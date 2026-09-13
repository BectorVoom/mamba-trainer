//! The fused rollout step: one kernel from logits to the next observation.
//!
//! A rollout step is a big computation followed by a long tail of tiny ones. The
//! big one is the policy — a stack of Mamba-3 blocks, tens of launches, each of
//! them doing real work. The tail is everything that turns its output back into the
//! next observation, and on a batch of environments every piece of it touches a few
//! hundred bytes:
//!
//! | | launches | what it reads |
//! |---|---|---|
//! | draw an action and score it | 1 | `[envs, actions]` logits |
//! | record the observation | 1 | `[envs, obs_dim]` |
//! | record action, log-probability, value, reward, termination | 5 | `[envs]` each |
//! | the environment transition | 1 | the environment's own state |
//!
//! Eight launches, none of which is limited by anything but the fact that it is a
//! launch, and — this is the part that matters — the same eight whatever the policy
//! and the game are. A step of the small policy `tests/rl_fused_footprint.rs`
//! measures is 75 launches, so its tail is 10% of it; halve the model and the tail
//! is a fifth of a step, because only the other 75 shrank.
//!
//! They collapse into one because they are the same computation per environment:
//! every one of them is a row of `envs`, and the tail is a straight line through
//! that row. One unit takes environment `e` from its logits to its next
//! observation without any of the intermediates reaching memory — the action, its
//! log-probability, the reward and the termination flag are all born and consumed
//! in registers, and only the trajectory buffer is written.
//!
//! ```text
//!  logits[e] ─▶ sample ─▶ a ─┬─▶ buffer.actions[e, t]
//!                            ├─▶ log π(a) ─▶ buffer.log_probs[e, t]
//!                            └─▶ G::transition ─┬─▶ reward ─▶ buffer.rewards[e, t]
//!  obs[e] ────▶ buffer.observations[e, t]       ├─▶ done ───▶ buffer.dones[e, t]
//!  value[e] ──▶ buffer.values[e, t]             └─▶ obs[e], for the next step
//! ```
//!
//! That is what [`GameLogic`] is for: the transition has to be device code the
//! crate can paste into the middle of its own kernel, and a [`super::VecEnv`] — a
//! host object that answers with tensors — cannot be.
//!
//! # What is *not* fused
//!
//! The policy. A Mamba-3 stack is matrix products and a recurrence over a state
//! that does not fit in one unit's registers, so it stays the multi-kernel pass it
//! is; fusing the tail into it would mean fusing it into the *last* of those
//! kernels, whose shape is set by the actor head and not by the environments. What
//! the fusion removes is the overhead *between* policy steps, which is the part
//! that does not shrink when the model does. On the configuration the footprint
//! test measures that is a step of 83 launches becoming one of 76; the remaining 75
//! are the policy, and cutting those is a separate piece of work on
//! [`crate::models::mamba3`] rather than on this loop.
//!
//! # It is the same rollout
//!
//! The draw is the same arithmetic as [`crate::tensor::ops::rl::sample_categorical`]
//! on the same seed schedule, and the transition is the same function
//! [`super::GameWorld`] runs unfused. So a fused window and an unfused one over the
//! same seeds hold *identical* bytes, which is what `tests/rl_fused.rs` asserts —
//! a fusion that is only approximately the same rollout would be a fusion that
//! silently changes what is being learned.

use cubecl::prelude::*;

use crate::autograd::Var;
use crate::backend::{FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::rl::{
    draw_action_with_mask, record_action, record_observation, record_outcome, write_step,
};

use super::buffer::Column;
use super::collect::{CollectReport, Collector};
use super::game::{GameLogic, GameSpec, GameWorld};

/// One environment, from its logits to its next observation.
///
/// One unit per environment, like every other kernel in
/// [`crate::tensor::ops::rl`] and for the same reason: the row is an action space
/// or an observation, tens of entries wide, and the decisions in it — which bucket
/// the draw landed in, whether the episode ended — are per-environment by
/// construction, so there is nothing for a vector lane to agree on.
#[allow(clippy::too_many_arguments)] // Six buffers of one window, three of one world.
#[cube(launch_unchecked)]
fn fused_step_kernel<F: Float + CubeElement, G: GameLogic<F>>(
    logits: &Array<F>,
    values: &Array<F>,
    ints: &mut Array<u32>,
    floats: &mut Array<F>,
    obs: &mut Array<F>,
    buf_obs: &mut Array<F>,
    buf_actions: &mut Array<u32>,
    buf_log_probs: &mut Array<F>,
    buf_values: &mut Array<F>,
    buf_rewards: &mut Array<F>,
    buf_dones: &mut Array<F>,
    buf_mask: &mut Array<F>,
    last_done: &mut Array<F>,
    envs: usize,
    steps: usize,
    t: usize,
    inv_temperature: F,
    draw_lo: u32,
    draw_hi: u32,
    game_lo: u32,
    game_hi: u32,
    #[comptime] spec: GameSpec,
    #[comptime] sample: bool,
) {
    if ABSOLUTE_POS < envs {
        let env = ABSOLUTE_POS;
        // The crate's only sampler, shared with `sample_categorical` so that the
        // fused rollout and a replay of it are scored by the same arithmetic.
        let legal_base = (env * steps + t) * spec.action_dim;
        if comptime!(spec.masked) {
            // The mask for the state the action is drawn from — before the
            // transition below moves it on — recorded into the window and read
            // by the draw in the same unit. `GameWorld::action_mask` computes the
            // same flags from the same state on the unfused path.
            for action in 0..spec.action_dim {
                let legal = G::legal(env as u32, action as u32, ints, floats, spec);
                buf_mask[legal_base + action] = select(legal, F::new(1.0_f32), F::new(0.0_f32));
            }
        }
        // Without a mask the comptime flag compiles the read of `buf_mask` away,
        // leaving exactly `draw_action`.
        let drawn = draw_action_with_mask::<F>(
            logits,
            buf_mask,
            legal_base,
            env,
            spec.action_dim,
            inv_temperature,
            draw_lo,
            draw_hi,
            sample,
            spec.masked,
        );

        // The observation the action was chosen from, copied into the window before
        // the transition below overwrites it. This ordering is the whole reason the
        // world's observation buffer can be written in place: the only reader that
        // still wants the old value is this copy, and it has already taken it.
        // Taken before the draw is handed to the recorder, which consumes it.
        let action = drawn.action;

        record_observation::<F>(buf_obs, obs, env, t, steps, spec.obs_dim);
        record_action::<F>(
            buf_actions,
            buf_log_probs,
            buf_values,
            env,
            t,
            steps,
            drawn,
            values[env],
        );

        let out = G::transition(
            env as u32, action, ints, floats, obs, game_lo, game_hi, spec,
        );

        // `record_outcome` also writes the termination flag the next step is driven
        // with and the next window starts from, in place — so the collector holds one
        // `[envs]` tensor for the life of the run rather than one per step.
        record_outcome::<F>(
            buf_rewards,
            buf_dones,
            last_done,
            env,
            t,
            steps,
            out.reward,
            out.done,
        );
    }
}

/// One step of a rollout whose kernel the caller launches.
///
/// Everything the crate has decided by the time the environment must act: what the
/// policy produced, where the step is to be recorded, the termination flag to read
/// as a reset and write as the next one, and the draw's seed and temperature. What
/// the caller supplies is the kernel that turns those into the next observation.
///
/// The tensors are the collector's own, not copies, so a kernel writes straight
/// into the window it will later learn from. Every field is flat: `logits` is
/// `[envs, action_dim]` and `values` `[envs]` as far as indexing is concerned,
/// whatever ranks they carry.
pub struct FusedStep<'a, R: Runtime, E: FloatElem> {
    /// `[envs, action_dim]` logits for the observation the environment is showing.
    pub logits: &'a Tensor<R, E>,
    /// `[envs]` critic estimates for the same observation.
    pub values: &'a Tensor<R, E>,
    /// `[envs, obs_dim]` observation the action is being chosen from.
    ///
    /// The caller's kernel is expected to overwrite this with the next one, after
    /// copying it into the window — see
    /// [`crate::tensor::ops::rl::record_observation`].
    pub observation: &'a Tensor<R, E>,
    /// The trajectory column to fill, and its index.
    pub column: Column<'a, R, E>,
    /// `[envs]` termination flags.
    ///
    /// Read as this step's reset mask — the policy has already been driven with
    /// it — and to be written with this step's terminations, which is what the next
    /// step and the next window read.
    pub last_done: &'a Tensor<R, E>,
    /// Seed for this step's action draw. Split into `(seed as u32, (seed >> 32) as u32)`
    /// for [`crate::tensor::ops::rl::draw_action`].
    pub draw_seed: u64,
    /// `1 / temperature`, or `1` when the draw is greedy.
    pub inv_temperature: f32,
    /// Whether to sample. `false` means take the `argmax`, and ignore the seed.
    pub sample: bool,
    /// Environments in the window.
    pub envs: usize,
    /// Columns in the buffer, which is the stride between one environment's steps.
    pub steps: usize,
}

impl<R: Runtime, E: FloatElem> FusedStep<'_, R, E> {
    /// The column being written.
    pub fn t(&self) -> usize {
        self.column.t
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for FusedStep<'_, R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "FusedStep(t={}/{}, envs={}, sample={})",
            self.column.t, self.steps, self.envs, self.sample
        )
    }
}

impl<R: Runtime, E: FloatElem> Collector<'_, R, E> {
    /// Collect one window, launching each step's rollout kernel yourself.
    ///
    /// The escape hatch for a game this crate cannot host. A [`GameLogic`] must fit
    /// one signature — two state arenas of the crate's element types, no read-only
    /// side inputs, and a transition callable from inside a divergent branch. A real
    /// simulator often fits none of those: its state may be arenas of `i64` and
    /// `u64`, it may need constant tables staged into shared memory once per cube
    /// with a `sync_cube()` that must not sit under a per-environment `if`, and it
    /// may already own and manage its buffers on the host.
    ///
    /// So this hands the loop back. The crate still runs the policy, keeps the
    /// recurrent state, opens and closes the window, and turns the result into a
    /// [`PpoBatch`](super::PpoBatch); `step` is called once per column with a
    /// [`FusedStep`] and is expected to queue *one* kernel that draws the action,
    /// records the step and advances the environment. Build that kernel out of
    /// [`crate::tensor::ops::rl::step`]'s functions rather than reimplementing them
    /// — that is what makes the window it collects the window a replay will score.
    ///
    /// `observation` is the `[envs, obs_dim]` buffer the caller's kernel writes the
    /// next observation into. It must already hold the first one: this method does
    /// not reset anything, because it does not know how. Pass the same buffer every
    /// window — it is written in place, and it is what the policy reads.
    ///
    /// ```no_run
    /// # use mamba3::prelude::*;
    /// # use mamba3::rl::FusedStep;
    /// # fn go<R: cubecl::prelude::Runtime>(
    /// #     collector: &mut Collector<'_, R, f32>,
    /// #     observation: &mamba3::tensor::Tensor<R, f32>,
    /// #     config: &PpoConfig,
    /// # ) -> Result<()> {
    /// let report = collector.collect_with(observation, |step: FusedStep<'_, R, f32>| {
    ///     // queue one kernel: draw_action, record_*, and your own transition
    ///     Ok(())
    /// })?;
    /// let batch = collector.ppo_batch(&report, config)?;
    /// # Ok(()) }
    /// ```
    pub fn collect_with<S>(
        &mut self,
        observation: &Tensor<R, E>,
        step: S,
    ) -> Result<CollectReport<R, E>>
    where
        S: FnMut(FusedStep<'_, R, E>) -> Result<()>,
    {
        let envs = self.buffer().envs();
        let obs_dim = self.buffer().obs_dim();
        if observation.len() != envs * obs_dim {
            return Err(Error::shape(format!(
                "the collector is built for {envs} environments of width {obs_dim}, \
                 which is {} elements; the observation holds {}",
                envs * obs_dim,
                observation.len()
            )));
        }
        self.adopt_observation(observation.clone());
        let result = self.drive(step, false);
        self.recover_from(&result);
        result
    }

    /// The window loop both fused paths share: policy, hand over, commit.
    ///
    /// When the buffer has a mask column and `writes_mask` is false — a kernel
    /// that does not restrict actions — each column's mask is written as
    /// all-legal first, so no step reads a stale row from an earlier window.
    fn drive<S>(&mut self, mut step: S, writes_mask: bool) -> Result<CollectReport<R, E>>
    where
        S: FnMut(FusedStep<'_, R, E>) -> Result<()>,
    {
        let envs = self.buffer().envs();
        let obs_dim = self.buffer().obs_dim();
        let initial = self.open()?;
        let steps = self.buffer().steps();
        let all_legal = match (writes_mask, self.buffer().action_mask()) {
            (false, Some(mask)) => Some(Tensor::<R, E>::ones(
                vec![envs, mask.shape().dim(2)],
                mask.device(),
            )),
            _ => None,
        };

        for _ in 0..steps {
            // The policy reads the observation buffer; the caller's kernel
            // overwrites it. Both are launches on one ordered queue, so the read is
            // complete before the write is issued — the same guarantee every other
            // in-place operation in this crate rests on.
            let observation = self
                .observation_handle()
                .ok_or_else(|| Error::config("nothing has been collected yet".to_string()))?;
            let windowed = Var::constant(observation.reshape(vec![envs, 1, obs_dim])?);
            // At `t == 0` this still holds the previous window's last flag, which is
            // exactly the carry; from then on it holds the step before's.
            let reset = self.last_done_handle();
            let out = self.step_policy(&windowed, &reset)?;

            let draw_seed = self.next_draw_seed();
            let temperature = self.temperature();
            let greedy = temperature == 0.0;
            let column = self.buffer().column()?;
            if let (Some(ones), Some(mask)) = (&all_legal, column.action_mask) {
                write_step(mask, ones, column.t)?;
            }
            step(FusedStep {
                logits: out.logits.tensor(),
                values: out.value.tensor(),
                observation: &observation,
                column,
                last_done: &reset,
                draw_seed,
                inv_temperature: if greedy { 1.0 } else { 1.0 / temperature },
                sample: !greedy,
                envs,
                steps,
            })?;
            self.commit_step();
        }

        self.finish(initial, steps)
    }

    /// Collect one window with the environment fused into the rollout step.
    ///
    /// Does what [`Collector::collect`] does — same window, same buffer, same
    /// report, and the result is usable through [`Collector::ppo_batch`] exactly
    /// the same way — but the eight small launches that follow each policy step
    /// become one. The saving is per *step*, so it scales with the window length
    /// and not with the size of the model.
    ///
    /// ```no_run
    /// # use mamba3::prelude::*;
    /// # use mamba3::rl::{GameLogic, GameSpec, GameWorld, Mamba3Policy, PpoConfig};
    /// # fn go<R: cubecl::prelude::Runtime, G: GameLogic<f32>>(
    /// #     collector: &mut Collector<'_, R, f32>,
    /// #     world: &mut GameWorld<R, f32, G>,
    /// #     config: &PpoConfig,
    /// # ) -> Result<()> {
    /// let report = collector.collect_fused(world)?;
    /// let batch = collector.ppo_batch(&report, config)?;
    /// # Ok(()) }
    /// ```
    ///
    /// # Not a DAgger rollout
    ///
    /// A [`GameLogic`] has no expert to mix with and no labels to record, so
    /// [`Collector::collect_with_expert`] has no fused counterpart. Imitation stays
    /// on the [`super::VecEnv`] path, where the expert is a tensor the environment
    /// hands over.
    pub fn collect_fused<G: GameLogic<E>>(
        &mut self,
        world: &mut GameWorld<R, E, G>,
    ) -> Result<CollectReport<R, E>> {
        let spec = world.spec();
        // `prepare` checks the environment count and the observation width, as it
        // does for any environment. The action space is the one thing only this path
        // can get wrong: the kernel samples over `spec.action_dim` columns of a logit
        // row the policy laid out, so a disagreement reads another environment's
        // logits rather than failing, and the rollout would look plausible and be
        // nonsense.
        let actions = self.engine().policy().config().action_dim;
        if spec.action_dim != actions {
            return Err(Error::shape(format!(
                "the policy chooses between {actions} actions and the game between {}",
                spec.action_dim
            )));
        }
        self.prepare(world)?;
        if spec.masked {
            self.buffer_mut().ensure_action_mask(spec.action_dim)?;
        }
        // The kernel's mask binding when the game has none: one element, never
        // read, because the kernel is compiled without the masked branch — and
        // left uninitialised, since filling it would be a launch per window that
        // an unmasked rollout never used to issue.
        let unbound = Tensor::<R, E>::empty(vec![1], &self.buffer().device().clone());

        let report = self.drive(
            |s| {
                let game_seed = world.next_seed();
                let FusedStep {
                    logits,
                    values,
                    observation,
                    column,
                    last_done,
                    draw_seed,
                    inv_temperature,
                    sample,
                    envs,
                    steps,
                } = s;
                let t = column.t;
                let (count, dim) = launch_1d(
                    logits.client(),
                    envs,
                    spec.action_dim * 3 + spec.obs_dim * 2,
                );
                unsafe {
                    fused_step_kernel::launch_unchecked::<E, G, R>(
                        logits.client(),
                        count,
                        dim,
                        logits.arg(),
                        values.arg(),
                        world.ints().arg(),
                        world.floats().arg(),
                        observation.arg(),
                        column.observations.arg(),
                        column.actions.arg(),
                        column.log_probs.arg(),
                        column.values.arg(),
                        column.rewards.arg(),
                        column.dones.arg(),
                        match (spec.masked, column.action_mask) {
                            (true, Some(mask)) => mask.arg(),
                            _ => unbound.arg(),
                        },
                        last_done.arg(),
                        envs,
                        steps,
                        t,
                        E::from_scalar(inv_temperature),
                        draw_seed as u32,
                        (draw_seed >> 32) as u32,
                        game_seed as u32,
                        (game_seed >> 32) as u32,
                        spec,
                        sample,
                    );
                }
                Ok(())
            },
            spec.masked,
        );
        self.recover_from(&report);
        let report = report?;

        // The world's observation buffer is written in place, so this is the same
        // handle throughout a fused run — but not if an unfused `collect` on the
        // same world replaced it in between, which is why it is adopted rather than
        // assumed.
        self.adopt_observation(world.observation().clone());
        Ok(report)
    }
}
