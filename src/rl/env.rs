//! Vectorised environments that live on the device.
//!
//! An environment is the one part of a reinforcement learning loop this crate does
//! not own — it is the user's simulator, their game, their robot. What the crate
//! can insist on is the *interface*, and the shape of that interface decides
//! whether the loop synchronises. [`VecEnv`] therefore speaks entirely in device
//! tensors: observations come back as a `[envs, obs_dim]` tensor, actions go in as
//! device ids, and nothing crosses the bus. An environment that genuinely runs on
//! the host can still implement it — it just pays an upload per step, and it will
//! be visible in `read_count()`.
//!
//! # The kernel here
//!
//! Every other `#[cube]` kernel in the crate lives in [`crate::tensor::ops`],
//! which is its reusable numerical vocabulary. [`RecallEnv`]'s transition function
//! is not vocabulary — it is one specific toy task — so it lives with the
//! environment it belongs to rather than being filed beside `matmul`.

use cubecl::prelude::*;

use crate::backend::{Device, FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::hash_u32;

/// What an environment returns when it is stepped.
pub struct EnvStep<R: Runtime, E: FloatElem> {
    /// `[envs, obs_dim]` observation the agent must now act on.
    ///
    /// Where `done` is set, this is already the first observation of the *next*
    /// episode — environments auto-reset, which is what lets a rollout of fixed
    /// length hold environments whose episodes have different ones.
    pub observation: Tensor<R, E>,
    /// `[envs]` reward earned by the action just taken.
    pub reward: Tensor<R, E>,
    /// `[envs]` flag: `1` where that action ended the episode.
    ///
    /// This is the `done[t]` convention [`super::TrajectoryBuffer`] stores and the
    /// advantage estimator consumes, one step earlier than the reset flag the
    /// policy is driven with.
    pub done: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> Clone for EnvStep<R, E> {
    fn clone(&self) -> Self {
        Self {
            observation: self.observation.clone(),
            reward: self.reward.clone(),
            done: self.done.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for EnvStep<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "EnvStep({})", self.observation.shape())
    }
}

/// A batch of environments advancing together.
pub trait VecEnv<R: Runtime, E: FloatElem> {
    /// How many environments run in parallel.
    fn envs(&self) -> usize;

    /// Width of one observation.
    fn obs_dim(&self) -> usize;

    /// Number of discrete actions.
    fn action_dim(&self) -> usize;

    /// Start every environment and return the first `[envs, obs_dim]` observation.
    fn reset(&mut self) -> Result<Tensor<R, E>>;

    /// Apply one `[envs]` action per environment.
    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, E>>;

    /// What an expert would do on the observation most recently returned, if the
    /// environment can say.
    ///
    /// This is what makes imitation learning testable without a human: an
    /// environment that knows its own optimal action is an expert that can label
    /// any state, which is exactly what DAgger needs and what a recorded
    /// demonstration set cannot provide.
    fn expert_actions(&self) -> Option<IdTensor<R>> {
        None
    }

    /// Which actions are legal on the observation most recently returned, before
    /// its action is drawn.
    ///
    /// `[envs, action_dim]`, `1` where the action is legal and `0` where it is
    /// not. `None` — the default — means every action is legal on this step,
    /// exactly as if this method did not exist; every caller of it must treat
    /// absence that way, not as "nothing is legal." An environment may return a
    /// mask on some steps and `None` on others.
    ///
    /// A row with no legal action at all, or a value other than `0`/`1`, is an
    /// invalid environment contract, refused rather than silently proceeding.
    /// A device mask is checked once per window, when the window becomes a
    /// batch ([`crate::rl::validate_action_mask`]), because checking it per step
    /// would be a host read per step; an environment that can check its mask on
    /// the host before uploading it (see
    /// [`crate::rl::check_action_mask_values`]) should, and return the error
    /// here — the collector stops before drawing from it.
    ///
    /// An error stops the collection before this step's action is drawn.
    fn action_mask(&self) -> Result<Option<Tensor<R, E>>> {
        Ok(None)
    }
}

/// A mutable reference to an environment is an environment.
///
/// Without this, every caller that reaches its environment through indirection —
/// a `&mut dyn VecEnv` behind a trait object, a handle owned by a binding to
/// another language — has to re-wrap it before [`super::Collector`] will take it,
/// because `collect` is generic over a sized `V`. This is the same courtesy
/// `std::io::Read` extends to `&mut R`, and it costs nothing: every method is a
/// forward the compiler inlines away.
impl<R: Runtime, E: FloatElem, V: VecEnv<R, E> + ?Sized> VecEnv<R, E> for &mut V {
    fn envs(&self) -> usize {
        (**self).envs()
    }

    fn obs_dim(&self) -> usize {
        (**self).obs_dim()
    }

    fn action_dim(&self) -> usize {
        (**self).action_dim()
    }

    fn reset(&mut self) -> Result<Tensor<R, E>> {
        (**self).reset()
    }

    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, E>> {
        (**self).step(actions)
    }

    fn expert_actions(&self) -> Option<IdTensor<R>> {
        (**self).expert_actions()
    }

    fn action_mask(&self) -> Result<Option<Tensor<R, E>>> {
        (**self).action_mask()
    }
}

// ---------------------------------------------------------------------------
// A task that cannot be solved without memory
// ---------------------------------------------------------------------------

/// Fill one environment's observation slot from its `(cue, clock)` state.
///
/// The cue is shown only at `clock == 0`. Everything after that is a blank
/// observation and a clock reading, so the action at the end of an episode can only
/// be right if the cue was *carried* there.
#[cube]
fn write_observation<F: Float + CubeElement>(
    obs: &mut Array<F>,
    env: usize,
    cue: u32,
    clock: u32,
    action_dim: usize,
    obs_dim: usize,
    inv_horizon: F,
) {
    let base = env * obs_dim;
    let showing = select(clock == 0u32, F::new(1.0_f32), F::new(0.0_f32));
    for i in 0..action_dim {
        let hit = select(i == cue as usize, F::new(1.0_f32), F::new(0.0_f32));
        obs[base + i] = showing * hit;
    }
    // Two channels that are always visible: how far into the episode this is, and
    // whether the cue is on screen. Both are information the agent may use freely;
    // neither tells it what the cue was.
    obs[base + action_dim] = F::cast_from(clock) * inv_horizon;
    obs[base + action_dim + 1] = showing;
}

/// A fresh cue for `env`, drawn from a stateless hash of its position and the seed.
#[cube]
fn draw_cue(env: u32, action_dim: usize, seed_lo: u32, seed_hi: u32) -> u32 {
    hash_u32(env, seed_lo, seed_hi) % (action_dim as u32)
}

#[cube(launch_unchecked)]
fn recall_reset_kernel<F: Float + CubeElement>(
    cues: &mut Array<u32>,
    clocks: &mut Array<u32>,
    obs: &mut Array<F>,
    expert: &mut Array<u32>,
    action_dim: usize,
    obs_dim: usize,
    inv_horizon: F,
    seed_lo: u32,
    seed_hi: u32,
) {
    if ABSOLUTE_POS < cues.len() {
        let cue = draw_cue(ABSOLUTE_POS as u32, action_dim, seed_lo, seed_hi);
        cues[ABSOLUTE_POS] = cue;
        clocks[ABSOLUTE_POS] = 0u32;
        expert[ABSOLUTE_POS] = cue;
        write_observation::<F>(
            obs,
            ABSOLUTE_POS,
            cue,
            0u32,
            action_dim,
            obs_dim,
            inv_horizon,
        );
    }
}

#[cube(launch_unchecked)]
fn recall_step_kernel<F: Float + CubeElement>(
    actions: &Array<u32>,
    cues: &mut Array<u32>,
    clocks: &mut Array<u32>,
    obs: &mut Array<F>,
    reward: &mut Array<F>,
    done: &mut Array<F>,
    expert: &mut Array<u32>,
    action_dim: usize,
    obs_dim: usize,
    horizon: u32,
    inv_horizon: F,
    seed_lo: u32,
    seed_hi: u32,
) {
    if ABSOLUTE_POS < cues.len() {
        let cue = cues[ABSOLUTE_POS];
        let clock = clocks[ABSOLUTE_POS];
        let terminal = clock + 1u32 == horizon;

        // The reward is sparse and it is at the end: naming the cue early costs
        // nothing and earns nothing, so the only way to score is to still know it
        // when the episode closes.
        let correct = select(
            actions[ABSOLUTE_POS] == cue,
            F::new(1.0_f32),
            F::new(0.0_f32),
        );
        reward[ABSOLUTE_POS] = select(terminal, correct, F::new(0.0_f32));
        done[ABSOLUTE_POS] = select(terminal, F::new(1.0_f32), F::new(0.0_f32));

        // Auto-reset: a terminated environment draws a new cue and restarts its
        // clock, so the rollout stays rectangular however the episodes fall.
        let fresh = draw_cue(ABSOLUTE_POS as u32, action_dim, seed_lo, seed_hi);
        let next_cue = select(terminal, fresh, cue);
        let next_clock = select(terminal, 0u32, clock + 1u32);
        cues[ABSOLUTE_POS] = next_cue;
        clocks[ABSOLUTE_POS] = next_clock;
        expert[ABSOLUTE_POS] = next_cue;
        write_observation::<F>(
            obs,
            ABSOLUTE_POS,
            next_cue,
            next_clock,
            action_dim,
            obs_dim,
            inv_horizon,
        );
    }
}

/// A cue-recall task: see a symbol once, name it `horizon` steps later.
///
/// This exists because most toy control problems do not actually test what a state
/// space policy is for. A pole balances on the observation in front of it; an agent
/// with no memory at all does nearly as well as one with perfect memory, so a run
/// that succeeds proves the plumbing works and nothing about the recurrence.
///
/// Here the cue is on screen at step `0` and gone from step `1`, and the only
/// reward in an episode is `1` for naming it at step `horizon - 1`. A memoryless
/// policy cannot exceed `1/action_dim` — it is guessing — and the ceiling is `1`.
/// The gap between those two numbers is the state buffer doing its job, and it is
/// what `tests/rl_learn.rs` measures.
///
/// It also comes with its own expert: the optimal action at every step is the cue,
/// which the environment knows, so [`VecEnv::expert_actions`] can label any state
/// the learner wanders into. That is what makes DAgger testable here.
pub struct RecallEnv<R: Runtime, E: FloatElem> {
    cues: IdTensor<R>,
    clocks: IdTensor<R>,
    expert: IdTensor<R>,
    envs: usize,
    action_dim: usize,
    horizon: usize,
    seed: u64,
    episodes: u64,
    device: Device<R>,
    _marker: core::marker::PhantomData<E>,
}

impl<R: Runtime, E: FloatElem> RecallEnv<R, E> {
    /// Build `envs` environments over `action_dim` symbols and episodes of
    /// `horizon` steps.
    pub fn new(
        envs: usize,
        action_dim: usize,
        horizon: usize,
        seed: u64,
        device: &Device<R>,
    ) -> Result<Self> {
        if envs == 0 || action_dim < 2 || horizon == 0 {
            return Err(Error::config(format!(
                "a recall task needs at least one environment, two symbols and one \
                 step, got envs={envs}, action_dim={action_dim}, horizon={horizon}"
            )));
        }
        Ok(Self {
            cues: IdTensor::empty(vec![envs], device),
            clocks: IdTensor::empty(vec![envs], device),
            expert: IdTensor::empty(vec![envs], device),
            envs,
            action_dim,
            horizon,
            seed,
            episodes: 0,
            device: device.clone(),
            _marker: core::marker::PhantomData,
        })
    }

    /// Steps in one episode.
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// The return a policy that guesses uniformly earns per episode, and the floor
    /// any run has to clear to have learned anything.
    pub fn chance_return(&self) -> f32 {
        1.0 / self.action_dim as f32
    }

    /// The return a perfect policy earns per episode.
    pub fn optimal_return(&self) -> f32 {
        1.0
    }

    /// Draws are seeded per episode so a run is reproducible, and per environment so
    /// two environments do not receive the same cue at the same time.
    fn next_seed(&mut self) -> u64 {
        self.episodes = self.episodes.wrapping_add(1);
        self.seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.episodes.wrapping_mul(1442695040888963407))
    }
}

impl<R: Runtime, E: FloatElem> VecEnv<R, E> for RecallEnv<R, E> {
    fn envs(&self) -> usize {
        self.envs
    }

    fn obs_dim(&self) -> usize {
        // One channel per symbol, plus the clock and the cue-present flag.
        self.action_dim + 2
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn reset(&mut self) -> Result<Tensor<R, E>> {
        let obs_dim = self.obs_dim();
        let obs = Tensor::<R, E>::empty(vec![self.envs, obs_dim], &self.device);
        let seed = self.next_seed();
        let (count, dim) = launch_1d(self.device.client(), self.envs, obs_dim);
        unsafe {
            recall_reset_kernel::launch_unchecked::<E, R>(
                self.device.client(),
                count,
                dim,
                self.cues.arg(),
                self.clocks.arg(),
                obs.arg(),
                self.expert.arg(),
                self.action_dim,
                obs_dim,
                E::from_scalar(1.0 / self.horizon as f32),
                seed as u32,
                (seed >> 32) as u32,
            );
        }
        Ok(obs)
    }

    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, E>> {
        if actions.len() != self.envs {
            return Err(Error::shape(format!(
                "the recall task has {} environments but was given {} actions",
                self.envs,
                actions.len()
            )));
        }
        let obs_dim = self.obs_dim();
        // Fresh each step rather than written in place. The pooled allocator hands
        // back the same buffers, so the footprint is just as flat, and nothing the
        // policy is still reading can be overwritten underneath it.
        let obs = Tensor::<R, E>::empty(vec![self.envs, obs_dim], &self.device);
        let reward = Tensor::<R, E>::empty(vec![self.envs], &self.device);
        let done = Tensor::<R, E>::empty(vec![self.envs], &self.device);
        let seed = self.next_seed();
        let (count, dim) = launch_1d(self.device.client(), self.envs, obs_dim);
        unsafe {
            recall_step_kernel::launch_unchecked::<E, R>(
                self.device.client(),
                count,
                dim,
                actions.arg(),
                self.cues.arg(),
                self.clocks.arg(),
                obs.arg(),
                reward.arg(),
                done.arg(),
                self.expert.arg(),
                self.action_dim,
                obs_dim,
                self.horizon as u32,
                E::from_scalar(1.0 / self.horizon as f32),
                seed as u32,
                (seed >> 32) as u32,
            );
        }
        Ok(EnvStep {
            observation: obs,
            reward,
            done,
        })
    }

    fn expert_actions(&self) -> Option<IdTensor<R>> {
        // The cue matching the observation most recently emitted. Valid until the
        // next `step`, which is when the buffer is rewritten.
        Some(self.expert.clone())
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for RecallEnv<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "RecallEnv(envs={}, symbols={}, horizon={})",
            self.envs, self.action_dim, self.horizon
        )
    }
}
