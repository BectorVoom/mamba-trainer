//! Kernels for reinforcement and imitation learning.
//!
//! Four operations sit between a policy and a learning rule, and every one of them
//! is a place a naive implementation reaches for the host:
//!
//! * **sampling an action** from a row of logits — the textbook version reads the
//!   logits back, softmaxes them in Rust and uploads an id;
//! * **generalized advantage estimation** — a backwards recurrence over time, which
//!   looks like a `for` loop over a downloaded `[B, T]` array;
//! * **recording a step into a trajectory buffer** — a write into the `t`-th column
//!   of a `[B, T, W]` buffer, which looks like an upload;
//! * **mixing a learner's action with an expert's** — DAgger's coin flip, which
//!   looks like a host RNG.
//!
//! Each one of those reads blocks until the whole queue drains, so the naive
//! rollout loop synchronises four times per step and the environment and the device
//! take turns idling. The kernels here exist so none of that happens: an action is
//! chosen, scored, stored, advantaged and mixed without a byte crossing the bus.
//! [`crate::rl`] composes them, and `tests/rl_footprint.rs` asserts the resulting
//! loop reads back exactly zero times.
//!
//! All four are one-unit-per-row kernels over small rows — an action space, an
//! episode, an observation. None of them vectorise: `N` lanes of a [`Vector`] would
//! have to agree on a branch that is per-row by construction (which bucket the draw
//! landed in, whether this step ended an episode), and the rows are short enough
//! that the load is not what costs.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::hash_u32;
use crate::tensor::shape::Shape;

// ---------------------------------------------------------------------------
// Acting: sample an action and score it
// ---------------------------------------------------------------------------

pub use step::{
    Draw, draw_action, draw_action_masked, draw_action_with_mask, record_action, record_observation,
    record_outcome,
};

/// The device half of a rollout step, as `#[cube]` functions.
///
/// These are what [`categorical_kernel`] and [`crate::rl::fused`] are built from,
/// and they are public so that a caller can build a *different* kernel from the
/// same pieces. That is the escape hatch for a game this crate cannot host: one
/// whose state is arenas of its own element types, or whose transition needs a
/// cube-wide cooperative prologue, cannot be a [`crate::rl::GameLogic`], but it can
/// perfectly well be a kernel that calls [`draw_action`] and the three recorders
/// around its own step. [`crate::rl::Collector::collect_with`] drives such a kernel
/// and the rest of the loop — the policy, the advantage estimate, PPO — is
/// unchanged.
///
/// Calling these rather than reimplementing them is what makes such a kernel
/// collect the *same* window: the draw here is the crate's only sampler, so a
/// rollout built on it is scored by the same arithmetic a replay will score it by.
///
/// A module of its own so one `missing_docs` allow covers the companion modules
/// `#[cube]` generates; every item is re-exported and documented.
#[allow(missing_docs)]
pub mod step {
    use cubecl::prelude::*;

    use crate::tensor::ops::random::hash_unit;

    /// One row's action draw.
    #[derive(CubeType)]
    pub struct Draw<F: Float> {
        /// The action chosen.
        pub action: u32,
        /// `log p(action)` under the *tempered* distribution it was drawn from,
        /// which is what a policy gradient's importance ratio has to divide by.
        pub log_prob: F,
    }

    /// Sample one action from row `row` of a `[rows, classes]` logit block, and
    /// score it.
    ///
    /// Three passes over the row rather than one, and that is the right trade here.
    /// Keeping the exponentials would need `classes` registers or a scratch buffer;
    /// re-reading them costs an L1 hit each on an action space that is tens of
    /// entries wide, and buys a function whose register use does not depend on the
    /// action space at all — which matters when it is inlined into a kernel that
    /// has its own demands.
    ///
    /// `sample` is comptime: `false` returns the `argmax` and ignores the seed.
    #[cube]
    pub fn draw_action<F: Float + CubeElement>(
        logits: &Array<F>,
        row: usize,
        classes: usize,
        inv_temperature: F,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] sample: bool,
    ) -> Draw<F> {
        draw_action_with_mask::<F>(
            logits, logits, 0, row, classes, inv_temperature, seed_lo, seed_hi, sample, false,
        )
    }

    /// [`draw_action`] under a legal-action mask: `legal[legal_base + i]` is `0`
    /// where action `i` is illegal. Reads an illegal action's logit as the value
    /// [`crate::tensor::ops::elemwise::mask_logits`] writes (`F::min_value()`), so
    /// a fused masked draw and an unfused `mask_logits` + [`draw_action`] are the
    /// same arithmetic on the same values.
    #[cube]
    pub fn draw_action_masked<F: Float + CubeElement>(
        logits: &Array<F>,
        legal: &Array<F>,
        legal_base: usize,
        row: usize,
        classes: usize,
        inv_temperature: F,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] sample: bool,
    ) -> Draw<F> {
        draw_action_with_mask::<F>(
            logits, legal, legal_base, row, classes, inv_temperature, seed_lo, seed_hi, sample, true,
        )
    }

    /// Logit `i` of the row at `base`, or `F::min_value()` where a mask says it is
    /// illegal — never `-inf`, which WGSL cannot express as a constant.
    #[cube]
    fn logit_at<F: Float + CubeElement>(
        logits: &Array<F>,
        legal: &Array<F>,
        base: usize,
        legal_base: usize,
        i: usize,
        #[comptime] masked: bool,
    ) -> F {
        let mut value = logits[base + i];
        if comptime!(masked) {
            if legal[legal_base + i] == F::new(0.0_f32) {
                value = F::min_value();
            }
        }
        value
    }

    /// [`draw_action`] (`masked` false: `legal` is never read) or
    /// [`draw_action_masked`] (`masked` true), chosen at comptime — for a kernel
    /// that is itself specialised on whether its game masks, so it can make one
    /// call either way.
    #[cube]
    #[allow(clippy::too_many_arguments)]
    pub fn draw_action_with_mask<F: Float + CubeElement>(
        logits: &Array<F>,
        legal: &Array<F>,
        legal_base: usize,
        row: usize,
        classes: usize,
        inv_temperature: F,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] sample: bool,
        #[comptime] masked: bool,
    ) -> Draw<F> {
        let base = row * classes;

        // Pass 1. Both branches need the row maximum: greedy as the answer, and
        // sampling as the shift that keeps every `exp` below it in range.
        let mut top = logit_at::<F>(logits, legal, base, legal_base, 0, masked) * inv_temperature;
        let mut best = 0u32;
        for i in 1..classes {
            let v = logit_at::<F>(logits, legal, base, legal_base, i, masked) * inv_temperature;
            if v > top {
                top = v;
                best = i as u32;
            }
        }

        // Pass 2. The normaliser, in the shifted frame.
        let mut total = F::new(0.0_f32);
        for i in 0..classes {
            total += F::exp(logit_at::<F>(logits, legal, base, legal_base, i, masked) * inv_temperature - top);
        }

        let mut chosen = best;
        if comptime!(sample) {
            let unit = hash_unit::<F>(row as u32, seed_lo, seed_hi);

            // Pass 3. Inverse CDF, with the target scaled by the normaliser rather
            // than every weight divided by it. `chosen` counts the buckets whose
            // prefix has not yet passed the target, which *is* the index of the
            // first one that does — no early exit, so every unit in the cube walks
            // the same trip count.
            let target = unit * total;
            let mut acc = F::new(0.0_f32);
            let mut last_possible = best;
            chosen = 0u32;
            for i in 0..classes {
                let weight = F::exp(logit_at::<F>(logits, legal, base, legal_base, i, masked) * inv_temperature - top);
                acc += weight;
                if acc <= target {
                    chosen = (i + 1) as u32;
                }
                if weight > F::new(0.0_f32) {
                    last_possible = i as u32;
                }
            }
            // The prefix sum is computed in a different order from `total`, so it
            // can fall a rounding error short of a target drawn just below 1. The
            // fallback is the last action that has any probability at all — not
            // simply the last action, which under a legal-action mask (a
            // `F::min_value()` logit, weight exactly 0) may be one the draw must
            // never return.
            // Without a mask every weight is positive and this is `classes - 1`.
            if chosen > last_possible {
                chosen = last_possible;
            }
        }

        Draw::<F> {
            action: chosen,
            // `log p = (l_a/T - max) - log sum exp(l/T - max)`, which is the
            // log-softmax of the *tempered* logits.
            log_prob: logit_at::<F>(logits, legal, base, legal_base, chosen as usize, masked) * inv_temperature - top - F::ln(total),
        }
    }

    /// Copy environment `env`'s observation into column `t` of a
    /// `[envs, steps, width]` trajectory buffer.
    ///
    /// Call this *before* the transition, which is what makes it safe for a game to
    /// write its next observation over the row it was just handed: the only reader
    /// that still wants the old value is this copy, and it has already taken it.
    #[cube]
    pub fn record_observation<F: Float + CubeElement>(
        buffer: &mut Array<F>,
        observation: &Array<F>,
        env: usize,
        t: usize,
        steps: usize,
        width: usize,
    ) {
        let src = env * width;
        let dst = (env * steps + t) * width;
        for i in 0..width {
            buffer[dst + i] = observation[src + i];
        }
    }

    /// Record what the policy decided: the action, its log-probability and the
    /// critic's estimate for the observation it was chosen from.
    #[cube]
    pub fn record_action<F: Float + CubeElement>(
        actions: &mut Array<u32>,
        log_probs: &mut Array<F>,
        values: &mut Array<F>,
        env: usize,
        t: usize,
        steps: usize,
        draw: Draw<F>,
        value: F,
    ) {
        let slot = env * steps + t;
        actions[slot] = draw.action;
        log_probs[slot] = draw.log_prob;
        values[slot] = value;
    }

    /// Record what the transition returned, and the termination flag the next step
    /// is to be driven with.
    ///
    /// `last_done` is both this step's reset mask and the next one's, so a kernel
    /// that reads it before calling this writes the value its successor needs — one
    /// `[envs]` tensor for the life of the run rather than one per step.
    #[cube]
    pub fn record_outcome<F: Float + CubeElement>(
        rewards: &mut Array<F>,
        dones: &mut Array<F>,
        last_done: &mut Array<F>,
        env: usize,
        t: usize,
        steps: usize,
        reward: F,
        done: F,
    ) {
        let slot = env * steps + t;
        rewards[slot] = reward;
        dones[slot] = done;
        last_done[env] = done;
    }
}

/// One unit per row, over [`draw_action`].
///
/// The sampler itself lives in [`step`] so that this kernel and a caller's own
/// fused kernel are provably drawing from the same distribution with the same
/// arithmetic, rather than from two implementations that have to be shown to agree.
#[cube(launch_unchecked)]
fn categorical_kernel<F: Float + CubeElement>(
    logits: &Array<F>,
    actions: &mut Array<u32>,
    logprobs: &mut Array<F>,
    classes: usize,
    inv_temperature: F,
    seed_lo: u32,
    seed_hi: u32,
    #[comptime] sample: bool,
) {
    if ABSOLUTE_POS < actions.len() {
        let drawn = draw_action::<F>(
            logits,
            ABSOLUTE_POS,
            classes,
            inv_temperature,
            seed_lo,
            seed_hi,
            sample,
        );
        actions[ABSOLUTE_POS] = drawn.action;
        logprobs[ABSOLUTE_POS] = drawn.log_prob;
    }
}

/// Sample one action per row of `logits` and return it with its log-probability.
///
/// `logits` is `[..., classes]`; the outputs carry the leading shape. A
/// `temperature` of `0` makes the choice greedy, in which case `seed` is unused and
/// the result is deterministic.
///
/// The log-probability is of the *tempered* distribution — the one the action was
/// actually drawn from — because that is what a policy gradient's importance ratio
/// has to be measured against.
pub fn sample_categorical<R: Runtime, E: FloatElem>(
    logits: &Tensor<R, E>,
    temperature: f32,
    seed: u64,
) -> Result<(IdTensor<R>, Tensor<R, E>)> {
    if logits.rank() == 0 {
        return Err(Error::shape(
            "sampling needs a trailing action axis, got a scalar".to_string(),
        ));
    }
    let classes = logits.shape().dim_from_end(0);
    if classes == 0 {
        return Err(Error::shape(
            "cannot sample from an empty action space".to_string(),
        ));
    }
    if temperature < 0.0 {
        return Err(Error::config(format!(
            "temperature must not be negative, got {temperature}"
        )));
    }
    let rows_shape = logits.shape().without(logits.rank() - 1);
    let actions = IdTensor::empty(rows_shape.clone(), logits.device());
    let logprobs = Tensor::<R, E>::empty(rows_shape, logits.device());
    let rows = actions.len();
    if rows == 0 {
        return Ok((actions, logprobs));
    }

    // A zero temperature is `argmax`, not a division by zero.
    let greedy = temperature == 0.0;
    let inv_temperature = if greedy { 1.0 } else { 1.0 / temperature };
    let (count, dim) = launch_1d(logits.client(), rows, classes * 3);
    unsafe {
        categorical_kernel::launch_unchecked::<E, R>(
            logits.client(),
            count,
            dim,
            logits.arg(),
            actions.arg(),
            logprobs.arg(),
            classes,
            E::from_scalar(inv_temperature),
            seed as u32,
            (seed >> 32) as u32,
            !greedy,
        );
    }
    Ok((actions, logprobs))
}

// ---------------------------------------------------------------------------
// Learning: generalized advantage estimation
// ---------------------------------------------------------------------------

/// One unit per environment, walking its own episode backwards.
///
/// The recurrence is serial in `t` and independent across `B`, which is the exact
/// shape a rollout produces: the parallelism is the environments, and there are as
/// many of them as the rollout was wide. Nothing here is a reduction, so no unit
/// waits on another.
#[cube(launch_unchecked)]
fn gae_kernel<F: Float + CubeElement>(
    rewards: &Array<F>,
    values: &Array<F>,
    dones: &Array<F>,
    bootstrap: &Array<F>,
    advantages: &mut Array<F>,
    returns: &mut Array<F>,
    steps: usize,
    envs: usize,
    gamma: F,
    lambda: F,
) {
    if ABSOLUTE_POS < envs {
        let base = ABSOLUTE_POS * steps;
        let mut carry = F::new(0.0_f32);
        // What follows the window's last step. The collector reads it from the
        // critic on the observation the rollout stopped at, which is what keeps a
        // truncated window an estimate of the infinite-horizon return rather than
        // one that pretends the world ends at `T`.
        let mut next_value = bootstrap[ABSOLUTE_POS];
        for back in 0..steps {
            let t = steps - 1 - back;
            let i = base + t;
            // `dones[t]` marks the transition that *ended* an episode. Past it the
            // value of the successor is zero rather than the next row's estimate,
            // and the eligibility trace restarts rather than carrying credit across
            // a boundary into an episode that did not earn it.
            let alive = F::new(1.0_f32) - dones[i];
            let delta = rewards[i] + gamma * next_value * alive - values[i];
            carry = delta + gamma * lambda * alive * carry;
            advantages[i] = carry;
            // The critic's target is the advantage put back on the value scale.
            // That is the λ-return by construction, so the two heads are trained
            // against the same estimator rather than two that can disagree.
            returns[i] = carry + values[i];
            next_value = values[i];
        }
    }
}

/// What [`generalized_advantage`] produces.
pub struct Advantages<R: Runtime, E: FloatElem> {
    /// `[envs, steps]` advantage estimates, the actor's weighting.
    pub advantages: Tensor<R, E>,
    /// `[envs, steps]` λ-returns, the critic's target.
    pub returns: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> Clone for Advantages<R, E> {
    fn clone(&self) -> Self {
        Self {
            advantages: self.advantages.clone(),
            returns: self.returns.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Advantages<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Advantages({})", self.advantages.shape())
    }
}

/// Generalized advantage estimation over a `[envs, steps]` rollout.
///
/// * `rewards[e, t]` — the reward for the action taken at `t`.
/// * `values[e, t]` — the critic's estimate for the observation at `t`.
/// * `dones[e, t]` — `1` if the transition at `t` ended the episode.
/// * `bootstrap[e]` — the critic's estimate for the observation *after* the last
///   stored step, ignored where `dones[e, steps-1]` is set.
///
/// `lambda` interpolates between the one-step TD estimate (`0`, low variance, biased
/// by the critic) and the Monte Carlo return (`1`, unbiased, high variance).
pub fn generalized_advantage<R: Runtime, E: FloatElem>(
    rewards: &Tensor<R, E>,
    values: &Tensor<R, E>,
    dones: &Tensor<R, E>,
    bootstrap: &Tensor<R, E>,
    gamma: f32,
    lambda: f32,
) -> Result<Advantages<R, E>> {
    rewards.shape().expect_rank(2)?;
    let envs = rewards.shape().dim(0);
    let steps = rewards.shape().dim(1);
    for (name, other) in [("values", values), ("dones", dones)] {
        if other.shape() != rewards.shape() {
            return Err(Error::shape(format!(
                "{name} is {} but rewards are {}",
                other.shape(),
                rewards.shape()
            )));
        }
    }
    if bootstrap.len() != envs {
        return Err(Error::shape(format!(
            "bootstrap must hold one value per environment: expected {envs}, got {}",
            bootstrap.shape()
        )));
    }
    if !(0.0..=1.0).contains(&gamma) || !(0.0..=1.0).contains(&lambda) {
        return Err(Error::config(format!(
            "gamma and lambda are discount factors in [0, 1], got {gamma} and {lambda}"
        )));
    }

    let advantages = Tensor::empty(rewards.shape().clone(), rewards.device());
    let returns = Tensor::empty(rewards.shape().clone(), rewards.device());
    if envs == 0 || steps == 0 {
        return Ok(Advantages {
            advantages,
            returns,
        });
    }

    let (count, dim) = launch_1d(rewards.client(), envs, steps);
    unsafe {
        gae_kernel::launch_unchecked::<E, R>(
            rewards.client(),
            count,
            dim,
            rewards.arg(),
            values.arg(),
            dones.arg(),
            bootstrap.arg(),
            advantages.arg(),
            returns.arg(),
            steps,
            envs,
            E::from_scalar(gamma),
            E::from_scalar(lambda),
        );
    }
    Ok(Advantages {
        advantages,
        returns,
    })
}

// ---------------------------------------------------------------------------
// Accounting: completed-episode returns across window boundaries
// ---------------------------------------------------------------------------

/// One unit per environment, walking its own window forward.
///
/// Mirrors [`gae_kernel`]'s shape — serial in `t`, independent across `envs` — but
/// forward rather than backward: a completed episode's return only exists once its
/// last reward has been added, so there is nothing to accumulate ahead of time the
/// way GAE's bootstrap lets it walk backwards.
#[cube(launch_unchecked)]
fn episode_return_kernel<F: Float + CubeElement>(
    rewards: &Array<F>,
    dones: &Array<F>,
    running_in: &Array<F>,
    running_out: &mut Array<F>,
    completed_sum: &mut Array<F>,
    completed_count: &mut Array<F>,
    steps: usize,
    envs: usize,
) {
    if ABSOLUTE_POS < envs {
        let base = ABSOLUTE_POS * steps;
        let mut running = running_in[ABSOLUTE_POS];
        let mut sum = F::new(0.0_f32);
        let mut count = F::new(0.0_f32);
        for t in 0..steps {
            let i = base + t;
            running += rewards[i];
            let done = dones[i];
            // `dones[t]` marks the transition that *completed* an episode: fold
            // this lane's running total into the window's tally there, and only
            // there, then start the next episode's total from zero.
            sum += running * done;
            count += done;
            running *= F::new(1.0_f32) - done;
        }
        running_out[ABSOLUTE_POS] = running;
        completed_sum[ABSOLUTE_POS] = sum;
        completed_count[ABSOLUTE_POS] = count;
    }
}

/// What [`episode_returns`] produces.
pub struct EpisodeReturns<R: Runtime, E: FloatElem> {
    /// `[envs]` in-progress return of the episode each lane is mid-way through,
    /// `0` where the last transition in the window completed one. Carry this into
    /// the next window's `running` so an episode that straddles the boundary is
    /// not double-counted or dropped.
    pub running: Tensor<R, E>,
    /// `[envs]` sum of the returns of episodes that *completed* within this
    /// window, `0` on a lane where none did.
    pub completed_sum: Tensor<R, E>,
    /// `[envs]` number of episodes each lane completed within this window.
    pub completed_count: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> Clone for EpisodeReturns<R, E> {
    fn clone(&self) -> Self {
        Self {
            running: self.running.clone(),
            completed_sum: self.completed_sum.clone(),
            completed_count: self.completed_count.clone(),
        }
    }
}

/// Fold a `[envs, steps]` window of rewards and terminations into completed
/// episode returns, continuing each lane's in-progress total from `running`.
///
/// This is what makes an episode reported correctly regardless of whether it
/// fits in one collection window: `running` carries the partial sum of a lane's
/// current episode across the boundary, so a reward earned in window `N` and an
/// episode-ending reward earned in window `N+1` are added together rather than
/// silently dropped (`running` truncated at a window edge) or reported alone
/// (the window's total reward standing in for the whole episode's).
pub fn episode_returns<R: Runtime, E: FloatElem>(
    rewards: &Tensor<R, E>,
    dones: &Tensor<R, E>,
    running: &Tensor<R, E>,
) -> Result<EpisodeReturns<R, E>> {
    rewards.shape().expect_rank(2)?;
    let envs = rewards.shape().dim(0);
    let steps = rewards.shape().dim(1);
    if dones.shape() != rewards.shape() {
        return Err(Error::shape(format!(
            "dones is {} but rewards are {}",
            dones.shape(),
            rewards.shape()
        )));
    }
    if running.len() != envs {
        return Err(Error::shape(format!(
            "running must hold one value per environment: expected {envs}, got {}",
            running.shape()
        )));
    }

    let lanes = Shape::new(vec![envs]);
    if envs == 0 || steps == 0 {
        return Ok(EpisodeReturns {
            running: running.clone(),
            completed_sum: Tensor::zeros(lanes.clone(), rewards.device()),
            completed_count: Tensor::zeros(lanes, rewards.device()),
        });
    }

    let running_out = Tensor::empty(lanes.clone(), rewards.device());
    let completed_sum = Tensor::empty(lanes.clone(), rewards.device());
    let completed_count = Tensor::empty(lanes, rewards.device());
    let (count, dim) = launch_1d(rewards.client(), envs, steps);
    unsafe {
        episode_return_kernel::launch_unchecked::<E, R>(
            rewards.client(),
            count,
            dim,
            rewards.arg(),
            dones.arg(),
            running.arg(),
            running_out.arg(),
            completed_sum.arg(),
            completed_count.arg(),
            steps,
            envs,
        );
    }
    Ok(EpisodeReturns {
        running: running_out,
        completed_sum,
        completed_count,
    })
}

// ---------------------------------------------------------------------------
// Collecting: write one step into a trajectory buffer
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn write_step_kernel<F: Float + CubeElement>(
    buffer: &mut Array<F>,
    step: &Array<F>,
    steps: usize,
    width: usize,
    t: usize,
) {
    if ABSOLUTE_POS < step.len() {
        let env = ABSOLUTE_POS / width;
        let col = ABSOLUTE_POS % width;
        buffer[(env * steps + t) * width + col] = step[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
fn write_step_ids_kernel(buffer: &mut Array<u32>, step: &Array<u32>, steps: usize, t: usize) {
    if ABSOLUTE_POS < step.len() {
        buffer[ABSOLUTE_POS * steps + t] = step[ABSOLUTE_POS];
    }
}

/// Write `step` (`[envs, width]`) into column `t` of `buffer` (`[envs, steps, width]`).
///
/// In place, so a collection loop of any length allocates nothing. The caller owns
/// `buffer` exclusively — see [`super::elemwise::add_assign_`] for the rule both
/// in-place families follow.
pub fn write_step<R: Runtime, E: FloatElem>(
    buffer: &Tensor<R, E>,
    step: &Tensor<R, E>,
    t: usize,
) -> Result<()> {
    let rank = buffer.rank();
    if rank < 2 {
        return Err(Error::shape(format!(
            "a trajectory buffer is at least [envs, steps], got {}",
            buffer.shape()
        )));
    }
    let envs = buffer.shape().dim(0);
    let steps = buffer.shape().dim(1);
    let width: usize = buffer.dims()[2..].iter().product();
    if t >= steps {
        return Err(Error::shape(format!(
            "step {t} is past the end of a {steps}-step buffer"
        )));
    }
    if step.len() != envs * width {
        return Err(Error::shape(format!(
            "a step of {} does not fill one column of {}",
            step.shape(),
            buffer.shape()
        )));
    }
    let n = step.len();
    if n == 0 {
        return Ok(());
    }
    let (count, dim) = launch_1d(buffer.client(), n, 1);
    unsafe {
        write_step_kernel::launch_unchecked::<E, R>(
            buffer.client(),
            count,
            dim,
            buffer.arg(),
            step.arg(),
            steps,
            width,
            t,
        );
    }
    Ok(())
}

/// Write `step` (`[envs]` ids) into column `t` of `buffer` (`[envs, steps]` ids).
pub fn write_step_ids<R: Runtime>(
    buffer: &IdTensor<R>,
    step: &IdTensor<R>,
    t: usize,
) -> Result<()> {
    buffer.shape().expect_rank(2)?;
    let envs = buffer.shape().dim(0);
    let steps = buffer.shape().dim(1);
    if t >= steps {
        return Err(Error::shape(format!(
            "step {t} is past the end of a {steps}-step buffer"
        )));
    }
    if step.len() != envs {
        return Err(Error::shape(format!(
            "a step of {} does not fill one column of {}",
            step.shape(),
            buffer.shape()
        )));
    }
    if envs == 0 {
        return Ok(());
    }
    let (count, dim) = launch_1d(buffer.client(), envs, 1);
    unsafe {
        write_step_ids_kernel::launch_unchecked::<R>(
            buffer.client(),
            count,
            dim,
            buffer.arg(),
            step.arg(),
            steps,
            t,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Imitation: DAgger's coin flip
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn mix_actions_kernel(
    learner: &Array<u32>,
    expert: &Array<u32>,
    output: &mut Array<u32>,
    threshold: u32,
    seed_lo: u32,
    seed_hi: u32,
) {
    if ABSOLUTE_POS < output.len() {
        // The draw and the threshold are both 24-bit, so the comparison is exact
        // and no float enters a decision that only has two outcomes.
        let draw = hash_u32(ABSOLUTE_POS as u32, seed_lo, seed_hi) >> 8;
        let mut chosen = learner[ABSOLUTE_POS];
        if draw < threshold {
            chosen = expert[ABSOLUTE_POS];
        }
        output[ABSOLUTE_POS] = chosen;
    }
}

/// DAgger's mixture policy: take the expert's action with probability `beta`,
/// the learner's otherwise, independently per row.
///
/// `beta` starts at `1` — the first dataset is the expert's own trajectories — and
/// decays towards `0`, so the states the learner is trained on drift from the
/// expert's distribution to its own. That drift is the whole point of DAgger:
/// behaviour cloning alone only ever sees states the expert visits, and the
/// compounding error of the states it does not is what it cannot fix.
pub fn mix_actions<R: Runtime>(
    learner: &IdTensor<R>,
    expert: &IdTensor<R>,
    beta: f32,
    seed: u64,
) -> Result<IdTensor<R>> {
    if learner.shape() != expert.shape() {
        return Err(Error::shape(format!(
            "the learner proposed {} actions and the expert {}",
            learner.shape(),
            expert.shape()
        )));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(Error::config(format!(
            "beta is a probability in [0, 1], got {beta}"
        )));
    }
    let out = IdTensor::empty(learner.shape().clone(), learner.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    // `beta = 1` must take the expert every time, and the draw's largest value is
    // `2^24 - 1`, so the threshold ranges over `0 ..= 2^24`.
    let threshold = (beta * 16_777_216.0).round() as u32;
    let (count, dim) = launch_1d(learner.client(), n, 1);
    unsafe {
        mix_actions_kernel::launch_unchecked::<R>(
            learner.client(),
            count,
            dim,
            learner.arg(),
            expert.arg(),
            out.arg(),
            threshold,
            seed as u32,
            (seed >> 32) as u32,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Masking
// ---------------------------------------------------------------------------

/// The one description of what is wrong with a legal-action mask, shared by the
/// device check ([`validate_action_mask`]) and the host one
/// ([`check_action_mask_values`]) so the two cannot drift apart.
pub(crate) fn action_mask_problem(non_finite: bool, non_binary: f32, empty_rows: f32) -> Result<()> {
    if non_finite {
        return Err(Error::config("the action mask holds a non-finite value".to_string()));
    }
    if non_binary > 0.0 {
        return Err(Error::config(format!(
            "the action mask holds {non_binary} value(s) other than 0 or 1; a mask marks \
             each action legal (1) or illegal (0)"
        )));
    }
    if empty_rows > 0.0 {
        return Err(Error::config(format!(
            "the action mask leaves no legal action at all on {empty_rows} \
             observation(s); every observation must leave at least one action legal"
        )));
    }
    Ok(())
}

/// Check a `[.., action_dim]` legal-action mask: every value finite and exactly
/// `0` or `1`, and no row entirely illegal.
///
/// A row with no legal action is not treated as "every action legal" — that
/// would silently train on a batch [`crate::rl::VecEnv::action_mask`]'s own
/// contract calls invalid. Values other than `0`/`1` are refused because
/// [`super::elemwise::mask_logits`] treats *any* nonzero value as legal: a `-1`
/// would pass as legal, a `NaN` too, and a row like `[1, -1]` would sum to zero.
///
/// One host read, and only when a caller actually asks — masking that is
/// never used costs nothing here, exactly like every other optional feature
/// in this crate.
pub fn validate_action_mask<R: Runtime, E: FloatElem>(mask: &Tensor<R, E>) -> Result<()> {
    let values = action_mask_counts(mask)?.to_f32();
    action_mask_problem(!values[2].is_finite(), values[1], values[0])
}

/// `[3]` device tensor of what [`validate_action_mask`] reads: empty rows,
/// non-binary values, and the mask's sum (non-finite iff a value is). Kept on
/// the device so a caller with more to check can pack it into the same read.
pub(crate) fn action_mask_counts<R: Runtime, E: FloatElem>(mask: &Tensor<R, E>) -> Result<Tensor<R, E>> {
    use super::{elemwise, movement, reduce};

    if mask.rank() == 0 {
        return Err(Error::shape(
            "an action mask needs a trailing action axis, got a scalar".to_string(),
        ));
    }
    let last = mask.rank() - 1;
    let per_row = reduce::sum_dim(mask, last)?;
    let empty_rows = reduce::sum_all(&elemwise::eq_scalar(&per_row, 0.0))?;
    // `eq` is false for NaN, so a NaN counts as non-binary here as well as
    // poisoning `total` below.
    let binary = elemwise::add(&elemwise::eq_scalar(mask, 0.0), &elemwise::eq_scalar(mask, 1.0))?;
    let non_binary = elemwise::sub(
        &Tensor::full(vec![1], mask.len() as f32, mask.device()),
        &reduce::sum_all(&binary)?.reshape(vec![1])?,
    )?;
    let total = reduce::sum_all(mask)?;
    movement::cat(
        &[empty_rows.reshape(vec![1])?, non_binary, total.reshape(vec![1])?],
        0,
    )
}

/// [`validate_action_mask`] for a mask still on the host, as `[rows * action_dim]`
/// row-major values. No device involved: this is what an environment adapter
/// runs before the mask is uploaded, so a bad one is refused before anything is
/// drawn from it.
pub fn check_action_mask_values(values: &[f32], action_dim: usize) -> Result<()> {
    if action_dim == 0 || values.len() % action_dim != 0 {
        return Err(Error::shape(format!(
            "an action mask of {} values does not divide into rows of {action_dim} actions",
            values.len()
        )));
    }
    let non_finite = values.iter().any(|v| !v.is_finite());
    let non_binary = values.iter().filter(|&&v| v != 0.0 && v != 1.0).count() as f32;
    let empty_rows = values
        .chunks_exact(action_dim)
        .filter(|row| row.iter().all(|&v| v == 0.0))
        .count() as f32;
    action_mask_problem(non_finite, non_binary, empty_rows)
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// Centre and rescale advantages to zero mean and unit variance, on the device.
///
/// PPO's step size is set by the clip range, which is a bound on the *ratio*; the
/// size of the update it licenses therefore scales with whatever units the
/// advantage happens to be in. Normalising fixes those units, which is what makes
/// one clip range work across tasks whose rewards differ by orders of magnitude.
///
/// Nothing is read back: the mean and the variance are reduced on the device and
/// consumed there, so this stays inside the queue like everything around it.
pub fn normalize<R: Runtime, E: FloatElem>(
    values: &Tensor<R, E>,
    eps: f32,
) -> Result<Tensor<R, E>> {
    use super::{elemwise, reduce};

    let n = values.len();
    if n == 0 {
        return Ok(values.clone());
    }
    let flat = values.reshape(Shape::new(vec![n]))?;
    let mean = reduce::mean_all(&flat)?;
    let centred = elemwise::sub(&flat, &mean.reshape(Shape::new(vec![1]))?)?;
    let variance = reduce::mean_all(&elemwise::mul(&centred, &centred)?)?;
    // `rsqrt(var + eps)` rather than `1/sqrt(var)`: a rollout in which every
    // advantage is identical — one where nothing has been learned yet, or a task
    // whose reward is constant — has zero variance, and must produce zeros rather
    // than infinities.
    let scale = elemwise::rsqrt(&elemwise::add_scalar(&variance, eps));
    elemwise::mul(&centred, &scale.reshape(Shape::new(vec![1]))?)?.reshape(values.shape().clone())
}
