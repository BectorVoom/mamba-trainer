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
use crate::tensor::ops::random::{hash_u32, hash_unit};
use crate::tensor::shape::Shape;

// ---------------------------------------------------------------------------
// Acting: sample an action and score it
// ---------------------------------------------------------------------------

/// One unit per row: the row maximum, the normaliser, then the inverse CDF.
///
/// Three passes over a row of logits rather than one, and that is the right trade
/// here. Keeping the exponentials would need `classes` registers or a scratch
/// buffer; re-reading them costs an L1 hit each on an action space that is tens of
/// entries wide, and buys a kernel whose register use does not depend on the
/// action space at all.
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
        let base = ABSOLUTE_POS * classes;

        // Pass 1. Both branches need the row maximum: greedy as the answer, and
        // sampling as the shift that keeps every `exp` below it in range.
        let mut top = logits[base] * inv_temperature;
        let mut best = 0u32;
        for i in 1..classes {
            let v = logits[base + i] * inv_temperature;
            if v > top {
                top = v;
                best = i as u32;
            }
        }

        // Pass 2. The normaliser, in the shifted frame.
        let mut total = F::new(0.0_f32);
        for i in 0..classes {
            total += F::exp(logits[base + i] * inv_temperature - top);
        }

        let mut chosen = best;
        if comptime!(sample) {
            let unit = hash_unit::<F>(ABSOLUTE_POS as u32, seed_lo, seed_hi);

            // Pass 3. Inverse CDF, with the target scaled by the normaliser rather
            // than every weight divided by it. `chosen` counts the buckets whose
            // prefix has not yet passed the target, which *is* the index of the
            // first one that does — no early exit, so every unit in the cube walks
            // the same trip count.
            let target = unit * total;
            let mut acc = F::new(0.0_f32);
            chosen = 0u32;
            for i in 0..classes {
                acc += F::exp(logits[base + i] * inv_temperature - top);
                if acc <= target {
                    chosen = (i + 1) as u32;
                }
            }
            // The prefix sum is computed in a different order from `total`, so it
            // can fall a rounding error short of a target drawn just below 1.
            if chosen as usize >= classes {
                chosen = (classes - 1) as u32;
            }
        }

        actions[ABSOLUTE_POS] = chosen;
        // `log p = (l_a/T - max) - log sum exp(l/T - max)`, which is the log-softmax
        // of the *tempered* logits — the distribution actually sampled from.
        logprobs[ABSOLUTE_POS] =
            logits[base + chosen as usize] * inv_temperature - top - F::ln(total);
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
