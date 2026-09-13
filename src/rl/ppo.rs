//! Proximal policy optimization over a recurrent policy.
//!
//! PPO is a policy gradient with a leash. The gradient itself is the old one —
//! push up the log-probability of actions that did better than the critic
//! expected — but the step is taken against a *ratio* to the policy that collected
//! the data, and the objective is flattened once that ratio leaves a trust region.
//! That is what lets the same rollout be reused for several gradient steps instead
//! of one, which is most of why PPO is sample-efficient enough to be worth running.
//!
//! ```text
//! r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
//! L      = -E[ min( r_t A_t , clip(r_t, 1-ε, 1+ε) A_t ) ]
//!          + c_v E[ (V_θ(s_t) - R_t)^2 ] - c_H E[ H(π_θ(· | s_t)) ]
//! ```
//!
//! # What is recurrent about it
//!
//! Nothing in the objective. Everything in how the terms are obtained: `π_θ(a_t|s_t)`
//! for a memoryful policy is not a function of `s_t` alone but of the whole episode
//! up to `t`, so the ratio is only meaningful if the recomputed pass sees the same
//! history the rollout did. Two things make that true here, and both come from
//! [`super`]:
//!
//! * the window is replayed through [`super::Mamba3Policy::forward`], the `O(T)`
//!   scan that agrees with the `O(1)` rollout step to `1e-4` (`tests/rl.rs`);
//! * it is replayed under the *same* episode mask, derived from the recorded
//!   terminations by [`super::TrajectoryBuffer::reset_mask`], so the recurrence is
//!   cut in the same places.
//!
//! Without the first, the ratio compares two different functions. Without the
//! second, credit leaks backwards across an episode boundary into a life the agent
//! did not live. Both are why a state space model is a good fit for this: the
//! replay is one wide parallel pass whose cost does not grow with how long the
//! episodes ran.

use std::cell::Cell;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::distributions::{Categorical, Distribution};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::MixerCache;
use crate::nn::module::Module;
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::rl::normalize;
use crate::tensor::ops::{elemwise, fused, movement, reduce};
use crate::train::checkpoint::Checkpoint;
use crate::train::trainer::TrainStep;

use super::buffer::TrajectoryBuffer;
use super::policy::{Mamba3Policy, PolicyOutput};

/// Hyperparameters of a PPO update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PpoConfig {
    /// Discount factor.
    pub gamma: f32,
    /// Eligibility trace decay of the advantage estimator.
    pub lambda: f32,
    /// Trust region half-width `ε` on the probability ratio.
    pub clip_coeff: f32,
    /// Weight of the value loss in the total.
    pub value_coeff: f32,
    /// Weight of the entropy bonus. Larger keeps the policy exploring for longer.
    pub entropy_coeff: f32,
    /// Clip the value function's own update to the same `ε`.
    pub clip_value_loss: bool,
    /// Centre and rescale advantages before they weight the gradient.
    pub normalize_advantages: bool,
    /// Weight of a penalty on moving away from a fixed *reference* policy.
    ///
    /// The clip bounds how far one update may move the policy from the weights
    /// that collected the data. Nothing in it bounds where the policy ends up
    /// after two hundred such updates, so a run that starts from a good cloned
    /// policy can walk away from it a fraction of a nat at a time and arrive
    /// somewhere much worse having never once tripped the trust region. This term
    /// is what stops that: it prices the distance from a policy chosen once and
    /// frozen, rather than from the one of a moment ago.
    ///
    /// `0` — the default — leaves it off, and PPO is what it was. It does nothing
    /// unless the batch also carries [`PpoBatch::reference_log_probs`].
    pub reference_coeff: f32,
}

impl Default for PpoConfig {
    /// The settings PPO is usually reported with, and a sane starting point.
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip_coeff: 0.2,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            clip_value_loss: true,
            normalize_advantages: true,
            reference_coeff: 0.0,
        }
    }
}

impl PpoConfig {
    /// Start from the defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the discount and trace decay.
    pub fn with_discount(mut self, gamma: f32, lambda: f32) -> Self {
        self.gamma = gamma;
        self.lambda = lambda;
        self
    }

    /// Anchor the policy to a fixed reference with weight `coeff`.
    ///
    /// Pair it with [`reference_log_probs`], which is what scores the reference on
    /// a collected window; without those the coefficient is inert.
    pub fn with_reference_penalty(mut self, coeff: f32) -> Self {
        self.reference_coeff = coeff;
        self
    }

    /// Set the trust region half-width.
    pub fn with_clip(mut self, eps: f32) -> Self {
        self.clip_coeff = eps;
        self
    }

    /// Set the value and entropy weights.
    pub fn with_coefficients(mut self, value: f32, entropy: f32) -> Self {
        self.value_coeff = value;
        self.entropy_coeff = entropy;
        self
    }

    /// Turn advantage normalisation on or off.
    pub fn with_normalized_advantages(mut self, on: bool) -> Self {
        self.normalize_advantages = on;
        self
    }

    /// Check internal consistency.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.gamma) || !(0.0..=1.0).contains(&self.lambda) {
            return Err(Error::config(format!(
                "gamma and lambda are discount factors in [0, 1], got {} and {}",
                self.gamma, self.lambda
            )));
        }
        if self.clip_coeff <= 0.0 {
            return Err(Error::config(format!(
                "the clip range must be positive, got {}; \
                 a zero trust region admits no update at all",
                self.clip_coeff
            )));
        }
        if !self.reference_coeff.is_finite() || self.reference_coeff < 0.0 {
            return Err(Error::config(format!(
                "the reference penalty is a distance price and cannot be negative or \
                 non-finite, got {}; a negative one pays the policy to leave",
                self.reference_coeff
            )));
        }
        Ok(())
    }
}

/// One window of experience, with everything a PPO update needs to score it.
///
/// Every field is `[envs, steps]` or `[envs, steps, obs_dim]` and lives on the
/// device. The `log_probs` and `values` are the *behaviour* policy's — what the
/// weights said when the data was collected — and are what the recomputed pass is
/// compared against.
pub struct PpoBatch<R: Runtime, E: FloatElem> {
    /// `[envs, steps, obs_dim]` observations.
    pub observations: Tensor<R, E>,
    /// `[envs, steps]` actions taken.
    pub actions: IdTensor<R>,
    /// `[envs, steps]` log-probabilities under the behaviour policy.
    pub log_probs: Tensor<R, E>,
    /// `[envs, steps]` advantage estimates.
    pub advantages: Tensor<R, E>,
    /// `[envs, steps]` λ-returns, the critic's target.
    pub returns: Tensor<R, E>,
    /// `[envs, steps]` critic estimates made during the rollout.
    pub values: Tensor<R, E>,
    /// `[envs, steps]` flags marking observations that begin an episode.
    pub reset: Option<Tensor<R, E>>,
    /// Recurrent state the window continues from, for truncated backpropagation.
    pub initial: Option<Vec<MixerCache<R, E>>>,
    /// `[envs, steps]` weights, `0` for a position that should not count.
    pub mask: Option<Tensor<R, E>>,
    /// `[envs, steps]` log-probabilities of the taken actions under a fixed
    /// reference policy, for [`PpoConfig::reference_coeff`].
    ///
    /// Scored once per window rather than once per epoch — the reference does not
    /// move, so neither does this — by [`reference_log_probs`].
    pub reference_log_probs: Option<Tensor<R, E>>,
    /// `[envs, steps, actions]` legal-action mask, `1` where an action was
    /// legal on the observation at that position and `0` where it was not.
    /// `None` means every action was legal throughout the window.
    ///
    /// Applied to the replay identically to how it was applied when the
    /// window was collected — see [`crate::rl::VecEnv::action_mask`] — so the
    /// distribution PPO's ratio divides by is the same one the actor's own
    /// log-probability was drawn from.
    pub action_mask: Option<Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> Clone for PpoBatch<R, E> {
    fn clone(&self) -> Self {
        Self {
            observations: self.observations.clone(),
            actions: self.actions.clone(),
            log_probs: self.log_probs.clone(),
            advantages: self.advantages.clone(),
            returns: self.returns.clone(),
            values: self.values.clone(),
            reset: self.reset.clone(),
            reference_log_probs: self.reference_log_probs.clone(),
            initial: self.initial.clone(),
            mask: self.mask.clone(),
            action_mask: self.action_mask.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for PpoBatch<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PpoBatch({})", self.observations.shape())
    }
}

impl<R: Runtime, E: FloatElem> PpoBatch<R, E> {
    /// Build a batch from a collected rollout.
    ///
    /// `bootstrap` is `[envs]`: the critic's estimate for the observation the
    /// rollout stopped at, which is what keeps a truncated window an estimate of
    /// the infinite-horizon return. Advantages are estimated here, once, and reused
    /// across every epoch of the update — recomputing them against the improving
    /// critic would be estimating the advantage of the new policy from the old
    /// policy's data.
    pub fn from_buffer(
        buffer: &TrajectoryBuffer<R, E>,
        bootstrap: &Tensor<R, E>,
        config: &PpoConfig,
    ) -> Result<Self> {
        config.validate()?;
        let window = buffer.len();
        if window == 0 {
            return Err(Error::config(
                "cannot build a PPO batch from an empty trajectory buffer".to_string(),
            ));
        }
        let estimate = buffer.advantages(bootstrap, config.gamma, config.lambda)?;
        let trim_f = |t: &Tensor<R, E>| -> Result<Tensor<R, E>> {
            if window == buffer.steps() {
                Ok(t.clone())
            } else {
                movement::slice(t, 1, 0, window)
            }
        };
        let action_mask = buffer
            .action_mask()
            .map(trim_f)
            .transpose()?
            .map(|mask| -> Result<Tensor<R, E>> {
                crate::rl::validate_action_mask(&mask)?;
                Ok(mask)
            })
            .transpose()?;
        Ok(Self {
            observations: trim_f(buffer.observations())?,
            actions: if window == buffer.steps() {
                buffer.actions().clone()
            } else {
                // Ids have no strided slice of their own; the trimming a partly
                // filled window needs is rare enough to go through the host.
                let all = buffer.actions().to_vec();
                let envs = buffer.envs();
                let kept: Vec<u32> = (0..envs)
                    .flat_map(|e| {
                        let base = e * buffer.steps();
                        all[base..base + window].to_vec()
                    })
                    .collect();
                IdTensor::from_slice(&kept, vec![envs, window], buffer.device())?
            },
            log_probs: trim_f(buffer.log_probs())?,
            advantages: estimate.advantages,
            returns: estimate.returns,
            values: trim_f(buffer.values())?,
            reset: Some(buffer.reset_mask()?),
            initial: None,
            mask: None,
            reference_log_probs: None,
            action_mask,
        })
    }

    /// Number of environments.
    pub fn envs(&self) -> usize {
        self.observations.shape().dim(0)
    }

    /// Number of steps.
    pub fn steps(&self) -> usize {
        self.observations.shape().dim(1)
    }

    /// Continue from a recurrent state, for truncated backpropagation through time.
    ///
    /// Pass the detached snapshot the rollout started from, so the replay sees the
    /// history the actor did rather than a zeroed state.
    pub fn continuing_from(mut self, initial: Vec<MixerCache<R, E>>) -> Self {
        self.initial = Some(initial);
        self
    }

    /// Weight the positions that count, `0` excluding one entirely.
    pub fn with_mask(mut self, mask: Tensor<R, E>) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Attach a frozen reference policy's scores, from [`reference_log_probs`].
    pub fn with_reference_log_probs(mut self, log_probs: Tensor<R, E>) -> Self {
        self.reference_log_probs = Some(log_probs);
        self
    }

    /// A contiguous run of `len` environments as a batch of its own.
    ///
    /// PPO's minibatches must be whole *sequences* for a recurrent policy — cutting
    /// across time would ask the replay to start mid-episode with no state — so the
    /// axis that can be split is the environment one, and this splits it.
    pub fn minibatch(&self, start: usize, len: usize) -> Result<Self> {
        let envs = self.envs();
        if len == 0 || start + len > envs {
            return Err(Error::shape(format!(
                "environments {start}..{} are outside a {envs}-environment batch",
                start + len
            )));
        }
        if len == envs {
            return Ok(self.clone());
        }
        let cut = |t: &Tensor<R, E>| movement::slice(t, 0, start, len);
        let steps = self.steps();
        let ids = self.actions.to_vec();
        let kept: Vec<u32> = ids[start * steps..(start + len) * steps].to_vec();
        Ok(Self {
            observations: cut(&self.observations)?,
            actions: IdTensor::from_slice(&kept, vec![len, steps], self.observations.device())?,
            log_probs: cut(&self.log_probs)?,
            advantages: cut(&self.advantages)?,
            returns: cut(&self.returns)?,
            values: cut(&self.values)?,
            reset: self.reset.as_ref().map(cut).transpose()?,
            initial: self
                .initial
                .as_ref()
                .map(|caches| {
                    caches
                        .iter()
                        .map(|c| slice_cache(c, start, len))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?,
            mask: self.mask.as_ref().map(cut).transpose()?,
            reference_log_probs: self.reference_log_probs.as_ref().map(cut).transpose()?,
            action_mask: self.action_mask.as_ref().map(cut).transpose()?,
        })
    }

    fn check(&self) -> Result<()> {
        self.observations.shape().expect_rank(3)?;
        let (envs, steps) = (self.envs(), self.steps());
        let want = envs * steps;
        for (name, len) in [
            ("actions", self.actions.len()),
            ("log_probs", self.log_probs.len()),
            ("advantages", self.advantages.len()),
            ("returns", self.returns.len()),
            ("values", self.values.len()),
        ] {
            if len != want {
                return Err(Error::shape(format!(
                    "a PPO batch's {name} holds {len} elements, \
                     expected {want} for [{envs}, {steps}]"
                )));
            }
        }
        for (name, field) in [
            ("reset", &self.reset),
            ("mask", &self.mask),
            ("reference_log_probs", &self.reference_log_probs),
        ] {
            if let Some(t) = field
                && t.len() != want
            {
                return Err(Error::shape(format!(
                    "a PPO batch's {name} holds {} elements, expected {want}",
                    t.len()
                )));
            }
        }
        if let Some(mask) = &self.action_mask {
            mask.shape().expect_rank(3)?;
            if mask.shape().dim(0) != envs || mask.shape().dim(1) != steps {
                return Err(Error::shape(format!(
                    "a PPO batch's action_mask is {}, expected [{envs}, {steps}, _]",
                    mask.shape()
                )));
            }
        }
        Ok(())
    }
}

/// One layer of recurrent state, restricted to a run of environments.
fn slice_cache<R: Runtime, E: FloatElem>(
    cache: &MixerCache<R, E>,
    start: usize,
    len: usize,
) -> Result<MixerCache<R, E>> {
    let cut = |v: &Var<R, E>| v.slice(0, start, len);
    Ok(MixerCache {
        ssm: crate::ssm::scan::SsmState {
            h: cut(&cache.ssm.h)?,
            last_u: cut(&cache.ssm.last_u)?,
            angle: cache.ssm.angle.as_ref().map(cut).transpose()?,
        },
        conv: cache.conv.as_ref().map(cut).transpose()?,
    })
}

/// The terms of one PPO update.
///
/// The three losses are on the tape; the diagnostics are plain device tensors taken
/// off it. Nothing here is read back — call `to_f32()` on a diagnostic when you want
/// to look, which is a synchronisation and belongs outside the hot path.
pub struct PpoLoss<R: Runtime, E: FloatElem> {
    /// The scalar that is differentiated.
    pub total: Var<R, E>,
    /// Clipped surrogate objective, negated so that lower is better.
    pub policy: Var<R, E>,
    /// Mean squared error of the critic against the λ-returns.
    pub value: Var<R, E>,
    /// Mean entropy of the policy. Larger means it is still exploring.
    pub entropy: Var<R, E>,
    /// Schulman's low-variance estimate of `KL(π_old || π_θ)`, as a `[1]` tensor.
    ///
    /// The number to watch. It is what the clip is a proxy for, and a run whose KL
    /// climbs across epochs is one whose ratio has left the region the data
    /// supports — reduce the epochs or the learning rate rather than the clip.
    pub approx_kl: Tensor<R, E>,
    /// Fraction of positions whose ratio was clipped, as a `[1]` tensor.
    pub clip_fraction: Tensor<R, E>,
    /// Estimated `KL(π_θ || π_ref)` against the frozen reference, as a `[1]`
    /// tensor. Zero when no reference was supplied.
    pub reference_kl: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for PpoLoss<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PpoLoss(total={})", self.total.shape())
    }
}

/// Mean of `x` over the positions `mask` keeps.
///
/// With no mask this is a plain mean. With one it is the weighted mean — the sum
/// divided by the weight, not by the count — so that excluding positions changes
/// which terms contribute and not how large the remaining ones are.
fn masked_mean<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    mask: Option<&Var<R, E>>,
) -> Result<Var<R, E>> {
    match mask {
        None => x.mean(),
        Some(mask) => {
            let kept = mask.sum()?;
            // A mask that keeps nothing would divide by zero; one kept position is
            // the floor, and the numerator is zero there anyway.
            let floor = Var::constant(Tensor::full(kept.shape().clone(), 1.0, kept.device()));
            x.mul(mask)?.sum()?.div(&kept.maximum(&floor)?)
        }
    }
}

/// The clipped surrogate, the value loss and the entropy bonus for one window.
///
/// `output` is what the policy produced when the window was replayed: `logits` of
/// `[envs, steps, actions]` and `value` of `[envs, steps]`.
pub fn ppo_objective<R: Runtime, E: FloatElem>(
    output: &PolicyOutput<R, E>,
    batch: &PpoBatch<R, E>,
    config: &PpoConfig,
) -> Result<PpoLoss<R, E>> {
    config.validate()?;
    batch.check()?;
    output.logits.shape().expect_rank(3)?;
    let (envs, steps) = (batch.envs(), batch.steps());
    let classes = output.logits.dims()[2];
    if output.logits.dims()[0] != envs || output.logits.dims()[1] != steps {
        return Err(Error::shape(format!(
            "the replayed logits are {} but the batch is [{envs}, {steps}]",
            output.logits.shape()
        )));
    }
    let rows = envs * steps;
    let flat = vec![rows];

    // -- the policy's own view of what it did ------------------------------
    //
    // Both terms come from one distribution, and each is a single fused kernel: the
    // chosen action's log-probability for the ratio, and the row's entropy for the
    // bonus. Composed out of tensor operations — a `log_softmax`, a gather, an
    // `exp`, a product and a reduction — the same two numbers cost seven launches
    // forward and as many back, and write six intermediates the width of the whole
    // window. See [`crate::distributions::categorical`].
    //
    // Masked identically to the draw that collected this window: an illegal
    // action's logit is `-inf` here too, so the ratio it would otherwise
    // contribute divides two numbers that were never really in competition —
    // see [`crate::rl::VecEnv::action_mask`].
    let replay_logits = output.logits.reshape(vec![rows, classes])?;
    let replay_logits = match &batch.action_mask {
        Some(action_mask) => {
            replay_logits.mask_logits(&action_mask.reshape(vec![rows, classes])?)?
        }
        None => replay_logits,
    };
    let policy = Categorical::from_logits(replay_logits)?;
    let chosen = policy.log_prob_ids(&batch.actions.reshape(flat.clone())?)?;
    let entropy = policy.entropy()?;

    let mask = batch
        .mask
        .as_ref()
        .map(|m| Var::constant(m.reshape(flat.clone()).unwrap()));
    let advantages = if config.normalize_advantages {
        normalize(&batch.advantages, 1e-8)?
    } else {
        batch.advantages.clone()
    };
    let advantages = advantages.reshape(flat.clone())?;

    // -- the clipped surrogate ---------------------------------------------
    //
    // The *minimum* of the clipped and unclipped terms, which is a pessimistic bound
    // rather than a symmetric one: the objective is flattened where the ratio has
    // moved too far in the direction the advantage points, and left alone where it
    // has moved too far against it — so a step that overshoots is stopped and a step
    // that recovers from an overshoot is not.
    //
    // One launch each way rather than six and ten: see [`Var::ppo_surrogate`]. The
    // ratio comes back with it because the diagnostics below want it and the adjoint
    // has already saved it.
    let old_log_probs = batch.log_probs.reshape(flat.clone())?;
    let (surrogate, ratio_t) = chosen.ppo_surrogate(
        &old_log_probs,
        &advantages,
        config.clip_coeff,
    )?;
    let policy = masked_mean(&surrogate, mask.as_ref())?.neg();

    // -- the critic --------------------------------------------------------
    //
    // With `clip_value_loss` the same trust region is applied to the critic, on the
    // value scale: a single update cannot move an estimate further than `ε` from what
    // it was when the data was collected, unless doing so is the *larger* error, in
    // which case the unclipped term is taken and the critic is not let off. Fused
    // either way — see [`Var::ppo_value_loss`].
    let value_pred = output.value.reshape(flat.clone())?;
    let returns = batch.returns.reshape(flat.clone())?;
    let old_values = batch.values.reshape(flat.clone())?;
    let squared = value_pred.ppo_value_loss(
        &returns,
        &old_values,
        config.clip_coeff,
        config.clip_value_loss,
    )?;
    let value = masked_mean(&squared, mask.as_ref())?.mul_scalar(0.5);

    let entropy_mean = masked_mean(&entropy, mask.as_ref())?;

    // -- the anchor --------------------------------------------------------
    //
    // `KL(π_θ || π_ref) ≈ E[exp(d) - d - 1]` with `d = log π_ref(a) - log π_θ(a)`,
    // the same estimator the diagnostic below uses against the behaviour policy and
    // for the same reason: non-negative for every `d`, zero exactly when the two
    // agree, and far lower variance than `-d`. Unlike the clip, which prices the
    // distance from the weights that collected this window, this prices the
    // distance from a policy fixed once — so two hundred small steps in one
    // direction cost what they actually are rather than nothing at all.
    let (total, reference_kl) = match (&batch.reference_log_probs, config.reference_coeff) {
        (Some(reference), coeff) if coeff != 0.0 => {
            let reference = Var::constant(reference.reshape(flat.clone())?);
            let d = reference.sub(&chosen)?;
            let per_position = d.exp().sub(&d)?.add_scalar(-1.0);
            let mean = masked_mean(&per_position, mask.as_ref())?;
            let total = policy
                .add(&value.mul_scalar(config.value_coeff))?
                .sub(&entropy_mean.mul_scalar(config.entropy_coeff))?
                .add(&mean.mul_scalar(coeff))?;
            (total, mean.tensor().clone())
        }
        _ => {
            let total = policy
                .add(&value.mul_scalar(config.value_coeff))?
                .sub(&entropy_mean.mul_scalar(config.entropy_coeff))?;
            let zero = Tensor::zeros(vec![1], output.logits.tensor().device());
            (total, zero)
        }
    };

    // -- diagnostics, off the tape -----------------------------------------
    //
    // Built from the recorded values rather than recorded themselves: they are read
    // by a human, never differentiated, and putting them on the tape would keep the
    // graph that produced them alive for no reason.
    //
    // `KL ≈ (r - 1) - log r`, which is non-negative for every `r` and has far less
    // variance than `-log r` alone — the estimator from Schulman's note on the
    // three ways to approximate a KL from samples. It and the clip flag come out of
    // one launch, which is all they should ever have cost.
    let (kl_terms, clipped_flags) = fused::ppo_diagnostics(
        chosen.tensor(),
        &old_log_probs,
        &ratio_t,
        config.clip_coeff,
    )?;
    let (approx_kl, clip_fraction) = match &batch.mask {
        None => (
            reduce::mean_all(&kl_terms)?,
            reduce::mean_all(&clipped_flags)?,
        ),
        Some(m) => {
            let m = m.reshape(flat)?;
            let kept = reduce::sum_all(&m)?;
            let kept = elemwise::clamp(&kept, 1.0, f32::MAX);
            (
                elemwise::div(&reduce::sum_all(&elemwise::mul(&kl_terms, &m)?)?, &kept)?,
                elemwise::div(
                    &reduce::sum_all(&elemwise::mul(&clipped_flags, &m)?)?,
                    &kept,
                )?,
            )
        }
    };

    Ok(PpoLoss {
        total,
        policy,
        value,
        entropy: entropy_mean,
        approx_kl: approx_kl.reshape(vec![1])?,
        clip_fraction: clip_fraction.reshape(vec![1])?,
        reference_kl: reference_kl.reshape(vec![1])?,
    })
}

/// Score a collected window under a frozen reference policy, continuing the
/// reference's *own* recurrent history rather than the behaviour policy's.
///
/// `initial` is the reference's own cache — `None` at the start of a run, then
/// whatever this function last returned — and is completely independent of
/// [`PpoBatch::initial`], which is the *actor's* snapshot and belongs to a
/// different set of weights. Feeding the actor's cache to the reference here
/// would score the reference as if it had lived the actor's recent history
/// instead of its own, which drifts further the more the two diverge.
///
/// The same episode-reset mask the batch was collected with (`batch.reset`) is
/// applied inside [`Mamba3Policy::forward`] regardless of whose cache is
/// passed in, so lane resets — at a window boundary or inside the window — cut
/// the reference's history exactly where they cut the actor's.
///
/// Returns the `[envs, steps]` log-probabilities of the actions actually taken,
/// and the reference's end-of-window cache — feed that back in as `initial` for
/// the next window. Call this once per window, not once per epoch or update:
/// the reference does not move, so its answer for a given window and a given
/// starting cache does not either, and calling it twice on the same window
/// would advance the saved history twice for data that was only lived once.
pub fn reference_log_probs_from<R: Runtime, E: FloatElem>(
    reference: &Mamba3Policy<R, E>,
    batch: &PpoBatch<R, E>,
    initial: Option<&[MixerCache<R, E>]>,
) -> Result<(Tensor<R, E>, Option<Vec<MixerCache<R, E>>>)> {
    let _guard = crate::autograd::no_grad();
    batch.check()?;
    let (envs, steps) = (batch.envs(), batch.steps());
    let (output, end) = reference.forward(
        &Var::constant(batch.observations.clone()),
        batch.reset.as_ref(),
        initial,
    )?;
    let classes = output.logits.dims()[2];
    let rows = envs * steps;
    // The same legal-action mask the batch was collected and replayed with:
    // masking describes the observation, not which policy is looking at it, so
    // the reference is scored over the same support the actor was.
    let logits = output.logits.reshape(vec![rows, classes])?;
    let logits = match &batch.action_mask {
        Some(action_mask) => logits.mask_logits(&action_mask.reshape(vec![rows, classes])?)?,
        None => logits,
    };
    let distribution = Categorical::from_logits(logits)?;
    let scored = distribution.log_prob_ids(&batch.actions.reshape(vec![rows])?)?;
    Ok((scored.into_tensor().reshape(vec![envs, steps])?, end))
}

/// Score a collected window under a frozen reference policy, continuing from
/// the *behaviour* policy's own snapshot (`batch.initial`) rather than any
/// history of the reference's own.
///
/// Honest only when that is genuinely what is wanted — a single-window score,
/// or a reference whose history is not being tracked across windows at all.
/// [`ReferencePolicy::score`] is the corrected, stateful path a multi-window
/// learner should use instead; this compatibility wrapper exists for callers,
/// such as the tests in `tests/rl_reference.rs`, that score one window in
/// isolation and have no reference cache of their own to continue.
///
/// ```no_run
/// # use mamba3::prelude::*;
/// # use mamba3::rl::{PpoBatch, reference_log_probs};
/// # fn go<R: cubecl::prelude::Runtime>(
/// #     clone: &mamba3::rl::Mamba3Policy<R, f32>, batch: PpoBatch<R, f32>,
/// # ) -> Result<()> {
/// let scores = reference_log_probs(clone, &batch)?;
/// let batch = batch.with_reference_log_probs(scores);
/// # Ok(()) }
/// ```
pub fn reference_log_probs<R: Runtime, E: FloatElem>(
    reference: &Mamba3Policy<R, E>,
    batch: &PpoBatch<R, E>,
) -> Result<Tensor<R, E>> {
    reference_log_probs_from(reference, batch, batch.initial.as_deref()).map(|(scores, _)| scores)
}

/// An immutable copy of a policy's weights, plus the recurrent cache the
/// reference accumulates across the windows it scores.
///
/// # Why a copy, and not a shared handle
///
/// [`Mamba3Policy`]'s parameters are `Rc<RefCell<_>>` under the hood (see
/// [`crate::nn::param::Param`]), so even a *distinct* [`Mamba3Policy`] value
/// obtained by cloning an `Rc` to the same instance mutates in place when the
/// original is trained or reloaded — sharing the reference with the policy
/// being optimised, directly or through one intervening clone, would make the
/// "frozen" reference drift with it. [`ReferencePolicy::snapshot`] builds a
/// genuinely independent copy — a fresh [`Mamba3Policy`] of the same
/// architecture with the source's weights copied in through
/// [`Checkpoint::capture`]/[`Checkpoint::restore`] — so it stays exactly what
/// it was even when `source` is the policy the learner is training, or is
/// later reloaded out from under the caller.
pub struct ReferencePolicy<R: Runtime, E: FloatElem> {
    policy: Mamba3Policy<R, E>,
    cache: Option<Vec<MixerCache<R, E>>>,
}

impl<R: Runtime, E: FloatElem> ReferencePolicy<R, E> {
    /// Deep-copy `source`'s weights. Safe even when `source` is the same
    /// object as (or shares storage with) a policy that is later trained.
    pub fn snapshot(source: &Mamba3Policy<R, E>, device: &Device<R>) -> Result<Self> {
        let fresh = source.config().init::<R, E>(device)?;
        Checkpoint::capture::<R, E, _>(source, 0).restore::<R, E, _>(&fresh, true)?;
        Ok(Self {
            policy: fresh,
            cache: None,
        })
    }

    /// The frozen weights.
    pub fn policy(&self) -> &Mamba3Policy<R, E> {
        &self.policy
    }

    /// Score `batch`'s actions, continuing this reference's own cache from the
    /// previous call rather than the batch's `initial` (the actor's). Advances
    /// the saved cache by exactly one window's worth of history.
    pub fn score(&mut self, batch: &PpoBatch<R, E>) -> Result<Tensor<R, E>> {
        let (scores, end) = reference_log_probs_from(&self.policy, batch, self.cache.as_deref())?;
        self.cache = end;
        Ok(scores)
    }

    /// Forget the accumulated cache, e.g. when the learner itself is reset.
    pub fn reset(&mut self) {
        self.cache = None;
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for ReferencePolicy<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ReferencePolicy(cached={})", self.cache.is_some())
    }
}

/// A [`TrainStep`] that optimises a [`Mamba3Policy`] with PPO.
///
/// One `loss` call is one epoch over one batch: the window is replayed through the
/// scan, scored against the behaviour policy's log-probabilities, and reduced to a
/// scalar. Running several epochs over the same batch is calling
/// [`crate::train::Trainer::step`] several times with it — which is the whole
/// reason the ratio is there.
pub struct PpoTask<'a, R: Runtime, E: FloatElem> {
    policy: &'a Mamba3Policy<R, E>,
    params: Vec<Param<R, E>>,
    config: PpoConfig,
    training: Cell<bool>,
    last: std::cell::RefCell<Option<[Tensor<R, E>; 6]>>,
}

/// The diagnostics of the most recent [`PpoTask::loss`], on the host.
#[derive(Debug, Clone, Copy, Default)]
pub struct PpoStats {
    /// Clipped surrogate objective, negated so lower is better.
    pub policy_loss: f32,
    /// Value loss.
    pub value_loss: f32,
    /// Mean policy entropy.
    pub entropy: f32,
    /// Approximate `KL(π_old || π_θ)`.
    pub approx_kl: f32,
    /// Fraction of positions whose ratio was clipped.
    pub clip_fraction: f32,
    /// Estimated `KL(π_θ || π_ref)`. Zero when no reference is anchoring the run.
    pub reference_kl: f32,
}

impl<'a, R: Runtime, E: FloatElem> PpoTask<'a, R, E> {
    /// Train every parameter of `policy`.
    pub fn new(policy: &'a Mamba3Policy<R, E>, config: PpoConfig) -> Self {
        Self {
            params: policy.parameters(),
            policy,
            config,
            training: Cell::new(true),
            last: std::cell::RefCell::new(None),
        }
    }

    /// Train only the parameters whose path contains one of `patterns`, e.g.
    /// `only(&["lora"])` to fine-tune a policy through adapters alone.
    pub fn only(mut self, patterns: &[&str]) -> Self {
        self.params = self
            .policy
            .named_parameters()
            .into_iter()
            .filter(|(name, _)| patterns.iter().any(|p| name.contains(p)))
            .map(|(_, p)| p)
            .collect();
        self
    }

    /// The policy being optimised.
    pub fn policy(&self) -> &'a Mamba3Policy<R, E> {
        self.policy
    }

    /// The configuration.
    pub fn config(&self) -> &PpoConfig {
        &self.config
    }

    /// Diagnostics of the most recent loss, or `None` before the first.
    ///
    /// Reading these is a device synchronisation, so it is deferred to here:
    /// [`PpoTask::loss`] only keeps the five tensors, and a loop that never asks
    /// never stalls. Ask once per optimizer step, after the step, and it costs what
    /// the trainer's own loss read already costs.
    pub fn stats(&self) -> Option<PpoStats> {
        self.last.borrow().as_ref().map(|t| PpoStats {
            policy_loss: t[0].to_f32()[0],
            value_loss: t[1].to_f32()[0],
            entropy: t[2].to_f32()[0],
            approx_kl: t[3].to_f32()[0],
            clip_fraction: t[4].to_f32()[0],
            reference_kl: t[5].to_f32()[0],
        })
    }

    /// Replay a window and score it, returning every term.
    pub fn evaluate(&self, batch: &PpoBatch<R, E>) -> Result<PpoLoss<R, E>> {
        let observations = if self.training.get() {
            Var::traced(batch.observations.clone())
        } else {
            Var::constant(batch.observations.clone())
        };
        let (output, _) = self.policy.forward(
            &observations,
            batch.reset.as_ref(),
            batch.initial.as_deref(),
        )?;
        ppo_objective(&output, batch, &self.config)
    }
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for PpoTask<'_, R, E> {
    type Batch = PpoBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>> {
        if !self.training.get() {
            let _guard = crate::autograd::no_grad();
            return Ok(self.evaluate(batch)?.total);
        }
        let loss = self.evaluate(batch)?;
        // Kept as tensors, not read. These are handles into the queue; turning them
        // into numbers is what [`PpoTask::stats`] does, and only when asked.
        *self.last.borrow_mut() = Some([
            loss.policy.tensor().clone(),
            loss.value.tensor().clone(),
            loss.entropy.tensor().clone(),
            loss.approx_kl.clone(),
            loss.clip_fraction.clone(),
            loss.reference_kl.clone(),
        ]);
        Ok(loss.total)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.policy.set_training(training);
    }
}
