//! Imitation learning: behaviour cloning and DAgger.
//!
//! Imitation learning is supervised learning wearing a disguise, and the disguise
//! is the only interesting part. Behaviour cloning takes an expert's trajectories
//! and fits the policy to its actions with cross entropy — the same loss a language
//! model is trained with, over an action space instead of a vocabulary. On a
//! recurrent policy it is the same replay PPO uses, scored differently:
//!
//! ```text
//! L = -E_{(s,a*) ~ expert} [ log π_θ(a* | s_{≤t}) ] - c_H E[ H(π_θ) ]
//! ```
//!
//! # Why cloning alone is not enough
//!
//! An expert's trajectories only ever visit states the expert reaches. A cloned
//! policy is good there and unconstrained everywhere else, so its first mistake
//! takes it somewhere it has never been trained, where it makes a larger mistake.
//! The error compounds quadratically in the horizon rather than linearly, which for
//! a long episode is the difference between a policy that works and one that does
//! not.
//!
//! **DAgger** ([`DaggerSchedule`]) fixes the distribution rather than the loss: roll
//! out a *mixture* of the expert and the learner, label every state the mixture
//! visits with what the expert would have done, and train on that. As `beta` decays
//! from `1` to `0` the states drift from the expert's distribution to the learner's
//! own, so the policy is eventually trained on exactly the states it will face.
//! [`crate::tensor::ops::rl::mix_actions`] is the coin flip, on the device.
//!
//! # Why this pairs with [`super::ppo`]
//!
//! Cloning reaches a competent policy quickly and then stops: it can be no better
//! than the expert, and nothing in its loss mentions reward. Reinforcement learning
//! can exceed the expert but starts from nothing and spends most of its sample
//! budget getting to where cloning arrives in a few updates. Running one and then
//! the other is the standard answer, and both halves here are the same policy, the
//! same replay and the same episode mask — so the handover is a change of
//! [`crate::train::TrainStep`] and nothing else. `examples/train_rl.rs` does it.

use std::cell::Cell;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::models::mamba3::MixerCache;
use crate::nn::module::Module;
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::train::trainer::TrainStep;

use super::policy::Mamba3Policy;

/// How the expert's share of the acting decays over DAgger's rounds.
///
/// Round `0` is always pure expert — the first dataset has to be the expert's own
/// trajectories, because a policy that has learned nothing yet visits nowhere worth
/// labelling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DaggerSchedule {
    /// `beta = decay^round`. The usual choice; `0.5` halves the expert's share each
    /// round, reaching the learner's own distribution in a handful of them.
    Exponential {
        /// Per-round multiplier in `(0, 1]`.
        decay: f32,
    },
    /// `beta = 1 - round / rounds`, reaching zero at `rounds` and staying there.
    Linear {
        /// Round at which the expert stops acting entirely.
        rounds: u32,
    },
    /// Only round `0` uses the expert. The most aggressive schedule, and the one
    /// that is plain behaviour cloning followed by training on the learner's states.
    OnlyFirst,
    /// A constant mixture, for ablations.
    Fixed {
        /// The expert's share at every round.
        beta: f32,
    },
}

impl Default for DaggerSchedule {
    fn default() -> Self {
        Self::Exponential { decay: 0.5 }
    }
}

impl DaggerSchedule {
    /// The expert's share of the acting at `round`, counted from zero.
    pub fn beta(&self, round: u32) -> f32 {
        match *self {
            Self::Exponential { decay } => decay.clamp(0.0, 1.0).powi(round as i32),
            Self::Linear { rounds } => {
                if rounds == 0 {
                    0.0
                } else {
                    (1.0 - round as f32 / rounds as f32).max(0.0)
                }
            }
            Self::OnlyFirst => {
                if round == 0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Fixed { beta } => beta.clamp(0.0, 1.0),
        }
    }
}

/// A window of observations and the actions an expert took on them.
pub struct ImitationBatch<R: Runtime, E: FloatElem> {
    /// `[envs, steps, obs_dim]` observations.
    pub observations: Tensor<R, E>,
    /// `[envs, steps]` actions the expert would have taken.
    pub expert_actions: IdTensor<R>,
    /// `[envs, steps]` flags marking observations that begin an episode.
    pub reset: Option<Tensor<R, E>>,
    /// Recurrent state the window continues from.
    pub initial: Option<Vec<MixerCache<R, E>>>,
    /// `[envs, steps]` weights, `0` for a position with no expert label.
    ///
    /// DAgger needs this: a state the learner reached after the expert stopped
    /// acting may be one the expert cannot label, and a guessed label is worse than
    /// no label.
    pub mask: Option<Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> Clone for ImitationBatch<R, E> {
    fn clone(&self) -> Self {
        Self {
            observations: self.observations.clone(),
            expert_actions: self.expert_actions.clone(),
            reset: self.reset.clone(),
            initial: self.initial.clone(),
            mask: self.mask.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for ImitationBatch<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ImitationBatch({})", self.observations.shape())
    }
}

impl<R: Runtime, E: FloatElem> ImitationBatch<R, E> {
    /// A batch of observations and their expert labels.
    pub fn new(observations: Tensor<R, E>, expert_actions: IdTensor<R>) -> Self {
        Self {
            observations,
            expert_actions,
            reset: None,
            initial: None,
            mask: None,
        }
    }

    /// Mark the observations that begin an episode.
    pub fn with_reset(mut self, reset: Tensor<R, E>) -> Self {
        self.reset = Some(reset);
        self
    }

    /// Continue from a recurrent state.
    pub fn continuing_from(mut self, initial: Vec<MixerCache<R, E>>) -> Self {
        self.initial = Some(initial);
        self
    }

    /// Weight the positions that carry a usable expert label.
    pub fn with_mask(mut self, mask: Tensor<R, E>) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Number of environments.
    pub fn envs(&self) -> usize {
        self.observations.shape().dim(0)
    }

    /// Number of steps.
    pub fn steps(&self) -> usize {
        self.observations.shape().dim(1)
    }
}

/// Cross entropy of a policy's logits against an expert's actions, with an optional
/// entropy bonus.
///
/// `logits` is `[envs, steps, actions]` and `expert_actions` `[envs, steps]`.
/// `mask` weights the positions, `0` dropping one entirely.
///
/// The entropy term is subtracted, as in [`super::ppo`]: cloning an expert with
/// cross entropy alone drives the policy towards a deterministic copy, and a
/// deterministic policy is a bad starting point for the reinforcement learning that
/// usually follows, because it explores nothing.
pub fn behaviour_cloning_loss<R: Runtime, E: FloatElem>(
    logits: &Var<R, E>,
    expert_actions: &IdTensor<R>,
    mask: Option<&Tensor<R, E>>,
    entropy_coeff: f32,
) -> Result<Var<R, E>> {
    logits.shape().expect_rank(3)?;
    let dims = logits.dims();
    let (rows, classes) = (dims[0] * dims[1], dims[2]);
    if expert_actions.len() != rows {
        return Err(Error::shape(format!(
            "{rows} logit rows but {} expert actions",
            expert_actions.len()
        )));
    }
    if let Some(mask) = mask
        && mask.len() != rows
    {
        return Err(Error::shape(format!(
            "the mask holds {} weights, expected {rows}",
            mask.len()
        )));
    }

    let flat = logits.reshape(vec![rows, classes])?;
    let targets = expert_actions.reshape(vec![rows])?;
    // The fused kernel, same as language modelling: one launch each way, and no
    // dense one-hot in the backward pass.
    let per_step = flat.cross_entropy_rows(&targets, 0.0)?;

    let loss = match mask {
        None => per_step.mean()?,
        Some(mask) => {
            let mask = Var::constant(mask.reshape(vec![rows])?);
            let kept = mask.sum()?;
            let floor = Var::constant(Tensor::full(kept.shape().clone(), 1.0, kept.device()));
            per_step.mul(&mask)?.sum()?.div(&kept.maximum(&floor)?)?
        }
    };

    if entropy_coeff == 0.0 {
        return Ok(loss);
    }
    let log_probs = flat.log_softmax(1)?;
    let entropy = log_probs
        .exp()
        .mul(&log_probs)?
        .sum_dim(1)?
        .squeeze(1)?
        .neg();
    let entropy = match mask {
        None => entropy.mean()?,
        Some(mask) => {
            let mask = Var::constant(mask.reshape(vec![rows])?);
            let kept = mask.sum()?;
            let floor = Var::constant(Tensor::full(kept.shape().clone(), 1.0, kept.device()));
            entropy.mul(&mask)?.sum()?.div(&kept.maximum(&floor)?)?
        }
    };
    loss.sub(&entropy.mul_scalar(entropy_coeff))
}

/// A [`TrainStep`] that fits a [`Mamba3Policy`] to an expert's actions.
///
/// The critic head is along for the ride: nothing in behaviour cloning trains it,
/// and it receives no gradient because no term of the loss mentions it. That is
/// deliberate — a PPO run that follows will fit it against real returns, and a
/// critic pretrained on an expert's states would be confidently wrong about the
/// states the policy actually visits.
pub struct BehaviourCloningTask<'a, R: Runtime, E: FloatElem> {
    policy: &'a Mamba3Policy<R, E>,
    params: Vec<Param<R, E>>,
    entropy_coeff: f32,
    training: Cell<bool>,
}

impl<'a, R: Runtime, E: FloatElem> BehaviourCloningTask<'a, R, E> {
    /// Clone an expert into every parameter of `policy`.
    pub fn new(policy: &'a Mamba3Policy<R, E>) -> Self {
        Self {
            params: policy.parameters(),
            policy,
            entropy_coeff: 0.0,
            training: Cell::new(true),
        }
    }

    /// Keep the cloned policy stochastic by rewarding entropy.
    pub fn with_entropy_bonus(mut self, coeff: f32) -> Self {
        self.entropy_coeff = coeff;
        self
    }

    /// Train only the parameters whose path contains one of `patterns`.
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

    /// The policy being trained.
    pub fn policy(&self) -> &'a Mamba3Policy<R, E> {
        self.policy
    }

    /// Fraction of positions where the policy's most likely action is the expert's.
    ///
    /// This reads back, so it is a synchronisation: call it on a held-out batch
    /// between updates, not inside one.
    pub fn agreement(&self, batch: &ImitationBatch<R, E>) -> Result<f32> {
        let _guard = crate::autograd::no_grad();
        let (output, _) = self.policy.forward(
            &Var::constant(batch.observations.clone()),
            batch.reset.as_ref(),
            batch.initial.as_deref(),
        )?;
        let predicted =
            crate::tensor::ops::reduce::argmax(output.logits.tensor(), output.logits.rank() - 1)?
                .to_vec();
        let expected = batch.expert_actions.to_vec();
        let weights = batch.mask.as_ref().map(|m| m.to_f32());
        let mut hits = 0.0f32;
        let mut total = 0.0f32;
        for (i, (got, want)) in predicted.iter().zip(&expected).enumerate() {
            let w = weights.as_ref().map(|v| v[i]).unwrap_or(1.0);
            total += w;
            if got == want {
                hits += w;
            }
        }
        Ok(if total == 0.0 { 0.0 } else { hits / total })
    }
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for BehaviourCloningTask<'_, R, E> {
    type Batch = ImitationBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>> {
        let training = self.training.get();
        let guard = (!training).then(crate::autograd::no_grad);
        let observations = if training {
            Var::traced(batch.observations.clone())
        } else {
            Var::constant(batch.observations.clone())
        };
        let (output, _) = self.policy.forward(
            &observations,
            batch.reset.as_ref(),
            batch.initial.as_deref(),
        )?;
        let loss = behaviour_cloning_loss(
            &output.logits,
            &batch.expert_actions,
            batch.mask.as_ref(),
            self.entropy_coeff,
        );
        drop(guard);
        loss
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.policy.set_training(training);
    }
}
