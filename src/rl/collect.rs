//! The collection loop: a policy, an environment, and the buffer between them.
//!
//! This is the part of a reinforcement learning run that executes millions of
//! times, and the only property that matters about it is that it does not
//! synchronise. Every operation in [`Collector::collect`] is queued — the policy
//! step, the action draw, the environment transition, the write into the trajectory
//! buffer — so the whole window goes to the device as one stream of launches and
//! the host never waits for it. `tests/rl_footprint.rs` holds it to that: a
//! thousand collected steps read back exactly zero times and reserve no additional
//! bytes.
//!
//! # Where a window starts and ends
//!
//! A window is not an episode. Environments auto-reset, episodes straddle window
//! boundaries, and the state buffer carries across them — which is precisely what a
//! state space policy makes cheap and what a transformer with a growing cache does
//! not. Three things are carried from one window to the next and each of them is a
//! bug if it is dropped:
//!
//! * the **recurrent state**, in the engine, so the policy remembers;
//! * the **observation** the rollout stopped at, so nothing is skipped;
//! * the **termination flag** of the last step, so the first observation of the new
//!   window is correctly marked as beginning an episode, or correctly not.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::MixerCache;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::rl::{mix_actions, sample_categorical, write_step_ids};

use super::buffer::{TrajectoryBuffer, Transition};
use super::env::VecEnv;
use super::imitation::ImitationBatch;
use super::policy::Mamba3Policy;
use super::ppo::{PpoBatch, PpoConfig};
use super::rollout::RolloutEngine;
use super::state::Mamba3StateBuffer;

/// What one collected window left behind.
pub struct CollectReport<R: Runtime, E: FloatElem> {
    /// `[envs]` critic estimate for the observation the window stopped at.
    ///
    /// The advantage estimator needs this to treat a truncated window as a window
    /// on a longer life rather than the end of the world. It is obtained from a
    /// *copy* of the recurrent state, so asking for it does not advance the rollout.
    pub bootstrap: Tensor<R, E>,
    /// The recurrent state the window started from, detached.
    ///
    /// Hand it to [`PpoBatch::continuing_from`] for truncated backpropagation
    /// through time, so the replay begins with the history the actor had.
    pub initial: Vec<MixerCache<R, E>>,
    /// Steps collected.
    pub steps: usize,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for CollectReport<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CollectReport(steps={})", self.steps)
    }
}

/// Drives a [`VecEnv`] with a [`Mamba3Policy`] and records what happens.
pub struct Collector<'a, R: Runtime, E: FloatElem> {
    engine: RolloutEngine<'a, R, E>,
    buffer: TrajectoryBuffer<R, E>,
    /// The observation the next step will act on. `None` until the first collect,
    /// which resets the environment.
    observation: Option<Tensor<R, E>>,
    last_done: Tensor<R, E>,
    expert_labels: Option<IdTensor<R>>,
    temperature: f32,
    seed: u64,
    draws: u64,
    device: Device<R>,
}

impl<'a, R: Runtime, E: FloatElem> Collector<'a, R, E> {
    /// Build a collector for windows of `steps` steps over `envs` environments.
    pub fn new(
        policy: &'a Mamba3Policy<R, E>,
        envs: usize,
        steps: usize,
        obs_dim: usize,
        device: &Device<R>,
    ) -> Result<Self> {
        Ok(Self {
            engine: RolloutEngine::new(policy, envs, device),
            buffer: TrajectoryBuffer::new(envs, steps, obs_dim, device)?,
            observation: None,
            last_done: Tensor::zeros(vec![envs], device),
            expert_labels: None,
            temperature: 1.0,
            seed: 0,
            draws: 0,
            device: device.clone(),
        })
    }

    /// Sampling temperature. `0` acts greedily, which is for evaluation — a greedy
    /// rollout collects no exploration and its log-probabilities make a degenerate
    /// importance ratio.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Seed for the action draws.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Also record what the environment's expert would have done at every step,
    /// which is what [`Collector::imitation_batch`] needs.
    pub fn recording_expert_labels(mut self) -> Self {
        let (envs, steps) = (self.buffer.envs(), self.buffer.steps());
        self.expert_labels = Some(IdTensor::empty(vec![envs, steps], &self.device));
        self
    }

    /// The buffer the last window was collected into.
    pub fn buffer(&self) -> &TrajectoryBuffer<R, E> {
        &self.buffer
    }

    /// The rollout engine, and through it the recurrent state.
    pub fn engine(&self) -> &RolloutEngine<'a, R, E> {
        &self.engine
    }

    /// Forget everything: zero the recurrent state and start the environment over
    /// on the next collect.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.observation = None;
        crate::tensor::ops::elemwise::fill_(&self.last_done, 0.0);
    }

    /// Collect one window, acting entirely on the policy.
    pub fn collect<V: VecEnv<R, E>>(&mut self, env: &mut V) -> Result<CollectReport<R, E>> {
        self.run(env, None)
    }

    /// Collect one window, acting on a mixture of the environment's expert and the
    /// policy — DAgger's rollout.
    ///
    /// `beta` is the expert's share. Whatever acts, the *label* recorded is always
    /// the expert's, because the point is to learn what the expert would do on the
    /// states the mixture reaches.
    ///
    /// # Not a PPO window
    ///
    /// Use the result through [`Collector::imitation_batch`], not
    /// [`Collector::ppo_batch`]. The recorded log-probability is the policy's for
    /// the action actually taken, so it is self-consistent — but the *behaviour*
    /// density here is the mixture's, `beta` of which came from the expert and not
    /// from any distribution the policy could have produced. A PPO ratio built from
    /// it would divide by the wrong denominator, and no amount of clipping notices.
    pub fn collect_with_expert<V: VecEnv<R, E>>(
        &mut self,
        env: &mut V,
        beta: f32,
    ) -> Result<CollectReport<R, E>> {
        if self.expert_labels.is_none() {
            return Err(Error::config(
                "this collector was not built with `recording_expert_labels`, \
                 so a DAgger rollout would have nothing to learn from"
                    .to_string(),
            ));
        }
        self.run(env, Some(beta))
    }

    fn run<V: VecEnv<R, E>>(
        &mut self,
        env: &mut V,
        beta: Option<f32>,
    ) -> Result<CollectReport<R, E>> {
        let envs = self.buffer.envs();
        let obs_dim = self.buffer.obs_dim();
        if env.envs() != envs || env.obs_dim() != obs_dim {
            return Err(Error::shape(format!(
                "the collector is built for {envs} environments of width {obs_dim}, \
                 the environment has {} of width {}",
                env.envs(),
                env.obs_dim()
            )));
        }

        if self.observation.is_none() {
            self.observation = Some(env.reset()?);
            crate::tensor::ops::elemwise::fill_(&self.last_done, 0.0);
        }
        let carry = self.last_done.clone();
        self.buffer.rewind(Some(&carry))?;
        // Taken before the first step, so a PPO replay can begin where the actor
        // did. Snapshotting copies, which is what makes it safe to hold while the
        // engine's own state is overwritten underneath it.
        let initial = self.engine.state().snapshot();

        let steps = self.buffer.steps();
        for t in 0..steps {
            let observation = self
                .observation
                .clone()
                .expect("the environment was reset above");
            let windowed = Var::constant(observation.reshape(vec![envs, 1, obs_dim])?);
            let reset_flag = if t == 0 { &carry } else { &self.last_done };
            let out = self.engine.step(&windowed, Some(reset_flag))?;

            let logits = out
                .logits
                .tensor()
                .reshape(vec![envs, out.logits.dims()[2]])?;
            self.draws = self.draws.wrapping_add(1);
            let draw_seed = self.seed ^ self.draws.wrapping_mul(0x9e3779b97f4a7c15);
            let (sampled, log_prob) = sample_categorical(&logits, self.temperature, draw_seed)?;

            // What the expert would do here, recorded before the environment moves
            // on — after `step` the label belongs to a different observation.
            let expert = env.expert_actions();
            let (executed, log_prob) = match (beta, &expert) {
                (Some(beta), Some(expert)) => {
                    let executed = mix_actions(&sampled, expert, beta, draw_seed.rotate_left(17))?;
                    // The mixture may have overridden the draw, and the recorded
                    // log-probability has to belong to the action that was actually
                    // taken — it is what a later importance ratio divides by. Scoring
                    // the sampled action instead would silently record the density of
                    // something that never happened.
                    let scored = Var::constant(logits.clone())
                        .log_softmax(1)?
                        .take_along_last(&executed)?;
                    (executed, scored.into_tensor())
                }
                (Some(_), None) => {
                    return Err(Error::config(
                        "a DAgger rollout needs an environment that can label a state, \
                         but this one does not implement `expert_actions`"
                            .to_string(),
                    ));
                }
                (None, _) => (sampled, log_prob),
            };
            if let (Some(labels), Some(expert)) = (&self.expert_labels, &expert) {
                write_step_ids(labels, expert, t)?;
            }

            let transition = env.step(&executed)?;
            self.buffer.push(Transition {
                observation: &observation,
                action: &executed,
                log_prob: &log_prob,
                value: &out.value.tensor().reshape(vec![envs])?,
                reward: &transition.reward,
                done: &transition.done,
            })?;

            self.last_done = transition.done;
            self.observation = Some(transition.observation);
        }

        Ok(CollectReport {
            bootstrap: self.bootstrap()?,
            initial,
            steps,
        })
    }

    /// The critic's estimate for the observation the window stopped at.
    ///
    /// Evaluated against a *copy* of the recurrent state. Stepping the engine itself
    /// would consume the observation the next window has to start from, and the two
    /// windows would then disagree about what the policy saw — the kind of
    /// off-by-one that shows up as a value function that never converges and never
    /// says why.
    fn bootstrap(&self) -> Result<Tensor<R, E>> {
        let envs = self.buffer.envs();
        let obs_dim = self.buffer.obs_dim();
        let observation = self
            .observation
            .as_ref()
            .ok_or_else(|| Error::config("nothing has been collected yet".to_string()))?;
        let mut scratch =
            Mamba3StateBuffer::new(self.engine.state().snapshot(), envs, &self.device);
        let windowed = Var::constant(observation.reshape(vec![envs, 1, obs_dim])?);
        let out = self
            .engine
            .policy()
            .step(&windowed, &mut scratch, Some(&self.last_done))?;
        out.value.tensor().reshape(vec![envs])
    }

    /// The collected window as a PPO batch.
    pub fn ppo_batch(
        &self,
        report: &CollectReport<R, E>,
        config: &PpoConfig,
    ) -> Result<PpoBatch<R, E>> {
        Ok(
            PpoBatch::from_buffer(&self.buffer, &report.bootstrap, config)?
                .continuing_from(report.initial.clone()),
        )
    }

    /// The collected window as an imitation batch, labelled by the expert.
    ///
    /// Requires [`Collector::recording_expert_labels`].
    pub fn imitation_batch(&self) -> Result<ImitationBatch<R, E>> {
        let labels = self.expert_labels.as_ref().ok_or_else(|| {
            Error::config(
                "this collector recorded no expert labels; \
                 build it with `recording_expert_labels`"
                    .to_string(),
            )
        })?;
        if self.buffer.is_empty() {
            return Err(Error::config("nothing has been collected yet".to_string()));
        }
        Ok(
            ImitationBatch::new(self.buffer.observations().clone(), labels.clone())
                .with_reset(self.buffer.reset_mask()?),
        )
    }

    /// Mean reward per completed episode in the last window, as a `[1]` device
    /// tensor.
    ///
    /// Computed on the device — reading it is the caller's synchronisation to make,
    /// and belongs between updates rather than inside one. Windows in which no
    /// episode finished report their total reward rather than dividing by zero.
    pub fn episode_return(&self) -> Result<Tensor<R, E>> {
        use crate::tensor::ops::{elemwise, reduce};
        let reward = reduce::sum_all(self.buffer.rewards())?;
        let episodes = reduce::sum_all(self.buffer.dones())?;
        elemwise::div(&reward, &elemwise::clamp(&episodes, 1.0, f32::MAX))
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Collector<'_, R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Collector(envs={}, steps={}, temperature={})",
            self.buffer.envs(),
            self.buffer.steps(),
            self.temperature,
        )
    }
}
