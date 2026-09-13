//! An actor-critic policy over a Mamba-3 stack.
//!
//! The policy is deliberately thin — an observation encoder, a stack of Mamba-3
//! blocks, and two linear heads — because the interesting part is that the *same*
//! stack serves both of the shapes reinforcement learning needs:
//!
//! * [`Mamba3Policy::step`] advances `B` environments by one observation each,
//!   in `O(1)` per step regardless of how long their episodes have run;
//! * [`Mamba3Policy::forward`] consumes a whole `[B, T]` rollout buffer through
//!   the chunked parallel scan, in `O(T)` with `O(log T)`-depth dependencies.
//!
//! Both honour the same `[B]` / `[B, T]` episode-termination mask, and they agree
//! numerically — which is the property that makes the pair usable, because a
//! policy gradient estimated from the second must be a gradient of what the first
//! actually did.
//!
//! # What is stored for the backward pass
//!
//! Nothing explicitly. The scan is written out of differentiable primitives, so
//! the tape already holds exactly the intermediates its own adjoint needs and
//! releases them when [`crate::autograd::Var::backward`] runs. There is no
//! separate trajectory-of-states buffer to size, fill or keep in sync.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::distributions::Categorical;
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Block, Mamba3BlockConfig, MixerCache};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::ssm::config::SsmConfig;
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

use super::state::Mamba3StateBuffer;

/// What the policy produces for every position it is given.
pub struct PolicyOutput<R: Runtime, E: FloatElem> {
    /// Action logits, `[batch, seq, action_dim]`.
    pub logits: Var<R, E>,
    /// State value, `[batch, seq]`.
    pub value: Var<R, E>,
}

impl<R: Runtime, E: FloatElem> PolicyOutput<R, E> {
    /// The action distribution these logits describe.
    ///
    /// The bridge from a policy to [`crate::distributions`]: everything a
    /// policy-gradient loss asks of a policy — the log-probability of what it did,
    /// the entropy of what it might have done, the divergence from what it used to
    /// be — is a method on the returned [`Categorical`], and each one is a single
    /// fused kernel over the logit rows.
    ///
    /// ```no_run
    /// # use mamba3::prelude::*;
    /// # use mamba3::distributions::Distribution;
    /// # fn go<R: cubecl::prelude::Runtime>(out: &mamba3::rl::PolicyOutput<R, f32>,
    /// #        actions: &mamba3::tensor::ops::index::IdTensor<R>) -> Result<()> {
    /// let policy = out.distribution()?;
    /// let log_prob = policy.log_prob_ids(actions)?;
    /// let bonus = policy.entropy()?;
    /// # Ok(()) }
    /// ```
    pub fn distribution(&self) -> Result<Categorical<R, E>> {
        Categorical::from_logits(self.logits.clone())
    }
}

impl<R: Runtime, E: FloatElem> Clone for PolicyOutput<R, E> {
    fn clone(&self) -> Self {
        Self {
            logits: self.logits.clone(),
            value: self.value.clone(),
        }
    }
}

/// Configuration for [`Mamba3Policy`].
#[derive(Debug, Clone)]
pub struct Mamba3PolicyConfig {
    /// Width of one observation vector.
    pub obs_dim: usize,
    /// Number of discrete actions.
    pub action_dim: usize,
    /// Number of Mamba-3 blocks.
    pub n_layers: usize,
    /// The mixer configuration; `d_model` comes from here.
    pub ssm: SsmConfig,
    /// Normalisation epsilon.
    pub norm_eps: f32,
    /// Initialisation seed.
    pub seed: u64,
}

impl Mamba3PolicyConfig {
    /// A policy sized for on-device rollouts.
    ///
    /// The defaults are the small end of the architecture — one `B`/`C` group per
    /// head, a rotational transition, real state tracking — with `d_model` split
    /// evenly into `n_heads` of `head_dim`.
    pub fn new(obs_dim: usize, action_dim: usize, d_model: usize, n_layers: usize) -> Self {
        let head_dim = 64.min(d_model.max(1));
        let n_heads = (d_model / head_dim).max(1);
        Self {
            obs_dim,
            action_dim,
            n_layers,
            ssm: SsmConfig {
                d_model,
                n_heads,
                head_dim,
                d_state: 16,
                n_groups: n_heads,
                ..SsmConfig::default()
            },
            norm_eps: 1e-5,
            seed: 0,
        }
    }

    /// Edit the mixer configuration.
    pub fn with_ssm(mut self, f: impl FnOnce(&mut SsmConfig)) -> Self {
        f(&mut self.ssm);
        self
    }

    /// Initialisation seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Check internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.obs_dim == 0 || self.action_dim == 0 {
            return Err(Error::config(
                "a policy needs a positive observation and action width".to_string(),
            ));
        }
        if self.n_layers == 0 {
            return Err(Error::config(
                "a policy needs at least one layer".to_string(),
            ));
        }
        self.ssm.validate()
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(&self, device: &Device<R>) -> Result<Mamba3Policy<R, E>> {
        let mut rng = Rng::seeded(self.seed);
        self.init_with_rng(device, &mut rng)
    }

    /// Instantiate with an explicit RNG.
    pub fn init_with_rng<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Mamba3Policy<R, E>> {
        self.validate()?;
        let d_model = self.ssm.d_model;
        let normal = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let block = Mamba3BlockConfig::new(self.ssm.clone())
            .with_norm_eps(self.norm_eps)
            .mixer(|m| m.with_depth(self.n_layers));
        let blocks = (0..self.n_layers)
            .map(|_| block.init(device, rng))
            .collect::<Result<Vec<_>>>()?;

        Ok(Mamba3Policy {
            encoder: LinearConfig::new(self.obs_dim, d_model)
                .with_initializer(normal)
                .init(device, rng),
            blocks,
            norm: RmsNormConfig::new(d_model)
                .with_eps(self.norm_eps)
                .init(device, rng),
            // A near-zero actor head starts the policy close to uniform, which is
            // what keeps the first rollouts exploratory instead of committed.
            actor: LinearConfig::new(d_model, self.action_dim)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.01,
                })
                .init(device, rng),
            critic: LinearConfig::new(d_model, 1)
                .with_initializer(normal)
                .init(device, rng),
            config: self.clone(),
        })
    }
}

/// A recurrent actor-critic policy.
pub struct Mamba3Policy<R: Runtime, E: FloatElem> {
    encoder: Linear<R, E>,
    blocks: Vec<Mamba3Block<R, E>>,
    norm: RmsNorm<R, E>,
    actor: Linear<R, E>,
    critic: Linear<R, E>,
    config: Mamba3PolicyConfig,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Mamba3Policy<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mamba3Policy(obs={}, actions={}, d_model={}, layers={})",
            self.config.obs_dim,
            self.config.action_dim,
            self.config.ssm.d_model,
            self.blocks.len(),
        )
    }
}

impl<R: Runtime, E: FloatElem> Mamba3Policy<R, E> {
    /// The configuration this policy was built from.
    pub fn config(&self) -> &Mamba3PolicyConfig {
        &self.config
    }

    /// The blocks in the stack.
    pub fn blocks(&self) -> &[Mamba3Block<R, E>] {
        &self.blocks
    }

    /// Allocate the rollout state for `envs` environments.
    ///
    /// This is the only allocation an RL loop needs to make: the returned buffer
    /// is written in place from then on. Do it before the loop starts.
    pub fn empty_state(&self, envs: usize, device: &Device<R>) -> Mamba3StateBuffer<R, E> {
        Mamba3StateBuffer::new(
            self.blocks
                .iter()
                .map(|b| b.empty_cache(envs, device))
                .collect(),
            envs,
            device,
        )
    }

    /// Check that an observation window has the shape the encoder expects.
    fn check_obs(&self, obs: &Var<R, E>) -> Result<(usize, usize)> {
        obs.shape().expect_rank(3)?;
        let dims = obs.dims();
        if dims[2] != self.config.obs_dim {
            return Err(Error::shape(format!(
                "Mamba3Policy expects obs_dim={}, got {}",
                self.config.obs_dim,
                obs.shape()
            )));
        }
        Ok((dims[0], dims[1]))
    }

    /// Heads over the stack's output.
    fn heads(&self, hidden: &Var<R, E>, batch: usize, seq: usize) -> Result<PolicyOutput<R, E>> {
        let hidden = self.norm.apply(hidden)?;
        Ok(PolicyOutput {
            logits: self.actor.apply(&hidden)?,
            value: self.critic.apply(&hidden)?.reshape(vec![batch, seq])?,
        })
    }

    /// One rollout step for `B` environments.
    ///
    /// `obs` is `[envs, 1, obs_dim]` and `reset` an optional `[envs]` mask holding
    /// `1` for an environment whose previous episode ended. The state buffer is
    /// advanced in place; nothing leaves the device and no host synchronisation
    /// happens, so the whole step is one queue of kernel launches.
    ///
    /// Runs with the tape disabled — a rollout is data collection, and a cache
    /// that kept a graph alive would grow without bound.
    pub fn step(
        &self,
        obs: &Var<R, E>,
        state: &mut Mamba3StateBuffer<R, E>,
        reset: Option<&Tensor<R, E>>,
    ) -> Result<PolicyOutput<R, E>> {
        let _guard = crate::autograd::no_grad();
        let (batch, seq) = self.check_obs(obs)?;
        if seq != 1 {
            return Err(Error::shape(format!(
                "step() takes a single observation per environment; got {seq} positions. \
                 Use forward() for a window"
            )));
        }
        if batch != state.envs() {
            return Err(Error::shape(format!(
                "observation is for {batch} environments but the state buffer holds {}",
                state.envs()
            )));
        }
        if state.len() != self.blocks.len() {
            return Err(Error::shape(format!(
                "state buffer has {} layers, the policy has {}",
                state.len(),
                self.blocks.len()
            )));
        }

        let mut x = self.encoder.apply(obs)?;
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, cache) = block.step_masked(&x, state.layer(i)?, reset)?;
            state.store(i, cache)?;
            x = out;
        }
        self.heads(&x, batch, 1)
    }

    /// A whole rollout window, through the parallel scan.
    ///
    /// `obs` is `[batch, seq, obs_dim]` and `reset` an optional `[batch, seq]`
    /// mask marking the positions that begin a new episode. `initial` continues
    /// from a state the rollout left behind — pass
    /// [`Mamba3StateBuffer::snapshot`] taken before the rollout began, so the
    /// gradient sees the same history the actor did.
    ///
    /// This is the differentiable path: the result carries a tape, and
    /// [`crate::autograd::Var::backward`] on a loss built from it produces the
    /// reverse scan over the whole window.
    ///
    /// The second return value is the state at the window's right edge, present
    /// only when `initial` was — truncated backpropagation through time hands it
    /// to the next window. It is on the tape, so
    /// [`MixerCache::detach`] it before keeping it, or the graph
    /// stays alive for as long as the state does.
    #[allow(clippy::type_complexity)] // One output plus an optional per-layer state.
    pub fn forward(
        &self,
        obs: &Var<R, E>,
        reset: Option<&Tensor<R, E>>,
        initial: Option<&[MixerCache<R, E>]>,
    ) -> Result<(PolicyOutput<R, E>, Option<Vec<MixerCache<R, E>>>)> {
        let (batch, seq) = self.check_obs(obs)?;
        if let Some(reset) = reset
            && reset.len() != batch * seq
        {
            return Err(Error::shape(format!(
                "reset mask must be [batch, seq] = [{batch}, {seq}], got {}",
                reset.shape()
            )));
        }
        if let Some(initial) = initial
            && initial.len() != self.blocks.len()
        {
            return Err(Error::shape(format!(
                "initial state has {} layers, the policy has {}",
                initial.len(),
                self.blocks.len()
            )));
        }

        let mut x = self.encoder.apply(obs)?;
        let mut ends = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = initial.map(|c| &c[i]);
            let (out, end) = block.apply_with_state_masked(&x, cache, reset)?;
            if let Some(end) = end {
                ends.push(end);
            }
            x = out;
        }
        let ends = (ends.len() == self.blocks.len()).then_some(ends);
        Ok((self.heads(&x, batch, seq)?, ends))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Mamba3Policy<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("encoder", &self.encoder);
        for (i, block) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, block);
        }
        visitor.child("norm", &self.norm);
        visitor.child("actor", &self.actor);
        visitor.child("critic", &self.critic);
    }
}
