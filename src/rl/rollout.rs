//! The rollout engine: `B` environments advancing one step at a time.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::Result;
use crate::tensor::Tensor;

use super::policy::{Mamba3Policy, PolicyOutput};
use super::state::Mamba3StateBuffer;

/// A policy plus the state of the environments it is driving.
///
/// The engine exists to make the allocation boundary explicit. Everything that
/// persists across steps — one hidden state and one convolution history per layer
/// per environment — is allocated when the engine is built, and every
/// [`RolloutEngine::step`] afterwards overwrites it. What a step allocates
/// instead is its own intermediates, which are transient by construction: they
/// come from CubeCL's pooled device allocator, are released as the step's tensors
/// drop, and are handed straight back to the next step. The steady-state
/// footprint is therefore flat, which is what
/// `rollout_footprint_is_flat_over_many_steps` in `tests/rl.rs` measures.
///
/// # Staying on the device
///
/// Nothing here reads a tensor back to the host. Observations and termination
/// flags arrive as device tensors and results leave as device tensors, so a full
/// step is a queue of kernel launches with no synchronisation point. Calling
/// `to_f32()` on an output is what would end that — do it outside the loop, or on
/// a summary, never per step.
pub struct RolloutEngine<'a, R: Runtime, E: FloatElem> {
    policy: &'a Mamba3Policy<R, E>,
    state: Mamba3StateBuffer<R, E>,
    steps: u64,
}

impl<'a, R: Runtime, E: FloatElem> RolloutEngine<'a, R, E> {
    /// Build an engine and pre-allocate the state for `envs` environments.
    pub fn new(policy: &'a Mamba3Policy<R, E>, envs: usize, device: &Device<R>) -> Self {
        Self {
            policy,
            state: policy.empty_state(envs, device),
            steps: 0,
        }
    }

    /// The policy being rolled out.
    pub fn policy(&self) -> &'a Mamba3Policy<R, E> {
        self.policy
    }

    /// The persistent state.
    pub fn state(&self) -> &Mamba3StateBuffer<R, E> {
        &self.state
    }

    /// The persistent state, mutably — for an explicit
    /// [`Mamba3StateBuffer::reset_env_states`] outside a step.
    pub fn state_mut(&mut self) -> &mut Mamba3StateBuffer<R, E> {
        &mut self.state
    }

    /// Number of environments.
    pub fn envs(&self) -> usize {
        self.state.envs()
    }

    /// How many steps have been taken.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Advance every environment by one observation.
    ///
    /// `obs` is `[envs, 1, obs_dim]`. `done` is an optional `[envs]` mask holding
    /// `1` for an environment whose episode ended *before* this observation — so
    /// the observation is the first of the new episode and must not see the old
    /// one. The mask is folded into the step's own kernels rather than applied as
    /// a separate pass over the state, so terminations are free.
    pub fn step(
        &mut self,
        obs: &Var<R, E>,
        done: Option<&Tensor<R, E>>,
    ) -> Result<PolicyOutput<R, E>> {
        let out = self.policy.step(obs, &mut self.state, done)?;
        self.steps += 1;
        Ok(out)
    }

    /// Zero every environment's state and the step counter.
    pub fn reset(&mut self) {
        self.state.reset_all();
        self.steps = 0;
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for RolloutEngine<'_, R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "RolloutEngine(envs={}, steps={}, state={:?})",
            self.envs(),
            self.steps,
            self.state,
        )
    }
}
