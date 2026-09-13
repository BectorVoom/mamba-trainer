//! The trajectory buffer: what a rollout writes and a learning step reads.
//!
//! A rollout produces `T` steps of `B` environments one step at a time, and a
//! learning step wants all of them at once as a `[B, T]` window. The buffer is the
//! transposition between those two shapes, and like [`super::state`] it is
//! allocated once and only ever overwritten: a step writes its own column in place
//! (see [`crate::tensor::ops::rl::write_step`]), so a collection loop of any length
//! allocates nothing after the first call.
//!
//! # The two masks
//!
//! Termination appears twice in reinforcement learning, one step apart, and mixing
//! them up is the classic off-by-one that produces a policy gradient nobody can
//! debug. The buffer names both:
//!
//! * **`done[t]`** — the transition at `t` *ended* an episode. This is what the
//!   advantage estimator needs, because it decides whether the value of what
//!   follows step `t` is the next step's estimate or zero.
//! * **`reset[t]`** — the observation at `t` *begins* an episode. This is what the
//!   policy needs, because it decides whether the recurrence and the convolution
//!   may see anything before `t`.
//!
//! They are the same events shifted by one: `reset[t] = done[t-1]`, with `reset[0]`
//! coming from whatever the previous window left behind. [`TrajectoryBuffer::reset_mask`]
//! performs exactly that shift, so no caller has to get it right twice.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::rl::{write_step, write_step_ids};
use crate::tensor::ops::{elemwise, movement};

/// One step of every environment, as a rollout produces it.
///
/// Every field is a device tensor and none of them is read on the way in, so
/// recording a step adds kernel launches to the queue and nothing else.
pub struct Transition<'a, R: Runtime, E: FloatElem> {
    /// `[envs, obs_dim]` observation the action was chosen from.
    pub observation: &'a Tensor<R, E>,
    /// `[envs]` action taken.
    pub action: &'a IdTensor<R>,
    /// `[envs]` log-probability of that action under the policy that took it.
    pub log_prob: &'a Tensor<R, E>,
    /// `[envs]` critic estimate for `observation`.
    pub value: &'a Tensor<R, E>,
    /// `[envs]` reward the action earned.
    pub reward: &'a Tensor<R, E>,
    /// `[envs]` flag: `1` if this transition ended the episode.
    pub done: &'a Tensor<R, E>,
    /// `[envs, action_dim]` legal-action mask for `observation`, if the
    /// environment provides one. `None` means every action was legal.
    ///
    /// Whether this is `Some` must not change across the life of one
    /// [`TrajectoryBuffer`]: the first transition it sees decides whether a
    /// mask column exists at all, and every later one is held to that.
    pub action_mask: Option<&'a Tensor<R, E>>,
}

/// Where one step of a rollout is stored, handed to a kernel that writes it all.
///
/// Every field is the buffer's own tensor rather than a copy, so a kernel holding
/// this writes straight into the window. `t` is the column, already checked to be
/// in range.
///
/// Public because [`super::Collector::collect_with`] hands one to a caller's own
/// fused rollout kernel; the six tensors are exactly the six
/// [`crate::tensor::ops::rl::step`]'s recorders write.
pub struct Column<'a, R: Runtime, E: FloatElem> {
    /// `[envs, steps, obs_dim]`.
    pub observations: &'a Tensor<R, E>,
    /// `[envs, steps]`.
    pub actions: &'a IdTensor<R>,
    /// `[envs, steps]`.
    pub log_probs: &'a Tensor<R, E>,
    /// `[envs, steps]`.
    pub values: &'a Tensor<R, E>,
    /// `[envs, steps]`.
    pub rewards: &'a Tensor<R, E>,
    /// `[envs, steps]`.
    pub dones: &'a Tensor<R, E>,
    /// The column to write.
    pub t: usize,
}

/// Fixed-size storage for a `[envs, steps]` rollout.
///
/// # Footprint
///
/// `envs * steps * (obs_dim + 5)` floats plus `envs * steps` ids, fixed at
/// construction. [`TrajectoryBuffer::rewind`] makes the buffer reusable without
/// reallocating it, which is what lets a training run of a million steps hold the
/// same bytes as its first window.
pub struct TrajectoryBuffer<R: Runtime, E: FloatElem> {
    observations: Tensor<R, E>,
    actions: IdTensor<R>,
    log_probs: Tensor<R, E>,
    values: Tensor<R, E>,
    rewards: Tensor<R, E>,
    dones: Tensor<R, E>,
    /// Whether each environment's episode had ended before the window's first
    /// observation. Carried from the previous window so `reset[0]` is right.
    initial_done: Tensor<R, E>,
    /// `[envs, steps, action_dim]`, allocated the first time a pushed
    /// transition carries a mask. `None` for the life of the buffer if the
    /// environment never does.
    action_mask: Option<Tensor<R, E>>,
    envs: usize,
    steps: usize,
    obs_dim: usize,
    cursor: usize,
    device: Device<R>,
}

impl<R: Runtime, E: FloatElem> TrajectoryBuffer<R, E> {
    /// Allocate storage for `steps` steps of `envs` environments.
    pub fn new(envs: usize, steps: usize, obs_dim: usize, device: &Device<R>) -> Result<Self> {
        if envs == 0 || steps == 0 || obs_dim == 0 {
            return Err(Error::config(format!(
                "a trajectory buffer needs positive dimensions, got \
                 envs={envs}, steps={steps}, obs_dim={obs_dim}"
            )));
        }
        let pair = || Tensor::<R, E>::zeros(vec![envs, steps], device);
        Ok(Self {
            observations: Tensor::zeros(vec![envs, steps, obs_dim], device),
            actions: IdTensor::empty(vec![envs, steps], device),
            log_probs: pair(),
            values: pair(),
            rewards: pair(),
            dones: pair(),
            initial_done: Tensor::zeros(vec![envs], device),
            action_mask: None,
            envs,
            steps,
            obs_dim,
            cursor: 0,
            device: device.clone(),
        })
    }

    /// Number of environments.
    pub fn envs(&self) -> usize {
        self.envs
    }

    /// Capacity in steps.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Observation width.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Steps recorded so far.
    pub fn len(&self) -> usize {
        self.cursor
    }

    /// Whether nothing has been recorded since the last [`TrajectoryBuffer::rewind`].
    pub fn is_empty(&self) -> bool {
        self.cursor == 0
    }

    /// Whether the buffer has no room for another step.
    pub fn is_full(&self) -> bool {
        self.cursor >= self.steps
    }

    /// The device the buffer lives on.
    pub fn device(&self) -> &Device<R> {
        &self.device
    }

    /// Bytes held on the device, fixed once an action mask has appeared or not
    /// — which happens on the buffer's very first push.
    pub fn bytes(&self) -> usize {
        let floats = self.observations.len()
            + self.log_probs.len()
            + self.values.len()
            + self.rewards.len()
            + self.dones.len()
            + self.initial_done.len()
            + self.action_mask.as_ref().map_or(0, Tensor::len);
        floats * core::mem::size_of::<E>() + self.actions.len() * 4
    }

    /// `[envs, steps, obs_dim]` observations.
    pub fn observations(&self) -> &Tensor<R, E> {
        &self.observations
    }

    /// `[envs, steps]` actions.
    pub fn actions(&self) -> &IdTensor<R> {
        &self.actions
    }

    /// `[envs, steps]` log-probabilities under the behaviour policy.
    pub fn log_probs(&self) -> &Tensor<R, E> {
        &self.log_probs
    }

    /// `[envs, steps]` critic estimates made during the rollout.
    pub fn values(&self) -> &Tensor<R, E> {
        &self.values
    }

    /// `[envs, steps]` rewards.
    pub fn rewards(&self) -> &Tensor<R, E> {
        &self.rewards
    }

    /// `[envs, steps]` termination flags: `1` where the transition ended an episode.
    pub fn dones(&self) -> &Tensor<R, E> {
        &self.dones
    }

    /// `[envs, steps, action_dim]` legal-action mask, if the environment
    /// provided one on the buffer's first push. `None` otherwise.
    pub fn action_mask(&self) -> Option<&Tensor<R, E>> {
        self.action_mask.as_ref()
    }

    /// Record one step. Fails once the buffer is full.
    pub fn push(&mut self, step: Transition<'_, R, E>) -> Result<()> {
        if self.is_full() {
            return Err(Error::config(format!(
                "the trajectory buffer already holds its {} steps; \
                 learn from it and rewind before collecting more",
                self.steps
            )));
        }
        self.check(
            step.observation.len(),
            self.envs * self.obs_dim,
            "observation",
        )?;
        self.check(step.action.len(), self.envs, "action")?;
        self.check(step.log_prob.len(), self.envs, "log_prob")?;
        self.check(step.value.len(), self.envs, "value")?;
        self.check(step.reward.len(), self.envs, "reward")?;
        self.check(step.done.len(), self.envs, "done")?;

        match step.action_mask {
            Some(mask) => {
                if self.action_mask.is_none() {
                    // The first mask this buffer has ever seen decides the
                    // action width, and allocates storage for the rest of its
                    // life -- absent for good if this branch is never taken.
                    let action_dim = mask.len() / self.envs.max(1);
                    if action_dim == 0 || mask.len() != self.envs * action_dim {
                        return Err(Error::shape(format!(
                            "an action mask holds {} elements, not a multiple \
                             of {} environments",
                            mask.len(),
                            self.envs
                        )));
                    }
                    self.action_mask =
                        Some(Tensor::zeros(vec![self.envs, self.steps, action_dim], &self.device));
                }
                let buffer = self.action_mask.as_ref().expect("just allocated above");
                self.check(mask.len(), buffer.len() / self.steps, "action_mask")?;
                write_step(buffer, mask, self.cursor)?;
            }
            None if self.action_mask.is_some() => {
                return Err(Error::config(
                    "this environment provided an action mask on an earlier step of \
                     this window but not this one; a mask must be all-or-nothing for \
                     the life of a collector"
                        .to_string(),
                ));
            }
            None => {}
        }

        let t = self.cursor;
        write_step(&self.observations, step.observation, t)?;
        write_step_ids(&self.actions, step.action, t)?;
        for (buffer, value) in [
            (&self.log_probs, step.log_prob),
            (&self.values, step.value),
            (&self.rewards, step.reward),
            (&self.dones, step.done),
        ] {
            write_step(buffer, value, t)?;
        }
        self.cursor += 1;
        Ok(())
    }

    /// The storage behind the next unwritten column, for a kernel that fills the
    /// whole column itself.
    ///
    /// [`TrajectoryBuffer::push`] writes six of these from six tensors the caller
    /// already has, one launch each. A fused rollout step has the same six values
    /// in registers and can store them directly, which is what this exposes: the
    /// buffers and the column index, with the bounds check `push` would have done.
    /// Pair it with [`TrajectoryBuffer::commit`] once the launch is queued.
    pub fn column(&self) -> Result<Column<'_, R, E>> {
        if self.is_full() {
            return Err(Error::config(format!(
                "the trajectory buffer already holds its {} steps; \
                 learn from it and rewind before collecting more",
                self.steps
            )));
        }
        Ok(Column {
            observations: &self.observations,
            actions: &self.actions,
            log_probs: &self.log_probs,
            values: &self.values,
            rewards: &self.rewards,
            dones: &self.dones,
            t: self.cursor,
        })
    }

    /// Record that the column [`TrajectoryBuffer::column`] handed out is written.
    pub(crate) fn commit(&mut self) {
        self.cursor += 1;
    }

    fn check(&self, got: usize, want: usize, what: &str) -> Result<()> {
        if got == want {
            return Ok(());
        }
        Err(Error::shape(format!(
            "a transition's {what} holds {got} elements, expected {want} \
             for {} environments",
            self.envs
        )))
    }

    /// Make the buffer ready for another window without reallocating it.
    ///
    /// `carry` is whether each environment's episode had ended on the last step of
    /// the window just finished — pass [`TrajectoryBuffer::last_done`] — so that the
    /// next window's first observation is correctly marked as beginning an episode.
    pub fn rewind(&mut self, carry: Option<&Tensor<R, E>>) -> Result<()> {
        match carry {
            Some(done) => {
                self.check(done.len(), self.envs, "carry")?;
                elemwise::fill_(&self.initial_done, 0.0);
                elemwise::add_assign_(&self.initial_done, &done.reshape(vec![self.envs])?);
            }
            None => elemwise::fill_(&self.initial_done, 0.0),
        }
        self.cursor = 0;
        Ok(())
    }

    /// `[envs]` termination flags of the last recorded step.
    ///
    /// This is what [`TrajectoryBuffer::rewind`] wants, and what a rollout loop
    /// hands to the policy as the reset flag for the next window's first step.
    pub fn last_done(&self) -> Result<Tensor<R, E>> {
        if self.cursor == 0 {
            return Ok(self.initial_done.clone());
        }
        movement::slice(&self.dones, 1, self.cursor - 1, 1)?.reshape(vec![self.envs])
    }

    /// `[envs, steps]` flags marking the observations that begin an episode.
    ///
    /// This is `done` shifted one step later, with the first column carried in from
    /// the previous window — the mask [`super::Mamba3Policy::forward`] wants, and
    /// the one the rollout was driven with, so the scan reproduces the rollout.
    pub fn reset_mask(&self) -> Result<Tensor<R, E>> {
        let filled = self.filled()?;
        let shifted = movement::shift_right(&filled, 1)?;
        // `shift_right` zero-fills the first column; the carry belongs there.
        let head = self.initial_done.reshape(vec![self.envs, 1])?;
        let tail = movement::slice(&shifted, 1, 1, shifted.shape().dim(1) - 1)?;
        movement::cat(&[head, tail], 1)
    }

    /// The recorded prefix of `dones`, which is the whole buffer once it is full.
    fn filled(&self) -> Result<Tensor<R, E>> {
        if self.cursor == self.steps {
            return Ok(self.dones.clone());
        }
        if self.cursor == 0 {
            return Err(Error::config(
                "the trajectory buffer holds no steps yet".to_string(),
            ));
        }
        movement::slice(&self.dones, 1, 0, self.cursor)
    }

    /// The recorded window as a learning batch, with the critic's estimate of what
    /// lies beyond it.
    ///
    /// `bootstrap` is `[envs]`: the value of the observation the rollout stopped at.
    /// Where the last step terminated, it is ignored.
    pub fn advantages(
        &self,
        bootstrap: &Tensor<R, E>,
        gamma: f32,
        lambda: f32,
    ) -> Result<crate::tensor::ops::rl::Advantages<R, E>> {
        let window = self.cursor;
        if window == 0 {
            return Err(Error::config(
                "cannot estimate advantages from an empty trajectory buffer".to_string(),
            ));
        }
        let trim = |t: &Tensor<R, E>| -> Result<Tensor<R, E>> {
            if window == self.steps {
                Ok(t.clone())
            } else {
                movement::slice(t, 1, 0, window)
            }
        };
        crate::tensor::ops::rl::generalized_advantage(
            &trim(&self.rewards)?,
            &trim(&self.values)?,
            &trim(&self.dones)?,
            &bootstrap.reshape(vec![self.envs])?,
            gamma,
            lambda,
        )
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for TrajectoryBuffer<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TrajectoryBuffer(envs={}, {}/{} steps, obs_dim={}, {} KiB)",
            self.envs,
            self.cursor,
            self.steps,
            self.obs_dim,
            self.bytes() / 1024,
        )
    }
}
