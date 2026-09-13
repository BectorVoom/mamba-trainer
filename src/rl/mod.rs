//! On-device reinforcement learning: rollouts and trajectory training.
//!
//! Reinforcement learning asks a sequence model for two things that pull in
//! opposite directions. Acting wants the smallest possible step: one observation,
//! `B` environments, latency that does not grow with the episode. Learning wants
//! the largest possible batch: a whole `[B, T]` trajectory through one wide,
//! parallel pass. A transformer can only do the first by carrying a cache that
//! grows with `T`; a state space model does it in constant state, and this module
//! is the pair of engines that exploits that.
//!
//! | | entry point | cost per step | state |
//! |---|---|---|---|
//! | acting | [`Mamba3Policy::step`] | `O(1)` | `[B, L, H, P, N]`, fixed |
//! | learning | [`Mamba3Policy::forward`] | `O(T)`, `O(log T)` depth | the tape |
//!
//! # Episode boundaries
//!
//! The thing that makes the pair usable is that both honour the same termination
//! mask, and agree on what it means. A reset at position `t` must make everything
//! before `t` invisible to `t` and to everything after it, while leaving the
//! observation *at* `t` to seed the new episode. Two paths reach backwards in a
//! Mamba-3 layer and both are cut:
//!
//! * the recurrence, where the reset zeroes the two coefficients that carry the
//!   previous state (rollout) or floors the log-decay (scan) — see
//!   [`crate::ssm::scan::mamba3_step`];
//! * the short causal convolution, where taps that reach across a boundary are
//!   dropped — see [`crate::nn::conv::CausalConv1d::apply_masked`].
//!
//! Neither costs a launch of its own. The result is that `T` reset-aware rollout
//! steps and one reset-aware scan over the same `[B, T]` buffer compute the same
//! numbers, so a policy gradient taken from the second is a gradient of what the
//! first actually did.
//!
//! # What is built on that pair
//!
//! | | what it is | where |
//! |---|---|---|
//! | [`RolloutEngine`] | the policy plus the state of the environments it drives | [`rollout`] |
//! | [`TrajectoryBuffer`] | fixed-size `[B, T]` storage a step writes in place | [`buffer`] |
//! | [`VecEnv`] | environments that speak in device tensors | [`mod@env`] |
//! | [`GameLogic`] | a user's transition function, as device code | [`game`] |
//! | [`Collector`] | the loop joining the three, with nothing read back | [`collect`] |
//! | [`Collector::collect_fused`] | the same loop, one kernel a step | [`fused`] |
//! | [`MultiSyncCollector`] | the same loop with the environments on worker threads | [`parallel`] |
//! | [`PpoTask`] | the clipped surrogate, the critic and the entropy bonus | [`ppo`] |
//! | [`BehaviourCloningTask`] | cross entropy against an expert, and DAgger | [`imitation`] |
//!
//! The four operations that normally drag such a loop back to the host — sampling
//! an action, scoring it, recording it, and estimating its advantage — are kernels
//! in [`crate::tensor::ops::rl`], so a collection loop is a queue of launches and
//! not a conversation. `tests/rl_collect_footprint.rs` holds it to that: zero host
//! reads, flat bytes, flat dispatch count, however long it runs.
//!
//! Being kernels is not the end of it, because eight small launches in a row cost
//! eight dispatches whatever they compute. A game written as a [`GameLogic`] —
//! device code rather than a host object answering with tensors — lets the crate
//! compile the transition into the *same* kernel as the draw and the write, and
//! [`Collector::collect_fused`] is the resulting loop: one launch a step where the
//! [`VecEnv`] path takes eight, collecting a byte-for-byte identical window.
//!
//! # A learning loop
//!
//! ```no_run
//! use mamba3::prelude::*;
//! use mamba3::rl::{Mamba3PolicyConfig, PpoTask, RecallEnv};
//! use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};
//!
//! type R = mamba3::backends::Auto;
//! # fn main() -> mamba3::error::Result<()> {
//! let device = Device::<R>::default();
//! let mut env = RecallEnv::<R, f32>::new(64, 4, 8, 0, &device)?;
//! let obs_dim = env.obs_dim();
//! let policy = Mamba3PolicyConfig::new(obs_dim, 4, 128, 2).init::<R, f32>(&device)?;
//!
//! // Everything that persists is allocated here: the recurrent state and the
//! // trajectory buffer. The loop below adds nothing to either.
//! let mut collector = Collector::new(&policy, 64, 128, obs_dim, &device)?;
//! let config = PpoConfig::default();
//! let task = PpoTask::new(&policy, config);
//! let mut trainer = Trainer::new(
//!     TrainerConfig::builder().learning_rate(3e-4).build()?,
//!     AdamWConfig::builder().learning_rate(3e-4).build().init::<R, f32>(),
//! );
//!
//! for _ in 0..1_000 {
//!     let report = collector.collect(&mut env)?;
//!     let batch = collector.ppo_batch(&report, &config)?;
//!     for _ in 0..4 {
//!         trainer.step(&task, std::slice::from_ref(&batch))?;
//!     }
//! }
//! # Ok(())
//! # }
//! ```

pub mod buffer;
pub mod collect;
pub mod env;
pub mod fused;
pub mod game;
pub mod imitation;
pub mod parallel;
pub mod policy;
pub mod ppo;
pub mod rollout;
pub mod state;

pub use buffer::{Column, TrajectoryBuffer, Transition};
pub use collect::{CollectReport, Collector};
pub use env::{EnvStep, RecallEnv, VecEnv};
pub use fused::FusedStep;
pub use game::{GameLogic, GameSpec, GameWorld, Outcome};
pub use imitation::{
    BehaviourCloningTask, DaggerSchedule, ImitationBatch, behaviour_cloning_loss,
};
pub use parallel::{MultiSyncCollector, ParallelEnvs};
pub use policy::{Mamba3Policy, Mamba3PolicyConfig, PolicyOutput};
pub use ppo::{PpoBatch, PpoConfig, PpoLoss, PpoStats, PpoTask, ppo_objective, reference_log_probs};
pub use rollout::RolloutEngine;
pub use state::Mamba3StateBuffer;

pub use crate::tensor::ops::rl::{
    Advantages, Draw, draw_action, generalized_advantage, record_action, record_observation,
    record_outcome, sample_categorical,
};
