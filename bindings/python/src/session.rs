//! A collector kept alive beside the policy it borrows.
//!
//! [`mamba3::rl::Collector`] borrows the policy for as long as it collects, which
//! is exactly right in Rust and impossible to express through a Python object:
//! `learner` and `policy` are two independent handles with independent lifetimes,
//! and Python will drop them in whichever order it likes. [`Session`] is the one
//! place in these bindings that resolves that, and it resolves it once so that
//! nothing above has to think about it.

use std::rc::Rc;

use mamba3::backend::Device;
use mamba3::error::Result;
use mamba3::rl::{Collector, Mamba3Policy};

use crate::{E, R};

/// The persistent half of a learning loop: the recurrent state of every
/// environment and the `[envs, steps]` trajectory buffer, allocated once.
pub struct Session {
    /// Declared first, so it is dropped first: fields drop in declaration order
    /// and this one holds a reference into `policy`.
    collector: Collector<'static, R, E>,
    policy: Rc<Mamba3Policy<R, E>>,
}

impl Session {
    /// Build a collector for windows of `steps` steps over `envs` environments.
    pub fn new(
        policy: Rc<Mamba3Policy<R, E>>,
        envs: usize,
        steps: usize,
        obs_dim: usize,
        temperature: f32,
        seed: u64,
        expert_labels: bool,
        device: &Device<R>,
    ) -> Result<Self> {
        // SAFETY: the policy is behind an `Rc`, so it is heap-allocated and does
        // not move while this struct holds a strong count; the collector is dropped
        // before that count is released, so the borrow cannot outlive what it
        // points at. It is a shared reference and stays one — a `Param` is
        // interior-mutable, so training updates the weights through `&`, and no
        // `&mut Mamba3Policy` exists anywhere to alias it.
        let borrowed: &'static Mamba3Policy<R, E> = unsafe { &*Rc::as_ptr(&policy) };
        let mut collector = Collector::new(borrowed, envs, steps, obs_dim, device)?
            .with_temperature(temperature)
            .with_seed(seed);
        if expert_labels {
            collector = collector.recording_expert_labels();
        }
        Ok(Self { collector, policy })
    }

    /// Another handle onto the weights being trained.
    pub fn policy(&self) -> Rc<Mamba3Policy<R, E>> {
        self.policy.clone()
    }

    /// The collector, for reading what the last window left behind.
    pub fn collector(&self) -> &Collector<'static, R, E> {
        &self.collector
    }

    /// The collector, for driving it.
    pub fn collector_mut(&mut self) -> &mut Collector<'static, R, E> {
        &mut self.collector
    }
}
