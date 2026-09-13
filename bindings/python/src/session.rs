//! A collector kept alive beside the policy it borrows.
//!
//! [`mamba3::rl::Collector`] borrows the policy for as long as it collects, which
//! is exactly right in Rust and impossible to express through a Python object:
//! `learner` and `policy` are two independent handles with independent lifetimes,
//! and Python will drop them in whichever order it likes. [`Session`] is the one
//! place in these bindings that resolves that, and it resolves it once so that
//! nothing above has to think about it.

use std::mem::ManuallyDrop;
use std::rc::Rc;

use mamba3::backend::Device;
use mamba3::error::Result;
use mamba3::rl::{Collector, Mamba3Policy};

use crate::{E, R};

/// The persistent half of a learning loop: the recurrent state of every
/// environment and the `[envs, steps]` trajectory buffer, allocated once.
///
/// Both fields are `ManuallyDrop` and dropped explicitly, in order, by
/// [`Session`]'s own `Drop` impl: `collector` borrows `policy` and must go
/// first. That makes the ordering a property of the code rather than of the
/// fields' declaration order, so it survives a future reordering or an added
/// field.
pub struct Session {
    collector: ManuallyDrop<Collector<'static, R, E>>,
    policy: ManuallyDrop<Rc<Mamba3Policy<R, E>>>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: `collector` is dropped first, before the `policy` it borrows;
        // neither field is used again afterwards.
        unsafe {
            ManuallyDrop::drop(&mut self.collector);
            ManuallyDrop::drop(&mut self.policy);
        }
    }
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
        Ok(Self {
            collector: ManuallyDrop::new(collector),
            policy: ManuallyDrop::new(policy),
        })
    }

    /// Another handle onto the weights being trained.
    pub fn policy(&self) -> Rc<Mamba3Policy<R, E>> {
        Rc::clone(&self.policy)
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
