//! Inverted dropout.

use std::cell::Cell;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::Result;
use crate::nn::module::{Layer, Module, ModuleVisitor};
use crate::tensor::ops::random::dropout_mask;

/// Drops activations with probability `p` during training and rescales the
/// survivors by `1/(1-p)`, so the layer is the identity at evaluation time.
///
/// Masks are generated on device from a counter, so no host round trip and no
/// shared RNG lock is involved.
pub struct Dropout {
    p: f32,
    seed: u64,
    counter: Cell<u64>,
    training: Cell<bool>,
}

impl core::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Dropout(p={}, training={})", self.p, self.training.get())
    }
}

impl Dropout {
    /// A dropout layer with drop probability `p`.
    pub fn new(p: f32) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            seed: 0x9E3779B97F4A7C15,
            counter: Cell::new(0),
            training: Cell::new(true),
        }
    }

    /// Use an explicit base seed so a run is reproducible.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Drop probability.
    pub fn p(&self) -> f32 {
        self.p
    }

    /// Apply dropout to a value.
    pub fn apply<R: Runtime, E: FloatElem>(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        if !self.training.get() || self.p <= 0.0 {
            return Ok(input.clone());
        }
        let step = self.counter.get();
        self.counter.set(step.wrapping_add(1));
        let mask = dropout_mask::<R, E>(
            input.shape().clone(),
            self.p,
            self.seed ^ step.wrapping_mul(0x2545F4914F6CDD1D),
            input.device(),
        );
        input.mul(&Var::constant(mask))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Dropout {
    fn visit(&self, _visitor: &mut ModuleVisitor<'_, R, E>) {}

    fn on_mode_change(&self, training: bool) {
        self.training.set(training);
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for Dropout {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}
