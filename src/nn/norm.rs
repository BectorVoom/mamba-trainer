//! Normalisation layers.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::init::Initializer;
use crate::nn::module::{Layer, Module, ModuleVisitor};
use crate::nn::param::Param;
use crate::tensor::ops::random::Rng;

/// Configuration for [`RmsNorm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormConfig {
    dim: usize,
    eps: f32,
    weight: bool,
    init: Initializer,
}

impl RmsNormConfig {
    /// Normalise over a trailing axis of width `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            eps: 1e-5,
            weight: true,
            init: Initializer::Ones,
        }
    }

    /// Numerical floor added to the mean square.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Include a learnable gain. Turning it off gives a pure normaliser, which is
    /// what Mamba-3's B/C normalisation uses when the gain is folded elsewhere.
    pub fn with_weight(mut self, weight: bool) -> Self {
        self.weight = weight;
        self
    }

    /// Gain initialiser.
    pub fn with_initializer(mut self, init: Initializer) -> Self {
        self.init = init;
        self
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> RmsNorm<R, E> {
        RmsNorm {
            weight: self
                .weight
                .then(|| Param::new(self.init.init(vec![self.dim], device, rng))),
            dim: self.dim,
            eps: self.eps,
        }
    }
}

/// Root-mean-square normalisation over the last axis.
///
/// `y = x / sqrt(mean(x^2) + eps) * g`
pub struct RmsNorm<R: Runtime, E: FloatElem> {
    weight: Option<Param<R, E>>,
    dim: usize,
    eps: f32,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for RmsNorm<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "RmsNorm({}, eps={})", self.dim, self.eps)
    }
}

impl<R: Runtime, E: FloatElem> RmsNorm<R, E> {
    /// The learnable gain, if present.
    pub fn weight(&self) -> Option<&Param<R, E>> {
        self.weight.as_ref()
    }

    /// Width of the normalised axis.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Normalise `[..., dim]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        if input.shape().dim_from_end(0) != self.dim {
            return Err(Error::shape(format!(
                "RmsNorm expects a trailing dimension of {}, got {}",
                self.dim,
                input.shape()
            )));
        }
        let last = input.rank() - 1;
        let mean_square = input.mul(input)?.mean_dim(last)?;
        let scale = mean_square.add_scalar(self.eps).rsqrt();
        let normed = input.mul(&scale)?;
        match &self.weight {
            Some(w) => normed.mul(&w.var(input)),
            None => Ok(normed),
        }
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for RmsNorm<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        if let Some(w) = &self.weight {
            visitor.param("weight", w);
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for RmsNorm<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}

/// Configuration for [`LayerNorm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    dim: usize,
    eps: f32,
    affine: bool,
}

impl LayerNormConfig {
    /// Normalise over a trailing axis of width `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            eps: 1e-5,
            affine: true,
        }
    }

    /// Numerical floor added to the variance.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Include learnable gain and shift.
    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> LayerNorm<R, E> {
        let (weight, bias) = if self.affine {
            (
                Some(Param::new(Initializer::Ones.init(vec![self.dim], device, rng))),
                Some(Param::new(Initializer::Zeros.init(vec![self.dim], device, rng))),
            )
        } else {
            (None, None)
        };
        LayerNorm {
            weight,
            bias,
            dim: self.dim,
            eps: self.eps,
        }
    }
}

/// Mean/variance normalisation over the last axis.
pub struct LayerNorm<R: Runtime, E: FloatElem> {
    weight: Option<Param<R, E>>,
    bias: Option<Param<R, E>>,
    dim: usize,
    eps: f32,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for LayerNorm<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "LayerNorm({}, eps={})", self.dim, self.eps)
    }
}

impl<R: Runtime, E: FloatElem> LayerNorm<R, E> {
    /// Normalise `[..., dim]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        let last = input.rank() - 1;
        let mean = input.mean_dim(last)?;
        let centred = input.sub(&mean)?;
        let var = centred.mul(&centred)?.mean_dim(last)?;
        let mut out = centred.mul(&var.add_scalar(self.eps).rsqrt())?;
        if let Some(w) = &self.weight {
            out = out.mul(&w.var(input))?;
        }
        if let Some(b) = &self.bias {
            out = out.add(&b.var(input))?;
        }
        Ok(out)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for LayerNorm<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        if let Some(w) = &self.weight {
            visitor.param("weight", w);
        }
        if let Some(b) = &self.bias {
            visitor.param("bias", b);
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for LayerNorm<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}
