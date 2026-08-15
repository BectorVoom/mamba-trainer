//! Quantization-aware training.
//!
//! The crate implements **fake quantization**: values are rounded onto a uniform
//! grid in the forward pass and the gradient flows through a straight-through
//! estimator ([`Var::round_ste`](crate::autograd::Var::round_ste)) with a clamp mask
//! outside the representable range. That is the standard QAT recipe, and because
//! it is expressed with ordinary differentiable ops it needs no custom kernel and
//! no custom backward.
//!
//! Two range policies are supported, matching how the two kinds of tensor behave:
//!
//! * **Dynamic** — the range is recomputed from the tensor on every call. Correct
//!   for weights, which change every step.
//! * **Observed** — an exponential moving average of the per-batch range, frozen at
//!   evaluation time. Correct for activations.
//!
//! ```no_run
//! # use mamba3::nn::quant::{QuantConfig, QuantScheme, Granularity};
//! let weights = QuantConfig::builder()
//!     .bits(4)
//!     .scheme(QuantScheme::Symmetric)
//!     .granularity(Granularity::PerChannel { axis: 1 })
//!     .dynamic(true)
//!     .build()
//!     .unwrap();
//! ```

use std::cell::{Cell, RefCell};

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::nn::module::{Module, ModuleVisitor};
use crate::tensor::Tensor;
use crate::tensor::ops::{elemwise, reduce};

/// Whether the grid is centred on zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuantScheme {
    /// Zero point fixed at zero; the grid is `[-2^(b-1)+1, 2^(b-1)-1] * scale`.
    Symmetric,
    /// Learned zero point; the grid covers the observed `[min, max]` exactly.
    Affine,
}

/// How many independent scales a tensor gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Granularity {
    /// One scale for the whole tensor.
    PerTensor,
    /// One scale per slice along `axis` (for a `[in, out]` weight, `axis = 1` gives
    /// per-output-channel scales).
    PerChannel {
        /// Axis that keeps its own scale.
        axis: usize,
    },
}

/// Quantizer settings.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QuantConfig {
    /// Bit width, 2..=16.
    pub bits: u32,
    /// Symmetric or affine grid.
    pub scheme: QuantScheme,
    /// Per-tensor or per-channel scales.
    pub granularity: Granularity,
    /// Recompute the range every call (weights) instead of observing it (activations).
    pub dynamic: bool,
    /// EMA momentum for the observed range. Ignored when `dynamic`.
    pub momentum: f32,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            scheme: QuantScheme::Symmetric,
            granularity: Granularity::PerTensor,
            dynamic: true,
            momentum: 0.01,
        }
    }
}

impl QuantConfig {
    /// Start a builder.
    pub fn builder() -> QuantConfigBuilder {
        QuantConfigBuilder::default()
    }

    /// A sensible 8-bit weight configuration: symmetric, per-output-channel, dynamic.
    pub fn int8_weights() -> Self {
        Self {
            bits: 8,
            scheme: QuantScheme::Symmetric,
            granularity: Granularity::PerChannel { axis: 1 },
            dynamic: true,
            momentum: 0.01,
        }
    }

    /// A sensible 8-bit activation configuration: affine, per-tensor, observed.
    pub fn int8_activations() -> Self {
        Self {
            bits: 8,
            scheme: QuantScheme::Affine,
            granularity: Granularity::PerTensor,
            dynamic: false,
            momentum: 0.01,
        }
    }

    /// Integer bounds of the grid.
    fn bounds(&self) -> (f32, f32) {
        let levels = 1u32 << self.bits;
        match self.scheme {
            // Drop the most-negative code so the grid is exactly symmetric.
            QuantScheme::Symmetric => {
                let q = (levels / 2) as f32 - 1.0;
                (-q, q)
            }
            QuantScheme::Affine => (0.0, (levels - 1) as f32),
        }
    }
}

/// Builder for [`QuantConfig`].
#[derive(Debug, Clone, Default)]
pub struct QuantConfigBuilder {
    bits: Option<u32>,
    scheme: Option<QuantScheme>,
    granularity: Option<Granularity>,
    dynamic: Option<bool>,
    momentum: Option<f32>,
}

impl QuantConfigBuilder {
    /// Bit width.
    pub fn bits(mut self, bits: u32) -> Self {
        self.bits = Some(bits);
        self
    }

    /// Grid centring.
    pub fn scheme(mut self, scheme: QuantScheme) -> Self {
        self.scheme = Some(scheme);
        self
    }

    /// Scale sharing.
    pub fn granularity(mut self, granularity: Granularity) -> Self {
        self.granularity = Some(granularity);
        self
    }

    /// Recompute the range each call.
    pub fn dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// EMA momentum for observed ranges.
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    /// Validate and build.
    pub fn build(self) -> Result<QuantConfig> {
        let defaults = QuantConfig::default();
        let bits = self.bits.unwrap_or(defaults.bits);
        if !(2..=16).contains(&bits) {
            return Err(Error::config(format!(
                "quantization bit width must be in 2..=16, got {bits}"
            )));
        }
        let momentum = self.momentum.unwrap_or(defaults.momentum);
        if !(0.0..=1.0).contains(&momentum) {
            return Err(Error::config(format!(
                "observer momentum must be in [0, 1], got {momentum}"
            )));
        }
        let dynamic = self.dynamic.unwrap_or(defaults.dynamic);
        let granularity = self.granularity.unwrap_or(defaults.granularity);
        if !dynamic && matches!(granularity, Granularity::PerChannel { .. }) {
            return Err(Error::config(
                "per-channel observed ranges are not supported; use dynamic ranges \
                 for weights and per-tensor observation for activations"
                    .to_string(),
            ));
        }
        Ok(QuantConfig {
            bits,
            scheme: self.scheme.unwrap_or(defaults.scheme),
            granularity,
            dynamic,
            momentum,
        })
    }
}

/// The observed range of an activation tensor.
#[derive(Debug, Clone, Copy, Default)]
struct Observed {
    min: f32,
    max: f32,
    seen: bool,
}

/// A fake quantizer.
///
/// Insert one before a matmul to simulate low-precision inference while training
/// in floating point.
pub struct Quantizer {
    config: QuantConfig,
    observed: RefCell<Observed>,
    enabled: Cell<bool>,
    training: Cell<bool>,
}

impl core::fmt::Debug for Quantizer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Quantizer({:?}, enabled={})", self.config, self.enabled.get())
    }
}

impl Quantizer {
    /// Build a quantizer from a configuration.
    pub fn new(config: QuantConfig) -> Self {
        Self {
            config,
            observed: RefCell::new(Observed::default()),
            enabled: Cell::new(true),
            training: Cell::new(true),
        }
    }

    /// The configuration in force.
    pub fn config(&self) -> &QuantConfig {
        &self.config
    }

    /// Turn quantization off without removing it from the graph, so a model can be
    /// warmed up in full precision before QAT begins.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
    }

    /// Whether quantization is currently applied.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// The currently observed activation range, if any has been recorded.
    pub fn observed_range(&self) -> Option<(f32, f32)> {
        let o = *self.observed.borrow();
        o.seen.then_some((o.min, o.max))
    }

    /// Quantize and immediately dequantize, keeping the value differentiable.
    pub fn quantize<R: Runtime, E: FloatElem>(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        if !self.enabled.get() {
            return Ok(x.clone());
        }
        let (qmin, qmax) = self.config.bounds();

        match self.config.granularity {
            Granularity::PerTensor => {
                let (min, max) = if self.config.dynamic {
                    tensor_range(x.tensor())?
                } else {
                    self.observe(x.tensor())?
                };
                let (scale, zero_point) = solve_scale(min, max, qmin, qmax, self.config.scheme);
                let scaled = x.mul_scalar(1.0 / scale).add_scalar(zero_point);
                let q = scaled.round_ste().clamp(qmin, qmax);
                Ok(q.add_scalar(-zero_point).mul_scalar(scale))
            }
            Granularity::PerChannel { axis } => {
                let (scale, zero_point) =
                    channel_scales(x.tensor(), axis, qmin, qmax, self.config.scheme)?;
                let scale = Var::constant(scale);
                let zero_point = Var::constant(zero_point);
                let scaled = x.div(&scale)?.add(&zero_point)?;
                let q = scaled.round_ste().clamp(qmin, qmax);
                q.sub(&zero_point)?.mul(&scale)
            }
        }
    }

    /// Update and read the exponential moving average of the range.
    fn observe<R: Runtime, E: FloatElem>(&self, x: &Tensor<R, E>) -> Result<(f32, f32)> {
        let mut state = self.observed.borrow_mut();
        if self.training.get() || !state.seen {
            let (min, max) = tensor_range(x)?;
            if !state.seen {
                state.min = min;
                state.max = max;
                state.seen = true;
            } else {
                let m = self.config.momentum;
                state.min = (1.0 - m) * state.min + m * min;
                state.max = (1.0 - m) * state.max + m * max;
            }
        }
        Ok((state.min, state.max))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Quantizer {
    fn visit(&self, _visitor: &mut ModuleVisitor<'_, R, E>) {}

    fn on_mode_change(&self, training: bool) {
        self.training.set(training);
    }
}

/// The min and max of a whole tensor, read back to the host.
fn tensor_range<R: Runtime, E: FloatElem>(x: &Tensor<R, E>) -> Result<(f32, f32)> {
    let flat = x.flatten();
    let mins = reduce::min_dim(&flat, 0)?.to_f32();
    let maxes = reduce::max_dim(&flat, 0)?.to_f32();
    Ok((mins[0], maxes[0]))
}

/// Scale and zero point for one shared range.
fn solve_scale(min: f32, max: f32, qmin: f32, qmax: f32, scheme: QuantScheme) -> (f32, f32) {
    const EPS: f32 = 1e-12;
    match scheme {
        QuantScheme::Symmetric => {
            let bound = min.abs().max(max.abs()).max(EPS);
            (bound / qmax, 0.0)
        }
        QuantScheme::Affine => {
            // Always include zero in the range so that padding stays exact.
            let lo = min.min(0.0);
            let hi = max.max(0.0);
            let scale = ((hi - lo) / (qmax - qmin)).max(EPS);
            let zero_point = (qmin - lo / scale).round();
            (scale, zero_point)
        }
    }
}

/// Per-channel scale and zero-point tensors, broadcastable against `x`.
fn channel_scales<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    axis: usize,
    qmin: f32,
    qmax: f32,
    scheme: QuantScheme,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    if axis >= x.rank() {
        return Err(Error::config(format!(
            "per-channel axis {axis} is out of range for {}",
            x.shape()
        )));
    }
    let mut mins = x.clone();
    let mut maxes = x.clone();
    for a in 0..x.rank() {
        if a != axis {
            mins = reduce::min_dim(&mins, a)?;
            maxes = reduce::max_dim(&maxes, a)?;
        }
    }
    // Both now have shape [1, .., C, .., 1] which broadcasts against `x`.
    let host_min = mins.to_f32();
    let host_max = maxes.to_f32();
    let mut scales = Vec::with_capacity(host_min.len());
    let mut zeros = Vec::with_capacity(host_min.len());
    for (lo, hi) in host_min.iter().zip(host_max.iter()) {
        let (s, z) = solve_scale(*lo, *hi, qmin, qmax, scheme);
        scales.push(s);
        zeros.push(z);
    }
    let shape = mins.shape().clone();
    Ok((
        Tensor::from_f32(&scales, shape.clone(), x.device())?,
        Tensor::from_f32(&zeros, shape, x.device())?,
    ))
}

/// Round a tensor's values onto the grid and report the mean absolute error.
///
/// Useful in tests and when sweeping bit widths.
pub fn quantization_error<R: Runtime, E: FloatElem>(
    original: &Tensor<R, E>,
    quantized: &Tensor<R, E>,
) -> Result<f32> {
    let diff = elemwise::abs(&elemwise::sub(original, quantized)?);
    Ok(reduce::mean_all(&diff)?.to_f32()[0])
}
