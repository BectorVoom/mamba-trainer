//! Weight initialisation schemes.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::tensor::Shape;
use crate::tensor::Tensor;
use crate::tensor::ops::random::{Rng, randn, uniform};

/// How to fill a freshly created parameter.
///
/// Fan-in and fan-out are taken from the first two dimensions of the shape
/// (`[fan_in, fan_out, ...]`), which matches the `[in, out]` weight layout used by
/// [`crate::nn::Linear`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Initializer {
    /// All zeros.
    Zeros,
    /// All ones.
    Ones,
    /// A fixed value.
    Constant(f32),
    /// `N(mean, std^2)`.
    Normal {
        /// Distribution mean.
        mean: f32,
        /// Distribution standard deviation.
        std: f32,
    },
    /// `U(lo, hi)`.
    Uniform {
        /// Lower bound.
        lo: f32,
        /// Upper bound.
        hi: f32,
    },
    /// `U(-b, b)` with `b = gain * sqrt(3 / fan_in)`.
    KaimingUniform {
        /// Multiplier applied to the bound.
        gain: f32,
    },
    /// `N(0, (gain / sqrt(fan_in))^2)`.
    KaimingNormal {
        /// Multiplier applied to the standard deviation.
        gain: f32,
    },
    /// `U(-b, b)` with `b = gain * sqrt(6 / (fan_in + fan_out))`.
    XavierUniform {
        /// Multiplier applied to the bound.
        gain: f32,
    },
    /// `N(0, (gain * sqrt(2 / (fan_in + fan_out)))^2)`.
    XavierNormal {
        /// Multiplier applied to the standard deviation.
        gain: f32,
    },
    /// `N(0, std^2)` scaled by `1/sqrt(2 * n_layers)`, the standard residual-branch
    /// rescaling used by GPT-2 and by Mamba.
    ResidualScaled {
        /// Base standard deviation before rescaling.
        std: f32,
        /// Number of residual blocks in the stack.
        n_layers: usize,
    },
}

impl Default for Initializer {
    fn default() -> Self {
        Initializer::KaimingUniform { gain: 1.0 }
    }
}

impl Initializer {
    /// Materialise a tensor of the given shape.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        shape: impl Into<Shape>,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Tensor<R, E> {
        let shape = shape.into();
        let fan_in = shape.dims().first().copied().unwrap_or(1) as f32;
        let fan_out = shape.dims().get(1).copied().unwrap_or(1) as f32;

        match *self {
            Initializer::Zeros => Tensor::zeros(shape, device),
            Initializer::Ones => Tensor::ones(shape, device),
            Initializer::Constant(v) => Tensor::full(shape, v, device),
            Initializer::Normal { mean, std } => randn(shape, mean, std, device, rng),
            Initializer::Uniform { lo, hi } => uniform(shape, lo, hi, device, rng),
            Initializer::KaimingUniform { gain } => {
                let bound = gain * (3.0 / fan_in).sqrt();
                uniform(shape, -bound, bound, device, rng)
            }
            Initializer::KaimingNormal { gain } => {
                randn(shape, 0.0, gain / fan_in.sqrt(), device, rng)
            }
            Initializer::XavierUniform { gain } => {
                let bound = gain * (6.0 / (fan_in + fan_out)).sqrt();
                uniform(shape, -bound, bound, device, rng)
            }
            Initializer::XavierNormal { gain } => {
                randn(shape, 0.0, gain * (2.0 / (fan_in + fan_out)).sqrt(), device, rng)
            }
            Initializer::ResidualScaled { std, n_layers } => {
                let scale = 1.0 / (2.0 * n_layers.max(1) as f32).sqrt();
                randn(shape, 0.0, std * scale, device, rng)
            }
        }
    }
}
