//! Temporal convolutional encoder (baseline B): residual blocks of dilated
//! depthwise causal convolutions with pointwise channel mixing.
//!
//! ```text
//! h = x + mix(gelu(dilated_causal_conv(rmsnorm(x))))     dilation 1, 2, 4, ...
//! ```

use mamba3::autograd::Var;
use mamba3::backend::{Device, FloatElem};
use mamba3::cubecl::prelude::Runtime;
use mamba3::error::Result;
use mamba3::nn::{
    Initializer, Linear, LinearConfig, Module, ModuleVisitor, Param, RmsNorm, RmsNormConfig,
    shift_by,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;
use serde::{Deserialize, Serialize};

use crate::mamba::{Pooling, pool};

/// TCN configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TcnConfig {
    /// Input features per step.
    pub input_dim: usize,
    /// Channels.
    pub channels: usize,
    /// Taps per convolution.
    pub kernel: usize,
    /// Blocks; block `i` has dilation `2^i`.
    pub num_layers: usize,
    /// Pooling.
    pub pooling: Pooling,
}

impl Default for TcnConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            channels: 128,
            kernel: 4,
            num_layers: 5,
            pooling: Pooling::Last,
        }
    }
}

struct TcnBlock<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    taps: Param<R, E>,
    bias: Param<R, E>,
    mix: Linear<R, E>,
    dilation: usize,
    kernel: usize,
    channels: usize,
}

impl<R: Runtime, E: FloatElem> TcnBlock<R, E> {
    fn apply(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        let h = self.norm.apply(x)?;
        let taps = self.taps.var(x);
        let mut acc: Option<Var<R, E>> = None;
        for j in 0..self.kernel {
            let w = taps.slice(0, j, 1)?.reshape(vec![1, 1, self.channels])?;
            let term = shift_by(&h, 1, j * self.dilation)?.mul(&w)?;
            acc = Some(match acc {
                Some(a) => a.add(&term)?,
                None => term,
            });
        }
        let conv = acc.expect("kernel >= 1").add(&self.bias.var(x))?;
        x.add(&self.mix.apply(&conv.gelu()?)?)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for TcnBlock<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        v.child("norm", &self.norm);
        v.param("taps", &self.taps);
        v.param("bias", &self.bias);
        v.child("mix", &self.mix);
    }
}

/// Input projection, TCN blocks, final norm, pooling.
pub struct TcnEncoder<R: Runtime, E: FloatElem> {
    input: Linear<R, E>,
    blocks: Vec<TcnBlock<R, E>>,
    norm: RmsNorm<R, E>,
    cfg: TcnConfig,
}

impl<R: Runtime, E: FloatElem> TcnEncoder<R, E> {
    /// A freshly initialised encoder.
    pub fn new(cfg: &TcnConfig, device: &Device<R>, rng: &mut Rng) -> Self {
        let c = cfg.channels;
        let blocks = (0..cfg.num_layers)
            .map(|i| {
                let bound = (3.0 / cfg.kernel as f32).sqrt();
                let taps = Initializer::Uniform {
                    lo: -bound,
                    hi: bound,
                }
                .init(vec![cfg.kernel, c], device, rng);
                TcnBlock {
                    norm: RmsNormConfig::new(c).init(device, rng),
                    taps: Param::new(taps),
                    bias: Param::new(Tensor::zeros(vec![c], device)),
                    mix: LinearConfig::new(c, c).init(device, rng),
                    dilation: 1 << i,
                    kernel: cfg.kernel,
                    channels: c,
                }
            })
            .collect();
        Self {
            input: LinearConfig::new(cfg.input_dim, c).init(device, rng),
            blocks,
            norm: RmsNormConfig::new(c).init(device, rng),
            cfg: cfg.clone(),
        }
    }

    /// Output width.
    pub fn out_dim(&self) -> usize {
        self.cfg.channels
    }

    /// The pooled state, `[B, T, F] -> [B, channels]`.
    pub fn apply(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        let mut h = self.input.apply(x)?;
        for b in &self.blocks {
            h = b.apply(&h)?;
        }
        pool(&self.norm.apply(&h)?, self.cfg.pooling)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for TcnEncoder<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        v.child("input", &self.input);
        for (i, b) in self.blocks.iter().enumerate() {
            v.child_at("blocks", i, b);
        }
        v.child("norm", &self.norm);
    }
}
