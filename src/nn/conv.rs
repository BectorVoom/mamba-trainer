//! Convolutions.
//!
//! Only two shapes are needed by the models in this crate, and both are expressed
//! with ordinary differentiable ops rather than dedicated kernels:
//!
//! * a **depthwise causal 1-D convolution** with a very short kernel (3 or 4 taps),
//!   which is a sum of shifted, scaled copies of the input;
//! * a **non-overlapping patch embedding**, which is a reshape, a permute and a
//!   matmul.

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Layer, Module, ModuleVisitor};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

/// Shift `x` towards larger indices along `axis` by `n`, filling with zeros.
pub fn shift_by<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    axis: usize,
    n: usize,
) -> Result<Var<R, E>> {
    if n == 0 {
        return Ok(x.clone());
    }
    let len = x.shape().dim(axis);
    if n >= len {
        return Ok(Var::constant(Tensor::zeros(
            x.shape().clone(),
            x.device(),
        )));
    }
    let head = x.slice(axis, 0, len - n)?;
    let pad = Var::constant(Tensor::zeros(x.shape().with_dim(axis, n), x.device()));
    cat(&[pad, head], axis)
}

/// Configuration for [`CausalConv1d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalConv1dConfig {
    channels: usize,
    kernel_size: usize,
    bias: bool,
    init: Initializer,
}

impl CausalConv1dConfig {
    /// A depthwise causal convolution over `channels` with `kernel_size` taps.
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        Self {
            channels,
            kernel_size,
            bias: true,
            init: Initializer::KaimingUniform { gain: 1.0 },
        }
    }

    /// Enable or disable the bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Choose the weight initialiser.
    pub fn with_initializer(mut self, init: Initializer) -> Self {
        self.init = init;
        self
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> CausalConv1d<R, E> {
        // Fan-in for a depthwise kernel is the kernel width, so initialise from a
        // `[kernel, channels]` view and transpose the interpretation.
        let weight = Param::new(self.init.init(
            vec![self.kernel_size, self.channels],
            device,
            rng,
        ));
        CausalConv1d {
            weight,
            bias: self
                .bias
                .then(|| Param::new(Tensor::zeros(vec![self.channels], device))),
            channels: self.channels,
            kernel_size: self.kernel_size,
        }
    }
}

/// Depthwise causal convolution over `[batch, seq, channels]`.
///
/// The weight is stored as `[kernel_size, channels]`: tap `j` multiplies the input
/// delayed by `kernel_size - 1 - j` steps.
pub struct CausalConv1d<R: Runtime, E: FloatElem> {
    weight: Param<R, E>,
    bias: Option<Param<R, E>>,
    channels: usize,
    kernel_size: usize,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for CausalConv1d<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "CausalConv1d(channels={}, kernel={})",
            self.channels, self.kernel_size
        )
    }
}

impl<R: Runtime, E: FloatElem> CausalConv1d<R, E> {
    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of taps.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// The `[kernel_size, channels]` weight.
    pub fn weight(&self) -> &Param<R, E> {
        &self.weight
    }

    /// Convolve `[batch, seq, channels]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        input.shape().expect_rank(3)?;
        if input.shape().dim(2) != self.channels {
            return Err(Error::shape(format!(
                "CausalConv1d expects {} channels, got {}",
                self.channels,
                input.shape()
            )));
        }
        let weight = self.weight.var(input);
        let mut acc: Option<Var<R, E>> = None;
        for delay in 0..self.kernel_size {
            let tap = weight
                .slice(0, self.kernel_size - 1 - delay, 1)?
                .reshape(vec![1, 1, self.channels])?;
            let term = shift_by(input, 1, delay)?.mul(&tap)?;
            acc = Some(match acc {
                Some(a) => a.add(&term)?,
                None => term,
            });
        }
        let mut out = acc.expect("kernel_size >= 1");
        if let Some(bias) = &self.bias {
            out = out.add(&bias.var(input).reshape(vec![1, 1, self.channels])?)?;
        }
        Ok(out)
    }

    /// Convolve a window that continues an earlier one.
    ///
    /// `history` holds the last `kernel_size - 1` inputs as `[batch, k-1, channels]`
    /// and `input` is the new window as `[batch, seq, channels]`. Returns the output
    /// for the new positions and the history to carry forward. Works for a single
    /// decoding step and for a whole prefill window alike.
    pub fn apply_with_history(
        &self,
        input: &Var<R, E>,
        history: &Var<R, E>,
    ) -> Result<(Var<R, E>, Var<R, E>)> {
        let carry = self.kernel_size - 1;
        let seq = input.shape().dim(1);
        let window = cat(&[history.clone(), input.clone()], 1)?;
        let out = self.apply(&window)?.slice(1, carry, seq)?;
        let total = window.shape().dim(1);
        let new_history = window.slice(1, total - carry, carry)?;
        Ok((out, new_history))
    }

    /// One decoding step; `input` must hold a single position.
    pub fn step(
        &self,
        input: &Var<R, E>,
        history: &Var<R, E>,
    ) -> Result<(Var<R, E>, Var<R, E>)> {
        self.apply_with_history(input, history)
    }

    /// An all-zero convolution history for a batch.
    pub fn empty_history(&self, batch: usize, device: &Device<R>) -> Var<R, E> {
        Var::constant(Tensor::zeros(
            vec![batch, self.kernel_size - 1, self.channels],
            device,
        ))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for CausalConv1d<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("weight", &self.weight);
        if let Some(b) = &self.bias {
            visitor.param("bias", b);
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for CausalConv1d<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}

/// Configuration for [`PatchEmbed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchEmbedConfig {
    image_size: usize,
    patch_size: usize,
    in_channels: usize,
    dim: usize,
    bias: bool,
}

impl PatchEmbedConfig {
    /// Split a square image into non-overlapping patches and project each one.
    pub fn new(image_size: usize, patch_size: usize, in_channels: usize, dim: usize) -> Self {
        Self {
            image_size,
            patch_size,
            in_channels,
            dim,
            bias: true,
        }
    }

    /// Enable or disable the projection bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Number of patches per image.
    pub fn num_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<PatchEmbed<R, E>> {
        if self.image_size % self.patch_size != 0 {
            return Err(Error::config(format!(
                "image size {} is not divisible by patch size {}",
                self.image_size, self.patch_size
            )));
        }
        let patch_dim = self.in_channels * self.patch_size * self.patch_size;
        Ok(PatchEmbed {
            proj: LinearConfig::new(patch_dim, self.dim)
                .with_bias(self.bias)
                .init(device, rng),
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
        })
    }
}

/// Non-overlapping patch embedding for `[batch, channels, height, width]` images.
pub struct PatchEmbed<R: Runtime, E: FloatElem> {
    proj: Linear<R, E>,
    image_size: usize,
    patch_size: usize,
    in_channels: usize,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for PatchEmbed<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "PatchEmbed({}x{} image, {}x{} patches)",
            self.image_size, self.image_size, self.patch_size, self.patch_size
        )
    }
}

impl<R: Runtime, E: FloatElem> PatchEmbed<R, E> {
    /// Number of patches per image.
    pub fn num_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Embedding width.
    pub fn dim(&self) -> usize {
        self.proj.out_features()
    }

    /// The patch projection.
    pub fn projection(&self) -> &Linear<R, E> {
        &self.proj
    }

    /// Turn `[batch, channels, height, width]` into `[batch, num_patches, dim]`.
    pub fn apply(&self, images: &Var<R, E>) -> Result<Var<R, E>> {
        images.shape().expect_rank(4)?;
        let dims = images.dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if c != self.in_channels || h != self.image_size || w != self.image_size {
            return Err(Error::shape(format!(
                "PatchEmbed expects [_, {}, {}, {}], got {}",
                self.in_channels,
                self.image_size,
                self.image_size,
                images.shape()
            )));
        }
        let p = self.patch_size;
        let gh = h / p;
        let gw = w / p;
        // [b, c, gh, p, gw, p] -> [b, gh, gw, c, p, p] -> [b, gh*gw, c*p*p]
        let grid = images.reshape(vec![b, c, gh, p, gw, p])?;
        let ordered = grid.permute(&[0, 2, 4, 1, 3, 5])?;
        let flat = ordered.reshape(vec![b, gh * gw, c * p * p])?;
        self.proj.apply(&flat)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for PatchEmbed<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("proj", &self.proj);
    }
}
