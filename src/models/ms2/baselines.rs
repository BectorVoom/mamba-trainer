//! Baseline spectrum-encoder block stacks (P9.3 of `docs/MS2_SUBSTRUCTURE_TASKS.md`).
//!
//! Standalone alternatives to the bidirectional Mamba-3 block stack of
//! architecture §4.1 (`docs/MS2_V0_ARCHITECTURE.md`), for the release
//! comparison of contracts §10. Each stack takes exactly what the Mamba
//! encoder's block stack takes — the embedded, conditioned, masked peak
//! sequence `x [B, N, d]`, the peak validity `valid [B, N]`, and nothing else —
//! and returns `[B, N, d]`, so a later integration task can swap only the
//! block stack while the peak embedding, metadata conditioning, output norm,
//! spectrum memory and pooling stay shared. Nothing here touches `Ms2Model`,
//! the trainer or the drivers.
//!
//! Padding discipline (architecture §3): every cross-position quantity is
//! derived from [`masked_mean`] or from attention scores masked with
//! [`Var::mask_logits`], never by multiplication with a mask, and every block
//! output is passed through [`Var::ms2_select_valid`]. Padding therefore holds
//! exact `+0.0` and cannot influence a valid output, even with NaN or huge
//! values in padded slots.
//!
//! Parameter matching at the V0 config (`ModelConfig::v0()`: `d = 128`, two
//! bidirectional blocks of the contracted SISO `SsmConfig`):
//!
//! * reference bidirectional stack: 4 [`Mamba3Block`]s × 140,716 = 562,864;
//! * [`SetEncoderStackConfig::matched`]: `L = 6`, `hidden = 240` → 555,936
//!   (−1.23%);
//! * [`TransformerEncoderStackConfig::matched`]: `L = 3`, 4 heads,
//!   `hidden = 480` → 567,840 (+0.88%);
//! * [`UnidirectionalMambaStackConfig::matched`]: `L = 4` forward-only blocks
//!   → 562,864 (exact).
//!
//! Fairness note (review Part D): the parameter-matched forward-only stack has
//! twice the depth of the reference bidirectional stack (4 forward-only blocks
//! against 2 bidirectional pairs), so that ablation changes depth as well as
//! direction; results should identify the depth change alongside the removal
//! of bidirectionality.
//!
//! [`UnidirectionalMambaStackConfig::matched`]: UnidirectionalMambaStackConfig::matched

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Block, Mamba3BlockConfig};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::ssm::SsmConfig;
use crate::tensor::Shape;
use crate::tensor::Tensor;
use crate::tensor::ops::{elemwise, reduce};
use crate::tensor::ops::random::Rng;

use super::contract::ModelConfig;

/// Which block stack an encoder runs. Serde support is for the later
/// integration task, which will select the stack from a config file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncoderStackKind {
    /// The V0 bidirectional Mamba-3 stack (reference; lives in `encoder.rs`).
    Mamba,
    /// Forward-only Mamba-3 blocks: the bidirectionality ablation.
    UnidirectionalMamba,
    /// Permutation-invariant DeepSets-style blocks.
    Set,
    /// Pre-norm self-attention blocks, no positional encoding.
    Transformer,
}

/// Masked mean of `x [B, N, d]` over the valid peaks: the sum of
/// [`Var::ms2_select_valid`] `x` along the peak axis divided by
/// `max(valid_count, 1)`. Padding never contributes: it is selected to zero
/// before the sum (a multiplication would let a NaN through), and an
/// all-padding spectrum divides the zero sum by 1, giving an exact-zero mean
/// rather than a NaN.
pub fn masked_mean<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    valid: &Tensor<R, E>,
) -> Result<Var<R, E>> {
    let dims = x.dims().to_vec();
    if dims.len() != 3 || valid.shape().rank() != 2 {
        return Err(Error::shape(format!(
            "masked_mean needs x [B, N, d] and valid [B, N], got {} and {}",
            x.shape(),
            valid.shape()
        )));
    }
    let b = dims[0];
    let device = x.device().clone();
    let clean = x.ms2_select_valid(valid)?;
    let sum = clean.sum_dim(1)?;
    let len = reduce::sum_dim(valid, 1)?.reshape(vec![b, 1, 1])?;
    let denom = Var::constant(len).maximum(&Var::constant(Tensor::ones(vec![b, 1, 1], &device)))?;
    sum.div(&denom)?.squeeze(1)
}

/// Parameters of one [`Mamba3Block`] for `ssm`: the fused input and output
/// projections (with bias only when `ssm.bias`: `in_w` entries on the input
/// side and `d_model` entries on the output side, since the output projection
/// maps `d_inner → d_model`), the `dt_bias`/`a_log`/`d_skip`
/// vectors, the optional `B`/`C` biases and norms, the optional convolution
/// (`kernel × channels` weights plus one bias per channel, matching
/// `CausalConv1dConfig::new(channels, kernel)` with its default bias), and the
/// block pre-norm gain.
pub fn mamba3_block_params(ssm: &SsmConfig) -> usize {
    let in_w = ssm.in_proj_width();
    let inner = ssm.d_inner();
    let mut n = ssm.d_model * in_w + inner * ssm.d_model;
    if ssm.bias {
        n += in_w + ssm.d_model;
    }
    n += ssm.n_heads;
    n += ssm.n_heads;
    if ssm.skip_connection {
        n += ssm.n_heads;
    }
    if ssm.bc_bias {
        n += 2 * ssm.n_heads * ssm.d_state;
    }
    if ssm.bc_norm {
        n += ssm.d_state;
    }
    if ssm.post_gate_norm {
        n += inner;
    }
    if let Some(k) = ssm.conv_kernel {
        let channels = inner + 2 * ssm.bc_width();
        n += k * channels + channels;
    }
    n += ssm.d_model;
    n
}

/// Parameters of the reference bidirectional stack: `encoder_blocks` pairs of
/// forward/backward [`Mamba3Block`]s.
pub fn bidirectional_reference_params(model: &ModelConfig) -> usize {
    model.encoder_blocks as usize * 2 * mamba3_block_params(&model.encoder)
}

// -- Set encoder --------------------------------------------------------

/// Configuration of [`SetEncoderStack`]: `n_blocks` pre-norm-free blocks of
/// `x <- select_valid(x + MLP([x, masked mean of x])))` with a trailing RMS
/// norm, where the MLP is `Linear(2d → hidden)`, SiLU, `Linear(hidden → d)`
/// (both with bias). No positional or order information enters anywhere: the
/// only cross-peak quantity is the masked mean.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetEncoderStackConfig {
    /// Residual stream width.
    pub d_model: usize,
    /// Block count.
    pub n_blocks: usize,
    /// MLP hidden width.
    pub hidden_dim: usize,
}

impl SetEncoderStackConfig {
    /// Parameters of the whole stack.
    pub fn params(&self) -> usize {
        self.n_blocks * (2 * self.d_model + self.hidden_dim * (3 * self.d_model + 1))
    }

    /// A config with `6` blocks whose hidden width (rounded down to a multiple
    /// of 8) puts the count closest to `target_params`. At the V0 target
    /// 562,864 and `d_model = 128` this is `L = 6`, `hidden = 240` → 555,936
    /// (−1.23%).
    pub fn matched(target_params: usize, d_model: usize) -> Self {
        let n_blocks = 6;
        let per = target_params / n_blocks;
        let hidden = per.saturating_sub(2 * d_model) / (3 * d_model + 1);
        let hidden = (hidden / 8 * 8).max(8);
        Self {
            d_model,
            n_blocks,
            hidden_dim: hidden,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.n_blocks == 0 || self.hidden_dim == 0 {
            return Err(Error::config(format!(
                "SetEncoderStackConfig needs positive d_model, n_blocks and hidden_dim, got {} and {} and {}",
                self.d_model, self.n_blocks, self.hidden_dim
            )));
        }
        Ok(())
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<SetEncoderStack<R, E>> {
        self.validate()?;
        let mut blocks = Vec::with_capacity(self.n_blocks);
        for _ in 0..self.n_blocks {
            blocks.push(SetBlock {
                up: LinearConfig::new(2 * self.d_model, self.hidden_dim).init(device, rng),
                down: LinearConfig::new(self.hidden_dim, self.d_model).init(device, rng),
                norm: RmsNormConfig::new(self.d_model).init(device, rng),
            });
        }
        Ok(SetEncoderStack {
            blocks,
            d_model: self.d_model,
            hidden_dim: self.hidden_dim,
        })
    }
}

struct SetBlock<R: Runtime, E: FloatElem> {
    up: Linear<R, E>,
    down: Linear<R, E>,
    norm: RmsNorm<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for SetBlock<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("up", &self.up);
        visitor.child("down", &self.down);
        visitor.child("norm", &self.norm);
    }
}

/// Permutation-equivariant DeepSets-style stack over `[B, N, d]`: permuting the
/// valid peak embeddings permutes the output rows the same way. (Only a pooled
/// vector such as [`masked_mean`] is permutation-invariant.)
pub struct SetEncoderStack<R: Runtime, E: FloatElem> {
    blocks: Vec<SetBlock<R, E>>,
    d_model: usize,
    hidden_dim: usize,
}

impl<R: Runtime, E: FloatElem> SetEncoderStack<R, E> {
    /// Block count.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Residual width.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// MLP hidden width.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Run the stack over the embedded, conditioned, masked peak sequence `x`
    /// with validity `valid`. Every block cleans its input with
    /// `select_valid` first (so NaN/huge padding never reaches the mean or the
    /// MLP), concatenates the masked mean, and returns exact zeros in padding.
    pub fn apply(&self, x: &Var<R, E>, valid: &Tensor<R, E>) -> Result<Var<R, E>> {
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::shape(format!(
                "SetEncoderStack::apply needs x [B, N, d], got {}",
                x.shape()
            )));
        }
        let (b, n) = (dims[0], dims[1]);
        let mut x = x.clone();
        for block in &self.blocks {
            let clean = x.ms2_select_valid(valid)?;
            let mean = masked_mean(&clean, valid)?.unsqueeze(1)?.expand(vec![b, n, self.d_model])?;
            let joined = cat(&[clean.clone(), mean], 2)?;
            let mlp = block.down.apply(&block.up.apply(&joined)?.silu()?)?;
            x = block.norm.apply(&clean.add(&mlp)?)?.ms2_select_valid(valid)?;
        }
        Ok(x)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for SetEncoderStack<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (i, block) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, block);
        }
    }
}

// -- Transformer encoder --------------------------------------------------

/// Configuration of [`TransformerEncoderStack`]: `n_blocks` pre-norm blocks of
/// multi-head self-attention (`q, k, v, o`: `Linear(d → d)`, no bias,
/// `n_heads` heads, key mask applied with `mask_logits`, no positional
/// encoding since m/z enters through the Fourier features) followed by a plain
/// MLP (`Linear(d → hidden)`, SiLU, `Linear(hidden → d)`, both with bias).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformerEncoderStackConfig {
    /// Residual stream width.
    pub d_model: usize,
    /// Block count.
    pub n_blocks: usize,
    /// Attention heads (`d_model` must be a multiple of it).
    pub n_heads: usize,
    /// MLP hidden width.
    pub hidden_dim: usize,
}

impl TransformerEncoderStackConfig {
    /// Parameters of the whole stack.
    pub fn params(&self) -> usize {
        let d = self.d_model;
        self.n_blocks * (4 * d * d + 3 * d + self.hidden_dim * (2 * d + 1))
    }

    /// A config with `3` blocks (4 heads when `d_model` is a multiple of 4)
    /// whose hidden width (rounded to a multiple of 16) puts the count closest
    /// to `target_params`. At the V0 target 562,864 and `d_model = 128` this is
    /// `L = 3`, 4 heads, `hidden = 480` → 567,840 (+0.88%).
    pub fn matched(target_params: usize, d_model: usize) -> Self {
        let n_heads = if d_model.is_multiple_of(4) {
            4
        } else if d_model.is_multiple_of(2) {
            2
        } else {
            1
        };
        let n_blocks = 3;
        let per = target_params / n_blocks;
        let hidden = per.saturating_sub(4 * d_model * d_model + 3 * d_model) / (2 * d_model + 1);
        let hidden = ((hidden + 8) / 16 * 16).max(16);
        Self {
            d_model,
            n_blocks,
            n_heads,
            hidden_dim: hidden,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.n_blocks == 0 || self.n_heads == 0 || self.hidden_dim == 0 {
            return Err(Error::config(format!(
                "TransformerEncoderStackConfig needs positive d_model, n_blocks, n_heads and hidden_dim, got {} and {} and {} and {}",
                self.d_model, self.n_blocks, self.n_heads, self.hidden_dim
            )));
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(Error::config(format!(
                "TransformerEncoderStackConfig needs d_model {} to be a multiple of n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        Ok(())
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<TransformerEncoderStack<R, E>> {
        self.validate()?;
        let d = self.d_model;
        let mut blocks = Vec::with_capacity(self.n_blocks);
        for _ in 0..self.n_blocks {
            blocks.push(TransformerBlock {
                norm1: RmsNormConfig::new(d).init(device, rng),
                q: LinearConfig::new(d, d).with_bias(false).init(device, rng),
                k: LinearConfig::new(d, d).with_bias(false).init(device, rng),
                v: LinearConfig::new(d, d).with_bias(false).init(device, rng),
                o: LinearConfig::new(d, d).with_bias(false).init(device, rng),
                norm2: RmsNormConfig::new(d).init(device, rng),
                up: LinearConfig::new(d, self.hidden_dim).init(device, rng),
                down: LinearConfig::new(self.hidden_dim, d).init(device, rng),
            });
        }
        Ok(TransformerEncoderStack {
            blocks,
            d_model: d,
            n_heads: self.n_heads,
            hidden_dim: self.hidden_dim,
        })
    }
}

struct TransformerBlock<R: Runtime, E: FloatElem> {
    norm1: RmsNorm<R, E>,
    q: Linear<R, E>,
    k: Linear<R, E>,
    v: Linear<R, E>,
    o: Linear<R, E>,
    norm2: RmsNorm<R, E>,
    up: Linear<R, E>,
    down: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for TransformerBlock<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm1", &self.norm1);
        visitor.child("q", &self.q);
        visitor.child("k", &self.k);
        visitor.child("v", &self.v);
        visitor.child("o", &self.o);
        visitor.child("norm2", &self.norm2);
        visitor.child("up", &self.up);
        visitor.child("down", &self.down);
    }
}

/// Pre-norm self-attention stack over `[B, N, d]`, composed from the crate's
/// `Linear`, `mask_logits` and `softmax` primitives exactly like the decoder's
/// cross-attention (`decoder.rs`), with memory set to the sequence itself.
pub struct TransformerEncoderStack<R: Runtime, E: FloatElem> {
    blocks: Vec<TransformerBlock<R, E>>,
    d_model: usize,
    n_heads: usize,
    hidden_dim: usize,
}

impl<R: Runtime, E: FloatElem> TransformerEncoderStack<R, E> {
    /// Block count.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Residual width.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Attention heads.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// MLP hidden width.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Run the stack over the embedded, conditioned, masked peak sequence `x`
    /// with validity `valid`. Padding values are selected to zero before the
    /// projections (so a zero attention weight can never meet a NaN value),
    /// padding keys are masked with `mask_logits`, and every residual lands
    /// through `select_valid`, leaving exact zeros in padding.
    pub fn apply(&self, x: &Var<R, E>, valid: &Tensor<R, E>) -> Result<Var<R, E>> {
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::shape(format!(
                "TransformerEncoderStack::apply needs x [B, N, d], got {}",
                x.shape()
            )));
        }
        let (b, n, d) = (dims[0], dims[1], dims[2]);
        let h = self.n_heads;
        let hd = d / h;
        let mut x = x.clone();
        for block in &self.blocks {
            let clean = x.ms2_select_valid(valid)?;
            let hn = block.norm1.apply(&clean)?.ms2_select_valid(valid)?;
            let heads = |proj: &Linear<R, E>| -> Result<Var<R, E>> {
                proj.apply(&hn)?
                    .reshape(vec![b, n, h, hd])?
                    .permute(&[0, 2, 1, 3])
            };
            let qh = heads(&block.q)?;
            let kh = heads(&block.k)?.permute(&[0, 1, 3, 2])?;
            let vh = heads(&block.v)?;
            let scores = qh.matmul(&kh)?.mul_scalar(1.0 / (hd as f32).sqrt());
            let key = valid.reshape(Shape::new(vec![b, 1, 1, n]))?;
            let full = elemwise::expand(&key, &Shape::new(vec![b, h, n, n]))?;
            let ctx = scores
                .mask_logits(&full)?
                .softmax(3)?
                .matmul(&vh)?
                .permute(&[0, 2, 1, 3])?
                .reshape(vec![b, n, d])?;
            let attended = block.o.apply(&ctx)?;
            let y = clean.add(&attended)?.ms2_select_valid(valid)?;
            let fed = block
                .down
                .apply(&block.up.apply(&block.norm2.apply(&y)?.ms2_select_valid(valid)?)?.silu()?)?;
            x = y.add(&fed)?.ms2_select_valid(valid)?;
        }
        Ok(x)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for TransformerEncoderStack<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (i, block) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, block);
        }
    }
}

// -- Unidirectional Mamba stack -------------------------------------------

/// Configuration of [`UnidirectionalMambaStack`]: `n_blocks` forward-only
/// [`Mamba3Block`]s with the given SSM config — the same blocks as the Mamba
/// encoder stack, without the reversed pass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnidirectionalMambaStackConfig {
    /// The SSM config of every block.
    pub ssm: SsmConfig,
    /// Block count.
    pub n_blocks: usize,
}

impl UnidirectionalMambaStackConfig {
    /// Parameters of the whole stack.
    pub fn params(&self) -> usize {
        self.n_blocks * mamba3_block_params(&self.ssm)
    }

    /// A config over a V0-style contracted SSM at `d_model` whose block count
    /// (nearest integer) matches `target_params`. At the V0 target 562,864 and
    /// `d_model = 128` this is `L = 4` → 562,864 (exact).
    ///
    /// Fairness note: matching parameters with forward-only blocks doubles the
    /// depth relative to the bidirectional reference (which holds two blocks
    /// per pair), so this ablation changes depth as well as direction.
    pub fn matched(target_params: usize, d_model: usize) -> Self {
        let mut ssm = ModelConfig::v0().encoder;
        ssm.d_model = d_model;
        let per = mamba3_block_params(&ssm).max(1);
        let n_blocks = ((target_params + per / 2) / per).max(1);
        Self { ssm, n_blocks }
    }

    fn validate(&self) -> Result<()> {
        if self.n_blocks == 0 {
            return Err(Error::config(
                "UnidirectionalMambaStackConfig needs at least one block".to_string(),
            ));
        }
        self.ssm.validate()?;
        Ok(())
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<UnidirectionalMambaStack<R, E>> {
        self.validate()?;
        let mut blocks = Vec::with_capacity(self.n_blocks);
        for _ in 0..self.n_blocks {
            blocks.push(Mamba3BlockConfig::new(self.ssm.clone()).init(device, rng)?);
        }
        Ok(UnidirectionalMambaStack { blocks })
    }
}

/// Forward-only Mamba-3 blocks over `[B, N, d]`: the bidirectionality ablation.
/// Padding is selected to zero before each block (restoring the encoder's
/// invariant that every block input is exact zeros in padding, so the scan
/// only ever multiplies finite numbers) and again after it, since the causal
/// scan carries valid states forward into the padding slots.
pub struct UnidirectionalMambaStack<R: Runtime, E: FloatElem> {
    blocks: Vec<Mamba3Block<R, E>>,
}

impl<R: Runtime, E: FloatElem> UnidirectionalMambaStack<R, E> {
    /// Block count.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Run the stack over the embedded, conditioned, masked peak sequence `x`
    /// with validity `valid`.
    pub fn apply(&self, x: &Var<R, E>, valid: &Tensor<R, E>) -> Result<Var<R, E>> {
        if x.rank() != 3 {
            return Err(Error::shape(format!(
                "UnidirectionalMambaStack::apply needs x [B, N, d], got {}",
                x.shape()
            )));
        }
        let mut x = x.ms2_select_valid(valid)?;
        for block in &self.blocks {
            x = block.apply(&x)?.ms2_select_valid(valid)?;
        }
        Ok(x)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for UnidirectionalMambaStack<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (i, block) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, block);
        }
    }
}
