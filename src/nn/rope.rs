//! Rotary position embedding, and the pairwise rotation it is built from.
//!
//! The same rotation primitive serves two purposes in this crate: position
//! encoding in attention layers, and the **complex (rotational) state update** of
//! Mamba-3, where the angles are data dependent rather than fixed. Sharing the
//! primitive keeps both paths differentiable through the ordinary autodiff rules.

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Rotate the halves of the last axis of `x` by the angles in `cos`/`sin`.
///
/// `x` has shape `[..., d]` with `d` even; `cos` and `sin` broadcast against
/// `[..., d/2]`. Splitting into halves (rather than adjacent pairs) matches the
/// usual RoPE implementation and lets the rotation be two slices and one concat.
pub fn rotate_halves<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    cos: &Var<R, E>,
    sin: &Var<R, E>,
) -> Result<Var<R, E>> {
    let last = x.rank() - 1;
    let d = x.shape().dim(last);
    if d % 2 != 0 {
        return Err(Error::shape(format!(
            "rotation needs an even trailing dimension, got {d}"
        )));
    }
    let half = d / 2;
    let x1 = x.slice(last, 0, half)?;
    let x2 = x.slice(last, half, half)?;
    let out1 = x1.mul(cos)?.sub(&x2.mul(sin)?)?;
    let out2 = x1.mul(sin)?.add(&x2.mul(cos)?)?;
    cat(&[out1, out2], last)
}

/// Precomputed `cos`/`sin` tables for fixed-frequency rotary embeddings.
pub struct RotaryEmbedding<R: Runtime, E: FloatElem> {
    cos: Tensor<R, E>,
    sin: Tensor<R, E>,
    dim: usize,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for RotaryEmbedding<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "RotaryEmbedding(dim={}, max_len={})",
            self.dim,
            self.cos.shape().dim(0)
        )
    }
}

impl<R: Runtime, E: FloatElem> RotaryEmbedding<R, E> {
    /// Build tables for `max_len` positions over a head dimension of `dim`.
    pub fn new(dim: usize, max_len: usize, base: f32, device: &Device<R>) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(Error::config(format!(
                "rotary dimension must be even, got {dim}"
            )));
        }
        let half = dim / 2;
        let mut cos = Vec::with_capacity(max_len * half);
        let mut sin = Vec::with_capacity(max_len * half);
        for pos in 0..max_len {
            for i in 0..half {
                let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
                let angle = pos as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok(Self {
            cos: Tensor::from_f32(&cos, vec![max_len, half], device)?,
            sin: Tensor::from_f32(&sin, vec![max_len, half], device)?,
            dim,
        })
    }

    /// Head dimension the tables were built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply to `[batch, heads, seq, dim]`, starting at position `offset`.
    pub fn apply(&self, x: &Var<R, E>, offset: usize) -> Result<Var<R, E>> {
        let seq = x.shape().dim(2);
        let max_len = self.cos.shape().dim(0);
        if offset + seq > max_len {
            return Err(Error::config(format!(
                "rotary tables cover {max_len} positions but {} were requested",
                offset + seq
            )));
        }
        let half = self.dim / 2;
        let shape = vec![1, 1, seq, half];
        let cos = Var::constant(
            crate::tensor::ops::movement::slice(&self.cos, 0, offset, seq)?.reshape(shape.clone())?,
        );
        let sin = Var::constant(
            crate::tensor::ops::movement::slice(&self.sin, 0, offset, seq)?.reshape(shape)?,
        );
        rotate_halves(x, &cos, &sin)
    }
}
