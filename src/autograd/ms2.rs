//! Autograd wrappers for the differentiable MS2 adapters.
//!
//! Only the adapters carry gradients: [`Var::ms2_select_valid`] re-applies the
//! selection to the upstream gradient, and [`Var::ms2_lookup`] scatters it
//! back into the table with [`crate::tensor::ops::ms2::lookup_backward`], so
//! neither pass reads ids back to the host. Peak selection and peak features
//! are constants to the tape and stay outside it.

use cubecl::prelude::Runtime;

use crate::backend::FloatElem;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;

use super::var::Var;

impl<R: Runtime, E: FloatElem> Var<R, E> {
    /// Keep `self` (`[.., n, d]`) where `valid` (`[.., n]`) is non-zero, exact
    /// zero elsewhere. The adjoint is the same selection of the gradient.
    pub fn ms2_select_valid(&self, valid: &Tensor<R, E>) -> Result<Self> {
        let value = crate::tensor::ops::ms2::select_valid(&self.value, valid)?;
        let saved = valid.clone();
        Ok(Self::record(value, &[self], || {
            Box::new(move |g: &Tensor<R, E>| {
                Ok(vec![Some(crate::tensor::ops::ms2::select_valid(
                    g, &saved,
                )?)])
            })
        }))
    }

    /// Single-pass cross-attention of the queries `q` (`[rows, d]`) over the
    /// keys `k` and values `v` (`[B, M, d]`) of the row's spectrum
    /// (`row / rows_per_spectrum`), `heads` heads, `mask` (`[B, M]`) naming
    /// the slots that take weight: `[rows, d]`, the composed
    /// `permute`/`matmul_nt`/`masked_softmax`/`matmul`/`permute` chain's value
    /// in two launches ([`ms2::attn_weights`], [`ms2::attn_context`]) and its
    /// gradients in three ([`ms2::attn_backward`]), with no head permute of
    /// `q`, `k`, `v` or the context. The weights `[rows, heads, M]` are kept
    /// for the backward pass.
    ///
    /// [`ms2::attn_weights`]: crate::tensor::ops::ms2::attn_weights
    /// [`ms2::attn_context`]: crate::tensor::ops::ms2::attn_context
    /// [`ms2::attn_backward`]: crate::tensor::ops::ms2::attn_backward
    pub fn ms2_attention(
        q: &Self,
        k: &Self,
        v: &Self,
        mask: &Tensor<R, E>,
        heads: usize,
        rows_per_spectrum: usize,
    ) -> Result<Self> {
        use crate::tensor::ops::ms2;
        let rows = q.shape().dim(0);
        let mem = k.shape().dim(1);
        let mut w = Tensor::<R, E>::empty(vec![rows, heads, mem], q.value.device());
        ms2::attn_weights(&q.value, &k.value, mask, &mut w, heads, rows_per_spectrum)?;
        let value = ms2::attn_context(&w, &v.value, rows_per_spectrum)?;
        let (qs, ks, vs) = (q.value.clone(), k.value.clone(), v.value.clone());
        Ok(Self::record_with_mask(value, &[q, k, v], |want| {
            let want = [want[0], want[1], want[2]];
            Box::new(move |g: &Tensor<R, E>| {
                let (dq, dk, dv) =
                    ms2::attn_backward(g, &qs, &ks, &vs, &w, heads, rows_per_spectrum, want)?;
                Ok(vec![dq, dk, dv])
            })
        }))
    }

    /// Rows of `table` (`[V, d]`) rearranged by `ids` (`[rows]`), a zero row
    /// for every out-of-range id, where no table row is taken twice.
    /// `inverse` (`[V]`) names the output row each table row went to
    /// (out of range for a row that was not taken), so the adjoint is the
    /// same rearrangement backwards — one gather — rather than the
    /// accumulation [`Var::ms2_lookup`] needs for repeated ids.
    pub fn ms2_take_rows(table: &Self, ids: &IdTensor<R>, inverse: &IdTensor<R>) -> Result<Self> {
        if inverse.len() != table.shape().dim(0) {
            return Err(crate::error::Error::shape(format!(
                "ms2_take_rows needs one inverse id per table row, got {} for {}",
                inverse.len(),
                table.shape()
            )));
        }
        let value = crate::tensor::ops::ms2::lookup(&table.value, ids)?;
        let saved = inverse.clone();
        Ok(Self::record(value, &[table], || {
            Box::new(move |g: &Tensor<R, E>| {
                Ok(vec![Some(crate::tensor::ops::ms2::lookup(g, &saved)?)])
            })
        }))
    }

    /// Look up `ids` (`[rows]`) in `table` (`[V, d]`), giving `[rows, d]`
    /// with a zero row for every out-of-range id. The adjoint accumulates the
    /// gradient back into a `[V, d]` table on the device.
    pub fn ms2_lookup(table: &Self, ids: &IdTensor<R>) -> Result<Self> {
        let value = crate::tensor::ops::ms2::lookup(&table.value, ids)?;
        let rows = table.shape().dim(0);
        let saved = ids.clone();
        Ok(Self::record(value, &[table], || {
            Box::new(move |g: &Tensor<R, E>| {
                Ok(vec![Some(crate::tensor::ops::ms2::lookup_backward(
                    g, &saved, rows,
                )?)])
            })
        }))
    }
}
