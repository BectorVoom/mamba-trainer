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
