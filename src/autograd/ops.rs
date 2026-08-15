//! Differentiable operations.
//!
//! Only genuinely primitive ops carry a hand-written adjoint. Everything else —
//! softmax, SiLU, GELU, RMS norm, RoPE, fake quantization, the whole SSD scan — is
//! *composed* from these, so its gradient is correct by construction. Fusing a
//! composed op later is a performance change, never a correctness one.

use cubecl::prelude::Runtime;

use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::{elemwise, index, matmul as mm, movement, reduce, scan};
use crate::tensor::{Shape, Tensor};

use super::var::Var;

/// Sum a gradient back down to `target`, undoing NumPy broadcasting.
pub(crate) fn reduce_grad_to<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    target: &Shape,
) -> Result<Tensor<R, E>> {
    if grad.shape() == target {
        return Ok(grad.clone());
    }
    let rank = grad.rank();
    let padded = target.left_padded(rank);
    let mut out = grad.clone();
    for axis in 0..rank {
        if padded.dim(axis) == 1 && out.shape().dim(axis) != 1 {
            out = reduce::sum_dim(&out, axis)?;
        }
    }
    out.reshape(target.clone())
}

macro_rules! rule {
    (|$g:ident| $body:block) => {
        Box::new(move |$g: &Tensor<R, E>| $body)
    };
}

impl<R: Runtime, E: FloatElem> Var<R, E> {
    // -- binary ------------------------------------------------------------

    /// Elementwise sum with broadcasting.
    pub fn add(&self, other: &Self) -> Result<Self> {
        let value = elemwise::add(&self.value, &other.value)?;
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record(value, &[self, other], || {
            rule!(|g| { Ok(vec![Some(reduce_grad_to(g, &ls)?), Some(reduce_grad_to(g, &rs)?)]) })
        }))
    }

    /// Elementwise difference with broadcasting.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let value = elemwise::sub(&self.value, &other.value)?;
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record(value, &[self, other], || {
            rule!(|g| {
                Ok(vec![
                    Some(reduce_grad_to(g, &ls)?),
                    Some(reduce_grad_to(&elemwise::neg(g), &rs)?),
                ])
            })
        }))
    }

    /// Elementwise product with broadcasting.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let value = elemwise::mul(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record(value, &[self, other], || {
            rule!(|g| {
                Ok(vec![
                    Some(reduce_grad_to(&elemwise::mul(g, &b)?, &ls)?),
                    Some(reduce_grad_to(&elemwise::mul(g, &a)?, &rs)?),
                ])
            })
        }))
    }

    /// Elementwise quotient with broadcasting.
    pub fn div(&self, other: &Self) -> Result<Self> {
        let value = elemwise::div(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record(value, &[self, other], || {
            rule!(|g| {
                let da = elemwise::div(g, &b)?;
                // d/db (a/b) = -a / b^2
                let b2 = elemwise::mul(&b, &b)?;
                let db = elemwise::neg(&elemwise::div(&elemwise::mul(g, &a)?, &b2)?);
                Ok(vec![
                    Some(reduce_grad_to(&da, &ls)?),
                    Some(reduce_grad_to(&db, &rs)?),
                ])
            })
        }))
    }

    /// Elementwise maximum with broadcasting.
    pub fn maximum(&self, other: &Self) -> Result<Self> {
        let value = elemwise::maximum(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record(value, &[self, other], || {
            rule!(|g| {
                let a_wins = elemwise::greater(&a, &b)?;
                let b_wins = elemwise::rsub_scalar(&a_wins, 1.0);
                Ok(vec![
                    Some(reduce_grad_to(&elemwise::mul(g, &a_wins)?, &ls)?),
                    Some(reduce_grad_to(&elemwise::mul(g, &b_wins)?, &rs)?),
                ])
            })
        }))
    }

    /// Batched matrix product with leading-dimension broadcasting.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let value = mm::matmul(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record(value, &[self, other], || {
            rule!(|g| {
                let bt = movement::transpose(&b)?;
                let at = movement::transpose(&a)?;
                let da = mm::matmul(g, &bt)?;
                let db = mm::matmul(&at, g)?;
                Ok(vec![
                    Some(reduce_grad_to(&da, &ls)?),
                    Some(reduce_grad_to(&db, &rs)?),
                ])
            })
        }))
    }

    // -- unary -------------------------------------------------------------

    /// Negation.
    pub fn neg(&self) -> Self {
        let value = elemwise::neg(&self.value);
        Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::neg(g))]) })
        })
    }

    /// `exp(x)`.
    pub fn exp(&self) -> Self {
        let value = elemwise::exp(&self.value);
        let out = value.clone();
        Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::mul(g, &out)?)]) })
        })
    }

    /// Natural logarithm.
    pub fn log(&self) -> Self {
        let value = elemwise::log(&self.value);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::div(g, &x)?)]) })
        })
    }

    /// Square root.
    pub fn sqrt(&self) -> Self {
        let value = elemwise::sqrt(&self.value);
        let out = value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let denom = elemwise::mul_scalar(&out, 2.0);
                Ok(vec![Some(elemwise::div(g, &denom)?)])
            })
        })
    }

    /// Reciprocal square root.
    pub fn rsqrt(&self) -> Self {
        let value = elemwise::rsqrt(&self.value);
        let out = value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                // d/dx x^-1/2 = -1/2 * (x^-1/2)^3
                let cube = elemwise::mul(&elemwise::mul(&out, &out)?, &out)?;
                let d = elemwise::mul_scalar(&cube, -0.5);
                Ok(vec![Some(elemwise::mul(g, &d)?)])
            })
        })
    }

    /// Reciprocal.
    pub fn recip(&self) -> Self {
        let value = elemwise::recip(&self.value);
        let out = value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let d = elemwise::neg(&elemwise::mul(&out, &out)?);
                Ok(vec![Some(elemwise::mul(g, &d)?)])
            })
        })
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        let value = elemwise::abs(&self.value);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::mul(g, &elemwise::sign(&x))?)]) })
        })
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Self {
        let value = elemwise::tanh(&self.value);
        let out = value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let d = elemwise::rsub_scalar(&elemwise::mul(&out, &out)?, 1.0);
                Ok(vec![Some(elemwise::mul(g, &d)?)])
            })
        })
    }

    /// Logistic sigmoid.
    pub fn sigmoid(&self) -> Self {
        let value = elemwise::sigmoid(&self.value);
        let out = value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let one_minus = elemwise::rsub_scalar(&out, 1.0);
                let d = elemwise::mul(&out, &one_minus)?;
                Ok(vec![Some(elemwise::mul(g, &d)?)])
            })
        })
    }

    /// Sine.
    pub fn sin(&self) -> Self {
        let value = elemwise::sin(&self.value);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::mul(g, &elemwise::cos(&x))?)]) })
        })
    }

    /// Cosine.
    pub fn cos(&self) -> Self {
        let value = elemwise::cos(&self.value);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let d = elemwise::neg(&elemwise::sin(&x));
                Ok(vec![Some(elemwise::mul(g, &d)?)])
            })
        })
    }

    /// Gauss error function.
    pub fn erf(&self) -> Self {
        let value = elemwise::erf(&self.value);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                // d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
                let x2 = elemwise::mul(&x, &x)?;
                let d = elemwise::mul_scalar(
                    &elemwise::exp(&elemwise::neg(&x2)),
                    2.0 / core::f32::consts::PI.sqrt(),
                );
                Ok(vec![Some(elemwise::mul(g, &d)?)])
            })
        })
    }

    /// `max(x, 0)`.
    pub fn relu(&self) -> Self {
        let value = elemwise::relu(&self.value);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let mask = elemwise::gt_scalar(&x, 0.0);
                Ok(vec![Some(elemwise::mul(g, &mask)?)])
            })
        })
    }

    // -- scalar ------------------------------------------------------------

    /// `x + a`.
    pub fn add_scalar(&self, a: f32) -> Self {
        let value = elemwise::add_scalar(&self.value, a);
        Self::record(value, &[self], || rule!(|g| { Ok(vec![Some(g.clone())]) }))
    }

    /// `x * a`.
    pub fn mul_scalar(&self, a: f32) -> Self {
        let value = elemwise::mul_scalar(&self.value, a);
        Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::mul_scalar(g, a))]) })
        })
    }

    /// `a - x`.
    pub fn rsub_scalar(&self, a: f32) -> Self {
        let value = elemwise::rsub_scalar(&self.value, a);
        Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::neg(g))]) })
        })
    }

    /// `x ^ a`.
    pub fn powf_scalar(&self, a: f32) -> Self {
        let value = elemwise::powf_scalar(&self.value, a);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let d = elemwise::mul_scalar(&elemwise::powf_scalar(&x, a - 1.0), a);
                Ok(vec![Some(elemwise::mul(g, &d)?)])
            })
        })
    }

    /// Clamp into `[lo, hi]`; gradient is zero outside the range.
    pub fn clamp(&self, lo: f32, hi: f32) -> Self {
        let value = elemwise::clamp(&self.value, lo, hi);
        let x = self.value.clone();
        Self::record(value, &[self], || {
            rule!(|g| {
                let above = elemwise::gt_scalar(&x, lo);
                let below = elemwise::lt_scalar(&x, hi);
                let mask = elemwise::mul(&above, &below)?;
                Ok(vec![Some(elemwise::mul(g, &mask)?)])
            })
        })
    }

    /// Round to the nearest integer with a **straight-through estimator**: the
    /// forward value is rounded, the gradient passes unchanged. This is what makes
    /// quantization-aware training differentiable.
    pub fn round_ste(&self) -> Self {
        let value = elemwise::round(&self.value);
        Self::record(value, &[self], || rule!(|g| { Ok(vec![Some(g.clone())]) }))
    }

    // -- reductions --------------------------------------------------------

    /// Sum along `axis`, keeping it with size 1.
    pub fn sum_dim(&self, axis: usize) -> Result<Self> {
        let value = reduce::sum_dim(&self.value, axis)?;
        let shape = self.shape().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::expand(g, &shape)?)]) })
        }))
    }

    /// Mean along `axis`, keeping it with size 1.
    pub fn mean_dim(&self, axis: usize) -> Result<Self> {
        let len = self.shape().dim(axis) as f32;
        let value = reduce::mean_dim(&self.value, axis)?;
        let shape = self.shape().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| {
                let spread = elemwise::expand(g, &shape)?;
                Ok(vec![Some(elemwise::mul_scalar(&spread, 1.0 / len))])
            })
        }))
    }

    /// Maximum along `axis`, keeping it with size 1. Ties share the gradient.
    pub fn max_dim(&self, axis: usize) -> Result<Self> {
        let value = reduce::max_dim(&self.value, axis)?;
        let x = self.value.clone();
        let maxes = value.clone();
        let shape = self.shape().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| {
                let broadcast = elemwise::expand(&maxes, &shape)?;
                let hit = elemwise::sub(&x, &broadcast)?;
                let mask = elemwise::eq_scalar(&hit, 0.0);
                let count = reduce::sum_dim(&mask, axis)?;
                let share = elemwise::div(&mask, &elemwise::expand(&count, &shape)?)?;
                let spread = elemwise::expand(g, &shape)?;
                Ok(vec![Some(elemwise::mul(&spread, &share)?)])
            })
        }))
    }

    /// Sum of every element, as a rank-0 value.
    pub fn sum(&self) -> Result<Self> {
        let value = reduce::sum_all(&self.value)?;
        let shape = self.shape().clone();
        let device = self.device().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| {
                let scale = g.to_f32()[0];
                Ok(vec![Some(Tensor::full(shape.clone(), scale, &device))])
            })
        }))
    }

    /// Mean of every element, as a rank-0 value.
    pub fn mean(&self) -> Result<Self> {
        let n = self.value.len().max(1) as f32;
        Ok(self.sum()?.mul_scalar(1.0 / n))
    }

    // -- movement ----------------------------------------------------------

    /// Reinterpret with a new shape.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Self> {
        let value = self.value.reshape(shape)?;
        let original = self.shape().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(g.reshape(original.clone())?)]) })
        }))
    }

    /// Insert a size-1 axis.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        self.reshape(self.shape().unsqueezed(axis))
    }

    /// Remove a size-1 axis.
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        if self.shape().dim(axis) != 1 {
            return Err(Error::shape(format!(
                "cannot squeeze axis {axis} of {}",
                self.shape()
            )));
        }
        self.reshape(self.shape().without(axis))
    }

    /// Reorder axes.
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let value = movement::permute(&self.value, perm)?;
        let mut inverse = vec![0usize; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inverse[p] = i;
        }
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(movement::permute(g, &inverse)?)]) })
        }))
    }

    /// Swap the last two axes.
    pub fn transpose(&self) -> Result<Self> {
        let rank = self.rank();
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(rank - 2, rank - 1);
        self.permute(&perm)
    }

    /// Swap two axes.
    pub fn swap_axes(&self, a: usize, b: usize) -> Result<Self> {
        let mut perm: Vec<usize> = (0..self.rank()).collect();
        perm.swap(a, b);
        self.permute(&perm)
    }

    /// Broadcast to a larger shape.
    pub fn expand(&self, target: impl Into<Shape>) -> Result<Self> {
        let target = target.into();
        let value = elemwise::expand(&self.value, &target)?;
        let original = self.shape().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(reduce_grad_to(g, &original)?)]) })
        }))
    }

    /// Reverse along `axis`.
    pub fn flip(&self, axis: usize) -> Result<Self> {
        let value = movement::flip(&self.value, axis)?;
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(movement::flip(g, axis)?)]) })
        }))
    }

    /// Take `len` entries starting at `start` along `axis`.
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> Result<Self> {
        let value = movement::slice(&self.value, axis, start, len)?;
        let shape = self.shape().clone();
        let device = self.device().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| {
                let total = shape.dim(axis);
                let mut parts = Vec::new();
                if start > 0 {
                    parts.push(Tensor::zeros(shape.with_dim(axis, start), &device));
                }
                parts.push(g.clone());
                let tail = total - start - len;
                if tail > 0 {
                    parts.push(Tensor::zeros(shape.with_dim(axis, tail), &device));
                }
                Ok(vec![Some(movement::cat(&parts, axis)?)])
            })
        }))
    }

    /// Shift by one step along `axis`, filling the first slot with zeros.
    pub fn shift_right(&self, axis: usize) -> Result<Self> {
        let len = self.shape().dim(axis);
        if len == 0 {
            return Ok(self.clone());
        }
        let head = self.slice(axis, 0, len - 1)?;
        let zeros = Var::constant(Tensor::zeros(
            self.shape().with_dim(axis, 1),
            self.device(),
        ));
        cat(&[zeros, head], axis)
    }

    // -- indexing ----------------------------------------------------------

    /// For each row of `self` (`[..., last]`), select the element named by `ids`.
    pub fn take_along_last(&self, ids: &IdTensor<R>) -> Result<Self> {
        let value = index::take_along_last(&self.value, ids)?;
        let shape = self.shape().clone();
        let last = shape.dim_from_end(0);
        let ids = ids.clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| {
                let onehot: Tensor<R, E> = index::one_hot(&ids, last)?;
                let spread = g.reshape(g.shape().unsqueezed(g.rank()))?;
                Ok(vec![Some(elemwise::mul(&onehot, &spread)?)])
            })
        }))
    }

    // -- scans -------------------------------------------------------------

    /// Inclusive prefix sum along `axis`.
    pub fn cumsum(&self, axis: usize) -> Result<Self> {
        let value = scan::cumsum(&self.value, axis)?;
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(scan::cumsum_reverse(g, axis)?)]) })
        }))
    }

    /// Exclusive prefix sum along `axis`.
    pub fn cumsum_exclusive(&self, axis: usize) -> Result<Self> {
        let value = scan::cumsum_exclusive(&self.value, axis)?;
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(scan::cumsum_reverse_exclusive(g, axis)?)]) })
        }))
    }

    // -- composed activations ---------------------------------------------

    /// `x * sigmoid(x)`.
    pub fn silu(&self) -> Result<Self> {
        self.mul(&self.sigmoid())
    }

    /// `log(1 + exp(x))`, computed stably.
    pub fn softplus(&self) -> Result<Self> {
        // softplus(x) = max(x, 0) + log(1 + exp(-|x|))
        let stable = self.abs().neg().exp().add_scalar(1.0).log();
        self.relu().add(&stable)
    }

    /// Exact GELU via the error function.
    pub fn gelu(&self) -> Result<Self> {
        let inner = self.mul_scalar(core::f32::consts::FRAC_1_SQRT_2).erf();
        let gate = inner.add_scalar(1.0).mul_scalar(0.5);
        self.mul(&gate)
    }

    /// Softmax along `axis`.
    pub fn softmax(&self, axis: usize) -> Result<Self> {
        // The max shift is a constant, so detaching it keeps the adjoint exact
        // while avoiding a needless max-routing term.
        let shifted = self.sub(&self.max_dim(axis)?.detach())?;
        let e = shifted.exp();
        let denom = e.sum_dim(axis)?;
        e.div(&denom)
    }

    /// Log-softmax along `axis`.
    pub fn log_softmax(&self, axis: usize) -> Result<Self> {
        let shifted = self.sub(&self.max_dim(axis)?.detach())?;
        let denom = shifted.exp().sum_dim(axis)?.log();
        shifted.sub(&denom)
    }
}

/// Concatenate along `axis`.
pub fn cat<R: Runtime, E: FloatElem>(parts: &[Var<R, E>], axis: usize) -> Result<Var<R, E>> {
    if parts.is_empty() {
        return Err(Error::shape("cat needs at least one value".to_string()));
    }
    let tensors: Vec<Tensor<R, E>> = parts.iter().map(|p| p.value.clone()).collect();
    let value = movement::cat(&tensors, axis)?;
    let sizes: Vec<usize> = parts.iter().map(|p| p.shape().dim(axis)).collect();
    let refs: Vec<&Var<R, E>> = parts.iter().collect();
    Ok(Var::record(value, &refs, || {
        rule!(|g| {
            let pieces = movement::split(g, &sizes, axis)?;
            Ok(pieces.into_iter().map(Some).collect())
        })
    }))
}

/// Sum a list of values elementwise.
pub fn sum_all_vars<R: Runtime, E: FloatElem>(parts: &[Var<R, E>]) -> Result<Var<R, E>> {
    let mut iter = parts.iter();
    let first = iter
        .next()
        .ok_or_else(|| Error::shape("sum of an empty list".to_string()))?
        .clone();
    iter.try_fold(first, |acc, v| acc.add(v))
}

/// Look up rows of an embedding table.
pub fn embedding<R: Runtime, E: FloatElem>(
    table: &Var<R, E>,
    ids: &IdTensor<R>,
) -> Result<Var<R, E>> {
    let value = index::gather_rows(&table.value, ids)?;
    let num_rows = table.shape().dim(0);
    let width = table.shape().dim(1);
    let ids = ids.clone();
    Ok(Var::record(value, &[table], || {
        rule!(|g| {
            let flat = g.reshape(Shape::new(vec![g.len() / width, width]))?;
            Ok(vec![Some(index::scatter_add_rows(&flat, &ids, num_rows)?)])
        })
    }))
}
