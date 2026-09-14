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
use crate::tensor::ops::{elemwise, fused, index, matmul as mm, movement, reduce, scan};
use crate::tensor::{Shape, Tensor};

use super::var::Var;

/// Gradient slots the pieces of a split fill in during the backward walk, shared
/// between the pieces' rules and the sink that concatenates them.
type SharedBands<T> = std::rc::Rc<std::cell::RefCell<T>>;

/// Sum a gradient back down to `target`, undoing NumPy broadcasting.
///
/// Adjacent axes that all have to go are summed in one pass rather than one each.
/// The tensor is contiguous, so a run of them is a single axis after a free reshape,
/// and the intermediate it would otherwise have written never exists: a per-head bias
/// of `[1, 1, heads, 1, state]` under a `[batch, seq, heads, 1, state]` gradient used
/// to take two launches and an intermediate the size of the batch axis, and now takes
/// one.
pub(crate) fn reduce_grad_to<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    target: &Shape,
) -> Result<Tensor<R, E>> {
    if grad.shape() == target {
        return Ok(grad.clone());
    }
    let rank = grad.rank();
    let padded = target.left_padded(rank);
    // Runs of axes that share a fate collapse into one axis of their product, which
    // is exact because the tensor is contiguous and the reshape is free.
    let mut merged: Vec<(usize, bool)> = Vec::with_capacity(rank);
    for axis in 0..rank {
        let extent = grad.shape().dim(axis);
        let drop = padded.dim(axis) == 1 && extent != 1;
        match merged.last_mut() {
            Some((size, was)) if *was == drop => *size *= extent,
            _ => merged.push((extent, drop)),
        }
    }

    let dims: Vec<usize> = merged.iter().map(|(size, _)| *size).collect();
    let mut out = grad.reshape(Shape::new(dims))?;
    for (axis, (_, drop)) in merged.iter().enumerate() {
        if *drop {
            // `sum_dim` keeps the axis at extent one, so later indices still line up.
            out = reduce::sum_dim(&out, axis)?;
        }
    }
    out.reshape(target.clone())
}

/// Reduce an RMS-norm bias gradient from `dx`'s shape (`[.., heads, rank, dim]`)
/// down to the bias's own `[heads, dim]`.
///
/// This is not [`reduce_grad_to`]: that helper only drops axes NumPy-style, by
/// left-padding the target and summing wherever the padded target reads `1`. Here
/// the bias's two axes are not adjacent in `dx` — a `rank` axis sits between them —
/// so the axis to keep (`heads`) is *in the middle*, not at the tail, and no
/// left-padding of `[heads, dim]` can express that.
fn rms_bias_grad<R: Runtime, E: FloatElem>(
    dx: &Tensor<R, E>,
    bias_shape: &Shape,
) -> Result<Tensor<R, E>> {
    let dim = dx.shape().dim_from_end(0);
    let rank = dx.shape().dim_from_end(1);
    let heads = dx.shape().dim_from_end(2);
    let outer = dx.len() / (heads * rank * dim).max(1);
    let reshaped = dx.reshape(Shape::new(vec![outer, heads, rank, dim]))?;
    let summed_outer =
        reduce::sum_dim(&reshaped, 0)?.reshape(Shape::new(vec![heads, rank, dim]))?;
    reduce::sum_dim(&summed_outer, 1)?.reshape(bias_shape.clone())
}

/// `Aᵀ G`, summed over every leading batch axis, for the adjoint of a product whose
/// right operand is a plain matrix.
///
/// That is what a `Linear` is: `[batch, seq, in] @ [in, out]`, where the weight is
/// broadcast across the batch. Taking the adjoint batch by batch and reducing
/// afterwards computes `batch` separate `[in, out]` products and then throws
/// `batch - 1` of them away — for a Mamba-3 input projection, four `[512, 4640]`
/// gradients written and a thirty-eight megabyte reduction to get back to one.
/// Folding the batch axes into the contraction instead makes it a single product
/// with a `batch` times longer inner dimension, which is the same arithmetic against
/// a quarter of the memory and no reduction at all.
fn weight_grad<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    grad: &Tensor<R, E>,
    target: &Shape,
) -> Result<Tensor<R, E>> {
    let batched = lhs.rank() > 2
        && target.rank() == 2
        && grad.rank() == lhs.rank()
        && lhs.dims()[..lhs.rank() - 2] == grad.dims()[..grad.rank() - 2];
    if batched {
        let k = lhs.shape().dim_from_end(0);
        let n = grad.shape().dim_from_end(0);
        let rows = lhs.len() / k;
        // Both operands are contiguous with the batch axes outermost, so stacking
        // them into one tall matrix is a reshape.
        if rows == grad.len() / n && target.dim(0) == k && target.dim(1) == n {
            let stacked_lhs = lhs.reshape(Shape::new(vec![rows, k]))?;
            let stacked_grad = grad.reshape(Shape::new(vec![rows, n]))?;
            return mm::matmul_tn(&stacked_lhs, &stacked_grad);
        }
    }
    reduce_grad_to(&mm::matmul_tn(lhs, grad)?, target)
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
        Ok(Self::record_with_mask(value, &[self, other], |want| {
            let (wl, wr) = (want[0], want[1]);
            rule!(|g| {
                Ok(vec![
                    if wl {
                        Some(reduce_grad_to(g, &ls)?)
                    } else {
                        None
                    },
                    if wr {
                        Some(reduce_grad_to(g, &rs)?)
                    } else {
                        None
                    },
                ])
            })
        }))
    }

    /// Elementwise difference with broadcasting.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let value = elemwise::sub(&self.value, &other.value)?;
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record_with_mask(value, &[self, other], |want| {
            let (wl, wr) = (want[0], want[1]);
            rule!(|g| {
                Ok(vec![
                    if wl {
                        Some(reduce_grad_to(g, &ls)?)
                    } else {
                        None
                    },
                    if wr {
                        Some(reduce_grad_to(&elemwise::neg(g), &rs)?)
                    } else {
                        None
                    },
                ])
            })
        }))
    }

    /// Elementwise product with broadcasting.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let value = elemwise::mul(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record_with_mask(value, &[self, other], |want| {
            let (wl, wr) = (want[0], want[1]);
            rule!(|g| {
                Ok(vec![
                    if wl {
                        Some(reduce_grad_to(&elemwise::mul(g, &b)?, &ls)?)
                    } else {
                        None
                    },
                    if wr {
                        Some(reduce_grad_to(&elemwise::mul(g, &a)?, &rs)?)
                    } else {
                        None
                    },
                ])
            })
        }))
    }

    /// Elementwise quotient with broadcasting.
    pub fn div(&self, other: &Self) -> Result<Self> {
        let value = elemwise::div(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record_with_mask(value, &[self, other], |want| {
            let (wl, wr) = (want[0], want[1]);
            rule!(|g| {
                let da = if wl {
                    Some(reduce_grad_to(&elemwise::div(g, &b)?, &ls)?)
                } else {
                    None
                };
                let db = if wr {
                    // d/db (a/b) = -a / b^2
                    let b2 = elemwise::mul(&b, &b)?;
                    let raw = elemwise::neg(&elemwise::div(&elemwise::mul(g, &a)?, &b2)?);
                    Some(reduce_grad_to(&raw, &rs)?)
                } else {
                    None
                };
                Ok(vec![da, db])
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

    /// Elementwise minimum with broadcasting.
    ///
    /// The gradient goes entirely to whichever operand won, which is exactly the
    /// behaviour PPO's clipped surrogate is built on: where the clipped branch is
    /// the smaller of the two, the unclipped one — and with it the policy — gets no
    /// gradient at all, and the update stops at the edge of the trust region.
    pub fn minimum(&self, other: &Self) -> Result<Self> {
        let value = elemwise::minimum(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        Ok(Self::record(value, &[self, other], || {
            rule!(|g| {
                // `a` wins a tie, matching `maximum`, so the two agree on `a == b`
                // and neither double-counts it.
                let b_wins = elemwise::greater(&a, &b)?;
                let a_wins = elemwise::rsub_scalar(&b_wins, 1.0);
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
        // `dA = G Bᵀ` and `dB = Aᵀ G`. Both transposes are read by the kernel rather
        // than materialised: a permutation that swaps the contiguous axis is the one
        // case the strided copy cannot vectorise, and doing two of them per matmul
        // was 14% of a training step.
        Ok(Self::record_with_mask(value, &[self, other], |want| {
            let (wl, wr) = (want[0], want[1]);
            rule!(|g| {
                let da = if wl {
                    Some(reduce_grad_to(&mm::matmul_nt(g, &b)?, &ls)?)
                } else {
                    None
                };
                let db = if wr {
                    Some(weight_grad(&a, g, &rs)?)
                } else {
                    None
                };
                Ok(vec![da, db])
            })
        }))
    }

    /// `self @ otherᵀ`, contracting the trailing axis of both operands.
    ///
    /// The forward-pass counterpart of the transpose [`Var::matmul`]'s own adjoint
    /// already reads instead of materialising. Any call site shaped
    /// `a.matmul(&b.transpose()?)` should be `a.matmul_nt(&b)` instead: the
    /// transpose there is the one case a strided copy cannot vectorise, because it
    /// swaps the contiguous axis.
    pub fn matmul_nt(&self, other: &Self) -> Result<Self> {
        let value = mm::matmul_nt(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        let (ls, rs) = (self.shape().clone(), other.shape().clone());
        // `y = A Bᵀ`; `dA = G B` (plain) and `dB = Gᵀ A` (both trailing axes stay
        // put, so neither adjoint needs a transpose of its own either). `dB` goes
        // through `weight_grad` for the same reason [`Var::matmul`]'s does: with a
        // two-dimensional right operand — the tied embedding table — the batched
        // form would materialise one `[vocab, d_model]` gradient per sequence
        // before summing them, which is the batch size times the memory the
        // stacked matmul needs.
        Ok(Self::record_with_mask(value, &[self, other], |want| {
            let (wl, wr) = (want[0], want[1]);
            rule!(|g| {
                let da = if wl {
                    Some(reduce_grad_to(&mm::matmul(g, &b)?, &ls)?)
                } else {
                    None
                };
                let db = if wr {
                    Some(weight_grad(g, &a, &rs)?)
                } else {
                    None
                };
                Ok(vec![da, db])
            })
        }))
    }

    /// Fused root-mean-square normalisation over the trailing axis.
    ///
    /// This is the one composed op in the crate that carries a hand-written adjoint,
    /// and it earns it: written out of primitives it is seven launches forward and
    /// about fifteen back, on a tensor small enough that all of them are dispatch
    /// overhead. Fused it is one and two. The adjoint is
    ///
    /// ```text
    /// dx_j = r * g_j * w_j - (r^3 / d) * x_j * sum_i(g_i * w_i * x_i)
    /// dw_i = sum over rows of g_i * x_i * r
    /// ```
    ///
    /// with `r = rsqrt(mean(x^2) + eps)`, and `tests/autograd.rs` checks it against
    /// central differences with and without a gain.
    pub fn rms_norm(&self, weight: Option<&Self>, eps: f32) -> Result<Self> {
        self.rms_norm_biased(None, weight, eps)
    }

    /// [`Var::rms_norm`], with a per-head bias added to the input before the norm's
    /// statistics are taken — the Mamba-3 `B`/`C` bias fused into the norm that
    /// reads them. `bias` is `[heads, dim]`; see [`fused::rms_norm`] for the exact
    /// activation shape it expects `self` to have.
    ///
    /// The bias enters additively, so `d/dbias` is `d/dself` (the same `dx` the
    /// input gets) summed back down to the bias's shape with [`rms_bias_grad`].
    pub fn rms_norm_biased(
        &self,
        bias: Option<&Self>,
        weight: Option<&Self>,
        eps: f32,
    ) -> Result<Self> {
        let bias_v = bias.map(|b| b.value.clone());
        let gain = weight.map(|w| w.value.clone());
        let (value, scale) = fused::rms_norm(&self.value, bias_v.as_ref(), gain.as_ref(), eps)?;
        let x = self.value.clone();
        let bias_shape = bias.map(|b| b.shape().clone());
        let mut parents: Vec<&Self> = vec![self];
        if let Some(b) = bias {
            parents.push(b);
        }
        if let Some(w) = weight {
            parents.push(w);
        }
        Ok(Self::record(value, &parents, || {
            rule!(|g| {
                let (dx, dw) =
                    fused::rms_norm_backward(g, &x, bias_v.as_ref(), gain.as_ref(), &scale)?;
                let mut out = vec![Some(dx.clone())];
                if let Some(shape) = &bias_shape {
                    out.push(Some(rms_bias_grad(&dx, shape)?));
                }
                if let Some(dw) = dw {
                    out.push(Some(dw));
                }
                Ok(out)
            })
        }))
    }

    /// Rotate the two halves of the trailing axis: the RoPE / rotating-frame
    /// primitive, fused.
    ///
    /// Falls back to the composed form when the angle tables broadcast in a way the
    /// fused kernel's row mapping cannot express — see
    /// [`fused::RotationLayout::resolve`]. Both call sites in the crate take the
    /// fused path.
    pub fn rotate_halves(&self, cos: &Self, sin: &Self) -> Result<Self> {
        let Some(layout) = fused::RotationLayout::resolve(self.dims(), cos.dims())
            .filter(|_| cos.shape() == sin.shape())
        else {
            return self.rotate_halves_composed(cos, sin);
        };
        let value = fused::rotate_halves(&self.value, &cos.value, &sin.value, layout, false)?;
        let (x, c, s) = (self.value.clone(), cos.value.clone(), sin.value.clone());
        let table_shape = cos.shape().clone();
        Ok(Self::record(value, &[self, cos, sin], || {
            rule!(|g| {
                let (dx, dcos, dsin) = fused::rotate_halves_backward(g, &x, &c, &s, layout, false)?;
                Ok(vec![
                    Some(dx),
                    Some(reduce_grad_to(&dcos, &table_shape)?),
                    Some(reduce_grad_to(&dsin, &table_shape)?),
                ])
            })
        }))
    }

    /// Rotate the two halves of the trailing axis by `-phi`, taking the angle
    /// directly instead of a precomputed `(cos, sin)` pair.
    ///
    /// The sine and cosine are computed inside the kernel. Two transcendentals per
    /// element cost far less than the three launches that materialising `cos phi`,
    /// `sin phi` and its negation would, and the Mamba-3 step rotates twice from the
    /// same angle.
    pub fn rotate_by_angle(&self, phi: &Self) -> Result<Self> {
        let Some(layout) = fused::RotationLayout::resolve(self.dims(), phi.dims()) else {
            let cos = phi.cos();
            let sin = phi.sin().neg();
            return self.rotate_halves_composed(&cos, &sin);
        };
        let value = fused::rotate_halves(&self.value, &phi.value, &phi.value, layout, true)?;
        let (x, p) = (self.value.clone(), phi.value.clone());
        let table_shape = phi.shape().clone();
        Ok(Self::record(value, &[self, phi], || {
            rule!(|g| {
                let (dx, dphi, _) = fused::rotate_halves_backward(g, &x, &p, &p, layout, true)?;
                Ok(vec![Some(dx), Some(reduce_grad_to(&dphi, &table_shape)?)])
            })
        }))
    }

    /// The primitive-by-primitive rotation, kept as the fallback and as the thing
    /// the fused kernel is tested against.
    pub fn rotate_halves_composed(&self, cos: &Self, sin: &Self) -> Result<Self> {
        let last = self.rank() - 1;
        let d = self.shape().dim(last);
        if !d.is_multiple_of(2) {
            return Err(Error::shape(format!(
                "rotation needs an even trailing dimension, got {d}"
            )));
        }
        let half = d / 2;
        let x1 = self.slice(last, 0, half)?;
        let x2 = self.slice(last, half, half)?;
        let out1 = x1.mul(cos)?.sub(&x2.mul(sin)?)?;
        let out2 = x1.mul(sin)?.add(&x2.mul(cos)?)?;
        super::cat(&[out1, out2], last)
    }

    /// A depthwise causal convolution over `[batch, seq, channels]`, fused.
    ///
    /// `history` holds the `taps - 1` positions before this window; `None` means
    /// they are zero. Returns only the output — the history to carry forward is
    /// [`Var::causal_conv1d_history`], which needs no gradient because it is a
    /// verbatim copy of positions this call already accounted for.
    ///
    /// `reset` is an optional `[batch, seq]` episode-termination mask; taps that
    /// would reach back across a set flag are dropped.
    pub fn causal_conv1d(
        &self,
        history: Option<&Self>,
        weight: &Self,
        bias: Option<&Self>,
        reset: Option<&Tensor<R, E>>,
    ) -> Result<Self> {
        let hist = history.map(|h| h.value.clone());
        let bias_value = bias.map(|b| b.value.clone());
        let saved_reset = reset.cloned();
        let value = fused::causal_conv1d(
            &self.value,
            hist.as_ref(),
            &weight.value,
            bias_value.as_ref(),
            reset,
        )?;

        let x = self.value.clone();
        let w = weight.value.clone();
        let mut parents: Vec<&Self> = vec![self, weight];
        if let Some(h) = history {
            parents.push(h);
        }
        if let Some(b) = bias {
            parents.push(b);
        }
        Ok(Self::record(value, &parents, || {
            rule!(|g| {
                let (dx, dh, dw, db) = fused::causal_conv1d_backward(
                    g,
                    &x,
                    hist.as_ref(),
                    &w,
                    bias_value.as_ref(),
                    saved_reset.as_ref(),
                )?;
                let mut out = vec![Some(dx), Some(dw)];
                if let Some(dh) = dh {
                    out.push(Some(dh));
                }
                if let Some(db) = db {
                    out.push(Some(db));
                }
                Ok(out)
            })
        }))
    }

    /// The `taps - 1` positions to hand the next [`Var::causal_conv1d`] call.
    ///
    /// Differentiable, because training over a sequence of windows backpropagates
    /// through the state that joins them.
    pub fn causal_conv1d_history(
        &self,
        history: Option<&Self>,
        weight: &Self,
        reset: Option<&Tensor<R, E>>,
    ) -> Result<Self> {
        let hist = history.map(|h| h.value.clone());
        let saved_reset = reset.cloned();
        let value = fused::causal_conv1d_history(&self.value, hist.as_ref(), &weight.value, reset)?;
        let x = self.value.clone();
        let w = weight.value.clone();
        let mut parents: Vec<&Self> = vec![self];
        if let Some(h) = history {
            parents.push(h);
        }
        Ok(Self::record(value, &parents, || {
            rule!(|g| {
                let (dx, dh) = fused::causal_conv1d_history_backward(
                    g,
                    &x,
                    hist.as_ref(),
                    &w,
                    saved_reset.as_ref(),
                )?;
                let mut out = vec![Some(dx)];
                if let Some(dh) = dh {
                    out.push(Some(dh));
                }
                Ok(out)
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

    /// Mask illegal positions' logits (to `F::min_value()`, a probability of
    /// exactly zero), for a legal-action mask over the
    /// trailing axis. `legal` is the same shape as `self`, `1` where an action
    /// is legal and `0` where it is not — see
    /// [`crate::tensor::ops::elemwise::mask_logits`], which computes the
    /// forward value.
    ///
    /// The gradient at an illegal position is exactly zero, and for the reason
    /// that actually matters here: that position's forward value came from the
    /// *constant* mask value, not from `self`, so nothing legitimately flows back
    /// to it regardless of what the loss upstream computed from the (already
    /// exactly zero) probability it produced. `elemwise::mul(g, legal)` states
    /// that directly rather than leaving it to fall out of the arithmetic.
    pub fn mask_logits(&self, legal: &Tensor<R, E>) -> Result<Self> {
        let value = elemwise::mask_logits(&self.value, legal)?;
        let legal = legal.clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::mul(g, &legal)?)]) })
        }))
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
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(elemwise::expand(g, &shape)?)]) })
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

    /// Reverse `axis` for only the listed inner channel bands, in one launch.
    ///
    /// See [`movement::reverse_bands`]. The operation is its own inverse and
    /// linear, so the adjoint is the same reversal applied to the gradient.
    pub fn reverse_bands(&self, axis: usize, reversed: &[(usize, usize)]) -> Result<Self> {
        let value = movement::reverse_bands(&self.value, axis, reversed)?;
        let reversed = reversed.to_vec();
        Ok(Self::record(value, &[self], move || {
            rule!(|g| { Ok(vec![Some(movement::reverse_bands(g, axis, &reversed)?)]) })
        }))
    }

    /// Take `len` entries starting at `start` along `axis`.
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> Result<Self> {
        // A slice that covers the whole axis is the identity, and recording it as a
        // slice is not free: the adjoint of a slice is a full-size buffer with the
        // gradient written into one band of it, so a rank-1 MIMO scan — which slices
        // `[b, t, h, p, 1]` on the rank axis once per rank pair — was paying a
        // full-size copy per slice in the backward pass to move a tensor onto itself.
        if start == 0 && len == self.shape().dim(axis) {
            return Ok(self.clone());
        }
        let value = movement::slice(&self.value, axis, start, len)?;
        let shape = self.shape().clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(fused::slice_backward(g, &shape, axis, start)?)]) })
        }))
    }

    /// Cut `axis` into consecutive pieces with the given sizes.
    ///
    /// Equivalent to one [`Var::slice`] per piece, and much cheaper to differentiate.
    /// The pieces tile the axis, so the gradient of the whole is exactly the
    /// *concatenation* of the pieces' gradients — whereas differentiating N separate
    /// slices builds N full-size buffers that are zero everywhere but one band and
    /// then adds them together. On the fused input projection of a Mamba-3 layer,
    /// `[4, 512, 4640]` split five ways, that is five full-size writes plus four
    /// full-size adds — some 650 MiB of traffic per layer — against one buffer written
    /// once, in bands, by this.
    ///
    /// The tape carries one output gradient per node, so the assembly happens in a
    /// *sink* node created before the pieces. Ids increase with creation order and
    /// the backward walk descends them, so the sink is visited after every piece has
    /// stashed its band. Pieces that receive no gradient leave a zero band.
    pub fn split(&self, sizes: &[usize], axis: usize) -> Result<Vec<Self>> {
        let values = movement::split(&self.value, sizes, axis)?;
        if values.len() < 2 {
            return Ok(values.into_iter().map(Var::constant).collect());
        }
        if self.trace.is_none() || !super::grad_mode::is_enabled() {
            return Ok(values.into_iter().map(Var::constant).collect());
        }

        let full = self.shape().clone();
        let sizes: Vec<usize> = sizes.to_vec();
        let device = self.device().clone();
        let bands: SharedBands<Vec<Option<Tensor<R, E>>>> =
            std::rc::Rc::new(std::cell::RefCell::new(vec![None; sizes.len()]));

        // The token a piece hands the sink to say "I contributed". Empty, so the
        // accumulating add in the backward walk allocates nothing and launches
        // nothing; all it does is put the sink on the worklist.
        let token = Tensor::empty(Shape::new(vec![0]), &device);

        let sink = {
            let bands = bands.clone();
            let full = full.clone();
            let sizes = sizes.clone();
            let device = device.clone();
            Self::record(token.clone(), &[self], move || {
                rule!(|_g| {
                    let mut slots = bands.borrow_mut();
                    let parts: Vec<Tensor<R, E>> = slots
                        .iter_mut()
                        .enumerate()
                        .map(|(i, slot)| {
                            slot.take().unwrap_or_else(|| {
                                Tensor::zeros(full.with_dim(axis, sizes[i]), &device)
                            })
                        })
                        .collect();
                    Ok(vec![Some(movement::cat(&parts, axis)?)])
                })
            })
        };

        Ok(values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let bands = bands.clone();
                let token = token.clone();
                Self::record(value, &[&sink], move || {
                    rule!(|g| {
                        bands.borrow_mut()[i] = Some(g.clone());
                        Ok(vec![Some(token.clone())])
                    })
                })
            })
            .collect())
    }

    /// Shift by one step along `axis`, filling the first slot with zeros.
    pub fn shift_right(&self, axis: usize) -> Result<Self> {
        let len = self.shape().dim(axis);
        if len == 0 {
            return Ok(self.clone());
        }
        let head = self.slice(axis, 0, len - 1)?;
        let zeros = Var::constant(Tensor::zeros(self.shape().with_dim(axis, 1), self.device()));
        cat(&[zeros, head], axis)
    }

    // -- indexing ----------------------------------------------------------

    /// The per-token cross entropy of `[rows, classes]` logits against `[rows]`
    /// targets, with optional label smoothing, fused.
    ///
    /// Composed out of primitives — [`Var::log_softmax`], [`Var::take_along_last`]
    /// and the smoothing mix, the oracle `train::cross_entropy_per_token_composed`
    /// keeps — the forward pass is six vocab-sized launches and the backward
    /// materialises a dense `[rows, classes]` one-hot before multiplying it away:
    /// the largest transient allocation in a training step. Fused it is one launch
    /// each way, and the backward rebuilds the softmax from a saved `[rows]`
    /// log-sum-exp vector instead of a logits-sized activation.
    pub fn cross_entropy_rows(&self, ids: &IdTensor<R>, smoothing: f32) -> Result<Self> {
        let (value, lse) = fused::cross_entropy_rows(&self.value, ids, smoothing)?;
        let x = self.value.clone();
        let ids = ids.clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| {
                Ok(vec![Some(fused::cross_entropy_rows_backward(
                    g, &x, &ids, &lse, smoothing,
                )?)])
            })
        }))
    }

    // -- reinforcement learning --------------------------------------------

    /// The clipped surrogate `min(r A, clip(r, 1±ε) A)` per position, fused.
    ///
    /// `self` is the new policy's log-probability of the action it took, `old` the
    /// behaviour policy's, and the ratio between them is what the trust region
    /// bounds. Composed — a subtraction, an `exp`, two products, a `clamp` and a
    /// `minimum` — this is six launches forward and about ten back, over vectors of
    /// a few thousand floats where a launch costs far more than the arithmetic.
    /// Fused it is one each way.
    ///
    /// The second return value is the `[rows]` ratio, off the tape: the adjoint reads
    /// it instead of keeping the chain that produced it, and
    /// [`crate::rl::ppo_objective`] reports its divergence as the diagnostic that
    /// says whether the clip is still doing its job.
    pub fn ppo_surrogate(
        &self,
        old: &Tensor<R, E>,
        advantages: &Tensor<R, E>,
        eps: f32,
    ) -> Result<(Self, Tensor<R, E>)> {
        let (value, ratio) = fused::ppo_surrogate(&self.value, old, advantages, eps)?;
        let saved = ratio.clone();
        let advantages = advantages.clone();
        let out = Self::record(value, &[self], || {
            rule!(|g| {
                Ok(vec![Some(fused::ppo_surrogate_backward(
                    g,
                    &saved,
                    &advantages,
                    eps,
                )?)])
            })
        });
        Ok((out, ratio))
    }

    /// The critic's per-position squared error, fused, optionally under the same
    /// trust region on the value scale.
    ///
    /// `self` is the replayed estimate, `returns` the λ-return it is fitted to and
    /// `old` the estimate made when the window was collected. With `clip` the loss is
    /// the *larger* of the plain error and the one a `±eps`-bounded estimate would
    /// have made, so a single update cannot move the critic further than `eps` unless
    /// staying put is the worse mistake.
    pub fn ppo_value_loss(
        &self,
        returns: &Tensor<R, E>,
        old: &Tensor<R, E>,
        eps: f32,
        clip: bool,
    ) -> Result<Self> {
        let value = fused::ppo_value_loss(&self.value, returns, old, eps, clip)?;
        let (x, returns, old) = (self.value.clone(), returns.clone(), old.clone());
        Ok(Self::record(value, &[self], || {
            rule!(|g| {
                Ok(vec![Some(fused::ppo_value_loss_backward(
                    g, &x, &returns, &old, eps, clip,
                )?)])
            })
        }))
    }

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
        let value = fused::silu(&self.value);
        let x = self.value.clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(fused::silu_backward(g, &x))]) })
        }))
    }

    /// `x * sigmoid(x)`, one primitive at a time. The reference
    /// [`Var::silu`] is checked against.
    pub fn silu_composed(&self) -> Result<Self> {
        self.mul(&self.sigmoid())
    }

    /// `self * silu(other)`, fused — the gate at the end of a Mamba-3 layer.
    pub fn swiglu(&self, other: &Self) -> Result<Self> {
        let value = fused::swiglu(&self.value, &other.value)?;
        let (a, b) = (self.value.clone(), other.value.clone());
        Ok(Self::record_with_mask(value, &[self, other], |want| {
            let (wa, wb) = (want[0], want[1]);
            rule!(|g| {
                let (da, db) = fused::swiglu_backward(g, &a, &b)?;
                Ok(vec![wa.then_some(da), wb.then_some(db)])
            })
        }))
    }

    /// `self * silu(other)`, one primitive at a time. The reference
    /// [`Var::swiglu`] is checked against.
    pub fn swiglu_composed(&self, other: &Self) -> Result<Self> {
        self.mul(&other.silu()?)
    }

    /// Reduce into `[-period/2, period/2)` by subtracting whole multiples of
    /// `period`.
    ///
    /// The multiple subtracted is locally constant, so the derivative is the
    /// identity — which is why this can be one kernel with a trivial adjoint rather
    /// than a rounding, two scalar multiplies and a subtraction.
    pub fn wrap_to(&self, period: f32) -> Result<Self> {
        let value = elemwise::wrap_to(&self.value, period);
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(g.clone())]) })
        }))
    }

    /// Advance the Mamba-3 rotating frame: `wrap(prev + dt * theta)`.
    ///
    /// `dt` is `[batch, heads]`, `theta` and `prev` are `[batch, heads, d_state / 2]`.
    pub fn ssm_angle(dt: &Self, theta: &Self, prev: Option<&Self>) -> Result<Self> {
        let value = fused::ssm_angle(&dt.value, &theta.value, prev.map(|p| &p.value))?;
        let (dtv, thetav) = (dt.value.clone(), theta.value.clone());
        let dt_shape = dt.shape().clone();
        let has_prev = prev.is_some();
        let mut parents: Vec<&Self> = vec![dt, theta];
        if let Some(p) = prev {
            parents.push(p);
        }
        Ok(Self::record(value, &parents, || {
            rule!(|g| {
                let (d_dt, d_theta, d_prev) =
                    fused::ssm_angle_backward(g, &dtv, &thetav, has_prev)?;
                let mut out = vec![Some(reduce_grad_to(&d_dt, &dt_shape)?), Some(d_theta)];
                if let Some(d_prev) = d_prev {
                    out.push(Some(d_prev));
                }
                Ok(out)
            })
        }))
    }

    /// `x0 * s0 + x1 * s1 + x2 * s2`: the Mamba-3 trapezoidal state update, where
    /// each `x` is a `[batch, heads, ...]` state and each `s` one value per head.
    pub fn ssm_state_update(x: [&Self; 3], scales: &Self) -> Result<Self> {
        let value =
            fused::ssm_state_update([&x[0].value, &x[1].value, &x[2].value], &scales.value)?;
        let xv = [x[0].value.clone(), x[1].value.clone(), x[2].value.clone()];
        let sv = scales.value.clone();
        let parents = [x[0], x[1], x[2], scales];
        Ok(Self::record(value, &parents, || {
            rule!(|g| {
                let (dx, ds) = fused::ssm_state_update_backward(g, [&xv[0], &xv[1], &xv[2]], &sv)?;
                let [dx0, dx1, dx2] = dx;
                Ok(vec![Some(dx0), Some(dx1), Some(dx2), Some(ds)])
            })
        }))
    }

    /// The Mamba-3 per-head coefficients `alpha`, `beta` and `g` for one step,
    /// packed as `[3, batch * heads]` — the layout [`Var::ssm_state_update`] reads.
    ///
    /// `reset` is an optional `[batch]` episode-termination flag that zeroes the
    /// two coefficients carrying the previous step's state. It is a plain tensor
    /// rather than a `Var` because a termination flag comes from the environment,
    /// not from the model: nothing differentiates it.
    pub fn ssm_coefficients(
        a_log: &Self,
        dt: &Self,
        lambda: &Self,
        reset: Option<&Tensor<R, E>>,
    ) -> Result<Self> {
        let value = fused::ssm_coefficients(&a_log.value, &dt.value, &lambda.value, reset)?;
        let (a, d, l) = (a_log.value.clone(), dt.value.clone(), lambda.value.clone());
        let saved_reset = reset.cloned();
        let (a_shape, dt_shape, lambda_shape) = (
            a_log.shape().clone(),
            dt.shape().clone(),
            lambda.shape().clone(),
        );
        Ok(Self::record(value, &[a_log, dt, lambda], || {
            rule!(|g| {
                let (da, ddt, dlambda) =
                    fused::ssm_coefficients_backward(g, &a, &d, &l, saved_reset.as_ref())?;
                Ok(vec![
                    Some(reduce_grad_to(&da, &a_shape)?),
                    Some(ddt.reshape(dt_shape.clone())?),
                    Some(dlambda.reshape(lambda_shape.clone())?),
                ])
            })
        }))
    }

    /// The chunked scan's intra-chunk band, `[rows, chunk, chunk]`, from three
    /// `[rows, chunk]` vectors.
    ///
    /// See [`fused::ssd_band`]. Composed of primitives this is seven passes over the
    /// band-sized tensor in the forward pass and rather more in the backward one;
    /// here it is one and one, and the band comes out already in the layout the
    /// batched matmul that consumes it wants.
    pub fn ssd_band(acum: &Self, w: &Self, g: &Self, floor: f32) -> Result<Self> {
        let value = fused::ssd_band(&acum.value, &w.value, &g.value, floor)?;
        let (a, wv, gv) = (acum.value.clone(), w.value.clone(), g.value.clone());
        Ok(Self::record(value, &[acum, w, g], || {
            rule!(|grad| {
                let (da, dw, dg) = fused::ssd_band_backward(grad, &a, &wv, &gv, floor)?;
                Ok(vec![Some(da), Some(dw), Some(dg)])
            })
        }))
    }

    /// `exp(clamp(a - b, floor, 0)) * m`, the scan's decay pattern, fused.
    ///
    /// `b` and `m` are optional and all three operands broadcast. Composed of
    /// primitives this is up to four launches forward, and backward the clamp's
    /// adjoint alone is four more plus three full-size intermediates; fused it is
    /// one launch each way, plus whatever reduction a broadcast operand's
    /// gradient needs. The clamp's gradient convention matches [`Var::clamp`]:
    /// zero at and outside the bounds.
    pub fn exp_decay(a: &Self, b: Option<&Self>, m: Option<&Self>, floor: f32) -> Result<Self> {
        let value = fused::exp_decay(&a.value, b.map(|b| &b.value), m.map(|m| &m.value), floor)?;
        let (av, bv, mv) = (
            a.value.clone(),
            b.map(|b| b.value.clone()),
            m.map(|m| m.value.clone()),
        );
        let a_shape = a.shape().clone();
        let b_shape = b.map(|b| b.shape().clone());
        let m_shape = m.map(|m| m.shape().clone());
        let mut parents: Vec<&Self> = vec![a];
        if let Some(b) = b {
            parents.push(b);
        }
        if let Some(m) = m {
            parents.push(m);
        }
        let has_b = b.is_some();
        Ok(Self::record_with_mask(value, &parents, |want| {
            let wa = want[0];
            let wb = has_b && want[1];
            let wm = m_shape.is_some() && want[1 + has_b as usize];
            rule!(|g| {
                let (da, db, dm) = fused::exp_decay_backward(
                    g,
                    &av,
                    bv.as_ref(),
                    mv.as_ref(),
                    floor,
                    [wa, wb, wm],
                )?;
                let reduced =
                    |full: Option<Tensor<R, E>>, shape: &Shape| -> Result<Option<Tensor<R, E>>> {
                        Ok(match full {
                            Some(full) => Some(reduce_grad_to(&full, shape)?),
                            None => None,
                        })
                    };
                let mut out = vec![reduced(da, &a_shape)?];
                if let Some(bs) = &b_shape {
                    out.push(reduced(db, bs)?);
                }
                if let Some(ms) = &m_shape {
                    out.push(reduced(dm, ms)?);
                }
                Ok(out)
            })
        }))
    }

    /// The decay pattern one primitive at a time. The reference
    /// [`Var::exp_decay`] is checked against.
    pub fn exp_decay_composed(
        a: &Self,
        b: Option<&Self>,
        m: Option<&Self>,
        floor: f32,
    ) -> Result<Self> {
        let diff = match b {
            Some(b) => a.sub(b)?,
            None => a.clone(),
        };
        let decay = diff.clamp(floor, 0.0).exp();
        match m {
            Some(m) => decay.mul(m),
            None => Ok(decay),
        }
    }

    /// The scan's trapezoid weights: `g = lambda * dt` and
    /// `w = g + shift_left((1 - lambda) * dt)`, both `[batch, seq, heads]`, fused.
    ///
    /// One launch producing both tensors, against the seven that the two
    /// multiplies, the scalar subtract and the materialised shift take composed.
    /// A tape node has one output, so the pair rides the same sink arrangement as
    /// [`Var::split`]: the outputs stash their gradients with a sink node created
    /// before them, which the backward walk therefore visits after both, and the
    /// sink runs the one-launch adjoint.
    pub fn trapezoid_weights(lambda: &Self, dt: &Self) -> Result<(Self, Self)> {
        let (g, w) = fused::trapezoid_weights(&lambda.value, &dt.value)?;
        if (lambda.trace.is_none() && dt.trace.is_none()) || !super::grad_mode::is_enabled() {
            return Ok((Self::constant(g), Self::constant(w)));
        }

        let (lam_v, dt_v) = (lambda.value.clone(), dt.value.clone());
        let device = lambda.device().clone();
        let bands: SharedBands<[Option<Tensor<R, E>>; 2]> =
            std::rc::Rc::new(std::cell::RefCell::new([None, None]));
        // As in `split`: an empty token, so the accumulating add in the backward
        // walk allocates and launches nothing — it only puts the sink on the
        // worklist.
        let token = Tensor::empty(Shape::new(vec![0]), &device);

        let sink = {
            let bands = bands.clone();
            Self::record(token.clone(), &[lambda, dt], move || {
                rule!(|_g| {
                    let mut slots = bands.borrow_mut();
                    let (gg, gw) = (slots[0].take(), slots[1].take());
                    let (d_lambda, d_dt) =
                        fused::trapezoid_weights_backward(gg.as_ref(), gw.as_ref(), &lam_v, &dt_v)?;
                    Ok(vec![Some(d_lambda), Some(d_dt)])
                })
            })
        };

        let stash = |value: Tensor<R, E>, slot: usize| {
            let bands = bands.clone();
            let token = token.clone();
            Self::record(value, &[&sink], move || {
                rule!(|g| {
                    bands.borrow_mut()[slot] = Some(g.clone());
                    Ok(vec![Some(token.clone())])
                })
            })
        };
        Ok((stash(g, 0), stash(w, 1)))
    }

    /// `log(1 + exp(x))`, computed stably.
    pub fn softplus(&self) -> Result<Self> {
        let value = fused::softplus(&self.value);
        let x = self.value.clone();
        Ok(Self::record(value, &[self], || {
            rule!(|g| { Ok(vec![Some(fused::softplus_backward(g, &x))]) })
        }))
    }

    /// `log(1 + exp(x))`, one primitive at a time. The reference
    /// [`Var::softplus`] is checked against.
    pub fn softplus_composed(&self) -> Result<Self> {
        // softplus(x) = max(x, 0) + log(1 + exp(-|x|))
        let stable = self.abs().neg().exp().add_scalar(1.0).log();
        self.relu().add(&stable)
    }

    /// `softplus(self + bias)`, fused — the `dt` projection's bias and activation
    /// in one launch. `bias` holds one value per element of `self`'s trailing axis.
    pub fn bias_softplus(&self, bias: &Self) -> Result<Self> {
        let value = fused::bias_softplus(&self.value, &bias.value)?;
        let (x, b) = (self.value.clone(), bias.value.clone());
        let bias_shape = bias.shape().clone();
        Ok(Self::record_with_mask(value, &[self, bias], |want| {
            let (wx, wb) = (want[0], want[1]);
            rule!(|g| {
                let dx = fused::bias_softplus_backward(g, &x, &b)?;
                let db = if wb {
                    Some(reduce_grad_to(&dx, &bias_shape)?)
                } else {
                    None
                };
                Ok(vec![wx.then_some(dx), db])
            })
        }))
    }

    /// `softplus(self + bias)`, one primitive at a time. The reference
    /// [`Var::bias_softplus`] is checked against.
    pub fn bias_softplus_composed(&self, bias: &Self) -> Result<Self> {
        self.add(bias)?.softplus()
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
