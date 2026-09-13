//! Distributions over the last axis of a tensor: [`Categorical`] and its relatives.
//!
//! Everything here is a **row kernel**: one unit owns one row of `k` classes and
//! walks it, because that is the only shape in which a softmax's normaliser can be
//! computed without either a second pass over global memory or a cross-unit
//! reduction. `k` is an action space, a vocabulary or a simplex — tens to tens of
//! thousands — and the row fits in cache either way.
//!
//! # What a policy gradient actually needs
//!
//! Three of these operations are the inner loop of PPO, and all three are fused:
//!
//! * [`Categorical::sample_with_log_prob`] draws an action **and** scores it in one
//!   pass over the logits. Drawing and then scoring reads the row twice and
//!   normalises it twice; there is no reason to.
//! * [`Categorical::log_prob`] is one launch, and its adjoint — `p − onehot(a)` — is
//!   one more. Composed out of a `log_softmax` and a `gather`, the same thing is
//!   five launches forward and six back.
//! * [`Categorical::entropy`] likewise, with the adjoint `−pᵢ(lᵢ − lse + H)`.
//!
//! # Numerics
//!
//! Every row subtracts its own maximum before exponentiating, so a logit of `+800`
//! is as safe as one of `0`. The log-density is `l_a − lse` rather than
//! `ln(softmax(l)_a)`, which is the same number computed without ever forming a
//! probability that could round to zero.

// See the note in `univariate.rs`: `#[cube] pub fn` synthesises an undocumented
// module and there is no way to document it.
#![allow(missing_docs)]

use cubecl::prelude::*;

use crate::autograd::Var;
use crate::autograd::ops::reduce_grad_to;
use crate::backend::{Device, FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::shape::Shape;

use super::rng;
use super::{Distribution, Support};

/// The maximum of a row of logits, scaled by an inverse temperature.
///
/// Both a greedy choice and a stable exponential need it, and it costs one pass
/// either way.
#[cube]
fn row_max<E: Float + CubeElement>(logits: &Array<E>, base: usize, classes: u32, scale: f32) -> f32 {
    let mut top = f32::cast_from(logits[base]) * scale;
    let mut i: u32 = 1;
    while i < classes {
        let v = f32::cast_from(logits[base + i as usize]) * scale;
        let mut next = top;
        if v > top {
            next = v;
        }
        top = next;
        i += 1u32;
    }
    top
}

/// `Σ exp(lᵢ·scale − top)` over a row.
#[cube]
fn row_sumexp<E: Float + CubeElement>(
    logits: &Array<E>,
    base: usize,
    classes: u32,
    scale: f32,
    top: f32,
) -> f32 {
    let mut total: f32 = 0.0;
    let mut i: u32 = 0;
    while i < classes {
        total += f32::exp(f32::cast_from(logits[base + i as usize]) * scale - top);
        i += 1u32;
    }
    total
}

// ---------------------------------------------------------------------------
// Categorical
// ---------------------------------------------------------------------------

/// Draw an action from each row, and optionally score it in the same pass.
///
/// Three passes over the row — maximum, normaliser, inverse CDF — rather than one,
/// which is the right trade at this width. Keeping the exponentials would cost `k`
/// registers or a scratch buffer; re-reading them costs an L1 hit each and buys a
/// kernel whose register use does not depend on the action space at all.
///
/// The inverse CDF counts the buckets whose prefix has not yet passed the target,
/// which *is* the index of the first one that does. No early exit, so every unit in
/// a cube walks the same trip count and a warp does not diverge.
#[cube(launch_unchecked)]
fn categorical_sample_kernel<E: Float + CubeElement>(
    logits: &Array<E>,
    actions: &mut Array<u32>,
    logprobs: &mut Array<E>,
    rows: u32,
    param_rows: u32,
    classes: u32,
    inv_temperature: f32,
    offset_lo: u32,
    offset_hi: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] stochastic: bool,
    #[comptime] want_logprob: bool,
    #[comptime] wide: bool,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        // The parameters tile: row `j·batch + i` of a `[samples, batch]` draw reads
        // the logits of batch element `i`.
        let base = (row % param_rows as usize) * classes as usize;
        let top = row_max::<E>(logits, base, classes, inv_temperature);
        let total = row_sumexp::<E>(logits, base, classes, inv_temperature, top);

        let mut chosen: u32 = 0;
        if comptime!(stochastic) {
            let index = offset_lo + row as u32;
            let mut hi = offset_hi;
            if index < offset_lo {
                hi += 1u32;
            }
            let u = rng::unit_open(rng::draw_lane(index, hi, 0u32, key_lo, key_hi, wide));
            // Scale the target by the normaliser rather than dividing every weight
            // by it: one multiply instead of `k` divides.
            let target = u * total;
            let mut acc: f32 = 0.0;
            let mut i: u32 = 0;
            while i < classes {
                acc += f32::exp(f32::cast_from(logits[base + i as usize]) * inv_temperature - top);
                let mut next = chosen;
                if acc <= target {
                    next = i + 1u32;
                }
                chosen = next;
                i += 1u32;
            }
            // The prefix sums in a different order from the normaliser, so it can
            // fall a rounding short of a target drawn just below one.
            let mut clamped = chosen;
            if chosen >= classes {
                clamped = classes - 1u32;
            }
            chosen = clamped;
        } else {
            let mut best: u32 = 0;
            let mut i: u32 = 1;
            while i < classes {
                let mut next = best;
                if f32::cast_from(logits[base + i as usize]) * inv_temperature
                    > f32::cast_from(logits[base + best as usize]) * inv_temperature
                {
                    next = i;
                }
                best = next;
                i += 1u32;
            }
            chosen = best;
        }

        actions[row] = chosen;
        if comptime!(want_logprob) {
            // The log-density of the *tempered* distribution — the one actually
            // drawn from — because that is what an importance ratio measures
            // against.
            logprobs[row] = E::cast_from(
                f32::cast_from(logits[base + chosen as usize]) * inv_temperature
                    - top
                    - f32::ln(total),
            );
        }
    }
}

/// `log softmax(l)_a` for one action per row.
#[cube(launch_unchecked)]
fn categorical_log_prob_kernel<E: Float + CubeElement>(
    logits: &Array<E>,
    actions: &Array<u32>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let base = row * classes as usize;
        let top = row_max::<E>(logits, base, classes, 1.0f32);
        let total = row_sumexp::<E>(logits, base, classes, 1.0f32, top);
        let a = actions[row];
        out[row] = E::cast_from(f32::cast_from(logits[base + a as usize]) - top - f32::ln(total));
    }
}

/// `−Σ pᵢ ln pᵢ` for one row.
///
/// A term whose probability is exactly `0` — a masked action (see
/// [`crate::tensor::ops::elemwise::mask_logits`]), or a logit so far below the
/// row's that `exp` underflows — contributes exactly `0`, which is the limit of
/// `p·ln p` as `p → 0`. It is short-circuited on `p == 0` rather than trusted to
/// the arithmetic, so a logit that is `-inf` (a caller's own, not the mask's) is
/// `0` too rather than `0 * -inf = NaN`. The test is on `p`, not on the logit
/// being `-inf`: an infinity literal does not compile on WGSL.
#[cube(launch_unchecked)]
fn categorical_entropy_kernel<E: Float + CubeElement>(
    logits: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let base = row * classes as usize;
        let top = row_max::<E>(logits, base, classes, 1.0f32);
        let total = row_sumexp::<E>(logits, base, classes, 1.0f32, top);
        let lse = top + f32::ln(total);
        let mut acc: f32 = 0.0;
        let mut i: u32 = 0;
        while i < classes {
            let shifted = f32::cast_from(logits[base + i as usize]) - lse;
            let p = f32::exp(shifted);
            let mut term = p * shifted;
            if p == 0.0 {
                term = 0.0;
            }
            acc -= term;
            i += 1u32;
        }
        out[row] = E::cast_from(acc);
    }
}

/// `softmax(l)`, and with it the adjoint of both of the above.
///
/// `which` picks the shape of the answer: the probabilities themselves, the
/// log-density's adjoint `g·(p − onehot(a))`, or the entropy's
/// `−g·pᵢ(lᵢ − lse + H)`. All three want the row's normaliser, so all three are the
/// same two passes with a different third.
#[cube(launch_unchecked)]
fn categorical_rowgrad_kernel<E: Float + CubeElement>(
    logits: &Array<E>,
    actions: &Array<u32>,
    upstream: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
    #[comptime] which: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let base = row * classes as usize;
        let top = row_max::<E>(logits, base, classes, 1.0f32);
        let total = row_sumexp::<E>(logits, base, classes, 1.0f32, top);
        let lse = top + f32::ln(total);

        if comptime!(which == 0) {
            let mut i: u32 = 0;
            while i < classes {
                out[base + i as usize] =
                    E::cast_from(f32::exp(f32::cast_from(logits[base + i as usize]) - lse));
                i += 1u32;
            }
        } else if comptime!(which == 1) {
            let g = f32::cast_from(upstream[row]);
            let a = actions[row];
            let mut i: u32 = 0;
            while i < classes {
                let p = f32::exp(f32::cast_from(logits[base + i as usize]) - lse);
                let mut hit: f32 = 0.0;
                if i == a {
                    hit = 1.0f32;
                }
                out[base + i as usize] = E::cast_from(g * (hit - p));
                i += 1u32;
            }
        } else {
            // The entropy first, then its adjoint; the second pass needs it, and
            // recomputing it here is cheaper than reading it back from a kernel that
            // already produced it. Both passes zero a term whose probability is
            // exactly `0` (see `categorical_entropy_kernel`), which is what a masked
            // action must contribute, and which `0 * -inf` would not be.
            let mut entropy: f32 = 0.0;
            let mut i: u32 = 0;
            while i < classes {
                let shifted = f32::cast_from(logits[base + i as usize]) - lse;
                let p = f32::exp(shifted);
                let mut term = p * shifted;
                if p == 0.0 {
                    term = 0.0;
                }
                entropy -= term;
                i += 1u32;
            }
            let g = f32::cast_from(upstream[row]);
            let mut j: u32 = 0;
            while j < classes {
                let shifted = f32::cast_from(logits[base + j as usize]) - lse;
                let p = f32::exp(shifted);
                let mut grad = -g * p * (shifted + entropy);
                if p == 0.0 {
                    grad = 0.0;
                }
                out[base + j as usize] = E::cast_from(grad);
                j += 1u32;
            }
        }
    }
}

/// A distribution over `k` classes, given by unnormalised logits.
///
/// The last axis of `logits` is the classes; every axis before it is the batch. A
/// draw is an index, not a one-hot vector — see [`Categorical::one_hot`] if you want
/// the other shape.
pub struct Categorical<R: Runtime, E: FloatElem = f32> {
    logits: Var<R, E>,
    batch: Shape,
    classes: usize,
}

impl<R: Runtime, E: FloatElem> Clone for Categorical<R, E> {
    fn clone(&self) -> Self {
        Self {
            logits: self.logits.clone(),
            batch: self.batch.clone(),
            classes: self.classes,
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Categorical<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Categorical({} over {})", self.batch, self.classes)
    }
}

impl<R: Runtime, E: FloatElem> Categorical<R, E> {
    /// From unnormalised logits, `[..batch, classes]`.
    ///
    /// The logits are kept as given rather than normalised at construction, as
    /// PyTorch normalises them: the kernels subtract a row maximum anyway, so
    /// normalising first would cost a pass and change nothing.
    pub fn from_logits(logits: impl Into<Var<R, E>>) -> Result<Self> {
        let logits = logits.into();
        if logits.rank() == 0 {
            return Err(Error::shape(
                "a categorical needs a trailing class axis, got a scalar".to_string(),
            ));
        }
        let classes = logits.shape().dim_from_end(0);
        if classes == 0 {
            return Err(Error::shape(
                "a categorical over an empty class axis has no support".to_string(),
            ));
        }
        let batch = logits.shape().without(logits.rank() - 1);
        Ok(Self {
            logits,
            batch,
            classes,
        })
    }

    /// From probabilities, which are converted to logits once, here.
    pub fn from_probs(probs: impl Into<Var<R, E>>) -> Result<Self> {
        let probs = probs.into();
        Self::from_logits(probs.log())
    }

    /// How many classes each row has.
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// The logits, as given.
    pub fn logits(&self) -> &Var<R, E> {
        &self.logits
    }

    /// The device the logits live on.
    pub fn device(&self) -> &Device<R> {
        self.logits.tensor().device()
    }

    fn rows(&self) -> usize {
        self.batch.num_elements()
    }

    /// The normalised probabilities, `[..batch, classes]`.
    pub fn probs(&self) -> Result<Tensor<R, E>> {
        self.rowgrad(None, None, 0)
    }

    /// Launch `categorical_rowgrad_kernel`.
    fn rowgrad(
        &self,
        actions: Option<&IdTensor<R>>,
        upstream: Option<&Tensor<R, E>>,
        which: u32,
    ) -> Result<Tensor<R, E>> {
        let logits = self.logits.tensor();
        let out = Tensor::<R, E>::empty(logits.shape().clone(), logits.device());
        let rows = self.rows();
        if rows == 0 {
            return Ok(out);
        }
        let ids_scratch;
        let ids = match actions {
            Some(a) => a,
            None => {
                ids_scratch = IdTensor::empty(Shape::new(vec![1]), logits.device());
                &ids_scratch
            }
        };
        let up_scratch;
        let up = match upstream {
            Some(u) => u,
            None => {
                up_scratch = Tensor::<R, E>::empty(Shape::new(vec![1]), logits.device());
                &up_scratch
            }
        };
        let (count, dim) = launch_1d(logits.client(), rows, self.classes * 3);
        unsafe {
            categorical_rowgrad_kernel::launch_unchecked::<E, R>(
                logits.client(),
                count,
                dim,
                logits.arg(),
                ids.arg(),
                up.arg(),
                out.arg(),
                rows as u32,
                self.classes as u32,
                which,
            );
        }
        Ok(out)
    }

    /// Draw one action per row and score it, in a single pass over the logits.
    ///
    /// A `temperature` of zero makes the choice greedy, in which case the draw is
    /// deterministic and `seed` is unused. The log-probability returned is of the
    /// *tempered* distribution, which is the one the action came from.
    pub fn sample_with_log_prob(
        &self,
        temperature: f32,
        seed: u64,
    ) -> Result<(IdTensor<R>, Tensor<R, E>)> {
        self.draw(temperature, seed, true)
    }

    /// Draw one action per row.
    pub fn sample_ids(&self, seed: u64) -> Result<IdTensor<R>> {
        Ok(self.draw(1.0, seed, false)?.0)
    }

    /// The most likely class in each row.
    pub fn mode_ids(&self) -> Result<IdTensor<R>> {
        Ok(self.draw(0.0, 0, false)?.0)
    }

    fn draw(
        &self,
        temperature: f32,
        seed: u64,
        want_logprob: bool,
    ) -> Result<(IdTensor<R>, Tensor<R, E>)> {
        self.draw_shaped(self.batch.clone(), temperature, seed, want_logprob)
    }

    /// The general draw: `shape` may prepend a sample axis to the batch, in which
    /// case one launch covers all of it and the parameters tile.
    fn draw_shaped(
        &self,
        shape: Shape,
        temperature: f32,
        seed: u64,
        want_logprob: bool,
    ) -> Result<(IdTensor<R>, Tensor<R, E>)> {
        if temperature < 0.0 {
            return Err(Error::config(format!(
                "temperature must not be negative, got {temperature}"
            )));
        }
        let logits = self.logits.tensor();
        let rows = shape.num_elements();
        let actions = IdTensor::empty(shape.clone(), logits.device());
        let logprobs = Tensor::<R, E>::empty(
            if want_logprob {
                shape
            } else {
                Shape::new(vec![1])
            },
            logits.device(),
        );
        if rows == 0 {
            return Ok((actions, logprobs));
        }
        // Zero is `argmax`, not a division by zero.
        let greedy = temperature == 0.0;
        let inv_temperature = if greedy { 1.0 } else { 1.0 / temperature };
        let (count, dim) = launch_1d(logits.client(), rows, self.classes * 3);
        unsafe {
            categorical_sample_kernel::launch_unchecked::<E, R>(
                logits.client(),
                count,
                dim,
                logits.arg(),
                actions.arg(),
                logprobs.arg(),
                rows as u32,
                self.rows().max(1) as u32,
                self.classes as u32,
                inv_temperature,
                0,
                0,
                seed as u32,
                (seed >> 32) as u32,
                !greedy,
                want_logprob,
                rng::wide_multiply(logits.client()),
            );
        }
        Ok((actions, logprobs))
    }

    /// `n` independent draws in one launch, shaped `[n, ..batch]`.
    pub fn sample_ids_n(&self, n: usize, seed: u64) -> Result<IdTensor<R>> {
        let mut dims = vec![n];
        dims.extend_from_slice(self.batch.dims());
        Ok(self.draw_shaped(Shape::new(dims), 1.0, seed, false)?.0)
    }

    /// The log-probability of one action per row.
    pub fn log_prob_ids(&self, actions: &IdTensor<R>) -> Result<Var<R, E>> {
        if actions.shape() != &self.batch {
            return Err(Error::shape(format!(
                "expected {} actions for a {self:?}, got {}",
                self.batch,
                actions.shape()
            )));
        }
        let logits = self.logits.tensor();
        let out = Tensor::<R, E>::empty(self.batch.clone(), logits.device());
        let rows = self.rows();
        if rows > 0 {
            let (count, dim) = launch_1d(logits.client(), rows, self.classes * 2);
            unsafe {
                categorical_log_prob_kernel::launch_unchecked::<E, R>(
                    logits.client(),
                    count,
                    dim,
                    logits.arg(),
                    actions.arg(),
                    out.arg(),
                    rows as u32,
                    self.classes as u32,
                );
            }
        }
        let owned = self.clone();
        let ids = actions.clone();
        Ok(Var::record(out, &[&self.logits], || {
            Box::new(move |g| {
                let grad = owned.rowgrad(Some(&ids), Some(g), 1)?;
                Ok(vec![Some(reduce_grad_to(&grad, owned.logits.shape())?)])
            })
        }))
    }

    /// A one-hot encoding of a draw, `[..batch, classes]`.
    pub fn one_hot(&self, actions: &IdTensor<R>) -> Result<Tensor<R, E>> {
        crate::tensor::ops::index::one_hot(actions, self.classes)
    }
}

impl<R: Runtime, E: FloatElem> Distribution<R, E> for Categorical<R, E> {
    fn batch_shape(&self) -> &Shape {
        &self.batch
    }

    fn event_shape(&self) -> Shape {
        Shape::scalar()
    }

    fn support(&self) -> Support {
        Support::Categories
    }

    fn has_rsample(&self) -> bool {
        false
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        // The index as a float, which is the shape the generic trait speaks in;
        // `sample_ids` is the one a gather wants.
        let ids = self.sample_ids(seed)?;
        Ok(crate::tensor::ops::index::ids_to_float(&ids))
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        Ok(crate::tensor::ops::index::ids_to_float(
            &self.sample_ids_n(n, seed)?,
        ))
    }

    fn rsample(&self, _seed: u64) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a categorical draw is an index, which no reparameterisation makes \
             differentiable; use RelaxedOneHotCategorical for a differentiable \
             relaxation of it"
                .to_string(),
        ))
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        let ids = crate::tensor::ops::index::float_to_ids(value.tensor());
        self.log_prob_ids(&ids)
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a categorical has no CDF: its classes are unordered".to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a categorical has no quantile: its classes are unordered".to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        let logits = self.logits.tensor();
        let out = Tensor::<R, E>::empty(self.batch.clone(), logits.device());
        let rows = self.rows();
        if rows > 0 {
            let (count, dim) = launch_1d(logits.client(), rows, self.classes * 3);
            unsafe {
                categorical_entropy_kernel::launch_unchecked::<E, R>(
                    logits.client(),
                    count,
                    dim,
                    logits.arg(),
                    out.arg(),
                    rows as u32,
                    self.classes as u32,
                );
            }
        }
        let owned = self.clone();
        Ok(Var::record(out, &[&self.logits], || {
            Box::new(move |g| {
                let grad = owned.rowgrad(None, Some(g), 2)?;
                Ok(vec![Some(reduce_grad_to(&grad, owned.logits.shape())?)])
            })
        }))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a categorical has no mean: its classes are unordered".to_string(),
        ))
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a categorical has no variance: its classes are unordered".to_string(),
        ))
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        Ok(crate::tensor::ops::index::ids_to_float(&self.mode_ids()?))
    }
}
