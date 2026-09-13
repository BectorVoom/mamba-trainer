//! Distributions on the simplex: [`Dirichlet`], and the one-hot relatives of
//! [`Categorical`].
//!
//! All row kernels, like [`super::categorical`], and for the same reason: the
//! normaliser of a simplex is a sum over the last axis, and a unit that owns the
//! whole row computes it without touching another unit.
//!
//! A Dirichlet draw is `k` independent Gammas divided by their sum. That is one unit
//! doing `k` rejection samples, which is why the sampler here is the only kernel in
//! the crate whose cost per unit grows with the event size rather than the batch —
//! and it is still the right shape, because the normalising sum makes the `k` draws
//! one indivisible piece of work.

// See the notes in `univariate.rs`: `#[cube]` synthesises an undocumented module per
// function, and a branching value has to be initialised before the branch that sets
// it, so its initialiser is dead by construction.
#![allow(missing_docs, unused_assignments)]

use cubecl::prelude::*;

use crate::autograd::Var;
use crate::autograd::ops::reduce_grad_to;
use crate::backend::{Device, FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

use super::categorical::Categorical;
use super::univariate::{SUB_DRAW, binomial_sample, std_gamma_sample};
use super::{Distribution, Support, rng, special};

// ---------------------------------------------------------------------------
// Dirichlet
// ---------------------------------------------------------------------------

/// `ln p(x | α) = Σ (αᵢ − 1) ln xᵢ + ln Γ(Σα) − Σ ln Γ(αᵢ)`.
#[cube(launch_unchecked)]
fn dirichlet_log_prob_kernel<E: Float + CubeElement>(
    concentration: &Array<E>,
    value: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
    param_rows: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let base = ABSOLUTE_POS * classes as usize;
        let pbase = (ABSOLUTE_POS % param_rows as usize) * classes as usize;
        let mut total: f32 = 0.0;
        let mut acc: f32 = 0.0;
        let mut i: u32 = 0;
        while i < classes {
            let a = f32::cast_from(concentration[pbase + i as usize]);
            total += a;
            acc += special::xlogy_f32(a - 1.0f32, f32::cast_from(value[base + i as usize]))
                - special::lgamma_f32(a);
            i += 1u32;
        }
        out[ABSOLUTE_POS] = E::cast_from(acc + special::lgamma_f32(total));
    }
}

/// The adjoint of `dirichlet_log_prob_kernel`, for both operands.
#[cube(launch_unchecked)]
fn dirichlet_log_prob_grad_kernel<E: Float + CubeElement>(
    concentration: &Array<E>,
    value: &Array<E>,
    upstream: &Array<E>,
    grad_conc: &mut Array<E>,
    grad_value: &mut Array<E>,
    rows: u32,
    classes: u32,
    param_rows: u32,
    #[comptime] want_conc: bool,
    #[comptime] want_value: bool,
) {
    if ABSOLUTE_POS < rows as usize {
        let base = ABSOLUTE_POS * classes as usize;
        let pbase = (ABSOLUTE_POS % param_rows as usize) * classes as usize;
        let mut total: f32 = 0.0;
        let mut i: u32 = 0;
        while i < classes {
            total += f32::cast_from(concentration[pbase + i as usize]);
            i += 1u32;
        }
        let psi_total = special::digamma_f32(total);
        let g = f32::cast_from(upstream[ABSOLUTE_POS]);
        let mut j: u32 = 0;
        while j < classes {
            let a = f32::cast_from(concentration[pbase + j as usize]);
            let x = f32::cast_from(value[base + j as usize]);
            if comptime!(want_conc) {
                grad_conc[base + j as usize] =
                    E::cast_from(g * (f32::ln(x) + psi_total - special::digamma_f32(a)));
            }
            if comptime!(want_value) {
                grad_value[base + j as usize] = E::cast_from(g * (a - 1.0f32) / x);
            }
            j += 1u32;
        }
    }
}

/// `H(α) = ln B(α) + (α₀ − k) ψ(α₀) − Σ (αᵢ − 1) ψ(αᵢ)`.
#[cube(launch_unchecked)]
fn dirichlet_entropy_kernel<E: Float + CubeElement>(
    concentration: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let base = ABSOLUTE_POS * classes as usize;
        let mut total: f32 = 0.0;
        let mut acc: f32 = 0.0;
        let mut i: u32 = 0;
        while i < classes {
            let a = f32::cast_from(concentration[base + i as usize]);
            total += a;
            acc += special::lgamma_f32(a) - (a - 1.0f32) * special::digamma_f32(a);
            i += 1u32;
        }
        let k = f32::cast_from(classes);
        out[ABSOLUTE_POS] = E::cast_from(
            acc - special::lgamma_f32(total) + (total - k) * special::digamma_f32(total),
        );
    }
}

/// The adjoint of `dirichlet_entropy_kernel`: `(α₀ − k)ψ'(α₀) − (αᵢ − 1)ψ'(αᵢ)`.
#[cube(launch_unchecked)]
fn dirichlet_entropy_grad_kernel<E: Float + CubeElement>(
    concentration: &Array<E>,
    upstream: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let base = ABSOLUTE_POS * classes as usize;
        let mut total: f32 = 0.0;
        let mut i: u32 = 0;
        while i < classes {
            total += f32::cast_from(concentration[base + i as usize]);
            i += 1u32;
        }
        let k = f32::cast_from(classes);
        let shared = (total - k) * special::trigamma_f32(total);
        let g = f32::cast_from(upstream[ABSOLUTE_POS]);
        let mut j: u32 = 0;
        while j < classes {
            let a = f32::cast_from(concentration[base + j as usize]);
            out[base + j as usize] =
                E::cast_from(g * (shared - (a - 1.0f32) * special::trigamma_f32(a)));
            j += 1u32;
        }
    }
}

/// `k` Gammas divided by their sum.
///
/// Each class draws from its own block of generator streams, so the `k` Gammas of a
/// row are independent of each other and of every other row's — and, because the
/// block is chosen from the class index rather than from a counter, independent of
/// the order the loop happens to run in.
#[cube(launch_unchecked)]
fn dirichlet_sample_kernel<E: Float + CubeElement>(
    concentration: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
    stride_row: u32,
    offset_lo: u32,
    offset_hi: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let base = row * classes as usize;
        let pbase = (row % stride_row as usize) * classes as usize;
        let index = offset_lo + row as u32;
        let mut hi = offset_hi;
        if index < offset_lo {
            hi += 1u32;
        }
        let mut total: f32 = 0.0;
        let mut i: u32 = 0;
        while i < classes {
            let a = f32::cast_from(concentration[pbase + i as usize]);
            let g = std_gamma_sample(a, index, hi, i * SUB_DRAW, key_lo, key_hi, wide);
            out[base + i as usize] = E::cast_from(g);
            total += g;
            i += 1u32;
        }
        let mut j: u32 = 0;
        while j < classes {
            out[base + j as usize] = E::cast_from(f32::cast_from(out[base + j as usize]) / total);
            j += 1u32;
        }
    }
}

/// The mean, variance or mode of a Dirichlet, per class.
#[cube(launch_unchecked)]
fn dirichlet_moment_kernel<E: Float + CubeElement>(
    concentration: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
    #[comptime] which: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let base = ABSOLUTE_POS * classes as usize;
        let mut total: f32 = 0.0;
        let mut i: u32 = 0;
        while i < classes {
            total += f32::cast_from(concentration[base + i as usize]);
            i += 1u32;
        }
        let k = f32::cast_from(classes);
        let mut j: u32 = 0;
        while j < classes {
            let a = f32::cast_from(concentration[base + j as usize]);
            let mut v: f32 = 0.0;
            if comptime!(which == 0) {
                v = a / total;
            } else if comptime!(which == 1) {
                v = a * (total - a) / (total * total * (total + 1.0f32));
            } else {
                v = (a - 1.0f32) / (total - k);
            }
            out[base + j as usize] = E::cast_from(v);
            j += 1u32;
        }
    }
}

/// A distribution over the probability simplex, conjugate to a categorical.
///
/// `concentration` is `[..batch, k]`; a draw is `[..batch, k]` summing to one along
/// the last axis.
pub struct Dirichlet<R: Runtime, E: FloatElem = f32> {
    concentration: Var<R, E>,
    batch: Shape,
    classes: usize,
}

impl<R: Runtime, E: FloatElem> Clone for Dirichlet<R, E> {
    fn clone(&self) -> Self {
        Self {
            concentration: self.concentration.clone(),
            batch: self.batch.clone(),
            classes: self.classes,
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Dirichlet<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Dirichlet({} over {})", self.batch, self.classes)
    }
}

impl<R: Runtime, E: FloatElem> Dirichlet<R, E> {
    /// From a `[..batch, k]` concentration.
    pub fn new(concentration: impl Into<Var<R, E>>) -> Result<Self> {
        let concentration = concentration.into();
        if concentration.rank() == 0 {
            return Err(Error::shape(
                "a Dirichlet needs a trailing class axis, got a scalar".to_string(),
            ));
        }
        let classes = concentration.shape().dim_from_end(0);
        if classes < 2 {
            return Err(Error::shape(format!(
                "a Dirichlet needs at least two classes, got {classes}"
            )));
        }
        let batch = concentration.shape().without(concentration.rank() - 1);
        Ok(Self {
            concentration,
            batch,
            classes,
        })
    }

    /// How many classes the simplex has.
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// The concentration, as given.
    pub fn concentration(&self) -> &Var<R, E> {
        &self.concentration
    }

    fn rows(&self) -> usize {
        self.batch.num_elements()
    }

    fn moment(&self, which: u32) -> Result<Tensor<R, E>> {
        let conc = self.concentration.tensor();
        let out = Tensor::<R, E>::empty(conc.shape().clone(), conc.device());
        let rows = self.rows();
        if rows == 0 {
            return Ok(out);
        }
        let (count, dim) = launch_1d(conc.client(), rows, self.classes * 2);
        unsafe {
            dirichlet_moment_kernel::launch_unchecked::<E, R>(
                conc.client(),
                count,
                dim,
                conc.arg(),
                out.arg(),
                rows as u32,
                self.classes as u32,
                which,
            );
        }
        Ok(out)
    }

    fn draw(&self, rows: usize, shape: Shape, seed: u64) -> Result<Tensor<R, E>> {
        let conc = self.concentration.tensor();
        let out = Tensor::<R, E>::empty(shape, conc.device());
        if rows == 0 {
            return Ok(out);
        }
        let (count, dim) = launch_1d(conc.client(), rows, self.classes * 200);
        unsafe {
            dirichlet_sample_kernel::launch_unchecked::<E, R>(
                conc.client(),
                count,
                dim,
                conc.arg(),
                out.arg(),
                rows as u32,
                self.classes as u32,
                self.rows().max(1) as u32,
                0,
                0,
                seed as u32,
                (seed >> 32) as u32,
                rng::wide_multiply(conc.client()),
            );
        }
        Ok(out)
    }
}

impl<R: Runtime, E: FloatElem> Distribution<R, E> for Dirichlet<R, E> {
    fn batch_shape(&self) -> &Shape {
        &self.batch
    }

    fn event_shape(&self) -> Shape {
        Shape::new(vec![self.classes])
    }

    fn support(&self) -> Support {
        Support::Simplex
    }

    fn has_rsample(&self) -> bool {
        false
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        self.draw(
            self.rows(),
            self.concentration.tensor().shape().clone(),
            seed,
        )
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        let mut dims = vec![n];
        dims.extend_from_slice(self.concentration.shape().dims());
        self.draw(n * self.rows(), Shape::new(dims), seed)
    }

    fn rsample(&self, _seed: u64) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a Dirichlet draw comes from a rejection sampler, whose control flow \
             depends on the concentration; PyTorch reparameterises it by implicit \
             differentiation of an incomplete gamma, which is not implemented here"
                .to_string(),
        ))
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        // The value may prepend a sample axis — scoring the output of `sample_n` is
        // the ordinary thing to want — so it is only required to end in the same
        // class axis and to be a whole number of rows.
        let classes = self.classes;
        let ends_right = value.rank() >= 1 && value.shape().dim_from_end(0) == classes;
        if !ends_right
            || !value
                .tensor()
                .len()
                .is_multiple_of(self.concentration.tensor().len())
        {
            return Err(Error::shape(format!(
                "a Dirichlet over {} cannot score a value of {}",
                self.concentration.shape(),
                value.shape()
            )));
        }
        let conc = self.concentration.tensor();
        let rows = value.tensor().len() / classes;
        let out_shape = value.shape().without(value.rank() - 1);
        let out = Tensor::<R, E>::empty(out_shape, conc.device());
        if rows > 0 {
            let (count, dim) = launch_1d(conc.client(), rows, self.classes * 60);
            unsafe {
                dirichlet_log_prob_kernel::launch_unchecked::<E, R>(
                    conc.client(),
                    count,
                    dim,
                    conc.arg(),
                    value.tensor().arg(),
                    out.arg(),
                    rows as u32,
                    self.classes as u32,
                    self.rows().max(1) as u32,
                );
            }
        }
        let owned = self.clone();
        let seen = value.tensor().clone();
        let value_shape = value.shape().clone();
        Ok(Var::record_with_mask(
            out,
            &[value, &self.concentration],
            |want| {
                let (want_value, want_conc) = (want[0], want[1]);
                Box::new(move |g| {
                    let conc = owned.concentration.tensor();
                    let scratch = Tensor::<R, E>::empty(Shape::new(vec![1]), conc.device());
                    // Both gradients are the value's shape; the concentration's is
                    // reduced back onto its own below, which is what sums a sample
                    // axis away.
                    let make = |on: bool| {
                        on.then(|| Tensor::<R, E>::empty(value_shape.clone(), conc.device()))
                    };
                    let gc = make(want_conc);
                    let gv = make(want_value);
                    let rows = seen.len() / owned.classes;
                    if rows > 0 {
                        let (count, dim) = launch_1d(conc.client(), rows, owned.classes * 60);
                        unsafe {
                            dirichlet_log_prob_grad_kernel::launch_unchecked::<E, R>(
                                conc.client(),
                                count,
                                dim,
                                conc.arg(),
                                seen.arg(),
                                g.arg(),
                                gc.as_ref().unwrap_or(&scratch).arg(),
                                gv.as_ref().unwrap_or(&scratch).arg(),
                                rows as u32,
                                owned.classes as u32,
                                owned.rows().max(1) as u32,
                                want_conc,
                                want_value,
                            );
                        }
                    }
                    Ok(vec![
                        gv.map(|t| reduce_grad_to(&t, &value_shape)).transpose()?,
                        gc.map(|t| reduce_grad_to(&t, owned.concentration.shape()))
                            .transpose()?,
                    ])
                })
            },
        ))
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a Dirichlet is multivariate and has no scalar CDF".to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a Dirichlet is multivariate and has no scalar quantile".to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        let conc = self.concentration.tensor();
        let out = Tensor::<R, E>::empty(self.batch.clone(), conc.device());
        let rows = self.rows();
        if rows > 0 {
            let (count, dim) = launch_1d(conc.client(), rows, self.classes * 80);
            unsafe {
                dirichlet_entropy_kernel::launch_unchecked::<E, R>(
                    conc.client(),
                    count,
                    dim,
                    conc.arg(),
                    out.arg(),
                    rows as u32,
                    self.classes as u32,
                );
            }
        }
        let owned = self.clone();
        Ok(Var::record(out, &[&self.concentration], || {
            Box::new(move |g| {
                let conc = owned.concentration.tensor();
                let grad = Tensor::<R, E>::empty(conc.shape().clone(), conc.device());
                let rows = owned.rows();
                if rows > 0 {
                    let (count, dim) = launch_1d(conc.client(), rows, owned.classes * 80);
                    unsafe {
                        dirichlet_entropy_grad_kernel::launch_unchecked::<E, R>(
                            conc.client(),
                            count,
                            dim,
                            conc.arg(),
                            g.arg(),
                            grad.arg(),
                            rows as u32,
                            owned.classes as u32,
                        );
                    }
                }
                Ok(vec![Some(reduce_grad_to(
                    &grad,
                    owned.concentration.shape(),
                )?)])
            })
        }))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        self.moment(0)
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        self.moment(1)
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        self.moment(2)
    }
}

// ---------------------------------------------------------------------------
// One-hot and relaxed categoricals
// ---------------------------------------------------------------------------

/// [`Categorical`] whose draws are one-hot vectors rather than indices.
///
/// The distribution is the same; only the shape of a sample and of a scored value
/// differ. Kept as a wrapper rather than a family of its own for exactly that
/// reason — everything but the encoding is delegated.
pub struct OneHotCategorical<R: Runtime, E: FloatElem = f32> {
    inner: Categorical<R, E>,
    event: Shape,
}

impl<R: Runtime, E: FloatElem> OneHotCategorical<R, E> {
    /// From unnormalised logits, `[..batch, classes]`.
    pub fn from_logits(logits: impl Into<Var<R, E>>) -> Result<Self> {
        let inner = Categorical::from_logits(logits)?;
        let event = Shape::new(vec![inner.classes()]);
        Ok(Self { inner, event })
    }

    /// From probabilities.
    pub fn from_probs(probs: impl Into<Var<R, E>>) -> Result<Self> {
        let inner = Categorical::from_probs(probs)?;
        let event = Shape::new(vec![inner.classes()]);
        Ok(Self { inner, event })
    }

    /// The categorical underneath.
    pub fn categorical(&self) -> &Categorical<R, E> {
        &self.inner
    }

    /// A draw, one-hot encoded.
    pub fn sample_one_hot(&self, seed: u64) -> Result<Tensor<R, E>> {
        let ids = self.inner.sample_ids(seed)?;
        self.inner.one_hot(&ids)
    }

    /// A draw, one-hot encoded, with a straight-through gradient.
    ///
    /// The value is the hard one-hot vector; the gradient is the soft probability's.
    /// That is the estimator "straight-through" names, and it is what PyTorch's
    /// `OneHotCategoricalStraightThrough` provides.
    pub fn sample_straight_through(&self, seed: u64) -> Result<Var<R, E>> {
        let ids = self.inner.sample_ids(seed)?;
        let hard = self.inner.one_hot(&ids)?;
        let probs = self.probs_var()?;
        // `hard − p` is detached, so the value is `hard` and the derivative is `p`'s.
        let offset = Var::constant(crate::tensor::ops::elemwise::sub(&hard, probs.tensor())?);
        probs.add(&offset)
    }

    /// The probabilities, on the tape.
    fn probs_var(&self) -> Result<Var<R, E>> {
        self.inner.logits().softmax(self.inner.logits().rank() - 1)
    }
}

impl<R: Runtime, E: FloatElem> Distribution<R, E> for OneHotCategorical<R, E> {
    fn batch_shape(&self) -> &Shape {
        self.inner.batch_shape()
    }

    fn event_shape(&self) -> Shape {
        self.event.clone()
    }

    fn support(&self) -> Support {
        Support::Simplex
    }

    fn has_rsample(&self) -> bool {
        false
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        self.sample_one_hot(seed)
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        let ids = self.inner.sample_ids_n(n, seed)?;
        self.inner.one_hot(&ids)
    }

    fn rsample(&self, _seed: u64) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a one-hot draw is discrete; use RelaxedOneHotCategorical, or \
             `sample_straight_through` for the biased estimator"
                .to_string(),
        ))
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        // The value is one-hot, so `Σ vᵢ log pᵢ` picks out the chosen class — and
        // does it differentiably in the value as well as in the logits.
        let log_probs = self
            .inner
            .logits()
            .log_softmax(self.inner.logits().rank() - 1)?;
        value
            .mul(&log_probs)?
            .sum_dim(self.inner.logits().rank() - 1)?
            .squeeze(self.inner.logits().rank() - 1)
    }

    fn cdf(&self, value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        self.inner.cdf(value)
    }

    fn icdf(&self, q: &Var<R, E>) -> Result<Var<R, E>> {
        self.inner.icdf(q)
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        self.inner.entropy()
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        self.inner.probs()
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        let p = self.inner.probs()?;
        let one_minus = crate::tensor::ops::elemwise::rsub_scalar(&p, 1.0);
        crate::tensor::ops::elemwise::mul(&p, &one_minus)
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        let ids = self.inner.mode_ids()?;
        self.inner.one_hot(&ids)
    }
}

/// The Gumbel-softmax: a differentiable stand-in for a one-hot draw.
///
/// A sample is `softmax((logits + Gumbel noise)/τ)`, which is a point *inside* the
/// simplex that approaches a vertex as `τ → 0`. Because the noise does not depend on
/// the logits, the sample has a path derivative — which is the whole point, and the
/// reason this exists where [`OneHotCategorical`] cannot be reparameterised.
pub struct RelaxedOneHotCategorical<R: Runtime, E: FloatElem = f32> {
    logits: Var<R, E>,
    temperature: f32,
    batch: Shape,
    classes: usize,
}

impl<R: Runtime, E: FloatElem> RelaxedOneHotCategorical<R, E> {
    /// From a temperature and unnormalised logits.
    pub fn new(temperature: f32, logits: impl Into<Var<R, E>>) -> Result<Self> {
        // Negated, so a `NaN` temperature is rejected rather than accepted.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(temperature > 0.0) {
            return Err(Error::config(format!(
                "a relaxation needs a positive temperature, got {temperature}"
            )));
        }
        let logits = logits.into();
        if logits.rank() == 0 {
            return Err(Error::shape(
                "a relaxed categorical needs a trailing class axis".to_string(),
            ));
        }
        let classes = logits.shape().dim_from_end(0);
        let batch = logits.shape().without(logits.rank() - 1);
        Ok(Self {
            logits,
            temperature,
            batch,
            classes,
        })
    }

    /// The logits, as given.
    pub fn logits(&self) -> &Var<R, E> {
        &self.logits
    }

    /// The temperature.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Standard Gumbel noise shaped like the logits, as a constant.
    fn noise(&self, seed: u64) -> Result<Tensor<R, E>> {
        let shape = self.logits.shape().clone();
        let device = self.logits.tensor().device();
        gumbel_noise::<R, E>(shape, seed, device)
    }
}

/// `−ln(−ln u)` for every element, drawn from the counter-based generator.
#[cube(launch_unchecked)]
fn gumbel_noise_kernel<E: Float + CubeElement>(
    out: &mut Array<E>,
    n: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) {
    if ABSOLUTE_POS < n as usize {
        let u = rng::unit_open(rng::draw_lane(
            ABSOLUTE_POS as u32,
            0u32,
            0u32,
            key_lo,
            key_hi,
            wide,
        ));
        out[ABSOLUTE_POS] = E::cast_from(-f32::ln(-f32::ln(u)));
    }
}

/// A tensor of standard Gumbel noise.
///
/// Public because the Gumbel-max trick is useful on its own: adding this to a row of
/// logits and taking the argmax is a categorical draw, and taking a softmax instead
/// is [`RelaxedOneHotCategorical`].
pub fn gumbel_noise<R: Runtime, E: FloatElem>(
    shape: impl Into<Shape>,
    seed: u64,
    device: &Device<R>,
) -> Result<Tensor<R, E>> {
    let out = Tensor::<R, E>::empty(shape, device);
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let (count, dim) = launch_1d(device.client(), n, 40);
    unsafe {
        gumbel_noise_kernel::launch_unchecked::<E, R>(
            device.client(),
            count,
            dim,
            out.arg(),
            n as u32,
            seed as u32,
            (seed >> 32) as u32,
            rng::wide_multiply(device.client()),
        );
    }
    Ok(out)
}

impl<R: Runtime, E: FloatElem> Distribution<R, E> for RelaxedOneHotCategorical<R, E> {
    fn batch_shape(&self) -> &Shape {
        &self.batch
    }

    fn event_shape(&self) -> Shape {
        Shape::new(vec![self.classes])
    }

    fn support(&self) -> Support {
        Support::Simplex
    }

    fn has_rsample(&self) -> bool {
        true
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        Ok(self.rsample(seed)?.into_tensor())
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        // One noise launch for the whole `[n, ..batch, k]` block, then one softmax.
        let mut dims = vec![n];
        dims.extend_from_slice(self.logits.shape().dims());
        let shape = Shape::new(dims);
        let noise = gumbel_noise::<R, E>(shape.clone(), seed, self.logits.tensor().device())?;
        let wide = self.logits.unsqueeze(0)?.expand(shape)?;
        let axis = wide.rank() - 1;
        Ok(wide
            .add(&Var::constant(noise))?
            .mul_scalar(1.0 / self.temperature)
            .softmax(axis)?
            .into_tensor())
    }

    fn rsample(&self, seed: u64) -> Result<Var<R, E>> {
        let axis = self.logits.rank() - 1;
        let noisy = self.logits.add(&Var::constant(self.noise(seed)?))?;
        noisy.mul_scalar(1.0 / self.temperature).softmax(axis)
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        // PyTorch's `ExpRelaxedCategorical` density, transformed back from the log
        // simplex: `ln Γ(k) + (k−1) ln τ + Σ(lᵢ − τ ln yᵢ) − k·lse(l − τ ln y)`,
        // minus `Σ ln yᵢ` for the change of variables out of the log.
        let axis = self.logits.rank() - 1;
        let k = self.classes as f32;
        let log_value = value.log();
        let score = self.logits.sub(&log_value.mul_scalar(self.temperature))?;
        let normalised = score.sub(&logsumexp_keep(&score, axis)?)?;
        let scale = special::host::lgamma_f32(k) + (k - 1.0) * self.temperature.ln();
        normalised
            .sum_dim(axis)?
            .squeeze(axis)?
            .sub(&log_value.sum_dim(axis)?.squeeze(axis)?)
            .map(|v| v.add_scalar(scale))
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a relaxed categorical is multivariate and has no scalar CDF".to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a relaxed categorical is multivariate and has no scalar quantile".to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a relaxed categorical has no closed-form entropy".to_string(),
        ))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a relaxed categorical has no closed-form mean".to_string(),
        ))
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a relaxed categorical has no closed-form variance".to_string(),
        ))
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a relaxed categorical has no closed-form mode".to_string(),
        ))
    }
}

/// `logsumexp` along `axis`, keeping the axis at extent one so it broadcasts back.
fn logsumexp_keep<R: Runtime, E: FloatElem>(x: &Var<R, E>, axis: usize) -> Result<Var<R, E>> {
    let top = x.max_dim(axis)?;
    x.sub(&top)?.exp().sum_dim(axis)?.log().add(&top)
}

/// Successes of each class in `total_count` independent categorical trials.
///
/// Sampled by the chain of conditional binomials: the first class's count is
/// `Binomial(n, p₁)`, the second's is `Binomial(n − c₁, p₂/(1 − p₁))`, and so on.
/// That is exact — it is the multinomial's own factorisation — and it costs `k`
/// binomial draws rather than `n` categorical ones, which matters when `n` is large.
pub struct Multinomial<R: Runtime, E: FloatElem = f32> {
    inner: Categorical<R, E>,
    total_count: usize,
}

impl<R: Runtime, E: FloatElem> Multinomial<R, E> {
    /// From a trial count and unnormalised logits.
    pub fn from_logits(total_count: usize, logits: impl Into<Var<R, E>>) -> Result<Self> {
        Ok(Self {
            inner: Categorical::from_logits(logits)?,
            total_count,
        })
    }

    /// How many trials each draw runs.
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// The categorical each trial draws from.
    pub fn categorical(&self) -> &Categorical<R, E> {
        &self.inner
    }
}

/// `k` conditional binomials along a row.
#[cube(launch_unchecked)]
fn multinomial_sample_kernel<E: Float + CubeElement>(
    logits: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
    total: f32,
    offset_lo: u32,
    offset_hi: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let base = row * classes as usize;
        let index = offset_lo + row as u32;
        let mut hi = offset_hi;
        if index < offset_lo {
            hi += 1u32;
        }
        // The row's normaliser, so each conditional probability is a ratio of
        // exponentials that never leaves the log domain until it has to.
        let mut top = f32::cast_from(logits[base]);
        let mut i: u32 = 1;
        while i < classes {
            let v = f32::cast_from(logits[base + i as usize]);
            let mut next = top;
            if v > top {
                next = v;
            }
            top = next;
            i += 1u32;
        }
        let mut remaining_mass: f32 = 0.0;
        let mut j: u32 = 0;
        while j < classes {
            remaining_mass += f32::exp(f32::cast_from(logits[base + j as usize]) - top);
            j += 1u32;
        }

        let mut left = total;
        let mut c: u32 = 0;
        while c < classes {
            let w = f32::exp(f32::cast_from(logits[base + c as usize]) - top);
            let mut drawn: f32 = 0.0;
            if left > 0.0f32 {
                if c + 1u32 == classes {
                    drawn = left;
                } else {
                    // `p = w / remaining_mass`, as a logit so the binomial sampler
                    // sees the parameterisation it wants.
                    let logit = f32::ln(w) - f32::ln(remaining_mass - w);
                    drawn = binomial_sample(
                        left,
                        logit,
                        index,
                        hi,
                        c * SUB_DRAW * 2u32,
                        key_lo,
                        key_hi,
                        wide,
                    );
                }
            }
            out[base + c as usize] = E::cast_from(drawn);
            left -= drawn;
            remaining_mass -= w;
            c += 1u32;
        }
    }
}

/// `ln n! − Σ ln xᵢ! + Σ xᵢ ln pᵢ`.
#[cube(launch_unchecked)]
fn multinomial_log_prob_kernel<E: Float + CubeElement>(
    logits: &Array<E>,
    value: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    classes: u32,
    total: f32,
) {
    if ABSOLUTE_POS < rows as usize {
        let base = ABSOLUTE_POS * classes as usize;
        let mut top = f32::cast_from(logits[base]);
        let mut i: u32 = 1;
        while i < classes {
            let v = f32::cast_from(logits[base + i as usize]);
            let mut next = top;
            if v > top {
                next = v;
            }
            top = next;
            i += 1u32;
        }
        let mut sum: f32 = 0.0;
        let mut j: u32 = 0;
        while j < classes {
            sum += f32::exp(f32::cast_from(logits[base + j as usize]) - top);
            j += 1u32;
        }
        let lse = top + f32::ln(sum);
        let mut acc = special::lgamma_f32(total + 1.0f32);
        let mut c: u32 = 0;
        while c < classes {
            let x = f32::cast_from(value[base + c as usize]);
            acc += x * (f32::cast_from(logits[base + c as usize]) - lse)
                - special::lgamma_f32(x + 1.0f32);
            c += 1u32;
        }
        out[ABSOLUTE_POS] = E::cast_from(acc);
    }
}

impl<R: Runtime, E: FloatElem> Distribution<R, E> for Multinomial<R, E> {
    fn batch_shape(&self) -> &Shape {
        self.inner.batch_shape()
    }

    fn event_shape(&self) -> Shape {
        Shape::new(vec![self.inner.classes()])
    }

    fn support(&self) -> Support {
        Support::BoundedCounting
    }

    fn has_rsample(&self) -> bool {
        false
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        let logits = self.inner.logits().tensor();
        let out = Tensor::<R, E>::empty(logits.shape().clone(), logits.device());
        let rows = self.inner.batch_shape().num_elements();
        if rows == 0 {
            return Ok(out);
        }
        let (count, dim) = launch_1d(logits.client(), rows, self.inner.classes() * 200);
        unsafe {
            multinomial_sample_kernel::launch_unchecked::<E, R>(
                logits.client(),
                count,
                dim,
                logits.arg(),
                out.arg(),
                rows as u32,
                self.inner.classes() as u32,
                self.total_count as f32,
                0,
                0,
                seed as u32,
                (seed >> 32) as u32,
                rng::wide_multiply(logits.client()),
            );
        }
        Ok(out)
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        let mut parts = Vec::with_capacity(n);
        for j in 0..n {
            parts.push(
                self.sample(
                    seed.wrapping_add(j as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )?,
            );
        }
        let mut dims = vec![n];
        dims.extend_from_slice(self.inner.logits().shape().dims());
        crate::tensor::ops::movement::cat(&parts, 0)?.reshape(Shape::new(dims))
    }

    fn rsample(&self, _seed: u64) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "multinomial counts are discrete".to_string(),
        ))
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        let logits = self.inner.logits().tensor();
        let out = Tensor::<R, E>::empty(self.inner.batch_shape().clone(), logits.device());
        let rows = self.inner.batch_shape().num_elements();
        if rows > 0 {
            let (count, dim) = launch_1d(logits.client(), rows, self.inner.classes() * 60);
            unsafe {
                multinomial_log_prob_kernel::launch_unchecked::<E, R>(
                    logits.client(),
                    count,
                    dim,
                    logits.arg(),
                    value.tensor().arg(),
                    out.arg(),
                    rows as u32,
                    self.inner.classes() as u32,
                    self.total_count as f32,
                );
            }
        }
        Ok(Var::constant(out))
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a multinomial is multivariate and has no scalar CDF".to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a multinomial is multivariate and has no scalar quantile".to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a multinomial has no closed-form entropy".to_string(),
        ))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        Ok(crate::tensor::ops::elemwise::mul_scalar(
            &self.inner.probs()?,
            self.total_count as f32,
        ))
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        let p = self.inner.probs()?;
        let one_minus = crate::tensor::ops::elemwise::rsub_scalar(&p, 1.0);
        Ok(crate::tensor::ops::elemwise::mul_scalar(
            &crate::tensor::ops::elemwise::mul(&p, &one_minus)?,
            self.total_count as f32,
        ))
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a multinomial's mode has no closed form".to_string(),
        ))
    }
}
