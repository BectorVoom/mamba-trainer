//! [`Univariate`]: the twenty-six scalar families, behind one type.
//!
//! One type rather than twenty-six, because twenty-six newtypes over the same three
//! parameter slots would differ only in their constructor's argument names — and
//! that difference is worth keeping, so it lives in the constructors. Every family
//! has its own, named after PyTorch's class and taking PyTorch's parameters in
//! PyTorch's order.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::autograd::ops::reduce_grad_to;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

use super::univariate::{self, Kind, Wants, moment, op};
use super::{Distribution, Param, Support};

/// A scalar distribution: a [`Kind`] plus up to three parameter tensors.
///
/// The parameters are [`Var`]s, so a distribution built from a network's output is
/// differentiable end to end — which is what a policy gradient is. A distribution
/// built from plain tensors carries no tape and costs nothing extra.
pub struct Univariate<R: Runtime, E: FloatElem = f32> {
    kind: Kind,
    /// The three slots. Unused slots alias the first one *as a constant*, which
    /// keeps them off the tape and out of the gradient without allocating anything.
    slots: [Var<R, E>; 3],
    batch: Shape,
}

impl<R: Runtime, E: FloatElem> Clone for Univariate<R, E> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            slots: self.slots.clone(),
            batch: self.batch.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Univariate<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}({})", self.kind, self.batch)
    }
}

impl<R: Runtime, E: FloatElem> Univariate<R, E> {
    /// Build a distribution of `kind` from parameters given in the order [`Kind`]'s
    /// table lists them.
    ///
    /// Prefer the named constructors; this is what they call, and what a caller who
    /// is choosing a family at run time wants.
    pub fn new(kind: Kind, params: Vec<Param<R, E>>, device: &Device<R>) -> Result<Self> {
        if params.len() != kind.arity() {
            return Err(Error::config(format!(
                "{kind:?} takes {} parameters, got {}",
                kind.arity(),
                params.len()
            )));
        }
        let given: Vec<Var<R, E>> = params.into_iter().map(|p| p.into_var(device)).collect();

        // The batch shape is the broadcast of every parameter that is not a single
        // value; a parameter that *is* one stays one and is bound with a zero
        // stride, rather than being expanded into a tensor of copies.
        let mut batch = Shape::scalar();
        for var in &given {
            if var.tensor().len() != 1 {
                batch = Shape::broadcast(&batch, var.shape())?;
            }
        }

        let mut slots: Vec<Var<R, E>> = Vec::with_capacity(3);
        for var in given {
            if var.tensor().len() == 1 || var.shape() == &batch {
                slots.push(var);
            } else {
                // A `[b, 1]` scale against a `[b, d]` mean: expand once, here, so the
                // kernels only ever see a stride of zero or one.
                slots.push(var.expand(batch.clone())?);
            }
        }
        // Unused slots point at the first parameter's buffer as an untracked
        // constant. The kernel's `#[comptime]` arm never reads them, and being
        // untracked keeps them out of the backward pass.
        let filler = Var::constant(slots[0].tensor().clone());
        while slots.len() < 3 {
            slots.push(filler.clone());
        }
        Ok(Self {
            kind,
            slots: [slots[0].clone(), slots[1].clone(), slots[2].clone()],
            batch,
        })
    }

    /// Which family this is.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The parameters, in the order [`Kind`]'s table lists them.
    pub fn params(&self) -> &[Var<R, E>] {
        &self.slots[..self.kind.arity()]
    }

    /// The device the parameters live on.
    pub fn device(&self) -> &Device<R> {
        self.slots[0].tensor().device()
    }

    fn tensors(&self) -> [&Tensor<R, E>; 3] {
        [
            self.slots[0].tensor(),
            self.slots[1].tensor(),
            self.slots[2].tensor(),
        ]
    }

    /// Reduce a full-size gradient back onto each parameter's own shape, dropping
    /// the slots that did not ask for one.
    fn fold(&self, grads: univariate::Grads<R, E>, value: Option<Shape>) -> Result<Vec<Option<Tensor<R, E>>>> {
        let mut out = Vec::with_capacity(4);
        out.push(match (grads.x, value) {
            (Some(g), Some(shape)) => Some(reduce_grad_to(&g, &shape)?),
            _ => None,
        });
        for (slot, grad) in [grads.a, grads.b, grads.c].into_iter().enumerate() {
            out.push(match grad {
                Some(g) => Some(reduce_grad_to(&g, self.slots[slot].shape())?),
                None => None,
            });
        }
        Ok(out)
    }

    /// The shape a value of this distribution must broadcast against, and whether it
    /// tiles the batch.
    fn value_shape(&self, value: &Shape) -> Result<Shape> {
        let out = Shape::broadcast(value, &self.batch)?;
        if &out != value {
            return Err(Error::shape(format!(
                "a value of {value} cannot be scored against a {} distribution: the \
                 result would be {out}, so broadcast the value first",
                self.batch
            )));
        }
        Ok(out)
    }

    /// A summary statistic, as one launch.
    fn summary(&self, which: u32, what: &str) -> Result<Tensor<R, E>> {
        univariate::parameterwise(self.tensors(), &self.batch, self.kind, which).map_err(|e| {
            Error::config(format!("{:?} has no {what}: {e}", self.kind))
        })
    }
}

/// Generate the named constructors from a table of families.
macro_rules! families {
    ($(
        $(#[$meta:meta])*
        $name:ident($($param:ident),+) => $kind:ident;
    )*) => {
        impl<R: Runtime, E: FloatElem> Univariate<R, E> {
            $(
                $(#[$meta])*
                pub fn $name(
                    $($param: impl Into<Param<R, E>>,)+
                    device: &Device<R>,
                ) -> Result<Self> {
                    Self::new(Kind::$kind, vec![$($param.into()),+], device)
                }
            )*
        }
    };
}

families! {
    /// The Gaussian, `N(loc, scale²)`.
    normal(loc, scale) => Normal;
    /// Flat on `[low, high)`.
    uniform(low, high) => Uniform;
    /// The waiting time of a Poisson process of the given rate.
    exponential(rate) => Exponential;
    /// Two exponential tails back to back.
    laplace(loc, scale) => Laplace;
    /// The ratio of two standard normals: heavy-tailed, and with no mean.
    cauchy(loc, scale) => Cauchy;
    /// The extreme-value distribution behind the Gumbel-max trick.
    gumbel(loc, scale) => Gumbel;
    /// A centred Gaussian folded at zero.
    half_normal(scale) => HalfNormal;
    /// A centred Cauchy folded at zero.
    half_cauchy(scale) => HalfCauchy;
    /// A Gaussian in the logarithm.
    log_normal(loc, scale) => LogNormal;
    /// A power law above a scale.
    pareto(scale, alpha) => Pareto;
    /// The Weibull, a stretched exponential.
    weibull(scale, concentration) => Weibull;
    /// A Beta-shaped density on the unit interval whose quantile is elementary.
    kumaraswamy(concentration1, concentration0) => Kumaraswamy;
    /// The Gamma, in shape-and-rate form.
    gamma(concentration, rate) => Gamma;
    /// The reciprocal of a Gamma.
    inverse_gamma(concentration, rate) => InverseGamma;
    /// The Beta, conjugate to a Bernoulli.
    beta(concentration1, concentration0) => Beta;
    /// Student's t, with a location and a scale.
    student_t(df, loc, scale) => StudentT;
    /// The ratio of two scaled chi-squares.
    fisher_snedecor(df1, df2) => FisherSnedecor;
    /// The circular analogue of a Gaussian.
    von_mises(loc, concentration) => VonMises;
    /// A continuous relaxation of a Bernoulli, supported on `[0, 1]`.
    continuous_bernoulli_logits(logits) => ContinuousBernoulli;
    /// A single coin flip, from its logit.
    bernoulli_logits(logits) => Bernoulli;
    /// Failures before the first success, from the success logit.
    geometric_logits(logits) => Geometric;
    /// Counts from a Poisson process.
    poisson(rate) => Poisson;
    /// Successes in `total_count` trials, from the success logit.
    binomial_logits(total_count, logits) => Binomial;
    /// Failures before `total_count` successes, from the success logit.
    negative_binomial_logits(total_count, logits) => NegativeBinomial;
    /// The pre-sigmoid Gumbel-softmax of one bit.
    logit_relaxed_bernoulli(temperature, logits) => LogitRelaxedBernoulli;
    /// A Gumbel-softmax relaxation of one bit, on `(0, 1)`.
    relaxed_bernoulli(temperature, logits) => RelaxedBernoulli;
}

impl<R: Runtime, E: FloatElem> Univariate<R, E> {
    /// `N(0, 1)`, shaped like `shape`.
    pub fn standard_normal(shape: impl Into<Shape>, device: &Device<R>) -> Result<Self> {
        let shape = shape.into();
        Self::normal(Tensor::<R, E>::zeros(shape, device), 1.0, device)
    }

    /// The chi-square distribution, which is `Gamma(df/2, ½)`.
    ///
    /// Stored as that Gamma rather than as a family of its own, exactly as PyTorch
    /// stores it, so every Gamma operation applies to it unchanged.
    pub fn chi2(df: impl Into<Param<R, E>>, device: &Device<R>) -> Result<Self> {
        let half = match df.into() {
            Param::Scalar(v) => Param::Scalar(0.5 * v),
            Param::Value(v) => Param::Value(v.mul_scalar(0.5)),
        };
        Self::gamma(half, 0.5, device)
    }

    /// A coin flip given its probability of heads rather than its logit.
    ///
    /// The probability is converted to a logit once, here — see the module docs on
    /// why the logit is what the kernels keep.
    pub fn bernoulli(probs: impl Into<Param<R, E>>, device: &Device<R>) -> Result<Self> {
        Self::bernoulli_logits(logits_of(probs, device)?, device)
    }

    /// [`Univariate::geometric_logits`] from a probability.
    pub fn geometric(probs: impl Into<Param<R, E>>, device: &Device<R>) -> Result<Self> {
        Self::geometric_logits(logits_of(probs, device)?, device)
    }

    /// [`Univariate::binomial_logits`] from a probability.
    pub fn binomial(
        total_count: impl Into<Param<R, E>>,
        probs: impl Into<Param<R, E>>,
        device: &Device<R>,
    ) -> Result<Self> {
        Self::binomial_logits(total_count, logits_of(probs, device)?, device)
    }

    /// [`Univariate::negative_binomial_logits`] from a probability.
    pub fn negative_binomial(
        total_count: impl Into<Param<R, E>>,
        probs: impl Into<Param<R, E>>,
        device: &Device<R>,
    ) -> Result<Self> {
        Self::negative_binomial_logits(total_count, logits_of(probs, device)?, device)
    }

    /// [`Univariate::continuous_bernoulli_logits`] from a probability.
    pub fn continuous_bernoulli(
        probs: impl Into<Param<R, E>>,
        device: &Device<R>,
    ) -> Result<Self> {
        Self::continuous_bernoulli_logits(logits_of(probs, device)?, device)
    }
}

/// `ln(p / (1 − p))`, differentiably, for a parameter given as a probability.
fn logits_of<R: Runtime, E: FloatElem>(
    probs: impl Into<Param<R, E>>,
    device: &Device<R>,
) -> Result<Param<R, E>> {
    Ok(match probs.into() {
        Param::Scalar(p) => Param::Scalar((p / (1.0 - p)).ln()),
        Param::Value(v) => {
            let one = Var::constant(Tensor::<R, E>::ones(vec![1], device));
            Param::Value(v.log().sub(&one.sub(&v)?.log())?)
        }
    })
}

impl<R: Runtime, E: FloatElem> Distribution<R, E> for Univariate<R, E> {
    fn batch_shape(&self) -> &Shape {
        &self.batch
    }

    fn event_shape(&self) -> Shape {
        Shape::scalar()
    }

    fn support(&self) -> Support {
        match self.kind {
            Kind::Normal
            | Kind::Laplace
            | Kind::Cauchy
            | Kind::Gumbel
            | Kind::StudentT
            | Kind::LogitRelaxedBernoulli => Support::Real,
            Kind::Uniform => Support::Interval,
            Kind::Exponential | Kind::HalfNormal | Kind::HalfCauchy => Support::NonNegative,
            Kind::LogNormal
            | Kind::Pareto
            | Kind::Weibull
            | Kind::Gamma
            | Kind::InverseGamma
            | Kind::FisherSnedecor => Support::Positive,
            Kind::Kumaraswamy | Kind::Beta | Kind::RelaxedBernoulli => Support::UnitOpen,
            Kind::ContinuousBernoulli => Support::UnitClosed,
            Kind::VonMises => Support::Circular,
            Kind::Bernoulli => Support::Boolean,
            Kind::Geometric | Kind::Poisson | Kind::NegativeBinomial => Support::Counting,
            Kind::Binomial => Support::BoundedCounting,
        }
    }

    fn has_rsample(&self) -> bool {
        self.kind.reparameterised()
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        univariate::sample(
            self.tensors(),
            self.batch.clone(),
            self.batch.num_elements(),
            0,
            seed,
            self.kind,
        )
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        let mut dims = vec![n];
        dims.extend_from_slice(self.batch.dims());
        univariate::sample(
            self.tensors(),
            Shape::new(dims),
            self.batch.num_elements(),
            0,
            seed,
            self.kind,
        )
    }

    fn rsample(&self, seed: u64) -> Result<Var<R, E>> {
        if !self.has_rsample() {
            return Err(Error::Unsupported(format!(
                "{:?} has no path derivative: its sampler's control flow depends on \
                 its parameters, so a draw is not a differentiable function of them. \
                 Use `sample` with a score-function estimator instead.",
                self.kind
            )));
        }
        let value = self.sample(seed)?;
        let batch = self.batch.num_elements();
        let kind = self.kind;
        let owned = self.clone();
        let tensors: Vec<Tensor<R, E>> = self.tensors().iter().map(|t| (*t).clone()).collect();
        Ok(Var::record_with_mask(
            value,
            &[&self.slots[0], &self.slots[1], &self.slots[2]],
            |want| {
                let wants = Wants {
                    x: false,
                    a: want[0],
                    b: want[1],
                    c: want[2],
                };
                Box::new(move |g| {
                    let refs = [&tensors[0], &tensors[1], &tensors[2]];
                    let grads =
                        univariate::param_grad(g, refs, batch, 0, seed, kind, 4, wants)?;
                    Ok(owned.fold(grads, None)?[1..].to_vec())
                })
            },
        ))
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        let shape = self.value_shape(value.shape())?;
        let out = univariate::pointwise(
            value.tensor(),
            self.tensors(),
            self.batch.num_elements(),
            self.kind,
            op::LOG_PROB,
        )?;
        let batch = self.batch.num_elements();
        let kind = self.kind;
        let owned = self.clone();
        let seen = value.tensor().clone();
        let tensors: Vec<Tensor<R, E>> = self.tensors().iter().map(|t| (*t).clone()).collect();
        let _ = shape;
        let value_shape = value.shape().clone();
        Ok(Var::record_with_mask(
            out,
            &[value, &self.slots[0], &self.slots[1], &self.slots[2]],
            |want| {
                let wants = Wants {
                    x: want[0],
                    a: want[1],
                    b: want[2],
                    c: want[3],
                };
                Box::new(move |g| {
                    let refs = [&tensors[0], &tensors[1], &tensors[2]];
                    let grads = univariate::pointwise_grad(
                        g,
                        &seen,
                        refs,
                        batch,
                        kind,
                        op::LOG_PROB,
                        wants,
                    )?;
                    owned.fold(grads, Some(value_shape.clone()))
                })
            },
        ))
    }

    fn cdf(&self, value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        if !self.kind.has_cdf() {
            return Err(Error::Unsupported(format!(
                "{:?} has no closed-form CDF",
                self.kind
            )));
        }
        self.value_shape(value.shape())?;
        univariate::pointwise(
            value,
            self.tensors(),
            self.batch.num_elements(),
            self.kind,
            op::CDF,
        )
    }

    fn icdf(&self, q: &Var<R, E>) -> Result<Var<R, E>> {
        if !self.kind.has_icdf() {
            return Err(Error::Unsupported(format!(
                "{:?} has no closed-form quantile",
                self.kind
            )));
        }
        self.value_shape(q.shape())?;
        let out = univariate::pointwise(
            q.tensor(),
            self.tensors(),
            self.batch.num_elements(),
            self.kind,
            op::ICDF,
        )?;
        if !self.kind.has_icdf_grad() {
            // The three discrete quantiles — continuous Bernoulli, Bernoulli,
            // geometric — are step functions of the probability and constants in
            // their parameters wherever they are differentiable at all, so the value
            // is returned off the tape rather than with a zero attached to it.
            return Ok(Var::constant(out));
        }
        let batch = self.batch.num_elements();
        let kind = self.kind;
        let owned = self.clone();
        let seen = q.tensor().clone();
        let tensors: Vec<Tensor<R, E>> = self.tensors().iter().map(|t| (*t).clone()).collect();
        let value_shape = q.shape().clone();
        Ok(Var::record_with_mask(
            out,
            &[q, &self.slots[0], &self.slots[1], &self.slots[2]],
            |want| {
                let wants = Wants {
                    x: want[0],
                    a: want[1],
                    b: want[2],
                    c: want[3],
                };
                Box::new(move |g| {
                    let refs = [&tensors[0], &tensors[1], &tensors[2]];
                    let grads =
                        univariate::pointwise_grad(g, &seen, refs, batch, kind, op::ICDF, wants)?;
                    owned.fold(grads, Some(value_shape.clone()))
                })
            },
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        if !self.kind.has_entropy() {
            return Err(Error::Unsupported(format!(
                "{:?} has no closed-form entropy",
                self.kind
            )));
        }
        let out = univariate::parameterwise(self.tensors(), &self.batch, self.kind, moment::MODE + 1)?;
        let batch = self.batch.num_elements();
        let kind = self.kind;
        let owned = self.clone();
        let tensors: Vec<Tensor<R, E>> = self.tensors().iter().map(|t| (*t).clone()).collect();
        Ok(Var::record_with_mask(
            out,
            &[&self.slots[0], &self.slots[1], &self.slots[2]],
            |want| {
                let wants = Wants {
                    x: false,
                    a: want[0],
                    b: want[1],
                    c: want[2],
                };
                Box::new(move |g| {
                    let refs = [&tensors[0], &tensors[1], &tensors[2]];
                    let grads = univariate::param_grad(g, refs, batch, 0, 0, kind, 3, wants)?;
                    Ok(owned.fold(grads, None)?[1..].to_vec())
                })
            },
        ))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        self.summary(moment::MEAN, "mean")
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        self.summary(moment::VARIANCE, "variance")
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        self.summary(moment::MODE, "mode")
    }
}
