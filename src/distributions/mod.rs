//! Probability distributions, sampled and scored on the device.
//!
//! A port of `torch.distributions` to CubeCL, built for the one thing a
//! reinforcement-learning loop needs that a general-purpose library usually does not
//! provide: **the whole of it stays on the device**. A policy draws an action,
//! scores it, records it, and later differentiates that score, without a byte
//! crossing the bus and without a host-side generator anywhere in the loop. That is
//! the same discipline [`crate::rl`] already holds its collector to, extended from
//! the four operations it hard-coded to the full family of distributions.
//!
//! ```no_run
//! use mamba3::prelude::*;
//! use mamba3::distributions::{Distribution, Univariate};
//!
//! # fn main() -> mamba3::error::Result<()> {
//! type R = mamba3::backends::Auto;
//! let device = Device::<R>::default();
//!
//! // A diagonal Gaussian policy over a batch of continuous actions.
//! let mean = Tensor::<R, f32>::zeros(vec![64, 6], &device);
//! let policy = Univariate::normal(&mean, 0.5, &device)?;
//!
//! let action = policy.rsample(0xC0FFEE)?;       // differentiable draw
//! let logp = policy.log_prob(&action)?;         // differentiable score
//! let bonus = policy.entropy()?;                // differentiable exploration bonus
//! # Ok(())
//! # }
//! ```
//!
//! # What is here
//!
//! | | |
//! |---|---|
//! | [`Univariate`] | twenty-six scalar families — see [`Kind`] for the table |
//! | [`Categorical`] | a distribution over the last axis of a logit tensor |
//! | [`OneHotCategorical`] | the same, drawn and scored as one-hot vectors |
//! | [`RelaxedOneHotCategorical`] | the Gumbel-softmax, which *is* differentiable |
//! | [`Multinomial`] | counts from repeated categorical trials |
//! | [`Dirichlet`] | the categorical's conjugate, over the simplex |
//! | [`MultivariateNormal`] | a Gaussian with a full or diagonal covariance |
//! | [`Independent`] | reinterprets batch axes as event axes |
//! | [`TransformedDistribution`] | a base pushed through a [`Transform`] |
//! | [`MixtureSameFamily`] | a categorical weighting of a batch of components |
//! | [`kl_divergence`] | the analytic pairs, where one exists; [`kl`] has the rest |
//! | [`elementwise`] | the special functions as differentiable tensor operations |
//!
//! [`crate::rl::PolicyOutput::distribution`] is the bridge from a policy to the
//! first of these, and [`crate::rl::ppo`] scores and regularises through it.
//!
//! # How it is fast
//!
//! **One kernel per question.** A normal log-density is one launch. Written out of
//! elementwise tensor operations — which is what PyTorch does, and what a
//! straightforward port would do — it is eight launches and seven intermediates the
//! width of the batch. The arithmetic is the same; the traffic is not. The
//! distribution is chosen at *compile* time through a `#[comptime]` discriminant, so
//! the generated kernel contains its own density and nothing else, with no dispatch
//! left to run.
//!
//! **Analytic gradients.** Every log-density, entropy and reparameterised draw has a
//! hand-derived adjoint that is again one kernel. This is the deliberate exception
//! to [`crate::autograd`]'s rule that only primitives carry a hand-written adjoint:
//! a composed `∂ log p/∂σ` is a dozen launches, and written out it is `(z² − 1)/σ`.
//!
//! **Scalar parameters cost nothing.** `Univariate::normal(&mean, 0.5, …)` does not
//! materialise a tensor of halves. A parameter that is one value is bound with a
//! stride of zero and read from a register.
//!
//! # How it is exact
//!
//! Everything random comes from Philox-4×32-10 ([`rng`]), keyed by the seed and
//! *counted by the element index*. A draw therefore depends on where it is and on
//! nothing else — not the cube geometry, not the batch size, not what else was in
//! the launch. Two consequences follow, and both are asserted in
//! `tests/distributions_bitexact.rs`:
//!
//! * the same call twice returns bit-identical results, on any launch shape;
//! * element `i` of a batch of eight is the same draw as element `i` of a batch of
//!   eight thousand.
//!
//! The mathematics is written once and compiled twice — once by CubeCL for the
//! device, once by rustc for the host — so "the device computes what the reference
//! computes" is a claim about one program rather than about two transcriptions of
//! one idea. See [`special`] for how, and for where the claim stops: `exp` and `ln`
//! are library calls, and a backend whose library differs from the host's will
//! differ in the last bit. The test names that backend rather than merely failing.
//!
//! # Where this departs from PyTorch
//!
//! Three deliberate differences, each because the alternative is worse here:
//!
//! * **Parameters are validated by shape, not by value.** PyTorch's
//!   `validate_args=True` checks that a scale is positive, which on a device means
//!   reading it back and stalling the queue. Shapes are checked on the host, where
//!   they already live; values are not.
//! * **Probabilities are stored as logits.** A distribution built from `probs`
//!   converts once, at construction. `log(1 − p)` from a `p` near one has no digits
//!   left, and `−softplus(logit)` has all of them.
//! * **`rsample` covers the inverse-CDF families only.** PyTorch reparameterises
//!   `Gamma`, `Beta` and `Dirichlet` too, through implicit differentiation of an
//!   incomplete gamma — a different thing from a path derivative, and a large
//!   apparatus. [`Distribution::has_rsample`] reports the truth for each family.
//!
//! Two places where the spelling is deliberately *better* than PyTorch's are marked
//! at the point they occur: a normal's CDF, and its quantile in the far tail.
//!
//! Three of PyTorch's classes are absent. `Wishart` and `LKJCholesky` are
//! distributions over matrices, need a decomposition per draw, and do not appear in
//! a reinforcement-learning loop. `LowRankMultivariateNormal` is absent for a
//! narrower reason: its density needs the inverse of an `r × r` capacitance matrix,
//! and a *differentiable* Cholesky is not something [`crate::autograd`] has — a
//! version whose parameters could not be trained would be worse than none.

use cubecl::prelude::Runtime;

pub mod categorical;
pub mod combinator;
pub mod elementwise;
pub mod kl;
pub mod multivariate;
pub mod rng;
pub mod scalar;
pub mod simplex;
pub mod special;
pub mod univariate;

pub use categorical::Categorical;
pub use combinator::{
    AffineTransform, ExpTransform, Independent, MixtureSameFamily, PowerTransform,
    SigmoidTransform, TanhTransform, Transform, TransformedDistribution,
};
pub use kl::kl_divergence;
pub use multivariate::MultivariateNormal;
pub use scalar::Univariate;
pub use simplex::{Dirichlet, Multinomial, OneHotCategorical, RelaxedOneHotCategorical};
pub use univariate::Kind;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::Result;
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// The set a distribution's samples live in.
///
/// Carried so a caller can check its own values if it wants to; nothing here checks
/// them, because checking a device tensor's values means reading it back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Support {
    /// The whole real line.
    Real,
    /// `(0, ∞)`.
    Positive,
    /// `[0, ∞)`.
    NonNegative,
    /// `(0, 1)`.
    UnitOpen,
    /// `[0, 1]`.
    UnitClosed,
    /// A half-open interval, which is only known once the parameters are.
    Interval,
    /// `{0, 1}`.
    Boolean,
    /// `{0, 1, 2, …}`.
    Counting,
    /// `{0, 1, …, n}` for a parameter `n`.
    BoundedCounting,
    /// `[−π, π)`.
    Circular,
    /// The probability simplex over the last axis.
    Simplex,
    /// `{0, …, k−1}` for a `k`-way choice.
    Categories,
}

/// A distribution parameter: a tensor, a tracked tensor, or one number.
///
/// `f32` is the case worth having. A scalar parameter is kept as one value and bound
/// to the kernel with a stride of zero, so `Univariate::normal(&mean, 1.0, …)`
/// allocates nothing for the scale and reads it from a register.
pub enum Param<R: Runtime, E: FloatElem = f32> {
    /// One number, broadcast over the whole batch.
    Scalar(f32),
    /// A tensor, possibly on the tape.
    Value(Var<R, E>),
}

impl<R: Runtime, E: FloatElem> From<f32> for Param<R, E> {
    fn from(v: f32) -> Self {
        Param::Scalar(v)
    }
}

impl<R: Runtime, E: FloatElem> From<f64> for Param<R, E> {
    fn from(v: f64) -> Self {
        Param::Scalar(v as f32)
    }
}

impl<R: Runtime, E: FloatElem> From<Tensor<R, E>> for Param<R, E> {
    fn from(t: Tensor<R, E>) -> Self {
        Param::Value(Var::constant(t))
    }
}

impl<R: Runtime, E: FloatElem> From<&Tensor<R, E>> for Param<R, E> {
    fn from(t: &Tensor<R, E>) -> Self {
        Param::Value(Var::constant(t.clone()))
    }
}

impl<R: Runtime, E: FloatElem> From<Var<R, E>> for Param<R, E> {
    fn from(v: Var<R, E>) -> Self {
        Param::Value(v)
    }
}

impl<R: Runtime, E: FloatElem> From<&Var<R, E>> for Param<R, E> {
    fn from(v: &Var<R, E>) -> Self {
        Param::Value(v.clone())
    }
}

impl<R: Runtime, E: FloatElem> Param<R, E> {
    /// The parameter as a `Var`, materialising a scalar into a one-element tensor.
    fn into_var(self, device: &Device<R>) -> Var<R, E> {
        match self {
            Param::Scalar(v) => Var::constant(
                Tensor::from_f32(&[v], vec![1], device).expect("one value fills one element"),
            ),
            Param::Value(v) => v,
        }
    }
}

/// What every distribution can be asked.
///
/// The shape of the trait follows `torch.distributions.Distribution`, with the
/// difference that anything a device cannot answer without stalling — validating a
/// parameter's value, say — is not asked. Operations a family has no closed form for
/// return [`Error::Unsupported`](crate::error::Error::Unsupported) rather than an
/// approximation.
pub trait Distribution<R: Runtime, E: FloatElem> {
    /// The shape of the parameters, and so of one draw.
    fn batch_shape(&self) -> &Shape;

    /// The shape of one *event*: empty for a scalar family, `[k]` for a categorical
    /// or a multivariate normal over `k` dimensions.
    fn event_shape(&self) -> Shape;

    /// The set a draw lives in.
    fn support(&self) -> Support;

    /// Whether [`Distribution::rsample`] exists for this family.
    fn has_rsample(&self) -> bool;

    /// One draw per batch element, detached from the tape.
    fn sample(&self, seed: u64) -> Result<Tensor<R, E>>;

    /// `n` independent draws, shaped `[n, ..batch_shape]`.
    ///
    /// Row `j` is what a `sample` with the same seed would have produced had it been
    /// asked for the `j`-th block of elements, which is what makes a sample's value
    /// independent of how many were asked for.
    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>>;

    /// One draw per batch element, differentiable with respect to the parameters.
    ///
    /// The path derivative, not the score function: the draw is written as a fixed
    /// function of a parameter-free uniform, and differentiated as one. Available
    /// only where [`Distribution::has_rsample`] is true.
    fn rsample(&self, seed: u64) -> Result<Var<R, E>>;

    /// The log-density, or log-mass, at `value`.
    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>>;

    /// The cumulative distribution function at `value`.
    fn cdf(&self, value: &Tensor<R, E>) -> Result<Tensor<R, E>>;

    /// The quantile function at `q ∈ (0, 1)`.
    ///
    /// Differentiable for the families a draw is reparameterised through — the
    /// result is literally what [`Distribution::rsample`] computes — and returned
    /// off the tape for the discrete ones, whose quantile is a step function.
    fn icdf(&self, q: &Var<R, E>) -> Result<Var<R, E>>;

    /// The differential or Shannon entropy.
    fn entropy(&self) -> Result<Var<R, E>>;

    /// The mean, where one exists.
    fn mean(&self) -> Result<Tensor<R, E>>;

    /// The variance, where one exists.
    fn variance(&self) -> Result<Tensor<R, E>>;

    /// The mode, where one exists.
    fn mode(&self) -> Result<Tensor<R, E>>;

    /// The standard deviation: the root of [`Distribution::variance`].
    fn stddev(&self) -> Result<Tensor<R, E>> {
        Ok(crate::tensor::ops::elemwise::sqrt(&self.variance()?))
    }

    /// The number of elements one draw holds.
    fn batch_len(&self) -> usize {
        self.batch_shape().num_elements()
    }
}
