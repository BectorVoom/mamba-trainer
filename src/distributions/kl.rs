//! Analytic Kullback–Leibler divergences.
//!
//! `KL(p ‖ q)` for the pairs that have a closed form — which for a policy-gradient
//! method is the pair that matters, since PPO's early stop, TRPO's trust region and
//! a variational bound are all "how far has the policy moved" measured exactly
//! rather than sampled.
//!
//! Everything here is **composed** out of ordinary tensor operations and the
//! differentiable special functions in [`super::elementwise`], not written as a
//! kernel. That is the opposite of the choice [`super::univariate`] makes, and it is
//! the right one here for two reasons: a divergence is evaluated once per update
//! rather than once per element of a rollout, so its handful of extra launches do
//! not show up; and composing it means its gradient comes from [`crate::autograd`]
//! and is correct by construction rather than by a second derivation. `KL(Gamma ‖
//! Gamma)` differentiates through a digamma without anyone having had to write down
//! a trigamma by hand.
//!
//! # Which pairs
//!
//! Same-family pairs for: normal, uniform, exponential, Laplace, Cauchy, Gumbel,
//! half-normal, half-Cauchy, log-normal, Pareto, gamma, inverse-gamma, beta,
//! Bernoulli, geometric and Poisson; plus [`categorical`] and [`dirichlet`], which
//! take their own arguments because their parameters are rows rather than scalars.
//! Anything else returns [`Error::Unsupported`] naming the pair, and the honest
//! answer there is a Monte Carlo estimate: `p.rsample()` scored under both.
//!
//! Where `p`'s support is not contained in `q`'s — a uniform wider than the one it
//! is compared against — the divergence is genuinely infinite, and that is what is
//! returned, arrived at by overflow rather than by a branch so that the shape of the
//! computation does not depend on the data.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::tensor::shape::Shape;

use super::elementwise::{digamma, lgamma};
use super::univariate::Kind;
use super::{Categorical, Dirichlet, Distribution, Univariate};

/// Half of `f32`'s exponent range, twice: `m·H·H` is `0` when `m` is and `+∞` when
/// `m` is one, with no branch and no `0 × ∞`.
const HALF_OVERFLOW: f32 = 1.0e38;

/// `+∞` wherever `mask` is one and `0` wherever it is zero.
fn blow_up<R: Runtime, E: FloatElem>(mask: &Var<R, E>) -> Var<R, E> {
    mask.mul_scalar(HALF_OVERFLOW).mul_scalar(HALF_OVERFLOW)
}

/// `1` where `lhs > rhs`, else `0`, as an untracked constant.
fn greater<R: Runtime, E: FloatElem>(lhs: &Var<R, E>, rhs: &Var<R, E>) -> Result<Var<R, E>> {
    Ok(Var::constant(crate::tensor::ops::elemwise::greater(
        lhs.tensor(),
        rhs.tensor(),
    )?))
}

/// `KL(p ‖ q)` for two scalar distributions of the same family.
///
/// The result has the broadcast of the two batch shapes, and is differentiable with
/// respect to both sets of parameters.
pub fn kl_divergence<R: Runtime, E: FloatElem>(
    p: &Univariate<R, E>,
    q: &Univariate<R, E>,
) -> Result<Var<R, E>> {
    if p.kind() != q.kind() {
        return Err(Error::Unsupported(format!(
            "no closed-form divergence between {:?} and {:?}; estimate it from \
             samples of p scored under both",
            p.kind(),
            q.kind()
        )));
    }
    let (pp, qq) = (p.params(), q.params());
    match p.kind() {
        Kind::Normal => normal_like(&pp[0], &pp[1], &qq[0], &qq[1]),
        Kind::LogNormal => normal_like(&pp[0], &pp[1], &qq[0], &qq[1]),
        Kind::HalfNormal => {
            let zero = zeros_like(&pp[0])?;
            normal_like(&zero, &pp[0], &zero, &qq[0])
        }
        Kind::Uniform => {
            // `ln((q.high − q.low)/(p.high − p.low))`, and infinite unless p's
            // interval sits inside q's.
            let ratio = qq[1].sub(&qq[0])?.div(&pp[1].sub(&pp[0])?)?.log();
            let below = greater(&qq[0], &pp[0])?;
            let above = greater(&pp[1], &qq[1])?;
            ratio.add(&blow_up(&below.add(&above)?))
        }
        Kind::Exponential => {
            let ratio = qq[0].div(&pp[0])?;
            ratio.sub(&ratio.log())?.add_scalar(-1.0).into_ok()
        }
        Kind::Laplace => {
            let scale_ratio = pp[1].div(&qq[1])?;
            let gap = pp[0].sub(&qq[0])?.abs();
            let t2 = gap.div(&qq[1])?;
            let t3 = scale_ratio.mul(&gap.div(&pp[1])?.neg().exp())?;
            scale_ratio
                .log()
                .neg()
                .add(&t2)?
                .add(&t3)?
                .add_scalar(-1.0)
                .into_ok()
        }
        Kind::Cauchy => cauchy_like(&pp[0], &pp[1], &qq[0], &qq[1]),
        Kind::HalfCauchy => {
            let zero = zeros_like(&pp[0])?;
            cauchy_like(&zero, &pp[0], &zero, &qq[0])
        }
        Kind::Gumbel => {
            // PyTorch's arrangement, which keeps every term finite for scales that
            // differ by orders of magnitude.
            let ct1 = pp[1].div(&qq[1])?;
            let ct2 = qq[0].div(&qq[1])?;
            let ct3 = pp[0].div(&qq[1])?;
            let t1 = ct1.log().neg().sub(&ct2)?.add(&ct3)?;
            let t2 = ct1.mul_scalar(super::special::EULER_GAMMA);
            let t3 = ct2.add(&lgamma(&ct1.add_scalar(1.0)))?.sub(&ct3)?.exp();
            t1.add(&t2)?
                .add(&t3)?
                .add_scalar(-(1.0 + super::special::EULER_GAMMA))
                .into_ok()
        }
        Kind::Pareto => {
            let scale_ratio = pp[0].div(&qq[0])?;
            let alpha_ratio = qq[1].div(&pp[1])?;
            let t1 = qq[1].mul(&scale_ratio.log())?;
            let t2 = alpha_ratio.log().neg();
            let below = greater(&qq[0], &pp[0])?;
            t1.add(&t2)?
                .add(&alpha_ratio)?
                .add_scalar(-1.0)
                .add(&blow_up(&below))
        }
        Kind::Gamma => gamma_like(&pp[0], &pp[1], &qq[0], &qq[1]),
        Kind::InverseGamma => gamma_like(&pp[0], &pp[1], &qq[0], &qq[1]),
        Kind::Beta => {
            let sum_p = pp[0].add(&pp[1])?;
            let sum_q = qq[0].add(&qq[1])?;
            let t1 = lgamma(&qq[0]).add(&lgamma(&qq[1]))?.add(&lgamma(&sum_p))?;
            let t2 = lgamma(&pp[0]).add(&lgamma(&pp[1]))?.add(&lgamma(&sum_q))?;
            let t3 = pp[0].sub(&qq[0])?.mul(&digamma(&pp[0]))?;
            let t4 = pp[1].sub(&qq[1])?.mul(&digamma(&pp[1]))?;
            let t5 = sum_q.sub(&sum_p)?.mul(&digamma(&sum_p))?;
            t1.sub(&t2)?.add(&t3)?.add(&t4)?.add(&t5)
        }
        Kind::Bernoulli => bernoulli_like(&pp[0], &qq[0]),
        Kind::Geometric => {
            // `−H(p) − ln(1 − q_p)/p_p − logit(q)`, PyTorch's arrangement, which
            // unwinds to `E₁[k]·ln((1−p₁)/(1−p₂)) + ln(p₁/p₂)` — the sum a geometric
            // divergence is.
            let entropy = p.entropy()?;
            let inv_probs = pp[0].sigmoid().recip();
            // `ln(1 − q_p) = ln σ(−logit) = −softplus(logit)`.
            let log1m_q = qq[0].softplus()?.neg();
            entropy.neg().sub(&log1m_q.mul(&inv_probs)?)?.sub(&qq[0])
        }
        Kind::Poisson => {
            let ratio = pp[0].log().sub(&qq[0].log())?;
            pp[0].mul(&ratio)?.sub(&pp[0])?.add(&qq[0])
        }
        other => Err(Error::Unsupported(format!(
            "no closed-form divergence for a pair of {other:?}; estimate it from \
             samples of p scored under both"
        ))),
    }
}

/// `KL(N(μ₁, σ₁) ‖ N(μ₂, σ₂))`.
fn normal_like<R: Runtime, E: FloatElem>(
    loc_p: &Var<R, E>,
    scale_p: &Var<R, E>,
    loc_q: &Var<R, E>,
    scale_q: &Var<R, E>,
) -> Result<Var<R, E>> {
    let ratio = scale_p.div(scale_q)?;
    let var_ratio = ratio.mul(&ratio)?;
    let shift = loc_p.sub(loc_q)?.div(scale_q)?;
    var_ratio
        .add(&shift.mul(&shift)?)?
        .sub(&var_ratio.log())?
        .add_scalar(-1.0)
        .mul_scalar(0.5)
        .into_ok()
}

/// `KL(Cauchy ‖ Cauchy)`.
fn cauchy_like<R: Runtime, E: FloatElem>(
    loc_p: &Var<R, E>,
    scale_p: &Var<R, E>,
    loc_q: &Var<R, E>,
    scale_q: &Var<R, E>,
) -> Result<Var<R, E>> {
    let sum = scale_p.add(scale_q)?;
    let shift = loc_p.sub(loc_q)?;
    let numer = sum.mul(&sum)?.add(&shift.mul(&shift)?)?;
    let denom = scale_p.mul(scale_q)?.mul_scalar(4.0);
    numer.log().sub(&denom.log())
}

/// `KL(Gamma ‖ Gamma)`, which is also the inverse-gamma's.
fn gamma_like<R: Runtime, E: FloatElem>(
    conc_p: &Var<R, E>,
    rate_p: &Var<R, E>,
    conc_q: &Var<R, E>,
    rate_q: &Var<R, E>,
) -> Result<Var<R, E>> {
    let t1 = conc_q.mul(&rate_p.div(rate_q)?.log())?;
    let t2 = lgamma(conc_q).sub(&lgamma(conc_p))?;
    let t3 = conc_p.sub(conc_q)?.mul(&digamma(conc_p))?;
    let t4 = rate_q.sub(rate_p)?.mul(&conc_p.div(rate_p)?)?;
    t1.add(&t2)?.add(&t3)?.add(&t4)
}

/// `KL(Bernoulli ‖ Bernoulli)` from two logits.
fn bernoulli_like<R: Runtime, E: FloatElem>(
    logit_p: &Var<R, E>,
    logit_q: &Var<R, E>,
) -> Result<Var<R, E>> {
    // `p(ln p₁ − ln p₂) + (1 − p)(ln(1 − p₁) − ln(1 − p₂))`, with every logarithm
    // spelled as a softplus so a probability of one keeps its digits.
    let p = logit_p.sigmoid();
    let head = logit_p.neg().softplus()?.sub(&logit_q.neg().softplus()?)?;
    let tail = logit_p.softplus()?.sub(&logit_q.softplus()?)?;
    let one_minus = p.rsub_scalar(1.0);
    p.mul(&head)?.add(&one_minus.mul(&tail)?)?.neg().into_ok()
}

/// A zero the same shape as `like`, untracked.
fn zeros_like<R: Runtime, E: FloatElem>(like: &Var<R, E>) -> Result<Var<R, E>> {
    Ok(Var::constant(crate::tensor::base::Tensor::zeros(
        like.shape().clone(),
        like.tensor().device(),
    )))
}

/// `KL(p ‖ q)` for two categoricals over the same class axis.
///
/// `Σ pᵢ (ln pᵢ − ln qᵢ)`, taken in log space so a class that `p` never visits
/// contributes nothing rather than a `0 × −∞`.
pub fn categorical<R: Runtime, E: FloatElem>(
    p: &Categorical<R, E>,
    q: &Categorical<R, E>,
) -> Result<Var<R, E>> {
    if p.classes() != q.classes() {
        return Err(Error::shape(format!(
            "cannot compare a {}-way categorical with a {}-way one",
            p.classes(),
            q.classes()
        )));
    }
    let axis = p.logits().rank() - 1;
    let log_p = p.logits().log_softmax(axis)?;
    let log_q = q.logits().log_softmax(axis)?;
    let probs = log_p.exp();
    probs.mul(&log_p.sub(&log_q)?)?.sum_dim(axis)?.squeeze(axis)
}

/// `KL(p ‖ q)` for two Dirichlets over the same simplex.
pub fn dirichlet<R: Runtime, E: FloatElem>(
    p: &Dirichlet<R, E>,
    q: &Dirichlet<R, E>,
) -> Result<Var<R, E>> {
    if p.classes() != q.classes() {
        return Err(Error::shape(format!(
            "cannot compare a {}-class Dirichlet with a {}-class one",
            p.classes(),
            q.classes()
        )));
    }
    let axis = p.concentration().rank() - 1;
    let cp = p.concentration();
    let cq = q.concentration();
    let sum_p = cp.sum_dim(axis)?;
    let sum_q = cq.sum_dim(axis)?;
    let t1 = lgamma(&sum_p).sub(&lgamma(&sum_q))?.squeeze(axis)?;
    let t2 = lgamma(cp).sub(&lgamma(cq))?.sum_dim(axis)?.squeeze(axis)?;
    let t3 = cp.sub(cq)?;
    let t4 = digamma(cp).sub(&digamma(&sum_p).expand(cp.shape().clone())?)?;
    t1.sub(&t2)?
        .add(&t3.mul(&t4)?.sum_dim(axis)?.squeeze(axis)?)
}

/// The batch shape a divergence between `p` and `q` would produce.
pub fn broadcast_shape<R: Runtime, E: FloatElem>(
    p: &Univariate<R, E>,
    q: &Univariate<R, E>,
) -> Result<Shape> {
    Shape::broadcast(p.batch_shape(), q.batch_shape())
}

/// A tiny helper so a chain of `?`-free arithmetic can end in a `Result`.
trait IntoOk<T> {
    fn into_ok(self) -> Result<T>;
}

impl<T> IntoOk<T> for T {
    fn into_ok(self) -> Result<T> {
        Ok(self)
    }
}
