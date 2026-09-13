//! Distributions built out of other distributions.
//!
//! Three combinators, and between them most of what a continuous-control policy is:
//!
//! * [`Independent`] turns a batch of scalars into one vector event, which is the
//!   difference between "six independent Gaussians" and "a six-dimensional diagonal
//!   Gaussian". It is what makes `log_prob` return one number per action rather than
//!   six.
//! * [`TransformedDistribution`] pushes a base through an invertible map and carries
//!   the Jacobian, which is how a squashed Gaussian — `tanh(N(μ, σ))`, the SAC
//!   policy — gets a correct density instead of an approximate one.
//! * [`MixtureSameFamily`] weights a batch of components by a categorical, which is
//!   how a mixture-density policy is written.
//!
//! Everything here is composition over [`Var`] operations, so the gradients come
//! from [`crate::autograd`] and are correct by construction. Nothing here needs a
//! kernel of its own, and none of it reads back.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

use super::{Categorical, Distribution, Support};

// ---------------------------------------------------------------------------
// Independent
// ---------------------------------------------------------------------------

/// Reinterpret the last `n` batch axes of a distribution as one event.
///
/// The distribution is unchanged; what changes is what counts as "one sample", and
/// therefore what [`Distribution::log_prob`] and [`Distribution::entropy`] sum over.
/// A `[batch, actions]` Gaussian wrapped with `n = 1` scores a whole action vector
/// with one number, which is what a policy gradient wants.
pub struct Independent<D> {
    base: D,
    reinterpreted: usize,
    batch: Shape,
    event: Shape,
}

impl<D> Independent<D> {
    /// Move the last `reinterpreted` batch axes into the event shape.
    pub fn new<R: Runtime, E: FloatElem>(base: D, reinterpreted: usize) -> Result<Self>
    where
        D: Distribution<R, E>,
    {
        let dims = base.batch_shape().dims();
        if reinterpreted > dims.len() {
            return Err(Error::shape(format!(
                "cannot reinterpret {reinterpreted} axes of a {} batch",
                base.batch_shape()
            )));
        }
        let split = dims.len() - reinterpreted;
        let batch = Shape::new(dims[..split].to_vec());
        let mut event = dims[split..].to_vec();
        event.extend_from_slice(base.event_shape().dims());
        Ok(Self {
            base,
            reinterpreted,
            batch,
            event: Shape::new(event),
        })
    }

    /// The distribution underneath.
    pub fn base(&self) -> &D {
        &self.base
    }
}

/// Sum the last `n` axes away.
fn sum_trailing<R: Runtime, E: FloatElem>(v: &Var<R, E>, n: usize) -> Result<Var<R, E>> {
    let mut out = v.clone();
    for _ in 0..n {
        let axis = out.rank() - 1;
        out = out.sum_dim(axis)?.squeeze(axis)?;
    }
    Ok(out)
}

impl<R: Runtime, E: FloatElem, D: Distribution<R, E>> Distribution<R, E> for Independent<D> {
    fn batch_shape(&self) -> &Shape {
        &self.batch
    }

    fn event_shape(&self) -> Shape {
        self.event.clone()
    }

    fn support(&self) -> Support {
        self.base.support()
    }

    fn has_rsample(&self) -> bool {
        self.base.has_rsample()
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        self.base.sample(seed)
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        self.base.sample_n(n, seed)
    }

    fn rsample(&self, seed: u64) -> Result<Var<R, E>> {
        self.base.rsample(seed)
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        sum_trailing(&self.base.log_prob(value)?, self.reinterpreted)
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "an Independent event is multivariate and has no scalar CDF".to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "an Independent event is multivariate and has no scalar quantile".to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        sum_trailing(&self.base.entropy()?, self.reinterpreted)
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        self.base.mean()
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        self.base.variance()
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        self.base.mode()
    }
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

/// An invertible map, with the log-Jacobian that turns a density into a density.
///
/// The contract is PyTorch's: `inverse` undoes `forward`, and
/// `log_abs_det_jacobian(x, y)` is `ln |dy/dx|`, summed over the transform's own
/// event axes if it has any.
pub trait Transform<R: Runtime, E: FloatElem> {
    /// `y = f(x)`.
    fn forward(&self, x: &Var<R, E>) -> Result<Var<R, E>>;

    /// `x = f⁻¹(y)`.
    fn inverse(&self, y: &Var<R, E>) -> Result<Var<R, E>>;

    /// `ln |dy/dx|`, elementwise unless the transform says otherwise.
    fn log_abs_det_jacobian(&self, x: &Var<R, E>, y: &Var<R, E>) -> Result<Var<R, E>>;

    /// How many trailing axes one application couples. Zero for an elementwise map.
    fn event_dim(&self) -> usize {
        0
    }
}

/// `y = loc + scale·x`.
pub struct AffineTransform<R: Runtime, E: FloatElem = f32> {
    loc: Var<R, E>,
    scale: Var<R, E>,
}

impl<R: Runtime, E: FloatElem> AffineTransform<R, E> {
    /// From a location and a scale, either of which may be a scalar tensor.
    pub fn new(loc: impl Into<Var<R, E>>, scale: impl Into<Var<R, E>>) -> Self {
        Self {
            loc: loc.into(),
            scale: scale.into(),
        }
    }
}

impl<R: Runtime, E: FloatElem> Transform<R, E> for AffineTransform<R, E> {
    fn forward(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        x.mul(&self.scale)?.add(&self.loc)
    }

    fn inverse(&self, y: &Var<R, E>) -> Result<Var<R, E>> {
        y.sub(&self.loc)?.div(&self.scale)
    }

    fn log_abs_det_jacobian(&self, x: &Var<R, E>, _y: &Var<R, E>) -> Result<Var<R, E>> {
        let ln = self.scale.abs().log();
        // Broadcast to the value's shape, so a scalar scale still contributes once
        // per element rather than once.
        ln.expand(x.shape().clone())
    }
}

/// `y = exp(x)`.
pub struct ExpTransform;

impl<R: Runtime, E: FloatElem> Transform<R, E> for ExpTransform {
    fn forward(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(x.exp())
    }

    fn inverse(&self, y: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(y.log())
    }

    fn log_abs_det_jacobian(&self, x: &Var<R, E>, _y: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(x.clone())
    }
}

/// `y = σ(x)`, mapping the line onto `(0, 1)`.
pub struct SigmoidTransform;

impl<R: Runtime, E: FloatElem> Transform<R, E> for SigmoidTransform {
    fn forward(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(x.sigmoid())
    }

    fn inverse(&self, y: &Var<R, E>) -> Result<Var<R, E>> {
        y.log().sub(&y.rsub_scalar(1.0).log())
    }

    fn log_abs_det_jacobian(&self, x: &Var<R, E>, _y: &Var<R, E>) -> Result<Var<R, E>> {
        // `ln σ'(x) = −softplus(−x) − softplus(x)`, which stays finite where
        // `ln(σ(x)(1 − σ(x)))` underflows to `−∞`.
        let a = x.neg().softplus()?;
        let b = x.softplus()?;
        a.add(&b).map(|v| v.neg())
    }
}

/// `y = tanh(x)`, mapping the line onto `(−1, 1)`.
///
/// The squashing every continuous-control policy with bounded actions uses. Its
/// log-Jacobian is written as `2(ln 2 − x − softplus(−2x))` rather than
/// `ln(1 − tanh(x)²)`: the second loses every digit once `|x|` passes about 8, where
/// `tanh(x)²` rounds to exactly one and the logarithm becomes `−∞`. The first is
/// exact out to the end of the float range, and it is the same expression PyTorch
/// settled on for the same reason.
pub struct TanhTransform;

impl<R: Runtime, E: FloatElem> Transform<R, E> for TanhTransform {
    fn forward(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(x.tanh())
    }

    fn inverse(&self, y: &Var<R, E>) -> Result<Var<R, E>> {
        // `atanh(y) = ½ ln((1+y)/(1−y))`.
        let num = y.add_scalar(1.0).log();
        let den = y.rsub_scalar(1.0).log();
        Ok(num.sub(&den)?.mul_scalar(0.5))
    }

    fn log_abs_det_jacobian(&self, x: &Var<R, E>, _y: &Var<R, E>) -> Result<Var<R, E>> {
        let soft = x.mul_scalar(-2.0).softplus()?;
        Ok(x.add(&soft)?
            .rsub_scalar(core::f32::consts::LN_2)
            .mul_scalar(2.0))
    }
}

/// `y = xᵏ` on the positive half-line.
pub struct PowerTransform {
    /// The exponent `k`.
    pub exponent: f32,
}

impl<R: Runtime, E: FloatElem> Transform<R, E> for PowerTransform {
    fn forward(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(x.powf_scalar(self.exponent))
    }

    fn inverse(&self, y: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(y.powf_scalar(1.0 / self.exponent))
    }

    fn log_abs_det_jacobian(&self, x: &Var<R, E>, y: &Var<R, E>) -> Result<Var<R, E>> {
        // `ln|k| + ln|y/x|`, which avoids forming `x^{k−1}` separately.
        Ok(y.div(x)?.abs().log().add_scalar(self.exponent.abs().ln()))
    }
}

/// A base distribution pushed through a chain of transforms.
///
/// The transforms apply in order, so `vec![AffineTransform, TanhTransform]` means
/// `tanh(loc + scale·x)`. A density is carried backwards through the chain: the value
/// is inverted transform by transform, and each step's log-Jacobian is subtracted.
pub struct TransformedDistribution<R: Runtime, E: FloatElem, D> {
    base: D,
    transforms: Vec<Box<dyn Transform<R, E>>>,
}

impl<R: Runtime, E: FloatElem, D> TransformedDistribution<R, E, D> {
    /// Compose `base` with `transforms`, applied left to right.
    pub fn new(base: D, transforms: Vec<Box<dyn Transform<R, E>>>) -> Self {
        Self { base, transforms }
    }

    /// The untransformed distribution.
    pub fn base(&self) -> &D {
        &self.base
    }

    /// Push a value forward through every transform.
    pub fn forward(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        let mut out = x.clone();
        for t in &self.transforms {
            out = t.forward(&out)?;
        }
        Ok(out)
    }
}

impl<R: Runtime, E: FloatElem, D: Distribution<R, E>> Distribution<R, E>
    for TransformedDistribution<R, E, D>
{
    fn batch_shape(&self) -> &Shape {
        self.base.batch_shape()
    }

    fn event_shape(&self) -> Shape {
        self.base.event_shape()
    }

    fn support(&self) -> Support {
        // A transform's image is not tracked; the honest answer is that it is
        // whatever the last transform maps the base's support to.
        Support::Real
    }

    fn has_rsample(&self) -> bool {
        self.base.has_rsample()
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        let x = Var::constant(self.base.sample(seed)?);
        Ok(self.forward(&x)?.into_tensor())
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        let x = Var::constant(self.base.sample_n(n, seed)?);
        Ok(self.forward(&x)?.into_tensor())
    }

    fn rsample(&self, seed: u64) -> Result<Var<R, E>> {
        self.forward(&self.base.rsample(seed)?)
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        // Walk backwards, inverting as we go, so each transform sees the value it
        // itself produced on the forward pass.
        let mut y = value.clone();
        let mut correction: Option<Var<R, E>> = None;
        for t in self.transforms.iter().rev() {
            let x = t.inverse(&y)?;
            let jac = t.log_abs_det_jacobian(&x, &y)?;
            correction = Some(match correction {
                Some(acc) => acc.add(&jac)?,
                None => jac,
            });
            y = x;
        }
        let base = self.base.log_prob(&y)?;
        match correction {
            Some(c) => base.sub(&c),
            None => Ok(base),
        }
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a transformed distribution's CDF is only defined for a monotone chain, \
             which is not tracked here"
                .to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a transformed distribution's quantile is only defined for a monotone \
             chain, which is not tracked here"
                .to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a transformed distribution's entropy has no closed form in general; \
             estimate it from `rsample` and `log_prob`"
                .to_string(),
        ))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a transformed distribution's mean is not the transform of the base's".to_string(),
        ))
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a transformed distribution's variance has no closed form in general".to_string(),
        ))
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        // A monotone transform does move the mode with it, and every transform here
        // is monotone, so this one *is* the transform of the base's.
        let m = Var::constant(self.base.mode()?);
        Ok(self.forward(&m)?.into_tensor())
    }
}

// ---------------------------------------------------------------------------
// Mixtures
// ---------------------------------------------------------------------------

/// A categorical mixture over a batch of components of one family.
///
/// `mixture` is a [`Categorical`] over `k`, and `components` is a distribution whose
/// batch shape ends in that same `k`. The last batch axis of the components is what
/// the mixture indexes, and it disappears from the mixture's own batch shape.
pub struct MixtureSameFamily<R: Runtime, E: FloatElem, D> {
    mixture: Categorical<R, E>,
    components: D,
    batch: Shape,
}

impl<R: Runtime, E: FloatElem, D: Distribution<R, E>> MixtureSameFamily<R, E, D> {
    /// Weight `components` by `mixture`.
    pub fn new(mixture: Categorical<R, E>, components: D) -> Result<Self> {
        let comp = components.batch_shape();
        if comp.rank() == 0 {
            return Err(Error::shape(
                "a mixture's components need a trailing component axis".to_string(),
            ));
        }
        let k = comp.dim_from_end(0);
        if k != mixture.classes() {
            return Err(Error::shape(format!(
                "a {}-way mixture cannot weight {k} components",
                mixture.classes()
            )));
        }
        let batch = comp.without(comp.rank() - 1);
        if mixture.batch_shape() != &batch {
            return Err(Error::shape(format!(
                "the mixture's batch {} does not match the components' {batch}",
                mixture.batch_shape()
            )));
        }
        Ok(Self {
            mixture,
            components,
            batch,
        })
    }

    /// The weighting categorical.
    pub fn mixture(&self) -> &Categorical<R, E> {
        &self.mixture
    }

    /// The components.
    pub fn components(&self) -> &D {
        &self.components
    }
}

impl<R: Runtime, E: FloatElem, D: Distribution<R, E>> Distribution<R, E>
    for MixtureSameFamily<R, E, D>
{
    fn batch_shape(&self) -> &Shape {
        &self.batch
    }

    fn event_shape(&self) -> Shape {
        self.components.event_shape()
    }

    fn support(&self) -> Support {
        self.components.support()
    }

    fn has_rsample(&self) -> bool {
        false
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        // Draw every component, then keep the one the mixture chose. Drawing all `k`
        // and discarding `k − 1` sounds wasteful and is not: the alternative is a
        // gather whose indices are data, which costs a launch of its own and breaks
        // the property that a draw depends only on its own index.
        let all = self.components.sample(seed)?;
        let ids = self
            .mixture
            .sample_ids(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5DEE_CE66)?;
        let k = self.mixture.classes();
        let picked = crate::tensor::ops::index::one_hot::<R, E>(&ids, k)?;
        let weighted = crate::tensor::ops::elemwise::mul(&all, &picked)?;
        let axis = weighted.rank() - 1;
        crate::tensor::ops::reduce::sum_dim(&weighted, axis)?.squeeze(axis)
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
        dims.extend_from_slice(self.batch.dims());
        crate::tensor::ops::movement::cat(&parts, 0)?.reshape(Shape::new(dims))
    }

    fn rsample(&self, _seed: u64) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "which component a mixture draw came from is discrete".to_string(),
        ))
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        // `ln Σ_k π_k p_k(x)`, in log space: broadcast the value across components,
        // score it under each, add the log-weights, and reduce with a logsumexp.
        let axis = value.rank();
        let spread = value
            .unsqueeze(axis)?
            .expand(self.components.batch_shape().clone())?;
        let per_component = self.components.log_prob(&spread)?;
        let weights = self
            .mixture
            .logits()
            .log_softmax(self.mixture.logits().rank() - 1)?;
        let joint = per_component.add(&weights)?;
        let top = joint.max_dim(axis)?;
        joint
            .sub(&top)?
            .exp()
            .sum_dim(axis)?
            .log()
            .add(&top)?
            .squeeze(axis)
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a mixture CDF would need the components' CDFs, which are not required \
             by the trait bound here"
                .to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a mixture has no closed-form quantile".to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a mixture has no closed-form entropy; estimate it from samples".to_string(),
        ))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        let per = self.components.mean()?;
        let probs = self.mixture.probs()?;
        let axis = per.rank() - 1;
        crate::tensor::ops::reduce::sum_dim(
            &crate::tensor::ops::elemwise::mul(&per, &probs)?,
            axis,
        )?
        .squeeze(axis)
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        // The law of total variance: the mean of the variances plus the variance of
        // the means.
        let probs = self.mixture.probs()?;
        let per_var = self.components.variance()?;
        let per_mean = self.components.mean()?;
        let axis = per_mean.rank() - 1;
        let mean = self.mean()?;
        let centred = crate::tensor::ops::elemwise::sub(&per_mean, &mean.unsqueeze(axis)?)?;
        let spread = crate::tensor::ops::elemwise::mul(&centred, &centred)?;
        let total = crate::tensor::ops::elemwise::add(&per_var, &spread)?;
        crate::tensor::ops::reduce::sum_dim(
            &crate::tensor::ops::elemwise::mul(&total, &probs)?,
            axis,
        )?
        .squeeze(axis)
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a mixture's mode has no closed form".to_string(),
        ))
    }
}
