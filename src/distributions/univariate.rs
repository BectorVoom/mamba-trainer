//! The scalar mathematics of every univariate distribution, in one place.
//!
//! Each of the seven things a distribution can be asked for — its log-density, its
//! CDF, its quantile, its entropy, its first two moments, its mode, and a draw from
//! it — is **one function** here that switches on a [`Kind`] at compile time. There
//! is no run-time branch in the generated code: `kind` is `#[comptime]`, so CubeCL
//! prunes every arm but one and each launch compiles to a kernel that knows only its
//! own distribution.
//!
//! # Why one function and not twenty-six
//!
//! Because the alternative is twenty-six copies of the same launch plumbing, the
//! same broadcast handling and the same `f32` casting, differing in four lines of
//! algebra each — and because a bug in the plumbing would then have to be found
//! twenty-six times. The `if comptime!(kind == …)` chain reads like a table of
//! densities, which is what it is.
//!
//! # One body, two compilations
//!
//! As in [`super::special`], the marked region below is compiled twice: once by
//! `#[cube]` and once by `rustc`, the second copy living in `univariate_host.rs`.
//! Three mechanical edits separate them — drop the `#[cube]` lines, drop the
//! `#[comptime] ` markers, and turn `comptime!(e)` into `(e)` — and
//! `distributions_bitexact::host_twin_is_the_same_source` re-derives the file and
//! fails if it has drifted. The two differ in one further place, outside the region:
//! their `use` lines, which point `rng` and `special` at the device functions in one
//! copy and at the host twins in the other.
//!
//! That the *samplers* are shared too is the part that matters. A rejection sampler
//! is a loop whose trip count depends on the draws it makes, so "the host agrees
//! with the device" is a statement about control flow as much as arithmetic, and it
//! is only worth making if both sides are running the same program.
//!
//! # Conventions
//!
//! * **Three parameter slots.** Every distribution reads `a`, `b` and `c`; most
//!   ignore the third and many the second. [`Kind::arity`] says how many are live,
//!   and the table on [`Kind`] says what each one means.
//! * **Logits, not probabilities.** Anything with a Bernoulli-shaped parameter takes
//!   it as a logit, because `log(1 − p)` computed from a `p` near one has no digits
//!   left and `−softplus(logit)` has all of them. A caller who has probabilities
//!   converts once, at construction.
//! * **`f32` throughout.** Storage may be `f16` or `bf16`; the arithmetic is not.
//!   See [`super::special`] for why.
//! * **Loop-carried assignment.** Inside a loop, a variable the loop carries is
//!   only ever assigned from an `if` that has an `else`, or through a fresh local
//!   declared in the body. An `if` without an `else` writing to a loop-carried
//!   variable is the one construction CubeCL's CPU backend miscompiles — the
//!   generated MLIR fails its own dominance check — and it fails at launch, not at
//!   build.
//! * **PyTorch's formulas.** Where `torch.distributions` picks one of several
//!   algebraically equal spellings, this picks the same one, so that a model ported
//!   across gets the same numbers. The two deliberate exceptions are noted where
//!   they occur — both are places where PyTorch's spelling loses the tail.

// Two lints that a `#[cube]` function's shape guarantees will fire. A branching
// value must be a `let mut` initialised before the branch, so the initialiser is
// dead; and every function here takes all three parameter slots whether or not the
// distribution it was specialised to reads them.
#![allow(unused_assignments, unused_variables)]
// Every `#[cube] pub fn` expands to a public module of the same name holding the
// macro's generated `expand` entry points. There is no way to attach documentation
// to a module a proc macro synthesises, so `missing_docs` fires on each of them and
// there is nothing to say in reply. Every item written by hand in this file is
// documented; the allow is for the ones that are not.
#![allow(missing_docs)]

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d};
use crate::error::Result;
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

use super::rng;
use super::special;

#[path = "univariate_host.rs"]
pub mod host;

/// A partial derivative with respect to the value and the three parameter slots.
///
/// Returned whole rather than as four calls because the four share almost all of
/// their work — a Beta's `∂/∂a` and `∂/∂b` differ in two digammas out of five terms
/// — and because the backward pass wants all of them at once anyway.
#[derive(CubeType, Clone, Copy, Debug)]
pub struct Grad4 {
    /// `∂/∂x`, the derivative with respect to the value.
    pub dx: f32,
    /// `∂/∂a`.
    pub da: f32,
    /// `∂/∂b`.
    pub db: f32,
    /// `∂/∂c`.
    pub dc: f32,
}

/// Infinity and NaN, for the results that are genuinely non-finite.
///
/// Passed in rather than written as `f32::INFINITY` and `f32::NAN`, because a
/// kernel cannot spell either: CubeCL emits a float constant as `f32(<value>)`,
/// and `f32(inf)` and `f32(NaN)` are not WGSL. On wgpu such a kernel fails to
/// compile — which [`crate::backend::check_launches`] now reports, and before it
/// every continuous distribution silently returned zeros there. A kernel instead
/// reads both from its scalar arguments, which no compiler can fold into a
/// literal, and the host twin passes [`NonFinite::HOST`].
#[derive(CubeType, Clone, Copy, Debug)]
pub struct NonFinite {
    /// `+∞`. Negated where `−∞` is meant.
    pub inf: f32,
    /// A quiet NaN.
    pub nan: f32,
}

impl NonFinite {
    /// The values themselves, for host code.
    pub const HOST: NonFinite = NonFinite {
        inf: f32::INFINITY,
        nan: f32::NAN,
    };
}

/// Which distribution a kernel is specialised to.
///
/// The discriminant is what the kernels switch on, so it is part of the compiled
/// kernel's identity; reordering this enum changes which cached kernel a launch
/// finds, and nothing else.
///
/// | kind | `a` | `b` | `c` |
/// |---|---|---|---|
/// | [`Normal`](Kind::Normal) | loc | scale | |
/// | [`Uniform`](Kind::Uniform) | low | high | |
/// | [`Exponential`](Kind::Exponential) | rate | | |
/// | [`Laplace`](Kind::Laplace) | loc | scale | |
/// | [`Cauchy`](Kind::Cauchy) | loc | scale | |
/// | [`Gumbel`](Kind::Gumbel) | loc | scale | |
/// | [`HalfNormal`](Kind::HalfNormal) | scale | | |
/// | [`HalfCauchy`](Kind::HalfCauchy) | scale | | |
/// | [`LogNormal`](Kind::LogNormal) | loc | scale | |
/// | [`Pareto`](Kind::Pareto) | scale | alpha | |
/// | [`Weibull`](Kind::Weibull) | scale | concentration | |
/// | [`Kumaraswamy`](Kind::Kumaraswamy) | concentration1 | concentration0 | |
/// | [`Gamma`](Kind::Gamma) | concentration | rate | |
/// | [`InverseGamma`](Kind::InverseGamma) | concentration | rate | |
/// | [`Beta`](Kind::Beta) | concentration1 | concentration0 | |
/// | [`StudentT`](Kind::StudentT) | df | loc | scale |
/// | [`FisherSnedecor`](Kind::FisherSnedecor) | df1 | df2 | |
/// | [`VonMises`](Kind::VonMises) | loc | concentration | |
/// | [`ContinuousBernoulli`](Kind::ContinuousBernoulli) | logits | | |
/// | [`Bernoulli`](Kind::Bernoulli) | logits | | |
/// | [`Geometric`](Kind::Geometric) | logits | | |
/// | [`Poisson`](Kind::Poisson) | rate | | |
/// | [`Binomial`](Kind::Binomial) | total_count | logits | |
/// | [`NegativeBinomial`](Kind::NegativeBinomial) | total_count | logits | |
/// | [`LogitRelaxedBernoulli`](Kind::LogitRelaxedBernoulli) | temperature | logits | |
/// | [`RelaxedBernoulli`](Kind::RelaxedBernoulli) | temperature | logits | |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Kind {
    /// The Gaussian.
    Normal = 0,
    /// Flat on `[low, high)`.
    Uniform = 1,
    /// The waiting time of a Poisson process.
    Exponential = 2,
    /// Two exponential tails back to back.
    Laplace = 3,
    /// The ratio of two standard normals; no mean.
    Cauchy = 4,
    /// The limit of a maximum; the noise behind the Gumbel-max trick.
    Gumbel = 5,
    /// The absolute value of a centred Gaussian.
    HalfNormal = 6,
    /// The absolute value of a centred Cauchy.
    HalfCauchy = 7,
    /// A Gaussian in the log.
    LogNormal = 8,
    /// A power-law tail above a scale.
    Pareto = 9,
    /// The Weibull, a stretched exponential.
    Weibull = 10,
    /// A Beta-shaped density on the unit interval with a closed-form quantile.
    Kumaraswamy = 11,
    /// The Gamma, in the shape-and-rate parameterisation.
    Gamma = 12,
    /// The reciprocal of a Gamma.
    InverseGamma = 13,
    /// The Beta, the conjugate of a Bernoulli.
    Beta = 14,
    /// Student's t, with location and scale.
    StudentT = 15,
    /// The ratio of two scaled chi-squares.
    FisherSnedecor = 16,
    /// The circular analogue of a Gaussian.
    VonMises = 17,
    /// A continuous relaxation of a Bernoulli, on `[0, 1]`.
    ContinuousBernoulli = 18,
    /// A single coin flip.
    Bernoulli = 19,
    /// Failures before the first success.
    Geometric = 20,
    /// Counts from a Poisson process.
    Poisson = 21,
    /// Successes in a fixed number of trials.
    Binomial = 22,
    /// Failures before a fixed number of successes.
    NegativeBinomial = 23,
    /// The pre-sigmoid Gumbel-softmax of a single bit.
    LogitRelaxedBernoulli = 24,
    /// [`LogitRelaxedBernoulli`](Kind::LogitRelaxedBernoulli) pushed through a
    /// sigmoid.
    RelaxedBernoulli = 25,
}

impl Kind {
    /// The discriminant the kernels switch on.
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// How many of the three parameter slots this distribution reads.
    pub const fn arity(self) -> usize {
        match self {
            Kind::Exponential
            | Kind::HalfNormal
            | Kind::HalfCauchy
            | Kind::ContinuousBernoulli
            | Kind::Bernoulli
            | Kind::Geometric
            | Kind::Poisson => 1,
            Kind::StudentT => 3,
            _ => 2,
        }
    }

    /// Whether a draw from this distribution is a differentiable function of its
    /// parameters — PyTorch's `has_rsample`.
    ///
    /// True exactly when the sampler is an inverse-CDF or location-scale transform
    /// of a draw that does not itself depend on the parameters. The rejection
    /// samplers are not: their *control flow* depends on the parameters, so there is
    /// no path derivative to take. (PyTorch reparameterises Gamma anyway, through an
    /// implicit-differentiation trick that is a different thing from a path
    /// derivative; that is not done here, and [`Kind::Gamma`] reports `false`.)
    pub const fn reparameterised(self) -> bool {
        matches!(
            self,
            Kind::Normal
                | Kind::Uniform
                | Kind::Exponential
                | Kind::Laplace
                | Kind::Cauchy
                | Kind::Gumbel
                | Kind::HalfNormal
                | Kind::HalfCauchy
                | Kind::LogNormal
                | Kind::Pareto
                | Kind::Weibull
                | Kind::Kumaraswamy
                | Kind::LogitRelaxedBernoulli
                | Kind::RelaxedBernoulli
        )
    }

    /// Whether [`cdf_of()`] has a closed form for this kind.
    ///
    /// The set is PyTorch's, plus the three discrete families whose CDF is a step,
    /// a geometric series and an incomplete gamma respectively — all cheap, all
    /// useful, and all left unimplemented upstream.
    pub const fn has_cdf(self) -> bool {
        !matches!(
            self,
            Kind::Beta
                | Kind::StudentT
                | Kind::FisherSnedecor
                | Kind::VonMises
                | Kind::Binomial
                | Kind::NegativeBinomial
                | Kind::LogitRelaxedBernoulli
                | Kind::RelaxedBernoulli
        )
    }

    /// Whether [`icdf_of()`] has a closed form for this kind.
    pub const fn has_icdf(self) -> bool {
        matches!(
            self,
            Kind::Normal
                | Kind::Uniform
                | Kind::Exponential
                | Kind::Laplace
                | Kind::Cauchy
                | Kind::Gumbel
                | Kind::HalfNormal
                | Kind::HalfCauchy
                | Kind::LogNormal
                | Kind::Pareto
                | Kind::Weibull
                | Kind::Kumaraswamy
                | Kind::ContinuousBernoulli
                | Kind::Bernoulli
                | Kind::Geometric
        )
    }

    /// [`Kind::has_icdf`] for a kind given by its [`Kind::code`], as the kernels
    /// see it; `false` for a code that names no kind.
    pub const fn code_has_icdf(code: u32) -> bool {
        let mut i = 0;
        while i < Kind::ALL.len() {
            if Kind::ALL[i].code() == code {
                return Kind::ALL[i].has_icdf();
            }
            i += 1;
        }
        false
    }

    /// Whether [`entropy_of()`] has a closed form for this kind.
    pub const fn has_entropy(self) -> bool {
        !matches!(
            self,
            Kind::FisherSnedecor
                | Kind::ContinuousBernoulli
                | Kind::Poisson
                | Kind::Binomial
                | Kind::NegativeBinomial
                | Kind::LogitRelaxedBernoulli
                | Kind::RelaxedBernoulli
        )
    }

    /// Whether [`icdf_grad_of()`] fills in a gradient for this kind, which is also
    /// what makes a quantile differentiable.
    pub const fn has_icdf_grad(self) -> bool {
        (self.code() <= Kind::Kumaraswamy.code()) && self.has_icdf()
    }

    /// Whether one draw costs exactly one uniform, so four consecutive elements can
    /// share a single Philox evaluation.
    ///
    /// The inverse-CDF families and the three that compare or transform a single
    /// draw. The rejection samplers cannot: their consumption depends on how many
    /// proposals they reject, which is data.
    pub const fn lane_sampled(self) -> bool {
        self.code() <= Kind::Kumaraswamy.code()
            || matches!(
                self,
                Kind::ContinuousBernoulli
                    | Kind::Bernoulli
                    | Kind::Geometric
                    | Kind::LogitRelaxedBernoulli
                    | Kind::RelaxedBernoulli
            )
    }

    /// Every kind, for tests and for iterating the table.
    pub const ALL: [Kind; 26] = [
        Kind::Normal,
        Kind::Uniform,
        Kind::Exponential,
        Kind::Laplace,
        Kind::Cauchy,
        Kind::Gumbel,
        Kind::HalfNormal,
        Kind::HalfCauchy,
        Kind::LogNormal,
        Kind::Pareto,
        Kind::Weibull,
        Kind::Kumaraswamy,
        Kind::Gamma,
        Kind::InverseGamma,
        Kind::Beta,
        Kind::StudentT,
        Kind::FisherSnedecor,
        Kind::VonMises,
        Kind::ContinuousBernoulli,
        Kind::Bernoulli,
        Kind::Geometric,
        Kind::Poisson,
        Kind::Binomial,
        Kind::NegativeBinomial,
        Kind::LogitRelaxedBernoulli,
        Kind::RelaxedBernoulli,
    ];
}

/// Which summary [`moment_of()`] should return.
pub mod moment {
    /// The mean.
    pub const MEAN: u32 = 0;
    /// The variance.
    pub const VARIANCE: u32 = 1;
    /// The mode.
    pub const MODE: u32 = 2;
}

/// Streams per sub-draw, and so the iteration cap of every rejection loop here.
///
/// A rejection sampler's `i`-th attempt reads stream `base + i`, and a distribution
/// that needs two independent draws — a Beta needs two Gammas — gives the second a
/// base of `SUB_DRAW`. Capping the loop is what makes the kernel terminate on a
/// device that cannot be interrupted; the caps below are chosen so the probability
/// of reaching one is smaller than the probability of the machine miscomputing.
pub const SUB_DRAW: u32 = 64;

/// The upper bound on rejection attempts. See [`SUB_DRAW`].
pub const MAX_TRIES: u32 = 60;

// The region below is the single source of truth for the scalar mathematics of every
// univariate distribution. `univariate_host.rs` is generated from it; see the module
// docs.
//
// >>> shared

/// The log-density (or log-mass) of `kind` at `x`.
#[cube]
pub fn log_prob_of(x: f32, a: f32, b: f32, c: f32, nf: NonFinite, #[comptime] kind: u32) -> f32 {
    let mut out: f32 = 0.0;
    if comptime!(kind == 0) {
        // Normal(loc = a, scale = b).
        let z = (x - a) / b;
        out = -0.5f32 * z * z - f32::ln(b) - special::HALF_LN_2PI;
    } else if comptime!(kind == 1) {
        // Uniform(low = a, high = b): flat inside, impossible outside.
        out = -nf.inf;
        if x >= a && x < b {
            out = -f32::ln(b - a);
        }
    } else if comptime!(kind == 2) {
        // Exponential(rate = a).
        out = f32::ln(a) - a * x;
    } else if comptime!(kind == 3) {
        // Laplace(loc = a, scale = b).
        out = -f32::ln(2.0f32 * b) - f32::abs(x - a) / b;
    } else if comptime!(kind == 4) {
        // Cauchy(loc = a, scale = b).
        let z = (x - a) / b;
        out = -f32::ln(special::PI) - f32::ln(b) - special::log1p_f32(z * z);
    } else if comptime!(kind == 5) {
        // Gumbel(loc = a, scale = b).
        let z = (x - a) / b;
        out = -(z + f32::exp(-z)) - f32::ln(b);
    } else if comptime!(kind == 6) {
        // HalfNormal(scale = a): the Gaussian folded at zero, so twice the density.
        let z = x / a;
        out = -nf.inf;
        if x >= 0.0f32 {
            out = -0.5f32 * z * z - f32::ln(a) - special::HALF_LN_2PI + special::LN_2;
        }
    } else if comptime!(kind == 7) {
        // HalfCauchy(scale = a).
        let z = x / a;
        out = -nf.inf;
        if x >= 0.0f32 {
            out = special::LN_2 - f32::ln(special::PI) - f32::ln(a) - special::log1p_f32(z * z);
        }
    } else if comptime!(kind == 8) {
        // LogNormal(loc = a, scale = b): a Normal in ln x, minus the Jacobian.
        let l = f32::ln(x);
        let z = (l - a) / b;
        out = -0.5f32 * z * z - f32::ln(b) - special::HALF_LN_2PI - l;
    } else if comptime!(kind == 9) {
        // Pareto(scale = a, alpha = b).
        out = f32::ln(b) + b * f32::ln(a) - (b + 1.0f32) * f32::ln(x);
    } else if comptime!(kind == 10) {
        // Weibull(scale = a, concentration = b).
        let z = x / a;
        out = f32::ln(b) - f32::ln(a) + (b - 1.0f32) * f32::ln(z) - f32::powf(z, b);
    } else if comptime!(kind == 11) {
        // Kumaraswamy(a, b): a Beta-shaped density whose CDF is elementary.
        let xa = f32::powf(x, a);
        out = f32::ln(a)
            + f32::ln(b)
            + (a - 1.0f32) * f32::ln(x)
            + special::xlog1py_f32(b - 1.0f32, -xa);
    } else if comptime!(kind == 12) {
        // Gamma(concentration = a, rate = b).
        out = special::xlogy_f32(a, b) + (a - 1.0f32) * f32::ln(x) - b * x - special::lgamma_f32(a);
    } else if comptime!(kind == 13) {
        // InverseGamma(concentration = a, rate = b).
        out = a * f32::ln(b) - special::lgamma_f32(a) - (a + 1.0f32) * f32::ln(x) - b / x;
    } else if comptime!(kind == 14) {
        // Beta(concentration1 = a, concentration0 = b).
        out = special::xlogy_f32(a - 1.0f32, x) + special::xlog1py_f32(b - 1.0f32, -x)
            - special::lbeta_f32(a, b);
    } else if comptime!(kind == 15) {
        // StudentT(df = a, loc = b, scale = c).
        let z = (x - b) / c;
        let half = 0.5f32 * a;
        out = special::lgamma_f32(half + 0.5f32)
            - special::lgamma_f32(half)
            - 0.5f32 * f32::ln(a * special::PI)
            - f32::ln(c)
            - (half + 0.5f32) * special::log1p_f32(z * z / a);
    } else if comptime!(kind == 16) {
        // FisherSnedecor(df1 = a, df2 = b), spelled as PyTorch spells it.
        let h1 = 0.5f32 * a;
        let h2 = 0.5f32 * b;
        let ratio = a / b;
        out = special::lgamma_f32(h1 + h2) - special::lgamma_f32(h1) - special::lgamma_f32(h2)
            + h1 * f32::ln(ratio)
            + (h1 - 1.0f32) * f32::ln(x)
            - (h1 + h2) * special::log1p_f32(ratio * x);
    } else if comptime!(kind == 17) {
        // VonMises(loc = a, concentration = b).
        out = b * f32::cos(x - a) - f32::ln(2.0f32 * special::PI) - special::log_i0_f32(b);
    } else if comptime!(kind == 18) {
        // ContinuousBernoulli(logits = a) on [0, 1].
        out = x * a - special::softplus_f32(a) + cont_bernoulli_log_norm(a);
    } else if comptime!(kind == 19) {
        // Bernoulli(logits = a): `x·logit − softplus(logit)`, which is the negative
        // binary cross entropy and is finite at every logit, unlike `x ln p`.
        out = x * a - special::softplus_f32(a);
    } else if comptime!(kind == 20) {
        // Geometric(logits = a), counting failures before the first success.
        out = x * special::log_sigmoid_f32(-a) + special::log_sigmoid_f32(a);
    } else if comptime!(kind == 21) {
        // Poisson(rate = a).
        out = special::xlogy_f32(x, a) - a - special::lgamma_f32(x + 1.0f32);
    } else if comptime!(kind == 22) {
        // Binomial(total_count = a, logits = b).
        out = special::log_binom_f32(a, x) + x * b - a * special::softplus_f32(b);
    } else if comptime!(kind == 23) {
        // NegativeBinomial(total_count = a, logits = b).
        out = a * special::log_sigmoid_f32(-b)
            + x * special::log_sigmoid_f32(b)
            + special::lgamma_f32(a + x)
            - special::lgamma_f32(1.0f32 + x)
            - special::lgamma_f32(a);
    } else if comptime!(kind == 24) {
        // LogitRelaxedBernoulli(temperature = a, logits = b).
        let diff = b - x * a;
        out = f32::ln(a) + diff - 2.0f32 * special::softplus_f32(diff);
    } else if comptime!(kind == 25) {
        // RelaxedBernoulli(temperature = a, logits = b): the logit-relaxed density
        // pushed through a sigmoid, so `x = σ(y)` and `dy/dx = 1/(x(1−x))`.
        let y = f32::ln(x) - special::log1p_f32(-x);
        let diff = b - y * a;
        out = f32::ln(a) + diff
            - 2.0f32 * special::softplus_f32(diff)
            - f32::ln(x)
            - special::log1p_f32(-x);
    }
    out
}

/// The mean of a continuous Bernoulli with natural parameter (logit) `t`.
///
/// Its own function because two callers need it: [`moment_of()`] and the score in
/// [`log_prob_grad_of()`], which is the value minus this mean.
#[cube]
pub fn cont_bernoulli_mean(t: f32) -> f32 {
    let mut out: f32 = 0.0;
    if f32::abs(t) > 0.02f32 {
        out = 1.0f32 / (-special::expm1_f32(-t)) - 1.0f32 / t;
    } else {
        out = 0.5f32 + t * (1.0f32 / 12.0f32 - t * t / 720.0f32);
    }
    out
}

/// `ln C(λ)`, the normaliser of a continuous Bernoulli with logit `t`.
///
/// `C(λ) = 2 tanh⁻¹(1 − 2λ)/(1 − 2λ)` has a removable singularity at `λ = ½`, where
/// both halves vanish; away from it the closed form is stable, and within a
/// thousandth of it the fourth-order Taylor series is better than the cancellation
/// would be. The switch is PyTorch's, and so is the series.
#[cube]
pub fn cont_bernoulli_log_norm(logits: f32) -> f32 {
    let p = 1.0f32 / (1.0f32 + f32::exp(-logits));
    let d = p - 0.5f32;
    let mut out: f32 = 0.0;
    if f32::abs(d) > 1.0e-3f32 {
        out = f32::ln(f32::abs(f32::ln(1.0f32 - p) - f32::ln(p)))
            - f32::ln(f32::abs(1.0f32 - 2.0f32 * p));
    } else {
        let d2 = d * d;
        out = special::LN_2 + (4.0f32 / 3.0f32 + 104.0f32 / 45.0f32 * d2) * d2;
    }
    out
}

/// The regularized lower incomplete gamma function `P(a, x)`.
///
/// Numerical Recipes' pairing: the ascending series where it converges quickly
/// (`x < a + 1`) and Lentz's continued fraction for the complement everywhere else.
/// Neither converges well in the other's half, which is why both are here.
///
/// This is the only place a distribution CDF needs an iteration rather than a
/// formula, and it is what makes [`Kind::Gamma`], [`Kind::InverseGamma`] and
/// [`Kind::Poisson`] answerable at all.
#[cube]
pub fn gammainc_p(a: f32, x: f32) -> f32 {
    let mut out: f32 = 0.0;
    if x > 0.0f32 {
        // The factor both branches share, formed in the log so that a large `a`
        // does not overflow on the way to a result in `[0, 1]`.
        let scale = f32::exp(a * f32::ln(x) - x - special::lgamma_f32(a));
        if x < a + 1.0f32 {
            let mut term = 1.0f32 / a;
            let mut sum = term;
            let mut n: u32 = 1;
            while n < 300u32 {
                term = term * x / (a + n as f32);
                sum += term;
                if f32::abs(term) < f32::abs(sum) * 1.0e-8f32 {
                    n = 300u32;
                } else {
                    n += 1u32;
                }
            }
            out = sum * scale;
        } else {
            // Lentz's algorithm for the continued fraction of the upper tail. The
            // two `tiny` guards write into locals rather than straight into `dj`
            // and `cj`: an `if` without an `else` that assigns to a variable the
            // loop carries is the one shape CubeCL's CPU backend miscompiles, and
            // it does so silently.
            let tiny = 1.0e-30f32;
            let mut bj = x + 1.0f32 - a;
            let mut cj = 1.0e30f32;
            let mut dj = 1.0f32 / bj;
            let mut h = dj;
            let mut i: u32 = 1;
            while i < 300u32 {
                let fi = i as f32;
                let an = fi * (a - fi);
                bj += 2.0f32;
                let mut den = an * dj + bj;
                if f32::abs(den) < tiny {
                    den = tiny;
                }
                let mut num = bj + an / cj;
                if f32::abs(num) < tiny {
                    num = tiny;
                }
                cj = num;
                dj = 1.0f32 / den;
                let delta = dj * cj;
                h *= delta;
                if f32::abs(delta - 1.0f32) < 1.0e-8f32 {
                    i = 300u32;
                } else {
                    i += 1u32;
                }
            }
            out = 1.0f32 - scale * h;
        }
    }
    out
}

/// The cumulative distribution function of `kind` at `x`.
///
/// `NaN` where the distribution has no closed-form CDF; the host layer refuses those
/// before it ever launches, so the value is never observed.
#[cube]
pub fn cdf_of(x: f32, a: f32, b: f32, c: f32, nf: NonFinite, #[comptime] kind: u32) -> f32 {
    let mut out: f32 = 0.0;
    out = nf.nan;
    if comptime!(kind == 0) {
        out = special::std_normal_cdf_f32((x - a) / b);
    } else if comptime!(kind == 1) {
        out = ((x - a) / (b - a)).clamp(0.0f32, 1.0f32);
    } else if comptime!(kind == 2) {
        out = -special::expm1_f32(-a * x);
    } else if comptime!(kind == 3) {
        // Laplace: written with `expm1` on the shared exponential tail so that the
        // near half of the distribution does not cancel against the leading half.
        let z = (x - a) / b;
        out = 0.5f32 - 0.5f32 * special::sign_f32(z) * special::expm1_f32(-f32::abs(z));
    } else if comptime!(kind == 4) {
        out = f32::atan((x - a) / b) / special::PI + 0.5f32;
    } else if comptime!(kind == 5) {
        out = f32::exp(-f32::exp(-(x - a) / b));
    } else if comptime!(kind == 6) {
        out = 0.0f32;
        if x > 0.0f32 {
            out = special::erf_f32(x / (a * special::SQRT_2));
        }
    } else if comptime!(kind == 7) {
        out = 0.0f32;
        if x > 0.0f32 {
            out = 2.0f32 / special::PI * f32::atan(x / a);
        }
    } else if comptime!(kind == 8) {
        out = 0.0f32;
        if x > 0.0f32 {
            out = special::std_normal_cdf_f32((f32::ln(x) - a) / b);
        }
    } else if comptime!(kind == 9) {
        out = 0.0f32;
        if x > a {
            out = -special::expm1_f32(b * (f32::ln(a) - f32::ln(x)));
        }
    } else if comptime!(kind == 10) {
        out = 0.0f32;
        if x > 0.0f32 {
            out = -special::expm1_f32(-f32::powf(x / a, b));
        }
    } else if comptime!(kind == 11) {
        out = 0.0f32;
        if x > 0.0f32 {
            out = -special::expm1_f32(b * special::log1p_f32(-f32::powf(x, a)));
        }
    } else if comptime!(kind == 12) {
        out = gammainc_p(a, b * x);
    } else if comptime!(kind == 13) {
        // 1/X with X ~ Gamma, so the CDF is the Gamma's upper tail at `rate/x`.
        out = 0.0f32;
        if x > 0.0f32 {
            out = 1.0f32 - gammainc_p(a, b / x);
        }
    } else if comptime!(kind == 18) {
        // ContinuousBernoulli, PyTorch's closed form, clamped to the unit interval.
        let p = 1.0f32 / (1.0f32 + f32::exp(-a));
        let t = x.clamp(0.0f32, 1.0f32);
        let mut v = t;
        if f32::abs(p - 0.5f32) > 1.0e-3f32 {
            v = (f32::powf(p, t) * f32::powf(1.0f32 - p, 1.0f32 - t) + p - 1.0f32)
                / (2.0f32 * p - 1.0f32);
        }
        out = v.clamp(0.0f32, 1.0f32);
    } else if comptime!(kind == 19) {
        // Bernoulli. PyTorch leaves this unimplemented; it is two lines and a step
        // function is a perfectly good CDF.
        out = 1.0f32;
        if x < 0.0f32 {
            out = 0.0f32;
        } else if x < 1.0f32 {
            out = 1.0f32 / (1.0f32 + f32::exp(a));
        }
    } else if comptime!(kind == 20) {
        // Geometric: `1 − (1−p)^(⌊x⌋+1)`.
        out = 0.0f32;
        if x >= 0.0f32 {
            out = -special::expm1_f32((f32::floor(x) + 1.0f32) * special::log_sigmoid_f32(-a));
        }
    } else if comptime!(kind == 21) {
        // Poisson: the upper regularized incomplete gamma at `⌊x⌋ + 1`.
        out = 0.0f32;
        if x >= 0.0f32 {
            out = 1.0f32 - gammainc_p(f32::floor(x) + 1.0f32, a);
        }
    }
    out
}

/// The quantile function of `kind` at `q ∈ (0, 1)`.
///
/// `NaN` where there is no closed form, as in [`cdf_of()`].
#[cube]
pub fn icdf_of(q: f32, a: f32, b: f32, c: f32, nf: NonFinite, #[comptime] kind: u32) -> f32 {
    let mut out: f32 = 0.0;
    out = nf.nan;
    if comptime!(Kind::code_has_icdf(kind)) {
        out = closed_form_icdf_of(q, a, b, c, kind);
    }
    out
}

/// [`icdf_of()`] for a kind that has a closed form, and `0` for one that does not.
///
/// Split out so the samplers, which only ever invert kinds that have one, need no
/// [`NonFinite`] to reach it.
#[cube]
pub fn closed_form_icdf_of(q: f32, a: f32, b: f32, c: f32, #[comptime] kind: u32) -> f32 {
    let mut out: f32 = 0.0;
    if comptime!(kind == 0) {
        out = a + b * special::std_normal_icdf_f32(q);
    } else if comptime!(kind == 1) {
        out = a + q * (b - a);
    } else if comptime!(kind == 2) {
        out = -special::log1p_f32(-q) / a;
    } else if comptime!(kind == 3) {
        let d = q - 0.5f32;
        out = a - b * special::sign_f32(d) * special::log1p_f32(-2.0f32 * f32::abs(d));
    } else if comptime!(kind == 4) {
        out = a + b * f32::tan(special::PI * (q - 0.5f32));
    } else if comptime!(kind == 5) {
        out = a - b * f32::ln(-f32::ln(q));
    } else if comptime!(kind == 6) {
        out = a * special::SQRT_2 * special::erfinv_f32(q);
    } else if comptime!(kind == 7) {
        out = a * f32::tan(0.5f32 * special::PI * q);
    } else if comptime!(kind == 8) {
        out = f32::exp(a + b * special::std_normal_icdf_f32(q));
    } else if comptime!(kind == 9) {
        out = a * f32::exp(-special::log1p_f32(-q) / b);
    } else if comptime!(kind == 10) {
        out = a * f32::powf(-special::log1p_f32(-q), 1.0f32 / b);
    } else if comptime!(kind == 11) {
        out = f32::powf(-special::expm1_f32(special::log1p_f32(-q) / b), 1.0f32 / a);
    } else if comptime!(kind == 18) {
        // ContinuousBernoulli, PyTorch's closed form with its `p = ½` limit.
        let p = 1.0f32 / (1.0f32 + f32::exp(-a));
        let mut v = q;
        if f32::abs(p - 0.5f32) > 1.0e-3f32 {
            let r = 1.0f32 - p;
            v = (special::log1p_f32(q * (2.0f32 * p - 1.0f32) / r)) / (f32::ln(p) - f32::ln(r));
        }
        out = v.clamp(0.0f32, 1.0f32);
    } else if comptime!(kind == 19) {
        out = 0.0f32;
        if q > 1.0f32 / (1.0f32 + f32::exp(a)) {
            out = 1.0f32;
        }
    } else if comptime!(kind == 20) {
        out = f32::floor(special::log1p_f32(-q) / special::log_sigmoid_f32(-a));
    }
    out
}

/// The differential (or Shannon) entropy of `kind`.
///
/// `NaN` for the distributions PyTorch leaves unimplemented, plus the three relaxed
/// ones, whose entropies have no elementary closed form. The host layer refuses
/// those before launching.
#[cube]
pub fn entropy_of(a: f32, b: f32, c: f32, nf: NonFinite, #[comptime] kind: u32) -> f32 {
    let mut out: f32 = 0.0;
    out = nf.nan;
    if comptime!(kind == 0) {
        out = f32::ln(b) + special::HALF_LN_2PI + 0.5f32;
    } else if comptime!(kind == 1) {
        out = f32::ln(b - a);
    } else if comptime!(kind == 2) {
        out = 1.0f32 - f32::ln(a);
    } else if comptime!(kind == 3) {
        out = 1.0f32 + f32::ln(2.0f32 * b);
    } else if comptime!(kind == 4) {
        out = f32::ln(4.0f32 * special::PI * b);
    } else if comptime!(kind == 5) {
        out = f32::ln(b) + 1.0f32 + special::EULER_GAMMA;
    } else if comptime!(kind == 6) {
        out = f32::ln(a) + special::HALF_LN_2PI + 0.5f32 - special::LN_2;
    } else if comptime!(kind == 7) {
        out = f32::ln(2.0f32 * special::PI * a);
    } else if comptime!(kind == 8) {
        out = a + f32::ln(b) + special::HALF_LN_2PI + 0.5f32;
    } else if comptime!(kind == 9) {
        out = f32::ln(a / b) + 1.0f32 + 1.0f32 / b;
    } else if comptime!(kind == 10) {
        out = special::EULER_GAMMA * (1.0f32 - 1.0f32 / b) + f32::ln(a / b) + 1.0f32;
    } else if comptime!(kind == 11) {
        // Kumaraswamy: `X^a` is Beta(1, b), which is where the harmonic number
        // `γ + ψ(b+1)` comes from.
        let harmonic = special::EULER_GAMMA + special::digamma_f32(b + 1.0f32);
        out = (1.0f32 - 1.0f32 / b) + (1.0f32 - 1.0f32 / a) * harmonic - f32::ln(a) - f32::ln(b);
    } else if comptime!(kind == 12) {
        out = a - f32::ln(b) + special::lgamma_f32(a) + (1.0f32 - a) * special::digamma_f32(a);
    } else if comptime!(kind == 13) {
        out = a + f32::ln(b) + special::lgamma_f32(a) - (1.0f32 + a) * special::digamma_f32(a);
    } else if comptime!(kind == 14) {
        out = special::lbeta_f32(a, b)
            - (a - 1.0f32) * special::digamma_f32(a)
            - (b - 1.0f32) * special::digamma_f32(b)
            + (a + b - 2.0f32) * special::digamma_f32(a + b);
    } else if comptime!(kind == 15) {
        let half = 0.5f32 * a;
        out = (half + 0.5f32) * (special::digamma_f32(half + 0.5f32) - special::digamma_f32(half))
            + 0.5f32 * f32::ln(a)
            + special::lbeta_f32(half, 0.5f32)
            + f32::ln(c);
    } else if comptime!(kind == 17) {
        // Von Mises. PyTorch has no entropy for this one; it is two Bessel calls.
        out = f32::ln(2.0f32 * special::PI) + special::log_i0_f32(b)
            - b * special::bessel_ratio_f32(b);
    } else if comptime!(kind == 19) {
        let p = 1.0f32 / (1.0f32 + f32::exp(-a));
        out = special::softplus_f32(a) - p * a;
    } else if comptime!(kind == 20) {
        // A geometric run is a sequence of Bernoulli trials, and its entropy is the
        // per-trial entropy divided by the probability one ends the run.
        let p = 1.0f32 / (1.0f32 + f32::exp(-a));
        out = (special::softplus_f32(a) - p * a) / p;
    }
    out
}

/// A summary statistic of `kind`: its mean, variance or mode, selected by `which`
/// from [`moment`].
///
/// `NaN` where the statistic does not exist — a Cauchy has no mean, a uniform no
/// mode, a Student's t with one degree of freedom neither — and `+∞` where it
/// diverges, which is a different thing and worth keeping distinct.
#[cube]
// The `if`s below nest a run-time domain test inside a `#[comptime]` arm. Merging
// the two with `&&` would make the arm a run-time branch and lose the pruning that
// is the point of the dispatch.
#[allow(clippy::collapsible_if)]
// Formatted by hand: without its `#[comptime]` markers the host copy of this
// signature fits on one line and rustfmt would join it, and the two copies must
// stay textually identical.
#[rustfmt::skip]
pub fn moment_of(
    a: f32,
    b: f32,
    c: f32,
    nf: NonFinite,
    #[comptime] kind: u32,
    #[comptime] which: u32,
) -> f32 {
    let mut out: f32 = 0.0;
    out = nf.nan;
    if comptime!(kind == 0) {
        if comptime!(which == 0) {
            out = a;
        } else if comptime!(which == 1) {
            out = b * b;
        } else {
            out = a;
        }
    } else if comptime!(kind == 1) {
        if comptime!(which == 0) {
            out = 0.5f32 * (a + b);
        } else if comptime!(which == 1) {
            let w = b - a;
            out = w * w / 12.0f32;
        }
    } else if comptime!(kind == 2) {
        if comptime!(which == 0) {
            out = 1.0f32 / a;
        } else if comptime!(which == 1) {
            out = 1.0f32 / (a * a);
        } else {
            out = 0.0f32;
        }
    } else if comptime!(kind == 3) {
        if comptime!(which == 0) {
            out = a;
        } else if comptime!(which == 1) {
            out = 2.0f32 * b * b;
        } else {
            out = a;
        }
    } else if comptime!(kind == 4) {
        if comptime!(which == 1) {
            out = nf.inf;
        } else if comptime!(which == 2) {
            out = a;
        }
    } else if comptime!(kind == 5) {
        if comptime!(which == 0) {
            out = a + b * special::EULER_GAMMA;
        } else if comptime!(which == 1) {
            out = special::PI * special::PI * b * b / 6.0f32;
        } else {
            out = a;
        }
    } else if comptime!(kind == 6) {
        if comptime!(which == 0) {
            out = a * f32::sqrt(2.0f32 / special::PI);
        } else if comptime!(which == 1) {
            out = a * a * (1.0f32 - 2.0f32 / special::PI);
        } else {
            out = 0.0f32;
        }
    } else if comptime!(kind == 7) {
        if comptime!(which == 2) {
            out = 0.0f32;
        } else {
            out = nf.inf;
        }
    } else if comptime!(kind == 8) {
        let v = b * b;
        if comptime!(which == 0) {
            out = f32::exp(a + 0.5f32 * v);
        } else if comptime!(which == 1) {
            out = special::expm1_f32(v) * f32::exp(2.0f32 * a + v);
        } else {
            out = f32::exp(a - v);
        }
    } else if comptime!(kind == 9) {
        if comptime!(which == 0) {
            out = nf.inf;
            if b > 1.0f32 {
                out = a * b / (b - 1.0f32);
            }
        } else if comptime!(which == 1) {
            out = nf.inf;
            if b > 2.0f32 {
                let d = b - 1.0f32;
                out = a * a * b / (d * d * (b - 2.0f32));
            }
        } else {
            out = a;
        }
    } else if comptime!(kind == 10) {
        let g1 = f32::exp(special::lgamma_f32(1.0f32 + 1.0f32 / b));
        if comptime!(which == 0) {
            out = a * g1;
        } else if comptime!(which == 1) {
            let g2 = f32::exp(special::lgamma_f32(1.0f32 + 2.0f32 / b));
            out = a * a * (g2 - g1 * g1);
        } else {
            out = 0.0f32;
            if b > 1.0f32 {
                out = a * f32::powf((b - 1.0f32) / b, 1.0f32 / b);
            }
        }
    } else if comptime!(kind == 11) {
        // Kumaraswamy's raw moments are Beta functions: `E[Xⁿ] = b·B(1 + n/a, b)`.
        let m1 = f32::exp(f32::ln(b) + special::lbeta_f32(1.0f32 + 1.0f32 / a, b));
        if comptime!(which == 0) {
            out = m1;
        } else if comptime!(which == 1) {
            let m2 = f32::exp(f32::ln(b) + special::lbeta_f32(1.0f32 + 2.0f32 / a, b));
            out = m2 - m1 * m1;
        } else if a >= 1.0f32 && b >= 1.0f32 {
            out = f32::powf((a - 1.0f32) / (a * b - 1.0f32), 1.0f32 / a);
        }
    } else if comptime!(kind == 12) {
        if comptime!(which == 0) {
            out = a / b;
        } else if comptime!(which == 1) {
            out = a / (b * b);
        } else {
            out = 0.0f32;
            if a >= 1.0f32 {
                out = (a - 1.0f32) / b;
            }
        }
    } else if comptime!(kind == 13) {
        if comptime!(which == 0) {
            out = nf.inf;
            if a > 1.0f32 {
                out = b / (a - 1.0f32);
            }
        } else if comptime!(which == 1) {
            out = nf.inf;
            if a > 2.0f32 {
                let d = a - 1.0f32;
                out = b * b / (d * d * (a - 2.0f32));
            }
        } else {
            out = b / (a + 1.0f32);
        }
    } else if comptime!(kind == 14) {
        let total = a + b;
        if comptime!(which == 0) {
            out = a / total;
        } else if comptime!(which == 1) {
            out = a * b / (total * total * (total + 1.0f32));
        } else if a > 1.0f32 && b > 1.0f32 {
            out = (a - 1.0f32) / (total - 2.0f32);
        }
    } else if comptime!(kind == 15) {
        if comptime!(which == 0) {
            if a > 1.0f32 {
                out = b;
            }
        } else if comptime!(which == 1) {
            if a > 2.0f32 {
                out = c * c * a / (a - 2.0f32);
            } else if a > 1.0f32 {
                out = nf.inf;
            }
        } else {
            out = b;
        }
    } else if comptime!(kind == 16) {
        if comptime!(which == 0) {
            if b > 2.0f32 {
                out = b / (b - 2.0f32);
            }
        } else if comptime!(which == 1) {
            if b > 4.0f32 {
                let d = b - 2.0f32;
                out = 2.0f32 * b * b * (a + b - 2.0f32) / (a * d * d * (b - 4.0f32));
            }
        }
    } else if comptime!(kind == 17) {
        if comptime!(which == 1) {
            // The circular variance, `1 − I₁(κ)/I₀(κ)`, which is what PyTorch
            // reports and is bounded by one rather than by the line.
            out = 1.0f32 - special::bessel_ratio_f32(b);
        } else {
            out = a;
        }
    } else if comptime!(kind == 18) {
        // A continuous Bernoulli is a one-parameter exponential family on `[0, 1]`
        // with natural parameter `t` and log-normaliser `ln((eᵗ − 1)/t)`, so its
        // mean and variance are that function's first two derivatives. Both cancel
        // badly near `t = 0`, where the distribution is uniform; the series below
        // are the Taylor expansions there.
        let t = a;
        if comptime!(which == 0) {
            out = cont_bernoulli_mean(t);
        } else if comptime!(which == 1) {
            if f32::abs(t) > 0.02f32 {
                let e = special::expm1_f32(t);
                out = 1.0f32 / (t * t) - (e + 1.0f32) / (e * e);
            } else {
                let t2 = t * t;
                out = 1.0f32 / 12.0f32 - t2 * (1.0f32 / 240.0f32 - t2 / 6048.0f32);
            }
        }
    } else if comptime!(kind == 19) {
        let p = 1.0f32 / (1.0f32 + f32::exp(-a));
        if comptime!(which == 0) {
            out = p;
        } else if comptime!(which == 1) {
            out = p * (1.0f32 - p);
        } else {
            out = 0.0f32;
            if a > 0.0f32 {
                out = 1.0f32;
            }
        }
    } else if comptime!(kind == 20) {
        // With `p = σ(a)`, the mean `(1−p)/p` is exactly `e^{−a}`.
        let odds = f32::exp(-a);
        if comptime!(which == 0) {
            out = odds;
        } else if comptime!(which == 1) {
            out = odds * (1.0f32 + odds);
        } else {
            out = 0.0f32;
        }
    } else if comptime!(kind == 21) {
        if comptime!(which == 2) {
            out = f32::floor(a);
        } else {
            out = a;
        }
    } else if comptime!(kind == 22) {
        let p = 1.0f32 / (1.0f32 + f32::exp(-b));
        if comptime!(which == 0) {
            out = a * p;
        } else if comptime!(which == 1) {
            out = a * p * (1.0f32 - p);
        } else {
            out = f32::floor((a + 1.0f32) * p).clamp(0.0f32, a);
        }
    } else if comptime!(kind == 23) {
        let odds = f32::exp(b);
        if comptime!(which == 0) {
            out = a * odds;
        } else if comptime!(which == 1) {
            out = a * odds * (1.0f32 + odds);
        } else {
            out = 0.0f32;
            if a > 1.0f32 {
                out = f32::floor((a - 1.0f32) * odds);
            }
        }
    }
    out
}

/// One uniform draw in `(0, 1)` for element `index` of stream `stream`.
#[cube]
pub fn uniform_at(
    index: u32,
    index_hi: u32,
    stream: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> f32 {
    rng::unit_open(rng::draw_lane(
        index, index_hi, stream, key_lo, key_hi, wide,
    ))
}

/// A draw from `Gamma(shape, 1)` by Marsaglia and Tsang's squeeze method.
///
/// The idea is that `Gamma(α, 1)` for `α ≥ 1` is a cubed linear function of a
/// standard normal, up to a rejection whose acceptance rate is above 98% for every
/// shape and rises towards one as the shape grows. Shapes below one are handled by
/// Marsaglia's boost: draw from `Gamma(α + 1, 1)` and scale by `U^{1/α}`.
///
/// The `1 − 0.0331 z⁴` test is the published squeeze: it implies the exact
/// condition below it, so taking it changes nothing but the cost — it skips a
/// logarithm on the overwhelming majority of attempts. The two tests are written as
/// separate arms with the same body rather than as one `||` precisely so that the
/// second, and its logarithm, are reached only when the first fails.
#[allow(clippy::if_same_then_else)]
///
/// The loop is capped at [`MAX_TRIES`]. Reaching the cap needs sixty consecutive
/// rejections, which at the *worst* acceptance rate is a probability below `1e-100`;
/// the fallback returns the distribution's own mode, which is at least in the
/// support. The alternative — an unbounded loop — is a kernel that can hang a
/// device that has no way to interrupt it.
#[cube]
pub fn std_gamma_sample(
    shape: f32,
    index: u32,
    index_hi: u32,
    base: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> f32 {
    let mut alpha = shape;
    let mut boost: f32 = 1.0;
    if shape < 1.0f32 {
        alpha = shape + 1.0f32;
        let u = uniform_at(index, index_hi, base + MAX_TRIES, key_lo, key_hi, wide);
        boost = f32::powf(u, 1.0f32 / shape);
    }
    let d = alpha - 1.0f32 / 3.0f32;
    let cc = 1.0f32 / f32::sqrt(9.0f32 * d);
    let mut acc = d;
    let mut i: u32 = 0;
    while i < MAX_TRIES {
        let bits = rng::draw_block(index, index_hi, base + i, key_lo, key_hi, wide);
        let z = special::std_normal_icdf_f32(rng::unit_open(bits.a));
        let u = rng::unit_open(bits.b);
        let root = 1.0f32 + cc * z;
        // Every write to the loop-carried `acc` and `i` goes through a local first;
        // see the module docs on loop-carried assignment.
        let mut next_acc = acc;
        let mut next_i = i + 1u32;
        if root > 0.0f32 {
            let v = root * root * root;
            let z2 = z * z;
            let mut taken: f32 = 0.0;
            if u < 1.0f32 - 0.0331f32 * z2 * z2 {
                taken = 1.0f32;
            } else if f32::ln(u) < 0.5f32 * z2 + d * (1.0f32 - v + f32::ln(v)) {
                taken = 1.0f32;
            }
            if taken > 0.5f32 {
                next_acc = d * v;
                next_i = MAX_TRIES;
            }
        }
        acc = next_acc;
        i = next_i;
    }
    boost * acc
}

/// A draw from `Poisson(rate)`.
///
/// Two algorithms, because no single one is good over the whole range. Below ten,
/// inversion: walk the CDF from zero with a single uniform, which costs one
/// generator call and about `rate` multiplies. Above ten that walk gets long and
/// `exp(−rate)` starts to underflow, so the transformed-rejection method PTRS takes
/// over — the same switch, at the same threshold, that NumPy and PyTorch make.
#[cube]
pub fn poisson_sample(
    rate: f32,
    index: u32,
    index_hi: u32,
    base: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> f32 {
    let mut out: f32 = 0.0;
    if rate < 10.0f32 {
        let u = uniform_at(index, index_hi, base, key_lo, key_hi, wide);
        let mut k: f32 = 0.0;
        let mut term = f32::exp(-rate);
        let mut cum = term;
        let mut i: u32 = 0;
        while i < 200u32 {
            let mut next_k = k;
            let mut next_term = term;
            let mut next_cum = cum;
            let mut next_i = i + 1u32;
            if u > cum {
                next_k = k + 1.0f32;
                next_term = term * rate / next_k;
                next_cum = cum + next_term;
            } else {
                next_i = 200u32;
            }
            k = next_k;
            term = next_term;
            cum = next_cum;
            i = next_i;
        }
        out = k;
    } else {
        // Hörmann's PTRS: a hat made of a triangle in `1/us` whose inverse is
        // closed-form, so a proposal costs two uniforms and no transcendental.
        let bb = 0.931f32 + 2.53f32 * f32::sqrt(rate);
        let aa = 0.02483f32 * bb - 0.059f32;
        let inv_alpha = 1.1239f32 + 1.1328f32 / (bb - 3.4f32);
        let v_r = 0.9277f32 - 3.6224f32 / (bb - 2.0f32);
        let log_rate = f32::ln(rate);
        let mut acc = f32::floor(rate);
        let mut i: u32 = 0;
        while i < MAX_TRIES {
            let bits = rng::draw_block(index, index_hi, base + i, key_lo, key_hi, wide);
            let u = rng::unit_open(bits.a) - 0.5f32;
            let v = rng::unit_open(bits.b);
            let us = 0.5f32 - f32::abs(u);
            let k = f32::floor((2.0f32 * aa / us + bb) * u + rate + 0.43f32);
            let mut taken: f32 = 0.0;
            if us >= 0.07f32 && v <= v_r {
                taken = 1.0f32;
            }
            if taken < 0.5f32 {
                let mut usable: f32 = 1.0;
                if k < 0.0f32 {
                    usable = 0.0f32;
                }
                if us < 0.013f32 && v > us {
                    usable = 0.0f32;
                }
                if usable > 0.5f32 {
                    let hat = f32::ln(v * inv_alpha / (aa / (us * us) + bb));
                    if hat <= k * log_rate - rate - special::lgamma_f32(k + 1.0f32) {
                        taken = 1.0f32;
                    }
                }
            }
            let mut next_acc = acc;
            let mut next_i = i + 1u32;
            if taken > 0.5f32 {
                next_acc = k;
                next_i = MAX_TRIES;
            }
            acc = next_acc;
            i = next_i;
        }
        out = acc;
    }
    out
}

/// A draw from `Binomial(trials, p)`, with `p` given as a logit.
///
/// Inversion while the mean is small, and Hörmann's BTRS above it — the same
/// two-regime split as [`poisson_sample()`] and for the same reason. BTRS is folded
/// about `p = ½` first: its hat is tuned for the smaller of `p` and `1 − p`, so a
/// `p` above a half is sampled as `n` minus a draw at `1 − p`.
///
/// The acceptance test is written as a ratio of exact log-densities against the
/// mode rather than through the Stirling-tail approximation the original paper uses.
/// It costs three `lgamma` calls on the slow path and removes an approximation from
/// a place where an approximation would bias the samples rather than blur them.
#[cube]
pub fn binomial_sample(
    trials: f32,
    logits: f32,
    index: u32,
    index_hi: u32,
    base: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> f32 {
    let raw_p = 1.0f32 / (1.0f32 + f32::exp(-logits));
    let mut p = raw_p;
    let mut flip: f32 = 0.0;
    if raw_p > 0.5f32 {
        p = 1.0f32 - raw_p;
        flip = 1.0f32;
    }
    let n = f32::floor(trials + 0.5f32);
    let q = 1.0f32 - p;
    let mean = n * p;
    let mut draw: f32 = 0.0;
    if mean < 10.0f32 {
        let u = uniform_at(index, index_hi, base, key_lo, key_hi, wide);
        let ratio = p / q;
        let mut k: f32 = 0.0;
        let mut term = f32::exp(n * f32::ln(q));
        let mut cum = term;
        let mut i: u32 = 0;
        while i < 1000u32 {
            let mut next_k = k;
            let mut next_term = term;
            let mut next_cum = cum;
            let mut next_i = i + 1u32;
            if u > cum {
                if k < n {
                    next_k = k + 1.0f32;
                    next_term = term * ratio * (n - k) / next_k;
                    next_cum = cum + next_term;
                } else {
                    next_i = 1000u32;
                }
            } else {
                next_i = 1000u32;
            }
            k = next_k;
            term = next_term;
            cum = next_cum;
            i = next_i;
        }
        draw = k;
    } else {
        let spq = f32::sqrt(mean * q);
        let bb = 1.15f32 + 2.53f32 * spq;
        let aa = 0.0248f32 * bb + 0.01f32 * p - 0.0873f32;
        let centre = mean + 0.5f32;
        let v_r = 0.92f32 - 4.2f32 / bb;
        let alpha = (2.83f32 + 5.1f32 / bb) * spq;
        let ln_p = f32::ln(p);
        let ln_q = f32::ln(q);
        let m = f32::floor((n + 1.0f32) * p);
        // `ln n!` is the same for every attempt, so it comes out of the loop; only
        // the two `k`-dependent log-gammas stay inside.
        let log_trials = special::lgamma_f32(n + 1.0f32);
        let log_mode =
            log_trials - special::lgamma_f32(m + 1.0f32) - special::lgamma_f32(n - m + 1.0f32)
                + m * ln_p
                + (n - m) * ln_q;
        let mut acc = m;
        let mut i: u32 = 0;
        while i < MAX_TRIES {
            let bits = rng::draw_block(index, index_hi, base + i, key_lo, key_hi, wide);
            let u = rng::unit_open(bits.a) - 0.5f32;
            let v = rng::unit_open(bits.b);
            let us = 0.5f32 - f32::abs(u);
            let k = f32::floor((2.0f32 * aa / us + bb) * u + centre);
            let mut usable: f32 = 1.0;
            if k < 0.0f32 {
                usable = 0.0f32;
            }
            if k > n {
                usable = 0.0f32;
            }
            let mut taken: f32 = 0.0;
            if usable > 0.5f32 {
                if us >= 0.07f32 && v <= v_r {
                    taken = 1.0f32;
                }
                if taken < 0.5f32 {
                    let hat = f32::ln(v * alpha / (aa / (us * us) + bb));
                    let target = log_trials
                        - special::lgamma_f32(k + 1.0f32)
                        - special::lgamma_f32(n - k + 1.0f32)
                        + k * ln_p
                        + (n - k) * ln_q
                        - log_mode;
                    if hat <= target {
                        taken = 1.0f32;
                    }
                }
            }
            let mut next_acc = acc;
            let mut next_i = i + 1u32;
            if taken > 0.5f32 {
                next_acc = k;
                next_i = MAX_TRIES;
            }
            acc = next_acc;
            i = next_i;
        }
        draw = acc;
    }
    let mut out = draw;
    if flip > 0.5f32 {
        out = n - draw;
    }
    out
}

/// A draw from `VonMises(0, concentration)`, in `[−π, π)`.
///
/// Best and Fisher's wrapped-Cauchy rejection method, which is what PyTorch uses.
/// The proposal is a wrapped Cauchy tuned so its acceptance rate stays above about
/// 66% for every concentration, and the accepted angle comes out of a single
/// `acos`. Below a concentration of `1e-4` the distribution is uniform to well
/// inside `f32`, and the tuning constants divide by the concentration, so that case
/// is taken directly.
///
/// As in [`std_gamma_sample()`], the two acceptance tests are separate arms with the
/// same body so the second — the one with the logarithm — is only reached when the
/// cheap one fails.
#[allow(clippy::if_same_then_else)]
#[cube]
pub fn von_mises_sample(
    concentration: f32,
    index: u32,
    index_hi: u32,
    base: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> f32 {
    let mut out: f32 = 0.0;
    if concentration < 1.0e-4f32 {
        out = special::PI
            * (2.0f32 * uniform_at(index, index_hi, base, key_lo, key_hi, wide) - 1.0f32);
    } else {
        let tau = 1.0f32 + f32::sqrt(1.0f32 + 4.0f32 * concentration * concentration);
        let rho = (tau - f32::sqrt(2.0f32 * tau)) / (2.0f32 * concentration);
        let r = (1.0f32 + rho * rho) / (2.0f32 * rho);
        let mut acc: f32 = 0.0;
        let mut i: u32 = 0;
        while i < MAX_TRIES {
            let bits = rng::draw_block(index, index_hi, base + i, key_lo, key_hi, wide);
            let u1 = rng::unit_open(bits.a);
            let u2 = rng::unit_open(bits.b);
            let u3 = rng::unit_open(bits.c);
            let z = f32::cos(special::PI * u1);
            let f = (1.0f32 + r * z) / (r + z);
            let cc = concentration * (r - f);
            let mut taken: f32 = 0.0;
            if u2 < cc * (2.0f32 - cc) {
                taken = 1.0f32;
            } else if f32::ln(cc / u2) + 1.0f32 - cc >= 0.0f32 {
                taken = 1.0f32;
            }
            let mut next_acc = acc;
            let mut next_i = i + 1u32;
            if taken > 0.5f32 {
                next_acc = special::sign_f32(u3 - 0.5f32) * f32::acos(f.clamp(-1.0f32, 1.0f32));
                next_i = MAX_TRIES;
            }
            acc = next_acc;
            i = next_i;
        }
        out = acc;
    }
    out
}

/// Wrap an angle into `[−π, π)`.
#[cube]
pub fn wrap_angle(x: f32) -> f32 {
    let two_pi = 2.0f32 * special::PI;
    let shifted = x + special::PI;
    shifted - two_pi * f32::floor(shifted / two_pi) - special::PI
}

/// A draw from `kind` out of one 32-bit word of randomness.
///
/// Split out of [`sample_of()`] so that the scalar path and the four-at-a-time path
/// cannot drift: both call this, and neither knows how the word was produced. Only
/// the families that need exactly one uniform appear here —
/// [`Kind::lane_sampled`] is the list — and every other kind returns zero, which no
/// caller ever sees because the host picks the path.
#[cube]
pub fn sample_from_bits_of(bits: u32, a: f32, b: f32, c: f32, #[comptime] kind: u32) -> f32 {
    let mut out: f32 = 0.0;
    if comptime!(kind <= 11) {
        // Everything with an elementary quantile is sampled by inverting it: one
        // uniform, no rejection, and a draw that is a differentiable function of the
        // parameters — which is exactly what makes these the reparameterisable ones.
        out = closed_form_icdf_of(rng::unit_open(bits), a, b, c, kind);
    } else if comptime!(kind == 18) {
        out = closed_form_icdf_of(rng::unit_open(bits), a, b, c, kind);
    } else if comptime!(kind == 19) {
        // Half-open, so a probability of exactly zero can never produce a one.
        let u = rng::unit_half_open(bits);
        out = 0.0f32;
        if u < 1.0f32 / (1.0f32 + f32::exp(-a)) {
            out = 1.0f32;
        }
    } else if comptime!(kind == 20) {
        // `⌊ln U / ln(1 − p)⌋`, with `ln(1 − p)` taken as `log σ(−logit)` so that a
        // `p` near one keeps its digits.
        out = f32::floor(f32::ln(rng::unit_open(bits)) / special::log_sigmoid_f32(-a));
    } else if comptime!(kind == 24) {
        let u = rng::unit_open(bits);
        out = (b + f32::ln(u) - special::log1p_f32(-u)) / a;
    } else if comptime!(kind == 25) {
        let u = rng::unit_open(bits);
        let y = (b + f32::ln(u) - special::log1p_f32(-u)) / a;
        out = 1.0f32 / (1.0f32 + f32::exp(-y));
    }
    out
}

/// One draw from `kind`, for element `index`.
///
/// The draw depends on `index`, the seed and nothing else — not on how the launch
/// was shaped, not on how many elements were asked for, and not on which other
/// elements a unit happened to also compute. That is what makes a sample bit-exact
/// rather than merely correct in distribution.
#[cube]
pub fn sample_of(
    a: f32,
    b: f32,
    c: f32,
    index: u32,
    index_hi: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] kind: u32,
    #[comptime] wide: bool,
) -> f32 {
    let mut out: f32 = 0.0;
    if comptime!(kind <= 11) {
        out = sample_from_bits_of(
            rng::draw_lane(index, index_hi, 0u32, key_lo, key_hi, wide),
            a,
            b,
            c,
            kind,
        );
    } else if comptime!(kind == 12) {
        out = std_gamma_sample(a, index, index_hi, 0u32, key_lo, key_hi, wide) / b;
    } else if comptime!(kind == 13) {
        out = b / std_gamma_sample(a, index, index_hi, 0u32, key_lo, key_hi, wide);
    } else if comptime!(kind == 14) {
        // A Beta is the share one of two independent Gammas takes of their sum. The
        // second draws from its own block of streams so the two never correlate.
        let g1 = std_gamma_sample(a, index, index_hi, 0u32, key_lo, key_hi, wide);
        let g2 = std_gamma_sample(b, index, index_hi, SUB_DRAW, key_lo, key_hi, wide);
        out = g1 / (g1 + g2);
    } else if comptime!(kind == 15) {
        // Student's t: a normal divided by the root of a scaled chi-square, which is
        // a Gamma of half the degrees of freedom.
        let z =
            special::std_normal_icdf_f32(uniform_at(index, index_hi, 0u32, key_lo, key_hi, wide));
        let g = std_gamma_sample(0.5f32 * a, index, index_hi, SUB_DRAW, key_lo, key_hi, wide);
        out = b + c * z * f32::sqrt(0.5f32 * a / g);
    } else if comptime!(kind == 16) {
        let g1 = std_gamma_sample(0.5f32 * a, index, index_hi, 0u32, key_lo, key_hi, wide);
        let g2 = std_gamma_sample(0.5f32 * b, index, index_hi, SUB_DRAW, key_lo, key_hi, wide);
        // Two scaled chi-squares; a chi-square with `k` degrees of freedom is
        // twice a `Gamma(k/2, 1)`, and the twos cancel out of the ratio.
        out = (g1 / (0.5f32 * a)) / (g2 / (0.5f32 * b));
    } else if comptime!(kind == 17) {
        out = wrap_angle(a + von_mises_sample(b, index, index_hi, 0u32, key_lo, key_hi, wide));
    } else if comptime!(kind == 21) {
        out = poisson_sample(a, index, index_hi, 0u32, key_lo, key_hi, wide);
    } else if comptime!(kind == 22) {
        out = binomial_sample(a, b, index, index_hi, 0u32, key_lo, key_hi, wide);
    } else if comptime!(kind == 23) {
        // A negative binomial is a Poisson whose rate is itself Gamma-distributed;
        // that mixture *is* the distribution, so this is exact rather than a
        // two-stage approximation.
        let rate = std_gamma_sample(a, index, index_hi, 0u32, key_lo, key_hi, wide) * f32::exp(b);
        out = poisson_sample(rate, index, index_hi, SUB_DRAW, key_lo, key_hi, wide);
    } else {
        out = sample_from_bits_of(
            rng::draw_lane(index, index_hi, 0u32, key_lo, key_hi, wide),
            a,
            b,
            c,
            kind,
        );
    }
    out
}

/// The logistic sigmoid, `σ(t)`.
#[cube]
pub fn sigmoid_f32(t: f32) -> f32 {
    1.0f32 / (1.0f32 + f32::exp(-t))
}

/// Every partial derivative of [`log_prob_of()`], in one pass.
///
/// Hand-derived rather than composed, which is the whole reason this module exists:
/// a normal's `∂ log p/∂σ` written out of differentiable tensor ops is a dozen
/// launches and a dozen intermediates, and written out is `(z² − 1)/σ`.
///
/// Slots a distribution does not use are left at zero, as is `dx` for a discrete
/// distribution whose support is not differentiable in any useful sense — except
/// where the density does extend continuously, which is exactly the case PyTorch's
/// own `log_prob` covers by accepting a non-integer `value`.
#[cube]
pub fn log_prob_grad_of(x: f32, a: f32, b: f32, c: f32, #[comptime] kind: u32) -> Grad4 {
    let mut dx: f32 = 0.0;
    let mut da: f32 = 0.0;
    let mut db: f32 = 0.0;
    let mut dc: f32 = 0.0;
    if comptime!(kind == 0) {
        let z = (x - a) / b;
        dx = -z / b;
        da = z / b;
        db = (z * z - 1.0f32) / b;
    } else if comptime!(kind == 1) {
        let w = 1.0f32 / (b - a);
        da = w;
        db = -w;
    } else if comptime!(kind == 2) {
        dx = -a;
        da = 1.0f32 / a - x;
    } else if comptime!(kind == 3) {
        let s = special::sign_f32(x - a) / b;
        dx = -s;
        da = s;
        db = (f32::abs(x - a) / b - 1.0f32) / b;
    } else if comptime!(kind == 4) {
        let z = (x - a) / b;
        let g = 2.0f32 * z / (1.0f32 + z * z);
        dx = -g / b;
        da = g / b;
        db = (z * g - 1.0f32) / b;
    } else if comptime!(kind == 5) {
        let z = (x - a) / b;
        let g = 1.0f32 - f32::exp(-z);
        dx = -g / b;
        da = g / b;
        db = (z * g - 1.0f32) / b;
    } else if comptime!(kind == 6) {
        let z = x / a;
        dx = -z / a;
        da = (z * z - 1.0f32) / a;
    } else if comptime!(kind == 7) {
        let z = x / a;
        let g = 2.0f32 * z / (1.0f32 + z * z);
        dx = -g / a;
        da = (z * g - 1.0f32) / a;
    } else if comptime!(kind == 8) {
        let l = f32::ln(x);
        let z = (l - a) / b;
        dx = (-z / b - 1.0f32) / x;
        da = z / b;
        db = (z * z - 1.0f32) / b;
    } else if comptime!(kind == 9) {
        dx = -(b + 1.0f32) / x;
        da = b / a;
        db = 1.0f32 / b + f32::ln(a) - f32::ln(x);
    } else if comptime!(kind == 10) {
        let z = x / a;
        let t = f32::powf(z, b);
        dx = (b - 1.0f32) / x - b * t / x;
        da = b * (t - 1.0f32) / a;
        db = 1.0f32 / b + (1.0f32 - t) * f32::ln(z);
    } else if comptime!(kind == 11) {
        let lx = f32::ln(x);
        let w = f32::powf(x, a);
        let rest = 1.0f32 - w;
        dx = (a - 1.0f32) / x - (b - 1.0f32) * a * w / (x * rest);
        da = 1.0f32 / a + lx - (b - 1.0f32) * w * lx / rest;
        db = 1.0f32 / b + special::log1p_f32(-w);
    } else if comptime!(kind == 12) {
        dx = (a - 1.0f32) / x - b;
        da = f32::ln(b) + f32::ln(x) - special::digamma_f32(a);
        db = a / b - x;
    } else if comptime!(kind == 13) {
        dx = b / (x * x) - (a + 1.0f32) / x;
        da = f32::ln(b) - special::digamma_f32(a) - f32::ln(x);
        db = a / b - 1.0f32 / x;
    } else if comptime!(kind == 14) {
        let psi_sum = special::digamma_f32(a + b);
        dx = (a - 1.0f32) / x - (b - 1.0f32) / (1.0f32 - x);
        da = f32::ln(x) - special::digamma_f32(a) + psi_sum;
        db = special::log1p_f32(-x) - special::digamma_f32(b) + psi_sum;
    } else if comptime!(kind == 15) {
        let z = (x - b) / c;
        let half = 0.5f32 * a;
        let scaled = z * z / a;
        let sq = 1.0f32 + scaled;
        let slope = (a + 1.0f32) * z / (a * sq);
        dx = -slope / c;
        db = slope / c;
        dc = (slope * z - 1.0f32) / c;
        da = 0.5f32 * (special::digamma_f32(half + 0.5f32) - special::digamma_f32(half))
            - 0.5f32 / a
            - 0.5f32 * special::log1p_f32(scaled)
            + (a + 1.0f32) * scaled / (2.0f32 * a * sq);
    } else if comptime!(kind == 16) {
        let h1 = 0.5f32 * a;
        let h2 = 0.5f32 * b;
        let r = a / b;
        let denom = 1.0f32 + r * x;
        let psi_sum = special::digamma_f32(h1 + h2);
        dx = (h1 - 1.0f32) / x - (h1 + h2) * r / denom;
        da = 0.5f32
            * (psi_sum - special::digamma_f32(h1) + f32::ln(r) + 1.0f32 + f32::ln(x)
                - special::log1p_f32(r * x))
            - (h1 + h2) * x / (b * denom);
        db = 0.5f32 * (psi_sum - special::digamma_f32(h2) - special::log1p_f32(r * x)) - h1 / b
            + (h1 + h2) * r * x / (b * denom);
    } else if comptime!(kind == 17) {
        let s = f32::sin(x - a);
        dx = -b * s;
        da = b * s;
        db = f32::cos(x - a) - special::bessel_ratio_f32(b);
    } else if comptime!(kind == 18) {
        // An exponential family in its natural parameter: the score is the value
        // minus the mean, and nothing else survives.
        dx = a;
        da = x - cont_bernoulli_mean(a);
    } else if comptime!(kind == 19) {
        dx = a;
        da = x - sigmoid_f32(a);
    } else if comptime!(kind == 20) {
        dx = special::log_sigmoid_f32(-a);
        da = sigmoid_f32(-a) - x * sigmoid_f32(a);
    } else if comptime!(kind == 21) {
        dx = f32::ln(a) - special::digamma_f32(x + 1.0f32);
        da = x / a - 1.0f32;
    } else if comptime!(kind == 22) {
        dx = b + special::digamma_f32(a - x + 1.0f32) - special::digamma_f32(x + 1.0f32);
        da = special::digamma_f32(a + 1.0f32)
            - special::digamma_f32(a - x + 1.0f32)
            - special::softplus_f32(b);
        db = x - a * sigmoid_f32(b);
    } else if comptime!(kind == 23) {
        let psi_sum = special::digamma_f32(a + x);
        dx = special::log_sigmoid_f32(b) + psi_sum - special::digamma_f32(1.0f32 + x);
        da = special::log_sigmoid_f32(-b) + psi_sum - special::digamma_f32(a);
        db = x * sigmoid_f32(-b) - a * sigmoid_f32(b);
    } else if comptime!(kind == 24) {
        let diff = b - x * a;
        let g = 1.0f32 - 2.0f32 * sigmoid_f32(diff);
        dx = -a * g;
        da = 1.0f32 / a - x * g;
        db = g;
    } else if comptime!(kind == 25) {
        let y = f32::ln(x) - special::log1p_f32(-x);
        let diff = b - y * a;
        let g = 1.0f32 - 2.0f32 * sigmoid_f32(diff);
        dx = -a * g / (x * (1.0f32 - x)) - 1.0f32 / x + 1.0f32 / (1.0f32 - x);
        da = 1.0f32 / a - y * g;
        db = g;
    }
    Grad4 { dx, da, db, dc }
}

/// Every partial derivative of [`entropy_of()`].
///
/// `dx` is always zero — an entropy has no value argument — and the slots of a
/// distribution whose entropy is not implemented are `NaN`, matching [`entropy_of()`]
/// so that a caller cannot silently differentiate something that was never computed.
#[cube]
pub fn entropy_grad_of(a: f32, b: f32, c: f32, nf: NonFinite, #[comptime] kind: u32) -> Grad4 {
    let mut da: f32 = 0.0;
    let mut db: f32 = 0.0;
    let mut dc: f32 = 0.0;
    da = nf.nan;
    db = nf.nan;
    if comptime!(kind == 0) {
        da = 0.0f32;
        db = 1.0f32 / b;
    } else if comptime!(kind == 1) {
        let w = 1.0f32 / (b - a);
        da = -w;
        db = w;
    } else if comptime!(kind == 2) {
        da = -1.0f32 / a;
    } else if comptime!(kind == 3 || kind == 4 || kind == 5) {
        // Laplace, Cauchy and Gumbel are location–scale families whose entropy is
        // `ln(scale)` plus a constant, so the location does not enter and the scale
        // enters the same way in all three.
        da = 0.0f32;
        db = 1.0f32 / b;
    } else if comptime!(kind == 6 || kind == 7) {
        // Half-normal and half-Cauchy, likewise, with the scale in the first slot.
        da = 1.0f32 / a;
    } else if comptime!(kind == 8) {
        da = 1.0f32;
        db = 1.0f32 / b;
    } else if comptime!(kind == 9) {
        da = 1.0f32 / a;
        db = -(1.0f32 + 1.0f32 / b) / b;
    } else if comptime!(kind == 10) {
        da = 1.0f32 / a;
        db = special::EULER_GAMMA / (b * b) - 1.0f32 / b;
    } else if comptime!(kind == 11) {
        let harmonic = special::EULER_GAMMA + special::digamma_f32(b + 1.0f32);
        da = harmonic / (a * a) - 1.0f32 / a;
        db = 1.0f32 / (b * b) + (1.0f32 - 1.0f32 / a) * special::trigamma_f32(b + 1.0f32)
            - 1.0f32 / b;
    } else if comptime!(kind == 12) {
        da = 1.0f32 + (1.0f32 - a) * special::trigamma_f32(a);
        db = -1.0f32 / b;
    } else if comptime!(kind == 13) {
        da = 1.0f32 - (1.0f32 + a) * special::trigamma_f32(a);
        db = 1.0f32 / b;
    } else if comptime!(kind == 14) {
        let tri_sum = special::trigamma_f32(a + b);
        da = (a + b - 2.0f32) * tri_sum - (a - 1.0f32) * special::trigamma_f32(a);
        db = (a + b - 2.0f32) * tri_sum - (b - 1.0f32) * special::trigamma_f32(b);
    } else if comptime!(kind == 15) {
        let half = 0.5f32 * a;
        let hi = half + 0.5f32;
        // Differentiating `((ν+1)/2)(ψ((ν+1)/2) − ψ(ν/2)) + ½ln ν + ln B(ν/2, ½)`,
        // the four digammas cancel in pairs and only the trigammas and `1/(2ν)`
        // survive.
        da = 0.25f32 * (a + 1.0f32) * (special::trigamma_f32(hi) - special::trigamma_f32(half))
            + 0.5f32 / a;
        db = 0.0f32;
        dc = 1.0f32 / c;
    } else if comptime!(kind == 17) {
        let ratio = special::bessel_ratio_f32(b);
        // `d/dκ ln I₀ = I₁/I₀`, and `d/dκ (κ I₁/I₀)` unwinds through the Bessel
        // recurrence `I₀' = I₁` and `I₁' = I₀ − I₁/κ` into the expression below.
        let dratio = 1.0f32 - ratio * ratio - ratio / b;
        da = 0.0f32;
        db = -b * dratio;
    } else if comptime!(kind == 19) {
        let p = sigmoid_f32(a);
        da = -a * p * (1.0f32 - p);
    } else if comptime!(kind == 20) {
        let p = sigmoid_f32(a);
        let bern = special::softplus_f32(a) - p * a;
        let dbern = -a * p * (1.0f32 - p);
        da = (dbern * p - bern * p * (1.0f32 - p)) / (p * p);
    }
    Grad4 {
        dx: 0.0f32,
        da,
        db,
        dc,
    }
}

/// Every partial derivative of [`icdf_of()`], in one pass.
///
/// This is what makes a *reparameterised* sample differentiable. A draw from one of
/// the inverse-CDF families is `x = F⁻¹(u; θ)` for a `u` that does not depend on
/// `θ`, so `∂x/∂θ` is exactly this — the path derivative, with no score-function
/// variance in it. [`Kind::reparameterised`] says which kinds are covered; the rest
/// leave every slot at zero, and the host layer refuses to build the node.
#[cube]
pub fn icdf_grad_of(q: f32, a: f32, b: f32, c: f32, #[comptime] kind: u32) -> Grad4 {
    let mut dq: f32 = 0.0;
    let mut da: f32 = 0.0;
    let mut db: f32 = 0.0;
    let mut dc: f32 = 0.0;
    dc = 0.0f32;
    if comptime!(kind == 0) {
        let z = special::std_normal_icdf_f32(q);
        da = 1.0f32;
        db = z;
        // `dΦ⁻¹/dq = 1/φ(z)`.
        dq = b * f32::exp(0.5f32 * z * z + special::HALF_LN_2PI);
    } else if comptime!(kind == 1) {
        da = 1.0f32 - q;
        db = q;
        dq = b - a;
    } else if comptime!(kind == 2) {
        let x = -special::log1p_f32(-q) / a;
        da = -x / a;
        dq = 1.0f32 / (a * (1.0f32 - q));
    } else if comptime!(kind == 3) {
        let d = q - 0.5f32;
        da = 1.0f32;
        db = -special::sign_f32(d) * special::log1p_f32(-2.0f32 * f32::abs(d));
        dq = 2.0f32 * b / (1.0f32 - 2.0f32 * f32::abs(d));
    } else if comptime!(kind == 4) {
        let t = f32::tan(special::PI * (q - 0.5f32));
        da = 1.0f32;
        db = t;
        dq = b * special::PI * (1.0f32 + t * t);
    } else if comptime!(kind == 5) {
        let l = -f32::ln(q);
        da = 1.0f32;
        db = -f32::ln(l);
        dq = b / (l * q);
    } else if comptime!(kind == 6) {
        let e = special::erfinv_f32(q);
        da = special::SQRT_2 * e;
        // `d erfinv/dq = (√π/2)·exp(erfinv(q)²)`.
        dq = a * special::SQRT_2 * 0.8862269f32 * f32::exp(e * e);
    } else if comptime!(kind == 7) {
        let t = f32::tan(0.5f32 * special::PI * q);
        da = t;
        dq = a * 0.5f32 * special::PI * (1.0f32 + t * t);
    } else if comptime!(kind == 8) {
        let z = special::std_normal_icdf_f32(q);
        let x = f32::exp(a + b * z);
        da = x;
        db = x * z;
        dq = x * b * f32::exp(0.5f32 * z * z + special::HALF_LN_2PI);
    } else if comptime!(kind == 9) {
        let l = special::log1p_f32(-q);
        let x = a * f32::exp(-l / b);
        da = x / a;
        db = x * l / (b * b);
        dq = x / (b * (1.0f32 - q));
    } else if comptime!(kind == 10) {
        let l = -special::log1p_f32(-q);
        let x = a * f32::powf(l, 1.0f32 / b);
        da = x / a;
        db = -x * f32::ln(l) / (b * b);
        dq = x / (b * l * (1.0f32 - q));
    } else if comptime!(kind == 11) {
        let l = special::log1p_f32(-q);
        let w = -special::expm1_f32(l / b);
        let x = f32::powf(w, 1.0f32 / a);
        let chain = x / (a * w);
        da = -x * f32::ln(w) / (a * a);
        db = chain * (1.0f32 - w) * l / (b * b);
        dq = chain * (1.0f32 - w) / (b * (1.0f32 - q));
    }
    Grad4 { dx: dq, da, db, dc }
}

/// The path derivative of a draw.
///
/// Re-derives the very uniform the forward pass used from `index` and the key, then
/// differentiates the quantile at it. Nothing has to be saved between the passes:
/// the generator is a pure function of where it is, so the backward pass can ask it
/// the same question and get the same answer.
#[cube]
pub fn rsample_grad_of(
    a: f32,
    b: f32,
    c: f32,
    index: u32,
    index_hi: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] kind: u32,
    #[comptime] wide: bool,
) -> Grad4 {
    let mut dq: f32 = 0.0;
    let mut da: f32 = 0.0;
    let mut db: f32 = 0.0;
    let mut dc: f32 = 0.0;
    if comptime!(kind <= 11) {
        let u = uniform_at(index, index_hi, 0u32, key_lo, key_hi, wide);
        let g = icdf_grad_of(u, a, b, c, kind);
        dq = g.dx;
        da = g.da;
        db = g.db;
        dc = g.dc;
    } else if comptime!(kind == 24) {
        // `(logits + logistic noise)/temperature`: linear in the logit, and inverse
        // in the temperature.
        let u = uniform_at(index, index_hi, 0u32, key_lo, key_hi, wide);
        let y = (b + f32::ln(u) - special::log1p_f32(-u)) / a;
        da = -y / a;
        db = 1.0f32 / a;
    } else if comptime!(kind == 25) {
        let u = uniform_at(index, index_hi, 0u32, key_lo, key_hi, wide);
        let y = (b + f32::ln(u) - special::log1p_f32(-u)) / a;
        let s = sigmoid_f32(y);
        let jac = s * (1.0f32 - s);
        da = -jac * y / a;
        db = jac / a;
    }
    // `dx` is the derivative with respect to the underlying uniform, which a caller
    // never differentiates through; it is carried only so the struct is one shape.
    Grad4 { dx: dq, da, db, dc }
}

// <<< shared

// ---------------------------------------------------------------------------
// Launching
// ---------------------------------------------------------------------------

/// Which of the three value-taking operations a `pointwise_kernel` performs.
pub mod op {
    /// The log-density.
    pub const LOG_PROB: u32 = 0;
    /// The cumulative distribution function.
    pub const CDF: u32 = 1;
    /// The quantile function.
    pub const ICDF: u32 = 2;
}

/// One value, up to three parameters, one result — fused.
///
/// The whole point of this kernel is that it is *one* kernel. A normal
/// log-density written out of elementwise tensor ops is eight launches and seven
/// intermediates the width of the batch; here it is one launch and no intermediate
/// leaves a register. The arithmetic is identical, so the saving is pure: memory
/// traffic and dispatch, not accuracy.
///
/// `stride_a`, `stride_b` and `stride_c` are `0` for a parameter that is one value
/// broadcast over the batch and `1` for one that varies with it, which is how a
/// scalar parameter costs a register instead of a materialised tensor.
#[cube(launch_unchecked)]
fn pointwise_kernel<E: Float + CubeElement>(
    value: &Array<E>,
    pa: &Array<E>,
    pb: &Array<E>,
    pc: &Array<E>,
    out: &mut Array<E>,
    n: u32,
    batch: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    inf: f32,
    nan: f32,
    #[comptime] kind: u32,
    #[comptime] which: u32,
    #[comptime] tiled: bool,
    #[comptime] group: u32,
) {
    let nf = NonFinite { inf, nan };
    #[unroll]
    for slot in 0..group {
        let i = ABSOLUTE_POS * group as usize + slot as usize;
        if i < n as usize {
            pointwise_at::<E>(
                value, pa, pb, pc, out, i, batch, stride_a, stride_b, stride_c, nf, kind, which,
                tiled,
            );
        }
    }
}

/// One element of `pointwise_kernel`.
///
/// A unit does four of these rather than one. The arithmetic is unchanged — each
/// element is computed exactly as it would have been alone, which is what keeps the
/// result bit-identical — but the index arithmetic is amortised and, where a
/// parameter is a broadcast scalar, its logarithm is loaded from the same address
/// four times running and the compiler keeps it in a register.
#[cube]
fn pointwise_at<E: Float + CubeElement>(
    value: &Array<E>,
    pa: &Array<E>,
    pb: &Array<E>,
    pc: &Array<E>,
    out: &mut Array<E>,
    i: usize,
    batch: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    nf: NonFinite,
    #[comptime] kind: u32,
    #[comptime] which: u32,
    #[comptime] tiled: bool,
) {
    {
        {
            // `tiled` is set when the value has more elements than the parameters — a
            // `[samples, batch]` score against a `[batch]` policy — and the modulo it
            // costs is compiled away in the usual case, where it does not.
            let mut p = i;
            if comptime!(tiled) {
                p = i % batch as usize;
            }
            let x = f32::cast_from(value[i]);
            let a = f32::cast_from(pa[p * stride_a as usize]);
            let b = f32::cast_from(pb[p * stride_b as usize]);
            let c = f32::cast_from(pc[p * stride_c as usize]);
            let mut r: f32 = 0.0;
            if comptime!(which == 0) {
                r = log_prob_of(x, a, b, c, nf, kind);
            } else if comptime!(which == 1) {
                r = cdf_of(x, a, b, c, nf, kind);
            } else {
                r = icdf_of(x, a, b, c, nf, kind);
            }
            out[i] = E::cast_from(r);
        }
    }
}

/// Up to three parameters in, one summary out: entropy, mean, variance or mode.
#[cube(launch_unchecked)]
fn param_kernel<E: Float + CubeElement>(
    pa: &Array<E>,
    pb: &Array<E>,
    pc: &Array<E>,
    out: &mut Array<E>,
    n: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    inf: f32,
    nan: f32,
    #[comptime] kind: u32,
    #[comptime] which: u32,
) {
    if ABSOLUTE_POS < n as usize {
        let i = ABSOLUTE_POS;
        let a = f32::cast_from(pa[i * stride_a as usize]);
        let b = f32::cast_from(pb[i * stride_b as usize]);
        let c = f32::cast_from(pc[i * stride_c as usize]);
        let nf = NonFinite { inf, nan };
        let mut r: f32 = 0.0;
        if comptime!(which == 3) {
            r = entropy_of(a, b, c, nf, kind);
        } else {
            r = moment_of(a, b, c, nf, kind, which);
        }
        out[i] = E::cast_from(r);
    }
}

/// One draw of `kind` written into `out[i]`, reading the parameters that element
/// needs.
///
/// Factored out so the scalar and the four-at-a-time paths of `sample_kernel` are
/// literally the same code with a different source of randomness.
#[cube]
fn emit_draw<E: Float + CubeElement>(
    pa: &Array<E>,
    pb: &Array<E>,
    pc: &Array<E>,
    out: &mut Array<E>,
    i: usize,
    batch: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    bits: u32,
    #[comptime] kind: u32,
) {
    let p = i % batch as usize;
    let a = f32::cast_from(pa[p * stride_a as usize]);
    let b = f32::cast_from(pb[p * stride_b as usize]);
    let c = f32::cast_from(pc[p * stride_c as usize]);
    out[i] = E::cast_from(sample_from_bits_of(bits, a, b, c, kind));
}

/// One draw per output element.
///
/// `offset` shifts the element index that seeds the generator, so that drawing `n`
/// independent batches is `n` launches into one buffer with `offset` stepping by the
/// batch size — and each of those draws is the same one it would have been had the
/// whole thing been drawn at once. `param_index` is what selects the parameters, and
/// it wraps at the batch size, which is what makes a `[samples, batch]` draw read a
/// `[batch]` parameter.
#[cube(launch_unchecked)]
fn sample_kernel<E: Float + CubeElement>(
    pa: &Array<E>,
    pb: &Array<E>,
    pc: &Array<E>,
    out: &mut Array<E>,
    n: u32,
    batch: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    offset_lo: u32,
    offset_hi: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] kind: u32,
    #[comptime] lane_batched: bool,
    #[comptime] wide: bool,
) {
    if comptime!(lane_batched) {
        // Four consecutive elements read four lanes of one generator evaluation, so
        // a unit that owns all four pays for one. `offset` is a multiple of four —
        // the host checks — so the group a unit owns is exactly a Philox counter.
        let group = ABSOLUTE_POS as u32;
        let base = (offset_lo >> 2u32) + group;
        let bits = rng::draw_lane_block(base, offset_hi, 0u32, key_lo, key_hi, wide);
        let first = 4 * ABSOLUTE_POS;
        if first < n as usize {
            emit_draw::<E>(
                pa, pb, pc, out, first, batch, stride_a, stride_b, stride_c, bits.a, kind,
            );
        }
        if first + 1 < n as usize {
            emit_draw::<E>(
                pa,
                pb,
                pc,
                out,
                first + 1,
                batch,
                stride_a,
                stride_b,
                stride_c,
                bits.b,
                kind,
            );
        }
        if first + 2 < n as usize {
            emit_draw::<E>(
                pa,
                pb,
                pc,
                out,
                first + 2,
                batch,
                stride_a,
                stride_b,
                stride_c,
                bits.c,
                kind,
            );
        }
        if first + 3 < n as usize {
            emit_draw::<E>(
                pa,
                pb,
                pc,
                out,
                first + 3,
                batch,
                stride_a,
                stride_b,
                stride_c,
                bits.d,
                kind,
            );
        }
    } else if ABSOLUTE_POS < n as usize {
        let i = ABSOLUTE_POS;
        let p = i % batch as usize;
        let a = f32::cast_from(pa[p * stride_a as usize]);
        let b = f32::cast_from(pb[p * stride_b as usize]);
        let c = f32::cast_from(pc[p * stride_c as usize]);
        // A 64-bit draw index, so a tensor with more than four billion elements — or
        // a long run of batches at stepped offsets — still gets distinct streams.
        let lo = offset_lo + i as u32;
        let mut hi = offset_hi;
        if lo < offset_lo {
            hi += 1u32;
        }
        out[i] = E::cast_from(sample_of(a, b, c, lo, hi, key_lo, key_hi, kind, wide));
    }
}

/// Elements one unit of `pointwise_kernel` handles.
///
/// Four, because that is where the measurement stops improving on the runtimes here:
/// two is most of the win and eight adds nothing but a longer tail on a batch that
/// does not divide. It changes only how the work is packed, never what is computed.
pub const POINTWISE_GROUP: u32 = 4;

/// A parameter as the kernels see it: a buffer plus the stride that says whether it
/// varies with the batch.
pub(crate) struct Bound<'a, R: Runtime, E: FloatElem> {
    /// The values.
    pub tensor: &'a Tensor<R, E>,
    /// `1` if the buffer is batch-shaped, `0` if it is a single broadcast value.
    pub stride: u32,
}

impl<'a, R: Runtime, E: FloatElem> Bound<'a, R, E> {
    /// Bind a parameter tensor against a batch of `batch` elements.
    pub(crate) fn new(tensor: &'a Tensor<R, E>, batch: usize) -> Self {
        let stride = u32::from(tensor.len() == batch && batch != 1);
        Self { tensor, stride }
    }
}

/// Launch `pointwise_kernel` for `kind` and `which`.
pub(crate) fn pointwise<R: Runtime, E: FloatElem>(
    value: &Tensor<R, E>,
    params: [&Tensor<R, E>; 3],
    batch: usize,
    kind: Kind,
    which: u32,
) -> Result<Tensor<R, E>> {
    let n = value.len();
    let out = Tensor::<R, E>::empty(value.shape().clone(), value.device());
    if n == 0 {
        return Ok(out);
    }
    let bound = params.map(|p| Bound::new(p, batch));
    let (count, dim) = launch_1d(
        value.client(),
        n.div_ceil(POINTWISE_GROUP as usize),
        work_per_element(kind) * POINTWISE_GROUP as usize,
    );
    unsafe {
        pointwise_kernel::launch_unchecked::<E, R>(
            value.client(),
            count,
            dim,
            value.arg(),
            bound[0].tensor.arg(),
            bound[1].tensor.arg(),
            bound[2].tensor.arg(),
            out.arg(),
            n as u32,
            batch.max(1) as u32,
            bound[0].stride,
            bound[1].stride,
            bound[2].stride,
            f32::INFINITY,
            f32::NAN,
            kind.code(),
            which,
            n != batch,
            POINTWISE_GROUP,
        );
    }
    Ok(out)
}

/// Launch `param_kernel` for `kind` and `which`.
pub(crate) fn parameterwise<R: Runtime, E: FloatElem>(
    params: [&Tensor<R, E>; 3],
    batch: &Shape,
    kind: Kind,
    which: u32,
) -> Result<Tensor<R, E>> {
    let n = batch.num_elements();
    let device = params[0].device();
    let out = Tensor::<R, E>::empty(batch.clone(), device);
    if n == 0 {
        return Ok(out);
    }
    let bound = params.map(|p| Bound::new(p, n));
    let (count, dim) = launch_1d(device.client(), n, work_per_element(kind));
    unsafe {
        param_kernel::launch_unchecked::<E, R>(
            device.client(),
            count,
            dim,
            bound[0].tensor.arg(),
            bound[1].tensor.arg(),
            bound[2].tensor.arg(),
            out.arg(),
            n as u32,
            bound[0].stride,
            bound[1].stride,
            bound[2].stride,
            f32::INFINITY,
            f32::NAN,
            kind.code(),
            which,
        );
    }
    Ok(out)
}

/// Launch `sample_kernel` into a buffer of `shape`, whose trailing `batch`
/// elements index the parameters.
pub(crate) fn sample<R: Runtime, E: FloatElem>(
    params: [&Tensor<R, E>; 3],
    shape: Shape,
    batch: usize,
    offset: u64,
    seed: u64,
    kind: Kind,
) -> Result<Tensor<R, E>> {
    let n = shape.num_elements();
    let device = params[0].device();
    let out = Tensor::<R, E>::empty(shape, device);
    if n == 0 {
        return Ok(out);
    }
    let bound = params.map(|p| Bound::new(p, batch));
    // The four-at-a-time path needs each unit's group of four to be a whole Philox
    // counter, which it is exactly when the offset is a multiple of four.
    let lane_batched = kind.lane_sampled() && offset.is_multiple_of(4);
    let wide = rng::wide_multiply(device.client());
    let units = if lane_batched { n.div_ceil(4) } else { n };
    let (count, dim) = launch_1d(device.client(), units, work_per_element(kind).max(64));
    unsafe {
        sample_kernel::launch_unchecked::<E, R>(
            device.client(),
            count,
            dim,
            bound[0].tensor.arg(),
            bound[1].tensor.arg(),
            bound[2].tensor.arg(),
            out.arg(),
            n as u32,
            batch.max(1) as u32,
            bound[0].stride,
            bound[1].stride,
            bound[2].stride,
            offset as u32,
            (offset >> 32) as u32,
            seed as u32,
            (seed >> 32) as u32,
            kind.code(),
            lane_batched,
            wide,
        );
    }
    Ok(out)
}

/// Roughly how many element operations one lane of a kernel for `kind` performs.
///
/// [`launch_1d`] uses this to decide how wide to make a cube on runtimes where a
/// unit is an operating-system thread and the answer depends on whether the work
/// amortises the dispatch. A normal log-density is a handful of operations; a
/// Student's t pays for two `lgamma`s, and a rejection sampler for as many as it
/// takes. Being roughly right is enough — the number only picks a thread count.
fn work_per_element(kind: Kind) -> usize {
    match kind {
        Kind::Gamma
        | Kind::InverseGamma
        | Kind::Beta
        | Kind::StudentT
        | Kind::FisherSnedecor
        | Kind::Poisson
        | Kind::Binomial
        | Kind::NegativeBinomial
        | Kind::Kumaraswamy => 200,
        Kind::VonMises => 120,
        Kind::Normal | Kind::LogNormal | Kind::HalfNormal => 40,
        _ => 16,
    }
}

/// The reverse pass of `pointwise_kernel`, fused the same way the forward is.
///
/// Four possible outputs, each gated by a `#[comptime]` flag: a caller that only
/// wants `∂/∂logits` compiles a kernel that only computes and only writes that. The
/// unwanted buffers are never touched, so they can be — and are — one element wide.
#[cube(launch_unchecked)]
fn pointwise_grad_kernel<E: Float + CubeElement>(
    upstream: &Array<E>,
    value: &Array<E>,
    pa: &Array<E>,
    pb: &Array<E>,
    pc: &Array<E>,
    gx: &mut Array<E>,
    ga: &mut Array<E>,
    gb: &mut Array<E>,
    gc: &mut Array<E>,
    n: u32,
    batch: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    #[comptime] kind: u32,
    #[comptime] which: u32,
    #[comptime] tiled: bool,
    #[comptime] want_x: bool,
    #[comptime] want_a: bool,
    #[comptime] want_b: bool,
    #[comptime] want_c: bool,
) {
    if ABSOLUTE_POS < n as usize {
        let i = ABSOLUTE_POS;
        let mut p = i;
        if comptime!(tiled) {
            p = i % batch as usize;
        }
        let x = f32::cast_from(value[i]);
        let a = f32::cast_from(pa[p * stride_a as usize]);
        let b = f32::cast_from(pb[p * stride_b as usize]);
        let c = f32::cast_from(pc[p * stride_c as usize]);
        let up = f32::cast_from(upstream[i]);
        // The struct comes apart into four scalars because a `#[cube]` variable
        // cannot be reassigned wholesale from a branch.
        let mut dx: f32 = 0.0;
        let mut da: f32 = 0.0;
        let mut db: f32 = 0.0;
        let mut dc: f32 = 0.0;
        if comptime!(which == 0) {
            let g = log_prob_grad_of(x, a, b, c, kind);
            dx = g.dx;
            da = g.da;
            db = g.db;
            dc = g.dc;
        } else {
            let g = icdf_grad_of(x, a, b, c, kind);
            dx = g.dx;
            da = g.da;
            db = g.db;
            dc = g.dc;
        }
        if comptime!(want_x) {
            gx[i] = E::cast_from(up * dx);
        }
        if comptime!(want_a) {
            ga[i] = E::cast_from(up * da);
        }
        if comptime!(want_b) {
            gb[i] = E::cast_from(up * db);
        }
        if comptime!(want_c) {
            gc[i] = E::cast_from(up * dc);
        }
    }
}

/// The reverse pass of an entropy, and of a reparameterised draw.
///
/// Both take no value argument, which is the only reason they share a kernel: an
/// entropy has none, and a draw's "value" is a uniform the caller never sees.
#[cube(launch_unchecked)]
fn param_grad_kernel<E: Float + CubeElement>(
    upstream: &Array<E>,
    pa: &Array<E>,
    pb: &Array<E>,
    pc: &Array<E>,
    ga: &mut Array<E>,
    gb: &mut Array<E>,
    gc: &mut Array<E>,
    n: u32,
    batch: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    offset_lo: u32,
    offset_hi: u32,
    key_lo: u32,
    key_hi: u32,
    inf: f32,
    nan: f32,
    #[comptime] kind: u32,
    #[comptime] which: u32,
    #[comptime] wide: bool,
    #[comptime] want_a: bool,
    #[comptime] want_b: bool,
    #[comptime] want_c: bool,
) {
    if ABSOLUTE_POS < n as usize {
        let i = ABSOLUTE_POS;
        let p = i % batch as usize;
        let a = f32::cast_from(pa[p * stride_a as usize]);
        let b = f32::cast_from(pb[p * stride_b as usize]);
        let c = f32::cast_from(pc[p * stride_c as usize]);
        let up = f32::cast_from(upstream[i]);
        let mut da: f32 = 0.0;
        let mut db: f32 = 0.0;
        let mut dc: f32 = 0.0;
        if comptime!(which == 3) {
            let nf = NonFinite { inf, nan };
            let g = entropy_grad_of(a, b, c, nf, kind);
            da = g.da;
            db = g.db;
            dc = g.dc;
        } else {
            let lo = offset_lo + i as u32;
            let mut hi = offset_hi;
            if lo < offset_lo {
                hi += 1u32;
            }
            let g = rsample_grad_of(a, b, c, lo, hi, key_lo, key_hi, kind, wide);
            da = g.da;
            db = g.db;
            dc = g.dc;
        }
        if comptime!(want_a) {
            ga[i] = E::cast_from(up * da);
        }
        if comptime!(want_b) {
            gb[i] = E::cast_from(up * db);
        }
        if comptime!(want_c) {
            gc[i] = E::cast_from(up * dc);
        }
    }
}

/// A one-element scratch buffer, bound where a kernel is told not to write.
fn dummy<R: Runtime, E: FloatElem>(device: &crate::backend::Device<R>) -> Tensor<R, E> {
    Tensor::empty(Shape::new(vec![1]), device)
}

/// Which of the four gradient slots a caller wants.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wants {
    /// `∂/∂value`.
    pub x: bool,
    /// `∂/∂a`.
    pub a: bool,
    /// `∂/∂b`.
    pub b: bool,
    /// `∂/∂c`.
    pub c: bool,
}

/// What a gradient launch produced: one full-batch tensor per requested slot.
#[derive(Debug)]
pub struct Grads<R: Runtime, E: FloatElem> {
    /// `∂/∂value`, if asked for.
    pub x: Option<Tensor<R, E>>,
    /// `∂/∂a`, if asked for.
    pub a: Option<Tensor<R, E>>,
    /// `∂/∂b`, if asked for.
    pub b: Option<Tensor<R, E>>,
    /// `∂/∂c`, if asked for.
    pub c: Option<Tensor<R, E>>,
}

/// Launch `pointwise_grad_kernel`.
pub(crate) fn pointwise_grad<R: Runtime, E: FloatElem>(
    upstream: &Tensor<R, E>,
    value: &Tensor<R, E>,
    params: [&Tensor<R, E>; 3],
    batch: usize,
    kind: Kind,
    which: u32,
    wants: Wants,
) -> Result<Grads<R, E>> {
    let n = value.len();
    let device = value.device();
    let make = |want: bool| want.then(|| Tensor::<R, E>::empty(value.shape().clone(), device));
    let out = Grads {
        x: make(wants.x),
        a: make(wants.a),
        b: make(wants.b),
        c: make(wants.c),
    };
    if n == 0 {
        return Ok(out);
    }
    let scratch = dummy::<R, E>(device);
    let bind = |slot: &Option<Tensor<R, E>>| slot.as_ref().unwrap_or(&scratch).arg();
    let bound = params.map(|p| Bound::new(p, batch));
    let (count, dim) = launch_1d(value.client(), n, work_per_element(kind));
    unsafe {
        pointwise_grad_kernel::launch_unchecked::<E, R>(
            value.client(),
            count,
            dim,
            upstream.arg(),
            value.arg(),
            bound[0].tensor.arg(),
            bound[1].tensor.arg(),
            bound[2].tensor.arg(),
            bind(&out.x),
            bind(&out.a),
            bind(&out.b),
            bind(&out.c),
            n as u32,
            batch.max(1) as u32,
            bound[0].stride,
            bound[1].stride,
            bound[2].stride,
            kind.code(),
            which,
            n != batch,
            wants.x,
            wants.a,
            wants.b,
            wants.c,
        );
    }
    Ok(out)
}

/// Launch `param_grad_kernel`.
pub(crate) fn param_grad<R: Runtime, E: FloatElem>(
    upstream: &Tensor<R, E>,
    params: [&Tensor<R, E>; 3],
    batch: usize,
    offset: u64,
    seed: u64,
    kind: Kind,
    which: u32,
    wants: Wants,
) -> Result<Grads<R, E>> {
    let n = upstream.len();
    let device = upstream.device();
    let make = |want: bool| want.then(|| Tensor::<R, E>::empty(upstream.shape().clone(), device));
    let out = Grads {
        x: None,
        a: make(wants.a),
        b: make(wants.b),
        c: make(wants.c),
    };
    if n == 0 {
        return Ok(out);
    }
    let scratch = dummy::<R, E>(device);
    let bind = |slot: &Option<Tensor<R, E>>| slot.as_ref().unwrap_or(&scratch).arg();
    let bound = params.map(|p| Bound::new(p, batch));
    let (count, dim) = launch_1d(device.client(), n, work_per_element(kind));
    unsafe {
        param_grad_kernel::launch_unchecked::<E, R>(
            device.client(),
            count,
            dim,
            upstream.arg(),
            bound[0].tensor.arg(),
            bound[1].tensor.arg(),
            bound[2].tensor.arg(),
            bind(&out.a),
            bind(&out.b),
            bind(&out.c),
            n as u32,
            batch.max(1) as u32,
            bound[0].stride,
            bound[1].stride,
            bound[2].stride,
            offset as u32,
            (offset >> 32) as u32,
            seed as u32,
            (seed >> 32) as u32,
            f32::INFINITY,
            f32::NAN,
            kind.code(),
            which,
            rng::wide_multiply(device.client()),
            wants.a,
            wants.b,
            wants.c,
        );
    }
    Ok(out)
}
