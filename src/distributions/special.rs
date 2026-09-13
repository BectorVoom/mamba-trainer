//! The special functions the distributions are built out of, written once.
//!
//! A distribution library lives or dies on `lgamma`, `digamma`, `erf`, `erfc` and
//! `erfinv`. None of them is a hardware instruction, and the two that CubeCL does
//! offer — `erf` is one — are *polyfilled* differently on different backends, so a
//! kernel that leans on them computes something slightly different depending on
//! where it runs. Everything here is therefore implemented from scratch out of the
//! four operations IEEE-754 pins down exactly (`+`, `-`, `*`, `/`) plus the two
//! libm calls (`exp`, `ln`) that no reasonable implementation can avoid.
//!
//! # One body, two compilations
//!
//! Every function below is compiled **twice**: once by `#[cube]` into device IR,
//! and once by `rustc` into a plain host function in [`host`]. The two share their
//! source, because a hand-written host "reference" beside each kernel tests only
//! that two transcriptions of the same idea agree, and quietly rots the first time
//! one side is edited. Sharing makes the question the tests ask the interesting one:
//! *does the same program produce the same bits when CubeCL compiles it as when
//! rustc does?*
//!
//! The sharing is mechanical rather than clever. `special_host.rs` is this file's
//! marked region with one edit — the `#[cube]` lines dropped — and
//! `distributions_bitexact::host_twin_is_the_same_source` re-derives it and fails if
//! the checked-in copy has drifted. A `macro_rules!` that emitted both
//! would be tidier to look at and does not work: a body arriving through a
//! metavariable carries its own hygiene context, and the `scope` binding `#[cube]`
//! synthesises around it is then invisible to it.
//!
//! It works at all because the two spellings overlap. `f32::exp(x)`, `f32::ln(x)`,
//! `f32::abs(x)`, `f32::sqrt(x)`, `f32::floor(x)`, `f32::sin(x)`, `f32::tan(x)` and
//! `a.max(b)` are valid in both languages and mean the same thing, and the CubeCL
//! house style — no `if` expressions, no early `return`, no recursion — is a subset
//! of Rust rather than a departure from it. What is *not* shared is integer code,
//! where the host panics on an overflow the device is expected to wrap through;
//! [`super::rng`] keeps a separate host twin for exactly that reason.
//!
//! # Calling these
//!
//! The functions in the module root are **device** functions: they exist to be
//! called from inside a `#[cube]` kernel, and calling one from ordinary Rust panics
//! with CubeCL's "unexpanded" message. Their host counterparts, with the same names
//! and the same source, are in [`host`].
//!
//! # Accuracy
//!
//! Everything is `f32`, and the polynomial fits were chosen so that the *evaluation*
//! error dominates the *approximation* error — i.e. so a longer polynomial would not
//! help. Measured against `f64` references over the ranges that matter:
//!
//! | function | worst relative error | in `f32` ulp |
//! |---|---|---|
//! | [`erf_f32()`], \|x\| ≤ 1 | 1.8e-7 | 1.6 |
//! | [`erfc_f32()`], x ≥ 1 | 3.3e-7 | 2.8 |
//! | [`erfinv_f32()`] | 1.2e-7 | 1.0 |
//! | [`lgamma_f32()`] | 3.8e-12 (fit) | rounding-limited |
//! | [`digamma_f32()`] | 9.4e-11 (fit) | rounding-limited |
//! | [`log_i0_f32()`], [`log_i1_f32()`] | 4.4e-7 | 3.7 |
//!
//! The `f32` arithmetic is deliberate: `f16` and `bf16` tensors are cast up on the
//! way in and back down on the way out, because a Lanczos series evaluated in `f16`
//! would be noise. `f64` is not an option — wgpu has no double at all.

// A `#[cube]` function cannot use `if` as an expression, so every branching value
// is a `let mut` initialised before the branch that sets it. The initialiser is dead
// by construction, and the lint that notices has nothing to offer here.
#![allow(clippy::excessive_precision, unused_assignments)]
// Every `#[cube] pub fn` expands to a public module of the same name holding the
// macro's generated `expand` entry points. There is no way to attach documentation
// to a module a proc macro synthesises, so `missing_docs` fires on each of them and
// there is nothing to say in reply. Every item written by hand in this file is
// documented; the allow is for the ones that are not.
#![allow(missing_docs)]

use cubecl::prelude::*;

/// π, rounded to `f32`.
pub const PI: f32 = core::f32::consts::PI;
/// `ln(2π)/2`, the constant term of a Gaussian log-density.
pub const HALF_LN_2PI: f32 = 0.918_938_5;
/// `ln(2)`.
pub const LN_2: f32 = core::f32::consts::LN_2;
/// `√2`.
pub const SQRT_2: f32 = core::f32::consts::SQRT_2;
/// `1/√2`, the scale that turns a Gaussian argument into an `erf` one.
pub const INV_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;
/// Euler–Mascheroni γ, which is `−ψ(1)`.
pub const EULER_GAMMA: f32 = 0.577_215_66;

// The region below is the single source of truth for every special function in the
// crate. `special_host.rs` is generated from it; see the module docs.
//
// >>> shared

/// `exp(−x²)`, without letting `x²`'s rounding error through the exponential.
///
/// `exp` amplifies an error in its argument by the argument itself: at `x = 5`
/// the `f32` rounding of `x²` is about `1e-6`, and the exponential turns that
/// into a part in `10⁶` — eight ulp — of the answer. Splitting `x` into a head
/// with at most eleven significant bits and a tail fixes it, because `xh²` is
/// then exact and the correction factor's argument is small enough that its own
/// rounding does not matter:
///
/// `exp(−x²) = exp(−xh²) · exp(−(x + xh)·xl)`.
///
/// The split is `floor(128x)/128`, not a bit mask, so the same expression works
/// on the host and on the device. Only defined for `x ≥ 0`.
#[cube]
pub fn exp_neg_square(x: f32) -> f32 {
    let head = f32::floor(x * 128.0f32) * 0.0078125f32;
    let tail = x - head;
    f32::exp(-(head * head)) * f32::exp(-((x + head) * tail))
}

/// The Gauss error function.
///
/// Two regimes. Below one, a degree-8 minimax polynomial in `x²` for `erf(x)/x`
/// — the odd symmetry is exact rather than approximated, so the sign of zero and
/// the behaviour near the origin come out right for free. Above one, the
/// complement, because `erf` there is a hair under one and a polynomial for it
/// would be spending all its accuracy reproducing the leading `1`.
#[cube]
pub fn erf_f32(x: f32) -> f32 {
    let a = f32::abs(x);
    let mut out: f32 = 0.0;
    if a < 1.0f32 {
        let t = x * x;
        let mut p: f32 = 1.058968905e-06;
        p = p * t - 1.390945818e-05f32;
        p = p * t + 1.195658042e-04f32;
        p = p * t - 8.542639553e-04f32;
        p = p * t + 5.223783664e-03f32;
        p = p * t - 2.686613239e-02f32;
        p = p * t + 1.128379107e-01f32;
        p = p * t - 3.761263788e-01f32;
        p = p * t + 1.128379226e+00f32;
        out = x * p;
    } else {
        out = 1.0f32 - erfc_positive(a);
        if x < 0.0f32 {
            out = -out;
        }
    }
    out
}

/// The complementary error function, `1 − erf(x)`.
///
/// Computed as its own quantity above one rather than by subtraction, which is
/// what keeps the tail meaningful: `1 − erf(4)` in `f32` is `0` on the nose,
/// while `erfc(4)` is `1.54e-8` and correct to three ulp. A Gaussian CDF far out
/// in its tail is exactly where a policy's log-likelihood needs this.
#[cube]
pub fn erfc_f32(x: f32) -> f32 {
    let mut out: f32 = 0.0;
    if x < 0.0f32 {
        out = 2.0f32 - erfc_f32_nonneg(-x);
    } else {
        out = erfc_f32_nonneg(x);
    }
    out
}

/// [`erfc_f32()`] restricted to `x ≥ 0`, split at one.
#[cube]
pub fn erfc_f32_nonneg(x: f32) -> f32 {
    let mut out: f32 = 0.0;
    if x < 1.0f32 {
        out = 1.0f32 - erf_f32(x);
    } else {
        out = erfc_positive(x);
    }
    out
}

/// `erfc(x)` for `x ≥ 1`, as `exp(−x²)·(1/x)·P(1/x)`.
///
/// `P` approximates `erfc(x)·exp(x²)·x`, which tends to `1/√π` and is smooth all
/// the way in. Two fits rather than one: a single polynomial over the whole of
/// `1/x ∈ (0, 1]` needs degree fourteen to fit, and by then its own alternating
/// coefficients cancel in `f32` worse than the extra terms buy — measured, the
/// one-piece degree-14 version is four times *less* accurate than these two
/// degree-7 pieces.
#[cube]
pub fn erfc_positive(x: f32) -> f32 {
    let s = 1.0f32 / x;
    let mut p: f32 = 0.0;
    if x <= 2.5f32 {
        p = -1.057080925e-02f32;
        p = p * s + 3.865620494e-02f32;
        p = p * s - 1.023724396e-02f32;
        p = p * s - 1.821996868e-01f32;
        p = p * s + 4.353404641e-01f32;
        p = p * s - 4.328425229e-01f32;
        p = p * s + 2.733356692e-02f32;
        p = p * s + 5.621036291e-01f32;
    } else {
        p = 4.157866240e-01f32;
        p = p * s - 3.084259629e-01f32;
        p = p * s - 3.761625886e-01f32;
        p = p * s + 5.029486418e-01f32;
        p = p * s - 8.538408205e-03f32;
        p = p * s - 2.816584706e-01f32;
        p = p * s - 8.388361493e-06f32;
        p = p * s + 5.641896129e-01f32;
    }
    exp_neg_square(x) * s * p
}

/// The shared core of [`erfinv_f32()`] and [`std_normal_icdf_f32()`], after
/// Giles (2012).
///
/// Two polynomials in `w = −ln(1 − y²)`, which is the variable that turns the
/// inverse's two very different regimes — nearly linear near the origin, growing
/// like `√w` towards the tails — into two well-conditioned fits. This is the
/// routine CUDA's own `erfinvf` uses.
///
/// `w` is taken as an argument rather than computed here because it is the *only*
/// place the tails lose accuracy, and its two callers can each form it without
/// cancelling. See [`std_normal_icdf_f32()`].
#[cube]
pub fn erfinv_core(w: f32) -> f32 {
    let mut z = w;
    let mut p: f32 = 0.0;
    if z < 5.0f32 {
        z -= 2.5f32;
        p = 2.81022636e-08f32;
        p = p * z + 3.43273939e-07f32;
        p = p * z - 3.5233877e-06f32;
        p = p * z - 4.39150654e-06f32;
        p = p * z + 0.00021858087f32;
        p = p * z - 0.00125372503f32;
        p = p * z - 0.00417768164f32;
        p = p * z + 0.246640727f32;
        p = p * z + 1.50140941f32;
    } else {
        z = f32::sqrt(z) - 3.0f32;
        p = -0.000200214257f32;
        p = p * z + 0.000100950558f32;
        p = p * z + 0.00134934322f32;
        p = p * z - 0.00367342844f32;
        p = p * z + 0.00573950773f32;
        p = p * z - 0.0076224613f32;
        p = p * z + 0.00943887047f32;
        p = p * z + 1.00167406f32;
        p = p * z + 2.83297682f32;
    }
    p
}

/// The inverse error function.
#[cube]
pub fn erfinv_f32(y: f32) -> f32 {
    erfinv_core(-f32::ln((1.0f32 - y) * (1.0f32 + y))) * y
}

/// `ln(1 + x)`, accurate when `x` is small.
///
/// Kahan's identity: the rounding `1 + x` commits is exactly the quantity
/// `(u − 1)` records, so scaling `ln u` by `x/(u − 1)` undoes it. Written out of
/// `ln` alone rather than calling a native `log1p`, which not every backend has
/// and no two spell the same way.
#[cube]
pub fn log1p_f32(x: f32) -> f32 {
    let u = 1.0f32 + x;
    let mut out = x;
    if u != 1.0f32 {
        out = f32::ln(u) * (x / (u - 1.0f32));
    }
    out
}

/// `exp(x) − 1`, accurate when `x` is small.
///
/// The mirror of [`log1p_f32()`]: away from the origin `exp(x) − 1` is already
/// exact to a rounding, and near it the same rescaling recovers the digits the
/// subtraction would cancel.
#[cube]
pub fn expm1_f32(x: f32) -> f32 {
    let u = f32::exp(x);
    let mut out = u - 1.0f32;
    if u == 1.0f32 {
        out = x;
    } else if f32::abs(x) < 0.7f32 {
        out = (u - 1.0f32) * x / f32::ln(u);
    }
    out
}

/// `ln(1 − exp(x))` for `x < 0`, without cancelling.
///
/// Which of the two spellings is stable flips at `−ln 2`: above it `exp(x)` is
/// close to one and `1 − exp(x)` cancels, so the `expm1` form is needed; below
/// it `exp(x)` is small and `log1p` is the accurate one. This is the function
/// that makes `log(1 − p)` usable when a Bernoulli's `p` is given as a logit.
#[cube]
pub fn log1mexp_f32(x: f32) -> f32 {
    let mut out: f32 = 0.0;
    if x > -LN_2 {
        out = f32::ln(-expm1_f32(x));
    } else {
        out = log1p_f32(-f32::exp(x));
    }
    out
}

/// `ln(exp(a) + exp(b))`, shifted so neither exponential overflows.
#[cube]
pub fn logaddexp_f32(a: f32, b: f32) -> f32 {
    let hi = a.max(b);
    let lo = a.min(b);
    hi + log1p_f32(f32::exp(lo - hi))
}

/// `ln(1 + exp(x))`, the softplus.
#[cube]
pub fn softplus_f32(x: f32) -> f32 {
    x.max(0.0f32) + log1p_f32(f32::exp(-f32::abs(x)))
}

/// `ln σ(x)`, the log of the logistic sigmoid.
#[cube]
pub fn log_sigmoid_f32(x: f32) -> f32 {
    -softplus_f32(-x)
}

/// `−1`, `0` or `+1`, matching the sign of `x`.
///
/// Written out rather than reached for: CubeCL has no scalar `signum`, and the
/// branchless spellings all disagree with each other at zero.
#[cube]
pub fn sign_f32(x: f32) -> f32 {
    let mut out: f32 = 0.0;
    if x > 0.0f32 {
        out = 1.0f32;
    } else if x < 0.0f32 {
        out = -1.0f32;
    }
    out
}

/// `x · ln(y)`, with `0 · ln 0 = 0`.
///
/// PyTorch's `xlogy`, and the reason a Bernoulli's log-density at a zero
/// probability is finite instead of `NaN`.
#[cube]
pub fn xlogy_f32(x: f32, y: f32) -> f32 {
    let mut out: f32 = 0.0;
    if x != 0.0f32 {
        out = x * f32::ln(y);
    }
    out
}

/// `x · ln(1 + y)`, with `0 · ln 1 = 0` and the small-`y` accuracy of
/// [`log1p_f32()`].
#[cube]
pub fn xlog1py_f32(x: f32, y: f32) -> f32 {
    let mut out: f32 = 0.0;
    if x != 0.0f32 {
        out = x * log1p_f32(y);
    }
    out
}

/// `ln |Γ(x)|`, by the Lanczos approximation with `g = 7` and nine coefficients.
///
/// Below a half the series is not valid, so the reflection formula moves the
/// argument across and pays for it with a sine. The sine's argument is reduced
/// modulo one first: `sin(πx)` for `x = −40.3` evaluated directly would be
/// asking `f32` to resolve a phase of 127 radians, and the reduction makes the
/// accuracy of `lgamma(−40.3)` the same as that of `lgamma(−0.3)`.
///
/// The reflection is written as a branch rather than a recursive call because a
/// `#[cube]` function cannot call itself.
#[cube]
pub fn lgamma_f32(x: f32) -> f32 {
    let mut z = x;
    let mut reflected: f32 = 0.0;
    if x < 0.5f32 {
        z = 1.0f32 - x;
        reflected = 1.0f32;
    }
    let w = z - 1.0f32;
    let t = w + 7.5f32;
    let mut a: f32 = 0.99999999999980993;
    a += 676.5203681218851f32 / (w + 1.0f32);
    a += -1259.1392167224028f32 / (w + 2.0f32);
    a += 771.32342877765313f32 / (w + 3.0f32);
    a += -176.61502916214059f32 / (w + 4.0f32);
    a += 12.507343278686905f32 / (w + 5.0f32);
    a += -0.13857109526572012f32 / (w + 6.0f32);
    a += 9.9843695780195716e-6f32 / (w + 7.0f32);
    a += 1.5056327351493116e-7f32 / (w + 8.0f32);
    let core = HALF_LN_2PI + (w + 0.5f32) * f32::ln(t) - t + f32::ln(a);
    let mut out = core;
    if reflected != 0.0f32 {
        let frac = x - f32::floor(x);
        out = f32::ln(PI) - f32::ln(f32::abs(f32::sin(PI * frac))) - core;
    }
    out
}

/// `ψ(x) = d/dx ln Γ(x)`, the digamma function.
///
/// The asymptotic series in `1/x` is only good for large `x`, so the recurrence
/// `ψ(x) = ψ(x + 1) − 1/x` walks the argument up to eight first — measured, that
/// threshold puts the series' own error at `9e-11`, two orders below what `f32`
/// can hold, while costing at most eight divisions. Negative arguments reflect
/// through `ψ(1 − x) − ψ(x) = π cot(πx)`, with the same modulo-one reduction
/// [`lgamma_f32()`] uses and for the same reason.
#[cube]
pub fn digamma_f32(x: f32) -> f32 {
    let mut z = x;
    let mut reflected: f32 = 0.0;
    if x < 0.0f32 {
        z = 1.0f32 - x;
        reflected = 1.0f32;
    }
    let mut acc: f32 = 0.0;
    let mut y = z;
    while y < 8.0f32 {
        acc -= 1.0f32 / y;
        y += 1.0f32;
    }
    let inv = 1.0f32 / y;
    let inv2 = inv * inv;
    acc += f32::ln(y) - 0.5f32 * inv;
    acc -= inv2
        * (0.083333333f32
            - inv2
                * (0.0083333333f32
                    - inv2
                        * (0.0039682540f32 - inv2 * (0.0041666667f32 - inv2 * 0.0075757576f32))));
    let mut out = acc;
    if reflected != 0.0f32 {
        let frac = x - f32::floor(x);
        out = acc - PI / f32::tan(PI * frac);
    }
    out
}

/// `ψ'(x) = d²/dx² ln Γ(x)`, the trigamma function.
///
/// The same shape as [`digamma_f32()`] — recurrence up to eight, then the Bernoulli
/// asymptotic series, then a reflection for negative arguments — and for the same
/// reasons. It exists because the *gradient* of an entropy needs it: differentiating
/// a Gamma's or a Beta's entropy with respect to its concentration differentiates a
/// digamma, and an entropy bonus in a policy-gradient loss is exactly that
/// derivative.
#[cube]
pub fn trigamma_f32(x: f32) -> f32 {
    let mut z = x;
    let mut reflected: f32 = 0.0;
    if x < 0.0f32 {
        z = 1.0f32 - x;
        reflected = 1.0f32;
    }
    let mut acc: f32 = 0.0;
    let mut y = z;
    while y < 8.0f32 {
        acc += 1.0f32 / (y * y);
        y += 1.0f32;
    }
    let inv = 1.0f32 / y;
    let inv2 = inv * inv;
    acc += inv + 0.5f32 * inv2;
    acc += inv2
        * inv
        * (0.16666667f32
            + inv2
                * (inv2 * (0.023809524f32 + inv2 * (inv2 * 0.075757576f32 - 0.033333333f32))
                    - 0.033333333f32));
    let mut out = acc;
    if reflected != 0.0f32 {
        let frac = x - f32::floor(x);
        let s = f32::sin(PI * frac);
        out = PI * PI / (s * s) - acc;
    }
    out
}

/// `ln B(a, b)`, the log of the beta function.
#[cube]
pub fn lbeta_f32(a: f32, b: f32) -> f32 {
    lgamma_f32(a) + lgamma_f32(b) - lgamma_f32(a + b)
}

/// `ln C(n, k)`, the log of a binomial coefficient, for real `n` and `k`.
#[cube]
pub fn log_binom_f32(n: f32, k: f32) -> f32 {
    lgamma_f32(n + 1.0f32) - lgamma_f32(k + 1.0f32) - lgamma_f32(n - k + 1.0f32)
}

/// `ln I₀(x)`, the log of the modified Bessel function of the first kind.
///
/// Abramowitz & Stegun 9.8.1 below `3.75` and 9.8.2 above it, the second already
/// factored as `exp(x)/√x` so the logarithm is taken of the slowly varying part
/// and `I₀(700)` — which overflows every float there is — costs nothing to
/// express. Von Mises needs this as its normaliser.
#[cube]
pub fn log_i0_f32(x: f32) -> f32 {
    let a = f32::abs(x);
    let mut out: f32 = 0.0;
    if a < 3.75f32 {
        let r = a / 3.75f32;
        let t = r * r;
        let mut p: f32 = 0.0045813;
        p = p * t + 0.0360768f32;
        p = p * t + 0.2659732f32;
        p = p * t + 1.2067492f32;
        p = p * t + 3.0899424f32;
        p = p * t + 3.5156229f32;
        p = p * t + 1.0f32;
        out = f32::ln(p);
    } else {
        let t = 3.75f32 / a;
        let mut p: f32 = 0.00392377;
        p = p * t - 0.01647633f32;
        p = p * t + 0.02635537f32;
        p = p * t - 0.02057706f32;
        p = p * t + 0.00916281f32;
        p = p * t - 0.00157565f32;
        p = p * t + 0.00225319f32;
        p = p * t + 0.01328592f32;
        p = p * t + 0.39894228f32;
        out = a - 0.5f32 * f32::ln(a) + f32::ln(p);
    }
    out
}

/// `ln I₁(|x|)`, the companion of [`log_i0_f32()`] (A&S 9.8.3 and 9.8.4).
#[cube]
pub fn log_i1_f32(x: f32) -> f32 {
    let a = f32::abs(x);
    let mut out: f32 = 0.0;
    if a < 3.75f32 {
        let r = a / 3.75f32;
        let t = r * r;
        let mut p: f32 = 0.00032411;
        p = p * t + 0.00301532f32;
        p = p * t + 0.02658733f32;
        p = p * t + 0.15084934f32;
        p = p * t + 0.51498869f32;
        p = p * t + 0.87890594f32;
        p = p * t + 0.5f32;
        let la = f32::ln(a);
        let lp = f32::ln(p);
        out = la + lp;
    } else {
        let t = 3.75f32 / a;
        let mut p: f32 = 0.0;
        p = -0.00420059f32;
        p = p * t + 0.01787654f32;
        p = p * t - 0.02895312f32;
        p = p * t + 0.02282967f32;
        p = p * t - 0.01031555f32;
        p = p * t + 0.00163801f32;
        p = p * t - 0.00362018f32;
        p = p * t - 0.03988024f32;
        p = p * t + 0.39894228f32;
        out = a - 0.5f32 * f32::ln(a) + f32::ln(p);
    }
    out
}

/// `I₁(x)/I₀(x)`, the mean resultant length of a von Mises distribution.
///
/// Below eight, formed in log space so the `exp(x)/√x` factor the two Bessel
/// functions share cancels before either is exponentiated — at a concentration of
/// 700 the ratio is a hair under one while `I₀` itself is `1e302`.
///
/// Above eight that route stops working, for the same reason a difference of large
/// logarithms always does: at `x = 700` both logs are near 697 and their `f32`
/// representation error alone is `4e-5`, which is fifty times the whole distance
/// between the ratio and one. The asymptotic branch fits the ratio itself as a
/// degree-6 polynomial in `1/x`, whose leading terms are the expansion
/// `1 − 1/(2x) − 1/(8x²) − …` that the fit reproduces to five digits. That takes the
/// worst case from 550 ulp to under one.
#[cube]
pub fn bessel_ratio_f32(x: f32) -> f32 {
    let a = f32::abs(x);
    let mut out: f32 = 0.0;
    if a >= 8.0f32 {
        let t = 1.0f32 / a;
        let mut p: f32 = 0.0;
        p = -8.631679535e+00f32;
        p = p * t + 1.464498997e+00f32;
        p = p * t - 3.843364120e-01f32;
        p = p * t - 1.159115955e-01f32;
        p = p * t - 1.251997799e-01f32;
        p = p * t - 4.999983907e-01f32;
        p = p * t + 1.000000000e+00f32;
        out = p;
    } else if a > 0.0f32 {
        out = f32::exp(log_i1_f32(a) - log_i0_f32(a));
    }
    out
}

/// The standard normal quantile, `Φ⁻¹(u)`.
///
/// Not `√2 · erfinv(2u − 1)`, though that is the identity. Written that way, a `u`
/// of `1e-6` produces a `y` of `−0.999998`, whose distance from `−1` — the only
/// thing the tail depends on — survives in `f32` to three digits, and the quantile
/// comes out wrong in its fourth. The fix is to notice that `erfinv`'s working
/// variable is `−ln((1 − y)(1 + y))`, which for `y = 2u − 1` is exactly
/// `−ln(4u(1 − u))`: computable straight from `u` with nothing to cancel. Only the
/// final sign-and-scale factor still goes through `2u − 1`, where a relative error
/// is harmless. Measured, this takes `Φ⁻¹(1e-6)` from 5600 ulp to under two.
#[cube]
pub fn std_normal_icdf_f32(u: f32) -> f32 {
    SQRT_2 * erfinv_core(-f32::ln(4.0f32 * u * (1.0f32 - u))) * (2.0f32 * u - 1.0f32)
}

/// The standard normal CDF, `Φ(x)`.
///
/// Spelled `½ erfc(−x/√2)` rather than the textbook `½(1 + erf(x/√2))`. The two
/// agree to a rounding in the bulk, but the textbook form computes
/// `1 + (−1 + ε)` in the left tail and returns a flat zero below `x ≈ −5`, while
/// this one is still returning correct digits at `x = −8`.
#[cube]
pub fn std_normal_cdf_f32(x: f32) -> f32 {
    0.5f32 * erfc_f32(-x * INV_SQRT_2)
}
// <<< shared

/// The host compilation of every function above.
///
/// Same source, different compiler — see the module docs. Public so a caller can
/// write its own reference check, and so the bit-exactness tests can.
#[path = "special_host.rs"]
pub mod host;
