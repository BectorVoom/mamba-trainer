//! The special functions as differentiable tensor operations.
//!
//! [`super::special`] is written for kernels; this is the same mathematics for
//! callers who are composing tensors. Every function here is one fused launch, and
//! every one carries its own adjoint, so `lgamma` and `digamma` are as usable inside
//! a loss as `exp` and `log` already are.
//!
//! That is what lets [`super::kl`] be written as ordinary arithmetic. A Gamma's
//! Kullback–Leibler divergence is a digamma, two log-gammas and some algebra; with
//! digamma as an operation, the divergence needs no kernel and no derivative table
//! of its own, and its gradient is correct because [`crate::autograd`] composed it.
//!
//! The adjoints are the classical ones: `lgamma′ = ψ`, `ψ′ = ψ₁`, `erf′ = 2e^{−x²}/√π`,
//! and so on. [`trigamma`] is where the chain stops — differentiating it would need
//! a tetragamma, which nothing here asks for — so it is the one operation that
//! returns a value with no gradient, and it says so.

// See the note in `univariate.rs` on the modules `#[cube]` synthesises.
#![allow(missing_docs)]

use cubecl::prelude::*;

use crate::autograd::Var;
use crate::backend::{FloatElem, launch_1d};
use crate::tensor::base::Tensor;

use super::special;

/// Which special function [`unary_kernel`] evaluates.
mod code {
    pub const LGAMMA: u32 = 0;
    pub const DIGAMMA: u32 = 1;
    pub const TRIGAMMA: u32 = 2;
    pub const ERF: u32 = 3;
    pub const ERFC: u32 = 4;
    pub const ERFINV: u32 = 5;
    pub const LOG1P: u32 = 6;
    pub const EXPM1: u32 = 7;
    pub const LOG_I0: u32 = 8;
    pub const LOG_I1: u32 = 9;
    pub const BESSEL_RATIO: u32 = 10;
    pub const NDTR: u32 = 11;
    pub const NDTRI: u32 = 12;
    pub const LOG_SIGMOID: u32 = 13;
    pub const LOG1MEXP: u32 = 14;
    /// The derivative of `erfinv`, needed as its own adjoint.
    pub const ERFINV_GRAD: u32 = 15;
    /// The derivative of `ndtri`, likewise.
    pub const NDTRI_GRAD: u32 = 16;
    /// The standard normal density, the adjoint of `ndtr`.
    pub const NDPDF: u32 = 17;
    /// `2e^{−x²}/√π`, the adjoint of `erf`.
    pub const ERF_GRAD: u32 = 18;
}

/// One special function over a flat buffer, chosen at compile time.
#[cube(launch_unchecked)]
fn unary_kernel<E: Float + CubeElement>(
    input: &Array<E>,
    out: &mut Array<E>,
    #[comptime] which: u32,
) {
    if ABSOLUTE_POS < out.len() {
        let x = f32::cast_from(input[ABSOLUTE_POS]);
        let mut r: f32 = 0.0;
        if comptime!(which == 0) {
            r = special::lgamma_f32(x);
        } else if comptime!(which == 1) {
            r = special::digamma_f32(x);
        } else if comptime!(which == 2) {
            r = special::trigamma_f32(x);
        } else if comptime!(which == 3) {
            r = special::erf_f32(x);
        } else if comptime!(which == 4) {
            r = special::erfc_f32(x);
        } else if comptime!(which == 5) {
            r = special::erfinv_f32(x);
        } else if comptime!(which == 6) {
            r = special::log1p_f32(x);
        } else if comptime!(which == 7) {
            r = special::expm1_f32(x);
        } else if comptime!(which == 8) {
            r = special::log_i0_f32(x);
        } else if comptime!(which == 9) {
            r = special::log_i1_f32(x);
        } else if comptime!(which == 10) {
            r = special::bessel_ratio_f32(x);
        } else if comptime!(which == 11) {
            r = special::std_normal_cdf_f32(x);
        } else if comptime!(which == 12) {
            r = special::std_normal_icdf_f32(x);
        } else if comptime!(which == 13) {
            r = special::log_sigmoid_f32(x);
        } else if comptime!(which == 14) {
            r = special::log1mexp_f32(x);
        } else if comptime!(which == 15) {
            // `d/dx erfinv = (√π/2)·exp(erfinv(x)²)`.
            let e = special::erfinv_f32(x);
            r = 0.8862269f32 * f32::exp(e * e);
        } else if comptime!(which == 16) {
            // `d/dq Φ⁻¹ = 1/φ(Φ⁻¹(q))`.
            let z = special::std_normal_icdf_f32(x);
            r = f32::exp(0.5f32 * z * z + special::HALF_LN_2PI);
        } else if comptime!(which == 17) {
            r = f32::exp(-0.5f32 * x * x - special::HALF_LN_2PI);
        } else if comptime!(which == 18) {
            r = core::f32::consts::FRAC_2_SQRT_PI * f32::exp(-x * x);
        }
        out[ABSOLUTE_POS] = E::cast_from(r);
    }
}

/// Evaluate one special function over a tensor.
fn eval<R: Runtime, E: FloatElem>(input: &Tensor<R, E>, which: u32) -> Tensor<R, E> {
    let out = Tensor::<R, E>::empty(input.shape().clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let (count, dim) = launch_1d(input.client(), n, 60);
    unsafe {
        unary_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            input.arg(),
            out.arg(),
            which,
        );
    }
    out
}

/// Generate a differentiable wrapper whose adjoint is another entry in the table.
macro_rules! differentiable {
    ($(
        $(#[$meta:meta])*
        $name:ident = $code:ident, d/dx = $grad:ident;
    )*) => {
        $(
            $(#[$meta])*
            pub fn $name<R: Runtime, E: FloatElem>(x: &Var<R, E>) -> Var<R, E> {
                let value = eval(x.tensor(), code::$code);
                let saved = x.tensor().clone();
                Var::record(value, &[x], || {
                    Box::new(move |g| {
                        let slope = eval(&saved, code::$grad);
                        Ok(vec![Some(crate::tensor::ops::elemwise::mul(g, &slope)?)])
                    })
                })
            }
        )*
    };
}

differentiable! {
    /// `ln Γ(x)`, whose derivative is [`digamma`].
    lgamma = LGAMMA, d/dx = DIGAMMA;
    /// `ψ(x) = d/dx ln Γ(x)`, whose derivative is [`trigamma`].
    digamma = DIGAMMA, d/dx = TRIGAMMA;
    /// The Gauss error function.
    erf = ERF, d/dx = ERF_GRAD;
    /// The inverse error function.
    erfinv = ERFINV, d/dx = ERFINV_GRAD;
    /// The standard normal CDF, `Φ(x)`.
    ndtr = NDTR, d/dx = NDPDF;
    /// The standard normal quantile, `Φ⁻¹(q)`.
    ndtri = NDTRI, d/dx = NDTRI_GRAD;
}

/// `ln(1 + x)`, accurate for small `x`.
pub fn log1p<R: Runtime, E: FloatElem>(x: &Var<R, E>) -> Var<R, E> {
    let value = eval(x.tensor(), code::LOG1P);
    let saved = x.tensor().clone();
    Var::record(value, &[x], || {
        Box::new(move |g| {
            let denom = crate::tensor::ops::elemwise::add_scalar(&saved, 1.0);
            Ok(vec![Some(crate::tensor::ops::elemwise::div(g, &denom)?)])
        })
    })
}

/// `exp(x) − 1`, accurate for small `x`.
pub fn expm1<R: Runtime, E: FloatElem>(x: &Var<R, E>) -> Var<R, E> {
    let value = eval(x.tensor(), code::EXPM1);
    let saved = x.tensor().clone();
    Var::record(value, &[x], || {
        Box::new(move |g| {
            let slope = crate::tensor::ops::elemwise::exp(&saved);
            Ok(vec![Some(crate::tensor::ops::elemwise::mul(g, &slope)?)])
        })
    })
}

/// `erfc(x) = 1 − erf(x)`, exact in the tail where the subtraction is not.
pub fn erfc<R: Runtime, E: FloatElem>(x: &Var<R, E>) -> Var<R, E> {
    let value = eval(x.tensor(), code::ERFC);
    let saved = x.tensor().clone();
    Var::record(value, &[x], || {
        Box::new(move |g| {
            let slope = eval(&saved, code::ERF_GRAD);
            let product = crate::tensor::ops::elemwise::mul(g, &slope)?;
            Ok(vec![Some(crate::tensor::ops::elemwise::neg(&product))])
        })
    })
}

/// `ln I₀(x)`, whose derivative is `I₁(x)/I₀(x)`.
pub fn log_i0<R: Runtime, E: FloatElem>(x: &Var<R, E>) -> Var<R, E> {
    let value = eval(x.tensor(), code::LOG_I0);
    let saved = x.tensor().clone();
    Var::record(value, &[x], || {
        Box::new(move |g| {
            let slope = eval(&saved, code::BESSEL_RATIO);
            Ok(vec![Some(crate::tensor::ops::elemwise::mul(g, &slope)?)])
        })
    })
}

/// `ψ₁(x) = d²/dx² ln Γ(x)`, the trigamma function.
///
/// The one operation here with no gradient: its adjoint would be a tetragamma, and
/// nothing in the crate needs one. Returned as a plain tensor rather than an
/// untracked `Var` so that a caller cannot differentiate it by accident.
pub fn trigamma<R: Runtime, E: FloatElem>(x: &Tensor<R, E>) -> Tensor<R, E> {
    eval(x, code::TRIGAMMA)
}

/// `ln I₁(|x|)`, as a plain tensor.
pub fn log_i1<R: Runtime, E: FloatElem>(x: &Tensor<R, E>) -> Tensor<R, E> {
    eval(x, code::LOG_I1)
}

/// `I₁(x)/I₀(x)`, as a plain tensor.
pub fn bessel_ratio<R: Runtime, E: FloatElem>(x: &Tensor<R, E>) -> Tensor<R, E> {
    eval(x, code::BESSEL_RATIO)
}

/// `ln σ(x)`, as a plain tensor. Use [`crate::autograd::Var::softplus`] for the
/// differentiable form, of which this is the negation at `−x`.
pub fn log_sigmoid<R: Runtime, E: FloatElem>(x: &Tensor<R, E>) -> Tensor<R, E> {
    eval(x, code::LOG_SIGMOID)
}

/// `ln(1 − exp(x))` for `x < 0`, as a plain tensor.
pub fn log1mexp<R: Runtime, E: FloatElem>(x: &Tensor<R, E>) -> Tensor<R, E> {
    eval(x, code::LOG1MEXP)
}
