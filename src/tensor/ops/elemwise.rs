//! Elementwise kernels: unary maps, scalar maps and broadcasting binary ops.
//!
//! Two kernels back every binary op:
//!
//! * a flat kernel used when both operands already share a shape, and
//! * a broadcasting kernel that decodes the output index into coordinates and
//!   re-offsets each operand with pre-computed (possibly zero) strides.
//!
//! The broadcast metadata buffer layout is
//! `[rank, out_shape[8], lhs_strides[8], rhs_strides[8]]`, matching
//! [`crate::tensor::shape::pack_broadcast_meta`].

use cubecl::prelude::*;

use crate::backend::{ELEMWISE_CUBE_DIM, FloatElem, cube_count_for};
use crate::error::Result;
use crate::tensor::base::Tensor;
use crate::tensor::shape::{Shape, pack_broadcast_meta};

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn fill_kernel<F: Float + CubeElement>(output: &mut Array<F>, value: F) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = value;
    }
}

/// Overwrite every element of `out` with `value`.
pub fn fill_<R: Runtime, E: FloatElem>(out: &Tensor<R, E>, value: f32) {
    let n = out.len();
    if n == 0 {
        return;
    }
    unsafe {
        fill_kernel::launch_unchecked::<E, R>(
            out.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            out.arg(),
            E::from_scalar(value),
        );
    }
}

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------

macro_rules! unary_op {
    ($(#[$meta:meta])* $name:ident, $kernel:ident, |$x:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        fn $kernel<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
            if ABSOLUTE_POS < output.len() {
                let $x = input[ABSOLUTE_POS];
                output[ABSOLUTE_POS] = $body;
            }
        }

        $(#[$meta])*
        pub fn $name<R: Runtime, E: FloatElem>(input: &Tensor<R, E>) -> Tensor<R, E> {
            let out = Tensor::empty(input.shape.clone(), input.device());
            let n = out.len();
            if n == 0 {
                return out;
            }
            unsafe {
                $kernel::launch_unchecked::<E, R>(
                    input.client(),
                    cube_count_for(n, ELEMWISE_CUBE_DIM),
                    CubeDim::new_1d(ELEMWISE_CUBE_DIM),
                    input.arg(),
                    out.arg(),
                );
            }
            out
        }
    };
}

unary_op!(
    /// Bit-for-bit copy into a fresh buffer.
    identity, identity_kernel, |x| x);
unary_op!(
    /// `-x`
    neg, neg_kernel, |x| -x);
unary_op!(
    /// `exp(x)`
    exp, exp_kernel, |x| x.exp());
unary_op!(
    /// Natural logarithm.
    log, log_kernel, |x| x.ln());
unary_op!(
    /// `sqrt(x)`
    sqrt, sqrt_kernel, |x| x.sqrt());
unary_op!(
    /// `1/sqrt(x)`
    rsqrt, rsqrt_kernel, |x| x.inverse_sqrt());
unary_op!(
    /// `1/x`
    recip, recip_kernel, |x| x.recip());
unary_op!(
    /// `|x|`
    abs, abs_kernel, |x| x.abs());
unary_op!(
    /// `tanh(x)`
    tanh, tanh_kernel, |x| x.tanh());
unary_op!(
    /// `sin(x)`
    sin, sin_kernel, |x| x.sin());
unary_op!(
    /// `cos(x)`
    cos, cos_kernel, |x| x.cos());
unary_op!(
    /// Gauss error function.
    erf, erf_kernel, |x| x.erf());
unary_op!(
    /// Round half away from zero. Used by fake quantization; the straight-through
    /// estimator lives in the autodiff layer.
    round, round_kernel, |x| x.round());
unary_op!(
    /// `floor(x)`
    floor, floor_kernel, |x| x.floor());
unary_op!(
    /// Logistic sigmoid.
    sigmoid, sigmoid_kernel, |x| F::new(1.0_f32) / (F::new(1.0_f32) + (-x).exp()));
unary_op!(
    /// `max(x, 0)`
    relu, relu_kernel, |x| F::max(x, F::new(0.0_f32)));
unary_op!(
    /// `-1`, `0` or `1`.
    sign, sign_kernel, |x| select(
        x > F::new(0.0_f32),
        F::new(1.0_f32),
        select(x < F::new(0.0_f32), F::new(-1.0_f32), F::new(0.0_f32))
    ));

// ---------------------------------------------------------------------------
// Unary with scalar operands
// ---------------------------------------------------------------------------

macro_rules! unary_scalar_op {
    ($(#[$meta:meta])* $name:ident, $kernel:ident, |$x:ident, $a:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        fn $kernel<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, $a: F) {
            if ABSOLUTE_POS < output.len() {
                let $x = input[ABSOLUTE_POS];
                output[ABSOLUTE_POS] = $body;
            }
        }

        $(#[$meta])*
        pub fn $name<R: Runtime, E: FloatElem>(input: &Tensor<R, E>, scalar: f32) -> Tensor<R, E> {
            let out = Tensor::empty(input.shape.clone(), input.device());
            let n = out.len();
            if n == 0 {
                return out;
            }
            unsafe {
                $kernel::launch_unchecked::<E, R>(
                    input.client(),
                    cube_count_for(n, ELEMWISE_CUBE_DIM),
                    CubeDim::new_1d(ELEMWISE_CUBE_DIM),
                    input.arg(),
                    out.arg(),
                    E::from_scalar(scalar),
                );
            }
            out
        }
    };
}

unary_scalar_op!(
    /// `x + a`
    add_scalar, add_scalar_kernel, |x, a| x + a);
unary_scalar_op!(
    /// `x * a`
    mul_scalar, mul_scalar_kernel, |x, a| x * a);
unary_scalar_op!(
    /// `a - x`
    rsub_scalar, rsub_scalar_kernel, |x, a| a - x);
unary_scalar_op!(
    /// `x ^ a`
    powf_scalar, powf_scalar_kernel, |x, a| F::powf(x, a));
unary_scalar_op!(
    /// `max(x, a)`
    clamp_min, clamp_min_kernel, |x, a| F::max(x, a));
unary_scalar_op!(
    /// `min(x, a)`
    clamp_max, clamp_max_kernel, |x, a| F::min(x, a));
unary_scalar_op!(
    /// `1` where `x > a`, else `0`.
    gt_scalar, gt_scalar_kernel, |x, a| select(x > a, F::new(1.0_f32), F::new(0.0_f32)));
unary_scalar_op!(
    /// `1` where `x < a`, else `0`.
    lt_scalar, lt_scalar_kernel, |x, a| select(x < a, F::new(1.0_f32), F::new(0.0_f32)));
unary_scalar_op!(
    /// `1` where `x == a`, else `0`.
    eq_scalar, eq_scalar_kernel, |x, a| select(x == a, F::new(1.0_f32), F::new(0.0_f32)));

#[cube(launch_unchecked)]
fn clamp_kernel<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, lo: F, hi: F) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = F::clamp(input[ABSOLUTE_POS], lo, hi);
    }
}

/// Clamp every element into `[lo, hi]`.
pub fn clamp<R: Runtime, E: FloatElem>(input: &Tensor<R, E>, lo: f32, hi: f32) -> Tensor<R, E> {
    let out = Tensor::empty(input.shape.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    unsafe {
        clamp_kernel::launch_unchecked::<E, R>(
            input.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            input.arg(),
            out.arg(),
            E::from_scalar(lo),
            E::from_scalar(hi),
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Binary (with NumPy broadcasting)
// ---------------------------------------------------------------------------

macro_rules! binary_op {
    ($(#[$meta:meta])* $name:ident, $flat:ident, $bcast:ident, |$a:ident, $b:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        fn $flat<F: Float + CubeElement>(lhs: &Array<F>, rhs: &Array<F>, output: &mut Array<F>) {
            if ABSOLUTE_POS < output.len() {
                let $a = lhs[ABSOLUTE_POS];
                let $b = rhs[ABSOLUTE_POS];
                output[ABSOLUTE_POS] = $body;
            }
        }

        #[cube(launch_unchecked)]
        fn $bcast<F: Float + CubeElement>(
            lhs: &Array<F>,
            rhs: &Array<F>,
            output: &mut Array<F>,
            meta: &Array<u32>,
            rank: usize,
        ) {
            if ABSOLUTE_POS < output.len() {
                let mut rem = ABSOLUTE_POS;
                let mut lhs_off = 0usize;
                let mut rhs_off = 0usize;
                for i in 0..rank {
                    let d = rank - 1 - i;
                    let size = meta[1 + d] as usize;
                    let coord = rem % size;
                    rem /= size;
                    lhs_off += coord * meta[9 + d] as usize;
                    rhs_off += coord * meta[17 + d] as usize;
                }
                let $a = lhs[lhs_off];
                let $b = rhs[rhs_off];
                output[ABSOLUTE_POS] = $body;
            }
        }

        $(#[$meta])*
        pub fn $name<R: Runtime, E: FloatElem>(
            lhs: &Tensor<R, E>,
            rhs: &Tensor<R, E>,
        ) -> Result<Tensor<R, E>> {
            if lhs.shape == rhs.shape {
                let out = Tensor::empty(lhs.shape.clone(), lhs.device());
                let n = out.len();
                if n == 0 {
                    return Ok(out);
                }
                unsafe {
                    $flat::launch_unchecked::<E, R>(
                        lhs.client(),
                        cube_count_for(n, ELEMWISE_CUBE_DIM),
                        CubeDim::new_1d(ELEMWISE_CUBE_DIM),
                        lhs.arg(),
                        rhs.arg(),
                        out.arg(),
                    );
                }
                return Ok(out);
            }

            let out_shape = Shape::broadcast(&lhs.shape, &rhs.shape)?;
            let meta = pack_broadcast_meta(
                &out_shape,
                &lhs.shape.broadcast_strides(&out_shape)?,
                &rhs.shape.broadcast_strides(&out_shape)?,
            );
            let rank = out_shape.rank();
            let out = Tensor::empty(out_shape, lhs.device());
            let n = out.len();
            if n == 0 {
                return Ok(out);
            }
            let meta_handle = lhs.client().create_from_slice(u32::as_bytes(&meta));
            unsafe {
                $bcast::launch_unchecked::<E, R>(
                    lhs.client(),
                    cube_count_for(n, ELEMWISE_CUBE_DIM),
                    CubeDim::new_1d(ELEMWISE_CUBE_DIM),
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    ArrayArg::from_raw_parts(meta_handle, meta.len()),
                    rank,
                );
            }
            Ok(out)
        }
    };
}

binary_op!(
    /// Elementwise sum with broadcasting.
    add, add_flat_kernel, add_bcast_kernel, |a, b| a + b);
binary_op!(
    /// Elementwise difference with broadcasting.
    sub, sub_flat_kernel, sub_bcast_kernel, |a, b| a - b);
binary_op!(
    /// Elementwise product with broadcasting.
    mul, mul_flat_kernel, mul_bcast_kernel, |a, b| a * b);
binary_op!(
    /// Elementwise quotient with broadcasting.
    div, div_flat_kernel, div_bcast_kernel, |a, b| a / b);
binary_op!(
    /// Elementwise maximum with broadcasting.
    maximum, max_flat_kernel, max_bcast_kernel, |a, b| F::max(a, b));
binary_op!(
    /// Elementwise minimum with broadcasting.
    minimum, min_flat_kernel, min_bcast_kernel, |a, b| F::min(a, b));
binary_op!(
    /// Elementwise power with broadcasting.
    powf, powf_flat_kernel, powf_bcast_kernel, |a, b| F::powf(a, b));
binary_op!(
    /// `1` where `lhs > rhs`, else `0`.
    greater, gt_flat_kernel, gt_bcast_kernel, |a, b| select(a > b, F::new(1.0_f32), F::new(0.0_f32)));

/// Fused multiply-add over three same-shaped tensors: `a * b + c`.
#[cube(launch_unchecked)]
fn mul_add_kernel<F: Float + CubeElement>(a: &Array<F>, b: &Array<F>, c: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS] + c[ABSOLUTE_POS];
    }
}

/// `a * b + c`, all three tensors sharing a shape.
pub fn mul_add<R: Runtime, E: FloatElem>(
    a: &Tensor<R, E>,
    b: &Tensor<R, E>,
    c: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    if a.shape != b.shape || a.shape != c.shape {
        return crate::tensor::ops::elemwise::add(&mul(a, b)?, c);
    }
    let out = Tensor::empty(a.shape.clone(), a.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    unsafe {
        mul_add_kernel::launch_unchecked::<E, R>(
            a.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            a.arg(),
            b.arg(),
            c.arg(),
            out.arg(),
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Broadcast / expand
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn expand_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    meta: &Array<u32>,
    rank: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let mut rem = ABSOLUTE_POS;
        let mut off = 0usize;
        for i in 0..rank {
            let d = rank - 1 - i;
            let size = meta[1 + d] as usize;
            let coord = rem % size;
            rem /= size;
            off += coord * meta[9 + d] as usize;
        }
        output[ABSOLUTE_POS] = input[off];
    }
}

/// Materialise `input` broadcast to `target`.
pub fn expand<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    target: &Shape,
) -> Result<Tensor<R, E>> {
    if &input.shape == target {
        return Ok(input.clone());
    }
    let strides = input.shape.broadcast_strides(target)?;
    let meta = pack_broadcast_meta(target, &strides, &strides);
    let rank = target.rank();
    let out = Tensor::empty(target.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let meta_handle = input.client().create_from_slice(u32::as_bytes(&meta));
    unsafe {
        expand_kernel::launch_unchecked::<E, R>(
            input.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            input.arg(),
            out.arg(),
            ArrayArg::from_raw_parts(meta_handle, meta.len()),
            rank,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// In-place accumulation (used by the optimizer and gradient accumulation)
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn add_assign_kernel<F: Float + CubeElement>(target: &mut Array<F>, source: &Array<F>) {
    if ABSOLUTE_POS < target.len() {
        target[ABSOLUTE_POS] += source[ABSOLUTE_POS];
    }
}

/// `target += source` in place. Both tensors must share a shape.
///
/// This is the one operation in the crate that mutates a buffer. It is only used
/// where the caller provably owns the destination (optimizer state, gradient
/// accumulators), never on values reachable from an autodiff graph.
pub fn add_assign_<R: Runtime, E: FloatElem>(target: &Tensor<R, E>, source: &Tensor<R, E>) {
    debug_assert_eq!(target.shape, source.shape);
    let n = target.len();
    if n == 0 {
        return;
    }
    unsafe {
        add_assign_kernel::launch_unchecked::<E, R>(
            target.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            target.arg(),
            source.arg(),
        );
    }
}
