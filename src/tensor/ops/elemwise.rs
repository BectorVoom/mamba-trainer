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
//!
//! # Vectorisation
//!
//! The flat kernels read and write [`Vector`]s rather than scalars: one unit handles
//! `line` adjacent elements, where `line` is the widest width the device likes that
//! divides the element count ([`line_size_for`]). On CubeCL's CPU runtime that maps
//! straight onto SIMD registers — a 256K-element add drops from 122 us to 25 us at
//! 16 lanes — and on GPUs it turns scalar loads into vector loads. A width of `1`
//! compiles to exactly the scalar kernel these used to be, so shapes that do not
//! divide evenly lose nothing.
//!
//! The broadcasting kernels vectorise too, but only when neither operand is
//! broadcast along the trailing axis — that is exactly the condition under which
//! adjacent output elements come from adjacent input elements on both sides. When it
//! holds, the trailing axis is folded into vectors and every stride above it is
//! rescaled, so the per-element index decomposition runs once per vector instead of
//! once per element ([`vectorise_broadcast`]).

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d, line_size_for};
use crate::error::Result;
use crate::tensor::base::Tensor;
use crate::tensor::shape::{Shape, pack_broadcast_meta};

/// The widest vector the trailing axis can be folded into, plus the output shape
/// and operand strides rewritten in those units.
///
/// Folding is only sound when both operands step by one along the trailing axis: a
/// stride of `0` there means one operand repeats a single value across the whole
/// vector, which a vector load cannot express. Every stride above the trailing axis
/// is a multiple of its extent, hence of the width, so rescaling them is exact.
fn vectorise_broadcast<R: Runtime, E: FloatElem>(
    client: &ComputeClient<R>,
    out: &Shape,
    lhs_strides: &[usize],
    rhs_strides: &[usize],
) -> (usize, Shape, Vec<usize>, Vec<usize>) {
    let scalar = || (1, out.clone(), lhs_strides.to_vec(), rhs_strides.to_vec());
    if out.rank() == 0 {
        return scalar();
    }
    let last = out.rank() - 1;
    if lhs_strides[last] != 1 || rhs_strides[last] != 1 {
        return scalar();
    }
    let line = line_size_for::<R, E>(client, out.dim(last));
    if line == 1 {
        return scalar();
    }
    let mut dims = out.dims().to_vec();
    dims[last] /= line;
    let rescale = |strides: &[usize]| {
        let mut v: Vec<usize> = strides.iter().map(|s| s / line).collect();
        v[last] = 1;
        v
    };
    (
        line,
        Shape::new(dims),
        rescale(lhs_strides),
        rescale(rhs_strides),
    )
}

/// Vector width and launch geometry for a flat kernel over `n` elements.
///
/// Buffers are still bound with their scalar element count — CubeCL divides that by
/// the vector width to get the number of vectors a kernel sees.
fn flat_launch<R: Runtime, E: FloatElem>(
    client: &ComputeClient<R>,
    n: usize,
) -> (usize, CubeCount, CubeDim) {
    let line = line_size_for::<R, E>(client, n);
    let (count, dim) = launch_1d(client, n / line, line);
    (line, count, dim)
}

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn fill_kernel<F: Float + CubeElement, N: Size>(output: &mut Array<Vector<F, N>>, value: F) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = Vector::<F, N>::new(value);
    }
}

/// Overwrite every element of `out` with `value`.
pub fn fill_<R: Runtime, E: FloatElem>(out: &Tensor<R, E>, value: f32) {
    let n = out.len();
    if n == 0 {
        return;
    }
    let (line, count, dim) = flat_launch::<R, E>(out.client(), n);
    unsafe {
        fill_kernel::launch_unchecked::<E, R>(
            out.client(),
            count,
            dim,
            line,
            out.arg(),
            E::from_scalar(value),
        );
    }
}

// ---------------------------------------------------------------------------
// Element-type conversion
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn cast_kernel<F1: Float + CubeElement, F2: Float + CubeElement, N: Size>(
    input: &Array<Vector<F1, N>>,
    output: &mut Array<Vector<F2, N>>,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = Vector::<F2, N>::cast_from(input[ABSOLUTE_POS]);
    }
}

/// Convert every element to another float type, rounding where it narrows.
///
/// This is the staging step of the mixed-precision matmul mode: `f32` operands are
/// rounded to `bf16` once, and the product kernels read the narrow copy.
pub fn cast<R: Runtime, E1: FloatElem, E2: FloatElem>(input: &Tensor<R, E1>) -> Tensor<R, E2> {
    let out = Tensor::<R, E2>::empty(input.shape().clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    // Both sides read and write vectors of the same element count, so the width
    // has to be one the device supports for *both* element types.
    let line = line_size_for::<R, E1>(input.client(), n).min(line_size_for::<R, E2>(input.client(), n));
    let (count, dim) = launch_1d(input.client(), n / line, line);
    unsafe {
        cast_kernel::launch_unchecked::<E1, E2, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------

macro_rules! unary_op {
    ($(#[$meta:meta])* $name:ident, $kernel:ident, |$x:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        fn $kernel<F: Float + CubeElement, N: Size>(
            input: &Array<Vector<F, N>>,
            output: &mut Array<Vector<F, N>>,
        ) {
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
            let (line, count, dim) = flat_launch::<R, E>(input.client(), n);
            unsafe {
                $kernel::launch_unchecked::<E, R>(
                    input.client(),
                    count,
                    dim,
                    line,
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
    ///
    /// Written as a multiply because CubeCL's C++ backends (CUDA, HIP) emit their
    /// vector types as plain structs and give them no unary `operator-`; scaling by
    /// `-1` is exact for floats, including the sign of zero.
    neg, neg_kernel, |x| x * Vector::<F, N>::new(F::new(-1.0_f32)));
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
    ///
    /// Spelled out rather than `inverse_sqrt`: HIP resolves a vectorised `rsqrt` to
    /// the `double` overload, and the result then fails to narrow inside the
    /// initializer list CubeCL builds its vector types from.
    rsqrt, rsqrt_kernel, |x| Vector::<F, N>::new(F::new(1.0_f32)) / x.sqrt());
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
    sigmoid, sigmoid_kernel, |x| Vector::<F, N>::new(F::new(1.0_f32))
        / (Vector::<F, N>::new(F::new(1.0_f32))
            + (x * Vector::<F, N>::new(F::new(-1.0_f32))).exp()));
unary_op!(
    /// `max(x, 0)`
    relu, relu_kernel, |x| x.max(Vector::<F, N>::new(F::new(0.0_f32))));
unary_op!(
    /// `-1`, `0` or `1`.
    sign, sign_kernel, |x| select_many(
        x.greater_than(Vector::<F, N>::new(F::new(0.0_f32))),
        Vector::<F, N>::new(F::new(1.0_f32)),
        select_many(
            x.less_than(Vector::<F, N>::new(F::new(0.0_f32))),
            Vector::<F, N>::new(F::new(-1.0_f32)),
            Vector::<F, N>::new(F::new(0.0_f32))
        )
    ));

// ---------------------------------------------------------------------------
// Unary with scalar operands
// ---------------------------------------------------------------------------

macro_rules! unary_scalar_op {
    ($(#[$meta:meta])* $name:ident, $kernel:ident, |$x:ident, $a:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        fn $kernel<F: Float + CubeElement, N: Size>(
            input: &Array<Vector<F, N>>,
            output: &mut Array<Vector<F, N>>,
            scalar: F,
        ) {
            if ABSOLUTE_POS < output.len() {
                let $x = input[ABSOLUTE_POS];
                let $a = Vector::<F, N>::new(scalar);
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
            let (line, count, dim) = flat_launch::<R, E>(input.client(), n);
            unsafe {
                $kernel::launch_unchecked::<E, R>(
                    input.client(),
                    count,
                    dim,
                    line,
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
    powf_scalar, powf_scalar_kernel, |x, a| x.powf(a));
unary_scalar_op!(
    /// `max(x, a)`
    clamp_min, clamp_min_kernel, |x, a| x.max(a));
unary_scalar_op!(
    /// `min(x, a)`
    clamp_max, clamp_max_kernel, |x, a| x.min(a));
unary_scalar_op!(
    /// `1` where `x > a`, else `0`.
    gt_scalar, gt_scalar_kernel, |x, a| select_many(
        x.greater_than(a),
        Vector::<F, N>::new(F::new(1.0_f32)),
        Vector::<F, N>::new(F::new(0.0_f32))
    ));
unary_scalar_op!(
    /// `1` where `x < a`, else `0`.
    lt_scalar, lt_scalar_kernel, |x, a| select_many(
        x.less_than(a),
        Vector::<F, N>::new(F::new(1.0_f32)),
        Vector::<F, N>::new(F::new(0.0_f32))
    ));
unary_scalar_op!(
    /// `x` reduced into `[-a/2, a/2)` by subtracting whole multiples of `a`.
    wrap_to, wrap_to_kernel, |x, a| x - (x / a).round() * a);
unary_scalar_op!(
    /// `1` where `x == a`, else `0`.
    eq_scalar, eq_scalar_kernel, |x, a| select_many(
        x.equal(a),
        Vector::<F, N>::new(F::new(1.0_f32)),
        Vector::<F, N>::new(F::new(0.0_f32))
    ));

#[cube(launch_unchecked)]
fn clamp_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    lo: F,
    hi: F,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].clamp(
            Vector::<F, N>::new(lo),
            Vector::<F, N>::new(hi),
        );
    }
}

/// Clamp every element into `[lo, hi]`.
pub fn clamp<R: Runtime, E: FloatElem>(input: &Tensor<R, E>, lo: f32, hi: f32) -> Tensor<R, E> {
    let out = Tensor::empty(input.shape.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let (line, count, dim) = flat_launch::<R, E>(input.client(), n);
    unsafe {
        clamp_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
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
        fn $flat<F: Float + CubeElement, N: Size>(
            lhs: &Array<Vector<F, N>>,
            rhs: &Array<Vector<F, N>>,
            output: &mut Array<Vector<F, N>>,
        ) {
            if ABSOLUTE_POS < output.len() {
                let $a = lhs[ABSOLUTE_POS];
                let $b = rhs[ABSOLUTE_POS];
                output[ABSOLUTE_POS] = $body;
            }
        }

        #[cube(launch_unchecked)]
        fn $bcast<F: Float + CubeElement, N: Size>(
            lhs: &Array<Vector<F, N>>,
            rhs: &Array<Vector<F, N>>,
            output: &mut Array<Vector<F, N>>,
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
                let (line, count, dim) = flat_launch::<R, E>(lhs.client(), n);
                unsafe {
                    $flat::launch_unchecked::<E, R>(
                        lhs.client(),
                        count,
                        dim,
                        line,
                        lhs.arg(),
                        rhs.arg(),
                        out.arg(),
                    );
                }
                return Ok(out);
            }

            let out_shape = Shape::broadcast(&lhs.shape, &rhs.shape)?;
            let rank = out_shape.rank();
            let out = Tensor::empty(out_shape.clone(), lhs.device());
            let n = out.len();
            if n == 0 {
                return Ok(out);
            }
            let (line, shape, lhs_strides, rhs_strides) = vectorise_broadcast::<R, E>(
                lhs.client(),
                &out_shape,
                &lhs.shape.broadcast_strides(&out_shape)?,
                &rhs.shape.broadcast_strides(&out_shape)?,
            );
            let meta = pack_broadcast_meta(&shape, &lhs_strides, &rhs_strides);
            let meta_handle = lhs.client().create_from_slice(u32::as_bytes(&meta));
            let (count, dim) = launch_1d(lhs.client(), n / line, rank * line);
            unsafe {
                $bcast::launch_unchecked::<E, R>(
                    lhs.client(),
                    count,
                    dim,
                    line,
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
    maximum, max_flat_kernel, max_bcast_kernel, |a, b| a.max(b));
binary_op!(
    /// Elementwise minimum with broadcasting.
    minimum, min_flat_kernel, min_bcast_kernel, |a, b| a.min(b));
binary_op!(
    /// Elementwise power with broadcasting.
    powf, powf_flat_kernel, powf_bcast_kernel, |a, b| a.powf(b));
binary_op!(
    /// `1` where `lhs > rhs`, else `0`.
    greater, gt_flat_kernel, gt_bcast_kernel, |a, b| select_many(
        a.greater_than(b),
        Vector::<F, N>::new(F::new(1.0_f32)),
        Vector::<F, N>::new(F::new(0.0_f32))
    ));

/// Fused multiply-add over three same-shaped tensors: `a * b + c`.
#[cube(launch_unchecked)]
fn mul_add_kernel<F: Float + CubeElement, N: Size>(
    a: &Array<Vector<F, N>>,
    b: &Array<Vector<F, N>>,
    c: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
) {
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
    let (line, count, dim) = flat_launch::<R, E>(a.client(), n);
    unsafe {
        mul_add_kernel::launch_unchecked::<E, R>(
            a.client(),
            count,
            dim,
            line,
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
fn expand_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
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
    let rank = target.rank();
    let out = Tensor::empty(target.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let (line, shape, strides, _) =
        vectorise_broadcast::<R, E>(input.client(), target, &strides, &strides);
    let meta = pack_broadcast_meta(&shape, &strides, &strides);
    let meta_handle = input.client().create_from_slice(u32::as_bytes(&meta));
    let (count, dim) = launch_1d(input.client(), n / line, rank * line);
    unsafe {
        expand_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
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
fn add_assign_kernel<F: Float + CubeElement, N: Size>(
    target: &mut Array<Vector<F, N>>,
    source: &Array<Vector<F, N>>,
) {
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
    let (line, count, dim) = flat_launch::<R, E>(target.client(), n);
    unsafe {
        add_assign_kernel::launch_unchecked::<E, R>(
            target.client(),
            count,
            dim,
            line,
            target.arg(),
            source.arg(),
        );
    }
}
