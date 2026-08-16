//! Reductions along a single axis.
//!
//! Every reduction views the tensor as `[outer, axis, inner]` and assigns one unit
//! per `(outer, inner)` pair. That is simple, correct for any axis, and good enough
//! while the axis is short (state dims, head dims, chunk lengths). Full-tensor
//! reductions iterate the same kernel in a tree so no single unit walks the whole
//! buffer.
//!
//! # Vectorisation
//!
//! Which axis is being reduced decides how a unit can use a vector register:
//!
//! * `inner > 1` — neighbouring lanes are independent reductions over the same
//!   `axis`, so a unit can own `line` of them at once and keep a vector
//!   accumulator. This is the "reduce a middle axis" case.
//! * `inner == 1` — the reduced axis is the contiguous one. A unit strides it
//!   `line` elements at a time into a vector accumulator and folds the lanes
//!   together at the end. This is the case softmax and RMS norm hit.
//!
//! Anything that divides evenly falls into one of the two; everything else runs the
//! scalar kernel, which is what all three compile to at a width of `1`. The second
//! case reassociates the sum — lane `j` accumulates every `j`-th element and the
//! lanes are folded at the end — so a wide `sum_dim` can differ from a narrow one in
//! the last bits.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d, line_size_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// Output elements below which a reduction is not wide enough to fill a GPU.
///
/// One unit per output element is the right shape when there are plenty of them, and
/// the wrong one when there are not: a gradient reduced down to a `[64]` gain vector
/// runs sixteen lanes over the whole device no matter how many elements it reads.
const MIN_OUTPUTS: usize = 8 * 1024;

/// Elements one lane should still walk after a split, so the second pass and the
/// extra launch are worth paying for.
const MIN_STEPS: usize = 32;

/// How many partials to split a reduction of `axis_len` into when it only produces
/// `outputs` of them, or `None` when one pass is already wide enough.
///
/// The factor has to divide `axis_len` exactly — the split is a reshape, and a
/// remainder would need a padded tail — so it is taken as a power of two, which is
/// what every axis this crate reduces happens to be.
fn split_factor(outputs: usize, axis_len: usize) -> Option<usize> {
    if outputs == 0 || outputs >= MIN_OUTPUTS || axis_len < 2 * MIN_STEPS {
        return None;
    }
    let want = MIN_OUTPUTS.div_ceil(outputs);
    let mut groups = 1;
    while groups * 2 <= want
        && axis_len.is_multiple_of(groups * 2)
        && axis_len / (groups * 2) >= MIN_STEPS
    {
        groups *= 2;
    }
    (groups > 1).then_some(groups)
}

/// Every kernel seeds its accumulator with the first element of the axis and starts
/// the loop at one, rather than with an identity constant.
///
/// That is not a micro-optimisation, it is what makes the same source compile on
/// every backend. `max_dim`'s identity is negative infinity, and an infinite float
/// *literal* is a shader-creation error in WGSL; passing it as a runtime scalar
/// instead compiles, but a vector splat of a runtime scalar that is then mutated in
/// a loop makes CubeCL's MLIR pipeline emit IR that fails its own dominance check.
/// A seed read from the buffer sidesteps both, and is exact for infinite inputs too.
/// The identity survives only on the host, for the empty-axis case.
macro_rules! reduce_op {
    ($(#[$meta:meta])* $name:ident, $scaled:ident, $kernel:ident, $folded:ident, $init:expr, |$acc:ident, $v:ident| $body:expr) => {
        /// One unit per output element, reducing `axis_len` strided values.
        ///
        /// With a vector width of `N` a unit owns `N` adjacent `inner` lanes.
        #[cube(launch_unchecked)]
        fn $kernel<F: Float + CubeElement, N: Size>(
            input: &Array<Vector<F, N>>,
            output: &mut Array<Vector<F, N>>,
            scale: F,
            axis_len: usize,
            inner: usize,
        ) {
            if ABSOLUTE_POS < output.len() {
                let o = ABSOLUTE_POS / inner;
                let i = ABSOLUTE_POS % inner;
                let base = o * axis_len * inner + i;
                let mut $acc = input[base];
                for step in 1..axis_len {
                    let $v = input[base + step * inner];
                    $acc = $body;
                }
                output[ABSOLUTE_POS] = $acc * Vector::<F, N>::new(scale);
            }
        }

        /// The `inner == 1` case: the reduced axis is contiguous, so a unit walks it
        /// `N` elements at a time and folds the lanes at the end.
        #[cube(launch_unchecked)]
        fn $folded<F: Float + CubeElement, N: Size>(
            input: &Array<Vector<F, N>>,
            output: &mut Array<F>,
            scale: F,
            axis_lines: usize,
        ) {
            if ABSOLUTE_POS < output.len() {
                let base = ABSOLUTE_POS * axis_lines;
                let mut lanes = input[base];
                for step in 1..axis_lines {
                    let $acc = lanes;
                    let $v = input[base + step];
                    lanes = $body;
                }
                let mut folded = lanes[0];
                #[unroll]
                for lane in 1..N::value() {
                    let $acc = folded;
                    let $v = lanes[lane];
                    folded = $body;
                }
                output[ABSOLUTE_POS] = folded * scale;
            }
        }

        $(#[$meta])*
        pub fn $name<R: Runtime, E: FloatElem>(
            input: &Tensor<R, E>,
            axis: usize,
        ) -> Result<Tensor<R, E>> {
            $scaled(input, axis, 1.0)
        }

        /// The same reduction, with the result scaled before it is written.
        ///
        /// The scale is free — one multiply on a value the unit already holds — and
        /// it saves the separate elementwise pass a mean would otherwise need.
        pub fn $scaled<R: Runtime, E: FloatElem>(
            input: &Tensor<R, E>,
            axis: usize,
            scale: f32,
        ) -> Result<Tensor<R, E>> {
            if axis >= input.rank() {
                return Err(Error::shape(format!(
                    "axis {axis} out of range for shape {}",
                    input.shape
                )));
            }
            crate::backend::trace_shape!("TRACE {} {} axis={axis}", stringify!($name), input.shape);
            let axis_len = input.shape.dim(axis);
            let inner = input.shape.inner(axis);
            let outer = input.shape.num_elements() / (axis_len * inner).max(1);
            if let Some(groups) = split_factor(outer * inner, axis_len) {
                // One unit per output element leaves the device idle when there are
                // few outputs and a long axis: the gain gradient of an RMS norm is
                // `[32768, 64]` summed over the rows, which is sixty-four units
                // walking half a million elements each. Reduce it in two passes
                // instead — `groups` partials per output first, then the partials —
                // and the first pass runs `groups` times as wide.
                let split = Shape::new(vec![outer, groups, axis_len / groups, inner]);
                let partial = $scaled(&input.reshape(split)?, 2, 1.0)?;
                let stacked = Shape::new(vec![outer, groups, inner]);
                let folded = $scaled(&partial.reshape(stacked)?, 1, scale)?;
                return folded.reshape(input.shape.with_dim(axis, 1));
            }
            let out_shape = input.shape.with_dim(axis, 1);
            let out = Tensor::empty(out_shape, input.device());
            let n = out.len();
            if n == 0 {
                return Ok(out);
            }
            if axis_len == 0 {
                // Nothing to seed from; the identity is only ever needed here.
                crate::tensor::ops::elemwise::fill_(&out, $init * scale);
                return Ok(out);
            }
            if inner == 1 {
                let line = line_size_for::<R, E>(input.client(), axis_len);
                let (count, dim) = launch_1d(input.client(), n, axis_len);
                unsafe {
                    $folded::launch_unchecked::<E, R>(
                        input.client(),
                        count,
                        dim,
                        line,
                        input.arg(),
                        out.arg(),
                        E::from_scalar(scale),
                        axis_len / line,
                    );
                }
            } else {
                let line = line_size_for::<R, E>(input.client(), inner);
                let (count, dim) = launch_1d(input.client(), n / line, axis_len * line);
                unsafe {
                    $kernel::launch_unchecked::<E, R>(
                        input.client(),
                        count,
                        dim,
                        line,
                        input.arg(),
                        out.arg(),
                        E::from_scalar(scale),
                        axis_len,
                        inner / line,
                    );
                }
            }
            Ok(out)
        }
    };
}

reduce_op!(
    /// Sum along `axis`, keeping it with size 1.
    sum_dim, sum_dim_scaled, sum_dim_kernel, sum_dim_folded_kernel, 0.0_f32, |acc, v| acc + v);
reduce_op!(
    /// Maximum along `axis`, keeping it with size 1.
    max_dim, max_dim_scaled, max_dim_kernel, max_dim_folded_kernel, f32::NEG_INFINITY, |acc, v| acc.max(v));
reduce_op!(
    /// Minimum along `axis`, keeping it with size 1.
    min_dim, min_dim_scaled, min_dim_kernel, min_dim_folded_kernel, f32::INFINITY, |acc, v| acc.min(v));
reduce_op!(
    /// Product along `axis`, keeping it with size 1.
    prod_dim, prod_dim_scaled, prod_dim_kernel, prod_dim_folded_kernel, 1.0_f32, |acc, v| acc * v);

/// Mean along `axis`, keeping it with size 1.
///
/// One launch, not two: the division rides along inside the reduction.
pub fn mean_dim<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<Tensor<R, E>> {
    let len = input.shape.dim(axis).max(1) as f32;
    sum_dim_scaled(input, axis, 1.0 / len)
}

/// Sum every element into a rank-0 tensor.
///
/// Implemented as a tree of axis reductions so no single unit walks more than
/// `CHUNK` elements.
pub fn sum_all<R: Runtime, E: FloatElem>(input: &Tensor<R, E>) -> Result<Tensor<R, E>> {
    const CHUNK: usize = 512;
    let mut current = input.flatten();
    while current.len() > 1 {
        let n = current.len();
        let groups = n.div_ceil(CHUNK);
        let padded = groups * CHUNK;
        if padded != n {
            let pad = Tensor::<R, E>::zeros(vec![padded - n], input.device());
            current = crate::tensor::ops::movement::cat(&[current, pad], 0)?;
        }
        let reshaped = current.reshape(Shape::new(vec![groups, CHUNK]))?;
        current = sum_dim(&reshaped, 1)?.reshape(Shape::new(vec![groups]))?;
    }
    current.reshape(Shape::scalar())
}

/// Mean of every element as a rank-0 tensor.
pub fn mean_all<R: Runtime, E: FloatElem>(input: &Tensor<R, E>) -> Result<Tensor<R, E>> {
    let n = input.len().max(1) as f32;
    let summed = sum_all(input)?;
    Ok(crate::tensor::ops::elemwise::mul_scalar(&summed, 1.0 / n))
}

#[cube(launch_unchecked)]
fn argmax_kernel<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<u32>, axis_len: usize, inner: usize) {
    if ABSOLUTE_POS < output.len() {
        let o = ABSOLUTE_POS / inner;
        let i = ABSOLUTE_POS % inner;
        let base = o * axis_len * inner + i;
        let mut best = F::cast_from(input[base]);
        let mut best_idx = u32::cast_from(0u32);
        for step in 1..axis_len {
            let v = input[base + step * inner];
            if v > best {
                best = v;
                best_idx = step as u32;
            }
        }
        output[ABSOLUTE_POS] = best_idx;
    }
}

/// Index of the maximum along `axis`, as `u32` ids.
///
/// Stays scalar: the answer is a position, and folding lanes back into a position
/// would cost more than the vector load saves at the widths ids are used at.
pub fn argmax<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<crate::tensor::ops::index::IdTensor<R>> {
    let axis_len = input.shape.dim(axis);
    let inner = input.shape.inner(axis);
    let out_shape = input.shape.without(axis);
    let out = crate::tensor::ops::index::IdTensor::empty(out_shape, input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let (count, dim) = launch_1d(input.client(), n, axis_len);
    unsafe {
        argmax_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            input.arg(),
            out.arg(),
            axis_len,
            inner,
        );
    }
    Ok(out)
}
