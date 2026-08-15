//! Reductions along a single axis.
//!
//! Every reduction views the tensor as `[outer, axis, inner]` and assigns one unit
//! per `(outer, inner)` pair. That is simple, correct for any axis, and good enough
//! while the axis is short (state dims, head dims, chunk lengths). Full-tensor
//! reductions iterate the same kernel in a tree so no single unit walks the whole
//! buffer.

use cubecl::prelude::*;

use crate::backend::{ELEMWISE_CUBE_DIM, FloatElem, cube_count_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

macro_rules! reduce_op {
    ($(#[$meta:meta])* $name:ident, $kernel:ident, $init:expr, |$acc:ident, $v:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        fn $kernel<F: Float + CubeElement>(
            input: &Array<F>,
            output: &mut Array<F>,
            axis_len: usize,
            inner: usize,
        ) {
            if ABSOLUTE_POS < output.len() {
                let o = ABSOLUTE_POS / inner;
                let i = ABSOLUTE_POS % inner;
                let base = o * axis_len * inner + i;
                let mut $acc = F::new($init);
                for step in 0..axis_len {
                    let $v = input[base + step * inner];
                    $acc = $body;
                }
                output[ABSOLUTE_POS] = $acc;
            }
        }

        $(#[$meta])*
        pub fn $name<R: Runtime, E: FloatElem>(
            input: &Tensor<R, E>,
            axis: usize,
        ) -> Result<Tensor<R, E>> {
            if axis >= input.rank() {
                return Err(Error::shape(format!(
                    "axis {axis} out of range for shape {}",
                    input.shape
                )));
            }
            let axis_len = input.shape.dim(axis);
            let inner = input.shape.inner(axis);
            let out_shape = input.shape.with_dim(axis, 1);
            let out = Tensor::empty(out_shape, input.device());
            let n = out.len();
            if n == 0 {
                return Ok(out);
            }
            unsafe {
                $kernel::launch_unchecked::<E, R>(
                    input.client(),
                    cube_count_for(n, ELEMWISE_CUBE_DIM),
                    CubeDim::new_1d(ELEMWISE_CUBE_DIM),
                    input.arg(),
                    out.arg(),
                    axis_len,
                    inner,
                );
            }
            Ok(out)
        }
    };
}

reduce_op!(
    /// Sum along `axis`, keeping it with size 1.
    sum_dim, sum_dim_kernel, 0.0_f32, |acc, v| acc + v);
reduce_op!(
    /// Maximum along `axis`, keeping it with size 1.
    max_dim, max_dim_kernel, f32::NEG_INFINITY, |acc, v| F::max(acc, v));
reduce_op!(
    /// Minimum along `axis`, keeping it with size 1.
    min_dim, min_dim_kernel, f32::INFINITY, |acc, v| F::min(acc, v));
reduce_op!(
    /// Product along `axis`, keeping it with size 1.
    prod_dim, prod_dim_kernel, 1.0_f32, |acc, v| acc * v);

/// Mean along `axis`, keeping it with size 1.
pub fn mean_dim<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<Tensor<R, E>> {
    let len = input.shape.dim(axis) as f32;
    let summed = sum_dim(input, axis)?;
    Ok(crate::tensor::ops::elemwise::mul_scalar(&summed, 1.0 / len))
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
    unsafe {
        argmax_kernel::launch_unchecked::<E, R>(
            input.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            input.arg(),
            out.arg(),
            axis_len,
            inner,
        );
    }
    Ok(out)
}
