//! Prefix scans.
//!
//! `cumsum` is the only sequential primitive the SSD algorithm needs: log-decays
//! are accumulated with it, and the segment-sum matrix is then built from
//! broadcast differences. Its adjoint is a reverse cumulative sum, which is why
//! both directions live here.

use cubecl::prelude::*;

use crate::backend::{ELEMWISE_CUBE_DIM, FloatElem, cube_count_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;

#[cube(launch_unchecked)]
fn cumsum_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    axis_len: usize,
    inner: usize,
    lanes: usize,
    #[comptime] reverse: bool,
    #[comptime] exclusive: bool,
) {
    // One unit per (outer, inner) lane, NOT per element: guarding on the output
    // length instead would let surplus units walk past the end of the buffer.
    if ABSOLUTE_POS < lanes {
        let o = ABSOLUTE_POS / inner;
        let i = ABSOLUTE_POS % inner;
        let base = o * axis_len * inner + i;
        let mut acc = F::new(0.0_f32);
        for step in 0..axis_len {
            let idx = if comptime!(reverse) {
                axis_len - 1 - step
            } else {
                step
            };
            let v = input[base + idx * inner];
            if comptime!(exclusive) {
                output[base + idx * inner] = acc;
                acc += v;
            } else {
                acc += v;
                output[base + idx * inner] = acc;
            }
        }
    }
}

fn scan_launch<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
    reverse: bool,
    exclusive: bool,
) -> Result<Tensor<R, E>> {
    if axis >= input.rank() {
        return Err(Error::shape(format!(
            "axis {axis} out of range for {}",
            input.shape
        )));
    }
    let axis_len = input.shape.dim(axis);
    let inner = input.shape.inner(axis);
    let out = Tensor::empty(input.shape.clone(), input.device());
    // One unit per (outer, inner) lane; each lane walks the scanned axis.
    let lanes = input.len() / axis_len.max(1);
    if lanes == 0 {
        return Ok(out);
    }
    unsafe {
        cumsum_kernel::launch_unchecked::<E, R>(
            input.client(),
            cube_count_for(lanes, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            input.arg(),
            out.arg(),
            axis_len,
            inner,
            lanes,
            reverse,
            exclusive,
        );
    }
    Ok(out)
}

/// Inclusive prefix sum along `axis`.
pub fn cumsum<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<Tensor<R, E>> {
    scan_launch(input, axis, false, false)
}

/// Exclusive prefix sum along `axis` (element `i` sums `[0, i)`).
pub fn cumsum_exclusive<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<Tensor<R, E>> {
    scan_launch(input, axis, false, true)
}

/// Inclusive suffix sum along `axis`. This is the adjoint of [`cumsum`].
pub fn cumsum_reverse<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<Tensor<R, E>> {
    scan_launch(input, axis, true, false)
}

/// Exclusive suffix sum along `axis`. This is the adjoint of [`cumsum_exclusive`].
pub fn cumsum_reverse_exclusive<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<Tensor<R, E>> {
    scan_launch(input, axis, true, true)
}
