//! Layout changes: permute, slice, concatenate, flip, repeat.
//!
//! Because tensors are always contiguous, each of these materialises a new buffer.
//! The cost is real but the payoff is that no kernel in the crate ever has to
//! reason about strides.

use cubecl::prelude::*;

use crate::backend::{ELEMWISE_CUBE_DIM, FloatElem, cube_count_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::{MAX_RANK, Shape};

#[cube(launch_unchecked)]
fn strided_copy_kernel<F: Float + CubeElement>(
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
            let size = meta[d] as usize;
            let coord = rem % size;
            rem /= size;
            off += coord * meta[MAX_RANK + d] as usize;
        }
        output[ABSOLUTE_POS] = input[off];
    }
}

/// Copy `input` into a fresh contiguous buffer of shape `out_shape`, reading each
/// output element at `sum(coord[d] * src_strides[d])`.
fn strided_copy<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    out_shape: Shape,
    src_strides: &[usize],
) -> Tensor<R, E> {
    let mut meta = vec![0u32; 2 * MAX_RANK];
    for (d, size) in out_shape.dims().iter().enumerate() {
        meta[d] = *size as u32;
        meta[MAX_RANK + d] = src_strides[d] as u32;
    }
    let rank = out_shape.rank();
    let out = Tensor::empty(out_shape, input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let meta_handle = input.client().create_from_slice(u32::as_bytes(&meta));
    unsafe {
        strided_copy_kernel::launch_unchecked::<E, R>(
            input.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            input.arg(),
            out.arg(),
            ArrayArg::from_raw_parts(meta_handle, meta.len()),
            rank,
        );
    }
    out
}

/// Reorder axes. `perm[i]` names the source axis that becomes output axis `i`.
pub fn permute<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    perm: &[usize],
) -> Result<Tensor<R, E>> {
    if perm.len() != input.rank() {
        return Err(Error::shape(format!(
            "permutation {perm:?} does not match rank {}",
            input.rank()
        )));
    }
    let mut seen = vec![false; perm.len()];
    for &p in perm {
        if p >= perm.len() || seen[p] {
            return Err(Error::shape(format!("{perm:?} is not a permutation")));
        }
        seen[p] = true;
    }
    if perm.iter().enumerate().all(|(i, p)| i == *p) {
        return Ok(input.clone());
    }

    let src_strides = input.shape.strides();
    let out_shape = Shape::new(perm.iter().map(|&p| input.shape.dim(p)).collect::<Vec<_>>());
    let strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();
    Ok(strided_copy(input, out_shape, &strides))
}

/// Swap the last two axes.
pub fn transpose<R: Runtime, E: FloatElem>(input: &Tensor<R, E>) -> Result<Tensor<R, E>> {
    let rank = input.rank();
    if rank < 2 {
        return Err(Error::shape("transpose needs rank >= 2".to_string()));
    }
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(rank - 2, rank - 1);
    permute(input, &perm)
}

/// Swap two arbitrary axes.
pub fn swap_axes<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    a: usize,
    b: usize,
) -> Result<Tensor<R, E>> {
    let mut perm: Vec<usize> = (0..input.rank()).collect();
    perm.swap(a, b);
    permute(input, &perm)
}

/// Reverse `input` along `axis`.
pub fn flip<R: Runtime, E: FloatElem>(input: &Tensor<R, E>, axis: usize) -> Result<Tensor<R, E>> {
    if axis >= input.rank() {
        return Err(Error::shape(format!(
            "axis {axis} out of range for {}",
            input.shape
        )));
    }
    let len = input.shape.dim(axis);
    // Reading with a negative stride is expressed by starting at the far end;
    // easier to express as an explicit index map.
    flip_impl(input, axis, len)
}

#[cube(launch_unchecked)]
fn flip_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    axis_len: usize,
    inner: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let i = ABSOLUTE_POS % inner;
        let rest = ABSOLUTE_POS / inner;
        let d = rest % axis_len;
        let o = rest / axis_len;
        let src = o * axis_len * inner + (axis_len - 1 - d) * inner + i;
        output[ABSOLUTE_POS] = input[src];
    }
}

fn flip_impl<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
    axis_len: usize,
) -> Result<Tensor<R, E>> {
    let inner = input.shape.inner(axis);
    let out = Tensor::empty(input.shape.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    unsafe {
        flip_kernel::launch_unchecked::<E, R>(
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

#[cube(launch_unchecked)]
fn slice_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    src_axis_len: usize,
    dst_axis_len: usize,
    inner: usize,
    start: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let i = ABSOLUTE_POS % inner;
        let rest = ABSOLUTE_POS / inner;
        let d = rest % dst_axis_len;
        let o = rest / dst_axis_len;
        let src = o * src_axis_len * inner + (start + d) * inner + i;
        output[ABSOLUTE_POS] = input[src];
    }
}

/// Take `len` entries starting at `start` along `axis`.
pub fn slice<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
    start: usize,
    len: usize,
) -> Result<Tensor<R, E>> {
    if axis >= input.rank() {
        return Err(Error::shape(format!(
            "axis {axis} out of range for {}",
            input.shape
        )));
    }
    let src_len = input.shape.dim(axis);
    if start + len > src_len {
        return Err(Error::shape(format!(
            "slice [{start}, {}) exceeds axis {axis} of {}",
            start + len,
            input.shape
        )));
    }
    if start == 0 && len == src_len {
        return Ok(input.clone());
    }
    let inner = input.shape.inner(axis);
    let out_shape = input.shape.with_dim(axis, len);
    let out = Tensor::empty(out_shape, input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    unsafe {
        slice_kernel::launch_unchecked::<E, R>(
            input.client(),
            cube_count_for(n, ELEMWISE_CUBE_DIM),
            CubeDim::new_1d(ELEMWISE_CUBE_DIM),
            input.arg(),
            out.arg(),
            src_len,
            len,
            inner,
            start,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn write_slice_kernel<F: Float + CubeElement>(
    src: &Array<F>,
    dst: &mut Array<F>,
    dst_axis_len: usize,
    src_axis_len: usize,
    inner: usize,
    start: usize,
) {
    if ABSOLUTE_POS < src.len() {
        let i = ABSOLUTE_POS % inner;
        let rest = ABSOLUTE_POS / inner;
        let d = rest % src_axis_len;
        let o = rest / src_axis_len;
        dst[o * dst_axis_len * inner + (start + d) * inner + i] = src[ABSOLUTE_POS];
    }
}

/// Concatenate tensors along `axis`. Every other dimension must match.
pub fn cat<R: Runtime, E: FloatElem>(
    parts: &[Tensor<R, E>],
    axis: usize,
) -> Result<Tensor<R, E>> {
    if parts.is_empty() {
        return Err(Error::shape("cat needs at least one tensor".to_string()));
    }
    if parts.len() == 1 {
        return Ok(parts[0].clone());
    }
    let rank = parts[0].rank();
    for p in parts {
        if p.rank() != rank {
            return Err(Error::shape("cat operands must share a rank".to_string()));
        }
        for d in 0..rank {
            if d != axis && p.shape.dim(d) != parts[0].shape.dim(d) {
                return Err(Error::shape(format!(
                    "cat operands disagree outside axis {axis}: {} vs {}",
                    parts[0].shape, p.shape
                )));
            }
        }
    }

    let total: usize = parts.iter().map(|p| p.shape.dim(axis)).sum();
    let out_shape = parts[0].shape.with_dim(axis, total);
    let inner = parts[0].shape.inner(axis);
    let out = Tensor::empty(out_shape, parts[0].device());

    let mut offset = 0usize;
    for p in parts {
        let n = p.len();
        if n > 0 {
            unsafe {
                write_slice_kernel::launch_unchecked::<E, R>(
                    p.client(),
                    cube_count_for(n, ELEMWISE_CUBE_DIM),
                    CubeDim::new_1d(ELEMWISE_CUBE_DIM),
                    p.arg(),
                    out.arg(),
                    total,
                    p.shape.dim(axis),
                    inner,
                    offset,
                );
            }
        }
        offset += p.shape.dim(axis);
    }
    Ok(out)
}

/// Split into `n` equal chunks along `axis`.
pub fn chunk<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    n: usize,
    axis: usize,
) -> Result<Vec<Tensor<R, E>>> {
    let len = input.shape.dim(axis);
    if len % n != 0 {
        return Err(Error::shape(format!(
            "axis {axis} of {} is not divisible into {n} chunks",
            input.shape
        )));
    }
    let step = len / n;
    (0..n).map(|i| slice(input, axis, i * step, step)).collect()
}

/// Split along `axis` into pieces with the given sizes.
pub fn split<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    sizes: &[usize],
    axis: usize,
) -> Result<Vec<Tensor<R, E>>> {
    let total: usize = sizes.iter().sum();
    if total != input.shape.dim(axis) {
        return Err(Error::shape(format!(
            "split sizes {sizes:?} sum to {total}, but axis {axis} of {} is {}",
            input.shape,
            input.shape.dim(axis)
        )));
    }
    let mut out = Vec::with_capacity(sizes.len());
    let mut off = 0;
    for &s in sizes {
        out.push(slice(input, axis, off, s)?);
        off += s;
    }
    Ok(out)
}

/// Shift along `axis` by one step towards larger indices, filling the first slot
/// with zeros. This is the `x_{t-1}` term of the trapezoidal recurrence.
pub fn shift_right<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
) -> Result<Tensor<R, E>> {
    let len = input.shape.dim(axis);
    if len == 0 {
        return Ok(input.clone());
    }
    let zeros = Tensor::<R, E>::zeros(input.shape.with_dim(axis, 1), input.device());
    let head = slice(input, axis, 0, len - 1)?;
    cat(&[zeros, head], axis)
}

/// Repeat a size-1 axis `times` times.
pub fn repeat<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
    times: usize,
) -> Result<Tensor<R, E>> {
    if input.shape.dim(axis) != 1 {
        return Err(Error::shape(format!(
            "repeat expects a size-1 axis, got {} at {axis}",
            input.shape.dim(axis)
        )));
    }
    crate::tensor::ops::elemwise::expand(input, &input.shape.with_dim(axis, times))
}
