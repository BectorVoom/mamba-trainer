//! Layout changes: permute, slice, concatenate, flip, repeat.
//!
//! Because tensors are always contiguous, each of these materialises a new buffer.
//! The cost is real but the payoff is that no kernel in the crate ever has to
//! reason about strides.
//!
//! [`slice`], [`cat`] and [`flip`] all move whole runs of `inner` adjacent elements
//! and only reorder the axes above them, so a unit can carry a [`Vector`] of `inner`
//! at a time: every index in those kernels is a multiple of `inner`, and dividing
//! them all by the vector width leaves the arithmetic unchanged.
//!
//! [`permute`] is the awkward one, because reordering axes is exactly what breaks
//! adjacency — but it breaks it less often than it looks. Two properties recover
//! most of the cost, and between them they took the strided copy from 12% of a
//! training step to under 4%:
//!
//! * **Axes that stayed adjacent can be merged.** Output axes `d` and `d+1` describe
//!   one contiguous run of source memory whenever `src_stride[d] == src_stride[d+1] *
//!   dim[d+1]`, so they can be folded into a single axis before the kernel ever runs.
//!   A rank-5 permutation like `[0, 1, 3, 2, 4]` — the one the scan uses to put heads
//!   in front of positions — collapses to rank 3, and the per-element index
//!   arithmetic is a division and a modulo per axis.
//! * **The innermost axis is often untouched.** If it is, the source stride for it is
//!   `1`, the copy moves whole vectors, and both the loads and the index arithmetic
//!   are divided by the vector width.
//!
//! What is left after that — a permutation that genuinely transposes the contiguous
//! axis, such as the trailing swap in `[0, 1, 2, 4, 3]` — still runs the scalar
//! kernel, which is the honest cost of moving those bytes.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d, line_size_for};
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

/// The same kernel over vectors, for the case where the innermost output axis is
/// also contiguous in the source.
///
/// Every extent and stride in `meta` is already divided by the vector width, which
/// is legal precisely because the innermost source stride is `1`: every other stride
/// is then a multiple of that axis's extent, and the width divides it.
#[cube(launch_unchecked)]
fn strided_copy_vec_kernel<F: Float + CubeElement, N: Size>(
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
            let size = meta[d] as usize;
            let coord = rem % size;
            rem /= size;
            off += coord * meta[MAX_RANK + d] as usize;
        }
        output[ABSOLUTE_POS] = input[off];
    }
}

/// Drop extent-1 axes and merge every pair that describes one contiguous run.
///
/// Both transformations leave the element-by-element mapping identical: an axis of
/// extent 1 contributes a coordinate that is always zero, and axes `d`, `d+1` with
/// `stride[d] == stride[d+1] * dim[d+1]` enumerate exactly the addresses a single
/// axis of extent `dim[d] * dim[d+1]` and stride `stride[d+1]` does.
fn coalesce_axes(dims: &[usize], strides: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut d: Vec<usize> = Vec::with_capacity(dims.len());
    let mut s: Vec<usize> = Vec::with_capacity(dims.len());
    for (dim, stride) in dims.iter().zip(strides) {
        if *dim == 1 {
            continue;
        }
        if let Some(last) = d.len().checked_sub(1)
            && s[last] == stride * dim
        {
            d[last] *= dim;
            s[last] = *stride;
            continue;
        }
        d.push(*dim);
        s.push(*stride);
    }
    if d.is_empty() {
        d.push(1);
        s.push(0);
    }
    (d, s)
}

/// Copy `input` into a fresh contiguous buffer of shape `out_shape`, reading each
/// output element at `sum(coord[d] * src_strides[d])`.
fn strided_copy<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    out_shape: Shape,
    src_strides: &[usize],
) -> Tensor<R, E> {
    let _op = crate::backend::tally_op_scope("strided_copy");
    crate::backend::trace_shape!("TRACE strided_copy {} -> {out_shape}", input.shape);
    let (mut dims, mut strides) = coalesce_axes(out_shape.dims(), src_strides);
    let out = Tensor::empty(out_shape, input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }

    // A trailing source stride of 1 means the copy moves whole runs, so it can move
    // them a vector at a time.
    let innermost = *dims.last().expect("at least one axis");
    let line = if strides.last() == Some(&1) {
        line_size_for::<R, E>(input.client(), innermost)
    } else {
        1
    };
    if line > 1 {
        let last = dims.len() - 1;
        dims[last] /= line;
        for stride in &mut strides[..last] {
            *stride /= line;
        }
    }

    let rank = dims.len();
    debug_assert!(rank <= MAX_RANK);
    let mut meta = vec![0u32; 2 * MAX_RANK];
    for (d, (size, stride)) in dims.iter().zip(&strides).enumerate() {
        meta[d] = *size as u32;
        meta[MAX_RANK + d] = *stride as u32;
    }
    let meta_handle = crate::backend::meta_handle(input.device(), &meta);
    let (count, dim) = launch_1d(input.client(), n / line, rank);
    unsafe {
        if line > 1 {
            strided_copy_vec_kernel::launch_unchecked::<E, R>(
                input.client(),
                count,
                dim,
                line,
                input.arg(),
                out.arg(),
                ArrayArg::from_raw_parts(meta_handle, meta.len()),
                rank,
            );
        } else {
            strided_copy_kernel::launch_unchecked::<E, R>(
                input.client(),
                count,
                dim,
                input.arg(),
                out.arg(),
                ArrayArg::from_raw_parts(meta_handle, meta.len()),
                rank,
            );
        }
    }
    out
}

/// Whether reading `dims` with `strides` walks memory in order, so that the view is
/// the same bytes in the same sequence and no copy is needed.
///
/// Axes of extent 1 are skipped: their stride is never used, so it cannot break
/// contiguity.
fn is_contiguous(dims: &[usize], strides: &[usize]) -> bool {
    let mut expected = 1;
    for axis in (0..dims.len()).rev() {
        if dims[axis] == 1 {
            continue;
        }
        if strides[axis] != expected {
            return false;
        }
        expected *= dims[axis];
    }
    true
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
    if is_contiguous(out_shape.dims(), &strides) {
        // The permutation moved only size-1 axes past the others, so the bytes are
        // already in the order the output wants and this is a relabelling. Worth
        // checking for: a SISO Mamba-3 rotates `[b, h, n, 1]` by swapping the last
        // two axes, and paying a full strided copy for that twice per token per
        // layer is most of what `permute` was costing at decode.
        return input.reshape(out_shape);
    }
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
fn flip_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
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
    let line = line_size_for::<R, E>(input.client(), inner);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    unsafe {
        flip_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
            axis_len,
            inner / line,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn reverse_bands_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    bands: &Array<u32>,
    n_bands: usize,
    axis_len: usize,
    inner: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let i = ABSOLUTE_POS % inner;
        let rest = ABSOLUTE_POS / inner;
        let d = rest % axis_len;
        let o = rest / axis_len;
        let mut src_d = d;
        for j in 0..n_bands {
            let start = bands[2 * j] as usize;
            let end = bands[2 * j + 1] as usize;
            if i >= start && i < end {
                src_d = axis_len - 1 - d;
            }
        }
        let src = o * axis_len * inner + src_d * inner + i;
        output[ABSOLUTE_POS] = input[src];
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Reverse `axis` for only the listed channel bands, in one launch.
///
/// `reversed` holds `(start, end)` ranges over the flattened extent inside `axis`;
/// elements whose inner offset falls in a range read from the mirrored position
/// along `axis`, everything else copies straight through. This is what the fused
/// bidirectional mixer uses to put its backward-direction bands into reversed
/// time: done as slices, flips and concatenations it is a launch per band edge,
/// done here it is one. The operation is linear and an involution — applying it
/// twice is the identity — which also makes it its own adjoint.
pub fn reverse_bands<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
    reversed: &[(usize, usize)],
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("reverse_bands");
    if axis >= input.rank() {
        return Err(Error::shape(format!(
            "axis {axis} out of range for {}",
            input.shape
        )));
    }
    let inner = input.shape.inner(axis);
    let axis_len = input.shape.dim(axis);
    for &(start, end) in reversed {
        if start >= end || end > inner {
            return Err(Error::shape(format!(
                "band {start}..{end} does not fit the inner extent {inner} of {}",
                input.shape
            )));
        }
    }
    if reversed.is_empty() || axis_len <= 1 {
        return Ok(input.clone());
    }
    let out = Tensor::empty(input.shape.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }

    // A vector must never straddle a band edge, so the width has to divide every
    // boundary as well as the inner extent.
    let mut aligned = inner;
    for &(start, end) in reversed {
        aligned = gcd(aligned, gcd(start, end));
    }
    let line = line_size_for::<R, E>(input.client(), aligned);

    let mut meta = Vec::with_capacity(reversed.len() * 2);
    for &(start, end) in reversed {
        meta.push((start / line) as u32);
        meta.push((end / line) as u32);
    }
    let meta_handle = crate::backend::meta_handle(input.device(), &meta);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    unsafe {
        reverse_bands_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
            ArrayArg::from_raw_parts(meta_handle, meta.len()),
            reversed.len(),
            axis_len,
            inner / line,
        );
    }
    Ok(out)
}

/// The true length of every row of a padded batch, for the reversals of a
/// bidirectional scan over rows of different lengths.
///
/// It carries what the host already knows — the padded length and whether
/// every row fills it — so that the plain reversal can be chosen without a
/// read. The lengths themselves stay on the device.
pub struct RaggedLengths<R: Runtime> {
    ids: crate::tensor::ops::index::IdTensor<R>,
    max: usize,
    all_full: bool,
}

impl<R: Runtime> Clone for RaggedLengths<R> {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids.clone(),
            max: self.max,
            all_full: self.all_full,
        }
    }
}

impl<R: Runtime> RaggedLengths<R> {
    /// Wrap device lengths `[rows]`, each at most `max`. `all_full` says every
    /// one equals `max`; it must not be set otherwise.
    ///
    /// A length above `max` is read as `max`, so wrong lengths cannot make a
    /// kernel read out of bounds.
    pub fn new(
        ids: crate::tensor::ops::index::IdTensor<R>,
        max: usize,
        all_full: bool,
    ) -> Result<Self> {
        if ids.shape().rank() != 1 || ids.is_empty() || max == 0 {
            return Err(Error::shape(format!(
                "ragged lengths must be a non-empty [rows] with a positive maximum, got {} \
                 and {max}",
                ids.shape()
            )));
        }
        Ok(Self { ids, max, all_full })
    }

    /// Upload host lengths, each at most `max`.
    pub fn from_host(
        lengths: &[u32],
        max: usize,
        device: &crate::backend::Device<R>,
    ) -> Result<Self> {
        if let Some(bad) = lengths.iter().find(|&&len| len as usize > max) {
            return Err(Error::shape(format!(
                "a row of length {bad} does not fit the padded length {max}"
            )));
        }
        let all_full = lengths.iter().all(|&len| len as usize == max);
        Self::new(
            crate::tensor::ops::index::IdTensor::from_slice(lengths, vec![lengths.len()], device)?,
            max,
            all_full,
        )
    }

    /// `[rows]` true lengths.
    pub fn ids(&self) -> &crate::tensor::ops::index::IdTensor<R> {
        &self.ids
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.ids.len()
    }

    /// The padded length.
    pub fn max(&self) -> usize {
        self.max
    }

    /// Whether every row fills the padded length.
    pub fn all_full(&self) -> bool {
        self.all_full
    }
}

#[cube(launch_unchecked)]
fn reverse_bands_ragged_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    bands: &Array<u32>,
    lengths: &Array<u32>,
    n_bands: usize,
    axis_len: usize,
    inner: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let i = ABSOLUTE_POS % inner;
        let rest = ABSOLUTE_POS / inner;
        let d = rest % axis_len;
        let o = rest / axis_len;
        let mut len = lengths[o] as usize;
        if len > axis_len {
            len = axis_len;
        }
        let mut src_d = d;
        // Positions past the row's length stay where they are, in every band.
        if d < len {
            for j in 0..n_bands {
                let start = bands[2 * j] as usize;
                let end = bands[2 * j + 1] as usize;
                if i >= start && i < end {
                    src_d = len - 1 - d;
                }
            }
        }
        output[ABSOLUTE_POS] = input[o * axis_len * inner + src_d * inner + i];
    }
}

/// [`reverse_bands`] over rows of different lengths: in the listed bands,
/// position `d` of a row of length `len` reads position `len − 1 − d` when
/// `d < len` and itself otherwise.
///
/// So each row is reversed **within its own length** and its padding stays
/// last — in both directions of a bidirectional scan, which is what keeps a
/// causal scan from ever carrying a pad into a real position. `axis` must be
/// the axis right after the rows (`axis = 1` of `[rows, seq, ..]`). Like the
/// plain reversal it is linear and an involution, hence its own adjoint. With
/// [`RaggedLengths::all_full`] it *is* the plain reversal.
pub fn reverse_bands_ragged<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    axis: usize,
    reversed: &[(usize, usize)],
    lengths: &RaggedLengths<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("reverse_bands_ragged");
    if axis == 0 || axis >= input.rank() {
        return Err(Error::shape(format!(
            "a ragged reversal needs a row axis before axis {axis} of {}",
            input.shape
        )));
    }
    let inner = input.shape.inner(axis);
    let axis_len = input.shape.dim(axis);
    if input.shape.outer(axis) != lengths.rows() || axis_len != lengths.max() {
        return Err(Error::shape(format!(
            "ragged lengths for {} rows padded to {} do not fit {}",
            lengths.rows(),
            lengths.max(),
            input.shape
        )));
    }
    if lengths.all_full() {
        return reverse_bands(input, axis, reversed);
    }
    for &(start, end) in reversed {
        if start >= end || end > inner {
            return Err(Error::shape(format!(
                "band {start}..{end} does not fit the inner extent {inner} of {}",
                input.shape
            )));
        }
    }
    if reversed.is_empty() || axis_len <= 1 {
        return Ok(input.clone());
    }
    let out = Tensor::empty(input.shape.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let mut aligned = inner;
    for &(start, end) in reversed {
        aligned = gcd(aligned, gcd(start, end));
    }
    let line = line_size_for::<R, E>(input.client(), aligned);
    let mut meta = Vec::with_capacity(reversed.len() * 2);
    for &(start, end) in reversed {
        meta.push((start / line) as u32);
        meta.push((end / line) as u32);
    }
    let meta_handle = crate::backend::meta_handle(input.device(), &meta);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    // SAFETY: one length per row (checked above), each clamped to the axis in
    // the kernel; bands lie inside the inner extent (checked above).
    unsafe {
        reverse_bands_ragged_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
            ArrayArg::from_raw_parts(meta_handle, meta.len()),
            lengths.ids().arg(),
            reversed.len(),
            axis_len,
            inner / line,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn slice_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
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
    crate::backend::trace_shape!("TRACE slice {} axis={axis} len={len}", input.shape);
    let inner = input.shape.inner(axis);
    let out_shape = input.shape.with_dim(axis, len);
    let out = Tensor::empty(out_shape, input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let line = line_size_for::<R, E>(input.client(), inner);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    unsafe {
        slice_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
            src_len,
            len,
            inner / line,
            start,
        );
    }
    Ok(out)
}

/// Whether [`split`] fuses its bands into one launch: `0` off, `1` on, `-1` not
/// yet read from the environment.
static FUSED_SPLIT: core::sync::atomic::AtomicI8 = core::sync::atomic::AtomicI8::new(-1);

/// Whether [`split`] fuses its bands into one launch.
///
/// On by default, and `MAMBA3_FUSED_SPLIT=0` turns it off.
fn fused_split_enabled() -> bool {
    use core::sync::atomic::Ordering;
    match FUSED_SPLIT.load(Ordering::Relaxed) {
        -1 => {
            let on = std::env::var("MAMBA3_FUSED_SPLIT").as_deref() != Ok("0");
            FUSED_SPLIT.store(on as i8, Ordering::Relaxed);
            on
        }
        flag => flag == 1,
    }
}

/// Choose whether [`split`] writes its bands in one launch or one per band.
///
/// Both paths produce identical tensors — `tests/tensor.rs` asserts it element by
/// element — so this changes only how many dispatches the split costs. It exists
/// because the two cannot be told apart by running a process twice: wall-clock
/// noise on this crate's benchmark machines is ±20% run to run, which is several
/// times the effect being measured. `examples/bench_split.rs` alternates them
/// inside one process instead, which is the only way the comparison means
/// anything.
pub fn set_fused_split(on: bool) {
    FUSED_SPLIT.store(on as i8, core::sync::atomic::Ordering::Relaxed);
}

/// How many bands [`split`] will write in a single launch.
///
/// Six covers every split in the crate — the widest is the Mamba-3 projection's
/// five-way cut into `z`, `xBC`, `dt`, `lambda` and `theta` — and a wider one
/// falls back to a band per launch rather than growing this signature.
const MAX_SPLIT_BANDS: usize = 6;

/// Every band of a split, written by one launch.
///
/// [`slice`] copies one band per dispatch, so the five-way cut of a projection is
/// five dispatches that between them read the input exactly once and compute
/// nothing at all. This reads it once and writes all the bands: identical memory
/// traffic, one launch. On a rollout step — where the whole policy is ~75
/// dispatches and eight of them are these splits per layer — that is the single
/// largest block of pure overhead in the step.
///
/// Slots past the end of the split are handed band 0's buffer with a width of `0`.
/// Every band is guarded by `a >= offset` *and* `a - offset < width`, so a
/// zero-width slot matches nothing and is never written through; the alias exists
/// only to fill the signature, and costs no allocation.
///
/// The two conditions are nested rather than combined with `&&` on purpose: the
/// generated code evaluates both sides of a `&&`, and `a - offset` underflows on
/// the bands below the one this element belongs to.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn split_bands_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    out0: &mut Array<Vector<F, N>>,
    out1: &mut Array<Vector<F, N>>,
    out2: &mut Array<Vector<F, N>>,
    out3: &mut Array<Vector<F, N>>,
    out4: &mut Array<Vector<F, N>>,
    out5: &mut Array<Vector<F, N>>,
    axis_len: usize,
    inner: usize,
    w0: usize,
    w1: usize,
    w2: usize,
    w3: usize,
    w4: usize,
    w5: usize,
) {
    if ABSOLUTE_POS < input.len() {
        let i = ABSOLUTE_POS % inner;
        let rest = ABSOLUTE_POS / inner;
        let a = rest % axis_len;
        let o = rest / axis_len;
        let value = input[ABSOLUTE_POS];

        let mut offset = 0usize;
        if a >= offset {
            let d = a - offset;
            if d < w0 {
                out0[o * w0 * inner + d * inner + i] = value;
            }
        }
        offset += w0;
        if a >= offset {
            let d = a - offset;
            if d < w1 {
                out1[o * w1 * inner + d * inner + i] = value;
            }
        }
        offset += w1;
        if a >= offset {
            let d = a - offset;
            if d < w2 {
                out2[o * w2 * inner + d * inner + i] = value;
            }
        }
        offset += w2;
        if a >= offset {
            let d = a - offset;
            if d < w3 {
                out3[o * w3 * inner + d * inner + i] = value;
            }
        }
        offset += w3;
        if a >= offset {
            let d = a - offset;
            if d < w4 {
                out4[o * w4 * inner + d * inner + i] = value;
            }
        }
        offset += w4;
        if a >= offset {
            let d = a - offset;
            if d < w5 {
                out5[o * w5 * inner + d * inner + i] = value;
            }
        }
    }
}

/// The inverse of [`split_bands_kernel`]: up to six parts written into their bands
/// of one output in a single launch. Extents are in vectors, as there.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cat_bands_kernel<F: Float + CubeElement, N: Size>(
    in0: &Array<Vector<F, N>>,
    in1: &Array<Vector<F, N>>,
    in2: &Array<Vector<F, N>>,
    in3: &Array<Vector<F, N>>,
    in4: &Array<Vector<F, N>>,
    in5: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    axis_len: usize,
    w0: usize,
    w1: usize,
    w2: usize,
    w3: usize,
    w4: usize,
    w5: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let a = ABSOLUTE_POS % axis_len;
        let o = ABSOLUTE_POS / axis_len;
        let e1 = w0;
        let e2 = e1 + w1;
        let e3 = e2 + w2;
        let e4 = e3 + w3;
        let e5 = e4 + w4;
        let value = if a < e1 {
            in0[o * w0 + a]
        } else if a < e2 {
            in1[o * w1 + a - e1]
        } else if a < e3 {
            in2[o * w2 + a - e2]
        } else if a < e4 {
            in3[o * w3 + a - e3]
        } else if a < e5 {
            in4[o * w4 + a - e4]
        } else {
            in5[o * w5 + a - e5]
        };
        output[ABSOLUTE_POS] = value;
    }
}

#[cube(launch_unchecked)]
fn write_slice_kernel<F: Float + CubeElement, N: Size>(
    src: &Array<Vector<F, N>>,
    dst: &mut Array<Vector<F, N>>,
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
pub fn cat<R: Runtime, E: FloatElem>(parts: &[Tensor<R, E>], axis: usize) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("cat");
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

    // Up to six non-empty parts: one launch over the output, moving vectors of any
    // width that divides every part's run of `width * inner` elements (so also on
    // the last axis, where `inner` is 1 — the projection split's adjoint).
    if parts.len() <= MAX_SPLIT_BANDS && parts.iter().all(|p| p.len() > 0) && fused_split_enabled() {
        let run = parts
            .iter()
            .fold(0usize, |acc, p| gcd(acc, p.shape.dim(axis) * inner));
        let line = line_size_for::<R, E>(parts[0].client(), run);
        let (count, dim) = launch_1d(parts[0].client(), out.len() / line, line);
        let slot = |i: usize| parts.get(i).unwrap_or(&parts[0]).arg();
        let width = |i: usize| parts.get(i).map_or(0, |p| p.shape.dim(axis) * inner / line);
        unsafe {
            cat_bands_kernel::launch_unchecked::<E, R>(
                parts[0].client(),
                count,
                dim,
                line,
                slot(0),
                slot(1),
                slot(2),
                slot(3),
                slot(4),
                slot(5),
                out.arg(),
                total * inner / line,
                width(0),
                width(1),
                width(2),
                width(3),
                width(4),
                width(5),
            );
        }
        return Ok(out);
    }

    let line = line_size_for::<R, E>(parts[0].client(), inner);
    let mut offset = 0usize;
    for p in parts {
        let n = p.len();
        if n > 0 {
            let (count, dim) = launch_1d(p.client(), n / line, line);
            unsafe {
                write_slice_kernel::launch_unchecked::<E, R>(
                    p.client(),
                    count,
                    dim,
                    line,
                    p.arg(),
                    out.arg(),
                    total,
                    p.shape.dim(axis),
                    inner / line,
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
    if !len.is_multiple_of(n) {
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
    if let Some(out) = split_fused(input, sizes, axis)? {
        return Ok(out);
    }
    let mut out = Vec::with_capacity(sizes.len());
    let mut off = 0;
    for &s in sizes {
        out.push(slice(input, axis, off, s)?);
        off += s;
    }
    Ok(out)
}

/// [`split`] in one launch, or `None` when this split is not one the fused kernel
/// takes.
///
/// It declines three cases, each of which the band-per-launch path already gets
/// right and none of which is worth a branch in the kernel: fewer than two bands
/// (there is no launch to save, and one band is a borrow rather than a copy), more
/// than [`MAX_SPLIT_BANDS`], and any empty band — a zero-length band would bind a
/// zero-sized buffer, which not every backend accepts.
fn split_fused<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    sizes: &[usize],
    axis: usize,
) -> Result<Option<Vec<Tensor<R, E>>>> {
    if sizes.len() < 2 || sizes.len() > MAX_SPLIT_BANDS || sizes.contains(&0) {
        return Ok(None);
    }
    if !fused_split_enabled() {
        return Ok(None);
    }
    let n = input.len();
    if n == 0 {
        return Ok(None);
    }

    crate::backend::trace_shape!("TRACE split {} axis={axis} bands={sizes:?}", input.shape);
    let out: Vec<Tensor<R, E>> = sizes
        .iter()
        .map(|&s| Tensor::empty(input.shape.with_dim(axis, s), input.device()))
        .collect();

    // Every band is `width * inner` contiguous elements of each outer row, so the
    // copy can move vectors of any width that divides all of those runs — also when
    // the split is on the last axis (`inner == 1`), which is the projection's case.
    // With the vector width folded in, the kernel sees `inner = 1` and every extent
    // in vectors; the element mapping is unchanged.
    let inner = input.shape.inner(axis);
    let run = sizes.iter().fold(0usize, |acc, &w| gcd(acc, w * inner));
    let line = line_size_for::<R, E>(input.client(), run);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    let axis_lines = input.shape.dim(axis) * inner / line;

    // Band 0 fills the unused slots. Paired with a width of zero below, which is
    // what keeps the kernel from ever writing through one.
    let slot = |i: usize| out.get(i).unwrap_or(&out[0]).arg();
    let width = |i: usize| sizes.get(i).copied().unwrap_or(0);
    unsafe {
        split_bands_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            slot(0),
            slot(1),
            slot(2),
            slot(3),
            slot(4),
            slot(5),
            axis_lines,
            1,
            width(0) * inner / line,
            width(1) * inner / line,
            width(2) * inner / line,
            width(3) * inner / line,
            width(4) * inner / line,
            width(5) * inner / line,
        );
    }
    Ok(Some(out))
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
