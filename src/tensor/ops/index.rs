//! Integer id tensors and the gather / scatter-add pair behind embeddings.
//!
//! Embedding gradients are accumulated with a **host-built bucket index** rather
//! than float atomics. That costs one small round trip per step but is exactly
//! reproducible run to run and portable to backends without `atomicAdd(float*)`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::backend::{Device, FloatElem, launch_1d, line_size_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// A dense `u32` tensor: token ids, class targets, argmax results.
pub struct IdTensor<R: Runtime> {
    handle: Handle,
    shape: Shape,
    device: Device<R>,
}

impl<R: Runtime> Clone for IdTensor<R> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
        }
    }
}

impl<R: Runtime> core::fmt::Debug for IdTensor<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "IdTensor({})", self.shape)
    }
}

impl<R: Runtime> IdTensor<R> {
    /// Allocate uninitialised ids.
    pub fn empty(shape: impl Into<Shape>, device: &Device<R>) -> Self {
        let shape = shape.into();
        let handle = device.client().empty(shape.num_elements() * 4);
        Self {
            handle,
            shape,
            device: device.clone(),
        }
    }

    /// Upload ids from the host.
    pub fn from_slice(ids: &[u32], shape: impl Into<Shape>, device: &Device<R>) -> Result<Self> {
        let shape = shape.into();
        if ids.len() != shape.num_elements() {
            return Err(Error::shape(format!(
                "{} ids do not fill shape {shape}",
                ids.len()
            )));
        }
        Ok(Self {
            handle: device.client().create_from_slice(u32::as_bytes(ids)),
            shape,
            device: device.clone(),
        })
    }

    /// Download ids to the host.
    ///
    /// # Panics
    ///
    /// If a kernel launched before the read failed to run; see
    /// [`IdTensor::try_to_vec`].
    pub fn to_vec(&self) -> Vec<u32> {
        self.try_to_vec().unwrap_or_else(|err| panic!("{err}"))
    }

    /// [`IdTensor::to_vec`], returning a failed launch as an error.
    pub fn try_to_vec(&self) -> Result<Vec<u32>> {
        crate::backend::check_launches(&self.device)?;
        crate::backend::count_read();
        let bytes = crate::backend::read_handle(&self.device, &self.handle);
        Ok(u32::from_bytes(&bytes)[..self.shape.num_elements()].to_vec())
    }

    /// Shape of the id tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Number of ids.
    pub fn len(&self) -> usize {
        self.shape.num_elements()
    }

    /// Whether there are no ids.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device the ids live on.
    pub fn device(&self) -> &Device<R> {
        &self.device
    }

    /// Reshape without moving data.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if shape.num_elements() != self.len() {
            return Err(Error::shape("id reshape changes element count".to_string()));
        }
        Ok(Self {
            handle: self.handle.clone(),
            shape,
            device: self.device.clone(),
        })
    }

    /// Kernel argument for this buffer.
    ///
    /// Public for the same reason [`crate::tensor::Tensor::arg`] is: a caller
    /// writing its own rollout kernel has to bind the trajectory buffer's action
    /// column, and that is an `IdTensor`.
    pub fn arg(&self) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.handle.clone(), self.len()) }
    }

    pub(crate) fn client(&self) -> &ComputeClient<R> {
        self.device.client()
    }
}

/// What [`read_together`] hands back: `I` id vectors, then `F` float vectors.
pub type HostReads<const I: usize, const F: usize> = ([Vec<u32>; I], [Vec<f32>; F]);

/// Read id tensors and float tensors back to the host under one synchronisation.
///
/// Returns the ids as `u32` and the floats as `f32`, each array in the order its
/// tensors were given:
///
/// ```ignore
/// let ([actions], [values, log_probs]) = read_together([&actions], [&values, &log_probs])?;
/// ```
///
/// Reading the same tensors one at a time is the same data for several times the
/// wall clock on a GPU: a read's cost is a fixed wait for the device, not the
/// bytes (see `examples/bench_host_read.rs`), and each read waits again. Use this
/// wherever a step hands several outputs to the host at once. Like every read, it
/// reports a kernel that failed to launch as an error instead of returning
/// whatever the buffer held.
pub fn read_together<R: Runtime, E: FloatElem, const I: usize, const F: usize>(
    ids: [&IdTensor<R>; I],
    floats: [&Tensor<R, E>; F],
) -> Result<HostReads<I, F>> {
    let Some(device) = ids
        .first()
        .map(|t| &t.device)
        .or_else(|| floats.first().map(|t| &t.device))
    else {
        // Nothing to read: both arrays are empty.
        return Ok((
            core::array::from_fn(|_| Vec::new()),
            core::array::from_fn(|_| Vec::new()),
        ));
    };
    crate::backend::check_launches(device)?;

    // An empty buffer has nothing to read, and the zero-length slice a read returns
    // is not aligned for its element, which `from_bytes` refuses by panicking.
    let handles = ids
        .iter()
        .filter(|t| !t.is_empty())
        .map(|t| t.handle.clone())
        .chain(
            floats
                .iter()
                .filter(|t| !t.is_empty())
                .map(|t| t.handle.clone()),
        )
        .collect();
    let mut bytes = crate::backend::read_handles(device, handles).into_iter();

    let ids = core::array::from_fn(|i| match ids[i].len() {
        0 => Vec::new(),
        len => u32::from_bytes(&bytes.next().expect("one read per id tensor"))[..len].to_vec(),
    });
    let floats = core::array::from_fn(|i| match floats[i].len() {
        0 => Vec::new(),
        len => E::slice_to_f32(
            &E::from_bytes(&bytes.next().expect("one read per float tensor"))[..len],
        ),
    });
    Ok((ids, floats))
}

#[cube(launch_unchecked)]
fn ids_to_float_kernel<F: Float + CubeElement>(ids: &Array<u32>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::cast_from(ids[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
fn float_to_ids_kernel<F: Float + CubeElement>(input: &Array<F>, out: &mut Array<u32>) {
    if ABSOLUTE_POS < out.len() {
        // Round to nearest rather than truncate: an id that arrived through a float
        // may be `2.9999998` rather than `3`, and truncating would silently pick the
        // wrong class.
        out[ABSOLUTE_POS] = u32::cast_from(F::round(input[ABSOLUTE_POS]));
    }
}

/// Ids as floats, so an integer-valued tensor can be handed to a float API.
pub fn ids_to_float<R: Runtime, E: FloatElem>(ids: &IdTensor<R>) -> Tensor<R, E> {
    let out = Tensor::<R, E>::empty(ids.shape().clone(), ids.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let (count, dim) = launch_1d(ids.client(), n, 1);
    unsafe {
        ids_to_float_kernel::launch_unchecked::<E, R>(
            ids.client(),
            count,
            dim,
            ids.arg(),
            out.arg(),
        );
    }
    out
}

/// The inverse of [`ids_to_float`], rounding to the nearest integer.
pub fn float_to_ids<R: Runtime, E: FloatElem>(input: &Tensor<R, E>) -> IdTensor<R> {
    let out = IdTensor::empty(input.shape().clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let (count, dim) = launch_1d(input.client(), n, 1);
    unsafe {
        float_to_ids_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            input.arg(),
            out.arg(),
        );
    }
    out
}

/// Rows are contiguous runs of `width`, so a unit can copy a whole [`Vector`] of a
/// row at a time; `width` is then counted in vectors.
#[cube(launch_unchecked)]
fn gather_rows_kernel<F: Float + CubeElement, N: Size>(
    table: &Array<Vector<F, N>>,
    ids: &Array<u32>,
    output: &mut Array<Vector<F, N>>,
    width: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let row = ABSOLUTE_POS / width;
        let col = ABSOLUTE_POS % width;
        let src = ids[row] as usize;
        output[ABSOLUTE_POS] = table[src * width + col];
    }
}

/// Look up rows of a `[num_rows, width]` table.
///
/// Output shape is `ids.shape ++ [width]`.
pub fn gather_rows<R: Runtime, E: FloatElem>(
    table: &Tensor<R, E>,
    ids: &IdTensor<R>,
) -> Result<Tensor<R, E>> {
    table.shape.expect_rank(2)?;
    let width = table.shape.dim(1);
    let mut out_dims = ids.shape.dims().to_vec();
    out_dims.push(width);
    let out = Tensor::empty(Shape::new(out_dims), table.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let line = line_size_for::<R, E>(table.client(), width);
    let (count, dim) = launch_1d(table.client(), n / line, line);
    unsafe {
        gather_rows_kernel::launch_unchecked::<E, R>(
            table.client(),
            count,
            dim,
            line,
            table.arg(),
            ids.arg(),
            out.arg(),
            width / line,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn bucket_scatter_add_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    rows: &Array<u32>,
    offsets: &Array<u32>,
    members: &Array<u32>,
    output: &mut Array<Vector<F, N>>,
    width: usize,
    num_buckets: usize,
) {
    if ABSOLUTE_POS < num_buckets * width {
        let bucket = ABSOLUTE_POS / width;
        let col = ABSOLUTE_POS % width;
        let target_row = rows[bucket] as usize;
        let start = offsets[bucket] as usize;
        let end = offsets[bucket + 1] as usize;
        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        for i in start..end {
            acc += grad[members[i] as usize * width + col];
        }
        output[target_row * width + col] = acc;
    }
}

/// Accumulate `grad` rows back into a `[num_rows, width]` table according to `ids`.
///
/// Rows that no id selects stay zero.
pub fn scatter_add_rows<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    ids: &IdTensor<R>,
    num_rows: usize,
) -> Result<Tensor<R, E>> {
    let width = grad.shape.dim_from_end(0);
    let out = Tensor::<R, E>::zeros(Shape::new(vec![num_rows, width]), grad.device());
    if ids.is_empty() {
        return Ok(out);
    }

    // Build buckets on the host: one bucket per distinct row id.
    let host_ids = ids.try_to_vec()?;
    let mut order: Vec<u32> = (0..host_ids.len() as u32).collect();
    order.sort_by_key(|&i| host_ids[i as usize]);

    let mut rows: Vec<u32> = Vec::new();
    let mut offsets: Vec<u32> = vec![0];
    let mut members: Vec<u32> = Vec::with_capacity(order.len());
    let mut current: Option<u32> = None;
    for pos in order {
        let id = host_ids[pos as usize];
        if id as usize >= num_rows {
            return Err(Error::shape(format!(
                "id {id} out of range for a table with {num_rows} rows"
            )));
        }
        if current != Some(id) {
            if current.is_some() {
                offsets.push(members.len() as u32);
            }
            rows.push(id);
            current = Some(id);
        }
        members.push(pos);
    }
    offsets.push(members.len() as u32);

    let num_buckets = rows.len();
    let client = grad.client();
    let rows_h = client.create_from_slice(u32::as_bytes(&rows));
    let offsets_h = client.create_from_slice(u32::as_bytes(&offsets));
    let members_h = client.create_from_slice(u32::as_bytes(&members));

    let line = line_size_for::<R, E>(client, width);
    let lanes = num_buckets * (width / line);
    let (count, dim) = launch_1d(client, lanes, members.len().div_ceil(num_buckets) * line);
    unsafe {
        bucket_scatter_add_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            line,
            grad.arg(),
            ArrayArg::from_raw_parts(rows_h, rows.len()),
            ArrayArg::from_raw_parts(offsets_h, offsets.len()),
            ArrayArg::from_raw_parts(members_h, members.len()),
            out.arg(),
            width / line,
            num_buckets,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn one_hot_kernel<F: Float + CubeElement>(ids: &Array<u32>, output: &mut Array<F>, classes: usize) {
    if ABSOLUTE_POS < output.len() {
        let row = ABSOLUTE_POS / classes;
        let col = ABSOLUTE_POS % classes;
        let mut v = F::new(0.0_f32);
        if ids[row] as usize == col {
            v = F::new(1.0_f32);
        }
        output[ABSOLUTE_POS] = v;
    }
}

/// One-hot encode ids into `ids.shape ++ [classes]`.
pub fn one_hot<R: Runtime, E: FloatElem>(
    ids: &IdTensor<R>,
    classes: usize,
) -> Result<Tensor<R, E>> {
    let mut dims = ids.shape.dims().to_vec();
    dims.push(classes);
    let out = Tensor::<R, E>::empty(Shape::new(dims), ids.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let (count, dim) = launch_1d(ids.client(), n, 1);
    unsafe {
        one_hot_kernel::launch_unchecked::<E, R>(
            ids.client(),
            count,
            dim,
            ids.arg(),
            out.arg(),
            classes,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn take_along_last_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    ids: &Array<u32>,
    output: &mut Array<F>,
    last: usize,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS * last + ids[ABSOLUTE_POS] as usize];
    }
}

/// For each row of `input` (`[..., last]`), pick the element named by `ids`
/// (`[...]`). Used by cross-entropy to read the target logit.
pub fn take_along_last<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    ids: &IdTensor<R>,
) -> Result<Tensor<R, E>> {
    let last = input.shape.dim_from_end(0);
    let out = Tensor::empty(ids.shape.clone(), input.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let (count, dim) = launch_1d(input.client(), n, 1);
    unsafe {
        take_along_last_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            input.arg(),
            ids.arg(),
            out.arg(),
            last,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Splitting and joining id tensors
// ---------------------------------------------------------------------------

/// Copy a contiguous run of ids from one buffer into another at an offset.
///
/// One kernel serves both directions a fan-out needs: taking a worker's slice out
/// of a batched action vector, and putting a worker's labels back into a batched
/// one.
#[cube(launch_unchecked)]
fn copy_ids_kernel(
    src: &Array<u32>,
    dst: &mut Array<u32>,
    src_offset: usize,
    dst_offset: usize,
    len: usize,
) {
    if ABSOLUTE_POS < len {
        dst[dst_offset + ABSOLUTE_POS] = src[src_offset + ABSOLUTE_POS];
    }
}

fn copy_ids_into<R: Runtime>(
    src: &IdTensor<R>,
    dst: &IdTensor<R>,
    src_offset: usize,
    dst_offset: usize,
    len: usize,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    if src_offset + len > src.len() || dst_offset + len > dst.len() {
        return Err(Error::shape(format!(
            "copying {len} ids from offset {src_offset} of {} to offset {dst_offset} of {} \
             runs off the end",
            src.shape(),
            dst.shape()
        )));
    }
    let (count, dim) = launch_1d(src.client(), len, 1);
    unsafe {
        copy_ids_kernel::launch_unchecked::<R>(
            src.client(),
            count,
            dim,
            src.arg(),
            dst.arg(),
            src_offset,
            dst_offset,
            len,
        );
    }
    Ok(())
}

/// `len` ids starting at `start`, as a tensor of their own.
///
/// Ids have no strided view — like every tensor in the crate they are contiguous —
/// so this copies. At the widths it is used for (one action per environment) that
/// is one small kernel, which is cheaper than the host round trip the alternative
/// would need.
pub fn slice_ids<R: Runtime>(input: &IdTensor<R>, start: usize, len: usize) -> Result<IdTensor<R>> {
    if start + len > input.len() {
        return Err(Error::shape(format!(
            "ids {start}..{} are outside {}",
            start + len,
            input.shape()
        )));
    }
    let out = IdTensor::empty(vec![len], input.device());
    copy_ids_into(input, &out, start, 0, len)?;
    Ok(out)
}

/// Join id tensors end to end.
pub fn cat_ids<R: Runtime>(parts: &[IdTensor<R>]) -> Result<IdTensor<R>> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let first = parts
        .first()
        .ok_or_else(|| Error::shape("cannot join an empty list of id tensors".to_string()))?;
    let out = IdTensor::empty(vec![total], first.device());
    let mut offset = 0;
    for part in parts {
        copy_ids_into(part, &out, 0, offset, part.len())?;
        offset += part.len();
    }
    Ok(out)
}
