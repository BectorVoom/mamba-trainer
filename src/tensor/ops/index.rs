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
    pub fn to_vec(&self) -> Vec<u32> {
        crate::backend::count_read();
        let bytes = self.device.client().read_one_unchecked(self.handle.clone());
        u32::from_bytes(&bytes)[..self.shape.num_elements()].to_vec()
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

    pub(crate) fn arg(&self) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.handle.clone(), self.len()) }
    }

    pub(crate) fn client(&self) -> &ComputeClient<R> {
        self.device.client()
    }
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
    let host_ids = ids.to_vec();
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
