//! The device tensor: a contiguous, row-major buffer plus a shape.
//!
//! Design notes
//! ------------
//! * Tensors are **always C-contiguous**. Views that would introduce non-unit
//!   strides (`permute`, `slice`, ...) materialise a new buffer. This keeps every
//!   kernel in the crate a flat `Array<E>` kernel, which is what makes the
//!   backend surface small enough to port to a new runtime in an afternoon.
//! * [`Tensor::clone`] is an **alias**, not a copy: the underlying CubeCL handle is
//!   reference counted. No operation ever mutates its inputs, so aliasing is safe.
//! * All shape math happens on the host. Kernels receive plain `usize` scalars.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::shape::Shape;

/// A dense tensor stored on a CubeCL device.
pub struct Tensor<R: Runtime, E: FloatElem = f32> {
    pub(crate) handle: Handle,
    pub(crate) shape: Shape,
    pub(crate) device: Device<R>,
    pub(crate) _elem: core::marker::PhantomData<E>,
}

impl<R: Runtime, E: FloatElem> Clone for Tensor<R, E> {
    /// Cheap alias of the same device buffer.
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            _elem: core::marker::PhantomData,
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Tensor<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Tensor<{}>({}, {})",
            E::DTYPE.name(),
            self.shape,
            self.device.name()
        )
    }
}

impl<R: Runtime, E: FloatElem> Tensor<R, E> {
    /// Wrap an existing handle. The caller guarantees the buffer holds
    /// `shape.num_elements()` contiguous elements of type `E`.
    pub(crate) fn from_handle(handle: Handle, shape: Shape, device: Device<R>) -> Self {
        Self {
            handle,
            shape,
            device,
            _elem: core::marker::PhantomData,
        }
    }

    /// Allocate uninitialised device memory.
    pub fn empty(shape: impl Into<Shape>, device: &Device<R>) -> Self {
        let shape = shape.into();
        let handle = device
            .client()
            .empty(shape.num_elements() * core::mem::size_of::<E>());
        Self::from_handle(handle, shape, device.clone())
    }

    /// Upload host data. `data.len()` must equal `shape.num_elements()`.
    pub fn from_data(data: &[E], shape: impl Into<Shape>, device: &Device<R>) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.num_elements() {
            return Err(Error::shape(format!(
                "data of length {} does not fill shape {shape}",
                data.len()
            )));
        }
        let handle = device.client().create_from_slice(E::as_bytes(data));
        Ok(Self::from_handle(handle, shape, device.clone()))
    }

    /// Upload `f32` host data, converting to `E`.
    pub fn from_f32(data: &[f32], shape: impl Into<Shape>, device: &Device<R>) -> Result<Self> {
        Self::from_data(&E::slice_from_f32(data), shape, device)
    }

    /// Download to the host in the tensor's own element type.
    pub fn to_data(&self) -> Vec<E> {
        let bytes = self.device.client().read_one_unchecked(self.handle.clone());
        E::from_bytes(&bytes)[..self.shape.num_elements()].to_vec()
    }

    /// Download to the host as `f32`.
    pub fn to_f32(&self) -> Vec<f32> {
        E::slice_to_f32(&self.to_data())
    }

    /// Read a scalar tensor (or the first element).
    pub fn scalar(&self) -> f32 {
        self.to_f32()[0]
    }

    /// The tensor's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The tensor's dimensions.
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// The tensor's rank.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.shape.num_elements()
    }

    /// Whether the tensor holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The device the tensor lives on.
    pub fn device(&self) -> &Device<R> {
        &self.device
    }

    /// The compute client for this tensor's device.
    pub fn client(&self) -> &ComputeClient<R> {
        self.device.client()
    }

    /// Kernel argument for this buffer.
    ///
    /// The length is always in scalar elements, even for a kernel that reads the
    /// buffer as `Array<Vector<E, N>>`: CubeCL divides by the vector width itself.
    ///
    /// # Safety
    /// The returned argument borrows the buffer for the duration of the launch.
    pub(crate) fn arg(&self) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.handle.clone(), self.len()) }
    }

    /// Reinterpret the buffer with a new shape of the same element count.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if shape.num_elements() != self.len() {
            return Err(Error::shape(format!(
                "cannot reshape {} ({} elements) into {shape} ({} elements)",
                self.shape,
                self.len(),
                shape.num_elements()
            )));
        }
        Ok(Self::from_handle(
            self.handle.clone(),
            shape,
            self.device.clone(),
        ))
    }

    /// Insert a size-1 axis at `axis`.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        self.reshape(self.shape.unsqueezed(axis))
    }

    /// Remove a size-1 axis at `axis`.
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        if self.shape.dim(axis) != 1 {
            return Err(Error::shape(format!(
                "cannot squeeze axis {axis} of {} (size {})",
                self.shape,
                self.shape.dim(axis)
            )));
        }
        self.reshape(self.shape.without(axis))
    }

    /// Collapse the tensor to rank 1.
    pub fn flatten(&self) -> Self {
        self.reshape(Shape::new(vec![self.len()]))
            .expect("flatten preserves element count")
    }

    /// An independent copy of the buffer.
    pub fn deep_clone(&self) -> Self {
        crate::tensor::ops::elemwise::identity(self)
    }
}

/// Convenience constructors that need no gradient bookkeeping.
impl<R: Runtime, E: FloatElem> Tensor<R, E> {
    /// A tensor filled with `value`.
    pub fn full(shape: impl Into<Shape>, value: f32, device: &Device<R>) -> Self {
        let shape = shape.into();
        let out = Self::empty(shape, device);
        crate::tensor::ops::elemwise::fill_(&out, value);
        out
    }

    /// A tensor of zeros.
    pub fn zeros(shape: impl Into<Shape>, device: &Device<R>) -> Self {
        Self::full(shape, 0.0, device)
    }

    /// A tensor of ones.
    pub fn ones(shape: impl Into<Shape>, device: &Device<R>) -> Self {
        Self::full(shape, 1.0, device)
    }

    /// `[0, 1, 2, ... n-1]` as a rank-1 tensor.
    pub fn arange(n: usize, device: &Device<R>) -> Self {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Self::from_f32(&data, vec![n], device).expect("arange shape is consistent")
    }

    /// A lower-triangular mask of shape `[n, n]`: `1` where `col <= row`.
    pub fn causal_mask(n: usize, device: &Device<R>) -> Self {
        let mut data = vec![0.0f32; n * n];
        for r in 0..n {
            for c in 0..=r {
                data[r * n + c] = 1.0;
            }
        }
        Self::from_f32(&data, vec![n, n], device).expect("mask shape is consistent")
    }

    /// A strictly-lower-triangular mask of shape `[n, n]`: `1` where `col < row`.
    pub fn strict_causal_mask(n: usize, device: &Device<R>) -> Self {
        let mut data = vec![0.0f32; n * n];
        for r in 0..n {
            for c in 0..r {
                data[r * n + c] = 1.0;
            }
        }
        Self::from_f32(&data, vec![n, n], device).expect("mask shape is consistent")
    }

    /// The `[n, n]` identity matrix.
    pub fn eye(n: usize, device: &Device<R>) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_f32(&data, vec![n, n], device).expect("eye shape is consistent")
    }
}
