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
    /// Fresh-buffer bytes this value accounts for in
    /// [`live_bytes`](crate::backend::live_bytes): the buffer's byte size
    /// when this value was created alongside a fresh device buffer
    /// ([`Tensor::empty`], `from_data`, `from_vec`), shared by its clones
    /// (each live value counts the bytes, so the high-water mark stays a
    /// conservative upper bound under sharing), zero for views cut by
    /// reshape-like constructors.
    pub(crate) owned_bytes: usize,
}

impl<R: Runtime, E: FloatElem> Clone for Tensor<R, E> {
    /// Cheap alias of the same device buffer.
    fn clone(&self) -> Self {
        if self.owned_bytes != 0 {
            crate::backend::note_live_clone(self.owned_bytes);
        }
        Self {
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            _elem: core::marker::PhantomData,
            // Shares the buffer: shares the count too (counted anew, so
            // every live value's drop balances).
            owned_bytes: self.owned_bytes,
        }
    }
}

impl<R: Runtime, E: FloatElem> Drop for Tensor<R, E> {
    fn drop(&mut self) {
        if self.owned_bytes != 0 {
            crate::backend::note_live_free(self.owned_bytes);
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
            // Shares a buffer made elsewhere: counts nothing.
            owned_bytes: 0,
        }
    }

    /// Allocate uninitialised device memory.
    ///
    /// When a [`ScratchArena`] is active on this thread for this device
    /// (see [`with_scratch`]), a retained buffer of exactly the same byte
    /// size is reused instead of allocating: no [`allocation_calls`] charge
    /// and no [`note_alloc`] peak update, since it is the same memory. The
    /// contents of a reused buffer are stale, which is already this
    /// function's contract ("uninitialised"): every caller must write the
    /// buffer before reading it (kernels write their whole output;
    /// [`Tensor::zeros`]/[`Tensor::full`] overwrite with a fill).
    ///
    /// [`ScratchArena`]: crate::backend::ScratchArena
    /// [`with_scratch`]: crate::backend::with_scratch
    /// [`allocation_calls`]: crate::backend::allocation_calls
    /// [`note_alloc`]: crate::backend::note_alloc
    ///
    /// # Panics
    ///
    /// If `device` cannot hold elements of type `E` — `bf16` on WGSL, for one;
    /// see [`crate::backend::supports_dtype`]. Every tensor, including every op's
    /// output, starts here, so this is the one place a narrow element type the
    /// device lacks can be stopped before a kernel for it is compiled.
    /// [`Tensor::from_data`] returns the same refusal as an error.
    pub fn empty(shape: impl Into<Shape>, device: &Device<R>) -> Self {
        if E::DTYPE != crate::backend::DType::F32
            && let Err(err) = crate::backend::ensure_dtype(device, E::DTYPE)
        {
            panic!("{err}");
        }
        let shape = shape.into();
        let bytes = shape.num_elements() * core::mem::size_of::<E>();
        // One cheap thread-local check when no scope is active (see
        // `scratch_scope_active`): outside a scope both the reuse lookup
        // and the retention below are skipped without touching the arena.
        let scoped = bytes > 0 && crate::backend::scratch_scope_active();
        if scoped {
            if let Some(handle) = crate::backend::scratch_reuse(device.id(), bytes) {
                return Self::from_handle(handle, shape, device.clone());
            }
        }
        crate::backend::note_alloc(bytes);
        crate::backend::count_allocation();
        let handle = device.client().empty(bytes);
        if scoped {
            crate::backend::scratch_retain(device.id(), bytes, &handle);
        }
        crate::backend::note_live_alloc(bytes);
        let mut out = Self::from_handle(handle, shape, device.clone());
        out.owned_bytes = bytes;
        out
    }

    /// Upload host data. `data.len()` must equal `shape.num_elements()`.
    pub fn from_data(data: &[E], shape: impl Into<Shape>, device: &Device<R>) -> Result<Self> {
        crate::backend::ensure_dtype(device, E::DTYPE)?;
        let shape = shape.into();
        if data.len() != shape.num_elements() {
            return Err(Error::shape(format!(
                "data of length {} does not fill shape {shape}",
                data.len()
            )));
        }
        crate::backend::count_upload(core::mem::size_of_val(data));
        crate::backend::note_alloc(core::mem::size_of_val(data));
        let handle = device.client().create_from_slice(E::as_bytes(data));
        crate::backend::note_live_alloc(core::mem::size_of_val(data));
        let mut out = Self::from_handle(handle, shape, device.clone());
        out.owned_bytes = core::mem::size_of_val(data);
        Ok(out)
    }

    /// Upload host data by value. `data.len()` must equal `shape.num_elements()`.
    ///
    /// [`Tensor::from_data`] borrows, so the runtime copies the slice into a
    /// buffer of its own before staging it; this hands the vector over as it
    /// is, which for a table the size of a dataset is one host copy saved and
    /// the vector's memory released as soon as the device has it.
    pub fn from_vec(data: Vec<E>, shape: impl Into<Shape>, device: &Device<R>) -> Result<Self> {
        crate::backend::ensure_dtype(device, E::DTYPE)?;
        let shape = shape.into();
        if data.len() != shape.num_elements() {
            return Err(Error::shape(format!(
                "data of length {} does not fill shape {shape}",
                data.len()
            )));
        }
        crate::backend::count_upload(core::mem::size_of_val(data.as_slice()));
        crate::backend::note_alloc(core::mem::size_of_val(data.as_slice()));
        let bytes = core::mem::size_of_val(data.as_slice());
        let handle = device
            .client()
            .create(cubecl::bytes::Bytes::from_elems(data));
        crate::backend::note_live_alloc(bytes);
        let mut out = Self::from_handle(handle, shape, device.clone());
        out.owned_bytes = bytes;
        Ok(out)
    }

    /// Upload `f32` host data, converting to `E`.
    pub fn from_f32(data: &[f32], shape: impl Into<Shape>, device: &Device<R>) -> Result<Self> {
        Self::from_data(&E::slice_from_f32(data), shape, device)
    }

    /// Download to the host in the tensor's own element type.
    ///
    /// # Panics
    ///
    /// If a kernel launched before the read failed to run, since the buffer would
    /// then hold whatever it held before rather than a result. Use
    /// [`Tensor::try_to_data`] to receive that as an error.
    pub fn to_data(&self) -> Vec<E> {
        self.try_to_data().unwrap_or_else(|err| panic!("{err}"))
    }

    /// Download to the host as `f32`. Panics as [`Tensor::to_data`] does.
    pub fn to_f32(&self) -> Vec<f32> {
        E::slice_to_f32(&self.to_data())
    }

    /// [`Tensor::to_data`], returning a failed launch as an error.
    pub fn try_to_data(&self) -> Result<Vec<E>> {
        crate::backend::check_launches(&self.device)?;
        // An empty buffer has nothing to read, and the zero-length slice a read
        // returns is not aligned for `E`, which `from_bytes` refuses by panicking.
        if self.shape.num_elements() == 0 {
            return Ok(Vec::new());
        }
        crate::backend::count_read();
        let bytes = crate::backend::read_handle(&self.device, &self.handle);
        Ok(E::from_bytes(&bytes)[..self.shape.num_elements()].to_vec())
    }

    /// [`Tensor::to_f32`], returning a failed launch as an error.
    pub fn try_to_f32(&self) -> Result<Vec<f32>> {
        Ok(E::slice_to_f32(&self.try_to_data()?))
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
    /// Public so that a caller can bind a tensor into a kernel of its own — which
    /// is what [`crate::rl::Collector::collect_with`] exists for, and it would be
    /// useless if the tensors it hands over could not be bound. Everything the
    /// crate launches for itself goes through this too.
    ///
    /// # Safety
    /// The returned argument borrows the buffer for the duration of the launch, and
    /// the launch is `unsafe` for the usual CubeCL reason: nothing checks that the
    /// kernel binding it reads the buffer as `E`, or that it stays in bounds.
    pub fn arg(&self) -> ArrayArg<R> {
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
    ///
    /// Uploaded once per device, element type and size and shared after that:
    /// the chunked scan asks for the same mask on every call, and a mask is
    /// never written to.
    pub fn strict_causal_mask(n: usize, device: &Device<R>) -> Self {
        if E::DTYPE != crate::backend::DType::F32
            && let Err(err) = crate::backend::ensure_dtype(device, E::DTYPE)
        {
            panic!("{err}");
        }
        let handle = crate::backend::constant_handle::<R, E>(
            device,
            crate::backend::ConstantKind::StrictCausalMask,
            n,
            || {
                let mut data = vec![0.0f32; n * n];
                for r in 0..n {
                    for c in 0..r {
                        data[r * n + c] = 1.0;
                    }
                }
                data
            },
        );
        Self::from_handle(handle, Shape::new(vec![n, n]), device.clone())
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
