//! Backend abstraction.
//!
//! The crate is generic over two axes:
//!
//! * `R: Runtime` — a CubeCL runtime (CPU, wgpu, CUDA, HIP). Everything above the
//!   kernel layer only ever touches [`cubecl::prelude::Runtime`], so adding a backend is a
//!   feature flag, not a code change.
//! * `E: FloatElem` — the storage/compute element type. `f32` today, `f16`/`bf16`
//!   are wired for mixed-precision experiments.
//!
//! Keeping both generic is what makes the same modules usable for language models,
//! vision models, quantization-aware training and inference without forking code.

use cubecl::prelude::*;

/// Numeric kind of a tensor's elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16.
    BF16,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    /// Short name used in checkpoints.
    pub const fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        }
    }
}

/// A float element type usable both on the host and inside CubeCL kernels.
///
/// Implemented for `f32`, `half::f16` and `half::bf16`. Host-side reductions
/// (loss accumulation, optimizer statistics) always go through `f32` so that
/// low-precision storage types stay numerically sane.
pub trait FloatElem: Float + CubeElement + Send + Sync + 'static {
    /// The corresponding [`DType`].
    const DTYPE: DType;

    /// Lossy conversion from `f32`.
    fn from_scalar(v: f32) -> Self;

    /// Widening conversion to `f32`.
    fn to_scalar(self) -> f32;

    /// Convert a host slice to `f32`.
    fn slice_to_f32(src: &[Self]) -> Vec<f32> {
        src.iter().map(|v| v.to_scalar()).collect()
    }

    /// Convert an `f32` host slice to this element type.
    fn slice_from_f32(src: &[f32]) -> Vec<Self> {
        src.iter().map(|v| Self::from_scalar(*v)).collect()
    }
}

impl FloatElem for f32 {
    const DTYPE: DType = DType::F32;

    #[inline]
    fn from_scalar(v: f32) -> Self {
        v
    }

    #[inline]
    fn to_scalar(self) -> f32 {
        self
    }
}

impl FloatElem for half::f16 {
    const DTYPE: DType = DType::F16;

    #[inline]
    fn from_scalar(v: f32) -> Self {
        half::f16::from_f32(v)
    }

    #[inline]
    fn to_scalar(self) -> f32 {
        half::f16::to_f32(self)
    }
}

impl FloatElem for half::bf16 {
    const DTYPE: DType = DType::BF16;

    #[inline]
    fn from_scalar(v: f32) -> Self {
        half::bf16::from_f32(v)
    }

    #[inline]
    fn to_scalar(self) -> f32 {
        half::bf16::to_f32(self)
    }
}

/// A device handle plus its compute client.
///
/// Cloning is cheap: the client is internally reference counted.
pub struct Device<R: Runtime> {
    device: R::Device,
    client: ComputeClient<R>,
}

impl<R: Runtime> core::fmt::Debug for Device<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Device({:?})", self.device)
    }
}

impl<R: Runtime> Clone for Device<R> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            client: self.client.clone(),
        }
    }
}

impl<R: Runtime> Default for Device<R> {
    fn default() -> Self {
        Self::new(&R::Device::default())
    }
}

impl<R: Runtime> Device<R> {
    /// Open a device and its compute client.
    pub fn new(device: &R::Device) -> Self {
        Self {
            device: device.clone(),
            client: R::client(device),
        }
    }

    /// The underlying runtime device.
    pub fn inner(&self) -> &R::Device {
        &self.device
    }

    /// The compute client used to allocate buffers and launch kernels.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Backend name, e.g. `"cpu"` or `"wgpu"`.
    pub fn name(&self) -> &'static str {
        R::name(&self.client)
    }

    /// Block until every queued kernel has completed.
    pub fn synchronize(&self) {
        let _ = cubecl::future::block_on(self.client.sync());
    }
}

/// Choose a cube count that covers `num_elems` items with `cube_dim` units each.
pub fn cube_count_for(num_elems: usize, cube_dim: u32) -> CubeCount {
    let groups = num_elems.div_ceil(cube_dim as usize).max(1) as u32;
    // Most backends cap a single grid dimension at 65535; fold the excess into y.
    const MAX: u32 = 32768;
    if groups <= MAX {
        CubeCount::Static(groups, 1, 1)
    } else {
        let y = groups.div_ceil(MAX);
        CubeCount::Static(MAX, y, 1)
    }
}

/// Default number of units per cube for elementwise kernels.
///
/// 64 rather than a GPU-typical 256: on CubeCL's CPU runtime a 4096-element add
/// costs 0.09 ms at 64 units per cube against 0.31 ms at 256, and GPUs are happy
/// with 64 as well.
pub(crate) const ELEMWISE_CUBE_DIM: u32 = 64;
