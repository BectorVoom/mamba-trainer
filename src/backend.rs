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
///
/// Prefer [`launch_1d`], which also picks the cube *dimension* from the device's
/// own properties. This helper stays for callers that already know the dimension.
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

/// Ceiling on the units per cube [`launch_1d`] will ask a CPU-like runtime for.
///
/// The width it picks is already bounded by the core count; this only guards
/// against a runtime that reports an implausible one.
pub(crate) const ELEMWISE_CUBE_DIM: u32 = 64;

/// Element operations one unit should be worth before another unit is asked for,
/// on runtimes where a "unit" is an operating-system thread.
///
/// CubeCL's CPU runtime dispatches one task per unit in the cube to a pool of
/// worker threads and blocks until all of them report back. Measured on that
/// runtime, an empty launch costs ~13 us at one unit per cube and ~85 us at 64,
/// i.e. roughly a microsecond of pure dispatch per extra unit. A microsecond buys
/// a few tens of thousands of element operations, so that is the threshold below
/// which a second thread is a loss.
const WORK_PER_CPU_UNIT: usize = 32 * 1024;

/// Whether `MAMBA3_TRACE` was set, read once.
///
/// The shape-level trace it gates is what per-kernel timing cannot give you: the
/// profiler says `StridedCopyKernel` cost 6% of a step, and this says which of the
/// permutations that was and how many megabytes it moved. Reading the environment on
/// every launch would itself be measurable at a few thousand launches a step, hence
/// the cache.
pub(crate) fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAMBA3_TRACE").is_some())
}

/// Print one line per launch of a shape-sensitive op when `MAMBA3_TRACE` is set.
///
/// Aggregating the output by line tells you where a training step's memory traffic
/// actually goes, which is the question the kernel-level profile leaves open.
macro_rules! trace_shape {
    ($($arg:tt)*) => {
        if $crate::backend::trace_enabled() {
            eprintln!($($arg)*);
        }
    };
}
pub(crate) use trace_shape;

/// Kernel launches counted since the last [`reset_launch_count`].
static LAUNCHES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Kernel launches issued so far.
///
/// Worth watching. Every backend here charges a fixed price per launch — about
/// 13 us on the CPU runtime, about 9 us on wgpu — and for a small model that price,
/// not the arithmetic, is what the wall clock measures. A change that halves this
/// number roughly halves single-token decoding.
pub fn launch_count() -> usize {
    LAUNCHES.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set [`launch_count`] back to zero.
pub fn reset_launch_count() {
    LAUNCHES.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Record a launch whose geometry did not come from [`launch_1d`].
///
/// Kernels with a fixed cube shape — the block-tiled matmul, whose geometry follows
/// its block size rather than an element count — call this so the counter still sees
/// every dispatch.
pub(crate) fn count_launch() {
    LAUNCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// Launch geometry for a kernel that assigns one unit to each of `lanes` items and
/// does roughly `work_per_lane` element operations per lane.
///
/// The cube dimension is derived from the device rather than fixed, because the two
/// runtime families want opposite things:
///
/// * GPU-like runtimes (`plane_size_max > 1`) want a cube that is a whole number of
///   planes wide; [`CubeDim::new`] sizes that from the hardware.
/// * CubeCL's CPU runtime has no hardware planes — every unit is a thread, and the
///   cube *count* is a serial loop inside each thread. There, extra units are pure
///   overhead until the kernel has enough work to amortise them, so the width grows
///   with the total work and stops at the core count.
pub(crate) fn launch_1d<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    work_per_lane: usize,
) -> (CubeCount, CubeDim) {
    LAUNCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

    let hardware = &client.properties().hardware;
    let cube_dim = if hardware.plane_size_max > 1 {
        CubeDim::new(client, lanes)
    } else {
        let cores = hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
        let total = lanes.saturating_mul(work_per_lane.max(1));
        let units = (total / WORK_PER_CPU_UNIT).clamp(1, cores.min(lanes.max(1)));
        CubeDim::new_1d((units as u32).min(ELEMWISE_CUBE_DIM))
    };
    (
        cubecl::calculate_cube_count_elemwise(client, lanes, cube_dim),
        cube_dim,
    )
}

/// The widest vector width the device likes for `E` that divides `num_elems`.
///
/// Kernels over flat, contiguous buffers read and write [`Vector`]s of this many
/// elements, which is what lets one unit issue a full SIMD load instead of a scalar
/// one. A width of `1` means "no vectorisation" and the scalar kernel is used. The
/// count must divide exactly: a partial trailing vector would read past the buffer.
pub(crate) fn line_size_for<R: Runtime, E: FloatElem>(
    client: &ComputeClient<R>,
    num_elems: usize,
) -> usize {
    if num_elems == 0 {
        return 1;
    }
    client
        .io_optimized_vector_sizes(core::mem::size_of::<E>())
        .find(|width| num_elems.is_multiple_of(*width))
        .unwrap_or(1)
}
