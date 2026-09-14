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
use cubecl::server::Handle;

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
    /// Identity shared by this handle's clones; see [`Device::id`].
    id: usize,
}

/// Source of [`Device::id`].
static NEXT_DEVICE_ID: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

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
            id: self.id,
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
            id: NEXT_DEVICE_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed),
        }
    }

    /// This handle's identity, which its clones share.
    ///
    /// Used to scope [`meta_handle`]'s cache, so a buffer allocated through one
    /// device is never handed to another. Two `Device::new` calls for the same
    /// physical device get different identities and so cache separately — which
    /// wastes a little memory and is never wrong, the trade this exists to make.
    pub(crate) fn id(&self) -> usize {
        self.id
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
    ///
    /// # Panics
    ///
    /// If the runtime reports a failed launch — see [`check_launches`]. Syncing
    /// consumes the runtime's record of the failure, so discarding it here would
    /// hide it from every later check; [`Device::try_synchronize`] returns it
    /// instead.
    pub fn synchronize(&self) {
        if let Err(err) = self.try_synchronize() {
            panic!("{err}");
        }
    }

    /// [`Device::synchronize`], returning a failed launch as an error.
    pub fn try_synchronize(&self) -> crate::error::Result<()> {
        cubecl::future::block_on(self.client.sync()).map_err(launch_error)
    }
}

/// Whether `device` can store elements of `dtype` in buffers and compute with them.
///
/// Asked of the runtime, not inferred from its name: CubeCL records, per storage
/// type, the usages each backend registered for the adapter it opened. The crate's
/// kernels need three — a buffer of the type, arithmetic on it, and conversion to
/// and from `f32`. On this crate's backends that means: the CPU runtime takes
/// `f16` and `bf16`; wgpu compiling WGSL takes `f16` only when the adapter has
/// `SHADER_F16`, and never `bf16`, a type WGSL does not have.
///
/// `f32` is always supported; it is what every backend computes in.
pub fn supports_dtype<R: Runtime>(device: &Device<R>, dtype: DType) -> bool {
    use cubecl::features::TypeUsage;
    use cubecl::ir::{ElemType, FloatKind, StorageType};

    let kind = match dtype {
        DType::F32 => return true,
        DType::F16 => FloatKind::F16,
        DType::BF16 => FloatKind::BF16,
    };
    let needed = TypeUsage::Buffer | TypeUsage::Arithmetic | TypeUsage::Conversion;
    device
        .client()
        .properties()
        .type_usage(StorageType::Scalar(ElemType::Float(kind)))
        .is_superset(needed)
}

/// [`supports_dtype`] as an error that says what to do instead.
///
/// Refusing up front matters because the alternative is not an error at all: WGSL's
/// compiler *panics* on a `bf16` element, on the device thread, once per launch,
/// and the caller's buffers keep their old contents.
pub fn ensure_dtype<R: Runtime>(device: &Device<R>, dtype: DType) -> crate::error::Result<()> {
    if supports_dtype(device, dtype) {
        return Ok(());
    }
    Err(crate::error::Error::Unsupported(format!(
        "the {} backend cannot store or compute {} elements; use f32{}, or build \
         for a backend that supports them (the cpu runtime does)",
        device.name(),
        dtype.name(),
        if dtype == DType::BF16 && supports_dtype(device, DType::F16) {
            " or f16"
        } else {
            ""
        },
    )))
}

/// Fail if a kernel launched from this thread could not run.
///
/// CubeCL launches are fire-and-forget: `launch_unchecked` returns nothing. When a
/// kernel's shader fails to compile — on wgpu, a WGSL module the validator rejects,
/// such as one spelling an infinite float literal — the runtime does not dispatch
/// it and parks the error on the launching thread's stream. The output buffer
/// keeps whatever it held, typically zeros, and a later read returns those zeros
/// without complaint, because reads do not look at the parked errors. Only a
/// flush that asks for them does, which is what this is.
///
/// Every host read in this crate ([`crate::tensor::Tensor::try_to_data`] and its
/// relatives) and every synchronisation point calls it first, so a broken kernel
/// surfaces as an error at the next place a value is observed instead of as a
/// silently wrong number. It does not read device memory: on wgpu it submits the
/// queued command buffer, the same work the following read would have done.
///
/// Errors are recorded per stream, and a stream is per thread. A kernel launched
/// on another thread is reported by a check on that thread.
pub fn check_launches<R: Runtime>(device: &Device<R>) -> crate::error::Result<()> {
    device.client().flush().map_err(launch_error)
}

/// Read a buffer back to the host, from whichever thread.
///
/// The read runs under the stream that allocated the buffer rather than the
/// calling thread's. CubeCL 0.10's CPU server looks a read's memory up in the
/// *caller's* stream (`cubecl-cpu` `compute/server.rs`, `read`:
/// `self.scheduler.stream(&stream_id)` where the wgpu server uses
/// `desc.handle.stream`), so a buffer allocated on one thread and read on another
/// panicked with "Memory slice N doesn't exist" — which is what happened to an
/// environment on a [`crate::rl::ParallelEnvs`] worker that read its actions, or
/// saved its state. Callers flush their own stream first
/// ([`check_launches`]), so work this thread queued against the buffer has run.
pub(crate) fn read_handle<R: Runtime>(device: &Device<R>, handle: &Handle) -> cubecl::bytes::Bytes {
    let client = device.client();
    handle
        .stream
        .executes(|| client.read_one_unchecked(handle.clone()))
}

/// A runtime failure as a crate error, keeping the part a caller can act on.
///
/// CubeCL's own `Display` for a failed launch nests every error inside an
/// "invalid state" wrapper and appends a backtrace to each; the reasons — which
/// name the kernel and quote the compiler — are what is worth reporting.
fn launch_error(err: cubecl::server::ServerError) -> crate::error::Error {
    use cubecl::server::{LaunchError, ServerError};

    fn reason(err: &ServerError) -> String {
        match err {
            ServerError::ServerUnhealthy { errors, .. } => {
                // A kernel that fails to compile fails at every launch, and each
                // launch parks its own copy of the same error.
                let mut distinct: Vec<(String, usize)> = Vec::new();
                for text in errors.iter().map(reason) {
                    match distinct.iter_mut().find(|(seen, _)| *seen == text) {
                        Some((_, count)) => *count += 1,
                        None => distinct.push((text, 1)),
                    }
                }
                distinct
                    .into_iter()
                    .map(|(text, count)| match count {
                        1 => text,
                        n => format!("{text} (reported by {n} launches)"),
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            }
            ServerError::Launch(LaunchError::CompilationError(
                cubecl::CompilationError::Generic { reason, .. }
                | cubecl::CompilationError::Validation { reason, .. }
                | cubecl::CompilationError::UnsupportedInstruction { reason, .. },
            )) => format!("kernel compilation failed: {reason}"),
            ServerError::Generic { reason, .. } => reason.clone(),
            other => other.to_string(),
        }
    }
    crate::error::Error::backend(reason(&err))
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

/// Per-call-site launch tally, when one has been started.
///
/// A bare launch *count* says a phase is dispatch-bound; it does not say which
/// operation is doing the dispatching, and on this crate's hot paths the answer is
/// rarely the one you would guess. Every launch already funnels through
/// [`launch_1d`] or [`count_launch`], so making both `#[track_caller]` attributes a
/// dispatch to the op that issued it at no cost to the kernels themselves.
///
/// Off by default and gated on a relaxed atomic, so a build that never starts a
/// tally pays one predictable load per launch and never touches the lock.
static TALLY: std::sync::Mutex<Option<std::collections::HashMap<(&'static str, u32), usize>>> =
    std::sync::Mutex::new(None);

/// Whether [`TALLY`] is recording, checked before the lock is taken.
static TALLY_ON: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Begin attributing launches to their call sites, discarding any previous tally.
///
/// Pair with [`launch_tally`] to read the result and [`stop_launch_tally`] to put
/// the hot path back to a single atomic increment.
pub fn start_launch_tally() {
    *TALLY.lock().expect("launch tally is not poisoned") = Some(std::collections::HashMap::new());
    TALLY_ON.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// Stop attributing launches. The tally collected so far stays readable.
pub fn stop_launch_tally() {
    TALLY_ON.store(false, core::sync::atomic::Ordering::Relaxed);
}

/// Launches recorded since [`start_launch_tally`], as `(file:line, count)` sorted
/// by descending count.
pub fn launch_tally() -> Vec<(String, usize)> {
    let guard = TALLY.lock().expect("launch tally is not poisoned");
    let Some(sites) = guard.as_ref() else {
        return Vec::new();
    };
    let mut rows: Vec<(String, usize)> = sites
        .iter()
        .map(|((file, line), count)| (format!("{file}:{line}"), *count))
        .collect();
    // Ties broken by name so a printed tally is stable run to run, which is the
    // whole point of counting launches rather than timing them.
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows
}

/// Forget every recorded site, leaving the tally recording if it already was.
pub fn reset_launch_tally() {
    if let Some(sites) = TALLY.lock().expect("launch tally is not poisoned").as_mut() {
        sites.clear();
    }
}

/// Charge one launch to `site`, if a tally is running.
#[inline]
fn record_site(site: &'static core::panic::Location<'static>) {
    if !TALLY_ON.load(core::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if let Some(sites) = TALLY.lock().expect("launch tally is not poisoned").as_mut() {
        *sites.entry((site.file(), site.line())).or_insert(0) += 1;
    }
}

/// Kernel launches counted since the last [`reset_launch_count`].
static LAUNCHES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Device-to-host reads counted since the last [`reset_read_count`].
static READS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Device-to-host reads issued so far.
///
/// Every one of these blocks until the whole queue drains, not just the buffer
/// being read — that is what makes a mid-step read a stall rather than a cost.
/// A step that only reads what [`crate::train::Trainer::step`] intends to (the loss and the
/// gradient-norm scale) should hold this at exactly the number of such reads it
/// issued; any more means something on the hot path synchronised that did not
/// need to.
pub fn read_count() -> usize {
    READS.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set [`read_count`] back to zero.
pub fn reset_read_count() {
    READS.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Record a device-to-host read.
pub(crate) fn count_read() {
    READS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

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

thread_local! {
    /// Cached `[shape, strides]` buffers, keyed by device and then by contents.
    ///
    /// Thread-local on purpose. A [`Handle`] records the stream it was created on,
    /// so keeping the cache per-thread means a buffer is only ever reused by the
    /// thread that uploaded it, and the cache needs no lock on a path this hot.
    static META_CACHE: core::cell::RefCell<
        std::collections::HashMap<usize, std::collections::HashMap<Vec<u32>, Handle>>,
    > = core::cell::RefCell::new(std::collections::HashMap::new());
}

/// Distinct metadata buffers held per device before the cache is dropped.
///
/// A training run uses a handful of shapes, so this is never reached in practice;
/// it is here so that a program which genuinely does use unbounded shapes leaks
/// nothing. Clearing wholesale rather than evicting one entry keeps the bookkeeping
/// to a comparison, which matters because that comparison is on the hot path.
const META_CACHE_LIMIT: usize = 512;

/// Upload a small shape/stride buffer, or hand back the one already on the device.
///
/// Every broadcasting binary op, strided copy and expand uploads one of these just
/// before it launches — roughly a hundred bytes describing the shapes involved.
/// Measured on wgpu, that upload is **22 us, 37% of the whole operation**
/// (`examples/bench_meta_upload.rs`), and a rollout step performs eleven of them
/// whose contents are *identical on every step*: the shapes of a policy step do not
/// change from one observation to the next.
///
/// So they are uploaded once and kept. This is the manual's first host-side lever —
/// hoist invariant uploads out of the loop — applied where the loop is the whole
/// training run and the invariance is discovered from the contents rather than
/// declared by the caller.
///
/// Safe to share: the kernels that read these buffers only ever read them, so two
/// launches holding the same handle cannot disagree about what is in it.
pub(crate) fn meta_handle<R: Runtime>(device: &Device<R>, meta: &[u32]) -> Handle {
    if !meta_cache_enabled() {
        return device.client().create_from_slice(u32::as_bytes(meta));
    }
    META_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let per_device = cache.entry(device.id()).or_default();
        if let Some(handle) = per_device.get(meta) {
            return handle.clone();
        }
        if per_device.len() >= META_CACHE_LIMIT {
            per_device.clear();
        }
        let handle = device.client().create_from_slice(u32::as_bytes(meta));
        per_device.insert(meta.to_vec(), handle.clone());
        handle
    })
}

/// Whether [`meta_handle`] caches: `0` off, `1` on, `-1` not yet read.
static META_CACHE_ON: core::sync::atomic::AtomicI8 = core::sync::atomic::AtomicI8::new(-1);

/// Whether metadata buffers are cached. On by default; `MAMBA3_META_CACHE=0` puts
/// the upload-every-time behaviour back.
fn meta_cache_enabled() -> bool {
    use core::sync::atomic::Ordering;
    match META_CACHE_ON.load(Ordering::Relaxed) {
        -1 => {
            let on = std::env::var("MAMBA3_META_CACHE").as_deref() != Ok("0");
            META_CACHE_ON.store(on as i8, Ordering::Relaxed);
            on
        }
        flag => flag == 1,
    }
}

/// Choose whether metadata buffers are cached between launches.
///
/// Off means every broadcasting op, strided copy and expand re-uploads its
/// `[shape, strides]` buffer, which is what the crate did before the cache existed.
/// The results are identical either way — the buffer holds the same bytes — so this
/// changes only cost, and exists so the two can be compared inside one process
/// rather than across runs that differ by more than the change under test.
pub fn set_meta_cache(on: bool) {
    META_CACHE_ON.store(on as i8, core::sync::atomic::Ordering::Relaxed);
    if !on {
        clear_meta_cache();
    }
}

/// Drop every cached metadata buffer.
///
/// Only the memory-footprint tests need this: they measure reserved bytes before
/// and after a loop, and a cache that is still filling during the first iteration
/// would look like a leak.
pub fn clear_meta_cache() {
    META_CACHE.with(|cache| cache.borrow_mut().clear());
}

/// Bytes the runtime has reserved on `device`, including pooled memory it is
/// holding for reuse.
///
/// This is the number that must stop growing for a loop to be safe to run
/// indefinitely. It counts reserved rather than in-use bytes on purpose: a step
/// that allocates and frees the same intermediates every time returns them to the
/// pool, so `bytes_in_use` falls back to the persistent state between steps while
/// `bytes_reserved` records the high-water mark the pool actually holds. A flat
/// high-water mark is the real statement that nothing leaks.
///
/// Returns `None` on a runtime that does not report memory.
pub fn reserved_bytes<R: Runtime>(device: &Device<R>) -> Option<u64> {
    device
        .client()
        .memory_usage()
        .ok()
        .map(|u| u.bytes_reserved)
}

/// Record a launch whose geometry did not come from [`launch_1d`].
///
/// Kernels with a fixed cube shape — the block-tiled matmul, whose geometry follows
/// its block size rather than an element count — call this so the counter still sees
/// every dispatch.
#[track_caller]
pub(crate) fn count_launch() {
    LAUNCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    record_site(core::panic::Location::caller());
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
#[track_caller]
pub(crate) fn launch_1d<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    work_per_lane: usize,
) -> (CubeCount, CubeDim) {
    LAUNCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    record_site(core::panic::Location::caller());

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
