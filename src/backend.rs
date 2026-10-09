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
    /// Held by this handle and its clones (every tensor keeps one): when the
    /// last is dropped, the per-device caches know the identity is gone.
    alive: std::sync::Arc<()>,
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
            alive: self.alive.clone(),
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
            alive: std::sync::Arc::new(()),
        }
    }

    /// A handle that tells whether this identity still has an owner.
    fn liveness(&self) -> std::sync::Weak<()> {
        std::sync::Arc::downgrade(&self.alive)
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
    ///
    /// This is the single place that performs *and* records a crate-level
    /// synchronisation: the [`synchronize_count`] increment lives here, on the
    /// actual drain-the-device operation, rather than as independent
    /// bookkeeping in a wrapper. Deleting the sync from a caller therefore
    /// removes its count with it, which is what the span-counter test pins
    /// (finding A3). Every attempt is counted, including one that surfaces a
    /// parked launch failure as `Err`.
    pub fn try_synchronize(&self) -> crate::error::Result<()> {
        SYNCHRONIZE_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
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

/// Host wall-clock label for [`sync_span`] durations (architecture §6.5).
///
/// A [`sync_span`] duration is measured with a host `Instant` around the body
/// plus the device synchronisation — never with device timestamps, even on a
/// runtime whose profiling capability is `DeviceTimestamps`. The runtime's own
/// capability (probed from what `client.profile` returns) belongs in a
/// separate field; see [`crate::models::ms2::workspace::TimingMethod`].
pub const SYNC_WALL_TIMER: &str = "SynchronizedHostWallClock";

/// Run `body` once, synchronise the device, and return its value with the
/// synchronised wall time (P2.6 host path, architecture §6.5).
///
/// CubeCL 0.10's only client-side profiler, `ComputeClient::profile`
/// (`cubecl-runtime-0.10.0/src/client.rs:886`), does not run its closure on
/// the calling thread: the client sends it through `device.exclusive`
/// (`client.rs:304`), which on `std` non-wasm builds is
/// `ChannelDeviceHandle::exclusive`
/// (`cubecl-common-0.10.0/src/device/handle/mod.rs:18` selects the channel
/// handle when `multi_threading` is set, which `build.rs` sets for every
/// `std` non-wasm build; `channel.rs:104` defines `exclusive` via
/// `run_scoped`, `channel.rs:125`, which enqueues the closure on the device
/// runner thread and blocks the caller until it returns). A closure borrowing
/// this crate's `Rc`-held model parameters is `!Send` and cannot be passed to
/// `client.profile` directly, and wrapping it in an `unsafe impl Send` would
/// be unsound: the body would run on another thread while `Rc`'s non-atomic
/// refcount assumes single-threaded access. That is a bound on borrowing
/// caller state across the runner-thread hop — it does not make device
/// timestamps unreachable from safe code: [`profile_session`] builds the
/// `Rc`-holding state on the runner thread, keeps it in runner-thread-local
/// storage, and profiles stages through `Send` callbacks that reach that
/// storage without capturing `Rc` (no `unsafe`). This helper records
/// synchronised host wall clock instead — the same clock `client.profile`
/// reports on runtimes whose `TimingMethod` is `System` (e.g. the CPU
/// runtime, confirmed by `Ms2Capabilities::probe`). It runs the body exactly
/// once, adds no kernel launch, and performs no device read beyond the
/// synchronisation, so a span around a call observes exactly that call's
/// counters. Nested spans are supported (each span synchronises in turn).
pub fn sync_span<R: Runtime, O>(device: &Device<R>, body: impl FnOnce() -> O) -> (O, std::time::Duration) {
    let started = std::time::Instant::now();
    let out = body();
    counted_synchronize(device);
    (out, started.elapsed())
}

/// Process-wide count of actual device synchronisations since the last
/// [`reset_synchronize_count`].
///
/// Incremented by [`Device::try_synchronize`] itself — the operation that
/// drains the device — never by a wrapper's independent bookkeeping. The
/// counter therefore instruments the actual synchronisation: a span that
/// stops synchronising stops counting, which a value-based completion test
/// cannot observe on the CPU runtime (every read blocks there, so a later
/// read would mask the missing boundary sync).
static SYNCHRONIZE_COUNT: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// Synchronisations through [`counted_synchronize`] so far.
pub fn synchronize_count() -> usize {
    SYNCHRONIZE_COUNT.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set [`synchronize_count`] back to zero.
pub fn reset_synchronize_count() {
    SYNCHRONIZE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// [`Device::synchronize`] for [`sync_span`]: the single wrapper through
/// which wall-clock spans synchronise.
///
/// The counting lives in [`Device::try_synchronize`], not here — there is no
/// separate increment to keep while dropping the call (the finding-A3
/// escaping mutant). A span whose synchronisation is removed advances
/// [`synchronize_count`] by zero, which the span-counter test pins.
fn counted_synchronize<R: Runtime>(device: &Device<R>) {
    device.synchronize();
}

/// Source of [`Profiler`]/slot session identities (finding 3).
static NEXT_SESSION_ID: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(1);

thread_local! {
    /// Runner-thread session state for [`profile_session`]: `Some` while a
    /// session runs on this thread, `None` otherwise.
    ///
    /// The stored value is created, used and dropped on the runner thread
    /// only; nothing `!Send` ever crosses a thread. Span and `with_state`
    /// callbacks reach the state through this slot: each takes it out for
    /// the duration of its body and puts it back afterwards, so no borrow is
    /// ever held while user code runs and sequential uses never alias. A
    /// nested session saves and restores any outer value. Callbacks must not
    /// overlap: a span inside a `with_state` body (or vice versa) finds the
    /// slot taken and fails with a clear error — use sequential calls,
    /// which is what the profile driver needs.
    ///
    /// Each entry carries the session id of the [`Profiler`] handle that
    /// installed it; an access through a handle whose id differs fails
    /// without consuming anything (see [`Profiler::with_state`]).
    static PROFILE_SLOT: std::cell::RefCell<Option<(u64, Box<dyn std::any::Any>)>> =
        const { std::cell::RefCell::new(None) };
}

/// Runner-local device-timing profiler handle of [`profile_session`].
///
/// Created on the runner thread and used there only. [`Profiler::span`] times
/// one stage with `client.profile`: the span body is a `fn` pointer over the
/// session state, so the profile closure captures nothing `!Send` and reaches
/// the state through runner-thread-local storage.
///
/// The handle carries the session id installed with the state (see
/// [`profile_session`]): an access while another session's entry occupies the
/// slot fails without consuming anything, so a stale outer handle can neither
/// silently profile the inner state (same type) nor destroy it through a
/// failing downcast (different type).
pub struct Profiler<R: Runtime> {
    client: ComputeClient<R>,
    session: u64,
}

impl<R: Runtime> Clone for Profiler<R> {
    /// Clone the handle: the clone names the same session, so it is stale as
    /// soon as a nested session installs its own entry (see the session
    /// identity test).
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            session: self.session,
        }
    }
}

/// One [`Profiler::span`] duration: milliseconds with the timing method of
/// the returned `ProfileDuration` (`DeviceTimestamps` for hardware
/// timestamps, `SystemTime` for host wall time).
///
/// Finding-A1 limitation on device-timestamp runtimes, verified against the
/// pinned `cubecl-wgpu-0.10.0` source: an ordinary `client.profile` span
/// reports the first timestamped compute pass's begin-to-end duration, not
/// the whole closure's elapsed device time, once the closure spans several
/// passes. `start_profile` flushes queued work and opens the token
/// (`compute/stream.rs:239,243`); each new pass takes its timestamp writes
/// from `register_profile_device` (`compute/stream.rs:548`), which drains
/// only newly initialised tokens (`compute/timings.rs:321`); the stream ends
/// the pass and submits once `tasks_count >= tasks_max`
/// (`compute/stream.rs:453,457`, default 32 in `src/runtime.rs:188`); and
/// the token's end resolves against `current`, i.e. its initial query set
/// (`compute/timings.rs:193`). A span is therefore whole-stage device time
/// only when it provably fits one pass; the profile driver marks every other
/// device-timestamp span `unavailable` instead of reporting the first pass.
#[derive(Debug, Clone, Copy)]
pub struct SpanTiming {
    /// Span duration in milliseconds.
    pub ms: f64,
    /// Which clock timed the span (never [`SYNC_WALL_TIMER`): that label is
    /// for host-`Instant` spans; this timing comes from `client.profile`).
    pub timer: &'static str,
}

impl<R: Runtime> Profiler<R> {
    /// Run `f` on the session state without timing it.
    ///
    /// For setup, warmup and teardown inside a session: the closure runs on
    /// the runner thread with the state taken out of thread-local storage,
    /// so unlike [`Profiler::span`] it may borrow anything (no `Send`
    /// bound — nothing here crosses a thread). Must not overlap a span body
    /// on the same session (see [`profile_session`]).
    ///
    /// Failures are values, never panics and never a lost state: no state on
    /// this thread, a state installed by another session's handle, a state
    /// of another type, or a panic inside `f` (caught here) all return `Err`
    /// and leave whatever occupied the slot in place. A restored state means
    /// the next span works and sees the state.
    pub fn with_state<S: 'static, T>(&self, f: impl FnOnce(&mut S) -> T) -> crate::error::Result<T> {
        let staged: Box<S> = PROFILE_SLOT.with(|slot| {
            let slot_ref = slot.borrow();
            let Some((id, _)) = slot_ref.as_ref() else {
                return Err(crate::error::Error::backend(
                    "Profiler::with_state: no profile_session state on this thread".to_string(),
                ));
            };
            if *id != self.session {
                return Err(crate::error::Error::backend(format!(
                    "Profiler::with_state: session mismatch (handle {}, slot holds {})",
                    self.session, id
                )));
            }
            if !slot_ref.as_ref().expect("slot holds a state").1.is::<S>() {
                return Err(crate::error::Error::backend(
                    "Profiler::with_state: session state type mismatch".to_string(),
                ));
            }
            drop(slot_ref);
            slot.borrow_mut()
                .take()
                .expect("the state was just validated")
                .1
                .downcast::<S>()
                .map_err(|_| {
                    crate::error::Error::backend(
                        "Profiler::with_state: session state type mismatch".to_string(),
                    )
                })
        })?;
        // An unwind-safe guard: the state goes back into the slot whether `f`
        // returns or panics (the slot is empty here — anything else would have
        // failed above — so restoring cannot clobber a nested session).
        struct Restore<S: 'static> {
            session: u64,
            state: Option<Box<S>>,
        }
        impl<S: 'static> Drop for Restore<S> {
            fn drop(&mut self) {
                if let Some(state) = self.state.take() {
                    PROFILE_SLOT.with(|slot| {
                        // The slot is empty on every path that reaches this
                        // guard with a state (anything else failed before the
                        // take, and a nested session restores what it saved);
                        // holding the state here keeps it alive on the runner
                        // thread rather than dropping live session state.
                        let mut slot = slot.borrow_mut();
                        if slot.is_none() {
                            *slot = Some((self.session, state));
                        }
                    });
                }
            }
        }
        let mut restore = Restore {
            session: self.session,
            state: Some(staged),
        };
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            f(restore.state.as_deref_mut().expect("state is present"))
        }))
        .map_err(|payload| {
            crate::error::Error::backend(format!(
                "Profiler::with_state: callback panicked: {}",
                panic_message(payload)
            ))
        });
        let staged = restore.state.take().expect("state is present");
        PROFILE_SLOT.with(|slot| {
            *slot.borrow_mut() = Some((self.session, staged));
        });
        // The guard's drop now finds `state: None` and does nothing.
        core::mem::forget(restore);
        out
    }

    /// Time `f` on the session state with `client.profile` and return its
    /// value with the device duration.
    ///
    /// On a device-timestamp runtime the duration is subject to the
    /// multi-pass limitation documented on [`SpanTiming`]: callers that
    /// present it (e.g. the profile driver) must check single-pass fit
    /// before calling it whole-stage device time.
    ///
    /// Stream identity: `client.profile` captures the calling thread's stream
    /// before dispatch, and the profile closure runs on the runner thread
    /// under the caller-propagated stream (`ChannelDeviceHandle::exclusive`
    /// wraps the dispatched closure in `StreamId::executes`; nested profile
    /// calls from the runner thread execute inline on that same stream,
    /// `channel.rs:151`). The span body therefore launches on the stream the
    /// profile token was opened for. `T` must be `Send` because the profile
    /// closure crosses to the runner thread; the session state itself never
    /// does (it lives in runner-thread-local storage). Must not overlap a
    /// `with_state` body on the same session (see [`profile_session`]).
    ///
    /// The `client.profile` closure is the only span mechanism the pinned
    /// client offers (`client.rs:886` builds its token with `start_profile`/
    /// `end_profile` through `submit_blocking`, but neither is public), so
    /// per-stage device times come from whole-closure spans around shared
    /// production stage functions — never from a callback that ends one
    /// token and starts the next.
    ///
    /// As with [`Profiler::with_state`], failures are values: a missing,
    /// foreign-session or wrong-typed state, a `client.profile` error, and a
    /// panic inside `f` all return `Err`. The callback panic is caught
    /// *inside* the profiling closure (so CubeCL still runs `end_profile`
    /// and closes its token) with the state already restored, and the next
    /// span works and sees the state.
    pub fn span<S: 'static, T: Send + 'static>(
        &self,
        name: &str,
        f: fn(&mut S) -> T,
    ) -> crate::error::Result<(T, SpanTiming)> {
        let session = self.session;
        let name_owned = name.to_string();
        let (outcome, duration) = self
            .client
            .profile(
                move || -> Result<T, String> {
                    let staged: Box<S> = PROFILE_SLOT.with(|slot| {
                        let slot_ref = slot.borrow();
                        let Some((id, _)) = slot_ref.as_ref() else {
                            return Err(format!(
                                "Profiler::span {name_owned}: no profile_session state on this thread"
                            ));
                        };
                        if *id != session {
                            return Err(format!(
                                "Profiler::span {name_owned}: session mismatch (handle {session}, slot holds {id})"
                            ));
                        }
                        if !slot_ref.as_ref().expect("slot holds a state").1.is::<S>() {
                            return Err(format!(
                                "Profiler::span {name_owned}: session state type mismatch"
                            ));
                        }
                        drop(slot_ref);
                        slot.borrow_mut()
                            .take()
                            .expect("the state was just validated")
                            .1
                            .downcast::<S>()
                            .map_err(|_| {
                                format!(
                                    "Profiler::span {name_owned}: session state type mismatch"
                                )
                            })
                    })?;
                    // Restore the state before returning, on every path: the
                    // closure returns normally (so `end_profile` always runs),
                    // and a callback panic becomes `Err`, never an unwind
                    // across the profile token.
                    struct Restore<S: 'static> {
                        session: u64,
                        state: Option<Box<S>>,
                    }
                    impl<S: 'static> Drop for Restore<S> {
                        fn drop(&mut self) {
                            if let Some(state) = self.state.take() {
                                PROFILE_SLOT.with(|slot| {
                                    *slot.borrow_mut() = Some((self.session, state));
                                });
                            }
                        }
                    }
                    let mut restore = Restore {
                        session,
                        state: Some(staged),
                    };
                    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        f(restore.state.as_deref_mut().expect("state is present"))
                    }))
                    .map_err(|payload| {
                        format!(
                            "Profiler::span {name_owned}: callback panicked: {}",
                            panic_message(payload)
                        )
                    })?;
                    let staged = restore.state.take().expect("state is present");
                    PROFILE_SLOT.with(|slot| {
                        *slot.borrow_mut() = Some((session, staged));
                    });
                    core::mem::forget(restore);
                    Ok(out)
                },
                name,
            )
            .map_err(|err| crate::error::Error::backend(format!("Profiler::span: {err}")))?;
        let outcome = outcome
            .map_err(|err| crate::error::Error::backend(format!("Profiler::span: {err}")))?;
        let timer = match duration.timing_method().to_string().as_str() {
            "device" => "DeviceTimestamps",
            "system" => "SystemTime",
            _ => "Unavailable",
        };
        let ticks = cubecl::future::block_on(duration.resolve());
        Ok((
            outcome,
            SpanTiming {
                ms: ticks.duration().as_secs_f64() * 1000.0,
                timer,
            },
        ))
    }
}

/// A caught callback panic as a message, for the harness `Err` values above.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|message| message.to_string())
        })
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Run a whole profiling session on the device runner thread and return its
/// `Send` result (architecture §6.5 device path).
///
/// `build` constructs the non-`Send` session state `S` (model, workspace,
/// trainer: all `Rc`-holding) and `run` uses it through the [`Profiler`];
/// both are `Send` closures themselves, so entering the runner once through
/// the client's `exclusive` is sound. `S` is created, used and dropped on
/// the runner thread only — the slot is cleared (dropping `S`) before the
/// session returns, so nothing `!Send` crosses a thread. No `unsafe`.
///
/// `run` receives only the profiler, not `&mut S`: every access to the
/// state — untimed work through [`Profiler::with_state`], timed stages
/// through [`Profiler::span`] — takes the state out of runner-thread-local
/// storage for the duration of its own body and puts it back afterwards.
/// Handing `run` a `&mut S` directly (as an earlier sketch did) cannot work
/// in safe Rust: while `run` holds that borrow, no span could reach the
/// state without aliasing it, and taking the state out for `run` leaves
/// spans with an empty slot. The take/put discipline here keeps sequential
/// uses sound without `unsafe`.
///
/// Inside `run`, [`Profiler::span`] times stages with real `client.profile`
/// spans: nested profile calls execute inline because the caller already is
/// the runner (`channel.rs:151`). Callbacks that overlap on the same session
/// (a span inside a `with_state` body or vice versa) fail with a clear error;
/// sequential spans and sessions nested inside `run` are supported (a nested
/// session saves and restores the outer state).
///
/// Bodies must return errors as values rather than panic: a panic inside a
/// span or `with_state` body is caught and returned as `Err` (the profiling
/// token is always closed and the state restored), but a panic anywhere else
/// in `run` still unwinds through the runner. Such a runner-thread panic does
/// *not* break the device channel for the process: `channel.rs` catches task
/// panics (`catch_unwind` around the task with a warning, then `CallError`,
/// which surfaces here through `exclusive` as a backend error), and a
/// cross-thread shim panic surfaces the same way when the reply never
/// arrives — the channel keeps serving later tasks either way.
pub fn profile_session<R: Runtime, S: 'static, O: Send + 'static>(
    device: &Device<R>,
    build: impl FnOnce() -> S + Send + 'static,
    run: impl FnOnce(&Profiler<R>) -> O + Send + 'static,
) -> crate::error::Result<O> {
    let client = device.client().clone();
    let session = NEXT_SESSION_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    device
        .client()
        .exclusive(move || {
            struct RestoreSlot {
                saved: Option<(u64, Box<dyn std::any::Any>)>,
            }
            impl Drop for RestoreSlot {
                fn drop(&mut self) {
                    PROFILE_SLOT.with(|slot| {
                        // Drop this session's state here, on the runner
                        // thread, then restore any outer session's value.
                        slot.borrow_mut().take();
                        if let Some(saved) = self.saved.take() {
                            *slot.borrow_mut() = Some(saved);
                        }
                    });
                }
            }
            let restore = RestoreSlot {
                saved: PROFILE_SLOT.with(|slot| slot.borrow_mut().take()),
            };
            PROFILE_SLOT.with(|slot| {
                *slot.borrow_mut() = Some((session, Box::new(build())));
            });
            let profiler = Profiler {
                client: client.clone(),
                session,
            };
            // The state stays in the slot while `run` uses it through the
            // profiler (`with_state` for untimed work, `span` for timed
            // stages); each takes it out and puts it back in turn.
            let out = run(&profiler);
            // Drop this session's state here, still on the runner thread,
            // and restore any outer session's value (`Drop` also covers
            // unwinding through `run`).
            drop(restore);
            out
        })
        .map_err(launch_error)
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
    let bytes = handle
        .stream
        .executes(|| client.read_one_unchecked(handle.clone()));
    count_runtime_read(bytes.len());
    bytes
}

/// Read several buffers back to the host under one synchronisation, counting it.
///
/// On wgpu every read is a queue submit, a staging map and a blocking poll for
/// the map's callback, and that wait is paid per read rather than per byte —
/// about 1.4 ms on Metal for a 128-byte buffer, the same for three buffers read
/// together (`examples/bench_host_read.rs`). One `client.read` of every handle
/// pays it once.
///
/// Handles allocated on different streams cannot share that read, for the reason
/// [`read_handle`] gives, so they fall back to one read each. Callers flush their
/// own stream first ([`check_launches`]).
pub(crate) fn read_handles<R: Runtime>(
    device: &Device<R>,
    handles: Vec<Handle>,
) -> Vec<cubecl::bytes::Bytes> {
    let Some(stream) = handles.first().map(|handle| handle.stream) else {
        return Vec::new();
    };
    if handles.iter().all(|handle| handle.stream == stream) {
        count_read();
        let client = device.client();
        let out = stream.executes(|| client.read(handles));
        count_runtime_read(out.iter().map(|b| b.len()).sum());
        out
    } else {
        handles
            .iter()
            .map(|handle| {
                count_read();
                read_handle(device, handle)
            })
            .collect()
    }
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

/// Turn on CubeCL's compiled-kernel cache for a process that has no `cubecl.toml`.
///
/// CubeCL reads its configuration from a `cubecl.toml` in the working directory or
/// one of its parents, and without one it compiles every kernel from scratch in
/// every process — minutes of JIT at the start of a Python training script run
/// from its own project directory, where this repository's `cubecl.toml` is not on
/// the path. This installs the cache in the user cache directory
/// (`$XDG_CACHE_HOME` or `~/.cache`, then `mamba3/kernels/<source hash>`) unless a
/// config file would be found, the configuration was already read, or
/// `MAMBA3_KERNEL_CACHE=0`. Call it before the first device is created.
///
/// The directory is named after a hash of this crate's sources (`build.rs`):
/// CubeCL keys cached kernels by type and comptime arguments only, so a cache
/// shared across builds would keep serving a kernel's old body after an edit.
pub fn default_kernel_cache() {
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    if std::env::var("MAMBA3_KERNEL_CACHE").as_deref() == Ok("0") {
        return;
    }
    let Ok(mut dir) = std::env::current_dir() else {
        return;
    };
    loop {
        for name in ["cubecl.toml", "CubeCL.toml", "burn.toml", "Burn.toml"] {
            if dir.join(name).exists() {
                return;
            }
        }
        if !dir.pop() {
            break;
        }
    }
    let mut storage = CubeClRuntimeConfig::storage().lock();
    if storage.is_some() {
        return;
    }
    let Some(root) = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
    else {
        return;
    };
    let mut config = CubeClRuntimeConfig::default().override_from_env();
    if config.compilation.cache.is_none() {
        config.compilation.cache = Some(CacheConfig::File(
            root.join("mamba3").join("kernels").join(env!("MAMBA3_SRC_HASH")),
        ));
    }
    *storage = Some(std::sync::Arc::new(config));
}

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
static TALLY: std::sync::Mutex<
    Option<std::collections::HashMap<(String, String, &'static str, u32), usize>>,
> = std::sync::Mutex::new(None);

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

thread_local! {
    /// Model-region label stack for the launch tally (`M0.2`). The top entry (or
    /// `"-"`) is charged with every launch while a tally runs. One Vec push/pop
    /// per scope; left in place permanently.
    static TALLY_LABELS: core::cell::RefCell<Vec<&'static str>> =
        const { core::cell::RefCell::new(Vec::new()) };
    /// Current public-op name for the launch tally. Public launching ops set this
    /// around their kernel launch so the tally can name the op, not just the
    /// helper that launched (e.g. `flat_launch`, shared by every flat kernel).
    static TALLY_OP: core::cell::RefCell<Vec<&'static str>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

/// Pops its label off [`TALLY_LABELS`] on drop.
pub struct TallyScope {
    is_label: bool,
}

impl Drop for TallyScope {
    fn drop(&mut self) {
        if self.is_label {
            TALLY_LABELS.with(|s| {
                s.borrow_mut().pop();
            });
        } else {
            TALLY_OP.with(|s| {
                s.borrow_mut().pop();
            });
        }
    }
}

/// Push a model-region label charged to every launch until the scope drops.
///
/// Nesting takes the innermost label. Used by the model (`mixer.project`,
/// `scan.intra`, `encoder`, `backward`, `optimizer`, …) so the tally prints
/// `label / op / file:line` instead of just the helper's line.
pub fn tally_scope(label: &'static str) -> TallyScope {
    TALLY_LABELS.with(|s| s.borrow_mut().push(label));
    TallyScope { is_label: true }
}

/// Push a public-op name charged to every launch until the scope drops.
///
/// Helpers (`flat_launch` and equivalents) forward to the charge; every `pub fn`
/// that launches wraps its body in this with its own name (`"silu"`,
/// `"strided_copy"`, `"sum_dim"`, …).
pub fn tally_op_scope(op: &'static str) -> TallyScope {
    TALLY_OP.with(|s| s.borrow_mut().push(op));
    TallyScope { is_label: false }
}

fn tally_label_top() -> String {
    TALLY_LABELS.with(|s| {
        s.borrow()
            .last()
            .copied()
            .unwrap_or("-")
            .to_string()
    })
}

fn tally_op_top(file: &'static str) -> String {
    TALLY_OP.with(|s| {
        s.borrow()
            .last()
            .copied()
            .map(str::to_string)
            .unwrap_or_else(|| {
                // Fall back to the helper's file stem so the op field is never
                // empty even where a public op has not been named yet.
                file.rsplit('/').next().unwrap_or(file).trim_end_matches(".rs").to_string()
            })
    })
}

/// One attributed tally row: model region, public op, and source site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyRow {
    /// Innermost [`tally_scope`] label, or `"-"`.
    pub label: String,
    /// Innermost [`tally_op_scope`] op, or the launching helper's file stem.
    pub op: String,
    /// `file:line` of the launch.
    pub site: String,
    /// Launches charged here.
    pub count: usize,
}

/// Launches recorded since [`start_launch_tally`], as `(file:line, count)` sorted
/// by descending count.
pub fn launch_tally() -> Vec<(String, usize)> {
    launch_tally_detailed()
        .into_iter()
        .map(|row| (format!("{} / {} / {}", row.label, row.op, row.site), row.count))
        .collect()
}

/// Launches recorded since [`start_launch_tally`] with the label, op and site
/// kept separate, sorted by descending count (ties by name for run-to-run
/// stability).
pub fn launch_tally_detailed() -> Vec<TallyRow> {
    let guard = TALLY.lock().expect("launch tally is not poisoned");
    let Some(sites) = guard.as_ref() else {
        return Vec::new();
    };
    let mut rows: Vec<TallyRow> = sites
        .iter()
        .map(|((label, op, file, line), count)| TallyRow {
            label: label.clone(),
            op: op.clone(),
            site: format!("{file}:{line}"),
            count: *count,
        })
        .collect();
    // Ties broken by name so a printed tally is stable run to run, which is the
    // whole point of counting launches rather than timing them.
    rows.sort_by(|a, b| {
        b.count.cmp(&a.count).then_with(|| {
            (&a.label, &a.op, &a.site).cmp(&(&b.label, &b.op, &b.site))
        })
    });
    rows
}

/// Forget every recorded site, leaving the tally recording if it already was.
pub fn reset_launch_tally() {
    if let Some(sites) = TALLY.lock().expect("launch tally is not poisoned").as_mut() {
        sites.clear();
    }
}

type TallyKey = (String, String, &'static str, u32);

/// Timed-tally state: the drain hook, the launch whose time is still running and
/// when it started, and the milliseconds charged per site.
struct TimedTally {
    drain: Box<dyn Fn() + Send>,
    pending: Option<(TallyKey, std::time::Instant)>,
    ms: std::collections::HashMap<TallyKey, f64>,
}

static TIMED: std::sync::Mutex<Option<TimedTally>> = std::sync::Mutex::new(None);

/// Also time every launch while the tally runs: before each launch `drain` is called
/// (it must wait for every queued kernel), and the wall time since the previous
/// launch is charged to that previous launch's site. This serialises the queue, so
/// the total is slower than an untimed step; the split between sites is what it is
/// for. `None` turns timing off.
pub fn set_launch_timer(drain: Option<Box<dyn Fn() + Send>>) {
    *TIMED.lock().expect("timed tally is not poisoned") = drain.map(|drain| TimedTally {
        drain,
        pending: None,
        ms: std::collections::HashMap::new(),
    });
}

/// Close the running launch (drain and charge it) so the timed tally is complete.
pub fn flush_launch_timer() {
    if let Some(t) = TIMED.lock().expect("timed tally is not poisoned").as_mut() {
        (t.drain)();
        if let Some((key, started)) = t.pending.take() {
            *t.ms.entry(key).or_insert(0.0) += started.elapsed().as_secs_f64() * 1e3;
        }
    }
}

/// Milliseconds charged per `label / op / file:line` since [`set_launch_timer`],
/// descending. Call [`flush_launch_timer`] first.
pub fn launch_time_tally() -> Vec<(String, f64)> {
    let guard = TIMED.lock().expect("timed tally is not poisoned");
    let Some(t) = guard.as_ref() else {
        return Vec::new();
    };
    let mut rows: Vec<(String, f64)> = t
        .ms
        .iter()
        .map(|((label, op, file, line), ms)| (format!("{label} / {op} / {file}:{line}"), *ms))
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    rows
}

/// Charge one launch to `site`, if a tally is running.
#[inline]
fn record_site(site: &'static core::panic::Location<'static>) {
    if !TALLY_ON.load(core::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let label = tally_label_top();
    let op = tally_op_top(site.file());
    let key: TallyKey = (label, op, site.file(), site.line());
    if let Some(t) = TIMED.lock().expect("timed tally is not poisoned").as_mut() {
        (t.drain)();
        let now = std::time::Instant::now();
        if let Some((prev, started)) = t.pending.take() {
            *t.ms.entry(prev).or_insert(0.0) += (now - started).as_secs_f64() * 1e3;
        }
        t.pending = Some((key.clone(), now));
    }
    if let Some(sites) = TALLY.lock().expect("launch tally is not poisoned").as_mut() {
        *sites.entry(key).or_insert(0) += 1;
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

/// Host-to-device uploads counted since the last [`reset_upload_count`].
static UPLOADS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Metadata buffers [`meta_handle`] had to upload since the last
/// [`reset_meta_miss_count`].
static META_MISSES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Buffers this crate has created from host data so far.
///
/// The mirror of [`read_count`] for the other direction: every
/// [`crate::tensor::Tensor::from_data`], [`crate::tensor::ops::index::IdTensor::from_slice`]
/// and their by-value forms, every metadata buffer [`meta_handle`] did not
/// already hold, and the index tables of
/// [`crate::tensor::ops::index::scatter_add_rows`]. Fills (`zeros`, `full`,
/// `ones`) are kernels and a launch's scalar arguments travel as uniforms:
/// neither is an upload.
///
/// A training step whose data lives on the device should hold this at zero once
/// every shape of the run has been seen.
pub fn upload_count() -> usize {
    UPLOADS.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set [`upload_count`] back to zero.
pub fn reset_upload_count() {
    UPLOADS.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Record a host-to-device upload of `bytes` bytes.
///
/// An upload always creates its buffer too, so this also charges
/// [`upload_bytes`] and [`allocation_calls`].
pub(crate) fn count_upload(bytes: usize) {
    UPLOADS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    UPLOAD_BYTES.fetch_add(bytes as u64, core::sync::atomic::Ordering::Relaxed);
    ALLOCATION_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// Every device-to-host read issued through [`read_handle`] or
/// [`read_handles`], one per runtime read call.
///
/// [`read_count`] counts the reads a step budget pins; this one counts every
/// runtime read, so a footprint test can prove a warmed call performs exactly
/// one read in total, and it moves with [`reset_transfer_counters`] rather than
/// [`reset_read_count`].
static RUNTIME_READS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Bytes returned by device-to-host reads since [`reset_transfer_counters`].
///
/// Summed from the byte lengths the runtime hands back, so any padding the
/// runtime adds is counted too; a lower bound on what the reads moved.
static DOWNLOAD_BYTES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Bytes passed to the runtime by host uploads since [`reset_transfer_counters`].
///
/// Charged where host data is copied in, so a memory estimate (for example
/// `models::ms2::workspace`) can be reconciled against what really moved.
static UPLOAD_BYTES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Device buffer creations since [`reset_transfer_counters`].
///
/// Every `empty` buffer and every upload creates one; a warmed loop whose
/// allocator is flat must hold this still.
static ALLOCATION_CALLS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Every device-to-host runtime read since [`reset_transfer_counters`].
pub fn runtime_read_count() -> usize {
    RUNTIME_READS.load(core::sync::atomic::Ordering::Relaxed)
}

/// Bytes returned by device-to-host reads since [`reset_transfer_counters`].
pub fn download_bytes() -> u64 {
    DOWNLOAD_BYTES.load(core::sync::atomic::Ordering::Relaxed)
}

/// Bytes passed to the runtime by host uploads since [`reset_transfer_counters`].
pub fn upload_bytes() -> u64 {
    UPLOAD_BYTES.load(core::sync::atomic::Ordering::Relaxed)
}

/// Device buffer creations since [`reset_transfer_counters`].
pub fn allocation_calls() -> usize {
    ALLOCATION_CALLS.load(core::sync::atomic::Ordering::Relaxed)
}

/// Zero the four transfer counters, leaving [`read_count`], [`upload_count`]
/// and [`launch_count`] alone.
///
/// The transfer counters measure reads, bytes and buffers moved; the launch and
/// step-sync counters measure dispatches, so footprint tests reset each group
/// on its own boundary.
pub fn reset_transfer_counters() {
    RUNTIME_READS.store(0, core::sync::atomic::Ordering::Relaxed);
    DOWNLOAD_BYTES.store(0, core::sync::atomic::Ordering::Relaxed);
    UPLOAD_BYTES.store(0, core::sync::atomic::Ordering::Relaxed);
    ALLOCATION_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Record a device buffer creation without host data (an `empty` buffer).
pub(crate) fn count_allocation() {
    ALLOCATION_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// Record one device-to-host runtime read that moved `bytes` bytes.
fn count_runtime_read(bytes: usize) {
    RUNTIME_READS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    DOWNLOAD_BYTES.fetch_add(bytes as u64, core::sync::atomic::Ordering::Relaxed);
}

/// Allocator state reported by the runtime, when it reports any.
///
/// `bytes_reserved` is the high-water mark the pool holds (see
/// [`reserved_bytes`]); `bytes_in_use` and `allocations` say what of it is
/// live. A runtime that stays silent yields `None` from [`memory_snapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Active device slices.
    pub allocations: u64,
    /// Bytes actually in use, excluding padding and pooled memory.
    pub bytes_in_use: u64,
    /// Total bytes reserved on the device, including pooled memory.
    pub bytes_reserved: u64,
}

/// [`MemorySnapshot`] for `device`, or `None` when the runtime stays silent.
pub fn memory_snapshot<R: Runtime>(device: &Device<R>) -> Option<MemorySnapshot> {
    device.client().memory_usage().ok().map(|u| MemorySnapshot {
        allocations: u.number_allocs,
        bytes_in_use: u.bytes_in_use,
        bytes_reserved: u.bytes_reserved,
    })
}

/// Metadata buffers uploaded because [`meta_handle`]'s cache did not hold them.
///
/// Each is also counted by [`upload_count`]; this singles them out, because a
/// steady stream of them means shapes that never repeat.
pub fn meta_miss_count() -> usize {
    META_MISSES.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set [`meta_miss_count`] back to zero.
pub fn reset_meta_miss_count() {
    META_MISSES.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Largest single device allocation since the last [`reset_peak_alloc`].
static PEAK_ALLOC: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// The largest single tensor allocated since the last [`reset_peak_alloc`], in
/// bytes.
///
/// A memory pool sorts allocations into size classes, and the class an
/// allocation falls in decides how large a page is reserved for it; a model
/// that sizes its batches to stay inside one class checks itself with this.
pub fn peak_alloc_bytes() -> usize {
    PEAK_ALLOC.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set [`peak_alloc_bytes`] back to zero.
pub fn reset_peak_alloc() {
    PEAK_ALLOC.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Record a device allocation of `bytes`.
#[inline]
pub(crate) fn note_alloc(bytes: usize) {
    PEAK_ALLOC.fetch_max(bytes, core::sync::atomic::Ordering::Relaxed);
}

/// Device bytes currently owned by live [`Tensor`](crate::tensor::Tensor) and
/// [`IdTensor`](crate::tensor::ops::index::IdTensor) values.
///
/// Each fresh device buffer contributes its byte size once — counted by the
/// [`Tensor::empty`](crate::tensor::Tensor::empty) / `from_data` /
/// [`IdTensor::empty`](crate::tensor::ops::index::IdTensor::empty) /
/// `from_slice` constructor that created its buffer — and every clone shares
/// its source's count while every value's drop subtracts its own. Sharing
/// therefore counts a buffer once per live value sharing it: the total is a
/// conservative upper bound of the tensor-owned device peak (never an
/// undercount through clones), which is the direction a peak estimate needs.
/// Views built by reshape-like constructors count nothing (their buffer is
/// counted through the value they were cut from while it lives).
///
/// What this does NOT see: the scratch arena's retained-but-idle buffers (the
/// creating value already dropped, the arena's stored clone holds the
/// memory), the runtime pool's own pages and padding, and handles the runtime
/// made outside these constructors (upload caches, constant tables). With the
/// arena off the high-water mark is the true transient peak up to view
/// aliasing (a few kilobytes against the test's margin); with it on,
/// retained idle memory is invisible. The MS2 carry-peak test (task F10 item
/// A2) runs with the arena off, so its high-water mark bounds the step's
/// live peak from above.
static LIVE_BYTES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// High-water mark of [`live_bytes`] since the last
/// [`reset_live_high_water`].
static LIVE_HIGH: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Device bytes currently owned by live tensor values; see the
/// `LIVE_BYTES` counter below.
pub fn live_bytes() -> u64 {
    LIVE_BYTES.load(core::sync::atomic::Ordering::Relaxed)
}

/// The largest [`live_bytes`] observed since the last
/// [`reset_live_high_water`]: the peak of tensor-owned device memory inside
/// the measured window, not what is retained afterwards.
pub fn live_high_water_bytes() -> u64 {
    LIVE_HIGH.load(core::sync::atomic::Ordering::Relaxed)
}

/// Point the [`live_high_water_bytes`] mark at the current [`live_bytes`],
/// so the next read measures the peak inside the window that starts here.
pub fn reset_live_high_water() {
    LIVE_HIGH.store(
        LIVE_BYTES.load(core::sync::atomic::Ordering::Relaxed),
        core::sync::atomic::Ordering::Relaxed,
    );
}

/// Record a fresh device buffer of `bytes` coming under tensor ownership.
#[inline]
pub(crate) fn note_live_alloc(bytes: usize) {
    note_live_add(bytes);
}

/// Record one more live value sharing `bytes` (a clone): counted anew, so
/// the clone's drop balances it.
#[inline]
pub(crate) fn note_live_clone(bytes: usize) {
    note_live_add(bytes);
}

#[inline]
fn note_live_add(bytes: usize) {
    let now = LIVE_BYTES.fetch_add(bytes as u64, core::sync::atomic::Ordering::Relaxed) + bytes as u64;
    if now > LIVE_HIGH.load(core::sync::atomic::Ordering::Relaxed) {
        LIVE_HIGH.fetch_max(now, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Record one value's drop: subtract the bytes it counted. Every counted
/// value drops exactly once, so the books balance under any aliasing.
#[inline]
pub(crate) fn note_live_free(bytes: usize) {
    LIVE_BYTES.fetch_sub(bytes as u64, core::sync::atomic::Ordering::Relaxed);
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

/// The small tables cached for one device identity.
struct DeviceTables {
    /// Dead once the device and every tensor on it have been dropped.
    alive: std::sync::Weak<()>,
    tables: std::collections::HashMap<Vec<u32>, Handle>,
}

/// Small tables on the device, keyed by a device identity and then by contents.
type TableCache<K> = core::cell::RefCell<std::collections::HashMap<K, DeviceTables>>;

thread_local! {
    /// Cached `[shape, strides]` buffers, keyed by device and then by contents.
    ///
    /// Thread-local on purpose. A [`Handle`] records the stream it was created on,
    /// so keeping the cache per-thread means a buffer is only ever reused by the
    /// thread that uploaded it, and the cache needs no lock on a path this hot.
    static META_CACHE: TableCache<usize> = core::cell::RefCell::new(std::collections::HashMap::new());

    /// The optimizer's per-chunk tables (slot lengths, sum-of-squares layout),
    /// kept apart from the shape tables: a model has one entry per chunk of
    /// trainable tensors, and sharing [`META_CACHE`]'s budget would let a model
    /// with many of them evict every shape table of the run, and back.
    static OPTIMIZER_TABLES: TableCache<usize> = core::cell::RefCell::new(std::collections::HashMap::new());
}

/// Distinct optimizer tables held per device before that cache is dropped.
const OPTIMIZER_TABLE_LIMIT: usize = 4096;

/// The cached table with these `contents` for the device `key` names, or the
/// one `upload` makes, kept for next time.
///
/// A hit is two lookups. A miss on a device this thread has not cached for
/// before first drops the tables of devices that no longer exist, so a process
/// that opens many devices (each `Device::new` is a new identity) does not keep
/// every one's tables for the life of the thread.
fn cached_table<K: core::hash::Hash + Eq + Copy, R: Runtime>(
    cache: &'static std::thread::LocalKey<TableCache<K>>,
    key: K,
    device: &Device<R>,
    contents: &[u32],
    limit: usize,
    upload: impl FnOnce() -> Handle,
) -> Handle {
    cache.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(handle) = cache.get(&key).and_then(|entry| entry.tables.get(contents)) {
            return handle.clone();
        }
        if !cache.contains_key(&key) {
            cache.retain(|_, entry| entry.alive.strong_count() > 0);
        }
        let entry = cache.entry(key).or_insert_with(|| DeviceTables {
            alive: device.liveness(),
            tables: std::collections::HashMap::new(),
        });
        // Clearing wholesale rather than evicting one entry keeps the
        // bookkeeping to a comparison.
        if entry.tables.len() >= limit {
            entry.tables.clear();
        }
        let handle = upload();
        entry.tables.insert(contents.to_vec(), handle.clone());
        handle
    })
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
    u32_table(&META_CACHE, META_CACHE_LIMIT, device, meta)
}

/// [`meta_handle`] for the optimizer's per-chunk tables, which have a cache
/// and a budget of their own ([`OPTIMIZER_TABLES`]).
pub(crate) fn optimizer_table_handle<R: Runtime>(device: &Device<R>, table: &[u32]) -> Handle {
    u32_table(&OPTIMIZER_TABLES, OPTIMIZER_TABLE_LIMIT, device, table)
}

fn u32_table<R: Runtime>(
    cache: &'static std::thread::LocalKey<TableCache<usize>>,
    limit: usize,
    device: &Device<R>,
    table: &[u32],
) -> Handle {
    let upload = || {
        count_upload(core::mem::size_of_val(table));
        META_MISSES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        device.client().create_from_slice(u32::as_bytes(table))
    };
    if !meta_cache_enabled() {
        return upload();
    }
    cached_table(cache, device.id(), device, table, limit, upload)
}

thread_local! {
    /// Cached small float tables, keyed by device and element type and then by
    /// the bits of their `f32` contents. The float twin of [`META_CACHE`].
    static FLOAT_META_CACHE: TableCache<(usize, DType)> =
        core::cell::RefCell::new(std::collections::HashMap::new());
}

/// Upload a small table of floats as `E`, or hand back the one already on the
/// device: [`meta_handle`] for values that are not integers.
///
/// The optimizer's per-slot weight-decay table is the case this exists for: a
/// handful of floats that depend only on which parameters are being updated,
/// uploaded again on every step of every chunk until they were kept. Keyed by
/// contents, so a parameter list that changes is simply a new entry.
pub(crate) fn float_meta_handle<R: Runtime, E: FloatElem>(
    device: &Device<R>,
    values: &[f32],
) -> Handle {
    let upload = || {
        count_upload(values.len() * core::mem::size_of::<E>());
        META_MISSES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        device
            .client()
            .create_from_slice(E::as_bytes(&E::slice_from_f32(values)))
    };
    if !meta_cache_enabled() {
        return upload();
    }
    let key: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    cached_table(
        &FLOAT_META_CACHE,
        (device.id(), E::DTYPE),
        device,
        &key,
        META_CACHE_LIMIT,
        upload,
    )
}

/// Which constant a [`constant_handle`] entry holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ConstantKind {
    /// The `[n, n]` mask that is one where `col < row`.
    StrictCausalMask,
}

thread_local! {
    /// Cached constant tensors, keyed by device, element type, kind and size.
    static CONSTANT_CACHE: core::cell::RefCell<
        std::collections::HashMap<(usize, DType, ConstantKind, usize), Handle>,
    > = core::cell::RefCell::new(std::collections::HashMap::new());
}

/// A constant that depends only on its kind and size, uploaded once and kept:
/// [`meta_handle`] for constants too large to key by their contents.
///
/// `build` produces the values on a miss. Safe to share for the reason
/// [`meta_handle`] gives: no operation mutates its inputs.
pub(crate) fn constant_handle<R: Runtime, E: FloatElem>(
    device: &Device<R>,
    kind: ConstantKind,
    size: usize,
    build: impl FnOnce() -> Vec<f32>,
) -> Handle {
    let upload = |values: Vec<f32>| {
        count_upload(values.len() * core::mem::size_of::<E>());
        device
            .client()
            .create_from_slice(E::as_bytes(&E::slice_from_f32(&values)))
    };
    if !meta_cache_enabled() {
        return upload(build());
    }
    CONSTANT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let key = (device.id(), E::DTYPE, kind, size);
        if let Some(handle) = cache.get(&key) {
            return handle.clone();
        }
        if cache.len() >= META_CACHE_LIMIT {
            cache.clear();
        }
        let handle = upload(build());
        cache.insert(key, handle.clone());
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

/// Device identities this thread's shape-table cache holds tables for.
///
/// For tests of the cache's lifetime: the tables of a device that has been
/// dropped, with every tensor on it, go the next time a new device is cached
/// for.
pub fn meta_cache_devices() -> usize {
    META_CACHE.with(|cache| cache.borrow().len())
}

/// Drop every cached metadata buffer.
///
/// Only the memory-footprint tests need this: they measure reserved bytes before
/// and after a loop, and a cache that is still filling during the first iteration
/// would look like a leak.
pub fn clear_meta_cache() {
    META_CACHE.with(|cache| cache.borrow_mut().clear());
    OPTIMIZER_TABLES.with(|cache| cache.borrow_mut().clear());
    FLOAT_META_CACHE.with(|cache| cache.borrow_mut().clear());
    CONSTANT_CACHE.with(|cache| cache.borrow_mut().clear());
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
        CubeDim::new_1d(cpu_units(client, lanes, work_per_lane))
    };
    (
        cubecl::calculate_cube_count_elemwise(client, lanes, cube_dim),
        cube_dim,
    )
}

/// Worker threads a CPU-like runtime should get for `lanes` lanes of
/// `work_per_lane` element operations each; see [`WORK_PER_CPU_UNIT`].
fn cpu_units<R: Runtime>(client: &ComputeClient<R>, lanes: usize, work_per_lane: usize) -> u32 {
    let cores = client
        .properties()
        .hardware
        .num_cpu_cores
        .unwrap_or(1)
        .max(1) as usize;
    let total = lanes.saturating_mul(work_per_lane.max(1));
    let units = (total / WORK_PER_CPU_UNIT).clamp(1, cores.min(lanes.max(1)));
    (units as u32).min(ELEMWISE_CUBE_DIM)
}

/// [`launch_1d`] for a kernel whose units each walk a *span* of consecutive
/// lanes: unit `pos` covers lanes `pos * span .. min((pos + 1) * span, lanes)`.
///
/// The two runtime families want the lanes dealt out differently:
///
/// * On a GPU-like runtime neighbouring lanes of one plane should touch
///   neighbouring addresses — that is what coalesces — so the span is 1 and
///   the geometry is exactly [`launch_1d`]'s.
/// * On CubeCL's CPU runtime every unit of a cube is its own worker thread and
///   the cube count is a serial loop inside each one, so with one lane per
///   unit, worker `u` handles lanes `u, u + units, u + 2·units, …`. A kernel
///   writing one scalar per lane then has every worker writing into every
///   cache line, and the line ping-pongs between cores on each store. Giving
///   each worker one contiguous run of `lanes / units` lanes instead (a single
///   cube) removes the sharing and lets each thread stream its own memory.
///
/// Returns the geometry and the span.
#[track_caller]
pub(crate) fn launch_1d_spans<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    work_per_lane: usize,
) -> (CubeCount, CubeDim, usize) {
    LAUNCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    record_site(core::panic::Location::caller());

    if client.properties().hardware.plane_size_max > 1 {
        let cube_dim = CubeDim::new(client, lanes);
        return (
            cubecl::calculate_cube_count_elemwise(client, lanes, cube_dim),
            cube_dim,
            1,
        );
    }
    let units = cpu_units(client, lanes, work_per_lane) as usize;
    let span = lanes.div_ceil(units).max(1);
    (
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(lanes.div_ceil(span) as u32),
        span,
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

// ---------------------------------------------------------------------------
// Scratch arena: a scope-activated recycler of device buffers above the
// runtime allocator (task T3).
// ---------------------------------------------------------------------------

/// Buffers of one byte size the arena holds past this count are not
/// retained: a plain allocation is used instead. The total is bounded by
/// [`ScratchArena::new`]'s `max_bytes` anyway; this only stops one hot size
/// from crowding out every other.
const SCRATCH_PER_SIZE_LIMIT: usize = 64;

/// Whether a scratch arena is active on the current thread: the single cheap
/// check [`crate::tensor::Tensor::empty`] and
/// [`crate::tensor::ops::index::IdTensor::empty`] consult before touching the
/// arena.
///
/// [`with_scratch`] sets this when it installs the arena and clears it when
/// the scope drops (including on unwind), so the flag mirrors the presence
/// of [`ACTIVE_SCRATCH`] exactly. Outside a scope one `Cell::get` — one
/// thread-local lookup, no `RefCell` borrow — decides that both the reuse
/// lookup and the retention are skipped, where the constructors previously
/// paid one thread-local lookup each.
pub(crate) fn scratch_scope_active() -> bool {
    SCRATCH_ACTIVE_FLAG.get()
}

/// A scope-activated recycler of device buffers that sits above the
/// runtime's allocator.
///
/// The decode loop allocates every float op output through
/// [`crate::tensor::Tensor::empty`] (and every id output through
/// [`crate::tensor::ops::index::IdTensor::empty`]), which allocates through
/// the runtime's memory pool on every call. A warmed `generate` therefore
/// created one device buffer per launch. Inside
/// [`with_scratch`]'s scope those two constructors instead reuse a retained
/// buffer of exactly the same byte size, so a warmed loop performs zero
/// [`allocation_calls`] while launching exactly as before.
///
/// Activation is per thread: [`with_scratch`] installs the arena for the
/// current thread for the duration of its closure. Outside a scope nothing
/// changes anywhere. Each generation bucket owns one arena, so shapes that
/// never repeat never share buffers.
///
/// # Confinement (task F10 item A1)
///
/// Reuse is only sound while every tensor allocated from the arena stays on
/// the allocating thread: a public tensor can be cloned and sent anywhere,
/// so confinement holds by construction, not by convention. The rule:
///
/// * the arena is activated ONLY inside the decode loops of
///   [`Ms2Model::generate_with_hook`](crate::models::ms2::generate::Ms2Model::generate_with_hook),
///   [`generate_packed_with_hook`](crate::models::ms2::generate::Ms2Model::generate_packed_with_hook)
///   and
///   [`generate_resident_with_hook`](crate::models::ms2::generate::Ms2Model::generate_resident_with_hook)
///   when no hook is installed — there the decoder state is private to the
///   call and is dropped before the call returns;
/// * the staged public API — every public entry point that returns or
///   exposes a [`DecoderState`](crate::models::ms2::decoder::DecoderState)
///   or tensors produced inside a step (`generate_decoder_init*`,
///   `generate_decode_step*`), and every `*_with_hook` entry point with a
///   hook installed — runs WITHOUT the arena (plain allocations, exactly as
///   with `MAMBA3_MS2_SCRATCH=0`);
/// * what the three owning calls return is never scratch-backed: `generate`
///   and `generate_packed` return host batches, and `generate_resident`
///   leases bucket buffers allocated outside any scope (see the audit on
///   [`Ms2Model::generate_with_hook`](crate::models::ms2::generate::Ms2Model::generate_with_hook)).
///   Each owning call ends with
///   [`ScratchArena::debug_assert_no_external_refs`], which proves in debug
///   builds that no retained buffer is still referenced from outside the
///   arena when the call returns.
///
/// # Safety
///
/// A retained handle is handed out only when [`Handle::can_mut`] is true,
/// i.e. the host holds no other reference to it: the tensor that used it
/// has been dropped (the arena's own stored clone does not count against
/// this; see `cubecl-runtime-0.10.0/.../memory_pool/handle.rs`, where
/// `can_mut` is `strong_count <= 2`). Kernels already queued that read the
/// buffer's old contents were submitted earlier and execute earlier —
/// submission order is execution order on every backend here — which is the
/// same guarantee the in-place recurrent step relies on. A tensor that
/// escapes the scope (returned to the caller, stored in the workspace) keeps
/// its handle alive, so `can_mut` is false and the arena never reuses it.
/// Contents of a recycled buffer are stale, which is already the contract
/// of `empty` ("uninitialised").
///
/// # Streams
///
/// Reuse is bound to the allocating stream as well as the device (finding
/// T3F-1). CubeCL 0.10 orders queued work per thread stream
/// (`cubecl-common-0.10.0/src/stream_id.rs`: with `multi_threading`, which
/// every `std` non-wasm build sets, each thread owns its stream id, and the
/// scheduler records each binding's creation stream): a buffer handed to a
/// tensor on another stream could be overwritten by the first stream's
/// still-queued write, while the handle still names the first stream, so
/// the second stream's later launch would not drain the first stream's
/// queue. Each retained buffer therefore records the stream it was
/// allocated on (`Handle::stream`, i.e. `StreamId::current()` of the
/// allocating thread), and [`scratch_reuse`] serves only buffers of the
/// current thread's stream (and the arena's device). A same-size buffer
/// that is free but lives on another stream falls through to a fresh
/// allocation, counted in [`ScratchStats::cross_stream_fallthroughs`] (as
/// well as [`ScratchStats::fell_through`]). Same-stream reuse — the whole
/// warmed decode loop — is unaffected. A mutex around checkout alone could
/// not order device work, which is why the stream, not a lock, is the
/// guard.
///
/// # Audit: no decode-step caller reads what `empty` left behind
///
/// Every `Tensor::empty`/`IdTensor::empty` in the decode loop was traced to
/// its call site (per-step multiset: the step embedding, the per-layer
/// norm/projection/context/output/residual chain, the head product, the
/// fused input projection, the norm scales, and the line-sized mixer
/// scratch — no id-tensor allocation at all in the fused loop):
///
/// * every op output is fully written by its kernel before any read
///   (elementwise kernels guard `ABSOLUTE_POS < len` over the whole
///   buffer; reductions initialise their accumulators locally);
/// * `zeros`/`full`/`ones` overwrite with a fill kernel;
/// * the option-absent placeholders (the norm gain placeholder, the
///   no-convolution history scratch, the coef kernel's reset/skip
///   placeholders) are never read: the kernels' reads are gated by
///   `comptime` bools derived from the same `Option`s.
///
/// So recycling changes no value: `generate` with the arena on equals off
/// bit-for-bit on the cpu runtime (pinned by `tests/ms2_generation_footprint.rs`).
///
/// [`Handle::can_mut`]: cubecl::server::Handle::can_mut
pub struct ScratchArena {
    shared: std::sync::Arc<ScratchShared>,
}

/// Reference-counted state behind [`ScratchArena`], so [`with_scratch`] can
/// install the active arena in thread-local storage without raw pointers or
/// `unsafe`.
struct ScratchShared {
    /// Buffers retained so far, guarded because `empty` may run on any
    /// thread while a scope on another thread is active.
    inner: std::sync::Mutex<ScratchInner>,
    /// Upper bound on retained bytes (constructor argument).
    max_bytes: usize,
}

/// The retained buffers and counters of one [`ScratchArena`].
struct ScratchInner {
    /// Device identity the retained buffers were allocated on (`None`
    /// until the first buffer is retained). Buffers are only ever handed
    /// to the device they came from.
    device: Option<usize>,
    /// One stored handle per retained buffer, keyed by exact byte size.
    entries: Vec<ScratchEntry>,
    /// Bytes currently retained.
    bytes_held: u64,
    /// `empty` calls served from the arena.
    served: u64,
    /// `empty` calls that allocated while an arena was active (nothing
    /// free, device mismatch, or over the bound).
    fell_through: u64,
    /// `empty` calls that fell through although a free buffer of the same
    /// size was retained, because that buffer lives on another stream (a
    /// subset of `fell_through`; see the "Streams" section on
    /// [`ScratchArena`]).
    cross_stream_fallthroughs: u64,
}

/// One retained device buffer.
struct ScratchEntry {
    /// Exact byte size (`empty` reuses only exact-size matches).
    bytes: usize,
    /// The stream the buffer was allocated on (`Handle::stream` at
    /// retention). Reuse serves only entries whose stream is the current
    /// thread's stream (see the "Streams" section on [`ScratchArena`]).
    stream: cubecl::stream_id::StreamId,
    /// Stored clone; handed-out clones alias this memory.
    handle: Handle,
}

/// Snapshot of [`ScratchArena::stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchStats {
    /// Retained buffers.
    pub buffers: usize,
    /// Retained bytes.
    pub bytes_held: u64,
    /// `empty` calls served from the arena since creation.
    pub served: u64,
    /// `empty` calls that allocated despite an active arena since creation.
    pub fell_through: u64,
    /// `empty` calls that fell through although a free buffer of the same
    /// size was retained, because that buffer lives on another stream (a
    /// subset of [`ScratchStats::fell_through`]).
    pub cross_stream_fallthroughs: u64,
}

impl ScratchArena {
    /// An empty arena retaining at most `max_bytes` bytes.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            shared: std::sync::Arc::new(ScratchShared {
                inner: std::sync::Mutex::new(ScratchInner {
                    device: None,
                    entries: Vec::new(),
                    bytes_held: 0,
                    served: 0,
                    fell_through: 0,
                    cross_stream_fallthroughs: 0,
                }),
                max_bytes,
            }),
        }
    }

    /// The retention bound in bytes.
    pub fn max_bytes(&self) -> usize {
        self.shared.max_bytes
    }

    /// How many buffers and bytes are retained, and how many `empty` calls
    /// were served from the arena or fell through to the allocator.
    pub fn stats(&self) -> ScratchStats {
        let inner = self.shared.inner.lock().expect("scratch arena is not poisoned");
        ScratchStats {
            buffers: inner.entries.len(),
            bytes_held: inner.bytes_held,
            served: inner.served,
            fell_through: inner.fell_through,
            cross_stream_fallthroughs: inner.cross_stream_fallthroughs,
        }
    }

    /// Retained buffers as `(bytes, count)` pairs, sorted by bytes.
    /// Test and profiling support for reconciling the retention against
    /// the shape-derived [`decode_scratch_bytes`] estimate.
    ///
    /// [`decode_scratch_bytes`]: crate::models::ms2::workspace::decode_scratch_bytes
    pub fn retained_sizes(&self) -> Vec<(usize, usize)> {
        let inner = self.shared.inner.lock().expect("scratch arena is not poisoned");
        let mut by_size: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for e in &inner.entries {
            *by_size.entry(e.bytes).or_insert(0) += 1;
        }
        by_size.into_iter().collect()
    }

    /// Drop every retained buffer, freeing the memory to the runtime.
    /// Counters are kept.
    pub fn clear(&self) {
        let mut inner = self.shared.inner.lock().expect("scratch arena is not poisoned");
        inner.entries.clear();
        inner.bytes_held = 0;
        inner.device = None;
    }

    /// Retained buffers still referenced from outside the arena: entries
    /// whose handle is not mutable, i.e. some tensor beyond the arena's own
    /// stored clone still holds the buffer.
    ///
    /// The owning generate calls assert this is zero once their (private)
    /// decoder state has dropped; see the confinement rule on
    /// [`ScratchArena`]. A scratch-backed tensor that escapes its call keeps
    /// its entry non-mutable forever, so the arena never reuses that buffer
    /// (the live-handle protection), and this count names it.
    pub fn externally_referenced_buffers(&self) -> usize {
        let inner = self.shared.inner.lock().expect("scratch arena is not poisoned");
        inner.entries.iter().filter(|e| !e.handle.can_mut()).count()
    }

    /// Debug-only confinement check for the end of an arena-scoped owning
    /// call (task F10 item A1): no retained buffer may still be referenced
    /// from outside the arena when the call that owns the arena returns.
    /// Every entry must be mutable through the arena's own stored clone
    /// alone; a live scratch-backed tensor outside would hold its buffer and
    /// fail the check, naming the buffer's byte size. Release builds skip
    /// the scan (the `can_mut` protection stays regardless).
    pub fn debug_assert_no_external_refs(&self) {
        if cfg!(debug_assertions) {
            let inner = self.shared.inner.lock().expect("scratch arena is not poisoned");
            for e in &inner.entries {
                debug_assert!(
                    e.handle.can_mut(),
                    "scratch arena leaked a {}-byte buffer past its owning call: a scratch-backed tensor is still referenced from outside",
                    e.bytes
                );
            }
        }
    }
}

impl Clone for ScratchArena {
    /// Cheap alias sharing the retained buffers (used by tests that hold
    /// the arena while a scope runs on the same thread).
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
        }
    }
}

thread_local! {
    /// The arena [`with_scratch`] activated on this thread, if any.
    static ACTIVE_SCRATCH: std::cell::RefCell<Option<std::sync::Arc<ScratchShared>>> =
        const { std::cell::RefCell::new(None) };
    /// Whether [`ACTIVE_SCRATCH`] holds an arena on this thread: the single
    /// cheap check behind [`scratch_scope_active`], kept in lockstep with
    /// [`ACTIVE_SCRATCH`] by [`with_scratch`] (set on install, cleared on
    /// scope drop, including on unwind).
    static SCRATCH_ACTIVE_FLAG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with `arena` activated for the current thread: [`Tensor::empty`]
/// and [`IdTensor::empty`] calls inside `f` on the arena's device reuse
/// retained buffers of the same byte size instead of allocating.
///
/// Production activates the arena only inside decode loops whose decoder
/// state is private to the call (see the confinement rule on
/// [`ScratchArena`]): the staged public API never calls this.
///
/// Re-entrancy: nesting the *same* arena is a no-op (the outer scope stays
/// active); nesting a *different* arena is refused panic-free — the outer
/// scope stays active and `f` runs under it. Either way `f` runs exactly
/// once and the previous activation (if any) is restored afterwards, even
/// on unwind.
///
/// [`Tensor::empty`]: crate::tensor::Tensor::empty
/// [`IdTensor::empty`]: crate::tensor::ops::index::IdTensor::empty
pub fn with_scratch<O>(arena: &ScratchArena, f: impl FnOnce() -> O) -> O {
    let same = ACTIVE_SCRATCH.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(|current| std::sync::Arc::ptr_eq(current, &arena.shared))
    });
    if same || is_other_arena_active(&arena.shared) {
        // Same arena: already active. Different arena: refuse to switch and
        // keep the outer scope, so buffers are never retained into (or
        // served from) the wrong bucket's arena.
        return f();
    }
    ACTIVE_SCRATCH.with(|active| {
        *active.borrow_mut() = Some(arena.shared.clone());
    });
    SCRATCH_ACTIVE_FLAG.with(|flag| flag.set(true));
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE_SCRATCH.with(|active| {
                *active.borrow_mut() = None;
            });
            SCRATCH_ACTIVE_FLAG.with(|flag| flag.set(false));
        }
    }
    let _restore = Restore;
    f()
}

/// Whether a *different* arena than `candidate` is active on this thread.
fn is_other_arena_active(candidate: &std::sync::Arc<ScratchShared>) -> bool {
    ACTIVE_SCRATCH.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(|current| !std::sync::Arc::ptr_eq(current, candidate))
    })
}

/// A retained buffer of exactly `bytes` on `device_id` from this thread's
/// active arena, if any. `None` outside a scope, on a device mismatch, when
/// nothing of that size is free, or when the free buffers of that size live
/// on another stream (see the "Streams" section on [`ScratchArena`]). Hits
/// and misses are counted in the arena's [`ScratchStats`].
pub(crate) fn scratch_reuse(device_id: usize, bytes: usize) -> Option<Handle> {
    ACTIVE_SCRATCH.with(|active| {
        active.borrow().as_ref()?.reuse(device_id, bytes)
    })
}

/// Retain `handle` (`bytes` on `device_id`) into this thread's active arena,
/// if any. Beyond the arena's bound the buffer is simply not retained.
pub(crate) fn scratch_retain(device_id: usize, bytes: usize, handle: &Handle) {
    ACTIVE_SCRATCH.with(|active| {
        let Some(arena) = active.borrow().as_ref().cloned() else {
            return;
        };
        arena.retain(arena.max_bytes, device_id, bytes, handle);
    });
}

impl ScratchShared {
    /// Hand out a clone of a retained buffer of exactly `bytes` on
    /// `device_id`, or `None` (counted as fell through).
    ///
    /// Only a buffer allocated on the current thread's stream is served: a
    /// free same-size buffer that lives on another stream falls through to
    /// a fresh allocation instead (counted in `cross_stream_fallthroughs`
    /// as well as `fell_through`), so a still-queued write on that stream
    /// can never overwrite the new tensor. See the "Streams" section on
    /// [`ScratchArena`].
    fn reuse(&self, device_id: usize, bytes: usize) -> Option<Handle> {
        let current = cubecl::stream_id::StreamId::current();
        let mut inner = self.inner.lock().expect("scratch arena is not poisoned");
        if inner.device != Some(device_id) {
            inner.fell_through += 1;
            return None;
        }
        let found = inner
            .entries
            .iter()
            .find(|e| e.bytes == bytes && e.stream == current && e.handle.can_mut())
            .map(|e| e.handle.clone());
        match found {
            Some(handle) => {
                inner.served += 1;
                Some(handle)
            }
            None => {
                if inner
                    .entries
                    .iter()
                    .any(|e| e.bytes == bytes && e.stream != current && e.handle.can_mut())
                {
                    inner.cross_stream_fallthroughs += 1;
                }
                inner.fell_through += 1;
                None
            }
        }
    }

    /// Retain a clone of `handle` for later reuse, unless bound to another
    /// device, full past `max_bytes`, or already holding
    /// [`SCRATCH_PER_SIZE_LIMIT`] buffers of this size.
    ///
    /// The entry records `handle`'s allocation stream (`Handle::stream`),
    /// which is what binds later reuse to the allocating stream (see the
    /// "Streams" section on [`ScratchArena`]).
    fn retain(&self, max_bytes: usize, device_id: usize, bytes: usize, handle: &Handle) {
        let mut inner = self.inner.lock().expect("scratch arena is not poisoned");
        match inner.device {
            None => inner.device = Some(device_id),
            Some(bound) if bound != device_id => return,
            _ => {}
        }
        if inner.bytes_held.saturating_add(bytes as u64) > max_bytes as u64 {
            return;
        }
        if inner.entries.iter().filter(|e| e.bytes == bytes).count() >= SCRATCH_PER_SIZE_LIMIT {
            return;
        }
        inner.entries.push(ScratchEntry {
            bytes,
            stream: handle.stream,
            handle: handle.clone(),
        });
        inner.bytes_held += bytes as u64;
    }
}

/// Whether the decode loop runs inside a scratch scope: `0` off, `1` on,
/// `-1` not yet read.
static SCRATCH_ON: core::sync::atomic::AtomicI8 = core::sync::atomic::AtomicI8::new(-1);

/// Whether the decode loop runs inside a scratch scope. On by default;
/// `MAMBA3_MS2_SCRATCH=0` (or [`set_scratch_enabled`] with `false`)
/// restores the allocate-every-output behaviour exactly.
pub fn scratch_enabled() -> bool {
    use core::sync::atomic::Ordering;
    match SCRATCH_ON.load(Ordering::Relaxed) {
        -1 => {
            let on = std::env::var("MAMBA3_MS2_SCRATCH").as_deref() != Ok("0");
            SCRATCH_ON.store(on as i8, Ordering::Relaxed);
            on
        }
        flag => flag == 1,
    }
}

/// Choose whether the decode loop runs inside a scratch scope.
///
/// Off means every op output allocates through the runtime's pool, which is
/// what the crate did before the arena existed. Results are identical
/// either way — a recycled buffer holds stale bytes but `empty` promises
/// uninitialised memory — so this changes only cost, and exists so the two
/// can be compared inside one process rather than across runs that differ
/// by more than the change under test.
pub fn set_scratch_enabled(on: bool) {
    SCRATCH_ON.store(on as i8, core::sync::atomic::Ordering::Relaxed);
}
