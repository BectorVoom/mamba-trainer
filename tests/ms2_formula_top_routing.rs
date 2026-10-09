//! Chunked top-F routing proofs (task F9 item B2) and workspace-scratch
//! reuse allocation counts (task F9 item B1).
//!
//! This binary OWNS the process-global selection switch
//! ([`ms2::set_formula_top_chunked`]) and the process-global launch counter
//! ([`launch_count`]): every test holds [`ROUTING_SERIAL`] first, so no two
//! tests in this process race on either, and every override goes through
//! [`RoutingGuard`], which restores the previous setting on drop — including
//! on panic. Cargo gives each integration test file its own process, so no
//! other binary's switch use can interfere either. The
//! `MAMBA3_MS2_TOP_CHUNKED` environment variable is never set in this
//! process (each test removes it on entry, before the first routed call, so
//! the once-per-process environment sample of task F9 item B3 always sees
//! it unset).
//!
//! Allocation counting uses the binary-local [`#[global_allocator]`]: host
//! allocations around the measured windows only (warm-up stays outside).

#![cfg(feature = "backend")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use mamba3::backend::{Device, check_launches, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

/// Serialises every test in this binary: the switch, the launch counter and
/// the allocation window are all process-global.
static ROUTING_SERIAL: Mutex<()> = Mutex::new(());

/// `alloc` calls inside the measured window.
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
/// Bytes handed out inside the measured window.
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Whether the allocation window is open (warm-up must not pollute it).
static COUNTING: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Scoped routing override (task F9 item B2): holds [`ROUTING_SERIAL`] for
/// its whole life (serialising this binary's tests), applies the requested
/// override (or clears it), and restores the PREVIOUS override — and the
/// previous environment-variable state — on drop, including on panic.
struct RoutingGuard {
    /// Held for the guard's life: serialises this binary's tests.
    _serial: MutexGuard<'static, ()>,
    /// Override before this guard ran (`None` = unset).
    prev_override: Option<bool>,
    /// Environment variable before this guard ran (`None` = unset).
    prev_env: Option<String>,
}

impl RoutingGuard {
    /// Hold the serial mutex, snapshot the previous switch state, remove the
    /// environment variable (this binary never sets it, so the B3
    /// once-per-process sample always sees it unset), then force the switch
    /// off (`Some(false)`), on (`Some(true)`) or back to unset (`None`).
    fn scoped(setting: Option<bool>) -> Self {
        let serial = ROUTING_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let prev_override = ms2::formula_top_chunked_override();
        let prev_env = std::env::var("MAMBA3_MS2_TOP_CHUNKED").ok();
        // SAFETY: ROUTING_SERIAL is already held (acquired above for the
        // guard's whole life), serialising this binary's tests, and no other
        // thread in this process touches the environment — so no concurrent
        // `getenv` can race this mutation.
        unsafe { std::env::remove_var("MAMBA3_MS2_TOP_CHUNKED") };
        match setting {
            Some(on) => ms2::set_formula_top_chunked(on),
            None => ms2::clear_formula_top_chunked_override(),
        }
        RoutingGuard {
            _serial: serial,
            prev_override,
            prev_env,
        }
    }
}

impl Drop for RoutingGuard {
    /// Restore the previous override and environment-variable state (runs on
    /// panic too, so one failing test cannot leak the switch into the next).
    fn drop(&mut self) {
        match self.prev_override {
            Some(on) => ms2::set_formula_top_chunked(on),
            None => ms2::clear_formula_top_chunked_override(),
        }
        if let Some(v) = self.prev_env.take() {
            // SAFETY: ROUTING_SERIAL is still held (released after drop),
            // serialising this binary's tests; no other thread in this
            // process touches the environment.
            unsafe { std::env::set_var("MAMBA3_MS2_TOP_CHUNKED", v) };
        }
    }
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Whether this runtime has planes (the GPU route) or not (the CPU route,
/// where the default keeps the old kernel).
fn has_planes(device: &Device<R>) -> bool {
    device.client().properties().hardware.plane_size_max > 1
}

/// Deterministic `(log_prob, cand)` fixture at `(batch, m)`: quantised ties
/// plus invalid scores, so old and chunked forms must agree bit for bit.
fn fixture(batch: usize, m: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut rng = Rng::seeded(seed);
    let u = rng.uniform_vec(batch * m * 2, 0.0, 1.0);
    let mut log_prob = vec![0.0f32; batch * m];
    let mut cand = vec![0u32; batch * m * 13];
    for b in 0..batch {
        for slot in 0..m {
            let r0 = u[(b * m + slot) * 2];
            let r1 = u[(b * m + slot) * 2 + 1];
            let flag = (r0 * 3.0) as u32;
            cand[(b * m + slot) * 13 + 11] = flag;
            cand[(b * m + slot) * 13 + 12] = 100 + slot as u32;
            log_prob[b * m + slot] = ((r1 * 8.0 - 4.0) * 2.0).round() / 2.0;
        }
    }
    (log_prob, cand)
}

/// Run [`ms2::formula_top`] once and return the selection-launch delta plus
/// the read-back outputs.
fn routed_call(
    device: &Device<R>,
    log_prob: &[f32],
    cand: &[u32],
    batch: usize,
    m: usize,
    f: usize,
) -> (usize, Vec<u32>, Vec<f32>, Vec<u32>) {
    let lp_t = Tensor::<R, E>::from_f32(log_prob, vec![batch, m], device).unwrap();
    let cand_t = IdTensor::from_slice(cand, vec![batch, m, 13], device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, device).unwrap();
    reset_launch_count();
    ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    device.synchronize();
    let launches = launch_count();
    check_launches(device).unwrap();
    (
        launches,
        out.top.try_to_vec().unwrap(),
        out.top_log_prob.try_to_f32().unwrap(),
        out.top_count.try_to_vec().unwrap(),
    )
}

/// Run [`ms2::formula_top_chunked`] directly (bypasses routing) and return
/// the launch delta plus the read-back outputs.
fn direct_chunked_call(
    device: &Device<R>,
    log_prob: &[f32],
    cand: &[u32],
    batch: usize,
    m: usize,
    f: usize,
) -> (usize, Vec<u32>, Vec<f32>, Vec<u32>) {
    let lp_t = Tensor::<R, E>::from_f32(log_prob, vec![batch, m], device).unwrap();
    let cand_t = IdTensor::from_slice(cand, vec![batch, m, 13], device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, device).unwrap();
    reset_launch_count();
    ms2::formula_top_chunked(&lp_t, &cand_t, &out).unwrap();
    device.synchronize();
    let launches = launch_count();
    check_launches(device).unwrap();
    (
        launches,
        out.top.try_to_vec().unwrap(),
        out.top_log_prob.try_to_f32().unwrap(),
        out.top_count.try_to_vec().unwrap(),
    )
}

/// Task F9 item B2: forced off routes to the old kernel — exactly 1 selection
/// launch at `M = 512` for every `F` in {1, 4, 8}. (A broken routing that
/// always ran chunked would launch `2F`, failing here.)
#[test]
fn routing_forced_off_is_one_launch() {
    let _guard = RoutingGuard::scoped(Some(false));
    let device = dev();
    let (batch, m) = (2usize, 512usize);
    for &f in &[1usize, 4, 8] {
        let (log_prob, cand) = fixture(batch, m, 7000 + f as u64);
        let (launches, top, lp, count) = routed_call(&device, &log_prob, &cand, batch, m, f);
        assert_eq!(
            launches, 1,
            "forced off at M = {m} F = {f}: the old kernel launches exactly once, got {launches}"
        );
        // The routed-off outputs equal the direct chunked outputs (the two
        // forms are bit-identical on every input).
        let (_, d_top, d_lp, d_count) = direct_chunked_call(&device, &log_prob, &cand, batch, m, f);
        assert_eq!(
            top, d_top,
            "forced off at F = {f}: top differs from chunked"
        );
        assert_eq!(
            lp, d_lp,
            "forced off at F = {f}: log-probs differ from chunked"
        );
        assert_eq!(
            count, d_count,
            "forced off at F = {f}: counts differ from chunked"
        );
        println!("ROUTING forced off M = {m} F = {f}: {launches} launch");
    }
}

/// Task F9 item B2: forced on routes to the chunked form — exactly `2F`
/// selection launches at `M = 512` for every `F` in {1, 4, 8}. (A broken
/// routing that always ran the old kernel would launch 1, failing here.)
#[test]
fn routing_forced_on_is_two_f_launches() {
    let _guard = RoutingGuard::scoped(Some(true));
    let device = dev();
    let (batch, m) = (2usize, 512usize);
    for &f in &[1usize, 4, 8] {
        let (log_prob, cand) = fixture(batch, m, 7000 + f as u64);
        let (launches, top, lp, count) = routed_call(&device, &log_prob, &cand, batch, m, f);
        assert_eq!(
            launches,
            2 * f,
            "forced on at M = {m} F = {f}: chunked launches 2F, got {launches}"
        );
        let (d_launches, d_top, d_lp, d_count) =
            direct_chunked_call(&device, &log_prob, &cand, batch, m, f);
        assert_eq!(
            d_launches,
            2 * f,
            "direct chunked at M = {m} F = {f}: must launch 2F, got {d_launches}"
        );
        assert_eq!(
            top, d_top,
            "forced on at F = {f}: routed top differs from direct chunked"
        );
        assert_eq!(lp, d_lp, "forced on at F = {f}: routed log-probs differ");
        assert_eq!(count, d_count, "forced on at F = {f}: routed counts differ");
        println!("ROUTING forced on M = {m} F = {f}: {launches} launches");
    }
}

/// Task F9 item B2: with the switch unset and the environment unset, the
/// default for the runtime applies — the old kernel (1 launch) on the CPU
/// runtime (no planes), the chunked form (`2F`) where planes exist.
#[test]
fn routing_default_matches_runtime() {
    let _guard = RoutingGuard::scoped(None);
    let device = dev();
    let (batch, m, f) = (2usize, 512usize, 4usize);
    let (log_prob, cand) = fixture(batch, m, 4242);
    let (launches, _, _, _) = routed_call(&device, &log_prob, &cand, batch, m, f);
    if has_planes(&device) {
        assert_eq!(
            launches,
            2 * f,
            "default on a plane device at M = {m} F = {f}: chunked launches 2F, got {launches}"
        );
        println!("ROUTING default (planes): {launches} launches");
    } else {
        assert_eq!(
            launches, 1,
            "default on the CPU runtime at M = {m} F = {f}: the old kernel launches once, got {launches}"
        );
        println!("ROUTING default (cpu, no planes): {launches} launch");
    }
}

/// Task F9 item B1: the chunk scratch lives in the formula workspace —
/// allocated once per bucket and reused per call. After warm-up, ten routed
/// (forced-on, production-chunk) calls over one shared workspace allocate
/// exactly the per-call scratch less than ten identical calls whose
/// workspace scratch shapes cannot match (forcing the pre-B1 per-call
/// allocation on the SAME chunk and work); the outputs stay bit-identical
/// either way.
#[test]
fn chunk_scratch_reused_not_reallocated() {
    let _guard = RoutingGuard::scoped(Some(true));
    let device = dev();
    let (batch, m, f) = (2usize, 512usize, 4usize);
    let (log_prob, cand) = fixture(batch, m, 9900);
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![batch, m], &device).unwrap();
    let cand_t = IdTensor::from_slice(&cand, vec![batch, m, 13], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, &device).unwrap();
    // Same buffers but sabotaged scratch shapes: `check_formula_top_shapes`
    // never looks at the scratch, so the call is valid — yet the shapes
    // cannot match `[B, nchunks]`, forcing the per-call scratch allocation
    // (the pre-B1 pattern) on identical work.
    let mut sabotaged = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, &device).unwrap();
    sabotaged.chunk_score = Tensor::<R, E>::empty(vec![batch, 1], &device);
    sabotaged.chunk_slot = IdTensor::empty(vec![batch, 1], &device);
    // Warm-up outside every window (lazy client/device allocation first).
    for _ in 0..5 {
        ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
        ms2::formula_top(&lp_t, &cand_t, &sabotaged).unwrap();
    }
    device.synchronize();
    let reference = (
        out.top.try_to_vec().unwrap(),
        out.top_log_prob.try_to_f32().unwrap(),
        out.top_count.try_to_vec().unwrap(),
    );

    // Reused workspace: ten routed calls over the same buffers.
    COUNTING.store(1, Ordering::Relaxed);
    for _ in 0..10 {
        ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    }
    device.synchronize();
    let reused_calls = ALLOC_CALLS.load(Ordering::Relaxed);
    let reused_bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    COUNTING.store(0, Ordering::Relaxed);
    let reused = (
        out.top.try_to_vec().unwrap(),
        out.top_log_prob.try_to_f32().unwrap(),
        out.top_count.try_to_vec().unwrap(),
    );
    assert_eq!(
        reused, reference,
        "reused-workspace calls must stay bit-identical"
    );
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);

    // Per-call scratch on identical work: ten routed calls over the
    // sabotaged buffers.
    COUNTING.store(1, Ordering::Relaxed);
    for _ in 0..10 {
        ms2::formula_top(&lp_t, &cand_t, &sabotaged).unwrap();
    }
    device.synchronize();
    let per_call_calls = ALLOC_CALLS.load(Ordering::Relaxed);
    let per_call_bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    COUNTING.store(0, Ordering::Relaxed);
    let per_call = (
        sabotaged.top.try_to_vec().unwrap(),
        sabotaged.top_log_prob.try_to_f32().unwrap(),
        sabotaged.top_count.try_to_vec().unwrap(),
    );
    assert_eq!(
        per_call, reference,
        "per-call-scratch calls must agree bit for bit"
    );
    println!(
        "ALLOC chunk scratch: reused-workspace 10 calls = {reused_calls} allocs / {reused_bytes} bytes; per-call-scratch 10 calls = {per_call_calls} allocs / {per_call_bytes} bytes"
    );
    // Identical work, so the ONLY allocation difference is the two scratch
    // buffers per call (at least 2 allocation calls each): the reused path
    // must save at least that. (A revert to per-call scratch makes both
    // paths allocate and fails here.)
    assert!(
        per_call_calls >= reused_calls + 20,
        "reused workspace ({reused_calls} allocs) must save at least the two per-call scratch buffers per call vs per-call scratch ({per_call_calls} allocs)"
    );
}
