//! P2.6 fallback tests: the synchronised wall-clock span in
//! `mamba3::backend::sync_span`.
//!
//! Counter discipline in this binary: two tests read process-wide counters
//! (the generate-span test reads launches; the device-equivalence test reads
//! launches, runtime reads and allocations), and every test holds the SERIAL
//! lock for its whole body, so each reset-plus-measure window is atomic
//! against the others. Cargo gives each integration test file its own
//! process, so no test outside this file can perturb the numbers. The
//! neighbours touch no process-global counter: they count body invocations
//! through a host cell only.
//!
//! Synchronisation regression coverage (R2-D), what remains unverified: the
//! span-counter test below pins that `Device::try_synchronize` is CALLED
//! (its increment lives on the actual drain operation, so deleting the call
//! removes its count), but no test can verify that the call DRAINS the
//! runtime. A mutant that keeps the increment while replacing the
//! `client.sync()` drain with `Ok(())` passes every test here: on the CPU
//! backend `flush` and `sync` execute the queued tasks identically (same
//! `StreamErrorMode`), so no execution-time failure surfaces at `sync` that
//! `check_launches` would not already surface, and neither backend exposes a
//! pending-work indicator through the crate's API that a test could poll
//! without itself synchronising (every device read synchronises, masking the
//! missing completion). Timing-based detection would be flaky, so no such
//! test is faked here. Closing this gap needs runtime support (e.g. a
//! non-synchronising pending-task query from cubecl) or a deterministic
//! sync-only failure channel.

#![cfg(feature = "backend")]

use std::cell::Cell;
use std::rc::Rc;

use mamba3::backend::{
    Device, Profiler, SYNC_WALL_TIMER, allocation_calls, launch_count, profile_session,
    reset_launch_count, reset_transfer_counters, runtime_read_count, synchronize_count, sync_span,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::contract::{Control, FormulaSource, GenerationConfig, GenerationMode, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION};
use mamba3::models::ms2::contract::{ModelConfig, SpectrumBatch};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::workspace::{Ms2Capabilities, Ms2MemoryEstimate, TimingMethod};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Process-wide serialisation for this binary's tests.
///
/// Two tests below read process-wide counters, and the harness tests drive
/// kernels on the same physical device. Cargo runs a binary's tests on
/// threads of one process, so every test holds this lock for its whole body;
/// contention only ever serialises, never fails.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().expect("test serialisation is not poisoned")
}

/// Tiny model (`d = 16`, 2 heads, state 8, `N = 16`, `n_raw = 64`), as in the
/// generation tests: fast enough to run whole `generate` calls here.
fn tiny_config() -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = 16;
    m.n_peaks = 16;
    m.encoder_blocks = 1;
    m.decoder_blocks = 1;
    m.attention_heads = 2;
    m.encoder.d_model = 16;
    m.encoder.n_heads = 2;
    m.encoder.head_dim = 8;
    m.encoder.d_state = 8;
    m.encoder.n_groups = 2;
    m.decoder.d_model = 16;
    m.decoder.n_heads = 2;
    m.decoder.head_dim = 8;
    m.decoder.d_state = 8;
    m.decoder.n_groups = 2;
    m
}

fn tiny_generation() -> GenerationConfig {
    GenerationConfig {
        schema_version: SCHEMA_VERSION,
        trajectories: 4,
        formulas: 2,
        seed: 7,
        temperature: 1.0,
        max_steps: 22,
        max_device_bytes: 2 * 1024 * 1024 * 1024,
        formula_rows_visited_max: u32::MAX,
        formula_rows_scored_max: 4096,
        mode: GenerationMode::Sampling,
        oracle_formula: false,
        control: Control::None,
        formula_source: FormulaSource::Table,
        formula_window: 32,
        enum_lanes_max: 262_144,
        enum_lane_visits_max: 65_536,
        enum_dispatch_visits_max: 4_000_000,
        allocation: mamba3::models::ms2::contract::AllocationMode::RoundRobin,
        identity: mamba3::models::ms2::contract::IdentityMode::TraceOnly,
        identity_work_max: 4096,
        returned: 0,
        evidence: false,
        ion_request_work_max: 268435456,
    }
}

/// Spectra with explicit precursors and seeded peaks below them.
fn make_spectra(
    spectrum_ids: &[u64],
    precursors: &[u32],
    n_raw: usize,
    peak_counts: &[u32],
    seed: u64,
) -> SpectrumBatch {
    let b = spectrum_ids.len();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for (bi, &count) in peak_counts.iter().enumerate() {
        let count = count as usize;
        peak_count[bi] = count as u32;
        raw_peak_count[bi] = count as u32;
        let precursor = precursors[bi];
        for i in 0..count {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursor - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            let u = rng.uniform_vec(1, 0.0, 1.0)[0];
            intensity[bi * n_raw + i] = 0.5 + 2.0 * u;
        }
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: spectrum_ids.to_vec(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursors.to_vec(),
        precursor_uncertainty_udalton: vec![50; b],
        adduct: vec![1; b],
        polarity: vec![1; b],
        collision_energy_ev: vec![30.0; b],
        collision_energy_known: vec![1; b],
        energy_count: vec![1; b],
        fragment_tolerance_ppm_tenths: vec![0; b],
        precursor_tolerance_ppm_tenths: vec![0; b],
        instrument_class: vec![0; b],
    }
}

#[test]
fn sync_span_runs_body_once_with_a_non_negative_wall_time() {
    let _serial = serial();
    // The span runs its body exactly once, returns its value, and reports a
    // non-negative synchronised wall time. The body below is `!Send` (it
    // holds an `Rc`), which is exactly why the span is wall clock rather
    // than `client.profile`: the profile closure would have to be `Send`.
    let device = dev();
    let calls = Rc::new(Cell::new(0usize));
    let probe = Rc::new(42u32);
    let inner = Rc::clone(&calls);
    let (out, wall) = sync_span(&device, || {
        inner.set(inner.get() + 1);
        // Touch the `Rc` payload so the closure really captures it.
        *probe + inner.get() as u32
    });
    assert_eq!(calls.get(), 1, "the body runs exactly once");
    assert_eq!(out, 43, "the span returns the body's value");
    // A wall duration is non-negative by construction; pin it so a future
    // clock change (e.g. a non-monotonic source) fails loudly.
    assert!(wall.as_secs_f64() >= 0.0, "wall time is non-negative: {wall:?}");
}

#[test]
fn sync_span_uses_the_probed_timing_method() {
    let _serial = serial();
    // The probe reports the runtime's profiling *capability* (never
    // `Unavailable` here: both backends answer the trivial closure). On the
    // CPU runtime that capability is `SystemTime`. The span's own wall time
    // is a host `Instant` around the body plus the synchronisation, so its
    // timer label is always `SynchronizedHostWallClock` — even on a
    // device-timestamp runtime. The two must not be conflated: capability in
    // one field, clock in the other.
    assert_eq!(
        SYNC_WALL_TIMER, "SynchronizedHostWallClock",
        "the host wall-clock label is fixed"
    );
    let device = dev();
    let caps = Ms2Capabilities::probe(&device);
    assert!(
        matches!(
            caps.timing,
            TimingMethod::SystemTime | TimingMethod::DeviceTimestamps
        ),
        "the timing probe reports a real capability, got {:?}",
        caps.timing
    );
    if device.name() == "cpu" {
        assert_eq!(
            caps.timing,
            TimingMethod::SystemTime,
            "the CPU runtime times through system time"
        );
    }
    println!("backend {} timing {:?}", device.name(), caps.timing);
}

#[test]
fn span_around_generate_changes_neither_launches_nor_results() {
    let _serial = serial();
    // A span around a real `generate` call observes exactly that call's
    // launches (the synchronisation launches nothing) and the candidate
    // batch is identical to the same call without a span (same seed).
    // Launch deltas are used throughout: the counter readers in this binary
    // are this test and the device-equivalence test below, serialised by the
    // file lock so each reset-plus-measure window is atomic.
    let device = dev();
    let mut cfg = tiny_config();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0], [3, 7, 1, 2, 0, 0, 0, 0, 0, 0]];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_generation();
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = make_spectra(&[501, 502], &precursors, 64, &[10, 12], 31);
    // Two warm-up calls: compilation and tuning settle before measurement.
    for _ in 0..2 {
        let out = model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
        out.validate().unwrap();
    }
    device.synchronize();
    reset_launch_count();
    let l0 = launch_count();
    let plain = model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
    plain.validate().unwrap();
    let plain_launches = launch_count() - l0;
    reset_launch_count();
    let l1 = launch_count();
    let (spanned, wall) = sync_span(&device, || {
        model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap()
    });
    spanned.validate().unwrap();
    let spanned_launches = launch_count() - l1;
    assert_eq!(
        spanned_launches, plain_launches,
        "a span adds no launch: {spanned_launches} vs {plain_launches}"
    );
    assert_eq!(
        spanned, plain,
        "the spanned call returns the identical candidate batch"
    );
    println!("generate launches {plain_launches}, span wall {wall:?}");
}

#[test]
fn nested_spans_are_supported() {
    let _serial = serial();
    // Nested spans are supported: each span synchronises in turn and reports
    // its own non-negative wall time. (CubeCL 0.10's `client.profile` would
    // instead re-enter `exclusive` from the device thread; wall spans have
    // no such protocol, so nesting is just two syncs.)
    let device = dev();
    let (inner_out, outer_wall) = sync_span(&device, || {
        sync_span(&device, || {
            let started = std::time::Instant::now();
            let out = model_free_work();
            (out, started.elapsed())
        })
    });
    let ((inner_ms, _extra), inner_wall) = inner_out;
    assert_eq!(inner_ms, 7, "the inner body returns its value");
    assert!(inner_wall.as_secs_f64() >= 0.0, "inner wall is non-negative");
    assert!(outer_wall >= inner_wall, "the outer span contains the inner one");
}

fn model_free_work() -> u32 {
    7
}

#[test]
fn sync_span_completes_queued_device_work_without_a_read_inside() {
    let _serial = serial();
    // Eight kernel launches are queued with no device read inside the span
    // body; when the span returns, the device's pending work is complete
    // (the trailing synchronisation drained it) and the un-read result is
    // correct. Counter discipline, not counters, is what this test observes:
    // the generate test above is this binary's other counter reader, and the
    // file lock keeps the two apart.
    //
    // Honestly stated for the CPU runtime: launches there complete before
    // any host observation anyway (every read blocks), so no host-observable
    // on this runtime can distinguish "synced at the span boundary" from
    // "synced by the later read" — removing the trailing synchronisation
    // would leave this value-based test passing here. What the test pins is
    // the contract a silent desync would break elsewhere: the body runs
    // exactly once, the span adds no work of its own, and the queued work is
    // complete when the span returns (on an asynchronous runtime the values
    // below would be undefined without the boundary sync). That removing the
    // boundary sync is caught at all is pinned by the synchronise-counter
    // test below (`sync_span_advances_the_synchronise_counter_by_one`), which
    // fails when the span stops synchronising. A deferred launch failure
    // would likewise surface at this boundary through `try_synchronize`.
    let device = dev();
    let a = Tensor::<R, f32>::from_data(&vec![1.0; 64], vec![64], &device).unwrap();
    let b = Tensor::<R, f32>::from_data(&vec![2.0; 64], vec![64], &device).unwrap();
    let (out, wall) = sync_span(&device, || {
        let mut t = mamba3::tensor::ops::elemwise::add(&a, &b).unwrap();
        for _ in 1..8 {
            t = mamba3::tensor::ops::elemwise::add(&t, &a).unwrap();
        }
        t
    });
    assert!(wall.as_secs_f64() >= 0.0, "wall time is non-negative: {wall:?}");
    device.try_synchronize().expect("no deferred launch failure is parked");
    let got = out.to_f32();
    assert_eq!(got.len(), 64, "the queued work produced a full result");
    for (i, v) in got.iter().enumerate() {
        assert_eq!(*v, 10.0, "element {i}: 1 + 2 + 7 more ones");
    }
}

// ---------------------------------------------------------------------------
// Runner-local profiling harness (`mamba3::backend::profile_session`).
//
// Counter discipline: the device-equivalence test at the end of this section
// reads the process-wide counters inside its reset-plus-measure window while
// holding the file lock, like the generate-span test above.
// ---------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Session state holding host-side call counters: the span bodies are `fn`
/// pointers, so they observe calls through the state, never through captures.
struct CounterSession {
    spans: Arc<AtomicUsize>,
    value: u32,
}

fn count_span_a(s: &mut CounterSession) -> u32 {
    s.spans.fetch_add(1, Ordering::SeqCst);
    s.value + 1
}

fn count_span_b(s: &mut CounterSession) -> u32 {
    s.spans.fetch_add(1, Ordering::SeqCst);
    s.value + 2
}

#[test]
fn harness_runs_build_once_and_each_span_once_returning_values() {
    let _serial = serial();
    // The harness constructs the session state once on the runner thread and
    // runs each span body once, returning each span's value with a timing.
    let device = dev();
    let builds = Arc::new(AtomicUsize::new(0));
    let spans = Arc::new(AtomicUsize::new(0));
    let builds_c = Arc::clone(&builds);
    let spans_c = Arc::clone(&spans);
    let out = profile_session(
        &device,
        move || {
            builds_c.fetch_add(1, Ordering::SeqCst);
            CounterSession {
                spans: Arc::clone(&spans_c),
                value: 40,
            }
        },
        |p: &Profiler<R>| {
            let (a, ta) = p.span("a", count_span_a).expect("span a runs");
            let (b, tb) = p.span("b", count_span_b).expect("span b runs");
            assert!(ta.ms >= 0.0 && tb.ms >= 0.0, "span timings are non-negative");
            (a, b, ta.timer, tb.timer)
        },
    )
    .expect("the session runs");
    assert_eq!(builds.load(Ordering::SeqCst), 1, "build runs exactly once");
    assert_eq!(spans.load(Ordering::SeqCst), 2, "each span runs exactly once");
    assert_eq!((out.0, out.1), (41, 42), "each span's value is returned");
    for timer in [out.2, out.3] {
        assert!(
            matches!(timer, "DeviceTimestamps" | "SystemTime"),
            "a span reports the ProfileDuration's own timing method, got {timer}"
        );
    }
}

/// Session state proving `!Send` payloads work: the `Rc` never crosses a
/// thread (built, used and dropped on the runner), and the drop is observed
/// with the dropping thread's id.
struct DropSession {
    #[allow(dead_code)]
    rc: Rc<u32>,
    dropped: Arc<AtomicBool>,
    dropped_on: Arc<std::sync::Mutex<Option<std::thread::ThreadId>>>,
}

impl Drop for DropSession {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
        *self.dropped_on.lock().unwrap() = Some(std::thread::current().id());
    }
}

fn drop_span_ok(_: &mut DropSession) -> Result<u32, String> {
    Err("staged failure".to_string())
}

#[test]
fn harness_span_error_is_a_value_and_state_drops_on_the_runner() {
    let _serial = serial();
    // An error inside a span body is returned as `Err` (not a panic, not a
    // hang), and the session state is still dropped on the runner thread —
    // never on the caller.
    let device = dev();
    let caller = std::thread::current().id();
    let dropped = Arc::new(AtomicBool::new(false));
    let dropped_on = Arc::new(std::sync::Mutex::new(None));
    let dropped_c = Arc::clone(&dropped);
    let dropped_on_c = Arc::clone(&dropped_on);
    let out = profile_session(
        &device,
        move || DropSession {
            rc: Rc::new(7),
            dropped: Arc::clone(&dropped_c),
            dropped_on: Arc::clone(&dropped_on_c),
        },
        |p: &Profiler<R>| p.span("failing", drop_span_ok).expect("span runs").0,
    )
    .expect("the session runs");
    assert_eq!(out, Err("staged failure".to_string()), "the span error is a value");
    assert!(dropped.load(Ordering::SeqCst), "the session state was dropped");
    let tid = dropped_on.lock().unwrap().expect("the drop recorded its thread");
    assert_ne!(tid, caller, "the state drops on the runner thread, not the caller");
}

struct OuterSession {
    count: Arc<AtomicUsize>,
}

struct InnerSession {
    count: Arc<AtomicUsize>,
}

fn outer_span(s: &mut OuterSession) -> u32 {
    s.count.fetch_add(1, Ordering::SeqCst) as u32 + 100
}

fn inner_span(s: &mut InnerSession) -> u32 {
    s.count.fetch_add(1, Ordering::SeqCst) as u32 + 200
}

#[test]
fn harness_supports_nested_sessions_and_nested_profiles() {
    let _serial = serial();
    // Nesting that the harness supports: a session inside another session's
    // `run` (the inner session saves and restores the outer state, even
    // across state types), sequential spans around it, and nested
    // `client.profile` calls with pure-`Send` bodies (which execute inline
    // because the caller already is the runner, channel.rs:151).
    //
    // Directly nested `Profiler::span` calls on the *same* session are not
    // supported: a span takes the state out of the runner-thread slot while
    // its body runs, so an inner span would find the slot taken. Use
    // sequential spans, which is what the profile driver needs.
    let device = dev();
    let outer = Arc::new(AtomicUsize::new(0));
    let inner = Arc::new(AtomicUsize::new(0));
    let outer_c = Arc::clone(&outer);
    let inner_c = Arc::clone(&inner);
    let device_c = device.clone();
    let out = profile_session(
        &device,
        move || OuterSession {
            count: Arc::clone(&outer_c),
        },
        move |p: &Profiler<R>| {
            let (a, _) = p.span("outer-a", outer_span).expect("outer span runs");
            // A nested session of a different state type inside `run`.
            let b = profile_session(
                &device_c,
                {
                    let inner_c = Arc::clone(&inner_c);
                    move || InnerSession {
                        count: inner_c,
                    }
                },
                |p: &Profiler<R>| {
                    p.span("inner", inner_span).expect("inner span runs").0
                },
            )
            .expect("the nested session runs");
            let (c, _) = p.span("outer-b", outer_span).expect("outer span runs");
            // A nested `client.profile` with `Send` bodies, inline on the runner.
            let nested = device_c
                .client()
                .profile(
                    || {
                        device_c
                            .client()
                            .profile(|| 7u32, "nested-inner")
                            .expect("nested profile runs")
                            .0
                            + 1
                    },
                    "nested-outer",
                )
                .expect("outer profile runs")
                .0;
            (a, b, c, nested)
        },
    )
    .expect("the session runs");
    assert_eq!((out.0, out.2), (100, 101), "sequential outer spans return values");
    assert_eq!(out.1, 200, "the nested session's span returns its value");
    assert_eq!(out.3, 8, "nested client.profile executes inline with values");
    assert_eq!(outer.load(Ordering::SeqCst), 2, "each outer span runs once");
    assert_eq!(inner.load(Ordering::SeqCst), 1, "the inner span runs once");
}

#[test]
fn harness_generate_equals_the_same_call_outside() {
    let _serial = serial();
    // A `generate` call inside the harness returns exactly what the same
    // call (same seed, same weights, same batch) returns outside it.
    let device = dev();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0], [3, 7, 1, 2, 0, 0, 0, 0, 0, 0]];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let mut cfg = tiny_config();
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = make_spectra(&[601, 602], &precursors, 64, &[10, 12], 33);
    let gcfg = tiny_generation();
    // Outside: same construction on the caller thread, warmed twice.
    let table_out = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table_out.rows as u32;
    cfg.formula_table.sha256 = table_out.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model_out = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants_out = Ms2Constants::new(&device);
    let mut ws_out = GenerationWorkspace::new();
    for _ in 0..2 {
        model_out
            .generate(&batch, &table_out, &gcfg, &mut ws_out, &constants_out)
            .unwrap();
    }
    let plain = model_out
        .generate(&batch, &table_out, &gcfg, &mut ws_out, &constants_out)
        .unwrap();
    // Inside: the same construction on the runner thread.
    let harnessed = profile_session(
        &device,
        {
            let device_c = device.clone();
            let cfg_c = cfg.clone();
            let host_table_c = host_table.clone();
            let batch_c = batch.clone();
            let gcfg_c = gcfg.clone();
            move || {
                let table =
                    DeviceFormulaTable::<R, f32>::upload(&host_table_c, &device_c).unwrap();
                let mut rng = Rng::seeded(5);
                let model = Ms2Model::<R, f32>::init(&cfg_c, &device_c, &mut rng).unwrap();
                let constants = Ms2Constants::new(&device_c);
                (
                    model,
                    table,
                    constants,
                    GenerationWorkspace::<R, f32>::new(),
                    batch_c,
                    gcfg_c,
                )
            }
        },
        |p: &Profiler<R>| {
            p.with_state(
                |s: &mut (
                    Ms2Model<R, f32>,
                    DeviceFormulaTable<R, f32>,
                    Ms2Constants<R>,
                    GenerationWorkspace<R, f32>,
                    SpectrumBatch,
                    GenerationConfig,
                )| {
                    for _ in 0..2 {
                        s.0.generate(&s.4, &s.1, &s.5, &mut s.3, &s.2).unwrap();
                    }
                    s.0.generate(&s.4, &s.1, &s.5, &mut s.3, &s.2).unwrap()
                },
            )
            .expect("the harnessed call runs")
        },
    )
    .expect("the session runs");
    assert_eq!(harnessed, plain, "the harnessed call returns the identical batch");
}

#[test]
fn sync_span_advances_the_synchronise_counter_by_one() {
    let _serial = serial();
    // The observable seam for the span's trailing synchronisation: the
    // crate-level synchronise counter advances by exactly one per span, and
    // the span body itself synchronises nothing. The count lives in
    // `Device::try_synchronize` itself (the actual drain), not in a wrapper,
    // so removing the span's `device.synchronize()` removes its count with
    // it and this test fails — the finding-A3 escaping mutant (sync deleted,
    // increment kept) is no longer expressible. The value-based completion
    // test above still cannot do this on the CPU runtime, which is why the
    // counter exists. This test is the only reader of that counter in
    // its binary besides the instrumentation test below, and the file lock
    // keeps every span inside the measured window this test's own.
    let device = dev();
    let before = synchronize_count();
    let (out, wall) = sync_span(&device, model_free_work);
    assert_eq!(out, 7, "the span returns the body's value");
    assert!(wall.as_secs_f64() >= 0.0, "wall time is non-negative: {wall:?}");
    assert_eq!(
        synchronize_count() - before,
        1,
        "one span performs exactly one crate-level synchronisation"
    );
    let before = synchronize_count();
    let (inner_out, outer_wall) = sync_span(&device, || sync_span(&device, model_free_work));
    assert!(outer_wall.as_secs_f64() >= 0.0, "outer wall is non-negative");
    assert_eq!(inner_out.0, 7, "nested spans return values");
    assert_eq!(
        synchronize_count() - before,
        2,
        "nested spans synchronise once each"
    );
}

fn panicking_span(_: &mut CounterSession) -> u32 {
    panic!("staged span panic");
}

fn panicking_state(_: &mut CounterSession) -> u32 {
    panic!("staged with_state panic");
}

/// A second session state type, for the wrong-type access test.
struct OtherSession {
    #[allow(dead_code)]
    value: u32,
}

fn other_span(_: &mut OtherSession) -> u32 {
    0
}

fn build_counter(spans: &Arc<AtomicUsize>, value: u32) -> CounterSession {
    CounterSession {
        spans: Arc::clone(spans),
        value,
    }
}

#[test]
fn harness_span_panic_is_an_error_and_the_session_recovers() {
    let _serial = serial();
    // A panic inside a span body is caught inside the profiling closure (so
    // CubeCL closes its profile token) and returned as `Err`; the state is
    // restored, so the next span works and sees the state.
    let device = dev();
    let spans = Arc::new(AtomicUsize::new(0));
    let spans_c = Arc::clone(&spans);
    let (failed, recovered, count) = profile_session(
        &device,
        move || build_counter(&spans_c, 40),
        |p: &Profiler<R>| {
            let failed = p.span("panicking", panicking_span).is_err();
            let recovered = p.span("after", count_span_a).expect("the next span runs").0;
            let count = p
                .with_state(|s: &mut CounterSession| {
                    s.spans.fetch_add(10, Ordering::SeqCst);
                    s.value
                })
                .expect("state is intact");
            (failed, recovered, count)
        },
    )
    .expect("the session runs");
    assert!(failed, "the panicking span returns Err");
    assert_eq!(recovered, 41, "the next span runs and sees the state");
    assert_eq!(count, 40, "the state value survived the panic");
    assert_eq!(
        spans.load(Ordering::SeqCst),
        11,
        "only the recovered span and the with_state ran bodies (1 + 10)"
    );
}

#[test]
fn harness_with_state_panic_is_an_error_and_the_session_recovers() {
    let _serial = serial();
    // Same for `with_state`: the panic becomes `Err`, the state is restored,
    // and the next span works and sees the state.
    let device = dev();
    let spans = Arc::new(AtomicUsize::new(0));
    let spans_c = Arc::clone(&spans);
    let (failed, recovered) = profile_session(
        &device,
        move || build_counter(&spans_c, 40),
        |p: &Profiler<R>| {
            let failed = p.with_state(panicking_state).is_err();
            let recovered = p.span("after", count_span_a).expect("the next span runs").0;
            (failed, recovered)
        },
    )
    .expect("the session runs");
    assert!(failed, "the panicking with_state returns Err");
    assert_eq!(recovered, 41, "the next span runs and sees the state");
    assert_eq!(spans.load(Ordering::SeqCst), 1, "only the recovered span ran");
}

#[test]
fn harness_wrong_type_access_is_an_error_and_leaves_the_state() {
    let _serial = serial();
    // A span or `with_state` naming another state type fails without
    // consuming anything: the slot keeps the state, and the next correctly
    // typed span works and sees it.
    let device = dev();
    let spans = Arc::new(AtomicUsize::new(0));
    let spans_c = Arc::clone(&spans);
    let (span_failed, state_failed, recovered) = profile_session(
        &device,
        move || build_counter(&spans_c, 40),
        |p: &Profiler<R>| {
            let span_failed = p.span("wrong-type", other_span).is_err();
            let state_failed = p
                .with_state(|_: &mut OtherSession| 0u32)
                .is_err();
            let recovered = p.span("after", count_span_a).expect("the next span runs").0;
            (span_failed, state_failed, recovered)
        },
    )
    .expect("the session runs");
    assert!(span_failed, "the wrong-typed span returns Err");
    assert!(state_failed, "the wrong-typed with_state returns Err");
    assert_eq!(recovered, 41, "the state survived both mismatches");
    assert_eq!(spans.load(Ordering::SeqCst), 1, "no mismatched body ran");
}

#[test]
fn harness_outer_handle_is_rejected_during_a_nested_session() {
    let _serial = serial();
    // Session identity: the outer handle used while an inner session of the
    // same type is active fails without running its body on (or otherwise
    // touching) the inner state; likewise with an inner session of a
    // different type, where the old code destroyed the inner state through
    // the failing downcast. The inner session then completes normally, and
    // so does the outer one afterwards.
    let device = dev();
    let outer_count = Arc::new(AtomicUsize::new(0));
    let inner_count = Arc::new(AtomicUsize::new(0));
    let outer_c = Arc::clone(&outer_count);
    let inner_c = Arc::clone(&inner_count);
    let device_c = device.clone();
    let out = profile_session(
        &device,
        move || build_counter(&outer_c, 40),
        move |outer_p: &Profiler<R>| {
            let stale = outer_p.clone();
            // Same-type inner session: the stale outer span and with_state
            // both fail, and neither body's counter moves.
            let same = profile_session(
                &device_c,
                {
                    let inner_c = Arc::clone(&inner_c);
                    move || build_counter(&inner_c, 50)
                },
                move |inner_p: &Profiler<R>| {
                    let span_failed = stale.span("stale", count_span_a).is_err();
                    let state_failed = stale.with_state(|_: &mut CounterSession| 0u32).is_err();
                    let inner_value = inner_p.span("inner", count_span_b).expect("inner runs").0;
                    (span_failed, state_failed, inner_value)
                },
            )
            .expect("the same-type inner session runs");
            // Different-type inner session: the stale outer access still
            // fails, and the inner state is intact afterwards.
            let device_c2 = device_c.clone();
            let stale2 = outer_p.clone();
            let different = profile_session(
                &device_c2,
                || OtherSession { value: 60 },
                move |inner_p: &Profiler<R>| {
                    let span_failed = stale2.span("stale", count_span_a).is_err();
                    let inner_value = inner_p.span("inner", other_span).expect("inner runs").0;
                    (span_failed, inner_value)
                },
            )
            .expect("the different-type inner session runs");
            let outer_value = outer_p.span("outer", count_span_a).expect("outer runs").0;
            (same, different, outer_value)
        },
    )
    .expect("the outer session runs");
    assert!(out.0.0, "stale outer span fails during a same-type inner session");
    assert!(out.0.1, "stale outer with_state fails during a same-type inner session");
    assert_eq!(out.0.2, 52, "the inner session's span ran on the inner state");
    assert!(out.1.0, "stale outer span fails during a different-type inner session");
    assert_eq!(out.1.1, 0, "the different-type inner session completed");
    assert_eq!(out.2, 41, "the outer session resumes after the nested ones");
    assert_eq!(outer_count.load(Ordering::SeqCst), 1, "no stale outer body ran");
    assert_eq!(inner_count.load(Ordering::SeqCst), 1, "the inner span ran exactly once");
}

// ---------------------------------------------------------------------------
// Device-mode production equivalence: per-stage device spans run the shared
// production stage functions (finding 1), so per-stage launch/read/
// allocation counters equal the host hook path's and the candidates equal
// production's for the same seed.
// ---------------------------------------------------------------------------

use mamba3::autograd::Var;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::decoder::DecoderState;
use mamba3::models::ms2::encoder::EncoderOutput;
use mamba3::models::ms2::generate::{GeneratePreflight, GenerateStage};

/// Runner-thread session mirroring the profile driver's device-mode session:
/// production pieces plus the intermediates stashed between spans, with
/// per-stage counter deltas recorded by the span bodies themselves.
struct EquivSession {
    device: Device<R>,
    model: Ms2Model<R, f32>,
    table: DeviceFormulaTable<R, f32>,
    constants: Ms2Constants<R>,
    workspace: GenerationWorkspace<R, f32>,
    batch: SpectrumBatch,
    gen_config: GenerationConfig,
    pre: GeneratePreflight,
    host_status: Vec<u32>,
    spectrum_ids: Vec<u64>,
    spectra: Option<DeviceSpectra<R, f32>>,
    encoded: Option<EncoderOutput<R, f32>>,
    decoder_state: Option<DecoderState<R, f32>>,
    bond_table: Option<Tensor<R, f32>>,
    traj_formula: Option<Var<R, f32>>,
    counts: Vec<(usize, usize, usize)>,
    readout: Option<mamba3::models::ms2::contract::CandidateBatch>,
}

fn equiv_counts() -> (usize, usize, usize) {
    (launch_count(), runtime_read_count(), allocation_calls())
}

fn equiv_push(s: &mut EquivSession, before: (usize, usize, usize)) {
    let (l, r, a) = equiv_counts();
    s.counts.push((l - before.0, r - before.1, a - before.2));
}

fn equiv_preprocess(s: &mut EquivSession) -> Result<(), String> {
    let before = equiv_counts();
    s.model
        .generate_preflight(&s.batch, &s.table, &s.gen_config)
        .map_err(|e| e.to_string())?;
    let spectra = s
        .model
        .generate_preprocess(&s.batch, &s.gen_config, &s.device)
        .map_err(|e| e.to_string())?;
    s.host_status = spectra.host_status.clone();
    s.spectrum_ids = spectra.spectrum_id.clone();
    s.spectra = Some(spectra);
    equiv_push(s, before);
    Ok(())
}

fn equiv_encoder(s: &mut EquivSession) -> Result<(), String> {
    let before = equiv_counts();
    let spectra = s.spectra.as_ref().ok_or("encoder span ran before preprocess")?;
    let encoded = s
        .model
        .generate_encode_ws(&mut s.workspace, spectra, s.gen_config.control, &s.pre, &s.device)
        .map_err(|e| e.to_string())?;
    s.encoded = Some(encoded);
    equiv_push(s, before);
    Ok(())
}

fn equiv_search(s: &mut EquivSession) -> Result<(), String> {
    let before = equiv_counts();
    let spectra = s.spectra.as_ref().ok_or("search span ran before preprocess")?;
    let pool = s
        .encoded
        .as_ref()
        .ok_or("search span ran before encoder")?
        .pool
        .clone();
    s.model
        .generate_search_ws(
            &mut s.workspace,
            spectra,
            &s.batch,
            &pool,
            &s.table,
            s.pre.spectra_n,
            s.pre.trajectories,
            s.pre.formulas,
            false,
            s.gen_config.formula_rows_visited_max,
            s.gen_config.formula_rows_scored_max,
            &s.gen_config,
            &s.pre,
            &s.device,
        )
        .map_err(|e| e.to_string())?;
    equiv_push(s, before);
    Ok(())
}

fn equiv_decoder_init(s: &mut EquivSession) -> Result<(), String> {
    let before = equiv_counts();
    let encoded = s.encoded.as_ref().ok_or("init span ran before encoder")?;
    let (state, bonds, traj) = s
        .model
        .generate_decoder_init_ws(&mut s.workspace, encoded, &s.pre, &s.device)
        .map_err(|e| e.to_string())?;
    s.decoder_state = Some(state);
    s.bond_table = Some(bonds);
    s.traj_formula = Some(traj);
    equiv_push(s, before);
    Ok(())
}

fn equiv_decode_step(s: &mut EquivSession, step: usize) -> Result<(), String> {
    let before = equiv_counts();
    let seed_lo = (s.gen_config.seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (s.gen_config.seed >> 32) as u32;
    let temperature = s.gen_config.temperature;
    let encoded = s.encoded.as_ref().ok_or("loop span ran before encoder")?;
    let traj = s.traj_formula.as_ref().ok_or("loop span ran before decoder init")?;
    let bonds = s.bond_table.as_ref().ok_or("loop span ran before decoder init")?;
    let mut state = s.decoder_state.take().ok_or("loop span ran before decoder init")?;
    let carry = s
        .model
        .generate_decode_step_ws(
            &mut s.workspace,
            encoded,
            traj,
            &mut state,
            bonds,
            &s.constants.atom_table,
            step,
            seed_lo,
            seed_hi,
            temperature,
            s.pre.trajectories,
            &s.pre,
            &s.device,
        )
        .map_err(|e| e.to_string())?;
    assert!(carry.is_none(), "capture stays off in the harness");
    s.decoder_state = Some(state);
    equiv_push(s, before);
    Ok(())
}

fn equiv_validate(s: &mut EquivSession) -> Result<(), String> {
    let before = equiv_counts();
    s.model
        .generate_validate_ws(
            &mut s.workspace,
            &s.constants.atom_table,
            s.pre.trajectories,
            &s.gen_config,
            &s.pre,
            &s.device,
        )
        .map_err(|e| e.to_string())?;
    equiv_push(s, before);
    Ok(())
}

fn equiv_readout(s: &mut EquivSession) -> Result<(), String> {
    let before = equiv_counts();
    let out = s
        .model
        .generate_readout_ws(
            &mut s.workspace,
            &s.host_status,
            &s.spectrum_ids,
            &s.gen_config,
            s.pre.trajectories,
            s.pre.formulas,
            &s.pre,
            &s.device,
        )
        .map_err(|e| e.to_string())?;
    s.readout = Some(out);
    equiv_push(s, before);
    Ok(())
}

#[test]
fn harness_device_stages_equal_production_counters_and_candidates() {
    let _serial = serial();
    // Device-mode stages are the production workload (finding 1): the same
    // workspace-level stage functions production calls, over warmed buckets,
    // under the same `no_grad` guard. Per-stage (launch, runtime-read,
    // allocation) counters therefore equal the host hook path's, and the
    // readout batch equals production's for the same seed.
    let device = dev();
    let mut cfg = tiny_config();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = make_spectra(&[701], &precursors, 64, &[10], 35);
    let gcfg = tiny_generation();
    let decode_steps = (gcfg.max_steps as usize - 1).max(1);

    // Host path: warmed production calls, then one hooked call with counter
    // snapshots at each real boundary.
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
    }
    device.synchronize();
    reset_launch_count();
    reset_transfer_counters();
    let plain = model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    reset_launch_count();
    reset_transfer_counters();
    let mut host_counts: Vec<(usize, usize, usize)> = Vec::new();
    let mut prev = equiv_counts();
    let hooked = {
        let mut hook = |_: GenerateStage| {
            let now = equiv_counts();
            host_counts.push((now.0 - prev.0, now.1 - prev.1, now.2 - prev.2));
            prev = now;
        };
        model
            .generate_with_hook(&batch, &table, &gcfg, &mut ws, &constants, Some(&mut hook))
            .unwrap()
    };
    let end = equiv_counts();
    host_counts.push((end.0 - prev.0, end.1 - prev.1, end.2 - prev.2));
    assert_eq!(hooked, plain, "the hooked call is production for the same seed");
    // Boundaries: preprocess, encoder, search, decoder init, one per decode
    // step, validate, readout; the readout hook fires right after the single
    // batched read, closed out by the final snapshot (which must show no
    // work of its own).
    assert_eq!(
        host_counts.len(),
        4 + decode_steps + 3,
        "every hook boundary fires, plus the close-out snapshot"
    );

    // Device path: the same construction on the runner thread, warmed with
    // the same production calls, then one span per production stage.
    let device_c = device.clone();
    let cfg_c = cfg.clone();
    let host_table_c = host_table.clone();
    let batch_c = batch.clone();
    let gcfg_c = gcfg.clone();
    let (device_counts, device_batch) = profile_session(
        &device,
        move || {
            let table =
                DeviceFormulaTable::<R, f32>::upload(&host_table_c, &device_c).unwrap();
            let mut rng = Rng::seeded(5);
            let model = Ms2Model::<R, f32>::init(&cfg_c, &device_c, &mut rng).unwrap();
            let pre = model
                .generate_preflight(&batch_c, &table, &gcfg_c)
                .expect("device session preflights");
            EquivSession {
                device: device_c.clone(),
                constants: Ms2Constants::new(&device_c),
                workspace: GenerationWorkspace::new(),
                model,
                table,
                batch: batch_c,
                gen_config: gcfg_c,
                pre,
                host_status: Vec::new(),
                spectrum_ids: Vec::new(),
                spectra: None,
                encoded: None,
                decoder_state: None,
                bond_table: None,
                traj_formula: None,
                counts: Vec::new(),
                readout: None,
            }
        },
        move |p: &Profiler<R>| {
            p.with_state(|s: &mut EquivSession| {
                for _ in 0..2 {
                    s.model
                        .generate(&s.batch, &s.table, &s.gen_config, &mut s.workspace, &s.constants)
                        .expect("device warmup runs");
                }
                reset_launch_count();
                reset_transfer_counters();
            })
            .expect("warmup and reset run");
            let span = |p: &Profiler<R>, name: &str, f: fn(&mut EquivSession) -> Result<(), String>| {
                let (r, _) = p.span(name, f).expect("profile span runs");
                r.expect("profiled stage runs");
            };
            span(p, "ms2.preprocess", equiv_preprocess);
            span(p, "ms2.encoder", equiv_encoder);
            span(p, "ms2.search", equiv_search);
            span(p, "ms2.decoder_init", equiv_decoder_init);
            for step in 1..=decode_steps {
                // One span per production decode step: each span body is a
                // `fn` pointer, so the step index travels in a per-step
                // monomorphised shim below.
                match step {
                    1 => span(p, "ms2.step1", equiv_step_1),
                    2 => span(p, "ms2.step2", equiv_step_2),
                    3 => span(p, "ms2.step3", equiv_step_3),
                    4 => span(p, "ms2.step4", equiv_step_4),
                    5 => span(p, "ms2.step5", equiv_step_5),
                    6 => span(p, "ms2.step6", equiv_step_6),
                    7 => span(p, "ms2.step7", equiv_step_7),
                    8 => span(p, "ms2.step8", equiv_step_8),
                    9 => span(p, "ms2.step9", equiv_step_9),
                    10 => span(p, "ms2.step10", equiv_step_10),
                    11 => span(p, "ms2.step11", equiv_step_11),
                    12 => span(p, "ms2.step12", equiv_step_12),
                    13 => span(p, "ms2.step13", equiv_step_13),
                    14 => span(p, "ms2.step14", equiv_step_14),
                    15 => span(p, "ms2.step15", equiv_step_15),
                    16 => span(p, "ms2.step16", equiv_step_16),
                    17 => span(p, "ms2.step17", equiv_step_17),
                    18 => span(p, "ms2.step18", equiv_step_18),
                    19 => span(p, "ms2.step19", equiv_step_19),
                    20 => span(p, "ms2.step20", equiv_step_20),
                    21 => span(p, "ms2.step21", equiv_step_21),
                    _ => panic!("tiny config has 21 decode steps"),
                }
            }
            span(p, "ms2.validate", equiv_validate);
            span(p, "ms2.readout", equiv_readout);
            p.with_state(|s: &mut EquivSession| {
                (s.counts.clone(), s.readout.clone().expect("readout ran"))
            })
            .expect("results read out")
        },
    )
    .expect("the session runs");
    // Same number of stages, same counters per stage, same candidates. The
    // host's final close-out entry (after the readout hook) must be empty on
    // both sides — the device path has no such entry by construction.
    assert_eq!(
        device_counts.len() + 1,
        host_counts.len(),
        "device spans cover every production stage"
    );
    for (i, (host, dev)) in host_counts.iter().zip(device_counts.iter()).enumerate() {
        assert_eq!(
            dev, host,
            "stage {i}: device span counters (launches, runtime reads, allocations) equal the host hook window's"
        );
    }
    assert_eq!(
        host_counts[device_counts.len()],
        (0, 0, 0),
        "the close-out after the readout hook launches, reads and allocates nothing"
    );
    device_batch.validate().expect("device candidates validate");
    assert_eq!(
        device_batch, plain,
        "the device-mode stages return production's candidates for the same seed"
    );
}

fn equiv_step_1(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 1)
}
fn equiv_step_2(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 2)
}
fn equiv_step_3(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 3)
}
fn equiv_step_4(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 4)
}
fn equiv_step_5(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 5)
}
fn equiv_step_6(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 6)
}
fn equiv_step_7(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 7)
}
fn equiv_step_8(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 8)
}
fn equiv_step_9(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 9)
}
fn equiv_step_10(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 10)
}
fn equiv_step_11(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 11)
}
fn equiv_step_12(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 12)
}
fn equiv_step_13(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 13)
}
fn equiv_step_14(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 14)
}
fn equiv_step_15(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 15)
}
fn equiv_step_16(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 16)
}
fn equiv_step_17(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 17)
}
fn equiv_step_18(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 18)
}
fn equiv_step_19(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 19)
}
fn equiv_step_20(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 20)
}
fn equiv_step_21(s: &mut EquivSession) -> Result<(), String> {
    equiv_decode_step(s, 21)
}

#[test]
fn synchronize_count_instruments_the_actual_synchronisation() {
    let _serial = serial();
    // Finding A3: the counter instruments the actual operation. A direct
    // `try_synchronize` — no wrapper, no span — advances the counter by
    // exactly one, proving the increment lives in the drain itself; a span
    // therefore cannot count without syncing. The file lock keeps the
    // measured windows this test's own.
    let device = dev();
    let before = synchronize_count();
    device
        .try_synchronize()
        .expect("an idle synchronisation succeeds");
    assert_eq!(
        synchronize_count() - before,
        1,
        "one real synchronisation counts exactly once"
    );
    let before = synchronize_count();
    let (out, wall) = sync_span(&device, model_free_work);
    assert_eq!(out, 7, "the span returns the body's value");
    assert!(wall.as_secs_f64() >= 0.0, "wall time is non-negative: {wall:?}");
    assert_eq!(
        synchronize_count() - before,
        1,
        "one span performs exactly one real synchronisation"
    );
    // Deferred-failure observability on the CPU runtime: none exists, so no
    // stronger behavioural test is available here. What was tried:
    // - the infinite-literal kernel of `tests/kernel_errors.rs` compiles on
    //   the CPU runtime (`rejects_non_finite_literals` is false there), so it
    //   runs and parks no error;
    // - the CPU stream (`cubecl-cpu-0.10.0/src/compute/stream.rs:63`)
    //   executes queued closures at flush and parks errors only through
    //   `stream.error()`, which no safe kernel launch produces — CPU kernels
    //   are compiled Rust closures with no shader-compile step to fail
    //   asynchronously.
    // Hence no safe-code launch can park a deferred error that only a real
    // synchronisation surfaces at the span boundary on this runtime (every
    // read blocks there anyway). The structural fix above — the count lives
    // in `try_synchronize`, so a removed sync takes its count with it — plus
    // the span-delta test is the strongest test available.
}

#[test]
fn enumerate_limit_between_estimates_refuses_before_any_launch_or_allocation() {
    let _serial = serial();
    // Finding A2 at the production seam the driver preflights: with
    // `FormulaSource::Enumerate`, a `max_device_bytes` between the
    // table-only estimate and the enumeration-inclusive estimate refuses in
    // `generate_preflight` — before any upload, allocation or launch — so
    // the driver's enumeration-inclusive preflight (base, slope, stability,
    // device session) can never admit a configuration the cold call panics
    // on. Counter discipline as elsewhere in this binary: this test holds
    // the file lock across its reset-plus-measure window.
    let device = dev();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let mut cfg = tiny_config();
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let domain = EnumDomain::from_compositions(comps.iter().copied(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.iter().copied(), 0).unwrap();
    model.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    let artifacts = model.enum_artifacts.as_ref().expect("artifacts uploaded");
    let enum_p = artifacts.p as u64;
    let enum_words = artifacts.bounds.len() as u64;
    assert!(enum_p > 0, "the fixture domain has resident lanes");
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = make_spectra(&[901], &precursors, 64, &[10], 41);
    let mut gcfg = tiny_generation();
    gcfg.formula_source = FormulaSource::Enumerate;
    // The driver's two estimates for exactly this shape: table-only (what
    // the driver used to preflight) versus enumeration-inclusive (what it
    // must preflight). The resident artifacts add strictly positive items
    // (`rare`, `bounds`, `lane_stats`, `offsets`), so the inclusive total
    // is strictly larger and a limit strictly between them exists.
    let table_only = Ms2MemoryEstimate::generation(
        &cfg,
        table.rows as u64,
        batch.len() as u64,
        u64::from(gcfg.trajectories),
        u64::from(batch.n_raw),
        u64::from(gcfg.max_steps),
        u64::from(gcfg.formula_window),
        u64::from(gcfg.formulas),
    )
    .and_then(|est| est.total())
    .expect("the table-only estimate fits u64");
    let with_enum = Ms2MemoryEstimate::generation_with_enum(
        &cfg,
        table.rows as u64,
        batch.len() as u64,
        u64::from(gcfg.trajectories),
        u64::from(batch.n_raw),
        u64::from(gcfg.max_steps),
        u64::from(gcfg.formula_window),
        u64::from(gcfg.formulas),
        enum_p,
        enum_words,
    )
    .and_then(|est| est.total())
    .expect("the enumeration-inclusive estimate fits u64");
    assert!(
        with_enum > table_only,
        "the enumeration-inclusive estimate ({with_enum}) exceeds the table-only one ({table_only})"
    );
    gcfg.max_device_bytes = table_only + (with_enum - table_only) / 2;
    assert!(
        gcfg.max_device_bytes > table_only && gcfg.max_device_bytes < with_enum,
        "the test limit sits strictly between the estimates"
    );
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    device.synchronize();
    reset_launch_count();
    reset_transfer_counters();
    let err = model
        .generate(&batch, &table, &gcfg, &mut ws, &constants)
        .expect_err("a limit between the estimates refuses");
    assert!(
        format!("{err}").contains("exceeds limit"),
        "the refusal is the memory-limit preflight, got: {err}"
    );
    assert_eq!(launch_count(), 0, "the preflight refuses before any launch");
    assert_eq!(
        allocation_calls(),
        0,
        "the preflight refuses before any allocation"
    );
}
