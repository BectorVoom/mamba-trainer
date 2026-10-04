//! MS2 substructure profile driver: stage spans, synchronised wall clock,
//! device-timestamp spans via the runner-local harness, warmup, cold-start
//! reporting and machine-readable JSON (architecture §6.5, P2.6).
//!
//! ```text
//! cargo run --release --no-default-features --features cpu --example profile_ms2_substructure -- \
//!   --mode both --n 64,128 --b 1,8 --k 8 --warmup 2 --repeats 3 --stability 20 --out /tmp/ms2_profile.json
//! ```
//!
//! Modes `--mode generate|train|both`; shapes `--n 64,128,256,512`
//! (raw peaks per spectrum) with `--b 1,8,32` (cartesian), `--k 8`
//! (trajectories), `--warmup 2`, `--repeats 10`, `--stability 0`,
//! `--profile-mode host|device` (default `host`), `--out <json>`,
//! `--max-device-bytes <bytes>` (default 4 GiB).
//!
//! Per configuration: cold first-call wall time; warm p50/p95 wall time over
//! `repeats` with a device sync per call; per stage (preprocess/upload,
//! encoder, formula search, decoder initialisation, decode steps all and per
//! step, validate, readout for generation; forward, backward, optimizer for
//! training) launches, reads, upload/download bytes, allocation calls,
//! synchronised wall time (`sync_wall_ms`) and device-span time
//! (`profile_ms`, device mode only).
//!
//! Timing clocks. `sync_wall_ms` is measured with a host `Instant` around the
//! stage plus the device synchronisation, so its timer is always
//! `"SynchronizedHostWallClock"` — even on a runtime whose profiling
//! capability is `DeviceTimestamps`. The runtime's own capability (probed
//! once from what `client.profile` returns: `DeviceTimestamps` for hardware
//! timestamps, `SystemTime` for host wall time) is recorded separately in
//! `timing_method`; the two fields must not be conflated.
//!
//! Host mode (default) isolates stages with the generation/training stage
//! hooks (`Ms2Model::generate_with_hook`, `Ms2Trainer::step_with_boundaries`):
//! one instrumented call runs exactly what production runs, and the hook
//! snapshots counters and synchronised wall time at each real boundary
//! (after forward/loss, after backward, after optimizer; after
//! preprocess/upload, encoder, formula search, decoder init, each decode
//! step, validate, readout). The device is drained (synchronised) before
//! counters are reset and any isolated timer starts, so no previously queued
//! work is charged to a stage. `profile_ms` is the JSON string
//! `"unavailable"` in host mode.
//!
//! Device mode (`--profile-mode device`) runs the same session on the device
//! runner thread through `mamba3::backend::profile_session` and times each
//! stage with a real `client.profile` span, so `profile_ms` is a number and
//! `timing_method` is the `ProfileDuration`'s own method — subject to the
//! single-pass limitation below. Each device-mode
//! stage calls the SAME workspace-level production stage functions
//! (`generate_*_ws`, `forward_state`/`backward_state`/`optimizer_step`) that
//! `generate_with_hook`/`step_with_boundaries` call — warmed workspace
//! buckets, the same `no_grad` guard — over a session warmed with the same
//! production calls. No replica stage code remains. The span bodies are
//! `fn` pointers over runner-thread-local session state (they capture no
//! `Rc`), which is why this works from safe code: `client.profile` runs its
//! closure on the device runner thread (`cubecl-runtime-0.10.0/src/client.rs`,
//! `profile` through `device.exclusive`; `cubecl-common-0.10.0/src/device/handle/channel.rs:151`
//! executes nested calls inline because the caller already is the runner),
//! so a closure borrowing the caller's `Rc`-held model cannot enter it — but
//! a session built on the runner thread can. Counts, wall times and every
//! other field are identical between the modes; the production cold/warm
//! measurements always stay outside the harness.
//!
//! Whole-stage device duration is UNAVAILABLE on the multi-pass wgpu path
//! (finding A1, verified against pinned `cubecl-wgpu-0.10.0`): an ordinary
//! `client.profile` span returns the first timestamped compute pass's
//! begin-to-end duration, not the stage's elapsed device time, once a stage
//! spans several passes (`compute/stream.rs:239` flushes queued work and
//! opens the token; `compute/stream.rs:548` attaches timestamp writes only
//! when a new pass opens; `compute/timings.rs:321` drains newly initialised
//! tokens, so later passes get no timestamp writes; `compute/stream.rs:457`
//! ends the pass once `tasks_count >= tasks_max`; `compute/timings.rs:193`
//! resolves the token's end against its initial query set). One pass holds
//! at most `device_pass_task_limit` tasks (default 32,
//! `cubecl-wgpu-0.10.0/src/runtime.rs:188`, overridable with
//! `CUBECL_WGPU_MAX_TASKS` at `runtime.rs:192`; recorded in every record and
//! device stage as `device_pass_task_limit`), and any mid-stage upload or
//! read forces a flush too (`compute/stream.rs:105` write path,
//! `read_resources` ends the pass). Hence on a `DeviceTimestamps` runtime a
//! stage whose host-measured launches exceed the limit, or that uploads or
//! reads mid-stage, reports `"profile_ms": "unavailable"` with a
//! `profile_scope` saying exactly that; a stage that provably fits one pass
//! (launches within the limit, no mid-stage upload/read) keeps its number
//! with the scope `"single timestamped compute pass"`. On the CPU runtime
//! (`SystemTime`) nothing changes. `device_span_plausible` remains only as
//! an extra self-check alongside the scope. A `--profile-mode device-sum`
//! that would sum per-launch-group spans is NOT built: the production stage
//! functions expose no launch-group decomposition (each `generate_*_ws`
//! stage is one closure; only the decode loop could be split per step, and a
//! step still exceeds one pass), so summing separately profiled groups
//! would change the measurement without becoming whole-stage elapsed time.
//!
//! Enumeration memory (finding A2): with `--formula-source enumerate` every
//! estimate and refusal in this driver — the base record, the T+1 slope, the
//! stability preflight and the device session preflight — uses the
//! enumeration-inclusive estimate (`Ms2MemoryEstimate::generation_with_enum`
//! / `training_with_enum`): the domain and bounds are fitted on the host
//! compositions first and sized exactly as `DeviceEnumArtifacts::upload`
//! sizes them, before any allocation, so a limit between the table-only and
//! the enumeration-inclusive estimate REFUSES instead of panicking in the
//! cold call.
//!
//! Allocation (P2 acceptance, NOT met in V0). The warmed decode loop
//! allocates device buffers every step (the mixer step and the composed
//! attention allocate their outputs functionally): the per-step allocation
//! slope (a warmed T+1-step call minus the warmed T-step call) is positive,
//! and `decoder_loop_allocation_free` is `false`. The P2 acceptance item
//! "workspace allocation happens outside the decoder hot loop" stays open
//! for P8.2/O2 work; V0 measures the slope and reports reserved bytes, it
//! does not claim the loop is allocation-free.
//!
//! Launch budget. `L_call = L_preprocess + L_encoder + L_search + L_init +
//! (T-1)*L_step + L_finalize` must equal the measured warmed total exactly:
//! `L_init` (decoder initialisation: K/V projections, cache fills,
//! atom-memory fills) is measured in its own hook window, `L_step` comes
//! from the warmed launch slope and is cross-checked against the hook's
//! decode-loop total, and the driver exits non-zero when the reconciliation
//! difference is not 0.
//!
//! Reserved bytes are recorded two ways: `reserved_bytes_after` is the
//! production endpoint, and `peak_reserved_bytes_sampled` is the high-water
//! mark sampled at every hook boundary of the instrumented call (a transient
//! high reservation during a call can exceed any single endpoint, so the
//! endpoint is never presented as a peak). Configurations whose
//! `Ms2MemoryEstimate` exceeds `--max-device-bytes` are recorded as refused
//! with the estimate, not run — including the `--stability` measurement,
//! which preflights its tested configurations with the same estimate before
//! any allocation.
//!
//! Stage profiling never changes the production read count: production
//! (cold/warm) numbers are measured first with their own counter windows,
//! and every profiling sync lives in a later, separately labelled
//! profiling window (P2.9). Weights are random-init (seeded), so timings are
//! structural, not quality numbers; spectra are synthetic from the chemistry
//! fixture's compositions plus seeded peaks (see `fixture_comps`).
//!
//! `--stability <calls>` (default 0 = skip) is the P2.3 serving measurement:
//! after warmup it runs that many identical `generate` calls and then an
//! alternating sequence over the first two `--b` values, recording the
//! reserved-bytes series (first/last/min/max over the calls), allocation
//! calls per call (min/max) and bucket counts captured at each phase
//! boundary (`GenerationWorkspace::bucket_keys().len()`). It changes no
//! behaviour: it runs in its own window after the production numbers with
//! counters reset around every call.

#![recursion_limit = "256"]

use std::time::Instant;

use mamba3::autograd::Var;
use mamba3::backend::{
    Device, Profiler, SYNC_WALL_TIMER, allocation_calls, download_bytes, launch_count,
    memory_snapshot, profile_session, read_count, reserved_bytes, reset_launch_count,
    reset_read_count, reset_transfer_counters, runtime_read_count, upload_bytes,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{Composition, composition_mass, element_index};
use mamba3::models::ms2::contract::{
    Control, GenerationConfig, ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::decoder::DecoderState;
use mamba3::models::ms2::encoder::EncoderOutput;
use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GeneratePreflight, GenerateStage, GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::train::{BackwardState, ForwardState, Ms2Trainer, StepPhase, TrainConfig};
use mamba3::models::ms2::workspace::{Ms2Capabilities, Ms2MemoryEstimate};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn usage() -> ! {
    eprintln!(
        "usage: profile_ms2_substructure [--mode generate|train|both] [--n 64,128,256,512] \
         [--b 1,8,32] [--k 8] [--warmup 2] [--repeats 10] [--stability 0] [--profile-mode host|device] \
         [--max-device-bytes 4294967296] --out <json>"
    );
    std::process::exit(2);
}

fn parse_list(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|s| s.trim().parse().unwrap_or_else(|_| usage()))
        .collect()
}

/// Parent compositions derived from `tests/fixtures/ms2/chemistry_v0.json`
/// (benzene, alanine, glucose, naphthalene by formula); falls back to the
/// same four literals when the fixture cannot be read, which is noted in the
/// JSON. Peaks are synthetic: a seeded RNG draws intensities and m/z below
/// each precursor (repeat/perturb has no peak fixture to repeat, since the
/// chemistry fixture carries structures, not spectra).
fn fixture_comps() -> (Vec<Composition>, bool) {
    let text = std::fs::read_to_string("tests/fixtures/ms2/chemistry_v0.json").or_else(|_| {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ms2/chemistry_v0.json"
        ))
    });
    let Ok(text) = text else {
        return (fallback_comps(), false);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return (fallback_comps(), false);
    };
    let mut out = Vec::new();
    if let Some(molecules) = value.get("molecules").and_then(|m| m.as_array()) {
        for m in molecules.iter().take(4) {
            let Some(formula) = m.get("formula") else {
                continue;
            };
            let mut c: Composition = [0; 10];
            let mut ok = false;
            if let Some(obj) = formula.as_object() {
                for (symbol, count) in obj {
                    if let Some(e) = element_index(symbol) {
                        c[e] = count.as_u64().unwrap_or(0) as u16;
                        ok = true;
                    }
                }
            }
            if ok {
                out.push(c);
            }
        }
    }
    if out.len() < 4 {
        return (fallback_comps(), false);
    }
    (out, true)
}

fn fallback_comps() -> Vec<Composition> {
    vec![
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
        [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
        [10, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
}

fn precursor_of(comp: &Composition) -> u32 {
    composition_mass(comp).unwrap() + 1_007_825 - 549
}

fn spectra_batch(comps: &[Composition], n_raw: usize, seed: u64, id_base: u64) -> SpectrumBatch {
    let b = comps.len();
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for bi in 0..b {
        let count = n_raw as u32;
        peak_count[bi] = count;
        raw_peak_count[bi] = count;
        for i in 0..n_raw {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursors[bi] - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + 2.0 * rng.uniform_vec(1, 0.0, 1.0)[0];
        }
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: (0..b as u64).map(|i| id_base + i).collect(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursors,
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

fn experiment_set(comps: &[Composition], n_raw: usize, seed: u64) -> ExperimentSet {
    use mamba3::models::ms2::graph::MolGraph;
    let batch = spectra_batch(comps, n_raw, seed, 9000);
    let spectra = (0..comps.len())
        .map(|i| {
            let base = i * n_raw;
            ExperimentSpectrum {
                molecule: i,
                spectrum: ExportSpectrum {
                    row: i as u64,
                    spectrum_id: batch.spectrum_id[i],
                    adduct: 1,
                    polarity: 1,
                    precursor_mz_udalton: batch.precursor_mz_udalton[i],
                    precursor_uncertainty_udalton: 50,
                    raw_peak_count: batch.raw_peak_count[i],
                    peak_id: batch.peak_id[base..base + n_raw].to_vec(),
                    mz_udalton: batch.mz_udalton[base..base + n_raw].to_vec(),
                    intensity: batch.intensity[base..base + n_raw]
                        .iter()
                        .map(|&v| v as f64)
                        .collect(),
                    mz_uncertainty_udalton: 50,
                    collision_energy_ev: 30.0,
                    collision_energy_known: 1,
                    energy_count: 1,
                    instrument_class: 0,
                },
                parent: MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
                parent_composition: comps[i],
                labels: None,
                domain: SpectrumDomain::InDomainUnlabeled,
            }
        })
        .collect();
    ExperimentSet {
        name: "profile-synthetic".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: (0..comps.len()).map(|i| format!("mol{i}")).collect(),
        spectra,
    }
}

fn p50p95(mut values: Vec<f64>) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(f64::total_cmp);
    let at = |p: f64| {
        let rank = (p / 100.0 * (values.len() - 1) as f64).round() as usize;
        values[rank.min(values.len() - 1)]
    };
    (at(50.0), at(95.0))
}

fn timing_name(device: &Device<R>) -> &'static str {
    match device
        .client()
        .profile(|| {}, "ms2-timing-probe")
        .map(|(_, d)| d.timing_method().to_string())
        .as_deref()
    {
        Ok("device") => "DeviceTimestamps",
        Ok("system") => "SystemTime",
        _ => "Unavailable",
    }
}

/// Tasks one wgpu compute pass holds before the stream ends it (finding A1).
///
/// `tasks_max`, default 32 (`cubecl-wgpu-0.10.0/src/runtime.rs:188`),
/// overridable with `CUBECL_WGPU_MAX_TASKS` (`runtime.rs:192`); the stream
/// submits and ends the current pass once `tasks_count >= tasks_max`
/// (`cubecl-wgpu-0.10.0/src/compute/stream.rs:453-465`). A device-timestamp
/// stage is whole-stage device time only when it provably fits one pass
/// (see the header); the limit travels in the JSON as
/// `device_pass_task_limit`.
fn device_pass_task_limit() -> usize {
    std::env::var("CUBECL_WGPU_MAX_TASKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(32)
}

/// Host-side enumeration artifact sizes for the memory estimate (finding A2).
///
/// Fits the domain and bounds on the given compositions on the host — no
/// device, no allocation — then sizes the resident artifacts exactly as
/// `DeviceEnumArtifacts::upload` does: `rare_table` rows are `P`
/// (`formula_head.rs:253`), `pack_device_bounds` words are the packed bounds
/// length. Returns `(enum_p, enum_bounds_words)` for
/// `Ms2MemoryEstimate::{generation_with_enum, training_with_enum}`.
fn enum_host_sizes(base_comps: &[Composition]) -> (u64, u64) {
    use mamba3::models::ms2::formula_enum::{
        EnumDomain, RatioBounds, pack_device_bounds, rare_table,
    };
    let domain = EnumDomain::from_compositions(base_comps.iter().copied(), 0)
        .expect("enum domain fits on fixture comps");
    let bounds = RatioBounds::fit(base_comps.iter().copied(), 0)
        .expect("ratio bounds fit on fixture comps");
    let p = rare_table(&domain, &bounds)
        .expect("rare table builds on fixture domain")
        .len() as u64;
    let words = pack_device_bounds(&domain, &bounds)
        .expect("bounds pack on fixture domain")
        .len() as u64;
    (p, words)
}

/// Counters, synchronised wall time and the reserved-bytes sample at one hook
/// boundary.
#[derive(Debug, Clone, Copy, Default)]
struct Boundary {
    launches: usize,
    reads: usize,
    runtime_reads: usize,
    upload_bytes: u64,
    download_bytes: u64,
    allocs: usize,
    wall_ms: f64,
    reserved: Option<u64>,
}

impl Boundary {
    /// Drain previously queued work, then snapshot counters and wall time.
    ///
    /// The leading synchronisation is what keeps a stage's window from being
    /// charged for work queued before it; the trailing device state is then
    /// exactly this boundary's. The reserved-bytes sample feeds the
    /// high-water mark (`peak_reserved_bytes_sampled`): the maximum over
    /// every hook boundary of the call, never a single endpoint.
    fn snapshot(device: &Device<R>, started: &Instant) -> Self {
        device.synchronize();
        Self {
            launches: launch_count(),
            reads: read_count(),
            runtime_reads: runtime_read_count(),
            upload_bytes: upload_bytes(),
            download_bytes: download_bytes(),
            allocs: allocation_calls(),
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            reserved: reserved_bytes(device),
        }
    }

    /// This boundary minus the previous one: the stage between them.
    fn delta(&self, prev: &Self) -> Self {
        Self {
            launches: self.launches - prev.launches,
            reads: self.reads - prev.reads,
            runtime_reads: self.runtime_reads - prev.runtime_reads,
            upload_bytes: self.upload_bytes - prev.upload_bytes,
            download_bytes: self.download_bytes - prev.download_bytes,
            allocs: self.allocs - prev.allocs,
            wall_ms: (self.wall_ms - prev.wall_ms).max(0.0),
            reserved: None,
        }
    }
}

/// Drain the device, zero the counters and start the wall clock: every
/// isolated timer (hook windows and production repeats alike) starts here so
/// no previously queued work is charged to it.
fn open_window(device: &Device<R>) -> Instant {
    device.synchronize();
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    Instant::now()
}

/// One host-timed stage object: counter deltas with the synchronised host
/// wall clock.
///
/// `timer` is always [`SYNC_WALL_TIMER`]: the wall time is a host `Instant`
/// around the stage plus its synchronisation. The runtime's profiling
/// capability (probed from what `client.profile` returns) travels in
/// `timing_method`, never conflated with the clock. In device mode the same
/// object additionally carries `profile_ms`/`timing_method` from the
/// harness span.
fn stage_json(
    label: &str,
    delta: &Boundary,
    capability: &str,
    note: &str,
) -> serde_json::Value {
    serde_json::json!({
        "stage": label,
        "launches": delta.launches,
        "reads": delta.reads,
        "runtime_reads": delta.runtime_reads,
        "upload_bytes": delta.upload_bytes,
        "download_bytes": delta.download_bytes,
        "allocation_calls": delta.allocs,
        "profile_ms": "unavailable",
        "profile_note": "host mode: synchronised wall clock (see sync_wall_ms); per-stage device spans need --profile-mode device",
        "sync_wall_ms": delta.wall_ms,
        "timer": SYNC_WALL_TIMER,
        "timing_method": capability,
        "window": "profiled (P2.9: separate from production)",
        "note": note,
    })
}

/// A byte count the runtime reports, or the JSON string `"unavailable"`
/// when it stays silent — never 0 for an unreported value.
fn json_bytes_or_unavailable(bytes: Option<u64>) -> serde_json::Value {
    match bytes {
        Some(b) => serde_json::Value::from(b),
        None => serde_json::Value::from("unavailable"),
    }
}

fn main() {
    let mut mode = "both".to_string();
    let mut ns = vec![64usize, 128, 256, 512];
    let mut bs = vec![1usize, 8, 32];
    let mut k: u32 = 8;
    let mut warmup: usize = 2;
    let mut repeats: usize = 10;
    let mut stability: usize = 0;
    let mut profile_mode = "host".to_string();
    let mut max_device_bytes: u64 = 4 * 1024 * 1024 * 1024;
    let mut out: Option<String> = None;
    let mut formula_source = mamba3::models::ms2::contract::FormulaSource::Table;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--mode" => mode = next(),
            "--n" => ns = parse_list(&next()),
            "--b" => bs = parse_list(&next()),
            "--k" => k = next().parse().unwrap_or_else(|_| usage()),
            "--warmup" => warmup = next().parse().unwrap_or_else(|_| usage()),
            "--repeats" => repeats = next().parse().unwrap_or_else(|_| usage()),
            "--stability" => stability = next().parse().unwrap_or_else(|_| usage()),
            "--profile-mode" => profile_mode = next(),
            "--max-device-bytes" => max_device_bytes = next().parse().unwrap_or_else(|_| usage()),
            "--formula-source" => {
                formula_source = match next().as_str() {
                    "table" => mamba3::models::ms2::contract::FormulaSource::Table,
                    "enumerate" => mamba3::models::ms2::contract::FormulaSource::Enumerate,
                    _ => usage(),
                };
            }
            "--out" => out = Some(next()),
            _ => usage(),
        }
    }
    if !["generate", "train", "both"].contains(&mode.as_str()) {
        usage();
    }
    if profile_mode == "device-sum" {
        eprintln!(
            "profile_ms2_substructure: --profile-mode device-sum is not supported: the production stage functions expose no launch-group decomposition (each generate_*_ws stage is one closure), so per-group spans would change the measurement without becoming whole-stage elapsed time (finding A1; see header)"
        );
        std::process::exit(2);
    }
    if !["host", "device"].contains(&profile_mode.as_str()) {
        eprintln!("profile_ms2_substructure: --profile-mode must be host|device, got {profile_mode}");
        std::process::exit(2);
    }
    let Some(out) = out else { usage() };
    for &n in &ns {
        if !matches!(n, 64 | 128 | 256 | 512) {
            eprintln!("profile_ms2_substructure: --n values must be in 64,128,256,512, got {n}");
            std::process::exit(2);
        }
    }

    let device = Device::<R>::default();
    let caps = Ms2Capabilities::probe(&device);
    let timing = timing_name(&device);
    let backend = device.name().to_string();
    let os = std::env::consts::OS.to_string();
    // The runtime's own identity string when it exposes one: CubeCL 0.10's
    // `ComputeClient::properties` carries features, memory, hardware and the
    // timing method but no adapter/driver name; `client.info()` is `()` on
    // the CPU runtime and the graphics backend enum on wgpu.
    let info = format!("{:?}", device.client().info());
    let adapter = if info == "()" {
        serde_json::Value::from("unavailable")
    } else {
        serde_json::Value::from(info)
    };
    println!("backend: {backend} timing: {timing} caps: {caps:?}");
    let (base_comps, from_fixture) = fixture_comps();
    let host_table_small = FormulaTable::from_compositions(base_comps.clone().into_iter())
        .expect("small table builds");

    let mut records = Vec::new();
    let modes: Vec<&str> = match mode.as_str() {
        "generate" => vec!["generate"],
        "train" => vec!["train"],
        _ => vec!["generate", "train"],
    };
    for m in modes {
        for &n in &ns {
            for &b in &bs {
                if m == "generate" {
                    records.push(profile_generate(
                        &device,
                        &base_comps,
                        &host_table_small,
                        n,
                        b,
                        k,
                        warmup,
                        repeats,
                        max_device_bytes,
                        timing,
                        &profile_mode,
                        formula_source,
                    ));
                } else {
                    records.push(profile_train(
                        &device,
                        &base_comps,
                        &host_table_small,
                        n,
                        b,
                        warmup,
                        repeats,
                        max_device_bytes,
                        timing,
                        &profile_mode,
                        formula_source,
                    ));
                }
            }
        }
    }

    // The P2.3 serving measurement, once per peak capacity: identical calls
    // then an alternating-bucket sequence, in its own window after the
    // production numbers.
    let stability_report = if stability > 0 && mode != "train" {
        let mut entries = Vec::new();
        for &n in &ns {
            entries.push(profile_stability(
                &device,
                &base_comps,
                &host_table_small,
                n,
                &bs,
                k,
                stability,
                max_device_bytes,
                timing,
                formula_source,
            ));
        }
        serde_json::Value::Array(entries)
    } else {
        serde_json::json!({
            "skipped": if stability == 0 {
                "stability is 0 (default: skip)"
            } else {
                "stability is a generation measurement; --mode train runs none"
            },
        })
    };

    let report = serde_json::json!({
        "machine": {
            "backend": backend,
            "device": backend,
            "adapter": adapter,
            "adapter_note": "CubeCL 0.10 exposes no adapter/driver string via ComputeClient properties (features, memory, hardware, timing method only); client.info() is () on the CPU runtime, so the adapter is unavailable there",
            "os": os,
            "timing_method": timing,
            "timing_note": "runtime profiling capability probed from what client.profile returns (device timestamps vs system time); every stage's sync_wall_ms is a synchronised host wall clock (timer SynchronizedHostWallClock), and profile_ms is unavailable in host mode (see header)",
            "profile_mode": profile_mode,
        },
        "config": {
            "mode": mode,
            "n": ns,
            "b": bs,
            "k": k,
            "warmup": warmup,
            "repeats": repeats,
            "stability": stability,
            "profile_mode": profile_mode,
            "max_device_bytes": max_device_bytes,
            "weights": "random-init (seeded); timings are structural, not quality numbers",
            "spectra": if from_fixture {
                "synthetic peaks (seeded RNG) over parent compositions from tests/fixtures/ms2/chemistry_v0.json"
            } else {
                "synthetic peaks (seeded RNG) over fallback parent compositions (fixture unreadable)"
            },
            "memory_snapshot_bytes_reserved": memory_snapshot(&device).map(|s| s.bytes_reserved),
        },
        "records": records,
        "stability": stability_report,
    });
    if let Some(parent) = std::path::Path::new(&out).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&report).expect("report serializes"),
    )
    .expect("write report");
    println!("wrote {}", out);
    // The launch budget must reconcile exactly: any mismatch means a stage
    // is misattributed, and the profile is refused rather than reported.
    let mut reconciled = true;
    for record in &records {
        if let Some(diff) = record
            .get("launch_budget")
            .and_then(|b| b.get("measured_minus_L_call"))
            .and_then(|d| d.as_i64())
            && diff != 0
        {
            eprintln!(
                "profile_ms2_substructure: launch budget mismatch (measured − L_call = {diff}) in record {}",
                record.get("mode").and_then(|m| m.as_str()).unwrap_or("?")
            );
            reconciled = false;
        }
    }
    if !reconciled {
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn profile_generate(
    device: &Device<R>,
    base_comps: &[Composition],
    host_table: &FormulaTable,
    n: usize,
    b: usize,
    k: u32,
    warmup: usize,
    repeats: usize,
    max_device_bytes: u64,
    timing: &str,
    profile_mode: &str,
    formula_source: mamba3::models::ms2::contract::FormulaSource,
) -> serde_json::Value {
    let model_config = ModelConfig::v0();
    // Finding A2: with Enumerate the artifacts are fitted on the host FIRST
    // and every estimate below sizes them in (`generation_with_enum`), so a
    // limit between the table-only and the enumeration-inclusive estimate
    // refuses here instead of panicking in the cold call's `expect`.
    let is_enumerate = matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    );
    let (enum_p, enum_bounds_words) = if is_enumerate {
        enum_host_sizes(base_comps)
    } else {
        (0, 0)
    };
    let estimate = if is_enumerate {
        Ms2MemoryEstimate::generation_with_enum(
            &model_config,
            host_table.len() as u64,
            b as u64,
            k as u64,
            n as u64,
            22,
            32,
            4,
            enum_p,
            enum_bounds_words,
        )
    } else {
        Ms2MemoryEstimate::generation(
            &model_config,
            host_table.len() as u64,
            b as u64,
            k as u64,
            n as u64,
            22,
            32,
            4,
        )
    };
    let estimate_total = estimate.as_ref().ok().and_then(|est| est.total().ok());
    let estimate_items = match &estimate {
        Ok(est) => est
            .items
            .iter()
            .map(|(name, bytes)| (name.to_string(), serde_json::Value::from(*bytes)))
            .collect::<serde_json::Map<String, serde_json::Value>>(),
        Err(_) => serde_json::Map::new(),
    };
    let estimate_bytes = estimate_total.unwrap_or(u64::MAX);
    if estimate_total
        .map(|t| t > max_device_bytes)
        .unwrap_or(true)
    {
        return serde_json::json!({
            "mode": "generate", "n": n, "b": b, "k": k,
            "status": "refused",
            "estimate_bytes": estimate_bytes,
            "estimate_items": estimate_items,
            "max_device_bytes": max_device_bytes,
            "timing_method": timing,
            "profile_mode": profile_mode,
            "warmup": warmup,
            "repeats": repeats,
        });
    }

    let table = DeviceFormulaTable::<R, E>::upload(host_table, device).expect("table uploads");
    let mut config = ModelConfig::v0();
    config.formula_table.rows = table.rows as u32;
    config.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(1000 + n as u64 * 131 + b as u64);
    let mut model = Ms2Model::<R, E>::init(&config, device, &mut rng).expect("model inits");
    // Enumerating source: fit the domain on the synthetic fixture
    // compositions, so the search stage's launches and time are measured
    // for this source too. The fit is the same host-side fit the base
    // estimate above already sized in (finding A2).
    if is_enumerate {
        let domain =
            mamba3::models::ms2::formula_enum::EnumDomain::from_compositions(
                base_comps.iter().copied(),
                0,
            )
            .expect("enum domain fits on fixture comps");
        let bounds = mamba3::models::ms2::formula_enum::RatioBounds::fit(
            base_comps.iter().copied(),
            0,
        )
        .expect("ratio bounds fit on fixture comps");
        model
            .upload_enum_artifacts(&domain, &bounds, device)
            .expect("enum artifacts upload");
    }
    let constants = Ms2Constants::new(device);
    let comps: Vec<Composition> = (0..b).map(|i| base_comps[i % base_comps.len()]).collect();
    let batch = spectra_batch(&comps, n, 5000 + n as u64, 7000);
    // The requested limit reaches every generation configuration this
    // profile builds: the production call, the T+1 slope configuration and
    // (below) the device-mode session, which clones this config.
    let gen_config = GenerationConfig {
        trajectories: k,
        max_device_bytes,
        formula_source,
        ..GenerationConfig::default()
    };
    let steps = gen_config.max_steps as usize;
    let decode_steps = (steps - 1).max(1);
    let mut workspace = GenerationWorkspace::new();

    // Cold first call (production window).
    let started = open_window(device);
    let out = model
        .generate(&batch, &table, &gen_config, &mut workspace, &constants)
        .expect("cold generate runs");
    out.validate().expect("cold output validates");
    device.synchronize();
    let cold_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cold = serde_json::json!({
        "wall_ms": cold_ms,
        "launches": launch_count(),
        "reads": read_count(),
        "runtime_reads": runtime_read_count(),
        "upload_bytes": upload_bytes(),
        "download_bytes": download_bytes(),
        "allocation_calls": allocation_calls(),
    });

    // Warm-up (untimed), then timed repeats with a device sync per call.
    for _ in 0..warmup {
        let out = model
            .generate(&batch, &table, &gen_config, &mut workspace, &constants)
            .expect("warmup runs");
        out.validate().expect("warmup validates");
    }
    device.synchronize();
    let mut walls = Vec::with_capacity(repeats);
    let mut reads = Vec::with_capacity(repeats);
    let mut launches = Vec::with_capacity(repeats);
    let mut allocs = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = open_window(device);
        let out = model
            .generate(&batch, &table, &gen_config, &mut workspace, &constants)
            .expect("timed generate runs");
        out.validate().expect("timed output validates");
        device.synchronize();
        walls.push(started.elapsed().as_secs_f64() * 1000.0);
        reads.push(read_count());
        launches.push(launch_count());
        allocs.push(allocation_calls());
    }
    let (p50, p95) = p50p95(walls.clone());
    let peak_reserved = reserved_bytes(device);

    // Stage profiling (separate P2.9 window): one instrumented call runs
    // exactly what production runs, and the hook snapshots counters and
    // synchronised wall time at each real boundary. The window opens with a
    // drain, so no previously queued work is charged to a stage; each hook
    // hit synchronises in turn, so a stage's wall time is its own kernels
    // plus one sync. No subtraction, no clamping: preprocess, encoder,
    // search, decoder initialisation, each decode step, validate and readout
    // each have measured launches, allocations and wall time.
    let started = open_window(device);
    let mut bounds: Vec<(GenerateStage, Boundary)> = Vec::new();
    {
        let mut hook = |stage: GenerateStage| {
            bounds.push((stage, Boundary::snapshot(device, &started)));
        };
        let out = model
            .generate_with_hook(
                &batch,
                &table,
                &gen_config,
                &mut workspace,
                &constants,
                Some(&mut hook),
            )
            .expect("instrumented generate runs");
        out.validate().expect("instrumented output validates");
    }
    let final_bound = Boundary::snapshot(device, &started);
    // Boundaries in pipeline order: preprocess, encoder, search, decoder
    // init, one per decode step, validate, readout (the readout hook fires
    // right after the single batched read, closed out by the final snapshot).
    let n_bounds = 4 + decode_steps + 2;
    if bounds.len() != n_bounds
        || bounds[4 + decode_steps].0 != GenerateStage::AfterValidate
        || bounds[5 + decode_steps].0 != GenerateStage::AfterReadout
    {
        eprintln!(
            "profile_ms2_substructure: expected {n_bounds} hook boundaries ending in AfterValidate, AfterReadout, got {:?}",
            bounds.iter().map(|b| b.0).collect::<Vec<_>>()
        );
        std::process::exit(1);
    }
    let zero = Boundary::default();
    let d_pre = bounds[0].1.delta(&zero);
    let d_enc = bounds[1].1.delta(&bounds[0].1);
    let d_search = bounds[2].1.delta(&bounds[1].1);
    let d_init = bounds[3].1.delta(&bounds[2].1);
    let mut d_steps: Vec<Boundary> = Vec::with_capacity(decode_steps);
    for i in 0..decode_steps {
        d_steps.push(bounds[4 + i].1.delta(&bounds[3 + i].1));
    }
    let d_validate = bounds[4 + decode_steps].1.delta(&bounds[3 + decode_steps].1);
    // The readout stage starts at the validate boundary, not the readout
    // hook: the single batched `read_all` runs between the two (validate
    // hook, then the read, then the readout hook), so measuring from the
    // readout hook would leave the read in no stage at all.
    let d_readout = final_bound.delta(&bounds[4 + decode_steps].1);
    let loop_launches: usize = d_steps.iter().map(|d| d.launches).sum();
    let loop_allocs: usize = d_steps.iter().map(|d| d.allocs).sum();
    let loop_ms: f64 = d_steps.iter().map(|d| d.wall_ms).sum();
    let step_launch_vec: Vec<usize> = d_steps.iter().map(|d| d.launches).collect();
    // Per-decode-step allocation slope: one warmed call at T+1 steps (a new
    // bucket; steady-state allocation counts are deterministic, so two
    // warmups plus one measured call suffice) minus the warmed T-step call.
    // The first T-1 decode steps draw the same RNG stream as the T-step
    // call, so exactly one extra step's allocations remain.
    //
    // The T+1 configuration is preflighted with the same estimate before any
    // allocation, exactly like the base record: a refused slope records
    // itself and the per-step costs fall back to the hook decode-loop
    // average (the reconciliation then telescopes by construction, said so in
    // the note).
    let mut gen_config_long = gen_config.clone();
    gen_config_long.max_steps += 1;
    // Finding A2: the T+1 slope is preflighted with the same
    // enumeration-inclusive estimate as the base record.
    let slope_estimate = if is_enumerate {
        Ms2MemoryEstimate::generation_with_enum(
            &model_config,
            host_table.len() as u64,
            b as u64,
            u64::from(k),
            n as u64,
            gen_config_long.max_steps as u64,
            gen_config_long.formula_window as u64,
            gen_config_long.formulas as u64,
            enum_p,
            enum_bounds_words,
        )
        .and_then(|est| est.total())
    } else {
        Ms2MemoryEstimate::generation(
            &model_config,
            host_table.len() as u64,
            b as u64,
            u64::from(k),
            n as u64,
            gen_config_long.max_steps as u64,
            gen_config_long.formula_window as u64,
            gen_config_long.formulas as u64,
        )
        .and_then(|est| est.total())
    };
    let slope_refused = slope_estimate.as_ref().map(|t| *t > max_device_bytes).unwrap_or(true);
    let (long_launches, long_allocs, slope_note) = if slope_refused {
        (
            0usize,
            0usize,
            format!(
                "refused: the T+1 estimate ({}) exceeds --max-device-bytes ({max_device_bytes}); per-step costs are the hook decode-loop average, so the budget telescopes by construction",
                slope_estimate.unwrap_or(u64::MAX),
            ),
        )
    } else {
        for _ in 0..2 {
            let out = model
                .generate(&batch, &table, &gen_config_long, &mut workspace, &constants)
                .expect("slope warmup runs");
            out.validate().expect("slope warmup validates");
        }
        let _slope_started = open_window(device);
        let out_long = model
            .generate(&batch, &table, &gen_config_long, &mut workspace, &constants)
            .expect("slope call runs");
        out_long.validate().expect("slope output validates");
        device.synchronize();
        (
            launch_count(),
            allocation_calls(),
            "warmed T+1-step call minus the warmed T-step call".to_string(),
        )
    };
    let base_allocs: usize = allocs.first().copied().unwrap_or(0);
    let measured_total_launches: usize = launches.first().copied().unwrap_or(0);
    let (alloc_per_step, l_step) = if slope_refused {
        if loop_launches % decode_steps != 0 {
            eprintln!(
                "profile_ms2_substructure: refused slope needs an evenly divisible decode loop, got {loop_launches} launches over {decode_steps} steps"
            );
            std::process::exit(1);
        }
        (
            loop_allocs / decode_steps,
            loop_launches / decode_steps,
        )
    } else {
        let alloc_per_step = long_allocs.saturating_sub(base_allocs);
        // The per-step launch cost, every scope included: the warmed T+1-step
        // call minus the warmed T-step call, cross-checked against the hook's
        // decode-loop total below (they must agree exactly).
        let l_step = (long_launches as i64 - measured_total_launches as i64).max(0) as usize;
        (alloc_per_step, l_step)
    };
    // The launch budget (architecture §5): every term is a measured counter
    // delta — counter snapshots at the hook boundaries for preprocess,
    // encoder, search, decoder init (`L_init`: K/V projections, cache fills,
    // atom-memory fills) and validate, the warmed slope for `L_step` — so
    // `L_call` must equal the warmed-call total exactly.
    let l_preprocess = d_pre.launches;
    let l_encoder = d_enc.launches;
    let l_search = d_search.launches;
    let l_init = d_init.launches;
    let l_finalize = d_validate.launches;
    let l_call = l_preprocess + l_encoder + l_search + l_init + l_step * decode_steps + l_finalize;
    let l_diff = measured_total_launches as i64 - l_call as i64;
    // The warm decode loop is allocation-free only when the per-step slope
    // is zero. It is not (the mixer step and the composed attention allocate
    // their outputs functionally in V0), so this stays `false` until P8.2/O2
    // makes the loop reuse its outputs.
    let decoder_loop_allocation_free = alloc_per_step == 0;
    let per_step_ms = loop_ms / decode_steps as f64;
    let mut stages = vec![
        stage_json(
            "ms2.preprocess",
            &d_pre,
            timing,
            "request upload: DeviceSpectra buffers from the host batch",
        ),
        stage_json(
            "ms2.encoder",
            &d_enc,
            timing,
            "spectrum encoder over the selected peaks",
        ),
        stage_json(
            "ms2.search",
            &d_search,
            timing,
            "formula window, scoring, top-F, trajectory initialisation and formula broadcast",
        ),
        stage_json(
            "ms2.decoder_init",
            &d_init,
            timing,
            "decoder start state: K/V projections, cache fills, atom-memory fills (L_init of the launch budget)",
        ),
        serde_json::json!({
            "stage": "decode_steps_all",
            "launches": l_step * decode_steps,
            "reads": 0, "runtime_reads": 0,
            "upload_bytes": 0, "download_bytes": 0, "allocation_calls": alloc_per_step * decode_steps,
            "allocation_note": "per-step slope times T-1 steps (warmed T+1 minus T calls, or the hook decode-loop average when the slope configuration is refused: see launch_budget.slope_note); the hook's in-call decode-loop total must equal this (see launch_budget)",
            "decode_loop_launches_hook": loop_launches,
            "decode_loop_allocations_hook": loop_allocs,
            "decode_step_launches_hook": step_launch_vec,
            "profile_ms": "unavailable",
            "profile_note": "host mode: synchronised wall clock (see sync_wall_ms); per-stage device spans need --profile-mode device",
            "sync_wall_ms": loop_ms,
            "timer": SYNC_WALL_TIMER,
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "time is the directly measured decode loop (hook boundaries around the T-1 sampling steps); launches are the per-step slope times T-1, cross-checked against the hook total",
        }),
        serde_json::json!({
            "stage": "decode_per_step",
            "launches": l_step,
            "reads": 0, "runtime_reads": 0,
            "upload_bytes": 0, "download_bytes": 0, "allocation_calls": alloc_per_step,
            "allocation_note": "warmed allocation_calls slope over one extra decode step (T+1 minus T, or the hook decode-loop average when the slope configuration is refused: see launch_budget.slope_note)",
            "profile_ms": "unavailable",
            "profile_note": "host mode: synchronised wall clock (see sync_wall_ms); per-stage device spans need --profile-mode device",
            "sync_wall_ms": per_step_ms,
            "timer": SYNC_WALL_TIMER,
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "decode_steps_all divided by T-1 steps; the decode loop performs no reads or transfers, but it does allocate device buffers every step (see decoder_loop_allocation_free)",
        }),
        stage_json(
            "validate",
            &d_validate,
            timing,
            "trajectory validation kernel over the preallocated scratch (in-place status)",
        ),
        stage_json(
            "readout",
            &d_readout,
            timing,
            "the single batched read_all plus host-side batch building",
        ),
    ];
    // I2 readout modes (V1 §4.4): one warmed packed call and one warmed
    // resident call over the same request, measured in their own windows.
    // Allocation already ran inside the search stage above; identity would
    // run inside validate (this profile uses the default TraceOnly, so no
    // identity launch fires here); the pack window (AfterValidate to
    // AfterPack) holds exactly the six pack-stage launches (scores_fill,
    // allocate_window, rank, record_pack, record_pack_f, pack) under
    // ms2.finalize. The budget reconciles as L_packed = L_call + 6, the
    // resident call performs no read and its deferred read performs one.
    let readout_modes = {
        for _ in 0..warmup.max(1) {
            model
                .generate_packed(&batch, &table, &gen_config, &mut workspace, &constants)
                .expect("packed warmup runs")
                .validate()
                .expect("packed warmup validates");
            model
                .generate_resident(&batch, &table, &gen_config, &mut workspace, &constants)
                .expect("resident warmup runs")
                .release_into(&mut workspace);
        }
        let p_started = open_window(device);
        let mut p_bounds: Vec<(GenerateStage, Boundary)> = Vec::new();
        let packed = {
            let mut hook = |stage: GenerateStage| {
                p_bounds.push((stage, Boundary::snapshot(device, &p_started)));
            };
            model
                .generate_packed_with_hook(
                    &batch,
                    &table,
                    &gen_config,
                    &mut workspace,
                    &constants,
                    Some(&mut hook),
                )
                .expect("packed profile runs")
        };
        packed.validate().expect("packed profile validates");
        let p_final = Boundary::snapshot(device, &p_started);
        // The packed hook sequence is the generate sequence plus AfterPack
        // between AfterValidate and AfterReadout.
        let n_p_bounds = 4 + decode_steps + 3;
        assert!(
            p_bounds.len() == n_p_bounds
                && p_bounds[4 + decode_steps].0 == GenerateStage::AfterValidate
                && p_bounds[5 + decode_steps].0 == GenerateStage::AfterPack
                && p_bounds[6 + decode_steps].0 == GenerateStage::AfterReadout,
            "profile_ms2_substructure: packed hook boundaries must be the generate sequence plus AfterPack, got {:?}",
            p_bounds.iter().map(|b| b.0).collect::<Vec<_>>()
        );
        let pack_window = p_bounds[5 + decode_steps].1.delta(&p_bounds[4 + decode_steps].1);
        let packed_total = p_final.launches;
        let packed_reads = p_final.runtime_reads;
        assert_eq!(
            pack_window.launches, 6,
            "the pack window holds exactly the six pack-stage launches"
        );
        assert_eq!(
            packed_total,
            measured_total_launches + 6,
            "packed launches reconcile as L_call + 6"
        );
        assert_eq!(packed_reads, 1, "one warmed packed call performs one read");
        open_window(device);
        let resident = model
            .generate_resident(&batch, &table, &gen_config, &mut workspace, &constants)
            .expect("resident profile runs");
        device.synchronize();
        let resident_reads = runtime_read_count();
        assert_eq!(resident_reads, 0, "one warmed resident call performs no read");
        let deferred = resident.read(&model).expect("deferred read runs");
        deferred.validate().expect("deferred output validates");
        device.synchronize();
        let deferred_reads = runtime_read_count();
        assert_eq!(deferred_reads, 1, "the deferred read performs one read");
        let r_launches = launch_count();
        assert_eq!(
            r_launches,
            measured_total_launches + 6,
            "resident launches reconcile as L_call + 6"
        );
        resident.release_into(&mut workspace);
        serde_json::json!({
            "pack_window_launches": pack_window.launches,
            "pack_window_note": "AfterValidate to AfterPack: scores_fill, allocate_window, rank, record_pack, record_pack_f, pack (ms2.finalize scope)",
            "packed_total_launches": packed_total,
            "packed_reads": packed_reads,
            "resident_launches": r_launches,
            "resident_reads": resident_reads,
            "deferred_reads": deferred_reads,
            "reconciliation_note": "packed/resident launches reconcile as L_call + 6; reads are 1 / 0 + deferred 1",
        })
    };
    // Device mode: per-stage profile_ms from the runner-local harness with
    // the timing method of each returned ProfileDuration. Counts, wall
    // times and every other field are unchanged.
    if profile_mode == "device" {
        let device_spans = device_generate_spans(device, &config, host_table, &batch, &gen_config);
        // Order: preprocess, encoder, search, decoder_init, decode loop,
        // validate, readout; decode_per_step derives from the loop span.
        let slots = [0usize, 1, 2, 3, 4, 6, 7];
        for (slot, (ms, timer)) in slots.iter().zip(device_spans.iter()) {
            set_device_profile(&mut stages[*slot], *ms, timer);
        }
        let (loop_ms_dev, loop_timer) = device_spans[4];
        set_device_profile(&mut stages[5], loop_ms_dev / decode_steps as f64, loop_timer);
    }

    println!(
        "generate n={n} b={b} k={k}: cold {cold_ms:.1}ms warm p50/p95 {p50:.1}/{p95:.1}ms launches {launches:?} reads {reads:?}"
    );
    // The reserved-bytes high-water mark is sampled, not an endpoint: the
    // maximum over every hook boundary sample plus the production endpoint. A
    // transient high reservation during the call can exceed any single
    // endpoint, so the endpoint travels separately as `reserved_bytes_after`.
    let mut peak_sampled = peak_reserved;
    for (_, bound) in bounds.iter().chain(std::iter::once(&(
        GenerateStage::AfterReadout,
        final_bound,
    ))) {
        match (peak_sampled, bound.reserved) {
            (Some(a), Some(b)) => peak_sampled = Some(a.max(b)),
            (None, b) => peak_sampled = b,
            _ => {}
        }
    }
    serde_json::json!({
        "mode": "generate", "n": n, "b": b, "k": k,
        "status": "ok",
        "estimate_bytes": estimate_bytes,
        "estimate_items": estimate_items,
        "max_device_bytes": max_device_bytes,
        "device_pass_task_limit": device_pass_task_limit(),
        "warmup": warmup,
        "repeats": repeats,
        "cold_wall_ms": cold_ms,
        "cold": cold,
        "warm_wall_ms": walls,
        "warm_p50_ms": p50, "warm_p95_ms": p95,
        "warm_reads_per_call": reads,
        "warm_launches_per_call": launches,
        "warm_allocation_calls_per_call": allocs,
        "reserved_bytes_after": json_bytes_or_unavailable(peak_reserved),
        "peak_reserved_bytes_sampled": json_bytes_or_unavailable(peak_sampled),
        "reserved_note": "reserved_bytes_after is the production endpoint; peak_reserved_bytes_sampled is the high-water mark over every hook boundary sample of the instrumented call",
        "timing_method": timing,
        "timer": SYNC_WALL_TIMER,
        "profile_mode": profile_mode,
        "decoder_loop_allocation_free": decoder_loop_allocation_free,
        "allocation_per_decode_step_slope": alloc_per_step,
        "allocation_note": "the warmed decode loop allocates device buffers every step in V0 (mixer step and composed attention allocate their outputs functionally); the P2 acceptance item \"workspace allocation happens outside the decoder hot loop\" is NOT met in V0 and stays open for P8.2/O2 work",
        "launch_budget": {
            "L_preprocess": l_preprocess,
            "L_encoder": l_encoder,
            "L_search": l_search,
            "L_init": l_init,
            "L_step": l_step,
            "L_finalize": l_finalize,
            "L_call": l_call,
            "L_loop_hook": loop_launches,
            "slope_refused": slope_refused,
            "slope_note": slope_note,
            "measured_total_launches": measured_total_launches,
            "measured_minus_L_call": l_diff,
            "reconciliation_note": "L_call sums the stage costs of one warmed call: preprocess/encoder/search/decoder-init/validate from their hook counter windows (every scope included), L_step from the warmed T+1-minus-T launch slope cross-checked against the hook decode-loop total (L_loop_hook). The difference against the warmed-call total must be 0; the driver exits non-zero otherwise",
        },
        "readout_modes": readout_modes,
        "stages": stages,
    })
}

#[allow(clippy::too_many_arguments)]
fn profile_train(
    device: &Device<R>,
    base_comps: &[Composition],
    host_table: &FormulaTable,
    n: usize,
    b: usize,
    warmup: usize,
    repeats: usize,
    max_device_bytes: u64,
    timing: &str,
    profile_mode: &str,
    formula_source: mamba3::models::ms2::contract::FormulaSource,
) -> serde_json::Value {
    let model_config = ModelConfig::v0();
    let slots = 2usize;
    // Finding A2: with Enumerate the artifacts are fitted on the host FIRST
    // and the estimate sizes them in (`training_with_enum`), so a limit
    // between the table-only and the enumeration-inclusive estimate refuses
    // here instead of failing later.
    let is_enumerate = matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    );
    let (enum_p, enum_bounds_words) = if is_enumerate {
        enum_host_sizes(base_comps)
    } else {
        (0, 0)
    };
    let estimate = if is_enumerate {
        Ms2MemoryEstimate::training_with_enum(
            &model_config,
            host_table.len() as u64,
            b as u64,
            slots as u64,
            n as u64,
            22,
            32,
            enum_p,
            enum_bounds_words,
        )
    } else {
        Ms2MemoryEstimate::training(
            &model_config,
            host_table.len() as u64,
            b as u64,
            slots as u64,
            n as u64,
            22,
            32,
        )
    };
    let estimate_total = estimate.as_ref().ok().and_then(|est| est.total().ok());
    let estimate_items = match &estimate {
        Ok(est) => est
            .items
            .iter()
            .map(|(name, bytes)| (name.to_string(), serde_json::Value::from(*bytes)))
            .collect::<serde_json::Map<String, serde_json::Value>>(),
        Err(_) => serde_json::Map::new(),
    };
    let fits = estimate_total
        .as_ref()
        .map(|t| *t <= max_device_bytes)
        .unwrap_or(false);
    if !fits {
        return serde_json::json!({
            "mode": "train", "n": n, "b": b, "slots": slots,
            "status": "refused",
            "estimate_bytes": estimate_total.unwrap_or(u64::MAX),
            "estimate_items": estimate_items,
            "max_device_bytes": max_device_bytes,
            "timing_method": timing,
            "profile_mode": profile_mode,
            "warmup": warmup,
            "repeats": repeats,
        });
    }
    let estimate_bytes = estimate_total.unwrap_or(0);
    let train_config = TrainConfig {
        batch: b,
        slots,
        formula_source,
                    lambda_assign: 0.0,
..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&model_config, host_table, &train_config, device)
        .expect("trainer builds");
    if matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) {
        let domain =
            mamba3::models::ms2::formula_enum::EnumDomain::from_compositions(
                base_comps.iter().copied(),
                0,
            )
            .expect("enum domain fits on fixture comps");
        let bounds = mamba3::models::ms2::formula_enum::RatioBounds::fit(
            base_comps.iter().copied(),
            0,
        )
        .expect("ratio bounds fit on fixture comps");
        trainer
            .upload_enum_artifacts(&domain, &bounds)
            .expect("enum artifacts upload");
    }
    let comps: Vec<Composition> = (0..b).map(|i| base_comps[i % base_comps.len()]).collect();
    let set = experiment_set(&comps, n, 6000 + n as u64);
    let indices: Vec<usize> = (0..b).collect();

    // Cold first step (production window; no report read on the cold call).
    let started = open_window(device);
    let _ = trainer.step(&set, &indices).expect("cold step runs");
    device.synchronize();
    let cold_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cold = serde_json::json!({
        "wall_ms": cold_ms,
        "launches": launch_count(),
        "reads": read_count(),
        "runtime_reads": runtime_read_count(),
        "upload_bytes": upload_bytes(),
        "download_bytes": download_bytes(),
        "allocation_calls": allocation_calls(),
    });

    for _ in 0..warmup {
        let _ = trainer.step(&set, &indices).expect("warmup runs");
    }
    device.synchronize();
    let mut walls = Vec::with_capacity(repeats);
    let mut launches = Vec::with_capacity(repeats);
    let mut allocs = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = open_window(device);
        let _ = trainer.step(&set, &indices).expect("timed step runs");
        device.synchronize();
        walls.push(started.elapsed().as_secs_f64() * 1000.0);
        launches.push(launch_count());
        allocs.push(allocation_calls());
    }
    let (p50, p95) = p50p95(walls.clone());
    let peak_reserved = reserved_bytes(device);

    // Stage profiling (separate P2.9 window): one instrumented step runs
    // exactly what production runs, and the hook snapshots counters and
    // synchronised wall time at the three real boundaries — after the
    // forward/loss build, after the backward pass, after the optimizer
    // update. No report is requested, so no stage reads the device. The old
    // `step − teacher_eval` subtraction is gone: the teacher pass performs
    // evaluation-only work and a device read the training forward never
    // does, so it could not isolate backward work.
    let started = open_window(device);
    let mut bounds: Vec<(StepPhase, Boundary)> = Vec::new();
    {
        let mut hook = |phase: StepPhase| {
            bounds.push((phase, Boundary::snapshot(device, &started)));
        };
        let _ = trainer
            .step_with_boundaries(&set, &indices, &mut hook)
            .expect("instrumented step runs");
    }
    let final_bound = Boundary::snapshot(device, &started);
    if bounds.len() != 3
        || bounds[0].0 != StepPhase::AfterForward
        || bounds[1].0 != StepPhase::AfterBackward
        || bounds[2].0 != StepPhase::AfterOptimizer
    {
        eprintln!(
            "profile_ms2_substructure: expected 3 train hook boundaries, got {:?}",
            bounds.iter().map(|b| b.0).collect::<Vec<_>>()
        );
        std::process::exit(1);
    }
    let zero = Boundary::default();
    let d_forward = bounds[0].1.delta(&zero);
    let d_backward = bounds[1].1.delta(&bounds[0].1);
    let d_optimizer = bounds[2].1.delta(&bounds[1].1);
    let d_close = final_bound.delta(&bounds[2].1);
    let mut stages = vec![
        stage_json(
            "forward",
            &d_forward,
            timing,
            "training forward pass with loss build and report packing (no report read: no report requested)",
        ),
        stage_json(
            "backward",
            &d_backward,
            timing,
            "gradient computation with the device-side clip scale",
        ),
        stage_json(
            "optimizer",
            &d_optimizer,
            timing,
            "AdamW update over every model parameter (measured, not zero)",
        ),
    ];
    // The close-out after the optimizer boundary enqueues nothing and reads
    // nothing without a pending report; anything it ever shows is a bug in
    // the phase split, surfaced here rather than hidden.
    stages[2]["close_out"] = serde_json::json!({
        "launches": d_close.launches,
        "reads": d_close.reads,
        "allocation_calls": d_close.allocs,
    });
    // Device mode: per-stage profile_ms from the runner-local harness with
    // the timing method of each returned ProfileDuration. Counts, wall
    // times and every other field are unchanged.
    if profile_mode == "device" {
        let device_spans = device_train_spans(
            device,
            &model_config,
            host_table,
            &comps,
            n,
            &train_config,
        );
        for (stage, (ms, timer)) in stages.iter_mut().zip(device_spans.iter()) {
            set_device_profile(stage, *ms, timer);
        }
    }

    println!("train n={n} b={b}: cold {cold_ms:.1}ms warm p50/p95 {p50:.1}/{p95:.1}ms");
    // Sampled high-water mark over the three phase boundaries plus the
    // production endpoint (see the generate record: same convention).
    let mut peak_sampled = peak_reserved;
    for (_, bound) in bounds.iter().chain(std::iter::once(&(
        StepPhase::AfterOptimizer,
        final_bound,
    ))) {
        match (peak_sampled, bound.reserved) {
            (Some(a), Some(b)) => peak_sampled = Some(a.max(b)),
            (None, b) => peak_sampled = b,
            _ => {}
        }
    }
    serde_json::json!({
        "mode": "train", "n": n, "b": b, "slots": slots,
        "status": "ok",
        "estimate_bytes": estimate_bytes,
        "estimate_items": estimate_items,
        "max_device_bytes": max_device_bytes,
        "device_pass_task_limit": device_pass_task_limit(),
        "warmup": warmup,
        "repeats": repeats,
        "cold_wall_ms": cold_ms,
        "cold": cold,
        "warm_wall_ms": walls,
        "warm_p50_ms": p50, "warm_p95_ms": p95,
        "warm_launches_per_call": launches,
        "warm_allocation_calls_per_call": allocs,
        "reserved_bytes_after": json_bytes_or_unavailable(peak_reserved),
        "peak_reserved_bytes_sampled": json_bytes_or_unavailable(peak_sampled),
        "reserved_note": "reserved_bytes_after is the production endpoint; peak_reserved_bytes_sampled is the high-water mark over every hook boundary sample of the instrumented step",
        "timing_method": timing,
        "timer": SYNC_WALL_TIMER,
        "profile_mode": profile_mode,
        "stages": stages,
    })
}

/// The P2.3 serving measurement (`--stability <calls>`): after warmup, run
/// `calls` identical `generate` calls, then `calls` calls alternating over
/// the first two `--b` values, recording the reserved-bytes series
/// (first/last/min/max over the calls), allocation calls per call (min/max)
/// and bucket counts captured at each phase boundary. Its own window after
/// the production numbers, with counters reset around every call; it changes
/// no behaviour.
///
/// The tested configurations are preflighted with the same
/// `Ms2MemoryEstimate` as the profile records before any allocation, so
/// `--max-device-bytes` refuses here exactly as it refuses a profile record.
#[allow(clippy::too_many_arguments)]
fn profile_stability(
    device: &Device<R>,
    base_comps: &[Composition],
    host_table: &FormulaTable,
    n: usize,
    bs: &[usize],
    k: u32,
    calls: usize,
    max_device_bytes: u64,
    timing: &str,
    formula_source: mamba3::models::ms2::contract::FormulaSource,
) -> serde_json::Value {
    fn min_max(values: &[usize]) -> (usize, usize) {
        values
            .iter()
            .fold((usize::MAX, 0usize), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }

    /// A reserved-bytes series summary: first/last/min/max over the calls,
    /// never a single endpoint.
    fn reserved_series(values: &[Option<u64>]) -> serde_json::Value {
        let present: Vec<u64> = values.iter().filter_map(|v| *v).collect();
        serde_json::json!({
            "first": values.first().copied().flatten().map_or(serde_json::Value::from("unavailable"), serde_json::Value::from),
            "last": values.last().copied().flatten().map_or(serde_json::Value::from("unavailable"), serde_json::Value::from),
            "min": present.iter().min().copied().map_or(serde_json::Value::from("unavailable"), serde_json::Value::from),
            "max": present.iter().max().copied().map_or(serde_json::Value::from("unavailable"), serde_json::Value::from),
            "per_call": values.iter().map(|v| v.map_or(serde_json::Value::from("unavailable"), serde_json::Value::from)).collect::<Vec<_>>(),
        })
    }

    // Finding A2: the stability preflight uses the same
    // enumeration-inclusive estimate as the profile records (fitted on the
    // host first), so `--max-device-bytes` refuses here exactly as it
    // refuses a profile record.
    let is_enumerate = matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    );
    let (enum_p, enum_bounds_words) = if is_enumerate {
        enum_host_sizes(base_comps)
    } else {
        (0, 0)
    };
    fn stability_estimate(
        host_table: &FormulaTable,
        b: usize,
        k: u32,
        n: usize,
        is_enumerate: bool,
        enum_p: u64,
        enum_bounds_words: u64,
    ) -> Result<u64, String> {
        if is_enumerate {
            Ms2MemoryEstimate::generation_with_enum(
                &ModelConfig::v0(),
                host_table.len() as u64,
                b as u64,
                u64::from(k),
                n as u64,
                22,
                32,
                4,
                enum_p,
                enum_bounds_words,
            )
            .and_then(|est| est.total())
            .map_err(|e| e.to_string())
        } else {
            Ms2MemoryEstimate::generation(
                &ModelConfig::v0(),
                host_table.len() as u64,
                b as u64,
                u64::from(k),
                n as u64,
                22,
                32,
                4,
            )
            .and_then(|est| est.total())
            .map_err(|e| e.to_string())
        }
    }

    let b0 = bs[0];
    // Preflight before any allocation: refuse exactly like a profile record
    // when the estimate exceeds the limit (both the identical shape and, for
    // the alternating phase, the second shape).
    let mut refused: Option<(usize, u64)> = None;
    for &b in bs.iter().take(if bs.len() >= 2 { 2 } else { 1 }) {
        match stability_estimate(host_table, b, k, n, is_enumerate, enum_p, enum_bounds_words) {
            Ok(total) if total <= max_device_bytes => {}
            Ok(total) => {
                refused = Some((b, total));
                break;
            }
            Err(_) => {
                refused = Some((b, u64::MAX));
                break;
            }
        }
    }
    if let Some((b, estimate_bytes)) = refused {
        return serde_json::json!({
            "n": n, "k": k, "calls": calls,
            "status": "refused",
            "b": b,
            "estimate_bytes": estimate_bytes,
            "max_device_bytes": max_device_bytes,
            "timing_method": timing,
        });
    }
    let table = DeviceFormulaTable::<R, E>::upload(host_table, device).expect("table uploads");
    let mut config = ModelConfig::v0();
    config.formula_table.rows = table.rows as u32;
    config.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(2000 + n as u64 * 131);
    let mut model = Ms2Model::<R, E>::init(&config, device, &mut rng).expect("model inits");
    // With Enumerate the stability run needs the same resident artifacts as
    // production (fitted above for the estimate); without them the warmed
    // calls would fail after the preflight admitted them (finding A2).
    if is_enumerate {
        let domain =
            mamba3::models::ms2::formula_enum::EnumDomain::from_compositions(
                base_comps.iter().copied(),
                0,
            )
            .expect("enum domain fits on fixture comps");
        let bounds = mamba3::models::ms2::formula_enum::RatioBounds::fit(
            base_comps.iter().copied(),
            0,
        )
        .expect("ratio bounds fit on fixture comps");
        model
            .upload_enum_artifacts(&domain, &bounds, device)
            .expect("enum artifacts upload");
    }
    let constants = Ms2Constants::new(device);
    let gen_config = GenerationConfig {
        trajectories: k,
        max_device_bytes,
        formula_source,
        ..GenerationConfig::default()
    };
    let batch0 = {
        let comps: Vec<Composition> =
            (0..b0).map(|i| base_comps[i % base_comps.len()]).collect();
        spectra_batch(&comps, n, 8000 + n as u64, 7000)
    };
    let mut workspace = GenerationWorkspace::new();
    for _ in 0..2 {
        let out = model
            .generate(&batch0, &table, &gen_config, &mut workspace, &constants)
            .expect("stability warmup runs");
        out.validate().expect("stability warmup validates");
    }
    device.synchronize();
    // Identical calls on the warmed bucket; bucket count and reserved bytes
    // are captured at this phase's boundaries.
    let identical_buckets_before = workspace.bucket_keys().len();
    let mut identical_allocs = Vec::with_capacity(calls);
    let mut identical_reserved = Vec::with_capacity(calls);
    for _ in 0..calls {
        reset_launch_count();
        reset_read_count();
        reset_transfer_counters();
        let out = model
            .generate(&batch0, &table, &gen_config, &mut workspace, &constants)
            .expect("stability call runs");
        out.validate().expect("stability output validates");
        identical_allocs.push(allocation_calls());
        identical_reserved.push(reserved_bytes(device));
    }
    device.synchronize();
    let identical_buckets_after = workspace.bucket_keys().len();
    let (identical_min, identical_max) = min_max(&identical_allocs);

    // Alternating buckets over the first two `--b` values, when there are
    // two (a second bucket is cached past the first alternating warmup while
    // memory stays bounded: at most 4 buckets are kept).
    let alternating = if bs.len() >= 2 {
        let b1 = bs[1];
        let batch1 = {
            let comps: Vec<Composition> =
                (0..b1).map(|i| base_comps[i % base_comps.len()]).collect();
            spectra_batch(&comps, n, 8100 + n as u64, 7100)
        };
        for _ in 0..2 {
            let out = model
                .generate(&batch1, &table, &gen_config, &mut workspace, &constants)
                .expect("alternating warmup runs");
            out.validate().expect("alternating warmup validates");
        }
        device.synchronize();
        let alternating_buckets_before = workspace.bucket_keys().len();
        let mut allocs = Vec::with_capacity(calls);
        let mut reserved = Vec::with_capacity(calls);
        for i in 0..calls {
            let batch = if i % 2 == 0 { &batch0 } else { &batch1 };
            reset_launch_count();
            reset_read_count();
            reset_transfer_counters();
            let out = model
                .generate(batch, &table, &gen_config, &mut workspace, &constants)
                .expect("alternating call runs");
            out.validate().expect("alternating output validates");
            allocs.push(allocation_calls());
            reserved.push(reserved_bytes(device));
        }
        device.synchronize();
        let (min, max) = min_max(&allocs);
        serde_json::json!({
            "b": [b0, b1],
            "calls": calls,
            "reserved_bytes_series": reserved_series(&reserved),
            "allocation_calls_per_call": allocs,
            "allocation_calls_min": min,
            "allocation_calls_max": max,
            "bucket_count_before": alternating_buckets_before,
            "bucket_count_after": workspace.bucket_keys().len(),
        })
    } else {
        serde_json::json!({
            "skipped": "a single --b value: no second bucket to alternate with",
            "bucket_count": workspace.bucket_keys().len(),
        })
    };

    println!(
        "stability n={n} b0={b0}: identical allocs min/max {identical_min}/{identical_max} buckets {}",
        workspace.bucket_keys().len()
    );
    serde_json::json!({
        "n": n, "k": k, "calls": calls,
        "status": "ok",
        "timer": SYNC_WALL_TIMER,
        "timing_method": timing,
        "identical": {
            "b": b0,
            "calls": calls,
            "reserved_bytes_series": reserved_series(&identical_reserved),
            "allocation_calls_per_call": identical_allocs,
            "allocation_calls_min": identical_min,
            "allocation_calls_max": identical_max,
            "bucket_count_before": identical_buckets_before,
            "bucket_count_after": identical_buckets_after,
        },
        "alternating": alternating,
        "bucket_keys": workspace.bucket_keys().iter().map(|key| {
            serde_json::json!({
                "batch": key.0, "trajectories": key.1, "steps": key.2,
                "n_raw": key.3, "formulas": key.4, "window": key.5,
            })
        }).collect::<Vec<_>>(),
    })
}

/// Record a harness span on a host-timed stage object: `profile_ms` becomes
/// the device duration and `timing_method` the span's own timing method. The
/// synchronised wall clock (`sync_wall_ms`, timer
/// [`SYNC_WALL_TIMER`]) is untouched.
///
/// Finding A1: on a `DeviceTimestamps` runtime an ordinary `client.profile`
/// span is whole-stage device time only when the stage provably fits one
/// timestamped compute pass — host-measured launches within
/// [`device_pass_task_limit`] and no mid-stage upload or read (either forces
/// a flush: `cubecl-wgpu-0.10.0/src/compute/stream.rs:105` write path,
/// `read_resources` ends the pass). Such a stage keeps its number with the
/// scope `"single timestamped compute pass"`. Any other device-timestamp
/// stage reports `"profile_ms": "unavailable"` with a `profile_scope` saying
/// exactly why (launch count vs the pass limit, mid-stage upload/read);
/// anything else would report the first pass's begin-to-end duration as the
/// stage's time (`compute/stream.rs:239,548,457`; `compute/timings.rs:321`
/// drains only newly initialised tokens; `compute/timings.rs:193` resolves
/// the token's end against its initial query set). On a `SystemTime` runtime
/// nothing changes: the span is host time over the closure including
/// dispatch. `device_span_plausible` stays only as an extra self-check: a
/// stage with more than 100 launches whose `profile_ms` is below 5% of its
/// `sync_wall_ms` prints a WARNING and records `false`; an unavailable span
/// records `true` (vacuous — there is no number to distrust).
fn set_device_profile(stage: &mut serde_json::Value, ms: f64, timer: &'static str) {
    let limit = device_pass_task_limit() as u64;
    stage["device_pass_task_limit"] = serde_json::Value::from(limit);
    let launches = stage
        .get("launches")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let wall_ms = stage
        .get("sync_wall_ms")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let upload_bytes = stage
        .get("upload_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let reads = stage.get("reads").and_then(|v| v.as_u64()).unwrap_or(0)
        + stage
            .get("runtime_reads")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
    let name = stage
        .get("stage")
        .and_then(|s| s.as_str())
        .unwrap_or("?")
        .to_string();
    // Multi-pass evidence from the host-measured window (counts always come
    // from the host hook path, identical between the modes).
    let mut reasons: Vec<String> = Vec::new();
    if launches > limit {
        reasons.push(format!("{launches} launches exceed one pass ({limit} tasks)"));
    }
    if upload_bytes > 0 {
        reasons.push(format!("{upload_bytes} upload bytes force a flush mid-stage"));
    }
    if reads > 0 {
        reasons.push(format!("{reads} reads end the pass mid-stage"));
    }
    if timer == "DeviceTimestamps" && !reasons.is_empty() {
        stage["profile_ms"] = serde_json::Value::from("unavailable");
        stage["profile_note"] = serde_json::Value::from(
            "device mode: whole-stage device duration is unavailable on this multi-pass path (see profile_scope)",
        );
        stage["profile_scope"] = serde_json::Value::from(format!(
            "unavailable: the stage spans more than one timestamped compute pass ({}); an ordinary client.profile token returns the first timestamped pass's begin-to-end duration, not the stage's elapsed device time (cubecl-wgpu-0.10.0 compute/stream.rs:239,548,457, compute/timings.rs:321,193); host-side dispatch, submission and wait stay in sync_wall_ms",
            reasons.join("; ")
        ));
        stage["device_span_plausible"] = serde_json::Value::from(true);
        eprintln!(
            "profile_ms2_substructure: stage {name} device span unavailable: {} (see profile_scope)",
            reasons.join("; "),
        );
        return;
    }
    stage["profile_ms"] = serde_json::Value::from(ms);
    stage["timing_method"] = serde_json::Value::from(timer);
    stage["profile_note"] = serde_json::Value::from(
        "device mode: real client.profile span on the runner thread (see header)",
    );
    stage["profile_scope"] = serde_json::Value::from(match timer {
        "DeviceTimestamps" => format!("single timestamped compute pass: {launches} launches fit one pass (at most {limit} tasks, no mid-stage upload or read), so the client.profile token covers the stage's only compute pass (queued work is flushed before the token opens and the resolve is submitted before it closes); host-side dispatch, submission and wait stay in sync_wall_ms, never in profile_ms"),
        "SystemTime" => "client.profile token around the stage closure on a runtime without device timestamps: host system time over the closure, including dispatch".to_string(),
        _ => "the runtime reported no timing method for this span".to_string(),
    });
    let plausible = !(launches > 100 && ms < 0.05 * wall_ms);
    stage["device_span_plausible"] = serde_json::Value::from(plausible);
    if !plausible {
        eprintln!(
            "profile_ms2_substructure: WARNING: stage {name} device span implausible: profile_ms {ms:.3} below 5% of sync_wall_ms {wall_ms:.3} with {launches} launches (see profile_scope)",
        );
    }
}

/// Runner-thread session for device-mode generate profiling: everything one
/// stage needs, built on the runner thread and never crossing it.
///
/// Device mode runs the PRODUCTION workload: the same workspace-level stage
/// functions [`Ms2Model::generate_with_hook`] calls (`generate_preprocess`,
/// `generate_encode_ws`, `generate_search_ws`, `generate_decoder_init_ws`,
/// `generate_decode_step_ws`, `generate_validate_ws`, `generate_readout_ws`)
/// over the session's own warmed [`GenerationWorkspace`] buckets and under
/// the same `no_grad` guard production uses (each stage holds it). No replica
/// stage code remains: the span bodies below are thin orchestrations over the
/// shared functions, and launches, allocations, reads and wall times always
/// come from the host hook path, which runs the real `generate`.
struct GenSession {
    device: Device<R>,
    model: Ms2Model<R, E>,
    table: DeviceFormulaTable<R, E>,
    constants: Ms2Constants<R>,
    workspace: GenerationWorkspace<R, E>,
    batch: SpectrumBatch,
    gen_config: GenerationConfig,
    pre: GeneratePreflight,
    host_status: Vec<u32>,
    spectrum_ids: Vec<u64>,
    spectra: Option<DeviceSpectra<R, E>>,
    encoded: Option<EncoderOutput<R, E>>,
    decoder_state: Option<DecoderState<R, E>>,
    bond_table: Option<Tensor<R, E>>,
    traj_formula: Option<Var<R, E>>,
}

fn gen_err(e: mamba3::error::Error) -> String {
    e.to_string()
}

fn gen_metadata_blind(s: &GenSession) -> bool {
    matches!(
        s.gen_config.control,
        Control::MetadataOnly | Control::StructurePrior
    )
}

fn gen_preprocess(s: &mut GenSession) -> Result<(), String> {
    // The estimate was checked at session build (preflight before any
    // allocation); re-checking here is host-side only and keeps the span
    // self-sufficient.
    s.model
        .generate_preflight(&s.batch, &s.table, &s.gen_config)
        .map_err(gen_err)?;
    let spectra = s
        .model
        .generate_preprocess(&s.batch, &s.gen_config, &s.device)
        .map_err(gen_err)?;
    s.host_status = spectra.host_status.clone();
    s.spectrum_ids = spectra.spectrum_id.clone();
    s.spectra = Some(spectra);
    Ok(())
}

fn gen_encoder(s: &mut GenSession) -> Result<(), String> {
    let spectra = s.spectra.as_ref().ok_or("encoder span ran before preprocess")?;
    // The warmed bucket the production call uses: the workspace was warmed
    // with the same shapes, so this is a cache hit, never a fresh buffer.
    let encoded = s
        .model
        .generate_encode_ws(&mut s.workspace, spectra, s.gen_config.control, &s.pre, &s.device)
        .map_err(gen_err)?;
    s.encoded = Some(encoded);
    Ok(())
}

fn gen_search(s: &mut GenSession) -> Result<(), String> {
    let spectra = s.spectra.as_ref().ok_or("search span ran before preprocess")?;
    let pool = s
        .encoded
        .as_ref()
        .ok_or("search span ran before encoder")?
        .pool
        .clone();
    let blind = gen_metadata_blind(s);
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
            blind,
            s.gen_config.formula_rows_visited_max,
            s.gen_config.formula_rows_scored_max,
            &s.gen_config,
            &s.pre,
            &s.device,
        )
        .map_err(gen_err)
}

fn gen_decoder_init(s: &mut GenSession) -> Result<(), String> {
    let encoded = s.encoded.as_ref().ok_or("init span ran before encoder")?;
    let (state, bonds, traj) = s
        .model
        .generate_decoder_init_ws(&mut s.workspace, encoded, &s.pre, &s.device)
        .map_err(gen_err)?;
    s.decoder_state = Some(state);
    s.bond_table = Some(bonds);
    s.traj_formula = Some(traj);
    Ok(())
}

fn gen_decode_loop(s: &mut GenSession) -> Result<(), String> {
    let seed_lo = (s.gen_config.seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (s.gen_config.seed >> 32) as u32;
    let temperature = s.gen_config.temperature;
    // Step 0 is not sampled: initialisation wrote START at position 0 — the
    // same loop production runs, over the same warmed bucket.
    for step in 1..s.pre.steps {
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
            .map_err(gen_err)?;
        // Carry snapshots are test support (device reads); the harness never
        // enables capture, so this stays `None` and changes no counter.
        assert!(carry.is_none(), "device profiling never captures carries");
        s.decoder_state = Some(state);
    }
    Ok(())
}

fn gen_validate(s: &mut GenSession) -> Result<(), String> {
    s.model
        .generate_validate_ws(
            &mut s.workspace,
            &s.constants.atom_table,
            s.pre.trajectories,
            &s.gen_config,
            &s.pre,
            &s.device,
        )
        .map_err(gen_err)
}

fn gen_readout(s: &mut GenSession) -> Result<(), String> {
    // The single batched read of the whole call, as production performs it;
    // values are dropped (counts and validation come from the host hook path
    // — this span times the read only).
    let _ = s
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
        .map_err(gen_err)?;
    Ok(())
}

/// Per-stage device times for one generate configuration: preprocess,
/// encoder, search, decoder init, decode loop, validate, readout.
///
/// The session (model, workspace, buffers: all `Rc`-holding) is built on the
/// runner thread; each stage runs in its own `client.profile` span through a
/// `fn` pointer that captures nothing. A warmup runs outside the spans; the
/// production cold/warm measurements always stay outside the harness.
#[allow(clippy::too_many_arguments)]
fn device_generate_spans(
    device: &Device<R>,
    config: &ModelConfig,
    host_table: &FormulaTable,
    batch: &SpectrumBatch,
    gen_config: &GenerationConfig,
) -> Vec<(f64, &'static str)> {
    // Finding A2: device-mode profiling with Enumerate is refused here,
    // before any allocation — the base record above already admitted the
    // configuration with the enumeration-inclusive estimate, but the device
    // session cannot refit the fixture artifacts on the runner thread, so
    // this path stays host-mode-only rather than failing mid-session.
    if matches!(
        gen_config.formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) {
        eprintln!("profile_ms2_substructure: device-mode profile with Enumerate is not supported in this driver (use --profile-mode host)");
        std::process::exit(1);
    }
    let device_c = device.clone();
    let config_c = config.clone();
    let table_c = host_table.clone();
    let batch_c = batch.clone();
    let gen_c = gen_config.clone();
    profile_session(
        device,
        move || {
            let table =
                DeviceFormulaTable::<R, E>::upload(&table_c, &device_c).expect("table uploads");
            let mut rng = Rng::seeded(1000 + batch_c.n_raw as u64 * 131 + batch_c.len() as u64);
            let mut model =
                Ms2Model::<R, E>::init(&config_c, &device_c, &mut rng).expect("model inits");
            if matches!(
                gen_c.formula_source,
                mamba3::models::ms2::contract::FormulaSource::Enumerate
            ) {
                // Unreachable: `device_generate_spans` refuses Enumerate
                // before the session builds (finding A2). Kept as a guard so
                // a future caller that bypasses that check fails loudly
                // instead of profiling without resident artifacts.
                panic!("device-mode profile with Enumerate is not supported in this driver (use --profile-mode host)");
            }
            // Preflight before any allocation, exactly like production: the
            // warmed workspace buckets below are only allocated after this.
            let pre = model
                .generate_preflight(&batch_c, &table, &gen_c)
                .expect("device session preflights");
            GenSession {
                device: device_c.clone(),
                constants: Ms2Constants::new(&device_c),
                workspace: GenerationWorkspace::new(),
                model,
                table,
                batch: batch_c,
                gen_config: gen_c,
                pre,
                host_status: Vec::new(),
                spectrum_ids: Vec::new(),
                spectra: None,
                encoded: None,
                decoder_state: None,
                bond_table: None,
                traj_formula: None,
            }
        },
        |p: &Profiler<R>| -> Vec<(f64, &'static str)> {
            // Warmup outside the spans (same shapes as the spans below, so
            // compilation and tuning settle before any span opens). The
            // production call populates the warmed workspace buckets the
            // spans then reuse — the spans time the production workload.
            p.with_state(|s: &mut GenSession| {
                for _ in 0..2 {
                    s.model
                        .generate(&s.batch, &s.table, &s.gen_config, &mut s.workspace, &s.constants)
                        .expect("device warmup runs");
                }
            })
            .expect("device warmup runs");
            let mut out = Vec::with_capacity(7);
            let span = |p: &Profiler<R>, name: &str, f: fn(&mut GenSession) -> Result<(), String>| {
                let (r, t) = p.span(name, f).expect("profile span runs");
                r.expect("profiled stage runs");
                (t.ms, t.timer)
            };
            out.push(span(p, "ms2.preprocess", gen_preprocess));
            out.push(span(p, "ms2.encoder", gen_encoder));
            out.push(span(p, "ms2.search", gen_search));
            out.push(span(p, "ms2.decoder_init", gen_decoder_init));
            out.push(span(p, "ms2.decode_loop", gen_decode_loop));
            out.push(span(p, "ms2.validate", gen_validate));
            out.push(span(p, "ms2.readout", gen_readout));
            out
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("profile_ms2_substructure: device session failed: {e}");
        std::process::exit(1);
    })
}

/// Runner-thread session for device-mode train profiling: the trainer plus
/// the batch, built on the runner thread and never crossing it. The forward
/// state is stashed between spans because the backward phase consumes the
/// forward phase's autograd graph — phases cannot run standalone.
struct TrainSession {
    trainer: Ms2Trainer<R, E>,
    set: ExperimentSet,
    indices: Vec<usize>,
    fwd: Option<ForwardState<R, E>>,
    bwd: Option<BackwardState<R, E>>,
}

fn train_forward_span(s: &mut TrainSession) -> Result<(), String> {
    let fwd = s
        .trainer
        .forward_state(&s.set, &s.indices)
        .map_err(|e| e.to_string())?;
    s.fwd = Some(fwd);
    Ok(())
}

fn train_backward_span(s: &mut TrainSession) -> Result<(), String> {
    let fwd = s.fwd.as_ref().ok_or("backward span ran before forward")?;
    let bwd = s
        .trainer
        .backward_state(fwd)
        .map_err(|e| e.to_string())?;
    s.bwd = Some(bwd);
    Ok(())
}

fn train_optimizer_span(s: &mut TrainSession) -> Result<(), String> {
    let bwd = s.bwd.as_ref().ok_or("optimizer span ran before backward")?;
    s.trainer.optimizer_step(bwd).map_err(|e| e.to_string())?;
    s.bwd = None;
    s.fwd = None;
    Ok(())
}

/// Per-stage device times for one train configuration: forward, backward,
/// optimizer. No report is requested, so no span reads the device — the same
/// window the host hook path measures.
fn device_train_spans(
    device: &Device<R>,
    model_config: &ModelConfig,
    host_table: &FormulaTable,
    comps: &[Composition],
    n: usize,
    train_config: &TrainConfig,
) -> Vec<(f64, &'static str)> {
    let device_c = device.clone();
    let model_c = model_config.clone();
    let table_c = host_table.clone();
    let train_c = train_config.clone();
    let comps_c = comps.to_vec();
    // Finding A2, like `device_generate_spans`: device-mode profiling with
    // Enumerate is refused before any allocation rather than failing in the
    // warmed session without resident artifacts.
    if matches!(
        train_c.formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) {
        eprintln!("profile_ms2_substructure: device-mode profile with Enumerate is not supported in this driver (use --profile-mode host)");
        std::process::exit(1);
    }
    profile_session(
        device,
        move || {
            let trainer =
                Ms2Trainer::<R, E>::new(&model_c, &table_c, &train_c, &device_c)
                    .expect("trainer builds");
            let set = experiment_set(&comps_c, n, 6000 + n as u64);
            let indices: Vec<usize> = (0..comps_c.len()).collect();
            TrainSession {
                trainer,
                set,
                indices,
                fwd: None,
                bwd: None,
            }
        },
        |p: &Profiler<R>| -> Vec<(f64, &'static str)> {
            p.with_state(|s: &mut TrainSession| {
                for _ in 0..2 {
                    s.trainer
                        .step(&s.set, &s.indices)
                        .expect("device warmup runs");
                }
            })
            .expect("device warmup runs");
            let mut out = Vec::with_capacity(3);
            let span = |p: &Profiler<R>, name: &str, f: fn(&mut TrainSession) -> Result<(), String>| {
                let (r, t) = p.span(name, f).expect("profile span runs");
                r.expect("profiled stage runs");
                (t.ms, t.timer)
            };
            out.push(span(p, "forward", train_forward_span));
            out.push(span(p, "backward", train_backward_span));
            out.push(span(p, "optimizer", train_optimizer_span));
            out
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("profile_ms2_substructure: device session failed: {e}");
        std::process::exit(1);
    })
}
