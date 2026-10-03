//! MS2 substructure profile driver: stage spans, CubeCL `client.profile`
//! timing, synchronised wall clock, warmup, cold-start reporting and
//! machine-readable JSON (architecture §6.5, P2.6).
//!
//! ```text
//! cargo run --release --no-default-features --features cpu --example profile_ms2_substructure -- \
//!   --mode both --n 64,128 --b 1,8 --k 8 --warmup 2 --repeats 3 --out /tmp/ms2_profile.json
//! ```
//!
//! Modes `--mode generate|train|both`; shapes `--n 64,128,256,512`
//! (raw peaks per spectrum) with `--b 1,8,32` (cartesian), `--k 8`
//! (trajectories), `--warmup 2`, `--repeats 10`, `--out <json>`,
//! `--max-device-bytes <bytes>` (default 4 GiB).
//!
//! Per configuration: cold first-call wall time; warm p50/p95 wall time over
//! `repeats` with a device sync per call; per stage (preprocess/upload,
//! encoder, formula search, decode steps all and per step, validate, readout
//! for generation; forward, backward, optimizer for training) launches,
//! reads, upload/download bytes, allocation calls and synchronised wall time
//! with the timing method recorded (`DeviceTimestamps` or `SystemTime`,
//! probed from what `client.profile` returns). Stage times are
//! synchronised wall clock rather than `client.profile` spans because the
//! model and trainer hold `Rc` parameters (`!Send` closures cannot enter
//! `client.profile`), and no non-`Send` profiling entry point exists in the
//! pinned CubeCL 0.10: the only client-side entry is
//! `ComputeClient::profile` (`cubecl-runtime-0.10.0/src/client.rs:886`),
//! whose closure bound is `FnOnce() -> O + Send`; the start/end token API
//! (`start_profile`/`end_profile`) lives on the `ComputeServer` trait
//! (`cubecl-runtime-0.10.0/src/server/base.rs:397,400`) and the client does
//! not surface it, so per-stage device timestamps cannot be recorded without
//! restructuring the model (`Rc` to `Arc`), which is out of scope. The
//! recorded method still says which clock that wall time is. Peak `reserved_bytes` is recorded. Configurations whose
//! `Ms2MemoryEstimate` exceeds `--max-device-bytes` are recorded as refused
//! with the estimate, not run.
//!
//! Stage profiling never changes the production read count: production
//! (cold/warm) numbers are measured first with their own counter windows,
//! and every `client.profile` sync lives in a later, separately labelled
//! profiling window (P2.9). Weights are random-init (seeded), so timings are
//! structural, not quality numbers; spectra are synthetic from the chemistry
//! fixture's compositions plus seeded peaks (see `fixture_comps`).

use std::time::Instant;

use mamba3::backend::{
    Device, allocation_calls, download_bytes, launch_count, launch_tally_detailed, memory_snapshot,
    read_count, reserved_bytes, reset_launch_count, reset_launch_tally, reset_read_count,
    reset_transfer_counters, runtime_read_count, start_launch_tally, stop_launch_tally,
    upload_bytes,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass, element_index};
use mamba3::models::ms2::contract::{
    Control, GenerationConfig, ModelConfig, SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::train::{Ms2Trainer, TrainConfig};
use mamba3::models::ms2::workspace::{Ms2Capabilities, Ms2MemoryEstimate};
use mamba3::tensor::ops::ms2::{self, Ms2Constants};
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn usage() -> ! {
    eprintln!(
        "usage: profile_ms2_substructure [--mode generate|train|both] [--n 64,128,256,512] \
         [--b 1,8,32] [--k 8] [--warmup 2] [--repeats 10] [--max-device-bytes 4294967296] \
         --out <json>"
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
        schema_version: SCHEMA_VERSION,
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

/// One profiled stage: counters are zeroed first, `op` runs, the device is
/// synchronised, and the deltas are read.
///
/// Stage times are synchronised wall clock (op plus device sync), not
/// `client.profile` spans: the model and trainer hold `Rc` parameters, so a
/// closure borrowing them is `!Send` and cannot enter `client.profile`'s
/// `Send` bound (`cubecl-runtime-0.10.0/src/client.rs:886), and the client
/// exposes no start/end token or guard API — that token API lives on the
/// `ComputeServer` trait (`cubecl-runtime-0.10.0/src/server/base.rs:397,400`)
/// and is not surfaced by the client. The `timing` field still records what
/// `client.profile` returns on this device (probed once with a trivial
/// closure: `DeviceTimestamps` is `TimingMethod::Device`, `SystemTime` is
/// `TimingMethod::System` per `cubecl-common-0.10.0/src/profile.rs:15`; the
/// CPU runtime timestamps through `TimestampProfiler`, whose `stop`
/// (`cubecl-runtime-0.10.0/src/timestamp_profiler.rs:37`) returns system
/// time), so a reader knows which clock the milliseconds are.
/// Production windows are measured separately, so these syncs never move
/// production read counts (P2.9).
fn profile_stage(
    device: &Device<R>,
    label: &str,
    timing: &str,
    op: impl FnOnce(),
) -> serde_json::Value {
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let started = Instant::now();
    op();
    device.synchronize();
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    serde_json::json!({
        "stage": label,
        "launches": launch_count(),
        "reads": read_count(),
        "runtime_reads": runtime_read_count(),
        "upload_bytes": upload_bytes(),
        "download_bytes": download_bytes(),
        "allocation_calls": allocation_calls(),
        "ms": ms,
        "timing_method": timing,
        "window": "profiled (P2.9: separate from production)",
    })
}

fn main() {
    let mut mode = "both".to_string();
    let mut ns = vec![64usize, 128, 256, 512];
    let mut bs = vec![1usize, 8, 32];
    let mut k: u32 = 8;
    let mut warmup: usize = 2;
    let mut repeats: usize = 10;
    let mut max_device_bytes: u64 = 4 * 1024 * 1024 * 1024;
    let mut out: Option<String> = None;
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
            "--max-device-bytes" => max_device_bytes = next().parse().unwrap_or_else(|_| usage()),
            "--out" => out = Some(next()),
            _ => usage(),
        }
    }
    if !["generate", "train", "both"].contains(&mode.as_str()) {
        usage();
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
                    ));
                }
            }
        }
    }

    let report = serde_json::json!({
        "machine": {
            "backend": backend,
            "adapter": serde_json::Value::Null,
            "adapter_note": "adapter name is not exposed by ComputeClient properties in CubeCL 0.10; backend name above is the device identity",
            "os": os,
            "timing_method": timing,
            "timing_note": "probed from what client.profile returns (device timestamps vs system time)",
        },
        "config": {
            "mode": mode,
            "n": ns,
            "b": bs,
            "k": k,
            "warmup": warmup,
            "repeats": repeats,
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
) -> serde_json::Value {
    let model_config = ModelConfig::v0();
    let estimate = Ms2MemoryEstimate::generation(
        &model_config,
        host_table.len() as u64,
        b as u64,
        k as u64,
        n as u64,
        22,
    )
    .and_then(|est| {
        let total = est.total()?;
        if total > max_device_bytes {
            return Err(mamba3::error::Error::Config(format!(
                "memory estimate {total} bytes exceeds limit {max_device_bytes} bytes"
            )));
        }
        Ok(total)
    });
    let estimate_total = Ms2MemoryEstimate::generation(
        &model_config,
        host_table.len() as u64,
        b as u64,
        k as u64,
        n as u64,
        22,
    )
    .and_then(|e| e.total());
    let Ok(estimate_bytes) = estimate else {
        return serde_json::json!({
            "mode": "generate", "n": n, "b": b, "k": k,
            "status": "refused",
            "estimate_bytes": estimate_total.unwrap_or(u64::MAX),
            "max_device_bytes": max_device_bytes,
            "timing_method": timing,
        });
    };

    let table = DeviceFormulaTable::<R, E>::upload(host_table, device).expect("table uploads");
    let mut config = ModelConfig::v0();
    config.formula_table.rows = table.rows as u32;
    config.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(1000 + n as u64 * 131 + b as u64);
    let model = Ms2Model::<R, E>::init(&config, device, &mut rng).expect("model inits");
    let constants = Ms2Constants::new(device);
    let comps: Vec<Composition> = (0..b).map(|i| base_comps[i % base_comps.len()]).collect();
    let batch = spectra_batch(&comps, n, 5000 + n as u64, 7000);
    let gen_config = GenerationConfig {
        trajectories: k,
        ..GenerationConfig::default()
    };
    let steps = gen_config.max_steps as usize;
    let mut workspace = GenerationWorkspace::new();

    // Cold first call (production window).
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let started = Instant::now();
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
    for _ in 0..repeats {
        reset_launch_count();
        reset_read_count();
        reset_transfer_counters();
        let started = Instant::now();
        let out = model
            .generate(&batch, &table, &gen_config, &mut workspace, &constants)
            .expect("timed generate runs");
        out.validate().expect("timed output validates");
        device.synchronize();
        walls.push(started.elapsed().as_secs_f64() * 1000.0);
        reads.push(read_count());
        launches.push(launch_count());
    }
    let (p50, p95) = p50p95(walls.clone());
    let peak_reserved = reserved_bytes(device);

    // Stage profiling (separate P2.9 window): separable prefix ops are
    // profiled directly; decode/validate/readout launches come from the
    // launch tally of one more call, and decode time is the profiled total
    // minus the prefix sum (floored at zero).
    let spectra = mamba3::models::ms2::batch::DeviceSpectra::<R, E>::upload(&batch, device)
        .expect("spectra upload");
    let peaks = ms2::PeakBuffers::<R, E>::new(b, n, model_config.n_peaks as usize, device);
    let s_pre = profile_stage(device, "ms2.preprocess", timing, || {
        mamba3::models::ms2::batch::DeviceSpectra::<R, E>::upload(&batch, device)
            .expect("preprocess runs");
    });
    let s_enc = profile_stage(device, "ms2.encoder", timing, || {
        model
            .encoder
            .encode(&spectra, &peaks, Control::None)
            .expect("encoder runs");
    });
    let encoded = model
        .encoder
        .encode(&spectra, &peaks, Control::None)
        .expect("encoder runs");
    let fbufs = ms2::FormulaBuffers::<R, E>::new(b, 32, gen_config.formulas as usize, device);
    let s_search = profile_stage(device, "ms2.search", timing, || {
        ms2::formula_window(
            &table.table,
            &spectra.meta,
            table.max_error,
            u32::MAX,
            4096,
            &fbufs,
        )
        .expect("window runs");
        let scored = model
            .formula
            .score(&table, &fbufs, &encoded.pool)
            .expect("score runs");
        ms2::formula_top(scored.log_prob.tensor(), &fbufs.window, &fbufs).expect("top runs");
    });
    // One more full call under the tally for per-label launches plus a
    // profiled total for the decode derivation.
    reset_launch_tally();
    start_launch_tally();
    let s_total = profile_stage(device, "ms2.generate_total", timing, || {
        model
            .generate(&batch, &table, &gen_config, &mut workspace, &constants)
            .expect("total runs");
    });
    stop_launch_tally();
    let mut per_label = std::collections::BTreeMap::new();
    for row in launch_tally_detailed() {
        *per_label.entry(row.label).or_insert(0) += row.count;
    }
    let total_ms = s_total["ms"].as_f64().unwrap_or(0.0);
    let prefix_ms: f64 = s_pre["ms"].as_f64().unwrap_or(0.0)
        + s_enc["ms"].as_f64().unwrap_or(0.0)
        + s_search["ms"].as_f64().unwrap_or(0.0);
    let decode_ms = (total_ms - prefix_ms).max(0.0);
    let per_step_ms = decode_ms / (steps - 1).max(1) as f64;
    let step_launches = per_label.get("ms2.step").copied().unwrap_or(0);
    let stages = vec![
        s_pre,
        s_enc,
        s_search,
        serde_json::json!({
            "stage": "decode_steps_all",
            "launches": step_launches,
            "reads": 0, "runtime_reads": 0,
            "upload_bytes": 0, "download_bytes": 0, "allocation_calls": 0,
            "ms": decode_ms,
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "time is the profiled generate total minus the prefix stages; launches from the tally label ms2.step",
        }),
        serde_json::json!({
            "stage": "decode_per_step",
            "launches": step_launches / (steps - 1).max(1),
            "reads": 0, "runtime_reads": 0,
            "upload_bytes": 0, "download_bytes": 0, "allocation_calls": 0,
            "ms": per_step_ms,
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "decode_steps_all divided by T-1 steps",
        }),
        serde_json::json!({
            "stage": "validate",
            "launches": per_label.get("ms2.finalize").copied().unwrap_or(0),
            "reads": 0, "runtime_reads": 0,
            "upload_bytes": 0, "download_bytes": 0, "allocation_calls": 0,
            "ms": 0.0,
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "launches from the tally label ms2.finalize (validate plus readout setup); isolated validate timing needs generate hooks, out of scope for the P2.6 driver",
        }),
        serde_json::json!({
            "stage": "readout",
            "launches": 0,
            "reads": 1, "runtime_reads": 1,
            "upload_bytes": 0, "download_bytes": s_total["download_bytes"].clone(),
            "allocation_calls": 0,
            "ms": 0.0,
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "one batched read_all per warmed call (production window); time included in the generate total",
        }),
    ];

    println!(
        "generate n={n} b={b} k={k}: cold {cold_ms:.1}ms warm p50/p95 {p50:.1}/{p95:.1}ms launches {launches:?} reads {reads:?}"
    );
    serde_json::json!({
        "mode": "generate", "n": n, "b": b, "k": k,
        "status": "ok",
        "estimate_bytes": estimate_bytes,
        "max_device_bytes": max_device_bytes,
        "cold_wall_ms": cold_ms,
        "cold": cold,
        "warm_wall_ms": walls,
        "warm_p50_ms": p50, "warm_p95_ms": p95,
        "warm_reads_per_call": reads,
        "warm_launches_per_call": launches,
        "peak_reserved_bytes": peak_reserved,
        "timing_method": timing,
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
) -> serde_json::Value {
    let model_config = ModelConfig::v0();
    let slots = 2usize;
    let estimate_total = Ms2MemoryEstimate::training(
        &model_config,
        host_table.len() as u64,
        b as u64,
        slots as u64,
        n as u64,
        22,
    )
    .and_then(|e| e.total());
    let fits = estimate_total
        .as_ref()
        .map(|t| *t <= max_device_bytes)
        .unwrap_or(false);
    if !fits {
        return serde_json::json!({
            "mode": "train", "n": n, "b": b, "slots": slots,
            "status": "refused",
            "estimate_bytes": estimate_total.unwrap_or(u64::MAX),
            "max_device_bytes": max_device_bytes,
            "timing_method": timing,
        });
    }
    let estimate_bytes = estimate_total.unwrap_or(0);
    let train_config = TrainConfig {
        batch: b,
        slots,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&model_config, host_table, &train_config, device)
        .expect("trainer builds");
    let comps: Vec<Composition> = (0..b).map(|i| base_comps[i % base_comps.len()]).collect();
    let set = experiment_set(&comps, n, 6000 + n as u64);
    let indices: Vec<usize> = (0..b).collect();

    // Cold first step (production window; no report read on the cold call).
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let started = Instant::now();
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
    for _ in 0..repeats {
        reset_launch_count();
        reset_read_count();
        reset_transfer_counters();
        let started = Instant::now();
        let _ = trainer.step(&set, &indices).expect("timed step runs");
        device.synchronize();
        walls.push(started.elapsed().as_secs_f64() * 1000.0);
        launches.push(launch_count());
    }
    let (p50, p95) = p50p95(walls.clone());
    let peak_reserved = reserved_bytes(device);

    // Stage profiling (separate P2.9 window): forward is the report-free
    // teacher pass; the full step adds backward plus the optimizer, so
    // backward time is the step total minus forward (the optimizer runs
    // inside the step and is reported as its own stage from the same
    // window with the derivation noted).
    let s_forward = profile_stage(device, "forward", timing, || {
        trainer.teacher_eval(&set, &indices).expect("forward runs");
    });
    let s_step = profile_stage(device, "step_total", timing, || {
        trainer.step(&set, &indices).expect("step runs");
    });
    let fwd_ms = s_forward["ms"].as_f64().unwrap_or(0.0);
    let step_ms = s_step["ms"].as_f64().unwrap_or(0.0);
    let fwd_launches = s_forward["launches"].as_u64().unwrap_or(0);
    let step_launches = s_step["launches"].as_u64().unwrap_or(0);
    let stages = vec![
        s_forward,
        serde_json::json!({
            "stage": "backward",
            "launches": step_launches.saturating_sub(fwd_launches),
            "reads": 0, "runtime_reads": 0,
            "upload_bytes": 0, "download_bytes": 0, "allocation_calls": 0,
            "ms": (step_ms - fwd_ms).max(0.0),
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "step total minus forward; includes the optimizer, which runs inside the step in V0",
        }),
        serde_json::json!({
            "stage": "optimizer",
            "launches": 0,
            "reads": 0, "runtime_reads": 0,
            "upload_bytes": 0, "download_bytes": 0, "allocation_calls": 0,
            "ms": 0.0,
            "timing_method": timing,
            "window": "profiled (P2.9: separate from production)",
            "note": "AdamW runs inside the step in V0 and is included in backward above; isolated optimizer timing needs train hooks, out of scope for the P2.6 driver",
        }),
    ];

    println!("train n={n} b={b}: cold {cold_ms:.1}ms warm p50/p95 {p50:.1}/{p95:.1}ms");
    serde_json::json!({
        "mode": "train", "n": n, "b": b, "slots": slots,
        "status": "ok",
        "estimate_bytes": estimate_bytes,
        "max_device_bytes": max_device_bytes,
        "cold_wall_ms": cold_ms,
        "cold": cold,
        "warm_wall_ms": walls,
        "warm_p50_ms": p50, "warm_p95_ms": p95,
        "warm_launches_per_call": launches,
        "peak_reserved_bytes": peak_reserved,
        "timing_method": timing,
        "stages": stages,
    })
}
