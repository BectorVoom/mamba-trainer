//! P2.2 tests: memory-estimate reconciliation for one generation call and
//! one training step at the V0 shapes.
//!
//! Generation compares the estimate total from `Ms2MemoryEstimate::generation`
//! against the measured increase of `reserved_bytes` from a fresh device
//! context to the end of the second warm call (band 0.5×–8×). Training
//! compares `Ms2MemoryEstimate::training` at G = 16 slots against the
//! measured `bytes_in_use` peak (sampled during warm steps) at B = 4, 8, 16
//! (band 0.67×–1.5×); the reserved increase is reported alongside for
//! documentation only. Both numbers and the ratio are printed. When the
//! runtime does not report memory the test prints "unavailable" and passes
//! without asserting (never a faked value).
//!
//! Why reserved bytes differ from logical bytes: the allocator rounds every
//! buffer up to its size class and keeps freed buffers pooled, so reserved
//! is the high-water mark of the pool, not the live sum; matmul autotune
//! scratch and one-time compilation buffers also stay pooled after the first
//! call. The generation band below (0.5×–8×) brackets that pooling: reserved
//! is at least the live state that survives (workspace buckets, weights,
//! table) and at most a small multiple of the logical total after two warm
//! calls. Training bands the live peak instead, which the pool cannot
//! inflate, so its band is tight.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, reserved_bytes};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{GenerationConfig, ModelConfig, SCHEMA_VERSION, SpectrumBatch};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::train::{Ms2Trainer, TrainConfig};
use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

const B: usize = 8;
const K: u32 = 8;

/// Parent compositions from well-known public structures (fixture-derived in
/// the sense of chemistry_v0.json: benzene, alanine, glucose, naphthalene,
/// repeated to fill `b`). All in-domain with distinct masses above the 50 Da
/// precursor floor.
fn parent_comps(b: usize) -> Vec<Composition> {
    let base = vec![
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
        [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
        [10, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    (0..b).map(|i| base[i % base.len()]).collect()
}

fn precursor_of(comp: &Composition) -> u32 {
    composition_mass(comp).unwrap() + 1_007_825 - 549
}

/// B spectra with explicit precursors and deterministic synthetic peaks below
/// them (seeded RNG; peak shapes only affect selection cost, not the
/// reconciliation logic).
fn spectra_batch(comps: &[Composition], n_raw: usize, seed: u64) -> SpectrumBatch {
    let b = comps.len();
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for bi in 0..b {
        let count = (n_raw / 2) as u32;
        peak_count[bi] = count;
        raw_peak_count[bi] = count;
        for i in 0..count as usize {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursors[bi] - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + 2.0 * rng.uniform_vec(1, 0.0, 1.0)[0];
        }
    }
    SpectrumBatch {
        schema_version: SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: (0..b as u64).map(|i| 1000 + i).collect(),
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

/// A minimal 8-spectrum experiment set with unlabeled spectra (empty slots):
/// the training step still runs the full forward, backward and optimizer
/// path with zero graph loss on the empty slots.
fn experiment_set(comps: &[Composition], n_raw: usize, seed: u64) -> ExperimentSet {
    let batch = spectra_batch(comps, n_raw, seed);
    let spectra = (0..comps.len())
        .map(|i| {
            let base = i * n_raw;
            let np = batch.peak_count[i] as usize;
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
                    peak_id: batch.peak_id[base..base + np].to_vec(),
                    mz_udalton: batch.mz_udalton[base..base + np].to_vec(),
                    intensity: batch.intensity[base..base + np]
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
        name: "footprint-synthetic".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: (0..comps.len()).map(|i| format!("mol{i}")).collect(),
        spectra,
    }
}

#[test]
fn generation_estimate_reconciles_with_reserved_bytes() {
    let device = Device::<R>::default();
    let Some(baseline) = reserved_bytes(&device) else {
        println!("generation footprint: reserved bytes unavailable");
        println!("unavailable");
        return;
    };
    let model_config = ModelConfig::v0();
    let comps = parent_comps(B);
    let host_table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let table = DeviceFormulaTable::<R, E>::upload(&host_table, &device).unwrap();
    let mut config = ModelConfig::v0();
    config.formula_table.rows = table.rows as u32;
    config.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, E>::init(&config, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let batch = spectra_batch(&comps, 64, 21);
    let gen_config = GenerationConfig {
        trajectories: K,
        ..GenerationConfig::default()
    };
    let estimate = Ms2MemoryEstimate::generation(
        &model_config,
        table.rows as u64,
        B as u64,
        K as u64,
        64,
        gen_config.max_steps as u64,
    )
    .unwrap()
    .total()
    .unwrap();
    let mut workspace = GenerationWorkspace::new();
    // Two warm calls: the first compiles kernels and settles the allocator
    // and autotune scratch; the measurement ends after the second.
    for _ in 0..2 {
        let out = model
            .generate(&batch, &table, &gen_config, &mut workspace, &constants)
            .unwrap();
        out.validate().unwrap();
    }
    let Some(end) = reserved_bytes(&device) else {
        println!("generation footprint: reserved bytes unavailable");
        println!("unavailable");
        return;
    };
    let measured = end.saturating_sub(baseline);
    let ratio = measured as f64 / estimate as f64;
    println!("generation estimate total: {estimate} bytes");
    println!("generation measured reserved increase: {measured} bytes");
    println!("generation ratio measured/estimate: {ratio:.3}");
    assert!(
        (0.5..=8.0).contains(&ratio),
        "generation reserved/estimate ratio {ratio:.3} outside 0.5..=8.0 \
         (estimate {estimate}, measured {measured}): the pool rounds and holds \
         matmul/autotune scratch, so reserved is a high-water mark, not the live sum"
    );
}

#[test]
fn training_estimate_tracks_in_use_peak() {
    // P2.2 reconciliation at the V0 shapes (G = 16 slots, T = 22 steps,
    // N = 64 raw peaks): for B = 4, 8, 16 the training estimate must track
    // the measured `bytes_in_use` peak within [0.67, 1.5].
    //
    // Method: a fresh device per B (clean allocator baseline), two warm
    // steps to compile kernels and settle the pool, then three measured
    // steps during which a sampler thread records the peak `bytes_in_use`.
    // `Ms2Trainer::step` is monolithic (forward, `backward_retain` and the
    // AdamW step in one call with no inter-phase hook), so the phase
    // boundaries inside one step are not observable through the public API:
    // instead the steady-state in-use before/after the steps (the fixed
    // part: weights, gradients, AdamW moments, buckets, table) and the
    // in-step peak (fixed plus retained activations, their gradients and
    // kernel scratch) separate the batch-proportional retained part from
    // the fixed part across the three batch sizes by slope vs intercept.
    // A forward-only peak (`teacher_eval` under `no_grad`: forward
    // temperatures plus outputs, no retention or backward) is measured the
    // same way for comparison, so the retained-plus-gradients part is the
    // step peak minus the forward peak.
    // The reserved-bytes increase is reported alongside for documentation
    // only: the allocator rounds every buffer to its size class and keeps
    // freed buffers pooled, so reserved is a high-water mark of the pool,
    // not the live sum (matmul autotune scratch stays pooled too).
    use mamba3::backend::memory_snapshot;
    use mamba3::cubecl::stream_id::StreamId;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    // The allocator must report live bytes at all for this to mean
    // anything; otherwise print "unavailable" and pass (never a faked
    // value), as the generation test does for reserved bytes.
    {
        let probe = Device::<R>::default();
        if memory_snapshot(&probe).is_none() {
            println!("training footprint: memory snapshot unavailable");
            println!("unavailable");
            return;
        }
    }
    const SLOTS: usize = 16;
    const N_RAW: usize = 64;
    const T: u64 = 22;
    println!("B | in_use_before | in_use_peak | estimate | peak/est | reserved_increase");
    for b in [4usize, 8, 16] {
        let device = Device::<R>::default();
        let model_config = ModelConfig::v0();
        let comps = parent_comps(b);
        let host_table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
        let train_config = TrainConfig {
            batch: b,
            slots: SLOTS,
            ..TrainConfig::default()
        };
        let mut trainer =
            Ms2Trainer::<R, E>::new(&model_config, &host_table, &train_config, &device).unwrap();
        let set = experiment_set(&comps, N_RAW, 23);
        let indices: Vec<usize> = (0..b).collect();
        let estimate_items = Ms2MemoryEstimate::training(
            &model_config,
            host_table.len() as u64,
            b as u64,
            SLOTS as u64,
            N_RAW as u64,
            T,
        )
        .unwrap();
        for (name, bytes) in &estimate_items.items {
            println!("    est item {name}: {bytes}");
        }
        let estimate = estimate_items.total().unwrap();
        // Two warm steps: the first compiles kernels, the second settles
        // the allocator pool.
        for _ in 0..2 {
            let _ = trainer.step(&set, &indices).unwrap();
        }
        device.synchronize();
        let before = memory_snapshot(&device)
            .map(|s| s.bytes_in_use)
            .unwrap_or(0);
        let reserved_before = memory_snapshot(&device)
            .map(|s| s.bytes_reserved)
            .unwrap_or(0);
        // Three measured steps with a sampler thread recording the peak
        // live bytes; the peak spans forward (retained activations),
        // backward (their gradients) and the optimizer step. CubeCL
        // streams are thread-local (`StreamId::current`), so the sampler
        // must query the main thread's stream explicitly via `executes`:
        // its own stream would stay empty and report nothing.
        let main_stream = StreamId::current();
        let peak = Arc::new(AtomicU64::new(before));
        let stop = Arc::new(AtomicBool::new(false));
        std::thread::scope(|s| {
            let dev = device.clone();
            let peak_ref = peak.clone();
            let stop_ref = stop.clone();
            s.spawn(move || {
                while !stop_ref.load(Ordering::Relaxed) {
                    let snap = main_stream.executes(|| memory_snapshot(&dev));
                    if let Some(snap) = snap {
                        peak_ref.fetch_max(snap.bytes_in_use, Ordering::Relaxed);
                    }
                }
            });
            for _ in 0..3 {
                let _ = trainer.step(&set, &indices).unwrap();
            }
            device.synchronize();
            stop.store(true, Ordering::Relaxed);
        });
        let in_use_peak = peak.load(Ordering::Relaxed);
        let after = memory_snapshot(&device)
            .map(|s| s.bytes_in_use)
            .unwrap_or(0);
        let reserved_after = memory_snapshot(&device)
            .map(|s| s.bytes_reserved)
            .unwrap_or(0);
        let reserved_increase = reserved_after.saturating_sub(reserved_before);
        let ratio = in_use_peak as f64 / estimate as f64;
        // Forward-only peak for comparison: `teacher_eval` runs the same
        // forward pass under `no_grad` (no retention, no backward, no
        // optimizer), so its peak is forward temperatures plus outputs while
        // the step peak above adds what autograd retains, the gradients and
        // the optimizer transient. Sampled on the main stream the same way.
        let fwd_peak = Arc::new(AtomicU64::new(before));
        let fwd_stop = Arc::new(AtomicBool::new(false));
        std::thread::scope(|s| {
            let dev = device.clone();
            let peak_ref = fwd_peak.clone();
            let stop_ref = fwd_stop.clone();
            s.spawn(move || {
                while !stop_ref.load(Ordering::Relaxed) {
                    let snap = main_stream.executes(|| memory_snapshot(&dev));
                    if let Some(snap) = snap {
                        peak_ref.fetch_max(snap.bytes_in_use, Ordering::Relaxed);
                    }
                }
            });
            let _ = trainer.teacher_eval(&set, &indices).unwrap();
            device.synchronize();
            fwd_stop.store(true, Ordering::Relaxed);
        });
        let fwd_peak = fwd_peak.load(Ordering::Relaxed);
        println!(
            "{b} | {before} | {in_use_peak} | {estimate} | {ratio:.3} | {reserved_increase} (after-step in-use {after}, forward-only peak {fwd_peak})"
        );
        assert!(
            (0.67..=1.5).contains(&ratio),
            "training in-use-peak/estimate ratio {ratio:.3} outside 0.67..=1.5 at B={b} \
             (estimate {estimate}, in-use peak {in_use_peak}, in-use before {before})"
        );
    }
}
