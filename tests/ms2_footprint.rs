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

use mamba3::backend::{Device, check_launches, reserved_bytes};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    GenerationConfig, ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

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
        schema_version: SPECTRUM_SCHEMA_VERSION,
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

/// File-level serialisation for counter-reading tests: the process-wide
/// launch/read/transfer counters (and the shared device allocator) are
/// perturbed by any test running beside these, so every test holds this
/// mutex. Poison-tolerant: a panicking holder still releases the lock.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}
#[test]
fn generation_estimate_reconciles_with_reserved_bytes() {
    let _serial = serial();
    // V1 §3.1 shape pin: the reconciliation band holds at the V1 candidate
    // shape (32, 8, 42) with small B and K. This is the only reserved-bytes
    // reader in this binary: `reserved_bytes` is the pool high-water mark,
    // so a second shape measured after this pool has been built would see
    // no growth. The V0 shape lives in `tests/ms2_footprint_v0.rs`, where it
    // is likewise the first allocation of its process.
    // The larger shape uses the full V1 candidate at small B and K: with
    // tiny widths the fixed autotune scratch would dominate the ratio, while
    // the full widths keep the same reconciliation physics as the V0 shape.
    for (mkm, b, k, t) in [(
        ModelConfig::v1_candidate as fn() -> ModelConfig,
        2usize,
        2u32,
        42u32,
    )] {
        let device = Device::<R>::default();
        let Some(baseline) = reserved_bytes(&device) else {
            println!("generation footprint: reserved bytes unavailable");
            println!("unavailable");
            return;
        };
        let model_config = mkm();
        let comps = parent_comps(b);
        let host_table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
        let table = DeviceFormulaTable::<R, E>::upload(&host_table, &device).unwrap();
        let mut config = model_config.clone();
        config.formula_table.rows = table.rows as u32;
        config.formula_table.sha256 = table.sha256.clone();
        let mut rng = Rng::seeded(5);
        let model = Ms2Model::<R, E>::init(&config, &device, &mut rng).unwrap();
        let constants = Ms2Constants::new(&device);
        let batch = spectra_batch(&comps, 64, 21);
        let gen_config = GenerationConfig {
            trajectories: k,
            max_steps: t,
            ..GenerationConfig::default()
        };
        let estimate = Ms2MemoryEstimate::generation(
            &model_config,
            table.rows as u64,
            b as u64,
            k as u64,
            64,
            gen_config.max_steps as u64,
            gen_config.formula_window as u64,
            gen_config.formulas as u64,
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
        println!("generation estimate total (b={b} k={k} t={t}): {estimate} bytes");
        println!("generation measured reserved increase (b={b} k={k} t={t}): {measured} bytes");
        println!("generation ratio measured/estimate (b={b} k={k} t={t}): {ratio:.3}");
        assert!(
            (0.5..=8.0).contains(&ratio),
            "generation reserved/estimate ratio {ratio:.3} outside 0.5..=8.0 \
             (estimate {estimate}, measured {measured}): the pool rounds and holds \
             matmul/autotune scratch, so reserved is a high-water mark, not the live sum"
        );
    }
}

#[test]
fn training_estimate_tracks_in_use_peak() {
    let _serial = serial();
    // The estimate is for a batch whose every target slot is occupied, which
    // is what the padded teacher pass allocates whatever the batch holds.
    // The compact pass allocates for the occupied slots only — none at all
    // in these unlabeled synthetic spectra — so it is switched off here and
    // the estimate is reconciled with the case it bounds.
    mamba3::models::ms2::train::set_compact_teacher(false);
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            mamba3::models::ms2::train::set_compact_teacher(true);
        }
    }
    let _restore = Restore;
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
                            lambda_assign: 0.0,
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
            32,
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

#[test]
fn conditioning_gold_embed_launches_only_in_composition() {
    let _serial = serial();
    // B1-fix2 finding 3/observation: structural proof, holding on every
    // backend, that `ScoredRowOrZero` launches zero gold-network kernels
    // while `Composition` launches more than zero. The gold row-network
    // call (`count_features` + `embed_rows` on the gold features) runs
    // under the `ms2.gold_embed` tally scope, entered only by the
    // `Composition` branch — so filtering the launch tally by that label
    // proves the property without comparing backend-sensitive total launch
    // counts. Label-filtered, so neighbouring tests (which never enter this
    // scope) cannot contaminate it.
    use mamba3::backend::{
        launch_tally_detailed, reset_launch_tally, start_launch_tally, stop_launch_tally,
    };
    let device = Device::<R>::default();
    let model_config = ModelConfig::v0();
    let comps = parent_comps(2);
    let host_table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let mk_trainer = |mode| {
        let train_config = TrainConfig {
            batch: 2,
            slots: 2,
            gold_formula_conditioning: mode,
                            lambda_assign: 0.0,
..TrainConfig::default()
    };
        Ms2Trainer::<R, E>::new(&model_config, &host_table, &train_config, &device).unwrap()
    };
    let mut trainer_zero = mk_trainer(GoldFormulaConditioning::ScoredRowOrZero);
    let mut trainer_comp = mk_trainer(GoldFormulaConditioning::Composition);
    let set = experiment_set(&comps, 64, 29);
    let indices = vec![0usize, 1];
    // Warm-up: compile kernels and settle tuning outside the tally windows.
    let _ = trainer_zero.conditioning_for_test(&set, &indices).unwrap();
    let _ = trainer_comp.conditioning_for_test(&set, &indices).unwrap();
    check_launches(&device).unwrap();
    let gold_launches = |rows: Vec<mamba3::backend::TallyRow>| {
        rows.into_iter()
            .filter(|row| row.label == "ms2.gold_embed")
            .map(|row| row.count)
            .sum::<usize>()
    };
    start_launch_tally();
    let _ = trainer_zero.conditioning_for_test(&set, &indices).unwrap();
    check_launches(&device).unwrap();
    let zero_gold = gold_launches(launch_tally_detailed());
    reset_launch_tally();
    let _ = trainer_comp.conditioning_for_test(&set, &indices).unwrap();
    check_launches(&device).unwrap();
    let comp_gold = gold_launches(launch_tally_detailed());
    stop_launch_tally();
    println!("gold_embed launches: ScoredRowOrZero {zero_gold}, Composition {comp_gold}");
    assert_eq!(
        zero_gold, 0,
        "ScoredRowOrZero launches no gold-network kernel"
    );
    assert!(
        comp_gold > 0,
        "Composition launches the gold row network ({comp_gold} gold_embed launches)"
    );
}

#[test]
fn warmed_training_step_pins_v0_launch_count() {
    let _serial = serial();
    // B1-fix2 item 6: a warmed training step at the V0 config
    // (`ModelConfig::v0`, default `TrainConfig` i.e. `ScoredRowOrZero`)
    // launches an exact pinned number of kernels on CPU. Any kernel added
    // to or removed from the V0 step fails here rather than drifting
    // silently.
    use mamba3::backend::{launch_count, reset_launch_count};
    let device = Device::<R>::default();
    let model_config = ModelConfig::v0();
    let comps = parent_comps(4);
    let host_table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let train_config = TrainConfig {
        batch: 4,
        slots: 16,
                    lambda_assign: 0.0,
..TrainConfig::default()
    };
    let mut trainer =
        Ms2Trainer::<R, E>::new(&model_config, &host_table, &train_config, &device).unwrap();
    let set = experiment_set(&comps, 64, 31);
    let indices: Vec<usize> = (0..4).collect();
    for _ in 0..2 {
        let _ = trainer.step(&set, &indices).unwrap();
    }
    check_launches(&device).unwrap();
    reset_launch_count();
    let _ = trainer.step(&set, &indices).unwrap();
    check_launches(&device).unwrap();
    let n = launch_count();
    println!("PINNED warmed V0 training step launches: {n}");
    // CPU-runtime number; wgpu (including the `vulkan`/`msl` builds, which
    // set the `wgpu` feature too) picks other kernels for the same ops, so it
    // pins its own measured number; any other backend prints and skips.
    if device.name() == "cpu" {
        // 1554 with the padded teacher pass and the padded encoder scans. The
        // ragged forms (the defaults) add the gathers into and out of their
        // packed rows, the gathers of the per-spectrum memory, the packed
        // layout's step lookup and the reset-aware scans, the masked slots that
        // round the attention memory up to whole vectors, and their adjoints.
        assert_eq!(n, 1609, "warmed V0 training step launches pinned on CPU");
    } else if device.name() == "wgpu" {
        assert_eq!(n, 920, "warmed V0 training step launches pinned on wgpu");
    } else {
        println!(
            "unpinned backend {}: warmed V0 training step launches {n}",
            device.name()
        );
    }
}

#[test]
fn training_enum_lane_refusal_leaves_counters_unchanged() {
    // D3: training refuses B*P > enum_lanes_max BEFORE any upload, bucket
    // allocation or encoder launch; the first refusal leaves allocation and
    // launch counters unchanged. Counter-owning binary behind its mutex, one
    // refused call, exact assertions.
    use mamba3::backend::{allocation_calls, launch_count};
    use mamba3::models::ms2::contract::FormulaSource;
    use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
    let _serial = serial();
    let device = Device::<R>::default();
    let comps = parent_comps(2);
    let host_table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let model_config = ModelConfig::v0();
    let mut train_config = TrainConfig {
        batch: 2,
        slots: 2,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        enum_lanes_max: 1,
                    lambda_assign: 0.0,
..TrainConfig::default()
    };
    // Lane limits are checkpointed (D3): round-trip through JSON keeps them.
    let json = serde_json::to_string(&train_config).unwrap();
    let back: TrainConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.enum_lanes_max, 1);
    assert_eq!(back.enum_lane_visits_max, train_config.enum_lane_visits_max);
    assert_eq!(back.enum_dispatch_visits_max, train_config.enum_dispatch_visits_max);
    let mut trainer =
        Ms2Trainer::<R, E>::new(&model_config, &host_table, &train_config, &device).unwrap();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let set = experiment_set(&comps, 64, 31);
    let indices = vec![0usize, 1];
    let a0 = allocation_calls();
    let l0 = launch_count();
    let err = trainer.step(&set, &indices).unwrap_err();
    assert!(
        matches!(err, mamba3::error::Error::Config(_)),
        "excessive lanes refusal is Config, got {err:?}"
    );
    assert!(err.to_string().contains("exceeds enum_lanes_max"), "{err}");
    assert_eq!(allocation_calls(), a0, "no allocation on first refusal");
    assert_eq!(launch_count(), l0, "no launch on first refusal");
}
