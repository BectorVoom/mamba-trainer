//! Integration of the enumerating formula source (V1 §1.4, task I1).
//!
//! CPU-only domains (kept small: the CPU runtime executes lanes on the
//! host). No process-global counter reads here: footprint counters live in
//! their own binaries.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, FloatElem};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    Control, FormulaSource, GenerationConfig, GenerationMode, ModelConfig, SCHEMA_VERSION,
    SPECTRUM_SCHEMA_VERSION, SpectrumBatch, candidate_status, request_status,
};
use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{
    DeviceEnumLimits, EnumDomain, RatioBounds, enumerate_device_order, EnumQuery,
};
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn comp(c: u16, h: u16, n: u16, o: u16) -> Composition {
    [c, h, n, o, 0, 0, 0, 0, 0, 0]
}

fn fixture_comps() -> Vec<Composition> {
    vec![
        comp(0, 2, 0, 1),
        comp(1, 0, 0, 2),
        comp(1, 4, 0, 0),
        comp(2, 6, 0, 1),
        comp(6, 6, 0, 0),
        comp(6, 12, 0, 6),
    ]
}

fn tiny_model() -> ModelConfig {
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

fn tiny_gen(window: u32) -> GenerationConfig {
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
        formula_source: FormulaSource::Enumerate,
        formula_window: window,
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

fn precursor_of(c: &Composition) -> u32 {
    composition_mass(c).unwrap() + 1_007_825 - 549
}

fn spectra_batch(
    comps: &[Composition],
    precursors: &[u32],
    uncs: &[u32],
    adducts: &[u16],
    peak_counts: &[u32],
    n_raw: usize,
    seed: u64,
) -> SpectrumBatch {
    let b = comps.len();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for bi in 0..b {
        let count = peak_counts[bi] as usize;
        peak_count[bi] = peak_counts[bi];
        raw_peak_count[bi] = peak_counts[bi];
        for i in 0..count {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursors[bi].saturating_sub(5_000_000)) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + 2.0 * rng.uniform_vec(1, 0.0, 1.0)[0];
        }
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: (0..b as u64).map(|i| 900 + i).collect(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursors.to_vec(),
        precursor_uncertainty_udalton: uncs.to_vec(),
        adduct: adducts.to_vec(),
        polarity: vec![1; b],
        collision_energy_ev: vec![30.0; b],
        collision_energy_known: vec![1; b],
        energy_count: vec![1; b],
        fragment_tolerance_ppm_tenths: vec![0; b],
        precursor_tolerance_ppm_tenths: vec![0; b],
        instrument_class: vec![0; b],
    }
}

fn setup_enum() -> (EnumDomain, RatioBounds, FormulaTable) {
    let comps = fixture_comps();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let table = FormulaTable::from_compositions(comps.into_iter()).unwrap();
    (domain, bounds, table)
}

fn setup_model(
    domain: &EnumDomain,
    bounds: &RatioBounds,
    table: &FormulaTable,
    device: &Device<R>,
) -> (Ms2Model<R, E>, DeviceFormulaTable<R, E>, Ms2Constants<R>) {
    let dtable = DeviceFormulaTable::<R, E>::upload(table, device).unwrap();
    let mut cfg = tiny_model();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, E>::init(&cfg, device, &mut rng).unwrap();
    model.upload_enum_artifacts(domain, bounds, device).unwrap();
    let constants = Ms2Constants::new(device);
    (model, dtable, constants)
}

#[test]
fn enumerate_special_rows_have_contract_statuses() {
    let device = dev();
    let (domain, bounds, table) = setup_enum();
    let (model, dtable, constants) = setup_model(&domain, &bounds, &table, &device);
    // B = 4: normal, wide window, unknown precision, failed row.
    let normal = comp(6, 6, 0, 0);
    let comps = vec![normal, normal, normal, normal];
    let precursors = vec![
        precursor_of(&normal),
        precursor_of(&normal),
        precursor_of(&normal),
        precursor_of(&normal),
    ];
    // Wide window via huge uncertainty; unknown via MAX; failed via peak 0.
    let uncs = vec![50, 5_000_000, u32::MAX, 50];
    let adducts = vec![1, 1, 1, 1];
    let peak_counts = vec![10, 10, 10, 0];
    let batch = spectra_batch(&comps, &precursors, &uncs, &adducts, &peak_counts, 64, 21);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_gen(32);
    let out = model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    out.validate().unwrap();
    // Row 0 (normal): unaffected by the other rows' special cases.
    assert_eq!(out.formula_source[0], 1);
    // Row 1 (wide): D4 exhausted with complete=0 and NO absent; abstains.
    assert!(
        out.request_status[1] & request_status::FORMULA_SEARCH_EXHAUSTED != 0,
        "wide window sets exhausted: {:b}",
        out.request_status[1]
    );
    assert_eq!(
        out.request_status[1] & request_status::FORMULA_ABSENT, 0,
        "wide window has NO formula_absent"
    );
    assert_eq!(out.rows_joined[1], 0);
    assert_eq!(out.rows_scored[1], 0);
    assert_eq!(out.formula_support_complete[1], 0);
    for kk in 0..4 {
        let r = 1 * 4 + kk;
        assert_eq!(out.length[r], 0);
        assert_eq!(
            out.status[r] & candidate_status::REQUEST_FAILED,
            candidate_status::REQUEST_FAILED,
            "wide window abstains (no trajectory starts)"
        );
    }
    // Row 2 (unknown): D4 unavailable+absent with complete=0; abstains.
    assert!(
        out.request_status[2] & request_status::EXACT_MASS_UNAVAILABLE != 0,
        "unknown precision sets exact_mass_unavailable"
    );
    assert!(
        out.request_status[2] & request_status::FORMULA_ABSENT != 0,
        "unknown precision sets formula_absent"
    );
    assert_eq!(out.rows_joined[2], 0);
    assert_eq!(out.rows_scored[2], 0);
    assert_eq!(out.formula_support_complete[2], 0);
    for kk in 0..4 {
        let r = 2 * 4 + kk;
        assert_eq!(out.length[r], 0);
        assert_eq!(
            out.status[r] & candidate_status::REQUEST_FAILED,
            candidate_status::REQUEST_FAILED
        );
    }
    // Row 3 (failed peak_count 0): fatal empty, no candidate.
    assert!(
        out.request_status[3] & request_status::FATAL_MASK != 0,
        "failed row is fatal"
    );
    for kk in 0..4 {
        let r = 3 * 4 + kk;
        assert_eq!(out.formula_rank[r], u32::MAX);
        assert!(out.formula_counts[r * 10..r * 10 + 10].iter().all(|&c| c == 0));
        assert_eq!(out.status[r] & candidate_status::REQUEST_FAILED, candidate_status::REQUEST_FAILED);
    }
}

#[test]
fn enumerate_mass_overflow_has_no_candidate() {
    let device = dev();
    let (domain, bounds, table) = setup_enum();
    let (model, dtable, constants) = setup_model(&domain, &bounds, &table, &device);
    let normal = comp(6, 6, 0, 0);
    // Adduct 0 (unknown) makes parent_mass fail -> mass_overflow.
    let comps = vec![normal, normal];
    let precursors = vec![precursor_of(&normal), precursor_of(&normal)];
    let uncs = vec![50, 50];
    let adducts = vec![1, 0];
    let peak_counts = vec![10, 10];
    let batch = spectra_batch(&comps, &precursors, &uncs, &adducts, &peak_counts, 64, 22);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_gen(32);
    let out = model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    out.validate().unwrap();
    // D4: overflow is mass_overflow with complete=0 and NO absent; abstains.
    assert!(
        out.request_status[1] & request_status::MASS_OVERFLOW != 0,
        "bad parent sets mass_overflow: {:b}",
        out.request_status[1]
    );
    assert_eq!(
        out.request_status[1] & request_status::FORMULA_ABSENT, 0,
        "overflow has NO formula_absent"
    );
    assert_eq!(out.formula_support_complete[1], 0);
    assert_eq!(out.rows_scored[1], 0);
    for kk in 0..4 {
        let r = 1 * 4 + kk;
        assert_eq!(out.formula_rank[r], u32::MAX);
        assert_eq!(out.length[r], 0);
        assert_eq!(
            out.status[r] & candidate_status::REQUEST_FAILED,
            candidate_status::REQUEST_FAILED
        );
    }
    // Other row unaffected.
    assert_eq!(out.formula_source[0], 1);
}

#[test]
fn enumerate_generate_matches_host_twin() {
    let device = dev();
    let (domain, bounds, table) = setup_enum();
    let (model, dtable, constants) = setup_model(&domain, &bounds, &table, &device);
    let comps = fixture_comps();
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let b = comps.len();
    let uncs = vec![50; b];
    let adducts = vec![1; b];
    let peak_counts = vec![10; b];
    let batch = spectra_batch(&comps, &precursors, &uncs, &adducts, &peak_counts, 64, 23);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_gen(32);
    let out = model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    out.validate().unwrap();
    for bb in 0..b {
        assert_eq!(out.formula_source[bb], 1);
        // Host twin for this spectrum.
        let query = EnumQuery {
            precursor_mz: precursors[bb],
            adduct: 1,
            ppm_tenths: 200,
            precursor_uncertainty: 50,
        };
        let limits = DeviceEnumLimits {
            lane_visits_max: 65_536,
            scored_cap: 32,
        };
        let twin = enumerate_device_order(&domain, &bounds, &query, &limits).unwrap();
        assert_eq!(out.rows_joined[bb] as u64, twin.joined as u64);
        assert_eq!(out.rows_scored[bb] as u64, twin.scored as u64);
        // Every record's rank/counts match the twin's scored candidate.
        for kk in 0..4 {
            let r = bb * 4 + kk;
            let rank = out.formula_rank[r];
            if rank == u32::MAX {
                assert!(out.formula_counts[r * 10..r * 10 + 10].iter().all(|&c| c == 0));
                assert_eq!(out.formula_row[r], u32::MAX);
            } else {
                assert!((rank as usize) < twin.compositions.len());
                let want = twin.compositions[rank as usize];
                let got = &out.formula_counts[r * 10..r * 10 + 10];
                for e in 0..10 {
                    assert_eq!(got[e], want[e], "b={bb} k={kk} e={e}");
                }
                assert_eq!(out.formula_row[r], u32::MAX);
                assert_eq!(out.formula_source[bb], 1);
            }
        }
        // Gold in the scored support exactly when the twin says so: the
        // twin rank (if any) lies below the device rows_scored, which
        // equals the twin scored count (checked above). Retained top-F may
        // drop it (neural ranking), so the retained candidates are not the
        // check.
        let twin_rank = twin
            .compositions
            .iter()
            .position(|c| c == &comps[bb]);
        let gold_in_twin = twin_rank.is_some();
        if let Some(rank) = twin_rank {
            assert!(
                (rank as u32) < out.rows_scored[bb],
                "b={bb} twin gold rank {rank} below rows_scored {}",
                out.rows_scored[bb]
            );
        }
        // When the twin lacks the gold, nothing more is asserted here (the
        // training gold_slot test pins the rank equality).
        let _ = gold_in_twin;
    }
}

fn experiment_set_for(comps: &[Composition], n_raw: usize, seed: u64) -> ExperimentSet {
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let mut rng = Rng::seeded(seed);
    let b = comps.len();
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
    let batch = SpectrumBatch {
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
    };
    let spectra = (0..b)
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
                    intensity: batch.intensity[base..base + np].iter().map(|&v| v as f64).collect(),
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
        name: "enum-integration".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: (0..b).map(|i| format!("mol{i}")).collect(),
        spectra,
    }
}

#[test]
fn enumerate_training_gold_slot_matches_twin() {
    let device = dev();
    let comps = fixture_comps();
    let (domain, bounds, table) = setup_enum();
    let train_config = TrainConfig {
        batch: comps.len(),
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
                    lambda_assign: 0.0,
..TrainConfig::default()
    };
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &train_config, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let set = experiment_set_for(&comps, 64, 29);
    let indices: Vec<usize> = (0..comps.len()).collect();
    // Forward conditioning hook reads gold slots; compare with the host twin.
    let cond = trainer.conditioning_for_test(&set, &indices).unwrap();
    for (bb, slot) in cond.slots.iter().enumerate() {
        let query = EnumQuery {
            precursor_mz: precursor_of(&comps[bb]),
            adduct: 1,
            ppm_tenths: 200,
            precursor_uncertainty: 50,
        };
        let limits = DeviceEnumLimits {
            lane_visits_max: 65_536,
            scored_cap: 32,
        };
        let twin = enumerate_device_order(&domain, &bounds, &query, &limits).unwrap();
        let want = twin
            .compositions
            .iter()
            .position(|c| c == &comps[bb])
            .map(|i| i as u32)
            .unwrap_or(u32::MAX);
        assert_eq!(*slot, want, "b={bb} gold_slot");
    }
    // Loss finite on a reporting step.
    trainer.request_report();
    let report = trainer.step(&set, &indices).unwrap().expect("report");
    assert!(report.loss.is_finite());
    assert!(report.formula.is_finite());
}

#[test]
fn enumerate_completed_absence_sets_absent_only() {
    // D4: formula_absent only for a COMPLETED search that joined nothing
    // (plus the unknown rule). A precursor far from every enumerated formula
    // completes with joined==0, exhausted==0, absent and complete==1, and
    // abstains (no trajectory starts, request_failed).
    let device = dev();
    let (domain, bounds, table) = setup_enum();
    let (model, dtable, constants) = setup_model(&domain, &bounds, &table, &device);
    // C7H8 is outside the fixture domain caps (max C6), so its window joins
    // nothing with a normal budget.
    let absent_comp = comp(7, 8, 0, 0);
    let comps = vec![absent_comp];
    let precursors = vec![precursor_of(&absent_comp)];
    let uncs = vec![50];
    let adducts = vec![1];
    let peak_counts = vec![10];
    let batch = spectra_batch(&comps, &precursors, &uncs, &adducts, &peak_counts, 64, 99);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_gen(32);
    let out = model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    out.validate().unwrap();
    assert!(
        out.request_status[0] & request_status::FORMULA_ABSENT != 0,
        "completed empty search sets absent"
    );
    assert_eq!(
        out.request_status[0] & request_status::FORMULA_SEARCH_EXHAUSTED, 0,
        "completed absence is not exhausted"
    );
    assert_eq!(out.rows_joined[0], 0);
    assert_eq!(out.rows_scored[0], 0);
    assert_eq!(out.formula_support_complete[0], 1, "completed absence is complete");
    for kk in 0..4 {
        let r = kk;
        assert_eq!(out.length[r], 0);
        assert_eq!(
            out.status[r] & candidate_status::REQUEST_FAILED,
            candidate_status::REQUEST_FAILED
        );
    }
}

#[test]
fn packed_and_resident_accept_enumeration_counters() {
    // Finding R1-C2 (device leg): legal enumeration output validates through
    // `generate`, `generate_packed`, `generate_resident` + read and host
    // `pack`, because packed validation uses the SAME source-specific
    // counter rule as unpacked validation (one shared function): with the
    // enumerating source `joined` may exceed `visited` (up to 4 hydrogen
    // counts per visited heavy vector).
    use mamba3::models::ms2::pack::{ScoreKind, pack};
    let device = dev();
    let (domain, bounds, table) = setup_enum();
    let (model, dtable, constants) = setup_model(&domain, &bounds, &table, &device);
    let comps = fixture_comps();
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let b = comps.len();
    let batch = spectra_batch(
        &comps,
        &precursors,
        &vec![50; b],
        &vec![1; b],
        &vec![10; b],
        64,
        23,
    );
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_gen(32);
    let out = model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    out.validate().unwrap();
    let packed = model
        .generate_packed(&batch, &dtable, &gcfg, &mut ws, &constants)
        .unwrap();
    packed.validate().unwrap();
    let resident = model
        .generate_resident(&batch, &dtable, &gcfg, &mut ws, &constants)
        .unwrap();
    let read = resident.read(&model).unwrap();
    read.validate().unwrap();
    assert_eq!(read, packed, "the resident read equals the packed read");
    resident.release_into(&mut ws);
    let want = pack(&out, None, ScoreKind::Raw, gcfg.effective_returned() as usize).unwrap();
    want.validate().unwrap();
    for bb in 0..b {
        assert_eq!(out.formula_source[bb], 1, "spectrum {bb} enumerates");
        for (name, visited, joined, scored) in [
            ("generate", out.rows_visited[bb], out.rows_joined[bb], out.rows_scored[bb]),
            ("generate_packed", packed.rows_visited[bb], packed.rows_joined[bb], packed.rows_scored[bb]),
            ("resident read", read.rows_visited[bb], read.rows_joined[bb], read.rows_scored[bb]),
            ("host pack", want.rows_visited[bb], want.rows_joined[bb], want.rows_scored[bb]),
        ] {
            assert!(
                (joined as u64) <= (visited as u64) * 4,
                "{name} spectrum {bb}: joined {joined} <= 4 * visited {visited}"
            );
            assert!(
                scored <= joined,
                "{name} spectrum {bb}: scored {scored} <= joined {joined}"
            );
        }
        assert_eq!(packed.rows_visited[bb], out.rows_visited[bb]);
        assert_eq!(packed.rows_joined[bb], out.rows_joined[bb]);
        assert_eq!(packed.rows_scored[bb], out.rows_scored[bb]);
    }
}
