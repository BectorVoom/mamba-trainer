//! Integration of the `Evidence` formula-feature layout (task E3).
//!
//! End to end on `backends::Auto`: the `Counts` regression gates (no
//! evidence buffers, the search-stage launch pin, the parameter names), the
//! `Evidence` search path on both formula sources against the E1 host twins,
//! the zero-init branch equality, branch gradients, the inference boundary,
//! training, precursor jitter, config round trips and checkpoint mismatch.
//!
//! Counter-reading tests share this binary's process-global counters, so
//! every test holds `SERIAL` for its whole body and all counter assertions
//! live in the single `evidence_read_budgets` test.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{
    Device, launch_count, read_count, reset_launch_count, reset_read_count,
    reset_transfer_counters, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    Control, FormulaFeatures, FormulaSource, GenerationConfig, GenerationMode, ModelConfig,
    SCHEMA_VERSION, SCHEMA_VERSION_V1, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, apply_precursor_jitter,
    jittered_set_for_eval, spectrum_batch_for, spectrum_batch_with_donors,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{
    EnumDomain, RatioBounds, build_enum_meta,
};
use mamba3::models::ms2::formula_evidence::{
    EVIDENCE_PEAKS, evidence_peaks as twin_peaks, formula_evidence as twin_evidence,
    formula_features as twin_features,
};
use mamba3::models::ms2::formula_evidence_ref::{jitter_precursor_mz, jitter_seed};
use mamba3::models::ms2::formula_head::{DeviceEnumArtifacts, DeviceFormulaTable, FormulaHead};
use mamba3::models::ms2::generate::{GenerateStage, GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, FormulaBuffers, Ms2Constants, PeakBuffers};
use mamba3::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};
use mamba3::tensor::ops::ms2_formula_evidence;
use mamba3::tensor::ops::movement;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// File-level serialisation: the process-wide launch/read counters (and the
/// shared device allocator) are perturbed by any test running beside these,
/// so every test holds this mutex for its whole body. Poison-tolerant: a
/// panicking holder still releases the lock.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
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

fn tiny_model(layout: FormulaFeatures) -> ModelConfig {
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
    m.formula_features = layout;
    m
}

/// Tiny generation config matching the `tests/ms2_launch_budget.rs` pin
/// shape (K=4, F=2, T=22, Table, M=32) so the `Counts` search pin can be
/// compared directly.
fn tiny_gen_table(window: u32) -> GenerationConfig {
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
        formula_evidence_work_max: 2048,
        formula_evidence_dispatch_max: 268435456,
    }
}

fn tiny_gen_enum(window: u32) -> GenerationConfig {
    let mut g = tiny_gen_table(window);
    g.formula_source = FormulaSource::Enumerate;
    g
}

fn precursor_of(c: &Composition) -> u32 {
    composition_mass(c).unwrap() + 1_007_825 - 549
}

/// E4F item 2: the wrapper's callers pass true host-known bounds without a
/// read — the table's largest hydrogen count + 3, the artifacts' hydrogen
/// bound + 3, and the batch's largest fragment tolerance (recomputed here
/// with independent `u64` arithmetic).
#[test]
fn evidence_dispatch_bounds_are_true_host_bounds() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (domain, bounds, table) = setup_enum();
    let hand_max_h: u16 = comps.iter().map(|c| c[1]).max().unwrap_or(0);
    assert_eq!(table.max_hydrogen(), hand_max_h);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    assert_eq!(dtable.hydrogen_cap_max(), u32::from(hand_max_h) + 3);
    let art = DeviceEnumArtifacts::<R>::upload(&domain, &bounds, &device).unwrap();
    assert_eq!(art.hydrogen_max, domain.hydrogen_max);
    assert_eq!(art.hydrogen_cap_max(), u32::from(domain.hydrogen_max) + 3);
    // The enumerating source covers every fixture composition.
    for c in comps.iter() {
        assert!(domain.contains(c), "domain covers {c:?}");
        assert!(
            u32::from(c[1]) + 3 <= art.hydrogen_cap_max(),
            "artifact bound covers {c:?}"
        );
    }
    // The batch tolerance bound matches the independent computation.
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let b = comps.len();
    let batch = spectra_batch(
        &precursors,
        &vec![50; b],
        &vec![1; b],
        &vec![10; b],
        64,
        23,
    );
    let mut want_tol = 0u32;
    let n_raw = batch.n_raw as usize;
    for bi in 0..b {
        let ppm = batch.fragment_tolerance(bi);
        let mut mz_max = 0u32;
        for k in 0..(batch.peak_count[bi] as usize).min(n_raw) {
            mz_max = mz_max.max(batch.mz_udalton[bi * n_raw + k]);
        }
        if mz_max == 0 {
            continue;
        }
        let tol = ((u64::from(mz_max) * u64::from(ppm)) / 10_000_000) as u32;
        want_tol = want_tol.max(tol);
    }
    assert_eq!(batch.max_fragment_tolerance(), want_tol);
    assert!(want_tol > 0, "non-degenerate batch tolerance");
    // Search-stage sizing at the defaults (M = 32, W = 2,048,
    // dispatch_max = 2^28, P = 32) for both sources, for the record.
    for (source, h_cap_max) in [
        ("table", dtable.hydrogen_cap_max()),
        ("enumerate", art.hydrogen_cap_max()),
    ] {
        let s_max = (u64::from(h_cap_max) * 7_825 + 2 * u64::from(want_tol)) / 1_000_000;
        let per_s = (2 * u64::from(want_tol)) / 7_825 + 2;
        let trials = ((s_max + 1) * per_s).min(u64::from(h_cap_max) + 1);
        let per_lane = 2048u64 * 32 * trials;
        let lanes_per_launch = 268_435_456u64 / per_lane;
        let launches = (192u64).div_ceil(lanes_per_launch);
        println!(
            "E4F-SEARCH {source}: h_cap_max={h_cap_max} tol_max={want_tol} trials={trials} per_lane={per_lane} lanes_per_launch={lanes_per_launch} launches={launches}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn spectra_batch(
    precursors: &[u32],
    uncs: &[u32],
    adducts: &[u16],
    peak_counts: &[u32],
    n_raw: usize,
    seed: u64,
) -> SpectrumBatch {
    let b = precursors.len();
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
    layout: FormulaFeatures,
    domain: &EnumDomain,
    bounds: &RatioBounds,
    table: &FormulaTable,
    device: &Device<R>,
) -> (Ms2Model<R, E>, DeviceFormulaTable<R, E>, Ms2Constants<R>) {
    let dtable = DeviceFormulaTable::<R, E>::upload(table, device).unwrap();
    let mut cfg = tiny_model(layout);
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, E>::init(&cfg, device, &mut rng).unwrap();
    model.upload_enum_artifacts(domain, bounds, device).unwrap();
    let constants = Ms2Constants::new(device);
    (model, dtable, constants)
}

fn experiment_set_for(comps: &[Composition], n_raw: usize, seed: u64) -> ExperimentSet {
    experiment_set_for_unc(comps, n_raw, seed, 50)
}

fn experiment_set_for_unc(
    comps: &[Composition],
    n_raw: usize,
    seed: u64,
    unc: u32,
) -> ExperimentSet {
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
        precursor_uncertainty_udalton: vec![unc; b],
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
                    precursor_uncertainty_udalton: unc,
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
        name: "e3-integration".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: (0..b).map(|i| format!("mol{i}")).collect(),
        spectra,
    }
}

/// Reconstruct the `[B, 8]` meta words of [`DeviceSpectra::upload`] for a
/// batch with no fatal spectrum (every status bit clear): what
/// `evidence_peaks` and `formula_evidence` read on the device.
fn meta_host(batch: &SpectrumBatch) -> Vec<u32> {
    let b = batch.len();
    let mut meta = vec![0u32; b * 8];
    for i in 0..b {
        meta[i * 8] = batch.peak_count[i];
        meta[i * 8 + 1] = batch.precursor_mz_udalton[i];
        meta[i * 8 + 2] = batch.precursor_uncertainty_udalton[i];
        meta[i * 8 + 3] = u32::from(batch.adduct[i]);
        meta[i * 8 + 4] = batch.fragment_tolerance(i);
        meta[i * 8 + 5] = batch.precursor_tolerance(i);
        let id = batch.spectrum_id[i];
        meta[i * 8 + 6] = (id & 0xFFFF_FFFF) as u32;
        meta[i * 8 + 7] = (id >> 32) as u32;
    }
    meta
}

/// The `[B, 2]` m/z-uncertainty buffer the search stage builds like
/// `generate_ion` does: word 0 is the stored uncertainty, word 1 reserved.
fn spec_host(batch: &SpectrumBatch) -> Vec<u32> {
    let b = batch.len();
    let mut spec = vec![0u32; b * 2];
    for (i, v) in batch.mz_uncertainty_udalton.iter().enumerate().take(b) {
        spec[i * 2] = *v;
    }
    spec
}

fn log_table_host() -> Vec<f32> {
    (0..1024).map(|n| (1.0 + n as f32).ln()).collect()
}

#[test]
fn counts_layout_has_no_evidence_buffers_and_names() {
    let _serial = serial();
    let device = dev();
    // `Counts`: no evidence buffer is allocated ...
    let plain = FormulaBuffers::<R, E>::new(2, 32, 2, &device);
    assert!(!plain.is_evidence());
    assert!(plain.cand_xfeat.is_none());
    assert!(plain.ev_peaks.is_none());
    assert!(plain.ev_w.is_none());
    assert!(plain.cand_ev.is_none());
    assert!(plain.cand_feat16.is_none());
    let ev = FormulaBuffers::<R, E>::new_evidence(2, 32, 2, &device);
    assert!(ev.is_evidence());
    // ... and the parameter name list is unchanged: exactly the §1.2 names,
    // nothing with `evidence` in it.
    let mut rng = Rng::seeded(5);
    let head = FormulaHead::<R, E>::init(&tiny_model(FormulaFeatures::Counts), &device, &mut rng)
        .unwrap();
    let mut names: Vec<String> = head
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "pool_query.bias",
            "pool_query.weight",
            "row_in.bias",
            "row_in.weight",
            "row_out.bias",
            "row_out.weight",
        ],
        "Counts head parameters"
    );
    // The `Evidence` head adds exactly `evidence_in` / `evidence_out`.
    let mut rng = Rng::seeded(5);
    let ev_head =
        FormulaHead::<R, E>::init(&tiny_model(FormulaFeatures::Evidence), &device, &mut rng)
            .unwrap();
    assert!(ev_head.has_evidence_branch());
    let mut ev_names: Vec<String> = ev_head
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    ev_names.sort();
    assert_eq!(
        ev_names,
        vec![
            "evidence_in.bias",
            "evidence_in.weight",
            "evidence_out.bias",
            "evidence_out.weight",
            "pool_query.bias",
            "pool_query.weight",
            "row_in.bias",
            "row_in.weight",
            "row_out.bias",
            "row_out.weight",
        ],
        "Evidence head parameters"
    );
}

/// Fill the `Evidence` search buffers on the device for the table source and
/// read back `cand`, `cand_feat`, `cand_ev`, `cand_xfeat`, `kept` and
/// `kept_f` for the host-twin comparison.
#[allow(clippy::too_many_arguments)]
fn run_table_evidence(
    batch: &SpectrumBatch,
    dtable: &DeviceFormulaTable<R, E>,
    device: &Device<R>,
    m: usize,
    work_max: u32,
    dispatch_max: u64,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>) {
    let b = batch.len();
    let n = 16usize;
    let spectra = mamba3::models::ms2::batch::DeviceSpectra::<R, E>::upload(batch, device).unwrap();
    let peaks = PeakBuffers::<R, E>::new(b, batch.n_raw as usize, n, device);
    ms2::peak_select(
        &spectra.mz,
        &spectra.intensity,
        &spectra.meta,
        spectra.intensity_scale,
        &peaks,
    )
    .unwrap();
    let mut fbuf = FormulaBuffers::<R, E>::poisoned_evidence(b, m, 2, device).unwrap();
    ms2::formula_window(
        &dtable.table,
        &spectra.meta,
        dtable.max_error,
        u32::MAX,
        4096,
        &fbuf,
    )
    .unwrap();
    ms2::formula_gather(&fbuf.window, &dtable.table, &dtable.counts, &mut fbuf.cand).unwrap();
    let spec_h = spec_host(batch);
    let spec_t = IdTensor::from_slice(&spec_h, vec![b, 2], device).unwrap();
    {
        let (Some(ev_peaks), Some(ev_w)) = (fbuf.ev_peaks.as_mut(), fbuf.ev_w.as_mut()) else {
            panic!("evidence buffers");
        };
        ms2_formula_evidence::evidence_peaks(
            &peaks.kept,
            &peaks.kept_f,
            &spectra.meta,
            &spec_t,
            ev_peaks,
            ev_w,
        )
        .unwrap();
    }
    {
        let (Some(ev_peaks), Some(ev_w), Some(cand_ev)) = (
            fbuf.ev_peaks.as_ref(),
            fbuf.ev_w.as_ref(),
            fbuf.cand_ev.as_mut(),
        ) else {
            panic!("evidence buffers");
        };
        ms2_formula_evidence::formula_evidence(
            &fbuf.cand,
            ev_peaks,
            ev_w,
            &spectra.meta,
            &spec_t,
            cand_ev,
            work_max,
            dispatch_max,
            dtable.hydrogen_cap_max(),
            batch.max_fragment_tolerance(),
        )
        .unwrap();
    }
    {
        let (Some(cand_ev), Some(feat16)) =
            (fbuf.cand_ev.as_ref(), fbuf.cand_feat16.as_mut())
        else {
            panic!("evidence buffers");
        };
        ms2_formula_evidence::formula_features(
            &fbuf.cand,
            cand_ev,
            &spectra.meta,
            &dtable.log_table,
            feat16,
        )
        .unwrap();
    }
    let feat16_t = fbuf.cand_feat16.as_ref().unwrap().clone();
    fbuf.cand_feat = movement::slice(&feat16_t, 2, 0, 10).unwrap();
    fbuf.cand_xfeat = Some(movement::slice(&feat16_t, 2, 10, 6).unwrap());
    let cand = fbuf.cand.try_to_vec().unwrap();
    let cand_feat = fbuf.cand_feat.try_to_f32().unwrap();
    let cand_ev = fbuf.cand_ev.as_ref().unwrap().try_to_f32().unwrap();
    let cand_xfeat = fbuf.cand_xfeat.as_ref().unwrap().try_to_f32().unwrap();
    let kept = peaks.kept.try_to_vec().unwrap();
    let kept_f = peaks.kept_f.try_to_f32().unwrap();
    (cand, cand_feat, cand_ev, cand_xfeat, kept, kept_f)
}

#[test]
fn evidence_xfeat_matches_host_twins_table() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (domain, bounds, table) = setup_enum();
    let (_model, dtable, _constants) =
        setup_model(FormulaFeatures::Evidence, &domain, &bounds, &table, &device);
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let b = comps.len();
    let batch = spectra_batch(
        &precursors,
        &vec![50; b],
        &vec![1; b],
        &vec![10; b],
        64,
        23,
    );
    let m = 32usize;
    let (cand, cand_feat, cand_ev, cand_xfeat, kept, kept_f) =
        run_table_evidence(&batch, &dtable, &device, m, 2048, 268435456);
    // Host twins from the same request.
    let meta_h = meta_host(&batch);
    let spec_h = spec_host(&batch);
    let (ev_peaks_h, ev_w_h) = twin_peaks(&kept, &kept_f, &meta_h, &spec_h, b, 16, EVIDENCE_PEAKS);
    let cand_ev_h = twin_evidence(&cand, &ev_peaks_h, &ev_w_h, &meta_h, &spec_h, b, m, EVIDENCE_PEAKS, 2048, u32::MAX);
    let feat16_h = twin_features(&cand, &cand_ev_h, &meta_h, &log_table_host(), b, m);
    // Columns 0..10 are what `count_features` would write; 10..16 are the
    // six evidence features.
    for s in 0..b * m {
        assert_eq!(
            &cand_feat[s * 10..(s + 1) * 10],
            &feat16_h[s * 16..s * 16 + 10],
            "cand_feat slot {s}"
        );
        assert_eq!(
            &cand_xfeat[s * 6..(s + 1) * 6],
            &feat16_h[s * 16 + 10..(s + 1) * 16],
            "cand_xfeat slot {s}"
        );
        assert_eq!(
            &cand_ev[s * 4..(s + 1) * 4],
            &cand_ev_h[s * 4..(s + 1) * 4],
            "cand_ev slot {s}"
        );
    }
}

#[test]
fn evidence_xfeat_matches_host_twins_enum() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (domain, bounds, table) = setup_enum();
    let (_model, dtable, _constants) =
        setup_model(FormulaFeatures::Evidence, &domain, &bounds, &table, &device);
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let b = comps.len();
    let batch = spectra_batch(
        &precursors,
        &vec![50; b],
        &vec![1; b],
        &vec![10; b],
        64,
        23,
    );
    let m = 32usize;
    let n = 16usize;
    let spectra = mamba3::models::ms2::batch::DeviceSpectra::<R, E>::upload(&batch, &device).unwrap();
    let peaks = PeakBuffers::<R, E>::new(b, 64, n, &device);
    ms2::peak_select(
        &spectra.mz,
        &spectra.intensity,
        &spectra.meta,
        spectra.intensity_scale,
        &peaks,
    )
    .unwrap();
    let art = DeviceEnumArtifacts::<R>::upload(&domain, &bounds, &device).unwrap();
    let mut fbuf = FormulaBuffers::<R, E>::poisoned_evidence(b, m, 2, &device).unwrap();
    let scored_cap = 4096u32.min(m as u32);
    let enum_meta = build_enum_meta(&batch, art.domain_max_error, 65_536, scored_cap);
    let meta_t = IdTensor::from_slice(&enum_meta, vec![b, 8], &device).unwrap();
    let launch = EnumLaunch::from_chemistry();
    let mut lane_stats = IdTensor::empty(vec![b * art.p, 2], &device);
    let mut offsets = IdTensor::empty(vec![b * art.p], &device);
    launch
        .count(&meta_t, &art.rare, &art.bounds, &lane_stats, 262_144, 4_000_000, 65_536)
        .unwrap();
    enum_offsets(&lane_stats, &meta_t, &offsets, &fbuf.counters, scored_cap, m, 262_144).unwrap();
    launch
        .fill(&meta_t, &art.rare, &art.bounds, &offsets, &mut fbuf.cand, scored_cap, 262_144, 4_000_000, 65_536)
        .unwrap();
    cand_pad(&fbuf.counters, &mut fbuf.cand, art.p, 262_144).unwrap();
    let spec_h = spec_host(&batch);
    let spec_t = IdTensor::from_slice(&spec_h, vec![b, 2], &device).unwrap();
    {
        let (Some(ev_peaks), Some(ev_w)) = (fbuf.ev_peaks.as_mut(), fbuf.ev_w.as_mut()) else {
            panic!("evidence buffers");
        };
        ms2_formula_evidence::evidence_peaks(
            &peaks.kept,
            &peaks.kept_f,
            &spectra.meta,
            &spec_t,
            ev_peaks,
            ev_w,
        )
        .unwrap();
    }
    {
        let (Some(ev_peaks), Some(ev_w), Some(cand_ev)) = (
            fbuf.ev_peaks.as_ref(),
            fbuf.ev_w.as_ref(),
            fbuf.cand_ev.as_mut(),
        ) else {
            panic!("evidence buffers");
        };
        ms2_formula_evidence::formula_evidence(
            &fbuf.cand,
            ev_peaks,
            ev_w,
            &spectra.meta,
            &spec_t,
            cand_ev,
            2048,
            268435456,
            art.hydrogen_cap_max(),
            batch.max_fragment_tolerance(),
        )
        .unwrap();
    }
    {
        let (Some(cand_ev), Some(feat16)) =
            (fbuf.cand_ev.as_ref(), fbuf.cand_feat16.as_mut())
        else {
            panic!("evidence buffers");
        };
        ms2_formula_evidence::formula_features(
            &fbuf.cand,
            cand_ev,
            &spectra.meta,
            &dtable.log_table,
            feat16,
        )
        .unwrap();
    }
    let feat16_t = fbuf.cand_feat16.as_ref().unwrap().clone();
    fbuf.cand_feat = movement::slice(&feat16_t, 2, 0, 10).unwrap();
    fbuf.cand_xfeat = Some(movement::slice(&feat16_t, 2, 10, 6).unwrap());
    let cand = fbuf.cand.try_to_vec().unwrap();
    let cand_xfeat = fbuf.cand_xfeat.as_ref().unwrap().try_to_f32().unwrap();
    let cand_ev = fbuf.cand_ev.as_ref().unwrap().try_to_f32().unwrap();
    let kept = peaks.kept.try_to_vec().unwrap();
    let kept_f = peaks.kept_f.try_to_f32().unwrap();
    let meta_h = meta_host(&batch);
    let (ev_peaks_h, ev_w_h) = twin_peaks(&kept, &kept_f, &meta_h, &spec_h, b, n, EVIDENCE_PEAKS);
    let cand_ev_h = twin_evidence(&cand, &ev_peaks_h, &ev_w_h, &meta_h, &spec_h, b, m, EVIDENCE_PEAKS, 2048, u32::MAX);
    let feat16_h = twin_features(&cand, &cand_ev_h, &meta_h, &log_table_host(), b, m);
    for s in 0..b * m {
        assert_eq!(
            &cand_xfeat[s * 6..(s + 1) * 6],
            &feat16_h[s * 16 + 10..(s + 1) * 16],
            "enum cand_xfeat slot {s}"
        );
        assert_eq!(
            &cand_ev[s * 4..(s + 1) * 4],
            &cand_ev_h[s * 4..(s + 1) * 4],
            "enum cand_ev slot {s}"
        );
    }
}

#[test]
fn evidence_zero_branch_matches_counts() {
    let _serial = serial();
    let device = dev();
    let (domain, bounds, table) = setup_enum();
    let (counts_model, counts_table, constants) =
        setup_model(FormulaFeatures::Counts, &domain, &bounds, &table, &device);
    let (ev_model, ev_table, _) =
        setup_model(FormulaFeatures::Evidence, &domain, &bounds, &table, &device);
    // Same request for both layouts.
    let comps = fixture_comps();
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let b = comps.len();
    let batch = spectra_batch(
        &precursors,
        &vec![50; b],
        &vec![1; b],
        &vec![10; b],
        64,
        23,
    );
    for source in [FormulaSource::Table, FormulaSource::Enumerate] {
        let g = match source {
            FormulaSource::Table => tiny_gen_table(32),
            FormulaSource::Enumerate => tiny_gen_enum(32),
        };
        let mut ws_c = GenerationWorkspace::new();
        let out_c = counts_model
            .generate(&batch, &counts_table, &g, &mut ws_c, &constants)
            .unwrap();
        out_c.validate().unwrap();
        let mut ws_e = GenerationWorkspace::new();
        let out_e = ev_model
            .generate(&batch, &ev_table, &g, &mut ws_e, &constants)
            .unwrap();
        out_e.validate().unwrap();
        // The freshly initialised branch (zero output layer) adds exactly 0:
        // formula log-probabilities match the `Counts` model bit-for-bit up
        // to float association (1e-6 gate).
        assert_eq!(out_c.formula_log_prob.len(), out_e.formula_log_prob.len());
        let mut worst = 0.0f32;
        for (a, c) in out_c.formula_log_prob.iter().zip(out_e.formula_log_prob.iter()) {
            worst = worst.max((a - c).abs());
        }
        assert!(
            worst <= 1e-6,
            "{source:?}: worst formula_log_prob drift {worst}"
        );
    }
}

#[test]
fn evidence_branch_gradients_finite_difference() {
    let _serial = serial();
    let device = dev();
    let (b, m, d) = (2usize, 4usize, 16usize);
    let mut rng = Rng::seeded(5);
    let head =
        FormulaHead::<R, E>::init(&tiny_model(FormulaFeatures::Evidence), &device, &mut rng)
            .unwrap();
    let param = |head: &FormulaHead<R, E>, want: &str| {
        head.named_parameters()
            .into_iter()
            .find(|(n, _)| n == want)
            .unwrap_or_else(|| panic!("missing param {want}"))
            .1
    };
    // Non-zero output layer (the fresh init is zero, hence a dead branch).
    let out_w = param(&head, "evidence_out.weight");
    let out_shape = out_w.shape().dims().to_vec();
    let out_n = out_w.shape().dims().iter().product();
    out_w
        .set(Tensor::<R, E>::from_f32(&vec![0.1; out_n], out_shape.clone(), &device).unwrap());
    // Scored candidates with distinct evidence features.
    let mut cand_h = vec![0u32; b * m * 13];
    for s in 0..b * m {
        for e in 0..10 {
            cand_h[s * 13 + e] = ((s + e) % 4) as u32;
        }
        cand_h[s * 13 + 10] = 1_000_000 + s as u32;
        cand_h[s * 13 + 11] = 1;
        cand_h[s * 13 + 12] = 0;
    }
    let mut feat_h = vec![0f32; b * m * 10];
    for (i, v) in feat_h.iter_mut().enumerate() {
        *v = 0.1 * ((i % 10) + 1) as f32;
    }
    let base_x = [0.5f32, -0.2, 0.5, 0.3, 0.405, 1.0];
    let mut xfeat_h = vec![0f32; b * m * 6];
    for s in 0..b * m {
        for k in 0..6 {
            xfeat_h[s * 6 + k] = base_x[k] + 0.01 * s as f32;
        }
    }
    let mut buffers = FormulaBuffers::<R, E>::new_evidence(b, m, 1, &device);
    buffers.cand = IdTensor::from_slice(&cand_h, vec![b, m, 13], &device).unwrap();
    buffers.cand_feat = Tensor::<R, E>::from_f32(&feat_h, vec![b, m, 10], &device).unwrap();
    buffers.cand_xfeat =
        Some(Tensor::<R, E>::from_f32(&xfeat_h, vec![b, m, 6], &device).unwrap());
    let mut prng = Rng::seeded(23);
    let pool_h = prng.uniform_vec(b * d, -0.5, 0.5);
    let pool_of =
        |host: &[f32]| Var::constant(Tensor::<R, E>::from_f32(host, vec![b, d], &device).unwrap());
    let gold_t = IdTensor::from_slice(&[1u32, 3], vec![b], &device).unwrap();
    let loss_of = |head: &FormulaHead<R, E>, buffers: &FormulaBuffers<R, E>, host: &[f32]| {
        let out = head.score(buffers, &pool_of(host)).unwrap();
        head.loss(&out, &gold_t).unwrap().to_f32()[0]
    };
    // Analytic gradients into the branch parameters.
    let out = head.score(&buffers, &pool_of(&pool_h)).unwrap();
    let loss = head.loss(&out, &gold_t).unwrap();
    let grads = loss.backward_retain().unwrap();
    let in_w = param(&head, "evidence_in.weight");
    let analytic = grads.get(in_w.id()).unwrap().to_f32();
    assert!(
        analytic.iter().any(|&g| g != 0.0),
        "the loss reaches evidence_in"
    );
    let out_analytic = grads.get(out_w.id()).unwrap().to_f32();
    assert!(
        out_analytic.iter().any(|&g| g != 0.0),
        "the loss reaches evidence_out"
    );
    // Finite differences of the formula loss w.r.t. `evidence_in` weights.
    let shape = in_w.shape().dims().to_vec();
    let base = in_w.value().to_f32();
    for &idx in &[0usize, 1, 2] {
        let mut up = base.clone();
        up[idx] += 1e-2;
        in_w.set(Tensor::<R, E>::from_f32(&up, shape.clone(), &device).unwrap());
        let fu = loss_of(&head, &buffers, &pool_h);
        let mut down = base.clone();
        down[idx] -= 1e-2;
        in_w.set(Tensor::<R, E>::from_f32(&down, shape.clone(), &device).unwrap());
        let fd = loss_of(&head, &buffers, &pool_h);
        in_w.set(Tensor::<R, E>::from_f32(&base, shape.clone(), &device).unwrap());
        let numeric = (fu - fd) / 2e-2;
        let a = analytic[idx];
        assert!(
            (a - numeric).abs() <= 2e-2 * numeric.abs() + 1e-3,
            "evidence_in.weight[{idx}]: analytic={a} numeric={numeric}"
        );
    }
    // No gradient reaches the features: with the output layer zeroed the
    // branch is dead — perturbing the features or `evidence_in` leaves every
    // bit of the loss unchanged, and the `evidence_in` gradients are exactly
    // 0 — while an active branch is sensitive to the features (the single
    // path they take into the score).
    let zero_n = out_w.value().to_f32().len();
    out_w.set(Tensor::<R, E>::from_f32(&vec![0.0; zero_n], out_shape, &device).unwrap());
    let l0 = loss_of(&head, &buffers, &pool_h);
    let mut x_up = xfeat_h.clone();
    x_up[0] += 1.0;
    buffers.cand_xfeat =
        Some(Tensor::<R, E>::from_f32(&x_up, vec![b, m, 6], &device).unwrap());
    let l1 = loss_of(&head, &buffers, &pool_h);
    assert_eq!(
        l0.to_bits(),
        l1.to_bits(),
        "dead branch: features do not move the loss"
    );
    buffers.cand_xfeat =
        Some(Tensor::<R, E>::from_f32(&xfeat_h, vec![b, m, 6], &device).unwrap());
    let mut w_up = base.clone();
    w_up[0] += 1.0;
    in_w.set(Tensor::<R, E>::from_f32(&w_up, shape.clone(), &device).unwrap());
    let l2 = loss_of(&head, &buffers, &pool_h);
    assert_eq!(
        l0.to_bits(),
        l2.to_bits(),
        "dead branch: evidence_in does not move the loss"
    );
    in_w.set(Tensor::<R, E>::from_f32(&base, shape, &device).unwrap());
    let out2 = head.score(&buffers, &pool_of(&pool_h)).unwrap();
    let loss2 = head.loss(&out2, &gold_t).unwrap();
    let grads2 = loss2.backward_retain().unwrap();
    let zeroed: Vec<f32> = grads2.get(in_w.id()).unwrap().to_f32();
    assert!(
        zeroed.iter().all(|&g| g == 0.0),
        "dead branch: evidence_in gradients are exactly 0"
    );
}

#[test]
fn evidence_inference_boundary_ignores_gold() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_domain, _bounds, table) = setup_enum();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    // Same spectra (ids, peaks, precursors), different gold compositions.
    let set_a = experiment_set_for(&comps, 64, 31);
    let mut set_b = experiment_set_for(&comps, 64, 31);
    for s in &mut set_b.spectra {
        s.parent_composition = comp(6, 6, 0, 0);
    }
    let indices: Vec<usize> = (0..comps.len()).collect();
    let gcfg = tiny_gen_table(32);
    let a = trainer.generate_candidates(&set_a, &indices, &gcfg).unwrap();
    let b = trainer.generate_candidates(&set_b, &indices, &gcfg).unwrap();
    assert_eq!(a, b, "gold payload does not enter generation");
}

/// All counter assertions of this binary live in this one test: the
/// `Counts` search-stage pin, one device read per `Evidence` generation
/// call (table and enumerating source), the `Evidence` search launch counts
/// (printed for the report), and zero reads in a warmed non-report
/// training step.
#[test]
fn evidence_read_budgets() {
    let _serial = serial();
    let device = dev();
    let backend = device.name();
    // `Counts` search pin (the `tests/ms2_launch_budget.rs` shape B=2).
    let pin_comps: Vec<Composition> = vec![comp(6, 6, 0, 0), comp(3, 7, 1, 2)];
    let pin_table = FormulaTable::from_compositions(pin_comps.clone()).unwrap();
    let mut pin_cfg = tiny_model(FormulaFeatures::Counts);
    let pin_dtable = DeviceFormulaTable::<R, E>::upload(&pin_table, &device).unwrap();
    pin_cfg.formula_table.rows = pin_dtable.rows as u32;
    pin_cfg.formula_table.sha256 = pin_dtable.sha256.clone();
    let mut pin_rng = Rng::seeded(5);
    let pin_model = Ms2Model::<R, E>::init(&pin_cfg, &device, &mut pin_rng).unwrap();
    let pin_constants = Ms2Constants::new(&device);
    let mut pin_ws = GenerationWorkspace::new();
    let pin_gcfg = tiny_gen_table(32);
    let pin_precursors: Vec<u32> = pin_comps.iter().map(precursor_of).collect();
    let pin_batch = spectra_batch(&pin_precursors, &vec![50; 2], &vec![1; 2], &vec![10; 2], 64, 22);
    for _ in 0..2 {
        pin_model
            .generate(&pin_batch, &pin_dtable, &pin_gcfg, &mut pin_ws, &pin_constants)
            .unwrap();
    }
    device.synchronize();
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let mut stages: Vec<GenerateStage> = Vec::new();
    let mut launches: Vec<usize> = Vec::new();
    {
        let mut hook = |stage: GenerateStage| {
            stages.push(stage);
            launches.push(launch_count());
        };
        pin_model
            .generate_with_hook(
                &pin_batch,
                &pin_dtable,
                &pin_gcfg,
                &mut pin_ws,
                &pin_constants,
                Some(&mut hook),
            )
            .unwrap();
    }
    let l_search = launches[2] - launches[1];
    println!("E3-SEARCH-LAUNCHES {backend} counts-table {l_search}");
    assert_eq!(stages[2], GenerateStage::AfterSearch);
    if backend == "cpu" || backend == "wgpu" {
        assert_eq!(l_search, 33, "Counts search launches the pinned 33");
    }
    // `Evidence` generation reads exactly once per call, both sources, and
    // the search-stage launch counts are reported (not pinned).
    let comps = fixture_comps();
    let (domain, bounds, table) = setup_enum();
    let (ev_model, ev_table, ev_constants) =
        setup_model(FormulaFeatures::Evidence, &domain, &bounds, &table, &device);
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let b = comps.len();
    let batch = spectra_batch(
        &precursors,
        &vec![50; b],
        &vec![1; b],
        &vec![10; b],
        64,
        23,
    );
    for source in [FormulaSource::Table, FormulaSource::Enumerate] {
        let g = match source {
            FormulaSource::Table => tiny_gen_table(32),
            FormulaSource::Enumerate => tiny_gen_enum(32),
        };
        let mut ws = GenerationWorkspace::new();
        for _ in 0..2 {
            ev_model
                .generate(&batch, &ev_table, &g, &mut ws, &ev_constants)
                .unwrap();
        }
        device.synchronize();
        reset_launch_count();
        reset_read_count();
        let mut ws2 = ws;
        let mut stages_e: Vec<GenerateStage> = Vec::new();
        let mut launches_e: Vec<usize> = Vec::new();
        {
            let mut hook = |stage: GenerateStage| {
                stages_e.push(stage);
                launches_e.push(launch_count());
            };
            ev_model
                .generate_with_hook(&batch, &ev_table, &g, &mut ws2, &ev_constants, Some(&mut hook))
                .unwrap();
        }
        device.synchronize();
        // One more call reads exactly once (the delta of this call alone).
        let r0 = runtime_read_count();
        ev_model
            .generate(&batch, &ev_table, &g, &mut ws2, &ev_constants)
            .unwrap();
        device.synchronize();
        assert_eq!(
            runtime_read_count() - r0,
            1,
            "{source:?}: Evidence generate reads exactly once"
        );
        let l_search_e = launches_e[2] - launches_e[1];
        println!("E3-SEARCH-LAUNCHES {backend} evidence-{source:?} {l_search_e}");
        assert_eq!(stages_e[2], GenerateStage::AfterSearch);
    }
    // Warmed non-report training step reads nothing.
    let tcfg = TrainConfig {
        batch: b,
        slots: 2,
        lr: 3e-3,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        ..TrainConfig::default()
    };
    let mut tcfg_ev = tiny_model(FormulaFeatures::Evidence);
    let dt = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    tcfg_ev.formula_table.rows = dt.rows as u32;
    tcfg_ev.formula_table.sha256 = dt.sha256.clone();
    let mut trainer = Ms2Trainer::<R, E>::new(&tcfg_ev, &table, &tcfg, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let set = experiment_set_for(&comps, 64, 29);
    let indices: Vec<usize> = (0..b).collect();
    for _ in 0..3 {
        assert!(trainer.step(&set, &indices).unwrap().is_none());
    }
    for _ in 0..5 {
        let r0 = read_count();
        let rr0 = runtime_read_count();
        assert!(trainer.step(&set, &indices).unwrap().is_none());
        assert_eq!(read_count() - r0, 0, "warmed non-report step reads nothing");
        assert_eq!(
            runtime_read_count() - rr0,
            0,
            "warmed non-report step reads nothing (runtime)"
        );
    }
}

#[test]
fn precursor_jitter_keys() {
    let _serial = serial();
    let comps = fixture_comps();
    let set = experiment_set_for(&comps[..4], 64, 11);
    let indices: Vec<usize> = (0..4).collect();
    // `sigma = 0` is the identity: every byte unchanged.
    let plain = spectrum_batch_for(&set, &indices, 64).unwrap();
    let same = mamba3::models::ms2::experiment::spectrum_batch_for_with_jitter(
        &set, &indices, 64, 0.0, 99, 7,
    )
    .unwrap();
    assert_eq!(plain, same, "sigma = 0 leaves every byte unchanged");
    // `sigma = 2` changes precursors by at most 6 ppm ...
    let j1 = mamba3::models::ms2::experiment::spectrum_batch_for_with_jitter(
        &set, &indices, 64, 2.0, 99, 7,
    )
    .unwrap();
    let mut changed = 0usize;
    for b in 0..4 {
        let old = plain.precursor_mz_udalton[b] as f64;
        let new = j1.precursor_mz_udalton[b] as f64;
        let rel = ((new - old).abs()) / old;
        assert!(
            rel <= 6.01e-6,
            "spectrum {b}: relative change {rel} within 6 ppm"
        );
        if new != old {
            changed += 1;
        }
        // ... and nothing else moves.
        assert_eq!(j1.mz_uncertainty_udalton, plain.mz_uncertainty_udalton);
        assert_eq!(j1.mz_udalton, plain.mz_udalton);
        assert_eq!(j1.intensity, plain.intensity);
    }
    assert!(changed > 0, "sigma = 2 moves at least one precursor");
    // ... is reproducible for the same keys ...
    let j1b = mamba3::models::ms2::experiment::spectrum_batch_for_with_jitter(
        &set, &indices, 64, 2.0, 99, 7,
    )
    .unwrap();
    assert_eq!(j1, j1b, "same keys give the same draw");
    // ... and differs across steps.
    let j2 = mamba3::models::ms2::experiment::spectrum_batch_for_with_jitter(
        &set, &indices, 64, 2.0, 99, 8,
    )
    .unwrap();
    assert_ne!(
        j1.precursor_mz_udalton, j2.precursor_mz_udalton,
        "different split tags draw differently"
    );
    assert_ne!(
        jitter_seed(99, 7, 3, (2.0f64).to_bits()),
        jitter_seed(99, 8, 3, (2.0f64).to_bits()),
        "split tags seed differently"
    );
    // Evaluation draws are independent of batch order.
    let rev: Vec<usize> = vec![3, 2, 1, 0];
    let jrev = mamba3::models::ms2::experiment::spectrum_batch_for_with_jitter(
        &set, &rev, 64, 2.0, 99, 0,
    )
    .unwrap();
    let jfwd = mamba3::models::ms2::experiment::spectrum_batch_for_with_jitter(
        &set, &indices, 64, 2.0, 99, 0,
    )
    .unwrap();
    for (pos, &idx) in rev.iter().enumerate() {
        assert_eq!(
            jrev.precursor_mz_udalton[pos], jfwd.precursor_mz_udalton[idx],
            "spectrum {idx} draws the same in any batch order"
        );
    }
    // The shuffled-spectrum control keeps the recipient's precursor (keyed
    // by the recipient index) while the peaks follow the donor.
    let donors = vec![1usize, 2, 3, 0];
    let dj = mamba3::models::ms2::experiment::spectrum_batch_with_donors_with_jitter(
        &set, &indices, &donors, 64, 2.0, 99, 7,
    )
    .unwrap();
    let d0 =
        spectrum_batch_with_donors(&set, &indices, &donors, 64).unwrap();
    for b in 0..4 {
        let donor = &set.spectra[donors[b]].spectrum;
        assert_eq!(
            dj.precursor_mz_udalton[b],
            jitter_precursor_mz(d0.precursor_mz_udalton[b], 2.0, 99, 7, b as u64),
            "row {b}: precursor is the jittered recipient precursor"
        );
        assert_eq!(
            dj.mz_udalton[b * 64..(b + 1) * 64].to_vec(),
            donor_mz_row(donor, 64),
            "row {b}: peaks follow the donor"
        );
    }
    // `apply_precursor_jitter` with `sigma = 0` is a no-op on donor batches.
    let mut dcopy = d0.clone();
    apply_precursor_jitter(&mut dcopy, &indices, 0.0, 99, 7);
    assert_eq!(dcopy, d0);
    // `jittered_set_for_eval` with `sigma = 0` is identical; with sigma = 2
    // only the precursors move.
    let js0 = jittered_set_for_eval(&set, 0.0, 99);
    let b0 = spectrum_batch_for(&js0, &indices, 64).unwrap();
    assert_eq!(b0, plain);
    let js2 = jittered_set_for_eval(&set, 2.0, 99);
    for (i, s) in js2.spectra.iter().enumerate() {
        assert_eq!(
            s.spectrum.precursor_mz_udalton,
            jitter_precursor_mz(set.spectra[i].spectrum.precursor_mz_udalton, 2.0, 99, 0, i as u64),
            "eval draw keyed by (seed, spectrum)"
        );
        assert_eq!(s.spectrum.mz_udalton, set.spectra[i].spectrum.mz_udalton);
        assert_eq!(s.parent_composition, set.spectra[i].parent_composition);
    }
}

/// Donor peak m/z as a width-`n_raw` row (padded with 0).
fn donor_mz_row(donor: &ExportSpectrum, n_raw: usize) -> Vec<u32> {
    let mut row = vec![0u32; n_raw];
    for (k, &mz) in donor.mz_udalton.iter().enumerate() {
        row[k] = mz;
    }
    row
}

#[test]
fn config_round_trips_and_mismatch() {
    let _serial = serial();
    // Version-1 documents still load and mean `Counts`.
    let mut v = serde_json::to_value(&ModelConfig::v0()).unwrap();
    v["schema_version"] = serde_json::Value::from(SCHEMA_VERSION_V1);
    v.as_object_mut().unwrap().remove("formula_features");
    let m1: ModelConfig = serde_json::from_value(v).unwrap();
    assert_eq!(m1.formula_features, FormulaFeatures::Counts);
    assert!(m1.validate().is_ok());
    let mut g = serde_json::to_value(&GenerationConfig::default()).unwrap();
    g["schema_version"] = serde_json::Value::from(SCHEMA_VERSION_V1);
    for f in [
        "formula_source",
        "formula_window",
        "formula_evidence_work_max",
        "formula_evidence_dispatch_max",
    ] {
        g.as_object_mut().unwrap().remove(f);
    }
    let g1: GenerationConfig = serde_json::from_value(g).unwrap();
    assert_eq!(g1.formula_source, FormulaSource::Table);
    assert_eq!(g1.formula_window, 32);
    assert_eq!(g1.formula_evidence_work_max, 2048);
    assert_eq!(g1.formula_evidence_dispatch_max, 268435456);
    assert!(g1.validate(16, 4).is_ok());
    // Existing version-2 documents round-trip with the new defaults.
    let t = TrainConfig::default();
    let tback: TrainConfig = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(tback, t);
    assert!(tback.validate().is_ok());
    // Invalid new fields are `Error::Config`.
    let mut bad = GenerationConfig::default();
    bad.formula_evidence_work_max = 0;
    let err = bad.validate(16, 4).unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("formula_evidence_work_max"), "{err}");
    let mut bad = GenerationConfig::default();
    bad.formula_evidence_dispatch_max = 0;
    let err = bad.validate(16, 4).unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("formula_evidence_dispatch_max"), "{err}");
    for jitter in [-1.0f32, f32::NAN, 5.1] {
        let mut bad = TrainConfig::default();
        bad.precursor_jitter_ppm = jitter;
        let err = bad.validate().unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err}");
        assert!(err.to_string().contains("precursor_jitter_ppm"), "{err}");
    }
    let mut bad = TrainConfig::default();
    bad.formula_evidence_work_max = 0;
    let err = bad.validate().unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
    // Checkpoints round-trip per layout and refuse layout mismatch.
    let device = dev();
    let comps = fixture_comps();
    let table = FormulaTable::from_compositions(comps.into_iter()).unwrap();
    let ev_path = std::env::temp_dir().join("ms2_e3_evidence_roundtrip.json");
    let count_path = std::env::temp_dir().join("ms2_e3_counts_roundtrip.json");
    let tcfg = TrainConfig {
        batch: 2,
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let mut ev_cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    ev_cfg.formula_table.rows = dtable.rows as u32;
    ev_cfg.formula_table.sha256 = dtable.sha256.clone();
    let trainer = Ms2Trainer::<R, E>::new(&ev_cfg, &table, &tcfg, &device).unwrap();
    let before: Vec<(String, Vec<f32>)> = trainer
        .model
        .named_parameters()
        .into_iter()
        .map(|(n, p)| (n, p.value().to_f32()))
        .collect();
    assert!(
        before.iter().any(|(n, _)| n.contains("evidence_in")),
        "Evidence checkpoint carries the branch"
    );
    trainer.save(&ev_path).unwrap();
    let loaded = Ms2Trainer::<R, E>::load(&ev_path, &table, &device).unwrap();
    let after: Vec<(String, Vec<f32>)> = loaded
        .model
        .named_parameters()
        .into_iter()
        .map(|(n, p)| (n, p.value().to_f32()))
        .collect();
    assert_eq!(before, after, "Evidence checkpoint round-trips");
    let mut count_cfg = tiny_model(FormulaFeatures::Counts);
    count_cfg.formula_table.rows = dtable.rows as u32;
    count_cfg.formula_table.sha256 = dtable.sha256.clone();
    let err = match Ms2Trainer::<R, E>::load_with_config(&ev_path, &count_cfg, &table, &device) {
        Err(e) => e,
        Ok(_) => panic!("Evidence checkpoint into a Counts config must be refused"),
    };
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("formula_features"), "{err}");
    let count_trainer = Ms2Trainer::<R, E>::new(&count_cfg, &table, &tcfg, &device).unwrap();
    count_trainer.save(&count_path).unwrap();
    let err = match Ms2Trainer::<R, E>::load_with_config(&count_path, &ev_cfg, &table, &device) {
        Err(e) => e,
        Ok(_) => panic!("Counts checkpoint into an Evidence config must be refused"),
    };
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("formula_features"), "{err}");
    let _ = std::fs::remove_file(&ev_path);
    let _ = std::fs::remove_file(&count_path);
}

#[test]
fn evidence_training_loss_decreases() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (domain, bounds, table) = setup_enum();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        lr: 3e-3,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        enum_lane_visits_max: 65_536,
        ..TrainConfig::default()
    };
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let set = experiment_set_for_unc(&comps, 64, 29, 500_000);
    let indices: Vec<usize> = (0..comps.len()).collect();
    trainer.request_report();
    let first = trainer.step(&set, &indices).unwrap().expect("report");
    assert!(first.loss.is_finite() && first.formula.is_finite());
    assert!(
        first.formula_present == comps.len(),
        "gold is scored for every spectrum: {:?}",
        first
    );
    for _ in 0..28 {
        assert!(trainer.step(&set, &indices).unwrap().is_none());
    }
    trainer.request_report();
    let last = trainer.step(&set, &indices).unwrap().expect("report");
    assert!(last.loss.is_finite() && last.formula.is_finite());
    assert!(
        last.loss < first.loss,
        "30 Evidence steps decrease the loss: {} -> {}",
        first.loss,
        last.loss
    );
}
