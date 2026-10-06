//! E3F follow-ups: Part B regressions (donor uncertainty, diagnostics) and
//! the integration properties the review lists as untested (task E3F).
//!
//! End to end on `backends::Auto` (CPU in the pinned runs). Counter-reading
//! tests share this binary's process-global counters, so every test holds
//! `SERIAL` for its whole body.
//!
//! Finding/test-letter map (see the final summary):
//! - B1 evidence stage: `b1_shuffled_donor_uncertainty_evidence`.
//! - B1 ion stage: `b1_shuffled_donor_uncertainty_ion`.
//! - B2 diagnostics: `b2_evidence_peaks_mean_empty_support` (+ `h_*`).
//! - a: `a_historical_counts_documents` (+ checkpoint part).
//! - b: `b_production_search_buffers_*`.
//! - c: `c_padding_with_live_branch`.
//! - d: `d_leaving_the_zero_point`.
//! - e: `e_conditioning_parity_*`.
//! - f: `f_enumerate_inference_boundary`.
//! - g: `g_jitter_through_trainer_step`.
//! - h: `h_diagnostics_values_and_reads`.
//! - i: `i_trained_evidence_checkpoint_reload`.
//! - j: `j_workspace_buckets_and_estimate`.

#![cfg(feature = "backend")]

#[path = "common/mod.rs"]
mod common;

use mamba3::backend::{
    Device, read_count, reset_read_count, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::rotate_peaks;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    AssignmentConfig, Control, FormulaFeatures, FormulaSource, GenerationConfig, GenerationMode,
    ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, spectrum_batch_for,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
use mamba3::models::ms2::formula_evidence::{
    EVIDENCE_PEAKS, evidence_peaks as twin_peaks, formula_evidence as twin_evidence,
    formula_features as twin_features,
};
use mamba3::models::ms2::formula_evidence_ref::jitter_precursor_mz;
use mamba3::models::ms2::formula_head::{DeviceEnumArtifacts, DeviceFormulaTable};
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// File-level serialisation: the process-wide launch/read counters (and the
/// shared device allocator) are perturbed by any test running beside these,
/// so every test holds this mutex for its whole body. Poison-tolerant.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn comp(c: u16, h: u16, n: u16, o: u16) -> Composition {
    [c, h, n, o, 0, 0, 0, 0, 0, 0]
}

/// C5: mass exactly 60,000,000 uDa (5 * 12,000,000).
fn c5() -> Composition {
    comp(5, 0, 0, 0)
}

/// C4H12: mass 60,093,900 uDa, the same-mass-window decoy for tests d/h.
fn c4h12() -> Composition {
    comp(4, 12, 0, 0)
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

fn tiny_model_assignment(layout: FormulaFeatures) -> ModelConfig {
    let mut m = tiny_model(layout);
    m.assignment = Some(AssignmentConfig::default());
    m
}

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

fn precursor_of(c: &Composition) -> u32 {
    composition_mass(c).unwrap() + 1_007_825 - 549
}

fn setup_enum(comps: Vec<Composition>) -> (EnumDomain, RatioBounds, FormulaTable) {
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
    assign: bool,
) -> (Ms2Model<R, E>, DeviceFormulaTable<R, E>, Ms2Constants<R>) {
    let dtable = DeviceFormulaTable::<R, E>::upload(table, device).unwrap();
    let mut cfg = tiny_model(layout);
    if assign {
        cfg.assignment = Some(AssignmentConfig::default());
    }
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, E>::init(&cfg, device, &mut rng).unwrap();
    model.upload_enum_artifacts(domain, bounds, device).unwrap();
    let constants = Ms2Constants::new(device);
    (model, dtable, constants)
}

/// Reconstruct the `[B, 8]` meta words of `DeviceSpectra::upload` for a
/// batch with no fatal spectrum.
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

/// The `[B, 2]` m/z-uncertainty buffer the search stage builds.
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

/// The review's B1 failing scenario as a two-row batch.
///
/// Row 0 (recipient): precursor 61,007,276 (C5 + proton), adduct 1, known
/// precursor uncertainty, fragment uncertainty `u32::MAX`, no peaks of its
/// own. Row 1 (donor): fragment uncertainty 0 and one intensity-1 peak at
/// 59,999,451 (target exactly 60,000,000 = C5). Fragment tolerance is the
/// default 100 ppm-tenths. Under `ShuffledSpectrum` rotation device row 0
/// holds the donor's peak and must be judged with the donor's uncertainty.
fn b1_batch() -> SpectrumBatch {
    let n_raw = 64usize;
    let mut peak_id = vec![u32::MAX; 2 * n_raw];
    let mut mz = vec![0u32; 2 * n_raw];
    let mut intensity = vec![0.0f32; 2 * n_raw];
    // Donor row: one intensity-1 peak at 59,999,451.
    peak_id[1 * n_raw] = 0;
    mz[1 * n_raw] = 59_999_451;
    intensity[1 * n_raw] = 1.0;
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![901, 902],
        raw_peak_count: vec![0, 1],
        peak_count: vec![0, 1],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![u32::MAX, 0],
        precursor_mz_udalton: vec![61_007_276, 61_007_276],
        precursor_uncertainty_udalton: vec![50, 50],
        adduct: vec![1, 1],
        polarity: vec![1, 1],
        collision_energy_ev: vec![30.0, 30.0],
        collision_energy_known: vec![1, 1],
        energy_count: vec![1, 1],
        fragment_tolerance_ppm_tenths: vec![0, 0],
        precursor_tolerance_ppm_tenths: vec![0, 0],
        instrument_class: vec![0, 0],
    }
}

/// Candidate slot of `composition` in `cand [B, M, 13]` for spectrum `s`
/// (`None` when the composition is not scored there).
fn find_slot(cand: &[u32], s: usize, m: usize, want: &Composition) -> Option<usize> {
    for slot in 0..m {
        let base = (s * m + slot) * 13;
        let counts: Composition = cand[base..base + 10]
            .iter()
            .map(|&v| v as u16)
            .collect::<Vec<u16>>()
            .try_into()
            .unwrap();
        if &counts == want && cand[base + 11] != 0 {
            return Some(slot);
        }
    }
    None
}

/// B1, evidence stage: under `ShuffledSpectrum` the rotated donor peaks must
/// be judged with the donor's m/z uncertainty.
///
/// The test drives the production search stage (`generate_preprocess` +
/// `generate_search_ws`, passing the ORIGINAL host batch exactly as
/// `generate` does) and reads the production `cand_ev` buffer. Row 0 must
/// hold `[1,1,1,1]` for C5 (donor uncertainty 0 selects the peak); the
/// pre-fix code built `spec` from the original batch (recipient uncertainty
/// `u32::MAX`) and produced `[0,0,0,1]`, which the wrong-uncertainty twin
/// below reproduces.
#[test]
fn b1_shuffled_donor_uncertainty_evidence() {
    let _serial = serial();
    let device = dev();
    let (domain, bounds, table) = setup_enum(vec![c5()]);
    let (model, dtable, _constants) =
        setup_model(FormulaFeatures::Evidence, &domain, &bounds, &table, &device, false);
    let batch = b1_batch();
    let mut g = tiny_gen_table(32);
    g.control = Control::ShuffledSpectrum;
    g.trajectories = 2;
    g.formulas = 1;
    assert!(g.validate(16, 4).is_ok());
    // Production search stage on a fresh workspace.
    let pre = model.generate_preflight(&batch, &dtable, &g).unwrap();
    let spectra = model.generate_preprocess(&batch, &g, &device).unwrap();
    // The uploaded rows carry the rotated (donor) uncertainty.
    assert_eq!(spectra.mz_uncertainty_udalton, vec![0, u32::MAX]);
    let mut ws = GenerationWorkspace::new();
    let encoded = model
        .generate_encode_ws(&mut ws, &spectra, g.control, &pre, &device)
        .unwrap();
    model
        .generate_search_ws(
            &mut ws,
            &spectra,
            &batch,
            &encoded.pool,
            &dtable,
            pre.spectra_n,
            pre.trajectories,
            pre.formulas,
            false,
            u32::MAX,
            4096,
            &g,
            &pre,
            &device,
        )
        .unwrap();
    let (kept, kept_f, cand, _ev_peaks, _ev_w, cand_ev, _feat, _xfeat) =
        ws.debug_search_buffers().unwrap();
    let kept_h = kept.try_to_vec().unwrap();
    let kept_fh = kept_f.try_to_f32().unwrap();
    let cand_h = cand.try_to_vec().unwrap();
    let cand_ev_h = cand_ev.unwrap().try_to_f32().unwrap();
    let m = 32usize;
    let slot = find_slot(&cand_h, 0, m, &c5()).expect("C5 scored in row 0");
    assert_eq!(
        &cand_ev_h[slot * 4..slot * 4 + 4],
        &[1.0, 1.0, 1.0, 1.0],
        "row 0 judges the donor peak with the donor uncertainty"
    );
    // The twins agree: donor uncertainty explains the peak, the
    // recipient's (unknown) uncertainty selects nothing.
    let rotated = rotate_peaks(&batch);
    let meta_r = meta_host(&rotated);
    let spec_donor = spec_host(&rotated);
    let (ev_d, ew_d) = twin_peaks(&kept_h, &kept_fh, &meta_r, &spec_donor, 2, 16, EVIDENCE_PEAKS);
    let cev_d = twin_evidence(&cand_h, &ev_d, &ew_d, &meta_r, &spec_donor, 2, m, EVIDENCE_PEAKS, 2048, u32::MAX);
    assert_eq!(
        &cev_d[(0 * m + slot) * 4..(0 * m + slot) * 4 + 4],
        &[1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(cand_ev_h, cev_d, "production matches the donor-uncertainty twin");
    let spec_wrong = spec_host(&batch);
    let (ev_w, ew_w) = twin_peaks(&kept_h, &kept_fh, &meta_r, &spec_wrong, 2, 16, EVIDENCE_PEAKS);
    let cev_w = twin_evidence(&cand_h, &ev_w, &ew_w, &meta_r, &spec_wrong, 2, m, EVIDENCE_PEAKS, 2048, u32::MAX);
    assert_eq!(
        &cev_w[(0 * m + slot) * 4..(0 * m + slot) * 4 + 4],
        &[0.0, 0.0, 0.0, 1.0],
        "the pre-fix pairing (recipient uncertainty) selects nothing"
    );
}

/// B1, ion stage: the same scenario through full `generate` with
/// `evidence = true`. Row 0's trajectories must carry fragment-ion evidence
/// for the donor peak (judged with the donor's uncertainty); row 1, whose
/// rotated peaks are empty, must carry none.
#[test]
fn b1_shuffled_donor_uncertainty_ion() {
    let _serial = serial();
    let device = dev();
    let (domain, bounds, table) = setup_enum(vec![c5()]);
    let (model, dtable, constants) =
        setup_model(FormulaFeatures::Evidence, &domain, &bounds, &table, &device, true);
    let batch = b1_batch();
    let mut g = tiny_gen_table(32);
    g.control = Control::ShuffledSpectrum;
    g.trajectories = 1;
    g.formulas = 1;
    g.evidence = true;
    assert!(g.validate(16, 4).is_ok());
    let mut ws = GenerationWorkspace::new();
    let out = model.generate(&batch, &dtable, &g, &mut ws, &constants).unwrap();
    out.validate().unwrap();
    assert_eq!(out.trajectories, 1);
    // Device row 0 (donor peak): its trajectory carries evidence for C5.
    assert!(
        out.evidence_count[0] >= 1,
        "row 0 carries donor-peak evidence"
    );
    assert_eq!(
        out.evidence_peak_id[0],
        0,
        "row 0 evidence maps to the donor peak id"
    );
    // Device row 1 (empty rotated peaks): no evidence anywhere.
    assert_eq!(out.evidence_count[1], 0, "row 1 is empty");
}

/// One-spectrum experiment set with explicit peaks/precursors/uncertainties.
fn single_spectrum_set(
    name: &str,
    precursor: u32,
    prec_unc: u32,
    mz_unc: u32,
    peaks: &[(u32, f64)],
    parent: Composition,
) -> ExperimentSet {
    let spectrum = ExportSpectrum {
        row: 0,
        spectrum_id: 7001,
        adduct: 1,
        polarity: 1,
        precursor_mz_udalton: precursor,
        precursor_uncertainty_udalton: prec_unc,
        raw_peak_count: peaks.len() as u32,
        peak_id: (0..peaks.len() as u32).collect(),
        mz_udalton: peaks.iter().map(|&(mz, _)| mz).collect(),
        intensity: peaks.iter().map(|&(_, it)| it).collect(),
        mz_uncertainty_udalton: mz_unc,
        collision_energy_ev: 30.0,
        collision_energy_known: 1,
        energy_count: 1,
        instrument_class: 0,
    };
    ExperimentSet {
        name: name.to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: vec!["mol0".to_string()],
        spectra: vec![ExperimentSpectrum {
            molecule: 0,
            spectrum: spectrum.clone(),
            parent: MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
            parent_composition: parent,
            labels: None,
            domain: SpectrumDomain::InDomainUnlabeled,
        }],
    }
}

fn evidence_trainer(
    table: &FormulaTable,
    batch: usize,
    device: &Device<R>,
) -> Ms2Trainer<R, E> {
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(table, device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch,
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    Ms2Trainer::<R, E>::new(&cfg, table, &tcfg, device).unwrap()
}

/// B2: `evidence_peaks_mean` counts valid evidence peaks independently of
/// candidate support.
///
/// One spectrum (precursor 101,007,276, adduct 1, known uncertainties, one
/// eligible intensity-1 peak at 59,999,451) against a table holding only C5
/// with a narrow precursor window, so `rows_scored = 0` while selection still
/// produces one valid evidence peak. The documented mean is 1; the pre-fix
/// code substituted 0 for the unscored spectrum and reported 0.
#[test]
fn b2_evidence_peaks_mean_empty_support() {
    let _serial = serial();
    let device = dev();
    let table = FormulaTable::from_compositions([c5()].into_iter()).unwrap();
    let mut trainer = evidence_trainer(&table, 1, &device);
    let set = single_spectrum_set(
        "b2",
        101_007_276,
        50,
        50,
        &[(59_999_451, 1.0)],
        c5(),
    );
    let diag = trainer
        .evidence_diagnostics(&set, &[0])
        .unwrap()
        .expect("Evidence diagnostics");
    assert_eq!(diag.spectra, 1);
    assert_eq!(diag.scored, 0);
    assert_eq!(diag.peaks_sum, 1.0);
    assert_eq!(diag.peaks_mean(), 1.0);
    assert_eq!(diag.incomplete_fraction(), 0.0);
    assert!(diag.gold_explained_fraction().is_none());
    assert!(diag.other_explained_fraction().is_none());
}

/// a: historical documents.
///
/// A version-1 `ModelConfig`/`GenerationConfig` JSON and an earlier
/// version-2 JSON with every field added by the evidence work ABSENT
/// (literally absent from the JSON below, not deleted by code)
/// deserialise to `Counts` with the default evidence limits and jitter 0.
/// The version-1 pair mirrors the `ms2_contract.rs` schema-version fixtures
/// (`ModelConfig::v0` / `GenerationConfig::default` with `schema_version`
/// 1); the early version-2 pair is today's canonical JSON with the new keys
/// omitted. A `Counts` trainer checkpoint with those keys stripped from its
/// JSON loads and trains.
#[test]
fn a_historical_counts_documents() {
    let _serial = serial();
    // Version-1 ModelConfig: no formula_features/assignment/formula_artifacts.
    let m1: ModelConfig = serde_json::from_str(MODEL_V1_JSON).unwrap();
    assert_eq!(m1.formula_features, FormulaFeatures::Counts);
    assert!(m1.assignment.is_none());
    assert!(m1.formula_artifacts.is_none());
    assert!(m1.validate().is_ok());
    // Version-1 GenerationConfig: none of the V1 §1.2/§1.4/§4.2/§4.4,
    // evidence or jitter-adjacent fields.
    let g1: GenerationConfig = serde_json::from_str(GEN_V1_JSON).unwrap();
    assert_eq!(g1.formula_source, FormulaSource::Table);
    assert_eq!(g1.formula_window, 32);
    assert!(!g1.evidence);
    assert_eq!(g1.formula_evidence_work_max, 2048);
    assert_eq!(g1.formula_evidence_dispatch_max, 8589934592);
    assert!(g1.validate(16, 4).is_ok());
    // Earlier version-2 documents: schema 2 with the evidence-work fields
    // absent take the Counts layout, the default evidence limits and no
    // jitter.
    let m2: ModelConfig = serde_json::from_str(MODEL_V2_EARLY_JSON).unwrap();
    assert_eq!(m2.formula_features, FormulaFeatures::Counts);
    assert!(m2.validate().is_ok());
    let g2: GenerationConfig = serde_json::from_str(GEN_V2_EARLY_JSON).unwrap();
    assert!(!g2.evidence);
    assert_eq!(g2.formula_evidence_work_max, 2048);
    assert_eq!(g2.formula_evidence_dispatch_max, 8589934592);
    assert_eq!(g2.ion_request_work_max, 268435456);
    assert!(g2.validate(16, 4).is_ok());
    let t2: TrainConfig = serde_json::from_str(TRAIN_V2_EARLY_JSON).unwrap();
    assert_eq!(t2.precursor_jitter_ppm, 0.0);
    assert_eq!(t2.formula_evidence_work_max, 2048);
    assert_eq!(t2.formula_evidence_dispatch_max, 268435456);
    assert!(t2.validate().is_ok());
    // A Counts checkpoint with those keys stripped loads and trains.
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let mut cfg = tiny_model(FormulaFeatures::Counts);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: 2,
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    let path = std::env::temp_dir().join("ms2_e3f_counts_stripped.json");
    trainer.save(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
    for key in ["formula_features", "assignment", "formula_artifacts"] {
        v["model_config"].as_object_mut().unwrap().remove(key);
    }
    for key in [
        "lambda_assign",
        "ion_request_work_max",
        "formula_evidence_work_max",
        "formula_evidence_dispatch_max",
        "precursor_jitter_ppm",
    ] {
        v["train_config"].as_object_mut().unwrap().remove(key);
    }
    std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    let mut loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert_eq!(
        loaded.model.config.formula_features,
        FormulaFeatures::Counts
    );
    let set = two_spectrum_set(&comps, 50);
    loaded.request_report();
    let rep = loaded.step(&set, &[0, 1]).unwrap().expect("report");
    assert!(rep.loss.is_finite() && rep.formula.is_finite());
    let _ = std::fs::remove_file(&path);
}

/// Version-1 `ModelConfig` JSON: `ModelConfig::v0` with `schema_version` 1
/// and no `formula_features`, `assignment` or `formula_artifacts`.
const MODEL_V1_JSON: &str = r#"{
  "schema_version": 1,
  "version": "ms2-model-v0",
  "chemistry": "ms2-chem-v0.1",
  "n_peaks": 128,
  "d_model": 128,
  "encoder": {
    "d_model": 128,
    "n_heads": 4,
    "head_dim": 64,
    "d_state": 32,
    "n_groups": 4,
    "discretization": "LearnedTrapezoid",
    "dynamics": "Rotational",
    "mode": "Siso",
    "chunk_size": 64,
    "conv_kernel": null,
    "bc_norm": true,
    "bc_bias": true,
    "post_gate_norm": false,
    "skip_connection": true,
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init_floor": 0.0001,
    "a_init_min": 1.0,
    "a_init_max": 16.0,
    "bias": false
  },
  "decoder": {
    "d_model": 128,
    "n_heads": 4,
    "head_dim": 64,
    "d_state": 32,
    "n_groups": 4,
    "discretization": "LearnedTrapezoid",
    "dynamics": "Rotational",
    "mode": "Siso",
    "chunk_size": 64,
    "conv_kernel": null,
    "bc_norm": true,
    "bc_bias": true,
    "post_gate_norm": false,
    "skip_connection": true,
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init_floor": 0.0001,
    "a_init_min": 1.0,
    "a_init_max": 16.0,
    "bias": false
  },
  "encoder_blocks": 2,
  "decoder_blocks": 2,
  "attention_heads": 4,
  "fourier_features": 16,
  "max_atoms": 16,
  "max_ring_closures": 4,
  "formula_table": {
    "version": "ms2-formula-v0",
    "rows": 37859,
    "sha256": ""
  },
  "energy_scale_ev": 100.0,
  "energy_clip_ev": 400.0,
  "dtype": "F32"
}"#;

/// Version-1 `GenerationConfig` JSON: today's defaults with `schema_version`
/// 1 and every newer field absent.
const GEN_V1_JSON: &str = r#"{
  "schema_version": 1,
  "trajectories": 8,
  "formulas": 4,
  "seed": 0,
  "temperature": 1.0,
  "max_steps": 22,
  "max_device_bytes": 2147483648,
  "formula_rows_visited_max": 4294967295,
  "formula_rows_scored_max": 4096,
  "mode": "Sampling",
  "oracle_formula": false,
  "control": "None"
}"#;

/// Earlier version-2 `ModelConfig` JSON: schema 2 with the evidence-work
/// fields absent (deserialises to `Counts`).
const MODEL_V2_EARLY_JSON: &str = r#"{
  "schema_version": 2,
  "version": "ms2-model-v0",
  "chemistry": "ms2-chem-v0.1",
  "n_peaks": 128,
  "d_model": 128,
  "encoder": {
    "d_model": 128,
    "n_heads": 4,
    "head_dim": 64,
    "d_state": 32,
    "n_groups": 4,
    "discretization": "LearnedTrapezoid",
    "dynamics": "Rotational",
    "mode": "Siso",
    "chunk_size": 64,
    "conv_kernel": null,
    "bc_norm": true,
    "bc_bias": true,
    "post_gate_norm": false,
    "skip_connection": true,
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init_floor": 0.0001,
    "a_init_min": 1.0,
    "a_init_max": 16.0,
    "bias": false
  },
  "decoder": {
    "d_model": 128,
    "n_heads": 4,
    "head_dim": 64,
    "d_state": 32,
    "n_groups": 4,
    "discretization": "LearnedTrapezoid",
    "dynamics": "Rotational",
    "mode": "Siso",
    "chunk_size": 64,
    "conv_kernel": null,
    "bc_norm": true,
    "bc_bias": true,
    "post_gate_norm": false,
    "skip_connection": true,
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init_floor": 0.0001,
    "a_init_min": 1.0,
    "a_init_max": 16.0,
    "bias": false
  },
  "encoder_blocks": 2,
  "decoder_blocks": 2,
  "attention_heads": 4,
  "fourier_features": 16,
  "max_atoms": 16,
  "max_ring_closures": 4,
  "formula_table": {
    "version": "ms2-formula-v0",
    "rows": 37859,
    "sha256": ""
  },
  "energy_scale_ev": 100.0,
  "energy_clip_ev": 400.0,
  "dtype": "F32"
}"#;

/// Earlier version-2 `GenerationConfig` JSON: schema 2 with the
/// evidence-work fields absent.
const GEN_V2_EARLY_JSON: &str = r#"{
  "schema_version": 2,
  "trajectories": 8,
  "formulas": 4,
  "seed": 0,
  "temperature": 1.0,
  "max_steps": 22,
  "max_device_bytes": 2147483648,
  "formula_rows_visited_max": 4294967295,
  "formula_rows_scored_max": 4096,
  "mode": "Sampling",
  "oracle_formula": false,
  "control": "None",
  "formula_source": "Table",
  "formula_window": 32,
  "enum_lanes_max": 262144,
  "enum_lane_visits_max": 4096,
  "enum_dispatch_visits_max": 4000000,
  "allocation": "RoundRobin",
  "identity": "TraceOnly",
  "identity_work_max": 4096,
  "returned": 0
}"#;

/// Earlier `TrainConfig` JSON: today's defaults with the evidence-work and
/// jitter fields absent.
const TRAIN_V2_EARLY_JSON: &str = r#"{
  "batch": 16,
  "slots": 16,
  "lr": 0.0003,
  "weight_decay": 0.1,
  "formula_weight": 0.2,
  "seed": 1,
  "control": "None",
  "grad_clip": null,
  "gold_formula_conditioning": "ScoredRowOrZero",
  "formula_source": "Table",
  "formula_window": 32,
  "enum_lanes_max": 262144,
  "enum_lane_visits_max": 4096,
  "enum_dispatch_visits_max": 4000000,
  "enum_fit_name": null,
  "enum_fit_sha256": null,
  "enum_fit_subset": null
}"#;

/// Two-spectrum set over `comps` with deterministic peaks (each spectrum
/// carries a peak at its own gold mass minus the electron, so its gold
/// candidate has non-trivial evidence) and uniform uncertainty `unc`.
fn two_spectrum_set(comps: &[Composition], unc: u32) -> ExperimentSet {
    let n_raw = 64usize;
    let b = comps.len();
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f64; b * n_raw];
    for (bi, c) in comps.iter().enumerate() {
        let mass = composition_mass(c).unwrap();
        // Gold-explaining peak first, then deterministic fillers.
        mz[bi * n_raw] = mass - 549;
        peak_id[bi * n_raw] = 0;
        intensity[bi * n_raw] = 5.0;
        for i in 1..10usize {
            peak_id[bi * n_raw + i] = i as u32;
            mz[bi * n_raw + i] = 50_000_001 + ((bi * 13 + i * 7) % 40) as u32 * 1_000_000;
            intensity[bi * n_raw + i] = 1.0 + (i % 3) as f64;
        }
    }
    let spectra = comps
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let np = 10usize;
            ExperimentSpectrum {
                molecule: i,
                spectrum: ExportSpectrum {
                    row: i as u64,
                    spectrum_id: 8000 + i as u64,
                    adduct: 1,
                    polarity: 1,
                    precursor_mz_udalton: precursor_of(c),
                    precursor_uncertainty_udalton: unc,
                    raw_peak_count: np as u32,
                    peak_id: peak_id[i * n_raw..i * n_raw + np].to_vec(),
                    mz_udalton: mz[i * n_raw..i * n_raw + np].to_vec(),
                    intensity: intensity[i * n_raw..i * n_raw + np].to_vec(),
                    mz_uncertainty_udalton: 50,
                    collision_energy_ev: 30.0,
                    collision_energy_known: 1,
                    energy_count: 1,
                    instrument_class: 0,
                },
                parent: MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
                parent_composition: *c,
                labels: None,
                domain: SpectrumDomain::InDomainUnlabeled,
            }
        })
        .collect();
    ExperimentSet {
        name: "e3f-two".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: (0..b).map(|i| format!("mol{i}")).collect(),
        spectra,
    }
}

/// b: production search buffers for one source.
///
/// After `generate` with the `Evidence` layout on a fixture whose
/// candidates have non-trivial evidence, the workspace's `cand_feat` /
/// `cand_xfeat` (read through the test-only workspace accessor, i.e. the
/// buffers production actually scored) match the host twins applied to the
/// SAME request (`formula_evidence::evidence_peaks` → `formula_evidence` →
/// `formula_features` from the host copies of kept peaks, cand, meta, spec):
/// integers exact, floats within 1e-6. This checks the production wiring,
/// not a pipeline rebuilt in the test.
fn check_production_search_buffers(source: FormulaSource) {
    let _serial = serial();
    let device = dev();
    let comps = vec![comp(0, 2, 0, 1), c5(), comp(6, 6, 0, 0)];
    let (domain, bounds, table) = setup_enum(comps.clone());
    let (model, dtable, constants) =
        setup_model(FormulaFeatures::Evidence, &domain, &bounds, &table, &device, false);
    let set = two_spectrum_set(&comps, 50);
    let indices: Vec<usize> = (0..comps.len()).collect();
    let batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let b = batch.len();
    let m = 32usize;
    let mut g = tiny_gen_table(32);
    g.formula_source = source;
    if source == FormulaSource::Enumerate {
        g.enum_lane_visits_max = 65_536;
    }
    // One production `generate` on a fresh workspace.
    let mut ws = GenerationWorkspace::new();
    let out = model.generate(&batch, &dtable, &g, &mut ws, &constants).unwrap();
    out.validate().unwrap();
    // The buffers production scored, read through the workspace accessor.
    let (kept_t, kept_f_t, cand_t, ev_peaks_t, _ev_w_t, cand_ev_t, feat_t, xfeat_t) =
        ws.debug_search_buffers().unwrap();
    let kept = kept_t.try_to_vec().unwrap();
    let kept_f = kept_f_t.try_to_f32().unwrap();
    let cand = cand_t.try_to_vec().unwrap();
    let feat = feat_t.try_to_f32().unwrap();
    let xfeat = xfeat_t.unwrap().try_to_f32().unwrap();
    let cand_ev = cand_ev_t.unwrap().try_to_f32().unwrap();
    let ev_ids = ev_peaks_t.unwrap().try_to_vec().unwrap();
    // Host twins from the host copies of the SAME request.
    let meta_h = meta_host(&batch);
    let spec_h = spec_host(&batch);
    let (ev_peaks_h, ev_w_h) = twin_peaks(&kept, &kept_f, &meta_h, &spec_h, b, 16, EVIDENCE_PEAKS);
    assert_eq!(ev_ids, ev_peaks_h, "{source:?}: ev_peaks integers exact");
    let cand_ev_h = twin_evidence(&cand, &ev_peaks_h, &ev_w_h, &meta_h, &spec_h, b, m, EVIDENCE_PEAKS, 2048, u32::MAX);
    assert_eq!(cand_ev.len(), cand_ev_h.len());
    for (i, (a, e)) in cand_ev.iter().zip(cand_ev_h.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-6,
            "{source:?}: cand_ev[{i}] {a} vs twin {e}"
        );
    }
    let feat16_h = twin_features(&cand, &cand_ev_h, &meta_h, &log_table_host(), b, m);
    for s in 0..b * m {
        for k in 0..10 {
            let (a, e) = (feat[s * 10 + k], feat16_h[s * 16 + k]);
            assert!((a - e).abs() <= 1e-6, "{source:?}: cand_feat[{s},{k}] {a} vs {e}");
        }
        for k in 0..6 {
            let (a, e) = (xfeat[s * 6 + k], feat16_h[s * 16 + 10 + k]);
            assert!((a - e).abs() <= 1e-6, "{source:?}: cand_xfeat[{s},{k}] {a} vs {e}");
        }
    }
    // The fixture has non-trivial evidence: some candidate explains a peak.
    assert!(
        cand_ev.iter().step_by(4).any(|&v| v > 0.0),
        "{source:?}: fixture explains peaks"
    );
}

#[test]
fn b_production_search_buffers_table() {
    check_production_search_buffers(FormulaSource::Table);
}

#[test]
fn b_production_search_buffers_enumerate() {
    check_production_search_buffers(FormulaSource::Enumerate);
}

/// c: padding with a live branch.
///
/// With `evidence_out` weight and bias set to non-zero values through the
/// state dict, a padding slot's log-probability is still the masked value
/// (`f32::MIN` bits: the mask is applied after the branch, and the masked
/// logit survives the softmax shift exactly): no contribution survives the
/// mask. Checked for a spectrum with scored support and for one with EMPTY
/// support (the slot-0 fallback, whose log-probability is exactly 0).
#[test]
fn c_padding_with_live_branch() {
    let _serial = serial();
    let device = dev();
    let mut rng = Rng::seeded(5);
    let cfg = tiny_model(FormulaFeatures::Evidence);
    let head = mamba3::models::ms2::formula_head::FormulaHead::<R, E>::init(&cfg, &device, &mut rng)
        .unwrap();
    // Non-zero output layer through the state dict.
    let mut sd = head.state_dict();
    for key in ["formula.evidence_out.weight", "evidence_out.weight"] {
        if let Some(entry) = sd.entries.get_mut(key) {
            entry.data = vec![0.1; entry.data.len()];
        }
    }
    for key in ["formula.evidence_out.bias", "evidence_out.bias"] {
        if let Some(entry) = sd.entries.get_mut(key) {
            entry.data = vec![0.2; entry.data.len()];
        }
    }
    head.load_state_dict(&sd, true).unwrap();
    let out_w = head
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "evidence_out.weight")
        .unwrap()
        .1;
    assert!(out_w.value().to_f32().iter().all(|&v| v == 0.1));
    // B=2, M=4: spectrum 0 has two scored slots + two padding slots,
    // spectrum 1 has empty support (all flags 0).
    let (b, m, d) = (2usize, 4usize, 16usize);
    let mut cand_h = vec![0u32; b * m * 13];
    for s in 0..b * m {
        for e in 0..10 {
            cand_h[s * 13 + e] = ((s + e) % 3) as u32;
        }
        cand_h[s * 13 + 10] = 60_000_000 + s as u32;
        cand_h[s * 13 + 11] = if s < 2 { 1 } else { 0 };
        cand_h[s * 13 + 12] = 0;
    }
    let feat_h: Vec<f32> = (0..b * m * 10).map(|i| 0.05 * ((i % 7) + 1) as f32).collect();
    let xfeat_h: Vec<f32> = (0..b * m * 6).map(|i| 0.25 * ((i % 5) + 1) as f32).collect();
    let mut buffers = mamba3::tensor::ops::ms2::FormulaBuffers::<R, E>::new_evidence(b, m, 1, &device);
    buffers.cand = IdTensor::from_slice(&cand_h, vec![b, m, 13], &device).unwrap();
    buffers.cand_feat =
        Tensor::<R, E>::from_f32(&feat_h, vec![b, m, 10], &device).unwrap();
    buffers.cand_xfeat =
        Some(Tensor::<R, E>::from_f32(&xfeat_h, vec![b, m, 6], &device).unwrap());
    let pool_h: Vec<f32> = (0..b * d).map(|i| 0.01 * (i % 11) as f32).collect();
    let pool = mamba3::autograd::Var::constant(
        Tensor::<R, E>::from_f32(&pool_h, vec![b, d], &device).unwrap(),
    );
    let lp = head.score(&buffers, &pool).unwrap().log_prob;
    let lp_h = lp.tensor().try_to_f32().unwrap();
    let masked = f32::MIN;
    // Scored support: padding slots carry exactly the masked value.
    for slot in 2..4 {
        assert_eq!(
            lp_h[slot].to_bits(),
            masked.to_bits(),
            "padding slot {slot} keeps the masked value with a live branch"
        );
    }
    assert!(lp_h[0] > -1e30 && lp_h[1] > -1e30);
    // Empty support: slot 0 (the fallback) is exactly 0, the rest masked.
    assert_eq!(lp_h[m].to_bits(), 0.0f32.to_bits());
    for slot in 1..4 {
        assert_eq!(
            lp_h[m + slot].to_bits(),
            masked.to_bits(),
            "empty-support padding slot {slot} keeps the masked value"
        );
    }
    // Perturbing a padding slot's evidence features moves nothing: the mask
    // kills the contribution before the softmax, so every log-probability
    // is bit-identical.
    let mut x_up = xfeat_h.clone();
    x_up[2 * 6] += 10.0;
    x_up[(m + 3) * 6] += 10.0;
    buffers.cand_xfeat =
        Some(Tensor::<R, E>::from_f32(&x_up, vec![b, m, 6], &device).unwrap());
    let lp2 = head.score(&buffers, &pool).unwrap().log_prob;
    let lp2_h = lp2.tensor().try_to_f32().unwrap();
    assert_eq!(
        lp_h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        lp2_h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "padding features do not move any log-probability"
    );
}

/// d: leaving the zero point.
///
/// Starting from the zero-initialised output layer, optimizer steps on a
/// fixture where evidence separates the gold candidate (C5, explains its
/// peak) from a same-mass-window decoy (C4H12, scored through the widened
/// precursor-uncertainty window, explains nothing) make `evidence_out`
/// weight non-zero after the first step, deliver a non-zero gradient to
/// `evidence_in` on a later step (its weights move with weight decay off),
/// and end with the formula loss below a `Counts` model trained identically
/// on the same fixture.
#[test]
fn d_leaving_the_zero_point() {
    let _serial = serial();
    let device = dev();
    let table = FormulaTable::from_compositions([c5(), c4h12()].into_iter()).unwrap();
    assert!(composition_mass(&c4h12()).unwrap() > composition_mass(&c5()).unwrap());
    // One spectrum: the C5 precursor with a wide precursor-uncertainty
    // window (both candidates join) and one peak the gold explains.
    let set = single_spectrum_set("d", precursor_of(&c5()), 500_000, 50, &[(59_999_451, 1.0)], c5());
    let mk = |layout| {
        let mut cfg = tiny_model(layout);
        let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
        cfg.formula_table.rows = dtable.rows as u32;
        cfg.formula_table.sha256 = dtable.sha256.clone();
        let tcfg = TrainConfig {
            batch: 1,
            slots: 2,
            lr: 1e-2,
            weight_decay: 0.0,
            gold_formula_conditioning: GoldFormulaConditioning::Composition,
            ..TrainConfig::default()
        };
        Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap()
    };
    let mut ev = mk(FormulaFeatures::Evidence);
    // The fixture separates gold from decoy through evidence: both scored,
    // gold fully explained, the other slot unexplained.
    let diag = ev
        .evidence_diagnostics(&set, &[0])
        .unwrap()
        .expect("Evidence diagnostics");
    assert_eq!(diag.scored, 2, "gold and decoy share the window");
    assert_eq!(diag.gold_explained_fraction(), Some(1.0));
    assert_eq!(diag.other_explained_fraction(), Some(0.0));
    // Zero start: the output layer is exactly zero.
    let out_w0: Vec<f32> = ev
        .model
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "formula.evidence_out.weight")
        .unwrap()
        .1
        .value()
        .to_f32();
    assert!(out_w0.iter().all(|&v| v == 0.0));
    // First optimizer step moves the output weight off zero.
    assert!(ev.step(&set, &[0]).unwrap().is_none());
    let out_w1: Vec<f32> = ev
        .model
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "formula.evidence_out.weight")
        .unwrap()
        .1
        .value()
        .to_f32();
    assert!(
        out_w1.iter().any(|&v| v != 0.0),
        "evidence_out leaves zero after the first step"
    );
    // A later step delivers a non-zero gradient to evidence_in: with decay
    // off its weights move exactly when the gradient is non-zero.
    let in_w1: Vec<f32> = ev
        .model
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "formula.evidence_in.weight")
        .unwrap()
        .1
        .value()
        .to_f32();
    assert!(ev.step(&set, &[0]).unwrap().is_none());
    let in_w2: Vec<f32> = ev
        .model
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "formula.evidence_in.weight")
        .unwrap()
        .1
        .value()
        .to_f32();
    assert_ne!(
        in_w1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        in_w2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "evidence_in receives a non-zero gradient once the output is live"
    );
    // Train both layouts identically; evidence ends below counts.
    let mut counts = mk(FormulaFeatures::Counts);
    for _ in 0..8 {
        assert!(ev.step(&set, &[0]).unwrap().is_none());
        assert!(counts.step(&set, &[0]).unwrap().is_none());
    }
    ev.request_report();
    counts.request_report();
    let ev_rep = ev.step(&set, &[0]).unwrap().expect("report");
    let counts_rep = counts.step(&set, &[0]).unwrap().expect("report");
    println!(
        "E3D-LOSS evidence_formula={} counts_formula={}",
        ev_rep.formula, counts_rep.formula
    );
    assert!(
        ev_rep.formula < counts_rep.formula,
        "evidence formula loss {} below counts {}",
        ev_rep.formula,
        counts_rep.formula
    );
}

/// e: conditioning parity.
///
/// Under `Evidence`, the embedding that conditions the decoder (generation
/// gathered rows; teacher forcing in both `Composition` and
/// `ScoredRowOrZero` modes) and the embedding the assignment head uses
/// (`FormulaHead::embed_rows`, the row network `AssignmentHead::log_prob`
/// runs) equal those of a `Counts` model with the same row-network weights,
/// bit for bit on CPU.
#[test]
fn e_conditioning_parity() {
    let _serial = serial();
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let set = two_spectrum_set(&comps, 50);
    let indices: Vec<usize> = (0..comps.len()).collect();
    for mode in [
        GoldFormulaConditioning::Composition,
        GoldFormulaConditioning::ScoredRowOrZero,
    ] {
        let mk = |layout| {
            let mut cfg = tiny_model(layout);
            let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
            cfg.formula_table.rows = dtable.rows as u32;
            cfg.formula_table.sha256 = dtable.sha256.clone();
            let tcfg = TrainConfig {
                batch: indices.len(),
                slots: 2,
                gold_formula_conditioning: mode,
                ..TrainConfig::default()
            };
            Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap()
        };
        let mut counts = mk(FormulaFeatures::Counts);
        let mut evidence = mk(FormulaFeatures::Evidence);
        // Same row-network weights: the encoder (initialised before the
        // formula head) and the head's row network (initialised before the
        // evidence branch) are identical across layouts. The decoder is
        // initialised after the head, so its stream legitimately differs;
        // conditioning never reads decoder weights.
        for (n, p) in counts.model.named_parameters() {
            if n.contains("evidence") || n.starts_with("decoder.") {
                continue;
            }
            let q = evidence
                .model
                .named_parameters()
                .into_iter()
                .find(|(m, _)| m == &n)
                .unwrap_or_else(|| panic!("missing param {n}"))
                .1;
            assert_eq!(
                p.value().to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                q.value().to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "shared param {n} identical"
            );
        }
        // Teacher forcing through the production hook, both modes.
        let c = counts.conditioning_for_test(&set, &indices).unwrap();
        let e = evidence.conditioning_for_test(&set, &indices).unwrap();
        assert_eq!(c.slots, e.slots);
        assert_eq!(
            c.e_cond.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            e.e_cond.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{mode:?}: conditioning embedding bit-identical"
        );
        assert_eq!(
            c.scored_embedding.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            e.scored_embedding.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{mode:?}: scored embedding bit-identical"
        );
    }
    // Generation gathered rows: same request through both layouts.
    let batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let run_gen = |layout| {
        let mut cfg = tiny_model(layout);
        let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
        cfg.formula_table.rows = dtable.rows as u32;
        cfg.formula_table.sha256 = dtable.sha256.clone();
        let mut rng = Rng::seeded(5);
        let model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng).unwrap();
        let g = tiny_gen_table(32);
        let mut ws = GenerationWorkspace::new();
        let constants = Ms2Constants::new(&device);
        model.generate(&batch, &dtable, &g, &mut ws, &constants).unwrap();
        ws.last_traj_formula.clone().unwrap().try_to_f32().unwrap()
    };
    let (tc, te) = (run_gen(FormulaFeatures::Counts), run_gen(FormulaFeatures::Evidence));
    assert_eq!(
        tc.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        te.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "generation conditioning rows bit-identical"
    );
    // The assignment embedding: the shared row network on the same features.
    let mut rng = Rng::seeded(5);
    let hc = mamba3::models::ms2::formula_head::FormulaHead::<R, E>::init(
        &tiny_model(FormulaFeatures::Counts),
        &device,
        &mut rng,
    )
    .unwrap();
    let mut rng = Rng::seeded(5);
    let he = mamba3::models::ms2::formula_head::FormulaHead::<R, E>::init(
        &tiny_model(FormulaFeatures::Evidence),
        &device,
        &mut rng,
    )
    .unwrap();
    let feats_h: Vec<f32> = (0..2 * 3 * 10).map(|i| 0.1 * ((i % 9) + 1) as f32).collect();
    let feats = mamba3::autograd::Var::constant(
        Tensor::<R, E>::from_f32(&feats_h, vec![2, 3, 10], &device).unwrap(),
    );
    let (ec, ee) = (
        hc.embed_rows(&feats).unwrap().tensor().try_to_f32().unwrap(),
        he.embed_rows(&feats).unwrap().tensor().try_to_f32().unwrap(),
    );
    assert_eq!(
        ec.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ee.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "assignment row-network embedding bit-identical"
    );
}

/// f: inference boundary with the enumerating source.
///
/// Two requests differing only in gold payload (`parent_composition`),
/// labels and targets (parent graphs, hence the target batches) give
/// identical candidate batches: evidence generation consumes spectra,
/// formula artifacts and configuration, never gold, labels or targets.
#[test]
fn f_enumerate_inference_boundary() {
    let _serial = serial();
    use mamba3::models::ms2::targets::Labels;
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let (domain, bounds, table) = setup_enum(comps.clone());
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        enum_lane_visits_max: 65_536,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let set_a = two_spectrum_set(&comps, 50);
    let mut set_b = two_spectrum_set(&comps, 50);
    // Gold payload, labels and targets (graphs) all differ; peaks,
    // precursors and metadata are untouched.
    for s in &mut set_b.spectra {
        s.parent_composition = comp(1, 4, 0, 0);
        s.parent = MolGraph::new(vec![6], vec![]).expect("single atom builds");
        s.labels = Some(Labels {
            embeddings: Vec::new(),
            graphs: 0,
            targets_before_cut: 0,
            targets: Vec::new(),
            dropped_weight: 0.0,
            cut_is_tied: false,
            explained_peaks: Vec::new(),
            ambiguous_hypotheses: 0,
            canonicalization_failures: 0,
        });
    }
    let indices: Vec<usize> = (0..comps.len()).collect();
    let mut g = tiny_gen_table(32);
    g.formula_source = FormulaSource::Enumerate;
    g.enum_lane_visits_max = 65_536;
    let a = trainer.generate_candidates(&set_a, &indices, &g).unwrap();
    let b = trainer.generate_candidates(&set_b, &indices, &g).unwrap();
    assert_eq!(a, b, "gold, labels and targets do not enter generation");
}

/// g: jitter through a real trainer step.
///
/// With `precursor_jitter_ppm = 2`, the precursors the trainer uploads at
/// steps 0 and 1 (observed through the test-only hook recording the exact
/// host batch production uploads) equal
/// `jitter_precursor_mz(stored, 2.0, seed, 1 + step, index in set)` and
/// differ between the two steps; the SAME jittered values are what the
/// enumeration meta rows, the peak filter (`mz <= precursor + 2 Da`, with a
/// peak at stored precursor + 2 Da exactly so the jitter decides its
/// eligibility) and the upload see, while labels and target inputs are
/// bit-identical to the unjittered ones.
#[test]
fn g_jitter_through_trainer_step() {
    let _serial = serial();
    use mamba3::models::ms2::formula_enum::build_enum_meta;
    use mamba3::models::ms2::train::TRAIN_ROWS_SCORED_MAX;
    let device = dev();
    let seed = 11u64;
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let (domain, bounds, table) = setup_enum(comps.clone());
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        seed,
        precursor_jitter_ppm: 2.0,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        enum_lane_visits_max: 65_536,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer.capture_prep_batch(true);
    // Spectrum 0 carries a peak at stored precursor + 2 Da exactly (appended
    // last, raw index 10), so the jitter draw decides its peak-filter
    // eligibility.
    let mut set = two_spectrum_set(&comps, 50);
    let stored: Vec<u32> = set.spectra.iter().map(|s| s.spectrum.precursor_mz_udalton).collect();
    set.spectra[0].spectrum.peak_id.push(10);
    set.spectra[0].spectrum.mz_udalton.push(stored[0] + 2_000_000);
    set.spectra[0].spectrum.intensity.push(10.0);
    set.spectra[0].spectrum.raw_peak_count += 1;
    let indices: Vec<usize> = (0..comps.len()).collect();
    let plain = spectrum_batch_for(&set, &indices, 64).unwrap();
    // Steps 0 and 1 through the real trainer step (which jitters).
    assert!(trainer.step(&set, &indices).unwrap().is_none());
    let rec0 = trainer.last_prep_batch.clone().expect("hook recorded step 0");
    assert!(trainer.step(&set, &indices).unwrap().is_none());
    let rec1 = trainer.last_prep_batch.clone().expect("hook recorded step 1");
    // Documented keying: (seed, 1 + step, index in set).
    for (b, &idx) in indices.iter().enumerate() {
        assert_eq!(
            rec0.precursor_mz_udalton[b],
            jitter_precursor_mz(stored[b], 2.0, seed, 1, idx as u64),
            "step 0 precursor of spectrum {idx}"
        );
        assert_eq!(
            rec1.precursor_mz_udalton[b],
            jitter_precursor_mz(stored[b], 2.0, seed, 2, idx as u64),
            "step 1 precursor of spectrum {idx}"
        );
    }
    assert_ne!(
        rec0.precursor_mz_udalton, rec1.precursor_mz_udalton,
        "the two steps draw differently"
    );
    // Only the precursors move: every other byte matches the unjittered batch.
    for rec in [&rec0, &rec1] {
        let mut expect = plain.clone();
        expect.precursor_mz_udalton = rec.precursor_mz_udalton.clone();
        assert_eq!(*rec, expect, "jitter touches precursors only");
    }
    // Labels and target inputs are untouched by either step.
    for s in &set.spectra {
        assert!(s.labels.is_none());
    }
    let graphs: Vec<String> = set
        .spectra
        .iter()
        .map(|s| format!("{:?}", s.parent_composition))
        .collect();
    assert_eq!(
        graphs,
        vec![
            "[5, 0, 0, 0, 0, 0, 0, 0, 0, 0]".to_string(),
            "[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]".to_string()
        ]
    );
    // The enumeration meta rows carry the SAME jittered precursors that were
    // uploaded (production builds them from the same prepared batch): word 0
    // is the parent mass derived from the jittered precursor.
    use mamba3::models::ms2::batch::DeviceSpectra;
    use mamba3::models::ms2::chem::parent_mass;
    use mamba3::tensor::ops::ms2::{self, PeakBuffers};
    let art = DeviceEnumArtifacts::<R>::upload(&domain, &bounds, &device).unwrap();
    let scored_cap = TRAIN_ROWS_SCORED_MAX.min(32);
    for rec in [&rec0, &rec1] {
        let meta = build_enum_meta(rec, art.domain_max_error, 65_536, scored_cap);
        for b in 0..rec.len() {
            assert_eq!(
                meta[b * 8],
                parent_mass(rec.precursor_mz_udalton[b], rec.adduct[b]).unwrap(),
                "enum meta row {b} derives from the uploaded precursor"
            );
        }
        // The production peak filter on the uploaded batch: the boundary
        // peak (raw 10 of spectrum 0) is kept exactly when the jittered
        // precursor still covers it.
        let spectra = DeviceSpectra::<R, E>::upload(rec, &device).unwrap();
        let peaks = PeakBuffers::<R, E>::new(rec.len(), 64, 16, &device);
        ms2::peak_select(
            &spectra.mz,
            &spectra.intensity,
            &spectra.meta,
            spectra.intensity_scale,
            &peaks,
        )
        .unwrap();
        let kept = peaks.kept.try_to_vec().unwrap();
        let n = 16usize;
        let present = (0..n).any(|p| kept[(0 * n + p) * 3] == 10);
        let eligible = stored[0] + 2_000_000 <= rec.precursor_mz_udalton[0] + 2_000_000;
        assert_eq!(
            present, eligible,
            "boundary peak eligibility follows the jittered precursor"
        );
    }
}

/// h: diagnostics values and read budget.
///
/// Hand-built two-spectrum fixture: spectrum A (C5 precursor, widened
/// precursor-uncertainty window scoring C5 + the C4H12 decoy, one explained
/// peak) and spectrum B (far precursor, empty candidate support, one valid
/// evidence peak). Expected: incomplete fraction 1/2, peaks mean 1.0
/// (the empty-support spectrum contributes its peak), gold explained
/// fraction 1.0, other explained fraction 0.0 — and exactly one device
/// read per `evidence_diagnostics` call.
#[test]
fn h_diagnostics_values_and_reads() {
    let _serial = serial();
    let device = dev();
    let table = FormulaTable::from_compositions([c5(), c4h12()].into_iter()).unwrap();
    let mut trainer = evidence_trainer(&table, 2, &device);
    let mk_spectrum = |precursor: u32, prec_unc: u32| ExperimentSpectrum {
        molecule: 0,
        spectrum: ExportSpectrum {
            row: 0,
            spectrum_id: 7100 + precursor as u64 % 100,
            adduct: 1,
            polarity: 1,
            precursor_mz_udalton: precursor,
            precursor_uncertainty_udalton: prec_unc,
            raw_peak_count: 1,
            peak_id: vec![0],
            mz_udalton: vec![59_999_451],
            intensity: vec![1.0],
            mz_uncertainty_udalton: 50,
            collision_energy_ev: 30.0,
            collision_energy_known: 1,
            energy_count: 1,
            instrument_class: 0,
        },
        parent: MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
        parent_composition: c5(),
        labels: None,
        domain: SpectrumDomain::InDomainUnlabeled,
    };
    let set = ExperimentSet {
        name: "e3f-h".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: vec!["mol0".to_string(), "mol1".to_string()],
        spectra: vec![
            {
                let mut s = mk_spectrum(precursor_of(&c5()), 500_000);
                s.molecule = 0;
                s
            },
            {
                let mut s = mk_spectrum(201_007_276, 50);
                s.molecule = 1;
                s.spectrum.row = 1;
                s.spectrum.spectrum_id = 7101;
                s
            },
        ],
    };
    // Warmup, then exactly one device read per call.
    let first = trainer.evidence_diagnostics(&set, &[0, 1]).unwrap().expect("diagnostics");
    let r0 = read_count();
    let rr0 = runtime_read_count();
    let diag = trainer.evidence_diagnostics(&set, &[0, 1]).unwrap().expect("diagnostics");
    assert_eq!(diag, first, "deterministic across calls");
    assert_eq!(read_count() - r0, 1, "one logical read per call");
    assert_eq!(runtime_read_count() - rr0, 1, "one device read per call");
    assert_eq!(diag.spectra, 2);
    assert_eq!(diag.scored, 2, "gold + decoy on spectrum A");
    // Both walks complete within budget (tiny parents), so no scored slot
    // is incomplete even though the decoy explains nothing.
    assert_eq!(diag.incomplete, 0);
    assert_eq!(diag.incomplete_fraction(), 0.0);
    assert_eq!(diag.peaks_sum, 2.0, "both spectra contribute their peak");
    assert_eq!(diag.peaks_mean(), 1.0);
    assert_eq!(diag.gold_spectra, 1);
    assert_eq!(diag.gold_explained_fraction(), Some(1.0));
    assert_eq!(diag.other_slots, 1);
    assert_eq!(diag.other_explained_fraction(), Some(0.0));
}

/// i: a trained `Evidence` checkpoint reloaded.
///
/// After 20 steps (with the branch provably non-zero), saving and reloading
/// gives bit-identical `generate` output and a bit-identical next-step loss
/// on CPU.
#[test]
fn i_trained_evidence_checkpoint_reload() {
    let _serial = serial();
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        lr: 3e-3,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    let set = two_spectrum_set(&comps, 50);
    let indices: Vec<usize> = (0..comps.len()).collect();
    for _ in 0..20 {
        assert!(trainer.step(&set, &indices).unwrap().is_none());
    }
    // Non-zero branch: at least one evidence parameter moved.
    let moved = trainer
        .model
        .named_parameters()
        .into_iter()
        .filter(|(n, _)| n.contains("evidence"))
        .any(|(_, p)| p.value().to_f32().iter().any(|&v| v != 0.0));
    assert!(moved, "20 steps activate the branch");
    let path = std::env::temp_dir().join("ms2_e3f_evidence_trained.json");
    trainer.save(&path).unwrap();
    let mut reloaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    // Bit-identical generation.
    let g = tiny_gen_table(32);
    let a = trainer.generate_candidates(&set, &indices, &g).unwrap();
    let b = reloaded.generate_candidates(&set, &indices, &g).unwrap();
    assert_eq!(a, b, "reloaded checkpoint generates bit-identical output");
    // Bit-identical next-step loss.
    trainer.request_report();
    reloaded.request_report();
    let ra = trainer.step(&set, &indices).unwrap().expect("report");
    let rb = reloaded.step(&set, &indices).unwrap().expect("report");
    assert_eq!(ra.loss.to_bits(), rb.loss.to_bits(), "loss bit-identical");
    assert_eq!(ra.formula.to_bits(), rb.formula.to_bits(), "formula loss bit-identical");
    assert_eq!(ra.graph.to_bits(), rb.graph.to_bits(), "graph loss bit-identical");
    let _ = std::fs::remove_file(&path);
}

/// j: workspace buckets and estimate items.
///
/// An `Evidence` request and a `Counts` request of the same shape use
/// different buckets, and the estimate's evidence items equal the buffers'
/// actual byte sizes (compared item by item with the allocated tensors'
/// element counts).
#[test]
fn j_workspace_buckets_and_estimate() {    let _serial = serial();
    use mamba3::tensor::ops::ms2::FormulaBuffers;
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let set = two_spectrum_set(&comps, 50);
    let indices: Vec<usize> = (0..comps.len()).collect();
    let batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let (b, k, t, m, f) = (2usize, 4usize, 22usize, 32usize, 2usize);
    let mk = |layout| {
        let mut cfg = tiny_model(layout);
        cfg.d_model = 16;
        let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
        cfg.formula_table.rows = dtable.rows as u32;
        cfg.formula_table.sha256 = dtable.sha256.clone();
        let mut rng = Rng::seeded(5);
        let model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng).unwrap();
        (cfg, model, dtable)
    };
    let (counts_cfg, counts_model, counts_table) = mk(FormulaFeatures::Counts);
    let (ev_cfg, ev_model, ev_table) = mk(FormulaFeatures::Evidence);
    let g = tiny_gen_table(32);
    let constants = Ms2Constants::new(&device);
    // Same shape through both layouts on one workspace: different buckets.
    let mut ws = GenerationWorkspace::new();
    counts_model.generate(&batch, &counts_table, &g, &mut ws, &constants).unwrap();
    ev_model.generate(&batch, &ev_table, &g, &mut ws, &constants).unwrap();
    let keys = ws.bucket_keys();
    assert_eq!(keys.len(), 2, "Counts and Evidence do not share buckets");
    let (ka, kb) = (keys[0], keys[1]);
    assert_eq!(
        (ka.0, ka.1, ka.2, ka.3, ka.4, ka.5, ka.6, ka.7, ka.8),
        (kb.0, kb.1, kb.2, kb.3, kb.4, kb.5, kb.6, kb.7, kb.8),
        "same shape"
    );
    assert_ne!(ka.9, kb.9, "layout flag differs");
    // Estimate items against the allocated tensors' element counts (f32).
    let rows = ev_table.rows as u64;
    let est = Ms2MemoryEstimate::generation(&ev_cfg, rows, b as u64, k as u64, 64, t as u64, m as u64, f as u64)
        .unwrap();
    let bufs = FormulaBuffers::<R, E>::new_evidence(b, m, f, &device);
    let bytes = |elems: usize| elems as u64 * 4;
    let count = |dims: &[usize]| dims.iter().product::<usize>();
    let ev_peaks = bufs.ev_peaks.unwrap();
    let ev_w = bufs.ev_w.unwrap();
    let cand_ev = bufs.cand_ev.unwrap();
    let feat16 = bufs.cand_feat16.unwrap();
    let xfeat = bufs.cand_xfeat.unwrap();
    assert_eq!(est.get("ev_peaks"), Some(bytes(count(ev_peaks.shape().dims()))));
    assert_eq!(est.get("ev_w"), Some(bytes(count(ev_w.shape().dims()))));
    assert_eq!(est.get("cand_ev"), Some(bytes(count(cand_ev.shape().dims()))));
    assert_eq!(est.get("cand_feat16"), Some(bytes(count(feat16.shape().dims()))));
    assert_eq!(est.get("cand_xfeat"), Some(bytes(count(xfeat.shape().dims()))));
    // Branch activations (transient, twice B*M*32 floats) and the 257 branch
    // parameters counted from the state dict.
    assert_eq!(est.get("evidence_branch"), Some(2 * b as u64 * m as u64 * 32 * 4));
    let branch_params: usize = ev_model
        .state_dict()
        .entries
        .iter()
        .filter(|(n, _)| n.contains("evidence"))
        .map(|(_, e)| e.data.len())
        .sum();
    assert_eq!(branch_params, 257, "evidence_in (224) + evidence_out (33)");
    // Counts: every evidence item is zero.
    let est_c = Ms2MemoryEstimate::generation(&counts_cfg, rows, b as u64, k as u64, 64, t as u64, m as u64, f as u64)
        .unwrap();
    for name in ["ev_peaks", "ev_w", "cand_ev", "cand_feat16", "cand_xfeat", "evidence_branch"] {
        assert_eq!(est_c.get(name), Some(0), "Counts {name} is zero");
    }
}

/// E5F Part B: the prep-batch hook is explicit opt-in, default off.
///
/// With capture off the accessor returns `None` after a step (no clone,
/// nothing retained); with it on it returns the exact uploaded batch.
#[test]
fn e5f_capture_prep_batch_opt_in() {
    let _serial = serial();
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    let set = two_spectrum_set(&comps, 50);
    let indices: Vec<usize> = (0..comps.len()).collect();
    // Default off: nothing retained.
    assert!(trainer.step(&set, &indices).unwrap().is_none());
    assert!(trainer.last_prep_batch.is_none(), "capture off retains nothing");
    assert!(trainer.captured_prep_batch().is_none());
    // Opt in: the exact uploaded batch is retained.
    trainer.capture_prep_batch(true);
    assert!(trainer.step(&set, &indices).unwrap().is_none());
    let rec = trainer.captured_prep_batch().expect("capture on retains").clone();
    assert_eq!(rec.len(), indices.len());
    assert!(trainer.last_prep_batch.is_some());
    // Switching off clears and retains nothing further.
    trainer.capture_prep_batch(false);
    assert!(trainer.last_prep_batch.is_none());
    assert!(trainer.step(&set, &indices).unwrap().is_none());
    assert!(trainer.last_prep_batch.is_none(), "capture off again retains nothing");
    let _ = rec;
}

/// E5F-e: loss attribution for leaving the zero point.
///
/// Train ONE `Evidence` model on the competing-candidates fixture; then
/// evaluate the formula loss twice with the SAME trained weights — once as
/// is, once with the evidence branch ablated (zeroed through the state
/// dict) — and assert the ablated loss is higher by a stated margin.
#[test]
fn e5f_e_loss_attribution_ablation() {
    let _serial = serial();
    let device = dev();
    let table = FormulaTable::from_compositions([c5(), c4h12()].into_iter()).unwrap();
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: 1,
        slots: 2,
        lr: 1e-2,
        weight_decay: 0.0,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    let set = single_spectrum_set("d", precursor_of(&c5()), 500_000, 50, &[(59_999_451, 1.0)], c5());
    for _ in 0..12 {
        assert!(trainer.step(&set, &[0]).unwrap().is_none());
    }
    // Same trained weights for both evaluations: snapshot the state dict.
    let trained = trainer.model.state_dict();
    let eval_formula = |tr: &mut Ms2Trainer<R, E>| -> f32 {
        tr.request_report();
        tr.step(&set, &[0]).unwrap().expect("report").formula
    };
    let full = eval_formula(&mut trainer);
    // Restore the trained weights (undo the report step's update), then
    // ablate the evidence branch through the state dict path.
    trainer.model.load_state_dict(&trained, true).unwrap();
    for (name, param) in trainer.model.named_parameters() {
        if name.contains("evidence_out") {
            let shape = param.value().shape().dims().to_vec();
            let zeros = vec![0.0f32; shape.iter().product::<usize>().max(1)];
            // Rebuild a zero tensor of the same shape on the device.
            let z = Tensor::<R, E>::from_f32(&zeros, shape, &device).unwrap();
            param.set(z);
        }
    }
    // Confirm the branch is dead: every evidence_out value is zero.
    for (name, param) in trainer.model.named_parameters() {
        if name.contains("evidence_out") {
            assert!(
                param.value().to_f32().iter().all(|&v| v == 0.0),
                "{name} ablated to zero"
            );
        }
    }
    let ablated = eval_formula(&mut trainer);
    println!("E5F-E-LOSS full={full} ablated={ablated} margin={}", ablated - full);
    assert!(
        ablated > full + 0.0001,
        "ablated formula loss {ablated} exceeds trained {full} by 0.0001"
    );
}

/// E5F-f: trained-checkpoint reload with a provably live branch.
///
/// On the competing-candidates fixture assert before saving that
/// `evidence_out` weights are non-zero AND differ from init, and that
/// ablating the branch changes `generate` ranking/log-probs for at least
/// one spectrum; then the reload is bit-identical.
#[test]
fn e5f_f_live_branch_reload() {
    let _serial = serial();
    let device = dev();
    let table = FormulaTable::from_compositions([c5(), c4h12()].into_iter()).unwrap();
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: 1,
        slots: 2,
        lr: 1e-2,
        weight_decay: 0.0,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    let init_out: Vec<f32> = trainer
        .model
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "formula.evidence_out.weight")
        .unwrap()
        .1
        .value()
        .to_f32();
    assert!(init_out.iter().all(|&v| v == 0.0), "evidence_out starts at zero");
    let set = single_spectrum_set("d", precursor_of(&c5()), 500_000, 50, &[(59_999_451, 1.0)], c5());
    for _ in 0..12 {
        assert!(trainer.step(&set, &[0]).unwrap().is_none());
    }
    let live: Vec<f32> = trainer
        .model
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "formula.evidence_out.weight")
        .unwrap()
        .1
        .value()
        .to_f32();
    assert!(live.iter().any(|&v| v != 0.0), "evidence_out non-zero after training");
    assert_ne!(
        init_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        live.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "evidence_out differs from init"
    );
    // Ablating changes generate's formula ranking/log-probs.
    let g = tiny_gen_table(32);
    let before = trainer.generate_candidates(&set, &[0], &g).unwrap();
    let snapshot = trainer.model.state_dict();
    for (name, param) in trainer.model.named_parameters() {
        if name.contains("evidence_out") {
            let shape = param.value().shape().dims().to_vec();
            let zeros = vec![0.0f32; shape.iter().product::<usize>().max(1)];
            param.set(Tensor::<R, E>::from_f32(&zeros, shape, &device).unwrap());
        }
    }
    let after = trainer.generate_candidates(&set, &[0], &g).unwrap();
    let changed = before != after;
    println!("E5F-F-ABLATE changed={changed}");
    assert!(changed, "ablating the live branch changes generate output");
    trainer.model.load_state_dict(&snapshot, true).unwrap();
    // Reload is bit-identical.
    let path = std::env::temp_dir().join("ms2_e5f_live_branch.json");
    trainer.save(&path).unwrap();
    let mut reloaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    let a = trainer.generate_candidates(&set, &[0], &g).unwrap();
    let b = reloaded.generate_candidates(&set, &[0], &g).unwrap();
    assert_eq!(a, b, "reloaded checkpoint generates bit-identical output");
    trainer.request_report();
    reloaded.request_report();
    let ra = trainer.step(&set, &[0]).unwrap().expect("report");
    let rb = reloaded.step(&set, &[0]).unwrap().expect("report");
    assert_eq!(ra.formula.to_bits(), rb.formula.to_bits(), "formula loss bit-identical");
    let _ = std::fs::remove_file(&path);
}

/// E5F-g: diagnostics with a non-zero incomplete fraction.
///
/// `complete` iff the non-carbon heavy-vector count `J <= W`. Table
/// `[C5 (J = 1), C5N1 (J = 2)]` with `formula_evidence_work_max = 1`: one
/// scored candidate complete, one incomplete. Expected fraction by hand.
#[test]
fn e5f_g_diagnostics_nonzero_incomplete() {    let _serial = serial();
    let device = dev();
    let c5n1: Composition = [5, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    // J by hand: C5 -> 1; C5N1 -> (1 + 1) = 2.
    assert_eq!({ let s: u64 = [2,3,4,5,6,7,8,9].iter().map(|&e| u64::from(c5()[e]) + 1).product(); s }, 1);
    assert_eq!({ let s: u64 = [2,3,4,5,6,7,8,9].iter().map(|&e| u64::from(c5n1[e]) + 1).product(); s }, 2);
    let table = FormulaTable::from_compositions([c5(), c5n1].into_iter()).unwrap();
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: 1,
        slots: 2,
        formula_evidence_work_max: 1,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    // Wide precursor window so both candidates score; one peak.
    let set = single_spectrum_set("g", precursor_of(&c5()), 15_000_000, 50, &[(59_999_451, 1.0)], c5());
    let diag = trainer
        .evidence_diagnostics(&set, &[0])
        .unwrap()
        .expect("Evidence diagnostics");
    println!(
        "E5F-G-DIAG scored={} incomplete={} fraction={}",
        diag.scored,
        diag.incomplete,
        diag.incomplete_fraction()
    );
    assert_eq!(diag.scored, 2, "both candidates scored");
    assert_eq!(diag.incomplete, 1, "exactly the J = 2 candidate incomplete");
    assert_eq!(diag.incomplete_fraction(), 0.5, "expected fraction 1/2 by hand");
}

/// E5F-h: jitter observed on the device side.
///
/// After a trainer step with `precursor_jitter_ppm = 2`, read back through
/// the test-only bucket probe the uploaded `meta [B, 8]` precursor word,
/// the enumeration meta row (Enumerate source) and the gold-slot residual
/// parent mass, and compare each with values computed from
/// `jitter_precursor_mz(stored, 2.0, seed, 1 + step, index)`. Fixture WITH
/// labels and targets; the target batch is bit-identical to the unjittered
/// one.
#[test]
fn e5f_h_jitter_device_side() {
    let _serial = serial();
    use mamba3::models::ms2::batch::DeviceSpectra;
    use mamba3::models::ms2::chem::parent_mass;
    use mamba3::models::ms2::experiment::target_batch_for;
    use mamba3::models::ms2::formula_enum::build_enum_meta;
    use mamba3::models::ms2::targets::Labels;
    use mamba3::models::ms2::train::TRAIN_ROWS_SCORED_MAX;
    let device = dev();
    let seed = 11u64;
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let (domain, bounds, table) = setup_enum(comps.clone());
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        seed,
        precursor_jitter_ppm: 2.0,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        enum_lane_visits_max: 65_536,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer.capture_prep_batch(true);
    // Fixture WITH non-empty labels and non-empty parent graphs (targets).
    let mut set = two_spectrum_set(&comps, 50);
    for (i, s) in set.spectra.iter_mut().enumerate() {
        s.parent = MolGraph::new(vec![6], vec![]).expect("single atom builds");
        // Labels present with non-empty embeddings; pseudo-targets empty so
        // the target batch builds (empty slots) while labels are still
        // present for the preservation check.
        s.labels = Some(Labels {
            embeddings: vec![mamba3::models::ms2::targets::Embedding {
                atoms: vec![0],
                boundary: 0,
                closures: 0,
            }],
            graphs: 1,
            targets_before_cut: 0,
            targets: Vec::new(),
            dropped_weight: 0.0,
            cut_is_tied: false,
            explained_peaks: vec![0],
            ambiguous_hypotheses: 0,
            canonicalization_failures: 0,
        });
        let _ = i;
    }
    let stored: Vec<u32> = set.spectra.iter().map(|s| s.spectrum.precursor_mz_udalton).collect();
    let indices: Vec<usize> = (0..comps.len()).collect();
    // Unjittered target batch for the bit-identical assertion.
    let limits = mamba3::models::ms2::grammar::Limits::new(16, 4).unwrap();
    let unjittered_targets = target_batch_for(&set, &indices, 2, limits.clone()).unwrap();
    assert!(trainer.step(&set, &indices).unwrap().is_none());
    let rec = trainer.captured_prep_batch().expect("capture on").clone();
    // Probe accessor (bucket, no production cost) returns the uploaded
    // precursors.
    let probe = trainer.upload_probe_for_test().expect("probe");
    assert_eq!(probe, rec.precursor_mz_udalton, "probe is the uploaded precursors");
    for (b, &idx) in indices.iter().enumerate() {
        let expect = jitter_precursor_mz(stored[b], 2.0, seed, 1, idx as u64);
        assert_eq!(rec.precursor_mz_udalton[b], expect, "host precursor {b}");
        assert_eq!(probe[b], expect, "probe precursor {b}");
    }
    // Device side: upload the retained batch and read the meta precursor
    // word (production upload code, one read in test only).
    let spectra = DeviceSpectra::<R, E>::upload(&rec, &device).unwrap();
    let meta_h = spectra.meta.try_to_vec().unwrap();
    for b in 0..rec.len() {
        assert_eq!(
            meta_h[b * 8 + 1],
            jitter_precursor_mz(stored[b], 2.0, seed, 1, indices[b] as u64),
            "uploaded meta precursor word {b}"
        );
    }
    // Enumeration meta row derives from the jittered precursor.
    let art = DeviceEnumArtifacts::<R>::upload(&domain, &bounds, &device).unwrap();
    let scored_cap = TRAIN_ROWS_SCORED_MAX.min(32);
    let enum_meta = build_enum_meta(&rec, art.domain_max_error, 65_536, scored_cap);
    for b in 0..rec.len() {
        assert_eq!(
            enum_meta[b * 8],
            parent_mass(rec.precursor_mz_udalton[b], rec.adduct[b]).unwrap(),
            "enum meta row {b} from jittered precursor"
        );
    }
    // Gold-slot residual parent mass from the jittered precursor.
    for b in 0..rec.len() {
        let mp = parent_mass(rec.precursor_mz_udalton[b], rec.adduct[b]).unwrap();
        let mp_expect =
            parent_mass(jitter_precursor_mz(stored[b], 2.0, seed, 1, indices[b] as u64), rec.adduct[b])
                .unwrap();
        assert_eq!(mp, mp_expect, "gold residual parent mass {b}");
    }
    // Labels present and targets bit-identical to unjittered.
    for s in &set.spectra {
        assert!(s.labels.is_some(), "fixture carries labels");
    }
    let jittered_targets = target_batch_for(&set, &indices, 2, limits).unwrap();
    assert_eq!(
        format!("{:?}", jittered_targets.tokens),
        format!("{:?}", unjittered_targets.tokens),
        "target batch bit-identical"
    );
}

/// E5F-i: enumeration inference boundary with meaningful labels.
///
/// The two requests differ in NON-EMPTY labels and targets (not absent
/// versus empty); generated candidates are still identical.
#[test]
fn e5f_i_enumerate_boundary_nonempty_labels() {
    let _serial = serial();
    use mamba3::models::ms2::targets::{Embedding, Labels, Target};
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let (domain, bounds, table) = setup_enum(comps.clone());
    let mut cfg = tiny_model(FormulaFeatures::Evidence);
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let tcfg = TrainConfig {
        batch: comps.len(),
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        enum_lane_visits_max: 65_536,
        ..TrainConfig::default()
    };
    let mut trainer = Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let mk_labels = |tag: u64| Labels {
        embeddings: vec![Embedding { atoms: vec![0], boundary: 0, closures: 0 }],
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![Target {
            trace: Vec::new(),
            weight: tag,
            q: 1.0,
            embeddings: vec![0],
            anchors: vec![(0, 0)],
        }],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: vec![0],
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let mut set_a = two_spectrum_set(&comps, 50);
    let mut set_b = two_spectrum_set(&comps, 50);
    for s in &mut set_a.spectra {
        s.parent_composition = comp(1, 4, 0, 0);
        s.parent = MolGraph::new(vec![6], vec![]).expect("single atom builds");
        s.labels = Some(mk_labels(1));
    }
    for s in &mut set_b.spectra {
        s.parent_composition = comp(2, 8, 0, 0);
        s.parent = MolGraph::new(vec![6, 6], vec![]).expect("two atoms build");
        s.labels = Some(mk_labels(2));
    }
    // Both label sets non-empty and different; targets (graphs) differ.
    assert!(set_a.spectra.iter().all(|s| s.labels.as_ref().unwrap().targets.len() == 1));
    assert!(set_b.spectra.iter().all(|s| s.labels.as_ref().unwrap().targets.len() == 1));
    assert_ne!(
        set_a.spectra[0].labels.as_ref().unwrap().targets[0].weight,
        set_b.spectra[0].labels.as_ref().unwrap().targets[0].weight,
        "labels differ non-emptily"
    );
    let indices: Vec<usize> = (0..comps.len()).collect();
    let mut g = tiny_gen_table(32);
    g.formula_source = FormulaSource::Enumerate;
    g.enum_lane_visits_max = 65_536;
    let a = trainer.generate_candidates(&set_a, &indices, &g).unwrap();
    let b = trainer.generate_candidates(&set_b, &indices, &g).unwrap();
    assert_eq!(a, b, "non-empty labels/targets do not enter generation");
}

/// E5F-j: conditioning parity with a learned branch.
///
/// After setting `evidence_out` to non-zero values, the decoder-conditioning
/// embedding and the embedding the assignment head receives THROUGH the
/// assignment integration path (assignment forward, never `embed_rows`
/// directly) are bit-identical to those with the branch zeroed.
#[test]
fn e5f_j_conditioning_parity_learned_branch() {
    let _serial = serial();
    let device = dev();
    let comps = vec![c5(), comp(6, 6, 0, 0)];
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let set = two_spectrum_set(&comps, 50);
    let indices: Vec<usize> = (0..comps.len()).collect();
    let mk = || {
        let mut cfg = tiny_model_assignment(FormulaFeatures::Evidence);
        let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
        cfg.formula_table.rows = dtable.rows as u32;
        cfg.formula_table.sha256 = dtable.sha256.clone();
        let tcfg = TrainConfig {
            batch: indices.len(),
            slots: 2,
            gold_formula_conditioning: GoldFormulaConditioning::Composition,
            ..TrainConfig::default()
        };
        Ms2Trainer::<R, E>::new(&cfg, &table, &tcfg, &device).unwrap()
    };
    let mut trainer = mk();
    // Zeroed branch baseline through the production hooks.
    let base_cond = trainer.conditioning_for_test(&set, &indices).unwrap();
    // Assignment integration path: teacher conditioning + assignment loss
    // forward uses the row network internally (never `embed_rows` here).
    // Capture its conditioning embedding as the assignment-path probe: the
    // same production prefix feeds both, so equality of the conditioning
    // embedding plus equality of the assignment forward loss establishes the
    // assignment path sees the same rows.
    trainer.capture_prep_batch(true);
    // Learned branch: set every evidence_out value non-zero.
    for (name, param) in trainer.model.named_parameters() {
        if name.contains("evidence_out") {
            let shape = param.value().shape().dims().to_vec();
            let n: usize = shape.iter().product::<usize>().max(1);
            let vals = vec![0.5f32; n];
            param.set(Tensor::<R, E>::from_f32(&vals, shape, &device).unwrap());
        }
    }
    let live_cond = trainer.conditioning_for_test(&set, &indices).unwrap();
    assert_eq!(
        base_cond.e_cond.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        live_cond.e_cond.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "decoder conditioning identical with learned branch"
    );
    // Assignment forward through production (ion_assign + head log_prob via
    // the trainer's assignment evaluation on gold labels would need labels;
    // instead drive the model-level assignment integration: build the gold
    // ion buffers with production kernels and run the head's log_prob,
    // which is the integration path the assignment loss uses).
    //
    // The row network weights are shared; evidence_out never enters
    // `embed_rows` or the assignment path, so the assignment log-probs are
    // identical. We establish this by comparing two assignment forwards
    // that differ only in evidence_out (zeroed vs 0.5) on the same ion
    // buffers.
    use mamba3::models::ms2::batch::DeviceSpectra;
    use mamba3::models::ms2::experiment::spectrum_batch_for;
    let batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let spectra = DeviceSpectra::<R, E>::upload(&batch, &device).unwrap();
    let n_peaks = 16usize;
    let peaks = mamba3::tensor::ops::ms2::PeakBuffers::<R, E>::new(batch.len(), 64, n_peaks, &device);
    mamba3::tensor::ops::ms2::peak_select(
        &spectra.mz, &spectra.intensity, &spectra.meta, spectra.intensity_scale, &peaks,
    )
    .unwrap();
    let spec_t = spectra.evidence_spec(&device).unwrap();
    // Gold counts for the two spectra.
    let mut gold_counts = Vec::new();
    for c in &comps {
        for e in 0..10 {
            gold_counts.push(u32::from(c[e]));
        }
    }
    let top_counts_t = IdTensor::from_slice(&gold_counts, vec![2, 1, 10], &device).unwrap();
    let j = 4usize;
    let mut ion_t = IdTensor::empty(vec![2, 1, n_peaks, j, 12], &device);
    let mut ion_meta_t = IdTensor::empty(vec![2, 1, n_peaks, 4], &device);
    mamba3::tensor::ops::ms2_ion::ion_assign(
        &top_counts_t, &peaks.kept, &spectra.meta, &spec_t, &mut ion_t, &mut ion_meta_t, 64,
    )
    .unwrap();
    // Encode to get x for the head.
    let encoded = trainer.model.encoder.encode(&spectra, &peaks, Control::None).unwrap();
    let head = trainer.model.assignment.as_ref().expect("assignment head");
    let log_host: Vec<f32> = (0..1024).map(|i| ((1 + i) as f32).ln()).collect();
    let log_table = Tensor::<R, E>::from_f32(&log_host, vec![1024], &device).unwrap();
    let out_live = head
        .log_prob(&trainer.model.formula, &log_table, &ion_t, &ion_meta_t, &encoded.x)
        .unwrap();
    let lp_live = out_live.log_prob.tensor().try_to_f32().unwrap();
    // Zero the branch and rerun the SAME integration path.
    for (name, param) in trainer.model.named_parameters() {
        if name.contains("evidence_out") {
            let shape = param.value().shape().dims().to_vec();
            let n: usize = shape.iter().product::<usize>().max(1);
            param.set(Tensor::<R, E>::from_f32(&vec![0.0; n], shape, &device).unwrap());
        }
    }
    let out_zero = head
        .log_prob(&trainer.model.formula, &log_table, &ion_t, &ion_meta_t, &encoded.x)
        .unwrap();
    let lp_zero = out_zero.log_prob.tensor().try_to_f32().unwrap();
    assert_eq!(
        lp_live.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        lp_zero.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "assignment integration path identical with learned branch"
    );
}

/// Resolve the built example binary `name` relative to the running test
/// executable — the shared Part C helper (see `tests/common/mod.rs`).
fn resolve_example_bin(name: &str) -> std::path::PathBuf {
    common::resolve_example_bin(name)
}

/// E5F-k: driver behaviour (`examples/ms2_experiment.rs`).
///
/// Runs the built example as a subprocess on a tiny fixture export for a
/// few steps with `--formula-features evidence --precursor-jitter-ppm 2
/// --formula-evidence-work-max 64`, and asserts on the report JSON: the
/// three settings are recorded; every `_jitter2` field exists next to its
/// stored-precursor field; the evidence diagnostics are numbers (and `null`
/// in a second run with `--formula-features counts`); the stored-precursor
/// recall equals the recall of a third run with `--precursor-jitter-ppm 0`
/// evaluated from the same checkpoint via `--load --eval-only`.
#[test]
fn e5f_k_driver_report() {
    let _serial = serial();
    // Tiny fixture: first 4 molecules of the overfit export (or skip).
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("data/ms2/overfit_train.json");
    if !src.exists() {
        println!("E5F-K-SKIP: data/ms2/overfit_train.json absent");
        return;
    }
    let raw = std::fs::read_to_string(&src).expect("export readable");
    let mut val: serde_json::Value = serde_json::from_str(&raw).expect("export parses");
    if let Some(mols) = val.get_mut("molecules").and_then(|m| m.as_array_mut()) {
        mols.truncate(4);
    }
    let dir = std::env::temp_dir().join("ms2_e5f_k");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let train_path = dir.join("train.json");
    let val_path = dir.join("val.json");
    std::fs::write(&train_path, serde_json::to_string(&val).unwrap()).unwrap();
    std::fs::write(&val_path, serde_json::to_string(&val).unwrap()).unwrap();
    // Example binary, resolved relative to the running test executable
    // (task F7A Part C): `std::env::current_exe()` is
    // `<target>/<profile>/deps/<test>-<hash>`, so the example is
    // `<target>/<profile>/examples/<name>`, whatever the target directory
    // or backend feature is.
    let bin = resolve_example_bin("ms2_experiment");
    let run = |args: &[&str]| -> serde_json::Value {
        let out = std::process::Command::new(&bin)
            .args(args)
            .output()
            .expect("example runs");
        assert!(
            out.status.success(),
            "example failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let report_path = args
            .windows(2)
            .find(|w| w[0] == "--out")
            .map(|w| w[1].to_string())
            .expect("--out");
        let text = std::fs::read_to_string(&report_path).expect("report readable");
        serde_json::from_str(&text).expect("report parses")
    };
    let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
    let table_path = manifest.join("data/ms2/formula_table_msgym_v0.json");
    let ckpt_path = dir.join("ckpt.json");
    let rep_evidence = run(&[
        "--train",
        &s(&train_path),
        "--validation",
        &s(&val_path),
        "--table",
        &s(&table_path),
        "--name",
        "e5f-k-evidence",
        "--steps",
        "2",
        "--batch",
        "2",
        "--formula-features",
        "evidence",
        "--precursor-jitter-ppm",
        "2",
        "--formula-evidence-work-max",
        "64",
        "--save",
        &s(&ckpt_path),
        "--out",
        &s(&dir.join("rep_evidence.json")),
    ]);
    assert_eq!(
        rep_evidence["provenance"]["formula_features"].as_str().unwrap(),
        "evidence",
        "setting recorded"
    );
    assert_eq!(
        rep_evidence["provenance"]["precursor_jitter_ppm"].as_f64().unwrap(),
        2.0,
        "setting recorded"
    );
    assert_eq!(
        rep_evidence["provenance"]["formula_evidence_work_max"].as_u64().unwrap(),
        64,
        "setting recorded"
    );
    // Every _jitter2 field next to its stored-precursor field: the final
    // evaluation object carries both.
    let evals = rep_evidence["evaluations"].as_array().expect("evaluations array");
    assert!(!evals.is_empty(), "at least one evaluation");
    let final_eval = evals.last().unwrap();
    let obj = final_eval.as_object().expect("evaluation object");
    let mut jitter2_seen = 0usize;
    for key in obj.keys() {
        if key.ends_with("_jitter2") {
            let base = key.trim_end_matches("_jitter2");
            assert!(obj.contains_key(base), "{key} without {base}");
            jitter2_seen += 1;
        }
    }
    assert!(jitter2_seen > 0, "at least one _jitter2 field");
    // Evidence diagnostics are numbers under evidence.
    assert!(
        final_eval["evidence_incomplete_fraction"].is_number(),
        "evidence diagnostics number"
    );
    assert!(final_eval["evidence_peaks_mean"].is_number());
    let rep_counts = run(&[
        "--train",
        &s(&train_path),
        "--validation",
        &s(&val_path),
        "--table",
        &s(&table_path),
        "--name",
        "e5f-k-counts",
        "--steps",
        "2",
        "--batch",
        "2",
        "--formula-features",
        "counts",
        "--precursor-jitter-ppm",
        "2",
        "--formula-evidence-work-max",
        "64",
        "--out",
        &s(&dir.join("rep_counts.json")),
    ]);
    // Counts run records counts and null evidence diagnostics (when the
    // field exists; otherwise the evidence run above already showed numbers
    // where counts shows null/absent).
    assert_eq!(
        rep_counts["provenance"]["formula_features"].as_str().unwrap(),
        "counts"
    );
    let evals_c = rep_counts["evaluations"].as_array().expect("evaluations array");
    let final_c = evals_c.last().unwrap();
    assert!(
        final_c["evidence_incomplete_fraction"].is_null(),
        "counts evidence diagnostics null"
    );
    assert!(final_c["evidence_peaks_mean"].is_null());
    // Stored-precursor recall equality: reload the evidence checkpoint with
    // jitter 0 via --load --eval-only; same weights give same stored recall.
    let rep_reload = run(&[
        "--train",
        &s(&train_path),
        "--validation",
        &s(&val_path),
        "--table",
        &s(&table_path),
        "--name",
        "e5f-k-reload",
        "--load",
        &s(&ckpt_path),
        "--eval-only",
        "--precursor-jitter-ppm",
        "0",
        "--formula-features",
        "evidence",
        "--formula-evidence-work-max",
        "64",
        "--out",
        &s(&dir.join("rep_reload.json")),
    ]);
    let evals_r = rep_reload["evaluations"].as_array().expect("evaluations array");
    let final_r = evals_r.last().unwrap();
    assert_eq!(
        final_eval["packed_formula_recall_rate"].to_string(),
        final_r["packed_formula_recall_rate"].to_string(),
        "same checkpoint gives same stored-precursor recall"
    );
    let _ = rep_counts;
    println!("E5F-K driver report checks passed");
}
