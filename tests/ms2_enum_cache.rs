//! Memoised device enumeration for training and evaluation (task T6).
//!
//! Exactness (cached `cand`/`counters` bit-identical to the device
//! enumeration), footprint (zero enumeration launches and no added reads on
//! a hit), header discipline, the compact encoding, refusal parity and the
//! fixed jitter pool. CPU and GPU via `backends::Auto`: integer fields and
//! cpu floats compare bit-equal, GPU floats within 1e-6.

#![cfg(feature = "backend")]

use std::sync::{Arc, Mutex};

use mamba3::backend::{
    Device, launch_count, launch_tally_detailed, read_count, reset_launch_count,
    reset_read_count, start_launch_tally, stop_launch_tally,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    AllocationMode, CandidateBatch, Control, FormulaSource, GenerationConfig, GenerationMode,
    IdentityMode, ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
    request_status,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::enum_cache::{EnumCache, EnumCacheHeader};
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, apply_precursor_jitter,
    spectrum_batch_for,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{
    EnumDomain, RatioBounds, build_enum_meta,
};
use mamba3::models::ms2::formula_evidence_ref::jitter_precursor_mz;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::pack::PackedCandidateBatch;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};
use mamba3::tensor::ops::random::Rng;

/// File-level serialisation: the process-wide launch/read counters are
/// perturbed by any test running beside these, so every test holds this
/// mutex.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Whether the test backend is the CPU runtime (bit-equal floats) rather
/// than a GPU one (1e-6).
fn is_cpu() -> bool {
    std::any::type_name::<R>().contains("Cpu")
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
        comp(3, 7, 1, 2),
        comp(6, 6, 0, 0),
        comp(6, 12, 0, 6),
        comp(4, 9, 1, 1),
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

fn tiny_gen(window: u32, rows_scored_max: u32) -> GenerationConfig {
    GenerationConfig {
        schema_version: SCHEMA_VERSION,
        trajectories: 4,
        formulas: 2,
        seed: 7,
        temperature: 1.0,
        max_steps: 22,
        max_device_bytes: 2 * 1024 * 1024 * 1024,
        formula_rows_visited_max: u32::MAX,
        formula_rows_scored_max: rows_scored_max,
        mode: GenerationMode::Sampling,
        oracle_formula: false,
        control: Control::None,
        formula_source: FormulaSource::Enumerate,
        formula_window: window,
        enum_lanes_max: 262_144,
        enum_lane_visits_max: 65_536,
        enum_dispatch_visits_max: 4_000_000,
        allocation: AllocationMode::RoundRobin,
        identity: IdentityMode::TraceOnly,
        identity_work_max: 4096,
        returned: 0,
        evidence: false,
        ion_request_work_max: 268435456,
        formula_evidence_work_max: 2048,
        formula_evidence_dispatch_max: 268435456,
    }
}

fn tiny_train() -> TrainConfig {
    TrainConfig {
        batch: 2,
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
        lambda_assign: 0.0,
        ..TrainConfig::default()
    }
}

/// Protonated precursor of a composition (adduct 1), as the other enum
/// tests form it.
fn precursor_of(c: &Composition) -> u32 {
    composition_mass(c).unwrap() + 1_007_825 - 549
}

/// One hand-built request batch: fixed peaks, per-spectrum precursor data.
fn spectrum_batch(
    precursors: &[u32],
    uncs: &[u32],
    adducts: &[u16],
    tol_tenths: &[u16],
    n_raw: usize,
) -> SpectrumBatch {
    let b = precursors.len();
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: (1..=b as u64).collect(),
        raw_peak_count: vec![8; b],
        peak_count: vec![8; b],
        peak_id: (0..b * n_raw).map(|i| (i % n_raw) as u32).collect(),
        mz_udalton: vec![60_000_000; b * n_raw],
        intensity: vec![1.0; b * n_raw],
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
        precursor_tolerance_ppm_tenths: tol_tenths.to_vec(),
        instrument_class: vec![0; b],
    }
}

/// Model, resident table and uploaded artifacts over the fixture domain.
fn setup_model(
    device: &Device<R>,
    comps: &[Composition],
    seed: u64,
) -> (FormulaTable, DeviceFormulaTable<R, E>, Ms2Model<R, E>) {
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, device).unwrap();
    let domain = EnumDomain::from_compositions(comps.to_vec(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.to_vec(), 0).unwrap();
    let mut cfg = tiny_model();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(seed);
    let mut model = Ms2Model::<R, E>::init(&cfg, device, &mut rng).unwrap();
    model
        .upload_enum_artifacts(&domain, &bounds, device)
        .unwrap();
    (table, dtable, model)
}

/// Minimal experiment set over these spectra for the trainer.
fn experiment_set(comps: &[Composition], batch: &SpectrumBatch) -> ExperimentSet {
    let n_raw = batch.n_raw as usize;
    let spectra = comps
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let base = i * n_raw;
            let np = 8usize;
            ExperimentSpectrum {
                molecule: i,
                spectrum: ExportSpectrum {
                    row: i as u64,
                    spectrum_id: batch.spectrum_id[i],
                    adduct: 1,
                    polarity: 1,
                    precursor_mz_udalton: batch.precursor_mz_udalton[i],
                    precursor_uncertainty_udalton: batch.precursor_uncertainty_udalton[i],
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
                parent_composition: *c,
                labels: None,
                domain: SpectrumDomain::InDomainUnlabeled,
            }
        })
        .collect();
    ExperimentSet {
        name: "enum-cache".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: (0..comps.len()).map(|i| format!("mol{i}")).collect(),
        spectra,
    }
}

/// Run the production enumeration for one batch and read `cand` and
/// `counters` back (the uncached device result).
fn device_enumerate(
    device: &Device<R>,
    model: &Ms2Model<R, E>,
    batch: &SpectrumBatch,
    scored_cap: u32,
    m: usize,
    lane_visits_max: u32,
) -> (Vec<u32>, Vec<u32>) {
    let b = batch.len();
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let meta_host = build_enum_meta(
        batch,
        artifacts.domain_max_error,
        lane_visits_max,
        scored_cap,
    );
    let meta_t = IdTensor::from_slice(&meta_host, vec![b, 8], device).unwrap();
    let launch = EnumLaunch::from_chemistry();
    let stats_t = IdTensor::empty(vec![b * artifacts.p, 2], device);
    let offsets_t = IdTensor::empty(vec![b * artifacts.p], device);
    let cand_t = IdTensor::empty(vec![b, m, 13], device);
    let counters_t = IdTensor::empty(vec![b, 5], device);
    launch
        .count(
            &meta_t,
            &artifacts.rare,
            &artifacts.bounds,
            &stats_t,
            262_144,
            4_000_000,
            lane_visits_max,
        )
        .unwrap();
    enum_offsets(
        &stats_t,
        &meta_t,
        &offsets_t,
        &counters_t,
        scored_cap,
        m,
        262_144,
    )
    .unwrap();
    launch
        .fill(
            &meta_t,
            &artifacts.rare,
            &artifacts.bounds,
            &offsets_t,
            &cand_t,
            scored_cap,
            262_144,
            4_000_000,
            lane_visits_max,
        )
        .unwrap();
    cand_pad(&counters_t, &cand_t, artifacts.p, 262_144).unwrap();
    let cand = cand_t.to_vec();
    let counters = counters_t.to_vec();
    (cand, counters)
}

fn meta_keys(batch_meta: &[u32]) -> Vec<[u32; 8]> {
    batch_meta
        .chunks_exact(8)
        .map(|r| {
            let mut k = [0u32; 8];
            k.copy_from_slice(r);
            k
        })
        .collect()
}

fn assert_f32_close(a: f32, b: f32, ctx: &str) {
    assert!(a.is_finite() && b.is_finite(), "{ctx}: non-finite {a} vs {b}");
    assert!(
        (a - b).abs() <= 1e-6,
        "{ctx}: {a} vs {b} differ by more than 1e-6"
    );
}

fn assert_candidate_batch_close(a: &CandidateBatch, b: &CandidateBatch) {
    if is_cpu() {
        assert_eq!(a, b, "cached generate differs (cpu demands bit-equality)");
        return;
    }
    assert_eq!(a.schema_version, b.schema_version);
    assert_eq!(a.batch, b.batch);
    assert_eq!(a.trajectories, b.trajectories);
    assert_eq!(a.max_steps, b.max_steps);
    assert_eq!(a.max_atoms, b.max_atoms);
    assert_eq!(a.max_ring_closures, b.max_ring_closures);
    assert_eq!(a.spectrum_id, b.spectrum_id);
    assert_eq!(a.trajectory, b.trajectory);
    assert_eq!(a.actions, b.actions);
    assert_eq!(a.length, b.length);
    assert_eq!(a.formula_row, b.formula_row);
    assert_eq!(a.formula_rank, b.formula_rank);
    assert_eq!(a.formula_counts, b.formula_counts);
    assert_eq!(a.status, b.status);
    assert_eq!(a.request_status, b.request_status);
    assert_eq!(a.rows_visited, b.rows_visited);
    assert_eq!(a.rows_joined, b.rows_joined);
    assert_eq!(a.rows_scored, b.rows_scored);
    assert_eq!(a.formula_support_complete, b.formula_support_complete);
    assert_eq!(a.peaks_kept, b.peaks_kept);
    assert_eq!(a.open_valence, b.open_valence);
    assert_eq!(a.attachment_partition, b.attachment_partition);
    assert_eq!(a.evidence_status, b.evidence_status);
    assert_eq!(a.identity_resolution, b.identity_resolution);
    assert_eq!(a.formula_source, b.formula_source);
    assert_eq!(a.evidence_count, b.evidence_count);
    assert_eq!(a.evidence_peak_id, b.evidence_peak_id);
    assert_eq!(a.evidence_hypothesis, b.evidence_hypothesis);
    assert_eq!(a.evidence_shift, b.evidence_shift);
    assert_eq!(a.evidence_residual, b.evidence_residual);
    for (i, (&x, &y)) in a.formula_log_prob.iter().zip(&b.formula_log_prob).enumerate() {
        assert_f32_close(x, y, &format!("formula_log_prob[{i}]"));
    }
    for (i, (&x, &y)) in a.trace_log_prob.iter().zip(&b.trace_log_prob).enumerate() {
        assert_f32_close(x, y, &format!("trace_log_prob[{i}]"));
    }
    for (i, (&x, &y)) in a
        .formula_mass_retained
        .iter()
        .zip(&b.formula_mass_retained)
        .enumerate()
    {
        assert_f32_close(x, y, &format!("formula_mass_retained[{i}]"));
    }
    for (i, (&x, &y)) in a
        .intensity_retained
        .iter()
        .zip(&b.intensity_retained)
        .enumerate()
    {
        assert_f32_close(x, y, &format!("intensity_retained[{i}]"));
    }
    for (i, (&x, &y)) in a
        .evidence_log_prob
        .iter()
        .zip(&b.evidence_log_prob)
        .enumerate()
    {
        assert_f32_close(x, y, &format!("evidence_log_prob[{i}]"));
    }
}

fn assert_packed_close(a: &PackedCandidateBatch, b: &PackedCandidateBatch) {
    if is_cpu() {
        assert_eq!(a, b, "cached generate_packed differs (cpu demands bit-equality)");
        return;
    }
    assert_eq!(a.schema_version, b.schema_version);
    assert_eq!(a.batch, b.batch);
    assert_eq!(a.returned, b.returned);
    assert_eq!(a.trajectories, b.trajectories);
    assert_eq!(a.max_steps, b.max_steps);
    assert_eq!(a.max_atoms, b.max_atoms);
    assert_eq!(a.max_ring_closures, b.max_ring_closures);
    assert_eq!(a.spectrum_id, b.spectrum_id);
    assert_eq!(a.trajectory, b.trajectory);
    assert_eq!(a.actions, b.actions);
    assert_eq!(a.length, b.length);
    assert_eq!(a.formula_row, b.formula_row);
    assert_eq!(a.formula_rank, b.formula_rank);
    assert_eq!(a.formula_counts, b.formula_counts);
    assert_eq!(a.status, b.status);
    assert_eq!(a.returned_count, b.returned_count);
    assert_eq!(a.request_status, b.request_status);
    assert_eq!(a.rows_visited, b.rows_visited);
    assert_eq!(a.rows_joined, b.rows_joined);
    assert_eq!(a.rows_scored, b.rows_scored);
    assert_eq!(a.formula_support_complete, b.formula_support_complete);
    assert_eq!(a.peaks_kept, b.peaks_kept);
    assert_eq!(a.open_valence, b.open_valence);
    assert_eq!(a.evidence_status, b.evidence_status);
    assert_eq!(a.evidence_count, b.evidence_count);
    assert_eq!(a.evidence_peak_id, b.evidence_peak_id);
    assert_eq!(a.evidence_hypothesis, b.evidence_hypothesis);
    assert_eq!(a.evidence_shift, b.evidence_shift);
    assert_eq!(a.evidence_residual, b.evidence_residual);
    assert_eq!(a.identity_resolution, b.identity_resolution);
    assert_eq!(a.attachment_partition, b.attachment_partition);
    assert_eq!(a.formula_source, b.formula_source);
    for (i, (&x, &y)) in a.formula_log_prob.iter().zip(&b.formula_log_prob).enumerate() {
        assert_f32_close(x, y, &format!("packed formula_log_prob[{i}]"));
    }
    for (i, (&x, &y)) in a.trace_log_prob.iter().zip(&b.trace_log_prob).enumerate() {
        assert_f32_close(x, y, &format!("packed trace_log_prob[{i}]"));
    }
    for (i, (&x, &y)) in a.score.iter().zip(&b.score).enumerate() {
        assert_f32_close(x, y, &format!("packed score[{i}]"));
    }
    for (i, (&x, &y)) in a
        .formula_mass_retained
        .iter()
        .zip(&b.formula_mass_retained)
        .enumerate()
    {
        assert_f32_close(x, y, &format!("packed formula_mass_retained[{i}]"));
    }
    for (i, (&x, &y)) in a
        .intensity_retained
        .iter()
        .zip(&b.intensity_retained)
        .enumerate()
    {
        assert_f32_close(x, y, &format!("packed intensity_retained[{i}]"));
    }
    for (i, (&x, &y)) in a
        .evidence_log_prob
        .iter()
        .zip(&b.evidence_log_prob)
        .enumerate()
    {
        assert_f32_close(x, y, &format!("packed evidence_log_prob[{i}]"));
    }
}

#[test]
fn round_trip_element_for_element() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, _dtable, model) = setup_model(&device, &comps, 5);
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let domain_max_error = artifacts.domain_max_error;
    // Batch A (M = 32): one productive spectrum, one with nothing joined,
    // one with unknown precision, one outside DEVICE_HALF_MAX.
    let productive = precursor_of(&comps[4]);
    let batch_a = spectrum_batch(
        &[productive, 4_000_000_000, productive, productive],
        &[50, 50, u32::MAX, 100_000_000],
        &[1, 1, 1, 1],
        &[0, 0, 0, 0],
        64,
    );
    let m_a = 32usize;
    let scored_a = 32u32;
    let (cand_dev, counters_dev) =
        device_enumerate(&device, &model, &batch_a, scored_a, m_a, 65_536);
    // Qualitative shape of the fixture (what makes each row its kind).
    assert!(
        counters_dev[2] > 0,
        "row 0 must be productive (rows_scored > 0), got {counters_dev:?}"
    );
    assert_eq!(counters_dev[1 * 5 + 1], 0, "row 1 must join nothing");
    assert_eq!(counters_dev[2 * 5 + 1], 0, "row 2 (unknown precision) joins nothing");
    assert_eq!(counters_dev[3 * 5 + 1], 0, "row 3 (half overflow) joins nothing");
    assert_ne!(
        counters_dev[3 * 5 + 3] & request_status::FORMULA_SEARCH_EXHAUSTED,
        0,
        "row 3 (half overflow) must report exhausted"
    );
    // Build the cache through the production build API and expand it.
    let gcfg = tiny_gen(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch_a].into_iter(), &gcfg, &mut cache)
        .unwrap();
    assert_eq!(cache.len(), 4);
    let meta_a = build_enum_meta(&batch_a, domain_max_error, 65_536, scored_a);
    let keys_a = meta_keys(&meta_a);
    let (cand_hit, counters_hit) = cache.expand_batch(&keys_a, m_a).expect("all hit");
    assert_eq!(cand_hit, cand_dev, "cached cand differs element for element");
    assert_eq!(
        counters_hit, counters_dev,
        "cached counters differ element for element (incl. status bits)"
    );
    // Batch B (M = 8, scored cap 0): a spectrum over the scored cap — its
    // single join is truncated to nothing (rows_scored 0 < joined), with
    // the exhausted bit set and every slot the fixed padding row.
    let batch_b = spectrum_batch(&[productive], &[50], &[1], &[1000], 64);
    let m_b = 8usize;
    let scored_b = 0u32;
    let (cand_dev_b, counters_dev_b) =
        device_enumerate(&device, &model, &batch_b, scored_b, m_b, 65_536);
    assert!(
        counters_dev_b[1] > counters_dev_b[2],
        "row must be over the scored cap (joined {} > scored {})",
        counters_dev_b[1],
        counters_dev_b[2]
    );
    assert_eq!(counters_dev_b[2], scored_b);
    let mut gcfg_b = tiny_gen(32, 4096);
    gcfg_b.formula_window = 8;
    gcfg_b.formula_rows_scored_max = 0;
    gcfg_b.enum_lane_visits_max = 65_536;
    let header_b = model.enum_cache_header(&gcfg_b).unwrap();
    let mut cache_b = EnumCache::new(header_b);
    model
        .build_enum_cache([&batch_b].into_iter(), &gcfg_b, &mut cache_b)
        .unwrap();
    let meta_b = build_enum_meta(&batch_b, domain_max_error, 65_536, scored_b);
    let (cand_hit_b, counters_hit_b) = cache_b
        .expand_batch(&meta_keys(&meta_b), m_b)
        .expect("all hit");
    assert_eq!(cand_hit_b, cand_dev_b, "over-cap cand differs");
    assert_eq!(counters_hit_b, counters_dev_b, "over-cap counters differ");
}

#[test]
fn generate_and_step_bit_identical() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (table, dtable, _) = setup_model(&device, &comps, 5);
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let gcfg = tiny_gen(32, 4096);
    let constants = Ms2Constants::new(&device);
    // Two identical models; one serves from a cache built over the batch.
    let mut rng_a = Rng::seeded(5);
    let mut rng_b = Rng::seeded(5);
    let mut cfg_a = tiny_model();
    cfg_a.formula_table.rows = dtable.rows as u32;
    cfg_a.formula_table.sha256 = dtable.sha256.clone();
    let mut model_a = Ms2Model::<R, E>::init(&cfg_a, &device, &mut rng_a).unwrap();
    let mut model_b = Ms2Model::<R, E>::init(&cfg_a, &device, &mut rng_b).unwrap();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    model_a.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    model_b.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    let header = model_b.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model_b
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    model_b.set_enum_cache(Some(Arc::new(cache)));
    let mut ws_a = GenerationWorkspace::new();
    let mut ws_b = GenerationWorkspace::new();
    for _ in 0..2 {
        model_a.generate(&batch, &dtable, &gcfg, &mut ws_a, &constants).unwrap();
        model_b.generate(&batch, &dtable, &gcfg, &mut ws_b, &constants).unwrap();
    }
    let out_a = model_a.generate(&batch, &dtable, &gcfg, &mut ws_a, &constants).unwrap();
    let out_b = model_b.generate(&batch, &dtable, &gcfg, &mut ws_b, &constants).unwrap();
    assert_candidate_batch_close(&out_a, &out_b);
    let packed_a = model_a
        .generate_packed(&batch, &dtable, &gcfg, &mut ws_a, &constants)
        .unwrap();
    let packed_b = model_b
        .generate_packed(&batch, &dtable, &gcfg, &mut ws_b, &constants)
        .unwrap();
    assert_packed_close(&packed_a, &packed_b);
    let resident_a = model_a
        .generate_resident(&batch, &dtable, &gcfg, &mut ws_a, &constants)
        .unwrap();
    let resident_b = model_b
        .generate_resident(&batch, &dtable, &gcfg, &mut ws_b, &constants)
        .unwrap();
    assert_packed_close(&resident_a.read(&model_a).unwrap(), &resident_b.read(&model_b).unwrap());
    // One training step with a report, bit-identical with and without cache.
    let set_comps = vec![comps[4], comps[6]];
    let set = experiment_set(&set_comps, &batch);
    let indices = vec![0usize, 1];
    let train_config = tiny_train();
    let mut trainer_a =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &train_config, &device).unwrap();
    let mut trainer_b =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &train_config, &device).unwrap();
    trainer_a.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer_b.upload_enum_artifacts(&domain, &bounds).unwrap();
    let n_raw = 64u32;
    let cache_batch = spectrum_batch_for(&set, &indices, n_raw).unwrap();
    let mut tcache = EnumCache::new(trainer_b.enum_cache_header(32).unwrap());
    trainer_b
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut tcache)
        .unwrap();
    trainer_b.set_enum_cache(Some(Arc::new(tcache)));
    trainer_a.request_report();
    trainer_b.request_report();
    let rep_a = trainer_a.step(&set, &indices).unwrap().unwrap();
    let rep_b = trainer_b.step(&set, &indices).unwrap().unwrap();
    if is_cpu() {
        assert_eq!(rep_a, rep_b, "cached training step differs (cpu demands bit-equality)");
    } else {
        assert_eq!(rep_a.step, rep_b.step);
        assert_eq!(rep_a.spectra, rep_b.spectra);
        assert_eq!(rep_a.formula_present, rep_b.formula_present);
        assert_f32_close(rep_a.loss, rep_b.loss, "loss");
        assert_f32_close(rep_a.graph, rep_b.graph, "graph");
        assert_f32_close(rep_a.formula, rep_b.formula, "formula");
    }
}

#[test]
fn cached_search_issues_zero_enum_launches_and_no_extra_reads() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, dtable, mut model) = setup_model(&device, &comps, 5);
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let gcfg = tiny_gen(32, 4096);
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    // Warm the bucket, then measure the uncached search stage.
    model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    start_launch_tally();
    reset_launch_count();
    model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    stop_launch_tally();
    let uncached_total = launch_count();
    let uncached_enum: usize = launch_tally_detailed()
        .iter()
        .filter(|row| row.site.contains("ms2_enum"))
        .map(|row| row.count)
        .sum();
    println!("uncached generate: {uncached_total} launches, {uncached_enum} of them ms2_enum");
    assert!(uncached_enum > 0, "fixture must enumerate on device");
    // Build the cache over the batch and measure the cached call.
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    model.set_enum_cache(Some(Arc::new(cache)));
    start_launch_tally();
    reset_launch_count();
    model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    stop_launch_tally();
    let cached_total = launch_count();
    let cached_enum: usize = launch_tally_detailed()
        .iter()
        .filter(|row| row.site.contains("ms2_enum"))
        .map(|row| row.count)
        .sum();
    println!("cached generate: {cached_total} launches, {cached_enum} of them ms2_enum");
    assert_eq!(cached_enum, 0, "cached search must issue zero enumeration launches");
    assert_eq!(
        cached_total,
        uncached_total - uncached_enum,
        "cached call drops exactly the enumeration launches"
    );
    // A warmed `generate` still reads exactly once (the readout).
    reset_read_count();
    model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    assert_eq!(read_count(), 1, "warmed cached generate reads once");
    // A warmed training step still reads zero times.
    let set = experiment_set(&vec![comps[4], comps[6]], &batch);
    let indices = vec![0usize, 1];
    let table = FormulaTable::from_compositions(fixture_comps().into_iter()).unwrap();
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &tiny_train(), &device).unwrap();
    let domain = EnumDomain::from_compositions(fixture_comps(), 0).unwrap();
    let bounds = RatioBounds::fit(fixture_comps(), 0).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let cache_batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let mut tcache = EnumCache::new(trainer.enum_cache_header(32).unwrap());
    trainer
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut tcache)
        .unwrap();
    trainer.set_enum_cache(Some(Arc::new(tcache)));
    for _ in 0..2 {
        trainer.step(&set, &indices).unwrap();
    }
    device.synchronize();
    reset_read_count();
    trainer.step(&set, &indices).unwrap();
    device.synchronize();
    assert_eq!(read_count(), 0, "warmed cached training step reads zero");
}

#[test]
fn partial_batch_takes_device_path_and_bump_misses() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, dtable, mut model) = setup_model(&device, &comps, 5);
    let pair = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let gcfg = tiny_gen(32, 4096);
    let constants = Ms2Constants::new(&device);
    // Cache only the first spectrum's row.
    let one = spectrum_batch(&[precursor_of(&comps[4])], &[50], &[1], &[0], 64);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&one].into_iter(), &gcfg, &mut cache)
        .unwrap();
    assert_eq!(cache.len(), 1);
    model.set_enum_cache(Some(Arc::new(cache)));
    let mut ws = GenerationWorkspace::new();
    let cached = model.generate(&pair, &dtable, &gcfg, &mut ws, &constants).unwrap();
    let (lookups, hits) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (1, 0), "partial batch is a lookup but no hit");
    model.set_enum_cache(None);
    let mut ws2 = GenerationWorkspace::new();
    let plain = model.generate(&pair, &dtable, &gcfg, &mut ws2, &constants).unwrap();
    assert_candidate_batch_close(&cached, &plain);
    // A different meta row (precursor changed by 1 unit) is a miss.
    let bumped = spectrum_batch(&[precursor_of(&comps[4]) + 1], &[50], &[1], &[0], 64);
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let meta = build_enum_meta(&bumped, artifacts.domain_max_error, 65_536, 32);
    let key = meta_keys(&meta);
    // Rebuild the one-entry cache to probe it (the model's cache is set
    // aside above; probe a fresh one identically built).
    let header2 = model.enum_cache_header(&gcfg).unwrap();
    let mut probe = EnumCache::new(header2);
    model
        .build_enum_cache([&one].into_iter(), &gcfg, &mut probe)
        .unwrap();
    assert!(probe.get(&key[0]).is_none(), "precursor + 1 must miss");
}

#[test]
fn header_mismatch_save_load_and_corrupt() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, _dtable, model) = setup_model(&device, &comps, 5);
    let gcfg = tiny_gen(32, 4096);
    let base = model.enum_cache_header(&gcfg).unwrap();
    // Every field mismatch is Error::Config naming the field.
    let mutants: Vec<(EnumCacheHeader, &str)> = vec![
        ({ let mut h = base.clone(); h.format_version += 1; h }, "`format_version`"),
        ({ let mut h = base.clone(); h.domain_sha256 = "00".to_string(); h }, "`domain_sha256`"),
        ({ let mut h = base.clone(); h.bounds_sha256 = "00".to_string(); h }, "`bounds_sha256`"),
        ({ let mut h = base.clone(); h.p += 1; h }, "`p`"),
        ({ let mut h = base.clone(); h.window_m += 1; h }, "`window_m`"),
        ({ let mut h = base.clone(); h.formula_rows_scored_max += 1; h }, "`formula_rows_scored_max`"),
        ({ let mut h = base.clone(); h.enum_lane_visits_max += 1; h }, "`enum_lane_visits_max`"),
        ({ let mut h = base.clone(); h.chemistry_version = "old".to_string(); h }, "`chemistry_version`"),
    ];
    for (mutant, field) in &mutants {
        let err = base.check_compatible(mutant).expect_err("mutant must mismatch");
        assert!(
            err.to_string().contains(field),
            "mismatch must name {field}, got {err}"
        );
    }
    // save → load → identical entries; bytes() is the file size.
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let mut cache = EnumCache::new(base.clone());
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    let path = std::env::temp_dir().join("ms2_enum_cache_test.bin");
    cache.save(&path).unwrap();
    let on_disk = std::fs::metadata(&path).unwrap().len() as usize;
    assert_eq!(cache.bytes(), on_disk, "bytes() must be the file size");
    let loaded = EnumCache::load(&path, &base).unwrap();
    assert_eq!(loaded.len(), cache.len());
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let meta = build_enum_meta(&batch, artifacts.domain_max_error, 65_536, 32);
    let keys = meta_keys(&meta);
    assert_eq!(
        loaded.expand_batch(&keys, 32),
        cache.expand_batch(&keys, 32),
        "save → load must preserve entries"
    );
    // A mismatching expected header is an error naming the field, never a
    // silent rebuild.
    let mut wrong = base.clone();
    wrong.p += 1;
    let err = EnumCache::load(&path, &wrong).expect_err("wrong header must fail");
    assert!(err.to_string().contains("`p`"), "got {err}");
    // A truncated file is an error, not a panic.
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    assert!(EnumCache::load(&path, &base).is_err(), "truncated file must error");
    // Bad magic is an error.
    let mut bad = bytes.clone();
    bad[0] ^= 0xFF;
    std::fs::write(&path, &bad).unwrap();
    assert!(EnumCache::load(&path, &base).is_err(), "bad magic must error");
    // A damaged header length is an error.
    let mut damaged = bytes.clone();
    damaged[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&path, &damaged).unwrap();
    assert!(EnumCache::load(&path, &base).is_err(), "damaged length must error");
    std::fs::remove_file(&path).ok();
}

#[test]
fn compact_encoding_edges() {
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        4,
        4096,
        65_536,
    );
    let mut cache = EnumCache::new(header);
    // A candidate with a heavy count of 255 and hydrogen 1023 survives.
    let mut counts = [0u16; 10];
    counts[0] = 255;
    counts[1] = 1023;
    let mass = composition_mass(&counts).unwrap();
    let mut row = [0u32; 13];
    row[0] = 255;
    row[1] = 1023;
    row[10] = mass;
    row[11] = 1;
    row[12] = u32::MAX;
    cache.insert([7u32; 8], [5, 3, 1, 0, 1], &row, 4).unwrap();
    let (cand, counters) = cache.expand_batch(&[[7u32; 8]], 4).expect("hit");
    assert_eq!(&cand[0..13], &row, "255/1023 candidate must survive");
    assert_eq!(&counters, &[5, 3, 1, 0, 1]);
    // Padding slots are the fixed padding row.
    assert_eq!(&cand[13..26], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u32::MAX]);
    // An out-of-range hydrogen count is refused at insert.
    let mut bad_h = row;
    bad_h[1] = 1024;
    assert!(
        cache.insert([8u32; 8], [0, 0, 1, 0, 0], &bad_h, 4).is_err(),
        "hydrogen 1024 must be refused"
    );
    // An out-of-range heavy count is refused at insert.
    let mut bad_c = row;
    bad_c[0] = 256;
    assert!(
        cache.insert([9u32; 8], [0, 0, 1, 0, 0], &bad_c, 4).is_err(),
        "heavy count 256 must be refused"
    );
    // A wrong mass is refused at insert.
    let mut bad_m = row;
    bad_m[10] += 1;
    assert!(
        cache.insert([10u32; 8], [0, 0, 1, 0, 0], &bad_m, 4).is_err(),
        "wrong mass must be refused"
    );
    assert!(cache.get(&[8u32; 8]).is_none());
}

#[test]
fn refusal_applies_cached() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, dtable, mut model) = setup_model(&device, &comps, 5);
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    // A lane budget the request exceeds, refused uncached ...
    let mut tight = tiny_gen(32, 4096);
    tight.enum_lanes_max = 1;
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let err = model
        .generate(&batch, &dtable, &tight, &mut ws, &constants)
        .expect_err("too many lanes must be refused uncached");
    assert!(err.to_string().contains("enum_lanes_max"), "got {err}");
    // ... and refused cached (here over a populated cache).
    let gcfg = tiny_gen(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    model.set_enum_cache(Some(Arc::new(cache)));
    let mut ws2 = GenerationWorkspace::new();
    let err = model
        .generate(&batch, &dtable, &tight, &mut ws2, &constants)
        .expect_err("too many lanes must be refused cached");
    assert!(err.to_string().contains("enum_lanes_max"), "got {err}");
}

#[test]
fn jitter_variants_fixed_pool_and_hits() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let set_comps = vec![comps[4], comps[6]];
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let set = experiment_set(&set_comps, &batch);
    let indices = vec![0usize, 1];
    let mut train_config = tiny_train();
    train_config.precursor_jitter_ppm = 2.0;
    train_config.precursor_jitter_variants = 3;
    train_config.seed = 9;
    // The three fixed draws per spectrum (split_tag = 1 + v).
    let mut variants: Vec<SpectrumBatch> = Vec::new();
    let base = spectrum_batch_for(&set, &indices, 64).unwrap();
    for v in 0..3 {
        let mut jb = base.clone();
        apply_precursor_jitter(&mut jb, &indices, 2.0, 9, 1 + v as u64);
        variants.push(jb);
    }
    let draws: Vec<Vec<u32>> = variants.iter().map(|b| b.precursor_mz_udalton.clone()).collect();
    assert!(
        draws[0] != draws[1] || draws[1] != draws[2],
        "the fixed draws must vary across variants"
    );
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_model(), &FormulaTable::from_compositions(set_comps.iter().copied()).unwrap(), &train_config, &device).unwrap();
    trainer.capture_prep_batch(true);
    let domain = EnumDomain::from_compositions(fixture_comps(), 0).unwrap();
    let bounds = RatioBounds::fit(fixture_comps(), 0).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let mut cache = EnumCache::new(trainer.enum_cache_header(32).unwrap());
    trainer
        .build_enum_cache(variants.iter(), 32, &mut cache)
        .unwrap();
    assert_eq!(cache.len(), 6, "all 3 variants of both spectra cached");
    trainer.set_enum_cache(Some(Arc::new(cache)));
    // Six steps: every precursor comes from the fixed pool, and every
    // training batch is a cache hit.
    let mut seen: Vec<Vec<u32>> = Vec::new();
    for _ in 0..6 {
        trainer.step(&set, &indices).unwrap();
        let prep = trainer.last_prep_batch.as_ref().unwrap();
        seen.push(prep.precursor_mz_udalton.clone());
        for (b, &idx) in indices.iter().enumerate() {
            let allowed: Vec<u32> = (0..3)
                .map(|v| {
                    jitter_precursor_mz(
                        base.precursor_mz_udalton[b],
                        2.0,
                        9,
                        1 + v as u64,
                        idx as u64,
                    )
                })
                .collect();
            assert!(
                allowed.contains(&prep.precursor_mz_udalton[b]),
                "step precursor {} not in the 3 fixed draws {allowed:?}",
                prep.precursor_mz_udalton[b]
            );
        }
    }
    let (lookups, hits) = trainer.enum_cache_stats();
    assert_eq!((lookups, hits), (6, 6), "all three variants are cache hits");
    // Deterministic: a second trainer draws the same sequence.
    let mut trainer2 =
        Ms2Trainer::<R, E>::new(&tiny_model(), &FormulaTable::from_compositions(set_comps.iter().copied()).unwrap(), &train_config, &device).unwrap();
    trainer2.capture_prep_batch(true);
    trainer2.upload_enum_artifacts(&domain, &bounds).unwrap();
    let mut seen2: Vec<Vec<u32>> = Vec::new();
    for _ in 0..6 {
        trainer2.step(&set, &indices).unwrap();
        seen2.push(trainer2.last_prep_batch.as_ref().unwrap().precursor_mz_udalton.clone());
    }
    assert_eq!(seen, seen2, "V = 3 uses only the 3 draws, deterministically");
}
