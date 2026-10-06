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
    AllocationMode, CandidateBatch, Control, FormulaFeatures, FormulaSource, GenerationConfig,
    GenerationMode, IdentityMode, ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION,
    SpectrumBatch, request_status,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::enum_cache::{
    ENUM_CACHE_FORMAT_VERSION, EnumCache, EnumCacheHeader, EvidenceInputs, EvidenceKey,
    EvidenceQuery, SavePhase, evidence_canonical_bytes, evidence_inputs_for_batch,
    evidence_key_for_batch, evidence_key_with_forced_hash, run_device_evidence_into,
    set_save_phase_hook,
};
use mamba3::models::ms2::formula_evidence::EVIDENCE_PEAKS;
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, apply_precursor_jitter,
    jitter_variant_index, jitter_variants_of_batch, jittered_set_for_eval, spectrum_batch_for,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{
    EnumDomain, RatioBounds, build_enum_meta, enum_lanes_per_dispatch,
};
use mamba3::models::ms2::formula_evidence_ref::jitter_precursor_mz;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::pack::PackedCandidateBatch;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::{IdTensor, read_all};
use mamba3::tensor::ops::ms2::{Ms2Constants, PeakBuffers, peak_select};
use mamba3::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};
use mamba3::tensor::ops::ms2_formula_evidence::{evidence_peaks, formula_evidence};
use mamba3::tensor::ops::random::Rng;

/// File-level serialisation: the process-wide launch/read counters are
/// perturbed by any test running beside these, so every test holds this
/// mutex.
static SERIAL: Mutex<()> = Mutex::new(());

/// Host-allocation probe for the task F9 laziness test (A3): a counting
/// [`#[global_allocator]`](std::alloc::GlobalAlloc) in THIS test binary only.
/// Tracks live bytes and their peak inside the window, so the test can prove
/// at most one jitter variant is alive at a time.
static ALLOC_LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ALLOC_PEAK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ALLOC_COUNTING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if ALLOC_COUNTING.load(std::sync::atomic::Ordering::Relaxed) == 1 {
            let live = ALLOC_LIVE.fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed)
                + layout.size();
            ALLOC_PEAK.fetch_max(live, std::sync::atomic::Ordering::Relaxed);
        }
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        if ALLOC_COUNTING.load(std::sync::atomic::Ordering::Relaxed) == 1 {
            // Saturating: ambient frees (allocated before the window, never
            // counted) must not wrap the live total.
            let _ = ALLOC_LIVE.fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |live| live.checked_sub(layout.size()),
            );
        }
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

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
    setup_model_with_layout(device, comps, seed, FormulaFeatures::Counts)
}

/// [`setup_model`] with an explicit formula-features layout (task T6B: the
/// evidence tests run the `Evidence` layout).
fn setup_model_with_layout(
    device: &Device<R>,
    comps: &[Composition],
    seed: u64,
    layout: FormulaFeatures,
) -> (FormulaTable, DeviceFormulaTable<R, E>, Ms2Model<R, E>) {
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, device).unwrap();
    let domain = EnumDomain::from_compositions(comps.to_vec(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.to_vec(), 0).unwrap();
    let mut cfg = tiny_model();
    cfg.formula_features = layout;
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
/// `counters` back (the uncached device result). `dispatch` is the dispatch
/// bound: results never depend on it (the lane takes the absolute lane
/// index), only the launch count does.
fn device_enumerate(
    device: &Device<R>,
    model: &Ms2Model<R, E>,
    batch: &SpectrumBatch,
    scored_cap: u32,
    m: usize,
    lane_visits_max: u32,
    dispatch: u32,
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
            dispatch,
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
            dispatch,
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
        device_enumerate(&device, &model, &batch_a, scored_a, m_a, 65_536, 4_000_000);
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
        device_enumerate(&device, &model, &batch_b, scored_b, m_b, 65_536, 4_000_000);
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
    model_b.set_enum_cache(Some(Arc::new(cache))).unwrap();
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
    trainer_b.set_enum_cache(Some(Arc::new(tcache))).unwrap();
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
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
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
    trainer.set_enum_cache(Some(Arc::new(tcache))).unwrap();
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
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let mut ws = GenerationWorkspace::new();
    let cached = model.generate(&pair, &dtable, &gcfg, &mut ws, &constants).unwrap();
    let (lookups, hits, ev_lookups, ev_hits) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (1, 0), "partial batch is a lookup but no hit");
    assert_eq!((ev_lookups, ev_hits), (0, 0), "Counts layout counts no evidence lookup");
    model.set_enum_cache(None).unwrap();
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
        16,
        "f32",
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
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
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
    trainer.set_enum_cache(Some(Arc::new(cache))).unwrap();
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
    let (lookups, hits, ev_lookups, ev_hits) = trainer.enum_cache_stats();
    assert_eq!((lookups, hits), (6, 6), "all three variants are cache hits");
    assert_eq!((ev_lookups, ev_hits), (0, 0), "Counts layout counts no evidence lookup");
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

// ---------------------------------------------------------------------------
// Task T6B: memoised formula evidence.
// ---------------------------------------------------------------------------

/// Classify `ms2_formula_evidence.rs` launches from the detailed tally by the
/// launch-site line: `evidence_peaks` launches from its wrapper (which starts
/// at line 371), `formula_evidence` from its wrapper (starts at line 1063),
/// `formula_features` from its wrapper (starts at line 1388). Kernel code is
/// frozen by this task, so these boundaries are stable.
fn evidence_tally() -> (usize, usize, usize) {
    let mut peaks = 0usize;
    let mut ev = 0usize;
    let mut feat = 0usize;
    for row in launch_tally_detailed() {
        if !row.site.contains("ms2_formula_evidence") {
            continue;
        }
        let line: usize = row
            .site
            .rsplit(':')
            .next()
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
        if line < 1063 {
            peaks += row.count;
        } else if line < 1388 {
            ev += row.count;
        } else {
            feat += row.count;
        }
    }
    (peaks, ev, feat)
}

/// Generation config for the T6B evidence tests: `Enumerate` source with an
/// explicit dispatch bound (results never depend on it) and the `Evidence`
/// walk defaults.
fn tiny_gen_evidence(window: u32, rows_scored_max: u32) -> GenerationConfig {
    let mut g = tiny_gen(window, rows_scored_max);
    g.formula_evidence_work_max = 2048;
    g.formula_evidence_dispatch_max = 268435456;
    g
}

/// Request batch with per-spectrum peak data for the evidence fixtures:
/// `mzs[i]` / `intensities[i]` are the uploaded rows (each `n_raw` wide, of
/// which the first `counts[i]` are valid), `uncs[i]` the m/z uncertainty.
#[allow(clippy::too_many_arguments)]
fn evidence_batch(
    precursors: &[u32],
    prec_uncs: &[u32],
    adducts: &[u16],
    frag_tenths: &[u16],
    prec_tenths: &[u16],
    uncs: &[u32],
    counts: &[u32],
    mzs: &[Vec<u32>],
    intensities: &[Vec<f32>],
    n_raw: usize,
) -> SpectrumBatch {
    let b = precursors.len();
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for bi in 0..b {
        let count = counts[bi] as usize;
        peak_count[bi] = counts[bi];
        raw_peak_count[bi] = counts[bi];
        for k in 0..count {
            peak_id[bi * n_raw + k] = k as u32;
            mz[bi * n_raw + k] = mzs[bi][k];
            intensity[bi * n_raw + k] = intensities[bi][k];
        }
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: (1..=b as u64).collect(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: uncs.to_vec(),
        precursor_mz_udalton: precursors.to_vec(),
        precursor_uncertainty_udalton: prec_uncs.to_vec(),
        adduct: adducts.to_vec(),
        polarity: vec![1; b],
        collision_energy_ev: vec![30.0; b],
        collision_energy_known: vec![1; b],
        energy_count: vec![1; b],
        fragment_tolerance_ppm_tenths: frag_tenths.to_vec(),
        precursor_tolerance_ppm_tenths: prec_tenths.to_vec(),
        instrument_class: vec![0; b],
    }
}

/// Run the PRODUCTION evidence pipeline (peak selection, enumeration,
/// `evidence_peaks`, `formula_evidence`) for one batch and read `cand_ev`,
/// `counters` and `ev_peaks` back: the uncached device reference the cache
/// entries must equal element for element.
#[allow(clippy::too_many_arguments)]
fn device_evidence(
    device: &Device<R>,
    model: &Ms2Model<R, E>,
    batch: &SpectrumBatch,
    scored_cap: u32,
    m: usize,
    lane_visits_max: u32,
    work_max: u32,
    dispatch_max: u64,
    kept: usize,
) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    let b = batch.len();
    let n_raw = batch.n_raw as usize;
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let spectra = DeviceSpectra::upload(batch, device).unwrap();
    let peaks = PeakBuffers::<R, E>::new(b, n_raw, kept, device);
    peak_select(
        &spectra.mz,
        &spectra.intensity,
        &spectra.meta,
        spectra.intensity_scale,
        &peaks,
    )
    .unwrap();
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
    let spec_t = spectra.evidence_spec(device).unwrap();
    let mut ev_peaks_t = IdTensor::empty(vec![b, EVIDENCE_PEAKS, 4], device);
    let mut ev_w_t = Tensor::<R, E>::empty(vec![b, EVIDENCE_PEAKS], device);
    evidence_peaks(
        &peaks.kept,
        &peaks.kept_f,
        &spectra.meta,
        &spec_t,
        &mut ev_peaks_t,
        &mut ev_w_t,
    )
    .unwrap();
    let mut cand_ev_t = Tensor::<R, E>::empty(vec![b, m, 4], device);
    formula_evidence(
        &cand_t,
        &ev_peaks_t,
        &ev_w_t,
        &spectra.meta,
        &spec_t,
        &mut cand_ev_t,
        work_max,
        dispatch_max,
        artifacts.hydrogen_cap_max(),
        spectra.uploaded_tol_max(),
    )
    .unwrap();
    let (ids, floats) = read_all(&[&counters_t, &ev_peaks_t], &[&cand_ev_t]).unwrap();
    (floats[0].clone(), ids[0].clone(), ids[1].clone())
}

/// Evidence fixture spectra over the standard fixture domain: a productive
/// row with explained and unexplained peaks, a row with unknown m/z
/// uncertainty (no evidence peaks), and a far-precursor row that joins
/// nothing. Peaks sit below the productive precursor (the peak filter keeps
/// `mz <= precursor + 2 Da`) and span a wide m/z range so the walk has both
/// explained and unexplained peaks.
fn evidence_fixture_batch(comps: &[Composition], n_raw: usize) -> SpectrumBatch {
    // comps[6] = [6, 12, 0, 6]: ~180 Da, so the 60..172 Da peaks survive the
    // precursor filter and the enumerated candidates can explain the small
    // ones while the large ones stay unexplained.
    let productive = precursor_of(&comps[6]);
    let mut mzs = Vec::new();
    let mut intensities = Vec::new();
    for _ in 0..3 {
        let mut mz_row = Vec::with_capacity(n_raw);
        let mut int_row = Vec::with_capacity(n_raw);
        for k in 0..8 {
            mz_row.push(60_000_000 + (k as u32) * 16_000_000);
            int_row.push(1.0 - 0.09 * k as f32);
        }
        mzs.push(mz_row);
        intensities.push(int_row);
    }
    evidence_batch(
        &[productive, productive, 4_000_000_000],
        &[50, 50, 50],
        &[1, 1, 1],
        &[0, 0, 0],
        &[0, 0, 0],
        &[50, u32::MAX, 50],
        &[8, 8, 8],
        &mzs,
        &intensities,
        n_raw,
    )
}

/// Verified evidence queries for a whole batch (task F8 item 3): the hash
/// key, the scored count and the borrowed canonical inputs, so expansion
/// verifies every hash hit in full.
fn evidence_queries_for<'b>(
    batch: &'b SpectrumBatch,
    meta_host: &[u32],
    uploaded_counts: &[u32],
    rows: &[usize],
    work_max: u32,
    h_cap_max: u32,
) -> Vec<EvidenceQuery<'b>> {
    let keys8 = meta_keys(meta_host);
    assert_eq!(keys8.len(), batch.len());
    assert_eq!(rows.len(), batch.len());
    keys8
        .iter()
        .enumerate()
        .map(|(i, key8)| {
            let (key, _) =
                evidence_key_for_batch(batch, i, uploaded_counts[i], *key8, work_max, h_cap_max);
            EvidenceQuery {
                key,
                rows_scored: rows[i],
                inputs: evidence_inputs_for_batch(
                    batch,
                    i,
                    uploaded_counts[i],
                    work_max,
                    h_cap_max,
                ),
            }
        })
        .collect()
}

/// Verified single-row evidence lookup (task F8 item 3): `None` on a hash
/// miss AND on a canonical mismatch despite a hash hit.
fn get_evidence_for<'s, 'b>(
    cache: &'s EnumCache,
    batch: &'b SpectrumBatch,
    i: usize,
    uploaded_peak_count: u32,
    meta_row: [u32; 8],
    work_max: u32,
    h_cap_max: u32,
) -> Option<mamba3::models::ms2::enum_cache::EvidenceRef<'s>> {
    let (key, _) =
        evidence_key_for_batch(batch, i, uploaded_peak_count, meta_row, work_max, h_cap_max);
    let inputs =
        evidence_inputs_for_batch(batch, i, uploaded_peak_count, work_max, h_cap_max);
    cache.get_evidence(&key, &inputs)
}

/// The suite's kept-peak capacity (`tiny_model().n_peaks`) and float dtype:
/// every model-derived header in these tests carries them.
const TEST_N_PEAKS: u32 = 16;
/// The suite's float dtype name (`E = f32`).
const TEST_DTYPE: &str = "f32";

#[test]
fn evidence_round_trip_element_for_element() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, _dtable, model) = setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let domain_max_error = artifacts.domain_max_error;
    let h_cap_max = artifacts.hydrogen_cap_max();
    let batch = evidence_fixture_batch(&comps, 64);
    let m = 32usize;
    let scored_cap = 32u32;
    let work_max = 2048u32;
    // Uncached device reference from the production pipeline.
    let (ev_dev, counters_dev, ev_peaks_dev) =
        device_evidence(&device, &model, &batch, scored_cap, m, 65_536, work_max, 268435456, 16);
    // Row 1 (unknown m/z uncertainty) selects no evidence peak.
    let stride = EVIDENCE_PEAKS * 4;
    let mut n_valid_1 = 0;
    for s in 0..EVIDENCE_PEAKS {
        if ev_peaks_dev[stride + s * 4 + 3] == 1 {
            n_valid_1 += 1;
        }
    }
    assert_eq!(n_valid_1, 0, "unknown-uncertainty row selects no evidence peak");
    // Row 0 is productive: some scored slot explains a peak.
    let scored0 = counters_dev[2] as usize;
    assert!(scored0 > 0, "row 0 must score candidates");
    let mut explained_some = false;
    for r in 0..scored0 {
        if ev_dev[r * 4] > 0.0 {
            explained_some = true;
        }
    }
    assert!(explained_some, "row 0 must explain a peak somewhere");
    // Build the cache through the production build API and expand it.
    let gcfg = tiny_gen_evidence(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    // Rows 0 and 1 share their enumeration meta row (same precursor and
    // tolerances; the m/z uncertainty is not part of it) but differ in
    // evidence content: 2 enum entries, 3 evidence entries.
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.evidence_len(), 3);
    let meta = build_enum_meta(&batch, domain_max_error, 65_536, scored_cap);
    // Uploaded peak counts: row 2 is fatal (precursor out of range)
    // and uploads with peak_count 0.
    let uploaded = vec![8u32, 8, 0];
    let rows: Vec<usize> = (0..3).map(|i| counters_dev[i * 5 + 2] as usize).collect();
    let queries = evidence_queries_for(&batch, &meta, &uploaded, &rows, work_max, h_cap_max);
    let ev_hit = cache
        .expand_evidence_batch(&queries, m, TEST_N_PEAKS, TEST_DTYPE)
        .expect("all evidence hit");
    assert_eq!(ev_hit.len(), ev_dev.len());
    for (i, (&got, &want)) in ev_hit.iter().zip(ev_dev.iter()).enumerate() {
        // Column 1 is the explained weight (float); the rest are integers.
        if i % 4 == 1 {
            if is_cpu() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "weight bit differs at flat {i}: {got} vs {want}"
                );
            } else {
                assert_f32_close(got, want, &format!("weight[{i}]"));
            }
        } else {
            assert_eq!(got, want, "integer word differs at flat {i}: {got} vs {want}");
        }
    }
    // A padding-heavy window (scored cap 1): padding slots regenerate zeros.
    let mut gcfg_pad = tiny_gen_evidence(32, 4096);
    gcfg_pad.formula_rows_scored_max = 1;
    let header_pad = model.enum_cache_header(&gcfg_pad).unwrap();
    let mut cache_pad = EnumCache::new(header_pad);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg_pad, &mut cache_pad)
        .unwrap();
    let meta_pad = build_enum_meta(&batch, domain_max_error, 65_536, 1);
    let (ev_dev_pad, counters_pad, _) =
        device_evidence(&device, &model, &batch, 1, m, 65_536, work_max, 268435456, 16);
    let rows_pad: Vec<usize> = (0..3).map(|i| counters_pad[i * 5 + 2] as usize).collect();
    assert!(
        rows_pad.iter().all(|&r| r <= 1),
        "scored cap 1 bounds rows_scored: {rows_pad:?}"
    );
    let queries_pad = evidence_queries_for(&batch, &meta_pad, &uploaded, &rows_pad, work_max, h_cap_max);
    let ev_hit_pad = cache_pad
        .expand_evidence_batch(&queries_pad, m, TEST_N_PEAKS, TEST_DTYPE)
        .expect("all evidence hit");
    assert_eq!(ev_hit_pad, ev_dev_pad, "padding-heavy cand_ev differs");
    // An incomplete walk (work_max 1 with real heteroatom counts): the
    // complete flag is 0 somewhere and still round-trips exactly.
    let mut gcfg_w1 = tiny_gen_evidence(32, 4096);
    gcfg_w1.formula_evidence_work_max = 1;
    let header_w1 = model.enum_cache_header(&gcfg_w1).unwrap();
    let mut cache_w1 = EnumCache::new(header_w1);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg_w1, &mut cache_w1)
        .unwrap();
    let queries_w1 = evidence_queries_for(&batch, &meta, &uploaded, &rows, 1, h_cap_max);
    let (ev_dev_w1, _, _) =
        device_evidence(&device, &model, &batch, scored_cap, m, 65_536, 1, 268435456, 16);
    let ev_hit_w1 = cache_w1
        .expand_evidence_batch(&queries_w1, m, TEST_N_PEAKS, TEST_DTYPE)
        .expect("all evidence hit");
    assert_eq!(ev_hit_w1, ev_dev_w1, "work_max-1 cand_ev differs");
    let mut incomplete_some = false;
    for (r, chunk) in ev_hit_w1.chunks_exact(4 * m).enumerate() {
        let scored = rows[r];
        for s in 0..scored {
            if chunk[s * 4 + 3] == 0.0 {
                incomplete_some = true;
            }
        }
    }
    assert!(incomplete_some, "work_max 1 must leave a lane incomplete");
}

#[test]
fn evidence_generate_and_step_bit_identical() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (table, dtable, _) = setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let batch = evidence_fixture_batch(&comps, 64);
    let gcfg = tiny_gen_evidence(32, 4096);
    let constants = Ms2Constants::new(&device);
    // Three models: uncached, enum-only cached (built under Counts so no
    // evidence entries exist — the header is layout-independent), fully
    // cached.
    let mut rng_a = Rng::seeded(5);
    let mut rng_b = Rng::seeded(5);
    let mut rng_c = Rng::seeded(5);
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut cfg_ev = tiny_model();
    cfg_ev.formula_features = FormulaFeatures::Evidence;
    cfg_ev.formula_table.rows = dtable.rows as u32;
    cfg_ev.formula_table.sha256 = dtable.sha256.clone();
    let mut model_plain = Ms2Model::<R, E>::init(&cfg_ev, &device, &mut rng_a).unwrap();
    let mut model_enum = Ms2Model::<R, E>::init(&cfg_ev, &device, &mut rng_b).unwrap();
    let mut model_full = Ms2Model::<R, E>::init(&cfg_ev, &device, &mut rng_c).unwrap();
    for m in [&mut model_plain, &mut model_enum, &mut model_full] {
        m.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    }
    // Enum-only cache: built by a Counts-layout model over the same batch.
    let mut cfg_counts = tiny_model();
    cfg_counts.formula_table.rows = dtable.rows as u32;
    cfg_counts.formula_table.sha256 = dtable.sha256.clone();
    let mut rng_d = Rng::seeded(5);
    let mut model_counts = Ms2Model::<R, E>::init(&cfg_counts, &device, &mut rng_d).unwrap();
    model_counts
        .upload_enum_artifacts(&domain, &bounds, &device)
        .unwrap();
    let header = model_enum.enum_cache_header(&gcfg).unwrap();
    let mut cache_enum = EnumCache::new(header);
    model_counts
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache_enum)
        .unwrap();
    assert_eq!(cache_enum.evidence_len(), 0, "Counts build stores no evidence");
    model_enum.set_enum_cache(Some(Arc::new(cache_enum))).unwrap();
    // Full cache: built by the Evidence model itself.
    let header_full = model_full.enum_cache_header(&gcfg).unwrap();
    let mut cache_full = EnumCache::new(header_full);
    model_full
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache_full)
        .unwrap();
    assert_eq!(cache_full.evidence_len(), 3);
    model_full.set_enum_cache(Some(Arc::new(cache_full))).unwrap();
    // The three readout modes agree across cache states (bit-identical on
    // cpu, 1e-6 on GPU floats).
    let mut ws_plain = GenerationWorkspace::new();
    let mut ws_enum = GenerationWorkspace::new();
    let mut ws_full = GenerationWorkspace::new();
    for _ in 0..2 {
        model_plain.generate(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
        model_enum.generate(&batch, &dtable, &gcfg, &mut ws_enum, &constants).unwrap();
        model_full.generate(&batch, &dtable, &gcfg, &mut ws_full, &constants).unwrap();
    }
    let out_plain = model_plain.generate(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
    let out_enum = model_enum.generate(&batch, &dtable, &gcfg, &mut ws_enum, &constants).unwrap();
    let out_full = model_full.generate(&batch, &dtable, &gcfg, &mut ws_full, &constants).unwrap();
    assert_candidate_batch_close(&out_plain, &out_enum);
    assert_candidate_batch_close(&out_plain, &out_full);
    let packed_plain = model_plain.generate_packed(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
    let packed_enum = model_enum.generate_packed(&batch, &dtable, &gcfg, &mut ws_enum, &constants).unwrap();
    let packed_full = model_full.generate_packed(&batch, &dtable, &gcfg, &mut ws_full, &constants).unwrap();
    assert_packed_close(&packed_plain, &packed_enum);
    assert_packed_close(&packed_plain, &packed_full);
    let resident_plain = model_plain.generate_resident(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
    let resident_enum = model_enum.generate_resident(&batch, &dtable, &gcfg, &mut ws_enum, &constants).unwrap();
    let resident_full = model_full.generate_resident(&batch, &dtable, &gcfg, &mut ws_full, &constants).unwrap();
    assert_packed_close(&resident_plain.read(&model_plain).unwrap(), &resident_enum.read(&model_enum).unwrap());
    assert_packed_close(&resident_plain.read(&model_plain).unwrap(), &resident_full.read(&model_full).unwrap());
    // Search-stage launch counts (uncached / enumeration cached / both
    // cached): the evidence kernels run on the first two, neither on the
    // third; `formula_features` runs on all three.
    for (name, model, ws) in [
        ("uncached", &model_plain, &mut ws_plain),
        ("enum-cached", &model_enum, &mut ws_enum),
        ("both-cached", &model_full, &mut ws_full),
    ] {
        device.synchronize();
        start_launch_tally();
        reset_launch_count();
        model.generate(&batch, &dtable, &gcfg, ws, &constants).unwrap();
        device.synchronize();
        stop_launch_tally();
        let (peaks, ev, feat) = evidence_tally();
        let n_enum: usize = launch_tally_detailed()
            .iter()
            .filter(|row| row.site.contains("ms2_enum"))
            .map(|row| row.count)
            .sum();
        println!("evidence search {name}: enum={n_enum} evidence_peaks={peaks} formula_evidence={ev} formula_features={feat}");
        match name {
            "uncached" => {
                assert!(n_enum > 0, "uncached generate must enumerate on device");
                assert_eq!(peaks, 1, "uncached generate launches evidence_peaks once");
                assert!(ev > 0, "uncached generate must run formula_evidence");
                assert!(feat > 0, "formula_features runs uncached");
            }
            "enum-cached" => {
                assert_eq!(n_enum, 0, "enum-cached generate launches no enumeration");
                assert_eq!(peaks, 1, "enum-cached generate still runs evidence_peaks");
                assert!(ev > 0, "enum-cached generate still runs formula_evidence");
            }
            _ => {
                assert_eq!(n_enum, 0, "full hit launches no enumeration");
                assert_eq!(peaks, 0, "full hit launches no evidence_peaks");
                assert_eq!(ev, 0, "full hit launches no formula_evidence");
                assert!(feat > 0, "formula_features still runs on a full hit");
            }
        }
    }
    // A warmed `generate` still reads exactly once in every cache state.
    for (model, ws) in [
        (&model_plain, &mut ws_plain),
        (&model_enum, &mut ws_enum),
        (&model_full, &mut ws_full),
    ] {
        reset_read_count();
        model.generate(&batch, &dtable, &gcfg, ws, &constants).unwrap();
        device.synchronize();
        assert_eq!(read_count(), 1, "warmed generate reads once");
    }
    // One training step under Evidence: bit-identical across cache states,
    // zero reads per warmed step.
    let set_comps = vec![comps[4], comps[4], comps[4]];
    let set_batch = evidence_fixture_batch(&comps, 64);
    let set = experiment_set(&set_comps, &set_batch);
    let indices = vec![0usize, 1, 2];
    let mut train_config = tiny_train();
    train_config.formula_window = 32;
    let mut ev_cfg = tiny_model();
    ev_cfg.formula_features = FormulaFeatures::Evidence;
    let mut trainer_plain =
        Ms2Trainer::<R, E>::new(&ev_cfg, &table, &train_config, &device).unwrap();
    let mut trainer_full =
        Ms2Trainer::<R, E>::new(&ev_cfg, &table, &train_config, &device).unwrap();
    trainer_plain.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer_full.upload_enum_artifacts(&domain, &bounds).unwrap();
    let cache_batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let mut tcache = EnumCache::new(trainer_full.enum_cache_header(32).unwrap());
    trainer_full
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut tcache)
        .unwrap();
    // Rows 0 and 1 of the trainer batch are content-identical (the set
    // helper stores one m/z uncertainty for all rows), so they share one
    // evidence entry; the fatal row has its own.
    assert_eq!(tcache.evidence_len(), 2, "trainer build stores evidence");
    trainer_full.set_enum_cache(Some(Arc::new(tcache))).unwrap();
    trainer_plain.request_report();
    trainer_full.request_report();
    let rep_plain = trainer_plain.step(&set, &indices).unwrap().unwrap();
    let rep_full = trainer_full.step(&set, &indices).unwrap().unwrap();
    if is_cpu() {
        assert_eq!(rep_plain, rep_full, "cached training step differs (cpu demands bit-equality)");
    } else {
        assert_eq!(rep_plain.step, rep_full.step);
        assert_f32_close(rep_plain.loss, rep_full.loss, "loss");
    }
    for _ in 0..2 {
        trainer_plain.step(&set, &indices).unwrap();
        trainer_full.step(&set, &indices).unwrap();
    }
    device.synchronize();
    reset_read_count();
    trainer_plain.step(&set, &indices).unwrap();
    device.synchronize();
    assert_eq!(read_count(), 0, "warmed plain training step reads zero");
    reset_read_count();
    trainer_full.step(&set, &indices).unwrap();
    device.synchronize();
    assert_eq!(read_count(), 0, "warmed cached training step reads zero");
    let (_, _, ev_lookups, ev_hits) = trainer_full.enum_cache_stats();
    assert!(ev_lookups > 0 && ev_hits > 0, "training steps count evidence lookups/hits");
}

#[test]
fn evidence_key_sensitivity() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, _dtable, model) = setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let h_cap_max = artifacts.hydrogen_cap_max();
    let batch = evidence_fixture_batch(&comps, 64);
    let gcfg = tiny_gen_evidence(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    let meta = build_enum_meta(&batch, artifacts.domain_max_error, 65_536, 32);
    // Uploaded peak counts: row 2 is fatal (precursor out of range) and
    // uploads with peak_count 0.
    let uploaded = vec![8u32, 8, 0];
    let keys8 = meta_keys(&meta);
    let (_, counters) = cache.expand_batch(&keys8, 32).expect("enum hit");
    let rows: Vec<usize> = (0..3).map(|i| counters[i * 5 + 2] as usize).collect();
    // The same row uploaded twice is a hit.
    let keys8 = meta_keys(&meta);
    for i in 0..3 {
        let got = get_evidence_for(&cache, &batch, i, uploaded[i], keys8[i], 2048, h_cap_max)
            .expect("same row must hit");
        assert_eq!(got.peak_count, uploaded[i], "same row must hit");
    }
    let queries = evidence_queries_for(&batch, &meta, &uploaded, &rows, 2048, h_cap_max);
    assert!(
        cache.expand_evidence_batch(&queries, 32, TEST_N_PEAKS, TEST_DTYPE).is_some(),
        "same batch must expand"
    );
    // One uploaded peak's m/z shifted by 1 unit is a miss.
    let mut bumped_mz = batch.clone();
    bumped_mz.mz_udalton[3] += 1;
    assert!(
        get_evidence_for(&cache, &bumped_mz, 0, uploaded[0], keys8[0], 2048, h_cap_max).is_none(),
        "m/z + 1 must miss"
    );
    let queries_b = evidence_queries_for(&bumped_mz, &meta, &uploaded, &rows, 2048, h_cap_max);
    assert!(
        cache.expand_evidence_batch(&queries_b, 32, TEST_N_PEAKS, TEST_DTYPE).is_none(),
        "m/z + 1 batch must miss"
    );
    // One intensity bit flipped is a miss.
    let mut bumped_int = batch.clone();
    let bits = bumped_int.intensity[5].to_bits() ^ 1;
    bumped_int.intensity[5] = f32::from_bits(bits);
    assert!(
        get_evidence_for(&cache, &bumped_int, 0, uploaded[0], keys8[0], 2048, h_cap_max).is_none(),
        "intensity 1-bit must miss"
    );
    // A changed fragment ppm is a miss (stored 0 resolves to 100; 200 differs).
    let mut bumped_ppm = batch.clone();
    bumped_ppm.fragment_tolerance_ppm_tenths[0] = 200;
    assert!(
        get_evidence_for(&cache, &bumped_ppm, 0, uploaded[0], keys8[0], 2048, h_cap_max).is_none(),
        "fragment ppm change must miss"
    );
    // A changed m/z uncertainty is a miss.
    let mut bumped_u = batch.clone();
    bumped_u.mz_uncertainty_udalton[0] = 51;
    assert!(
        get_evidence_for(&cache, &bumped_u, 0, uploaded[0], keys8[0], 2048, h_cap_max).is_none(),
        "m/z uncertainty change must miss"
    );
    // A changed work budget is a miss.
    let queries_w = evidence_queries_for(&batch, &meta, &uploaded, &rows, 1024, h_cap_max);
    assert!(
        cache.expand_evidence_batch(&queries_w, 32, TEST_N_PEAKS, TEST_DTYPE).is_none(),
        "work_max change must miss"
    );
    // A changed adduct is a miss.
    let mut bumped_ad = batch.clone();
    bumped_ad.adduct[0] = 2;
    assert!(
        get_evidence_for(&cache, &bumped_ad, 0, uploaded[0], keys8[0], 2048, h_cap_max).is_none(),
        "adduct change must miss"
    );
    // A forged entry with the right hash but different canonical inputs is a
    // miss (and counted): canonical bytes of a wider row stored under row
    // 0's key.
    let entry = get_evidence_for(&cache, &batch, 0, uploaded[0], keys8[0], 2048, h_cap_max).unwrap();
    let (key0, _) = evidence_key_for_batch(&batch, 0, uploaded[0], keys8[0], 2048, h_cap_max);
    let mut forged2 = EnumCache::new(model.enum_cache_header(&gcfg).unwrap());
    let forged_inputs = EvidenceInputs {
        mz_row: &batch.mz_udalton[0..9],
        intensity_row: &batch.intensity[0..9],
        intensity_scale: 0,
        precursor: batch.precursor_mz_udalton[0],
        adduct: 1,
        fragment_ppm: 100,
        mz_uncertainty: 50,
        peak_count: uploaded[0] + 1,
        work_max: 2048,
        p: EVIDENCE_PEAKS as u32,
        h_cap_max,
    };
    forged2
        .insert_evidence(
            key0,
            uploaded[0] + 1,
            forged_inputs.mz_row.iter().map(|&m| u64::from(m)).sum(),
            entry.n_ev,
            entry.explained.to_vec(),
            entry.weight_bits.to_vec(),
            entry.complete.to_vec(),
            32,
            evidence_canonical_bytes(&forged_inputs),
        )
        .unwrap();
    let forged_queries = evidence_queries_for(&batch, &meta, &uploaded, &rows, 2048, h_cap_max);
    assert!(
        forged2.expand_evidence_batch(&forged_queries[0..1], 32, TEST_N_PEAKS, TEST_DTYPE).is_none(),
        "wrong canonical inputs must miss despite the right hash"
    );
    assert!(
        forged2.collisions() > 0,
        "a verified-hash mismatch must count a collision"
    );
}

#[test]
fn evidence_diagnostics_equal_cached() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (table, _dtable, _) = setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let set_comps = vec![comps[4], comps[4], comps[4]];
    let set_batch = evidence_fixture_batch(&comps, 64);
    let set = experiment_set(&set_comps, &set_batch);
    let indices = vec![0usize, 1, 2];
    let mut train_config = tiny_train();
    train_config.formula_window = 32;
    let mut ev_cfg = tiny_model();
    ev_cfg.formula_features = FormulaFeatures::Evidence;
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut trainer_plain =
        Ms2Trainer::<R, E>::new(&ev_cfg, &table, &train_config, &device).unwrap();
    let mut trainer_cached =
        Ms2Trainer::<R, E>::new(&ev_cfg, &table, &train_config, &device).unwrap();
    trainer_plain.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer_cached.upload_enum_artifacts(&domain, &bounds).unwrap();
    let cache_batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let mut tcache = EnumCache::new(trainer_cached.enum_cache_header(32).unwrap());
    trainer_cached
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut tcache)
        .unwrap();
    trainer_cached.set_enum_cache(Some(Arc::new(tcache))).unwrap();
    let d_plain = trainer_plain.evidence_diagnostics(&set, &indices).unwrap();
    let d_cached = trainer_cached.evidence_diagnostics(&set, &indices).unwrap();
    assert!(d_plain.is_some() && d_cached.is_some(), "diagnostics run under Evidence");
    assert_eq!(d_plain, d_cached, "evidence_diagnostics must equal with and without the cache");
}

#[test]
fn evidence_save_load_and_version_refused() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, _dtable, model) = setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let batch = evidence_fixture_batch(&comps, 64);
    let gcfg = tiny_gen_evidence(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header.clone());
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    assert!(cache.evidence_len() > 0);
    let path = std::env::temp_dir().join("ms2_enum_cache_evidence_test.bin");
    cache.save(&path).unwrap();
    let on_disk = std::fs::metadata(&path).unwrap().len() as usize;
    assert_eq!(cache.bytes(), on_disk, "bytes() must be the file size");
    let loaded = EnumCache::load(&path, &header).unwrap();
    assert_eq!(loaded.len(), cache.len());
    assert_eq!(loaded.evidence_len(), cache.evidence_len());
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let meta = build_enum_meta(&batch, artifacts.domain_max_error, 65_536, 32);
    let keys = meta_keys(&meta);
    assert_eq!(
        loaded.expand_batch(&keys, 32),
        cache.expand_batch(&keys, 32),
        "save → load must preserve enum entries"
    );
    // Uploaded peak counts: row 2 is fatal (precursor out of range)
    // and uploads with peak_count 0.
    let uploaded = vec![8u32, 8, 0];
    let (_, counters_host) = cache.expand_batch(&keys, 32).expect("enum hit");
    let rows: Vec<usize> = (0..3).map(|i| counters_host[i * 5 + 2] as usize).collect();
    let queries = evidence_queries_for(
        &batch,
        &meta,
        &uploaded,
        &rows,
        2048,
        artifacts.hydrogen_cap_max(),
    );
    let got_cache = cache.expand_evidence_batch(&queries, 32, TEST_N_PEAKS, TEST_DTYPE);
    assert!(got_cache.is_some(), "evidence must hit before save");
    assert_eq!(
        loaded.expand_evidence_batch(&queries, 32, TEST_N_PEAKS, TEST_DTYPE),
        got_cache,
        "save → load must preserve evidence entries"
    );
    // Version 1 to 3 files are refused by name, never silently
    // reinterpreted (task F8 item 8: format version 4).
    for version in [1u32, 2, 3] {
        let mut old = std::fs::read(&path).unwrap();
        old[8..12].copy_from_slice(&version.to_le_bytes());
        std::fs::write(&path, &old).unwrap();
        let err = EnumCache::load(&path, &header)
            .expect_err(&format!("version {version} must be refused"));
        assert!(
            err.to_string().contains("format version"),
            "refusal must name the version, got {err}"
        );
        assert!(
            err.to_string().contains(&ENUM_CACHE_FORMAT_VERSION.to_string()),
            "refusal must name the needed version, got {err}"
        );
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn dispatch_default_sixteen_million_and_launch_formula() {
    let _serial = serial();
    // Part 1: the default is 16,000,000 in both configs and in documents
    // deserialised without the field.
    assert_eq!(GenerationConfig::default().enum_dispatch_visits_max, 16_000_000);
    assert_eq!(TrainConfig::default().enum_dispatch_visits_max, 16_000_000);
    let gen_json = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "trajectories": 4,
        "formulas": 2,
        "seed": 7,
        "temperature": 1.0,
        "max_steps": 22,
        "max_device_bytes": 2147483648u64,
        "formula_rows_visited_max": 4294967295u32,
        "formula_rows_scored_max": 4096,
        "mode": "Sampling",
        "oracle_formula": false,
        "control": "None",
    });
    let gen_cfg: GenerationConfig = serde_json::from_value(gen_json).unwrap();
    assert_eq!(gen_cfg.enum_dispatch_visits_max, 16_000_000);
    let train_json = serde_json::json!({
        "batch": 2,
        "slots": 2,
        "lr": 0.001,
        "weight_decay": 0.1,
        "formula_weight": 0.2,
        "seed": 1,
        "control": "None",
    });
    let train: TrainConfig = serde_json::from_value(train_json).unwrap();
    assert_eq!(train.enum_dispatch_visits_max, 16_000_000);
    // Launch counts follow `ceil(B * P / max(1, dispatch / lane_visits))`:
    // with the 16,000,000 default and an 8,000,000 lane budget each launch
    // covers 2 lanes.
    let device = dev();
    let launch = EnumLaunch::from_chemistry();
    let meta_rows: [[u32; 8]; 2] = [[0, 0, 0, 1, 0, 4096, 8, 0], [0, 0, 0, 1, 0, 4096, 8, 0]];
    let rare_rows: [[u32; 8]; 2] = [[0, 0, 0, 0, 0, 0, 0, 0]; 2];
    let packed = vec![0u32; 64];
    let meta_flat: Vec<u32> = meta_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let rare_flat: Vec<u32> = rare_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let dispatch = GenerationConfig::default().enum_dispatch_visits_max;
    let lane_visits = 8_000_000u32;
    let per = enum_lanes_per_dispatch(dispatch, lane_visits);
    assert_eq!(per, 2);
    let want = (4 + per - 1) / per;
    let meta_t = IdTensor::from_slice(&meta_flat, vec![2, 8], &device).unwrap();
    let rare_t = IdTensor::from_slice(&rare_flat, vec![2, 8], &device).unwrap();
    let bounds_t = IdTensor::from_slice(&packed, vec![packed.len()], &device).unwrap();
    let stats_t = IdTensor::from_slice(&vec![0u32; 2 * 2 * 2], vec![4, 2], &device).unwrap();
    reset_launch_count();
    launch
        .count(
            &meta_t,
            &rare_t,
            &bounds_t,
            &stats_t,
            262_144,
            dispatch,
            lane_visits,
        )
        .unwrap();
    device.synchronize();
    assert_eq!(
        launch_count(),
        want,
        "count launches ceil(4 / {per}) times at the 16,000,000 default"
    );
}

// ---------------------------------------------------------------------------
// Task F7A: Part A fixes (header discipline, payload checksum, unique temp
// files, bounded loading) and the missing Part A coverage.
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit (test-local copy of the file checksum's hash, for crafting
/// checksum-valid files by hand).
fn f7a_fnv1a64(bytes: &[u8], basis: u64) -> u64 {
    let mut h = basis;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Checksum-valid file for `body` (task F7A test helper): the two FNV-1a
/// passes over the body, appended as the trailing 128-bit checksum.
fn f7a_checksummed(mut body: Vec<u8>) -> Vec<u8> {
    let h0 = f7a_fnv1a64(&body, 0xcbf2_9ce4_8422_2325);
    let h1 = f7a_fnv1a64(&body, 0x3d0d_613b_7bde_ddda);
    body.extend_from_slice(&h0.to_le_bytes());
    body.extend_from_slice(&h1.to_le_bytes());
    body
}

/// One hand-inserted cache (no device): a single spectrum with one scored
/// candidate plus one evidence row, so every file offset is computable by
/// hand. Layout: header, one enum entry (`rows_scored = 1`), one evidence
/// entry (`rows = 1`).
fn f7a_hand_cache() -> (EnumCache, EnumCacheHeader) {
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        4,
        4096,
        65_536,
        16,
        "f32",
    );
    let mut cache = EnumCache::new(header.clone());
    let mut counts = [0u16; 10];
    counts[0] = 1;
    counts[1] = 4;
    let mass = composition_mass(&counts).unwrap();
    let row = [
        1u32, 4, 0, 0, 0, 0, 0, 0, 0, 0, mass, 1, u32::MAX,
    ];
    cache.insert([11u32; 8], [7, 5, 1, 0, 1], &row, 4).unwrap();
    cache
        .insert_evidence(
            EvidenceKey { meta: [12u32; 8], h0: 1, h1: 2 },
            8,
            100,
            3,
            vec![2],
            vec![1.0f32.to_bits()],
            vec![1],
            4,
            evidence_canonical_bytes(&EvidenceInputs {
                mz_row: &[],
                intensity_row: &[],
                intensity_scale: 0,
                precursor: 0,
                adduct: 0,
                fragment_ppm: 100,
                mz_uncertainty: 50,
                peak_count: 8,
                work_max: 0,
                p: 32,
                h_cap_max: 0,
            }),
        )
        .unwrap();
    (cache, header)
}

/// Byte offset of the hand cache's regions inside a saved file of header
/// JSON length `hlen`: the stored `complete` word, the first candidate
/// count byte, and the first evidence weight word.
fn f7a_hand_offsets(hlen: usize) -> (usize, usize, usize) {
    let complete = 16 + hlen + 8 + 32 + 16;
    let count_byte = 16 + hlen + 8 + 56;
    let ev_weight = 16 + hlen + 8 + 68 + 8 + 65 + 1;
    (complete, count_byte, ev_weight)
}

#[test]
fn f7a_checksum_rejects_single_bit_flips() {
    let _serial = serial();
    let (cache, header) = f7a_hand_cache();
    let dir = std::env::temp_dir().join("ms2_f7a_checksum");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    cache.save(&path).unwrap();
    let file = std::fs::read(&path).unwrap();
    assert_eq!(cache.bytes(), file.len(), "bytes() must be the file size");
    let hlen =
        u32::from_le_bytes(file[12..16].try_into().unwrap()) as usize;
    let (complete_off, count_off, weight_off) = f7a_hand_offsets(hlen);
    assert!(weight_off + 4 < file.len() - 16, "weight offset must be in the evidence section");
    // Flip one bit of a stored `complete` word, of a count byte, of an
    // evidence weight, and of the header: every load names the checksum.
    for (name, off) in [
        ("complete", complete_off),
        ("count", count_off),
        ("evidence weight", weight_off),
        ("header", 21usize),
    ] {
        let mut bad = file.clone();
        bad[off] ^= 0x01;
        std::fs::write(&path, &bad).unwrap();
        let err = EnumCache::load(&path, &header).expect_err(&format!("{name} bit flip must fail"));
        assert!(
            err.to_string().contains("checksum"),
            "{name} flip must name the checksum, got {err}"
        );
    }
    // The unmodified file still loads.
    std::fs::write(&path, &file).unwrap();
    assert_eq!(EnumCache::load(&path, &header).unwrap().len(), 1);
    std::fs::remove_file(&path).ok();
}

#[test]
fn f7a_old_versions_refused_by_name() {
    let _serial = serial();
    let (cache, header) = f7a_hand_cache();
    let dir = std::env::temp_dir().join("ms2_f7a_versions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    cache.save(&path).unwrap();
    let file = std::fs::read(&path).unwrap();
    // Versions 1 (no evidence section), 2 (no checksum) and 3 (no evidence
    // identity, unverified hash hits) are all refused by name (task F8 item
    // 8: format version 4).
    for version in [1u32, 2u32, 3u32] {
        let mut old = file.clone();
        old[8..12].copy_from_slice(&version.to_le_bytes());
        std::fs::write(&path, &old).unwrap();
        let err =
            EnumCache::load(&path, &header).expect_err("older version must be refused");
        assert!(
            err.to_string().contains("format version"),
            "refusal must name the version, got {err}"
        );
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn f7a_bounded_loading_million_entries() {
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        4,
        4096,
        65_536,
        16,
        "f32",
    );
    let dir = std::env::temp_dir().join("ms2_f7a_bounded");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    let hj = serde_json::to_string(&header).unwrap();
    // A tiny file advertising a million entries (checksum-valid, so the
    // bounded parse is what rejects it — not the checksum).
    let mut body = Vec::new();
    body.extend_from_slice(b"MS2ENUMC");
    body.extend_from_slice(&ENUM_CACHE_FORMAT_VERSION.to_le_bytes());
    body.extend_from_slice(&(hj.len() as u32).to_le_bytes());
    body.extend_from_slice(hj.as_bytes());
    body.extend_from_slice(&1_000_000u64.to_le_bytes());
    std::fs::write(&path, f7a_checksummed(body)).unwrap();
    let err = EnumCache::load(&path, &header).expect_err("million-entry stub must fail");
    let msg = err.to_string();
    assert!(
        !msg.contains("checksum") && !msg.contains("format version"),
        "bounded parse (not checksum/version) must reject it, got {msg}"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn f7a_duplicate_trailing_and_invalid_entries_rejected() {
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        4,
        4096,
        65_536,
        16,
        "f32",
    );
    let dir = std::env::temp_dir().join("ms2_f7a_malformed");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    let hj = serde_json::to_string(&header).unwrap();
    // One valid entry's bytes: key, counters (`rows_scored = 1`), one
    // candidate (C=1, H=4, flag 1).
    let mut entry = Vec::new();
    for _ in 0..8 {
        entry.extend_from_slice(&7u32.to_le_bytes());
    }
    for w in [5u32, 3, 1, 0, 1] {
        entry.extend_from_slice(&w.to_le_bytes());
    }
    entry.extend_from_slice(&1u32.to_le_bytes());
    entry.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0]);
    entry.extend_from_slice(&4u16.to_le_bytes());
    entry.push(1);
    let mut base = Vec::new();
    base.extend_from_slice(b"MS2ENUMC");
    base.extend_from_slice(&ENUM_CACHE_FORMAT_VERSION.to_le_bytes());
    base.extend_from_slice(&(hj.len() as u32).to_le_bytes());
    base.extend_from_slice(hj.as_bytes());
    // Duplicate keys: the same entry twice.
    let mut dup = base.clone();
    dup.extend_from_slice(&2u64.to_le_bytes());
    dup.extend_from_slice(&entry);
    dup.extend_from_slice(&entry);
    dup.extend_from_slice(&0u64.to_le_bytes());
    std::fs::write(&path, f7a_checksummed(dup)).unwrap();
    let err = EnumCache::load(&path, &header).expect_err("duplicate keys must fail");
    assert!(err.to_string().contains("duplicate"), "got {err}");
    // Trailing bytes: a valid single-entry file plus one junk byte.
    let mut single = base.clone();
    single.extend_from_slice(&1u64.to_le_bytes());
    single.extend_from_slice(&entry);
    single.extend_from_slice(&0u64.to_le_bytes());
    let mut trailing = single.clone();
    trailing.push(0xAA);
    std::fs::write(&path, f7a_checksummed(trailing)).unwrap();
    let err = EnumCache::load(&path, &header).expect_err("trailing bytes must fail");
    assert!(err.to_string().contains("trailing"), "got {err}");
    // The untainted single-entry file loads.
    std::fs::write(&path, f7a_checksummed(single)).unwrap();
    assert_eq!(EnumCache::load(&path, &header).unwrap().len(), 1);
    // Invalid entry: flag 3 (checksum recomputed so the flag check fires).
    let mut bad = std::fs::read(&path).unwrap();
    bad.truncate(bad.len() - 16);
    let hlen = u32::from_le_bytes(bad[12..16].try_into().unwrap()) as usize;
    let flag_off = 16 + hlen + 8 + 56 + 11;
    bad[flag_off] = 3;
    std::fs::write(&path, f7a_checksummed(bad)).unwrap();
    let err = EnumCache::load(&path, &header).expect_err("flag 3 must fail");
    assert!(err.to_string().contains("flag"), "got {err}");
    // Evidence duplicates are rejected on load (task F8 item 7): two
    // identical (key, canonical) records fail naming the duplicate, while
    // two colliding keys with different canonical bytes coexist.
    fn ev_record(meta: u32, h0: u64, h1: u64, pc: u32, canon: &[u8]) -> Vec<u8> {
        let mut rec = Vec::new();
        for _ in 0..8 {
            rec.extend_from_slice(&meta.to_le_bytes());
        }
        rec.extend_from_slice(&h0.to_le_bytes());
        rec.extend_from_slice(&h1.to_le_bytes());
        rec.extend_from_slice(&pc.to_le_bytes());
        rec.extend_from_slice(&0u64.to_le_bytes());
        rec.push(0);
        rec.extend_from_slice(&0u32.to_le_bytes());
        rec.extend_from_slice(&(canon.len() as u32).to_le_bytes());
        rec.extend_from_slice(canon);
        rec
    }
    // Canonical bytes for `peak_count = 1` with one slot (embedded
    // peak-count word 1, then one m/z + intensity-bit pair).
    fn canon_one_slot(ib: u32) -> Vec<u8> {
        let mut c = Vec::new();
        for w in [1u32, 0, 0, 0, 0, 0, 0, 0, 0] {
            c.extend_from_slice(&w.to_le_bytes());
        }
        c.extend_from_slice(&60_000_000u32.to_le_bytes());
        c.extend_from_slice(&ib.to_le_bytes());
        c
    }
    let hj_dup = serde_json::to_string(&header).unwrap();
    let mut ev_base = Vec::new();
    ev_base.extend_from_slice(b"MS2ENUMC");
    ev_base.extend_from_slice(&ENUM_CACHE_FORMAT_VERSION.to_le_bytes());
    ev_base.extend_from_slice(&(hj_dup.len() as u32).to_le_bytes());
    ev_base.extend_from_slice(hj_dup.as_bytes());
    ev_base.extend_from_slice(&0u64.to_le_bytes());
    let canon = canon_one_slot(1.0f32.to_bits());
    let mut dup_ev = ev_base.clone();
    dup_ev.extend_from_slice(&2u64.to_le_bytes());
    dup_ev.extend_from_slice(&ev_record(7, 11, 13, 1, &canon));
    dup_ev.extend_from_slice(&ev_record(7, 11, 13, 1, &canon));
    std::fs::write(&path, f7a_checksummed(dup_ev)).unwrap();
    let err = EnumCache::load(&path, &header).expect_err("duplicate evidence must fail");
    assert!(
        err.to_string().contains("duplicate evidence"),
        "got {err}"
    );
    // Same key, different canonical bytes: loads, both entries kept.
    let mut canon2 = canon.clone();
    let last = canon2.len() - 1;
    canon2[last] ^= 0x01;
    let mut ok_ev = ev_base.clone();
    ok_ev.extend_from_slice(&2u64.to_le_bytes());
    ok_ev.extend_from_slice(&ev_record(7, 11, 13, 1, &canon));
    ok_ev.extend_from_slice(&ev_record(7, 11, 13, 1, &canon2));
    std::fs::write(&path, f7a_checksummed(ok_ev)).unwrap();
    let loaded = EnumCache::load(&path, &header).unwrap();
    assert_eq!(loaded.evidence_len(), 2, "colliding keys must coexist through load");
    std::fs::remove_file(&path).ok();
}

/// State for the deterministic concurrent-save overlap
/// (`f7a_concurrent_saves_same_path`): rendezvous + in-save gauges driven by
/// the `test-support` save-phase hook.
static RENDEZVOUS_STATE: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
static RENDEZVOUS_CV: std::sync::Condvar = std::sync::Condvar::new();
/// Saves currently between `Created` and `Renamed`.
static SAVE_INSIDE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Maximum observed `SAVE_INSIDE` (proves overlap happened).
static SAVE_MAX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn save_overlap_hook(phase: SavePhase) {
    use std::sync::atomic::Ordering;
    match phase {
        SavePhase::Created => {
            let n = SAVE_INSIDE.fetch_add(1, Ordering::SeqCst) + 1;
            SAVE_MAX.fetch_max(n, Ordering::SeqCst);
        }
        SavePhase::Written => {
            // Force the overlap: both saves hold a complete temporary file
            // before either renames. A condvar rendezvous with a 30 s timeout
            // (not an un-timed barrier wait): if a worker fails early, its
            // peer rendezvouses out after 30 s instead of hanging the
            // regression suite forever (task F9 item A6). Only the first
            // `Written` per worker rendezvouses (later rounds already
            // proved overlap).
            let mut n = RENDEZVOUS_STATE.lock().unwrap_or_else(|e| e.into_inner());
            if *n < 2 {
                *n += 1;
                let (guard, _wait) = RENDEZVOUS_CV
                    .wait_timeout_while(n, std::time::Duration::from_secs(30), |n| *n < 2)
                    .unwrap_or_else(|e| e.into_inner());
                drop(guard);
            }
        }
        SavePhase::Renamed => {
            SAVE_INSIDE.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Reset the doc-hidden save hook on drop (a set hook would break every
/// other save in the process).
struct SaveHookGuard;
impl Drop for SaveHookGuard {
    fn drop(&mut self) {
        set_save_phase_hook(None);
    }
}

#[test]
fn f7a_concurrent_saves_same_path() {
    use std::sync::atomic::Ordering;
    let _serial = serial();
    let dir = std::env::temp_dir().join("ms2_f7a_concurrent_saves");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    // Hygiene from killed earlier runs (a crash between create and rename
    // leaves orphans by design): drop this destination's stale temporaries
    // before measuring that the overlapped run leaves none.
    for stale in EnumCache::stale_temp_files(&path) {
        std::fs::remove_file(&stale).ok();
    }
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        4,
        4096,
        65_536,
        16,
        "f32",
    );
    let mut cache_a = EnumCache::new(header.clone());
    let mut cache_b = EnumCache::new(header.clone());
    let mut counts = [0u16; 10];
    counts[0] = 1;
    counts[1] = 4;
    let mass = composition_mass(&counts).unwrap();
    let row = [1u32, 4, 0, 0, 0, 0, 0, 0, 0, 0, mass, 1, u32::MAX];
    cache_a.insert([21u32; 8], [7, 5, 1, 0, 1], &row, 4).unwrap();
    cache_b.insert([22u32; 8], [7, 5, 1, 0, 1], &row, 4).unwrap();
    let cache_a = Arc::new(cache_a);
    let cache_b = Arc::new(cache_b);
    *RENDEZVOUS_STATE.lock().unwrap_or_else(|e| e.into_inner()) = 0;
    SAVE_INSIDE.store(0, Ordering::SeqCst);
    SAVE_MAX.store(0, Ordering::SeqCst);
    set_save_phase_hook(Some(save_overlap_hook));
    let _guard = SaveHookGuard;
    let worker = |cache: Arc<EnumCache>, path: std::path::PathBuf| {
        std::thread::spawn(move || {
            for _ in 0..20 {
                cache.save(&path).expect("concurrent save must succeed");
            }
        })
    };
    let h1 = worker(cache_a, path.clone());
    let h2 = worker(cache_b, path.clone());
    h1.join().unwrap();
    h2.join().unwrap();
    // The barrier on `Written` forced genuine overlap inside `save`
    // (reverting the hook calls leaves this at 0; sharing one temporary
    // path fails the saves above instead).
    assert!(
        SAVE_MAX.load(Ordering::SeqCst) >= 2,
        "the two saves must overlap inside save"
    );
    // The surviving file is one complete save: it loads under the header.
    let loaded = EnumCache::load(&path, &header).unwrap();
    assert_eq!(loaded.len(), 1, "atomic saves never leave a partial file");
    // No temporary file remains (listed, not deleted, by production code).
    assert!(
        EnumCache::stale_temp_files(&path).is_empty(),
        "no temporary file may remain"
    );
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        assert!(
            !name.contains(".tmp-"),
            "temporary file left behind: {name}"
        );
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn f7a_concurrent_lookups_exact_totals() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    // `Ms2Model` is not `Sync` (workspace `Rc`), so concurrent production
    // serving happens one level down: `Arc<EnumCache>` readers. The cache's
    // own atomic totals are the production counters of that path (reverting
    // the counter updates leaves them at zero); every lookup also asserts
    // its returned buffer. Afterwards the same cache serves the model
    // sequentially, tying the model-level production counters to the same
    // buffers.
    let (_table, dtable, ev_model) =
        setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let ev_batch = evidence_fixture_batch(&comps, 64);
    let ev_gcfg = tiny_gen_evidence(32, 4096);
    let ev_header = ev_model.enum_cache_header(&ev_gcfg).unwrap();
    let mut ev_cache = EnumCache::new(ev_header);
    ev_model
        .build_enum_cache([&ev_batch].into_iter(), &ev_gcfg, &mut ev_cache)
        .unwrap();
    assert!(ev_cache.evidence_len() > 0, "the fixture must cache evidence");
    let ev_artifacts = ev_model.enum_artifacts.as_ref().unwrap();
    let ev_meta = build_enum_meta(&ev_batch, ev_artifacts.domain_max_error, 65_536, 32);
    let ev_keys = meta_keys(&ev_meta);
    let (_, ev_counters) = ev_cache.expand_batch(&ev_keys, 32).expect("enum hit");
    let ev_rows: Vec<usize> = (0..3).map(|i| ev_counters[i * 5 + 2] as usize).collect();
    let ev_uploaded = vec![8u32, 8, 0];
    let ev_queries = evidence_queries_for(
        &ev_batch,
        &ev_meta,
        &ev_uploaded,
        &ev_rows,
        2048,
        ev_artifacts.hydrogen_cap_max(),
    );
    let want_cand = ev_cache.expand_batch(&ev_keys, 32).expect("enum reference");
    let want_ev = ev_cache
        .expand_evidence_batch(&ev_queries, 32, TEST_N_PEAKS, TEST_DTYPE)
        .expect("evidence reference");
    // Baselines above counted (2, 1, 1, 1); snapshot them out.
    let base = ev_cache.lookup_stats();
    let ev_cache = Arc::new(ev_cache);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                for _ in 0..100 {
                    let (cand, counters) =
                        ev_cache.expand_batch(&ev_keys, 32).expect("all enum lookups hit");
                    assert_eq!((cand, counters), want_cand, "concurrent enum buffer differs");
                    let ev = ev_cache
                        .expand_evidence_batch(&ev_queries, 32, TEST_N_PEAKS, TEST_DTYPE)
                        .expect("all evidence lookups hit");
                    assert_eq!(ev, want_ev, "concurrent evidence buffer differs");
                }
            });
        }
    });
    // Production totals after workers finish: 8 x 100 batch lookups and
    // hits on each map — and no collisions on distinct rows.
    let (lookups, hits, ev_lookups, ev_hits) = ev_cache.lookup_stats();
    assert_eq!(
        (lookups - base.0, hits - base.1),
        (800, 800),
        "production enum totals must count every concurrent lookup"
    );
    assert_eq!(
        (ev_lookups - base.2, ev_hits - base.3),
        (800, 800),
        "production evidence totals must count every concurrent lookup"
    );
    assert_eq!(ev_cache.collisions(), 0, "no collision on distinct rows");
    // The same cache serves the model sequentially with identical buffers.
    let mut ev_model = ev_model;
    ev_model.set_enum_cache(Some(ev_cache)).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    ev_model.generate(&ev_batch, &dtable, &ev_gcfg, &mut ws, &constants).unwrap();
    let (mlookups, mhits, mev_lookups, mev_hits) = ev_model.enum_cache_stats();
    assert_eq!((mlookups, mhits), (1, 1));
    assert_eq!((mev_lookups, mev_hits), (1, 1));
}

#[test]
fn f7a_attach_rejects_foreign_bounds_generation_and_training() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds_a = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut wide = comps.clone();
    wide.push([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
    let bounds_b = RatioBounds::fit(wide, 0).unwrap();
    let (table, dtable, _) = setup_model(&device, &comps, 5);
    let mut rng_a = Rng::seeded(5);
    let mut rng_b = Rng::seeded(5);
    let mut cfg = tiny_model();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut model_a = Ms2Model::<R, E>::init(&cfg, &device, &mut rng_a).unwrap();
    let mut model_b = Ms2Model::<R, E>::init(&cfg, &device, &mut rng_b).unwrap();
    model_a.upload_enum_artifacts(&domain, &bounds_a, &device).unwrap();
    model_b.upload_enum_artifacts(&domain, &bounds_b, &device).unwrap();
    let gcfg = tiny_gen(32, 4096);
    let header_a = model_a.enum_cache_header(&gcfg).unwrap();
    let header_b = model_b.enum_cache_header(&gcfg).unwrap();
    assert_ne!(
        header_a.bounds_sha256, header_b.bounds_sha256,
        "the two bounds must hash differently"
    );
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let mut cache = EnumCache::new(header_a);
    model_a
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    // Generation: attaching the foreign cache fails early, naming the field.
    let err = model_b
        .set_enum_cache(Some(Arc::new(cache.clone())))
        .expect_err("foreign bounds must be refused at attach");
    assert!(err.to_string().contains("`bounds_sha256`"), "got {err}");
    // Training: the same attachment fails the same way.
    let mut trainer_b =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &tiny_train(), &device).unwrap();
    trainer_b.upload_enum_artifacts(&domain, &bounds_b).unwrap();
    let err = trainer_b
        .set_enum_cache(Some(Arc::new(cache)))
        .expect_err("foreign bounds must be refused at attach (training)");
    assert!(err.to_string().contains("`bounds_sha256`"), "got {err}");
}

#[test]
fn f7a_per_use_header_mismatch_is_config_error() {
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
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let constants = Ms2Constants::new(&device);
    // A different scored cap later: the request is Error::Config naming the
    // field — never a silent fallback, never a hit.
    let mut other = tiny_gen(32, 4096);
    other.formula_rows_scored_max = 16;
    let mut ws = GenerationWorkspace::new();
    let err = model
        .generate(&batch, &dtable, &other, &mut ws, &constants)
        .expect_err("stale scored cap must be refused");
    assert!(
        err.to_string().contains("`formula_rows_scored_max`"),
        "got {err}"
    );
    let (lookups, hits, _, _) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (0, 0), "a refused request is no lookup");
    // The matching config still hits.
    let mut ws2 = GenerationWorkspace::new();
    model.generate(&batch, &dtable, &gcfg, &mut ws2, &constants).unwrap();
    let (lookups, hits, _, _) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (1, 1));
}

#[test]
fn f7a_training_per_use_header_mismatch_is_config_error() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let set = experiment_set(&vec![comps[4], comps[6]], &batch);
    let indices = vec![0usize, 1];
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &tiny_train(), &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let cache_batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let mut cache = EnumCache::new(trainer.enum_cache_header(32).unwrap());
    trainer
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut cache)
        .unwrap();
    let cache = Arc::new(cache);
    // A trainer with a different lane visit budget: attachment succeeds
    // (artifacts match) but the step is refused at use, naming the field.
    let mut other_train = tiny_train();
    other_train.enum_lane_visits_max = trainer.train_config().enum_lane_visits_max + 1;
    let mut trainer_b =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &other_train, &device).unwrap();
    trainer_b.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer_b.set_enum_cache(Some(cache)).unwrap();
    let err = trainer_b
        .step(&set, &indices)
        .expect_err("stale lane budget must be refused");
    assert!(
        err.to_string().contains("`enum_lane_visits_max`"),
        "got {err}"
    );
    let (lookups, hits, _, _) = trainer_b.enum_cache_stats();
    assert_eq!((lookups, hits), (0, 0), "a refused step is no lookup");
}

#[test]
fn f7a_dispatch_invariance() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, dtable, mut model) = setup_model(&device, &comps, 5);
    let lane_visits = 65_536u32;
    // Four spectra (one far-precursor, genuinely exhausted) over `P` lanes
    // each: `B * P` lanes, so one-lane dispatches and one all-lane dispatch
    // give different chunk counts. (The old fixture fit both bounds in one
    // chunk, so a multi-chunk indexing defect passed silently.)
    let batch = spectrum_batch(
        &[
            precursor_of(&comps[4]),
            precursor_of(&comps[6]),
            4_000_000_000,
            precursor_of(&comps[2]),
        ],
        &[50, 50, 50, 50],
        &[1, 1, 1, 1],
        &[0, 0, 0, 0],
        64,
    );
    let lanes = 4 * model.enum_artifacts.as_ref().unwrap().p;
    assert!(lanes > 1, "the fixture needs several lanes");
    let dispatch_a = lane_visits;
    let dispatch_b = (lanes as u32).saturating_mul(lane_visits);
    let per_a = enum_lanes_per_dispatch(dispatch_a, lane_visits);
    let per_b = enum_lanes_per_dispatch(dispatch_b, lane_visits);
    assert_eq!(per_a, 1, "bound A must dispatch one lane at a time");
    assert!(per_b >= lanes, "bound B must cover every lane at once");
    assert_ne!(per_a, per_b, "the two bounds must give different chunk counts");
    // Raw buffers at both bounds, with launch counts proving the chunking
    // actually differed (reverting to one shared chunk count fails here).
    reset_launch_count();
    let (cand_a, counters_a) =
        device_enumerate(&device, &model, &batch, 32, 32, lane_visits, dispatch_a);
    device.synchronize();
    let launches_a = launch_count();
    reset_launch_count();
    let (cand_b, counters_b) =
        device_enumerate(&device, &model, &batch, 32, 32, lane_visits, dispatch_b);
    device.synchronize();
    let launches_b = launch_count();
    assert_ne!(
        launches_a, launches_b,
        "different chunk counts must launch differently ({launches_a} vs {launches_b})"
    );
    assert_eq!(cand_a, cand_b, "raw cand differs across dispatch bounds");
    assert_eq!(counters_a, counters_b, "raw counters differ across dispatch bounds");
    // The far-precursor row scores nothing (empty entry: the padding
    // regeneration edge) — and is identical across bounds.
    assert_eq!(counters_a[2 * 5 + 2], 0, "far-precursor row scores nothing");
    // The cache serves the raw buffers of both bounds, exhausted row
    // included.
    let gcfg = tiny_gen(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    let domain_max_error = model.enum_artifacts.as_ref().unwrap().domain_max_error;
    let meta = build_enum_meta(&batch, domain_max_error, 65_536, 32);
    let (cand_hit, counters_hit) =
        cache.expand_batch(&meta_keys(&meta), 32).expect("all hit");
    assert_eq!(cand_hit, cand_a, "cached cand differs from the raw buffers");
    assert_eq!(counters_hit, counters_a, "cached counters differ from the raw buffers");
    // The served search equals the uncached one, at this dispatch bound and
    // at another (dispatch is scheduling, never part of the key).
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws_hit = GenerationWorkspace::new();
    let hit = model.generate(&batch, &dtable, &gcfg, &mut ws_hit, &constants).unwrap();
    let mut gcfg16 = tiny_gen(32, 4096);
    gcfg16.enum_dispatch_visits_max = 16_000_000;
    let mut ws_hit16 = GenerationWorkspace::new();
    let hit16 = model.generate(&batch, &dtable, &gcfg16, &mut ws_hit16, &constants).unwrap();
    let (lookups, hits, _, _) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (2, 2), "dispatch is not part of the key");
    model.set_enum_cache(None).unwrap();
    let mut ws_plain = GenerationWorkspace::new();
    let plain = model.generate(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
    assert_candidate_batch_close(&hit, &plain);
    assert_candidate_batch_close(&hit16, &plain);
}

#[test]
fn f7a_flag2_round_trip_and_invalid_flags_refused() {
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        4,
        4096,
        65_536,
        16,
        "f32",
    );
    let mut cache = EnumCache::new(header);
    let mut counts = [0u16; 10];
    counts[0] = 2;
    counts[1] = 6;
    let mass = composition_mass(&counts).unwrap();
    // Flag 2 (ambiguous) survives the round trip explicitly.
    let row2 = [2u32, 6, 0, 0, 0, 0, 0, 0, 0, 0, mass, 2, u32::MAX];
    cache.insert([31u32; 8], [9, 4, 1, 0, 1], &row2, 4).unwrap();
    let (cand, _) = cache.expand_batch(&[[31u32; 8]], 4).expect("hit");
    assert_eq!(&cand[0..13], &row2, "flag 2 must survive");
    // Flag 0/3 and a wrong source id are refused at insert.
    let mut bad_flag = row2;
    bad_flag[11] = 3;
    assert!(cache.insert([32u32; 8], [0, 0, 1, 0, 0], &bad_flag, 4).is_err());
    let mut zero_flag = row2;
    zero_flag[11] = 0;
    assert!(
        cache.insert([34u32; 8], [0, 0, 1, 0, 0], &zero_flag, 4).is_err(),
        "flag 0 must be refused at insert"
    );
    let mut bad_src = row2;
    bad_src[12] = 0;
    assert!(cache.insert([33u32; 8], [0, 0, 1, 0, 0], &bad_src, 4).is_err());
    // Flag 2 survives save → load (the old test never left memory).
    let dir = std::env::temp_dir().join("ms2_f7a_flag2");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    cache.save(&path).unwrap();
    let loaded = EnumCache::load(&path, &cache.header().clone()).unwrap();
    let (cand_rt, _) = loaded.expand_batch(&[[31u32; 8]], 4).expect("hit");
    assert_eq!(&cand_rt[0..13], &row2, "flag 2 must survive save → load");
    // Flag 0 in a crafted file is refused, naming the flag (checksum
    // recomputed so the flag check fires, not the checksum).
    let mut bad = std::fs::read(&path).unwrap();
    bad.truncate(bad.len() - 16);
    let hlen = u32::from_le_bytes(bad[12..16].try_into().unwrap()) as usize;
    let flag_off = 16 + hlen + 8 + 56 + 11;
    bad[flag_off] = 0;
    std::fs::write(&path, f7a_checksummed(bad)).unwrap();
    let err = EnumCache::load(&path, &cache.header().clone()).expect_err("flag 0 must fail");
    assert!(err.to_string().contains("flag"), "got {err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn f7a_hit_miss_hit_matches_three_uncached_calls() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, dtable, mut model) = setup_model(&device, &comps, 5);
    // FIXED batch shape throughout (B = 2): only one precursor changes, so a
    // defect confined to reusing one unchanged bucket cannot hide (the old
    // test changed B 1 → 2 → 1).
    let pair_ab = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let pair_ac = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[2])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let gcfg = tiny_gen(32, 4096);
    let constants = Ms2Constants::new(&device);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&pair_ab].into_iter(), &gcfg, &mut cache)
        .unwrap();
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let mut ws = GenerationWorkspace::new();
    let hit1 = model.generate(&pair_ab, &dtable, &gcfg, &mut ws, &constants).unwrap();
    let miss = model.generate(&pair_ac, &dtable, &gcfg, &mut ws, &constants).unwrap();
    let hit2 = model.generate(&pair_ab, &dtable, &gcfg, &mut ws, &constants).unwrap();
    let (lookups, hits, _, _) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (3, 2), "hit, miss, hit at a fixed batch shape");
    model.set_enum_cache(None).unwrap();
    let mut ws2 = GenerationWorkspace::new();
    let plain1 = model.generate(&pair_ab, &dtable, &gcfg, &mut ws2, &constants).unwrap();
    let plain2 = model.generate(&pair_ac, &dtable, &gcfg, &mut ws2, &constants).unwrap();
    let plain3 = model.generate(&pair_ab, &dtable, &gcfg, &mut ws2, &constants).unwrap();
    assert_candidate_batch_close(&hit1, &plain1);
    assert_candidate_batch_close(&miss, &plain2);
    assert_candidate_batch_close(&hit2, &plain3);
}

#[test]
fn f7a_cached_step_weights_and_moments_bit_identical() {
    let _serial = serial();
    if !is_cpu() {
        println!("f7a weights/moments bit-identity is a cpu assertion; skipping on gpu");
        return;
    }
    let device = dev();
    let comps = fixture_comps();
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let set = experiment_set(&vec![comps[4], comps[6]], &batch);
    let indices = vec![0usize, 1];
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut trainer_a =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &tiny_train(), &device).unwrap();
    let mut trainer_b =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &tiny_train(), &device).unwrap();
    trainer_a.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer_b.upload_enum_artifacts(&domain, &bounds).unwrap();
    let cache_batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let mut tcache = EnumCache::new(trainer_b.enum_cache_header(32).unwrap());
    trainer_b
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut tcache)
        .unwrap();
    trainer_b.set_enum_cache(Some(Arc::new(tcache))).unwrap();
    trainer_a.request_report();
    trainer_b.request_report();
    let rep_a = trainer_a.step(&set, &indices).unwrap().unwrap();
    let rep_b = trainer_b.step(&set, &indices).unwrap().unwrap();
    assert_eq!(rep_a, rep_b, "cached training step report differs");
    // Weights AND optimizer moments: the checkpoint carries both, so equal
    // files mean bit-identical weights and moments.
    let dir = std::env::temp_dir().join("ms2_f7a_step_ident");
    std::fs::create_dir_all(&dir).unwrap();
    let pa = dir.join("a.json");
    let pb = dir.join("b.json");
    trainer_a.save(&pa).unwrap();
    trainer_b.save(&pb).unwrap();
    let ja: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pa).unwrap()).unwrap();
    let jb: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pb).unwrap()).unwrap();
    assert_eq!(ja, jb, "cached step must leave weights and moments bit-identical");
    std::fs::remove_file(&pa).ok();
    std::fs::remove_file(&pb).ok();
}

#[test]
fn f7a_exceptional_readout_cached() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, dtable, mut model) = setup_model(&device, &comps, 5);
    let productive = precursor_of(&comps[4]);
    // Invalid parent mass (far precursor), unknown precision, half-overflow,
    // plus a productive row: statuses and `complete` must equal uncached
    // through the full cached readout.
    let batch = spectrum_batch(
        &[productive, 4_000_000_000, productive, productive],
        &[50, 50, u32::MAX, 100_000_000],
        &[1, 1, 1, 1],
        &[0, 0, 0, 0],
        64,
    );
    let gcfg = tiny_gen(32, 4096);
    let constants = Ms2Constants::new(&device);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let mut ws = GenerationWorkspace::new();
    let hit = model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    // A hit was actually served: with cache serving disabled this fails
    // (the old test passed either way).
    let (lookups, hits, _, _) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (1, 1), "the exceptional batch must be served from the cache");
    model.set_enum_cache(None).unwrap();
    let mut ws2 = GenerationWorkspace::new();
    let plain = model.generate(&batch, &dtable, &gcfg, &mut ws2, &constants).unwrap();
    assert_candidate_batch_close(&hit, &plain);
    assert_eq!(hit.status, plain.status, "exceptional statuses must match");
    assert_eq!(
        hit.request_status, plain.request_status,
        "exceptional request statuses must match"
    );
    assert_eq!(
        hit.formula_support_complete, plain.formula_support_complete,
        "complete flags must match"
    );
    assert_ne!(
        hit.request_status.iter().sum::<u32>(),
        0,
        "the fixture must actually contain exceptional rows"
    );
    // Genuine exhaustion on the HIT output (not just parity with uncached):
    // the too-wide-window row (saturating m/z window) carries
    // FORMULA_SEARCH_EXHAUSTED with complete == 0 ...
    assert!(
        hit.request_status[3] & request_status::FORMULA_SEARCH_EXHAUSTED != 0,
        "wide-window row must be exhausted, got {:#x}",
        hit.request_status[3]
    );
    assert_eq!(
        hit.formula_support_complete[3], 0,
        "exhausted row must not claim complete"
    );
    // ... the invalid-mass row is exceptional, and the unknown-precision
    // row (skipped search) is exceptional too.
    assert_ne!(hit.request_status[1], 0, "invalid-mass row must be exceptional");
    assert_ne!(hit.request_status[2], 0, "unknown-precision row must be exceptional");
}

#[test]
fn f7a_cached_missing_artifact_and_config_refusals() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, dtable, model) = setup_model(&device, &comps, 5);
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let gcfg = tiny_gen(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    let cache = Arc::new(cache);
    // Attaching without resident artifacts is refused at attach.
    let mut rng = Rng::seeded(5);
    let mut bare = Ms2Model::<R, E>::init(&tiny_model(), &device, &mut rng).unwrap();
    let err = bare
        .set_enum_cache(Some(cache.clone()))
        .expect_err("attach without artifacts must fail");
    assert!(err.to_string().contains("resident enum artifacts"), "got {err}");
    // Config validation refuses identically with and without a cache.
    let mut bad = tiny_gen(32, 4096);
    bad.schema_version = 999;
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let plain_err = model
        .generate(&batch, &dtable, &bad, &mut ws, &constants)
        .expect_err("bad schema must be refused uncached");
    let mut cached_model = model;
    cached_model.set_enum_cache(Some(cache)).unwrap();
    let mut ws2 = GenerationWorkspace::new();
    let cached_err = cached_model
        .generate(&batch, &dtable, &bad, &mut ws2, &constants)
        .expect_err("bad schema must be refused cached");
    assert!(
        cached_err.to_string().contains("schema_version"),
        "got {cached_err}"
    );
    assert_eq!(
        plain_err.to_string(),
        cached_err.to_string(),
        "cached and uncached refusals must agree"
    );
}

#[test]
fn f7a_batch_reorder_and_subset_keep_jitter_hits() {
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
    let mut train_config = tiny_train();
    train_config.precursor_jitter_ppm = 2.0;
    train_config.precursor_jitter_variants = 3;
    train_config.seed = 9;
    let base = spectrum_batch_for(&set, &[0usize, 1], 64).unwrap();
    let mut variants: Vec<SpectrumBatch> = Vec::new();
    for v in 0..3 {
        let mut jb = base.clone();
        apply_precursor_jitter(&mut jb, &[0usize, 1], 2.0, 9, 1 + v as u64);
        variants.push(jb);
    }
    let table = FormulaTable::from_compositions(set_comps.iter().copied()).unwrap();
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &train_config, &device).unwrap();
    trainer.capture_prep_batch(true);
    let domain = EnumDomain::from_compositions(fixture_comps(), 0).unwrap();
    let bounds = RatioBounds::fit(fixture_comps(), 0).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let mut cache = EnumCache::new(trainer.enum_cache_header(32).unwrap());
    trainer
        .build_enum_cache(variants.iter(), 32, &mut cache)
        .unwrap();
    trainer.set_enum_cache(Some(Arc::new(cache))).unwrap();
    // Reordered, subset and full batches: the jitter-variant choice keys by
    // spectrum index, never by batch position, so every batch hits and every
    // precursor comes from the fixed pool of its own spectrum.
    // The FIRST step selects the exact (step, index) variant: step_tag is
    // `steps + 1 = 1`, draw `v = jitter_variant_index(seed, 1, idx, 3)` at
    // `split_tag = 1 + v` — asserted exactly, not by pool membership (a
    // wrong-member selection passed the old test).
    trainer.step(&set, &[1usize, 0]).unwrap();
    let prep = trainer.last_prep_batch.as_ref().unwrap();
    for (b, &idx) in [1usize, 0].iter().enumerate() {
        let v = jitter_variant_index(9, 1, idx as u64, 3);
        let want = jitter_precursor_mz(
            base.precursor_mz_udalton[idx],
            2.0,
            9,
            1 + u64::from(v),
            idx as u64,
        );
        assert_eq!(
            prep.precursor_mz_udalton[b], want,
            "first step spectrum {idx} must use exactly draw {v}"
        );
    }
    for indices in [vec![1usize], vec![0, 1]] {
        trainer.step(&set, &indices).unwrap();
        let prep = trainer.last_prep_batch.as_ref().unwrap();
        for (b, &idx) in indices.iter().enumerate() {
            let allowed: Vec<u32> = (0..3)
                .map(|v| {
                    jitter_precursor_mz(
                        base.precursor_mz_udalton[idx],
                        2.0,
                        9,
                        1 + v as u64,
                        idx as u64,
                    )
                })
                .collect();
            assert!(
                allowed.contains(&prep.precursor_mz_udalton[b]),
                "reordered precursor {} not in spectrum {idx}'s pool {allowed:?}",
                prep.precursor_mz_udalton[b]
            );
        }
    }
    let (lookups, hits, _, _) = trainer.enum_cache_stats();
    assert_eq!((lookups, hits), (3, 3), "reorder/subset batches all hit");
}

#[test]
fn f7a_v0_unchanged() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let set = experiment_set(&vec![comps[4], comps[6]], &batch);
    let indices = vec![0usize, 1];
    // V = 0: a fresh draw per step (`split_tag = 1 + step`), never the fixed
    // pool — the stored-precursor cache cannot serve it.
    let mut train_config = tiny_train();
    train_config.precursor_jitter_ppm = 2.0;
    train_config.precursor_jitter_variants = 0;
    train_config.seed = 9;
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut trainer_a =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &train_config, &device).unwrap();
    let mut trainer_b =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &train_config, &device).unwrap();
    trainer_a.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer_b.upload_enum_artifacts(&domain, &bounds).unwrap();
    let cache_batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    let mut cache = EnumCache::new(trainer_b.enum_cache_header(32).unwrap());
    trainer_b
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut cache)
        .unwrap();
    trainer_b.set_enum_cache(Some(Arc::new(cache))).unwrap();
    trainer_a.request_report();
    trainer_b.request_report();
    trainer_a.capture_prep_batch(true);
    trainer_b.capture_prep_batch(true);
    let rep_a = trainer_a.step(&set, &indices).unwrap().unwrap();
    let rep_b = trainer_b.step(&set, &indices).unwrap().unwrap();
    let (lookups, hits, _, _) = trainer_b.enum_cache_stats();
    assert_eq!((lookups, hits), (1, 0), "V = 0 draws miss the stored cache");
    // Independent oracle for the prepared precursor: V = 0 draws fresh at
    // `split_tag = 1 + step` (first step: tag 1), keyed by spectrum index —
    // never the fixed pool. (Reverting to pool draws fails this.)
    for trainer in [&trainer_a, &trainer_b] {
        let prep = trainer.last_prep_batch.as_ref().expect("prep captured");
        for (b, &idx) in indices.iter().enumerate() {
            let want = jitter_precursor_mz(
                cache_batch.precursor_mz_udalton[b],
                2.0,
                9,
                1,
                idx as u64,
            );
            assert_eq!(
                prep.precursor_mz_udalton[b], want,
                "V = 0 prepared precursor must be the fresh tag-1 draw"
            );
        }
    }
    if is_cpu() {
        assert_eq!(rep_a, rep_b, "V = 0 cached step must equal uncached");
    }
}

#[test]
fn f7a_eval_tag_never_training_tag() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let set = experiment_set(&vec![comps[4], comps[6]], &batch);
    // Evaluation always draws with `split_tag = 0`; training with
    // `split_tag = 1 + v` (fixed pool) or `1 + step` (V = 0). Through the
    // REAL evaluation preparation — `jittered_set_for_eval`, what the driver
    // evaluates — every precursor is the deterministic tag-0 draw, distinct
    // from the training tags. (Calling the jitter helper in isolation would
    // pass even if evaluation preparation reverted to a training tag.)
    let eval_set = jittered_set_for_eval(&set, 2.0, 9);
    for (i, s) in eval_set.spectra.iter().enumerate() {
        let stored = set.spectra[i].spectrum.precursor_mz_udalton;
        let eval_draw = s.spectrum.precursor_mz_udalton;
        assert_eq!(
            eval_draw,
            jitter_precursor_mz(stored, 2.0, 9, 0, i as u64),
            "evaluation preparation must draw at tag 0"
        );
        for tag in [1u64, 2, 3] {
            assert_ne!(
                eval_draw,
                jitter_precursor_mz(stored, 2.0, 9, tag, i as u64),
                "evaluation tag 0 must not coincide with training tag {tag}"
            );
        }
    }
    // The real evaluation over that prepared set is deterministic, and the
    // jitter actually moves the precursors under evaluation.
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &tiny_train(), &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let gcfg = tiny_gen(32, 4096);
    let indices = vec![0usize, 1];
    let (evals_a, _, _, _, _, _) = trainer
        .generate_eval_packed(&eval_set, &indices, &gcfg)
        .unwrap();
    let (evals_b, _, _, _, _, _) = trainer
        .generate_eval_packed(&eval_set, &indices, &gcfg)
        .unwrap();
    if is_cpu() {
        assert_eq!(evals_a, evals_b, "evaluation over the tag-0 set must be deterministic");
    } else {
        assert_eq!(evals_a.len(), evals_b.len());
    }
    let (evals_stored, _, _, _, _, _) = trainer
        .generate_eval_packed(&set, &indices, &gcfg)
        .unwrap();
    assert_ne!(
        evals_a, evals_stored,
        "the tag-0 evaluation must differ from the stored-precursor evaluation"
    );
}

#[test]
fn f7a_driver_pool_covers_training_two_epochs() {
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
    train_config.precursor_jitter_variants = 2;
    train_config.seed = 9;
    // The driver-built pool through the SHARED production pool-construction
    // function (`jitter_variants_of_batch` — what `ms2_experiment` calls):
    // every fixed variant plus the stored batch the evaluation draws use —
    // exactly the precursor rows training then uses. (Building an ad-hoc
    // pool here would pass even if the driver's pool construction reverted.)
    let base = spectrum_batch_for(&set, &indices, 64).unwrap();
    let mut pool: Vec<SpectrumBatch> = vec![base.clone()];
    // Task F9 item A3: the shared pool function yields variants LAZILY (one
    // at a time); collecting here is the test's choice, not the driver's.
    let variants: Vec<SpectrumBatch> = jitter_variants_of_batch(&base, &indices, 2.0, 9, 2).collect();
    assert_eq!(variants.len(), 2, "the shared pool function must emit V variants");
    // Each pool member is exactly its oracle draw.
    for (v, member) in variants.iter().enumerate() {
        for (b, &idx) in indices.iter().enumerate() {
            let want = jitter_precursor_mz(
                base.precursor_mz_udalton[b],
                2.0,
                9,
                1 + v as u64,
                idx as u64,
            );
            assert_eq!(
                member.precursor_mz_udalton[b], want,
                "pool variant {v} spectrum {idx} must be its oracle draw"
            );
        }
        pool.push(member.clone());
    }
    let table = FormulaTable::from_compositions(set_comps.iter().copied()).unwrap();
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_model(), &table, &train_config, &device).unwrap();
    let domain = EnumDomain::from_compositions(fixture_comps(), 0).unwrap();
    let bounds = RatioBounds::fit(fixture_comps(), 0).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    let mut cache = EnumCache::new(trainer.enum_cache_header(32).unwrap());
    trainer.build_enum_cache(pool.iter(), 32, &mut cache).unwrap();
    trainer.set_enum_cache(Some(Arc::new(cache))).unwrap();
    // Two epochs over the small export (four steps over the full two-spectrum
    // set): 100% hits, AND every step selects exactly its (step, spectrum)
    // draw — not just pool membership (task F9 item A7: the old test asserted
    // the exact variant for the first step only; a step that picked a
    // wrong-member draw passed it).
    trainer.capture_prep_batch(true);
    for step_tag in 1..=4u64 {
        trainer.step(&set, &indices).unwrap();
        let prep = trainer.last_prep_batch.as_ref().expect("prep captured");
        for (b, &idx) in indices.iter().enumerate() {
            let v = jitter_variant_index(9, step_tag, idx as u64, 2);
            let want = jitter_precursor_mz(
                base.precursor_mz_udalton[idx],
                2.0,
                9,
                1 + u64::from(v),
                idx as u64,
            );
            assert_eq!(
                prep.precursor_mz_udalton[b], want,
                "step {step_tag} spectrum {idx} must use exactly draw {v}"
            );
        }
    }
    let (lookups, hits, _, _) = trainer.enum_cache_stats();
    assert_eq!((lookups, hits), (4, 4), "the driver pool must cover training exactly");
}

#[test]
fn f7a_evidence_header_mismatch_refused() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds_a = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut wide = comps.clone();
    wide.push([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
    let bounds_b = RatioBounds::fit(wide, 0).unwrap();
    let (_table, dtable, _) = setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let mut rng_a = Rng::seeded(5);
    let mut rng_b = Rng::seeded(5);
    let mut cfg = tiny_model();
    cfg.formula_features = FormulaFeatures::Evidence;
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut model_a = Ms2Model::<R, E>::init(&cfg, &device, &mut rng_a).unwrap();
    let mut model_b = Ms2Model::<R, E>::init(&cfg, &device, &mut rng_b).unwrap();
    model_a.upload_enum_artifacts(&domain, &bounds_a, &device).unwrap();
    model_b.upload_enum_artifacts(&domain, &bounds_b, &device).unwrap();
    let batch = evidence_fixture_batch(&comps, 64);
    let gcfg = tiny_gen_evidence(32, 4096);
    let header_a = model_a.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header_a);
    model_a
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    assert!(cache.evidence_len() > 0, "the cache must carry evidence entries");
    // Attach under foreign bounds: refused before any entry (enumeration or
    // evidence) is served.
    let err = model_b
        .set_enum_cache(Some(Arc::new(cache.clone())))
        .expect_err("foreign evidence cache must be refused at attach");
    assert!(err.to_string().contains("`bounds_sha256`"), "got {err}");
    // A stale scored cap at use: refused, and the evidence stage never runs
    // on the stale header.
    model_a.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let mut other = tiny_gen_evidence(32, 4096);
    other.formula_rows_scored_max = 16;
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let err = model_a
        .generate(&batch, &dtable, &other, &mut ws, &constants)
        .expect_err("stale evidence cache must be refused at use");
    assert!(
        err.to_string().contains("`formula_rows_scored_max`"),
        "got {err}"
    );
    let (_, _, ev_lookups, ev_hits) = model_a.enum_cache_stats();
    assert_eq!((ev_lookups, ev_hits), (0, 0), "no evidence lookup on refusal");
}

// ---------------------------------------------------------------------------
// Task F8: evidence identity (n_peaks, dtype), verified hash hits, host
// memory budget, stale temporaries.
// ---------------------------------------------------------------------------

/// [`setup_model_with_layout`] with an explicit kept-peak capacity (task F8
/// item 1: the 17-peak spectrum needs models at `n_peaks` 16 and 32 over
/// identical artifacts).
fn setup_model_with_layout_and_peaks(
    device: &Device<R>,
    comps: &[Composition],
    seed: u64,
    layout: FormulaFeatures,
    n_peaks: u32,
) -> (FormulaTable, DeviceFormulaTable<R, E>, Ms2Model<R, E>) {
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, device).unwrap();
    let domain = EnumDomain::from_compositions(comps.to_vec(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.to_vec(), 0).unwrap();
    let mut cfg = tiny_model();
    cfg.n_peaks = n_peaks;
    cfg.formula_features = layout;
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(seed);
    let mut model = Ms2Model::<R, E>::init(&cfg, device, &mut rng).unwrap();
    model
        .upload_enum_artifacts(&domain, &bounds, device)
        .unwrap();
    (table, dtable, model)
}

/// One spectrum with 17 eligible peaks (task F8 item 1: more than 16, so the
/// kept-peak capacity decides what the evidence stage sees).
fn seventeen_peak_batch(comps: &[Composition], n_raw: usize) -> SpectrumBatch {
    let productive = precursor_of(&comps[6]);
    let mut mzs = vec![0u32; n_raw];
    let mut intensities = vec![0.0f32; n_raw];
    for k in 0..17 {
        mzs[k] = 60_000_000 + (k as u32) * 7_000_000;
        intensities[k] = 1.0 - 0.05 * k as f32;
    }
    evidence_batch(
        &[productive],
        &[50],
        &[1],
        &[0],
        &[0],
        &[50],
        &[17],
        &[mzs],
        &[intensities],
        n_raw,
    )
}

#[test]
fn f8_n_peaks_evidence_identity() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, _dtable, model16) =
        setup_model_with_layout_and_peaks(&device, &comps, 5, FormulaFeatures::Evidence, 16);
    let batch = seventeen_peak_batch(&comps, 64);
    // The reviewer's scenario, proved first: 17 eligible peaks give
    // different UNCACHED evidence at kept capacities 16 and 32 — so the
    // field matters and omitting it is incorrect, not just untested.
    let (ev16, _, peaks16) =
        device_evidence(&device, &model16, &batch, 32, 32, 65_536, 2048, 268435456, 16);
    let (ev32, _, peaks32) =
        device_evidence(&device, &model16, &batch, 32, 32, 65_536, 2048, 268435456, 32);
    assert_ne!(
        ev16, ev32,
        "17 peaks must give different uncached evidence at n_peaks 16 vs 32"
    );
    let stride = EVIDENCE_PEAKS * 4;
    assert_eq!(peaks16.len(), stride);
    assert_eq!(peaks32.len(), stride);
    let n16 = (0..EVIDENCE_PEAKS).filter(|&s| peaks16[s * 4 + 3] == 1).count();
    let n32 = (0..EVIDENCE_PEAKS).filter(|&s| peaks32[s * 4 + 3] == 1).count();
    assert_ne!(n16, n32, "valid evidence-peak counts must differ at 16 vs 32");
    // Build at `n_peaks = 16` ...
    let gcfg = tiny_gen_evidence(32, 4096);
    let header16 = model16.enum_cache_header(&gcfg).unwrap();
    assert_eq!(header16.n_peaks, 16, "the header must stamp the kept-peak capacity");
    let mut cache = EnumCache::new(header16);
    model16
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    assert!(cache.evidence_len() > 0, "the 17-peak build must store evidence");
    // ... and attach to a 32-kept-peak model over identical artifacts:
    // refused, naming `n_peaks` (reverting the attach check serves 16-peak
    // evidence to the 32-peak model).
    let (_t2, _d2, mut model32) =
        setup_model_with_layout_and_peaks(&device, &comps, 5, FormulaFeatures::Evidence, 32);
    let err = model32
        .set_enum_cache(Some(Arc::new(cache.clone())))
        .expect_err("n_peaks mismatch must be refused at attach");
    assert!(err.to_string().contains("`n_peaks`"), "got {err}");
    // Build enforcement: the 16-model cannot build into a 32-stamped cache.
    let header32 = model32.enum_cache_header(&gcfg).unwrap();
    assert_eq!(header32.n_peaks, 32);
    let mut cache32 = EnumCache::new(header32);
    let err = model16
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache32)
        .expect_err("n_peaks mismatch must be refused at build");
    assert!(err.to_string().contains("`n_peaks`"), "got {err}");
    // Lookup enforcement: the 16-built entries hit at 16 and miss at 32.
    let artifacts = model16.enum_artifacts.as_ref().unwrap();
    let meta = build_enum_meta(&batch, artifacts.domain_max_error, 65_536, 32);
    let (_, counters_host) = cache.expand_batch(&meta_keys(&meta), 32).expect("enum hit");
    let rows = vec![counters_host[2] as usize];
    let queries = evidence_queries_for(&batch, &meta, &[17], &rows, 2048, artifacts.hydrogen_cap_max());
    assert!(
        cache.expand_evidence_batch(&queries, 32, 16, TEST_DTYPE).is_some(),
        "same n_peaks must hit"
    );
    assert!(
        cache.expand_evidence_batch(&queries, 32, 32, TEST_DTYPE).is_none(),
        "foreign n_peaks must miss (device path runs)"
    );
}

#[test]
fn f8_evidence_dtype_policy() {
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (_table, _dtable, model) =
        setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let batch = evidence_fixture_batch(&comps, 64);
    let gcfg = tiny_gen_evidence(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    assert_eq!(header.evidence_dtype, "f32", "the header must stamp the evidence dtype");
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    assert!(cache.evidence_len() > 0);
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let meta = build_enum_meta(&batch, artifacts.domain_max_error, 65_536, 32);
    let keys = meta_keys(&meta);
    let (_, counters_host) = cache.expand_batch(&keys, 32).expect("enum hit");
    let rows: Vec<usize> = (0..3).map(|i| counters_host[i * 5 + 2] as usize).collect();
    let uploaded = vec![8u32, 8, 0];
    let queries = evidence_queries_for(&batch, &meta, &uploaded, &rows, 2048, artifacts.hydrogen_cap_max());
    // Policy (documented in the module docs): a cache built under f32 serves
    // ENUMERATION hits under bf16 and evidence MISSES — the step runs the
    // device evidence stage, so exactness is preserved.
    assert!(
        cache.expand_evidence_batch(&queries, 32, TEST_N_PEAKS, "f32").is_some(),
        "same dtype must hit"
    );
    assert!(
        cache.expand_evidence_batch(&queries, 32, TEST_N_PEAKS, "bf16").is_none(),
        "evidence must miss under another dtype"
    );
    assert!(
        cache.expand_batch(&keys, 32).is_some(),
        "integer enumeration entries stay usable across dtypes"
    );
    // Build enforcement: a bf16 evidence build into an f32-stamped cache is
    // refused naming the field (fires before any device work).
    let (cand_host, counters_host) = cache.expand_batch(&keys, 32).expect("enum hit");
    let mut cache_f32 = EnumCache::new(model.enum_cache_header(&gcfg).unwrap());
    let err = run_device_evidence_into::<R, half::bf16>(
        &device,
        &batch,
        &keys,
        &counters_host,
        &cand_host,
        32,
        2048,
        268435456,
        artifacts.hydrogen_cap_max(),
        16,
        &mut cache_f32,
    )
    .expect_err("bf16 build into an f32 cache must be refused");
    assert!(err.to_string().contains("`evidence_dtype`"), "got {err}");
    // Capability probe: a minimal bf16 device round-trip on this runtime.
    // Where supported the branch above already ran; where not, the refusal
    // itself is asserted here instead of skipping silently.
    match Tensor::<R, half::bf16>::from_f32(&[1.0, 2.0], vec![2], &device) {
        Ok(t) => assert_eq!(t.shape().dims(), &[2]),
        Err(e) => assert!(
            !e.to_string().is_empty(),
            "a bf16 capability refusal must still report"
        ),
    }
}

#[test]
fn f8_evidence_collision_buckets() {
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        32,
        4096,
        65_536,
        16,
        "f32",
    );
    // Two different uploaded rows (m/z uncertainty 50 vs unknown: different
    // canonical inputs) forced onto the SAME 128-bit key via the doc-hidden
    // hook — the injected collision of task F8 item 3.
    let comps = fixture_comps();
    let batch = evidence_fixture_batch(&comps, 64);
    let h_cap = 1023u32;
    let inputs_a = evidence_inputs_for_batch(&batch, 0, 8, 2048, h_cap);
    let inputs_b = evidence_inputs_for_batch(&batch, 1, 8, 2048, h_cap);
    let canon_a = evidence_canonical_bytes(&inputs_a);
    let canon_b = evidence_canonical_bytes(&inputs_b);
    assert_ne!(canon_a, canon_b, "the two rows must differ canonically");
    let key = evidence_key_with_forced_hash([9u32; 8], 0x1111_2222_3333_4444, 0x5555_6666_7777_8888);
    let sum_a: u64 = inputs_a.mz_row.iter().map(|&m| u64::from(m)).sum();
    let sum_b: u64 = inputs_b.mz_row.iter().map(|&m| u64::from(m)).sum();
    let mut cache = EnumCache::new(header.clone());
    assert!(
        cache
            .insert_evidence(
                key,
                8,
                sum_a,
                7,
                vec![3, 5],
                vec![1.5f32.to_bits(), 2.5f32.to_bits()],
                vec![1, 0],
                32,
                canon_a.clone(),
            )
            .unwrap(),
        "first colliding insert must store"
    );
    assert!(
        cache
            .insert_evidence(
                key,
                8,
                sum_b,
                7,
                vec![4, 6],
                vec![3.5f32.to_bits(), 4.5f32.to_bits()],
                vec![1, 1],
                32,
                canon_b.clone(),
            )
            .unwrap(),
        "second colliding insert must coexist, not overwrite"
    );
    assert_eq!(cache.evidence_len(), 2, "colliding keys must coexist");
    // Both are then served correctly through the verified lookup.
    let ra = cache.get_evidence(&key, &inputs_a).expect("row A must be served");
    assert_eq!(ra.explained, &[3, 5], "row A served row B's entry");
    assert_eq!(ra.weight_bits, &[1.5f32.to_bits(), 2.5f32.to_bits()]);
    let rb = cache.get_evidence(&key, &inputs_b).expect("row B must be served");
    assert_eq!(rb.explained, &[4, 6], "row B served row A's entry");
    assert_eq!(rb.complete, &[1, 1]);
    // The cross-bucket scan counted at least one verified-hash mismatch
    // (reverting verification to hash-only serves silently and counts
    // nothing).
    assert!(
        cache.collisions() >= 1,
        "a verified-hash mismatch must count a collision"
    );
    // Batch expansion serves the right member per row.
    let queries = [
        EvidenceQuery { key, rows_scored: 2, inputs: inputs_a },
        EvidenceQuery { key, rows_scored: 2, inputs: inputs_b },
    ];
    let out = cache
        .expand_evidence_batch(&queries, 32, 16, "f32")
        .expect("both rows hit");
    assert_eq!(out[0], 3.0, "row 0 explained must be A's");
    assert_eq!(out[32 * 4], 4.0, "row 1 explained must be B's");
    assert_eq!(out[1].to_bits(), 1.5f32.to_bits());
    assert_eq!(out[32 * 4 + 1].to_bits(), 3.5f32.to_bits());
    // Save → load preserves both bucket members.
    let dir = std::env::temp_dir().join("ms2_f8_collision");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    cache.save(&path).unwrap();
    let loaded = EnumCache::load(&path, &header).unwrap();
    assert_eq!(loaded.evidence_len(), 2, "both bucket members must survive save → load");
    assert_eq!(
        loaded.get_evidence(&key, &inputs_a).unwrap().explained,
        &[3, 5]
    );
    assert_eq!(
        loaded.get_evidence(&key, &inputs_b).unwrap().explained,
        &[4, 6]
    );
    std::fs::remove_file(&path).ok();
    // Memory cost per evidence entry (task F8 item 3): key 48 + canonical
    // `36 + 8 * slots` + explained/weight/complete vectors + overhead (see
    // `resident_bytes`). Before item 3 the canonical bytes did not exist.
    println!(
        "EVIDENCE entry bytes: canonical {} + explained 2 + weights 8 + complete 2 (slots 8, rows 2)",
        canon_a.len()
    );
    assert_eq!(canon_a.len(), 36 + 8 * 8);
}

#[test]
fn f8_resident_bytes_and_budget() {
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        32,
        4096,
        65_536,
        16,
        "f32",
    );
    let mut cache = EnumCache::with_max_resident_bytes(header, 1000);
    assert_eq!(cache.max_resident_bytes(), 1000);
    // Scored-0 entries are valid inserts (no candidate words).
    assert!(
        cache.insert([1u32; 8], [0, 0, 0, 0, 1], &[], 32).unwrap(),
        "first insert must store"
    );
    let resident = cache.resident_bytes();
    assert!(resident > 0, "resident accounting must observe the entry");
    println!("BUDGET resident {resident} file {} after one entry", cache.bytes());
    // Pin the budget at the current footprint: the next insert no longer
    // fits and is REFUSED (counted; the row stays uncached).
    cache.set_max_resident_bytes(resident as u64);
    assert!(
        !cache.insert([2u32; 8], [0, 0, 0, 0, 1], &[], 32).unwrap(),
        "over-budget insert must be refused, not stored"
    );
    assert_eq!(cache.budget_refusals(), 1, "refusals must be counted");
    assert_eq!(cache.len(), 1, "the refused row stays uncached");
    assert_eq!(cache.resident_bytes(), resident, "a refusal must not grow the footprint");
    // The refused rows run uncached at the model level with identical
    // results: a zero budget caches nothing, misses everything, and still
    // agrees with the device path.
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
    let mut poor = EnumCache::with_max_resident_bytes(model.enum_cache_header(&gcfg).unwrap(), 0);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut poor)
        .unwrap();
    assert_eq!(poor.len(), 0, "zero budget stores nothing");
    assert!(poor.budget_refusals() > 0, "zero budget refuses everything");
    model.set_enum_cache(Some(Arc::new(poor))).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let missed = model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    let (lookups, hits, _, _) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (1, 0), "budget-starved steps miss and run uncached");
    model.set_enum_cache(None).unwrap();
    let mut ws2 = GenerationWorkspace::new();
    let plain = model.generate(&batch, &dtable, &gcfg, &mut ws2, &constants).unwrap();
    assert_candidate_batch_close(&missed, &plain);
}

#[test]
fn f8_stale_temp_files_listed_not_deleted() {
    let _serial = serial();
    let dir = std::env::temp_dir().join("ms2_f8_stale");
    std::fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("cache.bin");
    std::fs::write(&dest, b"data").unwrap();
    let t1 = dir.join("cache.bin.tmp-1-2-3");
    let t2 = dir.join("cache.bin.tmp-9-9-9");
    let other = dir.join("other.bin.tmp-1-2-3");
    std::fs::write(&t1, b"x").unwrap();
    std::fs::write(&t2, b"y").unwrap();
    std::fs::write(&other, b"z").unwrap();
    // Orphans of THIS destination are listed (sorted), nothing is deleted.
    let stale = EnumCache::stale_temp_files(&dest);
    assert_eq!(stale, vec![t1.clone(), t2.clone()]);
    assert!(t1.exists() && t2.exists(), "stale files are printed, never auto-deleted");
    assert!(!stale.contains(&other), "another destination's temporary must not match");
    // An unreadable directory yields an empty list, never an error.
    assert!(EnumCache::stale_temp_files(&dir.join("nope").join("cache.bin")).is_empty());
    std::fs::remove_file(&t1).ok();
    std::fs::remove_file(&t2).ok();
    std::fs::remove_file(&other).ok();
    std::fs::remove_file(&dest).ok();
}

// ---------------------------------------------------------------------------
// Task F9, Part A.
// ---------------------------------------------------------------------------

/// Tiny deterministic xorshift64* for the randomised budget test: the
/// sequence is fixed by the seed, so the test is reproducible.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn f9_valid_cand_row(counts: &[u16; 10], flag: u32) -> [u32; 13] {
    let mass = composition_mass(counts).unwrap();
    let mut row = [0u32; 13];
    for (i, &c) in counts.iter().enumerate() {
        row[i] = u32::from(c);
    }
    row[10] = mass;
    row[11] = flag;
    row[12] = u32::MAX;
    row
}

#[test]
fn f9_prospective_budget_invariant_randomized() {
    // Task F9 item A2, re-homed on physical buckets by task F10 item B1:
    // admission is the prospective bucket-based footprint (the same
    // function `resident_bytes()` reports, evaluated for the state after
    // the insert). Over a randomised sequence of 2,000 enum/evidence
    // inserts and replacements at several budgets: every accepted insert
    // leaves `resident_bytes() <= max_resident_bytes`, every refusal is
    // counted exactly once, and after EVERY operation the tracked buckets
    // cover the buckets implied by the live usable capacities. Evidence
    // metas deliberately reuse replaced enumeration metas, so replacements
    // evict evidence (tombstones) on the way — the test asserts evictions
    // actually happened, by count.
    let _serial = serial();
    let header = || {
        EnumCacheHeader::new(
            "domain".to_string(),
            "bounds".to_string(),
            4,
            32,
            4096,
            65_536,
            16,
            "f32",
        )
    };
    // Valid scored rows (mass-checked at insert): scored 1..=3 plus scored 0.
    let row1 = f9_valid_cand_row(&[1, 4, 0, 0, 0, 0, 0, 0, 0, 0], 1);
    let row2a = f9_valid_cand_row(&[1, 4, 0, 0, 0, 0, 0, 0, 0, 0], 1);
    let row2b = f9_valid_cand_row(&[2, 6, 0, 1, 0, 0, 0, 0, 0, 0], 2);
    let row3a = f9_valid_cand_row(&[1, 4, 0, 0, 0, 0, 0, 0, 0, 0], 2);
    let row3b = f9_valid_cand_row(&[2, 6, 0, 1, 0, 0, 0, 0, 0, 0], 1);
    let row3c = f9_valid_cand_row(&[6, 6, 0, 0, 0, 0, 0, 0, 0, 0], 1);
    // One structurally valid canonical evidence input (reused under varying
    // keys: `insert_evidence` validates the canonical bytes, never the
    // key/canonical pairing).
    let comps = fixture_comps();
    let ev_batch = evidence_batch(
        &[precursor_of(&comps[4])],
        &[50],
        &[1],
        &[0],
        &[0],
        &[50],
        &[8],
        &[vec![60_000_000; 64]],
        &[vec![1.0; 64]],
        64,
    );
    let ev_inputs = evidence_inputs_for_batch(&ev_batch, 0, 8, 2048, 1023);
    let ev_canon = evidence_canonical_bytes(&ev_inputs);
    let ev_mz_sum: u64 = ev_inputs.mz_row.iter().map(|&m| u64::from(m)).sum();
    let ev_peak_count = ev_inputs.peak_count;
    for &budget in &[0u64, 124, 500, 5_000, 100_000, u64::MAX] {
        let mut cache = EnumCache::with_max_resident_bytes(header(), budget);
        let mut rng = XorShift(0x243F_6A88_85A3_08D3);
        let mut keys: Vec<[u32; 8]> = Vec::new();
        let mut ev_keys: Vec<EvidenceKey> = Vec::new();
        let mut refusals = 0u64;
        let mut evictions_observed = 0u64;
        // After every operation: the budget invariant and the bucket
        // invariant (task F10 item B1).
        let check_invariants = |cache: &EnumCache, budget: u64, step: usize, what: &str| {
            assert!(
                (cache.resident_bytes() as u64) <= budget,
                "budget {budget} step {step}: {what} leaves resident {} over budget",
                cache.resident_bytes()
            );
            assert!(
                cache.bucket_invariant_holds(),
                "budget {budget} step {step}: {what} breaks the bucket invariant (tracked {:?}, live capacities {:?})",
                cache.table_buckets(),
                cache.table_capacities()
            );
        };
        for step in 0..2_000 {
            let op = rng.next() % 100;
            if op < 55 {
                // Enum insert: 70% new key, 30% replacement of a live key.
                let replace = !keys.is_empty() && rng.next() % 100 < 30;
                let key = if replace {
                    keys[(rng.next() as usize) % keys.len()]
                } else {
                    let mut k = [0u32; 8];
                    for w in k.iter_mut() {
                        *w = (rng.next() & 0xFFFF_FFFF) as u32;
                    }
                    k
                };
                let scored = (rng.next() % 4) as usize;
                let words: Vec<u32> = match scored {
                    0 => Vec::new(),
                    1 => row1.to_vec(),
                    2 => [row2a.as_slice(), row2b.as_slice()].concat(),
                    _ => [row3a.as_slice(), row3b.as_slice(), row3c.as_slice()].concat(),
                };
                let counters = [7, 5, scored as u32, 0, 1];
                let ev_before = cache.evidence_len();
                let stored = cache.insert(key, counters, &words, 32).unwrap();
                if stored {
                    check_invariants(&cache, budget, step, "accepted enum insert");
                    // A replacement evicts every evidence entry sharing
                    // the meta: count the evictions so the test proves
                    // tombstones (removals) actually occurred below.
                    if replace {
                        evictions_observed += (ev_before as u64).saturating_sub(cache.evidence_len() as u64);
                    }
                    if !replace {
                        keys.push(key);
                    }
                } else {
                    refusals += 1;
                    check_invariants(&cache, budget, step, "refused enum insert");
                }
            } else {
                // Evidence insert: new or colliding forced-hash key, rows
                // 1..=3, or replacement of a live key/canonical pair. The
                // meta reuses a live enumeration meta one time in three, so
                // enum replacements above evict evidence (task F10 item B1:
                // the randomised tombstone coupling).
                let replace =
                    !ev_keys.is_empty() && rng.next() % 100 < 30;
                let key = if replace {
                    ev_keys[(rng.next() as usize) % ev_keys.len()]
                } else if !keys.is_empty() && rng.next() % 3 == 0 {
                    evidence_key_with_forced_hash(
                        keys[(rng.next() as usize) % keys.len()],
                        rng.next(),
                        rng.next(),
                    )
                } else {
                    evidence_key_with_forced_hash(
                        {
                            let mut m = [0u32; 8];
                            for w in m.iter_mut() {
                                *w = (rng.next() & 0xFFFF_FFFF) as u32;
                            }
                            m
                        },
                        rng.next(),
                        rng.next(),
                    )
                };
                let rows = 1 + (rng.next() % 3) as usize;
                let was_new = !replace;
                let stored = cache
                    .insert_evidence(
                        key,
                        ev_peak_count,
                        ev_mz_sum,
                        3,
                        vec![1u8; rows],
                        vec![1.0f32.to_bits(); rows],
                        vec![1u8; rows],
                        32,
                        ev_canon.clone(),
                    )
                    .unwrap();
                if stored {
                    check_invariants(&cache, budget, step, "accepted evidence insert");
                    if was_new {
                        ev_keys.push(key);
                    }
                } else {
                    refusals += 1;
                    check_invariants(&cache, budget, step, "refused evidence insert");
                }
            }
        }
        assert_eq!(
            cache.budget_refusals(),
            refusals,
            "budget {budget}: every refusal must be counted exactly once"
        );
        println!(
            "BUDGET invariant budget={budget}: entries={} evidence={} resident={} refusals={refusals} evictions={evictions_observed} buckets={:?}",
            cache.len(),
            cache.evidence_len(),
            cache.resident_bytes(),
            cache.table_buckets()
        );
        if budget == u64::MAX {
            assert!(
                evictions_observed > 0,
                "the unbounded run must evict evidence through enum replacements (tombstones exercised)"
            );
        }
    }
    // The reviewer's 124-byte case: an empty cache, 124-byte budget, a
    // zero-scored enumeration entry — refused, or within budget either way.
    let mut tiny = EnumCache::with_max_resident_bytes(header(), 124);
    let stored = tiny.insert([9u32; 8], [0, 0, 0, 0, 1], &[], 32).unwrap();
    assert!(
        !stored || (tiny.resident_bytes() as u64) <= 124,
        "the 124-byte case must be refused or stay within budget (resident {})",
        tiny.resident_bytes()
    );
    println!(
        "BUDGET 124-byte case: stored={stored} resident={}",
        tiny.resident_bytes()
    );
}

#[test]
fn f9_enum_replace_evicts_evidence() {
    // Task F9 item A5: replacing an enumeration entry invalidates the
    // evidence entries keyed with its meta (they were computed for the old
    // candidates). Host-only: no device work, no randomness.
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        32,
        4096,
        65_536,
        16,
        "f32",
    );
    let mut cache = EnumCache::new(header);
    let key = [7u32; 8];
    let row_a = f9_valid_cand_row(&[1, 4, 0, 0, 0, 0, 0, 0, 0, 0], 1);
    assert!(
        cache.insert(key, [7, 5, 1, 0, 1], &row_a, 32).unwrap(),
        "first insert must store"
    );
    // Evidence for K over a real uploaded row.
    let comps = fixture_comps();
    let batch = evidence_batch(
        &[precursor_of(&comps[4])],
        &[50],
        &[1],
        &[0],
        &[0],
        &[50],
        &[8],
        &[vec![60_000_000; 64]],
        &[vec![1.0; 64]],
        64,
    );
    let (ev_key, _) = evidence_key_for_batch(&batch, 0, 8, key, 2048, 1023);
    assert_eq!(ev_key.meta, key, "the evidence key must carry K as its meta");
    let inputs = evidence_inputs_for_batch(&batch, 0, 8, 2048, 1023);
    let mz_sum: u64 = inputs.mz_row.iter().map(|&m| u64::from(m)).sum();
    let peak_count = inputs.peak_count;
    let canonical = evidence_canonical_bytes(&inputs);
    assert!(
        cache
            .insert_evidence(
                ev_key,
                peak_count,
                mz_sum,
                3,
                vec![2u8],
                vec![1.5f32.to_bits()],
                vec![1u8],
                32,
                canonical,
            )
            .unwrap(),
        "evidence insert must store"
    );
    let inputs_again = evidence_inputs_for_batch(&batch, 0, 8, 2048, 1023);
    assert!(
        cache.get_evidence(&ev_key, &inputs_again).is_some(),
        "evidence must be served before the replacement"
    );
    // Replace K's candidates through the public overwrite API: different
    // valid candidates, same scored count.
    let row_b = f9_valid_cand_row(&[2, 6, 0, 1, 0, 0, 0, 0, 0, 0], 2);
    assert_ne!(row_a, row_b, "the replacement must differ");
    assert!(
        cache.insert(key, [7, 5, 1, 0, 0], &row_b, 32).unwrap(),
        "replacement insert must store"
    );
    let inputs_after = evidence_inputs_for_batch(&batch, 0, 8, 2048, 1023);
    assert!(
        cache.get_evidence(&ev_key, &inputs_after).is_none(),
        "old evidence must MISS after K's candidates were replaced"
    );
    assert_eq!(
        cache.evidence_len(),
        0,
        "the invalidated bucket must be gone"
    );
    println!("A5 replace evicts evidence: miss after overwrite");
}

#[test]
fn f9_load_budget_refused_then_adequate() {
    // Task F9 item A4: the budget governs loading. A cache saved under a
    // large budget is refused under a small one (before reading when the
    // file size alone exceeds it; during parsing when the running estimate
    // does) and loads under an adequate one — never partially.
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        32,
        4096,
        65_536,
        16,
        "f32",
    );
    let mut cache = EnumCache::new(header.clone());
    for i in 0u32..50 {
        let mut k = [0u32; 8];
        k[0] = i;
        cache.insert(k, [7, 5, 0, 0, 1], &[], 32).unwrap();
    }
    assert_eq!(cache.len(), 50);
    let dir = std::env::temp_dir().join("ms2_f9_load_budget");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    cache.save(&path).unwrap();
    let file_len = std::fs::metadata(&path).unwrap().len();
    let loaded = EnumCache::load(&path, &header).unwrap();
    let resident = loaded.resident_bytes() as u64;
    assert_eq!(loaded.len(), 50, "default load must serve every entry");
    println!("LOAD budget: file {file_len} bytes, resident {resident} bytes");
    assert!(
        resident > file_len,
        "the fixture needs resident > file size to separate the two refusal stages (resident {resident}, file {file_len})"
    );
    // Stage 1: the file size alone exceeds the budget — refused before
    // reading, naming the budget and the size.
    let err = EnumCache::load_with_budget(&path, &header, 1).expect_err("tiny budget must refuse");
    let msg = err.to_string();
    assert!(msg.contains("budget"), "refusal must name the budget, got {msg}");
    assert!(msg.contains(&file_len.to_string()), "refusal must name the size, got {msg}");
    // Stage 2: the file fits but the running resident estimate does not —
    // refused during parsing, naming the budget and the size.
    let err = EnumCache::load_with_budget(&path, &header, file_len)
        .expect_err("file-sized budget must refuse a larger resident cache");
    let msg = err.to_string();
    assert!(msg.contains("budget"), "refusal must name the budget, got {msg}");
    assert!(msg.contains("resident"), "refusal must name the resident size, got {msg}");
    // Boundary: exactly the resident size loads; one byte less refuses.
    let ok = EnumCache::load_with_budget(&path, &header, resident).unwrap();
    assert_eq!(ok.len(), 50, "an adequate budget must load every entry");
    assert_eq!(ok.max_resident_bytes(), resident, "the loaded cache carries the budget");
    assert!(
        EnumCache::load_with_budget(&path, &header, resident - 1).is_err(),
        "one byte under resident must refuse"
    );
    std::fs::remove_file(&path).ok();
}

/// Evidence-layout trainer over two fixture spectra, jitter off (steps use
/// the stored precursors, so the built cache covers them).
fn f9_evidence_trainer(
    device: &Device<R>,
    comps: &[Composition],
) -> (Ms2Trainer<R, E>, ExperimentSet, Vec<usize>) {
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        &[0, 0],
        64,
    );
    let set = experiment_set(&vec![comps[4], comps[6]], &batch);
    let indices = vec![0usize, 1];
    let mut model_cfg = tiny_model();
    model_cfg.formula_features = FormulaFeatures::Evidence;
    let table = FormulaTable::from_compositions(vec![comps[4], comps[6]].into_iter()).unwrap();
    let mut train_config = tiny_train();
    train_config.precursor_jitter_ppm = 0.0;
    train_config.precursor_jitter_variants = 0;
    train_config.seed = 11;
    let mut trainer =
        Ms2Trainer::<R, E>::new(&model_cfg, &table, &train_config, device).unwrap();
    let domain = EnumDomain::from_compositions(comps.to_vec(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.to_vec(), 0).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    (trainer, set, indices)
}

/// Bit-exact weights snapshot through the public state dict.
fn f9_weights_json<R2: cubecl::prelude::Runtime, E2: mamba3::backend::FloatElem>(
    trainer: &Ms2Trainer<R2, E2>,
) -> String
where
    Ms2Model<R2, E2>: mamba3::nn::Module<R2, E2>,
{
    use mamba3::nn::Module;
    serde_json::to_string(&trainer.model.state_dict()).unwrap()
}

#[test]
fn f9_evidence_build_survives_enum_refusal_zero_budget() {
    // Task F9 item A1: with `FormulaFeatures::Evidence` and a ZERO budget,
    // cache construction SUCCEEDS (rows uncached, refusals counted) instead
    // of aborting on the missing enumeration — and the full step (loss and,
    // on CPU, weights) equals the uncached step bit for bit.
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (mut trainer, set, indices) = f9_evidence_trainer(&device, &comps);
    let header = trainer.enum_cache_header(32).unwrap();
    let mut poor = EnumCache::with_max_resident_bytes(header, 0);
    let cache_batch = spectrum_batch_for(&set, &indices, 64).unwrap();
    trainer
        .build_enum_cache([&cache_batch].into_iter(), 32, &mut poor)
        .unwrap();
    assert_eq!(poor.len(), 0, "zero budget stores no enumeration");
    assert_eq!(poor.evidence_len(), 0, "zero budget stores no evidence");
    assert!(
        poor.budget_refusals() > 0,
        "zero budget must count its refusals"
    );
    trainer.set_enum_cache(Some(Arc::new(poor))).unwrap();
    trainer.request_report();
    let cached_rep = trainer.step(&set, &indices).unwrap().unwrap();
    let cached_weights = f9_weights_json(&trainer);
    let (lookups, hits, ev_lookups, ev_hits) = trainer.enum_cache_stats();
    assert_eq!((hits, ev_hits), (0, 0), "nothing can hit under zero budget");
    assert!(lookups > 0 && ev_lookups > 0, "the step must still look up");
    // Uncached twin: same seed, same config, no cache.
    let (mut plain_trainer, _, _) = f9_evidence_trainer(&device, &comps);
    plain_trainer.request_report();
    let plain_rep = plain_trainer.step(&set, &indices).unwrap().unwrap();
    let plain_weights = f9_weights_json(&plain_trainer);
    assert_eq!(cached_rep.loss, plain_rep.loss, "refused-cache loss must equal uncached");
    assert_eq!(cached_rep, plain_rep, "refused-cache report must equal uncached");
    if is_cpu() {
        assert_eq!(cached_weights, plain_weights, "refused-cache weights must equal uncached bit for bit");
    }
    println!(
        "A1 zero budget: refusals={} lookups={lookups}/{hits} ev={ev_lookups}/{ev_hits} loss={}",
        trainer.enum_cache_stats().0,
        cached_rep.loss
    );
}

#[test]
fn f9_evidence_build_survives_enum_refusal_partial_budget() {
    // Task F9 item A1, partial budget: admits one spectrum's rows, refuses
    // the other's — construction succeeds, counters report the refusals, and
    // the full step still equals the uncached step bit for bit (admitted
    // rows hit, refused rows run uncached).
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (mut trainer, set, indices) = f9_evidence_trainer(&device, &comps);
    // Size a budget between one spectrum's footprint and two: build each
    // spectrum alone under no limit and read its resident cost.
    let batch_a = spectrum_batch_for(&set, &[0], 64).unwrap();
    let batch_b = spectrum_batch_for(&set, &[1], 64).unwrap();
    let mut probe = EnumCache::new(trainer.enum_cache_header(32).unwrap());
    trainer.build_enum_cache([&batch_a].into_iter(), 32, &mut probe).unwrap();
    let cost_a = probe.resident_bytes() as u64;
    assert!(probe.len() > 0, "spectrum A must cache something");
    let mut probe_b = EnumCache::new(trainer.enum_cache_header(32).unwrap());
    trainer.build_enum_cache([&batch_b].into_iter(), 32, &mut probe_b).unwrap();
    let cost_b = probe_b.resident_bytes() as u64;
    assert!(probe_b.len() > 0, "spectrum B must cache something");
    // Fresh cache pinned at A's exact footprint: A fits, B is refused.
    let mut cache = EnumCache::with_max_resident_bytes(trainer.enum_cache_header(32).unwrap(), cost_a);
    trainer.build_enum_cache([&batch_a].into_iter(), 32, &mut cache).unwrap();
    assert!(cache.len() > 0, "A must be admitted");
    let refusals_before = cache.budget_refusals();
    trainer.build_enum_cache([&batch_b].into_iter(), 32, &mut cache).unwrap();
    assert!(
        cache.budget_refusals() > refusals_before,
        "B must be refused (costs {cost_a}/{cost_b})"
    );
    trainer.set_enum_cache(Some(Arc::new(cache))).unwrap();
    trainer.request_report();
    let cached_rep = trainer.step(&set, &indices).unwrap().unwrap();
    let cached_weights = f9_weights_json(&trainer);
    let (mut plain_trainer, _, _) = f9_evidence_trainer(&device, &comps);
    plain_trainer.request_report();
    let plain_rep = plain_trainer.step(&set, &indices).unwrap().unwrap();
    let plain_weights = f9_weights_json(&plain_trainer);
    assert_eq!(cached_rep.loss, plain_rep.loss, "partial-cache loss must equal uncached");
    assert_eq!(cached_rep, plain_rep, "partial-cache report must equal uncached");
    if is_cpu() {
        assert_eq!(cached_weights, plain_weights, "partial-cache weights must equal uncached bit for bit");
    }
    println!("A1 partial budget: cost_a={cost_a} cost_b={cost_b} loss={}", cached_rep.loss);
}

#[test]
fn f9_jitter_variants_lazy_one_alive() {
    // Task F9 item A3: `jitter_variants_of_batch` is lazy — one variant's
    // batch alive at a time. The counting allocator measures peak live host
    // bytes around (a) the eager pattern (all V variants retained, what the
    // old `Vec` return forced) and (b) the lazy driver loop (each variant
    // dropped before the next is materialised). The lazy peak must stay near
    // one variant while the eager peak holds all V — plus exactness: every
    // lazy variant equals its oracle draw.
    use std::sync::atomic::Ordering;
    let _serial = serial();
    let comps = fixture_comps();
    let precursors: Vec<u32> = (0..8).map(|i| precursor_of(&comps[i % comps.len()])).collect();
    let uncs = vec![50u32; 8];
    let batch = spectrum_batch(&precursors, &uncs, &[1, 1, 1, 1, 1, 1, 1, 1], &[0; 8], 256);
    let indices: Vec<usize> = (0..8).collect();
    let variants = 16u32;
    // One variant's heap bytes, measured directly.
    ALLOC_COUNTING.store(1, Ordering::Relaxed);
    let live_before = ALLOC_LIVE.load(Ordering::Relaxed);
    let one = {
        let mut v = batch.clone();
        apply_precursor_jitter(&mut v, &indices, 2.0, 9, 1);
        v
    };
    let variant_bytes = ALLOC_LIVE.load(Ordering::Relaxed).saturating_sub(live_before);
    drop(one);
    ALLOC_COUNTING.store(0, Ordering::Relaxed);
    assert!(variant_bytes > 0, "a variant must own heap bytes");
    // (a) Eager: retain all V (the old return forced this retention).
    ALLOC_LIVE.store(0, Ordering::Relaxed);
    ALLOC_PEAK.store(0, Ordering::Relaxed);
    ALLOC_COUNTING.store(1, Ordering::Relaxed);
    let eager: Vec<SpectrumBatch> = (0..variants)
        .map(|v| {
            let mut jb = batch.clone();
            apply_precursor_jitter(&mut jb, &indices, 2.0, 9, 1 + u64::from(v));
            jb
        })
        .collect();
    let eager_peak = ALLOC_PEAK.load(Ordering::Relaxed);
    ALLOC_COUNTING.store(0, Ordering::Relaxed);
    assert!(
        eager_peak >= 8 * variant_bytes,
        "eager must retain many variants at once (peak {eager_peak}, one {variant_bytes})"
    );
    // (b) Lazy driver loop: one variant alive at a time.
    ALLOC_LIVE.store(0, Ordering::Relaxed);
    ALLOC_PEAK.store(0, Ordering::Relaxed);
    ALLOC_COUNTING.store(1, Ordering::Relaxed);
    let mut seen = 0u32;
    for variant in jitter_variants_of_batch(&batch, &indices, 2.0, 9, variants) {
        // Consume without retaining: the variant drops at iteration end.
        let mut acc = 0u64;
        for &mz in variant.precursor_mz_udalton.iter() {
            acc = acc.wrapping_add(u64::from(mz));
        }
        std::hint::black_box(acc);
        seen += 1;
    }
    drop(batch);
    let lazy_peak = ALLOC_PEAK.load(Ordering::Relaxed);
    ALLOC_COUNTING.store(0, Ordering::Relaxed);
    assert_eq!(seen, variants, "the lazy pool must yield every variant");
    // The lazy loop holds the iterator's base clone plus one variant (~2
    // units); the eager pool holds V (~16). Demand a 4x separation.
    assert!(
        lazy_peak * 4 < eager_peak.max(1),
        "lazy peak {lazy_peak} must stay far below eager peak {eager_peak} (one {variant_bytes})"
    );
    // Exactness: every lazy variant is its oracle draw (not just pool
    // membership).
    let batch2 = spectrum_batch(&precursors, &uncs, &[1, 1, 1, 1, 1, 1, 1, 1], &[0; 8], 256);
    for (v, member) in jitter_variants_of_batch(&batch2, &indices, 2.0, 9, variants).enumerate() {
        for (b, &idx) in indices.iter().enumerate() {
            let want = jitter_precursor_mz(
                batch2.precursor_mz_udalton[idx],
                2.0,
                9,
                1 + v as u64,
                idx as u64,
            );
            assert_eq!(
                member.precursor_mz_udalton[b], want,
                "lazy variant {v} spectrum {idx} must be its oracle draw"
            );
        }
    }
    drop(eager);
    println!("LAZY jitter: one={variant_bytes} eager_peak={eager_peak} lazy_peak={lazy_peak}");
}

#[test]
fn f9_dispatch_invariance_productive_incomplete() {
    // Task F9 item A7: dispatch invariance with a PRODUCTIVE, INCOMPLETE
    // enumeration row (non-zero scored candidates, `complete == 0`) — the
    // old fixture's far-precursor row scores nothing, so incompleteness was
    // never exercised. Found by sweeping small lane-visit budgets over a
    // productive precursor; both are asserted, then the cached buffers and
    // the served search are proved equal across dispatch bounds.
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    // A dense enum domain (C/H/N/O grid, rich in isobars) plus a WIDE
    // precursor tolerance so a productive precursor joins MANY candidates:
    // with the 8-composition fixture domain (and zero tolerance) every walk
    // joins at most one candidate and always completes. The formula table
    // stays small (generate only needs it for init); enumeration runs off
    // the dense artifacts.
    let mut dense: Vec<Composition> = comps.clone();
    for c in 0..6u16 {
        for h in (0..20u16).step_by(2) {
            for n in 0..4u16 {
                for o in 0..6u16 {
                    dense.push([c, h, n, o, 0, 0, 0, 0, 0, 0]);
                }
            }
        }
    }
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let domain = EnumDomain::from_compositions(dense.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(dense.clone(), 0).unwrap();
    let mut model_cfg = tiny_model();
    model_cfg.formula_table.rows = dtable.rows as u32;
    model_cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, E>::init(&model_cfg, &device, &mut rng).unwrap();
    model.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    let productive4 = precursor_of(&comps[4]);
    let productive6 = precursor_of(&comps[6]);
    let lanes = 2 * model.enum_artifacts.as_ref().unwrap().p;
    // A productive, INCOMPLETE row via scored-cap truncation: when the walk
    // joins more than `scored_cap`, the excess is truncated (`complete ==
    // 0`) while the kept `scored_cap` rows are non-zero. Sweep small caps
    // over the productive precursors.
    let mut found: Option<(u32, Vec<u32>, Vec<u32>, SpectrumBatch)> = None;
    'outer: for &precursor in &[productive4, productive6] {
        for &tol in &[1000u16, 3000, 8000] {
            let batch =
                spectrum_batch(&[precursor, 4_000_000_000], &[50, 50], &[1, 1], &[tol, 0], 64);
            for &scored_cap in &[1u32, 2, 4, 8] {
                let dispatch = lanes as u32 * 65_536;
                let (cand, counters) =
                    device_enumerate(&device, &model, &batch, scored_cap, 32, 65_536, dispatch);
                let scored = counters[2] as usize;
                let complete = counters[4];
                println!("INCOMPLETE sweep precursor={precursor} tol={tol} scored_cap={scored_cap}: scored={scored} complete={complete}");
                if scored > 0 && complete == 0 {
                    found = Some((scored_cap, cand, counters, batch));
                    break 'outer;
                }
            }
        }
    }
    let (scored_cap, cand_raw, counters_raw, batch) = found.expect(
        "a productive precursor under a small scored cap must keep rows without completing",
    );
    let lane_visits = 65_536u32;
    // Raw buffers agree across dispatch bounds (one lane at a time vs all).
    let per_a = enum_lanes_per_dispatch(lane_visits, lane_visits);
    assert_eq!(per_a, 1, "bound A must dispatch one lane at a time");
    let (cand_a, counters_a) =
        device_enumerate(&device, &model, &batch, scored_cap, 32, lane_visits, lane_visits);
    assert_eq!(cand_a, cand_raw, "raw cand differs across dispatch bounds");
    assert_eq!(counters_a, counters_raw, "raw counters differ across dispatch bounds");
    // The cache serves the incomplete row's raw buffers exactly.
    let mut gcfg = tiny_gen(32, scored_cap);
    gcfg.enum_lane_visits_max = lane_visits;
    gcfg.enum_dispatch_visits_max = lanes as u32 * lane_visits;
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header);
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    let domain_max_error = model.enum_artifacts.as_ref().unwrap().domain_max_error;
    let meta = build_enum_meta(&batch, domain_max_error, lane_visits, scored_cap);
    let (cand_hit, counters_hit) = cache.expand_batch(&meta_keys(&meta), 32).expect("all hit");
    assert_eq!(cand_hit, cand_raw, "cached cand differs on the incomplete row");
    assert_eq!(counters_hit, counters_raw, "cached counters differ on the incomplete row");
    // The served search equals the uncached one.
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws_hit = GenerationWorkspace::new();
    let hit = model.generate(&batch, &dtable, &gcfg, &mut ws_hit, &constants).unwrap();
    model.set_enum_cache(None).unwrap();
    let mut ws_plain = GenerationWorkspace::new();
    let plain = model.generate(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
    assert_candidate_batch_close(&hit, &plain);
    println!("INCOMPLETE productive row at scored_cap={scored_cap}: served exactly");
}

#[test]
fn f9_bf16_cached_evidence_step_or_capability_refusal() {
    // Task F9 item A7: one full cached evidence step on an attached bf16
    // model (CPU runtime). The cache is built by the f32 model, then attached
    // to the bf16 model over identical artifacts: integer enumeration entries
    // stay reusable (enumeration HITS), evidence always misses its dtype
    // check (the device evidence stage runs), and the cached bf16 step equals
    // the uncached bf16 step. Where bf16 generation is unsupported on this
    // runtime, the test asserts the capability REFUSAL instead of skipping
    // silently.
    use half::bf16 as BF16;
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    type EB = BF16;
    // bf16 model over identical artifacts (mirrors `setup_model_with_layout`
    // at another dtype).
    let bf16_setup: Result<
        (
            DeviceFormulaTable<R, EB>,
            Ms2Model<R, EB>,
            SpectrumBatch,
            GenerationConfig,
        ),
        mamba3::error::Error,
    > = (|| {
        let table = FormulaTable::from_compositions(comps.iter().copied())?;
        let dtable = DeviceFormulaTable::<R, EB>::upload(&table, &device)?;
        let domain = EnumDomain::from_compositions(comps.to_vec(), 0)?;
        let bounds = RatioBounds::fit(comps.to_vec(), 0)?;
        let mut cfg = tiny_model();
        cfg.formula_features = FormulaFeatures::Evidence;
        cfg.formula_table.rows = dtable.rows as u32;
        cfg.formula_table.sha256 = dtable.sha256.clone();
        let mut rng = Rng::seeded(5);
        let mut model = Ms2Model::<R, EB>::init(&cfg, &device, &mut rng)?;
        model.upload_enum_artifacts(&domain, &bounds, &device)?;
        let batch = evidence_fixture_batch(&comps, 64);
        let gcfg = tiny_gen_evidence(32, 4096);
        Ok((dtable, model, batch, gcfg))
    })();
    let (dtable, mut model, batch, gcfg) = match bf16_setup {
        Err(e) => {
            assert!(
                !e.to_string().is_empty(),
                "a bf16 capability refusal must still report"
            );
            println!("BF16 unsupported at setup (capability refusal): {e}");
            return;
        }
        Ok(v) => v,
    };
    // Uncached bf16 reference first: if the runtime cannot run it, that is
    // the capability refusal (not a cache defect).
    let constants = Ms2Constants::new(&device);
    let mut ws_plain = GenerationWorkspace::new();
    let plain = match model.generate(&batch, &dtable, &gcfg, &mut ws_plain, &constants) {
        Err(e) => {
            assert!(!e.to_string().is_empty(), "a bf16 capability refusal must still report");
            println!("BF16 generate unsupported (capability refusal): {e}");
            return;
        }
        Ok(v) => v,
    };
    // The f32 model builds enum + evidence over the same batch; attaching it
    // to the bf16 model reuses enumeration, never evidence.
    let (_table, _dtable_f32, f32_model) =
        setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let f32_header = f32_model.enum_cache_header(&gcfg).unwrap();
    assert_eq!(f32_header.evidence_dtype, "f32");
    let mut cache = EnumCache::new(f32_header);
    f32_model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    assert!(cache.evidence_len() > 0, "the f32 build must store evidence");
    model.set_enum_cache(Some(Arc::new(cache))).unwrap();
    let mut ws_hit = GenerationWorkspace::new();
    let hit = model
        .generate(&batch, &dtable, &gcfg, &mut ws_hit, &constants)
        .unwrap();
    // Cached bf16 equals uncached bf16 (bit for bit on CPU); enumeration hit,
    // evidence missed (dtype policy), so the device evidence stage ran.
    if is_cpu() {
        assert_eq!(hit, plain, "cached bf16 evidence step must equal uncached bit for bit");
    } else {
        assert_eq!(hit.batch, plain.batch, "cached bf16 must match uncached structurally");
    }
    let (lookups, hits, ev_lookups, ev_hits) = model.enum_cache_stats();
    assert_eq!((lookups, hits), (1, 1), "integer enumeration must hit across dtypes");
    assert!(ev_lookups > 0 && ev_hits == 0, "evidence must miss under bf16 ({ev_lookups}/{ev_hits})");
    println!("BF16 cached evidence step: enum {lookups}/{hits} ev {ev_lookups}/{ev_hits}");
}

#[test]
fn f10_tombstone_sequence_at_budget_boundary() {
    // Task F10 item B1, the reviewer's sequence: fill the evidence table
    // with 28 keys (32 physical buckets), remove an interior key through an
    // enumeration replace (evicting its evidence — a DELETED tombstone in
    // the evidence table), then insert a key that takes an EMPTY slot at
    // the exact budget boundary. The insert must be refused or stay within
    // budget — never over — and the tracked buckets must cover the implied
    // physical buckets after every operation. Bucket counts are a pure
    // function of the length history (growth passes only through the
    // cache's own budgeted reserves), so the 28-fill pins exactly 32
    // buckets per table deterministically.
    let _serial = serial();
    let header = || {
        EnumCacheHeader::new(
            "domain".to_string(),
            "bounds".to_string(),
            4,
            32,
            4096,
            65_536,
            16,
            "f32",
        )
    };
    let comps = fixture_comps();
    let ev_batch = evidence_batch(
        &[precursor_of(&comps[4])],
        &[50],
        &[1],
        &[0],
        &[0],
        &[50],
        &[8],
        &[vec![60_000_000; 64]],
        &[vec![1.0; 64]],
        64,
    );
    let ev_inputs = evidence_inputs_for_batch(&ev_batch, 0, 8, 2048, 1023);
    let ev_canon = evidence_canonical_bytes(&ev_inputs);
    let new_evidence = |meta: [u32; 8], h: u64| {
        (
            EvidenceKey { meta, h0: h, h1: !h },
            ev_inputs.peak_count,
            ev_inputs.mz_row.iter().map(|&m| u64::from(m)).sum::<u64>(),
        )
    };
    let check = |cache: &EnumCache, budget: u64, what: &str| {
        assert!(
            (cache.resident_bytes() as u64) <= budget,
            "{what}: resident {} over budget {budget}",
            cache.resident_bytes()
        );
        assert!(
            cache.bucket_invariant_holds(),
            "{what}: bucket invariant broken (tracked {:?}, live {:?})",
            cache.table_buckets(),
            cache.table_capacities()
        );
    };
    // One post-evict state, then CLONED twins: a clone preserves the
    // tables, the tombstones and the hasher, so the probe measures exactly
    // what the pinned caches will do (two independently built caches could
    // evict to EMPTY vs DELETED and diverge — the budget invariant holds
    // for both, but the exact boundary needs identical layout).
    let metas: Vec<[u32; 8]> = (0..28)
        .map(|i| {
            let mut m = [0u32; 8];
            m[0] = i as u32;
            m[7] = 0xC0FFEE;
            m
        })
        .collect();
    let mut base = EnumCache::with_max_resident_bytes(header(), u64::MAX);
    for meta in &metas {
        assert!(base.insert(*meta, [7, 5, 0, 0, 1], &[], 32).unwrap());
        check(&base, u64::MAX, "enum fill");
    }
    assert_eq!(base.table_buckets().0, 32, "28 enum entries pin 32 buckets");
    for (i, meta) in metas.iter().enumerate() {
        let (key, pc, mz) = new_evidence(*meta, i as u64);
        assert!(base
            .insert_evidence(key, pc, mz, 3, vec![1u8; 2], vec![1.0f32.to_bits(); 2], vec![1u8; 2], 32, ev_canon.clone())
            .unwrap());
        check(&base, u64::MAX, "evidence fill");
    }
    assert_eq!(base.table_buckets().1, 32, "28 evidence entries pin 32 buckets");
    assert_eq!(base.evidence_len(), 28);
    // Remove an interior key through an enumeration replace: its evidence
    // is evicted (a tombstone, or an EMPTY restoration — either is legal),
    // the tables keep their buckets.
    assert!(base.insert(metas[7], [7, 5, 0, 0, 1], &[], 32).unwrap());
    assert_eq!(base.evidence_len(), 27, "the replace must evict the victim evidence");
    assert_eq!(base.table_buckets(), (32, 32), "removal never lowers the tracked buckets");
    check(&base, u64::MAX, "evicting replace");
    let mut probe = base.clone();
    let mut cache = base.clone();
    let mut poor = base.clone();
    // Measure the exact cost of the EMPTY-slot insert on the probe, then
    // pin the real cache's budget to it: admitted and exactly within
    // budget, never over. A full table resizes for `usable + 1` even when
    // the new key takes an EMPTY slot (or stays put when the eviction left
    // an EMPTY restoration) — either way the budget check prices exactly
    // the outcome, and the invariant holds throughout. (The old code
    // predicted 55 usable slots against 56 actual here.)
    let (fresh_key, fresh_pc, fresh_mz) = new_evidence(metas[0], 1_000_000);
    let before = probe.resident_bytes();
    let buckets_before = probe.table_buckets();
    assert!(probe
        .insert_evidence(fresh_key, fresh_pc, fresh_mz, 3, vec![1u8; 2], vec![1.0f32.to_bits(); 2], vec![1u8; 2], 32, ev_canon.clone())
        .unwrap());
    let cost = (probe.resident_bytes() as u64) - (before as u64);
    let buckets_after = probe.table_buckets();
    assert!(
        buckets_after.1 == buckets_before.1 || buckets_after.1 == buckets_before.1 * 2,
        "the EMPTY-slot insert either reuses the table or grows exactly one tier, got {buckets_before:?} -> {buckets_after:?}"
    );
    check(&probe, u64::MAX, "probe insert");
    cache.set_max_resident_bytes((before as u64) + cost);
    assert!(cache
        .insert_evidence(fresh_key, fresh_pc, fresh_mz, 3, vec![1u8; 2], vec![1.0f32.to_bits(); 2], vec![1u8; 2], 32, ev_canon.clone())
        .unwrap());
    assert_eq!(
        cache.resident_bytes() as u64,
        (before as u64) + cost,
        "the boundary insert lands exactly on budget"
    );
    assert_eq!(cache.table_buckets(), buckets_after, "twin layouts agree");
    check(&cache, (before as u64) + cost, "boundary insert");
    // One byte less refuses, counting the refusal, changing nothing.
    poor.set_max_resident_bytes((before as u64) + cost - 1);
    let refusals = poor.budget_refusals();
    assert!(
        !poor
            .insert_evidence(fresh_key, fresh_pc, fresh_mz, 3, vec![1u8; 2], vec![1.0f32.to_bits(); 2], vec![1u8; 2], 32, ev_canon.clone())
            .unwrap(),
        "one byte under the boundary must refuse"
    );
    assert_eq!(poor.budget_refusals(), refusals + 1, "the refusal is counted");
    assert_eq!(poor.resident_bytes(), before, "a refused insert changes nothing");
    check(&poor, (before as u64) + cost - 1, "refused insert");
    // Past the usable count the next insert needs one tier more: fill to
    // `len == capacity` (whatever tier the eviction outcome produced),
    // then the next key refuses at the pinned boundary instead of growing
    // past the budget — and admits once unpinned, on exactly one tier up.
    let mut h = 3_000_000u64;
    cache.set_max_resident_bytes(u64::MAX);
    while cache.evidence_len() < cache.table_capacities().1 {
        let (key, pc, mz) = new_evidence(metas[(h as usize) % metas.len()], h);
        h += 1;
        assert!(cache
            .insert_evidence(key, pc, mz, 3, vec![1u8; 2], vec![1.0f32.to_bits(); 2], vec![1u8; 2], 32, ev_canon.clone())
            .unwrap());
        check(&cache, u64::MAX, "tier fill");
    }
    let full_buckets = cache.table_buckets().1;
    let (k_next, pc_next, mz_next) = new_evidence(metas[2], 9_000_000);
    let pinned = cache.resident_bytes() as u64;
    cache.set_max_resident_bytes(pinned);
    let refusals_next = cache.budget_refusals();
    assert!(
        !cache
            .insert_evidence(k_next, pc_next, mz_next, 3, vec![1u8; 2], vec![1.0f32.to_bits(); 2], vec![1u8; 2], 32, ev_canon.clone())
            .unwrap(),
        "the next-tier key must refuse at the pinned boundary"
    );
    assert_eq!(cache.budget_refusals(), refusals_next + 1, "the refusal is counted");
    check(&cache, pinned, "growth refusal");
    cache.set_max_resident_bytes(u64::MAX);
    assert!(cache
        .insert_evidence(k_next, pc_next, mz_next, 3, vec![1u8; 2], vec![1.0f32.to_bits(); 2], vec![1u8; 2], 32, ev_canon.clone())
        .unwrap());
    assert_eq!(
        cache.table_buckets().1,
        full_buckets * 2,
        "the next-tier key grows exactly one tier"
    );
    check(&cache, u64::MAX, "growth admit");
    println!("TOMBSTONE boundary: cost={cost} buckets={:?}", cache.table_buckets());
}

#[test]
fn f10_load_read_bounded_by_declared_size() {
    // Task F10 item B2: the bounded read. A valid cache file plus trailing
    // junk, read through the handle-based loader with the declared size,
    // loads fine while the reader observes at most the declared bytes —
    // a file replaced or grown after the size check cannot be read past
    // the limit. A shorter declaration fails closed (checksum) instead of
    // loading partially.
    let _serial = serial();
    let header = EnumCacheHeader::new(
        "domain".to_string(),
        "bounds".to_string(),
        4,
        32,
        4096,
        65_536,
        16,
        "f32",
    );
    let mut cache = EnumCache::new(header.clone());
    assert!(cache.insert([3u32; 8], [0, 0, 0, 0, 1], &[], 32).unwrap());
    let dir = std::env::temp_dir().join("ms2_f10_bounded");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cache.bin");
    cache.save(&path).unwrap();
    let file_bytes = std::fs::read(&path).unwrap();
    let declared = file_bytes.len() as u64;
    // A reader longer than its declared size, counting every byte taken.
    struct Counting {
        data: Vec<u8>,
        pos: usize,
        read_bytes: usize,
    }
    impl std::io::Read for Counting {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let n = buf.len().min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            self.read_bytes += n;
            Ok(n)
        }
    }
    let mut junk = file_bytes.clone();
    junk.extend_from_slice(&[0xA5u8; 1 << 20]);
    let mut reader = Counting { data: junk, pos: 0, read_bytes: 0 };
    let loaded =
        EnumCache::load_from_reader(&mut reader, declared, &path, &header, u64::MAX).unwrap();
    assert_eq!(loaded.len(), 1, "the declared prefix loads");
    assert_eq!(
        reader.read_bytes as u64, declared,
        "the read stays bounded by the declared size (file is 1 MiB longer)"
    );
    assert!(
        (loaded.resident_bytes() as u64) <= u64::MAX,
        "loaded cache within budget"
    );
    // A shorter declaration truncates the body: checksum failure, never a
    // partial cache.
    let mut short = Counting {
        data: file_bytes.clone(),
        pos: 0,
        read_bytes: 0,
    };
    let err = EnumCache::load_from_reader(&mut short, declared - 1, &path, &header, u64::MAX)
        .unwrap_err();
    assert!(
        err.to_string().contains("truncated")
            || err.to_string().contains("checksum")
            || err.to_string().contains("corrupted"),
        "a short declaration must fail closed, got: {err}"
    );
    let _ = std::fs::remove_file(&path);
    println!("BOUNDED read: declared={declared} observed={}", reader.read_bytes);
}

#[test]
fn f10_evidence_build_survives_enum_refusal_generation() {
    // Task F10 item B4: the GENERATION branch of the refused-insert fix
    // (the existing tests guard the training branch only). With
    // `FormulaFeatures::Evidence` and a ZERO budget,
    // `Ms2Model::build_enum_cache` SUCCEEDS — rows stay uncached, refusals
    // are counted, and the evidence precompute skips batches whose
    // enumeration was refused — instead of aborting. Attaching the refused
    // cache, `generate` equals the uncached generate bit for bit on CPU.
    let _serial = serial();
    let device = dev();
    let comps = fixture_comps();
    let (table, dtable, _) = setup_model_with_layout(&device, &comps, 5, FormulaFeatures::Evidence);
    let batch = evidence_fixture_batch(&comps, 64);
    let gcfg = tiny_gen_evidence(32, 4096);
    let constants = Ms2Constants::new(&device);
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut cfg_ev = tiny_model();
    cfg_ev.formula_features = FormulaFeatures::Evidence;
    cfg_ev.formula_table.rows = dtable.rows as u32;
    cfg_ev.formula_table.sha256 = dtable.sha256.clone();
    let mut rng_a = Rng::seeded(5);
    let mut rng_b = Rng::seeded(5);
    let mut model_plain = Ms2Model::<R, E>::init(&cfg_ev, &device, &mut rng_a).unwrap();
    let mut model_cached = Ms2Model::<R, E>::init(&cfg_ev, &device, &mut rng_b).unwrap();
    for m in [&mut model_plain, &mut model_cached] {
        m.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    }
    // Zero budget through the GENERATION build path (this is the branch
    // under test — reverting its refused-insert skip fails below).
    let header = model_cached.enum_cache_header(&gcfg).unwrap();
    let mut poor = EnumCache::with_max_resident_bytes(header, 0);
    model_cached
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut poor)
        .unwrap();
    assert_eq!(poor.len(), 0, "zero budget stores no enumeration");
    assert_eq!(poor.evidence_len(), 0, "zero budget stores no evidence");
    assert!(
        poor.budget_refusals() > 0,
        "zero budget must count its refusals"
    );
    model_cached.set_enum_cache(Some(Arc::new(poor))).unwrap();
    let mut ws_plain = GenerationWorkspace::new();
    let mut ws_cached = GenerationWorkspace::new();
    for _ in 0..2 {
        model_plain.generate(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
        model_cached.generate(&batch, &dtable, &gcfg, &mut ws_cached, &constants).unwrap();
    }
    let out_plain =
        model_plain.generate(&batch, &dtable, &gcfg, &mut ws_plain, &constants).unwrap();
    let out_cached =
        model_cached.generate(&batch, &dtable, &gcfg, &mut ws_cached, &constants).unwrap();
    assert_candidate_batch_close(&out_plain, &out_cached);
    if is_cpu() {
        assert_eq!(
            out_plain.actions, out_cached.actions,
            "refused-cache generate actions must equal uncached bit for bit"
        );
    }
    let (lookups, hits, ev_lookups, ev_hits) = model_cached.enum_cache_stats();
    assert_eq!((hits, ev_hits), (0, 0), "nothing can hit under zero budget");
    assert!(lookups > 0 && ev_lookups > 0, "the step must still look up");
    println!("B4 generation refusal: refusals counted, generate bit-identical");
}
