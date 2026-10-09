//! Allocation probe for the enumeration/evidence cache hit path (task F8
//! item 5).
//!
//! A counting [`#[global_allocator]`](std::alloc::GlobalAlloc) in THIS test
//! binary only measures host allocations around (a) the per-use cache header
//! check and (b) one cached `generate` hit. The header check must perform
//! zero allocations; the hit-path total is printed for the before/after
//! comparison in the F8 summary.

#![cfg(feature = "backend")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bytes handed out by the global allocator since the last reset.
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
/// `alloc` calls since the last reset.
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
/// Whether counting is active (warm-up and device threads must not pollute
/// the window).
static COUNTING: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset_counters() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn start() {
    reset_counters();
    COUNTING.store(1, Ordering::Relaxed);
}

fn stop() -> (usize, usize) {
    COUNTING.store(0, Ordering::Relaxed);
    (
        ALLOC_CALLS.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    AllocationMode, Control, FormulaSource, GenerationConfig, GenerationMode, IdentityMode,
    ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::enum_cache::EnumCache;
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
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

fn precursor_of(c: &Composition) -> u32 {
    composition_mass(c).unwrap() + 1_007_825 - 549
}

fn spectrum_batch(
    precursors: &[u32],
    uncs: &[u32],
    adducts: &[u16],
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
        precursor_tolerance_ppm_tenths: vec![0; b],
        instrument_class: vec![0; b],
    }
}

/// One test only (no intra-binary parallelism noise): measure the header
/// check and one cached `generate` hit.
#[test]
fn alloc_probe_cached_hit() {
    let device = dev();
    let comps = fixture_comps();
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let mut cfg = tiny_model();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng).unwrap();
    model
        .upload_enum_artifacts(&domain, &bounds, &device)
        .unwrap();
    let batch = spectrum_batch(
        &[precursor_of(&comps[4]), precursor_of(&comps[6])],
        &[50, 50],
        &[1, 1],
        64,
    );
    let gcfg = tiny_gen(32, 4096);
    let header = model.enum_cache_header(&gcfg).unwrap();
    let mut cache = EnumCache::new(header.clone());
    model
        .build_enum_cache([&batch].into_iter(), &gcfg, &mut cache)
        .unwrap();
    let stored_header = cache.header().clone();
    model
        .set_enum_cache(Some(std::sync::Arc::new(cache)))
        .unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    // Warm-up (lazy device/client allocation stays outside the window).
    model
        .generate(&batch, &dtable, &gcfg, &mut ws, &constants)
        .unwrap();
    model
        .generate(&batch, &dtable, &gcfg, &mut ws, &constants)
        .unwrap();
    device.synchronize();

    // (a) The per-use header check in isolation: the borrowed expectation
    // plus the field-by-field comparison, exactly as the cached search path
    // does on every use. Must perform zero allocations (task F8 item 5).
    use mamba3::models::ms2::enum_cache::expected_header;
    let artifacts = model.enum_artifacts.as_ref().unwrap();
    let p = u32::try_from(artifacts.p).unwrap();
    start();
    let exp = expected_header(
        artifacts.domain_sha256.as_str(),
        artifacts.bounds_sha256.as_str(),
        p,
        gcfg.formula_window,
        gcfg.formula_rows_scored_max,
        gcfg.enum_lane_visits_max,
        16,
    );
    stored_header.check_expected(&exp).unwrap();
    let (header_calls, header_bytes) = stop();
    println!("ALLOC borrowed header check: {header_calls} allocs, {header_bytes} bytes");
    assert_eq!(
        (header_calls, header_bytes),
        (0, 0),
        "the borrowed per-use header check must not allocate"
    );

    // (b) One cached `generate` hit (minimum of 5 rounds).
    let mut best = (usize::MAX, usize::MAX);
    for _ in 0..5 {
        start();
        model
            .generate(&batch, &dtable, &gcfg, &mut ws, &constants)
            .unwrap();
        device.synchronize();
        let got = stop();
        best = (best.0.min(got.0), best.1.min(got.1));
    }
    println!(
        "ALLOC cached generate hit: {} allocs, {} bytes (min of 5)",
        best.0, best.1
    );
    let (lookups, hits, _, _) = model.enum_cache_stats();
    // Two warm-up hits plus five measured hits.
    assert_eq!(
        (lookups, hits),
        (7, 7),
        "every generate must be a cache hit"
    );

    // (c) Bounded loading (task F8 item 7): a tiny file advertising a
    // million entries must reject with a small allocation — the reservation
    // itself is bounded. Reverting the bound reserves a million map slots
    // (~tens of MB) and fails this assertion.
    fn fnv1a64(bytes: &[u8], mut h: u64) -> u64 {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        h
    }
    let hj = serde_json::to_string(&stored_header).unwrap();
    let mut body = Vec::new();
    body.extend_from_slice(b"MS2ENUMC");
    body.extend_from_slice(
        &mamba3::models::ms2::enum_cache::ENUM_CACHE_FORMAT_VERSION.to_le_bytes(),
    );
    body.extend_from_slice(&(hj.len() as u32).to_le_bytes());
    body.extend_from_slice(hj.as_bytes());
    body.extend_from_slice(&1_000_000u64.to_le_bytes());
    let h0 = fnv1a64(&body, 0xcbf2_9ce4_8422_2325);
    let h1 = fnv1a64(&body, 0x3d0d_613b_7bde_ddda);
    body.extend_from_slice(&h0.to_le_bytes());
    body.extend_from_slice(&h1.to_le_bytes());
    start();
    let err = mamba3::models::ms2::enum_cache::EnumCache::parse_with_stats(
        &body,
        std::path::Path::new("stub.bin"),
        Some(&stored_header),
        mamba3::models::ms2::enum_cache::DEFAULT_ENUM_CACHE_MAX_RESIDENT_BYTES,
    )
    .expect_err("million-entry stub must fail");
    let (parse_calls, parse_bytes) = stop();
    println!("ALLOC bounded parse: {parse_calls} allocs, {parse_bytes} bytes ({err})");
    assert!(
        err.to_string().contains("unexpected end"),
        "the bounded parse (not checksum/version) must reject it, got {err}"
    );
    assert!(
        parse_bytes < 1_000_000,
        "bounded parse must not reserve a million slots ({parse_bytes} bytes)"
    );
}
