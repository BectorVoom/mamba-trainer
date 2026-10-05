//! One read per warmed `generate` with `Enumerate` (own binary: reads the
//! process-global read counter, so no other counter test shares this
//! process).

#![cfg(feature = "backend")]

use mamba3::backend::{Device, read_count, reset_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    Control, FormulaSource, GenerationConfig, GenerationMode, ModelConfig, SCHEMA_VERSION,
    SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn comp(c: u16, h: u16, n: u16, o: u16) -> Composition {
    [c, h, n, o, 0, 0, 0, 0, 0, 0]
}

#[test]
fn warmed_enumerate_generate_reads_once() {
    let device = Device::<R>::default();
    let comps = vec![comp(6, 6, 0, 0), comp(3, 7, 1, 2)];
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let mut cfg = ModelConfig::v0();
    cfg.d_model = 16;
    cfg.n_peaks = 16;
    cfg.encoder_blocks = 1;
    cfg.decoder_blocks = 1;
    cfg.attention_heads = 2;
    cfg.encoder.d_model = 16;
    cfg.encoder.n_heads = 2;
    cfg.encoder.head_dim = 8;
    cfg.encoder.d_state = 8;
    cfg.encoder.n_groups = 2;
    cfg.decoder.d_model = 16;
    cfg.decoder.n_heads = 2;
    cfg.decoder.head_dim = 8;
    cfg.decoder.d_state = 8;
    cfg.decoder.n_groups = 2;
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng).unwrap();
    model.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    let constants = Ms2Constants::new(&device);
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let b = comps.len();
    let n_raw = 64;
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut rng = Rng::seeded(21);
    for bi in 0..b {
        for i in 0..32 {
            peak_id[bi * n_raw + i] = i as u32;
            mz[bi * n_raw + i] = (60_000_000 + bi as u32 * 1_000_000 + i as u32 * 1000).max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + rng.uniform_vec(1, 0.0, 1.0)[0];
        }
    }
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![1, 2],
        raw_peak_count: vec![32, 32],
        peak_count: vec![32, 32],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; 2],
        precursor_mz_udalton: precursors,
        precursor_uncertainty_udalton: vec![50; 2],
        adduct: vec![1; 2],
        polarity: vec![1; 2],
        collision_energy_ev: vec![30.0; 2],
        collision_energy_known: vec![1; 2],
        energy_count: vec![1; 2],
        fragment_tolerance_ppm_tenths: vec![0; 2],
        precursor_tolerance_ppm_tenths: vec![0; 2],
        instrument_class: vec![0; 2],
    };
    let gcfg = GenerationConfig {
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
        formula_window: 32,
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
    };
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    }
    device.synchronize();
    reset_read_count();
    model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    assert_eq!(read_count(), 1, "warmed enumerate generate reads once");
}
