//! Refusal before any allocation/launch with `Enumerate` (own binary).

#![cfg(feature = "backend")]

use mamba3::backend::{Device, allocation_calls, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
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

#[test]
fn enumerate_refuses_lanes_before_allocation_or_launch() {
    let device = Device::<R>::default();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]];
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let table = FormulaTable::from_compositions(comps.into_iter()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let mut cfg = ModelConfig::v0();
    cfg.formula_table.rows = dtable.rows as u32;
    cfg.formula_table.sha256 = dtable.sha256.clone();
    let mut rng = Rng::seeded(5);
    let mut model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng).unwrap();
    model.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    let p = model.enum_artifacts.as_ref().map(|a| a.p).unwrap_or(0);
    assert!(p > 0, "fixture domain has lanes");
    let constants = Ms2Constants::new(&device);
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: 64,
        spectrum_id: vec![1, 2],
        raw_peak_count: vec![10, 10],
        peak_count: vec![10, 10],
        peak_id: (0..128).map(|i| if i % 64 < 10 { (i % 64) as u32 } else { u32::MAX }).collect(),
        mz_udalton: vec![60_000_000; 128],
        intensity: vec![1.0; 128],
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50, 50],
        precursor_mz_udalton: vec![200_000_000, 200_000_000],
        precursor_uncertainty_udalton: vec![50, 50],
        adduct: vec![1, 1],
        polarity: vec![1, 1],
        collision_energy_ev: vec![30.0, 30.0],
        collision_energy_known: vec![1, 1],
        energy_count: vec![1, 1],
        fragment_tolerance_ppm_tenths: vec![0, 0],
        precursor_tolerance_ppm_tenths: vec![0, 0],
        instrument_class: vec![0, 0],
    };
    // B = 2, P = 1, so B * P = 2 > 1 refuses (lanes_max must be
    // non-zero, so 1 is the smallest refusing limit here).
    let lanes_max = 1u32;
    assert!((2 * p as u64) > lanes_max as u64, "test refuses: B*P={} > {lanes_max}", 2 * p);
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
        enum_lanes_max: lanes_max,
        enum_lane_visits_max: 65_536,
        enum_dispatch_visits_max: 4_000_000,
        allocation: mamba3::models::ms2::contract::AllocationMode::RoundRobin,
        identity: mamba3::models::ms2::contract::IdentityMode::TraceOnly,
        identity_work_max: 4096,
        returned: 0,
        evidence: false,
        ion_request_work_max: 268435456,
    };
    let mut ws = GenerationWorkspace::new();
    // Warm once with a fitting limit so the bucket exists; then refuse.
    let mut ok_cfg = gcfg.clone();
    ok_cfg.enum_lanes_max = 262_144;
    model.generate(&batch, &dtable, &ok_cfg, &mut ws, &constants).unwrap();
    device.synchronize();
    let launches_before = launch_count();
    let allocs_before = allocation_calls();
    reset_launch_count();
    let err = model
        .generate(&batch, &dtable, &gcfg, &mut ws, &constants)
        .expect_err("B*P over the limit refuses");
    device.synchronize();
    assert!(format!("{err}").contains("enum_lanes_max"));
    assert_eq!(launch_count(), 0, "no launch on refusal");
    assert_eq!(
        allocation_calls(),
        allocs_before,
        "no allocation on refusal"
    );
    let _ = launches_before;
}
