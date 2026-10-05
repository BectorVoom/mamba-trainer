//! Reserved bytes flat over repeated enumerate calls (own binary).

#![cfg(feature = "backend")]

use mamba3::backend::{Device, reserved_bytes};
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

#[test]
fn enumerate_reserved_flat_over_repeated_calls() {
    let device = Device::<R>::default();
    let Some(baseline) = reserved_bytes(&device) else {
        println!("unavailable");
        return;
    };
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]];
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let dtable = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let mut cfg = ModelConfig::v0();
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
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: 64,
        spectrum_id: vec![1],
        raw_peak_count: vec![16],
        peak_count: vec![16],
        peak_id: (0..64).map(|i| if i < 16 { i } else { u32::MAX }).collect(),
        mz_udalton: vec![60_000_000; 64],
        intensity: vec![1.0; 64],
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50],
        precursor_mz_udalton: precursors,
        precursor_uncertainty_udalton: vec![50],
        adduct: vec![1],
        polarity: vec![1],
        collision_energy_ev: vec![30.0],
        collision_energy_known: vec![1],
        energy_count: vec![1],
        fragment_tolerance_ppm_tenths: vec![0],
        precursor_tolerance_ppm_tenths: vec![0],
        instrument_class: vec![0],
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
    let Some(mid) = reserved_bytes(&device) else {
        println!("unavailable");
        return;
    };
    for _ in 0..3 {
        model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    }
    let Some(end) = reserved_bytes(&device) else {
        println!("unavailable");
        return;
    };
    println!("reserved baseline {baseline} mid {mid} end {end}");
    assert_eq!(mid, end, "reserved flat over repeated enumerate calls");
}
