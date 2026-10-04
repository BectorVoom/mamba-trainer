//! Launches per enumerate call constant (own binary).

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    Control, FormulaSource, GenerationConfig, GenerationMode, ModelConfig, SCHEMA_VERSION,
    SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{
    ENUM_LANES_MAX_DEFAULT, EnumDomain, RatioBounds, enum_lanes_per_dispatch,
};
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::ms2_enum::{EnumLaunch, enum_offsets};
use mamba3::tensor::ops::random::Rng;

/// File-level serialisation: the process-wide launch counter is perturbed
/// by any test running beside these, so every test holds this mutex.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

type R = Auto;
type E = f32;

#[test]
fn enumerate_launches_constant_per_call() {
    let _serial = serial();
    let device = Device::<R>::default();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0], [3, 7, 1, 2, 0, 0, 0, 0, 0, 0]];
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
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: 64,
        spectrum_id: vec![1, 2],
        raw_peak_count: vec![16, 16],
        peak_count: vec![16, 16],
        peak_id: (0..128).map(|i| (i % 64) as u32).collect(),
        mz_udalton: vec![60_000_000; 128],
        intensity: vec![1.0; 128],
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
    };
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    }
    device.synchronize();
    reset_launch_count();
    model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    let first = launch_count();
    reset_launch_count();
    model.generate(&batch, &dtable, &gcfg, &mut ws, &constants).unwrap();
    device.synchronize();
    let second = launch_count();
    println!("enumerate search launches per call: {first} then {second}");
    assert!(first > 0);
    assert_eq!(first, second, "launches per call constant");
}

#[test]
fn enum_count_and_fill_launch_once_per_dispatch_chunk() {
    // Finding R1-B2: count and fill each launch
    // `ceil(B*P / lanes_per_launch)` times, with `lanes_per_launch =
    // max(1, dispatch_visits_max / lane_visits_max)`. B = 2 spectra over a
    // two-lane domain (4 lanes), chunked 1 / 2 / 3 lanes per dispatch.
    let _serial = serial();
    let device = Device::<R>::default();
    let launch = EnumLaunch::from_chemistry();
    // Two rare lanes: meta rows join at least one lane each (the exact joins
    // do not matter here, only the launch counts).
    let meta_rows: [[u32; 8]; 2] = [[0, 0, 0, 1, 0, 4096, 8, 0], [0, 0, 0, 1, 0, 4096, 8, 0]];
    let rare_rows: [[u32; 8]; 2] = [[0, 0, 0, 0, 0, 0, 0, 0]; 2];
    let packed = vec![0u32; 64];
    let meta_flat: Vec<u32> = meta_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let rare_flat: Vec<u32> = rare_rows.iter().flat_map(|r| r.iter().copied()).collect();
    for (dispatch, lane_visits, per) in
        [(1u32, 4096u32, 1usize), (8192u32, 4096u32, 2usize), (12288u32, 4096u32, 3usize)]
    {
        assert_eq!(enum_lanes_per_dispatch(dispatch, lane_visits), per);
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
                ENUM_LANES_MAX_DEFAULT,
                dispatch,
                lane_visits,
            )
            .unwrap();
        device.synchronize();
        assert_eq!(
            launch_count(),
            want,
            "count launches ceil(4 / {per}) times (dispatch {dispatch})"
        );
        let offsets_t = IdTensor::from_slice(&vec![0u32; 4], vec![4], &device).unwrap();
        let counters_t = IdTensor::from_slice(&vec![0u32; 2 * 5], vec![2, 5], &device).unwrap();
        enum_offsets(&stats_t, &meta_t, &offsets_t, &counters_t, 8, 8, ENUM_LANES_MAX_DEFAULT)
            .unwrap();
        device.synchronize();
        let cand_t =
            IdTensor::from_slice(&vec![0u32; 2 * 8 * 13], vec![2, 8, 13], &device).unwrap();
        reset_launch_count();
        launch
            .fill(
                &meta_t,
                &rare_t,
                &bounds_t,
                &offsets_t,
                &cand_t,
                8,
                ENUM_LANES_MAX_DEFAULT,
                dispatch,
                lane_visits,
            )
            .unwrap();
        device.synchronize();
        assert_eq!(
            launch_count(),
            want,
            "fill launches ceil(4 / {per}) times (dispatch {dispatch})"
        );
        check_launches(&device).unwrap();
    }
}
