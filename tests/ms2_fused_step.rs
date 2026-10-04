//! The fused sampler step (P8 / O4) against the composed reference step.
//!
//! `Ms2Decoder::step_packed` replaces the composed tensor ops of
//! `Ms2Decoder::step_logits` plus the pack copies with one kernel per stage.
//! Both run the same arithmetic up to the summation order inside a dot
//! product, so a generation call driven by either must emit the same
//! trajectories and carry the same recurrent state within float tolerance.
//! These tests run the two forms of the same call side by side
//! (`GenerationWorkspace::composed_step`).

#![cfg(feature = "backend")]

use mamba3::backend::{check_launches, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::contract::{
    Control, FormulaSource, GenerationConfig, GenerationMode, ModelConfig, SCHEMA_VERSION,
    SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model, StepCarries};
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn tiny_config() -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = 16;
    m.n_peaks = 16;
    m.encoder_blocks = 1;
    m.decoder_blocks = 2;
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

/// The V1 capacities `(A, R_max) = (32, 8)` on the tiny widths.
fn v1_tiny_config() -> ModelConfig {
    let mut m = tiny_config();
    m.max_atoms = 32;
    m.max_ring_closures = 8;
    m.decoder_blocks = 3;
    m.decoder.n_heads = 4;
    m.decoder.head_dim = 8;
    m
}

fn generation(max_steps: u32, trajectories: u32, seed: u64) -> GenerationConfig {
    GenerationConfig {
        schema_version: SCHEMA_VERSION,
        trajectories,
        formulas: 2,
        seed,
        temperature: 1.0,
        max_steps,
        max_device_bytes: 2 * 1024 * 1024 * 1024,
        formula_rows_visited_max: u32::MAX,
        formula_rows_scored_max: 4096,
        mode: GenerationMode::Sampling,
        oracle_formula: false,
        control: Control::None,
        formula_source: FormulaSource::Table,
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
    }
}

fn make_spectra(
    spectrum_ids: &[u64],
    precursors: &[u32],
    n_raw: usize,
    peak_counts: &[u32],
    seed: u64,
) -> SpectrumBatch {
    let b = spectrum_ids.len();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for (bi, &count) in peak_counts.iter().enumerate() {
        let count = count as usize;
        peak_count[bi] = count as u32;
        raw_peak_count[bi] = count as u32;
        let precursor = precursors[bi];
        for i in 0..count {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursor - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            let u = rng.uniform_vec(1, 0.0, 1.0)[0];
            intensity[bi * n_raw + i] = 0.5 + 2.0 * u;
        }
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: spectrum_ids.to_vec(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursors.to_vec(),
        precursor_uncertainty_udalton: vec![50; b],
        adduct: vec![1; b],
        polarity: vec![1; b],
        collision_energy_ev: vec![30.0; b],
        collision_energy_known: vec![1; b],
        energy_count: vec![1; b],
        fragment_tolerance_ppm_tenths: vec![0; b],
        precursor_tolerance_ppm_tenths: vec![0; b],
        instrument_class: vec![0; b],
    }
}

/// Counter-reading tests in this file run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
            "{what}[{i}]: fused {x} against composed {y}"
        );
    }
}

fn carries_close(fused: &[StepCarries], composed: &[StepCarries], tol: f32) {
    assert_eq!(fused.len(), composed.len(), "one carry snapshot per step");
    for (f, c) in fused.iter().zip(composed.iter()) {
        assert_eq!(f.step, c.step);
        assert_eq!(f.layers.len(), c.layers.len());
        for (l, (fl, cl)) in f.layers.iter().zip(c.layers.iter()).enumerate() {
            let at = format!("step {} layer {l}", f.step);
            close(&fl.h, &cl.h, tol, &format!("{at} h"));
            close(&fl.last_u, &cl.last_u, tol, &format!("{at} last_u"));
            match (&fl.angle, &cl.angle) {
                (Some(a), Some(b)) => close(a, b, tol, &format!("{at} angle")),
                (None, None) => {}
                _ => panic!("{at}: angle present in one form only"),
            }
            match (&fl.conv, &cl.conv) {
                (Some(a), Some(b)) => close(a, b, tol, &format!("{at} conv")),
                (None, None) => {}
                _ => panic!("{at}: conv present in one form only"),
            }
        }
    }
}

/// One generation call in both forms over the same model, spectra and seed.
fn run_both(
    config: ModelConfig,
    max_steps: u32,
    trajectories: u32,
    seeds: &[u64],
) -> (usize, usize) {
    let device = mamba3::backend::Device::<R>::default();
    let mut cfg = config;
    let comps: Vec<Composition> = vec![
        [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
    ];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = make_spectra(&[601, 602, 603], &precursors, 64, &[10, 12, 9], 33);
    let mut launches = (0usize, 0usize);
    let mut stopped_rows = 0usize;
    let mut live_rows = 0usize;
    for &seed in seeds {
        let gcfg = generation(max_steps, trajectories, seed);
        let mut fused_ws = GenerationWorkspace::<R, f32>::new();
        fused_ws.capture_carry_trace = true;
        let mut composed_ws = GenerationWorkspace::<R, f32>::new();
        composed_ws.capture_carry_trace = true;
        composed_ws.composed_step = true;
        let fused = model
            .generate(&batch, &table, &gcfg, &mut fused_ws, &constants)
            .unwrap();
        check_launches(&device).unwrap();
        let composed = model
            .generate(&batch, &table, &gcfg, &mut composed_ws, &constants)
            .unwrap();
        check_launches(&device).unwrap();
        fused.validate().unwrap();
        composed.validate().unwrap();
        // The discrete outcome is identical: the two forms differ by float
        // rounding far below the sampler's decision margins at these seeds.
        assert_eq!(fused.actions, composed.actions, "seed {seed}: actions");
        assert_eq!(fused.length, composed.length, "seed {seed}: lengths");
        assert_eq!(fused.status, composed.status, "seed {seed}: status");
        assert_eq!(fused.formula_row, composed.formula_row, "seed {seed}: formula rows");
        assert_eq!(fused.open_valence, composed.open_valence, "seed {seed}: open valence");
        assert_eq!(
            fused.request_status, composed.request_status,
            "seed {seed}: request status"
        );
        close(
            &fused.trace_log_prob,
            &composed.trace_log_prob,
            1e-4,
            "trace_log_prob",
        );
        close(
            &fused.formula_log_prob,
            &composed.formula_log_prob,
            1e-5,
            "formula_log_prob",
        );
        // Every step's post-freeze carries agree, so stopped rows were
        // frozen to the same values and live rows advanced alike.
        carries_close(&fused_ws.carry_trace, &composed_ws.carry_trace, 1e-4);
        for &len in &fused.length {
            if (len as usize) < max_steps as usize && len > 0 {
                stopped_rows += 1;
            } else {
                live_rows += 1;
            }
        }
        // Warmed launch counts of the two forms (carry capture off).
        let mut fw = GenerationWorkspace::<R, f32>::new();
        let mut cw = GenerationWorkspace::<R, f32>::new();
        cw.composed_step = true;
        for _ in 0..2 {
            model.generate(&batch, &table, &gcfg, &mut fw, &constants).unwrap();
            model.generate(&batch, &table, &gcfg, &mut cw, &constants).unwrap();
        }
        check_launches(&device).unwrap();
        reset_launch_count();
        model.generate(&batch, &table, &gcfg, &mut fw, &constants).unwrap();
        let lf = launch_count();
        reset_launch_count();
        model.generate(&batch, &table, &gcfg, &mut cw, &constants).unwrap();
        let lc = launch_count();
        check_launches(&device).unwrap();
        launches = (lf, lc);
    }
    println!(
        "fused step parity: {stopped_rows} trajectories stopped before the horizon, {live_rows} did not; warmed launches fused {} composed {}",
        launches.0, launches.1
    );
    assert!(
        stopped_rows > 0,
        "the seeds must exercise the carry freeze (some trajectory stops early)"
    );
    launches
}

#[test]
fn fused_generate_matches_composed_v0_shape() {
    let _serial = serial();
    let (fused, composed) = run_both(tiny_config(), 22, 4, &[7, 8, 9, 10]);
    assert!(
        fused * 2 < composed,
        "the fused step at least halves the launches of a call: {fused} against {composed}"
    );
}

#[test]
fn fused_generate_matches_composed_v1_capacities() {
    let _serial = serial();
    let (fused, composed) = run_both(v1_tiny_config(), 42, 3, &[11, 12]);
    assert!(
        fused * 2 < composed,
        "the fused step at least halves the launches of a call: {fused} against {composed}"
    );
}
