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
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model, StepCarries, set_scratch_arena};
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
        formula_evidence_work_max: 2048,
        formula_evidence_dispatch_max: 268435456,
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
        let unobserved = model.generate(&batch, &table, &gcfg, &mut fw, &constants).unwrap();
        let lf = launch_count();
        // With no carry trace the fused loop steps its recurrent state in
        // place and freezes nothing: the trajectories are the ones of the
        // observed loop, whose carries were compared above.
        unobserved.validate().unwrap();
        assert_eq!(unobserved.actions, fused.actions, "seed {seed}: unobserved actions");
        assert_eq!(unobserved.length, fused.length, "seed {seed}: unobserved lengths");
        assert_eq!(unobserved.status, fused.status, "seed {seed}: unobserved status");
        assert_eq!(
            unobserved.open_valence, fused.open_valence,
            "seed {seed}: unobserved open valence"
        );
        close(
            &unobserved.trace_log_prob,
            &fused.trace_log_prob,
            1e-4,
            "unobserved trace_log_prob",
        );
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

/// Reconstruct the functional `last_u` (`[batch, heads, head_dim, state]`,
/// row-major) from the in-place step's just-written factors: the activated
/// `x` scalars of `act` times the `B` column of `bc` per lane (the same
/// outer product the state kernels read).
fn last_u_from_factors(
    act: &[f32],
    bc: &[f32],
    batch: usize,
    heads: usize,
    head_dim: usize,
    state: usize,
) -> Vec<f32> {
    let d_inner = heads * head_dim;
    let act_width = act.len() / batch;
    let mut out = vec![0.0f32; batch * heads * head_dim * state];
    for b in 0..batch {
        for h in 0..heads {
            for c in 0..head_dim {
                let x = act[b * act_width + d_inner + h * head_dim + c];
                for s in 0..state {
                    out[((b * heads + h) * head_dim + c) * state + s] =
                        x * bc[(b * heads + h) * 2 * state + s];
                }
            }
        }
    }
    out
}

fn exact_or_close(
    inplace: &[f32],
    functional: &[f32],
    bit_equal: bool,
    what: &str,
) {
    assert_eq!(
        inplace.len(),
        functional.len(),
        "{what}: lengths {} against {}",
        inplace.len(),
        functional.len()
    );
    if bit_equal {
        assert!(
            inplace == functional,
            "{what}: in-place against functional is not bit-equal on the cpu runtime"
        );
        return;
    }
    for (i, (x, y)) in inplace.iter().zip(functional.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-6 * (1.0 + x.abs().max(y.abs())),
            "{what}[{i}]: in-place {x} against functional {y}"
        );
    }
}

/// One row-slice of a row-major `[rows, ...]` host buffer.
fn live_rows<T: Clone>(values: &[T], stride: usize, live: &[bool]) -> Vec<T> {
    let rows = live.len();
    assert_eq!(values.len(), rows * stride, "row-major buffer shape");
    live.iter()
        .enumerate()
        .filter(|(_, keep)| **keep)
        .flat_map(|(r, _)| values[r * stride..(r + 1) * stride].to_vec())
        .collect()
}

/// P5.9: the in-place recurrent step against the functional step over a full
/// generation — every carry (`h`, `last_u` as its factors, `angle`, the
/// convolution history, the atom memory) after every step, at both capacity
/// shapes: bit-equal on the cpu runtime, within 1e-6 on a GPU backend.
///
/// Both states run the same fused step (`step_packed`) on the same inputs;
/// only the carry representation differs (stepped-in-place buffers against
/// per-step caches). The functional side is the observed fused state (the
/// carry trace of the parity tests); the in-place side is what production
/// drives when the trace is off.
///
/// Stopped rows are compared only while live: the functional path freezes a
/// stopped row's carries (they stay exactly as they were) while the
/// in-place path keeps stepping them unread — they differ by construction,
/// which is the point of the optimisation (nothing reads them: the sampler
/// ignores the row's logits from then on). Liveness is the exact freeze
/// predicate: the grammar state's stopped-flag column (`3A + 5`) read after
/// each step; a row with a nonzero flag was frozen and is excluded.
#[test]
fn in_place_carries_match_functional_every_step() {
    let _serial = serial();
    // The carry comparison runs with the scratch arena off: recycled
    // buffers hold stale bytes, and this test must attribute any
    // difference to the step kernels, not the allocator.
    set_scratch_arena(false);
    for (make_config, max_steps, trajectories) in
        [(tiny_config as fn() -> ModelConfig, 22u32, 8u32), (v1_tiny_config as fn() -> ModelConfig, 42u32, 6u32)]
    {
        for seed in [7u64, 8, 9] {
        let device = mamba3::backend::Device::<R>::default();
        let bit_equal = device.name() == "cpu";
        let mut cfg = make_config();
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
        let gcfg = generation(max_steps, trajectories, seed);        // Task F10 item A3: each staged call carries its own latched
        // mode — the capture workspace preflights the functional mode,
        // the plain workspace the production mode.
        let pre_func = model
            .generate_preflight_with_decode_mode(&batch, &table, &gcfg, false, true)
            .unwrap();
        let pre_in = model
            .generate_preflight_with_decode_mode(&batch, &table, &gcfg, false, false)
            .unwrap();
        let pre = model.generate_preflight(&batch, &table, &gcfg).unwrap();
        assert_eq!(pre_func.rows, pre.rows);
        assert_eq!(pre_in.rows, pre.rows);
        assert!(!pre_func.decode_in_place);
        assert!(pre_in.decode_in_place);
        let spectra = model.generate_preprocess(&batch, &gcfg, &device).unwrap();
        let mut ws_func = GenerationWorkspace::<R, f32>::new();
        ws_func.capture_carry_trace = true;
        let mut ws_in = GenerationWorkspace::<R, f32>::new();
        let encoded_f = model
            .generate_encode_ws(&mut ws_func, &spectra, gcfg.control, &pre_func, &device)
            .unwrap();
        let encoded_i = model
            .generate_encode_ws(&mut ws_in, &spectra, gcfg.control, &pre_in, &device)
            .unwrap();
        model
            .generate_search_ws(
                &mut ws_func,
                &spectra,
                &batch,
                &encoded_f.pool,
                &table,
                pre_func.spectra_n,
                pre_func.trajectories,
                pre_func.formulas,
                false,
                gcfg.formula_rows_visited_max,
                gcfg.formula_rows_scored_max,
                &gcfg,
                &pre_func,
                &device,
            )
            .unwrap();
        model
            .generate_search_ws(
                &mut ws_in,
                &spectra,
                &batch,
                &encoded_i.pool,
                &table,
                pre_in.spectra_n,
                pre_in.trajectories,
                pre_in.formulas,
                false,
                gcfg.formula_rows_visited_max,
                gcfg.formula_rows_scored_max,
                &gcfg,
                &pre_in,
                &device,
            )
            .unwrap();
        let (mut state_f, bonds_f, traj_f) = model
            .generate_decoder_init_ws(&mut ws_func, &encoded_f, &pre_func, &device)
            .unwrap();
        let (mut state_i, bonds_i, traj_i) = model
            .generate_decoder_init_ws(&mut ws_in, &encoded_i, &pre_in, &device)
            .unwrap();
        assert!(
            !state_f.carries_in_place(),
            "the observed state keeps per-step caches"
        );
        assert!(
            state_i.carries_in_place(),
            "the unobserved state steps its carries in place"
        );
        let seed_lo = (seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (seed >> 32) as u32;
        let rows = pre.rows;
        // Per-step snapshots as host floats, compared below with the exact
        // freeze predicate as the liveness mask: the grammar state's
        // stopped-flag column read after each step.
        // (functional trace, freeze flags, in-place h/act/bc/angle/history,
        // and both sides' atom keys, previous heads, previous outputs and
        // residual ids.)
        #[allow(clippy::type_complexity)]
        let mut snaps: Vec<(
            StepCarries,
            Vec<bool>,
            Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>, Vec<usize>)>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<u32>,
            Vec<u32>,
        )> = Vec::new();
        for step in 1..pre.steps {
            let carry_f = model
                .generate_decode_step_ws(
                    &mut ws_func,
                    &encoded_f,
                    &traj_f,
                    &mut state_f,
                    &bonds_f,
                    &constants.atom_table,
                    step,
                    seed_lo,
                    seed_hi,
                    gcfg.temperature,
                    pre_func.trajectories,
                    &pre_func,
                    &device,
                )
                .unwrap()
                .expect("capture is on");
            model
                .generate_decode_step_ws(
                    &mut ws_in,
                    &encoded_i,
                    &traj_i,
                    &mut state_i,
                    &bonds_i,
                    &constants.atom_table,
                    step,
                    seed_lo,
                    seed_hi,
                    gcfg.temperature,
                    pre_in.trajectories,
                    &pre_in,
                    &device,
                )
                .unwrap();
            check_launches(&device).unwrap();
            let inplace = state_i.in_place_carries().expect("carries are in place");
            assert_eq!(
                inplace.len(),
                carry_f.layers.len(),
                "step {step}: one in-place carry per functional layer"
            );
            let mut layers = Vec::with_capacity(inplace.len());
            for tensors in &inplace {
                layers.push((
                    tensors.h.try_to_f32().unwrap(),
                    tensors.act.try_to_f32().unwrap(),
                    tensors.bc.try_to_f32().unwrap(),
                    tensors.angle.as_ref().map(|a| a.try_to_f32().unwrap()),
                    tensors.history.as_ref().map(|h| h.try_to_f32().unwrap()),
                    tensors.h.shape().dims().to_vec(),
                ));
            }
            let fused_f = state_f.fused.as_ref().expect("functional state is fused");
            let fused_i = state_i.fused.as_ref().expect("in-place state is fused");
            // The exact freeze predicate for this step: rows whose
            // stopped flag (`3A + 5`) is set were frozen and are excluded
            // from the comparison.
            let state_width = 3 * pre.atoms + 16;
            let stop_col = 3 * pre.atoms + 5;
            let flags = ws_func
                .debug_grammar_state()
                .expect("a bucket is cached")
                .try_to_vec()
                .unwrap();
            let live: Vec<bool> = (0..rows)
                .map(|r| flags[r * state_width + stop_col] == 0)
                .collect();
            snaps.push((
                carry_f,
                live,
                layers,
                fused_i.atom_keys.try_to_f32().unwrap(),
                fused_f.atom_keys.try_to_f32().unwrap(),
                fused_i.prev_heads().try_to_f32().unwrap(),
                fused_f.prev_heads().try_to_f32().unwrap(),
                state_i.prev_h.try_to_f32().unwrap(),
                state_f.prev_h.try_to_f32().unwrap(),
                state_i.resid_ids.try_to_vec().unwrap(),
                state_f.resid_ids.try_to_vec().unwrap(),
            ));
        }
        let mut live_total = 0usize;
        let mut stopped_total = 0usize;
        let mut max_live_step = 0usize;
        for (step_idx, (carry_f, live, layers, atom_keys_i, atom_keys_f, prev_heads_i, prev_heads_f, prev_h_i, prev_h_f, resid_i, resid_f)) in
            snaps.iter().enumerate()
        {
            let step = step_idx + 1;
            let live_count = live.iter().filter(|&&v| v).count();
            if live_count > 0 {
                max_live_step = step;
            }
            live_total += live.iter().filter(|&&v| v).count();
            stopped_total += live.iter().filter(|&&v| !v).count();
            assert_eq!(layers.len(), carry_f.layers.len());
            for (l, ((h_i, act_i, bc_i, angle_i, hist_i, h_dims), layer)) in
                layers.iter().zip(carry_f.layers.iter()).enumerate()
            {
                let at = format!("step {step} layer {l}");
                assert_eq!(h_dims.len(), 4, "{at}: h is [batch, heads, head_dim, state]");
                let (b, heads, head_dim, state) = (h_dims[0], h_dims[1], h_dims[2], h_dims[3]);
                assert_eq!(b, rows, "{at}: carry batch is the trajectory rows");
                exact_or_close(
                    &live_rows(h_i, h_i.len() / rows, &live),
                    &live_rows(&layer.h, layer.h.len() / rows, &live),
                    bit_equal,
                    &format!("{at} h"),
                );
                let last_u = last_u_from_factors(act_i, bc_i, b, heads, head_dim, state);
                exact_or_close(
                    &live_rows(&last_u, last_u.len() / rows, &live),
                    &live_rows(&layer.last_u, layer.last_u.len() / rows, &live),
                    bit_equal,
                    &format!("{at} last_u"),
                );
                match (angle_i, &layer.angle) {
                    (Some(a), Some(b)) => exact_or_close(
                        &live_rows(a, a.len() / rows, &live),
                        &live_rows(b, b.len() / rows, &live),
                        bit_equal,
                        &format!("{at} angle"),
                    ),
                    (None, None) => {}
                    _ => panic!("{at}: angle present in one form only"),
                }
                match (hist_i, &layer.conv) {
                    (Some(a), Some(b)) => exact_or_close(
                        &live_rows(a, a.len() / rows, &live),
                        &live_rows(b, b.len() / rows, &live),
                        bit_equal,
                        &format!("{at} conv history"),
                    ),
                    (None, None) => {}
                    _ => panic!("{at}: conv history present in one form only"),
                }
            }
            // The atom memory both fused states keep projected: identical
            // rows, previous outputs and residual ids after every step.
            // `atom_keys`/`prev_heads`/`prev_h` are row-major with the row
            // as the outer axis, so the live-row mask applies; `resid_ids`
            // are refreshed identically for every row (no freeze), so all
            // rows compare.
            let at = format!("step {step} atom memory");
            exact_or_close(
                &live_rows(atom_keys_i, atom_keys_i.len() / rows, &live),
                &live_rows(atom_keys_f, atom_keys_f.len() / rows, &live),
                bit_equal,
                &at,
            );
            exact_or_close(
                &live_rows(prev_heads_i, prev_heads_i.len() / rows, &live),
                &live_rows(prev_heads_f, prev_heads_f.len() / rows, &live),
                bit_equal,
                &format!("step {step} prev heads"),
            );
            exact_or_close(
                &live_rows(prev_h_i, prev_h_i.len() / rows, &live),
                &live_rows(prev_h_f, prev_h_f.len() / rows, &live),
                bit_equal,
                &format!("step {step} prev_h"),
            );
            assert_eq!(
                resid_i, resid_f,
                "step {step}: residual ids"
            );
        }
        assert!(
            live_total > 0,
            "the comparison must cover live rows (else it is vacuous)"
        );
        assert_eq!(
            max_live_step,
            pre.steps - 1,
            "stopping is absorbing, so a live row at the horizon means every step compared live rows"
        );
        println!(
            "in-place carries match functional every step (seed {seed}, {} steps, {live_total} live-row checks, {stopped_total} stopped-row exclusions, last live step {max_live_step}, bit_equal={bit_equal})",
            pre.steps - 1
        );
        }
    }
    set_scratch_arena(true);
}

#[test]
fn composed_freeze_in_place_matches_select_add_bitwise() {
    // Task F10 item A2 (hard rule): the composed path's in-place freeze
    // computes bit-identical values to the previous select-and-add freeze
    // (`select_valid` of the alive/dead expansions, summed) on the cpu
    // runtime, for B·K in {8, 64} with both formula sources. The old
    // formula is replicated here as the reference oracle; the production
    // path under test is the in-place `freeze_rows` selection the composed
    // step now shares with the fused path. Bitwise equality holds because
    // both write exact copies of one input per row (the sum's `+0.0` is the
    // identity on the finite, non-negative-zero state this probe holds —
    // and the test would fail loudly if that ever stopped being true).
    let _serial = serial();
    set_scratch_arena(false);
    use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
    use mamba3::tensor::Shape;
    use mamba3::tensor::ops::elemwise;
    use mamba3::tensor::ops::index::{ids_to_float, slice_ids_along};
    use mamba3::tensor::ops::ms2 as ms2ops;
    for (b, k) in [(2usize, 4u32), (8usize, 8u32)] {
        for source in [FormulaSource::Table, FormulaSource::Enumerate] {
            let device = mamba3::backend::Device::<R>::default();
            let mut cfg = tiny_config();
            let comps: Vec<Composition> = vec![
                [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
                [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
            ];
            let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
            let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
            cfg.formula_table.rows = table.rows as u32;
            cfg.formula_table.sha256 = table.sha256.clone();
            let mut rng = Rng::seeded(41 + b as u64);
            let mut model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
            if source == FormulaSource::Enumerate {
                let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
                let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
                model.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
            }
            let constants = Ms2Constants::new(&device);
            let precursors: Vec<u32> = (0..b)
                .map(|i| {
                    mamba3::models::ms2::chem::composition_mass(&comps[i % comps.len()]).unwrap()
                        + 1_007_825
                        - 549
                })
                .collect();
            let ids: Vec<u64> = (0..b as u64).map(|i| 610 + i).collect();
            let batch = make_spectra(&ids, &precursors, 64, &vec![10u32; b], 42);
            let mut gcfg = generation(22, k, 43);
            gcfg.formula_source = source;
            // Composed + observed (caches, no in-place stepping): the
            // reference freeze's old bank, new bank and grammar state.
            let pre = model
                .generate_preflight_with_decode_mode(&batch, &table, &gcfg, true, true)
                .unwrap();
            assert!(!pre.decode_in_place);
            let mut ws = GenerationWorkspace::<R, f32>::new();
            ws.composed_step = true;
            ws.capture_carry_trace = true;
            let spectra = model.generate_preprocess(&batch, &gcfg, &device).unwrap();
            let encoded = model
                .generate_encode_ws(&mut ws, &spectra, gcfg.control, &pre, &device)
                .unwrap();
            model
                .generate_search_ws(
                    &mut ws,
                    &spectra,
                    &batch,
                    &encoded.pool,
                    &table,
                    pre.spectra_n,
                    pre.trajectories,
                    pre.formulas,
                    false,
                    gcfg.formula_rows_visited_max,
                    gcfg.formula_rows_scored_max,
                    &gcfg,
                    &pre,
                    &device,
                )
                .unwrap();
            let (mut state, bonds, traj) = model
                .generate_decoder_init_ws(&mut ws, &encoded, &pre, &device)
                .unwrap();
            assert!(!state.carries_in_place());
            let rows = pre.rows;
            assert_eq!(rows, b * k as usize);
            let seed_lo = 43u32;
            let seed_hi = 0u32;
            for step in [1usize, 2] {
                let old_caches = state.caches.clone();
                model
                    .generate_decode_step_ws(
                        &mut ws,
                        &encoded,
                        &traj,
                        &mut state,
                        &bonds,
                        &constants.atom_table,
                        step,
                        seed_lo,
                        seed_hi,
                        gcfg.temperature,
                        pre.trajectories,
                        &pre,
                        &device,
                    )
                    .unwrap();
                check_launches(&device).unwrap();
                // The production result: the new bank frozen in place.
                let new_caches = state.caches.clone();
                // The reference oracle: the removed select-and-add freeze.
                let replay = ws.debug_grammar_state().expect("bucket cached");
                let stopped_ids =
                    slice_ids_along(&replay, 1, 3 * pre.atoms + 5, 1).unwrap().reshape(vec![rows]).unwrap();
                let stopped_f = ids_to_float(&stopped_ids);
                let alive = elemwise::eq_scalar(&stopped_f, 0.0);
                let dead = elemwise::rsub_scalar(&alive, 1.0);
                let old_freeze = |new_t: &mamba3::tensor::Tensor<R, f32>,
                                  old_t: &mamba3::tensor::Tensor<R, f32>|
                 -> Vec<u32> {
                    let dims = new_t.shape().dims().to_vec();
                    let mut vdims = dims.clone();
                    vdims.pop();
                    let mut ashape = vec![rows];
                    ashape.extend(vec![1; vdims.len() - 1]);
                    let alive_v = elemwise::expand(
                        &alive.reshape(Shape::new(ashape.clone())).unwrap(),
                        &Shape::new(vdims.clone()),
                    )
                    .unwrap();
                    let dead_v = elemwise::expand(
                        &dead.reshape(Shape::new(ashape)).unwrap(),
                        &Shape::new(vdims),
                    )
                    .unwrap();
                    let kept_new = ms2ops::select_valid(new_t, &alive_v).unwrap();
                    let kept_old = ms2ops::select_valid(old_t, &dead_v).unwrap();
                    elemwise::add(&kept_new, &kept_old).unwrap().to_data().iter().map(|v| v.to_bits()).collect()
                };
                for (layer, (old, new)) in old_caches.iter().zip(new_caches.iter()).enumerate() {
                    // `new` is already frozen in place, but that is no
                    // obstacle: on live rows the frozen bank IS the
                    // pre-freeze new bank (untouched), and on stopped rows
                    // the oracle ignores its `new` input — so the oracle on
                    // (frozen, old) equals the old formula on (pre-freeze
                    // new, old). The assert below therefore compares the old
                    // and new formulas on identical inputs.
                    let pairs: Vec<(
                        &mamba3::tensor::Tensor<R, f32>,
                        &mamba3::tensor::Tensor<R, f32>,
                    )> = {
                        let mut v = Vec::with_capacity(4);
                        v.push((new.ssm.h.tensor(), old.ssm.h.tensor()));
                        v.push((new.ssm.last_u.tensor(), old.ssm.last_u.tensor()));
                        if let (Some(n), Some(o)) = (&new.ssm.angle, &old.ssm.angle) {
                            v.push((n.tensor(), o.tensor()));
                        }
                        if let (Some(n), Some(o)) = (&new.conv, &old.conv) {
                            v.push((n.tensor(), o.tensor()));
                        }
                        v
                    };
                    for (t, (new_t, old_t)) in pairs.iter().enumerate() {
                        // The production result, and the reference oracle on
                        // the same inputs: bitwise equality is the rule's
                        // demand.
                        let frozen_bits: Vec<u32> =
                            new_t.to_data().iter().map(|v| v.to_bits()).collect();
                        let oracle_bits = old_freeze(new_t, old_t);
                        assert_eq!(
                            oracle_bits, frozen_bits,
                            "B*K={} {source:?} step {step} layer {layer} tensor {t}: select-and-add oracle equals the in-place freeze bit-for-bit",
                            b * k as usize,
                        );
                    }
                }
            }
        }
    }
    set_scratch_arena(true);
}
