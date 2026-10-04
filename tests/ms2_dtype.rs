//! K9 (plan item P7.7): the reduced-precision capability matrix of the MS2 model.
//!
//! Design reference: `docs/MS2_SUBSTRUCTURE_DESIGN.md` §4.2 last paragraph —
//! exact-mass decisions live on the integer sidecar and never read a float of
//! the model dtype; reduced precision applies only after feature construction;
//! unsupported dtypes produce actionable errors, never a silent fallback.
//!
//! Validated set (contracts §3.3): f32 everywhere, bf16 where the device
//! supports it (the CPU runtime), f16 nowhere. For each validated dtype this
//! suite checks capability reporting, exact-mass integer identity with f32,
//! finite falling training loss, and a validating warmed `generate`. f16 is
//! refused on every backend with `Error::Unsupported` naming the dtype, the
//! backend and the reason — asserted as `Err`, so no test panics inside the
//! library.
//!
//! ```text
//! cargo test --release --no-default-features --features cpu --test ms2_dtype
//! ```

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{
    DType, Device, FloatElem, check_launches,
};
use mamba3::backends::Auto;
use mamba3::error::{Error, Result};
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{Composition, composition_mass, element_index};
use mamba3::models::ms2::contract::{
    AllocationMode, Control, FormulaSource, GenerationConfig, GenerationMode, IdentityMode,
    ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::decoder::{Ms2Decoder, ReplayView, graph_loss};
use mamba3::models::ms2::encoder::Ms2Encoder;
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::formula_head::{DeviceFormulaTable, FormulaHead, gold_slots_host};
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::grammar::{Limits, Token};
use mamba3::models::ms2::targets::{Labels, Target};
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::models::ms2::workspace::{Ms2Capabilities, TimingMethod};
use mamba3::nn::Module;
use mamba3::nn::param::Param;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, FormulaBuffers, Ms2Constants, PeakBuffers, ReplayBuffers};
use mamba3::tensor::ops::random::Rng;
use mamba3::train::optim::{AdamWConfig, Optimizer};

type R = Auto;

/// Training steps of the tiny overfit loop per dtype.
const TRAIN_STEPS: usize = 60;

/// Net hydrogen shift of contract §4.3, `m_H - m_e` (as in `overfit_smoke`).
const H_NET: u32 = 1_007_825 - 549;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn fixture() -> serde_json::Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/chemistry_v0.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn token_of(t: &serde_json::Value) -> Token {
    let a = t.as_array().expect("token array");
    Token {
        kind: a[0].as_u64().expect("kind") as u8,
        atom_type: a[1].as_u64().expect("atom_type") as u8,
        bond: a[2].as_u64().expect("bond") as u8,
        pointer: a[3].as_u64().expect("pointer") as u8,
    }
}

fn trace_of(t: &serde_json::Value) -> Vec<Token> {
    t.as_array()
        .expect("trace array")
        .iter()
        .map(token_of)
        .collect()
}

fn composition_of(formula: &serde_json::Value) -> Composition {
    let mut c: Composition = [0; 10];
    for (symbol, count) in formula.as_object().expect("formula object") {
        let e = element_index(symbol).expect("known element");
        c[e] = count.as_u64().expect("count") as u16;
    }
    c
}

/// A single-target [`Labels`] with weight 1 on `trace`.
fn single_target(trace: Vec<Token>) -> Labels {
    Labels {
        embeddings: Vec::new(),
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![Target {
            trace,
            weight: 1,
            q: 1.0,
            embeddings: Vec::new(),
            anchors: Vec::new(),
        }],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: Vec::new(),
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    }
}

/// Tiny model config (`d = 16`, 1 encoder/decoder block, `N = 16`), stamped
/// with the compute dtype under test so memory estimates price it.
fn tiny_config<E: FloatElem>() -> ModelConfig {
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
    m.dtype = E::DTYPE;
    m
}

/// The `[M+H]+` precursor of a neutral composition.
fn precursor_of(comp: &Composition) -> u32 {
    composition_mass(comp).unwrap() + H_NET
}

/// Two small training molecules and their parents.
fn train_molecules() -> (Vec<Composition>, Vec<Labels>) {
    let f = fixture();
    let names = ["ethanol", "acetonitrile"];
    let mols: Vec<serde_json::Value> = names
        .iter()
        .map(|n| {
            f["molecules"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["name"] == *n)
                .unwrap()
                .clone()
        })
        .collect();
    let parents: Vec<Composition> = mols.iter().map(|m| composition_of(&m["formula"])).collect();
    let labs: Vec<Labels> = mols
        .iter()
        .map(|m| single_target(trace_of(&m["whole_trace"]["trace"])))
        .collect();
    (parents, labs)
}

/// Deterministic 2-spectrum batch with precursors from the true parents, so
/// every gold formula joins the window.
fn make_batch(parents: &[Composition], spectrum_ids: &[u64], seed: u64) -> SpectrumBatch {
    let b = spectrum_ids.len();
    let n_raw = 64usize;
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    let mut precursor = vec![0u32; b];
    for (bi, parent) in parents.iter().enumerate() {
        precursor[bi] = precursor_of(parent);
        let n = 20usize;
        peak_count[bi] = n as u32;
        raw_peak_count[bi] = n as u32;
        for i in 0..n {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursor[bi] - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + 2.0 * rng.uniform_vec(1, 0.0, 1.0)[0];
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
        precursor_mz_udalton: precursor,
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

/// Formula table of the training parents plus two decoys.
fn train_table(parents: &[Composition]) -> FormulaTable {
    let mut comps = parents.to_vec();
    comps.push([6, 6, 0, 0, 0, 0, 0, 0, 0, 0]);
    comps.push([2, 6, 0, 1, 0, 0, 0, 0, 0, 0]);
    FormulaTable::from_compositions(comps).unwrap()
}

/// One step's training outcome, read back on the host every step.
struct TrainCurve {
    /// Total loss per step (`L_graph + 0.2 * L_formula`, pre-update).
    losses: Vec<f32>,
    /// Max `|gradient|` over every parameter per step (non-finite allowed).
    grad_max: Vec<f32>,
    /// First step with a non-finite loss or gradient, if any.
    first_nonfinite: Option<usize>,
    /// `"loss"` or `"grad:<param>"` for [`TrainCurve::first_nonfinite`].
    first_nonfinite_where: Option<String>,
}

/// `steps` steps of the tiny `overfit_smoke` loop in element type `E`:
/// encoder + formula head + decoder, AdamW lr 3e-3, oracle gold conditioning.
/// The loss is read every step (not every 50) and every gradient tensor is
/// read back, so finiteness is observed, not assumed.
fn train_curve<E: FloatElem>(steps: usize) -> TrainCurve {
    let device = dev();
    let (parents, labs) = train_molecules();
    let b = parents.len();
    let spectra_batch = make_batch(&parents, &[101, 102], 31);
    let table = train_table(&parents);
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let model = tiny_config::<E>();
    let d = model.d_model as usize;
    let mut rng = Rng::seeded(41);
    let encoder = Ms2Encoder::init(&model, &device, &mut rng).unwrap();
    let formula_head = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let decoder = Ms2Decoder::init(&model, &device, &mut rng).unwrap();
    let spectra = DeviceSpectra::upload(&spectra_batch, &device).unwrap();
    let peaks = PeakBuffers::<R, E>::new(b, 64, 16, &device);
    let m_window = 32usize;
    let mut buffers = FormulaBuffers::<R, E>::new(b, m_window, 4, &device);
    let mut search = vec![0u32; table.len() * 2];
    for row in 0..table.len() {
        search[row * 2] = table.mass(row);
        search[row * 2 + 1] =
            (mamba3::models::ms2::chem::composition_error_nda(table.composition(row)).div_ceil(1000))
                as u32;
    }
    let search_t = IdTensor::from_slice(&search, vec![table.len(), 2], &device).unwrap();
    let refs: Vec<Option<&Labels>> = labs.iter().map(Some).collect();
    let targets_batch = TargetBatch::build(&refs, &parents, 2, Limits::V0).unwrap();
    let targets = targets_batch.upload(&device).unwrap();
    let constants = Ms2Constants::new(&device);
    let t = Limits::V0.max_steps();
    let rows = b * 2;
    let replay_buffers = ReplayBuffers::poisoned(rows, t, 16, &device).unwrap();
    ms2::grammar_replay(
        &targets.tokens,
        &targets.meta,
        &constants,
        16,
        4,
        &replay_buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let precursor: Vec<u32> = parents.iter().map(precursor_of).collect();
    let queries: Vec<WindowQuery> = precursor
        .iter()
        .map(|&p| WindowQuery {
            precursor_mz: p,
            adduct: 1,
            ppm_tenths: 200,
            precursor_uncertainty: 50,
            rows_visited_max: u32::MAX,
            rows_scored_max: 4096,
        })
        .collect();
    let mut gold_rows = vec![u32::MAX; b];
    for (bi, parent) in parents.iter().enumerate() {
        for row in 0..table.len() {
            if table.composition(row) == parent {
                gold_rows[bi] = row as u32;
                break;
            }
        }
        assert_ne!(gold_rows[bi], u32::MAX, "parent {bi} is in the table");
    }
    let gold_slots = gold_slots_host(&table, &queries, &gold_rows, m_window);
    assert!(
        gold_slots.iter().all(|&s| s != u32::MAX),
        "every gold is scored: {gold_slots:?}"
    );
    let mut opt = AdamWConfig::builder()
        .learning_rate(3e-3)
        .build()
        .init::<R, E>();
    let mut params = encoder.named_parameters();
    params.extend(formula_head.named_parameters());
    params.extend(decoder.named_parameters());
    let only_values: Vec<Param<R, E>> = params.iter().map(|(_, p)| p.clone()).collect();
    let replay = ReplayView {
        replay: &replay_buffers.replay,
        atoms: &replay_buffers.atoms,
    };
    let mut losses = Vec::with_capacity(steps);
    let mut grad_max = Vec::with_capacity(steps);
    let mut first_nonfinite = None;
    let mut first_nonfinite_where = None;
    for step in 0..steps {
        let encoded = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        ms2::formula_window(
            &search_t,
            &spectra.meta,
            table.max_error(),
            u32::MAX,
            4096,
            &buffers,
        )
        .unwrap();
        ms2::formula_gather(
            &buffers.window,
            &uploaded.table,
            &uploaded.counts,
            &mut buffers.cand,
        )
        .unwrap();
        ms2::count_features(
            &buffers.cand.reshape(vec![b * m_window, 13]).unwrap(),
            &uploaded.log_table,
            &mut buffers.cand_feat.reshape(vec![b * m_window, 10]).unwrap(),
            13,
        )
        .unwrap();
        let scored = formula_head.score(&buffers, &encoded.pool).unwrap();
        let formula_loss = formula_head.loss(&scored, &IdTensor::from_slice(&gold_slots, vec![b], &device).unwrap()).unwrap();
        let gold_ids = IdTensor::from_slice(&gold_slots, vec![b], &device).unwrap();
        let e_gold = Var::gather_tokens(&scored.embedding, &gold_ids, 1)
            .unwrap()
            .reshape(vec![b, d])
            .unwrap();
        let tout = decoder
            .teacher(&encoded, &e_gold, &targets, &replay)
            .unwrap();
        let gloss = graph_loss(&tout, &targets.q, b).unwrap();
        let total = gloss.add(&formula_loss.mul_scalar(0.2)).unwrap();
        let loss_v = total.try_to_f32().unwrap()[0];
        losses.push(loss_v);
        if !loss_v.is_finite() && first_nonfinite.is_none() {
            first_nonfinite = Some(step);
            first_nonfinite_where = Some("loss".to_string());
        }
        let grads = total.backward_retain().unwrap();
        let mut step_max = 0.0f32;
        for (name, param) in &params {
            let g = grads
                .get(param.id())
                .unwrap_or_else(|| panic!("no gradient for {name}"))
                .try_to_f32()
                .unwrap();
            for v in &g {
                if !v.is_finite() && first_nonfinite.is_none() {
                    first_nonfinite = Some(step);
                    first_nonfinite_where = Some(format!("grad:{name}"));
                }
                step_max = step_max.max(v.abs());
            }
        }
        grad_max.push(step_max);
        opt.step(&only_values, &grads).unwrap();
    }
    check_launches(&device).unwrap();
    TrainCurve {
        losses,
        grad_max,
        first_nonfinite,
        first_nonfinite_where,
    }
}

/// Final loss and its ratio to the f32 run, for the log.
fn summarize(name: &str, curve: &TrainCurve, f32_final: f32) {
    let final_loss = *curve.losses.last().unwrap();
    let initial = curve.losses[0];
    println!(
        "{name}: {TRAIN_STEPS} steps, loss {initial:.6} -> {final_loss:.6} \
         (ratio to initial {:.4}, ratio to f32 final {:.4}), max|grad| {:.3e}, \
         first non-finite: {:?} ({:?})",
        final_loss / initial,
        final_loss / f32_final,
        curve
            .grad_max
            .iter()
            .fold(0.0f32, |m, v| m.max(*v)),
        curve.first_nonfinite,
        curve.first_nonfinite_where,
    );
}

#[test]
fn capability_matrix_names_dtype_support() {
    // Point 1: the validated set is f32 on every backend, bf16 on the CPU
    // backend only, f16 nowhere — independent of hardware capability.
    // `check` refuses anything outside it with `Error::Unsupported` naming
    // the dtype and saying it is "not validated for this backend" — never a
    // silent fallback to another precision, and never a kernel launch.
    let device = dev();
    let backend = device.name();
    let caps = Ms2Capabilities::probe(&device);
    println!(
        "backend {}: f32={} f16={} bf16={} max_bindings={} plane_size_max={}",
        caps.backend, caps.f32_supported, caps.f16_supported, caps.bf16_supported,
        caps.max_bindings, caps.plane_size_max,
    );
    assert!(caps.f32_supported, "f32 is always supported");
    // f32 validates everywhere.
    let mut cfg = tiny_config::<f32>();
    cfg.dtype = DType::F32;
    caps.check(&cfg)
        .unwrap_or_else(|e| panic!("f32 refused on {backend}: {e}"));
    // bf16 validates on the CPU backend only, whatever the device reports
    // (the allowlist is independent of hardware capability).
    let mut cfg = tiny_config::<f32>();
    cfg.dtype = DType::BF16;
    if backend == "cpu" {
        caps.check(&cfg)
            .unwrap_or_else(|e| panic!("bf16 refused on the CPU backend: {e}"));
    } else {
        let err = caps
            .check(&cfg)
            .expect_err("bf16 off the CPU backend must be refused");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "unsupported dtype refusal is Error::Unsupported: {err:?}"
        );
        assert!(
            err.to_string().contains("not validated for this backend"),
            "the refusal states bf16 is not validated for this backend: {err}"
        );
    }
    // f16 is refused everywhere, even where the runtime reports support.
    let mut cfg = tiny_config::<f32>();
    cfg.dtype = DType::F16;
    let err = caps
        .check(&cfg)
        .expect_err("f16 must be refused on every backend");
    assert!(
        matches!(err, Error::Unsupported(_)),
        "f16 refusal is Error::Unsupported: {err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains("f16"), "the refusal names the dtype: {msg}");
    assert!(
        msg.contains("not validated"),
        "the refusal states f16 is not validated: {msg}"
    );
    // The refusal wording is pinned even when the hardware supports every
    // dtype (as the CPU runtime does), via a denied capability set.
    let denied = Ms2Capabilities {
        backend: "test-backend".to_string(),
        f32_supported: true,
        f16_supported: true,
        bf16_supported: false,
        max_bindings: 64,
        reports_memory: false,
        timing: TimingMethod::SystemTime,
        reports_reserved_bytes: false,
        plane_size_max: 1,
    };
    let mut cfg = tiny_config::<f32>();
    cfg.dtype = DType::F16;
    let err = denied
        .check(&cfg)
        .expect_err("f16 must be refused even when reported supported");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(
        err.to_string().contains("not validated"),
        "the f16 refusal states it is not validated: {err}"
    );
    let mut cfg = tiny_config::<f32>();
    cfg.dtype = DType::BF16;
    let err = denied
        .check(&cfg)
        .expect_err("bf16 on a non-CPU backend must be refused");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(
        err.to_string().contains("not validated for this backend"),
        "the refusal states bf16 is not validated for this backend: {err}"
    );
}

/// Integer snapshots of the exact-mass path for one element type: peak
/// selection (`rank`, `position`), the formula window (`window`,
/// `counters`) and the candidate compositions (`cand`, before neural
/// scoring). All three kernels write these buffers from integer inputs; the
/// comparison against f32 pins that no model-dtype float leaks into them.
struct IntegerSnapshots {
    rank: Vec<u32>,
    position: Vec<u32>,
    window: Vec<u32>,
    counters: Vec<u32>,
    cand: Vec<u32>,
}

fn integer_snapshots<E: FloatElem>() -> IntegerSnapshots {
    let device = dev();
    let (parents, _) = train_molecules();
    let spectra_batch = make_batch(&parents, &[101, 102], 31);
    let table = train_table(&parents);
    let spectra = DeviceSpectra::<R, E>::upload(&spectra_batch, &device).unwrap();
    // Peak selection.
    let peaks = PeakBuffers::<R, E>::new(2, 64, 16, &device);
    ms2::peak_select(
        &spectra.mz,
        &spectra.intensity,
        &spectra.meta,
        u32::from(spectra_batch.intensity_scale),
        &peaks,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let rank = peaks.rank.try_to_vec().unwrap();
    let position = peaks.position.try_to_vec().unwrap();
    // Formula window + candidate gather.
    let mut buffers = FormulaBuffers::<R, E>::new(2, 32, 4, &device);
    let mut search = vec![0u32; table.len() * 2];
    for row in 0..table.len() {
        search[row * 2] = table.mass(row);
        search[row * 2 + 1] =
            (mamba3::models::ms2::chem::composition_error_nda(table.composition(row)).div_ceil(1000))
                as u32;
    }
    let search_t = IdTensor::from_slice(&search, vec![table.len(), 2], &device).unwrap();
    ms2::formula_window(
        &search_t,
        &spectra.meta,
        table.max_error(),
        u32::MAX,
        4096,
        &buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let window = buffers.window.try_to_vec().unwrap();
    let counters = buffers.counters.try_to_vec().unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    ms2::formula_gather(
        &buffers.window,
        &uploaded.table,
        &uploaded.counts,
        &mut buffers.cand,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let cand = buffers.cand.try_to_vec().unwrap();
    IntegerSnapshots {
        rank,
        position,
        window,
        counters,
        cand,
    }
}

fn assert_snapshots_equal(want: &IntegerSnapshots, got: &IntegerSnapshots, name: &str) {
    assert_eq!(got.rank, want.rank, "{name}: peak rank differs from f32");
    assert_eq!(
        got.position, want.position,
        "{name}: peak positions differ from f32"
    );
    assert_eq!(
        got.window, want.window,
        "{name}: formula window differs from f32"
    );
    assert_eq!(
        got.counters, want.counters,
        "{name}: formula counters differ from f32"
    );
    assert_eq!(
        got.cand, want.cand,
        "{name}: candidate compositions differ from f32"
    );
}

/// Tiny generation config: K=4 trajectories, F=2 formulas, T=22 (the derived
/// cap `2 + A + R_max` for A=16, R_max=4).
fn tiny_generation() -> GenerationConfig {
    GenerationConfig {
        schema_version: SCHEMA_VERSION,
        trajectories: 4,
        formulas: 2,
        seed: 99,
        temperature: 1.0,
        max_steps: 22,
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
        allocation: AllocationMode::RoundRobin,
        identity: IdentityMode::TraceOnly,
        identity_work_max: 4096,
        returned: 0,
        evidence: false,
        ion_request_work_max: 268435456,
    }
}

/// One warmed `generate` in element type `E`: a warmup call (autotune, bucket
/// allocation) then the measured call. Same seed and request as every dtype.
///
/// Returns the first error (upload, init or generate) instead of panicking,
/// so a refused dtype surfaces as `Err` and no test panics inside the
/// library.
fn try_warmed_generate<E: FloatElem>() -> Result<mamba3::models::ms2::contract::CandidateBatch> {
    let device = dev();
    let (parents, _) = train_molecules();
    let batch = make_batch(&parents, &[101, 102], 31);
    let table = train_table(&parents);
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device)?;
    let mut cfg = tiny_config::<E>();
    cfg.formula_table.rows = uploaded.rows as u32;
    cfg.formula_table.sha256 = uploaded.sha256.clone();
    let mut rng = Rng::seeded(11);
    let model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng)?;
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::<R, E>::new();
    let gcfg = tiny_generation();
    model.generate(&batch, &uploaded, &gcfg, &mut ws, &constants)?;
    check_launches(&device).unwrap();
    let out = model.generate(&batch, &uploaded, &gcfg, &mut ws, &constants)?;
    check_launches(&device).unwrap();
    Ok(out)
}

/// [`try_warmed_generate`] for dtypes whose pipeline succeeds.
fn warmed_generate<E: FloatElem>() -> mamba3::models::ms2::contract::CandidateBatch {
    try_warmed_generate::<E>().unwrap()
}

#[test]
fn exact_mass_integers_identical_across_dtypes() {
    // Point 2: integer kernel outputs are identical across the validated
    // dtypes. bf16 runs only where it is validated (the CPU runtime); f16
    // never runs.
    let reference = integer_snapshots::<f32>();
    if dev().name() == "cpu" {
        assert_snapshots_equal(&reference, &integer_snapshots::<half::bf16>(), "bf16");
    } else {
        println!("bf16 not validated here: kernel comparison skipped");
    }
}

#[test]
fn generation_bf16_valid_and_integers_match_f32() {
    // Point 4 (plus the end-to-end half of point 2) for bf16: on the CPU
    // runtime one warmed `generate` validates and the exact-integer
    // request/search fields match f32; on wgpu the dtype is refused before
    // any upload, allocation or launch. Sampled tokens and neural ranking
    // (`formula_row`, `actions`, `formula_counts`, `formula_rank`, float
    // fields) may differ across dtypes and are not compared.
    if dev().name() != "cpu" {
        let err = try_warmed_generate::<half::bf16>()
            .expect_err("bf16 generate off the CPU runtime must be refused");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "bf16 refusal is Error::Unsupported: {err:?}"
        );
        assert!(
            err.to_string().contains("bf16"),
            "the refusal names the dtype: {err}"
        );
        println!("bf16 refusal off CPU pinned: {err}");
        return;
    }
    let reference = warmed_generate::<f32>();
    reference.validate().expect("f32 candidates validate");
    assert!(
        reference.trace_log_prob.iter().all(|v| v.is_finite()),
        "f32 trace log-probs finite"
    );
    let out = warmed_generate::<half::bf16>();
    out.validate().expect("bf16 candidates validate");
    for (i, v) in out.trace_log_prob.iter().enumerate() {
        assert!(v.is_finite(), "bf16 record {i} trace log-prob {v}");
    }
    for (i, v) in out.formula_log_prob.iter().enumerate() {
        assert!(v.is_finite(), "bf16 record {i} formula log-prob {v}");
    }
    assert_eq!(
        out.request_status, reference.request_status,
        "bf16: request_status"
    );
    assert_eq!(out.rows_visited, reference.rows_visited, "bf16: rows_visited");
    assert_eq!(out.rows_joined, reference.rows_joined, "bf16: rows_joined");
    assert_eq!(out.rows_scored, reference.rows_scored, "bf16: rows_scored");
    assert_eq!(
        out.formula_support_complete, reference.formula_support_complete,
        "bf16: formula_support_complete"
    );
    assert_eq!(out.peaks_kept, reference.peaks_kept, "bf16: peaks_kept");
    println!(
        "bf16: validates; {} records, {} finished",
        out.status.len(),
        out.status
            .iter()
            .filter(|s| *s & mamba3::models::ms2::contract::candidate_status::FINISHED != 0)
            .count(),
    );
}

/// f16 is refused on every backend before any upload, allocation or launch:
/// the table upload, the model init and the `generate` entry points all
/// refuse with `Error::Unsupported` naming the dtype, the backend and the
/// reason. No test may panic inside the library, so the refusal is asserted
/// as `Err`, never observed as a panic or a NaN.
#[test]
fn generation_f16_refused_everywhere() {
    let backend = dev().name();
    let err = try_warmed_generate::<half::f16>()
        .expect_err("f16 must be refused on every backend");
    assert!(
        matches!(err, Error::Unsupported(_)),
        "f16 refusal is Error::Unsupported: {err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains("f16"), "the refusal names the dtype: {msg}");
    assert!(
        msg.contains("not validated"),
        "the refusal states f16 is not validated: {msg}"
    );
    println!("f16 refusal on {backend} pinned: {msg}");
    // The model constructor refuses on its own, before any parameter init.
    let device = dev();
    let cfg = tiny_config::<half::f16>();
    let mut rng = Rng::seeded(11);
    let err = match Ms2Model::<R, half::f16>::init(&cfg, &device, &mut rng) {
        Ok(_) => panic!("Ms2Model::init must refuse f16"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(
        err.to_string().contains("f16"),
        "the init refusal names the dtype: {err}"
    );
    // The table upload refuses on its own, before any device write.
    let (parents, _) = train_molecules();
    let table = train_table(&parents);
    let err = match DeviceFormulaTable::<R, half::f16>::upload(&table, &device) {
        Ok(_) => panic!("DeviceFormulaTable::upload must refuse f16"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}

#[test]
fn training_f32_loss_finite_and_falls() {
    // Point 3, f32 reference: finite every step, final below initial.
    let curve = train_curve::<f32>(TRAIN_STEPS);
    let f32_final = *curve.losses.last().unwrap();
    summarize("f32", &curve, f32_final);
    assert!(
        curve.first_nonfinite.is_none(),
        "f32 non-finite at step {:?} ({:?})",
        curve.first_nonfinite,
        curve.first_nonfinite_where,
    );
    assert!(
        f32_final < curve.losses[0],
        "f32 final {} not below initial {}",
        f32_final,
        curve.losses[0],
    );
}

#[test]
fn training_bf16_loss_finite_and_falls() {
    // Point 3, bf16: same assertions as f32, plus the ratio to f32 — on the
    // CPU runtime only. Elsewhere the dtype is refused before any work.
    if dev().name() != "cpu" {
        let device = dev();
        let mut cfg = tiny_config::<f32>();
        cfg.dtype = DType::BF16;
        let err = Ms2Capabilities::probe(&device)
            .check(&cfg)
            .expect_err("bf16 off CPU must be refused");
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        println!("bf16 training refusal off CPU pinned: {err}");
        return;
    }
    // Point 3, bf16: same assertions as f32, plus the ratio to f32.
    let f32_final = *train_curve::<f32>(TRAIN_STEPS).losses.last().unwrap();
    let curve = train_curve::<half::bf16>(TRAIN_STEPS);
    summarize("bf16", &curve, f32_final);
    assert!(
        curve.first_nonfinite.is_none(),
        "bf16 non-finite at step {:?} ({:?})",
        curve.first_nonfinite,
        curve.first_nonfinite_where,
    );
    let final_loss = *curve.losses.last().unwrap();
    assert!(
        final_loss < curve.losses[0],
        "bf16 final {} not below initial {}",
        final_loss,
        curve.losses[0],
    );
}

#[test]
fn dtype_allowlist_is_backend_based_not_capability_based() {
    // Finding R1-D2: f32 on every backend; bf16 on the CPU backend only;
    // f16 nowhere — independent of hardware capability, through the one
    // shared `Ms2Capabilities::check_dtype` function used by production,
    // tests and `examples/ms2_dtype_report.rs`.
    use mamba3::models::ms2::workspace::Ms2Capabilities;
    for backend in ["cpu", "cuda", "wgpu", "test-backend"] {
        Ms2Capabilities::check_dtype(backend, DType::F32)
            .unwrap_or_else(|e| panic!("f32 refused on {backend}: {e}"));
        let err = Ms2Capabilities::check_dtype(backend, DType::F16)
            .expect_err("f16 must be refused on every backend");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "f16 refusal is Unsupported on {backend}: {err:?}"
        );
        assert!(
            err.to_string().contains("not validated"),
            "f16 refusal states not validated on {backend}: {err}"
        );
    }
    Ms2Capabilities::check_dtype("cpu", DType::BF16)
        .expect("bf16 validates on the CPU backend");
    for backend in ["cuda", "wgpu", "test-backend"] {
        let err = Ms2Capabilities::check_dtype(backend, DType::BF16)
            .expect_err("bf16 off CPU must be refused");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "bf16 refusal is Unsupported on {backend}: {err:?}"
        );
        assert!(
            err.to_string().contains("not validated for this backend"),
            "bf16 refusal states not validated for this backend on {backend}: {err}"
        );
    }
}
