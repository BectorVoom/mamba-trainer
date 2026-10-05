//! K9 (plan item P7.7): reduced-precision capability matrix report.
//!
//! Validated set (contracts §3.3), enforced by the one shared policy
//! [`Ms2Capabilities::check_dtype`](mamba3::models::ms2::workspace::Ms2Capabilities::check_dtype):
//! f32 on every backend, bf16 on the CPU backend only, f16 nowhere. For each
//! validated dtype: the final loss of a fixed tiny training run, max
//! |gradient|, whether any non-finite value appeared and where, and the
//! warmed-generation validity rate. Refused dtypes record the refusal text
//! and are never built or run. The report distinguishes `hardware_supported`
//! (what the device reports) from `ms2_validated` (what the MS2 policy
//! allows) and carries a `policy` note; an earlier f16 measurement, taken
//! before the refusal policy, is kept under the explicitly named
//! `historical` key with its reason.
//! Aggregates only.
//!
//! Usage: `cargo run --release --no-default-features --features cpu
//! --example ms2_dtype_report -- --out <report.json>`.

use std::path::PathBuf;

use mamba3::autograd::Var;
use mamba3::backend::{DType, Device, FloatElem, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{Composition, composition_mass, element_index};
use mamba3::models::ms2::contract::{
    AllocationMode, Control, FormulaSource, GenerationConfig, GenerationMode, IdentityMode,
    ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch, candidate_status,
};
use mamba3::models::ms2::decoder::{Ms2Decoder, ReplayView, graph_loss};
use mamba3::models::ms2::encoder::Ms2Encoder;
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::formula_head::{DeviceFormulaTable, FormulaHead, gold_slots_host};
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::grammar::{Limits, Token};
use mamba3::models::ms2::targets::{Labels, Target};
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::models::ms2::workspace::Ms2Capabilities;
use mamba3::nn::Module;
use mamba3::nn::param::Param;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, FormulaBuffers, Ms2Constants, PeakBuffers, ReplayBuffers};
use mamba3::tensor::ops::random::Rng;
use mamba3::train::optim::{AdamWConfig, Optimizer};

type R = Auto;

/// Fixed tiny-run length: the same loop as `tests/ms2_dtype.rs`, fewer steps.
const STEPS: usize = 30;

/// Net hydrogen shift of contract §4.3, `m_H - m_e`.
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

fn trace_of(t: &serde_json::Value) -> Vec<Token> {
    t.as_array()
        .expect("trace array")
        .iter()
        .map(|a| {
            let a = a.as_array().expect("token array");
            Token {
                kind: a[0].as_u64().expect("kind") as u8,
                atom_type: a[1].as_u64().expect("atom_type") as u8,
                bond: a[2].as_u64().expect("bond") as u8,
                pointer: a[3].as_u64().expect("pointer") as u8,
            }
        })
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

fn precursor_of(comp: &Composition) -> u32 {
    composition_mass(comp).unwrap() + H_NET
}

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
        .map(|m| {
            let trace = trace_of(&m["whole_trace"]["trace"]);
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
        })
        .collect();
    (parents, labs)
}

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

fn train_table(parents: &[Composition]) -> FormulaTable {
    let mut comps = parents.to_vec();
    comps.push([6, 6, 0, 0, 0, 0, 0, 0, 0, 0]);
    comps.push([2, 6, 0, 1, 0, 0, 0, 0, 0, 0]);
    FormulaTable::from_compositions(comps).unwrap()
}

/// Per-dtype outcome of the fixed tiny run.
struct DtypeOutcome {
    /// What the device reports for this dtype.
    hardware_supported: bool,
    /// What the MS2 policy allows ([`Ms2Capabilities::check_dtype`]).
    ms2_validated: bool,
    error: Option<String>,
    param_bytes: Option<u64>,
    initial_loss: Option<f32>,
    final_loss: Option<f32>,
    max_grad_abs: Option<f32>,
    first_nonfinite_step: Option<usize>,
    first_nonfinite_where: Option<String>,
    generation_validate_ok: Option<bool>,
    generation_validity_rate: Option<f32>,
    /// `generate` refusal text, when the pipeline fails (finding F-K9-1).
    generation_error: Option<String>,
}

fn run_dtype<E: FloatElem>() -> DtypeOutcome {
    let device = dev();
    let (parents, labs) = train_molecules();
    let b = parents.len();
    let spectra_batch = make_batch(&parents, &[101, 102], 31);
    let table = train_table(&parents);
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device)
        .expect("table uploads for a supported dtype");
    let model_cfg = tiny_config::<E>();
    let d = model_cfg.d_model as usize;
    let mut rng = Rng::seeded(41);
    let encoder = Ms2Encoder::init(&model_cfg, &device, &mut rng).expect("encoder inits");
    let formula_head =
        FormulaHead::<R, E>::init(&model_cfg, &device, &mut rng).expect("head inits");
    let decoder = Ms2Decoder::init(&model_cfg, &device, &mut rng).expect("decoder inits");
    let mut param_bytes = 0u64;
    for (_, p) in encoder
        .named_parameters()
        .iter()
        .chain(formula_head.named_parameters().iter())
        .chain(decoder.named_parameters().iter())
    {
        param_bytes += p.numel() as u64 * E::DTYPE.size() as u64;
    }
    let spectra = DeviceSpectra::upload(&spectra_batch, &device).expect("spectra upload");
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
    ms2::grammar_replay(&targets.tokens, &targets.meta, &constants, 16, 4, &replay_buffers)
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
    }
    let gold_slots = gold_slots_host(&table, &queries, &gold_rows, m_window);
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
    let mut initial_loss = None;
    let mut final_loss = 0.0f32;
    let mut max_grad_abs = 0.0f32;
    let mut first_nonfinite_step = None;
    let mut first_nonfinite_where = None;
    for step in 0..STEPS {
        let encoded = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        ms2::formula_window(&search_t, &spectra.meta, table.max_error(), u32::MAX, 4096, &buffers)
            .unwrap();
        ms2::formula_gather(&buffers.window, &uploaded.table, &uploaded.counts, &mut buffers.cand)
            .unwrap();
        ms2::count_features(
            &buffers.cand.reshape(vec![b * m_window, 13]).unwrap(),
            &uploaded.log_table,
            &mut buffers.cand_feat.reshape(vec![b * m_window, 10]).unwrap(),
            13,
        )
        .unwrap();
        let scored = formula_head.score(&buffers, &encoded.pool).unwrap();
        let gold_t = IdTensor::from_slice(&gold_slots, vec![b], &device).unwrap();
        let formula_loss = formula_head.loss(&scored, &gold_t).unwrap();
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
        if step == 0 {
            initial_loss = Some(loss_v);
        }
        final_loss = loss_v;
        if !loss_v.is_finite() && first_nonfinite_step.is_none() {
            first_nonfinite_step = Some(step);
            first_nonfinite_where = Some("loss".to_string());
        }
        let grads = total.backward_retain().unwrap();
        for (name, param) in &params {
            let g = grads.get(param.id()).expect("gradient present").try_to_f32().unwrap();
            for v in &g {
                if !v.is_finite() && first_nonfinite_step.is_none() {
                    first_nonfinite_step = Some(step);
                    first_nonfinite_where = Some(format!("grad:{name}"));
                }
                max_grad_abs = max_grad_abs.max(v.abs());
            }
        }
        opt.step(&only_values, &grads).unwrap();
    }
    check_launches(&device).unwrap();
    // One warmed generate: warmup, then the measured call.
    let gen_cfg = GenerationConfig {
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
        formula_evidence_work_max: 2048,
        formula_evidence_dispatch_max: 268435456,
    };
    let mut gen_model_cfg = tiny_config::<E>();
    gen_model_cfg.formula_table.rows = uploaded.rows as u32;
    gen_model_cfg.formula_table.sha256 = uploaded.sha256.clone();
    let mut rng = Rng::seeded(11);
    let gen_model = Ms2Model::<R, E>::init(&gen_model_cfg, &device, &mut rng).expect("model inits");
    let mut ws = GenerationWorkspace::<R, E>::new();
    // A dtype whose numerics fail the pipeline's own validation (finding
    // F-K9-1 for f16) records the refusal instead of panicking: `generate`
    // validates its batch before returning.
    let gen_result: Result<(bool, f32), String> = (|| {
        gen_model
            .generate(&spectra_batch, &uploaded, &gen_cfg, &mut ws, &constants)
            .map_err(|e| format!("warmup: {e}"))?;
        let out = gen_model
            .generate(&spectra_batch, &uploaded, &gen_cfg, &mut ws, &constants)
            .map_err(|e| format!("measured: {e}"))?;
        check_launches(&device).map_err(|e| e.to_string())?;
        let validate_ok = out.validate().is_ok();
        let total_records = out.status.len();
        // Valid: finished with no failure bits and finite trace log-probability.
        let valid = out
            .status
            .iter()
            .zip(out.trace_log_prob.iter())
            .filter(|(s, lp)| {
                *s & candidate_status::FINISHED != 0
                    && *s
                        & (candidate_status::INVALID_FINAL
                            | candidate_status::REQUEST_FAILED
                            | candidate_status::NO_VALID_ACTION)
                        == 0
                    && lp.is_finite()
            })
            .count();
        Ok::<(bool, f32), String>((
            validate_ok,
            valid as f32 / total_records.max(1) as f32,
        ))
    })();
    let (generation_validate_ok, generation_validity_rate, generation_error) = match gen_result {
        Ok((ok, rate)) => (Some(ok), Some(rate), None),
        Err(text) => (Some(false), None, Some(text)),
    };
    DtypeOutcome {
        hardware_supported: true,
        ms2_validated: true,
        error: None,
        param_bytes: Some(param_bytes),
        initial_loss,
        final_loss: Some(final_loss),
        max_grad_abs: Some(max_grad_abs),
        first_nonfinite_step,
        first_nonfinite_where,
        generation_validate_ok,
        generation_validity_rate,
        generation_error,
    }
}

fn outcome_json(o: &DtypeOutcome, f32_final: Option<f32>) -> serde_json::Value {
    serde_json::json!({
        "hardware_supported": o.hardware_supported,
        "ms2_validated": o.ms2_validated,
        "error": o.error,
        "param_bytes": o.param_bytes,
        "initial_loss": o.initial_loss,
        "final_loss": o.final_loss,
        "final_to_f32_ratio": match (o.final_loss, f32_final) {
            (Some(got), Some(f32v)) => Some(got / f32v),
            _ => None,
        },
        "max_grad_abs": o.max_grad_abs,
        "first_nonfinite_step": o.first_nonfinite_step,
        "first_nonfinite_where": o.first_nonfinite_where,
        "generation_validate_ok": o.generation_validate_ok,
        "generation_validity_rate": o.generation_validity_rate,
        "generation_error": o.generation_error,
    })
}

fn usage() -> ! {
    eprintln!("usage: ms2_dtype_report --out <report.json>");
    std::process::exit(2);
}

fn main() {
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = args.next().map(PathBuf::from),
            _ => usage(),
        }
    }
    let Some(out) = out else { usage() };

    let device = dev();
    let caps = Ms2Capabilities::probe(&device);
    println!(
        "backend {}: f32={} f16={} bf16={}",
        caps.backend, caps.f32_supported, caps.f16_supported, caps.bf16_supported,
    );

    // Gate every dtype on the one shared policy (contracts §3.3): f32 on
    // every backend, bf16 on the CPU backend only, f16 nowhere. Validated
    // dtypes run; refused ones record the refusal text and are never built
    // or run. No silent fallback, no kernel launch for a refused dtype.
    let f32_outcome = run_dtype::<f32>();
    let f32_final = f32_outcome.final_loss;
    let mut dtypes = serde_json::Map::new();
    dtypes.insert("f32".to_string(), outcome_json(&f32_outcome, f32_final));

    let bf16_hw = caps.bf16_supported;
    let bf16_ok = Ms2Capabilities::check_dtype(&caps.backend, DType::BF16).is_ok();
    if bf16_ok {
        let o = run_dtype::<half::bf16>();
        println!(
            "bf16: loss {:?} -> {:?}, max|grad| {:?}, first non-finite {:?} ({:?}), \
             generation validate {:?}, validity rate {:?}, generation error {:?}",
            o.initial_loss,
            o.final_loss,
            o.max_grad_abs,
            o.first_nonfinite_step,
            o.first_nonfinite_where,
            o.generation_validate_ok,
            o.generation_validity_rate,
            o.generation_error,
        );
        dtypes.insert("bf16".to_string(), outcome_json(&o, f32_final));
    } else {
        let mut cfg = tiny_config::<f32>();
        cfg.dtype = DType::BF16;
        let text = caps
            .check(&cfg)
            .expect_err("bf16 off the CPU runtime must be refused")
            .to_string();
        assert!(
            text.contains("not validated for this backend"),
            "the bf16 refusal states it is not validated for this backend: {text}"
        );
        println!("bf16 refused: {text}");
        dtypes.insert(
            "bf16".to_string(),
            outcome_json(
                &DtypeOutcome {
                    hardware_supported: bf16_hw,
                    ms2_validated: false,
                    error: Some(text),
                    param_bytes: None,
                    initial_loss: None,
                    final_loss: None,
                    max_grad_abs: None,
                    first_nonfinite_step: None,
                    first_nonfinite_where: None,
                    generation_validate_ok: None,
                    generation_validity_rate: None,
                    generation_error: None,
                },
                f32_final,
            ),
        );
    }

    // f16 is not validated for the MS2 model on any backend: record the
    // refusal text without building or running anything.
    {
        let mut cfg = tiny_config::<f32>();
        cfg.dtype = DType::F16;
        let text = caps
            .check(&cfg)
            .expect_err("f16 must be refused on every backend")
            .to_string();
        assert!(
            text.contains("not validated"),
            "the f16 refusal states it is not validated: {text}"
        );
        println!("f16 refused: {text}");
        dtypes.insert(
            "f16".to_string(),
            outcome_json(
                &DtypeOutcome {
                    hardware_supported: caps.f16_supported,
                    ms2_validated: false,
                    error: Some(text),
                    param_bytes: None,
                    initial_loss: None,
                    final_loss: None,
                    max_grad_abs: None,
                    first_nonfinite_step: None,
                    first_nonfinite_where: None,
                    generation_validate_ok: None,
                    generation_validity_rate: None,
                    generation_error: None,
                },
                f32_final,
            ),
        );
    }

    let report = serde_json::json!({
        "backend": caps.backend,
        "steps": STEPS,
        "policy": "contracts §3.3 via Ms2Capabilities::check_dtype (the one function used by production, tests and this report): f32 on every backend; bf16 on the CPU backend only; f16 nowhere. hardware_supported is what the device reports; ms2_validated is what the policy allows; refused combinations carry the refusal text instead of measurements.",
        "capabilities": {
            "f32_supported": caps.f32_supported,
            "f16_supported": caps.f16_supported,
            "bf16_supported": caps.bf16_supported,
            "max_bindings": caps.max_bindings,
            "plane_size_max": caps.plane_size_max,
        },
        "dtypes": dtypes,
        "historical": {
            "reason": "measurements taken before the refusal policy, when f16 was still executed instead of refused; kept for the record, superseded by the current dtypes.f16 refusal.",
            "f16_pre_policy_cpu": {
                "error": null,
                "final_loss": null,
                "final_to_f32_ratio": null,
                "first_nonfinite_step": 0,
                "first_nonfinite_where": "loss",
                "generation_error": "warmup: invalid configuration: CandidateBatch::validate: spectrum 0 formula_mass_retained NaN is outside the retained-fraction range [0, 1 + 1e-4]",
                "generation_validate_ok": false,
                "generation_validity_rate": null,
                "initial_loss": null,
                "max_grad_abs": 0.0,
                "param_bytes": 24802
            }
        },
    });
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap_or_else(|e| {
        eprintln!("ms2_dtype_report: cannot write {}: {e}", out.display());
        std::process::exit(1);
    });
    println!("wrote {}", out.display());
}
