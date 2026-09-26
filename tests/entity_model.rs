//! G3 tests (ENTITY_MODEL_PLAN.md): the StepCausal leakage test and the
//! `crew_symmetric` coordination test. (The `Joint` + `autoregressive_on`
//! refusal is a spec unit test in `tests/entity_spec.rs`.)

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::model::EntityModel;
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityModelSpec, HeadSpec, QuerySetSpec, SetLayout,
};
use mamba3::autograd::Var;
use mamba3::tensor::Tensor;
use mamba3::train::TrainStep;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn frand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32) / (u64::MAX >> 11) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn ar_spec(crew_symmetric: bool) -> EntityModelSpec {
    EntityModelSpec {
        globals: 4,
        context: vec![ContextSetSpec::new("cells", 9, 4).with_layout(SetLayout::Grid {
            height: 3,
            width: 3,
            alternate_axes: true,
        })],
        queries: Some(
            QuerySetSpec::new("agents", 3, 3, 3)
                .with_anchor("cells")
                .with_autoregressive("tgt"),
        ),
        heads: vec![
            HeadSpec::pointer("tgt", "cells", 1),
            HeadSpec::categorical("kind", 2).condition_on("tgt"),
        ],
        d_model: 8,
        context_layers: 1,
        decoder_layers: 1,
        decoder: DecoderMode::StepCausal { crew_symmetric },
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 8,
            n_heads: 2,
            head_dim: 4,
            d_state: 4,
            n_groups: 2,
            chunk_size: 4,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 1,
    }
}

/// Run encode → queries → decode with teacher-forced `choices` (`[M*K]` host
/// ids, `IGNORE` allowed), returning `h` as `[M, K, d]` host floats.
fn decode_h(
    model: &EntityModel<R, f32>,
    device: &Device<R>,
    qfeats: &[f32],
    choices: &[u32],
) -> Vec<f32> {
    let (b, m, k, n) = (1usize, 3, 3, 9);
    let cells = Var::constant(
        Tensor::<R, f32>::from_f32(&frand(n * 4, 101), vec![b, n, 4], device).unwrap(),
    );
    let presence = Var::constant(
        Tensor::<R, f32>::from_f32(&vec![1.0; b * n], vec![b, n], device).unwrap(),
    );
    let glob = Var::constant(
        Tensor::<R, f32>::from_f32(&frand(b * 4, 102), vec![b, 4], device).unwrap(),
    );
    let ctx = model.encode(std::slice::from_ref(&cells), std::slice::from_ref(&presence), Some(&glob)).unwrap();
    let qf = Var::constant(
        Tensor::<R, f32>::from_f32(qfeats, vec![b, m, 3], device).unwrap(),
    );
    let qp = Var::constant(
        Tensor::<R, f32>::from_f32(&vec![1.0; b * m], vec![b, m], device).unwrap(),
    );
    let u = model.query_base(&qf, &qp).unwrap();
    let g = model.global_embed(Some(&glob)).unwrap();
    let anchors = vec![0u32; b * m];
    let anchor_tok = model.anchor_tokens(&anchors, b, m, &ctx).unwrap();
    let prev = model.prev_tokens(choices, b, m, k, &ctx).unwrap();
    let q = model.build_queries(&u, Some(&anchor_tok), g.as_ref(), &prev).unwrap();
    model.decode(&ctx, &q).unwrap().h.tensor().to_f32()
}

fn step_of(h: &[f32], _m: usize, k: usize, d: usize, mi: usize, j: usize) -> &[f32] {
    let start = (mi * k + j) * d;
    &h[start..start + d]
}

#[test]
fn step_causal_leakage() {
    for crew in [false, true] {
        let device = dev();
        let spec = ar_spec(crew);
        spec.validate().unwrap();
        let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
        let (m, k, d) = (3usize, 3, 8);
        let qfeats = frand(m * 3, 201);
        // Step-1 labels change to other valid ids; steps 0..=1 must be
        // exactly unchanged, step 2 must change.
        let mut c1 = vec![0u32; m * k];
        let mut c2 = vec![0u32; m * k];
        for i in 0..m * k {
            c1[i] = (i % 10) as u32;
            c2[i] = (i % 10) as u32;
        }
        for mi in 0..m {
            c2[mi * k + 1] = ((mi * k + 1) % 10 + 5) as u32 % 10;
        }
        assert_ne!(c1, c2);
        let h1 = decode_h(&model, &device, &qfeats, &c1);
        let h2 = decode_h(&model, &device, &qfeats, &c2);
        for mi in 0..m {
            for j in 0..=1 {
                assert_eq!(
                    step_of(&h1, m, k, d, mi, j),
                    step_of(&h2, m, k, d, mi, j),
                    "crew={crew}: step {j} of query {mi} moved when step-1 labels changed"
                );
            }
        }
        let mut moved = false;
        for mi in 0..m {
            if step_of(&h1, m, k, d, mi, 2) != step_of(&h2, m, k, d, mi, 2) {
                moved = true;
            }
        }
        assert!(moved, "crew={crew}: step 2 did not react to step-1 labels");
    }
}

#[test]
fn crew_symmetric_reaches_both_directions() {
    let device = dev();
    let spec = ar_spec(true);
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let (m, k) = (3usize, 3);
    let mut qfeats = frand(m * 3, 301);
    let choices = vec![2u32; m * k];
    let h1 = decode_h(&model, &device, &qfeats, &choices);
    // Change only query 1's features.
    for v in &mut qfeats[1 * 3..2 * 3] {
        *v += 2.0;
    }
    let h2 = decode_h(&model, &device, &qfeats, &choices);
    let d = 8;
    // m1 = 0 < m2 = 1 and m1 = 2 > m2 = 1 both move at step 0.
    assert_ne!(
        step_of(&h1, m, k, d, 0, 0),
        step_of(&h2, m, k, d, 0, 0),
        "query 0 (before query 1) did not see its step-0 change"
    );
    assert_ne!(
        step_of(&h1, m, k, d, 2, 0),
        step_of(&h2, m, k, d, 2, 0),
        "query 2 (after query 1) did not see its step-0 change"
    );
}

use mamba3::models::entity::model::Decode;
use mamba3::models::entity::{EntityBatch, HostArrays};

fn ar_batch() -> HostArrays {
    let mut a = HostArrays::new();
    a.insert_f32("cells", vec![1, 9, 4], frand(1 * 9 * 4, 401));
    a.insert_f32("globals", vec![1, 4], frand(4, 402));
    a.insert_f32("agents", vec![1, 3, 3], frand(1 * 3 * 3, 403));
    a.insert_int("agents.anchor", vec![1, 3], vec![0, 4, 8]);
    // Random valid labels: tgt 0..10 (9 cells + NONE), kind 0/1.
    let mut s = 404u64;
    let mut ri = |m: u64| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m) as i64
    };
    let tgt: Vec<i64> = (0..9).map(|_| ri(10)).collect();
    let kind: Vec<i64> = (0..9).map(|_| ri(2)).collect();
    a.insert_int("label.tgt", vec![1, 3, 3], tgt);
    a.insert_int("label.kind", vec![1, 3, 3], kind);
    a
}

#[test]
fn generate_equals_teacher_forced_on_own_choices() {
    let device = dev();
    let spec = ar_spec(false);
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &ar_batch(), &device).unwrap();
    let greedy = model.predict(&batch, Decode::Greedy, None).unwrap();
    // Feed the greedy choices back as labels.
    let mut a = ar_batch();
    let own: Vec<i64> = greedy.choices["tgt"].to_vec().into_iter().map(|v| v as i64).collect();
    a.insert_int("label.tgt", vec![1, 3, 3], own);
    let batch2 = EntityBatch::<R, f32>::from_host(&spec, &a, &device).unwrap();
    let forced = model.predict(&batch2, Decode::TeacherForced, None).unwrap();
    for (name, g) in &greedy.logits {
        assert_eq!(g.to_f32(), forced.logits[name].to_f32(), "head {name}");
    }
    assert_eq!(
        greedy.choices["tgt"].to_vec(),
        forced.choices["tgt"].to_vec()
    );
}

#[test]
fn save_load_predict_round_trip() {
    let device = dev();
    let spec = ar_spec(true);
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &ar_batch(), &device).unwrap();
    let before = model.predict(&batch, Decode::Greedy, None).unwrap();
    let path = std::env::temp_dir().join("mamba3_entity_roundtrip.m3ck");
    model.save(&path, 7).unwrap();
    let loaded = EntityModel::<R, f32>::load(&path, &device).unwrap();
    assert_eq!(loaded.spec(), &spec);
    let after = loaded.predict(&batch, Decode::Greedy, None).unwrap();
    for (name, b) in &before.logits {
        assert_eq!(b.to_f32(), after.logits[name].to_f32(), "head {name}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn chooser_is_honoured_and_conditions_next_steps() {
    let device = dev();
    let spec = ar_spec(true);
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &ar_batch(), &device).unwrap();
    let free = model.predict(&batch, Decode::Greedy, None).unwrap();
    // A chooser that forces entity 1 everywhere: honoured at every step,
    // and its choices condition the next step (the plan is autoregressive).
    let mut calls = 0;
    let mut force_one = |_step: usize, _logits: &mamba3::tensor::Tensor<R, f32>| {
        calls += 1;
        mamba3::tensor::ops::index::IdTensor::from_slice(&vec![1u32; 3], vec![1, 3], &dev())
    };
    let constrained = model.predict(&batch, Decode::Greedy, Some(&mut force_one)).unwrap();
    assert_eq!(calls, 3);
    let got = constrained.choices["tgt"].to_vec();
    assert!(got.iter().all(|&v| v == 1), "chooser violated: {got:?}");
    // The constrained step-0 choice feeds step 1: later steps must react.
    assert_ne!(
        free.choices["tgt"].to_vec(),
        got,
        "constrained choices identical to free ones"
    );
    // And the conditioned head moves with the constrained choices.
    assert_ne!(
        free.logits["kind"].to_f32(),
        constrained.logits["kind"].to_f32()
    );
}

// ---------------------------------------------------------------------------
// G6 ports of tests/planner.rs (same assertions, generic model, Kaggriculture
// shapes in Joint mode).
// ---------------------------------------------------------------------------

use mamba3::models::entity::EntityTask;
use mamba3::nn::Module;

/// Small Kaggriculture-shaped spec in Joint mode (no autoregression).
fn joint_kag_spec() -> mamba3::models::entity::EntityModelSpec {
    use mamba3::models::entity::{ContextSetSpec, DecoderMode, EntityModelSpec, HeadSpec, QuerySetSpec, SetLayout};
    EntityModelSpec {
        globals: 6,
        context: vec![ContextSetSpec::new("tiles", 4, 5).with_layout(SetLayout::Grid {
            height: 2,
            width: 2,
            alternate_axes: true,
        })],
        queries: Some(QuerySetSpec::new("units", 2, 4, 2).with_anchor("tiles")),
        heads: vec![
            HeadSpec::pointer("target", "tiles", 1),
            HeadSpec::categorical("op", 3).condition_on("target"),
            HeadSpec::multilabel("opset", 3).condition_on("target").loss_weight(0.3),
            HeadSpec::categorical("crop", 2).condition_on("target").loss_weight(0.3),
            HeadSpec::regression("eta", 1).first_step_only().loss_weight(0.1),
        ],
        d_model: 8,
        context_layers: 1,
        decoder_layers: 1,
        decoder: DecoderMode::Joint,
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 8,
            n_heads: 2,
            head_dim: 4,
            d_state: 4,
            n_groups: 2,
            chunk_size: 4,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 5,
    }
}

/// Random host arrays with valid labels for the Joint Kaggriculture spec.
fn random_joint_arrays(b: usize, active: usize, seed: u64) -> HostArrays {
    let (n, u, k, q) = (4usize, 2, 2, 4);
    let mut anchor = vec![-1i64; b * u];
    let mut tgt = vec![-1i64; b * q];
    let mut op = vec![-1i64; b * q];
    let mut crop = vec![-1i64; b * q];
    let mut opset = vec![0.0f32; b * q * 3];
    let mut eta = vec![f32::NAN; b * u];
    let mut s = seed.max(1);
    let mut ri = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    for bi in 0..b {
        for uu in 0..active.min(u) {
            anchor[bi * u + uu] = ri(n) as i64;
            eta[bi * u + uu] = (ri(6) as f32 + 1.0).ln();
            for j in 0..k {
                let f = bi * q + uu * k + j;
                let t = if ri(5) < 4 { ri(n) as i64 } else { n as i64 };
                tgt[f] = t;
                if t < n as i64 {
                    op[f] = ri(3) as i64;
                    if ri(2) == 0 {
                        crop[f] = ri(2) as i64;
                    }
                    opset[f * 3 + ri(3)] = 1.0;
                }
            }
        }
    }
    let mut a = HostArrays::new();
    // Guarantee one IGNORE row (real batches always have padding): the last
    // row's pointer label is IGNORE, exercising the none_prev token.
    tgt[q - 1] = -1;
    op[q - 1] = -1;
    a.insert_f32("tiles", vec![b, n, 5], frand(b * n * 5, seed + 1));
    a.insert_f32("globals", vec![b, 6], frand(b * 6, seed + 2));
    a.insert_f32("units", vec![b, u, 4], frand(b * u * 4, seed + 3));
    a.insert_int("units.anchor", vec![b, u], anchor);
    a.insert_int("label.target", vec![b, u, k], tgt);
    a.insert_int("label.op", vec![b, u, k], op);
    a.insert_f32("label.opset", vec![b, u, k, 3], opset);
    a.insert_int("label.crop", vec![b, u, k], crop);
    a.insert_f32("label.eta", vec![b, u, 1], eta);
    a
}

#[test]
fn forward_shapes() {
    let device = dev();
    let spec = joint_kag_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &random_joint_arrays(2, 2, 7), &device).unwrap();
    let (_, out) = model.forward_train(&batch).unwrap();
    assert_eq!(out.logits["target"].shape().dims(), &[2, 2, 2, 5]);
    assert_eq!(out.logits["op"].shape().dims(), &[2, 2, 2, 3]);
    assert_eq!(out.logits["opset"].shape().dims(), &[2, 2, 2, 3]);
    assert_eq!(out.logits["crop"].shape().dims(), &[2, 2, 2, 2]);
    assert_eq!(out.logits["eta"].shape().dims(), &[2, 2, 1, 1]);
    for (name, v) in &out.logits {
        assert!(v.tensor().to_f32().iter().all(|x| x.is_finite()), "head {name}");
    }
}

#[test]
fn gradient_reaches_every_parameter() {
    let device = dev();
    let spec = joint_kag_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &random_joint_arrays(1, 2, 11), &device).unwrap();
    let task = EntityTask::new(&model);
    let loss = task.loss(&batch).unwrap();
    let grads = loss.backward().unwrap();
    for (name, p) in model.named_parameters() {
        let g = grads.get(p.id()).unwrap_or_else(|| panic!("no gradient reached {name}"));
        let norm: f32 = g.to_f32().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "zero gradient at {name}");
    }
}

/// Central-difference gradient of the task loss wrt a few elements of a parameter.
fn check_param_grad(spec: &mamba3::models::entity::EntityModelSpec, name: &str, idxs: &[usize]) {
    let device = dev();
    let model = EntityModel::<R, f32>::init(spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(spec, &random_joint_arrays(1, 2, 13), &device).unwrap();
    let task = EntityTask::new(&model);
    let loss_of = || task.loss(&batch).unwrap().to_f32()[0];
    let params: std::collections::HashMap<String, _> = model.named_parameters().into_iter().collect();
    let p = params.get(name).unwrap_or_else(|| panic!("no param {name}"));
    let analytic = task.loss(&batch).unwrap().backward().unwrap().get(p.id()).unwrap().to_f32();
    let eps = 1e-3f32;
    for &i in idxs {
        let mut v = p.value().to_f32();
        v[i] += eps;
        p.set(Tensor::from_f32(&v, p.shape().clone(), &device).unwrap());
        let fp = loss_of();
        v[i] -= 2.0 * eps;
        p.set(Tensor::from_f32(&v, p.shape().clone(), &device).unwrap());
        let fm = loss_of();
        v[i] += eps;
        p.set(Tensor::from_f32(&v, p.shape().clone(), &device).unwrap());
        let numeric = (fp - fm) / (2.0 * eps);
        let tol = 2e-2 * (1.0 + numeric.abs());
        assert!((analytic[i] - numeric).abs() < tol, "{name}[{i}] analytic={} numeric={}", analytic[i], numeric);
    }
}

#[test]
fn loss_gradients_match_finite_differences() {
    let spec = joint_kag_spec();
    check_param_grad(&spec, "ctx_pos_0", &[0, 1, 5]);
    check_param_grad(&spec, "ptr_extra_0", &[0, 3]);
    check_param_grad(&spec, "ctx_in1_0.weight", &[0, 2]);
}

#[test]
fn anchor_gather_equals_one_hot_indexing() {
    use mamba3::tensor::ops::index::one_hot;
    let device = dev();
    let spec = joint_kag_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let (n, d) = (spec.n_ctx(), spec.d_model);
    let t = Tensor::<R, f32>::from_f32(&frand(n * d, 21), vec![1, n, d], &device).unwrap();
    let ctx = Var::constant(t.clone());
    // Global anchor ids (single context set, offset 0), then IGNORE.
    let anchors = vec![3u32, 2u32, mamba3::tensor::ops::entity_model::IGNORE];
    let got = model.anchor_tokens(&anchors, 1, 3, &ctx).unwrap().tensor().to_f32();
    let ids =
        mamba3::tensor::ops::index::IdTensor::from_slice(&anchors[..2], vec![1, 2], &device).unwrap();
    let oh = one_hot::<R, f32>(&ids, n).unwrap();
    let want = Var::constant(oh).matmul(&Var::constant(t)).unwrap().tensor().to_f32();
    for r in 0..2 {
        let row = &got[r * d..(r + 1) * d];
        let want_row = &want[r * d..(r + 1) * d];
        for (a, b) in row.iter().zip(want_row.iter()) {
            assert!((a - b).abs() < 1e-5, "row {r}: {a} != {b}");
        }
    }
    // IGNORE anchors gather the zero row.
    let zero = &got[2 * d..3 * d];
    assert!(zero.iter().all(|&v| v == 0.0));
}
