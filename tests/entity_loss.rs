//! G4 tests (ENTITY_MODEL_PLAN.md): every head type's loss against a host
//! f64 oracle, over-fitting, loss-scale invariance, and presence masking.

#![cfg(feature = "backend")]

use std::collections::BTreeMap;

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec,
};
use mamba3::nn::Module;
use mamba3::prelude::*;
use mamba3::train::{Optimizer, TrainStep};

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// 1 context set (4 entities, 3 feats), 2 queries, K = 2, one head of each
/// kind. No autoregression (Joint decoder).
fn four_head_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 2,
        context: vec![ContextSetSpec::new("cells", 4, 3)],
        queries: Some(QuerySetSpec::new("agents", 2, 2, 2)),
        heads: vec![
            HeadSpec::pointer("p", "cells", 1),
            HeadSpec::categorical("c", 3).condition_on("p"),
            HeadSpec::multilabel("ml", 2).loss_weight(0.5),
            HeadSpec::regression("r", 1)
                .first_step_only()
                .loss_weight(0.25),
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
        seed: 3,
    }
}

/// Hand-made batch: 1 sample, 2 queries, K = 2, with IGNORE / NaN rows.
fn handmade() -> HostArrays {
    let mut a = HostArrays::new();
    // cells: entity rows are one-hot-ish so pointer ids are meaningful.
    a.insert_f32(
        "cells",
        vec![1, 4, 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0],
    );
    a.insert_f32("globals", vec![1, 2], vec![0.3, -0.2]);
    a.insert_f32("agents", vec![1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);
    // label.p [1,2,2]: (q0: 1, 4=NONE-extra), (q1: IGNORE, 2).
    a.insert_int("label.p", vec![1, 2, 2], vec![1, 4, -1, 2]);
    // label.c: (q0: 2, 0), (q1: IGNORE, 1).
    a.insert_int("label.c", vec![1, 2, 2], vec![2, 0, -1, 1]);
    // label.ml [1,2,2,2]: NaN row at (q0, step 1).
    a.insert_f32(
        "label.ml",
        vec![1, 2, 2, 2],
        vec![1.0, 0.0, f32::NAN, f32::NAN, 0.0, 1.0, 1.0, 1.0],
    );
    // label.r [1,2,1] (First): q0 kept, q1 NaN.
    a.insert_f32("label.r", vec![1, 2, 1], vec![0.5, f32::NAN]);
    a
}

fn lse(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// Host f64 oracle over the device logits, mirroring the loss rules.
fn oracle(logits: &BTreeMap<String, Vec<f32>>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    // p: rows (q0s0: id 1), (q0s1: id 4), (q1s1: id 2); width 5.
    let p = &logits["p"];
    let mut acc = 0.0;
    for (row, id) in [(0usize, 1usize), (1, 4), (3, 2)] {
        let r: Vec<f64> = (0..5).map(|c| p[row * 5 + c] as f64).collect();
        acc += lse(&r) - r[id];
    }
    out.insert("p".to_string(), acc / 3.0);
    // c: rows (q0s0: 2), (q0s1: 0), (q1s1: 1); width 3.
    let c = &logits["c"];
    let mut acc = 0.0;
    for (row, id) in [(0usize, 2usize), (1, 0), (3, 1)] {
        let r: Vec<f64> = (0..3).map(|cc| c[row * 3 + cc] as f64).collect();
        acc += lse(&r) - r[id];
    }
    out.insert("c".to_string(), acc / 3.0);
    // ml: rows 0, 2, 3 kept (row 1 NaN); L = 2; BCE averaged over labels.
    let ml = &logits["ml"];
    let targets = [1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut acc = 0.0;
    for (ri, row) in [0usize, 2, 3].iter().enumerate() {
        for l in 0..2 {
            let x = ml[row * 2 + l] as f64;
            let y = targets[ri * 2 + l];
            acc += x.max(0.0) + (-x.abs()).exp().ln_1p() - x * y;
        }
    }
    out.insert("ml".to_string(), acc / (3.0 * 2.0));
    // r: only q0 step 0 kept, target 0.5.
    let r = &logits["r"];
    let d = r[0] as f64 - 0.5;
    out.insert("r".to_string(), d * d);
    out
}

#[test]
fn head_losses_match_host_oracle() {
    let device = dev();
    let spec = four_head_spec();
    spec.validate().unwrap();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &handmade(), &device).unwrap();
    let task = EntityTask::new(&model);
    let got = task.component_losses(&batch).unwrap();
    let (_, out) = model.forward_train(&batch).unwrap();
    let mut logits = BTreeMap::new();
    for (name, v) in &out.logits {
        logits.insert(name.clone(), v.tensor().to_f32());
    }
    let want = oracle(&logits);
    for (name, w) in &want {
        let g = got[name].to_f32()[0] as f64;
        assert!((g - w).abs() < 1e-4, "head {name}: device {g} != host {w}");
    }
}

#[test]
fn overfits_a_fixed_batch() {
    let device = dev();
    let spec = four_head_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &handmade(), &device).unwrap();
    let task = EntityTask::new(&model);
    let mut opt = AdamWConfig::builder()
        .learning_rate(3e-3)
        .weight_decay(0.0)
        .build()
        .init::<R, f32>();
    let params = task.parameters();
    let start = task.loss(&batch).unwrap().to_f32()[0];
    for _ in 0..200 {
        let loss = task.loss(&batch).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&params, &grads).unwrap();
    }
    let end = task.loss(&batch).unwrap().to_f32()[0];
    assert!(end < 0.1 * start, "loss {start} -> {end}, expected < 10%");
}

#[test]
fn loss_scale_leaves_updates_unchanged() {
    // Same shape as the planner's proof: one tight optimizer step with eps
    // and clip scaled, then the clipped case. One step proves the math; more
    // steps only compound 1-ulp chaos through the nonlinear dynamics.
    let run = |scale: f32, eps: f32, clip: f32| {
        let device = dev();
        let spec = four_head_spec();
        let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
        let batch = EntityBatch::<R, f32>::from_host(&spec, &handmade(), &device).unwrap();
        let task = EntityTask::new(&model).with_loss_scale(scale);
        let opt = AdamWConfig::builder()
            .learning_rate(1e-3)
            .eps(eps)
            .weight_decay(0.0)
            .build()
            .init::<R, f32>();
        let tcfg = TrainerConfig::builder()
            .learning_rate(1e-3)
            .max_grad_norm(clip)
            .build()
            .unwrap();
        let mut trainer = Trainer::new(tcfg, opt);
        trainer.step(&task, std::slice::from_ref(&batch)).unwrap();
        model.named_parameters()
    };
    let a = run(1.0, 1e-8, 0.0);
    let b = run(1024.0, 1e-8 * 1024.0, 0.0);
    for ((na, pa), (nb, pb)) in a.iter().zip(b.iter()) {
        assert_eq!(na, nb);
        let (x, y) = (pa.value().to_f32(), pb.value().to_f32());
        for (i, (u, v)) in x.iter().zip(y.iter()).enumerate() {
            let tol = 1e-5 * (1.0 + u.abs());
            assert!((u - v).abs() < tol, "{na}[{i}]: {u} != {v}");
        }
    }
}

#[test]
fn absent_entities_are_never_chosen() {
    let device = dev();
    let spec = four_head_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let mut a = handmade();
    // Entity 3 absent.
    a.insert_f32("cells.presence", vec![1, 4], vec![1.0, 1.0, 1.0, 0.0]);
    let batch = EntityBatch::<R, f32>::from_host(&spec, &a, &device).unwrap();
    let (_, out) = model.forward_train(&batch).unwrap();
    let p = out.logits["p"].tensor().to_f32();
    // Logits are [1,2,2,5]; entity 3 is column 3 of every row.
    for row in 0..4 {
        let v = p[row * 5 + 3];
        // -1e4 mask plus the row's unmasked dot product (O(1)).
        assert!(v < -9990.0, "row {row}: absent entity logit {v} not masked");
    }
}

#[test]
fn bad_keys_and_ids_are_rejected() {
    let device = dev();
    let spec = four_head_spec();
    let mut a = handmade();
    a.insert_f32("typo", vec![1], vec![0.0]);
    let err = EntityBatch::<R, f32>::from_host(&spec, &a, &device)
        .err()
        .expect("must fail")
        .to_string();
    assert!(err.contains("typo"), "{err}");
    let mut a = handmade();
    a.insert_int("label.p", vec![1, 2, 2], vec![1, 9, -1, 2]);
    let err = EntityBatch::<R, f32>::from_host(&spec, &a, &device)
        .err()
        .expect("must fail")
        .to_string();
    assert!(err.contains("label.p"), "{err}");
}
