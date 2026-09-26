//! Task planner (TASK_PLANNER_PLAN.md T1–T5): config, forward, batch, loss, checkpoint.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::planner::{HostBatch, PlannerBatch, PlannerTask, TaskPlanner, TaskPlannerConfig};
use mamba3::nn::Module;
use mamba3::prelude::*;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{Optimizer, TrainStep};

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn tiny_cfg() -> TaskPlannerConfig {
    let mut c = TaskPlannerConfig::default();
    c.grid = 2;
    c.max_units = 2;
    c.k = 2;
    c.d_model = 8;
    c.n_tile_layers = 1;
    c.n_joint_layers = 1;
    c.ssm.n_heads = 2;
    c.ssm.n_groups = 2;
    c.ssm.head_dim = 4;
    c.ssm.d_state = 4;
    c.ssm.chunk_size = 4;
    c.seed = 0;
    c
}

fn small_cfg() -> TaskPlannerConfig {
    let mut c = TaskPlannerConfig::default();
    c.d_model = 32;
    c.n_tile_layers = 1;
    c.n_joint_layers = 1;
    c.ssm.n_heads = 2;
    c.ssm.n_groups = 1;
    c.ssm.head_dim = 32;
    c.ssm.d_state = 8;
    c.ssm.chunk_size = 32;
    c.seed = 0;
    c
}

/// Deterministic pseudo-random floats in [-1, 1].
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

/// A random host batch with valid labels: B turns, `active` of U units work.
fn random_host(cfg: &TaskPlannerConfig, b: usize, active: usize, seed: u64) -> HostBatch {
    let (n, u, k) = (cfg.tiles(), cfg.max_units, cfg.k);
    let q = cfg.queries();
    let mut upos = vec![-1i32; b * u];
    let mut tgt = vec![-100i32; b * q];
    let mut op = vec![-100i32; b * q];
    let mut crop = vec![-100i32; b * q];
    let mut opset = vec![0u8; b * q * cfg.n_ops];
    let mut eta = vec![-1i32; b * u];
    let mut s = seed.max(1);
    let mut ri = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    for bi in 0..b {
        for uu in 0..active.min(u) {
            upos[bi * u + uu] = ri(n) as i32;
            eta[bi * u + uu] = ri(6) as i32;
            for j in 0..k {
                let qi = (bi * u + uu) * k + j;
                // 4/5 of visits are real tiles, 1/5 NONE.
                let t = if ri(5) < 4 { ri(n) as i32 } else { n as i32 };
                tgt[bi * q + uu * k + j] = t;
                if t < n as i32 {
                    op[bi * q + uu * k + j] = ri(cfg.n_ops) as i32;
                    if ri(2) == 0 {
                        crop[bi * q + uu * k + j] = ri(cfg.n_crops) as i32;
                    }
                    let nset = 1 + ri(3);
                    for _ in 0..nset {
                        opset[(bi * q + uu * k + j) * cfg.n_ops + ri(cfg.n_ops)] = 1;
                    }
                }
                let _ = qi;
            }
        }
    }
    HostBatch {
        turns: b,
        tiles: frand(b * n * cfg.c_tile, seed + 1),
        glob: frand(b * cfg.c_glob, seed + 2),
        units: frand(b * u * cfg.c_unit, seed + 3),
        upos,
        tgt,
        op,
        crop,
        opset,
        eta,
    }
}

#[test]
fn default_config_validates_and_round_trips_json() {
    let c = TaskPlannerConfig::default();
    c.validate().unwrap();
    assert_eq!(c.tiles(), 100);
    assert_eq!(c.queries(), 60);
    assert_eq!(c.aux_width(), 13 + 13 + 5 + 1);
    let s = serde_json::to_string(&c).unwrap();
    assert_eq!(serde_json::from_str::<TaskPlannerConfig>(&s).unwrap(), c);
}

#[test]
fn forward_shapes() {
    let cfg = TaskPlannerConfig::default();
    let model = cfg.init::<R, f32>(&dev()).unwrap();
    let h = random_host(&cfg, 2, 3, 7);
    let b = PlannerBatch::from_host(&cfg, &h, &dev()).unwrap();
    let out = model
        .forward(
            &Var::constant(b.tiles.clone()),
            &Var::constant(b.glob.clone()),
            &Var::constant(b.units.clone()),
            &b.unit_onehot,
        )
        .unwrap();
    assert_eq!(out.target_logits.shape().dims(), &[2, 60, 101]);
    let aux = model.heads(&out, &b.target_onehot).unwrap();
    assert_eq!(aux.shape().dims(), &[2, 60, 32]);
    assert!(out.target_logits.tensor().to_f32().iter().all(|v| v.is_finite()));
    assert!(aux.tensor().to_f32().iter().all(|v| v.is_finite()));
}

#[test]
fn gradient_reaches_every_parameter() {
    let cfg = small_cfg();
    let model = cfg.init::<R, f32>(&dev()).unwrap();
    let h = random_host(&cfg, 1, 2, 11);
    let b = PlannerBatch::from_host(&cfg, &h, &dev()).unwrap();
    let out = model
        .forward(
            &Var::constant(b.tiles.clone()),
            &Var::constant(b.glob.clone()),
            &Var::constant(b.units.clone()),
            &b.unit_onehot,
        )
        .unwrap();
    let aux = model.heads(&out, &b.target_onehot).unwrap();
    let loss = out.target_logits.sum().unwrap().add(&aux.sum().unwrap()).unwrap();
    let grads = loss.backward().unwrap();
    for (name, p) in model.named_parameters() {
        let g = grads
            .get(p.id())
            .unwrap_or_else(|| panic!("no gradient reached {name}"));
        let norm: f32 = g.to_f32().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "zero gradient at {name}");
    }
}

/// Central-difference gradient of the task loss wrt a few elements of a parameter.
fn check_param_grad(cfg: &TaskPlannerConfig, name: &str, idxs: &[usize]) {
    let model = cfg.init::<R, f32>(&dev()).unwrap();
    let h = random_host(cfg, 1, 2, 13);
    let b = PlannerBatch::from_host(cfg, &h, &dev()).unwrap();
    let task = PlannerTask::new(&model);
    let loss_of = || task.loss(&b).unwrap().to_f32()[0];
    let params: std::collections::HashMap<String, _> =
        model.named_parameters().into_iter().collect();
    let p = params.get(name).unwrap_or_else(|| panic!("no param {name}"));
    let analytic = task.loss(&b).unwrap().backward().unwrap().get(p.id()).unwrap().to_f32();
    let eps = 1e-3f32;
    for &i in idxs {
        let mut v = p.value().to_f32();
        v[i] += eps;
        p.set(Tensor::from_f32(&v, p.shape().clone(), &dev()).unwrap());
        let fp = loss_of();
        v[i] -= 2.0 * eps;
        p.set(Tensor::from_f32(&v, p.shape().clone(), &dev()).unwrap());
        let fm = loss_of();
        v[i] += eps;
        p.set(Tensor::from_f32(&v, p.shape().clone(), &dev()).unwrap());
        let numeric = (fp - fm) / (2.0 * eps);
        let tol = 2e-2 * (1.0 + numeric.abs());
        assert!(
            (analytic[i] - numeric).abs() < tol,
            "{name}[{i}] analytic={} numeric={}",
            analytic[i],
            numeric
        );
    }
}

#[test]
fn loss_gradients_match_finite_differences() {
    let cfg = tiny_cfg();
    check_param_grad(&cfg, "tile_pos", &[0, 1, 5]);
    check_param_grad(&cfg, "none_key", &[0, 3]);
    // One Linear weight (first tile input projection).
    check_param_grad(&cfg, "tile_in1.weight", &[0, 2]);
}

#[test]
fn onehot_gather_equals_indexing() {
    let cfg = TaskPlannerConfig::default();
    let (n, d, u) = (cfg.tiles(), cfg.d_model, cfg.max_units);
    let t = Tensor::<R, f32>::from_f32(&frand(n * d, 21), vec![1, n, d], &dev()).unwrap();
    let upos = vec![3u32, 3u32, 42u32, 0u32];
    let ids = IdTensor::from_slice(&upos, vec![1, u.min(4)], &dev()).unwrap();
    let oh = mamba3::tensor::ops::index::one_hot::<R, f32>(&ids, n).unwrap();
    let got = Var::constant(oh).matmul(&Var::constant(t.clone())).unwrap().tensor().to_f32();
    let t = t.to_f32();
    for (r, &id) in upos.iter().enumerate() {
        let row = &got[r * d..(r + 1) * d];
        let want = &t[id as usize * d..(id as usize + 1) * d];
        for (a, b) in row.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "row {r}: {a} != {b}");
        }
    }
    let _ = u;
}

/// Host f64 reference for one component row.
fn ref_components(
    logits: &[f32],
    aux: &[f32],
    h: &HostBatch,
    cfg: &TaskPlannerConfig,
) -> [f64; 5] {
    let (n, no, nc, aw) = (cfg.tiles(), cfg.n_ops, cfg.n_crops, cfg.aux_width());
    let (u, k, q) = (cfg.max_units, cfg.k, cfg.queries());
    let lse = |xs: &[f64]| {
        let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
    };
    let mut acc = [0.0; 5];
    let mut den = [0.0; 5];
    for bi in 0..h.turns {
        for uu in 0..u {
            for j in 0..k {
                let f = bi * q + uu * k + j;
                let t = h.tgt[f];
                if t != -100 {
                    let row: Vec<f64> = (0..n + 1).map(|c| logits[f * (n + 1) + c] as f64).collect();
                    let w = if k == 3 { [1.0, 0.5, 0.5][j] } else { 1.0 };
                    acc[0] += w * (lse(&row) - row[t as usize]);
                    den[0] += w;
                }
                let tgt_tile = t >= 0 && (t as usize) < n;
                let o = h.op[f];
                if tgt_tile && o >= 0 {
                    let row: Vec<f64> =
                        (0..no).map(|c| aux[f * aw + c] as f64).collect();
                    acc[1] += lse(&row) - row[o as usize];
                    den[1] += 1.0;
                    let mut bce = 0.0;
                    for i in 0..no {
                        let x = aux[f * aw + no + i] as f64;
                        let y = h.opset[f * no + i] as f64;
                        bce += x.max(0.0) + (-x.abs()).exp().ln_1p() - x * y;
                    }
                    acc[2] += bce;
                    den[2] += 1.0;
                }
                let c = h.crop[f];
                if c >= 0 {
                    let row: Vec<f64> =
                        (0..nc).map(|ci| aux[f * aw + 2 * no + ci] as f64).collect();
                    acc[3] += lse(&row) - row[c as usize];
                    den[3] += 1.0;
                }
                if j == 0 && h.eta[bi * u + uu] >= 0 {
                    let d = aux[f * aw + aw - 1] as f64 - ((h.eta[bi * u + uu] as f64) + 1.0).ln();
                    acc[4] += d * d;
                    den[4] += 1.0;
                }
            }
        }
    }
    // Order: [target, op, opset, crop, eta] to match component_losses.
    let norm = |a: f64, d: f64| a / d.max(1.0);
    [norm(acc[0], den[0]), norm(acc[1], den[1]), norm(acc[2], den[2] * no as f64), norm(acc[3], den[3]), norm(acc[4], den[4])]
}

#[test]
fn loss_matches_host_reference() {
    let cfg = small_cfg();
    let model = cfg.init::<R, f32>(&dev()).unwrap();
    // 1 turn, 2 active units: exercises kept, ignored, NONE and padding rows.
    let h = random_host(&cfg, 1, 2, 31);
    let b = PlannerBatch::from_host(&cfg, &h, &dev()).unwrap();
    let task = PlannerTask::new(&model);
    let got: Vec<f32> = task
        .component_losses(&b)
        .unwrap()
        .iter()
        .map(|v| v.to_f32()[0])
        .collect();
    // Host reference from the same forward values.
    let out = model
        .forward(
            &Var::constant(b.tiles.clone()),
            &Var::constant(b.glob.clone()),
            &Var::constant(b.units.clone()),
            &b.unit_onehot,
        )
        .unwrap();
    let logits = out.target_logits.tensor().to_f32();
    let aux = model.heads(&out, &b.target_onehot).unwrap().tensor().to_f32();
    let want = ref_components(&logits, &aux, &h, &cfg);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (*g as f64 - *w).abs() < 1e-4,
            "component {i}: device {g} != host {w}"
        );
    }
}

// NOTE: the no-host-reads check lives alone in tests/planner_footprint.rs:
// the launch/read counters are process-wide, and a test running beside it
// would add to them (see tests/rl_entity_footprint.rs).

#[test]
fn overfits_a_fixed_batch() {
    let cfg = small_cfg();
    let model = cfg.init::<R, f32>(&dev()).unwrap();
    let h = random_host(&cfg, 1, 2, 41);
    let b = PlannerBatch::from_host(&cfg, &h, &dev()).unwrap();
    let task = PlannerTask::new(&model);
    let mut opt = AdamWConfig::builder()
        .learning_rate(3e-3)
        .weight_decay(0.0)
        .build()
        .init::<R, f32>();
    let params = task.parameters();
    let start = task.loss(&b).unwrap().to_f32()[0];
    for _ in 0..200 {
        let loss = task.loss(&b).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&params, &grads).unwrap();
    }
    let end = task.loss(&b).unwrap().to_f32()[0];
    assert!(end < 0.1 * start, "loss {start} -> {end}, expected < 10%");
}

#[test]
fn loss_scale_leaves_updates_unchanged() {
    // NOTE: the tight phase runs ONE optimizer step. The §2.5 rescaling (eps,
    // clip) enters every AdamW update from step 1, so one step proves the math;
    // more steps only compound 1-ulp chaos through the nonlinear dynamics
    // (measured worst rel drift 3.7e-8 after 1 step, 1.2e-3 after 20).
    let run = |scale: f32, eps: f32, clip: f32, steps: usize| {
        let cfg = small_cfg();
        let model = cfg.init::<R, f32>(&dev()).unwrap();
        let h = random_host(&cfg, 1, 2, 43);
        let b = PlannerBatch::from_host(&cfg, &h, &dev()).unwrap();
        let task = PlannerTask::new(&model).with_loss_scale(scale);
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
        for _ in 0..steps {
            trainer.step(&task, std::slice::from_ref(&b)).unwrap();
        }
        model.named_parameters()
    };
    let a = run(1.0, 1e-8, 0.0, 1);
    let b = run(1024.0, 1e-8 * 1024.0, 0.0, 1);
    for ((na, pa), (nb, pb)) in a.iter().zip(b.iter()) {
        assert_eq!(na, nb);
        let (x, y) = (pa.value().to_f32(), pb.value().to_f32());
        for (i, (u, v)) in x.iter().zip(y.iter()).enumerate() {
            let tol = 1e-5 * (1.0 + u.abs());
            assert!((u - v).abs() < tol, "{na}[{i}]: {u} != {v}");
        }
    }
    // With the clip engaged the agreement is close but not exact: the shared
    // `clip_factor` kernel adds 1e-6 to the norm *after* the loss scale has
    // multiplied it (`clip = max_norm / (S*norm + 1e-6)` vs `1 / (norm + 1e-6)`),
    // so the two runs' effective gradients differ at ~1e-7 relative per step and
    // Adam's state compounds that. 1e-3 absolute still proves the clip is scaled
    // (an unscaled clip would shrink every update 1024x and diverge ~1e-2).
    // One step only: the step-1 grad norm (≈11) already engages the clip, and
    // further steps only compound 1-ulp chaos (see the NOTE above).
    let a = run(1.0, 1e-8, 1.0, 1);
    let b = run(1024.0, 1e-8 * 1024.0, 1024.0, 1);
    for ((na, pa), (nb, pb)) in a.iter().zip(b.iter()) {
        assert_eq!(na, nb);
        let (x, y) = (pa.value().to_f32(), pb.value().to_f32());
        for (i, (u, v)) in x.iter().zip(y.iter()).enumerate() {
            assert!((u - v).abs() < 1e-3, "{na}[{i}]: {u} != {v}");
        }
    }
}

#[test]
fn save_load_predict_round_trip() {
    let cfg = small_cfg();
    let model = cfg.init::<R, f32>(&dev()).unwrap();
    let h = random_host(&cfg, 2, 2, 61);
    let b = PlannerBatch::from_host(&cfg, &h, &dev()).unwrap();
    let before = model.predict(&b).unwrap();
    let path = std::env::temp_dir().join("mamba3_planner_roundtrip.m3ck");
    model.save(&path, 7).unwrap();
    let loaded = TaskPlanner::<R, f32>::load(&path, &dev()).unwrap();
    let after = loaded.predict(&b).unwrap();
    assert_eq!(before.0.to_f32(), after.0.to_f32());
    assert_eq!(before.1.to_f32(), after.1.to_f32());
    let _ = std::fs::remove_file(&path);
}
