//! f16 smoke for the planner loss (TASK_PLANNER_PLAN.md T4 test 5).
//!
//! Alone in its binary: the matmul precision is a process-global mode, and a
//! test running beside this one would compute its own matmuls in f16.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::planner::{HostBatch, PlannerBatch, PlannerTask, TaskPlannerConfig};
use mamba3::prelude::*;
use mamba3::train::{Optimizer, TrainStep};

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

#[test]
fn f16_smoke_on_capable_backends() {
    use mamba3::tensor::ops::matmul::{MatmulPrecision, supports_matmul_precision};
    let device = dev();
    if !supports_matmul_precision(&device, MatmulPrecision::F16) {
        return;
    }
    let mut cfg = TaskPlannerConfig::default();
    cfg.d_model = 32;
    cfg.n_tile_layers = 1;
    cfg.n_joint_layers = 1;
    cfg.ssm.n_heads = 2;
    cfg.ssm.n_groups = 1;
    cfg.ssm.head_dim = 32;
    cfg.ssm.d_state = 8;
    cfg.ssm.chunk_size = 32;
    cfg.seed = 51;
    // Random but valid labels: 2 active units, real-tile-or-NONE targets.
    let (n, u, k, q) = (cfg.tiles(), cfg.max_units, cfg.k, cfg.queries());
    let mut upos = vec![-1i32; u];
    let mut tgt = vec![-100i32; q];
    let mut op = vec![-100i32; q];
    let crop = vec![-100i32; q];
    let mut opset = vec![0u8; q * cfg.n_ops];
    let mut eta = vec![-1i32; u];
    let mut s = 53u64;
    let mut ri = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    for uu in 0..2 {
        upos[uu] = ri(n) as i32;
        eta[uu] = ri(6) as i32;
        for j in 0..k {
            let t = if ri(5) < 4 { ri(n) as i32 } else { n as i32 };
            tgt[uu * k + j] = t;
            if t < n as i32 {
                op[uu * k + j] = ri(cfg.n_ops) as i32;
                opset[(uu * k + j) * cfg.n_ops + ri(cfg.n_ops)] = 1;
            }
        }
    }
    let h = HostBatch {
        turns: 1,
        tiles: frand(n * cfg.c_tile, 54),
        glob: frand(cfg.c_glob, 55),
        units: frand(u * cfg.c_unit, 56),
        upos,
        tgt,
        op,
        crop,
        opset,
        eta,
    };
    mamba3::tensor::ops::matmul::set_matmul_precision(MatmulPrecision::F16);
    let result = (|| -> mamba3::error::Result<Vec<f32>> {
        let model = cfg.init::<R, f32>(&device)?;
        let b = PlannerBatch::from_host(&cfg, &h, &device)?;
        let task = PlannerTask::new(&model).with_loss_scale(1024.0);
        let mut opt = AdamWConfig::builder()
            .learning_rate(1e-3)
            .eps(1e-8 * 1024.0)
            .weight_decay(0.0)
            .build()
            .init::<R, f32>();
        let params = task.parameters();
        let mut losses = Vec::new();
        for _ in 0..50 {
            let loss = task.loss(&b)?;
            losses.push(loss.to_f32()[0] / 1024.0);
            let grads = loss.backward()?;
            opt.step(&params, &grads)?;
        }
        Ok(losses)
    })();
    mamba3::tensor::ops::matmul::set_matmul_precision(MatmulPrecision::F32);
    let losses = result.unwrap();
    assert!(losses.iter().all(|v| v.is_finite()), "non-finite f16 loss");
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "f16 losses did not decrease: {} -> {}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}
