//! The planner loss reads nothing back.
//!
//! Alone in its binary for the reason `rl_footprint.rs` gives: the launch and
//! read counters are process-wide, and a test running beside this one would add
//! to them.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::planner::{HostBatch, PlannerBatch, PlannerTask, TaskPlannerConfig};
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

#[test]
fn loss_and_backward_do_no_host_reads() {
    let mut cfg = TaskPlannerConfig::default();
    cfg.d_model = 32;
    cfg.n_tile_layers = 1;
    cfg.n_joint_layers = 1;
    cfg.ssm.n_heads = 2;
    cfg.ssm.n_groups = 1;
    cfg.ssm.head_dim = 32;
    cfg.ssm.d_state = 8;
    cfg.ssm.chunk_size = 32;
    let model = cfg.init::<R, f32>(&dev()).unwrap();
    let (n, u, k) = (cfg.tiles(), cfg.max_units, cfg.k);
    let q = cfg.queries();
    let h = HostBatch {
        turns: 1,
        tiles: frand(n * cfg.c_tile, 1),
        glob: frand(cfg.c_glob, 2),
        units: frand(u * cfg.c_unit, 3),
        upos: vec![-1; u],
        tgt: vec![-100; q],
        op: vec![-100; q],
        crop: vec![-100; q],
        opset: vec![0; q * cfg.n_ops],
        eta: vec![-1; u],
    };
    let _ = k;
    let b = PlannerBatch::from_host(&cfg, &h, &dev()).unwrap();
    let task = PlannerTask::new(&model);
    // Warm up once (lazy init), then measure.
    task.loss(&b).unwrap().backward().unwrap();
    let before = mamba3::backend::read_count();
    let loss = task.loss(&b).unwrap();
    loss.backward().unwrap();
    assert_eq!(mamba3::backend::read_count() - before, 0);
}
