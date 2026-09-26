//! Planner training-step profile: launches, reads and time per stage.
//!
//! One training step of the default config at batch 128 (the T8 gate shape),
//! in both `set_fused_planner` modes, forward / backward / update separately,
//! with the launch tally per source line. Until a K kernel lands both modes
//! run the composed oracle and must print the same counts.
//!
//! ```text
//! cargo run --release --no-default-features --features cpu --example profile_planner
//! ```
//!
//! `MAMBA3_PLANNER_BATCH` overrides the batch (turns); `MAMBA3_PLANNER_STEPS`
//! overrides the timed steps per stage.

use std::time::Instant;

use mamba3::backend::{
    launch_count, launch_tally, read_count, reset_launch_count, reset_launch_tally,
    reset_read_count, start_launch_tally, stop_launch_tally,
};
use mamba3::models::planner::{HostBatch, PlannerBatch, PlannerTask, TaskPlannerConfig};
use mamba3::models::set_fused_planner;
use mamba3::prelude::*;
use mamba3::train::TrainStep;

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
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

/// Random host batch with mostly-valid labels.
fn random_host(cfg: &TaskPlannerConfig, b: usize, seed: u64) -> HostBatch {
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
        for uu in 0..u {
            if ri(5) > 0 {
                upos[bi * u + uu] = ri(n) as i32;
                eta[bi * u + uu] = ri(20) as i32;
            }
            for j in 0..k {
                let f = bi * q + uu * k + j;
                let r = ri(10);
                if r < 7 {
                    tgt[f] = ri(n) as i32;
                    op[f] = ri(cfg.n_ops) as i32;
                    crop[f] = ri(cfg.n_crops) as i32;
                    opset[f * cfg.n_ops + ri(cfg.n_ops)] = 1;
                } else if r < 8 {
                    tgt[f] = n as i32;
                }
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

fn profile_stage(
    label: &str,
    steps: usize,
    device: &Device<R>,
    mut body: impl FnMut() -> Result<()>,
) -> Result<()> {
    for _ in 0..steps.min(2) {
        body()?;
    }
    device.synchronize();
    reset_launch_count();
    reset_read_count();
    reset_launch_tally();
    start_launch_tally();
    let started = Instant::now();
    for _ in 0..steps {
        body()?;
    }
    device.synchronize();
    let ms = started.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    stop_launch_tally();
    println!(
        "{label:<28} {:>8} launches {:>4} reads {:>10.1} ms/step",
        launch_count(),
        read_count(),
        ms
    );
    for (site, count) in launch_tally().into_iter().take(15) {
        println!("    {count:>8}  {site}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let batch_size = env_usize("MAMBA3_PLANNER_BATCH", 128);
    let steps = env_usize("MAMBA3_PLANNER_STEPS", 5);
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    let cfg = TaskPlannerConfig::default();
    let model = cfg.init::<R, f32>(&device)?;
    let host = random_host(&cfg, batch_size, 99);
    let batch = PlannerBatch::from_host(&cfg, &host, &device)?;
    let task = PlannerTask::new(&model);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(3e-4).build()?,
        AdamWConfig::builder().learning_rate(3e-4).build().init::<R, f32>(),
    );
    // Warm up (compiles kernels, settles the allocator).
    for _ in 0..3 {
        trainer.step(&task, std::slice::from_ref(&batch))?;
    }
    device.synchronize();

    for fused in [false, true] {
        set_fused_planner(fused);
        println!(
            "\n== fused={fused} (batch {batch_size}, d={} {}+{} layers) ==",
            cfg.d_model, cfg.n_tile_layers, cfg.n_joint_layers
        );
        profile_stage("forward", steps, &device, || {
            task.loss(&batch)?;
            Ok(())
        })?;
        profile_stage("forward+backward", steps, &device, || {
            task.loss(&batch)?.backward()?;
            Ok(())
        })?;
        profile_stage("optimizer step", steps, &device, || {
            trainer.step(&task, std::slice::from_ref(&batch))?;
            Ok(())
        })?;
    }
    Ok(())
}
