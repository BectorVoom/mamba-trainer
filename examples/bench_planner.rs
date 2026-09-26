//! Interleaved fused-vs-composed A/B for one planner optimizer step.
//!
//! Alternates `set_fused_planner(true/false)` every iteration and reports the
//! median step time per mode and the ratio. Interleaving in one process is
//! required: run-to-run wall noise is ±20%.
//!
//! ```text
//! cargo run --release --no-default-features --features vulkan --example bench_planner
//! ```
//!
//! `MAMBA3_PLANNER_BATCH` (default 128) and `MAMBA3_PLANNER_ITERS` (default
//! 100) override the batch and the iterations per mode.

use std::time::{Duration, Instant};

use mamba3::models::planner::{HostBatch, PlannerBatch, PlannerTask, TaskPlannerConfig};
use mamba3::models::set_fused_planner;
use mamba3::prelude::*;

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
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

fn main() -> Result<()> {
    let batch_size = env_usize("MAMBA3_PLANNER_BATCH", 128);
    let iters = env_usize("MAMBA3_PLANNER_ITERS", 100);
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    let cfg = TaskPlannerConfig::default();
    let model = cfg.init::<R, f32>(&device)?;
    let (n, u, k) = (cfg.tiles(), cfg.max_units, cfg.k);
    let q = cfg.queries();
    let host = HostBatch {
        turns: batch_size,
        tiles: frand(batch_size * n * cfg.c_tile, 1),
        glob: frand(batch_size * cfg.c_glob, 2),
        units: frand(batch_size * u * cfg.c_unit, 3),
        upos: vec![0; batch_size * u],
        tgt: vec![0; batch_size * q],
        op: vec![0; batch_size * q],
        crop: vec![0; batch_size * q],
        opset: vec![0; batch_size * q * cfg.n_ops],
        eta: vec![3; batch_size * u],
    };
    let _ = (n, u, k);
    let batch = PlannerBatch::from_host(&cfg, &host, &device)?;
    let task = PlannerTask::new(&model);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(3e-4).build()?,
        AdamWConfig::builder().learning_rate(3e-4).build().init::<R, f32>(),
    );
    for _ in 0..3 {
        trainer.step(&task, std::slice::from_ref(&batch))?;
    }

    let mut fused_times: Vec<Duration> = Vec::with_capacity(iters);
    let mut composed_times: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        for (fused, sink) in [
            (false, &mut composed_times),
            (true, &mut fused_times),
        ] {
            set_fused_planner(fused);
            let started = Instant::now();
            trainer.step(&task, std::slice::from_ref(&batch))?;
            device.synchronize();
            sink.push(started.elapsed());
        }
    }
    let median = |mut v: Vec<Duration>| {
        v.sort_unstable();
        v[v.len() / 2].as_secs_f64() * 1000.0
    };
    let (fused, composed) = (median(fused_times), median(composed_times));
    println!("batch {batch_size}: fused {fused:.1} ms/step, composed {composed:.1} ms/step");
    println!("ratio fused/composed: {:.3}", fused / composed);
    Ok(())
}
