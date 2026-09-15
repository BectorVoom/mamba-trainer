//! Where a PPO update's dispatches go.
//!
//! `profile_rollout` attributes a collected window; this attributes one optimizer
//! step of the update that follows — replay, objective, backward, clip and AdamW —
//! to the source lines that launched it, the way `PpoLearner.update()` takes it
//! (`Trainer::queue_step`), and prints the host time of queueing it against the
//! drain. At these sizes the update is host-bound, so the launch count is the
//! number that moves the wall clock.
//!
//! ```text
//! cargo run --release --features wgpu --example profile_update
//! ```
//!
//! `MAMBA3_PPO_ENVS`, `MAMBA3_PPO_WINDOW`, `MAMBA3_PPO_DMODEL` and
//! `MAMBA3_PPO_LAYERS` override the shape, matching `profile_ppo`.

use std::time::Instant;

use mamba3::backend::{
    launch_count, launch_tally, read_count, reset_launch_count, reset_launch_tally,
    reset_read_count, start_launch_tally, stop_launch_tally,
};
use mamba3::prelude::*;
use mamba3::rl::{Mamba3Policy, Mamba3PolicyConfig, PpoTask, RecallEnv};
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

const SYMBOLS: usize = 4;
const HORIZON: usize = 8;
const STEPS: usize = 16;
const TOP: usize = 40;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let envs = env_usize("MAMBA3_PPO_ENVS", 32);
    let window = env_usize("MAMBA3_PPO_WINDOW", HORIZON * 4);
    let d_model = env_usize("MAMBA3_PPO_DMODEL", 64);
    let layers = env_usize("MAMBA3_PPO_LAYERS", 2);
    println!("backend: {}", device.name());
    println!("shape:   {envs} envs x {window} steps, d_model {d_model}, {layers} layers\n");

    let mut environment = RecallEnv::<R, f32>::new(envs, SYMBOLS, HORIZON, 23, &device)?;
    let obs_dim = environment.obs_dim();
    let policy: Mamba3Policy<R, f32> = Mamba3PolicyConfig::new(obs_dim, SYMBOLS, d_model, layers)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.head_dim = 16;
            s.n_groups = 4;
            s.d_state = 8;
            s.chunk_size = 8;
            s.conv_kernel = Some(4);
        })
        .init::<R, f32>(&device)?;
    let config = PpoConfig::default();
    let task = PpoTask::new(&policy, config);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-3).build()?,
        AdamWConfig::builder()
            .learning_rate(1e-3)
            .build()
            .init::<R, f32>(),
    );
    let mut collector = Collector::new(&policy, envs, window, obs_dim, &device)?.with_seed(5);
    let report = collector.collect(&mut environment)?;
    let batch = collector.ppo_batch(&report, &config)?;

    // Warm-up: JIT compilation for every kernel shape a step uses.
    for _ in 0..3 {
        let queued = trainer.queue_step(&task, std::slice::from_ref(&batch))?;
        trainer.read_steps(std::slice::from_ref(&queued))?;
    }

    reset_launch_count();
    reset_read_count();
    start_launch_tally();
    reset_launch_tally();
    let queued = trainer.queue_step(&task, std::slice::from_ref(&batch))?;
    stop_launch_tally();
    trainer.read_steps(std::slice::from_ref(&queued))?;
    let per_step = launch_count();
    println!(
        "one optimizer step: {per_step} launches, {} read\n",
        read_count()
    );
    println!("{:>6} {:>6}  site", "count", "share");
    for (site, count) in launch_tally().into_iter().take(TOP) {
        println!(
            "{count:>6} {:>5.1}%  {site}",
            100.0 * count as f64 / per_step as f64
        );
    }

    let started = Instant::now();
    let mut queued = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        queued.push(trainer.queue_step(&task, std::slice::from_ref(&batch))?);
    }
    let submitted = started.elapsed();
    trainer.read_steps(&queued)?;
    let total = started.elapsed();
    println!(
        "\n{STEPS} steps: host submission {:.2} ms/step ({:.1} us/launch), drain {:.2} ms/step",
        submitted.as_secs_f64() * 1e3 / STEPS as f64,
        submitted.as_secs_f64() * 1e6 / (STEPS * per_step) as f64,
        (total - submitted).as_secs_f64() * 1e3 / STEPS as f64,
    );
    Ok(())
}
