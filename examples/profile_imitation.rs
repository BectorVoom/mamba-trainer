//! Where a DAgger round's time goes.
//!
//! `ImitationLearner.round()` is a collected window mixed with the expert, an
//! imitation batch, one behaviour-cloning optimizer step and — by default — an
//! agreement replay, read together with the step's loss under one synchronisation.
//! This splits one round into those phases the way the Python learner takes them,
//! and prints each phase's launches, reads and wall clock, then the launch tally of
//! the agreement replay and of a window collected with and without the expert.
//!
//! Every phase is timed to a flush, so its time includes its own drain, and the
//! read's is the wait for whatever the device had not finished. At these
//! sizes the round is host-bound (see `profile_ppo`), so launches and reads are
//! the numbers that move the wall clock: a launch is ~10-40 us, a read ~1.4 ms.
//!
//! ```text
//! cargo run --release --features wgpu --example profile_imitation
//! ```
//!
//! `MAMBA3_PPO_ENVS`, `MAMBA3_PPO_WINDOW`, `MAMBA3_PPO_DMODEL` and
//! `MAMBA3_PPO_LAYERS` override the shape, matching `profile_ppo`.

use std::time::{Duration, Instant};

use mamba3::backend::{
    check_launches, launch_count, launch_tally, read_count, reset_launch_count, reset_launch_tally,
    reset_read_count, start_launch_tally, stop_launch_tally,
};
use mamba3::prelude::*;
use mamba3::rl::{
    BehaviourCloningTask, Mamba3Policy, Mamba3PolicyConfig, QueuedAgreement, RecallEnv,
};
use mamba3::tensor::ops::index::read_all;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

const SYMBOLS: usize = 4;
const HORIZON: usize = 8;
const ROUNDS: usize = 20;
const TOP: usize = 25;
const BETA: f32 = 0.5;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Launches, reads and wall clock of one phase, flushed.
#[derive(Default, Clone, Copy)]
struct Phase {
    launches: usize,
    reads: usize,
    time: Duration,
}

fn measure<T>(device: &Device<R>, f: impl FnOnce() -> Result<T>) -> Result<(T, Phase)> {
    reset_launch_count();
    reset_read_count();
    let started = Instant::now();
    let out = f()?;
    check_launches(device)?;
    Ok((
        out,
        Phase {
            launches: launch_count(),
            reads: read_count(),
            time: started.elapsed(),
        },
    ))
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
    let task = BehaviourCloningTask::new(&policy).with_entropy_bonus(0.01);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(3e-3).build()?,
        AdamWConfig::builder()
            .learning_rate(3e-3)
            .build()
            .init::<R, f32>(),
    );
    let mut collector = Collector::new(&policy, envs, window, obs_dim, &device)?
        .with_seed(5)
        .recording_expert_labels();

    // Warm-up: JIT compilation for every kernel shape a round uses.
    for _ in 0..3 {
        collector.collect_with_expert(&mut environment, BETA)?;
        let batch = collector.imitation_batch()?;
        trainer.step(&task, std::slice::from_ref(&batch))?;
        task.agreement(&batch)?;
    }

    let names = ["collect", "batch", "queue step", "queue agree", "read"];
    let mut totals = [Phase::default(); 5];
    let mut round_time = Duration::ZERO;
    for _ in 0..ROUNDS {
        let started = Instant::now();
        let (_, collect) = measure(&device, || {
            collector.collect_with_expert(&mut environment, BETA)
        })?;
        let (batch, batch_phase) = measure(&device, || collector.imitation_batch())?;
        let (queued, queue) = measure(&device, || {
            trainer.queue_step(&task, std::slice::from_ref(&batch))
        })?;
        let (agreement, queue_agreement) = measure(&device, || task.queue_agreement(&batch))?;
        let (_, read) = measure(&device, || {
            let scalars: Vec<&Tensor<R, f32>> = queued
                .scalars()
                .chain(QueuedAgreement::scalars(&agreement))
                .collect();
            let (_, values) = read_all(&[], &scalars)?;
            let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
            let (step, agreed) = values.split_at(queued.scalars().count());
            trainer.report_steps(std::slice::from_ref(&queued), step);
            Ok(agreement.fraction(agreed))
        })?;
        round_time += started.elapsed();
        for (total, phase) in
            totals
                .iter_mut()
                .zip([collect, batch_phase, queue, queue_agreement, read])
        {
            total.launches += phase.launches;
            total.reads += phase.reads;
            total.time += phase.time;
        }
    }

    println!(
        "{:<12} {:>9} {:>7} {:>9}",
        "phase", "launches", "reads", "ms"
    );
    for (name, total) in names.iter().zip(&totals) {
        println!(
            "{name:<12} {:>9} {:>7} {:>9.2}",
            total.launches / ROUNDS,
            total.reads / ROUNDS,
            total.time.as_secs_f64() * 1e3 / ROUNDS as f64,
        );
    }
    println!(
        "{:<12} {:>9} {:>7} {:>9.2}\n",
        "round",
        totals.iter().map(|t| t.launches).sum::<usize>() / ROUNDS,
        totals.iter().map(|t| t.reads).sum::<usize>() / ROUNDS,
        round_time.as_secs_f64() * 1e3 / ROUNDS as f64,
    );

    // The plain collect is the reference the expert mixing is charged against.
    for (label, run) in [
        ("agreement", 0usize),
        ("collect with expert", 1),
        ("collect without expert", 2),
    ] {
        start_launch_tally();
        reset_launch_tally();
        reset_launch_count();
        match run {
            0 => {
                let batch = collector.imitation_batch()?;
                task.agreement(&batch)?;
            }
            1 => {
                collector.collect_with_expert(&mut environment, BETA)?;
            }
            _ => {
                collector.collect(&mut environment)?;
            }
        }
        stop_launch_tally();
        let launches = launch_count();
        println!("{label}: {launches} launches");
        println!("{:>6} {:>6}  site", "count", "share");
        for (site, count) in launch_tally().into_iter().take(TOP) {
            println!(
                "{count:>6} {:>5.1}%  {site}",
                100.0 * count as f64 / launches as f64
            );
        }
        println!();
    }
    Ok(())
}
