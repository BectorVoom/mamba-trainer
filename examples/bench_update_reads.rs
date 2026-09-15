//! What reading a PPO update's numbers once, instead of per step, is worth.
//!
//! `PpoLearner.update()` used to read each step's loss and gradient norm as the
//! step finished, cut each minibatch's actions through the host, and read the six
//! diagnostics one at a time: 14 reads for four epochs, 54 over four minibatches.
//! It now queues every step and reads everything once. Both patterns run here over
//! the same policy, interleaved round by round in one process — wall-clock noise
//! between separate runs on this machine is ±20%, which would hide the answer.
//!
//! ```text
//! cargo run --release --features wgpu --example bench_update_reads
//! ```

use std::time::Instant;

use mamba3::backend::{read_count, reset_read_count};
use mamba3::prelude::*;
use mamba3::rl::{Mamba3Policy, Mamba3PolicyConfig, PpoBatch, PpoTask, RecallEnv};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::{IdTensor, read_all};
use mamba3::train::{AdamW, AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

const ENVS: usize = 32;
const WINDOW: usize = 32;
const EPOCHS: usize = 4;
const ROUNDS: usize = 15;

/// The old per-step pattern: host-cut minibatches, two reads per step, six reads
/// for the diagnostics.
fn update_reading_each(
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    task: &PpoTask<'_, R, f32>,
    batch: &PpoBatch<R, f32>,
    minibatches: usize,
) -> Result<()> {
    let per = ENVS / minibatches;
    for _ in 0..EPOCHS {
        for index in 0..minibatches {
            let micro = if minibatches == 1 {
                batch.clone()
            } else {
                let mut micro = batch.minibatch(index * per, per)?;
                let ids = batch.actions.try_to_vec()?;
                let kept = &ids[index * per * WINDOW..(index + 1) * per * WINDOW];
                micro.actions =
                    IdTensor::from_slice(kept, vec![per, WINDOW], batch.observations.device())?;
                micro
            };
            let queued = trainer.queue_step(task, std::slice::from_ref(&micro))?;
            for scalar in queued.scalars() {
                scalar.try_to_f32()?;
            }
        }
    }
    for stat in task.stat_tensors().expect("a loss was taken") {
        stat.try_to_f32()?;
    }
    Ok(())
}

/// The new pattern, as `PpoLearner.update()` runs it.
fn update_reading_once(
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    task: &PpoTask<'_, R, f32>,
    batch: &PpoBatch<R, f32>,
    minibatches: usize,
) -> Result<()> {
    let per = ENVS / minibatches;
    let micros = (0..minibatches)
        .map(|index| batch.minibatch(index * per, per))
        .collect::<Result<Vec<_>>>()?;
    let mut queued = Vec::with_capacity(EPOCHS * minibatches);
    for _ in 0..EPOCHS {
        for micro in &micros {
            queued.push(trainer.queue_step(task, std::slice::from_ref(micro))?);
        }
    }
    let diagnostics = task.stat_tensors().expect("a loss was taken");
    let mut scalars: Vec<&Tensor<R, f32>> = queued.iter().flat_map(|q| q.scalars()).collect();
    let steps = scalars.len();
    scalars.extend(&diagnostics);
    let (_, values) = read_all(&[], &scalars)?;
    let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
    trainer.report_steps(&queued, &values[..steps]);
    Ok(())
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let mut env = RecallEnv::<R, f32>::new(ENVS, 4, 8, 23, &device)?;
    let obs_dim = env.obs_dim();
    let policy: Mamba3Policy<R, f32> = Mamba3PolicyConfig::new(obs_dim, 4, 64, 2)
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
        AdamWConfig::builder().learning_rate(1e-3).build().init(),
    );
    let mut collector = Collector::new(&policy, ENVS, WINDOW, obs_dim, &device)?.with_seed(5);
    let report = collector.collect(&mut env)?;
    let batch = collector.ppo_batch(&report, &config)?;

    println!(
        "backend: {}, {ENVS} envs x {WINDOW} steps, {EPOCHS} epochs\n",
        device.name()
    );
    println!(
        "{:<14} {:>14} {:>14} {:>9}",
        "minibatches", "each (ms, rd)", "once (ms, rd)", "saved"
    );
    for minibatches in [1, 4] {
        // Warm-up: compile every kernel shape either pattern uses.
        for _ in 0..2 {
            update_reading_each(&mut trainer, &task, &batch, minibatches)?;
            update_reading_once(&mut trainer, &task, &batch, minibatches)?;
        }
        let (mut each, mut once) = (Vec::new(), Vec::new());
        let (mut each_reads, mut once_reads) = (0, 0);
        for _ in 0..ROUNDS {
            device.synchronize();
            reset_read_count();
            let started = Instant::now();
            update_reading_each(&mut trainer, &task, &batch, minibatches)?;
            device.synchronize();
            each.push(started.elapsed().as_secs_f64() * 1e3);
            each_reads = read_count();

            reset_read_count();
            let started = Instant::now();
            update_reading_once(&mut trainer, &task, &batch, minibatches)?;
            device.synchronize();
            once.push(started.elapsed().as_secs_f64() * 1e3);
            once_reads = read_count();
        }
        let (each, once) = (median(each), median(once));
        println!(
            "{minibatches:<14} {each:>8.1} ({each_reads:>2}) {once:>8.1} ({once_reads:>2}) {:>8.0}%",
            100.0 * (each - once) / each
        );
    }
    Ok(())
}
