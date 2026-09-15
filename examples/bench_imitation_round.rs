//! A/B of a DAgger round's synchronisation, and of the mixture's scoring.
//!
//! Two comparisons, each interleaved inside one process because run-to-run noise
//! on a shared machine is larger than the effects:
//!
//! * **round** — `before` is what `ImitationLearner.round()` used to do: read the
//!   optimizer step, then replay for agreement and argmax on the device but compare
//!   on the host, reading the predictions and the labels separately (three reads).
//!   `after` queues the step and the agreement replay and reads both in one go.
//!   Both collect the same kind of window first.
//! * **score** — `before` scores the executed action of a `[envs, classes]` row the
//!   way a DAgger window used to, `log_softmax` then `take_along_last`; `after` is
//!   `Categorical::log_prob_ids`, what it does now. Timed to a flush.
//!
//! ```text
//! cargo run --release --features wgpu --example bench_imitation_round
//! ```
//!
//! `MAMBA3_PPO_ENVS`, `MAMBA3_PPO_WINDOW`, `MAMBA3_PPO_DMODEL` and
//! `MAMBA3_PPO_LAYERS` override the shape, matching `profile_imitation`.

use std::time::{Duration, Instant};

use mamba3::autograd::Var;
use mamba3::backend::{check_launches, read_count, reset_read_count};
use mamba3::distributions::categorical::Categorical;
use mamba3::prelude::*;
use mamba3::rl::{
    BehaviourCloningTask, ImitationBatch, Mamba3Policy, Mamba3PolicyConfig, RecallEnv,
};
use mamba3::tensor::ops::index::{IdTensor, read_all};
use mamba3::tensor::ops::reduce;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

const SYMBOLS: usize = 4;
const HORIZON: usize = 8;
const ROUNDS: usize = 30;
const SCORES: usize = 400;
const BETA: f32 = 0.5;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// The agreement as `BehaviourCloningTask::agreement` computed it before: argmax on
/// the device, every comparison on the host.
fn host_agreement(policy: &Mamba3Policy<R, f32>, batch: &ImitationBatch<R, f32>) -> Result<f32> {
    let _guard = mamba3::autograd::no_grad();
    let (output, _) = policy.forward(
        &Var::constant(batch.observations.clone()),
        batch.reset.as_ref(),
        batch.initial.as_deref(),
    )?;
    let predicted =
        reduce::argmax(output.logits.tensor(), output.logits.rank() - 1)?.try_to_vec()?;
    let expected = batch.expert_actions.try_to_vec()?;
    let weights = batch.mask.as_ref().map(|m| m.try_to_f32()).transpose()?;
    let (mut hits, mut total) = (0.0f32, 0.0f32);
    for (i, (got, want)) in predicted.iter().zip(&expected).enumerate() {
        let w = weights.as_ref().map_or(1.0, |v| v[i]);
        total += w;
        if got == want {
            hits += w;
        }
    }
    Ok(if total == 0.0 { 0.0 } else { hits / total })
}

/// `p10`, median and mean of a set of timings, in milliseconds.
fn summary(times: &mut [Duration]) -> (f64, f64, f64) {
    times.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mean = times.iter().map(|d| ms(*d)).sum::<f64>() / times.len() as f64;
    (
        ms(times[times.len() / 10]),
        ms(times[times.len() / 2]),
        mean,
    )
}

fn report(label: &str, before: &mut [Duration], after: &mut [Duration], unit: &str) {
    let (b10, b50, bmean) = summary(before);
    let (a10, a50, amean) = summary(after);
    println!("{label}: p10 / median / mean, {unit}");
    println!("  before {b10:>8.3} {b50:>8.3} {bmean:>8.3}");
    println!("  after  {a10:>8.3} {a50:>8.3} {amean:>8.3}");
    println!(
        "  median {:+.1}%, mean {:+.1}%\n",
        100.0 * (a50 / b50 - 1.0),
        100.0 * (amean / bmean - 1.0)
    );
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

    // The `before` variant's kernels; the first untimed rounds below warm the
    // queued variant's.
    for _ in 0..3 {
        collector.collect_with_expert(&mut environment, BETA)?;
        let batch = collector.imitation_batch()?;
        trainer.step(&task, std::slice::from_ref(&batch))?;
        host_agreement(&policy, &batch)?;
    }

    let mut before = Vec::with_capacity(ROUNDS);
    let mut after = Vec::with_capacity(ROUNDS);
    let mut reads = (0, 0);
    for i in 0..ROUNDS * 2 + 6 {
        reset_read_count();
        let started = Instant::now();
        collector.collect_with_expert(&mut environment, BETA)?;
        let batch = collector.imitation_batch()?;
        let queued_first = i % 2 == 1;
        if queued_first {
            let queued = trainer.queue_step(&task, std::slice::from_ref(&batch))?;
            let agreement = task.queue_agreement(&batch)?;
            let scalars: Vec<&Tensor<R, f32>> =
                queued.scalars().chain(agreement.scalars()).collect();
            let (_, values) = read_all(&[], &scalars)?;
            let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
            let (step, agreed) = values.split_at(queued.scalars().count());
            trainer.report_steps(std::slice::from_ref(&queued), step);
            agreement.fraction(agreed);
        } else {
            trainer.step(&task, std::slice::from_ref(&batch))?;
            host_agreement(&policy, &batch)?;
        }
        let elapsed = started.elapsed();
        if i < 6 {
            continue;
        }
        if queued_first {
            after.push(elapsed);
            reads.1 = read_count();
        } else {
            before.push(elapsed);
            reads.0 = read_count();
        }
    }
    report(
        &format!("round (reads: before {}, after {})", reads.0, reads.1),
        &mut before,
        &mut after,
        "ms",
    );

    let logits = Tensor::<R, f32>::from_f32(
        &(0..envs * SYMBOLS)
            .map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.0)
            .collect::<Vec<_>>(),
        vec![envs, SYMBOLS],
        &device,
    )?;
    let actions = IdTensor::<R>::from_slice(
        &(0..envs as u32)
            .map(|i| i % SYMBOLS as u32)
            .collect::<Vec<_>>(),
        vec![envs],
        &device,
    )?;
    let composite = || -> Result<Tensor<R, f32>> {
        Ok(Var::constant(logits.clone())
            .log_softmax(1)?
            .take_along_last(&actions)?
            .into_tensor())
    };
    let fused = || -> Result<Tensor<R, f32>> {
        Ok(Categorical::from_logits(Var::constant(logits.clone()))?
            .log_prob_ids(&actions)?
            .into_tensor())
    };
    let (_, [a]) = mamba3::tensor::ops::index::read_together([], [&composite()?])?;
    let (_, [b]) = mamba3::tensor::ops::index::read_together([], [&fused()?])?;
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("score: largest difference between the two {worst:e}");

    let mut before = Vec::with_capacity(SCORES);
    let mut after = Vec::with_capacity(SCORES);
    for i in 0..SCORES * 2 {
        let started = Instant::now();
        if i % 2 == 0 {
            composite()?;
        } else {
            fused()?;
        }
        check_launches(&device)?;
        let elapsed = started.elapsed() * 1000;
        if i % 2 == 0 {
            before.push(elapsed);
        } else {
            after.push(elapsed);
        }
    }
    report("score one step", &mut before, &mut after, "us");
    Ok(())
}
