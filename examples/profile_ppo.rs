//! Where a PPO round's time actually goes.
//!
//! A round is five distinguishable pieces of work — collect the window, build the
//! batch, replay it through the scan, score it, and take the gradient step — and
//! optimising the wrong one is the usual way to spend a day for nothing. This
//! prints wall time, kernel launches and host reads for each, so the next change
//! is aimed at the piece that is actually large.
//!
//! Launches are printed beside the time because at this size they *are* the time:
//! every backend here charges a fixed price per dispatch (~9 us on wgpu), and a
//! phase issuing 400 launches over `[32, 16]` tensors is paying dispatch, not
//! arithmetic. A phase whose time falls when its launch count does was
//! launch-bound; one whose time does not was not.
//!
//! ```text
//! cargo run --release --features wgpu --example profile_ppo
//! ```
//!
//! `MAMBA3_PPO_ENVS`, `MAMBA3_PPO_WINDOW`, `MAMBA3_PPO_DMODEL` and
//! `MAMBA3_PPO_LAYERS` override the shape.

use std::time::Instant;

use mamba3::autograd::Var;
use mamba3::backend::{launch_count, read_count, reset_launch_count, reset_read_count};
use mamba3::prelude::*;
use mamba3::rl::{Mamba3Policy, PpoBatch, PpoTask, RecallEnv, ppo_objective};
use mamba3::train::{AdamWConfig, Optimizer, grad_scale};

type R = mamba3::backends::Auto;

const SYMBOLS: usize = 4;
const HORIZON: usize = 8;
const EPOCHS: usize = 4;
const ROUNDS: usize = 8;
/// Extra pipelined rounds for an external sampling profiler to have something to see.
const SAMPLE_ROUNDS_KEY: &str = "MAMBA3_PPO_ROUNDS";

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// One phase's cost, accumulated over every round that ran it.
#[derive(Default, Clone, Copy)]
struct Phase {
    nanos: u128,
    launches: usize,
    reads: usize,
    calls: u32,
}

impl Phase {
    fn per_round(&self, rounds: u32) -> (f64, f64, f64) {
        (
            self.nanos as f64 / rounds as f64 / 1e6,
            self.launches as f64 / rounds as f64,
            self.reads as f64 / rounds as f64,
        )
    }
}

/// Run `body`, charging its wall time, launches and reads to `phase`.
///
/// The device is synchronised at both ends. That makes the phases add up to the
/// round — the point of the breakdown — at the cost of serialising work the real
/// loop would overlap, so the total here is an upper bound on the real one.
fn timed<T>(phase: &mut Phase, device: &Device<R>, body: impl FnOnce() -> Result<T>) -> Result<T> {
    let launches = launch_count();
    let reads = read_count();
    let started = Instant::now();
    let out = body()?;
    device.synchronize();
    phase.nanos += started.elapsed().as_nanos();
    phase.launches += launch_count() - launches;
    phase.reads += read_count() - reads;
    phase.calls += 1;
    Ok(out)
}

/// The device, optionally with CubeCL's exclusive-page allocator instead of the
/// default sub-slicing one.
///
/// Worth having because a sampling profile of this loop puts about half its host time
/// inside `client.empty` — every intermediate tensor is an allocation.
/// `MAMBA3_PPO_EXCLUSIVE_PAGES=1` swaps the default sub-slicing pool for the one that
/// hands back a whole page, which asks whether that half is the pool's bookkeeping.
///
/// Measured here, it is not: the two are indistinguishable across ten interleaved
/// runs. What `client.empty` costs is the enqueue onto the device channel and the
/// host-side layout policy, neither of which the preset changes — so the only lever
/// on it is allocating less, which means fusing. The knob stays because that is worth
/// re-checking on a backend whose allocator behaves differently.
#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "hip")))]
fn open_device() -> Device<R> {
    if std::env::var_os("MAMBA3_PPO_EXCLUSIVE_PAGES").is_none() {
        return Device::<R>::default();
    }
    use cubecl::wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, init_device, init_setup};
    let base = WgpuDevice::default();
    let setup = init_setup::<AutoGraphicsApi>(&base, RuntimeOptions::default());
    let tuned = init_device(
        setup,
        RuntimeOptions {
            memory_config: cubecl::MemoryConfiguration::ExclusivePages,
            ..RuntimeOptions::default()
        },
    );
    println!("(allocator: exclusive pages)");
    Device::<R>::new(&tuned)
}

#[cfg(not(all(feature = "wgpu", not(feature = "cuda"), not(feature = "hip"))))]
fn open_device() -> Device<R> {
    Device::<R>::default()
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::try_set_precision_from_env::<R>()?;
    let device = open_device();

    let envs = env_usize("MAMBA3_PPO_ENVS", 32);
    let window = env_usize("MAMBA3_PPO_WINDOW", HORIZON * 4);
    let d_model = env_usize("MAMBA3_PPO_DMODEL", 64);
    let layers = env_usize("MAMBA3_PPO_LAYERS", 2);

    println!("backend: {}", device.name());
    println!(
        "shape:   {envs} envs x {window} steps, d_model {d_model}, {layers} layers, \
         {EPOCHS} epochs/round\n"
    );

    let mut environment = RecallEnv::<R, f32>::new(envs, SYMBOLS, HORIZON, 23, &device)?;
    let obs_dim = environment.obs_dim();
    let policy: Mamba3Policy<R, f32> =
        mamba3::rl::Mamba3PolicyConfig::new(obs_dim, SYMBOLS, d_model, layers)
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

    let config = PpoConfig::default()
        .with_discount(0.99, 0.95)
        .with_clip(0.2)
        .with_coefficients(0.5, 0.01);
    let task = PpoTask::new(&policy, config);
    let parameters = <PpoTask<R, f32> as mamba3::train::TrainStep<R, f32>>::parameters(&task);
    let mut optimizer = AdamWConfig::builder()
        .learning_rate(1e-3)
        .build()
        .init::<R, f32>();
    let mut collector = Collector::new(&policy, envs, window, obs_dim, &device)?.with_seed(5);

    // Warm-up. The first round pays JIT compilation for every kernel shape the
    // round will ever use; timing it would measure the compiler.
    {
        let report = collector.collect(&mut environment)?;
        let batch = collector.ppo_batch(&report, &config)?;
        let loss = task.evaluate(&batch)?;
        let grads = loss.total.backward()?;
        let scale = grad_scale(&grads, 0.5, 1.0)?;
        optimizer.step_scaled(&parameters, &grads, scale.as_ref().map(|s| &s.factor))?;
        device.synchronize();
    }

    let mut collect = Phase::default();
    let mut build = Phase::default();
    let mut forward = Phase::default();
    let mut objective = Phase::default();
    let mut backward = Phase::default();
    let mut update = Phase::default();

    reset_launch_count();
    reset_read_count();
    let round_start = Instant::now();

    for _ in 0..ROUNDS {
        let report = timed(&mut collect, &device, || {
            collector.collect(&mut environment)
        })?;
        let batch: PpoBatch<R, f32> = timed(&mut build, &device, || {
            collector.ppo_batch(&report, &config)
        })?;

        for _ in 0..EPOCHS {
            let output = timed(&mut forward, &device, || {
                let observations = Var::traced(batch.observations.clone());
                policy
                    .forward(
                        &observations,
                        batch.reset.as_ref(),
                        batch.initial.as_deref(),
                    )
                    .map(|(output, _)| output)
            })?;
            let loss = timed(&mut objective, &device, || {
                ppo_objective(&output, &batch, &config)
            })?;
            let grads = timed(&mut backward, &device, || loss.total.backward())?;
            timed(&mut update, &device, || {
                let scale = grad_scale(&grads, 0.5, 1.0)?;
                optimizer.step_scaled(&parameters, &grads, scale.as_ref().map(|s| &s.factor))
            })?;
        }
    }
    device.synchronize();
    let total = round_start.elapsed();
    let rounds = ROUNDS as u32;
    // Read before the pipelined pass below resets the counters.
    let (phase_launches, phase_reads) = (launch_count(), read_count());

    // The same rounds again with nothing synchronised inside them, which is how the
    // real loop runs. Launches are buffered and the queue is only drained once, so the
    // host runs ahead of the device and the two overlap; the breakdown above pays a
    // full drain at every phase boundary and is therefore an upper bound. Optimise
    // against this number and attribute with that one.
    reset_launch_count();
    let pipelined_rounds = env_usize(SAMPLE_ROUNDS_KEY, ROUNDS);
    let pipelined_start = Instant::now();
    for _ in 0..pipelined_rounds {
        let report = collector.collect(&mut environment)?;
        let batch = collector.ppo_batch(&report, &config)?;
        for _ in 0..EPOCHS {
            let observations = Var::traced(batch.observations.clone());
            let (output, _) = policy.forward(
                &observations,
                batch.reset.as_ref(),
                batch.initial.as_deref(),
            )?;
            let loss = ppo_objective(&output, &batch, &config)?;
            let grads = loss.total.backward()?;
            let scale = grad_scale(&grads, 0.5, 1.0)?;
            optimizer.step_scaled(&parameters, &grads, scale.as_ref().map(|s| &s.factor))?;
        }
    }
    // Split the round in two: the time the host spends building and queueing the work,
    // and the time it then waits for the device to finish it. Launches are buffered, so
    // if the first number is most of the round the bottleneck is on this side of the
    // bus — host-side per-op cost — and no kernel will ever be fast enough to hide it.
    let submitted = pipelined_start.elapsed();
    device.synchronize();
    let n = pipelined_rounds as u32;
    let pipelined = pipelined_start.elapsed() / n;
    let submit = submitted / n;
    let pipelined_launches = launch_count() / pipelined_rounds;

    println!(
        "{:<12} {:>10} {:>7} {:>10} {:>8} {:>7}",
        "phase", "ms/round", "share", "launches", "us/each", "reads"
    );
    let whole = total.as_nanos() as f64 / rounds as f64 / 1e6;
    for (name, phase) in [
        ("collect", collect),
        ("batch", build),
        ("forward", forward),
        ("objective", objective),
        ("backward", backward),
        ("update", update),
    ] {
        let (ms, launches, reads) = phase.per_round(rounds);
        let each = if launches > 0.0 {
            ms * 1000.0 / launches
        } else {
            0.0
        };
        println!(
            "{name:<12} {ms:>10.2} {:>6.1}% {launches:>10.0} {each:>8.1} {reads:>7.1}",
            100.0 * ms / whole
        );
    }
    println!(
        "{:<12} {whole:>10.2} {:>6.1}% {:>10.0} {:>8.1} {:>7.1}",
        "round",
        100.0,
        phase_launches as f64 / rounds as f64,
        whole * 1000.0 * rounds as f64 / phase_launches as f64,
        phase_reads as f64 / rounds as f64,
    );
    println!(
        "\npipelined round (no intra-round sync): {:.2} ms, {pipelined_launches} launches, \
         {:.1} us/launch",
        pipelined.as_secs_f64() * 1e3,
        pipelined.as_secs_f64() * 1e6 / pipelined_launches as f64,
    );
    println!(
        "  of which host submission: {:.2} ms ({:.1} us/launch), device drain: {:.2} ms",
        submit.as_secs_f64() * 1e3,
        submit.as_secs_f64() * 1e6 / pipelined_launches as f64,
        (pipelined - submit).as_secs_f64() * 1e3,
    );
    Ok(())
}
