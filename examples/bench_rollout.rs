//! Rollout step latency, at the shape a reinforcement learning loop runs at.
//!
//! The target the RL engines are designed against is a sub-millisecond step for
//! `B = 64` environments at `D = 256`, `N = 16` — fast enough that the policy is
//! not what the environment waits for. This measures that, and prints the two
//! numbers that explain whatever it finds:
//!
//! * **dispatches per step** — at this size no backend is arithmetic-bound. A
//!   step moves a few megabytes and does a few megaflops; what it costs is the
//!   fixed price of each kernel launch (roughly 13 us on the CPU runtime, 9 us on
//!   wgpu), multiplied by this number. It is the number to attack.
//! * **reserved bytes** — flat, or the loop cannot run to a million steps.
//!
//! Run it against whichever backend you built for:
//!
//! ```text
//! cargo run --release --example bench_rollout                        # CPU
//! cargo run --release --no-default-features --features msl   --example bench_rollout
//! cargo run --release --no-default-features --features cuda  --example bench_rollout
//! ```

use std::time::Instant;

use mamba3::backend::{launch_count, read_count, reserved_bytes, reset_launch_count,
    reset_read_count};
use mamba3::prelude::*;
use mamba3::rl::{Mamba3PolicyConfig, RolloutEngine};
use mamba3::ssm::scan::{SsmState, mamba3_step};

type R = mamba3::backends::Auto;

/// The acceptance target, in microseconds.
const TARGET_US: u128 = 1_000;

/// The recurrence on its own, at the same shape.
///
/// Worth separating from the policy step below, because they answer different
/// questions. This is the rollout kernel proper — read the state, apply the
/// reset, advance it, read it out — and it is a handful of dispatches. The policy
/// step is that plus a whole Mamba-3 layer's projection, convolution,
/// normalisation and gate, four times over, which is where the dispatch count
/// actually goes.
fn bare_recurrence(device: &Device<R>, envs: usize, heads: usize, head_dim: usize, state: usize) {
    let z = |d: Vec<usize>| Var::constant(Tensor::<R, f32>::zeros(d, device));
    let (x, b, c) = (
        z(vec![envs, heads, head_dim, 1]),
        z(vec![envs, heads, state, 1]),
        z(vec![envs, heads, state, 1]),
    );
    let dt = Var::constant(Tensor::<R, f32>::full(vec![envs, heads], 0.1, device));
    let lambda = Var::constant(Tensor::<R, f32>::full(vec![envs, heads], 0.5, device));
    let a_log = z(vec![heads]);
    let theta = z(vec![envs, heads, state / 2]);
    let done = Tensor::<R, f32>::zeros(vec![envs], device);

    for (label, rotational) in [("real", false), ("rotational", true)] {
        let mut st = SsmState::zeros(envs, heads, head_dim, state, rotational, device);
        let th = rotational.then(|| theta.clone());
        let run = |st: &SsmState<R, f32>| {
            mamba3_step(&x, &b, &c, &dt, &lambda, &a_log, th.as_ref(), None, st, Some(&done))
                .expect("shapes are consistent")
                .1
        };
        for _ in 0..20 {
            st = run(&st);
        }
        device.synchronize();
        reset_launch_count();
        let iterations = 200;
        let t = Instant::now();
        for _ in 0..iterations {
            st = run(&st);
        }
        device.synchronize();
        let per_step = t.elapsed() / iterations;
        println!(
            "  recurrence, {label:<11} {per_step:>10.2?} per step   {:>4} dispatches   {}",
            launch_count() / iterations as usize,
            if per_step.as_micros() <= TARGET_US { "under 1 ms" } else { "OVER 1 ms" },
        );
    }
}

fn main() -> Result<()> {
    let device = Device::<R>::default();

    // B = 64 environments, D_inner = 256, N = 16.
    let (envs, obs_dim, actions, d_model, layers) = (64usize, 32usize, 6usize, 256usize, 4usize);
    let policy = Mamba3PolicyConfig::new(obs_dim, actions, d_model, layers)
        .with_seed(1)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.head_dim = 64;
            s.n_groups = 4;
            s.d_state = 16;
        })
        .init::<R, f32>(&device)?;

    let mut engine = RolloutEngine::new(&policy, envs, &device);
    println!("policy   {policy:?}");
    println!("state    {:?}", engine.state());
    println!(
        "         {} KiB, fixed for the life of the loop\n",
        engine.state().bytes() / 1024
    );

    // Observations and termination flags stay on the device: the point of the
    // exercise is that a step is a queue of launches with nothing crossing the
    // bus. A real loop writes into these buffers from a device-side environment.
    let obs = Var::constant(Tensor::zeros(vec![envs, 1, obs_dim], &device));
    let none_done = Tensor::<R, f32>::zeros(vec![envs], &device);
    let all_done = Tensor::<R, f32>::ones(vec![envs], &device);

    // Warm up: compile the kernels and let the allocator's pools reach the size
    // one step needs.
    for _ in 0..32 {
        engine.step(&obs, Some(&none_done))?;
    }
    device.synchronize();

    println!("B = {envs}, D_inner = {d_model}, N = 16\n");
    bare_recurrence(&device, envs, 4, 64, 16);
    println!();

    let baseline = reserved_bytes(&device);
    for (label, done) in [("no resets", &none_done), ("all reset", &all_done)] {
        let iterations = 200;
        reset_launch_count();
        reset_read_count();
        let t = Instant::now();
        for _ in 0..iterations {
            engine.step(&obs, Some(done))?;
        }
        device.synchronize();
        let elapsed = t.elapsed();

        let per_step = elapsed / iterations;
        let micros = per_step.as_micros();
        println!(
            "  policy, {label:<13} {per_step:>10.2?} per step   {:>4} dispatches   {} reads   {}",
            launch_count() / iterations as usize,
            read_count(),
            if micros <= TARGET_US { "under 1 ms" } else { "OVER 1 ms" },
        );
    }

    device.synchronize();
    if let (Some(before), Some(after)) = (baseline, reserved_bytes(&device)) {
        println!(
            "\nreserved  {} KiB before, {} KiB after 400 steps ({})",
            before / 1024,
            after / 1024,
            if before == after { "flat" } else { "GREW" },
        );
    }

    // What the same trajectory costs through the training engine, for scale: one
    // scan over T steps against T rollout steps. The scan is the shape the policy
    // gradient is taken at, and it is where the parallelism is.
    let horizon = 128;
    let window = Var::constant(Tensor::zeros(vec![envs, horizon, obs_dim], &device));
    let resets = Tensor::<R, f32>::zeros(vec![envs, horizon], &device);
    policy.forward(&window, Some(&resets), None)?;
    device.synchronize();
    reset_launch_count();
    let t = Instant::now();
    policy.forward(&window, Some(&resets), None)?;
    device.synchronize();
    println!(
        "\nscan over T={horizon}   {:>10.2?} total, {:>8.2?} per step, {} dispatches",
        t.elapsed(),
        t.elapsed() / horizon as u32,
        launch_count(),
    );

    Ok(())
}
