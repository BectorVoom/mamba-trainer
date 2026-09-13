//! Where a rollout step's dispatches actually go.
//!
//! `profile_ppo` says collection is about half a PPO round's launches. It does not
//! say which operation issues them, and at ~86 dispatches for one step of a
//! two-layer policy the answer is not obvious from reading the step: most of them
//! are not the scan, and several are shape moves that compute nothing.
//!
//! This attributes every launch to the source line that issued it, for one rollout
//! step and for one whole collected window, so a fusion has a measured target
//! rather than a plausible one. Launch counts are used throughout instead of wall
//! time because they are deterministic — run-to-run wall noise on this machine is
//! ±20%, which hides anything smaller than a large win.
//!
//! ```text
//! cargo run --release --features wgpu --example profile_rollout
//! ```
//!
//! `MAMBA3_PPO_ENVS`, `MAMBA3_PPO_WINDOW`, `MAMBA3_PPO_DMODEL` and
//! `MAMBA3_PPO_LAYERS` override the shape, matching `profile_ppo` so the two are
//! directly comparable.

use mamba3::backend::{
    launch_count, launch_tally, reset_launch_count, reset_launch_tally, start_launch_tally,
    stop_launch_tally,
};
use mamba3::prelude::*;
use mamba3::rl::{Mamba3Policy, Mamba3PolicyConfig, RecallEnv};

type R = mamba3::backends::Auto;

const SYMBOLS: usize = 4;
const HORIZON: usize = 8;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Run `body` with a fresh tally and print the sites it charged, densest first.
///
/// `scale` divides the counts, so a window of `T` steps reports per step and the
/// two tables below can be read against each other.
fn attribute(label: &str, scale: usize, body: impl FnOnce() -> Result<()>) -> Result<()> {
    reset_launch_count();
    reset_launch_tally();
    start_launch_tally();
    body()?;
    stop_launch_tally();

    let total = launch_count();
    println!("\n{label}: {total} launches ({} per step)\n", total / scale.max(1));
    println!("{:>8}  {:>7}  {}", "launches", "/step", "site");
    for (site, count) in launch_tally().into_iter().take(24) {
        println!(
            "{count:>8}  {:>7.1}  {site}",
            count as f64 / scale.max(1) as f64
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::try_set_precision_from_env::<R>()?;
    let device = Device::<R>::default();

    let envs = env_usize("MAMBA3_PPO_ENVS", 32);
    let window = env_usize("MAMBA3_PPO_WINDOW", HORIZON * 4);
    let d_model = env_usize("MAMBA3_PPO_DMODEL", 64);
    let layers = env_usize("MAMBA3_PPO_LAYERS", 2);

    println!("backend: {}", device.name());
    println!("shape:   {envs} envs x {window} steps, d_model {d_model}, {layers} layers");

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

    let mut collector = Collector::new(&policy, envs, window, obs_dim, &device)?.with_seed(5);

    // Warm-up. The first window compiles every kernel shape the loop will use and
    // grows the allocator's pools; tallying it would count the compiler's launches
    // alongside the loop's.
    collector.collect(&mut environment)?;
    device.synchronize();

    // The policy step alone, off the collection loop: what one observation costs in
    // dispatches before the draw, the environment and the buffer write are added.
    {
        let mut engine = mamba3::rl::RolloutEngine::new(&policy, envs, &device);
        let obs = Var::constant(Tensor::<R, f32>::zeros(vec![envs, 1, obs_dim], &device));
        let done = Tensor::<R, f32>::zeros(vec![envs], &device);
        for _ in 0..8 {
            engine.step(&obs, Some(&done))?;
        }
        device.synchronize();
        attribute("policy step", 1, || {
            engine.step(&obs, Some(&done))?;
            Ok(())
        })?;
    }

    // The whole window, which is the number `profile_ppo` charges to `collect`.
    attribute("collected window", window, || {
        collector.collect(&mut environment)?;
        Ok(())
    })?;
    device.synchronize();

    Ok(())
}
