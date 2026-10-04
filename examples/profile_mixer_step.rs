//! Where one incremental `Mamba3Mixer::step` spends its launches and its time.
//!
//! `bench_rollout` times a whole policy step; this isolates a single block at
//! the same shape (`B = 64`, `D = 256`, `N = 16`) and attributes every launch
//! — count and, with the queue drained between launches, wall time — to the
//! site that issued it.
//!
//! ```text
//! cargo run --release --example profile_mixer_step
//! ```
//!
//! `MAMBA3_STEP_ENVS`, `MAMBA3_STEP_DMODEL`, `MAMBA3_STEP_HEADS`,
//! `MAMBA3_STEP_STATE` and `MAMBA3_STEP_ROTATIONAL=0` override the shape.

use std::time::Instant;

use mamba3::backend::{
    flush_launch_timer, launch_count, launch_tally, launch_time_tally, reset_launch_count,
    reset_launch_tally, set_launch_timer, start_launch_tally, stop_launch_tally,
};
use mamba3::prelude::*;
use mamba3::tensor::ops::random::Rng;

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::try_set_kernel_from_env()?;
    let device = Device::<R>::default();
    let envs = env_usize("MAMBA3_STEP_ENVS", 64);
    let d_model = env_usize("MAMBA3_STEP_DMODEL", 256);
    let heads = env_usize("MAMBA3_STEP_HEADS", 4);
    let state = env_usize("MAMBA3_STEP_STATE", 16);
    let iterations = env_usize("MAMBA3_STEP_ITERS", 400);

    let mut ssm = SsmConfig {
        d_model,
        n_heads: heads,
        head_dim: d_model / heads,
        n_groups: heads,
        d_state: state,
        ..SsmConfig::default()
    };
    if env_usize("MAMBA3_STEP_ROTATIONAL", 1) == 0 {
        ssm.dynamics = StateDynamics::Real;
    }
    let mut rng = Rng::seeded(1);
    let block = Mamba3BlockConfig::new(ssm).init::<R, f32>(&device, &mut rng)?;
    println!("backend: {}", device.name());
    println!("block:   {:?}, {envs} envs", block.mixer());

    let input = Var::constant(Tensor::<R, f32>::full(vec![envs, 1, d_model], 0.1, &device));
    let done = Tensor::<R, f32>::zeros(vec![envs], &device);
    let _guard = mamba3::autograd::no_grad();

    let mut cache = block.empty_cache(envs, &device);
    for _ in 0..32 {
        cache = block.step_masked(&input, &cache, Some(&done))?.1;
    }
    device.synchronize();

    // Pipelined: the number a rollout loop sees.
    reset_launch_count();
    let t = Instant::now();
    for _ in 0..iterations {
        cache = block.step_masked(&input, &cache, Some(&done))?.1;
    }
    device.synchronize();
    let per_step = t.elapsed() / iterations as u32;
    println!(
        "\nblock step: {per_step:.2?} per step, {} launches",
        launch_count() / iterations
    );

    // Drained between launches: the split between sites.
    reset_launch_tally();
    start_launch_tally();
    let dev = device.clone();
    let probe = Tensor::<R, f32>::zeros(vec![1], &device);
    set_launch_timer(Some(Box::new(move || {
        dev.synchronize();
        let _ = probe.to_data();
    })));
    let timed = 50;
    for _ in 0..timed {
        cache = block.step_masked(&input, &cache, Some(&done))?.1;
    }
    flush_launch_timer();
    stop_launch_tally();
    let counts: std::collections::HashMap<String, usize> = launch_tally().into_iter().collect();
    println!("\n{:>9}  {:>6}  site", "us/step", "/step");
    let mut total = 0.0;
    for (site, ms) in launch_time_tally() {
        total += ms;
        println!(
            "{:>9.1}  {:>6.1}  {site}",
            ms * 1e3 / timed as f64,
            counts.get(&site).copied().unwrap_or(0) as f64 / timed as f64
        );
    }
    println!("{:>9.1}  total (serialised)", total * 1e3 / timed as f64);
    set_launch_timer(None);
    Ok(())
}
