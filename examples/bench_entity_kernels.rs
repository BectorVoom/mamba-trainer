//! Device time of each fused entity kernel alone, at the Kaggriculture shape.
//!
//! `bench_entity` answers whether the fused path beats the composed one end to
//! end; this answers which kernel inside it costs what, so a change to one
//! kernel can be judged on its own. Each kernel is launched `MAMBA3_BENCH_BURST`
//! times back to back between two synchronisations (so per-launch host latency
//! is amortised and the figure is dominated by the device work), and the median
//! of `MAMBA3_BENCH_ITERS` such bursts is reported per launch.
//!
//! ```text
//! cargo run --release --example bench_entity_kernels
//! ```
//!
//! `MAMBA3_ENTITY_ROWS` overrides the row count (default 1024, the BC window of
//! 64 envs × 16 steps).

use std::time::{Duration, Instant};

use mamba3::prelude::*;
use mamba3::tensor::ops::entity;
use mamba3::tensor::ops::index::IdTensor;

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Deterministic pseudo-random data in `[-1, 1)`.
fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn time(
    name: &str,
    device: &Device<R>,
    iters: usize,
    burst: usize,
    mut body: impl FnMut() -> Result<()>,
) -> Result<()> {
    for _ in 0..3 {
        body()?;
    }
    device.synchronize();
    let mut samples: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.synchronize();
        let started = Instant::now();
        for _ in 0..burst {
            body()?;
        }
        device.synchronize();
        samples.push(started.elapsed() / burst as u32);
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    println!("{name:<28} {:>10.1} us", median.as_secs_f64() * 1e6);
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    let rows = env_usize("MAMBA3_ENTITY_ROWS", 1024);
    let iters = env_usize("MAMBA3_BENCH_ITERS", 21);
    let burst = env_usize("MAMBA3_BENCH_BURST", 10);
    let (globals, n, f, d, h, kx) = (4usize, 100usize, 60usize, 48usize, 48usize, 1usize);
    let obs_dim = globals + n * (f + 1);
    println!("rows {rows}, N {n}, F {f}, d {d}, H {h}");

    // Presence: roughly two thirds of the slots are present.
    let mut obs = noise(rows * obs_dim, 1);
    for r in 0..rows {
        for s in 0..n {
            obs[r * obs_dim + globals + s * (f + 1) + f] = if (r + s) % 3 == 0 { 0.0 } else { 1.0 };
        }
    }
    let obs = Tensor::<R, f32>::from_f32(&obs, vec![rows, obs_dim], &device)?;
    let (feat, mean_w, legal, any) = entity::entity_prepare(&obs, globals, n, f)?;

    let e = Tensor::<R, f32>::from_f32(&noise(rows * n * d, 2), vec![rows, n, d], &device)?;
    let stride_w = globals + 2 * d;
    let joined = Tensor::<R, f32>::zeros(vec![rows, stride_w], &device);
    let argmax = IdTensor::<R>::from_slice(&vec![0u32; rows * d], vec![rows, d], &device)?;
    let g_joined =
        Tensor::<R, f32>::from_f32(&noise(rows * stride_w, 3), vec![rows, stride_w], &device)?;
    let g_feat = Tensor::<R, f32>::from_f32(&noise(rows * n * f, 4), vec![rows, n, f], &device)?;

    let k = Tensor::<R, f32>::from_f32(&noise(rows * n * h, 5), vec![rows, n, h], &device)?;
    let q = Tensor::<R, f32>::from_f32(&noise(rows * h, 6), vec![rows, h], &device)?;
    let v = Tensor::<R, f32>::from_f32(&noise(h, 7), vec![h], &device)?;
    let qd = Tensor::<R, f32>::from_f32(&noise(rows * d, 8), vec![rows, d], &device)?;
    let extra = Tensor::<R, f32>::from_f32(&noise(rows * kx, 9), vec![rows, kx], &device)?;
    let g_logits =
        Tensor::<R, f32>::from_f32(&noise(rows * (n + kx), 10), vec![rows, n + kx], &device)?;
    let pre = Tensor::<R, f32>::from_f32(&noise(rows * n * 64, 11), vec![rows * n, 64], &device)?;
    let bias = Tensor::<R, f32>::from_f32(&noise(64, 12), vec![64], &device)?;
    let y = entity::bias_relu(&pre, &bias)?;

    time("entity_prepare", &device, iters, burst, || {
        entity::entity_prepare(&obs, globals, n, f).map(|_| ())
    })?;
    time("entity_prepare_backward", &device, iters, burst, || {
        entity::entity_prepare_backward(&g_feat, &obs, globals, n, f).map(|_| ())
    })?;
    time("entity_pool (mean+max+G)", &device, iters, burst, || {
        entity::entity_pool(
            &e,
            &mean_w,
            &legal,
            &any,
            &obs,
            &joined,
            &argmax,
            globals,
            globals + d,
            globals,
            true,
            true,
            true,
        )
    })?;
    time("entity_pool_backward", &device, iters, burst, || {
        entity::entity_pool_backward(
            &g_joined,
            &mean_w,
            &legal,
            &any,
            &argmax,
            n,
            d,
            globals,
            globals + d,
            true,
            true,
        )
        .map(|_| ())
    })?;
    time("pointer_additive", &device, iters, burst, || {
        entity::pointer_additive(&k, &q, &v, &legal, Some(&extra), n).map(|_| ())
    })?;
    time("pointer_additive_bwd_dk", &device, iters, burst, || {
        entity::pointer_additive_backward_dk(&g_logits, &k, &q, &v, &legal, n, kx).map(|_| ())
    })?;
    time("pointer_additive_bwd_dq", &device, iters, burst, || {
        entity::pointer_additive_backward_dq(&g_logits, &k, &q, &v, &legal, n, kx).map(|_| ())
    })?;
    time("pointer_dot", &device, iters, burst, || {
        entity::pointer_dot(&e, &qd, &legal, Some(&extra), n).map(|_| ())
    })?;
    time("pointer_dot_bwd_de", &device, iters, burst, || {
        entity::pointer_dot_backward_de(&g_logits, &e, &qd, &legal, n, kx).map(|_| ())
    })?;
    time("pointer_dot_bwd_dqd", &device, iters, burst, || {
        entity::pointer_dot_backward_dqd(&g_logits, &e, &legal, n, kx).map(|_| ())
    })?;
    time("bias_relu", &device, iters, burst, || {
        entity::bias_relu(&pre, &bias).map(|_| ())
    })?;
    time("bias_relu_backward", &device, iters, burst, || {
        entity::bias_relu_backward(&pre, &y).map(|_| ())
    })?;
    drop(feat);
    Ok(())
}
