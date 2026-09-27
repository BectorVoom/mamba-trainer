//! K9: time `sum_dim` with the reduction split on and off.
//!
//! ```text
//! cargo run --release --features cpu --example bench_reduce
//! MAMBA3_REDUCE_SPLIT=0 cargo run --release --features cpu --example bench_reduce
//! MAMBA3_REDUCE_SPLIT=1 cargo run --release --features cpu --example bench_reduce
//! ```
//!
//! Prints µs per call (200 calls after warm-up) and launches per call for each
//! shape in §K9: `[15360, 128]` axis 1, `[128, 120, 128]` axis 1,
//! `[7680, 101]` axis 1, plus `[3840, 128]` (the batch-32 form of the first)
//! and `[2048, 1024]` axis 1 (long axis, few outputs: the case that still splits).

use std::time::Instant;

use mamba3::backend::{launch_count, reset_launch_count};
use mamba3::prelude::*;
use mamba3::tensor::ops::reduce::sum_dim;

type R = mamba3::backends::Auto;

fn bench(device: &Device<R>, shape: &[usize], axis: usize) {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01).collect();
    let input = Tensor::<R, f32>::from_f32(&data, shape.to_vec(), device).unwrap();
    // Warm up (compiles both vector widths the kernel may pick).
    for _ in 0..10 {
        let _ = sum_dim(&input, axis).unwrap();
    }
    device.synchronize();
    reset_launch_count();
    let started = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..200 {
        let out = sum_dim(&input, axis).unwrap();
        sink += out.to_f32()[0];
    }
    let us = started.elapsed().as_secs_f64() * 1e6 / 200.0;
    let launches = launch_count() as f64 / 200.0;
    println!(
        "sum_dim{:?} axis={axis}: {us:10.1} us/call  {launches:5.2} launches/call (sink {sink:.3})",
        shape
    );
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend: {}", device.name());
    bench(&device, &[15360, 128], 1);
    bench(&device, &[128, 120, 128], 1);
    bench(&device, &[7680, 101], 1);
    bench(&device, &[3840, 128], 1);
    bench(&device, &[2048, 1024], 1);
    Ok(())
}
