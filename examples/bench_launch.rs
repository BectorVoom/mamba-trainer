//! Per-launch cost of this backend: tiny and mid-size fills and adds, pipelined
//! (400 calls back to back, one drain), best of three rounds in microseconds.
//! Written to compare HIP, CUDA and wgpu on the same shapes.
use std::time::Instant;
use mamba3::prelude::*;
use mamba3::tensor::ops::elemwise::{add, fill_};
type R = mamba3::backends::Auto;
fn time(label: &str, device: &Device<R>, iters: usize, mut f: impl FnMut()) {
    for _ in 0..5 {
        f();
    }
    let probe = Tensor::<R, f32>::zeros(vec![1], device);
    let _ = probe.to_data();
    let mut rounds = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let _ = probe.to_data();
        rounds.push(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    let best = rounds.iter().cloned().fold(f64::MAX, f64::min);
    println!("{label:<34} {best:>8.1} us/call best, rounds {rounds:.0?}");
}
fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend {}", device.name());
    for &n in &[1024usize, 65_536, 1 << 20, 1 << 22] {
        let a = Tensor::<R, f32>::zeros(vec![n], &device);
        let b = Tensor::<R, f32>::zeros(vec![n], &device);
        time(&format!("fill {} KB", n * 4 / 1024), &device, 400, || fill_(&a, 1.0));
        time(&format!("add  {} KB (alloc out)", n * 4 / 1024), &device, 400, || {
            let _ = add(&a, &b).unwrap();
        });
    }
    Ok(())
}
