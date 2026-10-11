//! Allocation cost probe: a 52 MB tensor allocated per call against one reused
//! and against the scratch arena, then the checkpointing scan with and without
//! the arena. `MAMBA3_SCAN_SHAPE=b,t,h,p,n` as in bench_ssd_scan.
use std::time::Instant;
use mamba3::backend::{ScratchArena, with_scratch};
use mamba3::prelude::*;
use mamba3::tensor::ops::elemwise::fill_;
use mamba3::tensor::ops::ssd_scan::{ScanDecay, ssd_scan_saving};
type R = mamba3::backends::Auto;
fn frand(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 11) as f32 / (u64::MAX >> 11) as f32)
        })
        .collect()
}
fn time(label: &str, device: &Device<R>, iters: usize, mut f: impl FnMut()) {
    for _ in 0..2 {
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
        rounds.push(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
    }
    println!("{label:<40} {:>8.3} ms/call (rounds {rounds:.2?})", rounds.iter().sum::<f64>() / 3.0);
}
fn main() -> Result<()> {
    let device = Device::<R>::default();
    let dims: Vec<usize> = std::env::var("MAMBA3_SCAN_SHAPE")
        .unwrap_or_else(|_| "128,67,10,64,32".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let (b, t, h, p, n) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    println!("backend {}", device.name());
    let iters = 20;
    for &mb in &[8usize, 52, 72] {
        let elems = mb * 1_000_000 / 4;
        time(&format!("alloc + fill + drop {mb} MB"), &device, iters, || {
            let x = Tensor::<R, f32>::empty(vec![elems], &device);
            fill_(&x, 1.0);
        });
        let keep = Tensor::<R, f32>::empty(vec![elems], &device);
        time(&format!("reuse + fill {mb} MB"), &device, iters, || fill_(&keep, 1.0));
        let arena = ScratchArena::new(1 << 30);
        time(&format!("arena: alloc + fill + drop {mb} MB"), &device, iters, || {
            with_scratch(&arena, || {
                let x = Tensor::<R, f32>::empty(vec![elems], &device);
                fill_(&x, 1.0);
            });
        });
    }
    let mk = |shape: Vec<usize>, seed: u64, lo: f32, hi: f32| {
        let len = shape.iter().product();
        Tensor::<R, f32>::from_f32(&frand(len, seed, lo, hi), shape, &device)
    };
    let x = mk(vec![b, t, h, p], 1, -1.0, 1.0)?;
    let bb = mk(vec![b, t, h, n], 2, -1.0, 1.0)?;
    let c = mk(vec![b, t, h, n], 3, -1.0, 1.0)?;
    let a = mk(vec![b, t, h], 4, -0.5, -0.01)?;
    let g = mk(vec![b, t, h], 5, -1.0, 1.0)?;
    let w = mk(vec![b, t, h], 6, -1.0, 1.0)?;
    time("scan forward, saving", &device, iters, || {
        ssd_scan_saving(&x, &bb, &c, ScanDecay::Log(&a), &g, &w, None, None, true).unwrap();
    });
    time("scan forward, no saving", &device, iters, || {
        ssd_scan_saving(&x, &bb, &c, ScanDecay::Log(&a), &g, &w, None, None, false).unwrap();
    });
    Ok(())
}
