//! Time the fused SSD scan (forward and backward) against the chunked scan at the
//! entity model's decoder shape. `MAMBA3_SCAN_SHAPE=b,t,h,p,n` overrides the shape.

use std::time::Instant;

use mamba3::prelude::*;
use mamba3::ssm::scan::ssd_chunked;
use mamba3::tensor::ops::ssd_scan::{ScanDecay, ssd_scan, ssd_scan_backward, ssd_scan_saving};

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
    // Pipelined: `iters` calls back to back and one drain, three rounds. This is
    // how the calls run inside a training step; a min over single drained calls
    // catches the GPU at boost clock and overstates the kernel badly.
    let mut rounds = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let _ = probe.to_data();
        rounds.push(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
    }
    println!("{label:<24} {:>8.3} ms/call (rounds {:.2?})", rounds.iter().sum::<f64>() / 3.0, rounds);
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let dims: Vec<usize> = std::env::var("MAMBA3_SCAN_SHAPE")
        .unwrap_or_else(|_| "128,160,4,64,32".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let (b, t, h, p, n) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let hw = &device.client().properties().hardware;
    println!(
        "backend {} plane {}..{} shape b={b} t={t} h={h} p={p} n={n}",
        device.name(),
        hw.plane_size_min,
        hw.plane_size_max
    );
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
    let dy = mk(vec![b, t, h, p], 7, -1.0, 1.0)?;
    let iters = 20;
    time("fused forward", &device, iters, || {
        ssd_scan(&x, &bb, &c, ScanDecay::Log(&a), &g, &w, None, None).unwrap();
    });
    time("fused backward", &device, iters, || {
        ssd_scan_backward(&dy, &x, &bb, &c, ScanDecay::Log(&a), &g, &w, None, None, None).unwrap();
    });
    let (_, saved) =
        ssd_scan_saving(&x, &bb, &c, ScanDecay::Log(&a), &g, &w, None, None, true).unwrap();
    time("fused forward, saving", &device, iters, || {
        ssd_scan_saving(&x, &bb, &c, ScanDecay::Log(&a), &g, &w, None, None, true).unwrap();
    });
    time("fused backward, saved", &device, iters, || {
        ssd_scan_backward(&dy, &x, &bb, &c, ScanDecay::Log(&a), &g, &w, None, None, saved.as_ref())
            .unwrap();
    });
    if std::env::var_os("MAMBA3_SCAN_CHUNKED").is_some() {
        let v = |t: &Tensor<R, f32>| Var::traced(t.clone());
        time("chunked fwd+bwd", &device, iters, || {
            let (y, _) = ssd_chunked(&v(&x), &v(&bb), &v(&c), &v(&a), &v(&g), &v(&w), None, 40, true)
                .unwrap();
            y.mul(&Var::constant(dy.clone())).unwrap().sum().unwrap().backward().unwrap();
        });
    }
    Ok(())
}
