//! Same-shape versus broadcasting elementwise ops, at the same output size.
//!
//! The scan's hot intermediates are `[batch, chunks, chunk, chunk, heads]` tensors
//! built by broadcasting two `[batch, chunks, chunk, heads]` operands against each
//! other. If broadcasting costs the same per output element as a plain elementwise
//! op, there is nothing here. If it does not, the difference is per-call overhead,
//! and it is being paid a few dozen times per layer.
//!
//! ```text
//! cargo run --release --no-default-features --features hip --example bench_elemwise
//! ```

use std::time::Instant;

use mamba3::prelude::*;
use mamba3::tensor::ops::elemwise;

type R = mamba3::backends::Auto;

fn best(f: impl Fn(), device: &Device<R>) -> f64 {
    for _ in 0..3 {
        f();
    }
    device.synchronize();
    let mut best = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        f();
        device.synchronize();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend {}", device.name());

    // The shape the scan actually builds: batch 4, 8 chunks of 64, 16 heads.
    let (b, c, chunk, h) = (4usize, 8, 64, 16);
    let full = vec![b, c, chunk, chunk, h];
    let n = b * c * chunk * chunk * h;

    let x = Tensor::<R, f32>::zeros(full.clone(), &device);
    let y = Tensor::<R, f32>::zeros(full.clone(), &device);
    let row = Tensor::<R, f32>::zeros(vec![b, c, chunk, 1, h], &device);
    let col = Tensor::<R, f32>::zeros(vec![b, c, 1, chunk, h], &device);
    let mask = Tensor::<R, f32>::zeros(vec![1, 1, chunk, chunk, 1], &device);

    println!("{n} output elements ({:.1} MiB)", (n * 4) as f64 / 1048576.0);
    let report = |name: &str, run: &dyn Fn()| {
        let t = best(run, &device);
        println!(
            "{name} {:>8.3} ms   {:>6.1} GB/s of output",
            t * 1e3,
            (n * 4) as f64 / t / 1e9
        );
    };
    report("same shape ", &|| {
        let _ = elemwise::mul(&x, &y);
    });
    report("row x col  ", &|| {
        let _ = elemwise::sub(&row, &col);
    });
    report("full x mask", &|| {
        let _ = elemwise::mul(&x, &mask);
    });
    report("full x full", &|| {
        let _ = elemwise::add(&x, &y);
    });
    Ok(())
}
