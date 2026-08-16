//! Matmul throughput at the shapes a Mamba-3 training step actually issues.
//!
//! Square 1024³ is the shape everyone quotes and none of the shapes this crate runs.
//! What a training step issues is a handful of very wide, short-`k` products (the
//! fused input projection and its two adjoints) and a large batch of tiny ones (the
//! scan's per-chunk, per-head products). A kernel that wins one can lose the other,
//! so both are measured, for every kernel, in GFLOP/s.
//!
//! ```text
//! cargo run --release --no-default-features --features hip --example bench_matmul
//! ```

use std::time::Instant;

use mamba3::prelude::*;
use mamba3::tensor::ops::matmul::{MatmulKernel, matmul_3d, set_default_kernel};

type R = mamba3::backends::Auto;

struct Case {
    name: &'static str,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend {}", device.name());

    // d_model 512, in_proj width 4640, d_inner 1024, vocab 8192, 2048 tokens,
    // and the scan's 4 * 8 chunks * 16 heads of 64x64x64.
    let cases = [
        Case { name: "square 1024",   batch: 1,   m: 1024, n: 1024, k: 1024 },
        Case { name: "in_proj  fwd",  batch: 1,   m: 2048, n: 4640, k: 512 },
        Case { name: "in_proj  dW",   batch: 1,   m: 512,  n: 4640, k: 2048 },
        Case { name: "in_proj  dX",   batch: 1,   m: 2048, n: 512,  k: 4640 },
        Case { name: "out_proj fwd",  batch: 1,   m: 2048, n: 512,  k: 1024 },
        Case { name: "lm_head  fwd",  batch: 1,   m: 2048, n: 8192, k: 512 },
        Case { name: "scan  chunks",  batch: 512, m: 64,   n: 64,   k: 64 },
        Case { name: "scan  states",  batch: 512, m: 64,   n: 64,   k: 64 },
    ];

    let kernels = [
        // `Auto` is the tuned path: it measures the candidates once per shape and
        // caches the winner, so this column should track the best of the others.
        ("Auto", MatmulKernel::Auto),
        ("Simple", MatmulKernel::Simple),
        ("RowTiled", MatmulKernel::RowTiled),
        ("Tiled", MatmulKernel::Tiled),
        ("BlockTiled", MatmulKernel::BlockTiled),
    ];

    print!("{:<14}", "case");
    for (name, _) in &kernels {
        print!("{name:>12}");
    }
    println!("   (GFLOP/s)");

    for case in &cases {
        let lhs = Tensor::<R, f32>::zeros(vec![case.batch, case.m, case.k], &device);
        let rhs = Tensor::<R, f32>::zeros(vec![case.batch, case.n, case.k], &device)
            .reshape(vec![case.batch, case.k, case.n])?;
        let flops = 2.0 * (case.batch * case.m * case.n * case.k) as f64;
        print!("{:<14}", case.name);
        for (_, kernel) in &kernels {
            set_default_kernel(*kernel);
            let run = || {
                matmul_3d(
                    &lhs,
                    &rhs,
                    case.batch,
                    case.m,
                    case.n,
                    case.k,
                    case.m * case.k,
                    case.k * case.n,
                )
            };
            for _ in 0..2 {
                let _ = run();
            }
            device.synchronize();
            // Best of many, not the mean: the machine this runs on shares its memory
            // bandwidth between the GPU and everything else, and a mean is mostly a
            // measurement of the rest of the machine.
            let mut best = f64::INFINITY;
            for _ in 0..12 {
                let t = Instant::now();
                let _ = run();
                device.synchronize();
                best = best.min(t.elapsed().as_secs_f64());
            }
            print!("{:>12.0}", flops / best / 1e9);
        }
        println!();
    }
    set_default_kernel(MatmulKernel::Auto);
    Ok(())
}
