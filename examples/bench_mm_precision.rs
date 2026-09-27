//! The entity model step's matmul shapes under each matmul precision mode.
//!
//! Prints whether the matrix-core (CMMA) path is reachable per mode and the
//! pipelined time of every shape (the tuner picks the kernel, CMMA included), so
//! `MatmulPrecision::Bf16` / `F16` can be judged on the step's real products.
//!
//! ```text
//! cargo run --release --no-default-features --features vulkan --example bench_mm_precision
//! ```

use std::time::Instant;

use mamba3::prelude::*;
use mamba3::tensor::ops::matmul::{MatmulPrecision, cmma_available, matmul_3d_t, set_matmul_precision};

type R = mamba3::backends::Auto;

fn main() -> Result<()> {
    let device = Device::<R>::default();
    // (m, n, k, lhs_t, rhs_t): the decoder and context in/out projections and
    // their adjoints at batch 128.
    let shapes = [
        (20480, 648, 128, false, false),
        (20480, 128, 648, false, true),
        (128, 648, 20480, true, false),
        (20480, 128, 256, false, false),
        (20480, 256, 128, false, true),
        (256, 128, 20480, true, false),
        (12800, 1296, 128, false, false),
        (12800, 128, 1296, false, true),
        (128, 1296, 12800, true, false),
    ];
    for (name, mode) in [
        ("f32", MatmulPrecision::F32),
        ("bf16", MatmulPrecision::Bf16),
        ("f16", MatmulPrecision::F16),
    ] {
        if !mamba3::tensor::ops::matmul::supports_matmul_precision(&device, mode) {
            println!("== {name}: not supported on this backend");
            continue;
        }
        println!("== {name}: cmma reachable {}", cmma_available(&device, mode));
        set_matmul_precision(mode);
        let mut total = 0.0;
        for &(m, n, k, lt, rt) in &shapes {
            let a = Tensor::<R, f32>::from_f32(&vec![0.01; m * k], vec![m * k], &device)?;
            let b = Tensor::<R, f32>::from_f32(&vec![0.02; k * n], vec![k * n], &device)?;
            for _ in 0..3 {
                let _ = matmul_3d_t(&a, &b, 1, m, n, k, 0, 0, lt, rt);
            }
            let _ = matmul_3d_t(&a, &b, 1, m, n, k, 0, 0, lt, rt).to_f32();
            let iters = 20;
            let t = Instant::now();
            let mut last = None;
            for _ in 0..iters {
                last = Some(matmul_3d_t(&a, &b, 1, m, n, k, 0, 0, lt, rt));
            }
            let _ = last.map(|o| o.to_f32());
            let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            total += ms;
            let tflops = 2.0 * (m * n * k) as f64 / (ms * 1e-3) / 1e12;
            println!("  m={m:>5} n={n:>5} k={k:>5} lt={lt:<5} rt={rt:<5} {ms:7.3} ms  {tflops:5.2} TFLOP/s");
        }
        println!("  total {total:.2} ms");
    }
    Ok(())
}
