//! Which 16x16 output tiles the matrix-core (CMMA) matmul gets wrong on this
//! device, against the simple kernel on the same inputs (small integers, exact
//! in f16 and bf16). `M`, `N`, `K` override the shape (default 64x64x32, the
//! 64x64 block candidate); `PREC=f16|bf16` picks the fragment type.
//!
//! Written for the HIP WMMA lane-indexing fix in `vendor/cubecl-cpp`: before
//! it, every plane but the first of a multi-plane cube produced garbage.
use mamba3::prelude::*;
use mamba3::tensor::ops::matmul::{
    MatmulKernel, MatmulPrecision, matmul_3d, set_default_kernel, set_matmul_precision,
};
type R = mamba3::backends::Auto;
fn dim(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn main() -> Result<()> {
    let device = Device::<R>::default();
    let (m, n, k) = (dim("M", 64), dim("N", 64), dim("K", 32));
    // Small integers: exact in bf16, so the reference is exact too.
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 17) as f32 - 8.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 + 1) % 13) as f32 - 6.0).collect();
    let ta = Tensor::<R, f32>::from_f32(&a, vec![1, m, k], &device)?;
    let tb = Tensor::<R, f32>::from_f32(&b, vec![1, k, n], &device)?;
    set_matmul_precision(MatmulPrecision::F32);
    set_default_kernel(MatmulKernel::Simple);
    let reference = matmul_3d(&ta, &tb, 1, m, n, k, m * k, k * n).to_f32();
    // `PREC=f16` (default) or `PREC=bf16`: the operand type of the fragments.
    let precision = match std::env::var("PREC").as_deref() {
        Ok("bf16") => MatmulPrecision::Bf16,
        _ => MatmulPrecision::F16,
    };
    set_matmul_precision(precision);
    set_default_kernel(MatmulKernel::Cmma);
    let got = matmul_3d(&ta, &tb, 1, m, n, k, m * k, k * n).to_f32();
    println!("backend {}; m={m} n={n} k={k}: max abs error per 16x16 tile", device.name());
    let mut bad = 0usize;
    for tr in 0..m.div_ceil(16) {
        let mut line = String::new();
        for tc in 0..n.div_ceil(16) {
            let mut worst = 0.0f32;
            for r in tr * 16..((tr + 1) * 16).min(m) {
                for c in tc * 16..((tc + 1) * 16).min(n) {
                    worst = worst.max((got[r * n + c] - reference[r * n + c]).abs());
                }
            }
            if worst > 0.5 {
                bad += 1;
            }
            line.push_str(&format!(" {:>9.3e}", worst));
        }
        println!("{line}");
    }
    println!("{bad} of {} tiles wrong", m.div_ceil(16) * n.div_ceil(16));
    Ok(())
}
