//! Matmul routes that reshape the problem before any kernel runs: a broadcast
//! weight folded into one tall product, and split-K for `Xᵀ G` with a long `k`
//! (GPU-like devices only; elsewhere this checks the direct kernel).

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::matmul::{matmul, matmul_tn};

type R = Auto;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, a_t: bool) -> Vec<f32> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = if a_t { a[p * m + i] } else { a[i * k + p] } as f64;
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j] as f64;
            }
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

#[test]
fn broadcast_weight_is_folded_into_one_product() {
    let dev = Device::<R>::default();
    let (batch, m, k, n) = (5usize, 7, 16, 12);
    let a = noise(batch * m * k, 1);
    let w = noise(k * n, 2);
    let at = Tensor::<R, f32>::from_f32(&a, vec![batch, m, k], &dev).unwrap();
    let wt = Tensor::<R, f32>::from_f32(&w, vec![k, n], &dev).unwrap();
    let got = matmul(&at, &wt).unwrap();
    assert_eq!(got.dims(), &[batch, m, n]);
    let want = host_matmul(&a, &w, batch * m, k, n, false);
    assert_close(&got.to_f32(), &want, 1e-4, "folded");
}

#[test]
fn tuned_projection_shapes_match_host() {
    // The entity model's projection shapes through the broadcast-weight path of
    // `matmul`: a 3-D lhs `[2, m/2, k]` against a 2-D weight `[k, n]`, folded
    // into one tall product. The tuner must never return a plan that disagrees
    // with the simple kernel here — on Metal a dropped launch (e.g. the 512-unit
    // 128x128 block shape) used to time as the fastest and win, so every later
    // product of that shape returned stale memory. Three calls: the first tunes
    // the shape, the rest reuse the cached plan, and all three must agree with
    // the host reference.
    let dev = Device::<R>::default();
    for (m, n, k) in [
        (26usize, 80usize, 8usize),
        (32, 80, 8),
        (18, 8, 16),
        (128, 648, 128),
    ] {
        let (batch, rows) = (2usize, m / 2);
        assert_eq!(batch * rows, m);
        let a = noise(batch * rows * k, 100 + m as u64 * 1000 + k as u64);
        let w = noise(k * n, 200 + n as u64 * 1000 + k as u64);
        let at = Tensor::<R, f32>::from_f32(&a, vec![batch, rows, k], &dev).unwrap();
        let wt = Tensor::<R, f32>::from_f32(&w, vec![k, n], &dev).unwrap();
        let want = host_matmul(&a, &w, batch * rows, k, n, false);
        for call in 0..3 {
            let got = matmul(&at, &wt).unwrap();
            assert_eq!(got.dims(), &[batch, rows, n]);
            assert_close(
                &got.to_f32(),
                &want,
                1e-4,
                &format!("m={m} n={n} k={k} call={call}"),
            );
        }
    }
}

#[test]
fn long_k_transposed_left_matches_host() {
    let dev = Device::<R>::default();
    // k = 8192 splits (and 6000 splits unevenly-divisible candidates away).
    for (k, m, n) in [(8192usize, 24usize, 40usize), (6000, 16, 8)] {
        let a = noise(k * m, 3);
        let b = noise(k * n, 4);
        let at = Tensor::<R, f32>::from_f32(&a, vec![k, m], &dev).unwrap();
        let bt = Tensor::<R, f32>::from_f32(&b, vec![k, n], &dev).unwrap();
        let got = matmul_tn(&at, &bt).unwrap();
        assert_eq!(got.dims(), &[m, n]);
        let want = host_matmul(&a, &b, m, k, n, true);
        assert_close(&got.to_f32(), &want, 2e-3, &format!("split-k k={k}"));
    }
}
