//! The fused entity kernels against plain host references at a shape wide
//! enough to exercise their launch geometry.
//!
//! `entity_kernels` holds each kernel to the composed path at small shapes,
//! which run in one unit with narrow vectors. Here the widths divide by 4 and
//! 16, so the kernels read `Vector`s wider than one, and the work is large
//! enough that the CPU runtime deals it out to several units, each starting
//! its span partway through a row. Each reference is the definition written
//! out serially in `f32`.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::entity;
use mamba3::tensor::ops::index::IdTensor;

type R = Auto;

const ROWS: usize = 64;
const G: usize = 4;
const N: usize = 100;
const F: usize = 13;
const D: usize = 48;
const H: usize = 48;
const KX: usize = 3;
const OBS_DIM: usize = G + N * (F + 1);

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Deterministic pseudo-random values in `[-1, 1)`, tie-free in practice.
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

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// Presence per `(row, n)`: mostly 1, every third slot absent, one 0.5, and
/// row 5 entirely absent.
fn presence() -> Vec<f32> {
    let mut p = vec![0.0; ROWS * N];
    for r in 0..ROWS {
        for n in 0..N {
            p[r * N + n] = if r == 5 || (r + n) % 3 == 0 { 0.0 } else { 1.0 };
        }
    }
    p[2 * N + 1] = 0.5;
    p
}

fn observations(p: &[f32]) -> Vec<f32> {
    let mut obs = noise(ROWS * OBS_DIM, 41);
    for r in 0..ROWS {
        for n in 0..N {
            obs[r * OBS_DIM + G + n * (F + 1) + F] = p[r * N + n];
        }
    }
    obs
}

#[test]
fn prepare_and_its_adjoint_match_the_definition_across_units() {
    let device = dev();
    let p = presence();
    let raw = observations(&p);
    let obs = Tensor::<R, f32>::from_f32(&raw, vec![ROWS, OBS_DIM], &device).unwrap();
    let (feat, mean_w, legal, any) = entity::entity_prepare(&obs, G, N, F).unwrap();

    let mut want_feat = vec![0.0; ROWS * N * F];
    let mut want_mean = vec![0.0; ROWS * N];
    let mut want_any = vec![0.0; ROWS];
    for r in 0..ROWS {
        let sum: f32 = (0..N).map(|n| p[r * N + n]).fold(0.0, |a, b| a + b);
        want_any[r] = sum.min(1.0);
        for n in 0..N {
            want_mean[r * N + n] = p[r * N + n] / sum.max(1.0);
            for f in 0..F {
                want_feat[(r * N + n) * F + f] =
                    raw[r * OBS_DIM + G + n * (F + 1) + f] * p[r * N + n];
            }
        }
    }
    assert_close(&feat.to_f32(), &want_feat, 0.0, "features");
    assert_close(&mean_w.to_f32(), &want_mean, 0.0, "mean_w");
    assert_close(&legal.to_f32(), &p, 0.0, "legal");
    assert_close(&any.to_f32(), &want_any, 0.0, "any");

    let g = noise(ROWS * N * F, 42);
    let g_t = Tensor::<R, f32>::from_f32(&g, vec![ROWS, N, F], &device).unwrap();
    let d_obs = entity::entity_prepare_backward(&g_t, &obs, G, N, F).unwrap();
    let mut want = vec![0.0; ROWS * OBS_DIM];
    for r in 0..ROWS {
        for n in 0..N {
            for f in 0..F {
                want[r * OBS_DIM + G + n * (F + 1) + f] = g[(r * N + n) * F + f] * p[r * N + n];
            }
        }
    }
    assert_close(&d_obs.to_f32(), &want, 0.0, "d_obs");
}

#[test]
fn pool_and_its_adjoint_match_the_definition_across_units() {
    let device = dev();
    let p = presence();
    let raw = observations(&p);
    let obs = Tensor::<R, f32>::from_f32(&raw, vec![ROWS, OBS_DIM], &device).unwrap();
    let (_, mean_w, legal, any) = entity::entity_prepare(&obs, G, N, F).unwrap();
    let (mean_h, any_h) = (mean_w.to_f32(), any.to_f32());

    let e_h = noise(ROWS * N * D, 43);
    let e = Tensor::<R, f32>::from_f32(&e_h, vec![ROWS, N, D], &device).unwrap();
    let stride_w = G + 2 * D;
    let joined = Tensor::<R, f32>::zeros(vec![ROWS, stride_w], &device);
    let argmax = IdTensor::<R>::from_slice(&vec![0u32; ROWS * D], vec![ROWS, D], &device).unwrap();
    entity::entity_pool(
        &e,
        &mean_w,
        &legal,
        &any,
        &obs,
        &joined,
        &argmax,
        G,
        G + D,
        G,
        true,
        true,
        true,
    )
    .unwrap();

    let mut want = vec![0.0; ROWS * stride_w];
    let mut want_arg = vec![0u32; ROWS * D];
    for r in 0..ROWS {
        want[r * stride_w..r * stride_w + G].copy_from_slice(&raw[r * OBS_DIM..r * OBS_DIM + G]);
        for j in 0..D {
            let mut acc = 0.0f32;
            let mut best: Option<(f32, u32)> = None;
            for n in 0..N {
                let x = e_h[(r * N + n) * D + j];
                acc += mean_h[r * N + n] * x;
                if p[r * N + n] != 0.0 && best.is_none_or(|(b, _)| x > b) {
                    best = Some((x, n as u32));
                }
            }
            let (b, i) = best.unwrap_or((0.0, 0));
            want[r * stride_w + G + j] = acc;
            want[r * stride_w + G + D + j] = if any_h[r] != 0.0 { b } else { 0.0 };
            want_arg[r * D + j] = i;
        }
    }
    assert_close(&joined.to_f32(), &want, 1e-6, "joined");
    assert_eq!(argmax.to_vec(), want_arg, "argmax");

    let g = noise(ROWS * stride_w, 44);
    let g_t = Tensor::<R, f32>::from_f32(&g, vec![ROWS, stride_w], &device).unwrap();
    let d_e = entity::entity_pool_backward(
        &g_t,
        &mean_w,
        &legal,
        &any,
        &argmax,
        N,
        D,
        G,
        G + D,
        true,
        true,
    )
    .unwrap();
    let mut want = vec![0.0; ROWS * N * D];
    for r in 0..ROWS {
        for n in 0..N {
            for j in 0..D {
                let mut v = g[r * stride_w + G + j] * mean_h[r * N + n];
                if any_h[r] != 0.0 && want_arg[r * D + j] == n as u32 {
                    v += g[r * stride_w + G + D + j] * p[r * N + n];
                }
                want[(r * N + n) * D + j] = v;
            }
        }
    }
    assert_close(&d_e.to_f32(), &want, 1e-6, "d_e");
}

#[test]
fn pointer_scorers_and_adjoints_match_the_definition_across_units() {
    let device = dev();
    let p = presence();
    let legal = Tensor::<R, f32>::from_f32(&p, vec![ROWS, N], &device).unwrap();
    let k_h = noise(ROWS * N * H, 51);
    let q_h = noise(ROWS * H, 52);
    let v_h = noise(H, 53);
    let x_h = noise(ROWS * KX, 54);
    let k = Tensor::<R, f32>::from_f32(&k_h, vec![ROWS, N, H], &device).unwrap();
    let q = Tensor::<R, f32>::from_f32(&q_h, vec![ROWS, H], &device).unwrap();
    let v = Tensor::<R, f32>::from_f32(&v_h, vec![H], &device).unwrap();
    let x = Tensor::<R, f32>::from_f32(&x_h, vec![ROWS, KX], &device).unwrap();
    let cols = N + KX;
    let g = noise(ROWS * cols, 55);
    let g_t = Tensor::<R, f32>::from_f32(&g, vec![ROWS, cols], &device).unwrap();

    // Additive.
    let logits = entity::pointer_additive(&k, &q, &v, &legal, Some(&x), N)
        .unwrap()
        .to_f32();
    let (d_k, d_x) = entity::pointer_additive_backward_dk(&g_t, &k, &q, &v, &legal, N, KX).unwrap();
    let (d_q, dv) = entity::pointer_additive_backward_dq(&g_t, &k, &q, &v, &legal, N, KX).unwrap();
    let mut want = vec![0.0; ROWS * cols];
    let mut want_dk = vec![0.0; ROWS * N * H];
    let mut want_dq = vec![0.0; ROWS * H];
    let mut want_dv = vec![0.0; ROWS * H];
    for r in 0..ROWS {
        for c in 0..N {
            let present = p[r * N + c] != 0.0;
            let gl = g[r * cols + c] * p[r * N + c];
            let mut acc = 0.0f32;
            for h in 0..H {
                let pre = k_h[(r * N + c) * H + h] + q_h[r * H + h];
                acc += v_h[h] * pre.max(0.0);
                if pre > 0.0 {
                    want_dk[(r * N + c) * H + h] = gl * v_h[h];
                    want_dq[r * H + h] += gl * v_h[h];
                    want_dv[r * H + h] += gl * pre;
                }
            }
            want[r * cols + c] = if present { acc } else { f32::MIN };
        }
        for kk in 0..KX {
            want[r * cols + N + kk] = x_h[r * KX + kk];
        }
    }
    assert_close(&logits, &want, 1e-5, "additive logits");
    assert_close(&d_k.to_f32(), &want_dk, 1e-6, "d_k");
    let want_dx: Vec<f32> = (0..ROWS * KX)
        .map(|i| g[(i / KX) * cols + N + i % KX])
        .collect();
    assert_close(&d_x.to_f32(), &want_dx, 0.0, "additive d_extra");
    assert_close(&d_q.to_f32(), &want_dq, 1e-5, "d_q");
    assert_close(&dv.to_f32(), &want_dv, 1e-5, "dv_partial");

    // Dot, scoring the same keys against `q` as the query.
    let logits = entity::pointer_dot(&k, &q, &legal, Some(&x), N)
        .unwrap()
        .to_f32();
    let (d_e, d_x) = entity::pointer_dot_backward_de(&g_t, &k, &q, &legal, N, KX).unwrap();
    let d_qd = entity::pointer_dot_backward_dqd(&g_t, &k, &legal, N, KX).unwrap();
    let mut want_de = vec![0.0; ROWS * N * H];
    let mut want_dqd = vec![0.0; ROWS * H];
    for r in 0..ROWS {
        for c in 0..N {
            let gl = g[r * cols + c] * p[r * N + c];
            let mut acc = 0.0f32;
            for h in 0..H {
                acc += k_h[(r * N + c) * H + h] * q_h[r * H + h];
                want_de[(r * N + c) * H + h] = gl * q_h[r * H + h];
                want_dqd[r * H + h] += gl * k_h[(r * N + c) * H + h];
            }
            want[r * cols + c] = if p[r * N + c] != 0.0 { acc } else { f32::MIN };
        }
    }
    assert_close(&logits, &want, 1e-5, "dot logits");
    assert_close(&d_e.to_f32(), &want_de, 1e-6, "d_e");
    assert_close(&d_x.to_f32(), &want_dx, 0.0, "dot d_extra");
    assert_close(&d_qd.to_f32(), &want_dqd, 1e-5, "d_qd");
}

#[test]
fn bias_relu_and_its_adjoint_match_the_definition_across_units() {
    let device = dev();
    let (rows, h) = (ROWS * N, 64usize);
    let pre_h = noise(rows * h, 61);
    let bias_h = noise(h, 62);
    let g = noise(rows * h, 63);
    let pre = Tensor::<R, f32>::from_f32(&pre_h, vec![rows, h], &device).unwrap();
    let bias = Tensor::<R, f32>::from_f32(&bias_h, vec![h], &device).unwrap();
    let y = entity::bias_relu(&pre, &bias).unwrap();
    let want: Vec<f32> = (0..rows * h)
        .map(|i| (pre_h[i] + bias_h[i % h]).max(0.0))
        .collect();
    assert_close(&y.to_f32(), &want, 0.0, "bias_relu");

    let g_t = Tensor::<R, f32>::from_f32(&g, vec![rows, h], &device).unwrap();
    let d = entity::bias_relu_backward(&g_t, &y).unwrap();
    let want: Vec<f32> = (0..rows * h)
        .map(|i| if want[i] > 0.0 { g[i] } else { 0.0 })
        .collect();
    assert_close(&d.to_f32(), &want, 0.0, "bias_relu_backward");
}
