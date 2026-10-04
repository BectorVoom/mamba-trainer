//! K6 tests: the fragment-ion assignment head and its loss.
//!
//! Standalone module (plan item P4.2, neural part): `AssignmentHead::log_prob`
//! over `ion [B, F, N, J, 12]` / `ion_meta [B, F, N, 4]` with the formula
//! head's row network, and `AssignmentHead::loss` over `label_mask [B, N,
//! J + 1]` / `label_state [B, N]` for the training case `F = 1`, plus the
//! `ms2_ion_class_mask` kernel against its host twin.
//!
//! Every device call runs on poisoned outputs where a kernel writes one, is
//! followed by [`check_launches`], and opposed to an independent host
//! computation: a dropped launch (stale poison) or a wrong word fails. Sizes
//! stay small on the CPU runtime; the supervisor runs the same file on wgpu.
//!
//! Device-read discipline: `log_prob`/`loss` perform no device read. That is
//! asserted with the process-global transfer counters, so every test in this
//! binary serialises on the file-level [`LOCK`].

#![cfg(feature = "backend")]

use std::sync::Mutex;

use mamba3::autograd::Var;
use mamba3::backend::{
    Device, check_launches, download_bytes, reset_transfer_counters, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::assign::{
    AssignOutput, AssignmentHead, activation_bytes, ion_class_mask_host,
};
use mamba3::models::ms2::contract::ModelConfig;
use mamba3::models::ms2::formula_head::FormulaHead;
use mamba3::models::ms2::ion::{IonAssignment, IonHypothesis, IonLabel, IonLabels, label_mask};
use mamba3::nn::Module;
use mamba3::nn::param::Param;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::{IdTensor, slice_ids_along};
use mamba3::tensor::ops::movement;
use mamba3::tensor::ops::ms2_assign;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::optim::{AdamWConfig, Optimizer};

type R = Auto;
type E = f32;

/// File-level mutex: every test in this binary holds it, so the
/// process-global transfer counters isolate the no-device-read test.
/// Poison-tolerant: a failed test must not fail the rest of the binary.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Tiny model config: `d = 8`.
fn tiny_model() -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = 8;
    m
}

fn upload_ids(data: &[u32], shape: Vec<usize>, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(data, shape, device).unwrap()
}

fn upload_f(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, f32> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

/// The resident `[1024]` `ln(1 + n)` table, as
/// [`DeviceFormulaTable::upload`](mamba3::models::ms2::formula_head::DeviceFormulaTable::upload)
/// builds it.
fn log_table(device: &Device<R>) -> Tensor<R, f32> {
    let v: Vec<f32> = (0..1024).map(|n| (1.0 + n as f32).ln()).collect();
    upload_f(&v, vec![1024], device)
}

fn log_table_host() -> Vec<f32> {
    (0..1024).map(|n| (1.0 + n as f32).ln()).collect()
}

fn poison_f(len: usize) -> Vec<f32> {
    vec![f32::NAN; len]
}

// ---------------------------------------------------------------------------
// Independent host reference (f64, from the heads' parameters read back).
// ---------------------------------------------------------------------------

/// Host parameters read back from the two heads (row-major `[in, out]`
/// weights, matching `Linear`'s `y = x @ W + b` layout; the peak projection
/// `proj` has no bias per the specification
/// `logit_j = (e_j . W x_p) / sqrt(d)`).
struct HostParams {
    d: usize,
    row_in_w: Vec<f64>,
    row_in_b: Vec<f64>,
    row_out_w: Vec<f64>,
    row_out_b: Vec<f64>,
    proj_w: Vec<f64>,
    un_w: Vec<f64>,
    un_b: f64,
}

fn silu_f64(y: f64) -> f64 {
    y / (1.0 + (-y).exp())
}

fn read_params(
    head: &AssignmentHead<R, E>,
    formula: &FormulaHead<R, E>,
    d: usize,
) -> HostParams {
    fn one(params: &[(String, Param<R, E>)], name: &str) -> Vec<f64> {
        params
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("param {name}"))
            .1
            .value()
            .to_f32()
            .iter()
            .map(|&v| v as f64)
            .collect()
    }
    let an = head.named_parameters();
    let fnp = formula.named_parameters();
    HostParams {
        d,
        row_in_w: one(&fnp, "row_in.weight"),
        row_in_b: one(&fnp, "row_in.bias"),
        row_out_w: one(&fnp, "row_out.weight"),
        row_out_b: one(&fnp, "row_out.bias"),
        proj_w: one(&an, "proj.weight"),
        un_w: one(&an, "unassigned.weight"),
        un_b: one(&an, "unassigned.bias")[0],
    }
}

/// Bitwise equality of two `f32` slices (masks and counts are exactly
/// representable, so exactness is the assertion).
fn assert_bits_eq(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.to_bits() == e.to_bits(),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// Host `log_prob [B, F, N, J + 1]` in f64: masked entries are
/// `f64::NEG_INFINITY` (the device holds the finite masked value there).
#[allow(clippy::too_many_arguments)]
fn host_log_prob(
    hp: &HostParams,
    table: &[f32],
    ion: &[u32],
    ion_meta: &[u32],
    x: &[f32],
    b: usize,
    f: usize,
    n: usize,
    j: usize,
) -> Vec<f64> {
    let d = hp.d;
    let width = j + 1;
    let mut out = vec![f64::NEG_INFINITY; b * f * n * width];
    for bi in 0..b {
        for fi in 0..f {
            for pi in 0..n {
                let row = (bi * f + fi) * n + pi;
                let kept = ion_meta[row * 4 + 2] as usize;
                let status = ion_meta[row * 4 + 3];
                let eff = if status & 4 != 0 {
                    0
                } else {
                    kept.min(j)
                };
                let xb = (bi * n + pi) * d;
                // The specification's peak projection has no bias:
                // `q = W x_p` (written independently of the implementation).
                let mut q = vec![0.0f64; d];
                for (k, qk) in q.iter_mut().enumerate() {
                    let mut s = 0.0f64;
                    for (l, xv) in x[xb..xb + d].iter().enumerate() {
                        s += *xv as f64 * hp.proj_w[l * d + k];
                    }
                    *qk = s;
                }
                let mut u = hp.un_b;
                for (l, xv) in x[xb..xb + d].iter().enumerate() {
                    u += *xv as f64 * hp.un_w[l];
                }
                let mut logits = vec![f64::NEG_INFINITY; width];
                for (qq, slot) in logits.iter_mut().take(eff).enumerate() {
                    let hb = (row * j + qq) * 12;
                    let mut h = vec![0.0f64; d];
                    for (k, hk) in h.iter_mut().enumerate() {
                        let mut s = hp.row_in_b[k];
                        for (e, &c) in ion[hb..hb + 10].iter().enumerate() {
                            s += table[c as usize] as f64 * hp.row_in_w[e * d + k];
                        }
                        *hk = silu_f64(s);
                    }
                    let mut dot = 0.0f64;
                    for (k, qk) in q.iter().enumerate() {
                        let mut s = hp.row_out_b[k];
                        for (l, hl) in h.iter().enumerate() {
                            s += hl * hp.row_out_w[l * d + k];
                        }
                        dot += s * qk;
                    }
                    *slot = dot / (d as f64).sqrt();
                }
                logits[j] = u;
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0f64;
                for v in logits.iter() {
                    denom += (v - m).exp();
                }
                let base = row * width;
                for (c, v) in logits.iter().enumerate() {
                    if *v != f64::NEG_INFINITY {
                        out[base + c] = v - m - denom.ln();
                    }
                }
            }
        }
    }
    out
}

/// Host `L_assign` in f64 from host log-probabilities, plus the three
/// report counts as integers. Eligibility and partial come from the label
/// states alone (`1` or `3` eligible, `3` partial, `2` dropped), counted
/// independently of the device reductions.
fn host_loss(
    lp: &[f64],
    mask: &[f32],
    state: &[u32],
    b: usize,
    n: usize,
    j: usize,
) -> (f64, u64, u64, u64) {
    let width = j + 1;
    let mut total = 0.0f64;
    let (mut elig, mut partial, mut dropped) = (0u64, 0u64, 0u64);
    for bi in 0..b {
        for pi in 0..n {
            let row = bi * n + pi;
            let s = state[row];
            if s == 2 {
                dropped += 1;
            }
            if s == 3 {
                partial += 1;
            }
            if s != 1 && s != 3 {
                continue;
            }
            elig += 1;
            let base = row * width;
            let m = (0..width)
                .filter(|&c| mask[base + c] != 0.0)
                .map(|c| lp[base + c])
                .fold(f64::NEG_INFINITY, f64::max);
            let mut acc = 0.0f64;
            for c in 0..width {
                if mask[base + c] != 0.0 {
                    acc += (lp[base + c] - m).exp();
                }
            }
            total += -(m + acc.ln());
        }
    }
    let loss = total / (elig.max(1) as f64);
    (loss, elig, partial, dropped)
}

// ---------------------------------------------------------------------------
// The mask kernel against its host twin, on poisoned outputs.
// ---------------------------------------------------------------------------

#[test]
fn class_mask_kernel_matches_twin_on_poisoned_outputs() {
    let _guard = lock();
    let device = dev();
    // (b, f, n, j) shapes, from scalar edges to a wider sweep.
    let shapes = [
        (1usize, 1usize, 1usize, 1usize),
        (1, 1, 4, 2),
        (2, 3, 5, 3),
        (2, 2, 6, 8),
    ];
    for &(b, f, n, j) in &shapes {
        let width = j + 1;
        let mut rng = Rng::seeded((b * 100 + f * 10 + n) as u64 + j as u64);
        let statuses = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let mut meta = vec![0u32; b * f * n * 4];
        for row in 0..b * f * n {
            let u = rng.uniform_vec(2, 0.0, 1.0);
            // `kept` ranges past `J` to exercise the clamp; statuses cover
            // every bit combination so bits 0 and 1 prove inert and bit 2
            // proves dominant.
            meta[row * 4] = (u[0] * 41.0) as u32;
            meta[row * 4 + 1] = (u[1] * 17.0) as u32;
            meta[row * 4 + 2] = ((row * 5 + 3) % (j + 2)) as u32;
            meta[row * 4 + 3] = statuses[row % statuses.len()];
        }
        let want = ion_class_mask_host(&meta, b, f, n, j);
        let meta_t = upload_ids(&meta, vec![b, f, n, 4], &device);
        let mut out_t = upload_f(&poison_f(b * f * n * width), vec![b, f, n, width], &device);
        ms2_assign::ion_class_mask_into(&meta_t, &mut out_t).unwrap();
        check_launches(&device).unwrap();
        let got = out_t.try_to_f32().unwrap();
        assert_bits_eq(&got, &want, "mask words");
    }
    // Degenerate lane grid: no rows, no launch, still `Ok`.
    let meta_t = upload_ids(&[], vec![1, 1, 0, 4], &device);
    let mut out_t = upload_f(&[], vec![1, 1, 0, 3], &device);
    ms2_assign::ion_class_mask_into(&meta_t, &mut out_t).unwrap();
    check_launches(&device).unwrap();
    // Shape errors, never a panic.
    let meta_t = upload_ids(&[0u32; 8], vec![1, 2, 1, 4], &device);
    let mut bad_rank = upload_f(&[0.0; 8], vec![8], &device);
    assert!(ms2_assign::ion_class_mask_into(&meta_t, &mut bad_rank).is_err());
    // `ion_meta` last dim 3 instead of 4.
    let mut bad_last = upload_f(&[0.0; 4], vec![1, 2, 1, 2], &device);
    let meta3 = upload_ids(&[0u32; 6], vec![1, 2, 1, 3], &device);
    assert!(ms2_assign::ion_class_mask_into(&meta3, &mut bad_last).is_err());
    // `J = 0` (width 1) and `J = 9` (width 10) are refused.
    let mut w1 = upload_f(&[0.0; 2], vec![1, 2, 1, 1], &device);
    assert!(ms2_assign::ion_class_mask_into(&meta_t, &mut w1).is_err());
    let mut w10 = upload_f(&[0.0; 20], vec![1, 2, 1, 10], &device);
    assert!(ms2_assign::ion_class_mask_into(&meta_t, &mut w10).is_err());
    // Width mismatch against `ion_meta`.
    let mut w4 = upload_f(&[0.0; 8], vec![1, 2, 1, 4], &device);
    let meta_n2 = upload_ids(&[0u32; 16], vec![1, 2, 2, 4], &device);
    assert!(ms2_assign::ion_class_mask_into(&meta_n2, &mut w4).is_err());
}

// ---------------------------------------------------------------------------
// Shapes and the all-masked rule.
// ---------------------------------------------------------------------------

/// Fixed `(B, F, N, J) = (1, 1, 4, 2)` fixture: peak 0 has two kept
/// hypotheses, peak 1 one, peak 2 is `ion_unavailable` with stale-looking
/// kept data (the status must win), peak 3 has no hypothesis. Returns
/// `(ion, ion_meta, x, mask, state)`.
type PeakFixture = (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<u32>);

fn four_peak_fixture() -> PeakFixture {
    // J = 2.
    let (b, f, n, j) = (1usize, 1usize, 4usize, 2usize);
    let mut ion = vec![0u32; b * f * n * j * 12];
    // Peak 0, hyp 0: C6H12O6-ish; hyp 1: CH3.
    let h00: [u32; 10] = [6, 12, 0, 6, 0, 0, 0, 0, 0, 0];
    let h01: [u32; 10] = [1, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    for (e, &c) in h00.iter().enumerate() {
        ion[e] = c;
    }
    ion[10] = 180_000_000;
    ion[11] = 0x8000_0005;
    for (e, &c) in h01.iter().enumerate() {
        ion[12 + e] = c;
    }
    ion[22] = 15_000_000;
    ion[23] = 0x7fff_fffa;
    // Peak 1, hyp 0: C2H4O2-ish.
    let h10: [u32; 10] = [2, 4, 0, 2, 0, 0, 0, 0, 0, 0];
    let base1 = 2 * 12;
    for (e, &c) in h10.iter().enumerate() {
        ion[base1 + e] = c;
    }
    ion[base1 + 10] = 60_000_000;
    ion[base1 + 11] = 0x8000_0001;
    // Peak 2: stale kept data under `ion_unavailable` (status wins).
    let base2 = 2 * 2 * 12;
    for (e, &c) in h10.iter().enumerate() {
        ion[base2 + e] = c;
    }
    ion[base2 + 10] = 60_000_000;
    ion[base2 + 11] = 0x8000_0001;
    for (e, &c) in h01.iter().enumerate() {
        ion[base2 + 12 + e] = c;
    }
    ion[base2 + 22] = 15_000_000;
    ion[base2 + 23] = 0x7fff_fffa;
    // Peak 3: padding (zeroed row).
    let ion_meta = vec![
        2, 1, 2, 0, // peak 0: kept 2, complete
        1, 0, 1, 0, // peak 1: kept 1, complete
        0, 0, 2, 4, // peak 2: stale kept 2, unavailable
        0, 0, 0, 4, // peak 3: padding
    ];
    let x = vec![
        0.5, -0.3, 0.1, 0.0, 0.2, -0.1, 0.4, 0.05, //
        -0.2, 0.6, -0.4, 0.3, 0.0, 0.1, -0.5, 0.2, //
        0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    // One label on peak 0 (hyp 0) and peak 1 (hyp 0); peak 2 has no label;
    // peak 3 has labels but none kept.
    let mask = vec![
        1.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, //
        0.0, 0.0, 1.0,
    ];
    let state = vec![1, 1, 0, 2];
    (ion, ion_meta, x, mask, state)
}

#[test]
fn shapes_and_all_masked_rule() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(11);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let (ion, ion_meta, x, _mask, _state) = four_peak_fixture();
    let (b, f, n, j) = (1usize, 1usize, 4usize, 2usize);
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_t = upload_f(&x, vec![b, n, 8], &device);
    let out = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t))
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(out.log_prob.dims(), &[b, f, n, j + 1]);
    assert_eq!(out.mask.dims(), &[b, f, n, j + 1]);
    // The stored mask is the twin's mask exactly.
    let want_mask = ion_class_mask_host(&ion_meta, b, f, n, j);
    let got_mask = out.mask.to_f32();
    assert_bits_eq(&got_mask, &want_mask, "class mask");
    let lp = out.log_prob.try_to_f32().unwrap();
    for v in lp.iter() {
        assert!(!v.is_nan(), "no NaN anywhere in log_prob");
    }
    // Peaks 2 (unavailable, stale kept) and 3 (padding): probability 1 on
    // unassigned, log-prob 0 at class `J`, deeply negative elsewhere.
    for &p in &[2usize, 3usize] {
        let base = p * (j + 1);
        assert_eq!(
            lp[base + j].to_bits(),
            0.0f32.to_bits(),
            "peak {p}: unassigned log-prob is 0"
        );
        for c in 0..j {
            assert!(
                lp[base + c] < -1e30,
                "peak {p}: masked class {c} is deeply negative, got {}",
                lp[base + c]
            );
        }
    }
    // Peak 3's row never saw its stale data: an unavailable peak with kept
    // hypotheses in the buffer still scores unassigned-only (peak 2).
    // Probabilities of the existing classes sum to 1 (1e-5).
    for p in 0..n {
        let base = p * (j + 1);
        let mut s = 0.0f64;
        for c in 0..j + 1 {
            if want_mask[base + c] != 0.0 {
                s += (lp[base + c] as f64).exp();
            }
        }
        assert!(
            (s - 1.0).abs() <= 1e-5,
            "peak {p}: existing classes sum to 1, got {s}"
        );
    }
    // Shape errors, never a panic.
    let x_bad = upload_f(&x[..8], vec![b, 1, 8], &device);
    assert!(
        head.log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_bad))
            .is_err()
    );
    let ion4 = upload_ids(&ion[..8], vec![b, f, n, j], &device);
    assert!(
        head.log_prob(&formula, &lt, &ion4, &im_t, &Var::constant(upload_f(&x, vec![b, n, 8], &device)))
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Agreement with the independent host computation.
// ---------------------------------------------------------------------------

#[test]
fn agrees_with_host_f64() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(21);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let (b, f, n, j, d) = (2usize, 2usize, 6usize, 3usize, 8usize);
    let kept_pat = [3usize, 2, 1, 0, 2, 3];
    let status_pat = [0u32, 1, 0, 4, 2, 7];
    let mut ion = vec![0u32; b * f * n * j * 12];
    let mut ion_meta = vec![0u32; b * f * n * 4];
    let mut hrng = Rng::seeded(22);
    for row in 0..b * f * n {
        let p = row % n;
        let kept = kept_pat[p];
        let status = status_pat[p];
        ion_meta[row * 4] = (row as u32) % 5;
        ion_meta[row * 4 + 1] = (row as u32) % 3;
        ion_meta[row * 4 + 2] = kept as u32;
        ion_meta[row * 4 + 3] = status;
        if status & 4 == 0 {
            let u = hrng.uniform_vec(kept * 10, 0.0, 7.0);
            for (q, chunk) in u.chunks(10).enumerate().take(kept.min(j)) {
                let hb = (row * j + q) * 12;
                for (e, &v) in chunk.iter().enumerate() {
                    ion[hb + e] = v as u32;
                }
                ion[hb + 10] = 10_000_000 + (row as u32) * 1_000 + q as u32;
                ion[hb + 11] = 0x8000_0000u32.wrapping_add(row as u32);
            }
        }
    }
    let xu = hrng.uniform_vec(b * n * d, -1.0, 1.0);
    let x: Vec<f32> = xu;
    let table = log_table_host();
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_t = upload_f(&x, vec![b, n, d], &device);
    let out = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t))
        .unwrap();
    check_launches(&device).unwrap();
    let hp = read_params(&head, &formula, d);
    let want = host_log_prob(&hp, &table, &ion, &ion_meta, &x, b, f, n, j);
    let got = out.log_prob.try_to_f32().unwrap();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if *w == f64::NEG_INFINITY {
            assert!(
                *g < -1e30 && g.is_finite(),
                "index {i}: masked class is deeply negative and finite, got {g}"
            );
        } else {
            assert!(
                (*g as f64 - *w).abs() <= 1e-4,
                "index {i}: got {g}, want {w}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The loss against the independent computation, with counts.
// ---------------------------------------------------------------------------

#[test]
fn loss_matches_host_with_counts() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(31);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    // Peak 0: one label; peak 1: several labels (log of a sum); peak 2:
    // true partial (state 3: some label kept while another is not); peak 3:
    // state 2; peak 4 (unavailable, stale kept): state 0. J = 2.
    let (b, f, n, j) = (1usize, 1usize, 5usize, 2usize);
    let mut ion = vec![0u32; b * f * n * j * 12];
    let h0: [[u32; 10]; 2] = [[6, 12, 0, 6, 0, 0, 0, 0, 0, 0], [1, 3, 0, 0, 0, 0, 0, 0, 0, 0]];
    for p in 0..3usize {
        for (q, hyp) in h0.iter().enumerate() {
            let hb = (p * j + q) * 12;
            for (e, &c) in hyp.iter().enumerate() {
                ion[hb + e] = c;
            }
            ion[hb + 10] = 50_000_000 + p as u32;
            ion[hb + 11] = 0x8000_0000u32.wrapping_add(p as u32);
        }
    }
    // Peak 4 keeps stale data under `ion_unavailable`.
    for (q, hyp) in h0.iter().enumerate() {
        let hb = (4 * j + q) * 12;
        for (e, &c) in hyp.iter().enumerate() {
            ion[hb + e] = c;
        }
    }
    let ion_meta = vec![
        2, 0, 2, 0, // peak 0
        2, 0, 2, 0, // peak 1
        2, 0, 2, 0, // peak 2
        0, 0, 0, 0, // peak 3: no kept hypothesis
        0, 0, 2, 4, // peak 4: stale kept, unavailable
    ];
    let x = vec![
        0.4, -0.2, 0.1, 0.3, -0.1, 0.0, 0.2, -0.4, //
        -0.5, 0.1, 0.6, -0.3, 0.2, 0.4, -0.2, 0.1, //
        0.0, 0.5, -0.6, 0.1, 0.3, -0.4, 0.0, 0.2, //
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, //
        0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
    ];
    let mask = vec![
        1.0, 0.0, 0.0, // peak 0: one label
        1.0, 1.0, 0.0, // peak 1: several labels
        0.0, 1.0, 0.0, // peak 2: true partial (state 3)
        0.0, 0.0, 1.0, // peak 3: state 2
        0.0, 0.0, 1.0, // peak 4: state 0
    ];
    let state = vec![1, 1, 3, 2, 0];
    let table = log_table_host();
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_t = upload_f(&x, vec![b, n, 8], &device);
    let out = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t))
        .unwrap();
    check_launches(&device).unwrap();
    let mask_t = upload_f(&mask, vec![b, n, j + 1], &device);
    let state_t = upload_ids(&state, vec![b, n], &device);
    let al = head.loss(&out, &mask_t, &state_t).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(al.counts.dims(), &[3]);
    let hp = read_params(&head, &formula, 8);
    let host_lp = host_log_prob(&hp, &table, &ion, &ion_meta, &x, b, f, n, j);
    let (want_loss, want_elig, want_partial, want_dropped) =
        host_loss(&host_lp, &mask, &state, b, n, j);
    assert_eq!((want_elig, want_partial, want_dropped), (3, 1, 1));
    let got_loss = al.loss.try_to_f32().unwrap()[0];
    assert!(
        (got_loss as f64 - want_loss).abs() <= 1e-4,
        "loss: got {got_loss}, want {want_loss}"
    );
    let counts = al.counts.try_to_f32().unwrap();
    assert_bits_eq(&counts, &[3.0, 1.0, 1.0], "eligible/partial/dropped");
    // `F = 2` is refused: the loss is the training case `F = 1`.
    let fake_out = AssignOutput {
        log_prob: Var::constant(upload_f(&vec![0.0; b * 2 * n * (j + 1)], vec![b, 2, n, j + 1], &device)),
        mask: upload_f(&vec![0.0; b * 2 * n * (j + 1)], vec![b, 2, n, j + 1], &device),
    };
    assert!(head.loss(&fake_out, &mask_t, &state_t).is_err());
    // Zero eligible peaks give loss 0 and zero gradients.
    let state0 = upload_ids(&vec![0u32; b * n], vec![b, n], &device);
    let mask0 = upload_f(&[0.0, 0.0, 1.0].repeat(b * n), vec![b, n, j + 1], &device);
    let al0 = head.loss(&out, &mask0, &state0).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(al0.loss.try_to_f32().unwrap()[0].to_bits(), 0.0f32.to_bits());
    assert_bits_eq(&al0.counts.try_to_f32().unwrap(), &[0.0, 0.0, 0.0], "zero counts");
    let grads0 = al0.loss.backward_retain().unwrap();
    for (name, p) in head
        .named_parameters()
        .into_iter()
        .chain(formula.named_parameters())
    {
        match grads0.get(p.id()) {
            None => {}
            Some(g) => {
                assert!(
                    g.to_f32()
                        .iter()
                        .all(|&x| x.to_bits() == 0.0f32.to_bits()),
                    "{name}: zero-eligible loss has zero gradients"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// True-partial label retention (review finding A1).
// ---------------------------------------------------------------------------

/// The reviewer's two cases at `J = 4`, end to end through `label_mask` and
/// `AssignmentHead::loss`: one label fully retained gives partial 0, while
/// five labels with four retained give partial 1. Counts are checked against
/// an independent host count over labels versus kept hypotheses (never the
/// mask sum), so the old mask-sum proxy cannot pass.
#[test]
fn true_partial_counts_label_retention() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(97);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    // J = 4, N = 2. Peak 0 keeps one hypothesis with its only label;
    // peak 1 keeps four of its five labels.
    let (b, f, n, j) = (1usize, 1usize, 2usize, 4usize);
    let a: [u16; 10] = [6, 12, 0, 6, 0, 0, 0, 0, 0, 0];
    let c2: [u16; 10] = [1, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    let c3: [u16; 10] = [2, 4, 0, 2, 0, 0, 0, 0, 0, 0];
    let c4: [u16; 10] = [3, 5, 0, 1, 0, 0, 0, 0, 0, 0];
    let c5: [u16; 10] = [4, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let hyp = |counts: [u16; 10]| IonHypothesis {
        counts,
        mass: 50_000_000,
        residual: 0,
    };
    let hypotheses = vec![
        IonAssignment {
            accepted: 1,
            ambiguous: 0,
            kept: vec![hyp(a)],
            status: 0,
        },
        IonAssignment {
            accepted: 4,
            ambiguous: 0,
            kept: vec![hyp(a), hyp(c2), hyp(c3), hyp(c4)],
            status: 0,
        },
    ];
    let labels = IonLabels {
        labels: vec![
            IonLabel { raw_index: 0, counts: a },
            IonLabel { raw_index: 1, counts: a },
            IonLabel { raw_index: 1, counts: c2 },
            IonLabel { raw_index: 1, counts: c3 },
            IonLabel { raw_index: 1, counts: c4 },
            IonLabel { raw_index: 1, counts: c5 },
        ],
        overflow: 0,
    };
    let (mask, state) = label_mask(&[0, 1], &hypotheses, &labels, j);
    assert_eq!(state, vec![1, 3], "peak 0 fully retained, peak 1 true partial");
    assert_eq!(&mask[0..5], &[1.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&mask[5..10], &[1.0, 1.0, 1.0, 1.0, 0.0]);
    // Independent host count: labels versus kept hypotheses per peak.
    let mut want = (0u64, 0u64, 0u64);
    for (pi, assign) in hypotheses.iter().enumerate() {
        let raw = pi as u32;
        let lab: Vec<[u16; 10]> = labels
            .labels
            .iter()
            .filter(|l| l.raw_index == raw)
            .map(|l| l.counts)
            .collect();
        if lab.is_empty() {
            continue;
        }
        let kept_any = lab.iter().any(|c| assign.kept.iter().any(|h| h.counts == *c));
        let dropped_any = lab.iter().any(|c| !assign.kept.iter().any(|h| h.counts == *c));
        if !kept_any {
            want.2 += 1;
        } else {
            want.0 += 1;
            if dropped_any {
                want.1 += 1;
            }
        }
    }
    assert_eq!(want, (2, 1, 0), "independent count: eligible 2, partial 1");
    // The loss reduces the same counts on the device.
    let mut ion = vec![0u32; b * f * n * j * 12];
    for (p, assign) in hypotheses.iter().enumerate() {
        for (q, h) in assign.kept.iter().enumerate() {
            let hb = (p * j + q) * 12;
            for (e, &c) in h.counts.iter().enumerate() {
                ion[hb + e] = u32::from(c);
            }
            ion[hb + 10] = h.mass;
            ion[hb + 11] = (h.residual as u32).wrapping_add(0x8000_0000);
        }
    }
    let mut ion_meta = vec![0u32; b * f * n * 4];
    for (p, assign) in hypotheses.iter().enumerate() {
        ion_meta[p * 4] = assign.accepted;
        ion_meta[p * 4 + 1] = assign.ambiguous;
        ion_meta[p * 4 + 2] = assign.kept.len() as u32;
        ion_meta[p * 4 + 3] = assign.status;
    }
    let x = vec![0.25f32; b * n * 8];
    let table = log_table_host();
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_t = upload_f(&x, vec![b, n, 8], &device);
    let out = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t))
        .unwrap();
    check_launches(&device).unwrap();
    let mask_t = upload_f(&mask, vec![b, n, j + 1], &device);
    let state_t = upload_ids(&state, vec![b, n], &device);
    let al = head.loss(&out, &mask_t, &state_t).unwrap();
    check_launches(&device).unwrap();
    let hp = read_params(&head, &formula, 8);
    let host_lp = host_log_prob(&hp, &table, &ion, &ion_meta, &x, b, f, n, j);
    let (want_loss, want_elig, want_partial, want_dropped) =
        host_loss(&host_lp, &mask, &state, b, n, j);
    assert_eq!(
        (want_elig, want_partial, want_dropped),
        want,
        "host reference agrees with the independent count"
    );
    let got_loss = al.loss.try_to_f32().unwrap()[0];
    assert!(
        (got_loss as f64 - want_loss).abs() <= 1e-4,
        "loss: got {got_loss}, want {want_loss}"
    );
    let counts = al.counts.try_to_f32().unwrap();
    assert_bits_eq(&counts, &[2.0, 1.0, 0.0], "eligible/partial/dropped");
}

// ---------------------------------------------------------------------------
// Finite-difference gradient checks.
// ---------------------------------------------------------------------------

/// Central finite differences of the loss against the analytic gradients,
/// for the head's own parameters and the formula head's row-network
/// parameters, on the loss fixture above.
#[test]
fn gradients_match_finite_differences() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(41);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let (b, f, n, j) = (1usize, 1usize, 3usize, 2usize);
    let mut ion = vec![0u32; b * f * n * j * 12];
    let h0: [[u32; 10]; 2] = [[6, 12, 0, 6, 0, 0, 0, 0, 0, 0], [1, 3, 0, 0, 0, 0, 0, 0, 0, 0]];
    for p in 0..n {
        for (q, hyp) in h0.iter().enumerate() {
            let hb = (p * j + q) * 12;
            for (e, &c) in hyp.iter().enumerate() {
                ion[hb + e] = c;
            }
            ion[hb + 10] = 40_000_000 + p as u32;
            ion[hb + 11] = 0x8000_0000u32.wrapping_add(p as u32);
        }
    }
    let ion_meta = vec![2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0];
    let x = vec![
        0.4, -0.2, 0.1, 0.3, -0.1, 0.0, 0.2, -0.4, //
        -0.5, 0.1, 0.6, -0.3, 0.2, 0.4, -0.2, 0.1, //
        0.0, 0.5, -0.6, 0.1, 0.3, -0.4, 0.0, 0.2,
    ];
    let mask = vec![
        1.0, 0.0, 0.0, //
        1.0, 1.0, 0.0, //
        0.0, 1.0, 0.0,
    ];
    let state = vec![1, 1, 1];
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_t = upload_f(&x, vec![b, n, 8], &device);
    let mask_t = upload_f(&mask, vec![b, n, j + 1], &device);
    let state_t = upload_ids(&state, vec![b, n], &device);
    let loss_of = || {
        let out = head
            .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t.clone()))
            .unwrap();
        head.loss(&out, &mask_t, &state_t)
            .unwrap()
            .loss
            .try_to_f32()
            .unwrap()[0]
    };
    let out = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t.clone()))
        .unwrap();
    check_launches(&device).unwrap();
    let loss = head.loss(&out, &mask_t, &state_t).unwrap().loss;
    assert!(loss.try_to_f32().unwrap()[0].is_finite());
    let grads = loss.backward_retain().unwrap();
    // The head's own parameters and the formula head's row-network
    // parameters carry gradient; `pool_query` is not on this path.
    let mut names: Vec<(String, Param<R, E>)> = head.named_parameters();
    let fnames: Vec<(String, Param<R, E>)> = formula.named_parameters();
    for (name, p) in fnames {
        if name.starts_with("row_") {
            names.push((format!("formula.{name}"), p));
        } else {
            match grads.get(p.id()) {
                None => {}
                Some(g) => assert!(
                    g.to_f32()
                        .iter()
                        .all(|&v| v.to_bits() == 0.0f32.to_bits()),
                    "{name}: pool_query carries no assignment gradient"
                ),
            }
        }
    }
    let mut vacuous: Vec<String> = Vec::new();
    for (name, param) in &names {
        let shape = param.shape().dims().to_vec();
        let base = param.value().to_f32();
        let analytic = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("{name}: no gradient"))
            .to_f32();
        assert_eq!(analytic.len(), base.len(), "{name}: gradient length");
        let mut order: Vec<usize> = (0..base.len()).collect();
        order.sort_by(|&a, &b| {
            analytic[b]
                .abs()
                .partial_cmp(&analytic[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut exercised = false;
        for &idx in order.iter().take(3) {
            let mut up = base.clone();
            up[idx] += 1e-2;
            param.set(Tensor::<R, E>::from_f32(&up, shape.clone(), &device).unwrap());
            let fu = loss_of();
            let mut down = base.clone();
            down[idx] -= 1e-2;
            param.set(Tensor::<R, E>::from_f32(&down, shape.clone(), &device).unwrap());
            let fd = loss_of();
            param.set(Tensor::<R, E>::from_f32(&base, shape.clone(), &device).unwrap());
            let numeric = (fu - fd) / 2e-2;
            if numeric.abs() > 1e-4 {
                exercised = true;
            }
            assert!(
                (analytic[idx] - numeric).abs() <= 2e-2 * numeric.abs() + 1e-3,
                "{name}[{idx}]: analytic={} numeric={numeric}",
                analytic[idx]
            );
        }
        if !exercised {
            vacuous.push(name.clone());
        }
    }
    assert!(
        vacuous.is_empty(),
        "parameters with no sampled entry above 1e-4: {vacuous:?}"
    );
    check_launches(&device).unwrap();
}

/// Perturbing `x` at unavailable, padding or ineligible peaks does not
/// change the loss; perturbing it at an eligible peak does.
#[test]
fn no_gradient_from_unavailable_peaks() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(51);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let (ion, ion_meta, x, mask, state) = four_peak_fixture();
    let (b, f, n, j) = (1usize, 1usize, 4usize, 2usize);
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let mask_t = upload_f(&mask, vec![b, n, j + 1], &device);
    let state_t = upload_ids(&state, vec![b, n], &device);
    let loss_of = |xh: &[f32]| {
        let xt = upload_f(xh, vec![b, n, 8], &device);
        let out = head
            .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(xt))
            .unwrap();
        head.loss(&out, &mask_t, &state_t)
            .unwrap()
            .loss
            .try_to_f32()
            .unwrap()[0]
    };
    let base = loss_of(&x);
    assert!(base.is_finite());
    // Peaks 2 (unavailable, stale kept) and 3 (padding): large perturbations
    // leave the loss bitwise identical.
    for &p in &[2usize, 3usize] {
        let mut xp = x.clone();
        for v in xp.iter_mut().skip(p * 8).take(8) {
            *v += 2.5;
        }
        assert_eq!(
            loss_of(&xp).to_bits(),
            base.to_bits(),
            "peak {p}: perturbing x leaves the loss identical"
        );
    }
    // The check is not vacuous: peak 0 is eligible, so its perturbation moves
    // the loss.
    let mut x0 = x.clone();
    for v in x0.iter_mut().take(8) {
        *v += 2.5;
    }
    assert_ne!(
        loss_of(&x0).to_bits(),
        base.to_bits(),
        "eligible peak moves the loss"
    );
    // Analytic form: with a tracked `x`, the input gradient rows of the
    // gated peaks are exactly zero.
    let xt = Var::traced(upload_f(&x, vec![b, n, 8], &device));
    let out = head.log_prob(&formula, &lt, &ion_t, &im_t, &xt).unwrap();
    check_launches(&device).unwrap();
    let al = head.loss(&out, &mask_t, &state_t).unwrap();
    let grads = al.loss.backward_retain().unwrap();
    let dx = grads
        .node(xt.node().expect("tracked x has a node"))
        .expect("retained input gradient")
        .to_f32();
    assert_eq!(dx.len(), n * 8);
    for &p in &[2usize, 3usize] {
        for k in 0..8 {
            assert_eq!(
                dx[p * 8 + k].to_bits(),
                0.0f32.to_bits(),
                "peak {p}: analytic input gradient is 0"
            );
        }
    }
    assert!(
        dx[0..8].iter().any(|&v| v.to_bits() != 0.0f32.to_bits()),
        "eligible peak carries input gradient"
    );
}

// ---------------------------------------------------------------------------
// A learnable synthetic task.
// ---------------------------------------------------------------------------

#[test]
fn learnable_synthetic() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(61);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let (b, f, n, j, d) = (2usize, 1usize, 12usize, 2usize, 8usize);
    // Two fixed, well-separated hypotheses shared by every peak.
    let hyps: [[u32; 10]; 2] = [
        [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    let mut ion = vec![0u32; b * f * n * j * 12];
    for row in 0..b * f * n {
        for (q, hyp) in hyps.iter().enumerate() {
            let hb = (row * j + q) * 12;
            for (e, &c) in hyp.iter().enumerate() {
                ion[hb + e] = c;
            }
            ion[hb + 10] = 50_000_000 + q as u32;
            ion[hb + 11] = 0x8000_0000u32;
        }
    }
    let mut ion_meta = vec![0u32; b * f * n * 4];
    for row in 0..b * f * n {
        ion_meta[row * 4] = 2;
        ion_meta[row * 4 + 2] = 2;
    }
    // The label alternates by peak; `x` carries it on its first two dims.
    let mut labels = vec![0usize; b * n];
    let mut xrng = Rng::seeded(62);
    let jit = xrng.uniform_vec(b * n * (d - 2), -0.03, 0.03);
    let mut x = vec![0.0f32; b * n * d];
    for (i, lab) in labels.iter_mut().enumerate() {
        let l = i % 2;
        *lab = l;
        x[i * d] = if l == 0 { 0.7 } else { -0.7 };
        x[i * d + 1] = -x[i * d];
        for (k, xv) in x.iter_mut().skip(i * d + 2).take(d - 2).enumerate() {
            *xv = jit[i * (d - 2) + k];
        }
    }
    let mut mask = vec![0.0f32; b * n * (j + 1)];
    for (i, &l) in labels.iter().enumerate() {
        mask[i * (j + 1) + l] = 1.0;
    }
    let state = vec![1u32; b * n];
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_var = Var::constant(upload_f(&x, vec![b, n, d], &device));
    let mask_t = upload_f(&mask, vec![b, n, j + 1], &device);
    let state_t = upload_ids(&state, vec![b, n], &device);
    let mut opt = AdamWConfig::builder()
        .learning_rate(1e-2)
        .build()
        .init::<R, E>();
    let mut params = head.named_parameters();
    params.extend(formula.named_parameters());
    let only: Vec<Param<R, E>> = params.iter().map(|(_, p)| p.clone()).collect();
    let mut curve = Vec::new();
    for step in 0..=300 {
        let out = head.log_prob(&formula, &lt, &ion_t, &im_t, &x_var).unwrap();
        let al = head.loss(&out, &mask_t, &state_t).unwrap();
        if step % 50 == 0 {
            let v = al.loss.try_to_f32().unwrap()[0];
            assert!(v.is_finite(), "step {step}: loss is finite");
            curve.push(v);
        }
        if step == 300 {
            break;
        }
        let grads = al.loss.backward().unwrap();
        opt.step(&only, &grads).unwrap();
    }
    check_launches(&device).unwrap();
    assert!(
        *curve.last().unwrap() < 0.3 * curve[0],
        "loss falls below 30% of its initial value: {curve:?}"
    );
    // The labelled hypothesis has the largest probability on (at least) 90%
    // of peaks.
    let out = head.log_prob(&formula, &lt, &ion_t, &im_t, &x_var).unwrap();
    check_launches(&device).unwrap();
    let lp = out.log_prob.try_to_f32().unwrap();
    let mut correct = 0usize;
    for (i, &lab) in labels.iter().enumerate() {
        let row = &lp[i * (j + 1)..(i + 1) * (j + 1)];
        let mut best = 0usize;
        for (c, &v) in row.iter().enumerate().skip(1) {
            if v > row[best] {
                best = c;
            }
        }
        if best == lab {
            correct += 1;
        }
    }
    assert!(
        correct * 10 >= 9 * b * n,
        "labelled hypothesis wins on {correct}/{} peaks",
        b * n
    );
}

// ---------------------------------------------------------------------------
// No device read in `log_prob`/`loss`.
// ---------------------------------------------------------------------------

#[test]
fn no_device_read_in_log_prob_or_loss() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(71);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let (ion, ion_meta, x, mask, state) = four_peak_fixture();
    let (b, f, n, j) = (1usize, 1usize, 4usize, 2usize);
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_t = upload_f(&x, vec![b, n, 8], &device);
    let mask_t = upload_f(&mask, vec![b, n, j + 1], &device);
    let state_t = upload_ids(&state, vec![b, n], &device);
    // Warm up (compile, autotune) outside the measured section.
    let out = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t.clone()))
        .unwrap();
    let al = head.loss(&out, &mask_t, &state_t).unwrap();
    al.loss.backward_retain().unwrap();
    let _ = out.log_prob.try_to_f32().unwrap();
    check_launches(&device).unwrap();
    reset_transfer_counters();
    let out = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t))
        .unwrap();
    let al = head.loss(&out, &mask_t, &state_t).unwrap();
    al.loss.backward_retain().unwrap();
    check_launches(&device).unwrap();
    assert_eq!(runtime_read_count(), 0, "no device read in log_prob/loss");
    assert_eq!(download_bytes(), 0, "no download bytes in log_prob/loss");
    // The values are still the all-masked-rule rows of the fixture test.
    let lp = al.loss.try_to_f32().unwrap()[0];
    assert!(lp.is_finite());
}

// ---------------------------------------------------------------------------
// Row independence: `B = 2` equals two `B = 1` runs.
// ---------------------------------------------------------------------------

#[test]
fn rows_are_independent_across_batches() {
    let _guard = lock();
    let device = dev();
    let model = tiny_model();
    let mut rng = Rng::seeded(81);
    let formula = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let head = AssignmentHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    // `F = 1` with every peak eligible, so the loss is comparable too.
    let (b, f, n, j, d) = (2usize, 1usize, 4usize, 2usize, 8usize);
    let mut hrng = Rng::seeded(82);
    let iu = hrng.uniform_vec(b * f * n * j * 10, 0.0, 6.0);
    let mut ion = vec![0u32; b * f * n * j * 12];
    for (i, &v) in iu.iter().enumerate() {
        let q = i % (j * 10);
        let row = i / (j * 10);
        let hb = row * j * 12 + (q / 10) * 12 + (q % 10);
        ion[hb] = v as u32;
    }
    for q in 0..b * f * n * j {
        ion[q * 12 + 10] = 30_000_000 + q as u32;
        ion[q * 12 + 11] = 0x8000_0000u32;
    }
    let mut ion_meta = vec![0u32; b * f * n * 4];
    for row in 0..b * f * n {
        ion_meta[row * 4] = 2;
        ion_meta[row * 4 + 2] = j as u32;
    }
    let x = hrng.uniform_vec(b * n * d, -1.0, 1.0);
    let mask: Vec<f32> = [1.0, 0.0, 0.0].repeat(b * n);
    let state = vec![1u32; b * n];
    let lt = log_table(&device);
    let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
    let x_t = upload_f(&x, vec![b, n, d], &device);
    let mask_t = upload_f(&mask, vec![b, n, j + 1], &device);
    let state_t = upload_ids(&state, vec![b, n], &device);
    let full = head
        .log_prob(&formula, &lt, &ion_t, &im_t, &Var::constant(x_t.clone()))
        .unwrap();
    check_launches(&device).unwrap();
    let full_lp = full.log_prob.try_to_f32().unwrap();
    let full_loss = head
        .loss(&full, &mask_t, &state_t)
        .unwrap()
        .loss
        .try_to_f32()
        .unwrap()[0];
    // The joint loss is the mean over all eligible peaks; each split scores
    // the same per-peak values (compared above), so the joint loss equals
    // the mean of the splits here (equal eligible counts per batch) within
    // reduction order.
    let mut split_sum = 0.0f32;
    for bi in 0..b {
        let ion_b = slice_ids_along(&ion_t, 0, bi, 1).unwrap();
        let im_b = slice_ids_along(&im_t, 0, bi, 1).unwrap();
        let x_b = movement::slice(&x_t, 0, bi, 1).unwrap();
        let one = head
            .log_prob(&formula, &lt, &ion_b, &im_b, &Var::constant(x_b))
            .unwrap();
        check_launches(&device).unwrap();
        let one_lp = one.log_prob.try_to_f32().unwrap();
        let base = bi * f * n * (j + 1);
        for (k, &v) in one_lp.iter().enumerate() {
            let g = full_lp[base + k];
            assert!(
                (v - g).abs() <= 1e-6,
                "batch {bi} word {k}: split {v} versus joint {g}"
            );
        }
        let mask_b = movement::slice(&mask_t, 0, bi, 1).unwrap();
        let state_b = slice_ids_along(&state_t, 0, bi, 1).unwrap();
        split_sum += head
            .loss(&one, &mask_b, &state_b)
            .unwrap()
            .loss
            .try_to_f32()
            .unwrap()[0];
    }
    assert!(
        (full_loss - split_sum / b as f32).abs() <= 1e-5,
        "joint {full_loss} versus mean of splits {}",
        split_sum / b as f32
    );
}

// ---------------------------------------------------------------------------
// Memory estimate.
// ---------------------------------------------------------------------------

#[test]
fn activation_bytes_counts_listed_tensors() {
    let _guard = lock();
    // `B = 8, F = 4, N = 128, J = 4, d = 128`, FP32: features
    // `8*4*128*4*10*4` = 655,360; two row-network activations
    // `2*8*4*128*4*128*4` = 16,777,216 (8 MiB per layer, as in §2.2);
    // logits and log-probs `2*8*4*128*5*4` = 163,840; total 17,596,416.
    assert_eq!(activation_bytes(8, 4, 128, 4, 128, 4).unwrap(), 17_596_416);
    assert_eq!(activation_bytes(0, 4, 128, 4, 128, 4).unwrap(), 0);
    assert!(activation_bytes(usize::MAX, usize::MAX, 128, 4, 128, 4).is_err());
}
