//! The Mamba-1 selective scan kernels against the sequential `f64` reference
//! (`mamba3::ssm::selective`): forward in both schedules, every gradient through
//! the autograd op, checkpoint reuse, and causality.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::ssm::selective::{
    SelectiveDims, SelectiveInputs, selective_scan_reference, selective_scan_reference_backward,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::selective_scan::{
    SelectiveScanMode, selective_scan, selective_scan_backward,
};

type R = Auto;
type V = Var<R, f32>;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn uniform(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

struct Case {
    d: SelectiveDims,
    x: Vec<f32>,
    dt: Vec<f32>,
    a: Vec<f32>,
    bm: Vec<f32>,
    cm: Vec<f32>,
    skip: Vec<f32>,
}

impl Case {
    fn new(batch: usize, seq: usize, dim: usize, state: usize, seed: u64) -> Self {
        let act = batch * seq * dim;
        let bc = batch * seq * state;
        Self {
            d: SelectiveDims {
                batch,
                seq,
                dim,
                state,
            },
            x: uniform(act, seed, -1.0, 1.0),
            dt: uniform(act, seed + 1, 0.01, 0.6),
            a: uniform(dim * state, seed + 2, -3.0, -0.05),
            bm: uniform(bc, seed + 3, -1.0, 1.0),
            cm: uniform(bc, seed + 4, -1.0, 1.0),
            skip: uniform(dim, seed + 5, -1.0, 1.0),
        }
    }

    fn f64s(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| x as f64).collect()
    }

    fn reference(&self, with_skip: bool) -> Vec<f64> {
        let (x, dt, a, bm, cm, skip) = (
            Self::f64s(&self.x),
            Self::f64s(&self.dt),
            Self::f64s(&self.a),
            Self::f64s(&self.bm),
            Self::f64s(&self.cm),
            Self::f64s(&self.skip),
        );
        let inp = SelectiveInputs {
            x: &x,
            dt: &dt,
            a: &a,
            bm: &bm,
            cm: &cm,
            skip: with_skip.then_some(&skip[..]),
        };
        selective_scan_reference(self.d, &inp)
    }

    fn tensors(&self, dev: &Device<R>) -> [Tensor<R, f32>; 6] {
        let d = self.d;
        [
            Tensor::from_data(&self.x, vec![d.batch, d.seq, d.dim], dev).unwrap(),
            Tensor::from_data(&self.dt, vec![d.batch, d.seq, d.dim], dev).unwrap(),
            Tensor::from_data(&self.a, vec![d.dim, d.state], dev).unwrap(),
            Tensor::from_data(&self.bm, vec![d.batch, d.seq, d.state], dev).unwrap(),
            Tensor::from_data(&self.cm, vec![d.batch, d.seq, d.state], dev).unwrap(),
            Tensor::from_data(&self.skip, vec![d.dim], dev).unwrap(),
        ]
    }
}

fn assert_close(name: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (*g as f64 - w).abs();
        assert!(
            err <= tol * scale,
            "{name}[{i}]: got {g}, want {w} (err {err:.3e}, scale {scale:.3e})"
        );
    }
}

const SHAPES: &[(usize, usize, usize, usize)] = &[
    (1, 1, 1, 1),
    (1, 5, 3, 4),
    (2, 17, 40, 16),
    (3, 33, 7, 8),
    (1, 64, 33, 16),
    (2, 9, 64, 5),
];

#[test]
fn forward_matches_reference_in_every_schedule() {
    let dev = dev();
    for (k, &(b, t, d, n)) in SHAPES.iter().enumerate() {
        let case = Case::new(b, t, d, n, 10 + k as u64);
        let [x, dt, a, bm, cm, skip] = case.tensors(&dev);
        for with_skip in [false, true] {
            let want = case.reference(with_skip);
            let modes = [
                SelectiveScanMode::Sequential,
                SelectiveScanMode::Chunked(1),
                SelectiveScanMode::Chunked(3),
                SelectiveScanMode::Chunked(16),
                SelectiveScanMode::Chunked(t + 5),
                SelectiveScanMode::Auto,
            ];
            for mode in modes {
                let y = selective_scan(&x, &dt, &a, &bm, &cm, with_skip.then_some(&skip), mode)
                    .unwrap()
                    .to_f32();
                assert_close(&format!("y {b}x{t}x{d}x{n} skip={with_skip} {mode:?}"), &y, &want, 2e-5);
            }
        }
    }
}

#[test]
fn gradients_match_reference_through_autograd() {
    let dev = dev();
    for (k, &(b, t, d, n)) in SHAPES.iter().enumerate() {
        let case = Case::new(b, t, d, n, 50 + k as u64);
        let w = uniform(b * t * d, 77 + k as u64, -1.0, 1.0);
        let [x, dt, a, bm, cm, skip] = case.tensors(&dev);
        for mode in [SelectiveScanMode::Sequential, SelectiveScanMode::Chunked(4)] {
            let vars: Vec<V> = [&x, &dt, &a, &bm, &cm, &skip]
                .into_iter()
                .map(|t| V::traced(t.clone()))
                .collect();
            let y = V::selective_scan(&vars[0], &vars[1], &vars[2], &vars[3], &vars[4], Some(&vars[5]), mode)
                .unwrap();
            let wv = V::constant(Tensor::from_data(&w, vec![b, t, d], &dev).unwrap());
            let loss = y.mul(&wv).unwrap().sum().unwrap();
            let grads = loss.backward_retain().unwrap();

            let f = Case::f64s;
            let (xs, dts, as_, bms, cms, sks, ws) =
                (f(&case.x), f(&case.dt), f(&case.a), f(&case.bm), f(&case.cm), f(&case.skip), f(&w));
            let inp = SelectiveInputs {
                x: &xs,
                dt: &dts,
                a: &as_,
                bm: &bms,
                cm: &cms,
                skip: Some(&sks),
            };
            let want = selective_scan_reference_backward(case.d, &inp, &ws);
            let tag = format!("{b}x{t}x{d}x{n} {mode:?}");
            let get = |v: &V| grads.node(v.node().unwrap()).expect("gradient").to_f32();
            assert_close(&format!("dx {tag}"), &get(&vars[0]), &want.dx, 5e-5);
            assert_close(&format!("ddt {tag}"), &get(&vars[1]), &want.ddt, 5e-5);
            assert_close(&format!("dA {tag}"), &get(&vars[2]), &want.da, 5e-5);
            assert_close(&format!("dBm {tag}"), &get(&vars[3]), &want.dbm, 5e-5);
            assert_close(&format!("dCm {tag}"), &get(&vars[4]), &want.dcm, 5e-5);
            assert_close(&format!("dskip {tag}"), &get(&vars[5]), &want.dskip, 5e-5);
        }
    }
}

/// The backward without saved checkpoints (it re-runs the forward) gives the same
/// gradients as with them.
#[test]
fn backward_without_checkpoints_recomputes() {
    let dev = dev();
    let case = Case::new(2, 21, 35, 16, 5);
    let [x, dt, a, bm, cm, skip] = case.tensors(&dev);
    let dy = Tensor::from_data(&uniform(2 * 21 * 35, 9, -1.0, 1.0), vec![2, 21, 35], &dev).unwrap();
    let saved = mamba3::tensor::ops::selective_scan::selective_scan_saving(
        &x,
        &dt,
        &a,
        &bm,
        &cm,
        Some(&skip),
        SelectiveScanMode::Chunked(8),
        true,
    )
    .unwrap()
    .1;
    let with = selective_scan_backward(&dy, &x, &dt, &a, &bm, &cm, Some(&skip), saved.as_ref()).unwrap();
    let without = selective_scan_backward(&dy, &x, &dt, &a, &bm, &cm, Some(&skip), None).unwrap();
    for (name, p, q) in [
        ("dx", &with.dx, &without.dx),
        ("ddt", &with.ddt, &without.ddt),
        ("da", &with.da, &without.da),
        ("dbm", &with.dbm, &without.dbm),
        ("dcm", &with.dcm, &without.dcm),
    ] {
        let (p, q) = (p.to_f32(), q.to_f32());
        let want: Vec<f64> = q.iter().map(|&v| v as f64).collect();
        assert_close(name, &p, &want, 1e-5);
    }
}

/// Changing any input after step `t0` leaves `y[..=t0]` bit-identical, in both
/// schedules.
#[test]
fn kernel_is_causal() {
    let dev = dev();
    let (b, t, d, n) = (2, 24, 9, 4);
    let base = Case::new(b, t, d, n, 31);
    for mode in [SelectiveScanMode::Sequential, SelectiveScanMode::Chunked(5)] {
        let [x, dt, a, bm, cm, skip] = base.tensors(&dev);
        let y0 = selective_scan(&x, &dt, &a, &bm, &cm, Some(&skip), mode).unwrap().to_f32();
        for cut in [0usize, 4, 5, 11, 23] {
            let mut p = Case::new(b, t, d, n, 31);
            for bi in 0..b {
                for tt in cut + 1..t {
                    for ch in 0..d {
                        let i = (bi * t + tt) * d + ch;
                        p.x[i] += 3.0;
                        p.dt[i] *= 2.0;
                    }
                    for s in 0..n {
                        let i = (bi * t + tt) * n + s;
                        p.bm[i] = -p.bm[i];
                        p.cm[i] += 1.0;
                    }
                }
            }
            let [x, dt, a, bm, cm, skip] = p.tensors(&dev);
            let y = selective_scan(&x, &dt, &a, &bm, &cm, Some(&skip), mode).unwrap().to_f32();
            for bi in 0..b {
                for tt in 0..=cut {
                    for ch in 0..d {
                        let i = (bi * t + tt) * d + ch;
                        assert_eq!(y[i].to_bits(), y0[i].to_bits(), "{mode:?} cut {cut} at b{bi} t{tt} d{ch}");
                    }
                }
            }
        }
    }
}

#[test]
fn rejects_bad_shapes() {
    let dev = dev();
    let case = Case::new(1, 4, 3, 2, 1);
    let [x, dt, a, bm, cm, _] = case.tensors(&dev);
    let wrong_a = Tensor::<R, f32>::zeros(vec![4, 2], &dev);
    assert!(selective_scan(&x, &dt, &wrong_a, &bm, &cm, None, SelectiveScanMode::Auto).is_err());
    let wrong_b = Tensor::<R, f32>::zeros(vec![1, 4, 3], &dev);
    assert!(selective_scan(&x, &dt, &a, &wrong_b, &cm, None, SelectiveScanMode::Auto).is_err());
    let too_wide = Tensor::<R, f32>::zeros(vec![3, 65], &dev);
    let bm65 = Tensor::<R, f32>::zeros(vec![1, 4, 65], &dev);
    assert!(selective_scan(&x, &dt, &too_wide, &bm65, &bm65, None, SelectiveScanMode::Auto).is_err());
}

/// Checkpoints saved for one set of inputs are not used for another: the
/// backward recomputes, so the gradients are those of the new inputs.
#[test]
fn checkpoints_of_other_inputs_are_not_reused() {
    let dev = dev();
    let a_case = Case::new(2, 19, 33, 8, 70);
    let b_case = Case::new(2, 19, 33, 8, 71);
    let [x1, dt1, a1, bm1, cm1, s1] = a_case.tensors(&dev);
    let [x2, dt2, a2, bm2, cm2, s2] = b_case.tensors(&dev);
    let dy = Tensor::from_data(&uniform(2 * 19 * 33, 3, -1.0, 1.0), vec![2, 19, 33], &dev).unwrap();
    let saved1 = mamba3::tensor::ops::selective_scan::selective_scan_saving(
        &x1, &dt1, &a1, &bm1, &cm1, Some(&s1), SelectiveScanMode::Sequential, true,
    )
    .unwrap()
    .1;
    let misuse = selective_scan_backward(&dy, &x2, &dt2, &a2, &bm2, &cm2, Some(&s2), saved1.as_ref()).unwrap();
    let fresh = selective_scan_backward(&dy, &x2, &dt2, &a2, &bm2, &cm2, Some(&s2), None).unwrap();
    assert_eq!(misuse.dx.to_f32(), fresh.dx.to_f32());
    assert_eq!(misuse.da.to_f32(), fresh.da.to_f32());
}

/// A long chunk: the chunk decay is the product of the per-step factors, so the
/// chunked result stays with the sequential one (an `exp(A * sum dt)` summary
/// drifted here).
#[test]
fn long_chunks_do_not_drift() {
    let dev = dev();
    let t = 40_000;
    let x: Vec<f32> = (0..t).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let dt = vec![0.1f32; t];
    let ones = vec![1.0f32; t];
    let xt = Tensor::<R, f32>::from_data(&x, vec![1, t, 1], &dev).unwrap();
    let dtt = Tensor::<R, f32>::from_data(&dt, vec![1, t, 1], &dev).unwrap();
    let a = Tensor::<R, f32>::from_data(&[-1e-4], vec![1, 1], &dev).unwrap();
    let bm = Tensor::<R, f32>::from_data(&ones, vec![1, t, 1], &dev).unwrap();
    let cm = bm.clone();
    let seq = selective_scan(&xt, &dtt, &a, &bm, &cm, None, SelectiveScanMode::Sequential).unwrap().to_f32();
    let chunked = selective_scan(&xt, &dtt, &a, &bm, &cm, None, SelectiveScanMode::Chunked(5_000)).unwrap().to_f32();
    for i in (0..t).step_by(997) {
        let rel = (seq[i] - chunked[i]).abs() / seq[i].abs().max(1e-12);
        assert!(rel < 1e-5, "t {i}: sequential {} chunked {} (rel {rel:.2e})", seq[i], chunked[i]);
    }
    // An absurd chunk length is capped rather than overflowing.
    let capped = selective_scan(&xt, &dtt, &a, &bm, &cm, None, SelectiveScanMode::Chunked(usize::MAX)).unwrap().to_f32();
    assert_eq!(capped.len(), t);
}

/// Half-precision storage (f16, bf16), f32 accumulation: forward and gradients
/// stay within the storage type's rounding of the f64 reference. Runs for each
/// type the backend supports and says which it skipped.
#[test]
fn half_precision_storage_stays_close() {
    use mamba3::backend::{DType, supports_dtype};
    let dev = dev();
    let case = Case::new(2, 33, 40, 16, 90);
    let d = case.d;
    let f = Case::f64s;
    let (xs, dts, as_, bms, cms, sks) = (f(&case.x), f(&case.dt), f(&case.a), f(&case.bm), f(&case.cm), f(&case.skip));
    let w = uniform(d.batch * d.seq * d.dim, 4, -1.0, 1.0);
    let inp = SelectiveInputs { x: &xs, dt: &dts, a: &as_, bm: &bms, cm: &cms, skip: Some(&sks) };
    let want_y = selective_scan_reference(d, &inp);
    let want_g = selective_scan_reference_backward(d, &inp, &f(&w));
    fn run<E: mamba3::backend::FloatElem>(
        case: &Case,
        w: &[f32],
        dev: &Device<R>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = case.d;
        let t = |v: &[f32], s: Vec<usize>| Tensor::<R, E>::from_f32(v, s, dev).unwrap();
        let (x, dt, a, bm, cm, skip) = (
            t(&case.x, vec![d.batch, d.seq, d.dim]),
            t(&case.dt, vec![d.batch, d.seq, d.dim]),
            t(&case.a, vec![d.dim, d.state]),
            t(&case.bm, vec![d.batch, d.seq, d.state]),
            t(&case.cm, vec![d.batch, d.seq, d.state]),
            t(&case.skip, vec![d.dim]),
        );
        let y = selective_scan(&x, &dt, &a, &bm, &cm, Some(&skip), SelectiveScanMode::Sequential).unwrap();
        let g = selective_scan_backward(&t(w, vec![d.batch, d.seq, d.dim]), &x, &dt, &a, &bm, &cm, Some(&skip), None).unwrap();
        (y.to_f32(), g.dx.to_f32(), g.da.to_f32())
    }
    let mut ran = Vec::new();
    for (name, dtype, tol) in [("f16", DType::F16, 2e-2), ("bf16", DType::BF16, 6e-2)] {
        if !supports_dtype(&dev, dtype) {
            eprintln!("half_precision_storage_stays_close: {name} not supported by this backend, skipped");
            continue;
        }
        let (y, dx, da) = match dtype {
            DType::F16 => run::<half::f16>(&case, &w, &dev),
            _ => run::<half::bf16>(&case, &w, &dev),
        };
        assert_close(&format!("{name} y"), &y, &want_y, tol);
        assert_close(&format!("{name} dx"), &dx, &want_g.dx, tol);
        assert_close(&format!("{name} dA"), &da, &want_g.da, tol);
        ran.push(name);
    }
    eprintln!("half_precision_storage_stays_close: checked {ran:?}");
}
