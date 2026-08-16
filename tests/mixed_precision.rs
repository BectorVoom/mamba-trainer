//! The reduced-precision matmul modes: `bf16` and `f16` operands, `f32` accumulate.
//!
//! Two questions, in order. First, does this backend compile and run narrow
//! float kernels *at all*? That is the probe the mixed mode was designed
//! around, and its answer decides where the mode can be tested — on CubeCL's
//! CPU (MLIR) runtime the answer turns out to be yes for both types, so
//! everything below runs locally rather than only on a GPU.
//!
//! Second, does the mode compute what it claims: every operand rounded once,
//! every product accumulated in `f32`? That is checked against a host-side
//! reference that rounds the same way, on every kernel the tuner can reach —
//! forced explicitly, because the `Auto` path on a CPU only ever picks one of
//! them and an instantiation that never launches is never compiled, let alone
//! verified.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::matmul::{
    MatmulKernel, MatmulPrecision, matmul, matmul_nt, matmul_precision, set_default_kernel,
    set_matmul_precision,
};

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Serialises the tests in this file.
///
/// The precision mode and the kernel choice are process-global — that is the
/// point of them, a run opts in once — and the test harness runs the functions
/// in one binary *concurrently*. Two tests each setting the mode and then
/// checking a product against a reference rounded their own way will read each
/// other's mode and disagree by exactly one narrow-type rounding, which looks
/// like a kernel bug and is not one.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take the lock, ignoring poisoning: one failing test should report its own
/// assertion, not turn every later one into a poison panic.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The two narrow modes, with the host-side rounding each one must match.
///
/// `f16` is the more accurate of the two per product (three more mantissa bits)
/// and the one that can overflow (five fewer exponent bits); the tolerances
/// below are the same for both because they are all relative to a reference
/// rounded the same way.
type NarrowMode = (MatmulPrecision, &'static str, fn(f32) -> f32);

const NARROW: [NarrowMode; 2] = [
    (MatmulPrecision::Bf16, "bf16", |v| {
        half::bf16::from_f32(v).to_f32()
    }),
    (MatmulPrecision::F16, "f16", |v| {
        half::f16::from_f32(v).to_f32()
    }),
];

#[test]
fn narrow_elemwise_kernels_compile_and_run() {
    let _serial = serial();
    let a_data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let b_data: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.5).collect();

    macro_rules! check {
        ($narrow:ty, $name:literal) => {{
            let a = Tensor::<R, $narrow>::from_f32(&a_data, vec![4, 16], &dev()).unwrap();
            let b = Tensor::<R, $narrow>::from_f32(&b_data, vec![4, 16], &dev()).unwrap();

            let sum = mamba3::tensor::ops::elemwise::add(&a, &b).unwrap().to_f32();
            for ((s, x), y) in sum.iter().zip(&a_data).zip(&b_data) {
                // One rounding of each operand plus one of the sum.
                let want = x + y;
                assert!(
                    (s - want).abs() <= 0.02 * (1.0 + want.abs()),
                    concat!($name, " add: {} vs {}"),
                    s,
                    want
                );
            }

            let e = mamba3::tensor::ops::elemwise::exp(&a).to_f32();
            for (v, x) in e.iter().zip(&a_data) {
                let want = x.exp();
                assert!(
                    (v - want).abs() <= 0.02 * (1.0 + want.abs()),
                    concat!($name, " exp: {} vs {}"),
                    v,
                    want
                );
            }
        }};
    }
    check!(half::bf16, "bf16");
    check!(half::f16, "f16");
}

#[test]
fn narrow_matmul_kernels_compile_and_run() {
    let _serial = serial();
    let a_data: Vec<f32> = (0..48).map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.125).collect();
    let b_data: Vec<f32> = (0..60).map(|i| ((i * 11 % 19) as f32 - 9.0) * 0.125).collect();

    macro_rules! check {
        ($narrow:ty, $name:literal) => {{
            let a = Tensor::<R, $narrow>::from_f32(&a_data, vec![4, 12], &dev()).unwrap();
            let b = Tensor::<R, $narrow>::from_f32(&b_data, vec![12, 5], &dev()).unwrap();
            let got = mamba3::tensor::ops::matmul::matmul(&a, &b).unwrap().to_f32();
            for (r, row) in got.chunks(5).enumerate() {
                for (c, v) in row.iter().enumerate() {
                    let want: f32 = (0..12)
                        .map(|k| a_data[r * 12 + k] * b_data[k * 5 + c])
                        .sum();
                    // Narrow storage rounds every operand and every partial; a
                    // 12-term dot is good to a couple of percent.
                    assert!(
                        (v - want).abs() <= 0.05 * (1.0 + want.abs()),
                        concat!($name, " matmul [{},{}]: {} vs {}"),
                        r,
                        c,
                        v,
                        want
                    );
                }
            }
        }};
    }
    check!(half::bf16, "bf16");
    check!(half::f16, "f16");
}

/// The mixed-precision modes end to end: an `f32` call with a mode on must
/// round its operands once and accumulate in `f32`, on every kernel the tuner
/// can pick, and a whole model forward pass must stay within that mode's error
/// bounds of the `f32` one.
///
/// One test rather than several because the mode is process-global state.
#[test]
fn mixed_precision_modes() {
    let _serial = serial();
    assert_eq!(
        matmul_precision(),
        MatmulPrecision::F32,
        "the mode must default to off"
    );

    // Values chosen to be representable in neither narrow type, so a mode that
    // silently did nothing would fail the `moved` assertion below.
    let a_data: Vec<f32> = (0..64).map(|i| 0.1 + 0.013 * (i as f32)).collect();
    let b_data: Vec<f32> = (0..96).map(|i| 0.3 + 0.007 * ((i * 5 % 23) as f32)).collect();
    let a = Tensor::<R, f32>::from_f32(&a_data, vec![8, 8], &dev()).unwrap();
    let b = Tensor::<R, f32>::from_f32(&b_data, vec![8, 12], &dev()).unwrap();
    let bt = Tensor::<R, f32>::from_f32(&b_data, vec![12, 8], &dev()).unwrap();
    let exact = matmul(&a, &b).unwrap().to_f32();

    // A `k` the vector width does not divide, which is what sends the block
    // kernel down its scalar-staged path instead of the vectorised one.
    let a9_data: Vec<f32> = (0..72).map(|i| 0.1 + 0.013 * (i as f32)).collect();
    let b9_data: Vec<f32> = (0..108).map(|i| 0.3 + 0.007 * ((i * 5 % 23) as f32)).collect();
    let a9 = Tensor::<R, f32>::from_f32(&a9_data, vec![8, 9], &dev()).unwrap();
    let b9 = Tensor::<R, f32>::from_f32(&b9_data, vec![9, 12], &dev()).unwrap();

    for (mode, name, round) in NARROW {
        // The reference: round each operand on the host, accumulate in f32. The
        // product of two rounded values is exact in f32, so the only slack the
        // tolerance has to cover is summation order.
        let ar: Vec<f32> = a_data.iter().copied().map(round).collect();
        let br: Vec<f32> = b_data.iter().copied().map(round).collect();
        let mut want = vec![0.0f32; 8 * 12];
        for r in 0..8 {
            for c in 0..12 {
                want[r * 12 + c] = (0..8).map(|p| ar[r * 8 + p] * br[p * 12 + c]).sum();
            }
        }

        set_matmul_precision(mode);
        let mut moved = 0.0f32;
        for kernel in [
            MatmulKernel::Simple,
            MatmulKernel::RowTiled,
            MatmulKernel::Tiled,
            MatmulKernel::BlockTiled,
        ] {
            set_default_kernel(kernel);
            let got = matmul(&a, &b).unwrap().to_f32();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
                    "{name} matmul[{i}] under {kernel:?}: {g} vs rounded reference {w}"
                );
            }
            moved = moved.max(
                got.iter()
                    .zip(&exact)
                    .map(|(g, e)| (g - e).abs())
                    .fold(0.0f32, f32::max),
            );

            // The transposed entry points resolve the mode too; under
            // `BlockTiled` this is the transposed block kernel reading the
            // narrow operand in place rather than staging a copy.
            let got_nt = matmul_nt(&a, &bt).unwrap().to_f32();
            for r in 0..8 {
                for c in 0..12 {
                    let w: f32 = (0..8).map(|p| ar[r * 8 + p] * br[c * 8 + p]).sum();
                    let g = got_nt[r * 12 + c];
                    assert!(
                        (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
                        "{name} matmul_nt[{r},{c}] under {kernel:?}: {g} vs {w}"
                    );
                }
            }
        }
        assert!(
            moved > 0.0,
            "{name} mode changed nothing — operands were not rounded"
        );

        set_default_kernel(MatmulKernel::BlockTiled);
        let got9 = matmul(&a9, &b9).unwrap().to_f32();
        set_default_kernel(MatmulKernel::Auto);
        for r in 0..8 {
            for c in 0..12 {
                let w: f32 = (0..9)
                    .map(|p| round(a9_data[r * 9 + p]) * round(b9_data[p * 12 + c]))
                    .sum();
                let g = got9[r * 12 + c];
                assert!(
                    (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
                    "{name} scalar-staged block matmul[{r},{c}]: {g} vs {w}"
                );
            }
        }
        set_matmul_precision(MatmulPrecision::F32);
    }
}

/// A whole forward pass under each mode: finite, and within that mode's error
/// bounds of the `f32` run.
#[test]
fn mixed_precision_forward_pass_stays_close() {
    let _serial = serial();
    let model = mamba3::prelude::Mamba3LmConfig::builder()
        .vocab_size(16)
        .d_model(16)
        .n_layers(2)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 4;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .seed(7)
        .build()
        .unwrap()
        .init::<R, f32>(&dev())
        .unwrap();
    let tokens = mamba3::tensor::ops::index::IdTensor::from_slice(
        &[1, 2, 3, 4, 5, 6, 7, 8],
        vec![1, 8],
        &dev(),
    )
    .unwrap();

    set_matmul_precision(MatmulPrecision::F32);
    let reference = model.forward(&tokens, false).unwrap().to_f32();
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    for (mode, name, _) in NARROW {
        set_matmul_precision(mode);
        let got = model.forward(&tokens, false).unwrap().to_f32();
        set_matmul_precision(MatmulPrecision::F32);

        let worst = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Both types keep 8 or 11 mantissa bits (~0.4% and ~0.05% relative);
        // a two-layer stack of products compounds that, so the bound is loose
        // but still an order of magnitude tighter than "wrong".
        assert!(
            worst <= 0.05 * (1.0 + scale),
            "{name}-mode forward drifted {worst} at logit scale {scale}"
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{name}-mode forward produced non-finite logits"
        );
    }
}

/// The matrix-core path, on devices that have matrix cores.
///
/// This *skips* rather than fails where the hardware is absent, which on the
/// CPU runtime is always: whether a machine has tensor cores is not something
/// the crate gets to decide. On a GPU that reports the 16×16×16 fragment it
/// forces the path and checks it against the same host-rounded reference every
/// other kernel is checked against — the fragment pipeline computes the same
/// product or it is wrong.
///
/// The tuner covers this too, and more thoroughly: with a mode on,
/// `MAMBA3_TUNE_CHECK=1` verifies the matrix-core candidate against the simple
/// kernel on every shape a real training step issues, with real operands.
#[test]
fn cmma_matches_the_reference_where_the_device_has_it() {
    let _serial = serial();
    use mamba3::tensor::ops::matmul::cmma_available;

    // Big enough to cover several fragments per block and to leave a ragged
    // tail on both axes, which is the case the guarded copy exists for.
    let (m, k, n) = (40usize, 48usize, 36usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| 0.05 + 0.011 * ((i % 37) as f32)).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| -0.2 + 0.013 * ((i % 41) as f32)).collect();
    let a = Tensor::<R, f32>::from_f32(&a_data, vec![m, k], &dev()).unwrap();
    let b = Tensor::<R, f32>::from_f32(&b_data, vec![k, n], &dev()).unwrap();
    // The same right operand stored transposed, for the `nt` form the scan and
    // every weight adjoint use.
    let mut bt_data = vec![0.0f32; n * k];
    for r in 0..k {
        for c in 0..n {
            bt_data[c * k + r] = b_data[r * n + c];
        }
    }
    let bt = Tensor::<R, f32>::from_f32(&bt_data, vec![n, k], &dev()).unwrap();

    let mut ran = 0;
    for (mode, name, round) in NARROW {
        if !cmma_available(&dev(), mode) {
            continue;
        }
        ran += 1;
        let ar: Vec<f32> = a_data.iter().copied().map(round).collect();
        let br: Vec<f32> = b_data.iter().copied().map(round).collect();

        set_matmul_precision(mode);
        set_default_kernel(MatmulKernel::Cmma);
        let got = matmul(&a, &b).unwrap().to_f32();
        let got_nt = matmul_nt(&a, &bt).unwrap().to_f32();
        set_default_kernel(MatmulKernel::Auto);
        set_matmul_precision(MatmulPrecision::F32);

        for r in 0..m {
            for c in 0..n {
                let want: f32 = (0..k).map(|p| ar[r * k + p] * br[p * n + c]).sum();
                // Fragments accumulate in an order the hardware chooses, so
                // this is looser than the register kernels' 1e-5: it is the
                // reassociation of a k-term f32 sum, not a precision claim.
                let tol = 1e-4 * (k as f32).sqrt() * (1.0 + want.abs());
                assert!(
                    (got[r * n + c] - want).abs() <= tol,
                    "{name} cmma matmul[{r},{c}]: {} vs {want}",
                    got[r * n + c]
                );
                assert!(
                    (got_nt[r * n + c] - want).abs() <= tol,
                    "{name} cmma matmul_nt[{r},{c}]: {} vs {want}",
                    got_nt[r * n + c]
                );
            }
        }
    }
    if ran == 0 {
        eprintln!("no matrix-core fragment reported by this device — cmma path skipped");
    }

    // The fallback, which every device can run: asking for matrix cores where
    // there are none must still compute the product, not fail and not return
    // something else. On a machine without them this is the only part of this
    // test that executes, and it is the part that keeps the request safe.
    let reference = matmul(&a, &b).unwrap().to_f32();
    set_default_kernel(MatmulKernel::Cmma);
    let fallback = matmul(&a, &b).unwrap().to_f32();
    set_default_kernel(MatmulKernel::Auto);
    for (i, (g, w)) in fallback.iter().zip(&reference).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
            "cmma fallback[{i}]: {g} vs {w}"
        );
    }
}
