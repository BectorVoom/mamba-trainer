//! Baseline encoder-stack tests (task K8): shapes, padding discipline,
//! batch behaviour, invariance/equivariance/causality, gradients, parameter
//! matching and the all-padding edge case — for every baseline stack.
//!
//! Run with `cargo test --release --no-default-features --features cpu
//! --test ms2_baselines`.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::baselines::{
    EncoderStackKind, SetEncoderStackConfig, TransformerEncoderStackConfig,
    UnidirectionalMambaStackConfig, bidirectional_reference_params, mamba3_block_params,
    masked_mean,
};
use mamba3::models::ms2::contract::ModelConfig;
use mamba3::nn::Module;
use mamba3::ssm::SsmConfig;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

const D: usize = 8;
const N: usize = 6;

#[derive(Clone, Copy)]
enum Kind {
    Set,
    Transformer,
    Uni,
}

const KINDS: [Kind; 3] = [Kind::Set, Kind::Transformer, Kind::Uni];

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Set => "set",
        Kind::Transformer => "transformer",
        Kind::Uni => "uni",
    }
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn small_ssm(d: usize) -> SsmConfig {
    let mut s = ModelConfig::v0().encoder;
    s.d_model = d;
    if d == D {
        s.n_heads = 2;
        s.head_dim = 4;
        s.d_state = 4;
        s.n_groups = 2;
    }
    s
}

fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    Rng::seeded(seed).uniform_vec(len, -0.5, 0.5)
}

fn valid_vec(lens: &[usize], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(lens.len() * n);
    for &len in lens {
        for p in 0..n {
            out.push(if p < len { 1.0 } else { 0.0 });
        }
    }
    out
}

fn upload(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, f32> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

fn apply_stack(kind: Kind, seed: u64, x: &Tensor<R, f32>, valid: &Tensor<R, f32>) -> Vec<f32> {
    let device = x.device().clone();
    let mut rng = Rng::seeded(seed);
    let xv = Var::constant(x.clone());
    let out = match kind {
        Kind::Set => SetEncoderStackConfig {
            d_model: D,
            n_blocks: 2,
            hidden_dim: 16,
        }
        .init(&device, &mut rng)
        .unwrap()
        .apply(&xv, valid)
        .unwrap(),
        Kind::Transformer => TransformerEncoderStackConfig {
            d_model: D,
            n_blocks: 2,
            n_heads: 2,
            hidden_dim: 16,
        }
        .init(&device, &mut rng)
        .unwrap()
        .apply(&xv, valid)
        .unwrap(),
        Kind::Uni => UnidirectionalMambaStackConfig {
            ssm: small_ssm(D),
            n_blocks: 1,
        }
        .init(&device, &mut rng)
        .unwrap()
        .apply(&xv, valid)
        .unwrap(),
    };
    check_launches(&device).unwrap();
    out.try_to_f32().unwrap()
}

fn run_case(kind: Kind, seed: u64, lens: &[usize], poison: Option<f32>) -> Vec<f32> {
    let device = dev();
    let b = lens.len();
    let mut x = random_vec(b * N * D, seed);
    if let Some(p) = poison {
        for bi in 0..b {
            for pos in lens[bi]..N {
                for c in 0..D {
                    x[(bi * N + pos) * D + c] = p;
                }
            }
        }
    }
    let xt = upload(&x, vec![b, N, D], &device);
    let vt = upload(&valid_vec(lens, N), vec![b, N], &device);
    apply_stack(kind, 7, &xt, &vt)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.abs() <= 1.0 {
            (a - e).abs() <= tol
        } else {
            (a - e).abs() <= tol * e.abs()
        };
        assert!(ok, "{what}: index {i} got {a}, want {e}");
    }
}

fn assert_bits_equal(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits(),
            "{what}: index {i} differs: {a} vs {e}"
        );
    }
}

#[test]
fn shapes_exact_zeros_and_finite() {
    for kind in KINDS {
        let out = run_case(kind, 11, &[N, 3], None);
        assert_eq!(out.len(), 2 * N * D, "{}: shape", kind_name(kind));
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{}: non-finite output",
            kind_name(kind)
        );
        for pos in 3..N {
            for c in 0..D {
                let v = out[(N + pos) * D + c];
                assert!(
                    v.to_bits() == 0,
                    "{}: padding (1,{pos}) not +0.0: {v}",
                    kind_name(kind)
                );
            }
        }
    }
}

#[test]
fn padding_independence_with_poison() {
    for kind in KINDS {
        let name = kind_name(kind);
        let clean = run_case(kind, 21, &[5, 2], None);
        let nan = run_case(kind, 21, &[5, 2], Some(f32::NAN));
        let huge = run_case(kind, 21, &[5, 2], Some(1e30));
        assert_bits_equal(&nan, &clean, &format!("{name}: NaN poison changed outputs"));
        assert_bits_equal(&huge, &clean, &format!("{name}: huge poison changed outputs"));
    }
}

#[test]
fn alone_vs_batch_and_batch_permutation() {
    for kind in KINDS {
        let name = kind_name(kind);
        let device = dev();
        // One spectrum alone (len 5).
        let x_single = random_vec(N * D, 31);
        let v_single = valid_vec(&[5], N);
        let single = apply_stack(
            kind,
            7,
            &upload(&x_single, vec![1, N, D], &device),
            &upload(&v_single, vec![1, N], &device),
        );
        // The same spectrum at position 1 of a batch of 3.
        let mut xb = random_vec(3 * N * D, 32);
        xb[N * D..2 * N * D].copy_from_slice(&x_single);
        let batch = apply_stack(
            kind,
            7,
            &upload(&xb, vec![3, N, D], &device),
            &upload(&valid_vec(&[2, 5, 4], N), vec![3, N], &device),
        );
        assert_close(
            &batch[N * D..2 * N * D],
            &single,
            1e-5,
            &format!("{name}: alone vs in-batch"),
        );
        // Permuting the batch permutes the outputs.
        let order = [2usize, 0, 1];
        let mut xp = vec![0.0f32; 3 * N * D];
        let mut vp = vec![0.0f32; 3 * N];
        let vall = valid_vec(&[2, 5, 4], N);
        for (nb, &ob) in order.iter().enumerate() {
            xp[nb * N * D..(nb + 1) * N * D]
                .copy_from_slice(&xb[ob * N * D..(ob + 1) * N * D]);
            vp[nb * N..(nb + 1) * N].copy_from_slice(&vall[ob * N..(ob + 1) * N]);
        }
        let perm = apply_stack(
            kind,
            7,
            &upload(&xp, vec![3, N, D], &device),
            &upload(&vp, vec![3, N], &device),
        );
        for (nb, &ob) in order.iter().enumerate() {
            assert_close(
                &perm[nb * N * D..(nb + 1) * N * D],
                &batch[ob * N * D..(ob + 1) * N * D],
                1e-5,
                &format!("{name}: batch permutation row {nb}"),
            );
        }
    }
}

#[test]
fn set_stack_permutation_equivariance() {
    let device = dev();
    let x = random_vec(N * D, 41);
    let valid = valid_vec(&[N], N);
    let base = apply_stack(
        Kind::Set,
        7,
        &upload(&x, vec![1, N, D], &device),
        &upload(&valid, vec![1, N], &device),
    );
    let perm = [5usize, 3, 0, 4, 1, 2];
    let mut xp = vec![0.0f32; N * D];
    for (np, &op) in perm.iter().enumerate() {
        xp[np * D..(np + 1) * D].copy_from_slice(&x[op * D..(op + 1) * D]);
    }
    let moved = apply_stack(
        Kind::Set,
        7,
        &upload(&xp, vec![1, N, D], &device),
        &upload(&valid, vec![1, N], &device),
    );
    for (np, &op) in perm.iter().enumerate() {
        assert_close(
            &moved[np * D..(np + 1) * D],
            &base[op * D..(op + 1) * D],
            1e-5,
            "set: permuted output row {np}",
        );
    }
    // The masked mean itself does not move.
    let mean_of = |data: &[f32]| {
        let xv = Var::constant(upload(data, vec![1, N, D], &device));
        let vt = upload(&valid, vec![1, N], &device);
        masked_mean(&xv, &vt).unwrap().try_to_f32().unwrap()
    };
    assert_close(&mean_of(&xp), &mean_of(&x), 1e-5, "set: masked mean");
}

#[test]
fn masked_mean_pooled_vector_is_permutation_invariant() {
    // The set stack itself is permutation-EQUIVARIANT (rows move with their
    // inputs, tested above); only the masked-mean pooled spectrum vector is
    // permutation-INVARIANT.
    let device = dev();
    let x = random_vec(N * D, 41);
    let valid = valid_vec(&[N], N);
    let mean_of = |data: &[f32]| {
        let xv = Var::constant(upload(data, vec![1, N, D], &device));
        let vt = upload(&valid, vec![1, N], &device);
        masked_mean(&xv, &vt).unwrap().try_to_f32().unwrap()
    };
    let base = mean_of(&x);
    for perm in [
        [5usize, 3, 0, 4, 1, 2],
        [1usize, 0, 2, 3, 4, 5],
        [5usize, 4, 3, 2, 1, 0],
    ] {
        let mut xp = vec![0.0f32; N * D];
        for (np, &op) in perm.iter().enumerate() {
            xp[np * D..(np + 1) * D].copy_from_slice(&x[op * D..(op + 1) * D]);
        }
        assert_close(&mean_of(&xp), &base, 1e-5, "pooled masked mean under permutation");
    }
}

#[test]
fn transformer_stack_permutation_equivariance() {
    let device = dev();
    let x = random_vec(N * D, 51);
    let valid = valid_vec(&[N], N);
    let base = apply_stack(
        Kind::Transformer,
        7,
        &upload(&x, vec![1, N, D], &device),
        &upload(&valid, vec![1, N], &device),
    );
    let perm = [4usize, 0, 5, 2, 1, 3];
    let mut xp = vec![0.0f32; N * D];
    for (np, &op) in perm.iter().enumerate() {
        xp[np * D..(np + 1) * D].copy_from_slice(&x[op * D..(op + 1) * D]);
    }
    let moved = apply_stack(
        Kind::Transformer,
        7,
        &upload(&xp, vec![1, N, D], &device),
        &upload(&valid, vec![1, N], &device),
    );
    for (np, &op) in perm.iter().enumerate() {
        assert_close(
            &moved[np * D..(np + 1) * D],
            &base[op * D..(op + 1) * D],
            1e-5,
            "transformer: permuted output row {np}",
        );
    }
}

#[test]
fn unidirectional_stack_is_causal() {
    let device = dev();
    let n = 8usize;
    let x = random_vec(n * D, 61);
    let valid = valid_vec(&[n], n);
    let vt = upload(&valid, vec![1, n], &device);
    // Change peak j = 5 only. Positions before j must be bit-identical.
    let mut xm = x.clone();
    for c in 0..D {
        xm[5 * D + c] += 2.0;
    }
    // Two bit-identical stacks (same seed): one sees the base input, the
    // other the mutated one.
    let mut rng = Rng::seeded(7);
    let xv = Var::constant(upload(&xm, vec![1, n, D], &device));
    let stack = UnidirectionalMambaStackConfig {
        ssm: small_ssm(D),
        n_blocks: 1,
    }
    .init(&device, &mut rng)
    .unwrap();
    // Rebuild the base with the same weights for a fair comparison.
    let mut rng2 = Rng::seeded(7);
    let stack2 = UnidirectionalMambaStackConfig {
        ssm: small_ssm(D),
        n_blocks: 1,
    }
    .init(&device, &mut rng2)
    .unwrap();
    let xb = Var::constant(upload(&x, vec![1, n, D], &device));
    let base8 = stack2.apply(&xb, &vt).unwrap().try_to_f32().unwrap();
    let moved = stack.apply(&xv, &vt).unwrap().try_to_f32().unwrap();
    assert_bits_equal(&base8[..5 * D], &moved[..5 * D], "uni: prefix before peak 5");
    assert!(
        base8[5 * D..6 * D]
            .iter()
            .zip(&moved[5 * D..6 * D])
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "uni: peak 5 left its own output unchanged (vacuous test)"
    );
    check_launches(&device).unwrap();
}

fn finite_difference_check(
    device: &Device<R>,
    what: &str,
    params: &[(String, mamba3::nn::Param<R, f32>)],
    apply: &dyn Fn() -> Var<R, f32>,
    targets: &[&str],
) {
    let loss = apply().sum().unwrap();
    check_launches(device).unwrap();
    let grads = loss.backward_retain().unwrap();
    let find = |name: &str| {
        params
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("missing param {name}"))
            .1
            .clone()
    };
    for name in targets {
        let param = find(name);
        let analytic = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("no gradient for {name}"))
            .try_to_f32()
            .unwrap();
        let shape = param.shape().dims().to_vec();
        let base = param.value().try_to_f32().unwrap();
        for &idx in &[0usize, 1, 2] {
            assert!(idx < base.len(), "{name} too small");
            let mut up = base.clone();
            up[idx] += 1e-2;
            param.set(Tensor::<R, f32>::from_f32(&up, shape.clone(), device).unwrap());
            let fu = apply().sum().unwrap().try_to_f32().unwrap()[0];
            let mut down = base.clone();
            down[idx] -= 1e-2;
            param.set(Tensor::<R, f32>::from_f32(&down, shape.clone(), device).unwrap());
            let fd = apply().sum().unwrap().try_to_f32().unwrap()[0];
            param.set(Tensor::<R, f32>::from_f32(&base, shape.clone(), device).unwrap());
            let numeric = (fu - fd) / 2e-2;
            let a = analytic[idx];
            assert!(
                (a - numeric).abs() <= 3e-2 * numeric.abs() + 2e-3,
                "{what} {name}[{idx}]: analytic={a} numeric={numeric}"
            );
        }
    }
    check_launches(device).unwrap();
}

#[test]
fn set_stack_gradients_match_finite_differences() {
    let device = dev();
    let (b, n) = (1usize, 4usize);
    let xt = upload(&random_vec(b * n * D, 71), vec![b, n, D], &device);
    let vt = upload(&valid_vec(&[n], n), vec![b, n], &device);
    let mut rng = Rng::seeded(7);
    let stack = SetEncoderStackConfig {
        d_model: D,
        n_blocks: 1,
        hidden_dim: 16,
    }
    .init(&device, &mut rng)
    .unwrap();
    let params = stack.named_parameters();
    let apply = || {
        stack
            .apply(&Var::constant(xt.clone()), &vt)
            .unwrap()
    };
    finite_difference_check(
        &device,
        "set",
        &params,
        &apply,
        &[
            "blocks.0.up.weight",
            "blocks.0.down.weight",
            "blocks.0.norm.weight",
        ],
    );
}

#[test]
fn transformer_stack_gradients_match_finite_differences() {
    let device = dev();
    let (b, n) = (1usize, 4usize);
    let xt = upload(&random_vec(b * n * D, 71), vec![b, n, D], &device);
    let vt = upload(&valid_vec(&[n], n), vec![b, n], &device);
    let mut rng = Rng::seeded(7);
    let stack = TransformerEncoderStackConfig {
        d_model: D,
        n_blocks: 1,
        n_heads: 2,
        hidden_dim: 16,
    }
    .init(&device, &mut rng)
    .unwrap();
    let params = stack.named_parameters();
    let apply = || {
        stack
            .apply(&Var::constant(xt.clone()), &vt)
            .unwrap()
    };
    finite_difference_check(
        &device,
        "transformer",
        &params,
        &apply,
        &[
            "blocks.0.q.weight",
            "blocks.0.up.weight",
            "blocks.0.norm1.weight",
        ],
    );
}

#[test]
fn unidirectional_stack_gradients_match_finite_differences() {
    let device = dev();
    let (b, n) = (1usize, 4usize);
    let xt = upload(&random_vec(b * n * D, 71), vec![b, n, D], &device);
    let vt = upload(&valid_vec(&[n], n), vec![b, n], &device);
    let mut rng = Rng::seeded(7);
    let stack = UnidirectionalMambaStackConfig {
        ssm: small_ssm(D),
        n_blocks: 1,
    }
    .init(&device, &mut rng)
    .unwrap();
    let params = stack.named_parameters();
    let apply = || {
        stack
            .apply(&Var::constant(xt.clone()), &vt)
            .unwrap()
    };
    finite_difference_check(
        &device,
        "uni",
        &params,
        &apply,
        &[
            "blocks.0.mixer.in_proj.weight",
            "blocks.0.mixer.out_proj.weight",
            "blocks.0.norm.weight",
        ],
    );
}

#[test]
fn parameter_matching_within_five_percent_at_v0() {
    let v0 = ModelConfig::v0();
    let target = bidirectional_reference_params(&v0);
    assert_eq!(mamba3_block_params(&v0.encoder), 140_716);
    assert_eq!(target, 562_864);
    let within = |count: usize| {
        let diff = count.abs_diff(target) as f64 / target as f64;
        assert!(
            diff <= 0.05,
            "count {count} differs from {target} by {diff:.4}"
        );
        diff
    };
    let set_cfg = SetEncoderStackConfig::matched(target, 128);
    assert_eq!((set_cfg.n_blocks, set_cfg.hidden_dim), (6, 240));
    assert_eq!(set_cfg.params(), 555_936);
    let d_set = within(set_cfg.params());
    let tf_cfg = TransformerEncoderStackConfig::matched(target, 128);
    assert_eq!((tf_cfg.n_blocks, tf_cfg.n_heads, tf_cfg.hidden_dim), (3, 4, 480));
    assert_eq!(tf_cfg.params(), 567_840);
    let d_tf = within(tf_cfg.params());
    let uni_cfg = UnidirectionalMambaStackConfig::matched(target, 128);
    assert_eq!(uni_cfg.n_blocks, 4);
    assert_eq!(
        UnidirectionalMambaStackConfig {
            ssm: v0.encoder.clone(),
            n_blocks: 4
        }
        .params(),
        target
    );
    let d_uni = within(uni_cfg.params());
    // The analytic counts equal the live modules' counts.
    let device = dev();
    let mut rng = Rng::seeded(3);
    let set_live = set_cfg.init::<R, f32>(&device, &mut rng).unwrap();
    assert_eq!(set_live.num_parameters(), set_cfg.params());
    let tf_live = tf_cfg.init::<R, f32>(&device, &mut rng).unwrap();
    assert_eq!(tf_live.num_parameters(), tf_cfg.params());
    let uni_live = UnidirectionalMambaStackConfig {
        ssm: v0.encoder.clone(),
        n_blocks: 4,
    }
    .init::<R, f32>(&device, &mut rng)
    .unwrap();
    assert_eq!(uni_live.num_parameters(), target);
    println!("matched diffs: set={d_set:.4} transformer={d_tf:.4} uni={d_uni:.4}");
}

#[test]
fn mamba3_block_params_match_live_across_options() {
    use mamba3::models::mamba3::Mamba3BlockConfig;
    // The output projection bias has d_model entries (not d_inner): with
    // bias on, the analytic count must equal the live module's count, as it
    // must with bias off and across other optional settings (convolution,
    // B/C bias, skip, norms).
    let device = dev();
    let base = small_ssm(D);
    let mut variants = vec![
        ("bias off", {
            let mut s = base.clone();
            s.bias = false;
            s
        }),
        ("bias on", {
            let mut s = base.clone();
            s.bias = true;
            s
        }),
    ];
    // Two more optional settings flipped relative to the small config.
    let mut with_conv = base.clone();
    with_conv.conv_kernel = Some(3);
    variants.push(("conv 3", with_conv));
    let mut no_bc_bias = base.clone();
    no_bc_bias.bc_bias = false;
    variants.push(("no bc_bias", no_bc_bias));
    let mut no_skip = base.clone();
    no_skip.skip_connection = false;
    variants.push(("no skip", no_skip));
    let mut gate_norm = base.clone();
    gate_norm.post_gate_norm = true;
    variants.push(("post_gate_norm", gate_norm));
    for (name, ssm) in &variants {
        let analytic = mamba3_block_params(ssm);
        let mut rng = Rng::seeded(5);
        // Mamba3BlockConfig::init validates the SSM config.
        let live = Mamba3BlockConfig::new(ssm.clone())
            .init::<R, f32>(&device, &mut rng)
            .unwrap();
        assert_eq!(
            live.num_parameters(),
            analytic,
            "{name}: analytic {analytic} != live {}",
            live.num_parameters()
        );
    }
    // Bias on adds exactly in_w + d_model over bias off.
    let (mut off, mut on) = (base.clone(), base.clone());
    off.bias = false;
    on.bias = true;
    assert_eq!(
        mamba3_block_params(&on) - mamba3_block_params(&off),
        on.in_proj_width() + on.d_model,
        "output bias width must be d_model"
    );
}

#[test]
fn all_padding_spectrum_is_exact_zeros() {
    for kind in KINDS {
        let name = kind_name(kind);
        let out = run_case(kind, 81, &[0, 3], None);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{name}: non-finite with an all-padding spectrum"
        );
        for pos in 0..N {
            for c in 0..D {
                let v = out[pos * D + c];
                assert!(
                    v.to_bits() == 0,
                    "{name}: all-padding output ({pos}) not +0.0: {v}"
                );
            }
        }
        let empty = run_case(kind, 82, &[0], None);
        assert!(
            empty.iter().all(|v| v.to_bits() == 0 && v.is_finite()),
            "{name}: len-0 batch not exact zeros"
        );
    }
}

#[test]
fn stack_kind_serde_roundtrip() {
    for kind in [
        EncoderStackKind::Mamba,
        EncoderStackKind::UnidirectionalMamba,
        EncoderStackKind::Set,
        EncoderStackKind::Transformer,
    ] {
        let s = serde_json::to_string(&kind).unwrap();
        let back: EncoderStackKind = serde_json::from_str(&s).unwrap();
        assert_eq!(back, kind);
    }
}
