//! Gradient checks against central finite differences.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::{Shape, Tensor};

type R = Auto;
type V = Var<R, f32>;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Compare the analytic gradient of `f` at `x` with central differences.
fn check_grad<F>(name: &str, data: &[f32], shape: impl Into<Shape> + Clone, f: F)
where
    F: Fn(&V) -> V,
{
    let shape = shape.into();
    let x = V::traced(Tensor::from_f32(data, shape.clone(), &dev()).unwrap());
    let y = f(&x);
    assert_eq!(y.shape().num_elements(), 1, "{name}: f must return a scalar");
    let grads = y.backward_retain().unwrap();
    let analytic = grads
        .node(x.node().unwrap())
        .unwrap_or_else(|| panic!("{name}: no gradient reached the input"))
        .to_f32();

    let eps = 1e-3f32;
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let fp = f(&V::constant(
            Tensor::from_f32(&plus, shape.clone(), &dev()).unwrap(),
        ))
        .to_f32()[0];
        let fm = f(&V::constant(
            Tensor::from_f32(&minus, shape.clone(), &dev()).unwrap(),
        ))
        .to_f32()[0];
        let numeric = (fp - fm) / (2.0 * eps);
        let tol = 2e-2 * (1.0 + numeric.abs());
        assert!(
            (analytic[i] - numeric).abs() < tol,
            "{name}: grad[{i}] analytic={} numeric={}",
            analytic[i],
            numeric
        );
    }
}

#[test]
fn elementwise_gradients() {
    let data = [0.3f32, -0.7, 1.2, 2.0, -1.5, 0.05];
    check_grad("exp", &data, vec![2, 3], |x| x.exp().sum().unwrap());
    check_grad("sigmoid", &data, vec![2, 3], |x| x.sigmoid().sum().unwrap());
    check_grad("tanh", &data, vec![2, 3], |x| x.tanh().sum().unwrap());
    check_grad("silu", &data, vec![2, 3], |x| {
        x.silu().unwrap().sum().unwrap()
    });
    check_grad("gelu", &data, vec![2, 3], |x| {
        x.gelu().unwrap().sum().unwrap()
    });
    check_grad("softplus", &data, vec![2, 3], |x| {
        x.softplus().unwrap().sum().unwrap()
    });
    check_grad("abs", &data, vec![2, 3], |x| x.abs().sum().unwrap());
    check_grad("square", &data, vec![2, 3], |x| {
        x.mul(x).unwrap().sum().unwrap()
    });
    check_grad("recip", &data, vec![2, 3], |x| {
        x.add_scalar(3.0).recip().sum().unwrap()
    });
    check_grad("sqrt", &data, vec![2, 3], |x| {
        x.add_scalar(3.0).sqrt().sum().unwrap()
    });
    check_grad("rsqrt", &data, vec![2, 3], |x| {
        x.add_scalar(3.0).rsqrt().sum().unwrap()
    });
    check_grad("sin", &data, vec![2, 3], |x| x.sin().sum().unwrap());
    check_grad("powf", &data, vec![2, 3], |x| {
        x.add_scalar(3.0).powf_scalar(1.7).sum().unwrap()
    });
}

#[test]
fn reduction_and_movement_gradients() {
    let data = [0.3f32, -0.7, 1.2, 2.0, -1.5, 0.05];
    check_grad("sum_dim", &data, vec![2, 3], |x| {
        x.sum_dim(1).unwrap().mul(&x.sum_dim(1).unwrap()).unwrap().sum().unwrap()
    });
    check_grad("mean_dim", &data, vec![2, 3], |x| {
        x.mean_dim(0).unwrap().exp().sum().unwrap()
    });
    check_grad("max_dim", &data, vec![2, 3], |x| {
        x.max_dim(1).unwrap().exp().sum().unwrap()
    });
    check_grad("transpose", &data, vec![2, 3], |x| {
        x.transpose().unwrap().exp().sum().unwrap()
    });
    check_grad("flip", &data, vec![2, 3], |x| {
        x.flip(1).unwrap().mul_scalar(2.0).exp().sum().unwrap()
    });
    check_grad("slice", &data, vec![2, 3], |x| {
        x.slice(1, 1, 2).unwrap().exp().sum().unwrap()
    });
    check_grad("shift_right", &data, vec![2, 3], |x| {
        x.shift_right(1).unwrap().exp().sum().unwrap()
    });
    check_grad("cumsum", &data, vec![2, 3], |x| {
        x.cumsum(1).unwrap().exp().sum().unwrap()
    });
    check_grad("cumsum_exclusive", &data, vec![2, 3], |x| {
        x.cumsum_exclusive(1).unwrap().exp().sum().unwrap()
    });
    check_grad("expand", &data, vec![2, 1, 3], |x| {
        x.expand(vec![2, 4, 3]).unwrap().exp().sum().unwrap()
    });
    check_grad("softmax", &data, vec![2, 3], |x| {
        let p = x.softmax(1).unwrap();
        p.mul(&p).unwrap().sum().unwrap()
    });
    check_grad("log_softmax", &data, vec![2, 3], |x| {
        x.log_softmax(1).unwrap().exp().sum().unwrap()
    });
}

#[test]
fn matmul_gradient() {
    let a_data = [1.0f32, 2.0, -1.0, 0.5, 0.25, -2.0];
    let b = Tensor::from_f32(&[0.5f32, -1.0, 2.0, 1.5], vec![2, 2], &dev()).unwrap();
    check_grad("matmul_lhs", &a_data, vec![3, 2], |x| {
        x.matmul(&V::constant(b.clone()))
            .unwrap()
            .exp()
            .sum()
            .unwrap()
    });

    let a = Tensor::from_f32(&a_data, vec![3, 2], &dev()).unwrap();
    check_grad("matmul_rhs", &[0.5f32, -1.0, 2.0, 1.5], vec![2, 2], |x| {
        V::constant(a.clone()).matmul(x).unwrap().exp().sum().unwrap()
    });
}

#[test]
fn matmul_nt_matches_transpose_then_matmul() {
    let a_data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.03).collect();
    let b_data: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.025).collect();
    let a = Tensor::from_f32(&a_data, vec![2, 3, 4], &dev()).unwrap();
    let b = Tensor::from_f32(&b_data, vec![2, 5, 4], &dev()).unwrap();

    let composed = V::constant(a.clone())
        .matmul(&V::constant(b.clone()).transpose().unwrap())
        .unwrap();
    let fused = V::constant(a.clone()).matmul_nt(&V::constant(b.clone())).unwrap();
    assert_eq!(composed.shape(), fused.shape());
    let (want, got) = (composed.to_f32(), fused.to_f32());
    for (w, g) in want.iter().zip(&got) {
        assert!((w - g).abs() < 1e-4, "matmul_nt value mismatch: {w} vs {g}");
    }

    check_grad("matmul_nt_lhs", &a_data, vec![2, 3, 4], |x| {
        x.matmul_nt(&V::constant(b.clone())).unwrap().exp().sum().unwrap()
    });
    check_grad("matmul_nt_rhs", &b_data, vec![2, 5, 4], |x| {
        V::constant(a.clone()).matmul_nt(x).unwrap().exp().sum().unwrap()
    });

    // The tied-head shape: a batched activation against one shared `[rows, k]`
    // table, whose gradient is summed over the batch rather than kept per
    // sequence.
    let table_data: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.04).collect();
    let table = Tensor::from_f32(&table_data, vec![5, 4], &dev()).unwrap();
    let shared = V::constant(a.clone())
        .matmul_nt(&V::constant(table.clone()))
        .unwrap();
    let shared_composed = V::constant(a.clone())
        .matmul(&V::constant(table.clone()).transpose().unwrap())
        .unwrap();
    assert_eq!(shared.shape(), shared_composed.shape());
    for (w, g) in shared_composed.to_f32().iter().zip(&shared.to_f32()) {
        assert!((w - g).abs() < 1e-4, "matmul_nt (shared rhs) {w} vs {g}");
    }
    check_grad("matmul_nt_shared_rhs", &table_data, vec![5, 4], |x| {
        V::constant(a.clone()).matmul_nt(x).unwrap().exp().sum().unwrap()
    });
}

#[test]
fn broadcast_gradient() {
    let bias = [0.5f32, -1.0, 2.0];
    let base = Tensor::from_f32(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &dev()).unwrap();
    // `tanh` keeps the objective O(1); an `exp` here would put the finite
    // difference below f32 resolution and produce a bogus zero.
    check_grad("broadcast_add", &bias, vec![3], |x| {
        V::constant(base.clone())
            .add(x)
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });
    check_grad("broadcast_mul", &bias, vec![3], |x| {
        V::constant(base.clone())
            .mul(x)
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });
}

#[test]
fn cat_gradient() {
    let data = [0.3f32, -0.7, 1.2, 2.0];
    check_grad("cat", &data, vec![2, 2], |x| {
        let a = x.slice(1, 0, 1).unwrap();
        let b = x.mul_scalar(2.0);
        mamba3::autograd::cat(&[a, b], 1).unwrap().exp().sum().unwrap()
    });
}

#[test]
fn straight_through_round_is_identity_in_backward() {
    let x = V::traced(Tensor::from_f32(&[0.2f32, 1.7, -2.4], vec![3], &dev()).unwrap());
    let y = x.round_ste().mul_scalar(3.0).sum().unwrap();
    assert_eq!(y.to_f32()[0], (0.0 + 2.0 - 2.0) * 3.0);
    let grads = y.backward_retain().unwrap();
    let g = grads.node(x.node().unwrap()).unwrap().to_f32();
    assert_eq!(g, vec![3.0, 3.0, 3.0]);
}

#[test]
fn embedding_gradient_accumulates_repeats() {
    use mamba3::tensor::ops::index::IdTensor;

    let table = V::traced(
        Tensor::from_f32(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &dev()).unwrap(),
    );
    let ids = IdTensor::from_slice(&[0, 2, 0], vec![3], &dev()).unwrap();
    let out = mamba3::autograd::embedding(&table, &ids).unwrap();
    let loss = out.sum().unwrap();
    let grads = loss.backward_retain().unwrap();
    let g = grads.node(table.node().unwrap()).unwrap().to_f32();
    // row 0 used twice, row 1 never, row 2 once.
    assert_eq!(g, vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn graphs_merge_across_independent_roots() {
    let a = V::traced(Tensor::from_f32(&[2.0f32], vec![1], &dev()).unwrap());
    let b = V::traced(Tensor::from_f32(&[3.0f32], vec![1], &dev()).unwrap());
    let y = a.mul(&b).unwrap().sum().unwrap();
    let grads = y.backward_retain().unwrap();
    assert_eq!(grads.node(a.node().unwrap()).unwrap().to_f32(), vec![3.0]);
    assert_eq!(grads.node(b.node().unwrap()).unwrap().to_f32(), vec![2.0]);
}

#[test]
fn untracked_inputs_stay_cheap() {
    let x = V::constant(Tensor::from_f32(&[1.0f32, 2.0], vec![2], &dev()).unwrap());
    let y = x.exp().mul_scalar(2.0);
    assert!(!y.is_tracked());
    assert!(y.backward().is_err());
}

/// The fused RMS norm is the one composed op in the crate that carries a
/// hand-written adjoint, so it gets a hand-written check: both the input and the
/// gain gradient against central differences, with and without a gain.
#[test]
fn fused_rms_norm_gradients() {
    use mamba3::nn::RmsNormConfig;
    use mamba3::tensor::ops::random::Rng;

    let data = [0.7f32, -1.3, 2.1, 0.4, -0.9, 1.6, 0.2, -2.4];

    // Gradient with respect to the input, gain on and off.
    for gain in [true, false] {
        let norm = RmsNormConfig::new(4)
            .with_weight(gain)
            .init::<R, f32>(&dev(), &mut Rng::seeded(7));
        // A non-trivial gain, so its contribution cannot cancel out.
        if let Some(w) = norm.weight() {
            w.set(Tensor::from_f32(&[1.3f32, 0.6, -0.8, 1.1], vec![4], &dev()).unwrap());
        }
        check_grad(
            if gain { "rms_norm(x) with gain" } else { "rms_norm(x)" },
            &data,
            vec![2, 4],
            |x| norm.apply(x).unwrap().tanh().sum().unwrap(),
        );
    }

    // Gradient with respect to the gain: hold the input fixed and differentiate the
    // same objective with the gain as the traced variable.
    let x = Tensor::from_f32(&data, vec![2, 4], &dev()).unwrap();
    let norm = RmsNormConfig::new(4).init::<R, f32>(&dev(), &mut Rng::seeded(7));
    let weight = norm.weight().expect("gain requested");
    let gain_data = [1.3f32, 0.6, -0.8, 1.1];
    let eps = 1e-3f32;
    weight.set(Tensor::from_f32(&gain_data, vec![4], &dev()).unwrap());
    let y = norm
        .apply(&V::constant(x.clone()))
        .unwrap()
        .tanh()
        .sum()
        .unwrap();
    let grads = y.backward_retain().unwrap();
    let analytic = grads
        .get(weight.id())
        .expect("the gain is a parameter of the graph")
        .to_f32();
    for i in 0..gain_data.len() {
        let mut plus = gain_data;
        let mut minus = gain_data;
        plus[i] += eps;
        minus[i] -= eps;
        let at = |g: &[f32]| {
            weight.set(Tensor::from_f32(g, vec![4], &dev()).unwrap());
            norm.apply(&V::constant(x.clone()))
                .unwrap()
                .tanh()
                .sum()
                .unwrap()
                .to_f32()[0]
        };
        let numeric = (at(&plus) - at(&minus)) / (2.0 * eps);
        assert!(
            (analytic[i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
            "d gain[{i}]: analytic={} numeric={numeric}",
            analytic[i]
        );
    }
}

/// The fused rotation's adjoint, against central differences — for the tensor and
/// for both angle tables, which the Mamba-3 scan learns through `theta`.
#[test]
fn fused_rotation_gradients() {
    let x = [0.3f32, -0.7, 1.2, 2.0, -1.5, 0.05, 0.9, -0.2];
    let table = [0.4f32, -1.1, 0.8, 0.25];

    // `[2, 4]` against `[2, 2]`: no broadcast.
    let angles = |data: &[f32]| V::constant(Tensor::from_f32(data, vec![2, 2], &dev()).unwrap());
    check_grad("rotate d/dx", &x, vec![2, 4], |v| {
        v.rotate_halves(&angles(&table), &angles(&[0.6f32, 0.1, -0.9, 1.4]))
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });

    // `[2, 3, 4]` against `[2, 1, 2]`: the rotating-frame broadcast, differentiated
    // with respect to the table, so the reduction back down to it is exercised.
    let field: Vec<f32> = (0..24).map(|i| ((i % 9) as f32 - 4.0) * 0.2).collect();
    let big = V::constant(Tensor::from_f32(&field, vec![2, 3, 4], &dev()).unwrap());
    let other = V::constant(Tensor::from_f32(&[0.2f32, -0.5, 1.0, 0.7], vec![2, 1, 2], &dev()).unwrap());
    check_grad("rotate d/dcos", &table, vec![2, 1, 2], |c| {
        big.rotate_halves(c, &other).unwrap().tanh().sum().unwrap()
    });
    check_grad("rotate d/dsin", &table, vec![2, 1, 2], |s| {
        big.rotate_halves(&other, s).unwrap().tanh().sum().unwrap()
    });
}

/// Every gradient the fused convolution produces — input, taps, bias and carried
/// history — against central differences.
#[test]
fn fused_causal_conv_gradients() {
    use mamba3::nn::conv::CausalConv1dConfig;
    use mamba3::tensor::ops::random::Rng;

    let channels = 3;
    let taps = 3;
    let carry = taps - 1;
    let seq = 4;
    let batch = 2;
    let conv = CausalConv1dConfig::new(channels, taps)
        .with_bias(true)
        .init::<R, f32>(&dev(), &mut Rng::seeded(11));

    let hn = batch * carry * channels;
    let hdata: Vec<f32> = (0..hn).map(|i| ((i % 5) as f32 - 2.0) * 0.35).collect();
    let history = V::constant(Tensor::from_f32(&hdata, vec![batch, carry, channels], &dev()).unwrap());

    let n = batch * seq * channels;
    let xdata: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();

    // Input, through both outputs so the carried history's adjoint is exercised too.
    check_grad("conv d/dx", &xdata, vec![batch, seq, channels], |x| {
        let (out, next) = conv.apply_with_history(x, &history).unwrap();
        out.tanh().sum().unwrap().add(&next.tanh().sum().unwrap()).unwrap()
    });

    // History, likewise.
    check_grad("conv d/dhistory", &hdata, vec![batch, carry, channels], |h| {
        let x = V::constant(Tensor::from_f32(&xdata, vec![batch, seq, channels], &dev()).unwrap());
        let (out, next) = conv.apply_with_history(&x, h).unwrap();
        out.tanh().sum().unwrap().add(&next.tanh().sum().unwrap()).unwrap()
    });

    // Taps and bias: differentiate the parameters by finite differences on the host.
    let x = Tensor::from_f32(&xdata, vec![batch, seq, channels], &dev()).unwrap();
    let objective = || {
        conv.apply_with_history(&V::constant(x.clone()), &history)
            .unwrap()
            .0
            .tanh()
            .sum()
            .unwrap()
    };
    let grads = objective().backward_retain().unwrap();
    let eps = 1e-3f32;
    for (name, param, shape) in [
        ("weight", conv.weight(), vec![taps, channels]),
        ("bias", conv.bias().expect("bias requested"), vec![channels]),
    ] {
        let base = param.value().to_f32();
        let analytic = grads.get(param.id()).expect("parameter reached").to_f32();
        for i in 0..base.len() {
            let at = |delta: f32| {
                let mut d = base.clone();
                d[i] += delta;
                param.set(Tensor::from_f32(&d, shape.clone(), &dev()).unwrap());
                objective().to_f32()[0]
            };
            let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
            param.set(Tensor::from_f32(&base, shape.clone(), &dev()).unwrap());
            assert!(
                (analytic[i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
                "conv d/d{name}[{i}]: analytic={} numeric={numeric}",
                analytic[i]
            );
        }
    }
}

/// The fused SiLU and the fused state update, both against central differences and
/// against the composed forms they replace.
#[test]
fn fused_silu_and_state_update_gradients() {
    let data = [0.3f32, -0.7, 1.2, 2.0, -1.5, 0.05];

    let x = V::constant(Tensor::from_f32(&data, vec![2, 3], &dev()).unwrap());
    assert_eq!(x.silu().unwrap().to_f32(), x.silu_composed().unwrap().to_f32());
    check_grad("silu", &data, vec![2, 3], |v| v.silu().unwrap().sum().unwrap());

    let softplus_data = [-40.0f32, -3.0, -0.7, 0.0, 0.7, 3.0, 40.0];
    let x = V::constant(Tensor::from_f32(&softplus_data, vec![7], &dev()).unwrap());
    let (fused, composed) = (x.softplus().unwrap().to_f32(), x.softplus_composed().unwrap().to_f32());
    for (f, c) in fused.iter().zip(&composed) {
        assert!((f - c).abs() < 1e-4, "softplus fused={f} composed={c}");
    }
    assert!(fused.iter().all(|v| v.is_finite()), "softplus overflowed: {fused:?}");
    check_grad("softplus", &softplus_data[1..6], vec![5], |v| {
        v.softplus().unwrap().sum().unwrap()
    });

    // `[2, 2, 3]` states scaled by one coefficient per `(batch, head)`.
    let state: Vec<f32> = (0..12).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
    let others: Vec<Vec<f32>> = (1..3)
        .map(|k| (0..12).map(|i| ((i * k % 5) as f32 - 2.0) * 0.4).collect())
        .collect();
    let scales = [0.7f32, -0.4, 1.1, 0.2];
    let tensor = |d: &[f32]| V::constant(Tensor::from_f32(d, vec![2, 2, 3], &dev()).unwrap());
    let scale = |d: &[f32]| V::constant(Tensor::from_f32(d, vec![4], &dev()).unwrap());

    let composed = |x: &[&V; 3], s: &[&V; 3]| -> V {
        let shaped = |v: &V| v.reshape(vec![2, 2, 1]).unwrap();
        x[0].mul(&shaped(s[0]))
            .unwrap()
            .add(&x[1].mul(&shaped(s[1])).unwrap())
            .unwrap()
            .add(&x[2].mul(&shaped(s[2])).unwrap())
            .unwrap()
    };
    let (x0, x1, x2) = (tensor(&state), tensor(&others[0]), tensor(&others[1]));
    let packed: Vec<f32> = scales
        .iter()
        .chain(&[0.1f32, 0.9, -0.6, 0.5])
        .chain(&[1.0f32, 0.3, 0.8, -0.2])
        .copied()
        .collect();
    let all = V::constant(Tensor::from_f32(&packed, vec![3, 4], &dev()).unwrap());
    let (s0, s1, s2) = (scale(&packed[0..4]), scale(&packed[4..8]), scale(&packed[8..12]));
    let fused = V::ssm_state_update([&x0, &x1, &x2], &all).unwrap();
    let want = composed(&[&x0, &x1, &x2], &[&s0, &s1, &s2]);
    for (a, b) in fused.to_f32().iter().zip(want.to_f32()) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    check_grad("state update d/dx", &state, vec![2, 2, 3], |v| {
        V::ssm_state_update([v, &x1, &x2], &all)
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });
    check_grad("state update d/ds", &packed, vec![3, 4], |v| {
        V::ssm_state_update([&x0, &x1, &x2], v)
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });
}

/// The three Mamba-3 step kernels that carry hand-written adjoints across several
/// inputs: the per-head coefficients, the rotating frame's angle, and the rotation
/// that takes that angle directly.
#[test]
fn fused_step_kernels_match_and_differentiate() {
    let heads = 3usize;
    let batch = 2usize;
    let a_log = [0.2f32, -0.5, 0.8];
    let dt: Vec<f32> = (0..batch * heads).map(|i| 0.2 + 0.1 * i as f32).collect();
    let lambda: Vec<f32> = (0..batch * heads).map(|i| 0.3 + 0.05 * i as f32).collect();
    let av = |d: &[f32]| V::constant(Tensor::from_f32(d, vec![heads], &dev()).unwrap());
    let fv = |d: &[f32]| V::constant(Tensor::from_f32(d, vec![batch, heads], &dev()).unwrap());

    // Against the arithmetic it replaces.
    let packed = V::ssm_coefficients(&av(&a_log), &fv(&dt), &fv(&lambda))
        .unwrap()
        .to_f32();
    for b in 0..batch {
        for (h, log) in a_log.iter().enumerate() {
            let i = b * heads + h;
            let a = -log.exp();
            let alpha = (dt[i] * a).exp();
            for (k, want) in [
                alpha,
                (1.0 - lambda[i]) * dt[i] * alpha,
                lambda[i] * dt[i],
            ]
            .into_iter()
            .enumerate()
            {
                let got = packed[k * batch * heads + i];
                assert!((got - want).abs() < 1e-5, "coefficient {k}[{i}]: {got} != {want}");
            }
        }
    }

    check_grad("coefficients d/d a_log", &a_log, vec![heads], |v| {
        V::ssm_coefficients(v, &fv(&dt), &fv(&lambda))
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });
    check_grad("coefficients d/d dt", &dt, vec![batch, heads], |v| {
        V::ssm_coefficients(&av(&a_log), v, &fv(&lambda))
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });
    check_grad("coefficients d/d lambda", &lambda, vec![batch, heads], |v| {
        V::ssm_coefficients(&av(&a_log), &fv(&dt), v)
            .unwrap()
            .tanh()
            .sum()
            .unwrap()
    });

    // The angle advance, with and without a previous frame.
    let theta: Vec<f32> = (0..batch * heads * 2).map(|i| 0.4 + 0.3 * i as f32).collect();
    let prev: Vec<f32> = (0..batch * heads * 2).map(|i| 1.1 - 0.2 * i as f32).collect();
    let tv = |d: &[f32]| V::constant(Tensor::from_f32(d, vec![batch, heads, 2], &dev()).unwrap());
    let angle = V::ssm_angle(&fv(&dt), &tv(&theta), Some(&tv(&prev)))
        .unwrap()
        .to_f32();
    let two_pi = 2.0 * std::f32::consts::PI;
    for i in 0..theta.len() {
        let raw = prev[i] + dt[i / 2] * theta[i];
        let want = raw - (raw / two_pi).round() * two_pi;
        assert!((angle[i] - want).abs() < 1e-5, "angle[{i}]: {} != {want}", angle[i]);
    }
    check_grad("angle d/d theta", &theta, vec![batch, heads, 2], |v| {
        V::ssm_angle(&fv(&dt), v, Some(&tv(&prev))).unwrap().sum().unwrap()
    });
    check_grad("angle d/d dt", &dt, vec![batch, heads], |v| {
        V::ssm_angle(v, &tv(&theta), Some(&tv(&prev))).unwrap().sum().unwrap()
    });

    // Rotation straight from the angle, against the cos/sin form.
    let field: Vec<f32> = (0..batch * heads * 4).map(|i| ((i % 9) as f32 - 4.0) * 0.2).collect();
    let xv = |d: &[f32]| V::constant(Tensor::from_f32(d, vec![batch, heads, 4], &dev()).unwrap());
    let phi = tv(&theta);
    let composed = xv(&field)
        .rotate_halves_composed(&phi.cos(), &phi.sin().neg())
        .unwrap();
    let direct = xv(&field).rotate_by_angle(&phi).unwrap();
    for (a, b) in direct.to_f32().iter().zip(composed.to_f32()) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
    check_grad("rotate_by_angle d/dx", &field, vec![batch, heads, 4], |v| {
        v.rotate_by_angle(&phi).unwrap().tanh().sum().unwrap()
    });
    check_grad("rotate_by_angle d/dphi", &theta, vec![batch, heads, 2], |v| {
        xv(&field).rotate_by_angle(v).unwrap().tanh().sum().unwrap()
    });
}

/// The scan's fused band, against central differences in each of its three inputs.
///
/// `Var::ssd_band` replaces a broadcast subtract, a clamp, an exponential, two masked
/// multiplies and an add, and its adjoint is written rather than generated — so it is
/// checked directly here as well as through the scan. The cumulative decays are given
/// a spread wide enough that some entries land past the clamp's floor and some do not,
/// because the clamp's derivative is the part of the rule that is easy to get wrong.
#[test]
fn fused_ssd_band_gradients() {
    const FLOOR: f32 = -60.0;
    let rows = 2;
    let chunk = 4;

    // Decreasing along each row, as a cumulative sum of negative steps is, with the
    // second row falling far enough to cross the floor.
    let acum = [0.0f32, -0.4, -1.1, -1.9, 0.0, -20.0, -45.0, -70.0];
    let w = [0.6f32, -1.2, 0.9, 0.3, 1.4, -0.7, 0.5, -0.2];
    let g = [1.1f32, 0.4, -0.9, 0.7, -0.3, 1.2, 0.8, -1.5];
    let shape = vec![rows, chunk];

    // A non-symmetric objective, so no term can cancel another out.
    let weights: Vec<f32> = (0..rows * chunk * chunk)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.31 + 0.17)
        .collect();
    let mask = V::constant(
        Tensor::from_f32(&weights, vec![rows, chunk, chunk], &dev()).unwrap(),
    );

    let objective = |a: &V, wv: &V, gv: &V| -> V {
        V::ssd_band(a, wv, gv, FLOOR)
            .unwrap()
            .mul(&mask)
            .unwrap()
            .sum()
            .unwrap()
    };

    let cst = |d: &[f32]| V::constant(Tensor::from_f32(d, shape.clone(), &dev()).unwrap());
    for (name, data) in [("acum", &acum[..]), ("w", &w[..]), ("g", &g[..])] {
        check_grad(&format!("ssd_band d/d {name}"), data, shape.clone(), |v| {
            match name {
                "acum" => objective(v, &cst(&w), &cst(&g)),
                "w" => objective(&cst(&acum), v, &cst(&g)),
                _ => objective(&cst(&acum), &cst(&w), v),
            }
        });
    }
}

/// `Var::split` against the run of slices it replaces, including a piece that
/// receives no gradient at all.
///
/// The pieces share a sink node that assembles their bands into one buffer, and the
/// sink is only reached because each piece hands it an empty token — so the two
/// things worth checking are that the bands land in the right places and that a
/// piece nobody uses leaves zeros rather than stale values.
#[test]
fn split_gradient_matches_slices() {
    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.37 - 4.0).collect();
    let shape = vec![2, 3, 4];
    let sizes = [1usize, 2, 1];

    // A different nonlinearity per piece, so no two bands can be confused.
    let objective = |pieces: &[V]| -> V {
        pieces[0]
            .tanh()
            .sum()
            .unwrap()
            .add(&pieces[1].mul(&pieces[1]).unwrap().sum().unwrap())
            .unwrap()
            .add(&pieces[2].sigmoid().sum().unwrap())
            .unwrap()
    };

    check_grad("split, every piece used", &data, shape.clone(), |x| {
        objective(&x.split(&sizes, 2).unwrap())
    });
    check_grad("slices, every piece used", &data, shape.clone(), |x| {
        let pieces: Vec<V> = [(0, 1), (1, 2), (3, 1)]
            .iter()
            .map(|(start, len)| x.slice(2, *start, *len).unwrap())
            .collect();
        objective(&pieces)
    });

    // The middle piece is dropped on the floor: its band of the gradient must be
    // zero, not whatever the buffer happened to hold.
    check_grad("split, middle piece unused", &data, shape, |x| {
        let pieces = x.split(&sizes, 2).unwrap();
        pieces[0]
            .tanh()
            .sum()
            .unwrap()
            .add(&pieces[2].sigmoid().sum().unwrap())
            .unwrap()
    });
}

/// The scan's fused decay pattern `exp(clamp(a - b, floor, 0)) * m` against its
/// composed form, on the broadcast shapes the scan actually uses, and its
/// gradients away from the clamp's corners.
#[test]
fn fused_exp_decay_matches_composed_and_differentiates() {
    const FLOOR: f32 = -5.0;
    let close = |name: &str, fused: &V, composed: &V| {
        assert_eq!(fused.shape(), composed.shape(), "{name}: shape");
        for (f, c) in fused.to_f32().iter().zip(&composed.to_f32()) {
            assert!((f - c).abs() < 1e-6, "{name}: fused={f} composed={c}");
        }
    };

    // Values straddling both clamp bounds, in the "decay to chunk end" shape:
    // a `[2, 3, 1, 2]` broadcast against b and m at `[2, 3, 4, 2]`.
    let a_data: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.9).collect();
    let b_data: Vec<f32> = (0..48).map(|i| ((i * 7 % 13) as f32 - 6.0) * 1.1).collect();
    let m_data: Vec<f32> = (0..48).map(|i| ((i % 5) as f32 - 2.0) * 0.6).collect();
    let a = V::constant(Tensor::from_f32(&a_data, vec![2, 3, 1, 2], &dev()).unwrap());
    let b = V::constant(Tensor::from_f32(&b_data, vec![2, 3, 4, 2], &dev()).unwrap());
    let m = V::constant(Tensor::from_f32(&m_data, vec![2, 3, 4, 2], &dev()).unwrap());
    close(
        "decay to end",
        &V::exp_decay(&a, Some(&b), Some(&m), FLOOR).unwrap(),
        &V::exp_decay_composed(&a, Some(&b), Some(&m), FLOOR).unwrap(),
    );

    // The transfer-matrix shape: the operands broadcast against each other and
    // the mask is broadcast along the trailing axis.
    let q = V::constant(Tensor::from_f32(&a_data, vec![2, 1, 3, 2], &dev()).unwrap());
    let p = V::constant(Tensor::from_f32(&a_data, vec![2, 3, 1, 2], &dev()).unwrap());
    let mask_data: Vec<f32> = (0..9).map(|i| ((i % 3) != 0) as u8 as f32).collect();
    let mask = V::constant(Tensor::from_f32(&mask_data, vec![1, 3, 3, 1], &dev()).unwrap());
    close(
        "transfer",
        &V::exp_decay(&p, Some(&q), Some(&mask), FLOOR).unwrap(),
        &V::exp_decay_composed(&p, Some(&q), Some(&mask), FLOOR).unwrap(),
    );

    // b and m omitted: plain exp(clamp(a, floor, 0)).
    close(
        "single operand",
        &V::exp_decay(&b, None, None, FLOOR).unwrap(),
        &V::exp_decay_composed(&b, None, None, FLOOR).unwrap(),
    );

    // Gradients, with the difference held strictly inside (FLOOR, 0) so the
    // central difference never steps across the clamp's kink.
    let ga: Vec<f32> = (0..6).map(|i| -1.5 + 0.16 * (i as f32)).collect();
    let gb: Vec<f32> = (0..24).map(|i| 0.6 + 0.07 * ((i * 5 % 11) as f32)).collect();
    let gm: Vec<f32> = (0..24).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let ac = V::constant(Tensor::from_f32(&ga, vec![2, 1, 3], &dev()).unwrap());
    let bc = V::constant(Tensor::from_f32(&gb, vec![2, 4, 3], &dev()).unwrap());
    let mc = V::constant(Tensor::from_f32(&gm, vec![2, 4, 3], &dev()).unwrap());
    check_grad("exp_decay d/da", &ga, vec![2, 1, 3], |v| {
        V::exp_decay(v, Some(&bc), Some(&mc), FLOOR).unwrap().sum().unwrap()
    });
    check_grad("exp_decay d/db", &gb, vec![2, 4, 3], |v| {
        V::exp_decay(&ac, Some(v), Some(&mc), FLOOR).unwrap().sum().unwrap()
    });
    check_grad("exp_decay d/dm", &gm, vec![2, 4, 3], |v| {
        V::exp_decay(&ac, Some(&bc), Some(v), FLOOR).unwrap().sum().unwrap()
    });
    let solo: Vec<f32> = (0..6).map(|i| -4.2 + 0.7 * (i as f32)).collect();
    check_grad("exp_decay d/da (solo)", &solo, vec![2, 3], |v| {
        V::exp_decay(v, None, None, FLOOR).unwrap().sum().unwrap()
    });
}

/// The fused trapezoid weights against the composed shift-and-add, and their
/// gradients — including through the cross term where `w` reads position `t + 1`.
#[test]
fn fused_trapezoid_weights_match_composed_and_differentiate() {
    let (batch, seq, heads) = (2usize, 4, 3);
    let n = batch * seq * heads;
    let lam_data: Vec<f32> = (0..n).map(|i| 0.1 + 0.8 * ((i * 5 % 9) as f32 / 9.0)).collect();
    let dt_data: Vec<f32> = (0..n).map(|i| 0.05 + ((i * 3 % 7) as f32) * 0.12).collect();
    let lam = V::constant(Tensor::from_f32(&lam_data, vec![batch, seq, heads], &dev()).unwrap());
    let dt = V::constant(Tensor::from_f32(&dt_data, vec![batch, seq, heads], &dev()).unwrap());

    let (g, w) = V::trapezoid_weights(&lam, &dt).unwrap();
    let g_composed = lam.mul(&dt).unwrap();
    let next = lam.rsub_scalar(1.0).mul(&dt).unwrap();
    let tail = next.slice(1, 1, seq - 1).unwrap();
    let pad = V::constant(Tensor::zeros(vec![batch, 1, heads], &dev()));
    let shifted = mamba3::autograd::cat(&[tail, pad], 1).unwrap();
    let w_composed = g_composed.add(&shifted).unwrap();
    for (f, c) in g.to_f32().iter().zip(&g_composed.to_f32()) {
        assert!((f - c).abs() < 1e-6, "trapezoid g fused={f} composed={c}");
    }
    for (f, c) in w.to_f32().iter().zip(&w_composed.to_f32()) {
        assert!((f - c).abs() < 1e-6, "trapezoid w fused={f} composed={c}");
    }

    check_grad("trapezoid d/dlambda", &lam_data, vec![batch, seq, heads], |v| {
        let (g, w) = V::trapezoid_weights(v, &dt).unwrap();
        g.mul(&w).unwrap().sum().unwrap()
    });
    check_grad("trapezoid d/ddt", &dt_data, vec![batch, seq, heads], |v| {
        let (g, w) = V::trapezoid_weights(&lam, v).unwrap();
        g.mul(&w).unwrap().sum().unwrap()
    });
    // One output dropped on the floor: the sink must treat its missing gradient
    // as zero, not read stale state.
    check_grad("trapezoid d/dlambda (w only)", &lam_data, vec![batch, seq, heads], |v| {
        let (_g, w) = V::trapezoid_weights(v, &dt).unwrap();
        w.tanh().sum().unwrap()
    });
    check_grad("trapezoid d/ddt (g only)", &dt_data, vec![batch, seq, heads], |v| {
        let (g, _w) = V::trapezoid_weights(&lam, v).unwrap();
        g.tanh().sum().unwrap()
    });
}

/// The fused per-token cross entropy against the composed log-softmax + gather
/// oracle, with and without label smoothing, and its logits gradient.
#[test]
fn fused_cross_entropy_matches_composed_and_differentiates() {
    use mamba3::tensor::ops::index::IdTensor;
    use mamba3::train::cross_entropy_per_token_composed;

    let (rows, classes) = (3usize, 5usize);
    let data: Vec<f32> = (0..15).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.7).collect();
    let ids = IdTensor::from_slice(&[2, 0, 4], vec![rows], &dev()).unwrap();
    for smoothing in [0.0f32, 0.3] {
        let x = V::constant(Tensor::from_f32(&data, vec![rows, classes], &dev()).unwrap());
        let fused = x.cross_entropy_rows(&ids, smoothing).unwrap();
        let composed = cross_entropy_per_token_composed(&x, &ids, smoothing).unwrap();
        assert_eq!(fused.shape().dims(), &[rows]);
        for (f, c) in fused.to_f32().iter().zip(&composed.to_f32()) {
            assert!(
                (f - c).abs() < 1e-4,
                "cross entropy fused={f} composed={c} (smoothing {smoothing})"
            );
        }
        check_grad("cross_entropy_rows", &data, vec![rows, classes], |v| {
            v.cross_entropy_rows(&ids, smoothing).unwrap().sum().unwrap()
        });
    }
}
