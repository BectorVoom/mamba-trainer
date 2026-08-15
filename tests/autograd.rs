//! Gradient checks against central finite differences.

#![cfg(feature = "cpu")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Cpu;
use mamba3::tensor::{Shape, Tensor};

type R = Cpu;
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
