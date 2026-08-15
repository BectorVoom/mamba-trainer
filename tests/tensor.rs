//! Numerical checks for the raw kernel layer.

#![cfg(feature = "cpu")]

use mamba3::backend::Device;
use mamba3::backends::Cpu;
use mamba3::tensor::Shape;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::*;

type R = Cpu;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor<R, f32> {
    Tensor::from_f32(data, shape, &dev()).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= eps * (1.0 + e.abs()),
            "at {i}: {a} != {e}\nactual:   {actual:?}\nexpected: {expected:?}"
        );
    }
}

#[test]
fn fill_and_roundtrip() {
    let x = Tensor::<R, f32>::full(vec![2, 3], 1.5, &dev());
    assert_close(&x.to_f32(), &[1.5; 6], 1e-6);
}

#[test]
fn elementwise_unary() {
    let x = t(&[-1.0, 0.0, 1.0, 2.0], vec![4]);
    assert_close(&exp(&x).to_f32(), &[(-1.0f32).exp(), 1.0, 1.0f32.exp(), 2.0f32.exp()], 1e-5);
    assert_close(&sigmoid(&x).to_f32(), &[0.26894143, 0.5, 0.7310586, 0.880797], 1e-5);
    assert_close(&abs(&x).to_f32(), &[1.0, 0.0, 1.0, 2.0], 1e-6);
    assert_close(&sign(&x).to_f32(), &[-1.0, 0.0, 1.0, 1.0], 1e-6);
    assert_close(&relu(&x).to_f32(), &[0.0, 0.0, 1.0, 2.0], 1e-6);
}

#[test]
fn binary_broadcast() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = t(&[10.0, 20.0, 30.0], vec![3]);
    assert_close(
        &add(&a, &b).unwrap().to_f32(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        1e-6,
    );

    let c = t(&[1.0, 2.0], vec![2, 1]);
    assert_close(
        &mul(&a, &c).unwrap().to_f32(),
        &[1.0, 2.0, 3.0, 8.0, 10.0, 12.0],
        1e-6,
    );
}

#[test]
fn matmul_matches_reference() {
    // [2, 3] @ [3, 2]
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = t(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let c = matmul(&a, &b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_close(&c.to_f32(), &[58.0, 64.0, 139.0, 154.0], 1e-5);
}

#[test]
fn matmul_batched_broadcast() {
    // [2, 2, 3] @ [3, 4] -> [2, 2, 4]
    let mut a_data = Vec::new();
    for i in 0..12 {
        a_data.push(i as f32);
    }
    let a = t(&a_data, vec![2, 2, 3]);
    let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect();
    let b = t(&b_data, vec![3, 4]);
    let c = matmul(&a, &b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 4]);

    // Reference on the host.
    let mut expected = vec![0.0f32; 2 * 2 * 4];
    for batch in 0..2 {
        for m in 0..2 {
            for n in 0..4 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += a_data[batch * 6 + m * 3 + k] * b_data[k * 4 + n];
                }
                expected[batch * 8 + m * 4 + n] = acc;
            }
        }
    }
    assert_close(&c.to_f32(), &expected, 1e-5);
}

#[test]
fn matmul_larger_than_one_tile() {
    let (m, k, n) = (33, 40, 17);
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.25).collect();
    let a = t(&a_data, vec![m, k]);
    let b = t(&b_data, vec![k, n]);
    let c = matmul(&a, &b).unwrap();

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a_data[i * k + p] * b_data[p * n + j];
            }
            expected[i * n + j] = acc;
        }
    }
    assert_close(&c.to_f32(), &expected, 1e-4);
}

#[test]
fn reductions() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_close(&sum_dim(&x, 1).unwrap().to_f32(), &[6.0, 15.0], 1e-6);
    assert_close(&sum_dim(&x, 0).unwrap().to_f32(), &[5.0, 7.0, 9.0], 1e-6);
    assert_close(&max_dim(&x, 1).unwrap().to_f32(), &[3.0, 6.0], 1e-6);
    assert_close(&mean_dim(&x, 1).unwrap().to_f32(), &[2.0, 5.0], 1e-6);
    assert_close(&sum_all(&x).unwrap().to_f32(), &[21.0], 1e-6);
}

#[test]
fn sum_all_over_many_chunks() {
    let n = 5000;
    let x = Tensor::<R, f32>::ones(vec![n], &dev());
    assert_close(&sum_all(&x).unwrap().to_f32(), &[n as f32], 1e-3);
}

#[test]
fn movement_ops() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_close(
        &transpose(&x).unwrap().to_f32(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        1e-6,
    );
    assert_close(&flip(&x, 1).unwrap().to_f32(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0], 1e-6);
    assert_close(&slice(&x, 1, 1, 2).unwrap().to_f32(), &[2.0, 3.0, 5.0, 6.0], 1e-6);
    assert_close(
        &shift_right(&x, 1).unwrap().to_f32(),
        &[0.0, 1.0, 2.0, 0.0, 4.0, 5.0],
        1e-6,
    );

    let a = t(&[1.0, 2.0], vec![1, 2]);
    let b = t(&[3.0, 4.0], vec![1, 2]);
    assert_close(&cat(&[a, b], 0).unwrap().to_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-6);
}

#[test]
fn permute_rank3() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = t(&data, vec![2, 3, 4]);
    let y = permute(&x, &[2, 0, 1]).unwrap();
    assert_eq!(y.dims(), &[4, 2, 3]);
    let got = y.to_f32();
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let src = data[i * 12 + j * 4 + k];
                let dst = got[k * 6 + i * 3 + j];
                assert_eq!(src, dst, "mismatch at ({i},{j},{k})");
            }
        }
    }
}

#[test]
fn cumulative_sums() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    assert_close(&cumsum(&x, 1).unwrap().to_f32(), &[1.0, 3.0, 6.0, 10.0], 1e-6);
    assert_close(
        &cumsum_exclusive(&x, 1).unwrap().to_f32(),
        &[0.0, 1.0, 3.0, 6.0],
        1e-6,
    );
    assert_close(
        &cumsum_reverse(&x, 1).unwrap().to_f32(),
        &[10.0, 9.0, 7.0, 4.0],
        1e-6,
    );
    assert_close(
        &cumsum_reverse_exclusive(&x, 1).unwrap().to_f32(),
        &[9.0, 7.0, 4.0, 0.0],
        1e-6,
    );
}

#[test]
fn embedding_gather_and_scatter() {
    let table = t(&[0.0, 1.0, 10.0, 11.0, 20.0, 21.0], vec![3, 2]);
    let ids = IdTensor::from_slice(&[2, 0, 2], vec![3], &dev()).unwrap();
    let rows = gather_rows(&table, &ids).unwrap();
    assert_eq!(rows.dims(), &[3, 2]);
    assert_close(&rows.to_f32(), &[20.0, 21.0, 0.0, 1.0, 20.0, 21.0], 1e-6);

    let grad = t(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![3, 2]);
    let table_grad = scatter_add_rows(&grad, &ids, 3).unwrap();
    // row 0 <- id position 1, row 2 <- positions 0 and 2, row 1 untouched.
    assert_close(&table_grad.to_f32(), &[2.0, 2.0, 0.0, 0.0, 4.0, 4.0], 1e-6);
}

#[test]
fn dropout_mask_statistics() {
    let mask = dropout_mask::<R, f32>(vec![4096], 0.5, 1234, &dev());
    let values = mask.to_f32();
    let kept = values.iter().filter(|v| **v > 0.0).count() as f32 / values.len() as f32;
    assert!((kept - 0.5).abs() < 0.05, "keep rate {kept} far from 0.5");
    // Surviving entries are rescaled so the mean stays 1.
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    assert!((mean - 1.0).abs() < 0.1, "mean {mean} far from 1");
}
