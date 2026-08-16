//! Numerical checks for the raw kernel layer.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Shape;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::*;

type R = Auto;

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

/// Kernels over flat buffers pick a vector width from the device, and only widths
/// that divide the element count exactly. A width that is wrong by a factor is a
/// *silent* wrong answer, not a crash, so sweep lengths that land on every width a
/// backend is likely to offer (1, 2, 4, 8, 16) and on lengths that divide none of
/// them cleanly.
#[test]
fn kernels_agree_across_every_vector_width() {
    for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17, 31, 32, 48, 64, 96, 129] {
        let data: Vec<f32> = (0..n).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let other: Vec<f32> = (0..n).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let x = t(&data, vec![n]);
        let y = t(&other, vec![n]);

        assert_close(
            &Tensor::<R, f32>::zeros(vec![n], &dev()).to_f32(),
            &vec![0.0; n],
            0.0,
        );
        assert_close(&identity(&x).to_f32(), &data, 0.0);
        assert_close(
            &mul_scalar(&x, 3.0).to_f32(),
            &data.iter().map(|v| v * 3.0).collect::<Vec<_>>(),
            1e-6,
        );
        assert_close(
            &add(&x, &y).unwrap().to_f32(),
            &data.iter().zip(&other).map(|(a, b)| a + b).collect::<Vec<_>>(),
            1e-6,
        );
        assert_close(
            &relu(&x).to_f32(),
            &data.iter().map(|v| v.max(0.0)).collect::<Vec<_>>(),
            1e-6,
        );
        assert_close(
            &sign(&x).to_f32(),
            &data.iter().map(|v| v.signum() * (*v != 0.0) as u8 as f32).collect::<Vec<_>>(),
            1e-6,
        );
        // Reduce the contiguous axis: the fold-the-lanes path.
        assert_close(
            &sum_dim(&x, 0).unwrap().to_f32(),
            &[data.iter().sum::<f32>()],
            1e-4,
        );
        assert_close(
            &max_dim(&x, 0).unwrap().to_f32(),
            &[data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)],
            1e-6,
        );
        assert_close(&cumsum(&x, 0).unwrap().to_f32(), &running_sum(&data), 1e-4);

        // Reduce a leading axis, so the trailing extent is what gets vectorised.
        let rows = t(&data, vec![1, n]);
        let stacked = cat(&[rows.clone(), rows.clone()], 0).unwrap();
        assert_close(
            &sum_dim(&stacked, 0).unwrap().to_f32(),
            &data.iter().map(|v| v * 2.0).collect::<Vec<_>>(),
            1e-5,
        );
        // Broadcasting along a leading axis keeps the trailing axis contiguous.
        assert_close(
            &mul(&stacked, &rows).unwrap().to_f32(),
            &data
                .iter()
                .chain(data.iter())
                .map(|v| v * v)
                .collect::<Vec<_>>(),
            1e-5,
        );
        assert_close(&flip(&rows, 1).unwrap().to_f32(), &reversed(&data), 0.0);

        // `[1, n] @ [n, 1]` exercises a k-loop; `[n, 1] @ [1, n]` a wide output.
        let col = t(&other, vec![n, 1]);
        assert_close(
            &matmul(&rows, &col).unwrap().to_f32(),
            &[data.iter().zip(&other).map(|(a, b)| a * b).sum::<f32>()],
            1e-4,
        );
        let outer = matmul(&col, &rows).unwrap();
        assert_eq!(outer.dims(), &[n, n]);
        assert_close(
            &outer.to_f32()[..n],
            &data.iter().map(|v| v * other[0]).collect::<Vec<_>>(),
            1e-5,
        );
    }
}

fn running_sum(data: &[f32]) -> Vec<f32> {
    let mut acc = 0.0;
    data.iter()
        .map(|v| {
            acc += v;
            acc
        })
        .collect()
}

fn reversed(data: &[f32]) -> Vec<f32> {
    let mut v = data.to_vec();
    v.reverse();
    v
}

/// Every matmul kernel must agree. They differ only in how work is assigned to
/// units, so a disagreement beyond float reassociation is a bug in one of them.
///
/// `Tiled` is exercised only where a cube barrier is a hardware instruction: on the
/// CPU runtime it is emulated, and the module documentation records what that costs.
#[test]
fn matmul_kernels_agree() {
    use mamba3::tensor::ops::matmul::{MatmulKernel, set_default_kernel};

    let planes = dev().client().properties().hardware.plane_size_max > 1;
    let mut kernels = vec![
        MatmulKernel::Auto,
        MatmulKernel::Simple,
        MatmulKernel::RowTiled,
    ];
    if planes {
        kernels.push(MatmulKernel::Tiled);
        kernels.push(MatmulKernel::BlockTiled);
    }

    // Shapes chosen so `m` is and is not a multiple of the row tile, and `n` is and
    // is not a multiple of a plausible vector width.
    for (b, m, n, k) in [
        (1usize, 1usize, 8usize, 8usize),
        (1, 3, 5, 7),
        (1, 8, 16, 32),
        (2, 7, 12, 9),
        (3, 16, 4, 20),
    ] {
        let ld: Vec<f32> = (0..b * m * k).map(|i| (i % 11) as f32 - 5.0).collect();
        let rd: Vec<f32> = (0..b * k * n).map(|i| (i % 7) as f32 - 3.0).collect();
        let lhs = t(&ld, vec![b, m, k]);
        let rhs = t(&rd, vec![b, k, n]);

        let mut want = vec![0.0f32; b * m * n];
        for (bi, out) in want.chunks_mut(m * n).enumerate() {
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += ld[bi * m * k + r * k + p] * rd[bi * k * n + p * n + c];
                    }
                    out[r * n + c] = acc;
                }
            }
        }

        for kernel in &kernels {
            set_default_kernel(*kernel);
            let got = matmul(&lhs, &rhs).unwrap();
            assert_eq!(got.dims(), &[b, m, n], "{kernel:?} on {b}x{m}x{n}x{k}");
            assert_close(&got.to_f32(), &want, 1e-5);
        }
    }
    set_default_kernel(MatmulKernel::Auto);
}

/// Reading an operand transposed must give the same product as transposing it first.
///
/// The adjoint of a matrix product is `dA = G Bᵀ` and `dB = Aᵀ G`, and both are
/// computed by handing the kernel an untransposed buffer and a flag rather than by
/// materialising the transpose. That is only sound if the two agree exactly, which is
/// what this checks — over shapes that do and do not divide the block tile, and with
/// and without a batch, since the transposed kernel indexes the batch itself.
#[test]
fn transposed_operands_match_materialised_transposes() {
    use mamba3::tensor::ops::matmul::{MatmulKernel, matmul_nt, matmul_tn, set_default_kernel};

    for (b, m, n, k) in [
        (1usize, 4usize, 8usize, 8usize),
        (1, 3, 5, 7),
        (2, 8, 16, 32),
        (3, 130, 68, 20),
        (5, 64, 64, 64),
    ] {
        // `lhs` is stored [b, k, m] for the `tn` form and [b, m, k] for `nt`.
        let a: Vec<f32> = (0..b * m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
        let g: Vec<f32> = (0..b * m * n).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect();
        let bb: Vec<f32> = (0..b * k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.75).collect();

        let a_t = t(&a, vec![b, m, k]); // used as [b, k, m] by matmul_tn
        let g_t = t(&g, vec![b, m, n]);
        let b_t = t(&bb, vec![b, k, n]);

        // dA = G Bᵀ: contract the trailing axis of both.
        let want_da = matmul(&g_t, &transpose(&b_t).unwrap()).unwrap();
        // dB = Aᵀ G: contract the leading matrix axis of both.
        let want_db = matmul(&transpose(&a_t).unwrap(), &g_t).unwrap();

        for kernel in [MatmulKernel::Auto, MatmulKernel::BlockTiled, MatmulKernel::Simple] {
            set_default_kernel(kernel);
            let da = matmul_nt(&g_t, &b_t).unwrap();
            let db = matmul_tn(&a_t, &g_t).unwrap();
            assert_eq!(da.dims(), want_da.dims(), "{kernel:?} nt on {b}x{m}x{n}x{k}");
            assert_eq!(db.dims(), want_db.dims(), "{kernel:?} tn on {b}x{m}x{n}x{k}");
            assert_close(&da.to_f32(), &want_da.to_f32(), 1e-5);
            assert_close(&db.to_f32(), &want_db.to_f32(), 1e-5);
        }
    }
    set_default_kernel(MatmulKernel::Auto);
}

/// The plane-per-output-element kernel is only ever chosen by the tuner, so no
/// `MatmulKernel` variant can pin it the way the other kernels are pinned above.
/// Instead, `MAMBA3_TUNE_CHECK` makes the tuner verify *every* candidate against the
/// simple kernel while it probes — so issuing a shape that admits the plane-dot plan
/// (small output, long contraction, `rhs` transposed) under that flag is what checks
/// it, along with every other candidate for the shape. On the CPU runtime there is
/// no tuner and this reduces to an ordinary matmul test.
#[test]
fn skinny_adjoint_candidates_agree() {
    use mamba3::tensor::ops::matmul::matmul_nt;

    // Safety: worst case a concurrent test's tuner probe also runs checked.
    unsafe { std::env::set_var("MAMBA3_TUNE_CHECK", "1") };

    let (b, m, n, k) = (3usize, 8usize, 8usize, 384usize);
    let g: Vec<f32> = (0..b * m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
    let w: Vec<f32> = (0..b * n * k).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect();
    let g_t = t(&g, vec![b, m, k]);
    let w_t = t(&w, vec![b, n, k]);

    let want = matmul(&g_t, &transpose(&w_t).unwrap()).unwrap();
    let got = matmul_nt(&g_t, &w_t).unwrap();
    assert_close(&got.to_f32(), &want.to_f32(), 1e-4);
}

/// `permute` takes a vectorised path when the innermost axis survives the reordering
/// and a scalar one when it does not, and it merges axes that stayed adjacent before
/// either. All three have to produce the same bytes as the definition.
#[test]
fn permute_agrees_with_the_definition_on_every_path() {
    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        // Innermost axis untouched, outer axes mergeable: the vectorised path.
        (vec![2, 3, 4, 5, 8], vec![0, 1, 3, 2, 4]),
        // Innermost axis moved: the scalar path.
        (vec![2, 3, 4, 5, 8], vec![0, 1, 2, 4, 3]),
        // Innermost axis untouched but nothing merges.
        (vec![3, 4, 5, 8], vec![2, 0, 1, 3]),
        // A width that does not divide any vector size.
        (vec![2, 3, 4, 5, 7], vec![0, 1, 3, 2, 4]),
        // Size-1 axes, which are dropped before anything else happens.
        (vec![2, 1, 6, 4], vec![2, 0, 1, 3]),
    ];
    for (dims, perm) in cases {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let got = permute(&t(&data, dims.clone()), &perm).unwrap();

        let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        assert_eq!(got.dims(), out_dims.as_slice(), "{dims:?} by {perm:?}");

        // Source strides, reordered the way `permute` reorders them.
        let mut src_strides = vec![1usize; dims.len()];
        for axis in (0..dims.len() - 1).rev() {
            src_strides[axis] = src_strides[axis + 1] * dims[axis + 1];
        }
        let strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();

        let mut want = vec![0.0f32; n];
        for (flat, slot) in want.iter_mut().enumerate() {
            let mut rem = flat;
            let mut off = 0usize;
            for axis in (0..out_dims.len()).rev() {
                off += (rem % out_dims[axis]) * strides[axis];
                rem /= out_dims[axis];
            }
            *slot = data[off];
        }
        assert_close(&got.to_f32(), &want, 0.0);
    }
}

/// The fused rotation must agree with the two-slice, four-multiply form it replaces,
/// on both broadcast shapes the crate uses: RoPE shares one table across batch and
/// heads, the Mamba-3 scan shares one across the state rank.
#[test]
fn fused_rotation_matches_the_composed_form() {
    use mamba3::autograd::Var;

    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![2, 3, 4, 8], vec![1, 1, 4, 4]), // RoPE
        (vec![2, 3, 5, 8], vec![2, 3, 1, 4]), // rotating state frame
        (vec![2, 3, 4, 8], vec![2, 3, 4, 4]), // no broadcast at all
        (vec![6, 2], vec![6, 1]),             // rank 2, trailing broadcast
    ];
    for (x_dims, t_dims) in cases {
        let n: usize = x_dims.iter().product();
        let m: usize = t_dims.iter().product();
        let xd: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let cd: Vec<f32> = (0..m).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let sd: Vec<f32> = (0..m).map(|i| ((i % 7) as f32 - 3.0) * 0.4).collect();
        let x = Var::constant(t(&xd, x_dims.clone()));
        let c = Var::constant(t(&cd, t_dims.clone()));
        let s = Var::constant(t(&sd, t_dims.clone()));

        let fused = x.rotate_halves(&c, &s).unwrap();
        let composed = x.rotate_halves_composed(&c, &s).unwrap();
        assert_eq!(fused.dims(), composed.dims(), "{x_dims:?} / {t_dims:?}");
        assert_close(&fused.to_f32(), &composed.to_f32(), 1e-6);
    }
}

/// The fused depthwise convolution against the sum-of-shifts form it replaces, with
/// and without carried history, for windows longer and shorter than the kernel.
#[test]
fn fused_causal_conv_matches_the_composed_form() {
    use mamba3::autograd::{Var, cat};
    use mamba3::nn::conv::CausalConv1dConfig;
    use mamba3::tensor::ops::random::Rng;

    for taps in [1usize, 2, 4] {
        for bias in [true, false] {
            let channels = 6;
            let conv = CausalConv1dConfig::new(channels, taps)
                .with_bias(bias)
                .init::<R, f32>(&dev(), &mut Rng::seeded(3));

            for seq in [1usize, 2, 5, 8] {
                let batch = 2;
                let n = batch * seq * channels;
                let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.3).collect();
                let x = Var::constant(t(&data, vec![batch, seq, channels]));

                // No history: both forms see the same zero padding.
                assert_close(
                    &conv.apply(&x).unwrap().to_f32(),
                    &conv.apply_composed(&x).unwrap().to_f32(),
                    1e-5,
                );

                let carry = taps - 1;
                if carry == 0 {
                    continue;
                }
                let hn = batch * carry * channels;
                let hdata: Vec<f32> = (0..hn).map(|i| ((i % 7) as f32 - 3.0) * 0.4).collect();
                let history = Var::constant(t(&hdata, vec![batch, carry, channels]));

                let (out, next) = conv.apply_with_history(&x, &history).unwrap();
                let window = cat(&[history.clone(), x.clone()], 1).unwrap();
                let want = conv
                    .apply_composed(&window)
                    .unwrap()
                    .slice(1, carry, seq)
                    .unwrap();
                assert_close(&out.to_f32(), &want.to_f32(), 1e-5);

                // The carried history is the tail of the window, verbatim.
                let want_next = window.slice(1, seq, carry).unwrap();
                assert_eq!(next.dims(), want_next.dims());
                assert_close(&next.to_f32(), &want_next.to_f32(), 0.0);
            }
        }
    }
}
