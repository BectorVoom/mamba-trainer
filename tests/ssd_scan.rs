//! The fused recurrence scan (`Var::ssd_scan`) against the chunked decomposition
//! (`ssd_chunked`), forward and every gradient.
//!
//! `ssd_chunked` with `want_state = true` always takes the chunked path, which is
//! what makes it the oracle here regardless of the device default.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::ssm::scan::ssd_chunked;
use mamba3::tensor::Tensor;

type R = Auto;
type V = Var<R, f32>;

fn dev() -> Device<R> {
    Device::<R>::default()
}

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

fn assert_close(actual: &[f32], expected: &[f32], eps: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let scale = expected.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let err = (a - e).abs() / scale;
        if err > worst {
            worst = err;
            at = i;
        }
    }
    assert!(
        worst <= eps,
        "{what}: worst error {worst} (relative to max |want| {scale}) at {at}: got {}, want {}",
        actual[at],
        expected[at]
    );
}

struct Case {
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state: usize,
    chunk: usize,
    with_init: bool,
    reset_at: Option<usize>,
}

fn run(case: &Case, seed: u64) {
    let Case {
        batch,
        seq,
        heads,
        head_dim,
        state,
        chunk,
        with_init,
        reset_at,
    } = *case;
    let x = noise(batch * seq * heads * head_dim, seed);
    let b = noise(batch * seq * heads * state, seed + 1);
    let c = noise(batch * seq * heads * state, seed + 2);
    let mut a: Vec<f32> = noise(batch * seq * heads, seed + 3)
        .iter()
        .map(|v| -0.05 - 0.4 * (v + 1.0))
        .collect();
    if let Some(t) = reset_at {
        for bi in 0..batch {
            for h in 0..heads {
                a[(bi * seq + t) * heads + h] = -60.0;
            }
        }
    }
    let g = noise(batch * seq * heads, seed + 4);
    let w = noise(batch * seq * heads, seed + 5);
    let init = noise(batch * heads * head_dim * state, seed + 6);
    let weights = noise(batch * seq * heads * head_dim, seed + 7);

    let shapes = [
        vec![batch, seq, heads, head_dim],
        vec![batch, seq, heads, state],
        vec![batch, seq, heads, state],
        vec![batch, seq, heads],
        vec![batch, seq, heads],
        vec![batch, seq, heads],
        vec![batch, heads, head_dim, state],
    ];
    let data = [&x, &b, &c, &a, &g, &w, &init];
    let names = ["x", "b", "c", "a", "g", "w", "init"];

    let eval = |fused: bool| -> (Vec<f32>, Vec<Vec<f32>>) {
        let leaves: Vec<V> = data
            .iter()
            .zip(&shapes)
            .map(|(d, s)| V::traced(Tensor::from_f32(d, s.clone(), &dev()).unwrap()))
            .collect();
        let init = with_init.then_some(&leaves[6]);
        let y = if fused {
            V::ssd_scan(
                &leaves[0], &leaves[1], &leaves[2], &leaves[3], None, &leaves[4], &leaves[5], init,
                None,
            )
            .unwrap()
        } else {
            ssd_chunked(
                &leaves[0], &leaves[1], &leaves[2], &leaves[3], &leaves[4], &leaves[5], init,
                chunk, true,
            )
            .unwrap()
            .0
        };
        let mask = V::constant(Tensor::from_f32(&weights, shapes[0].clone(), &dev()).unwrap());
        let loss = y.mul(&mask).unwrap().sum().unwrap();
        let grads = loss.backward_retain().unwrap();
        let n = if with_init { 7 } else { 6 };
        let gs = leaves[..n]
            .iter()
            .enumerate()
            .map(|(i, l)| grads.node(l.node().unwrap()).unwrap_or_else(|| panic!("no gradient for {} (fused={fused})", names[i])).to_f32())
            .collect();
        (y.to_f32(), gs)
    };

    let (y_want, g_want) = eval(false);
    let (y_got, g_got) = eval(true);
    assert_close(&y_got, &y_want, 1e-4, "y");
    for (i, (got, want)) in g_got.iter().zip(&g_want).enumerate() {
        assert_close(got, want, 2e-4, &format!("d{}", names[i]));
    }
}

#[test]
fn fused_scan_matches_chunked_small() {
    run(
        &Case {
            batch: 2,
            seq: 37,
            heads: 3,
            head_dim: 5,
            state: 6,
            chunk: 8,
            with_init: false,
            reset_at: None,
        },
        11,
    );
}

#[test]
fn fused_scan_matches_chunked_with_initial_state_and_reset() {
    run(
        &Case {
            batch: 2,
            seq: 21,
            heads: 2,
            head_dim: 4,
            state: 4,
            chunk: 5,
            with_init: true,
            reset_at: Some(9),
        },
        23,
    );
}

/// The entity model's decoder shape at batch 1: head_dim 64, d_state 32, 160 steps.
#[test]
fn fused_scan_matches_chunked_model_shape() {
    run(
        &Case {
            batch: 1,
            seq: 160,
            heads: 2,
            head_dim: 64,
            state: 32,
            chunk: 40,
            with_init: false,
            reset_at: None,
        },
        37,
    );
}

/// The decay given as `dt * A[h]` and the `D[h] x` skip folded in, against the
/// same arithmetic spelled out around the chunked scan — forward and the
/// gradients of every input, including the two per-head parameters.
#[test]
fn fused_scan_rate_and_skip_match_composed() {
    rate_and_skip_case(2, 19, 3, 6, 8);
}

/// The same at shapes the chunked backward kernel takes (`head_dim`, `d_state`
/// multiples of 16): a sequence that ends mid-chunk, and one of whole chunks.
#[test]
fn chunked_backward_rate_and_skip_match_composed() {
    rate_and_skip_case(2, 37, 2, 16, 16);
    rate_and_skip_case(1, 32, 3, 64, 32);
}

/// An initial state (and its gradient) through the chunked backward kernel.
#[test]
fn chunked_backward_with_initial_state_and_reset() {
    run(
        &Case {
            batch: 2,
            seq: 45,
            heads: 2,
            head_dim: 16,
            state: 16,
            chunk: 8,
            with_init: true,
            reset_at: Some(20),
        },
        29,
    );
}

fn rate_and_skip_case(batch: usize, seq: usize, heads: usize, head_dim: usize, state: usize) {
    let x = noise(batch * seq * heads * head_dim, 51);
    let b = noise(batch * seq * heads * state, 52);
    let c = noise(batch * seq * heads * state, 53);
    let dt: Vec<f32> = noise(batch * seq * heads, 54)
        .iter()
        .map(|v| 0.02 + 0.1 * (v + 1.0))
        .collect();
    let a_head: Vec<f32> = [-0.7f32, -2.5, -4.0].iter().cycle().take(heads).copied().collect();
    let g = noise(batch * seq * heads, 55);
    let w = noise(batch * seq * heads, 56);
    let skip: Vec<f32> = [1.0f32, -0.4, 0.8].iter().cycle().take(heads).copied().collect();
    let weights = noise(batch * seq * heads * head_dim, 57);
    let shapes = [
        vec![batch, seq, heads, head_dim],
        vec![batch, seq, heads, state],
        vec![batch, seq, heads, state],
        vec![batch, seq, heads],
        vec![heads],
        vec![batch, seq, heads],
        vec![batch, seq, heads],
        vec![heads],
    ];
    let data = [&x, &b, &c, &dt, &a_head, &g, &w, &skip];
    let names = ["x", "b", "c", "dt", "a_head", "g", "w", "skip"];

    let eval = |fused: bool| -> (Vec<f32>, Vec<Vec<f32>>) {
        let l: Vec<V> = data
            .iter()
            .zip(&shapes)
            .map(|(d, s)| V::traced(Tensor::from_f32(d, s.clone(), &dev()).unwrap()))
            .collect();
        let y = if fused {
            V::ssd_scan(&l[0], &l[1], &l[2], &l[3], Some(&l[4]), &l[5], &l[6], None, Some(&l[7]))
                .unwrap()
        } else {
            let a = l[3].mul(&l[4].reshape(vec![1, 1, heads]).unwrap()).unwrap();
            let (y, _) = ssd_chunked(&l[0], &l[1], &l[2], &a, &l[5], &l[6], None, 5, true).unwrap();
            y.add(&l[0].mul(&l[7].reshape(vec![1, 1, heads, 1]).unwrap()).unwrap())
                .unwrap()
        };
        let mask = V::constant(Tensor::from_f32(&weights, shapes[0].clone(), &dev()).unwrap());
        let grads = y.mul(&mask).unwrap().sum().unwrap().backward_retain().unwrap();
        let gs = l
            .iter()
            .enumerate()
            .map(|(i, v)| {
                grads
                    .node(v.node().unwrap())
                    .unwrap_or_else(|| panic!("no gradient for {} (fused={fused})", names[i]))
                    .to_f32()
            })
            .collect();
        (y.to_f32(), gs)
    };
    let (y_want, g_want) = eval(false);
    let (y_got, g_got) = eval(true);
    assert_close(&y_got, &y_want, 1e-4, "y");
    for (i, (got, want)) in g_got.iter().zip(&g_want).enumerate() {
        assert_close(got, want, 3e-4, &format!("d{}", names[i]));
    }
}

/// A whole Mamba-3 mixer (rotation, trapezoid, skip, B/C norm) with the fused scan
/// forced on and off: same output, same parameter gradients.
#[test]
fn mixer_with_fused_scan_matches_chunked() {
    use mamba3::models::{Mamba3Mixer, Mamba3MixerConfig};
    use mamba3::ssm::config::SsmConfig;
    use mamba3::ssm::set_fused_scan;
    use mamba3::tensor::ops::random::Rng;

    let ssm = SsmConfig {
        d_model: 16,
        n_heads: 2,
        n_groups: 1,
        head_dim: 8,
        d_state: 8,
        chunk_size: 4,
        ..SsmConfig::default()
    };
    let mixer: Mamba3Mixer<R, f32> = Mamba3MixerConfig::new(ssm)
        .init(&dev(), &mut Rng::seeded(5))
        .unwrap();
    let data: Vec<f32> = (0..2 * 11 * 16).map(|i| (i as f32 * 0.17).sin()).collect();
    let run = |fused: bool| -> (Vec<f32>, Vec<Vec<f32>>) {
        set_fused_scan(Some(fused));
        let input = V::traced(Tensor::from_f32(&data, vec![2, 11, 16], &dev()).unwrap());
        let out = mixer.apply(&input).unwrap();
        let grads = out.tanh().sum().unwrap().backward().unwrap();
        let mut gs: Vec<(u64, Vec<f32>)> = grads
            .iter()
            .map(|(id, t)| (id.raw(), t.to_f32()))
            .collect();
        gs.sort_by_key(|(id, _)| *id);
        (out.to_f32(), gs.into_iter().map(|(_, g)| g).collect())
    };
    let (y_want, g_want) = run(false);
    let (y_got, g_got) = run(true);
    set_fused_scan(None);
    assert_close(&y_got, &y_want, 1e-4, "mixer output");
    assert_eq!(g_got.len(), g_want.len(), "same parameters get gradients");
    for (i, (got, want)) in g_got.iter().zip(&g_want).enumerate() {
        assert_close(got, want, 5e-4, &format!("parameter {i}"));
    }
}
