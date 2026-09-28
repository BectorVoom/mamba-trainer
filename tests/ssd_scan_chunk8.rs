//! Chunk-8 twin of `tests/ssd_scan.rs`.
//!
//! This is a separate test binary (not a module of `ssd_scan.rs`) because the
//! chunk length is process-wide and read once: [`scan_chunk`](crate) caches
//! `MAMBA3_SCAN_CHUNK` in a `std::sync::OnceLock` on first use, like
//! `chunked_backward`'s `FORCE_RECURRENT`. Two values cannot coexist in one
//! process, so the chunk-16 tests stay in `tests/ssd_scan.rs` (env unset,
//! default 16) and the chunk-8 tests live here (env set to `"8"` before the
//! first scan runs).
//!
//! Helpers are copied from `tests/ssd_scan.rs` rather than made public there.
//!
//! Gate note (see `chunked_backward` in `src/tensor/ops/ssd_scan.rs`): the
//! chunked kernel needs `(chunk * head_dim)` and `(chunk * state)` multiples
//! of 256, `(head_dim * state)` a multiple of 256, plus head_dim % 4 == 0 and
//! state % 2 == 0, and shared memory within limits (huge on the CPU backend,
//! so divisibility decides). With chunk 8 that means head_dim and state must
//! each be multiples of 32. The (64, 32) model shape qualifies; the (16, 16)
//! shapes used for the rate/skip and init/reset cases do not (8*16 = 128),
//! on any backend, so under chunk 8 they fall back to the recurrent kernel.
//! Each test below asserts the gate where it should hold; where it cannot
//! hold the test documents the fallback and a chunk-8-capable twin covers the
//! same path through the chunked kernel.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::ssm::scan::ssd_chunked;
use mamba3::tensor::Tensor;

type R = Auto;
type V = Var<R, f32>;

/// Pin this binary to chunk 8 before the first scan reads the setting.
///
/// # Safety
///
/// `std::env::set_var` is `unsafe` because it races with other threads
/// reading the environment. The tests here run single-threaded up to the
/// first scan, set the same value every time, and never change it afterwards,
/// so the once-read caching in `scan_chunk` always observes `"8"`.
fn ensure_chunk8() {
    unsafe {
        std::env::set_var("MAMBA3_SCAN_CHUNK", "8");
    }
}

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

/// The `chunked_backward` gate for chunk 8, minus the shared-memory check.
///
/// Shared memory for the shapes here is at most ~29 KB (chunk 8, pd 64,
/// n 32), far below the 48 KB cap the gate enforces, and the CPU backend
/// reports gigabytes, so divisibility plus the small-multiple rules decide.
/// `MAMBA3_SCAN_BACKWARD` must also be unset (not `"recurrent"`); that is the
/// process default and no test here sets it. Panics when the shape would fall
/// back to the recurrent kernel, i.e. when this test would pass vacuously.
fn assert_chunked8_taken(head_dim: usize, state: usize) {
    const UNITS: usize = 256;
    const CHUNK: usize = 8;
    assert!(
        !std::env::var("MAMBA3_SCAN_BACKWARD").is_ok_and(|v| v == "recurrent"),
        "MAMBA3_SCAN_BACKWARD=recurrent forces the recurrent kernel"
    );
    assert!(
        (CHUNK * head_dim).is_multiple_of(UNITS),
        "chunk 8 needs 8*head_dim multiple of 256, got head_dim {head_dim}"
    );
    assert!(
        (CHUNK * state).is_multiple_of(UNITS),
        "chunk 8 needs 8*state multiple of 256, got state {state}"
    );
    assert!(
        (head_dim * state).is_multiple_of(UNITS),
        "chunked kernel needs head_dim*state multiple of 256"
    );
    assert!(head_dim.is_multiple_of(4), "head_dim must be a multiple of 4");
    assert!(state.is_multiple_of(2), "state must be a multiple of 2");
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

/// The entity model's decoder shape at batch 1: head_dim 64, d_state 32, 160 steps.
#[test]
fn fused_scan_matches_chunked_model_shape() {
    ensure_chunk8();
    // 8*64 and 8*32 are multiples of 256: the chunked kernel runs on the CPU
    // backend (shared ~29 KB, `MAMBA3_SCAN_BACKWARD` unset).
    assert_chunked8_taken(64, 32);
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
/// same arithmetic spelled out around the chunked scan.
///
/// The (16, 16) subcase cannot take the chunked kernel with chunk 8 on any
/// backend (8*16 = 128 is not a multiple of 256), so there it checks the
/// recurrent kernel; the (64, 32) subcase asserts the chunked kernel ran.
#[test]
fn chunked_backward_rate_and_skip_match_composed() {
    ensure_chunk8();
    rate_and_skip_case(2, 37, 2, 16, 16);
    assert_chunked8_taken(64, 32);
    rate_and_skip_case(1, 32, 3, 64, 32);
}

/// An initial state (and its gradient) through the chunked backward kernel.
///
/// At (16, 16) chunk 8 cannot take the chunked kernel (see above), so this
/// shape checks the recurrent kernel; the `_chunk8_shape` twin below covers
/// the same initial-state-plus-reset path through the chunk-8 kernel.
#[test]
fn chunked_backward_with_initial_state_and_reset() {
    ensure_chunk8();
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

/// Same as above at a chunk-8-capable shape (multiples of 32), so the
/// initial-state gradient and reset go through the chunk-8 kernel.
#[test]
fn chunked_backward_with_initial_state_and_reset_chunk8_shape() {
    ensure_chunk8();
    assert_chunked8_taken(32, 32);
    run(
        &Case {
            batch: 2,
            seq: 45,
            heads: 2,
            head_dim: 32,
            state: 32,
            chunk: 8,
            with_init: true,
            reset_at: Some(20),
        },
        29,
    );
}
