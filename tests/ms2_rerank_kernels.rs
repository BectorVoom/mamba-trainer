//! K5 kernel-versus-twin tests for the reranker features.
//!
//! Every device call runs on poisoned outputs, is followed by
//! [`check_launches`], and is compared element-for-element (bit-exact `f32`)
//! with the host twin [`compute_features`]: a dropped launch (stale poison)
//! or a wrong word fails. Sizes stay small on the CPU runtime; the supervisor
//! runs the same file on wgpu.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::ms2::pack::SCORE_FINITE_MAX;
use mamba3::models::ms2::rerank::{EVIDENCE_STRIDE, N_FEATURES, compute_features};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2_rerank;

type R = Auto;
type E = f32;

const T: usize = 6;
const A: usize = 8;

/// Tiny deterministic generator (SplitMix64): no new dependencies.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n.max(1))) as u32
    }

    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 11) as f32 / 9007199254740992.0)
    }
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn upload_ids(data: &[u32], shape: Vec<usize>, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(data, shape, device).unwrap()
}

fn upload_f(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, E> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

/// Random candidate records: finished/unfinished, with and without evidence,
/// the incomplete-support bit, and non-finite scores.
#[allow(clippy::too_many_lines)]
fn random_inputs(rows: usize, seed: u64) -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let stride = T * 4 + A + 4;
    let mut actions = vec![0u32; rows * stride];
    let mut scores = vec![0.0f32; rows * 2];
    let mut evidence = vec![0u32; rows * EVIDENCE_STRIDE];
    let mut evidence_f = vec![0.0f32; rows * 2];
    for r in 0..rows {
        let finished = rng.below(4) != 0;
        let length = if finished {
            2 + rng.below(T as u32 - 1)
        } else {
            rng.below(T as u32)
        };
        for s in 0..length {
            // Token kinds 1 (START), 2 (ADD_ATOM), 3 (CLOSE_RING), 4 (STOP).
            let kind = 1 + rng.below(4);
            let base = r * stride + (s as usize) * 4;
            actions[base] = kind;
            actions[base + 1] = rng.below(18);
            actions[base + 2] = rng.below(4);
            actions[base + 3] = rng.below(4);
        }
        for a in 0..A {
            actions[r * stride + T * 4 + a] = rng.below(4);
        }
        let len_field = r * stride + T * 4 + A;
        actions[len_field] = length;
        actions[len_field + 1] = u32::from(finished);
        actions[len_field + 2] = rng.next() as u32;
        actions[len_field + 3] = rng.below(8);
        // Scores: log-probabilities, with every sixth row non-finite.
        let poison = r % 6;
        let (tlp, flp) = match poison {
            1 => (f32::NAN, rng.f32(-20.0, -0.01)),
            2 => (rng.f32(-20.0, -0.01), f32::INFINITY),
            3 => (f32::NEG_INFINITY, f32::NEG_INFINITY),
            4 => (f32::MAX, rng.f32(-20.0, -0.01)),
            _ => (rng.f32(-20.0, -0.01), rng.f32(-20.0, -0.01)),
        };
        scores[r * 2] = tlp;
        scores[r * 2 + 1] = flp;
        // Evidence: status (base 0/1/2 with the incomplete bit sometimes),
        // count (sometimes beyond E), random filler words.
        let base_status = [0u32, 1, 2, 0, 1, 2][r % 6];
        let incomplete = u32::from(rng.below(3) == 0);
        evidence[r * EVIDENCE_STRIDE] = base_status + incomplete * 128;
        let count = if r % 3 == 0 { 0 } else { rng.below(7) };
        evidence[r * EVIDENCE_STRIDE + 1] = count;
        for w in 2..EVIDENCE_STRIDE {
            evidence[r * EVIDENCE_STRIDE + w] = rng.next() as u32;
        }
        // Evidence floats: spec defaults when there is no evidence, one
        // non-finite row with evidence present (the kernel must sanitize it).
        if count == 0 {
            evidence_f[r * 2] = 0.0;
            evidence_f[r * 2 + 1] = 1.0;
        } else if r % 11 == 5 {
            evidence_f[r * 2] = f32::NAN;
            evidence_f[r * 2 + 1] = f32::INFINITY;
        } else {
            evidence_f[r * 2] = rng.f32(-5.0, 0.0);
            evidence_f[r * 2 + 1] = rng.f32(0.0, 2.0);
        }
    }
    (actions, scores, evidence, evidence_f)
}

fn check_config(rows: usize, seed: u64, what: &str) {
    let device = dev();
    let stride = T * 4 + A + 4;
    let (actions, scores, evidence, evidence_f) = random_inputs(rows, seed);
    let (want_f, want_ok) =
        compute_features(&actions, T, A as u32, &scores, &evidence, &evidence_f).unwrap();
    assert_eq!(want_f.len(), rows * N_FEATURES, "{what}: twin features length");
    assert_eq!(want_ok.len(), rows, "{what}: twin ok length");
    // Device on poisoned outputs.
    let actions_t = upload_ids(&actions, vec![rows, stride], &device);
    let scores_t = upload_f(&scores, vec![rows, 2], &device);
    let evidence_t = upload_ids(&evidence, vec![rows, EVIDENCE_STRIDE], &device);
    let evidence_f_t = upload_f(&evidence_f, vec![rows, 2], &device);
    let mut features_t = upload_f(&vec![f32::NAN; rows * N_FEATURES], vec![rows, N_FEATURES], &device);
    let mut ok_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_rerank::features(
        &actions_t,
        &scores_t,
        &evidence_t,
        &evidence_f_t,
        &mut features_t,
        &mut ok_t,
        T,
        A as u32,
    )
    .unwrap();
    check_launches(&device).unwrap();
    // Every element, bit-exact.
    let got_f = features_t.try_to_f32().unwrap();
    assert_eq!(got_f.len(), want_f.len(), "{what}: features length");
    for (i, (g, w)) in got_f.iter().zip(want_f.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: feature word {i} differs");
    }
    let got_ok = ok_t.try_to_vec().unwrap();
    assert_eq!(got_ok, want_ok, "{what}: feature_ok differs");
    // Non-finite scores zero the row and clear the flag; the flag is 1
    // exactly where both scores pass the twin's range rule.
    let in_range = |x: f32| x > -SCORE_FINITE_MAX && x < SCORE_FINITE_MAX;
    for r in 0..rows {
        let finite = in_range(scores[r * 2]) && in_range(scores[r * 2 + 1]);
        assert_eq!(want_ok[r] == 1, finite, "{what}: row {r} ok flag");
        if !finite {
            assert!(
                want_f[r * N_FEATURES..(r + 1) * N_FEATURES]
                    .iter()
                    .all(|&v| v == 0.0),
                "{what}: row {r} is not all zeros"
            );
        }
    }
}

#[test]
fn rerank_features_match_twin() {
    for rows in [1usize, 8, 24] {
        for trial in 0..3 {
            check_config(rows, 0x5eed_0000 + rows as u64 * 16 + trial, &format!("rows{rows} t{trial}"));
        }
    }
}

#[test]
fn rerank_features_reject_bad_shapes() {
    let device = dev();
    let rows = 4usize;
    let stride = T * 4 + A + 4;
    let (actions, scores, evidence, evidence_f) = random_inputs(rows, 7);
    let actions_t = upload_ids(&actions, vec![rows, stride], &device);
    let scores_t = upload_f(&scores, vec![rows, 2], &device);
    let evidence_t = upload_ids(&evidence, vec![rows, EVIDENCE_STRIDE], &device);
    let evidence_f_t = upload_f(&evidence_f, vec![rows, 2], &device);
    let mut features_t = upload_f(&vec![0.0f32; rows * N_FEATURES], vec![rows, N_FEATURES], &device);
    let mut ok_t = upload_ids(&vec![0u32; rows], vec![rows], &device);
    let run = |a: &IdTensor<R>,
               s: &Tensor<R, E>,
               e: &IdTensor<R>,
               ef: &Tensor<R, E>,
               f: &mut Tensor<R, E>,
               o: &mut IdTensor<R>,
               steps: usize,
               atoms: u32| {
        ms2_rerank::features(a, s, e, ef, f, o, steps, atoms)
    };
    // Wrong rank on `actions`.
    let flat = upload_ids(&actions, vec![rows * stride], &device);
    assert!(matches!(
        run(&flat, &scores_t, &evidence_t, &evidence_f_t, &mut features_t, &mut ok_t, T, A as u32),
        Err(Error::Shape(_))
    ));
    // Wrong last dimension on `scores`.
    let bad_scores = upload_f(&vec![0.0f32; rows * 3], vec![rows, 3], &device);
    assert!(matches!(
        run(&actions_t, &bad_scores, &evidence_t, &evidence_f_t, &mut features_t, &mut ok_t, T, A as u32),
        Err(Error::Shape(_))
    ));
    // Wrong `evidence` width.
    let bad_ev = upload_ids(&vec![0u32; rows * 17], vec![rows, 17], &device);
    assert!(matches!(
        run(&actions_t, &scores_t, &bad_ev, &evidence_f_t, &mut features_t, &mut ok_t, T, A as u32),
        Err(Error::Shape(_))
    ));
    // Wrong `evidence_f` width.
    let bad_ef = upload_f(&vec![0.0f32; rows], vec![rows, 1], &device);
    assert!(matches!(
        run(&actions_t, &scores_t, &evidence_t, &bad_ef, &mut features_t, &mut ok_t, T, A as u32),
        Err(Error::Shape(_))
    ));
    // Wrong `features` width.
    let mut bad_f = upload_f(&vec![0.0f32; rows * 7], vec![rows, 7], &device);
    assert!(matches!(
        run(&actions_t, &scores_t, &evidence_t, &evidence_f_t, &mut bad_f, &mut ok_t, T, A as u32),
        Err(Error::Shape(_))
    ));
    // Wrong `feature_ok` length.
    let mut bad_ok = upload_ids(&vec![0u32; rows + 1], vec![rows + 1], &device);
    assert!(matches!(
        run(&actions_t, &scores_t, &evidence_t, &evidence_f_t, &mut features_t, &mut bad_ok, T, A as u32),
        Err(Error::Shape(_))
    ));
    // Stride that does not match the caps.
    assert!(matches!(
        run(&actions_t, &scores_t, &evidence_t, &evidence_f_t, &mut features_t, &mut ok_t, T + 1, A as u32),
        Err(Error::Shape(_))
    ));
    // Degenerate caps.
    assert!(matches!(
        run(&actions_t, &scores_t, &evidence_t, &evidence_f_t, &mut features_t, &mut ok_t, T, 0),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        run(&actions_t, &scores_t, &evidence_t, &evidence_f_t, &mut features_t, &mut ok_t, T, 33),
        Err(Error::Shape(_))
    ));
    // Row-count mismatch between buffers.
    let short_scores = upload_f(&scores[..(rows - 1) * 2], vec![rows - 1, 2], &device);
    assert!(matches!(
        run(&actions_t, &short_scores, &evidence_t, &evidence_f_t, &mut features_t, &mut ok_t, T, A as u32),
        Err(Error::Shape(_))
    ));
    // The twin refuses the same shapes.
    assert!(matches!(
        compute_features(&actions, T, 0, &scores, &evidence, &evidence_f),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        compute_features(&actions, T, A as u32, &scores[..scores.len() - 1], &evidence, &evidence_f),
        Err(Error::Shape(_))
    ));
    // Empty buckets pass through without launching.
    let empty_a = upload_ids(&[], vec![0, stride], &device);
    let empty_s = upload_f(&[], vec![0, 2], &device);
    let empty_e = upload_ids(&[], vec![0, EVIDENCE_STRIDE], &device);
    let empty_ef = upload_f(&[], vec![0, 2], &device);
    let mut empty_f = upload_f(&[], vec![0, N_FEATURES], &device);
    let mut empty_ok = upload_ids(&[], vec![0], &device);
    run(&empty_a, &empty_s, &empty_e, &empty_ef, &mut empty_f, &mut empty_ok, T, A as u32).unwrap();
    check_launches(&device).unwrap();
    let (twin_f, twin_ok) = compute_features(&[], T, A as u32, &[], &[], &[]).unwrap();
    assert!(twin_f.is_empty() && twin_ok.is_empty());
}

/// Hand-computed evidence features per spec §4.3, independent of the twin:
/// feature 4 is the RETAINED evidence count over E = 4 (the buffer's count
/// word is the total number of qualifying peaks, unbounded), and with no
/// evidence the incomplete flag is 0 with features 5/6 at their defaults.
#[test]
fn evidence_features_match_hand_computed_spec() {
    // One record: length 1 with a single ADD_ATOM token (kind 2) and open
    // valences [1, 0, ...] over A slots, so f2 = 1/A and f3 = 1/(2A).
    let t = 2usize;
    let a = 2u32;
    let stride = t * 4 + a as usize + 4;
    let mut actions = vec![0u32; 2 * stride];
    for r in 0..2 {
        let base = r * stride;
        actions[base] = 2; // ADD_ATOM at step 0
        actions[base + t * 4] = 1; // one open valence
        actions[base + t * 4 + a as usize] = 1; // length = 1
        actions[base + t * 4 + a as usize + 1] = 1;
    }
    let scores = vec![-1.0f32, -2.0, -3.0, -4.0];
    // Row 0: six qualifying peaks in the buffer (beyond E = 4) with status 0:
    // retained = min(6, 4) = 4, so f4 = 1.0, never 1.5.
    // Row 1: no evidence (count 0) with the incomplete-support bit set: the
    // flag must still be 0 and features 5/6 the defaults (0, 1).
    let mut evidence = vec![0u32; 2 * EVIDENCE_STRIDE];
    evidence[1] = 6;
    evidence[EVIDENCE_STRIDE] = 128;
    evidence[EVIDENCE_STRIDE + 1] = 0;
    let evidence_f = vec![-0.5f32, 0.25, 99.0, 99.0];
    let want = vec![
        -1.0f32, -2.0, 0.5, 0.25, 1.0, -0.5, 0.25, 0.0,
        -3.0f32, -4.0, 0.5, 0.25, 0.0, 0.0, 1.0, 0.0,
    ];
    // The twin matches the hand values (not the other way round).
    let (twin_f, twin_ok) =
        compute_features(&actions, t, a, &scores, &evidence, &evidence_f).unwrap();
    assert_eq!(twin_ok, vec![1, 1]);
    assert_eq!(twin_f.len(), want.len());
    for (i, (g, w)) in twin_f.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "twin feature word {i} differs");
    }
    // The kernel matches the same hand values.
    let device = dev();
    let actions_t = upload_ids(&actions, vec![2, stride], &device);
    let scores_t = upload_f(&scores, vec![2, 2], &device);
    let evidence_t = upload_ids(&evidence, vec![2, EVIDENCE_STRIDE], &device);
    let evidence_f_t = upload_f(&evidence_f, vec![2, 2], &device);
    let mut features_t = upload_f(&[f32::NAN; 2 * N_FEATURES], vec![2, N_FEATURES], &device);
    let mut ok_t = upload_ids(&[0xDEAD_BEEF; 2], vec![2], &device);
    ms2_rerank::features(
        &actions_t,
        &scores_t,
        &evidence_t,
        &evidence_f_t,
        &mut features_t,
        &mut ok_t,
        t,
        a,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let got_f = features_t.try_to_f32().unwrap();
    for (i, (g, w)) in got_f.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "kernel feature word {i} differs");
    }
    assert_eq!(ok_t.try_to_vec().unwrap(), vec![1, 1]);
}
