//! H6.2 host-only tests for trajectory allocation
//! (`mamba3::models::ms2::allocate`). No tensors, no kernels.
//!
//! The tests pin the `docs/MS2_V1_ARCHITECTURE.md` §3.2 rule exactly: round
//! robin is `k mod top_count`; proportional renormalises
//! `exp(top_log_prob - max)` in `f32`, gives every retained formula one
//! trajectory first, shares the rest by largest remainder with ties by
//! smaller slot, and fills trajectories in slot order. The contract ranges
//! `K <= 64` and `top_count <= F <= 8` are refused with `Error::Config` by
//! the host wrapper and are a guarded sentinel no-op in the lane; a floor-sum
//! overshoot is reconciled before any unsigned subtraction (helper tested
//! directly); a retained log-probability outside the validated domain falls back to round
//! robin. The lane emits complete `[B, K, 12]` records (slot, source id, 10
//! counts) with sentinel slot/source and zero counts when `top_count == 0`.

use mamba3::models::ms2::allocate::{
    ALLOC_F_MAX, ALLOC_K_MAX, ALLOC_PROPORTIONAL, ALLOC_RECORD, ALLOC_ROUND_ROBIN,
    allocate_checked, allocate_lane, reconcile_extra_shares,
};

/// Lane buffers for `F` retained formulas: `top` is `[F, 2]` (source id,
/// window slot), `top_counts` is `[F, 10]`. Sources are `100 + slot` so the
/// 12-word expansion is checked, and counts encode the slot.
fn buffers(f: usize) -> (Vec<u32>, Vec<u32>) {
    let mut top = Vec::with_capacity(f * 2);
    let mut top_counts = vec![0u32; f * 10];
    let mut s = 0u32;
    while s < f as u32 {
        top.push(100 + s);
        top.push(s);
        let mut e = 0u32;
        while e < 10 {
            top_counts[s as usize * 10 + e as usize] = s * 10 + e;
            e += 1;
        }
        s += 1;
    }
    (top, top_counts)
}

/// Run the lane with these inputs and return the `K` written slots (word 0 of
/// each 12-word record).
fn run(top_log_prob: &[f32], top_count: u32, k: usize, mode: u32) -> Vec<u32> {
    run_full(top_log_prob, top_count, k, mode)
        .iter()
        .step_by(12)
        .copied()
        .collect()
}

/// Run the lane and return the complete `K * 12` words.
fn run_full(top_log_prob: &[f32], top_count: u32, k: usize, mode: u32) -> Vec<u32> {
    let f = top_log_prob.len();
    let (top, top_counts) = buffers(f);
    let top_count_v = vec![top_count];
    let mut out = vec![0u32; k * 12];
    allocate_lane(
        &top,
        &top_counts,
        top_log_prob,
        &top_count_v,
        0,
        f as u32,
        k as u32,
        mode,
        &mut out,
    );
    out
}

/// Slots of a full record buffer.
fn slots_of(full: &[u32]) -> Vec<u32> {
    full.iter().step_by(12).copied().collect()
}

/// Check one 12-word record: slot, source, counts.
fn check_record(full: &[u32], t: usize, slot: u32, f: usize) {
    let base = t * 12;
    assert_eq!(full[base], slot, "t {t} slot");
    if slot == u32::MAX {
        assert_eq!(full[base + 1], u32::MAX, "t {t} source sentinel");
        for e in 0..10 {
            assert_eq!(full[base + 2 + e], 0, "t {t} count {e} zero");
        }
    } else {
        assert_eq!(full[base + 1], 100 + slot, "t {t} source");
        for e in 0..10 {
            assert_eq!(full[base + 2 + e], slot * 10 + e as u32, "t {t} count {e}");
        }
    }
    let _ = f;
}

/// Per-slot trajectory counts of an assignment, in slot order.
fn counts_of(out: &[u32], slots: usize) -> Vec<u32> {
    let mut counts = vec![0u32; slots];
    for slot in out {
        counts[*slot as usize] += 1;
    }
    counts
}

#[test]
fn round_robin_is_k_mod_count() {
    let probs = [0.0f32; 4];
    for count in [1u32, 2, 3, 4] {
        for k in [1usize, 2, 3, 5, 8, 9, 13] {
            let got = run(&probs[..count as usize], count, k, ALLOC_ROUND_ROBIN);
            assert_eq!(got.len(), k, "exactly K writes");
            for (t, slot) in got.iter().enumerate() {
                assert_eq!(*slot, t as u32 % count, "count {count} K {k} t {t}");
            }
            let full = run_full(&probs[..count as usize], count, k, ALLOC_ROUND_ROBIN);
            assert_eq!(full.len(), k * 12, "complete records");
            for (t, slot) in got.iter().enumerate() {
                check_record(&full, t, *slot, count as usize);
            }
        }
    }
}

#[test]
fn round_robin_ignores_probabilities() {
    let skewed = vec![100.0f32, -100.0, 50.0, -50.0];
    let got = run(&skewed, 4, 9, ALLOC_ROUND_ROBIN);
    let plain = run(&[0.0f32; 4], 4, 9, ALLOC_ROUND_ROBIN);
    assert_eq!(got, plain);
    assert_eq!(got, vec![0, 1, 2, 3, 0, 1, 2, 3, 0]);
}

#[test]
fn count_zero_writes_sentinels() {
    for mode in [ALLOC_ROUND_ROBIN, ALLOC_PROPORTIONAL, 7] {
        let full = run_full(&[0.0f32; 4], 0, 6, mode);
        assert_eq!(full.len(), 6 * 12, "mode {mode}");
        for t in 0..6 {
            check_record(&full, t, u32::MAX, 4);
        }
        assert_eq!(slots_of(&full), vec![u32::MAX; 6], "mode {mode}");
    }
}

#[test]
fn proportional_sums_to_k_with_at_least_one_each() {
    let prob_sets: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, -1.0, -2.0, -3.0],
        vec![5.0, 4.9999, -10.0],
        vec![-0.5, -0.50001],
        vec![2.0],
    ];
    for probs in &prob_sets {
        let count = probs.len() as u32;
        for k in [1usize, 2, 3, 4, 5, 7, 8, 13, 17, 64] {
            let got = run(probs, count, k, ALLOC_PROPORTIONAL);
            assert_eq!(got.len(), k, "exactly K writes");
            assert!(got.iter().all(|slot| *slot < count), "slots in range");
            // Slot order fill: trajectories of one slot are contiguous.
            let mut seen = vec![false; count as usize];
            let mut last = got[0];
            seen[last as usize] = true;
            for slot in &got[1..] {
                if *slot != last {
                    assert!(!seen[*slot as usize], "slot order fill");
                    seen[*slot as usize] = true;
                    last = *slot;
                }
            }
            if k >= count as usize {
                let counts = counts_of(&got, count as usize);
                assert_eq!(counts.iter().sum::<u32>(), k as u32, "sums to K");
                assert!(counts.iter().all(|c| *c >= 1), "at least one each: {counts:?}");
            } else {
                assert_eq!(got, (0..k as u32).collect::<Vec<u32>>(), "first K one each");
            }
            let full = run_full(probs, count, k, ALLOC_PROPORTIONAL);
            for (t, slot) in got.iter().enumerate() {
                check_record(&full, t, *slot, count as usize);
            }
        }
    }
}

#[test]
fn proportional_k_less_than_count_gives_first_k_one_each() {
    let probs = vec![0.0f32, -0.2, -0.4, -0.6, -0.8];
    for k in [1usize, 2, 4] {
        let got = run(&probs, 5, k, ALLOC_PROPORTIONAL);
        assert_eq!(got, (0..k as u32).collect::<Vec<u32>>());
    }
}

#[test]
fn equal_probabilities_break_ties_by_smaller_slot() {
    // Three equal formulas, K = 7: one each, then K' = 4 shared as 1/1/1 with
    // one remainder; every fractional part is 1/3, so slot 0 takes it.
    let got = run(&[0.0f32; 3], 3, 7, ALLOC_PROPORTIONAL);
    assert_eq!(got, vec![0, 0, 0, 1, 1, 2, 2]);
    // Four equal formulas, K = 6: one each, K' = 2 shared 0/0/0/0 with two
    // remainders to slots 0 and 1.
    let got = run(&[0.0f32; 4], 4, 6, ALLOC_PROPORTIONAL);
    assert_eq!(got, vec![0, 0, 1, 1, 2, 3]);
}

#[test]
fn hand_computed_example() {
    // Slots with log-probs 0, -1, -2 (about ln 1, ln 1/e, ln 1/e^2), K = 8:
    // one each leaves K' = 5; exp gives [1, 0.3679, 0.1353], sum 1.5032,
    // quotas [3.326, 1.224, 0.450], floors [3, 1, 0], one remainder to the
    // largest fractional part (slot 2, 0.450). Counts [4, 2, 2]. No quota is
    // within 1e-4 of an integer and no two fractional parts tie, so a twin
    // comparison is meaningful here.
    let probs = vec![0.0f32, -1.0, -2.0];
    let got = run(&probs, 3, 8, ALLOC_PROPORTIONAL);
    assert_eq!(got, vec![0, 0, 0, 0, 1, 1, 2, 2]);
    // A second hand case: log-probs 0, -ln 2 (shares 2/3, 1/3), K = 5:
    // one each leaves K' = 3; quotas [2.0, 1.0], no remainder. Counts [3, 2].
    let probs = vec![0.0f32, -core::f32::consts::LN_2];
    let got = run(&probs, 2, 5, ALLOC_PROPORTIONAL);
    assert_eq!(got, vec![0, 0, 0, 1, 1]);
}

#[test]
fn larger_probability_never_gets_fewer_trajectories() {
    // Deterministic pseudo-random descending log-probabilities.
    let mut x = 0x1234_5678_9abc_def1u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 32) as u32
    };
    for _ in 0..50 {
        let count = 1 + next() % 8;
        let k = 1 + next() % 64;
        let mut probs = Vec::new();
        let mut v = 0.0f32;
        for _ in 0..count {
            v -= (next() % 1000) as f32 / 500.0;
            probs.push(v);
        }
        for mode in [ALLOC_ROUND_ROBIN, ALLOC_PROPORTIONAL] {
            let got = run(&probs, count, k as usize, mode);
            let counts = counts_of(&got, count as usize);
            for w in 1..counts.len() {
                assert!(
                    counts[w] <= counts[w - 1],
                    "monotonic: {counts:?} from {probs:?} K {k} mode {mode}"
                );
            }
        }
    }
}

#[test]
fn allocation_is_deterministic() {
    let probs = vec![0.3f32, -0.1, -1.7, -0.9];
    for mode in [ALLOC_ROUND_ROBIN, ALLOC_PROPORTIONAL] {
        let first = run(&probs, 4, 19, mode);
        let second = run(&probs, 4, 19, mode);
        assert_eq!(first, second, "mode {mode}");
    }
}

#[test]
fn single_formula_gets_everything() {
    for mode in [ALLOC_ROUND_ROBIN, ALLOC_PROPORTIONAL] {
        assert_eq!(run(&[1.5f32], 1, 8, mode), vec![0; 8], "mode {mode}");
        let full = run_full(&[1.5f32], 1, 8, mode);
        for t in 0..8 {
            check_record(&full, t, 0, 1);
        }
    }
}

#[test]
fn unknown_mode_behaves_as_proportional() {
    let probs = vec![0.0f32, -1.0, -2.0, -3.0];
    assert_eq!(
        run(&probs, 4, 11, 7),
        run(&probs, 4, 11, ALLOC_PROPORTIONAL)
    );
}

#[test]
fn top_count_past_f_is_rejected() {
    // Spec §3.2 (amended): `top_count <= F` is enforced. The lane is a
    // guarded sentinel no-op; the host wrapper reports `Error::Config`.
    let (top, top_counts) = buffers(2);
    let probs = vec![0.0f32, -1.0];
    for mode in [ALLOC_ROUND_ROBIN, ALLOC_PROPORTIONAL] {
        let mut out = vec![0u32; 5 * 12];
        allocate_lane(&top, &top_counts, &probs, &[9], 0, 2, 5, mode, &mut out);
        for t in 0..5 {
            check_record(&out, t, u32::MAX, 2);
        }
        let mut out = vec![0u32; 5 * 12];
        assert!(
            allocate_checked(&top, &top_counts, &probs, &[9], 1, 2, 5, mode, &mut out).is_err(),
            "wrapper rejects, mode {mode}"
        );
    }
}

#[test]
fn k_above_64_is_rejected() {
    // `K = 65`: the lane writes sentinels and the wrapper refuses.
    let (top, top_counts) = buffers(2);
    let probs = vec![0.0f32, -1.0];
    let mut out = vec![0u32; 65 * 12];
    allocate_lane(&top, &top_counts, &probs, &[2], 0, 2, 65, ALLOC_PROPORTIONAL, &mut out);
    for t in 0..65 {
        check_record(&out, t, u32::MAX, 2);
    }
    let mut out = vec![0u32; 65 * 12];
    assert!(
        allocate_checked(&top, &top_counts, &probs, &[2], 1, 2, 65, ALLOC_PROPORTIONAL, &mut out)
            .is_err()
    );
    // The boundary itself works: exactly 64 slots, every one in range.
    let got = run(&probs, 2, 64, ALLOC_PROPORTIONAL);
    assert_eq!(got.len(), 64);
    assert!(got.iter().all(|slot| *slot < 2));
    assert_eq!(ALLOC_K_MAX, 64);
}

#[test]
fn f_above_8_is_rejected() {
    // Nine retained formulas: the lane writes sentinels, the wrapper refuses.
    let (top, top_counts) = buffers(9);
    let probs = vec![0.0f32; 9];
    let mut out = vec![0u32; 9 * 12];
    allocate_lane(&top, &top_counts, &probs, &[9], 0, 9, 9, ALLOC_PROPORTIONAL, &mut out);
    for t in 0..9 {
        check_record(&out, t, u32::MAX, 9);
    }
    let mut out = vec![0u32; 9 * 12];
    assert!(
        allocate_checked(&top, &top_counts, &probs, &[9], 1, 9, 9, ALLOC_PROPORTIONAL, &mut out)
            .is_err()
    );
    assert_eq!(ALLOC_F_MAX, 8);
}

#[test]
fn short_layouts_are_rejected() {
    // `top` / `top_counts` shorter than `[F, 2]` / `[F, 10]`: sentinels in
    // the lane, `Error::Config` in the wrapper.
    let probs = vec![0.0f32, -1.0];
    let short_top = vec![0u32; 3];
    let (_, top_counts) = buffers(2);
    let mut out = vec![0u32; 4 * 12];
    allocate_lane(&short_top, &top_counts, &probs, &[2], 0, 2, 4, ALLOC_PROPORTIONAL, &mut out);
    for t in 0..4 {
        check_record(&out, t, u32::MAX, 2);
    }
    let mut out = vec![0u32; 4 * 12];
    assert!(
        allocate_checked(&short_top, &top_counts, &probs, &[2], 1, 2, 4, ALLOC_PROPORTIONAL, &mut out)
            .is_err()
    );
    let (top, _) = buffers(2);
    let short_counts = vec![0u32; 19];
    let mut out = vec![0u32; 4 * 12];
    assert!(
        allocate_checked(&top, &short_counts, &probs, &[2], 1, 2, 4, ALLOC_PROPORTIONAL, &mut out)
            .is_err()
    );
    // An `out` shorter than `[B, K, 12]` is refused by the wrapper as well.
    let mut out = vec![0u32; 3];
    assert!(
        allocate_checked(&top, &top_counts, &probs, &[2], 1, 2, 4, ALLOC_PROPORTIONAL, &mut out)
            .is_err()
    );
}

#[test]
fn floor_overshoot_reconcile_helper() {
    // Crafted overshoot (the `K = 16_777_221` shape at small scale): floors
    // sum past `rest_k`. The excess is shaved in slot order and no picks
    // remain, while the baselines (outside `floors`) are untouched.
    let mut floors = [6u32, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(reconcile_extra_shares(&mut floors, 2, 3), 0);
    assert_eq!(&floors[..2], &[3, 0]);
    // No overshoot: shares unchanged, the remainder reported exactly.
    let mut floors = [2u32, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(reconcile_extra_shares(&mut floors, 2, 5), 2);
    assert_eq!(&floors[..2], &[2, 1]);
    // Exact fit: nothing to shave, nothing left.
    let mut floors = [3u32, 2, 0, 0, 0, 0, 0, 0];
    assert_eq!(reconcile_extra_shares(&mut floors, 2, 5), 0);
    assert_eq!(&floors[..2], &[3, 2]);
    // Shave spreads across slots in order when the first slot cannot cover
    // the excess alone.
    let mut floors = [1u32, 4, 0, 0, 0, 0, 0, 0];
    assert_eq!(reconcile_extra_shares(&mut floors, 2, 2), 0);
    assert_eq!(&floors[..2], &[0, 2]);
    // Review finding 6: the adversarial sum 2^32 (floors of 2^31 each with
    // rest_k = u32::MAX) shaves one without wrapping and reports 0, in both
    // build profiles (debug would panic on the plain-u32 sum).
    let mut floors = [2_147_483_648u32, 2_147_483_648, 0, 0, 0, 0, 0, 0];
    assert_eq!(reconcile_extra_shares(&mut floors, 2, u32::MAX), 0);
    assert_eq!(&floors[..2], &[2_147_483_647, 2_147_483_648]);
    // Review finding 6: n = 9 cannot index past the fixed array; the lane
    // caps (F <= 8) clamp the scan and the remainder is reported exactly.
    let mut floors = [0u32; 8];
    assert_eq!(reconcile_extra_shares(&mut floors, 9, 5), 5);
    assert_eq!(&floors, &[0u32; 8]);
    let mut floors = [7u32, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(reconcile_extra_shares(&mut floors, 9, 3), 0);
    assert_eq!(&floors[..2], &[3, 0]);
}

#[test]
fn every_retained_keeps_baseline_and_k_is_exact_at_cap() {
    // At the `K = 64` cap every retained formula still owns at least its
    // baseline trajectory and exactly 64 slots are written.
    let prob_sets: Vec<Vec<f32>> = vec![
        vec![0.0, -0.5, -1.0, -1.5],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![3.0, -7.0],
    ];
    for probs in &prob_sets {
        let count = probs.len() as u32;
        let got = run(probs, count, 64, ALLOC_PROPORTIONAL);
        assert_eq!(got.len(), 64);
        let counts = counts_of(&got, count as usize);
        assert_eq!(counts.iter().sum::<u32>(), 64);
        assert!(counts.iter().all(|c| *c >= 1), "{counts:?} keeps baselines");
    }
}

#[test]
fn outside_validated_domain_falls_back_to_round_robin() {
    // NaN, +inf and -inf among the retained entries: the whole spectrum
    // falls back to round robin (never a saturating-cast artefact).
    let cases: Vec<Vec<f32>> = vec![
        vec![0.0, f32::NAN],
        vec![f32::NAN, -1.0],
        vec![f32::INFINITY, -1.0],
        vec![0.0, f32::NEG_INFINITY],
        vec![f32::NEG_INFINITY, f32::NAN],
    ];
    for probs in &cases {
        let got = run(probs, 2, 7, ALLOC_PROPORTIONAL);
        assert_eq!(got, vec![0, 1, 0, 1, 0, 1, 0], "{probs:?}");
    }
    // An entry outside the validated domain past `top_count` is not read and changes nothing.
    let full = vec![0.0f32, -1.0, f32::NAN];
    assert_eq!(
        run(&full, 2, 7, ALLOC_PROPORTIONAL),
        run(&full[..2], 2, 7, ALLOC_PROPORTIONAL)
    );
}

#[test]
fn finite_boundary_falls_back_to_round_robin() {
    // The validated input domain of `top_log_prob` is the open interval
    // (-3e38, 3e38) — the `ms2.rs` FINITE_MAX range test, required by
    // fast-math backends. Finite values at or beyond ±3e38 take the
    // round-robin fallback together with NaN and ±infinity.
    for probs in [
        vec![0.0f32, -3e38],
        vec![0.0f32, 3e38],
        vec![-3e38f32, -3e38],
        vec![3e38f32, 0.0],
    ] {
        let got = run(&probs, 2, 5, ALLOC_PROPORTIONAL);
        assert_eq!(got, vec![0, 1, 0, 1, 0], "{probs:?}");
    }
    // Extreme but strictly inside the domain still allocates proportionally:
    // exp(-3e37) underflows to 0, so slot 0 takes every extra trajectory.
    let got = run(&[0.0f32, -3e37], 2, 5, ALLOC_PROPORTIONAL);
    assert_eq!(got, vec![0, 0, 0, 0, 1]);
}

#[test]
fn checked_wrapper_accepts_legal_inputs() {
    let (top, top_counts) = buffers(3);
    let probs = vec![0.0f32, -1.0, -2.0];
    let mut out = vec![0u32; 8 * 12];
    assert!(
        allocate_checked(&top, &top_counts, &probs, &[3], 1, 3, 8, ALLOC_PROPORTIONAL, &mut out)
            .is_ok()
    );
    assert_eq!(slots_of(&out), vec![0, 0, 0, 0, 1, 1, 2, 2]);
    for (t, slot) in [0, 0, 0, 0, 1, 1, 2, 2].iter().enumerate() {
        check_record(&out, t, *slot, 3);
    }
    assert_eq!(ALLOC_RECORD, 12);
}
