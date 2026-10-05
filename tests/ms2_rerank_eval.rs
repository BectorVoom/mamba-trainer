//! Host-only tests for [`mamba3::models::ms2::rerank_eval`]: hand-computed
//! expectations for `roc_auc`, `top1_precision`, `precision_at` and
//! `paired_bootstrap`, including ties, a single-class input, groups of one
//! and bootstrap determinism.
//!
//! No device code: this binary runs without any backend feature.

use mamba3::models::ms2::calibration::brier_by_stratum;
use mamba3::models::ms2::rerank_eval::{
    FitProvenance, SearchPolicyInputs, build_search_policy, check_generator_fit, find_shared_key,
    paired_bootstrap, precision_at, roc_auc, top1_precision, trace_atom_count, validate_bootstrap,
};

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() < 1e-12
}

#[test]
fn auc_hand_computed_with_ties() {
    // Scores [0.1(n), 0.4(n), 0.35(p), 0.8(p)]: ascending ranks 1, 3, 2, 4;
    // positive ranks 2 + 4 = 6; AUC = (6 − 3) / 4 = 0.75.
    let auc = roc_auc(&[0.1, 0.4, 0.35, 0.8], &[0.0, 0.0, 1.0, 1.0]).unwrap();
    assert!(close(auc, 0.75), "AUC without ties is 0.75, got {auc}");
    // Tied pair at 0.5 shares the average rank 1.5: positive ranks
    // 1.5 + 3 = 4.5; AUC = (4.5 − 3) / 2 = 0.75.
    let tied = roc_auc(&[0.5, 0.5, 0.9], &[0.0, 1.0, 1.0]).unwrap();
    assert!(close(tied, 0.75), "tied AUC is 0.75, got {tied}");
    // All tied: every rank is the average (n + 1) / 2, AUC is 0.5.
    let all_tied = roc_auc(&[0.2, 0.2, 0.2, 0.2], &[0.0, 1.0, 0.0, 1.0]).unwrap();
    assert!(close(all_tied, 0.5), "all-tied AUC is 0.5, got {all_tied}");
    // Perfect separation is 1.0, perfect inversion is 0.0.
    assert!(close(
        roc_auc(&[0.1, 0.2, 0.8, 0.9], &[0.0, 0.0, 1.0, 1.0]).unwrap(),
        1.0
    ));
    assert!(close(
        roc_auc(&[0.9, 0.8, 0.2, 0.1], &[0.0, 0.0, 1.0, 1.0]).unwrap(),
        0.0
    ));
}

#[test]
fn auc_single_class_is_none() {
    assert_eq!(roc_auc(&[0.1, 0.9], &[0.0, 0.0]), None);
    assert_eq!(roc_auc(&[0.1, 0.9], &[1.0, 1.0]), None);
    assert_eq!(roc_auc(&[], &[]), None);
}

#[test]
fn top1_ties_break_to_smaller_index() {
    // Group 0: the tie at score 0.7 breaks to index 0 (label 1): hit.
    // Group 1: the top score sits on index 3 (label 0): miss.
    // Group 2: a single example (label 1): hit.
    let scores = vec![0.7, 0.7, 0.2, 0.9, 0.1, 0.4];
    let labels = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let groups = vec![0, 0, 1, 1, 1, 2];
    let (prec, n) = top1_precision(&scores, &labels, &groups);
    assert_eq!(n, 3);
    assert!(close(prec, 2.0 / 3.0), "top-1 precision is 2/3, got {prec}");
    // The second group on its own: top score at its first example.
    let (solo, n) = top1_precision(&[0.3], &[1.0], &[7]);
    assert_eq!((solo, n), (1.0, 1));
    // No group at all.
    assert_eq!(top1_precision(&[], &[], &[]), (0.0, 0));
}

#[test]
fn precision_at_matches_hand_counts() {
    // Group 0 (scores 0.5, 0.9, 0.1; labels 1, 0, 1): order 0.9(miss),
    // 0.5(hit), 0.1(hit). Group 1 (scores 0.4; label 0): one example.
    let scores = vec![0.5, 0.9, 0.1, 0.4];
    let labels = vec![1.0, 0.0, 1.0, 0.0];
    let groups = vec![0, 0, 0, 1];
    // r = 1 is the top-1 fraction: 0/1 and 0/1 -> 0.0.
    assert!(close(precision_at(&scores, &labels, &groups, 1), 0.0));
    let (top1, _) = top1_precision(&scores, &labels, &groups);
    assert!(close(precision_at(&scores, &labels, &groups, 1), top1));
    // r = 2: group 0 has 1/2, group 1 has 0/1 -> mean 0.25.
    assert!(close(precision_at(&scores, &labels, &groups, 2), 0.25));
    // r = 8 (past every group size) is the per-group base rate mean:
    // group 0 has 2/3, group 1 has 0/1 -> (2/3 + 0) / 2.
    assert!(close(precision_at(&scores, &labels, &groups, 8), 1.0 / 3.0));
    // r = 0 and empty groups give 0.0.
    assert_eq!(precision_at(&scores, &labels, &groups, 0), 0.0);
    assert_eq!(precision_at(&[], &[], &[], 4), 0.0);
}

#[test]
fn paired_bootstrap_point_and_determinism() {
    // Molecule 0: paired diffs 0.5 and 0.5, molecule 1: paired diff -1.0.
    // Point: (0.5 − 1.0) / 2 = −0.25.
    let a = vec![0.5, 1.0, 0.0];
    let b = vec![0.0, 0.5, 1.0];
    let mol = vec![0, 0, 1];
    let (point, lo, hi) = paired_bootstrap(&a, &b, &mol, 2000, 42).expect("paired groups exist");
    assert!(close(point, -0.25), "paired point is −0.25, got {point}");
    assert!(lo <= point && point <= hi, "interval [{lo}, {hi}] misses {point}");
    // Deterministic for a seed, with a non-degenerate interval on spread data.
    assert_eq!(
        paired_bootstrap(&a, &b, &mol, 2000, 42),
        paired_bootstrap(&a, &b, &mol, 2000, 42)
    );
    assert!(lo < hi, "spread diffs give a non-degenerate interval");
    // No group at all: no paired difference.
    assert_eq!(paired_bootstrap(&[], &[], &[], 100, 1), None);
}

#[test]
fn paired_bootstrap_zero_repetitions_returns_point() {
    // B3: `n == 0` returns the measured point with `lo == hi == point` —
    // never zero in place of a measured value.
    let a = vec![0.5, 1.0, 0.0];
    let b = vec![0.0, 0.5, 1.0];
    let mol = vec![0, 0, 1];
    assert_eq!(
        paired_bootstrap(&a, &b, &mol, 0, 1),
        Some((-0.25, -0.25, -0.25))
    );
    assert!(validate_bootstrap(0).is_err(), "--bootstrap 0 is rejected");
    assert!(
        validate_bootstrap(0).unwrap_err().contains("non-zero"),
        "the refusal says non-zero"
    );
    assert!(validate_bootstrap(100).is_ok());
}

#[test]
fn paired_bootstrap_no_paired_group_is_none() {
    // B4 (reviewer's case): `a=[1, NaN]`, `b=[NaN, 0]`, molecules `[0, 0]`.
    // No spectrum is defined in both rankings, so there is no paired
    // difference — not a `(1, 1, 1)` from averaging different spectra.
    assert_eq!(
        paired_bootstrap(&[1.0, f64::NAN], &[f64::NAN, 0.0], &[0, 0], 100, 7),
        None
    );
}

#[test]
fn paired_bootstrap_asymmetric_nan_keeps_paired_groups() {
    // B4: groups defined in only one ranking are skipped, but paired groups
    // still aggregate. Groups 0 and 2 are paired (diffs 1.0 and −1.0 in
    // molecules 0 and 1); groups 1 (a NaN) and 3 (b NaN) are skipped.
    let a = vec![1.0, f64::NAN, 0.0, 0.5];
    let b = vec![0.0, 0.5, 1.0, f64::NAN];
    let mol = vec![0, 0, 1, 1];
    let (point, lo, hi) =
        paired_bootstrap(&a, &b, &mol, 2000, 3).expect("two paired groups");
    assert!(close(point, 0.0), "paired point is 0.0, got {point}");
    assert!(lo <= point && point <= hi);
    // A single surviving paired group gives its own difference as the point.
    let (solo, solo_lo, solo_hi) = paired_bootstrap(
        &[1.0, f64::NAN, 0.5],
        &[0.0, 0.5, f64::NAN],
        &[0, 0, 1],
        100,
        3,
    )
    .expect("one paired group");
    assert!(close(solo, 1.0), "single paired diff is 1.0, got {solo}");
    assert!(solo_lo <= solo && solo <= solo_hi);
}

#[test]
fn shared_key_finds_first_overlap_in_order() {
    // B1: the key-overlap check names the first shared molecule key.
    let a = ["m1".to_string(), "m2".to_string(), "m3".to_string()];
    let b = ["m9".to_string(), "m3".to_string(), "m2".to_string()];
    assert_eq!(
        find_shared_key(&a, &b),
        Some("m2".to_string()),
        "first shared key in `a` order"
    );
    assert_eq!(
        find_shared_key(&a, &["m9".to_string()]),
        None,
        "disjoint key sets"
    );
    assert_eq!(find_shared_key(&[], &b), None, "empty input is disjoint");
}

#[test]
fn generator_fit_provenance_match_and_mismatch() {
    // B1: recorded fit provenance must match the supplied `--generator-fit`
    // (name or SHA-256); nothing recorded is reported, not refused.
    assert_eq!(
        check_generator_fit(Some("fit.json"), Some("abc123"), "fit.json", "abc123"),
        Ok(FitProvenance::Matches)
    );
    let err = check_generator_fit(Some("fit.json"), Some("abc123"), "other.json", "abc123")
        .unwrap_err();
    assert!(
        err.contains("fit.json") && err.contains("other.json"),
        "name mismatch names both sides: {err}"
    );
    let err =
        check_generator_fit(Some("fit.json"), Some("abc123"), "fit.json", "def456").unwrap_err();
    assert!(
        err.contains("abc123") && err.contains("def456"),
        "sha mismatch names both sides: {err}"
    );
    assert_eq!(
        check_generator_fit(None, None, "fit.json", "abc123"),
        Ok(FitProvenance::Unrecorded)
    );
}

#[test]
fn search_policy_separates_lane_visit_limits() {
    // B6: two configs differing only in `enum_lane_visits_max` get different
    // keys, so calibration cannot cross them.
    let base = SearchPolicyInputs {
        formula_source: "enumerate",
        window_m: 2048,
        formula_rows_scored_max: 4096,
        formula_rows_visited_max: u32::MAX,
        enum_lane_visits_max: 4096,
        enum_lanes_max: 262_144,
        allocation: "round_robin",
        identity: "graph",
        returned: 8,
        table_sha256_16: "0123456789abcdef",
        enum_domain_sha256_16: Some("aaaabbbbccccdddd"),
        enum_bounds_sha256_16: Some("1111222233334444"),
    };
    let same = SearchPolicyInputs { enum_lane_visits_max: 4096, ..base };
    assert_eq!(build_search_policy(&base), build_search_policy(&same));
    let other_lane_visits = SearchPolicyInputs { enum_lane_visits_max: 1, ..base };
    assert_ne!(
        build_search_policy(&base),
        build_search_policy(&other_lane_visits),
        "lane visit limits 1 vs 4096 must key differently"
    );
}

#[test]
fn size_stratum_uses_trace_atoms_not_parent_replay() {
    // B5 (reviewer's case): a finished, valid three-atom candidate generated
    // under an incorrect retained formula fails replay under the true
    // parent's composition, so evaluation reports atoms 0 — but the
    // candidate's own trace still has three atoms. With one positive and
    // this negative, both at probability 0.9, the 3–5 stratum Brier is 0.41
    // (not 0.01 with the negative omitted).
    let words: Vec<u32> = vec![
        2, 4, 0, 0, // ADD_ATOM below length
        2, 4, 0, 0, // ADD_ATOM below length
        2, 4, 0, 0, // ADD_ATOM below length
        4, 0, 0, 0, // STOP
    ];
    assert_eq!(trace_atom_count(&words, 4), 3);
    assert_eq!(
        trace_atom_count(&words, 2),
        2,
        "steps at or after length add no atoms"
    );
    assert_eq!(trace_atom_count(&[1, 0, 0, 0, 4, 0, 0, 0], 2), 0);
    let brier = brier_by_stratum(&[0.9, 0.9], &[1.0, 0.0], &[3, 3]).unwrap();
    assert!(
        (brier[0] - 0.41).abs() < 1e-12,
        "3–5 stratum Brier with both candidates is 0.41, got {}",
        brier[0]
    );
    let dropped = brier_by_stratum(&[0.9, 0.9], &[1.0, 0.0], &[3, 0]).unwrap();
    assert!(
        (dropped[0] - 0.01).abs() < 1e-12,
        "parent-replay atoms would omit the negative (0.01), got {}",
        dropped[0]
    );
}
