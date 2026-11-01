//! Leakage injection: hand-built splits that break C4 must be rejected by the
//! run-time guard.

use clm_data::types::SampleIndex;
use clm_eval::folds::{Fold, FoldConfig, Scheme, Segment, SegmentKind, make_folds};
use clm_eval::purging::{
    FoldSamples, LeakageError, check_folds_disjoint, check_no_leakage, split_samples,
};

const H: usize = 2;
const E: usize = 1;

/// One fold of a 60-day calendar of 10-day years: train [0,40), val [40,45),
/// cal [45,50), test [50,60).
fn fold() -> Fold {
    let years: Vec<i32> = (0..60).map(|t| 2000 + t / 10).collect();
    let cfg = FoldConfig {
        scheme: Scheme::Expanding,
        first_test_year: 2005,
        last_test_year: 2005,
        val_days: 5,
        cal_days: 5,
        horizon: H,
        embargo: E,
        min_train_days: 1,
    };
    let f = make_folds(&years, 60, &cfg).remove(0);
    assert_eq!(
        (f.train, f.val, f.cal, f.test),
        (
            Segment::new(0, 40),
            Segment::new(40, 45),
            Segment::new(45, 50),
            Segment::new(50, 60)
        )
    );
    f
}

fn si(t: usize) -> SampleIndex {
    SampleIndex::new(0, t)
}

fn all_samples(n: usize) -> Vec<SampleIndex> {
    (0..n).map(si).collect()
}

#[test]
fn clean_split_passes() {
    let f = fold();
    let fs = split_samples(&f, &all_samples(60));
    // train keep: t + 1 + 2 + 1 < 40 -> t <= 35
    assert_eq!(fs.train.last().unwrap().t, 35);
    assert_eq!(fs.val.last().unwrap().t, 40);
    assert_eq!(fs.cal.last().unwrap().t, 45);
    // test keep: t + 3 <= 59 -> t <= 56
    assert_eq!(fs.test.last().unwrap().t, 56);
    check_no_leakage(&f, &fs, H, E).unwrap();
}

#[test]
fn train_label_overlapping_validation_is_rejected() {
    let f = fold();
    let mut fs = split_samples(&f, &all_samples(60));
    // t = 38: label indices 39..=41 reach into val (starts at 40).
    fs.train.push(si(38));
    let err = check_no_leakage(&f, &fs, H, E).unwrap_err();
    assert!(
        matches!(
            err,
            LeakageError::LabelOverlap {
                kind: SegmentKind::Train,
                later: SegmentKind::Val,
                label_end: 41,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn embargo_violation_is_rejected() {
    let f = fold();
    let mut fs = split_samples(&f, &all_samples(60));
    // t = 36: labels 37..=39 < 40 (no overlap) but 39 + e = 40 is not < 40.
    fs.train.push(si(36));
    let err = check_no_leakage(&f, &fs, H, E).unwrap_err();
    assert!(
        matches!(
            err,
            LeakageError::LabelOverlap {
                label_end: 39,
                embargo: 1,
                ..
            }
        ),
        "{err:?}"
    );
    // Without embargo the same sample is fine.
    check_no_leakage(&f, &fs, H, 0).unwrap();
}

#[test]
fn cal_label_reaching_test_is_rejected() {
    let f = fold();
    let mut fs = split_samples(&f, &all_samples(60));
    fs.cal.push(si(48));
    let err = check_no_leakage(&f, &fs, H, E).unwrap_err();
    assert!(
        matches!(
            err,
            LeakageError::LabelOverlap {
                kind: SegmentKind::Cal,
                later: SegmentKind::Test,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn label_reaching_a_non_adjacent_later_segment_is_rejected() {
    // val is empty, so a train label crossing into cal must be caught against cal and
    // against the empty val (whose start equals cal's start).
    let mut f = fold();
    f.val = Segment::new(45, 45);
    f.train = Segment::new(0, 45);
    let fs = FoldSamples {
        train: vec![si(43)],
        ..Default::default()
    };
    assert!(check_no_leakage(&f, &fs, H, E).is_err());
}

#[test]
fn guard_uses_the_run_horizon_not_the_fold() {
    // The split was made for h = 2; checking it with h = 5 must fail at the boundary.
    let f = fold();
    let fs = split_samples(&f, &all_samples(60));
    assert!(check_no_leakage(&f, &fs, 5, E).is_err());
    assert!(check_no_leakage(&f, &fs, H, E + 3).is_err());
}

#[test]
fn overlapping_segments_are_rejected() {
    let mut f = fold();
    f.val = Segment::new(39, 45); // overlaps train [0, 40)
    let err = check_no_leakage(&f, &FoldSamples::default(), H, E).unwrap_err();
    assert!(
        matches!(
            err,
            LeakageError::SegmentOrder {
                first: SegmentKind::Train,
                second: SegmentKind::Val,
                ..
            }
        ),
        "{err:?}"
    );
    let mut f = fold();
    f.cal = Segment::new(52, 55); // after test start
    assert!(matches!(
        check_no_leakage(&f, &FoldSamples::default(), H, E),
        Err(LeakageError::SegmentOrder { .. })
    ));
}

#[test]
fn segment_past_n_is_rejected() {
    let mut f = fold();
    f.test = Segment::new(50, 61);
    assert!(matches!(
        check_no_leakage(&f, &FoldSamples::default(), H, E),
        Err(LeakageError::SegmentOutOfRange {
            kind: SegmentKind::Test,
            ..
        })
    ));
}

#[test]
fn test_label_past_end_is_rejected() {
    let f = fold();
    let mut fs = split_samples(&f, &all_samples(60));
    fs.test.push(si(57)); // label end 57 + 3 = 60 > 59
    let err = check_no_leakage(&f, &fs, H, E).unwrap_err();
    assert!(
        matches!(
            err,
            LeakageError::TestLabelPastEnd {
                label_end: 60,
                n_days: 60,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn sample_in_wrong_segment_is_rejected() {
    let f = fold();
    let fs = FoldSamples {
        train: vec![si(52)], // a test day listed as training
        ..Default::default()
    };
    assert!(matches!(
        check_no_leakage(&f, &fs, H, E),
        Err(LeakageError::SampleOutsideSegment {
            kind: SegmentKind::Train,
            ..
        })
    ));
}

#[test]
fn overlapping_test_segments_across_folds_are_rejected() {
    let a = fold();
    let mut b = fold();
    b.test = Segment::new(55, 60);
    assert!(matches!(
        check_folds_disjoint(&[a.clone(), b]),
        Err(LeakageError::TestSegmentsOverlap { .. })
    ));
    // out of order
    let mut c = fold();
    c.test = Segment::new(10, 20);
    assert!(check_folds_disjoint(&[a, c]).is_err());
}
