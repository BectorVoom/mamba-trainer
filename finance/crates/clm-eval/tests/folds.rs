//! Walk-forward folds: hand-worked calendars and seeded property tests of C4
//! properties 1, 2, 3, 5 checked directly from the definitions.
// Conditions are written as in docs/CONTRACTS.md (`t + 1 + h <= N - 1`), on purpose.
#![allow(clippy::int_plus_one, clippy::needless_range_loop)]

use clm_data::types::SampleIndex;
use clm_eval::folds::{Fold, FoldConfig, Scheme, Segment, SegmentKind, make_folds, render_folds};
use clm_eval::purging::{check_folds_disjoint, check_no_leakage, split_samples};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 6 "years" 2000..=2005 of 10 days each (N = 60).
fn six_years() -> Vec<i32> {
    (0..60).map(|t| 2000 + t / 10).collect()
}

fn base_cfg(scheme: Scheme) -> FoldConfig {
    FoldConfig {
        scheme,
        first_test_year: 2000,
        last_test_year: 2006,
        val_days: 2,
        cal_days: 1,
        horizon: 1,
        embargo: 1,
        min_train_days: 5,
    }
}

#[test]
fn expanding_hand_worked() {
    let folds = make_folds(&six_years(), 60, &base_cfg(Scheme::Expanding));
    // 2000: ts = 0 < cal + val = 3 -> skipped; 2006 not in the calendar -> skipped.
    // train/val/cal keep: t + 1 + 1 + 1 < s_next -> t < s_next - 3; test: t + 2 <= 59.
    let expect = [
        "fold 2001 train 0 7 val 7 9 cal 9 10 test 10 20 keep 0 4 7 7 9 9 10 20",
        "fold 2002 train 0 17 val 17 19 cal 19 20 test 20 30 keep 0 14 17 17 19 19 20 30",
        "fold 2003 train 0 27 val 27 29 cal 29 30 test 30 40 keep 0 24 27 27 29 29 30 40",
        "fold 2004 train 0 37 val 37 39 cal 39 40 test 40 50 keep 0 34 37 37 39 39 40 50",
        "fold 2005 train 0 47 val 47 49 cal 49 50 test 50 60 keep 0 44 47 47 49 49 50 58",
    ];
    assert_eq!(render_folds(&folds), expect);
    assert!(folds.iter().all(|f| f.n_days == 60));
}

#[test]
fn rolling_hand_worked_and_min_train_skip() {
    let mut cfg = base_cfg(Scheme::Rolling { train_days: 8 });
    cfg.min_train_days = 8;
    let folds = make_folds(&six_years(), 60, &cfg);
    // 2001: train [max(0, 7-8), 7) = [0, 7), 7 days < 8 -> skipped.
    let expect = [
        "fold 2002 train 9 17 val 17 19 cal 19 20 test 20 30 keep 9 14 17 17 19 19 20 30",
        "fold 2003 train 19 27 val 27 29 cal 29 30 test 30 40 keep 19 24 27 27 29 29 30 40",
        "fold 2004 train 29 37 val 37 39 cal 39 40 test 40 50 keep 29 34 37 37 39 39 40 50",
        "fold 2005 train 39 47 val 47 49 cal 49 50 test 50 60 keep 39 44 47 47 49 49 50 58",
    ];
    assert_eq!(render_folds(&folds), expect);
}

#[test]
fn non_empty_val_cal_keep_and_empty_train_keep() {
    let cfg = FoldConfig {
        scheme: Scheme::Expanding,
        first_test_year: 2001,
        last_test_year: 2001,
        val_days: 5,
        cal_days: 4,
        horizon: 1,
        embargo: 0,
        min_train_days: 1,
    };
    let folds = make_folds(&six_years(), 60, &cfg);
    // cal [6,10), val [1,6), train [0,1); keep t + 2 < s_next.
    assert_eq!(
        render_folds(&folds),
        ["fold 2001 train 0 1 val 1 6 cal 6 10 test 10 20 keep 0 0 1 4 6 8 10 20"]
    );
}

#[test]
fn gaps_in_years_and_short_last_year() {
    // years 2000 (10 days), 2002 (10 days), 2003 (3 days); 2001 is absent.
    let mut years: Vec<i32> = vec![2000; 10];
    years.extend([2002; 10]);
    years.extend([2003; 3]);
    let cfg = FoldConfig {
        scheme: Scheme::Expanding,
        first_test_year: 2001,
        last_test_year: 2003,
        val_days: 1,
        cal_days: 1,
        horizon: 2,
        embargo: 0,
        min_train_days: 0,
    };
    let folds = make_folds(&years, 23, &cfg);
    // 2001 skipped (absent). 2002: ts 10, cal [9,10), val [8,9), train [0,8).
    // keep train t + 3 < 8 -> [0,5); test t + 3 <= 22 -> t <= 19 -> [10,20).
    // 2003: ts 20, test [20,23), keep t <= 19 -> empty [20,20).
    assert_eq!(
        render_folds(&folds),
        [
            "fold 2002 train 0 8 val 8 9 cal 9 10 test 10 20 keep 0 5 8 8 9 9 10 20",
            "fold 2003 train 0 18 val 18 19 cal 19 20 test 20 23 keep 0 15 18 18 19 19 20 20",
        ]
    );
}

#[test]
fn no_folds_when_range_empty_or_reversed() {
    let mut cfg = base_cfg(Scheme::Expanding);
    cfg.first_test_year = 2004;
    cfg.last_test_year = 2003;
    assert!(make_folds(&six_years(), 60, &cfg).is_empty());
    assert!(make_folds(&[], 0, &base_cfg(Scheme::Expanding)).is_empty());
}

#[test]
#[should_panic(expected = "non-decreasing")]
fn decreasing_years_panic() {
    make_folds(&[2001, 2000], 2, &base_cfg(Scheme::Expanding));
}

// ---------------------------------------------------------------- properties

/// A random run-length calendar (some years may be absent) and config.
pub fn random_case(rng: &mut StdRng) -> (Vec<i32>, FoldConfig) {
    let y0 = rng.random_range(1990..2010);
    let nyears = rng.random_range(1..9);
    let mut years = Vec::new();
    for k in 0..nyears {
        let c = if rng.random_bool(0.15) {
            0
        } else {
            rng.random_range(1..40)
        };
        years.extend(std::iter::repeat_n(y0 + k, c));
    }
    let scheme = if rng.random_bool(0.5) {
        Scheme::Expanding
    } else {
        Scheme::Rolling {
            train_days: rng.random_range(0..80),
        }
    };
    let first = y0 + rng.random_range(-2..nyears + 1);
    let cfg = FoldConfig {
        scheme,
        first_test_year: first,
        last_test_year: first + rng.random_range(-1..nyears + 2),
        val_days: rng.random_range(0..15),
        cal_days: rng.random_range(0..15),
        horizon: rng.random_range(1..10),
        embargo: rng.random_range(0..6),
        min_train_days: rng.random_range(0..50),
    };
    (years, cfg)
}

/// Independent re-derivation of which test years emit a fold (signed arithmetic).
fn expected_emitted(years: &[i32], cfg: &FoldConfig) -> Vec<i32> {
    let n = years.len() as i64;
    let mut out = Vec::new();
    for y in cfg.first_test_year..=cfg.last_test_year {
        let Some(ts) = years.iter().position(|&v| v == y) else {
            continue;
        };
        let ts = ts as i64;
        let vs = ts - cfg.cal_days as i64 - cfg.val_days as i64;
        if vs < 0 {
            continue;
        }
        let train_len = match cfg.scheme {
            Scheme::Expanding => vs,
            Scheme::Rolling { train_days } => vs - (vs - train_days as i64).max(0),
        };
        if train_len < cfg.min_train_days as i64 {
            continue;
        }
        let _ = n;
        out.push(y);
    }
    out
}

fn check_properties(years: &[i32], cfg: &FoldConfig, folds: &[Fold]) {
    let n = years.len();
    let (h, e) = (cfg.horizon as i64, cfg.embargo as i64);
    let ctx = format!("years={years:?} cfg={cfg:?}");
    assert_eq!(
        folds.iter().map(|f| f.test_year).collect::<Vec<_>>(),
        expected_emitted(years, cfg),
        "{ctx}"
    );
    for f in folds {
        let segs = f.segments();
        // Property 1: well-formed, inside [0, N), ordered, disjoint, contiguous.
        for s in segs {
            assert!(s.start <= s.end && s.end <= n, "{ctx}");
        }
        for i in 0..4 {
            for j in i + 1..4 {
                assert!(segs[i].end <= segs[j].start, "{ctx}");
            }
        }
        for t in 0..n {
            let owners = segs.iter().filter(|s| s.contains(t)).count();
            assert!(owners <= 1, "{ctx}");
        }
        // Test is exactly the year.
        for t in 0..n {
            assert_eq!(f.test.contains(t), years[t] == f.test_year, "{ctx}");
        }
        // Lengths per definition.
        assert_eq!(f.cal.len(), cfg.cal_days, "{ctx}");
        assert_eq!(f.val.len(), cfg.val_days, "{ctx}");
        if let Scheme::Rolling { train_days } = cfg.scheme {
            assert!(f.train.len() <= train_days, "{ctx}");
        } else {
            assert_eq!(f.train.start, 0, "{ctx}");
        }
        // Keep sets equal the definition, and property 2 holds for every kept day.
        for (i, &kind) in SegmentKind::ALL.iter().enumerate() {
            let seg = segs[i];
            for t in seg.days() {
                let ti = t as i64;
                let def = if i < 3 {
                    ti + 1 + h + e < segs[i + 1].start as i64
                } else {
                    ti + 1 + h <= n as i64 - 1
                };
                assert_eq!(f.is_kept(t), def, "{ctx} t={t} kind={kind:?}");
                assert_eq!(f.kept(kind).contains(t), def, "{ctx}");
                if def {
                    for later in &segs[i + 1..] {
                        for li in ti + 1..=ti + 1 + h {
                            assert!(li < later.start as i64 - e, "{ctx}");
                            for d in later.days() {
                                assert!(later.start as i64 - e <= d as i64, "{ctx}");
                            }
                        }
                    }
                    if i == 3 {
                        assert!(ti + 1 + h < n as i64, "{ctx}");
                    }
                }
            }
            // keep is a prefix of the segment
            let k = f.kept(kind);
            assert!(
                k.start == seg.start && k.end <= seg.end.max(seg.start),
                "{ctx}"
            );
        }
        assert_eq!(f.segment_of(n), None);
        // split + run-time guard agree with the definition.
        let samples: Vec<SampleIndex> = (0..n)
            .flat_map(|t| (0..2).map(move |s| SampleIndex::new(s, t)))
            .collect();
        let fs = split_samples(f, &samples);
        check_no_leakage(f, &fs, cfg.horizon, cfg.embargo)
            .unwrap_or_else(|err| panic!("{ctx}: {err}"));
        let total = fs.train.len() + fs.val.len() + fs.cal.len() + fs.test.len();
        let kept_days: usize = f.keep.iter().map(Segment::len).sum();
        assert_eq!(total, 2 * kept_days, "{ctx}");
    }
    // Property 3: test segments disjoint and increasing.
    check_folds_disjoint(folds).unwrap();
    for w in folds.windows(2) {
        assert!(w[0].test.end <= w[1].test.start, "{ctx}");
        // Property 5 (expanding): train(k) ⊆ train(k+1).
        if cfg.scheme == Scheme::Expanding {
            assert!(
                w[1].train.start <= w[0].train.start && w[0].train.end <= w[1].train.end,
                "{ctx}"
            );
        }
    }
}

#[test]
fn properties_hold_for_random_configs() {
    let mut rng = StdRng::seed_from_u64(20261022);
    let mut emitted = 0;
    for _ in 0..3000 {
        let (years, cfg) = random_case(&mut rng);
        let folds = make_folds(&years, years.len(), &cfg);
        emitted += folds.len();
        check_properties(&years, &cfg, &folds);
    }
    assert!(
        emitted > 1000,
        "random cases should emit many folds, got {emitted}"
    );
}
