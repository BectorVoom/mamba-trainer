//! Stationary bootstrap and comparison helpers (C6).

use clm_data::types::Action;
use clm_eval::backtest::{BacktestError, DailySeries, backtest_metrics, run_backtest};
use clm_eval::folds::Segment;
use clm_eval::statistics::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn cfg(seed: u64) -> BootstrapConfig {
    BootstrapConfig {
        replicates: 500,
        seed,
        ..Default::default()
    }
}

fn noise(seed: u64, n: usize, mu: f64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| mu + rng.random_range(-0.02..0.02)).collect()
}

#[test]
fn defaults_match_contract() {
    let d = BootstrapConfig::default();
    assert_eq!((d.replicates, d.mean_block, d.level), (2000, 20.0, 0.95));
}

#[test]
fn constant_series_has_zero_width_intervals() {
    let r = vec![0.0013; 300];
    let b = bootstrap_series(&r, &cfg(1)).unwrap();
    assert_eq!(b.mean.lo, b.mean.hi);
    assert_eq!(b.mean.lo, b.mean.estimate);
    assert_eq!(b.sharpe.lo, b.sharpe.hi);
    assert_eq!(b.sharpe.estimate, 0.0); // std = 0 -> Sharpe defined as 0
}

#[test]
fn paired_difference_of_identical_series_is_zero() {
    let a = noise(3, 400, 0.0005);
    let days: Vec<usize> = (10..410).collect();
    let p = paired_bootstrap(&days, &a, &days, &a, &cfg(5)).unwrap();
    for iv in [p.sharpe_diff, p.mean_diff] {
        assert_eq!((iv.estimate, iv.lo, iv.hi), (0.0, 0.0, 0.0));
    }
    assert_eq!(p.sharpe_a, p.sharpe_b);
    assert_eq!(p.n_days, 400);
}

#[test]
fn paired_uses_same_indices() {
    // b = a + c (constant shift): mean(A - B) is exactly -c in every resample.
    let a = noise(4, 300, 0.0);
    let b: Vec<f64> = a.iter().map(|x| x + 0.25).collect();
    let days: Vec<usize> = (0..300).collect();
    let p = paired_bootstrap(&days, &a, &days, &b, &cfg(6)).unwrap();
    assert!((p.mean_diff.lo + 0.25).abs() < 1e-12 && (p.mean_diff.hi + 0.25).abs() < 1e-12);
}

#[test]
fn seeded_reproducibility() {
    let a = noise(8, 250, 0.001);
    let x = bootstrap_series(&a, &cfg(42)).unwrap();
    let y = bootstrap_series(&a, &cfg(42)).unwrap();
    let z = bootstrap_series(&a, &cfg(43)).unwrap();
    assert_eq!(x, y);
    assert_ne!(x, z);
    assert!(x.mean.lo <= x.mean.estimate && x.mean.estimate <= x.mean.hi);
    assert!(x.sharpe.lo < x.sharpe.hi);
}

#[test]
fn mean_block_length_is_about_twenty() {
    let mut rng = StdRng::seed_from_u64(77);
    let n = 200_000;
    let idx = stationary_bootstrap_indices(n, 20.0, &mut rng);
    assert_eq!(idx.len(), n);
    assert!(idx.iter().all(|&i| i < n));
    let breaks = idx.windows(2).filter(|w| w[1] != (w[0] + 1) % n).count();
    let mean_len = n as f64 / (breaks + 1) as f64;
    assert!(
        (mean_len - 20.0).abs() < 1.0,
        "mean block length {mean_len}"
    );
}

#[test]
fn misaligned_days_are_an_error() {
    let a = noise(1, 10, 0.0);
    let d1: Vec<usize> = (0..10).collect();
    let d2: Vec<usize> = (1..11).collect();
    assert!(matches!(
        paired_bootstrap(&d1, &a, &d2, &a, &cfg(1)),
        Err(StatsError::DaysDiffer(0))
    ));
    assert!(matches!(
        paired_bootstrap(&d1, &a, &d1[..9], &a[..9], &cfg(1)),
        Err(StatsError::DaysDiffer(9))
    ));
    assert!(matches!(
        bootstrap_series(&[], &cfg(1)),
        Err(StatsError::Empty)
    ));
    let bad = BootstrapConfig {
        replicates: 0,
        ..Default::default()
    };
    assert!(matches!(
        bootstrap_series(&a, &bad),
        Err(StatsError::BadConfig(_))
    ));
}

#[test]
fn per_year_comparison_hand_computed() {
    let c = paired_by_year(
        &[2010, 2011, 2012],
        &[1.0, 0.5, 0.2],
        &[0.5, 0.5, 0.4],
        true,
    )
    .unwrap();
    assert_eq!((c.wins_a, c.ties), (1, 1));
    // diffs 0.5, 0, -0.2: mean 0.1, sample sd
    assert!((c.mean_diff - 0.1).abs() < 1e-15);
    let sd = ((0.16 + 0.01 + 0.09) / 2.0f64).sqrt();
    assert!((c.sd_diff - sd).abs() < 1e-12);
    // log loss: lower is better
    let l = paired_by_year(&[1, 2], &[0.9, 1.0], &[1.0, 0.95], false).unwrap();
    assert_eq!(l.wins_a, 1);
}

#[test]
fn paired_sharpe_by_year_splits_on_calendar() {
    let year_of_day: Vec<i32> = (0..20).map(|d| 2000 + d / 10).collect();
    let days: Vec<usize> = (2..18).collect();
    let mk = |r: Vec<f64>| DailySeries {
        day: days.clone(),
        turnover: vec![0.0; r.len()],
        gross_exposure: vec![1.0; r.len()],
        ret: r,
    };
    let a = mk(noise(1, 16, 0.002));
    let b = mk(noise(2, 16, 0.0));
    let c = paired_sharpe_by_year(&a, &b, &year_of_day).unwrap();
    assert_eq!(c.years, vec![2000, 2001]);
    let (_, groups) = split_by_year(&a, &year_of_day);
    assert_eq!(groups[0].len(), 8);
    assert_eq!(groups[1].len(), 8);
}

#[test]
fn cost_sensitivity_grid() {
    let opens = vec![vec![
        10.0, 10.0, 10.5, 10.2, 10.8, 10.4, 11.0, 11.2, 11.1, 11.5,
    ]];
    let dec = vec![vec![
        Some(Action::Buy),
        Some(Action::Sell),
        Some(Action::Buy),
        Some(Action::Sell),
        Some(Action::Buy),
        Some(Action::Hold),
        Some(Action::Sell),
    ]];
    let pts = cost_sensitivity(|k| -> Result<_, BacktestError> {
        Ok(backtest_metrics(
            &run_backtest(&opens, Segment::new(0, 7), 10, &dec, k)?.daily,
        ))
    })
    .unwrap();
    assert_eq!(
        pts.iter().map(|p| p.kappa_bps).collect::<Vec<_>>(),
        COST_GRID_BPS.to_vec()
    );
    assert!((pts[2].kappa - 5e-4).abs() < 1e-18);
    assert!(
        pts.windows(2)
            .all(|w| w[1].metrics.ann_return < w[0].metrics.ann_return)
    );
}

#[test]
fn regime_flags_and_split() {
    let (m, flags) = regime_flags(&[0.1, 0.3, 0.2, 0.25], &[0.1, 0.2, 0.3, 0.4]).unwrap();
    assert!((m - 0.25).abs() < 1e-15);
    assert_eq!(flags, vec![false, true, false, false]); // strictly above the median
    assert!(regime_flags(&[f64::NAN], &[0.1]).is_err());
    assert!(regime_flags(&[0.1], &[]).is_err());

    let daily = DailySeries {
        day: vec![1, 2, 3, 4],
        ret: vec![0.01, -0.02, 0.03, 0.0],
        turnover: vec![0.0; 4],
        gross_exposure: vec![1.0; 4],
    };
    let r = regime_split(&daily, &flags).unwrap();
    assert_eq!(r.high.n_days, 1);
    assert_eq!(r.low.n_days, 3);
    assert!((r.high.ann_return - 252.0 * -0.02).abs() < 1e-12);
    assert!(regime_split(&daily, &[true]).is_err());
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(median(&[]), None);
}

// ------------------------------------------------------ non-finite returns are rejected

const BAD: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

#[test]
fn bootstrap_rejects_non_finite() {
    for v in BAD {
        let mut a = noise(1, 50, 0.0);
        a[17] = v;
        assert_eq!(
            bootstrap_series(&a, &cfg(1)),
            Err(StatsError::NonFinite(17)),
            "{v}"
        );
        assert_eq!(
            bootstrap_statistic(&a, &|x: &[f64]| x[0], &cfg(1)),
            Err(StatsError::NonFinite(17))
        );
    }
}

#[test]
fn paired_bootstrap_rejects_non_finite_in_either_series() {
    let days: Vec<usize> = (0..40).collect();
    for v in BAD {
        let good = noise(2, 40, 0.0);
        let mut bad = good.clone();
        bad[3] = v;
        assert_eq!(
            paired_bootstrap(&days, &bad, &days, &good, &cfg(1)),
            Err(StatsError::NonFinite(3))
        );
        assert_eq!(
            paired_bootstrap(&days, &good, &days, &bad, &cfg(1)),
            Err(StatsError::NonFinite(3))
        );
        let s = |r: Vec<f64>| DailySeries {
            day: days.clone(),
            turnover: vec![0.0; 40],
            gross_exposure: vec![0.0; 40],
            ret: r,
        };
        assert_eq!(
            paired_bootstrap_series(&s(good.clone()), &s(bad.clone()), &cfg(1)),
            Err(StatsError::NonFinite(3))
        );
        let year_of_day = vec![2000; 40];
        assert_eq!(
            paired_sharpe_by_year(&s(bad.clone()), &s(good.clone()), &year_of_day),
            Err(StatsError::NonFinite(3))
        );
        assert_eq!(
            regime_split(&s(bad), &[false; 40]),
            Err(StatsError::NonFinite(3))
        );
    }
}

#[test]
fn per_year_comparison_rejects_non_finite() {
    for v in BAD {
        assert_eq!(
            paired_by_year(&[1, 2], &[0.1, v], &[0.0, 0.0], true),
            Err(StatsError::NonFinite(1))
        );
        assert_eq!(
            paired_by_year(&[1, 2], &[0.1, 0.2], &[v, 0.0], false),
            Err(StatsError::NonFinite(0))
        );
    }
    assert_eq!(check_finite(&[0.0, -1.0, 1e300]), Ok(()));
}
