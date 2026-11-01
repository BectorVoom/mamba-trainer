//! Backtest accounting (C5) against a hand-computed example and the proved identities.
// Conditions are written as in docs/CONTRACTS.md (`t + 1 + h <= N - 1`), on purpose.
#![allow(clippy::int_plus_one, clippy::needless_range_loop)]

use clm_data::types::Action::{self, Buy, Hold, Sell};
use clm_eval::backtest::*;
use clm_eval::folds::Segment;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const NAN: f64 = f64::NAN;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

fn assert_vec(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(close(*g, *w), "{got:?} vs {want:?}");
    }
}

/// N = 8, test segment [1, 7): decision days 1..=6; t = 6 has t + 2 = 8 > 7.
fn example() -> (Vec<Vec<f64>>, Vec<Vec<Option<Action>>>) {
    let opens = vec![
        vec![10.0, 10.0, 11.0, 12.0, NAN, 12.0, 13.0, 14.0],
        vec![20.0, 20.0, 19.0, 19.0, 18.0, 20.0, 21.0, 22.0],
    ];
    let dec = vec![
        vec![
            Some(Buy),
            Some(Hold),
            Some(Sell),
            Some(Buy),
            None,
            Some(Buy),
        ],
        vec![
            None,
            Some(Buy),
            Some(Buy),
            Some(Hold),
            Some(Sell),
            Some(Sell),
        ],
    ];
    (opens, dec)
}

#[test]
fn hand_computed_two_securities() {
    let (opens, dec) = example();
    let k = 0.001;
    let r = run_backtest(&opens, Segment::new(1, 7), 8, &dec, k).unwrap();
    let a = &r.securities[0];
    assert_eq!(a.position, vec![1, 1, 0, 1, 1, 1]);
    assert_eq!(a.trade, vec![1, 0, 1, 1, 0, 0]);
    assert_eq!(a.traded, vec![true, true, true, true, true, false]);
    // t=1: O[3]/O[2]-1 = 12/11-1; t=2: long but O[4] (= O[t+2]) missing -> 0;
    // t=3: SELL executed at the missing O[4] -> flat, cost still charged;
    // t=4: 13/12-1; t=5: 14/13-1; t=6: t+2 = 8 > N-1 = 7 -> untraded
    assert_vec(
        &a.gross,
        &[
            12.0 / 11.0 - 1.0,
            0.0,
            0.0,
            13.0 / 12.0 - 1.0,
            14.0 / 13.0 - 1.0,
            0.0,
        ],
    );
    assert_vec(&a.cost, &[k, 0.0, k, k, 0.0, 0.0]);
    let b = &r.securities[1];
    assert_eq!(b.position, vec![0, 1, 1, 1, 0, 0]);
    assert_eq!(b.trade, vec![0, 1, 0, 0, 1, 0]);
    assert_vec(
        &b.gross,
        &[
            0.0,
            18.0 / 19.0 - 1.0,
            20.0 / 18.0 - 1.0,
            21.0 / 20.0 - 1.0,
            0.0,
            0.0,
        ],
    );
    assert_vec(&b.cost, &[0.0, k, 0.0, 0.0, k, 0.0]);

    let d = &r.daily;
    assert_eq!(d.day, vec![2, 3, 4, 5, 6]);
    let want_ret = [
        (12.0 / 11.0 - 1.0 - k) / 2.0,
        (18.0 / 19.0 - 1.0 - k) / 2.0,
        (-k + 20.0 / 18.0 - 1.0) / 2.0,
        (13.0 / 12.0 - 1.0 - k + 0.05) / 2.0,
        (14.0 / 13.0 - 1.0 - k) / 2.0,
    ];
    assert_vec(&d.ret, &want_ret);
    assert_vec(&d.turnover, &[0.5; 5]);
    assert_vec(&d.gross_exposure, &[0.5, 1.0, 0.5, 1.0, 0.5]);
}

#[test]
fn untraded_tail_keeps_position_and_ignores_action() {
    // N = 4: decisions at t = 0, 1 are traded (t + 2 <= 3); t = 2, 3 are not.
    let opens = vec![vec![1.0, 1.0, 1.1, 1.2]];
    let dec = vec![vec![Some(Buy), Some(Hold), Some(Sell), Some(Sell)]];
    let r = run_backtest(&opens, Segment::new(0, 4), 4, &dec, 0.01).unwrap();
    let s = &r.securities[0];
    assert_eq!(s.position, vec![1, 1, 1, 1]);
    assert_eq!(s.trade, vec![1, 0, 0, 0]);
    assert_eq!(s.traded, vec![true, true, false, false]);
    assert_vec(&s.cost, &[0.01, 0.0, 0.0, 0.0]);
    assert_eq!(r.daily.day, vec![1, 2]);
}

#[test]
fn identities_on_random_backtests() {
    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..300 {
        let n = rng.random_range(3..40);
        let ns = rng.random_range(1..5);
        let ts = rng.random_range(0..n);
        let te = rng.random_range(ts..=n);
        let opens: Vec<Vec<f64>> = (0..ns)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        if rng.random_bool(0.1) {
                            NAN
                        } else {
                            rng.random_range(5.0..15.0)
                        }
                    })
                    .collect()
            })
            .collect();
        let dec: Vec<Vec<Option<Action>>> = (0..ns)
            .map(|_| {
                (ts..te)
                    .map(|_| match rng.random_range(0..4) {
                        3 => None,
                        i => Action::from_id(i),
                    })
                    .collect()
            })
            .collect();
        let kappa = rng.random_range(0.0..0.01);
        let r = run_backtest(&opens, Segment::new(ts, te), n, &dec, kappa).unwrap();
        let mut sum_trade_all = 0.0;
        for (s, l) in r.securities.iter().enumerate() {
            let g: f64 = l.gross.iter().sum();
            let nt: f64 = l.net.iter().sum();
            let tr: f64 = l.trade.iter().map(|&x| x as f64).sum();
            sum_trade_all += tr;
            assert!((nt - (g - kappa * tr)).abs() < 1e-12);
            let mut prev = 0u8;
            for i in 0..l.position.len() {
                assert!(l.position[i] <= 1);
                assert_eq!(l.trade[i], l.position[i].abs_diff(prev));
                let a = dec[s][i].unwrap_or(Hold);
                if a == Hold || !l.traded[i] {
                    assert_eq!(l.trade[i], 0, "HOLD never trades");
                    assert_eq!(l.position[i], prev);
                }
                if !l.traded[i] {
                    assert_eq!((l.gross[i], l.cost[i]), (0.0, 0.0));
                    assert!(ts + i + 2 > n - 1);
                }
                prev = l.position[i];
            }
        }
        let d = &r.daily;
        let ns_f = ns as f64;
        let turnover_sum: f64 = d.turnover.iter().sum();
        assert!((turnover_sum * ns_f - sum_trade_all).abs() < 1e-9);
        assert!(d.gross_exposure.iter().all(|&g| (0.0..=1.0).contains(&g)));
        assert!(d.day.windows(2).all(|w| w[0] < w[1]));
        assert!(d.day.iter().all(|&x| x + 1 <= n - 1 && x >= ts + 1));
    }
}

#[test]
fn metrics_hand_computed() {
    let daily = DailySeries {
        day: vec![1, 2, 3, 4],
        ret: vec![0.01, -0.02, 0.03, 0.0],
        turnover: vec![1.0, 0.0, 0.5, 0.5],
        gross_exposure: vec![1.0; 4],
    };
    let m = backtest_metrics(&daily);
    let sd = (1.3e-3f64 / 4.0).sqrt();
    assert!((m.ann_return - 252.0 * 0.005).abs() < 1e-12);
    assert!((m.sharpe - 252f64.sqrt() * 0.005 / sd).abs() < 1e-9);
    assert!((m.sortino - 252f64.sqrt() * 0.005 / 0.01).abs() < 1e-9);
    assert!((m.max_drawdown - (1.0 - 1.01 * 0.98 / 1.01)).abs() < 1e-12);
    assert!((m.profit_factor.unwrap() - 2.0).abs() < 1e-12);
    assert!((m.mean_turnover - 0.5).abs() < 1e-15);
    assert!((m.total_net_return - (1.01 * 0.98 * 1.03 - 1.0)).abs() < 1e-12);
    assert_eq!(m.n_days, 4);
    // Degenerate cases.
    assert_eq!(profit_factor(&[0.01, 0.02]), None);
    assert_eq!(sharpe(&[0.001; 10]), 0.0);
    assert_eq!(sortino(&[0.01, 0.02]), 0.0);
    assert_eq!(max_drawdown(&[0.1, 0.1]), 0.0);
    let e = backtest_metrics(&DailySeries::default());
    assert_eq!((e.n_days, e.sharpe, e.total_net_return), (0, 0.0, 0.0));
}

#[test]
fn buy_and_hold_pays_entry_once() {
    let (opens, _) = example();
    let k = 0.002;
    let r = buy_and_hold(&opens, Segment::new(1, 7), 8, k).unwrap();
    for l in &r.securities {
        assert_eq!(l.position, vec![1; 6]);
        assert_eq!(l.trade, vec![1, 0, 0, 0, 0, 0]);
        assert!((l.cost.iter().sum::<f64>() - k).abs() < 1e-15);
    }
    // day 2: (12/11-1 - k + 19/20... ) security B gross at t=1 = O[3]/O[2]-1 = 19/19-1 = 0
    assert!((r.daily.ret[0] - (12.0 / 11.0 - 1.0 - k + 0.0 - k) / 2.0).abs() < 1e-12);
    assert!(r.daily.gross_exposure.iter().all(|&g| g == 1.0));
}

#[test]
fn concat_requires_increasing_days() {
    let (opens, dec) = example();
    let a = run_backtest(
        &opens,
        Segment::new(1, 3),
        8,
        &[dec[0][..2].to_vec(), dec[1][..2].to_vec()],
        0.0,
    )
    .unwrap()
    .daily;
    let b = run_backtest(
        &opens,
        Segment::new(3, 6),
        8,
        &[dec[0][2..5].to_vec(), dec[1][2..5].to_vec()],
        0.0,
    )
    .unwrap()
    .daily;
    let c = DailySeries::concat(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(c.day, vec![2, 3, 4, 5, 6]);
    assert_eq!(c.ret.len(), 5);
    assert!(matches!(
        DailySeries::concat(&[b, a]),
        Err(BacktestError::DaysNotIncreasing(_))
    ));
}

#[test]
fn bad_inputs_are_errors() {
    let (opens, dec) = example();
    assert!(matches!(
        run_backtest(&[], Segment::new(0, 0), 8, &[], 0.0),
        Err(BacktestError::NoSecurities)
    ));
    assert!(matches!(
        run_backtest(&opens, Segment::new(1, 9), 8, &dec, 0.0),
        Err(BacktestError::BadSegment { .. })
    ));
    assert!(matches!(
        run_backtest(&opens, Segment::new(1, 6), 8, &dec, 0.0),
        Err(BacktestError::BadLength {
            what: "decisions",
            ..
        })
    ));
    assert!(matches!(
        run_backtest(&opens, Segment::new(1, 7), 8, &dec, -0.1),
        Err(BacktestError::BadKappa(_))
    ));
    assert!(matches!(
        run_backtest(&opens, Segment::new(1, 7), 8, &dec[..1], 0.0),
        Err(BacktestError::SecurityCount { .. })
    ));
}

#[test]
fn policy_and_render() {
    let acts = [Hold, Buy, Hold, Sell, Sell, Buy, Buy, Hold];
    let (p, t) = policy(&acts);
    assert_eq!(p, vec![0, 1, 1, 0, 0, 1, 1, 1]);
    assert_eq!(t, vec![0, 1, 0, 1, 0, 1, 0, 0]);
    assert_eq!(render_backtest_command(&acts), "backtest 8 2 0 2 1 1 0 0 2");
    assert_eq!(
        render_backtest(&acts),
        [
            "pos 0 1 1 0 0 1 1 1".to_string(),
            "trade 0 1 0 1 0 1 0 0".to_string()
        ]
    );
    assert_eq!(
        render_backtest(&[]),
        ["pos".to_string(), "trade".to_string()]
    );
}
