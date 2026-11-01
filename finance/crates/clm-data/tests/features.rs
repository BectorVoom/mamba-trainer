//! Feature set v1 against hand-computed values (C2) and validity edges.

mod common;

use clm_data::features::{col, zscores};
use clm_data::{FEATURE_NAMES, Panel, compute_features};
use common::*;

const LN101: f64 = 0.009_950_330_853_168_092; // ln 1.01

fn expected_row(i: usize, t: usize) -> Vec<(usize, f64)> {
    let lg = 0.001 * (i + 1) as f64;
    let a = 1.01f64.ln();
    assert!((a - LN101).abs() < 1e-15);
    let odd = t % 2 == 1;
    let sgn = if odd { 1.0 } else { -1.0 };
    let r1 = lg + sgn * a;
    let intraday = r1 - GAP.ln();
    let mkt_r1 = 0.001 + sgn * 1.02f64.ln();
    // hi52: max of H over t-251..=t straight from the hand OHLC.
    let closes: Vec<f64> = (0..=t).map(|k| stock_close(i, k)).collect();
    let (_, high, _) = ohlc(&closes);
    let hmax = high[t - 251..=t].iter().cloned().fold(f64::MIN, f64::max);
    let hi52 = (closes[t] / hmax).ln();
    if !odd {
        // Even t: the max is H[t] = O[t]*1.02 (closed form).
        assert!((hi52 - (intraday - 1.02f64.ln())).abs() < 1e-12);
    }
    let spike_mean = (19_000.0 + 1000.0 * (i + 1) as f64) / 20.0;
    let relvol = if t == SPIKE {
        (1000.0 * (i + 1) as f64 / spike_mean).ln()
    } else if t - 19 <= SPIKE && SPIKE < t {
        (1000.0 / spike_mean).ln()
    } else {
        0.0
    };
    // amihud: explicit sum with the reported (unadjusted) close.
    let mut sum = 0.0;
    for k in t - 19..=t {
        let r = (stock_close(i, k) / stock_close(i, k - 1)).ln().abs();
        sum += r / ((stock_close(i, k) / ADJ) * stock_volume(i, k));
    }
    let amihud = (1e-12 + sum / 20.0).ln();
    vec![
        (col::R1, r1),
        (col::R5, 5.0 * lg + sgn * a),
        (col::R20, 20.0 * lg),
        (col::R60, 60.0 * lg),
        (col::GAP, GAP.ln()),
        (col::INTRADAY, intraday),
        (col::RANGE, (1.02f64 * 1.01).ln() + intraday.abs()),
        (col::RV20, a * 252f64.sqrt()),
        (col::RV60, a * 252f64.sqrt()),
        (col::RELVOL, relvol),
        (col::MOM, 100.0 * lg),
        (col::HI52, hi52),
        (col::AMIHUD, amihud),
        (col::MKT_R1, mkt_r1),
        (col::MKT_R20, 0.02),
        (col::MKT_RV20, 1.02f64.ln() * 252f64.sqrt()),
        (col::SEC_R1, r1 - 0.0005),
        (col::SEC_R20, 20.0 * lg - 0.01),
        // r20_i = 0.02 (i+1): z = (i - 1.5) / sqrt(1.25)
        (col::XS_R20, (i as f64 - 1.5) / 1.25f64.sqrt()),
    ]
}

#[track_caller]
fn check(got: f32, want: f64, what: &str) {
    let tol = 3e-7 * want.abs().max(1e-3);
    assert!(
        (got as f64 - want).abs() <= tol,
        "{what}: got {got}, expected {want}"
    );
}

#[test]
fn every_feature_matches_hand_values() {
    let p = hand_panel();
    let f = compute_features(&p);
    assert_eq!((f.num_securities, f.num_days, f.num_features), (4, N, 20));
    assert_eq!(f.names, FEATURE_NAMES.map(String::from).to_vec());
    for t in [SPIKE, SPIKE + 1, 299] {
        for i in 0..4 {
            assert!(f.is_valid(i, t), "row ({i},{t}) valid");
            for (c, want) in expected_row(i, t) {
                check(
                    f.get(i, t, c),
                    want,
                    &format!("{} s{i} t{t}", FEATURE_NAMES[c]),
                );
            }
        }
    }
    // xs_relvol on the spike day: z-scores of the four relvols.
    let rv: Vec<f64> = (0..4)
        .map(|i| {
            let m = (19_000.0 + 1000.0 * (i + 1) as f64) / 20.0;
            (1000.0 * (i + 1) as f64 / m).ln()
        })
        .collect();
    let mean = rv.iter().sum::<f64>() / 4.0;
    let sd = (rv.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 4.0).sqrt();
    for (i, r) in rv.iter().enumerate() {
        check(
            f.get(i, SPIKE, col::XS_RELVOL),
            (r - mean) / sd,
            "xs_relvol",
        );
    }
}

#[test]
fn amihud_uses_reported_close_not_adjusted() {
    // With adj = 0.5 the reported close is twice the adjusted one, so amihud is ln 2
    // lower than for the same adjusted prices with adj = 1 (reported = adjusted).
    let p = hand_panel();
    let f = compute_features(&p);
    let (m, sec, mut st) = hand_bars(N, 4);
    for b in &mut st {
        b.open = b.open.iter().map(|x| x * ADJ).collect();
        b.high = b.high.iter().map(|x| x * ADJ).collect();
        b.low = b.low.iter().map(|x| x * ADJ).collect();
        b.close = b.adj_close.clone();
    }
    let g = compute_features(&panel_from((m, sec, st)));
    let d = f.get(0, SPIKE, col::AMIHUD) as f64 - g.get(0, SPIKE, col::AMIHUD) as f64;
    // (up to the 1e-12 inside the log: mean |r|/DV is ~5e-8 here)
    assert!((d + 2f64.ln()).abs() < 1e-4, "difference {d}");
    // Every ratio feature is unchanged by the adjustment factor.
    for c in (0..20).filter(|&c| c != col::AMIHUD) {
        assert!(
            (f.get(0, SPIKE, c) - g.get(0, SPIKE, c)).abs() < 1e-6,
            "{}",
            FEATURE_NAMES[c]
        );
    }
}

#[test]
fn first_valid_day_is_251() {
    let f = compute_features(&hand_panel());
    for s in 0..4 {
        assert!((0..251).all(|t| !f.is_valid(s, t)));
        assert!((251..N).all(|t| f.is_valid(s, t)));
        assert!(f.row(s, 250).iter().all(|x| x.is_nan()));
    }
}

/// 600-day panel with: stock 0 missing day 20, stock 1 zero volume on day 400, market bad
/// price on day 450, sector missing day 500, stocks 2 and 3 missing day 550.
fn edge_panel() -> (Panel, clm_data::LoadReport) {
    let (mut m, mut sec, mut st) = hand_bars(600, 4);
    drop_row(&mut st[0], 20);
    st[1].volume[400] = 0.0;
    m.close[450] = 0.0;
    drop_row(&mut sec[0], 500);
    drop_row(&mut st[2], 550);
    drop_row(&mut st[3], 550);
    Panel::from_bars(&m, &sec, &st, &[0; 4]).unwrap()
}

#[test]
fn validity_edges() {
    let (p, report) = edge_panel();
    assert_eq!(report.symbols[0].dropped_bad_price, 1);
    assert_eq!(report.symbols[0].missing_days, 1);
    assert_eq!(report.symbols[1].missing_days, 1);
    assert_eq!(report.symbols[2].missing_days, 1);
    assert_eq!(report.symbols[3].zero_volume, 1);
    assert!(!p.market.has(450));
    let f = compute_features(&p);
    let v = |s, t| f.is_valid(s, t);
    // Missing stock bar on day 20: rows 20..=271 invalid (hi52 reads 252 days).
    assert!((20..=271).all(|t| !v(0, t)));
    assert!(v(0, 272));
    // Zero volume on day 400: rows 400..=419 (relvol, amihud windows).
    assert!(v(1, 399) && v(1, 420));
    assert!((400..=419).all(|t| !v(1, t)));
    // Market missing on day 450: market r20/rv20 windows, rows 450..=470, every stock.
    for s in 0..2 {
        assert!(v(s, 449) && v(s, 471));
        assert!((450..=470).all(|t| !v(s, t)));
    }
    // Sector missing on day 500: sec_r1 reads t, t-1; sec_r20 reads t, t-20.
    for t in [500, 501, 520] {
        assert!(!v(0, t), "day {t}");
    }
    for t in [499, 502, 519, 521] {
        assert!(v(0, t), "day {t}");
    }
    // Day 550: only stocks 0 and 1 in the cross-section, so xs features are 0.
    assert!(v(0, 550) && v(1, 550) && !v(2, 550) && !v(3, 550));
    assert_eq!(f.get(0, 550, col::XS_R20), 0.0);
    assert_eq!(f.get(1, 550, col::XS_RELVOL), 0.0);
    // ... while on day 549 the four stocks give the hand z-scores.
    check(
        f.get(3, 549, col::XS_R20),
        1.5 / 1.25f64.sqrt(),
        "xs_r20 day 549",
    );
}

#[test]
fn zscore_rules() {
    assert_eq!(zscores(&[1.0, 2.0]), vec![0.0, 0.0]);
    assert_eq!(zscores(&[3.0, 3.0, 3.0]), vec![0.0; 3]);
    let z = zscores(&[1.0, 2.0, 3.0]);
    let sd = (2.0f64 / 3.0).sqrt();
    assert_close(z[0], -1.0 / sd, 1e-15, "z0");
    assert_close(z[2], 1.0 / sd, 1e-15, "z2");
}
