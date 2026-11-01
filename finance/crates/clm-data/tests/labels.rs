//! Labels (C3): hand values, existence, strict class thresholds, causal direction.

mod common;

use clm_data::features::{SeriesCache, realized_vol};
use clm_data::{Action, LabelConfig, classify, compute_labels, label_audit};
use common::*;

fn cfg(h: usize, kappa: f64, m: f64) -> LabelConfig {
    LabelConfig {
        horizon: h,
        kappa,
        margin_mult: m,
        stop_frac: 0.02,
    }
}

#[test]
fn hand_values_h2() {
    let p = hand_panel();
    let c = cfg(2, 0.0005, 0.5);
    let l = compute_labels(&p, &c).unwrap();
    let (i, t) = (1usize, 100usize);
    // Adjusted opens/lows of the hand panel (common::ohlc).
    let open = |k: usize| stock_close(i, k - 1) * GAP;
    let low = |k: usize| open(k).min(stock_close(i, k)) / 1.01;
    let r_h = (open(t + 3) / open(t + 1)).ln();
    let mae = (low(t + 1).min(low(t + 2)) / open(t + 1)).ln();
    let sigma = 1.01f64.ln() * 252f64.sqrt() * (2.0f64 / 252.0).sqrt();
    let theta = 2.0 * 0.0005 + 0.5 * sigma;
    assert_close(l.fwd_ret_of(i, t), r_h, 1e-12, "r_h");
    assert_close(l.mae_of(i, t), mae, 1e-12, "mae");
    assert_close(l.sigma_h[i * N + t], sigma, 1e-12, "sigma_h");
    assert_close(l.theta[i * N + t], theta, 1e-12, "theta");
    assert_close(l.vol_ret_of(i, t), r_h / sigma, 1e-12, "vol_ret");
    // r_h = 2 ln g_1 = 0.004 here, theta ~ 0.0087: HOLD.
    assert_close(r_h, 0.004, 1e-12, "r_h closed form");
    assert_eq!(l.class_of(i, t), Some(Action::Hold));
    // mae = ln(L[t+1]/O[t+1]); t+1 odd: C > O so L = O/1.01.
    assert_close(mae, -(1.01f64.ln()), 1e-12, "mae closed form");
    assert!(!l.stop_of(i, t)); // -0.00995 > -0.02
    let l2 = compute_labels(
        &p,
        &LabelConfig {
            stop_frac: 0.005,
            ..c
        },
    )
    .unwrap();
    assert!(l2.stop_of(i, t));
}

#[test]
fn existence_edges() {
    let p = hand_panel();
    let h = 5;
    let l = compute_labels(&p, &cfg(h, 0.0, 0.0)).unwrap();
    // rv20(t) needs closes t-20..=t; the label needs bars t+1..=t+1+h.
    assert!(!l.exists(0, 19));
    assert!(l.exists(0, 20));
    assert!(l.exists(0, N - 2 - h));
    assert!(!l.exists(0, N - 1 - h));
    // A missing bar at day d kills labels with d in t+1..=t+1+h, and those whose rv20 reads d.
    let (m, sec, mut st) = hand_bars(N, 4);
    drop_row(&mut st[0], 200);
    let p = panel_from((m, sec, st));
    let l = compute_labels(&p, &cfg(h, 0.0, 0.0)).unwrap();
    assert!(l.exists(0, 200 - 2 - h));
    assert!((200 - 1 - h..=199).all(|t| !l.exists(0, t)));
    assert!((200..=220).all(|t| !l.exists(0, t)));
    assert!(l.exists(0, 221));
    assert!(l.class_of(0, 205).is_none() && l.fwd_ret_of(0, 205).is_nan());
}

#[test]
fn classes_use_strict_inequalities() {
    let th = 0.01;
    assert_eq!(classify(0.01, th), Action::Hold);
    assert_eq!(classify(-0.01, th), Action::Hold);
    assert_eq!(
        classify(f64::from_bits(0.01f64.to_bits() + 1), th),
        Action::Buy
    );
    assert_eq!(
        classify(-f64::from_bits(0.01f64.to_bits() + 1), th),
        Action::Sell
    );
    assert_eq!(classify(0.0, 0.0), Action::Hold);

    // End to end: with m = 0, theta = 2*kappa exactly; kappa = r_h/2 puts r_h on the boundary.
    let p = hand_panel();
    let (i, t, h) = (2usize, 150usize, 3usize);
    let base = compute_labels(&p, &cfg(h, 0.0, 0.0)).unwrap();
    let r = base.fwd_ret_of(i, t);
    assert!(r > 0.0);
    let at = |kappa: f64| {
        compute_labels(&p, &cfg(h, kappa, 0.0))
            .unwrap()
            .class_of(i, t)
    };
    assert_eq!(at(r / 2.0), Some(Action::Hold));
    assert_eq!(
        at(f64::from_bits((r / 2.0).to_bits() - 1)),
        Some(Action::Buy)
    );
    assert_eq!(
        at(f64::from_bits((r / 2.0).to_bits() + 1)),
        Some(Action::Hold)
    );
    assert_eq!(base.class_of(i, t), Some(Action::Buy));
}

#[test]
fn sell_when_price_falls() {
    let (m, sec, mut st) = hand_bars(N, 4);
    // Crash the adjusted (and raw) prices from day 103 on by 10%.
    let b = &mut st[0];
    for k in 103..N {
        for col in [
            &mut b.open,
            &mut b.high,
            &mut b.low,
            &mut b.close,
            &mut b.adj_close,
        ] {
            col[k] *= 0.9;
        }
    }
    let p = panel_from((m, sec, st));
    let l = compute_labels(&p, &cfg(2, 0.0005, 0.0)).unwrap();
    assert_eq!(l.class_of(0, 100), Some(Action::Sell)); // O[103]/O[101] ~ 0.9
    assert!(l.fwd_ret_of(0, 100) < -0.09);
}

#[test]
fn past_bars_do_not_change_forward_return() {
    let p = hand_panel();
    let c = cfg(5, 0.0005, 0.0);
    let t = 150;
    let base = compute_labels(&p, &c).unwrap();
    // Rewrite every bar <= t (keep them present): r_h and mae must not move.
    let mut q = p.clone();
    for x in q.stocks.iter_mut() {
        for k in 0..=t {
            let f = 1.0 + 0.03 * ((k * 7919) % 13) as f64;
            x.open[k] *= f;
            x.high[k] *= f * 1.01;
            x.low[k] *= f / 1.02;
            x.close[k] *= f;
        }
    }
    let changed = compute_labels(&q, &c).unwrap();
    for s in 0..4 {
        assert_eq!(
            base.fwd_ret_of(s, t).to_bits(),
            changed.fwd_ret_of(s, t).to_bits()
        );
        assert_eq!(base.mae_of(s, t).to_bits(), changed.mae_of(s, t).to_bits());
        // sigma_h reads rv20(t), so it may (and here does) change.
        let rv = realized_vol(&SeriesCache::new(&q.stocks[s]), t, 20).unwrap();
        assert_close(
            changed.sigma_h[s * N + t],
            rv * (5.0f64 / 252.0).sqrt(),
            1e-12,
            "sigma",
        );
    }
    // Bars after t+1+h do not matter either.
    let mut q = p.clone();
    for x in q.stocks.iter_mut() {
        for k in t + 7..N {
            x.open[k] *= 3.0;
            x.low[k] *= 0.1;
        }
    }
    let after = compute_labels(&q, &c).unwrap();
    for s in 0..4 {
        assert_eq!(
            base.fwd_ret_of(s, t).to_bits(),
            after.fwd_ret_of(s, t).to_bits()
        );
        assert_eq!(base.mae_of(s, t).to_bits(), after.mae_of(s, t).to_bits());
        assert_eq!(base.class_of(s, t), after.class_of(s, t));
    }
    // The generic label audit agrees on many days.
    let days: Vec<usize> = (0..N).step_by(7).collect();
    let mm = label_audit(&p, &c, &days, 3);
    assert!(mm.is_empty(), "{:?}", &mm[..mm.len().min(5)]);
}

#[test]
fn rejects_bad_config() {
    let p = hand_panel();
    assert!(compute_labels(&p, &cfg(0, 0.0, 0.0)).is_err());
    assert!(compute_labels(&p, &cfg(1, -0.1, 0.0)).is_err());
    assert!(compute_labels(&p, &cfg(1, 0.0, f64::NAN)).is_err());
}
