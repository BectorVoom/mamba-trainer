//! Synthetic panel: determinism, bar consistency and the planted signal.

use clm_data::features::col;
use clm_data::synthetic::generate;
use clm_data::{LabelConfig, PlantedSignal, SyntheticConfig, compute_features, compute_labels};

fn config(signal: Option<PlantedSignal>) -> SyntheticConfig {
    SyntheticConfig {
        seed: 3,
        num_days: 1500,
        num_sectors: 2,
        stocks_per_sector: 4,
        missing_prob: 0.0,
        signal,
    }
}

fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx * syy).sqrt()
}

#[test]
fn deterministic_and_consistent() {
    let a = generate(&config(None));
    let b = generate(&config(None));
    assert_eq!(a.panel, b.panel);
    assert_eq!(a.regime, b.regime);
    let c = generate(&SyntheticConfig {
        seed: 4,
        ..config(None)
    });
    assert_ne!(a.panel, c.panel);
    // Same draws with and without the signal: regimes are identical.
    let s = generate(&config(Some(PlantedSignal::default())));
    assert_eq!(a.regime, s.regime);
    let p = &a.panel;
    assert_eq!(p.num_securities(), 8);
    assert_eq!(p.sectors.len(), 2);
    assert_eq!(p.sector_of, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    for x in std::iter::once(&p.market)
        .chain(&p.sectors)
        .chain(&p.stocks)
    {
        for t in 0..p.num_days() {
            assert!(x.present[t]);
            let (o, h, l, c) = (x.open[t], x.high[t], x.low[t], x.close[t]);
            assert!(
                l > 0.0 && l <= o.min(c) && h >= o.max(c),
                "{} day {t}",
                x.symbol
            );
            assert!(x.volume[t] > 0.0);
        }
    }
    // Stocks carry the 0.9 adjustment factor: traded value uses the reported close.
    let x = &p.stocks[0];
    assert!(
        (x.dollar_volume[10] - x.close[10] / 0.9 * x.volume[10]).abs() < 1e-6 * x.dollar_volume[10]
    );
}

#[test]
fn missing_bars_follow_probability() {
    let sp = generate(&SyntheticConfig {
        missing_prob: 0.05,
        ..config(None)
    });
    let p = &sp.panel;
    let missing = p
        .stocks
        .iter()
        .map(|x| x.present.iter().filter(|v| !**v).count())
        .sum::<usize>();
    let total = p.num_securities() * p.num_days();
    let frac = missing as f64 / total as f64;
    assert!((0.04..0.06).contains(&frac), "missing fraction {frac}");
    assert!(p.market.present.iter().all(|v| *v));
}

/// relvol(t) reveals z[t], and z[t] shifts r_1(t): so corr(relvol, r_1) > 0 with the
/// signal and ~0 without.
#[test]
fn planted_signal_is_learnable_from_features() {
    let h1 = LabelConfig {
        horizon: 1,
        ..LabelConfig::default()
    };
    let mut seen = Vec::new();
    for signal in [None, Some(PlantedSignal::default())] {
        let sp = generate(&config(signal));
        let f = compute_features(&sp.panel);
        let l = compute_labels(&sp.panel, &h1).unwrap();
        let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
        for s in 0..f.num_securities {
            for t in 0..f.num_days {
                if f.is_valid(s, t) && l.exists(s, t) {
                    xs.push(f.get(s, t, col::RELVOL) as f64);
                    ys.push(l.fwd_ret_of(s, t));
                    zs.push(sp.regime[s * f.num_days + t] as f64);
                }
            }
        }
        let (c_feat, c_reg) = (corr(&xs, &ys), corr(&zs, &xs));
        println!(
            "signal {signal:?}: corr(relvol, r_1) = {c_feat:.3}, corr(z, relvol) = {c_reg:.3}, n = {}",
            xs.len()
        );
        if signal.is_some() {
            assert!(c_reg > 0.5, "relvol reveals the regime: {c_reg}");
        } else {
            assert!(c_reg.abs() < 0.05, "no loading, no relation: {c_reg}");
        }
        seen.push(c_feat);
    }
    assert!(seen[0].abs() < 0.05, "no signal: {}", seen[0]);
    assert!(seen[1] > 0.1, "signal: {}", seen[1]);
}
