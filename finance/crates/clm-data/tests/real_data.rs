//! Real-data smoke and causality audits (skipped when `finance/data/bars` is absent).

mod common;

use std::time::Instant;

use clm_data::{
    FEATURE_NAMES, LabelConfig, Normalizer, Panel, compute_features, compute_labels, label_audit,
    perturbation_audit, truncation_audit, valid_samples,
};

fn load() -> Option<Panel> {
    let dir = common::real_dir()?;
    let t0 = Instant::now();
    let (p, report) = Panel::load_dir(&dir).unwrap();
    println!(
        "loaded {} stocks, {} sectors, {} days ({}..{}) in {:?}; dropped bad {} off-calendar {}",
        p.num_securities(),
        p.sectors.len(),
        p.num_days(),
        p.dates[0],
        p.dates[p.num_days() - 1],
        t0.elapsed(),
        report.total_dropped_bad(),
        report.total_off_calendar()
    );
    for r in report
        .symbols
        .iter()
        .filter(|r| r.missing_days > 0 || r.zero_volume > 0 || r.inconsistent_ohlc > 0)
    {
        println!("  {r:?}");
    }
    Some(p)
}

/// ~50 days: the lookback edge, then spread over the calendar, plus the last days.
fn audit_days(n: usize) -> Vec<usize> {
    let mut d = vec![0, 100, 249, 250, 251, 252, 253, 260, 300, 371, 372];
    d.extend((400..n).step_by((n - 400) / 36));
    d.extend([n - 3, n - 2, n - 1]);
    d.sort();
    d.dedup();
    d
}

#[test]
fn real_panel_smoke_counts() {
    let Some(p) = load() else {
        println!("no real data; skipped");
        return;
    };
    assert_eq!(p.num_securities(), 56);
    assert_eq!(p.sectors.len(), 9);
    let t0 = Instant::now();
    let f = compute_features(&p);
    println!(
        "features in {:?}: {} valid rows of {}",
        t0.elapsed(),
        f.num_valid(),
        f.valid.len()
    );
    let cfg = LabelConfig {
        horizon: 5,
        kappa: 0.0005,
        margin_mult: 0.0,
        stop_frac: 0.05,
    };
    let l = compute_labels(&p, &cfg).unwrap();
    let samples = valid_samples(&f, &l, 64);
    let mut counts = [0usize; 3];
    let mut stops = 0;
    for smp in &samples {
        counts[l.class_of(smp.security(), smp.day()).unwrap().id()] += 1;
        stops += l.stop_of(smp.security(), smp.day()) as usize;
    }
    let n = samples.len() as f64;
    println!(
        "h=5 kappa=5bp: labels {}, samples(W=64) {}, BUY {} ({:.1}%), SELL {} ({:.1}%), HOLD {} ({:.1}%), stop {:.1}%",
        l.num_existing(),
        samples.len(),
        counts[0],
        100.0 * counts[0] as f64 / n,
        counts[1],
        100.0 * counts[1] as f64 / n,
        counts[2],
        100.0 * counts[2] as f64 / n,
        100.0 * stops as f64 / n
    );
    let all = l.class_counts();
    println!(
        "all existing labels: BUY {} SELL {} HOLD {}",
        all[0], all[1], all[2]
    );
    for (m, name) in [(0.25, "m=0.25"), (0.5, "m=0.5")] {
        let lm = compute_labels(
            &p,
            &LabelConfig {
                margin_mult: m,
                ..cfg
            },
        )
        .unwrap();
        let c = lm.class_counts();
        let tot = (c[0] + c[1] + c[2]) as f64;
        println!(
            "  {name}: BUY {:.1}% SELL {:.1}% HOLD {:.1}%",
            100.0 * c[0] as f64 / tot,
            100.0 * c[1] as f64 / tot,
            100.0 * c[2] as f64 / tot
        );
    }
    // Every stock's rows become valid from day 251 (complete data).
    assert!((0..56).all(|s| !f.is_valid(s, 250) && f.is_valid(s, 251)));
    assert!(samples.len() > 300_000);
    // With m = 0 the HOLD band is only 2*kappa wide, so HOLD is rare (a few %).
    assert!(counts.iter().all(|&c| c > 0));
    assert!(counts[0] as f64 > 0.3 * n && counts[1] as f64 > 0.3 * n);
    assert_eq!(l.class_of(0, p.num_days() - 1), None);
    // Feature summary over the valid rows (train-like fit on everything).
    let norm = Normalizer::fit(&f, 0..p.num_days()).unwrap();
    for (k, name) in FEATURE_NAMES.iter().enumerate() {
        println!(
            "  {name:>10}: mean {:+.5} sd {:.5}",
            norm.mean[k], norm.sd[k]
        );
        assert!(norm.sd[k] > 1e-4, "{name} is degenerate");
    }
}

#[test]
fn real_panel_truncation_audit() {
    let Some(p) = load() else { return };
    let days = audit_days(p.num_days());
    println!("auditing {} days", days.len());
    let t0 = Instant::now();
    let mm = truncation_audit(&p, &days, compute_features);
    println!("truncation audit in {:?}", t0.elapsed());
    assert!(
        mm.is_empty(),
        "{} mismatches, first {:?}",
        mm.len(),
        &mm[..mm.len().min(5)]
    );
}

#[test]
fn real_panel_perturbation_and_label_audits() {
    let Some(p) = load() else { return };
    let days = audit_days(p.num_days());
    let t0 = Instant::now();
    let spread: Vec<usize> = days.iter().copied().step_by(3).collect();
    let mm = perturbation_audit(&p, &spread, 17, compute_features);
    assert!(
        mm.is_empty(),
        "{} mismatches, first {:?}",
        mm.len(),
        &mm[..mm.len().min(5)]
    );
    let mm = label_audit(&p, &LabelConfig::default(), &days, 5);
    assert!(
        mm.is_empty(),
        "{} mismatches, first {:?}",
        mm.len(),
        &mm[..mm.len().min(5)]
    );
    println!("perturbation + label audits in {:?}", t0.elapsed());
}
