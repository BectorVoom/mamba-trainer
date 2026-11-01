//! Look-ahead audits on synthetic panels, and injected leaks that the audits must catch.

use clm_data::features::col;
use clm_data::synthetic::generate;
use clm_data::{
    FeatureTensor, LabelConfig, Panel, PlantedSignal, SyntheticConfig, compute_features,
    label_audit, perturbation_audit, truncation_audit,
};

fn synth(missing_prob: f64) -> Panel {
    generate(&SyntheticConfig {
        seed: 11,
        num_days: 700,
        num_sectors: 2,
        stocks_per_sector: 3,
        missing_prob,
        signal: Some(PlantedSignal::default()),
    })
    .panel
}

/// Every day from 0 to the end in steps, plus the edges around the 252-day lookback.
fn audit_days(n: usize) -> Vec<usize> {
    let mut d: Vec<usize> = (0..n).step_by(9).collect();
    d.extend([250, 251, 252, 253, n - 2, n - 1]);
    d.sort();
    d.dedup();
    d
}

#[test]
fn features_pass_truncation_audit_on_synthetic() {
    for missing in [0.0, 0.01] {
        let p = synth(missing);
        let f = compute_features(&p);
        assert!(f.num_valid() > 0);
        let mm = truncation_audit(&p, &audit_days(p.num_days()), compute_features);
        assert!(
            mm.is_empty(),
            "missing {missing}: {:?}",
            &mm[..mm.len().min(5)]
        );
    }
}

#[test]
fn features_pass_perturbation_audit_on_synthetic() {
    let p = synth(0.01);
    let mm = perturbation_audit(&p, &audit_days(p.num_days()), 5, compute_features);
    assert!(mm.is_empty(), "{:?}", &mm[..mm.len().min(5)]);
}

#[test]
fn labels_pass_label_audit_on_synthetic() {
    let p = synth(0.01);
    for h in [1, 5, 20] {
        let c = LabelConfig {
            horizon: h,
            ..LabelConfig::default()
        };
        let mm = label_audit(&p, &c, &audit_days(p.num_days()), 9);
        assert!(mm.is_empty(), "h={h}: {:?}", &mm[..mm.len().min(5)]);
    }
}

/// r1 replaced by the *next* day's return where it exists.
fn leaky_next_close(p: &Panel) -> FeatureTensor {
    let mut f = compute_features(p);
    for s in 0..p.num_securities() {
        let x = &p.stocks[s];
        for t in 0..p.num_days() {
            if !f.is_valid(s, t) {
                continue;
            }
            if x.has(t + 1) {
                f.row_mut(s, t)[col::R1] = (x.close[t + 1] / x.close[t]).ln() as f32;
            } else {
                f.invalidate(s, t);
            }
        }
    }
    f
}

/// A feature centred by the whole-sample mean of r20 (the classic full-sample leak).
fn leaky_full_sample_mean(p: &Panel) -> FeatureTensor {
    let mut f = compute_features(p);
    let (mut sum, mut n) = (0.0f64, 0usize);
    for s in 0..f.num_securities {
        for t in 0..f.num_days {
            if f.is_valid(s, t) {
                sum += f.get(s, t, col::R20) as f64;
                n += 1;
            }
        }
    }
    let mean = (sum / n.max(1) as f64) as f32;
    for s in 0..f.num_securities {
        for t in 0..f.num_days {
            if f.is_valid(s, t) {
                f.row_mut(s, t)[col::R20] -= mean;
            }
        }
    }
    f
}

#[test]
fn injected_next_close_leak_is_caught() {
    let p = synth(0.0);
    let days = audit_days(p.num_days());
    let mm = truncation_audit(&p, &days, leaky_next_close);
    assert!(!mm.is_empty());
    // Truncation makes day t the last day, so the leaky row becomes invalid.
    assert!(mm.iter().all(|m| m.field == "valid" && m.day >= 251));
    let audited_valid = days
        .iter()
        .filter(|&&d| d >= 251 && d < p.num_days() - 1)
        .count();
    assert_eq!(mm.len(), audited_valid * p.num_securities());
    // The perturbation audit sees the value change itself.
    let mm = perturbation_audit(&p, &days, 1, leaky_next_close);
    assert!(mm.iter().any(|m| m.field == "r1"));
}

#[test]
fn injected_full_sample_leak_is_caught_by_truncation() {
    let p = synth(0.0);
    let mm = truncation_audit(&p, &audit_days(p.num_days()), leaky_full_sample_mean);
    assert!(mm.iter().any(|m| m.field == "r20"));
    assert!(mm.iter().all(|m| m.field == "r20"));
}

#[test]
fn injected_future_cross_section_leak_is_caught() {
    // xs_r20 computed from the next day's cross-section.
    let leaky = |p: &Panel| {
        let mut f = compute_features(p);
        let g = f.clone();
        for s in 0..f.num_securities {
            for t in 0..f.num_days {
                if f.is_valid(s, t) {
                    let next = if t + 1 < g.num_days && g.is_valid(s, t + 1) {
                        g.get(s, t + 1, col::XS_R20)
                    } else {
                        0.0
                    };
                    f.row_mut(s, t)[col::XS_R20] = next;
                }
            }
        }
        f
    };
    let p = synth(0.0);
    let mm = truncation_audit(&p, &audit_days(p.num_days()), leaky);
    assert!(mm.iter().any(|m| m.field == "xs_r20"));
}
