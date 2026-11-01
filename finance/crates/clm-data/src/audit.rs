//! Look-ahead audits.
//!
//! *Truncation audit*: for a day `t`, recompute the features on the panel cut after `t`
//! and compare row `t` of every stock with the full-panel result — validity flag and
//! every value bit for bit. A feature that reads any day `> t` (directly, through the
//! panel length, or through another stock's future) shows up as a mismatch.
//! *Perturbation audit*: same comparison, but the days after `t` are kept and scrambled
//! (prices and volumes rescaled at random, some bars removed); this also catches reads
//! of the future that do not depend on the panel length.
//!
//! Both take the feature function as an argument, so the tests can check that a
//! deliberately leaky function is caught. Labels have their own audit
//! ([`label_audit`]): they must read nothing after `t+1+h`, and `r_h`, `mae_h` nothing
//! at or before `t`.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::bars::{Panel, Series};
use crate::features::FeatureTensor;
use crate::labels::{LabelConfig, Labels, compute_labels};
use crate::synthetic::normal;

/// One disagreement between the full-panel result and the audited variant.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    /// Stock.
    pub security: usize,
    /// Audited day.
    pub day: usize,
    /// Column (feature or label field name); `"valid"` for a validity disagreement.
    pub field: String,
    /// Full-panel value (as `f64`; `1/0` for flags).
    pub full: f64,
    /// Value from the truncated/perturbed panel.
    pub audited: f64,
}

/// Compares row `t` of `full` and `variant` for every stock.
fn compare_rows(full: &FeatureTensor, variant: &FeatureTensor, t: usize, out: &mut Vec<Mismatch>) {
    assert_eq!(
        full.num_securities, variant.num_securities,
        "security count"
    );
    for s in 0..full.num_securities {
        let (vf, va) = (full.is_valid(s, t), variant.is_valid(s, t));
        if vf != va {
            out.push(Mismatch {
                security: s,
                day: t,
                field: "valid".into(),
                full: vf as u8 as f64,
                audited: va as u8 as f64,
            });
            continue;
        }
        if !vf {
            continue;
        }
        for (f, (a, b)) in full.row(s, t).iter().zip(variant.row(s, t)).enumerate() {
            if a.to_bits() != b.to_bits() {
                out.push(Mismatch {
                    security: s,
                    day: t,
                    field: full
                        .names
                        .get(f)
                        .cloned()
                        .unwrap_or_else(|| format!("f{f}")),
                    full: *a as f64,
                    audited: *b as f64,
                });
            }
        }
    }
}

/// Generic audit: for each `t` in `days`, compare row `t` of `features(panel)` with row
/// `t` of `features(variant(panel, t))`.
pub fn audit_with<F, V>(panel: &Panel, days: &[usize], features: F, variant: V) -> Vec<Mismatch>
where
    F: Fn(&Panel) -> FeatureTensor,
    V: Fn(&Panel, usize) -> Panel,
{
    let full = features(panel);
    let mut out = Vec::new();
    for &t in days {
        assert!(t < panel.num_days(), "audit day {t} past the calendar");
        let v = features(&variant(panel, t));
        compare_rows(&full, &v, t, &mut out);
    }
    out
}

/// Truncation audit (see the module docs). Returns every mismatch; empty = no look-ahead
/// detected on these days.
pub fn truncation_audit<F>(panel: &Panel, days: &[usize], features: F) -> Vec<Mismatch>
where
    F: Fn(&Panel) -> FeatureTensor,
{
    audit_with(panel, days, features, |p, t| p.truncated(t))
}

/// Perturbation audit (see the module docs): days after `t` are scrambled with `seed`.
pub fn perturbation_audit<F>(panel: &Panel, days: &[usize], seed: u64, features: F) -> Vec<Mismatch>
where
    F: Fn(&Panel) -> FeatureTensor,
{
    audit_with(panel, days, features, |p, t| {
        let n = p.num_days();
        scramble(
            p,
            t + 1..n,
            0.1,
            seed ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        )
    })
}

/// A copy of the panel whose bars on `days` are rescaled at random (each series and day:
/// prices by `exp(N(0, 0.05))`, volume by `exp(N(0, 0.5))`) and, with probability
/// `drop_prob`, removed.
pub fn scramble(panel: &Panel, days: std::ops::Range<usize>, drop_prob: f64, seed: u64) -> Panel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut p = panel.clone();
    let mut series: Vec<&mut Series> = vec![&mut p.market];
    series.extend(p.sectors.iter_mut());
    series.extend(p.stocks.iter_mut());
    for x in series {
        for t in days.clone() {
            if !x.present[t] {
                continue;
            }
            if rand::Rng::random::<f64>(&mut rng) < drop_prob {
                x.set_missing(t);
                continue;
            }
            for col in [&mut x.open, &mut x.high, &mut x.low, &mut x.close] {
                col[t] *= (0.05 * normal(&mut rng)).exp();
            }
            let fv = (0.5 * normal(&mut rng)).exp();
            x.volume[t] *= fv;
            x.dollar_volume[t] *= fv;
        }
    }
    p
}

fn label_fields(l: &Labels, s: usize, t: usize) -> [(&'static str, f64); 7] {
    let i = s * l.num_days + t;
    [
        ("class", l.class[i].map_or(-1.0, |a| a.id() as f64)),
        ("fwd_ret", l.fwd_ret[i]),
        ("sigma_h", l.sigma_h[i]),
        ("theta", l.theta[i]),
        ("vol_ret", l.vol_ret[i]),
        ("mae", l.mae[i]),
        ("stop", l.stop[i] as u8 as f64),
    ]
}

/// Label audit for days `days`:
/// 1. the panel cut after `t+1+h` gives the same label at `t` (every field, bitwise);
/// 2. with the bars of days `<= t` rescaled (none removed), `fwd_ret` and `mae` at `t`
///    and the label's existence are unchanged (`sigma_h`, `theta`, `class` and
///    `vol_ret` legitimately read `rv20(t)`).
pub fn label_audit(
    panel: &Panel,
    config: &LabelConfig,
    days: &[usize],
    seed: u64,
) -> Vec<Mismatch> {
    let full = compute_labels(panel, config).expect("label config");
    let h = config.horizon;
    let mut out = Vec::new();
    let mut push = |s, t, field: &str, a: f64, b: f64| {
        if a.to_bits() != b.to_bits() {
            out.push(Mismatch {
                security: s,
                day: t,
                field: field.into(),
                full: a,
                audited: b,
            });
        }
    };
    for &t in days {
        let last = (t + 1 + h).min(panel.num_days() - 1);
        let cut = compute_labels(&panel.truncated(last), config).expect("label config");
        let past = scramble(panel, 0..t + 1, 0.0, seed ^ t as u64);
        let past = compute_labels(&past, config).expect("label config");
        for s in 0..panel.num_securities() {
            let f = label_fields(&full, s, t);
            for ((name, a), (_, b)) in f.iter().zip(label_fields(&cut, s, t)) {
                push(s, t, name, *a, b);
            }
            let p = label_fields(&past, s, t);
            push(
                s,
                t,
                "exists",
                full.exists(s, t) as u8 as f64,
                past.exists(s, t) as u8 as f64,
            );
            push(s, t, "fwd_ret(past scrambled)", f[1].1, p[1].1);
            push(s, t, "mae(past scrambled)", f[5].1, p[5].1);
        }
    }
    out
}
