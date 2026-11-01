//! Classification, ranking and regression metrics.
//!
//! All inputs are `f32` (as stored in out-of-fold records); every sum is accumulated
//! in `f64`. Class ids follow [`clm_data::types::Action`]: BUY 0, SELL 1, HOLD 2.
//!
//! Every function taking probability rows and labels **panics** on invalid input
//! instead of returning a misleading number: lengths that differ, a label `>= 3`, or a
//! row that is non-finite, outside `[0, 1]` or does not sum to 1 within
//! [`ROW_SUM_TOL`] ([`check_prob_row`]). (A NaN row would otherwise pass as BUY in the
//! argmax and silently corrupt accuracy, Brier, ECE and AUC.)

use clm_data::types::NUM_ACTIONS;
use serde::{Deserialize, Serialize};

/// Probability floor of the log loss.
pub const LOG_LOSS_EPS: f64 = 1e-12;
/// Number of equal-width confidence bins of the ECE.
pub const ECE_BINS: usize = 15;
/// Days with fewer securities than this are skipped by the rank IC.
pub const IC_MIN_SECURITIES: usize = 5;

/// Softmax of one row of logits (max-shifted, in `f64`).
pub fn softmax(logits: &[f32; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    let m = logits
        .iter()
        .fold(f64::NEG_INFINITY, |m, &z| m.max(z as f64));
    let mut e = [0.0f64; NUM_ACTIONS];
    let mut s = 0.0;
    for k in 0..NUM_ACTIONS {
        e[k] = (logits[k] as f64 - m).exp();
        s += e[k];
    }
    e.map(|v| v / s)
}

/// Softmax of every row, as `f32`.
pub fn softmax_rows(logits: &[[f32; NUM_ACTIONS]]) -> Vec<[f32; NUM_ACTIONS]> {
    logits
        .iter()
        .map(|z| softmax(z).map(|p| p as f32))
        .collect()
}

/// Index of the largest probability (the first one on ties).
pub fn argmax(p: &[f32; NUM_ACTIONS]) -> usize {
    let mut best = 0;
    for k in 1..NUM_ACTIONS {
        if p[k] > p[best] {
            best = k;
        }
    }
    best
}

/// Tolerance on `|sum_k p_k - 1|` of a probability row.
pub const ROW_SUM_TOL: f64 = 1e-4;

/// Check one probability row: every entry finite and in `[0, 1]`, and the row sums to
/// 1 within [`ROW_SUM_TOL`]. Returns a description of the first problem.
pub fn check_prob_row(p: &[f32; NUM_ACTIONS]) -> Result<(), String> {
    if let Some(k) = p.iter().position(|v| !v.is_finite()) {
        return Err(format!("probability {k} is not finite ({})", p[k]));
    }
    if let Some(k) = p.iter().position(|&v| !(0.0..=1.0).contains(&v)) {
        return Err(format!("probability {k} = {} is outside [0, 1]", p[k]));
    }
    let s: f64 = p.iter().map(|&v| v as f64).sum();
    if (s - 1.0).abs() > ROW_SUM_TOL {
        return Err(format!("row sums to {s}, not 1"));
    }
    Ok(())
}

/// Validate metric inputs; panics with the offending row on any problem.
fn check_inputs(probs: &[[f32; NUM_ACTIONS]], labels: &[u8]) {
    assert_eq!(
        probs.len(),
        labels.len(),
        "probs and labels differ in length"
    );
    if let Some(i) = labels.iter().position(|&y| (y as usize) >= NUM_ACTIONS) {
        panic!("label out of range at row {i}: {}", labels[i]);
    }
    for (i, p) in probs.iter().enumerate() {
        if let Err(e) = check_prob_row(p) {
            panic!("invalid probabilities at row {i} {p:?}: {e}");
        }
    }
}

/// Fraction of rows whose argmax equals the label (0 for no rows).
pub fn accuracy(probs: &[[f32; NUM_ACTIONS]], labels: &[u8]) -> f64 {
    check_inputs(probs, labels);
    if labels.is_empty() {
        return 0.0;
    }
    let hits = probs
        .iter()
        .zip(labels)
        .filter(|(p, y)| argmax(p) == **y as usize)
        .count();
    hits as f64 / labels.len() as f64
}

/// Mean per-class recall over the classes that occur in `labels` (0 for no rows).
pub fn balanced_accuracy(probs: &[[f32; NUM_ACTIONS]], labels: &[u8]) -> f64 {
    check_inputs(probs, labels);
    let mut hit = [0usize; NUM_ACTIONS];
    let mut cnt = [0usize; NUM_ACTIONS];
    for (p, &y) in probs.iter().zip(labels) {
        cnt[y as usize] += 1;
        if argmax(p) == y as usize {
            hit[y as usize] += 1;
        }
    }
    let present: Vec<f64> = (0..NUM_ACTIONS)
        .filter(|&k| cnt[k] > 0)
        .map(|k| hit[k] as f64 / cnt[k] as f64)
        .collect();
    if present.is_empty() {
        0.0
    } else {
        present.iter().sum::<f64>() / present.len() as f64
    }
}

/// Mean multiclass log loss `-ln max(p_y, 1e-12)` (0 for no rows). Rows are used as
/// given (not renormalised).
pub fn log_loss(probs: &[[f32; NUM_ACTIONS]], labels: &[u8]) -> f64 {
    check_inputs(probs, labels);
    if labels.is_empty() {
        return 0.0;
    }
    let s: f64 = probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| {
            let py = p[y as usize] as f64;
            -py.max(LOG_LOSS_EPS).ln()
        })
        .sum();
    s / labels.len() as f64
}

/// Mean multiclass Brier score `sum_k (p_k - 1[y = k])^2` (range 0..2; 0 for no rows).
pub fn brier(probs: &[[f32; NUM_ACTIONS]], labels: &[u8]) -> f64 {
    check_inputs(probs, labels);
    if labels.is_empty() {
        return 0.0;
    }
    let s: f64 = probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| {
            (0..NUM_ACTIONS)
                .map(|k| {
                    let o = if k == y as usize { 1.0 } else { 0.0 };
                    (p[k] as f64 - o).powi(2)
                })
                .sum::<f64>()
        })
        .sum();
    s / labels.len() as f64
}

/// Average ranks (1-based) of `x`, ties sharing the mean of their ranks.
///
/// # Panics
/// If `x` holds a NaN.
pub fn average_ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).expect("NaN in average_ranks"));
    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        // positions i..j (0-based) -> ranks i+1..=j, mean (i+1+j)/2
        let r = (i + 1 + j) as f64 / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = r;
        }
        i = j;
    }
    ranks
}

/// Binary ROC-AUC by the rank-sum (Mann–Whitney) formula with tied scores averaged.
/// `None` when there is no positive or no negative.
///
/// # Panics
/// If the lengths differ or a score is NaN.
pub fn roc_auc(scores: &[f64], positive: &[bool]) -> Option<f64> {
    assert_eq!(scores.len(), positive.len());
    let n_pos = positive.iter().filter(|&&b| b).count();
    let n_neg = positive.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let ranks = average_ranks(scores);
    let rank_sum: f64 = ranks
        .iter()
        .zip(positive)
        .filter(|(_, p)| **p)
        .map(|(r, _)| r)
        .sum();
    let u = rank_sum - (n_pos as f64) * (n_pos as f64 + 1.0) / 2.0;
    Some(u / (n_pos as f64 * n_neg as f64))
}

/// One-vs-rest ROC-AUC of each class (score = that class's probability) and their
/// macro mean over the classes where it is defined.
pub fn roc_auc_ovr(
    probs: &[[f32; NUM_ACTIONS]],
    labels: &[u8],
) -> ([Option<f64>; NUM_ACTIONS], Option<f64>) {
    check_inputs(probs, labels);
    let mut per = [None; NUM_ACTIONS];
    for (k, slot) in per.iter_mut().enumerate() {
        let s: Vec<f64> = probs.iter().map(|p| p[k] as f64).collect();
        let pos: Vec<bool> = labels.iter().map(|&y| y as usize == k).collect();
        *slot = roc_auc(&s, &pos);
    }
    let def: Vec<f64> = per.iter().flatten().copied().collect();
    let macro_auc = if def.is_empty() {
        None
    } else {
        Some(def.iter().sum::<f64>() / def.len() as f64)
    };
    (per, macro_auc)
}

/// One bin of a reliability diagram.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// Lower edge of the confidence bin.
    pub lo: f64,
    /// Upper edge.
    pub hi: f64,
    /// Rows whose top-label confidence falls in the bin.
    pub count: usize,
    /// Mean top-label confidence of those rows (0 if empty).
    pub mean_confidence: f64,
    /// Fraction of those rows whose top label is correct (0 if empty).
    pub accuracy: f64,
}

/// Top-label expected calibration error with [`ECE_BINS`] equal-width bins on `[0, 1]`
/// and the bins. Bin `b` holds confidences in `[b/15, (b+1)/15)`, the last bin also
/// holds `1.0`. ECE = `sum_b (count_b / n) * |accuracy_b - mean_confidence_b|`.
pub fn ece_top_label(probs: &[[f32; NUM_ACTIONS]], labels: &[u8]) -> (f64, Vec<ReliabilityBin>) {
    check_inputs(probs, labels);
    let nb = ECE_BINS;
    let mut cnt = vec![0usize; nb];
    let mut conf = vec![0.0f64; nb];
    let mut acc = vec![0.0f64; nb];
    for (p, &y) in probs.iter().zip(labels) {
        let k = argmax(p);
        let c = (p[k] as f64).clamp(0.0, 1.0);
        let b = ((c * nb as f64).floor() as usize).min(nb - 1);
        cnt[b] += 1;
        conf[b] += c;
        if k == y as usize {
            acc[b] += 1.0;
        }
    }
    let n = labels.len();
    let mut ece = 0.0;
    let bins = (0..nb)
        .map(|b| {
            let (mc, ma) = if cnt[b] > 0 {
                (conf[b] / cnt[b] as f64, acc[b] / cnt[b] as f64)
            } else {
                (0.0, 0.0)
            };
            if n > 0 {
                ece += cnt[b] as f64 / n as f64 * (ma - mc).abs();
            }
            ReliabilityBin {
                lo: b as f64 / nb as f64,
                hi: (b + 1) as f64 / nb as f64,
                count: cnt[b],
                mean_confidence: mc,
                accuracy: ma,
            }
        })
        .collect();
    (ece, bins)
}

/// Every classification metric of a set of predictions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Number of rows.
    pub n: usize,
    /// Argmax accuracy.
    pub accuracy: f64,
    /// Mean recall over present classes.
    pub balanced_accuracy: f64,
    /// Multiclass log loss (p clipped at 1e-12).
    pub log_loss: f64,
    /// Multiclass Brier score.
    pub brier: f64,
    /// One-vs-rest ROC-AUC per class (`None` if undefined).
    pub auc: [Option<f64>; NUM_ACTIONS],
    /// Mean of the defined per-class AUCs.
    pub macro_auc: Option<f64>,
    /// Top-label ECE (15 bins).
    pub ece: f64,
    /// The reliability bins of the ECE.
    pub reliability: Vec<ReliabilityBin>,
}

/// All classification metrics from probabilities.
///
/// # Panics
/// If the lengths differ, a label is `>= 3`, or a probability row is invalid (see
/// [`check_prob_row`]: non-finite, outside `[0, 1]`, or not summing to 1).
pub fn classification_metrics(
    probs: &[[f32; NUM_ACTIONS]],
    labels: &[u8],
) -> ClassificationMetrics {
    let (auc, macro_auc) = roc_auc_ovr(probs, labels);
    let (ece, reliability) = ece_top_label(probs, labels);
    ClassificationMetrics {
        n: labels.len(),
        accuracy: accuracy(probs, labels),
        balanced_accuracy: balanced_accuracy(probs, labels),
        log_loss: log_loss(probs, labels),
        brier: brier(probs, labels),
        auc,
        macro_auc,
        ece,
        reliability,
    }
}

/// [`classification_metrics`] of `softmax(logits)`.
pub fn classification_metrics_from_logits(
    logits: &[[f32; NUM_ACTIONS]],
    labels: &[u8],
) -> ClassificationMetrics {
    classification_metrics(&softmax_rows(logits), labels)
}

/// Pearson correlation; `None` if either side has zero variance or `n < 2`.
fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// Spearman rank correlation (Pearson of average ranks); `None` if undefined.
pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    assert_eq!(x.len(), y.len());
    pearson(&average_ranks(x), &average_ranks(y))
}

/// Summary of daily cross-sectional rank ICs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IcSummary {
    /// Mean daily IC.
    pub mean: f64,
    /// Sample standard deviation (`n - 1`) of the daily ICs (0 if fewer than 2 days).
    pub sd: f64,
    /// `mean / sd * sqrt(n_days)` (0 if `sd == 0`).
    pub t_stat: f64,
    /// Days that produced an IC.
    pub n_days: usize,
    /// Days skipped (fewer than 5 securities, or a constant score/return).
    pub n_skipped: usize,
    /// The IC of each used day, `(day, ic)` in day order.
    pub daily: Vec<(u32, f64)>,
}

/// Mean daily Spearman IC between `score` and realised forward return `ret` across
/// securities, per decision day `day`. Pairs with a non-finite score or return are
/// ignored; days with fewer than [`IC_MIN_SECURITIES`] remaining pairs, or where either
/// side is constant (IC undefined), are skipped.
pub fn rank_ic(day: &[u32], score: &[f64], ret: &[f64]) -> IcSummary {
    assert!(day.len() == score.len() && day.len() == ret.len());
    let mut by_day: std::collections::BTreeMap<u32, (Vec<f64>, Vec<f64>)> =
        std::collections::BTreeMap::new();
    let mut all_days = std::collections::BTreeSet::new();
    for i in 0..day.len() {
        all_days.insert(day[i]);
        if score[i].is_finite() && ret[i].is_finite() {
            let e = by_day.entry(day[i]).or_default();
            e.0.push(score[i]);
            e.1.push(ret[i]);
        }
    }
    let mut daily = Vec::new();
    for (d, (s, r)) in &by_day {
        if s.len() < IC_MIN_SECURITIES {
            continue;
        }
        if let Some(ic) = spearman(s, r) {
            daily.push((*d, ic));
        }
    }
    let n = daily.len();
    let mean = if n > 0 {
        daily.iter().map(|x| x.1).sum::<f64>() / n as f64
    } else {
        0.0
    };
    let sd = if n > 1 {
        (daily.iter().map(|x| (x.1 - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    let t_stat = if sd > 0.0 {
        mean / sd * (n as f64).sqrt()
    } else {
        0.0
    };
    IcSummary {
        mean,
        sd,
        t_stat,
        n_days: n,
        n_skipped: all_days.len() - n,
        daily,
    }
}

/// Error metrics of return predictions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionMetrics {
    /// Pairs used (both finite).
    pub n: usize,
    /// Root mean squared error (0 for no pairs).
    pub rmse: f64,
    /// Mean absolute error (0 for no pairs).
    pub mae: f64,
}

/// RMSE and MAE over the pairs where both values are finite.
pub fn regression_metrics(pred: &[f64], target: &[f64]) -> RegressionMetrics {
    assert_eq!(pred.len(), target.len());
    let (mut n, mut se, mut ae) = (0usize, 0.0, 0.0);
    for (p, y) in pred.iter().zip(target) {
        if p.is_finite() && y.is_finite() {
            n += 1;
            se += (p - y).powi(2);
            ae += (p - y).abs();
        }
    }
    if n == 0 {
        return RegressionMetrics {
            n,
            rmse: 0.0,
            mae: 0.0,
        };
    }
    RegressionMetrics {
        n,
        rmse: (se / n as f64).sqrt(),
        mae: ae / n as f64,
    }
}
