//! Platt scaling and reliability for the reranker (plan item P7.9, machinery
//! only).
//!
//! Pure host `f64`: no tensors, no kernels, no device. The calibrated
//! probability is `p = sigmoid(a * logit + b)`, fitted by Newton/IRLS on the
//! `calibration` split only and judged on `report` (architecture §4.3).
//!
//! Fitting details (documented, deterministic: fixed initialization, a fixed
//! iteration cap, no randomness):
//!
//! * Target smoothing is Platt's own regulariser: a positive example gets
//!   `(N+ + 1) / (N+ + 2)` and a negative one `1 / (N- + 2)` instead of 1 and
//!   0, so a finite sample never demands an infinite logit.
//! * The 2×2 Newton system carries a small diagonal ridge ([`PLATT_RIDGE`])
//!   and accepts a step only when the penalised smoothed negative
//!   log-likelihood does not increase (halving otherwise, up to
//!   [`PLATT_MAX_HALVINGS`] halvings); convergence is the gradient norm below
//!   [`PLATT_GRAD_TOL`], and a non-converged fit is [`Error::Config`], never
//!   a silent wild pair.
//! * The logits are standardized with overflow-resistant arithmetic
//!   (`max |logit|` scaling, so values around `1e200`/`1e308` stay finite);
//!   the fitted `(a, b)` are mapped back to the caller's logit scale, and a
//!   non-finite fit is rejected with an error.
//! * Degenerate inputs stay finite: an empty input returns the neutral map
//!   `(a, b) = (0, 0)`; a single point or a single class still has smoothed
//!   targets and a ridged Hessian, so Newton converges to finite values;
//!   extreme logits are tamed by the standardization. `apply` maps `+inf` to
//!   1 and `-inf` to 0; only a NaN logit gives a NaN probability.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// Platt scaling
// ---------------------------------------------------------------------------

/// Version string stored with every calibration artifact.
pub const CALIBRATION_VERSION: &str = "ms2-calibration-v1";
/// Fixed Newton iteration cap of [`fit_platt`].
pub const PLATT_MAX_ITER: usize = 100;
/// Diagonal ridge of the 2×2 Newton system (the second regulariser after
/// Platt's target smoothing).
pub const PLATT_RIDGE: f64 = 1e-6;
/// Newton stops early when both parameter updates are below this (kept for
/// compatibility; convergence is judged by [`PLATT_GRAD_TOL`]).
pub const PLATT_TOL: f64 = 1e-10;
/// Convergence threshold of [`fit_platt`]: the mean (per-example) Euclidean
/// norm of the smoothed negative log-likelihood gradient (plus ridge) in the
/// standardized coordinates must fall below this. A fit that never gets there
/// is [`Error::Config`], never a silent wild parameter pair.
pub const PLATT_GRAD_TOL: f64 = 1e-6;
/// Floor for backtracking in [`fit_platt`]: halving stops after this many
/// halvings without an objective decrease, and the fit is reported as
/// non-converged ([`Error::Config`]).
pub const PLATT_MAX_HALVINGS: usize = 64;

/// Platt scaling parameters: `p = sigmoid(a * logit + b)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlattParams {
    /// Slope on the caller's logit scale.
    pub a: f64,
    /// Intercept on the caller's logit scale.
    pub b: f64,
    /// Whether the Newton fit converged (gradient norm below
    /// [`PLATT_GRAD_TOL`]). [`fit_platt`] returns [`Error::Config`] when the
    /// fit does not converge, so a successfully returned value always carries
    /// `converged = true`; callers that construct parameters by hand must set
    /// this explicitly.
    pub converged: bool,
}

/// Stable logistic sigmoid.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Fit Platt scaling `p = sigmoid(a * logit + b)` by damped Newton/IRLS with
/// [`PLATT_MAX_ITER`] iterations, objective-based backtracking (a step is
/// accepted only when the ridge-penalised smoothed negative log-likelihood
/// does not increase; the step is halved otherwise, up to
/// [`PLATT_MAX_HALVINGS`] halvings) and an explicit convergence test (the
/// gradient norm in standardized coordinates below [`PLATT_GRAD_TOL`]).
///
/// Standardization is overflow-resistant: with `S = max |logit|` the
/// computation runs in `u = logit / S`, so inputs around `1e200` or `1e308`
/// stay finite throughout. A fit whose parameters are non-finite, or that
/// never converges, is [`Error::Config`] — never a silent wild pair.
///
/// `logits` must be finite, `labels` hold `0.0`/`1.0` only, and both slices
/// share a length (anything else is [`Error::Config`]).
pub fn fit_platt(logits: &[f64], labels: &[f64]) -> Result<PlattParams> {
    if logits.len() != labels.len() {
        return Err(Error::config(format!(
            "fit_platt: {} logits for {} labels",
            logits.len(),
            labels.len()
        )));
    }
    for (i, &f) in logits.iter().enumerate() {
        if !f.is_finite() {
            return Err(Error::config(format!(
                "fit_platt: logit {i} is {f} (only finite logits are fitted)"
            )));
        }
    }
    for (i, &y) in labels.iter().enumerate() {
        if y != 0.0 && y != 1.0 {
            return Err(Error::config(format!(
                "fit_platt: label {i} is {y} (only 0.0/1.0 labels are fitted)"
            )));
        }
    }
    if logits.is_empty() {
        return Ok(PlattParams { a: 0.0, b: 0.0, converged: true });
    }
    let n = logits.len() as f64;
    let n_pos = labels.iter().filter(|&&y| y == 1.0).count() as f64;
    let n_neg = n - n_pos;
    // Platt's target smoothing: positives (N+ + 1)/(N+ + 2), negatives
    // 1/(N- + 2).
    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);
    // Overflow-resistant standardization. Summing raw logits (or squaring raw
    // deviations) overflows for values around 1e200/1e308, so scale by
    // S = max|logit| first: u = logit/S stays in [-1, 1], and the mean and
    // variance of u cannot overflow. std = S * std(u); the Newton features
    // (u - mean(u))/std(u) equal (f - mean)/std exactly in real arithmetic.
    let scale = logits.iter().fold(0.0f64, |m, &f| m.max(f.abs()));
    if !scale.is_finite() {
        return Err(Error::config(
            "fit_platt: logit scale is non-finite (only finite logits are fitted)".to_string(),
        ));
    }
    let (mean, std) = if scale == 0.0 {
        (0.0, 1.0)
    } else {
        let inv_n = 1.0 / n;
        let mut mean_u = 0.0;
        for &f in logits.iter() {
            mean_u += (f / scale) * inv_n;
        }
        let mut var_u = 0.0;
        for &f in logits.iter() {
            let d = f / scale - mean_u;
            var_u += d * d * inv_n;
        }
        if var_u <= 0.0 || !var_u.is_finite() {
            (mean_u * scale, 1.0)
        } else {
            (mean_u * scale, scale * var_u.sqrt())
        }
    };
    if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
        return Err(Error::config(
            "fit_platt: non-finite standardization (only finite logits are fitted)".to_string(),
        ));
    }
    // Smoothed targets and standardized features.
    let targets: Vec<f64> = labels
        .iter()
        .map(|&y| if y == 1.0 { t_pos } else { t_neg })
        .collect();
    let feats: Vec<f64> = logits.iter().map(|&f| (f - mean) / std).collect();
    if feats.iter().any(|f| !f.is_finite()) {
        return Err(Error::config(
            "fit_platt: non-finite standardized features (inputs too extreme to standardize)"
                .to_string(),
        ));
    }
    // Stable smoothed negative log-likelihood plus the ridge penalty (the
    // objective backtracking judges), with the log terms clamped away from
    // 0/1 so extreme linear predictors stay finite.
    let objective = |a: f64, b: f64| -> f64 {
        let mut s = 0.5 * PLATT_RIDGE * (a * a + b * b);
        for (&f, &t) in feats.iter().zip(targets.iter()) {
            let z = a * f + b;
            let p = sigmoid(z).clamp(1e-300, 1.0 - 1e-15);
            s += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
        }
        s
    };
    let grad_norm = |a: f64, b: f64| -> f64 {
        let mut ga = PLATT_RIDGE * a;
        let mut gb = PLATT_RIDGE * b;
        for (&f, &t) in feats.iter().zip(targets.iter()) {
            let r = t - sigmoid(a * f + b);
            ga += r * f;
            gb += r;
        }
        ga.hypot(gb) / n
    };
    // Prior log-odds with Laplace smoothing as the intercept start.
    let mut a = 0.0;
    let mut b = ((n_pos + 1.0) / (n_neg + 1.0)).ln();
    if grad_norm(a, b) < PLATT_GRAD_TOL {
        return Ok(PlattParams { a: a / std, b: b - a * mean / std, converged: true });
    }
    for _ in 0..PLATT_MAX_ITER {
        if grad_norm(a, b) < PLATT_GRAD_TOL {
            let (pa, pb) = (a / std, b - a * mean / std);
            if !pa.is_finite() || !pb.is_finite() {
                return Err(Error::config(
                    "fit_platt: non-finite fitted parameters (inputs too extreme)".to_string(),
                ));
            }
            return Ok(PlattParams { a: pa, b: pb, converged: true });
        }
        let mut ga = PLATT_RIDGE * a;
        let mut gb = PLATT_RIDGE * b;
        let mut haa = PLATT_RIDGE;
        let mut hab = 0.0;
        let mut hbb = PLATT_RIDGE;
        for (&f, &t) in feats.iter().zip(targets.iter()) {
            let p = sigmoid(a * f + b);
            let r = t - p;
            ga += r * f;
            gb += r;
            let w = p * (1.0 - p);
            haa += w * f * f;
            hab += w * f;
            hbb += w;
        }
        let det = haa * hbb - hab * hab;
        if !det.is_finite() || det <= 0.0 {
            return Err(Error::config(
                "fit_platt: Newton system is singular (no convergence)".to_string(),
            ));
        }
        let da = (ga * hbb - gb * hab) / det;
        let db = (gb * haa - ga * hab) / det;
        if !da.is_finite() || !db.is_finite() {
            return Err(Error::config(
                "fit_platt: Newton step is non-finite (no convergence)".to_string(),
            ));
        }
        // Objective-based backtracking: accept the step only when the
        // penalised smoothed NLL does not increase; halve otherwise.
        let obj = objective(a, b);
        if !obj.is_finite() {
            return Err(Error::config(
                "fit_platt: non-finite objective (no convergence)".to_string(),
            ));
        }
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..PLATT_MAX_HALVINGS {
            let trial = objective(a + step * da, b + step * db);
            if trial.is_finite() && trial <= obj {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            return Err(Error::config(
                "fit_platt: backtracking failed to decrease the objective (no convergence)"
                    .to_string(),
            ));
        }
        a += step * da;
        b += step * db;
        if !a.is_finite() || !b.is_finite() {
            return Err(Error::config(
                "fit_platt: non-finite iterate (no convergence)".to_string(),
            ));
        }
    }
    Err(Error::config(format!(
        "fit_platt: no convergence in {PLATT_MAX_ITER} iterations (gradient norm above {PLATT_GRAD_TOL})"
    )))
}

/// Calibrated probabilities for `logits` under `params` (stable sigmoid).
pub fn apply_platt(params: &PlattParams, logits: &[f64]) -> Vec<f64> {
    logits.iter().map(|&f| sigmoid(params.a * f + params.b)).collect()
}

// ---------------------------------------------------------------------------
// Reliability
// ---------------------------------------------------------------------------

/// Binning rule for the reliability table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binning {
    /// `nbins` bins of equal width over `[0, 1]`.
    EqualWidth,
    /// `nbins` groups of (near-)equal size after sorting by confidence.
    EqualMass,
}

/// One reliability bin: edges, count, mean confidence, empirical frequency.
///
/// Only non-empty bins are reported (an empty bin has no confidence or
/// frequency, so it is omitted rather than filled with NaNs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// Lower edge (inclusive): `0.0` for the first equal-width bin, the
    /// group's minimum probability for equal-mass bins.
    pub lo: f64,
    /// Upper edge (exclusive, except the last equal-width bin which holds
    /// `1.0`): `1.0` for the last equal-width bin, the group's maximum
    /// probability for equal-mass bins.
    pub hi: f64,
    /// Examples in the bin (always `>= 1`).
    pub count: usize,
    /// Mean predicted probability in the bin.
    pub mean_conf: f64,
    /// Fraction of positives in the bin.
    pub freq: f64,
}

/// Size stratum of an atom count (contracts §10): 3–5, 6–9, 10–16 atoms;
/// `None` outside every stratum. Out-of-stratum examples are excluded from
/// the per-stratum metrics, never merged into a neighbour.
pub fn stratum_of(atoms: usize) -> Option<usize> {
    match atoms {
        3..=5 => Some(0),
        6..=9 => Some(1),
        10..=16 => Some(2),
        _ => None,
    }
}

/// Reliability table of `probs` against `labels` (`0.0`/`1.0`).
///
/// Length mismatches, `nbins == 0`, non-finite probabilities or labels
/// outside `0.0`/`1.0` are [`Error::Config`]. Probabilities marginally
/// outside `[0, 1]` are clamped into the edge bins (documented; fitted
/// probabilities cannot leave the range, so this only bites on
/// caller-supplied values).
pub fn reliability_table(
    probs: &[f64],
    labels: &[f64],
    nbins: usize,
    binning: Binning,
) -> Result<Vec<ReliabilityBin>> {
    check_reliability_inputs(probs, labels, nbins)?;
    if probs.is_empty() {
        return Ok(Vec::new());
    }
    match binning {
        Binning::EqualWidth => Ok(equal_width(probs, labels, nbins)),
        Binning::EqualMass => Ok(equal_mass(probs, labels, nbins)),
    }
}

/// Expected calibration error: `sum_bins count/N * |freq − mean_conf|`.
/// Same input rules as [`reliability_table`]; an empty input has ECE `0.0`.
pub fn expected_calibration_error(
    probs: &[f64],
    labels: &[f64],
    nbins: usize,
    binning: Binning,
) -> Result<f64> {
    let bins = reliability_table(probs, labels, nbins, binning)?;
    if probs.is_empty() {
        return Ok(0.0);
    }
    let n = probs.len() as f64;
    Ok(bins
        .iter()
        .map(|b| b.count as f64 / n * (b.freq - b.mean_conf).abs())
        .sum())
}

/// Brier score: the mean squared error between probabilities and labels.
/// Same input rules as [`reliability_table`] except `nbins` (none); an empty
/// input scores `0.0`.
pub fn brier_score(probs: &[f64], labels: &[f64]) -> Result<f64> {
    check_reliability_inputs(probs, labels, 1)?;
    if probs.is_empty() {
        return Ok(0.0);
    }
    Ok(probs
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| (p - y) * (p - y))
        .sum::<f64>()
        / probs.len() as f64)
}

/// Reliability tables per size stratum (3–5, 6–9, 10–16 atoms).
/// `atoms` holds one atom count per example (a length mismatch is
/// [`Error::Config`]); examples outside every stratum are excluded. An empty
/// stratum yields an empty table.
pub fn reliability_by_stratum(
    probs: &[f64],
    labels: &[f64],
    atoms: &[usize],
    nbins: usize,
    binning: Binning,
) -> Result<[Vec<ReliabilityBin>; 3]> {
    let split = split_strata(probs, labels, atoms, nbins)?;
    Ok([
        reliability_table(&split[0].0, &split[0].1, nbins, binning)?,
        reliability_table(&split[1].0, &split[1].1, nbins, binning)?,
        reliability_table(&split[2].0, &split[2].1, nbins, binning)?,
    ])
}

/// Expected calibration error per size stratum. An empty stratum has ECE
/// `0.0` (no miscalibration over no data).
pub fn ece_by_stratum(
    probs: &[f64],
    labels: &[f64],
    atoms: &[usize],
    nbins: usize,
    binning: Binning,
) -> Result<[f64; 3]> {
    let split = split_strata(probs, labels, atoms, nbins)?;
    Ok([
        expected_calibration_error(&split[0].0, &split[0].1, nbins, binning)?,
        expected_calibration_error(&split[1].0, &split[1].1, nbins, binning)?,
        expected_calibration_error(&split[2].0, &split[2].1, nbins, binning)?,
    ])
}

/// Brier score per size stratum. An empty stratum scores `0.0`.
pub fn brier_by_stratum(
    probs: &[f64],
    labels: &[f64],
    atoms: &[usize],
) -> Result<[f64; 3]> {
    let split = split_strata(probs, labels, atoms, 1)?;
    Ok([
        brier_score(&split[0].0, &split[0].1)?,
        brier_score(&split[1].0, &split[1].1)?,
        brier_score(&split[2].0, &split[2].1)?,
    ])
}

/// Shared input checks of the reliability metrics.
fn check_reliability_inputs(probs: &[f64], labels: &[f64], nbins: usize) -> Result<()> {
    if probs.len() != labels.len() {
        return Err(Error::config(format!(
            "reliability: {} probabilities for {} labels",
            probs.len(),
            labels.len()
        )));
    }
    if nbins == 0 {
        return Err(Error::config(
            "reliability: nbins is 0 (at least one bin is required)".to_string(),
        ));
    }
    for (i, &p) in probs.iter().enumerate() {
        if !p.is_finite() {
            return Err(Error::config(format!(
                "reliability: probability {i} is {p} (only finite probabilities are scored)"
            )));
        }
    }
    for (i, &y) in labels.iter().enumerate() {
        if y != 0.0 && y != 1.0 {
            return Err(Error::config(format!(
                "reliability: label {i} is {y} (only 0.0/1.0 labels are scored)"
            )));
        }
    }
    Ok(())
}

/// Clamp into `[0, 1]` for binning (documented edge-bin rule). Inputs are
/// finite (checked by [`check_reliability_inputs`]), so `clamp` cannot panic.
fn clamp01(p: f64) -> f64 {
    p.clamp(0.0, 1.0)
}

/// Equal-width bins over `[0, 1]`; empty bins omitted.
fn equal_width(probs: &[f64], labels: &[f64], nbins: usize) -> Vec<ReliabilityBin> {
    let width = 1.0 / nbins as f64;
    let mut counts = vec![0usize; nbins];
    let mut conf = vec![0.0f64; nbins];
    let mut pos = vec![0usize; nbins];
    for (&p, &y) in probs.iter().zip(labels.iter()) {
        let c = clamp01(p);
        let mut idx = (c / width).floor() as usize;
        if idx >= nbins {
            idx = nbins - 1;
        }
        counts[idx] += 1;
        conf[idx] += c;
        if y == 1.0 {
            pos[idx] += 1;
        }
    }
    let mut out = Vec::new();
    for b in 0..nbins {
        if counts[b] == 0 {
            continue;
        }
        out.push(ReliabilityBin {
            lo: b as f64 * width,
            hi: if b + 1 == nbins { 1.0 } else { (b + 1) as f64 * width },
            count: counts[b],
            mean_conf: conf[b] / counts[b] as f64,
            freq: pos[b] as f64 / counts[b] as f64,
        });
    }
    out
}

/// Equal-mass groups after sorting by probability; empty groups omitted
/// (fewer examples than bins). Ties keep their input order (stable sort), so
/// the grouping is deterministic.
fn equal_mass(probs: &[f64], labels: &[f64], nbins: usize) -> Vec<ReliabilityBin> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        clamp01(probs[a])
            .total_cmp(&clamp01(probs[b]))
            .then_with(|| a.cmp(&b))
    });
    let n = probs.len();
    // Group sizes differ by at most one: the first `rem` groups take one extra.
    let base = n / nbins;
    let rem = n % nbins;
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for b in 0..nbins {
        let size = base + usize::from(b < rem);
        if size == 0 {
            continue;
        }
        let group = &order[cursor..cursor + size];
        cursor += size;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut conf = 0.0;
        let mut positives = 0usize;
        for &i in group {
            let c = clamp01(probs[i]);
            if c < lo {
                lo = c;
            }
            if c > hi {
                hi = c;
            }
            conf += c;
            if labels[i] == 1.0 {
                positives += 1;
            }
        }
        out.push(ReliabilityBin {
            lo,
            hi,
            count: size,
            mean_conf: conf / size as f64,
            freq: positives as f64 / size as f64,
        });
    }
    out
}

/// Split probabilities/labels into the three size strata.
fn split_strata(
    probs: &[f64],
    labels: &[f64],
    atoms: &[usize],
    nbins: usize,
) -> Result<[(Vec<f64>, Vec<f64>); 3]> {
    check_reliability_inputs(probs, labels, nbins)?;
    if atoms.len() != probs.len() {
        return Err(Error::config(format!(
            "reliability: {} atom counts for {} examples",
            atoms.len(),
            probs.len()
        )));
    }
    let mut split: [(Vec<f64>, Vec<f64>); 3] = [
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
    ];
    for ((&p, &y), &a) in probs.iter().zip(labels.iter()).zip(atoms.iter()) {
        if let Some(s) = stratum_of(a) {
            split[s].0.push(p);
            split[s].1.push(y);
        }
    }
    Ok(split)
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// The version keys the plan lists (domain, K, F, precision, ranking, search
/// policy), as strings supplied by the caller. [`CalibrationArtifact`] refuses
/// to apply under different keys.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigKeys {
    /// Chemistry domain version.
    pub domain: String,
    /// Trajectories per spectrum (`K`).
    pub k: String,
    /// Retained formulas per spectrum (`F`).
    pub f: String,
    /// Neural precision.
    pub precision: String,
    /// Ranking rule (raw or reranker version).
    pub ranking: String,
    /// Formula search policy.
    pub search_policy: String,
}

/// A fitted calibration: the Platt parameters, the fit split's name and
/// counts, and the version keys it was fitted under.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationArtifact {
    /// Artifact version ([`CALIBRATION_VERSION`]).
    pub version: String,
    /// Fitted slope.
    pub a: f64,
    /// Fitted intercept.
    pub b: f64,
    /// Name of the split the fit ran on (architecture §4.3: `calibration`).
    pub fit_split: String,
    /// Examples in the fit split.
    pub fit_count: usize,
    /// Positives in the fit split.
    pub fit_positives: usize,
    /// Version keys the fit ran under.
    pub config: ConfigKeys,
}

impl CalibrationArtifact {
    /// Build an artifact from fitted parameters, the fit split's name and
    /// counts, and the caller's version keys.
    pub fn new(
        params: &PlattParams,
        fit_split: &str,
        fit_count: usize,
        fit_positives: usize,
        config: ConfigKeys,
    ) -> Self {
        Self {
            version: CALIBRATION_VERSION.to_string(),
            a: params.a,
            b: params.b,
            fit_split: fit_split.to_string(),
            fit_count,
            fit_positives,
            config,
        }
    }

    /// Refuse an artifact under different configuration keys
    /// ([`Error::Config`] naming the first mismatched key).
    pub fn validate_matches(&self, keys: &ConfigKeys) -> Result<()> {
        for (name, mine, theirs) in [
            ("domain", &self.config.domain, &keys.domain),
            ("k", &self.config.k, &keys.k),
            ("f", &self.config.f, &keys.f),
            ("precision", &self.config.precision, &keys.precision),
            ("ranking", &self.config.ranking, &keys.ranking),
            ("search_policy", &self.config.search_policy, &keys.search_policy),
        ] {
            if mine != theirs {
                return Err(Error::config(format!(
                    "CalibrationArtifact::validate_matches: {name} mismatch (artifact {mine}, caller {theirs})"
                )));
            }
        }
        Ok(())
    }

    /// Calibrated probabilities for `logits` after [`validate_matches`]:
    /// refuses a different configuration instead of silently rescaling it.
    ///
    /// [`validate_matches`]: CalibrationArtifact::validate_matches
    pub fn apply(&self, keys: &ConfigKeys, logits: &[f64]) -> Result<Vec<f64>> {
        self.validate_matches(keys)?;
        Ok(apply_platt(
            &PlattParams { a: self.a, b: self.b, converged: true },
            logits,
        ))
    }

    /// Save the artifact as JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(&self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Load an artifact saved by [`CalibrationArtifact::save`]. A version
    /// mismatch is [`Error::Config`].
    ///
    /// [`CalibrationArtifact::save`]: CalibrationArtifact::save
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let artifact: CalibrationArtifact = serde_json::from_str(&text)?;
        if artifact.version != CALIBRATION_VERSION {
            return Err(Error::config(format!(
                "CalibrationArtifact::load: unknown version {} (expected {CALIBRATION_VERSION})",
                artifact.version
            )));
        }
        Ok(artifact)
    }
}
