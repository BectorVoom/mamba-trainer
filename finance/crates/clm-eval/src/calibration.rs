//! Post-hoc probability calibration, fitted on a fold's calibration segment only (C4
//! property 4) and saved per fold (serde).
//!
//! * **Temperature** — one `T > 0`, `p = softmax(z / T)`. The NLL is convex in
//!   `s = 1/T`, so it is unimodal in `ln T`: golden-section search over
//!   `ln T ∈ [ln 1e-3, ln 1e3]`, then a guarded Newton polish in `s`.
//! * **Vector scaling** (multiclass Platt) — `p = softmax(a ∘ z + b)` with per-class
//!   `a_k, b_k`. The objective `mean NLL + λ/2 (|a - 1|² + |b|²)`, `λ = 1e-4`, is
//!   strictly convex; it is minimised by **damped Newton** (exact 6×6 Hessian,
//!   Cholesky solve, Armijo backtracking) from the identity `a = 1, b = 0`, stopping
//!   when `max|grad| < 1e-10`, the objective stops decreasing, or after 100 steps. The
//!   small ridge fixes the softmax shift invariance of `b` and keeps the solution
//!   finite on separable data.
//! * **Isotonic** — per class, one-vs-rest isotonic regression (pool-adjacent-violators,
//!   tied scores pooled first) of `1[y = k]` on that class's softmax probability; the
//!   fitted monotone step function is kept. At apply time each class's value is floored
//!   at [`ISOTONIC_FLOOR`] and the row is renormalised to sum to 1.
//!
//! [`Calibrator::fit`] never returns a calibrator whose NLL on the fit set is worse
//! than the identity's: if one would be, it returns [`Calibrator::Identity`] instead
//! (by construction this can only happen for isotonic, whose per-class fits do not
//! control the renormalised multiclass NLL).

use clm_data::types::NUM_ACTIONS;
use serde::{Deserialize, Serialize};

use crate::metrics::{log_loss, softmax};

const K: usize = NUM_ACTIONS;

/// Floor of each isotonic class value before renormalisation (keeps every class
/// possible, so a test-time log loss cannot be infinite).
pub const ISOTONIC_FLOOR: f64 = 1e-6;
/// Ridge of vector scaling.
pub const VECTOR_RIDGE: f64 = 1e-4;
const T_MIN: f64 = 1e-3;
const T_MAX: f64 = 1e3;

/// Which calibrator to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// No calibration (`softmax(z)`).
    Identity,
    /// Temperature scaling.
    Temperature,
    /// Vector scaling (multiclass Platt).
    Vector,
    /// One-vs-rest isotonic regression.
    Isotonic,
}

/// A monotone non-decreasing step function: `value(x) = values[i]` for the last `i`
/// with `xs[i] <= x` (`values[0]` below `xs[0]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsotonicStep {
    /// Lower x edge of each block, strictly increasing.
    pub xs: Vec<f64>,
    /// Fitted value of each block, non-decreasing.
    pub values: Vec<f64>,
}

impl IsotonicStep {
    /// Fit by pool-adjacent-violators: the non-decreasing function minimising the
    /// squared error to `y` at the points `x` (equal `x` share one value).
    pub fn fit(x: &[f64], y: &[f64]) -> Self {
        assert_eq!(x.len(), y.len());
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        // blocks: (x_min, sum_y, weight)
        let mut blocks: Vec<(f64, f64, f64)> = Vec::new();
        let mut i = 0;
        while i < idx.len() {
            let x0 = x[idx[i]];
            let (mut s, mut w) = (0.0, 0.0);
            while i < idx.len() && x[idx[i]] == x0 {
                s += y[idx[i]];
                w += 1.0;
                i += 1;
            }
            blocks.push((x0, s, w));
            while blocks.len() >= 2 {
                let n = blocks.len();
                let (_, s1, w1) = blocks[n - 2];
                let (_, s2, w2) = blocks[n - 1];
                if s1 / w1 >= s2 / w2 {
                    blocks.pop();
                    let last = blocks.last_mut().expect("two blocks");
                    last.1 = s1 + s2;
                    last.2 = w1 + w2;
                } else {
                    break;
                }
            }
        }
        Self {
            xs: blocks.iter().map(|b| b.0).collect(),
            values: blocks.iter().map(|b| b.1 / b.2).collect(),
        }
    }

    /// Evaluate the step function (`x` itself if it was fitted on no data).
    pub fn eval(&self, x: f64) -> f64 {
        if self.xs.is_empty() {
            return x;
        }
        let i = self.xs.partition_point(|&v| v <= x);
        self.values[i.saturating_sub(1)]
    }
}

/// A fitted calibrator mapping logits to probabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Calibrator {
    /// `softmax(z)`.
    Identity,
    /// `softmax(z / t)`.
    Temperature {
        /// The temperature, `> 0`.
        t: f64,
    },
    /// `softmax(a ∘ z + b)`.
    Vector {
        /// Per-class scales.
        a: [f64; K],
        /// Per-class biases.
        b: [f64; K],
    },
    /// Per-class isotonic maps of the softmax probabilities, then renormalised.
    Isotonic {
        /// One step function per class.
        classes: Vec<IsotonicStep>,
    },
}

fn lse(l: &[f64; K]) -> f64 {
    let m = l.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    m + l.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

fn to_f64(z: &[f32; K]) -> [f64; K] {
    z.map(|v| v as f64)
}

fn softmax64(l: &[f64; K]) -> [f64; K] {
    let m = l.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let e = l.map(|v| (v - m).exp());
    let s: f64 = e.iter().sum();
    e.map(|v| v / s)
}

impl Calibrator {
    /// Fit a calibrator of `method` on `(logits, labels)` (the calibration segment).
    /// An empty fit set gives the identity (temperature 1). The result is never worse
    /// than the identity in NLL on the fit set (see the module docs).
    ///
    /// # Panics
    /// If the lengths differ, a label is `>= 3` or a logit is not finite.
    pub fn fit(method: CalibrationMethod, logits: &[[f32; K]], labels: &[u8]) -> Self {
        assert_eq!(logits.len(), labels.len());
        assert!(
            labels.iter().all(|&y| (y as usize) < K),
            "label out of range"
        );
        if let Some(i) = logits.iter().position(|z| z.iter().any(|v| !v.is_finite())) {
            panic!("non-finite logits at row {i}: {:?}", logits[i]);
        }
        if logits.is_empty() {
            return match method {
                CalibrationMethod::Temperature => Calibrator::Temperature { t: 1.0 },
                _ => Calibrator::Identity,
            };
        }
        let z: Vec<[f64; K]> = logits.iter().map(to_f64).collect();
        let cal = match method {
            CalibrationMethod::Identity => return Calibrator::Identity,
            CalibrationMethod::Temperature => Calibrator::Temperature {
                t: fit_temperature(&z, labels),
            },
            CalibrationMethod::Vector => {
                let (a, b) = fit_vector(&z, labels);
                Calibrator::Vector { a, b }
            }
            CalibrationMethod::Isotonic => Calibrator::Isotonic {
                classes: fit_isotonic(&z, labels),
            },
        };
        let id = Calibrator::Identity.apply(logits);
        let fitted = cal.apply(logits);
        if log_loss(&fitted, labels) > log_loss(&id, labels) {
            Calibrator::Identity
        } else {
            cal
        }
    }

    /// Calibrated probabilities of one row of logits, in `f64`.
    pub fn apply_row(&self, z: &[f32; K]) -> [f64; K] {
        match self {
            Calibrator::Identity => softmax(z),
            Calibrator::Temperature { t } => softmax64(&to_f64(z).map(|v| v / t)),
            Calibrator::Vector { a, b } => {
                let z = to_f64(z);
                softmax64(&std::array::from_fn(|k| a[k] * z[k] + b[k]))
            }
            Calibrator::Isotonic { classes } => {
                let p = softmax(z);
                let q: [f64; K] =
                    std::array::from_fn(|k| classes[k].eval(p[k]).max(ISOTONIC_FLOOR));
                let s: f64 = q.iter().sum();
                q.map(|v| v / s)
            }
        }
    }

    /// Calibrated probabilities of every row, as `f32`.
    pub fn apply(&self, logits: &[[f32; K]]) -> Vec<[f32; K]> {
        logits
            .iter()
            .map(|z| self.apply_row(z).map(|v| v as f32))
            .collect()
    }
}

/// Mean NLL of `softmax(s * z)` and its first two derivatives in `s`.
fn temp_nll(z: &[[f64; K]], y: &[u8], s: f64) -> (f64, f64, f64) {
    let (mut f, mut g, mut h) = (0.0, 0.0, 0.0);
    for (zi, &yi) in z.iter().zip(y) {
        let l = zi.map(|v| v * s);
        let p = softmax64(&l);
        f += lse(&l) - l[yi as usize];
        let m1: f64 = (0..K).map(|k| p[k] * zi[k]).sum();
        let m2: f64 = (0..K).map(|k| p[k] * zi[k] * zi[k]).sum();
        g += m1 - zi[yi as usize];
        h += m2 - m1 * m1;
    }
    let n = z.len() as f64;
    (f / n, g / n, h / n)
}

fn fit_temperature(z: &[[f64; K]], y: &[u8]) -> f64 {
    let f = |u: f64| temp_nll(z, y, (-u).exp()).0;
    let (mut lo, mut hi) = (T_MIN.ln(), T_MAX.ln());
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = hi - r * (hi - lo);
    let mut d = lo + r * (hi - lo);
    let (mut fc, mut fd) = (f(c), f(d));
    while hi - lo > 1e-7 {
        if fc <= fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - r * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + r * (hi - lo);
            fd = f(d);
        }
    }
    // Newton polish in s = 1/T (convex), each step accepted only if it lowers the NLL.
    let mut s = (-(lo + hi) / 2.0).exp();
    let (mut fs, mut g, mut h) = temp_nll(z, y, s);
    for _ in 0..20 {
        if h <= 0.0 || g.abs() < 1e-12 {
            break;
        }
        let s_new = s - g / h;
        if !(s_new > 1.0 / T_MAX && s_new < 1.0 / T_MIN) {
            break;
        }
        let (f2, g2, h2) = temp_nll(z, y, s_new);
        if f2 >= fs {
            break;
        }
        (s, fs, g, h) = (s_new, f2, g2, h2);
    }
    // T = 1 if the search did not beat it (flat objective, e.g. all-zero logits).
    if temp_nll(z, y, 1.0).0 <= fs {
        1.0
    } else {
        1.0 / s
    }
}

/// Objective, gradient and Hessian of vector scaling at `th = [a; b]`.
fn vector_obj(
    z: &[[f64; K]],
    y: &[u8],
    th: &[f64; 2 * K],
    want_h: bool,
) -> (f64, [f64; 6], [[f64; 6]; 6]) {
    let n = z.len() as f64;
    let mut f = 0.0;
    let mut g = [0.0; 2 * K];
    let mut h = [[0.0; 2 * K]; 2 * K];
    for (zi, &yi) in z.iter().zip(y) {
        let l: [f64; K] = std::array::from_fn(|k| th[k] * zi[k] + th[K + k]);
        let p = softmax64(&l);
        f += lse(&l) - l[yi as usize];
        for k in 0..K {
            let r = p[k] - if k == yi as usize { 1.0 } else { 0.0 };
            g[k] += r * zi[k];
            g[K + k] += r;
        }
        if want_h {
            // d l_k / d a_k = z_k, d l_k / d b_k = 1; H_l = diag(p) - p p^T.
            let d: [f64; 2 * K] = std::array::from_fn(|i| if i < K { zi[i] } else { 1.0 });
            for i in 0..2 * K {
                for j in 0..2 * K {
                    let (ki, kj) = (i % K, j % K);
                    let hl = if ki == kj { p[ki] } else { 0.0 } - p[ki] * p[kj];
                    h[i][j] += hl * d[i] * d[j];
                }
            }
        }
    }
    f /= n;
    for (i, gi) in g.iter_mut().enumerate() {
        *gi /= n;
        let c = if i < K { th[i] - 1.0 } else { th[i] };
        *gi += VECTOR_RIDGE * c;
        f += 0.5 * VECTOR_RIDGE * c * c;
    }
    for (i, row) in h.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v /= n;
        }
        row[i] += VECTOR_RIDGE;
    }
    (f, g, h)
}

/// Solve `H x = r` for symmetric positive definite `H` (Cholesky); `None` if not SPD.
#[allow(clippy::needless_range_loop)] // textbook index form is clearer here
fn cholesky_solve(h: &[[f64; 6]; 6], r: &[f64; 6]) -> Option<[f64; 6]> {
    let n = 6;
    let mut l = [[0.0; 6]; 6];
    for i in 0..n {
        for j in 0..=i {
            let mut s = h[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut x = [0.0; 6];
    for i in 0..n {
        let mut s = r[i];
        for k in 0..i {
            s -= l[i][k] * x[k];
        }
        x[i] = s / l[i][i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in i + 1..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

fn fit_vector(z: &[[f64; K]], y: &[u8]) -> ([f64; K], [f64; K]) {
    let mut th = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let (mut f, mut g, mut h) = vector_obj(z, y, &th, true);
    for _ in 0..100 {
        if g.iter().all(|v| v.abs() < 1e-10) {
            break;
        }
        let neg_g = g.map(|v| -v);
        // Newton direction; steepest descent if the Hessian is numerically not SPD.
        let dir = cholesky_solve(&h, &neg_g).unwrap_or(neg_g);
        let slope: f64 = dir.iter().zip(&g).map(|(d, g)| d * g).sum();
        if slope >= 0.0 {
            break;
        }
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let cand: [f64; 6] = std::array::from_fn(|i| th[i] + alpha * dir[i]);
            let fc = vector_obj(z, y, &cand, false).0;
            if fc.is_finite() && fc <= f + 1e-4 * alpha * slope {
                accepted = Some((cand, fc));
                break;
            }
            alpha *= 0.5;
        }
        let Some((cand, fc)) = accepted else { break };
        let done = f - fc <= 1e-15 * (1.0 + f.abs());
        th = cand;
        (f, g, h) = vector_obj(z, y, &th, true);
        if done {
            break;
        }
    }
    ([th[0], th[1], th[2]], [th[3], th[4], th[5]])
}

fn fit_isotonic(z: &[[f64; K]], y: &[u8]) -> Vec<IsotonicStep> {
    let p: Vec<[f64; K]> = z.iter().map(softmax64).collect();
    (0..K)
        .map(|k| {
            let x: Vec<f64> = p.iter().map(|r| r[k]).collect();
            let t: Vec<f64> = y
                .iter()
                .map(|&v| if v as usize == k { 1.0 } else { 0.0 })
                .collect();
            IsotonicStep::fit(&x, &t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pav_hand_example() {
        // x sorted: y = 1, 3, 2, 4 -> blocks 1, 2.5, 2.5, 4
        let s = IsotonicStep::fit(&[0.1, 0.2, 0.3, 0.4], &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(s.xs, vec![0.1, 0.2, 0.4]);
        assert_eq!(s.values, vec![1.0, 2.5, 4.0]);
        assert_eq!(s.eval(0.0), 1.0);
        assert_eq!(s.eval(0.25), 2.5);
        assert_eq!(s.eval(0.35), 2.5);
        assert_eq!(s.eval(0.9), 4.0);
    }

    #[test]
    fn pav_ties_pooled() {
        // two points at x=0.5 with y 0 and 1 -> one block of 0.5
        let s = IsotonicStep::fit(&[0.5, 0.5, 0.9], &[0.0, 1.0, 1.0]);
        assert_eq!(s.xs, vec![0.5, 0.9]);
        assert_eq!(s.values, vec![0.5, 1.0]);
        // ties that violate get merged with the next block
        let s = IsotonicStep::fit(&[0.1, 0.5, 0.5], &[1.0, 0.0, 1.0]);
        assert_eq!(s.values, vec![2.0 / 3.0]);
    }

    #[test]
    fn cholesky_identity() {
        let mut h = [[0.0; 6]; 6];
        for (i, row) in h.iter_mut().enumerate() {
            row[i] = 2.0;
        }
        let x = cholesky_solve(&h, &[2.0; 6]).unwrap();
        assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-15));
    }
}
