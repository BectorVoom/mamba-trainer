//! Labels (CONTRACTS C3): forward returns from the next open, target-exposure classes,
//! vol-scaled return, max adverse excursion and stop flag.
//!
//! For decision day `t` with horizon `h`, prices are read only on days
//! `t+1 ..= t+1+h`, except `sigma_h(t) = rv20(t) * sqrt(h/252)` which reads the stock's
//! closes `t-20 ..= t` (known at decision time; it only sets the class threshold and the
//! vol scale). A label exists iff every bar `t+1 ..= t+1+h` exists **and** `rv20(t)` is
//! available (bars `t-20 ..= t`).

use serde::{Deserialize, Serialize};

use crate::bars::Panel;
use crate::error::{DataError, Result};
use crate::features::{SeriesCache, TRADING_DAYS, realized_vol};
use crate::types::Action;

/// Label parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LabelConfig {
    /// Horizon `h >= 1` in days.
    pub horizon: usize,
    /// One-way cost `kappa` (fraction, e.g. `0.0005` = 5 bp).
    pub kappa: f64,
    /// Margin multiplier `m >= 0` on `sigma_h`.
    pub margin_mult: f64,
    /// Stop threshold: `stop = mae_h < -stop_frac`.
    pub stop_frac: f64,
}

impl Default for LabelConfig {
    /// `h = 5`, `kappa = 5 bp`, `m = 0`, `stop_frac = 0.05`.
    fn default() -> Self {
        Self {
            horizon: 5,
            kappa: 0.0005,
            margin_mult: 0.0,
            stop_frac: 0.05,
        }
    }
}

impl LabelConfig {
    /// Checks `h >= 1` and that `kappa`, `m`, `stop_frac` are finite and `>= 0`.
    pub fn validate(&self) -> Result<()> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        if self.horizon == 0 || !ok(self.kappa) || !ok(self.margin_mult) || !ok(self.stop_frac) {
            return Err(DataError::Invalid(format!("bad label config {self:?}")));
        }
        Ok(())
    }

    /// `theta = 2*kappa + m*sigma_h`.
    pub fn threshold(&self, sigma_h: f64) -> f64 {
        2.0 * self.kappa + self.margin_mult * sigma_h
    }
}

/// The class of a forward return against threshold `theta` (strict inequalities).
pub fn classify(r_h: f64, theta: f64) -> Action {
    if r_h > theta {
        Action::Buy
    } else if r_h < -theta {
        Action::Sell
    } else {
        Action::Hold
    }
}

/// Labels of every (stock, day), `[s][t]` row-major. Fields of a missing label are NaN
/// (`class = None`, `stop = false`). Equality is bitwise on the floats.
#[derive(Debug, Clone)]
pub struct Labels {
    /// The config they were made with.
    pub config: LabelConfig,
    /// Stocks.
    pub num_securities: usize,
    /// Calendar days.
    pub num_days: usize,
    /// Class, `None` if the label does not exist.
    pub class: Vec<Option<Action>>,
    /// `r_h = log(O[t+1+h] / O[t+1])`.
    pub fwd_ret: Vec<f64>,
    /// `sigma_h = rv20(t) * sqrt(h / 252)`.
    pub sigma_h: Vec<f64>,
    /// `theta = 2*kappa + m*sigma_h`.
    pub theta: Vec<f64>,
    /// `r_h / max(sigma_h, 1e-4)`.
    pub vol_ret: Vec<f64>,
    /// `mae_h = log(min(L[t+1 ..= t+h]) / O[t+1])`.
    pub mae: Vec<f64>,
    /// `mae_h < -stop_frac`.
    pub stop: Vec<bool>,
}

impl PartialEq for Labels {
    fn eq(&self, o: &Self) -> bool {
        let bits = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        self.config == o.config
            && (self.num_securities, self.num_days) == (o.num_securities, o.num_days)
            && self.class == o.class
            && self.stop == o.stop
            && bits(&self.fwd_ret, &o.fwd_ret)
            && bits(&self.sigma_h, &o.sigma_h)
            && bits(&self.theta, &o.theta)
            && bits(&self.vol_ret, &o.vol_ret)
            && bits(&self.mae, &o.mae)
    }
}

impl Labels {
    #[inline]
    fn idx(&self, s: usize, t: usize) -> usize {
        debug_assert!(s < self.num_securities && t < self.num_days);
        s * self.num_days + t
    }

    /// Whether the label of `(s, t)` exists.
    #[inline]
    pub fn exists(&self, s: usize, t: usize) -> bool {
        self.class[self.idx(s, t)].is_some()
    }

    /// The class of `(s, t)`.
    #[inline]
    pub fn class_of(&self, s: usize, t: usize) -> Option<Action> {
        self.class[self.idx(s, t)]
    }

    /// `r_h` of `(s, t)`.
    #[inline]
    pub fn fwd_ret_of(&self, s: usize, t: usize) -> f64 {
        self.fwd_ret[self.idx(s, t)]
    }

    /// Vol-scaled return of `(s, t)`.
    #[inline]
    pub fn vol_ret_of(&self, s: usize, t: usize) -> f64 {
        self.vol_ret[self.idx(s, t)]
    }

    /// `mae_h` of `(s, t)`.
    #[inline]
    pub fn mae_of(&self, s: usize, t: usize) -> f64 {
        self.mae[self.idx(s, t)]
    }

    /// Stop flag of `(s, t)`.
    #[inline]
    pub fn stop_of(&self, s: usize, t: usize) -> bool {
        self.stop[self.idx(s, t)]
    }

    /// Number of existing labels.
    pub fn num_existing(&self) -> usize {
        self.class.iter().filter(|c| c.is_some()).count()
    }

    /// Counts of BUY, SELL, HOLD among existing labels.
    pub fn class_counts(&self) -> [usize; 3] {
        let mut c = [0; 3];
        for a in self.class.iter().flatten() {
            c[a.id()] += 1;
        }
        c
    }
}

/// One label of the stock behind `c` (`None` if it does not exist): `(r_h, sigma_h, mae_h)`.
pub fn label_values(c: &SeriesCache, t: usize, h: usize) -> Option<(f64, f64, f64)> {
    let x = c.x;
    if !c.has_all(t + 1, t + 1 + h) {
        return None;
    }
    let rv20 = realized_vol(c, t, 20)?;
    let o1 = x.open[t + 1];
    let r_h = (x.open[t + 1 + h] / o1).ln();
    let mut low = f64::INFINITY;
    for k in t + 1..=t + h {
        low = low.min(x.low[k]);
    }
    let mae = (low / o1).ln();
    let sigma = rv20 * (h as f64 / TRADING_DAYS).sqrt();
    [r_h, sigma, mae]
        .iter()
        .all(|v| v.is_finite())
        .then_some((r_h, sigma, mae))
}

/// Computes labels for every stock and day.
pub fn compute_labels(panel: &Panel, config: &LabelConfig) -> Result<Labels> {
    config.validate()?;
    let (ns, nd, h) = (panel.num_securities(), panel.num_days(), config.horizon);
    let n = ns * nd;
    let mut out = Labels {
        config: *config,
        num_securities: ns,
        num_days: nd,
        class: vec![None; n],
        fwd_ret: vec![f64::NAN; n],
        sigma_h: vec![f64::NAN; n],
        theta: vec![f64::NAN; n],
        vol_ret: vec![f64::NAN; n],
        mae: vec![f64::NAN; n],
        stop: vec![false; n],
    };
    for s in 0..ns {
        let c = SeriesCache::new(&panel.stocks[s]);
        for t in 0..nd {
            let Some((r_h, sigma, mae)) = label_values(&c, t, h) else {
                continue;
            };
            let i = out.idx(s, t);
            let theta = config.threshold(sigma);
            out.class[i] = Some(classify(r_h, theta));
            out.fwd_ret[i] = r_h;
            out.sigma_h[i] = sigma;
            out.theta[i] = theta;
            out.vol_ret[i] = r_h / sigma.max(1e-4);
            out.mae[i] = mae;
            out.stop[i] = mae < -config.stop_frac;
        }
    }
    Ok(out)
}
