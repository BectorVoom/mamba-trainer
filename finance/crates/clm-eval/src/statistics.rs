//! Statistical comparison of strategies (C6): stationary bootstrap intervals, paired
//! per-year comparison, cost sensitivity and volatility regimes.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backtest::{BacktestMetrics, DailySeries, backtest_metrics, mean, sharpe};

/// Invalid inputs to a statistic.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum StatsError {
    /// The series is empty.
    #[error("empty series")]
    Empty,
    /// Paired series are not aligned day by day.
    #[error("paired series cover different days (first difference at position {0})")]
    DaysDiffer(usize),
    /// Inputs that must have equal length do not.
    #[error("length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),
    /// A bootstrap setting is out of range.
    #[error("bad bootstrap config: {0}")]
    BadConfig(&'static str),
    /// A value that must be finite is not (position in the offending series; for
    /// paired inputs the first series is checked first).
    #[error("non-finite value at position {0}")]
    NonFinite(usize),
}

/// `Err(NonFinite(i))` for the first non-finite value. Every statistic validates its
/// inputs with this before resampling: one NaN would otherwise poison every replicate
/// (and NaN-unaware sorting would make the percentiles meaningless).
pub fn check_finite(x: &[f64]) -> Result<(), StatsError> {
    match x.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(StatsError::NonFinite(i)),
        None => Ok(()),
    }
}

/// Stationary bootstrap settings (C6 defaults: B = 2000, mean block 20, 95%).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Number of resamples `B`.
    pub replicates: usize,
    /// Mean block length (geometric block lengths with `p = 1 / mean_block`).
    pub mean_block: f64,
    /// RNG seed (`StdRng::seed_from_u64`).
    pub seed: u64,
    /// Coverage of the percentile interval.
    pub level: f64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            replicates: 2000,
            mean_block: 20.0,
            seed: 0x5EED_C1A5,
            level: 0.95,
        }
    }
}

impl BootstrapConfig {
    fn validate(&self) -> Result<(), StatsError> {
        if self.replicates == 0 {
            return Err(StatsError::BadConfig("replicates must be > 0"));
        }
        if !(self.mean_block >= 1.0 && self.mean_block.is_finite()) {
            return Err(StatsError::BadConfig("mean_block must be finite and >= 1"));
        }
        if !(self.level > 0.0 && self.level < 1.0) {
            return Err(StatsError::BadConfig("level must be in (0, 1)"));
        }
        Ok(())
    }
}

/// A point estimate with a percentile interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// The statistic on the original series.
    pub estimate: f64,
    /// Lower percentile bound.
    pub lo: f64,
    /// Upper percentile bound.
    pub hi: f64,
}

/// One stationary-bootstrap (Politis–Romano) resample of `0..n`: start uniformly,
/// then at each step start a new block at a uniform index with probability
/// `1 / mean_block`, else continue to the next index (circularly).
pub fn stationary_bootstrap_indices(n: usize, mean_block: f64, rng: &mut StdRng) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let p = 1.0 / mean_block;
    let mut idx = Vec::with_capacity(n);
    let mut cur = rng.random_range(0..n);
    idx.push(cur);
    for _ in 1..n {
        cur = if rng.random::<f64>() < p {
            rng.random_range(0..n)
        } else {
            (cur + 1) % n
        };
        idx.push(cur);
    }
    idx
}

/// Type-7 (linear interpolation) quantile of sorted data.
fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    let h = (s.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    s[lo] + (h - lo as f64) * (s[hi] - s[lo])
}

fn percentile_interval(estimate: f64, mut reps: Vec<f64>, level: f64) -> Interval {
    reps.sort_by(f64::total_cmp);
    let a = (1.0 - level) / 2.0;
    Interval {
        estimate,
        lo: quantile_sorted(&reps, a),
        hi: quantile_sorted(&reps, 1.0 - a),
    }
}

/// A statistic of a return series.
pub type Statistic<'a> = &'a dyn Fn(&[f64]) -> f64;

/// Bootstrap several statistics of one series with shared resampled indices.
fn bootstrap_many<const M: usize>(
    x: &[f64],
    stats: [Statistic<'_>; M],
    cfg: &BootstrapConfig,
) -> Result<[Interval; M], StatsError> {
    cfg.validate()?;
    if x.is_empty() {
        return Err(StatsError::Empty);
    }
    check_finite(x)?;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut reps: [Vec<f64>; M] = std::array::from_fn(|_| Vec::with_capacity(cfg.replicates));
    let mut buf = vec![0.0; x.len()];
    for _ in 0..cfg.replicates {
        let idx = stationary_bootstrap_indices(x.len(), cfg.mean_block, &mut rng);
        for (b, &i) in buf.iter_mut().zip(&idx) {
            *b = x[i];
        }
        for (m, st) in stats.iter().enumerate() {
            reps[m].push(st(&buf));
        }
    }
    let mut out = [Interval {
        estimate: 0.0,
        lo: 0.0,
        hi: 0.0,
    }; M];
    for (m, r) in reps.into_iter().enumerate() {
        out[m] = percentile_interval(stats[m](x), r, cfg.level);
    }
    Ok(out)
}

/// Percentile bootstrap interval of any statistic of one series.
pub fn bootstrap_statistic(
    x: &[f64],
    stat: Statistic<'_>,
    cfg: &BootstrapConfig,
) -> Result<Interval, StatsError> {
    Ok(bootstrap_many(x, [stat], cfg)?[0])
}

/// Intervals for the mean daily return and the annualised Sharpe of one series.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeriesBootstrap {
    /// Mean daily return.
    pub mean: Interval,
    /// Annualised Sharpe (population std, 0 if std = 0).
    pub sharpe: Interval,
}

/// Bootstrap the mean daily return and Sharpe of one series (same resamples).
pub fn bootstrap_series(r: &[f64], cfg: &BootstrapConfig) -> Result<SeriesBootstrap, StatsError> {
    let [m, s] = bootstrap_many(r, [&mean, &sharpe], cfg)?;
    Ok(SeriesBootstrap { mean: m, sharpe: s })
}

/// Paired bootstrap of two strategies over the same days.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairedBootstrap {
    /// Sharpe of A.
    pub sharpe_a: Interval,
    /// Sharpe of B.
    pub sharpe_b: Interval,
    /// `Sharpe(A) - Sharpe(B)` with the same resampled days for both.
    pub sharpe_diff: Interval,
    /// `mean(A - B)` (daily).
    pub mean_diff: Interval,
    /// Number of days.
    pub n_days: usize,
}

/// Paired stationary bootstrap of returns `a` and `b` on days `days_a` / `days_b`;
/// errors unless the days are identical.
pub fn paired_bootstrap(
    days_a: &[usize],
    a: &[f64],
    days_b: &[usize],
    b: &[f64],
    cfg: &BootstrapConfig,
) -> Result<PairedBootstrap, StatsError> {
    cfg.validate()?;
    if days_a.len() != a.len() {
        return Err(StatsError::LengthMismatch(days_a.len(), a.len()));
    }
    if days_b.len() != b.len() {
        return Err(StatsError::LengthMismatch(days_b.len(), b.len()));
    }
    if let Some(i) = (0..days_a.len().max(days_b.len())).find(|&i| days_a.get(i) != days_b.get(i)) {
        return Err(StatsError::DaysDiffer(i));
    }
    let n = a.len();
    if n == 0 {
        return Err(StatsError::Empty);
    }
    check_finite(a)?;
    check_finite(b)?;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut reps: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(cfg.replicates));
    let (mut ra, mut rb, mut rd) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for _ in 0..cfg.replicates {
        let idx = stationary_bootstrap_indices(n, cfg.mean_block, &mut rng);
        for (k, &i) in idx.iter().enumerate() {
            ra[k] = a[i];
            rb[k] = b[i];
            rd[k] = a[i] - b[i];
        }
        let (sa, sb) = (sharpe(&ra), sharpe(&rb));
        reps[0].push(sa);
        reps[1].push(sb);
        reps[2].push(sa - sb);
        reps[3].push(mean(&rd));
    }
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let [r0, r1, r2, r3] = reps;
    Ok(PairedBootstrap {
        sharpe_a: percentile_interval(sharpe(a), r0, cfg.level),
        sharpe_b: percentile_interval(sharpe(b), r1, cfg.level),
        sharpe_diff: percentile_interval(sharpe(a) - sharpe(b), r2, cfg.level),
        mean_diff: percentile_interval(mean(&d), r3, cfg.level),
        n_days: n,
    })
}

/// [`paired_bootstrap`] of two portfolio series.
pub fn paired_bootstrap_series(
    a: &DailySeries,
    b: &DailySeries,
    cfg: &BootstrapConfig,
) -> Result<PairedBootstrap, StatsError> {
    paired_bootstrap(&a.day, &a.ret, &b.day, &b.ret, cfg)
}

/// Per-year comparison of a metric of A and B.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YearComparison {
    /// The years compared.
    pub years: Vec<i32>,
    /// Metric of A per year.
    pub a: Vec<f64>,
    /// Metric of B per year.
    pub b: Vec<f64>,
    /// Years where A is strictly better.
    pub wins_a: usize,
    /// Years with equal values.
    pub ties: usize,
    /// Mean of `a - b`.
    pub mean_diff: f64,
    /// Sample standard deviation (`n - 1`) of `a - b` (0 for fewer than 2 years).
    pub sd_diff: f64,
}

/// Compare per-year values of a metric; `higher_is_better` is true for Sharpe, false
/// for log loss. Differences are always `a - b`.
pub fn paired_by_year(
    years: &[i32],
    a: &[f64],
    b: &[f64],
    higher_is_better: bool,
) -> Result<YearComparison, StatsError> {
    if years.len() != a.len() {
        return Err(StatsError::LengthMismatch(years.len(), a.len()));
    }
    if a.len() != b.len() {
        return Err(StatsError::LengthMismatch(a.len(), b.len()));
    }
    check_finite(a)?;
    check_finite(b)?;
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let wins_a = a
        .iter()
        .zip(b)
        .filter(|(x, y)| if higher_is_better { x > y } else { x < y })
        .count();
    let ties = a.iter().zip(b).filter(|(x, y)| x == y).count();
    let n = d.len();
    let m = mean(&d);
    let sd_diff = if n > 1 {
        (d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    Ok(YearComparison {
        years: years.to_vec(),
        a: a.to_vec(),
        b: b.to_vec(),
        wins_a,
        ties,
        mean_diff: m,
        sd_diff,
    })
}

/// Group a daily series by the calendar year of its days (`year_of_day[d]`), returning
/// `(years, returns per year)` in day order.
pub fn split_by_year(daily: &DailySeries, year_of_day: &[i32]) -> (Vec<i32>, Vec<Vec<f64>>) {
    let mut years: Vec<i32> = Vec::new();
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for (&d, &r) in daily.day.iter().zip(&daily.ret) {
        let y = year_of_day[d];
        if years.last() != Some(&y) {
            years.push(y);
            groups.push(Vec::new());
        }
        groups.last_mut().expect("pushed").push(r);
    }
    (years, groups)
}

/// Per-year Sharpe comparison of two aligned portfolio series (days must match).
pub fn paired_sharpe_by_year(
    a: &DailySeries,
    b: &DailySeries,
    year_of_day: &[i32],
) -> Result<YearComparison, StatsError> {
    if let Some(i) = (0..a.len().max(b.len())).find(|&i| a.day.get(i) != b.day.get(i)) {
        return Err(StatsError::DaysDiffer(i));
    }
    check_finite(&a.ret)?;
    check_finite(&b.ret)?;
    let (years, ga) = split_by_year(a, year_of_day);
    let (_, gb) = split_by_year(b, year_of_day);
    let sa: Vec<f64> = ga.iter().map(|r| sharpe(r)).collect();
    let sb: Vec<f64> = gb.iter().map(|r| sharpe(r)).collect();
    paired_by_year(&years, &sa, &sb, true)
}

/// The C6 cost grid, one-way, in basis points.
pub const COST_GRID_BPS: [f64; 5] = [0.0, 2.5, 5.0, 10.0, 20.0];

/// Metrics at one cost level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostPoint {
    /// One-way cost in basis points.
    pub kappa_bps: f64,
    /// One-way cost as a fraction (`bps * 1e-4`).
    pub kappa: f64,
    /// Backtest metrics at that cost.
    pub metrics: BacktestMetrics,
}

/// Re-run a backtest closure (taking `kappa` as a fraction) at every cost of
/// [`COST_GRID_BPS`].
pub fn cost_sensitivity<E>(
    mut run: impl FnMut(f64) -> Result<BacktestMetrics, E>,
) -> Result<Vec<CostPoint>, E> {
    COST_GRID_BPS
        .iter()
        .map(|&bps| {
            let kappa = bps * 1e-4;
            Ok(CostPoint {
                kappa_bps: bps,
                kappa,
                metrics: run(kappa)?,
            })
        })
        .collect()
}

/// Median of a slice (mean of the two middle values for even length); `None` if empty
/// or any value is not finite.
pub fn median(x: &[f64]) -> Option<f64> {
    if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    Some(if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    })
}

/// Point-in-time regime flags (C6): `true` (high volatility) iff the market `rv20` at a
/// decision day is strictly above its median over the fold's **training** days.
/// Returns `(threshold, flags)` aligned with `decision_rv`.
pub fn regime_flags(decision_rv: &[f64], train_rv: &[f64]) -> Result<(f64, Vec<bool>), StatsError> {
    if train_rv.is_empty() {
        return Err(StatsError::Empty);
    }
    if let Some(i) = train_rv.iter().position(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite(i));
    }
    if let Some(i) = decision_rv.iter().position(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite(i));
    }
    let m = median(train_rv).ok_or(StatsError::Empty)?;
    Ok((m, decision_rv.iter().map(|&v| v > m).collect()))
}

/// Metrics of a daily series split by a regime flag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegimeMetrics {
    /// Metrics over the days flagged `true` (high volatility).
    pub high: BacktestMetrics,
    /// Metrics over the days flagged `false`.
    pub low: BacktestMetrics,
}

/// Split a daily series by per-day flags (`flags[i]` belongs to `daily.day[i]`, i.e. to
/// decision day `day - 1`) and compute metrics per regime.
pub fn regime_split(daily: &DailySeries, flags: &[bool]) -> Result<RegimeMetrics, StatsError> {
    if flags.len() != daily.len() {
        return Err(StatsError::LengthMismatch(flags.len(), daily.len()));
    }
    check_finite(&daily.ret)?;
    check_finite(&daily.turnover)?;
    let low: Vec<bool> = flags.iter().map(|f| !f).collect();
    Ok(RegimeMetrics {
        high: backtest_metrics(&daily.select(flags)),
        low: backtest_metrics(&daily.select(&low)),
    })
}
