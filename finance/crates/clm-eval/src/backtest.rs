//! The deterministic long-only policy and the backtest accounting of C5.
//!
//! Per security, the action of decision day `t` sets the position held from the open
//! of `t+1` to the open of `t+2`: BUY → 1, SELL → 0, HOLD → unchanged (start 0 at the
//! first decision day of the segment). With cost `κ` one way, all simple returns:
//!
//! ```text
//! trade(t) = |p(t) - p(t-1)|                 executed at O[t+1]
//! gross(t) = p(t) * (O[t+2] / O[t+1] - 1)    accrues on day t+1 (0 if either open is missing)
//! cost(t)  = κ * trade(t)
//! net(t)   = gross(t) - cost(t)
//! ```
//!
//! A decision with `t + 2 > N - 1` is not traded (position unchanged, gross = cost =
//! 0) and produces no portfolio day: the portfolio series holds one day `d = t + 1` per
//! traded decision day, `R(d) = (1/N_s) Σ_s net_s(t)`.
//!
//! Nothing here reads labels: a day is HOLD only because the caller passed `None`
//! (invalid input window, C2) or chose HOLD; label availability (C3) never affects
//! trading.

use clm_data::types::Action;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::folds::Segment;

/// Trading days per year used for annualisation.
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;

/// Invalid backtest inputs.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum BacktestError {
    /// No security was given.
    #[error("backtest needs at least one security")]
    NoSecurities,
    /// The test segment is not inside `[0, N)`.
    #[error("test segment [{}, {}) is not inside [0, {n_days})", seg.start, seg.end)]
    BadSegment {
        /// The segment.
        seg: Segment,
        /// Calendar length.
        n_days: usize,
    },
    /// A per-security input has the wrong length.
    #[error("security {security}: {what} has length {got}, expected {expected}")]
    BadLength {
        /// Security index.
        security: usize,
        /// Which input.
        what: &'static str,
        /// Its length.
        got: usize,
        /// The required length.
        expected: usize,
    },
    /// `decisions.len() != opens.len()`.
    #[error("{decisions} decision series for {securities} securities")]
    SecurityCount {
        /// Number of decision series.
        decisions: usize,
        /// Number of open series.
        securities: usize,
    },
    /// The cost is negative or not finite.
    #[error("kappa must be finite and >= 0, got {0}")]
    BadKappa(f64),
    /// Daily series to concatenate are not in strictly increasing day order.
    #[error("daily series overlap or are out of order at day {0}")]
    DaysNotIncreasing(usize),
}

/// The positions and trades of the C5 policy for one security's actions, starting
/// flat. `pos[i] ∈ {0, 1}`, `trade[i] = |pos[i] - pos[i-1]|`.
pub fn policy(actions: &[Action]) -> (Vec<u8>, Vec<u8>) {
    let mut prev = 0u8;
    let mut pos = Vec::with_capacity(actions.len());
    let mut trade = Vec::with_capacity(actions.len());
    for &a in actions {
        let p = match a {
            Action::Buy => 1,
            Action::Sell => 0,
            Action::Hold => prev,
        };
        trade.push(p.abs_diff(prev));
        pos.push(p);
        prev = p;
    }
    (pos, trade)
}

/// The `backtest` command of the Lean `clmcheck` oracle: `backtest n a1 .. an`.
pub fn render_backtest_command(actions: &[Action]) -> String {
    let mut s = format!("backtest {}", actions.len());
    for a in actions {
        s.push_str(&format!(" {}", a.id()));
    }
    s
}

/// The oracle's output lines of a `backtest` command: `pos p1 .. pn`, `trade t1 .. tn`.
pub fn render_backtest(actions: &[Action]) -> [String; 2] {
    let (pos, trade) = policy(actions);
    let line = |name: &str, v: &[u8]| {
        let mut s = name.to_string();
        for x in v {
            s.push_str(&format!(" {x}"));
        }
        s
    };
    [line("pos", &pos), line("trade", &trade)]
}

/// One security's ledger over the decision days `ts..te` of the test segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityLedger {
    /// Position `p(t) ∈ {0, 1}` chosen at each decision day.
    pub position: Vec<u8>,
    /// `|p(t) - p(t-1)|`.
    pub trade: Vec<u8>,
    /// Gross simple return of the sleeve.
    pub gross: Vec<f64>,
    /// `κ * trade`.
    pub cost: Vec<f64>,
    /// `gross - cost`.
    pub net: Vec<f64>,
    /// False for decisions with `t + 2 > N - 1` (not traded).
    pub traded: Vec<bool>,
}

/// The portfolio's daily series (one entry per traded decision day `t`, on day `t+1`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DailySeries {
    /// Day index `d = t + 1`, strictly increasing.
    pub day: Vec<usize>,
    /// Portfolio net simple return `R(d)`.
    pub ret: Vec<f64>,
    /// Turnover `(1/N_s) Σ_s trade_s(t)`.
    pub turnover: Vec<f64>,
    /// Gross exposure `(1/N_s) Σ_s p_s(t)`.
    pub gross_exposure: Vec<f64>,
}

impl DailySeries {
    /// Number of days.
    pub fn len(&self) -> usize {
        self.day.len()
    }

    /// True when there is no day.
    pub fn is_empty(&self) -> bool {
        self.day.is_empty()
    }

    /// Concatenate the series of several folds (test segments disjoint and increasing);
    /// errors unless the days are strictly increasing across all parts.
    pub fn concat(parts: &[DailySeries]) -> Result<DailySeries, BacktestError> {
        let mut out = DailySeries::default();
        for p in parts {
            for i in 0..p.len() {
                if let Some(&last) = out.day.last()
                    && p.day[i] <= last
                {
                    return Err(BacktestError::DaysNotIncreasing(p.day[i]));
                }
                out.day.push(p.day[i]);
                out.ret.push(p.ret[i]);
                out.turnover.push(p.turnover[i]);
                out.gross_exposure.push(p.gross_exposure[i]);
            }
        }
        Ok(out)
    }

    /// The days `i` with `keep[i]` (e.g. one regime).
    pub fn select(&self, keep: &[bool]) -> DailySeries {
        assert_eq!(keep.len(), self.len());
        let mut out = DailySeries::default();
        for i in (0..self.len()).filter(|&i| keep[i]) {
            out.day.push(self.day[i]);
            out.ret.push(self.ret[i]);
            out.turnover.push(self.turnover[i]);
            out.gross_exposure.push(self.gross_exposure[i]);
        }
        out
    }
}

/// A backtest over one test segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BacktestResult {
    /// The decision days `[ts, te)`.
    pub segment: Segment,
    /// One-way cost used.
    pub kappa: f64,
    /// Per-security ledgers, aligned to `ts..te`.
    pub securities: Vec<SecurityLedger>,
    /// The portfolio series.
    pub daily: DailySeries,
}

fn valid_price(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Run the C5 backtest.
///
/// * `opens[s][t]` — open of security `s` on day `t` (NaN, non-finite or `<= 0` =
///   missing), each of length `n_days`;
/// * `test` — the decision days `[ts, te)`;
/// * `decisions[s][i]` — action of security `s` on day `ts + i`. `None` → HOLD, and
///   per C5 the caller passes `None` **only** when the security's *input* window is
///   invalid that day (C2). Label existence (C3) must never decide this: it depends on
///   bars after `t`. Build decisions from every input-valid day of `[ts, te)`, not from
///   [`crate::purging::FoldSamples::test`] / the fold's test keep range, which are
///   label-gated and meant for prediction-quality metrics only;
/// * `kappa` — one-way cost as a fraction.
pub fn run_backtest(
    opens: &[Vec<f64>],
    test: Segment,
    n_days: usize,
    decisions: &[Vec<Option<Action>>],
    kappa: f64,
) -> Result<BacktestResult, BacktestError> {
    if opens.is_empty() {
        return Err(BacktestError::NoSecurities);
    }
    if decisions.len() != opens.len() {
        return Err(BacktestError::SecurityCount {
            decisions: decisions.len(),
            securities: opens.len(),
        });
    }
    if test.start > test.end || test.end > n_days {
        return Err(BacktestError::BadSegment { seg: test, n_days });
    }
    if !(kappa.is_finite() && kappa >= 0.0) {
        return Err(BacktestError::BadKappa(kappa));
    }
    let len = test.len();
    for (s, (o, d)) in opens.iter().zip(decisions).enumerate() {
        if o.len() != n_days {
            return Err(BacktestError::BadLength {
                security: s,
                what: "opens",
                got: o.len(),
                expected: n_days,
            });
        }
        if d.len() != len {
            return Err(BacktestError::BadLength {
                security: s,
                what: "decisions",
                got: d.len(),
                expected: len,
            });
        }
    }

    let mut ledgers = Vec::with_capacity(opens.len());
    for (o, d) in opens.iter().zip(decisions) {
        let mut l = SecurityLedger {
            position: Vec::with_capacity(len),
            trade: Vec::with_capacity(len),
            gross: Vec::with_capacity(len),
            cost: Vec::with_capacity(len),
            net: Vec::with_capacity(len),
            traded: Vec::with_capacity(len),
        };
        let mut prev = 0u8;
        for (i, t) in test.days().enumerate() {
            // t + 2 <= N - 1, written without underflow.
            let traded = t + 3 <= n_days;
            let a = if traded {
                d[i].unwrap_or(Action::Hold)
            } else {
                Action::Hold
            };
            let p = match a {
                Action::Buy => 1u8,
                Action::Sell => 0,
                Action::Hold => prev,
            };
            let trade = p.abs_diff(prev);
            let gross = if traded && p == 1 && valid_price(o[t + 1]) && valid_price(o[t + 2]) {
                o[t + 2] / o[t + 1] - 1.0
            } else {
                0.0
            };
            let cost = kappa * trade as f64;
            l.position.push(p);
            l.trade.push(trade);
            l.gross.push(gross);
            l.cost.push(cost);
            l.net.push(gross - cost);
            l.traded.push(traded);
            prev = p;
        }
        ledgers.push(l);
    }

    let ns = opens.len() as f64;
    let mut daily = DailySeries::default();
    for (i, t) in test.days().enumerate() {
        if !ledgers[0].traded[i] {
            continue;
        }
        let (mut r, mut tr, mut ge) = (0.0, 0.0, 0.0);
        for l in &ledgers {
            r += l.net[i];
            tr += l.trade[i] as f64;
            ge += l.position[i] as f64;
        }
        daily.day.push(t + 1);
        daily.ret.push(r / ns);
        daily.turnover.push(tr / ns);
        daily.gross_exposure.push(ge / ns);
    }
    Ok(BacktestResult {
        segment: test,
        kappa,
        securities: ledgers,
        daily,
    })
}

/// Equal-weight buy-and-hold over the same decision days: BUY on the first decision
/// day, HOLD afterwards (each sleeve pays the entry cost once).
pub fn buy_and_hold(
    opens: &[Vec<f64>],
    test: Segment,
    n_days: usize,
    kappa: f64,
) -> Result<BacktestResult, BacktestError> {
    let len = test.len();
    let dec: Vec<Option<Action>> = (0..len)
        .map(|i| if i == 0 { Some(Action::Buy) } else { None })
        .collect();
    run_backtest(opens, test, n_days, &vec![dec; opens.len()], kappa)
}

/// Mean (0 for an empty slice).
pub fn mean(r: &[f64]) -> f64 {
    if r.is_empty() {
        0.0
    } else {
        r.iter().sum::<f64>() / r.len() as f64
    }
}

/// Population standard deviation (0 for an empty or constant slice; exactly 0 for a
/// constant one, so ratios over it are not blown up by rounding in the mean).
pub fn pop_std(r: &[f64]) -> f64 {
    if r.is_empty() || r.iter().all(|&x| x == r[0]) {
        return 0.0;
    }
    let m = mean(r);
    (r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / r.len() as f64).sqrt()
}

/// Annualised Sharpe `sqrt(252) * mean / std` (population std); 0 when `std == 0`.
pub fn sharpe(r: &[f64]) -> f64 {
    let sd = pop_std(r);
    if sd > 0.0 {
        TRADING_DAYS_PER_YEAR.sqrt() * mean(r) / sd
    } else {
        0.0
    }
}

/// Annualised Sortino `sqrt(252) * mean / sqrt(mean(min(R,0)^2))`; 0 when there is
/// no negative day.
pub fn sortino(r: &[f64]) -> f64 {
    if r.is_empty() {
        return 0.0;
    }
    let dd = (r.iter().map(|x| x.min(0.0).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
    if dd > 0.0 {
        TRADING_DAYS_PER_YEAR.sqrt() * mean(r) / dd
    } else {
        0.0
    }
}

/// Maximum drawdown of the equity `prod(1 + R)` starting at 1, as a positive fraction
/// `max_k (1 - E_k / max_{j<=k} E_j)`.
pub fn max_drawdown(r: &[f64]) -> f64 {
    let (mut eq, mut peak, mut mdd) = (1.0f64, 1.0f64, 0.0f64);
    for x in r {
        eq *= 1.0 + x;
        peak = peak.max(eq);
        mdd = mdd.max(1.0 - eq / peak);
    }
    mdd
}

/// `sum(R > 0) / |sum(R < 0)|`; `None` when no day lost money (undefined).
pub fn profit_factor(r: &[f64]) -> Option<f64> {
    let gain: f64 = r.iter().filter(|&&x| x > 0.0).sum();
    let loss: f64 = r.iter().filter(|&&x| x < 0.0).sum();
    if loss < 0.0 { Some(gain / -loss) } else { None }
}

/// Compounded total return `prod(1 + R) - 1`.
pub fn total_return(r: &[f64]) -> f64 {
    r.iter().fold(1.0, |e, x| e * (1.0 + x)) - 1.0
}

/// The reported metrics of C5 on a daily portfolio series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BacktestMetrics {
    /// `252 * mean(R)`.
    pub ann_return: f64,
    /// `sqrt(252) * mean / std` (population; 0 if std = 0).
    pub sharpe: f64,
    /// `sqrt(252) * mean / downside deviation` (0 if no losing day).
    pub sortino: f64,
    /// Max drawdown of the compounded equity, positive fraction.
    pub max_drawdown: f64,
    /// Gains over losses; `None` if no losing day.
    pub profit_factor: Option<f64>,
    /// Mean daily turnover.
    pub mean_turnover: f64,
    /// Compounded total net return.
    pub total_net_return: f64,
    /// Number of days.
    pub n_days: usize,
}

/// C5 metrics of a daily series.
pub fn backtest_metrics(daily: &DailySeries) -> BacktestMetrics {
    let r = &daily.ret;
    BacktestMetrics {
        ann_return: TRADING_DAYS_PER_YEAR * mean(r),
        sharpe: sharpe(r),
        sortino: sortino(r),
        max_drawdown: max_drawdown(r),
        profit_factor: profit_factor(r),
        mean_turnover: mean(&daily.turnover),
        total_net_return: total_return(r),
        n_days: r.len(),
    }
}
