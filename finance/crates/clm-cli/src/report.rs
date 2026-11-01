//! Aggregate a CV run: prediction quality, calibration, trading after costs,
//! cost sensitivity, regimes, and paired comparisons with uncertainty.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use clm_data::{Action, FEATURE_NAMES};
use clm_eval::backtest::{BacktestMetrics, DailySeries, backtest_metrics, buy_and_hold, run_backtest};
use clm_eval::calibration::{CalibrationMethod, Calibrator};
use clm_eval::folds::{Fold, make_folds};
use clm_eval::metrics::{ClassificationMetrics, IcSummary, classification_metrics, rank_ic, regression_metrics};
use clm_eval::oof::{OofRecord, read_json, read_oof_parquet, write_json};
use clm_eval::statistics::{
    BootstrapConfig, PairedBootstrap, SeriesBootstrap, YearComparison, bootstrap_series, paired_bootstrap_series,
    paired_by_year, paired_sharpe_by_year, regime_flags, regime_split,
};
use serde::Serialize;

use crate::config::ExperimentConfig;
use crate::config::anyhow_like::{Result, err, wrap};
use crate::cv::{FoldModelResult, decide};
use crate::data::Dataset;

/// Everything reported for one model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelReport {
    /// Model name.
    pub model: String,
    /// Folds found.
    pub folds: Vec<i32>,
    /// Prediction quality with the raw softmax.
    pub raw: ClassificationMetrics,
    /// Prediction quality after the per-fold calibration.
    pub calibrated: ClassificationMetrics,
    /// Test log loss / Brier / ECE per calibration method (refitted on each fold's `cal`).
    pub calibration_methods: Vec<(String, f64, f64, f64)>,
    /// Daily rank IC of `p_buy - p_sell` against the forward return.
    pub ic: IcSummary,
    /// RMSE of the predicted vol-scaled return, if the model has the head.
    pub ret_rmse: Option<f64>,
    /// Trading metrics at the configured cost, primary policy (fixed tau).
    pub trading: BacktestMetrics,
    /// Trading metrics with each fold's tau tuned on its calibration segment.
    pub trading_tuned_tau: BacktestMetrics,
    /// Bootstrap intervals of the daily returns.
    pub bootstrap: SeriesBootstrap,
    /// Trading metrics per one-way cost (bps).
    pub costs: Vec<(f64, BacktestMetrics)>,
    /// High / low market-volatility regimes.
    pub regimes: Option<(BacktestMetrics, BacktestMetrics)>,
    /// Sharpe per test year.
    pub yearly_sharpe: Vec<(i32, f64)>,
    /// Action shares on test days under the primary policy (BUY, SELL, HOLD).
    pub action_shares: [f64; 3],
    /// Mean decision threshold chosen on `cal`.
    pub mean_tau: f64,
    /// Mean training seconds per fold.
    pub mean_seconds: f64,
    /// Parameters (last fold).
    pub num_params: usize,
    /// Mean training samples per second.
    pub samples_per_second: f64,
    /// Largest tensor-owned device memory of a fold's fit, MB (0 when the run
    /// predates the measurement).
    pub peak_device_mb: f64,
}

/// A paired comparison against the reference model.
#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    /// Model.
    pub model: String,
    /// Reference.
    pub reference: String,
    /// Paired stationary bootstrap of the daily returns.
    pub bootstrap: PairedBootstrap,
    /// Per-year Sharpe.
    pub yearly_sharpe: YearComparison,
    /// Per-year calibrated log loss (lower is better).
    pub yearly_log_loss: YearComparison,
}

struct Loaded {
    results: Vec<FoldModelResult>,
    test: Vec<Vec<OofRecord>>,
    cal: Vec<Vec<OofRecord>>,
    folds: Vec<Fold>,
}

fn load_model(run: &Path, name: &str, folds: &[Fold]) -> Option<Loaded> {
    let mut out = Loaded {
        results: vec![],
        test: vec![],
        cal: vec![],
        folds: vec![],
    };
    for f in folds {
        let d = run.join(format!("fold_{}", f.test_year)).join(name);
        let Ok(result) = read_json::<FoldModelResult>(d.join("model.json")) else {
            continue;
        };
        let (Ok(test), Ok(cal)) = (read_oof_parquet(d.join("oof_test.parquet")), read_oof_parquet(d.join("oof_cal.parquet"))) else {
            continue;
        };
        out.results.push(result);
        out.test.push(test);
        out.cal.push(cal);
        out.folds.push(f.clone());
    }
    (!out.results.is_empty()).then_some(out)
}

/// The threshold a backtest uses: fixed for every model (primary), or the one
/// each fold chose on its calibration segment (secondary).
#[derive(Clone, Copy)]
enum Tau {
    Fixed(f64),
    Tuned,
}

fn backtest_folds(
    ds: &Dataset,
    opens: &[Vec<f64>],
    m: &Loaded,
    kappa: f64,
    tau: Tau,
) -> Result<(DailySeries, [usize; 3])> {
    let ns = ds.panel.stocks.len();
    let mut parts = Vec::new();
    let mut counts = [0usize; 3];
    for ((fold, recs), res) in m.folds.iter().zip(&m.test).zip(&m.results) {
        let mut decisions = vec![vec![None; fold.test.len()]; ns];
        for r in recs {
            let t = r.t as usize;
            if fold.test.contains(t) {
                let tau = match tau {
                    Tau::Fixed(v) => v,
                    Tau::Tuned => res.tau,
                };
                decisions[r.s as usize][t - fold.test.start] = Some(decide(r.calibrated, tau));
            }
        }
        // Every security x test day, an absent prediction (invalid input) is HOLD.
        for row in &decisions {
            for d in row {
                counts[d.unwrap_or(Action::Hold).id()] += 1;
            }
        }
        let bt = run_backtest(opens, fold.test, ds.dates.len(), &decisions, kappa).map_err(wrap("backtest"))?;
        parts.push(bt.daily);
    }
    Ok((DailySeries::concat(&parts).map_err(wrap("concat"))?, counts))
}

fn market_rv(ds: &Dataset) -> Vec<f64> {
    let f = FEATURE_NAMES.iter().position(|n| *n == "mkt_rv20").expect("mkt_rv20 feature");
    (0..ds.dates.len())
        .map(|t| {
            (0..ds.features.num_securities)
                .find(|&s| ds.features.is_valid(s, t))
                .map_or(f64::NAN, |s| ds.features.get(s, t, f) as f64)
        })
        .collect()
}

/// Build the report for `run` into `run/report.{json,md}`. The run must have been
/// made with this config and data; a different source version is allowed (report
/// code changes do not change stored predictions) but recorded. Every configured
/// model must have results unless `allow_partial`.
pub fn run(cfg: &ExperimentConfig, run: &Path, allow_partial: bool) -> Result<()> {
    let ds = Dataset::load(cfg)?;
    let meta: serde_json::Value = read_json(run.join("run.json")).map_err(wrap("run.json"))?;
    let run_cfg: ExperimentConfig = serde_json::from_value(meta.get("config").cloned().unwrap_or_default())
        .map_err(wrap("run.json config"))?;
    let same_cfg = serde_json::to_value(&run_cfg).ok() == serde_json::to_value(cfg).ok();
    if !same_cfg || meta.get("dataset_version").and_then(|v| v.as_str()) != Some(ds.version.as_str()) {
        return Err(err(format!(
            "{} was produced with another config or dataset; report it with the config it was run with",
            run.display()
        )));
    }
    let run_source = meta.get("source_hash").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    if run_source != env!("CLM_SOURCE_HASH") {
        eprintln!(
            "note: predictions were produced by source {run_source}, this report by {}",
            env!("CLM_SOURCE_HASH")
        );
    }
    let opens = ds.opens();
    let folds = make_folds(&ds.years, ds.dates.len(), &cfg.fold_config());
    let rv = market_rv(&ds);
    // Daily returns are reported on the accrual day `t + 1`; years are those of the
    // decision day `t`, so a test year's last decision stays in its year.
    let decision_year: Vec<i32> = (0..ds.years.len()).map(|d| ds.years[d.saturating_sub(1)]).collect();
    let boot = BootstrapConfig::default();
    let mut reports = Vec::new();
    let mut series: BTreeMap<String, DailySeries> = BTreeMap::new();
    let mut yearly_ll: BTreeMap<String, BTreeMap<i32, f64>> = BTreeMap::new();
    // Every table covers the same test years: the folds all reported models finished.
    let mut loaded = Vec::new();
    for spec in &cfg.models {
        match load_model(run, &spec.name, &folds) {
            Some(m) => loaded.push((spec, m)),
            None => {
                if !allow_partial {
                    return Err(err(format!("{}: no finished folds (pass --allow-partial to report without it)", spec.name)));
                }
                eprintln!("{}: no finished folds, left out of the report", spec.name);
            }
        }
    }
    let common: Vec<i32> = folds
        .iter()
        .map(|f| f.test_year)
        .filter(|y| loaded.iter().all(|(_, m)| m.results.iter().any(|r| r.test_year == *y)))
        .collect();
    if common.is_empty() {
        return Err(err("the reported models share no finished fold"));
    }
    for (spec, m) in &mut loaded {
        let before = m.results.len();
        let keep: Vec<bool> = m.results.iter().map(|r| common.contains(&r.test_year)).collect();
        retain_by(&mut m.results, &keep);
        retain_by(&mut m.test, &keep);
        retain_by(&mut m.cal, &keep);
        retain_by(&mut m.folds, &keep);
        if m.results.len() < before {
            eprintln!("{}: {} finished folds outside the common set left out", spec.name, before - m.results.len());
        }
    }
    let excluded: Vec<i32> = folds.iter().map(|f| f.test_year).filter(|y| !common.contains(y)).collect();
    for (spec, m) in loaded {
        let labelled: Vec<&OofRecord> = m.test.iter().flatten().filter(|r| r.label.is_some()).collect();
        let labels: Vec<u8> = labelled.iter().map(|r| r.label.unwrap()).collect();
        let raw_p: Vec<[f32; 3]> = labelled.iter().map(|r| r.probs).collect();
        let cal_p: Vec<[f32; 3]> = labelled.iter().map(|r| r.calibrated).collect();
        let raw = classification_metrics(&raw_p, &labels);
        let calibrated = classification_metrics(&cal_p, &labels);
        // Calibration methods, each refitted per fold on that fold's `cal` logits.
        let mut calibration_methods = Vec::new();
        for (name, method) in [
            ("none", CalibrationMethod::Identity),
            ("temperature", CalibrationMethod::Temperature),
            ("vector", CalibrationMethod::Vector),
            ("isotonic", CalibrationMethod::Isotonic),
        ] {
            let mut p_all = Vec::new();
            let mut l_all = Vec::new();
            for (cal, test) in m.cal.iter().zip(&m.test) {
                let cl: Vec<[f32; 3]> = cal.iter().map(|r| r.logits).collect();
                let cy: Vec<u8> = cal.iter().map(|r| r.label.expect("cal is labelled")).collect();
                let c = Calibrator::fit(method, &cl, &cy);
                for r in test.iter().filter(|r| r.label.is_some()) {
                    p_all.push(c.apply_row(&r.logits).map(|v| v as f32));
                    l_all.push(r.label.unwrap());
                }
            }
            let cm = classification_metrics(&p_all, &l_all);
            calibration_methods.push((name.to_string(), cm.log_loss, cm.brier, cm.ece));
        }
        let ic = rank_ic(
            &labelled.iter().map(|r| r.t).collect::<Vec<_>>(),
            &labelled.iter().map(|r| (r.calibrated[0] - r.calibrated[1]) as f64).collect::<Vec<_>>(),
            &labelled.iter().map(|r| r.fwd_ret as f64).collect::<Vec<_>>(),
        );
        let ret_rmse = labelled.first().and_then(|r| r.pred_ret).map(|_| {
            let pred: Vec<f64> = labelled.iter().map(|r| r.pred_ret.unwrap_or(0.0) as f64).collect();
            let target: Vec<f64> = labelled
                .iter()
                .map(|r| ds.labels.vol_ret_of(r.s as usize, r.t as usize).clamp(-5.0, 5.0))
                .collect();
            regression_metrics(&pred, &target).rmse
        });
        let primary = Tau::Fixed(cfg.policy_tau);
        let (daily, action_counts) = backtest_folds(&ds, &opens, &m, cfg.kappa(), primary)?;
        let tuned = backtest_metrics(&backtest_folds(&ds, &opens, &m, cfg.kappa(), Tau::Tuned)?.0);
        let trading = backtest_metrics(&daily);
        let bootstrap = bootstrap_series(&daily.ret, &boot).map_err(wrap("bootstrap"))?;
        let mut costs = Vec::new();
        for bps in [0.0, 2.5, 5.0, 10.0, 20.0] {
            costs.push((bps, backtest_metrics(&backtest_folds(&ds, &opens, &m, bps * 1e-4, primary)?.0)));
        }
        // Regimes: market rv20 at the decision day against the median over each
        // fold's training days (point in time).
        let mut flags = Vec::new();
        for (fold, part) in m.folds.iter().zip(split_by_fold(&daily, &m.folds)) {
            let train_rv: Vec<f64> = fold.train.days().map(|t| rv[t]).filter(|v| v.is_finite()).collect();
            let decision_rv: Vec<f64> = part.iter().map(|&d| rv[d - 1]).map(|v| if v.is_finite() { v } else { 0.0 }).collect();
            match regime_flags(&decision_rv, &train_rv) {
                Ok((_, f)) => flags.extend(f),
                Err(_) => flags.extend(std::iter::repeat_n(false, part.len())),
            }
        }
        let regimes = regime_split(&daily, &flags).ok().map(|r| (r.high, r.low));
        let mut by_year: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
        for (d, r) in daily.day.iter().zip(&daily.ret) {
            by_year.entry(decision_year[*d]).or_default().push(*r);
        }
        let yearly_sharpe = by_year
            .iter()
            .map(|(y, r)| (*y, clm_eval::backtest::sharpe(r)))
            .collect();
        let mut ll: BTreeMap<i32, f64> = BTreeMap::new();
        for (res, test) in m.results.iter().zip(&m.test) {
            let l: Vec<&OofRecord> = test.iter().filter(|r| r.label.is_some()).collect();
            let p: Vec<[f32; 3]> = l.iter().map(|r| r.calibrated).collect();
            let y: Vec<u8> = l.iter().map(|r| r.label.unwrap()).collect();
            ll.insert(res.test_year, clm_eval::metrics::log_loss(&p, &y));
        }
        yearly_ll.insert(spec.name.clone(), ll);
        let shares = action_counts.map(|c| c as f64);
        let total: f64 = shares.iter().sum();
        let report = ModelReport {
            model: spec.name.clone(),
            folds: m.results.iter().map(|r| r.test_year).collect(),
            raw,
            calibrated,
            calibration_methods,
            ic,
            ret_rmse,
            trading,
            trading_tuned_tau: tuned,
            bootstrap,
            costs,
            regimes,
            yearly_sharpe,
            action_shares: shares.map(|s| s / total.max(1.0)),
            mean_tau: m.results.iter().map(|r| r.tau).sum::<f64>() / m.results.len() as f64,
            mean_seconds: m.results.iter().map(|r| r.seconds).sum::<f64>() / m.results.len() as f64,
            num_params: m.results.last().map_or(0, |r| r.num_params),
            samples_per_second: {
                let v: Vec<f64> = m.results.iter().filter_map(|r| r.fit.as_ref().map(|f| f.samples_per_second)).collect();
                if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
            },
            peak_device_mb: m
                .results
                .iter()
                .filter_map(|r| r.fit.as_ref().map(|f| f.peak_device_bytes as f64 / 1e6))
                .fold(0.0, f64::max),
        };
        eprintln!(
            "{}: {} folds, cal log loss {:.4}, Sharpe {:.2}",
            report.model,
            report.folds.len(),
            report.calibrated.log_loss,
            report.trading.sharpe
        );
        series.insert(spec.name.clone(), daily);
        reports.push(report);
    }
    if reports.is_empty() {
        return Err(err("no model has finished folds"));
    }
    // Buy and hold over the reference's test days.
    let reference = cfg.reference.clone();
    let bh = {
        let mut parts = Vec::new();
        for f in &folds {
            if reports.iter().any(|r| r.model == reference && r.folds.contains(&f.test_year)) {
                parts.push(buy_and_hold(&opens, f.test, ds.dates.len(), cfg.kappa()).map_err(wrap("buy and hold"))?.daily);
            }
        }
        DailySeries::concat(&parts).map_err(wrap("concat"))?
    };
    let mut comparisons = Vec::new();
    if let Some(ref_series) = series.get(&reference) {
        for (name, s) in &series {
            if *name == reference || s.day != ref_series.day {
                continue;
            }
            let bootstrap = paired_bootstrap_series(s, ref_series, &boot).map_err(wrap("paired bootstrap"))?;
            let yearly_sharpe = paired_sharpe_by_year(s, ref_series, &decision_year).map_err(wrap("yearly"))?;
            let (a, b) = (&yearly_ll[name], &yearly_ll[&reference]);
            let years: Vec<i32> = a.keys().filter(|y| b.contains_key(y)).copied().collect();
            let yearly_log_loss = paired_by_year(
                &years,
                &years.iter().map(|y| a[y]).collect::<Vec<_>>(),
                &years.iter().map(|y| b[y]).collect::<Vec<_>>(),
                false,
            )
            .map_err(wrap("yearly log loss"))?;
            comparisons.push(Comparison {
                model: name.clone(),
                reference: reference.clone(),
                bootstrap,
                yearly_sharpe,
                yearly_log_loss,
            });
        }
    }
    // Every model against buy and hold, paired by day: is timing the market worth it?
    let mut vs_bh = Vec::new();
    for (name, s) in &series {
        if s.day == bh.day {
            vs_bh.push((name.clone(), paired_bootstrap_series(s, &bh, &boot).map_err(wrap("paired bootstrap vs buy and hold"))?));
        }
    }
    let bh_metrics = backtest_metrics(&bh);
    write_json(
        run.join("report.json"),
        &serde_json::json!({
            "models": reports,
            "comparisons": comparisons,
            "buy_and_hold": bh_metrics,
            "vs_buy_and_hold": vs_bh,
            "common_folds": common,
            "excluded_folds": excluded,
            "run_source_hash": run_source,
            "report_source_hash": env!("CLM_SOURCE_HASH"),
            "signature": meta.get("signature"),
        }),
    )
    .map_err(wrap("report.json"))?;
    std::fs::write(run.join("report.md"), markdown(cfg, &reports, &comparisons, &vs_bh, &bh_metrics, &common, &excluded))
        .map_err(wrap("report.md"))?;
    eprintln!("wrote {}", run.join("report.md").display());
    Ok(())
}

/// The accrual days of `daily` grouped by the fold whose test segment made them.
fn split_by_fold(daily: &DailySeries, folds: &[Fold]) -> Vec<Vec<usize>> {
    folds
        .iter()
        .map(|f| daily.day.iter().copied().filter(|&d| f.test.contains(d - 1)).collect())
        .collect()
}

fn pct(x: f64) -> String {
    format!("{:.1}%", 100.0 * x)
}

fn markdown(
    cfg: &ExperimentConfig,
    reports: &[ModelReport],
    comps: &[Comparison],
    vs_bh: &[(String, PairedBootstrap)],
    bh: &BacktestMetrics,
    common: &[i32],
    excluded: &[i32],
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# CV report: {}\n", cfg.name);
    let _ = writeln!(
        s,
        "Horizon {} days, label cost {} bp, margin m = {}, policy cost {} bp one way, window {}. Reference: `{}`.\n",
        cfg.labels.horizon, cfg.labels.kappa_bps, cfg.labels.margin_mult, cfg.cost_bps, cfg.window, cfg.reference
    );
    let _ = writeln!(
        s,
        "Test years in every table (finished by every reported model): {}.{}\n",
        common.iter().map(|y| y.to_string()).collect::<Vec<_>>().join(", "),
        if excluded.is_empty() {
            String::new()
        } else {
            format!(
                " Left out (not finished by every model): {}.",
                excluded.iter().map(|y| y.to_string()).collect::<Vec<_>>().join(", ")
            )
        }
    );
    let _ = writeln!(s, "## Prediction quality (test, calibrated)\n");
    let _ = writeln!(s, "| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |");
    let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|---|");
    for r in reports {
        let _ = writeln!(
            s,
            "| {} | {} | {:.3} | {:.3} | {:.4} → {:.4} | {:.4} | {:.4} | {} | {:.4} | {:.2} |",
            r.model,
            r.calibrated.n,
            r.calibrated.accuracy,
            r.calibrated.balanced_accuracy,
            r.raw.log_loss,
            r.calibrated.log_loss,
            r.calibrated.brier,
            r.calibrated.ece,
            r.calibrated.macro_auc.map_or("-".into(), |v| format!("{v:.4}")),
            r.ic.mean,
            r.ic.t_stat
        );
    }
    let _ = writeln!(s, "\n## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)\n");
    let _ = writeln!(s, "| model | none | temperature | vector | isotonic |");
    let _ = writeln!(s, "|---|---|---|---|---|");
    for r in reports {
        let cells: Vec<String> = r
            .calibration_methods
            .iter()
            .map(|(_, ll, b, e)| format!("{ll:.4} / {b:.4} / {e:.4}"))
            .collect();
        let _ = writeln!(s, "| {} | {} |", r.model, cells.join(" | "));
    }
    let _ = writeln!(
        s,
        "\n## Trading after costs ({} bp one way; primary policy: fixed tau = {} for every model)\n",
        cfg.cost_bps, cfg.policy_tau
    );
    let _ = writeln!(s, "| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |");
    let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|");
    for r in reports {
        let t = &r.trading;
        let _ = writeln!(
            s,
            "| {} | {} | {:.2} [{:.2}, {:.2}] | {:.2} | {} | {} | {:.3} | {} | {:.0}/{:.0}/{:.0}% |",
            r.model,
            pct(t.ann_return),
            t.sharpe,
            r.bootstrap.sharpe.lo,
            r.bootstrap.sharpe.hi,
            t.sortino,
            pct(t.max_drawdown),
            t.profit_factor.map_or("-".into(), |v| format!("{v:.3}")),
            t.mean_turnover,
            pct(t.total_net_return),
            100.0 * r.action_shares[Action::Buy.id()],
            100.0 * r.action_shares[Action::Sell.id()],
            100.0 * r.action_shares[Action::Hold.id()],
        );
    }
    let _ = writeln!(
        s,
        "| buy & hold (equal weight) | {} | {:.2} | {:.2} | {} | {} | {:.3} | {} | |",
        pct(bh.ann_return),
        bh.sharpe,
        bh.sortino,
        pct(bh.max_drawdown),
        bh.profit_factor.map_or("-".into(), |v| format!("{v:.3}")),
        bh.mean_turnover,
        pct(bh.total_net_return)
    );
    let _ = writeln!(s, "\n## Secondary policy: tau tuned per fold on the calibration segment\n");
    let _ = writeln!(s, "| model | ann. return | Sharpe | max DD | turnover/day | mean tau |");
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for r in reports {
        let t = &r.trading_tuned_tau;
        let _ = writeln!(
            s,
            "| {} | {} | {:.2} | {} | {:.3} | {:.2} |",
            r.model,
            pct(t.ann_return),
            t.sharpe,
            pct(t.max_drawdown),
            t.mean_turnover,
            r.mean_tau
        );
    }
    let _ = writeln!(s, "\n## Paired comparison with `{}` (stationary bootstrap, mean block 20 days, 2000 resamples)\n", cfg.reference);
    let _ = writeln!(s, "| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |");
    let _ = writeln!(s, "|---|---|---|---|---|");
    for c in comps {
        let b = &c.bootstrap;
        let _ = writeln!(
            s,
            "| {} | {:+.2} [{:+.2}, {:+.2}] | {:+.2} [{:+.2}, {:+.2}] | {}/{} | {}/{} |",
            c.model,
            b.sharpe_diff.estimate,
            b.sharpe_diff.lo,
            b.sharpe_diff.hi,
            1e4 * b.mean_diff.estimate,
            1e4 * b.mean_diff.lo,
            1e4 * b.mean_diff.hi,
            c.yearly_sharpe.wins_a,
            c.yearly_sharpe.years.len(),
            c.yearly_log_loss.wins_a,
            c.yearly_log_loss.years.len()
        );
    }
    if !vs_bh.is_empty() {
        let _ = writeln!(
            s,
            "\n## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)\n"
        );
        let _ = writeln!(s, "| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |");
        let _ = writeln!(s, "|---|---|---|");
        for (name, b) in vs_bh {
            let _ = writeln!(
                s,
                "| {} | {:+.2} [{:+.2}, {:+.2}] | {:+.2} [{:+.2}, {:+.2}] |",
                name,
                b.sharpe_diff.estimate,
                b.sharpe_diff.lo,
                b.sharpe_diff.hi,
                1e4 * b.mean_diff.estimate,
                1e4 * b.mean_diff.lo,
                1e4 * b.mean_diff.hi
            );
        }
    }
    let _ = writeln!(s, "\n## Cost sensitivity (Sharpe)\n");
    let _ = writeln!(s, "| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |");
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for r in reports {
        let cells: Vec<String> = r.costs.iter().map(|(_, m)| format!("{:.2}", m.sharpe)).collect();
        let _ = writeln!(s, "| {} | {} |", r.model, cells.join(" | "));
    }
    let _ = writeln!(s, "\n## Market-volatility regimes (Sharpe high / low)\n");
    let _ = writeln!(s, "| model | high vol | low vol |");
    let _ = writeln!(s, "|---|---|---|");
    for r in reports {
        if let Some((h, l)) = &r.regimes {
            let _ = writeln!(s, "| {} | {:.2} ({} d) | {:.2} ({} d) |", r.model, h.sharpe, h.n_days, l.sharpe, l.n_days);
        }
    }
    let _ = writeln!(s, "\n## Sharpe per test year\n");
    let years: Vec<i32> = reports.first().map(|r| r.yearly_sharpe.iter().map(|(y, _)| *y).collect()).unwrap_or_default();
    let _ = writeln!(s, "| model | {} |", years.iter().map(|y| y.to_string()).collect::<Vec<_>>().join(" | "));
    let _ = writeln!(s, "|---|{}", "---|".repeat(years.len()));
    for r in reports {
        let cells: Vec<String> = r.yearly_sharpe.iter().map(|(_, v)| format!("{v:.2}")).collect();
        let _ = writeln!(s, "| {} | {} |", r.model, cells.join(" | "));
    }
    let _ = writeln!(s, "\n## Cost of each model\n");
    let _ = writeln!(s, "| model | parameters | mean seconds per fold | training samples/s | peak device MB |");
    let _ = writeln!(s, "|---|---|---|---|---|");
    for r in reports {
        let _ = writeln!(
            s,
            "| {} | {} | {:.0} | {:.0} | {} |",
            r.model,
            r.num_params,
            r.mean_seconds,
            r.samples_per_second,
            if r.peak_device_mb > 0.0 { format!("{:.0}", r.peak_device_mb) } else { "-".into() }
        );
    }
    s
}

/// Keep `v[i]` where `keep[i]`.
fn retain_by<T>(v: &mut Vec<T>, keep: &[bool]) {
    let mut i = 0;
    v.retain(|_| {
        i += 1;
        keep[i - 1]
    });
}
