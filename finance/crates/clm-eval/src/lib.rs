//! Leakage-free evaluation for Mamba-CLM (docs/CONTRACTS.md C4–C6).
//!
//! * [`folds`] — walk-forward folds (expanding / rolling) with purging and embargo (C4);
//! * [`purging`] — splitting samples into a fold and the run-time leakage guard;
//! * [`metrics`] — classification, calibration (ECE), rank-IC and regression metrics;
//! * [`calibration`] — temperature, vector (multiclass Platt) and isotonic calibrators;
//! * [`backtest`] — the long-only policy and cost accounting (C5) and its metrics;
//! * [`statistics`] — stationary bootstrap, paired per-year comparison, cost and
//!   regime analyses (C6);
//! * [`oof`] — out-of-fold prediction records (Parquet) and fold metadata (JSON).
//!
//! Everything fitted on a fold (calibrators, thresholds, regime medians) must be fitted
//! on that fold's kept `train`/`val`/`cal` samples only (C4 property 4); the functions
//! here take exactly the data they are given, so that discipline lies with the caller,
//! guarded by [`purging::check_no_leakage`].

pub mod backtest;
pub mod calibration;
pub mod folds;
pub mod metrics;
pub mod oof;
pub mod purging;
pub mod statistics;

pub use backtest::{
    BacktestError, BacktestMetrics, BacktestResult, DailySeries, SecurityLedger, backtest_metrics,
    buy_and_hold, policy, render_backtest, render_backtest_command, run_backtest,
};
pub use calibration::{CalibrationMethod, Calibrator, IsotonicStep};
pub use folds::{
    Fold, FoldConfig, Scheme, Segment, SegmentKind, make_folds, render_folds, render_folds_command,
    render_folds_line,
};
pub use metrics::{
    ClassificationMetrics, IcSummary, RegressionMetrics, ReliabilityBin, classification_metrics,
    rank_ic, regression_metrics,
};
pub use oof::{FoldMetadata, OofError, OofRecord, read_oof_parquet, write_oof_parquet};
pub use purging::{
    FoldSamples, LeakageError, check_folds_disjoint, check_no_leakage, split_samples,
};
pub use statistics::{
    BootstrapConfig, Interval, PairedBootstrap, StatsError, YearComparison, bootstrap_series,
    paired_bootstrap, paired_bootstrap_series,
};
