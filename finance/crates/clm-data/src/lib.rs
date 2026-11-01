//! Bars, point-in-time features, labels, normalisation and model windows for Mamba-CLM.
//!
//! The rules are in `finance/docs/CONTRACTS.md` (C1–C3, C4 normalisation); this crate
//! implements them:
//!
//! * [`bars`]: Parquet bars, the day-aligned [`Panel`] on the market's calendar, C1
//!   price adjustment, load validation ([`LoadReport`]).
//! * [`features`]: feature set v1 (C2) → [`FeatureTensor`] `[s][t][f]`.
//! * [`labels`]: C3 labels → [`Labels`].
//! * [`normalization`]: per-fold [`Normalizer`] (C4).
//! * [`windows`]: valid samples and `[B][W][F]` input windows.
//! * [`audit`]: truncation/perturbation look-ahead audits and the label audit.
//! * [`synthetic`]: seeded synthetic panels with an optional planted signal.
//! * [`export`]: Parquet sample table for external baselines and their predictions.
//!
//! Known deviation from the letter of C2 (conservative reading): `amihud` divides by the
//! traded value computed with the *reported* close, not the adjusted close, because the
//! adjusted level depends on future dividends (see [`features`]).

pub mod audit;
pub mod bars;
pub mod error;
pub mod export;
pub mod features;
pub mod labels;
pub mod normalization;
pub mod synthetic;
pub mod types;
pub mod windows;

pub use audit::{Mismatch, label_audit, perturbation_audit, truncation_audit};
pub use bars::{Bars, LoadReport, Panel, Series, Universe, load_parquet, write_parquet};
pub use error::{DataError, Result};
pub use export::{Predictions, read_predictions, write_predictions, write_samples};
pub use features::{FEATURE_NAMES, FeatureTensor, LOOKBACK, NUM_FEATURES, compute_features};
pub use labels::{LabelConfig, Labels, classify, compute_labels};
pub use normalization::Normalizer;
pub use synthetic::{PlantedSignal, SyntheticConfig, SyntheticPanel};
pub use types::{Action, DayIndex, NUM_ACTIONS, SampleIndex};
pub use windows::{gather_last, gather_windows, valid_samples};
