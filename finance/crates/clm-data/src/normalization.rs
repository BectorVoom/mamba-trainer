//! Per-feature normalisation (CONTRACTS C4, "Normalisation").
//!
//! `mu, sd` per feature over the feature rows of the days in the fit range (days, not
//! samples: every valid row of every stock whose day is in the range); population `sd`,
//! floored at `1e-6`; accumulation in `f64` (two passes). Rows map to
//! `clamp((x - mu) / sd, -8, 8)`.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::error::{DataError, Result};
use crate::features::FeatureTensor;

/// Floor of the per-feature standard deviation.
pub const SD_FLOOR: f64 = 1e-6;
/// Normalised values are clamped to `[-CLAMP, CLAMP]`.
pub const CLAMP: f64 = 8.0;

/// Fitted per-feature mean and standard deviation (stored with a checkpoint).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Normalizer {
    /// Per-feature mean.
    pub mean: Vec<f64>,
    /// Per-feature standard deviation (floored).
    pub sd: Vec<f64>,
    /// The day range the statistics were fitted on.
    pub fit_days: Range<usize>,
    /// Number of rows used.
    pub rows: u64,
}

impl Normalizer {
    /// Fits on every valid row `(s, t)` with `t` in `days`. Errors if there is none.
    pub fn fit(features: &FeatureTensor, days: Range<usize>) -> Result<Self> {
        let f = features.num_features;
        let end = days.end.min(features.num_days);
        let rows = || {
            (0..features.num_securities).flat_map(move |s| {
                (days.start..end)
                    .filter(move |&t| features.is_valid(s, t))
                    .map(move |t| features.row(s, t))
            })
        };
        let mut sum = vec![0.0f64; f];
        let mut n = 0u64;
        for row in rows() {
            for (a, &x) in sum.iter_mut().zip(row) {
                *a += x as f64;
            }
            n += 1;
        }
        if n == 0 {
            return Err(DataError::Invalid(format!(
                "no valid feature rows in days {days:?}"
            )));
        }
        let mean: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();
        let mut ss = vec![0.0f64; f];
        for row in rows() {
            for ((a, &x), m) in ss.iter_mut().zip(row).zip(&mean) {
                let d = x as f64 - m;
                *a += d * d;
            }
        }
        let sd = ss
            .iter()
            .map(|s| (s / n as f64).sqrt().max(SD_FLOOR))
            .collect();
        Ok(Self {
            mean,
            sd,
            fit_days: days,
            rows: n,
        })
    }

    /// Normalises one value of feature `f`.
    #[inline]
    pub fn transform(&self, f: usize, x: f32) -> f32 {
        ((x as f64 - self.mean[f]) / self.sd[f]).clamp(-CLAMP, CLAMP) as f32
    }

    /// Normalises one row in place.
    pub fn transform_row(&self, row: &mut [f32]) {
        debug_assert_eq!(row.len(), self.mean.len());
        for (f, x) in row.iter_mut().enumerate() {
            *x = self.transform(f, *x);
        }
    }

    /// Normalised copy of the tensor; invalid rows stay invalid (NaN).
    pub fn apply(&self, features: &FeatureTensor) -> FeatureTensor {
        assert_eq!(features.num_features, self.mean.len(), "feature count");
        let mut out = features.clone();
        for s in 0..out.num_securities {
            for t in 0..out.num_days {
                if out.is_valid(s, t) {
                    self.transform_row(out.row_mut(s, t));
                }
            }
        }
        out
    }
}
