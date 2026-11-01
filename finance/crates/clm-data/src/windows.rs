//! Model input windows (CONTRACTS C2): sample `(s, t)` uses feature rows `t-W+1 ..= t`.

use crate::features::FeatureTensor;
use crate::labels::Labels;
use crate::types::SampleIndex;

/// Every sample whose input window `t-W+1 ..= t` is all valid and whose label exists,
/// ordered by security, then day.
///
/// # Panics
/// If `w == 0` or the tensors have different shapes.
pub fn valid_samples(features: &FeatureTensor, labels: &Labels, w: usize) -> Vec<SampleIndex> {
    assert!(w >= 1, "window length must be >= 1");
    assert_eq!(
        (features.num_securities, features.num_days),
        (labels.num_securities, labels.num_days),
        "features and labels shapes differ"
    );
    let mut out = Vec::new();
    for s in 0..features.num_securities {
        let mut run = 0usize; // consecutive valid rows ending at t
        for t in 0..features.num_days {
            run = if features.is_valid(s, t) { run + 1 } else { 0 };
            if run >= w && labels.exists(s, t) {
                out.push(SampleIndex::new(s, t));
            }
        }
    }
    out
}

/// Gathers the windows of `samples` as `[B][W][F]` row-major (row `w-1` is day `t`).
///
/// # Panics
/// If a sample has `t < W-1`, is out of range, or its window holds an invalid row.
pub fn gather_windows(features: &FeatureTensor, samples: &[SampleIndex], w: usize) -> Vec<f32> {
    assert!(w >= 1, "window length must be >= 1");
    let f = features.num_features;
    let mut out = Vec::with_capacity(samples.len() * w * f);
    for smp in samples {
        let (s, t) = (smp.security(), smp.day());
        assert!(t + 1 >= w, "sample {smp:?}: window starts before day 0");
        for k in t + 1 - w..=t {
            assert!(features.is_valid(s, k), "sample {smp:?}: invalid row {k}");
            out.extend_from_slice(features.row(s, k));
        }
    }
    out
}

/// The last row (day `t`) of each sample, `[B][F]` (for non-sequential baselines).
///
/// # Panics
/// If a sample's row is invalid or out of range.
pub fn gather_last(features: &FeatureTensor, samples: &[SampleIndex]) -> Vec<f32> {
    gather_windows(features, samples, 1)
}
