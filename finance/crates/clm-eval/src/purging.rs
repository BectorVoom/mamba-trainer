//! Splitting samples into a fold's segments and the run-time leakage guard (C4).
//!
//! [`split_samples`] applies the fold's keep rule. [`check_no_leakage`] re-verifies the
//! produced split from the contract's definitions alone (it does not trust
//! [`Fold::keep`]), so a bug in the fold builder or in a hand-made split is caught
//! before a model is fitted.

use clm_data::types::SampleIndex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::folds::{Fold, Segment, SegmentKind};

/// The kept samples of one fold, by segment.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldSamples {
    /// Kept training samples.
    pub train: Vec<SampleIndex>,
    /// Kept validation samples.
    pub val: Vec<SampleIndex>,
    /// Kept calibration samples.
    pub cal: Vec<SampleIndex>,
    /// Kept test samples (label-gated: for prediction-quality metrics only; the
    /// backtest trades every input-valid test day, see [`crate::backtest::run_backtest`]).
    pub test: Vec<SampleIndex>,
}

impl FoldSamples {
    /// The samples of a segment kind.
    pub fn get(&self, kind: SegmentKind) -> &[SampleIndex] {
        match kind {
            SegmentKind::Train => &self.train,
            SegmentKind::Val => &self.val,
            SegmentKind::Cal => &self.cal,
            SegmentKind::Test => &self.test,
        }
    }

    fn get_mut(&mut self, kind: SegmentKind) -> &mut Vec<SampleIndex> {
        match kind {
            SegmentKind::Train => &mut self.train,
            SegmentKind::Val => &mut self.val,
            SegmentKind::Cal => &mut self.cal,
            SegmentKind::Test => &mut self.test,
        }
    }
}

/// Assign (data-valid) samples to the fold's segments, keeping only those that pass
/// the purge/embargo rule ([`Fold::is_kept`]). Samples outside every segment or
/// purged are dropped; the input order is preserved within each segment.
pub fn split_samples(fold: &Fold, samples: &[SampleIndex]) -> FoldSamples {
    let mut out = FoldSamples::default();
    for &si in samples {
        let t = si.day();
        if let Some(kind) = fold.segment_of(t)
            && fold.kept(kind).contains(t)
        {
            out.get_mut(kind).push(si);
        }
    }
    out
}

/// A violation of C4 properties 1–3 found by [`check_no_leakage`] or
/// [`check_folds_disjoint`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LeakageError {
    /// A segment has `end < start` or ends past `N`.
    #[error("segment {kind:?} [{}, {}) is malformed or not inside [0, {n_days})", seg.start, seg.end)]
    SegmentOutOfRange {
        /// Which segment.
        kind: SegmentKind,
        /// Its range.
        seg: Segment,
        /// Calendar length.
        n_days: usize,
    },
    /// Two segments overlap or are out of order.
    #[error("segment {first:?} [{}, {}) does not end before {second:?} [{}, {}) starts", a.start, a.end, b.start, b.end)]
    SegmentOrder {
        /// The earlier kind.
        first: SegmentKind,
        /// Its range.
        a: Segment,
        /// The later kind.
        second: SegmentKind,
        /// Its range.
        b: Segment,
    },
    /// A sample listed under a segment does not lie in it.
    #[error("sample (s={}, t={}) listed in {kind:?} lies outside [{}, {})", sample.s, sample.t, seg.start, seg.end)]
    SampleOutsideSegment {
        /// The segment it was listed under.
        kind: SegmentKind,
        /// The segment's range.
        seg: Segment,
        /// The sample.
        sample: SampleIndex,
    },
    /// A kept sample's label (or its embargo) reaches into a later segment.
    #[error(
        "sample (s={}, t={}) of {kind:?}: label end {label_end} + embargo {embargo} is not < start {later_start} of later segment {later:?}",
        sample.s, sample.t
    )]
    LabelOverlap {
        /// The sample's segment.
        kind: SegmentKind,
        /// The sample.
        sample: SampleIndex,
        /// `t + 1 + h`.
        label_end: usize,
        /// The embargo used.
        embargo: usize,
        /// The later segment it reaches.
        later: SegmentKind,
        /// That segment's start.
        later_start: usize,
    },
    /// A test sample's label reads a day past `N - 1`.
    #[error("test sample (s={}, t={}): label end {label_end} > N - 1 = {}", sample.s, sample.t, n_days.saturating_sub(1))]
    TestLabelPastEnd {
        /// The sample.
        sample: SampleIndex,
        /// `t + 1 + h`.
        label_end: usize,
        /// Calendar length.
        n_days: usize,
    },
    /// Test segments of two folds overlap or are not increasing.
    #[error("test segments of folds {first} [{}, {}) and {second} [{}, {}) overlap or are out of order", a.start, a.end, b.start, b.end)]
    TestSegmentsOverlap {
        /// Index of the earlier fold.
        first: usize,
        /// Its test segment.
        a: Segment,
        /// Index of the later fold.
        second: usize,
        /// Its test segment.
        b: Segment,
    },
}

/// Re-verify C4 properties 1–3 on a produced split, from the definitions:
///
/// 1. the four segments are well-formed, inside `[0, N)` and ordered without overlap;
///    every listed sample lies in the segment it is listed under;
/// 2. for every sample of a segment and **every** later segment `S` (empty ones
///    included), every label index `t+1 ..= t+1+h` is `< S.start - e`, i.e.
///    `t + 1 + h + e < S.start`;
/// 3. every test label lies inside the data: `t + 1 + h <= N - 1`.
///
/// `horizon` and `embargo` are passed separately from the fold on purpose: the guard
/// checks against the run's configured values, not whatever built the fold.
pub fn check_no_leakage(
    fold: &Fold,
    samples: &FoldSamples,
    horizon: usize,
    embargo: usize,
) -> Result<(), LeakageError> {
    let n = fold.n_days;
    let segs = fold.segments();
    for (k, &seg) in SegmentKind::ALL.iter().zip(segs.iter()) {
        if seg.end < seg.start || seg.end > n {
            return Err(LeakageError::SegmentOutOfRange {
                kind: *k,
                seg,
                n_days: n,
            });
        }
    }
    for i in 0..4 {
        for j in i + 1..4 {
            if segs[i].end > segs[j].start {
                return Err(LeakageError::SegmentOrder {
                    first: SegmentKind::ALL[i],
                    a: segs[i],
                    second: SegmentKind::ALL[j],
                    b: segs[j],
                });
            }
        }
    }
    for (i, &kind) in SegmentKind::ALL.iter().enumerate() {
        let seg = segs[i];
        for &sample in samples.get(kind) {
            let t = sample.day();
            if !seg.contains(t) {
                return Err(LeakageError::SampleOutsideSegment { kind, seg, sample });
            }
            // Saturating: an overflowing label end is certainly past every segment.
            let label_end = t.saturating_add(1).saturating_add(horizon);
            for (j, &later) in segs.iter().enumerate().skip(i + 1) {
                if label_end.saturating_add(embargo) >= later.start {
                    return Err(LeakageError::LabelOverlap {
                        kind,
                        sample,
                        label_end,
                        embargo,
                        later: SegmentKind::ALL[j],
                        later_start: later.start,
                    });
                }
            }
            if kind == SegmentKind::Test && (n == 0 || label_end > n - 1) {
                return Err(LeakageError::TestLabelPastEnd {
                    sample,
                    label_end,
                    n_days: n,
                });
            }
        }
    }
    Ok(())
}

/// C4 property 3: test segments of different folds are pairwise disjoint and
/// increasing (in the given order).
pub fn check_folds_disjoint(folds: &[Fold]) -> Result<(), LeakageError> {
    for (i, a) in folds.iter().enumerate() {
        for (j, b) in folds.iter().enumerate().skip(i + 1) {
            if a.test.end > b.test.start || a.test.start >= b.test.start {
                return Err(LeakageError::TestSegmentsOverlap {
                    first: i,
                    a: a.test,
                    second: j,
                    b: b.test,
                });
            }
        }
    }
    Ok(())
}
