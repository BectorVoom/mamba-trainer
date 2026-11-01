//! Walk-forward folds with purging and embargo (CONTRACTS C4).
//!
//! A fold has four contiguous, ordered segments `train < val < cal < test`. Within the
//! first three, a decision day `t` is *kept* iff `t + 1 + h + e < s_next` (the start of
//! the next segment); in `test` iff `t + 1 + h <= N - 1`. Because every segment ends
//! where the next starts, the kept days of each segment form a contiguous prefix,
//! stored in [`Fold::keep`].

use serde::{Deserialize, Serialize};

/// How the training window grows from fold to fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scheme {
    /// `train = [0, val.start)`.
    Expanding,
    /// `train = [max(0, val.start - train_days), val.start)`.
    Rolling {
        /// Maximum length of the training segment in days.
        train_days: usize,
    },
}

/// Walk-forward configuration (C4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldConfig {
    /// Expanding or rolling training window.
    pub scheme: Scheme,
    /// First test year (inclusive).
    pub first_test_year: i32,
    /// Last test year (inclusive).
    pub last_test_year: i32,
    /// Length of the validation segment (early stopping) in days.
    pub val_days: usize,
    /// Length of the calibration segment in days.
    pub cal_days: usize,
    /// Label horizon `h >= 1` in days.
    pub horizon: usize,
    /// Embargo `e >= 0` in days.
    pub embargo: usize,
    /// Folds whose training segment is shorter than this are skipped.
    pub min_train_days: usize,
}

/// A half-open day range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Segment {
    /// First day (inclusive).
    pub start: usize,
    /// One past the last day.
    pub end: usize,
}

impl Segment {
    /// The range `[start, end)`.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Number of days (0 if `end <= start`).
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// True when the segment holds no day.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True when `start <= t < end`.
    pub fn contains(&self, t: usize) -> bool {
        self.start <= t && t < self.end
    }

    /// The days of the segment.
    pub fn days(&self) -> std::ops::Range<usize> {
        self.start..self.end.max(self.start)
    }
}

/// Which segment of a fold a day belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentKind {
    /// Model fitting.
    Train = 0,
    /// Early stopping / model selection.
    Val = 1,
    /// Probability calibration and thresholds.
    Cal = 2,
    /// Out-of-sample evaluation.
    Test = 3,
}

impl SegmentKind {
    /// All kinds in segment order.
    pub const ALL: [SegmentKind; 4] = [
        SegmentKind::Train,
        SegmentKind::Val,
        SegmentKind::Cal,
        SegmentKind::Test,
    ];

    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            SegmentKind::Train => "train",
            SegmentKind::Val => "val",
            SegmentKind::Cal => "cal",
            SegmentKind::Test => "test",
        }
    }
}

/// One walk-forward fold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fold {
    /// The calendar year of the test segment.
    pub test_year: i32,
    /// Number of days `N` of the calendar the fold was cut from.
    pub n_days: usize,
    /// Training segment.
    pub train: Segment,
    /// Validation segment.
    pub val: Segment,
    /// Calibration segment.
    pub cal: Segment,
    /// Test segment (all days of `test_year`).
    pub test: Segment,
    /// Kept decision days of `[train, val, cal, test]` (each a prefix of its segment).
    /// The test entry is label-gated (`t + 1 + h <= N - 1`): it selects samples for
    /// prediction-quality metrics, never the days the backtest trades (C5).
    pub keep: [Segment; 4],
}

impl Fold {
    /// The segments in order `[train, val, cal, test]`.
    pub fn segments(&self) -> [Segment; 4] {
        [self.train, self.val, self.cal, self.test]
    }

    /// The segment of a kind.
    pub fn segment(&self, kind: SegmentKind) -> Segment {
        self.segments()[kind as usize]
    }

    /// The kept days of a kind.
    pub fn kept(&self, kind: SegmentKind) -> Segment {
        self.keep[kind as usize]
    }

    /// The segment containing day `t`, if any.
    pub fn segment_of(&self, t: usize) -> Option<SegmentKind> {
        SegmentKind::ALL
            .into_iter()
            .find(|&k| self.segment(k).contains(t))
    }

    /// True when `t` is a kept decision day of some segment.
    pub fn is_kept(&self, t: usize) -> bool {
        self.segment_of(t).is_some_and(|k| self.kept(k).contains(t))
    }
}

/// Kept prefix of a non-test segment `[start, end)` followed by a segment starting at
/// `end`: `t + 1 + h + e < end`, i.e. `t < end - (1 + h + e)`.
fn keep_before_next(seg: Segment, horizon: usize, embargo: usize) -> Segment {
    let gap = 1usize.saturating_add(horizon).saturating_add(embargo);
    let last_excl = seg.end.saturating_sub(gap);
    Segment::new(
        seg.start,
        last_excl.clamp(seg.start, seg.end.max(seg.start)),
    )
}

/// Kept prefix of the test segment: `t + 1 + h <= N - 1`, i.e. `t < N - 1 - h`.
fn keep_test(seg: Segment, n_days: usize, horizon: usize) -> Segment {
    let last_excl = n_days.saturating_sub(1usize.saturating_add(horizon));
    Segment::new(
        seg.start,
        last_excl.clamp(seg.start, seg.end.max(seg.start)),
    )
}

/// Cut walk-forward folds from a calendar (C4).
///
/// `years[t]` is the calendar year of day `t`; it must be non-decreasing and have
/// `n_days` entries. Folds for years missing from the calendar, with
/// `ts < cal_days + val_days`, or with a training segment shorter than
/// `min_train_days` are skipped.
///
/// # Panics
/// If `years.len() != n_days` or `years` decreases somewhere.
pub fn make_folds(years: &[i32], n_days: usize, cfg: &FoldConfig) -> Vec<Fold> {
    assert_eq!(
        years.len(),
        n_days,
        "make_folds: years.len() must equal n_days"
    );
    assert!(
        years.windows(2).all(|w| w[0] <= w[1]),
        "make_folds: years must be non-decreasing"
    );
    let mut folds = Vec::new();
    if cfg.first_test_year > cfg.last_test_year {
        return folds;
    }
    for y in cfg.first_test_year..=cfg.last_test_year {
        let Some(ts) = years.iter().position(|&v| v == y) else {
            continue;
        };
        let te = years.iter().position(|&v| v > y).unwrap_or(n_days);
        let Some(pre) = cfg.cal_days.checked_add(cfg.val_days) else {
            continue;
        };
        if ts < pre {
            continue;
        }
        let cal = Segment::new(ts - cfg.cal_days, ts);
        let val = Segment::new(cal.start - cfg.val_days, cal.start);
        let train = match cfg.scheme {
            Scheme::Expanding => Segment::new(0, val.start),
            Scheme::Rolling { train_days } => {
                Segment::new(val.start.saturating_sub(train_days), val.start)
            }
        };
        if train.len() < cfg.min_train_days {
            continue;
        }
        let test = Segment::new(ts, te);
        let keep = [
            keep_before_next(train, cfg.horizon, cfg.embargo),
            keep_before_next(val, cfg.horizon, cfg.embargo),
            keep_before_next(cal, cfg.horizon, cfg.embargo),
            keep_test(test, n_days, cfg.horizon),
        ];
        folds.push(Fold {
            test_year: y,
            n_days,
            train,
            val,
            cal,
            test,
            keep,
        });
    }
    folds
}

/// The `folds` command of the Lean `clmcheck` oracle for this calendar and config
/// (years run-length encoded).
///
/// # Panics
/// If `years.len() != n_days`.
pub fn render_folds_command(years: &[i32], n_days: usize, cfg: &FoldConfig) -> String {
    assert_eq!(years.len(), n_days);
    let (scheme, train_days) = match cfg.scheme {
        Scheme::Expanding => (0, 0),
        Scheme::Rolling { train_days } => (1, train_days),
    };
    let mut runs: Vec<(i32, usize)> = Vec::new();
    for &y in years {
        match runs.last_mut() {
            Some((ly, c)) if *ly == y => *c += 1,
            _ => runs.push((y, 1)),
        }
    }
    let mut s = format!(
        "folds {} {} {} {} {} {} {} {} {} {} {}",
        n_days,
        scheme,
        train_days,
        cfg.first_test_year,
        cfg.last_test_year,
        cfg.val_days,
        cfg.cal_days,
        cfg.horizon,
        cfg.embargo,
        cfg.min_train_days,
        runs.len()
    );
    for (y, c) in runs {
        s.push_str(&format!(" {y} {c}"));
    }
    s
}

/// One output line of the Lean oracle's `folds` command:
/// `fold Y train a b val a b cal a b test a b keep ta tb va vb ca cb xa xb`.
pub fn render_folds_line(fold: &Fold) -> String {
    let [tr, va, ca, te] = fold.segments();
    let [kt, kv, kc, kx] = fold.keep;
    format!(
        "fold {} train {} {} val {} {} cal {} {} test {} {} keep {} {} {} {} {} {} {} {}",
        fold.test_year,
        tr.start,
        tr.end,
        va.start,
        va.end,
        ca.start,
        ca.end,
        te.start,
        te.end,
        kt.start,
        kt.end,
        kv.start,
        kv.end,
        kc.start,
        kc.end,
        kx.start,
        kx.end
    )
}

/// All output lines of a `folds` command (one per emitted fold, without the final `end`).
pub fn render_folds(folds: &[Fold]) -> Vec<String> {
    folds.iter().map(render_folds_line).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(scheme: Scheme) -> FoldConfig {
        FoldConfig {
            scheme,
            first_test_year: 2001,
            last_test_year: 2005,
            val_days: 2,
            cal_days: 1,
            horizon: 1,
            embargo: 1,
            min_train_days: 5,
        }
    }

    #[test]
    fn keep_helpers() {
        // [0, 10), h=1, e=1: t + 3 < 10 -> t <= 6.
        assert_eq!(
            keep_before_next(Segment::new(0, 10), 1, 1),
            Segment::new(0, 7)
        );
        // too short: empty prefix at start.
        assert_eq!(
            keep_before_next(Segment::new(5, 7), 1, 1),
            Segment::new(5, 5)
        );
        assert_eq!(
            keep_before_next(Segment::new(0, 0), 3, 0),
            Segment::new(0, 0)
        );
        // test: t + 2 <= 9 -> t <= 7.
        assert_eq!(keep_test(Segment::new(5, 10), 10, 1), Segment::new(5, 8));
        assert_eq!(keep_test(Segment::new(0, 2), 2, 5), Segment::new(0, 0));
    }

    #[test]
    fn segment_of_and_is_kept() {
        let years: Vec<i32> = (0..60).map(|t| 2000 + t / 10).collect();
        let folds = make_folds(&years, 60, &cfg(Scheme::Expanding));
        let f = &folds[0];
        assert_eq!(f.segment_of(0), Some(SegmentKind::Train));
        assert_eq!(f.segment_of(7), Some(SegmentKind::Val));
        assert_eq!(f.segment_of(9), Some(SegmentKind::Cal));
        assert_eq!(f.segment_of(10), Some(SegmentKind::Test));
        assert_eq!(f.segment_of(20), None);
        assert!(f.is_kept(3) && !f.is_kept(4));
        assert!(!f.is_kept(20));
    }
}
