//! Ranked-shortlist recovery metrics for molecular completion (host only).
//!
//! [`score_query`] ranks a [`QueryOutcome`](super::completion_model::QueryOutcome)'s
//! shortlist against the query's target molecule: `rank` is the 1-based
//! position of the target under strict typed-graph identity
//! ([`same_identity`](super::completion_data::same_identity); an unresolved
//! comparison is not a hit), `skeleton_rank` the same under the
//! Kekulé-insensitive [`skeleton`](super::completion_data::skeleton).
//! The skeleton merges distinct bond-order isomers, so it is a relaxed
//! identity and never the headline metric. [`recovery_report`] aggregates
//! per-query scores into top-1/10/25 and skeleton-top-25 hit rates with
//! percentile bootstrap intervals over identity groups.
//!
//! Denominators are all scored queries: a query with no candidates is a
//! miss, never removed.
//!
//! The bootstrap resamples identity groups (the examples'
//! `identity_group`), keeping all queries of a group together, with the
//! shared [`SplitMix64`](super::completion_data::SplitMix64) generator (its
//! unbiased `below`). [`super::metrics`] keeps its own private bootstrap
//! helper and this task must not modify that module, so the few lines are
//! carried here rather than shared.

use serde::{Deserialize, Serialize};

use super::completion_data::{SplitMix64, same_identity, skeleton};
use super::completion_model::QueryOutcome;
use super::graph::MolGraph;

/// The count fields of a [`QueryOutcome`](super::completion_model::QueryOutcome),
/// carried into [`QueryScore`] so reports need no device handle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeCounts {
    /// Shortlist length actually returned.
    pub candidates: u32,
    /// Accepted identities before the `returned` cut.
    pub distinct: u32,
    /// Trajectories sampled (`K`).
    pub trajectories: u32,
    /// Device status FINISHED.
    pub finished: u32,
    /// Device status `no_valid_action`.
    pub dead_end: u32,
    /// Device status TRUNCATED.
    pub truncated: u32,
    /// FINISHED on the device but the host exact replay did not end stopped
    /// and complete.
    pub rejected_replay: u32,
    /// Complete but a required pattern is not contained.
    pub rejected_containment: u32,
    /// A containment check hit its node limit.
    pub containment_unresolved: u32,
    /// An identity comparison hit its work limit.
    pub identity_unresolved: u32,
    /// Trajectories with none of the three terminal status bits.
    pub other_status: u32,
}

impl From<&QueryOutcome> for OutcomeCounts {
    /// Copy the accounting fields of an outcome.
    fn from(outcome: &QueryOutcome) -> Self {
        Self {
            candidates: outcome.candidates.len() as u32,
            distinct: outcome.distinct,
            trajectories: outcome.trajectories,
            finished: outcome.finished,
            dead_end: outcome.dead_end,
            truncated: outcome.truncated,
            rejected_replay: outcome.rejected_replay,
            rejected_containment: outcome.rejected_containment,
            containment_unresolved: outcome.containment_unresolved,
            identity_unresolved: outcome.identity_unresolved,
            other_status: outcome.other_status,
        }
    }
}

/// One query's score: where the target lands in the returned shortlist.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryScore {
    /// 1-based rank of the target among the returned candidates under
    /// strict typed-graph identity ([`same_identity`](super::completion_data::same_identity));
    /// `None` when absent (an unresolved comparison is not a hit).
    pub rank: Option<u32>,
    /// 1-based rank under the Kekulé-insensitive
    /// [`skeleton`](super::completion_data::skeleton) identity; `None` when
    /// absent. Relaxed (it merges bond-order isomers), never the headline.
    pub skeleton_rank: Option<u32>,
    /// The outcome's accounting, for the report's denominators and fractions.
    pub outcome_counts: OutcomeCounts,
}

/// Rank `outcome`'s candidates against `target`.
///
/// The first candidate with `same_identity(candidate, target, work_limit) ==
/// Some(true)` decides `rank` (1-based); `None` comparisons are skipped, so
/// an unresolved comparison is never a hit. `skeleton_rank` repeats the scan
/// on [`skeleton`](super::completion_data::skeleton) graphs: a skeleton that
/// fails to build (or a target whose skeleton fails) simply never matches.
/// Both scans are over the returned shortlist only.
pub fn score_query(outcome: &QueryOutcome, target: &MolGraph, work_limit: usize) -> QueryScore {
    let mut rank = None;
    for (i, candidate) in outcome.candidates.iter().enumerate() {
        if same_identity(&candidate.graph, target, work_limit) == Some(true) {
            rank = Some(i as u32 + 1);
            break;
        }
    }
    let skeleton_rank = match skeleton(target) {
        Ok(target_skeleton) => {
            let mut found = None;
            for (i, candidate) in outcome.candidates.iter().enumerate() {
                let Ok(candidate_skeleton) = skeleton(&candidate.graph) else {
                    continue;
                };
                if same_identity(&candidate_skeleton, &target_skeleton, work_limit) == Some(true) {
                    found = Some(i as u32 + 1);
                    break;
                }
            }
            found
        }
        Err(_) => None,
    };
    QueryScore {
        rank,
        skeleton_rank,
        outcome_counts: OutcomeCounts::from(outcome),
    }
}

/// One hit rate with its 95% percentile bootstrap interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HitRate {
    /// Queries scoring a hit.
    pub hits: usize,
    /// All scored queries (an empty shortlist is a miss, never removed).
    pub queries: usize,
    /// `hits / queries` (0 when there are no queries).
    pub rate: f64,
    /// 2.5th percentile of the group-bootstrap means.
    pub lo: f64,
    /// 97.5th percentile of the group-bootstrap means.
    pub hi: f64,
}

/// Top-1/10/25 recovery over scored queries with group bootstrap intervals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecoveryReport {
    /// Scored queries (the denominator of every hit rate).
    pub queries: usize,
    /// Target ranked first.
    pub top1: HitRate,
    /// Target within the first 10.
    pub top10: HitRate,
    /// Target anywhere in the shortlist.
    pub top25: HitRate,
    /// Target's skeleton anywhere in the shortlist (relaxed, never the
    /// headline).
    pub skeleton_top25: HitRate,
    /// Mean accepted identities per query (before the `returned` cut).
    pub mean_distinct: f64,
    /// Mean unresolved trajectories per query (the `unresolved` shortlist of
    /// [`QueryOutcome`](super::completion_model::QueryOutcome), which is
    /// never a hit).
    pub mean_unresolved: f64,
    /// Mean FINISHED fraction of trajectories.
    pub mean_finished_fraction: f64,
    /// Mean `no_valid_action` fraction of trajectories.
    pub mean_dead_end_fraction: f64,
    /// Mean TRUNCATED fraction of trajectories.
    pub mean_truncated_fraction: f64,
    /// Mean rejected fraction of trajectories
    /// (`rejected_replay + rejected_containment + containment_unresolved`
    /// over trajectories; identity-unresolved rows are still accepted, so
    /// they are not rejected).
    pub mean_rejected_fraction: f64,
    /// Queries whose shortlist is empty.
    pub zero_candidate_queries: usize,
}

/// The `p`th percentile of a sorted slice by linear interpolation.
fn percentile_of(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (sorted.len() - 1) as f64 * p / 100.0;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }
}

/// Aggregate per-query scores into [`RecoveryReport`].
///
/// `groups[i]` is the identity group of `scores[i]` (the examples'
/// `identity_group`); the two slices must have equal lengths. Hit rates are
/// means over all scored queries: top-1 needs `rank == 1`, top-10
/// `rank <= 10`, top-25 `rank <= 25` (a rank past the shortlist cut is a
/// miss), and skeleton-top-25 `skeleton_rank <= 25`. Intervals resample
/// *groups* with replacement `bootstrap` times (seeded by `seed`), keeping
/// all queries of a group together: each replicate draws `n_groups` groups
/// by rejection sampling (no modulo bias), pools their queries (a group
/// drawn twice counts twice), and records the four hit rates; `lo`/`hi` are
/// the 2.5th/97.5th percentiles. With one group every replicate pools the
/// same queries, so each interval collapses to its point estimate.
/// `bootstrap == 0` reports the point as both bounds. The whole computation
/// is a pure function of its inputs: the same seed gives the same report bit
/// for bit. `mean_unresolved` is the mean per-query count of trajectories in
/// the outcome's `unresolved` shortlist (never hits, reported next to the
/// other diagnostics).
pub fn recovery_report(
    scores: &[QueryScore],
    groups: &[u64],
    bootstrap: usize,
    seed: u64,
) -> RecoveryReport {
    assert_eq!(
        scores.len(),
        groups.len(),
        "recovery_report: {} scores for {} groups",
        scores.len(),
        groups.len()
    );
    let queries = scores.len();
    let hit = |score: &QueryScore, which: u8| -> bool {
        match which {
            0 => score.rank == Some(1),
            1 => score.rank.is_some_and(|r| r <= 10),
            2 => score.rank.is_some_and(|r| r <= 25),
            _ => score.skeleton_rank.is_some_and(|r| r <= 25),
        }
    };
    let rate_of = |which: u8, member: &[usize]| -> f64 {
        if member.is_empty() {
            return 0.0;
        }
        member.iter().filter(|i| hit(&scores[**i], which)).count() as f64 / member.len() as f64
    };
    let all: Vec<usize> = (0..queries).collect();
    let point = [
        rate_of(0, &all),
        rate_of(1, &all),
        rate_of(2, &all),
        rate_of(3, &all),
    ];
    // Unique groups in first-appearance order with their member queries.
    let mut order: Vec<u64> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        match order.iter().position(|g| g == group) {
            Some(pos) => members[pos].push(i),
            None => {
                order.push(*group);
                members.push(vec![i]);
            }
        }
    }
    let mut replicate_rates: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    if !order.is_empty() && bootstrap > 0 {
        let mut rng = SplitMix64::new(seed);
        let n_groups = order.len();
        for _ in 0..bootstrap {
            let mut pooled: Vec<usize> = Vec::new();
            for _ in 0..n_groups {
                let draw = rng.below(n_groups as u64) as usize;
                pooled.extend_from_slice(&members[draw]);
            }
            for (which, rates) in replicate_rates.iter_mut().enumerate() {
                rates.push(rate_of(which as u8, &pooled));
            }
        }
        for rates in replicate_rates.iter_mut() {
            rates.sort_by(|a, b| a.total_cmp(b));
        }
    }
    let interval = |which: usize| -> (f64, f64) {
        if replicate_rates[which].is_empty() {
            (point[which], point[which])
        } else {
            (
                percentile_of(&replicate_rates[which], 2.5),
                percentile_of(&replicate_rates[which], 97.5),
            )
        }
    };
    let hit_rate = |which: usize| -> HitRate {
        let hits = all
            .iter()
            .filter(|i| hit(&scores[**i], which as u8))
            .count();
        let (lo, hi) = interval(which);
        HitRate {
            hits,
            queries,
            rate: point[which],
            lo,
            hi,
        }
    };
    let fraction = |pick: fn(&OutcomeCounts) -> u32| -> f64 {
        if queries == 0 {
            return 0.0;
        }
        scores
            .iter()
            .map(|score| {
                let counts = &score.outcome_counts;
                if counts.trajectories == 0 {
                    0.0
                } else {
                    f64::from(pick(counts)) / f64::from(counts.trajectories)
                }
            })
            .sum::<f64>()
            / queries as f64
    };
    RecoveryReport {
        queries,
        top1: hit_rate(0),
        top10: hit_rate(1),
        top25: hit_rate(2),
        skeleton_top25: hit_rate(3),
        mean_distinct: if queries == 0 {
            0.0
        } else {
            scores
                .iter()
                .map(|score| f64::from(score.outcome_counts.distinct))
                .sum::<f64>()
                / queries as f64
        },
        mean_unresolved: if queries == 0 {
            0.0
        } else {
            scores
                .iter()
                .map(|score| f64::from(score.outcome_counts.identity_unresolved))
                .sum::<f64>()
                / queries as f64
        },
        mean_finished_fraction: fraction(|counts| counts.finished),
        mean_dead_end_fraction: fraction(|counts| counts.dead_end),
        mean_truncated_fraction: fraction(|counts| counts.truncated),
        mean_rejected_fraction: fraction(|counts| {
            counts
                .rejected_replay
                .saturating_add(counts.rejected_containment)
                .saturating_add(counts.containment_unresolved)
        }),
        zero_candidate_queries: scores
            .iter()
            .filter(|score| score.outcome_counts.candidates == 0)
            .count(),
    }
}
