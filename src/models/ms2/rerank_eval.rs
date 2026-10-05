//! Ranking metrics and paired molecule bootstrap for the reranker experiment
//! (task RK; `docs/MS2_V1_ARCHITECTURE.md` §4.3).
//!
//! Pure host `f64` on slices: no tensors, no kernels, no device reads. The
//! driver (`examples/ms2_rerank_experiment.rs`) calls these on the report
//! split (and, labelled as not held out, on the calibration split) for the
//! `raw` and `ms2-reranker-v1` rankings.

use std::collections::BTreeMap;

use super::rerank::ADD_ATOM_KIND;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic 64-bit generator for the bootstrap (SplitMix64, the same
/// constants as [`crate::models::ms2::metrics`]).
struct SplitMix64 {
    /// Current state.
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next `u64`.
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The `p`th percentile of a sorted slice by linear interpolation (the same
/// rule as [`crate::models::ms2::metrics`]).
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

/// Indices of one group in example order.
fn group_positions(groups: &[usize], group: usize) -> Vec<usize> {
    groups
        .iter()
        .enumerate()
        .filter_map(|(i, &g)| (g == group).then_some(i))
        .collect()
}

// ---------------------------------------------------------------------------
// Ranking metrics
// ---------------------------------------------------------------------------

/// ROC AUC of `scores` against binary `labels`.
///
/// A label counts as positive when it is not `0.0` (callers pass `0.0`/`1.0`).
/// Tied scores share the average of their ranks (1-based, ascending), so ties
/// contribute one half. Returns `None` when either class is absent (including
/// the empty input). NaN scores sort largest (via `total_cmp`); NaN labels
/// count as positive, so callers must not pass them.
///
/// Panics when the slices differ in length.
pub fn roc_auc(scores: &[f64], labels: &[f64]) -> Option<f64> {
    assert_eq!(
        scores.len(),
        labels.len(),
        "roc_auc: {} scores for {} labels",
        scores.len(),
        labels.len()
    );
    let n = scores.len();
    let n_pos = labels.iter().filter(|&&y| y != 0.0).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    // Ascending ranks with average ranks for ties.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        // Positions i..j (0-based) share the 1-based average rank.
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for &pos in &order[i..j] {
            if labels[pos] != 0.0 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }
    let n_pos = n_pos as f64;
    let n_neg = n_neg as f64;
    Some((rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))
}

/// Top-1 precision over groups (spectra).
///
/// `groups` holds one group id per example. Over groups with at least one
/// example, the fraction whose highest-scored example has a non-zero label;
/// ties break to the smaller example index (the first maximum in example
/// order). Returns the fraction and the group count; with no group the
/// fraction is `0.0` and the count is `0`.
///
/// Panics when the slices differ in length.
pub fn top1_precision(scores: &[f64], labels: &[f64], groups: &[usize]) -> (f64, usize) {
    assert_eq!(
        scores.len(),
        labels.len(),
        "top1_precision: {} scores for {} labels",
        scores.len(),
        labels.len()
    );
    assert_eq!(
        scores.len(),
        groups.len(),
        "top1_precision: {} scores for {} groups",
        scores.len(),
        groups.len()
    );
    let mut ids: Vec<usize> = groups.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return (0.0, 0);
    }
    let mut hits = 0usize;
    for &g in &ids {
        let pos = group_positions(groups, g);
        debug_assert!(!pos.is_empty());
        // Strict improvement keeps the first (smallest-index) maximum.
        let mut best = pos[0];
        for &i in &pos[1..] {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        if labels[best] != 0.0 {
            hits += 1;
        }
    }
    (hits as f64 / ids.len() as f64, ids.len())
}

/// Precision at `r` over groups (spectra).
///
/// The mean over groups with at least one example of the label mean of the
/// top `min(r, group size)` examples by score (ties break to the smaller
/// example index, as in [`top1_precision`]). With `r == 0` or no group the
/// value is `0.0`.
///
/// Panics when the slices differ in length.
pub fn precision_at(scores: &[f64], labels: &[f64], groups: &[usize], r: usize) -> f64 {
    assert_eq!(
        scores.len(),
        labels.len(),
        "precision_at: {} scores for {} labels",
        scores.len(),
        labels.len()
    );
    assert_eq!(
        scores.len(),
        groups.len(),
        "precision_at: {} scores for {} groups",
        scores.len(),
        groups.len()
    );
    if r == 0 {
        return 0.0;
    }
    let mut ids: Vec<usize> = groups.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &g in &ids {
        let mut pos = group_positions(groups, g);
        debug_assert!(!pos.is_empty());
        // Descending by score, ties by smaller example index.
        pos.sort_by(|&a, &b| {
            scores[b]
                .total_cmp(&scores[a])
                .then_with(|| a.cmp(&b))
        });
        let take = r.min(pos.len());
        let mut hits = 0.0;
        for &i in &pos[..take] {
            if labels[i] != 0.0 {
                hits += 1.0;
            }
        }
        sum += hits / take as f64;
    }
    sum / ids.len() as f64
}

// ---------------------------------------------------------------------------
// Paired molecule bootstrap
// ---------------------------------------------------------------------------

/// Paired bootstrap interval of the per-group difference `a − b`.
///
/// `per_group_a` and `per_group_b` hold one value per group (in the same group
/// order) and `molecule_of_group` the molecule index of each group. Only
/// groups defined (non-NaN) in BOTH rankings are kept; their paired
/// differences are averaged per molecule, then molecules are resampled with
/// replacement `n` times (seeded SplitMix64, so a seed gives the same
/// interval on any platform). The point is the mean over molecules of the
/// per-molecule mean differences; the interval is the 95% percentile
/// interval (2.5th/97.5th by linear interpolation, the same rule as
/// [`crate::models::ms2::metrics`]).
///
/// Returns `None` when no group is defined in both rankings (there is no
/// paired difference to average). With `n == 0` the measured point is
/// returned with `lo == hi == point` — never zero in place of a measured
/// value.
///
/// Panics when the three slices differ in length.
pub fn paired_bootstrap(
    per_group_a: &[f64],
    per_group_b: &[f64],
    molecule_of_group: &[usize],
    n: usize,
    seed: u64,
) -> Option<(f64, f64, f64)> {
    assert_eq!(
        per_group_a.len(),
        per_group_b.len(),
        "paired_bootstrap: {} values for a but {} for b",
        per_group_a.len(),
        per_group_b.len()
    );
    assert_eq!(
        per_group_a.len(),
        molecule_of_group.len(),
        "paired_bootstrap: {} values for {} molecules",
        per_group_a.len(),
        molecule_of_group.len()
    );
    // Pair first: keep only groups defined in both rankings, form their
    // paired differences, then average differences per molecule.
    let mut per_mol: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for (i, &m) in molecule_of_group.iter().enumerate() {
        let a = per_group_a[i];
        let b = per_group_b[i];
        if a.is_nan() || b.is_nan() {
            continue;
        }
        let e = per_mol.entry(m).or_insert((0.0, 0));
        e.0 += a - b;
        e.1 += 1;
    }
    let mut diffs: Vec<f64> = Vec::new();
    for (_, (sum, count)) in &per_mol {
        debug_assert!(*count > 0);
        diffs.push(sum / *count as f64);
    }
    if diffs.is_empty() {
        return None;
    }
    let point = diffs.iter().sum::<f64>() / diffs.len() as f64;
    if n == 0 {
        return Some((point, point, point));
    }
    let mut rng = SplitMix64::new(seed);
    let m = diffs.len();
    let mut means: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut sum = 0.0;
        for _ in 0..m {
            let j = (rng.next() % m as u64) as usize;
            sum += diffs[j];
        }
        means.push(sum / m as f64);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    Some((point, percentile_of(&means, 2.5), percentile_of(&means, 97.5)))
}

// ---------------------------------------------------------------------------
// Leakage guard (B1): molecule-key overlap and fit provenance
// ---------------------------------------------------------------------------

/// First molecule key of `a` that also occurs in `b`, in `a`'s order.
///
/// Both slices hold molecule keys in export order. Returns `None` when the
/// two key sets are disjoint.
pub fn find_shared_key(a: &[String], b: &[String]) -> Option<String> {
    for key in a {
        if b.iter().any(|other| other == key) {
            return Some(key.clone());
        }
    }
    None
}

/// Whether the checkpoint's recorded fit export was verified against the
/// caller-supplied `--generator-fit` export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitProvenance {
    /// The checkpoint records no fit export identity: the supplied export
    /// is taken on the caller's word and reported as such.
    Unrecorded,
    /// Every recorded fit identifier matches the supplied export.
    Matches,
}

/// Compare the checkpoint's recorded fit export identity (`enum_fit_name` /
/// `enum_fit_sha256`, or any other export identity the train config records)
/// against the caller-supplied `--generator-fit` export (`supplied_name` /
/// `supplied_sha256`).
///
/// Every recorded identifier must equal its supplied counterpart; the first
/// mismatch is `Err` naming both sides. When the checkpoint records nothing,
/// the caller-supplied export is accepted as [`FitProvenance::Unrecorded`]
/// (the driver reports it as supplied-by-the-caller, not recorded in the
/// checkpoint).
pub fn check_generator_fit(
    fit_name: Option<&str>,
    fit_sha256: Option<&str>,
    supplied_name: &str,
    supplied_sha256: &str,
) -> Result<FitProvenance, String> {
    if let Some(recorded) = fit_name
        && recorded != supplied_name
    {
        return Err(format!(
            "checkpoint fit export mismatch: recorded enum_fit_name '{recorded}' != supplied --generator-fit '{supplied_name}'"
        ));
    }
    if let Some(recorded) = fit_sha256
        && recorded != supplied_sha256
    {
        return Err(format!(
            "checkpoint fit export mismatch: recorded enum_fit_sha256 '{recorded}' != supplied --generator-fit sha256 '{supplied_sha256}'"
        ));
    }
    if fit_name.is_none() && fit_sha256.is_none() {
        return Ok(FitProvenance::Unrecorded);
    }
    Ok(FitProvenance::Matches)
}

/// Validate the bootstrap repetition count: `--bootstrap 0` cannot produce
/// an interval, so it is rejected (the driver exits 2) instead of reporting
/// zero in place of measured values.
pub fn validate_bootstrap(n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("--bootstrap must be non-zero (zero repetitions cannot form an interval)".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Size stratum (B5): atoms of the candidate's own trace
// ---------------------------------------------------------------------------

/// Atoms of a candidate's own trace: the atom-adding actions below `length`.
///
/// `token_words` holds one record's `steps * 4` token words of
/// `(kind, atom_type, bond_order, pointer)`; an `ADD_ATOM` kind below
/// `length` adds one atom. This is independent of any parent budget (unlike
/// replaying under the true parent composition, which fails — with atom
/// count 0 — for a candidate outside that budget), so a valid finished
/// record keeps its own size stratum while containment under the true parent
/// stays the label.
pub fn trace_atom_count(token_words: &[u32], length: u32) -> usize {
    let steps = token_words.len() / 4;
    let mut atoms = 0usize;
    let mut s = 0usize;
    while s < steps {
        if (s as u32) < length && token_words[s * 4] == ADD_ATOM_KIND {
            atoms += 1;
        }
        s += 1;
    }
    atoms
}

// ---------------------------------------------------------------------------
// Search policy (B6): configuration fingerprint of the scored support
// ---------------------------------------------------------------------------

/// Inputs of [`build_search_policy`]: the configuration values that shape the
/// scored support a calibration was fitted under.
#[derive(Clone, Copy)]
pub struct SearchPolicyInputs<'a> {
    /// Formula source: `"table"` or `"enumerate"`.
    pub formula_source: &'a str,
    /// Scored-candidate capacity per spectrum (`M`).
    pub window_m: u32,
    /// Rows scored before the search reports exhausted.
    pub formula_rows_scored_max: u32,
    /// Table source's visit limit (`formula_rows_visited_max`).
    pub formula_rows_visited_max: u32,
    /// Enumeration per-lane visit budget (`enum_lane_visits_max`).
    pub enum_lane_visits_max: u32,
    /// Enumeration submitted-lane cap (`enum_lanes_max`).
    pub enum_lanes_max: u32,
    /// Trajectory allocation (`AllocationMode`).
    pub allocation: &'a str,
    /// Identity mode (`IdentityMode`).
    pub identity: &'a str,
    /// Packed slots per spectrum (effective `R`).
    pub returned: u32,
    /// First 16 hex digits of the formula table file's SHA-256.
    pub table_sha256_16: &'a str,
    /// First 16 hex digits of the enumeration domain SHA-256, when the
    /// checkpoint carries enumeration artifacts.
    pub enum_domain_sha256_16: Option<&'a str>,
    /// First 16 hex digits of the enumeration ratio-bounds SHA-256, when the
    /// checkpoint carries enumeration artifacts.
    pub enum_bounds_sha256_16: Option<&'a str>,
}

/// Build the `search_policy` configuration key: the canonical fingerprint
/// of the configuration values that produce the scored support.
///
/// Two runs that could score different supports must get different keys, so
/// `validate_matches` refuses calibration across them: the formula source,
/// window `M`, `formula_rows_scored_max`, the source's work limits
/// (`formula_rows_visited_max` for the table; `enum_lane_visits_max` and
/// `enum_lanes_max` for enumeration), allocation, identity mode, `returned`,
/// and the SHA-256 of the formula table file and of the enumeration
/// artifacts when present (first 16 hex digits each).
pub fn build_search_policy(inp: &SearchPolicyInputs<'_>) -> String {
    let mut key = format!(
        "{}/M={}/scored_max={}/visited_max={}/lane_visits_max={}/lanes_max={}/alloc={}/identity={}/returned={}/table={}",
        inp.formula_source,
        inp.window_m,
        inp.formula_rows_scored_max,
        inp.formula_rows_visited_max,
        inp.enum_lane_visits_max,
        inp.enum_lanes_max,
        inp.allocation,
        inp.identity,
        inp.returned,
        inp.table_sha256_16,
    );
    if let Some(sha) = inp.enum_domain_sha256_16 {
        key.push_str(&format!("/enum_domain={sha}"));
    }
    if let Some(sha) = inp.enum_bounds_sha256_16 {
        key.push_str(&format!("/enum_bounds={sha}"));
    }
    key
}
