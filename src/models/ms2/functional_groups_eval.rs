//! Functional-group evaluation `ms2-fg-v4` (§4 of the FG2 task): pure host
//! functions over candidate records.
//!
//! Inputs per spectrum are the true parent [`MolGraph`] (giving the type set
//! `T`) and the candidates of one generation call. A candidate counts when it
//! is finished and device-valid and not a trace or graph duplicate — the same
//! eligibility as [`eligible_examples`](super::rerank::eligible_examples),
//! minus the containment label. The evaluated family is every eligible
//! candidate with at least one atom (the model may STOP after a single
//! atom); no size filtering is applied. Its graph is rebuilt from ITS OWN
//! trace and ITS OWN conditioning formula ([`CandidateBatch::formula_counts`]
//! of that record as the replay budget) — NOT under the true parent's
//! composition. `P_k` is the union of determined types over the first `k`
//! eligible candidates in raw-score order (`formula_log_prob +
//! trace_log_prob`, ties by trajectory). Every set metric is additionally
//! reported restricted to candidates of at least 3 atoms (`min_atoms_3`,
//! next to the unrestricted one). The candidate-size distribution
//! (fractions with 1, 2, 3–5, 6–9, 10–16 atoms over all eligible candidates)
//! is reported alongside.
//!
//! Every spectrum of the split stays in every denominator: a spectrum without
//! eligible candidates has `P = {}`, and a candidate whose own-formula replay
//! fails contributes the empty type set (it stays eligible for the
//! candidate-level denominators). A record with no conditioning formula
//! (all-zero counts) replays under the zero budget, so a non-empty trace is
//! invalid there and contributes nothing.
//!
//! Bootstrap intervals are 95% percentile intervals over MOLECULES with the
//! same SplitMix64 generator and linear-interpolation percentile rule as
//! [`summarize`](super::metrics::summarize): resample molecules with
//! replacement, recompute the metric over the resampled spectra, take the
//! 2.5th/97.5th percentiles. The macro denominator (types with at least 10
//! true spectra on the original split) is chosen once and held fixed in
//! every replicate; a held type with zero support in a replicate contributes
//! 0 to that replicate's macro average (the documented zero-denominator
//! convention for replicates; point-estimate tables show `null`/n/a for
//! undefined ratios).
//!
//! Zero-denominator conventions (applied everywhere, `null` in JSON, `n/a`
//! in print): micro precision with no prediction is `null`; micro recall
//! with no truth is `null`; micro F1 is `null` when either side is `null`
//! and 0 when precision and recall are both defined and both zero; per-type
//! precision/recall/F1 follow the same rule; macro
//! averages over an empty held set are `null` (otherwise `null` components
//! contribute 0, keeping the denominator fixed); mean Jaccard is 1 on a
//! both-empty spectrum and the mean over zero spectra is `null`. Micro
//! precision is always read next to the empty-`P` fraction and next to
//! recall.
//!
//! Reference rows: `prior` (train-frequency threshold, empty set included,
//! ties to the smaller set), `candidate_formula_prior` (the plain prior set
//! minus types whose needed elements are absent from the spectrum's
//! best-`formula_log_prob` eligible record — the top-ranked formula
//! hypothesis among allocated trajectories; generation output does not
//! expose unallocated hypotheses), `label_ceiling` (union over pseudo-label
//! targets), and `recipe_fragments` (union over the label recipe's candidate
//! fragments: at most two bond cuts, 3 to 16 atoms — coverage of the recipe,
//! not a ceiling for the model). The last two use the parent structure and
//! are not baselines.
//!
//! Analytic note on recall: the determined rule itself does not cap recall
//! for the model's fragment family — any group instance fits with its
//! closing neighbours in a 16-atom fragment (pattern atoms plus the
//! neighbours needed to make exclusions and bond statuses certain). This is
//! verified by [`closing_fragment_not_found`] on the validation parents and
//! reported as `closing_fragment_not_found` (a count of instances for which
//! the layer-expansion search found no determined fragment of at most 16
//! atoms — a search failure, not a proof — never assumed to be zero).

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::chem::Composition;
use super::contract::{CandidateBatch, candidate_status};
use super::functional_groups::{
    FG_NAMES, FULL_MASK, HETEROATOM_MASK, N_FG, SPECIFIC_MASK, fg_instances, functional_groups_v4,
};
use super::grammar::{Limits, Token, replay};
use super::graph::MolGraph;
use super::targets::{Candidates, Labels, RecipeLimits};

/// One candidate record parsed out of a [`CandidateBatch`].
#[derive(Clone, Debug)]
pub struct EvalRecord {
    /// The trace ended with STOP (the batch `finished` bit).
    pub finished: bool,
    /// Finished without a device-side validity failure: `FINISHED` set,
    /// `INVALID_FINAL`, `TRUNCATED` and `REQUEST_FAILED` unset.
    pub device_valid: bool,
    /// A trace or graph duplicate (`DUPLICATE_TRACE` or `DUPLICATE_GRAPH`).
    pub duplicate: bool,
    /// The emitted token prefix.
    pub tokens: Vec<Token>,
    /// The record's own conditioning formula (all zero when there is none).
    pub formula: Composition,
    /// `log p(formula | spectrum)` of the record.
    pub formula_log_prob: f32,
    /// Summed action log-probabilities of the record.
    pub trace_log_prob: f32,
    /// The trajectory index (raw-score tie-break).
    pub trajectory: u32,
}

impl EvalRecord {
    /// Whether the candidate counts: finished, device-valid, not a duplicate.
    pub fn eligible(&self) -> bool {
        self.finished && self.device_valid && !self.duplicate
    }

    /// Raw score ordering key: `formula_log_prob + trace_log_prob`.
    pub fn score(&self) -> f64 {
        f64::from(self.formula_log_prob) + f64::from(self.trace_log_prob)
    }
}

/// Parse every record of a [`CandidateBatch`] into per-spectrum rows.
///
/// Layout errors (a `length` past `max_steps`, a token field past `u8`) are
/// [`Error::Config`]. Device validity and duplication come from the status
/// bits; an unresolved graph identity stays eligible (it is not a duplicate
/// flag, as in [`eligible_examples`](super::rerank::eligible_examples)).
pub fn eval_records(batch: &CandidateBatch) -> Result<Vec<Vec<EvalRecord>>> {
    let k = batch.trajectories;
    let t = batch.max_steps;
    let mut out: Vec<Vec<EvalRecord>> = Vec::with_capacity(batch.batch);
    for b in 0..batch.batch {
        let mut rows = Vec::with_capacity(k);
        for kk in 0..k {
            let r = b * k + kk;
            let st = batch.status[r];
            let finished = st & candidate_status::FINISHED != 0;
            let device_valid = finished
                && st & candidate_status::INVALID_FINAL == 0
                && st & candidate_status::TRUNCATED == 0
                && st & candidate_status::REQUEST_FAILED == 0;
            let duplicate = st & candidate_status::DUPLICATE_TRACE != 0
                || st & candidate_status::DUPLICATE_GRAPH != 0;
            let len = batch.length[r] as usize;
            if len > t {
                return Err(Error::config(format!(
                    "eval_records: spectrum {b} candidate {kk} length {len} past T={t}"
                )));
            }
            let base = r * t * 4;
            let mut tokens = Vec::with_capacity(len);
            for step in 0..len {
                let f = &batch.actions[base + step * 4..base + step * 4 + 4];
                let conv = |v: u32, what: &str| {
                    u8::try_from(v).map_err(|_| {
                        Error::config(format!(
                            "eval_records: spectrum {b} candidate {kk} step {step} {what} {v} past u8"
                        ))
                    })
                };
                tokens.push(Token {
                    kind: conv(f[0], "kind")?,
                    atom_type: conv(f[1], "type")?,
                    bond: conv(f[2], "bond")?,
                    pointer: conv(f[3], "pointer")?,
                });
            }
            let mut formula: Composition = [0; 10];
            if !batch.formula_counts.is_empty() {
                for e in 0..10 {
                    formula[e] = batch.formula_counts[r * 10 + e];
                }
            }
            rows.push(EvalRecord {
                finished,
                device_valid,
                duplicate,
                tokens,
                formula,
                formula_log_prob: batch.formula_log_prob[r],
                trace_log_prob: batch.trace_log_prob[r],
                trajectory: batch.trajectory[r],
            });
        }
        out.push(rows);
    }
    Ok(out)
}

/// One eligible candidate's determined groups: per-type instance counts, the
/// undetermined mask, the heavy-atom count, and the raw-score ordering key.
#[derive(Clone, Debug)]
pub struct FgCandidate {
    /// Determined instance counts per type (index `id - 1`).
    pub counts: [u32; N_FG],
    /// Types with a matched but undetermined instance.
    pub undet: u32,
    /// Heavy-atom count of the replayed graph (0 when replay failed).
    pub atoms: usize,
    /// Raw score (`formula_log_prob + trace_log_prob`).
    pub score: f64,
    /// Trajectory index (score tie-break).
    pub trajectory: u32,
}

impl FgCandidate {
    /// Determined presence mask.
    pub fn mask(&self) -> u32 {
        let mut m = 0u32;
        for (i, c) in self.counts.iter().enumerate() {
            if *c > 0 {
                m |= 1u32 << i;
            }
        }
        m
    }

    /// Determined instances in a vocabulary mask.
    pub fn instances_in(&self, vocab: u32) -> u32 {
        let mut n = 0u32;
        for (i, c) in self.counts.iter().enumerate() {
            if vocab & (1u32 << i) != 0 {
                n += *c;
            }
        }
        n
    }
}

/// One spectrum's functional-group evaluation input.
#[derive(Clone, Debug)]
pub struct FgSpectrumDatum {
    /// Molecule index (for the molecule bootstrap).
    pub molecule: usize,
    /// Determined types of the true parent.
    pub parent_mask: u32,
    /// Eligible candidates in raw-score order.
    pub candidates: Vec<FgCandidate>,
    /// Union of determined types over the pseudo-label target graphs.
    pub label_union: u32,
    /// Union over the label recipe's candidate fragments.
    pub oracle_union: u32,
    /// Conditioning formula of the eligible record with the largest
    /// `formula_log_prob` (the top-ranked formula hypothesis among allocated
    /// trajectories; `None` when there is no eligible candidate).
    /// Generation output does not expose a spectrum's top formula when no
    /// trajectory was allocated to it.
    pub top_formula: Option<Composition>,
}

/// Build one spectrum's datum: replay every eligible record under its own
/// formula, detect its groups, and sort by raw score (ties by trajectory).
///
/// `label_union`/`oracle_union` are computed by the caller (see
/// [`label_union`] and [`oracle_union`]) so this function stays free of
/// recipe work.
pub fn spectrum_datum(
    parent: &MolGraph,
    molecule: usize,
    records: &[EvalRecord],
    limits: Limits,
    label_union: u32,
    oracle_union: u32,
) -> FgSpectrumDatum {
    let parent_mask = functional_groups_v4(parent).mask();
    let mut candidates = Vec::new();
    for rec in records.iter().filter(|r| r.eligible()) {
        let (counts, undet, atoms) = match replay(&rec.tokens, limits, Some(rec.formula)) {
            Ok(state) if state.stopped() && state.atoms() > 0 => match state.graph() {
                Ok(graph) => {
                    let set = functional_groups_v4(&graph);
                    (
                        *set.counts(),
                        super::functional_groups::undetermined(&graph),
                        graph.atoms().len(),
                    )
                }
                Err(_) => ([0; N_FG], 0, 0),
            },
            _ => ([0; N_FG], 0, 0),
        };
        candidates.push(FgCandidate {
            counts,
            undet,
            atoms,
            score: rec.score(),
            trajectory: rec.trajectory,
        });
    }
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.trajectory.cmp(&b.trajectory))
    });
    // Top-ranked formula hypothesis among allocated trajectories: largest
    // `formula_log_prob`, independent of trace scores.
    let top_formula = records
        .iter()
        .filter(|r| r.eligible())
        .max_by(|a, b| {
            a.formula_log_prob
                .total_cmp(&b.formula_log_prob)
                .then_with(|| b.trajectory.cmp(&a.trajectory))
        })
        .map(|r| r.formula);
    FgSpectrumDatum {
        molecule,
        parent_mask,
        candidates,
        label_union,
        oracle_union,
        top_formula,
    }
}

/// Union of determined types over a spectrum's pseudo-label target graphs
/// (`labels.targets`, the graphs the model is trained to produce).
///
/// Each target trace replays under [`Limits::V0`] with no budget; targets
/// that fail to replay are skipped.
pub fn label_union(labels: &Labels) -> u32 {
    let mut union = 0u32;
    for target in &labels.targets {
        let Ok(state) = replay(&target.trace, Limits::V0, None) else {
            continue;
        };
        if !state.stopped() || state.atoms() == 0 {
            continue;
        }
        let Ok(graph) = state.graph() else {
            continue;
        };
        union |= functional_groups_v4(&graph).mask();
    }
    union
}

/// Union of determined types over the label recipe's candidate fragments
/// (at most two bond cuts, 3 to 16 atoms): coverage of the recipe, not a
/// ceiling for the model.
///
/// Enumerates [`enumerate_embeddings`](super::targets::enumerate_embeddings)
/// under [`RecipeLimits::V0`].
///
/// Returns the union and whether any spectrum-level work limit was hit (a
/// non-zero canonicalization-failure count, or a preparation failure, which
/// yields the empty union).
pub fn recipe_fragments_union(parent: &MolGraph) -> (u32, bool) {
    let limits = RecipeLimits::V0;
    let candidates = match Candidates::new(parent, &limits) {
        Ok(c) => c,
        Err(_) => return (0, true),
    };
    let limited = candidates.canonicalization_failures() > 0;
    let mut union = 0u32;
    for emb in candidates.embeddings() {
        let Ok(sub) = parent.induced(&emb.atoms) else {
            continue;
        };
        union |= functional_groups_v4(&sub).mask();
    }
    (union, limited)
}

/// Backwards-compatible alias of [`recipe_fragments_union`] (the v1 row
/// name `oracle`). New code should use `recipe_fragments_union`.
pub fn oracle_union(parent: &MolGraph) -> (u32, bool) {
    recipe_fragments_union(parent)
}

// ---------------------------------------------------------------------------
// Metrics (`None`/`null` for zero denominators)
// ---------------------------------------------------------------------------

/// One point estimate with its 95% percentile bootstrap interval over
/// molecules. `None` (`null` in JSON, `n/a` in print) marks an undefined
/// ratio (zero denominator); see the module docs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    /// The metric on all spectra (`None` when undefined).
    pub point: Option<f64>,
    /// 2.5th percentile of the molecule-bootstrap values.
    pub lo: Option<f64>,
    /// 97.5th percentile of the molecule-bootstrap values.
    pub hi: Option<f64>,
}

/// One row of the per-type table: point estimates with molecule-bootstrap
/// intervals for precision and recall (`None` bounds when the point estimate
/// is undefined — null when the denominator is zero in the point estimate —
/// or when no replicate is defined).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeRow {
    /// 1-based vocabulary id.
    pub id: usize,
    /// Vocabulary name.
    pub name: String,
    /// Spectra with the type in `T` (the support).
    pub true_count: usize,
    /// Spectra with the type in `P`.
    pub predicted: usize,
    /// Spectra with the type in both.
    pub tp: usize,
    /// `tp / predicted` (`None` when never predicted).
    pub precision: Option<f64>,
    /// 2.5th percentile of the molecule-bootstrap precisions.
    pub precision_lo: Option<f64>,
    /// 97.5th percentile of the molecule-bootstrap precisions.
    pub precision_hi: Option<f64>,
    /// `tp / true_count` (`None` when the support is 0).
    pub recall: Option<f64>,
    /// 2.5th percentile of the molecule-bootstrap recalls.
    pub recall_lo: Option<f64>,
    /// 97.5th percentile of the molecule-bootstrap recalls.
    pub recall_hi: Option<f64>,
    /// Paired bootstrap interval of the recall difference (own − donor) for
    /// the donor-peaks ablation: 2.5th percentile (`None` without donor data
    /// or when the support is 0).
    pub recall_diff_lo: Option<f64>,
    /// Paired bootstrap interval of the recall difference (own − donor):
    /// 97.5th percentile.
    pub recall_diff_hi: Option<f64>,
}

/// Set-level metrics shared by model and reference rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SetMetrics {
    /// Micro precision `sum |P ∩ T| / sum |P|` (`None` when no prediction).
    pub micro_precision: MetricPoint,
    /// Micro recall `sum |P ∩ T| / sum |T|` (`None` when no truth).
    pub micro_recall: MetricPoint,
    /// Micro F1 (harmonic mean; `None` when undefined).
    pub micro_f1: MetricPoint,
    /// Macro precision over the held supported types.
    pub macro_precision: MetricPoint,
    /// Macro recall over the held supported types.
    pub macro_recall: MetricPoint,
    /// Macro F1 over the held supported types.
    pub macro_f1: MetricPoint,
    /// Type ids with at least 10 true spectra on the original split (the
    /// held macro denominator).
    pub types_used: Vec<usize>,
    /// True-spectrum support of every type (index `id - 1`).
    pub supports: Vec<usize>,
    /// Per-type table.
    pub per_type: Vec<TypeRow>,
    /// Mean Jaccard `|P ∩ T| / |P ∪ T|` per spectrum (1 when both empty;
    /// `None` over zero spectra).
    pub jaccard: MetricPoint,
    /// Exact-set-match rate (`None` over zero spectra).
    pub exact_match: MetricPoint,
    /// Fraction of spectra with empty `P` (`None` over zero spectra).
    pub empty_p: MetricPoint,
}

/// Model metrics on one vocabulary: set metrics plus the instance and
/// candidate levels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VocabReport {
    /// Set-level metrics (unrestricted candidates).
    pub set: SetMetrics,
    /// Set-level metrics restricted to candidates of at least 3 atoms.
    pub set_min_atoms_3: SetMetrics,
    /// Of all determined instances in all eligible candidates, the fraction
    /// whose type is in `T` (`None` when there is no determined instance).
    pub instance_precision: MetricPoint,
    /// Mean determined instances per eligible candidate.
    pub mean_instances: MetricPoint,
    /// Mean undetermined-mask bits per eligible candidate.
    pub mean_undet: MetricPoint,
    /// Fraction of eligible candidates with at least one determined group.
    pub cand_with_group: MetricPoint,
    /// Among those, the fraction ALL of whose determined types are in `T`
    /// (`None` when no eligible candidate carries a group).
    pub cand_all_real: MetricPoint,
}

/// Candidate-size distribution over all eligible candidates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateSizes {
    /// Eligible candidate count.
    pub n: usize,
    /// Fraction with exactly 1 atom (`None` when `n == 0`).
    pub frac_1: Option<f64>,
    /// Fraction with exactly 2 atoms.
    pub frac_2: Option<f64>,
    /// Fraction with 3–5 atoms.
    pub frac_3_5: Option<f64>,
    /// Fraction with 6–9 atoms.
    pub frac_6_9: Option<f64>,
    /// Fraction with 10–16 atoms.
    pub frac_10_16: Option<f64>,
}

/// Donor-peaks ablation of one `k`: the same checkpoint generating with
/// donor peaks (an input ablation, not a retraining), plus paired own−donor
/// differences sharing one molecule bootstrap (same spectra, same resamples).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DonorReport {
    /// Full 28-type vocabulary under donor peaks.
    pub full: VocabReport,
    /// Specific subset under donor peaks.
    pub specific: VocabReport,
    /// Heteroatom subset under donor peaks.
    pub heteroatom: VocabReport,
    /// Paired own−donor differences, full vocabulary.
    pub diff_full: PairedSetDiff,
    /// Paired own−donor differences, specific subset.
    pub diff_specific: PairedSetDiff,
    /// Paired own−donor differences, heteroatom subset.
    pub diff_heteroatom: PairedSetDiff,
}

/// Model metrics at one `k`: all three vocabularies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KReport {
    /// Candidates unioned (`P_k`).
    pub k: usize,
    /// Full 28-type vocabulary.
    pub full: VocabReport,
    /// Specific subset (no generic `carbonyl`).
    pub specific: VocabReport,
    /// Heteroatom subset (no `carbonyl`, `alkene`, `alkyne`, `arene_ring`).
    pub heteroatom: VocabReport,
    /// Candidate-size distribution (k-independent: all eligible candidates).
    pub sizes: CandidateSizes,
    /// Donor-peaks ablation (`None` unless evaluated with donor peaks).
    pub donor: Option<DonorReport>,
}

/// Reference rows: fixed type sets per spectrum, hence k-independent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RefRow {
    /// Row name (`prior`, `candidate_formula_prior`, `label_ceiling`,
    /// `recipe_fragments`).
    pub name: String,
    /// Full-vocabulary set metrics.
    pub full: SetMetrics,
    /// Specific-subset set metrics.
    pub specific: SetMetrics,
    /// Heteroatom-subset set metrics.
    pub heteroatom: SetMetrics,
}

/// The fixed prior set plus its train fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriorBaseline {
    /// Frequency threshold chosen on train molecules (2.0 marks the empty
    /// set: above every attainable frequency).
    pub tau: f64,
    /// Type ids with train frequency at least `tau` (empty when `tau` is
    /// above 1).
    pub set: Vec<usize>,
}

/// Micro F1 of fixed prediction sets against truths (point only): the
/// objective the prior threshold maximises on train molecules. `None` when
/// undefined (zero denominators on either side of the harmonic mean).
pub fn micro_f1_point(preds: &[u32], truths: &[u32]) -> Option<f64> {
    let mut inter = 0u64;
    let mut p = 0u64;
    let mut t = 0u64;
    for (pr, tr) in preds.iter().zip(truths.iter()) {
        inter += (pr & tr).count_ones() as u64;
        p += pr.count_ones() as u64;
        t += tr.count_ones() as u64;
    }
    f1_of(
        if p == 0 {
            None
        } else {
            Some(inter as f64 / p as f64)
        },
        if t == 0 {
            None
        } else {
            Some(inter as f64 / t as f64)
        },
    )
}

/// Harmonic mean: `None` when either side is `None` (a zero denominator).
/// When precision and recall are both defined and both zero, F1 is 0.
fn f1_of(p: Option<f64>, r: Option<f64>) -> Option<f64> {
    let (p, r) = (p?, r?);
    if p + r == 0.0 {
        Some(0.0)
    } else {
        Some(2.0 * p * r / (p + r))
    }
}

/// Choose the prior baseline: type ids whose frequency among train molecules
/// is at least `tau`, with `tau` maximising micro F1 on the train molecules.
/// The empty prediction set is always a candidate (at `tau = 2.0`, above
/// every attainable frequency). Ties break to the highest `tau` (the
/// smallest set). `train_masks` holds one parent type set per train
/// molecule. A `None` F1 never beats a defined one; ties among `None`
/// break to the smaller set.
pub fn choose_prior(train_masks: &[u32]) -> PriorBaseline {
    if train_masks.is_empty() {
        return PriorBaseline {
            tau: 1.0,
            set: Vec::new(),
        };
    }
    let n = train_masks.len() as f64;
    let mut freq = [0u32; N_FG];
    for m in train_masks {
        for i in 0..N_FG {
            if m & (1u32 << i) != 0 {
                freq[i] += 1;
            }
        }
    }
    let mut taus: Vec<f64> = freq.iter().map(|c| f64::from(*c) / n).collect();
    taus.sort_by(|a, b| a.total_cmp(b));
    taus.dedup();
    // The empty set is always available.
    taus.push(2.0);
    let mut best_f1: Option<f64> = None;
    let mut best_tau = f64::NEG_INFINITY;
    let mut best_set = Vec::new();
    let mut first = true;
    for tau in taus {
        let set: Vec<usize> = (1..=N_FG)
            .filter(|&id| f64::from(freq[id - 1]) / n >= tau)
            .collect();
        let pred = set.iter().fold(0u32, |m, id| m | (1u32 << (id - 1)));
        let f1 = micro_f1_point(&vec![pred; train_masks.len()], train_masks);
        let better = if first {
            true
        } else {
            match (f1, best_f1) {
                (Some(a), Some(b)) => a > b || (a == b && tau > best_tau),
                (Some(_), None) => true,
                (None, None) => tau > best_tau,
                (None, Some(_)) => false,
            }
        };
        if better {
            best_f1 = f1;
            best_tau = tau;
            best_set = set;
            first = false;
        }
    }
    PriorBaseline {
        tau: best_tau,
        set: best_set,
    }
}

/// The formula-aware prior: the plain prior set minus every type whose
/// needed elements ([`fg_elements`](super::functional_groups::fg_elements))
/// are absent from the spectrum's top formula — the largest `formula_log_prob`
/// among the spectrum's ELIGIBLE records (finished, device-valid,
/// non-duplicate candidates only; unallocated formula hypotheses are not
/// exposed). `None` (no eligible candidate) falls back to the plain prior set.
pub fn formula_aware_set(prior_set: &[usize], formula: Option<Composition>) -> Vec<usize> {
    let Some(formula) = formula else {
        return prior_set.to_vec();
    };
    prior_set
        .iter()
        .copied()
        .filter(|&id| {
            super::functional_groups::fg_elements(id)
                .iter()
                .all(|&e| formula[e] > 0)
        })
        .collect()
}

/// Evaluate fixed prediction sets: set-level metrics for one vocabulary.
///
/// `preds`/`truths`/`molecules` run in parallel (one entry per spectrum);
/// both masks are pre-restricted to the vocabulary under test. The macro
/// denominator is chosen once on the given split and held fixed in every
/// bootstrap replicate.
pub fn evaluate_sets(
    preds: &[u32],
    truths: &[u32],
    molecules: &[usize],
    bootstrap: usize,
    seed: u64,
) -> SetMetrics {
    assert_eq!(preds.len(), truths.len(), "evaluate_sets: preds vs truths");
    assert_eq!(
        preds.len(),
        molecules.len(),
        "evaluate_sets: preds vs molecules"
    );
    let point = set_point(preds, truths);
    let held = point.types_used.clone();
    // Bootstrap over molecules: resample molecules, pool their spectra.
    let mut mol_ids: Vec<usize> = molecules.to_vec();
    mol_ids.sort_unstable();
    mol_ids.dedup();
    let per_mol: Vec<Vec<usize>> = mol_ids
        .iter()
        .map(|m| {
            molecules
                .iter()
                .enumerate()
                .filter_map(|(i, mm)| (*mm == *m).then_some(i))
                .collect()
        })
        .collect();
    let boot = |stat: &dyn Fn(&[u32], &[u32]) -> Option<f64>| -> (Option<f64>, Option<f64>) {
        if mol_ids.is_empty() || bootstrap == 0 {
            let v = stat(preds, truths);
            return (v, v);
        }
        let mut rng = SplitMix64::new(seed);
        let m = per_mol.len();
        let mut values: Vec<f64> = Vec::with_capacity(bootstrap);
        for _ in 0..bootstrap {
            let mut bp = Vec::new();
            let mut bt = Vec::new();
            for _ in 0..m {
                let j = (rng.next() % m as u64) as usize;
                for &i in &per_mol[j] {
                    bp.push(preds[i]);
                    bt.push(truths[i]);
                }
            }
            if let Some(v) = stat(&bp, &bt) {
                values.push(v);
            }
        }
        if values.is_empty() {
            return (None, None);
        }
        values.sort_by(|a, b| a.total_cmp(b));
        (
            Some(percentile_of(&values, 2.5)),
            Some(percentile_of(&values, 97.5)),
        )
    };
    let bp = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).0;
    let br = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).1;
    let bf = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).2;
    let mp = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).3;
    let mr = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).4;
    let mf = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).5;
    let ja = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).6;
    let ex = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).7;
    let em = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).8;
    let (plo, phi) = boot(&bp);
    let (rlo, rhi) = boot(&br);
    let (flo, fhi) = boot(&bf);
    let (mplo, mphi) = boot(&mp);
    let (mrlo, mrhi) = boot(&mr);
    let (mflo, mfhi) = boot(&mf);
    let (jlo, jhi) = boot(&ja);
    let (elo, ehi) = boot(&ex);
    let (emlo, emhi) = boot(&em);
    // Per-type bootstrap intervals for precision and recall (null bounds
    // when the point estimate is undefined: the denominator is zero there).
    let mut per_type = point.per_type;
    for row in per_type.iter_mut() {
        let bit = 1u32 << (row.id - 1);
        if row.precision.is_some() {
            let stat = |p: &[u32], t: &[u32]| {
                let mut tp = 0u64;
                let mut pr = 0u64;
                for (a, b) in p.iter().zip(t.iter()) {
                    if a & bit != 0 {
                        pr += 1;
                        if b & bit != 0 {
                            tp += 1;
                        }
                    }
                }
                if pr == 0 {
                    None
                } else {
                    Some(tp as f64 / pr as f64)
                }
            };
            let (lo, hi) = boot(&stat);
            row.precision_lo = lo;
            row.precision_hi = hi;
        }
        if row.recall.is_some() {
            let stat = |p: &[u32], t: &[u32]| {
                let mut tp = 0u64;
                let mut tr = 0u64;
                for (a, b) in p.iter().zip(t.iter()) {
                    if b & bit != 0 {
                        tr += 1;
                        if a & bit != 0 {
                            tp += 1;
                        }
                    }
                }
                if tr == 0 {
                    None
                } else {
                    Some(tp as f64 / tr as f64)
                }
            };
            let (lo, hi) = boot(&stat);
            row.recall_lo = lo;
            row.recall_hi = hi;
        }
    }
    SetMetrics {
        micro_precision: MetricPoint {
            point: point.micro_precision.point,
            lo: plo,
            hi: phi,
        },
        micro_recall: MetricPoint {
            point: point.micro_recall.point,
            lo: rlo,
            hi: rhi,
        },
        micro_f1: MetricPoint {
            point: point.micro_f1.point,
            lo: flo,
            hi: fhi,
        },
        macro_precision: MetricPoint {
            point: point.macro_precision.point,
            lo: mplo,
            hi: mphi,
        },
        macro_recall: MetricPoint {
            point: point.macro_recall.point,
            lo: mrlo,
            hi: mrhi,
        },
        macro_f1: MetricPoint {
            point: point.macro_f1.point,
            lo: mflo,
            hi: mfhi,
        },
        jaccard: MetricPoint {
            point: point.jaccard.point,
            lo: jlo,
            hi: jhi,
        },
        exact_match: MetricPoint {
            point: point.exact_match.point,
            lo: elo,
            hi: ehi,
        },
        empty_p: MetricPoint {
            point: point.empty_p.point,
            lo: emlo,
            hi: emhi,
        },
        types_used: point.types_used,
        supports: point.supports,
        per_type,
    }
}

/// Paired bootstrap interval of the difference (own − donor) for one scalar
/// set metric: the same molecule resamples feed both arms, so the interval
/// measures the ablation on the same spectra. `None` marks an undefined
/// side (zero denominator there).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffPoint {
    /// Point difference on all spectra (`None` when undefined on either side).
    pub point: Option<f64>,
    /// 2.5th percentile of the paired bootstrap differences.
    pub lo: Option<f64>,
    /// 97.5th percentile of the paired bootstrap differences.
    pub hi: Option<f64>,
}

/// Paired own−donor differences for every set-level scalar plus per-type
/// recall, sharing one molecule bootstrap (same spectra, same resamples).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedSetDiff {
    /// Micro precision difference.
    pub micro_precision: DiffPoint,
    /// Micro recall difference.
    pub micro_recall: DiffPoint,
    /// Micro F1 difference.
    pub micro_f1: DiffPoint,
    /// Macro precision difference (held denominator, as in [`SetMetrics`]).
    pub macro_precision: DiffPoint,
    /// Macro recall difference.
    pub macro_recall: DiffPoint,
    /// Macro F1 difference.
    pub macro_f1: DiffPoint,
    /// Mean Jaccard difference.
    pub jaccard: DiffPoint,
    /// Exact-set-match rate difference.
    pub exact_match: DiffPoint,
    /// Empty-`P` fraction difference.
    pub empty_p: DiffPoint,
    /// Per-type recall difference (own − donor), index `id - 1`.
    pub per_type_recall_diff: Vec<DiffPoint>,
}

/// Paired own−donor set differences for one vocabulary.
///
/// `own_preds`/`donor_preds`/`truths`/`molecules` run in parallel (one entry
/// per spectrum); both prediction masks are pre-restricted to the vocabulary
/// under test. The macro denominator is the support-based held set of the
/// shared truths. Resampling reuses the same molecule order and seed
/// convention as [`evaluate_sets`] so paired arms share resamples.
pub fn paired_set_diff(
    own_preds: &[u32],
    donor_preds: &[u32],
    truths: &[u32],
    molecules: &[usize],
    bootstrap: usize,
    seed: u64,
) -> PairedSetDiff {
    assert_eq!(
        own_preds.len(),
        truths.len(),
        "paired_set_diff: own vs truths"
    );
    assert_eq!(
        donor_preds.len(),
        truths.len(),
        "paired_set_diff: donor vs truths"
    );
    assert_eq!(
        molecules.len(),
        truths.len(),
        "paired_set_diff: molecules vs truths"
    );
    let held: Vec<usize> = {
        let mut support = [0usize; N_FG];
        for t in truths {
            for id in 0..N_FG {
                if t & (1u32 << id) != 0 {
                    support[id] += 1;
                }
            }
        }
        (1..=N_FG).filter(|&id| support[id - 1] >= 10).collect()
    };
    // One paired bootstrap: the same resampled spectra feed both arms.
    let mut mol_ids: Vec<usize> = molecules.to_vec();
    mol_ids.sort_unstable();
    mol_ids.dedup();
    let per_mol: Vec<Vec<usize>> = mol_ids
        .iter()
        .map(|m| {
            molecules
                .iter()
                .enumerate()
                .filter_map(|(i, mm)| (*mm == *m).then_some(i))
                .collect()
        })
        .collect();
    let scalar = |pick: &dyn Fn(&[u32], &[u32]) -> Option<f64>| -> DiffPoint {
        let point = match (pick(own_preds, truths), pick(donor_preds, truths)) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        if mol_ids.is_empty() || bootstrap == 0 {
            return DiffPoint {
                point,
                lo: point,
                hi: point,
            };
        }
        let mut rng = SplitMix64::new(seed);
        let m = per_mol.len();
        let mut values: Vec<f64> = Vec::with_capacity(bootstrap);
        for _ in 0..bootstrap {
            let mut op = Vec::new();
            let mut dp = Vec::new();
            let mut bt = Vec::new();
            for _ in 0..m {
                let j = (rng.next() % m as u64) as usize;
                for &i in &per_mol[j] {
                    op.push(own_preds[i]);
                    dp.push(donor_preds[i]);
                    bt.push(truths[i]);
                }
            }
            if let (Some(a), Some(b)) = (pick(&op, &bt), pick(&dp, &bt)) {
                values.push(a - b);
            }
        }
        if values.is_empty() {
            return DiffPoint {
                point,
                lo: None,
                hi: None,
            };
        }
        values.sort_by(|a, b| a.total_cmp(b));
        DiffPoint {
            point,
            lo: Some(percentile_of(&values, 2.5)),
            hi: Some(percentile_of(&values, 97.5)),
        }
    };
    let sp = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).0;
    let sr = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).1;
    let sf = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).2;
    let mp = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).3;
    let mr = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).4;
    let mf = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).5;
    let ja = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).6;
    let ex = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).7;
    let em = |p: &[u32], t: &[u32]| metrics_with_held(p, t, &held).8;
    // Per-type recall differences (the recall denominator is the shared
    // truth support, so both arms are defined together).
    let mut per_type_recall_diff = Vec::with_capacity(N_FG);
    for id in 1..=N_FG {
        let bit = 1u32 << (id - 1);
        let rec = |p: &[u32], t: &[u32]| {
            let mut tp = 0u64;
            let mut tr = 0u64;
            for (a, b) in p.iter().zip(t.iter()) {
                if b & bit != 0 {
                    tr += 1;
                    if a & bit != 0 {
                        tp += 1;
                    }
                }
            }
            if tr == 0 {
                None
            } else {
                Some(tp as f64 / tr as f64)
            }
        };
        let both = |op: &[u32], dp: &[u32], bt: &[u32]| -> Option<f64> {
            match (rec(op, bt), rec(dp, bt)) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            }
        };
        let point = both(own_preds, donor_preds, truths);
        let (lo, hi) = if mol_ids.is_empty() || bootstrap == 0 || point.is_none() {
            (point, point)
        } else {
            let mut rng = SplitMix64::new(seed.wrapping_add(id as u64));
            let m = per_mol.len();
            let mut values: Vec<f64> = Vec::with_capacity(bootstrap);
            for _ in 0..bootstrap {
                let mut op = Vec::new();
                let mut dp = Vec::new();
                let mut bt = Vec::new();
                for _ in 0..m {
                    let j = (rng.next() % m as u64) as usize;
                    for &i in &per_mol[j] {
                        op.push(own_preds[i]);
                        dp.push(donor_preds[i]);
                        bt.push(truths[i]);
                    }
                }
                if let Some(v) = both(&op, &dp, &bt) {
                    values.push(v);
                }
            }
            if values.is_empty() {
                (None, None)
            } else {
                values.sort_by(|a, b| a.total_cmp(b));
                (
                    Some(percentile_of(&values, 2.5)),
                    Some(percentile_of(&values, 97.5)),
                )
            }
        };
        per_type_recall_diff.push(DiffPoint { point, lo, hi });
    }
    PairedSetDiff {
        micro_precision: scalar(&sp),
        micro_recall: scalar(&sr),
        micro_f1: scalar(&sf),
        macro_precision: scalar(&mp),
        macro_recall: scalar(&mr),
        macro_f1: scalar(&mf),
        jaccard: scalar(&ja),
        exact_match: scalar(&ex),
        empty_p: scalar(&em),
        per_type_recall_diff,
    }
}

/// Scalar metrics with a held macro denominator: (micro_p, micro_r,
/// micro_f1, macro_p, macro_r, macro_f1, jaccard, exact, empty). Macro
/// averages treat undefined per-type components as 0, keeping the held
/// denominator fixed; an empty held set yields `None` macro values.
#[allow(clippy::type_complexity)]
fn metrics_with_held(
    preds: &[u32],
    truths: &[u32],
    held: &[usize],
) -> (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
) {
    let n = preds.len();
    let mut inter = 0u64;
    let mut p_tot = 0u64;
    let mut t_tot = 0u64;
    let mut j_sum = 0.0;
    let mut exact = 0usize;
    let mut empty = 0usize;
    let mut true_c = [0usize; N_FG];
    let mut pred_c = [0usize; N_FG];
    let mut tp_c = [0usize; N_FG];
    for (pr, tr) in preds.iter().zip(truths.iter()) {
        let i = (pr & tr).count_ones() as u64;
        inter += i;
        p_tot += pr.count_ones() as u64;
        t_tot += tr.count_ones() as u64;
        let u = (pr | tr).count_ones() as f64;
        j_sum += if u == 0.0 { 1.0 } else { i as f64 / u };
        if pr == tr {
            exact += 1;
        }
        if *pr == 0 {
            empty += 1;
        }
        for id in 0..N_FG {
            let bit = 1u32 << id;
            if tr & bit != 0 {
                true_c[id] += 1;
            }
            if pr & bit != 0 {
                pred_c[id] += 1;
            }
            if pr & tr & bit != 0 {
                tp_c[id] += 1;
            }
        }
    }
    let micro_p = if p_tot == 0 {
        None
    } else {
        Some(inter as f64 / p_tot as f64)
    };
    let micro_r = if t_tot == 0 {
        None
    } else {
        Some(inter as f64 / t_tot as f64)
    };
    let micro_f = f1_of(micro_p, micro_r);
    let (macro_p, macro_r, macro_f) = if held.is_empty() {
        (None, None, None)
    } else {
        let mut sp = 0.0;
        let mut sr = 0.0;
        let mut sf = 0.0;
        for &id in held {
            let tp = tp_c[id - 1] as f64;
            let pr = pred_c[id - 1] as f64;
            let tr = true_c[id - 1] as f64;
            let p = if pr == 0.0 { None } else { Some(tp / pr) };
            let r = if tr == 0.0 { None } else { Some(tp / tr) };
            sp += p.unwrap_or(0.0);
            sr += r.unwrap_or(0.0);
            sf += f1_of(p, r).unwrap_or(0.0);
        }
        let m = held.len() as f64;
        (Some(sp / m), Some(sr / m), Some(sf / m))
    };
    let j = if n == 0 { None } else { Some(j_sum / n as f64) };
    let e = if n == 0 {
        None
    } else {
        Some(exact as f64 / n as f64)
    };
    let em = if n == 0 {
        None
    } else {
        Some(empty as f64 / n as f64)
    };
    (
        micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, j, e, em,
    )
}

/// Point estimates of the set metrics (intervals degenerate to the point).
fn set_point(preds: &[u32], truths: &[u32]) -> SetMetrics {
    let n = preds.len();
    let mut inter = 0u64;
    let mut p_tot = 0u64;
    let mut t_tot = 0u64;
    let mut j_sum = 0.0;
    let mut exact = 0usize;
    let mut empty = 0usize;
    let mut true_c = [0usize; N_FG];
    let mut pred_c = [0usize; N_FG];
    let mut tp_c = [0usize; N_FG];
    for (pr, tr) in preds.iter().zip(truths.iter()) {
        let i = (pr & tr).count_ones() as u64;
        let p = pr.count_ones() as u64;
        let t = tr.count_ones() as u64;
        inter += i;
        p_tot += p;
        t_tot += t;
        let u = (pr | tr).count_ones() as f64;
        j_sum += if u == 0.0 { 1.0 } else { i as f64 / u };
        if pr == tr {
            exact += 1;
        }
        if *pr == 0 {
            empty += 1;
        }
        for id in 0..N_FG {
            let bit = 1u32 << id;
            if tr & bit != 0 {
                true_c[id] += 1;
            }
            if pr & bit != 0 {
                pred_c[id] += 1;
            }
            if pr & tr & bit != 0 {
                tp_c[id] += 1;
            }
        }
    }
    let micro_p = if p_tot == 0 {
        None
    } else {
        Some(inter as f64 / p_tot as f64)
    };
    let micro_r = if t_tot == 0 {
        None
    } else {
        Some(inter as f64 / t_tot as f64)
    };
    let used: Vec<usize> = (1..=N_FG).filter(|&id| true_c[id - 1] >= 10).collect();
    let (macro_p, macro_r, macro_f) = if used.is_empty() {
        (None, None, None)
    } else {
        let mut sp = 0.0;
        let mut sr = 0.0;
        let mut sf = 0.0;
        for &id in &used {
            let tp = tp_c[id - 1] as f64;
            let pr = pred_c[id - 1] as f64;
            let tr = true_c[id - 1] as f64;
            let p = if pr == 0.0 { None } else { Some(tp / pr) };
            let r = if tr == 0.0 { None } else { Some(tp / tr) };
            sp += p.unwrap_or(0.0);
            sr += r.unwrap_or(0.0);
            sf += f1_of(p, r).unwrap_or(0.0);
        }
        let m = used.len() as f64;
        (Some(sp / m), Some(sr / m), Some(sf / m))
    };
    let per_type: Vec<TypeRow> = (1..=N_FG)
        .map(|id| {
            let tp = tp_c[id - 1];
            let pr = pred_c[id - 1];
            let tr = true_c[id - 1];
            TypeRow {
                id,
                name: FG_NAMES[id - 1].to_string(),
                true_count: tr,
                predicted: pr,
                tp,
                precision: if pr == 0 {
                    None
                } else {
                    Some(tp as f64 / pr as f64)
                },
                // Point-only construction: intervals are filled by
                // `evaluate_sets` (molecule bootstrap) and the donor paired
                // recall difference by the paired evaluation.
                precision_lo: None,
                precision_hi: None,
                recall: if tr == 0 {
                    None
                } else {
                    Some(tp as f64 / tr as f64)
                },
                recall_lo: None,
                recall_hi: None,
                recall_diff_lo: None,
                recall_diff_hi: None,
            }
        })
        .collect();
    let one = |v: Option<f64>| MetricPoint {
        point: v,
        lo: v,
        hi: v,
    };
    SetMetrics {
        micro_precision: one(micro_p),
        micro_recall: one(micro_r),
        micro_f1: one(f1_of(micro_p, micro_r)),
        macro_precision: one(macro_p),
        macro_recall: one(macro_r),
        macro_f1: one(macro_f),
        types_used: used,
        supports: true_c.to_vec(),
        per_type,
        jaccard: one(if n == 0 { None } else { Some(j_sum / n as f64) }),
        exact_match: one(if n == 0 {
            None
        } else {
            Some(exact as f64 / n as f64)
        }),
        empty_p: one(if n == 0 {
            None
        } else {
            Some(empty as f64 / n as f64)
        }),
    }
}

/// The `p`th percentile by linear interpolation (the same rule as
/// [`super::metrics`]).
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

/// Deterministic 64-bit generator for the bootstrap (SplitMix64, the same
/// constants as [`super::metrics`]).
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

/// Candidate-size distribution over the eligible candidates of `data`.
pub fn candidate_size_dist(data: &[FgSpectrumDatum]) -> CandidateSizes {
    let mut n = 0usize;
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    let mut c35 = 0usize;
    let mut c69 = 0usize;
    let mut c1016 = 0usize;
    for d in data {
        for c in &d.candidates {
            // Replay failures carry `atoms == 0` and sit outside every bin;
            // they still count in `n` (they occupy top-k slots as empty
            // predictions) but in no size fraction.
            n += 1;
            match c.atoms {
                1 => c1 += 1,
                2 => c2 += 1,
                3..=5 => c35 += 1,
                6..=9 => c69 += 1,
                10..=16 => c1016 += 1,
                _ => {}
            }
        }
    }
    let frac = |k: usize| {
        if n == 0 {
            None
        } else {
            Some(k as f64 / n as f64)
        }
    };
    CandidateSizes {
        n,
        frac_1: frac(c1),
        frac_2: frac(c2),
        frac_3_5: frac(c35),
        frac_6_9: frac(c69),
        frac_10_16: frac(c1016),
    }
}

/// Evaluate model data at every `k` in `ks`: `P_k` is the union over the
/// first `k` eligible candidates in raw-score order, for the full, specific
/// and heteroatom vocabularies, each also restricted to candidates of at
/// least 3 atoms (`min_atoms_3`).
///
/// Instance and candidate levels pool ALL eligible candidates (not just the
/// top `k`). Returns one [`KReport`] per `k`, in `ks` order.
///
/// `donor` holds the donor-peaks ablation data (same spectra, same order as
/// `data`): each spectrum's peaks replaced by a donor spectrum's from another
/// molecule. When present, every report carries the donor vocabularies plus
/// paired own−donor bootstrap intervals for every set metric, and each
/// per-type row of the own arm gains the paired recall-difference interval.
pub fn evaluate_fg(
    data: &[FgSpectrumDatum],
    donor: Option<&[FgSpectrumDatum]>,
    ks: &[usize],
    bootstrap: usize,
    seed: u64,
) -> Vec<KReport> {
    if let Some(d) = donor {
        assert_eq!(d.len(), data.len(), "evaluate_fg: donor vs data spectra");
    }
    let molecules: Vec<usize> = data.iter().map(|d| d.molecule).collect();
    let sizes = candidate_size_dist(data);
    let mut out = Vec::with_capacity(ks.len());
    for &k in ks {
        let mk = |rows: &[FgSpectrumDatum], vocab: u32, min3: bool| -> Vec<u32> {
            rows.iter()
                .map(|d| {
                    d.candidates
                        .iter()
                        .take(k)
                        .filter(|c| !min3 || c.atoms >= 3)
                        .fold(0u32, |m, c| m | (c.mask() & vocab))
                })
                .collect()
        };
        let t = |vocab: u32| -> Vec<u32> { data.iter().map(|d| d.parent_mask & vocab).collect() };
        let full_p = mk(data, FULL_MASK, false);
        let full_p3 = mk(data, FULL_MASK, true);
        let spec_p = mk(data, SPECIFIC_MASK, false);
        let spec_p3 = mk(data, SPECIFIC_MASK, true);
        let het_p = mk(data, HETEROATOM_MASK, false);
        let het_p3 = mk(data, HETEROATOM_MASK, true);
        let full_t = t(FULL_MASK);
        let spec_t = t(SPECIFIC_MASK);
        let het_t = t(HETEROATOM_MASK);
        let sized = || CandidateSizes {
            n: sizes.n,
            frac_1: sizes.frac_1,
            frac_2: sizes.frac_2,
            frac_3_5: sizes.frac_3_5,
            frac_6_9: sizes.frac_6_9,
            frac_10_16: sizes.frac_10_16,
        };
        let mut full = vocab_report(
            &full_p, &full_p3, &full_t, data, FULL_MASK, &molecules, bootstrap, seed,
        );
        let mut specific = vocab_report(
            &spec_p,
            &spec_p3,
            &spec_t,
            data,
            SPECIFIC_MASK,
            &molecules,
            bootstrap,
            seed,
        );
        let mut heteroatom = vocab_report(
            &het_p,
            &het_p3,
            &het_t,
            data,
            HETEROATOM_MASK,
            &molecules,
            bootstrap,
            seed,
        );
        let donor_rep = donor.map(|dd| {
            let dfull_p = mk(dd, FULL_MASK, false);
            let dfull_p3 = mk(dd, FULL_MASK, true);
            let dspec_p = mk(dd, SPECIFIC_MASK, false);
            let dspec_p3 = mk(dd, SPECIFIC_MASK, true);
            let dhet_p = mk(dd, HETEROATOM_MASK, false);
            let dhet_p3 = mk(dd, HETEROATOM_MASK, true);
            let dfull = vocab_report(
                &dfull_p, &dfull_p3, &full_t, dd, FULL_MASK, &molecules, bootstrap, seed,
            );
            let dspec = vocab_report(
                &dspec_p,
                &dspec_p3,
                &spec_t,
                dd,
                SPECIFIC_MASK,
                &molecules,
                bootstrap,
                seed,
            );
            let dhet = vocab_report(
                &dhet_p,
                &dhet_p3,
                &het_t,
                dd,
                HETEROATOM_MASK,
                &molecules,
                bootstrap,
                seed,
            );
            let diff_full =
                paired_set_diff(&full_p, &dfull_p, &full_t, &molecules, bootstrap, seed);
            let diff_specific =
                paired_set_diff(&spec_p, &dspec_p, &spec_t, &molecules, bootstrap, seed);
            let diff_heteroatom =
                paired_set_diff(&het_p, &dhet_p, &het_t, &molecules, bootstrap, seed);
            (
                dfull,
                dspec,
                dhet,
                diff_full,
                diff_specific,
                diff_heteroatom,
            )
        });
        // Copy paired per-type recall differences into the own arm's rows.
        if let Some((_, _, _, ref diff_full, ref diff_specific, ref diff_heteroatom)) = donor_rep {
            for (row, d) in full
                .set
                .per_type
                .iter_mut()
                .zip(diff_full.per_type_recall_diff.iter())
            {
                row.recall_diff_lo = d.lo;
                row.recall_diff_hi = d.hi;
            }
            for (row, d) in specific
                .set
                .per_type
                .iter_mut()
                .zip(diff_specific.per_type_recall_diff.iter())
            {
                row.recall_diff_lo = d.lo;
                row.recall_diff_hi = d.hi;
            }
            for (row, d) in heteroatom
                .set
                .per_type
                .iter_mut()
                .zip(diff_heteroatom.per_type_recall_diff.iter())
            {
                row.recall_diff_lo = d.lo;
                row.recall_diff_hi = d.hi;
            }
        }
        out.push(KReport {
            k,
            full,
            specific,
            heteroatom,
            sizes: sized(),
            donor: donor_rep.map(
                |(dfull, dspec, dhet, diff_full, diff_specific, diff_heteroatom)| DonorReport {
                    full: dfull,
                    specific: dspec,
                    heteroatom: dhet,
                    diff_full,
                    diff_specific,
                    diff_heteroatom,
                },
            ),
        });
    }
    out
}

/// One vocabulary's model report: set metrics (plus the `min_atoms_3`
/// restriction) plus instance/candidate levels pooled over all eligible
/// candidates.
#[allow(clippy::too_many_arguments)]
fn vocab_report(
    preds: &[u32],
    preds_min3: &[u32],
    truths: &[u32],
    data: &[FgSpectrumDatum],
    vocab: u32,
    molecules: &[usize],
    bootstrap: usize,
    seed: u64,
) -> VocabReport {
    let set = evaluate_sets(preds, truths, molecules, bootstrap, seed);
    let set_min_atoms_3 = evaluate_sets(preds_min3, truths, molecules, bootstrap, seed);
    // Per-candidate molecule assignment for the bootstrap.
    let mut cand_mol: Vec<usize> = Vec::new();
    for d in data {
        for _ in &d.candidates {
            cand_mol.push(d.molecule);
        }
    }
    let mut mol_ids = cand_mol.clone();
    mol_ids.sort_unstable();
    mol_ids.dedup();
    let per_mol: Vec<Vec<usize>> = mol_ids
        .iter()
        .map(|m| {
            cand_mol
                .iter()
                .enumerate()
                .filter_map(|(i, mm)| (*mm == *m).then_some(i))
                .collect()
        })
        .collect();
    let boot = |stat: &dyn Fn(&[usize]) -> Option<f64>| -> MetricPoint {
        let point = stat(&(0..cand_mol.len()).collect::<Vec<usize>>());
        if mol_ids.is_empty() || bootstrap == 0 {
            return MetricPoint {
                point,
                lo: point,
                hi: point,
            };
        }
        let mut rng = SplitMix64::new(seed.wrapping_add(0x1234_5678_9ABC_DEF0));
        let m = per_mol.len();
        let mut values: Vec<f64> = Vec::with_capacity(bootstrap);
        for _ in 0..bootstrap {
            let mut idx = Vec::new();
            for _ in 0..m {
                let j = (rng.next() % m as u64) as usize;
                idx.extend_from_slice(&per_mol[j]);
            }
            if let Some(v) = stat(&idx) {
                values.push(v);
            }
        }
        if values.is_empty() {
            return MetricPoint {
                point,
                lo: None,
                hi: None,
            };
        }
        values.sort_by(|a, b| a.total_cmp(b));
        MetricPoint {
            point,
            lo: Some(percentile_of(&values, 2.5)),
            hi: Some(percentile_of(&values, 97.5)),
        }
    };
    let flat: Vec<&FgCandidate> = data.iter().flat_map(|d| &d.candidates).collect();
    let truth_of = |i: usize| -> u32 {
        // Candidate `i` in flat order belongs to a spectrum; recover its T.
        let mut acc = 0usize;
        for d in data {
            if i < acc + d.candidates.len() {
                return d.parent_mask & vocab;
            }
            acc += d.candidates.len();
        }
        0
    };
    let ip = boot(&|idx: &[usize]| {
        let mut inter = 0u64;
        let mut tot = 0u64;
        for &i in idx {
            let c = flat[i];
            for t in 0..N_FG {
                if vocab & (1u32 << t) == 0 {
                    continue;
                }
                let n = u64::from(c.counts[t]);
                tot += n;
                if truth_of(i) & (1u32 << t) != 0 {
                    inter += n;
                }
            }
        }
        if tot == 0 {
            None
        } else {
            Some(inter as f64 / tot as f64)
        }
    });
    let mp = boot(&|idx: &[usize]| {
        if idx.is_empty() {
            return Some(0.0);
        }
        let s: u64 = idx
            .iter()
            .map(|&i| u64::from(flat[i].instances_in(vocab)))
            .sum();
        Some(s as f64 / idx.len() as f64)
    });
    let up = boot(&|idx: &[usize]| {
        if idx.is_empty() {
            return Some(0.0);
        }
        let s: u64 = idx
            .iter()
            .map(|&i| (flat[i].undet & vocab).count_ones() as u64)
            .sum();
        Some(s as f64 / idx.len() as f64)
    });
    let wp = boot(&|idx: &[usize]| {
        if idx.is_empty() {
            return Some(0.0);
        }
        let w = idx
            .iter()
            .filter(|&&i| flat[i].instances_in(vocab) > 0)
            .count();
        Some(w as f64 / idx.len() as f64)
    });
    let ap = boot(&|idx: &[usize]| {
        let with: Vec<usize> = idx
            .iter()
            .copied()
            .filter(|&i| flat[i].instances_in(vocab) > 0)
            .collect();
        if with.is_empty() {
            return None;
        }
        let good = with
            .iter()
            .filter(|&&i| flat[i].mask() & vocab & !truth_of(i) == 0)
            .count();
        Some(good as f64 / with.len() as f64)
    });
    VocabReport {
        set,
        set_min_atoms_3,
        instance_precision: ip,
        mean_instances: mp,
        mean_undet: up,
        cand_with_group: wp,
        cand_all_real: ap,
    }
}

/// Number of determined parent group instances for which the closing-fragment
/// search found no determined fragment of at most 16 atoms (a search
/// failure, not a proof).
///
/// Reported as `closing_fragment_not_found`: instances for which the staged
/// search found no witness — a search failure, not a proof that no such
/// fragment exists.
pub fn closing_fragment_not_found(parents: &[MolGraph]) -> usize {
    let mut bad = 0usize;
    for parent in parents {
        for (id, anchor) in fg_instances(parent) {
            if !closable(parent, &anchor, id) {
                bad += 1;
            }
        }
    }
    bad
}

/// Whether the determined instance (`id` on `anchor` atoms of `parent`)
/// is determined in some connected induced fragment of at most 16 atoms
/// containing the anchor.
///
/// The search tries, in order: (i) the pattern atoms plus every neighbour
/// (a superset of the neighbours the exclusions consult); (ii) stage (i)
/// plus every ring of at most 6 atoms containing an atom within distance 1
/// of the pattern (completing rings the five-ring and arene rules consult);
/// (iii) stage (ii) plus the whole π-graph components of the consulted bonds
/// (bonds incident to the pattern atoms) when they fit in 16 atoms; and
/// finally the legacy whole-layer expansion. A negative answer is a search
/// failure, not a proof (see [`closing_fragment_not_found`]).
fn closable(parent: &MolGraph, anchor: &[usize], id: usize) -> bool {
    use std::collections::BTreeSet;
    let n = parent.atoms().len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (a, b, _) in parent.bonds() {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    // Stage (i): pattern atoms plus every neighbour.
    let anchor_set: BTreeSet<usize> = anchor.iter().copied().collect();
    let mut stage1 = anchor_set.clone();
    for u in anchor_set.iter() {
        for v in &adj[*u] {
            stage1.insert(*v);
        }
    }
    // Stage (ii): plus every ring of at most 6 atoms through distance ≤ 1.
    let mut near: BTreeSet<usize> = anchor_set.clone();
    for u in anchor_set.iter() {
        for v in &adj[*u] {
            near.insert(*v);
        }
    }
    let mut stage2 = stage1.clone();
    for r in small_rings_through(&adj, &near, 6, 2000) {
        for a in r {
            stage2.insert(a);
        }
    }
    // Stage (iii): plus whole π-graph components of consulted bonds (bonds
    // incident to the pattern atoms) when they fit.
    let mut stage3 = stage2.clone();
    let mut consulted_h: BTreeSet<usize> = BTreeSet::new();
    {
        use super::functional_groups::pi_graph_components;
        // Atom -> component index.
        let comps = pi_graph_components(parent);
        let mut comp_of: Vec<Option<usize>> = vec![None; n];
        for (ci, c) in comps.iter().enumerate() {
            for a in c {
                comp_of[*a] = Some(ci);
            }
        }
        let mut need_comps: BTreeSet<usize> = BTreeSet::new();
        for u in anchor_set.iter() {
            for v in &adj[*u] {
                for x in [*u, *v] {
                    if let Some(ci) = comp_of[x] {
                        need_comps.insert(ci);
                    }
                }
            }
        }
        for ci in need_comps {
            for a in &comps[ci] {
                consulted_h.insert(*a);
            }
        }
    }
    let mut fits = true;
    for a in consulted_h.iter() {
        stage3.insert(*a);
        if stage3.len() > 16 {
            fits = false;
            break;
        }
    }
    // Try the stages in order (skipping stage (iii) when it does not fit),
    // then fall back to legacy whole-layer expansion.
    for members in [stage1, stage2] {
        if determined_in(parent, &members, anchor, id) {
            return true;
        }
    }
    if fits && determined_in(parent, &stage3, anchor, id) {
        return true;
    }
    // Legacy fallback: whole neighbouring layers up to 16 atoms.
    let mut inside: BTreeSet<usize> = anchor.iter().copied().collect();
    loop {
        let members: Vec<usize> = inside.iter().copied().collect();
        if members.len() <= 16 && determined_in(parent, &inside, anchor, id) {
            return true;
        }
        if inside.len() >= 16 {
            return false;
        }
        // Expand by one neighbouring layer.
        let mut next = inside.clone();
        for u in inside.iter() {
            for v in &adj[*u] {
                next.insert(*v);
                if next.len() > 32 {
                    break;
                }
            }
            if next.len() > 32 {
                break;
            }
        }
        if next == inside {
            return false;
        }
        // Cap the search at 17 atoms (one past the limit proves failure).
        let capped: Vec<usize> = next.into_iter().take(17).collect();
        if capped.len() > 16 {
            // One more determination attempt on the first-16 truncation is
            // skipped: exceeding 16 already fails the "at most 16" check.
            return false;
        }
        inside = capped.into_iter().collect();
    }
}

/// Whether the instance (`id` on `anchor` of `parent`) is determined in the
/// induced fragment on `members` (which must contain the anchor and hold at
/// most 16 atoms).
fn determined_in(
    parent: &MolGraph,
    members: &std::collections::BTreeSet<usize>,
    anchor: &[usize],
    id: usize,
) -> bool {
    if members.len() > 16 {
        return false;
    }
    let members_vec: Vec<usize> = members.iter().copied().collect();
    let Ok(sub) = parent.induced(&members_vec) else {
        return false;
    };
    // Remap the anchor into fragment coordinates.
    let mut pos: Vec<usize> = Vec::with_capacity(anchor.len());
    for a in anchor {
        match members_vec.iter().position(|x| x == a) {
            Some(p) => pos.push(p),
            None => return false,
        }
    }
    pos.sort_unstable();
    fg_instances(&sub)
        .into_iter()
        .any(|(i2, a2)| i2 == id && a2 == pos)
}

/// Simple rings (atom lists, without closure repetition) of length at most
/// `max_len` through any seed atom, deduplicated by sorted atom set, capped
/// at `cap` rings (a search aid: incompleteness only risks a search failure,
/// never a wrong verdict).
fn small_rings_through(
    adj: &[Vec<usize>],
    seeds: &std::collections::BTreeSet<usize>,
    max_len: usize,
    cap: usize,
) -> Vec<Vec<usize>> {
    use std::collections::BTreeSet;
    let n = adj.len();
    let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut out: Vec<Vec<usize>> = Vec::new();
    fn rec(
        adj: &[Vec<usize>],
        start: usize,
        cur: usize,
        path: &mut Vec<usize>,
        vis: &mut [bool],
        max_len: usize,
        cap: usize,
        seen: &mut BTreeSet<Vec<usize>>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if out.len() >= cap {
            return;
        }
        for &nxt in &adj[cur] {
            if out.len() >= cap {
                return;
            }
            if nxt == start {
                if path.len() >= 3 && path.len() <= max_len {
                    let mut key = path.clone();
                    key.sort_unstable();
                    if seen.insert(key) {
                        out.push(path.clone());
                    }
                }
                continue;
            }
            // No smallest-atom canonicalisation: seeds may not hold the
            // ring's smallest atom, and duplicates are removed by set.
            if vis[nxt] || path.len() >= max_len {
                continue;
            }
            vis[nxt] = true;
            path.push(nxt);
            rec(adj, start, nxt, path, vis, max_len, cap, seen, out);
            path.pop();
            vis[nxt] = false;
        }
    }
    for s in seeds.iter() {
        if out.len() >= cap {
            break;
        }
        let mut path = vec![*s];
        let mut vis = vec![false; n];
        vis[*s] = true;
        rec(
            adj, *s, *s, &mut path, &mut vis, max_len, cap, &mut seen, &mut out,
        );
    }
    out
}
