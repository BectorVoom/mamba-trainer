//! Evaluation metrics of contracts §10 (V0.5 prerequisites).
//!
//! [`evaluate_candidates`] replays each finished candidate trace, decides
//! induced containment against the parent and matches targets by canonical
//! trace. [`summarize`] aggregates per spectrum, then per molecule, then over
//! molecules, with percentile bootstrap intervals over molecules.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::contain::{Containment, contains_induced};
use super::contract::{CandidateBatch, candidate_status};
use super::experiment::{ExperimentSet, SpectrumDomain};
use super::grammar::{
    ADD_ATOM, CANONICAL_WORK_LIMIT, Limits, STOP, Token, canonical_trace, replay,
};

// ---------------------------------------------------------------------------
// Evaluation records
// ---------------------------------------------------------------------------

/// One candidate of one spectrum, as evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEval {
    /// The trace ended with STOP (the batch `finished` bit).
    pub finished: bool,
    /// The finished trace replays legally under the parent-formula budget.
    pub valid: bool,
    /// A finished candidate with the same raw trace and formula row as an
    /// earlier finished candidate of the spectrum (excluded from precision).
    pub duplicate: bool,
    /// Atoms of the replayed graph (0 when there is none).
    pub atoms: usize,
    /// Induced containment against the parent (`WorkLimit` counts as not
    /// contained and is reported as its own rate).
    pub contained: Containment,
    /// Canonical trace of the replayed graph (`None` when there is none or
    /// canonicalization fails).
    pub canonical: Option<Vec<Token>>,
}

/// One spectrum's evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumEval {
    /// Molecule index (into the experiment set's molecule list).
    pub molecule: usize,
    /// The spectrum's domain.
    pub domain: SpectrumDomain,
    /// One entry per considered candidate (in trajectory order).
    pub candidates: Vec<CandidateEval>,
    /// True when the spectrum has no finished candidate at all.
    pub abstained: bool,
    /// Whether the gold formula is among the retained hypotheses. Always
    /// `None` here: without the formula table the row ids of
    /// [`CandidateBatch::formula_row`] cannot be mapped to compositions, so
    /// recall needs the table and is set by the caller when it is available.
    pub formula_recall: Option<bool>,
    /// `q` mass of the spectrum's targets whose canonical trace is among the
    /// finished candidates (0 for unlabeled and out-of-domain spectra, which
    /// are misses in the full-dataset denominator).
    pub q_found: f64,
    /// Per size stratum (targets of 3–5, 6–9, 10–16 atoms): the `q` mass of
    /// the stratum's targets found among the finished candidates.
    pub q_found_by_stratum: [f64; 3],
    /// Per size stratum: the total `q` mass of the stratum's targets (0 for
    /// unlabeled and out-of-domain spectra).
    pub q_total_by_stratum: [f64; 3],
}

// ---------------------------------------------------------------------------
// Candidate evaluation
// ---------------------------------------------------------------------------

/// Replay each finished candidate trace, decide containment and match targets.
///
/// Each finished trace is replayed with [`replay`] under the parent-formula
/// budget (`Some(parent composition)`), so a candidate that exceeds the
/// parent composition is invalid. Its graph is checked with
/// [`contains_induced`] against the parent under `work_limit` and
/// canonicalized; targets match by canonical trace. Unfinished traces get
/// `finished = false`, `valid = false`, no graph and `NotContained`.
pub fn evaluate_candidates(
    set: &ExperimentSet,
    indices: &[usize],
    batch: &CandidateBatch,
    work_limit: usize,
) -> Result<Vec<SpectrumEval>> {
    if indices.len() != batch.batch {
        return Err(Error::config(format!(
            "evaluate_candidates: {} indices for a batch of {} spectra",
            indices.len(),
            batch.batch
        )));
    }
    let limits = Limits::new(batch.max_atoms, batch.max_ring_closures)
        .map_err(|e| Error::config(format!("evaluate_candidates: batch limits rejected: {e}")))?;
    let k = batch.trajectories;
    let t = batch.max_steps;
    let mut out: Vec<SpectrumEval> = Vec::with_capacity(indices.len());
    for (b, &idx) in indices.iter().enumerate() {
        let entry = set.spectra.get(idx).ok_or_else(|| {
            Error::config(format!(
                "evaluate_candidates: spectrum index {idx} outside {} spectra",
                set.spectra.len()
            ))
        })?;
        let expect = entry.spectrum.spectrum_id;
        for kk in 0..k {
            let r = b * k + kk;
            if batch.spectrum_id[r] != expect {
                return Err(Error::config(format!(
                    "evaluate_candidates: spectrum {b} record {kk} id {} is not the set id {expect}",
                    batch.spectrum_id[r]
                )));
            }
        }
        // Raw traces and formula rows for duplicate detection (same raw trace
        // and formula row as an earlier finished candidate).
        let mut raw_of: Vec<Option<Vec<Token>>> = Vec::with_capacity(k);
        let mut finished_of: Vec<bool> = Vec::with_capacity(k);
        for kk in 0..k {
            let r = b * k + kk;
            let finished = batch.status[r] & candidate_status::FINISHED != 0;
            finished_of.push(finished);
            if !finished {
                raw_of.push(None);
                continue;
            }
            let len = batch.length[r] as usize;
            if len > t {
                return Err(Error::config(format!(
                    "evaluate_candidates: spectrum {b} candidate {kk} length {len} past T={t}"
                )));
            }
            let base = r * t * 4;
            let mut raw: Vec<Token> = Vec::with_capacity(len);
            for step in 0..len {
                let f = &batch.actions[base + step * 4..base + step * 4 + 4];
                raw.push(Token {
                    kind: u8::try_from(f[0]).map_err(|_| {
                        Error::config(format!(
                            "evaluate_candidates: spectrum {b} candidate {kk} step {step} kind {} past u8",
                            f[0]
                        ))
                    })?,
                    atom_type: u8::try_from(f[1]).map_err(|_| {
                        Error::config(format!(
                            "evaluate_candidates: spectrum {b} candidate {kk} step {step} type {} past u8",
                            f[1]
                        ))
                    })?,
                    bond: u8::try_from(f[2]).map_err(|_| {
                        Error::config(format!(
                            "evaluate_candidates: spectrum {b} candidate {kk} step {step} bond {} past u8",
                            f[2]
                        ))
                    })?,
                    pointer: u8::try_from(f[3]).map_err(|_| {
                        Error::config(format!(
                            "evaluate_candidates: spectrum {b} candidate {kk} step {step} pointer {} past u8",
                            f[3]
                        ))
                    })?,
                });
            }
            raw_of.push(Some(raw));
        }
        let mut candidates: Vec<CandidateEval> = Vec::with_capacity(k);
        let mut seen: BTreeSet<(Vec<Token>, u32)> = BTreeSet::new();
        for kk in 0..k {
            let r = b * k + kk;
            let finished = finished_of[kk];
            if !finished {
                candidates.push(CandidateEval {
                    finished: false,
                    valid: false,
                    duplicate: false,
                    atoms: 0,
                    contained: Containment::NotContained,
                    canonical: None,
                });
                continue;
            }
            let raw = raw_of[kk].clone().expect("finished has a raw trace");
            let key = (raw.clone(), batch.formula_row[r]);
            let duplicate = !seen.insert(key);
            // Replay under the parent budget; any failure is invalid with no
            // graph. The empty placeholder parent of out-of-domain molecules
            // has the zero composition, so every non-empty candidate is
            // invalid there, which is the intended miss.
            let replayed = replay(&raw, limits, Some(entry.parent_composition));
            let (valid, atoms, contained, canonical) = match replayed {
                Ok(state) if state.stopped() && state.atoms() > 0 => {
                    let graph = match state.graph() {
                        Ok(g) => g,
                        Err(_) => {
                            candidates.push(CandidateEval {
                                finished: true,
                                valid: false,
                                duplicate,
                                atoms: 0,
                                contained: Containment::NotContained,
                                canonical: None,
                            });
                            continue;
                        }
                    };
                    let atoms = graph.atoms().len();
                    let contained = contains_induced(&entry.parent, &graph, work_limit);
                    let canonical = canonical_trace(&graph, limits, CANONICAL_WORK_LIMIT)
                        .ok()
                        .map(|c| c.trace);
                    (true, atoms, contained, canonical)
                }
                _ => (false, 0, Containment::NotContained, None),
            };
            candidates.push(CandidateEval {
                finished: true,
                valid,
                duplicate,
                atoms,
                contained,
                canonical,
            });
        }
        let abstained = !candidates.iter().any(|c| c.finished);
        // Target coverage: the q mass of targets whose canonical trace is
        // among the finished candidates' canonical traces, overall and per
        // size stratum (by the target's atom count: ADD_ATOM tokens in its
        // canonical trace). A target outside 3–16 atoms (never produced by
        // the recipe, which keeps 3–16 atoms) counts in `q_found` but in no
        // stratum.
        let (q_found, q_found_by_stratum, q_total_by_stratum) = match &entry.labels {
            None => (0.0, [0.0; 3], [0.0; 3]),
            Some(labels) => {
                let have: BTreeSet<Vec<Token>> = candidates
                    .iter()
                    .filter(|c| c.finished)
                    .filter_map(|c| c.canonical.clone())
                    .collect();
                let mut found = 0.0;
                let mut found_s = [0.0; 3];
                let mut total_s = [0.0; 3];
                for target in &labels.targets {
                    let atoms = target.trace.iter().filter(|t| t.kind == ADD_ATOM).count();
                    let hit = have.contains(&target.trace);
                    if hit {
                        found += target.q;
                    }
                    if let Some(s) = stratum(atoms) {
                        total_s[s] += target.q;
                        if hit {
                            found_s[s] += target.q;
                        }
                    }
                }
                (found, found_s, total_s)
            }
        };
        out.push(SpectrumEval {
            molecule: entry.molecule,
            domain: entry.domain.clone(),
            candidates,
            abstained,
            formula_recall: None,
            q_found,
            q_found_by_stratum,
            q_total_by_stratum,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// One point estimate with its 95% percentile bootstrap interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// Mean over molecules of per-molecule means.
    pub point: f64,
    /// 2.5th percentile of the bootstrap means.
    pub lo: f64,
    /// 97.5th percentile of the bootstrap means.
    pub hi: f64,
}

/// One metric overall and per size stratum (3–5, 6–9, 10–16 atoms).
///
/// Only metrics with a per-candidate or per-target size use this: precision,
/// coverage (and conditional coverage), size-aware precision and the
/// work-limit rate. Candidate-level strata restrict to candidates whose
/// atom count falls in the range. Coverage strata use the per-stratum `q`
/// masses of [`SpectrumEval`]: stratum `s` of a spectrum is
/// `q_found_by_stratum[s] / q_total_by_stratum[s]` (undefined when the
/// spectrum has no target in the stratum). Spectrum-level flags (formula
/// recall, abstention) use [`FlagMetric`] (null strata); validity and
/// uniqueness report the overall value only (see their fields).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stratified {
    /// All sizes.
    pub overall: Interval,
    /// Candidates of 3–5 atoms.
    pub s3_5: Interval,
    /// Candidates of 6–9 atoms.
    pub s6_9: Interval,
    /// Candidates of 10–16 atoms.
    pub s10_16: Interval,
}

/// One spectrum-level flag metric: an overall value with explicitly null
/// size strata.
///
/// Formula recall (gold formula among the retained hypotheses or not) and
/// abstention (no finished candidate or not) are per-spectrum flags with no
/// candidate size, so copying the overall value into every size stratum
/// would fabricate size information. The strata serialize as `null`;
/// readers must use `overall`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlagMetric {
    /// All sizes.
    pub overall: Interval,
    /// Always `None`: a flag has no 3–5-atom value.
    pub s3_5: Option<Interval>,
    /// Always `None`: a flag has no 6–9-atom value.
    pub s6_9: Option<Interval>,
    /// Always `None`: a flag has no 10–16-atom value.
    pub s10_16: Option<Interval>,
}

impl FlagMetric {
    /// A flag metric from its overall interval; every stratum is `None`
    /// (serializes as `null`) because a flag has no size.
    fn of(overall: Interval) -> Self {
        Self {
            overall,
            s3_5: None,
            s6_9: None,
            s10_16: None,
        }
    }
}

/// Metrics at K with bootstrap intervals (contracts §10).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Spectra evaluated.
    pub n_spectra: usize,
    /// Molecules evaluated.
    pub n_molecules: usize,
    /// Candidates considered per spectrum.
    pub k: usize,
    /// Fraction of finished, distinct-by-trace candidates contained in the
    /// parent (`WorkLimit` counts as not contained).
    pub precision: Stratified,
    /// Target `q` mass found (full denominator: out-of-domain and unlabeled
    /// spectra are misses at 0).
    pub coverage: Stratified,
    /// Target `q` mass found among in-domain labeled spectra only (reported
    /// next to the full value, never instead).
    pub coverage_conditional: Stratified,
    /// Fraction of considered candidates finished and valid.
    ///
    /// Overall only, with no size strata: only finished traces have an atom
    /// count (unfinished candidates carry atoms 0, outside every stratum),
    /// so strata would divide finished-and-valid by finished-in-range and
    /// read near 1.0 while the overall is lower. The strata answer a
    /// different question, so they are dropped rather than reported.
    pub validity: Interval,
    /// Distinct finished traces over finished candidates.
    ///
    /// Overall only, for the same reason as `validity`: unfinished
    /// candidates have no size, so strata would exclude them and read
    /// higher than the overall.
    pub uniqueness: Interval,
    /// Gold formula among the retained hypotheses (spectra with `None` are
    /// excluded).
    ///
    /// A per-spectrum flag with no size: the overall value with null strata.
    pub formula_recall: FlagMetric,
    /// Contained candidates weighted by `atoms / 16`, over distinct finished.
    pub size_aware_precision: Stratified,
    /// Fraction of spectra with no finished candidate.
    ///
    /// A per-spectrum flag with no size: the overall value with null strata.
    pub abstention: FlagMetric,
    /// Fraction of distinct finished candidates hitting `WorkLimit`.
    pub worklimit_rate: Stratified,
}

/// Size stratum of an atom count: 3–5, 6–9, 10–16; `None` outside.
fn stratum(atoms: usize) -> Option<usize> {
    match atoms {
        3..=5 => Some(0),
        6..=9 => Some(1),
        10..=16 => Some(2),
        _ => None,
    }
}

/// Aggregate [`SpectrumEval`]s at K.
///
/// Per spectrum the first `k` candidates are considered; duplicates (same raw
/// trace and formula as an earlier finished candidate) are excluded from
/// precision, size-aware precision and the work-limit rate. Per-molecule
/// values are means over that molecule's spectra with defined values;
/// overall points are means over molecules. Intervals are percentile
/// bootstraps over molecules (`bootstrap` resamples with replacement, seeded
/// by `seed`; the 2.5th and 97.5th percentiles by linear interpolation).
/// Spectra with no defined value for a metric are excluded from that
/// metric's denominator (documented per metric below); coverage overall is
/// always defined (misses are 0), coverage strata are defined only for
/// spectra holding a target in the stratum, formula recall excludes `None`.
/// Validity and uniqueness aggregate the overall value only (their strata
/// are dropped, see their fields); formula recall and abstention aggregate
/// the overall value with null strata (see [`FlagMetric`]).
pub fn summarize(evals: &[SpectrumEval], k: usize, bootstrap: usize, seed: u64) -> MetricSummary {
    // Molecule order of first appearance.
    let mut mol_index: BTreeMap<usize, usize> = BTreeMap::new();
    for e in evals {
        if !mol_index.contains_key(&e.molecule) {
            let n = mol_index.len();
            mol_index.insert(e.molecule, n);
        }
    }
    let n_mol = mol_index.len();
    // Per-spectrum values: overall + 3 strata, NaN when undefined.
    // Metrics order: precision, coverage, coverage_cond(flag), validity,
    // uniqueness, recall, size-aware, worklimit. Abstention and coverage are
    // handled alongside.
    // Per-spectrum values: stratified metrics carry overall + 3 strata (NaN
    // when undefined); validity and uniqueness carry the overall value
    // only; recall and abstention are spectrum-level flags.
    // Metrics order: precision, coverage, validity, uniqueness, size-aware,
    // work-limit. Recall, abstention and conditional coverage are handled
    // alongside.
    #[allow(clippy::type_complexity)]
    let mut per_spec: Vec<(
        [f64; 4],
        [f64; 4],
        f64,
        f64,
        [f64; 4],
        [f64; 4],
        bool,
        Option<bool>,
        bool,
    )> = Vec::with_capacity(evals.len());
    // Conditional coverage needs the domain, tracked per spectrum below.
    for e in evals {
        let n = e.candidates.len().min(k);
        // Distinct finished among the first k (skip duplicates).
        let mut distinct: Vec<&CandidateEval> = Vec::new();
        for c in e.candidates.iter().take(n) {
            if c.finished && !c.duplicate {
                distinct.push(c);
            }
        }
        let finished_count = e.candidates.iter().take(n).filter(|c| c.finished).count();
        // Precision overall + strata.
        let mut prec = [f64::NAN; 4];
        prec[0] = if distinct.is_empty() {
            f64::NAN
        } else {
            distinct
                .iter()
                .filter(|c| c.contained == Containment::Contained)
                .count() as f64
                / distinct.len() as f64
        };
        for s in 0..3 {
            let in_s: Vec<&&CandidateEval> = distinct
                .iter()
                .filter(|c| stratum(c.atoms) == Some(s))
                .collect();
            prec[s + 1] = if in_s.is_empty() {
                f64::NAN
            } else {
                in_s.iter()
                    .filter(|c| c.contained == Containment::Contained)
                    .count() as f64
                    / in_s.len() as f64
            };
        }
        // Validity overall only: finished and valid over considered. No
        // strata (see the `validity` field docs): unfinished candidates
        // have atoms 0, outside every stratum, so strata would divide by
        // finished-in-range and read near 1.0 while the overall is lower.
        let val = if n == 0 {
            f64::NAN
        } else {
            e.candidates
                .iter()
                .take(n)
                .filter(|c| c.finished && c.valid)
                .count() as f64
                / n as f64
        };
        // Uniqueness overall only: distinct over finished. No strata, for
        // the same reason as validity (unfinished candidates have no size
        // and strata would exclude them).
        let uniq = if finished_count == 0 {
            f64::NAN
        } else {
            distinct.len() as f64 / finished_count as f64
        };
        // Size-aware: sum(atoms/16) over contained distinct over distinct.
        let mut saw = [f64::NAN; 4];
        saw[0] = if distinct.is_empty() {
            f64::NAN
        } else {
            distinct
                .iter()
                .filter(|c| c.contained == Containment::Contained)
                .map(|c| c.atoms as f64 / 16.0)
                .sum::<f64>()
                / distinct.len() as f64
        };
        for s in 0..3 {
            let in_s: Vec<&&CandidateEval> = distinct
                .iter()
                .filter(|c| stratum(c.atoms) == Some(s))
                .collect();
            saw[s + 1] = if in_s.is_empty() {
                f64::NAN
            } else {
                in_s.iter()
                    .filter(|c| c.contained == Containment::Contained)
                    .map(|c| c.atoms as f64 / 16.0)
                    .sum::<f64>()
                    / in_s.len() as f64
            };
        }
        // Work-limit rate over distinct.
        let mut wlr = [f64::NAN; 4];
        wlr[0] = if distinct.is_empty() {
            f64::NAN
        } else {
            distinct
                .iter()
                .filter(|c| c.contained == Containment::WorkLimit)
                .count() as f64
                / distinct.len() as f64
        };
        for s in 0..3 {
            let in_s: Vec<&&CandidateEval> = distinct
                .iter()
                .filter(|c| stratum(c.atoms) == Some(s))
                .collect();
            wlr[s + 1] = if in_s.is_empty() {
                f64::NAN
            } else {
                in_s.iter()
                    .filter(|c| c.contained == Containment::WorkLimit)
                    .count() as f64
                    / in_s.len() as f64
            };
        }
        // Coverage is spectrum-level but stratified by target size: stratum
        // `s` is found / total `q` mass of that stratum, NaN when the
        // spectrum holds no target there. Recall and abstention are flags
        // with null strata (see [`FlagMetric`]).
        let mut cov = [f64::NAN; 4];
        cov[0] = e.q_found;
        for s in 0..3 {
            cov[s + 1] = if e.q_total_by_stratum[s] > 0.0 {
                e.q_found_by_stratum[s] / e.q_total_by_stratum[s]
            } else {
                f64::NAN
            };
        }
        let labeled = e.domain == SpectrumDomain::InDomainLabeled;
        per_spec.push((
            prec,
            cov,
            val,
            uniq,
            saw,
            wlr,
            e.abstained,
            e.formula_recall,
            labeled,
        ));
    }
    // Group spectrum positions by molecule.
    let mut by_mol: Vec<Vec<usize>> = vec![Vec::new(); n_mol];
    for (pos, e) in evals.iter().enumerate() {
        let m = mol_index[&e.molecule];
        by_mol[m].push(pos);
    }
    // Helper: per-molecule means for one metric slot, skipping NaN.
    let mean_of_defined = |vals: &[f64]| -> Option<f64> {
        let mut sum = 0.0;
        let mut n = 0usize;
        for v in vals {
            if !v.is_nan() {
                sum += *v;
                n += 1;
            }
        }
        if n == 0 { None } else { Some(sum / n as f64) }
    };
    // Collect per-molecule means: 4 stratified metrics (precision,
    // coverage, size-aware, work-limit) x4 slots, then the overall-only
    // validity, uniqueness, recall and abstention, then conditional
    // coverage x4.
    // Index: metric 0 precision, 1 coverage, 2 size-aware, 3 worklimit,
    // each x4 slots; 16 validity, 17 uniqueness, 18 recall, 19 abstention;
    // 20..24 conditional coverage x4.
    let mut mol_means: Vec<Vec<f64>> = vec![Vec::new(); 4 * 4 + 4 + 4];
    let slot_index = |metric: usize, slot: usize| metric * 4 + slot;
    const VALIDITY: usize = 16;
    const UNIQUENESS: usize = 17;
    const RECALL: usize = 18;
    const ABSTENTION: usize = 19;
    const COND_BASE: usize = 20;
    for mol in &by_mol {
        let mut slots_prec: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut slots_cov: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut slots_covc: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut slots_saw: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut slots_wlr: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut vals_val: Vec<f64> = Vec::new();
        let mut vals_uniq: Vec<f64> = Vec::new();
        let mut vals_rec: Vec<f64> = Vec::new();
        let mut vals_abst: Vec<f64> = Vec::new();
        for &pos in mol {
            let (prec, cov, val, uniq, saw, wlr, abstained, recall, labeled) = &per_spec[pos];
            for s in 0..4 {
                slots_prec[s].push(prec[s]);
                slots_cov[s].push(cov[s]);
                if *labeled {
                    slots_covc[s].push(cov[s]);
                }
                slots_saw[s].push(saw[s]);
                slots_wlr[s].push(wlr[s]);
            }
            vals_val.push(*val);
            vals_uniq.push(*uniq);
            vals_rec.push(match recall {
                None => f64::NAN,
                Some(true) => 1.0,
                Some(false) => 0.0,
            });
            vals_abst.push(if *abstained { 1.0 } else { 0.0 });
        }
        for s in 0..4 {
            if let Some(m) = mean_of_defined(&slots_prec[s]) {
                mol_means[slot_index(0, s)].push(m);
            }
            if let Some(m) = mean_of_defined(&slots_cov[s]) {
                mol_means[slot_index(1, s)].push(m);
            }
            if let Some(m) = mean_of_defined(&slots_saw[s]) {
                mol_means[slot_index(2, s)].push(m);
            }
            if let Some(m) = mean_of_defined(&slots_wlr[s]) {
                mol_means[slot_index(3, s)].push(m);
            }
            if let Some(m) = mean_of_defined(&slots_covc[s]) {
                mol_means[COND_BASE + s].push(m);
            }
        }
        if let Some(m) = mean_of_defined(&vals_val) {
            mol_means[VALIDITY].push(m);
        }
        if let Some(m) = mean_of_defined(&vals_uniq) {
            mol_means[UNIQUENESS].push(m);
        }
        if let Some(m) = mean_of_defined(&vals_rec) {
            mol_means[RECALL].push(m);
        }
        if let Some(m) = mean_of_defined(&vals_abst) {
            mol_means[ABSTENTION].push(m);
        }
    }
    let interval_of = |vals: &[f64]| -> Interval {
        if vals.is_empty() {
            return Interval {
                point: 0.0,
                lo: 0.0,
                hi: 0.0,
            };
        }
        let point = vals.iter().sum::<f64>() / vals.len() as f64;
        let (lo, hi) = bootstrap_interval(vals, bootstrap, seed);
        Interval { point, lo, hi }
    };
    let strat_of = |metric: usize| -> Stratified {
        Stratified {
            overall: interval_of(&mol_means[slot_index(metric, 0)]),
            s3_5: interval_of(&mol_means[slot_index(metric, 1)]),
            s6_9: interval_of(&mol_means[slot_index(metric, 2)]),
            s10_16: interval_of(&mol_means[slot_index(metric, 3)]),
        }
    };
    let covc_strat = Stratified {
        overall: interval_of(&mol_means[COND_BASE]),
        s3_5: interval_of(&mol_means[COND_BASE + 1]),
        s6_9: interval_of(&mol_means[COND_BASE + 2]),
        s10_16: interval_of(&mol_means[COND_BASE + 3]),
    };
    MetricSummary {
        n_spectra: evals.len(),
        n_molecules: n_mol,
        k,
        precision: strat_of(0),
        coverage: strat_of(1),
        coverage_conditional: covc_strat,
        validity: interval_of(&mol_means[VALIDITY]),
        uniqueness: interval_of(&mol_means[UNIQUENESS]),
        formula_recall: FlagMetric::of(interval_of(&mol_means[RECALL])),
        size_aware_precision: strat_of(2),
        abstention: FlagMetric::of(interval_of(&mol_means[ABSTENTION])),
        worklimit_rate: strat_of(3),
    }
}

/// Percentile bootstrap interval over molecule means.
///
/// Resamples `vals` with replacement `resamples` times (seeded SplitMix64),
/// takes the mean of each resample, and returns the 2.5th / 97.5th
/// percentiles by linear interpolation. A constant input collapses to the
/// point.
fn bootstrap_interval(vals: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    if vals.is_empty() || resamples == 0 {
        return (0.0, 0.0);
    }
    let mut rng = SplitMix64::new(seed);
    let n = vals.len();
    let mut means: Vec<f64> = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let j = (rng.next() % n as u64) as usize;
            sum += vals[j];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    (percentile_of(&means, 2.5), percentile_of(&means, 97.5))
}

/// The `p`th percentile by linear interpolation (same rule as
/// `dataset::percentile`).
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

/// Deterministic 64-bit generator for the bootstrap (SplitMix64).
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

/// Teacher-forced NLL per scored token, aggregated per molecule then over
/// molecules, with a percentile bootstrap over molecules.
///
/// Per spectrum `sum_g q_g nll_g / sum_g q_g tokens_g` over its `slots`
/// target slots (`scored_tokens` are the `use` positions of the target).
/// Unlabeled spectra (denominator 0: no `q`-weighted scored token) are
/// excluded from this metric. `molecules[i]` is the molecule index of
/// spectrum `i`. The interval uses 1000 resamples with seed 0; callers that
/// need other bootstrap settings use [`summarize`] on spectrum-level values
/// instead.
pub fn teacher_nll_per_token(
    nll: &[f32],
    q: &[f32],
    scored_tokens: &[u32],
    spectra: usize,
    slots: usize,
    molecules: &[usize],
) -> (f64, (f64, f64)) {
    assert_eq!(
        molecules.len(),
        spectra,
        "teacher_nll_per_token: molecules length must equal spectra"
    );
    assert_eq!(
        nll.len(),
        spectra * slots,
        "teacher_nll_per_token: nll length must equal spectra * slots"
    );
    assert_eq!(
        q.len(),
        spectra * slots,
        "teacher_nll_per_token: q length must equal spectra * slots"
    );
    assert_eq!(
        scored_tokens.len(),
        spectra * slots,
        "teacher_nll_per_token: scored_tokens length must equal spectra * slots"
    );
    // Per-spectrum values, skipping unlabeled (zero denominator).
    let mut per_spec: Vec<(usize, f64)> = Vec::new();
    for (b, m) in molecules.iter().enumerate().take(spectra) {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for g in 0..slots {
            let row = b * slots + g;
            num += f64::from(q[row]) * f64::from(nll[row]);
            den += f64::from(q[row]) * f64::from(scored_tokens[row]);
        }
        if den > 0.0 {
            per_spec.push((*m, num / den));
        }
    }
    // Per-molecule means.
    let mut by_mol: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (m, v) in per_spec {
        by_mol.entry(m).or_default().push(v);
    }
    let mut mol_means: Vec<f64> = Vec::with_capacity(by_mol.len());
    for vals in by_mol.values() {
        mol_means.push(vals.iter().sum::<f64>() / vals.len() as f64);
    }
    if mol_means.is_empty() {
        return (0.0, (0.0, 0.0));
    }
    let point = mol_means.iter().sum::<f64>() / mol_means.len() as f64;
    let (lo, hi) = bootstrap_interval(&mol_means, 1000, 0);
    (point, (lo, hi))
}

// ---------------------------------------------------------------------------
// Field-split teacher NLL for `--diagnose`
// ---------------------------------------------------------------------------

/// Teacher NLL per token overall and split by field, for `--diagnose`.
///
/// Each bucket is `sum q * (-log p)` over that bucket's used positions
/// divided by `sum q * (count of that bucket's used positions)`, per
/// spectrum, then per molecule, then over molecules with percentile bootstrap
/// intervals over molecules. Buckets: kind with STOP separated (kind tokens
/// whose target kind is STOP vs other kinds), atom type, bond, pointer.
/// Uses `TeacherOutput.field_log_prob` (`[B*G, T, 4]`) with the host
/// `use_mask` and target tokens. Unlabeled spectra (zero denominator) are
/// excluded per bucket, exactly as [`teacher_nll_per_token`] excludes them
/// overall.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldSplit {
    /// Overall teacher NLL per token (same definition as
    /// [`teacher_nll_per_token` but with the caller's bootstrap settings).
    pub overall: Interval,
    /// Kind field where the target kind is STOP.
    pub kind_stop: Interval,
    /// Kind field where the target kind is not STOP.
    pub kind_other: Interval,
    /// Atom-type field.
    pub atom_type: Interval,
    /// Bond field.
    pub bond: Interval,
    /// Pointer field.
    pub pointer: Interval,
}

/// Paired per-molecule difference with its bootstrap interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedDiff {
    /// Mean over molecules of per-molecule (donor − own) differences.
    pub point: f64,
    /// 2.5th percentile of the bootstrap means.
    pub lo: f64,
    /// 97.5th percentile of the bootstrap means.
    pub hi: f64,
}

/// Field-split teacher NLL with configurable bootstrap.
///
/// `nll`, `q`, `scored_tokens` are the per-target teacher outputs
/// (`[B*G]`); `field_log_prob` is `[B*G*T*4]` row-major; `use_mask` is the
/// host `[B*G*T*4]` field-use indicators; `tokens` are the host target tokens
/// (`[B*G*T*4]`); `molecules[b]` is the molecule of spectrum `b`. Lengths are
/// asserted. `bootstrap` resamples with replacement seeded by `seed`.
#[allow(clippy::too_many_arguments)]
pub fn teacher_field_split(
    nll: &[f32],
    q: &[f32],
    scored_tokens: &[u32],
    field_log_prob: &[f32],
    use_mask: &[f32],
    tokens: &[u32],
    spectra: usize,
    slots: usize,
    max_steps: usize,
    molecules: &[usize],
    bootstrap: usize,
    seed: u64,
) -> FieldSplit {
    assert_eq!(
        molecules.len(),
        spectra,
        "field split: molecules vs spectra"
    );
    assert_eq!(nll.len(), spectra * slots, "field split: nll length");
    assert_eq!(q.len(), spectra * slots, "field split: q length");
    assert_eq!(
        scored_tokens.len(),
        spectra * slots,
        "field split: scored_tokens length"
    );
    assert_eq!(
        field_log_prob.len(),
        spectra * slots * max_steps * 4,
        "field split: field_log_prob length"
    );
    assert_eq!(
        use_mask.len(),
        spectra * slots * max_steps * 4,
        "field split: use_mask length"
    );
    assert_eq!(
        tokens.len(),
        spectra * slots * max_steps * 4,
        "field split: tokens length"
    );
    // Per-spectrum overall (same rule as `teacher_nll_per_token`) plus the
    // five field buckets. NaN when the spectrum has no denominator there.
    let mut spec_overall = vec![f64::NAN; spectra];
    let mut spec_kind_stop = vec![f64::NAN; spectra];
    let mut spec_kind_other = vec![f64::NAN; spectra];
    let mut spec_type = vec![f64::NAN; spectra];
    let mut spec_bond = vec![f64::NAN; spectra];
    let mut spec_ptr = vec![f64::NAN; spectra];
    for b in 0..spectra {
        let mut num_all = 0.0f64;
        let mut den_all = 0.0f64;
        let mut num_ks = 0.0f64;
        let mut den_ks = 0.0f64;
        let mut num_ko = 0.0f64;
        let mut den_ko = 0.0f64;
        let mut num_ty = 0.0f64;
        let mut den_ty = 0.0f64;
        let mut num_bo = 0.0f64;
        let mut den_bo = 0.0f64;
        let mut num_pt = 0.0f64;
        let mut den_pt = 0.0f64;
        for g in 0..slots {
            let row = b * slots + g;
            let qw = f64::from(q[row]);
            if qw == 0.0 {
                continue;
            }
            num_all += qw * f64::from(nll[row]);
            den_all += qw * f64::from(scored_tokens[row]);
            for i in 0..max_steps {
                let base = (row * max_steps + i) * 4;
                // Kind (field 0), split by the target kind at token `i + 1`.
                if use_mask[base] != 0.0 {
                    // The target token at position `i + 1`; at the last
                    // output position there is no next token, but `use` is 0
                    // there, so this branch never reads out of bounds for a
                    // used position.
                    let pos = i + 1;
                    let kind = if pos < max_steps {
                        tokens[(row * max_steps + pos) * 4] as u8
                    } else {
                        0
                    };
                    let v = -f64::from(field_log_prob[base]);
                    if kind == STOP {
                        num_ks += qw * v;
                        den_ks += qw;
                    } else {
                        num_ko += qw * v;
                        den_ko += qw;
                    }
                }
                if use_mask[base + 1] != 0.0 {
                    num_ty += qw * -f64::from(field_log_prob[base + 1]);
                    den_ty += qw;
                }
                if use_mask[base + 2] != 0.0 {
                    num_bo += qw * -f64::from(field_log_prob[base + 2]);
                    den_bo += qw;
                }
                if use_mask[base + 3] != 0.0 {
                    num_pt += qw * -f64::from(field_log_prob[base + 3]);
                    den_pt += qw;
                }
            }
        }
        if den_all > 0.0 {
            spec_overall[b] = num_all / den_all;
        }
        if den_ks > 0.0 {
            spec_kind_stop[b] = num_ks / den_ks;
        }
        if den_ko > 0.0 {
            spec_kind_other[b] = num_ko / den_ko;
        }
        if den_ty > 0.0 {
            spec_type[b] = num_ty / den_ty;
        }
        if den_bo > 0.0 {
            spec_bond[b] = num_bo / den_bo;
        }
        if den_pt > 0.0 {
            spec_ptr[b] = num_pt / den_pt;
        }
    }
    let interval_of_specs = |specs: &[f64]| -> Interval {
        let mut by_mol: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for (b, v) in specs.iter().enumerate() {
            if !v.is_nan() {
                by_mol.entry(molecules[b]).or_default().push(*v);
            }
        }
        let mut means: Vec<f64> = Vec::with_capacity(by_mol.len());
        for vals in by_mol.values() {
            means.push(vals.iter().sum::<f64>() / vals.len() as f64);
        }
        interval_of(&means, bootstrap, seed)
    };
    FieldSplit {
        overall: interval_of_specs(&spec_overall),
        kind_stop: interval_of_specs(&spec_kind_stop),
        kind_other: interval_of_specs(&spec_kind_other),
        atom_type: interval_of_specs(&spec_type),
        bond: interval_of_specs(&spec_bond),
        pointer: interval_of_specs(&spec_ptr),
    }
}

/// Per-spectrum field values behind a [`FieldSplit`]: one `[overall,
/// kind_stop, kind_other, atom_type, bond, pointer]` row per spectrum (NaN
/// when the spectrum has no denominator there).
///
/// `field_log_prob` is `[B*G*T*4]` row-major, `use_mask` the host
/// `[B*G*T*4]` indicators, `tokens` the host target tokens. The kind split
/// reads the target kind at token `i + 1` for output position `i`.
pub fn field_per_spectrum(
    nll: &[f32],
    q: &[f32],
    scored_tokens: &[u32],
    field_log_prob: &[f32],
    use_mask: &[f32],
    tokens: &[u32],
    spectra: usize,
    slots: usize,
    max_steps: usize,
) -> Vec<[f64; 6]> {
    assert_eq!(nll.len(), spectra * slots, "field split: nll length");
    assert_eq!(q.len(), spectra * slots, "field split: q length");
    assert_eq!(
        scored_tokens.len(),
        spectra * slots,
        "field split: scored_tokens length"
    );
    assert_eq!(
        field_log_prob.len(),
        spectra * slots * max_steps * 4,
        "field split: field_log_prob length"
    );
    assert_eq!(
        use_mask.len(),
        spectra * slots * max_steps * 4,
        "field split: use_mask length"
    );
    assert_eq!(
        tokens.len(),
        spectra * slots * max_steps * 4,
        "field split: tokens length"
    );
    let mut out = vec![[f64::NAN; 6]; spectra];
    for (b, row_out) in out.iter_mut().enumerate().take(spectra) {
        let mut num_all = 0.0f64;
        let mut den_all = 0.0f64;
        let mut num_ks = 0.0f64;
        let mut den_ks = 0.0f64;
        let mut num_ko = 0.0f64;
        let mut den_ko = 0.0f64;
        let mut num_ty = 0.0f64;
        let mut den_ty = 0.0f64;
        let mut num_bo = 0.0f64;
        let mut den_bo = 0.0f64;
        let mut num_pt = 0.0f64;
        let mut den_pt = 0.0f64;
        for g in 0..slots {
            let row = b * slots + g;
            let qw = f64::from(q[row]);
            if qw == 0.0 {
                continue;
            }
            num_all += qw * f64::from(nll[row]);
            den_all += qw * f64::from(scored_tokens[row]);
            for i in 0..max_steps {
                let base = (row * max_steps + i) * 4;
                if use_mask[base] != 0.0 {
                    let pos = i + 1;
                    let kind = if pos < max_steps {
                        tokens[(row * max_steps + pos) * 4] as u8
                    } else {
                        0
                    };
                    let v = -f64::from(field_log_prob[base]);
                    if kind == STOP {
                        num_ks += qw * v;
                        den_ks += qw;
                    } else {
                        num_ko += qw * v;
                        den_ko += qw;
                    }
                }
                if use_mask[base + 1] != 0.0 {
                    num_ty += qw * -f64::from(field_log_prob[base + 1]);
                    den_ty += qw;
                }
                if use_mask[base + 2] != 0.0 {
                    num_bo += qw * -f64::from(field_log_prob[base + 2]);
                    den_bo += qw;
                }
                if use_mask[base + 3] != 0.0 {
                    num_pt += qw * -f64::from(field_log_prob[base + 3]);
                    den_pt += qw;
                }
            }
        }
        if den_all > 0.0 {
            row_out[0] = num_all / den_all;
        }
        if den_ks > 0.0 {
            row_out[1] = num_ks / den_ks;
        }
        if den_ko > 0.0 {
            row_out[2] = num_ko / den_ko;
        }
        if den_ty > 0.0 {
            row_out[3] = num_ty / den_ty;
        }
        if den_bo > 0.0 {
            row_out[4] = num_bo / den_bo;
        }
        if den_pt > 0.0 {
            row_out[5] = num_pt / den_pt;
        }
    }
    out
}

/// Per-molecule means of per-spectrum values, for the paired peak
/// sensitivity: means in molecule order of first appearance, skipping
/// molecules with no defined spectrum. Callers difference the two inputs
/// molecule by molecule and pass the diffs to [`paired_interval`].
pub fn per_molecule_means(values: &[f64], molecules: &[usize]) -> Vec<f64> {
    assert_eq!(
        values.len(),
        molecules.len(),
        "per_molecule_means: values vs molecules"
    );
    let mut by_mol: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (b, v) in values.iter().enumerate() {
        if !v.is_nan() {
            by_mol.entry(molecules[b]).or_default().push(*v);
        }
    }
    by_mol
        .values()
        .map(|vals| vals.iter().sum::<f64>() / vals.len() as f64)
        .collect()
}

/// Mean of paired per-molecule differences with a paired bootstrap interval.
///
/// Resamples the diffs with replacement (`bootstrap` times, seeded by `seed`)
/// and reports the 2.5th/97.5th percentiles by linear interpolation (the same
/// rule as [`summarize`]). Pairing is by molecule: the caller differences
/// per-molecule means, so each resample keeps own/donor pairs together.
pub fn paired_interval(diffs: &[f64], bootstrap: usize, seed: u64) -> PairedDiff {
    if diffs.is_empty() {
        return PairedDiff {
            point: 0.0,
            lo: 0.0,
            hi: 0.0,
        };
    }
    let point = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let (lo, hi) = bootstrap_interval(diffs, bootstrap, seed);
    PairedDiff { point, lo, hi }
}

/// Interval helper shared by the field split: mean over molecule means with
/// a percentile bootstrap (empty input gives zeros, as in [`summarize`]).
fn interval_of(vals: &[f64], bootstrap: usize, seed: u64) -> Interval {
    if vals.is_empty() {
        return Interval {
            point: 0.0,
            lo: 0.0,
            hi: 0.0,
        };
    }
    let point = vals.iter().sum::<f64>() / vals.len() as f64;
    let (lo, hi) = bootstrap_interval(vals, bootstrap, seed);
    Interval { point, lo, hi }
}
