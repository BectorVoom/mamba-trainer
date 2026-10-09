//! Mass-to-formulas-to-generation shared path for the completion model.
//!
//! One library function both the trained-model API (`completion_api`) and the
//! mass experiment arm (`completion_experiment`) use: enumerate formula
//! hypotheses from a neutral mass (or a precursor ion), filter them through
//! the model-domain, substructure and completability stages, sample the
//! survivors under a fixed total trajectory budget, and pool the accepted
//! identities.
//!
//! Selection is deterministic and by default carries no learned prior:
//! survivors are ordered by verdict (`Accept` before `Ambiguous`), then
//! absolute mass residual, then the enumerator's canonical order. This order
//! is a deterministic default, not a formula probability. A caller that has
//! predicted element counts for the query can pass an [`ElementPrior`]
//! ([`run_mass_completion_search_with_prior`]), which orders the same
//! survivors by distance to that prediction. The enumerator capacity
//! is unbounded on this path, so it cannot bind before selection; a search
//! that hits the node budget reports `search_exhausted`, never a silently
//! complete truncation.
//!
//! Pruning ([`FormulaPruning`]) separates necessary (exact) filters from the
//! empirical train-fit bounds; allocation ([`FormulaAllocation`]) separates
//! the even trajectory split from the training-frequency prior weights.
//!
//! There is no second implementation: the API and the experiment both call
//! [`run_mass_completion`].

use cubecl::prelude::Runtime;

use crate::backend::Device;
use crate::error::{Error, Result};
use crate::tensor::ops::ms2::Ms2Constants;

use super::chem::{ADDUCTS, Composition, ELEMENTS, HYDROGEN};
use super::completion_model::{
    CompletionGenerationConfig, CompletionModel, CompletionRequest, FormulaArtifacts, QueryOutcome,
    SubstructureSemantics,
};
use super::formula_enum::{EnumDomain, EnumLimits, EnumResult, enumerate, enumerate_neutral};
use super::grammar::{Limits, TraceState};

/// Formula-search pruning: which enumeration bounds apply on top of the exact
/// chemical filters.
///
/// The exact filters (saturated-acyclic hydrogen ceiling, parity/integer-DBE
/// rule, DBE `>= 0`) are necessary for the V0 chemistry domain and always
/// apply. The [`FormulaPruning`] choice only adds or removes the empirical
/// train-fit bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulaPruning {
    /// Exact chemical filters plus the checkpoint's train-fit [`RatioBounds`]
    /// and [`EnumDomain`] caps (today's behaviour). The bounds are fitted on
    /// training molecules with margin 0: they can exclude the true formula
    /// of a new molecule. An empirical modelling choice, not chemistry.
    TrainFit,
    /// Exact chemical filters only: element caps come from the model's
    /// `max_atoms` alone (no [`RatioBounds`], no fitted domain caps).
    ChemicalOnly,
}

impl FormulaPruning {
    /// The request/response spelling of the pruning.
    pub fn as_str(self) -> &'static str {
        match self {
            FormulaPruning::TrainFit => "train_fit",
            FormulaPruning::ChemicalOnly => "chemical_only",
        }
    }

    /// Parse a request spelling.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "train_fit" => Some(FormulaPruning::TrainFit),
            "chemical_only" => Some(FormulaPruning::ChemicalOnly),
            _ => None,
        }
    }
}

impl Default for FormulaPruning {
    /// `train_fit` (today's behaviour).
    fn default() -> Self {
        FormulaPruning::TrainFit
    }
}

/// Trajectory allocation over the selected formulas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulaAllocation {
    /// Even split with largest-remainder redistribution (the default).
    Equal,
    /// Training-frequency prior weights from the checkpoint's composition
    /// counts (explicit estimate ranking; still not calibrated).
    TrainFrequency,
}

impl FormulaAllocation {
    /// The request/response spelling of the allocation.
    pub fn as_str(self) -> &'static str {
        match self {
            FormulaAllocation::Equal => "equal",
            FormulaAllocation::TrainFrequency => "train_frequency",
        }
    }

    /// Parse a request spelling.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "equal" => Some(FormulaAllocation::Equal),
            "train_frequency" => Some(FormulaAllocation::TrainFrequency),
            _ => None,
        }
    }
}

impl Default for FormulaAllocation {
    /// `equal` (the even split).
    fn default() -> Self {
        FormulaAllocation::Equal
    }
}

/// Row cap for the second (non-primary pruning) enumeration behind
/// [`FormulaSearchReport::joined_chemical`].
///
/// The rerun joins a superset of the primary search, so a primary search
/// that already joined more than this many rows would need at least as much
/// memory again: the rerun is skipped with a reason instead.
pub const CHEMICAL_RERUN_JOIN_CAP: usize = 200_000;

/// Node budget for the diagnostic rerun behind
/// [`FormulaSearchReport::joined_chemical`].
///
/// The rerun has its own bounded budget (independent of how much of the
/// request's node budget the primary search consumed, capped by what is left
/// of it): when its own limit binds, the chemical-only count is `None` with
/// a reason instead of an unbounded second search.
pub const CHEMICAL_RERUN_NODES_MAX: u64 = 2_000_000;

/// Chemical-only enumeration domain for `max_atoms` heavy atoms.
///
/// Per heavy element the cap is `max_atoms` with the heavy total capped at
/// `max_atoms` (exactly the model-domain condition), hydrogen `0` to
/// `6 * max_atoms + 2` (saturating). The hydrogen bound is sound: the exact
/// hydrogen ceiling is `sum v*n - 2*n + 2` with per-element maximum valence
/// `v <= 6` (S v6), so no composition within the heavy cap can need more
/// hydrogens; the exact ceiling filter (not this bound) does the real work.
pub fn chemical_only_domain(max_atoms: u32) -> EnumDomain {
    let cap = max_atoms.min(u32::from(u16::MAX)) as u16;
    let hydrogen_max = (u32::from(cap) * 6 + 2).min(u32::from(u16::MAX)) as u16;
    EnumDomain {
        version: super::formula_enum::ENUM_DOMAIN_VERSION.to_string(),
        heavy_caps: [cap; 9],
        heavy_max: cap,
        hydrogen_min: 0,
        hydrogen_max,
    }
}

/// A mass query for [`run_mass_completion`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassQuery {
    /// Already-neutral mass: `value` is the neutral mass in micro-dalton.
    Neutral {
        /// Neutral mass in micro-dalton.
        value: u32,
        /// Tolerance in tenths of a ppm.
        ppm_tenths: u32,
        /// Uncertainty in micro-dalton; `None` means unknown precision.
        uncertainty: Option<u32>,
    },
    /// Precursor ion: `value` is the precursor m/z in micro-m/z.
    Precursor {
        /// Precursor m/z in micro-m/z.
        value: u32,
        /// Adduct id (one of [`ADDUCTS`]).
        adduct: u16,
        /// Tolerance in tenths of a ppm.
        ppm_tenths: u32,
        /// Uncertainty in micro-dalton; `None` means unknown precision.
        uncertainty: Option<u32>,
    },
}

/// One selected formula with its search verdict and sampling assignment.
#[derive(Clone, Debug)]
pub struct FormulaEntry {
    /// The composition.
    pub composition: Composition,
    /// Canonical formula text (for example `C2H6O`).
    pub formula: String,
    /// Integer neutral mass.
    pub computed_uda: u32,
    /// Absolute residual against the parent neutral mass.
    pub residual_uda: u32,
    /// `true` when the enumerator verdict was ambiguous.
    pub ambiguous: bool,
    /// Selection weight: uniform `1 / selected` under
    /// [`FormulaAllocation::Equal`], the training-frequency prior
    /// `(count + 1) / sum(count + 1)` over the selected formulas under
    /// [`FormulaAllocation::TrainFrequency`] (a prior, not a calibrated
    /// probability).
    pub weight: f64,
    /// Trajectories assigned (`0` means `not_sampled`).
    pub trajectories: u32,
    /// Finished trajectories (`0` when not sampled).
    pub finished: u32,
    /// Accepted candidates from this formula (`0` when not sampled).
    pub accepted_candidates: u32,
    /// Actual search counters for this formula under beam search; `None`
    /// under sampling or when no decoding budget was assigned.
    pub beam_stats: Option<super::completion_model::BeamStats>,
}

/// Enumerator counters behind a mass search.
#[derive(Clone, Debug)]
pub struct EnumeratorReport {
    /// Heavy prefixes expanded.
    pub nodes_visited: u64,
    /// Hydrogen verdicts run.
    pub hydrogen_checks: u64,
    /// Rows joined before caps.
    pub rows_joined: u64,
    /// Rows scored (kept output length).
    pub rows_scored: u64,
    /// Verdict-passing compositions removed by the hydrogen ceiling.
    pub rejected_h_max: u64,
    /// Removed by the parity rule.
    pub rejected_parity: u64,
    /// Removed by DBE `>= 0`.
    pub rejected_dbe: u64,
    /// Removed by the mass verdict itself.
    pub rejected_mass: u64,
    /// DFS branches skipped by the (i) caps.
    pub pruned_ratio_cap: u64,
    /// DFS branches skipped by the (iii) maxima.
    pub pruned_rare: u64,
    /// Leaf completions failing the (i) refinement.
    pub rejected_ratio_cap: u64,
    /// Leaf completions failing the (iii) stages.
    pub rejected_ratio_rare: u64,
    /// Removed by the (ii) H/C stage.
    pub rejected_ratio_hc: u64,
    /// Removed by the (ii) N/C stage.
    pub rejected_ratio_nc: u64,
    /// Removed by the (ii) O/C stage.
    pub rejected_ratio_oc: u64,
    /// Removed by the (ii) hal/C stage.
    pub rejected_ratio_hal: u64,
    /// Removed by the (ii) S/C stage.
    pub rejected_ratio_s: u64,
    /// Removed by the (ii) P/C stage.
    pub rejected_ratio_p: u64,
    /// Removed by the (iv) DBE stage.
    pub rejected_ratio_dbe: u64,
    /// A work limit stopped the search.
    pub exhausted: bool,
}

/// One formula-search stage: its class separates necessary stages (mass
/// verdict, exact chemical filters, model domain, substructure bound,
/// completability) from empirical ones (train-fit bounds) and budget ones
/// (first-F selection, trajectory allocation).
#[derive(Clone, Debug)]
pub struct SearchStage {
    /// Stage name (`mass_verdict`, `exact_chemical_filters`,
    /// `train_fit_bounds`, `model_domain`, `substructure_bound`,
    /// `completability`, `first_F_selection`, `trajectory_allocation`).
    pub stage: String,
    /// `necessary`, `empirical` or `budget`.
    pub class: String,
    /// Rows entering the stage.
    pub entering: usize,
    /// Rows leaving the stage.
    pub leaving: usize,
    /// Qualifier (for example why a stage was skipped or truncated).
    pub note: String,
}

/// The formula-search stages behind a mass request.
#[derive(Clone, Debug)]
pub struct FormulaSearchReport {
    /// `complete`, `truncated`, `search_exhausted`, `unavailable` or
    /// `mass_overflow`. `search_exhausted` takes precedence over `truncated`:
    /// when both limits bind, the search may have missed a nearer formula.
    /// The two facts are also exposed independently as
    /// [`FormulaSearchReport::truncated`] and
    /// [`FormulaSearchReport::search_exhausted`].
    pub status: String,
    /// Whether the first-`hypotheses` selection dropped survivors.
    pub truncated: bool,
    /// Whether the enumerator reported exhaustion (node budget bound).
    pub search_exhausted: bool,
    /// Why nothing was sampled although the search ran (`None` when every
    /// selected formula was sampled): the first stage that left zero rows —
    /// `by_train_fit` (the primary train-fit search joined nothing while
    /// the chemical-only rerun joined rows), `all_excluded_by_domain`,
    /// `by_substructures`, `by_completability` — or `budget` when at least
    /// one selected formula received zero trajectories.
    pub unsampled_reason: Option<String>,
    /// The pruning used (`train_fit` or `chemical_only`).
    pub pruning: String,
    /// The trajectory allocation (`equal` or `train_frequency`).
    pub allocation: String,
    /// Compositions joined by the enumerator (under [`FormulaSearchReport::pruning`]).
    pub joined: usize,
    /// Compositions the exact chemical filters alone admit (the
    /// chemical-only enumeration), when cheap to obtain: the rerun runs when
    /// the primary search joined fewer than
    /// [`CHEMICAL_RERUN_JOIN_CAP`] rows and the node budget allows;
    /// otherwise `None` with [`FormulaSearchReport::joined_chemical_reason`].
    /// Equals `joined` when the pruning itself is `chemical_only`.
    pub joined_chemical: Option<usize>,
    /// Why [`FormulaSearchReport::joined_chemical`] is `None` (`None` when
    /// it is available).
    pub joined_chemical_reason: Option<String>,
    /// `joined_chemical - joined`: rows the train-fit bounds excluded
    /// (`None` when `joined_chemical` is unavailable).
    pub excluded_by_train_fit: Option<usize>,
    /// Survivors after the model-domain filter.
    pub after_domain: usize,
    /// Survivors after the substructure lower bound.
    pub after_substructures: usize,
    /// Survivors after the completability pre-check.
    pub after_completability: usize,
    /// Formulas selected (at most `hypotheses`).
    pub selected: usize,
    /// Formulas actually sampled (trajectories `> 0`).
    pub sampled: usize,
    /// Ranking rule (no learned prior; a deterministic default, not a
    /// formula probability).
    pub ranking: String,
    /// Per-stage entering/leaving counts with their necessary / empirical /
    /// budget class.
    pub stages: Vec<SearchStage>,
    /// Selected formulas in rank order.
    pub formulas: Vec<FormulaEntry>,
    /// Enumerator counters.
    pub enumerator: EnumeratorReport,
}

/// One pooled candidate with its formula mass evidence.
#[derive(Clone, Debug)]
pub struct PooledCandidate {
    /// Atom type ids of the molecule.
    pub atoms: Vec<u8>,
    /// Bonds as `(a, b, order)`.
    pub bonds: Vec<(usize, usize, u8)>,
    /// Element composition of the molecule.
    pub composition: Composition,
    /// The first trace producing this identity.
    pub trace: Vec<super::grammar::Token>,
    /// Trajectories producing this identity.
    pub samples: u32,
    /// Best trace log-probability.
    pub best_log_prob: f32,
    /// Formula text of the source formula.
    pub formula: String,
    /// Computed neutral mass of the source formula.
    pub computed_uda: u32,
    /// Residual against the parent neutral mass.
    pub residual_uda: u32,
    /// `accepted` or `boundary_ambiguous` (the source verdict).
    pub mass_status: String,
    /// Rank order of the source formula.
    pub formula_order: usize,
}

/// Summed accounting over the sampled formulas.
#[derive(Clone, Debug, Default)]
pub struct PooledAccounting {
    /// Trajectories executed (sum over formulas).
    pub trajectories: u32,
    /// Device status FINISHED (sum).
    pub finished: u32,
    /// Device status `no_valid_action` (sum).
    pub dead_end: u32,
    /// Device status TRUNCATED (sum).
    pub truncated: u32,
    /// Other device statuses (sum).
    pub other_status: u32,
    /// Rejected by host replay (sum).
    pub rejected_replay: u32,
    /// Rejected by containment (sum).
    pub rejected_containment: u32,
    /// Containment work-limit (sum).
    pub containment_unresolved: u32,
    /// Identity work-limit (sum).
    pub identity_unresolved: u32,
    /// Accepted identities before the pooled cut (sum).
    pub distinct: u32,
    /// Unresolved shortlist entries (sum of lengths).
    pub unresolved: u32,
    /// Rejected for extra functional groups under `CompleteFunctionalGroups`
    /// (sum; inside `rejected_containment`).
    pub rejected_extra_groups: u32,
    /// Rejected for missing groups under `CompleteFunctionalGroups` (sum;
    /// inside `rejected_containment`).
    pub rejected_missing_groups: u32,
    /// Finished trajectories passing the `Contained` rule (sum).
    pub pass_contained: u32,
    /// Finished trajectories passing the `DisjointOccurrences` rule (sum).
    pub pass_disjoint: u32,
    /// Finished trajectories passing the `CompleteFunctionalGroups` rule
    /// (sum).
    pub pass_complete: u32,
}

/// Named parts of the §5 verdict error bound behind a mass request, in
/// micro-dalton: `error(composition) = observation + composition +
/// neutralisation` (saturating).
#[derive(Clone, Debug)]
pub struct MassErrorTerms {
    /// The supplied mass's rounding half-width (`uncertainty_uda`).
    pub observation: u32,
    /// The largest `ceil(composition_error_nda / 1000)` over the joined rows
    /// (`0` when nothing joined). Each row's own verdict uses its own
    /// composition term, so the three parts sum to an upper bound, exact for
    /// the worst-case joined row.
    pub composition: u32,
    /// The adduct conversion's rounding bound: `1` on the precursor path
    /// (contract §4.3), `0` for an already-neutral mass (no conversion).
    pub neutralisation: u32,
}

/// How a mass-completion run searches each formula hypothesis.
///
/// Both modes spend the same rows: `total_trajectories` is split across the
/// selected formulas by [`allocate_trajectories`], and a formula's share is
/// its sample count under [`Sampling`](CompletionSearch::Sampling) or its
/// beam width under [`Beam`](CompletionSearch::Beam). Equal totals are
/// therefore equal decoder rows, which is what makes the two comparable;
/// [`MassCompletionResult::beam_row_steps`] reports the rows actually
/// executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionSearch {
    /// Independent samples under the exact-completion rule
    /// ([`CompletionModel::generate_with_spectra`]).
    Sampling,
    /// A beam over the grammar
    /// ([`CompletionModel::generate_beam_with_spectra`]), which keeps an
    /// action by rank instead of drawing it, so a single unlikely step does
    /// not cost the whole trace. `temperature` and `seed` are then unused.
    Beam,
}

/// The pooled result of [`run_mass_completion`].
#[derive(Debug)]
pub struct MassCompletionResult {
    /// Pooled candidates, ranked and cut to `returned`.
    pub candidates: Vec<PooledCandidate>,
    /// Accepted identities before the pooled cut (pooled length).
    pub distinct_before_cut: u32,
    /// Summed accounting.
    pub accounting: PooledAccounting,
    /// `accepted`, `boundary_ambiguous`, `rejected`, `search_incomplete`,
    /// `unavailable` or `mass_overflow`, derived only from the formula
    /// search itself — never from whether sampling happened:
    ///
    /// * `accepted`: the completed search joined at least one `Accept`
    ///   verdict row.
    /// * `boundary_ambiguous`: the completed search joined rows but only
    ///   `Ambiguous` ones.
    /// * `rejected`: the search completed and no formula passed the mass
    ///   verdict (under `train_fit` pruning this additionally requires the
    ///   chemical-only rerun to complete empty — train-fit-only exclusions
    ///   are reported via `unsampled_reason`, never as mass rejection).
    /// * `search_incomplete`: a node budget or capacity bound stopped the
    ///   search (primary or the disambiguating rerun), so absence proves
    ///   nothing.
    /// * `unavailable`: unknown mass precision (`null` uncertainty) — no
    ///   search ran and no claim is made.
    /// * `mass_overflow`: precursor neutralisation left the `u32` range —
    ///   the enumerator's status, propagated, never a search around a
    ///   substituted zero mass.
    pub mass_evidence_status: String,
    /// Named §5 error parts (`None` when no search ran).
    pub error_terms: Option<MassErrorTerms>,
    /// The formula-search stages.
    pub formula_search: FormulaSearchReport,
    /// Compositions of every scored (joined) row, in canonical order, for
    /// the experiment's true-formula absence check.
    pub joined_compositions: Vec<Composition>,
    /// Decoder rows actually executed by a
    /// [`Beam`](CompletionSearch::Beam) run, summed over the formula
    /// hypotheses (0 under [`Sampling`](CompletionSearch::Sampling), whose
    /// work is `trajectories` times the step limit by construction).
    pub beam_row_steps: u64,
    /// Legal continuations a [`Beam`](CompletionSearch::Beam) run's width
    /// dropped, summed over the formula hypotheses: what a wider beam would
    /// have had room for.
    pub beam_candidates_dropped: u64,
}

/// Canonical formula text, matching the request layer's `composition_text`
/// style for C/H/N/O (count `1` omitted) and extending it to the remaining
/// elements in table order. For example ethanol renders `C2H6O`.
pub fn formula_text(c: &Composition) -> String {
    let suffix = |symbol: &str, count: u16| {
        if count == 1 {
            symbol.to_string()
        } else {
            format!("{symbol}{count}")
        }
    };
    let mut out = String::new();
    if c[0] > 0 {
        out.push_str(&suffix(ELEMENTS[0].symbol, c[0]));
    }
    // Hydrogen always renders (even zero mirrors `composition_text`).
    out.push_str(&suffix(ELEMENTS[1].symbol, c[1]));
    for e in [2usize, 3, 4, 5, 6, 7, 8, 9] {
        if c[e] > 0 {
            out.push_str(&suffix(ELEMENTS[e].symbol, c[e]));
        }
    }
    out
}

/// Heavy-atom total of a composition.
fn heavy_total(c: &Composition) -> u32 {
    let mut total: u32 = 0;
    for e in [0usize, 2, 3, 4, 5, 6, 7, 8, 9] {
        total += u32::from(c[e]);
    }
    total
}

/// Substructure lower bound: necessary for containment under the active
/// [`SubstructureSemantics`].
///
/// Under `Contained`, every pattern's per-element heavy count must not exceed
/// the composition's, and the pattern's hydrogens (summed over its atom
/// types) must not exceed the composition's hydrogens. Under
/// `DisjointOccurrences` and `CompleteFunctionalGroups` the occurrences are
/// disjoint, so the check sums over all patterns instead of taking the
/// per-pattern maximum; under `CompleteFunctionalGroups` the summed
/// heteroatom counts must additionally equal the composition's
/// (`functional-groups-ertl-v1` marks every heteroatom).
pub fn passes_substructure(
    patterns: &[super::graph::MolGraph],
    c: &Composition,
    semantics: SubstructureSemantics,
) -> bool {
    if semantics == SubstructureSemantics::Contained {
        for pattern in patterns {
            let mut need_heavy = [0u16; 10];
            let mut need_h: u32 = 0;
            for &id in pattern.atoms() {
                let Some(t) = super::chem::atom_type(id) else {
                    return false;
                };
                if t.element == HYDROGEN {
                    need_h += 1;
                } else {
                    need_heavy[t.element] += 1;
                    need_h += u32::from(t.hydrogens);
                }
            }
            for e in [0usize, 2, 3, 4, 5, 6, 7, 8, 9] {
                if need_heavy[e] > c[e] {
                    return false;
                }
            }
            if need_h > u32::from(c[HYDROGEN]) {
                return false;
            }
        }
        return true;
    }
    let mut need_heavy = [0u32; 10];
    let mut need_h: u32 = 0;
    for pattern in patterns {
        for &id in pattern.atoms() {
            let Some(t) = super::chem::atom_type(id) else {
                return false;
            };
            if t.element == HYDROGEN {
                need_h += 1;
            } else {
                need_heavy[t.element] += 1;
                need_h += u32::from(t.hydrogens);
            }
        }
    }
    for e in [0usize, 2, 3, 4, 5, 6, 7, 8, 9] {
        if need_heavy[e] > u32::from(c[e]) {
            return false;
        }
    }
    if need_h > u32::from(c[HYDROGEN]) {
        return false;
    }
    if semantics == SubstructureSemantics::CompleteFunctionalGroups {
        for e in [2usize, 3, 4, 5, 6, 7, 8, 9] {
            if need_heavy[e] != u32::from(c[e]) {
                return false;
            }
        }
    }
    true
}

/// First 8 bytes of the SHA-256 of `text`, as a `u64` (big-endian).
fn numeric_id_hex(text: &str) -> u64 {
    let hex = super::experiment::sha256_hex(text.as_bytes());
    u64::from_str_radix(&hex[..16], 16).expect("hex of a hash parses")
}

/// Empty enumerator report (no search ran).
fn empty_enumerator() -> EnumeratorReport {
    EnumeratorReport {
        nodes_visited: 0,
        hydrogen_checks: 0,
        rows_joined: 0,
        rows_scored: 0,
        rejected_h_max: 0,
        rejected_parity: 0,
        rejected_dbe: 0,
        rejected_mass: 0,
        pruned_ratio_cap: 0,
        pruned_rare: 0,
        rejected_ratio_cap: 0,
        rejected_ratio_rare: 0,
        rejected_ratio_hc: 0,
        rejected_ratio_nc: 0,
        rejected_ratio_oc: 0,
        rejected_ratio_hal: 0,
        rejected_ratio_s: 0,
        rejected_ratio_p: 0,
        rejected_ratio_dbe: 0,
        exhausted: false,
    }
}

/// Report from an [`EnumResult`].
fn enumerator_report(r: &EnumResult) -> EnumeratorReport {
    EnumeratorReport {
        nodes_visited: r.nodes_visited,
        hydrogen_checks: r.hydrogen_checks,
        rows_joined: r.rows_joined,
        rows_scored: r.rows_scored,
        rejected_h_max: r.rejected_h_max,
        rejected_parity: r.rejected_parity,
        rejected_dbe: r.rejected_dbe,
        rejected_mass: r.rejected_mass,
        pruned_ratio_cap: r.pruned_ratio_cap,
        pruned_rare: r.pruned_rare,
        rejected_ratio_cap: r.rejected_ratio_cap,
        rejected_ratio_rare: r.rejected_rare,
        rejected_ratio_hc: r.rejected_ratio_hc,
        rejected_ratio_nc: r.rejected_ratio_nc,
        rejected_ratio_oc: r.rejected_ratio_oc,
        rejected_ratio_hal: r.rejected_ratio_hal,
        rejected_ratio_s: r.rejected_ratio_s,
        rejected_ratio_p: r.rejected_ratio_p,
        rejected_ratio_dbe: r.rejected_ratio_dbe,
        exhausted: r.exhausted,
    }
}

/// Run one enumeration under `pruning` with `nodes_visited_max`.
///
/// The enumerator capacity and scored cap are unbounded by design (fix: the
/// capacity cannot bind before selection; a search that hits the node budget
/// reports `exhausted`, which the caller renders as `search_exhausted`,
/// never as a silently complete truncation).
fn run_enumeration(
    artifacts: &FormulaArtifacts,
    chem_domain: &EnumDomain,
    mass: &MassQuery,
    ppm_tenths: u32,
    uncertainty: u32,
    pruning: FormulaPruning,
    nodes_visited_max: u64,
) -> Result<EnumResult> {
    let enum_limits = EnumLimits {
        nodes_visited_max,
        capacity: usize::MAX,
        scored_max: usize::MAX,
        filter_h_max: true,
        filter_parity: true,
        filter_dbe: true,
        ratio: match pruning {
            FormulaPruning::TrainFit => Some(artifacts.bounds.clone()),
            FormulaPruning::ChemicalOnly => None,
        },
    };
    let domain = match pruning {
        FormulaPruning::TrainFit => &artifacts.domain,
        FormulaPruning::ChemicalOnly => chem_domain,
    };
    match mass {
        MassQuery::Neutral { value, .. } => {
            enumerate_neutral(domain, *value, ppm_tenths, uncertainty, &enum_limits)
        }
        MassQuery::Precursor { value, adduct, .. } => {
            let query = super::formula_enum::EnumQuery {
                precursor_mz: *value,
                adduct: *adduct,
                ppm_tenths,
                precursor_uncertainty: uncertainty,
            };
            enumerate(domain, &query, &enum_limits)
        }
    }
}

/// Assign `total` trajectories over the selected formulas with weights
/// `weights` (one per selected formula, non-negative).
///
/// Largest-remainder rounding of `total * w`: each formula gets
/// `floor(total * w)` and the leftover goes one each to the largest
/// fractional parts (ties in selected order), so the assigned sum is exactly
/// `total`. With at least one for every selected formula while the total
/// allows (`total >= selected`): any zero share takes one from the formula
/// holding the most (which then holds at least two, so the loop
/// terminates); when `total < selected` only the first `total` formulas get
/// one each. Uniform weights give the even split with the remainder
/// redistributed one each to the first formulas: a fixed budget is spent,
/// never silently dropped.
fn allocate_trajectories(weights: &[f64], total: u32) -> Vec<u32> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    if (total as usize) < n {
        return (0..n).map(|i| u32::from(i < total as usize)).collect();
    }
    let mut base = vec![0u32; n];
    let mut frac = vec![0.0f64; n];
    let mut assigned: u64 = 0;
    for (i, w) in weights.iter().enumerate() {
        // `w >= 0` and `total * w <= total * 1.0000001 < u32::MAX`.
        let quota = f64::from(total) * w.max(0.0);
        let b = quota.floor() as u32;
        base[i] = b;
        frac[i] = quota - quota.floor();
        assigned += u64::from(b);
    }
    let mut rest = u64::from(total).saturating_sub(assigned);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| frac[*b].total_cmp(&frac[*a]).then_with(|| a.cmp(b)));
    for &i in &order {
        if rest == 0 {
            break;
        }
        base[i] += 1;
        rest -= 1;
    }
    // Repair zeros while the total allows: the shares sum to `total >= n`,
    // so a zero share implies some other share holds at least two.
    while let Some(zero) = base.iter().position(|&t| t == 0) {
        // Donor: the fullest share above one, ties by selected order.
        let mut donor: Option<usize> = None;
        for (i, &t) in base.iter().enumerate() {
            if t > 1 && donor.is_none_or(|d| t > base[d]) {
                donor = Some(i);
            }
        }
        let donor = donor.expect("a zero share with total >= selected implies a donor above one");
        base[donor] -= 1;
        base[zero] += 1;
    }
    base
}

/// Largest `ceil(composition_error_nda / 1000)` over `rows` (`0` when empty).
fn max_composition_term(rows: &[Composition]) -> u32 {
    rows.iter()
        .map(|c| {
            (super::chem::composition_error_nda(c).div_ceil(1000)).min(u64::from(u32::MAX)) as u32
        })
        .max()
        .unwrap_or(0)
}

/// The [`MassCompletionResult`] for a precursor whose neutralisation left
/// the `u32` range: the enumerator's `mass_overflow` status propagated as
/// `mass_evidence.status = "mass_overflow"`, with no search, no candidates
/// and no substituted zero mass.
fn mass_overflow_result(
    enum_result: &EnumResult,
    pruning: FormulaPruning,
    allocation: FormulaAllocation,
    ranking_text: String,
) -> MassCompletionResult {
    MassCompletionResult {
        candidates: Vec::new(),
        distinct_before_cut: 0,
        accounting: PooledAccounting::default(),
        mass_evidence_status: "mass_overflow".to_string(),
        error_terms: Some(MassErrorTerms {
            observation: enum_result.error_observation,
            composition: 0,
            neutralisation: enum_result.error_neutralisation,
        }),
        formula_search: FormulaSearchReport {
            status: "mass_overflow".to_string(),
            truncated: false,
            search_exhausted: false,
            unsampled_reason: None,
            pruning: pruning.as_str().to_string(),
            allocation: allocation.as_str().to_string(),
            joined: 0,
            joined_chemical: None,
            joined_chemical_reason: Some(
                "precursor neutralisation left the u32 range: no search was performed".to_string(),
            ),
            excluded_by_train_fit: None,
            after_domain: 0,
            after_substructures: 0,
            after_completability: 0,
            selected: 0,
            sampled: 0,
            ranking: ranking_text,
            stages: Vec::new(),
            formulas: Vec::new(),
            enumerator: enumerator_report(enum_result),
        },
        joined_compositions: Vec::new(),
        beam_row_steps: 0,
        beam_candidates_dropped: 0,
    }
}

/// Run the mass-to-formulas-to-generation path.
///
/// Enumerates with every exact chemical filter on and the `pruning` bounds
/// (the checkpoint's train-fit bounds, or the model-`max_atoms` chemical
/// domain only), filters through the model-domain, substructure and
/// completability stages, orders survivors by verdict (`Accept` before
/// `Ambiguous`), then absolute mass residual, then the enumerator's
/// canonical order — a deterministic default, not a formula probability —
/// and samples the first `hypotheses` under the total `trajectories` budget
/// per `allocation` (the assigned sum is exactly `total` whenever anything
/// is selected; when `total < selected` only the first `total` formulas get
/// one trajectory each and the rest are `not_sampled`). Formulas sharing one
/// assigned trajectory count generate in one batched call. Pooled identities
/// rank by `samples` descending, then `best_log_prob` descending, then
/// formula order, then trace order under [`FormulaAllocation::Equal`]; under
/// [`FormulaAllocation::TrainFrequency`] they rank by the explicit estimate
/// `weight * samples / trajectories` first (ties as in the equal order).
/// Different formulas never share an identity (asserted in debug builds).
/// There is no learned formula ranker.
#[allow(clippy::too_many_arguments)]
pub fn run_mass_completion<R: Runtime>(
    model: &CompletionModel<R, f32>,
    constants: &Ms2Constants<R>,
    device: &Device<R>,
    artifacts: &FormulaArtifacts,
    max_atoms: u32,
    max_ring_closures: u32,
    patterns: &[super::graph::MolGraph],
    acceptance_patterns: Option<&[super::graph::MolGraph]>,
    mass: &MassQuery,
    hypotheses: u32,
    nodes_visited_max: u64,
    total_trajectories: u32,
    temperature: f32,
    seed: u64,
    returned: u32,
    request_id: &str,
    condition_on_patterns: bool,
    pruning: FormulaPruning,
    allocation: FormulaAllocation,
    semantics: SubstructureSemantics,
    fingerprint: Option<&super::completion_fingerprint::SparseFingerprint>,
) -> Result<MassCompletionResult> {
    run_mass_completion_with_spectrum(
        model,
        constants,
        device,
        artifacts,
        max_atoms,
        max_ring_closures,
        patterns,
        acceptance_patterns,
        mass,
        hypotheses,
        nodes_visited_max,
        total_trajectories,
        temperature,
        seed,
        returned,
        request_id,
        condition_on_patterns,
        pruning,
        allocation,
        semantics,
        fingerprint,
        None,
    )
}

/// [`run_mass_completion`] with optional spectral evidence: every formula
/// hypothesis is sampled conditioned on the same
/// [`SpectrumEvidence`](super::completion_spectrum::SpectrumEvidence)
/// (peaks, adduct, neutral mass). The evidence never changes the formula
/// search, acceptance or ranking; a model without a spectrum encoder given
/// evidence is [`Error::Config`].
#[allow(clippy::too_many_arguments)]
pub fn run_mass_completion_with_spectrum<R: Runtime>(
    model: &CompletionModel<R, f32>,
    constants: &Ms2Constants<R>,
    device: &Device<R>,
    artifacts: &FormulaArtifacts,
    max_atoms: u32,
    max_ring_closures: u32,
    patterns: &[super::graph::MolGraph],
    acceptance_patterns: Option<&[super::graph::MolGraph]>,
    mass: &MassQuery,
    hypotheses: u32,
    nodes_visited_max: u64,
    total_trajectories: u32,
    temperature: f32,
    seed: u64,
    returned: u32,
    request_id: &str,
    condition_on_patterns: bool,
    pruning: FormulaPruning,
    allocation: FormulaAllocation,
    semantics: SubstructureSemantics,
    fingerprint: Option<&super::completion_fingerprint::SparseFingerprint>,
    spectrum: Option<&super::completion_spectrum::SpectrumEvidence>,
) -> Result<MassCompletionResult> {
    run_mass_completion_search(
        model,
        constants,
        device,
        artifacts,
        max_atoms,
        max_ring_closures,
        patterns,
        acceptance_patterns,
        mass,
        hypotheses,
        nodes_visited_max,
        total_trajectories,
        temperature,
        seed,
        returned,
        request_id,
        condition_on_patterns,
        pruning,
        allocation,
        semantics,
        fingerprint,
        spectrum,
        CompletionSearch::Sampling,
    )
}

/// Predicted element counts of one query: a learned ordering of the formula
/// hypotheses of its mass.
///
/// The default selection orders hypotheses by absolute mass residual and
/// keeps the nearest `hypotheses`. Many formulas fit one measured mass (tens
/// at 10 ppm near 350 Da), so the answer's formula is often not among the
/// nearest few and each kept formula receives a small share of the search.
/// A model that predicts the element counts from the spectrum orders the
/// same hypotheses far better; this type carries that prediction.
///
/// With a prior, hypotheses are ordered by verdict, then by
/// [`ElementPrior::distance`], then by mass residual and canonical order, and
/// the trajectory budget is split in proportion to
/// `exp(-(distance - nearest) / temperature)` (largest remainder; as under
/// every allocation, a selected formula keeps at least one row while the
/// budget allows), so a formula far from the prediction receives a single
/// row. Nothing else changes: enumeration, pruning, acceptance and pooling
/// are the same code.
/// The prior is an estimate, not a calibrated formula probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementPrior {
    /// Predicted `ln(1 + count)` per element in [`ELEMENTS`] order
    /// (C, H, N, O, F, P, S, Cl, Br, I).
    pub log1p_counts: [f32; 10],
    /// Temperature of the trajectory weights; must be positive and finite.
    pub temperature: f32,
}

impl ElementPrior {
    /// L1 distance between a composition and the prediction in
    /// `ln(1 + count)` space (the space the predictor is trained in).
    pub fn distance(&self, c: &Composition) -> f64 {
        let mut total = 0.0f64;
        for e in 0..10 {
            total += ((f64::from(c[e])).ln_1p() - f64::from(self.log1p_counts[e])).abs();
        }
        total
    }

    /// Trajectory weights of hypotheses at the given distances:
    /// `exp(-(d - min d) / temperature)`, normalised. Empty for no hypothesis.
    pub fn weights(&self, distances: &[f64]) -> Vec<f64> {
        let nearest = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let t = f64::from(self.temperature);
        let raw: Vec<f64> = distances.iter().map(|d| (-(d - nearest) / t).exp()).collect();
        let total: f64 = raw.iter().sum();
        raw.iter().map(|w| w / total).collect()
    }
}

/// [`run_mass_completion_with_spectrum`] with the search mode chosen by the
/// caller: `Sampling` is that function, and `Beam` replaces each
/// hypothesis's independent samples by a beam of the same number of rows
/// ([`CompletionModel::generate_beam_with_spectra`]). The formula
/// enumeration, the trajectory allocation, acceptance, pooling and the
/// ranking are the same code either way; only the generator differs, so the
/// two arms are comparable at equal rows.
#[allow(clippy::too_many_arguments)]
pub fn run_mass_completion_search<R: Runtime>(
    model: &CompletionModel<R, f32>,
    constants: &Ms2Constants<R>,
    device: &Device<R>,
    artifacts: &FormulaArtifacts,
    max_atoms: u32,
    max_ring_closures: u32,
    patterns: &[super::graph::MolGraph],
    acceptance_patterns: Option<&[super::graph::MolGraph]>,
    mass: &MassQuery,
    hypotheses: u32,
    nodes_visited_max: u64,
    total_trajectories: u32,
    temperature: f32,
    seed: u64,
    returned: u32,
    request_id: &str,
    condition_on_patterns: bool,
    pruning: FormulaPruning,
    allocation: FormulaAllocation,
    semantics: SubstructureSemantics,
    fingerprint: Option<&super::completion_fingerprint::SparseFingerprint>,
    spectrum: Option<&super::completion_spectrum::SpectrumEvidence>,
    search: CompletionSearch,
) -> Result<MassCompletionResult> {
    run_mass_completion_search_with_prior(
        model,
        constants,
        device,
        artifacts,
        max_atoms,
        max_ring_closures,
        patterns,
        acceptance_patterns,
        mass,
        hypotheses,
        nodes_visited_max,
        total_trajectories,
        temperature,
        seed,
        returned,
        request_id,
        condition_on_patterns,
        pruning,
        allocation,
        semantics,
        fingerprint,
        spectrum,
        search,
        None,
    )
}

/// [`run_mass_completion_search`] with an optional [`ElementPrior`] that
/// orders the formula hypotheses and weights their trajectory shares. With
/// `None` this is exactly [`run_mass_completion_search`].
#[allow(clippy::too_many_arguments)]
pub fn run_mass_completion_search_with_prior<R: Runtime>(
    model: &CompletionModel<R, f32>,
    constants: &Ms2Constants<R>,
    device: &Device<R>,
    artifacts: &FormulaArtifacts,
    max_atoms: u32,
    max_ring_closures: u32,
    patterns: &[super::graph::MolGraph],
    acceptance_patterns: Option<&[super::graph::MolGraph]>,
    mass: &MassQuery,
    hypotheses: u32,
    nodes_visited_max: u64,
    total_trajectories: u32,
    temperature: f32,
    seed: u64,
    returned: u32,
    request_id: &str,
    condition_on_patterns: bool,
    pruning: FormulaPruning,
    allocation: FormulaAllocation,
    semantics: SubstructureSemantics,
    fingerprint: Option<&super::completion_fingerprint::SparseFingerprint>,
    spectrum: Option<&super::completion_spectrum::SpectrumEvidence>,
    search: CompletionSearch,
    element_prior: Option<&ElementPrior>,
) -> Result<MassCompletionResult> {
    if let Some(prior) = element_prior {
        if !(prior.temperature.is_finite() && prior.temperature > 0.0)
            || prior.log1p_counts.iter().any(|v| !v.is_finite())
        {
            return Err(Error::config(
                "element prior needs finite counts and a positive finite temperature".to_string(),
            ));
        }
    }
    let mut beam_row_steps = 0u64;
    let mut beam_candidates_dropped = 0u64;
    let ranking_text = match allocation {
        FormulaAllocation::Equal => "verdict, then absolute mass residual, then canonical order; trajectories split evenly with the remainder redistributed; deterministic default, not a formula probability".to_string(),
        FormulaAllocation::TrainFrequency => "verdict, then absolute mass residual, then canonical order; trajectories by training-frequency largest-remainder weights; weights are a training-frequency prior, not a calibrated probability".to_string(),
    };
    let ranking_text = match element_prior {
        Some(prior) => format!(
            "verdict, then L1 distance to the predicted ln(1+count) element counts, then absolute mass residual, then canonical order; trajectories by exp(-(distance - nearest)/{}) largest-remainder weights; an estimate, not a calibrated formula probability",
            prior.temperature
        ),
        None => ranking_text,
    };
    // Unknown precision: no search, no candidates, never an exception.
    let unknown = match mass {
        MassQuery::Neutral { uncertainty, .. } => uncertainty.is_none(),
        MassQuery::Precursor { uncertainty, .. } => uncertainty.is_none(),
    };
    if unknown {
        return Ok(MassCompletionResult {
            candidates: Vec::new(),
            distinct_before_cut: 0,
            accounting: PooledAccounting::default(),
            mass_evidence_status: "unavailable".to_string(),
            error_terms: None,
            formula_search: FormulaSearchReport {
                status: "unavailable".to_string(),
                truncated: false,
                search_exhausted: false,
                unsampled_reason: None,
                pruning: pruning.as_str().to_string(),
                allocation: allocation.as_str().to_string(),
                joined: 0,
                joined_chemical: None,
                joined_chemical_reason: Some(
                    "unknown mass precision: no search was performed".to_string(),
                ),
                excluded_by_train_fit: None,
                after_domain: 0,
                after_substructures: 0,
                after_completability: 0,
                selected: 0,
                sampled: 0,
                ranking: ranking_text,
                stages: Vec::new(),
                formulas: Vec::new(),
                enumerator: empty_enumerator(),
            },
            joined_compositions: Vec::new(),
            beam_row_steps: 0,
            beam_candidates_dropped: 0,
        });
    }
    let (ppm_tenths, uncertainty) = match mass {
        MassQuery::Neutral {
            ppm_tenths,
            uncertainty,
            ..
        } => (*ppm_tenths, uncertainty.unwrap()),
        MassQuery::Precursor {
            ppm_tenths,
            uncertainty,
            ..
        } => (*ppm_tenths, uncertainty.unwrap()),
    };
    let limits = Limits::new(max_atoms as usize, max_ring_closures as usize)?;
    let chem_domain = chemical_only_domain(max_atoms);
    // Primary enumeration under the requested pruning.
    let enum_result: EnumResult = run_enumeration(
        artifacts,
        &chem_domain,
        mass,
        ppm_tenths,
        uncertainty,
        pruning,
        nodes_visited_max,
    )?;
    // Precursor neutralisation that leaves the `u32` range: propagate the
    // enumerator's `mass_overflow` status. Never substitute a zero mass and
    // search around it.
    let Some(parent) = enum_result.parent_mass else {
        return Ok(mass_overflow_result(
            &enum_result,
            pruning,
            allocation,
            ranking_text,
        ));
    };
    let joined = enum_result.compositions.len();
    // Second enumeration under the other pruning, for the chemical-only
    // count and the train-fit exclusion difference. It runs when the primary
    // search joined fewer than the rerun row cap and the node budget allows;
    // otherwise the counts are null with a reason. The rerun has its own
    // bounded capacity (`CHEMICAL_RERUN_JOIN_CAP + 1`, so an over-cap join
    // reports exhaustion instead of growing memory) and its own node budget
    // (at most `CHEMICAL_RERUN_NODES_MAX`, capped by what the primary left);
    // when its own limit binds the counts are null with a reason naming it.
    let other = match pruning {
        FormulaPruning::TrainFit => FormulaPruning::ChemicalOnly,
        FormulaPruning::ChemicalOnly => FormulaPruning::TrainFit,
    };
    let remaining_nodes = nodes_visited_max.saturating_sub(enum_result.nodes_visited);
    enum ChemicalCount {
        /// The rerun's joined count plus its verdict split (accept rows).
        Available { rerun_joined: usize, accept: usize },
        /// Unavailable with the reason.
        Unavailable(String),
    }
    let chemical_count = if joined > CHEMICAL_RERUN_JOIN_CAP {
        ChemicalCount::Unavailable(format!(
            "primary search joined {joined} rows above the {CHEMICAL_RERUN_JOIN_CAP} rerun row cap"
        ))
    } else if remaining_nodes == 0 {
        ChemicalCount::Unavailable("node budget exhausted by the primary search".to_string())
    } else {
        let rerun_nodes = remaining_nodes.min(CHEMICAL_RERUN_NODES_MAX);
        let rerun_limits = EnumLimits {
            nodes_visited_max: rerun_nodes,
            capacity: CHEMICAL_RERUN_JOIN_CAP.saturating_add(1),
            scored_max: CHEMICAL_RERUN_JOIN_CAP.saturating_add(1),
            filter_h_max: true,
            filter_parity: true,
            filter_dbe: true,
            ratio: match other {
                FormulaPruning::TrainFit => Some(artifacts.bounds.clone()),
                FormulaPruning::ChemicalOnly => None,
            },
        };
        let rerun_domain = match other {
            FormulaPruning::TrainFit => &artifacts.domain,
            FormulaPruning::ChemicalOnly => &chem_domain,
        };
        let rerun = match mass {
            MassQuery::Neutral { value, .. } => super::formula_enum::enumerate_neutral(
                rerun_domain,
                *value,
                ppm_tenths,
                uncertainty,
                &rerun_limits,
            )?,
            MassQuery::Precursor { value, adduct, .. } => {
                let query = super::formula_enum::EnumQuery {
                    precursor_mz: *value,
                    adduct: *adduct,
                    ppm_tenths,
                    precursor_uncertainty: uncertainty,
                };
                super::formula_enum::enumerate(rerun_domain, &query, &rerun_limits)?
            }
        };
        if rerun.exhausted {
            ChemicalCount::Unavailable(format!(
                "the {} rerun bound its own limit (node budget {rerun_nodes} or row capacity {})",
                other.as_str(),
                CHEMICAL_RERUN_JOIN_CAP.saturating_add(1),
            ))
        } else {
            let rerun_joined = rerun.compositions.len();
            let accept = rerun.ambiguous.iter().filter(|a| !**a).count();
            ChemicalCount::Available {
                rerun_joined,
                accept,
            }
        }
    };
    // The rerun verdicts behind the mass-evidence fallback: under
    // `train_fit` pruning the rerun is the chemical-only (superset) search,
    // whose verdicts say whether any formula passed the mass verdict at
    // all; under `chemical_only` the primary already is that search.
    let rerun_verdicts: Option<(usize, usize)> = match &chemical_count {
        ChemicalCount::Available {
            rerun_joined,
            accept,
        } => Some((*accept, rerun_joined.saturating_sub(*accept))),
        ChemicalCount::Unavailable(_) => None,
    };
    let (joined_chemical, joined_chemical_reason, excluded_by_train_fit) = match chemical_count {
        ChemicalCount::Available { rerun_joined, .. } => {
            let (chemical, train) = match pruning {
                FormulaPruning::TrainFit => (rerun_joined, joined),
                FormulaPruning::ChemicalOnly => (joined, rerun_joined),
            };
            // The train-fit bounds only remove rows, so the chemical-only
            // count dominates; saturate rather than wrap on any surprise.
            debug_assert!(
                chemical >= train,
                "train-fit bounds only remove rows: chemical {chemical} under train-fit {train}"
            );
            (Some(chemical), None, Some(chemical.saturating_sub(train)))
        }
        ChemicalCount::Unavailable(reason) => (None, Some(reason), None),
    };
    // Stage b: model domain (heavy atoms <= max_atoms).
    let mut after_domain_idx: Vec<usize> = Vec::new();
    for (i, c) in enum_result.compositions.iter().enumerate() {
        if heavy_total(c) <= max_atoms {
            after_domain_idx.push(i);
        }
    }
    let after_domain = after_domain_idx.len();
    // Stage c: substructure lower bound (necessary for containment under
    // the active semantics: summed over patterns for the disjoint modes,
    // with heteroatom equality for the complete mode).
    let mut after_sub_idx: Vec<usize> = Vec::new();
    let accept_patterns: &[super::graph::MolGraph] = acceptance_patterns.unwrap_or(patterns);
    for &i in &after_domain_idx {
        if passes_substructure(accept_patterns, &enum_result.compositions[i], semantics) {
            after_sub_idx.push(i);
        }
    }
    let after_substructures = after_sub_idx.len();
    // Stage d: completability pre-check on the empty state.
    let mut after_comp_idx: Vec<usize> = Vec::new();
    for &i in &after_sub_idx {
        let c = enum_result.compositions[i];
        let state = TraceState::new_exact(limits, c);
        if state.feasibility().all() {
            after_comp_idx.push(i);
        }
    }
    let after_completability = after_comp_idx.len();
    // Stage e: order by verdict (Accept before Ambiguous), then absolute
    // mass residual, then the enumerator's canonical order. Ambiguous rows
    // are never dropped. This order is a deterministic default, not a
    // formula probability.
    let mut ordered: Vec<(bool, u32, usize)> = Vec::with_capacity(after_comp_idx.len());
    for &i in &after_comp_idx {
        let mass_i = enum_result.masses[i];
        let residual = parent.abs_diff(mass_i);
        ordered.push((enum_result.ambiguous[i], residual, i));
    }
    match element_prior {
        None => ordered.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        }),
        // With an element prior the predicted-count distance leads inside a
        // verdict; the default keys break its ties.
        Some(prior) => ordered.sort_by(|a, b| {
            let da = prior.distance(&enum_result.compositions[a.2]);
            let db = prior.distance(&enum_result.compositions[b.2]);
            a.0.cmp(&b.0)
                .then_with(|| da.total_cmp(&db))
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        }),
    }
    let hypotheses_n = hypotheses as usize;
    let selected_n = ordered.len().min(hypotheses_n);
    let truncated = ordered.len() > selected_n;
    // `search_exhausted` takes precedence over `truncated`: when both bind,
    // a nearer formula may remain unvisited. Both facts stay visible as
    // independent booleans.
    let search_exhausted = enum_result.exhausted;
    let status = if search_exhausted {
        "search_exhausted".to_string()
    } else if truncated {
        "truncated".to_string()
    } else {
        "complete".to_string()
    };
    let selected_ordered: Vec<usize> = ordered
        .iter()
        .take(selected_n)
        .map(|(_, _, i)| *i)
        .collect();
    // Selection weights: uniform under `equal`, the training-frequency
    // prior under `train_frequency` (a prior, not a calibrated
    // probability; unseen formulas count 0, smoothed by alpha = 1).
    let weights: Vec<f64> = match allocation {
        FormulaAllocation::Equal => {
            vec![
                if selected_n == 0 {
                    0.0
                } else {
                    1.0 / selected_n as f64
                };
                selected_n
            ]
        }
        FormulaAllocation::TrainFrequency => {
            let mut smoothed = Vec::with_capacity(selected_n);
            for &enum_pos in &selected_ordered {
                let text = formula_text(&enum_result.compositions[enum_pos]);
                smoothed.push(artifacts.composition_count(&text) as f64 + 1.0);
            }
            let total: f64 = smoothed.iter().sum();
            if total > 0.0 {
                smoothed.iter().map(|w| w / total).collect()
            } else {
                vec![
                    if selected_n == 0 {
                        0.0
                    } else {
                        1.0 / selected_n as f64
                    };
                    selected_n
                ]
            }
        }
    };
    // An element prior replaces the weights of either allocation.
    let weights: Vec<f64> = match element_prior {
        Some(prior) if selected_n > 0 => {
            let distances: Vec<f64> = selected_ordered
                .iter()
                .map(|&enum_pos| prior.distance(&enum_result.compositions[enum_pos]))
                .collect();
            prior.weights(&distances)
        }
        _ => weights,
    };
    // Stage f: budget split. Largest-remainder rounding of the weights (an
    // even split with the remainder redistributed under `equal`); the
    // assigned sum is exactly the total whenever anything is selected.
    let assigned = allocate_trajectories(&weights, total_trajectories);
    debug_assert_eq!(assigned.len(), selected_n);
    let sampled_idx: Vec<usize> = (0..selected_n).filter(|&pos| assigned[pos] > 0).collect();
    let sampled_n = sampled_idx.len();
    // Generate grouped by assigned trajectory count, so each batched call
    // has a uniform `K`. Ids stay distinct per formula (request id, formula
    // text, selected position); the seed is shared, so sampling stays
    // deterministic without sharing streams.
    let mut outcomes: Vec<QueryOutcome> = Vec::new();
    let mut outcome_by_pos: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut beam_stats_by_pos = std::collections::HashMap::new();
    if !sampled_idx.is_empty() {
        let mut groups: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for &pos in &sampled_idx {
            groups.entry(assigned[pos]).or_default().push(pos);
        }
        for (&k, positions) in &groups {
            let gen_config = CompletionGenerationConfig {
                trajectories: k,
                temperature,
                seed,
                returned,
                containment_node_limit: 100_000,
                identity_work_limit: 100_000,
                condition_on_patterns,
                substructure_semantics: semantics,
            };
            let req_comps: Vec<Composition> = positions
                .iter()
                .map(|&pos| enum_result.compositions[selected_ordered[pos]])
                .collect();
            let mut owned_ids: Vec<u64> = Vec::with_capacity(positions.len());
            for &pos in positions {
                let enum_pos = selected_ordered[pos];
                let text = formula_text(&enum_result.compositions[enum_pos]);
                owned_ids.push(numeric_id_hex(&format!("{request_id}:{text}:{pos}")));
            }
            {
                let mut seen = std::collections::HashSet::new();
                for id in &owned_ids {
                    debug_assert!(seen.insert(*id), "derived formula ids must be distinct");
                }
            }
            let mut reqs: Vec<CompletionRequest<'_>> = Vec::with_capacity(positions.len());
            for (n, &id) in owned_ids.iter().enumerate() {
                reqs.push(CompletionRequest {
                    id,
                    composition: req_comps[n],
                    patterns,
                    acceptance_patterns,
                    fingerprint,
                });
            }
            let spectra = vec![spectrum; reqs.len()];
            let mut group_outcomes = match search {
                CompletionSearch::Sampling => {
                    model.generate_with_spectra(&reqs, &spectra, &gen_config, constants, device)?
                }
                CompletionSearch::Beam => {
                    // The formula's share of the total rows is its width.
                    let (group, stats) = model.generate_beam_with_spectra(
                        &reqs,
                        &spectra,
                        &gen_config,
                        k,
                        constants,
                        device,
                    )?;
                    for (&pos, report) in positions.iter().zip(stats.iter()) {
                        beam_row_steps += report.row_steps;
                        beam_candidates_dropped += report.candidates_dropped;
                        beam_stats_by_pos.insert(pos, report.clone());
                    }
                    group
                }
            };
            for (&pos, outcome) in positions.iter().zip(group_outcomes.drain(..)) {
                outcome_by_pos.insert(pos, outcomes.len());
                outcomes.push(outcome);
            }
        }
    }
    // Assemble formula entries in selected order.
    let mut formulas: Vec<FormulaEntry> = Vec::with_capacity(selected_n);
    for (pos, &enum_pos) in selected_ordered.iter().enumerate() {
        let comp = enum_result.compositions[enum_pos];
        let computed = enum_result.masses[enum_pos];
        let residual = parent.abs_diff(computed);
        let ambiguous = enum_result.ambiguous[enum_pos];
        let (traj, fin, acc) = match outcome_by_pos.get(&pos) {
            Some(&o) => {
                let oc = &outcomes[o];
                (oc.trajectories, oc.finished, oc.candidates.len() as u32)
            }
            None => (0, 0, 0),
        };
        formulas.push(FormulaEntry {
            composition: comp,
            formula: formula_text(&comp),
            computed_uda: computed,
            residual_uda: residual,
            ambiguous,
            weight: weights[pos],
            trajectories: traj,
            finished: fin,
            accepted_candidates: acc,
            beam_stats: beam_stats_by_pos.remove(&pos),
        });
    }
    // Pool accepted identities across formulas.
    struct Pooled {
        graph: super::graph::MolGraph,
        trace: Vec<super::grammar::Token>,
        samples: u32,
        best: f32,
        formula_order: usize,
        formula: String,
        computed: u32,
        residual: u32,
        ambiguous: bool,
        /// Explicit ranking estimate under `train_frequency`
        /// (`weight * samples / trajectories` of the source formula).
        estimate: f64,
    }
    let mut pooled: Vec<Pooled> = Vec::new();
    let mut accounting = PooledAccounting::default();
    // F1: index outcomes by selected position (`outcome_by_pos`), never by
    // batch order. Grouped generation stores outcomes in trajectory-count
    // group order, so `outcomes[o]` in selected-formula order mislabels
    // pooled candidates with another formula's mass metadata and ranking
    // weight whenever allocations differ (e.g. `[22, 21, 21]`).
    for &pos in &sampled_idx {
        let o = outcome_by_pos[&pos];
        let oc = &outcomes[o];
        accounting.trajectories += oc.trajectories;
        accounting.finished += oc.finished;
        accounting.dead_end += oc.dead_end;
        accounting.truncated += oc.truncated;
        accounting.other_status += oc.other_status;
        accounting.rejected_replay += oc.rejected_replay;
        accounting.rejected_containment += oc.rejected_containment;
        accounting.containment_unresolved += oc.containment_unresolved;
        accounting.rejected_extra_groups += oc.rejected_extra_groups;
        accounting.rejected_missing_groups += oc.rejected_missing_groups;
        accounting.pass_contained += oc.pass_contained;
        accounting.pass_disjoint += oc.pass_disjoint;
        accounting.pass_complete += oc.pass_complete;
        accounting.identity_unresolved += oc.identity_unresolved;
        accounting.distinct += oc.distinct;
        accounting.unresolved += oc.unresolved.len() as u32;
        let entry = &formulas[pos];
        for cand in &oc.candidates {
            let rebuilt = super::graph::MolGraph::new(
                cand.graph.atoms().to_vec(),
                cand.graph.bonds().to_vec(),
            )
            .expect("pooled candidate graph rebuilds");
            // Release-build invariant (an `Error`, not a `debug_assert`):
            // every pooled candidate's graph composition equals its source
            // formula. A mismatch means pooled mass metadata or ranking
            // weights are misattributed (F1) and must fail loudly.
            if rebuilt.composition() != entry.composition {
                return Err(Error::config(format!(
                    "mass completion pooled a candidate of composition {} under source formula {}",
                    formula_text(&rebuilt.composition()),
                    entry.formula,
                )));
            }
            // The estimate behind the `train_frequency` ranking: the prior
            // weight times the empirical hit rate of the source formula.
            // `entry.trajectories > 0` here (only sampled formulas produce
            // candidates), so the division is exact.
            let estimate = if entry.trajectories > 0 {
                entry.weight * f64::from(cand.samples) / f64::from(entry.trajectories)
            } else {
                0.0
            };
            pooled.push(Pooled {
                graph: rebuilt,
                trace: cand.trace.clone(),
                samples: cand.samples,
                best: cand.best_log_prob,
                formula_order: pos,
                formula: entry.formula.clone(),
                computed: entry.computed_uda,
                residual: entry.residual_uda,
                ambiguous: entry.ambiguous,
                estimate,
            });
        }
    }
    // Different formulas never share an identity (compositions differ).
    debug_assert!(
        {
            let mut ok = true;
            for i in 0..pooled.len() {
                for j in (i + 1)..pooled.len() {
                    if pooled[i].formula != pooled[j].formula
                        && pooled[i].graph.composition() == pooled[j].graph.composition()
                    {
                        ok = false;
                    }
                }
            }
            ok
        },
        "different formulas can never be the same identity"
    );
    pooled.sort_by(|a, b| {
        // Under `train_frequency` the explicit estimate leads; ties fall
        // through to the equal-allocation order. Under `equal` the estimate
        // is skipped outright, so the order is exactly samples, log-prob,
        // formula order, trace order.
        if allocation == FormulaAllocation::TrainFrequency {
            let ord = b.estimate.total_cmp(&a.estimate);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        b.samples
            .cmp(&a.samples)
            .then_with(|| b.best.total_cmp(&a.best))
            .then_with(|| a.formula_order.cmp(&b.formula_order))
            .then_with(|| a.trace.cmp(&b.trace))
    });
    let distinct_before_cut = pooled.len() as u32;
    let take_n = (returned as usize).min(pooled.len());
    let mut candidates: Vec<PooledCandidate> = Vec::with_capacity(take_n);
    for p in pooled.into_iter().take(take_n) {
        let mass_status = if p.ambiguous {
            "boundary_ambiguous".to_string()
        } else {
            "accepted".to_string()
        };
        let composition = p.graph.composition();
        candidates.push(PooledCandidate {
            atoms: p.graph.atoms().to_vec(),
            bonds: p.graph.bonds().to_vec(),
            composition,
            trace: p.trace,
            samples: p.samples,
            best_log_prob: p.best,
            formula: p.formula,
            computed_uda: p.computed,
            residual_uda: p.residual,
            mass_status,
            formula_order: p.formula_order,
        });
    }
    // Mass evidence, derived only from the formula search itself — never
    // from whether sampling happened. Per-row verdicts are exact, so joined
    // rows speak even when the search hit a bound; `rejected` needs a
    // completed search with nothing joining on mass (under `train_fit`
    // pruning the chemical-only rerun must complete empty too —
    // train-fit-only exclusions are `by_train_fit`, never mass rejection).
    let primary_accept = enum_result.ambiguous.iter().filter(|a| !**a).count();
    let mass_evidence_status = if joined > 0 {
        if primary_accept > 0 {
            "accepted".to_string()
        } else {
            "boundary_ambiguous".to_string()
        }
    } else if enum_result.exhausted {
        "search_incomplete".to_string()
    } else {
        match pruning {
            FormulaPruning::ChemicalOnly => "rejected".to_string(),
            FormulaPruning::TrainFit => match rerun_verdicts {
                Some((accept, ambiguous)) => {
                    if accept > 0 {
                        "accepted".to_string()
                    } else if ambiguous > 0 {
                        "boundary_ambiguous".to_string()
                    } else {
                        "rejected".to_string()
                    }
                }
                // The diagnostic rerun could not disambiguate (its own
                // limit bound or the row-cap skip): absence proves nothing.
                None => "search_incomplete".to_string(),
            },
        }
    };
    // Why formulas did not all sample: `budget` when at least one selected
    // formula received zero trajectories (the total did not cover the
    // selection); otherwise, when nothing sampled although mass-level
    // formulas joined, the first stage that left zero rows. `None` when
    // every selected formula sampled.
    let unsampled_reason = if sampled_n < selected_n {
        Some("budget".to_string())
    } else if sampled_n == 0 {
        if joined == 0 {
            match pruning {
                FormulaPruning::TrainFit => match joined_chemical {
                    Some(c) if c > 0 => Some("by_train_fit".to_string()),
                    _ => None,
                },
                FormulaPruning::ChemicalOnly => None,
            }
        } else if after_domain == 0 {
            Some("all_excluded_by_domain".to_string())
        } else if after_substructures == 0 {
            Some("by_substructures".to_string())
        } else if after_completability == 0 {
            Some("by_completability".to_string())
        } else {
            // `selected == 0` with survivors is unreachable
            // (`hypotheses >= 1` always selects); kept as `budget` rather
            // than silence.
            Some("budget".to_string())
        }
    } else {
        None
    };
    // Named §5 error parts: the query's observation and neutralisation terms
    // plus the largest composition term over the joined rows.
    let error_terms = Some(MassErrorTerms {
        observation: enum_result.error_observation,
        composition: max_composition_term(&enum_result.compositions),
        neutralisation: enum_result.error_neutralisation,
    });
    // Per-stage entering/leaving counts with their necessary / empirical /
    // budget class.
    let verdict_passing = enum_result
        .hydrogen_checks
        .saturating_sub(enum_result.rejected_mass);
    let exact_passing = verdict_passing.saturating_sub(
        enum_result
            .rejected_h_max
            .saturating_add(enum_result.rejected_parity)
            .saturating_add(enum_result.rejected_dbe),
    );
    let ratio_rejects = enum_result
        .rejected_ratio_cap
        .saturating_add(enum_result.rejected_rare)
        .saturating_add(enum_result.rejected_ratio_hc)
        .saturating_add(enum_result.rejected_ratio_nc)
        .saturating_add(enum_result.rejected_ratio_oc)
        .saturating_add(enum_result.rejected_ratio_hal)
        .saturating_add(enum_result.rejected_ratio_s)
        .saturating_add(enum_result.rejected_ratio_p)
        .saturating_add(enum_result.rejected_ratio_dbe);
    debug_assert_eq!(
        exact_passing.saturating_sub(ratio_rejects) as usize,
        joined,
        "enumerator joins are verdict-passing rows minus exact and train-fit rejects"
    );
    let train_fit_note = match pruning {
        FormulaPruning::TrainFit => match joined_chemical {
            Some(chemical) => format!(
                "leaf stages excluded {chemical} - {joined} chemical-only rows (DFS skips in train_fit_dfs_pruning)"
            ),
            None => format!(
                "chemical-only count unavailable: {}",
                joined_chemical_reason.as_deref().unwrap_or("unknown")
            ),
        },
        FormulaPruning::ChemicalOnly => {
            "skipped under chemical_only pruning: joined counts are chemical-only".to_string()
        }
    };
    // The DFS pruning line reports the branches skipped before any verdict
    // (empirical, not rows): no row count changes across it.
    let dfs_note = match pruning {
        FormulaPruning::TrainFit => format!(
            "DFS branches skipped by the train-fit (i)/(iii) maxima before any verdict: pruned_ratio_cap {} + pruned_rare {} (no rows change across this line)",
            enum_result.pruned_ratio_cap, enum_result.pruned_rare
        ),
        FormulaPruning::ChemicalOnly => {
            "skipped under chemical_only pruning: no train-fit DFS maxima apply".to_string()
        }
    };
    let allocation_note = match allocation {
        FormulaAllocation::Equal => {
            "even split with the remainder redistributed one each to the first formulas".to_string()
        }
        FormulaAllocation::TrainFrequency => {
            "largest-remainder of total * weight with at least one per formula while the total allows".to_string()
        }
    };
    let stages = vec![
        SearchStage {
            stage: "mass_verdict".to_string(),
            class: "necessary".to_string(),
            entering: enum_result.hydrogen_checks as usize,
            leaving: verdict_passing as usize,
            note: "contract §5 verdict per hydrogen count".to_string(),
        },
        SearchStage {
            stage: "exact_chemical_filters".to_string(),
            class: "necessary".to_string(),
            entering: verdict_passing as usize,
            leaving: exact_passing as usize,
            note: "hydrogen ceiling, parity, DBE >= 0 (within-traversal count)".to_string(),
        },
        SearchStage {
            stage: "train_fit_dfs_pruning".to_string(),
            class: "empirical".to_string(),
            entering: exact_passing as usize,
            leaving: exact_passing as usize,
            note: dfs_note,
        },
        SearchStage {
            stage: "train_fit_bounds".to_string(),
            class: "empirical".to_string(),
            entering: exact_passing as usize,
            leaving: joined,
            note: train_fit_note,
        },
        SearchStage {
            stage: "model_domain".to_string(),
            class: "necessary".to_string(),
            entering: joined,
            leaving: after_domain,
            note: format!("heavy atoms <= max_atoms {max_atoms}"),
        },
        SearchStage {
            stage: "substructure_bound".to_string(),
            class: "necessary".to_string(),
            entering: after_domain,
            leaving: after_substructures,
            note: match semantics {
                SubstructureSemantics::Contained => {
                    "per-element lower bound, necessary for containment".to_string()
                }
                SubstructureSemantics::DisjointOccurrences => {
                    "summed-element lower bound, necessary for disjoint containment".to_string()
                }
                SubstructureSemantics::CompleteFunctionalGroups => {
                    "summed-element lower bound with heteroatom equality, necessary for complete functional groups"
                        .to_string()
                }
            },
        },
        SearchStage {
            stage: "completability".to_string(),
            class: "necessary".to_string(),
            entering: after_substructures,
            leaving: after_completability,
            note: "empty-state feasibility pre-check".to_string(),
        },
        SearchStage {
            stage: "first_F_selection".to_string(),
            class: "budget".to_string(),
            entering: after_completability,
            leaving: selected_n,
            note: if truncated {
                format!("truncated to {hypotheses} hypotheses")
            } else {
                "all survivors selected".to_string()
            },
        },
        SearchStage {
            stage: "trajectory_allocation".to_string(),
            class: "budget".to_string(),
            entering: selected_n,
            leaving: sampled_n,
            note: allocation_note,
        },
    ];
    Ok(MassCompletionResult {
        candidates,
        distinct_before_cut,
        accounting,
        mass_evidence_status,
        error_terms,
        formula_search: FormulaSearchReport {
            status,
            truncated,
            search_exhausted,
            unsampled_reason,
            pruning: pruning.as_str().to_string(),
            allocation: allocation.as_str().to_string(),
            joined,
            joined_chemical,
            joined_chemical_reason,
            excluded_by_train_fit,
            after_domain,
            after_substructures: after_substructures,
            after_completability,
            selected: selected_n,
            sampled: sampled_n,
            ranking: ranking_text,
            stages,
            formulas,
            enumerator: enumerator_report(&enum_result),
        },
        joined_compositions: enum_result.compositions.clone(),
        beam_row_steps,
        beam_candidates_dropped,
    })
}

/// Adduct id by name (one of [`ADDUCTS`]).
pub fn adduct_id_by_name(name: &str) -> Option<u16> {
    ADDUCTS.iter().find(|a| a.name == name).map(|a| a.id)
}

/// Whether an adduct name is known.
pub fn known_adduct(name: &str) -> bool {
    adduct_id_by_name(name).is_some()
}
