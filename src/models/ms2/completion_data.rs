//! Supervised molecular-completion data: deterministic open-substructure
//! extraction plus the molecule-level dataset behind the training pairs.
//!
//! A training pair conditions on the molecule's exact composition and a few
//! *open substructures* cut from it (atom types keep the parent's hydrogen
//! counts) and predicts the molecule's canonical BFS trace. This module
//! builds those pairs deterministically from an export file and measures how
//! much of the data fits a candidate domain:
//!
//! * [`ExtractionConfig`] / [`extract_patterns`] cut connected induced
//!   subgraphs with a seeded generator. Pattern atom order is a seeded
//!   shuffle of the chosen atoms and pattern order is generation order, so
//!   neither carries information about the parent's atom order or its
//!   canonical trace. [`Pattern::parent_atoms`] is audit metadata (the oracle
//!   correspondence); it must never be a model input.
//! * [`skeleton`] drops bond orders for a Kekulé-insensitive identity. It
//!   also merges distinct bond-order isomers, so it is a relaxed identity,
//!   never the primary one.
//! * [`CompletionSet`] keeps, per molecule, the canonical trace of the
//!   molecule in canonical placement order, but only when that trace replays
//!   under the exact-completion grammar to a stopped, complete state.
//! * [`same_identity`] is exact typed, bond-order-preserving graph
//!   isomorphism: an order-independent invariant hash first, then a bounded
//!   exact search. It returns `None` when the work limit is spent, never a
//!   guess, so a generated candidate never needs a canonical trace to be
//!   compared with a target.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{Error, Result};

use super::chem::Composition;
use super::completion::stable_hash;
use super::dataset::ExportFile;
use super::functional_groups::{aromatic_rings, functional_groups};
use super::grammar::{Limits, Token, canonical_trace, replay, replay_exact};
use super::graph::MolGraph;

/// Version of the supervised completion-data recipe in this module.
pub const COMPLETION_DATA_VERSION: &str = "completion-data-v2";

/// How patterns are cut from a parent molecule.
///
/// The bounds mirror the request layer's input limits: at most 8
/// substructures and at most 24 pattern atoms per query.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExtractionConfig {
    /// Fewest patterns per parent.
    pub min_patterns: usize,
    /// Most patterns per parent (at most 8).
    pub max_patterns: usize,
    /// Fewest atoms per pattern (at least 1).
    pub min_pattern_atoms: usize,
    /// Most atoms per pattern.
    pub max_pattern_atoms: usize,
    /// Most pattern atoms in total across all patterns of one parent.
    pub max_total_atoms: usize,
}

impl Default for ExtractionConfig {
    /// One to four patterns of two to eight atoms, 24 atoms at most.
    fn default() -> Self {
        Self {
            min_patterns: 1,
            max_patterns: 4,
            min_pattern_atoms: 2,
            max_pattern_atoms: 8,
            max_total_atoms: 24,
        }
    }
}

impl ExtractionConfig {
    /// Check the bounds against the request layer's input limits:
    /// `min_patterns <= max_patterns <= 8`,
    /// `1 <= min_pattern_atoms <= max_pattern_atoms` and
    /// `max_pattern_atoms <= max_total_atoms <= 24`.
    ///
    /// `max_patterns == 0` (with `min_patterns == 0`) is valid and means
    /// "no patterns": the no-substructure control. Anything else outside the
    /// bounds is [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        if self.min_patterns > self.max_patterns || self.max_patterns > 8 {
            return Err(Error::config(format!(
                "ExtractionConfig::validate: pattern count {}..={} is not within 0..=8",
                self.min_patterns, self.max_patterns
            )));
        }
        if self.min_pattern_atoms < 1 || self.min_pattern_atoms > self.max_pattern_atoms {
            return Err(Error::config(format!(
                "ExtractionConfig::validate: pattern atoms {}..={} need 1 <= min <= max",
                self.min_pattern_atoms, self.max_pattern_atoms
            )));
        }
        if self.max_pattern_atoms > self.max_total_atoms || self.max_total_atoms > 24 {
            return Err(Error::config(format!(
                "ExtractionConfig::validate: max pattern atoms {} with total {} need max_pattern_atoms <= max_total_atoms <= 24",
                self.max_pattern_atoms, self.max_total_atoms
            )));
        }
        Ok(())
    }
}

/// How functional-group patterns are taken from a parent molecule.
///
/// All functional groups (`functional-groups-ertl-v1`) of the parent become
/// candidate patterns; the caps mirror the request layer's input limits.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FunctionalGroupConfig {
    /// Most groups kept per parent (at most 8).
    pub max_groups: usize,
    /// Most pattern atoms in total across one parent's groups (at most 24).
    pub max_total_atoms: usize,
    /// Groups larger than this are dropped (at most 24).
    pub max_group_atoms: usize,
    /// Per-group keep probability in percent (training-time robustness to
    /// incomplete lists; evaluation uses 100).
    pub keep_probability_percent: u8,
    /// Add each aromatic ring without a marked atom as its own group.
    pub aromatic_rings_as_groups: bool,
}

impl Default for FunctionalGroupConfig {
    /// Up to 8 groups, 24 atoms at most, every group kept, no ring groups.
    fn default() -> Self {
        Self {
            max_groups: 8,
            max_total_atoms: 24,
            max_group_atoms: 24,
            keep_probability_percent: 100,
            aromatic_rings_as_groups: false,
        }
    }
}

impl FunctionalGroupConfig {
    /// Check the bounds: `max_groups <= 8`,
    /// `max_group_atoms <= max_total_atoms <= 24` and
    /// `keep_probability_percent <= 100`. Anything else is [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        if self.max_groups > 8 {
            return Err(Error::config(format!(
                "FunctionalGroupConfig::validate: max_groups {} exceeds 8",
                self.max_groups
            )));
        }
        if self.max_group_atoms > self.max_total_atoms || self.max_total_atoms > 24 {
            return Err(Error::config(format!(
                "FunctionalGroupConfig::validate: max group atoms {} with total {} need max_group_atoms <= max_total_atoms <= 24",
                self.max_group_atoms, self.max_total_atoms
            )));
        }
        if self.keep_probability_percent > 100 {
            return Err(Error::config(format!(
                "FunctionalGroupConfig::validate: keep_probability_percent {} exceeds 100",
                self.keep_probability_percent
            )));
        }
        Ok(())
    }
}

/// Where the patterns of a query come from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternSource {
    /// Random connected patches ([`extract_patterns`]).
    RandomPatches(ExtractionConfig),
    /// Functional groups (Ertl) of the target.
    FunctionalGroups(FunctionalGroupConfig),
}

/// One functional-group pattern draw: the kept patterns plus audit counts.
pub struct PatternDraw {
    /// Kept patterns (induced subgraphs, shuffled atom order).
    pub patterns: Vec<Pattern>,
    /// Functional groups found on the parent (before any drop or fit).
    pub groups_found: usize,
    /// Patterns kept.
    pub groups_kept: usize,
    /// Whether the fit dropped a candidate that otherwise fit
    /// (`max_groups` / `max_total_atoms` truncation).
    pub truncated: bool,
    /// Candidates dropped for exceeding `max_group_atoms`.
    pub dropped_oversized: usize,
}

/// One open substructure cut from a parent molecule.
///
/// Atom types keep the parent's hydrogen counts: the pattern is an induced
/// subgraph of the parent, never a re-capped molecule.
pub struct Pattern {
    /// The pattern graph; pattern atom `i` is the parent's
    /// [`parent_atoms[i]`][Pattern::parent_atoms].
    pub graph: MolGraph,
    /// Parent atom behind each pattern atom, in pattern atom order.
    ///
    /// Audit metadata (the oracle correspondence): it tells which parent
    /// atoms a pattern covers and must never be a model input. Pattern atom
    /// order itself is a seeded shuffle, so it carries no information about
    /// the parent's atom order or the canonical trace.
    pub parent_atoms: Vec<usize>,
}

/// Deterministic 64-bit generator for the completion modules (SplitMix64,
/// Steele et al.).
///
/// The single shared generator of the completion modules
/// (`completion_data`, `completion_eval`, `completion_model`,
/// `completion_experiment`): extraction mixes its inputs through full rounds
/// of it, the eval bootstrap and the train shuffle draw indices with
/// [`below`][SplitMix64::below], and generation mixes seeds and request ids
/// through one round each. Fixed constants, so the same seed gives the same
/// stream on any platform.
pub(crate) struct SplitMix64 {
    /// Current state.
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next `u64`.
    pub(crate) fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer below `bound` by rejection (no modulo bias).
    ///
    /// The caller guarantees `bound > 0`; values are accepted below the
    /// largest multiple of `bound` that fits in `u64`, the rest are redrawn.
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "SplitMix64::below: empty range");
        let accept = (u64::MAX / bound) * bound;
        loop {
            let x = self.next();
            if x < accept {
                return x % bound;
            }
        }
    }
}

/// Fold `seed`, a stable hash of `key` and `draw` into one generator seed.
///
/// Each value goes through a full SplitMix64 round; raw small integers are
/// never combined directly (no `seed ^ draw`), so nearby seeds still spread.
fn mix_seed(seed: u64, key: &str, draw: u64) -> u64 {
    let mut first = SplitMix64::new(seed);
    let a = first.next();
    let mut second = SplitMix64::new(a.wrapping_add(stable_hash(&[key])));
    let b = second.next();
    let mut third = SplitMix64::new(b.wrapping_add(draw));
    third.next()
}

/// Uniform integer in `lo..=hi` by rejection (no modulo bias).
///
/// The caller guarantees `lo <= hi`.
fn uniform_range(rng: &mut SplitMix64, lo: usize, hi: usize) -> usize {
    assert!(lo <= hi, "uniform_range: empty range");
    lo + rng.below((hi - lo + 1) as u64) as usize
}

/// Cut open substructures from `parent`, deterministically in the arguments.
///
/// `seed` is the sampling seed, `key` the molecule's stable key (so two
/// molecules never share a stream) and `draw` the resampling index. The same
/// arguments always give the same patterns; different arguments spread
/// through the seed mix above.
///
/// The generator draws a pattern count in
/// `min_patterns..=max_patterns`, then per pattern a size in
/// `min_pattern_atoms..=min(max_pattern_atoms, parent atoms, remaining
/// total)` (the pattern slot is skipped when that upper bound is below
/// `min_pattern_atoms`), a uniformly random start atom, and grows the set by
/// repeatedly adding a uniformly random atom of the *sorted* frontier (atoms
/// bonded to the chosen set and not in it) until the size is reached or the
/// frontier is empty. The pattern's atom order is a seeded Fisher–Yates
/// shuffle of the chosen atoms and the graph is the parent's induced
/// subgraph on that order. Patterns may overlap each other; the overlap is
/// unknown to the model.
///
/// Every returned pattern is connected, holds between 1 and
/// `max_pattern_atoms` atoms, the total stays within `max_total_atoms`, and
/// atom types are the parent's.
pub fn extract_patterns(
    parent: &MolGraph,
    config: &ExtractionConfig,
    seed: u64,
    key: &str,
    draw: u64,
) -> Result<Vec<Pattern>> {
    config.validate()?;
    let mut rng = SplitMix64::new(mix_seed(seed, key, draw));
    let count = uniform_range(&mut rng, config.min_patterns, config.max_patterns);
    let n_atoms = parent.atoms().len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
    for (a, b, _) in parent.bonds() {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    let mut out = Vec::with_capacity(count);
    let mut total = 0usize;
    for _ in 0..count {
        let remaining = config.max_total_atoms.saturating_sub(total);
        let hi = config.max_pattern_atoms.min(n_atoms).min(remaining);
        if hi < config.min_pattern_atoms {
            continue;
        }
        let size = uniform_range(&mut rng, config.min_pattern_atoms, hi);
        let start = rng.below(n_atoms as u64) as usize;
        let mut chosen = vec![false; n_atoms];
        chosen[start] = true;
        let mut held = 1usize;
        while held < size {
            let mut frontier: Vec<usize> = Vec::new();
            for (atom, &inside) in chosen.iter().enumerate() {
                if inside {
                    continue;
                }
                if adj[atom].iter().any(|&nbr| chosen[nbr]) {
                    frontier.push(atom);
                }
            }
            if frontier.is_empty() {
                break;
            }
            let pick = frontier[rng.below(frontier.len() as u64) as usize];
            chosen[pick] = true;
            held += 1;
        }
        let mut order: Vec<usize> = chosen
            .iter()
            .enumerate()
            .filter(|(_, inside)| **inside)
            .map(|(atom, _)| atom)
            .collect();
        for i in (1..order.len()).rev() {
            let j = rng.below((i + 1) as u64) as usize;
            order.swap(i, j);
        }
        let graph = parent
            .induced(&order)
            .map_err(|e| Error::config(format!("extract_patterns: induced pattern failed: {e}")))?;
        total += order.len();
        out.push(Pattern {
            graph,
            parent_atoms: order,
        });
    }
    Ok(out)
}

/// Take the functional groups of `parent` as patterns, deterministically in
/// the arguments.
///
/// All functional groups (`functional-groups-ertl-v1`) of the parent are
/// candidates as induced subgraphs with the parent's atom types. With
/// `aromatic_rings_as_groups`, each aromatic ring (per
/// [`AROMATICITY_VERSION`](super::functional_groups::AROMATICITY_VERSION))
/// without a marked atom is added as its own group. Groups larger than
/// `max_group_atoms` are dropped and counted in
/// [`dropped_oversized`][PatternDraw::dropped_oversized]. With
/// `keep_probability_percent < 100` each remaining group is kept
/// independently with that probability (seeded) before the fit. When the
/// survivors exceed `max_groups` or `max_total_atoms`, a seeded subset that
/// fits is kept: the survivors in a seeded Fisher–Yates order, taken while
/// each fits, and [`truncated`][PatternDraw::truncated] is set.
///
/// Atom order inside each pattern and the order of the patterns are seeded
/// Fisher–Yates shuffles from the same `SplitMix64` stream mixed from
/// `seed`/`key`/`draw` as [`extract_patterns`], so no parent order leaks.
/// Molecules without any functional group give an empty pattern set. The
/// version behind the groups is [`FUNCTIONAL_GROUPS_VERSION`].
pub fn functional_group_patterns(
    parent: &MolGraph,
    config: &FunctionalGroupConfig,
    seed: u64,
    key: &str,
    draw: u64,
) -> Result<PatternDraw> {
    config.validate()?;
    let mut rng = SplitMix64::new(mix_seed(seed, key, draw));
    let groups = functional_groups(parent).map_err(|e| {
        Error::config(format!("functional_group_patterns: functional groups failed: {e}"))
    })?;
    let marked_union: std::collections::HashSet<usize> =
        groups.iter().flat_map(|g| g.atoms.iter().copied()).collect();
    let mut atom_lists: Vec<Vec<usize>> = groups.into_iter().map(|g| g.atoms).collect();
    if config.aromatic_rings_as_groups {
        for ring in aromatic_rings(parent) {
            if ring.iter().any(|a| marked_union.contains(a)) {
                continue;
            }
            let mut atoms = ring;
            atoms.sort_unstable();
            atoms.dedup();
            if !atom_lists.contains(&atoms) {
                atom_lists.push(atoms);
            }
        }
        atom_lists.sort_by_key(|atoms| atoms[0]);
    }
    let groups_found = atom_lists.len();
    // Drop oversized groups.
    let mut dropped_oversized = 0usize;
    let mut candidates: Vec<Vec<usize>> = Vec::new();
    for atoms in atom_lists {
        if atoms.len() > config.max_group_atoms {
            dropped_oversized += 1;
            continue;
        }
        candidates.push(atoms);
    }
    // Seeded keep-probability filter before the fit.
    let mut survivors: Vec<Vec<usize>> = Vec::new();
    if config.keep_probability_percent >= 100 {
        survivors = candidates;
    } else if config.keep_probability_percent == 0 {
        // Still draw to keep the stream形状? No draws needed: nothing kept.
        survivors = Vec::new();
    } else {
        for atoms in candidates {
            let roll = rng.below(100) as u8;
            if roll < config.keep_probability_percent {
                survivors.push(atoms);
            }
        }
    }
    // Seeded Fisher–Yates order, then take groups while each fits.
    for i in (1..survivors.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        survivors.swap(i, j);
    }
    let mut kept_lists: Vec<Vec<usize>> = Vec::new();
    let mut total = 0usize;
    let mut truncated = false;
    for atoms in survivors {
        if kept_lists.len() >= config.max_groups || total + atoms.len() > config.max_total_atoms {
            truncated = true;
            continue;
        }
        // A later smaller group may still fit: skip this one but keep going.
        // `truncated` records that the full survivor list did not fit; when
        // every skipped group is oversized for the remaining budget this is
        // still a truncation of the seeded order.
        total += atoms.len();
        kept_lists.push(atoms);
    }
    let groups_kept = kept_lists.len();
    let mut patterns = Vec::with_capacity(kept_lists.len());
    for mut atoms in kept_lists {
        for i in (1..atoms.len()).rev() {
            let j = rng.below((i + 1) as u64) as usize;
            atoms.swap(i, j);
        }
        let graph = parent.induced(&atoms).map_err(|e| {
            Error::config(format!(
                "functional_group_patterns: induced pattern failed: {e}"
            ))
        })?;
        patterns.push(Pattern {
            graph,
            parent_atoms: atoms,
        });
    }
    Ok(PatternDraw {
        patterns,
        groups_found,
        groups_kept,
        truncated,
        dropped_oversized,
    })
}

/// The skeleton of a graph: the same atoms and bonds with every bond order
/// set to 1.
///
/// Used for a Kekulé-insensitive identity: two Kekulé forms of one aromatic
/// system share a skeleton. It also merges distinct bond-order isomers, so it
/// is a relaxed identity, never the primary one.
pub fn skeleton(graph: &MolGraph) -> Result<MolGraph> {
    let bonds: Vec<(usize, usize, u8)> =
        graph.bonds().iter().map(|(a, b, _)| (*a, *b, 1)).collect();
    MolGraph::new(graph.atoms().to_vec(), bonds)
        .map_err(|e| Error::config(format!("skeleton: rebuilding with unit bonds failed: {e}")))
}

/// One supervised completion example: a molecule with its canonical trace.
pub struct CompletionExample {
    /// Stable molecule key from the export file.
    pub key: String,
    /// Identity group for the frozen splits, from the export file.
    pub identity_group: u64,
    /// Position in the export file's molecule list (`ExportFile.molecules`
    /// index): the fingerprint sidecar's `bits_by_molecule` lookup key.
    /// Keyed (`<key>|<identity_group>`) lookup misattributes fingerprints
    /// when several structures share one key, so the index is authoritative.
    pub source_index: usize,
    /// The molecule's atoms in canonical placement order (the graph behind
    /// [`trace`][CompletionExample::trace]).
    pub target: MolGraph,
    /// Exact composition of the target, hydrogens included.
    pub composition: Composition,
    /// Canonical trace of the molecule: START .. STOP.
    pub trace: Vec<Token>,
    /// Canonical trace of [`skeleton`] of the target; empty when that
    /// canonicalization hit the work limit.
    pub skeleton_trace: Vec<Token>,
}

impl CompletionExample {
    /// Cut open substructures from this example's target for one draw.
    ///
    /// Shorthand for [`extract_patterns`] with the example's key, so draws
    /// for different molecules never share a stream.
    pub fn patterns(
        &self,
        config: &ExtractionConfig,
        seed: u64,
        draw: u64,
    ) -> Result<Vec<Pattern>> {
        extract_patterns(&self.target, config, seed, &self.key, draw)
    }

    /// Cut patterns from this example's target through a [`PatternSource`].
    ///
    /// Random patches return every drawn patch with `groups_found` equal to
    /// the pattern count and no truncation; functional groups return the
    /// full [`PatternDraw`].
    pub fn patterns_from(
        &self,
        source: &PatternSource,
        seed: u64,
        draw: u64,
    ) -> Result<PatternDraw> {
        match source {
            PatternSource::RandomPatches(config) => {
                let patterns = extract_patterns(&self.target, config, seed, &self.key, draw)?;
                let kept = patterns.len();
                Ok(PatternDraw {
                    patterns,
                    groups_found: kept,
                    groups_kept: kept,
                    truncated: false,
                    dropped_oversized: 0,
                })
            }
            PatternSource::FunctionalGroups(config) => {
                functional_group_patterns(&self.target, config, seed, &self.key, draw)
            }
        }
    }
}

/// The kept canonical examples of an export file under one domain.
pub struct CompletionSet {
    /// Trace size limits the set was built with.
    pub limits: Limits,
    /// Kept examples, in export file order.
    pub examples: Vec<CompletionExample>,
    /// Molecules left out, by reason (`graph_error`, `not_connected`,
    /// `too_many_atoms`, `too_many_closures`, `canonicalization_work_limit`,
    /// `duplicate_identity`).
    pub skipped: BTreeMap<String, u64>,
    /// Expansion budget passed to [`canonical_trace`].
    pub max_expansions: usize,
}

impl CompletionSet {
    /// Build the set from an export file: canonicalize every molecule in file
    /// order, keeping only complete canonical examples.
    ///
    /// The first applicable reason leaves a molecule out and counts it in
    /// [`skipped`][CompletionSet::skipped]: `graph_error`
    /// ([`MolGraph::new`] failed), `not_connected`, `too_many_atoms`,
    /// `too_many_closures`, `canonicalization_work_limit`, then
    /// `duplicate_identity` (an earlier kept example has the same canonical
    /// trace; the first is kept). The work limit is [`canonical_trace`]'s
    /// third argument; hitting it is a counted skip, never an error.
    ///
    /// Every kept example is checked, and any failure is an [`Error`]
    /// naming the molecule key, never a silent skip:
    /// [`replay_exact`][crate::models::ms2::grammar::replay_exact] of the
    /// trace under `limits` with the stated composition succeeds, the end
    /// state is stopped and complete (a legal prefix is not enough), and the
    /// replayed graph is connected with a composition equal to the stated
    /// one.
    pub fn from_export(file: &ExportFile, limits: Limits, work_limit: usize) -> Result<Self> {
        let mut examples = Vec::new();
        let mut skipped: BTreeMap<String, u64> = BTreeMap::new();
        let mut kept_traces: Vec<Vec<Token>> = Vec::new();
        for (source_index, mol) in file.molecules.iter().enumerate() {
            let graph = match mol.graph() {
                Ok(graph) => graph,
                Err(_) => {
                    *skipped.entry("graph_error".to_string()).or_default() += 1;
                    continue;
                }
            };
            if !graph.is_connected() {
                *skipped.entry("not_connected".to_string()).or_default() += 1;
                continue;
            }
            if graph.atoms().len() > limits.max_atoms() {
                *skipped.entry("too_many_atoms".to_string()).or_default() += 1;
                continue;
            }
            if graph.ring_closures() > limits.max_closures() {
                *skipped.entry("too_many_closures".to_string()).or_default() += 1;
                continue;
            }
            let canonical = match canonical_trace(&graph, limits, work_limit) {
                Ok(canonical) => canonical,
                Err(e) => {
                    if e.to_string().contains("canonicalization_budget_exceeded") {
                        *skipped
                            .entry("canonicalization_work_limit".to_string())
                            .or_default() += 1;
                        continue;
                    }
                    return Err(Error::config(format!(
                        "CompletionSet::from_export: molecule {} canonicalization failed: {e}",
                        mol.key
                    )));
                }
            };
            if kept_traces.contains(&canonical.trace) {
                *skipped.entry("duplicate_identity".to_string()).or_default() += 1;
                continue;
            }
            let key_error = |what: &str| {
                Error::config(format!(
                    "CompletionSet::from_export: molecule {key} {what}",
                    key = mol.key
                ))
            };
            let replayed = replay(&canonical.trace, limits, None)
                .map_err(|e| key_error(&format!("canonical trace does not replay: {e}")))?;
            let target = replayed
                .graph()
                .map_err(|e| key_error(&format!("replayed graph is invalid: {e}")))?;
            let composition = target.composition();
            let exact = replay_exact(&canonical.trace, limits, composition)
                .map_err(|e| key_error(&format!("canonical trace is not exact-legal: {e}")))?;
            if !exact.stopped() || !exact.is_complete() {
                return Err(key_error(
                    "canonical trace stops short of a complete molecule",
                ));
            }
            let end = exact
                .graph()
                .map_err(|e| key_error(&format!("exact end graph is invalid: {e}")))?;
            if !end.is_connected() {
                return Err(key_error("exact end graph is disconnected"));
            }
            if end.composition() != composition {
                return Err(key_error("exact end graph composition differs"));
            }
            let flat = skeleton(&target)
                .map_err(|e| key_error(&format!("skeleton rebuild failed: {e}")))?;
            let skeleton_trace = match canonical_trace(&flat, limits, work_limit) {
                Ok(canonical) => canonical.trace,
                Err(e) => {
                    if e.to_string().contains("canonicalization_budget_exceeded") {
                        Vec::new()
                    } else {
                        return Err(key_error(&format!("skeleton canonicalization failed: {e}")));
                    }
                }
            };
            kept_traces.push(canonical.trace.clone());
            examples.push(CompletionExample {
                key: mol.key.clone(),
                identity_group: mol.identity_group,
                source_index,
                target,
                composition,
                trace: canonical.trace,
                skeleton_trace,
            });
        }
        Ok(Self {
            limits,
            examples,
            skipped,
            max_expansions: work_limit,
        })
    }

    /// Read an export file from disk and build the set.
    pub fn load(path: &Path, limits: Limits, work_limit: usize) -> Result<Self> {
        Self::from_export(&ExportFile::load(path)?, limits, work_limit)
    }

    /// Trace overlap of this set with another: how many of this set's
    /// examples have a canonical trace equal to one of `other`'s, and how
    /// many have an equal non-empty skeleton trace.
    pub fn overlap(&self, other: &CompletionSet) -> (usize, usize) {
        let strict = self
            .examples
            .iter()
            .filter(|e| other.examples.iter().any(|o| o.trace == e.trace))
            .count();
        let relaxed = self
            .examples
            .iter()
            .filter(|e| {
                !e.skeleton_trace.is_empty()
                    && other
                        .examples
                        .iter()
                        .any(|o| o.skeleton_trace == e.skeleton_trace)
            })
            .count();
        (strict, relaxed)
    }
}

/// Exact typed, bond-order-preserving graph isomorphism.
///
/// The policy mirrors the host side of `identity.rs`: an order-independent
/// invariant hash first (differing hashes decide inequality), then a bounded
/// exact backtracking search over a connectivity order of `a`, pruning by
/// atom type and by exact bond agreement (presence and order) with every
/// already-mapped atom. Each tried image costs one unit of `work_limit`;
/// spending it returns `None`, never a guess. That is what lets generated
/// candidates be compared with targets without a canonical trace.
///
/// The lanes of `identity.rs` work on packed device records rather than
/// [`MolGraph`], so they cannot run here directly; this is the same two-stage
/// policy adapted to the host graph.
pub fn same_identity(a: &MolGraph, b: &MolGraph, work_limit: usize) -> Option<bool> {
    if a.atoms().len() != b.atoms().len() || a.bonds().len() != b.bonds().len() {
        return Some(false);
    }
    let mut need: BTreeMap<u8, usize> = BTreeMap::new();
    let mut have: BTreeMap<u8, usize> = BTreeMap::new();
    for t in a.atoms() {
        *need.entry(*t).or_default() += 1;
    }
    for t in b.atoms() {
        *have.entry(*t).or_default() += 1;
    }
    if need != have {
        return Some(false);
    }
    if invariant_hash(a) != invariant_hash(b) {
        return Some(false);
    }
    exact_isomorphism(a, b, work_limit)
}

/// Order-independent invariant hash: per-atom labels refined over bond
/// neighbourhoods, summed commutatively.
///
/// Four refinement rounds mix each atom's label with the wrapped sum of its
/// neighbours' `(bond order, label)` pairs, exactly like the lane's
/// refinement sum; the total folds in the atom and bond counts. Equal up to
/// graph isomorphism by construction (summation order never matters), so a
/// hash difference proves non-isomorphism while equality proves nothing.
fn invariant_hash(graph: &MolGraph) -> u64 {
    fn mix(first: u64, second: u64) -> u64 {
        let mut z = first
            .wrapping_add(second)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let n = graph.atoms().len();
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for (a, b, order) in graph.bonds() {
        adj[*a].push((*b, *order));
        adj[*b].push((*a, *order));
    }
    let mut labels: Vec<u64> = graph
        .atoms()
        .iter()
        .enumerate()
        .map(|(i, &t)| mix(u64::from(t), adj[i].len() as u64))
        .collect();
    for _ in 0..4 {
        let mut next = vec![0u64; n];
        for (i, label) in labels.iter().enumerate().take(n) {
            let mut acc = 0u64;
            for &(nbr, order) in &adj[i] {
                acc = acc.wrapping_add(mix(u64::from(order), labels[nbr]));
            }
            next[i] = mix(*label, acc);
        }
        labels = next;
    }
    let mut sum = 0u64;
    for label in &labels {
        sum = sum.wrapping_add(*label);
    }
    mix(sum, mix(n as u64, graph.bonds().len() as u64))
}

/// Connectivity order of `a`: each next atom adjacent to an earlier one where
/// the graph allows, restarting at the smallest unmapped atom. Shared by the
/// first-match search and the enumerating search so both visit atoms in the
/// same order.
fn connectivity_order(a_adj: &[Vec<u8>], na: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(na);
    let mut visited = vec![false; na];
    while order.len() < na {
        let Some(start) = (0..na).find(|&i| !visited[i]) else {
            break;
        };
        visited[start] = true;
        order.push(start);
        loop {
            let mut next = None;
            for c in 0..na {
                if visited[c] {
                    continue;
                }
                if (0..na).any(|d| visited[d] && a_adj[c][d] != 0) {
                    next = Some(c);
                    break;
                }
            }
            match next {
                Some(v) => {
                    visited[v] = true;
                    order.push(v);
                }
                None => break,
            }
        }
    }
    order
}
/// Bounded exact isomorphism search behind [`same_identity`].
///
/// Connectivity order of `a` (see [`connectivity_order`]), mapped
/// injectively into `b` with exact agreement of atom type and of every bond
/// (presence and order) to already-mapped atoms. Equal atom and bond counts
/// plus agreement on every `a`-bond already force full induced agreement, so
/// the first complete map decides equality. `None` on work exhaustion.
fn exact_isomorphism(a: &MolGraph, b: &MolGraph, work_limit: usize) -> Option<bool> {
    let na = a.atoms().len();
    let nb = b.atoms().len();
    if na != nb {
        return Some(false);
    }
    let mut a_adj = vec![vec![0u8; na]; na];
    for (x, y, o) in a.bonds() {
        a_adj[*x][*y] = *o;
        a_adj[*y][*x] = *o;
    }
    let mut b_adj = vec![vec![0u8; nb]; nb];
    for (x, y, o) in b.bonds() {
        b_adj[*x][*y] = *o;
        b_adj[*y][*x] = *o;
    }
    let order = connectivity_order(&a_adj, na);
    let mut map: Vec<Option<usize>> = vec![None; na];
    let mut used = vec![false; nb];
    let mut attempts = 0usize;
    let mut over_budget = false;
    let found = iso_dfs(
        0,
        &order,
        a.atoms(),
        b.atoms(),
        &a_adj,
        &b_adj,
        &mut map,
        &mut used,
        &mut attempts,
        work_limit,
        &mut over_budget,
    );
    if over_budget { None } else { Some(found) }
}

/// Depth-first backtracking over the connectivity order.
///
/// `map[d]` is the `b` image of `order[d]` for `d < depth`.
#[allow(clippy::too_many_arguments)]
fn iso_dfs(
    depth: usize,
    order: &[usize],
    a_atoms: &[u8],
    b_atoms: &[u8],
    a_adj: &[Vec<u8>],
    b_adj: &[Vec<u8>],
    map: &mut [Option<usize>],
    used: &mut [bool],
    attempts: &mut usize,
    work_limit: usize,
    over_budget: &mut bool,
) -> bool {
    if *over_budget {
        return false;
    }
    if depth == order.len() {
        return true;
    }
    let c = order[depth];
    for image in 0..b_atoms.len() {
        if used[image] || b_atoms[image] != a_atoms[c] {
            continue;
        }
        let mut ok = true;
        for d in 0..depth {
            let c2 = order[d];
            let image2 = map[d].expect("earlier depths are mapped");
            if a_adj[c][c2] != b_adj[image][image2] {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        *attempts += 1;
        if *attempts > work_limit {
            *over_budget = true;
            return false;
        }
        map[depth] = Some(image);
        used[image] = true;
        if iso_dfs(
            depth + 1,
            order,
            a_atoms,
            b_atoms,
            a_adj,
            b_adj,
            map,
            used,
            attempts,
            work_limit,
            over_budget,
        ) {
            return true;
        }
        map[depth] = None;
        used[image] = false;
        if *over_budget {
            return false;
        }
    }
    false
}

/// Every complete typed, bond-order-preserving isomorphism of `a` onto `b`.
///
/// The same bounded backtracking search as [`exact_isomorphism`] (same
/// connectivity order, same pruning, each tried image costing one unit of
/// `work_limit`), but visiting every complete mapping instead of stopping at
/// the first. Collection stops early with `truncated` once `max_mappings`
/// maps are held; spending `work_limit` first sets `over_budget`. Either
/// flag means the enumeration is incomplete, never a guess. When both flags
/// are false, `maps` holds every isomorphism, so an empty `maps` proves
/// non-isomorphism.
///
/// `maps[i][j]` is the `b` image of `a` atom `j`. The visit order is
/// deterministic (ascending image trials over [`connectivity_order`]), so
/// the identity — when it is an isomorphism — is always `maps[0]`.
///
/// [`same_identity`] keeps its own first-match search and behaviour; this
/// enumerator exists for symmetry analysis (stereo orbits), where stopping
/// at the first map would be wrong.
pub fn enumerate_isomorphisms(
    a: &MolGraph,
    b: &MolGraph,
    work_limit: usize,
    max_mappings: usize,
) -> IsomorphismEnumeration {
    let na = a.atoms().len();
    let nb = b.atoms().len();
    if na != nb {
        return IsomorphismEnumeration {
            maps: Vec::new(),
            over_budget: false,
            truncated: false,
        };
    }
    let mut a_adj = vec![vec![0u8; na]; na];
    for (x, y, o) in a.bonds() {
        a_adj[*x][*y] = *o;
        a_adj[*y][*x] = *o;
    }
    let mut b_adj = vec![vec![0u8; nb]; nb];
    for (x, y, o) in b.bonds() {
        b_adj[*x][*y] = *o;
        b_adj[*y][*x] = *o;
    }
    let order = connectivity_order(&a_adj, na);
    let mut out = IsomorphismEnumeration {
        maps: Vec::new(),
        over_budget: false,
        truncated: false,
    };
    let mut map: Vec<Option<usize>> = vec![None; na];
    let mut used = vec![false; nb];
    let mut attempts = 0usize;
    iso_dfs_all(
        0,
        &order,
        a.atoms(),
        b.atoms(),
        &a_adj,
        &b_adj,
        &mut map,
        &mut used,
        &mut attempts,
        work_limit,
        &mut out,
        max_mappings,
    );
    out
}

/// Outcome of [`enumerate_isomorphisms`].
pub struct IsomorphismEnumeration {
    /// Complete maps `a -> b`: `maps[i][j]` is the `b` image of `a` atom `j`,
    /// in deterministic visit order.
    pub maps: Vec<Vec<usize>>,
    /// True when the work limit was spent before the search completed: `maps`
    /// is a prefix of the visit order, never the full set.
    pub over_budget: bool,
    /// True when the search stopped early at the mapping cap: at least the
    /// cap many isomorphisms exist.
    pub truncated: bool,
}

/// Depth-first backtracking over the connectivity order, collecting every
/// complete mapping.
///
/// `map[d]` is the `b` image of `order[d]` for `d < depth`. A complete
/// mapping is translated to atom order and pushed, and the search continues
/// (it never stops at a first match). It stops — setting the matching flag
/// on `out` — once `max_mappings` maps are held or `work_limit` attempts are
/// spent. Attempts are counted exactly like [`iso_dfs`], so the two searches
/// agree on what fits a budget.
#[allow(clippy::too_many_arguments)]
fn iso_dfs_all(
    depth: usize,
    order: &[usize],
    a_atoms: &[u8],
    b_atoms: &[u8],
    a_adj: &[Vec<u8>],
    b_adj: &[Vec<u8>],
    map: &mut [Option<usize>],
    used: &mut [bool],
    attempts: &mut usize,
    work_limit: usize,
    out: &mut IsomorphismEnumeration,
    max_mappings: usize,
) {
    if out.over_budget || out.truncated {
        return;
    }
    if depth == order.len() {
        let mut full = vec![0usize; order.len()];
        for (d, c) in order.iter().enumerate() {
            full[*c] = map[d].expect("earlier depths are mapped");
        }
        out.maps.push(full);
        if out.maps.len() >= max_mappings {
            out.truncated = true;
        }
        return;
    }
    let c = order[depth];
    for image in 0..b_atoms.len() {
        if used[image] || b_atoms[image] != a_atoms[c] {
            continue;
        }
        let mut ok = true;
        for d in 0..depth {
            let c2 = order[d];
            let image2 = map[d].expect("earlier depths are mapped");
            if a_adj[c][c2] != b_adj[image][image2] {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        *attempts += 1;
        if *attempts > work_limit {
            out.over_budget = true;
            return;
        }
        map[depth] = Some(image);
        used[image] = true;
        iso_dfs_all(
            depth + 1,
            order,
            a_atoms,
            b_atoms,
            a_adj,
            b_adj,
            map,
            used,
            attempts,
            work_limit,
            out,
            max_mappings,
        );
        map[depth] = None;
        used[image] = false;
        if out.over_budget || out.truncated {
            return;
        }
    }
}
