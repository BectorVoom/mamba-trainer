//! Bounded molecular-completion ambiguity audit
//! (`docs/MOLECULAR_COMPLETION_EXPERIMENT.md`).
//!
//! Given an observed neutral mass with explicit precision metadata, a declared
//! small chemical domain, and required open substructures (with unknown
//! cross-substructure overlap inferred by the typed injective embeddings, or
//! pinned by a globally consistent correspondence oracle), enumerate the
//! distinct complete closed-shell molecular graphs that agree with every
//! constraint. Graph identity is the canonical BFS trace; mass arithmetic,
//! tolerance and rounding bounds reuse `chem` exactly. Every budget is a hard
//! deterministic counter, a wall-clock watchdog is separate, and a memory
//! estimate aborts with its own reason. No incomplete or unresolved query
//! ever certifies zero compatible graphs.
//!
//! The enumerator only extends the existing grammar: [`TraceState`] is never
//! seeded with a partial substructure, and an existing caller's STOP semantics
//! (`TraceState` used for open subgraph tasks) is untouched.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

use super::chem::{self, Composition};
use super::contain;
use super::grammar::{self, Limits, Token, TraceState};
use super::graph::MolGraph;

/// Version of this bounded completion protocol.
pub const COMPLETION_VERSION: &str = "completion-bounded-v1";

/// FNV-1a 64-bit: a stable, dependency-free hash for provenance.
pub fn stable_hash(parts: &[&str]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for part in parts {
        for b in part.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The declared small chemical domain of one query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionDomain {
    elements: Vec<usize>,
    min_heavy: usize,
    max_heavy: usize,
    max_ring_closures: usize,
}

impl CompletionDomain {
    /// Validate and store a declared domain: a subset of the supported
    /// C/N/O elements with a bounded connected-graph size.
    pub fn new(
        elements: &[usize],
        min_heavy: usize,
        max_heavy: usize,
        max_ring_closures: usize,
    ) -> Result<Self> {
        let supported = [0usize, 2, 3]; // C, N, O
        if elements.is_empty() || elements.iter().any(|e| !supported.contains(e)) {
            return Err(Error::Unsupported(format!(
                "unsupported_input: domain elements must be a non-empty subset of C/N/O, got {elements:?}"
            )));
        }
        let mut seen = elements.to_vec();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != elements.len() {
            return Err(Error::Unsupported(
                "unsupported_input: duplicate domain elements".to_string(),
            ));
        }
        if min_heavy == 0 || min_heavy > max_heavy || max_heavy > 32 {
            return Err(Error::Unsupported(format!(
                "unsupported_input: heavy-atom bounds {min_heavy}..={max_heavy} invalid"
            )));
        }
        if max_ring_closures > 8 {
            return Err(Error::Unsupported(
                "unsupported_input: max_ring_closures above the declared bound".to_string(),
            ));
        }
        Ok(Self {
            elements: seen,
            min_heavy,
            max_heavy,
            max_ring_closures,
        })
    }

    /// Allowed heavy elements as `ELEMENTS` indices.
    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    /// Inclusive heavy-atom bounds.
    pub fn heavy_range(&self) -> (usize, usize) {
        (self.min_heavy, self.max_heavy)
    }

    /// Maximum independent cycles (ring closures) per target.
    pub fn max_ring_closures(&self) -> usize {
        self.max_ring_closures
    }

    /// Whether the atom type is inside this domain.
    pub fn admits_type(&self, id: u8) -> bool {
        let Some(t) = chem::atom_type(id) else {
            return false;
        };
        self.elements.contains(&t.element)
    }

    /// Stable text form of the domain, used for hashing and reports.
    pub fn describe(&self) -> String {
        let symbols: Vec<&str> = self
            .elements
            .iter()
            .map(|e| chem::ELEMENTS[*e].symbol)
            .collect();
        format!(
            "elements={},heavy={}..={},max_ring_closures={}",
            symbols.join("/"),
            self.min_heavy,
            self.max_heavy,
            self.max_ring_closures
        )
    }
}

/// Per-stage work budgets plus the wall-clock watchdog and memory bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionBudgets {
    /// Count-vector visits of the formula enumeration.
    pub formula_visits: usize,
    /// Applied grammar tokens (graph action extensions).
    pub graph_extensions: usize,
    /// Embedding-match extension candidates.
    pub embedding_nodes: usize,
    /// Canonicalization branch-and-bound expansions, cumulative.
    pub canonical_expansions: usize,
    /// Retained unique accepted graph identities.
    pub retained_graphs: usize,
    /// Estimated bytes retained as unique graphs, tokens and maps.
    pub memory_bytes: usize,
    /// Wall-clock watchdog; deterministic counters stay primary.
    pub watchdog: Duration,
}

impl Default for CompletionBudgets {
    fn default() -> Self {
        Self {
            formula_visits: 100_000,
            graph_extensions: 100_000,
            embedding_nodes: 100_000,
            canonical_expansions: 100_000,
            retained_graphs: 10_000,
            memory_bytes: 64 * 1024 * 1024,
            watchdog: Duration::from_secs(30),
        }
    }
}

/// Neutral-mass precision of the observed value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MassEvidence {
    /// Tolerance at the observed m/z, in tenths of a ppm.
    pub ppm_tenths: u32,
    /// Absolute observation uncertainty in integer micro-dalton, or `None`
    /// when precision is unavailable (then no mass decision is possible).
    pub uncertainty_uda: Option<u32>,
    /// `synthetic` or `instrument`; provenance only.
    pub source: String,
}

/// One globally applied atom-correspondence constraint:
/// `(s1, a1)` must land on the same target atom as `(s2, a2)`.
///
/// Semantics are "fully known": the pairs induce equivalence classes over
/// pattern atom occurrences, every class must map to one target atom, and
/// distinct classes must map to distinct target atoms (classes are matched
/// injectively into the target). An empty pair list is therefore *known
/// disjoint* — cross-substructure embeddings may not share target atoms.
/// Use `None` correspondence for unknown overlap, which allows arbitrary
/// sharing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Correspondence {
    /// Pairs of `(substructure index, atom index)`.
    pub pairs: Vec<((usize, usize), (usize, usize))>,
}

impl Correspondence {
    /// Consistency check: every referenced pair must exist and name equal
    /// typed atoms; equal atoms form classes across all pairs; each class
    /// must be injective within each substructure (two distinct atoms of one
    /// substructure cannot share a target atom).
    pub fn validate(&self, substructures: &[MolGraph]) -> Result<()> {
        let mut sizes = vec![0usize; substructures.len()];
        for (i, s) in substructures.iter().enumerate() {
            sizes[i] = s.atoms().len();
        }
        let offset = |s: usize, a: usize| -> usize { sizes[..s].iter().sum::<usize>() + a };
        let total: usize = sizes.iter().sum();
        let mut parent: Vec<usize> = (0..total).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for &(a, b) in &self.pairs {
            let (s1, a1) = a;
            let (s2, a2) = b;
            let g1 = substructures.get(s1).ok_or_else(|| {
                Error::Unsupported(format!(
                    "unsupported_input: correspondence references substructure {s1}"
                ))
            })?;
            let g2 = substructures.get(s2).ok_or_else(|| {
                Error::Unsupported(format!(
                    "unsupported_input: correspondence references substructure {s2}"
                ))
            })?;
            let t1 = g1.atoms().get(a1).copied();
            let t2 = g2.atoms().get(a2).copied();
            let (Some(t1), Some(t2)) = (t1, t2) else {
                return Err(Error::Unsupported(format!(
                    "unsupported_input: correspondence references atom ({a:?}) or ({b:?}) out of range"
                )));
            };
            if t1 != t2 {
                return Err(Error::Unsupported(
                    "unsupported_input: correspondence pairs incompatible shared atom types"
                        .to_string(),
                ));
            }
            let ra = find(&mut parent, offset(s1, a1));
            let rb = find(&mut parent, offset(s2, a2));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let mut by_root: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (s, size) in sizes.iter().enumerate() {
            for a in 0..*size {
                let r = find(&mut parent, offset(s, a));
                by_root.entry(r).or_default().push((s, a));
            }
        }
        for members in by_root.values() {
            let mut per_sub: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
            for (s, a) in members {
                per_sub.entry(*s).or_default().insert(*a);
            }
            for atoms in per_sub.values() {
                if atoms.len() > 1 {
                    return Err(Error::Unsupported(
                        "unsupported_input: correspondence forces two atoms of one substructure onto one target atom"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// One completion query: identity, mass, evidence, and required substructures.
pub struct CompletionQuery {
    /// Stable query id.
    pub id: String,
    /// `synthetic` or `instrument`, plus a free-text note.
    pub provenance: String,
    /// Observed neutral target mass in integer micro-dalton.
    pub observed_mass_uda: u32,
    /// Mass decision metadata.
    pub mass: MassEvidence,
    /// Required open substructures in input order.
    pub substructures: Vec<MolGraph>,
    /// Optional globally-applied typed correspondence. `Some` is a fully
    /// known equivalence relation (distinct classes map to distinct target
    /// atoms; `Some` with no pairs means known-disjoint). `None` leaves
    /// overlap unknown and permits arbitrary sharing.
    pub correspondence: Option<Correspondence>,
    /// The target domain.
    pub domain: CompletionDomain,
    /// Optional fixture reference graph for ground-truth recovery.
    pub reference: Option<MolGraph>,
}

/// Work counters, frozen at termination.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Counters {
    /// Count-vector visits during formula enumeration.
    pub formula_visits: usize,
    /// Token applications during the trace search.
    pub graph_extensions: usize,
    /// Embedding-match candidates tried.
    pub embedding_nodes: usize,
    /// Canonical branch-and-bound expansions.
    pub canonical_expansions: usize,
    /// Retained unique graph identities.
    pub retained_graphs: usize,
    /// Estimated retained bytes of the unique identity set: retained graph types,
    /// bonds and canonical traces, plus modeled DFS-path frontier (sum of each
    /// in-flight trace's backing store) at its last checked point. Modeled
    /// storage only; peak RSS is not sampled on this platform path.
    pub memory_estimate_bytes: usize,
    /// Completed graphs passing the closed-molecule filters.
    pub completed_candidates: usize,
}

/// Three-valued mass status of the mass constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassStatus {
    /// Every visited formula got a definitive accept/reject and at least one
    /// was accepted; accepted ones are the only candidates.
    Accepted,
    /// At least one formula sat on the tolerance boundary band.
    Ambiguous,
    /// Observation precision is unavailable; no mass decision is possible.
    Unavailable,
    /// Resolved but every candidate formula is rejected; the observation is
    /// mass-inconsistent with the entire restricted domain, yet that verdict
    /// is itemized just like any other decision.
    Rejected,
}

impl MassStatus {
    /// Stable name used in the report.
    pub fn name(&self) -> &'static str {
        match self {
            MassStatus::Accepted => "accepted",
            MassStatus::Ambiguous => "ambiguous",
            MassStatus::Unavailable => "unavailable",
            MassStatus::Rejected => "rejected",
        }
    }
}

/// The precursor arm is explicitly not evaluated here.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PrecursorArm {
    /// Always `not_evaluated` in this audit.
    pub status: &'static str,
    /// Why it is not evaluated.
    pub reason: &'static str,
}

/// One row of the per-query completion report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompletionReport {
    /// Protocol version.
    pub protocol: &'static str,
    /// Query id.
    pub query_id: String,
    /// `synthetic`/`instrument` note.
    pub provenance: String,
    /// Domain description.
    pub domain: String,
    /// FNV-1a over the canonical input serialization.
    pub input_hash: String,
    /// Seed (enumeration order is deterministic; recorded for provenance).
    pub seed: u64,
    /// Primary status (see the doc's status list).
    pub status: String,
    /// Every applicable doc status.
    pub statuses: Vec<String>,
    /// Resource termination reasons (limits named here).
    pub termination_reasons: Vec<String>,
    /// `accepted` | `ambiguous` | `unavailable`.
    pub mass_status: String,
    /// Accepted candidate formulas, canonical text form.
    pub accepted_formulas: Vec<String>,
    /// Mass-boundary-ambiguous formulas.
    pub ambiguous_formulas: Vec<String>,
    /// Unresolved hypotheses that block a mass-constrained zero count.
    pub unresolved_hypotheses: usize,
    /// Retained unique accepted graphs.
    pub unique_graphs: usize,
    /// Reference recovery (`Some` when a fixture reference exists).
    pub recovery: Option<bool>,
    /// All work counters.
    pub counters: Counters,
    /// Elapsed milliseconds.
    pub elapsed_ms: u128,
    /// Precursor arm: never silently fabricated.
    pub precursor: PrecursorArm,
    /// Whether the result certifies zero compatible graphs.
    pub certifies_zero: bool,
    /// Canonical traces of the accepted graphs (identities, typed order),
    /// rendered as printable token strings.
    pub accepted_identities: Vec<String>,
}

fn suffix(symbol: &str, count: u16) -> String {
    if count == 1 {
        symbol.to_string()
    } else {
        format!("{symbol}{count}")
    }
}

fn formula_text(c: &Composition) -> String {
    let mut parts = String::new();
    if c[0] > 0 {
        parts.push_str(&suffix("C", c[0]));
    }
    parts.push_str(&suffix("H", c[chem::HYDROGEN]));
    if c[2] > 0 {
        parts.push_str(&suffix("N", c[2]));
    }
    if c[3] > 0 {
        parts.push_str(&suffix("O", c[3]));
    }
    parts
}

/// Backtracking result with work-limit exhaustion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchHit {
    Yes,
    No,
    WorkLimit,
}

/// Per-pattern result with a budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContainKind {
    Contained,
    Missing,
    WorkLimit,
}

/// BFS (by path extension) connectivity order of a pattern.
fn connectivity_order(p: &[u8], p_adj: &[Vec<u8>]) -> Vec<usize> {
    let mut order = Vec::with_capacity(p.len());
    let mut visited = vec![false; p.len()];
    while let Some(start) = (0..p.len()).find(|&i| !visited[i]) {
        visited[start] = true;
        order.push(start);
        loop {
            let mut next = None;
            for c in 0..p.len() {
                if visited[c] {
                    continue;
                }
                if (0..p.len()).any(|d| visited[d] && p_adj[c][d] != 0) {
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
        if order.len() == p.len() {
            break;
        }
    }
    order
}

/// Existence of a typed subgraph embedding of `pattern` into `target`:
/// injective within the pattern, pattern bonds mapped to equal-order bonds.
/// Non-induced: the target may have extra bonds.
fn pattern_contained(
    target: &MolGraph,
    pattern: &MolGraph,
    counter: &mut Counters,
    node_limit: usize,
) -> Result<ContainKind> {
    let p = pattern.atoms();
    let t = target.atoms();
    if p.is_empty() || p.len() > t.len() {
        return Ok(ContainKind::Missing);
    }
    let mut p_adj = vec![vec![0u8; p.len()]; p.len()];
    for (a, b, o) in pattern.bonds() {
        p_adj[*a][*b] = *o;
        p_adj[*b][*a] = *o;
    }
    let mut t_adj = vec![vec![0u8; t.len()]; t.len()];
    for (a, b, o) in target.bonds() {
        t_adj[*a][*b] = *o;
        t_adj[*b][*a] = *o;
    }
    let order = connectivity_order(p, &p_adj);
    let mut map: Vec<Option<usize>> = vec![None; p.len()];
    let mut used = vec![false; t.len()];
    let found = cb_dfs(
        0, &order, p, t, &p_adj, &t_adj, &mut map, &mut used, counter, node_limit,
    );
    Ok(match found {
        SearchHit::Yes => ContainKind::Contained,
        SearchHit::No => ContainKind::Missing,
        SearchHit::WorkLimit => ContainKind::WorkLimit,
    })
}

#[allow(clippy::too_many_arguments)]
fn cb_dfs(
    depth: usize,
    order: &[usize],
    p: &[u8],
    t: &[u8],
    p_adj: &[Vec<u8>],
    t_adj: &[Vec<u8>],
    map: &mut [Option<usize>],
    used: &mut [bool],
    counter: &mut Counters,
    node_limit: usize,
) -> SearchHit {
    if depth == order.len() {
        return SearchHit::Yes;
    }
    let c = order[depth];
    for tt in 0..t.len() {
        if used[tt] || t[tt] != p[c] {
            continue;
        }
        let mut ok = true;
        for d in 0..depth {
            let c2 = order[d];
            let t2 = map[d].expect("prefix mapped");
            let need = p_adj[c][c2];
            if need != 0 && t_adj[tt][t2] != need {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        counter.embedding_nodes += 1;
        if counter.embedding_nodes > node_limit {
            return SearchHit::WorkLimit;
        }
        used[tt] = true;
        map[depth] = Some(tt);
        match cb_dfs(
            depth + 1,
            order,
            p,
            t,
            p_adj,
            t_adj,
            map,
            used,
            counter,
            node_limit,
        ) {
            SearchHit::Yes => return SearchHit::Yes,
            SearchHit::WorkLimit => return SearchHit::WorkLimit,
            SearchHit::No => {}
        }
        map[depth] = None;
        used[tt] = false;
    }
    SearchHit::No
}

/// Whether `pattern` embeds into `target` as a non-induced typed subgraph.
///
/// Thin public wrapper over the audit's [`pattern_contained`][self] search:
/// the map is injective, every pattern bond lands on a target bond of the
/// same order, and the target may hold more bonds. Returns
/// [`contain::Containment`]: `Contained`, `NotContained`, or `WorkLimit` when
/// `node_limit` extension attempts are spent first. The audit's behaviour is
/// unchanged.
pub fn contains_pattern(
    target: &MolGraph,
    pattern: &MolGraph,
    node_limit: usize,
) -> contain::Containment {
    let mut counters = Counters::default();
    // `pattern_contained` never errors today; a future error degrades to
    // `NotContained` rather than a guess of containment.
    let kind = pattern_contained(target, pattern, &mut counters, node_limit)
        .unwrap_or(ContainKind::Missing);
    match kind {
        ContainKind::Contained => contain::Containment::Contained,
        ContainKind::Missing => contain::Containment::NotContained,
        ContainKind::WorkLimit => contain::Containment::WorkLimit,
    }
}

/// Whether every pattern embeds into `target` on pairwise disjoint atom
/// sets: one joint injective embedding of all patterns with disjoint images.
///
/// Thin public wrapper over the audit's [`embeddings_compatible`][self]
/// joint search with the known-disjoint correspondence (an empty pair list:
/// distinct pattern atom occurrences must land on distinct target atoms).
/// Returns [`contain::Containment`]: `Contained`, `NotContained`, or
/// `WorkLimit` when `node_limit` extension attempts are spent first. An
/// empty pattern list is vacuously `Contained`.
pub fn contains_patterns_disjoint(
    target: &MolGraph,
    patterns: &[MolGraph],
    node_limit: usize,
) -> contain::Containment {
    let disjoint = Correspondence { pairs: Vec::new() };
    let mut counters = Counters::default();
    // `embeddings_compatible` never errors today; a future error degrades to
    // `NotContained` rather than a guess of containment.
    let kind =
        embeddings_compatible(target, patterns, Some(&disjoint), &mut counters, node_limit)
            .unwrap_or(ContainKind::Missing);
    match kind {
        ContainKind::Contained => contain::Containment::Contained,
        ContainKind::Missing => contain::Containment::NotContained,
        ContainKind::WorkLimit => contain::Containment::WorkLimit,
    }
}

/// All oracle-consistent joint embedding possibilities for `substructures`
/// in `target`; `Ok(Contained)` iff some assignment embeds every pattern with
/// the supplied forced-equality relation.
fn embeddings_compatible(
    target: &MolGraph,
    substructures: &[MolGraph],
    correspondence: Option<&Correspondence>,
    counter: &mut Counters,
    node_limit: usize,
) -> Result<ContainKind> {
    if correspondence.is_none() {
        for s in substructures {
            match pattern_contained(target, s, counter, node_limit)? {
                ContainKind::Contained => {}
                ContainKind::Missing => return Ok(ContainKind::Missing),
                ContainKind::WorkLimit => return Ok(ContainKind::WorkLimit),
            }
        }
        return Ok(ContainKind::Contained);
    }
    let pairs = &correspondence.expect("checked").pairs;
    // Fully-known equivalence relation: each pattern atom occurrence has a
    // union-find class; occurrences in different classes must land on
    // distinct target atoms (injective quotient map).
    let sizes: Vec<usize> = substructures.iter().map(|s| s.atoms().len()).collect();
    let total: usize = sizes.iter().sum();
    let offset = |s: usize, a: usize| -> usize { sizes[..s].iter().sum::<usize>() + a };
    let mut parent: Vec<usize> = (0..total).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for &(a, b) in pairs {
        let ra = find(&mut parent, offset(a.0, a.1));
        let rb = find(&mut parent, offset(b.0, b.1));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let root_of: Vec<Vec<usize>> = substructures
        .iter()
        .enumerate()
        .map(|(s, g)| {
            (0..g.atoms().len())
                .map(|a| find(&mut parent, offset(s, a)))
                .collect()
        })
        .collect();
    let mut maps: Vec<Option<Vec<usize>>> = vec![None; substructures.len()];
    match joint_dfs(
        0,
        target,
        substructures,
        pairs,
        &root_of,
        &mut maps,
        counter,
        node_limit,
    )? {
        SearchHit::Yes => Ok(ContainKind::Contained),
        SearchHit::No => Ok(ContainKind::Missing),
        SearchHit::WorkLimit => Ok(ContainKind::WorkLimit),
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn joint_dfs(
    s: usize,
    target: &MolGraph,
    substructures: &[MolGraph],
    pairs: &[((usize, usize), (usize, usize))],
    root_of: &[Vec<usize>],
    maps: &mut [Option<Vec<usize>>],
    counter: &mut Counters,
    node_limit: usize,
) -> Result<SearchHit> {
    if s == substructures.len() {
        for &(a, b) in pairs {
            let (s1, a1) = a;
            let (s2, a2) = b;
            let m1 = maps[s1].as_ref().expect("all mapped");
            let m2 = maps[s2].as_ref().expect("all mapped");
            if m1[a1] != m2[a2] {
                return Ok(SearchHit::No);
            }
        }
        return Ok(SearchHit::Yes);
    }
    let mut forced: BTreeMap<usize, usize> = BTreeMap::new();
    for &(a, b) in pairs {
        let (s1, a1) = a;
        let (s2, a2) = b;
        if s1 == s
            && let Some(m) = &maps[s2]
        {
            forced.insert(a1, m[a2]);
        }
        if s2 == s
            && let Some(m) = &maps[s1]
        {
            forced.insert(a2, m[a1]);
        }
    }
    let solutions = enumerate_one(target, &substructures[s], &forced, counter, node_limit)?;
    match solutions {
        (SearchHit::WorkLimit, _) => Ok(SearchHit::WorkLimit),
        (_, sols) if sols.is_empty() => Ok(SearchHit::No),
        (_, sols) => {
            for m in sols {
                // Distinct classes must land on distinct target atoms: any
                // collision with an earlier substructure's mapping rejects
                // this joint map before descending.
                let mut classes_ok = true;
                'outer: for sp in 0..s {
                    let mp = maps[sp].as_ref().expect("earlier mapped");
                    for a in 0..substructures[s].atoms().len() {
                        for ap in 0..substructures[sp].atoms().len() {
                            if root_of[s][a] != root_of[sp][ap] && m[a] == mp[ap] {
                                classes_ok = false;
                                break 'outer;
                            }
                        }
                    }
                }
                if !classes_ok {
                    continue;
                }
                maps[s] = Some(m);
                match joint_dfs(
                    s + 1,
                    target,
                    substructures,
                    pairs,
                    root_of,
                    maps,
                    counter,
                    node_limit,
                )? {
                    SearchHit::Yes => return Ok(SearchHit::Yes),
                    SearchHit::WorkLimit => return Ok(SearchHit::WorkLimit),
                    SearchHit::No => {}
                }
            }
            maps[s] = None;
            Ok(SearchHit::No)
        }
    }
}

/// Every embedding of one pattern, constrained by forced assignments.
fn enumerate_one(
    target: &MolGraph,
    pattern: &MolGraph,
    forced: &BTreeMap<usize, usize>,
    counter: &mut Counters,
    node_limit: usize,
) -> Result<(SearchHit, Vec<Vec<usize>>)> {
    let p = pattern.atoms();
    let t = target.atoms();
    if p.is_empty() || p.len() > t.len() {
        return Ok((SearchHit::No, Vec::new()));
    }
    let mut p_adj = vec![vec![0u8; p.len()]; p.len()];
    for (a, b, o) in pattern.bonds() {
        p_adj[*a][*b] = *o;
        p_adj[*b][*a] = *o;
    }
    let mut t_adj = vec![vec![0u8; t.len()]; t.len()];
    for (a, b, o) in target.bonds() {
        t_adj[*a][*b] = *o;
        t_adj[*b][*a] = *o;
    }
    let order = connectivity_order(p, &p_adj);
    let mut map: Vec<Option<usize>> = vec![None; p.len()];
    let mut used = vec![false; t.len()];
    let mut out = Vec::new();
    let res = enum_collect(
        0, &order, p, t, &p_adj, &t_adj, forced, &mut map, &mut used, counter, node_limit, &mut out,
    );
    let hit = if res == SearchHit::WorkLimit {
        SearchHit::WorkLimit
    } else if out.is_empty() {
        SearchHit::No
    } else {
        SearchHit::Yes
    };
    Ok((hit, out))
}

#[allow(clippy::too_many_arguments)]
fn enum_collect(
    depth: usize,
    order: &[usize],
    p: &[u8],
    t: &[u8],
    p_adj: &[Vec<u8>],
    t_adj: &[Vec<u8>],
    forced: &BTreeMap<usize, usize>,
    map: &mut [Option<usize>],
    used: &mut [bool],
    counter: &mut Counters,
    node_limit: usize,
    out: &mut Vec<Vec<usize>>,
) -> SearchHit {
    if depth == order.len() {
        // Reindex by pattern atom: `map` is progression-indexed (`order`),
        // callers reason in terms of the pattern's own atom indices.
        let mut position = vec![0usize; p.len()];
        for (d, &a) in order.iter().enumerate() {
            position[a] = d;
        }
        let mut by_atom = Vec::with_capacity(p.len());
        for depth_of_atom in &position {
            by_atom.push(map[*depth_of_atom].expect("placed"));
        }
        out.push(by_atom);
        return SearchHit::Yes;
    }
    let c = order[depth];
    for tt in 0..t.len() {
        if used[tt] || t[tt] != p[c] {
            continue;
        }
        if let Some(&want) = forced.get(&c)
            && want != tt
        {
            continue;
        }
        let mut ok = true;
        for d in 0..depth {
            let c2 = order[d];
            let t2 = map[d].expect("prefix mapped");
            let need = p_adj[c][c2];
            if need != 0 && t_adj[tt][t2] != need {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        counter.embedding_nodes += 1;
        if counter.embedding_nodes > node_limit {
            return SearchHit::WorkLimit;
        }
        used[tt] = true;
        map[depth] = Some(tt);
        if enum_collect(
            depth + 1,
            order,
            p,
            t,
            p_adj,
            t_adj,
            forced,
            map,
            used,
            counter,
            node_limit,
            out,
        ) == SearchHit::WorkLimit
        {
            return SearchHit::WorkLimit;
        }
        map[depth] = None;
        used[tt] = false;
    }
    SearchHit::No
}

fn query_input_hash(query: &CompletionQuery) -> String {
    let mut input_parts: Vec<String> = Vec::new();
    input_parts.push(query.id.clone());
    input_parts.push(format!("prov={}", query.provenance));
    input_parts.push(query.domain.describe());
    input_parts.push(format!("mass={}", query.observed_mass_uda));
    input_parts.push(format!("ppm={}", query.mass.ppm_tenths));
    input_parts.push(format!("msource={}", query.mass.source));
    match query.mass.uncertainty_uda {
        Some(u) => input_parts.push(format!("unc={u}")),
        None => input_parts.push("unc=none".to_string()),
    }
    for s in &query.substructures {
        let atoms: Vec<String> = s.atoms().iter().map(|a| a.to_string()).collect();
        let bonds: Vec<String> = s
            .bonds()
            .iter()
            .map(|(a, b, o)| format!("{a}-{b}:{o}"))
            .collect();
        input_parts.push(format!("pat[{}|{}]", atoms.join(","), bonds.join("|")));
    }
    if let Some(c) = &query.correspondence {
        for ((s1, a1), (s2, a2)) in &c.pairs {
            input_parts.push(format!("eq[{s1},{a1}]=[{s2},{a2}]"));
        }
    }
    if let Some(r) = &query.reference {
        let atoms: Vec<String> = r.atoms().iter().map(|a| a.to_string()).collect();
        let bonds: Vec<String> = r
            .bonds()
            .iter()
            .map(|(a, b, o)| format!("{a}-{b}:{o}"))
            .collect();
        input_parts.push(format!("ref[{}|{}]", atoms.join(","), bonds.join("|")));
    }
    format!(
        "{:016x}",
        stable_hash(&input_parts.iter().map(|s| s.as_str()).collect::<Vec<_>>())
    )
}

/// Validation of one entire query before any search.
fn validate_query(query: &CompletionQuery, budgets: &CompletionBudgets) -> Result<()> {
    if query.substructures.len() > 8 {
        return Err(Error::Unsupported(
            "unsupported_input: at most 8 required substructures per query".to_string(),
        ));
    }
    if query.substructures.iter().any(|s| s.atoms().is_empty()) {
        return Err(Error::Unsupported(
            "unsupported_input: empty substructure".to_string(),
        ));
    }
    for s in &query.substructures {
        if s.atoms().len() > query.domain.heavy_range().1 {
            return Err(Error::Unsupported(
                "unsupported_input: substructure exceeds the declared heavy-atom bound".to_string(),
            ));
        }
        for id in s.atoms() {
            if !query.domain.admits_type(*id) {
                return Err(Error::Unsupported(format!(
                    "unsupported_input: substructure uses atom type {id} outside the declared domain"
                )));
            }
        }
    }
    let joint_atoms: usize = query.substructures.iter().map(|s| s.atoms().len()).sum();
    if joint_atoms > 24 {
        return Err(Error::Unsupported(
            "unsupported_input: total required pattern atoms exceed the audit cap of 24"
                .to_string(),
        ));
    }
    if let Some(c) = &query.correspondence
        && c.pairs.len() > 16
    {
        return Err(Error::Unsupported(
            "unsupported_input: correspondence exceeds the audit cap of 16 pairs".to_string(),
        ));
    }
    if query.domain.heavy_range().1 > 6 {
        return Err(Error::Unsupported(
            "unsupported_input: domain max heavy atoms must be <= 6 for the V0 audit".to_string(),
        ));
    }
    if query.mass.ppm_tenths > 1000 {
        return Err(Error::Unsupported(
            "unsupported_input: ppm_tenths exceeds the 1000 proof bound".to_string(),
        ));
    }
    if budgets.formula_visits == 0
        || budgets.graph_extensions == 0
        || budgets.embedding_nodes == 0
        || budgets.canonical_expansions == 0
        || budgets.retained_graphs == 0
        || budgets.memory_bytes == 0
    {
        return Err(Error::Unsupported(
            "unsupported_input: budget fields must be positive".to_string(),
        ));
    }
    if let Some(reference) = &query.reference {
        for id in reference.atoms() {
            if !query.domain.admits_type(*id) {
                return Err(Error::Unsupported(
                    "unsupported_input: reference uses atom type outside the declared domain"
                        .to_string(),
                ));
            }
        }
        let heavy = reference.atoms().len();
        if heavy < query.domain.heavy_range().0 || heavy > query.domain.heavy_range().1 {
            return Err(Error::Unsupported(
                "unsupported_input: reference heavy-atom count outside the declared domain"
                    .to_string(),
            ));
        }
        if reference.residual_valence().iter().any(|&r| r != 0) || !reference.is_connected() {
            return Err(Error::Unsupported(
                "unsupported_input: reference must be closed-shell and connected".to_string(),
            ));
        }
        if reference.ring_closures() > query.domain.max_ring_closures() {
            return Err(Error::Unsupported(
                "unsupported_input: reference exceeds the domain ring bound".to_string(),
            ));
        }
    }
    if let Some(c) = &query.correspondence {
        c.validate(&query.substructures)?;
    }
    Ok(())
}

/// Identity of an accepted target under the remaining canonicalization budget.
fn identity_of(
    graph: &MolGraph,
    limits: Limits,
    budget: usize,
    counters: &mut Counters,
) -> Result<Vec<Token>> {
    match grammar::canonical_trace(graph, limits, budget) {
        Ok(c) => {
            counters.canonical_expansions += c.expansions;
            Ok(c.trace)
        }
        Err(Error::Unsupported(msg)) if msg.contains("canonicalization_budget_exceeded") => {
            counters.canonical_expansions += budget.saturating_add(1);
            Err(Error::Unsupported(
                "canonicalization_budget_exceeded".to_string(),
            ))
        }
        Err(e) => Err(e),
    }
}

fn estimate_bytes(g: &MolGraph, trace: &[Token]) -> usize {
    g.atoms().len() * 8 + g.bonds().len() * 32 + trace.len() * 8 + 256
}

/// Depth-first trace expansion from START with all budgets enforced. Each
/// STOP state is validated for exact composition, zero residual valence,
/// connectedness, any-domain types, and at-most-one-cycle bones before
/// patterns and identity filtering.
fn complete_search(
    query: &CompletionQuery,
    budget_comp: &Composition,
    budgets: &CompletionBudgets,
    counters: &mut Counters,
    graphs: &mut BTreeMap<Vec<Token>, MolGraph>,
    reasons: &mut Vec<String>,
    deadline: Instant,
    memory_level: &mut usize,
) -> Result<()> {
    let limits = Limits::new(
        query.domain.heavy_range().1,
        query.domain.max_ring_closures(),
    )?;
    let start_tok = Token {
        kind: grammar::START,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    let mut stack: Vec<Vec<Token>> = vec![vec![start_tok]];
    while let Some(path) = stack.pop() {
        if Instant::now() >= deadline {
            reasons.push("watchdog".to_string());
            return Ok(());
        }
        // Frontier accounting: every in-flight path is a partial result
        // structure of length startTok..stop, bounded by 4+max_atoms+max_closures.
        let frontier_bytes: usize = stack
            .iter()
            .map(|p| p.len() * std::mem::size_of::<Token>() + std::mem::size_of::<Vec<Token>>())
            .sum();
        counters.memory_estimate_bytes = counters
            .memory_estimate_bytes
            .max(*memory_level + frontier_bytes);
        if *memory_level + frontier_bytes > budgets.memory_bytes {
            reasons.push("memory_bound".to_string());
            return Ok(());
        }
        let state = match grammar::replay(&path, limits, Some(*budget_comp)) {
            Ok(s) => s,
            Err(e) => return Err(Error::Config(e.to_string())),
        };
        if state.stopped() {
            counters.completed_candidates += 1;
            let mol = state.graph()?;
            let heavy = mol.atoms().len();
            if heavy < query.domain.heavy_range().0 || heavy > query.domain.heavy_range().1 {
                continue;
            }
            if mol.composition() != *budget_comp {
                continue;
            }
            if mol.residual_valence().iter().any(|&r| r != 0) {
                continue;
            }
            if !mol.is_connected() || mol.ring_closures() > query.domain.max_ring_closures() {
                continue;
            }
            if mol.atoms().iter().any(|t| !query.domain.admits_type(*t)) {
                continue;
            }
            let mut ok = true;
            if Instant::now() >= deadline {
                reasons.push("watchdog".to_string());
                return Ok(());
            }
            if query.correspondence.is_none() {
                for s in &query.substructures {
                    match pattern_contained(&mol, s, counters, budgets.embedding_nodes)? {
                        ContainKind::Contained => {}
                        ContainKind::Missing => {
                            ok = false;
                            break;
                        }
                        ContainKind::WorkLimit => {
                            reasons.push("embedding_node_limit".to_string());
                            return Ok(());
                        }
                    }
                }
            } else {
                match embeddings_compatible(
                    &mol,
                    &query.substructures,
                    query.correspondence.as_ref(),
                    counters,
                    budgets.embedding_nodes,
                )? {
                    ContainKind::Contained => {}
                    ContainKind::Missing => ok = false,
                    ContainKind::WorkLimit => {
                        reasons.push("embedding_node_limit".to_string());
                        return Ok(());
                    }
                }
            }
            if !ok {
                continue;
            }
            if Instant::now() >= deadline {
                reasons.push("watchdog".to_string());
                return Ok(());
            }
            let budget_left = budgets
                .canonical_expansions
                .saturating_sub(counters.canonical_expansions);
            let identity = match identity_of(&mol, limits, budget_left, counters) {
                Ok(id) => id,
                Err(e) => {
                    if e.to_string().contains("canonicalization_budget_exceeded") {
                        reasons.push("canonicalization_limit".to_string());
                        return Ok(());
                    }
                    return Err(e);
                }
            };
            if !graphs.contains_key(&identity) {
                *memory_level += estimate_bytes(&mol, &identity);
                if *memory_level > budgets.memory_bytes {
                    counters.memory_estimate_bytes =
                        counters.memory_estimate_bytes.max(*memory_level);
                    reasons.push("memory_bound".to_string());
                    return Ok(());
                }
                if graphs.len() >= budgets.retained_graphs {
                    counters.memory_estimate_bytes =
                        counters.memory_estimate_bytes.max(*memory_level);
                    reasons.push("retained_graph_limit".to_string());
                    return Ok(());
                }
                graphs.insert(identity, mol);
                counters.retained_graphs += 1;
                counters.memory_estimate_bytes = counters.memory_estimate_bytes.max(*memory_level);
            }
            continue;
        }
        let mut candidates: Vec<Token> = Vec::new();
        for kind in [grammar::ADD_ATOM, grammar::CLOSE_RING, grammar::STOP] {
            for atom_type in 0u8..=17 {
                for bond in 0u8..=3 {
                    for pointer in 0u8..16 {
                        let tok = Token {
                            kind,
                            atom_type,
                            bond,
                            pointer,
                        };
                        if state.is_legal(tok) {
                            candidates.push(tok);
                        }
                    }
                }
            }
        }
        for tok in candidates {
            counters.graph_extensions += 1;
            if counters.graph_extensions > budgets.graph_extensions {
                reasons.push("graph_extension_limit".to_string());
                return Ok(());
            }
            if Instant::now() >= deadline {
                reasons.push("watchdog".to_string());
                return Ok(());
            }
            let mut next_path = path.clone();
            next_path.push(tok);
            let mut st = TraceState::new(limits, Some(*budget_comp));
            for t in &next_path {
                st.apply(*t)?;
            }
            stack.push(next_path);
        }
        let _ = state;
    }
    Ok(())
}

/// Enumerate all domain-bounded formulas within the visit budget.
fn enumerate_formulas(
    domain: &CompletionDomain,
    observed: u32,
    evidence: &MassEvidence,
    counters: &mut Counters,
    budgets: &CompletionBudgets,
    reasons: &mut Vec<String>,
    deadline: Instant,
) -> (Vec<Composition>, Vec<Composition>, MassStatus) {
    let deadline_start = deadline;
    let mut accepted = Vec::new();
    let mut ambiguous = Vec::new();
    let mut status = MassStatus::Accepted;
    // Deterministic lexicographic order over (C, N, O, H).
    for c in 0u16..=6 {
        for n in 0u16..=6 {
            for o in 0u16..=6 {
                let heavy = (c + n + o) as usize;
                if heavy < domain.heavy_range().0 || heavy > domain.heavy_range().1 {
                    continue;
                }
                if !domain.elements().contains(&0) && c > 0 {
                    continue;
                }
                if !domain.elements().contains(&2) && n > 0 {
                    continue;
                }
                if !domain.elements().contains(&3) && o > 0 {
                    continue;
                }
                // Closed-shell valence budget: H <= 2C + N + 2 with parity H
                // congruent to N (mod 2); tighter rings only reduce H, and
                // the exact sum is enforced later by zero residual valence.
                let max_h = 2 * c + n + 2;
                let mut h = if n % 2 == 0 { 0u16 } else { 1u16 };
                while h <= max_h {
                    counters.formula_visits += 1;
                    if counters.formula_visits > budgets.formula_visits {
                        reasons.push("formula_visit_limit".to_string());
                        return (accepted, ambiguous, status);
                    }
                    if Instant::now() >= deadline_start {
                        reasons.push("watchdog".to_string());
                        return (accepted, ambiguous, status);
                    }
                    let mut comp: Composition = [0; 10];
                    comp[0] = c;
                    comp[chem::HYDROGEN] = h;
                    comp[2] = n;
                    comp[3] = o;
                    let verdict = match evidence.uncertainty_uda {
                        None => {
                            status = MassStatus::Unavailable;
                            None
                        }
                        Some(unc) => {
                            let mass = match chem::composition_mass(&comp) {
                                Ok(m) => m,
                                Err(_) => return (accepted, ambiguous, status),
                            };
                            let nda: u64 = comp
                                .iter()
                                .enumerate()
                                .map(|(e, k)| {
                                    u64::from(*k) * u64::from(chem::ELEMENTS[e].residual_nda)
                                })
                                .sum();
                            let error = (nda + 999) / 1000 + u64::from(unc);
                            let tol = u64::from(chem::tolerance(observed, evidence.ppm_tenths));
                            let r = observed.abs_diff(mass) as u64;
                            Some(match (r + error <= tol, r > tol + error) {
                                (true, _) => chem::Verdict::Accept,
                                (false, true) => chem::Verdict::Reject,
                                _ => chem::Verdict::Ambiguous,
                            })
                        }
                    };
                    if let Some(v) = verdict {
                        match v {
                            chem::Verdict::Accept => accepted.push(comp),
                            chem::Verdict::Reject => {}
                            chem::Verdict::Ambiguous => {
                                ambiguous.push(comp);
                                status = MassStatus::Ambiguous;
                            }
                        }
                    }
                    h += 2;
                }
            }
        }
    }
    if reasons.is_empty() && status == MassStatus::Accepted && accepted.is_empty() {
        status = MassStatus::Rejected;
    }
    (accepted, ambiguous, status)
}

/// Run the bounded completion: validate, enumerate formulas, then traces,
/// then pattern match, then canonicalize, then recover the reference.
pub fn run(
    query: &CompletionQuery,
    budgets: &CompletionBudgets,
    seed: u64,
) -> Result<CompletionReport> {
    let started = Instant::now();
    let deadline = match started.checked_add(budgets.watchdog) {
        Some(d) => d,
        None => {
            return Err(Error::Unsupported(format!(
                "unsupported_input: watchdog duration {:?} overflow the runtime clock",
                budgets.watchdog
            )));
        }
    };
    let input_hash = query_input_hash(query);
    if let Err(e) = validate_query(query, budgets) {
        let status = if e.to_string().contains("unsupported_input") {
            "unsupported_input"
        } else {
            "reference_error"
        };
        return Ok(CompletionReport {
            protocol: COMPLETION_VERSION,
            query_id: query.id.clone(),
            provenance: query.provenance.clone(),
            domain: query.domain.describe(),
            input_hash,
            seed,
            status: status.to_string(),
            statuses: vec![status.to_string()],
            termination_reasons: vec![e.to_string()],
            mass_status: "not_evaluated".to_string(),
            accepted_formulas: Vec::new(),
            ambiguous_formulas: Vec::new(),
            unresolved_hypotheses: 0,
            unique_graphs: 0,
            recovery: None,
            counters: Counters::default(),
            elapsed_ms: started.elapsed().as_millis(),
            precursor: PrecursorArm {
                status: "not_evaluated",
                reason: "not evaluated because the query failed input validation",
            },
            certifies_zero: false,
            accepted_identities: Vec::new(),
        });
    }
    let mut counters = Counters::default();
    let mut reasons: Vec<String> = Vec::new();

    let (accepted, ambiguous, mut mass_status) = enumerate_formulas(
        &query.domain,
        query.observed_mass_uda,
        &query.mass,
        &mut counters,
        budgets,
        &mut reasons,
        deadline,
    );
    if query.mass.uncertainty_uda.is_none() {
        mass_status = MassStatus::Unavailable;
    }
    if reasons.is_empty() && Instant::now() >= deadline {
        reasons.push("watchdog".to_string());
    }

    let mut graphs: BTreeMap<Vec<Token>, MolGraph> = BTreeMap::new();
    let mut memory_level = 0usize;
    for comp in &accepted {
        if !reasons.is_empty() {
            break;
        }
        complete_search(
            query,
            comp,
            budgets,
            &mut counters,
            &mut graphs,
            &mut reasons,
            deadline,
            &mut memory_level,
        )?;
    }

    let recovery: Option<bool> = query.reference.as_ref().and_then(|reference| {
        // Reference auditing shares the search budget; exhausted resources
        // mean recovery is unknown rather than a negative match.
        if Instant::now() >= deadline
            || counters.canonical_expansions >= budgets.canonical_expansions
            || reasons
                .iter()
                .any(|r| r == "memory_bound" || r == "watchdog" || r == "canonicalization_limit")
        {
            return None;
        }
        let limits = Limits::new(
            query.domain.heavy_range().1,
            query.domain.max_ring_closures(),
        )
        .expect("validated");
        let budget_left = budgets
            .canonical_expansions
            .saturating_sub(counters.canonical_expansions);
        match identity_of(reference, limits, budget_left, &mut counters) {
            Ok(id) => Some(graphs.contains_key(&id)),
            Err(e) => {
                if e.to_string().contains("canonicalization_budget_exceeded") {
                    reasons.push("canonicalization_limit".to_string());
                }
                None
            }
        }
    });

    // Final wall-clock gate: no result may claim `complete` after the
    // deadline, even if the elapsed time was consumed by late matching or
    // the reference canonicalization above.
    if Instant::now() >= deadline && !reasons.iter().any(|r| r == "watchdog") {
        reasons.push("watchdog".to_string());
    }

    let mut statuses: Vec<String> = Vec::new();
    if !reasons.is_empty() {
        statuses.push("search_budget_exhausted".to_string());
    }
    match mass_status {
        MassStatus::Accepted | MassStatus::Rejected => {}
        MassStatus::Ambiguous => statuses.push("mass_evidence_unresolved".to_string()),
        MassStatus::Unavailable => statuses.push("mass_evidence_unresolved".to_string()),
    }
    if statuses.is_empty() {
        statuses.push("complete".to_string());
    }
    let primary = if reasons.is_empty()
        && matches!(mass_status, MassStatus::Accepted | MassStatus::Rejected)
    {
        "complete".to_string()
    } else if !reasons.is_empty() {
        "search_budget_exhausted".to_string()
    } else {
        "mass_evidence_unresolved".to_string()
    };

    let permitted_zero = statuses.len() == 1 && statuses[0] == "complete";

    let elapsed_ms = started.elapsed().as_millis();
    let unresolved_hypotheses = match mass_status {
        MassStatus::Accepted | MassStatus::Rejected => 0,
        MassStatus::Ambiguous => ambiguous.len(),
        MassStatus::Unavailable => counters.formula_visits,
    };

    let accepted_identities = graphs
        .keys()
        .map(|toks| {
            toks.iter()
                .map(|t| format!("{}/{}/{}/{}", t.kind, t.atom_type, t.bond, t.pointer))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>();

    Ok(CompletionReport {
        protocol: COMPLETION_VERSION,
        query_id: query.id.clone(),
        provenance: query.provenance.clone(),
        domain: query.domain.describe(),
        input_hash,
        seed,
        status: primary,
        statuses,
        termination_reasons: reasons,
        mass_status: mass_status.name().to_string(),
        accepted_formulas: accepted.iter().map(formula_text).collect(),
        ambiguous_formulas: ambiguous.iter().map(formula_text).collect(),
        unresolved_hypotheses,
        unique_graphs: graphs.len(),
        recovery,
        counters,
        elapsed_ms,
        precursor: PrecursorArm {
            status: "not_evaluated",
            reason: "no independently validated compatible parent/target pairs exist in this audit; a synthetic precursor mass is not fragmentation evidence",
        },
        certifies_zero: graphs.is_empty() && permitted_zero,
        accepted_identities,
    })
}

/// Verdict expectation for one fixture, checked by tests and parity.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FixtureExpected {
    /// Doc status string.
    pub status: String,
    /// `accepted` | `ambiguous` | `unavailable`.
    pub mass_status: String,
    /// Unique graphs.
    pub unique_graphs: usize,
    /// Accepted formulas.
    pub accepted_formulas: Vec<String>,
    /// Ambiguous formulas.
    pub ambiguous_formulas: Vec<String>,
    /// Optional check of unresolved count.
    pub unresolved_hypotheses: Option<usize>,
    /// Optional check of recovery.
    pub recovery: Option<bool>,
    /// Zero-certification expectation.
    pub certifies_zero: bool,
    /// Optional exact list of expected termination reasons (documented only
    /// for the genuinely truncated fixtures).
    #[serde(default)]
    pub termination_reasons: Vec<String>,
}

/// A loaded fixture: the query plus its manually established expectation.
pub struct Fixture {
    /// Name.
    pub name: String,
    /// Category tag for stratification.
    pub category: String,
    /// Provenance string.
    pub provenance: String,
    /// The query.
    pub query: CompletionQuery,
    /// Expectation.
    pub expected: FixtureExpected,
}

fn element_id(s: &str) -> Result<usize> {
    chem::element_index(s)
        .ok_or_else(|| Error::Unsupported(format!("unsupported_input: unknown element {s}")))
}

fn build_graph(doc: &serde_json::Value, what: &str) -> Result<MolGraph> {
    let atom_id = |v: &serde_json::Value| -> Result<u8> {
        match v.as_u64() {
            Some(n) if n <= u64::from(u8::MAX) => Ok(n as u8),
            _ => Err(Error::Unsupported(format!(
                "{what}: atom type ids must be non-negative integers <= 255"
            ))),
        }
    };
    let atoms: Vec<u8> = doc["atoms"]
        .as_array()
        .ok_or_else(|| Error::Unsupported(format!("{what}: atoms must be a list")))?
        .iter()
        .map(atom_id)
        .collect::<Result<_>>()?;
    let bonds = doc["bonds"]
        .as_array()
        .ok_or_else(|| Error::Unsupported(format!("{what}: bonds must be a list")))?
        .iter()
        .map(|b| {
            let parts = b
                .as_array()
                .ok_or_else(|| Error::Unsupported(format!("{what}: bond must be a list")))?;
            if parts.len() != 3 {
                return Err(Error::Unsupported(format!("{what}: bond has 3 fields")));
            }
            let a = parts[0].as_u64().ok_or_else(|| {
                Error::Unsupported(format!("{what}: bond endpoint must be an integer"))
            })?;
            let b_ = parts[1].as_u64().ok_or_else(|| {
                Error::Unsupported(format!("{what}: bond endpoint must be an integer"))
            })?;
            let o = parts[2].as_u64().ok_or_else(|| {
                Error::Unsupported(format!("{what}: bond order must be an integer"))
            })?;
            Ok((
                usize::try_from(a)
                    .map_err(|_| Error::Unsupported(format!("{what}: bond endpoint too large")))?,
                usize::try_from(b_)
                    .map_err(|_| Error::Unsupported(format!("{what}: bond endpoint too large")))?,
                u8::try_from(o)
                    .map_err(|_| Error::Unsupported(format!("{what}: bond order too large")))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    MolGraph::new(atoms, bonds)
}

/// Load the fixture set from its provenance JSON.
pub fn load_fixture_set(text: &str) -> Result<Vec<Fixture>> {
    let root: serde_json::Value = serde_json::from_str(text)?;
    let items = root["fixtures"]
        .as_array()
        .ok_or_else(|| Error::Unsupported("fixtures: missing 'fixtures' list".to_string()))?;
    let mut out = Vec::new();
    for f in items {
        let name = f["name"].as_str().unwrap_or("").to_string();
        let category = f["category"].as_str().unwrap_or("").to_string();
        let provenance = f["provenance"].as_str().unwrap_or("synthetic").to_string();
        let observed_mass_uda = f["observed_mass_uda"]
            .as_u64()
            .ok_or_else(|| {
                Error::Unsupported(format!("{name}: missing or invalid observed_mass_uda"))
            })
            .and_then(|v| {
                u32::try_from(v).map_err(|_| {
                    Error::Unsupported(format!("{name}: observed_mass_uda {v} overflows u32"))
                })
            })?;
        let m = &f["mass"];
        let mass = MassEvidence {
            ppm_tenths: match m["ppm_tenths"].as_u64() {
                Some(v) => u32::try_from(v)
                    .map_err(|_| {
                        Error::Unsupported(format!("{name}: mass.ppm_tenths overflows u32"))
                    })
                    .and_then(|p| {
                        if p > 1000 {
                            Err(Error::Unsupported(format!(
                                "{name}: mass.ppm_tenths {p} exceeds the 1000 proof bound"
                            )))
                        } else {
                            Ok(p)
                        }
                    })?,
                None => 100,
            },
            uncertainty_uda: match m["uncertainty_uda"].as_u64() {
                Some(v) => Some(u32::try_from(v).map_err(|_| {
                    Error::Unsupported(format!("{name}: mass.uncertainty_uda overflows u32"))
                })?),
                None => {
                    if m["uncertainty_uda"].is_null() || m.get("uncertainty_uda").is_none() {
                        None
                    } else {
                        return Err(Error::Unsupported(format!(
                            "{name}: mass.uncertainty_uda must be null or a non-negative integer"
                        )));
                    }
                }
            },
            source: m["source"].as_str().unwrap_or("synthetic").to_string(),
        };
        let d = &f["domain"];
        let dom_field = |k: &str, default: u32| -> Result<u32> {
            match d.get(k) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        Error::Unsupported(format!(
                            "{name}: domain.{k} must be a non-negative integer"
                        ))
                    }),
            }
        };
        let elements: Vec<usize> = d["elements"]
            .as_array()
            .ok_or_else(|| Error::Unsupported(format!("{name}: domain.elements")))?
            .iter()
            .map(|e| element_id(e.as_str().unwrap_or("")))
            .collect::<Result<_>>()?;
        let domain = CompletionDomain::new(
            &elements,
            dom_field("min_heavy", 2)? as usize,
            dom_field("max_heavy", 6)? as usize,
            dom_field("max_ring_closures", 1)? as usize,
        )?;
        let mut substructures = Vec::new();
        if f.get("patterns").is_some_and(|p| !p.is_array()) {
            return Err(Error::Unsupported(format!(
                "{name}: patterns must be an array"
            )));
        }
        if let Some(patterns) = f["patterns"].as_array() {
            for (i, p) in patterns.iter().enumerate() {
                substructures.push(build_graph(p, &format!("{name}:patterns[{i}]"))?);
            }
        }
        if f.get("correspondence").is_some_and(|c| !c.is_array()) {
            return Err(Error::Unsupported(format!(
                "{name}: correspondence must be an array"
            )));
        }
        let mut correspondence = None;
        if let Some(c) = f["correspondence"].as_array() {
            let mut pairs = Vec::new();
            for pair in c {
                let get_idx = |v: &serde_json::Value, what: &str| -> Result<usize> {
                    match v.as_u64() {
                        Some(n) => usize::try_from(n).map_err(|_| {
                            Error::Unsupported(format!("{name}: correspondence {what} too large"))
                        }),
                        None => Err(Error::Unsupported(format!(
                            "{name}: correspondence must be pairs of integer indices"
                        ))),
                    }
                };
                let endpoints = pair.as_array().ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{name}: correspondence entries must be pairs ofpairs"
                    ))
                })?;
                if endpoints.len() != 2 {
                    return Err(Error::Unsupported(format!(
                        "{name}: correspondence entry must have exactly two endpoints"
                    )));
                }
                let read_pair = |v: &serde_json::Value| -> Result<(usize, usize)> {
                    let ap = v.as_array().ok_or_else(|| {
                        Error::Unsupported(format!(
                            "{name}: correspondence endpoint must be [sub, atom]"
                        ))
                    })?;
                    if ap.len() != 2 {
                        return Err(Error::Unsupported(format!(
                            "{name}: correspondence endpoint must be [sub, atom]"
                        )));
                    }
                    Ok((get_idx(&ap[0], "substructure")?, get_idx(&ap[1], "atom")?))
                };
                pairs.push((read_pair(&endpoints[0])?, read_pair(&endpoints[1])?));
            }
            correspondence = Some(Correspondence { pairs });
        }
        let reference = f["reference"]
            .as_object()
            .map(|obj| {
                build_graph(
                    &serde_json::Value::Object(obj.clone()),
                    &format!("{name}:reference"),
                )
            })
            .transpose()?;
        let expected: FixtureExpected = serde_json::from_value(f["expected"].clone())?;
        out.push(Fixture {
            name: name.clone(),
            category,
            provenance,
            query: CompletionQuery {
                id: name,
                provenance: f["provenance"].as_str().unwrap_or("synthetic").to_string(),
                observed_mass_uda,
                mass,
                substructures,
                correspondence,
                domain,
                reference,
            },
            expected,
        });
    }
    Ok(out)
}

/// Parse a single request (fixture-shaped query plus optional `budgets` and
/// `seed`) and return a full JSON report. Budget fields absent from the
/// request keep their declared defaults.
pub fn run_request_json(text: &str) -> Result<String> {
    let root: serde_json::Value = serde_json::from_str(text)?;
    // Assertions live on fixture sets, not in a single query: the same
    // machine-readable report is returned either way.
    let mut fixture_v = match root.get("fixtures").and_then(|f| f.as_array()) {
        Some(list) => list
            .first()
            .cloned()
            .ok_or_else(|| Error::Unsupported("request: empty fixtures list".to_string()))?,
        None => root.clone(),
    };
    if !fixture_v.is_object() {
        return Err(Error::Unsupported(
            "request must be an object or a fixtures list".to_string(),
        ));
    }
    if let Some(obj) = fixture_v.as_object_mut() {
        obj.entry("expected").or_insert(serde_json::json!({
            "status": "",
            "mass_status": "",
            "unique_graphs": 0,
            "accepted_formulas": [],
            "ambiguous_formulas": [],
            "certifies_zero": false
        }));
    }
    let doc = serde_json::json!({"fixtures": [fixture_v]});
    let mut fixtures = load_fixture_set(&doc.to_string())?;
    let fixture = fixtures.pop().expect("one fixture");
    let mut budgets = CompletionBudgets::default();
    if let Some(b) = root.get("budgets") {
        if !b.is_object() {
            return Err(Error::Unsupported("budgets must be an object".to_string()));
        }
        let num = |k: &str| -> Result<usize> {
            match b.get(k) {
                None => Ok(0),
                Some(v) => v.as_u64().map(|n| n as usize).ok_or_else(|| {
                    Error::Unsupported(format!("budgets.{k} must be a non-negative integer"))
                }),
            }
        };
        if b.get("formula_visits").is_some() {
            budgets.formula_visits = num("formula_visits")?;
        }
        if b.get("graph_extensions").is_some() {
            budgets.graph_extensions = num("graph_extensions")?;
        }
        if b.get("embedding_nodes").is_some() {
            budgets.embedding_nodes = num("embedding_nodes")?;
        }
        if b.get("canonical_expansions").is_some() {
            budgets.canonical_expansions = num("canonical_expansions")?;
        }
        if b.get("retained_graphs").is_some() {
            budgets.retained_graphs = num("retained_graphs")?;
        }
        if b.get("memory_bytes").is_some() {
            budgets.memory_bytes = num("memory_bytes")?;
        }
        if let Some(v) = b.get("watchdog_ms") {
            let ms = v.as_u64().ok_or_else(|| {
                Error::Unsupported("budgets.watchdog_ms must be a non-negative integer".into())
            })?;
            budgets.watchdog = Duration::from_millis(ms);
        }
        let unknown = b
            .as_object()
            .map(|m| {
                m.keys()
                    .filter(|k| {
                        !matches!(
                            k.as_str(),
                            "formula_visits"
                                | "graph_extensions"
                                | "embedding_nodes"
                                | "canonical_expansions"
                                | "retained_graphs"
                                | "memory_bytes"
                                | "watchdog_ms"
                        )
                    })
                    .count()
            })
            .unwrap_or(0);
        if unknown != 0 {
            return Err(Error::Unsupported(
                "budgets carries unknown fields; refused to guess semantics".to_string(),
            ));
        }
        if budgets.formula_visits == 0
            || budgets.graph_extensions == 0
            || budgets.embedding_nodes == 0
            || budgets.canonical_expansions == 0
            || budgets.retained_graphs == 0
            || budgets.memory_bytes == 0
        {
            return Err(Error::Unsupported(
                "budgets fields must be positive".to_string(),
            ));
        }
    }
    let seed = match root.get("seed") {
        None => 1,
        Some(value) => value
            .as_u64()
            .ok_or_else(|| Error::Unsupported("seed must be a non-negative integer".to_string()))?,
    };
    let report = run(&fixture.query, &budgets, seed)?;
    Ok(serde_json::to_string_pretty(&report)?)
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn stable_hash_distinguishes_partition_formats() {
        assert_eq!(stable_hash(&["a", "bc"]), stable_hash(&["a", "bc"]));
        assert_ne!(stable_hash(&["a", "bc"]), stable_hash(&["ab", "c"]));
    }

    /// Byte-level proof: `enumerate_one` produces solutions indexed by
    /// pattern atom, even when the pattern's connectivity traversal places
    /// out-of-order components (cross-edge 0-2, 1-3).
    #[test]
    fn enumerated_solutions_use_pattern_atom_indices() {
        let target = MolGraph::new(
            vec![3, 3, 3, 3],
            vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)],
        )
        .unwrap();
        let pattern = MolGraph::new(vec![3, 3, 3, 3], vec![(0, 2, 1), (1, 3, 1)]).unwrap();
        let mut counter = Counters::default();
        let (_, sols) =
            enumerate_one(&target, &pattern, &BTreeMap::new(), &mut counter, 1000).unwrap();
        assert!(!sols.is_empty());
        for m in &sols {
            for (a, b, order) in pattern.bonds() {
                let have = target.bonds().iter().any(|(x, y, o)| {
                    (*x == m[*a] && *y == m[*b] || *x == m[*b] && *y == m[*a]) && o == order
                });
                assert!(
                    have,
                    "solution record misread: map {m:?} order pair {a}-{b}"
                );
            }
        }
    }

    /// Reordering pattern atom indices does not change joint containment.
    #[test]
    fn joint_membership_is_insensitive_to_pattern_index_permutation() {
        let target = MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
        let pert_a = MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
        let pert_b = MolGraph::new(vec![9, 3, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
        // pert_b: O-atom first, then C-chain reversed: (O <- C(H2) <- C(H3)).
        let mut ca = Counters::default();
        let mut cb = Counters::default();
        let a = embeddings_compatible(&target, std::slice::from_ref(&pert_a), None, &mut ca, 1000)
            .unwrap();
        let b = embeddings_compatible(&target, std::slice::from_ref(&pert_b), None, &mut cb, 1000)
            .unwrap();
        assert_eq!(a, ContainKind::Contained);
        assert_eq!(b, ContainKind::Contained);
    }

    /// An oracle chain forcing two distinct atoms of one substructure onto a
    /// single target atom is inconsistent before any work is done.
    #[test]
    fn correspondence_chain_collapse_is_rejected() {
        let s = [
            MolGraph::new(vec![4, 4], vec![(0, 1, 1)]).unwrap(),
            MolGraph::new(vec![4], vec![]).unwrap(),
            MolGraph::new(vec![4], vec![]).unwrap(),
        ];
        let bad = Correspondence {
            pairs: vec![((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (0, 1))],
        };
        assert!(bad.validate(&s).is_err());
    }

    /// An oracle requiring incompatible shared atom types is rejected.
    #[test]
    fn correspondence_with_unequal_types_is_rejected() {
        let s = [
            MolGraph::new(vec![4], Vec::new()).unwrap(),
            MolGraph::new(vec![9], Vec::new()).unwrap(),
        ];
        let bad = Correspondence {
            pairs: vec![((0, 0), (1, 0))],
        };
        assert!(bad.validate(&s).is_err());
    }

    /// Original grammar STOP semantics remain usable by subgraph callers:
    /// early STOP with open valence and unused budget is still legal.
    #[test]
    fn trace_state_keeps_subgraph_stop_legality() {
        let limits = Limits::V0;
        let mut st = TraceState::new(limits, None);
        st.apply(Token {
            kind: grammar::START,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        })
        .unwrap();
        st.apply(Token {
            kind: grammar::ADD_ATOM,
            atom_type: 4,
            bond: 0,
            pointer: 0,
        })
        .unwrap();
        assert!(st.is_legal(Token {
            kind: grammar::STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        }));
        st.apply(Token {
            kind: grammar::STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        })
        .unwrap();
        assert!(st.stopped());
        assert!(!st.residual_valence().iter().all(|&r| r == 0));
    }

    /// Every accepted-identity token string returned by the public report
    /// decodes through the existing grammar into a closed, connected target
    /// whose composition satisfies the domain.
    #[test]
    fn reported_identities_decode_to_closed_connected_targets() {
        for name in ["c2h6o_mass_only", "c2h7n_mass_only", "c4h8_mass_only"] {
            let text = std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json"
            ))
            .unwrap();
            let fixtures = load_fixture_set(&text).unwrap();
            let f = fixtures.iter().find(|f| f.name == name).unwrap();
            let report = run(&f.query, &CompletionBudgets::default(), 1).unwrap();
            for id in &report.accepted_identities {
                let mut toks = Vec::new();
                for part in id.split(',') {
                    let mut fields = part.split('/');
                    let kind: u8 = fields.next().unwrap().parse().unwrap();
                    let atom_type = fields.next().unwrap().parse().unwrap();
                    let bond = fields.next().unwrap().parse().unwrap();
                    let pointer = fields.next().unwrap().parse().unwrap();
                    toks.push(Token {
                        kind,
                        atom_type,
                        bond,
                        pointer,
                    });
                }
                let state = grammar::replay(&toks, Limits::new(6, 1).unwrap(), None).unwrap();
                assert!(state.stopped(), "{name}: not stopped");
                let mol = state.graph().unwrap();
                assert!(mol.residual_valence().iter().all(|&r| r == 0), "{name}");
                assert!(mol.is_connected(), "{name}");
            }
        }
    }
}
