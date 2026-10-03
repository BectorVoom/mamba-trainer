//! Pseudo-label recipe `q-cut-v1` (contract §7.2): subgraph enumeration,
//! peak filtering and the mass-match weights the GPU pipeline trains on.
//!
//! Pure host Rust with integer masses only: no kernels, no tensors, no new
//! dependencies. Every mass decision goes through [`decide`] in integer
//! units, exactly as the reference `tools/ms2/ms2_reference.py` does.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};

use super::chem::{Composition, Ion, Verdict, adduct, decide, ion, tolerance};
use super::grammar::{CANONICAL_WORK_LIMIT, Limits, Token, canonical_trace};
use super::graph::MolGraph;

/// Recipe version of contract §7.2, stored with every label file.
pub const RECIPE_VERSION: &str = "q-cut-v1";

/// Intensity units of contract §7.2 step 5: a relative intensity is stored
/// as `floor(r * 2^20 + 0.5)` integer units.
pub const INTENSITY_UNITS: u64 = 1 << 20;
/// Split factor of contract §7.2 step 5: `4096 * lcm(1..=16)`, so splitting
/// one peak's units among up to 16 graphs is exact.
pub const SPLIT_UNITS: u64 = 2_952_069_120;

/// A relative intensity in integer units of 2^-20, rounded half up
/// (contract §7.2 step 5).
pub fn intensity_units(relative: f64) -> u64 {
    (relative * INTENSITY_UNITS as f64 + 0.5).floor() as u64
}

/// One peak's share per explaining graph: `intensity_units(relative) *
/// SPLIT_UNITS / graphs` (contract §7.2 step 5).
///
/// Errors on `graphs == 0` or on overflow of the multiplication, so weight
/// arithmetic never wraps.
pub fn peak_share(relative: f64, graphs: usize) -> Result<u64> {
    if graphs == 0 {
        return Err(Error::config(
            "peak_share: graphs is 0: a peak explained by no graph has no share".to_string(),
        ));
    }
    let units = intensity_units(relative);
    let product = units.checked_mul(SPLIT_UNITS).ok_or_else(|| {
        Error::config(format!(
            "peak_share: {units} intensity units times {SPLIT_UNITS} overflows u64"
        ))
    })?;
    Ok(product / graphs as u64)
}

/// Candidate-size limits of the recipe; [`RecipeLimits::V0`] is frozen.
#[derive(Clone, Copy, Debug)]
pub struct RecipeLimits {
    /// Fewest atoms per candidate embedding.
    pub min_atoms: usize,
    /// Most atoms per candidate embedding.
    pub max_atoms: usize,
    /// Most ring closures (`internal bonds − atoms + 1`) per embedding.
    pub max_closures: usize,
    /// Most parent bonds removed per cut set (the empty set is included).
    pub max_cuts: usize,
    /// Most hydrogens shifted in an ion hypothesis (each way).
    pub max_shift: i32,
    /// Most graphs kept by the retention step.
    pub max_targets: usize,
}

impl RecipeLimits {
    /// The frozen V0 limits: 3–16 atoms, 4 closures, 2 cuts, shift 2, 16 targets.
    pub const V0: Self = Self {
        min_atoms: 3,
        max_atoms: 16,
        max_closures: 4,
        max_cuts: 2,
        max_shift: 2,
        max_targets: 16,
    };
}

/// One candidate subgraph: a connected induced atom set of the parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Embedding {
    /// Sorted parent atom indices.
    pub atoms: Vec<usize>,
    /// Parent bonds with exactly one end inside (a property of the atom set).
    pub boundary: usize,
    /// Ring closures: internal bonds minus atoms plus one.
    pub closures: usize,
}

/// Disjoint-set union over the parent atoms, used to cut bonds.
struct UnionFind {
    /// Parent pointers; a root points to itself.
    parent: Vec<usize>,
}

impl UnionFind {
    /// One singleton set per atom.
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    /// Root of `x`'s set, with path halving.
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merge the sets of `a` and `b`.
    fn union(&mut self, a: usize, b: usize) {
        let root = self.find(a);
        let other = self.find(b);
        if root != other {
            self.parent[root] = other;
        }
    }
}

/// Every cut set of at most `max_cuts` bond indices, empty set first.
fn for_each_cut_set(m: usize, max_cuts: usize, f: &mut impl FnMut(&[usize])) {
    fn rec(start: usize, m: usize, left: usize, cur: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
        f(cur);
        if left == 0 {
            return;
        }
        for i in start..m {
            cur.push(i);
            rec(i + 1, m, left - 1, cur, f);
            cur.pop();
        }
    }
    rec(0, m, max_cuts, &mut Vec::new(), f);
}

/// Every recipe candidate of the parent (contract §7.2 step 2).
///
/// Each set of at most [`RecipeLimits::max_cuts`] parent bonds (including the
/// empty set) whose removal leaves each removed bond with its two ends in
/// different components contributes every resulting component with
/// `min_atoms..=max_atoms` atoms and at most `max_closures` ring closures.
/// The map key is the atom set, so duplicates from different cut sets merge;
/// the map order is lexicographic by atom set.
pub fn enumerate_embeddings(parent: &MolGraph, limits: &RecipeLimits) -> Vec<Embedding> {
    let n = parent.atoms().len();
    let bonds = parent.bonds();
    let m = bonds.len();
    let mut found: BTreeMap<Vec<usize>, Embedding> = BTreeMap::new();
    let mut is_cut = vec![false; m];
    let mut handle = |cut: &[usize]| {
        for &i in cut {
            is_cut[i] = true;
        }
        let mut sets = UnionFind::new(n);
        for (i, (a, b, _)) in bonds.iter().enumerate() {
            if !is_cut[i] {
                sets.union(*a, *b);
            }
        }
        let roots: Vec<usize> = (0..n).map(|i| sets.find(i)).collect();
        let separated = cut.iter().all(|&i| {
            let (a, b, _) = bonds[i];
            roots[a] != roots[b]
        });
        if separated {
            let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (i, root) in roots.iter().enumerate() {
                groups.entry(*root).or_default().push(i);
            }
            for members in groups.into_values() {
                if members.len() < limits.min_atoms || members.len() > limits.max_atoms {
                    continue;
                }
                let mut inside = vec![false; n];
                for &a in &members {
                    inside[a] = true;
                }
                let mut internal = 0usize;
                let mut boundary = 0usize;
                for (a, b, _) in bonds {
                    match (inside[*a], inside[*b]) {
                        (true, true) => internal += 1,
                        (false, false) => {}
                        _ => boundary += 1,
                    }
                }
                // Components are connected through the kept bonds, so
                // `internal >= members.len() - 1` and the subtraction holds.
                let closures = internal - (members.len() - 1);
                if closures <= limits.max_closures {
                    found.entry(members.clone()).or_insert(Embedding {
                        atoms: members,
                        boundary,
                        closures,
                    });
                }
            }
        }
        for &i in cut {
            is_cut[i] = false;
        }
    };
    for_each_cut_set(m, limits.max_cuts, &mut handle);
    found.into_values().collect()
}

/// One filtered spectrum peak with its caller-side identity.
#[derive(Clone, Copy, Debug)]
pub struct Peak {
    /// Index of the peak in the caller's original list.
    pub id: u32,
    /// m/z in integer units of 10⁻⁶ dalton.
    pub mz: u32,
    /// Linear relative intensity (1.0 is the spectrum maximum).
    pub intensity: f64,
}

/// Contract §2 without the cap: keep `0 < mz <= precursor + 2 Da`, divide by
/// the maximum of the kept peaks, then keep relative intensities `>= 1e-3`.
///
/// Empty when no peak passes the mass bound or the maximum is not positive.
/// Order and ids are preserved.
///
/// Errors ([`Error::Config`]) when the three slices differ in length, or
/// when an intensity is NaN, infinite or negative: request validation
/// (contract §3.1) is supposed to have rejected those before this point, so
/// they are reported with their index instead of being silently dropped.
pub fn filter_peaks(
    ids: &[u32],
    mz: &[u32],
    intensity: &[f64],
    precursor_mz: u32,
) -> Result<Vec<Peak>> {
    if ids.len() != mz.len() || ids.len() != intensity.len() {
        return Err(Error::config(format!(
            "filter_peaks: ids/mz/intensity length mismatch ({} / {} / {})",
            ids.len(),
            mz.len(),
            intensity.len()
        )));
    }
    for (i, &v) in intensity.iter().enumerate() {
        if v.is_nan() || v.is_infinite() || v < 0.0 {
            return Err(Error::config(format!(
                "filter_peaks: intensity[{i}] is {v}: validate the request first"
            )));
        }
    }
    let bound = precursor_mz.saturating_add(2_000_000);
    let kept: Vec<usize> = (0..mz.len())
        .filter(|&i| mz[i] != 0 && mz[i] <= bound)
        .collect();
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    let top = kept
        .iter()
        .map(|&i| intensity[i])
        .fold(f64::NEG_INFINITY, f64::max);
    // The intensities are finite and non-negative here, so a non-positive
    // maximum means every kept peak is exactly zero: nothing to normalise by.
    if top <= 0.0 {
        return Ok(Vec::new());
    }
    Ok(kept
        .into_iter()
        .filter(|&i| intensity[i] / top >= 1e-3)
        .map(|i| Peak {
            id: ids[i],
            mz: mz[i],
            intensity: intensity[i] / top,
        })
        .collect())
}

/// One retained pseudo-label: a graph with its weight and peak evidence.
#[derive(Clone, Debug)]
pub struct Target {
    /// Canonical trace identifying the graph.
    pub trace: Vec<Token>,
    /// Raw weight: the `u64` sum of the peak shares (contract §7.2 step 5).
    pub weight: u64,
    /// `weight as f64 / sum of kept weights as f64`.
    pub q: f64,
    /// Indices into the [`Labels::embeddings`] list behind this graph.
    pub embeddings: Vec<usize>,
    /// `(peak id, shift)` pairs with an accepted hypothesis, sorted.
    pub anchors: Vec<(u32, i32)>,
}

/// The pseudo-labels of one spectrum: steps 3–7 of contract §7.2.
#[derive(Clone, Debug)]
pub struct Labels {
    /// All recipe candidates, in [`enumerate_embeddings`] order.
    pub embeddings: Vec<Embedding>,
    /// Distinct canonical traces among the embeddings.
    pub graphs: usize,
    /// Graphs with non-zero weight before the top-`max_targets` cut.
    pub targets_before_cut: usize,
    /// Kept targets, by decreasing weight, ties by smaller trace.
    pub targets: Vec<Target>,
    /// Fraction of total weight outside the kept targets (0 without targets).
    pub dropped_weight: f64,
    /// Whether the top-16 cut falls between two graphs of equal integer
    /// weight: more than 16 graphs have weight and the 16th and 17th largest
    /// weights are equal (the report's `cut_is_tied`).
    pub cut_is_tied: bool,
    /// Peak ids explained by at least one graph, ascending.
    pub explained_peaks: Vec<u32>,
    /// `(peak, embedding, shift)` triples with an Ambiguous verdict.
    pub ambiguous_hypotheses: usize,
    /// Embeddings dropped before matching on the canonicalization budget.
    pub canonicalization_failures: usize,
}

/// Molecule-level preparation of contract §7.2 steps 2–3, reused for every
/// spectrum of the molecule.
///
/// Canonicalization dominates the cost and does not depend on the spectrum,
/// so [`Candidates::new`] runs it once while [`Candidates::label`] runs the
/// per-spectrum matching (steps 4–7) on the cached traces and compositions.
#[derive(Clone, Debug)]
pub struct Candidates {
    /// All recipe candidates, in [`enumerate_embeddings`] order.
    embeddings: Vec<Embedding>,
    /// Graph index per embedding (`None` for budget failures).
    graph_of_emb: Vec<Option<usize>>,
    /// Distinct canonical traces, in first-seen order.
    graph_traces: Vec<Vec<Token>>,
    /// Element composition per embedding, in embedding order.
    compositions: Vec<Composition>,
    /// Embeddings dropped on the canonicalization budget.
    canonicalization_failures: usize,
    /// Largest `Canonical::expansions` seen while canonicalizing.
    max_expansions: usize,
    /// Recipe limits the candidates were prepared with.
    limits: RecipeLimits,
}

impl Candidates {
    /// Enumerate the parent's embeddings and canonicalize each one once.
    ///
    /// Embeddings are grouped into graphs by canonical trace with a map keyed
    /// by trace (first-seen order), and a budget failure drops the embedding
    /// before any matching and is counted.
    pub fn new(parent: &MolGraph, limits: &RecipeLimits) -> Result<Self> {
        let embeddings = enumerate_embeddings(parent, limits);
        let canon = Limits::new(limits.max_atoms, limits.max_closures)?;
        let mut compositions: Vec<Composition> = Vec::with_capacity(embeddings.len());
        let mut graph_of_emb: Vec<Option<usize>> = Vec::with_capacity(embeddings.len());
        let mut graph_traces: Vec<Vec<Token>> = Vec::new();
        let mut index_of: BTreeMap<Vec<Token>, usize> = BTreeMap::new();
        let mut canonicalization_failures = 0usize;
        let mut max_expansions = 0usize;
        for emb in &embeddings {
            let sub = parent.induced(&emb.atoms)?;
            compositions.push(sub.composition());
            match canonical_trace(&sub, canon, CANONICAL_WORK_LIMIT) {
                Ok(found) => {
                    max_expansions = max_expansions.max(found.expansions);
                    let graph = match index_of.get(&found.trace) {
                        Some(&g) => g,
                        None => {
                            let g = graph_traces.len();
                            graph_traces.push(found.trace.clone());
                            index_of.insert(found.trace, g);
                            g
                        }
                    };
                    graph_of_emb.push(Some(graph));
                }
                Err(e) => {
                    if e.to_string().contains("canonicalization_budget_exceeded") {
                        canonicalization_failures += 1;
                        graph_of_emb.push(None);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Ok(Self {
            embeddings,
            graph_of_emb,
            graph_traces,
            compositions,
            canonicalization_failures,
            max_expansions,
            limits: *limits,
        })
    }

    /// All recipe candidates, in [`enumerate_embeddings`] order.
    pub fn embeddings(&self) -> &[Embedding] {
        &self.embeddings
    }

    /// Distinct canonical traces among the embeddings.
    pub fn graphs(&self) -> usize {
        self.graph_traces.len()
    }

    /// Embeddings dropped before matching on the canonicalization budget.
    pub fn canonicalization_failures(&self) -> usize {
        self.canonicalization_failures
    }

    /// The largest `Canonical::expansions` seen while canonicalizing.
    pub fn max_expansions(&self) -> usize {
        self.max_expansions
    }

    /// Match peaks and retain targets (contract §7.2 steps 4–7).
    ///
    /// Each peak of relative intensity `r` contributes
    /// [`peak_share`]`(r, n)` integer units to each of the `n` graphs it
    /// accepts (any embedding, any allowed shift); the `max_targets`
    /// heaviest graphs are kept, comparing the integers, ties broken by the
    /// smaller trace. Ion hypotheses are computed once per embedding, not
    /// once per peak.
    ///
    /// An unknown observation precision (`mz_uncertainty == u32::MAX`,
    /// contract §5) disables exact-mass decisions: no hypothesis is
    /// evaluated, so the labels hold no target, no explained peak and no
    /// ambiguous hypothesis.
    pub fn label(
        &self,
        peaks: &[Peak],
        adduct_id: u16,
        ppm_tenths: u32,
        mz_uncertainty: u32,
    ) -> Result<Labels> {
        if adduct(adduct_id).is_none() {
            return Err(Error::Unsupported(format!(
                "pseudo_labels: unknown adduct id {adduct_id}"
            )));
        }
        let idle = || Labels {
            embeddings: self.embeddings.clone(),
            graphs: self.graph_traces.len(),
            targets_before_cut: 0,
            targets: Vec::new(),
            dropped_weight: 0.0,
            cut_is_tied: false,
            explained_peaks: Vec::new(),
            ambiguous_hypotheses: 0,
            canonicalization_failures: self.canonicalization_failures,
        };
        if mz_uncertainty == u32::MAX {
            return Ok(idle());
        }
        let limits = &self.limits;
        let n_graphs = self.graph_traces.len();
        let mut hyps: Vec<Vec<(i32, Ion)>> = Vec::with_capacity(self.embeddings.len());
        for (i, emb) in self.embeddings.iter().enumerate() {
            let mut cached = Vec::new();
            if self.graph_of_emb[i].is_some() {
                let cap = emb.boundary.min(limits.max_shift.max(0) as usize) as i32;
                for s in -cap..=cap {
                    if let Some(hyp) = ion(&self.compositions[i], adduct_id, s)? {
                        cached.push((s, hyp));
                    }
                }
            }
            hyps.push(cached);
        }
        let mut weights = vec![0u64; n_graphs];
        let mut hit_ever = vec![false; n_graphs];
        let mut anchors: Vec<BTreeSet<(u32, i32)>> = vec![BTreeSet::new(); n_graphs];
        let mut explained: BTreeSet<u32> = BTreeSet::new();
        let mut ambiguous_hypotheses = 0usize;
        for peak in peaks {
            let tol = tolerance(peak.mz, ppm_tenths);
            let mut hit: Vec<BTreeSet<i32>> = (0..n_graphs).map(|_| BTreeSet::new()).collect();
            for (i, cached) in hyps.iter().enumerate() {
                let Some(graph) = self.graph_of_emb[i] else {
                    continue;
                };
                for (s, hyp) in cached {
                    let error = hyp.error.saturating_add(mz_uncertainty);
                    match decide(peak.mz, hyp.mz, error, tol) {
                        Verdict::Accept => {
                            hit[graph].insert(*s);
                        }
                        Verdict::Ambiguous => {
                            ambiguous_hypotheses += 1;
                        }
                        Verdict::Reject => {}
                    }
                }
            }
            let n_hit = hit.iter().filter(|h| !h.is_empty()).count();
            if n_hit == 0 {
                continue;
            }
            explained.insert(peak.id);
            let share = peak_share(peak.intensity, n_hit)?;
            for (g, shifts) in hit.iter().enumerate() {
                if shifts.is_empty() {
                    continue;
                }
                hit_ever[g] = true;
                weights[g] = weights[g].checked_add(share).ok_or_else(|| {
                    Error::config(format!("pseudo_labels: weight of graph {g} overflows u64"))
                })?;
                anchors[g].extend(shifts.iter().map(|&s| (peak.id, s)));
            }
        }
        let mut total: u64 = 0;
        for w in &weights {
            total = total.checked_add(*w).ok_or_else(|| {
                Error::config("pseudo_labels: total weight overflows u64".to_string())
            })?;
        }
        let mut order: Vec<usize> = (0..n_graphs).filter(|&g| hit_ever[g]).collect();
        order.sort_by(|&a, &b| {
            weights[b]
                .cmp(&weights[a])
                .then(self.graph_traces[a].cmp(&self.graph_traces[b]))
        });
        let targets_before_cut = order.len();
        // The report's `cut_is_tied`: the frozen 16-target cut falls between
        // two graphs of equal integer weight.
        let cut_is_tied = targets_before_cut > RecipeLimits::V0.max_targets
            && weights[order[RecipeLimits::V0.max_targets - 1]]
                == weights[order[RecipeLimits::V0.max_targets]];
        let keep = order.len().min(limits.max_targets);
        let mut kept_sum: u64 = 0;
        for &g in &order[..keep] {
            kept_sum = kept_sum.checked_add(weights[g]).ok_or_else(|| {
                Error::config("pseudo_labels: kept weight overflows u64".to_string())
            })?;
        }
        let mut targets = Vec::with_capacity(keep);
        for &g in &order[..keep] {
            let members: Vec<usize> = self
                .graph_of_emb
                .iter()
                .enumerate()
                .filter_map(|(i, og)| (*og == Some(g)).then_some(i))
                .collect();
            targets.push(Target {
                trace: self.graph_traces[g].clone(),
                weight: weights[g],
                q: if kept_sum > 0 {
                    weights[g] as f64 / kept_sum as f64
                } else {
                    0.0
                },
                embeddings: members,
                anchors: anchors[g].iter().copied().collect(),
            });
        }
        Ok(Labels {
            embeddings: self.embeddings.clone(),
            graphs: n_graphs,
            targets_before_cut,
            targets,
            dropped_weight: if total > 0 && keep > 0 {
                1.0 - kept_sum as f64 / total as f64
            } else {
                0.0
            },
            cut_is_tied,
            explained_peaks: explained.into_iter().collect(),
            ambiguous_hypotheses,
            canonicalization_failures: self.canonicalization_failures,
        })
    }
}

/// Group embeddings into graphs, match peaks and retain targets.
///
/// Preparation (enumeration and canonicalization) runs once per molecule in
/// [`Candidates::new`]; this wrapper keeps the one-call shape for existing
/// callers.
pub fn pseudo_labels(
    parent: &MolGraph,
    peaks: &[Peak],
    adduct_id: u16,
    ppm_tenths: u32,
    mz_uncertainty: u32,
    limits: &RecipeLimits,
) -> Result<Labels> {
    Candidates::new(parent, limits)?.label(peaks, adduct_id, ppm_tenths, mz_uncertainty)
}
