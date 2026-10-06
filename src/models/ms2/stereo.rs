//! Stereo-aware perception and enumeration for completion candidates
//! (`stereo-perception-v2`).
//!
//! The training data carries no stereo labels and the model's evidence (a
//! formula and substructures) carries none, so nothing here predicts a
//! stereoisomer. Instead every candidate graph is made stereo-aware: which
//! stereo elements its constitution supports, how many distinct
//! stereoisomers those elements generate, and each of them on request, under
//! a convention an external tool can turn into isomeric SMILES. `MolGraph`
//! identity stays constitutional; stereo is a separate layer, and ranking is
//! unchanged (constitutional).
//!
//! ## Potential elements (local, cheap, over-inclusive)
//!
//! * **Tetrahedral carbon**: a carbon whose total coordination is four —
//!   heavy neighbours plus hydrogens equals four, i.e. every bond is single
//!   — with at most one hydrogen. Three heavy neighbours and one hydrogen
//!   counts (bromochlorofluoromethane is a centre).
//! * **Double bond**: a C=C, C=N or N=N bond of order 2 where each end has,
//!   besides the partner, either two single-bonded heavy substituents, or one
//!   heavy substituent and one hydrogen, or (nitrogen) one heavy substituent
//!   or one hydrogen plus its lone pair. Excluded: an end with two
//!   hydrogens; an end that carries another multiple bond (cumulene/allene);
//!   any bond whose smallest ring has fewer than 8 atoms; any bond in a ring
//!   of 8 or more atoms that is not isolated (see below).
//!
//! A ring double bond whose smallest ring has 8 or more atoms is a potential
//! stereo element only when it is **isolated in every ring through it**:
//! neither end atom may have a *ring* neighbour (other than its partner,
//! within the same ring system) that carries a double bond or is a lone-pair
//! donor (N, O, S with only single bonds). A non-isolated large-ring double
//! bond is excluded and the molecule gets
//! `unsupported: "conjugated_large_ring"`. Ring systems come from ring-bond
//! connectivity (bonds lying on some cycle), so the verdict never depends on
//! which shortest ring a search happens to find first.
//!
//! Nothing else is a potential element in this version. In particular
//! amines are never centres (nitrogen inversion is fast, not stereo).
//!
//! ## Convention
//!
//! * Tetrahedral: the ligand list `L` is the centre's heavy neighbours in
//!   ascending atom index, then `H` when the centre has a hydrogen. Value
//!   `cw` (1) means: looking from `L[0]` toward the centre,
//!   `L[1] -> L[2] -> L[3]` runs clockwise; `ccw` (0) the opposite.
//! * Double bond `(a, b)` with `a < b`: each end's *reference ligand* is its
//!   lowest-index heavy substituent, else its hydrogen, else (nitrogen) its
//!   lone pair. Value `cis` (0): the two reference ligands are on the same
//!   side; `trans` (1): opposite sides.
//! * An *assignment* is one value per potential element, in element order:
//!   tetrahedral centres by atom index, then double bonds by `(a, b)`.
//!
//! ## Exact stereogenicity by automorphisms
//!
//! The automorphisms of the constitutional graph (bijections preserving atom
//! type and bond order) are enumerated with the bounded DFS of
//! [`enumerate_isomorphisms`](super::completion_data::enumerate_isomorphisms).
//! An automorphism acts on an assignment: a centre maps to its image with a
//! flip for an odd ligand permutation; a double bond flips iff exactly one
//! end's mapped reference ligand differs from the image end's reference.
//! Two assignments are equivalent iff some automorphism maps one to the
//! other. The orbits of all `2^k` assignments (`k` potential elements) are
//! the exact distinct stereoisomers within the supported element kinds; the
//! canonical representative of an orbit is its lexicographically smallest
//! assignment. An element is stereogenic iff flipping it alone changes the
//! orbit for at least one assignment.
//!
//! Caps (`max_elements`, `max_automorphisms`, `work_limit`) make the result
//! `unresolved` when exceeded — never a guess.
//!
//! ## Out of scope
//!
//! `unsupported` names kinds present in the molecule that this version does
//! not model, so nobody reads the count as molecule-wide truth when it is
//! not:
//!
//! * `axial_cumulene`: a carbon or nitrogen with two double bonds.
//! * `conjugated_large_ring`: a non-isolated double bond in a ring of 8 or
//!   more atoms (possible aromaticity or conjugation, including through
//!   lone-pair donors).
//! * `constrained_nitrogen_center`: a nitrogen with three single bonds (to
//!   heavy atoms or hydrogens) in a ring of 3 or 4 atoms, or at the
//!   bridgehead of a bicyclic system (flagged as uncertain, never counted).
//! * `atropisomer_axis_possible`: a single bond between two sp2 ring atoms
//!   in different rings where at least three of the four ortho positions
//!   carry a heavy substituent (a candidate hindered axis).
//! * `phosphorus_center`: any phosphorus with three or more heavy
//!   neighbours.
//! * `sulfur_center`: any sulfur with three or more heavy neighbours.
//!
//! Atropisomerism (hindered rotation) and conformational stereo (ring
//! puckers, rotamers) are otherwise out of scope entirely.
//!
//! `molecule_wide_exact` is true iff the computation resolved **and** no
//! unmodelled kind above is present: any molecule that may carry stereo of
//! an unmodelled kind is never reported molecule-wide exact.
//!
//! Deterministic and pure host. The orbit computation holds two `usize`
//! assignment tables of `2^k` entries (`k` potential elements) plus one
//! image/flip action table per automorphism and the canonical
//! representatives; `max_elements` is clamped to an absolute cap of 16
//! inside [`perceive`] and every shift and allocation size is checked, so an
//! excessive caller limit reports `unresolved` instead of panicking or
//! over-allocating.

use std::collections::BTreeMap;

use super::completion_data::enumerate_isomorphisms;
use super::graph::MolGraph;

/// Version string of the stereo perception recipe in this module.
pub const STEREO_VERSION: &str = "stereo-perception-v2";

/// Element index of carbon in [`chem::ELEMENTS`](super::chem::ELEMENTS).
const CARBON: usize = 0;
/// Element index of nitrogen.
const NITROGEN: usize = 2;
/// Element index of oxygen.
const OXYGEN: usize = 3;
/// Element index of phosphorus.
const PHOSPHORUS: usize = 5;
/// Element index of sulfur.
const SULFUR: usize = 6;

/// Absolute cap on analysed potential elements, enforced inside
/// [`perceive`] (and [`equivalent`]) no matter what the caller passes:
/// more elements report `unresolved` instead of panicking or allocating
/// `2^k` tables for an excessive `k`.
const ABSOLUTE_MAX_ELEMENTS: usize = 16;

/// One stereo ligand: a heavy neighbour atom, an implicit hydrogen, or a
/// nitrogen lone pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ligand {
    /// Heavy neighbour atom index.
    Atom(usize),
    /// The centre's/end's implicit hydrogen.
    Hydrogen,
    /// A nitrogen lone pair (double-bond ends only).
    LonePair,
}

/// One potential stereo element, in assignment order (tetrahedral centres by
/// atom index, then double bonds by `(a, b)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StereoElement {
    /// Potential tetrahedral carbon: `ligands` is `L` of the convention
    /// (heavy neighbours ascending, then `H` when present).
    Tetrahedral {
        /// Centre atom index.
        atom: usize,
        /// Ligand list `L`.
        ligands: Vec<Ligand>,
    },
    /// Potential stereo double bond with `a < b` and each end's reference
    /// ligand (lowest-index heavy substituent, else hydrogen, else a
    /// nitrogen lone pair).
    DoubleBond {
        /// First endpoint (`a < b`).
        a: usize,
        /// Second endpoint.
        b: usize,
        /// Reference ligand of end `a`.
        ref_a: Ligand,
        /// Reference ligand of end `b`.
        ref_b: Ligand,
    },
}

/// A stereogenic element with its index among the potential elements (the
/// assignment order position).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedElement {
    /// Index among [`StereoReport::potential`].
    pub potential_index: usize,
    /// The element itself.
    pub element: StereoElement,
}

/// Budgets for [`perceive`] and [`equivalent`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StereoLimits {
    /// Most potential elements analysed: more gives
    /// `unresolved: too_many_elements`.
    pub max_elements: usize,
    /// Most automorphisms enumerated: more gives
    /// `unresolved: too_many_automorphisms`.
    pub max_automorphisms: usize,
    /// Work units for the automorphism search: spending it gives
    /// `unresolved: work_limit_exceeded`.
    pub work_limit: usize,
    /// Most canonical isomers kept in [`StereoReport::isomers`].
    pub max_expanded: usize,
}

impl Default for StereoLimits {
    /// Ten elements, 20,000 automorphisms, a 100,000-unit work limit, and no
    /// isomer expansion.
    fn default() -> Self {
        Self {
            max_elements: 10,
            max_automorphisms: 20_000,
            work_limit: 100_000,
            max_expanded: 0,
        }
    }
}

/// Whether the stereo analysis resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// Orbits computed exactly within the supported kinds.
    Resolved,
    /// A cap was hit; the string names it (`too_many_elements`,
    /// `too_many_automorphisms`, `work_limit_exceeded`, or
    /// `automorphism_mapping`).
    Unresolved(String),
}

impl Resolution {
    /// The protocol word: `"resolved"` or `"unresolved: <reason>"`.
    pub fn text(&self) -> String {
        match self {
            Resolution::Resolved => "resolved".to_string(),
            Resolution::Unresolved(reason) => format!("unresolved: {reason}"),
        }
    }

    /// True for [`Resolution::Resolved`].
    pub fn is_resolved(&self) -> bool {
        matches!(self, Resolution::Resolved)
    }
}

/// Stereo analysis of one candidate graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StereoReport {
    /// Every potential element, in assignment order.
    pub potential: Vec<StereoElement>,
    /// The stereogenic elements among them, in assignment order, each with
    /// its index among [`potential`](StereoReport::potential).
    pub elements: Vec<IndexedElement>,
    /// Potential elements that are not stereogenic (counted, but dropped
    /// from [`elements`](StereoReport::elements)).
    pub not_stereogenic: usize,
    /// Unmodelled kinds present in the molecule (`axial_cumulene`,
    /// `conjugated_large_ring`, `phosphorus_center`, `sulfur_center`),
    /// sorted.
    pub unsupported: Vec<String>,
    /// `2^k` for `k` potential elements, or `None` when unresolved.
    pub raw_assignments: Option<u64>,
    /// The number of orbits (exact distinct stereoisomers within the
    /// supported kinds), or `None` when unresolved.
    pub distinct: Option<u64>,
    /// Whether the orbits were computed exactly.
    pub resolution: Resolution,
    /// The automorphism count, or `None` when unresolved.
    pub automorphisms: Option<u64>,
    /// Canonical representatives (lexicographically smallest assignment of
    /// each orbit) projected onto the stereogenic elements, in lexicographic
    /// order, at most `max_expanded` entries.
    pub isomers: Vec<Vec<u8>>,
    /// True when more orbits exist than [`isomers`](StereoReport::isomers)
    /// holds (only when expansion was requested).
    pub isomers_truncated: bool,
}

impl StereoReport {
    /// True iff the computation resolved (exact within the supported kinds).
    pub fn complete_within_supported_kinds(&self) -> bool {
        self.resolution.is_resolved()
    }

    /// True iff the computation resolved and no unmodelled kind is present.
    pub fn molecule_wide_exact(&self) -> bool {
        self.resolution.is_resolved() && self.unsupported.is_empty()
    }
}

/// An unresolved report over a known potential set: counts are `None`,
/// nothing is claimed.
fn unresolved_report(
    potential: Vec<StereoElement>,
    unsupported: Vec<String>,
    reason: &str,
) -> StereoReport {
    StereoReport {
        potential,
        elements: Vec::new(),
        not_stereogenic: 0,
        unsupported,
        raw_assignments: None,
        distinct: None,
        resolution: Resolution::Unresolved(reason.to_string()),
        automorphisms: None,
        isomers: Vec::new(),
        isomers_truncated: false,
    }
}
struct AtomView {
    /// Index into the element table.
    element: usize,
    /// Parent hydrogen count from the atom type.
    hydrogens: u8,
    /// Incident `(neighbour, order)` pairs.
    neighbours: Vec<(usize, u8)>,
}

/// Local views of every atom of `graph`.
fn views(graph: &MolGraph) -> Vec<AtomView> {
    use super::chem::atom_type;
    let n = graph.atoms().len();
    let mut out: Vec<AtomView> = Vec::with_capacity(n);
    for id in graph.atoms().iter() {
        let t = atom_type(*id).expect("stored type ids are valid");
        out.push(AtomView {
            element: t.element,
            hydrogens: t.hydrogens,
            neighbours: Vec::new(),
        });
    }
    for (a, b, order) in graph.bonds() {
        out[*a].neighbours.push((*b, *order));
        out[*b].neighbours.push((*a, *order));
    }
    out
}

/// Shortest path of atoms from `from` to `to` avoiding the direct edge
/// (`from`, `to`), by BFS over ascending neighbours (deterministic). `None`
/// when disconnected without that edge.
fn shortest_path_avoiding(adj: &[Vec<usize>], from: usize, to: usize) -> Option<Vec<usize>> {
    let n = adj.len();
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut queue = std::collections::VecDeque::new();
    parent[from] = Some(from);
    queue.push_back(from);
    while let Some(u) = queue.pop_front() {
        if u == to {
            break;
        }
        for &v in &adj[u] {
            if (u == from && v == to) || (u == to && v == from) {
                continue;
            }
            if parent[v].is_none() {
                parent[v] = Some(u);
                queue.push_back(v);
            }
        }
    }
    parent[to]?;
    let mut path = vec![to];
    while *path.last().expect("path is non-empty") != from {
        let u = *path.last().expect("path is non-empty");
        path.push(parent[u].expect("visited atoms have parents"));
    }
    path.reverse();
    Some(path)
}

/// Whether `a` is lexicographically smaller than `b` as assignment vectors
/// over `k` elements with element 0 most significant (bit `j` of the mask
/// is element `j`'s value).
fn mask_lex_less(a: usize, b: usize, k: usize) -> bool {
    for j in 0..k {
        let (x, y) = ((a >> j) & 1, (b >> j) & 1);
        if x != y {
            return x < y;
        }
    }
    false
}

/// Bond-indexed adjacency of `graph`: per atom `(neighbour, bond index)`.
fn adjacency_with_bonds(graph: &MolGraph) -> Vec<Vec<(usize, usize)>> {
    let n = graph.atoms().len();
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (idx, (a, b, _)) in graph.bonds().iter().enumerate() {
        adj[*a].push((*b, idx));
        adj[*b].push((*a, idx));
    }
    adj
}

/// Whether `from` reaches `to` without using bond `skip` (breadth-first
/// search; connectivity only, so the answer never depends on neighbour
/// visit order).
fn connected_avoiding(
    adj: &[Vec<(usize, usize)>],
    from: usize,
    to: usize,
    skip: usize,
) -> bool {
    if from == to {
        return true;
    }
    let mut seen = vec![false; adj.len()];
    let mut queue = std::collections::VecDeque::new();
    seen[from] = true;
    queue.push_back(from);
    while let Some(u) = queue.pop_front() {
        for &(v, b) in &adj[u] {
            if b == skip || seen[v] {
                continue;
            }
            if v == to {
                return true;
            }
            seen[v] = true;
            queue.push_back(v);
        }
    }
    false
}

/// Ring-bond flags, one per bond of [`MolGraph::bonds`]: bond `i` lies on
/// some cycle iff its ends stay connected without it. A pure graph
/// invariant — independent of atom numbering, unlike one arbitrarily chosen
/// shortest ring.
fn ring_bond_flags(graph: &MolGraph, adj: &[Vec<(usize, usize)>]) -> Vec<bool> {
    graph
        .bonds()
        .iter()
        .enumerate()
        .map(|(idx, (a, b, _))| connected_avoiding(adj, *a, *b, idx))
        .collect()
}

/// Ring-system component per atom: connectivity through ring bonds only.
/// Two atoms share a component iff some chain of cycle-lying bonds joins
/// them; fused rings are one system, disjoint rings are not.
fn ring_components(
    n: usize,
    bonds: &[(usize, usize, u8)],
    ring_bond: &[bool],
) -> Vec<usize> {
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for ((a, b, _), is_ring) in bonds.iter().zip(ring_bond.iter()) {
        if !is_ring {
            continue;
        }
        let (ra, rb) = (find(&mut parent, *a), find(&mut parent, *b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    for i in 0..n {
        let r = find(&mut parent, i);
        parent[i] = r;
    }
    parent
}

/// Whether atom `v` is a lone-pair donor for the conjugation check: N, O
/// or S carrying only single bonds (any hydrogen count).
fn is_lone_pair_donor(views: &[AtomView], v: usize) -> bool {
    matches!(views[v].element, NITROGEN | OXYGEN | SULFUR)
        && views[v].neighbours.iter().all(|&(_, o)| o == 1)
}

/// Whether the large-ring double bond `(a, b)` is isolated: neither end has
/// a ring neighbour (other than its partner, in the same ring system) that
/// carries a double bond or is a lone-pair donor. Inspects every ring
/// through the bond at once, so the answer never depends on which shortest
/// ring a search happens to find first.
fn large_ring_double_bond_isolated(
    views: &[AtomView],
    bond_index: &BTreeMap<(usize, usize), usize>,
    ring_bond: &[bool],
    ring_comp: &[usize],
    a: usize,
    b: usize,
) -> bool {
    for (end, partner) in [(a, b), (b, a)] {
        for &(nbr, _) in &views[end].neighbours {
            if nbr == partner {
                continue;
            }
            let key = if end < nbr { (end, nbr) } else { (nbr, end) };
            let Some(&idx) = bond_index.get(&key) else {
                continue;
            };
            if !ring_bond[idx] || ring_comp[nbr] != ring_comp[end] {
                continue;
            }
            if views[nbr].neighbours.iter().any(|&(_, o)| o == 2)
                || is_lone_pair_donor(views, nbr)
            {
                return false;
            }
        }
    }
    true
}

/// Whether atom `v` lies on a ring of 3 or 4 atoms: some neighbour pair of
/// `v` is directly bonded (a triangle with `v`) or shares another neighbour
/// besides `v` (a four-ring through `v`).
fn in_small_ring(adj: &[Vec<usize>], neighbours: &[usize], v: usize) -> bool {
    for (i, &x) in neighbours.iter().enumerate() {
        for &y in &neighbours[i + 1..] {
            if x == y {
                continue;
            }
            // A direct `x - y` bond closes a triangle with `v`.
            if adj[x].contains(&y) {
                return true;
            }
            // A shared neighbour `w` besides `v` closes a four-ring.
            if adj[x].iter().any(|&w| w != v && adj[w].contains(&y)) {
                return true;
            }
        }
    }
    false
}

/// Potential elements plus unmodelled kinds of `graph`.
///
/// Element order: tetrahedral centres by atom index, then double bonds by
/// `(a, b)`. `unsupported` is sorted and deduplicated.
fn potential_elements(graph: &MolGraph) -> (Vec<StereoElement>, Vec<String>) {
    let atom_views = views(graph);
    let n = atom_views.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (a, b, _) in graph.bonds() {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    for list in adj.iter_mut() {
        list.sort_unstable();
    }

    // Tetrahedral carbons: coordination four, all single, at most one H.
    let mut tetra: Vec<StereoElement> = Vec::new();
    for (i, view) in atom_views.iter().enumerate() {
        if view.element != CARBON || view.hydrogens > 1 {
            continue;
        }
        if view.neighbours.len() + usize::from(view.hydrogens) != 4 {
            continue;
        }
        if view.neighbours.iter().any(|&(_, o)| o != 1) {
            continue;
        }
        let mut heavy: Vec<usize> = view.neighbours.iter().map(|&(v, _)| v).collect();
        heavy.sort_unstable();
        let mut ligands: Vec<Ligand> = heavy.into_iter().map(Ligand::Atom).collect();
        if view.hydrogens == 1 {
            ligands.push(Ligand::Hydrogen);
        }
        tetra.push(StereoElement::Tetrahedral { atom: i, ligands });
    }

    // Double bonds: C=C, C=N, N=N with qualifying ends, outside small
    // rings and, in large rings, only when isolated (numbering-free).
    let mut bonds: Vec<StereoElement> = Vec::new();
    let mut conjugated_large_ring = false;
    let adj_bonds = adjacency_with_bonds(graph);
    let ring_bond = ring_bond_flags(graph, &adj_bonds);
    let ring_comp = ring_components(n, graph.bonds(), &ring_bond);
    let mut bond_index: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (idx, (a, b, _)) in graph.bonds().iter().enumerate() {
        bond_index.insert((*a.min(b), *a.max(b)), idx);
    }
    // Ring degree per atom (incident ring bonds): three or more marks a
    // bicyclic bridgehead candidate for the nitrogen check.
    let mut ring_degree = vec![0usize; n];
    for ((a, b, _), is_ring) in graph.bonds().iter().zip(ring_bond.iter()) {
        if *is_ring {
            ring_degree[*a] += 1;
            ring_degree[*b] += 1;
        }
    }
    for (a, b, order) in graph.bonds() {
        if *order != 2 {
            continue;
        }
        let (ea, eb) = (atom_views[*a].element, atom_views[*b].element);
        if !matches!(ea, CARBON | NITROGEN) || !matches!(eb, CARBON | NITROGEN) {
            continue;
        }
        // Smallest ring through the bond (its size is a graph invariant);
        // below 8 atoms it is excluded. At 8 or more it counts only when
        // isolated in every ring through it — otherwise the bond is
        // excluded and flagged, without ever naming one chosen ring.
        if let Some(path) = shortest_path_avoiding(&adj, *a, *b) {
            let size = path.len();
            if size < 8 {
                continue;
            }
            if !large_ring_double_bond_isolated(
                &atom_views,
                &bond_index,
                &ring_bond,
                &ring_comp,
                *a,
                *b,
            ) {
                conjugated_large_ring = true;
                continue;
            }
        }
        let Some(ref_a) = double_bond_end(&atom_views, *a, *b) else {
            continue;
        };
        let Some(ref_b) = double_bond_end(&atom_views, *b, *a) else {
            continue;
        };
        bonds.push(StereoElement::DoubleBond {
            a: *a,
            b: *b,
            ref_a,
            ref_b,
        });
    }
    bonds.sort_by_key(|e| match e {
        StereoElement::DoubleBond { a, b, .. } => (*a, *b),
        StereoElement::Tetrahedral { .. } => unreachable!("bond list holds bonds only"),
    });

    // Unmodelled kinds present in the molecule. Every condition is
    // conservative: flagging uncertainty is enough to keep
    // `molecule_wide_exact` honest, and every condition is a pure graph
    // invariant (never dependent on atom numbering).
    let mut unsupported: Vec<String> = Vec::new();
    if atom_views.iter().any(|v| {
        matches!(v.element, CARBON | NITROGEN)
            && v.neighbours.iter().filter(|&&(_, o)| o == 2).count() >= 2
    }) {
        unsupported.push("axial_cumulene".to_string());
    }
    if conjugated_large_ring {
        unsupported.push("conjugated_large_ring".to_string());
    }
    // Constrained pyramidal nitrogen: three single bonds (heavy neighbours
    // plus hydrogens) in a ring of 3 or 4 atoms, or at a bicyclic
    // bridgehead (three or more ring bonds). RDKit counts such nitrogens
    // (e.g. `CN1OC1`, `CN1C(C)C1`); this version never models them.
    if atom_views.iter().enumerate().any(|(v, view)| {
        view.element == NITROGEN
            && view.neighbours.iter().all(|&(_, o)| o == 1)
            && view.neighbours.len() + usize::from(view.hydrogens) == 3
            && (in_small_ring(
                &adj,
                &view.neighbours.iter().map(|&(w, _)| w).collect::<Vec<_>>(),
                v,
            ) || ring_degree[v] >= 3)
    }) {
        unsupported.push("constrained_nitrogen_center".to_string());
    }
    // Candidate atropisomeric axis: a single bond between two sp2 ring atoms
    // (each carrying a double bond) in different ring systems, with at
    // least three of the four ortho positions carrying a heavy substituent
    // (a ring neighbour with a third heavy neighbour).
    let mut atropisomer_axis = false;
    for ((a, b, order), is_ring) in graph.bonds().iter().zip(ring_bond.iter()) {
        if *order != 1 || *is_ring || ring_comp[*a] == ring_comp[*b] {
            continue;
        }
        let sp2 = |v: usize| {
            ring_degree[v] >= 1 && atom_views[v].neighbours.iter().any(|&(_, o)| o == 2)
        };
        if !sp2(*a) || !sp2(*b) {
            continue;
        }
        let mut substituted = 0usize;
        for (end, partner) in [(*a, *b), (*b, *a)] {
            for &(nbr, _) in &atom_views[end].neighbours {
                if nbr == partner {
                    continue;
                }
                let key = if end < nbr { (end, nbr) } else { (nbr, end) };
                if !bond_index.get(&key).is_some_and(|&idx| ring_bond[idx]) {
                    continue;
                }
                // Ortho ring neighbour: substituted when it carries a third
                // heavy neighbour (a fused third ring bond counts too —
                // conservatively bulky).
                if atom_views[nbr].neighbours.len() >= 3 {
                    substituted += 1;
                }
            }
        }
        if substituted >= 3 {
            atropisomer_axis = true;
            break;
        }
    }
    if atropisomer_axis {
        unsupported.push("atropisomer_axis_possible".to_string());
    }
    // Unmodelled P/S coordination: any phosphorus or sulfur with three or
    // more heavy neighbours (five-coordinate phosphorus and six-coordinate
    // all-single-bond sulfur are silent under narrower conditions).
    if atom_views
        .iter()
        .any(|v| v.element == PHOSPHORUS && v.neighbours.len() >= 3)
    {
        unsupported.push("phosphorus_center".to_string());
    }
    if atom_views
        .iter()
        .any(|v| v.element == SULFUR && v.neighbours.len() >= 3)
    {
        unsupported.push("sulfur_center".to_string());
    }
    unsupported.sort();
    unsupported.dedup();

    tetra.extend(bonds);
    (tetra, unsupported)
}

/// Reference ligand of double-bond end `end` (partner `partner`), or `None`
/// when the end does not qualify: it must carry no other multiple bond, and
/// hold two single heavy substituents (carbon, no H), one heavy plus one H
/// (carbon), or one heavy / one H plus the lone pair (nitrogen).
fn double_bond_end(views: &[AtomView], end: usize, partner: usize) -> Option<Ligand> {
    let view = &views[end];
    let mut heavy: Vec<usize> = Vec::new();
    for &(nbr, order) in &view.neighbours {
        if nbr == partner {
            continue;
        }
        if order != 1 {
            // Another multiple bond (cumulene/allene): excluded.
            return None;
        }
        heavy.push(nbr);
    }
    heavy.sort_unstable();
    match (view.element, view.hydrogens, heavy.len()) {
        (CARBON, 0, 2) => Some(Ligand::Atom(heavy[0])),
        (CARBON, 1, 1) => Some(Ligand::Atom(heavy[0])),
        (NITROGEN, 0, 1) => Some(Ligand::Atom(heavy[0])),
        (NITROGEN, 1, 0) => Some(Ligand::Hydrogen),
        _ => None,
    }
}

/// Map one ligand through an atom relabeling (`Hydrogen`/`LonePair` are
/// fixed points).
fn map_ligand(ligand: &Ligand, sigma: &[usize]) -> Ligand {
    match ligand {
        Ligand::Atom(i) => Ligand::Atom(sigma[*i]),
        Ligand::Hydrogen => Ligand::Hydrogen,
        Ligand::LonePair => Ligand::LonePair,
    }
}

/// Parity (0 even, 1 odd) of the permutation taking the distinct-element list
/// `from` into the order of `to`. `None` when the lists hold different
/// elements (not a permutation at all).
fn permutation_parity(from: &[Ligand], to: &[Ligand]) -> Option<u8> {
    if from.len() != to.len() {
        return None;
    }
    let mut perm: Vec<usize> = Vec::with_capacity(from.len());
    for item in from {
        let pos = to.iter().position(|t| t == item)?;
        perm.push(pos);
    }
    let mut inversions = 0usize;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    Some((inversions % 2) as u8)
}

/// How one automorphism moves every potential element: the image element
/// index plus whether its value flips.
struct ElementAction {
    /// `image[j]`: potential index of the image of element `j`.
    image: Vec<usize>,
    /// `flip[j]`: the value of element `j` flips when moved.
    flip: Vec<bool>,
}

/// Action of the atom relabeling `sigma` on the potential elements, or
/// `None` when an image falls outside the potential set (a conservative
/// failure: the caller reports `unresolved`, never a guess).
fn element_action(
    potential: &[StereoElement],
    tetra_of_atom: &[Option<usize>],
    bond_of_pair: &BTreeMap<(usize, usize), usize>,
    sigma: &[usize],
) -> Option<ElementAction> {
    let mut image = vec![0usize; potential.len()];
    let mut flip = vec![false; potential.len()];
    for (j, element) in potential.iter().enumerate() {
        match element {
            StereoElement::Tetrahedral { atom, ligands } => {
                let target = sigma[*atom];
                let tj = tetra_of_atom[target]?;
                let StereoElement::Tetrahedral {
                    ligands: target_ligands,
                    ..
                } = &potential[tj]
                else {
                    return None;
                };
                let moved: Vec<Ligand> = ligands.iter().map(|l| map_ligand(l, sigma)).collect();
                let parity = permutation_parity(&moved, target_ligands.as_slice())?;
                image[j] = tj;
                flip[j] = parity == 1;
            }
            StereoElement::DoubleBond { a, b, ref_a, ref_b } => {
                let (sa, sb) = (sigma[*a], sigma[*b]);
                let key = if sa < sb { (sa, sb) } else { (sb, sa) };
                let tj = bond_of_pair.get(&key).copied()?;
                let StereoElement::DoubleBond {
                    a: ta,
                    ref_a: tref_a,
                    ref_b: tref_b,
                    ..
                } = &potential[tj]
                else {
                    return None;
                };
                // Which image end corresponds to `a`: the side holding `sa`.
                let a_side_is_first = sa == *ta;
                let mapped_a = map_ligand(ref_a, sigma);
                let mapped_b = map_ligand(ref_b, sigma);
                let flip_a = mapped_a != *if a_side_is_first { tref_a } else { tref_b };
                let flip_b = mapped_b != *if a_side_is_first { tref_b } else { tref_a };
                // An end swap by itself changes nothing; the value flips iff
                // exactly one end flipped.
                image[j] = tj;
                flip[j] = flip_a != flip_b;
            }
        }
    }
    // The images must permute the elements (an automorphism maps the
    // potential set onto itself bijectively).
    let mut seen = vec![false; potential.len()];
    for &tj in &image {
        if seen[tj] {
            return None;
        }
        seen[tj] = true;
    }
    Some(ElementAction { image, flip })
}

/// Image of the assignment mask `mask` under one element action: bit
/// `image[j]` of the result is bit `j` of `mask`, flipped when `flip[j]`.
fn apply_action(action: &ElementAction, mask: usize) -> usize {
    let mut out = 0usize;
    for (j, (&tj, &f)) in action.image.iter().zip(action.flip.iter()).enumerate() {
        let mut bit = (mask >> j) & 1;
        if f {
            bit ^= 1;
        }
        out |= bit << tj;
    }
    out
}

/// Union-find root with path halving.
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Analyse one candidate graph: potential elements, exact orbits under the
/// automorphism group, stereogenicity and canonical isomers.
///
/// Deterministic and pure host. Caps in `limits` make the report
/// `unresolved` instead of a guess; `max_elements` is additionally clamped
/// to an absolute cap of 16 inside this function, and every shift and every
/// allocation size is checked, so an excessive caller limit reports
/// `unresolved` instead of panicking or over-allocating.
pub fn perceive(graph: &MolGraph, limits: &StereoLimits) -> StereoReport {
    let (potential, unsupported) = potential_elements(graph);
    let k = potential.len();
    // The caller's limit never authorises more than the absolute cap: clamp
    // first, so no path below shifts or allocates by an excessive `k`.
    let max_elements = limits.max_elements.min(ABSOLUTE_MAX_ELEMENTS);
    if k > max_elements {
        return unresolved_report(potential, unsupported, "too_many_elements");
    }
    // Index the potential set for the action computation.
    let n = graph.atoms().len();
    let mut tetra_of_atom: Vec<Option<usize>> = vec![None; n];
    let mut bond_of_pair: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (j, element) in potential.iter().enumerate() {
        match element {
            StereoElement::Tetrahedral { atom, .. } => tetra_of_atom[*atom] = Some(j),
            StereoElement::DoubleBond { a, b, .. } => {
                bond_of_pair.insert((*a, *b), j);
            }
        }
    }
    // Every automorphism of the constitutional graph.
    let cap = limits.max_automorphisms.saturating_add(1);
    let enumeration = enumerate_isomorphisms(graph, graph, limits.work_limit, cap);
    if enumeration.over_budget {
        return unresolved_report(potential, unsupported, "work_limit_exceeded");
    }
    if enumeration.truncated || enumeration.maps.len() > limits.max_automorphisms {
        return unresolved_report(potential, unsupported, "too_many_automorphisms");
    }
    let mut actions: Vec<ElementAction> = Vec::with_capacity(enumeration.maps.len());
    for sigma in &enumeration.maps {
        match element_action(&potential, &tetra_of_atom, &bond_of_pair, sigma) {
            Some(action) => actions.push(action),
            None => return unresolved_report(potential, unsupported, "automorphism_mapping"),
        }
    }
    // Orbits of all 2^k assignments under the actions. Every size is
    // checked: `k` is at most the absolute cap (16), but the shift and the
    // two `usize` tables are still fallible by construction — refusal is
    // `unresolved`, never a panic or a hug allocation.
    let total = match 1usize.checked_shl(k as u32) {
        Some(total) => total,
        None => return unresolved_report(potential, unsupported, "too_many_elements"),
    };
    if total
        .checked_mul(std::mem::size_of::<usize>())
        .is_none()
    {
        return unresolved_report(potential, unsupported, "too_many_elements");
    }
    let mut parent: Vec<usize> = (0..total).collect();
    for action in &actions {
        for mask in 0..total {
            let image = apply_action(action, mask);
            let a = find_root(&mut parent, mask);
            let b = find_root(&mut parent, image);
            if a != b {
                parent[a] = b;
            }
        }
    }
    let mut root_of = vec![0usize; total];
    for (mask, slot) in root_of.iter_mut().enumerate() {
        *slot = find_root(&mut parent, mask);
    }
    // Canonical representative of each orbit: its lexicographically
    // smallest assignment vector (first element most significant).
    let mut canonical_of_root: BTreeMap<usize, usize> = BTreeMap::new();
    for (mask, &root) in root_of.iter().enumerate() {
        canonical_of_root
            .entry(root)
            .and_modify(|c| {
                if mask_lex_less(mask, *c, k) {
                    *c = mask;
                }
            })
            .or_insert(mask);
    }
    // An element is stereogenic iff flipping it alone changes the orbit for
    // at least one assignment.
    let mut is_stereogenic = vec![false; k];
    for j in 0..k {
        for mask in 0..total {
            if root_of[mask] != root_of[mask ^ (1usize << j)] {
                is_stereogenic[j] = true;
                break;
            }
        }
    }
    let mut elements: Vec<IndexedElement> = Vec::new();
    for (j, element) in potential.iter().enumerate() {
        if is_stereogenic[j] {
            elements.push(IndexedElement {
                potential_index: j,
                element: element.clone(),
            });
        }
    }
    let not_stereogenic = k - elements.len();
    // Canonical orbits projected onto the stereogenic elements.
    if limits.max_expanded > 0 {
        let mut projected: Vec<Vec<u8>> = canonical_of_root
            .values()
            .map(|&mask| {
                elements
                    .iter()
                    .map(|e| ((mask >> e.potential_index) & 1) as u8)
                    .collect()
            })
            .collect();
        projected.sort();
        projected.dedup();
        let truncated = projected.len() > limits.max_expanded;
        projected.truncate(limits.max_expanded);
        let distinct = canonical_of_root.len() as u64;
        return StereoReport {
            potential,
            elements,
            not_stereogenic,
            unsupported,
            raw_assignments: Some(total as u64),
            distinct: Some(distinct),
            resolution: Resolution::Resolved,
            automorphisms: Some(enumeration.maps.len() as u64),
            isomers_truncated: truncated,
            isomers: projected,
        };
    }
    StereoReport {
        potential,
        elements,
        not_stereogenic,
        unsupported,
        raw_assignments: Some(total as u64),
        distinct: Some(canonical_of_root.len() as u64),
        resolution: Resolution::Resolved,
        automorphisms: Some(enumeration.maps.len() as u64),
        isomers: Vec::new(),
        isomers_truncated: false,
    }
}

/// Whether two assignments over the potential elements are equivalent under
/// the graph's automorphisms.
///
/// `a` and `b` hold one value per potential element in assignment order.
/// `None` means undecided: a length mismatch, a value outside `{0, 1}`, or
/// the same caps as [`perceive`] were hit (too many elements, too many
/// automorphisms, the work limit, or an unmappable automorphism).
pub fn equivalent(graph: &MolGraph, a: &[u8], b: &[u8], limits: &StereoLimits) -> Option<bool> {
    let (potential, _) = potential_elements(graph);
    let k = potential.len();
    if a.len() != k || b.len() != k {
        return None;
    }
    if a.iter().any(|&v| v > 1) || b.iter().any(|&v| v > 1) {
        return None;
    }
    // The same absolute cap as [`perceive`]: a caller limit above it never
    // authorises the shifts and tables below.
    if k > limits.max_elements.min(ABSOLUTE_MAX_ELEMENTS) {
        return None;
    }
    let n = graph.atoms().len();
    let mut tetra_of_atom: Vec<Option<usize>> = vec![None; n];
    let mut bond_of_pair: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (j, element) in potential.iter().enumerate() {
        match element {
            StereoElement::Tetrahedral { atom, .. } => tetra_of_atom[*atom] = Some(j),
            StereoElement::DoubleBond { a, b, .. } => {
                bond_of_pair.insert((*a, *b), j);
            }
        }
    }
    let cap = limits.max_automorphisms.saturating_add(1);
    let enumeration = enumerate_isomorphisms(graph, graph, limits.work_limit, cap);
    if enumeration.over_budget
        || enumeration.truncated
        || enumeration.maps.len() > limits.max_automorphisms
    {
        return None;
    }
    let mut mask_a = 0usize;
    let mut mask_b = 0usize;
    for (j, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        mask_a |= usize::from(x) << j;
        mask_b |= usize::from(y) << j;
    }
    for sigma in &enumeration.maps {
        let action = element_action(&potential, &tetra_of_atom, &bond_of_pair, sigma)?;
        if apply_action(&action, mask_a) == mask_b {
            return Some(true);
        }
    }
    Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2-butanol: `C(H3)-C(H1)-C(H2)-C(H3)` with `O(H1)` on atom 1.
    fn butan_2_ol() -> MolGraph {
        MolGraph::new(
            vec![4, 2, 3, 4, 9],
            vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (1, 4, 1)],
        )
        .unwrap()
    }

    #[test]
    fn butanol_has_one_centre_and_two_isomers() {
        let report = perceive(&butan_2_ol(), &StereoLimits::default());
        assert_eq!(report.potential.len(), 1);
        assert_eq!(report.elements.len(), 1);
        assert_eq!(report.distinct, Some(2));
        assert!(report.resolution.is_resolved());
        assert!(report.molecule_wide_exact());
    }

    #[test]
    fn enantiomers_differ_singletons_agree() {
        let graph = butan_2_ol();
        let limits = StereoLimits::default();
        assert_eq!(equivalent(&graph, &[0], &[0], &limits), Some(true));
        assert_eq!(equivalent(&graph, &[0], &[1], &limits), Some(false));
    }
}
