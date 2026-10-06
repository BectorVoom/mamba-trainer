//! Functional-group extraction on typed graphs (`functional-groups-ertl-v1`)
//! with a documented aromaticity perception (`aromatic-ring-v2`).
//!
//! Ertl's algorithm (P. Ertl, "An algorithm to identify functional groups in
//! organic molecules", J. Cheminform. 9:36, 2017) as implemented in RDKit's
//! `Contrib/IFG/ifg.py`: mark every heteroatom, mark the four carbon kinds
//! (non-aromatic double/triple to hetero, non-aromatic C=C/C#C, acetal
//! carbons, oxirane/aziridine/thiirane atoms) and merge bonded marked atoms.
//! A functional group here is the induced typed subgraph on its marked atoms
//! with the parent's atom types (hydrogen counts included); the unmarked
//! carbon environment that Ertl's type strings add is not included.
//!
//! Where this text and `ifg.py` differ, `ifg.py` wins. Known difference: the
//! carbon of a C=O/C=N whose ring atom is aromatic is not marked (the SMARTS
//! `A=,#[!#6]` needs an aliphatic first atom), so an aromatic carbonyl such
//! as 2-pyridone or caffeine contributes its O alone. The acetal SMARTS
//! `[CX4](-[O,N,S])-[O,N,S]` is matched literally: a tetracoordinate carbon
//! (hydrogens plus heavy neighbours equals four) with two single-bonded O/N/S
//! neighbours, without checking those neighbours' other bonds (RDKit matches
//! e.g. `CC(N=C)OC`).
//!
//! SMARTS qualifier audit (MC17; `ifg.py` `PATT_*` vs the rules below):
//!
//! * `A=,#[!#6]` (rule 2a): `A` = aliphatic — enforced by requiring the
//!   carbon non-aromatic; `=`/`#` — enforced by accepting only orders 2/3;
//!   `[!#6]` — the partner is a heteroatom (already marked in rule 1, so
//!   marking the carbon is the whole effect).
//! * `C=,#C` (rule 2b): both `C` aliphatic — enforced on both ends; bond
//!   order 2/3 only.
//! * `[CX4](-[O,N,S])-[O,N,S]` (rule 2c): `C` aliphatic, `X4` total
//!   connections (heavy neighbours plus hydrogens) equals four, `-`
//!   explicit single bonds to the two neighbours, `[O,N,S]` aliphatic —
//!   all enforced; the neighbours' other bonds are unchecked, exactly as
//!   the SMARTS (which constrains only the two `-` bonds) does.
//! * `[O,N,S]1CC1` (rule 2d): atoms aliphatic, implicit bonds
//!   single-or-aromatic — since the atoms are aliphatic, every triangle bond
//!   must be single; all three bonds are checked for order 1.
//!
//! Aromaticity (`aromatic-ring-v2`) is an approximation of RDKit's model, not
//! a re-implementation. Rings of 5 and 6 atoms are enumerated as the smallest
//! cycle through each ring bond (bounded breadth-first search, depth 5). In
//! addition every **pair of rings sharing exactly one bond** is evaluated as
//! one circuit over the union of their atoms (the shared atoms count once),
//! for fused systems whose rings are aromatic only jointly. A circuit
//! (single ring or pair union) is aromatic when every atom contributes as
//! below and the electron count is 4n+2: (i) sp2 carbon or nitrogen with
//! exactly one double bond in this circuit or in another already-aromatic
//! circuit fused to it (one electron), (ii) a heteroatom with only single
//! bonds and a lone pair in the circuit (N with a hydrogen or three single
//! bonds, O, S; two electrons), (iii) a ring carbon with an exocyclic C=O/C=N
//! (zero electrons, e.g. 2-pyridone). Judgement iterates to a fixed point
//! for fused systems. Agreement with RDKit is measured by
//! `tools/ms2/completion_functional_groups_check.py`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::error::Result;

use super::chem;
use super::graph::MolGraph;

/// Version of the functional-group recipe in this module.
pub const FUNCTIONAL_GROUPS_VERSION: &str = "functional-groups-ertl-v1";
/// Version of the aromaticity perception in this module.
pub const AROMATICITY_VERSION: &str = "aromatic-ring-v2";

/// One functional group: ascending parent atom indices plus an
/// order-independent signature of the group's typed subgraph.
pub struct FunctionalGroup {
    /// Ascending parent atom indices of the group.
    pub atoms: Vec<usize>,
    /// Order-independent text of the group's typed subgraph, for statistics.
    pub signature: String,
}

/// One enumerated ring: cycle-order atoms and the bond indices of the cycle.
struct Ring {
    /// Atoms in cycle order.
    atoms: Vec<usize>,
    /// Bond indices (`graph.bonds()` positions) of the cycle edges.
    bonds: Vec<usize>,
    /// Atom set, for fusion tests and dedup.
    set: BTreeSet<usize>,
}

/// Build the adjacency with bond indices: per atom `(neighbour, order, bond)`.
fn adjacency(graph: &MolGraph) -> Vec<Vec<(usize, u8, usize)>> {
    let n = graph.atoms().len();
    let mut adj: Vec<Vec<(usize, u8, usize)>> = vec![Vec::new(); n];
    for (idx, (a, b, order)) in graph.bonds().iter().enumerate() {
        adj[*a].push((*b, *order, idx));
        adj[*b].push((*a, *order, idx));
    }
    adj
}

/// Element index of atom `v` (`ELEMENTS` position).
fn element_of(graph: &MolGraph, v: usize) -> usize {
    chem::atom_type(graph.atoms()[v])
        .map(|t| t.element)
        .unwrap_or(0)
}

/// Hydrogen count of atom `v`.
fn hydrogens_of(graph: &MolGraph, v: usize) -> u8 {
    chem::atom_type(graph.atoms()[v])
        .map(|t| t.hydrogens)
        .unwrap_or(0)
}

/// Enumerate the rings of 5 and 6 atoms: the smallest cycle through each
/// ring bond, bounded to depth 5.
fn enumerate_rings(graph: &MolGraph, adj: &[Vec<(usize, u8, usize)>]) -> Vec<Ring> {
    let n = graph.atoms().len();
    let mut rings: Vec<Ring> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    for (skipped, (sa, sb, _)) in graph.bonds().iter().enumerate() {
        let (start, goal) = (*sa, *sb);
        // Bounded BFS from `start` to `goal` without the skipped bond.
        let mut dist: Vec<Option<usize>> = vec![None; n];
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        dist[start] = Some(0);
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            let d = dist[v].expect("visited");
            if d >= 5 {
                continue;
            }
            for (nbr, _, bidx) in &adj[v] {
                if *bidx == skipped {
                    continue;
                }
                if dist[*nbr].is_none() {
                    dist[*nbr] = Some(d + 1);
                    parent[*nbr] = Some((v, *bidx));
                    queue.push_back(*nbr);
                }
            }
        }
        let Some(path_len) = dist[goal] else {
            continue;
        };
        if !(4..=5).contains(&path_len) {
            continue;
        }
        // Reconstruct the path from `start` to `goal`.
        let mut path_nodes: Vec<usize> = vec![goal];
        let mut path_bonds: Vec<usize> = Vec::new();
        let mut cur = goal;
        while cur != start {
            let (prev, bidx) = parent[cur].expect("path connected");
            path_bonds.push(bidx);
            cur = prev;
            path_nodes.push(cur);
        }
        path_nodes.reverse();
        path_bonds.reverse();
        let cycle_len = path_len + 1;
        if cycle_len != 5 && cycle_len != 6 {
            continue;
        }
        let mut key = path_nodes.clone();
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        let mut set = BTreeSet::new();
        for v in &path_nodes {
            set.insert(*v);
        }
        let mut bonds = path_bonds;
        bonds.push(skipped);
        rings.push(Ring {
            atoms: path_nodes,
            bonds,
            set,
        });
    }
    rings
}

/// Pair circuits for fused systems: every pair of rings sharing exactly
/// one bond, evaluated as one circuit over the union of their atoms (the
/// shared atoms count once). For fused systems whose rings are aromatic
/// only jointly (neither bootstraps the other alone).
fn pair_circuits(rings: &[Ring]) -> Vec<Ring> {
    let mut out: Vec<Ring> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    for i in 0..rings.len() {
        for j in (i + 1)..rings.len() {
            let shared = rings[i]
                .bonds
                .iter()
                .filter(|b| rings[j].bonds.contains(b))
                .count();
            if shared != 1 {
                continue;
            }
            let set: BTreeSet<usize> =
                rings[i].set.union(&rings[j].set).copied().collect();
            let key: Vec<usize> = set.iter().copied().collect();
            if !seen.insert(key) {
                continue;
            }
            let mut bonds: Vec<usize> = rings[i]
                .bonds
                .iter()
                .chain(rings[j].bonds.iter())
                .copied()
                .collect();
            bonds.sort_unstable();
            bonds.dedup();
            let atoms: Vec<usize> = set.iter().copied().collect();
            out.push(Ring { atoms, bonds, set });
        }
    }
    out
}

/// Single rings plus pair circuits: returns the single-ring count (the
/// prefix `circuits[..singles]` holds the single rings) and all circuits.
/// Judgement runs to a fixed point over all of them, so pairs and singles
/// bootstrap each other.
fn enumerate_circuits(
    graph: &MolGraph,
    adj: &[Vec<(usize, u8, usize)>],
) -> (usize, Vec<Ring>) {
    let singles = enumerate_rings(graph, adj);
    let single_count = singles.len();
    let mut circuits = singles;
    let pairs = pair_circuits(&circuits);
    circuits.extend(pairs);
    (single_count, circuits)
}

/// The aromatic rings of the graph (atom lists in cycle order).
///
/// Separated from [`aromatic_atoms`] so the pattern source can add aromatic
/// rings as groups without recomputing the enumeration. Only single rings
/// are returned (pair unions have no cycle order); pairs still contribute
/// through [`aromatic_atoms`] and the fixed-point judgement.
pub fn aromatic_rings(graph: &MolGraph) -> Vec<Vec<usize>> {
    let adj = adjacency(graph);
    let (single_count, circuits) = enumerate_circuits(graph, &adj);
    judge_aromatic(graph, &adj, &circuits)
        .into_iter()
        .filter(|idx| *idx < single_count)
        .map(|idx| circuits[idx].atoms.clone())
        .collect()
}

/// Judge which enumerated rings are aromatic, to a fixed point.
///
/// Returns the indices into `rings` judged aromatic.
fn judge_aromatic(
    graph: &MolGraph,
    adj: &[Vec<(usize, u8, usize)>],
    rings: &[Ring],
) -> Vec<usize> {
    let mut aromatic = vec![false; rings.len()];
    // Bond index -> aromatic ring positions using it, for the fused rule.
    loop {
        let mut next = aromatic.clone();
        // Bond sets of already-aromatic rings for the fused double-bond rule.
        let mut aromatic_bond_sets: Vec<HashSet<usize>> = Vec::new();
        let mut aromatic_atom_sets: Vec<BTreeSet<usize>> = Vec::new();
        for (i, ring) in rings.iter().enumerate() {
            if aromatic[i] {
                aromatic_bond_sets.push(ring.bonds.iter().copied().collect());
                aromatic_atom_sets.push(ring.set.clone());
            }
        }
        for (i, ring) in rings.iter().enumerate() {
            if aromatic[i] {
                continue;
            }
            if ring_is_aromatic(graph, adj, ring, &aromatic_bond_sets, &aromatic_atom_sets) {
                next[i] = true;
            }
        }
        if next == aromatic {
            break;
        }
        aromatic = next;
    }
    aromatic
        .iter()
        .enumerate()
        .filter(|(_, a)| **a)
        .map(|(i, _)| i)
        .collect()
}

/// Whether one ring is aromatic under the current fixed-point state.
fn ring_is_aromatic(
    graph: &MolGraph,
    adj: &[Vec<(usize, u8, usize)>],
    ring: &Ring,
    aromatic_bond_sets: &[HashSet<usize>],
    aromatic_atom_sets: &[BTreeSet<usize>],
) -> bool {
    let in_ring: HashSet<usize> = ring.set.iter().copied().collect();
    let ring_bonds: HashSet<usize> = ring.bonds.iter().copied().collect();
    let mut electrons = 0u32;
    for v in &ring.atoms {
        let v = *v;
        let element = element_of(graph, v);
        let hydrogens = hydrogens_of(graph, v);
        let incident = &adj[v];
        let orders: Vec<u8> = incident.iter().map(|(_, o, _)| *o).collect();
        let all_single = orders.iter().all(|o| *o == 1);
        let degree = incident.len();
        // (iii) Ring carbon with an exocyclic C=O / C=N: zero electrons.
        if element == 0 {
            let mut exocyclic = false;
            for (nbr, order, _) in incident {
                if *order == 2 && !in_ring.contains(nbr) {
                    let e = element_of(graph, *nbr);
                    if e == 2 || e == 3 {
                        exocyclic = true;
                    }
                }
            }
            if exocyclic {
                continue;
            }
        }
        // (ii) Heteroatom with only single bonds and a lone pair.
        if all_single {
            if element == 2 && (hydrogens >= 1 || degree == 3) {
                electrons += 2;
                continue;
            }
            if element == 3 || element == 6 {
                electrons += 2;
                continue;
            }
        }
        // (i) sp2 carbon or nitrogen with exactly one double bond in this
        // ring or in an already-aromatic ring fused to it.
        if element == 0 || element == 2 {
            let doubles: Vec<usize> = incident
                .iter()
                .filter(|(_, o, _)| *o == 2)
                .map(|(_, _, b)| *b)
                .collect();
            let triples = incident.iter().filter(|(_, o, _)| *o == 3).count();
            if triples == 0 && doubles.len() == 1 {
                let b = doubles[0];
                let in_this = ring_bonds.contains(&b);
                let mut in_fused = false;
                if !in_this {
                    for (set, atoms) in aromatic_bond_sets.iter().zip(aromatic_atom_sets.iter()) {
                        if set.contains(&b) && atoms.intersection(&ring.set).next().is_some() {
                            in_fused = true;
                            break;
                        }
                    }
                }
                if in_this || in_fused {
                    electrons += 1;
                    continue;
                }
            }
        }
        return false;
    }
    electrons >= 2 && (electrons - 2).is_multiple_of(4)
}

/// Whether atom `v` is aromatic: part of a circuit (single ring or fused
/// pair) judged aromatic.
pub fn aromatic_atoms(graph: &MolGraph) -> Vec<bool> {
    let adj = adjacency(graph);
    let (_, circuits) = enumerate_circuits(graph, &adj);
    let aromatic = judge_aromatic(graph, &adj, &circuits);
    let in_aromatic: HashSet<usize> = aromatic
        .into_iter()
        .flat_map(|i| circuits[i].atoms.iter().copied())
        .collect();
    (0..graph.atoms().len())
        .map(|v| in_aromatic.contains(&v))
        .collect()
}

/// Whether bond `i` (`graph.bonds()[i]`) is aromatic: part of a circuit
/// (single ring or fused pair) judged aromatic.
pub fn aromatic_bonds(graph: &MolGraph) -> Vec<bool> {
    let adj = adjacency(graph);
    let (_, circuits) = enumerate_circuits(graph, &adj);
    let aromatic = judge_aromatic(graph, &adj, &circuits);
    let in_aromatic: HashSet<usize> = aromatic
        .into_iter()
        .flat_map(|i| circuits[i].bonds.iter().copied())
        .collect();
    (0..graph.bonds().len())
        .map(|i| in_aromatic.contains(&i))
        .collect()
}

/// Order-independent signature of a group's typed subgraph.
///
/// Sorted atom type ids plus sorted `(low type, high type, order)` bond
/// triples. Generic chemistry for statistics, never data rows.
fn group_signature(graph: &MolGraph, atoms: &[usize]) -> String {
    let mut types: Vec<u8> = atoms.iter().map(|a| graph.atoms()[*a]).collect();
    types.sort_unstable();
    let pos: HashMap<usize, usize> = atoms.iter().enumerate().map(|(i, a)| (*a, i)).collect();
    let mut triples: Vec<(u8, u8, u8)> = Vec::new();
    for (a, b, order) in graph.bonds() {
        if let (Some(_), Some(_)) = (pos.get(a), pos.get(b)) {
            let (ta, tb) = (graph.atoms()[*a], graph.atoms()[*b]);
            let (lo, hi) = if ta <= tb { (ta, tb) } else { (tb, ta) };
            triples.push((lo, hi, *order));
        }
    }
    triples.sort_unstable();
    let type_text: Vec<String> = types.iter().map(|t| t.to_string()).collect();
    let bond_text: Vec<String> = triples
        .iter()
        .map(|(a, b, o)| format!("{a}-{b}:{o}"))
        .collect();
    format!("types[{}]|bonds[{}]", type_text.join(","), bond_text.join(","))
}

/// The functional groups of the graph, ordered by smallest atom index.
///
/// Deterministic: marked atoms merge by connectivity; each group's atoms are
/// ascending and groups are ordered by their smallest atom.
pub fn functional_groups(graph: &MolGraph) -> Result<Vec<FunctionalGroup>> {
    let n = graph.atoms().len();
    let adj = adjacency(graph);
    let aromatic_atom = aromatic_atoms(graph);
    let aromatic_bond = aromatic_bonds(graph);
    let mut marked = vec![false; n];
    // Rule 1: every heteroatom (every atom that is not carbon).
    for (v, slot) in marked.iter_mut().enumerate() {
        if element_of(graph, v) != 0 {
            *slot = true;
        }
    }
    // Helper: bond index of the (a, b) pair.
    let mut bond_index: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (idx, (a, b, _)) in graph.bonds().iter().enumerate() {
        bond_index.insert((*a.min(b), *a.max(b)), idx);
    }
    // Rules 2(a)/2(b): non-aromatic double/triple bonds. The ifg.py SMARTS
    // need aliphatic (non-aromatic) atoms: `A=,#[!#6]` and `C=,#C`.
    for (idx, (a, b, order)) in graph.bonds().iter().enumerate() {
        if *order != 2 && *order != 3 {
            continue;
        }
        if aromatic_bond[idx] {
            continue;
        }
        let (ea, eb) = (element_of(graph, *a), element_of(graph, *b));
        let (aa, ab) = (aromatic_atom[*a], aromatic_atom[*b]);
        // 2(a): aliphatic carbon double/triple-bonded to a heteroatom.
        if ea == 0 && eb != 0 && !aa {
            marked[*a] = true;
        }
        if eb == 0 && ea != 0 && !ab {
            marked[*b] = true;
        }
        // 2(b): aliphatic carbon-carbon double/triple bond.
        if ea == 0 && eb == 0 && !aa && !ab {
            marked[*a] = true;
            marked[*b] = true;
        }
    }
    // Rule 2(c): acetal carbons. Literal `[CX4](-[O,N,S])-[O,N,S]`: carbon
    // with hydrogens plus heavy neighbours equal to four, single-bonded to
    // two or more aliphatic O/N/S. The neighbours' other bonds are not
    // checked (ifg.py matches e.g. `CC(N=C)OC`).
    for (v, nbrs) in adj.iter().enumerate() {
        if element_of(graph, v) != 0 || aromatic_atom[v] {
            continue;
        }
        let degree = nbrs.len();
        let total = usize::from(hydrogens_of(graph, v)) + degree;
        if total != 4 {
            continue;
        }
        let mut count = 0usize;
        for (nbr, order, _) in nbrs {
            if *order != 1 {
                continue;
            }
            let e = element_of(graph, *nbr);
            if (e == 2 || e == 3 || e == 6) && !aromatic_atom[*nbr] {
                count += 1;
            }
        }
        if count >= 2 {
            marked[v] = true;
        }
    }
    // Rule 2(d): oxirane, aziridine and thiirane rings. The ifg.py SMARTS
    // `[O,N,S]1CC1` uses implicit single-or-aromatic bonds between aliphatic
    // atoms; since every matched atom must be aliphatic (non-aromatic),
    // aromatic bonds are impossible and every triangle bond must be single.
    // An unsaturated triangle such as 2H-azirine (`C1=NC1`) does not match:
    // its C=N pair is marked by rule 2(a) instead.
    let mut triangle_seen: HashSet<Vec<usize>> = HashSet::new();
    for (v, nbrs) in adj.iter().enumerate() {
        let ids: Vec<usize> = nbrs.iter().map(|(w, _, _)| *w).collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = (ids[i], ids[j]);
                if bond_index.contains_key(&(a.min(b), a.max(b))) {
                    let mut tri = vec![v, a, b];
                    tri.sort_unstable();
                    if !triangle_seen.insert(tri.clone()) {
                        continue;
                    }
                    let elements: Vec<usize> =
                        tri.iter().map(|w| element_of(graph, *w)).collect();
                    let hetero = elements
                        .iter()
                        .filter(|e| **e == 2 || **e == 3 || **e == 6)
                        .count();
                    let carbons = elements.iter().filter(|e| **e == 0).count();
                    if hetero != 1 || carbons != 2 {
                        continue;
                    }
                    // All three atoms aliphatic (the SMARTS atoms are
                    // uppercase) and all three triangle bonds single.
                    if tri.iter().any(|w| aromatic_atom[*w]) {
                        continue;
                    }
                    let mut all_single = true;
                    for (x, y) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])] {
                        let key = (x.min(y), x.max(y));
                        let Some(&idx) = bond_index.get(&key) else {
                            all_single = false;
                            break;
                        };
                        if graph.bonds()[idx].2 != 1 {
                            all_single = false;
                            break;
                        }
                    }
                    if all_single {
                        for w in &tri {
                            marked[*w] = true;
                        }
                    }
                }
            }
        }
    }
    // Rule 3: merge bonded marked atoms (connected components).
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (a, b, _) in graph.bonds() {
        if marked[*a] && marked[*b] {
            let ra = find(&mut parent, *a);
            let rb = find(&mut parent, *b);
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (v, is_marked) in marked.iter().enumerate() {
        if *is_marked {
            let r = find(&mut parent, v);
            by_root.entry(r).or_default().push(v);
        }
    }
    let mut groups: Vec<FunctionalGroup> = Vec::new();
    for atoms in by_root.into_values() {
        let signature = group_signature(graph, &atoms);
        groups.push(FunctionalGroup { atoms, signature });
    }
    groups.sort_by_key(|g| g.atoms[0]);
    Ok(groups)
}

// ---------------------------------------------------------------------------
// NOTE (main merge): this module hosts two recipes. The Ertl recipe above
// (`functional-groups-ertl-v1`) is upstream; `functional_groups_v4` below is
// the local `ms2-fg-v4` pattern vocabulary, renamed from `functional_groups`
// so both callers compile. Unify or split the module as a follow-up.
// ---------------------------------------------------------------------------
// Functional-group vocabulary `ms2-fg-v4`: a kekulé-invariant evaluation
// level for predicted molecular graphs.
//
// A functional group is a small pattern over heavy atoms with their element,
// parent hydrogen count and kekulized bond orders. Patterns are data (the
// [`PATTERNS`] table) matched by one generic backtracking matcher, plus a
// small set of structural rules (delocalised bonds, five-rings, carbonyl
// remaining-substituent checks) shared with the RDKit reference. Each
// pattern names required atoms (element; hydrogen-count constraint: exact,
// "H1 or H2", or "any"), required bonds with orders, and **exclusions** on
// named pattern atoms. Every "double bond" below means a **fixed** double
// bond (see the delocalised-bond rule); triple bonds are never delocalised.
//
// # The 28 types (`[Hn]` = parent hydrogen count)
//
// Ids were assigned once and are never reused: v1 ids 1–26 keep their
// meaning (with corrected chemistry), 27 and 28 are new.
//
// | id | name | pattern in words | includes / excludes |
// |---|---|---|
// | 1 | `carbonyl` | C=O (fixed) | every ketone/acid/ester/amide carbonyl; CO2 counts twice; generic overlap by design |
// | 2 | `carboxylic_acid` | C(=O)–O[H1], carbonyl C's remaining substituent C or H | formic/acetic/benzoic count; carbonic, carbamic, carbonate flanks do NOT (remaining O/N); fragment needs the carbon closed |
// | 3 | `ester` | C(=O)–O[H0]–C, carbonyl C's remaining C/H, alkoxy C no fixed double to O,S,N | ethyl acetate, methyl benzoate, lactones count; anhydride flanks, carbonates, carbamates do NOT |
// | 4 | `amide` | C(=O)–N, carbonyl C's remaining C/H | acetamide, benzamide, lactams count (imides once per C(=O)–N); ureas/carbamates do NOT (go to 26 only) |
// | 5 | `aldehyde` | C[H1 or H2]=O (fixed), no single bond to N,O,S,halogen | acetaldehyde, benzaldehyde, propanal, formaldehyde (H2) count; formic acid/formates excluded |
// | 6 | `ketone` | C[H0](=O)(–C)(–C) | acetone, cyclohexanone, quinone carbonyls count |
// | 7 | `hydroxyl` | C–O[H1], that C no fixed double to O,S,N | ethanol, phenols (aromatic C counts), enols count; acid OH excluded |
// | 8 | `ether` | C–O[H0]–C, neither C fixed double to O,S,N; O not a five-ring heteroatom | diethyl ether, anisole, THF count; ester/anhydride oxygens and furan O excluded |
// | 9 | `primary_amine` | C–N[H2], that C no fixed double | ethylamine, aniline, 2-aminopyridine (aromatic C counts) count; amides excluded |
// | 10 | `secondary_amine` | C–N[H1]–C, neither C fixed double; N not a five-ring heteroatom | dialkylamines, piperidine count; pyrrole N excluded; amides excluded |
// | 11 | `tertiary_amine` | N[H0](–C)3, no C fixed double; N not a five-ring heteroatom | trialkylamines count; N-methylpyrrole excluded |
// | 12 | `nitrile` | C≡N | acetonitrile, benzonitrile count |
// | 13 | `imine` | C=N (fixed), not in a five-ring | acyclic imines/oximes count; pyridine/pyridazine ring bonds (delocalised) do NOT |
// | 14 | `alkene` | C=C (fixed), not in a five-ring | isolated alkenes count; arene (delocalised) and furan/pyrrole ring bonds do NOT |
// | 15 | `alkyne` | C≡C | 2-butyne, acetylene count |
// | 16 | `thiol` | C–S[H1], that C no fixed double to O,S,N | ethanethiol, thiophenol count; thioacids excluded |
// | 17 | `thioether` | C–S[H0]–C, S valence 2, S not a five-ring heteroatom | dialkyl sulphides count; thiophene S excluded |
// | 18 | `sulfonyl` | S[H0](=O)(=O) | sulphones, sulphonic-acid/sulphate motifs count; sulphoxides outside the atom-type domain |
// | 19 | `sulfonamide` | S[H0](=O)(=O)–N | methanesulphonamide and analogues count |
// | 20 | `phosphoryl` | P[H0]=O | trimethyl phosphate and analogues count (P is valence 5 in-domain) |
// | 21 | `fluoride` | C–F (covalent organofluorine motif) | — |
// | 22 | `chloride` | C–Cl | — |
// | 23 | `bromide` | C–Br | — |
// | 24 | `arene_ring` | six-membered C/N ring, all six ring bonds delocalised | benzene, pyridine, naphthalene (2), anthracene rings count; benzoquinone does NOT (fixed bonds); cyclooctatetraene does NOT (no six-ring) |
// | 25 | `iodide` | C–I | — |
// | 26 | `carbamate_or_urea` | N–C(=O)–N or N–C(=O)–O | ureas, carbamates, carbamic acid count |
// | 27 | `anhydride_or_carbonate` | C(=O)–O–C(=O), or O–C(=O)–O with both single O H0/H1 | anhydrides, carbonates, carbonic acid count; one generic type so these acyl-oxygen motifs are not lost |
// | 28 | `heteroaromatic_five_ring` | five-ring, 1–2 heteroatoms (N,O,S), each atom either a single-only heteroatom or carrying a ring double | pyrrole, furan, thiophene, imidazole, pyrazole, oxazole, thiazole, indole/benzofuran five-rings count; count = rings |
//
// Groups overlap by design: an acid also contains a `carbonyl`, a
// sulphonamide also contains a `sulfonyl`, urea also contains
// `carbamate_or_urea` (but no longer `amide`). Evaluation is reported on
// all 28 types (**full**), on the **specific** subset (no generic
// `carbonyl`, id 1) and on the **heteroatom** subset (every type except
// `carbonyl`, `alkene`, `alkyne`, `arene_ring`: the groups a chemist would
// list as functional groups proper).
//
// Chemistry notes: `aldehyde` allows H2 so formaldehyde counts; pyrrolic
// nitrogen (the single-bonded heteroatom of a five-ring) is not a
// conventional amine; furan oxygen is not an ether; thiophene sulphur is
// not a thioether; carbonic/carbamic acids are not carboxylic acids;
// carbonate/carbamate/anhydride flanks are not esters (anhydrides and
// carbonates have their own type 27); thioacids are not thiols. Tautomers
// are different labels: 2-pyridone (`O=c1cccc[nH]1`) gives
// `{carbonyl, amide, alkene}` while 2-hydroxypyridine (`Oc1ccccn1`) gives
// `{hydroxyl, arene_ring}` — kekulé-invariant, NOT tautomer-invariant.
//
// # Kekulé invariance: the delocalised-bond rule
//
// A kekulé form of the molecule is an assignment of single/double to the
// bonds that are single or double in the stored form (triple bonds stay)
// such that every atom keeps its number of double bonds `d(v)` (its count
// in the stored form). This is a degree-constrained subgraph (an `f`-factor)
// problem with `f(v) = d(v)`; the v3 π graph is the special case `d ≤ 1`
// and silently drops ring atoms with `d = 2` (hypervalent S, P). Two kekulé
// forms differ by flipping bond orders around **alternating cycles** (even
// cycles whose bonds alternate single, double, single, double, … in the
// stored orders — at a `d = 2` vertex such a cycle still passes through one
// single and one double edge, so strict alternation holds there too). The
// symmetric difference of two valid assignments is a union of alternating
// cycles. Delocalisation is decided exactly, with no cycle-length bound and
// no work limit, through the **candidate graph** and Tutte's gadget (see
// [`delocalised_bonds`]):
//
// * The **candidate graph** `G'` of a kekulized graph has as vertices the
//   atoms with `d(v) ≥ 1`, and as edges the single or double bonds between
//   two such vertices. A bond with an end of `d = 0` is fixed single in
//   every valid assignment. Triple bonds stay triple in every valid
//   assignment — but the single/double bonds incident to a triple-bonded
//   atom are NOT fixed by valence in general: `C#S1=NC=CC=C1` keeps its
//   triple fixed while all six ring bonds are delocalised (regression in
//   the fixture).
// * Decide "bond `e` is double in some valid assignment and single in
//   another" exactly for general `d(v)` by reduction to perfect matching
//   with **Tutte's gadget**: for vertex `v` with `k = deg_{G'}(v)` and
//   requirement `d(v)`, build `k` port vertices (one per incident edge of
//   `G'`) and `k − d(v)` core vertices, with every core adjacent to every
//   port of `v`; each original edge `(u, v)` of `G'` becomes one edge
//   between its two ports. (Vertices with `d(v) > k` cannot occur in a
//   valid form: the stored doubles sit inside `G'`, so `d(v) ≤ k` always.
//   Vertices with `d(v) = k` have no cores: every incident bond is fixed
//   double.) **Correspondence:** in any perfect matching of the gadget,
//   exactly `d(v)` ports of `v` match outward (the `k − d(v)` cores absorb
//   the rest), so the set of matched port–port edges is an assignment with
//   the prescribed degrees — an edge is double iff its port–port edge is
//   matched — and every valid assignment arises this way. The reduction is
//   therefore exact: the gadget has a perfect matching per assignment and
//   vice versa. The stored form gives the starting perfect matching `M`
//   (ports of single bonds matched to cores, in order).
// * Hence (a) **the delocalised set is the same in every kekulé form**:
//   a bond of `G'` is **delocalised** when it is double in some valid
//   assignment but single in another — for a double bond `e = (u, v)` (its
//   port–port edge is in `M`) iff the gadget minus that edge still has a
//   perfect matching; for a single bond `e = (u, v)` of `G'` iff forcing
//   its port–port edge (removing both ports, freeing their matched cores)
//   leaves a graph with a perfect matching — and every bond outside `G'`,
//   or inside `G'` failing its test, is **fixed**. Each test is one
//   augmenting-path search from the current matching (Edmonds' blossom
//   search, see `blossom` below): for `e ∈ M` remove the port–port edge
//   and unmatch its ports, then search an augmenting path between them;
//   for `e ∉ M` remove both ports, free their partners, then search an
//   augmenting path between the freed cores.
// * (b) **Atoms with two double bonds are NOT pinned in general**: only
//   atoms whose incident bonds admit no alternative (e.g. an allene centre
//   or terminal cumulene carbon, whose neighbours cannot take doubles; a
//   sulfonyl sulphur whose methyl neighbours have `d = 0`; a nitrile
//   carbon) are fixed — the gadget decides this per bond exactly, instead
//   of the v3 claim that every such atom is fixed by valence, which is
//   false for hypervalent atoms in rings (sulphur/phosphorus ring members
//   whose doubles can migrate around the ring).
//
// A kekulé form here keeps connectivity, hydrogen counts, valences and
// TRIPLE BONDS fixed: with those fixed,
// `d(v) = valence(v) − H(v) − heavy_degree(v) − 2·triple_count(v)`, so no
// further single/double rearrangement can change an atom's double count.
// The triple restriction is load-bearing: `C1#CC=C1` and `C1=C=CC=1` have
// identical connectivity, hydrogen counts and valences but different
// triple placements, hence different counts (`{alkene: 1, alkyne: 1}` vs
// `{alkene: 3}`) — they are different molecules, outside the module's
// equivalence relation (both in the fixture).
//
// ## Validated matchings only
//
// The matching search NEVER contributes a reconstructed search path as
// evidence. A decision is made only from a **validated perfect matching**:
// after the search, every matched pair must be a real edge of the graph
// given to the search, the matching must be perfect on the intended vertex
// set, and the tested edge must be absent (double test) or present (single
// test, after re-adding it). If validation fails, `debug_assert!` fails
// loudly in test builds, and in release the bond is reported UNDECIDED
// (never delocalised, never fixed).
//
// The witness of a delocalised bond `e` is the component through `e`
// of the symmetric difference `M Δ M'` between the stored form's matching
// `M` and the validated alternative matching `M'` — an alternating closed
// trail by construction (both matchings are perfect, so following the two
// mates alternately from `e` returns to its start; at the gadget level its
// vertices are distinct, as [`valid_witness`] checks). Projected to
// original atoms it may revisit an atom: it is NOT necessarily a simple
// even atom-cycle. The fixture's two triangles sharing one `S(H0,v6)`
// exchange their double assignments with no simple even atom-cycle witness
// (each triangle is odd; any cycle through both revisits the sulphur).
// The projected trail is validated too (closed, alternating, every edge
// real) before the fragment rule uses its vertices, and it is derived from
// the validated mate arrays — never from a search-internal vertex list.
//
// Every pattern below is defined on fixed and delocalised bonds, never on
// raw orders inside a delocalised system: "double bond" always means a fixed
// double bond. `alkene` is a fixed C=C, `imine` a fixed C=N, and the carbon
// exclusions consult fixed doubles only. `arene_ring` is a six-membered
// ring of C and/or N atoms all six of whose ring bonds are delocalised
// (naphthalene has two such six-rings in every kekulé form). A carbon in a
// delocalised ring is an aromatic carbon: a hydroxyl on it is a phenol-type
// hydroxyl and counts; a primary amine on it counts (2-aminopyridine has
// one in every form); pyridazine has zero `imine` in every form (its ring
// doubles are delocalised).
//
// The definition is about kekulé equivalence, not aromaticity:
// cyclooctatetraene's eight ring bonds are delocalised (they lie on an
// alternating eight-cycle) yet no six-ring test fires, so it has zero
// `arene_ring`; benzoquinone's ring bonds are fixed (its carbonyl carbons
// carry two ring singles, breaking alternation), so it is correctly not an
// arene.
//
// ## Exact matching test (no length bound, no work limit)
//
// There is no bound on the witnessing alternating cycle: hexacene needs a
// 26-cycle witness for some bonds, four-cycles count (biphenylene), and no
// search budget may silently change labels. Each bond test is decided by
// Edmonds' blossom algorithm for maximum matching in general graphs (odd
// cycles exist: five-rings, azulene), implemented in this module with no new
// dependency. The blossom search is the standard Edmonds formulation with
// `base[]`, `parent[]`, `used[]` and `blossom[]` sets, an LCA walk over
// bases, and `mark_path(v, b, child)` setting `parent[v] = child` while
// walking `v → base` (the classical O(V^3) form), kept self-contained in
// the `blossom` submodule. `arene_ring` stays "a six-membered ring of C/N
// atoms whose six ring bonds are all delocalised".
//
// ## Fragments: decided bond status and soundness
//
// In a fragment, atoms carry open valence ([`MolGraph::residual_valence`]).
// A bond's status is **decided** in the fragment iff it is the same in every
// completion of the fragment consistent with the element, hydrogen count and
// open valence of each atom; otherwise it is unknown, and every instance
// using it or consulting it is undetermined. Being undecided more often
// than necessary is acceptable; one wrong decided verdict is not. The
// implemented rule is a sound sufficient condition (it reports unknown more
// often than strictly necessary, never a wrong decided verdict):
//
// * Build `G'` and its gadget from the fragment's inside bond orders exactly
//   as for a whole molecule, with inside double counts `d(v)`. An atom's
//   `d` in a completion may exceed its inside `d` (a further double bond
//   outside needs residual valence ≥ 2: sulphur, phosphorus, and any atom
//   whose residual valence allows it), so such atoms are never stable
//   witness vertices.
// * A bond decided **delocalised** inside the fragment (gadget matching test
//   on the fragment) is reported decided only when a validated `M Δ M'`
//   witness cycle exists whose vertices are all **stable** — every original
//   atom owning a gadget vertex of the cycle has open valence ≤ 1, hence
//   cannot gain a second double or a triple outside, so its `d` is final,
//   every cycle edge stays in `G'` with the same orders in every completion,
//   and the flip preserves `d` at every vertex. The stable witness is
//   decided existentially: when the first search's witness routes through
//   unstable atoms, a constrained search (unstable-owned gadget vertices
//   deleted, stored doubles touching them forced) decides whether any
//   stable witness exists, so the verdict cannot depend on which witness
//   the first search happened to find (FG7). The flipped inside
//   assignment therefore extends (outside unchanged) to a valid assignment
//   of every completion with the bond flipped — the bond is delocalised in
//   every completion. (On a closed graph every vertex is stable, so
//   whole-molecule delocalised bonds are always decided.)
// * A bond decided **fixed** inside the fragment is reported decided only
//   when at least one of its ends is **sealed**: the end can take neither
//   its required first step (opposite order to the bond: a single when the
//   bond is double needs residual ≥ 1, a double when the bond is single
//   needs residual ≥ 2) directly outside, nor start an alternating path
//   (first step of the opposite order, then strictly alternating
//   single/double, never reusing the bond) through inside edges that reaches
//   an atom with open valence. Any alternating flip cycle through the bond
//   in any completion alternates single/double in the stored orders (even at
//   `d = 2` vertices, where it passes through one single and one double),
//   so it must leave through both ends: a sealed end leaves only an inside
//   alternating cycle — but flipping it would be a valid inside reassignment
//   moving the bond, contradicting the failed gadget test — or an inside
//   alternating path to the boundary or an immediate outside exit (cut the
//   cycle at its first exit), a contradiction. In particular a bond with a
//   closed degree-one endpoint (an exocyclic C=O whose oxygen leaf is
//   closed) is always decided fixed, even in fragments.
//
// *Soundness.* Let F be an induced fragment of a parent P (same atom
// types, same bond orders on shared bonds, hence the parent's stored form
// restricts to the fragment's stored form) and let I be a determined
// instance in F. Then the type of I is present in P, for every kekulé form
// of P and of F. Proof sketch: required atoms/bonds of I are inside F,
// hence inside P with the same raw orders, and delocalised status is
// form-independent so the stored forms decide. If a required double of I is
// fixed-decided in F, any alternating flip cycle through that bond in any
// completion — in particular in P — would restrict to an inside alternating
// cycle of F (contradicting the failed gadget test: flipping it moves the
// bond inside) or to an inside alternating path to the boundary or an
// immediate outside exit (cut the cycle at its first exit), contradicting
// the sealed end — so the bond is fixed in P as well. If it is
// delocalised-decided in F, the validated witness cycle sits inside P
// unchanged (all its atoms stable: same `d`, same `G'` edges) and flips the
// bond there, so the bond is delocalised in P too. The same cut argument
// applies to exclusion consultations: a fixed-double exclusion certain in F
// (no fixed double inside, residual below 2, no undecided double to the
// relevant elements, no alternating path to the boundary that could supply
// one) cannot gain a fixed double in P. Single-neighbour exclusions need
// residual 0, hence all neighbours are inside. Five-ring exclusions are
// decided only when the five-ring is entirely inside F or no five-ring can
// complete outside (no open atom within graph distance 2 of the instance,
// and a five-ring through the instance stays within distance 2), so the
// exclusion verdict transfers. Carbonyl-remaining checks need the carbon
// closed with only C/H neighbours inside. Hence every determined verdict in
// F holds in P, and since the verdicts depend only on the delocalised set
// (form-independent), they hold for every kekulé form of both graphs.
//
// *Scope of the two claims.* (a) Closed-molecule kekulé invariance holds,
// as far as the review's attacks establish, for valid closed V0 atom-type
// graphs with connectivity, hydrogen counts, valences, and triple bonds
// fixed. (b) Fragment soundness holds, as far as those attacks establish:
// determined types are a subset of parent types, and decided bond statuses
// agree with the parent — for an order-preserving induced fragment of a
// valid closed parent. Decidedness is sufficient and conservative (undecided
// more often than necessary is acceptable; one wrong decided verdict is
// not). "Delocalised" is a combinatorial label (it includes non-aromatic
// cases such as cyclobutadiene), not chemical resonance generally, and
// nothing here extends across tautomers (2-pyridone and 2-hydroxypyridine
// are different labels) or across arbitrary valence-preserving triple
// rearrangements.
//
// # Determined instances and open valence
//
// A candidate is a fragment: [`MolGraph::residual_valence`] is the open
// valence of each atom. An instance is counted ([`functional_groups_v4`])
// only when it is **determined**: every required atom and bond is inside
// the graph, every consulted bond status is decided, and every exclusion is
// certain:
//
// * a fixed-double exclusion is certain when no fixed double to the
//   relevant elements is inside, no undecided double to them is inside,
//   and the atom's open valence is below 2;
// * a single-bond exclusion is certain when no such neighbour is inside
//   AND the atom's open valence is 0;
// * a carbonyl-remaining (C/H) check is certain when no hetero neighbour
//   is inside AND the carbon's open valence is 0;
// * a five-ring exclusion is certain when no inside heteroaromatic
//   five-ring covers the heteroatom/bond AND no open atom sits within
//   graph distance 2 of the instance (so no such ring completes outside).
//
// In the parent graph every atom has open valence 0 and every bond status
// is decided, so every instance is determined. Candidate-side detection is
// therefore conservative.
//
// # Anchors and counting
//
// The anchor of a pattern is the sorted set of all its matched graph atoms;
// per-type counts in [`FgSet`] count distinct anchor sets, so symmetric
// matches are never double counted. A carbon dioxide carbon carries two
// distinct C=O anchors and counts two `carbonyl`s. `arene_ring` and
// `heteroaromatic_five_ring` count distinct ring-atom sets.
//
// [`undetermined`] reports the mask of types with at least one matched but
// not determined instance. That is either a full match whose exclusion or
// bond status is uncertain (and not positively violated), or a partial
// core whose missing bonds could lie outside: a `C(=O)–O[H0]` core with an
// open oxygen is an undetermined `ester` (an `O[H1]` core would be an
// undetermined acid; hydrogen counts are respected, so an `O[H0]` core is
// never an undetermined acid); a `C–N[H1]` pair whose nitrogen has room for
// a second carbon is an undetermined `secondary_amine`; likewise
// `N[H0]`/`N[H2]` with room for more carbons, a `C–O[H0]` pair with an open
// oxygen (ether), a lone open `O[H1]` (hydroxyl), and a `C–S[H0]` pair with
// an open sulphur (thioether). A positively violated exclusion rejects the
// instance outright. Undetermined matches are reported, never scored.



/// Version string of this vocabulary.
pub const FG_VERSION: &str = "ms2-fg-v4";
/// Functional-group names in id order (id `i` is `FG_NAMES[i - 1]`).
pub const FG_NAMES: [&str; 28] = [
    "carbonyl",
    "carboxylic_acid",
    "ester",
    "amide",
    "aldehyde",
    "ketone",
    "hydroxyl",
    "ether",
    "primary_amine",
    "secondary_amine",
    "tertiary_amine",
    "nitrile",
    "imine",
    "alkene",
    "alkyne",
    "thiol",
    "thioether",
    "sulfonyl",
    "sulfonamide",
    "phosphoryl",
    "fluoride",
    "chloride",
    "bromide",
    "arene_ring",
    "iodide",
    "carbamate_or_urea",
    "anhydride_or_carbonate",
    "heteroaromatic_five_ring",
];
/// Type count of the vocabulary.
pub const N_FG: usize = 28;
/// Presence mask of the full vocabulary (all 28 bits).
pub const FULL_MASK: u32 = (1u32 << N_FG) - 1;
/// Presence mask of the **specific** subset: every type except id 1
/// (`carbonyl`).
pub const SPECIFIC_MASK: u32 = FULL_MASK ^ (1u32 << 0);
/// Presence mask of the **heteroatom** subset: every type except `carbonyl`,
/// `alkene`, `alkyne` and `arene_ring` (ids 1, 14, 15, 24) — the groups a
/// chemist would list as functional groups proper.
pub const HETEROATOM_MASK: u32 = FULL_MASK ^ ((1u32 << 0) | (1u32 << 13) | (1u32 << 14) | (1u32 << 23));

/// Whether a 1-based type id belongs to the specific subset (all but id 1).
pub fn is_specific(id: usize) -> bool {
    id >= 2 && id <= N_FG
}

/// Whether a 1-based type id belongs to the heteroatom subset (all but ids
/// 1, 14, 15 and 24).
pub fn is_heteroatom(id: usize) -> bool {
    id >= 1 && id <= N_FG && HETEROATOM_MASK & (1u32 << (id - 1)) != 0
}

/// Elements (element ids in [`super::chem::ELEMENTS`] order) a type needs
/// beyond C, H and O: N for amides, amines, nitriles, imines, sulfonamides
/// and carbamates/ureas; S for thiols, thioethers, sulfonyls and
/// sulfonamides; P for phosphoryl; one entry per halogen. Types whose
/// chemistry spans several heteroatoms (acids, esters, the anhydride/
/// carbonate composite and the five-ring) need nothing: they are never
/// filtered by the formula-aware prior. Used by the formula-aware prior: a
/// type is kept only if the spectrum's top formula contains every listed
/// element.
pub fn fg_elements(id: usize) -> &'static [usize] {
    match id {
        4 | 9 | 10 | 11 | 26 => &[2],
        12 | 13 => &[2],
        16 | 17 | 18 => &[6],
        19 => &[2, 6],
        20 => &[5],
        21 => &[4],
        22 => &[7],
        23 => &[8],
        25 => &[9],
        _ => &[],
    }
}

/// Determined functional-group instances of one graph: per-type counts over
/// distinct anchor-atom sets plus a presence bitmask.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FgSet {
    /// Determined instance count per type (index `id - 1`).
    counts: [u32; N_FG],
    /// Presence bitmask (bit `id - 1`).
    mask: u32,
}

impl FgSet {
    /// The empty set: no instance of any type.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Determined instance counts per type (index `id - 1`).
    pub fn counts(&self) -> &[u32; N_FG] {
        &self.counts
    }

    /// Presence bitmask (bit `id - 1` set when the type is present).
    pub fn mask(&self) -> u32 {
        self.mask
    }

    /// Determined instance count of a 1-based type id.
    pub fn count(&self, id: usize) -> u32 {
        self.counts[id - 1]
    }

    /// Whether a 1-based type id is present.
    pub fn present(&self, id: usize) -> bool {
        self.mask & (1u32 << (id - 1)) != 0
    }

    /// Present 1-based type ids, ascending.
    pub fn types(&self) -> Vec<usize> {
        (1..=N_FG).filter(|&id| self.present(id)).collect()
    }

    /// Total determined instances over the full vocabulary.
    pub fn total(&self) -> u32 {
        self.counts.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Pattern table
// ---------------------------------------------------------------------------

/// Element ids (`ELEMENTS` order) used by the table.
const C: usize = 0;
const N: usize = 2;
const O: usize = 3;
const F: usize = 4;
const P: usize = 5;
const S: usize = 6;
const CL: usize = 7;
const BR: usize = 8;
const I: usize = 9;

/// Hydrogen-count requirement of a pattern atom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HReq {
    /// Any parent hydrogen count.
    Any,
    /// Exactly this parent hydrogen count.
    Exact(u8),
    /// H1 or H2 (the aldehyde carbon: acetaldehyde H1, formaldehyde H2).
    H1H2,
}

/// One required pattern atom: element, hydrogen constraint, valence constraint.
#[derive(Clone, Copy, Debug)]
struct PatNode {
    /// Element id (`ELEMENTS` order).
    element: usize,
    /// Hydrogen-count constraint.
    h: HReq,
    /// Required valence (`None` means any).
    valence: Option<u8>,
}

/// One required pattern bond.
#[derive(Clone, Copy, Debug)]
struct PatBond {
    /// First pattern atom.
    a: usize,
    /// Second pattern atom.
    b: usize,
    /// Kekulized bond order.
    order: u8,
}

/// One exclusion on a named pattern atom: no bond whose order is in `orders`
/// to an element in `elements`. For double exclusions only *fixed* doubles
/// count as violations; certainty needs open valence below `min(orders)`
/// and every consulted double bond decided. A matching inside bond rejects
/// the instance outright.
#[derive(Clone, Copy, Debug)]
struct Exclusion {
    /// Pattern atom the exclusion is about.
    node: usize,
    /// Forbidden bond orders.
    orders: &'static [u8],
    /// Forbidden neighbour elements.
    elements: &'static [usize],
}

/// Forbidden double bond to O, S or N (the shared carbon exclusion; fixed
/// doubles only).
const NO_DOUBLE_OSN: &[usize] = &[O, S, N];
/// Forbidden single bond to N, O, S or a halogen (the aldehyde exclusion).
const NO_SINGLE_NOSH_HAL: &[usize] = &[N, O, S, F, CL, BR, I];
/// Double-bond orders (shared slice for exclusions).
const ORDER_DOUBLE: &[u8] = &[2];
/// Single-bond orders (shared slice for exclusions).
const ORDER_SINGLE: &[u8] = &[1];

/// One vocabulary entry: 1-based id, required atoms and bonds, exclusions,
/// and rule flags.
struct Pattern {
    /// 1-based vocabulary id.
    id: usize,
    /// Required atoms.
    nodes: &'static [PatNode],
    /// Required bonds.
    bonds: &'static [PatBond],
    /// Exclusions on named atoms.
    exclusions: &'static [Exclusion],
    /// Extra per-type rule (remaining checks, five-ring interaction, …).
    extra: Extra,
}

/// Extra per-type rule beyond the generic core + exclusions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Extra {
    /// No extra rule.
    None,
    /// Carbonyl-remaining C/H check on node 0 (acid, ester, amide).
    CarbonylRemaining,
    /// Alkene: fixed C=C, not in a five-ring.
    Alkene,
    /// Imine: fixed C=N, not in a five-ring.
    Imine,
    /// Thiol carbon exclusion is handled via generic exclusions; no extra.
    Thiol,
}

/// Shorthand for a pattern node without a valence constraint.
const fn nd(element: usize, h: HReq) -> PatNode {
    PatNode {
        element,
        h,
        valence: None,
    }
}

/// Shorthand for a required bond.
const fn bd(a: usize, b: usize, order: u8) -> PatBond {
    PatBond { a, b, order }
}

// Pattern node lists (one per vocabulary entry; shared where identical).
const NODES_C_O: &[PatNode] = &[nd(C, HReq::Any), nd(O, HReq::Any)];
const NODES_ACID: &[PatNode] = &[nd(C, HReq::Any), nd(O, HReq::Any), nd(O, HReq::Exact(1))];
const NODES_ESTER: &[PatNode] = &[
    nd(C, HReq::Any),
    nd(O, HReq::Any),
    nd(O, HReq::Exact(0)),
    nd(C, HReq::Any),
];
const NODES_AMIDE: &[PatNode] = &[nd(C, HReq::Any), nd(O, HReq::Any), nd(N, HReq::Any)];
const NODES_ALDEHYDE: &[PatNode] = &[nd(C, HReq::H1H2), nd(O, HReq::Any)];
const NODES_KETONE: &[PatNode] = &[
    nd(C, HReq::Exact(0)),
    nd(O, HReq::Any),
    nd(C, HReq::Any),
    nd(C, HReq::Any),
];
const NODES_HYDROXYL: &[PatNode] = &[nd(C, HReq::Any), nd(O, HReq::Exact(1))];
const NODES_ETHER: &[PatNode] = &[nd(C, HReq::Any), nd(O, HReq::Exact(0)), nd(C, HReq::Any)];
const NODES_PRI_AMINE: &[PatNode] = &[nd(C, HReq::Any), nd(N, HReq::Exact(2))];
const NODES_SEC_AMINE: &[PatNode] = &[
    nd(C, HReq::Any),
    nd(N, HReq::Exact(1)),
    nd(C, HReq::Any),
];
const NODES_TER_AMINE: &[PatNode] = &[
    nd(N, HReq::Exact(0)),
    nd(C, HReq::Any),
    nd(C, HReq::Any),
    nd(C, HReq::Any),
];
const NODES_C_N: &[PatNode] = &[nd(C, HReq::Any), nd(N, HReq::Any)];
const NODES_C_C: &[PatNode] = &[nd(C, HReq::Any), nd(C, HReq::Any)];
const NODES_THIOL: &[PatNode] = &[nd(C, HReq::Any), nd(S, HReq::Exact(1))];
const NODES_THIOETHER: &[PatNode] = &[
    nd(C, HReq::Any),
    PatNode {
        element: S,
        h: HReq::Exact(0),
        valence: Some(2),
    },
    nd(C, HReq::Any),
];
const NODES_SULFONYL: &[PatNode] = &[nd(S, HReq::Exact(0)), nd(O, HReq::Any), nd(O, HReq::Any)];
const NODES_SULFONAMIDE: &[PatNode] = &[
    nd(S, HReq::Exact(0)),
    nd(O, HReq::Any),
    nd(O, HReq::Any),
    nd(N, HReq::Any),
];
const NODES_PHOSPHORYL: &[PatNode] = &[nd(P, HReq::Exact(0)), nd(O, HReq::Any)];
const NODES_FLUORIDE: &[PatNode] = &[nd(C, HReq::Any), nd(F, HReq::Any)];
const NODES_CHLORIDE: &[PatNode] = &[nd(C, HReq::Any), nd(CL, HReq::Any)];
const NODES_BROMIDE: &[PatNode] = &[nd(C, HReq::Any), nd(BR, HReq::Any)];
const NODES_IODIDE: &[PatNode] = &[nd(C, HReq::Any), nd(I, HReq::Any)];
const NODES_UREA: &[PatNode] = &[
    nd(N, HReq::Any),
    nd(C, HReq::Any),
    nd(O, HReq::Any),
    nd(N, HReq::Any),
];
const NODES_CARBAMATE: &[PatNode] = &[
    nd(N, HReq::Any),
    nd(C, HReq::Any),
    nd(O, HReq::Any),
    nd(O, HReq::Any),
];

// Pattern bond lists.
const BONDS_DOUBLE: &[PatBond] = &[bd(0, 1, 2)];
const BONDS_SINGLE: &[PatBond] = &[bd(0, 1, 1)];
const BONDS_TRIPLE: &[PatBond] = &[bd(0, 1, 3)];
const BONDS_ACID: &[PatBond] = &[bd(0, 1, 2), bd(0, 2, 1)];
const BONDS_ESTER: &[PatBond] = &[bd(0, 1, 2), bd(0, 2, 1), bd(2, 3, 1)];
const BONDS_KETONE: &[PatBond] = &[bd(0, 1, 2), bd(0, 2, 1), bd(0, 3, 1)];
const BONDS_ETHER_AMINE2: &[PatBond] = &[bd(0, 1, 1), bd(1, 2, 1)];
const BONDS_TER_AMINE: &[PatBond] = &[bd(0, 1, 1), bd(0, 2, 1), bd(0, 3, 1)];
const BONDS_SULFONYL: &[PatBond] = &[bd(0, 1, 2), bd(0, 2, 2)];
const BONDS_SULFONAMIDE: &[PatBond] = &[bd(0, 1, 2), bd(0, 2, 2), bd(0, 3, 1)];
const BONDS_UREA: &[PatBond] = &[bd(0, 1, 1), bd(1, 2, 2), bd(1, 3, 1)];

// Exclusion lists (double exclusions consult fixed doubles only).
const EXCL_ALDEHYDE: &[Exclusion] = &[Exclusion {
    node: 0,
    orders: ORDER_SINGLE,
    elements: NO_SINGLE_NOSH_HAL,
}];
const EXCL_HYDROXYL: &[Exclusion] = &[Exclusion {
    node: 0,
    orders: ORDER_DOUBLE,
    elements: NO_DOUBLE_OSN,
}];
const EXCL_ETHER: &[Exclusion] = &[
    Exclusion {
        node: 0,
        orders: ORDER_DOUBLE,
        elements: NO_DOUBLE_OSN,
    },
    Exclusion {
        node: 2,
        orders: ORDER_DOUBLE,
        elements: NO_DOUBLE_OSN,
    },
];
const EXCL_PRI_AMINE: &[Exclusion] = &[Exclusion {
    node: 0,
    orders: ORDER_DOUBLE,
    elements: NO_DOUBLE_OSN,
}];
const EXCL_SEC_AMINE: &[Exclusion] = &[
    Exclusion {
        node: 0,
        orders: ORDER_DOUBLE,
        elements: NO_DOUBLE_OSN,
    },
    Exclusion {
        node: 2,
        orders: ORDER_DOUBLE,
        elements: NO_DOUBLE_OSN,
    },
];
const EXCL_TER_AMINE: &[Exclusion] = &[
    Exclusion {
        node: 1,
        orders: ORDER_DOUBLE,
        elements: NO_DOUBLE_OSN,
    },
    Exclusion {
        node: 2,
        orders: ORDER_DOUBLE,
        elements: NO_DOUBLE_OSN,
    },
    Exclusion {
        node: 3,
        orders: ORDER_DOUBLE,
        elements: NO_DOUBLE_OSN,
    },
];
const EXCL_THIOL: &[Exclusion] = &[Exclusion {
    node: 0,
    orders: ORDER_DOUBLE,
    elements: NO_DOUBLE_OSN,
}];
const EXCL_ESTER_ALKOXY: &[Exclusion] = &[Exclusion {
    node: 3,
    orders: ORDER_DOUBLE,
    elements: NO_DOUBLE_OSN,
}];

/// The pattern table: ids 1–23, 25–26 as generic rows (24, 27, 28 are
/// structural and handled by dedicated code); id 26 has two rows.
const PATTERNS: &[Pattern] = &[
    Pattern { id: 1, nodes: NODES_C_O, bonds: BONDS_DOUBLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 2, nodes: NODES_ACID, bonds: BONDS_ACID, exclusions: &[], extra: Extra::CarbonylRemaining },
    Pattern { id: 3, nodes: NODES_ESTER, bonds: BONDS_ESTER, exclusions: EXCL_ESTER_ALKOXY, extra: Extra::CarbonylRemaining },
    Pattern { id: 4, nodes: NODES_AMIDE, bonds: BONDS_ACID, exclusions: &[], extra: Extra::CarbonylRemaining },
    Pattern { id: 5, nodes: NODES_ALDEHYDE, bonds: BONDS_DOUBLE, exclusions: EXCL_ALDEHYDE, extra: Extra::None },
    Pattern { id: 6, nodes: NODES_KETONE, bonds: BONDS_KETONE, exclusions: &[], extra: Extra::None },
    Pattern { id: 7, nodes: NODES_HYDROXYL, bonds: BONDS_SINGLE, exclusions: EXCL_HYDROXYL, extra: Extra::None },
    Pattern { id: 8, nodes: NODES_ETHER, bonds: BONDS_ETHER_AMINE2, exclusions: EXCL_ETHER, extra: Extra::None },
    Pattern { id: 9, nodes: NODES_PRI_AMINE, bonds: BONDS_SINGLE, exclusions: EXCL_PRI_AMINE, extra: Extra::None },
    Pattern { id: 10, nodes: NODES_SEC_AMINE, bonds: BONDS_ETHER_AMINE2, exclusions: EXCL_SEC_AMINE, extra: Extra::None },
    Pattern { id: 11, nodes: NODES_TER_AMINE, bonds: BONDS_TER_AMINE, exclusions: EXCL_TER_AMINE, extra: Extra::None },
    Pattern { id: 12, nodes: NODES_C_N, bonds: BONDS_TRIPLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 13, nodes: NODES_C_N, bonds: BONDS_DOUBLE, exclusions: &[], extra: Extra::Imine },
    Pattern { id: 14, nodes: NODES_C_C, bonds: BONDS_DOUBLE, exclusions: &[], extra: Extra::Alkene },
    Pattern { id: 15, nodes: NODES_C_C, bonds: BONDS_TRIPLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 16, nodes: NODES_THIOL, bonds: BONDS_SINGLE, exclusions: EXCL_THIOL, extra: Extra::Thiol },
    Pattern { id: 17, nodes: NODES_THIOETHER, bonds: BONDS_ETHER_AMINE2, exclusions: &[], extra: Extra::None },
    Pattern { id: 18, nodes: NODES_SULFONYL, bonds: BONDS_SULFONYL, exclusions: &[], extra: Extra::None },
    Pattern { id: 19, nodes: NODES_SULFONAMIDE, bonds: BONDS_SULFONAMIDE, exclusions: &[], extra: Extra::None },
    Pattern { id: 20, nodes: NODES_PHOSPHORYL, bonds: BONDS_DOUBLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 21, nodes: NODES_FLUORIDE, bonds: BONDS_SINGLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 22, nodes: NODES_CHLORIDE, bonds: BONDS_SINGLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 23, nodes: NODES_BROMIDE, bonds: BONDS_SINGLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 25, nodes: NODES_IODIDE, bonds: BONDS_SINGLE, exclusions: &[], extra: Extra::None },
    Pattern { id: 26, nodes: NODES_UREA, bonds: BONDS_UREA, exclusions: &[], extra: Extra::None },
    Pattern { id: 26, nodes: NODES_CARBAMATE, bonds: BONDS_UREA, exclusions: &[], extra: Extra::None },
];

// ---------------------------------------------------------------------------
// Graph view, delocalised bonds, rings
// ---------------------------------------------------------------------------

/// One graph atom as the patterns see it.
#[derive(Clone, Copy)]
struct AtomView {
    /// Element id (`ELEMENTS` order).
    element: usize,
    /// Parent hydrogen count.
    h: u8,
    /// Valence (hydrogens plus kekulized bond orders in the parent).
    valence: u8,
}

/// Bond status: fixed (in no perfect matching of the π graph), delocalised
/// (in some but not all), or undecided in a fragment (could complete outside).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BondStatus {
    /// In no perfect matching of the π graph (also every bond outside it and
    /// all triples).
    Fixed,
    /// In some but not all perfect matchings of the π graph.
    Delocalised,
    /// Undecided: the inside matching test does not transfer to every
    /// completion, so an outside completion is possible.
    Unknown,
}

/// The pattern-visible view of a graph.
struct View<'a> {
    /// Metadata per graph atom.
    atoms: Vec<AtomView>,
    /// Neighbours with (neighbour, order, bond index) per graph atom.
    adj: Vec<Vec<(usize, u8, usize)>>,
    /// Open valence per graph atom.
    residual: Vec<u8>,
    /// Raw bond orders per bond index.
    bond_orders: Vec<u8>,
    /// Status per bond index.
    status: Vec<BondStatus>,
    /// The underlying graph (for helpers).
    _graph: &'a MolGraph,
}

impl<'a> View<'a> {
    /// Build the view, computing delocalised bond statuses.
    fn of(graph: &'a MolGraph) -> core::result::Result<Self, String> {
        let mut atoms = Vec::with_capacity(graph.atoms().len());
        for (i, id) in graph.atoms().iter().enumerate() {
            let t = chem::atom_type(*id)
                .ok_or_else(|| format!("functional_groups_v4: unknown atom type id {id} at atom {i}"))?;
            atoms.push(AtomView {
                element: t.element,
                h: t.hydrogens,
                valence: t.valence,
            });
        }
        let n = atoms.len();
        let nb = graph.bonds().len();
        let mut adj: Vec<Vec<(usize, u8, usize)>> = vec![Vec::new(); n];
        let mut bond_orders = vec![0u8; nb];
        for (idx, (a, b, order)) in graph.bonds().iter().enumerate() {
            adj[*a].push((*b, *order, idx));
            adj[*b].push((*a, *order, idx));
            bond_orders[idx] = *order;
        }
        let residual = graph.residual_valence();
        let mut bond_ends = vec![(0usize, 0usize); nb];
        for (idx, (a, b, _)) in graph.bonds().iter().enumerate() {
            bond_ends[idx] = (*a, *b);
        }
        let status = compute_statuses(&adj, &bond_orders, &bond_ends, &residual);
        Ok(Self {
            atoms,
            adj,
            residual,
            bond_orders,
            status,
            _graph: graph,
        })
    }

    /// Bond index between adjacent atoms, if bonded.
    fn bond_idx(&self, a: usize, b: usize) -> Option<usize> {
        self.adj[a].iter().find(|(x, _, _)| *x == b).map(|(_, _, i)| *i)
    }

    /// Bond order between adjacent atoms, if bonded.
    fn order(&self, a: usize, b: usize) -> Option<u8> {
        self.bond_idx(a, b).map(|i| self.bond_orders[i])
    }

    /// Whether the bond between adjacent atoms is a fixed double.
    fn is_fixed_double(&self, a: usize, b: usize) -> bool {
        match self.bond_idx(a, b) {
            Some(i) => self.bond_orders[i] == 2 && self.status[i] == BondStatus::Fixed,
            None => false,
        }
    }

    /// Whether the bond between adjacent atoms is an undecided double
    /// (raw double whose status is unknown).
    fn is_unknown_double(&self, a: usize, b: usize) -> bool {
        match self.bond_idx(a, b) {
            Some(i) => self.bond_orders[i] == 2 && self.status[i] == BondStatus::Unknown,
            None => false,
        }
    }

    /// Whether a graph atom satisfies a pattern node predicate.
    fn node_ok(&self, g: usize, node: &PatNode) -> bool {
        let a = &self.atoms[g];
        if a.element != node.element {
            return false;
        }
        match node.h {
            HReq::Any => {}
            HReq::Exact(h) => {
                if a.h != h {
                    return false;
                }
            }
            HReq::H1H2 => {
                if a.h != 1 && a.h != 2 {
                    return false;
                }
            }
        }
        if let Some(v) = node.valence
            && a.valence != v
        {
            return false;
        }
        true
    }
}

/// Tutte's gadget for the degree-constrained (f-factor) kekulé problem (see
/// the module docs): for vertex `v` of the candidate graph `G'` with
/// `k = deg(v)` and required doubles `d(v)`, `k` port vertices (one per
/// incident edge) plus `k − d(v)` core vertices, every core adjacent to every
/// port of `v`, and one port–port edge per original edge of `G'`. Perfect
/// matchings of the gadget are exactly the valid single/double assignments
/// (an edge is double iff its port–port edge is matched); the stored form
/// gives the starting perfect matching (ports of single bonds matched to
/// cores). Shared by the detector and the closing-fragment search.
struct Gadget {
    /// Gadget adjacency (port and core vertices).
    adj: Vec<Vec<usize>>,
    /// Original atom owning each gadget vertex.
    owner: Vec<usize>,
    /// Whether each gadget vertex is a core vertex (else a port).
    is_core: Vec<bool>,
    /// Gadget port endpoints per original bond index, for bonds of `G'`.
    bond_ports: Vec<Option<(usize, usize)>>,
    /// The stored form's perfect matching of the gadget.
    mate0: Vec<Option<usize>>,
}

impl Gadget {
    /// Build the gadget from the atom count, bond endpoints and raw orders.
    fn build(n_atoms: usize, ends: &[(usize, usize)], orders: &[u8]) -> Self {
        // Double-bond counts `d(v)` over all bonds (a double bond always has
        // both ends at `d ≥ 1`, so every stored double sits inside `G'` and
        // `d(v) ≤ deg_{G'}(v)` holds below).
        let mut d = vec![0u8; n_atoms];
        for (idx, (a, b)) in ends.iter().enumerate() {
            if orders[idx] == 2 {
                d[*a] = d[*a].saturating_add(1);
                d[*b] = d[*b].saturating_add(1);
            }
        }
        let in_g: Vec<bool> = d.iter().map(|&x| x >= 1).collect();
        // `G'` edges (original bond indices): order 1 or 2 with both ends in `G'`.
        let mut g_edge_of: Vec<Option<usize>> = vec![None; orders.len()];
        let mut g_edges: Vec<usize> = Vec::new();
        let mut deg = vec![0usize; n_atoms];
        for (idx, (a, b)) in ends.iter().enumerate() {
            if (orders[idx] == 1 || orders[idx] == 2) && in_g[*a] && in_g[*b] {
                g_edge_of[idx] = Some(g_edges.len());
                g_edges.push(idx);
                deg[*a] += 1;
                deg[*b] += 1;
            }
        }
        debug_assert!(
            (0..n_atoms).all(|v| !in_g[v] || usize::from(d[v]) <= deg[v]),
            "gadget: stored doubles sit inside G'"
        );
        // Ports: one per incident `G'` edge, in bond-index order (deterministic).
        let mut port_of: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
        let mut owner: Vec<usize> = Vec::new();
        let mut is_core: Vec<bool> = Vec::new();
        let mut bond_ports: Vec<Option<(usize, usize)>> = vec![None; orders.len()];
        // Per-vertex incident `G'` bonds in index order.
        let mut incid: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
        for &bi in &g_edges {
            let (a, b) = ends[bi];
            incid[a].push(bi);
            incid[b].push(bi);
        }
        for v in 0..n_atoms {
            if in_g[v] {
                for _ in &incid[v] {
                    port_of[v].push(owner.len());
                    owner.push(v);
                    is_core.push(false);
                }
            }
        }
        // Cores after the ports.
        let mut cores_of: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
        for v in 0..n_atoms {
            if in_g[v] {
                let n_core = deg[v] - usize::from(d[v]);
                for _ in 0..n_core {
                    cores_of[v].push(owner.len());
                    owner.push(v);
                    is_core.push(true);
                }
            }
        }
        let gn = owner.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); gn];
        // Complete bipartite cores(v)–ports(v).
        for v in 0..n_atoms {
            for &c in &cores_of[v] {
                for &p in &port_of[v] {
                    adj[c].push(p);
                    adj[p].push(c);
                }
            }
        }
        // One port–port edge per `G'` bond.
        for &bi in &g_edges {
            let (a, b) = ends[bi];
            // The port of `bi` at each end: position of `bi` in `incid`.
            let pa = port_of[a][incid[a].iter().position(|&x| x == bi).expect("gadget: port exists")];
            let pb = port_of[b][incid[b].iter().position(|&x| x == bi).expect("gadget: port exists")];
            adj[pa].push(pb);
            adj[pb].push(pa);
            bond_ports[bi] = Some((pa, pb));
        }
        // The stored form's perfect matching: double bonds match port–port,
        // single bonds match each port to a distinct core of its owner.
        let mut mate0: Vec<Option<usize>> = vec![None; gn];
        for &bi in &g_edges {
            let (pa, pb) = bond_ports[bi].expect("gadget: G' bond has ports");
            if orders[bi] == 2 {
                mate0[pa] = Some(pb);
                mate0[pb] = Some(pa);
            }
        }
        for v in 0..n_atoms {
            if !in_g[v] {
                continue;
            }
            // Single-bond ports of `v` pair with its cores in order.
            let singles: Vec<usize> = incid[v]
                .iter()
                .filter(|&&bi| orders[bi] == 1)
                .map(|&bi| {
                    let (a, _) = ends[bi];
                    if a == v {
                        bond_ports[bi].expect("gadget: G' bond has ports").0
                    } else {
                        bond_ports[bi].expect("gadget: G' bond has ports").1
                    }
                })
                .collect();
            debug_assert_eq!(
                singles.len(),
                cores_of[v].len(),
                "gadget: single-bond ports match the core count"
            );
            for (p, &c) in singles.iter().zip(cores_of[v].iter()) {
                mate0[*p] = Some(c);
                mate0[c] = Some(*p);
            }
        }
        Self { adj, owner, is_core, bond_ports, mate0 }
    }
}

/// The component through `e` of `mate_a Δ mate_b` as a gadget-vertex cycle
/// `c` with `c[0] == c[c.len() - 1]`.
///
/// Both mates must be perfect matchings on the same vertex set with `e` in
/// exactly one of them; the component is then an even alternating cycle by
/// construction. Returns `None` (defensive) when the walk does not close
/// properly — callers treat that as an undecided bond, never as evidence.
fn witness_cycle(
    mate_a: &[Option<usize>],
    mate_b: &[Option<usize>],
    e: (usize, usize),
) -> Option<Vec<usize>> {
    let (u, v) = e;
    if u >= mate_a.len() || v >= mate_a.len() || mate_a.len() != mate_b.len() {
        return None;
    }
    let in_a = mate_a[u] == Some(v);
    let in_b = mate_b[u] == Some(v);
    if in_a == in_b {
        return None;
    }
    // First step from `v` uses the mating NOT containing `e`; then alternate.
    let (first, second) = if in_a { (mate_b, mate_a) } else { (mate_a, mate_b) };
    let mut cyc = vec![u, v];
    let mut cur = v;
    loop {
        let n1 = first[cur]?;
        cyc.push(n1);
        cur = n1;
        if cur == u {
            break;
        }
        let n2 = second[cur]?;
        cyc.push(n2);
        cur = n2;
        if cur == u {
            break;
        }
        if cyc.len() > mate_a.len() + 2 {
            return None;
        }
    }
    Some(cyc)
}

/// Whether `cyc` (closed: first == last) is a genuine alternating cycle for
/// the mate pair: an even edge count ≥ 4 (vertex list is one longer, hence
/// odd), distinct inner vertices, every consecutive pair a real edge of `adj`
/// matched in exactly one of the two mates.
fn valid_witness(
    adj: &[Vec<usize>],
    mate_a: &[Option<usize>],
    mate_b: &[Option<usize>],
    cyc: &[usize],
) -> bool {
    if cyc.len() < 5 || cyc.len() % 2 == 0 {
        return false;
    }
    if cyc[0] != cyc[cyc.len() - 1] {
        return false;
    }
    let inner = &cyc[..cyc.len() - 1];
    if inner.len() != {
        let mut s = std::collections::BTreeSet::new();
        for &x in inner {
            s.insert(x);
        }
        s.len()
    } {
        return false;
    }
    for w in cyc.windows(2) {
        let (x, y) = (w[0], w[1]);
        if x >= adj.len() || !adj[x].contains(&y) {
            return false;
        }
        let xa = mate_a[x] == Some(y);
        let xb = mate_b[x] == Some(y);
        if xa == xb {
            return false;
        }
    }
    true
}

/// Connected components of the candidate graph `G'` as sorted graph-atom
/// lists: atoms with at least one double bond, joined by the single/double
/// bonds between two such atoms.
///
/// Used by the closing-fragment search (stage iii): a consulted bond is
/// decided once its whole `G'` component is inside the fragment. Every `G'`
/// vertex carries a double bond, hence an incident `G'` edge, so there are
/// no isolated vertices.
pub fn pi_graph_components(graph: &MolGraph) -> Vec<Vec<usize>> {
    let ends: Vec<(usize, usize)> = graph.bonds().iter().map(|(a, b, _)| (*a, *b)).collect();
    let orders: Vec<u8> = graph.bonds().iter().map(|(_, _, o)| *o).collect();
    let n = graph.atoms().len();
    let mut d = vec![0u8; n];
    for (idx, (a, b)) in ends.iter().enumerate() {
        if orders[idx] == 2 {
            d[*a] = d[*a].saturating_add(1);
            d[*b] = d[*b].saturating_add(1);
        }
    }
    let mut gadj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, (a, b)) in ends.iter().enumerate() {
        if (orders[idx] == 1 || orders[idx] == 2) && d[*a] >= 1 && d[*b] >= 1 {
            gadj[*a].push(*b);
            gadj[*b].push(*a);
        }
    }
    let mut seen = vec![false; n];
    let mut out: Vec<Vec<usize>> = Vec::new();
    for h in 0..n {
        if d[h] == 0 || seen[h] {
            continue;
        }
        let mut stack = vec![h];
        seen[h] = true;
        let mut comp = Vec::new();
        while let Some(v) = stack.pop() {
            comp.push(v);
            for &w in &gadj[v] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    out.sort();
    out
}

/// Compute per-bond statuses with the exact gadget matching rule plus the
/// fragment decided-rule (see the module docs). No length bound, no work
/// limit: every inside test is decided by a blossom search, and no decision
/// ever rests on a reconstructed search path — only on validated perfect
/// matchings and their validated `M Δ M'` witness cycles.
fn compute_statuses(
    adj: &[Vec<(usize, u8, usize)>],
    orders: &[u8],
    ends: &[(usize, usize)],
    residual: &[u8],
) -> Vec<BondStatus> {
    let gadget = Gadget::build(adj.len(), ends, orders);
    debug_assert!(
        blossom::is_valid_matching(&gadget.adj, &gadget.mate0),
        "gadget: stored doubles must perfectly match the gadget"
    );
    debug_assert!(
        blossom::is_perfect(&gadget.mate0),
        "gadget: stored matching must be perfect"
    );
    let mut status = vec![BondStatus::Fixed; orders.len()];
    for (idx, order) in orders.iter().enumerate() {
        if *order != 1 && *order != 2 {
            // Triple bonds (and anything else) are never delocalised, in
            // every completion: decided fixed.
            status[idx] = BondStatus::Fixed;
            continue;
        }
        let (a, b) = ends[idx];
        let Some((pu, pv)) = gadget.bond_ports[idx] else {
            // Outside G': fixed inside; decided iff no completion can put it
            // on a flip cycle.
            status[idx] = if fixed_decided(a, b, *order, adj, residual, idx) {
                BondStatus::Fixed
            } else {
                BondStatus::Unknown
            };
            continue;
        };
        if *order == 2 {
            status[idx] = test_gadget_double(idx, pu, pv, &gadget, adj, residual);
        } else {
            status[idx] = test_gadget_single(idx, pu, pv, &gadget, adj, residual);
        }
    }
    status
}

/// Outcome of one alternative-matching search, split as the module docs
/// promise: an INVALID returned matching (pairs that are not real edges, or
/// that do not cover the searched vertex set) is a loud `debug_assert!`
/// failure in tests and UNDECIDED in release — never fixed, never
/// delocalised; a VALID matching that is simply not perfect means no
/// alternative assignment exists, so the bond is fixed inside (subject to
/// the fragment decided-rule).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOutcome {
    /// The search returned a matching that fails validation.
    Invalid,
    /// The search returned a valid matching that is not perfect.
    NoAlternative,
    /// The search returned a validated perfect matching (a candidate
    /// alternative assignment; the caller still checks the tested edge).
    Candidate,
}

/// Pure decision on one alternative-matching search result: `valid` is
/// whether the returned matching holds real edges over the searched vertex
/// set, `perfect` whether it covers every vertex. Tested directly by
/// `search_outcome_decision` in `tests/ms2_functional_groups.rs`.
#[doc(hidden)]
pub fn classify_search_result(valid: bool, perfect: bool) -> SearchOutcome {
    if !valid {
        SearchOutcome::Invalid
    } else if !perfect {
        SearchOutcome::NoAlternative
    } else {
        SearchOutcome::Candidate
    }
}

/// FG8 negative-control switch: when `true`, the fragment rule uses only the
/// first witness found and skips the FG7 existential constrained search, so a
/// first witness routed through unstable atoms yields UNDECIDED instead of a
/// decided-delocalised verdict. Thread-local default off; used ONLY by the
/// FG8 dense-fragment numbering test's negative control.
thread_local! {
    static FIRST_WITNESS_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set the FG8 first-witness-only negative-control switch (test hook only).
#[doc(hidden)]
pub fn set_first_witness_only(on: bool) {
    FIRST_WITNESS_ONLY.with(|c| c.set(on));
}

/// Whether the FG8 first-witness-only switch is on (test hook only).
fn first_witness_only() -> bool {
    FIRST_WITNESS_ONLY.with(|c| c.get())
}

/// Pure decision on one constrained-search (remainder) result, shared by the
/// `stable_alternative_double` / `stable_alternative_single` guards and
/// [`confirm_no_perfect`]: an INVALID returned matching (fails validation) is
/// loud in tests and yields UNDECIDED in release, exactly as the
/// [`SearchOutcome`] split does on the main path; only a valid non-perfect
/// matching is a quiet negative. Tested directly by
/// `constrained_outcome_decision` in `tests/ms2_functional_groups.rs`.
#[doc(hidden)]
pub fn classify_constrained_result(valid: bool, perfect: bool) -> SearchOutcome {
    classify_search_result(valid, perfect)
}

/// Independent confirmation of a negative seeded search (FG7 belt-and-braces,
/// a SEPARATE layer from the fix itself): a from-scratch maximum matching of
/// the modified graph (empty start, same [`blossom::complete`] core but a
/// different search order from the seeded path). Returns `true` when it
/// agrees — the maximum size is below `n/2`, i.e. no perfect matching exists.
/// A disagreement (from-scratch finds a validated perfect matching the seeded
/// search missed) is a loud `debug_assert!` failure in tests and UNDECIDED in
/// release — never fixed. Kept only while the 714-spectrum validation-set
/// evaluation (`--limit-spectra 64` cpu driver) shows no more than 10%
/// slowdown with it on.
fn confirm_no_perfect(adj: &[Vec<usize>]) -> bool {
    if adj.is_empty() {
        return true;
    }
    let mut fresh: Vec<Option<usize>> = vec![None; adj.len()];
    blossom::complete(adj, &mut fresh);
    match classify_constrained_result(
        blossom::is_valid_matching(adj, &fresh),
        blossom::is_perfect(&fresh),
    ) {
        SearchOutcome::Invalid => {
            // The confirmation returned a matching that fails validation:
            // loud in tests, UNDECIDED (disagreement) in release.
            debug_assert!(
                false,
                "gadget test: from-scratch matching must be a valid matching of real edges"
            );
            false
        }
        SearchOutcome::NoAlternative => true,
        SearchOutcome::Candidate => {
            debug_assert!(
                false,
                "gadget test: from-scratch matching disagrees with the seeded search (missed augmenting path)"
            );
            false
        }
    }
}

/// Gadget vertices no stable witness cycle may use (FG7, layer (f)):
/// vertices owned by atoms with open valence above 1 (their double-bond
/// count `d` may still grow outside), except the tested edge's own ports.
/// Returns `None` when an end itself is unstable: every witness cycle
/// through the edge uses that end, so no stable witness exists.
fn forbidden_vertices(
    gadget: &Gadget,
    residual: &[u8],
    pu: usize,
    pv: usize,
) -> Option<Vec<bool>> {
    if residual[gadget.owner[pu]] > 1 || residual[gadget.owner[pv]] > 1 {
        return None;
    }
    Some(
        gadget
            .owner
            .iter()
            .enumerate()
            .map(|(g, &o)| g != pu && g != pv && residual[o] > 1)
            .collect(),
    )
}

/// Stored double-bond pairs (gadget port–port edges of the stored matching)
/// touching a forbidden vertex: any witness cycle avoiding the forbidden
/// vertices leaves these pairs matched as stored, so the constrained search
/// below forces them. The tested edge itself is never among them (its ends
/// are stable, checked by the caller).
fn forced_stored_pairs(gadget: &Gadget, forbidden: &[bool]) -> Vec<(usize, usize)> {
    let mut forced = Vec::new();
    for g in 0..gadget.adj.len() {
        if let Some(w) = gadget.mate0[g] {
            if g < w && !gadget.is_core[g] && !gadget.is_core[w] && (forbidden[g] || forbidden[w]) {
                forced.push((g, w));
            }
        }
    }
    forced
}

/// Remainder graph for the stable-witness search: the gadget minus the
/// forbidden vertices and both ends of every forced pair (reindexed), with
/// the stored matching restricted to it. Every remaining vertex keeps its
/// stored mate inside the remainder — a stored pair touching a removed
/// vertex is either internal to the removed set (same-owner port–core) or a
/// forced pair (removed together) — except the designated free vertices the
/// caller unmatched for its test. Returns the remainder adjacency, the seed
/// matching, the reindex maps, and the forced pairs.
fn stable_remainder(
    gadget: &Gadget,
    forbidden: &[bool],
    forced: &[(usize, usize)],
    free: &[usize],
) -> (Vec<Vec<usize>>, Vec<Option<usize>>, Vec<Option<usize>>, Vec<usize>) {
    let mut removed = forbidden.to_vec();
    for &(x, y) in forced {
        removed[x] = true;
        removed[y] = true;
    }
    let mut new_of: Vec<Option<usize>> = vec![None; gadget.adj.len()];
    let mut count = 0usize;
    for g in 0..gadget.adj.len() {
        if !removed[g] {
            new_of[g] = Some(count);
            count += 1;
        }
    }
    let mut adjr: Vec<Vec<usize>> = vec![Vec::new(); count];
    for g in 0..gadget.adj.len() {
        if let Some(ng) = new_of[g] {
            for &w in &gadget.adj[g] {
                if let Some(nw) = new_of[w] {
                    adjr[ng].push(nw);
                }
            }
        }
    }
    let mut seed: Vec<Option<usize>> = vec![None; count];
    for g in 0..gadget.adj.len() {
        if let (Some(ng), Some(mp)) = (new_of[g], gadget.mate0[g]) {
            if free.contains(&g) {
                seed[ng] = None;
            } else {
                seed[ng] = new_of[mp];
            }
        }
    }
    let mut inv = vec![0usize; count];
    for g in 0..gadget.adj.len() {
        if let Some(ng) = new_of[g] {
            inv[ng] = g;
        }
    }
    (adjr, seed, new_of, inv)
}

/// Whether the restricted seed covers every remainder vertex but the
/// designated free ones (the removal lemma of [`stable_remainder`]).
/// Called only inside `debug_assert!`, like the other validators here.
#[allow(dead_code)]
fn seed_covers_remainder(
    gadget: &Gadget,
    new_of: &[Option<usize>],
    seed: &[Option<usize>],
    free: &[usize],
) -> bool {
    for g in 0..gadget.adj.len() {
        if let Some(ng) = new_of[g] {
            if free.contains(&g) {
                if seed[ng].is_some() {
                    return false;
                }
            } else if seed[ng].is_none() {
                return false;
            }
        }
    }
    true
}

/// Lift a remainder matching to the full vertex set: remainder pairs as
/// found, forced pairs as stored, and every other removed vertex with its
/// stored mate (internal to the removed set by the removal lemma).
fn lift_stable(
    gadget: &Gadget,
    new_of: &[Option<usize>],
    inv: &[usize],
    found: &[Option<usize>],
    forced: &[(usize, usize)],
    removed: &[bool],
) -> Vec<Option<usize>> {
    let mut malt: Vec<Option<usize>> = vec![None; gadget.adj.len()];
    for g in 0..gadget.adj.len() {
        if let Some(ng) = new_of[g] {
            malt[g] = found[ng].map(|nmp| inv[nmp]);
        }
    }
    for &(x, y) in forced {
        malt[x] = Some(y);
        malt[y] = Some(x);
    }
    let forced_end = |g: usize| forced.iter().any(|&(x, y)| x == g || y == g);
    for g in 0..gadget.adj.len() {
        if removed[g] && !forced_end(g) {
            malt[g] = gadget.mate0[g];
        }
    }
    malt
}

/// Whether a stable witness cycle through the tested double-bond edge
/// exists (FG7, layer (f)): an alternative perfect matching without the
/// edge whose `M Δ M'` cycle avoids every unstable-owned gadget vertex.
/// Decided by one constrained blossom search over the remainder of
/// [`stable_remainder`] (forbidden vertices deleted, stored doubles touching
/// them forced), so the answer is existential — independent of which witness
/// the unconstrained search happened to find first — and every returned
/// matching passes the same validation gates as the first search.
fn stable_alternative_double(
    pu: usize,
    pv: usize,
    gadget: &Gadget,
    residual: &[u8],
) -> bool {
    let Some(forbidden) = forbidden_vertices(gadget, residual, pu, pv) else {
        return false;
    };
    let forced = forced_stored_pairs(gadget, &forbidden);
    debug_assert!(
        !forced.iter().any(|&(x, y)| x == pu && y == pv || x == pv && y == pu),
        "stable double search: the tested edge is never forced"
    );
    let mut removed = forbidden.clone();
    for &(x, y) in &forced {
        removed[x] = true;
        removed[y] = true;
    }
    // The tested edge stays removed (as in the first search).
    let (mut adjr, mut seed, new_of, inv) =
        stable_remainder(gadget, &forbidden, &forced, &[pu, pv]);
    let (Some(npu), Some(npv)) = (new_of[pu], new_of[pv]) else {
        return false;
    };
    adjr[npu].retain(|&x| x != npv);
    adjr[npv].retain(|&x| x != npu);
    debug_assert!(
        seed_covers_remainder(gadget, &new_of, &seed, &[pu, pv]),
        "stable double search: the restricted seed covers the remainder but the freed ports"
    );
    blossom::complete(&adjr, &mut seed);
    match classify_constrained_result(
        blossom::is_valid_matching(&adjr, &seed),
        blossom::is_perfect(&seed),
    ) {
        SearchOutcome::Invalid => {
            // The constrained search returned a matching that fails
            // validation: loud in tests, UNDECIDED (no stable witness) in
            // release — exactly as the `SearchOutcome` split on the main path.
            debug_assert!(
                false,
                "stable double search: remainder matching must be a valid matching of real edges"
            );
            return false;
        }
        SearchOutcome::NoAlternative => return false,
        SearchOutcome::Candidate => {}
    }
    if seed[npu] == Some(npv) {
        return false;
    }
    let malt = lift_stable(gadget, &new_of, &inv, &seed, &forced, &removed);
    if !blossom::is_valid_matching(&gadget.adj, &malt) {
        debug_assert!(
            false,
            "stable double search: lifted matching must be a valid matching of real edges"
        );
        return false;
    }
    if !(blossom::is_perfect(&malt) && malt[pu] != Some(pv)) {
        return false;
    }
    // The witness must validate (as in the first search) and avoid unstable
    // owners (by construction; rechecked explicitly — soundness never rests
    // on the construction alone).
    debug_assert!(
        witness_cycle(&gadget.mate0, &malt, (pu, pv))
            .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &malt, c))
            .is_some(),
        "stable double search: witness cycle must validate"
    );
    witness_cycle(&gadget.mate0, &malt, (pu, pv))
        .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &malt, c))
        .is_some_and(|c| c.iter().all(|&g| residual[gadget.owner[g]] <= 1))
}

/// Whether a stable witness cycle through the tested single-bond edge
/// exists (FG7, layer (f)): the remainder analogue of
/// [`stable_alternative_double`] for the forcing test — the tested edge plus
/// the stored doubles touching unstable atoms are forced, the freed cores
/// are the designated free vertices, and the lifted matching must contain
/// the tested edge with a stable `M Δ M'` cycle through it.
fn stable_alternative_single(
    pu: usize,
    pv: usize,
    gadget: &Gadget,
    residual: &[u8],
) -> bool {
    let Some(forbidden) = forbidden_vertices(gadget, residual, pu, pv) else {
        return false;
    };
    let (Some(cu), Some(cv)) = (gadget.mate0[pu], gadget.mate0[pv]) else {
        return false;
    };
    let mut forced = vec![(pu, pv)];
    forced.extend(forced_stored_pairs(gadget, &forbidden));
    let mut removed = forbidden.clone();
    for &(x, y) in &forced {
        removed[x] = true;
        removed[y] = true;
    }
    let (adjr, mut seed, new_of, inv) =
        stable_remainder(gadget, &forbidden, &forced, &[cu, cv]);
    if new_of[cu].is_none() || new_of[cv].is_none() {
        return false;
    }
    debug_assert!(
        seed_covers_remainder(gadget, &new_of, &seed, &[cu, cv]),
        "stable single search: the restricted seed covers the remainder but the freed cores"
    );
    blossom::complete(&adjr, &mut seed);
    match classify_constrained_result(
        blossom::is_valid_matching(&adjr, &seed),
        blossom::is_perfect(&seed),
    ) {
        SearchOutcome::Invalid => {
            // The constrained search returned a matching that fails
            // validation: loud in tests, UNDECIDED (no stable witness) in
            // release — exactly as the `SearchOutcome` split on the main path.
            debug_assert!(
                false,
                "stable single search: remainder matching must be a valid matching of real edges"
            );
            return false;
        }
        SearchOutcome::NoAlternative => return false,
        SearchOutcome::Candidate => {}
    }
    let malt = lift_stable(gadget, &new_of, &inv, &seed, &forced, &removed);
    if !blossom::is_valid_matching(&gadget.adj, &malt) {
        debug_assert!(
            false,
            "stable single search: lifted matching must be a valid matching of real edges"
        );
        return false;
    }
    if !(blossom::is_perfect(&malt) && malt[pu] == Some(pv)) {
        return false;
    }
    debug_assert!(
        witness_cycle(&gadget.mate0, &malt, (pu, pv))
            .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &malt, c))
            .is_some(),
        "stable single search: witness cycle must validate"
    );
    witness_cycle(&gadget.mate0, &malt, (pu, pv))
        .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &malt, c))
        .is_some_and(|c| c.iter().all(|&g| residual[gadget.owner[g]] <= 1))
}

/// Delocalisation test for a double bond of `G'`: its port–port edge is in
/// `M`; the bond is delocalised iff the gadget minus that edge still has a
/// perfect matching (one augmenting search between the freed ports).
///
/// The alternative matching is validated (real edges, perfect, tested edge
/// absent); the witness is the validated `M Δ M'` cycle through the edge,
/// decided only over stable owner atoms (open valence ≤ 1: `d` final, no
/// second double or triple outside). Validation failure is loud in tests and
/// UNDECIDED in release — never delocalised, never fixed.
fn test_gadget_double(
    idx: usize,
    pu: usize,
    pv: usize,
    gadget: &Gadget,
    adj: &[Vec<(usize, u8, usize)>],
    residual: &[u8],
) -> BondStatus {
    let mut adj2 = gadget.adj.clone();
    adj2[pu].retain(|&x| x != pv);
    adj2[pv].retain(|&x| x != pu);
    let mut m2 = gadget.mate0.clone();
    m2[pu] = None;
    m2[pv] = None;
    blossom::complete(&adj2, &mut m2);
    match classify_search_result(
        blossom::is_valid_matching(&adj2, &m2),
        blossom::is_perfect(&m2),
    ) {
        SearchOutcome::Invalid => {
            // The search returned a matching that fails validation: loud
            // in tests, UNDECIDED in release — never fixed.
            debug_assert!(
                false,
                "gadget double test: alternative matching must be a valid matching of real edges"
            );
            return BondStatus::Unknown;
        }
        SearchOutcome::NoAlternative => {
            // A valid matching that is not perfect: confirm the negative
            // with an independent from-scratch matching (FG7 belt-and-braces:
            // a disagreement is loud in tests and UNDECIDED in release).
            if !confirm_no_perfect(&adj2) {
                return BondStatus::Unknown;
            }
            // No alternative perfect matching without `e` exists, so the bond
            // is fixed inside; the fragment still needs the no-outside-cycle
            // check.
            return if fixed_decided(gadget.owner[pu], gadget.owner[pv], 2,
                adj,
                residual,
                idx,
            ) {
                BondStatus::Fixed
            } else {
                BondStatus::Unknown
            };
        }
        SearchOutcome::Candidate => {}
    }
    debug_assert!(
        m2[pu] != Some(pv),
        "gadget double test: the tested edge must be absent from the alternative matching"
    );
    if m2[pu] == Some(pv) {
        return BondStatus::Unknown;
    }
    // Witness cycle = the validated `M Δ M'` component through the edge.
    let ok = witness_cycle(&gadget.mate0, &m2, (pu, pv))
        .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &m2, c))
        .is_some_and(|c| {
            c.iter().all(|&g| residual[gadget.owner[g]] <= 1)
        });
    debug_assert!(
        witness_cycle(&gadget.mate0, &m2, (pu, pv))
            .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &m2, c))
            .is_some(),
        "gadget double test: witness cycle must validate"
    );
    if ok {
        BondStatus::Delocalised
    } else if !first_witness_only() && stable_alternative_double(pu, pv, gadget, residual) {
        // FG7 (f): the first witness routed through atoms whose `d` may
        // still grow outside, but a stable witness exists (decided
        // existentially, so the verdict cannot depend on which witness the
        // first search happened to find).
        BondStatus::Delocalised
    } else {
        BondStatus::Unknown
    }
}

/// Delocalisation test for a single bond of `G'`: force its port–port edge
/// (remove both ports, freeing their matched cores) and search a perfect
/// matching of the remainder (one augmenting search between the freed
/// cores). The full alternative matching is validated (real edges, perfect
/// on the gadget, tested edge present); the witness and stability rule are
/// those of [`test_gadget_double`].
fn test_gadget_single(
    idx: usize,
    pu: usize,
    pv: usize,
    gadget: &Gadget,
    adj: &[Vec<(usize, u8, usize)>],
    residual: &[u8],
) -> BondStatus {
    let gn = gadget.adj.len();
    // Reindex the gadget minus {pu, pv}.
    let mut new_of: Vec<Option<usize>> = vec![None; gn];
    let mut count = 0usize;
    for g in 0..gn {
        if g != pu && g != pv {
            new_of[g] = Some(count);
            count += 1;
        }
    }
    let mut adj3: Vec<Vec<usize>> = vec![Vec::new(); count];
    for g in 0..gn {
        if let Some(ng) = new_of[g] {
            for &w in &gadget.adj[g] {
                if let Some(nw) = new_of[w] {
                    adj3[ng].push(nw);
                }
            }
        }
    }
    // The ports of a single bond are matched to cores; those cores are freed
    // and every other vertex keeps its partner (which survives: `M` pairs
    // inside the remainder).
    let (Some(cu), Some(cv)) = (gadget.mate0[pu], gadget.mate0[pv]) else {
        debug_assert!(false, "gadget single test: single-bond ports match cores");
        return BondStatus::Unknown;
    };
    debug_assert!(
        !gadget.is_core[pu] && !gadget.is_core[pv] && gadget.is_core[cu] && gadget.is_core[cv],
        "gadget single test: ports match cores"
    );
    let mut m3: Vec<Option<usize>> = vec![None; count];
    for g in 0..gn {
        if let (Some(ng), Some(mp)) = (new_of[g], gadget.mate0[g]) {
            if g == cu || g == cv {
                m3[ng] = None;
            } else {
                m3[ng] = new_of[mp];
            }
        }
    }
    let (Some(_ncu), Some(_ncv)) = (new_of[cu], new_of[cv]) else {
        debug_assert!(false, "gadget single test: freed cores survive");
        return BondStatus::Unknown;
    };
    blossom::complete(&adj3, &mut m3);
    match classify_search_result(
        blossom::is_valid_matching(&adj3, &m3),
        blossom::is_perfect(&m3),
    ) {
        SearchOutcome::Invalid => {
            // The search returned a matching that fails validation: loud
            // in tests, UNDECIDED in release — never fixed.
            debug_assert!(
                false,
                "gadget single test: remainder matching must be a valid matching of real edges"
            );
            return BondStatus::Unknown;
        }
        SearchOutcome::NoAlternative => {
            // A valid matching that is not perfect: confirm the negative
            // with an independent from-scratch matching (FG7 belt-and-braces:
            // a disagreement is loud in tests and UNDECIDED in release).
            if !confirm_no_perfect(&adj3) {
                return BondStatus::Unknown;
            }
            // Forcing the edge leaves no perfect matching of the remainder,
            // so the bond is fixed inside (subject to the fragment
            // decided-rule).
            return if fixed_decided(gadget.owner[pu], gadget.owner[pv], 1,
                adj,
                residual,
                idx,
            ) {
                BondStatus::Fixed
            } else {
                BondStatus::Unknown
            };
        }
        SearchOutcome::Candidate => {}
    }
    // Lift to the full alternative matching by re-adding the forced edge.
    let mut inv = vec![0usize; count];
    for g in 0..gn {
        if let Some(ng) = new_of[g] {
            inv[ng] = g;
        }
    }
    let mut malt: Vec<Option<usize>> = vec![None; gn];
    for g in 0..gn {
        if let Some(ng) = new_of[g] {
            malt[g] = m3[ng].map(|nmp| inv[nmp]);
        }
    }
    malt[pu] = Some(pv);
    malt[pv] = Some(pu);
    if !(blossom::is_valid_matching(&gadget.adj, &malt)
        && blossom::is_perfect(&malt)
        && malt[pu] == Some(pv))
    {
        debug_assert!(
            false,
            "gadget single test: lifted matching must be a validated perfect matching with the edge"
        );
        return BondStatus::Unknown;
    }
    let ok = witness_cycle(&gadget.mate0, &malt, (pu, pv))
        .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &malt, c))
        .is_some_and(|c| {
            c.iter().all(|&g| residual[gadget.owner[g]] <= 1)
        });
    debug_assert!(
        witness_cycle(&gadget.mate0, &malt, (pu, pv))
            .filter(|c| valid_witness(&gadget.adj, &gadget.mate0, &malt, c))
            .is_some(),
        "gadget single test: witness cycle must validate"
    );
    if ok {
        BondStatus::Delocalised
    } else if !first_witness_only() && stable_alternative_single(pu, pv, gadget, residual) {
        // FG7 (f): as in the double test, the stable witness is decided
        // existentially.
        BondStatus::Delocalised
    } else {
        BondStatus::Unknown
    }
}


/// Whether a bond judged fixed on inside orders is fixed in every completion
/// consistent with the atom types, hydrogen counts and open valences.
///
/// Checks the sound sufficient condition of the module docs. An end of the
/// bond is **sealed** when no alternating exit is possible there: the end
/// cannot take its required first step (opposite order to the bond: a single
/// needs residual ≥ 1, a double needs residual ≥ 2) directly outside, and no
/// alternating path — first step of the opposite order, then strictly
/// alternating single/double, never reusing the bond — through inside edges
/// reaches an atom with open valence. The bond is decided fixed when at
/// least one end is sealed: any alternating cycle through it in any
/// completion must leave through both ends, so a sealed end leaves only an
/// inside alternating cycle (contradicting the failed matching test) — cut
/// any outside cycle at its first exit to get the inside path. The path
/// search is an exact breadth-first search over (atom, expected-order)
/// states, with no length bound. In particular a bond with a closed
/// degree-one endpoint (such as an exocyclic C=O with its oxygen leaf
/// closed) is always decided fixed, even in fragments.
fn fixed_decided(
    a: usize,
    b: usize,
    order: u8,
    adj: &[Vec<(usize, u8, usize)>],
    residual: &[u8],
    skip: usize,
) -> bool {
    /// Whether any alternating exit is possible from end `s`.
    fn exit_possible(
        s: usize,
        need: u8,
        adj: &[Vec<(usize, u8, usize)>],
        residual: &[u8],
        skip: usize,
    ) -> bool {
        if residual[s] >= need {
            return true;
        }
        let n = adj.len();
        // seen[v][o]: atom v reached needing an edge of order o next.
        let mut seen = vec![[false; 4]; n];
        let mut queue = std::collections::VecDeque::new();
        seen[s][usize::from(need)] = true;
        queue.push_back((s, need));
        while let Some((v, exp)) = queue.pop_front() {
            for (nbr, o, bi) in &adj[v] {
                if *bi == skip || *o != exp {
                    continue;
                }
                if residual[*nbr] > 0 {
                    return true;
                }
                let next = 3 - exp;
                if !seen[*nbr][next as usize] {
                    seen[*nbr][next as usize] = true;
                    queue.push_back((*nbr, next));
                }
            }
        }
        false
    }
    let need: u8 = if order == 2 { 1 } else { 2 };
    // Decided fixed iff at least one end is sealed.
    !exit_possible(a, need, adj, residual, skip) || !exit_possible(b, need, adj, residual, skip)
}

/// Edmonds' blossom algorithm for maximum matching in general graphs.
///
/// A matching is a set of edges without shared vertices; an augmenting path
/// alternates non-matching/matching edges between two free (unmatched)
/// vertices, and augmenting along one grows the matching by one edge. The
/// naive alternating-tree search fails on odd cycles ("blossoms": an odd
/// cycle reached by an alternating path from a free vertex, where the search
/// enters the cycle at its base and cannot decide which way around to leave);
/// Edmonds' algorithm contracts each blossom to a single vertex, continues
/// the search on the contracted graph, and lifts the path on augmentation.
/// Odd cycles occur in molecules (five-rings, azulene), so a bipartite
/// matcher would be wrong.
///
/// This is the standard Edmonds formulation with `base[]`, `parent[]`,
/// `used[]` and `blossom[]` sets, an LCA walk over bases, and
/// `mark_path(v, b, child)` setting `parent[v] = child` while walking
/// `v → base` (the classical O(V^3) form), kept self-contained in this
/// module with no new dependency. The search never contributes a
/// reconstructed path as evidence: every decision in this crate rests on a
/// validated perfect matching (see [`is_valid_matching`]), and every witness
/// cycle is re-derived from validated mate arrays (see [`witness_cycle`]).
///
/// Complexity: each augmenting search scans every edge once per contracted
/// base (blossom contraction/expansion is linear per scan), i.e. O(n·m) time
/// and O(n + m) memory; a maximum matching needs at most n/2 augmentations.
/// The gadget graphs here hold tens to hundreds of vertices, and every
/// delocalisation test is a single search from a near-perfect matching, so
/// this is easily fast enough.
mod blossom {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Searches (calls to [`find_path`]) that performed at least one blossom
    /// contraction since the last reset.
    static SEARCHES_WITH_CONTRACTION: AtomicU64 = AtomicU64::new(0);
    /// Searches that performed at least two contractions (a contraction
    /// inside an already-contracted search: nested blossoms) since reset.
    static SEARCHES_NESTED: AtomicU64 = AtomicU64::new(0);

    /// Reset the blossom-search statistics counters.
    pub(super) fn reset_stats() {
        SEARCHES_WITH_CONTRACTION.store(0, Ordering::Relaxed);
        SEARCHES_NESTED.store(0, Ordering::Relaxed);
    }

    /// `(searches_with_contraction, searches_nested)` since the last reset.
    pub(super) fn stats() -> (u64, u64) {
        (
            SEARCHES_WITH_CONTRACTION.load(Ordering::Relaxed),
            SEARCHES_NESTED.load(Ordering::Relaxed),
        )
    }

    /// Lowest common ancestor of `a` and `b` in the alternating forest,
    /// comparing base representatives.
    ///
    /// Marks the base-walk from `a` up to the free root, then walks `b` up
    /// to the first marked base. Defensive fallbacks (never panicking on
    /// unexpected forest shapes) return the current base; the validation
    /// layer above rejects any matching built on a wrong contraction.
    fn lca(
        mate: &[Option<usize>],
        base: &[usize],
        parent: &[Option<usize>],
        mut a: usize,
        mut b: usize,
    ) -> usize {
        let n = mate.len();
        let mut used = vec![false; n];
        loop {
            a = base[a];
            used[a] = true;
            match mate[a] {
                None => break,
                Some(m) => match parent[m] {
                    None => break,
                    Some(p) => a = p,
                },
            }
        }
        loop {
            b = base[b];
            if used[b] {
                return b;
            }
            match mate[b] {
                None => return b,
                Some(mb) => match parent[mb] {
                    None => return b,
                    Some(p) => b = p,
                },
            }
        }
    }

    /// Mark the blossom on the path from `v` up to base `b`.
    ///
    /// Classical form: walks `v → base` following matched edges upward,
    /// setting `parent[v] = child` where `child` starts as the closing-edge
    /// endpoint and then propagates the matched vertex at each step, so the
    /// lifted augmenting path stays alternating in the original graph.
    /// Defensive breaks (never panicking) leave the contraction partial; the
    /// validation layer above rejects any matching built on it.
    fn mark_path(
        mate: &[Option<usize>],
        base: &[usize],
        in_blossom: &mut [bool],
        parent: &mut [Option<usize>],
        mut v: usize,
        b: usize,
        mut child: usize,
    ) {
        while base[v] != b {
            let Some(m) = mate[v] else { break };
            in_blossom[base[v]] = true;
            in_blossom[base[m]] = true;
            parent[v] = Some(child);
            child = m;
            let Some(pm) = parent[m] else { break };
            v = pm;
        }
    }

    /// Search an augmenting path from the free vertex `root`, contracting
    /// blossoms as they appear.
    ///
    /// On success augments `mate` along the path and returns the free
    /// endpoint reached. On failure `mate` is unchanged. The returned vertex
    /// is informational only: callers must validate the resulting matching
    /// (see [`is_valid_matching`]) and never use a search-internal vertex
    /// list as evidence.
    fn find_path(
        adj: &[Vec<usize>],
        mate: &mut [Option<usize>],
        root: usize,
    ) -> Option<usize> {
        let n = adj.len();
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut base: Vec<usize> = (0..n).collect();
        let mut used = vec![false; n];
        let mut in_blossom = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        used[root] = true;
        queue.push_back(root);
        // Contractions in THIS search: the first marks a blossom search,
        // the second a nested one (counted through the statistics hook).
        let mut contractions = 0usize;
        while let Some(v) = queue.pop_front() {
            for &to in &adj[v] {
                if base[v] == base[to] || mate[v] == Some(to) {
                    continue;
                }
                if to == root || (mate[to].is_some_and(|mt| parent[mt].is_some())) {
                    // Odd cycle: contract the blossom with the closing edge
                    // `(v, to)`; each side walks up to the base carrying the
                    // other side's endpoint as its initial child.
                    let b = lca(mate, &base, &parent, v, to);
                    in_blossom.fill(false);
                    mark_path(mate, &base, &mut in_blossom, &mut parent, v, b, to);
                    mark_path(mate, &base, &mut in_blossom, &mut parent, to, b, v);
                    contractions += 1;
                    if contractions == 1 {
                        SEARCHES_WITH_CONTRACTION.fetch_add(1, Ordering::Relaxed);
                    } else if contractions == 2 {
                        SEARCHES_NESTED.fetch_add(1, Ordering::Relaxed);
                    }
                    for i in 0..n {
                        if in_blossom[base[i]] {
                            base[i] = b;
                            if !used[i] {
                                used[i] = true;
                                queue.push_back(i);
                            }
                        }
                    }
                } else if parent[to].is_none() {
                    parent[to] = Some(v);
                    match mate[to] {
                        None => {
                            // Collect the parent/mate chain from `to` back
                            // to `root` BEFORE mutating `mate` (non-matching
                            // parent links alternate with matching mate
                            // links), then augment atomically along it. A
                            // chain that does not close at the root is never
                            // used as evidence: the search simply continues.
                            let mut chain = vec![to];
                            let mut cur = to;
                            let closed = loop {
                                let Some(pv) = parent[cur] else { break false };
                                chain.push(pv);
                                if pv == root {
                                    break true;
                                }
                                let Some(nv) = mate[pv] else { break false };
                                chain.push(nv);
                                cur = nv;
                                if chain.len() > 2 * n + 2 {
                                    break false;
                                }
                            };
                            if closed && chain.len() % 2 == 0 {
                                for w in chain.chunks_exact(2) {
                                    mate[w[0]] = Some(w[1]);
                                    mate[w[1]] = Some(w[0]);
                                }
                                return Some(to);
                            }
                        }
                        Some(m) => {
                            if !used[m] {
                                used[m] = true;
                                queue.push_back(m);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Grow `mate` in place until no augmenting search from a free vertex
    /// succeeds (repeated passes over the free vertices).
    pub(super) fn complete(adj: &[Vec<usize>], mate: &mut [Option<usize>]) {
        loop {
            let mut progressed = false;
            for r in 0..adj.len() {
                if mate[r].is_none() && find_path(adj, mate, r).is_some() {
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
    }

    /// Maximum matching from an empty start (repeated search until stuck).
    fn max_matching(n: usize, adj: &[Vec<usize>]) -> Vec<Option<usize>> {
        let mut mate: Vec<Option<usize>> = vec![None; n];
        complete(adj, &mut mate);
        mate
    }

    /// Whether `mate` is a consistent matching whose every pair is a real
    /// edge of `adj`: symmetric pairs, no self-pairs, each pair adjacent.
    ///
    /// This is the evidence gate for every decision in this crate: a search
    /// result is trusted only when this holds (plus perfectness on the
    /// intended vertex set, checked by the caller).
    pub(super) fn is_valid_matching(adj: &[Vec<usize>], mate: &[Option<usize>]) -> bool {
        if mate.len() != adj.len() {
            return false;
        }
        for (v, m) in mate.iter().enumerate() {
            if let Some(w) = m {
                if *w == v || mate[*w] != Some(v) {
                    return false;
                }
                if !adj[v].contains(w) {
                    return false;
                }
            }
        }
        true
    }

    /// Whether `mate` covers every vertex.
    pub(super) fn is_perfect(mate: &[Option<usize>]) -> bool {
        mate.iter().all(|m| m.is_some())
    }

    /// Expose the matcher core for the public test helpers below.
    pub(super) fn core_max_matching(n: usize, adj: &[Vec<usize>]) -> Vec<Option<usize>> {
        max_matching(n, adj)
    }
}

/// Build simple-graph adjacency from unordered pairs, skipping self-loops.
fn build_adj(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (a, b) in edges {
        if *a >= n || *b >= n || *a == *b {
            continue;
        }
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    adj
}

/// A maximum matching of the general graph with `n` vertices and `edges`
/// (unordered pairs), as sorted unordered vertex pairs.
///
/// Decided with Edmonds' blossom algorithm (see `blossom`), exact on graphs
/// with odd cycles; the result is order-independent (adjacency order must not
/// matter). The returned pairs are validated (disjoint, real edges) with a
/// `debug_assert!`. Exposed publicly so integration tests can check maximum
/// size and matching validity against brute-force enumeration on abstract
/// graphs; the detector applies the same search to every gadget graph.
pub fn maximum_matching(n: usize, edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let adj = build_adj(n, edges);
    let mate = blossom::core_max_matching(n, &adj);
    debug_assert!(
        blossom::is_valid_matching(&adj, &mate),
        "maximum_matching: result must be a valid matching of real edges"
    );
    let mut out = Vec::new();
    for v in 0..n {
        if let Some(w) = mate[v] {
            if v < w {
                out.push((v, w));
            }
        }
    }
    out.sort_unstable();
    out
}

/// Whether the general graph with `n` vertices and `edges` (unordered pairs)
/// has a perfect matching.
///
/// Decided with Edmonds' blossom algorithm (see `blossom`), exact on graphs
/// with odd cycles. Exposed publicly so integration tests can check the
/// matcher's decisions against brute-force enumeration on abstract graphs
/// that are not realizable as molecules; the detector applies the same search
/// to every gadget graph.
pub fn has_perfect_matching(n: usize, edges: &[(usize, usize)]) -> bool {
    if n % 2 == 1 {
        return false;
    }
    let adj = build_adj(n, edges);
    let mate = blossom::core_max_matching(n, &adj);
    debug_assert!(
        blossom::is_valid_matching(&adj, &mate),
        "has_perfect_matching: result must be a valid matching of real edges"
    );
    blossom::is_perfect(&mate)
}

/// For a general graph with `n` vertices, `edges` and a known perfect
/// matching `matching` (unordered vertex pairs), flag per edge of `edges`
/// whether the edge is allowed: in some but not all perfect matchings.
///
/// Each edge is tested exactly as the detector tests a gadget bond: an edge
/// of the matching is allowed iff the graph minus that edge still has a
/// perfect matching (one augmenting search between its freed endpoints); any
/// other edge `(u, v)` is allowed iff forcing it (removing `u` and `v`,
/// freeing their partners) still admits a perfect matching of the remainder.
/// A decision is `true` only for a validated perfect matching with the
/// tested edge absent (matched case) or present after re-adding (unmatched
/// case); anything else reports `false`. Edges in no perfect matching and
/// edges in every perfect matching both report `false`. Pass an empty
/// `matching` when no perfect matching exists (every edge then reports
/// `false`). Integration tests check these decisions against brute-force
/// enumeration of all perfect matchings.
pub fn matching_allowed_edges(
    n: usize,
    edges: &[(usize, usize)],
    matching: &[(usize, usize)],
) -> Vec<bool> {
    let adj = build_adj(n, edges);
    let mut mate: Vec<Option<usize>> = vec![None; n];
    for (a, b) in matching {
        if *a < n && *b < n && *a != *b {
            mate[*a] = Some(*b);
            mate[*b] = Some(*a);
        }
    }
    debug_assert!(
        blossom::is_valid_matching(&adj, &mate)
            && (blossom::is_perfect(&mate) || matching.is_empty()),
        "matching_allowed_edges: the given matching must be a perfect matching of real edges (or empty when none exists)"
    );
    if !(blossom::is_valid_matching(&adj, &mate)
        && (blossom::is_perfect(&mate) || matching.is_empty()))
    {
        // Invalid input (never in tests): report nothing allowed, never a
        // wrong `true`.
        return vec![false; edges.len()];
    }
    let is_matched = |u: usize, v: usize| mate[u] == Some(v);
    edges
        .iter()
        .map(|(u, v)| {
            if *u >= n || *v >= n || *u == *v {
                return false;
            }
            if is_matched(*u, *v) {
                let mut adj2 = adj.clone();
                adj2[*u].retain(|&x| x != *v);
                adj2[*v].retain(|&x| x != *u);
                let mut m2 = mate.clone();
                m2[*u] = None;
                m2[*v] = None;
                blossom::complete(&adj2, &mut m2);
                // Validated: perfect on the vertex set, real edges, and the
                // tested edge absent.
                blossom::is_valid_matching(&adj2, &m2)
                    && blossom::is_perfect(&m2)
                    && m2[*u] != Some(*v)
            } else {
                // Both endpoints must be matched (the matching is perfect);
                // otherwise the edge is in no perfect matching.
                let (Some(pu), Some(pv)) = (mate[*u], mate[*v]) else {
                    return false;
                };
                let mut new_of: Vec<Option<usize>> = vec![None; n];
                let mut count = 0usize;
                for h in 0..n {
                    if h != *u && h != *v {
                        new_of[h] = Some(count);
                        count += 1;
                    }
                }
                let mut adj3: Vec<Vec<usize>> = vec![Vec::new(); count];
                for h in 0..n {
                    if let Some(nh) = new_of[h] {
                        for &w in &adj[h] {
                            if let Some(nw) = new_of[w] {
                                adj3[nh].push(nw);
                            }
                        }
                    }
                }
                let mut m3: Vec<Option<usize>> = vec![None; count];
                for h in 0..n {
                    if let (Some(nh), Some(mp)) = (new_of[h], mate[h]) {
                        if h == pu || h == pv {
                            m3[nh] = None;
                        } else {
                            m3[nh] = new_of[mp];
                        }
                    }
                }
                if new_of[pu].is_none() || new_of[pv].is_none() {
                    return false;
                }
                blossom::complete(&adj3, &mut m3);
                if !(blossom::is_valid_matching(&adj3, &m3) && blossom::is_perfect(&m3)) {
                    return false;
                }
                // Lift by re-adding the forced edge and validate on the full
                // graph: perfect, real edges, tested edge present.
                let mut inv = vec![0usize; count];
                for h in 0..n {
                    if let Some(nh) = new_of[h] {
                        inv[nh] = h;
                    }
                }
                let mut malt: Vec<Option<usize>> = vec![None; n];
                for h in 0..n {
                    if let Some(nh) = new_of[h] {
                        malt[h] = m3[nh].map(|nmp| inv[nmp]);
                    }
                }
                malt[*u] = Some(*v);
                malt[*v] = Some(*u);
                blossom::is_valid_matching(&adj, &malt)
                    && blossom::is_perfect(&malt)
                    && malt[*u] == Some(*v)
            }
        })
        .collect()
}

/// Reset the blossom-search statistics counters (see [`blossom_stats`]).
/// Test hook only: the counters are process-global.
#[doc(hidden)]
pub fn reset_blossom_stats() {
    blossom::reset_stats();
}

/// Blossom-search statistics since the last reset: `(searches_with_contraction,
/// searches_nested)` — how many augmenting-path searches performed at least
/// one blossom contraction, and how many performed at least two (a
/// contraction inside an already-contracted search: nested blossoms).
/// Process-global: parallel tests also contribute. Test hook only.
#[doc(hidden)]
pub fn blossom_stats() -> (u64, u64) {
    blossom::stats()
}

/// Tutte-gadget size `(vertices, edges)` for `graph`: the perfect-matching
/// instance the detector searches per bond test. Exposed so tests can report
/// the reduction size on the largest fixture molecules.
pub fn kekule_gadget_size(graph: &MolGraph) -> (usize, usize) {
    let ends: Vec<(usize, usize)> = graph.bonds().iter().map(|(a, b, _)| (*a, *b)).collect();
    let orders: Vec<u8> = graph.bonds().iter().map(|(_, _, o)| *o).collect();
    let g = Gadget::build(graph.atoms().len(), &ends, &orders);
    let e: usize = g.adj.iter().map(|v| v.len()).sum::<usize>() / 2;
    (g.adj.len(), e)
}

/// Per-bond delocalisation flags in [`MolGraph::bonds`] order, decided by the
/// exact gadget matching rule (no length bound, no work limit).
/// Triple bonds are always reported fixed (`false`).
pub fn delocalised_bonds(graph: &MolGraph) -> Vec<bool> {
    decided_bonds(graph)
        .into_iter()
        .map(|s| s == Some(true))
        .collect()
}

/// Per-bond decided statuses in [`MolGraph::bonds`] order: `Some(true)` is a
/// decided delocalised bond, `Some(false)` a decided fixed bond (triples are
/// always `Some(false)`), and `None` is undecided in a fragment (some
/// completion disagrees; every instance using or consulting the bond is then
/// undetermined). On closed graphs (every residual valence 0) every entry is
/// decided. Exposed so tests can check fragment decided-status soundness
/// bond by bond.
pub fn decided_bonds(graph: &MolGraph) -> Vec<Option<bool>> {
    let view = View::of(graph).expect("decided_bonds: validated graphs hold known atom types");
    view.status
        .iter()
        .map(|s| match s {
            BondStatus::Delocalised => Some(true),
            BondStatus::Fixed => Some(false),
            BondStatus::Unknown => None,
        })
        .collect()
}

/// Distinct six-membered C/N rings whose six ring bonds are all delocalised
/// (sorted atom sets, one per ring). Naphthalene yields two in every kekulé
/// form; benzoquinone and cyclooctatetraene yield none.
pub fn arene_rings(graph: &MolGraph) -> Vec<Vec<usize>> {
    let view = View::of(graph).expect("arene_rings: validated graphs hold known atom types");
    find_arene_rings(&view)
}

/// Distinct heteroaromatic five-rings (sorted atom sets, one per ring); see
/// the module docs for the rule.
pub fn hetero_five_rings(graph: &MolGraph) -> Vec<Vec<usize>> {
    let view = View::of(graph).expect("hetero_five_rings: validated graphs hold known atom types");
    find_five_rings(&view)
}

/// Enumerate simple six-cycles of C/N atoms whose six bonds are all
/// delocalised.
fn find_arene_rings(view: &View<'_>) -> Vec<Vec<usize>> {
    let n = view.atoms.len();
    let mut out: BTreeSet<Vec<usize>> = BTreeSet::new();
    // Depth-first simple cycles of length 6 from each smallest atom.
    for s in 0..n {
        if view.atoms[s].element != C && view.atoms[s].element != N {
            continue;
        }
        let mut path = vec![s];
        let mut vis = vec![false; n];
        vis[s] = true;
        dfs_six(view, s, s, &mut path, &mut vis, &mut out);
    }
    out.into_iter().collect()
}

#[allow(clippy::too_many_arguments)]
fn dfs_six(
    view: &View<'_>,
    start: usize,
    cur: usize,
    path: &mut Vec<usize>,
    vis: &mut [bool],
    out: &mut BTreeSet<Vec<usize>>,
) {
    if path.len() == 6 {
        // Close the cycle: cur must bond to start.
        let Some(bi) = view.bond_idx(cur, start) else {
            return;
        };
        if view.bond_orders[bi] != 1 && view.bond_orders[bi] != 2 {
            return;
        }
        // All six ring bonds delocalised?
        let mut ring = path.clone();
        ring.push(start);
        for w in ring.windows(2) {
            let Some(i) = view.bond_idx(w[0], w[1]) else {
                return;
            };
            if view.status[i] != BondStatus::Delocalised {
                return;
            }
        }
        let mut key = path.clone();
        key.sort_unstable();
        out.insert(key);
        return;
    }
    for (nxt, o, _) in &view.adj[cur] {
        if *o != 1 && *o != 2 {
            continue;
        }
        if *nxt <= start && *nxt != start {
            // Canonicalise by smallest atom: only the smallest starts.
            continue;
        }
        if vis[*nxt] {
            continue;
        }
        let a = view.atoms[*nxt];
        if a.element != C && a.element != N {
            continue;
        }
        vis[*nxt] = true;
        path.push(*nxt);
        dfs_six(view, start, *nxt, path, vis, out);
        path.pop();
        vis[*nxt] = false;
    }
}

/// Enumerate heteroaromatic five-rings: simple five-cycles with exactly one
/// or two heteroatoms (N, O, S) in which every ring atom is either a
/// heteroatom whose two ring bonds are both single, or carries a ring bond
/// that is a fixed double or delocalised (raw double or delocalised single:
/// double in some kekule form — hence kekule-invariant; a fixed single
/// never changes). Rings with an undecided qualifying bond are skipped
/// (conservative: never counted, never excluding, in fragments).
fn find_five_rings(view: &View<'_>) -> Vec<Vec<usize>> {
    find_five_rings2(view).0
}

/// Determined five-rings plus undetermined five-ring candidates (fully
/// inside five-cycles that could qualify if undecided single bonds turn
/// out delocalised in the parent, but do not qualify decidedly).
///
/// Returned rings are sorted atom sets (one per ring); cycle order is not
/// preserved here — callers that need the actual ring bonds must use
/// [`find_five_ring_cycles`], because adjacency in sorted order is NOT ring
/// adjacency (assuming it is makes the detector depend on atom numbering).
fn find_five_rings2(view: &View<'_>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let (yes, maybe) = find_five_ring_cycles(view);
    let mut yes_sorted: Vec<Vec<usize>> = yes
        .iter()
        .map(|r| {
            let mut k = r.clone();
            k.sort_unstable();
            k
        })
        .collect();
    let mut maybe_sorted: Vec<Vec<usize>> = maybe
        .iter()
        .map(|r| {
            let mut k = r.clone();
            k.sort_unstable();
            k
        })
        .collect();
    yes_sorted.sort();
    yes_sorted.dedup();
    maybe_sorted.sort();
    maybe_sorted.dedup();
    // A decided ring is never also reported undecided.
    maybe_sorted.retain(|k| !yes_sorted.contains(k));
    (yes_sorted, maybe_sorted)
}

/// Determined / possible five-rings in cycle order (each ring starts at its
/// smallest atom; one representative cycle per sorted atom set). The ring
/// bonds are consecutive pairs (plus closure) of each returned vector.
/// Derived exclusion sets (five-ring heteroatoms and bonds) must be built
/// from these cycle orders — never from sorted-order adjacency — so the
/// detector depends only on the labelled graph up to isomorphism.
fn find_five_ring_cycles(view: &View<'_>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = view.atoms.len();
    let mut yes_seen: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut maybe_seen: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut yes: Vec<Vec<usize>> = Vec::new();
    let mut maybe: Vec<Vec<usize>> = Vec::new();
    for s in 0..n {
        let mut path = vec![s];
        let mut vis = vec![false; n];
        vis[s] = true;
        dfs_five_cycles(view, s, s, &mut path, &mut vis, &mut yes_seen, &mut maybe_seen, &mut yes, &mut maybe);
    }
    (yes, maybe)
}

fn dfs_five_cycles(
    view: &View<'_>,
    start: usize,
    cur: usize,
    path: &mut Vec<usize>,
    vis: &mut [bool],
    yes_seen: &mut BTreeSet<Vec<usize>>,
    maybe_seen: &mut BTreeSet<Vec<usize>>,
    yes: &mut Vec<Vec<usize>>,
    maybe: &mut Vec<Vec<usize>>,
) {
    if path.len() == 5 {
        let Some(bi) = view.bond_idx(cur, start) else {
            return;
        };
        if view.bond_orders[bi] != 1 && view.bond_orders[bi] != 2 {
            return;
        }
        // Full five atoms; classify the cycle in cycle order.
        let ring: Vec<usize> = path.clone();
        match five_ring_classify(view, &ring) {
            FiveRing::No => {}
            FiveRing::Yes => {
                let mut key = ring.clone();
                key.sort_unstable();
                if yes_seen.insert(key) {
                    yes.push(ring);
                }
            }
            FiveRing::Maybe => {
                let mut key = ring.clone();
                key.sort_unstable();
                if maybe_seen.insert(key) {
                    maybe.push(ring);
                }
            }
        }
        return;
    }
    for (nxt, o, _) in &view.adj[cur] {
        if *o != 1 && *o != 2 {
            continue;
        }
        if *nxt < start {
            continue;
        }
        if vis[*nxt] {
            continue;
        }
        vis[*nxt] = true;
        path.push(*nxt);
        dfs_five_cycles(view, start, *nxt, path, vis, yes_seen, maybe_seen, yes, maybe);
        path.pop();
        vis[*nxt] = false;
    }
}

/// Five-ring classification of a fully-inside five-cycle.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FiveRing {
    /// Impossible in every completion (wrong hetero count, a hetero with
    /// a ring double, or a carbon with only fixed singles).
    No,
    /// Qualifies decidedly (every non-hetero atom carries a raw double or
    /// a delocalised bond).
    Yes,
    /// Could qualify if undecided singles turn out delocalised (every
    /// non-hetero atom carries a decided qualifier or an undecided single,
    /// with at least one of the latter).
    Maybe,
}

/// Classify a five-cycle (in cycle order) under the kekule-invariant rule.
fn five_ring_classify(view: &View<'_>, ring: &[usize]) -> FiveRing {
    debug_assert_eq!(ring.len(), 5);
    let hetero = ring
        .iter()
        .filter(|a| matches!(view.atoms[**a].element, N | O | S))
        .count();
    if hetero != 1 && hetero != 2 {
        return FiveRing::No;
    }
    let mut maybe = false;
    for (k, a) in ring.iter().enumerate() {
        let prev_atom = ring[(k + 4) % 5];
        let next_atom = ring[(k + 1) % 5];
        let Some(pi) = view.bond_idx(prev_atom, *a) else {
            return FiveRing::No;
        };
        let Some(ni) = view.bond_idx(*a, next_atom) else {
            return FiveRing::No;
        };
        let el = view.atoms[*a].element;
        if matches!(el, N | O | S)
            && view.bond_orders[pi] == 1
            && view.bond_orders[ni] == 1
        {
            // Single-only heteroatom (pyrrole/furan/thiophene type).
            continue;
        }
        // Otherwise (carbon or pyridine-type heteroatom): needs a decided
        // qualifier (raw double or delocalised bond) to count, else an
        // undecided single keeps the ring possible-but-undecided.
        let qual = |i: usize| -> u8 {
            // 2 = decided qualifier, 1 = possible (undecided single), 0 = no.
            if view.bond_orders[i] == 2 || view.status[i] == BondStatus::Delocalised {
                2
            } else if view.status[i] == BondStatus::Unknown {
                1
            } else {
                0
            }
        };
        match (qual(pi), qual(ni)) {
            (2, _) | (_, 2) => {}
            _ => {
                if qual(pi) == 1 || qual(ni) == 1 {
                    maybe = true;
                } else {
                    return FiveRing::No;
                }
            }
        }
    }
    if maybe {
        FiveRing::Maybe
    } else {
        FiveRing::Yes
    }
}

// ---------------------------------------------------------------------------
// Generic matcher
// ---------------------------------------------------------------------------

/// Outcome of the exclusion / status check on one fully matched instance.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExclOutcome {
    /// Positively violated: reject outright.
    Violated,
    /// Not violated, but some check is not certain.
    Uncertain,
    /// Every check is certain: the instance is determined.
    Certain,
}

/// Check a pattern's required doubles are fixed and decided, its exclusions
/// certain, its carbonyl-remaining check (if any) certain, and its
/// five-ring interaction decided. `map[p]` is the graph atom of pattern
/// node `p`. `five_hetero`/`five_bond` mark determined five-ring coverage;
/// `maybe_hetero`/`maybe_bond` mark possible-but-undecided coverage (which
/// forces undetermined, never rejection).
fn check_instance(
    view: &View<'_>,
    pattern: &Pattern,
    map: &[usize],
    five_hetero: &[bool],
    five_bond: &BTreeSet<(usize, usize)>,
    maybe_hetero: &[bool],
    maybe_bond: &BTreeSet<(usize, usize)>,
) -> ExclOutcome {
    // Required doubles must be fixed and decided; required triples are
    // always fixed. An undecided double makes the instance undetermined;
    // a delocalised double where a fixed one is required rejects it.
    for b in pattern.bonds {
        if b.order == 2 {
            let g1 = map[b.a];
            let g2 = map[b.b];
            let Some(bi) = view.bond_idx(g1, g2) else {
                return ExclOutcome::Violated;
            };
            match view.status[bi] {
                BondStatus::Fixed => {}
                BondStatus::Delocalised => return ExclOutcome::Violated,
                BondStatus::Unknown => return ExclOutcome::Uncertain,
            }
        }
    }
    // Alkene / imine inside a five-ring are not that type; inside a
    // possible-but-undecided ring they are undetermined.
    if pattern.extra == Extra::Alkene || pattern.extra == Extra::Imine {
        let g1 = map[0];
        let g2 = map[1];
        let key = if g1 < g2 { (g1, g2) } else { (g2, g1) };
        if five_bond.contains(&key) {
            return ExclOutcome::Violated;
        }
        if maybe_bond.contains(&key) {
            return ExclOutcome::Uncertain;
        }
        if near_open(view, &[g1, g2], 2) {
            return ExclOutcome::Uncertain;
        }
    }
    // Carbonyl-remaining C/H check (acid, ester, amide: node 0).
    if pattern.extra == Extra::CarbonylRemaining {
        match carbonyl_remaining(view, map[0], &[map[1], map[2]]) {
            Remaining::Violated => return ExclOutcome::Violated,
            Remaining::Uncertain => return ExclOutcome::Uncertain,
            Remaining::Certain => {}
        }
        // Ester alkoxy exclusion is a generic double exclusion (below);
        // amides/acids have no further checks here.
    }
    // Generic exclusions over fixed doubles / singles.
    let mut out = ExclOutcome::Certain;
    for e in pattern.exclusions {
        let g = map[e.node];
        let min_order = e.orders.iter().min().copied().unwrap_or(1);
        if min_order == 2 {
            // Fixed-double exclusion: a fixed double inside violates; an
            // undecided double to the same elements leaves uncertainty;
            // otherwise residual >= 2 leaves uncertainty.
            let mut violated = false;
            let mut undecided = false;
            for (nbr, _o, _bi) in &view.adj[g] {
                if !e.elements.contains(&view.atoms[*nbr].element) {
                    continue;
                }
                if view.is_fixed_double(g, *nbr) {
                    violated = true;
                    break;
                }
                if view.is_unknown_double(g, *nbr) {
                    undecided = true;
                }
            }
            if violated {
                return ExclOutcome::Violated;
            }
            if undecided || view.residual[g] >= 2 {
                out = ExclOutcome::Uncertain;
            }
        } else {
            // Single exclusion: any such neighbour inside violates;
            // otherwise residual 0 is needed for certainty.
            let mut violated = false;
            for (nbr, o, _) in &view.adj[g] {
                if *o == 1 && e.elements.contains(&view.atoms[*nbr].element) {
                    violated = true;
                    break;
                }
            }
            if violated {
                return ExclOutcome::Violated;
            }
            if view.residual[g] > 0 {
                out = ExclOutcome::Uncertain;
            }
        }
    }
    // Five-ring heteroatom exclusions for secondary/tertiary amine, ether
    // and thioether: the single-bonded heteroatom of such a ring is not
    // that type; of a possible-but-undecided ring it is undetermined.
    match pattern.id {
        8 => {
            // Ether oxygen is node 1.
            let o = map[1];
            if five_hetero[o] {
                return ExclOutcome::Violated;
            }
            if maybe_hetero[o] {
                return ExclOutcome::Uncertain;
            }
            if near_open(view, &[o], 2) {
                return ExclOutcome::Uncertain;
            }
        }
        10 => {
            // Secondary amine nitrogen is node 1.
            let x = map[1];
            if five_hetero[x] {
                return ExclOutcome::Violated;
            }
            if maybe_hetero[x] {
                return ExclOutcome::Uncertain;
            }
            if near_open(view, &[x], 2) {
                return ExclOutcome::Uncertain;
            }
        }
        11 => {
            // Tertiary amine nitrogen is node 0.
            let x = map[0];
            if five_hetero[x] {
                return ExclOutcome::Violated;
            }
            if maybe_hetero[x] {
                return ExclOutcome::Uncertain;
            }
            if near_open(view, &[x], 2) {
                return ExclOutcome::Uncertain;
            }
        }
        17 => {
            // Thioether sulphur is node 1.
            let x = map[1];
            if five_hetero[x] {
                return ExclOutcome::Violated;
            }
            if maybe_hetero[x] {
                return ExclOutcome::Uncertain;
            }
            if near_open(view, &[x], 2) {
                return ExclOutcome::Uncertain;
            }
        }
        _ => {}
    }
    // Thiol carbon exclusion already ran as a generic double exclusion.
    out
}

/// Whether any atom within graph distance `d` of `seeds` has open valence
/// (the seeds themselves included).
fn near_open(view: &View<'_>, seeds: &[usize], d: usize) -> bool {
    let mut seen = vec![false; view.atoms.len()];
    let mut frontier: Vec<usize> = Vec::new();
    for s in seeds {
        if view.residual[*s] > 0 {
            return true;
        }
        seen[*s] = true;
        frontier.push(*s);
    }
    for _ in 0..d {
        let mut next = Vec::new();
        for u in &frontier {
            for (v, _, _) in &view.adj[*u] {
                if seen[*v] {
                    continue;
                }
                seen[*v] = true;
                if view.residual[*v] > 0 {
                    return true;
                }
                next.push(*v);
            }
        }
        frontier = next;
    }
    false
}

/// Outcome of the carbonyl-remaining check.
enum Remaining {
    /// A hetero extra neighbour (or two extra carbons): reject.
    Violated,
    /// No violation inside, but the carbon is open: undecided.
    Uncertain,
    /// Closed with only C/H remainder: determined.
    Certain,
}

/// The carbonyl carbon's remaining substituent must be C or H: no heavy
/// neighbour outside `designated` may be a heteroatom or carry anything but
/// a single bond, and at most one extra carbon is allowed. An open carbon
/// is undecided (outside could supply a heteroatom).
fn carbonyl_remaining(view: &View<'_>, c: usize, designated: &[usize]) -> Remaining {
    let mut extra_c = 0usize;
    for (nbr, o, _) in &view.adj[c] {
        if designated.contains(nbr) {
            continue;
        }
        let el = view.atoms[*nbr].element;
        if el != C || *o != 1 {
            return Remaining::Violated;
        }
        extra_c += 1;
        if extra_c > 1 {
            return Remaining::Violated;
        }
    }
    if view.residual[c] > 0 {
        return Remaining::Uncertain;
    }
    // Closed: consistency follows from valence (extra C ⟺ H0, none ⟺ H1).
    Remaining::Certain
}

/// All distinct anchor sets (sorted graph-atom lists) of one pattern:
/// generic backtracking over injective maps that satisfy every node
/// predicate and every required bond order (raw orders; fixed/delocalised
/// filtering happens in [`check_instance`]).
fn match_anchors(view: &View<'_>, pattern: &Pattern) -> BTreeSet<Vec<usize>> {
    let p = pattern.nodes.len();
    let mut linked: Vec<Vec<(usize, u8)>> = vec![Vec::new(); p];
    for b in pattern.bonds {
        linked[b.a].push((b.b, b.order));
        linked[b.b].push((b.a, b.order));
    }
    let mut anchors = BTreeSet::new();
    let mut map: Vec<Option<usize>> = vec![None; p];
    let mut used = vec![false; view.atoms.len()];
    fn rec(
        view: &View<'_>,
        pattern: &Pattern,
        linked: &[Vec<(usize, u8)>],
        map: &mut [Option<usize>],
        used: &mut [bool],
        anchors: &mut BTreeSet<Vec<usize>>,
    ) {
        if map.iter().all(|m| m.is_some()) {
            let mut anchor: Vec<usize> = map.iter().map(|m| m.expect("all mapped")).collect();
            anchor.sort_unstable();
            anchors.insert(anchor);
            return;
        }
        let mut best: Option<usize> = None;
        let mut best_score = (false, 0usize);
        for (q, m) in map.iter().enumerate() {
            if m.is_some() {
                continue;
            }
            let mapped_nbrs = linked[q].iter().filter(|(r, _)| map[*r].is_some()).count();
            let has_mapped = mapped_nbrs > 0;
            let score = (has_mapped, mapped_nbrs);
            if best.is_none_or(|_| score > best_score) {
                best_score = score;
                best = Some(q);
            }
        }
        let q = best.expect("an unmapped node exists");
        for g in 0..view.atoms.len() {
            if used[g] || !view.node_ok(g, &pattern.nodes[q]) {
                continue;
            }
            let mut ok = true;
            for (r, order) in &linked[q] {
                if let Some(gr) = map[*r] {
                    if view.order(g, gr) != Some(*order) {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            map[q] = Some(g);
            used[g] = true;
            rec(view, pattern, linked, map, used, anchors);
            map[q] = None;
            used[g] = false;
        }
    }
    rec(view, pattern, &linked, &mut map, &mut used, &mut anchors);
    anchors
}

// ---------------------------------------------------------------------------
// Public detection
// ---------------------------------------------------------------------------

/// Determined functional-group instances of `graph`.
///
/// Every required atom and bond of a counted instance lies inside the graph,
/// every consulted bond status is decided, and every exclusion is certain
/// (see the module docs); fragments therefore only show groups they contain
/// completely. Kekulé-invariant: identical counts on every kekulé form.
/// Isomorphism-invariant: the result depends only on the labelled graph up
/// to isomorphism — atom numbering, bond list order and bond endpoint order
/// must not matter (five-ring exclusions are built from ring bonds in cycle
/// order, never from sorted-order adjacency).
pub fn functional_groups_v4(graph: &MolGraph) -> FgSet {
    let (set, _) = detect(graph);
    set
}

/// Mask of types with at least one matched but NOT determined instance
/// (reported, never scored). On closed graphs (every residual valence 0
/// and no undecided bond) this carries only five-ring/arene edge cases
/// that are fully inside yet undecided — in practice 0 on whole molecules.
pub fn undetermined(graph: &MolGraph) -> u32 {
    let (_, undet) = detect(graph);
    undet
}

/// Determined instances as `(1-based type id, sorted anchor atoms)`, one
/// entry per counted instance. Used by the closing-fragment verification.
pub fn fg_instances(graph: &MolGraph) -> Vec<(usize, Vec<usize>)> {
    let view = View::of(graph).expect("functional_groups_v4: validated graphs hold known atom types");
    instances_of(&view)
        .into_iter()
        .map(|(id, anchor, _)| (id, anchor))
        .collect()
}

/// Joint detection: determined counts plus the undetermined mask.
fn detect(graph: &MolGraph) -> (FgSet, u32) {
    let view = View::of(graph).expect("functional_groups_v4: validated graphs hold known atom types");
    let inst = instances_of(&view);
    let mut counts = [0u32; N_FG];
    let mut mask = 0u32;
    for (id, _, u) in &inst {
        if *u {
            continue;
        }
        counts[id - 1] += 1;
        mask |= 1u32 << (id - 1);
    }
    let mut undet = 0u32;
    for (id, _, u) in &inst {
        if *u {
            undet |= 1u32 << (id - 1);
        }
    }
    // Partial cores whose missing bonds could lie outside the fragment.
    undet |= partial_undetermined(&view);
    let mut set = FgSet::empty();
    set.counts = counts;
    set.mask = mask;
    (set, undet)
}

/// (id, anchor, undetermined-flag) over every anchor of every pattern plus
/// the structural types: determined entries have the flag false and are
/// counted; undetermined entries (flag true) feed only the mask.
fn instances_of(view: &View<'_>) -> Vec<(usize, Vec<usize>, bool)> {
    // Exclusion sets MUST be built from ring bonds in cycle order (see
    // `find_five_ring_cycles`): adjacency of sorted atom sets is not ring
    // adjacency and depends on atom numbering.
    let (five_cycles, five_maybe_cycles) = find_five_ring_cycles(view);
    let (five, five_maybe) = find_five_rings2(view);
    let mut five_hetero = vec![false; view.atoms.len()];
    let mut five_bond: BTreeSet<(usize, usize)> = BTreeSet::new();
    for ring in &five_cycles {
        debug_assert_eq!(ring.len(), 5);
        for (k, a) in ring.iter().enumerate() {
            let b = ring[(k + 1) % 5];
            let el = view.atoms[*a].element;
            let prev = ring[(k + 4) % 5];
            let prev_o = view.order(prev, *a).unwrap_or(0);
            let next_o = view.order(*a, b).unwrap_or(0);
            if matches!(el, N | O | S) && prev_o == 1 && next_o == 1 {
                five_hetero[*a] = true;
            }
            let key = if *a < b { (*a, b) } else { (b, *a) };
            five_bond.insert(key);
        }
    }
    // Possible-but-undecided coverage: heteroatoms that are single-only in
    // a Maybe ring, and every ring bond of a Maybe ring.
    let mut maybe_hetero = vec![false; view.atoms.len()];
    let mut maybe_bond: BTreeSet<(usize, usize)> = BTreeSet::new();
    for ring in &five_maybe_cycles {
        debug_assert_eq!(ring.len(), 5);
        for (k, a) in ring.iter().enumerate() {
            let b = ring[(k + 1) % 5];
            let el = view.atoms[*a].element;
            let prev = ring[(k + 4) % 5];
            let prev_o = view.order(prev, *a).unwrap_or(0);
            let next_o = view.order(*a, b).unwrap_or(0);
            if matches!(el, N | O | S) && prev_o == 1 && next_o == 1 {
                maybe_hetero[*a] = true;
            }
            let key = if *a < b { (*a, b) } else { (b, *a) };
            maybe_bond.insert(key);
        }
    }
    let mut out: Vec<(usize, Vec<usize>, bool)> = Vec::new();
    for pattern in PATTERNS {
        let anchors = match_anchors(view, pattern);
        for anchor in &anchors {
            let Some(map) = map_into(view, pattern, anchor) else {
                continue;
            };
            match check_instance(view, pattern, &map, &five_hetero, &five_bond, &maybe_hetero, &maybe_bond) {
                ExclOutcome::Violated => {}
                ExclOutcome::Uncertain => {
                    out.push((pattern.id, anchor.clone(), true));
                }
                ExclOutcome::Certain => {
                    // Five-ring heteroatoms are never amines/ethers: the
                    // check above already rejected those; nothing further.
                    out.push((pattern.id, anchor.clone(), false));
                }
            }
        }
    }
    // Arene rings: one instance per all-delocalised six-ring; a fully
    // inside six-ring with an undecided bond is undetermined arene.
    let (arene_det, arene_undet) = arene_instances(view);
    for a in arene_det {
        out.push((24, a, false));
    }
    for a in arene_undet {
        out.push((24, a, true));
    }
    // Anhydride / carbonate composite (id 27).
    for a in anhydride_instances(view) {
        out.push((27, a, false));
    }
    // Heteroaromatic five-rings (id 28): determined rings count; possible
    // rings with undecided qualifiers are undetermined (never scored).
    for ring in &five {
        out.push((28, ring.clone(), false));
    }
    for ring in &five_maybe {
        // A decided ring is never also reported undecided (defensive: the
        // enumerator keeps the sets disjoint, but dedup below enforces it).
        out.push((28, ring.clone(), true));
    }
    // Deduplicate determined anchors per type (the matcher already
    // deduplicates per pattern, but id 26 has two rows).
    let mut seen: BTreeSet<(usize, Vec<usize>)> = BTreeSet::new();
    let mut dedup: Vec<(usize, Vec<usize>, bool)> = Vec::new();
    for (id, anchor, u) in out {
        if u {
            dedup.push((id, anchor, true));
            continue;
        }
        if seen.insert((id, anchor.clone())) {
            dedup.push((id, anchor, false));
        }
    }
    // Collapse to one entry per (id, anchor): a determined entry beats an
    // undetermined one for the same anchor (an anchor fully witnessed as
    // determined is not also reported undetermined).
    let mut det_set: BTreeSet<(usize, Vec<usize>)> = BTreeSet::new();
    for (id, anchor, u) in &dedup {
        if !u {
            det_set.insert((*id, anchor.clone()));
        }
    }
    let mut fin: Vec<(usize, Vec<usize>, bool)> = Vec::new();
    for (id, anchor, u) in dedup {
        if u && det_set.contains(&(id, anchor.clone())) {
            continue;
        }
        fin.push((id, anchor, u));
    }
    fin
}

/// Arene instances: determined all-delocalised six-rings, plus undetermined
/// six-rings (fully inside C/N six-cycle with an undecided bond and no
/// fixed bond).
fn arene_instances(view: &View<'_>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = view.atoms.len();
    let mut det: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut undet: BTreeSet<Vec<usize>> = BTreeSet::new();
    for s in 0..n {
        if view.atoms[s].element != C && view.atoms[s].element != N {
            continue;
        }
        let mut path = vec![s];
        let mut vis = vec![false; n];
        vis[s] = true;
        dfs_arene(view, s, s, &mut path, &mut vis, &mut det, &mut undet);
    }
    (det.into_iter().collect(), undet.into_iter().collect())
}

fn dfs_arene(
    view: &View<'_>,
    start: usize,
    cur: usize,
    path: &mut Vec<usize>,
    vis: &mut [bool],
    det: &mut BTreeSet<Vec<usize>>,
    undet: &mut BTreeSet<Vec<usize>>,
) {
    if path.len() == 6 {
        let Some(bi) = view.bond_idx(cur, start) else {
            return;
        };
        if view.bond_orders[bi] != 1 && view.bond_orders[bi] != 2 {
            return;
        }
        let mut ring = path.clone();
        ring.push(start);
        let mut has_fixed = false;
        let mut has_unknown = false;
        for w in ring.windows(2) {
            let Some(i) = view.bond_idx(w[0], w[1]) else {
                return;
            };
            match view.status[i] {
                BondStatus::Fixed => has_fixed = true,
                BondStatus::Unknown => has_unknown = true,
                BondStatus::Delocalised => {}
            }
        }
        let mut key = path.clone();
        key.sort_unstable();
        if !has_fixed && !has_unknown {
            det.insert(key);
        } else if !has_fixed && has_unknown {
            undet.insert(key);
        }
        return;
    }
    for (nxt, o, _) in &view.adj[cur] {
        if *o != 1 && *o != 2 {
            continue;
        }
        if *nxt <= start && *nxt != start {
            continue;
        }
        if vis[*nxt] {
            continue;
        }
        let a = view.atoms[*nxt];
        if a.element != C && a.element != N {
            continue;
        }
        vis[*nxt] = true;
        path.push(*nxt);
        dfs_arene(view, start, *nxt, path, vis, det, undet);
        path.pop();
        vis[*nxt] = false;
    }
}

/// Anhydride / carbonate instances (id 27): `C(=O)–O–C(=O)` anchors of five
/// atoms, and `O–C(=O)–O` anchors of four atoms with both single oxygens
/// H0/H1. All required doubles must be fixed and decided; any undecided
/// required double makes the anchor undetermined (reported via the generic
/// path by the caller treating these as determined-only: undecided cores
/// are simply absent here and surface through partial witnesses — none for
/// this rare type — so fragments show only complete certain instances).
fn anhydride_instances(view: &View<'_>) -> Vec<Vec<usize>> {
    let n = view.atoms.len();
    let mut out: BTreeSet<Vec<usize>> = BTreeSet::new();
    // Subpattern A: C1(=O1)–O–C2(=O2), bridge O H0 with two C singles.
    for o in 0..n {
        if view.atoms[o].element != O || view.atoms[o].h != 0 {
            continue;
        }
        let cs: Vec<usize> = view.adj[o]
            .iter()
            .filter(|t| t.1 == 1 && view.atoms[t.0].element == C)
            .map(|t| t.0)
            .collect();
        if cs.len() != 2 {
            continue;
        }
        let (c1, c2) = (cs[0], cs[1]);
        let mut has_both = true;
        for &c in &[c1, c2] {
            // Each carbonyl carbon needs exactly one fixed double to O.
            let dbl: Vec<usize> = view.adj[c]
                .iter()
                .filter(|t| view.is_fixed_double(c, t.0) && view.atoms[t.0].element == O)
                .map(|t| t.0)
                .collect();
            if dbl.len() != 1 {
                has_both = false;
                break;
            }
        }
        if !has_both {
            continue;
        }
        // Both carbonyl doubles decided fixed (no unknown double on c1/c2
        // to O); the bridge singles are raw singles (always decided).
        let mut ok = true;
        for &c in &[c1, c2] {
            for t in &view.adj[c] {
                if view.atoms[t.0].element == O && view.is_unknown_double(c, t.0) {
                    ok = false;
                }
            }
        }
        if !ok {
            continue;
        }
        // Carbonyl-remaining: each carbonyl's extra neighbour (besides its
        // double O and the bridge) is not checked here — anhydrides carry
        // alkyl/aryl remainders, carbonates do not match this subpattern
        // (their carbonyls have two O singles). Accept any remainder.
        let d1 = view.adj[c1]
            .iter()
            .find(|t| view.is_fixed_double(c1, t.0) && view.atoms[t.0].element == O)
            .map(|t| t.0);
        let d2 = view.adj[c2]
            .iter()
            .find(|t| view.is_fixed_double(c2, t.0) && view.atoms[t.0].element == O)
            .map(|t| t.0);
        let (Some(dd1), Some(dd2)) = (d1, d2) else {
            continue;
        };
        let mut anchor = vec![c1, dd1, o, c2, dd2];
        anchor.sort_unstable();
        out.insert(anchor);
    }
    // Subpattern B: O–C(=O)–O with both single oxygens H0 or H1.
    for c in 0..n {
        if view.atoms[c].element != C {
            continue;
        }
        let dbl: Vec<usize> = view.adj[c]
            .iter()
            .filter(|t| view.is_fixed_double(c, t.0) && view.atoms[t.0].element == O)
            .map(|t| t.0)
            .collect();
        if dbl.len() != 1 {
            continue;
        }
        let singles: Vec<usize> = view.adj[c]
            .iter()
            .filter(|t| {
                t.1 == 1
                    && view.atoms[t.0].element == O
                    && (view.atoms[t.0].h == 0 || view.atoms[t.0].h == 1)
            })
            .map(|t| t.0)
            .collect();
        if singles.len() != 2 {
            continue;
        }
        // No undecided double on this carbon.
        if view.adj[c].iter().any(|t| view.is_unknown_double(c, t.0)) {
            continue;
        }
        let mut anchor = vec![c, dbl[0], singles[0], singles[1]];
        anchor.sort_unstable();
        out.insert(anchor);
    }
    out.into_iter().collect()
}

/// One injective map of a pattern's nodes into an anchor set, satisfying the
/// node predicates and bond orders (`None` when the set is not an embedding
/// — unreachable for anchors the matcher produced).
fn map_into(view: &View<'_>, pattern: &Pattern, anchor: &[usize]) -> Option<Vec<usize>> {
    let p = pattern.nodes.len();
    if anchor.len() != p {
        return None;
    }
    let mut linked: Vec<Vec<(usize, u8)>> = vec![Vec::new(); p];
    for b in pattern.bonds {
        linked[b.a].push((b.b, b.order));
        linked[b.b].push((b.a, b.order));
    }
    let mut map: Vec<Option<usize>> = vec![None; p];
    let mut taken = vec![false; p];
    fn rec(
        view: &View<'_>,
        pattern: &Pattern,
        linked: &[Vec<(usize, u8)>],
        anchor: &[usize],
        map: &mut [Option<usize>],
        taken: &mut [bool],
    ) -> bool {
        if map.iter().all(|m| m.is_some()) {
            return true;
        }
        let mut best: Option<usize> = None;
        let mut best_score = (false, 0usize);
        for (q, m) in map.iter().enumerate() {
            if m.is_some() {
                continue;
            }
            let n = linked[q].iter().filter(|(r, _)| map[*r].is_some()).count();
            let score = (n > 0, n);
            if best.is_none_or(|_| score > best_score) {
                best_score = score;
                best = Some(q);
            }
        }
        let q = best.expect("an unmapped node exists");
        for (slot, &g) in anchor.iter().enumerate() {
            if taken[slot] || !view.node_ok(g, &pattern.nodes[q]) {
                continue;
            }
            if linked[q].iter().any(|(r, order)| {
                map[*r].is_some_and(|gr| view.order(g, gr) != Some(*order))
            }) {
                continue;
            }
            map[q] = Some(g);
            taken[slot] = true;
            if rec(view, pattern, linked, anchor, map, taken) {
                return true;
            }
            map[q] = None;
            taken[slot] = false;
        }
        false
    }
    if rec(view, pattern, &linked, anchor, &mut map, &mut taken) {
        Some(map.into_iter().map(|m| m.expect("mapped")).collect())
    } else {
        None
    }
}

/// Undetermined flags from partial cores: a matched core whose missing bonds
/// could lie outside the fragment, with no positively violated exclusion.
/// Hydrogen counts are respected: a `C(=O)–O[H0]` core with an open oxygen
/// is an undetermined `ester` only (never an undetermined acid); a
/// `C(=O)–O[H1]` core is an undetermined acid only.
fn partial_undetermined(view: &View<'_>) -> u32 {
    let mut undet = 0u32;
    // Carbonyl-carbon/oxygen witness for acid/ester, split by the oxygen's
    // hydrogen count.
    for c in 0..view.atoms.len() {
        if view.atoms[c].element != C {
            continue;
        }
        let mut has_fixed_dbl_o = false;
        let mut single_o: Vec<usize> = Vec::new();
        for (nbr, _o, _) in &view.adj[c] {
            if view.atoms[*nbr].element == O {
                if view.is_fixed_double(c, *nbr) {
                    has_fixed_dbl_o = true;
                } else if view.order(c, *nbr) == Some(1) {
                    single_o.push(*nbr);
                }
            }
        }
        if !has_fixed_dbl_o || single_o.is_empty() {
            continue;
        }
        for &o in &single_o {
            if view.residual[c] == 0 && view.residual[o] == 0 {
                // A complete C(=O)–O core is handled by the full matcher
                // (determined or rejected there); partial logic skips it.
                continue;
            }
            if view.residual[c] > 0 || view.residual[o] > 0 {
                match view.atoms[o].h {
                    0 => undet |= 1u32 << (3 - 1),
                    1 => undet |= 1u32 << (2 - 1),
                    _ => {}
                }
            }
        }
    }
    // Amine witnesses: a nitrogen with room for more carbons outside.
    for x in 0..view.atoms.len() {
        if view.atoms[x].element != N {
            continue;
        }
        // Pyrrolic nitrogens never witness amines.
        let h = view.atoms[x].h;
        let carbons: Vec<usize> = view.adj[x]
            .iter()
            .filter(|t| t.1 == 1 && view.atoms[t.0].element == C)
            .map(|t| t.0)
            .collect();
        let excluded_inside = carbons.iter().any(|&c| fixed_double_to_osn(view, c));
        if excluded_inside {
            continue;
        }
        if h == 2 && carbons.is_empty() && view.residual[x] > 0 {
            undet |= 1u32 << (9 - 1);
        } else if h == 1 && carbons.len() == 1 && view.residual[x] > 0 {
            undet |= 1u32 << (10 - 1);
        } else if h == 0 && !carbons.is_empty() && carbons.len() < 3 && view.residual[x] > 0 {
            undet |= 1u32 << (11 - 1);
        }
    }
    // Ether witness: C–O[H0] with the oxygen short of its second carbon.
    for o in 0..view.atoms.len() {
        if view.atoms[o].element != O || view.atoms[o].h != 0 {
            continue;
        }
        let carbons: Vec<usize> = view.adj[o]
            .iter()
            .filter(|t| t.1 == 1 && view.atoms[t.0].element == C)
            .map(|t| t.0)
            .collect();
        if carbons.len() == 1 && view.residual[o] > 0
            && !fixed_double_to_osn(view, carbons[0])
        {
            undet |= 1u32 << (8 - 1);
        }
    }
    // Hydroxyl witness: a lone open O[H1] could gain its carbon outside.
    for o in 0..view.atoms.len() {
        if view.atoms[o].element != O || view.atoms[o].h != 1 {
            continue;
        }
        let has_carbon = view.adj[o]
            .iter()
            .any(|t| t.1 == 1 && view.atoms[t.0].element == C);
        if !has_carbon && view.residual[o] > 0 {
            undet |= 1u32 << (7 - 1);
        }
    }
    // Thioether witness: C–S[H0] with the sulphur short of its second carbon.
    for s in 0..view.atoms.len() {
        if view.atoms[s].element != S || view.atoms[s].h != 0 || view.atoms[s].valence != 2 {
            continue;
        }
        let carbons = view.adj[s]
            .iter()
            .filter(|t| t.1 == 1 && view.atoms[t.0].element == C)
            .count();
        if carbons == 1 && view.residual[s] > 0 {
            undet |= 1u32 << (17 - 1);
        }
    }
    undet
}

/// Whether a carbon has a fixed double bond to O, S or N inside the graph
/// (a positive exclusion violation for the hydroxyl/ether/amine/thiol
/// patterns; delocalised doubles do not violate).
fn fixed_double_to_osn(view: &View<'_>, c: usize) -> bool {
    view.adj[c].iter().any(|t| {
        view.is_fixed_double(c, t.0) && matches!(view.atoms[t.0].element, O | S | N)
    })
}

