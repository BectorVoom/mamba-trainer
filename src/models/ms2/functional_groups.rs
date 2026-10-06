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
