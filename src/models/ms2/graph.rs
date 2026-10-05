//! Molecules and domain classification of `docs/MS2_CONTRACTS.md` (§§4.2, 4.5, 4.6).
//!
//! [`RawMolecule`] is a kekulized heavy-atom graph with per-atom metadata as
//! RDKit reports it; [`MolGraph`] is the validated in-domain labeled graph of
//! atom type ids and bond orders that the grammar replays.

use std::collections::HashSet;

use crate::error::{Error, Result};

use super::chem::{self, Composition, HYDROGEN};

/// One heavy atom as reported by RDKit: element, charge and radical state,
///
/// plus the total hydrogen count and the valence (hydrogens plus the sum of
/// kekulized bond orders).
pub struct RawAtom {
    /// Element symbol.
    pub element: String,
    /// Formal charge.
    pub charge: i32,
    /// Total (implicit plus explicit) hydrogen count.
    pub hydrogens: u8,
    /// Isotope label, 0 when absent.
    pub isotope: u32,
    /// Radical electron count.
    pub radical_electrons: u8,
    /// Hydrogens plus the sum of bond orders.
    pub valence: u8,
}

/// A kekulized molecule before domain validation: heavy atoms and bonds.
pub struct RawMolecule {
    /// Heavy atoms; hydrogens are counts on them, never vertices.
    pub atoms: Vec<RawAtom>,
    /// Bonds as `(a, b, order)` triples.
    pub bonds: Vec<(usize, usize, u8)>,
}

/// One reason a molecule is outside the V0 structure domain (contract §4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainReason {
    /// More than one connected component.
    Disconnected,
    /// Symbol outside §4.1, or an explicit hydrogen vertex.
    ElementOutsideDomain,
    /// Non-zero formal charge.
    FormalCharge,
    /// Non-zero isotope label.
    IsotopeLabel,
    /// Non-zero radical electron count.
    Radical,
    /// An otherwise clean atom whose triple is not in §4.2.
    UnsupportedAtomType,
}

impl DomainReason {
    /// The snake_case name used by the fixture.
    pub fn name(&self) -> &'static str {
        match self {
            DomainReason::Disconnected => "disconnected",
            DomainReason::ElementOutsideDomain => "element_outside_domain",
            DomainReason::FormalCharge => "formal_charge",
            DomainReason::IsotopeLabel => "isotope_label",
            DomainReason::Radical => "radical",
            DomainReason::UnsupportedAtomType => "unsupported_atom_type",
        }
    }
}

impl RawMolecule {
    /// Every reason the molecule is outside the V0 domain, sorted by name.
    ///
    /// Empty means in domain. `unsupported_atom_type` is reported only for an
    /// atom with none of the other four per-atom defects.
    pub fn classify(&self) -> Vec<DomainReason> {
        let mut reasons: Vec<DomainReason> = Vec::new();
        if component_count(self.atoms.len(), &self.bonds) != 1 {
            reasons.push(DomainReason::Disconnected);
        }
        for a in &self.atoms {
            let mut own = Vec::new();
            if chem::element_index(&a.element).is_none() || a.element == "H" {
                own.push(DomainReason::ElementOutsideDomain);
            }
            if a.charge != 0 {
                own.push(DomainReason::FormalCharge);
            }
            if a.isotope != 0 {
                own.push(DomainReason::IsotopeLabel);
            }
            if a.radical_electrons != 0 {
                own.push(DomainReason::Radical);
            }
            if own.is_empty() {
                let known = chem::element_index(&a.element)
                    .is_some_and(|e| chem::atom_type_of(e, a.hydrogens, a.valence).is_some());
                if !known {
                    own.push(DomainReason::UnsupportedAtomType);
                }
            }
            reasons.extend(own);
        }
        reasons.sort_by_key(|r| r.name());
        reasons.dedup_by_key(|r| r.name());
        reasons
    }

    /// Validate into the labeled in-domain graph; out-of-domain molecules
    /// are an error naming every reason.
    pub fn to_graph(&self) -> Result<MolGraph> {
        let reasons = self.classify();
        if !reasons.is_empty() {
            let names: Vec<&str> = reasons.iter().map(DomainReason::name).collect();
            return Err(Error::Unsupported(format!(
                "out of domain molecule: {}",
                names.join(", ")
            )));
        }
        let mut ids = Vec::with_capacity(self.atoms.len());
        for (i, a) in self.atoms.iter().enumerate() {
            let element = chem::element_index(&a.element).ok_or_else(|| {
                Error::config(format!(
                    "to_graph: atom {i} has unknown element {:?}",
                    a.element
                ))
            })?;
            let t = chem::atom_type_of(element, a.hydrogens, a.valence).ok_or_else(|| {
                Error::config(format!(
                    "to_graph: atom {i} ({}, H{}, v{}) has no atom type",
                    a.element, a.hydrogens, a.valence
                ))
            })?;
            ids.push(t.id);
        }
        MolGraph::new(ids, self.bonds.clone())
    }
}

/// A molecule or subgraph is limited to 4,096 atoms on the host (contract
/// §4.5); larger inputs are [`Error::Config`].
pub const MAX_GRAPH_ATOMS: usize = 4096;

/// Count connected components over `n` vertices, skipping bonds that name
/// no vertex (raw input is never trusted for indexing here).
fn component_count(n: usize, bonds: &[(usize, usize, u8)]) -> usize {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (a, b, _) in bonds {
        if *a < n && *b < n {
            let ra = find(&mut parent, *a);
            let rb = find(&mut parent, *b);
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut roots = HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    roots.len()
}

/// A validated in-domain labeled graph: atom type ids and kekulized orders.
///
/// Bond endpoints are stored with `a < b`; the ending order of [`bonds`]
/// is sorted. Connectivity is not required: disconnected graphs build fine
/// and fail only where the contract needs connectivity (canonicalization).
#[derive(Clone, Debug)]
pub struct MolGraph {
    /// Atom type id per atom.
    types: Vec<u8>,
    /// Bonds as `(a, b, order)` with `a < b`, sorted.
    bonds: Vec<(usize, usize, u8)>,
}

impl MolGraph {
    /// Validate and store a labeled graph.
    ///
    /// Rejects more than [`MAX_GRAPH_ATOMS`] atoms (contract §4.5), unknown
    /// type ids, out-of-range indices, self-bonds, orders outside 1–3,
    /// duplicate bonds in either orientation, and any atom whose residual
    /// valence (`valence − hydrogens − bond orders`) is negative.
    /// Connectivity is not checked.
    ///
    /// The atom cap keeps every element count in [`composition`] inside
    /// `u16`: at most 4,096 heavy atoms of one element, and at most
    /// `4,096 × 3` hydrogens (C H3 is the most hydrogenous type).
    pub fn new(atom_types: Vec<u8>, bonds: Vec<(usize, usize, u8)>) -> Result<Self> {
        let n = atom_types.len();
        if n > MAX_GRAPH_ATOMS {
            return Err(Error::config(format!(
                "MolGraph::new: {n} atoms exceed the host limit {MAX_GRAPH_ATOMS}"
            )));
        }
        for (i, id) in atom_types.iter().enumerate() {
            if chem::atom_type(*id).is_none() {
                return Err(Error::config(format!(
                    "MolGraph::new: unknown atom type id {id} at atom {i}"
                )));
            }
        }
        let mut seen = HashSet::new();
        let mut norm = Vec::with_capacity(bonds.len());
        for (a, b, order) in &bonds {
            if *a >= n || *b >= n {
                return Err(Error::config(format!(
                    "MolGraph::new: bond ({a}, {b}) names no atom of {n}"
                )));
            }
            if a == b {
                return Err(Error::config(format!(
                    "MolGraph::new: self-bond on atom {a}"
                )));
            }
            if !matches!(order, 1..=3) {
                return Err(Error::config(format!(
                    "MolGraph::new: bond order {order} outside 1..=3"
                )));
            }
            let (lo, hi) = if a < b { (*a, *b) } else { (*b, *a) };
            if !seen.insert((lo, hi)) {
                return Err(Error::config(format!(
                    "MolGraph::new: duplicate bond ({lo}, {hi})"
                )));
            }
            norm.push((lo, hi, *order));
        }
        norm.sort();
        let mut order_sum = vec![0u32; n];
        for (a, b, order) in &norm {
            order_sum[*a] += u32::from(*order);
            order_sum[*b] += u32::from(*order);
        }
        for (i, id) in atom_types.iter().enumerate() {
            let t = chem::atom_type(*id).expect("type ids checked above");
            let capacity = u32::from(t.valence) - u32::from(t.hydrogens);
            if order_sum[i] > capacity {
                return Err(Error::config(format!(
                    "MolGraph::new: atom {i} (type {id}) carries orders {} above residual {}",
                    order_sum[i], capacity
                )));
            }
        }
        Ok(Self {
            types: atom_types,
            bonds: norm,
        })
    }

    /// Atom type ids, one per atom.
    pub fn atoms(&self) -> &[u8] {
        &self.types
    }

    /// Bonds as `(a, b, order)` with `a < b`, sorted.
    pub fn bonds(&self) -> &[(usize, usize, u8)] {
        &self.bonds
    }

    /// Element composition: heavy atoms plus their parent hydrogens.
    pub fn composition(&self) -> Composition {
        let mut c: Composition = [0; 10];
        for id in &self.types {
            let t = chem::atom_type(*id).expect("stored type ids are valid");
            c[t.element] += 1;
            c[HYDROGEN] += u16::from(t.hydrogens);
        }
        c
    }

    /// `valence − hydrogens − bond orders` per atom; non-negative by
    /// construction. At STOP this is the open attachment valence (§4.5).
    pub fn residual_valence(&self) -> Vec<u8> {
        let mut residual = Vec::with_capacity(self.types.len());
        for (i, id) in self.types.iter().enumerate() {
            let t = chem::atom_type(*id).expect("stored type ids are valid");
            let mut used = u32::from(t.hydrogens);
            for (a, b, order) in &self.bonds {
                if *a == i || *b == i {
                    used += u32::from(*order);
                }
            }
            residual.push(t.valence - used as u8);
        }
        residual
    }

    /// Cyclomatic number: `bonds − atoms + components`.
    pub fn ring_closures(&self) -> usize {
        self.bonds.len() + component_count(self.types.len(), &self.bonds) - self.types.len()
    }

    /// Whether every atom sits in one component (empty graphs are not).
    pub fn is_connected(&self) -> bool {
        !self.types.is_empty() && component_count(self.types.len(), &self.bonds) == 1
    }

    /// The induced subgraph on `atoms`: result atom `i` is `atoms[i]`.
    ///
    /// Errors on duplicates and out-of-range indices.
    pub fn induced(&self, atoms: &[usize]) -> Result<MolGraph> {
        let n = self.types.len();
        let mut pos: Vec<Option<usize>> = vec![None; n];
        for (i, a) in atoms.iter().enumerate() {
            if *a >= n {
                return Err(Error::config(format!(
                    "induced: atom {a} outside graph of {n} atoms"
                )));
            }
            if pos[*a].is_some() {
                return Err(Error::config(format!("induced: duplicate atom {a}")));
            }
            pos[*a] = Some(i);
        }
        let types: Vec<u8> = atoms.iter().map(|a| self.types[*a]).collect();
        let mut bonds = Vec::new();
        for (a, b, order) in &self.bonds {
            if let (Some(i), Some(j)) = (pos[*a], pos[*b]) {
                bonds.push((i, j, *order));
            }
        }
        MolGraph::new(types, bonds)
    }

    /// The relabeled graph with result atom `i` holding self atom `perm[i]`.
    ///
    /// Errors unless `perm` is a permutation of all atom indices.
    pub fn permuted(&self, perm: &[usize]) -> Result<MolGraph> {
        let n = self.types.len();
        if perm.len() != n {
            return Err(Error::config(format!(
                "permuted: length {} is not the atom count {n}",
                perm.len()
            )));
        }
        let mut seen = vec![false; n];
        for p in perm {
            if *p >= n {
                return Err(Error::config(format!(
                    "permuted: atom {p} outside graph of {n} atoms"
                )));
            }
            if seen[*p] {
                return Err(Error::config(format!("permuted: duplicate atom {p}")));
            }
            seen[*p] = true;
        }
        let mut inverse = vec![0usize; n];
        for (i, p) in perm.iter().enumerate() {
            inverse[*p] = i;
        }
        let types: Vec<u8> = perm.iter().map(|p| self.types[*p]).collect();
        let bonds: Vec<(usize, usize, u8)> = self
            .bonds
            .iter()
            .map(|(a, b, o)| (inverse[*a], inverse[*b], *o))
            .collect();
        MolGraph::new(types, bonds)
    }
}
