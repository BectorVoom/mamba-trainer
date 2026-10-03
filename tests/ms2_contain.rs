//! Host-only tests for induced-subgraph containment (contracts §7.3).

use std::path::PathBuf;

use serde_json::Value;

use mamba3::models::ms2::contain::{Containment, contains_induced};
use mamba3::models::ms2::{MolGraph, RawAtom, RawMolecule};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json")
}

fn fixture() -> Value {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn raw_molecule(m: &Value) -> RawMolecule {
    let atoms = m["raw_atoms"]
        .as_array()
        .expect("raw_atoms")
        .iter()
        .map(|a| RawAtom {
            element: a["element"].as_str().expect("element").to_string(),
            charge: a["charge"].as_i64().expect("charge") as i32,
            hydrogens: a["hydrogens"].as_u64().expect("hydrogens") as u8,
            isotope: a["isotope"].as_u64().expect("isotope") as u32,
            radical_electrons: a["radical_electrons"].as_u64().expect("radical") as u8,
            valence: a["valence"].as_u64().expect("valence") as u8,
        })
        .collect();
    let bonds = m["bonds"]
        .as_array()
        .expect("bonds")
        .iter()
        .map(|b| {
            let b = b.as_array().expect("bond triple");
            (
                b[0].as_u64().expect("a") as usize,
                b[1].as_u64().expect("b") as usize,
                b[2].as_u64().expect("order") as u8,
            )
        })
        .collect();
    RawMolecule { atoms, bonds }
}

fn graph_of(m: &Value) -> MolGraph {
    raw_molecule(m)
        .to_graph()
        .expect("in-domain molecule builds")
}

fn members_of(g: &Value) -> Vec<usize> {
    g["atoms"]
        .as_array()
        .expect("atoms")
        .iter()
        .map(|a| a.as_u64().expect("atom") as usize)
        .collect()
}

/// Deterministic SplitMix64 for permutations and random graphs.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn permute(rng: &mut SplitMix64, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        p.swap(i, j);
    }
    p
}

#[test]
fn fixture_subgraphs_are_contained() {
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let parent = graph_of(m);
        for g in m["subgraphs"].as_array().expect("subgraphs") {
            let members = members_of(g);
            let sub = parent.induced(&members).expect("induced subgraph builds");
            assert_eq!(
                contains_induced(&parent, &sub, 1_000_000),
                Containment::Contained,
                "{name} {members:?} contained"
            );
        }
    }
}

#[test]
fn fixture_subgraphs_contained_after_permutation() {
    let f = fixture();
    let mut rng = SplitMix64::new(0xC0A1);
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let parent = graph_of(m);
        let n_p = parent.atoms().len();
        let perm_p = permute(&mut rng, n_p);
        let parent_p = parent.permuted(&perm_p).expect("parent permutes");
        for g in m["subgraphs"].as_array().expect("subgraphs").iter().take(8) {
            let members = members_of(g);
            let sub = parent.induced(&members).expect("induced builds");
            let n_c = sub.atoms().len();
            let perm_c = permute(&mut rng, n_c);
            let sub_p = sub.permuted(&perm_c).expect("candidate permutes");
            assert_eq!(
                contains_induced(&parent_p, &sub_p, 1_000_000),
                Containment::Contained,
                "{name} {members:?} contained after permutation"
            );
        }
    }
}

#[test]
fn removing_ring_bond_breaks_induced_containment() {
    let f = fixture();
    let mut found = 0;
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let parent = graph_of(m);
        for g in m["subgraphs"].as_array().expect("subgraphs") {
            if g["closures"].as_u64().unwrap() == 0 {
                continue;
            }
            let members = members_of(g);
            let sub = parent.induced(&members).expect("induced builds");
            // Try removing each bond: keep the first removal that stays
            // connected (a cycle edge), so the failure is the missing bond,
            // not disconnectivity.
            for drop in 0..sub.bonds().len() {
                let mut bonds: Vec<(usize, usize, u8)> = sub.bonds().to_vec();
                bonds.remove(drop);
                let edited = MolGraph::new(sub.atoms().to_vec(), bonds);
                let Ok(edited) = edited else { continue };
                if !edited.is_connected() {
                    continue;
                }
                assert_eq!(
                    contains_induced(&parent, &edited, 1_000_000),
                    Containment::NotContained,
                    "{name} {members:?} ring-bond removal is not contained"
                );
                found += 1;
                break;
            }
            if found >= 3 {
                break;
            }
        }
        if found >= 3 {
            break;
        }
    }
    assert!(found >= 3, "at least three ring-bond cases");
}

#[test]
fn changing_atom_type_breaks_containment() {
    let f = fixture();
    let mut found = 0;
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let parent = graph_of(m);
        let parent_types: std::collections::BTreeSet<u8> = parent.atoms().iter().copied().collect();
        for g in m["subgraphs"].as_array().expect("subgraphs") {
            let members = members_of(g);
            let sub = parent.induced(&members).expect("induced builds");
            // Degrees of the candidate atoms.
            let mut deg = vec![0usize; sub.atoms().len()];
            for (a, b, _) in sub.bonds() {
                deg[*a] += 1;
                deg[*b] += 1;
            }
            let mut edited = None;
            for (idx, d) in deg.iter().enumerate() {
                // A replacement type absent from the parent with enough bond
                // capacity: the type-count reject then proves NotContained.
                for id in [16u8, 15, 12, 11, 10] {
                    if parent_types.contains(&id) {
                        continue;
                    }
                    let t = mamba3::models::ms2::atom_type(id).expect("known type");
                    if (t.valence - t.hydrogens) as usize >= *d {
                        let mut types = sub.atoms().to_vec();
                        types[idx] = id;
                        if let Ok(g) = MolGraph::new(types, sub.bonds().to_vec()) {
                            edited = Some(g);
                            break;
                        }
                    }
                }
                if edited.is_some() {
                    break;
                }
            }
            if let Some(edited) = edited {
                assert_eq!(
                    contains_induced(&parent, &edited, 1_000_000),
                    Containment::NotContained,
                    "{name} {members:?} retyped candidate is not contained"
                );
                found += 1;
                if found >= 5 {
                    break;
                }
            }
        }
        if found >= 5 {
            break;
        }
    }
    assert!(found >= 5, "at least five retype cases");
}

/// Brute-force reference: every injective map, type and induced bonds exact.
fn brute_force(parent: &MolGraph, candidate: &MolGraph) -> Containment {
    let p = parent.atoms();
    let c = candidate.atoms();
    if c.is_empty() {
        return Containment::NotContained;
    }
    if c.len() > p.len() {
        return Containment::NotContained;
    }
    let mut p_adj = vec![vec![0u8; p.len()]; p.len()];
    for (a, b, o) in parent.bonds() {
        p_adj[*a][*b] = *o;
        p_adj[*b][*a] = *o;
    }
    let mut c_adj = vec![vec![0u8; c.len()]; c.len()];
    for (a, b, o) in candidate.bonds() {
        c_adj[*a][*b] = *o;
        c_adj[*b][*a] = *o;
    }
    // All injective maps via index permutations of parent choices.
    let n_p = p.len();
    let n_c = c.len();
    let choice: Vec<usize> = (0..n_c).collect();
    // Iterate over all ordered distinct tuples by odometer with dedup check.
    let mut total = 1usize;
    for i in 0..n_c {
        total *= n_p - i;
    }
    // Simple recursive enumeration.
    fn rec(
        depth: usize,
        n_p: usize,
        n_c: usize,
        used: &mut [bool],
        image: &mut [usize],
        p: &[u8],
        c: &[u8],
        p_adj: &[Vec<u8>],
        c_adj: &[Vec<u8>],
    ) -> bool {
        if depth == n_c {
            return true;
        }
        for q in 0..n_p {
            if used[q] {
                continue;
            }
            if p[q] != c[depth] {
                continue;
            }
            let mut ok = true;
            for d in 0..depth {
                if c_adj[depth][d] != p_adj[q][image[d]] {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            used[q] = true;
            image[depth] = q;
            if rec(depth + 1, n_p, n_c, used, image, p, c, p_adj, c_adj) {
                return true;
            }
            used[q] = false;
        }
        false
    }
    let mut used = vec![false; n_p];
    let mut image = vec![0usize; n_c];
    let _ = (choice, total);
    if rec(0, n_p, n_c, &mut used, &mut image, p, c, &p_adj, &c_adj) {
        Containment::Contained
    } else {
        Containment::NotContained
    }
}

#[test]
fn agrees_with_brute_force_on_random_graphs() {
    let mut rng = SplitMix64::new(20261003);
    let type_pool = [1u8, 2, 3, 4, 6, 8, 9, 10];
    let mut cases = 0;
    while cases < 500 {
        let n_p = 1 + rng.below(6);
        let n_c = rng.below(5);
        let p_types: Vec<u8> = (0..n_p)
            .map(|_| type_pool[rng.below(type_pool.len())])
            .collect();
        let c_types: Vec<u8> = (0..n_c)
            .map(|_| type_pool[rng.below(type_pool.len())])
            .collect();
        // Random bonds: each pair with ~30% chance, order 1 or 2.
        let mut p_bonds = Vec::new();
        for a in 0..n_p {
            for b in (a + 1)..n_p {
                if rng.below(10) < 3 {
                    p_bonds.push((a, b, if rng.below(6) == 0 { 2 } else { 1 }));
                }
            }
        }
        let mut c_bonds = Vec::new();
        for a in 0..n_c {
            for b in (a + 1)..n_c {
                if rng.below(10) < 3 {
                    c_bonds.push((a, b, if rng.below(6) == 0 { 2 } else { 1 }));
                }
            }
        }
        let (Ok(parent), Ok(candidate)) = (
            MolGraph::new(p_types, p_bonds),
            MolGraph::new(c_types, c_bonds),
        ) else {
            continue;
        };
        let fast = contains_induced(&parent, &candidate, 10_000_000);
        let slow = brute_force(&parent, &candidate);
        assert_eq!(fast, slow, "case {cases}: brute-force agreement");
        assert_ne!(fast, Containment::WorkLimit, "large limit never exhausts");
        cases += 1;
    }
}

#[test]
fn tiny_work_limit_gives_work_limit() {
    let f = fixture();
    let m = f["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .find(|m| m["name"].as_str().unwrap() == "benzene")
        .expect("benzene");
    let parent = graph_of(m);
    let members = members_of(&m["subgraphs"].as_array().expect("subgraphs")[0]);
    let sub = parent.induced(&members).expect("induced builds");
    assert!(sub.atoms().len() >= 2, "non-trivial candidate");
    assert_eq!(
        contains_induced(&parent, &sub, 1_000_000),
        Containment::Contained,
        "large limit contains"
    );
    assert_eq!(
        contains_induced(&parent, &sub, 1),
        Containment::WorkLimit,
        "work_limit = 1 exhausts on a non-trivial case"
    );
    // Empty candidates never claim containment, whatever the limit.
    let empty = MolGraph::new(Vec::new(), Vec::new()).expect("empty builds");
    assert_eq!(
        contains_induced(&parent, &empty, 1_000_000),
        Containment::NotContained,
        "empty is NotContained"
    );
}
