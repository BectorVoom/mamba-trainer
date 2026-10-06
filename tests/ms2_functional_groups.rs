//! MC15 tests: functional groups (Ertl) on typed graphs (host only).
//!
//! Graphs are built by hand from `chem::ATOM_TYPES` ids. No export is read.

use std::collections::{BTreeSet, HashSet};

use mamba3::models::ms2::chem::ATOM_TYPES;
use mamba3::models::ms2::completion::contains_pattern;
use mamba3::models::ms2::completion_data::{
    FunctionalGroupConfig, PatternSource, functional_group_patterns,
};
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::functional_groups::{aromatic_atoms, functional_groups};
use mamba3::models::ms2::graph::MolGraph;

const WORK: usize = 100_000;

fn graph(atoms: Vec<u8>, bonds: Vec<(usize, usize, u8)>) -> MolGraph {
    MolGraph::new(atoms, bonds).unwrap()
}

/// Named fixtures: (atoms, bonds, expected groups, expected aromatic flags).
fn fixtures() -> Vec<(
    &'static str,
    Vec<u8>,
    Vec<(usize, usize, u8)>,
    Vec<Vec<usize>>,
    Vec<bool>,
)> {
    vec![
        (
            "ethanol",
            vec![4, 3, 9],
            vec![(0, 1, 1), (1, 2, 1)],
            vec![vec![2]],
            vec![false; 3],
        ),
        (
            "diethyl ether",
            vec![4, 3, 8, 3, 4],
            vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)],
            vec![vec![2]],
            vec![false; 5],
        ),
        (
            "acetaldehyde",
            vec![4, 2, 8],
            vec![(0, 1, 1), (1, 2, 2)],
            vec![vec![1, 2]],
            vec![false; 3],
        ),
        (
            "acetone",
            vec![4, 1, 4, 8],
            vec![(0, 1, 1), (1, 2, 1), (1, 3, 2)],
            vec![vec![1, 3]],
            vec![false; 4],
        ),
        (
            "acetic acid",
            vec![4, 1, 8, 9],
            vec![(0, 1, 1), (1, 2, 2), (1, 3, 1)],
            vec![vec![1, 2, 3]],
            vec![false; 4],
        ),
        (
            "methyl acetate",
            vec![4, 8, 1, 4, 8],
            vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (2, 4, 2)],
            vec![vec![1, 2, 4]],
            vec![false; 5],
        ),
        (
            "acetamide",
            vec![4, 1, 8, 7],
            vec![(0, 1, 1), (1, 2, 2), (1, 3, 1)],
            vec![vec![1, 2, 3]],
            vec![false; 4],
        ),
        (
            "acetonitrile",
            vec![4, 1, 5],
            vec![(0, 1, 1), (1, 2, 3)],
            vec![vec![1, 2]],
            vec![false; 3],
        ),
        (
            "ethylamine",
            vec![4, 3, 7],
            vec![(0, 1, 1), (1, 2, 1)],
            vec![vec![2]],
            vec![false; 3],
        ),
        (
            "trimethylamine",
            vec![4, 5, 4, 4],
            vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)],
            vec![vec![1]],
            vec![false; 4],
        ),
        (
            "propene",
            vec![4, 2, 3],
            vec![(0, 1, 1), (1, 2, 2)],
            vec![vec![1, 2]],
            vec![false; 3],
        ),
        (
            "propyne",
            vec![4, 1, 2],
            vec![(0, 1, 1), (1, 2, 3)],
            vec![vec![1, 2]],
            vec![false; 3],
        ),
        (
            "benzene",
            vec![2, 2, 2, 2, 2, 2],
            vec![
                (0, 1, 1),
                (0, 5, 2),
                (1, 2, 2),
                (2, 3, 1),
                (3, 4, 2),
                (4, 5, 1),
            ],
            vec![],
            vec![true; 6],
        ),
        (
            "toluene",
            vec![4, 1, 2, 2, 2, 2, 2],
            vec![
                (0, 1, 1),
                (1, 2, 2),
                (1, 6, 1),
                (2, 3, 1),
                (3, 4, 2),
                (4, 5, 1),
                (5, 6, 2),
            ],
            vec![],
            vec![false, true, true, true, true, true, true],
        ),
        (
            "phenol",
            vec![9, 1, 2, 2, 2, 2, 2],
            vec![
                (0, 1, 1),
                (1, 2, 2),
                (1, 6, 1),
                (2, 3, 1),
                (3, 4, 2),
                (4, 5, 1),
                (5, 6, 2),
            ],
            vec![vec![0]],
            vec![false, true, true, true, true, true, true],
        ),
        (
            "anisole",
            vec![4, 8, 1, 2, 2, 2, 2, 2],
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (2, 3, 2),
                (2, 7, 1),
                (3, 4, 1),
                (4, 5, 2),
                (5, 6, 1),
                (6, 7, 2),
            ],
            vec![vec![1]],
            vec![false, false, true, true, true, true, true, true],
        ),
        (
            "pyridine",
            vec![2, 2, 2, 5, 2, 2],
            vec![
                (0, 1, 2),
                (0, 5, 1),
                (1, 2, 1),
                (2, 3, 2),
                (3, 4, 1),
                (4, 5, 2),
            ],
            vec![vec![3]],
            vec![true; 6],
        ),
        (
            "pyrrole",
            vec![2, 2, 2, 6, 2],
            vec![(0, 1, 1), (0, 4, 2), (1, 2, 2), (2, 3, 1), (3, 4, 1)],
            vec![vec![3]],
            vec![true; 5],
        ),
        (
            "furan",
            vec![2, 2, 2, 8, 2],
            vec![(0, 1, 1), (0, 4, 2), (1, 2, 2), (2, 3, 1), (3, 4, 1)],
            vec![vec![3]],
            vec![true; 5],
        ),
        (
            "thiophene",
            vec![2, 2, 2, 13, 2],
            vec![(0, 1, 1), (0, 4, 2), (1, 2, 2), (2, 3, 1), (3, 4, 1)],
            vec![vec![3]],
            vec![true; 5],
        ),
    ]
}

#[test]
fn named_fixtures_match() {
    for (name, atoms, bonds, want_groups, want_aromatic) in fixtures() {
        let parent = graph(atoms, bonds);
        let groups = functional_groups(&parent).unwrap();
        let got: Vec<Vec<usize>> = groups.iter().map(|g| g.atoms.clone()).collect();
        assert_eq!(got, want_groups, "{name}: group atom sets");
        assert_eq!(
            aromatic_atoms(&parent),
            want_aromatic,
            "{name}: aromatic flags"
        );
    }
}

/// F7: the `[O,N,S]1CC1` SMARTS needs single bonds between aliphatic atoms.
/// 2H-azirine (`C1=NC1`) is the C=N pair, not the triangle; saturated
/// oxirane/aziridine/thiirane stay whole triangles.
#[test]
fn unsaturated_triangles_are_not_three_membered_groups() {
    // 2H-azirine: the double bond marks its C=N pair only.
    let azirine = graph(vec![2, 5, 3], vec![(0, 1, 2), (0, 2, 1), (1, 2, 1)]);
    let groups = functional_groups(&azirine).unwrap();
    let got: Vec<Vec<usize>> = groups.iter().map(|g| g.atoms.clone()).collect();
    assert_eq!(got, vec![vec![0, 1]], "azirine: C=N pair, not the triangle");
    // Saturated three-membered rings are unchanged.
    for (name, atoms) in [
        ("oxirane", vec![3, 3, 8]),
        ("aziridine", vec![3, 3, 6]),
        ("thiirane", vec![3, 3, 13]),
    ] {
        let ring = graph(atoms, vec![(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
        let groups = functional_groups(&ring).unwrap();
        let got: Vec<Vec<usize>> = groups.iter().map(|g| g.atoms.clone()).collect();
        assert_eq!(got, vec![vec![0, 1, 2]], "{name}: whole triangle");
    }
}

#[test]
fn hydrogen_counts_follow_the_parent() {
    // Hydroxyl oxygen (id 9, one hydrogen) versus ether oxygen (id 8, none).
    let ethanol = graph(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]);
    let ether = graph(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]);
    let config = FunctionalGroupConfig::default();
    for (parent, want_type) in [(&ethanol, 9u8), (&ether, 8u8)] {
        let draw = functional_group_patterns(parent, &config, 1, "k", 0).unwrap();
        assert_eq!(draw.patterns.len(), 1);
        assert_eq!(draw.patterns[0].graph.atoms(), &[want_type]);
    }
    // Aldehyde carbon (id 2, one hydrogen) versus ketone carbon (id 1, none).
    let aldehyde = graph(vec![4, 2, 8], vec![(0, 1, 1), (1, 2, 2)]);
    let ketone = graph(vec![4, 1, 4, 8], vec![(0, 1, 1), (1, 2, 1), (1, 3, 2)]);
    for (parent, want_carbon) in [(&aldehyde, 2u8), (&ketone, 1u8)] {
        let draw = functional_group_patterns(parent, &config, 1, "k", 0).unwrap();
        assert_eq!(draw.patterns.len(), 1);
        let mut types = draw.patterns[0].graph.atoms().to_vec();
        types.sort_unstable();
        assert!(types.contains(&want_carbon), "carbon type {want_carbon}");
        assert!(types.contains(&8u8), "oxygen present");
    }
    // ATOM_TYPES is consulted (uses the import).
    assert_eq!(ATOM_TYPES.len(), 17);
}

#[test]
fn groups_are_invariant_under_permutation() {
    for (name, atoms, bonds, _, _) in fixtures() {
        let parent = graph(atoms.clone(), bonds.clone());
        let groups = functional_groups(&parent).unwrap();
        let n = parent.atoms().len();
        let perm: Vec<usize> = (0..n).rev().collect();
        let relabeled = parent.permuted(&perm).unwrap();
        let regrouped = functional_groups(&relabeled).unwrap();
        // Map regrouped atoms back through the permutation.
        let mut mapped: Vec<BTreeSet<usize>> = regrouped
            .iter()
            .map(|g| g.atoms.iter().map(|a| perm[*a]).collect())
            .collect();
        let mut want: Vec<BTreeSet<usize>> = groups
            .iter()
            .map(|g| g.atoms.iter().copied().collect())
            .collect();
        mapped.sort_by_key(|s| s.iter().next().copied());
        want.sort_by_key(|s| s.iter().next().copied());
        assert_eq!(mapped, want, "{name}: groups map to groups");
        // Signatures agree as multisets.
        let mut a: Vec<String> = groups.iter().map(|g| g.signature.clone()).collect();
        let mut b: Vec<String> = regrouped.iter().map(|g| g.signature.clone()).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "{name}: signatures agree");
    }
}

#[test]
fn extraction_is_deterministic() {
    let parent = graph(vec![4, 2, 8], vec![(0, 1, 1), (1, 2, 2)]);
    let config = FunctionalGroupConfig::default();
    let first = functional_group_patterns(&parent, &config, 7, "mol", 0).unwrap();
    let again = functional_group_patterns(&parent, &config, 7, "mol", 0).unwrap();
    assert_eq!(first.patterns.len(), again.patterns.len());
    for (a, b) in first.patterns.iter().zip(again.patterns.iter()) {
        assert_eq!(a.parent_atoms, b.parent_atoms);
        assert_eq!(a.graph.atoms(), b.graph.atoms());
        assert_eq!(a.graph.bonds(), b.graph.bonds());
    }
}

/// Ten hydroxyls: ten disconnected C(H3)-O(H1) units, ten one-atom groups.
fn ten_hydroxyls() -> MolGraph {
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    for i in 0..10 {
        atoms.push(4);
        atoms.push(9);
        bonds.push((2 * i, 2 * i + 1, 1));
    }
    graph(atoms, bonds)
}

#[test]
fn all_groups_returned_when_they_fit() {
    let parent = graph(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]);
    let config = FunctionalGroupConfig::default();
    let draw = functional_group_patterns(&parent, &config, 3, "eth", 0).unwrap();
    assert_eq!(draw.groups_found, 1);
    assert_eq!(draw.groups_kept, 1);
    assert!(!draw.truncated);
    assert_eq!(draw.dropped_oversized, 0);
    assert_eq!(draw.patterns.len(), 1);
}

#[test]
fn seeded_fitting_subset_with_truncation() {
    let parent = ten_hydroxyls();
    let config = FunctionalGroupConfig::default();
    assert_eq!(config.max_groups, 8);
    let draw = functional_group_patterns(&parent, &config, 3, "ten", 0).unwrap();
    assert_eq!(draw.groups_found, 10);
    assert_eq!(draw.groups_kept, 8);
    assert!(draw.truncated);
    assert_eq!(draw.patterns.len(), 8);
    // Same seed gives the same subset; another seed gives (likely) another.
    let again = functional_group_patterns(&parent, &config, 3, "ten", 0).unwrap();
    let same: HashSet<Vec<usize>> = draw
        .patterns
        .iter()
        .map(|p| {
            let mut v = p.parent_atoms.clone();
            v.sort_unstable();
            v
        })
        .collect();
    let same_again: HashSet<Vec<usize>> = again
        .patterns
        .iter()
        .map(|p| {
            let mut v = p.parent_atoms.clone();
            v.sort_unstable();
            v
        })
        .collect();
    assert_eq!(same, same_again);
    let mut seen = HashSet::new();
    for seed in 0..16u64 {
        let d = functional_group_patterns(&parent, &config, seed, "ten", 0).unwrap();
        let key: BTreeSet<usize> = d
            .patterns
            .iter()
            .flat_map(|p| p.parent_atoms.clone())
            .collect();
        seen.insert(key);
    }
    assert!(seen.len() >= 2, "seeds spread the fitting subset");
}

#[test]
fn keep_probability_edges_and_fraction() {
    let parent = ten_hydroxyls();
    let mut zero = FunctionalGroupConfig::default();
    zero.keep_probability_percent = 0;
    let draw = functional_group_patterns(&parent, &zero, 5, "ten", 0).unwrap();
    assert_eq!(draw.groups_found, 10);
    assert_eq!(draw.patterns.len(), 0);
    let mut full = FunctionalGroupConfig::default();
    full.keep_probability_percent = 100;
    let draw = functional_group_patterns(&parent, &full, 5, "ten", 0).unwrap();
    assert_eq!(draw.patterns.len(), 8);
    // At 50 over many draws the mean kept is near 5 (generous tolerance).
    let mut half = FunctionalGroupConfig::default();
    half.keep_probability_percent = 50;
    let mut total = 0usize;
    let draws = 200u64;
    for draw_idx in 0..draws {
        let d = functional_group_patterns(&parent, &half, 9, "ten", draw_idx).unwrap();
        total += d.patterns.len();
    }
    let mean = total as f64 / draws as f64;
    assert!(
        (3.5..=6.5).contains(&mean),
        "mean kept at 50 is near 5: {mean}"
    );
}

#[test]
fn aromatic_rings_as_groups() {
    let benzene = graph(
        vec![2, 2, 2, 2, 2, 2],
        vec![
            (0, 1, 1),
            (0, 5, 2),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
        ],
    );
    let cyclohexane = graph(
        vec![3, 3, 3, 3, 3, 3],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 0, 1),
        ],
    );
    let mut with_rings = FunctionalGroupConfig::default();
    with_rings.aromatic_rings_as_groups = true;
    let plain = FunctionalGroupConfig::default();
    assert_eq!(
        functional_group_patterns(&benzene, &plain, 1, "b", 0)
            .unwrap()
            .patterns
            .len(),
        0
    );
    assert_eq!(
        functional_group_patterns(&benzene, &with_rings, 1, "b", 0)
            .unwrap()
            .patterns
            .len(),
        1
    );
    assert_eq!(
        functional_group_patterns(&cyclohexane, &with_rings, 1, "c", 0)
            .unwrap()
            .patterns
            .len(),
        0
    );
}

#[test]
fn pattern_order_is_shuffled_and_contained() {
    // Acetaldehyde's group has two atoms: over many draws the first pattern
    // atom is sometimes not the smallest parent index.
    let parent = graph(vec![4, 2, 8], vec![(0, 1, 1), (1, 2, 2)]);
    let config = FunctionalGroupConfig::default();
    let mut ever_shuffled = false;
    for draw in 0..50u64 {
        let out = functional_group_patterns(&parent, &config, 1, "ald", draw).unwrap();
        assert_eq!(out.patterns.len(), 1);
        let atoms = &out.patterns[0].parent_atoms;
        assert_eq!(atoms.len(), 2);
        let min = atoms.iter().min().unwrap();
        if &atoms[0] != min {
            ever_shuffled = true;
        }
        assert_eq!(
            contains_pattern(&parent, &out.patterns[0].graph, WORK),
            Containment::Contained
        );
    }
    assert!(ever_shuffled, "atom order is shuffled");
    // Pattern order across draws varies for the ten-hydroxyl molecule.
    let parent = ten_hydroxyls();
    let mut seen = HashSet::new();
    for draw in 0..16u64 {
        let out = functional_group_patterns(&parent, &config, 7, "ten", draw).unwrap();
        seen.insert(
            out.patterns
                .iter()
                .map(|p| p.parent_atoms.clone())
                .collect::<Vec<_>>(),
        );
    }
    assert!(seen.len() >= 2, "pattern order varies");
}

#[test]
fn config_validation() {
    FunctionalGroupConfig::default().validate().unwrap();
    for bad in [
        FunctionalGroupConfig {
            max_groups: 9,
            ..FunctionalGroupConfig::default()
        },
        FunctionalGroupConfig {
            max_total_atoms: 25,
            ..FunctionalGroupConfig::default()
        },
        FunctionalGroupConfig {
            max_group_atoms: 25,
            max_total_atoms: 25,
            ..FunctionalGroupConfig::default()
        },
        FunctionalGroupConfig {
            keep_probability_percent: 101,
            ..FunctionalGroupConfig::default()
        },
    ] {
        assert!(bad.validate().is_err());
    }
    // PatternSource round-trips through serde.
    let source = PatternSource::FunctionalGroups(FunctionalGroupConfig::default());
    let text = serde_json::to_string(&source).unwrap();
    let back: PatternSource = serde_json::from_str(&text).unwrap();
    assert_eq!(source, back);
}
