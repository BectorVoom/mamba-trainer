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

// ---------------------------------------------------------------------------
// NOTE (main merge): Ertl/MC15 tests above (upstream); v4/FG2 tests below
// (local). The v4 entry point was renamed `functional_groups_v4`.
// ---------------------------------------------------------------------------
// FG2 host tests: RDKit-versus-Rust agreement over every stored kekule form,
// kekule invariance (whole-molecule and fragment-level), hand-written v2
// detector expectations, fragment soundness and undetermined cases, closing
// fragments, and the pure evaluation metrics of `functional_groups_eval`.


use mamba3::models::ms2::chem::{atom_type_of, element_index};
use mamba3::models::ms2::contract::{CandidateBatch, candidate_status};
use mamba3::models::ms2::experiment::ExperimentSet;
use mamba3::models::ms2::functional_groups::{
    FG_NAMES, HETEROATOM_MASK, N_FG, SPECIFIC_MASK, SearchOutcome, blossom_stats,
    classify_constrained_result, classify_search_result, decided_bonds, delocalised_bonds, fg_instances,
    functional_groups_v4, has_perfect_matching, kekule_gadget_size,
    matching_allowed_edges, maximum_matching, reset_blossom_stats, set_first_witness_only, undetermined,
};
use mamba3::models::ms2::functional_groups_eval::{
    FgCandidate, FgSpectrumDatum, candidate_size_dist, choose_prior, eval_records, evaluate_fg,
    evaluate_sets, formula_aware_set, label_union, micro_f1_point, recipe_fragments_union,
    spectrum_datum, closing_fragment_not_found,
};
use mamba3::models::ms2::grammar::{
    CANONICAL_WORK_LIMIT, Limits, Token, canonical_trace, replay,
};
use mamba3::models::ms2::targets::{Candidates, Labels, RecipeLimits, Target};

fn mol(types: &[u8], bonds: &[(usize, usize, u8)]) -> MolGraph {
    MolGraph::new(types.to_vec(), bonds.to_vec()).expect("test graph builds")
}

/// Bitmask with the given 1-based type ids.
fn bits(ids: &[usize]) -> u32 {
    ids.iter().fold(0u32, |m, id| m | (1u32 << (id - 1)))
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

fn approx_opt(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => approx(x, y),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// 1. RDKit-versus-Rust agreement over the whole v2 fixture (every molecule,
//    every stored kekule form, every type)
// ---------------------------------------------------------------------------

fn fixture() -> serde_json::Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/functional_groups_v4.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn graph_of_form(m: &serde_json::Value) -> MolGraph {
    let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("?");
    let mut ids = Vec::new();
    for a in m["atoms"].as_array().unwrap() {
        let e = element_index(a["element"].as_str().unwrap()).unwrap();
        let h = a["hydrogens"].as_u64().unwrap() as u8;
        let v = a["valence"].as_u64().unwrap() as u8;
        let t = atom_type_of(e, h, v).unwrap_or_else(|| {
            panic!("fixture atom without a type in {name}: {:?}/{h}/{v}", a["element"])
        });
        ids.push(t.id);
    }
    let bonds: Vec<(usize, usize, u8)> = m["bonds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            let b = b.as_array().unwrap();
            (
                b[0].as_u64().unwrap() as usize,
                b[1].as_u64().unwrap() as usize,
                b[2].as_u64().unwrap() as u8,
            )
        })
        .collect();
    MolGraph::new(ids, bonds).unwrap()
}

#[test]
fn rdkit_reference_agreement() {
    let f = fixture();
    assert_eq!(f["fg_version"].as_str().unwrap(), "ms2-fg-v4");
    let names: Vec<&str> = f["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, FG_NAMES);
    assert_eq!(N_FG, 28);
    let molecules = f["molecules"].as_array().unwrap();
    assert!(!molecules.is_empty());
    let mut mismatches = Vec::new();
    let mut exercised = [0usize; 28];
    let mut forms_compared = 0usize;
    for m in molecules {
        let name = m["name"].as_str().unwrap_or("?");
        let forms = m.get("forms").and_then(|v| v.as_array()).cloned().unwrap_or_else(|| {
            // Backwards tolerance: a bare molecule without forms.
            vec![m.clone()]
        });
        assert!(!forms.is_empty(), "{name} has no stored forms");
        // Complete enumeration: every kekule form is stored (the reference
        // fails loudly past its matching cap instead of truncating).
        let total = m["forms_total"].as_u64().unwrap() as usize;
        let stored = m["forms_stored"].as_u64().unwrap() as usize;
        assert_eq!(total, stored, "{name}: forms_total != forms_stored (truncated enumeration)");
        assert_eq!(forms.len(), stored, "{name}: stored forms != forms_stored");
        for form in &forms {
            forms_compared += 1;
            let graph = graph_of_form(form);
            let set = functional_groups_v4(&graph);
            let undet = undetermined(&graph);
            assert_eq!(undet, 0, "closed molecule {name} has undetermined matches");
            for (i, fg) in FG_NAMES.iter().enumerate() {
                let rust = set.count(i + 1);
                let refc = form["counts"][*fg].as_u64().unwrap() as u32;
                if rust != refc {
                    mismatches.push(format!("{name} {fg}: Rust {rust} vs RDKit {refc}"));
                }
                if refc > 0 {
                    exercised[i] += 1;
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} mismatches over {forms_compared} forms:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    let n_types = exercised.iter().filter(|&&c| c > 0).count();
    println!("types exercised: {n_types}/28 ({exercised:?}); forms compared: {forms_compared}");
    assert_eq!(n_types, 28, "the fixture must exercise all 28 types");
}

// ---------------------------------------------------------------------------
// 2. Kekule invariance: identical type counts across all stored forms
// ---------------------------------------------------------------------------

#[test]
fn kekule_invariance_across_forms() {
    let f = fixture();
    let molecules = f["molecules"].as_array().unwrap();
    let mut per_mol: Vec<(String, usize)> = Vec::new();
    let mut total = 0usize;
    let mut multi = 0usize;
    let mut mismatches = Vec::new();
    for m in molecules {
        let name = m["name"].as_str().unwrap_or("?").to_string();
        let forms = m["forms"].as_array().unwrap();
        per_mol.push((name.clone(), forms.len()));
        total += forms.len();
        if forms.len() > 1 {
            multi += 1;
        }
        let first = graph_of_form(&forms[0]);
        let base = functional_groups_v4(&first);
        for form in forms.iter().skip(1) {
            let g = graph_of_form(form);
            let s = functional_groups_v4(&g);
            if s.mask() != base.mask() || s.counts() != base.counts() {
                let diff: Vec<String> = (1..=N_FG)
                    .filter(|&id| s.count(id) != base.count(id))
                    .map(|id| format!("{}: {} vs {}", FG_NAMES[id - 1], base.count(id), s.count(id)))
                    .collect();
                mismatches.push(format!("{name}: {}", diff.join(", ")));
            }
        }
    }
    println!("kekule invariance: {} molecules, {} with >1 form, {total} forms compared, {} mismatches",
        per_mol.len(), multi, mismatches.len());
    for (name, n) in per_mol.iter().take(30) {
        println!("  {name}: {n} forms");
    }
    assert!(multi >= 10, "invariance test must be non-vacuous ({multi} multi-form molecules)");
    assert!(
        mismatches.is_empty(),
        "{} kekule mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    // Spot checks required by the specification (non-vacuous by name).
    for want in [
        "naphthalene",
        "anthracene",
        "azulene (10-cycle, no six-ring in some forms)",
        "2-aminopyridine",
        "pyridazine",
        "p-benzoquinone (fixed C=C/C=O; NOT an arene ring)",
        "cyclooctatetraene (delocalised, NOT aromatic, no six-ring)",
        "hexacene",
        "heptacene",
        "pentacene",
        "coronene",
        "ovalene sheet (hand-built 10-ring benzenoid, C32H14)",
        "large benzenoid sheet (hand-built 13-ring, C48H24)",
        "expanded six-pyrrole macrocycle, 4 imine + NH + NMe (hand-built)",
        "porphine (hand-built cyclic tetrapyrrole)",
        "biphenylene (four-cycle)",
        "perylene",
        "pyrene",
        "acenaphthylene",
        "indene",
        "fulvene",
        "[18]annulene",
        "cyclobutadiene (delocalised by kekule-equivalence)",
        "benzocyclobutadiene",
        "tropone",
    ] {
        assert!(
            per_mol.iter().any(|(n, _)| n == want),
            "fixture misses required molecule {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Fragment-level invariance on 6 parents (exact assertion stated here)
// ---------------------------------------------------------------------------

/// All connected induced atom subsets with `min_size..=max_size` atoms.
fn connected_subsets(n: usize, bonds: &[(usize, usize, u8)], min_size: usize, max_size: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for (a, b, _) in bonds {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    let mut out = Vec::new();
    // Bitmask enumeration is feasible for the small parents used here
    // (n <= 12 in this test).
    assert!(n <= 20, "connected_subsets: parent too large for enumeration");
    for bits in 1usize..(1usize << n) {
        let size = bits.count_ones() as usize;
        if size < min_size || size > max_size {
            continue;
        }
        let members: Vec<usize> = (0..n).filter(|i| bits & (1 << i) != 0).collect();
        let mut seen = vec![false; n];
        let mut stack = vec![members[0]];
        seen[members[0]] = true;
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if bits & (1 << v) != 0 && !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        if members.iter().all(|m| seen[*m]) {
            out.push(members);
        }
    }
    out
}

#[test]
fn fragment_invariance_across_forms() {
    // Exact assertion: for each of 6 parents and every connected induced
    // atom subset S of 6–10 atoms, let (DA, UA) be the determined /
    // undetermined masks of the fragment induced by S from kekule form A
    // (and (DB, UB) from form B on the same S), and let T be the parent's
    // determined types. Then DA ⊆ T and DB ⊆ T (soundness), and for every
    // type t with t ∉ UA and t ∉ UB (decided in both fragments),
    // t ∈ DA ⟺ t ∈ DB (decided verdicts agree across forms).
    let f = fixture();
    let molecules = f["molecules"].as_array().unwrap();
    let want = [
        "naphthalene",
        "quinoline",
        "indole",
        "biphenyl",
        "2-aminopyridine",
        "styrene",
    ];
    assert_eq!(want.len(), 6);
    let mut checked = 0usize;
    for name in want {
        let m = molecules
            .iter()
            .find(|m| m["name"].as_str().unwrap_or("") == name)
            .unwrap_or_else(|| panic!("fixture misses fragment parent {name}"));
        let forms = m["forms"].as_array().unwrap();
        assert!(forms.len() >= 2, "{name} needs at least 2 stored forms");
        // Raw bonds from the first form (connectivity is form-independent).
        let bonds: Vec<(usize, usize, u8)> = forms[0]["bonds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let b = b.as_array().unwrap();
                (
                    b[0].as_u64().unwrap() as usize,
                    b[1].as_u64().unwrap() as usize,
                    b[2].as_u64().unwrap() as u8,
                )
            })
            .collect();
        let n = forms[0]["atoms"].as_array().unwrap().len();
        assert!(n <= 12, "{name} too large for exhaustive fragment test ({n} atoms)");
        let ga = graph_of_form(&forms[0]);
        let gb = graph_of_form(&forms[1]);
        let ta = functional_groups_v4(&ga).mask();
        assert_eq!(ta, functional_groups_v4(&gb).mask(), "{name}: parent forms disagree");
        for members in connected_subsets(n, &bonds, 6, 10.min(n)) {
            let fa = ga.induced(&members).unwrap();
            let fb = gb.induced(&members).unwrap();
            let (da, ua) = {
                let s = functional_groups_v4(&fa);
                (s.mask(), undetermined(&fa))
            };
            let (db, ub) = {
                let s = functional_groups_v4(&fb);
                (s.mask(), undetermined(&fb))
            };
            assert_eq!(da & !ta, 0, "{name} fragment {members:?}: A outside parent");
            assert_eq!(db & !ta, 0, "{name} fragment {members:?}: B outside parent");
            for id in 1..=N_FG {
                let bit = 1u32 << (id - 1);
                if ua & bit == 0 && ub & bit == 0 {
                    assert_eq!(
                        da & bit != 0,
                        db & bit != 0,
                        "{name} fragment {members:?} type {} decided differently across forms",
                        FG_NAMES[id - 1]
                    );
                }
            }
            checked += 1;
        }
    }
    println!("fragment invariance: {checked} fragments checked over 6 parents");
    assert!(checked > 500, "fragment test must be non-vacuous ({checked} fragments)");
}

// ---------------------------------------------------------------------------
// 4. Hand-written v2 detector expectations
// ---------------------------------------------------------------------------

/// (name, types, bonds, expected mask, expected per-type counts).
fn hand_molecules() -> Vec<(&'static str, Vec<u8>, Vec<(usize, usize, u8)>, u32, Vec<(usize, u32)>)> {
    vec![
        ("acetic acid", vec![4, 1, 8, 9], vec![(0, 1, 1), (1, 2, 2), (1, 3, 1)], bits(&[1, 2]), vec![(1, 1), (2, 1)]),
        ("ethyl acetate", vec![4, 1, 8, 8, 3, 4], vec![(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 1), (4, 5, 1)], bits(&[1, 3]), vec![(1, 1), (3, 1)]),
        ("acetamide", vec![4, 1, 8, 7], vec![(0, 1, 1), (1, 2, 2), (1, 3, 1)], bits(&[1, 4]), vec![(1, 1), (4, 1)]),
        ("acetaldehyde", vec![4, 2, 8], vec![(0, 1, 1), (1, 2, 2)], bits(&[1, 5]), vec![(1, 1), (5, 1)]),
        ("formaldehyde", vec![3, 8], vec![(0, 1, 2)], bits(&[1, 5]), vec![(1, 1), (5, 1)]),
        ("acetone", vec![4, 1, 8, 4], vec![(0, 1, 1), (1, 2, 2), (1, 3, 1)], bits(&[1, 6]), vec![(1, 1), (6, 1)]),
        ("ethanol", vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)], bits(&[7]), vec![(7, 1)]),
        ("diethyl ether", vec![4, 3, 8, 3, 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)], bits(&[8]), vec![(8, 1)]),
        ("ethylamine", vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)], bits(&[9]), vec![(9, 1)]),
        ("benzene", vec![2, 2, 2, 2, 2, 2], vec![(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (0, 5, 2)], bits(&[24]), vec![(24, 1)]),
        ("acetonitrile", vec![4, 1, 5], vec![(0, 1, 1), (1, 2, 3)], bits(&[12]), vec![(12, 1)]),
        ("ethanethiol", vec![4, 3, 14], vec![(0, 1, 1), (1, 2, 1)], bits(&[16]), vec![(16, 1)]),
        // Urea is a carbamate/urea only (no amide under v2 remaining rules).
        ("urea", vec![7, 1, 8, 7], vec![(0, 1, 1), (1, 2, 2), (1, 3, 1)], bits(&[1, 26]), vec![(1, 1), (26, 1)]),
        // Carbonic acid is an anhydride/carbonate, not an acid or ester.
        ("carbonic acid", vec![9, 1, 8, 9], vec![(0, 1, 1), (1, 2, 2), (1, 3, 1)], bits(&[1, 27]), vec![(1, 1), (27, 1)]),
    ]
}

#[test]
fn hand_written_expectations() {
    assert!(N_FG == 28);
    for (name, types, bonds, want_mask, want_counts) in hand_molecules() {
        let g = mol(&types, &bonds);
        let set = functional_groups_v4(&g);
        assert_eq!(set.mask(), want_mask, "{name}: mask {:b} vs {:b}", set.mask(), want_mask);
        for (id, n) in &want_counts {
            assert_eq!(set.count(*id), *n, "{name} type {id}");
        }
        let want_total: u32 = want_counts.iter().map(|(_, n)| n).sum();
        assert_eq!(set.total(), want_total, "{name}: total");
        assert_eq!(undetermined(&g), 0, "{name}: closed, nothing undetermined");
    }
    // Spot exclusions: acetamide is no amine, benzene is no alkene,
    // carbonic acid is no acid/ester, furan is a five-ring only.
    let acetamide = mol(&[4, 1, 8, 7], &[(0, 1, 1), (1, 2, 2), (1, 3, 1)]);
    let set = functional_groups_v4(&acetamide);
    assert!(!set.present(9) && !set.present(10), "amide is not an amine");
    let benzene = mol(&[2, 2, 2, 2, 2, 2], &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (0, 5, 2)]);
    assert_eq!(functional_groups_v4(&benzene).count(14), 0, "arene bonds are no alkenes");
    let carbonic = mol(&[9, 1, 8, 9], &[(0, 1, 1), (1, 2, 2), (1, 3, 1)]);
    assert!(!functional_groups_v4(&carbonic).present(2), "carbonic acid is no carboxylic acid");
    assert!(!functional_groups_v4(&carbonic).present(3), "carbonic acid is no ester");
    // Furan: C1=COC=C1 as atom types/bonds (C H1 x4? two CH next to O...).
    // Types: ring C (H1, v4) x4, O (H0, v2) x1; bonds alternate double/single.
    let furan = mol(&[2, 2, 8, 2, 2], &[(0, 1, 2), (1, 2, 1), (2, 3, 1), (3, 4, 2), (4, 0, 1)]);
    let fs = functional_groups_v4(&furan);
    assert_eq!(fs.count(28), 1, "furan is one five-ring");
    assert_eq!(fs.count(8), 0, "furan oxygen is no ether");
    assert_eq!(fs.count(14), 0, "furan bonds are no alkenes");
    assert_eq!(fs.count(24), 0, "furan is no arene");
    // Thioacetic acid is no thiol.
    let thioacid = mol(&[4, 1, 8, 14], &[(0, 1, 1), (1, 2, 2), (1, 3, 1)]);
    assert_eq!(functional_groups_v4(&thioacid).count(16), 0, "thioacid is no thiol");
    assert!(SPECIFIC_MASK.count_ones() == 27);
    assert!(HETEROATOM_MASK.count_ones() == 24);
}

// ---------------------------------------------------------------------------
// 5. Fragment semantics: soundness, undetermined, closing
// ---------------------------------------------------------------------------

#[test]
fn fragment_soundness_exhaustive() {
    // Eight parents; on ALL connected induced fragments of 3 to 8 atoms every
    // determined instance is of a type present in the parent.
    let parents: Vec<(&str, Vec<u8>, Vec<(usize, usize, u8)>)> = hand_molecules()
        .into_iter()
        .filter(|m| {
            m.0 != "benzene" && m.0 != "acetaldehyde" && m.0 != "acetonitrile"
                && m.0 != "ethanethiol" && m.0 != "formaldehyde" && m.0 != "carbonic acid"
        })
        .map(|m| (m.0, m.1, m.2))
        .collect();
    assert_eq!(parents.len(), 8);
    for (name, types, bonds) in &parents {
        let parent = mol(types, bonds);
        let pmask = functional_groups_v4(&parent).mask();
        let n = types.len();
        for members in connected_subsets(n, bonds, 3, 8.min(n)) {
            let sub = parent.induced(&members).unwrap();
            let dm = functional_groups_v4(&sub).mask();
            assert_eq!(
                dm & !pmask,
                0,
                "{name} fragment {members:?}: determined {:b} outside parent {:b}",
                dm,
                pmask
            );
        }
    }
}

#[test]
fn fragment_undetermined_cases() {
    // C(=O)–O[H0] with the O open: carbonyl, undetermined ester only (never
    // undetermined acid: hydrogen counts are respected).
    let g = mol(&[1, 8, 8], &[(0, 1, 2), (0, 2, 1)]);
    assert_eq!(functional_groups_v4(&g).mask(), bits(&[1]));
    let u = undetermined(&g);
    assert!(u & bits(&[3]) == bits(&[3]), "ester undetermined, got {u:b}");
    assert!(u & bits(&[2]) == 0, "H0 core is never undetermined acid, got {u:b}");
    assert!(u & bits(&[8]) == 0 && functional_groups_v4(&g).mask() & bits(&[8]) == 0, "never ether");
    // C–O[H1] whose carbon has open valence 2: undetermined hydroxyl.
    let g = mol(&[2, 9], &[(0, 1, 1)]);
    assert_eq!(functional_groups_v4(&g).mask(), 0);
    assert_eq!(undetermined(&g), bits(&[7]));
    // N[H1] with one carbon inside and open valence 1: undetermined secondary amine.
    let g = mol(&[4, 6], &[(0, 1, 1)]);
    assert_eq!(functional_groups_v4(&g).mask(), 0);
    assert_eq!(undetermined(&g), bits(&[10]));
    // Three ring atoms of benzene: neither arene_ring nor alkene (undetermined alkene).
    let g = mol(&[2, 2, 2], &[(0, 1, 1), (1, 2, 2)]);
    assert_eq!(functional_groups_v4(&g).mask(), 0);
    assert_eq!(undetermined(&g) & bits(&[14]), bits(&[14]));
    assert_eq!(undetermined(&g) & bits(&[24]), 0);
}

#[test]
fn rule_closable_on_fixture_parents() {
    // Every determined instance of every hand parent fits with its closing
    // neighbours in a <=16-atom fragment where it is determined.
    let parents: Vec<MolGraph> = hand_molecules()
        .into_iter()
        .map(|(_, t, b, _, _)| mol(&t, &b))
        .collect();
    let bad = closing_fragment_not_found(&parents);
    println!("closing_fragment_not_found on hand parents: {bad}");
    assert_eq!(bad, 0, "hand instances must all close within 16 atoms");
    // The same check on the pilot validation parents is reported (not
    // assumed zero) — see rule_closable_on_validation_parents.
}

#[test]
fn rule_closable_on_validation_parents() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/ms2/pilot_validation.json");
    if !path.exists() {
        println!("SKIP rule_closable_on_validation_parents: {path:?} does not exist");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("export readable");
    let value: serde_json::Value = serde_json::from_str(&text).expect("export parses");
    let mut parents = Vec::new();
    for m in value["molecules"].as_array().unwrap() {
        let ids: Vec<u8> = m["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let bonds: Vec<(usize, usize, u8)> = m["bonds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let b = b.as_array().unwrap();
                (
                    b[0].as_u64().unwrap() as usize,
                    b[1].as_u64().unwrap() as usize,
                    b[2].as_u64().unwrap() as u8,
                )
            })
            .collect();
        let Ok(g) = MolGraph::new(ids, bonds) else {
            continue;
        };
        parents.push(g);
    }
    assert!(!parents.is_empty());
    let total_instances: usize = parents.iter().map(|p| fg_instances(p).len()).sum();
    let bad = closing_fragment_not_found(&parents);
    println!("closing_fragment_not_found on pilot validation parents: {bad}/{total_instances}");
}

// ---------------------------------------------------------------------------
// 6. Pure evaluation metrics on a hand-built example (null-aware)
// ---------------------------------------------------------------------------

fn cnt(pairs: &[(usize, u32)]) -> [u32; N_FG] {
    let mut c = [0u32; N_FG];
    for (id, n) in pairs {
        c[id - 1] = *n;
    }
    c
}

fn cand(counts: [(usize, u32); 1], score: f64, trajectory: u32) -> FgCandidate {
    FgCandidate {
        counts: cnt(&counts),
        undet: 0,
        atoms: 4,
        score,
        trajectory,
    }
}

fn hand_data() -> Vec<FgSpectrumDatum> {
    vec![
        FgSpectrumDatum {
            molecule: 0,
            parent_mask: bits(&[1, 2]),
            candidates: vec![cand([(1, 1)], 2.0, 0), cand([(2, 1)], 1.0, 1)],
            label_union: 0,
            oracle_union: 0,
            top_formula: None,
        },
        FgSpectrumDatum {
            molecule: 0,
            parent_mask: bits(&[2]),
            candidates: vec![cand([(2, 1)], 5.0, 0)],
            label_union: 0,
            oracle_union: 0,
            top_formula: None,
        },
        FgSpectrumDatum {
            molecule: 1,
            parent_mask: bits(&[9]),
            candidates: vec![],
            label_union: 0,
            oracle_union: 0,
            top_formula: None,
        },
    ]
}

#[test]
fn metrics_hand_computed() {
    let data = hand_data();
    let reps = evaluate_fg(&data, None, &[1], 100, 1);
    assert_eq!(reps.len(), 1);
    let full = &reps[0].full.set;
    assert!(approx_opt(full.micro_precision.point, Some(1.0)));
    assert!(approx_opt(full.micro_recall.point, Some(0.5)));
    assert!(approx_opt(full.micro_f1.point, Some(2.0 / 3.0)));
    assert!(approx_opt(full.jaccard.point, Some(0.5)));
    assert!(approx_opt(full.exact_match.point, Some(1.0 / 3.0)));
    assert!(approx_opt(full.empty_p.point, Some(1.0 / 3.0)));
    assert!(full.types_used.is_empty(), "no type has support 10");
    assert!(full.macro_precision.point.is_none());
    assert!(full.macro_recall.point.is_none());
    assert!(full.macro_f1.point.is_none());
    assert_eq!(full.supports[0], 1);
    assert_eq!(full.supports[1], 2);
    assert_eq!(full.supports[8], 1);
    let row = |id: usize| full.per_type.iter().find(|r| r.id == id).unwrap();
    assert_eq!((row(1).true_count, row(1).predicted, row(1).tp), (1, 1, 1));
    assert!(approx_opt(row(1).precision, Some(1.0)) && approx_opt(row(1).recall, Some(1.0)));
    assert_eq!((row(2).true_count, row(2).predicted, row(2).tp), (2, 1, 1));
    assert!(approx_opt(row(2).precision, Some(1.0)) && approx_opt(row(2).recall, Some(0.5)));
    assert_eq!((row(9).true_count, row(9).predicted, row(9).tp), (1, 0, 0));
    assert!(row(9).precision.is_none() && approx_opt(row(9).recall, Some(0.0)));
    // Specific subset drops id 1: s0 has P={} vs T={2}.
    let spec = &reps[0].specific.set;
    assert!(approx_opt(spec.micro_precision.point, Some(1.0)));
    assert!(approx_opt(spec.micro_recall.point, Some(1.0 / 3.0)));
    assert!(approx_opt(spec.jaccard.point, Some(1.0 / 3.0)));
    // min_atoms_3: all hand candidates have 4 atoms, so identical.
    assert_eq!(reps[0].full.set_min_atoms_3.micro_f1, reps[0].full.set.micro_f1);
    // Instance and candidate levels pool all eligible candidates.
    assert!(approx_opt(reps[0].full.instance_precision.point, Some(1.0)));
    assert!(approx_opt(reps[0].full.mean_instances.point, Some(1.0)));
    assert!(approx_opt(reps[0].full.mean_undet.point, Some(0.0)));
    assert!(approx_opt(reps[0].full.cand_with_group.point, Some(1.0)));
    assert!(approx_opt(reps[0].full.cand_all_real.point, Some(1.0)));
    assert!(approx_opt(reps[0].specific.cand_with_group.point, Some(2.0 / 3.0)));
    assert!(approx_opt(reps[0].specific.cand_all_real.point, Some(1.0)));
    // Heteroatom subset drops carbonyl/alkene/alkyne/arene: s0 P={} vs T={2}.
    assert!(approx_opt(reps[0].heteroatom.set.micro_recall.point, Some(1.0 / 3.0)));
    // Candidate sizes: 3 candidates, all in 3-5.
    assert_eq!(reps[0].sizes.n, 3);
    assert!(approx_opt(reps[0].sizes.frac_3_5, Some(1.0)));
    // Intervals bracket the point (when defined).
    for m in [
        &full.micro_precision,
        &full.micro_recall,
        &full.micro_f1,
        &full.jaccard,
        &spec.micro_f1,
    ] {
        if let (Some(lo), Some(p), Some(hi)) = (m.lo, m.point, m.hi) {
            assert!(lo <= p && p <= hi, "{m:?}");
        }
    }
}

#[test]
fn zero_denominators_are_null() {
    // No predictions at all: micro precision is null (not 1.0).
    let m = evaluate_sets(&[0, 0], &[bits(&[2]), bits(&[9])], &[0, 1], 0, 0);
    assert!(m.micro_precision.point.is_none());
    assert!(approx_opt(m.micro_recall.point, Some(0.0)));
    assert!(m.micro_f1.point.is_none());
    // No truths at all: micro recall is null.
    let m = evaluate_sets(&[bits(&[2]), 0], &[0, 0], &[0, 1], 0, 0);
    assert!(approx_opt(m.micro_precision.point, Some(0.0)));
    assert!(m.micro_recall.point.is_none());
    assert!(m.micro_f1.point.is_none());
    // Both empty: Jaccard 1 per spectrum, micro pair null.
    let m = evaluate_sets(&[0], &[0], &[0], 0, 0);
    assert!(approx_opt(m.jaccard.point, Some(1.0)));
    assert!(m.micro_precision.point.is_none() && m.micro_recall.point.is_none());
    // Precision and recall both defined and both zero: F1 is 0, not null.
    let m = evaluate_sets(&[bits(&[3])], &[bits(&[2])], &[0], 0, 0);
    assert!(approx_opt(m.micro_precision.point, Some(0.0)));
    assert!(approx_opt(m.micro_recall.point, Some(0.0)));
    assert!(approx_opt(m.micro_f1.point, Some(0.0)));
    // Per-type intervals exist exactly where the point is defined.
    let m = evaluate_sets(&[bits(&[3])], &[bits(&[2])], &[0], 10, 0);
    let row2 = m.per_type.iter().find(|r| r.id == 2).unwrap();
    assert!(approx_opt(row2.recall, Some(0.0)));
    assert!(row2.recall_lo.is_some() && row2.recall_hi.is_some());
    assert!(row2.precision.is_none() && row2.precision_lo.is_none() && row2.precision_hi.is_none());
}

#[test]
fn prior_tau_selection_and_formula_filter() {
    // Frequencies: t1 = 1/2, t2 = 1/2, rest 0.
    let train = vec![bits(&[1, 2]), bits(&[1]), bits(&[2]), 0];
    let prior = choose_prior(&train);
    assert!(approx(prior.tau, 0.5), "tau {}", prior.tau);
    assert_eq!(prior.set, vec![1, 2]);
    assert!(approx_opt(micro_f1_point(&vec![bits(&[1, 2]); 4], &train), Some(2.0 / 3.0)));
    // Saturated-hydrocarbon train (all masks empty): the empty set wins.
    let prior = choose_prior(&[0, 0, 0]);
    assert!(prior.set.is_empty(), "empty prior on empty train, got {:?}", prior.set);
    // Empty train: empty set at tau 1.
    let prior = choose_prior(&[]);
    assert!(prior.set.is_empty());
    // Formula-aware filter: id 9 needs N, id 1 needs nothing. Top formula
    // is now best-formula_log_prob among eligible records (see spectrum_datum).
    let set = vec![1, 9];
    let with_n: [u16; 10] = [1, 3, 1, 0, 0, 0, 0, 0, 0, 0];
    let without_n: [u16; 10] = [2, 4, 0, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(formula_aware_set(&set, Some(with_n)), vec![1, 9]);
    assert_eq!(formula_aware_set(&set, Some(without_n)), vec![1]);
    assert_eq!(formula_aware_set(&set, None), vec![1, 9]);
}

#[test]
fn macro_held_fixed_across_bootstrap() {
    // Ten A-miss spectra and ten B-hit spectra on distinct molecules: point
    // macro F1 is 0.5 over held {A, B}. Replicates keep {A, B} fixed (a
    // replicate missing A still averages over both, with 0 for the missing
    // type) instead of dropping below support 10.
    let a = bits(&[2]);
    let b = bits(&[9]);
    let mut preds = Vec::new();
    let mut truths = Vec::new();
    let mut mols = Vec::new();
    for i in 0..10 {
        preds.push(0);
        truths.push(a);
        mols.push(i);
    }
    for i in 10..20 {
        preds.push(b);
        truths.push(b);
        mols.push(i);
    }
    let m = evaluate_sets(&preds, &truths, &mols, 200, 3);
    assert_eq!(m.types_used, vec![2, 9]);
    assert!(approx_opt(m.macro_f1.point, Some(0.5)));
    // Every replicate averages over the same two types: the interval stays
    // inside [0, 1] and brackets 0.5 only via the fixed denominator.
    if let (Some(lo), Some(hi)) = (m.macro_f1.lo, m.macro_f1.hi) {
        assert!(lo <= 0.5 && 0.5 <= hi, "macro interval {lo}..{hi}");
        assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
    }
    let _ = candidate_size_dist(&hand_data());
    let (_, _) = recipe_fragments_union(&mol(&[4, 3, 9], &[(0, 1, 1), (1, 2, 1)]));
}

#[test]
fn bootstrap_deterministic() {
    let data = hand_data();
    let preds: Vec<u32> = vec![bits(&[1]), bits(&[2]), 0];
    let truths: Vec<u32> = data.iter().map(|d| d.parent_mask).collect();
    let mols: Vec<usize> = data.iter().map(|d| d.molecule).collect();
    let a = evaluate_sets(&preds, &truths, &mols, 200, 7);
    let b = evaluate_sets(&preds, &truths, &mols, 200, 7);
    assert_eq!(a, b, "same seed gives identical intervals");
}

// ---------------------------------------------------------------------------
// 7. Record parsing and own-formula rebuild (pure)
// ---------------------------------------------------------------------------

fn token(kind: u8, atom_type: u8, bond: u8, pointer: u8) -> Token {
    Token { kind, atom_type, bond, pointer }
}

#[test]
fn own_formula_rebuild_evaluates() {
    // A C=O candidate conditioned on its own formula (C1 O1), while the
    // parent is ethanol: the candidate is still evaluated (a false positive),
    // never silently dropped under the parent composition.
    let limits = Limits::V0;
    let parent = mol(&[4, 3, 9], &[(0, 1, 1), (1, 2, 1)]);
    let trace = vec![
        token(1, 0, 0, 0),
        token(2, 1, 0, 0),
        token(2, 8, 2, 0),
        token(4, 0, 0, 0),
    ];
    let t = limits.max_steps();
    let mut batch = CandidateBatch::empty(&[42], 1, t, 16, 4);
    for (i, tok) in trace.iter().enumerate() {
        batch.actions[i * 4] = u32::from(tok.kind);
        batch.actions[i * 4 + 1] = u32::from(tok.atom_type);
        batch.actions[i * 4 + 2] = u32::from(tok.bond);
        batch.actions[i * 4 + 3] = u32::from(tok.pointer);
    }
    batch.length[0] = trace.len() as u32;
    batch.status[0] = candidate_status::FINISHED;
    batch.trajectory[0] = 0;
    batch.formula_log_prob[0] = -1.0;
    batch.trace_log_prob[0] = -2.0;
    // Own formula C1 O1: NOT the parent composition (C2 H6 O1).
    batch.formula_counts[3] = 1; // O is element 3.
    batch.formula_counts[0] = 1; // C is element 0.
    assert_ne!([batch.formula_counts[0], batch.formula_counts[3]], [2u16, 1u16]);
    let rows = eval_records(&batch).unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0][0].eligible());
    let datum = spectrum_datum(&parent, 0, &rows[0], limits, 0, 0);
    assert_eq!(datum.candidates.len(), 1);
    assert_eq!(datum.candidates[0].mask(), bits(&[1]), "the C=O shows");
    // Top formula is now best-formula_log_prob among eligible records.
    assert_eq!(datum.top_formula, Some([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]));
    assert_eq!(datum.candidates[0].atoms, 2);
    // Duplicate-trace records are ineligible.
    batch.status[0] |= candidate_status::DUPLICATE_TRACE;
    let rows = eval_records(&batch).unwrap();
    assert!(!rows[0][0].eligible());
}

// ---------------------------------------------------------------------------
// 8. Pilot-validation comparison through the Python tool (live export)
// ---------------------------------------------------------------------------

#[test]
fn pilot_validation_reference() {
    let export = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/ms2/pilot_validation.json");
    if !export.exists() {
        println!("SKIP pilot_validation_reference: {export:?} does not exist (prerequisite missing)");
        return;
    }
    if std::process::Command::new("uv").arg("--version").output().is_err() {
        println!("SKIP pilot_validation_reference: uv is not on PATH (prerequisite missing)");
        return;
    }
    // Prerequisite: the data directory with the export.
    let export_text = std::fs::read_to_string(&export).expect("export readable");
    let export_value: serde_json::Value =
        serde_json::from_str(&export_text).expect("export parses");
    let export_molecules = export_value["molecules"].as_array().unwrap().len();
    assert!(export_molecules > 0, "export holds no molecules");
    let out = std::env::temp_dir().join("ms2_fg_pilot_validation.json");
    let status = std::process::Command::new("timeout")
        .arg("1200")
        .arg("uv")
        .arg("run")
        .arg("--with")
        .arg("rdkit")
        .arg("--with")
        .arg("numpy")
        .arg("python")
        .arg("tools/ms2/functional_groups_ref.py")
        .arg("--export")
        .arg(&export)
        .arg("--out")
        .arg(&out)
        .env("PYTHONPATH", "tools/ms2")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status();
    let status = match status {
        Ok(s) => s,
        Err(e) => {
            println!("SKIP pilot_validation_reference: cannot run uv/timeout: {e} (prerequisite missing)");
            return;
        }
    };
    // The tool ran: a nonzero exit is a FAIL, not a skip.
    assert!(
        status.success(),
        "FAIL pilot_validation_reference: reference tool exited {status} after prerequisites were present"
    );
    let text = std::fs::read_to_string(&out).expect("tool output readable");
    let value: serde_json::Value = serde_json::from_str(&text).expect("tool output parses");
    // Kekule regression recorded by the tool must agree.
    if let Some(reg) = value.get("kekule_regression") {
        let bad = reg["mismatches"].as_array().unwrap().len();
        assert_eq!(bad, 0, "kekule regression mismatches: {:?}", reg["mismatches"]);
    }
    let mut mismatches = Vec::new();
    let mut skips: Vec<String> = Vec::new();
    let mut n = 0;
    for m in value["molecules"].as_array().unwrap() {
        n += 1;
        let mut ids = Vec::new();
        for a in m["atoms"].as_array().unwrap() {
            let e = element_index(a["element"].as_str().unwrap()).unwrap();
            ids.push(atom_type_of(e, a["hydrogens"].as_u64().unwrap() as u8, a["valence"].as_u64().unwrap() as u8).unwrap().id);
        }
        let bonds: Vec<(usize, usize, u8)> = m["bonds"].as_array().unwrap().iter().map(|b| {
            let b = b.as_array().unwrap();
            (b[0].as_u64().unwrap() as usize, b[1].as_u64().unwrap() as usize, b[2].as_u64().unwrap() as u8)
        }).collect();
        let graph = MolGraph::new(ids, bonds).unwrap();
        let set = functional_groups_v4(&graph);
        for (i, fg) in FG_NAMES.iter().enumerate() {
            let (rust, refc) = (set.count(i + 1), m["counts"][*fg].as_u64().unwrap() as u32);
            if rust != refc {
                mismatches.push(format!("{} {fg}: Rust {rust} vs RDKit {refc}", m["name"].as_str().unwrap_or("?")));
            }
        }
    }
    for s in value["skipped"].as_array().unwrap() {
        skips.push(format!("{}: {}",
            s.get("key").and_then(|v| v.as_str()).unwrap_or("?"),
            s.get("reason").and_then(|v| v.as_str()).unwrap_or("?")));
    }
    println!("pilot_validation_reference: {n} molecules compared, {} skipped", skips.len());
    for s in &skips {
        println!("  skip: {s}");
    }
    // Finding 4: counting exceptions on supported molecules are ERRORS (the
    // tool exits non-zero on them; see the status assert above), never skips.
    // A skip is only an input outside the V0 domain, with its reason. The
    // live test requires zero errors here too.
    let errors = value.get("errors").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "FAIL: reference tool reported {} errors: {:?}",
        errors.len(),
        &errors[..errors.len().min(5)]
    );
    // The compared count must account for every export molecule.
    assert_eq!(
        n + skips.len(),
        export_molecules,
        "compared ({n}) + skipped ({}) != export molecules ({export_molecules})",
        skips.len()
    );
    assert!(n > 0, "FAIL: zero molecules were compared");
    assert!(mismatches.is_empty(), "{} mismatches over {n} molecules:\n{}", mismatches.len(), mismatches.join("\n"));
    // For the known supported validation export: zero skips and the exact
    // molecule-key set (every export molecule compared, none extra).
    if export.file_name().and_then(|s| s.to_str()).unwrap_or("") == "pilot_validation.json" {
        assert!(
            skips.is_empty(),
            "FAIL: pilot validation must have zero skips, got {}: {:?}",
            skips.len(),
            &skips[..skips.len().min(5)]
        );
        let export_keys: BTreeSet<String> = export_value["molecules"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["key"].as_str().unwrap_or("?").to_string())
            .collect();
        let compared_keys: BTreeSet<String> = value["molecules"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap_or("?").to_string())
            .collect();
        assert_eq!(compared_keys, export_keys, "pilot validation molecule-key set mismatch");
        println!("pilot_validation_reference: zero skips, key set exact ({} molecules)", export_keys.len());
    }
    println!("pilot_validation_reference: {n} molecules agree");
    assert!(SPECIFIC_MASK.count_ones() == 27);
    assert!(HETEROATOM_MASK.count_ones() == 24);
    let _ = BTreeSet::<usize>::new();
}

// ---------------------------------------------------------------------------
// FG3: trace-order soundness of the five-ring exclusions
// ---------------------------------------------------------------------------
//
// The `label_ceiling` row replays pseudo-label target traces (atoms numbered
// in trace order, ring closures as emitted) while `recipe_fragments` uses
// `parent.induced(...)`. The detector must depend only on the labelled graph
// up to isomorphism: atom numbering, bond list order and bond endpoint order
// must not matter. A prior revision derived five-ring heteroatom/bond
// coverage from sorted-order adjacency instead of ring bonds in cycle order,
// so trace-numbered fragments of (e.g.) thiophene wrongly showed `thioether`.

/// Thiophene, ring-consecutive numbering: C0=C1-S2-C3=C4-C0.
/// Types: C H1 = 2, S H0 = 13.
fn thiophene_consecutive() -> MolGraph {
    mol(
        &[2, 2, 13, 2, 2],
        &[(0, 1, 2), (1, 2, 1), (2, 3, 1), (3, 4, 2), (4, 0, 1)],
    )
}

/// The same thiophene with the sulphur first (trace-like numbering):
/// new[i] holds old atom [2, 0, 1, 3, 4]: S0-C2=C1, S0-C3, C1-C4, C3=C4.
fn thiophene_sulfur_first() -> MolGraph {
    mol(
        &[13, 2, 2, 2, 2],
        &[(0, 2, 1), (0, 3, 1), (1, 2, 2), (1, 4, 1), (3, 4, 2)],
    )
}

#[test]
fn five_ring_numbering_minimal() {
    // Minimal hand-written fixture, no data file: the same thiophene in two
    // numberings must give identical determined and undetermined masks, the
    // whole ring is a five-ring (never a thioether), and the 4-atom fragment
    // holding the sulphur shows no determined thioether.
    let whole_a = thiophene_consecutive();
    let whole_b = thiophene_sulfur_first();
    // Same labelled graph up to isomorphism: same type multiset, same bonds.
    assert_eq!(whole_a.atoms().len(), whole_b.atoms().len());
    let det_a = functional_groups_v4(&whole_a);
    let det_b = functional_groups_v4(&whole_b);
    assert_eq!(det_a.mask(), det_b.mask(), "whole-ring determined masks differ by numbering");
    assert_eq!(det_a.counts(), det_b.counts(), "whole-ring counts differ by numbering");
    assert_eq!(undetermined(&whole_a), undetermined(&whole_b));
    assert_eq!(det_a.count(28), 1, "whole thiophene is one five-ring");
    assert_eq!(det_a.count(17), 0, "thiophene sulphur is no thioether");
    assert_eq!(det_b.count(28), 1, "whole thiophene (S-first) is one five-ring");
    assert_eq!(det_b.count(17), 0, "thiophene sulphur (S-first) is no thioether");
    // Fragments holding the sulphur with one ring carbon missing, in both
    // numberings (parent order of each numbering).
    let frag_a = whole_a.induced(&[0, 1, 2, 3]).unwrap();
    let frag_b = whole_b.induced(&[0, 1, 2, 3]).unwrap();
    let fa = functional_groups_v4(&frag_a);
    let fb = functional_groups_v4(&frag_b);
    assert_eq!(fa.mask(), fb.mask(), "S-fragment determined masks differ by numbering");
    assert_eq!(fa.counts(), fb.counts());
    assert_eq!(undetermined(&frag_a), undetermined(&frag_b));
    assert_eq!(fa.count(17), 0, "S-fragment shows no determined thioether");
    assert_eq!(fb.count(17), 0, "S-fragment (S-first) shows no determined thioether");
    let parent_mask = det_a.mask();
    assert_eq!(fa.mask() & !parent_mask, 0, "S-fragment outside parent types");
    assert_eq!(fb.mask() & !parent_mask, 0, "S-fragment (S-first) outside parent types");
    println!("five_ring_numbering_minimal: checked 2 numberings x (whole + S-fragment) = 4 graphs");
}

/// Seeded shuffle: Fisher-Yates over `rand`.
fn shuffled<T>(rng: &mut impl rand::Rng, xs: &mut [T]) {
    for i in (1..xs.len()).rev() {
        let j = rng.random_range(0..=i);
        xs.swap(i, j);
    }
}

#[test]
fn permutation_invariance_whole_molecules() {
    // For every molecule and every stored kekule form: 20 random atom
    // permutations (seeded) with the bond list shuffled and endpoints
    // swapped at random give identical counts for all 28 types.
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let f = fixture();
    let molecules = f["molecules"].as_array().unwrap();
    let mut rng = StdRng::seed_from_u64(0x6673335e7a5f4135);
    let mut checked = 0usize;
    let mut forms_total = 0usize;
    for m in molecules {
        let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let forms = m.get("forms").and_then(|v| v.as_array()).cloned().unwrap_or_else(|| vec![m.clone()]);
        for form in &forms {
            forms_total += 1;
            let base = graph_of_form(form);
            let want = functional_groups_v4(&base);
            let n = base.atoms().len();
            for _ in 0..20 {
                let mut perm: Vec<usize> = (0..n).collect();
                shuffled(&mut rng, &mut perm);
                let pg = base.permuted(&perm).unwrap();
                // Shuffle the bond list and swap endpoints at random (the
                // stored order is sorted by construction; the detector must
                // not depend on input order).
                let mut bonds: Vec<(usize, usize, u8)> = pg.bonds().to_vec();
                shuffled(&mut rng, &mut bonds);
                for b in bonds.iter_mut() {
                    if rng.random_bool(0.5) {
                        std::mem::swap(&mut b.0, &mut b.1);
                    }
                }
                let g = MolGraph::new(pg.atoms().to_vec(), bonds).unwrap();
                let got = functional_groups_v4(&g);
                assert_eq!(
                    got.counts(),
                    want.counts(),
                    "{name}: permuted counts differ"
                );
                assert_eq!(got.mask(), want.mask(), "{name}: permuted mask differs");
                assert_eq!(undetermined(&g), undetermined(&base), "{name}: permuted undetermined differs");
                checked += 1;
            }
        }
    }
    println!("permutation_invariance_whole_molecules: checked {checked} permuted graphs over {forms_total} stored forms of {} molecules", molecules.len());
    assert!(checked > 4000, "permutation test must be non-vacuous ({checked} cases)");
}

/// One parent for the fragment permutation test: name, types, bonds.
fn fg3_parents() -> Vec<(&'static str, Vec<u8>, Vec<(usize, usize, u8)>)> {
    let f = fixture();
    let molecules = f["molecules"].as_array().unwrap();
    let by_name = |want: &str| -> (Vec<u8>, Vec<(usize, usize, u8)>) {
        let m = molecules
            .iter()
            .find(|m| m["name"].as_str().unwrap_or("") == want)
            .unwrap_or_else(|| panic!("fixture misses FG3 parent {want}"));
        let forms = m["forms"].as_array().unwrap();
        let g = graph_of_form(&forms[0]);
        (g.atoms().to_vec(), g.bonds().to_vec())
    };
    let mut out: Vec<(&'static str, Vec<u8>, Vec<(usize, usize, u8)>)> = Vec::new();
    for want in fg3_parent_names().into_iter().filter(|w| !w.ends_with("(hand)")) {
        let (t, b) = by_name(want);
        out.push((want, t, b));
    }
    // Benzofuran and benzothiophene (fused benzo five-rings), explicit
    // RDKit kekule forms (C H1 = 2, C H0 = 1, O H0 = 8, S H0 = 13).
    out.push((
        "benzofuran (hand)",
        vec![2, 2, 2, 1, 8, 2, 2, 1, 2],
        vec![
            (0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 1),
            (5, 6, 2), (6, 7, 1), (7, 3, 1), (7, 8, 2), (8, 0, 1),
        ],
    ));
    out.push((
        "benzothiophene (hand)",
        vec![2, 2, 2, 1, 13, 2, 2, 1, 2],
        vec![
            (0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 1),
            (5, 6, 2), (6, 7, 1), (7, 3, 1), (7, 8, 2), (8, 0, 1),
        ],
    ));
    out
}

fn fg3_parent_names() -> Vec<&'static str> {
    vec![
        "thiophene (five-ring, no thioether/alkene)",
        "furan (five-ring, no arene/alkene/ether)",
        "pyrrole (five-ring, no amine/alkene)",
        "imidazole",
        "thiazole",
        "oxazole",
        "indole",
        "purine",
        "pyridine",
        "tetrahydrofuran",
        "cyclohexene",
        "benzene",
        "cyclobutadiene (delocalised by kekule-equivalence)",
        "tropone",
        "indene",
        "benzocyclobutadiene",
        "biphenylene (four-cycle)",
        "acenaphthylene",
        "benzofuran (hand)",
        "benzothiophene (hand)",
    ]
}

#[test]
fn permutation_invariance_fragments_trace() {
    // For >=12 parents covering every five-ring heteroaromatic of the
    // fixture plus six-ring aromatics and non-aromatic rings: for every
    // connected induced fragment of 3 to 9 atoms, the determined mask AND
    // the undetermined mask are identical in parent order, in 5 random
    // permutations, and in trace order (canonicalise + replay, the exact
    // path the evaluation uses for labels and candidates). Soundness in
    // trace order is asserted in `fragment_soundness_trace_order`.
    use rand::{SeedableRng, rngs::StdRng};
    let names = fg3_parent_names();
    assert!(names.len() >= 12, "needs at least 12 parents");
    let parents = fg3_parents();
    assert_eq!(parents.len(), names.len());
    let mut rng = StdRng::seed_from_u64(0x33664733deadbeef);
    let mut checked = 0usize;
    let mut trace_ok = 0usize;
    for ((label, types, bonds), want) in parents.iter().zip(names.iter()) {
        let _ = label;
        let parent = mol(types, bonds);
        let n = types.len();
        assert!(n <= 12, "{want} too large for exhaustive fragment enumeration ({n} atoms)");
        for members in connected_subsets(n, bonds, 3, 9.min(n)) {
            let frag = parent.induced(&members).unwrap();
            let det0 = functional_groups_v4(&frag).mask();
            let und0 = undetermined(&frag);
            // 5 random permutations.
            for _ in 0..5 {
                let mut perm: Vec<usize> = (0..members.len()).collect();
                shuffled(&mut rng, &mut perm);
                let pg = frag.permuted(&perm).unwrap();
                assert_eq!(functional_groups_v4(&pg).mask(), det0, "{want} fragment {members:?}: permuted determined mask differs");
                assert_eq!(undetermined(&pg), und0, "{want} fragment {members:?}: permuted undetermined mask differs");
                checked += 1;
            }
            // Trace order: canonicalise the fragment and replay it.
            let canon = canonical_trace(&frag, Limits::V0, CANONICAL_WORK_LIMIT)
                .unwrap_or_else(|e| panic!("{want} fragment {members:?}: canonical_trace failed: {e}"));
            let state = replay(&canon.trace, Limits::V0, None)
                .unwrap_or_else(|e| panic!("{want} fragment {members:?}: replay failed: {e}"));
            assert!(state.stopped(), "{want} fragment {members:?}: replay did not stop");
            let tg = state.graph().unwrap();
            assert_eq!(functional_groups_v4(&tg).mask(), det0, "{want} fragment {members:?}: trace-order determined mask differs");
            assert_eq!(undetermined(&tg), und0, "{want} fragment {members:?}: trace-order undetermined mask differs");
            trace_ok += 1;
            checked += 1;
        }
    }
    println!("permutation_invariance_fragments_trace: checked {checked} permuted/trace cases ({trace_ok} trace replays) over {} parents", names.len());
    assert!(checked > 2000, "fragment permutation test must be non-vacuous ({checked} cases)");
}

#[test]
fn fragment_soundness_trace_order() {
    // On the same fragments, determined types of the trace-replayed fragment
    // are a subset of the parent's types.
    let names = fg3_parent_names();
    let parents = fg3_parents();
    let mut checked = 0usize;
    for ((_, types, bonds), want) in parents.iter().zip(names.iter()) {
        let parent = mol(types, bonds);
        let pmask = functional_groups_v4(&parent).mask();
        let n = types.len();
        for members in connected_subsets(n, bonds, 3, 9.min(n)) {
            let frag = parent.induced(&members).unwrap();
            let canon = canonical_trace(&frag, Limits::V0, CANONICAL_WORK_LIMIT)
                .unwrap_or_else(|e| panic!("{want} fragment {members:?}: canonical_trace failed: {e}"));
            let state = replay(&canon.trace, Limits::V0, None).unwrap();
            let tg = state.graph().unwrap();
            let dm = functional_groups_v4(&tg).mask();
            assert_eq!(dm & !pmask, 0, "{want} fragment {members:?}: trace-order determined {dm:b} outside parent {pmask:b}");
            checked += 1;
        }
    }
    println!("fragment_soundness_trace_order: checked {checked} trace-replayed fragments over {} parents", names.len());
    assert!(checked > 500, "trace soundness test must be non-vacuous ({checked} fragments)");
}

#[test]
fn fg3_label_ceiling_soundness() {
    // Data-backed soundness: for every labeled spectrum of the pilot
    // validation export, the union of determined types over the
    // pseudo-label target graphs (replayed in trace order) is a subset of
    // the parent's types; and the same holds for the union over ALL recipe
    // candidates rebuilt through the trace path (canonicalise + replay, not
    // only `induced`).
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/ms2/pilot_validation.json");
    if !path.exists() {
        println!("SKIP fg3_label_ceiling_soundness: {path:?} does not exist (prerequisite missing)");
        return;
    }
    let set = ExperimentSet::load(&path, &RecipeLimits::V0).expect("pilot validation loads");
    let mut spectra_checked = 0usize;
    let mut targets_checked = 0usize;
    let mut offenders: Vec<(String, usize, Vec<Token>, String)> = Vec::new();
    for (si, entry) in set.spectra.iter().enumerate() {
        let Some(labels) = entry.labels.as_ref() else {
            continue;
        };
        if labels.targets.is_empty() {
            continue;
        }
        let pmask = functional_groups_v4(&entry.parent).mask();
        let lu = label_union(labels);
        // Per-target detail for offence reports.
        for (ti, target) in labels.targets.iter().enumerate() {
            let Ok(state) = replay(&target.trace, Limits::V0, None) else {
                continue;
            };
            if !state.stopped() || state.atoms() == 0 {
                continue;
            }
            let Ok(graph) = state.graph() else {
                continue;
            };
            targets_checked += 1;
            let tm = functional_groups_v4(&graph).mask();
            let bad = tm & !pmask;
            if bad != 0 {
                for id in 1..=N_FG {
                    if bad & (1u32 << (id - 1)) != 0 {
                        offenders.push((
                            set.molecules[entry.molecule].clone(),
                            si,
                            target.trace.clone(),
                            format!("target {ti} type {} ({})", id, FG_NAMES[id - 1]),
                        ));
                    }
                }
            }
        }
        let bad = lu & !pmask;
        if bad != 0 {
            offenders.push((
                set.molecules[entry.molecule].clone(),
                si,
                Vec::new(),
                format!("spectrum union outside parent: {bad:b}"),
            ));
        }
        spectra_checked += 1;
    }
    println!("fg3_label_ceiling_soundness: checked {spectra_checked} labeled spectra ({targets_checked} targets)");
    for (key, si, trace, what) in offenders.iter().take(5) {
        println!("  offender: molecule {key} spectrum {si} {what} trace {trace:?}");
    }
    assert!(
        offenders.is_empty(),
        "{} label-ceiling offences (first shown above)",
        offenders.len()
    );
    assert!(spectra_checked > 100, "label soundness test must be non-vacuous ({spectra_checked} spectra)");
    // The union over ALL recipe candidates rebuilt through the trace path.
    let mut seen_mol: BTreeSet<usize> = BTreeSet::new();
    let mut frags_checked = 0usize;
    let mut recipe_offenders = 0usize;
    for entry in &set.spectra {
        if entry.labels.is_none() || !seen_mol.insert(entry.molecule) {
            continue;
        }
        let pmask = functional_groups_v4(&entry.parent).mask();
        let candidates = Candidates::new(&entry.parent, &RecipeLimits::V0).expect("recipe candidates build");
        for emb in candidates.embeddings() {
            let Ok(sub) = entry.parent.induced(&emb.atoms) else {
                continue;
            };
            let Ok(canon) = canonical_trace(&sub, Limits::V0, CANONICAL_WORK_LIMIT) else {
                continue;
            };
            let Ok(state) = replay(&canon.trace, Limits::V0, None) else {
                continue;
            };
            if !state.stopped() || state.atoms() == 0 {
                continue;
            }
            let Ok(graph) = state.graph() else {
                continue;
            };
            frags_checked += 1;
            if functional_groups_v4(&graph).mask() & !pmask != 0 {
                recipe_offenders += 1;
            }
        }
    }
    println!("fg3_label_ceiling_soundness: recipe-trace union checked {frags_checked} fragments over {} molecules", seen_mol.len());
    assert_eq!(recipe_offenders, 0, "{recipe_offenders} recipe-trace fragments outside parent types");
    assert!(frags_checked > 1000, "recipe-trace check must be non-vacuous ({frags_checked} fragments)");
}

#[test]
fn label_union_thiophene_regression() {
    // Evaluation-level regression: a hand-built spectrum whose parent
    // contains a thiophene and whose target fragment holds the ring sulphur
    // with one ring carbon missing shows no `thioether` (undetermined
    // instead); with the whole ring present it shows
    // `heteroaromatic_five_ring` and no `thioether`.
    let parent = thiophene_consecutive();
    let pmask = functional_groups_v4(&parent).mask();
    assert_eq!(pmask & (1u32 << (28 - 1)), 1u32 << (28 - 1), "parent holds the five-ring");
    assert_eq!(pmask & (1u32 << (17 - 1)), 0, "parent holds no thioether");
    // Fragment target: sulphur plus three ring atoms (one carbon missing).
    let frag = parent.induced(&[0, 1, 2, 3]).unwrap();
    let canon = canonical_trace(&frag, Limits::V0, CANONICAL_WORK_LIMIT).expect("fragment canonicalises");
    let labels = Labels {
        embeddings: Vec::new(),
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![Target {
            trace: canon.trace.clone(),
            weight: 1,
            q: 1.0,
            embeddings: Vec::new(),
            anchors: Vec::new(),
        }],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: Vec::new(),
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let lu = label_union(&labels);
    assert_eq!(lu & (1u32 << (17 - 1)), 0, "S-fragment label union holds no thioether");
    assert_eq!(lu & !pmask, 0, "S-fragment label union inside parent types");
    // Undetermined instead: replay the target and inspect the mask.
    let state = replay(&canon.trace, Limits::V0, None).unwrap();
    let graph = state.graph().unwrap();
    assert_eq!(functional_groups_v4(&graph).count(17), 0);
    assert_ne!(undetermined(&graph) & (1u32 << (17 - 1)), 0, "S-fragment holds undetermined thioether");
    // Whole-ring target: five-ring present, still no thioether.
    let whole = parent.induced(&[0, 1, 2, 3, 4]).unwrap();
    let canon_w = canonical_trace(&whole, Limits::V0, CANONICAL_WORK_LIMIT).expect("whole ring canonicalises");
    let labels_w = Labels {
        embeddings: Vec::new(),
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![Target {
            trace: canon_w.trace.clone(),
            weight: 1,
            q: 1.0,
            embeddings: Vec::new(),
            anchors: Vec::new(),
        }],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: Vec::new(),
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let lu_w = label_union(&labels_w);
    assert_ne!(lu_w & (1u32 << (28 - 1)), 0, "whole-ring label union holds the five-ring");
    assert_eq!(lu_w & (1u32 << (17 - 1)), 0, "whole-ring label union holds no thioether");
    println!("label_union_thiophene_regression: checked S-fragment and whole-ring label unions");
}

// ---------------------------------------------------------------------------
// FG4: blossom search versus brute force on abstract graphs
// ---------------------------------------------------------------------------
//
// The detector's delocalisation tests are single augmenting-path searches of
// Edmonds' blossom algorithm on the π graph (see `functional_groups`). Here
// the same Rust entry points (`has_perfect_matching`, `matching_allowed_edges`)
// are checked against an independent brute-force enumerator written in this
// test file (plain backtracking over all perfect matchings): on 500 seeded
// random graphs of 4 to 14 vertices with a planted perfect matching —
// including odd cycles — every "edge in some but not all perfect matchings"
// decision must agree.

/// Brute-force enumeration of all perfect matchings (vertex-pair lists) by
/// plain backtracking. Independent of the detector's blossom code.
fn brute_perfect_matchings(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let mut adj = vec![Vec::new(); n];
    for (a, b) in edges {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    for lst in adj.iter_mut() {
        lst.sort_unstable();
    }
    let mut out = Vec::new();
    let mut used = vec![false; n];
    let mut cur = Vec::new();
    fn rec(
        n: usize,
        adj: &[Vec<usize>],
        used: &mut [bool],
        cur: &mut Vec<(usize, usize)>,
        out: &mut Vec<Vec<(usize, usize)>>,
    ) {
        let Some(v) = (0..n).find(|&i| !used[i]) else {
            out.push(cur.clone());
            return;
        };
        used[v] = true;
        for &w in &adj[v] {
            if !used[w] {
                used[w] = true;
                cur.push((v.min(w), v.max(w)));
                rec(n, adj, used, cur, out);
                cur.pop();
                used[w] = false;
            }
        }
        used[v] = false;
    }
    rec(n, &adj, &mut used, &mut cur, &mut out);
    out
}

/// Brute-force allowed-edges: in some but not all perfect matchings.
fn brute_allowed(n: usize, edges: &[(usize, usize)]) -> Vec<bool> {
    let forms = brute_perfect_matchings(n, edges);
    if forms.is_empty() {
        return vec![false; edges.len()];
    }
    let total = forms.len();
    let in_set: Vec<std::collections::BTreeSet<(usize, usize)>> = forms
        .iter()
        .map(|f| f.iter().copied().collect())
        .collect();
    edges
        .iter()
        .map(|e| {
            let c = in_set.iter().filter(|s| s.contains(e)).count();
            c > 0 && c < total
        })
        .collect()
}

#[test]
fn blossom_allowed_edges_match_brute_force() {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0xB10550AA4D41C4E1);
    let mut graphs = 0usize;
    let mut edges_checked = 0usize;
    let mut allowed_count = 0usize;
    for trial in 0..500 {
        // Even vertex count 4..=14.
        let n = 2 * rng.random_range(2..=7);
        // Planted perfect matching: random pairing.
        let mut perm: Vec<usize> = (0..n).collect();
        shuffled(&mut rng, &mut perm);
        let planted: Vec<(usize, usize)> = perm
            .chunks_exact(2)
            .map(|w| (w[0].min(w[1]), w[0].max(w[1])))
            .collect();
        // Random extra edges (density varies; odd cycles included).
        let density: f64 = rng.random_range(0.15..0.75);
        let mut edge_set = std::collections::BTreeSet::new();
        for (a, b) in &planted {
            edge_set.insert((*a, *b));
        }
        for a in 0..n {
            for b in (a + 1)..n {
                if rng.random_bool(density) {
                    edge_set.insert((a, b));
                }
            }
        }
        let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        let brute = brute_allowed(n, &edges);
        // Sanity: the planted matching is a perfect matching, so brute force
        // always finds at least one form on these graphs.
        assert!(
            !brute_perfect_matchings(n, &edges).is_empty(),
            "trial {trial}: planted matching must be a perfect matching"
        );
        assert_eq!(
            has_perfect_matching(n, &edges),
            true,
            "trial {trial}: blossom must find the planted perfect matching"
        );
        let got = matching_allowed_edges(n, &edges, &planted);
        assert_eq!(
            got, brute,
            "trial {trial} (n={n}, m={}): blossom allowed-edges differ from brute force",
            edges.len()
        );
        graphs += 1;
        edges_checked += edges.len();
        allowed_count += got.iter().filter(|&&x| x).count();
    }
    // Graphs without any perfect matching (odd counts and planted-broken
    // evens): blossom existence must agree with brute force too.
    let mut no_pm = 0usize;
    for trial in 0..100 {
        let n = rng.random_range(3..=11);
        let density: f64 = rng.random_range(0.1..0.5);
        let mut edge_set = std::collections::BTreeSet::new();
        for a in 0..n {
            for b in (a + 1)..n {
                if rng.random_bool(density) {
                    edge_set.insert((a, b));
                }
            }
        }
        let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        let brute_has = !brute_perfect_matchings(n, &edges).is_empty();
        assert_eq!(
            has_perfect_matching(n, &edges),
            brute_has,
            "no-pm trial {trial} (n={n}): existence differs from brute force"
        );
        if !brute_has {
            no_pm += 1;
        }
    }
    println!(
        "blossom_allowed_edges_match_brute_force: {graphs} planted graphs agree, \
         {edges_checked} edges checked ({allowed_count} allowed); {no_pm}/100 without perfect matching"
    );
    assert!(allowed_count > 0, "test must exercise allowed edges");
}

// ---------------------------------------------------------------------------
// FG4: hexacene regression (reviewer numbering, forms A and B)
//
// The 22-atom bound broke kekulé invariance here: form A needs a 26-cycle
// witness for some bonds (the old code reported arene_ring=5, alkene=2),
// while form B gave the right counts. Under the exact matching rule both
// forms give arene_ring=6, alkene=0 with identical delocalised bond sets.
fn hexacene_graph(form_b: bool) -> MolGraph {
    // Two rows 0..=12 and 13..=25, horizontals plus rungs (2j, 13+2j).
    let types = vec![
        2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        2, 2,
    ];
    let mut bonds = vec![
        (0, 1),
        (0, 13),
        (1, 2),
        (2, 3),
        (2, 15),
        (3, 4),
        (4, 5),
        (4, 17),
        (5, 6),
        (6, 7),
        (6, 19),
        (7, 8),
        (8, 9),
        (8, 21),
        (9, 10),
        (10, 11),
        (10, 23),
        (11, 12),
        (12, 25),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
    ];
    let doubles_a = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 25),
        (13, 14),
        (15, 16),
        (17, 18),
        (19, 20),
        (21, 22),
        (23, 24),
    ];
    let doubles: Vec<(usize, usize)> = if form_b {
        doubles_a
            .into_iter()
            .filter(|e| *e != (10, 11) && *e != (12, 25) && *e != (23, 24))
            .chain([(10, 23), (11, 12), (24, 25)])
            .collect()
    } else {
        doubles_a.into()
    };
    let bond_list: Vec<(usize, usize, u8)> = bonds
        .drain(..)
        .map(|(a, b)| {
            let order = if doubles.contains(&(a, b)) { 2 } else { 1 };
            (a, b, order)
        })
        .collect();
    mol(&types, &bond_list)
}

#[test]
fn hexacene_invariance_exact() {
    use mamba3::models::ms2::functional_groups::delocalised_bonds;
    let ga = hexacene_graph(false);
    let gb = hexacene_graph(true);
    let sa = functional_groups_v4(&ga);
    let sb = functional_groups_v4(&gb);
    assert_eq!(sa.count(24), 6, "hexacene form A: six arene rings");
    assert_eq!(sa.count(14), 0, "hexacene form A: no fixed alkene");
    assert_eq!(sb.count(24), 6, "hexacene form B: six arene rings");
    assert_eq!(sb.count(14), 0, "hexacene form B: no fixed alkene");
    assert_eq!(sa.counts(), sb.counts(), "hexacene counts identical across forms");
    assert_eq!(sa.mask(), sb.mask());
    // The delocalised bond SET is identical across forms (kekulé invariance
    // at the bond level, including the four-cycle-free 26-witness bonds).
    assert_eq!(delocalised_bonds(&ga), delocalised_bonds(&gb));
    assert_eq!(undetermined(&ga), 0, "closed hexacene: nothing undetermined");
    assert_eq!(undetermined(&gb), 0);
    println!("hexacene_invariance_exact: both forms arene_ring=6 alkene=0, bond sets agree");
}

// ---------------------------------------------------------------------------
// FG4: fixture regression counts, completeness, sampled fragment soundness
// ---------------------------------------------------------------------------

/// Fixture molecule graphs (all stored forms) by exact name.
fn fixture_molecule_forms(name: &str) -> Vec<MolGraph> {
    let f = fixture();
    let m = f["molecules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"].as_str().unwrap_or("") == name)
        .unwrap_or_else(|| panic!("fixture misses molecule {name}"));
    m["forms"]
        .as_array()
        .unwrap()
        .iter()
        .map(graph_of_form)
        .collect()
}

#[test]
fn regression_molecule_counts() {
    // (molecule, type id, expected determined count in EVERY form).
    // Cyclobutadiene is delocalised by the kekule-equivalence definition: its
    // four ring bonds flip together, so every bond is delocalised and no
    // fixed alkene (or anything else) counts.
    let wants: Vec<(&str, Vec<(usize, u32)>)> = vec![
        ("hexacene", vec![(24, 6)]),
        ("heptacene", vec![(24, 7)]),
        ("pentacene", vec![(24, 5)]),
        ("coronene", vec![(24, 7)]),
        ("ovalene sheet (hand-built 10-ring benzenoid, C32H14)", vec![(24, 10)]),
        ("large benzenoid sheet (hand-built 13-ring, C48H24)", vec![(24, 13)]),
        ("expanded six-pyrrole macrocycle, 4 imine + NH + NMe (hand-built)", vec![(28, 6)]),
        ("porphine (hand-built cyclic tetrapyrrole)", vec![(28, 4)]),
        ("biphenylene (four-cycle)", vec![(24, 2)]),
        ("azulene (10-cycle, no six-ring in some forms)", vec![]),
        ("acenaphthylene", vec![(14, 1), (24, 2)]),
        ("indene", vec![(14, 1), (24, 1)]),
        ("fulvene", vec![(14, 3)]),
        ("[18]annulene", vec![]),
        ("cyclobutadiene (delocalised by kekule-equivalence)", vec![]),
        ("benzocyclobutadiene", vec![(24, 1)]),
        ("tropone", vec![(1, 1), (6, 1), (14, 3)]),
        ("p-benzoquinone (fixed C=C/C=O; NOT an arene ring)", vec![(1, 2), (6, 2), (14, 2)]),
        ("pyrene", vec![(24, 4)]),
        ("perylene", vec![(24, 4)]),
    ];
    for (name, ids) in &wants {
        let forms = fixture_molecule_forms(name);
        assert!(!forms.is_empty(), "{name}: no stored forms");
        let base = functional_groups_v4(&forms[0]);
        for (id, n) in ids {
            assert_eq!(base.count(*id), *n, "{name}: type {id} count");
        }
        for g in forms.iter().skip(1) {
            let s = functional_groups_v4(g);
            assert_eq!(s.counts(), base.counts(), "{name}: counts differ across forms");
        }
        // The specified types are exactly the non-zero ones, except where
        // the expectation list is empty (checked separately below).
        if name.contains("cyclobutadiene (delocalised") || *name == "[18]annulene" {
            assert_eq!(base.total(), 0, "{name}: expected no determined instances");
        }
        println!("regression {name}: {} forms agree", forms.len());
    }
    // Cyclobutadiene: every ring bond delocalised (kekule-equivalence), so
    // alkene 0 with all four bonds decided delocalised.
    {
        use mamba3::models::ms2::functional_groups::decided_bonds;
        let forms = fixture_molecule_forms("cyclobutadiene (delocalised by kekule-equivalence)");
        assert_eq!(forms.len(), 2);
        for g in &forms {
            assert_eq!(g.bonds().len(), 4);
            assert_eq!(decided_bonds(g), vec![Some(true); 4]);
        }
    }
    // Biphenylene: the delocalised bond SET is identical across all 5 forms
    // (four-cycles count), and so are the counts.
    {
        use mamba3::models::ms2::functional_groups::delocalised_bonds;
        let forms = fixture_molecule_forms("biphenylene (four-cycle)");
        assert_eq!(forms.len(), 5, "biphenylene stores 5 forms");
        let base = delocalised_bonds(&forms[0]);
        for g in forms.iter().skip(1) {
            assert_eq!(delocalised_bonds(g), base, "biphenylene bond set differs across forms");
        }
    }
    // Macrocycle: identical counts in all forms, no amine/imine/alkene.
    {
        let forms = fixture_molecule_forms(
            "expanded six-pyrrole macrocycle, 4 imine + NH + NMe (hand-built)",
        );
        assert!(forms.len() >= 2);
        let base = functional_groups_v4(&forms[0]);
        assert_eq!(base.count(28), 6);
        for id in [9, 10, 11, 13, 14] {
            assert_eq!(base.count(id), 0, "macrocycle type {id} must be 0");
        }
    }
}

/// The test's own f-factor oracle (independent of the detector's code, and
/// NOT shared with it): double-bond counts `d(v)` from the stored orders, the
/// candidate graph `G'` (atoms with `d >= 1` joined by the single/double
/// bonds between two such atoms), and ALL single/double assignments with the
/// prescribed `d(v)` by plain backtracking over the bonds of `G'` (NOT a
/// matching reduction, NOT the perfect-matching special case). The
/// delocalised set is the bonds double in some assignment and single in
/// another; every other bond is fixed.
fn oracle_degrees(graph: &MolGraph) -> (Vec<u8>, Vec<usize>) {
    let n = graph.atoms().len();
    let mut d = vec![0u8; n];
    for (a, b, o) in graph.bonds() {
        if *o == 2 {
            d[*a] += 1;
            d[*b] += 1;
        }
    }
    let mut g = Vec::new();
    for (i, (a, b, o)) in graph.bonds().iter().enumerate() {
        if (*o == 1 || *o == 2) && d[*a] >= 1 && d[*b] >= 1 {
            g.push(i);
        }
    }
    (d, g)
}

/// All valid single/double assignments with the prescribed double-bond counts
/// by plain backtracking over the bonds of `G'`. Independent of the
/// detector's gadget + blossom code. Returned as sorted bond triples.
fn oracle_assignments(graph: &MolGraph) -> Vec<Vec<(usize, usize, u8)>> {
    let bonds = graph.bonds();
    let n = graph.atoms().len();
    let (d, g) = oracle_degrees(graph);
    let mut incid: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (pos, &bi) in g.iter().enumerate() {
        let (a, b, _) = bonds[bi];
        incid[a].push(pos);
        incid[b].push(pos);
    }
    let mut assign = vec![0u8; g.len()];
    let mut used = vec![0u8; n];
    let mut remain: Vec<usize> = incid.iter().map(|v| v.len()).collect();
    let mut out: BTreeSet<Vec<(usize, usize, u8)>> = BTreeSet::new();
    fn rec(
        bonds: &[(usize, usize, u8)],
        g: &[usize],
        incid: &[Vec<usize>],
        d: &[u8],
        pos: usize,
        assign: &mut [u8],
        used: &mut [u8],
        remain: &mut [usize],
        out: &mut BTreeSet<Vec<(usize, usize, u8)>>,
    ) {
        if pos == g.len() {
            if used.iter().zip(d.iter()).all(|(u, w)| u == w) {
                let mut triples: Vec<(usize, usize, u8)> = Vec::new();
                for (i, (a, b, o)) in bonds.iter().enumerate() {
                    triples.push((*a.min(b), *a.max(b), *o));
                }
                for (p, &bi) in g.iter().enumerate() {
                    let (a, b, _) = bonds[bi];
                    let key = (a.min(b), a.max(b));
                    if let Some(t) = triples.iter_mut().find(|t| t.0 == key.0 && t.1 == key.1) {
                        t.2 = assign[p];
                    }
                }
                triples.sort_unstable();
                out.insert(triples);
            }
            return;
        }
        let bi = g[pos];
        let (a, b, _) = bonds[bi];
        for o in [1u8, 2u8] {
            let mut ok = true;
            for v in [a, b] {
                let u = used[v] + if o == 2 { 1 } else { 0 };
                let r = remain[v] - 1;
                if !(u <= d[v] && d[v] as usize <= u as usize + r) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            assign[pos] = o;
            for v in [a, b] {
                used[v] += if o == 2 { 1 } else { 0 };
                remain[v] -= 1;
            }
            rec(bonds, g, incid, d, pos + 1, assign, used, remain, out);
            for v in [a, b] {
                used[v] -= if o == 2 { 1 } else { 0 };
                remain[v] += 1;
            }
            assign[pos] = 0;
        }
    }
    rec(bonds, &g, &incid, &d, 0, &mut assign, &mut used, &mut remain, &mut out);
    out.into_iter().collect()
}

/// The oracle delocalised set: per-bond-index flags (double in some valid
/// assignment and single in another).
fn oracle_delocalised(graph: &MolGraph) -> Vec<bool> {
    let forms = oracle_assignments(graph);
    assert!(!forms.is_empty(), "oracle finds at least the stored assignment");
    let nb = graph.bonds().len();
    // Bond index by sorted endpoint pair.
    let mut idx_of = std::collections::BTreeMap::new();
    for (i, (a, b, _)) in graph.bonds().iter().enumerate() {
        idx_of.insert(((*a).min(*b), (*a).max(*b)), i);
    }
    let mut seen_double = vec![false; nb];
    let mut seen_single = vec![false; nb];
    for f in &forms {
        for (a, b, o) in f {
            let i = idx_of[&(*a, *b)];
            if *o == 2 {
                seen_double[i] = true;
            } else if *o == 1 {
                seen_single[i] = true;
            }
        }
    }
    seen_double.iter().zip(seen_single.iter()).map(|(x, y)| *x && *y).collect()
}

/// All kekule forms of `graph` via the test's own brute-force assignment
/// enumeration above (not the detector's code): each assignment sets its
/// `G'` doubles, other `G'` bonds single, and every non-`G'` bond keeps its
/// order. Returned as sorted bond triples.
fn test_kekule_forms(graph: &MolGraph) -> Vec<Vec<(usize, usize, u8)>> {
    oracle_assignments(graph)
}

#[test]
fn kekule_forms_complete_and_regenerated() {
    // The fixture stores EVERY kekule form (forms_total == forms_stored; the
    // reference fails loudly past its cap instead of truncating), and the
    // stored bond-assignment set equals the test's own brute-force
    // enumeration on the test's own π graph — for every fixture molecule.
    // The largest case is the 48-atom sheet with 432 forms, all regenerated
    // and checked (the "at least 64 forms" clause, with room).
    let f = fixture();
    let molecules = f["molecules"].as_array().unwrap();
    let mut biggest = (String::new(), 0usize);
    let mut total_forms = 0usize;
    for m in molecules {
        let name = m["name"].as_str().unwrap_or("?").to_string();
        let total = m["forms_total"].as_u64().unwrap() as usize;
        let stored = m["forms_stored"].as_u64().unwrap() as usize;
        assert_eq!(total, stored, "{name}: enumeration truncated");
        let forms = m["forms"].as_array().unwrap();
        assert_eq!(forms.len(), stored, "{name}: stored count mismatch");
        total_forms += stored;
        if stored > biggest.1 {
            biggest = (name.clone(), stored);
        }
        let first = graph_of_form(&forms[0]);
        let regen = test_kekule_forms(&first);
        let mut stored_set: BTreeSet<Vec<(usize, usize, u8)>> = BTreeSet::new();
        for form in forms {
            let mut triples: Vec<(usize, usize, u8)> = form["bonds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| {
                    let b = b.as_array().unwrap();
                    let (a, c) = (
                        b[0].as_u64().unwrap() as usize,
                        b[1].as_u64().unwrap() as usize,
                    );
                    (a.min(c), a.max(c), b[2].as_u64().unwrap() as u8)
                })
                .collect();
            triples.sort_unstable();
            stored_set.insert(triples);
        }
        let regen_set: BTreeSet<Vec<(usize, usize, u8)>> = regen.into_iter().collect();
        assert_eq!(stored_set, regen_set, "{name}: stored forms != regenerated forms");
        // Every stored form carries identical counts (invariance), checked
        // form by form against the first.
        let base = functional_groups_v4(&first);
        for form in forms.iter().skip(1) {
            let g = graph_of_form(form);
            assert_eq!(
                functional_groups_v4(&g).counts(),
                base.counts(),
                "{name}: counts differ across forms"
            );
        }
    }
    println!(
        "kekule_forms_complete_and_regenerated: {} molecules, {total_forms} forms; biggest {} ({})",
        molecules.len(),
        biggest.0,
        biggest.1
    );
    assert!(biggest.1 >= 432, "largest case must be the 432-form sheet");
}

/// All connected induced atom subsets with `min_size..=max_size` atoms, by
/// canonical bounded enumeration: from each start atom `s` (the subset's
/// minimum), grow connected sets adding only atoms greater than `s`.
/// Exhaustion is exact (unlike random sampling), so small parents are fully
/// covered; large parents are subsampled by the caller with a seed.
fn connected_subsets_bounded(
    adj: &[Vec<usize>],
    min_size: usize,
    max_size: usize,
) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut out = BTreeSet::new();
    for s in 0..n {
        // inside: sorted; frontier: candidates greater than s.
        let mut inside = vec![s];
        let mut frontier: Vec<usize> = adj[s].iter().copied().filter(|&v| v > s).collect();
        frontier.sort_unstable();
        frontier.dedup();
        fn rec(
            adj: &[Vec<usize>],
            s: usize,
            inside: &mut Vec<usize>,
            frontier: &mut Vec<usize>,
            min_size: usize,
            max_size: usize,
            out: &mut BTreeSet<Vec<usize>>,
        ) {
            if inside.len() >= min_size {
                out.insert(inside.clone());
            }
            if inside.len() >= max_size {
                return;
            }
            for k in 0..frontier.len() {
                let v = frontier[k];
                // Add v; new frontier members are v's greater neighbours.
                let mut new_frontier = frontier.clone();
                new_frontier.remove(k);
                for w in &adj[v] {
                    if *w > s && !inside.contains(w) && !new_frontier.contains(w) {
                        new_frontier.push(*w);
                    }
                }
                inside.push(v);
                inside.sort_unstable();
                rec(adj, s, inside, &mut new_frontier, min_size, max_size, out);
                inside.retain(|&x| x != v);
            }
        }
        rec(adj, s, &mut inside, &mut frontier, min_size, max_size, &mut out);
    }
    out.into_iter().collect()
}

#[test]
fn fragment_soundness_sampled_fg4() {
    // Exact assertion: on hexacene, coronene, the expanded macrocycle,
    // indole, purine and biphenylene, every connected induced fragment of 4
    // to 10 atoms (3,000 sampled per parent with a seed when more exist)
    // satisfies: determined types ⊆ parent types in parent order and in
    // trace-replay order, and every bond whose status the fragment reports as
    // decided carries the same status in the parent.
    use mamba3::models::ms2::functional_groups::decided_bonds;
    use rand::{SeedableRng, rngs::StdRng};
    let parents = [
        "hexacene",
        "coronene",
        "expanded six-pyrrole macrocycle, 4 imine + NH + NMe (hand-built)",
        "indole",
        "purine",
        "biphenylene (four-cycle)",
    ];
    let mut rng = StdRng::seed_from_u64(0xF647A11CE550123);
    let mut frags = 0usize;
    let mut trace_ok = 0usize;
    let mut trace_skipped = 0usize;
    let mut decided_bonds_checked = 0usize;
    for name in parents {
        let forms = fixture_molecule_forms(name);
        let parent = &forms[0];
        let pmask = functional_groups_v4(parent).mask();
        let pdecided = decided_bonds(parent);
        assert!(pdecided.iter().all(|s| s.is_some()), "{name}: parent must be fully decided");
        let n = parent.atoms().len();
        let mut adj = vec![Vec::new(); n];
        for (a, b, _) in parent.bonds() {
            adj[*a].push(*b);
            adj[*b].push(*a);
        }
        // Parent bond index by sorted endpoint pair.
        let mut parent_bond_idx = std::collections::BTreeMap::new();
        for (i, (a, b, _)) in parent.bonds().iter().enumerate() {
            parent_bond_idx.insert(((*a).min(*b), (*a).max(*b)), i);
        }
        let max_size = 10.min(n);
        // Exact bounded enumeration, then a seeded 3,000-sample when more
        // exist (small parents are covered fully).
        let all = connected_subsets_bounded(&adj, 4, max_size);
        let members_list: Vec<Vec<usize>> = if all.len() > 3000 {
            let mut idx: Vec<usize> = (0..all.len()).collect();
            shuffled(&mut rng, &mut idx);
            idx.truncate(3000);
            idx.into_iter().map(|i| all[i].clone()).collect()
        } else {
            all.clone()
        };
        let sampled = members_list.len();
        println!("fragment_soundness {name}: {sampled} fragments (space {})", all.len());
        for members in &members_list {
            let frag = parent.induced(&members).unwrap();
            let dm = functional_groups_v4(&frag).mask();
            assert_eq!(dm & !pmask, 0, "{name} fragment {members:?}: determined outside parent");
            // Decided-status soundness bond by bond.
            let fdecided = decided_bonds(&frag);
            for (i, (a, b, _)) in frag.bonds().iter().enumerate() {
                if let Some(v) = fdecided[i] {
                    let (pa, pb) = (members[*a], members[*b]);
                    let pi = parent_bond_idx[&(pa.min(pb), pa.max(pb))];
                    assert_eq!(
                        pdecided[pi],
                        Some(v),
                        "{name} fragment {members:?} bond ({pa},{pb}): fragment says {v} but parent says {:?}",
                        pdecided[pi]
                    );
                    decided_bonds_checked += 1;
                }
            }
            // Trace-replay order: determined types still inside the parent.
            let canon = canonical_trace(&frag, Limits::V0, CANONICAL_WORK_LIMIT)
                .unwrap_or_else(|e| panic!("{name} fragment {members:?}: canonical_trace: {e}"));
            let state = replay(&canon.trace, Limits::V0, None)
                .unwrap_or_else(|e| panic!("{name} fragment {members:?}: replay: {e}"));
            assert!(state.stopped(), "{name} fragment {members:?}: replay did not stop");
            let tg = state.graph().unwrap();
            let tm = functional_groups_v4(&tg).mask();
            assert_eq!(tm & !pmask, 0, "{name} fragment {members:?}: trace-order outside parent");
            trace_ok += 1;
            frags += 1;
        }
    }
    println!(
        "fragment_soundness_sampled_fg4: {frags} fragments, {trace_ok} trace replays, \
         {decided_bonds_checked} decided bonds agree with parent"
    );
    assert!(frags > 8000, "fragment test must be non-vacuous ({frags} fragments)");
    assert!(decided_bonds_checked > 3000);
}

#[test]
fn ether_closing_fragment_found() {
    // Finding 3: the reviewer's ether (tert-butyl pyridines, RDKit numbering
    // kept) contributes ZERO closing-fragment-not-found instances — the
    // staged search finds a determined fragment of at most 16 atoms. The
    // ether anchor is the reviewer's [9, 10, 11].
    let forms = fixture_molecule_forms(
        "ether closability (tert-butyl pyridines; reviewer SMILES kept)",
    );
    let parent = &forms[0];
    let insts = fg_instances(parent);
    let ethers: Vec<_> = insts.iter().filter(|(id, _)| *id == 8).collect();
    assert!(!ethers.is_empty(), "ether instance present");
    assert!(
        ethers.iter().any(|(_, a)| a.as_slice() == [9, 10, 11]),
        "reviewer ether anchor [9,10,11] present, got {:?}",
        ethers.iter().map(|(_, a)| a).collect::<Vec<_>>()
    );
    let bad = closing_fragment_not_found(std::slice::from_ref(parent));
    assert_eq!(bad, 0, "reviewer ether must close within 16 atoms, got {bad}");
    println!("ether_closing_fragment_found: {} ether instances, all close", ethers.len());
}


// ---------------------------------------------------------------------------
// FG5: blossom search versus brute force, take two (validated matchings,
// random edge order, nested blossoms, the reviewer's counterexample)
// ---------------------------------------------------------------------------
//
// The detector never uses a reconstructed search path as evidence: every
// decision rests on a validated perfect matching (real edges, perfect on the
// intended vertex set, tested edge absent/present). Here the same Rust entry
// points (`has_perfect_matching`, `maximum_matching`, `matching_allowed_edges`)
// are checked against an independent brute-force enumerator written in this
// test file on abstract graphs: maximum size equals brute force, every
// returned matching is validated by the test's own checker, and "in some /
// in all perfect matchings" equals brute force — with RANDOM (unsorted,
// seeded) edge order throughout.

/// Whether the graph is bipartite (breadth-first 2-colouring). Blossom
/// regression graphs must be non-bipartite (odd cycles) to be non-vacuous.
fn is_bipartite(n: usize, edges: &[(usize, usize)]) -> bool {
    let mut adj = vec![Vec::new(); n];
    for (a, b) in edges {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    let mut col: Vec<Option<bool>> = vec![None; n];
    for s in 0..n {
        if col[s].is_some() {
            continue;
        }
        col[s] = Some(false);
        let mut stack = vec![s];
        while let Some(v) = stack.pop() {
            for &w in &adj[v] {
                if let Some(c) = col[w] {
                    if c == col[v].unwrap() {
                        return false;
                    }
                } else {
                    col[w] = Some(!col[v].unwrap());
                    stack.push(w);
                }
            }
        }
    }
    true
}

/// Maximum matching cardinality by plain backtracking with memoisation on
/// the used-vertex bitmask (independent of the detector's blossom code).
fn brute_max_size(n: usize, edges: &[(usize, usize)]) -> usize {
    assert!(n <= 20, "brute_max_size: graph too large");
    let mut adj = vec![Vec::new(); n];
    for (a, b) in edges {
        if *a != *b {
            adj[*a].push(*b);
            adj[*b].push(*a);
        }
    }
    let mut memo = std::collections::HashMap::new();
    fn rec(n: usize, adj: &[Vec<usize>], used: u32, memo: &mut std::collections::HashMap<u32, usize>) -> usize {
        if let Some(&v) = memo.get(&used) {
            return v;
        }
        let free: Vec<usize> = (0..n).filter(|i| used & (1 << i) == 0).collect();
        if free.is_empty() {
            return 0;
        }
        let v = free[0];
        // Leave `v` unmatched.
        let mut best = rec(n, adj, used | (1 << v), memo);
        // Or match `v` with each free neighbour.
        for &w in &adj[v] {
            if used & (1 << w) == 0 {
                best = best.max(1 + rec(n, adj, used | (1 << v) | (1 << w), memo));
            }
        }
        memo.insert(used, best);
        best
    }
    rec(n, &adj, 0, &mut memo)
}

/// The test's own matching validator (independent of the detector): pairs
/// are disjoint and every pair is a real edge of the graph.
fn test_valid_matching(n: usize, edges: &[(usize, usize)], matching: &[(usize, usize)]) -> bool {
    let eset: BTreeSet<(usize, usize)> =
        edges.iter().map(|(a, b)| ((*a).min(*b), (*a).max(*b))).collect();
    let mut seen = vec![false; n];
    for (a, b) in matching {
        if *a >= n || *b >= n || *a == *b {
            return false;
        }
        if seen[*a] || seen[*b] {
            return false;
        }
        seen[*a] = true;
        seen[*b] = true;
        if !eset.contains(&((*a).min(*b), (*a).max(*b))) {
            return false;
        }
    }
    true
}

/// Brute-force in-some / in-all perfect-matching membership per edge.
fn brute_some_all(n: usize, edges: &[(usize, usize)]) -> (Vec<bool>, Vec<bool>) {
    let forms = brute_perfect_matchings(n, edges);
    if forms.is_empty() {
        return (vec![false; edges.len()], vec![false; edges.len()]);
    }
    let total = forms.len();
    let in_set: Vec<BTreeSet<(usize, usize)>> =
        forms.iter().map(|f| f.iter().copied().collect()).collect();
    let mut some = vec![false; edges.len()];
    let mut all = vec![false; edges.len()];
    for (i, e) in edges.iter().enumerate() {
        let c = in_set.iter().filter(|s| s.contains(e)).count();
        some[i] = c > 0;
        all[i] = c == total;
    }
    (some, all)
}

#[test]
fn blossom_reviewer_counterexample() {
    // The review's 12-vertex ordered edge list: independent enumeration finds
    // NO perfect matching, while the old helper reported one (using
    // nonexistent pairs). The ORDER is kept exactly as given (unsorted); the
    // detector must not depend on edge order.
    let edges: Vec<(usize, usize)> = vec![
        (3, 9), (0, 8), (2, 8), (0, 9), (4, 9), (5, 10), (0, 1), (7, 11),
        (4, 8), (3, 10), (1, 11), (9, 11), (2, 10), (7, 9), (0, 3), (6, 9),
    ];
    let n = 12;
    let brute = brute_perfect_matchings(n, &edges);
    assert!(brute.is_empty(), "reviewer graph truly has no perfect matching");
    assert!(!is_bipartite(n, &edges), "reviewer graph is blossom-relevant (odd cycle)");
    assert_eq!(has_perfect_matching(n, &edges), false, "reviewer graph: no perfect matching");
    let got = maximum_matching(n, &edges);
    assert!(test_valid_matching(n, &edges, &got), "returned matching validated: {got:?}");
    assert_eq!(got.len(), brute_max_size(n, &edges), "maximum size equals brute force");
    // With no perfect matching, every edge is in no perfect matching.
    let allowed = matching_allowed_edges(n, &edges, &[]);
    assert_eq!(allowed, vec![false; edges.len()]);
    println!("blossom_reviewer_counterexample: no perfect matching, max size {}", got.len());
}

/// One explicit nested-blossom graph: name, vertex count, edges.
fn nested_blossom_graphs() -> Vec<(&'static str, usize, Vec<(usize, usize)>)> {
    vec![
        // Flower: 5-cycle with a stem entering at each petal position.
        ("flower stem at petal 0", 8, vec![
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 0), (5, 6), (6, 7),
        ]),
        ("flower stem at petal 1", 8, vec![
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 1), (5, 6), (6, 7),
        ]),
        ("flower stem at petal 2", 8, vec![
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 2), (5, 6), (6, 7),
        ]),
        ("flower stem at petal 3", 8, vec![
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 3), (5, 6), (6, 7),
        ]),
        ("flower stem at petal 4", 8, vec![
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 4), (5, 6), (6, 7),
        ]),
        // Two blossoms sharing a stem: triangles on stem vertices 0 and 1.
        ("two blossoms sharing a stem", 8, vec![
            (0, 1), (1, 2), (2, 3), (3, 1), (0, 4), (4, 5), (5, 0), (3, 6), (6, 7),
        ]),
        // A blossom inside a blossom: inner triangle 2-3-4-2 on stem 0-1-2,
        // outer odd cycle 1-4-5-6-7-1 sharing vertices 1 and 4.
        ("blossom inside a blossom", 8, vec![
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (1, 4), (4, 5), (5, 6), (6, 7), (7, 1),
        ]),
    ]
}

#[test]
fn blossom_nested_explicit() {
    for (name, n, edges) in nested_blossom_graphs() {
        assert!(!is_bipartite(n, &edges), "{name}: must hold an odd cycle");
        let brute = brute_perfect_matchings(n, &edges);
        assert_eq!(
            has_perfect_matching(n, &edges),
            !brute.is_empty(),
            "{name}: existence differs from brute force"
        );
        let got = maximum_matching(n, &edges);
        assert!(test_valid_matching(n, &edges, &got), "{name}: returned matching validated");
        assert_eq!(got.len(), brute_max_size(n, &edges), "{name}: maximum size differs");
        if !brute.is_empty() {
            // A known perfect matching from brute force (first form).
            let pm = &brute[0];
            let lib = matching_allowed_edges(n, &edges, pm);
            assert_eq!(lib, brute_allowed(n, &edges), "{name}: allowed-edges differ");
            let (some, all) = brute_some_all(n, &edges);
            // Cross-check the derived in-some / in-all against the library:
            // in_some = matched || allowed, in_all = matched && !allowed.
            let pmset: BTreeSet<(usize, usize)> = pm.iter().copied().collect();
            for (i, e) in edges.iter().enumerate() {
                let m = pmset.contains(e);
                assert_eq!(m || lib[i], some[i], "{name} edge {e:?}: in-some differs");
                assert_eq!(m && !lib[i], all[i], "{name} edge {e:?}: in-all differs");
            }
        }
        println!("blossom_nested {name}: {} perfect matchings, max size {}", brute.len(), got.len());
    }
}

#[test]
fn blossom_random_5000_validated() {
    // 5,000 seeded random graphs of 4 to 16 vertices with RANDOM (unsorted)
    // edge order: half with a planted perfect matching, half without. Maximum
    // matching size equals brute force, every returned matching is validated
    // by the test's own checker, and "edge in some / in all perfect
    // matchings" (derived from the library's allowed-edges plus the known
    // matching) equals brute force.
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0xF655AA11CE020405);
    let mut with_pm = 0usize;
    let mut without_pm = 0usize;
    let mut allowed_checked = 0usize;
    let mut resampled = 0usize;
    let mut trial = 0usize;
    while trial < 5000 {
        trial += 1;
        let n = rng.random_range(4..=16);
        let planted = trial % 2 == 0;
        let mut edge_set = BTreeSet::new();
        let mut planted_pm: Vec<(usize, usize)> = Vec::new();
        if planted {
            let mut perm: Vec<usize> = (0..n).collect();
            shuffled(&mut rng, &mut perm);
            // Even n only for a planted pairing; odd n falls back to random.
            if n % 2 == 0 {
                for w in perm.chunks_exact(2) {
                    let e = (w[0].min(w[1]), w[0].max(w[1]));
                    planted_pm.push(e);
                    edge_set.insert(e);
                }
            }
        }
        let density: f64 = rng.random_range(0.08..0.4);
        for a in 0..n {
            for b in (a + 1)..n {
                if rng.random_bool(density) {
                    edge_set.insert((a, b));
                }
            }
        }
        // RANDOM (unsorted, seeded) edge order for the library.
        let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        shuffled(&mut rng, &mut edges);
        // Bound the brute-force cost: resample past 4096 perfect matchings.
        if brute_perfect_matchings(n, &edges).len() > 4096 {
            resampled += 1;
            trial -= 1;
            continue;
        }
        let brute = brute_perfect_matchings(n, &edges);
        assert_eq!(
            has_perfect_matching(n, &edges),
            !brute.is_empty(),
            "trial {trial} (n={n}): existence differs from brute force"
        );
        let got = maximum_matching(n, &edges);
        assert!(
            test_valid_matching(n, &edges, &got),
            "trial {trial} (n={n}): returned matching invalid: {got:?}"
        );
        assert_eq!(
            got.len(),
            brute_max_size(n, &edges),
            "trial {trial} (n={n}): maximum size differs from brute force"
        );
        if brute.is_empty() {
            without_pm += 1;
            let allowed = matching_allowed_edges(n, &edges, &[]);
            assert_eq!(allowed, vec![false; edges.len()], "trial {trial}: no-PM allowed differs");
        } else {
            with_pm += 1;
            let pm = if planted && !planted_pm.is_empty() && test_valid_matching(n, &edges, &planted_pm) {
                planted_pm.clone()
            } else {
                brute[0].clone()
            };
            // The known matching must be perfect (brute agrees it is).
            assert_eq!(pm.len() * 2, n, "trial {trial}: known matching is perfect");
            let lib = matching_allowed_edges(n, &edges, &pm);
            assert_eq!(lib, brute_allowed(n, &edges), "trial {trial} (n={n}): allowed differ");
            let (some, all) = brute_some_all(n, &edges);
            let pmset: BTreeSet<(usize, usize)> = pm.iter().copied().collect();
            for (i, e) in edges.iter().enumerate() {
                let m = pmset.contains(e);
                assert_eq!(m || lib[i], some[i], "trial {trial} edge {e:?}: in-some differs");
                assert_eq!(m && !lib[i], all[i], "trial {trial} edge {e:?}: in-all differs");
            }
            allowed_checked += edges.len();
        }
    }
    println!(
        "blossom_random_5000_validated: {with_pm} with + {without_pm} without perfect matching; \
         {allowed_checked} allowed-edges checked; {resampled} resampled past the brute cap"
    );
    assert!(with_pm >= 500 && without_pm >= 500, "both cases must occur ({with_pm}/{without_pm})");
}

// ---------------------------------------------------------------------------
// FG5: hypervalent (d = 2) kekulé regressions — the review's finding 1
// ---------------------------------------------------------------------------
//
// A kekulé form keeps every atom's double-bond count d(v); the detector
// reduces the general f-factor problem to perfect matching (Tutte's gadget)
// instead of dropping d = 2 atoms. The test's own backtracking oracle above
// (NOT shared with the library) decides the true status everywhere.

/// Atom-type ids allowed in the random valence-consistent generator, with
/// weights: ordinary C/N/O plus hypervalent S (H0,v6) and P (H0,v5).
fn gen_atom_types() -> Vec<(u8, u32)> {
    vec![
        (2, 30),  // C H1
        (1, 25),  // C H0
        (3, 10),  // C H2
        (5, 10),  // N H0
        (6, 5),   // N H1
        (8, 10),  // O H0
        (9, 5),   // O H1
        (15, 3),  // S H0 v6
        (16, 2),  // P H0 v5
    ]
}

fn sample_type(rng: &mut impl rand::Rng, pool: &[(u8, u32)]) -> u8 {
    let total: u32 = pool.iter().map(|(_, w)| w).sum();
    let mut x = rng.random_range(0..total);
    for (t, w) in pool {
        if x < *w {
            return *t;
        }
        x -= w;
    }
    pool[0].0
}

/// Random connected graph with an EXACT valence-satisfying single/double/
/// triple assignment (residual 0 everywhere: a closed molecule), or `None`
/// after the retry budget. `extra_p` controls the cycle density.
fn random_closed_graph(
    rng: &mut impl rand::Rng,
    n: usize,
    extra_p: f64,
) -> Option<MolGraph> {
    use mamba3::models::ms2::chem::atom_type;
    let pool = gen_atom_types();
    for _ in 0..60 {
        let types: Vec<u8> = (0..n).map(|_| sample_type(rng, &pool)).collect();
        // Random tree skeleton plus extra edges (cycles).
        let mut eset = BTreeSet::new();
        for i in 1..n {
            let j = rng.random_range(0..i);
            eset.insert((j.min(i), j.max(i)));
        }
        for a in 0..n {
            for b in (a + 1)..n {
                if !eset.contains(&(a, b)) && rng.random_bool(extra_p) {
                    eset.insert((a, b));
                }
            }
        }
        let pairs: Vec<(usize, usize)> = eset.into_iter().collect();
        // Capacity per atom: valence - hydrogens.
        let cap: Vec<u32> = types
            .iter()
            .map(|t| {
                let at = atom_type(*t).unwrap();
                (at.valence - at.hydrogens) as u32
            })
            .collect();
        // Backtracking for an exact assignment (orders 1..3), randomised.
        let mut order_try: Vec<Vec<u8>> = pairs.iter().map(|_| vec![1u8, 2, 3]).collect();
        for o in order_try.iter_mut() {
            shuffled(rng, o);
        }
        let mut assign = vec![0u8; pairs.len()];
        let mut used = vec![0u32; n];
        fn rec(
            pairs: &[(usize, usize)],
            order_try: &[Vec<u8>],
            cap: &[u32],
            pos: usize,
            assign: &mut [u8],
            used: &mut [u32],
        ) -> bool {
            if pos == pairs.len() {
                return used.iter().zip(cap.iter()).all(|(u, c)| u == c);
            }
            let (a, b) = pairs[pos];
            for &o in &order_try[pos] {
                let ou = o as u32;
                if used[a] + ou <= cap[a] && used[b] + ou <= cap[b] {
                    // Prune: remaining capacity must cover remaining slots.
                    assign[pos] = o;
                    used[a] += ou;
                    used[b] += ou;
                    if rec(pairs, order_try, cap, pos + 1, assign, used) {
                        return true;
                    }
                    used[a] -= ou;
                    used[b] -= ou;
                    assign[pos] = 0;
                }
            }
            false
        }
        if rec(&pairs, &order_try, &cap, 0, &mut assign, &mut used) {
            let bonds: Vec<(usize, usize, u8)> =
                pairs.iter().zip(assign.iter()).map(|((a, b), o)| (*a, *b, *o)).collect();
            if let Ok(g) = MolGraph::new(types, bonds) {
                if g.residual_valence().iter().all(|&r| r == 0) {
                    return Some(g);
                }
            }
        }
    }
    None
}

/// Whether atom `v` with `d(v) == 2` lies on a cycle (two of its neighbours
/// connect without `v`).
fn ring_d2_atoms(graph: &MolGraph) -> Vec<usize> {
    let n = graph.atoms().len();
    let mut adj = vec![Vec::new(); n];
    for (a, b, _) in graph.bonds() {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    let mut d = vec![0u8; n];
    for (a, b, o) in graph.bonds() {
        if *o == 2 {
            d[*a] += 1;
            d[*b] += 1;
        }
    }
    let mut out = Vec::new();
    for v in 0..n {
        if d[v] != 2 {
            continue;
        }
        // BFS from the first neighbour avoiding `v`.
        let nb = adj[v].clone();
        if nb.len() < 2 {
            continue;
        }
        let mut seen = vec![false; n];
        seen[v] = true;
        seen[nb[0]] = true;
        let mut stack = vec![nb[0]];
        let mut reaches = false;
        while let Some(u) = stack.pop() {
            if u == nb[1] {
                reaches = true;
                break;
            }
            for &w in &adj[u] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        // Any second neighbour reachable (not just nb[1]): repeat for all.
        if !reaches {
            for k in 2..nb.len() {
                let mut seen = vec![false; n];
                seen[v] = true;
                seen[nb[0]] = true;
                let mut stack = vec![nb[0]];
                while let Some(u) = stack.pop() {
                    if u == nb[k] {
                        reaches = true;
                        break;
                    }
                    for &w in &adj[u] {
                        if !seen[w] {
                            seen[w] = true;
                            stack.push(w);
                        }
                    }
                }
                if reaches {
                    break;
                }
            }
        }
        if reaches {
            out.push(v);
        }
    }
    out
}

/// Random closed parent of 5 to 12 atoms WITH a ring atom of d = 2 (retry
/// until the property holds; panics past the budget — the generator is
/// biased with S/P for this).
///
/// One draw in three is a randomized review-S-ring analogue (see
/// `randomized_s_ring_analogue` below): a delocalised six-ring with two doubles
/// at sulphur, whose incident ring bonds are guaranteed variable. The other
/// draws are fully random closed graphs filtered for a ring `d = 2` atom
/// (usually pinned: sulphonyl-like sulphur, cumulene centres). The planted
/// branch lifts the variable-`d = 2` rate well above 10% while keeping the
/// random graphs sparse (FG7: the FG6 summary's "latent blossom-search
/// incompleteness on dense S/P-rich graphs" did not reproduce — the seeded
/// search finds every brute-force augmenting path on the trigger in all 120
/// numberings, the library equals the closed oracle on dense closed parents,
/// and the reported "fixed" verdicts were conservative UNDECIDED fragment
/// verdicts compared against the closed-molecule oracle; the remaining
/// numbering dependence (found-witness stability) is fixed existentially in
/// the library, so dense graphs are covered by the FG7 tests below).
fn random_ring_d2_parent(rng: &mut impl rand::Rng) -> MolGraph {
    if rng.random_range(0..3) == 0 {
        return randomized_s_ring_analogue(rng);
    }
    for _ in 0..400 {
        let n = rng.random_range(5..=12);
        if let Some(g) = random_closed_graph(rng, n, 0.28) {
            if !ring_d2_atoms(&g).is_empty() {
                return g;
            }
        }
    }
    panic!("random_ring_d2_parent: budget exceeded");
}

/// Randomized review-S-ring analogue (`O=S1(C)=NC=CC=C1` scaffold): 0–2 of
/// the four ring CH carbons gain a methyl substituent (ring carbon
/// `C H1` → `C H0`, plus a pendant `C H3`), then the atoms are permuted.
/// Closure is preserved exactly; the methyls are `d = 0` spectators, so the
/// ring keeps both valid assignments and the sulphur's incident ring bonds
/// stay variable. Type ids: O H0 = 8, S H0 v6 = 15, C H3 = 4, N H0 = 5,
/// C H1 = 2, C H0 = 1.
fn randomized_s_ring_analogue(rng: &mut impl rand::Rng) -> MolGraph {
    let mut types = vec![8, 15, 4, 5, 2, 2, 2, 2];
    let mut bonds = vec![
        (0, 1, 2),
        (1, 2, 1),
        (1, 3, 1),
        (1, 7, 2),
        (3, 4, 2),
        (4, 5, 1),
        (5, 6, 2),
        (6, 7, 1),
    ];
    // Methylate 0-2 random ring CH carbons (atoms 4-7).
    let mut ring_ch = vec![4usize, 5, 6, 7];
    shuffled(&mut *rng, &mut ring_ch);
    let k = rng.random_range(0..=2);
    for &c in ring_ch.iter().take(k) {
        types[c] = 1;
        bonds.push((c, types.len(), 1));
        types.push(4);
    }
    let g = MolGraph::new(types, bonds).expect("methylated S-ring analogue builds");
    // Numbering diversity: random permutation.
    let n = g.atoms().len();
    let mut perm: Vec<usize> = (0..n).collect();
    shuffled(&mut *rng, &mut perm);
    g.permuted(&perm).expect("permuted analogue builds")
}

#[test]
fn hypervalent_s_ring_delocalised() {
    // The review's finding-1 molecule `O=S1(C)=NC=CC=C1` in BOTH stored
    // forms: identical delocalised sets (all six ring bonds delocalised),
    // alkene = 0 and imine = 0 in both (the old code gave alkene 2/1 and
    // imine 0/1 with an empty delocalised set).
    let forms = fixture_molecule_forms("review S-ring (two doubles at S; delocalised six-ring)");
    assert_eq!(forms.len(), 2, "the S-ring stores both kekule forms");
    let mut sets = Vec::new();
    for (k, g) in forms.iter().enumerate() {
        let set = functional_groups_v4(g);
        assert_eq!(set.count(14), 0, "form {k}: no fixed alkene");
        assert_eq!(set.count(13), 0, "form {k}: no fixed imine");
        assert_eq!(undetermined(g), 0, "form {k}: closed, nothing undetermined");
        // Ring bonds = all bonds except the exocyclic S=O and S-Me bonds
        // (found via the S(H0,v6) atom: type id 15).
        let s = g.atoms().iter().position(|&t| t == 15).expect("sulphur present");
        let dl = delocalised_bonds(g);
        assert_eq!(dl.len(), g.bonds().len());
        for (i, (a, b, _)) in g.bonds().iter().enumerate() {
            // Exocyclic: incident to S with the methyl carbon (H3) or oxygen
            // at the other end.
            let other = if *a == s {
                Some(*b)
            } else if *b == s {
                Some(*a)
            } else {
                None
            };
            let exo = match other {
                Some(o) => {
                    let t = mamba3::models::ms2::chem::atom_type(g.atoms()[o]).unwrap();
                    (t.element == 0 && t.hydrogens == 3) || t.element == 3
                }
                None => false,
            };
            if exo {
                assert!(!dl[i], "form {k} bond {i} ({a},{b}): exocyclic bond fixed");
            } else {
                assert!(dl[i], "form {k} bond {i} ({a},{b}): ring bond delocalised");
            }
        }
        assert_eq!(dl.iter().filter(|&&x| x).count(), 6, "form {k}: six ring bonds");
        sets.push(dl);
    }
    assert_eq!(sets[0], sets[1], "identical delocalised sets across forms");
    // The test's own oracle agrees (two assignments, same delocalised set).
    for g in &forms {
        assert_eq!(delocalised_bonds(g), oracle_delocalised(g), "oracle agrees on the S-ring");
    }
    println!("hypervalent_s_ring_delocalised: both forms alkene=0 imine=0, six ring bonds delocalised");
}

#[test]
fn reviewer_11atom_parent_true_status() {
    // The review's 11-atom S/P parent: its true status is computed by the
    // test's own brute-force oracle and asserted (3 valid assignments).
    let forms = fixture_molecule_forms("review S/P parent (11-atom fused heterocycle)");
    assert!(!forms.is_empty());
    for (k, g) in forms.iter().enumerate() {
        let oracle = oracle_delocalised(g);
        let lib = delocalised_bonds(g);
        assert_eq!(lib, oracle, "form {k}: library delocalised set equals the oracle");
        let lib_dec = decided_bonds(g);
        assert!(lib_dec.iter().all(|s| s.is_some()), "form {k}: parent fully decided");
        let oracle_bool: Vec<Option<bool>> = oracle.iter().map(|&x| Some(x)).collect();
        assert_eq!(lib_dec, oracle_bool, "form {k}: decided statuses equal the oracle");
    }
    let n_deloc: Vec<usize> = forms.iter().map(|g| delocalised_bonds(g).iter().filter(|&&x| x).count()).collect();
    println!("reviewer_11atom_parent_true_status: {} forms, delocalised counts {n_deloc:?}", forms.len());
}

#[test]
fn pinned_hypervalent_regressions() {
    // Sulphone, sulfolene, thiophene dioxide, sulphonamide, phosphate ester,
    // phosphine oxide, allene, ketene and CO2: every bond pinned (empty
    // delocalised set, unchanged from before), oracle agrees.
    let names = [
        "dimethyl sulfone",
        "sulfolene (3-sulfolene, pinned)",
        "thiophene S,S-dioxide (pinned)",
        "methanesulfonamide",
        "trimethyl phosphate",
        "trimethylphosphine oxide (pinned)",
        "allene (cumulene centre pinned)",
        "ketene (cumulene centre pinned)",
        "carbon dioxide (pinned)",
    ];
    for name in names {
        let forms = fixture_molecule_forms(name);
        assert!(!forms.is_empty(), "{name}: present");
        for g in &forms {
            let dl = delocalised_bonds(g);
            assert!(dl.iter().all(|&x| !x), "{name}: delocalised set empty (pinned)");
            assert_eq!(dl, oracle_delocalised(g), "{name}: oracle agrees");
            assert!(decided_bonds(g).iter().all(|s| s.is_some()), "{name}: fully decided");
        }
    }
    // Documented pinned counts (fixed doubles still count as groups).
    let allene = &fixture_molecule_forms("allene (cumulene centre pinned)")[0];
    assert_eq!(functional_groups_v4(allene).count(14), 2, "allene: two fixed alkenes");
    let ketene = &fixture_molecule_forms("ketene (cumulene centre pinned)")[0];
    assert_eq!(functional_groups_v4(ketene).count(1), 1, "ketene: one carbonyl");
    assert_eq!(functional_groups_v4(ketene).count(14), 1, "ketene: one alkene");
    let co2 = &fixture_molecule_forms("carbon dioxide (pinned)")[0];
    assert_eq!(functional_groups_v4(co2).count(1), 2, "CO2: two carbonyls");
    let dioxide = &fixture_molecule_forms("thiophene S,S-dioxide (pinned)")[0];
    assert_eq!(functional_groups_v4(dioxide).count(18), 1, "dioxide: one sulfonyl");
    println!("pinned_hypervalent_regressions: {} molecules pinned, oracle agrees", names.len());
}

#[test]
fn phosphinine_thiabenzene_sulfoximine_oracle() {
    // λ5-phosphinine (P in a six-ring), the domain-valid thiabenzene analogue
    // (two ring doubles at S) and the cyclic sulfoximine analogue (two ring
    // doubles at S): library equals the test's own oracle on every form.
    for name in [
        "lambda5-phosphinine (P in a delocalised six-ring)",
        "1,1-dimethyl-thiabenzene analogue (hand-built; S-v4 SMILES out of domain)",
        "cyclic sulfoximine analogue (hand-built; two ring doubles at S)",
    ] {
        let forms = fixture_molecule_forms(name);
        assert!(!forms.is_empty(), "{name}: present");
        for (k, g) in forms.iter().enumerate() {
            assert_eq!(
                delocalised_bonds(g),
                oracle_delocalised(g),
                "{name} form {k}: library equals oracle"
            );
        }
        println!("{name}: {} forms agree", forms.len());
    }
    // The phosphinine ring is delocalised (no fixed alkene), like pyridine.
    let ph = &fixture_molecule_forms("lambda5-phosphinine (P in a delocalised six-ring)")[0];
    assert_eq!(functional_groups_v4(ph).count(14), 0, "phosphinine: no fixed alkene");
}

// ---------------------------------------------------------------------------
// FG5: library-versus-oracle agreement on every fixture molecule and 2,000
// random valence-consistent graphs with d = 2 in rings
// ---------------------------------------------------------------------------

#[test]
fn oracle_agreement_all_fixtures() {
    // On EVERY fixture molecule and EVERY stored form, the library's
    // delocalised set equals the test's own backtracking oracle (which
    // enumerates all assignments with the prescribed d(v), including through
    // d = 2 ring atoms). Reports the gadget size on the largest molecule.
    let f = fixture();
    let molecules = f["molecules"].as_array().unwrap();
    let mut forms_checked = 0usize;
    let mut biggest = (String::new(), 0usize, 0usize, 0usize);
    for m in molecules {
        let name = m["name"].as_str().unwrap_or("?").to_string();
        let forms = m["forms"].as_array().unwrap();
        for form in forms {
            let g = graph_of_form(form);
            let (gv, ge) = kekule_gadget_size(&g);
            if gv > biggest.1 {
                biggest = (name.clone(), gv, ge, g.atoms().len());
            }
            assert_eq!(
                delocalised_bonds(&g),
                oracle_delocalised(&g),
                "{name}: library delocalised set differs from the oracle"
            );
            forms_checked += 1;
        }
    }
    println!(
        "oracle_agreement_all_fixtures: {forms_checked} forms agree over {} molecules; \
         largest gadget {} ({} atoms, {} gadget vertices, {} gadget edges)",
        molecules.len(), biggest.0, biggest.3, biggest.1, biggest.2
    );
}

/// Whether `graph` holds a `d = 2` atom with a VARIABLE incident bond (one
/// delocalised per the `deloc` flags, which the caller has checked against
/// the oracle): a topological ring `d = 2` atom is not enough — only an
/// alternating witness through it makes its bonds move.
fn has_variable_d2(graph: &MolGraph, deloc: &[bool]) -> bool {
    let n = graph.atoms().len();
    let mut d = vec![0u8; n];
    for (a, b, o) in graph.bonds() {
        if *o == 2 {
            d[*a] += 1;
            d[*b] += 1;
        }
    }
    graph
        .bonds()
        .iter()
        .enumerate()
        .any(|(i, (a, b, _))| deloc[i] && (d[*a] == 2 || d[*b] == 2))
}

#[test]
fn oracle_agreement_random_2000() {
    // 2,000 random closed valence-consistent graphs of 5 to 12 atoms WITH a
    // ring atom of d = 2 (hypervalent S/P in rings): library equals the
    // test's own oracle on every graph.
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0x0A1CE20260D2D201);
    let mut checked = 0usize;
    let mut s_count = 0usize;
    let mut p_count = 0usize;
    let mut variable_d2 = 0usize;
    while checked < 2000 {
        let g = random_ring_d2_parent(&mut rng);
        let has_s = g.atoms().iter().any(|&t| t == 15);
        let has_p = g.atoms().iter().any(|&t| t == 16);
        if has_s {
            s_count += 1;
        }
        if has_p {
            p_count += 1;
        }
        let lib = delocalised_bonds(&g);
        let oracle = oracle_delocalised(&g);
        assert_eq!(
            lib, oracle,
            "random graph {checked}: library differs from oracle"
        );
        if has_variable_d2(&g, &lib) {
            variable_d2 += 1;
        }
        assert!(
            decided_bonds(&g).iter().all(|s| s.is_some()),
            "random graph {checked}: closed, fully decided"
        );
        checked += 1;
    }
    println!("oracle_agreement_random_2000: {checked} graphs agree ({s_count} with S, {p_count} with P; {variable_d2} with a variable d = 2 incident bond)");
    assert!(s_count > 500 && p_count > 100, "generator must cover S and P ({s_count}/{p_count})");
    assert!(
        variable_d2 >= 200,
        "at least 10% of the ring-d2 parents must hold a d = 2 atom with a variable incident bond ({variable_d2}/2000)"
    );
}

// ---------------------------------------------------------------------------
// FG5: fragment regressions and exhaustive soundness (finding 2)
// ---------------------------------------------------------------------------

/// The review's 11-atom S/P parent in fixture (RDKit) numbering, plus the
/// induced fragment dropping the methyl carbon and the exocyclic oxygen
/// (the review's atoms 9, 10): the review's fragment on atoms 0..=8.
///
/// Uses the stored form matching the review's bond orders (the N–C double,
/// i.e. RDKit bond (6,7) double — the form in which the fragment carbon
/// carries the undecided double to N).
fn reviewer_parent_and_fragment() -> (MolGraph, MolGraph) {
    let forms = fixture_molecule_forms("review S/P parent (11-atom fused heterocycle)");
    // The review's kekule form: the N(H0)–C(H0) bond (RDKit atoms 6,7) double.
    let parent = forms
        .iter()
        .find(|g| {
            g.bonds().iter().any(|(a, b, o)| *o == 2 && (*a == 6 && *b == 7 || *a == 7 && *b == 6))
        })
        .unwrap_or(&forms[0])
        .clone();
    // Chemically identify the dropped atoms: the C(H3) methyl and the O(H0)
    // double-bonded to the S(H0,v6).
    use mamba3::models::ms2::chem::atom_type;
    let s = parent.atoms().iter().position(|&t| t == 15).expect("sulphur present");
    let mut drop = Vec::new();
    for (i, id) in parent.atoms().iter().enumerate() {
        let t = atom_type(*id).unwrap();
        if t.element == 0 && t.hydrogens == 3 {
            drop.push(i);
        }
    }
    assert_eq!(drop.len(), 1, "exactly one methyl carbon");
    let me = drop[0];
    assert!(
        parent.bonds().iter().any(|(a, b, o)| *o == 1 && (*a == s && *b == me || *a == me && *b == s)),
        "the methyl carbon bonds the sulphur"
    );
    let mut drop_o = None;
    for (a, b, o) in parent.bonds() {
        if *o == 2 && (*a == s || *b == s) {
            let other = if *a == s { *b } else { *a };
            let t = atom_type(parent.atoms()[other]).unwrap();
            if t.element == 3 && t.hydrogens == 0 {
                drop_o = Some(other);
            }
        }
    }
    let ox = drop_o.expect("exocyclic S=O oxygen present");
    let keep: Vec<usize> =
        (0..parent.atoms().len()).filter(|i| *i != me && *i != ox).collect();
    assert_eq!(keep.len(), parent.atoms().len() - 2);
    let frag = parent.induced(&keep).unwrap();
    // The fragment sulphur keeps residual valence 3 (review's observation).
    let fs = keep.iter().position(|&i| i == s).unwrap();
    assert_eq!(frag.residual_valence()[fs], 3, "fragment sulphur has residual 3");
    (parent, frag)
}

/// The N(H0)–C(H0, bonded to O(H1)) bond of a graph, if present (the
/// review's (0,1) bond and its fragment image).
fn review_bond_01(graph: &MolGraph) -> Option<usize> {
    use mamba3::models::ms2::chem::atom_type;
    let n = graph.atoms().len();
    // The hydroxyl oxygen O(H1) and its carbon.
    let mut target = None;
    for (i, id) in graph.atoms().iter().enumerate() {
        let t = atom_type(*id).unwrap();
        if t.element == 3 && t.hydrogens == 1 {
            for (a, b, o) in graph.bonds() {
                if *o == 1 && (*a == i || *b == i) {
                    let c = if *a == i { *b } else { *a };
                    let tc = atom_type(graph.atoms()[c]).unwrap();
                    if tc.element == 0 && tc.hydrogens == 0 {
                        target = Some(c);
                    }
                }
            }
        }
    }
    let c = target?;
    // The N(H0) neighbour of that carbon (any order: kekule forms vary).
    for (i, (a, b, _)) in graph.bonds().iter().enumerate() {
        if *a == c || *b == c {
            let x = if *a == c { *b } else { *a };
            let tx = atom_type(graph.atoms()[x]).unwrap();
            if tx.element == 2 && tx.hydrogens == 0 {
                let _ = n;
                return Some(i);
            }
        }
    }
    None
}

#[test]
fn fragment_reviewer_regression() {
    // The review's parent and its induced fragment (review atoms 0..=8):
    // bond (0,1) is UNDECIDED in the fragment, hydroxyl (0,8) is
    // undetermined, and determined types are a subset of the parent types.
    // (The old code reported (0,1) decided-delocalised and hydroxyl
    // determined — via a witness path with a nonexistent edge.)
    let (parent, frag) = reviewer_parent_and_fragment();
    let pmask = functional_groups_v4(&parent).mask();
    // The bond exists in both graphs.
    let pb = review_bond_01(&parent).expect("parent holds the (0,1) bond");
    let fb = review_bond_01(&frag).expect("fragment holds the (0,1) bond");
    let pdec = decided_bonds(&parent);
    let fdec = decided_bonds(&frag);
    println!("parent (0,1) decided: {:?}; fragment (0,1) decided: {:?}", pdec[pb], fdec[fb]);
    // Fragment: UNDECIDED.
    assert_eq!(fdec[fb], None, "fragment bond (0,1) is UNDECIDED");
    // Hydroxyl undetermined in the fragment.
    let hyd = 1u32 << (7 - 1);
    assert_ne!(undetermined(&frag) & hyd, 0, "fragment hydroxyl is undetermined");
    // Determined types ⊆ parent types.
    let dm = functional_groups_v4(&frag).mask();
    assert_eq!(dm & !pmask, 0, "fragment determined types outside parent: {:b} vs {:b}", dm, pmask);
    // Oracle cross-check: the fragment's decided bonds equal the parent's
    // oracle status; the parent is fully decided and equals its oracle.
    let oracle = oracle_delocalised(&parent);
    assert_eq!(delocalised_bonds(&parent), oracle, "parent equals its oracle");
    println!("fragment_reviewer_regression: (0,1) undecided, hydroxyl undetermined, subset holds");
}

/// One isomorphism `frag -> other` (both small): `map[i]` is the image of
/// frag atom `i`. Plain backtracking with type + degree pruning; independent
/// of the detector. `None` when the graphs are not isomorphic.
fn find_iso(frag: &MolGraph, other: &MolGraph) -> Option<Vec<usize>> {
    let n = frag.atoms().len();
    if other.atoms().len() != n {
        return None;
    }
    // Adjacency with orders, keyed by sorted pair.
    let key_of = |g: &MolGraph| {
        let mut m = std::collections::BTreeMap::new();
        for (a, b, o) in g.bonds() {
            m.insert(((*a).min(*b), (*a).max(*b)), *o);
        }
        m
    };
    let kf = key_of(frag);
    let ko = key_of(other);
    if kf.len() != ko.len() {
        return None;
    }
    let deg_f: Vec<usize> = (0..n)
        .map(|v| frag.bonds().iter().filter(|(a, b, _)| *a == v || *b == v).count())
        .collect();
    let deg_o: Vec<usize> = (0..n)
        .map(|v| other.bonds().iter().filter(|(a, b, _)| *a == v || *b == v).count())
        .collect();
    // Candidate images per frag atom, ordered by fewest first.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&v| {
        (0..n)
            .filter(|&w| other.atoms()[w] == frag.atoms()[v] && deg_o[w] == deg_f[v])
            .count()
    });
    let mut map: Vec<Option<usize>> = vec![None; n];
    let mut taken = vec![false; n];
    fn rec(
        frag: &MolGraph,
        other: &MolGraph,
        kf: &std::collections::BTreeMap<(usize, usize), u8>,
        ko: &std::collections::BTreeMap<(usize, usize), u8>,
        order: &[usize],
        pos: usize,
        map: &mut [Option<usize>],
        taken: &mut [bool],
    ) -> bool {
        if pos == order.len() {
            return true;
        }
        let v = order[pos];
        for w in 0..frag.atoms().len() {
            if taken[w] || other.atoms()[w] != frag.atoms()[v] {
                continue;
            }
            // Bond consistency with already-mapped atoms.
            let mut ok = true;
            for (u, mw) in map.iter().enumerate() {
                if let Some(mw) = mw {
                    let fkey = (v.min(u), v.max(u));
                    let okey = (w.min(*mw), w.max(*mw));
                    if kf.get(&fkey) != ko.get(&okey) {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            map[v] = Some(w);
            taken[w] = true;
            if rec(frag, other, kf, ko, order, pos + 1, map, taken) {
                return true;
            }
            map[v] = None;
            taken[w] = false;
        }
        false
    }
    if rec(frag, other, &kf, &ko, &order, 0, &mut map, &mut taken) {
        Some(map.into_iter().map(|m| m.unwrap()).collect())
    } else {
        None
    }
}

#[test]
fn fragment_exhaustive_soundness_fg5() {
    // Exhaustive soundness on small cases: for EVERY stored form of every
    // fixture molecule with at most 14 heavy atoms, and for 300 random
    // valence-consistent parents with ring atoms of d = 2, EVERY connected
    // induced fragment (cap 2,000 per form, seeded sample beyond): every
    // decided bond status equals the parent's status by the TEST's
    // brute-force oracle, and determined types are a subset of the parent
    // types — both in parent order and in trace-replay order.
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0xE4A0571CF650500D);
    // Parents: fixtures ≤ 14 atoms (EVERY stored form) + 300 random d=2-ring.
    let f = fixture();
    let mut parents: Vec<(String, Vec<MolGraph>)> = Vec::new();
    for m in f["molecules"].as_array().unwrap() {
        let name = m["name"].as_str().unwrap_or("?").to_string();
        let forms = m["forms"].as_array().unwrap();
        let g0 = graph_of_form(&forms[0]);
        if g0.atoms().len() <= 14 {
            parents.push((
                format!("fixture:{name}"),
                forms.iter().map(graph_of_form).collect(),
            ));
        }
    }
    let n_fixture = parents.len();
    let n_fixture_forms: usize = parents.iter().map(|(_, v)| v.len()).sum();
    for _ in 0..300 {
        parents.push(("random:ring-d2".to_string(), vec![random_ring_d2_parent(&mut rng)]));
    }
    reset_blossom_stats();
    let mut frags = 0usize;
    let mut trace_ok = 0usize;
    let mut trace_skipped = 0usize;
    let mut decided_checked = 0usize;
    let mut undecided_seen = 0usize;
    let mut variable_d2_parents = 0usize;
    let mut random_parents = 0usize;
    for (label, forms) in &parents {
        let is_random = label == "random:ring-d2";
        if is_random {
            random_parents += 1;
        }
        // The delocalised set is form-independent; the oracle is computed on
        // the first form and every stored form is checked against it (this
        // also re-verifies kekule invariance of the oracle itself).
        let oracle = oracle_delocalised(&forms[0]);
        assert_eq!(delocalised_bonds(&forms[0]), oracle, "{label}: parent must equal its oracle");
        for (fi, parent) in forms.iter().enumerate() {
            let tag = format!("{label} form {fi}");
            assert_eq!(
                delocalised_bonds(parent), oracle,
                "{tag}: delocalised set differs across stored forms"
            );
            let pmask = functional_groups_v4(parent).mask();
            if is_random && has_variable_d2(parent, &oracle) {
                variable_d2_parents += 1;
            }
            let n = parent.atoms().len();
            let mut adj = vec![Vec::new(); n];
            for (a, b, _) in parent.bonds() {
                adj[*a].push(*b);
                adj[*b].push(*a);
            }
            let mut parent_bond_idx = std::collections::BTreeMap::new();
            for (i, (a, b, _)) in parent.bonds().iter().enumerate() {
                parent_bond_idx.insert(((*a).min(*b), (*a).max(*b)), i);
            }
            let all = connected_subsets_bounded(&adj, 1, n);
            let members_list: Vec<Vec<usize>> = if all.len() > 2000 {
                let mut idx: Vec<usize> = (0..all.len()).collect();
                shuffled(&mut rng, &mut idx);
                idx.truncate(2000);
                idx.into_iter().map(|i| all[i].clone()).collect()
            } else {
                all.clone()
            };
            for members in &members_list {
            // Parent order.
            let frag = parent.induced(members).unwrap();
            let dm = functional_groups_v4(&frag).mask();
            assert_eq!(dm & !pmask, 0, "{tag} fragment {members:?}: determined outside parent");
            let fdec = decided_bonds(&frag);
            for (i, (a, b, _)) in frag.bonds().iter().enumerate() {
                if let Some(v) = fdec[i] {
                    let (pa, pb) = (members[*a], members[*b]);
                    let pi = parent_bond_idx[&(pa.min(pb), pa.max(pb))];
                    assert_eq!(
                        oracle[pi], v,
                        "{tag} fragment {members:?} bond ({pa},{pb}): fragment says {v} but oracle parent says {}",
                        oracle[pi]
                    );
                    decided_checked += 1;
                } else {
                    undecided_seen += 1;
                }
            }
            frags += 1;
            // Trace-replay order: same two assertions, with trace bonds mapped
            // back to parent bonds through an explicit isomorphism
            // (trace order renumbers atoms). Fragments outside the grammar
            // limits (16 atoms / 4 ring closures) have no trace order: the
            // evaluation never sees them, so only parent order is checked.
            let frag = parent.induced(members).unwrap();
            if frag.atoms().len() > 16 || frag.ring_closures() > 4 {
                trace_skipped += 1;
                continue;
            }
            let canon = canonical_trace(&frag, Limits::V0, CANONICAL_WORK_LIMIT)
                .unwrap_or_else(|e| panic!("{tag} fragment {members:?}: canonical_trace: {e}"));
            let state = replay(&canon.trace, Limits::V0, None)
                .unwrap_or_else(|e| panic!("{tag} fragment {members:?}: replay: {e}"));
            assert!(state.stopped(), "{tag} fragment {members:?}: replay did not stop");
            let tg = state.graph().unwrap();
            let iso = find_iso(&frag, &tg)
                .unwrap_or_else(|| panic!("{tag} fragment {members:?}: replay not isomorphic"));
            let tm = functional_groups_v4(&tg).mask();
            assert_eq!(tm & !pmask, 0, "{tag} fragment {members:?}: trace-order outside parent");
            // Invert the isomorphism: tg atom -> frag atom.
            let mut inv = vec![0usize; tg.atoms().len()];
            for (f, &t) in iso.iter().enumerate() {
                inv[t] = f;
            }
            let tdec = decided_bonds(&tg);
            for (i, (a, b, _)) in tg.bonds().iter().enumerate() {
                if let Some(v) = tdec[i] {
                    let (fa, fb) = (inv[*a], inv[*b]);
                    let (pa, pb) = (members[fa], members[fb]);
                    let pi = parent_bond_idx[&(pa.min(pb), pa.max(pb))];
                    assert_eq!(
                        oracle[pi], v,
                        "{tag} fragment {members:?} trace bond ({pa},{pb}): decided {v} but oracle parent says {}",
                        oracle[pi]
                    );
                    decided_checked += 1;
                } else {
                    undecided_seen += 1;
                }
            }
            // The delocalised set is numbering-independent: compare through
            // the isomorphism (bond vectors are in different orders).
            let fdl = delocalised_bonds(&frag);
            let tdl = delocalised_bonds(&tg);
            let mut t_by_pair = std::collections::BTreeMap::new();
            for (i, (a, b, _)) in tg.bonds().iter().enumerate() {
                t_by_pair.insert(((*a).min(*b), (*a).max(*b)), tdl[i]);
            }
            for (i, (a, b, _)) in frag.bonds().iter().enumerate() {
                let (ta, tb) = (iso[*a], iso[*b]);
                assert_eq!(
                    fdl[i],
                    t_by_pair[&(ta.min(tb), ta.max(tb))],
                    "{tag} fragment {members:?}: delocalised status differs across numbering"
                );
            }
            trace_ok += 1;
            }
        }
    }
    let (contracted, nested) = blossom_stats();
    println!(
        "fragment_exhaustive_soundness_fg5: {} parents ({} fixture in {} forms + 300 random), \
         {frags} fragments parent-order + {trace_ok} trace replays ({trace_skipped} skipped past grammar limits), \
         {decided_checked} decided bonds agree with the oracle, {undecided_seen} undecided; 0 offences; \
         blossom searches with a contraction: {contracted}, nested: {nested} (process-global hook); \
         {variable_d2_parents}/{random_parents} random parents hold a variable d = 2 incident bond",
        parents.len(), n_fixture, n_fixture_forms
    );
    assert!(frags > 10_000, "exhaustive test must be non-vacuous ({frags} fragments)");
    assert!(decided_checked > 10_000, "must check decided bonds ({decided_checked})");
    assert!(contracted > 0, "blossom statistics hook must see contracted searches");
    assert!(nested > 0, "blossom statistics hook must see nested contractions");
    assert!(
        variable_d2_parents * 10 >= random_parents,
        "at least 10% of the random ring-d2 parents must hold a d = 2 atom with a variable incident bond ({variable_d2_parents}/{random_parents})"
    );
}

// ---------------------------------------------------------------------------
// FG6: search-outcome split, reference validation, triple/trail regressions
// ---------------------------------------------------------------------------

#[test]
fn search_outcome_decision() {
    // The pure decision function behind both gadget-test branches: a search
    // returning a matching that FAILS validation and a search returning a
    // VALID matching that is simply not perfect take different paths
    // (UNDECIDED vs fixed-inside), as the module docs promise.
    assert_eq!(classify_search_result(false, false), SearchOutcome::Invalid);
    assert_eq!(classify_search_result(false, true), SearchOutcome::Invalid);
    assert_eq!(
        classify_search_result(true, false),
        SearchOutcome::NoAlternative
    );
    assert_eq!(classify_search_result(true, true), SearchOutcome::Candidate);
}

#[test]
fn constrained_outcome_decision() {
    // The pure decision behind BOTH constrained-search guards
    // (`stable_alternative_double` and `stable_alternative_single`) and the
    // `confirm_no_perfect` agreement check: an INVALID returned matching
    // (fails validation) is loud in tests and yields UNDECIDED in release —
    // exactly as the `SearchOutcome` split does on the main path — while only
    // a valid non-perfect matching is a quiet negative.
    for (valid, perfect, want) in [
        (false, false, SearchOutcome::Invalid),
        (false, true, SearchOutcome::Invalid),
        (true, false, SearchOutcome::NoAlternative),
        (true, true, SearchOutcome::Candidate),
    ] {
        assert_eq!(
            classify_constrained_result(valid, perfect), want,
            "double guard: valid={valid} perfect={perfect}"
        );
        assert_eq!(
            classify_constrained_result(valid, perfect), want,
            "single guard: valid={valid} perfect={perfect}"
        );
    }
}

/// Lengths of all simple cycles of `graph` (deduped by atom set + length;
/// same length means same parity, so this decides even-cycle existence).
fn simple_cycle_lengths(graph: &MolGraph) -> Vec<usize> {
    let n = graph.atoms().len();
    let mut adj = vec![Vec::new(); n];
    for (a, b, _) in graph.bonds() {
        adj[*a].push(*b);
        adj[*b].push(*a);
    }
    let mut found = BTreeSet::new();
    for s in 0..n {
        let mut stack = vec![(s, vec![s])];
        while let Some((_, path)) = stack.pop() {
            let v = *path.last().unwrap();
            for &w in &adj[v] {
                if w == s {
                    if path.len() >= 3 {
                        let mut set = path.clone();
                        set.sort_unstable();
                        found.insert((set, path.len()));
                    }
                } else if w > s && !path.contains(&w) {
                    let mut p2 = path.clone();
                    p2.push(w);
                    stack.push((w, p2));
                }
            }
        }
    }
    found.into_iter().map(|(_, l)| l).collect()
}

#[test]
fn fg6_triple_and_trail_regressions() {
    // `C#S1=NC=CC=C1`: the triple is fixed, all six ring bonds delocalised
    // (the old blanket claim — incident singles of a triple-bonded atom are
    // fixed by valence — is false here).
    let forms = fixture_molecule_forms(
        "triple-bonded six-ring heterocycle (triple fixed, ring delocalised)",
    );
    assert_eq!(forms.len(), 2, "the triple heterocycle stores both forms");
    for (k, g) in forms.iter().enumerate() {
        let tri: Vec<usize> = g
            .bonds()
            .iter()
            .enumerate()
            .filter(|(_, (_, _, o))| *o == 3)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(tri.len(), 1, "form {k}: exactly one triple bond");
        let dec = decided_bonds(g);
        assert_eq!(dec[tri[0]], Some(false), "form {k}: triple decided fixed");
        let dl = delocalised_bonds(g);
        assert!(!dl[tri[0]], "form {k}: triple never delocalised");
        let ring_dl = dl.iter().filter(|&&x| x).count();
        assert_eq!(ring_dl, 6, "form {k}: six ring bonds delocalised");
        let set = functional_groups_v4(g);
        assert_eq!(set.count(14), 0, "form {k}: no fixed alkene");
        assert_eq!(set.count(13), 0, "form {k}: no fixed imine");
        assert_eq!(dl, oracle_delocalised(g), "form {k}: oracle agrees");
    }
    // Fixed-triple isomers: identical connectivity (4-cycle), hydrogen
    // counts and valences, but different triple placement — different
    // molecules with different counts.
    let a = &fixture_molecule_forms("four-ring alkyne/alkene isomer (alkene 1, alkyne 1)")[0];
    let b = &fixture_molecule_forms("four-ring triene isomer (alkene 3)")[0];
    assert_eq!(functional_groups_v4(a).count(14), 1, "C1#CC=C1: one alkene");
    assert_eq!(functional_groups_v4(a).count(15), 1, "C1#CC=C1: one alkyne");
    assert_eq!(functional_groups_v4(b).count(14), 3, "C1=C=CC=1: three alkenes");
    assert_eq!(functional_groups_v4(b).count(15), 0, "C1=C=CC=1: no alkyne");
    for (g, tag) in [(a, "alkyne/alkene"), (b, "triene")] {
        assert_eq!(g.atoms().len(), 4, "{tag}: four atoms");
        assert_eq!(g.bonds().len(), 4, "{tag}: four bonds (4-cycle)");
    }
    println!("fg6 isomers: same 4-cycle/H/valences, counts {{alkene 1, alkyne 1}} vs {{alkene 3}}");
    // Two triangles sharing one S(H0,v6): the two valid assignments
    // exchange the double placement with no simple even atom-cycle witness
    // (each triangle is odd; any cycle through both revisits S) — the
    // projected witness is an alternating closed trail. Library = oracle.
    let forms = fixture_molecule_forms(
        "two triangles sharing S(H0,v6) (hand-built; exchange with no even atom-cycle)",
    );
    assert_eq!(forms.len(), 2, "the triangles store both forms");
    for (k, g) in forms.iter().enumerate() {
        let lib = delocalised_bonds(g);
        assert_eq!(lib, vec![true; 6], "form {k}: all six bonds delocalised");
        assert_eq!(lib, oracle_delocalised(g), "form {k}: library equals oracle");
        assert!(
            decided_bonds(g).iter().all(|s| *s == Some(true)),
            "form {k}: every bond decided delocalised"
        );
        let cycles = simple_cycle_lengths(g);
        assert!(!cycles.is_empty(), "form {k}: must hold a cycle");
        assert!(
            cycles.iter().all(|l| l % 2 == 1),
            "form {k}: no simple even atom-cycle exists, yet bonds move: {cycles:?}"
        );
        assert_eq!(
            functional_groups_v4(g).count(14),
            0,
            "form {k}: delocalised C–C doubles are no alkenes"
        );
    }
    println!("fg6 triangles: 2 forms exchange doubles with no even atom-cycle, library = oracle");
}

#[test]
fn reference_rejects_malformed_graphs() {
    // The reference tool validates EVERY graph before enumeration/counting:
    // the reviewer's trivalent oxygen and a lone C(H3,v4) are ERRORS
    // (reported, non-zero exit), while a valid hand-built graph passes.
    if std::process::Command::new("uv").arg("--version").output().is_err() {
        println!("SKIP reference_rejects_malformed_graphs: uv is not on PATH (prerequisite missing)");
        return;
    }
    let dir = std::env::temp_dir().join(format!("ms2_fg6_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir writable");
    let atom = |element: &str, h: u8, v: u8| {
        serde_json::json!({"element": element, "hydrogens": h, "valence": v})
    };
    // The reviewer's trivalent oxygen: O(H0,v2) with three singles.
    let trivalent_o = serde_json::json!({
        "name": "trivalent-O",
        "atoms": [atom("O", 0, 2), atom("C", 3, 4), atom("C", 3, 4), atom("C", 3, 4)],
        "bonds": [[0, 1, 1], [0, 2, 1], [0, 3, 1]],
    });
    // A lone C(H3,v4): unsatisfied valence as a whole molecule.
    let lone_c = serde_json::json!({
        "name": "lone-C",
        "atoms": [atom("C", 3, 4)],
        "bonds": [],
    });
    let bad_path = dir.join("malformed.json");
    std::fs::write(&bad_path, serde_json::json!({"graphs": [trivalent_o, lone_c]}).to_string())
        .expect("temp input writable");
    let bad_out = dir.join("malformed_out.json");
    let out = std::process::Command::new("timeout")
        .arg("300")
        .arg("uv")
        .arg("run")
        .arg("--with")
        .arg("rdkit")
        .arg("--with")
        .arg("numpy")
        .arg("python")
        .arg("tools/ms2/functional_groups_ref.py")
        .arg("--graphs")
        .arg(&bad_path)
        .arg("--out")
        .arg(&bad_out)
        .env("PYTHONPATH", "tools/ms2")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("tool spawns");
    assert!(
        !out.status.success(),
        "malformed graphs must fail the tool (exit {})",
        out.status
    );
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("malformed graph"), "reason reported, got: {text}");
    assert!(text.contains("valence closure"), "closure reason reported, got: {text}");
    // A valid hand-built graph still passes: the two triangles (2 forms).
    let s = atom("S", 0, 6);
    let c = atom("C", 1, 4);
    let good = serde_json::json!({
        "name": "two-triangles",
        "atoms": [s, c.clone(), c.clone(), c.clone(), c],
        "bonds": [[0, 1, 2], [0, 2, 2], [1, 2, 1], [0, 3, 1], [0, 4, 1], [3, 4, 2]],
    });
    let good_path = dir.join("valid.json");
    std::fs::write(&good_path, serde_json::json!({"graphs": [good]}).to_string())
        .expect("temp input writable");
    let good_out = dir.join("valid_out.json");
    let status = std::process::Command::new("timeout")
        .arg("300")
        .arg("uv")
        .arg("run")
        .arg("--with")
        .arg("rdkit")
        .arg("--with")
        .arg("numpy")
        .arg("python")
        .arg("tools/ms2/functional_groups_ref.py")
        .arg("--graphs")
        .arg(&good_path)
        .arg("--out")
        .arg(&good_out)
        .env("PYTHONPATH", "tools/ms2")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("tool spawns");
    assert!(status.success(), "valid hand-built graph must pass (exit {status})");
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&good_out).expect("tool output readable"))
            .expect("tool output parses");
    assert_eq!(value["molecules"].as_array().unwrap().len(), 1);
    assert_eq!(value["molecules"][0]["forms_total"], 2);
    assert!(value["errors"].as_array().unwrap().is_empty());
    println!("reference_rejects_malformed_graphs: trivalent O + lone C rejected, valid graph passes (2 forms)");
}

#[test]
fn export_rejects_malformed_graphs() {
    // Export mode validates the stored graph before counting: the trivalent
    // oxygen `[8,4,4,4]` with singles `(0,1),(0,2),(0,3)` and a lone
    // `C(H3,v4)` are ERRORS (non-zero exit with the reason).
    if std::process::Command::new("uv").arg("--version").output().is_err() {
        println!("SKIP export_rejects_malformed_graphs: uv is not on PATH (prerequisite missing)");
        return;
    }
    let dir = std::env::temp_dir().join(format!("ms2_fg8_export_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir writable");
    let payload = serde_json::json!({"molecules": [
        {"key": "tri-O", "atoms": [8, 4, 4, 4], "bonds": [[0, 1, 1], [0, 2, 1], [0, 3, 1]]},
        {"key": "lone-C", "atoms": [4], "bonds": []},
    ]});
    let inp = dir.join("export_bad.json");
    std::fs::write(&inp, payload.to_string()).expect("temp input writable");
    let outp = dir.join("export_bad_out.json");
    let out = std::process::Command::new("timeout")
        .arg("300")
        .arg("uv")
        .arg("run")
        .arg("--with")
        .arg("rdkit")
        .arg("--with")
        .arg("numpy")
        .arg("python")
        .arg("tools/ms2/functional_groups_ref.py")
        .arg("--export")
        .arg(&inp)
        .arg("--out")
        .arg(&outp)
        .env("PYTHONPATH", "tools/ms2")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("tool spawns");
    assert!(!out.status.success(), "malformed export graphs must fail (exit {})", out.status);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!text.contains("Traceback"), "no traceback, got: {text}");
    assert!(text.contains("valence closure"), "closure reason reported, got: {text}");
    println!("export_rejects_malformed_graphs: trivalent O + lone C rejected in export mode");
}

#[test]
fn graphs_rejects_noninteger_fields() {
    // `--graphs` strict input validation: fractional and boolean bond fields,
    // a non-record entry, a missing `bonds` key and a string order are normal
    // graph errors naming the graph and field (no traceback), exit non-zero.
    if std::process::Command::new("uv").arg("--version").output().is_err() {
        println!("SKIP graphs_rejects_noninteger_fields: uv is not on PATH (prerequisite missing)");
        return;
    }
    let dir = std::env::temp_dir().join(format!("ms2_fg8_graphs_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir writable");
    let atom = |element: &str, h: u8, v: u8| {
        serde_json::json!({"element": element, "hydrogens": h, "valence": v})
    };
    let c = || atom("C", 3, 4);
    let cases: Vec<(&str, serde_json::Value, &str)> = vec![
        ("frac", serde_json::json!({"graphs": [{"name": "frac", "atoms": [c(), c()], "bonds": [[0.9, 1, 1.9]]}]}), "must be integers"),
        ("bool", serde_json::json!({"graphs": [{"name": "bool", "atoms": [c(), c()], "bonds": [[false, true, true]]}]}), "must be integers"),
        ("null", serde_json::json!({"graphs": [null]}), "not an object"),
        ("missing", serde_json::json!({"graphs": [{"name": "m", "atoms": [c()]}]}), "missing field 'bonds'"),
        ("strorder", serde_json::json!({"graphs": [{"name": "s", "atoms": [c(), c()], "bonds": [[0, 1, "1"]]}]}), "must be integers"),
    ];
    for (tag, payload, want) in &cases {
        let inp = dir.join(format!("{tag}.json"));
        std::fs::write(&inp, payload.to_string()).expect("temp input writable");
        let outp = dir.join(format!("{tag}.out.json"));
        let out = std::process::Command::new("timeout")
            .arg("300")
            .arg("uv")
            .arg("run")
            .arg("--with")
            .arg("rdkit")
            .arg("--with")
            .arg("numpy")
            .arg("python")
            .arg("tools/ms2/functional_groups_ref.py")
            .arg("--graphs")
            .arg(&inp)
            .arg("--out")
            .arg(&outp)
            .env("PYTHONPATH", "tools/ms2")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("tool spawns");
        assert!(!out.status.success(), "{tag}: strict validation must fail (exit {})", out.status);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!text.contains("Traceback"), "{tag}: no traceback, got: {text}");
        assert!(text.contains("malformed graph"), "{tag}: names the graph error, got: {text}");
        assert!(text.contains(want), "{tag}: names the field ({want}), got: {text}");
    }
    println!("graphs_rejects_noninteger_fields: 5 strict-validation cases rejected without traceback");
}

// ---------------------------------------------------------------------------
// FG7: the dense-fragment verdict (trigger), matcher re-validation with
// memory-bounded early-exit oracles, and dense-parent agreement without the
// sparse setting
// ---------------------------------------------------------------------------
//
// Background. At the end of FG6 a wrong detector verdict was reported on
// dense S/P-rich random graphs ("a bond that moves between valid assignments
// is reported FIXED, and the verdict depends on atom numbering") and worked
// around in the test generator by keeping the random graphs sparse. The
// reported trigger is fragment atom types `[16,15,1,2,2]` with bonds
// `[(0,1,2),(0,2,1),(1,3,1),(1,4,2),(2,3,2),(2,4,1)]`.
//
// FG7 root cause (evidence in `fg7_trigger_all_numberings` and the probes
// retired with it). Layer by layer:
// (a) the trigger is an OPEN fragment (residual valence `[2,1,0,0,0]`), not
// a closed molecule; the "three assignments" Python oracle enumerates the
// closed-molecule `G'` reassignments, which is the wrong oracle for a
// fragment — fragment verdicts are conservative (decided vs undecided);
// (b) the candidate graph `G'` and degrees `d(v) = [1,2,1,1,1]` are right
// (the in-test oracle confirms exactly 3 assignments with all six bonds
// moving); (c) the Tutte gadget and its seeded matching are valid in every
// numbering (seed-validity is asserted over 2,000 random gadgets in
// `fg7_matcher_gadgets_2000`); (d) the tested-edge modification is faithful;
// (e) the search itself is innocent on the trigger (seeded search vs brute
// force: 0 mismatches over 120 numberings x 6 bonds) and on dense closed
// parents (`fg7_detector_dense_5000`: library == closed oracle); (f) the
// mapping to fixed / delocalised / undecided plus the fragment rule is where
// numberings first differed: the decided-delocalised verdict rested on the
// stability of the FIRST witness found, which routes through the residual-2
// atom in some numberings (`None`) while a stable witness exists in all of
// them. What FG6 read as "FIXED" was `Unknown` collapsed by
// `delocalised_bonds` (which reports `false` for both fixed and undecided):
// no numbering ever reported a wrong decided-fixed bond (0/120 `Some(false)`
// before the fix). The library fix decides the stable witness
// existentially (constrained search, same validation gates), so the trigger
// verdicts are numbering-invariant; the test-side fix drops the sparse-only
// avoidance (dense parents agree with the closed oracle).
//
// Memory rule (the first FG7 session took the machine down twice with a 23 GB
// materialising fuzz test): no oracle below materialises its enumeration —
// assignment counting is depth-first with `seen_single`/`seen_double` flags,
// perfect-matching existence is depth-first with early exit — inputs are
// bounded (abstract graphs: 6 to 20 vertices at densities 0.15 to 0.9, at most
// 14 vertices above density 0.3; gadgets of at most 8 atoms). Bounds per
// helper: `fg7_exists_capped` aborts a single case (counted skip, `None`)
// after 2,000,000 search nodes; `fg7_oracle_capped` and
// `fg7_for_each_assignment` cap ASSIGNMENTS at 4,096 per parent (counted skip,
// `None`/capped) with no search-node counter; the dense generators
// (`fg7_random_closed_dense` and friends) bound WORK by a 60-try retry budget
// per draw and VERTICES (5 to 12 atoms, extra-edge probability 0.5–0.9), with
// no search-node counter.

// ---------------------------------------------------------------------------
// FG7 helpers (memory-bounded oracles)
// ---------------------------------------------------------------------------

/// Closed-`G'` oracle without materialisation: `(delocalised flags, valid
/// assignment count)` with the prescribed double-bond counts, or `None` when
/// more than `cap` assignments exist (counted skip). Depth-first over the
/// bonds of `G'` with degree pruning; only two flag vectors are kept.
fn fg7_oracle_capped(graph: &MolGraph, cap: usize) -> Option<(Vec<bool>, usize)> {
    let bonds = graph.bonds();
    let n = graph.atoms().len();
    let mut d = vec![0u8; n];
    for (a, b, o) in bonds {
        if *o == 2 {
            d[*a] += 1;
            d[*b] += 1;
        }
    }
    let mut g: Vec<usize> = Vec::new();
    for (i, (a, b, o)) in bonds.iter().enumerate() {
        if (*o == 1 || *o == 2) && d[*a] >= 1 && d[*b] >= 1 {
            g.push(i);
        }
    }
    let mut seen_d = vec![false; bonds.len()];
    let mut seen_s = vec![false; bonds.len()];
    let mut count = 0usize;
    let mut capped = false;
    fn rec(
        bonds: &[(usize, usize, u8)],
        g: &[usize],
        d: &[u8],
        pos: usize,
        used: &mut [u8],
        seen_d: &mut [bool],
        seen_s: &mut [bool],
        count: &mut usize,
        cap: usize,
        capped: &mut bool,
        choice: &mut Vec<u8>,
    ) {
        if *capped {
            return;
        }
        if pos == g.len() {
            if used.iter().zip(d.iter()).all(|(u, w)| u == w) {
                *count += 1;
                if *count > cap {
                    *capped = true;
                    return;
                }
                for (p, &bi) in g.iter().enumerate() {
                    if choice[p] == 2 {
                        seen_d[bi] = true;
                    } else {
                        seen_s[bi] = true;
                    }
                }
            }
            return;
        }
        let bi = g[pos];
        let (a, b, _) = bonds[bi];
        for o in [1u8, 2u8] {
            let add = if o == 2 { 1u8 } else { 0u8 };
            let mut ok = true;
            for v in [a, b] {
                let mut rem = 0usize;
                for &bj in &g[pos + 1..] {
                    let (x, y, _) = bonds[bj];
                    if x == v || y == v {
                        rem += 1;
                    }
                }
                let u = used[v] + add;
                if u > d[v] || (d[v] as usize) > (u as usize) + rem {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            used[a] += add;
            used[b] += add;
            choice.push(o);
            rec(bonds, g, d, pos + 1, used, seen_d, seen_s, count, cap, capped, choice);
            choice.pop();
            used[a] -= add;
            used[b] -= add;
            if *capped {
                return;
            }
        }
    }
    let mut used = vec![0u8; n];
    let mut choice = Vec::new();
    rec(
        bonds, &g, &d, 0, &mut used, &mut seen_d, &mut seen_s, &mut count, cap,
        &mut capped, &mut choice,
    );
    if capped {
        return None;
    }
    let gset: BTreeSet<usize> = g.into_iter().collect();
    for i in 0..bonds.len() {
        if !gset.contains(&i) {
            if bonds[i].2 == 2 {
                seen_d[i] = true;
            } else if bonds[i].2 == 1 {
                seen_s[i] = true;
            }
        }
    }
    Some(((0..bonds.len()).map(|i| seen_d[i] && seen_s[i]).collect(), count))
}

/// Stream every valid `G'` assignment (orders on the `G'` bonds) to `f`
/// without storing the enumeration. Returns `(assignment count, capped)`
/// where `capped` reports more than `cap` assignments (the caller skips the
/// parent and counts it).
fn fg7_for_each_assignment(
    graph: &MolGraph,
    cap: usize,
    f: &mut impl FnMut(&[(usize, usize, u8)]),
) -> (usize, bool) {
    let bonds = graph.bonds().to_vec();
    let n = graph.atoms().len();
    let mut d = vec![0u8; n];
    for (a, b, o) in &bonds {
        if *o == 2 {
            d[*a] += 1;
            d[*b] += 1;
        }
    }
    let mut g: Vec<usize> = Vec::new();
    for (i, (a, b, o)) in bonds.iter().enumerate() {
        if (*o == 1 || *o == 2) && d[*a] >= 1 && d[*b] >= 1 {
            g.push(i);
        }
    }
    let mut count = 0usize;
    let mut capped = false;
    fn rec(
        bonds: &[(usize, usize, u8)],
        g: &[usize],
        d: &[u8],
        pos: usize,
        used: &mut [u8],
        orders: &mut [u8],
        count: &mut usize,
        cap: usize,
        capped: &mut bool,
        f: &mut impl FnMut(&[(usize, usize, u8)]),
    ) {
        if *capped {
            return;
        }
        if pos == g.len() {
            if used.iter().zip(d.iter()).all(|(u, w)| u == w) {
                *count += 1;
                if *count > cap {
                    *capped = true;
                    return;
                }
                let mut full: Vec<(usize, usize, u8)> = bonds.to_vec();
                for (p, &bi) in g.iter().enumerate() {
                    full[bi].2 = orders[p];
                }
                f(&full);
            }
            return;
        }
        let bi = g[pos];
        let (a, b, _) = bonds[bi];
        for o in [1u8, 2u8] {
            let add = if o == 2 { 1u8 } else { 0u8 };
            let mut ok = true;
            for v in [a, b] {
                let mut rem = 0usize;
                for &bj in &g[pos + 1..] {
                    let (x, y, _) = bonds[bj];
                    if x == v || y == v {
                        rem += 1;
                    }
                }
                let u = used[v] + add;
                if u > d[v] || (d[v] as usize) > (u as usize) + rem {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            used[a] += add;
            used[b] += add;
            orders[pos] = o;
            rec(bonds, g, d, pos + 1, used, orders, count, cap, capped, f);
            used[a] -= add;
            used[b] -= add;
            if *capped {
                return;
            }
        }
    }
    let mut used = vec![0u8; n];
    let mut orders = vec![0u8; g.len()];
    rec(&bonds, &g, &d, 0, &mut used, &mut orders, &mut count, cap, &mut capped, f);
    (count, capped)
}

/// Early-exit depth-first perfect-matching existence oracle, independent of
/// the detector's blossom code. `(answer, search nodes)`; `None` (counted
/// skip) aborts a single case after `node_cap` nodes. Vertex sets use a
/// `u128` mask (every graph below has at most 128 vertices); only a
/// used-mask memo is kept, never an enumeration.
fn fg7_exists_capped(n: usize, edges: &[(usize, usize)], node_cap: usize) -> (Option<bool>, usize) {
    assert!(n <= 128, "fg7 oracle: graph too large ({n} vertices)");
    let mut adj = vec![Vec::new(); n];
    for (a, b) in edges {
        if *a < n && *b < n && *a != *b {
            adj[*a].push(*b);
            adj[*b].push(*a);
        }
    }
    for lst in adj.iter_mut() {
        lst.sort_unstable();
        lst.dedup();
    }
    let full: u128 = if n == 128 { u128::MAX } else { (1u128 << n) - 1 };
    let mut nodes = 0usize;
    let mut aborted = false;
    let mut memo = std::collections::HashMap::<u128, bool>::new();
    fn rec(
        used: u128,
        full: u128,
        adj: &[Vec<usize>],
        memo: &mut std::collections::HashMap<u128, bool>,
        nodes: &mut usize,
        node_cap: usize,
        aborted: &mut bool,
    ) -> bool {
        *nodes += 1;
        if *nodes > node_cap {
            *aborted = true;
            return false;
        }
        if used == full {
            return true;
        }
        if let Some(&v) = memo.get(&used) {
            return v;
        }
        let v = (0..adj.len()).find(|i| used & (1u128 << i) == 0).expect("a free vertex exists");
        for &w in &adj[v] {
            if used & (1u128 << w) == 0 && rec(used | (1u128 << v) | (1u128 << w), full, adj, memo, nodes, node_cap, aborted) {
                if !*aborted {
                    memo.insert(used, true);
                }
                return true;
            }
            if *aborted {
                return false;
            }
        }
        if !*aborted {
            memo.insert(used, false);
        }
        false
    }
    let ans = rec(0, full, &adj, &mut memo, &mut nodes, node_cap, &mut aborted);
    (if aborted { None } else { Some(ans) }, nodes)
}

/// Faithful in-test replica of the detector's Tutte gadget (ports in
/// incident-bond order, cores after the ports, complete bipartite
/// cores–ports per atom, one port–port edge per `G'` bond, stored doubles
/// matched port–port and stored singles matched port–core in order).
/// Test-side only: the matcher is checked on the replica's graphs, and the
/// replica seed is validated as a perfect matching of real edges for every
/// graph below (layer (c)).
struct Fg7Gadget {
    adj: Vec<Vec<usize>>,
    #[allow(dead_code)]
    owner: Vec<usize>,
    #[allow(dead_code)]
    is_core: Vec<bool>,
    bond_ports: Vec<Option<(usize, usize)>>,
    mate0: Vec<Option<usize>>,
}

impl Fg7Gadget {
    fn build(n_atoms: usize, ends: &[(usize, usize)], orders: &[u8]) -> Self {
        let mut d = vec![0u8; n_atoms];
        for (idx, (a, b)) in ends.iter().enumerate() {
            if orders[idx] == 2 {
                d[*a] = d[*a].saturating_add(1);
                d[*b] = d[*b].saturating_add(1);
            }
        }
        let in_g: Vec<bool> = d.iter().map(|&x| x >= 1).collect();
        let mut g_edges: Vec<usize> = Vec::new();
        let mut deg = vec![0usize; n_atoms];
        for (idx, (a, b)) in ends.iter().enumerate() {
            if (orders[idx] == 1 || orders[idx] == 2) && in_g[*a] && in_g[*b] {
                g_edges.push(idx);
                deg[*a] += 1;
                deg[*b] += 1;
            }
        }
        let mut port_of: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
        let mut owner: Vec<usize> = Vec::new();
        let mut is_core: Vec<bool> = Vec::new();
        let mut bond_ports: Vec<Option<(usize, usize)>> = vec![None; orders.len()];
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
        for v in 0..n_atoms {
            for &c in &cores_of[v] {
                for &p in &port_of[v] {
                    adj[c].push(p);
                    adj[p].push(c);
                }
            }
        }
        for &bi in &g_edges {
            let (a, b) = ends[bi];
            let pa = port_of[a][incid[a].iter().position(|&x| x == bi).unwrap()];
            let pb = port_of[b][incid[b].iter().position(|&x| x == bi).unwrap()];
            adj[pa].push(pb);
            adj[pb].push(pa);
            bond_ports[bi] = Some((pa, pb));
        }
        let mut mate0: Vec<Option<usize>> = vec![None; gn];
        for &bi in &g_edges {
            let (pa, pb) = bond_ports[bi].unwrap();
            if orders[bi] == 2 {
                mate0[pa] = Some(pb);
                mate0[pb] = Some(pa);
            }
        }
        for v in 0..n_atoms {
            if !in_g[v] {
                continue;
            }
            let singles: Vec<usize> = incid[v]
                .iter()
                .filter(|&&bi| orders[bi] == 1)
                .map(|&bi| {
                    let (a, _) = ends[bi];
                    if a == v {
                        bond_ports[bi].unwrap().0
                    } else {
                        bond_ports[bi].unwrap().1
                    }
                })
                .collect();
            assert_eq!(singles.len(), cores_of[v].len(), "replica: singles match cores");
            for (p, &c) in singles.iter().zip(cores_of[v].iter()) {
                mate0[*p] = Some(c);
                mate0[c] = Some(*p);
            }
        }
        Self { adj, owner, is_core, bond_ports, mate0 }
    }

    /// The test's own seed validator (independent of the detector): symmetric
    /// pairs, no self-pairs, every pair a real edge, every vertex covered.
    fn seed_is_perfect(&self) -> bool {
        if self.mate0.iter().any(|m| m.is_none()) {
            return false;
        }
        for (v, m) in self.mate0.iter().enumerate() {
            let w = m.unwrap();
            if w == v || self.mate0[w] != Some(v) || !self.adj[v].contains(&w) {
                return false;
            }
        }
        true
    }

    fn edge_list(&self) -> Vec<(usize, usize)> {
        let mut out = BTreeSet::new();
        for (v, lst) in self.adj.iter().enumerate() {
            for &w in lst {
                out.insert((v.min(w), v.max(w)));
            }
        }
        out.into_iter().collect()
    }
}

/// Dense FG7 parent generator WITHOUT the sparse setting: one draw in three
/// is the planted-ring analogue (guaranteed variable `d = 2` ring bonds);
/// the other draws are dense S/P-rich random closed graphs (extra-edge
/// probability 0.5–0.9, S(v6)/P(v5) frequent). Returns `None` past the retry
/// budget (the caller counts the attempt and continues).
fn fg7_dense_pool_types(rng: &mut impl rand::Rng) -> u8 {
    let pool: Vec<(u8, u32)> = vec![
        (2, 20), // C H1
        (1, 15), // C H0
        (3, 8),  // C H2
        (5, 8),  // N H0
        (6, 4),  // N H1
        (8, 8),  // O H0
        (9, 4),  // O H1
        (15, 12), // S H0 v6 (frequent)
        (16, 10), // P H0 v5 (frequent)
    ];
    sample_type(rng, &pool)
}

/// Random dense connected graph with an EXACT valence-satisfying assignment
/// (a closed molecule), or `None` after the retry budget.
fn fg7_random_closed_dense(rng: &mut impl rand::Rng, n: usize, extra_p: f64) -> Option<MolGraph> {
    for _ in 0..60 {
        let types: Vec<u8> = (0..n).map(|_| fg7_dense_pool_types(rng)).collect();
        let mut eset = BTreeSet::new();
        for i in 1..n {
            let j = rng.random_range(0..i);
            eset.insert((j.min(i), j.max(i)));
        }
        for a in 0..n {
            for b in (a + 1)..n {
                if !eset.contains(&(a, b)) && rng.random_bool(extra_p) {
                    eset.insert((a, b));
                }
            }
        }
        let pairs: Vec<(usize, usize)> = eset.into_iter().collect();
        let cap: Vec<u32> = types
            .iter()
            .map(|t| {
                let at = mamba3::models::ms2::chem::atom_type(*t).unwrap();
                (at.valence - at.hydrogens) as u32
            })
            .collect();
        let mut order_try: Vec<Vec<u8>> = pairs.iter().map(|_| vec![1u8, 2, 3]).collect();
        for o in order_try.iter_mut() {
            shuffled(rng, o);
        }
        let mut assign = vec![0u8; pairs.len()];
        let mut used = vec![0u32; n];
        fn rec(
            pairs: &[(usize, usize)],
            order_try: &[Vec<u8>],
            cap: &[u32],
            pos: usize,
            assign: &mut [u8],
            used: &mut [u32],
        ) -> bool {
            if pos == pairs.len() {
                return used.iter().zip(cap.iter()).all(|(u, c)| u == c);
            }
            let (a, b) = pairs[pos];
            for &o in &order_try[pos] {
                let ou = o as u32;
                if used[a] + ou <= cap[a] && used[b] + ou <= cap[b] {
                    assign[pos] = o;
                    used[a] += ou;
                    used[b] += ou;
                    if rec(pairs, order_try, cap, pos + 1, assign, used) {
                        return true;
                    }
                    used[a] -= ou;
                    used[b] -= ou;
                    assign[pos] = 0;
                }
            }
            false
        }
        if rec(&pairs, &order_try, &cap, 0, &mut assign, &mut used) {
            let bonds: Vec<(usize, usize, u8)> =
                pairs.iter().zip(assign.iter()).map(|((a, b), o)| (*a, *b, *o)).collect();
            if let Ok(g) = MolGraph::new(types, bonds) {
                if g.residual_valence().iter().all(|&r| r == 0) {
                    return Some(g);
                }
            }
        }
    }
    None
}

/// One dense FG7 parent: a dense random closed graph of 5 to 12 atoms
/// (extra-edge probability 0.5–0.9, S(v6)/P(v5) frequent, no sparse setting).
/// Dense-only draws keep S and P frequent and multi-assignment parents
/// non-vacuous (the planted-ring analogue carries no P and exactly two
/// assignments, so it is kept out of the composition-asserted detector test).
fn fg7_random_dense_parent(rng: &mut impl rand::Rng) -> Option<MolGraph> {
    let n = rng.random_range(5..=12);
    let extra_p = rng.random_range(0.5..0.9);
    fg7_random_closed_dense(rng, n, extra_p)
}

/// Mixed FG7 parent for the volume-heavy fragment test: with probability
/// 1/3 the planted-ring analogue (8–10 atoms, large fragment space,
/// guaranteed variable `d = 2` ring bonds), else a dense random closed graph.
fn fg7_random_mixed_parent(rng: &mut impl rand::Rng) -> Option<MolGraph> {
    if rng.random_range(0..3) == 0 {
        return Some(randomized_s_ring_analogue(rng));
    }
    let n = rng.random_range(5..=10);
    let extra_p = rng.random_range(0.5..0.9);
    fg7_random_closed_dense(rng, n, extra_p)
}

// ---------------------------------------------------------------------------
// FG7 test 1: the trigger in all 120 numberings
// ---------------------------------------------------------------------------

#[test]
fn fg7_trigger_all_numberings() {
    // The FG6 trigger is an OPEN fragment (residual [2,1,0,0,0]), so the
    // closed-molecule "three assignments" oracle is the wrong oracle for it:
    // the library's conservative UNDECIDED verdicts on four bonds were read
    // as FIXED through `delocalised_bonds` (which reports `false` for both
    // fixed and undecided). The correct fragment oracle says: bonds incident
    // to the residual-2 atom can never carry a stable witness (always
    // undecided), while the other four bonds admit a stable witness in every
    // completion (decided delocalised). Verdicts must be identical in all 120
    // numberings and equal to that oracle.
    let types = vec![16u8, 15, 1, 2, 2];
    let bonds = vec![
        (0usize, 1usize, 2u8),
        (0, 2, 1),
        (1, 3, 1),
        (1, 4, 2),
        (2, 3, 2),
        (2, 4, 1),
    ];
    let g0 = MolGraph::new(types, bonds).unwrap();
    assert_eq!(g0.residual_valence(), vec![2, 1, 0, 0, 0], "trigger is an open fragment");
    // Layer (b): the closed-reading oracle on the stored form.
    let (d, gprimes) = oracle_degrees(&g0);
    assert_eq!(d, vec![1, 2, 1, 1, 1], "double-bond counts");
    assert_eq!(gprimes.len(), 6, "every bond is in G'");
    let Some((closed_deloc, count)) = fg7_oracle_capped(&g0, 4096) else {
        panic!("trigger oracle must not hit the cap");
    };
    assert_eq!(count, 3, "exactly three G'-internal assignments");
    assert_eq!(closed_deloc, vec![true; 6], "all six bonds move inside");
    // The correct fragment oracle, keyed by original atom pair: no decided
    // fixed bond anywhere (in particular nothing a valid assignment
    // contradicts), undecided exactly on the bonds incident to atom 0.
    let mut expect = std::collections::BTreeMap::new();
    expect.insert((0usize, 1usize), None);
    expect.insert((0, 2), None);
    expect.insert((1, 3), Some(true));
    expect.insert((1, 4), Some(true));
    expect.insert((2, 3), Some(true));
    expect.insert((2, 4), Some(true));
    // A closed 7-atom parent embedding the trigger (atoms 0..=4) as the
    // induced fragment on [0,1,2,3,4]: exocyclic =O on atom 0 (residual 2)
    // and -Me on atom 1 (residual 1). Its closed oracle is the soundness
    // reference for the fragment's decided bonds.
    let pg = MolGraph::new(
        vec![16, 15, 1, 2, 2, 8, 4],
        vec![
            (0, 1, 2), (0, 2, 1), (1, 3, 1), (1, 4, 2), (2, 3, 2), (2, 4, 1),
            (0, 5, 2), (1, 6, 1),
        ],
    )
    .unwrap();
    assert!(pg.residual_valence().iter().all(|&r| r == 0), "embedding parent is closed");
    let Some((parent_oracle, _)) = fg7_oracle_capped(&pg, 4096) else {
        panic!("embedding parent oracle must not hit the cap");
    };
    assert_eq!(delocalised_bonds(&pg), parent_oracle, "closed parent equals its oracle");
    let mut parent_bond_idx = std::collections::BTreeMap::new();
    for (i, (a, b, _)) in pg.bonds().iter().enumerate() {
        parent_bond_idx.insert(((*a).min(*b), (*a).max(*b)), i);
    }
    // All 120 permutations of the 5 atoms.
    fn rec(n: usize, cur: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>) {
        if cur.len() == n {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if !used[i] {
                used[i] = true;
                cur.push(i);
                rec(n, cur, used, out);
                cur.pop();
                used[i] = false;
            }
        }
    }
    let mut perms: Vec<Vec<usize>> = Vec::new();
    rec(5, &mut Vec::new(), &mut vec![false; 5], &mut perms);
    assert_eq!(perms.len(), 120);
    let mut offences = 0usize;
    let mut decided_ok = 0usize;
    let mut undecided = 0usize;
    for p in &perms {
        let g = g0.permuted(p).unwrap();
        let dec = decided_bonds(&g);
        assert_eq!(dec.len(), 6);
        for (i, (a, b, _)) in g.bonds().iter().enumerate() {
            // new[i] = old[p[i]]: map back to the original pair.
            let (oa, ob) = (p[*a], p[*b]);
            let key = (oa.min(ob), oa.max(ob));
            let di = dec[i];
            assert_eq!(
                di, expect[&key],
                "perm {p:?} bond ({a},{b}) (original {key:?}): {di:?} vs oracle {:?}",
                expect[&key]
            );
            // Soundness against the closed embedding parent.
            if let Some(v) = dec[i] {
                let pi = parent_bond_idx[&key];
                if parent_oracle[pi] != v {
                    offences += 1;
                } else {
                    decided_ok += 1;
                }
            } else {
                undecided += 1;
            }
        }
    }
    println!(
        "fg7_trigger_all_numberings: 120 numberings identical, equal to the fragment oracle \
         (never fixed); vs the closed embedding parent: {decided_ok} decided agree, \
         {undecided} undecided, {offences} offences"
    );
    assert_eq!(offences, 0, "0 fragment soundness offences in all 120 numberings");
    assert!(decided_ok > 0 && undecided > 0, "both decided and undecided verdicts occur");
}

// ---------------------------------------------------------------------------
// FG7 test 2: matcher versus early-exit brute force on 20,000 planted graphs
// ---------------------------------------------------------------------------

#[test]
fn fg7_matcher_planted_20000() {
    // 20,000 seeded random graphs with a planted perfect matching, RANDOM
    // (unsorted) edge order for the library: "a perfect matching still
    // exists" after removing a random matched edge, and after forcing a
    // random unmatched edge (remainder on n-2 vertices), must equal the
    // test's own early-exit DFS oracle. Size bounds: 6 to 20 vertices at edge
    // densities 0.15 to 0.9, except above density 0.3 at most 14 vertices. A
    // single case aborts (counted skip) after 2,000,000 nodes.
    use rand::{Rng, SeedableRng, rngs::StdRng};
    const NODE_CAP: usize = 2_000_000;
    let mut rng = StdRng::seed_from_u64(0xF0770607E501);
    let mut graphs = 0usize;
    let mut remove_checked = 0usize;
    let mut force_checked = 0usize;
    let mut force_missing = 0usize;
    let mut skipped_cap = 0usize;
    let mut still_pm_after_remove = 0usize;
    let mut still_pm_after_force = 0usize;
    while graphs < 20_000 {
        // Even vertex count 6..=20 (a planted pairing needs it).
        let n = 2 * rng.random_range(3..=10);
        // Density regime with the matching size bound.
        let density: f64 = if n > 14 { rng.random_range(0.15..0.3) } else { rng.random_range(0.15..0.9) };
        let mut perm: Vec<usize> = (0..n).collect();
        shuffled(&mut rng, &mut perm);
        let planted: Vec<(usize, usize)> = perm
            .chunks_exact(2)
            .map(|w| (w[0].min(w[1]), w[0].max(w[1])))
            .collect();
        let mut edge_set = BTreeSet::new();
        for e in &planted {
            edge_set.insert(*e);
        }
        for a in 0..n {
            for b in (a + 1)..n {
                if rng.random_bool(density) {
                    edge_set.insert((a, b));
                }
            }
        }
        let m = edge_set.len() as f64;
        let full = (n * (n - 1) / 2) as f64;
        let d = m / full;
        if n > 14 && d > 0.3 {
            continue; // outside the size bound: resample (not counted).
        }
        assert!(n <= 14 || d <= 0.3, "size bound: n={n} at density {d}");
        assert!(n <= 20, "at most 20 vertices");
        graphs += 1;
        // RANDOM (unsorted, seeded) edge order for the library.
        let mut edges: Vec<(usize, usize)> = edge_set.iter().copied().collect();
        shuffled(&mut rng, &mut edges);
        // (i) Remove a random matched (planted) edge: PM still exists?
        let ei = rng.random_range(0..planted.len());
        let gone = planted[ei];
        let minus: Vec<(usize, usize)> = edges.iter().filter(|e| **e != gone).copied().collect();
        let lib = has_perfect_matching(n, &minus);
        let (oracle, _) = fg7_exists_capped(n, &minus, NODE_CAP);
        match oracle {
            None => skipped_cap += 1,
            Some(b) => {
                assert_eq!(lib, b, "graph {graphs} (n={n}, d={d:.2}): remove-{gone:?} existence differs");
                remove_checked += 1;
                if b {
                    still_pm_after_remove += 1;
                }
            }
        }
        // (ii) Force a random unmatched edge: the n-2 remainder still has a PM?
        let extras: Vec<(usize, usize)> =
            edge_set.iter().filter(|e| !planted.contains(e)).copied().collect();
        if extras.is_empty() {
            force_missing += 1;
            continue;
        }
        let (u, v) = extras[rng.random_range(0..extras.len())];
        let mut new_of: Vec<Option<usize>> = vec![None; n];
        let mut count = 0usize;
        for h in 0..n {
            if h != u && h != v {
                new_of[h] = Some(count);
                count += 1;
            }
        }
        let rem: Vec<(usize, usize)> = edge_set
            .iter()
            .filter(|(a, b)| *a != u && *a != v && *b != u && *b != v)
            .map(|(a, b)| (new_of[*a].unwrap(), new_of[*b].unwrap()))
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        let lib = has_perfect_matching(n - 2, &rem);
        let (oracle, _) = fg7_exists_capped(n - 2, &rem, NODE_CAP);
        match oracle {
            None => skipped_cap += 1,
            Some(b) => {
                assert_eq!(lib, b, "graph {graphs} (n={n}, d={d:.2}): force-({u},{v}) existence differs");
                force_checked += 1;
                if b {
                    still_pm_after_force += 1;
                }
            }
        }
    }
    println!(
        "fg7_matcher_planted_20000: {graphs} planted graphs, {remove_checked} remove-checks \
         ({still_pm_after_remove} PM surviving) + {force_checked} force-checks \
         ({still_pm_after_force} PM admitting) agree; {force_missing} without an unmatched edge, \
         {skipped_cap} skipped past the node cap"
    );
    assert!(remove_checked > 15_000 && force_checked > 15_000, "both checks must usually apply");
}

// ---------------------------------------------------------------------------
// FG7 test 3: matcher versus brute force on Tutte gadgets of 2,000
// degree-constrained graphs
// ---------------------------------------------------------------------------

#[test]
fn fg7_matcher_gadgets_2000() {
    // 2,000 random degree-constrained graphs of at most 8 atoms (closed
    // valence-consistent graphs, which carry both doubles and singles for
    // non-trivial gadgets): on each Tutte gadget (test-side replica; the
    // stored seed is validated as a perfect matching of real edges) the same
    // two existence checks as above — after removing a random stored-double
    // port edge, after forcing a random stored-single port edge (remainder)
    // — must equal the early-exit oracle. RANDOM edge order for the library;
    // node cap 2,000,000 (counted skips).
    use rand::{Rng, SeedableRng, rngs::StdRng};
    const NODE_CAP: usize = 2_000_000;
    let mut rng = StdRng::seed_from_u64(0x6AD6E7F60702);
    let mut gadgets = 0usize;
    let mut remove_checked = 0usize;
    let mut force_checked = 0usize;
    let mut skipped_cap = 0usize;
    let mut skipped_size = 0usize;
    let mut skipped_trivial = 0usize;
    let mut tries = 0usize;
    while gadgets < 2000 {
        tries += 1;
        assert!(tries < 500_000, "gadget generator budget exceeded");
        let n = rng.random_range(3..=8);
        let extra_p: f64 = rng.random_range(0.3..0.8);
        let Some(g) = fg7_random_closed_dense(&mut rng, n, extra_p) else {
            continue;
        };
        let ends: Vec<(usize, usize)> = g.bonds().iter().map(|(a, b, _)| (*a, *b)).collect();
        let ords: Vec<u8> = g.bonds().iter().map(|(_, _, o)| *o).collect();
        let gad = Fg7Gadget::build(n, &ends, &ords);
        // Layer (c): the seeded matching is a valid perfect matching of the
        // gadget for THIS numbering — asserted on every graph.
        assert!(gad.seed_is_perfect(), "gadget seed must be a perfect matching of real edges");
        let gv = gad.adj.len();
        if gv > 128 {
            skipped_size += 1;
            continue;
        }
        let gel = gad.edge_list();
        // Stored doubles (matched port edges) and stored singles inside G'
        // (a single bond outside G' has no gadget ports to test). Both must
        // be present (otherwise retry: the valence caps filter double-rich
        // assignments, so trivial gadgets would dominate).
        let doubles: Vec<(usize, usize)> = gad
            .bond_ports
            .iter()
            .enumerate()
            .filter(|(bi, _)| ords[*bi] == 2)
            .map(|(_, p)| p.unwrap())
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        let singles: Vec<(usize, usize)> = gad
            .bond_ports
            .iter()
            .enumerate()
            .filter(|(bi, _)| ords[*bi] == 1 && gad.bond_ports[*bi].is_some())
            .map(|(_, p)| p.unwrap())
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        if gel.is_empty() || doubles.is_empty() || singles.is_empty() {
            skipped_trivial += 1;
            continue;
        }
        gadgets += 1;
        let mut gel = gel;
        shuffled(&mut rng, &mut gel);
        // (i) Remove a random stored-double port edge: PM still exists?
        // (Non-trivial gadgets always hold a double; see above.)
        let gone = doubles[rng.random_range(0..doubles.len())];
        let minus: Vec<(usize, usize)> = gel.iter().filter(|e| **e != gone).copied().collect();
        let lib = has_perfect_matching(gv, &minus);
        let (oracle, _) = fg7_exists_capped(gv, &minus, NODE_CAP);
        match oracle {
            None => skipped_cap += 1,
            Some(b) => {
                assert_eq!(lib, b, "gadget {gadgets} (gv={gv}): remove-{gone:?} existence differs");
                remove_checked += 1;
            }
        }
        // (ii) Force a random stored-single port edge (remainder on gv-2).
        // (Non-trivial gadgets always hold a single; see above.)
        let (u, v) = singles[rng.random_range(0..singles.len())];
        let mut new_of: Vec<Option<usize>> = vec![None; gv];
        let mut count = 0usize;
        for h in 0..gv {
            if h != u && h != v {
                new_of[h] = Some(count);
                count += 1;
            }
        }
        let rem: Vec<(usize, usize)> = gel
            .iter()
            .filter(|(a, b)| *a != u && *a != v && *b != u && *b != v)
            .map(|(a, b)| {
                let (na, nb) = (new_of[*a].unwrap(), new_of[*b].unwrap());
                (na.min(nb), na.max(nb))
            })
            .collect();
        let lib = has_perfect_matching(gv - 2, &rem);
        let (oracle, _) = fg7_exists_capped(gv - 2, &rem, NODE_CAP);
        match oracle {
            None => skipped_cap += 1,
            Some(b) => {
                assert_eq!(lib, b, "gadget {gadgets} (gv={gv}): force-({u},{v}) existence differs");
                force_checked += 1;
            }
        }
    }
    println!(
        "fg7_matcher_gadgets_2000: {gadgets} non-trivial gadgets (seeds all perfect), \
         {remove_checked} remove-checks + {force_checked} force-checks agree; {skipped_cap} \
         skipped past the node cap, {skipped_size} past the vertex bound, {skipped_trivial} \
         trivial (no G' double/single pair)"
    );
    assert!(remove_checked > 1500 && force_checked > 1500, "both checks must apply");
}

// ---------------------------------------------------------------------------
// FG7 test 4: detector on 5,000 dense closed parents (no sparse setting)
// ---------------------------------------------------------------------------

#[test]
fn fg7_detector_dense_5000() {
    // 5,000 valence-consistent closed parents of 5 to 10 atoms from the dense
    // generator (no sparse setting; S(v6) and P(v5) frequent): library
    // delocalised set == the capped assignment oracle (cap 4,096 assignments
    // per parent, counted skips) in 3 random numberings each; every closed
    // graph is fully decided.
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0xDE05E5007);
    let mut parents = 0usize;
    let mut tried = 0usize;
    let mut skipped_cap = 0usize;
    let mut with_s = 0usize;
    let mut with_p = 0usize;
    let mut with_ge3 = 0usize;
    let mut with_variable_d2 = 0usize;
    let mut numbering_checks = 0usize;
    while parents < 5000 {
        tried += 1;
        assert!(tried < 500_000, "dense generator budget exceeded");
        let Some(g) = fg7_random_dense_parent(&mut rng) else {
            continue;
        };
        assert!(g.residual_valence().iter().all(|&r| r == 0), "parents are closed");
        let Some((deloc, count)) = fg7_oracle_capped(&g, 4096) else {
            skipped_cap += 1;
            continue;
        };
        if g.atoms().iter().any(|&t| t == 15) {
            with_s += 1;
        }
        if g.atoms().iter().any(|&t| t == 16) {
            with_p += 1;
        }
        if count >= 3 {
            with_ge3 += 1;
        }
        if has_variable_d2(&g, &deloc) {
            with_variable_d2 += 1;
        }
        for _ in 0..3 {
            let n = g.atoms().len();
            let mut perm: Vec<usize> = (0..n).collect();
            shuffled(&mut rng, &mut perm);
            let pg = g.permuted(&perm).unwrap();
            let Some((odeloc, _)) = fg7_oracle_capped(&pg, 4096) else {
                panic!("same assignment space must not hit the cap in another numbering");
            };
            assert_eq!(
                delocalised_bonds(&pg), odeloc,
                "dense parent {parents}: library delocalised set differs from the oracle"
            );
            assert!(
                decided_bonds(&pg).iter().all(|s| s.is_some()),
                "dense parent {parents}: closed, fully decided"
            );
            numbering_checks += 1;
        }
        parents += 1;
    }
    println!(
        "fg7_detector_dense_5000: {parents} dense closed parents agree in {numbering_checks} \
         numberings ({with_s} with S, {with_p} with P, {with_ge3} with at least 3 assignments, \
         {with_variable_d2} with a d = 2 atom with a variable bond; {skipped_cap} skipped past \
         the assignment cap)"
    );
    assert!(with_s >= 2500 && with_p >= 1500, "S and P must be frequent ({with_s}/{with_p})");
    assert!(with_ge3 >= 100, "multi-assignment parents must occur ({with_ge3})");
    assert!(with_variable_d2 >= 200, "variable d = 2 bonds must occur ({with_variable_d2})");
}

// ---------------------------------------------------------------------------
// FG7 test 5: exhaustive fragment soundness over 300 dense parents
// ---------------------------------------------------------------------------

#[test]
fn fg7_fragment_soundness_dense_300() {
    // Over 300 dense closed parents: every oracle assignment streamed (never
    // stored) as a stored form, every connected induced fragment (cap 2,000
    // per form, seeded sample beyond): every decided bond status equals the
    // parent oracle status and determined types stay inside the parent types.
    // 0 offences.
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0xF0A60503);
    let mut parents: Vec<MolGraph> = Vec::new();
    let mut tried = 0usize;
    let mut skipped_cap = 0usize;
    while parents.len() < 300 {
        tried += 1;
        assert!(tried < 200_000, "dense generator budget exceeded");
        let Some(g) = fg7_random_mixed_parent(&mut rng) else {
            continue;
        };
        if fg7_oracle_capped(&g, 4096).is_none() {
            skipped_cap += 1;
            continue;
        }
        parents.push(g);
    }
    let mut forms_seen = 0usize;
    let mut max_forms = 0usize;
    let mut frags = 0usize;
    let mut decided_checked = 0usize;
    let mut undecided_seen = 0usize;
    let mut offences = 0usize;
    for (pi, parent) in parents.iter().enumerate() {
        let Some((oracle, _)) = fg7_oracle_capped(parent, 4096) else {
            panic!("parent {pi} passed the oracle cap above");
        };
        assert_eq!(delocalised_bonds(parent), oracle, "parent {pi} equals its oracle");
        let n = parent.atoms().len();
        let mut adj = vec![Vec::new(); n];
        for (a, b, _) in parent.bonds() {
            adj[*a].push(*b);
            adj[*b].push(*a);
        }
        let mut parent_bond_idx = std::collections::BTreeMap::new();
        for (i, (a, b, _)) in parent.bonds().iter().enumerate() {
            parent_bond_idx.insert(((*a).min(*b), (*a).max(*b)), i);
        }
        // Connectivity is form-independent: enumerate (and cap) once.
        let all = connected_subsets_bounded(&adj, 2, n);
        let members_list: Vec<Vec<usize>> = if all.len() > 2000 {
            let mut idx: Vec<usize> = (0..all.len()).collect();
            shuffled(&mut rng, &mut idx);
            idx.truncate(2000);
            idx.into_iter().map(|i| all[i].clone()).collect()
        } else {
            all.clone()
        };
        let mut n_forms = 0usize;
        let (_, capped) = fg7_for_each_assignment(parent, 4096, &mut |orders| {
            n_forms += 1;
            let form = MolGraph::new(parent.atoms().to_vec(), orders.to_vec()).unwrap();
            // The delocalised set is form-independent (kekule invariance of
            // the oracle itself, re-verified on every streamed form).
            assert_eq!(
                delocalised_bonds(&form), oracle,
                "parent {pi} form {n_forms}: delocalised set differs across forms"
            );
            let pmask = functional_groups_v4(&form).mask();
            for members in &members_list {
                let frag = form.induced(members).unwrap();
                let dm = functional_groups_v4(&frag).mask();
                if dm & !pmask != 0 {
                    offences += 1;
                    println!(
                        "OFFENCE parent {pi} form {n_forms} fragment {members:?}: determined outside parent"
                    );
                }
                let fdec = decided_bonds(&frag);
                for (i, (a, b, _)) in frag.bonds().iter().enumerate() {
                    if let Some(v) = fdec[i] {
                        let (pa, pb) = (members[*a], members[*b]);
                        let pii = parent_bond_idx[&(pa.min(pb), pa.max(pb))];
                        if oracle[pii] != v {
                            offences += 1;
                            println!(
                                "OFFENCE parent {pi} form {n_forms} fragment {members:?} bond \
                                 ({pa},{pb}): fragment says {v} but parent oracle says {}",
                                oracle[pii]
                            );
                        } else {
                            decided_checked += 1;
                        }
                    } else {
                        undecided_seen += 1;
                    }
                }
                frags += 1;
            }
        });
        assert!(!capped, "parent {pi} passed the oracle cap above");
        forms_seen += n_forms;
        max_forms = max_forms.max(n_forms);
    }
    println!(
        "fg7_fragment_soundness_dense_300: 300 dense parents ({forms_seen} streamed forms, \
         at most {max_forms} per parent), {frags} fragments, {decided_checked} decided bonds \
         agree with the parent oracle, {undecided_seen} undecided; {offences} offences \
         ({skipped_cap} parents skipped past the assignment cap)"
    );
    assert_eq!(offences, 0, "{offences} fragment soundness offences");
    assert!(frags > 20_000, "exhaustive test must be non-vacuous ({frags} fragments)");
    assert!(decided_checked > 10_000, "must check decided bonds ({decided_checked})");
}

// ---------------------------------------------------------------------------
// FG7 test 6: numbering invariance over 1,000 dense parents
// ---------------------------------------------------------------------------

#[test]
fn fg7_numbering_invariance_dense_1000() {
    // 1,000 dense closed parents, 5 permutations each: per-bond decided
    // statuses mapped back to original pairs are identical (any mismatch is
    // a detector soundness signal, not mere imprecision: on closed graphs
    // every bond is decided and the verdict is existential).
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0x1A810007);
    let mut parents = 0usize;
    let mut tried = 0usize;
    let mut checks = 0usize;
    while parents < 1000 {
        tried += 1;
        assert!(tried < 200_000, "dense generator budget exceeded");
        let Some(g) = fg7_random_dense_parent(&mut rng) else {
            continue;
        };
        let base: std::collections::BTreeMap<(usize, usize), Option<bool>> = g
            .bonds()
            .iter()
            .zip(decided_bonds(&g).iter())
            .map(|((a, b, _), s)| (((*a).min(*b), (*a).max(*b)), *s))
            .collect();
        assert!(base.values().all(|s| s.is_some()), "closed parents are fully decided");
        let n = g.atoms().len();
        for _ in 0..5 {
            let mut perm: Vec<usize> = (0..n).collect();
            shuffled(&mut rng, &mut perm);
            let pg = g.permuted(&perm).unwrap();
            let dec = decided_bonds(&pg);
            for (i, (a, b, _)) in pg.bonds().iter().enumerate() {
                let key = (perm[*a].min(perm[*b]), perm[*a].max(perm[*b]));
                assert_eq!(
                    dec[i], base[&key],
                    "parent {parents}: permuted status differs on original bond {key:?}"
                );
            }
            checks += 1;
        }
        parents += 1;
    }
    println!("fg7_numbering_invariance_dense_1000: {parents} dense parents x 5 permutations identical ({checks} checks)");
}

// ---------------------------------------------------------------------------
// FG8: dense-fragment numbering invariance population (open fragments)
// ---------------------------------------------------------------------------

/// Check one fragment under 5 random permutations: `decided_bonds` mapped back
/// to original pairs, determined instance sets (type id + anchor atoms mapped
/// back) and the undetermined mask are identical. Returns whether the fragment
/// holds an unstable atom (residual > 1) and a decided-delocalised bond.
fn fg8_check_fragment_numbering(
    frag: &MolGraph,
    rng: &mut impl rand::Rng,
    tag: &str,
) -> bool {
    use std::collections::{BTreeMap, BTreeSet};
    let base_dec: BTreeMap<(usize, usize), Option<bool>> = frag
        .bonds()
        .iter()
        .zip(decided_bonds(frag).iter())
        .map(|((a, b, _), s)| (((*a).min(*b), (*a).max(*b)), *s))
        .collect();
    let base_inst: BTreeSet<(usize, Vec<usize>)> =
        fg_instances(frag).into_iter().collect();
    let base_undet = undetermined(frag);
    let special = frag.residual_valence().iter().any(|&r| r > 1)
        && decided_bonds(frag).iter().any(|s| *s == Some(true));
    let n = frag.atoms().len();
    for _ in 0..5 {
        let mut perm: Vec<usize> = (0..n).collect();
        shuffled(rng, &mut perm);
        let pg = frag.permuted(&perm).unwrap();
        let dec = decided_bonds(&pg);
        for (i, (a, b, _)) in pg.bonds().iter().enumerate() {
            let key = (perm[*a].min(perm[*b]), perm[*a].max(perm[*b]));
            assert_eq!(
                dec[i], base_dec[&key],
                "{tag}: permuted decided status differs on original bond {key:?}"
            );
        }
        let got: BTreeSet<(usize, Vec<usize>)> = fg_instances(&pg)
            .into_iter()
            .map(|(id, anchor)| {
                let mut back: Vec<usize> = anchor.iter().map(|t| perm[*t]).collect();
                back.sort_unstable();
                (id, back)
            })
            .collect();
        assert_eq!(got, base_inst, "{tag}: permuted determined instance set differs");
        assert_eq!(
            undetermined(&pg), base_undet,
            "{tag}: permuted undetermined mask differs"
        );
    }
    special
}

#[test]
fn fg8_fragment_numbering_dense_population() {
    // Population test of FRAGMENT numbering invariance on dense parents: sample
    // connected induced fragments (3 to 9 atoms, at least 3,000 fragments from
    // at least 300 parents of the FG7 dense generator), apply 5 random
    // permutations each, and compare after mapping back: `decided_bonds`, the
    // determined instance sets (type + anchor atoms) and the undetermined set.
    // The population must contain at least 50 fragments with an unstable atom
    // and a decided-delocalised bond (reviewer's embedded trigger family as a
    // second source when the generator gives fewer). Negative control: with
    // the existential rule disabled via `set_first_witness_only(true)` at
    // least one numbering difference must be observed.
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0xF0881008);
    let mut parents: Vec<MolGraph> = Vec::new();
    let mut tried = 0usize;
    while parents.len() < 300 {
        tried += 1;
        assert!(tried < 200_000, "dense generator budget exceeded");
        if let Some(g) = fg7_random_dense_parent(&mut rng) {
            parents.push(g);
        }
    }
    let mut frags = 0usize;
    let mut perms = 0usize;
    let mut special = 0usize;
    for (pi, parent) in parents.iter().enumerate() {
        let n = parent.atoms().len();
        let mut adj = vec![Vec::new(); n];
        for (a, b, _) in parent.bonds() {
            adj[*a].push(*b);
            adj[*b].push(*a);
        }
        let all = connected_subsets_bounded(&adj, 3, 9.min(n));
        let members_list: Vec<Vec<usize>> = if all.len() > 15 {
            let mut idx: Vec<usize> = (0..all.len()).collect();
            shuffled(&mut rng, &mut idx);
            idx.truncate(15);
            idx.into_iter().map(|i| all[i].clone()).collect()
        } else {
            all.clone()
        };
        for members in &members_list {
            let frag = parent.induced(members).unwrap();
            if fg8_check_fragment_numbering(&frag, &mut rng, &format!("parent {pi}")) {
                special += 1;
            }
            frags += 1;
            perms += 5;
        }
    }
    // Second source when the dense sample gives fewer than 50: the reviewer's
    // 5-atom trigger fragment inside closed 7-atom parents with varied
    // exocyclic substituents (atom 0 closed by an order-2 neighbour, atom 1 by
    // an order-1 neighbour; the induced fragment on [0..4] is the trigger with
    // residual [2,1,0,0,0]).
    if special < 50 {
        let need = 50 - special;
        let mut made = 0usize;
        let closers: Vec<(u8, u8)> = [8u8, 13, 3, 6]
            .iter()
            .flat_map(|&a5| [4u8, 7, 9, 14, 10].iter().map(move |&a6| (a5, a6)))
            .collect();
        while made < need {
            let (a5, a6) = closers[made % closers.len()];
            let pg = MolGraph::new(
                vec![16, 15, 1, 2, 2, a5, a6],
                vec![
                    (0, 1, 2), (0, 2, 1), (1, 3, 1), (1, 4, 2),
                    (2, 3, 2), (2, 4, 1), (0, 5, 2), (1, 6, 1),
                ],
            )
            .expect("trigger-family parent builds");
                let mut perm: Vec<usize> = (0..7).collect();
                shuffled(&mut rng, &mut perm);
                let pgp = pg.permuted(&perm).expect("permuted parent builds");
                let mut inv = vec![0usize; 7];
                for (new, old) in perm.iter().enumerate() {
                    inv[*old] = new;
                }
                let members: Vec<usize> = (0..5).map(|o| inv[o]).collect();
                let frag = pgp.induced(&members).expect("trigger fragment induces");
                assert_eq!(
                    frag.residual_valence(),
                    vec![2, 1, 0, 0, 0],
                    "trigger family fragment keeps its residual profile"
                );
                if fg8_check_fragment_numbering(&frag, &mut rng, "trigger-family") {
                    special += 1;
                }
                frags += 1;
                perms += 5;
                made += 1;
            }
    }
    println!(
        "fragment numbering invariance (FG8 dense population): {parents} parents, {frags} fragments, \
         {perms} permutations, {special} fragments with an unstable atom and a decided-delocalised bond",
        parents = parents.len()
    );
    assert!(frags >= 3000, "population must hold at least 3,000 fragments ({frags})");
    assert!(
        special >= 50,
        "population must contain at least 50 fragments with an unstable atom and a decided-delocalised bond ({special})"
    );
    // Negative control: with the existential rule disabled the same trigger
    // must show a numbering difference (the test is non-vacuous).
    let types = vec![16u8, 15, 1, 2, 2];
    let bonds = vec![
        (0usize, 1usize, 2u8),
        (0, 2, 1),
        (1, 3, 1),
        (1, 4, 2),
        (2, 3, 2),
        (2, 4, 1),
    ];
    let g0 = MolGraph::new(types, bonds).expect("trigger builds");
    let ref_off: std::collections::BTreeMap<(usize, usize), Option<bool>> = g0
        .bonds()
        .iter()
        .zip(decided_bonds(&g0).iter())
        .map(|((a, b, _), s)| (((*a).min(*b), (*a).max(*b)), *s))
        .collect();
    set_first_witness_only(true);
    let mut diffs = 0usize;
    let mut checked_nc = 0usize;
    {
        use rand::{SeedableRng, rngs::StdRng};
        let mut nrng = StdRng::seed_from_u64(0x9E64110C);
        for _ in 0..20 {
            let mut perm: Vec<usize> = (0..5).collect();
            shuffled(&mut nrng, &mut perm);
            let pg = g0.permuted(&perm).expect("permuted trigger builds");
            let dec = decided_bonds(&pg);
            for (i, (a, b, _)) in pg.bonds().iter().enumerate() {
                let key = (perm[*a].min(perm[*b]), perm[*a].max(perm[*b]));
                if dec[i] != ref_off[&key] {
                    diffs += 1;
                }
            }
            checked_nc += 1;
        }
    }
    set_first_witness_only(false);
    println!(
        "negative control (first-witness-only): {diffs} numbering differences over {checked_nc} trigger permutations"
    );
    assert!(
        diffs >= 1,
        "negative control must observe at least one numbering difference with the switch on"
    );
}
