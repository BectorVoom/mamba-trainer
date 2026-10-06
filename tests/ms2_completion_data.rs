//! MC2 tests: supervised molecular-completion data (host reference).
//!
//! Hand-built molecules only (from [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)
//! ids) plus the repository fixture `tests/fixtures/ms2/chemistry_v0.json`:
//! no CASMI export is read here. Host only, fast and deterministic; no
//! `backend` feature is needed.

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

use mamba3::models::ms2::chem::CHEMISTRY_VERSION;
use mamba3::models::ms2::completion::contains_pattern;
use mamba3::models::ms2::completion_data::{
    CompletionSet, ExtractionConfig, extract_patterns, same_identity, skeleton,
};
use mamba3::models::ms2::contain::{Containment, contains_induced};
use mamba3::models::ms2::dataset::{ExportFile, ExportMolecule};
use mamba3::models::ms2::grammar::{
    CANONICAL_WORK_LIMIT, Limits, TraceState, canonical_trace, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;

const WORK: usize = 100_000;

/// Ethanol `[C(H3), C(H2), O(H1)]`.
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Kekulized toluene analogue: a six ring with alternating single/double
/// bonds and a methyl on atom 0. Ring atoms are `C(H1)` (id 2) except the
/// ipso `C(H0)` (id 1); the methyl is `C(H3)` (id 4).
fn toluene_like() -> MolGraph {
    MolGraph::new(
        vec![1, 2, 2, 2, 2, 2, 4],
        vec![
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 0, 2),
            (0, 6, 1),
        ],
    )
    .unwrap()
}

/// Cyclohexane: six `C(H2)` (id 3) in a single-bond ring.
fn cyclohexane() -> MolGraph {
    MolGraph::new(
        vec![3, 3, 3, 3, 3, 3],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 0, 1),
        ],
    )
    .unwrap()
}

/// Branched amine: three `C(H3)` (id 4) and one `N(H2)` (id 7) on a central
/// `C(H0)` (id 1).
fn branched_amine() -> MolGraph {
    MolGraph::new(
        vec![4, 4, 4, 1, 7],
        vec![(3, 0, 1), (3, 1, 1), (3, 2, 1), (3, 4, 1)],
    )
    .unwrap()
}

/// Eight-carbon chain (8 atoms): ends `C(H3)` (id 4), middles `C(H2)` (id 3).
fn chain8() -> MolGraph {
    MolGraph::new(
        vec![4, 3, 3, 3, 3, 3, 3, 4],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 6, 1),
            (6, 7, 1),
        ],
    )
    .unwrap()
}

/// Kekulé form A of an ortho-xylene-like ring: `C(H0)` (id 1) at 0 and 1
/// with methyls (`C(H3)`, id 4) at 6 and 7, `C(H1)` (id 2) elsewhere. The
/// adjacent methyls break every ring symmetry that would map one Kekulé
/// shift onto the other (a plain even ring's shifts are always related by a
/// reflection), so the two forms below are genuinely non-isomorphic while
/// sharing one skeleton.
fn kekule_a() -> MolGraph {
    MolGraph::new(
        vec![1, 1, 2, 2, 2, 2, 4, 4],
        vec![
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 0, 2),
            (0, 6, 1),
            (1, 7, 1),
        ],
    )
    .unwrap()
}

/// Kekulé form B: the ring double bonds shifted by one position. Same atoms
/// and skeleton as [`kekule_a`], different bond orders, not isomorphic to
/// it (checked by exhaustive search in development).
fn kekule_b() -> MolGraph {
    MolGraph::new(
        vec![1, 1, 2, 2, 2, 2, 4, 4],
        vec![
            (0, 1, 2),
            (1, 2, 1),
            (2, 3, 2),
            (3, 4, 1),
            (4, 5, 2),
            (5, 0, 1),
            (0, 6, 1),
            (1, 7, 1),
        ],
    )
    .unwrap()
}

/// Five atoms with two ring closures: a 5-cycle plus the (0,2) chord, all
/// `C(H1)` (id 2, capacity 3).
fn two_closures() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 2, 2, 2],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 0, 1),
            (0, 2, 1),
        ],
    )
    .unwrap()
}

/// Serialize extracted patterns for equality across draws.
fn pattern_key(
    graph: &MolGraph,
    parents: &[usize],
) -> (Vec<u8>, Vec<(usize, usize, u8)>, Vec<usize>) {
    (
        graph.atoms().to_vec(),
        graph.bonds().to_vec(),
        parents.to_vec(),
    )
}

#[test]
fn extraction_is_deterministic_and_valid() {
    let config = ExtractionConfig::default();
    let ring = toluene_like();
    // Same arguments give the same patterns.
    let first = extract_patterns(&ring, &config, 7, "mol", 0).unwrap();
    let again = extract_patterns(&ring, &config, 7, "mol", 0).unwrap();
    assert_eq!(first.len(), again.len(), "same draw, same count");
    for (a, b) in first.iter().zip(again.iter()) {
        assert_eq!(
            pattern_key(&a.graph, &a.parent_atoms),
            pattern_key(&b.graph, &b.parent_atoms),
            "same draw, same patterns"
        );
    }
    assert!(!first.is_empty(), "the default config extracts patterns");
    // Over 32 variants each differing dimension takes at least two distinct
    // values: the seed mix spreads nearby inputs.
    let mut seen_draws = HashSet::new();
    for draw in 0..32u64 {
        let out = extract_patterns(&ring, &config, 7, "mol", draw).unwrap();
        seen_draws.insert(
            out.iter()
                .map(|p| pattern_key(&p.graph, &p.parent_atoms))
                .collect::<Vec<_>>(),
        );
    }
    assert!(
        seen_draws.len() >= 2,
        "32 draws give at least two distinct results"
    );
    let mut seen_keys = HashSet::new();
    for k in 0..32u64 {
        let out = extract_patterns(&ring, &config, 7, &format!("mol-{k}"), 0).unwrap();
        seen_keys.insert(
            out.iter()
                .map(|p| pattern_key(&p.graph, &p.parent_atoms))
                .collect::<Vec<_>>(),
        );
    }
    assert!(
        seen_keys.len() >= 2,
        "32 keys give at least two distinct results"
    );
    let mut seen_seeds = HashSet::new();
    for seed in 0..32u64 {
        let out = extract_patterns(&ring, &config, seed, "mol", 0).unwrap();
        seen_seeds.insert(
            out.iter()
                .map(|p| pattern_key(&p.graph, &p.parent_atoms))
                .collect::<Vec<_>>(),
        );
    }
    assert!(
        seen_seeds.len() >= 2,
        "32 seeds give at least two distinct results"
    );
    // Every pattern of several draws is valid.
    for draw in 0..8u64 {
        let out = extract_patterns(&ring, &config, 7, "mol", draw).unwrap();
        assert!(
            out.len() <= config.max_patterns,
            "draw {draw}: at most max_patterns"
        );
        let total: usize = out.iter().map(|p| p.graph.atoms().len()).sum();
        assert!(
            total <= config.max_total_atoms,
            "draw {draw}: total {total} within max_total_atoms"
        );
        for p in &out {
            let n = p.graph.atoms().len();
            assert!(
                (1..=config.max_pattern_atoms).contains(&n),
                "draw {draw}: pattern size {n} in 1..=max_pattern_atoms"
            );
            assert!(p.graph.is_connected(), "draw {draw}: pattern connected");
            assert_eq!(
                n,
                p.parent_atoms.len(),
                "draw {draw}: parent_atoms names every pattern atom"
            );
            // The pattern is the induced subgraph on its parent atoms, with
            // the parent's atom types.
            let induced = ring.induced(&p.parent_atoms).unwrap();
            assert_eq!(induced.atoms(), p.graph.atoms(), "draw {draw}: types kept");
            assert_eq!(
                induced.bonds(),
                p.graph.bonds(),
                "draw {draw}: induced bonds"
            );
            assert_eq!(
                contains_pattern(&ring, &p.graph, WORK),
                Containment::Contained,
                "draw {draw}: non-induced containment"
            );
            assert_eq!(
                contains_induced(&ring, &p.graph, WORK),
                Containment::Contained,
                "draw {draw}: induced containment"
            );
        }
    }
}

#[test]
fn pattern_order_carries_no_parent_order() {
    let config = ExtractionConfig::default();
    let parent = chain8();
    assert!(parent.atoms().len() >= 8);
    // The shuffle moves the first pattern atom off the smallest parent index
    // on some draw: pattern order carries no parent-order information.
    let mut ever_shuffled = false;
    for draw in 0..200u64 {
        let out = extract_patterns(&parent, &config, 1, "chain", draw).unwrap();
        assert!(!out.is_empty(), "draw {draw}: a first pattern exists");
        let atoms = &out[0].parent_atoms;
        let min = atoms.iter().min().expect("pattern non-empty");
        if &atoms[0] != min {
            ever_shuffled = true;
        }
    }
    assert!(
        ever_shuffled,
        "over 200 draws the first pattern atom is sometimes not the smallest parent index"
    );
    // Extraction from a relabeled parent still cuts substructures of the
    // original: each pattern embeds in it. Exact equality across relabelings
    // is not required (the generator indexes atoms).
    let n = parent.atoms().len();
    let perm: Vec<usize> = (0..n).rev().collect();
    let relabeled = parent.permuted(&perm).unwrap();
    for draw in 0..20u64 {
        let out = extract_patterns(&relabeled, &config, 3, "chain", draw).unwrap();
        for p in &out {
            assert_eq!(
                contains_pattern(&parent, &p.graph, WORK),
                Containment::Contained,
                "draw {draw}: relabeled pattern embeds in the original"
            );
        }
    }
}

#[test]
fn config_validation() {
    let good = ExtractionConfig::default();
    assert!(good.validate().is_ok());
    let bad = |config: ExtractionConfig| {
        let err = config.validate().expect_err("bound must be rejected");
        assert!(
            matches!(err, mamba3::error::Error::Config(_)),
            "bound rejection is Error::Config: {err}"
        );
    };
    // min_patterns > max_patterns.
    bad(ExtractionConfig {
        min_patterns: 3,
        max_patterns: 2,
        ..ExtractionConfig::default()
    });
    // max_patterns > 8 (the request layer's substructure cap).
    bad(ExtractionConfig {
        max_patterns: 9,
        min_patterns: 9,
        ..ExtractionConfig::default()
    });
    // min_pattern_atoms < 1.
    bad(ExtractionConfig {
        min_pattern_atoms: 0,
        ..ExtractionConfig::default()
    });
    // min_pattern_atoms > max_pattern_atoms.
    bad(ExtractionConfig {
        min_pattern_atoms: 9,
        max_pattern_atoms: 8,
        max_total_atoms: 24,
        ..ExtractionConfig::default()
    });
    // max_pattern_atoms > max_total_atoms.
    bad(ExtractionConfig {
        max_pattern_atoms: 9,
        max_total_atoms: 8,
        ..ExtractionConfig::default()
    });
    // max_total_atoms > 24 (the request layer's pattern-atom cap).
    bad(ExtractionConfig {
        max_total_atoms: 25,
        ..ExtractionConfig::default()
    });
    // max_patterns == 0 is the valid no-substructure control.
    let none = ExtractionConfig {
        min_patterns: 0,
        max_patterns: 0,
        ..ExtractionConfig::default()
    };
    assert!(none.validate().is_ok());
    assert!(
        extract_patterns(&ethanol(), &none, 1, "ethanol", 0)
            .unwrap()
            .is_empty(),
        "max_patterns == 0 yields no patterns"
    );
}

/// One export molecule row with no spectra.
fn export_molecule(key: &str, graph: &MolGraph, identity_group: u64) -> ExportMolecule {
    ExportMolecule {
        key: key.to_string(),
        identity_group,
        fold_identity: 0,
        atoms: graph.atoms().to_vec(),
        bonds: graph.bonds().to_vec(),
        spectra: Vec::new(),
    }
}

fn export_file(molecules: Vec<ExportMolecule>) -> ExportFile {
    ExportFile {
        schema_version: 1,
        chemistry: CHEMISTRY_VERSION.to_string(),
        rdkit: "test".to_string(),
        source: "test".to_string(),
        seed: 0,
        n_raw: 0,
        spectra_per_molecule: 0,
        skipped_spectra: Default::default(),
        subset: "test".to_string(),
        molecules,
    }
}

#[test]
fn set_from_export_keeps_complete_canonical_examples() {
    let limits = Limits::new(6, 1).unwrap();
    let ethanol_graph = ethanol();
    // The same molecule under a different key and atom order: the second
    // copy is a duplicate identity.
    let permuted = ethanol_graph.permuted(&[2, 1, 0]).unwrap();
    let file = export_file(vec![
        export_molecule("ethanol", &ethanol_graph, 11),
        export_molecule("amine", &branched_amine(), 12),
        export_molecule("ethanol-copy", &permuted, 13),
        export_molecule("too-many-atoms", &chain8(), 14),
        export_molecule("too-many-closures", &two_closures(), 15),
        export_molecule(
            "disconnected",
            &MolGraph::new(vec![4, 3, 9], Vec::new()).unwrap(),
            16,
        ),
    ]);
    let set = CompletionSet::from_export(&file, limits, WORK).unwrap();
    assert_eq!(set.examples.len(), 2, "two valid molecules kept");
    assert_eq!(
        set.examples[0].key, "ethanol",
        "file order kept, first wins"
    );
    assert_eq!(set.examples[1].key, "amine");
    for (reason, want) in [
        ("duplicate_identity", 1u64),
        ("too_many_atoms", 1),
        ("too_many_closures", 1),
        ("not_connected", 1),
    ] {
        assert_eq!(
            set.skipped.get(reason).copied().unwrap_or(0),
            want,
            "skipped[{reason}]"
        );
    }
    assert_eq!(set.skipped.len(), 4, "no other skip reasons");
    for example in &set.examples {
        // Each kept example replays under the exact rule to a stopped,
        // complete state with the stated composition.
        let end = replay_exact(&example.trace, limits, example.composition).unwrap();
        assert!(end.stopped(), "{}: replay ends stopped", example.key);
        assert!(end.is_complete(), "{}: replay is complete", example.key);
        let replayed = end.graph().unwrap();
        assert!(replayed.is_connected(), "{}: replay connected", example.key);
        assert_eq!(
            replayed.composition(),
            example.composition,
            "{}: replay composition",
            example.key
        );
        assert_eq!(
            replayed.atoms(),
            example.target.atoms(),
            "{}: target is the replayed graph",
            example.key
        );
        assert_eq!(
            replayed.bonds(),
            example.target.bonds(),
            "{}: target bonds match",
            example.key
        );
        // Token-by-token legality under the exact grammar (what
        // replay_exact checks).
        let mut state = TraceState::new_exact(limits, example.composition);
        for token in &example.trace {
            assert!(
                state.is_legal(*token),
                "{}: token legal under new_exact",
                example.key
            );
            state.apply(*token).unwrap();
        }
    }
    // Two Kekulé forms of one ring: different traces, one skeleton trace.
    // They hold 8 atoms, so they go in their own file under wider limits.
    let kek_limits = Limits::new(8, 1).unwrap();
    let kek_file = export_file(vec![
        export_molecule("kek-a", &kekule_a(), 21),
        export_molecule("kek-b", &kekule_b(), 22),
    ]);
    let kek_set = CompletionSet::from_export(&kek_file, kek_limits, WORK).unwrap();
    assert_eq!(kek_set.examples.len(), 2, "both Kekulé forms kept");
    assert_ne!(
        kek_set.examples[0].trace, kek_set.examples[1].trace,
        "bond-order isomers have different canonical traces"
    );
    assert!(
        !kek_set.examples[0].skeleton_trace.is_empty(),
        "skeleton canonicalization succeeds"
    );
    assert_eq!(
        kek_set.examples[0].skeleton_trace, kek_set.examples[1].skeleton_trace,
        "Kekulé forms share a skeleton trace"
    );
    // The skeleton is the same atoms and bonds with unit orders.
    let flat = skeleton(&kekule_a()).unwrap();
    assert_eq!(flat.atoms(), kekule_a().atoms());
    assert!(
        flat.bonds().iter().all(|(_, _, o)| *o == 1),
        "skeleton sets every bond order to 1"
    );
}

#[test]
fn overlap_counts_strict_and_skeleton() {
    // Eight atoms fit alongside ethanol under (8, 1).
    let limits = Limits::new(8, 1).unwrap();
    let set_a = CompletionSet::from_export(
        &export_file(vec![
            export_molecule("a-ethanol", &ethanol(), 31),
            export_molecule("a-kek", &kekule_a(), 32),
        ]),
        limits,
        WORK,
    )
    .unwrap();
    let set_b = CompletionSet::from_export(
        &export_file(vec![
            export_molecule("b-ethanol", &ethanol(), 33),
            export_molecule("b-kek", &kekule_b(), 34),
        ]),
        limits,
        WORK,
    )
    .unwrap();
    assert_eq!(set_a.examples.len(), 2);
    assert_eq!(set_b.examples.len(), 2);
    // One molecule shared exactly (ethanol), one only up to Kekulé form.
    assert_eq!(set_a.overlap(&set_b), (1, 2));
}

#[test]
fn contains_pattern_is_non_induced() {
    let triangle = cyclohexane_like_triangle();
    // A C–C–C path embeds non-induced (the target may hold more bonds) but
    // not induced (the extra triangle bond is a mismatch there).
    let path = MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    assert_eq!(
        contains_pattern(&triangle, &path, WORK),
        Containment::Contained,
        "path embeds non-induced in the triangle"
    );
    assert_eq!(
        contains_induced(&triangle, &path, WORK),
        Containment::NotContained,
        "the extra triangle bond breaks the induced match"
    );
    // A double bond where the target has only singles never matches.
    let double_path = MolGraph::new(vec![2, 2, 3], vec![(0, 1, 2), (1, 2, 1)]).unwrap();
    assert_eq!(
        contains_pattern(&triangle, &double_path, WORK),
        Containment::NotContained,
        "wrong bond order is not contained"
    );
    // An atom type the target lacks never matches.
    let nitrogen = MolGraph::new(vec![6], Vec::new()).unwrap();
    assert_eq!(
        contains_pattern(&triangle, &nitrogen, WORK),
        Containment::NotContained,
        "wrong atom type is not contained"
    );
    // A node limit of 1 on a non-trivial pair spends the budget first.
    assert_eq!(
        contains_pattern(&cyclohexane(), &path, 1),
        Containment::WorkLimit,
        "node_limit 1 is exhausted before any embedding finishes"
    );
}

/// Cyclopropane: three `C(H2)` (id 3) in a triangle.
fn cyclohexane_like_triangle() -> MolGraph {
    MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (0, 2, 1)]).unwrap()
}

#[test]
fn same_identity_agrees_with_canonical_traces() {
    let limits = Limits::new(16, 4).unwrap();
    let mut graphs = vec![ethanol(), toluene_like(), cyclohexane(), branched_amine()];
    // One fixed non-identity relabeling (reversal) of each molecule.
    for i in 0..4 {
        let n = graphs[i].atoms().len();
        let perm: Vec<usize> = (0..n).rev().collect();
        graphs.push(graphs[i].permuted(&perm).unwrap());
    }
    let traces: Vec<_> = graphs
        .iter()
        .map(|g| canonical_trace(g, limits, WORK).unwrap().trace)
        .collect();
    for (i, a) in graphs.iter().enumerate() {
        for (j, b) in graphs.iter().enumerate() {
            let want = traces[i] == traces[j];
            assert_eq!(
                same_identity(a, b, WORK),
                Some(want),
                "pair ({i},{j}): same_identity matches canonical-trace equality"
            );
        }
    }
    // The two Kekulé forms are distinct bond-order isomers.
    assert_eq!(
        same_identity(&kekule_a(), &kekule_b(), WORK),
        Some(false),
        "Kekulé forms are not the same identity"
    );
    assert_eq!(
        same_identity(
            &kekule_a(),
            &kekule_a().permuted(&[7, 6, 5, 4, 3, 2, 1, 0]).unwrap(),
            WORK
        ),
        Some(true),
        "a Kekulé form matches its relabeling"
    );
}

/// Pinned extraction outputs: three fixed `(seed, key, draw)` triples give
/// these exact pattern atom lists on the toluene-like ring under the default
/// config. Guards the generator refactor (shared `SplitMix64`): any stream
/// change fails here.
#[test]
fn extraction_pinned_values() {
    let config = ExtractionConfig::default();
    let parent = toluene_like();
    let atoms_of = |seed: u64, key: &str, draw: u64| -> Vec<Vec<usize>> {
        extract_patterns(&parent, &config, seed, key, draw)
            .unwrap()
            .iter()
            .map(|p| p.parent_atoms.clone())
            .collect()
    };
    assert_eq!(atoms_of(7, "mol", 0), vec![vec![4usize, 3]]);
    assert_eq!(
        atoms_of(7, "mol", 1),
        vec![vec![3usize, 1, 4, 2], vec![1, 6, 5, 4, 0, 3]]
    );
    assert_eq!(atoms_of(42, "chain", 3), vec![vec![0usize, 6, 2, 1]]);
}

#[test]
fn skip_reason_precedence() {
    // A molecule that is both disconnected and too large is `not_connected`:
    // the connectivity check runs first.
    let limits = Limits::new(6, 1).unwrap();
    let disconnected_big = MolGraph::new(
        vec![4, 3, 9, 4, 3, 9, 4],
        vec![(0, 1, 1), (1, 2, 1), (3, 4, 1), (4, 5, 1)],
    )
    .unwrap();
    assert!(!disconnected_big.is_connected());
    assert!(disconnected_big.atoms().len() > limits.max_atoms());
    let set = CompletionSet::from_export(
        &export_file(vec![export_molecule("big-split", &disconnected_big, 1)]),
        limits,
        WORK,
    )
    .unwrap();
    assert!(set.examples.is_empty());
    assert_eq!(set.skipped.get("not_connected").copied().unwrap_or(0), 1);
    // A molecule too large in atoms and closures is `too_many_atoms`: the
    // atom check precedes the closure check.
    let big_closed = MolGraph::new(
        vec![2, 2, 2, 2, 2, 4, 4],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 0, 1),
            (0, 2, 1),
            (3, 5, 1),
            (4, 6, 1),
        ],
    )
    .unwrap();
    assert!(big_closed.atoms().len() > limits.max_atoms());
    assert!(big_closed.ring_closures() > limits.max_closures());
    let set = CompletionSet::from_export(
        &export_file(vec![export_molecule("big-closed", &big_closed, 2)]),
        limits,
        WORK,
    )
    .unwrap();
    assert!(set.examples.is_empty());
    assert_eq!(set.skipped.get("too_many_atoms").copied().unwrap_or(0), 1);
    // A work limit of 1 spends the canonicalization budget on any real
    // molecule: a counted skip, never an error.
    let wide = Limits::new(16, 4).unwrap();
    let set = CompletionSet::from_export(
        &export_file(vec![export_molecule("ethanol", &ethanol(), 3)]),
        wide,
        1,
    )
    .unwrap();
    assert!(set.examples.is_empty());
    assert_eq!(
        set.skipped
            .get("canonicalization_work_limit")
            .copied()
            .unwrap_or(0),
        1
    );
}

#[test]
fn extraction_capacity_edges() {
    // A zero total pattern budget holds no pattern, on any parent.
    let none = ExtractionConfig {
        min_patterns: 0,
        max_patterns: 0,
        ..ExtractionConfig::default()
    };
    assert!(
        extract_patterns(&chain8(), &none, 5, "edge", 0)
            .unwrap()
            .is_empty()
    );
    // A parent smaller than `min_pattern_atoms` holds no pattern.
    let lone = MolGraph::new(vec![4], vec![]).unwrap();
    assert!(
        extract_patterns(&lone, &ExtractionConfig::default(), 5, "edge", 0)
            .unwrap()
            .is_empty()
    );
    // `max_patterns = 8`, `max_total_atoms = 8`: never more than 8 atoms in
    // total, over 100 draws.
    let capped = ExtractionConfig {
        min_patterns: 1,
        max_patterns: 8,
        min_pattern_atoms: 2,
        max_pattern_atoms: 8,
        max_total_atoms: 8,
    };
    capped.validate().unwrap();
    for draw in 0..100u64 {
        let out = extract_patterns(&chain8(), &capped, 5, "edge", draw).unwrap();
        assert!(!out.is_empty(), "draw {draw}: the first slot always fits");
        assert!(out.len() <= 8, "draw {draw}: at most 8 patterns");
        let total: usize = out.iter().map(|p| p.graph.atoms().len()).sum();
        assert!(total <= 8, "draw {draw}: total {total} within 8");
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json")
}

#[test]
fn kept_examples_follow_the_closed_prefix_rule() {
    // Every kept example's trace is legal token by token under
    // `TraceState::new_exact`: training would silently lose any molecule
    // whose canonical trace the exact grammar forbade, so this asserts the
    // property explicitly on the fixture molecules.
    let text = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    let molecules = fixture["molecules"].as_array().expect("molecules");
    let limits = Limits::new(32, 6).unwrap();
    let mut export_molecules = Vec::with_capacity(molecules.len());
    let mut traceable = 0usize;
    for m in molecules {
        let atoms: Vec<u8> = m["atoms"]
            .as_array()
            .expect("atoms")
            .iter()
            .map(|a| a.as_u64().expect("atom") as u8)
            .collect();
        let bonds: Vec<(usize, usize, u8)> = m["bonds"]
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
        let name = m["name"].as_str().expect("name");
        export_molecules.push(export_molecule(
            name,
            &MolGraph::new(atoms, bonds).unwrap(),
            0,
        ));
        traceable += 1;
    }
    assert!(traceable >= 20, "the fixture holds molecules");
    let set =
        CompletionSet::from_export(&export_file(export_molecules), limits, CANONICAL_WORK_LIMIT)
            .unwrap();
    assert!(
        set.examples.len() >= 10,
        "most fixture molecules are kept: {} kept",
        set.examples.len()
    );
    let mut seen = BTreeSet::new();
    for example in &set.examples {
        seen.insert(example.key.clone());
        let mut state = TraceState::new_exact(limits, example.composition);
        for (i, token) in example.trace.iter().enumerate() {
            assert!(
                state.is_legal(*token),
                "{} token {i}: legal under new_exact",
                example.key
            );
            state.apply(*token).unwrap();
        }
        assert!(state.stopped(), "{}: ends stopped", example.key);
        assert!(state.is_complete(), "{}: ends complete", example.key);
    }
    assert_eq!(seen.len(), set.examples.len(), "kept keys are distinct");
}

/// The enumerating DFS visits every complete mapping: cyclopropane's
/// triangle has 6 automorphisms (S3), ethanol's typed chain has 1, and a
/// non-isomorphic pair has none. `same_identity` keeps its first-match
/// behaviour on the same inputs.
#[test]
fn enumerate_isomorphisms_visits_every_mapping() {
    use mamba3::models::ms2::completion_data::enumerate_isomorphisms;
    let triangle = cyclohexane_like_triangle();
    let out = enumerate_isomorphisms(&triangle, &triangle, WORK, 100);
    assert!(!out.over_budget && !out.truncated);
    assert_eq!(out.maps.len(), 6, "triangle automorphisms");
    let mut sorted = out.maps.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 6, "all maps distinct");
    // The identity is always first (ascending image trials).
    assert_eq!(out.maps[0], vec![0, 1, 2]);
    for map in &out.maps {
        let image: MolGraph = triangle
            .permuted(&invert(map))
            .expect("automorphisms permute validly");
        assert_eq!(
            same_identity(&triangle, &image, WORK),
            Some(true),
            "every enumerated map is an isomorphism"
        );
    }
    let ethanol_out = enumerate_isomorphisms(&ethanol(), &ethanol(), WORK, 100);
    assert_eq!(ethanol_out.maps.len(), 1, "typed chain has one map");
    assert_eq!(ethanol_out.maps[0], vec![0, 1, 2]);
    let none = enumerate_isomorphisms(&ethanol(), &triangle, WORK, 100);
    assert!(none.maps.is_empty() && !none.over_budget && !none.truncated);
    // The mapping cap truncates instead of guessing.
    let capped = enumerate_isomorphisms(&triangle, &triangle, WORK, 4);
    assert!(capped.truncated);
    assert_eq!(capped.maps.len(), 4);
    // A work limit of 0 spends the budget on the first tried image.
    let rushed = enumerate_isomorphisms(&triangle, &triangle, 0, 100);
    assert!(rushed.over_budget);
}

/// Inverse permutation: `invert(p)[p[i]] == i`.
fn invert(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}
