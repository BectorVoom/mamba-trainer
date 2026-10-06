//! Host tests for stereo perception and enumeration
//! (`stereo-perception-v2`, `src/models/ms2/stereo.rs`).
//!
//! Hand-built graphs only (atom type ids of `chem::ATOM_TYPES`): no CASMI
//! export is read here. Host only, fast and deterministic; no `backend`
//! feature is needed.

use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::stereo::{
    Ligand, Resolution, StereoElement, StereoLimits, StereoReport, equivalent, perceive,
};

/// Limits with full isomer expansion for the table molecules.
fn expanded() -> StereoLimits {
    StereoLimits {
        max_elements: 12,
        max_automorphisms: 20_000,
        work_limit: 100_000,
        max_expanded: 64,
    }
}

/// Ethanol `[C(H3), C(H2), O(H1)]`.
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Propane.
fn propane() -> MolGraph {
    MolGraph::new(vec![4, 3, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Kekulized toluene: six ring with alternating single/double bonds and a
/// methyl on atom 0.
fn toluene() -> MolGraph {
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

/// Bromochlorofluoromethane: `C(H1)` with Br, Cl, F.
fn bromochlorofluoromethane() -> MolGraph {
    MolGraph::new(vec![2, 12, 11, 10], vec![(0, 1, 1), (0, 2, 1), (0, 3, 1)]).unwrap()
}

/// 2-butanol: `C(H3)-C(H1)-C(H2)-C(H3)` with `O(H1)` on atom 1.
fn butan_2_ol() -> MolGraph {
    MolGraph::new(
        vec![4, 2, 3, 4, 9],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (1, 4, 1)],
    )
    .unwrap()
}

/// Isopropanol: central `C(H1)` with two methyls and `O(H1)`.
fn isopropanol() -> MolGraph {
    MolGraph::new(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]).unwrap()
}

/// 2,3-butanediol with the two centres first (atoms 0, 1), then the methyls
/// (2, 3) and the hydroxyl oxygens (4, 5).
fn butane_2_3_diol() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 4, 4, 9, 9],
        vec![(0, 1, 1), (0, 2, 1), (0, 4, 1), (1, 3, 1), (1, 5, 1)],
    )
    .unwrap()
}

/// Cyclohexane ring atoms 0..=5 with methyls at the given ring positions.
fn dimethylcyclohexane(methyl_a: usize, methyl_b: usize) -> MolGraph {
    let mut atoms = vec![3, 3, 3, 3, 3, 3];
    atoms[methyl_a] = 2;
    atoms[methyl_b] = 2;
    atoms.push(4);
    atoms.push(4);
    let (m0, m1) = (6, 7);
    let mut bonds = vec![
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 0, 1),
        (methyl_a, m0, 1),
        (methyl_b, m1, 1),
    ];
    bonds.sort();
    MolGraph::new(atoms, bonds).unwrap()
}

/// Methylcyclohexane: methyl on ring atom 0.
fn methylcyclohexane() -> MolGraph {
    MolGraph::new(
        vec![2, 3, 3, 3, 3, 3, 4],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 0, 1),
            (0, 6, 1),
        ],
    )
    .unwrap()
}

/// `CC(Cl)C(F)C(Cl)C`: chain atoms 0..=4 with Cl on 1 and 3, F on 2.
fn pseudo_asymmetric() -> MolGraph {
    MolGraph::new(
        vec![4, 2, 2, 2, 4, 11, 10, 11],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (1, 5, 1),
            (2, 6, 1),
            (3, 7, 1),
        ],
    )
    .unwrap()
}

/// 2-butene: `C(H3)-C(H1)=C(H1)-C(H3)`.
fn but_2_ene() -> MolGraph {
    MolGraph::new(vec![4, 2, 2, 4], vec![(0, 1, 1), (1, 2, 2), (2, 3, 1)]).unwrap()
}

/// Hexa-2,4-diene.
fn hexa_2_4_diene() -> MolGraph {
    MolGraph::new(
        vec![4, 2, 2, 2, 2, 4],
        vec![(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1)],
    )
    .unwrap()
}

/// Acetaldoxime `CC=NO`.
fn acetaldoxime() -> MolGraph {
    MolGraph::new(vec![4, 2, 5, 9], vec![(0, 1, 1), (1, 2, 2), (2, 3, 1)]).unwrap()
}

/// `CC=NN` hydrazone: the double bond is C=N; the terminal `N(H2)` is the
/// nitrogen end's heavy substituent.
fn hydrazone() -> MolGraph {
    MolGraph::new(vec![4, 2, 5, 7], vec![(0, 1, 1), (1, 2, 2), (2, 3, 1)]).unwrap()
}

/// Cyclohexene: double bond (0, 1) in a six ring.
fn cyclohexene() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 3, 3, 3, 3],
        vec![
            (0, 1, 2),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 0, 1),
        ],
    )
    .unwrap()
}

/// Kekulized benzene.
fn benzene() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 2, 2, 2, 2],
        vec![
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 0, 2),
        ],
    )
    .unwrap()
}

/// Cyclooctene: double bond (0, 1) in an eight ring.
fn cyclooctene() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 3, 3, 3, 3, 3, 3],
        vec![
            (0, 1, 2),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 6, 1),
            (6, 7, 1),
            (7, 0, 1),
        ],
    )
    .unwrap()
}

/// 1,1-dichloroethene.
fn dichloroethene() -> MolGraph {
    MolGraph::new(vec![1, 3, 11, 11], vec![(0, 1, 2), (0, 2, 1), (0, 3, 1)]).unwrap()
}

/// Propene.
fn propene() -> MolGraph {
    MolGraph::new(vec![3, 2, 4], vec![(0, 1, 2), (1, 2, 1)]).unwrap()
}

/// Allene `C=C=C`.
fn allene() -> MolGraph {
    MolGraph::new(vec![3, 1, 3], vec![(0, 1, 2), (1, 2, 2)]).unwrap()
}

/// Phosphine oxide with four different substituents: `P(=O)(C)(N)(Cl)`.
fn phosphine_oxide() -> MolGraph {
    MolGraph::new(
        vec![16, 8, 4, 7, 11],
        vec![(0, 1, 2), (0, 2, 1), (0, 3, 1), (0, 4, 1)],
    )
    .unwrap()
}

/// Sulfone analogue: `S(=O)2(C)2` (S with coordination 4 and double bonds).
fn sulfone() -> MolGraph {
    MolGraph::new(
        vec![15, 8, 8, 4, 4],
        vec![(0, 1, 2), (0, 2, 2), (0, 3, 1), (0, 4, 1)],
    )
    .unwrap()
}

/// Kekulized [18]annulene: 18 ring with alternating single/double bonds.
fn annulene18() -> MolGraph {
    let atoms = vec![2; 18];
    let mut bonds = Vec::new();
    for i in 0..18 {
        let j = (i + 1) % 18;
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        bonds.push((a, b, if i % 2 == 0 { 2 } else { 1 }));
    }
    bonds.sort();
    MolGraph::new(atoms, bonds).unwrap()
}

/// (name, graph, potential count, stereogenic count, distinct count).
fn table() -> Vec<(&'static str, MolGraph, usize, usize, u64)> {
    vec![
        ("ethanol", ethanol(), 0, 0, 1),
        ("propane", propane(), 0, 0, 1),
        ("toluene", toluene(), 0, 0, 1),
        (
            "bromochlorofluoromethane",
            bromochlorofluoromethane(),
            1,
            1,
            2,
        ),
        ("2-butanol", butan_2_ol(), 1, 1, 2),
        ("isopropanol", isopropanol(), 1, 0, 1),
        ("2,3-butanediol", butane_2_3_diol(), 2, 2, 3),
        (
            "1,4-dimethylcyclohexane",
            dimethylcyclohexane(0, 3),
            2,
            2,
            2,
        ),
        (
            "1,2-dimethylcyclohexane",
            dimethylcyclohexane(0, 1),
            2,
            2,
            3,
        ),
        ("methylcyclohexane", methylcyclohexane(), 1, 0, 1),
        ("pseudo-asymmetric", pseudo_asymmetric(), 3, 3, 4),
        ("2-butene", but_2_ene(), 1, 1, 2),
        ("hexa-2,4-diene", hexa_2_4_diene(), 2, 2, 3),
        ("acetaldoxime", acetaldoxime(), 1, 1, 2),
        ("hydrazone", hydrazone(), 1, 1, 2),
        ("cyclohexene", cyclohexene(), 0, 0, 1),
        ("benzene", benzene(), 0, 0, 1),
        ("cyclooctene", cyclooctene(), 1, 1, 2),
        ("1,1-dichloroethene", dichloroethene(), 0, 0, 1),
        ("propene", propene(), 0, 0, 1),
    ]
}

#[test]
fn potential_stereogenic_and_distinct_counts() {
    let limits = expanded();
    for (name, graph, potential, stereogenic, distinct) in table() {
        let report = perceive(&graph, &limits);
        assert_eq!(report.potential.len(), potential, "{name}: potential");
        assert_eq!(report.elements.len(), stereogenic, "{name}: stereogenic");
        assert_eq!(
            report.not_stereogenic,
            potential - stereogenic,
            "{name}: not_stereogenic"
        );
        assert_eq!(report.distinct, Some(distinct), "{name}: distinct");
        assert_eq!(
            report.raw_assignments,
            Some(1u64 << potential),
            "{name}: raw"
        );
        assert!(report.resolution.is_resolved(), "{name}: resolved");
        assert_eq!(
            report.automorphisms,
            Some(report.automorphisms.unwrap()),
            "{name}: automorphism count present"
        );
        assert!(report.unsupported.is_empty(), "{name}: no unsupported");
        assert!(report.molecule_wide_exact(), "{name}: molecule-wide exact");
        assert!(report.complete_within_supported_kinds(), "{name}: complete");
        // Expanded isomers: one canonical representative per orbit, over
        // the stereogenic elements, lexicographic.
        assert_eq!(report.isomers.len() as u64, distinct, "{name}: isomers");
        assert!(!report.isomers_truncated, "{name}: not truncated");
        let mut sorted = report.isomers.clone();
        sorted.sort();
        assert_eq!(sorted, report.isomers, "{name}: lexicographic");
    }
}

#[test]
fn unsupported_kinds_are_flagged_not_modelled() {
    let limits = expanded();
    let report = perceive(&allene(), &limits);
    assert_eq!(report.potential.len(), 0, "allene: no potential bond");
    assert_eq!(report.unsupported, vec!["axial_cumulene".to_string()]);
    assert_eq!(report.distinct, Some(1));
    assert!(report.resolution.is_resolved());
    assert!(!report.molecule_wide_exact());
    assert!(report.complete_within_supported_kinds());

    let report = perceive(&phosphine_oxide(), &limits);
    assert_eq!(report.potential.len(), 0, "phosphine oxide: no potential");
    assert_eq!(report.unsupported, vec!["phosphorus_center".to_string()]);
    assert!(!report.molecule_wide_exact());

    let report = perceive(&sulfone(), &limits);
    assert_eq!(report.potential.len(), 0, "sulfone: no potential");
    assert_eq!(report.unsupported, vec!["sulfur_center".to_string()]);
    assert!(!report.molecule_wide_exact());

    let report = perceive(&annulene18(), &limits);
    assert_eq!(report.potential.len(), 0, "annulene: no potential bond");
    assert_eq!(
        report.unsupported,
        vec!["conjugated_large_ring".to_string()]
    );
    assert_eq!(report.distinct, Some(1));
    assert!(!report.molecule_wide_exact());

    // Amines never count: dimethylamine's nitrogen is not a centre.
    let amine = MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let report = perceive(&amine, &limits);
    assert_eq!(report.potential.len(), 0, "amine: no centre");
    assert!(report.unsupported.is_empty());
}

#[test]
fn too_many_elements_is_unresolved_with_null_counts() {
    let graph = butane_2_3_diol();
    let limits = StereoLimits {
        max_elements: 1,
        ..expanded()
    };
    let report = perceive(&graph, &limits);
    assert_eq!(
        report.resolution,
        Resolution::Unresolved("too_many_elements".to_string())
    );
    assert_eq!(report.raw_assignments, None);
    assert_eq!(report.distinct, None);
    assert_eq!(report.automorphisms, None);
    assert!(report.isomers.is_empty());
    assert!(!report.complete_within_supported_kinds());
    // The potential set is still reported (2 centres were found).
    assert_eq!(report.potential.len(), 2);
}

#[test]
fn spent_work_limit_is_unresolved_never_a_guess() {
    let graph = butane_2_3_diol();
    let limits = StereoLimits {
        work_limit: 1,
        ..expanded()
    };
    let report = perceive(&graph, &limits);
    assert_eq!(
        report.resolution,
        Resolution::Unresolved("work_limit_exceeded".to_string())
    );
    assert_eq!(report.raw_assignments, None);
    assert_eq!(report.distinct, None);
}

#[test]
fn tiny_automorphism_cap_is_unresolved() {
    // Kekulized benzene has 6 automorphisms (D3 of the alternating ring);
    // a cap of 2 must fail rather than guess.
    let limits = StereoLimits {
        max_automorphisms: 2,
        ..expanded()
    };
    let report = perceive(&benzene(), &limits);
    assert_eq!(
        report.resolution,
        Resolution::Unresolved("too_many_automorphisms".to_string())
    );
    assert_eq!(report.distinct, None);
}

#[test]
fn perceive_is_deterministic() {
    let limits = expanded();
    for (_, graph, _, _, _) in table() {
        assert_eq!(perceive(&graph, &limits), perceive(&graph, &limits));
    }
}

/// Deterministic shuffle with a fixed seed (SplitMix64-style, `u64`
/// wrapping arithmetic, so identical on any platform).
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (next() % ((i + 1) as u64)) as usize;
        perm.swap(i, j);
    }
    perm
}

/// Transfer a full-length assignment on the permuted graph `report_new`
/// back to the original numbering: `perm[i]` is the original atom behind
/// permuted atom `i`. Values travel with the parity flips of the transfer,
/// exactly like the automorphism action.
fn transfer_assignment(
    report_old: &StereoReport,
    report_new: &StereoReport,
    perm: &[usize],
    full_new: &[u8],
) -> Vec<u8> {
    /// Old index behind new ligand.
    fn old_of(ligand: &Ligand, perm: &[usize]) -> Ligand {
        match ligand {
            Ligand::Atom(i) => Ligand::Atom(perm[*i]),
            Ligand::Hydrogen => Ligand::Hydrogen,
            Ligand::LonePair => Ligand::LonePair,
        }
    }
    /// Parity between moved-new ligands and old ligands.
    fn flip_between(moved: &[Ligand], target: &[Ligand]) -> u8 {
        let mut inversions = 0usize;
        let mut pos = Vec::with_capacity(moved.len());
        for item in moved {
            pos.push(target.iter().position(|t| t == item).unwrap());
        }
        for i in 0..pos.len() {
            for j in (i + 1)..pos.len() {
                if pos[i] > pos[j] {
                    inversions += 1;
                }
            }
        }
        (inversions % 2) as u8
    }
    // New potential index keyed by new atoms.
    let mut new_tetra: Vec<Option<usize>> = vec![None; perm.len()];
    let mut new_bond: std::collections::BTreeMap<(usize, usize), usize> =
        std::collections::BTreeMap::new();
    for (j, element) in report_new.potential.iter().enumerate() {
        match element {
            StereoElement::Tetrahedral { atom, .. } => new_tetra[*atom] = Some(j),
            StereoElement::DoubleBond { a, b, .. } => {
                new_bond.insert((*a, *b), j);
            }
        }
    }
    let mut full_old = vec![0u8; report_old.potential.len()];
    for (j, element) in report_old.potential.iter().enumerate() {
        match element {
            StereoElement::Tetrahedral { atom, ligands } => {
                // New atom holding old `atom`: the inverse permutation.
                let new_atom = perm.iter().position(|&p| p == *atom).unwrap();
                let nj = new_tetra[new_atom].unwrap();
                let StereoElement::Tetrahedral {
                    ligands: new_ligands,
                    ..
                } = &report_new.potential[nj]
                else {
                    panic!("tetrahedral maps to tetrahedral");
                };
                let moved: Vec<Ligand> = new_ligands.iter().map(|l| old_of(l, perm)).collect();
                full_old[j] = full_new[nj] ^ flip_between(&moved, ligands);
            }
            StereoElement::DoubleBond { a, b, ref_a, ref_b } => {
                // Old ends in new numbering: invert the permutation.
                let mut new_of_old = vec![0usize; perm.len()];
                for (ni, &oi) in perm.iter().enumerate() {
                    new_of_old[oi] = ni;
                }
                let (na, nb) = (new_of_old[*a], new_of_old[*b]);
                let nkey = if na < nb { (na, nb) } else { (nb, na) };
                let nj = new_bond[&nkey];
                let StereoElement::DoubleBond {
                    a: ta,
                    ref_a: nref_a,
                    ref_b: nref_b,
                    ..
                } = &report_new.potential[nj]
                else {
                    panic!("bond maps to bond");
                };
                let a_first = na == *ta;
                let mapped_a = old_of(if a_first { nref_a } else { nref_b }, perm);
                let mapped_b = old_of(if a_first { nref_b } else { nref_a }, perm);
                let flip = u8::from(mapped_a != *ref_a) ^ u8::from(mapped_b != *ref_b);
                full_old[j] = full_new[nj] ^ flip;
            }
        }
    }
    full_old
}

#[test]
fn permuted_graphs_agree_on_counts_and_isomers() {
    let limits = expanded();
    let check_limits = StereoLimits::default();
    for (name, graph, _, _, _) in table() {
        let n = graph.atoms().len();
        let report = perceive(&graph, &limits);
        // Full-length assignments behind the original isomers
        // (non-stereogenic positions read 0).
        let mut original_full: Vec<Vec<u8>> = Vec::new();
        for isomer in &report.isomers {
            let mut full = vec![0u8; report.potential.len()];
            for (e, &v) in report.elements.iter().zip(isomer.iter()) {
                full[e.potential_index] = v;
            }
            original_full.push(full);
        }
        if original_full.is_empty() {
            original_full.push(vec![0u8; report.potential.len()]);
        }
        for seed in [7u64, 19, 101] {
            let perm = shuffled(n, seed);
            let permuted = graph.permuted(&perm).unwrap();
            let preport = perceive(&permuted, &limits);
            assert_eq!(
                preport.potential.len(),
                report.potential.len(),
                "{name}: potential count under permutation"
            );
            assert_eq!(
                preport.distinct, report.distinct,
                "{name}: distinct under permutation"
            );
            assert_eq!(
                preport.not_stereogenic, report.not_stereogenic,
                "{name}: not_stereogenic under permutation"
            );
            assert_eq!(
                preport.unsupported, report.unsupported,
                "{name}: unsupported under permutation"
            );
            // Every permuted isomer transfers back onto an original isomer.
            let mut permuted_full: Vec<Vec<u8>> = Vec::new();
            for isomer in &preport.isomers {
                let mut full = vec![0u8; preport.potential.len()];
                for (e, &v) in preport.elements.iter().zip(isomer.iter()) {
                    full[e.potential_index] = v;
                }
                permuted_full.push(full);
            }
            if permuted_full.is_empty() {
                permuted_full.push(vec![0u8; preport.potential.len()]);
            }
            for full_new in &permuted_full {
                let back = transfer_assignment(&report, &preport, &perm, full_new);
                let ok = original_full.iter().any(|full_old| {
                    equivalent(&graph, &back, full_old, &check_limits) == Some(true)
                });
                assert!(ok, "{name}: permuted isomer maps back (seed {seed})");
            }
        }
    }
}

#[test]
fn equivalent_distinguishes_diastereomers() {
    // 2,3-butanediol (centres first): (cw, ccw) is the meso pair's other
    // half, (cw, cw) and (ccw, ccw) are the chiral pair.
    let graph = butane_2_3_diol();
    let limits = StereoLimits::default();
    assert_eq!(equivalent(&graph, &[1, 0], &[0, 1], &limits), Some(true));
    assert_eq!(equivalent(&graph, &[1, 1], &[0, 0], &limits), Some(false));
    assert_eq!(equivalent(&graph, &[1, 1], &[1, 1], &limits), Some(true));
    assert_eq!(equivalent(&graph, &[1, 0], &[1, 1], &limits), Some(false));
    // Wrong lengths and non-binary values are undecided, never a guess.
    assert_eq!(equivalent(&graph, &[1], &[0, 1], &limits), None);
    assert_eq!(equivalent(&graph, &[2, 0], &[0, 1], &limits), None);
}

#[test]
fn equivalent_unresolved_under_caps() {
    let graph = butane_2_3_diol();
    let tight = StereoLimits {
        max_elements: 1,
        ..expanded()
    };
    assert_eq!(equivalent(&graph, &[1, 0], &[0, 1], &tight), None);
    let rushed = StereoLimits {
        work_limit: 1,
        ..expanded()
    };
    assert_eq!(equivalent(&graph, &[1, 0], &[0, 1], &rushed), None);
}

/// F12: canonical representatives are the lexicographically smallest
/// assignment vectors (first element most significant). Hexa-2,4-diene has
/// three orbits; the middle orbit's minimum is `[0, 1]`, not `[1, 0]`.
#[test]
fn hexa_representatives_are_lexicographically_minimal() {
    let report = perceive(&hexa_2_4_diene(), &expanded());
    assert_eq!(report.distinct, Some(3));
    assert_eq!(report.isomers, vec![vec![0, 0], vec![0, 1], vec![1, 1]]);
}

/// F6: a 64-centre chain with `max_elements = 64` must report unresolved,
/// never panic or over-allocate (the absolute in-function cap binds first).
fn fluorinated_chain(centres: usize) -> MolGraph {
    // Ends: `C(H0)` with three fluorines each (four heavy neighbours);
    // interiors: `C(H1)` with one fluorine each (three heavy + one H).
    // Every carbon is a tetrahedral centre.
    let mut atoms: Vec<u8> = Vec::new();
    for i in 0..centres {
        atoms.push(if i == 0 || i + 1 == centres { 1 } else { 2 });
    }
    let mut bonds: Vec<(usize, usize, u8)> = Vec::new();
    for i in 0..centres.saturating_sub(1) {
        bonds.push((i, i + 1, 1));
    }
    let mut next = centres;
    let mut attach =
        |bonds: &mut Vec<(usize, usize, u8)>, c: usize, count: usize, next: &mut usize| {
            for _ in 0..count {
                atoms.push(10);
                bonds.push((c, *next, 1));
                *next += 1;
            }
        };
    for c in 0..centres {
        let count = if c == 0 || c + 1 == centres { 3 } else { 1 };
        attach(&mut bonds, c, count, &mut next);
    }
    bonds.sort();
    MolGraph::new(atoms, bonds).unwrap()
}

#[test]
fn oversized_caller_limits_are_refused_without_panic() {
    let graph = fluorinated_chain(64);
    let probe = perceive(&graph, &expanded());
    assert_eq!(probe.potential.len(), 64);
    let limits = StereoLimits {
        max_elements: 64,
        ..expanded()
    };
    let report = perceive(&graph, &limits);
    assert_eq!(
        report.resolution,
        Resolution::Unresolved("too_many_elements".to_string())
    );
    assert_eq!(report.raw_assignments, None);
    assert_eq!(report.distinct, None);
    // Just above the absolute cap: still unresolved, never an allocation.
    let graph17 = fluorinated_chain(17);
    let report17 = perceive(
        &graph17,
        &StereoLimits {
            max_elements: 17,
            ..expanded()
        },
    );
    assert_eq!(
        report17.resolution,
        Resolution::Unresolved("too_many_elements".to_string())
    );
}

/// F3: heteroaromatic macrocycle `C1=CC=CNC=CC=CNC=CC=CN1` — every large-ring
/// double bond is conjugated through lone-pair donors, so no bond is a
/// potential element and the molecule is flagged (never 16 isomers).
fn heteroaromatic_macrocycle() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 2, 2, 6, 2, 2, 2, 2, 6, 2, 2, 2, 2, 6],
        vec![
            (0, 1, 2),
            (0, 14, 1),
            (1, 2, 1),
            (2, 3, 2),
            (3, 4, 1),
            (4, 5, 1),
            (5, 6, 2),
            (6, 7, 1),
            (7, 8, 2),
            (8, 9, 1),
            (9, 10, 1),
            (10, 11, 2),
            (11, 12, 1),
            (12, 13, 2),
            (13, 14, 1),
        ],
    )
    .unwrap()
}

/// F3: fused `C1=CC=CC2=C(C=C1)CCCCCC2` — the shared double bond sits in two
/// equally short eight-member rings; the verdict must not depend on which
/// one the search finds first.
fn fused_large_ring() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
        vec![
            (0, 1, 2),
            (0, 7, 1),
            (1, 2, 1),
            (2, 3, 2),
            (3, 4, 1),
            (4, 5, 2),
            (4, 13, 1),
            (5, 6, 1),
            (5, 8, 1),
            (6, 7, 2),
            (8, 9, 1),
            (9, 10, 1),
            (10, 11, 1),
            (11, 12, 1),
            (12, 13, 1),
        ],
    )
    .unwrap()
}

#[test]
fn large_ring_conjugation_is_unsupported_not_counted() {
    let report = perceive(&heteroaromatic_macrocycle(), &expanded());
    assert_eq!(
        report.unsupported,
        vec!["conjugated_large_ring".to_string()]
    );
    assert_eq!(report.potential.len(), 0);
    assert_eq!(report.distinct, Some(1));
    assert!(!report.molecule_wide_exact());

    let fused = perceive(&fused_large_ring(), &expanded());
    assert!(
        fused
            .unsupported
            .contains(&"conjugated_large_ring".to_string()),
        "fused large ring is flagged: {:?}",
        fused.unsupported
    );
    assert!(!fused.molecule_wide_exact());
    // Cyclooctene stays a stereo element (isolated large-ring bond).
    let oct = perceive(&cyclooctene(), &expanded());
    assert_eq!(oct.distinct, Some(2));
    assert!(oct.unsupported.is_empty());
}

#[test]
fn large_ring_verdicts_do_not_depend_on_atom_numbering() {
    let molecules = vec![
        heteroaromatic_macrocycle(),
        fused_large_ring(),
        cyclooctene(),
        annulene18(),
        toluene(),
        benzene(),
        butan_2_ol(),
        hexa_2_4_diene(),
        but_2_ene(),
    ];
    let limits = expanded();
    for (m, graph) in molecules.iter().enumerate() {
        let n = graph.atoms().len();
        let report = perceive(graph, &limits);
        for seed in [3u64, 42, 777] {
            let perm = shuffled(n, seed);
            let permuted = graph.permuted(&perm).unwrap();
            let preport = perceive(&permuted, &limits);
            assert_eq!(
                preport.potential.len(),
                report.potential.len(),
                "molecule {m}: potential count under permutation (seed {seed})"
            );
            assert_eq!(
                preport.distinct, report.distinct,
                "molecule {m}: distinct under permutation (seed {seed})"
            );
            assert_eq!(
                preport.unsupported, report.unsupported,
                "molecule {m}: unsupported under permutation (seed {seed})"
            );
        }
    }
}

/// F4: `CN1OC1` — the ring nitrogen is a constrained pyramidal centre RDKit
/// counts (2 isomers); it must be flagged, never molecule-wide exact.
fn methyl_oxaziridine() -> MolGraph {
    MolGraph::new(
        vec![4, 5, 8, 3],
        vec![(0, 1, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)],
    )
    .unwrap()
}

/// F4: `CN1C(C)C1` — RDKit counts 4 (two constrained nitrogens / centres).
fn methyl_methylaziridine() -> MolGraph {
    MolGraph::new(
        vec![4, 5, 2, 4, 3],
        vec![(0, 1, 1), (1, 2, 1), (1, 4, 1), (2, 3, 1), (2, 4, 1)],
    )
    .unwrap()
}

/// Kekulized pyrrole: `N(H)` with two single ring bonds in a five-ring —
/// not a constrained centre (negative control).
fn pyrrole() -> MolGraph {
    MolGraph::new(
        vec![6, 2, 2, 2, 2],
        vec![(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 0, 1)],
    )
    .unwrap()
}

/// 2,2',6-trimethylbiphenyl axis: three of four ortho positions carry a
/// heavy substituent (positive atropisomer control).
fn trimethylbiphenyl() -> MolGraph {
    MolGraph::new(
        vec![4, 1, 2, 2, 2, 1, 4, 1, 1, 1, 4, 2, 2, 2, 1, 4],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (1, 7, 2),
            (2, 3, 2),
            (3, 4, 1),
            (4, 5, 2),
            (5, 6, 1),
            (5, 7, 1),
            (7, 8, 1),
            (8, 9, 2),
            (8, 14, 1),
            (9, 10, 1),
            (9, 11, 1),
            (11, 12, 2),
            (12, 13, 1),
            (13, 14, 2),
            (14, 15, 1),
        ],
    )
    .unwrap()
}

/// 2,2'-dimethylbiphenyl axis: two of four ortho positions carry a heavy
/// substituent — below the three-substituent bar (negative control).
fn biphenyl() -> MolGraph {
    MolGraph::new(
        vec![4, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 4],
        vec![
            (0, 1, 1),
            (1, 2, 2),
            (1, 6, 1),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 6, 2),
            (6, 7, 1),
            (7, 8, 2),
            (7, 12, 1),
            (8, 9, 1),
            (9, 10, 2),
            (10, 11, 1),
            (11, 12, 2),
            (12, 13, 1),
        ],
    )
    .unwrap()
}

#[test]
fn unmodelled_kinds_disable_molecule_wide_exactness() {
    // Constrained ring nitrogens (review counterexamples).
    for (name, graph) in [
        ("CN1OC1", methyl_oxaziridine()),
        ("CN1C(C)C1", methyl_methylaziridine()),
    ] {
        let report = perceive(&graph, &expanded());
        assert!(
            report
                .unsupported
                .contains(&"constrained_nitrogen_center".to_string()),
            "{name}: flagged: {:?}",
            report.unsupported
        );
        assert!(!report.molecule_wide_exact(), "{name}: not exact");
    }
    // Acyclic amines and five-ring `N(H)` stay unflagged.
    let amine = MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    assert!(perceive(&amine, &expanded()).unsupported.is_empty());
    assert!(perceive(&pyrrole(), &expanded()).unsupported.is_empty());
    // Candidate atropisomeric axis.
    let tri = perceive(&trimethylbiphenyl(), &expanded());
    assert!(
        tri.unsupported
            .contains(&"atropisomer_axis_possible".to_string()),
        "trimethylbiphenyl flagged: {:?}",
        tri.unsupported
    );
    assert!(!tri.molecule_wide_exact());
    assert!(
        perceive(&biphenyl(), &expanded())
            .unsupported
            .iter()
            .all(|u| u != "atropisomer_axis_possible"),
        "2,2'-dimethylbiphenyl axis is unhindered (2 of 4 ortho substituted)"
    );
    // Phosphorus with three single bonds (no double bond anywhere).
    let p3 = MolGraph::new(vec![16, 4, 4, 4], vec![(0, 1, 1), (0, 2, 1), (0, 3, 1)]).unwrap();
    let report = perceive(&p3, &expanded());
    assert_eq!(report.unsupported, vec!["phosphorus_center".to_string()]);
    assert!(!report.molecule_wide_exact());
    // Phosphorus with two heavy neighbours stays unflagged.
    let p2 = MolGraph::new(vec![16, 1, 4, 8], vec![(0, 1, 3), (1, 2, 1), (0, 3, 2)]).unwrap();
    assert!(perceive(&p2, &expanded()).unsupported.is_empty());
    // Six-coordinate sulfur with only single bonds.
    let s6 = MolGraph::new(
        vec![15, 4, 4, 4, 4, 4, 4],
        vec![
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (0, 4, 1),
            (0, 5, 1),
            (0, 6, 1),
        ],
    )
    .unwrap();
    let report = perceive(&s6, &expanded());
    assert_eq!(report.unsupported, vec!["sulfur_center".to_string()]);
    assert!(!report.molecule_wide_exact());
    // Dimethyl sulfide (two heavy neighbours) stays unflagged.
    let sulfide = MolGraph::new(vec![4, 13, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    assert!(perceive(&sulfide, &expanded()).unsupported.is_empty());
}
