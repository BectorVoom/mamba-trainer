//! MC16 tests: functional-group semantics in host candidate acceptance.
//!
//! Host-only: [`accepts`](mamba3::models::ms2::completion_model::accepts) and
//! [`check_feasibility`](mamba3::models::ms2::completion_model::check_feasibility)
//! are pure host computations, so no device is needed. The C3H8O2 pair
//! (propane-1,2-diol vs 2-methoxyethanol) shows what each rule accepts for a
//! query of two hydroxyl patterns.

#![cfg(feature = "backend")]

use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion_model::{
    Acceptance, CompletionGenerationConfig, SubstructureSemantics, accepts, check_feasibility,
};
use mamba3::models::ms2::functional_groups::functional_groups;
use mamba3::models::ms2::graph::MolGraph;

const NODE_LIMIT: usize = 100_000;
const IDENTITY_LIMIT: usize = 100_000;

/// One hydroxyl pattern: the O(H1) atom alone.
fn hydroxyl() -> MolGraph {
    MolGraph::new(vec![9], vec![]).unwrap()
}

/// Two hydroxyl patterns (a query listing two hydroxyl groups).
fn two_hydroxyls() -> Vec<MolGraph> {
    vec![hydroxyl(), hydroxyl()]
}

/// Propane-1,2-diol `HO-CH2-CH(OH)-CH3` (C3H8O2, two hydroxyl groups).
fn propane_1_2_diol() -> MolGraph {
    MolGraph::new(
        vec![3, 9, 2, 9, 4],
        vec![(0, 1, 1), (0, 2, 1), (2, 3, 1), (2, 4, 1)],
    )
    .unwrap()
}

/// 2-methoxyethanol `CH3-O-CH2-CH2-OH` (C3H8O2, one hydroxyl group plus an
/// ether oxygen).
fn methoxyethanol() -> MolGraph {
    MolGraph::new(
        vec![4, 8, 3, 3, 9],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)],
    )
    .unwrap()
}

/// Ethanol (C2H6O, one hydroxyl group).
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethyl ether (C2H6O, one ether group, no hydroxyl).
fn dimethyl_ether() -> MolGraph {
    MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Allyl alcohol `CH2=CH-CH2-OH` (an extra alkene next to the hydroxyl).
fn allyl_alcohol() -> MolGraph {
    MolGraph::new(vec![3, 2, 3, 9], vec![(0, 1, 2), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Hydroxyacetone `CH3-C(=O)-CH2-OH` (an extra carbonyl next to the
/// hydroxyl).
fn hydroxyacetone() -> MolGraph {
    MolGraph::new(
        vec![4, 1, 3, 9, 8],
        vec![(0, 1, 1), (1, 4, 2), (1, 2, 1), (2, 3, 1)],
    )
    .unwrap()
}

fn composition(c: u16, h: u16, o: u16) -> Composition {
    let mut out: Composition = [0; 10];
    out[0] = c;
    out[1] = h;
    out[3] = o;
    out
}

/// Kekulized benzene: six `C(H1)` in an alternating single/double ring.
fn benzene() -> MolGraph {
    MolGraph::new(
        vec![2, 2, 2, 2, 2, 2],
        vec![
            (0, 1, 2),
            (1, 2, 1),
            (2, 3, 2),
            (3, 4, 1),
            (4, 5, 2),
            (5, 0, 1),
        ],
    )
    .unwrap()
}

/// Kekulized phenol: benzene ring plus an `O(H1)` on an `C(H0)` ipso carbon.
fn phenol() -> MolGraph {
    MolGraph::new(
        vec![1, 2, 2, 2, 2, 2, 9],
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

/// Butane chain `C(H3)-C(H2)-C(H2)-C(H3)` for the disjoint joint search.
fn butane() -> MolGraph {
    MolGraph::new(
        vec![4, 3, 3, 4],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)],
    )
    .unwrap()
}

/// Propane chain `C(H3)-C(H2)-C(H3)`: too short for two disjoint edges.
fn propane_chain() -> MolGraph {
    MolGraph::new(vec![4, 3, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// One `C(H3)-C(H2)` single-bond pattern (matches a terminal ethyl edge).
fn ethyl_edge() -> MolGraph {
    MolGraph::new(vec![4, 3], vec![(0, 1, 1)]).unwrap()
}

/// The full Ertl group list of `molecule` as typed induced subgraphs.
fn full_groups(molecule: &MolGraph) -> Vec<MolGraph> {
    functional_groups(molecule)
        .unwrap()
        .iter()
        .map(|group| molecule.induced(&group.atoms).unwrap())
        .collect()
}

#[test]
fn extraction_marks_the_expected_groups() {
    // Guard for the constructions below: the diol has two single-oxygen
    // groups, the ether an ether oxygen plus a hydroxyl oxygen.
    let diol = functional_groups(&propane_1_2_diol()).unwrap();
    assert_eq!(diol.len(), 2, "the diol carries two groups");
    assert!(diol.iter().all(|g| g.atoms.len() == 1));
    let ether = functional_groups(&methoxyethanol()).unwrap();
    assert_eq!(ether.len(), 2, "the ether carries two groups");
    let allyl = functional_groups(&allyl_alcohol()).unwrap();
    assert_eq!(allyl.len(), 2, "allyl alcohol carries alkene plus hydroxyl");
    let keto = functional_groups(&hydroxyacetone()).unwrap();
    assert_eq!(
        keto.len(),
        2,
        "hydroxyacetone carries carbonyl plus hydroxyl"
    );
    // Compositions are what the formulas say.
    assert_eq!(propane_1_2_diol().composition(), composition(3, 8, 2));
    assert_eq!(methoxyethanol().composition(), composition(3, 8, 2));
}

#[test]
fn contained_accepts_both_c3h8o2_isomers() {
    // Under `Contained` each pattern is checked on its own: both hydroxyl
    // patterns map onto the ether's single hydroxyl oxygen.
    let patterns = two_hydroxyls();
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &patterns,
            SubstructureSemantics::Contained,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "the diol holds two hydroxyls"
    );
    assert_eq!(
        accepts(
            &methoxyethanol(),
            &patterns,
            SubstructureSemantics::Contained,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "one ether hydroxyl satisfies both patterns when sharing is allowed"
    );
}

#[test]
fn disjoint_needs_two_occurrences() {
    // Under `DisjointOccurrences` the two patterns need disjoint images: the
    // diol has two hydroxyl oxygens, the ether only one.
    let patterns = two_hydroxyls();
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &patterns,
            SubstructureSemantics::DisjointOccurrences,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "the diol holds two disjoint hydroxyls"
    );
    assert_eq!(
        accepts(
            &methoxyethanol(),
            &patterns,
            SubstructureSemantics::DisjointOccurrences,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedContainment,
        "the ether has only one hydroxyl oxygen"
    );
}

#[test]
fn complete_compares_full_group_lists() {
    // The full group list of the diol is two hydroxyls: it accepts the diol
    // and rejects the ether (which lacks the second hydroxyl).
    let patterns = two_hydroxyls();
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &patterns,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "the diol's groups are exactly two hydroxyls"
    );
    assert_eq!(
        accepts(
            &methoxyethanol(),
            &patterns,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedMissingGroups,
        "the ether lacks the second hydroxyl"
    );
}

#[test]
fn complete_counts_multiset_occurrences() {
    // Two identical groups need two occurrences: ethanol (one hydroxyl)
    // against two hydroxyl patterns is missing, and the diol (two
    // hydroxyls) against one pattern has an extra group.
    let two = two_hydroxyls();
    let one = vec![hydroxyl()];
    assert_eq!(
        accepts(
            &ethanol(),
            &two,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedMissingGroups,
        "one hydroxyl cannot cover two"
    );
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &one,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedExtraGroups,
        "the second hydroxyl is extra"
    );
    assert_eq!(
        accepts(
            &ethanol(),
            &one,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "ethanol's only group is the hydroxyl"
    );
    assert_eq!(
        accepts(
            &dimethyl_ether(),
            &one,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedMissingGroups,
        "the ether oxygen is not a hydroxyl"
    );
}

#[test]
fn complete_rejects_extra_alkene_and_carbonyl() {
    // A candidate carrying an extra alkene or carbonyl has a different
    // group list, even though it contains the hydroxyl disjointly.
    let one = vec![hydroxyl()];
    assert_eq!(
        accepts(
            &allyl_alcohol(),
            &one,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedExtraGroups,
        "the extra alkene changes the list"
    );
    assert_eq!(
        accepts(
            &hydroxyacetone(),
            &one,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedExtraGroups,
        "the extra carbonyl changes the list"
    );
    // Both still pass the weaker rules for the single hydroxyl.
    for candidate in [allyl_alcohol(), hydroxyacetone()] {
        assert_eq!(
            accepts(
                &candidate,
                &one,
                SubstructureSemantics::Contained,
                NODE_LIMIT,
                IDENTITY_LIMIT
            ),
            Acceptance::Accepted,
        );
        assert_eq!(
            accepts(
                &candidate,
                &one,
                SubstructureSemantics::DisjointOccurrences,
                NODE_LIMIT,
                IDENTITY_LIMIT
            ),
            Acceptance::Accepted,
        );
    }
}

#[test]
fn empty_patterns_mean_no_groups_under_complete() {
    let empty: Vec<MolGraph> = Vec::new();
    assert_eq!(
        accepts(
            &ethanol(),
            &empty,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedExtraGroups,
        "ethanol's hydroxyl is extra against an empty list"
    );
    assert_eq!(
        accepts(
            &ethanol(),
            &empty,
            SubstructureSemantics::Contained,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "an empty list is vacuously contained"
    );
    assert_eq!(
        accepts(
            &ethanol(),
            &empty,
            SubstructureSemantics::DisjointOccurrences,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "an empty list is vacuously disjoint"
    );
}

#[test]
fn work_limit_is_unresolved_never_acceptance() {
    // A spent budget resolves nothing: all three rules report unresolved.
    let patterns = two_hydroxyls();
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &patterns,
            SubstructureSemantics::Contained,
            0,
            IDENTITY_LIMIT
        ),
        Acceptance::ContainmentUnresolved,
    );
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &patterns,
            SubstructureSemantics::DisjointOccurrences,
            0,
            IDENTITY_LIMIT
        ),
        Acceptance::ContainmentUnresolved,
    );
    assert_eq!(
        accepts(
            &ethanol(),
            &[hydroxyl()],
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            0
        ),
        Acceptance::ContainmentUnresolved,
    );
}

#[test]
fn contained_never_fails_the_precheck() {
    // Patterns may share atoms, so `Contained` imposes no composition fit.
    let patterns = two_hydroxyls();
    assert_eq!(
        check_feasibility(
            &composition(2, 6, 1),
            &patterns,
            SubstructureSemantics::Contained
        ),
        None,
    );
}

#[test]
fn disjoint_sums_element_counts() {
    // Two hydroxyl oxygens need two oxygens in the composition.
    let patterns = two_hydroxyls();
    assert_eq!(
        check_feasibility(
            &composition(3, 8, 2),
            &patterns,
            SubstructureSemantics::DisjointOccurrences
        ),
        None,
        "C3H8O2 fits two disjoint hydroxyls"
    );
    let reason = check_feasibility(
        &composition(2, 6, 1),
        &patterns,
        SubstructureSemantics::DisjointOccurrences,
    )
    .expect("C2H6O cannot hold two disjoint hydroxyl oxygens");
    assert!(reason.contains('O'), "the reason names oxygen: {reason}");
    // Hydrogens sum too: eleven hydroxyls need eleven hydrogens.
    let many: Vec<MolGraph> = (0..11).map(|_| hydroxyl()).collect();
    let reason = check_feasibility(
        &composition(0, 10, 11),
        &many,
        SubstructureSemantics::DisjointOccurrences,
    )
    .expect("ten hydrogens cannot hold eleven hydroxyls");
    assert!(
        reason.contains("hydrogen"),
        "the reason names hydrogen: {reason}"
    );
}

#[test]
fn complete_needs_every_heteroatom_accounted() {
    // Ertl marks every heteroatom: under `CompleteFunctionalGroups` the
    // patterns' heteroatom counts must equal the composition's.
    assert_eq!(
        check_feasibility(
            &composition(2, 6, 1),
            &[hydroxyl()],
            SubstructureSemantics::CompleteFunctionalGroups
        ),
        None,
        "one hydroxyl accounts for ethanol's oxygen"
    );
    let reason = check_feasibility(
        &composition(2, 6, 2),
        &[hydroxyl()],
        SubstructureSemantics::CompleteFunctionalGroups,
    )
    .expect("one hydroxyl cannot account for two oxygens");
    assert!(reason.contains('O'), "the reason names oxygen: {reason}");
    // Carbon and hydrogen have unmarked atoms, so they only need to fit.
    assert_eq!(
        check_feasibility(
            &composition(3, 8, 2),
            &two_hydroxyls(),
            SubstructureSemantics::CompleteFunctionalGroups
        ),
        None,
        "carbons and hydrogens may live outside the groups"
    );
    // An empty list is feasible only with no heteroatoms at all.
    let empty: Vec<MolGraph> = Vec::new();
    assert_eq!(
        check_feasibility(
            &composition(3, 6, 0),
            &empty,
            SubstructureSemantics::CompleteFunctionalGroups
        ),
        None,
        "a hydrocarbon may have no groups"
    );
    assert!(
        check_feasibility(
            &composition(2, 6, 1),
            &empty,
            SubstructureSemantics::CompleteFunctionalGroups
        )
        .is_some(),
        "ethanol's oxygen must be accounted for"
    );
}

#[test]
fn default_semantics_is_contained() {    assert_eq!(
        CompletionGenerationConfig::default().substructure_semantics,
        SubstructureSemantics::Contained,
    );
    assert_eq!(
        SubstructureSemantics::parse("contained"),
        Some(SubstructureSemantics::Contained)
    );
    assert_eq!(
        SubstructureSemantics::parse("disjoint_occurrences"),
        Some(SubstructureSemantics::DisjointOccurrences)
    );
    assert_eq!(
        SubstructureSemantics::parse("complete_functional_groups"),
        Some(SubstructureSemantics::CompleteFunctionalGroups)
    );
    assert_eq!(SubstructureSemantics::parse("disjoint"), None);
    assert_eq!(SubstructureSemantics::Contained.as_str(), "contained");
    assert_eq!(
        SubstructureSemantics::DisjointOccurrences.as_str(),
        "disjoint_occurrences"
    );
    assert_eq!(
        SubstructureSemantics::CompleteFunctionalGroups.as_str(),
        "complete_functional_groups"
    );
}

#[test]
fn target_self_acceptance_sweep() {
    // Every target accepts its own full Ertl group list under every rule,
    // in either pattern order: the extraction/acceptance contract behind
    // the complete-rule driver (which must pass the untruncated Ertl list,
    // never the optional aromatic-ring additions).
    let molecules = [
        ("diol", propane_1_2_diol()),
        ("ethanol", ethanol()),
        ("allyl", allyl_alcohol()),
        ("keto", hydroxyacetone()),
        ("benzene", benzene()),
        ("phenol", phenol()),
    ];
    let rules = [
        SubstructureSemantics::Contained,
        SubstructureSemantics::DisjointOccurrences,
        SubstructureSemantics::CompleteFunctionalGroups,
    ];
    for (name, molecule) in molecules {
        let groups = full_groups(&molecule);
        for rule in rules {
            assert_eq!(
                accepts(&molecule, &groups, rule, NODE_LIMIT, IDENTITY_LIMIT),
                Acceptance::Accepted,
                "{name} self-accepts under {rule:?}",
            );
        }
        // Permuted pattern order changes nothing: rebuild reversed via
        // fresh induced subgraphs (MolGraph is not Clone).
        let order: Vec<usize> = (0..groups.len()).rev().collect();
        let group_atoms: Vec<Vec<usize>> = functional_groups(&molecule)
            .unwrap()
            .iter()
            .map(|group| group.atoms.clone())
            .collect();
        let reversed: Vec<MolGraph> = order
            .iter()
            .map(|&k| molecule.induced(&group_atoms[k]).unwrap())
            .collect();
        assert_eq!(
            accepts(
                &molecule,
                &reversed,
                SubstructureSemantics::CompleteFunctionalGroups,
                NODE_LIMIT,
                IDENTITY_LIMIT
            ),
            Acceptance::Accepted,
            "{name} self-accepts with reordered groups",
        );
        // The true composition fits its own groups under every rule.
        for rule in rules {
            assert_eq!(
                check_feasibility(&molecule.composition(), &groups, rule),
                None,
                "{name} groups fit its composition under {rule:?}",
            );
        }
    }
    // Benzene carries no Ertl groups (aromatic bonds are not marked).
    assert!(full_groups(&benzene()).is_empty());
    // Phenol carries exactly its hydroxyl oxygen.
    assert_eq!(full_groups(&phenol()).len(), 1);
}

#[test]
fn optional_aromatic_rings_break_complete_self_acceptance() {
    // An unmarked aromatic ring as an extra pattern breaks complete
    // self-acceptance (extra group), while contained still holds: the
    // driver must keep optional rings in encoder conditioning only.
    let benzene_groups = full_groups(&benzene());
    assert!(benzene_groups.is_empty());
    let ring_atoms: Vec<usize> = (0..6).collect();
    let ring = benzene().induced(&ring_atoms).unwrap();
    let with_ring = vec![ring];
    assert_eq!(
        accepts(
            &benzene(),
            &with_ring,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedMissingGroups,
        "the Ertl list has no ring to cover the ring pattern",
    );
    assert_eq!(
        accepts(
            &benzene(),
            &with_ring,
            SubstructureSemantics::Contained,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "the ring is contained in its own molecule",
    );
}

#[test]
fn methoxyethanol_per_rule_verdicts() {
    // The review's example: two hydroxyl patterns are independently
    // contained in methoxyethanol (shared image allowed), need two disjoint
    // images under disjoint, and a full-list comparison under complete.
    let patterns = two_hydroxyls();
    assert_eq!(
        accepts(
            &methoxyethanol(),
            &patterns,
            SubstructureSemantics::Contained,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "one hydroxyl satisfies both patterns when sharing is allowed",
    );
    assert_eq!(
        accepts(
            &methoxyethanol(),
            &patterns,
            SubstructureSemantics::DisjointOccurrences,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedContainment,
        "the ether has a single hydroxyl oxygen",
    );
    assert_eq!(
        accepts(
            &methoxyethanol(),
            &patterns,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedMissingGroups,
        "the ether's groups are hydroxyl plus ether, not two hydroxyls",
    );
}

#[test]
fn disjoint_joint_search_needs_backtracking() {
    // Two ethyl-edge patterns in a butane chain: the joint search must place
    // them on disjoint edges (backtracking over embeddings). A propane
    // chain has no two disjoint edges and rejects.
    let two_edges = vec![ethyl_edge(), ethyl_edge()];
    assert_eq!(
        accepts(
            &butane(),
            &two_edges,
            SubstructureSemantics::DisjointOccurrences,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "butane holds two disjoint ethyl edges",
    );
    assert_eq!(
        accepts(
            &propane_chain(),
            &two_edges,
            SubstructureSemantics::DisjointOccurrences,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::RejectedContainment,
        "propane has no two disjoint edges",
    );
    // Non-induced: a bondless two-carbon pattern is contained in bonded
    // carbons (only existing pattern bonds must match).
    let loose = MolGraph::new(vec![3, 3], vec![]).unwrap();
    assert_eq!(
        accepts(
            &butane(),
            &[loose],
            SubstructureSemantics::Contained,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
        "unbonded carbons match bonded ones (non-induced)",
    );
}

#[test]
fn complete_mixed_unresolved_is_never_acceptance() {
    // A spent identity budget resolves nothing, even with a certified match
    // elsewhere in the list: the diol against two hydroxyls is unresolved
    // at limit 0 but accepted with budget.
    let patterns = two_hydroxyls();
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &patterns,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            0
        ),
        Acceptance::ContainmentUnresolved,
    );
    assert_eq!(
        accepts(
            &propane_1_2_diol(),
            &patterns,
            SubstructureSemantics::CompleteFunctionalGroups,
            NODE_LIMIT,
            IDENTITY_LIMIT
        ),
        Acceptance::Accepted,
    );
}
