//! P1-A tests: every expected value comes from
//! `tests/fixtures/ms2/chemistry_v0.json`, derived independently by
//! `tools/ms2/make_fixtures.py`. The fixture is the specification.

use std::path::PathBuf;

use serde_json::Value;

use mamba3::models::ms2::{
    ADDUCTS, ATOM_TYPES, CANONICAL_WORK_LIMIT, CHEMISTRY_VERSION, CLOSE_RING, Composition,
    ELECTRON_EXACT, ELECTRON_MASS, ELECTRON_RESIDUAL_NDA, ELEMENTS, GRAMMAR_VERSION, HYDROGEN,
    Limits, MASS_SCALE, MAX_GRAPH_ATOMS, MolGraph, RawAtom, RawMolecule, TRAVERSAL_VERSION, Token,
    TraceState, Verdict, adduct, atom_type, canonical_trace, composition_error_nda,
    composition_mass, decide, element_index, first_illegal_step, ion, parent_mass, parse_decimal,
    replay, tolerance, tolerance_u32,
};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json")
}

fn fixture() -> Value {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn as_u32(v: &Value) -> u32 {
    v.as_u64().expect("u32 in fixture") as u32
}

fn token_of(t: &Value) -> Token {
    let a = t.as_array().expect("token array");
    Token {
        kind: a[0].as_u64().expect("kind") as u8,
        atom_type: a[1].as_u64().expect("atom_type") as u8,
        bond: a[2].as_u64().expect("bond") as u8,
        pointer: a[3].as_u64().expect("pointer") as u8,
    }
}

fn trace_of(t: &Value) -> Vec<Token> {
    t.as_array()
        .expect("trace array")
        .iter()
        .map(token_of)
        .collect()
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

fn composition_of_formula(formula: &Value) -> Composition {
    let mut c: Composition = [0; 10];
    for (symbol, count) in formula.as_object().expect("formula object") {
        let e = element_index(symbol).expect("known element");
        c[e] = count.as_u64().expect("count") as u16;
    }
    c
}

fn parent_composition(m: &Value) -> Composition {
    graph_of(m).composition()
}

#[test]
fn element_table_matches_fixture() {
    let f = fixture();
    assert_eq!(f["mass_scale"].as_u64().unwrap(), u64::from(MASS_SCALE));
    assert_eq!(f["chemistry"].as_str().unwrap(), CHEMISTRY_VERSION);
    assert_eq!(f["grammar"].as_str().unwrap(), GRAMMAR_VERSION);
    assert_eq!(f["traversal"].as_str().unwrap(), TRAVERSAL_VERSION);
    let elements = f["elements"].as_array().expect("elements");
    assert_eq!(elements.len(), ELEMENTS.len());
    for (entry, e) in elements.iter().zip(ELEMENTS.iter()) {
        assert_eq!(entry["symbol"].as_str().unwrap(), e.symbol);
        assert_eq!(entry["exact"].as_str().unwrap(), e.exact);
        assert_eq!(as_u32(&entry["udalton"]), e.mass);
        assert_eq!(as_u32(&entry["residual_nda"]), e.residual_nda);
    }
    assert_eq!(f["electron"]["exact"].as_str().unwrap(), ELECTRON_EXACT);
    assert_eq!(as_u32(&f["electron"]["udalton"]), ELECTRON_MASS);
    assert_eq!(
        as_u32(&f["electron"]["residual_nda"]),
        ELECTRON_RESIDUAL_NDA
    );
    let types = f["atom_types"].as_array().expect("atom_types");
    assert_eq!(types.len(), ATOM_TYPES.len());
    for (entry, t) in types.iter().zip(ATOM_TYPES.iter()) {
        assert_eq!(entry["id"].as_u64().unwrap(), u64::from(t.id));
        assert_eq!(
            element_index(entry["element"].as_str().unwrap()),
            Some(t.element)
        );
        assert_eq!(entry["hydrogens"].as_u64().unwrap(), u64::from(t.hydrogens));
        assert_eq!(entry["valence"].as_u64().unwrap(), u64::from(t.valence));
        assert_eq!(atom_type(t.id).map(|a| a.id), Some(t.id));
    }
    assert!(atom_type(0).is_none());
    let adducts = f["adducts"].as_array().expect("adducts");
    assert_eq!(adducts.len(), ADDUCTS.len());
    for (entry, a) in adducts.iter().zip(ADDUCTS.iter()) {
        assert_eq!(entry["id"].as_u64().unwrap(), u64::from(a.id));
        assert_eq!(entry["name"].as_str().unwrap(), a.name);
        assert_eq!(entry["hydrogens"].as_i64().unwrap(), i64::from(a.hydrogens));
        assert_eq!(entry["charge"].as_i64().unwrap(), i64::from(a.charge));
        assert_eq!(adduct(a.id).map(|x| x.id), Some(a.id));
    }
    assert!(adduct(0).is_none());
}

#[test]
fn parse_decimal_matches_fixture_and_rejects() {
    let f = fixture();
    for case in f["adduct_cases"].as_array().expect("adduct_cases") {
        let text = case["mz_decimal"].as_str().unwrap();
        assert_eq!(
            parse_decimal(text).unwrap(),
            as_u32(&case["mz_udalton"]),
            "{text}"
        );
    }
    for bad in ["", "-1", "1e3", "1.2.3", "4294.967296", "abc"] {
        assert!(parse_decimal(bad).is_err(), "{bad:?} rejected");
    }
    assert_eq!(parse_decimal("1.0000005").unwrap(), 1_000_000);
    assert_eq!(parse_decimal("1.0000015").unwrap(), 1_000_002);
    assert_eq!(parse_decimal("0.0000004").unwrap(), 0);
}

#[test]
fn parent_mass_matches_fixture() {
    let f = fixture();
    for case in f["adduct_cases"].as_array().expect("adduct_cases") {
        let mz = as_u32(&case["mz_udalton"]);
        assert_eq!(
            parent_mass(mz, 1).unwrap(),
            as_u32(&case["parent_udalton"]["1"])
        );
        assert_eq!(
            parent_mass(mz, 2).unwrap(),
            as_u32(&case["parent_udalton"]["2"])
        );
    }
    assert!(parent_mass(100_000_000, 0).is_err());
    assert!(parent_mass(100_000_000, 7).is_err());
}

#[test]
fn tolerance_matches_fixture_and_sweep() {
    let f = fixture();
    for case in f["tolerance_cases"].as_array().expect("tolerance_cases") {
        let mz = as_u32(&case["mz_udalton"]);
        let t = as_u32(&case["ppm_tenths"]);
        assert_eq!(tolerance(mz, t), as_u32(&case["tolerance_udalton"]));
        assert_eq!(tolerance_u32(mz, t).unwrap(), tolerance(mz, t));
    }
    assert!(tolerance_u32(100_000_000, 1001).is_err());
    // Deterministic sweep: fixed-seed LCG over the full u32 range, plus extremes.
    let mut state: u64 = 20261002;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    for _ in 0..200_000 {
        let mz = next();
        let t = next() % 1001;
        assert_eq!(
            tolerance_u32(mz, t).unwrap(),
            tolerance(mz, t),
            "mz={mz} t={t}"
        );
    }
    for mz in [0, 1, u32::MAX] {
        for t in [0, 1, 1000] {
            assert_eq!(tolerance_u32(mz, t).unwrap(), tolerance(mz, t));
        }
    }
}

#[test]
fn decide_matches_fixture() {
    let f = fixture();
    for case in f["decision_cases"].as_array().expect("decision_cases") {
        let verdict = decide(
            as_u32(&case["observed"]),
            as_u32(&case["computed"]),
            as_u32(&case["error"]),
            as_u32(&case["tolerance"]),
        );
        let expected = match case["verdict"].as_str().unwrap() {
            "accept" => Verdict::Accept,
            "reject" => Verdict::Reject,
            "ambiguous" => Verdict::Ambiguous,
            other => panic!("unknown verdict {other}"),
        };
        assert_eq!(verdict, expected);
    }
}

#[test]
fn molecule_masses_match_fixture() {
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let raw = raw_molecule(m);
        assert!(raw.classify().is_empty(), "{name} in domain");
        let graph = raw.to_graph().expect("in-domain builds");
        let fixture_atoms: Vec<u8> = m["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(graph.atoms(), fixture_atoms.as_slice(), "{name} atom types");
        let expected = composition_of_formula(&m["formula"]);
        assert_eq!(graph.composition(), expected, "{name} formula");
        let mass = composition_mass(&graph.composition()).unwrap();
        assert_eq!(mass, as_u32(&m["mass_udalton"]), "{name} mass");
        let error = composition_error_nda(&graph.composition());
        assert_eq!(
            error,
            m["mass_error_nda"].as_u64().unwrap(),
            "{name} error bound"
        );
        // Integer mass agrees with the decimal exact mass within the bound,
        // with integer arithmetic only.
        let exact = parse_decimal(m["mass_exact"].as_str().unwrap()).unwrap();
        let bound = (error + 999) / 1000;
        assert!(
            mass.abs_diff(exact) as u64 <= bound,
            "{name} exact agreement"
        );
        // RDKit's older table is within 1.2 micro-dalton per heavy atom + 1.
        let rdkit = parse_decimal(m["rdkit_mass"].as_str().unwrap()).unwrap();
        let heavy = graph.atoms().len() as u64;
        assert!(
            5 * mass.abs_diff(rdkit) as u64 <= 6 * heavy + 5,
            "{name} rdkit agreement"
        );
    }
}

#[test]
fn out_of_domain_classification_matches_fixture() {
    let f = fixture();
    for m in f["out_of_domain"].as_array().expect("out_of_domain") {
        let name = m["name"].as_str().unwrap();
        let raw = raw_molecule(m);
        let names: Vec<&str> = raw.classify().iter().map(|r| r.name()).collect();
        let expected: Vec<&str> = m["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        assert_eq!(names, expected, "{name}");
        assert!(raw.to_graph().is_err(), "{name} has no graph");
    }
}

#[test]
fn canonical_traces_match_fixture() {
    let f = fixture();
    let mut worst = 0usize;
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let graph = graph_of(m);
        for g in m["subgraphs"].as_array().expect("subgraphs") {
            let Some(fixture_trace) = g.get("canonical_trace") else {
                continue;
            };
            let members: Vec<usize> = g["atoms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() as usize)
                .collect();
            let sub = graph.induced(&members).expect("induced subgraph");
            let found =
                canonical_trace(&sub, Limits::V0, CANONICAL_WORK_LIMIT).expect("canonicalizes");
            worst = worst.max(found.expansions);
            assert!(found.expansions < CANONICAL_WORK_LIMIT, "{name} work limit");
            let expected = trace_of(fixture_trace);
            assert_eq!(found.trace, expected, "{name} {members:?} trace");
            // Replaying gives the fixture open valence, and the replayed
            // graph canonicalizes back to the same trace.
            let state = replay(&found.trace, Limits::V0, None).expect("trace replays");
            let open: Vec<u64> = g["open_valence_canonical_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            let residual: Vec<u64> = state
                .residual_valence()
                .iter()
                .map(|v| u64::from(*v))
                .collect();
            assert_eq!(residual, open, "{name} {members:?} open valence");
            let back = canonical_trace(&state.graph().unwrap(), Limits::V0, CANONICAL_WORK_LIMIT)
                .expect("regraph canonicalizes");
            assert_eq!(back.trace, expected, "{name} {members:?} round trip");
        }
    }
    println!("largest expansions in canonical_traces_match_fixture: {worst}");
}

#[test]
fn legality_masks_match_fixture() {
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let budget = parent_composition(m);
        for g in m["subgraphs"].as_array().expect("subgraphs") {
            let Some(fixture_trace) = g.get("canonical_trace") else {
                continue;
            };
            let trace = trace_of(fixture_trace);
            for (key, use_budget) in [
                ("legal_masks_parent_budget", true),
                ("legal_masks_no_budget", false),
            ] {
                let mut state = TraceState::new(Limits::V0, use_budget.then_some(budget));
                let expected_masks = g[key].as_array().expect("masks");
                for (i, token) in trace.iter().enumerate() {
                    let masks = state.masks(*token);
                    let expected = expected_masks[i].as_array().expect("mask row");
                    assert_eq!(
                        masks.kinds,
                        as_u32(&expected[0]),
                        "{name} step {i} kinds {key}"
                    );
                    assert_eq!(
                        masks.atom_types,
                        as_u32(&expected[1]),
                        "{name} step {i} types {key}"
                    );
                    assert_eq!(
                        masks.bonds,
                        as_u32(&expected[2]),
                        "{name} step {i} bonds {key}"
                    );
                    assert_eq!(
                        masks.pointers,
                        as_u32(&expected[3]),
                        "{name} step {i} ptrs {key}"
                    );
                    assert!(state.is_legal(*token), "{name} step {i} legal {key}");
                    state.apply(*token).expect("trace applies");
                }
            }
        }
    }
}

#[test]
fn identity_classes_match_fixture() {
    use std::collections::BTreeMap;
    let f = fixture();
    let mut worst = 0usize;
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let graph = graph_of(m);
        let mut by_trace: BTreeMap<Vec<Token>, Vec<u64>> = BTreeMap::new();
        for g in m["subgraphs"].as_array().expect("subgraphs") {
            let members: Vec<usize> = g["atoms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() as usize)
                .collect();
            let sub = graph.induced(&members).expect("induced subgraph");
            let found =
                canonical_trace(&sub, Limits::V0, CANONICAL_WORK_LIMIT).expect("canonicalizes");
            worst = worst.max(found.expansions);
            let class = g["class"].as_u64().unwrap();
            by_trace.entry(found.trace).or_default().push(class);
        }
        let expected_classes = m["identity_classes"].as_u64().unwrap() as usize;
        assert_eq!(by_trace.len(), expected_classes, "{name} class count");
        for (trace, classes) in &by_trace {
            assert!(
                classes.windows(2).all(|w| w[0] == w[1]),
                "{name} {trace:?} one class"
            );
        }
    }
    println!("largest expansions in identity_classes_match_fixture: {worst}");
}

#[test]
fn canonical_trace_permutation_invariant() {
    let f = fixture();
    let mut seed: u64 = 20261002;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let graph = graph_of(m);
        if graph.atoms().len() > Limits::V0.max_atoms()
            || graph.ring_closures() > Limits::V0.max_closures()
        {
            continue;
        }
        let n = graph.atoms().len();
        let original =
            canonical_trace(&graph, Limits::V0, CANONICAL_WORK_LIMIT).expect("canonicalizes");
        for _ in 0..20 {
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (next() as usize) % (i + 1);
                perm.swap(i, j);
            }
            let shuffled = graph.permuted(&perm).expect("permutation builds");
            let found = canonical_trace(&shuffled, Limits::V0, CANONICAL_WORK_LIMIT)
                .expect("permuted canonicalizes");
            assert_eq!(found.trace, original.trace, "{name} perm {perm:?}");
        }
    }
}

#[test]
fn root_step_is_legal_only_when_a_type_fits_the_budget() {
    let f = fixture();
    let start = Token {
        kind: 1,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    for case in f["root_budget_cases"]
        .as_array()
        .expect("root_budget_cases")
    {
        let budget = composition_of_formula(&case["budget"]);
        let mut state = TraceState::new(Limits::V0, Some(budget));
        state.apply(start).expect("START applies");
        let masks = state.masks(Token {
            kind: 2,
            atom_type: 1,
            bond: 0,
            pointer: 0,
        });
        assert_eq!(
            masks.kinds,
            as_u32(&case["kinds"]),
            "{} kinds",
            case["budget"]
        );
        assert_eq!(
            masks.atom_types,
            as_u32(&case["atom_types"]),
            "{} types",
            case["budget"]
        );
        assert_eq!(
            state.has_legal_action(),
            masks.kinds != 0,
            "{}",
            case["budget"]
        );
    }
}

#[test]
fn invalid_traces_rejected() {
    let f = fixture();
    for entry in f["invalid_traces"].as_array().expect("invalid_traces") {
        let name = entry["name"].as_str().unwrap();
        let trace = trace_of(&entry["trace"]);
        let expected = entry["first_illegal_step"].as_u64().unwrap() as usize;
        assert_eq!(
            first_illegal_step(&trace, Limits::V0, None),
            Some(expected),
            "{name}"
        );
        assert!(
            replay(&trace, Limits::V0, None).is_err(),
            "{name} replay errors"
        );
    }
}

#[test]
fn graph_edge_cases() {
    // Duplicate bond in either orientation.
    assert!(MolGraph::new(vec![3, 3], vec![(0, 1, 1), (1, 0, 1)]).is_err());
    // Self-bond.
    assert!(MolGraph::new(vec![1], vec![(0, 0, 1)]).is_err());
    // Five single bonds on a C H0 (capacity 4).
    assert!(
        MolGraph::new(
            vec![1, 4, 4, 4, 4, 4],
            vec![(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1)]
        )
        .is_err()
    );
    // Bond order 4.
    assert!(MolGraph::new(vec![1, 1], vec![(0, 1, 4)]).is_err());
    // Disconnected graphs build but do not canonicalize.
    let split = MolGraph::new(vec![4, 4], vec![]).expect("disconnected builds");
    assert!(!split.is_connected());
    assert!(canonical_trace(&split, Limits::V0, CANONICAL_WORK_LIMIT).is_err());
    // An incomplete trace (no STOP) replays without stopping.
    let open = replay(
        &[
            Token {
                kind: 1,
                atom_type: 0,
                bond: 0,
                pointer: 0,
            },
            Token {
                kind: 2,
                atom_type: 1,
                bond: 0,
                pointer: 0,
            },
        ],
        Limits::V0,
        None,
    )
    .expect("prefix replays");
    assert!(!open.stopped());
    // A 17-atom chain exceeds the atom limit.
    let mut chain_types = vec![4u8];
    chain_types.extend(std::iter::repeat_n(3u8, 15));
    chain_types.push(4u8);
    let chain_bonds: Vec<(usize, usize, u8)> = (0..16).map(|i| (i, i + 1, 1)).collect();
    let chain = MolGraph::new(chain_types, chain_bonds).expect("chain builds");
    assert!(canonical_trace(&chain, Limits::V0, CANONICAL_WORK_LIMIT).is_err());
    // A hand-built 6-atom graph with 5 closures (10 bonds, 6 atoms).
    let bicyclic = MolGraph::new(
        vec![1, 1, 1, 1, 1, 1],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 0, 1),
            (0, 2, 1),
            (0, 3, 1),
            (1, 3, 1),
            (1, 4, 1),
        ],
    )
    .expect("5-closure graph builds");
    assert_eq!(bicyclic.ring_closures(), 5);
    assert!(canonical_trace(&bicyclic, Limits::V0, CANONICAL_WORK_LIMIT).is_err());
    // Same formula, different structures: 2-butanol vs isobutanol.
    let f = fixture();
    let molecules = f["molecules"].as_array().expect("molecules");
    let by_name = |want: &str| {
        molecules
            .iter()
            .find(|m| m["name"].as_str().unwrap() == want)
            .expect("named molecule")
    };
    let butanol = graph_of(by_name("2-butanol"));
    let isobutanol = graph_of(by_name("isobutanol"));
    assert_eq!(butanol.composition(), isobutanol.composition());
    let a = canonical_trace(&butanol, Limits::V0, CANONICAL_WORK_LIMIT).expect("canonicalizes");
    let b = canonical_trace(&isobutanol, Limits::V0, CANONICAL_WORK_LIMIT).expect("canonicalizes");
    assert_ne!(a.trace, b.trace);
}

#[test]
fn open_valence_reports_residuals() {
    let f = fixture();
    let molecules = f["molecules"].as_array().expect("molecules");
    let aspirin = molecules
        .iter()
        .find(|m| m["name"].as_str().unwrap() == "aspirin")
        .unwrap();
    let graph = graph_of(aspirin);
    let mut checked = false;
    for g in aspirin["subgraphs"].as_array().unwrap() {
        let Some(fixture_trace) = g.get("canonical_trace") else {
            continue;
        };
        let open: Vec<u64> = g["open_valence_canonical_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        if !open.contains(&2) {
            continue;
        }
        let trace = trace_of(fixture_trace);
        let state = replay(&trace, Limits::V0, None).expect("trace replays");
        let residual: Vec<u64> = state
            .residual_valence()
            .iter()
            .map(|v| u64::from(*v))
            .collect();
        assert_eq!(residual, open);
        assert!(residual.contains(&2));
        let boundary = g["boundary"].as_u64().unwrap();
        assert!(residual.iter().sum::<u64>() >= boundary);
        // Hydrogens are never added to cap an open valence: the composition
        // hydrogen count is exactly the atom types' own hydrogens.
        let members: Vec<usize> = g["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_u64().unwrap() as usize)
            .collect();
        let sub = graph.induced(&members).unwrap();
        let own_hydrogens: u16 = sub
            .atoms()
            .iter()
            .map(|id| u16::from(atom_type_of_hydrogens(*id)))
            .sum();
        assert_eq!(sub.composition()[HYDROGEN], own_hydrogens);
        checked = true;
        break;
    }
    assert!(checked, "aspirin subgraph with residual 2");
}

fn atom_type_of_hydrogens(id: u8) -> u8 {
    atom_type(id).expect("known type").hydrogens
}

#[test]
fn canonical_work_limit_triggers() {
    let f = fixture();
    let molecules = f["molecules"].as_array().expect("molecules");
    let naphthalene = molecules
        .iter()
        .find(|m| m["name"].as_str().unwrap() == "naphthalene")
        .unwrap();
    let graph = graph_of(naphthalene);
    assert_eq!(graph.atoms().len(), 10);
    let err = canonical_trace(&graph, Limits::V0, 3).expect_err("tiny budget fails");
    assert!(
        err.to_string().contains("canonicalization_budget_exceeded"),
        "{err}"
    );
}

#[test]
fn ion_cases_match_fixture_with_decimal_interval() {
    let f = fixture();
    for case in f["ion_cases"].as_array().expect("ion_cases") {
        let composition = composition_of_formula(&case["formula"]);
        let adduct = case["adduct"].as_u64().unwrap() as u16;
        let shift = case["shift"].as_i64().unwrap() as i32;
        let found = ion(&composition, adduct, shift).expect("ion computes");
        if case.get("none").is_some() {
            assert!(found.is_none(), "{case} has no ion");
            continue;
        }
        let hyp = found.expect("ion exists");
        assert_eq!(hyp.mz, as_u32(&case["mz_udalton"]), "{case} m/z");
        assert_eq!(hyp.error, as_u32(&case["error_udalton"]), "{case} error");
        // The reported interval covers the exact decimal mass in nano-units.
        let floor = case["exact_nda_floor"].as_u64().unwrap();
        let ceil = case["exact_nda_ceil"].as_u64().unwrap();
        let lo = u64::from(hyp.mz) * 1000 - u64::from(hyp.error) * 1000;
        let hi = u64::from(hyp.mz) * 1000 + u64::from(hyp.error) * 1000;
        assert!(lo <= floor, "{case} interval low");
        assert!(ceil <= hi, "{case} interval high");
    }
}

#[test]
fn ion_and_parent_mass_errors() {
    let f = fixture();
    for case in f["ion_errors"].as_array().expect("ion_errors") {
        let composition = composition_of_formula(&case["formula"]);
        let adduct = case["adduct"].as_u64().unwrap() as u16;
        let shift = case["shift"].as_i64().unwrap() as i32;
        let err = match ion(&composition, adduct, shift) {
            Err(err) => err,
            Ok(_) => panic!("{case} ion fails"),
        };
        let want = case["error"].as_str().unwrap();
        assert!(err.to_string().contains(want), "{case}: {err}");
    }
    for case in f["parent_mass_errors"]
        .as_array()
        .expect("parent_mass_errors")
    {
        let precursor = as_u32(&case["precursor_mz_udalton"]);
        let adduct = case["adduct"].as_u64().unwrap() as u16;
        assert!(
            parent_mass(precursor, adduct).is_err(),
            "{case} parent mass fails"
        );
    }
}

#[test]
fn whole_traces_replay_with_fixture_masks() {
    let f = fixture();
    let mut molecules = 0usize;
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let Some(whole) = m.get("whole_trace") else {
            continue;
        };
        molecules += 1;
        let graph = graph_of(m);
        let trace = trace_of(&whole["trace"]);
        let budget = graph.composition();
        // The trace replays legally, with the fixture masks at every step
        // under both budgets and the fixture residual valences at its end.
        for (key, use_budget) in [
            ("legal_masks_parent_budget", true),
            ("legal_masks_no_budget", false),
        ] {
            let mut state = TraceState::new(Limits::V0, use_budget.then_some(budget));
            let expected_masks = whole[key].as_array().expect("masks");
            assert_eq!(expected_masks.len(), trace.len(), "{name} {key} steps");
            for (i, token) in trace.iter().enumerate() {
                let masks = state.masks(*token);
                let expected = expected_masks[i].as_array().expect("mask row");
                assert_eq!(
                    masks.kinds,
                    as_u32(&expected[0]),
                    "{name} step {i} kinds {key}"
                );
                assert_eq!(
                    masks.atom_types,
                    as_u32(&expected[1]),
                    "{name} step {i} types {key}"
                );
                assert_eq!(
                    masks.bonds,
                    as_u32(&expected[2]),
                    "{name} step {i} bonds {key}"
                );
                assert_eq!(
                    masks.pointers,
                    as_u32(&expected[3]),
                    "{name} step {i} ptrs {key}"
                );
                assert!(state.is_legal(*token), "{name} step {i} legal {key}");
                state.apply(*token).expect("trace applies");
            }
        }
        let state = replay(&trace, Limits::V0, None).expect("trace replays");
        assert!(state.stopped(), "{name} whole trace stops");
        let open: Vec<u64> = whole["open_valence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let residual: Vec<u64> = state
            .residual_valence()
            .iter()
            .map(|v| u64::from(*v))
            .collect();
        assert_eq!(residual, open, "{name} open valence");
        if name == "pyrene" {
            assert_eq!(graph.atoms().len(), 16, "pyrene atoms");
            assert_eq!(trace.len(), Limits::V0.max_steps(), "pyrene longest trace");
            assert_eq!(
                trace.iter().filter(|t| t.kind == CLOSE_RING).count(),
                4,
                "pyrene closures"
            );
            // After the fourth CLOSE_RING the closure budget is spent: no
            // later step offers CLOSE_RING.
            let mut state = TraceState::new(Limits::V0, None);
            let mut closes = 0usize;
            for (i, token) in trace.iter().enumerate() {
                if closes >= 4 {
                    assert_eq!(
                        state.masks(*token).kinds & (1u32 << CLOSE_RING),
                        0,
                        "pyrene step {i}: no CLOSE_RING past the fourth"
                    );
                }
                assert!(state.is_legal(*token), "pyrene step {i} legal");
                if token.kind == CLOSE_RING {
                    closes += 1;
                }
                state.apply(*token).expect("trace applies");
            }
        }
    }
    assert!(molecules > 20, "{molecules} whole traces");
}

#[test]
fn stereo_diol_has_32_identity_classes() {
    use std::collections::BTreeMap;
    let f = fixture();
    let molecules = f["molecules"].as_array().expect("molecules");
    let diol = molecules
        .iter()
        .find(|m| m["name"].as_str().unwrap() == "stereo diol")
        .expect("stereo diol");
    // Stereochemistry is ignored by identity: 32 classes, not 33.
    assert_eq!(diol["identity_classes"].as_u64().unwrap(), 32);
    let graph = graph_of(diol);
    let mut by_trace: BTreeMap<Vec<Token>, Vec<u64>> = BTreeMap::new();
    for g in diol["subgraphs"].as_array().expect("subgraphs") {
        let members: Vec<usize> = g["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_u64().unwrap() as usize)
            .collect();
        let sub = graph.induced(&members).expect("induced subgraph");
        let found = canonical_trace(&sub, Limits::V0, CANONICAL_WORK_LIMIT).expect("canonicalizes");
        by_trace
            .entry(found.trace)
            .or_default()
            .push(g["class"].as_u64().unwrap());
    }
    assert_eq!(by_trace.len(), 32, "stereo diol class count");
}

#[test]
fn graph_and_limit_sizes_rejected() {
    // 4,097 atoms exceed the 4,096-atom host limit; 4,096 build fine.
    assert!(MolGraph::new(vec![4u8; 4097], vec![]).is_err());
    assert!(MolGraph::new(vec![4u8; 4096], vec![]).is_ok());
    assert_eq!(MAX_GRAPH_ATOMS, 4096);
    assert!(Limits::new(0, 4).is_err());
    assert!(Limits::new(33, 4).is_err());
    assert!(Limits::new(16, 33).is_err());
    assert!(Limits::new(1, 0).is_ok());
    assert_eq!(Limits::V0.max_atoms(), 16);
    assert_eq!(Limits::V0.max_closures(), 4);
}
