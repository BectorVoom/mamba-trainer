//! Host reference tests for the bounded molecular-completion audit
//! (`docs/MOLECULAR_COMPLETION_EXPERIMENT.md`).
//!
//! Fixtures carry manually established completion sets; every test reads the
//! same fixture JSON the Python audit mirror uses. Budget assertions target
//! the counter names the supervision notes asked to exercise explicitly.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use mamba3::models::ms2::completion::{
    CompletionBudgets, CompletionDomain, CompletionQuery, MassEvidence, load_fixture_set, run,
};

fn fixtures() -> Vec<mamba3::models::ms2::completion::Fixture> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json");
    let text = fs::read_to_string(&path).expect("fixtures readable");
    load_fixture_set(&text).expect("fixtures parse")
}

fn fixture<'a>(name: &str) -> mamba3::models::ms2::completion::Fixture {
    fixtures().into_iter().find(|f| f.name == name).expect(name)
}

fn evaluate(name: &str) -> mamba3::models::ms2::completion::CompletionReport {
    let f = fixture(name);
    run(&f.query, &CompletionBudgets::default(), 1).expect("audit ok")
}

#[test]
fn fixture_expectations_hold() {
    for f in fixtures() {
        let report = run(&f.query, &CompletionBudgets::default(), 1).expect("audit ok");
        assert_eq!(report.status, f.expected.status, "{} status", f.name);
        assert_eq!(
            report.mass_status, f.expected.mass_status,
            "{} mass_status",
            f.name
        );
        assert_eq!(
            report.unique_graphs, f.expected.unique_graphs,
            "{} graphs",
            f.name
        );
        assert_eq!(
            report.accepted_formulas, f.expected.accepted_formulas,
            "{} formulas",
            f.name
        );
        assert_eq!(
            report.ambiguous_formulas, f.expected.ambiguous_formulas,
            "{} ambiguous",
            f.name
        );
        assert_eq!(
            report.certifies_zero, f.expected.certifies_zero,
            "{} certifies_zero",
            f.name
        );
        for term in &f.expected.termination_reasons {
            assert!(
                report.termination_reasons.contains(term),
                "{} reasons {:?}",
                f.name,
                report.termination_reasons
            );
        }
        match (f.expected.recovery, report.recovery) {
            (Some(a), Some(b)) => assert_eq!(a, b, "{} recovery", f.name),
            (None, _) => {}
            _ => panic!("{} missing recovery rendering", f.name),
        }
    }
}

#[test]
fn unresolved_and_unavailable_mass_never_certify_zero() {
    for name in [
        "c2h6o_mass_boundary_ambiguous",
        "c2h6o_precision_unavailable",
    ] {
        let r = evaluate(name);
        assert_eq!(r.status, "mass_evidence_unresolved", "{name}");
        assert!(!r.certifies_zero, "{name}");
    }
}

#[test]
fn rejected_mass_certifies_zero_only_within_domain() {
    let r = evaluate("c2h6o_mass_far_off");
    assert_eq!(r.status, "complete");
    assert_eq!(r.unique_graphs, 0);
    assert!(r.certifies_zero);
}

#[test]
fn incompatible_pattern_is_complete_zero_and_certifies_within_domain() {
    let r = evaluate("c2h6o_incompatible_n_atom");
    assert_eq!(r.status, "complete");
    assert_eq!(r.unique_graphs, 0);
    assert!(r.certifies_zero);
}

#[test]
fn oracle_and_unknown_overlap_arms_agree_on_count() {
    let with_oracle = evaluate("c2h6o_oracle_overlap_shares_ch2");
    let unknown = evaluate("c2h6o_overlap_unknown");
    assert_eq!(with_oracle.status, "complete");
    assert_eq!(unknown.status, "complete");
    assert_eq!(with_oracle.unique_graphs, unknown.unique_graphs);
    assert_eq!(with_oracle.unique_graphs, 1);
}

#[test]
fn known_disjoint_oracle_excludes_unreported_overlap_and_backtracks() {
    let unknown = evaluate("c2h6o_two_methyls_unknown_overlap");
    let known = evaluate("c2h6o_two_methyls_known_disjoint");
    assert_eq!(unknown.status, "complete");
    assert_eq!(known.status, "complete");
    assert_eq!(unknown.unique_graphs, 2);
    assert_eq!(known.unique_graphs, 1);
    assert_eq!(known.recovery, Some(true));
    assert!(
        known
            .accepted_identities
            .iter()
            .all(|id| unknown.accepted_identities.contains(id))
    );
}

#[test]
fn empty_oracle_without_patterns_does_not_depend_on_reference_metadata() {
    let mut query = fixture("c2h6o_mass_only").query;
    query.correspondence = Some(mamba3::models::ms2::completion::Correspondence { pairs: vec![] });
    let with_reference = run(&query, &CompletionBudgets::default(), 1).unwrap();
    query.reference = None;
    let without_reference = run(&query, &CompletionBudgets::default(), 1).unwrap();
    assert_eq!(without_reference.status, "complete");
    assert_eq!(
        without_reference.unique_graphs,
        with_reference.unique_graphs
    );
    assert_eq!(without_reference.unique_graphs, 2);
}

#[test]
fn request_rejects_malformed_seed_and_budget_objects() {
    let text = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json"),
    )
    .unwrap();
    let fixtures: serde_json::Value = serde_json::from_str(&text).unwrap();
    let query = fixtures["fixtures"][0].clone();
    for seed in [
        serde_json::json!(-1),
        serde_json::json!("1"),
        serde_json::json!(0.5),
        serde_json::Value::Null,
    ] {
        let mut request = query.clone();
        request["seed"] = seed;
        assert!(mamba3::models::ms2::completion::run_request_json(&request.to_string()).is_err());
    }
    for budgets in [
        serde_json::json!("default"),
        serde_json::json!([]),
        serde_json::Value::Null,
    ] {
        let mut request = query.clone();
        request["budgets"] = budgets;
        assert!(mamba3::models::ms2::completion::run_request_json(&request.to_string()).is_err());
    }
}

#[test]
fn incompatible_correspondence_is_unsupported_input() {
    let r = evaluate("oracle_requires_incompatible_types");
    assert_eq!(r.status, "unsupported_input");
    assert!(!r.statuses.is_empty() && r.statuses.iter().all(|s| s == "unsupported_input"));
    assert_eq!(r.unique_graphs, 0);
    assert!(!r.certifies_zero);
}

#[test]
fn invalid_reports_never_claim_mass_accepted() {
    for f in fixtures() {
        let r = run(&f.query, &CompletionBudgets::default(), 1).expect("audit ok");
        if r.status == "unsupported_input" {
            assert_eq!(r.mass_status, "not_evaluated", "{}", f.name);
            assert_eq!(r.precursor.status, "not_evaluated", "{}", f.name);
            assert!(!r.certifies_zero, "{}", f.name);
        }
    }
    // Direct invalid query: same id/domain, out-of-range ppm.
    let mut q = fixture("c2h6o_mass_only").query;
    q.mass.ppm_tenths = 100_000;
    let r = run(&q, &CompletionBudgets::default(), 1).unwrap();
    assert_eq!(r.status, "unsupported_input");
    assert_eq!(r.mass_status, "not_evaluated");
    assert!(!r.certifies_zero);
}

#[test]
fn invalid_evidence_changes_input_hash() {
    let mut a = fixture("c2h6o_mass_only").query;
    let mut b = fixture("c2h6o_mass_only").query;
    // Same id, domain, provenance; different invalid mass evidence.
    a.observed_mass_uda = 46_041_865;
    b.observed_mass_uda = 46_041_900;
    a.mass.ppm_tenths = 100_000;
    b.mass.ppm_tenths = 100_001;
    let ra = run(&a, &CompletionBudgets::default(), 1).unwrap();
    let rb = run(&b, &CompletionBudgets::default(), 1).unwrap();
    assert_eq!(ra.status, "unsupported_input");
    assert_eq!(rb.status, "unsupported_input");
    assert_ne!(ra.input_hash, rb.input_hash);
    // And a different source tag alone changes the hash.
    let mut c = fixture("c2h6o_mass_only").query;
    c.mass.source = "instrument".to_string();
    c.mass.ppm_tenths = 100_000;
    let rc = run(&c, &CompletionBudgets::default(), 1).unwrap();
    let mut a2 = fixture("c2h6o_mass_only").query;
    a2.mass.ppm_tenths = 100_000;
    let ra2 = run(&a2, &CompletionBudgets::default(), 1).unwrap();
    assert_ne!(rc.input_hash, ra2.input_hash);
}

#[test]
fn memory_estimate_tracks_peak_not_last_reset() {
    // The reported peak estimate must cover every retained graph byte the
    // audit kept; it is never reset downward by later frontier/insert
    // bookkeeping.
    for name in [
        "c2h6o_mass_only",
        "c4h8_mass_only",
        "c3h4_mass_only_unsaturation",
    ] {
        let r = evaluate(name);
        assert!(
            r.counters.memory_estimate_bytes >= 256 * r.counters.retained_graphs,
            "{name}: estimate {} < 256 * retained {}",
            r.counters.memory_estimate_bytes,
            r.counters.retained_graphs
        );
        assert!(r.counters.retained_graphs > 0, "{name}");
        // Deterministic across identical runs.
        let r2 = evaluate(name);
        assert_eq!(
            r.counters.memory_estimate_bytes, r2.counters.memory_estimate_bytes,
            "{name}"
        );
    }
    // On a memory-bound abort the attempted estimate is reported, not
    // silently dropped: the recorded peak is at least the frontier that
    // triggered the bound.
    let r = limited("c2h6o_mass_only", |b| b.memory_bytes = 64);
    assert_eq!(r.status, "search_budget_exhausted");
    assert!(r.termination_reasons.contains(&"memory_bound".to_string()));
    assert!(r.counters.memory_estimate_bytes >= 64);
}

#[test]
fn symmetry_counts_graph_once_per_identity() {
    // dimethyl ether carries two embeddings of an isolated CH3 atom, yet is
    // one graph.
    let r = evaluate("c2h6o_ch3_symmetry");
    assert_eq!(r.unique_graphs, 2);
    assert_eq!(r.status, "complete");
}

#[test]
fn ring_fixture_separates_ring_from_acyclics() {
    let ring = evaluate("c3h6_ring_cyclopropane");
    assert_eq!(ring.unique_graphs, 1);
    let ring4 = evaluate("c4h8_square_ring_selects_cyclobutane");
    assert_eq!(ring4.unique_graphs, 1);
    let square_vs_chain = evaluate("c4h8_mass_only");
    assert_eq!(square_vs_chain.unique_graphs, 5);
}

#[test]
fn n_fixtures_hold() {
    assert_eq!(evaluate("c2h7n_mass_only").unique_graphs, 2);
    assert_eq!(evaluate("c2h7n_nh2_selects_ethylamine").unique_graphs, 1);
}

#[test]
fn unsaturation_fixtures() {
    assert_eq!(evaluate("c3h4_mass_only_unsaturation").unique_graphs, 3);
    assert_eq!(
        evaluate("c3h4_triple_bond_selects_propyne").unique_graphs,
        1
    );
}

/// Every returned identity decodes to a target whose composition is one of
/// the supplied accepted formula strings.
#[test]
fn every_filtered_identity_matches_an_accepted_formula() {
    use mamba3::models::ms2::grammar::{Limits, Token, replay};

    fn render_comp(mol: &mamba3::models::ms2::graph::MolGraph) -> String {
        let comp = mol.composition();
        let mut out = String::new();
        let part = |label: &str, n: u16, out: &mut String| {
            if n == 0 {
                return;
            }
            if n == 1 {
                out.push_str(label);
            } else {
                out.push_str(&format!("{label}{n}"));
            }
        };
        part("C", comp[0], &mut out);
        part("H", comp[1], &mut out);
        part("N", comp[2], &mut out);
        part("O", comp[3], &mut out);
        out
    }

    for name in [
        "c2h6o_mass_only",
        "c2h7n_mass_only",
        "c4h8_mass_only",
        "c3h4_mass_only_unsaturation",
        "c3h8o_mass_only",
    ] {
        let rep = evaluate(name);
        for id in &rep.accepted_identities {
            let mut toks = Vec::new();
            for part in id.split(',') {
                let mut fields = part.split('/');
                toks.push(Token {
                    kind: fields.next().unwrap().parse().unwrap(),
                    atom_type: fields.next().unwrap().parse().unwrap(),
                    bond: fields.next().unwrap().parse().unwrap(),
                    pointer: fields.next().unwrap().parse().unwrap(),
                });
            }
            let state = replay(&toks, Limits::V0, None).unwrap();
            let mol = state.graph().unwrap();
            let text = render_comp(&mol);
            assert!(
                rep.accepted_formulas.iter().any(|f| f == &text),
                "{name}: {text} not in {:?}",
                rep.accepted_formulas
            );
        }
    }
}

/// Direct canonical-form check: the accepted set equals ethanol plus
/// dimethyl ether typed exactly.
#[test]
fn c2h6o_exact_closure_set_is_ethanol_plus_dimethyl_ether() {
    use mamba3::models::ms2::grammar::{Limits, canonical_trace};
    use mamba3::models::ms2::graph::MolGraph;
    let rep = evaluate("c2h6o_mass_only");
    let render = |g: &MolGraph| {
        let c = canonical_trace(g, Limits::V0, 100_000).unwrap();
        c.trace
            .iter()
            .map(|t| format!("{}/{}/{}/{}", t.kind, t.atom_type, t.bond, t.pointer))
            .collect::<Vec<_>>()
            .join(",")
    };
    let ethanol = MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let dimethyl_ether = MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let mut got = rep.accepted_identities.clone();
    got.sort();
    let mut want = vec![render(&ethanol), render(&dimethyl_ether)];
    want.sort();
    assert_eq!(got, want);
}

/// Direct canonical-form check for the N domain: ethylamine plus
/// dimethylamine.
#[test]
fn c2h7n_exact_closure_set_is_ethylamine_plus_dimethylamine() {
    use mamba3::models::ms2::grammar::{Limits, canonical_trace};
    use mamba3::models::ms2::graph::MolGraph;
    let rep = evaluate("c2h7n_mass_only");
    let render = |g: &MolGraph| {
        let c = canonical_trace(g, Limits::V0, 100_000).unwrap();
        c.trace
            .iter()
            .map(|t| format!("{}/{}/{}/{}", t.kind, t.atom_type, t.bond, t.pointer))
            .collect::<Vec<_>>()
            .join(",")
    };
    let ethylamine = MolGraph::new(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let dimethylamine = MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let mut got = rep.accepted_identities.clone();
    got.sort();
    let mut want = vec![render(&ethylamine), render(&dimethylamine)];
    want.sort();
    assert_eq!(got, want);
}

/// Reordering the atom indices of a supplied pattern set must not change
/// the count: typed embeddings are reads of the pattern graph, not of its
/// indexing.
#[test]
fn pattern_index_reorder_preserves_the_count() {
    let text = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json"),
    )
    .unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let fixtures = value["fixtures"].as_array_mut().unwrap();
    for f in fixtures.iter_mut() {
        if f["name"].as_str().unwrap() != "c2h6o_ch2oh_bond" {
            continue;
        }
        f["patterns"][0]["atoms"] = serde_json::json!([9, 3]);
        f["patterns"][0]["bonds"] = serde_json::json!([[0, 1, 1]]);
    }
    let text = serde_json::to_string(&value).unwrap();
    let mut list = load_fixture_set(&text).unwrap();
    let f = list.remove(
        list.iter()
            .position(|f| f.name == "c2h6o_ch2oh_bond")
            .unwrap(),
    );
    let swapped = run(&f.query, &CompletionBudgets::default(), 1).unwrap();
    let rep = evaluate("c2h6o_ch2oh_bond");
    assert_eq!(swapped.unique_graphs, rep.unique_graphs);
    assert_eq!(swapped.status, rep.status);
}

#[test]
fn loader_rejects_malformed_input_without_panicking() {
    let bad = [
        // bond has only two fields
        r#"{"fixtures":[{"name":"x","observed_mass_uda":0,"mass":{},"domain":{"elements":["C"]},"patterns":[{"atoms":[4,9],"bonds":[[0,1]]}]}]}"#,
        // atom id too large for u8
        r#"{"fixtures":[{"name":"x","observed_mass_uda":0,"mass":{},"domain":{"elements":["C"]},"patterns":[{"atoms":[0,1,2,3,4,65536],"bonds":[]}]}]}"#,
        // observed mass overflowing u32
        r#"{"fixtures":[{"name":"x","observed_mass_uda":4294967296,"mass":{},"domain":{"elements":["C"]},"patterns":[]}]}"#,
        // ppm as float
        r#"{"fixtures":[{"name":"x","observed_mass_uda":0,"mass":{"ppm_tenths":10.5,"uncertainty_uda":50},"domain":{"elements":["C"]},"patterns":[]}]}"#,
        // uncertainty as string
        r#"{"fixtures":[{"name":"x","observed_mass_uda":0,"mass":{"uncertainty_uda":"50"},"domain":{"elements":["C"]},"patterns":[]}]}"#,
        // correspondence with single endpoint
        r#"{"fixtures":[{"name":"x","observed_mass_uda":0,"mass":{},"domain":{"elements":["C"]},"patterns":[{"atoms":[4],"bonds":[]}],"correspondence":[[[0,0]]]}]}"#,
    ];
    for text in bad {
        assert!(
            load_fixture_set(text).is_err(),
            "expected error for: {text}"
        );
    }
}

fn limited(
    name: &str,
    mutate: impl Fn(&mut CompletionBudgets),
) -> mamba3::models::ms2::completion::CompletionReport {
    let f = fixture(name);
    let mut budgets = CompletionBudgets::default();
    mutate(&mut budgets);
    run(&f.query, &budgets, 1).expect("audit ok")
}

#[test]
fn formula_visit_limit() {
    let r = limited("c2h6o_mass_only", |b| b.formula_visits = 4);
    assert_eq!(r.status, "search_budget_exhausted");
    assert!(
        r.termination_reasons
            .contains(&"formula_visit_limit".to_string())
    );
}

#[test]
fn graph_extension_limit() {
    let r = limited("c2h6o_mass_only", |b| b.graph_extensions = 8);
    assert_eq!(r.status, "search_budget_exhausted");
    assert!(
        r.termination_reasons
            .contains(&"graph_extension_limit".to_string())
    );
}

#[test]
fn embedding_node_limit() {
    let r = limited("c2h6o_ch2oh_bond", |b| b.embedding_nodes = 1);
    assert_eq!(r.status, "search_budget_exhausted");
    assert!(
        r.termination_reasons
            .contains(&"embedding_node_limit".to_string())
    );
}

#[test]
fn canonicalization_limit() {
    let r = limited("c2h6o_mass_only", |b| b.canonical_expansions = 1);
    assert_eq!(r.status, "search_budget_exhausted");
    assert_eq!(r.recovery, None);
    assert_eq!(r.counters.canonical_expansions, 2);
    assert!(
        r.termination_reasons
            .contains(&"canonicalization_limit".to_string())
    );
}

#[test]
fn retained_graph_limit_keeps_lower_bound() {
    let r = limited("c2h6o_mass_only", |b| b.retained_graphs = 1);
    assert_eq!(r.status, "search_budget_exhausted");
    assert!(
        r.termination_reasons
            .contains(&"retained_graph_limit".to_string())
    );
    assert_eq!(r.unique_graphs, 1);
    assert!(!r.certifies_zero);
}

#[test]
fn memory_bound_explicit_reason() {
    let r = limited("c2h6o_mass_only", |b| b.memory_bytes = 1);
    assert_eq!(r.status, "search_budget_exhausted");
    assert!(r.termination_reasons.contains(&"memory_bound".to_string()));
    assert_eq!(r.unique_graphs, 0);
    assert!(!r.certifies_zero);
}

#[test]
fn watchdog_zero_truncates_any_query() {
    let r = limited("c2h6o_mass_only", |b| b.watchdog = Duration::ZERO);
    assert_eq!(r.status, "search_budget_exhausted");
    assert!(r.termination_reasons.iter().any(|s| s == "watchdog"));
    assert!(!r.certifies_zero);

    // Zero-watchdog neither helps rejection nor unknown mass: both reasons appear
    // and no zero is certified.
    let rejected = limited("c2h6o_mass_far_off", |b| b.watchdog = Duration::ZERO);
    assert_eq!(rejected.status, "search_budget_exhausted");
    assert!(rejected.termination_reasons.iter().any(|s| s == "watchdog"));
    assert!(!rejected.certifies_zero);
    let unknown = limited("c2h6o_precision_unavailable", |b| {
        b.watchdog = Duration::ZERO
    });
    assert_eq!(unknown.status, "search_budget_exhausted");
    assert!(
        unknown
            .statuses
            .iter()
            .any(|s| s == "mass_evidence_unresolved")
    );
    assert!(!unknown.certifies_zero);
}

#[test]
fn results_are_identical_across_runs() {
    let mut a = serde_json::to_value(evaluate("c2h6o_mass_only")).unwrap();
    let mut b = serde_json::to_value(evaluate("c2h6o_mass_only")).unwrap();
    a["counters"]["memory_estimate_bytes"].take();
    b["counters"]["memory_estimate_bytes"].take();
    a["elapsed_ms"].take();
    b["elapsed_ms"].take();
    assert_eq!(a, b);
}

#[test]
fn adding_constraints_never_increases_counts() {
    let observed_constraints = [
        ("c2h6o_mass_only", "c2h6o_ch2oh_bond"),
        ("c2h6o_mass_only", "c2h6o_ch3_symmetry"),
        ("c3h8o_mass_only", "c3h8o_disjoint_ch3_and_ch2oh"),
        (
            "c3h4_mass_only_unsaturation",
            "c3h4_triple_bond_selects_propyne",
        ),
        ("c4h8_mass_only", "c4h8_square_ring_selects_cyclobutane"),
    ];
    for (mass_only, constrained) in observed_constraints {
        let wide = evaluate(mass_only);
        let narrow = evaluate(constrained);
        assert_eq!(wide.status, "complete");
        assert_eq!(narrow.status, "complete");
        assert!(wide.unique_graphs >= narrow.unique_graphs);
    }
}

#[test]
fn standard_domain_excludes_out_of_scope_graphs() {
    // the restricted 2..4 domain on C4H8 finds 5 graphs; the acyclic-only
    // text does not: using rings<=0 is unavailable via CompletionDomain but
    // the predetermined domain equality on C4H8 supports 5.
    let r = evaluate("c4h8_mass_only");
    assert!(r.unique_graphs >= 5);
}

#[test]
fn domain_validation_rejects() {
    assert!(CompletionDomain::new(&[], 2, 6, 1).is_err());
    assert!(CompletionDomain::new(&[1], 2, 6, 1).is_err()); // Cl
    assert!(CompletionDomain::new(&[0, 0, 2], 2, 6, 1).is_err());
    assert!(CompletionDomain::new(&[0, 2, 3], 0, 6, 1).is_err());
    assert!(CompletionDomain::new(&[0, 2, 3], 3, 2, 1).is_err());

    let mass = MassEvidence {
        ppm_tenths: 100,
        uncertainty_uda: Some(50),
        source: "synthetic".to_string(),
    };
    let domain = CompletionDomain::new(&[0, 2, 3], 2, 6, 1).unwrap();
    let q = CompletionQuery {
        id: "bad_ppm".to_string(),
        provenance: "synthetic".to_string(),
        observed_mass_uda: 1_000_000,
        mass: MassEvidence {
            ppm_tenths: 100_000,
            ..mass
        },
        substructures: Vec::new(),
        correspondence: None,
        domain,
        reference: None,
    };
    let r = run(&q, &CompletionBudgets::default(), 1).unwrap();
    assert_eq!(r.status, "unsupported_input");
}

#[test]
fn watchdog_overflow_duration_is_rejected_explicitly() {
    let f = fixture("c2h6o_mass_only");
    let mut budgets = CompletionBudgets::default();
    budgets.watchdog = Duration::MAX;
    let r = run(&f.query, &budgets, 1);
    match r {
        Err(e) => assert!(e.to_string().contains("overflow") || e.to_string().contains("watchdog")),
        Ok(_) => panic!("watchdog overflow must not produce a report"),
    }
}

/// The independent Python mirror is run over the same fixture file; every
/// fixture that the Python brute-force enumerator reaches must agree with
/// the Rust report on the semantic fields. Giant-domain fixtures keep the
/// Python mirror out-of-scope and are validated by the Rust report alone.
#[test]
fn rust_report_matches_python_mirror_semantics() {
    let text = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json"),
    )
    .unwrap();
    let fixtures = load_fixture_set(&text).unwrap();
    let out = std::process::Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/ms2_completion_audit.py"
        ))
        .output()
        .expect("python3 is required for the mirror parity test");
    assert!(
        out.status.success(),
        "python mirror failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut report: serde_json::Value = serde_json::from_slice(&out.stdout).expect("python json");
    let rows = report["fixtures"].as_array_mut().unwrap();
    let mut compared = 0;
    for row in rows {
        if row.get("match").and_then(|m| m.as_bool()).is_none() {
            continue;
        }
        compared += 1;
        let name = row["name"].as_str().unwrap();
        let fixture = fixtures.iter().find(|f| f.name == name).unwrap();
        let report = run(&fixture.query, &CompletionBudgets::default(), 1).unwrap();
        assert_eq!(
            row["got"]["status"].as_str().unwrap(),
            report.status,
            "{name}"
        );
        assert_eq!(
            row["got"]["mass_status"].as_str().unwrap(),
            report.mass_status,
            "{name}"
        );
        assert_eq!(
            row["got"]["unique_graphs"].as_u64().unwrap() as usize,
            report.unique_graphs,
            "{name}"
        );
        assert_eq!(
            row["got"]["certifies_zero"].as_bool().unwrap(),
            report.certifies_zero,
            "{name}"
        );
    }
    assert!(compared >= 20, "only {compared} python comparisons");
}
