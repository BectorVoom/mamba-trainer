//! Integration tests for the strict `molecular-completion-request-v1`
//! request layer (`src/models/ms2/completion_request.rs`). Does not
//! touch the prior bounded audit tests or expectations.

use mamba3::models::ms2::completion_request::molecular_completion_run;

const EXAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/molecular_completion_request.json"
);

fn example() -> serde_json::Value {
    let text = std::fs::read_to_string(EXAMPLE).expect("example fixture readable");
    serde_json::from_str(&text).expect("example fixture parses")
}

fn run_value(doc: &serde_json::Value) -> Result<serde_json::Value, String> {
    molecular_completion_run(&doc.to_string())
        .map(|s| serde_json::from_str(&s).expect("response is JSON"))
        .map_err(|e| e.to_string())
}

#[test]
fn normal_ethanol_fixture_completes_with_bounded_shape() {
    let out = run_value(&example()).expect("example accepted");
    assert_eq!(out["protocol"], "molecular-completion-request-v1");
    assert_eq!(out["query_id"], "ethanol_target_molecule");
    assert!(out["input_hash"].as_str().unwrap().len() == 16);
    assert_eq!(out["ranking"]["status"], "not_evaluated");
    assert_eq!(
        out["physical_verification"]["status"], "not_evaluated"
    );
    assert!(out["identity_ordering"]
        .as_str()
        .unwrap()
        .contains("not calibrated confidence"));
    let audit = &out["audit"];
    assert_eq!(audit["protocol"], "completion-bounded-v1");
    assert_eq!(audit["status"], "complete");
    assert_eq!(audit["mass_status"], "accepted");
    assert_eq!(audit["unique_graphs"], 1);
    assert_eq!(audit["accepted_formulas"], serde_json::json!(["C2H6O"]));
    assert_eq!(audit["precursor"]["status"], "not_evaluated");
    assert_eq!(audit["certifies_zero"], false);
    // Decoded typed graph/trace from the existing canonical identity.
    let decoded = out["decoded_graphs"].as_array().unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0]["composition"], "C2H6O");
    assert_eq!(decoded[0]["atoms"].as_array().unwrap().len(), 3);
    assert!(decoded[0]["bonds"].as_array().unwrap().len() == 2);
}

#[test]
fn mass_role_is_required_and_fragment_roles_are_actionably_rejected() {
    let mut doc = example();
    doc.as_object_mut().unwrap().remove("mass_role");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("mass_role"), "{err}");

    for role in ["target_fragment", "neutral_loss"] {
        let mut doc = example();
        doc["mass_role"] = serde_json::json!(role);
        let err = run_value(&doc).unwrap_err();
        assert!(err.contains(role), "{err}");
        assert!(err.contains("precursor"), "{err}");
    }
}

#[test]
fn neutralization_convention_is_required_and_enforced() {
    let mut doc = example();
    doc.as_object_mut().unwrap().remove("neutralization");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("neutralization"), "{err}");

    let mut doc = example();
    doc["neutralization"] = serde_json::json!("protonated");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("already_neutral"), "{err}");
}

#[test]
fn unknown_precision_is_unavailable_never_zero() {
    let mut doc = example();
    doc["target_mass"]["uncertainty_uda"] = serde_json::Value::Null;
    let out = run_value(&doc).expect("null uncertainty is allowed");
    assert_eq!(out["audit"]["mass_status"], "unavailable");
    assert_eq!(out["audit"]["certifies_zero"], false);
    assert_eq!(out["audit"]["status"], "mass_evidence_unresolved");
}

#[test]
fn absent_correspondence_is_unknown_empty_is_known_disjoint() {
    // Two free CH3 patterns: unknown overlap keeps both ethanol and
    // dimethyl ether; a supplied empty correspondence (known disjoint)
    // keeps only dimethyl ether. `None` vs `Some([])` must differ.
    let mut doc = example();
    doc["substructures"] = serde_json::json!([
        {"atoms": [4], "bonds": [], "parent_hydrogen_semantics": "v0_parent_hydrogen_counts", "provenance": "synthetic", "certainty": "confirmed"},
        {"atoms": [4], "bonds": [], "parent_hydrogen_semantics": "v0_parent_hydrogen_counts", "provenance": "synthetic", "certainty": "confirmed"}
    ]);
    doc.as_object_mut().unwrap().remove("correspondence");
    let unknown = run_value(&doc).unwrap();
    assert_eq!(unknown["audit"]["status"], "complete");
    assert_eq!(unknown["audit"]["unique_graphs"], 2);

    doc["correspondence"] = serde_json::json!([]);
    let known_disjoint = run_value(&doc).unwrap();
    assert_eq!(known_disjoint["audit"]["status"], "complete");
    assert_eq!(known_disjoint["audit"]["unique_graphs"], 1);

    // Null is neither: it is rejected rather than guessed.
    doc["correspondence"] = serde_json::Value::Null;
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("correspondence"), "{err}");
}

#[test]
fn strict_types_and_unknown_fields_are_rejected() {
    let mut doc = example();
    doc["target_mass"]["value"] = serde_json::json!("46041865");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("target_mass.value"), "{err}");

    let mut doc = example();
    doc["target_mass"]["ppm_tenths"] = serde_json::json!(10.5);
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("ppm_tenths"), "{err}");

    let mut doc = example();
    doc["domain"]["elements"] = serde_json::json!("CNO");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("elements"), "{err}");

    let mut doc = example();
    doc["surprise"] = serde_json::json!(1);
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("unknown field 'surprise'"), "{err}");

    let mut doc = example();
    doc["substructures"][0]["weight"] = serde_json::json!(0.5);
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("unknown field 'weight'"), "{err}");

    let mut doc = example();
    doc["budgets"]["retained_graphs"] = serde_json::json!("many");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("retained_graphs"), "{err}");
}

#[test]
fn unknown_parent_hydrogen_semantics_and_tentative_patterns_are_rejected() {
    let mut doc = example();
    doc["substructures"][0]["parent_hydrogen_semantics"] = serde_json::json!("unknown");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("parent_hydrogen_semantics"), "{err}");

    let mut doc = example();
    doc["substructures"][0]
        .as_object_mut()
        .unwrap()
        .remove("parent_hydrogen_semantics");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("parent_hydrogen_semantics"), "{err}");

    let mut doc = example();
    doc["substructures"][0]["certainty"] = serde_json::json!("tentative");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("tentative"), "{err}");

    let mut doc = example();
    doc["substructures"][0]["certainty"] = serde_json::json!("unknown");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("certainty"), "{err}");
}

#[test]
fn provenance_and_id_are_required_nonempty() {
    let mut doc = example();
    doc["provenance"] = serde_json::json!("");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("provenance"), "{err}");

    let mut doc = example();
    doc["id"] = serde_json::json!("");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("id"), "{err}");

    let mut doc = example();
    doc["substructures"][0]
        .as_object_mut()
        .unwrap()
        .remove("provenance");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("provenance"), "{err}");
}

#[test]
fn any_supplyed_precursor_mass_is_rejected_not_ignored() {
    let mut doc = example();
    doc["precursor"] = serde_json::json!({"mass_uda": 92083730});
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("precursor"), "{err}");

    // Explicit null means "no precursor" and stays accepted.
    let mut doc = example();
    doc["precursor"] = serde_json::Value::Null;
    let out = run_value(&doc).expect("null precursor accepted");
    assert_eq!(out["audit"]["protocol"], "completion-bounded-v1");
}

#[test]
fn domain_is_pinned_to_cno_2_to_6_at_most_one_cycle() {
    let mut doc = example();
    doc["domain"]["version"] = serde_json::json!("completion-unbounded");
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("domain.version"), "{err}");

    let mut doc = example();
    doc["domain"]["elements"] = serde_json::json!(["C", "Cl"]);
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("C/N/O"), "{err}");

    let mut doc = example();
    doc["domain"]["max_heavy"] = serde_json::json!(7);
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("max_heavy"), "{err}");

    let mut doc = example();
    doc["domain"]["max_ring_closures"] = serde_json::json!(2);
    let err = run_value(&doc).unwrap_err();
    assert!(err.contains("max_ring_closures"), "{err}");
}

#[test]
fn exhausted_budget_surfaces_search_budget_exhausted_without_zero_certification() {
    let mut doc = example();
    doc["budgets"] = serde_json::json!({"graph_extensions": 8});
    let out = run_value(&doc).expect("budget-limited request still returns a report");
    assert_eq!(out["audit"]["status"], "search_budget_exhausted");
    assert!(out["audit"]["termination_reasons"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r == "graph_extension_limit"));
    assert_eq!(out["audit"]["certifies_zero"], false);
}

#[test]
fn deterministic_across_runs() {
    let a = molecular_completion_run(&example().to_string()).unwrap();
    let b = molecular_completion_run(&example().to_string()).unwrap();
    let mut a: serde_json::Value = serde_json::from_str(&a).unwrap();
    let mut b: serde_json::Value = serde_json::from_str(&b).unwrap();
    a["audit"]["elapsed_ms"].take();
    b["audit"]["elapsed_ms"].take();
    a["audit"]["counters"]["memory_estimate_bytes"].take();
    b["audit"]["counters"]["memory_estimate_bytes"].take();
    assert_eq!(a, b);
}
