//! Integration tests for the trained-model generation API
//! (`molecular-completion-generate-v1`, `src/models/ms2/completion_api.rs`).
//! Shares the committed fixture with the Python suite.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{self, CHEMISTRY_VERSION};
use mamba3::models::ms2::completion::contains_pattern;
use mamba3::models::ms2::completion_api::{CompletionService, GENERATE_PROTOCOL};
use mamba3::models::ms2::completion_data::COMPLETION_DATA_VERSION;
use mamba3::models::ms2::completion_model::{COMPLETION_MODEL_VERSION, PATTERN_SLOTS};
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::grammar::COMPLETION_GRAMMAR_VERSION;
use mamba3::models::ms2::graph::MolGraph;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn ckpt_path() -> String {
    format!(
        "{}/tests/fixtures/ms2/completion_tiny.ckpt",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn requests_path() -> String {
    format!(
        "{}/tests/fixtures/ms2/completion_tiny_requests.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn responses_path() -> String {
    format!(
        "{}/tests/fixtures/ms2/completion_tiny_responses.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn load_fixture() -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let reqs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(requests_path()).unwrap()).unwrap();
    let resps: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(responses_path()).unwrap()).unwrap();
    (
        reqs.as_array().unwrap().clone(),
        resps.as_array().unwrap().clone(),
    )
}

fn service() -> CompletionService<R> {
    CompletionService::load(std::path::Path::new(&ckpt_path()), &dev()).unwrap()
}

/// Compare JSON with exact integers/strings and 1e-5 tolerance for floats.
fn assert_json_close(a: &serde_json::Value, b: &serde_json::Value, path: &str) {
    match (a, b) {
        (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
            if let (Some(xi), Some(yi)) = (x.as_u64(), y.as_u64()) {
                assert_eq!(xi, yi, "{path}: integer mismatch");
            } else if let (Some(xi), Some(yi)) = (x.as_i64(), y.as_i64()) {
                assert_eq!(xi, yi, "{path}: integer mismatch");
            } else {
                let xf = x.as_f64().unwrap();
                let yf = y.as_f64().unwrap();
                assert!(
                    (xf - yf).abs() <= 1e-5,
                    "{path}: float {xf} vs {yf} beyond 1e-5"
                );
            }
        }
        (serde_json::Value::String(x), serde_json::Value::String(y)) => {
            assert_eq!(x, y, "{path}: string mismatch");
        }
        (serde_json::Value::Bool(x), serde_json::Value::Bool(y)) => {
            assert_eq!(x, y, "{path}: bool mismatch");
        }
        (serde_json::Value::Null, serde_json::Value::Null) => {}
        (serde_json::Value::Array(x), serde_json::Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: array length");
            for (i, (xi, yi)) in x.iter().zip(y.iter()).enumerate() {
                assert_json_close(xi, yi, &format!("{path}[{i}]"));
            }
        }
        (serde_json::Value::Object(x), serde_json::Value::Object(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: object size");
            for (k, xv) in x.iter() {
                let yv = y
                    .get(k)
                    .unwrap_or_else(|| panic!("{path}: missing key '{k}'"));
                assert_json_close(xv, yv, &format!("{path}.{k}"));
            }
        }
        _ => panic!("{path}: type mismatch {a} vs {b}"),
    }
}

fn first_request() -> serde_json::Value {
    load_fixture().0[0].clone()
}

#[test]
fn fixture_exact_match_on_cpu() {
    #[cfg(feature = "cpu")]
    {
        if dev().name() != "cpu" {
            eprintln!(
                "exact fixture is CPU-defined; skipping exact comparison on backend '{}'",
                dev().name()
            );
            return;
        }
        let (reqs, expected) = load_fixture();
        assert_eq!(reqs.len(), 8, "eight fixture requests");
        assert_eq!(expected.len(), 8, "eight fixture responses");
        let svc = service();
        for (i, (req, want)) in reqs.iter().zip(expected.iter()).enumerate() {
            let out: serde_json::Value =
                serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
            assert_json_close(&out, want, &format!("fixture[{i}]"));
        }
    }
    #[cfg(not(feature = "cpu"))]
    {
        eprintln!(
            "exact fixture is CPU-defined; skipping exact comparison without the cpu feature"
        );
    }
}

#[test]
fn structural_invariants_determinism_and_seed_hash() {
    let (reqs, _) = load_fixture();
    let svc = service();
    // Composition requests only (mass requests have their own invariants).
    let reqs: Vec<serde_json::Value> = reqs
        .into_iter()
        .filter(|r| r.get("composition").is_some())
        .collect();
    for (i, req) in reqs.iter().enumerate() {
        let out: serde_json::Value =
            serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
        let at = format!("fixture[{i}]");
        assert_eq!(out["protocol"], GENERATE_PROTOCOL, "{at}: protocol");
        assert_eq!(out["query_id"], req["id"], "{at}: query_id");
        assert_eq!(out["provenance"], req["provenance"], "{at}: provenance");
        assert!(
            out["input_hash"].as_str().unwrap().len() == 64,
            "{at}: input_hash"
        );
        let status = out["status"].as_str().unwrap();
        assert!(
            ["ok", "no_candidates", "unsupported_input"].contains(&status),
            "{at}: status"
        );
        // Requested composition for containment checks.
        let comp_obj = req["composition"].as_object().unwrap();
        let mut want_comp = [0u16; 10];
        for (k, v) in comp_obj.iter() {
            want_comp[chem::element_index(k).unwrap()] = v.as_u64().unwrap() as u16;
        }
        let cands = out["candidates"].as_array().unwrap();
        for (r, c) in cands.iter().enumerate() {
            assert_eq!(c["rank"], (r + 1) as u64, "{at}: rank");
            let atoms: Vec<u8> = c["atoms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            let bonds: Vec<(usize, usize, u8)> = c["bonds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| {
                    let t = b.as_array().unwrap();
                    (
                        t[0].as_u64().unwrap() as usize,
                        t[1].as_u64().unwrap() as usize,
                        t[2].as_u64().unwrap() as u8,
                    )
                })
                .collect();
            let graph = MolGraph::new(atoms, bonds).unwrap();
            assert!(graph.is_connected(), "{at}: candidate connected");
            assert_eq!(graph.composition(), want_comp, "{at}: composition");
            for (p, pat) in req["substructures"].as_array().unwrap().iter().enumerate() {
                let patoms: Vec<u8> = pat["atoms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect();
                let pbonds: Vec<(usize, usize, u8)> = pat["bonds"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| {
                        let t = b.as_array().unwrap();
                        (
                            t[0].as_u64().unwrap() as usize,
                            t[1].as_u64().unwrap() as usize,
                            t[2].as_u64().unwrap() as u8,
                        )
                    })
                    .collect();
                let pattern = MolGraph::new(patoms, pbonds).unwrap();
                assert_eq!(
                    contains_pattern(&graph, &pattern, 100_000),
                    Containment::Contained,
                    "{at}: pattern {p} contained"
                );
            }
            let samples = c["samples"].as_u64().unwrap() as f64;
            let traj = out["accounting"]["trajectories"].as_u64().unwrap() as f64;
            let frac = c["sample_fraction"].as_f64().unwrap();
            assert!(
                (frac - samples / traj).abs() <= 1e-9,
                "{at}: sample_fraction"
            );
        }
        let acc = &out["accounting"];
        assert_eq!(
            acc["requested_trajectories"], req["generation"]["trajectories"],
            "{at}: requested_trajectories mirrors the request"
        );
        let finished = acc["finished"].as_u64().unwrap();
        let accepted: u64 = cands.iter().map(|c| c["samples"].as_u64().unwrap()).sum();
        let unresolved = acc["identity_unresolved"].as_u64().unwrap();
        let accounted = acc["rejected_replay"].as_u64().unwrap()
            + acc["rejected_containment"].as_u64().unwrap()
            + acc["containment_unresolved"].as_u64().unwrap()
            + accepted
            + unresolved;
        if acc["distinct"].as_u64().unwrap() == cands.len() as u64 {
            assert_eq!(finished, accounted, "{at}: accounting identity");
        } else {
            assert!(accounted <= finished, "{at}: accounting bound under cut");
        }
        assert_eq!(
            out["ranking"]["status"], "sample_frequency",
            "{at}: ranking"
        );
        assert_eq!(out["ranking"]["calibrated"], false, "{at}: ranking");
        assert_eq!(
            out["mass_evidence"]["status"], "not_evaluated",
            "{at}: mass"
        );
        assert_eq!(out["search"]["status"], "sampled", "{at}: search");
        assert_eq!(out["search"]["exhaustive"], false, "{at}: search");
        assert_eq!(
            out["physical_verification"]["status"], "not_evaluated",
            "{at}: physical"
        );
        assert_eq!(
            out["stereochemistry"]["status"], "enumerated_not_predicted",
            "{at}: stereo status"
        );
        assert!(
            out["stereochemistry"]["reason"]
                .as_str()
                .unwrap()
                .contains("no preference"),
            "{at}: stereo reason"
        );
        // Determinism: same request twice gives identical response
        // (exact for integers/strings, 1e-5 for floats).
        let again: serde_json::Value =
            serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
        assert_json_close(&out, &again, &format!("{at}: deterministic"));
        // A different seed changes input_hash.
        let mut other = req.clone();
        other["generation"]["seed"] = serde_json::json!(999_999);
        let other_out: serde_json::Value =
            serde_json::from_str(&svc.generate_json(&other.to_string()).unwrap()).unwrap();
        assert_ne!(
            out["input_hash"], other_out["input_hash"],
            "{at}: seed changes input_hash"
        );
    }
}

fn err_of(doc: &serde_json::Value) -> String {
    service()
        .generate_json(&doc.to_string())
        .expect_err("rejected")
        .to_string()
}

#[test]
fn rejection_rules() {
    let base = first_request();
    // id and provenance non-empty.
    let mut doc = base.clone();
    doc["id"] = serde_json::json!("");
    assert!(err_of(&doc).contains("request.id"), "empty id");
    let mut doc = base.clone();
    doc["provenance"] = serde_json::json!("");
    assert!(err_of(&doc).contains("provenance"), "empty provenance");
    // mass_role.
    for role in ["target_fragment", "neutral_loss"] {
        let mut doc = base.clone();
        doc["mass_role"] = serde_json::json!(role);
        let err = err_of(&doc);
        assert!(err.contains(role), "{role}: {err}");
        assert!(err.contains("precursor"), "{role}: {err}");
    }
    let mut doc = base.clone();
    doc["mass_role"] = serde_json::json!("molecule");
    assert!(err_of(&doc).contains("mass_role"), "other mass_role");
    // composition.
    let mut doc = base.clone();
    doc["composition"] = serde_json::json!({"X": 1, "H": 2});
    assert!(err_of(&doc).contains('X'), "unknown element");
    let mut doc = base.clone();
    doc["composition"] = serde_json::json!({"H": 4});
    assert!(err_of(&doc).contains("heavy"), "zero heavy atoms");
    let mut doc = base.clone();
    doc["composition"] = serde_json::json!({"C": 70000, "H": 2});
    assert!(err_of(&doc).contains("u16"), "count does not fit");
    let mut doc = base.clone();
    doc["composition"] = serde_json::json!({"C": "2", "H": 6, "O": 1});
    assert!(
        err_of(&doc).contains("composition.C"),
        "wrong composition type"
    );
    // substructures: too many, too many atoms, bad ids, bad bonds, disconnected.
    let mut doc = base.clone();
    let many = (0..9)
        .map(|_| serde_json::json!({"atoms": [4], "bonds": [], "parent_hydrogen_semantics": "v0_parent_hydrogen_counts", "certainty": "confirmed", "provenance": "s"}))
        .collect::<Vec<_>>();
    doc["substructures"] = serde_json::Value::Array(many);
    assert!(err_of(&doc).contains("8"), "at most 8");
    let mut doc = base.clone();
    // Enough seven-atom chains to overrun the slots, whatever the width is.
    let big: Vec<serde_json::Value> = (0..PATTERN_SLOTS / 7 + 1)
        .map(|_| {
            serde_json::json!({"atoms": [3, 3, 3, 3, 3, 3, 3], "bonds": [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1]], "parent_hydrogen_semantics": "v0_parent_hydrogen_counts", "certainty": "confirmed", "provenance": "s"})
        })
        .collect();
    doc["substructures"] = serde_json::Value::Array(big);
    assert!(
        err_of(&doc).contains(&format!("limit of {PATTERN_SLOTS}")),
        "at most {PATTERN_SLOTS} pattern atoms"
    );
    let mut doc = base.clone();
    doc["substructures"][0]["atoms"] = serde_json::json!([99]);
    assert!(err_of(&doc).contains("atom type"), "bad atom id");
    let mut doc = base.clone();
    doc["substructures"][0]["bonds"] = serde_json::json!([[0, 5, 1]]);
    assert!(err_of(&doc).contains("substructures[0]"), "bad bonds");
    let mut doc = base.clone();
    doc["substructures"] = serde_json::json!([
        {"atoms": [4, 4], "bonds": [], "parent_hydrogen_semantics": "v0_parent_hydrogen_counts", "certainty": "confirmed", "provenance": "s"}
    ]);
    assert!(err_of(&doc).contains("connected"), "disconnected");
    let mut doc = base.clone();
    doc["substructures"][0]["parent_hydrogen_semantics"] = serde_json::json!("unknown");
    assert!(
        err_of(&doc).contains("parent_hydrogen_semantics"),
        "bad semantics"
    );
    let mut doc = base.clone();
    doc["substructures"][0]["certainty"] = serde_json::json!("tentative");
    assert!(err_of(&doc).contains("tentative"), "tentative");
    let mut doc = base.clone();
    doc["substructures"][0]["certainty"] = serde_json::json!("maybe");
    assert!(err_of(&doc).contains("certainty"), "other certainty");
    let mut doc = base.clone();
    doc["substructures"][0]["provenance"] = serde_json::json!("");
    assert!(err_of(&doc).contains("provenance"), "empty sub provenance");
    // generation.
    let mut doc = base.clone();
    doc["generation"]["trajectories"] = serde_json::json!(0);
    assert!(err_of(&doc).contains("trajectories"), "bad trajectories");
    let mut doc = base.clone();
    doc["generation"]["temperature"] = serde_json::json!(0.0);
    assert!(err_of(&doc).contains("temperature"), "bad temperature");
    let mut doc = base.clone();
    doc["generation"]["seed"] = serde_json::json!("1");
    assert!(err_of(&doc).contains("seed"), "bad seed type");
    let mut doc = base.clone();
    doc["generation"]["returned"] = serde_json::json!(26);
    assert!(err_of(&doc).contains("returned"), "bad returned");
    // Unknown fields at every level.
    let mut doc = base.clone();
    doc["surprise"] = serde_json::json!(1);
    assert!(
        err_of(&doc).contains("unknown field 'surprise'"),
        "top unknown"
    );
    let mut doc = base.clone();
    doc["substructures"][0]["weight"] = serde_json::json!(0.5);
    assert!(
        err_of(&doc).contains("unknown field 'weight'"),
        "sub unknown"
    );
    let mut doc = base.clone();
    doc["generation"]["extra"] = serde_json::json!(1);
    assert!(
        err_of(&doc).contains("unknown field 'extra'"),
        "gen unknown"
    );
    // Protocol and malformed JSON.
    let mut doc = base.clone();
    doc["protocol"] = serde_json::json!("bogus");
    assert!(err_of(&doc).contains("protocol"), "bad protocol");
    let err = service().generate_json("{not json").expect_err("malformed");
    assert!(err.to_string().contains("valid JSON"), "malformed JSON");
    let mut doc = base.clone();
    doc["generation"]["trajectories"] = serde_json::json!("64");
    assert!(
        err_of(&doc).contains("trajectories"),
        "wrong type names field"
    );
}

#[test]
fn unsupported_input_for_13_heavy_atoms() {
    let mut doc = first_request();
    doc["composition"] = serde_json::json!({"C": 13, "H": 28});
    doc["substructures"] = serde_json::json!([]);
    let out: serde_json::Value =
        serde_json::from_str(&service().generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "unsupported_input");
    assert_eq!(out["candidates"].as_array().unwrap().len(), 0);
    // Nothing was sampled: the request count is carried as
    // `requested_trajectories`, every executed counter is 0.
    let acc = &out["accounting"];
    assert_eq!(acc["requested_trajectories"], 64);
    assert_eq!(acc["trajectories"], 0);
    for key in [
        "finished",
        "dead_end",
        "truncated",
        "other_status",
        "rejected_replay",
        "rejected_containment",
        "containment_unresolved",
        "identity_unresolved",
        "distinct",
    ] {
        assert_eq!(acc[key], 0, "unsupported accounting.{key}");
    }
    assert_eq!(out["search"]["status"], "not_evaluated");
    assert!(
        out["search"]["reason"].as_str().unwrap().len() > 8,
        "search names why nothing ran"
    );
    assert_eq!(out["ranking"]["status"], "not_evaluated");
    assert!(
        out["ranking"]["reason"].as_str().unwrap().len() > 8,
        "ranking names why nothing ran"
    );
    assert_eq!(out["unsupported"]["limit"], "max_atoms");
    assert_eq!(out["unsupported"]["allowed"], 12);
    assert_eq!(out["unsupported"]["observed"], 13);
}

#[test]
fn overlapping_patterns_are_not_summed() {
    // Two copies of a seven-atom ring pattern fit one seven-atom molecule:
    // their sizes (14 total) sum above `max_atoms` (12) but each single
    // pattern fits, so the request generates normally.
    let ring7 = || {
        serde_json::json!({
            "atoms": [3, 3, 3, 3, 3, 3, 3],
            "bonds": [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1], [6, 0, 1]],
            "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
            "certainty": "confirmed",
            "provenance": "synthetic: seven-ring twice"
        })
    };
    let mut doc = first_request();
    doc["composition"] = serde_json::json!({"C": 7, "H": 14});
    doc["substructures"] = serde_json::Value::Array(vec![ring7(), ring7()]);
    let out: serde_json::Value =
        serde_json::from_str(&service().generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_ne!(
        out["status"], "unsupported_input",
        "overlapping patterns share target atoms: {}",
        out
    );
    assert!(
        ["ok", "no_candidates"].contains(&out["status"].as_str().unwrap()),
        "generates normally: {}",
        out["status"]
    );
}

#[test]
fn single_pattern_above_composition_heavy_is_unsupported() {
    // One four-atom pattern against a three-heavy-atom composition: the
    // pattern alone (4 <= max_atoms 12) fits the model, but it cannot fit a
    // three-heavy-atom target.
    let mut doc = first_request();
    doc["composition"] = serde_json::json!({"C": 2, "H": 6, "O": 1});
    doc["substructures"] = serde_json::json!([
        {"atoms": [4, 3, 3, 9], "bonds": [[0, 1, 1], [1, 2, 1], [2, 3, 1]],
         "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
         "certainty": "confirmed", "provenance": "synthetic: too big"}
    ]);
    let out: serde_json::Value =
        serde_json::from_str(&service().generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "unsupported_input");
    assert_eq!(out["unsupported"]["limit"], "composition_heavy_atoms");
    assert_eq!(out["unsupported"]["allowed"], 3);
    assert_eq!(out["unsupported"]["observed"], 4);
}

#[test]
fn input_hash_ignores_key_order_and_whitespace() {
    let svc = service();
    let base = first_request();
    let base_out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&base.to_string()).unwrap()).unwrap();
    // Same request with top-level, composition, generation and substructure
    // keys reordered, plus different whitespace, hashes identically.
    let reordered = serde_json::json!({
        "substructures": [
            {"provenance": "synthetic: C(H2)-O(H1)", "certainty": "confirmed",
             "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
             "bonds": [[0, 1, 1]], "atoms": [3, 9]}
        ],
        "provenance": "synthetic fixture",
        "protocol": "molecular-completion-generate-v1",
        "mass_role": "target_molecule",
        "id": base["id"],
        "generation": {"seed": 1, "returned": 25, "trajectories": 64, "temperature": 1.0},
        "composition": {"O": 1, "H": 6, "C": 2},
    });
    let compact = serde_json::to_string(&reordered).unwrap();
    let spaced = serde_json::to_string_pretty(&reordered).unwrap();
    assert_ne!(compact, spaced, "whitespace really differs");
    let a: serde_json::Value = serde_json::from_str(&svc.generate_json(&compact).unwrap()).unwrap();
    let b: serde_json::Value = serde_json::from_str(&svc.generate_json(&spaced).unwrap()).unwrap();
    assert_eq!(base_out["input_hash"], a["input_hash"], "key order");
    assert_eq!(a["input_hash"], b["input_hash"], "whitespace");
    // Any value change changes the hash.
    let mut other = base.clone();
    other["generation"]["seed"] = serde_json::json!(2);
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&other.to_string()).unwrap()).unwrap();
    assert_ne!(base_out["input_hash"], c["input_hash"], "value change");
    let mut other = base.clone();
    other["composition"]["C"] = serde_json::json!(3);
    let d: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&other.to_string()).unwrap()).unwrap();
    assert_ne!(
        base_out["input_hash"], d["input_hash"],
        "composition change"
    );
}

#[test]
fn huge_bond_index_is_a_schema_error() {
    for huge in [u64::from(u32::MAX) + 1, u64::MAX] {
        let mut doc = first_request();
        doc["substructures"][0]["bonds"] = serde_json::json!([[0, huge, 1]]);
        let err = err_of(&doc);
        assert!(
            err.contains("substructures[0]") && err.contains("atom"),
            "huge index {huge} is a schema error: {err}"
        );
    }
}

#[test]
fn describe_names_versions() {
    let doc: serde_json::Value = serde_json::from_str(&service().describe()).unwrap();
    assert_eq!(doc["protocol"], GENERATE_PROTOCOL);
    assert_eq!(doc["model_version"], COMPLETION_MODEL_VERSION);
    assert_eq!(doc["grammar_version"], COMPLETION_GRAMMAR_VERSION);
    assert_eq!(doc["data_version"], COMPLETION_DATA_VERSION);
    assert_eq!(doc["chemistry_version"], CHEMISTRY_VERSION);
    assert!(doc["checkpoint_sha256"].as_str().unwrap().len() == 64);
    assert!(doc["trained_steps"].as_u64().is_some());
    assert_eq!(doc["domain"]["max_atoms"], 12);
    assert_eq!(doc["domain"]["max_ring_closures"], 2);
}

#[test]
fn truncated_or_foreign_file_is_error_not_panic() {
    let dir = std::env::temp_dir();
    let bytes = std::fs::read(ckpt_path()).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let foreign_text = text.replacen("completion-checkpoint-v1", "bogus-format", 1);
    let foreign = dir.join(format!("mc9_foreign_{}.json", std::process::id()));
    std::fs::write(&foreign, foreign_text).unwrap();
    let err = match CompletionService::<R>::load(&foreign, &dev()) {
        Err(e) => e,
        Ok(_) => panic!("foreign file must be rejected"),
    };
    assert!(err.to_string().contains("format"), "foreign: {err}");
    let truncated = dir.join(format!("mc9_trunc_{}.json", std::process::id()));
    let bytes = std::fs::read(ckpt_path()).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    let err = match CompletionService::<R>::load(&truncated, &dev()) {
        Err(e) => e,
        Ok(_) => panic!("truncated file must be rejected"),
    };
    assert!(!err.to_string().is_empty(), "truncated: {err}");
    // A non-checkpoint file (not JSON at all) is an error, never a crash.
    let garbage = dir.join(format!("mc9_garbage_{}.json", std::process::id()));
    std::fs::write(&garbage, "this is not a checkpoint").unwrap();
    let err = match CompletionService::<R>::load(&garbage, &dev()) {
        Err(e) => e,
        Ok(_) => panic!("a non-checkpoint file must be rejected"),
    };
    assert!(!err.to_string().is_empty(), "garbage: {err}");
}

// ---------------------------------------------------------------------------
// MC12: mass input.
// ---------------------------------------------------------------------------

/// A wide-window neutral request whose train-fit search selects at least 8
/// formulas with at least one excluded by the train-fit bounds, cached across
/// tests (the scan is deterministic: same fixture, same seeds).
static WIDE_REQ: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn wide_window_request_text() -> String {
    WIDE_REQ
        .get_or_init(|| {
            let svc = service();
            for value in (20_000_000u32..120_000_000).step_by(2_000_000) {
                for unc in [2_000_000u32, 10_000_000u32] {
                    let mut req = mass_request_neutral();
                    req["target_mass"] = serde_json::json!({"units": "microdalton", "value": value, "ppm_tenths": 1000u32, "uncertainty_uda": unc, "source": "synthetic"});
                    req["substructures"] = serde_json::json!([]);
                    req["formula_search"] =
                        serde_json::json!({"hypotheses": 32u32, "nodes_visited_max": 2000000u32});
                    req["generation"] = serde_json::json!({"trajectories": 64u32, "temperature": 1.0, "seed": 1u64, "returned": 25u32});
                    let out: serde_json::Value =
                        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap())
                            .unwrap();
                    let fs = &out["formula_search"];
                    let selected = fs["selected"].as_u64().unwrap_or(0);
                    let excluded = fs["excluded_by_train_fit"].as_u64().unwrap_or(0);
                    if selected >= 8 && excluded >= 1 {
                        return req.to_string();
                    }
                }
            }
            panic!("no wide window with >=8 selected and >=1 train-fit exclusion");
        })
        .clone()
}

fn wide_window_request() -> serde_json::Value {
    serde_json::from_str(&wide_window_request_text()).unwrap()
}

#[test]
fn mass_error_terms_split_observation_composition_neutralisation() {
    let svc = service();
    // Neutral: observation 50, composition 1 (C2H6O: ceil(579/1000)), no
    // neutralisation term.
    let req = mass_request_neutral();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(
        out["mass_evidence"]["error_terms_uda"],
        serde_json::json!({"observation_uda": 50, "composition_uda": 1, "neutralisation_uda": 0})
    );
    // Precursor: the same observation and composition terms plus the single
    // +1 adduct-conversion bound.
    let req = mass_request_precursor();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(
        out["mass_evidence"]["error_terms_uda"],
        serde_json::json!({"observation_uda": 50, "composition_uda": 1, "neutralisation_uda": 1})
    );
}

#[test]
fn mass_pruning_chemical_only_admits_superset() {
    let svc = service();
    let mut train = wide_window_request();
    train["formula_search"]["hypotheses"] = serde_json::json!(8u32);
    let tout: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&train.to_string()).unwrap()).unwrap();
    let mut chem = train.clone();
    chem["id"] = serde_json::json!("mc12b-wide-chemical-only");
    chem["formula_search"]["pruning"] = serde_json::json!("chemical_only");
    let cout: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&chem.to_string()).unwrap()).unwrap();
    let (tfs, cfs) = (&tout["formula_search"], &cout["formula_search"]);
    assert_eq!(tfs["pruning"], "train_fit");
    assert_eq!(cfs["pruning"], "chemical_only");
    // The chemical-only search admits a superset: its joined count is the
    // exact-filter count, and the train-fit difference is exact.
    let (tjoined, cjoined) = (
        tfs["joined"].as_u64().unwrap(),
        cfs["joined"].as_u64().unwrap(),
    );
    assert_eq!(cfs["joined_chemical"], cjoined);
    assert!(
        cjoined >= tjoined,
        "chemical {cjoined} under train-fit {tjoined}"
    );
    assert_eq!(tfs["joined_chemical"], cjoined);
    assert_eq!(
        tfs["excluded_by_train_fit"],
        serde_json::json!(cjoined - tjoined)
    );
    assert!(tfs["excluded_by_train_fit"].as_u64().unwrap() >= 1);
    assert_eq!(
        cfs["excluded_by_train_fit"],
        serde_json::json!(cjoined - tjoined)
    );
    // Stage classes distinguish necessary, empirical and budget stages.
    let classes: Vec<String> = tfs["stages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["class"].as_str().unwrap().to_string())
        .collect();
    assert!(classes.contains(&"necessary".to_string()));
    assert!(classes.contains(&"empirical".to_string()));
    assert!(classes.contains(&"budget".to_string()));
    let train_fit_stage = tfs["stages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["stage"] == "train_fit_bounds")
        .unwrap();
    assert_eq!(train_fit_stage["class"], "empirical");
    // Unknown pruning/allocation spellings are schema errors.
    let mut bad = mass_request_neutral();
    bad["formula_search"]["pruning"] = serde_json::json!("none");
    let err = svc
        .generate_json(&bad.to_string())
        .expect_err("bad pruning");
    assert!(err.to_string().contains("pruning"), "bad pruning: {err}");
    let mut bad = mass_request_neutral();
    bad["formula_search"]["allocation"] = serde_json::json!("none");
    let err = svc
        .generate_json(&bad.to_string())
        .expect_err("bad allocation");
    assert!(
        err.to_string().contains("allocation"),
        "bad allocation: {err}"
    );
}

#[test]
fn mass_selection_orders_verdict_then_residual() {
    let svc = service();
    let mut req = wide_window_request();
    req["formula_search"]["hypotheses"] = serde_json::json!(8u32);
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let formulas = out["formula_search"]["formulas"].as_array().unwrap();
    assert!(formulas.len() >= 2);
    // Verdict (Accept before Ambiguous), then residual, is the selected order.
    let mut keys: Vec<(bool, u64)> = Vec::new();
    for f in formulas {
        let ambiguous = f["verdict"] == "boundary_ambiguous";
        keys.push((ambiguous, f["residual_uda"].as_u64().unwrap()));
    }
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "selected in (verdict, residual) order");
    // The ranking text states the deterministic default (not a probability).
    assert!(
        out["formula_search"]["ranking"]
            .as_str()
            .unwrap()
            .contains("deterministic default, not a formula probability")
    );
}

#[test]
fn mass_truncation_is_prefix_of_full_selection() {
    let svc = service();
    let base = wide_window_request();
    // Full selection (hypotheses 32 covers every survivor here).
    let mut full = base.clone();
    full["formula_search"]["hypotheses"] = serde_json::json!(32u32);
    let fout: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&full.to_string()).unwrap()).unwrap();
    // Truncated to one hypothesis: never a silently wrong "complete".
    let mut one = base.clone();
    one["id"] = serde_json::json!("mc12b-wide-truncated-one");
    one["formula_search"]["hypotheses"] = serde_json::json!(1u32);
    let oout: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&one.to_string()).unwrap()).unwrap();
    let (full_fs, one_fs) = (&fout["formula_search"], &oout["formula_search"]);
    assert!(full_fs["selected"].as_u64().unwrap() > 1);
    assert_eq!(one_fs["status"], "truncated");
    assert_eq!(one_fs["selected"], 1);
    assert_eq!(
        one_fs["formulas"][0]["formula"], full_fs["formulas"][0]["formula"],
        "the truncated selection is the prefix of the full selection"
    );
}

#[test]
fn mass_redistribution_spends_the_budget() {
    let svc = service();
    let mut req = wide_window_request();
    req["id"] = serde_json::json!("mc12b-wide-redistribute");
    req["formula_search"]["hypotheses"] = serde_json::json!(3u32);
    req["generation"] = serde_json::json!({"trajectories": 64u32, "temperature": 1.0, "seed": 1u64, "returned": 25u32});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let fs = &out["formula_search"];
    assert_eq!(fs["selected"], 3);
    let trajs: Vec<u64> = fs["formulas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["trajectories"].as_u64().unwrap())
        .collect();
    // 64 over 3: even split with the remainder redistributed one each to the
    // first formulas — the fixed budget is spent, never silently dropped.
    // The allocations differ (22 vs 21), so generation batches group out of
    // selected order: exactly the F1 shape.
    assert_eq!(trajs, vec![22, 21, 21]);
    assert_eq!(out["accounting"]["trajectories"], 64);
    assert_eq!(out["accounting"]["requested_trajectories"], 64);
    assert_eq!(out["accounting"]["unused_trajectories"], 0);
    // Every selected formula reports its stage, weight and mass status.
    for f in fs["formulas"].as_array().unwrap() {
        assert_eq!(f["stage"], "sampled");
        assert!((f["weight"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!(["accepted", "boundary_ambiguous"].contains(&f["mass_status"].as_str().unwrap()));
    }
    // F1: pooled candidates carry their own source formula's mass metadata,
    // even when trajectory allocations differ across formulas ([3, 2, 2]
    // groups the generation batches out of selected order).
    let candidates = out["candidates"].as_array().unwrap();
    assert!(
        !candidates.is_empty(),
        "the mixed allocation must pool candidates for the invariant to bite"
    );
    for c in candidates {
        assert_eq!(
            c["composition"], c["mass"]["formula"],
            "pooled candidate composition equals its source formula"
        );
    }
}

#[test]
fn mass_exhausted_search_is_not_rejection() {
    // F2: a search stopped by the node budget reports `search_incomplete`,
    // never a definitive `rejected` — and the top-level status stays
    // `no_candidates`.
    let svc = service();
    let mut req = mass_request_neutral();
    req["id"] = serde_json::json!("mc17-ethanol-exhausted");
    req["formula_search"] = serde_json::json!({"hypotheses": 8u32, "nodes_visited_max": 1u32});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "no_candidates");
    assert_eq!(out["mass_evidence"]["status"], "search_incomplete");
    assert_eq!(out["formula_search"]["status"], "search_exhausted");
    assert_eq!(out["formula_search"]["search_exhausted"], true);
}

#[test]
fn mass_overflow_is_propagated_never_zero_mass() {
    // F2: a precursor whose neutralisation leaves the u32 range reports
    // `mass_overflow` (the enumerator's status), never a search around a
    // substituted zero mass.
    let svc = service();
    let mut req = mass_request_neutral();
    req["id"] = serde_json::json!("mc17-mass-overflow");
    req["target_mass"] = serde_json::json!({"units": "microdalton", "value": 4294967295u32, "ppm_tenths": 100u32, "uncertainty_uda": 50u32, "source": "synthetic"});
    req["neutralization"] = serde_json::json!({"precursor_ion": {"adduct": "[M-H]-"}});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "no_candidates");
    assert_eq!(out["mass_evidence"]["status"], "mass_overflow");
    assert!(out["candidates"].as_array().unwrap().is_empty());
}

#[test]
fn mass_u32max_sentinel_is_a_schema_error() {
    // F2 (stricter reading): the unknown-precision sentinel `u32::MAX`
    // passed as an integer is rejected at the API boundary — unknown
    // precision must be spelled `null` (which yields `unavailable`).
    let svc = service();
    let mut req = mass_request_neutral();
    req["target_mass"] = serde_json::json!({"units": "microdalton", "value": 46041865u32, "ppm_tenths": 100u32, "uncertainty_uda": 4294967295u32, "source": "synthetic"});
    let err = svc
        .generate_json(&req.to_string())
        .expect_err("u32::MAX integer");
    assert!(
        err.to_string().contains("uncertainty_uda"),
        "sentinel names the field: {err}"
    );
}

#[test]
fn mass_search_exhausted_takes_precedence_over_truncation() {
    // F8: when both limits bind, the top-level status is `search_exhausted`
    // (a nearer formula may remain unvisited); both facts stay visible as
    // independent booleans. The scan finds a node budget where the wide
    // window exhausts after at least two survivors joined (so hypotheses=1
    // also truncates); the stop node is a deterministic function of the
    // inputs, so the scan is stable.
    let svc = service();
    let base = wide_window_request();
    let mut full = base.clone();
    full["id"] = serde_json::json!("mc17-wide-full-nodes");
    full["formula_search"]["hypotheses"] = serde_json::json!(1u32);
    let fout: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&full.to_string()).unwrap()).unwrap();
    let full_nodes = fout["formula_search"]["enumerator"]["nodes_visited"]
        .as_u64()
        .unwrap();
    assert!(
        fout["formula_search"]["after_completability"]
            .as_u64()
            .unwrap()
            >= 2
    );
    let mut found: Option<serde_json::Value> = None;
    let mut budgets = vec![full_nodes.saturating_sub(1)];
    let mut half = full_nodes / 2;
    while half >= 10 {
        budgets.push(half);
        half /= 2;
    }
    for nodes in budgets {
        if nodes < 1 {
            continue;
        }
        let mut req = base.clone();
        req["id"] = serde_json::json!("mc17-wide-truncated-exhausted");
        req["formula_search"]["hypotheses"] = serde_json::json!(1u32);
        req["formula_search"]["nodes_visited_max"] = serde_json::json!(nodes);
        let out: serde_json::Value =
            serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
        let fs = &out["formula_search"];
        if fs["search_exhausted"] == true && fs["after_completability"].as_u64().unwrap() >= 2 {
            found = Some(out);
            break;
        }
    }
    let out = found.expect("a budget exhausting after >= 2 survivors joined");
    let fs = &out["formula_search"];
    assert_eq!(fs["status"], "search_exhausted");
    assert_eq!(fs["truncated"], true);
    assert_eq!(fs["search_exhausted"], true);
}

#[test]
fn mass_stage_counts_come_from_one_traversal() {
    // F9: `exact_chemical_filters` reports the within-traversal count
    // (verdict-passing minus exact rejects), not the train-fit-pruned
    // `joined`; the DFS empirical pruning is its own line.
    let svc = service();
    let mut req = wide_window_request();
    req["id"] = serde_json::json!("mc17-wide-stages");
    req["formula_search"]["hypotheses"] = serde_json::json!(8u32);
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let fs = &out["formula_search"];
    let en = &fs["enumerator"];
    let stages = fs["stages"].as_array().unwrap();
    let at = |name: &str| {
        stages
            .iter()
            .find(|s| s["stage"] == name)
            .unwrap_or_else(|| panic!("stage {name}"))
    };
    let verdict = at("mass_verdict");
    let exact = at("exact_chemical_filters");
    let dfs = at("train_fit_dfs_pruning");
    let bounds = at("train_fit_bounds");
    let exact_rejects = en["rejected_h_max"].as_u64().unwrap()
        + en["rejected_parity"].as_u64().unwrap()
        + en["rejected_dbe"].as_u64().unwrap();
    assert_eq!(
        exact["entering"], verdict["leaving"],
        "exact filters start where the verdict ends"
    );
    assert_eq!(
        exact["leaving"].as_u64().unwrap(),
        verdict["leaving"].as_u64().unwrap() - exact_rejects,
        "exact filters leave the within-traversal count"
    );
    assert_eq!(exact["class"], "necessary");
    assert_eq!(dfs["class"], "empirical");
    assert_eq!(dfs["entering"], exact["leaving"]);
    assert_eq!(dfs["leaving"], exact["leaving"]);
    assert_eq!(bounds["class"], "empirical");
    assert_eq!(bounds["entering"], exact["leaving"]);
    assert_eq!(
        bounds["leaving"], fs["joined"],
        "train-fit bounds leave the joined count"
    );
}

#[test]
fn mass_train_fit_exclusion_is_not_mass_rejection() {
    // F2: when the train-fit bounds remove every mass-level formula, the
    // response says why (`unsampled_reason = "by_train_fit"`) and the mass
    // evidence follows the chemical-only rerun verdicts — never `rejected`.
    // The scan finds a window where the primary search joins nothing but the
    // chemical-only rerun joins rows (deterministic: same fixture, seeds).
    static REQ: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let text = REQ
        .get_or_init(|| {
            let svc = service();
            for value in (20_000_000u32..200_000_000).step_by(2_000_000) {
                for unc in [50_000u32, 200_000u32, 2_000_000u32] {
                    let mut req = mass_request_neutral();
                    req["target_mass"] = serde_json::json!({"units": "microdalton", "value": value, "ppm_tenths": 100u32, "uncertainty_uda": unc, "source": "synthetic"});
                    req["substructures"] = serde_json::json!([]);
                    req["formula_search"] =
                        serde_json::json!({"hypotheses": 8u32, "nodes_visited_max": 2000000u32});
                    let out: serde_json::Value =
                        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap())
                            .unwrap();
                    let fs = &out["formula_search"];
                    if fs["joined"].as_u64().unwrap_or(1) == 0
                        && fs["joined_chemical"].as_u64().unwrap_or(0) > 0
                    {
                        return req.to_string();
                    }
                }
            }
            panic!("no window with joined 0 but chemical-only rows");
        })
        .clone();
    let svc = service();
    let req: serde_json::Value = serde_json::from_str(&text).unwrap();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let fs = &out["formula_search"];
    assert_eq!(fs["joined"], 0);
    assert!(fs["joined_chemical"].as_u64().unwrap() > 0);
    assert_eq!(fs["unsampled_reason"], "by_train_fit");
    assert!(
        ["accepted", "boundary_ambiguous"]
            .contains(&out["mass_evidence"]["status"].as_str().unwrap()),
        "mass evidence follows the rerun verdicts: {}",
        out["mass_evidence"]["status"]
    );
    assert_eq!(out["status"], "no_candidates");
}

#[test]
fn mass_budget_starvation_reports_budget() {
    // F2/F9: when the total budget does not cover the selection, sampled
    // formulas report `sampled` and the rest `not_sampled`, with
    // `unsampled_reason = "budget"`.
    let svc = service();
    let mut req = wide_window_request();
    req["id"] = serde_json::json!("mc17-wide-budget-starved");
    req["formula_search"]["hypotheses"] = serde_json::json!(8u32);
    req["generation"] = serde_json::json!({"trajectories": 4u32, "temperature": 1.0, "seed": 1u64, "returned": 25u32});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let fs = &out["formula_search"];
    assert_eq!(fs["selected"], 8);
    assert_eq!(fs["sampled"], 4);
    assert_eq!(fs["unsampled_reason"], "budget");
    let stages: Vec<&str> = fs["formulas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["stage"].as_str().unwrap())
        .collect();
    assert_eq!(
        stages,
        vec![
            "sampled",
            "sampled",
            "sampled",
            "sampled",
            "not_sampled",
            "not_sampled",
            "not_sampled",
            "not_sampled"
        ]
    );
}

#[test]
fn mass_substructure_wipeout_reports_stage() {
    // F2: formulas joined on mass but a heavy pattern excludes them all at
    // the substructure bound: `unsampled_reason = "by_substructures"`, mass
    // evidence still from the search (not `rejected`), status no_candidates.
    let svc = service();
    let mut req = wide_window_request();
    req["id"] = serde_json::json!("mc17-wide-substructure-wipeout");
    req["formula_search"]["hypotheses"] = serde_json::json!(8u32);
    // Twelve heavy carbons: valid for the encoder, heavier than every
    // joined formula of the tiny artifact domain (heavy_max 7).
    let chain: Vec<serde_json::Value> = (0..12).map(|_| serde_json::json!(3u32)).collect();
    let bonds: Vec<serde_json::Value> = (0..11).map(|i| serde_json::json!([i, i + 1, 1])).collect();
    req["substructures"] = serde_json::json!([{
        "atoms": chain,
        "bonds": bonds,
        "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
        "certainty": "confirmed",
        "provenance": "synthetic: C12 chain",
    }]);
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let fs = &out["formula_search"];
    assert!(fs["joined"].as_u64().unwrap() > 0);
    assert_eq!(fs["after_substructures"], 0);
    assert_eq!(fs["unsampled_reason"], "by_substructures");
    assert!(
        ["accepted", "boundary_ambiguous"]
            .contains(&out["mass_evidence"]["status"].as_str().unwrap())
    );
    assert_eq!(out["status"], "no_candidates");
}

#[test]
fn mass_chemical_rerun_own_limit_has_reason() {
    // F10: the diagnostic rerun has its own node budget and row capacity.
    // With a node budget the primary almost spends, the primary completes
    // but the rerun binds its own limit: `joined_chemical` is null with a
    // reason naming the rerun's limit.
    let svc = service();
    let mut full = wide_window_request();
    full["id"] = serde_json::json!("mc17-wide-rerun-full");
    full["formula_search"]["hypotheses"] = serde_json::json!(8u32);
    let fout: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&full.to_string()).unwrap()).unwrap();
    let full_nodes = fout["formula_search"]["enumerator"]["nodes_visited"]
        .as_u64()
        .unwrap();
    assert!(fout["formula_search"]["joined_chemical"].is_number());
    let mut found: Option<serde_json::Value> = None;
    // Just above the primary's full node cost the primary completes
    // identically (deterministic stop node) with almost nothing left, so
    // the bigger chemical-only rerun binds its own node budget.
    let mut budget = full_nodes.saturating_add(1);
    for _ in 0..24 {
        let mut req = wide_window_request();
        req["id"] = serde_json::json!("mc17-wide-rerun-bound");
        req["formula_search"]["hypotheses"] = serde_json::json!(8u32);
        req["formula_search"]["nodes_visited_max"] = serde_json::json!(budget);
        let out: serde_json::Value =
            serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
        let fs = &out["formula_search"];
        if fs["status"] != "search_exhausted"
            && fs["joined_chemical"].is_null()
            && fs["joined"].as_u64().unwrap() <= 200_000
        {
            found = Some(out);
            break;
        }
        budget = budget.saturating_mul(2);
    }
    let out = found.expect("a budget where the primary completes but the rerun binds");
    let fs = &out["formula_search"];
    assert!(fs["joined_chemical"].is_null());
    let reason = fs["joined_chemical_reason"].as_str().unwrap();
    assert!(
        reason.contains("rerun"),
        "reason names the rerun's own limit: {reason}"
    );
}

#[test]
fn mass_allocation_train_frequency_weights_and_estimate() {
    let svc = service();
    let mut req = wide_window_request();
    req["id"] = serde_json::json!("mc12b-wide-train-frequency");
    req["formula_search"]["hypotheses"] = serde_json::json!(8u32);
    req["formula_search"]["allocation"] = serde_json::json!("train_frequency");
    req["generation"] = serde_json::json!({"trajectories": 64u32, "temperature": 1.0, "seed": 1u64, "returned": 25u32});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let fs = &out["formula_search"];
    assert_eq!(fs["allocation"], "train_frequency");
    assert!(
        fs["ranking"]
            .as_str()
            .unwrap()
            .contains("training-frequency prior, not a calibrated probability")
    );
    let formulas = fs["formulas"].as_array().unwrap();
    // Weights are a distribution over the selected formulas.
    let wsum: f64 = formulas.iter().map(|f| f["weight"].as_f64().unwrap()).sum();
    assert!((wsum - 1.0).abs() < 1e-9, "weights sum to 1: {wsum}");
    // The fixed budget is spent exactly, at least one per formula.
    let tsum: u64 = formulas
        .iter()
        .map(|f| f["trajectories"].as_u64().unwrap())
        .sum();
    assert_eq!(tsum, 64);
    assert_eq!(out["accounting"]["unused_trajectories"], 0);
    for f in formulas {
        assert!(f["trajectories"].as_u64().unwrap() >= 1);
        assert!(f["weight"].as_f64().unwrap() > 0.0);
    }
    // Candidates rank by the explicit estimate weight * samples /
    // trajectories (non-increasing across the shortlist).
    let ests: Vec<f64> = out["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let form = &c["mass"]["formula"];
            let f = formulas.iter().find(|f| f["formula"] == *form).unwrap();
            f["weight"].as_f64().unwrap() * c["samples"].as_f64().unwrap()
                / f["trajectories"].as_f64().unwrap()
        })
        .collect();
    for pair in ests.windows(2) {
        assert!(
            pair[0] + 1e-12 >= pair[1],
            "estimate non-increasing: {ests:?}"
        );
    }
    // F11: the ranking metadata names the rule actually used.
    assert_eq!(
        out["ranking"]["status"], "train_frequency_weighted_estimate",
        "top-level ranking names the weighted estimate"
    );
    assert!(
        out["ranking"]["reason"]
            .as_str()
            .unwrap()
            .contains("weight * samples / trajectories"),
        "ranking reason carries the formula: {}",
        out["ranking"]["reason"]
    );
    assert_eq!(out["ranking"]["calibrated"], false);
}

#[test]
fn mass_ranking_equal_is_sample_frequency() {
    // F11 (other half): under the default equal allocation both the
    // top-level and the formula-search ranking say `sample_frequency`.
    let svc = service();
    let mut req = wide_window_request();
    req["id"] = serde_json::json!("mc17-wide-equal-ranking");
    req["formula_search"]["hypotheses"] = serde_json::json!(3u32);
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["ranking"]["status"], "sample_frequency");
    assert!(
        out["formula_search"]["ranking"]
            .as_str()
            .unwrap()
            .contains("evenly")
            || out["formula_search"]["ranking"]
                .as_str()
                .unwrap()
                .contains("even split")
    );
}

fn mass_request_neutral() -> serde_json::Value {
    serde_json::json!({
        "protocol": "molecular-completion-generate-v1",
        "id": "mc12-ethanol-neutral",
        "provenance": "synthetic test",
        "mass_role": "target_molecule",
        "target_mass": {"units": "microdalton", "value": 46041865u32, "ppm_tenths": 100u32, "uncertainty_uda": 50u32, "source": "synthetic"},
        "neutralization": "already_neutral",
        "formula_search": {"hypotheses": 8u32, "nodes_visited_max": 2000000u32},
        "substructures": [
            {"atoms": [3, 9], "bonds": [[0, 1, 1]],
             "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
             "certainty": "confirmed", "provenance": "synthetic: C(H2)-O(H1)"}
        ],
        "generation": {"trajectories": 64u32, "temperature": 1.0, "seed": 1u64, "returned": 25u32}
    })
}

fn mass_request_precursor() -> serde_json::Value {
    let mut req = mass_request_neutral();
    req["id"] = serde_json::json!("mc12-ethanol-protonated");
    req["target_mass"] = serde_json::json!({"units": "microdalton", "value": 47049141u32, "ppm_tenths": 100u32, "uncertainty_uda": 50u32, "source": "synthetic"});
    req["neutralization"] = serde_json::json!({"precursor_ion": {"adduct": "[M+H]+"}});
    req
}

#[test]
fn mass_ethanol_neutral_returns_ethanol_accepted() {
    let svc = service();
    let req = mass_request_neutral();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_ne!(out["status"], "unsupported_input", "needs artifacts: {out}");
    let cands = out["candidates"].as_array().unwrap();
    assert!(!cands.is_empty(), "ethanol mass returns candidates");
    // Ethanol present.
    let has_ethanol = cands.iter().any(|c| c["composition"] == "C2H6O");
    assert!(has_ethanol, "formula list contains C2H6O: {out}");
    for c in cands {
        assert!(
            ["accepted", "boundary_ambiguous"].contains(&c["mass"]["status"].as_str().unwrap())
        );
        assert_eq!(c["mass"]["formula"], c["composition"]);
    }
    assert_eq!(out["mass_evidence"]["status"], "accepted");
    let fs = &out["formula_search"];
    let formulas: Vec<String> = fs["formulas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["formula"].as_str().unwrap().to_string())
        .collect();
    assert!(
        formulas.contains(&"C2H6O".to_string()),
        "formulas: {formulas:?}"
    );
    let (joined, after_domain, after_sub, after_comp, selected, sampled) = (
        fs["joined"].as_u64().unwrap(),
        fs["after_domain"].as_u64().unwrap(),
        fs["after_substructures"].as_u64().unwrap(),
        fs["after_completability"].as_u64().unwrap(),
        fs["selected"].as_u64().unwrap(),
        fs["sampled"].as_u64().unwrap(),
    );
    assert!(joined >= after_domain);
    assert!(after_domain >= after_sub);
    assert!(after_sub >= after_comp);
    assert!(after_comp >= selected);
    assert!(selected >= sampled);
    assert_eq!(
        fs["ranking"],
        "verdict, then absolute mass residual, then canonical order; trajectories split evenly with the remainder redistributed; deterministic default, not a formula probability"
    );
}

#[test]
fn mass_ethanol_precursor_matches_neutral() {
    let svc = service();
    let req = mass_request_precursor();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_ne!(out["status"], "unsupported_input");
    let cands = out["candidates"].as_array().unwrap();
    assert!(!cands.is_empty());
    assert!(cands.iter().any(|c| c["composition"] == "C2H6O"));
    assert_eq!(out["mass_evidence"]["status"], "accepted");
}

#[test]
fn mass_far_from_any_formula_is_rejected() {
    let svc = service();
    let mut req = mass_request_neutral();
    req["target_mass"] = serde_json::json!({"units": "microdalton", "value": 1000000000u32, "ppm_tenths": 100u32, "uncertainty_uda": 50u32, "source": "synthetic"});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["mass_evidence"]["status"], "rejected");
    assert_eq!(out["status"], "no_candidates");
}

#[test]
fn mass_unknown_precision_is_unavailable_without_search() {
    let svc = service();
    let mut req = mass_request_neutral();
    req["target_mass"] = serde_json::json!({"units": "microdalton", "value": 46041865u32, "ppm_tenths": 100u32, "uncertainty_uda": null, "source": "synthetic"});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["mass_evidence"]["status"], "unavailable");
    assert_eq!(out["candidates"].as_array().unwrap().len(), 0);
    let en = &out["formula_search"]["enumerator"];
    for key in [
        "nodes_visited",
        "hydrogen_checks",
        "rows_joined",
        "rejected_mass",
        "rejected_h_max",
        "rejected_parity",
        "rejected_dbe",
    ] {
        assert_eq!(en[key], 0, "no search counter {key} above zero");
    }
}

#[test]
fn mass_composition_exclusivity_is_schema_error() {
    // Both.
    let mut both = mass_request_neutral();
    both["composition"] = serde_json::json!({"C": 2, "H": 6, "O": 1});
    let err = service()
        .generate_json(&both.to_string())
        .expect_err("both is an error");
    assert!(err.to_string().contains("exactly one"), "both: {err}");
    // Neither.
    let mut neither = mass_request_neutral();
    neither.as_object_mut().unwrap().remove("target_mass");
    let err = service()
        .generate_json(&neither.to_string())
        .expect_err("neither is an error");
    assert!(err.to_string().contains("exactly one"), "neither: {err}");
}

#[test]
fn mass_without_artifacts_is_unsupported() {
    use mamba3::models::ms2::completion_data::{ExtractionConfig, PatternSource};
    use mamba3::models::ms2::completion_model::{CompletionModelConfig, CompletionTrainConfig};
    let device = dev();
    let model_config = CompletionModelConfig::tiny();
    let train_config = CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction: ExtractionConfig::default(),
        extraction_seed: 11,
        pattern_source: PatternSource::RandomPatches(ExtractionConfig::default()),
        fingerprint_mode: None,
        fingerprint_threshold: 0.1,
    };
    let trainer = mamba3::models::ms2::completion_model::CompletionTrainer::<R, f32>::new(
        &model_config,
        &train_config,
        &device,
    )
    .unwrap();
    assert!(trainer.formula_artifacts().is_none());
    let path = std::env::temp_dir().join(format!("mc12_noart_{}.ckpt", std::process::id()));
    trainer.save(&path).unwrap();
    let svc = CompletionService::load(&path, &device).unwrap();
    let req = mass_request_neutral();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "unsupported_input");
    assert_eq!(out["unsupported"]["limit"], "formula_artifacts");
}

#[test]
fn mass_budget_split_five_over_eight() {
    let svc = service();
    // Find a mass with at least 8 selected (wide window), then use total 5.
    // Tiny domain is tight; use a very wide uncertainty to join many.
    let mut chosen: Option<serde_json::Value> = None;
    for value in (20_000_000u32..120_000_000).step_by(2_000_000) {
        for unc in [2_000_000u32, 10_000_000u32] {
            let mut req = mass_request_neutral();
            req["target_mass"] = serde_json::json!({"units": "microdalton", "value": value, "ppm_tenths": 1000u32, "uncertainty_uda": unc, "source": "synthetic"});
            req["substructures"] = serde_json::json!([]);
            req["formula_search"] =
                serde_json::json!({"hypotheses": 32u32, "nodes_visited_max": 2000000u32});
            req["generation"] = serde_json::json!({"trajectories": 64u32, "temperature": 1.0, "seed": 1u64, "returned": 25u32});
            let out: serde_json::Value =
                serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
            let sel = out["formula_search"]["selected"].as_u64().unwrap_or(0);
            if sel >= 8 {
                // Rebuild with hypotheses 8 for the actual budget test.
                req["formula_search"] =
                    serde_json::json!({"hypotheses": 8u32, "nodes_visited_max": 2000000u32});
                chosen = Some(req);
                break;
            }
        }
        if chosen.is_some() {
            break;
        }
    }
    let mut req = chosen.expect("a mass with >=8 selected");
    req["generation"] = serde_json::json!({"trajectories": 5u32, "temperature": 1.0, "seed": 1u64, "returned": 25u32});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let fs = &out["formula_search"];
    assert_eq!(fs["selected"], 8);
    assert_eq!(fs["sampled"], 5);
    let formulas = fs["formulas"].as_array().unwrap();
    assert_eq!(formulas.len(), 8);
    let sampled = formulas
        .iter()
        .filter(|f| f["trajectories"].as_u64().unwrap() > 0)
        .count();
    let not_sampled = formulas
        .iter()
        .filter(|f| f["sampling"] == "not_sampled")
        .count();
    assert_eq!(sampled, 5);
    assert_eq!(not_sampled, 3);
    for f in formulas
        .iter()
        .filter(|f| f["trajectories"].as_u64().unwrap() > 0)
    {
        assert_eq!(f["trajectories"], 1);
    }
    let acc = &out["accounting"];
    assert_eq!(acc["requested_trajectories"], 5);
    assert_eq!(acc["trajectories"], 5);
}

#[test]
fn mass_absent_element_pattern_gives_no_candidates() {
    let svc = service();
    // Fluorine is absent from every joined formula of the tiny domain.
    let mut f_id: Option<u8> = None;
    for id in 0..18u8 {
        if let Some(t) = chem::atom_type(id) {
            if t.element == 4 {
                f_id = Some(id);
                break;
            }
        }
    }
    let fid = f_id.expect("a fluorine atom type exists");
    let mut req = mass_request_neutral();
    req["substructures"] = serde_json::json!([
        {"atoms": [fid], "bonds": [], "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
         "certainty": "confirmed", "provenance": "synthetic: F"}
    ]);
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["formula_search"]["after_substructures"], 0);
    assert_eq!(out["status"], "no_candidates");
}

#[test]
fn mass_determinism_and_hash_sensitivity() {
    let svc = service();
    let req = mass_request_neutral();
    let a: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let b: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(a["input_hash"], b["input_hash"]);
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
    // Every new field changes the hash.
    let mut changed = req.clone();
    changed["target_mass"]["value"] = serde_json::json!(46041866u32);
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&changed.to_string()).unwrap()).unwrap();
    assert_ne!(a["input_hash"], c["input_hash"], "target_mass.value");
    let mut changed = req.clone();
    changed["target_mass"]["ppm_tenths"] = serde_json::json!(101u32);
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&changed.to_string()).unwrap()).unwrap();
    assert_ne!(a["input_hash"], c["input_hash"], "ppm_tenths");
    let mut changed = req.clone();
    changed["target_mass"]["uncertainty_uda"] = serde_json::json!(51u32);
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&changed.to_string()).unwrap()).unwrap();
    assert_ne!(a["input_hash"], c["input_hash"], "uncertainty");
    let mut changed = req.clone();
    changed["target_mass"]["source"] = serde_json::json!("other");
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&changed.to_string()).unwrap()).unwrap();
    assert_ne!(a["input_hash"], c["input_hash"], "source");
    let mut changed = req.clone();
    changed["neutralization"] = serde_json::json!({"precursor_ion": {"adduct": "[M-H]-"}});
    changed["target_mass"] = serde_json::json!({"units": "microdalton", "value": 45034389u32, "ppm_tenths": 100u32, "uncertainty_uda": 50u32, "source": "synthetic"});
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&changed.to_string()).unwrap()).unwrap();
    assert_ne!(a["input_hash"], c["input_hash"], "neutralization");
    let mut changed = req.clone();
    changed["formula_search"]["hypotheses"] = serde_json::json!(7u32);
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&changed.to_string()).unwrap()).unwrap();
    assert_ne!(a["input_hash"], c["input_hash"], "hypotheses");
    let mut changed = req.clone();
    changed["formula_search"]["nodes_visited_max"] = serde_json::json!(1000000u32);
    let c: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&changed.to_string()).unwrap()).unwrap();
    assert_ne!(a["input_hash"], c["input_hash"], "nodes_visited_max");
}

#[test]
fn mass_pooled_ranking_is_samples_then_logprob_then_formula() {
    let svc = service();
    let req = mass_request_neutral();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let cands = out["candidates"].as_array().unwrap();
    let mut last_samples = u64::MAX;
    for c in cands {
        let s = c["samples"].as_u64().unwrap();
        assert!(s <= last_samples, "non-increasing in samples: {cands:?}");
        last_samples = s;
    }
}

// ---------------------------------------------------------------------------
// MC13: stereo-aware candidates (`stereo-perception-v2`).
// ---------------------------------------------------------------------------

/// Check one candidate's `stereo` block shape: versioned, typed element
/// lists, counts, resolution and the molecule-wide flag.
fn check_stereo_block(stereo: &serde_json::Value, at: &str, expand: usize) {
    assert_eq!(stereo["version"], "stereo-perception-v2", "{at}: version");
    assert_eq!(stereo["assignment"], "unspecified", "{at}: assignment");
    let tetra = stereo["tetrahedral_centers"].as_array().unwrap();
    for (i, t) in tetra.iter().enumerate() {
        assert!(t["atom"].as_u64().is_some(), "{at}: centre[{i}].atom");
        for ligand in t["ligands"].as_array().unwrap() {
            assert!(
                ligand.as_u64().is_some() || ligand == "H",
                "{at}: centre[{i}] ligand {ligand}"
            );
        }
    }
    let bonds = stereo["double_bonds"].as_array().unwrap();
    for (i, b) in bonds.iter().enumerate() {
        assert_eq!(b["atoms"].as_array().unwrap().len(), 2, "{at}: bond[{i}]");
        for ligand in b["reference"].as_array().unwrap() {
            assert!(
                ligand.as_u64().is_some() || ligand == "H" || ligand == "lone_pair",
                "{at}: bond[{i}] reference {ligand}"
            );
        }
    }
    assert!(stereo["not_stereogenic"].as_u64().is_some(), "{at}: count");
    for u in stereo["unsupported"].as_array().unwrap() {
        assert!(u.as_str().is_some(), "{at}: unsupported entry");
    }
    let resolution = stereo["resolution"].as_str().unwrap();
    assert!(
        resolution == "resolved" || resolution.starts_with("unresolved: "),
        "{at}: resolution {resolution}"
    );
    let resolved = resolution == "resolved";
    if resolved {
        let raw = stereo["raw_assignments"].as_u64().unwrap();
        let distinct = stereo["distinct_stereoisomers"].as_u64().unwrap();
        assert_eq!(
            raw,
            1u64 << (tetra.len()
                + bonds.len()
                + stereo["not_stereogenic"].as_u64().unwrap() as usize),
            "{at}: raw is 2^potential"
        );
        assert!((1..=raw).contains(&distinct), "{at}: distinct in range");
        let wide = stereo["molecule_wide_exact"].as_bool().unwrap();
        assert_eq!(
            wide,
            stereo["unsupported"].as_array().unwrap().is_empty(),
            "{at}: molecule_wide_exact"
        );
    } else {
        assert!(stereo["raw_assignments"].is_null(), "{at}: null raw");
        assert!(
            stereo["distinct_stereoisomers"].is_null(),
            "{at}: null distinct"
        );
        assert_eq!(stereo["molecule_wide_exact"], false, "{at}: not exact");
    }
    if expand > 0 {
        let isomers = stereo["stereoisomers"].as_array().unwrap();
        assert!(isomers.len() <= expand, "{at}: at most expand isomers");
        assert!(
            stereo["stereoisomers_truncated"].as_bool().is_some(),
            "{at}: truncation flag"
        );
        for (i, iso) in isomers.iter().enumerate() {
            assert_eq!(
                iso["tetrahedral"].as_array().unwrap().len(),
                tetra.len(),
                "{at}: isomer[{i}] tetrahedral"
            );
            assert_eq!(
                iso["double_bonds"].as_array().unwrap().len(),
                bonds.len(),
                "{at}: isomer[{i}] bonds"
            );
            for v in iso["tetrahedral"].as_array().unwrap() {
                assert!(v == "cw" || v == "ccw", "{at}: isomer[{i}] value {v}");
            }
            for v in iso["double_bonds"].as_array().unwrap() {
                assert!(v == "cis" || v == "trans", "{at}: isomer[{i}] value {v}");
            }
        }
    } else {
        assert!(
            stereo.get("stereoisomers").is_none(),
            "{at}: no stereoisomers without expand"
        );
    }
}

#[test]
fn candidates_carry_stereo_blocks_matching_perceive() {
    use mamba3::models::ms2::stereo::{StereoLimits, perceive};
    let svc = service();
    // Composition requests only; mass requests carry the same block (typed
    // check below covers one).
    let (reqs, _) = load_fixture();
    let reqs: Vec<serde_json::Value> = reqs
        .into_iter()
        .filter(|r| r.get("composition").is_some())
        .collect();
    assert!(!reqs.is_empty());
    for (i, req) in reqs.iter().enumerate() {
        let at = format!("stereo[{i}]");
        let out: serde_json::Value =
            serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
        assert_eq!(
            out["stereochemistry"]["status"], "enumerated_not_predicted",
            "{at}: status"
        );
        assert!(
            out["stereochemistry"]["reason"]
                .as_str()
                .unwrap()
                .contains("`unsupported`"),
            "{at}: reason names unsupported"
        );
        for (r, c) in out["candidates"].as_array().unwrap().iter().enumerate() {
            check_stereo_block(&c["stereo"], &format!("{at}.rank{r}"), 0);
            // The block agrees with a direct perception of the graph.
            let atoms: Vec<u8> = c["atoms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            let bonds: Vec<(usize, usize, u8)> = c["bonds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| {
                    let t = b.as_array().unwrap();
                    (
                        t[0].as_u64().unwrap() as usize,
                        t[1].as_u64().unwrap() as usize,
                        t[2].as_u64().unwrap() as u8,
                    )
                })
                .collect();
            let graph = MolGraph::new(atoms, bonds).unwrap();
            let report = perceive(
                &graph,
                &StereoLimits {
                    max_elements: 10,
                    max_automorphisms: 20_000,
                    work_limit: 100_000,
                    max_expanded: 0,
                },
            );
            assert_eq!(
                c["stereo"]["distinct_stereoisomers"],
                match report.distinct {
                    Some(d) => serde_json::json!(d),
                    None => serde_json::Value::Null,
                },
                "{at}.rank{r}: distinct matches perceive"
            );
        }
    }
}

#[test]
fn stereo_expand_adds_isomers_with_documented_shapes() {
    let svc = service();
    let mut req = first_request();
    req["stereo"] = serde_json::json!({"expand": 3});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    let cands = out["candidates"].as_array().unwrap();
    assert!(!cands.is_empty(), "fixture answers candidates");
    for (r, c) in cands.iter().enumerate() {
        check_stereo_block(&c["stereo"], &format!("expand.rank{r}"), 3);
    }
    // `expand` is part of the hashed input.
    let plain: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&first_request().to_string()).unwrap()).unwrap();
    assert_ne!(
        out["input_hash"], plain["input_hash"],
        "expand changes hash"
    );
}

#[test]
fn stereo_request_schema_rejects_out_of_range_and_unknown() {
    let svc = service();
    let mut doc = first_request();
    doc["stereo"] = serde_json::json!({"expand": 65});
    let err = svc.generate_json(&doc.to_string()).expect_err("expand 65");
    assert!(err.to_string().contains("expand"), "expand range: {err}");
    let mut doc = first_request();
    doc["stereo"] = serde_json::json!({"max_elements": 0});
    let err = svc
        .generate_json(&doc.to_string())
        .expect_err("max_elements 0");
    assert!(
        err.to_string().contains("max_elements"),
        "elements range: {err}"
    );
    let mut doc = first_request();
    doc["stereo"] = serde_json::json!({"max_elements": 13});
    let err = svc
        .generate_json(&doc.to_string())
        .expect_err("max_elements 13");
    assert!(
        err.to_string().contains("max_elements"),
        "elements range: {err}"
    );
    let mut doc = first_request();
    doc["stereo"] = serde_json::json!({"surprise": 1});
    let err = svc
        .generate_json(&doc.to_string())
        .expect_err("unknown stereo field");
    assert!(
        err.to_string().contains("unknown field 'surprise'"),
        "stereo unknown: {err}"
    );
    let mut doc = first_request();
    doc["stereo"] = serde_json::json!({"expand": "3"});
    let err = svc
        .generate_json(&doc.to_string())
        .expect_err("wrong expand type");
    assert!(err.to_string().contains("expand"), "expand type: {err}");
    let mut doc = first_request();
    doc["surprise"] = serde_json::json!(1);
    doc["stereo"] = serde_json::json!({"expand": 1});
    let err = svc
        .generate_json(&doc.to_string())
        .expect_err("top unknown kept");
    assert!(
        err.to_string().contains("unknown field 'surprise'"),
        "top unknown: {err}"
    );
}

#[test]
fn mass_candidates_carry_stereo_blocks() {
    let svc = service();
    let req = mass_request_neutral();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(
        out["stereochemistry"]["status"], "enumerated_not_predicted",
        "mass: stereo status"
    );
    let cands = out["candidates"].as_array().unwrap();
    assert!(!cands.is_empty());
    for (r, c) in cands.iter().enumerate() {
        check_stereo_block(&c["stereo"], &format!("mass.rank{r}"), 0);
        assert!(c["mass"]["formula"].as_str().is_some());
    }
}

#[test]
fn unsupported_input_reports_enumerated_stereo_status() {
    let mut doc = first_request();
    doc["composition"] = serde_json::json!({"C": 13, "H": 28});
    doc["substructures"] = serde_json::json!([]);
    let out: serde_json::Value =
        serde_json::from_str(&service().generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "unsupported_input");
    assert_eq!(
        out["stereochemistry"]["status"], "enumerated_not_predicted",
        "unsupported: stereo status"
    );
}

#[test]
fn fixture_butanol_request_answers_a_stereocentre() {
    // The fixture's `tiny-butanol-c4h10o` request (C4H10O under a
    // 2-butanol fragment) is answered by 2-butanol itself: no retraining
    // was needed, the nine-molecule set and the checkpoint are untouched.
    let (reqs, _) = load_fixture();
    let req = reqs
        .iter()
        .find(|r| r["id"] == "tiny-butanol-c4h10o")
        .expect("butanol request in fixture");
    let out: serde_json::Value =
        serde_json::from_str(&service().generate_json(&req.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "ok");
    let top = &out["candidates"][0];
    assert_eq!(top["composition"], "C4H10O");
    let stereo = &top["stereo"];
    assert_eq!(stereo["resolution"], "resolved");
    assert_eq!(stereo["tetrahedral_centers"].as_array().unwrap().len(), 1);
    assert!(stereo["double_bonds"].as_array().unwrap().is_empty());
    assert_eq!(stereo["distinct_stereoisomers"], 2);
    assert_eq!(stereo["molecule_wide_exact"], true);
    // With expansion the two enantiomers are listed.
    let mut expanded_req = req.clone();
    expanded_req["stereo"] = serde_json::json!({"expand": 2});
    let expanded: serde_json::Value =
        serde_json::from_str(&service().generate_json(&expanded_req.to_string()).unwrap()).unwrap();
    let isomers = expanded["candidates"][0]["stereo"]["stereoisomers"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(isomers.len(), 2);
    let mut values: Vec<String> = isomers
        .iter()
        .map(|iso| iso["tetrahedral"][0].as_str().unwrap().to_string())
        .collect();
    values.sort();
    assert_eq!(values, vec!["ccw".to_string(), "cw".to_string()]);
    assert_eq!(
        expanded["candidates"][0]["stereo"]["stereoisomers_truncated"],
        false
    );
}

#[test]
fn substructure_semantics_three_values_echoed() {
    let svc = service();
    let mut base = first_request();
    // Absent means `contained`, echoed with its one-line meaning.
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&base.to_string()).unwrap()).unwrap();
    assert_eq!(out["substructure_semantics"]["value"], "contained");
    assert_eq!(
        out["substructure_semantics"]["meaning"],
        "every pattern is contained somewhere; patterns may share atoms"
    );
    for (value, meaning) in [
        (
            "contained",
            "every pattern is contained somewhere; patterns may share atoms",
        ),
        (
            "disjoint_occurrences",
            "patterns are distinct occurrences on pairwise disjoint atom sets",
        ),
        (
            "complete_functional_groups",
            "patterns are the molecule's complete functional-group list (functional-groups-ertl-v1)",
        ),
    ] {
        base["substructure_semantics"] = serde_json::json!(value);
        let out: serde_json::Value =
            serde_json::from_str(&svc.generate_json(&base.to_string()).unwrap()).unwrap();
        assert!(
            ["ok", "no_candidates"].contains(&out["status"].as_str().unwrap()),
            "{value}: runnable status"
        );
        assert_eq!(
            out["substructure_semantics"]["value"], value,
            "{value}: echo"
        );
        assert_eq!(
            out["substructure_semantics"]["meaning"], meaning,
            "{value}: meaning"
        );
    }
    // Anything else is a schema error naming the field.
    base["substructure_semantics"] = serde_json::json!("disjoint");
    let err = err_of(&base);
    assert!(
        err.contains("substructure_semantics"),
        "schema error: {err}"
    );
    let mut typed = first_request();
    typed["substructure_semantics"] = serde_json::json!(7);
    assert!(
        err_of(&typed).contains("substructure_semantics"),
        "wrong type names the field"
    );
}

#[test]
fn substructure_semantics_changes_input_hash() {
    let svc = service();
    let base = first_request();
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&base.to_string()).unwrap()).unwrap();
    let mut other = base.clone();
    other["substructure_semantics"] = serde_json::json!("disjoint_occurrences");
    let other_out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&other.to_string()).unwrap()).unwrap();
    assert_ne!(
        out["input_hash"], other_out["input_hash"],
        "the field is covered by input_hash"
    );
}

#[test]
fn infeasible_semantics_returns_no_candidates_with_reason() {
    // C2H6O cannot hold two disjoint hydroxyl oxygens: well-formed but
    // unsatisfiable, so `no_candidates` with an `infeasible` reason (not
    // `unsupported_input`), no sampling, and the echo still present.
    let svc = service();
    let hydroxyl = serde_json::json!({"atoms": [9], "bonds": [], "parent_hydrogen_semantics": "v0_parent_hydrogen_counts", "certainty": "confirmed", "provenance": "s"});
    let mut doc = first_request();
    doc["substructures"] = serde_json::json!([hydroxyl, hydroxyl]);
    doc["substructure_semantics"] = serde_json::json!("disjoint_occurrences");
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "no_candidates");
    assert!(out["candidates"].as_array().unwrap().is_empty());
    let reason = out["infeasible"]["reason"].as_str().unwrap();
    assert!(reason.contains('O'), "the reason names oxygen: {reason}");
    assert_eq!(out["accounting"]["trajectories"], 0);
    assert_eq!(out["accounting"]["requested_trajectories"], 64);
    assert_eq!(
        out["substructure_semantics"]["value"],
        "disjoint_occurrences"
    );
    // The complete mode additionally needs every heteroatom accounted for:
    // one hydroxyl pattern against C2H6O2 leaves an oxygen unexplained.
    let mut doc = first_request();
    doc["composition"] = serde_json::json!({"C": 2, "H": 6, "O": 2});
    doc["substructures"] = serde_json::json!([hydroxyl]);
    doc["substructure_semantics"] = serde_json::json!("complete_functional_groups");
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "no_candidates");
    assert!(out["infeasible"]["reason"].as_str().unwrap().contains('O'));
    // The contained rule never fails the pre-check: the same query runs.
    let mut doc = first_request();
    doc["substructures"] = serde_json::json!([hydroxyl, hydroxyl]);
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&doc.to_string()).unwrap()).unwrap();
    assert!(out.get("infeasible").is_none(), "contained runs");
    assert_eq!(out["accounting"]["trajectories"], 64);
}

#[test]
fn fixture_complete_request_uses_complete_semantics() {
    // The committed `tiny-ethanol-fg-complete` fixture request carries the
    // complete mode; its response echoes it.
    let (reqs, expected) = load_fixture();
    assert_eq!(reqs.len(), 8, "eight fixture requests");
    assert_eq!(expected.len(), 8, "eight fixture responses");
    let pos = reqs
        .iter()
        .position(|r| r["id"] == "tiny-ethanol-fg-complete")
        .expect("complete fixture request");
    assert_eq!(
        reqs[pos]["substructure_semantics"],
        "complete_functional_groups"
    );
    assert_eq!(
        expected[pos]["substructure_semantics"]["value"],
        "complete_functional_groups"
    );
    assert_eq!(
        expected[pos]["substructure_semantics"]["meaning"],
        "patterns are the molecule's complete functional-group list (functional-groups-ertl-v1)"
    );
}

#[test]
fn fingerprint_field_validation_echo_and_unsupported() {
    // The committed tiny model keeps `fingerprint_slots = 0`: any fingerprint
    // is `unsupported_input` naming it. Validation rejects bad names,
    // indices and thresholds before that.
    let svc = service();
    let (reqs, _) = load_fixture();
    let mut base = reqs[0].clone();
    // Bad name.
    let mut doc = base.clone();
    doc["fingerprint"] =
        serde_json::json!({"name": "ecfp4", "bits": [[1, 0.9]], "threshold": 0.1});
    assert!(svc.generate_json(&doc.to_string()).is_err());
    // Bad index.
    let mut doc = base.clone();
    doc["fingerprint"] =
        serde_json::json!({"name": "morgan4096", "bits": [[4096, 0.9]], "threshold": 0.1});
    assert!(svc.generate_json(&doc.to_string()).is_err());
    // Bad threshold.
    let mut doc = base.clone();
    doc["fingerprint"] =
        serde_json::json!({"name": "morgan4096", "bits": [[1, 0.9]], "threshold": 0.0});
    assert!(svc.generate_json(&doc.to_string()).is_err());
    // Unknown field.
    let mut doc = base.clone();
    doc["fingerprint"] = serde_json::json!({
        "name": "morgan4096", "bits": [[1, 0.9]], "threshold": 0.1, "surprise": 1
    });
    assert!(svc.generate_json(&doc.to_string()).is_err());
    // Supported model check happens after validation: the tiny model names
    // the fingerprint in `unsupported`.
    let mut doc = base.clone();
    doc["fingerprint"] =
        serde_json::json!({"name": "morgan4096", "bits": [[1, 0.9], [2, 0.5]], "threshold": 0.1});
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "unsupported_input");
    assert_eq!(out["unsupported"]["limit"], "fingerprint");
    assert_eq!(out["unsupported"]["allowed"], 0);
    assert_eq!(out["unsupported"]["observed"], 2);
    // The echo is present on the unsupported response (slots 0: nothing
    // selected, everything dropped).
    assert_eq!(out["fingerprint"]["name"], "morgan4096");
    assert_eq!(out["fingerprint"]["tokens_used"], 0);
    assert_eq!(out["fingerprint"]["entries_dropped"], 2);
    // Without the field the response carries no fingerprint echo (the
    // committed fixture stays byte-identical).
    let out: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&base.to_string()).unwrap()).unwrap();
    assert!(out.get("fingerprint").is_none());
    // The field is covered by `input_hash`.
    let plain: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&base.to_string()).unwrap()).unwrap();
    let mut doc = base.clone();
    doc["fingerprint"] =
        serde_json::json!({"name": "morgan4096", "bits": [[1, 0.9]], "threshold": 0.1});
    // Unsupported either way, but the hashes differ.
    let _ = plain;
    let h1 = plain["input_hash"].as_str().unwrap().to_string();
    let with_fp: serde_json::Value =
        serde_json::from_str(&svc.generate_json(&doc.to_string()).unwrap()).unwrap();
    assert_ne!(h1, with_fp["input_hash"].as_str().unwrap());
}

#[test]
fn fingerprint_enabled_model_generates_with_echo() {
    // Successful fingerprint-enabled behavior (the committed fixture only
    // exercises the unsupported path): a model with a fingerprint encoder
    // accepts a fingerprint, echoes the selection, and conditions on it
    // (different bits give different candidates).
    use mamba3::models::ms2::completion_data::{ExtractionConfig, PatternSource};
    use mamba3::models::ms2::completion_fingerprint::FingerprintMode;
    use mamba3::models::ms2::completion_model::{CompletionModelConfig, CompletionTrainConfig};
    let device = dev();
    let mut model_config = CompletionModelConfig::tiny();
    model_config.fingerprint_slots = 8;
    let train_config = CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction: ExtractionConfig::default(),
        extraction_seed: 11,
        pattern_source: PatternSource::RandomPatches(ExtractionConfig::default()),
        fingerprint_mode: Some(FingerprintMode::Exact),
        fingerprint_threshold: 0.1,
    };
    let trainer = mamba3::models::ms2::completion_model::CompletionTrainer::<R, f32>::new(
        &model_config,
        &train_config,
        &device,
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!("mc12_fp_{}.ckpt", std::process::id()));
    trainer.save(&path).unwrap();
    let svc = CompletionService::load(&path, &device).unwrap();
    let (reqs, _) = load_fixture();
    let base = reqs.iter().find(|r| r.get("composition").is_some()).unwrap().clone();
    let with_fp = |bits: serde_json::Value| {
        let mut doc = base.clone();
        doc["fingerprint"] = serde_json::json!({
            "name": "morgan4096", "bits": bits, "threshold": 0.1
        });
        serde_json::from_str::<serde_json::Value>(&svc.generate_json(&doc.to_string()).unwrap())
            .unwrap()
    };
    let out_a = with_fp(serde_json::json!([[1, 0.9], [2, 0.5]]));
    assert_ne!(out_a["status"], "unsupported_input");
    assert_eq!(out_a["fingerprint"]["name"], "morgan4096");
    assert_eq!(out_a["fingerprint"]["tokens_used"], 2);
    assert_eq!(out_a["fingerprint"]["entries_dropped"], 0);
    // Different fingerprint bits condition differently.
    let out_b = with_fp(serde_json::json!([[3000, 0.9], [3001, 0.5]]));
    assert!(
        !out_a["candidates"].as_array().unwrap().is_empty()
            || !out_b["candidates"].as_array().unwrap().is_empty(),
        "at least one fingerprint yields candidates"
    );
    assert_ne!(out_a["candidates"], out_b["candidates"]);
}
