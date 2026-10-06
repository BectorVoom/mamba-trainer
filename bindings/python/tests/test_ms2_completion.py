"""Python/Rust feature-parity tests for the bounded completion audit API.

The fixtures are the shared provenance file of the audit; every check here
calls through the authoritative Rust implementation via the PyO3 binding
(mamba3_rl.completion_run / completion_run_fixture_set). No parallel
implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import mamba3_rl

FIX = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "molecular_completion"
    / "20261004_completion_ambiguity"
    / "fixtures.json"
)


def load():
    return json.loads(FIX.read_text())


def test_fixture_set_matches_expectations_via_rust_api():
    results = mamba3_rl.completion_run_fixture_set(FIX.read_text())
    expected = {f["name"]: f["expected"] for f in load()["fixtures"]}
    ok = True
    for name, report in results:
        rep = json.loads(report)
        want = expected[name]
        ok &= rep["status"] == want["status"]
        ok &= rep["mass_status"] == want["mass_status"]
        ok &= rep["unique_graphs"] == want["unique_graphs"]
        ok &= rep["accepted_formulas"] == want["accepted_formulas"]
        ok &= rep["certifies_zero"] == want["certifies_zero"]
        if want.get("recovery") is not None:
            ok &= rep["recovery"] == want["recovery"]
        for reason in want.get("termination_reasons", []):
            ok &= reason in rep["termination_reasons"]
    assert ok


def query_by_name(name):
    return next(f for f in load()["fixtures"] if f["name"] == name)


def run_query(query: dict, **overrides):
    payload = dict(query)
    payload.pop("expected", None)
    if overrides:
        payload["budgets"] = overrides.pop("budgets", {})
    if overrides:
        payload.update(overrides)
    return json.loads(mamba3_rl.completion_run(json.dumps(payload)))


def test_normal_path_mass_accepted():
    report = run_query(query_by_name("c2h6o_mass_only"))
    assert report["status"] == "complete"
    assert report["mass_status"] == "accepted"
    assert report["unique_graphs"] == 2
    assert report["accepted_formulas"] == ["C2H6O"]
    assert report["certifies_zero"] is False
    assert report["precursor"]["status"] == "not_evaluated"


EXAMPLE_REQUEST = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "molecular_completion_request.json"
)


def load_example():
    return json.loads(EXAMPLE_REQUEST.read_text())


def test_molecular_completion_run_parity_with_rust():
    req = json.dumps(load_example())
    out = json.loads(mamba3_rl.molecular_completion_run(req))
    assert out["protocol"] == "molecular-completion-request-v1"
    assert out["ranking"]["status"] == "not_evaluated"
    assert out["physical_verification"]["status"] == "not_evaluated"
    assert out["audit"]["protocol"] == "completion-bounded-v1"
    assert out["audit"]["status"] == "complete"
    assert out["audit"]["mass_status"] == "accepted"
    assert out["audit"]["unique_graphs"] == 1
    assert out["audit"]["accepted_formulas"] == ["C2H6O"]
    assert out["decoded_graphs"][0]["composition"] == "C2H6O"
    # Deterministic: identical request gives identical response modulo
    # wall-clock/memory counters.
    again = json.loads(mamba3_rl.molecular_completion_run(req))
    for obj in (out, again):
        obj["audit"]["elapsed_ms"] = 0
        obj["audit"]["counters"]["memory_estimate_bytes"] = 0
    assert out == again


def test_molecular_completion_run_strict_errors_match_rust():
    import pytest

    doc = load_example()
    doc["mass_role"] = "target_fragment"
    with pytest.raises((ValueError, NotImplementedError)) as exc:
        mamba3_rl.molecular_completion_run(json.dumps(doc))
    assert "precursor" in str(exc.value)

    doc = load_example()
    doc["substructures"][0]["certainty"] = "tentative"
    with pytest.raises((ValueError, NotImplementedError)) as exc:
        mamba3_rl.molecular_completion_run(json.dumps(doc))
    assert "tentative" in str(exc.value)

    doc = load_example()
    doc["target_mass"]["uncertainty_uda"] = None
    out = json.loads(mamba3_rl.molecular_completion_run(json.dumps(doc)))
    assert out["audit"]["mass_status"] == "unavailable"
    assert out["audit"]["certifies_zero"] is False

    doc = load_example()
    doc["correspondence"] = []
    out = json.loads(mamba3_rl.molecular_completion_run(json.dumps(doc)))
    assert out["audit"]["status"] == "complete"


def test_unknown_precision_is_unavailable_never_zero():
    report = run_query(query_by_name("c2h6o_precision_unavailable"))
    assert report["status"] == "mass_evidence_unresolved"
    assert report["mass_status"] == "unavailable"
    assert report["unique_graphs"] == 0
    assert report["certifies_zero"] is False
    assert report["unresolved_hypotheses"] > 0


def test_invalid_input_surfaces_unsupported_input():
    payload = query_by_name("c2h6o_mass_only")
    payload.pop("expected", None)
    payload["domain"]["elements"] = ["Cl"]  # Not in the C/N/O vocabulary
    import pytest

    with pytest.raises(Exception):
        mamba3_rl.completion_run(json.dumps(payload))


def test_exhausted_budget_is_lower_bound():
    payload = query_by_name("c2h6o_mass_only")
    payload.pop("expected", None)
    payload["budgets"] = {"retained_graphs": 1}
    report = json.loads(mamba3_rl.completion_run(json.dumps(payload)))
    assert report["status"] == "search_budget_exhausted"
    assert "retained_graph_limit" in report["termination_reasons"]
    assert report["unique_graphs"] == 1
    assert report["certifies_zero"] is False


def test_zero_rejected_mass_certifies_within_domain():
    report = run_query(query_by_name("c2h6o_mass_far_off"))
    assert report["status"] == "complete"
    assert report["mass_status"] == "rejected"
    assert report["unique_graphs"] == 0
    assert report["certifies_zero"] is True


def test_known_disjoint_oracle_reduces_unknown_overlap_candidates():
    unknown = run_query(query_by_name("c2h6o_two_methyls_unknown_overlap"))
    known = run_query(query_by_name("c2h6o_two_methyls_known_disjoint"))
    assert (unknown["status"], known["status"]) == ("complete", "complete")
    assert (unknown["unique_graphs"], known["unique_graphs"]) == (2, 1)
    assert set(known["accepted_identities"]) < set(unknown["accepted_identities"])
