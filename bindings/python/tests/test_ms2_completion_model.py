"""Python/Rust parity tests for the trained-model generation API.

Loads the same checkpoint and request/response fixtures as
``tests/ms2_completion_api.rs`` and calls through the PyO3 binding only.
"""

from __future__ import annotations

import json
from pathlib import Path

import mamba3_rl

ROOT = Path(__file__).resolve().parents[3]
FIX_DIR = ROOT / "tests" / "fixtures" / "ms2"
CKPT = FIX_DIR / "completion_tiny.ckpt"
REQS = FIX_DIR / "completion_tiny_requests.json"
RESPS = FIX_DIR / "completion_tiny_responses.json"


def load():
    return json.loads(REQS.read_text()), json.loads(RESPS.read_text())


def model():
    return mamba3_rl.MolecularCompletionModel(str(CKPT))


def assert_close(a, b, path=""):
    if isinstance(a, bool) or isinstance(b, bool):
        assert a is b or a == b, f"{path}: {a!r} vs {b!r}"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            assert a == b, f"{path}: {a} vs {b}"
        else:
            assert abs(float(a) - float(b)) <= 1e-5, f"{path}: {a} vs {b}"
    elif isinstance(a, str):
        assert a == b, f"{path}: {a!r} vs {b!r}"
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a)} vs {set(b)}"
        for k in a:
            assert_close(a[k], b[k], f"{path}.{k}")
    elif a is None:
        assert b is None, f"{path}: None vs {b!r}"
    else:
        raise AssertionError(f"{path}: type {type(a)} vs {type(b)}")


def test_exact_fixture_matches_rust():
    reqs, expected = load()
    assert len(reqs) == 8 and len(expected) == 8
    m = model()
    for i, (req, want) in enumerate(zip(reqs, expected)):
        out = json.loads(m.generate(json.dumps(req)))
        assert_close(out, want, f"fixture[{i}]")


def test_mass_fixture_requests_reproduce():
    reqs, expected = load()
    m = model()
    # The two mass requests are the last two fixture entries.
    for i in (5, 6):
        out = json.loads(m.generate(json.dumps(reqs[i])))
        assert_close(out, expected[i], f"fixture[{i}]")
        assert out["mass_evidence"]["status"] == "accepted"
        formulas = [f["formula"] for f in out["formula_search"]["formulas"]]
        assert "C2H6O" in formulas


def test_mass_rejected_unavailable_and_exclusivity():
    import pytest

    reqs, _ = load()
    m = model()
    mass_req = [r for r in reqs if "target_mass" in r][0]
    # (c) far mass.
    doc = json.loads(json.dumps(mass_req))
    doc["target_mass"]["value"] = 1_000_000_000
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["mass_evidence"]["status"] == "rejected"
    assert out["status"] == "no_candidates"
    # (d) unknown precision.
    doc = json.loads(json.dumps(mass_req))
    doc["target_mass"]["uncertainty_uda"] = None
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["mass_evidence"]["status"] == "unavailable"
    assert out["candidates"] == []
    # (e) both / neither are schema errors.
    doc = json.loads(json.dumps(mass_req))
    doc["composition"] = {"C": 2, "H": 6, "O": 1}
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "exactly one" in str(exc.value)
    doc = json.loads(json.dumps(mass_req))
    del doc["target_mass"]
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "exactly one" in str(exc.value)


def test_rejections_raise_value_error():
    import pytest

    reqs, _ = load()
    m = model()
    doc = dict(reqs[0])
    doc["mass_role"] = "target_fragment"
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "precursor" in str(exc.value)

    doc = json.loads(json.dumps(reqs[0]))
    doc["substructures"][0]["certainty"] = "tentative"
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "tentative" in str(exc.value)

    doc = json.loads(json.dumps(reqs[0]))
    doc["surprise"] = 1
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "unknown field" in str(exc.value)


def test_rejections_cover_every_object_level():
    # Eight cases, one per object level, each mirroring an assertion in
    # rejection_rules in tests/ms2_completion_api.rs.
    import pytest

    reqs, _ = load()
    m = model()

    def fails(mutator, needle):
        doc = json.loads(json.dumps(reqs[0]))
        mutator(doc)
        with pytest.raises(ValueError) as exc:
            m.generate(json.dumps(doc))
        assert needle in str(exc.value), f"{needle}: {exc.value}"

    fails(lambda d: d.update(surprise=1), "unknown field 'surprise'")
    fails(lambda d: d.update(protocol="bogus"), "protocol")
    fails(lambda d: d.update(id=""), "request.id")
    fails(lambda d: d.update(mass_role="molecule"), "mass_role")
    fails(lambda d: d.update(composition={"X": 1, "H": 2}), "X")
    fails(
        lambda d: d["substructures"][0].update(certainty="tentative"),
        "tentative",
    )
    fails(
        lambda d: d["substructures"][0].update(weight=0.5),
        "unknown field 'weight'",
    )
    fails(lambda d: d["generation"].update(trajectories=0), "trajectories")


def test_unsupported_input_for_13_heavy_atoms():
    reqs, _ = load()
    m = model()
    doc = json.loads(json.dumps(reqs[0]))
    doc["composition"] = {"C": 13, "H": 28}
    doc["substructures"] = []
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] == "unsupported_input"
    assert out["candidates"] == []
    # Mirrors unsupported_input_for_13_heavy_atoms in
    # tests/ms2_completion_api.rs: nothing ran.
    acc = out["accounting"]
    assert acc["requested_trajectories"] == 64
    assert acc["trajectories"] == 0
    for key in (
        "finished",
        "dead_end",
        "truncated",
        "other_status",
        "rejected_replay",
        "rejected_containment",
        "containment_unresolved",
        "identity_unresolved",
        "distinct",
    ):
        assert acc[key] == 0, key
    assert out["search"]["status"] == "not_evaluated"
    assert len(out["search"]["reason"]) > 8
    assert out["ranking"]["status"] == "not_evaluated"
    assert len(out["ranking"]["reason"]) > 8
    assert out["unsupported"]["limit"] == "max_atoms"
    assert out["unsupported"]["allowed"] == 12
    assert out["unsupported"]["observed"] == 13


def test_same_request_twice_is_identical_and_seed_changes_hash():
    # Mirrors the determinism / seed-hash block of
    # structural_invariants_determinism_and_seed_hash in
    # tests/ms2_completion_api.rs.
    reqs, _ = load()
    m = model()
    first = json.loads(m.generate(json.dumps(reqs[0])))
    again = json.loads(m.generate(json.dumps(reqs[0])))
    assert_close(first, again, "deterministic")
    other = json.loads(json.dumps(reqs[0]))
    other["generation"]["seed"] = 999_999
    changed = json.loads(m.generate(json.dumps(other)))
    assert first["input_hash"] != changed["input_hash"]


def test_overlapping_patterns_generate_normally():
    # Mirrors overlapping_patterns_are_not_summed in
    # tests/ms2_completion_api.rs: two seven-atom ring patterns (14 atoms
    # total, above max_atoms 12) share one seven-atom target.
    reqs, _ = load()
    m = model()
    ring = {
        "atoms": [3, 3, 3, 3, 3, 3, 3],
        "bonds": [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1], [6, 0, 1]],
        "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
        "certainty": "confirmed",
        "provenance": "synthetic: seven-ring twice",
    }
    doc = json.loads(json.dumps(reqs[0]))
    doc["composition"] = {"C": 7, "H": 14}
    doc["substructures"] = [ring, json.loads(json.dumps(ring))]
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] != "unsupported_input"
    assert out["status"] in ("ok", "no_candidates")


def test_single_pattern_above_composition_heavy_is_unsupported():
    # Mirrors single_pattern_above_composition_heavy_is_unsupported in
    # tests/ms2_completion_api.rs.
    reqs, _ = load()
    m = model()
    doc = json.loads(json.dumps(reqs[0]))
    doc["composition"] = {"C": 2, "H": 6, "O": 1}
    doc["substructures"] = [
        {
            "atoms": [4, 3, 3, 9],
            "bonds": [[0, 1, 1], [1, 2, 1], [2, 3, 1]],
            "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
            "certainty": "confirmed",
            "provenance": "synthetic: too big",
        }
    ]
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] == "unsupported_input"
    assert out["unsupported"]["limit"] == "composition_heavy_atoms"
    assert out["unsupported"]["allowed"] == 3
    assert out["unsupported"]["observed"] == 4


def test_input_hash_ignores_key_order_and_whitespace():
    # Mirrors input_hash_ignores_key_order_and_whitespace in
    # tests/ms2_completion_api.rs.
    reqs, _ = load()
    m = model()
    base = reqs[0]
    base_hash = json.loads(m.generate(json.dumps(base)))["input_hash"]
    reordered = {
        "substructures": [
            {
                "provenance": "synthetic: C(H2)-O(H1)",
                "certainty": "confirmed",
                "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                "bonds": [[0, 1, 1]],
                "atoms": [3, 9],
            }
        ],
        "provenance": "synthetic fixture",
        "protocol": "molecular-completion-generate-v1",
        "mass_role": "target_molecule",
        "id": base["id"],
        "generation": {"seed": 1, "returned": 25, "trajectories": 64, "temperature": 1.0},
        "composition": {"O": 1, "H": 6, "C": 2},
    }
    compact = json.dumps(reordered)
    spaced = json.dumps(reordered, indent=2, sort_keys=True)
    assert compact != spaced
    assert json.loads(m.generate(compact))["input_hash"] == base_hash
    assert json.loads(m.generate(spaced))["input_hash"] == base_hash
    other = json.loads(json.dumps(base))
    other["generation"]["seed"] = 2
    assert json.loads(m.generate(json.dumps(other)))["input_hash"] != base_hash
    other = json.loads(json.dumps(base))
    other["composition"]["C"] = 3
    assert json.loads(m.generate(json.dumps(other)))["input_hash"] != base_hash


def test_huge_bond_index_is_a_schema_error():
    # Mirrors huge_bond_index_is_a_schema_error in
    # tests/ms2_completion_api.rs.
    import pytest

    reqs, _ = load()
    m = model()
    for huge in (2**32, 2**64 - 1):
        doc = json.loads(json.dumps(reqs[0]))
        doc["substructures"][0]["bonds"] = [[0, huge, 1]]
        with pytest.raises(ValueError) as exc:
            m.generate(json.dumps(doc))
        assert "substructures[0]" in str(exc.value)
        assert "atom" in str(exc.value)


def test_truncated_and_non_checkpoint_files_raise():
    # Mirrors truncated_or_foreign_file_is_error_not_panic in
    # tests/ms2_completion_api.rs: errors, never a crash.
    import tempfile

    import pytest

    raw = CKPT.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        trunc = Path(tmp) / "trunc.ckpt"
        trunc.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(Exception):
            mamba3_rl.MolecularCompletionModel(str(trunc))
        garbage = Path(tmp) / "garbage.ckpt"
        garbage.write_text("this is not a checkpoint")
        with pytest.raises(Exception):
            mamba3_rl.MolecularCompletionModel(str(garbage))


def test_describe_names_versions():
    m = model()
    doc = json.loads(m.describe())
    assert doc["protocol"] == mamba3_rl.molecular_completion_generate_protocol()
    assert doc["protocol"] == "molecular-completion-generate-v1"
    assert doc["model_version"] == "completion-model-v1"
    assert doc["grammar_version"] == "completion-exact-v2"
    assert doc["data_version"] == "completion-data-v2"
    assert "chemistry_version" in doc
    assert len(doc["checkpoint_sha256"]) == 64


def _mass_req(idx=4):
    reqs, _ = load()
    return json.loads(json.dumps([r for r in reqs if "target_mass" in r][idx]))


def _wide_req(m, hypotheses=8, trajectories=64):
    # First wide window with >=8 selected and >=1 train-fit exclusion.
    for value in range(20_000_000, 120_000_000, 2_000_000):
        for unc in (2_000_000, 10_000_000):
            doc = _mass_req(0)
            doc["id"] = "py-wide-scan"
            doc["target_mass"] = {
                "units": "microdalton",
                "value": value,
                "ppm_tenths": 1000,
                "uncertainty_uda": unc,
                "source": "synthetic",
            }
            doc["substructures"] = []
            doc["formula_search"] = {"hypotheses": 32, "nodes_visited_max": 2000000}
            doc["generation"] = {
                "trajectories": 64,
                "temperature": 1.0,
                "seed": 1,
                "returned": 25,
            }
            out = json.loads(m.generate(json.dumps(doc)))
            fs = out["formula_search"]
            if fs["selected"] >= 8 and (fs["excluded_by_train_fit"] or 0) >= 1:
                doc["formula_search"] = {
                    "hypotheses": hypotheses,
                    "nodes_visited_max": 2000000,
                }
                doc["generation"] = {
                    "trajectories": trajectories,
                    "temperature": 1.0,
                    "seed": 1,
                    "returned": 25,
                }
                return doc
    raise AssertionError("no wide window with >=8 selected and >=1 exclusion")


def test_mass_error_terms_split():
    m = model()
    out = json.loads(m.generate(json.dumps(_mass_req(0))))
    assert out["mass_evidence"]["error_terms_uda"] == {
        "observation_uda": 50,
        "composition_uda": 1,
        "neutralisation_uda": 0,
    }
    out = json.loads(m.generate(json.dumps(_mass_req(1))))
    assert out["mass_evidence"]["error_terms_uda"] == {
        "observation_uda": 50,
        "composition_uda": 1,
        "neutralisation_uda": 1,
    }


def test_mass_pruning_chemical_only_and_bad_spellings():
    import pytest

    m = model()
    doc = _mass_req(0)
    train = json.loads(m.generate(json.dumps(doc)))
    assert train["formula_search"]["pruning"] == "train_fit"
    doc = _mass_req(0)
    doc["formula_search"]["pruning"] = "chemical_only"
    chem = json.loads(m.generate(json.dumps(doc)))
    assert chem["formula_search"]["pruning"] == "chemical_only"
    assert chem["formula_search"]["joined_chemical"] == chem["formula_search"]["joined"]
    tjoined = train["formula_search"]["joined"]
    cjoined = chem["formula_search"]["joined"]
    assert cjoined >= tjoined
    assert train["formula_search"]["excluded_by_train_fit"] == cjoined - tjoined
    doc = _mass_req(0)
    doc["formula_search"]["pruning"] = "none"
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "pruning" in str(exc.value)
    doc = _mass_req(0)
    doc["formula_search"]["allocation"] = "none"
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "allocation" in str(exc.value)


def test_mass_selection_order_and_redistribution():
    m = model()
    doc = _wide_req(m, hypotheses=8, trajectories=64)
    out = json.loads(m.generate(json.dumps(doc)))
    formulas = out["formula_search"]["formulas"]
    assert len(formulas) >= 2
    keys = [
        (f["verdict"] == "boundary_ambiguous", f["residual_uda"]) for f in formulas
    ]
    assert keys == sorted(keys), "selected in (verdict, residual) order"
    assert "deterministic default, not a formula probability" in out["formula_search"][
        "ranking"
    ]
    # Redistribution: 64 trajectories over 3 hypotheses spend the budget,
    # with differing allocations (22 vs 21: the F1 shape).
    doc = _wide_req(m, hypotheses=3, trajectories=64)
    out = json.loads(m.generate(json.dumps(doc)))
    trajs = [f["trajectories"] for f in out["formula_search"]["formulas"]]
    assert trajs == [22, 21, 21]
    assert out["accounting"]["trajectories"] == 64
    assert out["accounting"]["unused_trajectories"] == 0
    for f in out["formula_search"]["formulas"]:
        assert f["stage"] == "sampled"
        assert abs(f["weight"] - 1.0 / 3.0) < 1e-12
        assert f["mass_status"] in ("accepted", "boundary_ambiguous")
    # F1: pooled candidates carry their own source formula's mass metadata,
    # even when trajectory allocations differ.
    assert out["candidates"], "pooled candidates exist for the invariant"
    for c in out["candidates"]:
        assert c["composition"] == c["mass"]["formula"]
    # F11: the equal allocation ranks by sample frequency.
    assert out["ranking"]["status"] == "sample_frequency"


def test_mass_train_frequency_weights_and_estimate():
    m = model()
    doc = _wide_req(m, hypotheses=8, trajectories=64)
    doc["formula_search"]["allocation"] = "train_frequency"
    out = json.loads(m.generate(json.dumps(doc)))
    fs = out["formula_search"]
    assert fs["allocation"] == "train_frequency"
    assert "training-frequency prior, not a calibrated probability" in fs["ranking"]
    formulas = fs["formulas"]
    assert abs(sum(f["weight"] for f in formulas) - 1.0) < 1e-9
    assert sum(f["trajectories"] for f in formulas) == 64
    assert out["accounting"]["unused_trajectories"] == 0
    for f in formulas:
        assert f["trajectories"] >= 1
        assert f["weight"] > 0.0
    by_formula = {f["formula"]: f for f in formulas}
    ests = [
        by_formula[c["mass"]["formula"]]["weight"]
        * c["samples"]
        / by_formula[c["mass"]["formula"]]["trajectories"]
        for c in out["candidates"]
    ]
    for a, b in zip(ests, ests[1:]):
        assert a + 1e-12 >= b, f"estimate non-increasing: {ests}"
    # F11: the ranking metadata names the rule actually used.
    assert out["ranking"]["status"] == "train_frequency_weighted_estimate"
    assert "weight * samples / trajectories" in out["ranking"]["reason"]
    assert out["ranking"]["calibrated"] is False


def test_mass_evidence_states_and_search_exhaustion():
    # Mirrors the F2/F8 Rust tests: incomplete/overflow/sentinel requests
    # never claim completed rejection; exhaustion beats truncation.
    import pytest

    reqs, _ = load()
    m = model()
    mass_req = [r for r in reqs if "target_mass" in r][0]
    # Exhausted search: search_incomplete + no_candidates, never rejected.
    doc = json.loads(json.dumps(mass_req))
    doc["formula_search"] = {"hypotheses": 8, "nodes_visited_max": 1}
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] == "no_candidates"
    assert out["mass_evidence"]["status"] == "search_incomplete"
    assert out["formula_search"]["status"] == "search_exhausted"
    assert out["formula_search"]["search_exhausted"] is True
    # Precursor neutralisation leaving u32: mass_overflow, never zero mass.
    doc = json.loads(json.dumps(mass_req))
    doc["target_mass"]["value"] = 4294967295
    doc["neutralization"] = {"precursor_ion": {"adduct": "[M-H]-"}}
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] == "no_candidates"
    assert out["mass_evidence"]["status"] == "mass_overflow"
    assert out["candidates"] == []
    # The u32::MAX integer sentinel is a schema error (use null).
    doc = json.loads(json.dumps(mass_req))
    doc["target_mass"]["uncertainty_uda"] = 4294967295
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "uncertainty_uda" in str(exc.value)


def _check_stereo_block(stereo, at, expand):
    assert stereo["version"] == "stereo-perception-v2", at
    assert stereo["assignment"] == "unspecified", at
    tetra = stereo["tetrahedral_centers"]
    bonds = stereo["double_bonds"]
    for t in tetra:
        assert isinstance(t["atom"], int), f"{at}: centre atom"
        for ligand in t["ligands"]:
            assert isinstance(ligand, int) or ligand == "H", f"{at}: ligand {ligand}"
    for b in bonds:
        assert len(b["atoms"]) == 2, f"{at}: bond atoms"
        for ligand in b["reference"]:
            assert isinstance(ligand, int) or ligand in ("H", "lone_pair"), (
                f"{at}: reference {ligand}"
            )
    assert isinstance(stereo["not_stereogenic"], int), at
    assert isinstance(stereo["unsupported"], list), at
    resolution = stereo["resolution"]
    assert resolution == "resolved" or resolution.startswith("unresolved: "), (
        f"{at}: {resolution}"
    )
    if resolution == "resolved":
        raw = stereo["raw_assignments"]
        distinct = stereo["distinct_stereoisomers"]
        assert raw == 2 ** (len(tetra) + len(bonds) + stereo["not_stereogenic"]), at
        assert 1 <= distinct <= raw, at
        assert stereo["molecule_wide_exact"] == (stereo["unsupported"] == []), at
    else:
        assert stereo["raw_assignments"] is None, at
        assert stereo["distinct_stereoisomers"] is None, at
        assert stereo["molecule_wide_exact"] is False, at
    if expand > 0:
        isomers = stereo["stereoisomers"]
        assert len(isomers) <= expand, at
        assert isinstance(stereo["stereoisomers_truncated"], bool), at
        for iso in isomers:
            assert len(iso["tetrahedral"]) == len(tetra), at
            assert len(iso["double_bonds"]) == len(bonds), at
            for v in iso["tetrahedral"]:
                assert v in ("cw", "ccw"), f"{at}: {v}"
            for v in iso["double_bonds"]:
                assert v in ("cis", "trans"), f"{at}: {v}"
    else:
        assert "stereoisomers" not in stereo, at


def test_stereo_block_shape_and_expand():
    # Mirrors candidates_carry_stereo_blocks_matching_perceive and
    # stereo_expand_adds_isomers_with_documented_shapes in
    # tests/ms2_completion_api.rs.
    reqs, _ = load()
    m = model()
    comp_reqs = [r for r in reqs if "composition" in r]
    assert comp_reqs
    for i, req in enumerate(comp_reqs):
        out = json.loads(m.generate(json.dumps(req)))
        assert out["stereochemistry"]["status"] == "enumerated_not_predicted"
        assert "no preference" in out["stereochemistry"]["reason"]
        for r, c in enumerate(out["candidates"]):
            _check_stereo_block(c["stereo"], f"stereo[{i}].rank{r}", 0)
    doc = json.loads(json.dumps(reqs[0]))
    doc["stereo"] = {"expand": 3}
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["candidates"]
    for r, c in enumerate(out["candidates"]):
        _check_stereo_block(c["stereo"], f"expand.rank{r}", 3)
    plain = json.loads(m.generate(json.dumps(reqs[0])))
    assert out["input_hash"] != plain["input_hash"]


def test_stereo_butanol_request_answers_a_stereocentre():
    # Mirrors fixture_butanol_request_answers_a_stereocentre in
    # tests/ms2_completion_api.rs: the C4H10O fixture request is answered
    # by 2-butanol without any retraining.
    reqs, _ = load()
    m = model()
    req = next(r for r in reqs if r["id"] == "tiny-butanol-c4h10o")
    out = json.loads(m.generate(json.dumps(req)))
    assert out["status"] == "ok"
    stereo = out["candidates"][0]["stereo"]
    assert stereo["resolution"] == "resolved"
    assert len(stereo["tetrahedral_centers"]) == 1
    assert stereo["double_bonds"] == []
    assert stereo["distinct_stereoisomers"] == 2
    assert stereo["molecule_wide_exact"] is True
    doc = json.loads(json.dumps(req))
    doc["stereo"] = {"expand": 2}
    expanded = json.loads(m.generate(json.dumps(doc)))
    isomers = expanded["candidates"][0]["stereo"]["stereoisomers"]
    assert sorted(i["tetrahedral"][0] for i in isomers) == ["ccw", "cw"]
    assert expanded["candidates"][0]["stereo"]["stereoisomers_truncated"] is False


def test_stereo_schema_rejections():
    # Mirrors stereo_request_schema_rejects_out_of_range_and_unknown in
    # tests/ms2_completion_api.rs.
    import pytest

    reqs, _ = load()
    m = model()

    def fails(mutator, needle):
        doc = json.loads(json.dumps(reqs[0]))
        mutator(doc)
        with pytest.raises(ValueError) as exc:
            m.generate(json.dumps(doc))
        assert needle in str(exc.value), f"{needle}: {exc.value}"

    fails(lambda d: d.update(stereo={"expand": 65}), "expand")
    fails(lambda d: d.update(stereo={"max_elements": 0}), "max_elements")
    fails(lambda d: d.update(stereo={"max_elements": 13}), "max_elements")
    fails(lambda d: d.update(stereo={"surprise": 1}), "unknown field 'surprise'")


def test_substructure_semantics_echo_and_values():
    # Mirrors substructure_semantics_three_values_echoed in
    # tests/ms2_completion_api.rs.
    import pytest

    reqs, _ = load()
    m = model()
    out = json.loads(m.generate(json.dumps(reqs[0])))
    assert out["substructure_semantics"]["value"] == "contained"
    assert out["substructure_semantics"]["meaning"] == (
        "every pattern is contained somewhere; patterns may share atoms"
    )
    meanings = {
        "contained": "every pattern is contained somewhere; patterns may share atoms",
        "disjoint_occurrences": (
            "patterns are distinct occurrences on pairwise disjoint atom sets"
        ),
        "complete_functional_groups": (
            "patterns are the molecule's complete functional-group list "
            "(functional-groups-ertl-v1)"
        ),
    }
    for value, meaning in meanings.items():
        doc = json.loads(json.dumps(reqs[0]))
        doc["substructure_semantics"] = value
        out = json.loads(m.generate(json.dumps(doc)))
        assert out["status"] in ("ok", "no_candidates"), value
        assert out["substructure_semantics"]["value"] == value
        assert out["substructure_semantics"]["meaning"] == meaning
    doc = json.loads(json.dumps(reqs[0]))
    doc["substructure_semantics"] = "disjoint"
    with pytest.raises(ValueError) as exc:
        m.generate(json.dumps(doc))
    assert "substructure_semantics" in str(exc.value)


def test_substructure_semantics_infeasible():
    # Mirrors infeasible_semantics_returns_no_candidates_with_reason in
    # tests/ms2_completion_api.rs: C2H6O cannot hold two disjoint hydroxyls.
    reqs, _ = load()
    m = model()
    hydroxyl = {
        "atoms": [9],
        "bonds": [],
        "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
        "certainty": "confirmed",
        "provenance": "s",
    }
    doc = json.loads(json.dumps(reqs[0]))
    doc["substructures"] = [hydroxyl, json.loads(json.dumps(hydroxyl))]
    doc["substructure_semantics"] = "disjoint_occurrences"
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] == "no_candidates"
    assert out["candidates"] == []
    assert "O" in out["infeasible"]["reason"]
    assert out["accounting"]["trajectories"] == 0
    assert out["accounting"]["requested_trajectories"] == 64
    # The committed complete-mode fixture request echoes its semantics.
    req = next(r for r in reqs if r["id"] == "tiny-ethanol-fg-complete")
    out = json.loads(m.generate(json.dumps(req)))
    assert out["substructure_semantics"]["value"] == "complete_functional_groups"


def test_fingerprint_unsupported_and_validation():
    # Mirrors fingerprint_field_validation_echo_and_unsupported in
    # tests/ms2_completion_api.rs: the tiny model has no fingerprint encoder.
    import pytest

    reqs, _ = load()
    m = model()
    doc = json.loads(json.dumps(reqs[0]))
    doc["fingerprint"] = {"name": "morgan4096", "bits": [[1, 0.9], [2, 0.5]], "threshold": 0.1}
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] == "unsupported_input"
    assert out["unsupported"]["limit"] == "fingerprint"
    assert out["fingerprint"]["name"] == "morgan4096"
    plain = json.loads(m.generate(json.dumps(reqs[0])))
    assert "fingerprint" not in plain
    assert plain["input_hash"] != out["input_hash"]
    for bad in (
        {"name": "ecfp4", "bits": [[1, 0.9]], "threshold": 0.1},
        {"name": "morgan4096", "bits": [[4096, 0.9]], "threshold": 0.1},
        {"name": "morgan4096", "bits": [[1, 0.9]], "threshold": 0.0},
    ):
        doc = json.loads(json.dumps(reqs[0]))
        doc["fingerprint"] = bad
        with pytest.raises(ValueError):
            m.generate(json.dumps(doc))


def _spectrum_request(doc, adduct="[M+H]+"):
    doc = json.loads(json.dumps(doc))
    doc["spectrum"] = {
        "peaks": [[15000000, 100.0], [22000000, 40.0], [31000000, 10.0]],
        "precursor_mz_uda": 47049141,
        "adduct": adduct,
    }
    return doc


def test_spectrum_unsupported_and_validation():
    # Mirrors json_protocol_carries_spectral_evidence in
    # tests/ms2_completion_spectrum.rs: the tiny model has no spectrum encoder,
    # so evidence is unsupported_input naming the field, never ignored.
    import pytest

    reqs, _ = load()
    m = model()
    out = json.loads(m.generate(json.dumps(_spectrum_request(reqs[0]))))
    assert out["status"] == "unsupported_input"
    assert out["unsupported"]["limit"] == "spectrum"
    assert out["accounting"]["trajectories"] == 0
    assert out["spectrum"]["adduct"] == "[M+H]+"
    assert out["spectrum"]["neutral_mass_uda"] == 46041865
    plain = json.loads(m.generate(json.dumps(reqs[0])))
    assert "spectrum" not in plain
    assert plain["input_hash"] != out["input_hash"]

    def change(key, value):
        doc = _spectrum_request(reqs[0])
        doc["spectrum"][key] = value
        return doc

    for bad in (
        change("extra", 1),
        change("adduct", "[M+Li]+"),
        change("adduct", "unknown"),
        change("precursor_mz_uda", 0),
        change("peaks", "none"),
        change("peaks", [[0, 1.0]]),
        change("peaks", [[15000000, -1.0]]),
    ):
        with pytest.raises(ValueError):
            m.generate(json.dumps(bad))


def test_spectrum_checkpoint_generates_from_all_four_inputs():
    # A checkpoint with the fingerprint and spectrum encoders, named by
    # MAMBA3_SPECTRUM_CKPT (a trained run of examples/ms2_spectral_completion;
    # not committed): fingerprint, peaks, adduct and neutral mass in one
    # request. Every candidate must satisfy the mass.
    import os

    import pytest

    path = os.environ.get("MAMBA3_SPECTRUM_CKPT")
    if not path:
        pytest.skip("MAMBA3_SPECTRUM_CKPT is not set")
    m = mamba3_rl.MolecularCompletionModel(path)
    neutral = 242049724
    doc = {
        "protocol": "molecular-completion-generate-v1",
        "id": "python-four-inputs",
        "provenance": "python parity test",
        "mass_role": "target_molecule",
        "target_mass": {"units": "microdalton", "value": neutral, "ppm_tenths": 100,
                        "uncertainty_uda": 51, "source": "precursor minus adduct shift"},
        "neutralization": "already_neutral",
        "formula_search": {"hypotheses": 8, "nodes_visited_max": 2000000},
        "substructures": [],
        "generation": {"trajectories": 64, "temperature": 1.0, "seed": 7, "returned": 25},
        "fingerprint": {"name": "morgan4096", "threshold": 0.1,
                        "bits": [[b, 1.0] for b in (25, 31, 216, 389, 561, 1088, 1308, 1380)]},
        "spectrum": {"peaks": [[243057100, 1.0], [165010400, 0.56], [105033600, 0.12]],
                     "precursor_mz_uda": 243057000, "adduct": "[M+H]+"},
    }
    out = json.loads(m.generate(json.dumps(doc)))
    assert out["status"] in ("ok", "no_candidates"), out.get("unsupported")
    assert out["spectrum"] == {"peaks_used": 3, "peaks_dropped": 0, "adduct": "[M+H]+",
                               "precursor_mz_uda": 243057000, "neutral_mass_uda": neutral}
    assert out["fingerprint"]["tokens_used"] == 8
    assert out["accounting"]["trajectories"] == 64
    tolerance = neutral * 10 // 1000000 + 51 + 1
    for candidate in out["candidates"]:
        assert abs(candidate["mass"]["computed_uda"] - neutral) <= tolerance
    again = json.loads(m.generate(json.dumps(doc)))
    assert again["candidates"] == out["candidates"]

