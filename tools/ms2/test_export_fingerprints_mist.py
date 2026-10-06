"""Tests for `export_fingerprints_mist.py` (task MC20a).

Runnable with:
    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/test_export_fingerprints_mist.py

No competition dataset needed: the fingerprint is checked against the
vendored definition spelled out independently, `noise` is checked on a tiny
hand-computed synthetic matrix, and `panel`/`noise` wiring is checked on
synthetic pickle/TSV/parquet files in a temporary directory. The tool is
called as an imported module (no subprocess). Plain asserts only, plus a
`main()` so the file runs with and without pytest.
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

from rdkit.Chem import AllChem, DataStructs
from rdkit import Chem

import pyarrow as pa
import pyarrow.parquet as pq

import export_fingerprints_mist as ex

SMALL = ["CCO", "c1ccccc1", "CC(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"]


def vendored_call_directly(smiles: str) -> list[int]:
    """The vendored `morgan4096` definition, spelled out independently."""
    mol = Chem.MolFromSmiles(smiles)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
    arr = np.zeros((4096,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return sorted(int(i) for i in np.nonzero(arr)[0])


def test_fingerprint_matches_vendored_definition():
    assert ex.FINGERPRINT == "morgan4096"
    assert ex.N_BITS == 4096
    assert ex.DEFINITION == "AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096)"
    for smi in SMALL:
        assert ex.fingerprint_bits(smi) == vendored_call_directly(smi), smi
    print("PASS fingerprint of small molecules equals the vendored definition")


def test_bits_deterministic_and_sorted():
    for smi in SMALL:
        first, second = ex.fingerprint_bits(smi), ex.fingerprint_bits(smi)
        assert first == second, smi
        assert first == sorted(first) and len(set(first)) == len(first), smi
        assert all(0 <= b < 4096 for b in first), smi
    assert ex.entry_key("ABCDEF12345678", 12) == "ABCDEF12345678|12"
    print("PASS bits deterministic, sorted, in range; entry key is <key>|<identity_group>")


def test_noise_synthetic_hand_computed():
    # Two spectra, each its own molecule; true-on probs [0.92, 0.83, 0.61],
    # true-off probs [0.12, 0.24, 0.37].
    truth = [{0, 2}, {1}]
    probs = np.array([[0.92, 0.12, 0.83],
                      [0.24, 0.61, 0.37]])
    stats = ex.compute_noise(truth, probs, np.array([0, 1]))
    assert stats["n_spectra"] == 2 and stats["n_molecules"] == 2
    assert stats["mean_true_on_bits_spectrum"] == 1.5
    assert stats["mean_true_on_bits_molecule"] == 1.5
    assert abs(stats["mean_pred_prob_mass_spectrum"] - 1.545) < 1e-12

    on = [0] * 20
    for b in (12, 16, 18):
        on[b] = 1
    off = [0] * 20
    for b in (2, 4, 7):
        off[b] = 1
    assert stats["hist_pred_given_true_on"] == on, stats["hist_pred_given_true_on"]
    assert stats["hist_pred_given_true_off"] == off, stats["hist_pred_given_true_off"]

    for level in ("spectrum", "molecule"):
        t = stats[f"tanimoto_{level}"]
        assert (t["mean"], t["median"], t["p10"], t["p90"], t["n"]) == \
            (1.0, 1.0, 1.0, 1.0, 2), (level, t)
        pr = stats[f"precision_recall_{level}"]
        assert pr["0.5"] == {"precision": 1.0, "recall": 1.0}, pr["0.5"]
        assert pr["0.7"] == {"precision": 1.0, "recall": 0.5}, pr["0.7"]
        assert pr["0.1"] == {"precision": 0.5, "recall": 1.0}, pr["0.1"]
        assert pr["0.3"] == {"precision": 0.75, "recall": 1.0}, pr["0.3"]
    print("PASS noise on synthetic matrix gives the hand-computed histogram/precision/recall")


def test_build_bits_dedup_first_wins():
    molecules = [
        {"key": "K1", "smiles": "CCO", "identity_group": 3},
        {"key": "K2", "smiles": "c1ccccc1", "identity_group": 5},
        {"key": "K1", "smiles": "CCO", "identity_group": 3},  # identical dup
        {"key": "K1", "smiles": "CC(=O)O", "identity_group": 4},  # other group
    ]
    bits, stats, by_molecule = ex.build_bits(molecules)
    assert stats == {"n_molecules": 4, "n_unique_keys": 3,
                     "n_duplicate_keys": 1, "n_duplicate_molecules": 1,
                     "duplicate_keys_fp_identical": 1}, stats
    assert bits["K1|3"] == ex.fingerprint_bits("CCO")
    assert bits["K1|4"] == ex.fingerprint_bits("CC(=O)O")
    assert len(by_molecule) == 4
    assert by_molecule[0] == ex.fingerprint_bits("CCO")
    assert by_molecule[2] == ex.fingerprint_bits("CCO")
    assert by_molecule[3] == ex.fingerprint_bits("CC(=O)O")
    print("PASS build_bits dedups duplicate keys, first in file order wins")


def write_synthetic_inputs(root: Path) -> dict:
    """Two compounds (A: 2 spectra, B: 1 spectrum), 4096-bit probs."""
    rng = np.random.default_rng(7)
    names = np.array(["s1", "s2", "s3"])
    preds = np.zeros((3, 4096))
    preds[0, [10, 200]] = [0.9, 0.8]
    preds[1, [10, 3000]] = [0.6, 0.05]
    preds[2, [500]] = 0.95
    preds += rng.random((3, 4096)) * 1e-4
    with open(root / "preds.p", "wb") as fh:
        pickle.dump({"names": names, "preds": preds, "targs": [None] * 3}, fh)
    (root / "labels.tsv").write_text(
        "spec\tformula\tionization\tdataset\tcompound\tparentmass\tinstrument\n"
        "s1\tC2H6O\t[M+H]+\t-panel_\tAAAAAAAAAAAAAAAA\t47.0497\tunknown\n"
        "s2\tC2H6O\t[M+H]+\t-panel_\tAAAAAAAAAAAAAAAA\t47.0497\tunknown\n"
        "s3\tC6H6\t[M+H]+\t-panel_\tBBBBBBBBBBBBBBBB\t79.0548\tunknown\n")
    table = pa.table({
        "inchikey14": ["AAAAAAAAAAAAAAAA", "BBBBBBBBBBBBBBBB"],
        "smiles": ["CCO", "c1ccccc1"],
        "formula": ["C2H6O", "C6H6"],
        "n_heavy": [3, 6],
        "identity_group": [7, 9],
        "fold_identity": [2, 1],
    })
    pq.write_table(table, root / "structures.parquet")
    return {"preds": root / "preds.p", "labels": root / "labels.tsv",
            "structures": root / "structures.parquet"}


def test_noise_wiring():
    tmp = Path(tempfile.mkdtemp())
    src = write_synthetic_inputs(tmp)
    payload = ex.cmd_noise(src["preds"], src["labels"], src["structures"],
                           tmp / "noise.json")
    assert payload["n_spectra"] == 3 and payload["n_molecules"] == 2
    assert payload["fingerprint"] == "morgan4096"
    assert payload["thresholds"] == [0.1, 0.3, 0.5, 0.7]
    assert len(payload["hist_pred_given_true_on"]) == 20
    assert sum(payload["hist_pred_given_true_on"]) + \
        sum(payload["hist_pred_given_true_off"]) == 3 * 4096
    doc = json.loads((tmp / "noise.json").read_text())
    assert doc["n_spectra"] == 3
    print("PASS noise wiring: counts, histogram mass, thresholds")


def test_panel_schema_and_exclusion_list():
    tmp = Path(tempfile.mkdtemp())
    src = write_synthetic_inputs(tmp)
    report = ex.cmd_panel(src["preds"], src["labels"], src["structures"],
                          tmp / "panel.json")
    doc = json.loads((tmp / "panel.json").read_text())
    assert doc["schema_version"] == 1 and doc["chemistry"] == "ms2-chem-v0.1"
    assert doc["panel_identity_groups"] == [7, 9]
    assert report["written"] == 2 and report["n_found"] == 2
    by_key = {m["key"]: m for m in doc["molecules"]}
    assert set(by_key) == {"AAAAAAAAAAAAAAAA", "BBBBBBBBBBBBBBBB"}
    for m in doc["molecules"]:
        for field in ("key", "smiles", "identity_group", "fold_identity",
                      "atoms", "bonds", "fp_true", "fp_pred_mean",
                      "spectra", "formula"):
            assert field in m, field
        assert m["fp_true"] == ex.fingerprint_bits(m["smiles"])
        assert m["fp_true"] == sorted(m["fp_true"])
        bits = [b for b, _ in m["fp_pred_mean"]]
        assert bits == sorted(bits)
        assert all(p >= 0.01 for _, p in m["fp_pred_mean"])
        assert len(m["atoms"]) > 0 and len(m["bonds"]) > 0
    assert by_key["AAAAAAAAAAAAAAAA"]["spectra"] == 2
    assert by_key["BBBBBBBBBBBBBBBB"]["spectra"] == 1
    print("PASS panel writes the molecule-export schema plus the exclusion list")


def test_bits_keys_by_molecule_binding():
    # Alignment binding (review answer 1): bits files carry the export's
    # ordered entry keys so a reordered same-length sidecar is detectable.
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    export = tmp / "export.json"
    molecules = [
        {"key": "K1", "smiles": "CCO", "identity_group": 3},
        {"key": "K2", "smiles": "c1ccccc1", "identity_group": 5},
    ]
    export.write_text(json.dumps({"molecules": molecules}))
    orig = ex.vendored_bits
    ex.vendored_bits = lambda smi, d: ex.fingerprint_bits(smi)
    try:
        ex.cmd_bits(export, tmp / "bits.json", Path("/nonexistent"))
    finally:
        ex.vendored_bits = orig
    doc = json.loads((tmp / "bits.json").read_text())
    assert doc["keys_by_molecule"] == ["K1|3", "K2|5"], doc["keys_by_molecule"]
    assert len(doc["bits_by_molecule"]) == 2
    assert doc["bits_by_molecule"][0] == ex.fingerprint_bits("CCO")
    print("PASS bits files bind the export order with keys_by_molecule")




def test_noise_empty_truth_no_crash():
    # Finding 9 (additional defect): compute_noise crashed when all truth
    # sets were empty (concatenate of an empty list). Review's in-memory repro.
    truth = [set(), set()]
    probs = np.zeros((2, 4096))
    stats = ex.compute_noise(truth, probs, np.array([0, 1]))
    assert stats["n_spectra"] == 2 and stats["n_molecules"] == 2
    assert stats["hist_pred_given_true_on"] == [0] * 20
    assert sum(stats["hist_pred_given_true_off"]) == 2 * 4096
    assert stats["hist_pred_given_true_on_molecule"] == [0] * 20
    assert sum(stats["hist_pred_given_true_off_molecule"]) == 2 * 4096
    print("PASS noise with all-empty truth does not crash; histograms defined")


def test_noise_molecule_histograms():
    # Finding 5: the noise file carries both histogram sets. Two spectra of
    # one molecule: spectrum histograms pool 2 rows, molecule histograms pool
    # the single averaged row.
    truth = [{0}, {0}]
    probs = np.array([[0.9] + [0.0] * 4095, [0.1] + [0.0] * 4095])
    stats = ex.compute_noise(truth, probs, np.array([0, 0]))
    assert stats["n_spectra"] == 2 and stats["n_molecules"] == 1
    assert sum(stats["hist_pred_given_true_on"]) == 2
    assert sum(stats["hist_pred_given_true_on_molecule"]) == 1
    assert sum(stats["hist_pred_given_true_off"]) == 2 * 4095
    assert sum(stats["hist_pred_given_true_off_molecule"]) == 4095
    # The averaged on-probability (0.5) lands in bin 10, unlike the two
    # spectrum values (bins 18 and 2).
    assert stats["hist_pred_given_true_on_molecule"][10] == 1
    print("PASS noise exports per-spectrum and per-molecule histograms")


def test_pred_bundle_validation():
    # Finding 9 (additional): width / bounds / length mismatches are errors,
    # never silent zip truncation.
    good = {"names": ["a", "b"], "preds": np.zeros((2, 4096))}
    ex.validate_pred_bundle(good)
    for bad, why in [
        ({"names": ["a"], "preds": np.zeros((2, 4096))}, "length"),
        ({"names": ["a"], "preds": np.zeros((1, 100))}, "width"),
        ({"names": ["a"], "preds": np.full((1, 4096), 1.5)}, "bounds"),
        ({"names": ["a"], "preds": np.full((1, 4096), float("nan"))}, "nan"),
    ]:
        try:
            ex.validate_pred_bundle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid bundle accepted ({why})")
    print("PASS pred bundle validation rejects length/width/bounds mismatches")


def test_bits_fails_on_vendored_disagreement():
    # Finding 10: bits must fail export on vendored disagreement instead of
    # writing the output anyway.
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    export = tmp / "export.json"
    export.write_text(json.dumps({"molecules": [
        {"key": "K1", "smiles": "CCO", "identity_group": 1}]}))
    orig = ex.vendored_bits
    ex.vendored_bits = lambda smi, d: [999]  # force disagreement
    try:
        try:
            ex.cmd_bits(export, tmp / "bits.json", Path("/nonexistent"))
        except ValueError:
            pass
        else:
            raise AssertionError("disagreement did not fail the export")
        assert not (tmp / "bits.json").exists(), "output written despite disagreement"
    finally:
        ex.vendored_bits = orig
    print("PASS bits fails (and writes nothing) on vendored disagreement")


def write_ambiguous_inputs(root: Path) -> dict:
    # One compound with two spectra whose panel_spectra structures differ
    # (ethanol vs benzene: distinct fingerprints, review finding 4 pattern).
    rng = np.random.default_rng(11)
    preds = np.zeros((2, 4096))
    preds[0, [10, 200]] = [0.9, 0.8]
    preds[1, [500]] = 0.95
    preds += rng.random((2, 4096)) * 1e-4
    with open(root / "preds.p", "wb") as fh:
        pickle.dump({"names": np.array(["s1", "s2"]), "preds": preds,
                     "targs": [None] * 2}, fh)
    (root / "labels.tsv").write_text(
        "spec\tformula\tionization\tdataset\tcompound\tparentmass\tinstrument\n"
        "s1\tC2H6O\t[M+H]+\t-panel_\tAAAAAAAAAAAAAAAA\t47.0497\tunknown\n"
        "s2\tC2H6O\t[M+H]+\t-panel_\tAAAAAAAAAAAAAAAA\t47.0497\tunknown\n")
    table = pa.table({
        "inchikey14": ["AAAAAAAAAAAAAAAA"],
        "smiles": ["CCO"],
        "formula": ["C2H6O"],
        "n_heavy": [3],
        "identity_group": [7],
        "fold_identity": [2],
    })
    pq.write_table(table, root / "structures.parquet")
    spec_table = pa.table({
        "spec_id": ["s1", "s2"],
        "molecule": ["AAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAA"],
        "smiles": ["CCO", "c1ccccc1"],
        "formula": ["C2H6O", "C6H6"],
    })
    pq.write_table(spec_table, root / "panel_spectra.parquet")
    return {"preds": root / "preds.p", "labels": root / "labels.tsv",
            "structures": root / "structures.parquet",
            "panel_spectra": root / "panel_spectra.parquet"}


def test_panel_ambiguous_structure_dropped():
    # Finding 4: a panel molecule whose spectra map to distinct fingerprints
    # is dropped and counted, never silently first-wins.
    tmp = Path(tempfile.mkdtemp())
    src = write_ambiguous_inputs(tmp)
    report = ex.cmd_panel(src["preds"], src["labels"], src["structures"],
                          tmp / "panel.json",
                          panel_spectra_path=src["panel_spectra"])
    doc = json.loads((tmp / "panel.json").read_text())
    assert report["written"] == 0, report
    assert report["ambiguous"] == 1, report
    assert doc["molecules"] == []
    assert doc["skipped_panel_molecules"].get("ambiguous_structure") == 1
    print("PASS panel drops (and counts) ambiguous-structure molecules")


def test_panel_per_spectrum_entries():
    # Finding 5: --per-spectrum writes one entry per spectrum with that
    # spectrum's own probabilities and structure (same identity group).
    tmp = Path(tempfile.mkdtemp())
    src = write_ambiguous_inputs(tmp)
    report = ex.cmd_panel(src["preds"], src["labels"], src["structures"],
                          tmp / "panel_ps.json",
                          panel_spectra_path=src["panel_spectra"],
                          per_spectrum=True)
    doc = json.loads((tmp / "panel_ps.json").read_text())
    assert report["written"] == 2, report
    assert doc["per_spectrum"] is True
    by_spec = {m["spec_id"]: m for m in doc["molecules"]}
    assert set(by_spec) == {"s1", "s2"}
    assert by_spec["s1"]["smiles"] == "CCO"
    assert by_spec["s2"]["smiles"] == "c1ccccc1"
    assert by_spec["s1"]["fp_true"] == ex.fingerprint_bits("CCO")
    assert by_spec["s2"]["fp_true"] == ex.fingerprint_bits("c1ccccc1")
    assert by_spec["s1"]["identity_group"] == by_spec["s2"]["identity_group"] == 7
    assert by_spec["s1"]["spectra"] == 1 and by_spec["s2"]["spectra"] == 1
    print("PASS panel --per-spectrum writes one entry per spectrum")


def test_noise_uses_panel_spectra_truth():
    # Finding 4 (noise side): per-spectrum truth comes from the panel bundle,
    # so the two spectra above score against different truths.
    tmp = Path(tempfile.mkdtemp())
    src = write_ambiguous_inputs(tmp)
    payload = ex.cmd_noise(src["preds"], src["labels"], src["structures"],
                           tmp / "noise.json",
                           panel_spectra_path=src["panel_spectra"])
    assert payload["n_spectra"] == 2 and payload["n_molecules"] == 1
    assert "hist_pred_given_true_on_molecule" in payload
    print("PASS noise uses per-spectrum panel-bundle truth")

def main() -> None:
    test_fingerprint_matches_vendored_definition()
    test_bits_deterministic_and_sorted()
    test_build_bits_dedup_first_wins()
    test_noise_synthetic_hand_computed()
    test_noise_wiring()
    test_panel_schema_and_exclusion_list()
    test_noise_empty_truth_no_crash()
    test_noise_molecule_histograms()
    test_pred_bundle_validation()
    test_bits_fails_on_vendored_disagreement()
    test_panel_ambiguous_structure_dropped()
    test_panel_per_spectrum_entries()
    test_noise_uses_panel_spectra_truth()
    test_bits_keys_by_molecule_binding()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
