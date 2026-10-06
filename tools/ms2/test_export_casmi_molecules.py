"""Tests for `export_casmi_molecules.py` (task MC8).

Runnable with:
    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/test_export_casmi_molecules.py

A tiny in-memory structures table is written to a temporary parquet and the
tool is called as an imported module (no subprocess). Plain asserts only,
plus a `main()` so the file runs with and without pytest.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

import pyarrow as pa
import pyarrow.parquet as pq

import ms2_reference as ref
import export_casmi as exo
import export_casmi_molecules as ex

SEED = 20261005

# (smiles, inchikey14, identity_group, fold_identity)
TRAIN = [
    ("CCO", "TRAINKEY000001", 10, 2),
    ("c1ccccc1", "TRAINKEY000002", 11, 3),
    ("CC(=O)O", "TRAINKEY000003", 12, 4),
    ("CN", "TRAINKEY000004", 13, 2),
    ("CCC", "TRAINKEY000005", 14, 3),
    ("CCN", "TRAINKEY000006", 15, 4),
    ("CCCCCCCCCCCC", "TRAINKEY000007", 16, 2),  # 12 heavy atoms (for --max-heavy)
]
VALIDATION = [
    ("C1CCCCC1", "VALIDKEY000001", 99, 1),   # 99 % 3 == 0
    ("CCOCC", "VALIDKEY000002", 102, 1),     # 102 % 3 == 0
]
NEVER = [
    ("CCF", "FOLD0KEY000001", 20, 0),        # fold 0 is never exported
    ("CCBr", "FOLD1KEY000001", 22, 1),       # fold 1 with group % 3 != 0 is never exported
]
OUT_OF_DOMAIN = [
    ("C[N+](C)(C)C", "CHARGEDKEY00001", 30, 2),  # formal_charge
    ("C[Si](C)(C)C", "SILICONKEY00001", 31, 3),  # element_outside_domain
    ("C(C", "UNPARSEDKEY0001", 32, 4),            # unparseable SMILES
]


def write_structures(path: Path) -> None:
    rows = TRAIN + VALIDATION + NEVER + OUT_OF_DOMAIN
    table = pa.table({
        "inchikey14": [r[1] for r in rows],
        "smiles": [r[0] for r in rows],
        "identity_group": [r[2] for r in rows],
        "fold_identity": [r[3] for r in rows],
    })
    (path / "folds").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path / "folds" / "structures.parquet")


def load_doc(out_dir: Path, name: str, subset: str) -> dict:
    return json.loads((out_dir / f"{name}_{subset}.json").read_text())


def smiles_of(doc: dict) -> list:
    return [m["smiles"] for m in doc["molecules"]]


def test_subset_rule_uses_export_casmi():
    assert ex.subset_of is exo.subset_of, "the subset rule must be imported, not copied"
    print("PASS subset rule is imported from export_casmi")


def test_subsets_and_skips():
    tmp = Path(tempfile.mkdtemp())
    write_structures(tmp / "data")
    out = tmp / "out1"
    ex.export(tmp / "data", out, "t", 32, SEED)
    train, val = load_doc(out, "t", "train"), load_doc(out, "t", "validation")
    assert train["subset"] == "train" and val["subset"] == "validation"
    # Expected top-level fields of the existing schema, plus the two extras.
    for doc in (train, val):
        assert doc["schema_version"] == 1
        assert doc["chemistry"] == "ms2-chem-v0.1"
        assert doc["source"] == "CASMI 2026 structures.parquet (molecules only)"
        assert doc["seed"] == SEED and doc["n_raw"] == 512
        assert doc["spectra_per_molecule"] == 0
        assert doc["spectrum_sampling"] == "none" and doc["skipped_spectra"] == {}
        assert doc["max_heavy"] == 32 and isinstance(doc["skipped_structures"], dict)
        for m in doc["molecules"]:
            assert m["spectra"] == [] and isinstance(m["key"], str)
    got_train, got_val = set(smiles_of(train)), set(smiles_of(val))
    assert got_train == {r[0] for r in TRAIN}, got_train
    assert got_val == {r[0] for r in VALIDATION}, got_val
    for smi, _, _, _ in NEVER + OUT_OF_DOMAIN:
        assert smi not in got_train and smi not in got_val, smi
    # Skip reasons land in skipped_structures under the first classify reason.
    assert train["skipped_structures"].get("formal_charge", 0) == 1
    assert train["skipped_structures"].get("element_outside_domain", 0) == 1
    assert train["skipped_structures"].get("unparsed", 0) == 1
    assert val["skipped_structures"] == {}, val["skipped_structures"]
    # The written subsets share no identity_group.
    groups_train = {m["identity_group"] for m in train["molecules"]}
    groups_val = {m["identity_group"] for m in val["molecules"]}
    assert not groups_train & groups_val
    print("PASS subsets follow subset_of; out-of-domain rows skipped with reasons")


def test_max_heavy():
    tmp = Path(tempfile.mkdtemp())
    write_structures(tmp / "data")
    out = tmp / "out2"
    ex.export(tmp / "data", out, "t", 6, SEED)
    train = load_doc(out, "t", "train")
    got = set(smiles_of(train))
    assert "CCCCCCCCCCCC" not in got
    assert train["skipped_structures"].get("too_many_atoms", 0) == 1
    assert got == {r[0] for r in TRAIN} - {"CCCCCCCCCCCC"}, got
    print("PASS --max-heavy skips large molecules as too_many_atoms")


def test_determinism():
    tmp = Path(tempfile.mkdtemp())
    write_structures(tmp / "data")
    ex.export(tmp / "data", tmp / "a", "t", 32, SEED)
    ex.export(tmp / "data", tmp / "b", "t", 32, SEED)
    for subset in ("train", "validation"):
        assert (tmp / "a" / f"t_{subset}.json").read_bytes() == \
               (tmp / "b" / f"t_{subset}.json").read_bytes(), subset
    ex.export(tmp / "data", tmp / "c", "t", 32, SEED + 1)
    a = load_doc(tmp / "a", "t", "train")
    c = load_doc(tmp / "c", "t", "train")
    assert set(smiles_of(c)) == set(smiles_of(a))
    assert smiles_of(c) != smiles_of(a), "a different seed must reorder the molecules"
    print("PASS same seed byte-identical; different seed reorders the same set")


def test_limit_prefix():
    tmp = Path(tempfile.mkdtemp())
    write_structures(tmp / "data")
    ex.export(tmp / "data", tmp / "full", "t", 32, SEED)
    ex.export(tmp / "data", tmp / "lim", "t", 32, SEED, limit_train=2)
    full = smiles_of(load_doc(tmp / "full", "t", "train"))
    lim = smiles_of(load_doc(tmp / "lim", "t", "train"))
    assert lim == full[:2], (lim, full)
    print("PASS --limit-train keeps a prefix of the unlimited order")


def test_graph_consistency():
    tmp = Path(tempfile.mkdtemp())
    write_structures(tmp / "data")
    out = tmp / "out3"
    ex.export(tmp / "data", out, "t", 32, SEED)
    for subset in ("train", "validation"):
        for m in load_doc(out, "t", subset)["molecules"]:
            mol = ref.kekulized(m["smiles"])
            assert len(m["atoms"]) == mol.GetNumAtoms(), m["smiles"]
            bond_sum: dict = {}
            for a, b, order in m["bonds"]:
                bond_sum[a] = bond_sum.get(a, 0) + order
                bond_sum[b] = bond_sum.get(b, 0) + order
            for i, t in enumerate(m["atoms"]):
                _, h, valence = ref.ATOM_TYPES[t]
                assert bond_sum.get(i, 0) + h == valence, (m["smiles"], i, t)
    print("PASS atoms match the heavy-atom count; bond orders + H match the valence")


def main() -> None:
    test_subset_rule_uses_export_casmi()
    test_subsets_and_skips()
    test_max_heavy()
    test_determinism()
    test_limit_prefix()
    test_graph_consistency()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
