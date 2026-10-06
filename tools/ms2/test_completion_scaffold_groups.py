"""Tests for `tools/ms2/completion_scaffold_groups.py` (MC4 scaffold subgroups).

Runnable with:
    uv run --project /Users/ods/Documents/Enveda_CASMI python -m pytest -q \
        tools/ms2/test_completion_scaffold_groups.py

A 5-molecule in-memory validation case against a 2-molecule train set:
toluene sees benzene's scaffold, pyridine and cyclohexane are novel,
propane is acyclic, and one SMILES does not parse.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import completion_scaffold_groups as groups


def molecule(key, smiles):
    return {"key": key, "smiles": smiles}


def train_molecules():
    return [
        molecule("TRAIN_BENZENE", "c1ccccc1"),
        molecule("TRAIN_HEXANE", "CCCCCC"),
    ]


def validation_molecules():
    return [
        molecule("VAL_TOLUENE", "Cc1ccccc1"),
        molecule("VAL_PYRIDINE", "c1ccncc1"),
        molecule("VAL_CYCLOHEXANE", "C1CCCCC1"),
        molecule("VAL_PROPANE", "CCC"),
        molecule("VAL_BROKEN", "not-a-smiles(((("),
    ]


def test_scaffold_convention_matches_export_msgym():
    """The scaffold helper agrees with export_msgym's murcko function."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import export_msgym as ex  # noqa: E402
    from rdkit import Chem  # noqa: E402

    for smiles in ("c1ccccc1", "Cc1ccccc1", "CCC", "C1CCCCC1"):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        assert groups.murcko_scaffold_smiles(mol) == ex.murcko_scaffold_smiles(mol), smiles
    assert groups.scaffold_of("not-a-smiles((((") is None
    assert groups.scaffold_of("CCC") == ""


def test_five_molecule_grouping():
    """Toluene is seen, pyridine/cyclohexane novel, propane acyclic."""
    document = groups.group_keys(train_molecules(), validation_molecules())
    assert document["scaffold_seen"] == ["VAL_TOLUENE"]
    assert sorted(document["scaffold_novel"]) == ["VAL_CYCLOHEXANE", "VAL_PYRIDINE"]
    assert document["acyclic"] == ["VAL_PROPANE"]
    summary = document["_summary"]
    assert summary["scaffold_novel"] == 2
    assert summary["scaffold_seen"] == 1
    assert summary["acyclic"] == 1
    assert summary["unparsed"] == 1
    assert isinstance(summary["rdkit_version"], str) and summary["rdkit_version"]
    assert set(document) == {"scaffold_novel", "scaffold_seen", "acyclic", "_summary"}


def test_cli_writes_groups_json(tmp_path):
    """The CLI writes the same document to --out."""
    _check_cli(tmp_path)


def _check_cli(tmp: Path):
    train_path = tmp / "train.json"
    validation_path = tmp / "validation.json"
    out_path = tmp / "groups.json"
    train_path.write_text(json.dumps({"molecules": train_molecules()}))
    validation_path.write_text(json.dumps({"molecules": validation_molecules()}))
    code = groups.main(
        ["--train", str(train_path), "--validation", str(validation_path),
         "--out", str(out_path)]
    )
    assert code == 0
    written = json.loads(out_path.read_text())
    assert written == groups.group_keys(train_molecules(), validation_molecules())


def main() -> None:
    """Direct runner (no pytest needed): plain asserts in a temp dir."""
    import tempfile

    test_scaffold_convention_matches_export_msgym()
    test_five_molecule_grouping()
    with tempfile.TemporaryDirectory() as tmp:
        _check_cli(Path(tmp))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
