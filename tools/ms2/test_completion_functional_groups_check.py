"""Tests for the functional-group RDKit cross-check (plain functions + main)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from completion_functional_groups_check import (
    NAMED_FIXTURES,
    build_rust_binary,
    check_named_fixtures,
    reference_of_smiles,
    rust_results,
    typed_of_smiles,
)


def test_named_fixtures_present():
    assert len(NAMED_FIXTURES) >= 34
    for name in [
        "ethanol",
        "benzene",
        "phenol",
        "pyridine",
        "oxirane",
        "1,1-dimethoxyethane",
        "acrolein",
        "urea",
        "2-pyridone",
        "indole",
        "caffeine-like purine",
    ]:
        assert name in NAMED_FIXTURES


def test_typed_graphs_follow_rdkit_order():
    for name, smiles in list(NAMED_FIXTURES.items())[:5]:
        atoms, bonds = typed_of_smiles(smiles)
        assert len(atoms) > 0, name
        assert isinstance(bonds, list), name


def test_reference_groups_are_sorted():
    for name, smiles in list(NAMED_FIXTURES.items())[:5]:
        _, groups = reference_of_smiles(smiles)
        assert groups == sorted([sorted(g) for g in groups]), name


def test_rust_matches_named_fixtures():
    binary = build_rust_binary()
    failures = check_named_fixtures(binary)
    assert failures == [], f"named fixtures disagree: {failures}"


def test_rust_result_shape():
    binary = build_rust_binary()
    atoms, bonds = typed_of_smiles("CCO")
    out = rust_results([{"atoms": atoms, "bonds": bonds}], binary)[0]
    assert set(out.keys()) == {"aromatic_atoms", "groups"}
    assert len(out["aromatic_atoms"]) == len(atoms)


def main() -> int:
    test_named_fixtures_present()
    print("test_named_fixtures_present ok")
    test_typed_graphs_follow_rdkit_order()
    print("test_typed_graphs_follow_rdkit_order ok")
    test_reference_groups_are_sorted()
    print("test_reference_groups_are_sorted ok")
    test_rust_result_shape()
    print("test_rust_result_shape ok")
    test_rust_matches_named_fixtures()
    print("test_rust_matches_named_fixtures ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
