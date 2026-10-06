"""Tests for the stereo RDKit cross-check (plain functions + main)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from completion_stereo_check import (
    NAMED_FIXTURES,
    LostAssignmentError,
    build_rust_binary,
    canonical,
    check_isomer_smiles_distinct,
    check_named_fixtures,
    parity_of,
    rdkit_unique_smiles,
    rust_reports,
    stereo_block_of,
    to_rdkit,
    typed_of_smiles,
)
from rdkit import Chem


def test_named_fixtures_present():
    assert len(NAMED_FIXTURES) == 20
    for name in [
        "ethanol",
        "2-butanol",
        "2,3-butanediol",
        "1,4-dimethylcyclohexane",
        "pseudo-asymmetric",
        "2-butene",
        "hexa-2,4-diene",
        "acetaldoxime",
        "hydrazone",
        "cyclooctene",
    ]:
        assert name in NAMED_FIXTURES


def test_named_fixtures_agree_three_ways():
    binary = build_rust_binary()
    assert check_named_fixtures(binary) == []


def test_rust_report_shape():
    binary = build_rust_binary()
    atoms, bonds = typed_of_smiles("CCO")
    report = rust_reports([{"atoms": atoms, "bonds": bonds}], binary)[0]
    assert set(report.keys()) == {
        "potential",
        "stereogenic",
        "not_stereogenic",
        "unsupported",
        "raw_assignments",
        "distinct_stereoisomers",
        "resolution",
        "automorphisms",
        "isomers",
        "isomers_truncated",
    }
    assert report["resolution"] == "resolved"
    assert report["distinct_stereoisomers"] == 1


def test_wrong_parity_control_changes_smiles():
    # 2-butanol isomer 0 with its ligand order swapped (same value) must give
    # the mirror image: a different SMILES, equal to the flipped value.
    binary = build_rust_binary()
    atoms, bonds = typed_of_smiles("CCC(O)C")
    report = rust_reports([{"atoms": atoms, "bonds": bonds}], binary)[0]
    assert report["distinct_stereoisomers"] == 2
    block = stereo_block_of(report)
    good = canonical(Chem.MolToSmiles(to_rdkit(atoms, bonds, block, 0), isomericSmiles=True))
    other = canonical(Chem.MolToSmiles(to_rdkit(atoms, bonds, block, 1), isomericSmiles=True))
    assert good != other, "enantiomers differ"
    ligands = list(block["tetrahedral_centers"][0]["ligands"])
    ligands[0], ligands[1] = ligands[1], ligands[0]
    swapped = {
        "tetrahedral_centers": [
            {
                "atom": block["tetrahedral_centers"][0]["atom"],
                "ligands": ligands,
            }
        ],
        "double_bonds": [],
        "isomers": [{"tetrahedral": [0], "double_bonds": []}],
    }
    mirrored = canonical(
        Chem.MolToSmiles(to_rdkit(atoms, bonds, swapped, 0), isomericSmiles=True)
    )
    assert mirrored != good, "swapped ligand order changes the SMILES"
    assert mirrored == other, "swapping two ligands mirrors (odd permutation)"
    assert parity_of([0, 1], [1, 0]) == 1


def test_lost_assignment_error_path():
    # A lone-pair reference with no ligand at all cannot be represented:
    # no dummy atom is invented, an error is raised instead.
    atoms = [5, 5]
    bonds = [[0, 1, 2]]
    block = {
        "tetrahedral_centers": [],
        "double_bonds": [{"atoms": [0, 1], "reference": ["lone_pair", "lone_pair"]}],
        "isomers": [{"tetrahedral": [], "double_bonds": [0]}],
    }
    try:
        to_rdkit(atoms, bonds, block, 0)
    except LostAssignmentError:
        pass
    else:
        raise AssertionError("lone pair without ligand must raise")


def test_assignment_length_and_value_mismatch_raise():
    # F5: no silent `zip` truncation — a short, long or out-of-range
    # assignment is a LostAssignmentError naming the element.
    atoms, bonds = typed_of_smiles("CCC(O)C")
    binary = build_rust_binary()
    report = rust_reports([{"atoms": atoms, "bonds": bonds}], binary)[0]
    block = stereo_block_of(report)
    assert len(block["tetrahedral_centers"]) == 1
    short = {
        "tetrahedral_centers": block["tetrahedral_centers"],
        "double_bonds": [],
        "isomers": [{"tetrahedral": [], "double_bonds": []}],
    }
    for bad in (
        short,
        {
            "tetrahedral_centers": block["tetrahedral_centers"],
            "double_bonds": [],
            "isomers": [{"tetrahedral": [0, 1], "double_bonds": []}],
        },
        {
            "tetrahedral_centers": block["tetrahedral_centers"],
            "double_bonds": [],
            "isomers": [{"tetrahedral": [2], "double_bonds": []}],
        },
    ):
        try:
            to_rdkit(atoms, bonds, bad, 0)
        except LostAssignmentError:
            pass
        else:
            raise AssertionError(f"malformed assignment must raise: {bad['isomers']}")


def test_forged_kekule_bond_is_refused():
    # F5: a designated stereo bond that RDKit aromatises on sanitisation
    # (kekulized benzene) is refused before assigning, naming the bond.
    atoms, bonds = typed_of_smiles("c1ccccc1")
    block = {
        "tetrahedral_centers": [],
        "double_bonds": [{"atoms": [0, 1], "reference": [5, 2]}],
        "isomers": [{"tetrahedral": [], "double_bonds": [0]}],
    }
    try:
        to_rdkit(atoms, bonds, block, 0)
    except LostAssignmentError as e:
        assert "double_bonds[0]" in str(e), f"names the bond: {e}"
    else:
        raise AssertionError("aromatised bond must raise")


def test_heteroaromatic_macrocycle_loses_assignments():
    # F5 (review counterexample): forcing E/Z onto the heteroaromatic
    # macrocycle's conjugated ring bonds does not survive RDKit — every
    # failure raises LostAssignmentError naming the element.
    atoms, bonds = typed_of_smiles("C1=CC=CNC=CC=CNC=CC=CN1")
    block = {
        "tetrahedral_centers": [],
        "double_bonds": [{"atoms": [0, 1], "reference": [14, 2]}],
        "isomers": [{"tetrahedral": [], "double_bonds": [0]}],
    }
    try:
        to_rdkit(atoms, bonds, block, 0)
    except LostAssignmentError as e:
        assert "double_bonds[0]" in str(e), f"names the bond: {e}"
    else:
        raise AssertionError("macrocycle assignment must raise")


def test_wrong_ligands_are_refused():
    # F5: a centre whose documented ligands are not its neighbours is
    # refused (a LostAssignmentError, not an AssertionError).
    atoms, bonds = typed_of_smiles("CCC(O)C")
    block = {
        "tetrahedral_centers": [{"atom": 1, "ligands": [0, 2, 3, "H"]}],
        "double_bonds": [],
        "isomers": [{"tetrahedral": [0], "double_bonds": []}],
    }
    try:
        to_rdkit(atoms, bonds, block, 0)
    except LostAssignmentError as e:
        assert "tetrahedral_centers[0]" in str(e), f"names the centre: {e}"
    else:
        raise AssertionError("ligand mismatch must raise")


def test_collapsed_isomers_raise():
    # F5 (block level): two reported-distinct isomers serialising to one
    # SMILES raise LostAssignmentError. The pseudo-asymmetric molecule's
    # [0, 0, 1] and its same-orbit middle-flip [0, 1, 1] coincide in SMILES
    # (conditional stereogenicity — the flip stays in the orbit), so a
    # block claiming both as distinct isomers collapses.
    binary = build_rust_binary()
    atoms, bonds = typed_of_smiles("CC(Cl)C(F)C(Cl)C")
    report = rust_reports([{"atoms": atoms, "bonds": bonds}], binary)[0]
    block = stereo_block_of(report)
    assert check_isomer_smiles_distinct(atoms, bonds, block) is not None
    forged = {
        "tetrahedral_centers": block["tetrahedral_centers"],
        "double_bonds": [],
        "isomers": [
            {"tetrahedral": [0, 0, 1], "double_bonds": []},
            {"tetrahedral": [0, 1, 1], "double_bonds": []},
        ],
    }
    try:
        check_isomer_smiles_distinct(atoms, bonds, forged)
    except LostAssignmentError as e:
        assert "same SMILES" in str(e), f"names the collapse: {e}"
    else:
        raise AssertionError("collapsed isomers must raise")


def test_permutation_robustness():
    # 2,3-butanediol with atoms renumbered (reversed) gives the same set of
    # isomeric SMILES.
    binary = build_rust_binary()
    atoms, bonds = typed_of_smiles("CC(O)C(O)C")
    n = len(atoms)
    perm = list(reversed(range(n)))
    renumbered_atoms = [atoms[perm[i]] for i in range(n)]
    position = [0] * n
    for i, p in enumerate(perm):
        position[p] = i
    renumbered_bonds = sorted(
        [position[a], position[b], o] if position[a] < position[b] else [position[b], position[a], o]
        for a, b, o in bonds
    )
    report = rust_reports([{"atoms": atoms, "bonds": bonds}], binary)[0]
    re_report = rust_reports(
        [{"atoms": renumbered_atoms, "bonds": renumbered_bonds}], binary
    )[0]
    assert report["distinct_stereoisomers"] == re_report["distinct_stereoisomers"] == 3
    block = stereo_block_of(report)
    re_block = stereo_block_of(re_report)
    before = {
        canonical(Chem.MolToSmiles(to_rdkit(atoms, bonds, block, i), isomericSmiles=True))
        for i in range(len(block["isomers"]))
    }
    after = {
        canonical(
            Chem.MolToSmiles(
                to_rdkit(renumbered_atoms, renumbered_bonds, re_block, i),
                isomericSmiles=True,
            )
        )
        for i in range(len(re_block["isomers"]))
    }
    assert before == after
    assert len(before) == 3
    # And RDKit's own enumeration agrees on the renumbered graph too.
    bare = to_rdkit(renumbered_atoms, renumbered_bonds, re_block, None)
    assert rdkit_unique_smiles(bare) == before


def main() -> int:
    test_named_fixtures_present()
    print("test_named_fixtures_present ok")
    test_rust_report_shape()
    print("test_rust_report_shape ok")
    test_wrong_parity_control_changes_smiles()
    print("test_wrong_parity_control_changes_smiles ok")
    test_lost_assignment_error_path()
    print("test_lost_assignment_error_path ok")
    test_assignment_length_and_value_mismatch_raise()
    print("test_assignment_length_and_value_mismatch_raise ok")
    test_forged_kekule_bond_is_refused()
    print("test_forged_kekule_bond_is_refused ok")
    test_heteroaromatic_macrocycle_loses_assignments()
    print("test_heteroaromatic_macrocycle_loses_assignments ok")
    test_wrong_ligands_are_refused()
    print("test_wrong_ligands_are_refused ok")
    test_collapsed_isomers_raise()
    print("test_collapsed_isomers_raise ok")
    test_permutation_robustness()
    print("test_permutation_robustness ok")
    test_named_fixtures_agree_three_ways()
    print("test_named_fixtures_agree_three_ways ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
