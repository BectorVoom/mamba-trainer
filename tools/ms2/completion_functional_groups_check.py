"""Cross-check of `functional-groups-ertl-v1` against RDKit's IFG reference.

Rust result comes from the `functional_groups_report` example (subprocess,
built once). Reference is RDKit's `Contrib/IFG/ifg.py`
(`identify_functional_groups`, field `atomIds`) plus RDKit's own aromaticity
on the molecule built from the export's SMILES. Atom indices align through
`ms2_reference.kekulized` + `graph_of` (atom order = RDKit atom order).

Exit non-zero only if a named fixture disagrees.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Contrib.IFG import ifg as rdkit_ifg

sys.path.insert(0, str(Path(__file__).parent))
from ms2_reference import graph_of, kekulized

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "target" / "release" / "examples" / "functional_groups_report"

NAMED_FIXTURES = {
    "ethanol": "CCO",
    "diethyl ether": "CCOCC",
    "acetaldehyde": "CC=O",
    "acetone": "CC(C)=O",
    "acetic acid": "CC(=O)O",
    "methyl acetate": "COC(C)=O",
    "acetamide": "CC(=O)N",
    "acetonitrile": "CC#N",
    "ethylamine": "CCN",
    "trimethylamine": "CN(C)C",
    "propene": "CC=C",
    "propyne": "CC#C",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "anisole": "COc1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1c[nH]cn1",
    "chlorobenzene": "Clc1ccccc1",
    "oxirane": "C1CO1",
    "1,1-dimethoxyethane": "CC(OC)OC",
    "acrolein": "C=CC=O",
    "urea": "NC(=O)N",
    "dimethyl sulfide": "CSC",
    "dimethyl disulfide": "CSSC",
    "benzaldehyde": "O=Cc1ccccc1",
    "styrene": "C=Cc1ccccc1",
    "cyclohexene": "C1CCC=CC1",
    "2-pyridone": "O=c1[nH]cccc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "caffeine-like purine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
}


def build_rust_binary() -> Path:
    proc = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--no-default-features",
            "--features",
            "cpu",
            "--example",
            "functional_groups_report",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("cargo build of functional_groups_report failed")
    return EXAMPLE


def rust_results(entries: list[dict], binary: Path) -> list[dict]:
    payload = "\n".join(json.dumps({"atoms": e["atoms"], "bonds": e["bonds"]}) for e in entries)
    proc = subprocess.run(
        [str(binary)], input=payload, capture_output=True, text=True, cwd=str(REPO)
    )
    if proc.returncode != 0:
        raise RuntimeError(f"functional_groups_report failed: {proc.stderr[:2000]}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != len(entries):
        raise RuntimeError(f"rust returned {len(lines)} lines for {len(entries)} molecules")
    return [json.loads(line) for line in lines]


def reference_of_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES fixture")
    aromatic = [a.GetIsAromatic() for a in mol.GetAtoms()]
    groups = sorted([sorted(g.atomIds) for g in rdkit_ifg.identify_functional_groups(mol)])
    return aromatic, groups


def typed_of_smiles(smiles: str):
    mol = kekulized(smiles)
    atoms, bonds = graph_of(mol)
    # Atom order = RDKit atom order: graph_of walks GetAtoms/GetBonds in order.
    assert len(atoms) == mol.GetNumAtoms(), "atom count follows RDKit order"
    return atoms, bonds


def has_fused_or_hetero_aromatic(mol) -> bool:
    rings = mol.GetRingInfo().AtomRings()
    aromatic_rings = [
        r for r in rings if r and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    ]
    for ring in aromatic_rings:
        for i in ring:
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 6:
                return True
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(set(rings[i]) & set(rings[j])) >= 2:
                return True
    return False


def check_named_fixtures(binary: Path) -> list[str]:
    failures = []
    entries = []
    names = list(NAMED_FIXTURES.keys())
    for name in names:
        atoms, bonds = typed_of_smiles(NAMED_FIXTURES[name])
        entries.append({"atoms": atoms, "bonds": bonds})
    rust = rust_results(entries, binary)
    for name, out in zip(names, rust):
        _, want_groups = reference_of_smiles(NAMED_FIXTURES[name])
        got_groups = sorted([sorted(g) for g in out["groups"]])
        if got_groups != want_groups:
            failures.append(f"{name}: want {want_groups} got {got_groups}")
    return failures


def check_export(export: Path, limit: int, binary: Path) -> int:
    data = json.loads(export.read_text())
    molecules = data["molecules"][:limit]
    entries = []
    references = []
    flags = []
    ring_counts = []
    for mol in molecules:
        smiles = mol["smiles"]
        # Verify index alignment through the exporter conversion.
        atoms, bonds = typed_of_smiles(smiles)
        assert atoms == mol["atoms"], "export atoms match kekulized graph_of"
        assert bonds == mol["bonds"], "export bonds match kekulized graph_of"
        entries.append({"atoms": mol["atoms"], "bonds": mol["bonds"]})
        ref_mol = Chem.MolFromSmiles(smiles)
        aromatic = [a.GetIsAromatic() for a in ref_mol.GetAtoms()]
        groups = sorted([sorted(g.atomIds) for g in rdkit_ifg.identify_functional_groups(ref_mol)])
        references.append((aromatic, groups))
        flags.append(has_fused_or_hetero_aromatic(ref_mol))
        # Precompute ring counts before the Rust subprocess below.
        ring_counts.append(ref_mol.GetRingInfo().NumRings())
    rust = rust_results(entries, binary)
    n = len(molecules)
    arom_ok = 0
    groups_ok = 0
    arom_ok_flag = [0, 0]
    groups_ok_flag = [0, 0]
    count_flag = [0, 0]
    groups_per: list[int] = []
    atoms_per: list[int] = []
    disagreements = []
    for idx, out in enumerate(rust):
        want_arom, want_groups = references[idx]
        got_arom = [bool(v) for v in out["aromatic_atoms"]]
        got_groups = sorted([sorted(g) for g in out["groups"]])
        arom_same = got_arom == want_arom
        groups_same = got_groups == want_groups
        arom_ok += int(arom_same)
        groups_ok += int(groups_same)
        flag = 1 if flags[idx] else 0
        count_flag[flag] += 1
        arom_ok_flag[flag] += int(arom_same)
        groups_ok_flag[flag] += int(groups_same)
        groups_per.append(len(got_groups))
        for grp in out["groups"]:
            atoms_per.append(len(grp))
        if not (arom_same and groups_same):
            heavy = len(entries[idx]["atoms"])
            ring_count = ring_counts[idx]
            if not arom_same:
                kind = "aromaticity"
            elif {a for g in got_groups for a in g} != {a for g in want_groups for a in g}:
                kind = "marking"
            else:
                kind = "merging"
            disagreements.append((heavy, ring_count, kind))
    def frac(ok: int, total: int) -> float:
        return ok / total if total else 0.0
    print(f"molecules: {n}")
    print(f"aromatic exact agreement: {arom_ok}/{n} {frac(arom_ok, n):.4f}")
    print(f"groups exact agreement: {groups_ok}/{n} {frac(groups_ok, n):.4f}")
    for flag, label in [(1, "fused-or-hetero-aromatic=yes"), (0, "fused-or-hetero-aromatic=no")]:
        total = count_flag[flag]
        print(
            f"  {label}: n={total} aromatic {frac(arom_ok_flag[flag], total):.4f} "
            f"groups {frac(groups_ok_flag[flag], total):.4f}"
        )
    if groups_per:
        print(
            f"groups per molecule: mean {sum(groups_per)/len(groups_per):.3f} "
            f"max {max(groups_per)}"
        )
    if atoms_per:
        print(f"atoms per group: mean {sum(atoms_per)/len(atoms_per):.3f} max {max(atoms_per)}")
    else:
        print("atoms per group: no groups")
    no_group = sum(1 for c in groups_per if c == 0)
    print(f"fraction with no functional group: {no_group/max(n,1):.4f}")
    print(f"disagreements: {len(disagreements)} (showing up to 20)")
    for heavy, rings, kind in disagreements[:20]:
        print(f"  heavy_atoms={heavy} rings={rings} kind={kind}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args(argv)
    binary = build_rust_binary()
    failures = check_named_fixtures(binary)
    for failure in failures:
        print(f"named fixture disagreement: {failure}", file=sys.stderr)
    if failures:
        print(f"named fixtures: {len(NAMED_FIXTURES)-len(failures)}/{len(NAMED_FIXTURES)} agree")
        return 1
    print(f"named fixtures: {len(NAMED_FIXTURES)}/{len(NAMED_FIXTURES)} agree")
    if args.export is not None:
        return check_export(args.export, args.limit, binary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
