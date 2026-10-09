"""Murcko scaffolds as pattern sidecars for the completion model.

Writes, for every molecule of an `export_msgym_spectral.py` export, the atom
indices of its Bemis-Murcko scaffold **within that molecule's own atom
numbering**, so the Rust side can build the scaffold as an induced subgraph
carrying the parent's atom types (hydrogen counts included). That is the
`v0_parent_hydrogen_counts` semantics the completion model's patterns
require: a scaffold atom keeps the hydrogen count it has in the parent, not
the count it would have as an isolated molecule.

The scaffold is RDKit's `MurckoScaffold.GetScaffoldForMol` expressed as an
atom set: the ring atoms plus the atoms on the shortest paths between rings
(the linkers). It is computed by matching the scaffold's own atoms back to
the parent through a substructure match, so the indices are the parent's.
Acyclic molecules have an empty scaffold and are written as an empty list.

Output `<name>_scaffold.json`:

    {"scaffold": "bemis-murcko (RDKit MurckoScaffold.GetScaffoldForMol)",
     "rdkit": "...", "export": "...", "n_molecules": N,
     "keys_by_molecule": ["<key>|<identity_group>", ...],
     "atoms_by_molecule": [[3, 4, 5, ...], ...],
     "summary": {"acyclic": n, "unmatched": n, "mean_scaffold_atoms": x}}

`atoms_by_molecule[i]` is aligned with `export["molecules"][i]`, like the
fingerprint sidecar, and `keys_by_molecule` lets the loader check that
alignment.

    PYTHONPATH=tools/ms2 python tools/ms2/export_scaffold_patterns.py \
        --export data/ms2/specgen/msgym_validation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")


def scaffold_atoms(smiles: str) -> list[int] | None:
    """Parent atom indices of the Murcko scaffold, or `None` when unmatched.

    The parent is kekulized exactly as the export's graph was
    (`ms2_reference.kekulized`), so the indices line up with the export's
    `atoms` list. An acyclic molecule gives `[]`.
    """
    parent = ref.kekulized(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(parent)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return []
    # Match the scaffold back onto the parent. The scaffold is a subgraph of
    # the parent by construction, so a match must exist; a failure is
    # reported rather than guessed around.
    match = parent.GetSubstructMatch(scaffold)
    if not match:
        return None
    return sorted(int(i) for i in match)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    export = json.loads(args.export.read_text())
    molecules = export["molecules"]
    atoms_by_molecule: list[list[int]] = []
    keys: list[str] = []
    acyclic = unmatched = 0
    total = 0
    for molecule in molecules:
        keys.append(f"{molecule['key']}|{molecule['identity_group']}")
        found = scaffold_atoms(molecule["smiles"])
        if found is None:
            unmatched += 1
            atoms_by_molecule.append([])
            continue
        if not found:
            acyclic += 1
        # The indices must exist in the export's own atom list.
        if found and max(found) >= len(molecule["atoms"]):
            raise SystemExit(
                f"{molecule['key']}: scaffold atom {max(found)} is past the "
                f"molecule's {len(molecule['atoms'])} atoms"
            )
        total += len(found)
        atoms_by_molecule.append(found)

    out = args.out or args.export.with_name(args.export.stem + "_scaffold.json")
    document = {
        "scaffold": "bemis-murcko (RDKit MurckoScaffold.GetScaffoldForMol), matched back to the parent's atom numbering",
        "rdkit": rdkit.__version__,
        "export": args.export.name,
        "n_molecules": len(molecules),
        "keys_by_molecule": keys,
        "atoms_by_molecule": atoms_by_molecule,
        "summary": {
            "molecules": len(molecules),
            "acyclic": acyclic,
            "unmatched": unmatched,
            "mean_scaffold_atoms": round(total / max(len(molecules), 1), 2),
        },
    }
    out.write_text(json.dumps(document, separators=(",", ":")))
    print(json.dumps({**document["summary"], "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
