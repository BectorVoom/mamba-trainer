#!/usr/bin/env python3
"""Scaffold subgroups for the molecular-completion experiment (MC4).

For every validation molecule, the Bemis-Murcko scaffold SMILES of its
``smiles`` is compared against the scaffolds of every train molecule::

    uv run --project /Users/ods/Documents/Enveda_CASMI python \\
        tools/ms2/completion_scaffold_groups.py \\
        --train scale20k_train.json --validation scale20k_validation.json \\
        --out scaffold_groups.json

writes ``{"scaffold_novel": [...], "scaffold_seen": [...], "acyclic": [...],
"_summary": {...}}`` where the lists hold validation molecule ``key``
values:

* ``scaffold_novel``: keys whose scaffold is non-empty and absent from
  every train molecule;
* ``scaffold_seen``: keys whose non-empty scaffold occurs in train;
* ``acyclic``: keys with no scaffold (``''``);
* ``_summary``: the three counts, the ``unparsed`` count and the RDKit
  version. Keys starting with ``_`` are reserved for such metadata; the
  experiment driver ignores them.

The scaffold convention is the ``murcko_scaffold_smiles`` function of
``tools/ms2/export_msgym.py`` (``''`` for acyclic molecules). Unparseable
SMILES are counted under ``"_summary".unparsed`` and left out of every
group.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold_smiles(mol) -> str:
    """Bemis-Murcko scaffold SMILES; '' for acyclic molecules (no scaffold).

    Same convention as ``murcko_scaffold_smiles`` in
    ``tools/ms2/export_msgym.py``.
    """
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def scaffold_of(smiles: str) -> str | None:
    """The scaffold SMILES of ``smiles``, or None when it does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return murcko_scaffold_smiles(mol)


def group_keys(train_molecules, validation_molecules) -> dict:
    """Sort validation keys into scaffold groups.

    ``train_molecules`` and ``validation_molecules`` are sequences of
    mappings with ``key`` and ``smiles`` entries. Returns the output
    document: ``scaffold_novel``, ``scaffold_seen`` and ``acyclic`` key
    lists plus the ``_summary`` object. Unparseable validation SMILES are
    counted in ``_summary["unparsed"]`` and left out; unparseable train
    SMILES contribute no scaffold.
    """
    train_scaffolds = set()
    for molecule in train_molecules:
        scaffold = scaffold_of(molecule.get("smiles", ""))
        if scaffold:
            train_scaffolds.add(scaffold)
    novel, seen, acyclic, unparsed = [], [], [], 0
    for molecule in validation_molecules:
        scaffold = scaffold_of(molecule.get("smiles", ""))
        if scaffold is None:
            unparsed += 1
            continue
        if not scaffold:
            acyclic.append(molecule["key"])
        elif scaffold in train_scaffolds:
            seen.append(molecule["key"])
        else:
            novel.append(molecule["key"])
    return {
        "scaffold_novel": novel,
        "scaffold_seen": seen,
        "acyclic": acyclic,
        "_summary": {
            "scaffold_novel": len(novel),
            "scaffold_seen": len(seen),
            "acyclic": len(acyclic),
            "unparsed": unparsed,
            "rdkit_version": rdBase.rdkitVersion,
        },
    }


def load_molecules(path: Path):
    """The ``molecules`` array of an export file."""
    with open(path) as fh:
        document = json.load(fh)
    molecules = document.get("molecules")
    if not isinstance(molecules, list):
        raise ValueError(f"{path}: no molecules array")
    return molecules


def main(argv=None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="train export.json")
    parser.add_argument("--validation", required=True, help="validation export.json")
    parser.add_argument("--out", required=True, help="groups.json to write")
    args = parser.parse_args(argv)
    train_molecules = load_molecules(Path(args.train))
    validation_molecules = load_molecules(Path(args.validation))
    document = group_keys(train_molecules, validation_molecules)
    with open(args.out, "w") as fh:
        json.dump(document, fh, indent=2, sort_keys=True)
        fh.write("\n")
    summary = document["_summary"]
    print(
        f"novel {summary['scaffold_novel']}  seen {summary['scaffold_seen']}  "
        f"acyclic {summary['acyclic']}  unparsed {summary['unparsed']}  "
        f"rdkit {summary['rdkit_version']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
