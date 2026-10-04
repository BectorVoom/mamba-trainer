"""The V0 formula table (docs/MS2_CONTRACTS.md section 9) and its coverage report.

Rows are the distinct molecular formulas of the in-domain structures of the train
subset (`fold_identity` in {2, 3, 4}), sorted by integer mass, then by element
counts. The report gives the row count, bytes, how many rows a 20 ppm window
holds, and how many validation-fold structures have their formula in the table.
The test fold is not read.

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/formula_table.py \
        --data /Users/ods/Documents/Enveda_CASMI/kobayashi/exp-casmi-26-from-spectra-to-structures/data \
        --out bench/results/ms2/formula_table.json [--table-out <path>.json]

`--table-out` writes the table itself (derived from the CASMI structures, so it
stays outside the repository).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")
TRAIN_FOLDS, VALIDATION_FOLD = (2, 3, 4), 1
ROW_BYTES = 2 * len(ref.ELEMENT_ORDER) + 4
WINDOW_PPM_TENTHS = 200


def build_table(row_keys: set[tuple]) -> list:
    """Sorted (integer mass, element-count key) rows of distinct formula keys."""
    return sorted((sum(n * ref.MASS[e] for n, e in zip(key, ref.ELEMENT_ORDER)), key)
                  for key in row_keys)


def table_payload(table) -> str:
    """The `--table-out` JSON: elements and rows."""
    return json.dumps({"elements": ref.ELEMENT_ORDER, "rows": [[m, list(k)] for m, k in table]},
                      separators=(",", ":"))


def build_report(table, row_keys: set[tuple], train_structures: int,
                 validation: list, source: str, payload: str) -> dict:
    """The `--out` JSON: row count, bytes, window occupancy, validation coverage."""
    masses = np.array([m for m, _ in table], dtype=np.int64)
    tol = masses * WINDOW_PPM_TENTHS // 10_000_000
    joined = np.searchsorted(masses, masses + tol, "right") - np.searchsorted(masses, masses - tol, "left")
    return {
        "schema_version": 1, "version": "ms2-formula-v0", "chemistry": "ms2-chem-v0.1",
        "source": source,
        "train_structures": train_structures, "rows": len(table), "row_bytes": ROW_BYTES,
        "bytes": len(table) * ROW_BYTES, "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "mass_range_udalton": [int(masses.min()), int(masses.max())],
        "max_count_per_element": {e: int(max(k[i] for _, k in table)) for i, e in enumerate(ref.ELEMENT_ORDER)},
        "window_ppm_tenths": WINDOW_PPM_TENTHS,
        "rows_in_window_around_a_row": {f"p{p}": float(np.percentile(joined, p)) for p in (50, 95, 99, 100)},
        "validation_fold_structures": len(validation),
        "validation_fold_formula_in_table": float(np.mean([k in row_keys for k in validation])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table-out", type=Path, default=None)
    args = ap.parse_args()

    t = pq.read_table(args.data / "folds" / "structures.parquet",
                      columns=["smiles", "fold_identity"]).to_pandas()
    t = t[t.fold_identity.isin(TRAIN_FOLDS + (VALIDATION_FOLD,))]
    rows, train_structures = set(), 0
    validation = []
    for smi, fold in zip(t.smiles.values, t.fold_identity.values):
        mol = Chem.MolFromSmiles(smi)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        if ref.classify(mol):
            continue
        atoms, _ = ref.graph_of(mol)
        counts = ref.composition(atoms)
        key = tuple(counts[e] for e in ref.ELEMENT_ORDER)
        if fold in TRAIN_FOLDS:
            rows.add(key)
            train_structures += 1
        else:
            validation.append(key)
    table = build_table(rows)
    payload = table_payload(table)
    report = build_report(table, rows, train_structures, validation,
                          "in-domain structures of fold_identity 2, 3, 4", payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.table_out:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        args.table_out.write_text(payload)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
