"""Export in-domain CASMI molecules (no spectra) for the completion experiment.

Reads `folds/structures.parquet` only (never `raw/train.parquet`, never
`raw/test.parquet`) and writes one JSON file per exported subset of
`export_casmi.subset_of` (train, validation) in the schema
`tools/ms2/export_casmi.py` writes, with empty spectrum lists so the Rust
`ExportFile` reader loads them unchanged. The completion-conditioned model
trains on molecules only; this exports all in-domain training molecules
instead of the ~19k covered by the spectrum exports.

The output is derived from CC BY-NC competition data and stays outside the
repository (`--out-dir`, by default next to the data).

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/export_casmi_molecules.py \
        --data /Users/ods/Documents/Enveda_CASMI/kobayashi/exp-casmi-26-from-spectra-to-structures/data \
        --name molecules_v1 --max-heavy 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem, RDLogger

import ms2_reference as ref
from export_casmi import subset_of

RDLogger.DisableLog("rdApp.*")

# As hardcoded in export_casmi.py (== Rust CHEMISTRY_VERSION in
# src/models/ms2/chem.rs); ms2_reference defines no version constant.
CHEMISTRY = "ms2-chem-v0.1"
N_RAW = 512
SOURCE = "CASMI 2026 structures.parquet (molecules only)"
SUBSETS = ("train", "validation")


def export(data_dir: Path, out_dir: Path, name: str, max_heavy: int, seed: int,
           limit_train: int | None = None, limit_validation: int | None = None) -> dict:
    """Write `<name>_<subset>.json` for train and validation; return a report.

    Molecule order inside each subset is a seeded permutation over that
    subset's structures sorted by SMILES, so it does not depend on the
    parquet row order. A `--limit-*` keeps the first N structures of that
    order before filtering. Exits non-zero when the written subsets share an
    `identity_group` (contract: the subsets are identity-disjoint).
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = {"train": limit_train, "validation": limit_validation}
    rows = pq.read_table(data_dir / "folds" / "structures.parquet",
                         columns=["inchikey14", "smiles", "identity_group",
                                  "fold_identity"]).to_pylist()
    rng = np.random.default_rng(seed)
    report: dict = {"subsets": {}, "shared_identity_groups": []}

    for subset in SUBSETS:
        cand = sorted((r for r in rows
                       if subset_of(int(r["fold_identity"]), int(r["identity_group"])) == subset),
                      key=lambda r: r["smiles"])
        cand = [cand[i] for i in rng.permutation(len(cand))]
        limit = limits[subset]
        if limit is not None:
            cand = cand[:limit]
        considered = len(cand)
        skipped: Counter = Counter()
        molecules = []
        for row in cand:
            key, smi = row["inchikey14"], row["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped["unparsed"] += 1
                continue
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                skipped["kekulize_failed"] += 1
                continue
            reasons = ref.classify(mol)
            if reasons:
                skipped[reasons[0]] += 1
                continue
            if mol.GetNumAtoms() > max_heavy:
                skipped["too_many_atoms"] += 1
                continue
            atoms, bonds = ref.graph_of(mol)
            molecules.append({"key": key, "smiles": smi,
                              "identity_group": int(row["identity_group"]),
                              "fold_identity": int(row["fold_identity"]),
                              "atoms": atoms, "bonds": bonds, "spectra": []})
        payload = {"schema_version": 1, "chemistry": CHEMISTRY, "rdkit": rdkit.__version__,
                   "source": SOURCE, "seed": seed, "n_raw": N_RAW,
                   "spectra_per_molecule": 0, "spectrum_sampling": "none",
                   "skipped_spectra": {}, "subset": subset,
                   "skipped_structures": dict(skipped), "max_heavy": max_heavy,
                   "molecules": molecules}
        text = json.dumps(payload, separators=(",", ":"))
        path = out_dir / f"{name}_{subset}.json"
        path.write_text(text)
        print(f"{path}: {len(molecules)} molecules, {len(text) / 1e6:.1f} MB, "
              f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
        print(f"  {subset}: considered {considered}, written {len(molecules)}, "
              f"skipped {dict(skipped)}")
        report["subsets"][subset] = {"considered": considered, "written": len(molecules),
                                     "skipped": dict(skipped),
                                     "groups": sorted({m["identity_group"] for m in molecules})}

    shared = (set(report["subsets"]["train"]["groups"])
              & set(report["subsets"]["validation"]["groups"]))
    report["shared_identity_groups"] = sorted(shared)
    print(f"shared identity_group values between written subsets: {len(shared)}")
    if shared:
        raise SystemExit(f"identity_group overlap between written subsets: {sorted(shared)}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-heavy", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-validation", type=int, default=None)
    args = ap.parse_args()
    export(args.data, args.out_dir or args.data / "ms2", args.name,
           args.max_heavy, args.seed, args.limit_train, args.limit_validation)


if __name__ == "__main__":
    main()
