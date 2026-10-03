"""Export in-domain CASMI spectra and parent graphs for the Rust MS2 pipeline.

Writes one JSON file per subset of docs/MS2_CONTRACTS.md section 1 (never the
test fold): the kekulized parent graph of each sampled molecule as atom-type ids
and bonds, and its spectra as the integer fields of `SpectrumBatch` (section 3.1).
Labels are not exported: the Rust reference builds them from these inputs, so
there is one implementation of the recipe in the training path.

The output is derived from the competition data and stays outside the repository
(`--out-dir`, by default next to the data).

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/export_casmi.py \
        --data /Users/ods/Documents/Enveda_CASMI/kobayashi/exp-casmi-26-from-spectra-to-structures/data \
        --name pilot --train-molecules 2000 --validation-molecules 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem, RDLogger

import ms2_reference as ref
from audit_casmi import instrument_class

RDLogger.DisableLog("rdApp.*")

ADDUCT_ID = {name: i for i, (name, _, _) in ref.ADDUCTS.items()}
INSTRUMENT_ID = {"timstof": 1, "orbitrap": 2, "qtof": 3, "other": 4}
N_RAW = 512


def request_skip_reason(row: dict) -> str | None:
    """The request-level domain checks of an exported spectrum (contracts §4.3, §4.6)."""
    want = {"positive": "+", "negative": "-"}.get(row["ionization_mode"], "?")
    if row["adduct"] not in ADDUCT_ID:
        return "adduct"
    if not row["adduct"].endswith(want):
        return "polarity_conflict"
    if not 50.0 <= row["precursor_mz"] <= 2000.0:
        return "precursor_range"
    return None


def subset_of(fold: int, group: int) -> str | None:
    """Contract section 1; the test fold and the ranking/calibration parts are not exported."""
    if fold in (2, 3, 4):
        return "train"
    if fold == 1 and group % 3 == 0:
        return "validation"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-molecules", type=int, default=2000)
    ap.add_argument("--validation-molecules", type=int, default=400)
    ap.add_argument("--spectra-per-molecule", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--rows-from", type=Path, default=None,
                    help="a pilot report (target_pilot.json): export exactly its sampled rows as one subset")
    args = ap.parse_args()
    out_dir = args.out_dir or args.data / "ms2"
    out_dir.mkdir(parents=True, exist_ok=True)

    structures = pq.read_table(args.data / "folds" / "structures.parquet", columns=[
        "inchikey14", "smiles", "identity_group", "fold_identity"]).to_pandas()
    # A molecule is one stored structure, keyed by its own SMILES. InChIKey14 is not a
    # key: 1,612 of them cover several tautomeric structures, and a spectrum must be
    # labeled against the structure of its own row.
    structures = structures.set_index("smiles")
    rng = np.random.default_rng(args.seed)

    pinned_rows = None
    if args.rows_from:
        # The pilot's rows: a second implementation is compared on exactly the same spectra.
        pinned_rows = set(json.loads(args.rows_from.read_text())["sample"]["rows"])
        wanted = {"train": len(pinned_rows)}
        chosen = {}
        smiles_column = pq.read_table(args.data / "raw" / "train.parquet", columns=["normalized_smiles"])
        keys_of_rows = {smiles_column["normalized_smiles"][r].as_py() for r in pinned_rows}
        for key in sorted(keys_of_rows):
            row = structures.loc[key]
            # A pinned row is exported as training data: refuse anything outside the train subset.
            if subset_of(int(row.fold_identity), int(row.identity_group)) != "train":
                raise SystemExit(f"pinned row of {key} is in fold {row.fold_identity}, not the train subset")
            mol = Chem.MolFromSmiles(key)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            assert not ref.classify(mol), key
            atoms, bonds = ref.graph_of(mol)
            chosen[key] = {"subset": "train", "key": row.inchikey14, "smiles": key,
                           "identity_group": int(row.identity_group),
                           "fold_identity": int(row.fold_identity), "atoms": atoms, "bonds": bonds, "spectra": []}
    else:
        # Sample molecules per subset, in-domain only, in a seeded order.
        wanted = {"train": args.train_molecules, "validation": args.validation_molecules}
        chosen: dict[str, dict] = {}
        have = Counter()
        for key in structures.index[rng.permutation(len(structures))]:
            if all(have[s] >= n for s, n in wanted.items()):
                break
            row = structures.loc[key]
            subset = subset_of(int(row.fold_identity), int(row.identity_group))
            if subset is None or have[subset] >= wanted[subset]:
                continue
            mol = Chem.MolFromSmiles(key)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if ref.classify(mol):
                continue
            atoms, bonds = ref.graph_of(mol)
            have[subset] += 1
            chosen[key] = {"subset": subset, "key": row.inchikey14, "smiles": key,
                           "identity_group": int(row.identity_group),
                           "fold_identity": int(row.fold_identity), "atoms": atoms, "bonds": bonds, "spectra": []}

    f = pq.ParquetFile(args.data / "raw" / "train.parquet")
    skipped = Counter()
    selected = None
    if pinned_rows is None:
        # Each molecule's spectra are drawn uniformly (seeded) from all of its in-domain rows. Taking
        # the first rows in file order, as the first version did, picked peak-rich sources: the pilot
        # export's median was 174 peaks against 42 over the whole file, and its labeled rate 96.8%
        # against 82.3% on a row-group sample.
        candidates: dict[str, list[int]] = {k: [] for k in chosen}
        offset = 0
        for g in range(f.metadata.num_row_groups):
            t = f.read_row_group(g, columns=["normalized_smiles", "adduct", "ionization_mode",
                                             "precursor_mz"]).to_pylist()
            for i, row in enumerate(t):
                if row["normalized_smiles"] not in chosen:
                    continue
                reason = request_skip_reason(row)
                if reason:
                    skipped[reason] += 1
                else:
                    candidates[row["normalized_smiles"]].append(offset + i)
            offset += len(t)
        selected = set()
        for key in sorted(candidates):
            rows = candidates[key]
            take = min(args.spectra_per_molecule, len(rows))
            selected.update(rows[j] for j in rng.choice(len(rows), size=take, replace=False))

    # One pass over the spectra, reading the chosen rows (or the pinned ones).
    offset = 0
    cols = ["normalized_smiles", "adduct", "ionization_mode", "instrument_type", "precursor_mz", "ms2_mzs",
            "ms2_normalized_intensities", "collision_energy_ev"]
    for g in range(f.metadata.num_row_groups):
        n_rows = f.metadata.row_group(g).num_rows
        wanted_rows = pinned_rows if pinned_rows is not None else selected
        hits = [i for i in range(n_rows) if offset + i in wanted_rows]
        if hits:
            t = f.read_row_group(g, columns=cols).take(hits).to_pylist()
            for i, row in zip(hits, t):
                mol = chosen[row["normalized_smiles"]]
                want = {"positive": "+", "negative": "-"}.get(row["ionization_mode"], "?")
                reason = request_skip_reason(row)
                if reason:
                    skipped[reason] += 1
                    continue
                mz, it = list(row["ms2_mzs"] or []), list(row["ms2_normalized_intensities"] or [])
                if not mz:
                    skipped["empty"] += 1
                    continue
                raw = len(mz)
                order = sorted(range(raw), key=lambda j: (-it[j], j))[:N_RAW]
                order.sort()
                ce = row["collision_energy_ev"] or []
                mol["spectra"].append({
                    "row": offset + i, "spectrum_id": offset + i,
                    "adduct": ADDUCT_ID[row["adduct"]], "polarity": 1 if want == "+" else -1,
                    "precursor_mz_udalton": round(row["precursor_mz"] * ref.SCALE),
                    "precursor_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals([row["precursor_mz"]])),
                    "raw_peak_count": raw, "peak_id": order,
                    "mz_udalton": [round(mz[j] * ref.SCALE) for j in order],
                    "intensity": [float(it[j]) for j in order],
                    "mz_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals(mz)),
                    "collision_energy_ev": float(np.mean(ce)) if ce else 0.0,
                    "collision_energy_known": 1 if ce else 0,
                    "energy_count": min(len(ce), 8),
                    "instrument_class": INSTRUMENT_ID[instrument_class(row["instrument_type"])],
                })
        offset += n_rows

    provenance = {
        "schema_version": 1, "chemistry": "ms2-chem-v0.1", "rdkit": rdkit.__version__,
        "source": "CASMI 2026 train.parquet", "seed": args.seed, "n_raw": N_RAW,
        "spectra_per_molecule": args.spectra_per_molecule, "spectrum_sampling": "uniform-in-domain-v2" if pinned_rows is None else "pinned-rows",
        "skipped_spectra": dict(skipped),
    }
    for subset in wanted:
        molecules = [{k: v for k, v in m.items() if k != "subset"}
                     for m in chosen.values() if m["subset"] == subset and m["spectra"]]
        molecules.sort(key=lambda m: (m["key"], m["smiles"]))
        payload = json.dumps({**provenance, "subset": subset, "molecules": molecules}, separators=(",", ":"))
        path = out_dir / f"{args.name}_{subset}.json"
        path.write_text(payload)
        print(f"{path}: {len(molecules)} molecules, {sum(len(m['spectra']) for m in molecules)} spectra, "
              f"{len(payload) / 1e6:.1f} MB, sha256 {hashlib.sha256(payload.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
