"""Export MassSpecGym molecules with spectra for the spectrum-conditioned completion model.

Writes, per fold of `data/pinned/MassSpecGym1.5.tsv` (`train`, `val` -> validation,
`test`), one file in the `export_msgym.py` schema (so the Rust loaders
`ExportFile` / `CompletionSet` read it unchanged) and one fingerprint sidecar
in the `FingerprintStore` schema:

    <out-dir>/<name>_<fold>.json      molecules with their spectra
    <out-dir>/<name>_<fold>_fp.json   true morgan4096 on-bits, aligned by molecule index

Differences from `export_msgym.py` (all intentional):

- one molecule per InChIKey first block per fold (first SMILES in file order);
  a block seen in more than one fold is excluded everywhere and counted;
- the `test` fold is exported (to its own file) because this tool is also the
  evaluation export; nothing here trains on it;
- adducts follow the completion adduct table
  (`src/models/ms2/completion_spectrum.rs` `COMPLETION_ADDUCTS`): `[M+H]+` is
  id 1 and `[M+Na]+` is id 3, so sodium adducts are kept;
- `spectrum_id` is the numeric part of the MassSpecGym identifier
  (`MassSpecGymID0000042` -> 42), the key external fingerprint predictions
  are joined on; `row` is the TSV row;
- at most `--peaks` peaks per spectrum (the most intense, ties by lower index);
- molecules are filtered to the V0 atom vocabulary and to `--max-atoms` heavy
  atoms; ring closures are left to the Rust loader, which counts its skips;
- each molecule additionally carries `smiles` and `formula` (ignored by the
  Rust loader, used by the evaluation tool).

Fingerprint: `AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2,
nBits=4096)`, the `morgan4096` target of MIST (see
`tools/ms2/export_fingerprints_mist.py`).

The summary printed at the end also reports how far each measured precursor
is from the mass its structure implies, which is what the mass tolerance of an
evaluation has to cover.

    PYTHONPATH=tools/ms2 python tools/ms2/export_msgym_spectral.py --name msgym
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")

# Completion adduct table: name -> (id, m/z - neutral mass in micro-dalton).
ADDUCTS = {"[M+H]+": (1, 1_007_276), "[M+Na]+": (3, 22_989_221)}
FOLDS = {"train": ("train", 2), "val": ("validation", 1), "test": ("test", 0)}
INSTRUMENT_ID = {"Orbitrap": 2, "QTOF": 3}
U32_MAX = 4294967295


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_floats(text: str):
    try:
        return [float(v) for v in text.split(",") if v.strip() != ""]
    except ValueError:
        return None


def morgan4096(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
    return sorted(int(i) for i in fp.GetOnBits())


def spectrum_record(row_index: int, row: dict, peaks: int):
    """One export spectrum, or a skip reason."""
    adduct = row["adduct"]
    if adduct not in ADDUCTS:
        return "adduct"
    try:
        precursor = float(row["precursor_mz"])
    except ValueError:
        return "precursor_range"
    if not math.isfinite(precursor) or not 50.0 <= precursor <= 2000.0:
        return "precursor_range"
    mz = parse_floats(row["mzs"])
    it = parse_floats(row["intensities"])
    if mz is None or it is None or len(mz) != len(it):
        return "invalid_peaks"
    if not mz:
        return "empty"
    for m, v in zip(mz, it):
        if not (math.isfinite(m) and m > 0 and round(m * ref.SCALE) <= U32_MAX):
            return "invalid_peaks"
        if not (math.isfinite(v) and v >= 0):
            return "invalid_peaks"
    if not any(v > 0 for v in it):
        return "invalid_peaks"
    order = sorted(range(len(mz)), key=lambda j: (-it[j], j))[:peaks]
    order.sort()
    ce_text = row["collision_energy"].strip()
    try:
        ce = float(ce_text) if ce_text else None
    except ValueError:
        ce = None
    known = 1 if ce is not None and math.isfinite(ce) else 0
    identifier = row["identifier"]
    assert identifier.startswith("MassSpecGymID"), identifier
    return {
        "row": row_index, "spectrum_id": int(identifier[len("MassSpecGymID"):]),
        "adduct": ADDUCTS[adduct][0], "polarity": 1,
        "precursor_mz_udalton": round(precursor * ref.SCALE),
        "precursor_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals([precursor])),
        "raw_peak_count": len(mz), "peak_id": order,
        "mz_udalton": [round(mz[j] * ref.SCALE) for j in order],
        "intensity": [float(it[j]) for j in order],
        "mz_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals(mz)),
        "collision_energy_ev": float(ce) if known else 0.0,
        "collision_energy_known": known, "energy_count": 1 if known else 0,
        "instrument_class": INSTRUMENT_ID.get(row["instrument_type"], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tsv", type=Path, default=Path("data/pinned/MassSpecGym1.5.tsv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/ms2/specgen"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-atoms", type=int, default=32)
    ap.add_argument("--train-spectra", type=int, default=8,
                    help="spectra kept per training molecule")
    ap.add_argument("--eval-spectra", type=int, default=2,
                    help="spectra kept per validation/test molecule")
    ap.add_argument("--peaks", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    # Pass 1: rows per (fold, block); cross-fold blocks are excluded.
    rows_of: dict[str, list[int]] = {}
    fold_of: dict[str, set[str]] = {}
    smiles_of: dict[str, str] = {}
    formula_of: dict[str, str] = {}
    with open(args.tsv, newline="") as fh:
        for index, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            block = row["inchikey"]
            if row["fold"] not in FOLDS or len(block) != 14:
                continue
            rows_of.setdefault(block, []).append(index)
            fold_of.setdefault(block, set()).add(row["fold"])
            smiles_of.setdefault(block, row["smiles"])
            formula_of.setdefault(block, row["formula"])
    cross_fold = sorted(b for b, folds in fold_of.items() if len(folds) > 1)
    blocks = sorted(b for b in rows_of if b not in set(cross_fold))
    identity_group = {b: i for i, b in enumerate(sorted(rows_of))}

    # Structure domain.
    skipped_molecules: Counter = Counter()
    graphs: dict[str, tuple] = {}
    for block in blocks:
        smiles = smiles_of[block]
        if Chem.MolFromSmiles(smiles) is None:
            skipped_molecules["unparsable"] += 1
            continue
        try:
            kek = ref.kekulized(smiles)
        except Exception:
            skipped_molecules["kekulize_failed"] += 1
            continue
        reasons = ref.classify(kek)
        if reasons:
            skipped_molecules[reasons[0]] += 1
            continue
        atoms, bonds = ref.graph_of(kek)
        if len(atoms) > args.max_atoms:
            skipped_molecules["too_many_atoms"] += 1
            continue
        graphs[block] = (atoms, bonds)

    # Pass 2: spectra of the kept molecules.
    rng = random.Random(args.seed)
    wanted: dict[int, str] = {}
    candidates_by_block: dict[str, list[int]] = {b: rows_of[b] for b in graphs}
    del rows_of
    spectra: dict[str, list[dict]] = {b: [] for b in graphs}
    skipped_spectra: Counter = Counter()
    all_rows = {i: b for b, rows in candidates_by_block.items() for i in rows}
    parsed: dict[str, list[dict]] = {b: [] for b in graphs}
    with open(args.tsv, newline="") as fh:
        for index, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            block = all_rows.get(index)
            if block is None:
                continue
            record = spectrum_record(index, row, args.peaks)
            if isinstance(record, str):
                skipped_spectra[record] += 1
                continue
            parsed[block].append(record)
    del wanted
    ppm_errors: list[float] = []
    for block in sorted(graphs):
        fold = next(iter(fold_of[block]))
        take = args.train_spectra if fold == "train" else args.eval_spectra
        records = parsed[block]
        if len(records) > take:
            records = [records[j] for j in sorted(rng.sample(range(len(records)), take))]
        spectra[block] = records
        atoms, _ = graphs[block]
        exact = ref.mass_of(ref.composition(atoms))
        for record in records:
            shift = next(s for (i, s) in ADDUCTS.values() if i == record["adduct"])
            neutral = record["precursor_mz_udalton"] - shift
            ppm_errors.append(abs(neutral - exact) / exact * 1e6)

    tsv_sha = sha256_of(args.tsv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"cross_fold_blocks": len(cross_fold), "skipped_molecules": dict(skipped_molecules),
               "skipped_spectra": dict(skipped_spectra), "folds": {}}
    for fold, (subset, fold_identity) in FOLDS.items():
        molecules, bits, keys = [], [], []
        without_spectrum = 0
        for block in sorted(graphs):
            if next(iter(fold_of[block])) != fold:
                continue
            if not spectra[block]:
                without_spectrum += 1
                continue
            atoms, bonds = graphs[block]
            molecules.append({
                "key": block, "smiles": smiles_of[block], "formula": formula_of[block],
                "identity_group": identity_group[block], "fold_identity": fold_identity,
                "atoms": atoms, "bonds": bonds, "spectra": spectra[block],
            })
            bits.append(morgan4096(smiles_of[block]))
            keys.append(f"{block}|{identity_group[block]}")
        export = {
            "schema_version": 1, "chemistry": "ms2-chem-v0.1", "rdkit": rdkit.__version__,
            "source": "MassSpecGym1.5.tsv", "tsv_sha256": tsv_sha, "fold": fold,
            "seed": args.seed, "n_raw": args.peaks,
            "spectra_per_molecule": args.train_spectra if fold == "train" else args.eval_spectra,
            "spectrum_sampling": "seeded_without_replacement",
            "adducts": {name: i for name, (i, _) in ADDUCTS.items()},
            "max_atoms": args.max_atoms,
            "skipped_spectra": dict(skipped_spectra), "subset": subset, "molecules": molecules,
        }
        path = args.out_dir / f"{args.name}_{subset}.json"
        path.write_text(json.dumps(export, separators=(",", ":")))
        fp_path = args.out_dir / f"{args.name}_{subset}_fp.json"
        fp_path.write_text(json.dumps({
            "fingerprint": "morgan4096",
            "definition": "AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=4096)",
            "rdkit": rdkit.__version__, "export": path.name, "export_sha256": sha256_of(path),
            "n_molecules": len(molecules), "keys_by_molecule": keys, "bits_by_molecule": bits,
        }, separators=(",", ":")))
        n_spectra = sum(len(m["spectra"]) for m in molecules)
        summary["folds"][subset] = {
            "molecules": len(molecules), "spectra": n_spectra,
            "molecules_without_usable_spectrum": without_spectrum,
            "adducts": dict(Counter(s["adduct"] for m in molecules for s in m["spectra"])),
            "mean_on_bits": round(sum(len(b) for b in bits) / max(len(bits), 1), 2),
        }
    ppm_errors.sort()
    quantile = lambda q: round(ppm_errors[min(len(ppm_errors) - 1, int(q * len(ppm_errors)))], 3)
    summary["precursor_vs_structure_mass_ppm"] = {
        "n": len(ppm_errors), "median": quantile(0.5), "p90": quantile(0.9),
        "p95": quantile(0.95), "p99": quantile(0.99), "max": round(ppm_errors[-1], 3),
        "within_5ppm": round(sum(e <= 5 for e in ppm_errors) / len(ppm_errors), 4),
        "within_10ppm": round(sum(e <= 10 for e in ppm_errors) / len(ppm_errors), 4),
        "within_20ppm": round(sum(e <= 20 for e in ppm_errors) / len(ppm_errors), 4),
    }
    (args.out_dir / f"{args.name}_export_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
