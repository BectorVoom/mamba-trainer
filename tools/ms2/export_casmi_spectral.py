"""Export CASMI 2026 molecules with spectra for the spectrum-conditioned completion model.

The competition counterpart of `export_msgym_spectral.py`: the same schema (so
`ExportFile` / `CompletionSet` read it unchanged) and the same `morgan4096`
sidecar, built from the competition training file through a cleaned-spectrum
index (`spec_meta.parquet`: one row per training spectrum in file order with
its identity, fold and the split of the fingerprint predictor).

Subsets, all structure-disjoint by the scorer identity (tautomer-canonical
InChIKey first block):

* `train`: every identity the fingerprint predictor trained on. Its spectra
  are sampled (`--train-spectra`, seeded) from the rows with a completion
  adduct.
* one subset per `--queries NAME=FILE.npz`: identities the predictor never
  saw, with exactly the spectra of that query file (the rows the predictor's
  molecule-level prediction was aggregated from), in the file's order, so a
  prediction and the peaks a query supplies describe the same measurements.

An identity is skipped and counted when its structure is outside the V0 atom
vocabulary, has more than `--max-atoms` heavy atoms, or keeps no spectrum.
`spectrum_id` is the row of the spectrum in the training file.

    PYTHONPATH=tools/ms2 python tools/ms2/export_casmi_spectral.py \
        --train-parquet ../Enveda_CASMI/data/competition/train.parquet \
        --meta ../Enveda_CASMI/data/work/spec_meta.parquet \
        --folds ../Enveda_CASMI/data/folds/structures.parquet \
        --queries dev=../Enveda_CASMI/data/work/fp_hold_dev.npz \
        --name casmi --out-dir data/ms2/specgen/casmi
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")

# Completion adduct table (`src/models/ms2/completion_spectrum.rs` `COMPLETION_ADDUCTS`):
# name -> (id, m/z - neutral mass in micro-dalton).
ADDUCTS = {"[M+H]+": (1, 1_007_276), "[M-H]-": (2, -1_007_276), "[M+Na]+": (3, 22_989_221),
           "[M+NH4]+": (4, 18_033_826), "[M+K]+": (5, 38_963_158)}
U32_MAX = 4294967295


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def morgan4096(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
    return sorted(int(i) for i in fp.GetOnBits())


def spectrum_record(row_index: int, row: dict, peaks: int):
    """One export spectrum, or a skip reason."""
    if row["adduct"] not in ADDUCTS:
        return "adduct"
    polarity = {"positive": 1, "negative": -1}.get(row["ionization_mode"])
    if polarity is None or (row["adduct"][-1] == "+") != (polarity == 1):
        return "polarity_conflict"
    precursor = row["precursor_mz"]
    if precursor is None or not np.isfinite(precursor) or not 50.0 <= precursor <= 2000.0:
        return "precursor_range"
    mz, it = row["ms2_mzs"], row["ms2_normalized_intensities"]
    if mz is None or it is None or len(mz) != len(it):
        return "invalid_peaks"
    if not len(mz):
        return "empty"
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    if not (np.isfinite(mz).all() and (mz > 0).all() and (np.round(mz * ref.SCALE) <= U32_MAX).all()):
        return "invalid_peaks"
    if not (np.isfinite(it).all() and (it >= 0).all() and (it > 0).any()):
        return "invalid_peaks"
    # the most intense peaks, ties by lower index, back in file order
    order = np.sort(np.lexsort((np.arange(len(mz)), -it))[:peaks]).tolist()
    ce = row["collision_energy_ev"]
    ce = [float(v) for v in ce] if ce is not None else []
    return {
        "row": row_index, "spectrum_id": row_index,
        "adduct": ADDUCTS[row["adduct"]][0], "polarity": polarity,
        "precursor_mz_udalton": round(precursor * ref.SCALE),
        "precursor_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals([precursor])),
        "raw_peak_count": len(mz), "peak_id": order,
        "mz_udalton": [round(float(mz[j]) * ref.SCALE) for j in order],
        "intensity": [float(it[j]) for j in order],
        "mz_uncertainty_udalton": ref.mz_uncertainty(ref.stored_decimals(mz.tolist())),
        "collision_energy_ev": float(np.mean(ce)) if ce else 0.0,
        "collision_energy_known": 1 if ce else 0, "energy_count": min(len(ce), 8),
        "instrument_class": 1 if "timstof" in (row["instrument_type"] or "").lower() else 4,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--queries", action="append", default=[], metavar="NAME=FILE.npz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-atoms", type=int, default=32)
    ap.add_argument("--train-spectra", type=int, default=8)
    ap.add_argument("--no-train", action="store_true", help="export the query subsets only")
    ap.add_argument("--peaks", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = pd.read_parquet(args.meta, columns=["scorer_key", "adduct", "fp_split", "identity_group",
                                               "fold_identity", "n_heavy", "n_clean"])
    structures = pd.read_parquet(args.folds, columns=["scorer_key", "smiles", "formula"])
    structures = structures.drop_duplicates("scorer_key").set_index("scorer_key")

    # Rows wanted per subset and identity, in export order.
    wanted: dict[str, dict[str, list[int]]] = {}
    query_keys: set[str] = set()
    for spec in args.queries:
        name, path = spec.split("=", 1)
        q = np.load(path, allow_pickle=True)
        wanted[name] = {str(k): [int(r) for r in rows] for k, rows in zip(q["keys"], q["rows"])}
        assert not query_keys & set(wanted[name]), "query files share an identity"
        query_keys |= set(wanted[name])
    if not args.no_train:
        usable = (meta.fp_split.values == "train") & meta.adduct.isin(list(ADDUCTS)).values \
            & (meta.n_clean.values > 0) & (meta.n_heavy.values <= args.max_atoms)
        rows = np.flatnonzero(usable)
        groups = pd.Series(rows).groupby(meta.scorer_key.values[rows]).apply(lambda s: s.values)
        train = {}
        for key in sorted(groups.index):
            r = groups[key]
            if len(r) > args.train_spectra:
                r = np.sort(rng.choice(r, args.train_spectra, replace=False))
            train[key] = [int(v) for v in r]
        assert not query_keys & set(train), "a query identity is a training identity"
        wanted["train"] = train
    held_out = set(meta.scorer_key.values[meta.fp_split.values != "train"])
    assert all(k in held_out for k in query_keys), "a query identity was seen by the predictor"

    # Structure domain.
    skipped_molecules: dict[str, Counter] = {name: Counter() for name in wanted}
    graphs: dict[str, tuple] = {}
    for name, by_key in wanted.items():
        for key in by_key:
            smiles = structures.smiles[key]
            try:
                kek = ref.kekulized(smiles)
            except Exception:
                skipped_molecules[name]["kekulize_failed"] += 1
                continue
            reasons = ref.classify(kek)
            if reasons:
                skipped_molecules[name][reasons[0]] += 1
                continue
            atoms, bonds = ref.graph_of(kek)
            if len(atoms) > args.max_atoms:
                skipped_molecules[name]["too_many_atoms"] += 1
                continue
            graphs[key] = (atoms, bonds)

    # One pass over the training file for the rows of the kept identities.
    owner = {r: (name, key) for name, by_key in wanted.items() for key, rows in by_key.items()
             if key in graphs for r in rows}
    records: dict[int, dict] = {}
    skipped_spectra: dict[str, Counter] = {name: Counter() for name in wanted}
    pf = pq.ParquetFile(args.train_parquet)
    cols = ["adduct", "ionization_mode", "instrument_type", "precursor_mz", "ms2_mzs",
            "ms2_normalized_intensities", "collision_energy_ev"]
    offset = 0
    all_rows = np.array(sorted(owner), dtype=np.int64)
    for g in range(pf.metadata.num_row_groups):
        n_rows = pf.metadata.row_group(g).num_rows
        hits = all_rows[(all_rows >= offset) & (all_rows < offset + n_rows)] - offset
        if len(hits):
            table = pf.read_row_group(g, columns=cols).take(hits).to_pylist()
            for i, row in zip(hits.tolist(), table):
                record = spectrum_record(offset + i, row, args.peaks)
                if isinstance(record, str):
                    skipped_spectra[owner[offset + i][0]][record] += 1
                else:
                    records[offset + i] = record
        offset += n_rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_of(args.train_parquet)
    summary = {}
    for name, by_key in wanted.items():
        molecules, bits, keys = [], [], []
        ppm = []
        for key in sorted(k for k in by_key if k in graphs):
            spectra = [records[r] for r in by_key[key] if r in records]
            if not spectra:
                skipped_molecules[name]["no_usable_spectrum"] += 1
                continue
            atoms, bonds = graphs[key]
            group = int(meta.identity_group.values[by_key[key][0]])
            molecules.append({
                "key": key, "smiles": structures.smiles[key], "formula": structures.formula[key],
                "identity_group": group, "fold_identity": int(meta.fold_identity.values[by_key[key][0]]),
                "atoms": atoms, "bonds": bonds, "spectra": spectra,
            })
            bits.append(morgan4096(structures.smiles[key]))
            keys.append(f"{key}|{group}")
            exact = ref.mass_of(ref.composition(atoms))
            shift = {i: s for i, s in ADDUCTS.values()}
            ppm += [abs(s["precursor_mz_udalton"] - shift[s["adduct"]] - exact) / exact * 1e6 for s in spectra]
        export = {
            "schema_version": 1, "chemistry": "ms2-chem-v0.1", "rdkit": rdkit.__version__,
            "source": "CASMI 2026 train.parquet", "source_sha256": source_sha, "seed": args.seed,
            "n_raw": args.peaks, "spectra_per_molecule": args.train_spectra if name == "train" else 0,
            "spectrum_sampling": "seeded_without_replacement" if name == "train" else "query_file_rows",
            "adducts": {a: i for a, (i, _) in ADDUCTS.items()}, "max_atoms": args.max_atoms,
            "skipped_spectra": dict(skipped_spectra[name]), "subset": name, "molecules": molecules,
        }
        path = args.out_dir / f"{args.name}_{name}.json"
        path.write_text(json.dumps(export, separators=(",", ":")))
        (args.out_dir / f"{args.name}_{name}_fp.json").write_text(json.dumps({
            "fingerprint": "morgan4096",
            "definition": "AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=4096)",
            "rdkit": rdkit.__version__, "export": path.name, "export_sha256": sha256_of(path),
            "n_molecules": len(molecules), "keys_by_molecule": keys, "bits_by_molecule": bits,
        }, separators=(",", ":")))
        ppm = np.sort(np.array(ppm))
        summary[name] = {
            "identities_requested": len(by_key), "molecules": len(molecules),
            "spectra": sum(len(m["spectra"]) for m in molecules),
            "skipped_molecules": dict(skipped_molecules[name]), "skipped_spectra": dict(skipped_spectra[name]),
            "adducts": dict(Counter(s["adduct"] for m in molecules for s in m["spectra"])),
            "mean_on_bits": round(float(np.mean([len(b) for b in bits])), 2) if bits else 0.0,
            "precursor_vs_structure_mass_ppm": {
                "median": round(float(np.median(ppm)), 3), "p95": round(float(np.quantile(ppm, 0.95)), 3),
                "within_10ppm": round(float((ppm <= 10).mean()), 4),
            } if len(ppm) else {},
        }
    (args.out_dir / f"{args.name}_export_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
