"""P0.7 / P0.8 pilot: what the pseudo-label recipe yields on real spectra.

Applies the Python reference of recipe `q-cut-v1` (`ms2_reference.py`, the same
code the Rust fixtures come from) to a fixed sample of in-domain CASMI spectra
from the **training folds only**, and reports how much of a spectrum the recipe
explains, how often the same peaks are explained by the subgraphs of an
unrelated molecule, what the top-16 cut drops, and the action-trace sizes the
targets need. A grid over the cut budget and the tolerance shows the trade-off
the frozen setting (2 cuts, 10 ppm) was chosen from.

Every rate uses the same denominators: all sampled spectra and all their
filtered peaks, whether or not the molecule has a candidate subgraph.

These are pseudo-labels: a mass match under a hydrogen-shift rule, not an
experimentally assigned fragment. Graph identity here is RDKit's canonical
fragment SMILES with every atom tagged by its type id; the Rust reference uses
canonical traces, and P1.9 compares its counts on this sample.

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/pilot_targets.py \
        --data /Users/ods/Documents/Enveda_CASMI/kobayashi/exp-casmi-26-from-spectra-to-structures/data \
        --out bench/results/ms2/target_pilot.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")

ADDUCT_ID = {name: i for i, (name, _, _) in ref.ADDUCTS.items()}
TRAIN_FOLDS = (2, 3, 4)
GRID = [(cuts, ppm_tenths) for cuts in (1, 2, 3) for ppm_tenths in (50, 100, 200)]
FROZEN = (2, 100)
N_RAW = 512


def candidates(mol, atoms, bonds, max_cuts):
    """Recipe subgraphs of one molecule as `targets()` wants them."""
    ref.MAX_CUTS = max_cuts
    found = ref.enumerate_subgraphs(atoms, bonds)
    out = []
    for members, (boundary, closures) in sorted(found.items()):
        out.append({
            "atoms": members, "boundary": boundary, "closures": closures,
            "counts": ref.composition([atoms[i] for i in members]),
            "class": ref.fragment_smiles(mol, atoms, members),
        })
    return out


def quant(values):
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return {}
    return {f"p{p}": float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95, 100)} | {"mean": float(v.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--molecules", type=int, default=300)
    ap.add_argument("--max-parent-heavy", type=int, default=60)
    ap.add_argument("--row-group", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    f = pq.ParquetFile(args.data / "raw" / "train.parquet")
    t = f.read_row_group(args.row_group, columns=[
        "normalized_smiles", "inchikey14", "adduct", "precursor_mz", "ms2_mzs",
        "ms2_normalized_intensities", "ionization_mode"]).to_pandas()
    folds = pq.read_table(args.data / "folds" / "structures.parquet",
                          columns=["inchikey14", "fold_identity"]).to_pandas()
    fold_of = dict(zip(folds.inchikey14.values, folds.fold_identity.values))
    row_offset = sum(f.metadata.row_group(g).num_rows for g in range(args.row_group))
    t["row"] = np.arange(len(t)) + row_offset
    t = t[t.adduct.isin(ADDUCT_ID)]
    t = t[[fold_of.get(k, -1) in TRAIN_FOLDS for k in t.inchikey14.values]]
    rng = np.random.default_rng(args.seed)
    t = t.iloc[rng.permutation(len(t))].drop_duplicates("inchikey14")

    # Fix the sample first, so every grid cell and both nulls see the same spectra.
    sample = []
    skipped = Counter()
    truncated = 0
    for row in t.itertuples():
        if len(sample) >= args.molecules:
            break
        want = {"positive": "+", "negative": "-"}.get(row.ionization_mode, "?")
        if not row.adduct.endswith(want):
            skipped["polarity_conflict"] += 1
            continue
        if not 50.0 <= row.precursor_mz <= 2000.0:
            skipped["precursor_range"] += 1
            continue
        mol = Chem.MolFromSmiles(row.normalized_smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        if ref.classify(mol):
            skipped["outside_structure_domain"] += 1
            continue
        if mol.GetNumHeavyAtoms() > args.max_parent_heavy:
            skipped["parent_too_large_for_pilot"] += 1
            continue
        raw_mz = list(row.ms2_mzs)
        raw_it = list(row.ms2_normalized_intensities)
        # The host adapter's pre-selection (contract section 3.1): the N_RAW most intense, ties by index.
        order = sorted(sorted(range(len(raw_mz)), key=lambda j: (-raw_it[j], j))[:N_RAW])
        truncated += len(raw_mz) > N_RAW
        mz = [round(raw_mz[j] * ref.SCALE) for j in order]
        keep, rel = ref.filter_peaks(mz, [raw_it[j] for j in order], round(row.precursor_mz * ref.SCALE))
        if not keep:
            skipped["empty_spectrum"] += 1
            continue
        atoms, bonds = ref.graph_of(mol)
        sample.append({
            "row": int(row.row), "mol": mol, "atoms": atoms, "bonds": bonds,
            "adduct": ADDUCT_ID[row.adduct],
            "peaks": [(mz[i], r) for i, r in zip(keep, rel)],
            "uncertainty": ref.mz_uncertainty(ref.stored_decimals(raw_mz)),
        })

    total_peaks = sum(len(s["peaks"]) for s in sample)
    cells = []
    frozen_detail = {}
    for cuts in (1, 2, 3):
        graphs = [candidates(s["mol"], s["atoms"], s["bonds"], cuts) for s in sample]
        for ppm_tenths in (50, 100, 200):
            c = Counter()
            explained_int, wrong_int, overlap_int, n_targets, dropped = [], [], [], [], []
            sizes = Counter(); closures = Counter(); boundary = Counter(); types = Counter()
            for i, s in enumerate(sample):
                total = sum(r for _, r in s["peaks"])
                weight, _, ambiguous, explained = ref.targets(
                    s["peaks"], graphs[i], s["adduct"], ppm_tenths, s["uncertainty"])
                # The unrelated molecule is the next one of the sample (cyclic), same peaks.
                other = graphs[(i + 1) % len(sample)]
                _, _, _, wrong = ref.targets(s["peaks"], other, s["adduct"], ppm_tenths, s["uncertainty"])
                kept, drop = ref.retain(weight)
                c["matched_peaks"] += len(explained)
                c["wrong_parent_matched_peaks"] += len(wrong)
                c["matched_by_both"] += len(explained & wrong)
                c["ambiguous_hypotheses"] += ambiguous
                c["spectra_with_target"] += bool(weight)
                c["spectra_without_candidate"] += not graphs[i]
                c["spectra_over_16_targets"] += len(weight) > ref.MAX_TARGETS
                explained_int.append(sum(s["peaks"][p][1] for p in explained) / total)
                wrong_int.append(sum(s["peaks"][p][1] for p in wrong) / total)
                overlap_int.append(sum(s["peaks"][p][1] for p in explained & wrong) / total)
                n_targets.append(len(weight))
                dropped.append(float(drop))
                if (cuts, ppm_tenths) == FROZEN:
                    by_class = {}
                    for g in graphs[i]:
                        by_class.setdefault(g["class"], g)
                    for cls in kept:
                        g = by_class[cls]
                        sizes[len(g["atoms"])] += 1; closures[g["closures"]] += 1; boundary[g["boundary"]] += 1
                        for a in g["atoms"]:
                            types[s["atoms"][a]] += 1
            n = len(sample)
            cells.append({
                "max_cuts": cuts, "ppm_tenths": ppm_tenths,
                "matched_peak_fraction": c["matched_peaks"] / total_peaks,
                "wrong_parent_matched_peak_fraction": c["wrong_parent_matched_peaks"] / total_peaks,
                "matched_peaks_also_matched_by_wrong_parent": c["matched_by_both"] / max(1, c["matched_peaks"]),
                "explained_intensity": quant(explained_int),
                "wrong_parent_explained_intensity": quant(wrong_int),
                "explained_intensity_shared_with_wrong_parent": quant(overlap_int),
                "spectra_with_target_fraction": c["spectra_with_target"] / n,
                "spectra_without_candidate_fraction": c["spectra_without_candidate"] / n,
                "targets_per_spectrum": quant(n_targets),
                "spectra_over_16_targets_fraction": c["spectra_over_16_targets"] / n,
                "weight_dropped_by_top16": quant(dropped),
                "ambiguous_hypotheses": int(c["ambiguous_hypotheses"]),
                "subgraphs_per_molecule": quant([len(g) for g in graphs]),
            })
            if (cuts, ppm_tenths) == FROZEN:
                frozen_detail = {
                    "retained_targets": int(sum(sizes.values())),
                    "size_histogram": {str(k): int(v) for k, v in sorted(sizes.items())},
                    "ring_closure_histogram": {str(k): int(v) for k, v in sorted(closures.items())},
                    "boundary_bond_histogram": {str(k): int(v) for k, v in sorted(boundary.items())},
                    "atom_type_histogram": {str(k): int(v) for k, v in sorted(types.items())},
                }
    report = {
        "schema_version": 2,
        "recipe": "q-cut-v1 (docs/MS2_CONTRACTS.md section 7.2) via tools/ms2/ms2_reference.py",
        "identity": "RDKit canonical fragment SMILES, atoms tagged with their type id",
        "sample": {
            "row_group": args.row_group, "seed": args.seed, "folds": list(TRAIN_FOLDS),
            "molecules": len(sample), "peaks": total_peaks, "rows": [s["row"] for s in sample],
            "skipped": dict(skipped), "n_raw": N_RAW, "spectra_over_n_raw": int(truncated),
            "mz_uncertainty_udalton": {str(k): int(v) for k, v in
                                       sorted(Counter(s["uncertainty"] for s in sample).items())},
        },
        "limits": {"min_atoms": ref.MIN_ATOMS, "max_atoms": ref.MAX_ATOMS, "max_closures": ref.MAX_CLOSURES,
                   "max_shift": ref.MAX_SHIFT, "max_targets": ref.MAX_TARGETS},
        "wrong_parent": "the same peaks matched against the next sampled molecule's subgraphs",
        "grid": cells,
        "frozen_setting": {"max_cuts": FROZEN[0], "ppm_tenths": FROZEN[1], **frozen_detail},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}; molecules {len(sample)}, peaks {total_peaks}")


if __name__ == "__main__":
    main()
