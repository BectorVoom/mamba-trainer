"""How much of a spectrum's pseudo-labels (recipe q-cut-v1) depends on the spectrum.

The V0 pilot found that real spectra do not lower the held-out teacher-forced NLL below the shuffled-spectrum,
metadata-only and structure-prior controls (docs/MS2_SUBSTRUCTURE_TASKS.md). This diagnostic asks whether the
labels themselves carry spectrum-specific information: each spectrum's molecule is labelled with

- its own peaks;
- a sibling spectrum of the same molecule (when the export has one);
- another molecule's peaks, restricted to this spectrum's precursor window (realistic fragment masses);
- random peaks: the same count, m/z uniform over this spectrum's peak range, its own intensities permuted.

For each, the q distribution after the top-16 retention is compared with the own-peak one by the overlap
sum_g min(q_own(g), q_x(g)) (1 = identical, 0 = disjoint). If decoys overlap the own labels about as much as
siblings do, the recipe's targets are mostly a function of the parent structure, and a spectrum-conditioned model
has little to learn from the spectrum.

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/label_specificity.py \
        --export /Users/ods/Documents/Enveda_CASMI/kobayashi/exp-casmi-26-from-spectra-to-structures/data/ms2/pilot_validation.json \
        --out bench/results/ms2/label_specificity.json

The output holds aggregates only (the export is CC BY-NC).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

import ms2_reference as ref
from pilot_targets import FROZEN, candidates

RDLogger.DisableLog("rdApp.*")


def peaks_of(spectrum):
    keep, rel = ref.filter_peaks(spectrum["mz_udalton"], spectrum["intensity"], spectrum["precursor_mz_udalton"])
    return [(spectrum["mz_udalton"][i], r) for i, r in zip(keep, rel)]


def window(peaks, precursor):
    """Peaks a decoy may contribute: at most this spectrum's precursor + 2 Da, renormalised."""
    kept = [(m, r) for m, r in peaks if 0 < m <= precursor + 2 * ref.SCALE]
    if not kept:
        return []
    top = max(r for _, r in kept)
    return [(m, r / top) for m, r in kept if r / top >= 1e-3]


def labels(peaks, graphs, spectrum):
    weight, _, _, explained = ref.targets(peaks, graphs, spectrum["adduct"], FROZEN[1],
                                         spectrum["mz_uncertainty_udalton"])
    q, _ = ref.retain(weight)
    total = sum(r for _, r in peaks) or 1.0
    explained_intensity = sum(r for i, (_, r) in enumerate(peaks) if i in explained) / total
    return q, explained_intensity


def overlap(a, b):
    return sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in set(a) | set(b))


def summary(values):
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)), "p75": float(np.percentile(v, 75))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-molecules", type=int, default=0, help="0: all")
    args = ap.parse_args()
    started = time.time()
    data = json.loads(args.export.read_text())
    molecules = data["molecules"]
    if args.max_molecules:
        molecules = molecules[:args.max_molecules]
    rng = np.random.default_rng(args.seed)
    ref.MAX_CUTS = FROZEN[0]

    graphs = []
    for m in molecules:
        mol = Chem.MolFromSmiles(m["smiles"])
        Chem.Kekulize(mol, clearAromaticFlags=True)
        graphs.append(candidates(mol, m["atoms"], m["bonds"], FROZEN[0]))

    all_spectra = [(mi, s) for mi, m in enumerate(molecules) for s in m["spectra"]]
    rows = []
    for mi, m in enumerate(molecules):
        for si, s in enumerate(m["spectra"]):
            own_peaks = peaks_of(s)
            if not own_peaks:
                continue
            q_own, expl_own = labels(own_peaks, graphs[mi], s)
            row = {"molecule": mi, "labeled": bool(q_own), "targets": len(q_own), "explained": expl_own,
                   "candidates": len({g["class"] for g in graphs[mi]})}
            if len(m["spectra"]) > 1:
                sib = m["spectra"][1 - si] if len(m["spectra"]) == 2 else m["spectra"][(si + 1) % len(m["spectra"])]
                q_sib, _ = labels(window(peaks_of(sib), s["precursor_mz_udalton"]), graphs[mi], s)
                row["sibling"] = overlap(q_own, q_sib) if q_own and q_sib else None
            while True:
                oj, other = all_spectra[rng.integers(len(all_spectra))]
                if oj != mi:
                    break
            dec = window(peaks_of(other), s["precursor_mz_udalton"])
            q_dec, expl_dec = labels(dec, graphs[mi], s) if dec else ({}, 0.0)
            row["other_molecule"] = overlap(q_own, q_dec) if q_own and q_dec else None
            row["other_molecule_labeled"] = bool(q_dec)
            row["other_molecule_explained"] = expl_dec
            mz = [p for p, _ in own_peaks]
            lo, hi = min(mz), max(mz)
            rnd_mz = sorted(int(x) for x in rng.integers(lo, hi + 1, size=len(own_peaks)))
            rnd = list(zip(rnd_mz, rng.permutation([r for _, r in own_peaks]).tolist()))
            q_rnd, expl_rnd = labels(rnd, graphs[mi], s)
            row["random"] = overlap(q_own, q_rnd) if q_own and q_rnd else None
            row["random_labeled"] = bool(q_rnd)
            row["random_explained"] = expl_rnd
            rows.append(row)

    def col(key, labeled_only=True):
        return [r[key] for r in rows if r.get(key) is not None and (r["labeled"] or not labeled_only)]

    report = {
        "export": args.export.name, "export_spectrum_sampling": data.get("spectrum_sampling"),
        "recipe": "q-cut-v1", "cuts": FROZEN[0], "ppm_tenths": FROZEN[1], "seed": args.seed,
        "molecules": len(molecules), "spectra": len(rows),
        "labeled_fraction": {
            "own": float(np.mean([r["labeled"] for r in rows])),
            "other_molecule": float(np.mean([r["other_molecule_labeled"] for r in rows])),
            "random": float(np.mean([r["random_labeled"] for r in rows])),
        },
        "explained_intensity": {
            "own": summary(col("explained", False)),
            "other_molecule": summary(col("other_molecule_explained", False)),
            "random": summary(col("random_explained", False)),
        },
        "q_overlap_with_own": {
            "sibling_spectrum": summary(col("sibling")),
            "other_molecule_peaks": summary(col("other_molecule")),
            "random_peaks": summary(col("random")),
        },
        "targets_per_labeled_spectrum": summary([r["targets"] for r in rows if r["labeled"]]),
        "candidate_graphs_per_molecule": summary([r["candidates"] for r in rows]),
        "seconds": time.time() - started,
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("labeled_fraction", "explained_intensity", "q_overlap_with_own",
                                              "targets_per_labeled_spectrum", "candidate_graphs_per_molecule")},
                     indent=1))


if __name__ == "__main__":
    main()
