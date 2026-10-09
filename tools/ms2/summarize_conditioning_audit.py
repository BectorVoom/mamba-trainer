"""Score a conditioning audit against its independent molecule roster.

Includes empty lists in every per-query metric, checks RDKit validity, scores
tautomer-canonical competition keys, and bootstraps paired molecule differences.
These are diagnostic results; MIST formula inputs and precursor masses remain
favorable. Run using data/ms2/specgen/venv/bin/python.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from competition_eval import fingerprint, mol_of, rank_of, scorer_key

RDLogger.DisableLog("rdApp.*")

FIXED_METRICS = ["best_tanimoto_top25", "mean_tanimoto_top25",
                 "best_tanimoto_top25_logprob", "mean_tanimoto_top25_logprob"]


def fixed_metrics(row):
    """Equal candidate-position budget; missing and invalid slots score zero."""
    target = mol_of(row["target"]["atoms"], row["target"]["bonds"])
    target_fp = fingerprint(target)
    candidates = row["candidates"]
    model_order = list(range(min(25, len(candidates))))
    lp_order = sorted(range(len(candidates)), key=lambda i: -candidates[i]["best_log_prob"])[:25]
    selected = set(model_order) | set(lp_order)
    similarities = {}
    rings3 = rings4 = 0
    for index, candidate in enumerate(candidates):
        try:
            mol = mol_of(candidate["atoms"], candidate["bonds"])
            rings = mol.GetRingInfo().AtomRings()
            rings3 += any(len(ring) == 3 for ring in rings)
            rings4 += any(len(ring) == 4 for ring in rings)
            if index in selected:
                fp = fingerprint(mol)
                similarities[index] = float(DataStructs.TanimotoSimilarity(target_fp, fp)) if fp else 0.0
        except Exception:
            if index in selected:
                similarities[index] = 0.0
    result = {}
    for suffix, order in [("top25", model_order), ("top25_logprob", lp_order)]:
        values = [similarities.get(index, 0.0) for index in order]
        result[f"best_tanimoto_{suffix}"] = max(values, default=0.0)
        result[f"mean_tanimoto_{suffix}"] = sum(values) / 25
    target_rings = target.GetRingInfo().AtomRings()
    result.update({"candidate_3rings": rings3, "candidate_4rings": rings4,
                   "target_3ring": int(any(len(r) == 3 for r in target_rings)),
                   "target_4ring": int(any(len(r) == 4 for r in target_rings))})
    return result


def score(row):
    target = mol_of(row["target"]["atoms"], row["target"]["bonds"])
    target_key, target_fp = scorer_key(target), fingerprint(target)
    if not target_key or target_fp is None:
        raise ValueError(f"unresolvable roster target: {row['key']}")
    target_bits = set(target_fp.GetOnBits())
    input_bits = {int(bit) for bit, _ in row["inputs"]["fingerprint"]["bits"]}
    keys, similarities, logprobs = [], [], []
    invalid = disconnected = strain = 0
    for candidate in row["candidates"]:
        logprobs.append(float(candidate["best_log_prob"]))
        try:
            mol = mol_of(candidate["atoms"], candidate["bonds"])
            disconnected += len(Chem.GetMolFrags(mol)) != 1
            fp = fingerprint(mol)
            key = scorer_key(mol)
            if fp is None or not key:
                raise ValueError("unresolvable candidate")
            # A limited, explicitly labeled strain heuristic, not a claim of
            # chemical impossibility: triple bonds in <=7-membered rings, or
            # two adjacent ring double bonds sharing the same ring atom.
            flagged = False
            for ring in mol.GetRingInfo().AtomRings():
                ring_atoms = set(ring)
                double_degree = {}
                for bond in mol.GetBonds():
                    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if a not in ring_atoms or b not in ring_atoms:
                        continue
                    if bond.GetBondType() == Chem.BondType.TRIPLE and len(ring) <= 7:
                        flagged = True
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        double_degree[a] = double_degree.get(a, 0) + 1
                        double_degree[b] = double_degree.get(b, 0) + 1
                flagged |= any(degree > 1 for degree in double_degree.values())
            strain += flagged
            similarities.append(float(DataStructs.TanimotoSimilarity(target_fp, fp)))
            keys.append(key)
        except Exception:
            invalid += 1
            keys.append(None)
            similarities.append(0.0)
    rank = rank_of([key for key in keys if key], target_key)
    lp_order = sorted(range(len(keys)), key=lambda i: -logprobs[i])
    lp_rank = rank_of([keys[i] for i in lp_order if keys[i]], target_key)
    valid = len(keys) - invalid
    return {"key": row["key"], "spectrum_id": row["spectrum_id"],
            "candidates": len(keys), "invalid": invalid, "disconnected": disconnected,
            "strain_flagged": strain,
            "strain_fraction": strain / valid if valid else 0.0,
            "best_tanimoto": max(similarities, default=0.0),
            "mean_tanimoto": float(np.mean(similarities)) if similarities else 0.0,
            "pool_hit": int(target_key in keys),
            "mrr25": 1.0 / rank if rank else 0.0,
            "logprob_mrr25": 1.0 / lp_rank if lp_rank else 0.0,
            "top1": int(rank == 1), "top25": int(rank is not None),
            "input_precision": len(input_bits & target_bits) / len(input_bits) if input_bits else 0.0,
            "input_recall": len(input_bits & target_bits) / len(target_bits) if target_bits else 0.0}


def seen_keys(data):
    root = data.parent / "mist/downloads/canopus_extract/canopus_train_export"
    with (root / "splits/canopus_hplus_100_0.tsv").open() as stream:
        split = {r["name"]: r["split"] for r in csv.DictReader(stream, delimiter="\t")}
    seen, trained = set(), set()
    with (root / "labels.tsv").open() as stream:
        for r in csv.DictReader(stream, delimiter="\t"):
            key = r["inchikey"][:14]
            if split.get(r["spec"]) in {"train", "val"}:
                seen.add(key)
            if split.get(r["spec"]) == "train":
                trained.add(key)
    with (root / "aug_iceberg_canopus_train/biomols_filtered_smiles_canopus_train_labels.tsv").open() as stream:
        seen.update(r["inchikey"][:14] for r in csv.DictReader(stream, delimiter="\t"))
    return trained, seen


def aggregate(rows):
    n = len(rows)
    if not n:
        return {"queries": 0}
    return {"queries": n, "nonempty": sum(r["candidates"] > 0 for r in rows),
            "candidates": sum(r["candidates"] for r in rows),
            "invalid": sum(r["invalid"] for r in rows),
            "disconnected": sum(r["disconnected"] for r in rows),
            "pool_hits": sum(r["pool_hit"] for r in rows),
            "top1": sum(r["top1"] for r in rows), "top25": sum(r["top25"] for r in rows),
            "candidate_3ring_fraction": sum(r["candidate_3rings"] for r in rows) / max(1, sum(r["candidates"] for r in rows)),
            "candidate_4ring_fraction": sum(r["candidate_4rings"] for r in rows) / max(1, sum(r["candidates"] for r in rows)),
            "target_3ring_fraction": float(np.mean([r["target_3ring"] for r in rows])),
            "target_4ring_fraction": float(np.mean([r["target_4ring"] for r in rows])),
            **{name: float(np.mean([r[name] for r in rows])) for name in
               ["best_tanimoto", "mean_tanimoto", "mrr25", "logprob_mrr25", "strain_fraction",
                "input_precision", "input_recall", *FIXED_METRICS]}}


def paired(rows, baseline, mask):
    assert [(r["key"], r["spectrum_id"]) for r in rows] == [
        (r["key"], r["spectrum_id"]) for r in baseline]
    result = {}
    rng = np.random.default_rng(20261007)
    for metric in ["best_tanimoto", "mean_tanimoto", "pool_hit", "mrr25", *FIXED_METRICS]:
        difference = np.array([r[metric] - b[metric] for r, b in zip(rows, baseline)])[mask]
        sampled = difference[rng.integers(0, len(difference), size=(2000, len(difference)))].mean(axis=1)
        result[metric] = {"difference": float(difference.mean()),
                          "paired_ci95": np.quantile(sampled, [0.025, 0.975]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--upgrade-verified-cache", action="store_true",
                        help="Reuse unversioned base metrics only after independent verification; recompute new metrics")
    args = parser.parse_args()
    roster = json.loads((args.out / "roster.json").read_text())
    trained, seen = seen_keys(args.data)
    scored, summaries = {}, {}
    unseen_mask = np.array([r["key"] not in seen for r in roster])
    base_hash = hashlib.sha256("".join(inspect.getsource(fn) for fn in
                              [score, mol_of, fingerprint, scorer_key, rank_of]).encode()).hexdigest()
    fixed_hash = hashlib.sha256(inspect.getsource(fixed_metrics).encode()).hexdigest()
    for path in sorted(args.out.glob("*_predictions.jsonl")):
        name = path.name.removesuffix("_predictions.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [(r["key"], r["spectrum_id"]) for r in rows] == [
            (r["key"], r["spectrum_id"]) for r in roster], f"{name}: roster mismatch"
        assert len({r["key"] for r in rows}) == 300
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cache = args.out / f"{name}_metrics.json"
        existing = json.loads(cache.read_text()) if cache.exists() else None
        verified_legacy = bool(args.upgrade_verified_cache and existing and "base_score_sha256" not in existing)
        reuse_base = bool(existing and existing["predictions_sha256"] == digest and
                          (existing.get("base_score_sha256") == base_hash or verified_legacy))
        if reuse_base:
            metrics = existing["rows"]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                metrics = list(pool.map(score, rows, chunksize=4))
        if not reuse_base or existing.get("fixed_metrics_sha256") != fixed_hash:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for metric, extra in zip(metrics, pool.map(fixed_metrics, rows, chunksize=4)):
                    metric.update(extra)
        cache.write_text(json.dumps({"predictions_sha256": digest, "base_score_sha256": base_hash,
                                    "fixed_metrics_sha256": fixed_hash,
                                    "legacy_base_independently_verified": verified_legacy or bool(existing and existing.get("legacy_base_independently_verified")),
                                    "rows": metrics}, indent=2))
        scored[name] = metrics
        report = json.loads((args.out / f"{name}_report.json").read_text())["evaluation"]
        summaries[name] = {"all": aggregate(metrics),
                           "mist_unseen": aggregate([r for r in metrics if r["key"] not in seen]),
                           "mist_seen": aggregate([r for r in metrics if r["key"] in seen]),
                           "nll_per_molecule": report["teacher_forced_nll"]["per_molecule"],
                           "formula_source": report["formula_source"],
                           "true_formula_selected": sum(any(isinstance(f, dict) and f["formula"] == r["formula_search"]["true_formula"]
                                                            for f in r["formula_search"]["formulas"]) for r in rows) if report["formula_source"] == "mass" else 300,
                           "beam_stats": report["beam_stats_totals"],
                           "acceptance": report["acceptance_accounting"],
                           "true_formula_finished": report["true_formula_sampled"]}
        print(name, json.dumps(summaries[name]["all"]), flush=True)
    for control, label in [("B_removed", "removed"), ("C_mist", "mist"),
                           ("F_shuffled", "shuffled"), ("I_binary01_shuffled", "binary_shuffled")]:
        if control in scored:
            for name, rows in scored.items():
                if summaries[name]["formula_source"] != summaries[control]["formula_source"]:
                    continue
                summaries[name][f"paired_against_{label}"] = paired(rows, scored[control], np.ones(300, dtype=bool))
                summaries[name][f"paired_against_{label}_mist_unseen"] = paired(rows, scored[control], unseen_mask)
    target_baseline = None
    if scored:
        path = sorted(args.out.glob("*_predictions.jsonl"))[0]
        targets = [json.loads(line) for line in path.read_text().splitlines()]
        target_flags = [score({**row, "candidates": [{**row["target"], "best_log_prob": 0.0}]})["strain_flagged"]
                        for row in targets]
        target_baseline = {"queries": len(targets), "strain_flag_fraction": float(np.mean(target_flags)),
                           "ring3_fraction": summaries[next(iter(scored))]["all"]["target_3ring_fraction"],
                           "ring4_fraction": summaries[next(iter(scored))]["all"]["target_4ring_fraction"]}
    result = {"denominator": "independent export roster, 300 distinct molecules, one spectrum each; empty lists included",
              "mist_real_training_overlap": sum(r["key"] in trained for r in roster),
              "mist_train_validation_augmentation_overlap": sum(r["key"] in seen for r in roster),
              "target_baseline": target_baseline,
              "limitations": ["Stored MIST predictions use true formulas", "MassSpecGym precursor masses nearly exact",
                              "One checkpoint and one spectrum per molecule", "Strain flag is a heuristic"],
              "metric_notes": {
                  "whole_list_similarity": "reach over varying returned-list lengths; not a fixed-budget comparison",
                  "top25_similarity": "first 25 candidate positions; missing/invalid positions padded with zero; mean denominator is always 25",
                  "signal_controls": "C_mist versus F_shuffled, D_binary01 versus I_binary01_shuffled; removed fingerprint is a baseline, not the specificity control",
                  "pairing": "pairs with different formula sources omitted; intervals exploratory and unadjusted for multiple comparisons",
                  "validity": "RDKit sanitization and connectivity do not establish chemical plausibility",
              },
              "arms": summaries}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
