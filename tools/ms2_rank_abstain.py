"""Cheap ranker with calibrated abstention (plan: ranking before Mamba-3).

Fits a linear score on cheap exact features and abstains when the top-1
margin is thin or the pool is large. Features use only pre-retrieval
evidence plus database-presence counts; the true structure is NEVER a
feature (no oracle leakage). Train/test splits are file-order halves.

Run:   python3 tools/ms2_rank_abstain.py --corpus chebi --max-queries 100
       python3 tools/ms2_rank_abstain.py --corpus msgym --max-queries 300
Test:  python3 -m unittest tools.ms2_rank_abstain
"""
from __future__ import annotations

import csv
import gzip
import math
import os
import sys
import time
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ms2_database_retrieval import (
    AROM_ORDER,
    DatabaseIndex,
    WorkCounters,
    _contains_with_prep,
    _joint_with_prep,
    _prepare_joint,
    _prepare_single,
    count_embeddings_prep,
    edge_fingerprint,
    fingerprint_screen,
    formula_key,
    mass_verdict,
    pattern_fingerprint,
    prepare_target,
    BudgetExhausted,
)
from tools.ms2_chebi_corpus import (
    first_prop,
    iter_sdf_records,
    standardize_chebi_record,
)
from tools.ms2_msgym_corpus import (
    extract_patterns,
    index_val_by_smiles,
    iter_json_entries,
    parse_formula,
    standardize_smiles,
)

EMBED_CAP = 25
NODE_BUDGET = 100_000


def stage_pools(index, target, patterns, counters):
    """Reproduce run_query stages 1-3 while capturing per-candidate data."""
    observed = target["mass"]
    s1 = [r for r in index.records
          if mass_verdict(observed, r["formula"]) != "reject"]
    tkey = formula_key(target["formula"])
    s2 = [r for r in s1 if index.fkeys.get(r["id"]) == tkey
          and r["charge"] == target["charge"]]
    pat_fps = [pattern_fingerprint(pt, pe) for pt, pe in patterns]
    singles = [((pt, pe) + _prepare_single(pt, pe)) for pt, pe in patterns]
    joint = _prepare_joint(patterns) if len(patterns) > 1 else None
    from collections import Counter as _C

    s3 = []
    truncated = False
    try:
        for r in s2:
            if not all(fingerprint_screen(index.fps[r["id"]], pf) for pf in pat_fps):
                continue
            at, ed = r["atom_types"], r["edges"]
            if any(len(pt) > len(at) for pt, _, _, _ in singles):
                continue
            nbr, pos = prepare_target(at, ed)
            occupied = [False] * len(at)
            if not all(_contains_with_prep(nbr, pos, pt, order, cons, occupied, counters)
                       for pt, _, order, cons in singles):
                continue
            if joint is not None:
                ft, fp, fa, dg = joint
                if not _joint_with_prep(nbr, pos, _C(at), ft, fp, fa, dg,
                                        len(patterns),
                                        [set() for _ in patterns], counters):
                    continue
            embeds = [count_embeddings_prep(nbr, pos, pt, order, cons, EMBED_CAP)
                      for pt, _, order, cons in singles]
            s3.append((r, embeds))
    except BudgetExhausted:
        truncated = True
    return s1, s2, s3, truncated


def chebi_table(sdf_path, max_records, max_queries):
    """Rows: (qid, cand_id, is_target, residual_ppm, log_syn, log_xref,
    embed0, pool_n). Pool = whole-index formula matches (mass_only arm)."""
    opener = gzip.open if sdf_path.endswith(".gz") else open
    records = []
    with opener(sdf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for n, (mol, props) in enumerate(iter_sdf_records(handle)):
            if n >= max_records:
                break
            cid = first_prop(props, "ChEBI ID") or f"ROW{n}"
            key = first_prop(props, "INCHIKEY")[:14]
            rec, reason = standardize_chebi_record(mol, props, cid)
            if rec is not None:
                rec["_dedup"] = key
                records.append(rec)
    seen, unique = set(), []
    for r in records:
        if r["_dedup"] and r["_dedup"] in seen:
            continue
        seen.add(r["_dedup"])
        unique.append(r)
    index = DatabaseIndex(unique)
    import random
    import zlib

    rows = []
    n_truncated = 0
    for qi, target in enumerate(unique[:max_queries]):
        single, pair = extract_patterns(target["atom_types"], target["edges"])
        pats = [single, pair] if pair else [single]
        counters = WorkCounters(limit=NODE_BUDGET)
        _, _, s3, truncated = stage_pools(index, target, pats, counters)
        if truncated:
            # Incomplete pools would corrupt calibration (partial size and
            # margins presented as exact); excluded and counted instead.
            n_truncated += 1
            continue
        # Seeded shuffle: tie-breaking must not see ambient order.
        positions = list(range(len(s3)))
        random.Random(zlib.crc32(target["id"].encode())).shuffle(positions)
        pos_of = {idx: p for p, idx in enumerate(positions)}
        for idx, (r, embeds) in enumerate(s3):
            ppm = abs(r["mass"] - target["mass"]) / max(target["mass"], 1) * 1e6
            rows.append({
                "qid": target["id"],
                "cand": r["id"],
                "pos": pos_of[idx],
                "is_target": r["id"] == target["id"],
                "residual_ppm": ppm,
                "log_syn": math.log1p(r["original"].get("n_synonyms", 0)),
                "log_xref": math.log1p(r["original"].get("n_xrefs", 0)),
                "embed0": embeds[0] if embeds else 0,
                "pool_n": len(s3),
            })
    return rows, {"n_truncated_queries": n_truncated}


def msgym_table(prefix_path, tsv_path, max_queries):
    """Same shape; no synonym columns (log_syn/log_xref = 0)."""
    val_groups, _ = index_val_by_smiles(tsv_path)
    entries = [(q, c) for q, c in iter_json_entries(prefix_path, max_queries * 20)]
    joined = [(q, c) for q, c in entries if q in val_groups][:max_queries]
    rows = []
    n_truncated = 0
    for qi, (qsmiles, cands) in enumerate(joined):
        row = val_groups[qsmiles][0]
        target, reason = standardize_smiles(
            qsmiles, "msgym-val", f"Q{qi}", parse_formula(row["formula"]))
        if target is None:
            continue
        pool = []
        for ci, s in enumerate(cands):
            rec, _ = standardize_smiles(s, "msgym-pool", f"Q{qi}-C{ci}")
            if rec is not None:
                pool.append(rec)
        sub = DatabaseIndex(pool)
        single, pair = extract_patterns(target["atom_types"], target["edges"])
        pats = [single, pair] if pair else [single]
        counters = WorkCounters(limit=NODE_BUDGET)
        _, _, s3, truncated = stage_pools(sub, target, pats, counters)
        if truncated:
            n_truncated += 1
            continue
        import random
        import zlib

        positions = list(range(len(s3)))
        random.Random(zlib.crc32(f"Q{qi}".encode())).shuffle(positions)
        pos_of = {idx: p for p, idx in enumerate(positions)}
        for idx, (r, embeds) in enumerate(s3):
            ppm = abs(r["mass"] - target["mass"]) / max(target["mass"], 1) * 1e6
            rows.append({
                "qid": f"Q{qi}",
                "cand": r["id"],
                "pos": pos_of[idx],
                "is_target": r["original"].get("smiles") == qsmiles,
                "residual_ppm": ppm,
                "log_syn": 0.0,
                "log_xref": 0.0,
                "embed0": embeds[0] if embeds else 0,
                "pool_n": len(s3),
            })
    return rows, {"n_truncated_queries": n_truncated}


FEATURES = ("neg_residual", "log_syn", "log_xref", "neg_embed0")


def featurize(row):
    return {"neg_residual": -row["residual_ppm"],
            "log_syn": row["log_syn"],
            "log_xref": row["log_xref"],
            "neg_embed0": -row["embed0"]}


def split_queries(rows):
    qids = sorted(set(r["qid"] for r in rows))
    cut = len(qids) // 2
    return set(qids[:cut]), set(qids[cut:])


def standardize_params(rows, weights_hint=None):
    stats = {}
    for f in FEATURES:
        vals = [featurize(r)[f] for r in rows]
        mu = sum(vals) / max(len(vals), 1)
        sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)) ** 0.5 or 1.0
        stats[f] = (mu, sd)
    return stats


def score_row(row, weights, stats):
    f = featurize(row)
    return sum(weights[k] * (f[k] - stats[k][0]) / stats[k][1] for k in FEATURES)


def rank_pool(cands, weights, stats):
    # Ties break by preassigned shuffled position ("pos"), never by
    # ambient file or supplier order: without this, index order (ChEBI IDs)
    # or self-first supplied lists (MassSpecGym) leak into "top-1".
    return sorted(cands,
                  key=lambda r: (-score_row(r, weights, stats), r["pos"]))


def topk_hit(ranked, k):
    return any(r["is_target"] for r in ranked[:k])


def fit_weights(train_rows, grid=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)):
    """Coordinate ascent on train top-1 (deterministic, tiny grid)."""
    stats = standardize_params(train_rows)
    by_q = {}
    for r in train_rows:
        by_q.setdefault(r["qid"], []).append(r)
    pools = list(by_q.values())
    weights = {f: 0.0 for f in FEATURES}
    weights["neg_residual"] = 1.0

    def top1(w):
        return sum(1 for p in pools if rank_pool(p, w, stats)[0]["is_target"])

    best = top1(weights)
    improved = True
    while improved:
        improved = False
        for f in FEATURES:
            for g in grid:
                trial = dict(weights)
                trial[f] = g
                s = top1(trial)
                if s > best:
                    best, weights, improved = s, trial, True
    return weights, stats, best, len(pools)


def abstention_report(rows, weights, stats, tau, kmax):
    """Predict top-1 unless margin < tau or pool > kmax. Returns metrics."""
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], []).append(r)
    pred, good, top1_all, top1_pred = 0, 0, 0, 0
    for q, pool in by_q.items():
        ranked = rank_pool(pool, weights, stats)
        if topk_hit(ranked, 1):
            top1_all += 1
        margin = (score_row(ranked[0], weights, stats)
                  - (score_row(ranked[1], weights, stats) if len(ranked) > 1 else -1e18))
        if margin >= tau and len(pool) <= kmax:
            pred += 1
            if ranked[0]["is_target"]:
                good += 1
                top1_pred += 1
    n = len(by_q)
    return {"n": n, "predicted": pred,
            "coverage": pred / n if n else 0.0,
            "precision": good / pred if pred else None,
            "top1_all": top1_all / n if n else 0.0,
            "top1_predicted": top1_pred / pred if pred else None}


def fit_abstention(train_rows, weights, stats, target_precision=0.90):
    """Largest coverage with train precision >= target; deterministic.

    Falls back to a reject-all gate when nothing meets the target, so a
    failed calibration can never silently enable unrestricted prediction.
    """
    best = (0.0, 0.0, 1 << 30)  # coverage, tau, kmax
    margins = set()
    pools = {}
    for r in train_rows:
        pools.setdefault(r["qid"], []).append(r)
    for pool in pools.values():
        ranked = rank_pool(pool, weights, stats)
        if len(ranked) > 1:
            margins.add(score_row(ranked[0], weights, stats)
                        - score_row(ranked[1], weights, stats))
    taus = sorted(margins) + [float("inf")]
    kmaxs = sorted(set([1 << 30] + [len(p) for p in pools.values()]))
    for tau in taus:
        for kmax in kmaxs:
            rep = abstention_report(train_rows, weights, stats, tau, kmax)
            if rep["precision"] is not None and rep["precision"] >= target_precision:
                if rep["coverage"] > best[0]:
                    best = (rep["coverage"], tau, kmax)
    if best[0] <= 0.0:
        return {"tau": 1e18, "kmax": -1, "train_coverage": 0.0}
    return {"tau": best[1], "kmax": best[2], "train_coverage": best[0]}


def run_experiment(rows):
    train_ids, test_ids = split_queries(rows)
    train = [r for r in rows if r["qid"] in train_ids]
    test = [r for r in rows if r["qid"] in test_ids]
    weights, stats, train_top1, n_train = fit_weights(train)
    gate = fit_abstention(train, weights, stats)
    test_rep = abstention_report(test, weights, stats, gate["tau"], gate["kmax"])
    train_rep = abstention_report(train, weights, stats, gate["tau"], gate["kmax"])
    # Risk-coverage curve on test (sweep tau at fitted kmax).
    curve = []
    for tau in sorted(set([0.0] + [m for m in _margins(test, weights, stats)])):
        rep = abstention_report(test, weights, stats, tau, gate["kmax"])
        curve.append((round(tau, 4), round(rep["coverage"], 3),
                      rep["precision"] and round(rep["precision"], 3)))
    return {"weights": weights, "gate": gate, "n_train": n_train,
            "n_test": len(set(r["qid"] for r in test)),
            "train_top1": train_top1 / max(n_train, 1),
            "train_rep": train_rep, "test_rep": test_rep, "curve": curve}


def _margins(rows, weights, stats):
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], []).append(r)
    out = []
    for pool in by_q.values():
        ranked = rank_pool(pool, weights, stats)
        if len(ranked) > 1:
            out.append(score_row(ranked[0], weights, stats)
                       - score_row(ranked[1], weights, stats))
    return out


def main(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("chebi", "msgym"), required=True)
    ap.add_argument("--sdf", default="data/pinned/chebi_3_stars.sdf.gz")
    ap.add_argument("--prefix",
                    default="data/pinned/msgym_candidates_formula_prefix64.json")
    ap.add_argument("--tsv", default="data/pinned/MassSpecGym1.5.tsv")
    ap.add_argument("--max-records", type=int, default=60000)
    ap.add_argument("--max-queries", type=int, default=100)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    if args.corpus == "chebi":
        rows, table_info = chebi_table(args.sdf, args.max_records, args.max_queries)
    else:
        rows, table_info = msgym_table(args.prefix, args.tsv, args.max_queries)
    result = run_experiment(rows)
    result["n_rows"] = len(rows)
    result["table_info"] = table_info
    result["wall_s"] = time.perf_counter() - t0
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    import json

    sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")


class RankTests(unittest.TestCase):
    def _toy(self):
        # Only log_syn separates q1/q2 (in opposite pool orders); q3 is an
        # exact tie. Residual ties everywhere, so fitting must use log_syn.
        return [
            {"qid": "q1", "cand": "a", "pos": 0, "is_target": True, "residual_ppm": 0.0,
             "log_syn": 3.0, "log_xref": 1.0, "embed0": 2, "pool_n": 2},
            {"qid": "q1", "cand": "b", "pos": 1, "is_target": False, "residual_ppm": 0.0,
             "log_syn": 0.0, "log_xref": 0.0, "embed0": 2, "pool_n": 2},
            {"qid": "q2", "cand": "c", "pos": 0, "is_target": False, "residual_ppm": 0.0,
             "log_syn": 0.0, "log_xref": 0.0, "embed0": 1, "pool_n": 2},
            {"qid": "q2", "cand": "d", "pos": 1, "is_target": True, "residual_ppm": 0.0,
             "log_syn": 3.0, "log_xref": 0.0, "embed0": 1, "pool_n": 2},
            {"qid": "q3", "cand": "e", "pos": 0, "is_target": False, "residual_ppm": 0.0,
             "log_syn": 1.0, "log_xref": 0.0, "embed0": 1, "pool_n": 2},
            {"qid": "q3", "cand": "f", "pos": 1, "is_target": True, "residual_ppm": 0.0,
             "log_syn": 1.0, "log_xref": 0.0, "embed0": 1, "pool_n": 2},
        ]

    def test_fit_prefers_separating_feature(self):
        rows = self._toy()
        w, _, best, n = fit_weights(rows)
        self.assertEqual(n, 3)
        self.assertGreater(w["log_syn"], 0.0)
        self.assertEqual(best, 2)  # q1+q2; q3 tie is unrankable

    def test_abstention_gating(self):
        rows = self._toy()
        w, stats, _, _ = fit_weights(rows)
        # q3 is an exact tie (margin 0); a hair above 0 predicts q1+q2 only.
        rep = abstention_report(rows, w, stats, tau=1e-9, kmax=1 << 30)
        self.assertEqual(rep["predicted"], 2)
        self.assertEqual(rep["precision"], 1.0)
        rep0 = abstention_report(rows, w, stats, tau=0.0, kmax=1 << 30)
        self.assertEqual(rep0["predicted"], 3)
        self.assertAlmostEqual(rep0["precision"], 2 / 3)

    def test_split_deterministic(self):
        rows = self._toy()
        a = split_queries(rows)
        self.assertEqual(split_queries(rows), a)
        self.assertEqual(len(a[0]), 1)

    def test_no_oracle_features(self):
        # featurize must not read is_target/qid/cand.
        import inspect

        src = inspect.getsource(featurize)
        self.assertNotIn("is_target", src)

    def test_failed_calibration_abstains(self):
        # Review regression: with no gate meeting the precision target,
        # fitting returned predict-everything. It must reject all.
        rows = [
            {"qid": "q", "cand": "a", "pos": 0, "is_target": False,
             "residual_ppm": 0.0, "log_syn": 0.0, "log_xref": 0.0,
             "embed0": 1, "pool_n": 1},
        ]
        w = {f: 0.0 for f in FEATURES}
        stats = {f: (0.0, 1.0) for f in FEATURES}
        gate = fit_abstention(rows, w, stats, target_precision=0.9)
        rep = abstention_report(rows, w, stats, gate["tau"], gate["kmax"])
        self.assertEqual(rep["predicted"], 0)
        self.assertIsNone(rep["precision"])


if __name__ == "__main__":
    main(sys.argv[1:])
