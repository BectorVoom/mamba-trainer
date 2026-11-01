#!/usr/bin/env python3
"""LightGBM baseline on the Rust pipeline's exported folds.

Not part of the production path: it reads the per-fold sample tables written by
`mamba-clm export-lgbm` (last-row features normalised on each fold's training
days; the same purged train/val/cal splits and the same test days as every other
model), trains a multiclass LightGBM on `train`, early-stops and picks one of two
candidate settings on `val` log loss (the same two-candidate budget the neural
models get), and writes predictions for `cal` and `test`:

    <out>/fold_<year>/pred_cal.parquet   s, t, p_buy, p_sell, p_hold
    <out>/fold_<year>/pred_test.parquet
    <out>/fold_<year>/meta.json          the export's manifest (echoed for provenance),
                                         both candidates' val log loss, chosen params

The Rust CV runner then calibrates on `cal`, picks the decision threshold on `cal`
and backtests on `test` exactly as for the other models.

    python3 finance/tools/lgbm_baseline.py --export finance/results/lgbm_export --out finance/results/lgbm_pred
"""
import argparse
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CANDIDATES = [
    {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 0.8,
     "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0},
    {"learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 500, "feature_fraction": 0.8,
     "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0},
]


def load(path: pathlib.Path):
    t = pq.read_table(path).to_pandas()
    feats = [c for c in t.columns if c.startswith("f") and c[1:].isdigit()]
    feats.sort(key=lambda c: int(c[1:]))
    return t, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    root = pathlib.Path(args.export)
    out = pathlib.Path(args.out)
    for fold in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("fold_")):
        tr, feats = load(fold / "train.parquet")
        va, _ = load(fold / "val.parquet")
        ca, _ = load(fold / "cal.parquet")
        te, _ = load(fold / "test.parquet")
        for name, d in [("train", tr), ("val", va), ("cal", ca)]:
            if (d["label"] < 0).any():
                raise SystemExit(f"{fold.name}/{name}: unlabelled rows")
        dtr = lgb.Dataset(tr[feats].to_numpy(np.float32), label=tr["label"].to_numpy())
        dva = lgb.Dataset(va[feats].to_numpy(np.float32), label=va["label"].to_numpy(), reference=dtr)
        best = None
        losses = []
        for i, cand in enumerate(CANDIDATES):
            params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbose": -1,
                      "seed": args.seed + i, "num_threads": args.threads, "deterministic": True, **cand}
            booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            score = booster.best_score["valid_0"]["multi_logloss"]
            losses.append(score)
            if best is None or score < best[0]:
                best = (score, i, booster)
        score, idx, booster = best
        dest = out / fold.name
        dest.mkdir(parents=True, exist_ok=True)
        for name, d in [("cal", ca), ("test", te)]:
            p = booster.predict(d[feats].to_numpy(np.float32), num_iteration=booster.best_iteration)
            table = pa.table({
                "s": pa.array(d["s"].to_numpy(), pa.uint32()),
                "t": pa.array(d["t"].to_numpy(), pa.uint32()),
                "p_buy": p[:, 0], "p_sell": p[:, 1], "p_hold": p[:, 2],
            })
            pq.write_table(table, dest / f"pred_{name}.parquet")
        manifest = json.loads((fold / "manifest.json").read_text())
        (dest / "meta.json").write_text(json.dumps({
            "manifest": manifest, "candidate_val_nll": losses,
            "val_nll": score, "candidate": idx, "params": CANDIDATES[idx],
            "best_iteration": booster.best_iteration, "lightgbm": lgb.__version__,
            "n_train": len(tr), "n_val": len(va), "n_cal": len(ca), "n_test": len(te),
        }, indent=2))
        print(f"{fold.name}: candidate {idx} val logloss {score:.4f} iters {booster.best_iteration}")


if __name__ == "__main__":
    main()
