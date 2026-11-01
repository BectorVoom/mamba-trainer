#!/usr/bin/env python3
"""Test log loss of a constant class-share forecast, on the same labelled test samples
as a run's models (the prior every model has to beat).

Two priors per fold: the class shares of that fold's calibration year (uses only data
before the test year) and the shares of the whole labelled pool (`class_counts` in
run.json; includes later years, so it is slightly optimistic).

    python3 tools/prior_logloss.py results/runs/main C_mamba_mlp
"""
import glob
import json
import sys

import numpy as np
import pyarrow.parquet as pq

run, model = sys.argv[1:3]
pool = np.array(json.load(open(f"{run}/run.json"))["class_counts"], float)
pool /= pool.sum()
cal_ll, pool_ll = [], []
for d in sorted(glob.glob(f"{run}/fold_*/")):
    te = pq.read_table(f"{d}{model}/oof_test.parquet", columns=["label"]).to_pandas()["label"].to_numpy()
    ca = pq.read_table(f"{d}{model}/oof_cal.parquet", columns=["label"]).to_pandas()["label"].to_numpy()
    te, ca = te[te >= 0], ca[ca >= 0]
    shares = np.bincount(ca, minlength=3) / len(ca)
    cal_ll += list(-np.log(shares[te]))
    pool_ll += list(-np.log(pool[te]))
print(f"{len(cal_ll)} labelled test samples")
print(f"prior from each fold's calibration year: test log loss {np.mean(cal_ll):.4f}")
print(f"prior from the whole labelled pool:      test log loss {np.mean(pool_ll):.4f}")
