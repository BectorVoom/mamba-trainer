#!/usr/bin/env python3
"""Compare two runs' out-of-fold test predictions for one model, prediction by prediction.

Used to check that the same config and seed trained on two backends (e.g. wgpu locally
and CUDA on Kaggle) gives the same model:

    python3 tools/compare_oof.py results/runs/main results/kaggle/mclm-head/out/run C_mamba_mlp
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

a_dir, b_dir, model = sys.argv[1:4]
cols = ["s", "t", "p_buy", "p_sell", "p_hold", "cal_buy", "cal_sell", "cal_hold"]
worst = {c: 0.0 for c in cols[2:]}
n = 0
same_action = 0
for fa in sorted(glob.glob(f"{a_dir}/fold_*/{model}/oof_test.parquet")):
    fold = fa.split("/")[-3]
    fb = f"{b_dir}/{fold}/{model}/oof_test.parquet"
    if not os.path.exists(fb):
        sys.exit(f"{fold}: missing in {b_dir}")
    a = pq.read_table(fa, columns=cols).to_pandas()
    b = pq.read_table(fb, columns=cols).to_pandas()
    if not (np.array_equal(a.s, b.s) and np.array_equal(a.t, b.t)):
        sys.exit(f"{fold}: different sample keys")
    for c in worst:
        worst[c] = max(worst[c], float(np.abs(a[c] - b[c]).max()))
    ea = a.cal_buy - a.cal_sell
    eb = b.cal_buy - b.cal_sell
    same_action += int(((ea > 0) == (eb > 0)).sum())
    n += len(a)
print(f"{model}: {n} test predictions")
for c, v in worst.items():
    print(f"  max |diff| {c}: {v:.2e}")
print(f"  same primary-policy sign of edge: {same_action}/{n}")
