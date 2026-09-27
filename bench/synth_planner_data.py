"""Synthetic task-planner data with the schema of Kaggriculture's `train.npz`, for timing the planner step where the
real data is not available (a Colab VM): same keys, shapes, dtypes, value ranges and label couplings, random content.

Schema and rates measured on `runs/20260926_task_planner/data/train.npz` (first 20,000 turns):
  tiles [S,100,48] f16 in [0, 2.7] (39 of 48 columns binary); glob [S,114] f32 in [-1, 1.4]; units [S,20,36] f16
  upos [S,20] i16: -1 (absent unit, 48%) or the anchor tile 0..99
  tgt [S,20,3] i16: -100 for absent units; else a tile 0..99 or NONE = 100 (8% / 23% / 38% at steps 1-3, and a plan
      that reached NONE stays NONE)
  op [S,20,3] i16: 0..12 where the target is a tile, else -100; crop: 0..4 on 17% of tile targets, else -100
  opset [S,20,3,13] u8: 14% ones where the target is a tile, else 0; eta [S,20] i16: -1 where the first target is not a
      tile, else 0..15; day / hour / traj: i8 / i8 / i32
Timing does not depend on the values; the couplings keep every loss term populated as in the real data.

    python bench/synth_planner_data.py OUT_DIR [--turns 5120] [--seed 0]    # writes OUT_DIR/train.npz and dev16.npz
"""
import argparse
from pathlib import Path

import numpy as np

NT, NU, K, OPS, CROPS = 100, 20, 3, 13, 5
NONE_RATE = (0.078, 0.232, 0.382)


def synth(n, rng):
    tiles = rng.random((n, NT, 48), dtype=np.float32)
    tiles[:, :, :39] = tiles[:, :, :39] < 0.3                      # binary flag columns
    tiles[:, :, 39:] *= 2.6
    glob = rng.uniform(-1.0, 1.36, (n, 114)).astype(np.float32)
    units = (rng.random((n, NU, 36), dtype=np.float32) * 1.1).astype(np.float16)
    present = rng.random((n, NU)) < 0.524
    upos = np.where(present, rng.integers(0, NT, (n, NU)), -1).astype(np.int16)

    tgt = rng.integers(0, NT, (n, NU, K)).astype(np.int16)
    done = np.zeros((n, NU), bool)
    for s in range(K):
        done |= rng.random((n, NU)) < NONE_RATE[s] - (NONE_RATE[s - 1] if s else 0.0)
        tgt[:, :, s][done] = NT                                   # NONE, and it stays NONE
    tgt[~present] = -100
    has = (tgt >= 0) & (tgt < NT)
    op = np.where(has, rng.integers(0, OPS, tgt.shape), -100).astype(np.int16)
    crop = np.where(has & (rng.random(tgt.shape) < 0.173), rng.integers(0, CROPS, tgt.shape), -100).astype(np.int16)
    opset = ((rng.random(tgt.shape + (OPS,)) < 0.141) & has[..., None]).astype(np.uint8)
    eta = np.where(has[:, :, 0], rng.integers(0, 16, (n, NU)), -1).astype(np.int16)
    return dict(tiles=tiles.astype(np.float16), glob=glob, units=units, upos=upos, tgt=tgt, op=op, crop=crop,
                opset=opset, eta=eta, day=rng.integers(0, 30, n).astype(np.int8),
                hour=rng.integers(0, 24, n).astype(np.int8), traj=rng.integers(0, 75, n).astype(np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--turns", type=int, default=5120); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    np.savez(out / "train.npz", **synth(a.turns, rng))
    np.savez(out / "dev16.npz", **synth(max(512, a.turns // 10), rng))   # train_entity.py scores a dev set per epoch
    print(f"wrote {out}/train.npz ({a.turns} turns) and dev16.npz")


if __name__ == "__main__":
    main()
