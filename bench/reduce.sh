#!/usr/bin/env bash
# K9: time `sum_dim` on the model's shapes with the split on and off.
#
# Prints µs per call (200 calls after warm-up) for the default thresholds,
# `MAMBA3_REDUCE_SPLIT=0` (never split) and `MAMBA3_REDUCE_SPLIT=1` (the
# pre-K9 decision). `FEATURES` selects the backend (default `cpu`).
set -euo pipefail
cd "$(dirname "$0")/.."

FEATURES="${FEATURES:-cpu}"
cargo build --release --features "$FEATURES" --example bench_reduce
BIN="./target/release/examples/bench_reduce"

echo "--- default (K9 thresholds) ---"
"$BIN"
echo "--- MAMBA3_REDUCE_SPLIT=0 (never split) ---"
MAMBA3_REDUCE_SPLIT=0 "$BIN"
echo "--- MAMBA3_REDUCE_SPLIT=1 (legacy split) ---"
MAMBA3_REDUCE_SPLIT=1 "$BIN"
