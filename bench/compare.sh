#!/usr/bin/env bash
# Alternate this crate and PyTorch on the same model, and report the best step each
# achieved.
#
# Alternating matters. This is an integrated GPU that shares its memory bandwidth
# with the rest of the machine, so running one implementation to completion and then
# the other measures the machine's mood as much as the code. Interleaving gives both
# the same distribution of interference, and taking the best step of each run reports
# what the hardware can do rather than what it happened to be doing.
#
# The reference needs a PyTorch built for your ROCm. It is a large download and it
# does not belong in this repository, so point `PYTHON` at a virtualenv holding one:
#
#   python -m venv .torch
#   .torch/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm7.1
#   PYTHON=.torch/bin/python bench/compare.sh 4
#
# Other knobs:
#
#   SEQ=1024 BATCH=2 bench/compare.sh 3    # any shape bench_train takes
#   TORCH_ARGS=--compile bench/compare.sh 3
set -uo pipefail

ROUNDS=${1:-3}
PYTHON=${PYTHON:-.torch/bin/python}
RUST_BIN=${RUST_BIN:-./target/release/examples/bench_train}
export ITERS=${ITERS:-15}

if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "no PyTorch at '$PYTHON' — set PYTHON to a virtualenv that has one" >&2
    exit 1
fi
if [ ! -x "$RUST_BIN" ]; then
    echo "no benchmark at '$RUST_BIN' — cargo build --release \\" >&2
    echo "    --no-default-features --features hip --example bench_train" >&2
    exit 1
fi

best_of() { sort -g | head -1; }

rust_runs=()
torch_runs=()

for round in $(seq 1 "$ROUNDS"); do
    echo "round $round/$ROUNDS" >&2
    r=$("$RUST_BIN" 2>/dev/null | awk '/^tokens\/s/ {print $3}')
    t=$($PYTHON bench/torch_mamba3.py ${TORCH_ARGS:-} 2>/dev/null | awk '/^tokens\/s/ {print $3}')
    echo "  mamba3 $r tok/s   pytorch $t tok/s" >&2
    rust_runs+=("$r")
    torch_runs+=("$t")
done

rust_best=$(printf '%s\n' "${rust_runs[@]}" | sort -gr | head -1)
torch_best=$(printf '%s\n' "${torch_runs[@]}" | sort -gr | head -1)

echo
echo "best over $ROUNDS rounds of $ITERS steps"
echo "  mamba3   ${rust_best} tokens/s"
echo "  pytorch  ${torch_best} tokens/s"
awk -v a="$rust_best" -v b="$torch_best" 'BEGIN { printf "  speedup  %.2fx\n", a / b }'
