#!/usr/bin/env bash
# M0.3: one command for the gate numbers.
#
# Builds the vulkan profiler and runs it with MAMBA3_ENTITY_STEPS=8 at batch
# 8, 32 and 128 (fused path), appending one Markdown row per batch to
# bench/results/entity_step.md with: date, commit, batch, ms/step,
# launches/step, reads/step and the top-5 tally labels with their counts.
#
# Refuses to run with a dirty tree: numbers must belong to a commit.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
  echo "refusing: git tree is dirty; commit first so the numbers belong to a commit" >&2
  exit 1
fi

COMMIT="$(git rev-parse --short HEAD)"
DATE="$(date -u +%F)"
OUT="bench/results/entity_step.md"
mkdir -p bench/results
if [ ! -f "$OUT" ]; then
  printf '| date | commit | batch | ms/step | launches/step | reads/step | top labels |\n' > "$OUT"
  printf '|---|---|---|---|---|---|---|\n' >> "$OUT"
fi

cargo build --release --features vulkan --example profile_entity_model

for BATCH in 8 32 128; do
  LOG="$(mktemp)"
  MAMBA3_ENTITY_BATCH="$BATCH" MAMBA3_ENTITY_STEPS=8 \
    ./target/release/examples/profile_entity_model > "$LOG" 2>&1
  # The fused optimizer-stage block: lines after "== fused=true" up to the
  # "optimizer step" row, plus the following tally rows.
  ROW="$(awk '/== fused=true/,0' "$LOG" | awk '/optimizer step/{ms=$NF; launches=$(NF-4); reads=$(NF-2)} END{print ms, launches, reads}')"
  LABELS="$(awk '/== fused=true/,0' "$LOG" | grep '\[label\]' | sort -rn | head -5 | awk '{printf "%s:%s ", $3, $2}' )"
  # shellcheck disable=SC2086
  set -- $ROW
  printf '| %s | %s | %s | %s ms | %s | %s | %s |\n' \
    "$DATE" "$COMMIT" "$BATCH" "${1:-?}" "${2:-?}" "${3:-?}" "${LABELS:-?}" >> "$OUT"
  rm -f "$LOG"
done
echo "appended 3 rows to $OUT"
