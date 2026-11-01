# Mamba-CLM: Mamba encoder + contrastive decision learning for daily equities

A Rust/CubeCL study of whether contrastive state–action learning ("CLM") on a
Mamba-1 time-series encoder trades better out of sample than the same encoder with
a supervised head, a TCN, an MLP, a linear model and LightGBM — under leakage-free
walk-forward cross-validation. Results: [`RESULTS.md`](RESULTS.md). Plan and
architecture decisions: [`PLAN.md`](PLAN.md). Exact rules (features, labels, folds,
purging, policy, accounting, scan maths): [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Layout

| Path | What |
|---|---|
| `crates/clm-data` | Bars (Parquet), the 20 point-in-time features, labels, per-fold normalisation, windows, look-ahead audits, synthetic panels, LightGBM export |
| `crates/clm-eval` | Walk-forward folds, purging/embargo + run-time leakage guard, metrics, calibration (temperature / vector / isotonic), backtest, stationary bootstrap, out-of-fold Parquet |
| `crates/clm-model` | Mamba-1 encoder, TCN, contrastive state/action head, MLP/linear heads, auxiliary heads, losses, trainer, `f64` CPU reference |
| `crates/clm-cli` | `mamba-clm`: `cv`, `train`, `report`, `infer`, `export-lgbm`, `folds`, `bench` |
| `lean/ClmVerify` | Lean 4 proofs of the algorithms (scan algebra, chunked scan, causality, adjoint; folds/purging; label windows; backtest accounting) and the `clmcheck` oracle the Rust tests compare against |
| `tools/` | Python outside the production path: `fetch_yahoo.py` (data), `lgbm_baseline.py` (LightGBM baseline), `prior_logloss.py` (class-prior baseline), `compare_oof.py` (prediction-level comparison of two runs, e.g. CUDA vs wgpu) |
| `configs/` | Experiment configs (`main.toml` is the study) |
| `docs/reviews/` | Codex review reports |

The selective-scan kernels live in the core crate, next to the other scans:
`src/tensor/ops/selective_scan.rs` (CubeCL forward — sequential and chunked
parallel associative scan — and backward), `src/ssm/selective.rs` (`f64`
reference), `Var::selective_scan` in `src/autograd/ops.rs`, tests in
`tests/selective_scan.rs`.

## Running

```bash
# data (Yahoo daily bars; survivorship-biased universe, see RESULTS.md)
python3 finance/tools/fetch_yahoo.py --out finance/data/bars

cd finance
cargo build --release -p clm-cli --no-default-features --features wgpu   # or cuda / cpu
B=target/release/mamba-clm

$B folds  --config configs/main.toml                          # print the folds
$B cv     --config configs/main.toml --out results/runs/main  # resumable; refuses a changed config/data/source
$B export-lgbm --config configs/main.toml --out results/lgbm_export
python3 tools/lgbm_baseline.py --export results/lgbm_export --out results/lgbm_pred
$B cv     --config configs/main.toml --out results/runs/main --models lgbm
$B report --config configs/main.toml --run results/runs/main  # report.md / report.json
$B infer  --config configs/main.toml --run results/runs/main --model D_mamba_clm --date 2025-06-02
$B bench  --out results/bench                                 # scan kernels, T = 64..1024
```

Run from `finance/` (its `cubecl.toml` turns CubeCL's source-blind kernel cache off).

## Tests

```bash
# core crate: scan kernels vs the f64 reference (CPU runtime; add --no-default-features --features wgpu|cuda for GPU)
cargo test --release --test selective_scan && cargo test --release --lib ssm::selective
# study crates
cd finance
cargo test --release -p clm-data -p clm-eval
cargo test --release -p clm-eval -- --include-ignored      # + Lean oracle (needs lean/ClmVerify built)
cargo test --release -p clm-model --no-default-features --features wgpu --test model
# Lean
cd lean/ClmVerify && lake build && .lake/build/bin/clmcheck < test/cases.txt | diff - test/expected.txt
```

There are no Python bindings for this module (the production path is Rust only),
so the repository's Python/Rust parity check does not apply to it.
