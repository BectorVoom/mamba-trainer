# Mamba-CLM: plan and architecture decisions

Goal: find out, with leakage-free walk-forward CV, whether contrastive state-action
learning on a Mamba encoder ("Mamba + CLM") trades better out of sample than the same
encoder with a supervised head ("Mamba + MLP") and than LightGBM. Negative results are
results. The exact rules every part implements are in [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Layout

```
finance/                     this module (own Cargo workspace; depends on ../ = mamba3)
├── Cargo.toml               workspace: crates/*
├── crates/
│   ├── clm-data/            bars (Parquet), features (C2), labels (C3), normalisation, windows, synthetic panels
│   ├── clm-eval/            folds + purging (C4), metrics, calibration, backtest (C5), statistics (C6)
│   ├── clm-model/           Mamba-1 encoder, state/action encoders, heads, losses, baselines, trainer, CPU reference
│   └── clm-cli/             `mamba-clm`: build-dataset, train, infer, cv, bench, report
├── lean/ClmVerify/          Lean 4 proofs + `clmcheck` executable (differential oracle for Rust tests)
├── configs/                 experiment TOML
├── tools/                   Python, outside the production path: data download, LightGBM baseline, numeric checks
├── data/                    downloaded bars (gitignored)
├── results/                 run outputs (Parquet/JSON); RESULTS.md
└── docs/CONTRACTS.md
```

Kernels live in the core crate, beside the existing scans, because the autograd tape
records custom ops only from inside it:
`src/tensor/ops/selective_scan.rs` (CubeCL forward: sequential and chunked; backward),
`src/ssm/selective.rs` (sequential f64 CPU reference), `Var::selective_scan` (autograd op).

## Architecture decisions

| # | Decision | Why |
|---|----------|-----|
| AD1 | CubeCL **0.10.0**, not 0.11 | Kept by the user on 2026-10-17: 0.11 fuses `a*b+c` into FMA ignoring settings, folds IEEE-invalid expressions, and mis-lowers signed vector shifts on CPU — all break CPU/GPU agreement. The 0.11 port is stashed. |
| AD2 | Gradients: the crate's own tape autograd (`mamba3::autograd`) plus hand-written backward kernels for the selective scan | Already used by every model here; custom ops register a backward closure. No CPU fallback in the GPU path. |
| AD3 | GPU backends: CUDA (primary, `--features cuda`), wgpu on this PC's Radeon, CubeCL CPU runtime for tests; f32 first | CUDA via Kaggle/Colab; local GPU is AMD (wgpu works, vulkan/SPIR-V crashes RADV). |
| AD4 | Layouts: `[B, T, D]` row-major activations; scan state `[B, D, N]`; `A [D, N]` | Matches the existing kernels and `Linear` (last axis is features). |
| AD5 | Labels are position-independent target exposures (BUY = be long, SELL = be flat, HOLD = edge below cost) | The plan forbids BUY/SELL labels that ignore position; with target exposures the label does not depend on the policy's own past. The policy (C5) restores position dependence. |
| AD6 | Execution at the next open | Bars are known after the close; trading at that close would use information not yet available. |
| AD7 | Data: Yahoo daily bars of a fixed list of large US stocks + SPY + sector SPDRs, 2000–present | Free and point-in-time for prices. Survivorship bias (today's large caps) and static sector map are documented limitations. |
| AD8 | LightGBM via Python (`tools/lgbm_baseline.py`) on features exported by the Rust pipeline, with the Rust folds; predictions come back as Parquet and go through the same Rust metrics/backtest | It is a baseline, not the production path; identical features, folds and accounting. |
| AD9 | Lean proves the algorithms (scan algebra, folds/purging, label windows, backtest accounting) on its own model; Rust is *tested* against the Lean executable, not proven | Same arrangement as `lean/MotifDecoder`. |

## Phases (priority: correctness → leakage-free CV → baselines → model improvements → speed)

1. Scaffold workspace, contracts, data download. ✔ when `cargo check` passes and bars exist.
2. Selective scan: CPU reference; CubeCL forward (sequential, chunked); backward; autograd op; CPU/GPU equivalence + gradient checks; causality test.
3. clm-data + clm-eval: features, labels, folds, purging, normalisation, metrics, calibration, backtest, bootstrap. Leakage-injection tests.
4. Lean: proofs + `clmcheck`; Rust differential tests against it.
5. clm-model: Mamba-1 block/encoder, state projection, action encoder (cached), contrastive scoring, InfoNCE, SupCon, aux heads; baselines (linear, MLP, TCN, Mamba+MLP); gradient checks; synthetic learnability test; overfit test.
6. Trainer: AdamW, clipping, cosine schedule, early stopping on val, checkpoints, seeds, NaN guard, logging.
7. CV runner on real data; LightGBM; calibration; backtest; statistics; ablations A–F; RESULTS.md.
8. Profiling-driven optimisation and kernel benchmarks (T = 64 … 1024).

Reviews: codex (`codex exec -s read-only`) after phases 2, 3+4, 5+6, 7.

### Status (2026-10-22)

All eight phases are done. Main study (14 folds, 8 models) plus seed replicates and
ablations on Kaggle T4; findings in [`RESULTS.md`](RESULTS.md) (negative: CLM does
not beat Mamba+MLP; linear/LightGBM/MLP beat every Mamba variant; nothing beats
buy-and-hold after costs). Four codex reviews in `docs/reviews/`, all findings fixed
or documented. Not built: quantile heads, Python bindings, within-batch padding masks.
