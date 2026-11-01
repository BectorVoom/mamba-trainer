# Results: does contrastive decision learning help a Mamba encoder trade?

**Short answer: no evidence that it does.** The study ran 14 walk-forward test years
(2012–2025) on 56 large US stocks. On Sharpe, the Mamba encoder with the contrastive
state–action head (D) cannot be told apart from the same encoder with a plain MLP
head (C):

* Main run: D − C Sharpe −0.00 [−0.16, +0.15].
* Four seeds: average +0.055, and every 95% interval contains zero.
* No ablation changes this (window 128, d_state 32, 4 layers, horizon 20, four head
  variants).

D consistently trades less than C (about 13% lower turnover). Its post-cost advantage
is larger than its before-cost one, which fits lower cost exposure, but the study does
not isolate that cause. Its log loss is no better on average.

**The simple baselines do better than any Mamba variant.**

* Linear, LightGBM and a last-row MLP have the best log loss and the best Sharpe.
* Against Mamba+MLP, their paired Sharpe differences are +0.20 to +0.24, with
  intervals above zero (before any multiple-comparison adjustment).

**No model beats equal-weight buy-and-hold after costs.**

* Linear, LightGBM and the MLP tie with it on Sharpe (paired differences −0.02 to
  −0.07, intervals around zero), at a lower drawdown.
* The TCN and every Mamba variant except E have a significantly *lower* Sharpe than
  buy-and-hold (C −0.27 [−0.44, −0.10]). E is lower too, but its interval
  (−0.14 [−0.29, +0.02]) includes zero.
* Every model's forecasts are barely better than a constant class-share forecast:
  test log loss 1.066–1.074, against 1.077–1.079 for the constant forecast.

These are negative results and are reported as such. The engineering parts (scan
kernels, leakage controls, proofs, cross-backend agreement) are listed in §7 with
the evidence for each.

Reports used here are in [`results/reports/`](results/reports/): `main.md` is the main
study, `kaggle_*.md` are the replicates and ablations, with gzip JSON alongside. Two
small checks sit next to them, made by the scripts in `tools/`:

* `prior_logloss.txt` — the class-prior baseline;
* `cuda_vs_wgpu_C_mamba_mlp.txt` — the cross-backend agreement check.

## 1. Hypotheses

The hypotheses, horizon, label margin, costs, window, reference model and primary
policy were written into `configs/main.toml` before the main run. They are recorded
in that file's comments and in each run's `run.json`. This is not an external
preregistration. Two choices were made before the main run:

* The label margin m = 0.5 was chosen from the label class shares, not from any model
  result.
* The primary policy rule was fixed after a two-fold smoke run, where a first rule
  turned every model into buy-and-hold (see C5 in the contracts).

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Mamba+CLM (D) has lower test log loss than Mamba+MLP (C) | **Not supported.** Main 1.0741 vs 1.0733 (D worse). Across the 12 paired runs D is lower in 6, tied in 1, worse in 5, never by more than 0.0022; four-seed mean 1.0717 vs 1.0722, no interval computed |
| H2 | D has a higher Sharpe after costs than C | **Not supported.** All 12 paired Sharpe intervals include 0 (main −0.00 [−0.16, +0.15]) |
| H3 | A Mamba encoder beats non-sequential baselines (linear, LightGBM, last-row MLP) | **Not supported; the reverse holds nominally.** All three have higher Sharpe than C (+0.20 to +0.24, intervals above 0 before multiplicity adjustment) and lower log loss in 12–13 of 14 years |
| H4 | Some model beats buy-and-hold after costs | **Not supported.** Best paired Sharpe difference −0.02 [−0.16, +0.12] (linear) |
| H5 | SupCon (E) or auxiliary return/risk heads (F) help CLM | Not shown. E is the best Mamba variant (Sharpe 0.85, log loss 1.0692): +0.13 [−0.03, +0.29] Sharpe vs C, and +1.06 bp [+0.25, +1.83] mean daily return (nominal). F +0.05 [−0.11, +0.19] |

## 2. Setup

**Data.** Yahoo daily bars from 1999-01-04 to 2026-10-21 (6,993 trading days):

* 56 large US stocks — the traded universe;
* SPY and the 9 sector SPDRs, used as market and sector context.

`tools/fetch_yahoo.py` downloads them, and `universe.json` holds the list.

**Features** (C2 in [`docs/CONTRACTS.md`](docs/CONTRACTS.md)): 20 point-in-time
features per stock and day:

* log returns over 1/5/20/60 days, the overnight gap and the intraday return;
* high–low range and realised volatility over 20 and 60 days;
* relative volume, momentum, distance to the 52-week high and Amihud illiquidity;
* market return and volatility, and sector-relative returns;
* cross-sectional z-scores.

`compute_features` is audited against bars truncated at the audited day. The real-data
test (`clm-data/tests/real_data.rs`) checks 51 days spread over the whole history,
including the warm-up edges, for all securities. It is in the test logs (§7). Synthetic
tests with injected leaks show the audit catches them.

**Labels** (C3): target exposures over h = 5 days, executed at the next open.

* BUY if the forward return is above θ = 2κ + 0.5·σ_h.
* SELL if it is below −θ.
* HOLD otherwise.

κ = 5 bp is the one-way cost. Class shares in the labelled pool are 32 / 26 / 42%
(124,230 / 101,055 / 164,867). The label does not depend on the current position;
the policy (C5) adds that.

**Folds** (C4): expanding window with one test year per fold, from 2012 to 2025. Each
fold has four segments:

* train, from 1999;
* validation, 252 days;
* calibration, 252 days;
* test, one year.

A sample is kept only if its label window ends before the next segment starts, with
a 5-day embargo (`t + 1 + h + e < s_next`). This purges 11 days at each boundary.
Before training, every fold runs `check_no_leakage` on the actual sample indices.
Normalisation statistics come from the fold's training days only.

**Training.** Every model trained by the Rust runner gets the same settings:

* AdamW with a cosine schedule and 100 warm-up steps;
* at most 2,000 steps of batch 256, gradient clipping at 1;
* early stopping on validation log loss (patience: 3 evaluations, 200 steps apart);
* two candidate learning rates, chosen on validation log loss.

LightGBM is trained outside the runner (`tools/lgbm_baseline.py`). That script also
tries two candidate settings and chooses by validation log loss on the same exported
folds. The runner checks that LightGBM's predictions match the fold manifest, but it
cannot enforce how they were tuned. Every model's predictions are temperature-calibrated
on the calibration year.

**Models** (main study, window 64):

| | Model | Parameters |
|---|---|---|
| linear | multinomial logistic regression on the last row | 63 |
| lgbm | LightGBM on the last row (external, Python, same folds) | – |
| A | MLP on the last row | 5,699 |
| B | TCN (4 layers, d 64) + MLP head | 23,939 |
| C | Mamba-1 (2 layers, d 64, d_state 16, expand 2) + MLP head — reference | 71,171 |
| D | same encoder + contrastive state/action head (InfoNCE, learned temperature) | 79,745 |
| E | D + supervised contrastive loss between states (weight 0.5) | 79,745 |
| F | D + auxiliary heads: clipped vol-scaled return and stop-hit (weights 0.5 / 0.2) | 88,195 |

**Policy and accounting** (C5): edge = p_buy − p_sell.

* Long if edge > τ, flat if edge < −τ, otherwise keep the position.
* Positions are equal weight across the 56 stocks and trade at the next open, at 5 bp
  one-way cost.
* Primary policy: τ = 0 for every model. Secondary policy: τ tuned on the calibration
  year.

**Statistics** (C6): stationary block bootstrap (mean block 20 days, 2,000 resamples),
paired by day when comparing two strategies. No interval is adjusted for multiple
comparisons.

## 3. Main study (`results/reports/main.md`)

All 8 models finished all 14 folds, so every table covers the same 197,120 test
samples over 3,520 days.

### Forecast quality (test, temperature-calibrated)

| model | log loss | Brier | ECE | macro AUC | IC mean (t) |
|---|---|---|---|---|---|
| linear | 1.0675 | 0.6446 | 0.011 | 0.559 | 0.0073 (1.99) |
| lgbm | 1.0680 | 0.6452 | 0.005 | 0.556 | 0.0020 (0.57) |
| **A_mlp** | **1.0663** | **0.6435** | 0.004 | **0.564** | **0.0092 (2.56)** |
| B_tcn_mlp | 1.0708 | 0.6468 | 0.004 | 0.551 | 0.0005 (0.15) |
| C_mamba_mlp | 1.0733 | 0.6487 | 0.004 | 0.551 | 0.0044 (1.34) |
| D_mamba_clm | 1.0741 | 0.6490 | **0.002** | 0.545 | 0.0036 (1.14) |
| E_mamba_supcon | 1.0692 | 0.6458 | 0.003 | 0.555 | 0.0043 (1.31) |
| F_mamba_clm_aux | 1.0731 | 0.6485 | 0.005 | 0.546 | 0.0063 (1.94) |
| constant class shares | 1.0794 (calibration-year shares) / 1.0767 (pool shares) | | | | |

The pool-share prior uses later years, so it is slightly optimistic.

The signal is weak for every model. The sequence models (B–F) see 64 days of history,
yet forecast worse than the models that see only the last day. D has the lowest ECE
(0.002) and the lowest AUC.

**Calibration.** Temperature scaling gives the lowest ECE for every model. Isotonic
regression has a higher test log loss than temperature scaling for every model (by
0.008–0.021), and vector scaling is mixed. For the linear model and LightGBM,
temperature scaling raises log loss slightly (linear 1.0658 → 1.0675).

### Trading after costs (primary policy, τ = 0, 5 bp one way)

| model | ann. return | Sharpe [95% CI] | max DD | turnover/day | long share |
|---|---|---|---|---|---|
| linear | 12.6% | 0.97 [0.47, 1.52] | 27.6% | 0.113 | 87% |
| lgbm | 12.6% | 0.96 [0.48, 1.47] | 25.2% | 0.123 | 86% |
| A_mlp | 11.5% | 0.92 [0.42, 1.48] | 28.4% | 0.143 | 84% |
| B_tcn_mlp | 9.7% | 0.81 [0.30, 1.35] | 27.0% | 0.195 | 77% |
| C_mamba_mlp | 8.2% | 0.72 [0.24, 1.27] | 25.8% | 0.259 | 70% |
| D_mamba_clm | 8.7% | 0.72 [0.23, 1.26] | 28.0% | 0.205 | 76% |
| E_mamba_supcon | 10.8% | 0.85 [0.34, 1.39] | 29.1% | 0.143 | 81% |
| F_mamba_clm_aux | 9.2% | 0.77 [0.28, 1.30] | 27.7% | 0.204 | 75% |
| buy & hold, equal weight | 15.1% | 0.99 | 34.3% | 0.004 | 100% |

**Paired against buy-and-hold** (same days, same costs):

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| linear | −0.02 [−0.16, +0.12] | −0.99 [−1.96, −0.05] |
| lgbm | −0.03 [−0.24, +0.16] | −1.01 [−2.10, +0.16] |
| A_mlp | −0.07 [−0.22, +0.09] | −1.43 [−2.57, −0.40] |
| E_mamba_supcon | −0.14 [−0.29, +0.02] | −1.70 [−2.78, −0.66] |
| B_tcn_mlp | −0.18 [−0.37, −0.02] | −2.13 [−3.21, −1.09] |
| F_mamba_clm_aux | −0.22 [−0.40, −0.06] | −2.33 [−3.42, −1.20] |
| C_mamba_mlp | −0.27 [−0.44, −0.10] | −2.76 [−3.96, −1.57] |
| D_mamba_clm | −0.27 [−0.45, −0.10] | −2.53 [−3.71, −1.41] |

All models are long most of the time, so the question is whether stepping aside on
SELL days avoids more losses than it gives up. For the best models it roughly breaks
even on risk-adjusted terms: lower drawdown, but a lower mean return. For the Mamba
models it does not.

Across the eight models, higher turnover goes with lower Sharpe. The comparison does
not isolate turnover's effect, because the models differ in other ways too. Before
costs (0 bp), linear, LightGBM and the MLP reach Sharpe 1.07–1.08. At 20 bp,
Mamba+MLP's Sharpe is −0.15.

Tuning τ on the calibration year (the secondary policy) cuts turnover 3–6×, but the
results are inconsistent. C rises to 0.88, while linear falls to 0.66.

### Paired comparison with Mamba+MLP (C)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years better Sharpe | years lower log loss |
|---|---|---|---|---|
| linear | +0.24 [+0.11, +0.38] | +1.77 [+1.09, +2.50] | 12/14 | 12/14 |
| lgbm | +0.23 [+0.05, +0.41] | +1.75 [+0.82, +2.80] | 9/14 | 12/14 |
| A_mlp | +0.20 [+0.07, +0.33] | +1.33 [+0.70, +2.00] | 11/14 | 13/14 |
| E_mamba_supcon | +0.13 [−0.03, +0.29] | +1.06 [+0.25, +1.83] | 11/14 | 13/14 |
| B_tcn_mlp | +0.08 [−0.06, +0.23] | +0.62 [−0.04, +1.35] | 8/14 | 11/14 |
| F_mamba_clm_aux | +0.05 [−0.11, +0.19] | +0.43 [−0.26, +1.14] | 7/14 | 6/14 |
| D_mamba_clm | −0.00 [−0.16, +0.15] | +0.23 [−0.50, +0.96] | 6/14 | 4/14 |

These are seven unadjusted comparisons. A Bonferroni correction would need 99.3%
intervals, which the report does not compute. None of the seven point estimates
favours the Mamba reference.

## 4. Seed replicates of the key comparison (D vs C)

The same config was rerun with three other model seeds (`replicate_seed{1,2,3}.toml`)
on Kaggle T4 GPUs (CUDA backend). The folds, data and costs are identical.

| seed | log loss C / D | Sharpe C / D | D − C Sharpe [95% CI] | D − C mean daily, bp [95% CI] | D − C Sharpe at 0 bp | turnover C / D |
|---|---|---|---|---|---|---|
| main | 1.0733 / 1.0741 | 0.72 / 0.72 | −0.00 [−0.16, +0.15] | +0.23 [−0.50, +0.96] | −0.08 | 0.259 / 0.205 |
| 1 | 1.0722 / 1.0716 | 0.73 / 0.79 | +0.07 [−0.05, +0.21] | +0.53 [−0.13, +1.30] | +0.02 | 0.250 / 0.215 |
| 2 | 1.0709 / 1.0710 | 0.78 / 0.81 | +0.03 [−0.09, +0.14] | +0.19 [−0.36, +0.73] | +0.02 | 0.233 / 0.223 |
| 3 | 1.0723 / 1.0701 | 0.65 / 0.77 | +0.12 [−0.01, +0.26] | +0.82 [+0.14, +1.56] | +0.06 | 0.257 / 0.223 |
| mean | 1.0722 / 1.0717 | 0.72 / 0.77 | +0.055 | +0.44 | +0.005 | 0.250 / 0.217 |

The seed alone moves C's Sharpe between 0.65 and 0.78, which is as large as any
D − C difference. Seed 3's mean-return interval excludes zero nominally. It is one
of 12 unadjusted comparisons, and its Sharpe interval does not exclude zero.

D's consistent property is about 13% lower turnover. Its log loss is mixed: lower in
two seeds, higher in two, slightly lower on the four-seed mean. Its average Sharpe gap
over C is +0.055 after 5 bp costs and +0.005 before costs.

## 5. Ablations (each against its own Mamba+MLP twin; Kaggle T4)

| ablation (config) | C Sharpe | D Sharpe | D − C [95% CI] | log loss C / D |
|---|---|---|---|---|
| window 128 (`window128.toml`) | 0.77 | 0.78 | +0.01 [−0.10, +0.14] | 1.0730 / 1.0730 |
| d_state 32 (`dstate32.toml`) | 0.79 | 0.84 | +0.05 [−0.07, +0.18] | 1.0725 / 1.0714 |
| 4 layers (`layers4.toml`) | 0.75 | 0.72 | −0.03 [−0.21, +0.12] | 1.0736 / 1.0739 |
| horizon 20 (`horizon20.toml`) | 0.80 | 0.79 | −0.01 [−0.09, +0.08] | 1.0799 / 1.0780 |

The contrastive-head variants (`clm_head.toml`) are each compared with the main
config's C (Sharpe 0.72):

| variant | Sharpe | vs C [95% CI] | log loss |
|---|---|---|---|
| fixed temperature 0.1 | 0.70 | −0.02 [−0.18, +0.13] | 1.0744 |
| embedding width 16 | 0.80 | +0.07 [−0.08, +0.23] | 1.0728 |
| bilinear scoring | 0.67 | −0.05 [−0.20, +0.09] | 1.0739 |
| multi-positive (ambiguity band 0.25) | 0.79 | +0.06 [−0.07, +0.20] | 1.0723 |

Every model in these runs, C twins and contrastive variants alike, has a lower Sharpe
than buy-and-hold. The paired interval excludes zero for all of them except D at
d_state 32 (−0.15 [−0.32, +0.00]). Each `kaggle_*.md` has the table.

**Cross-backend agreement.** The `clm_head` run trained C with the main study's seed
on Kaggle's T4 (CUDA). The main study trained it on the local Radeon (wgpu); the
Kaggle build log shows the `cubecl-cuda` backend. Compared prediction by prediction
(`tools/compare_oof.py`, saved in `cuda_vs_wgpu_C_mamba_mlp.txt`):

* the largest difference over all 197,120 test probabilities is 1.2e-6;
* every primary-policy decision is the same.

## 6. What the evidence supports

* **CLM vs MLP head.** In these configurations, the contrastive state–action head
  shows no Sharpe or log-loss difference from an MLP head on the same Mamba encoder.
  This holds in 12 paired runs (4 seeds, 4 ablations, 4 head variants). Its one
  consistent difference is lower turnover.
* **Sequence models vs last-row baselines.** Under the same training budget, the
  last-row baselines (logistic regression, LightGBM, MLP) have better forecast and
  trading results than the sequence models. The study compares whole configurations,
  so it does not identify why. The gap over the constant forecast is small for every
  model: the best test log loss (1.0663) is 0.010 below the pool-share prior and
  0.013 below the calibration-year prior.
* **Pipeline checks.** The pipeline does recover a planted signal on a synthetic panel
  (clm-model `clm_learns_planted_signal`). That rules out some failure modes. It does
  not show the real-data results are free of every error.

## 7. Engineering verification

**Selective-scan kernels** (core crate, `src/tensor/ops/selective_scan.rs`):

* forward — sequential, and a three-phase chunked associative scan;
* a checkpointed backward;
* all written in CubeCL.

`tests/selective_scan.rs` compares the kernels with the `f64` reference in
`src/ssm/selective.rs`, with finite-difference gradients and causality checks. The
suite runs on whichever runtime the build selects. Its logs for the CPU runtime,
wgpu and CUDA (T4, on Kaggle) are in `results/reports/test_logs/`. In the half-precision test, f16 runs on wgpu; bf16 is skipped because this
GPU does not support it.

**Lean 4** (`lean/ClmVerify`; about 1,850 lines; 71 main theorems plus supporting
lemmas, 126 in all; no `sorry`, no added axioms, no `native_decide`). The proofs are
about abstract definitions of:

* the scan algebra, chunked scan = sequential scan, causality, and the backward's
  adjoint identity;
* fold, purge and embargo disjointness and the keep rule;
* label windows;
* backtest accounting.

The GPU and Rust implementations are not proven. `REPORT.md` lists what the Lean model
leaves out: checkpoint layout, padded segments, shared-memory reductions and
parameter-gradient sums. The Lean executable `clmcheck` is a differential oracle for
fold construction and backtest ledgers. Those two clm-eval tests are `#[ignore]` by
default (they need `lake build`); their run is in the test logs. The scan kernels are
tested numerically against the `f64` reference instead.

**Tests** (logs in `results/reports/test_logs/`):

* **clm-data:**
  * injected leaks must be caught: a feature reading the next close, a full-sample
    mean, or a future cross-section;
  * conditional real-data audits: truncation, perturbation, labels.
* **clm-eval:**
  * the purging guard must reject a training label overlapping validation, an embargo
    violation, and a calibration label reaching the test year;
  * the two Lean-oracle differential tests (ignored by default).
* **clm-model:**
  * the encoder against the `f64` reference in both scan schedules, and causality;
  * finite-difference gradients and a checkpoint round trip;
  * learning a planted signal, fixed-seed reproducibility, and every model overfitting
    a small batch.
* **Scan tests** (core crate).

Results of the final runs (2026-10-22). Every suite passed with no failures:

| suite | CPU runtime | wgpu (Radeon 860M) | CUDA (T4, Kaggle) |
|---|---|---|---|
| clm-data + clm-eval, incl. Lean oracle and real-data audits (`--include-ignored`) | 125 | – | – |
| clm-model (`tests/model.rs`) | 8 | 8 | – |
| scan kernels (`tests/selective_scan.rs`) | 8 | 8 | 8 |
| scan `f64` reference unit tests (`ssm::selective`) | 2 | – | – |

On the CPU runtime the model tests took 8 hours: that runtime emulates the GPU
kernels. On wgpu they took 6 s.

**Reviews.** Five codex reviews are in `docs/reviews/`: four of the code, and the fifth
of this document. All findings were fixed, except one documented on purpose: the CLM
action attribute uses the constant 2κ threshold.

**Leakage and provenance controls in the runner:**

* `cv` refuses to resume when the computed config, data or source signature changes.
* `report` requires the stored config and dataset version to match. A different
  source version is allowed and recorded, because report-code changes do not change
  stored predictions. The reports here were regenerated that way, to add the
  buy-and-hold comparison.
* External (LightGBM) predictions must match the fold manifest exactly: data, split
  keys and normaliser hash, with exact key counts.

### Speed

All timings are on this PC's Radeon 860M (wgpu backend, f32). Logs are in
`results/reports/bench/`.

**Selective scan at the study's shape** (batch 256 × 128 channels, d_state 16; ms per
call; `mamba-clm bench`). Every output is within 1.1e-7 relative error of the
reference.

| T | forward, sequential | forward, chunked (64) | backward | checkpoints |
|---|---|---|---|---|
| 64 | 1.1 | 2.1 | 4.5 | 34 MB |
| 128 | 1.9 | 3.0 | 11.9 | 67 MB |
| 256 | 7.7 | 6.9 | 19.5 | 134 MB |
| 512 | 17.7 | 18.9 | 39.9 | 268 MB |
| 1024 | 37.4 | 54.6 | 72.3 | 537 MB |

The backward grows linearly in T, and the forward does too from T = 256 on. On this GPU the chunked parallel scan pays off only at
T = 256; elsewhere it is slower than the sequential scan. At this shape (32,768
independent channels) the default `Auto` mode already picks the sequential scan. Auto
switches to the chunked scan only when there are fewer than 8,192 channels *and* the
sequence is longer than two checkpoint segments. Even at batch 32 (4,096
channels; second log) the forced chunk-64 scan is slower here, so that threshold is
not tuned for this GPU.

The CubeCL profiler (`CUBECL_DEBUG_OPTION=profile`) showed the backward as the
largest kernel in a training step; that profile was not kept. An interleaved
in-process A/B (`mamba-clm bench --ab-flush`; log
`wgpu_radeon_backward_ab_b256_d128.log`, three rounds per T) then compared segment
lengths and reduction intervals. The default (4-step segments, flush every 2 steps):

* is 3–7× faster than 8-step segments: T = 64, 2.7–6.7 vs 18 ms; T = 1024, 90–95 vs
  300–305 ms;
* is about as fast as 2-step segments, and faster at T = 128–512, while 2-step
  segments need twice the checkpoint memory.

An 8-step segment keeps 128 history floats per thread against 64 for a 4-step one, so
the slowdown fits register spill. The profiler output does not show that directly.

**Models** (`mamba-clm bench-model`, batch 256, window 64):

| model | ms per training step | samples/s | inference, 1 / 56 / 1,024 windows (ms) |
|---|---|---|---|
| linear | 2.2 | 114,000 | 0.10 / 0.14 / 6.9 |
| A_mlp | 1.2 | 206,000 | 0.37 / 0.44 / 6.9 |
| B_tcn_mlp | 39.2 | 6,500 | 1.5 / 3.7 / 73 |
| C_mamba_mlp | 58.9 | 4,300 | 0.80 / 3.2 / 64 |
| D_mamba_clm | 55.3 | 4,600 | 0.82 / 2.9 / 69 |
| E_mamba_supcon | 59.6 | 4,300 | 1.07 / 3.5 / 71 |
| F_mamba_clm_aux | 54.9 | 4,700 | 1.58 / 3.0 / 65 |

The contrastive head costs nothing measurable over the MLP head: action embeddings
are cached, and scoring three actions is tiny. A full day's decision for the
56-stock universe takes about 3 ms. One fold of a Mamba model (two candidates, up to
2,000 steps each, plus calibration and prediction) took 132–155 s on average in the
main run (`main.md`, "Cost of each model").

## 8. Limitations

* **Survivorship bias.** The universe is today's large caps, back-filled to 1999. This
  flatters buy-and-hold and every long-biased model alike. Comparisons between models
  are less affected than absolute returns.
* **Data.** Static sector map, no corporate-action checks beyond Yahoo's adjusted
  prices, and one data source.
* **Tuning.** One inner split per fold: one validation year to choose between two
  candidates, and one calibration year. The small tuning budget is deliberate, since a
  bigger one would favour the bigger models, but it leaves some performance
  unexplored.
* **Model scope.** The action set has three actions, so "hard negatives" are just the
  other two actions. The plan's quantile heads were not built.
* **Statistics and costs.** No interval is adjusted for multiple comparisons. The 5 bp
  cost model has no market impact or borrowing costs.
* **Sequence length.** The batch runner varies sequence length only between batches;
  there is no padding mask within a batch.
* **No Python bindings.** This module has none, so the repository's Python/Rust parity
  tests do not apply.
