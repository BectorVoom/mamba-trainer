# CV report: replicate_seed2

Horizon 5 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 64. Reference: `C_mamba_mlp_s2`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s2 | 197120 | 0.426 | 0.361 | 1.0705 → 1.0709 | 0.6470 | 0.0018 | 0.5540 | 0.0109 | 3.37 |
| D_mamba_clm_s2 | 197120 | 0.423 | 0.358 | 1.0703 → 1.0710 | 0.6471 | 0.0042 | 0.5537 | 0.0075 | 2.21 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp_s2 | 1.0705 / 0.6465 / 0.0153 | 1.0709 / 0.6470 / 0.0018 | 1.0723 / 0.6477 / 0.0104 | 1.0794 / 0.6476 / 0.0060 |
| D_mamba_clm_s2 | 1.0703 / 0.6467 / 0.0149 | 1.0710 / 0.6471 / 0.0042 | 1.0731 / 0.6484 / 0.0118 | 1.0851 / 0.6486 / 0.0075 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s2 | 9.4% | 0.78 [0.28, 1.33] | 1.09 | 29.3% | 1.169 | 0.233 | 235.8% | 75/25/0% |
| D_mamba_clm_s2 | 9.9% | 0.81 [0.31, 1.34] | 1.14 | 29.5% | 1.173 | 0.223 | 257.7% | 76/24/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp_s2 | 7.9% | 0.66 | 34.0% | 0.044 | 0.12 |
| D_mamba_clm_s2 | 8.6% | 0.73 | 30.7% | 0.038 | 0.12 |

## Paired comparison with `C_mamba_mlp_s2` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_mamba_clm_s2 | +0.03 [-0.09, +0.14] | +0.19 [-0.36, +0.73] | 8/14 | 7/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp_s2 | -0.21 [-0.35, -0.07] | -2.26 [-3.28, -1.29] |
| D_mamba_clm_s2 | -0.18 [-0.35, -0.04] | -2.07 [-3.07, -1.08] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp_s2 | 1.02 | 0.90 | 0.78 | 0.54 | 0.05 |
| D_mamba_clm_s2 | 1.04 | 0.92 | 0.81 | 0.58 | 0.12 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp_s2 | 0.87 (1196 d) | 0.78 (2324 d) |
| D_mamba_clm_s2 | 1.02 (1196 d) | 0.69 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s2 | 1.02 | 2.42 | 1.33 | 0.02 | 1.16 | 2.89 | -0.06 | 2.37 | 0.33 | 1.89 | -0.19 | 0.57 | 1.10 | 0.80 |
| D_mamba_clm_s2 | 1.26 | 2.49 | 0.93 | -0.06 | 1.06 | 2.62 | -0.05 | 2.42 | 0.52 | 1.88 | -0.03 | 0.64 | 1.08 | 0.95 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp_s2 | 71171 | 37 | 19784 | 702 |
| D_mamba_clm_s2 | 79745 | 33 | 19419 | 702 |
