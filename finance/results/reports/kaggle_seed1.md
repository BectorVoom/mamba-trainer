# CV report: replicate_seed1

Horizon 5 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 64. Reference: `C_mamba_mlp_s1`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s1 | 197120 | 0.423 | 0.360 | 1.0730 → 1.0722 | 0.6478 | 0.0023 | 0.5511 | 0.0090 | 2.85 |
| D_mamba_clm_s1 | 197120 | 0.424 | 0.360 | 1.0710 → 1.0716 | 0.6474 | 0.0016 | 0.5507 | 0.0060 | 1.87 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp_s1 | 1.0730 / 0.6481 / 0.0190 | 1.0722 / 0.6478 / 0.0023 | 1.0716 / 0.6473 / 0.0088 | 1.0803 / 0.6479 / 0.0061 |
| D_mamba_clm_s1 | 1.0710 / 0.6471 / 0.0113 | 1.0716 / 0.6474 / 0.0016 | 1.0715 / 0.6473 / 0.0074 | 1.0782 / 0.6476 / 0.0052 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s1 | 8.5% | 0.73 [0.24, 1.27] | 1.01 | 26.2% | 1.155 | 0.250 | 197.5% | 73/27/0% |
| D_mamba_clm_s1 | 9.8% | 0.79 [0.29, 1.34] | 1.12 | 29.3% | 1.171 | 0.215 | 254.0% | 77/23/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp_s1 | 6.3% | 0.60 | 29.4% | 0.078 | 0.12 |
| D_mamba_clm_s1 | 8.7% | 0.73 | 30.2% | 0.055 | 0.11 |

## Paired comparison with `C_mamba_mlp_s1` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_mamba_clm_s1 | +0.07 [-0.05, +0.21] | +0.53 [-0.13, +1.30] | 7/14 | 8/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp_s1 | -0.26 [-0.44, -0.10] | -2.62 [-3.76, -1.50] |
| D_mamba_clm_s1 | -0.20 [-0.35, -0.05] | -2.09 [-3.02, -1.19] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp_s1 | 0.99 | 0.86 | 0.73 | 0.46 | -0.08 |
| D_mamba_clm_s1 | 1.01 | 0.90 | 0.79 | 0.57 | 0.14 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp_s1 | 0.90 (1196 d) | 0.63 (2324 d) |
| D_mamba_clm_s1 | 0.99 (1196 d) | 0.69 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s1 | 1.22 | 2.68 | 0.86 | -0.10 | 1.04 | 2.67 | -0.11 | 2.31 | 0.01 | 2.18 | 0.01 | 0.81 | 0.95 | 0.70 |
| D_mamba_clm_s1 | 1.20 | 2.57 | 1.03 | -0.02 | 1.32 | 2.72 | -0.26 | 2.29 | 0.64 | 2.11 | -0.09 | 0.85 | 0.98 | 0.65 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp_s1 | 71171 | 36 | 19818 | - |
| D_mamba_clm_s1 | 79745 | 32 | 19519 | - |
