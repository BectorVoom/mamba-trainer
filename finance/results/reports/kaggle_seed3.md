# CV report: replicate_seed3

Horizon 5 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 64. Reference: `C_mamba_mlp_s3`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s3 | 197120 | 0.421 | 0.358 | 1.0727 → 1.0723 | 0.6480 | 0.0043 | 0.5500 | 0.0061 | 1.92 |
| D_mamba_clm_s3 | 197120 | 0.424 | 0.357 | 1.0700 → 1.0701 | 0.6467 | 0.0036 | 0.5545 | 0.0091 | 2.69 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp_s3 | 1.0727 / 0.6481 / 0.0208 | 1.0723 / 0.6480 / 0.0043 | 1.0723 / 0.6477 / 0.0127 | 1.0823 / 0.6479 / 0.0077 |
| D_mamba_clm_s3 | 1.0700 / 0.6466 / 0.0170 | 1.0701 / 0.6467 / 0.0036 | 1.0721 / 0.6476 / 0.0124 | 1.0780 / 0.6477 / 0.0081 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s3 | 7.5% | 0.65 [0.13, 1.22] | 0.89 | 30.6% | 1.141 | 0.257 | 159.7% | 71/29/0% |
| D_mamba_clm_s3 | 9.6% | 0.77 [0.25, 1.36] | 1.07 | 31.5% | 1.167 | 0.223 | 241.7% | 76/24/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp_s3 | 6.7% | 0.60 | 31.9% | 0.035 | 0.14 |
| D_mamba_clm_s3 | 8.6% | 0.72 | 35.1% | 0.041 | 0.12 |

## Paired comparison with `C_mamba_mlp_s3` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_mamba_clm_s3 | +0.12 [-0.01, +0.26] | +0.82 [+0.14, +1.56] | 8/14 | 9/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp_s3 | -0.34 [-0.50, -0.18] | -3.01 [-4.14, -1.94] |
| D_mamba_clm_s3 | -0.22 [-0.37, -0.06] | -2.19 [-3.18, -1.22] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp_s3 | 0.93 | 0.79 | 0.65 | 0.37 | -0.19 |
| D_mamba_clm_s3 | 0.99 | 0.88 | 0.77 | 0.54 | 0.09 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp_s3 | 0.66 (1196 d) | 0.72 (2324 d) |
| D_mamba_clm_s3 | 0.86 (1196 d) | 0.77 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_s3 | 1.19 | 2.54 | 0.88 | -0.04 | 1.34 | 2.90 | -0.62 | 2.26 | -0.11 | 2.31 | -0.31 | 0.31 | 1.28 | 0.68 |
| D_mamba_clm_s3 | 1.66 | 2.50 | 0.99 | -0.16 | 1.29 | 2.79 | -0.44 | 2.73 | 0.35 | 1.81 | -0.15 | 0.62 | 1.29 | 0.49 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp_s3 | 71171 | 35 | 21027 | 702 |
| D_mamba_clm_s3 | 79745 | 28 | 20646 | 702 |
