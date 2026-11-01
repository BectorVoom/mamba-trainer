# CV report: dstate32

Horizon 5 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 64. Reference: `C_mamba_mlp_n32`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_n32 | 197120 | 0.423 | 0.360 | 1.0729 → 1.0725 | 0.6482 | 0.0022 | 0.5511 | 0.0065 | 2.02 |
| D_mamba_clm_n32 | 197120 | 0.425 | 0.359 | 1.0714 → 1.0714 | 0.6474 | 0.0035 | 0.5522 | 0.0071 | 2.14 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp_n32 | 1.0729 / 0.6483 / 0.0196 | 1.0725 / 0.6482 / 0.0022 | 1.0734 / 0.6483 / 0.0081 | 1.0812 / 0.6479 / 0.0067 |
| D_mamba_clm_n32 | 1.0714 / 0.6474 / 0.0167 | 1.0714 / 0.6474 / 0.0035 | 1.0722 / 0.6474 / 0.0086 | 1.0844 / 0.6481 / 0.0071 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_n32 | 9.2% | 0.79 [0.27, 1.35] | 1.12 | 27.7% | 1.167 | 0.247 | 230.5% | 75/25/0% |
| D_mamba_clm_n32 | 10.0% | 0.84 [0.34, 1.37] | 1.19 | 25.9% | 1.180 | 0.223 | 266.5% | 74/26/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp_n32 | 8.2% | 0.69 | 33.8% | 0.040 | 0.11 |
| D_mamba_clm_n32 | 9.8% | 0.81 | 26.2% | 0.054 | 0.09 |

## Paired comparison with `C_mamba_mlp_n32` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_mamba_clm_n32 | +0.05 [-0.07, +0.18] | +0.31 [-0.26, +0.89] | 7/14 | 9/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp_n32 | -0.20 [-0.37, -0.03] | -2.32 [-3.31, -1.31] |
| D_mamba_clm_n32 | -0.15 [-0.32, +0.00] | -2.02 [-3.05, -1.02] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp_n32 | 1.06 | 0.93 | 0.79 | 0.53 | -0.01 |
| D_mamba_clm_n32 | 1.08 | 0.96 | 0.84 | 0.60 | 0.13 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp_n32 | 0.90 (1196 d) | 0.78 (2324 d) |
| D_mamba_clm_n32 | 1.06 (1196 d) | 0.72 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_n32 | 1.16 | 2.43 | 0.88 | -0.32 | 1.38 | 3.16 | -0.44 | 2.54 | 0.83 | 1.84 | -0.48 | 0.68 | 0.92 | 1.08 |
| D_mamba_clm_n32 | 1.15 | 2.26 | 1.02 | -0.04 | 1.51 | 2.51 | -0.01 | 2.25 | 0.72 | 1.82 | -0.09 | 0.84 | 0.76 | 1.14 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp_n32 | 83459 | 53 | 13343 | - |
| D_mamba_clm_n32 | 92033 | 45 | 13227 | - |
