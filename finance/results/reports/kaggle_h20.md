# CV report: horizon20

Horizon 20 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 64. Reference: `C_mamba_mlp_h20`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_h20 | 197120 | 0.409 | 0.369 | 1.0806 → 1.0799 | 0.6533 | 0.0145 | 0.5473 | -0.0026 | -0.86 |
| D_mamba_clm_h20 | 197120 | 0.412 | 0.372 | 1.0777 → 1.0780 | 0.6520 | 0.0164 | 0.5492 | -0.0003 | -0.10 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp_h20 | 1.0806 / 0.6535 / 0.0298 | 1.0799 / 0.6533 / 0.0145 | 1.0895 / 0.6594 / 0.0350 | 1.1053 / 0.6584 / 0.0332 |
| D_mamba_clm_h20 | 1.0777 / 0.6517 / 0.0246 | 1.0780 / 0.6520 / 0.0164 | 1.0857 / 0.6570 / 0.0301 | 1.1113 / 0.6575 / 0.0282 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_h20 | 9.7% | 0.80 [0.29, 1.35] | 1.13 | 28.0% | 1.167 | 0.179 | 251.4% | 80/20/0% |
| D_mamba_clm_h20 | 9.9% | 0.79 [0.28, 1.34] | 1.11 | 27.4% | 1.165 | 0.160 | 259.2% | 82/18/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp_h20 | 10.3% | 0.85 | 28.0% | 0.048 | 0.10 |
| D_mamba_clm_h20 | 9.6% | 0.77 | 28.7% | 0.023 | 0.14 |

## Paired comparison with `C_mamba_mlp_h20` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_mamba_clm_h20 | -0.01 [-0.09, +0.08] | +0.08 [-0.34, +0.51] | 6/14 | 6/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp_h20 | -0.19 [-0.32, -0.06] | -2.13 [-3.14, -1.17] |
| D_mamba_clm_h20 | -0.20 [-0.32, -0.08] | -2.04 [-3.06, -1.14] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp_h20 | 0.99 | 0.89 | 0.80 | 0.62 | 0.24 |
| D_mamba_clm_h20 | 0.95 | 0.87 | 0.79 | 0.63 | 0.31 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp_h20 | 0.88 (1196 d) | 0.81 (2324 d) |
| D_mamba_clm_h20 | 0.92 (1196 d) | 0.75 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_h20 | 0.96 | 2.33 | 1.54 | 0.19 | 0.89 | 3.37 | -0.36 | 2.55 | -0.05 | 1.99 | -0.28 | 0.79 | 1.32 | 1.11 |
| D_mamba_clm_h20 | 0.94 | 2.58 | 1.17 | -0.05 | 0.99 | 3.29 | -0.17 | 2.50 | -0.07 | 2.27 | -0.17 | 0.94 | 1.05 | 1.05 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp_h20 | 71171 | 33 | 18406 | 702 |
| D_mamba_clm_h20 | 79745 | 28 | 19121 | 702 |
