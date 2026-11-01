# CV report: layers4

Horizon 5 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 64. Reference: `C_mamba_mlp_l4`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_l4 | 197120 | 0.421 | 0.361 | 1.0770 → 1.0736 | 0.6490 | 0.0041 | 0.5509 | 0.0025 | 0.80 |
| D_mamba_clm_l4 | 197120 | 0.421 | 0.355 | 1.0745 → 1.0739 | 0.6491 | 0.0023 | 0.5454 | 0.0013 | 0.41 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp_l4 | 1.0770 / 0.6507 / 0.0264 | 1.0736 / 0.6490 / 0.0041 | 1.0732 / 0.6483 / 0.0071 | 1.0784 / 0.6488 / 0.0067 |
| D_mamba_clm_l4 | 1.0745 / 0.6494 / 0.0212 | 1.0739 / 0.6491 / 0.0023 | 1.0729 / 0.6483 / 0.0078 | 1.0824 / 0.6488 / 0.0090 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_l4 | 8.2% | 0.75 [0.22, 1.34] | 1.03 | 29.6% | 1.158 | 0.264 | 188.9% | 71/29/0% |
| D_mamba_clm_l4 | 8.4% | 0.72 [0.24, 1.24] | 1.01 | 27.0% | 1.153 | 0.238 | 195.2% | 73/27/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp_l4 | 6.9% | 0.65 | 32.6% | 0.062 | 0.12 |
| D_mamba_clm_l4 | 8.1% | 0.70 | 31.6% | 0.067 | 0.11 |

## Paired comparison with `C_mamba_mlp_l4` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_mamba_clm_l4 | -0.03 [-0.21, +0.12] | +0.10 [-0.61, +0.79] | 7/14 | 8/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp_l4 | -0.24 [-0.40, -0.07] | -2.74 [-3.83, -1.68] |
| D_mamba_clm_l4 | -0.27 [-0.43, -0.12] | -2.64 [-3.73, -1.53] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp_l4 | 1.05 | 0.90 | 0.75 | 0.44 | -0.16 |
| D_mamba_clm_l4 | 0.97 | 0.84 | 0.72 | 0.46 | -0.05 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp_l4 | 0.72 (1196 d) | 0.86 (2324 d) |
| D_mamba_clm_l4 | 0.91 (1196 d) | 0.61 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_l4 | 1.36 | 2.58 | 1.14 | -0.31 | 1.30 | 3.25 | -0.43 | 2.27 | 0.03 | 1.79 | -0.31 | 0.52 | 1.30 | 1.06 |
| D_mamba_clm_l4 | 1.18 | 2.12 | 1.00 | -0.11 | 0.87 | 1.57 | 0.02 | 2.47 | 0.25 | 1.99 | 0.00 | 0.37 | 0.55 | 1.06 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp_l4 | 136579 | 59 | 11073 | 703 |
| D_mamba_clm_l4 | 145153 | 49 | 10935 | 703 |
