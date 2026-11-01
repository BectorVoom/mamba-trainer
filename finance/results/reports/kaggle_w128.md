# CV report: window128

Horizon 5 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 128. Reference: `C_mamba_mlp_w128`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_w128 | 197120 | 0.423 | 0.359 | 1.0735 → 1.0730 | 0.6483 | 0.0025 | 0.5487 | 0.0006 | 0.18 |
| D_mamba_clm_w128 | 197120 | 0.424 | 0.358 | 1.0735 → 1.0730 | 0.6482 | 0.0018 | 0.5480 | 0.0002 | 0.07 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp_w128 | 1.0735 / 0.6483 / 0.0208 | 1.0730 / 0.6483 / 0.0025 | 1.0733 / 0.6481 / 0.0098 | 1.0797 / 0.6481 / 0.0085 |
| D_mamba_clm_w128 | 1.0735 / 0.6484 / 0.0195 | 1.0730 / 0.6482 / 0.0018 | 1.0727 / 0.6478 / 0.0104 | 1.0794 / 0.6488 / 0.0101 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_w128 | 8.8% | 0.77 [0.27, 1.29] | 1.08 | 25.3% | 1.162 | 0.253 | 212.5% | 72/28/0% |
| D_mamba_clm_w128 | 9.4% | 0.78 [0.29, 1.31] | 1.10 | 29.4% | 1.168 | 0.224 | 237.7% | 75/25/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp_w128 | 6.5% | 0.57 | 30.6% | 0.050 | 0.13 |
| D_mamba_clm_w128 | 9.1% | 0.76 | 31.4% | 0.066 | 0.08 |

## Paired comparison with `C_mamba_mlp_w128` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_mamba_clm_w128 | +0.01 [-0.10, +0.14] | +0.25 [-0.28, +0.85] | 8/14 | 7/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp_w128 | -0.22 [-0.42, -0.04] | -2.49 [-3.67, -1.30] |
| D_mamba_clm_w128 | -0.21 [-0.40, -0.03] | -2.24 [-3.40, -1.14] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp_w128 | 1.04 | 0.91 | 0.77 | 0.49 | -0.06 |
| D_mamba_clm_w128 | 1.01 | 0.90 | 0.78 | 0.55 | 0.08 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp_w128 | 0.86 (1196 d) | 0.76 (2324 d) |
| D_mamba_clm_w128 | 0.90 (1196 d) | 0.76 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp_w128 | 1.27 | 2.40 | 1.30 | -0.35 | 0.81 | 2.82 | -0.33 | 2.45 | 0.28 | 1.80 | -0.30 | 0.99 | 1.22 | 1.22 |
| D_mamba_clm_w128 | 1.36 | 2.60 | 1.07 | -0.10 | 0.99 | 1.61 | 0.03 | 2.23 | 0.23 | 1.73 | -0.13 | 1.03 | 1.53 | 1.05 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp_w128 | 71171 | 63 | 10480 | - |
| D_mamba_clm_w128 | 79745 | 59 | 10337 | - |
