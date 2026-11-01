# CV report: clm_head

Horizon 5 days, label cost 5 bp, margin m = 0.5, policy cost 5 bp one way, window 64. Reference: `C_mamba_mlp`.

Test years in every table (finished by every reported model): 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Prediction quality (test, calibrated)

| model | n | acc | bal acc | log loss (raw → cal) | Brier | ECE | macro AUC | IC mean | IC t |
|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp | 197120 | 0.422 | 0.362 | 1.0743 → 1.0733 | 0.6487 | 0.0035 | 0.5512 | 0.0044 | 1.34 |
| D_fixed_temp | 197120 | 0.423 | 0.358 | 1.0745 → 1.0744 | 0.6492 | 0.0025 | 0.5446 | 0.0039 | 1.23 |
| D_embed16 | 197120 | 0.426 | 0.359 | 1.0716 → 1.0728 | 0.6483 | 0.0050 | 0.5488 | 0.0047 | 1.47 |
| D_bilinear | 197120 | 0.422 | 0.358 | 1.0739 → 1.0739 | 0.6488 | 0.0018 | 0.5462 | 0.0026 | 0.80 |
| D_multi_positive | 197120 | 0.423 | 0.357 | 1.0772 → 1.0723 | 0.6481 | 0.0027 | 0.5494 | 0.0045 | 1.41 |

## Calibration methods (test log loss / Brier / ECE, fitted on each fold's calibration segment)

| model | none | temperature | vector | isotonic |
|---|---|---|---|---|
| C_mamba_mlp | 1.0743 / 0.6492 / 0.0185 | 1.0733 / 0.6487 / 0.0035 | 1.0725 / 0.6478 / 0.0076 | 1.0841 / 0.6486 / 0.0086 |
| D_fixed_temp | 1.0745 / 0.6491 / 0.0159 | 1.0744 / 0.6492 / 0.0025 | 1.0736 / 0.6486 / 0.0082 | 1.0845 / 0.6496 / 0.0085 |
| D_embed16 | 1.0716 / 0.6475 / 0.0099 | 1.0728 / 0.6483 / 0.0050 | 1.0726 / 0.6479 / 0.0078 | 1.0823 / 0.6486 / 0.0077 |
| D_bilinear | 1.0739 / 0.6487 / 0.0172 | 1.0739 / 0.6488 / 0.0018 | 1.0730 / 0.6481 / 0.0112 | 1.0859 / 0.6491 / 0.0070 |
| D_multi_positive | 1.0772 / 0.6512 / 0.0412 | 1.0723 / 0.6481 / 0.0027 | 1.0719 / 0.6476 / 0.0089 | 1.0788 / 0.6487 / 0.0098 |

## Trading after costs (5 bp one way; primary policy: fixed tau = 0 for every model)

| model | ann. return | Sharpe [95% CI] | Sortino | max DD | profit factor | turnover/day | total net | BUY/SELL/HOLD |
|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp | 8.2% | 0.72 [0.24, 1.27] | 1.01 | 25.8% | 1.155 | 0.259 | 185.6% | 70/30/0% |
| D_fixed_temp | 8.5% | 0.70 [0.21, 1.24] | 0.98 | 28.2% | 1.148 | 0.209 | 195.4% | 75/25/0% |
| D_embed16 | 9.6% | 0.80 [0.30, 1.35] | 1.12 | 28.1% | 1.170 | 0.201 | 244.8% | 77/23/0% |
| D_bilinear | 8.1% | 0.67 [0.19, 1.21] | 0.93 | 29.7% | 1.141 | 0.208 | 180.5% | 75/25/0% |
| D_multi_positive | 9.6% | 0.79 [0.27, 1.33] | 1.10 | 28.7% | 1.165 | 0.220 | 242.9% | 78/22/0% |
| buy & hold (equal weight) | 15.1% | 0.99 | 1.39 | 34.3% | 1.206 | 0.004 | 599.2% | |

## Secondary policy: tau tuned per fold on the calibration segment

| model | ann. return | Sharpe | max DD | turnover/day | mean tau |
|---|---|---|---|---|---|
| C_mamba_mlp | 9.0% | 0.88 | 28.4% | 0.052 | 0.13 |
| D_fixed_temp | 8.2% | 0.68 | 33.8% | 0.042 | 0.12 |
| D_embed16 | 5.1% | 0.48 | 35.5% | 0.040 | 0.12 |
| D_bilinear | 7.2% | 0.63 | 33.4% | 0.030 | 0.14 |
| D_multi_positive | 6.3% | 0.54 | 36.7% | 0.044 | 0.10 |

## Paired comparison with `C_mamba_mlp` (stationary bootstrap, mean block 20 days, 2000 resamples)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] | years with higher Sharpe | years with lower log loss |
|---|---|---|---|---|
| D_bilinear | -0.05 [-0.20, +0.09] | -0.01 [-0.69, +0.66] | 5/14 | 5/14 |
| D_embed16 | +0.07 [-0.08, +0.23] | +0.57 [-0.17, +1.27] | 9/14 | 9/14 |
| D_fixed_temp | -0.02 [-0.18, +0.13] | +0.13 [-0.59, +0.87] | 6/14 | 4/14 |
| D_multi_positive | +0.06 [-0.07, +0.20] | +0.56 [-0.08, +1.24] | 8/14 | 8/14 |

## Paired comparison with equal-weight buy and hold (same days and costs, same bootstrap)

| model | Sharpe diff [95% CI] | mean daily diff, bp [95% CI] |
|---|---|---|
| C_mamba_mlp | -0.27 [-0.44, -0.10] | -2.76 [-3.96, -1.57] |
| D_bilinear | -0.32 [-0.50, -0.16] | -2.77 [-3.96, -1.69] |
| D_embed16 | -0.19 [-0.37, -0.02] | -2.19 [-3.44, -1.02] |
| D_fixed_temp | -0.29 [-0.46, -0.12] | -2.62 [-3.80, -1.51] |
| D_multi_positive | -0.20 [-0.37, -0.05] | -2.19 [-3.31, -1.14] |

## Cost sensitivity (Sharpe)

| model | 0 bp | 2.5 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| C_mamba_mlp | 1.01 | 0.87 | 0.72 | 0.43 | -0.15 |
| D_fixed_temp | 0.92 | 0.81 | 0.70 | 0.48 | 0.05 |
| D_embed16 | 1.01 | 0.90 | 0.80 | 0.59 | 0.17 |
| D_bilinear | 0.89 | 0.78 | 0.67 | 0.45 | 0.02 |
| D_multi_positive | 1.01 | 0.90 | 0.79 | 0.56 | 0.10 |

## Market-volatility regimes (Sharpe high / low)

| model | high vol | low vol |
|---|---|---|
| C_mamba_mlp | 0.81 (1196 d) | 0.72 (2324 d) |
| D_fixed_temp | 0.80 (1196 d) | 0.68 (2324 d) |
| D_embed16 | 0.95 (1196 d) | 0.73 (2324 d) |
| D_bilinear | 0.69 (1196 d) | 0.73 (2324 d) |
| D_multi_positive | 0.90 (1196 d) | 0.76 (2324 d) |

## Sharpe per test year

| model | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_mamba_mlp | 1.32 | 2.18 | 1.21 | -0.16 | 0.84 | 2.79 | -0.30 | 2.28 | 0.04 | 1.70 | -0.16 | 0.91 | 1.27 | 0.97 |
| D_fixed_temp | 1.38 | 2.19 | 1.15 | -0.24 | 1.18 | 2.21 | -0.22 | 1.95 | -0.07 | 2.24 | -0.26 | 0.79 | 1.09 | 1.09 |
| D_embed16 | 1.13 | 2.26 | 1.08 | -0.17 | 1.58 | 2.37 | -0.04 | 2.48 | -0.26 | 2.11 | -0.04 | 0.96 | 1.31 | 1.42 |
| D_bilinear | 1.42 | 2.11 | 1.08 | -0.20 | 1.18 | 2.36 | -0.35 | 1.96 | -0.11 | 1.87 | -0.24 | 0.92 | 0.94 | 1.21 |
| D_multi_positive | 1.27 | 2.39 | 0.91 | -0.08 | 1.24 | 2.27 | 0.11 | 1.99 | 0.10 | 2.16 | -0.27 | 0.76 | 1.41 | 1.10 |

## Cost of each model

| model | parameters | mean seconds per fold | training samples/s | peak device MB |
|---|---|---|---|---|
| C_mamba_mlp | 71171 | 35 | 19292 | 702 |
| D_fixed_temp | 79745 | 28 | 20267 | 702 |
| D_embed16 | 73505 | 29 | 20241 | 702 |
| D_bilinear | 83841 | 29 | 20021 | 702 |
| D_multi_positive | 79745 | 24 | 20167 | 702 |
