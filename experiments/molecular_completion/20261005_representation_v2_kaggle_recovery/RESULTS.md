# Representation v2: full Kaggle T4 verification

Training used 54,665 measured spectra across 4,953 molecular connectivities. Evaluation used 100 test queries.

| Arm | Generation top-1 / 10 / 25 | Supplied-pool retrieval top-1 / 10 / 25 |
|---|---|---|
| full | 0.0% / 0.0% / 0.0% | 3.0% / 26.0% / 44.0% |
| transformer_decoder | 0.0% / 0.0% / 0.0% | 3.0% / 27.0% / 43.0% |
| no_substructures | 0.0% / 0.0% / 0.0% | 3.0% / 27.0% / 43.0% |

Shuffled-pool retrieval top-25: 44.0%.

Measured spectrum substitution results are in `paired_spectrum_audit.json`; same-formula and unmatched-formula donor results are reported separately.

This experiment measures identification and graph validity. Physical stability was not evaluated. The decoder remains connectivity only, and the new neural features are implemented in the Python experiment.

Archive SHA-256: `2099f58611a510375a45ebfc52fa56b0e7dcfe75447757eb13942e123b6c4d6d`. Verified 116 manifest files, 12 model checkpoints and 4 vocabularies.

## Interpretation and diagnostics

All three arms completed all 100 source-ordered test queries (87 distinct
connectivity identities). Scores above use the complete 100-query denominator.
The primary model generated completed graphs for 12 queries, producing 17
distinct candidates in total; none recovered the correct connectivity.
Formula top-4 recovery was 14/100. Only 15/100 reference structures fell inside
the declared generation domain, and accuracy remains zero even for them.

The primary retrieval top-25 score equals the shuffled supplied-pool baseline:
44/100. The connectivity-clustered bootstrap for the difference is 0 percentage
points, with a 95% interval of approximately -4.04 to +4.04 percentage points.
Both matched controls scored 43/100; the observed one-query advantage does not
establish improved identification quality.

The same-formula spectrum-substitution audit had only two queries; neither
showed lower target NLL with its correct spectrum. This small sample provides
no positive evidence of same-formula spectral discrimination. The other 13
donors had unmatched formulas and are reported separately.

All primary checkpoint hashes equal the seven models recovered from the
original failed run. Both controls completed their declared training schedules.
The recovered primary models were evaluated with the explicit-hydrogen feature
fix, preserving the original training snapshots and checkpoint provenance.

Local verification on 2026-10-05 checked the archive SHA-256, ZIP CRC, all
116 manifest files, 12 strictly loaded finite model states, and four
vocabularies. Metrics were independently recomputed from all three arms'
predictions and matched the report. The collector's earlier local Torch import
failure was resolved by rerunning the verification with the available project
interpreter; its historical error is retained in
`collector_verification_failure_resolved.json`.

The full workflow succeeded technically. The benchmark did not demonstrate
successful de novo identification or retrieval improvement over its shuffled
baseline. Physical stability was not evaluated.
