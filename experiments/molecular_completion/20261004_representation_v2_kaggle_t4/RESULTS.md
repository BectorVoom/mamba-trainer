# Representation v2 Kaggle T4: incomplete run

Confirmed on 2026-10-05 JST from downloaded Kaggle outputs.
Kaggle reports COMPLETE because the wrapper finished normally to preserve
artifacts. The experiment itself reports FAILED in the training stage.

All primary training schedules completed: upstream folds 0/1/2 and final
predictor, 8 epochs each; molecular prior, 10 epochs; conditional model,
15 epochs; same-formula ranker, 30 epochs. Training retained 54,665 measured
spectra across 4,953 connectivity identities and 945 calibration spectra.

Evaluation stopped on query 26 of 100. Retrieval candidate encoding raised
`KeyError: ('H', 0, 1)` in `canonical_graph_features` while assigning typed
atom features to an explicit hydrogen. The exception propagated through
`graph_tensors`, `ranking_features`, and `retrieval_prediction`.

Only the first 25 source-ordered test queries were persisted:

| Partial observation | Count |
|---|---:|
| Generation top-1 / top-10 / top-25 hits | 0 / 0 / 0 of 25 |
| Queries with valid generated candidates | 0 of 25 |
| Supplied-pool retrieval top-1 / top-10 / top-25 hits | 2 / 7 / 13 of 25 |

These are incomplete, source-ordered observations, not scores for the
100-query benchmark. Do not compare them with completed runs as final metrics.

Diagnostics, spectrum-substitution audit, matched Transformer and
no-substructure controls, retrieval audit, and final report did not run.
There is no verified final results archive or summary.json.

Seven primary models strictly loaded on CPU with finite weights, all four
vocabularies loaded, and 20 source snapshots matched their recorded SHA-256
hashes. Evidence is in `partial_artifact_verification.json`; checkpoint hashes
are local integrity records, not comparisons with a final remote manifest.

The graph-feature error remains unresolved. The primary weights are saved
locally and in Kaggle outputs, so recovery can preserve them. Any evaluation
fix must retain the training provenance and checkpoint protocol guard rather
than silently changing the checkpoint's declared training source.

The local collector was no longer active when results were checked. No
continuation or additional GPU run was launched during this confirmation.
