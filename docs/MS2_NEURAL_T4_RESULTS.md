# MS2 neural ranker: Colab T4 results

The tested neural retrieval pilot did not outperform the frozen Ridge baseline,
and auxiliary neural spectrum reconstruction showed no consistent benefit.
This does not validate the full completion/stability proposal. It does show
spectral signal above uniform ranking on the previously inspected validation
slice. No quantum calculations were performed.

## Measured recovery

The denominator is all 200 selected validation queries, with identical supplied
formula pools. Neural results are arithmetic means over seeds 42, 43 and 44.

| Method | Top-1 | Top-10 | Top-25 |
| --- | ---: | ---: | ---: |
| Uniform | 4.0% | 21.5% | 34.0% |
| Mass residual | 5.0% | 22.5% | 36.0% |
| Mean-fingerprint prior | 4.0% | 18.0% | 37.0% |
| Ridge spectrum-to-fingerprint | 6.0% | 24.5% | 45.5% |
| Spectrum/GNN contrastive ranker | 4.2% | 23.8% | 42.5% |
| Contrastive + forward-spectrum auxiliary loss | 4.2% | 23.7% | 42.2% |

Contrastive top-25 per seed: 87/200, 84/200, 84/200. Auxiliary-loss arm:
84/200, 85/200, 84/200. Standard deviations across seeds are 0.87 and 0.29
percentage points, respectively. Ridge recovered 91/200.

Paired query-bootstrap differences in top-25 recovery, averaging the three
neural seeds per query before resampling:

| Comparison | Difference | 95% interval |
| --- | ---: | ---: |
| Contrastive minus Ridge | -3.00 pp | [-8.67, +2.33] pp |
| Auxiliary arm minus Ridge | -3.33 pp | [-9.00, +1.83] pp |
| Auxiliary arm minus contrastive | -0.33 pp | [-1.50, +0.83] pp |
| Contrastive minus uniform | +8.50 pp | [+3.33, +14.00] pp |

These intervals are conditional on the three fitted models and this reused
validation sample; they do not establish generalization to an untouched test
set or physical stability. Neither comparison against Ridge establishes a
significant advantage or disadvantage. The auxiliary-loss comparison likewise
does not establish an effect.

Thirty-five pools have at most 25 candidates, so all their candidates fit in the
shortlist regardless of ranking. For the remaining 165 queries, mean top-25
recovery is 30.3% for contrastive and 29.9% for the auxiliary arm. Twelve of 200
queries have no scored supported graph candidates and remain in every
denominator with deterministic fallback ranking. Results are raw retrieval
recovery, not calibrated accepted predictions.

## Training and provenance

- Hardware: Colab Tesla T4, 15,360 MiB, NVIDIA driver 580.82.07; PyTorch
  2.11.0+cu130, Python 3.13.15. CLI session: `mamba-ms2-t4`.
- Data: pinned MassSpecGym revision
  `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`; table and 64 MiB pool-prefix
  byte counts and SHA-256 checked before execution.
- Training: 5,000 capped molecules; 3,269 usable fit molecules and 1,036
  held-out training calibration molecules after chemistry and duplicate-content
  exclusions. Connectivity groups are disjoint; scaffold disjointness is not
  established. Test-fold rows were scanned but never fit or evaluated.
- Each of six neural fits: width 128, three message-passing layers, batch 32,
  20 epochs, FP32, AdamW at 3e-4. The auxiliary weight is fixed at 0.2.
  Checkpoint selection uses minimum held-out training calibration loss.
- Validation outcomes did not change architecture, hyperparameters, or seeds.
  Seeds 43 and 44 were declared before the seed-42 validation result.
- Ridge and cheap baselines were rerun, rather than copied from old reports;
  query IDs and pool sizes were asserted identical for comparisons.

## What remains unverified

This experiment trains a candidate ranker, not the Mamba molecular-completion
decoder. It does not use predicted substructure graphs or a dedicated learned
molecular prior. Training negatives are in-batch molecules, not deliberately
matched same-formula isomers. These omissions mean this pilot is not a complete
test of the proposed architecture.

Formula pools are supplied by the benchmark: formula inference and candidate
generation are outside the evaluation. The binned spectrum representation
differs from pretrained FPNet. Missing energy-count information is represented
as unknown, and stereochemistry is absent from the graph encoder. Stability
labels and a defined stability endpoint are absent, so chemical persistence
and physical stability cannot be claimed from these results.

A useful next experiment would add same-formula training negatives and real,
out-of-fold predicted substructures, then evaluate an untouched dataset.
The current results do not justify replacing Ridge or promoting the auxiliary
loss. Any stability head needs independent endpoint-specific labels.

## Validation, fixes and saved artifacts

All nine pilot checks passed on the T4, including actual CUDA training and
CPU/CUDA graph-encoder parity. Final local regression: 107 passed, one CUDA
skip. No Rust API changed; the pilot is standalone PyTorch.

The first benchmark attempt failed before its first completed epoch because
the graph encoder omitted the shared parser's aromatic bond code 4. Four
bond channels and a benzene regression test fixed the root cause. Successful
runs all use the corrected source, with source hashes in each summary.
Notebook subprocesses now stream and persist stderr/stdout for actionable
CLI failures. The runtime was released after successful artifact retrieval.

Artifacts: [aggregate summary](../experiments/molecular_completion/20261004_neural_t4/artifacts/multi_seed_summary.json),
[results archive](../experiments/molecular_completion/20261004_neural_t4/results.zip).
The archive contains six neural checkpoints, per-query ranks, histories,
GPU/environment records, baseline outputs, and execution logs. All ZIP entries
passed CRC validation. Archive SHA-256 matches the remote file:
`80aa4e4a850085c262fdf57626d90d65e66b1e0730a6e64c67f8d4526fd5df0c`
(20,403,127 bytes).
