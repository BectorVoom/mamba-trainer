# Representation v2 full T4 run

## Final recovery results confirmed on 2026-10-05

The Kaggle T4 recovery completed all 100 test queries and both matched controls.
Archive SHA-256, ZIP CRC, 116 manifest files, 12 model checkpoints and four
vocabularies were verified locally. Recomputed prediction metrics match the
report; the original seven primary checkpoints retain identical hashes.

| Arm | Generation top-1 / 10 / 25 | Supplied-pool retrieval top-1 / 10 / 25 |
|---|---|---|
| Full model | 0% / 0% / 0% | 3% / 26% / 44% |
| Transformer decoder | 0% / 0% / 0% | 3% / 27% / 43% |
| No substructures | 0% / 0% / 0% | 3% / 27% / 43% |

Shuffled supplied-pool retrieval top-25 is also 44%. The benchmark provides no
measured improvement over that baseline. The full model generated completed
graphs for 12 queries, but none matched the correct connectivity. Only 15 of
100 reference structures are inside the declared generation domain; formula
top-4 recovery is 14%. Physical stability was not evaluated.

The collector initially failed its local checkpoint check because Torch was
unavailable in its interpreter. Local verification was rerun successfully;
the prior error is retained as historical evidence. The recovery itself has
no failed remote stage.

Final report:

`experiments/molecular_completion/20261005_representation_v2_kaggle_recovery/RESULTS.md`

Verification evidence:

`experiments/molecular_completion/20261005_representation_v2_kaggle_recovery/verified/local_artifact_verification.json`

The launch and initial failed-run sections below are historical.


## Recovery launched on 2026-10-05

The explicit-hydrogen encoder crash is fixed: canonical graph features remove
hydrogens retained by RDKit for bond-stereo notation before assigning
heavy-atom types. Regression cases include explicit alkene and imine
hydrogens, equivalent implicit-hydrogen inputs, CPU/CUDA encoder agreement,
and preservation of supported stereo distinctions.

The private [Kaggle T4 recovery notebook](https://www.kaggle.com/code/boomvector/ms2-representation-v2-t4-recovery-s1)
attaches the completed first notebook's outputs. Original primary checkpoints
and source hashes are retained. The recovery notebook archives the original
source snapshot as `training_sources/`, checks that each of the seven primary
checkpoints identifies that source hash and has the complete epoch schedule,
and checks the pinned data and split identities. It loads the trained primary
models directly for evaluation; it does not bypass the training protocol guard.
Evaluation reruns all 100 queries, replacing the incomplete 25-query outputs.
Matched controls train using the corrected code and the original full schedule.

Local checks passed 33 tests with two CUDA skips, including Rust grammar
parity. Six Kaggle workflow tests also pass, covering source preservation,
continuation and collector restart. On the actual T4, 30 tests passed with one
Rust-executable skip, including the explicit-hydrogen regression on CUDA.

Recovery artifacts and live progress:

`experiments/molecular_completion/20261005_representation_v2_kaggle_recovery/`

The collector now names continuations within the recovery notebook series,
resumes monitoring the latest successfully submitted segment after restart,
and limits each monitored segment to 13 hours. Local dependencies live in a
persistent cache, and caffeinate prevents idle sleep while collection runs.
A shutdown or lost network can still interrupt local collection; restart it
using the saved artifact directory. No final recovery score is available yet.


## Result confirmation on 2026-10-05

The Kaggle notebook finished, but the experiment **failed during evaluation**
after completing all primary training epochs. Query 26 triggered
`KeyError: ('H', 0, 1)` in `canonical_graph_features` during supplied-pool
retrieval. Only 25 of 100 query predictions were saved. No complete benchmark
score, matched controls, final report or final archive is available.

On those 25 queries, generation produced no valid candidates; retrieval
recorded 2 / 7 / 13 top-1 / top-10 / top-25 hits. These source-ordered partial
observations are not final benchmark scores. Seven primary model checkpoints
strictly loaded with finite weights; four vocabularies and the source snapshot
hashes were checked locally. The graph-feature error remains unresolved.

Downloaded evidence and the incomplete-run report:

`experiments/molecular_completion/20261004_representation_v2_kaggle_t4/RESULTS.md`

`experiments/molecular_completion/20261004_representation_v2_kaggle_t4/partial_artifact_verification.json`

Kaggle's COMPLETE status describes the notebook wrapper, which exits normally
to save artifacts; `run_status.json` correctly records the experiment failure.
The local collector was no longer active when checked. No further GPU session
was launched during this confirmation. The launch description below is
historical; its automatic continuation is not currently running.


## Kaggle T4 restart on 2026-10-04

The benchmark was submitted as a private Kaggle notebook, requesting
`NvidiaTeslaT4` (T4 ×2). The experiment uses one T4; its existing training
code does not distribute work across both devices. The notebook requires
CUDA and checks that the device name contains T4 before training. Live logs
confirmed two Tesla T4 devices with Torch 2.11.0+cu128. Both pinned data
downloads passed SHA-256 verification, and the remote suite passed 27 tests
with one Rust-executable skip. The supervisor has started training.

Notebook: [MS2 representation v2 T4 s1](https://www.kaggle.com/code/boomvector/ms2-representation-v2-t4-s1).

The full epoch schedule, seed, retained-measurement protocol, pinned dataset
hashes, primary model and both matched controls are unchanged. This is a
restart: the lost Colab weights could not be restored. No completed v2
identification result is available yet.

Local launch and monitoring artifacts are in:

`experiments/molecular_completion/20261004_representation_v2_kaggle_t4/`

- `kernel/`: private notebook and metadata, with current sources embedded.
- `collector_status.json`: current Kaggle status and active segment.
- `kaggle_live.log`: notebook logs downloaded while the run is active.
- `collector.log` and `collector_process.json`: detached local collection process.
- `segment_N/`: outputs downloaded after each Kaggle session finishes.
- `run_status.json`: experiment status recovered from the latest finished segment.
- `verified/` and `RESULTS.md`: created only after final archive verification.

The notebook ends each unfinished session after 10.5 hours of supervised work,
preserving the last completed epoch's weights, optimizer and RNG states in
Kaggle outputs. The local collector downloads those outputs and launches the
next private T4 notebook with the previous notebook attached as an input.
Completed stages are reused; interrupted stages resume from saved epoch
checkpoints. No epoch counts are shortened. Automatic continuation is bounded
to eight segments and stops on an experiment failure.

Kaggle exposes live logs, but output files are persisted at session completion;
this cannot back up every epoch off a running VM as the Colab collector did.
A sudden VM loss before a session finishes can still lose that session's
progress. The local collector must remain running for automatic continuation.

On full completion, collection checks archive SHA-256, ZIP CRC and manifest
hashes, then strictly loads all 12 models and four vocabularies before writing
the local verification record and results. Kaggle releases the runtime when
the notebook exits.

The first setup attempt mixed a newly installed NumPy with extensions already
loaded in Kaggle's notebook process. A subsequent virtual-environment attempt
failed because Kaggle's interpreter could not bootstrap ensurepip. Version 3
installs dependencies into a separate directory and runs the experiment in a
fresh Python subprocess using that directory. Kaggle reports dependency conflicts
for unrelated preinstalled packages; the experiment imports and its GPU tests
succeeded with the pinned dependencies. Neither failed setup reached
training. Their logs remain in the local artifacts.

Local validation passed 30 regression/collector tests with three skips, plus
four Kaggle workflow tests. Rust/Python grammar parity was then run separately
with the existing Rust executable and passed. The remote supervisor runs the
CUDA tests before training; live logs and `launch_verification.json` record
the successful Kaggle checks.

Rebuild the launch notebook:

```sh
python3 tools/ms2_kaggle_prepare.py   --out experiments/molecular_completion/20261004_representation_v2_kaggle_t4/kernel
```

## Historical Colab launch


The full benchmark was resumed on 2026-10-04 using the Colab CLI session
`ms2-representation-v2-full-t4`, an actual Tesla T4 with Torch 2.11.0+cu130.
This document records the launch. Live status is in `run_status.json`; final
scores will be written to `RESULTS.md` in the local artifact directory below.
No new identification score was available at launch.

## Interruption observed on 2026-10-04

The original runtime disappeared. The last downloaded log reached conditional
training epoch 7 of 15; no final archive or verified benchmark results were
recovered. The original collector copied logs but did not copy intermediate
model checkpoints off the VM. That omission made the remote epoch checkpoints
unrecoverable after runtime loss.

The collector now copies completed models and newly completed epoch checkpoints
into a sibling `_checkpoints` directory while training runs. Atomic local
transfers retain the last good checkpoint if a download fails. Five collector
tests pass, including epoch backup and interrupted-transfer recovery. This fix
does not recover the previous run's weights. A restart was attempted, but Colab
T4 allocation returned `Service Unavailable`; no replacement run started.

The pinned dataset audit retained 54,665 measured training spectra across 4,953
connectivity identities, 945 calibration spectra, and the same source-ordered
100 official test queries. The earlier v1 run retained one measurement per
training connectivity. This is a changed training protocol, not an isolated
ablation of one representation component.

The upstream predictors train on retained spectra; motif vocabulary mining and
molecular-prior training count each connectivity once. Same-formula candidate
sets are shared across measurements of a graph. Conditional training updates
the spectrum encoder, graph encoder and decoder using the paired candidate loss.
Reranker training selects shuffled measurement records before applying its
1,500-group cap, rather than taking a contiguous block of repeated measurements.

All components retain the full epoch schedule: upstream three folds plus final
model at 8 epochs each, molecular prior at 10 epochs, conditional model at 15
epochs, and reranker at 30 epochs. Seed is 42 and batch size is 16. Matched
Transformer-decoder and no-substructure controls follow the primary model.
Generation uses at most four inferred formulas and 32 trajectories per formula.
No quantum calculations or physical stability labels are used.

Profiling measured the batched same-formula loss at 0.3318 seconds versus 1.7149
seconds for the original per-query loop on T4, with identical loss values.
CPU profiling also identified repeated molecule parsing and grammar replay;
bounded immutable caches and compact cached teacher examples remove repeated
work. Tests verify peakwise/vectorized feature equivalence, batched/unbatched
loss and gradient equivalence, input distinctions, and exact epoch recovery.
The latest 28-test suite passed locally (26 passed, two CUDA skips) and on T4
(27 passed, one Rust-executable skip). Rust grammar parity passed locally.

The detached supervisor runs the primary training, component diagnostics,
correct-spectrum/substituted-spectrum audit, both matched controls, retrieval
audit, and final report. Same-formula spectrum donors and unmatched-formula
donors are reported separately. Epoch checkpoints include optimizer and RNG
states. Completed stages can be restored without retraining; changed source or
training data is rejected by the checkpoint protocol guard.

Local artifacts and progress:

`experiments/molecular_completion/20261004_representation_v2_full_t4/`

- `run_status.json`: periodically downloaded remote stage status.
- `training.log`: periodically downloaded training progress.
- `collector.log` and `collector_process.json`: detached local collector status.
- `preflight.json`: dataset audit and measured T4 profiling results.
- `local_tests.log`: local regression and Rust grammar parity evidence.

On completion, the collector downloads the ZIP, checks SHA-256, ZIP CRC and
manifest hashes, strictly loads all 12 model checkpoints and four vocabularies,
writes `local_artifact_verification.json` and `RESULTS.md`, and then stops the
T4 runtime. A failed remote stage is recorded explicitly; the collector does
not stop a failed runtime before diagnosis. The source snapshot and historical
v1 artifacts remain separate.

The neural representation remains a Python implementation, and the graph
decoder produces connectivity without stereo assignments. Rust parity covers
the unchanged connectivity grammar. Identification results cannot validate
physical stability.
