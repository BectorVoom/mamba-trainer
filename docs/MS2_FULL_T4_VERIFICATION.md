# Full neural MS2 completion experiment on Colab T4

This report describes the archived **v1** experiment. Current code and the
notebook use representation v2; see `MS2_REPRESENTATION_FIX.md` for changes and
their validation. Historical scores below must not be attributed to v2.
Reproducing v1 requires its archived source snapshots.

This experiment exercises the implemented neural components: an upstream
spectrum-to-substructure predictor, confidence-weighted substructure GNN, a
pretrained molecular graph prior, spectrum-conditioned Mamba-3 graph generation,
neural forward spectrum prediction, and same-formula reranking. It uses no
quantum calculations. Completion is evaluated separately from retrieval.

**The full proposed training procedure was not verified.** Conditional graph
likelihood and forward spectrum prediction use measured spectrum–molecule pairs,
but same-formula negatives train only a final reranker on frozen features.
`ranking_features` runs under `torch.no_grad()` and converts its outputs to
NumPy; the ranking loss therefore cannot update the spectrum encoder, graph
encoder, or generator. No end-to-end same-formula contrastive alignment loss
was included. The spectrum–graph dot product used by the reranker consequently
has no explicit alignment objective. This is an implementation gap, not an
experimentally isolated explanation for all of the poor results. A corrected
joint training run and paired-versus-shuffled-spectrum control are still needed.

**This is an architecture and identification experiment, not a validation of
physical stability.** Observed molecules provide a plausibility prior. Graph
validity, precursor-mass agreement, and RDKit sanitization do not establish
thermodynamic stability, persistence, or stability under particular conditions.
No experimentally labeled stability endpoint was supplied.

## Implementation and scope

- `tools/ms2_full_model.py`: PyTorch Mamba-3 reference, spectrum encoder, typed
  graph encoder, confidence-weighted predicted motifs, explicit partial-graph
  memory and pointer heads, constrained generation, and forward spectrum head.
- `tools/ms2_full_experiment.py`: split audit, three-fold upstream cross-fitting,
  prior pretraining, conditional training, same-formula negatives and evaluation.
- `tools/ms2_full_controls.py`: matched Transformer **decoder** control and a
  separately trained control without substructure inputs.
- `tools/test_ms2_full_model.py`, `tools/test_ms2_full_controls.py`: CPU/GPU
  learning and numerical checks, causality, grammar closure, graph invariance,
  reference-label independence and control capacity.
- `examples/ms2_full_grammar_parity.rs`: compares Python action masks with native
  Rust chemistry/grammar source under the experiment's closure rule.

The domain is explicitly limited to connected, neutral, non-isotopic CHNO
molecules with no formally charged atoms, 2–32 heavy atoms, at most 8 independent cycles, C≥1, N≤12 and
O≤16. Stereochemistry is excluded from connectivity identity. Other elements,
molecules with formal charges (including net-neutral nitro groups/zwitterions),
and larger molecules remain in the test denominator.
The traversal is a deterministic BFS of RDKit canonical connectivity SMILES;
it is not the frozen V0 exhaustive lexicographically minimal BFS traversal.

Training uses the repository's **PyTorch Mamba-3 reference**, not native Rust CUDA
training. Native Rust/Python graph-mask parity and CPU/CUDA neural parity are
separate checks. This prototype does not provide a native Rust implementation
of every new neural component.

## Inputs and leakage controls

Peaks retain continuous m/z and sub-Dalton Fourier features after intensity
filtering, top-160 selection and square-root scaling. Spectrum encoders also use
precursor m/z, explicit adduct embeddings, polarity, collision energy with a
known flag, and energy count. The pinned dataset does not supply an energy-count
column, so its value is unknown/zero; the model supports supplied counts up to 8.
The forward head receives the graph plus these measurement conditions, including
an adduct embedding. Its output uses 2,000 one-Dalton bins, which remains a
limitation of the forward consistency score.

No provided formula or reference graph enters generation. Formula hypotheses
come from precursor m/z and adduct, integer mass arithmetic with rounding/error
bounds and a declared 20 ppm window. The neural spectrum context ranks those
hypotheses; the top four define generation budgets. A graph can finish only
with exact composition and zero residual valence, and must then sanitize.
Substructures are self-trained typed one-, two- and three-node graph patterns,
with parent hydrogen counts preserved. They are not ground-truth query motifs
or a supplied FPNet checkpoint.

The official test fold supplies the first 100 query entries joining the pinned
candidate prefix. These contain 87 distinct connectivity identities after
stereochemistry removal; validation contains 178 distinct identities across
188 retained records. Training contains no connectivity duplicates. There is
no canonical identity overlap between training, validation and test. Validation
and test share one ring-scaffold group; both groups are excluded from training.
Repeated connectivity queries require clustered uncertainty estimates. These differ from the earlier pilot's inspected validation
queries. Training excludes their canonical identities, ring scaffolds and peak
content, as well as those of the first 1,000 official validation molecules.
Acyclic compounds are grouped by connectivity because their Murcko scaffold is
empty. This is not a strict scaffold split for acyclic compounds.
After declared chemistry and mass exclusions, 4,922 training and 188 validation
molecules remain. Source-order selection and the restricted domain limit
representativeness.

Upstream predictors are cross-fitted over three scaffold groups. Each fold's
motif vocabulary is mined only from that fold's training molecules. Completion
training receives the out-of-fold predicted graphs and confidence values.
The final upstream model predicts validation/test motifs. Validation labels
select checkpoints; final test labels are used only for evaluation.

## Frozen training/evaluation settings

| Component | Settings |
|---|---|
| Upstream | Three fold models + final model, 8 epochs each |
| Molecular prior | Formula-conditioned legal graph likelihood, 10 epochs |
| Conditional generator | Graph likelihood + inferred-formula classification + 0.2 forward cosine loss, 15 epochs |
| Reranker | Same-formula observed peers and valid formula-preserving rewiring decoys, at most 4 negatives, 30 epochs |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01, gradient norm ≤1 |
| Batch / seed | 16 / 42 |
| Generation | At most 4 inferred formulas ×32 trajectories; rank up to 25 distinct outputs |

Rewiring decoys are identification negatives, not labels of chemical instability.
The prior is pretrained on the known training molecules, not on a separate
large chemistry corpus. The experiment is a single-seed prototype.

The Transformer control replaces two decoder Mamba blocks (69,730 parameters)
with one causal Transformer layer (69,766 parameters): width 96, four heads,
feedforward width 166, zero dropout. The spectrum encoder remains the same
Mamba encoder. Total parameter count differs by less than 0.01%. Prior
pretraining, conditional training, ranking, split and generation budgets match.
The no-substructure control shares the pretrained Mamba prior, uses empty
substructure inputs, and retrains its conditional generator and reranker.

Generation top-1/10/25 uses all 100 queries, including domain, mass and generation
failures. Supplied same-formula pools are used only for a separate retrieval
diagnostic; they never supply generation candidates or an oracle formula.
Paired query bootstrap intervals compare controls. A supplied-pool retrieval
hit is not evidence that the generator recovered the target.

## Verification and reproduction

Local CPU tests include native Rust action-mask parity. Colab tests check CPU,
T4 CUDA and full conditional/forward numerical parity. The Transformer control
also has causal-prefix and parameter-count checks. Profiles are archived:
CPU profiling identified repeated motif-label extraction; CUDA profiling
identified per-trajectory action-head overhead. Labels are cached as immutable
sets and generation batches conditional fields while retaining exact prefix
snapshots. These changes do not alter legal actions or available inputs.

The run uses the authenticated `colab` CLI rather than browser execution. The
self-contained notebook embeds source snapshots, pinned dependency installs,
verified data downloads, tests, training, controls and archive creation. It can
also be executed through `colab exec -s SESSION --timeout 7200 -f SCRIPT`.
The runtime must report a Tesla T4; other accelerators are rejected.

Pinned MassSpecGym revision: `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`.
TSV SHA-256: `50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a`.
The first 64 MiB of the formula candidate JSON has SHA-256
`6066de3b1f2a5c013a6d37959b0c6f2f8a4867dc2fdef07fdbe3eaceaa9a6864`.

## Results

The main pipeline was trained and evaluated on an actual Colab Tesla T4.
The full architecture did **not** demonstrate useful de novo identification in
this experiment: top-1/10/25 generation was 0/100 queries. Only 15/100 targets were
inside the declared graph domain, and none of those 15 was generated either.
There were 17 completed, sanitized, mass-consistent graphs across 13 queries;
6,223 trajectories ended without a legal continuation. No query exceeded the
step limit. The completed-trajectory fraction was 17/(6,223+17) ≈0.27%.

Supplied-pool retrieval top-1/10/25 was 3%/26%/42%. Shuffled ordering achieved
3%/27%/44%. The paired top-25 difference was −2 percentage points, with a 95%
query bootstrap interval [−5, 0] percentage points. A clustered estimate is
reported separately because connectivity queries repeat. This does not demonstrate
an identification advantage over the baseline.

On the 15 supported test molecules, conditional graph NLL/token was 1.7167,
versus 1.6847 for the pretrained unconditional prior. A post-training removal
of substructure inputs gave 1.7078; this perturbation alone is not a trained
ablation. True-graph forward spectrum cosine was 0.2022, versus 0.2035 for the
mean training spectrum. The 90 selected motif predictions had 97.78% precision,
but common local motifs were insufficient to recover complete connectivity.

These results distinguish several problems: the limited chemistry/size domain
excludes 85% of this source-ordered test selection; exact-composition sampling
frequently reaches dead ends; and conditional spectrum learning did not improve
held-out graph likelihood or forward spectrum prediction. The conditional
validation objective was best early in training and worsened thereafter,
consistent with overfitting. Checkpoint selection used validation loss.

All matched controls completed on the same Tesla T4:

| Arm | Generation top-25 | Retrieval top-25 | Queries with completed graphs | Candidates, summed per query |
|---|---:|---:|---:|---:|
| Full Mamba pipeline | 0/100 | 42% | 13/100 | 17 |
| Matched Transformer decoder | 0/100 | 43% | 12/100 | 19 |
| Mamba without substructures | 0/100 | 42% | 14/100 | 31 |
| Shuffled supplied-pool ordering | N/A | 44% | N/A | N/A |

The corrected retrieval scores use original-label domain checks. Resampling the
87 connectivity groups gives a 95% interval of [−5.21, 0] percentage points for
full-minus-shuffled top-25, and [−3.16, 0] for full-minus-Transformer. Full and
no-substructure had identical per-query top-25 retrieval hits. No arm recovered
a target by generation, so the observed generation-hit contrasts are all zero;
this does not establish equivalence or rule out differences in a larger study.

There is no demonstrated advantage from the Mamba decoder, substructure inputs
or forward spectrum consistency in this run. The completed architecture is a
failed identification prototype under this split/search/domain protocol. The
experiment neither validates nor falsifies physical stability of its outputs,
because stability endpoints were not labeled. Results are not directly
comparable with the earlier pilot's different validation query cohort.

A post-training parser audit found isotope labels in 78 raw supplied-pool
candidates, but none in the training, retained validation or test targets.
Canonical connectivity normalization could remove isotope labels before
domain validation. The parser now checks original labels and validates raw
candidates before deduplication. Regression tests cover isotope-only groups
and groups containing both an isotope-labeled and an allowed representative.
Retrieval was rechecked with unchanged weights. Eligibility/ranks changed for
10 full-pipeline queries, 9 Transformer queries and 10 no-substructure queries,
but top-1/10/25 metrics did not change. Original diagnostic predictions and
training-source snapshots are retained. Generation and its results are
unaffected by this correction.

Local verification ran 16 tests: 15 passed, with the unavailable local CUDA test
skipped. Native Rust grammar parity passed in that run. Colab ran the same 16:
15 passed, including CPU/CUDA conditional and forward parity, with native Rust
executable parity skipped because that binary was checked on the local host.
The final notebook's code cells compile and its embedded sources match the
repository files. The archive has per-file SHA-256 checksums, and the download
is checked for archive integrity, checksum agreement and checkpoint loading. The limited chemistry
domain, single seed, connectivity-only identity and absence of physical stability
labels apply to every result.

## Colab CLI reproduction

From the repository root, extract the notebook's code cells and execute them in
an authenticated CLI session:

```sh
python3 - <<'PY'
import json
from pathlib import Path
notebook = json.loads(Path('notebooks/ms2_full_t4_colab.ipynb').read_text())
code = "import os\nos.environ['COLAB_CLI_RUN'] = '1'\n"
code += '\n\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
Path('/tmp/ms2_full_reproduce.py').write_text(code)
PY
colab new -s ms2-full-reproduce --gpu T4
colab exec -s ms2-full-reproduce --timeout 7200 -f /tmp/ms2_full_reproduce.py
colab download -s ms2-full-reproduce /content/ms2_neural_pilot/ms2_full_t4_results.zip ./ms2_full_t4_results.zip
colab stop -s ms2-full-reproduce
```

Check the execution output for Python failures as well as the CLI exit code.
The actual verification run used detached training/supervisor processes to
retain work across intermittent CLI connection failures. Avoid simultaneous
CLI execution clients against the same kernel, and retry a lost connection
without stopping the training runtime. The archive records the actual Python,
Torch, CUDA device and library versions; the notebook uses the Colab-provided
Torch build rather than reinstalling a large CUDA distribution.
