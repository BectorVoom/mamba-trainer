# MS2 neural ranking T4 pilot

Status: trained through the authenticated Colab CLI on a Tesla T4 on 2026-10-04,
using seeds 42, 43 and 44. See [measured results](MS2_NEURAL_T4_RESULTS.md).
The pilot did not outperform Ridge; physical stability remains unevaluated.

Open `notebooks/ms2_neural_ranker_t4_colab.ipynb` in Google Colab, select
Runtime → Change runtime type → T4 GPU, and run all cells. The notebook embeds
its Python sources, so these uncommitted files do not need publishing. It
downloads the repository-pinned MassSpecGym revision, verifies byte counts and
SHA-256 hashes, checks actual CUDA training and CPU/CUDA graph-encoder parity,
and saves a downloadable results archive. Installation/download failures stop
the run. Downloads have timeouts and bounded retries.

The experiment compares a spectrum/GNN contrastive retrieval model against the
same model with a fixed 0.2-weight neural forward-spectrum auxiliary loss.
Graph message passing distinguishes single, double, triple and aromatic bonds and pools
without dependence on atom order. Spectrum features reproduce the documented
FPNet filter, floor, 160-peak cap and square-root normalization, then use fixed
1-Da bins; this is not FPNet or its pretrained representation. Inputs include
precursor m/z, a fit-only adduct vocabulary, polarity, collision energy and its
known flag, and energy count. The benchmark does not supply energy count, so it
is unknown (0). Percent or otherwise unparseable collision energies remain
unknown; numerical collision energies follow the benchmark's eV convention.

Both neural arms use 5,000 capped training molecules, batch size 32, width 128,
20 epochs, seed 42, FP32 and AdamW at 3e-4. Checkpoint selection uses mean
held-out training calibration loss. It uses no validation target in selection.
The existing Ridge, mean-fingerprint prior, mass-residual and uniform baselines
run on identical pools. Comparisons assert matching query IDs and pool sizes.
Top-1/10/25 and paired query-bootstrap top-25 intervals are saved. The default
200-query slice has already been inspected in prior experiments; results are
exploratory, not an untouched test claim. Run multiple predeclared seeds and a
new evaluation set before making a general accuracy claim.

Training/calibration reuse connectivity-group and spectrum-content leakage
exclusions from `ms2_fp_predictor.py`, with additional empty-feature exclusions.
They do not establish scaffold disjointness. One spectrum is retained per
molecule. Candidate graphs use the existing restricted neutral typed parser;
unsupported candidates remain in a deterministic seeded tail. All selected
queries remain in the denominator, including queries without usable scores.
Fallback hits are raw retrieval results and do not indicate accepted confident
predictions. The 1-Da representation loses high-resolution mass information.
Formula pools are supplied by the benchmark; this pilot does not evaluate
formula inference from precursor/adduct inputs.
The graph encoder does not represent stereochemistry.

## Proposal verification boundary

This pilot tests learned graph/spectrum alignment and the benefit of neural
forward-spectrum supervision. It does not implement the Mamba completion
decoder, same-formula training negatives, predicted-substructure conditioning,
a dedicated pretrained molecular-plausibility prior, confidence calibration,
or physical-stability prediction. Those proposal components remain unverified.
No quantum calculations are performed. Physical stability requires a defined
endpoint and stability labels; benchmark molecule–spectrum pairs do not provide
them. Predicted-substructure conditioning also needs upstream predictions with
confidence and provenance, generated without training on the evaluation targets.

## Local validation

```sh
.venv-ability/bin/python -m unittest tools.test_ms2_neural_ranker -v
```

Seven CPU tests passed, including a file-backed training/evaluation/checkpoint
smoke run, training-loss improvement, atom-permutation/padding invariance,
missing-energy semantics, invalid inputs, unsupported-candidate accounting and
paired-bootstrap determinism. One actual CUDA training/parity test was skipped
because this host has no CUDA device. The Colab notebook runs that same test
on its allocated GPU before training. No Rust API is introduced or modified;
Rust/Python API parity does not apply to this standalone PyTorch pilot.

The synthetic smoke run verifies execution only; it is not scientific evidence
of molecular prediction accuracy. Actual benchmark training results are in the
linked report. A Colab training failure exposed a missing aromatic-bond channel;
this was fixed with a regression test before any successful benchmark fitting.

The broader local run covering this pilot plus the fingerprint, spectral,
corpus and retrieval helpers ran 107 tests: 106 passed and one CUDA test was
skipped. SciPy was initially absent; the 1.15.3 macOS wheel then failed to load
its PROPACK extension. Installing SciPy 1.16.3 into a temporary dependency
directory resolved the local environment failure. Colab's Linux dependency
pin remains 1.15.3. Existing spectral-helper tests emit unclosed-file resource
warnings; these are recorded as preexisting test cleanup debt, and do not
affect the passing assertions.

After the aromatic fix, the local regression run passed 107 of 108 tests with
one unavailable-CUDA skip. All nine pilot tests passed on Colab, including
actual CUDA training and CPU/CUDA parity.

## Colab CLI

The installed CLI's bundled manual is available with `colab readme`. It was
already authenticated; no new credentials were requested or stored by this work.
To run the notebook's default seed-42 pilot without a browser download:

```sh
colab new -s mamba-ms2-t4 --gpu T4
colab exec -s mamba-ms2-t4 --timeout 7200 --env COLAB_CLI_RUN=1 \
  -f notebooks/ms2_neural_ranker_t4_colab.ipynb
colab download -s mamba-ms2-t4 /content/ms2_neural_t4_results.zip ./results.zip
colab stop -s mamba-ms2-t4
```

The measured run additionally invoked the same trainer with `--seed 43` and
`--seed 44` and separate `--out` paths. The plan, per-seed predictions,
checkpoints, logs and aggregate paired comparisons are saved under
`experiments/molecular_completion/20261004_neural_t4/`.
