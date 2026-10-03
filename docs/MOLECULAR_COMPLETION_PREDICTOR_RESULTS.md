# Trained spectrum-to-fingerprint predictor: formula-pool ranking (bounded val slice)

Run date: 2026-10-03. Source: `tools/ms2_fp_predictor.py` (new; shared
`ms2_*` modules unchanged — all preexisting hashes verified identical).
Plan: [database experiments](MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md).
Prior result driving this run:
[spectral NN ranking](MOLECULAR_COMPLETION_SPECTRAL_RESULTS.md) — TRAIN
reference coverage is 5% of val queries and 0% of val targets, so no
reference-similarity scorer can rank val pools; the benchmark's own next
direction is a small trained spectrum→fingerprint predictor evaluated on the
IDENTICAL pools with honest denominators. This run implements exactly that.

Honest headline: on the 200-query val slice the trained predictor's raw
all-query top-1 (12/200 = 0.060) versus uniform/prior (8/200 = 0.040) is a
small exploratory difference of +4 hits — not evidence of a statistically
demonstrated improvement. The predeclared fixed 90%-precision gate fails
closed (abstain-all, 0/200 accepted) on held-out TRAIN calibration, so no
calibrated claim is made. True-eligible (target-parseable, n=156) and
capability (≥1 parseable candidate, n=188) numbers are reported separately
below with checkpoint/collision ceilings; no claim of "no model signal" is
drawn from aggregates alone (calibration carries a small in-domain signal,
see below).

## Method (declared representation, fixed a priori)

- Spectrum: 1.0 Da bins over [0, 2000) = 2000 dims; per-bin max intensity,
  sqrt transform, per-spectrum L2 normalization; sparse CSR. Fixed bounds,
  no huge dense tensors, per-sample transform only (no fitted vocabulary).
- Fingerprint: fixed stable-hash atom / typed-bond / length-2-path counts,
  dim 256, `zlib.crc32` (deterministic, never Python-hash),
  permutation-invariant by construction (multiset counts only). A LOCAL fixed
  hash fingerprint — NOT a standard Morgan/ECFP or official benchmark
  fingerprint; no such claim is made. Coarse counts alias isomers: collisions
  measured below and the ceiling admitted.
- Target fingerprint from each TRAIN molecule graph only; candidate
  fingerprint from each CANDIDATE graph at inference, via the existing
  `standardize_smiles` typed-graph parser with explicit domain exclusions.
  Query graphs, formula labels, IDs, target-in-pool flags and supplier order
  are never model inputs (query spectra only; measured `parent_mass` used
  solely by the declared `massresid` baseline, as given query evidence).
- Model: sparse multi-output Ridge, alpha=1.0, solver lsqr, tol=1e-3,
  max_iter=200, fit_intercept=False — fixed a priori, fit on FIT molecules
  only. No hyperparameter search on val; no refit on calibration.
- Scoring: cosine between L2-normalized predicted fp and L2-normalized
  candidate fp (signed Ridge output; negative scores allowed). Missing or
  non-finite scores rank in seeded deterministic tail (`zlib.crc32(qid)`,
  same convention as `ms2_rank_abstain.py` / `ms2_spectral_rank.py`),
  retained in every denominator, never dropped. Confidence margin is the
  top1–top2 gap over ACTUAL FINITE SCORED candidates only (None/NaN/inf
  excluded); a single finite score yields a conservative 0.0
  (`single_scored`); no finite scores yields null (unrankable).
- Inference capability (shared calibration/val predicate): nonempty usable
  spectrum features + finite nonzero prediction (zero/NaN/out-of-range
  never accepted, even at tau=0) + ≥1 finite candidate score + finite
  margin. Capability deliberately does NOT consult target parseability,
  identity or hit (those are not inference inputs); target-parseable is an
  EVALUATION-only subgroup. Non-capable queries are explicit
  `excluded_no_features` / `predictor_unavailable` (val) or
  `rankable=False` (calibration), never silently scored.
- Arms on IDENTICAL full distinct formula pools, same query order
  (VAL-0000..VAL-0199), no self-first bias: `uniform` (seeded shuffle),
  `massresid` (existing `candidate_residuals` — same function as the
  spectral run, so scores are directly comparable), `predictor` (trained,
  uncalibrated all-pool ranking), `prior` (untrained TRAIN-fit-mean
  fingerprint, same scoring — isolates spectral signal from popularity).
- Identity: exact provider-canonical SMILES strings for dedup/grouping/audit
  plus InChIKey-14 connectivity groups for split disjointness. No RDKit.
  Val queries lacking a connectivity key are retained as
  `excluded_identity_missing` with pool accounting, abstain, and no
  predictor result (observed 0/200 in this slice; file-backed regression
  test proves no output shrinking).

## Sample, caps and selection (persisted)

- Val: EXACT same 200 queries as the spectral run — file-order prefix
  entries joining first-row-per-SMILES val groups, first 200; distinct pools
  deduplicated in supplied order. Verified byte-level: 200/200 qids, SMILES
  and `n_pool` identical to `spectral_rows.csv`; `rank_uniform` matches on
  all 200 rows; total distinct pool occurrences 30,289 (identical).
- Train cap: 5000 distinct SMILES molecules (one spectrum per molecule, file
  order), group-capped by InChIKey-14 first-appearance order; 80/20 group
  split → 2994 fit groups / 749 calibration groups. After exclusions:
  fit 3269, calibration 1036 molecules (see leakage audit).
- Test fold: 17,556 rows scanned-but-unused (never fit, scored or selected);
  train rows scanned 194,119.
- Bias (selected 200, same as spectral run by construction): 194 [M+H]+ /
  6 [M+Na]+; 113 QTOF / 87 Orbitrap (val overall 76% [M+H]+, 13% QTOF —
  file-order slice bias, reported not corrected).
- Pins: TSV sha256 `50cfdd1d…06a` (261,197,365 bytes), prefix64 sha256
  `6066de3b…a9a6864` (67,108,864 bytes), code sha256 in summary. 0 download
  bytes, 0 API calls.

## Leakage audit (enforced, all counts published)

- Train/val connectivity overlap excluded: SMILES or InChIKey-14 match →
  train molecule excluded (observed 0 — folds are structure-disjoint;
  residual overlap would abort via assertion).
- Duplicate exact canonical spectrum content (`content_fp`, rounded
  mz/intensity pairs): fit first-wins (7 excluded), calibration duplicating
  fit content (2 excluded), train duplicating any val query content (0).
  Same-molecule spectra never straddle fit/calibration (one spectrum per
  molecule; split by connectivity group — asserted zero shared groups).
- Missing connectivity keys excluded (train 0 in capped set; val 0).
- Spectrum-unparseable train rows excluded with reasons (counts in summary).
- Unsupported chemistry (target fp needed for Y) excluded with reasons:
  unsupported_charge 178, unsupported_element 269, hypervalent 237,
  bracket_valence 2 (total 686 of capped 5000; fit 3269 / calib 1036 remain).
- Fit groups hash `dc15367000b07bc4`, calibration groups hash `ceba8b4230b09a38` (first 16 hex in
  `predictor_summary.json`); no shared groups asserted. Fit/calibration/val
  SMILES disjointness asserted.
- Feature provenance: spectrum bins fixed, fingerprint hash fixed, Ridge
  hyperparameters fixed — no vocabulary, statistics or thresholds fit on
  val/test. Calibration used held-out TRAIN molecules only; final model was
  NOT refit on calibration.

## Calibration (held-out TRAIN pools only; fail closed as designed)

- Calibration queries: 1036 calibration molecules, of which 897 have a
  formula pool in the 64 MiB prefix (139 `no_pool_in_prefix`, admitted as
  limited prefix coverage — identical ranking/pool domain otherwise: full
  pools, seeded ties, missing-fp tail). Rankability uses the shared
  inference-capability predicate only (897 rankable); target parseability
  is recorded evaluation-only and never gates calibration.
- Calibration ranking (rankable n=897): predictor top-1 0.040 (36/897),
  top-3 0.081, top-10 0.194; prior top-1 0.013, top-3 0.033, top-10 0.086;
  uniform top-1 0.011, top-3 0.021, top-10 0.062. The predictor carries a
  small in-domain signal (~3–4× uniform) at low absolute level — reported
  as an exploratory observation, not a validated guarantee.
- Gate grid (margin tau × pool-size kmax) at 90% target precision: best
  achievable precision 0.286 (tau=0.05, kmax=50, 7 predicted, 2/7 correct).
  No threshold meets 0.90 → FAIL CLOSED: `abstain_all` (tau=null,
  strict-JSON null, kmax=0, reason `no_threshold_meets_target`,
  n_eligible=897; serialized with `allow_nan=False`, no `Infinity`).
  Accepted val predictions: 0/200 (coverage 0.0). The uncalibrated all-pool
  ranking below is reported separately from accepted predictions, as
  required. A small calibration empirical precision is not equated with a
  confidence guarantee.

## Denominators (val, 200 selected, identical formula pools)

- Selected 200, complete 200, spectrum-excluded 0, identity-missing 0.
  Pool recall (all selected, full pools) 200/200 = 1.0.
- Every row persists `target_parseable` (bool) + `target_fp_status/reason`
  (evaluation-only) alongside capability fields (`rankable`,
  `rankable_reason`, `has_features`, `pred_norm`, `n_scored_finite`,
  `margin_pred/margin_status`). Val rankable 188, model-failure
  (complete but not rankable) 12 — all 12 are zero-parseable-candidate
  pools, i.e. `predictor_unavailable`, never silently dropped.
- Candidate chemistry: 6520/30,289 pool occurrences (21.5%) have no
  fingerprint (unparseable by the typed parser) — ranked in seeded tail,
  retained and counted. 12/200 queries have zero parseable candidates;
  capability_any_candidate (≥1 parseable) n=188.
- Target chemistry (EVALUATION-only): 44/200 val targets unparseable (no
  target fp; ranking unaffected — query graphs are never inputs — but
  collision analysis marks them). True-eligible
  (present + target-parseable) n=156; all four arms are compared on this
  exact subgroup. Of 156 parseable targets, 132 have a unique fp in their
  pool and 24 (15%) collide with ≥1 other candidate (exact fp equality
  including the target) — the representational ceiling of the 256-dim hash
  is admitted. Checkpoint: predictor_model.npz (Ridge 256×2000 float64 +
  256-dim prior) persisted with per-query rows.
- All-selected method failures: predictor produced usable finite scores on
  188/200 (12 `predictor_unavailable` with abstention); calibrated
  acceptance 0 with fail-closed abstention. Eligible present/capability/
  target-parseable denominators reported separately below.

## Ranking (val)

All-selected (n=200, full pools):

| Arm | top-1 | top-3 | top-10 |
|---|---|---|---|
| uniform | 0.040 (8/200) | 0.095 (19/200) | 0.215 (43/200) |
| massresid | 0.050 (10/200) | 0.105 (21/200) | 0.225 (45/200) |
| predictor (uncalibrated) | 0.060 (12/200) | 0.115 (23/200) | 0.245 (49/200) |
| prior (fit-mean) | 0.040 (8/200) | 0.075 (15/200) | 0.180 (36/200) |

Eligible_parseable i.e. TRUE-eligible (n=156, present + target-parseable;
all arms on this exact subgroup):

| Arm | top-1 | top-3 | top-10 |
|---|---|---|---|
| uniform | 0.026 (4/156) | 0.064 (10/156) | 0.173 (27/156) |
| massresid | 0.038 (6/156) | 0.090 (14/156) | 0.212 (33/156) |
| predictor | 0.051 (8/156) | 0.103 (16/156) | 0.237 (37/156) |
| prior | 0.026 (4/156) | 0.051 (8/156) | 0.154 (24/156) |

Capability_any_candidate (n=188, ≥1 parseable candidate; inference
capability, NOT the true-eligible denominator):

| Arm | top-1 | top-3 | top-10 |
|---|---|---|---|
| uniform | 0.021 (4/188) | 0.069 (13/188) | 0.170 (32/188) |
| massresid | 0.032 (6/188) | 0.080 (15/188) | 0.181 (34/188) |
| predictor | 0.043 (8/188) | 0.090 (17/188) | 0.202 (38/188) |
| prior | 0.021 (4/188) | 0.048 (9/188) | 0.133 (25/188) |

Accepted (calibrated gate): n=0, all arms null, coverage 0.0 — the gate
abstained everywhere rather than emit low-precision predictions.

- Paired target ranks (val): all-200 predictor better on 94, uniform
  better on 87, tie on 19; true-eligible-156 predictor better on 94,
  uniform better on 58, tie on 4. The +4-hit all-query edge
  (12 vs 8 top-1) is noise-scale on 200 queries — reported as an
  exploratory difference, not a demonstrated improvement.
- Predictor vs prior: +0.020 top-1 / +0.040 top-3 / +0.065 top-10
  all-selected — a marginal edge over pure fingerprint popularity, far below
  any useful ranking level. No "no model signal" claim is drawn from these
  aggregates alone: calibration shows a small in-domain edge (0.040 vs
  0.011 uniform) at low absolute level, and the checkpoint/collision
  ceilings above bound what this representation can show.
- Missing fingerprints never shrank a pool: all 200 queries stay in every
  `all` denominator with full pools.

## Costs

- Wall 38.5 s / CPU 38.4 s (cpu/wall 0.998 on 16 observed cores;
  OMP/OPENBLAS/MKL best-effort capped to 1 at start — observed ratio
  reported, no strict single-threaded claim), timer captured AFTER env
  imports and all row-CSV + checkpoint writes, BEFORE summary-JSON
  serialization (declared scope: full experiment incl. TSV scans, Ridge
  fit, fingerprinting 5000 train + ~30k pool SMILES, sha256, CSV/model
  writes; only the summary-JSON write itself falls outside, ms-scale).
  Final timings persist in the saved JSON, not stdout-only. Ridge fit
  alone 1.03 s (lsqr on 3269×2000 sparse, nnz 76,244 → 256 outputs).
  Peak RSS 2,792,276 KiB (~2.66 GiB), measured for the full process.
  Allocation attribution was not profiled, so its causes are unverified.
  Persisted outputs ~4.9 MB (model npz 4.1 MB = 256×2000 float64 coef +
  256-dim prior; CSVs ~0.76 MB; summary 15.6 KB with byte accounting for
  every output incl. summary_bytes). CPU budget ≤10 min met. No truncation
  or search budgets apply (no combinatorial search). Downloads 0 bytes,
  API calls 0 (persisted). torch deliberately NOT imported (not used;
  `env.torch="not_imported_by_design"`); numpy 2.4.6 / scipy 1.18.0 /
  sklearn 1.9.0 suffice and are recorded.

## Tests (121 total across all eight suites, exit 0)

- 31 in `tools/ms2_fp_predictor.py`: fingerprint determinism, permutation
  invariance, structural distinction, no-Python-hash; prediction-path
  leakage (scores never from query structures; evaluation-only target use);
  group/content overlap rejection; missing-fp tail retention + absent-target
  Nones + full denominators; seeded ties + fit-mean prior; calibration fail
  closed (no-threshold and insufficient-eligible) + strict-JSON gate
  (tau null, `allow_nan=False`, no `Infinity`); fit-only preprocessing
  (fixed per-sample transform, bounded bins); file-backed malformed-query
  integration (bad spectrum first position + train/val overlap fixture runs
  full output writing incl. model npz); NEW repair regressions:
  target-parseable vs capability split incl. unsupported-target-with-
  parseable-candidates, identity-missing retention without shrinking,
  shared capability (zero-norm / all-missing / NaN / out-of-range /
  no-features never accepted; target-fp independence), finite-only margin
  (signed negative-score gap 0.3, single-score conservative 0.0,
  no-finite-score null), resource/strict-JSON (no torch import, null gate
  round-trip).
- All pre-existing suites unchanged and passing: completion_ambiguity 3,
  database_retrieval 11, msgym_corpus 22, chebi_corpus 13, rank_abstain 5,
  oop_enumerate 1, spectral_rank 35 (90 pre-existing + 31 new = 121).
- Command: `python3 -m unittest tools.ms2_completion_ambiguity
  tools.ms2_database_retrieval tools.ms2_msgym_corpus tools.ms2_chebi_corpus
  tools.ms2_rank_abstain tools.ms2_oop_enumerate tools.ms2_spectral_rank
  tools.ms2_fp_predictor` (exit 0).
- Per-suite exit 0 individually (each `Ran N tests … OK`).
- CPU scientific-Python baseline only: GPU execution and Rust/Python parity
  explicitly NOT APPLICABLE (no GPU tensors, no Rust port for this
  experiment; recorded as limitations, not passing checks). No performance
  optimization performed (none needed; no profiling trigger).

## Caveats

- 256-dim atom/bond/path-2 hash aliases same-formula isomers (15% of
  parseable val targets collide; 21.5% of pool occurrences unparseable) —
  a representational ceiling independent of the spectral model.
- One spectrum per molecule (first in file order) ignores
  instrument/collision-energy variation; refs pooled without
  stratification (bias counts persisted).
- File-order train cap (first 5000) and 64 MiB prefix for calibration pools
  are admitted coverage limits, not corrected.
- Calibration precision grid is empirical on 897 held-out TRAIN queries, not
  a confidence guarantee; it failed closed rather than certify 90%.
- Val top-1 gains are small in absolute counts. Paired rank changes
  (94-vs-87 all, 94-vs-58 target-parseable) are descriptive and do not
  establish either a general performance gain or absence of spectral
  signal. No formal significance analysis or independent confirmation
  was performed; the failed calibration gate prevents a high-precision
  deployment claim.

## Decision

A bounded Ridge spectrum→hash-fingerprint predictor (5000-molecule cap,
3269 fit / 1036 calib after domain exclusions, 897 calibration-rankable)
shows a small in-domain calibration signal (top-1 0.040 vs 0.011 uniform)
but only a small exploratory val difference (all-200 top-1 0.060 vs
0.040 uniform / 0.040 prior; true-eligible-156 top-1 0.051 vs 0.026
uniform / 0.026 prior), and the predeclared 90%-precision gate correctly
abstains on all 200 val queries (tau=null, strict JSON). The pools contain all 200 targets, so pool coverage is not the limiting
factor in this slice. The learned model remains weak despite bypassing
reference-spectrum coverage. Parser exclusions and fingerprint collisions
are measured limitations, but this experiment does not isolate their
contribution from the small training sample or the Ridge model itself.

Next step, if
authorized: external spectra (MassBank/GNPS) joined by explicit identity,
a wider chemistry parser, or a controlled representation/model comparison
with training-only selection. The present experiment does not establish
which change would improve generalization. Test fold stays sealed.

## Reproduction

- `python3 tools/ms2_fp_predictor.py --max-queries 200 --max-train 5000
  --out-dir experiments/molecular_completion/20261003_predictor` (exit 0;
  wall ~38.5 s; see `RUN_COMMANDS.txt`, `run.stdout.log`, `run.stderr.log`,
  `alltests.*.log`).
- `python3 -m unittest tools.ms2_completion_ambiguity
  tools.ms2_database_retrieval tools.ms2_msgym_corpus tools.ms2_chebi_corpus
  tools.ms2_rank_abstain tools.ms2_oop_enumerate tools.ms2_spectral_rank
  tools.ms2_fp_predictor` (121 tests, exit 0).
- Outputs: `predictor_rows.csv` (200 val per-query rows),
  `calibration_rows.csv` (1036 calibration rows, 897 rankable),
  `train_fit_rows.csv` (capped fit/calibration molecules with split/status),
  `predictor_model.npz` (Ridge coef 256×2000 + fit-mean prior, no pickle),
  `predictor_summary.json` (caps, hashes, groups, gate table, metrics, costs,
  env, output bytes).
