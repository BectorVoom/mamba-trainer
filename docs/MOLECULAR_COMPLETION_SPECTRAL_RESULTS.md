# Measured-spectrum ranking: MassSpecGym NN retrieval (bounded val slice)

Run date: 2026-10-03 (repaired rerun; first run same day). Source:
`tools/ms2_spectral_rank.py`.
Plan: [database experiments](MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md).
Prior decision driving this run:
[database results, fourth run](MOLECULAR_COMPLETION_DATABASE_RESULTS.md) —
cheap metadata features cannot separate same-formula isomers (top-1
0.14/0.06 train/test, calibrated gate abstains almost everywhere), so the
located gap is *spectral ranking*, not structure generation.

This is the first HONEST measured-spectrum ranking in the series: earlier
runs reported spectral similarity as `unavailable_no_spectrum`. The pinned
TSV ships one measured spectrum per row (`mzs`, `intensities`), so this run
bins those spectra and scores supplied formula-pool candidates by cosine
against TRAIN-fold reference spectra of the same structure. Nothing is
fitted on val; nothing is fabricated for candidates without references.

An independent review of the first pass found four defect classes
(exclusion accounting, unenforced audit, control overclaim, documentation
errors). All are repaired here; changed numbers and weakened claims are
marked REPAIRED below. No scoring was tuned on val outcomes at any point.

## Method (declared representation, fixed a priori)

- Bins: 0.1 m/z over [0, 2000) = 20,000 bins; per-bin max intensity;
  L2-normalized sparse vectors; cosine similarity.
- Candidate score = max cosine over that candidate's TRAIN reference
  spectra; `None` when the candidate has no TRAIN reference.
- Identity: exact provider-canonical SMILES string equality. Sound (equal
  strings => same structure) but conservative: stereo/tautomer variants
  written differently do not match, so coverage is a LOWER BOUND on true
  connectivity coverage, NOT a graph-canonicalization guarantee. RDKit is
  not installed (no new dependencies); a crude no-stereo character count
  is a rough sensitivity approximation (neither an upper nor a tight
  bound), never used for ranking.
- Abstention is parameter-free: predict iff a complete row has >= 1
  referenced candidate. No thresholds exist, so train-only threshold
  calibration is NOT APPLICABLE (stated, not skipped silently).
- Arms on IDENTICAL formula pools, no oracle subgraphs (primary condition):
  `uniform` (seeded shuffle, `zlib.crc32(qid)` — same convention as
  `tools/ms2_rank_abstain.py`), `spectral` (TRAIN max-cosine; unreferenced
  candidates trail in shuffle order, never dropped), `coverage`
  (REPAIRED, new: coverage-only baseline — seeded shuffle among
  reference-backed candidates first, then unreferenced ones; uses no
  spectral score, isolating what reference PRESENCE alone buys),
  `massresid` (replacement cheap exact baseline for the formula-pool
  condition — NOT a head-to-head rerun of the prior learned ranker, whose
  oracle subgraph features do not exist here: rank by
  |graph-theoretical mass − query `parent_mass`| via the existing SMILES
  standardizer; unparseable candidates rank last and are counted).
- Leakage controls (REPAIRED — audit now ENFORCES): queries from val only,
  references from train only; train spectra matching a query SMILES are
  excluded and counted; val train/query InChIKey-14 overlap ABORTS the run
  (fail safe; observed 0) and queries with a missing SMILES/connectivity
  key are excluded conservatively before scoring (observed 0);
  same-CONTENT spectra are excluded across identifiers (query-content guard
  per query; reference content-dedup within structure, first wins; both
  counted); the target is never injected; test-fold rows pass through the
  shared TSV scanner but are never used for fitting, scoring, or
  selection.
- Row schema is uniform across complete/excluded rows (writer also unions
  fields as belt-and-braces); spectrum-free ranks (uniform, massresid) are
  still computed for spectrum-excluded queries and documented as such.

## Sample and pins

- Pins (verified, match `data/pinned/PIN.json`): TSV sha256
  `50cfdd1d…06a` (261,197,365 bytes), prefix64 sha256 `6066de3b…a9a6864`
  (67,108,864 bytes). 0 download bytes, 0 API calls beyond pinned files.
- Val: 200 queries = first 200 file-order prefix entries joining a val row;
  one spectrum per structure (first TSV row in file order). Train corpus: all
  194,119 train rows scanned; only spectra whose SMILES occurs in the
  selected pools retained (248 spectra / 24 structures after content
  dedup — explicitly bounded; 9 same-content duplicates removed, counted).
- Sample bias (file-order slice vs all val spectra): selected is 97%
  [M+H]+ (val overall 76%) and 57% QTOF (overall 13%); Orbitrap 44% vs 85%.
  Reported, not corrected — a randomized slice is future work.
- Supplied pools used verbatim (deduplicated to distinct strings; target in
  pool 200/200 by exact string — a pool property, not a method result).

## Denominators (val, 200 selected, formula pools, no oracle subgraphs)

- Selected 200, complete 200, spectrum-excluded 0, missing-key-excluded 0.
- All-query pool recall 200/200 (computed over ALL selected queries,
  including any excluded spectra — here trivially complete).
- Leak exclusions (train spectrum matching a query SMILES): 0.
  Query-content guard exclusions: 0 refs. Missing train keys: 0.
- Candidates with a TRAIN reference: 10/200 queries have >= 1 (5.0%);
  0/200 targets have a reference (`eligible_targetref` n = 0).
- Persisted coverage detail over the SELECTED 200 only (REPAIRED — the
  earlier draft mixed an unpersisted full-349 scan with the 200-sample):
  30,289 distinct candidate occurrences, 25 exact-covered (0.08%),
  54 further no-stereo-approximate matches. Formula pools are same-formula
  isomers of structures the train fold never measured — reference coverage
  is sparse under exact-string matching and the stated no-stereo
  approximation; neither establishes general connectivity coverage.

## Ranking (val)

| Pool (n) | uniform t1/t3/t10 | spectral t1/t3/t10 | coverage t1/t3/t10 | massresid t1/t3/t10 |
|---|---|---|---|---|
| all (200) | 0.040/0.095/0.215 | 0.040/0.095/0.215 | 0.040/0.095/0.215 | 0.050/0.105/0.225 |
| eligible_anyref (10) | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 |
| refcount_1 (6) | 0.0 | 0.0 | 0.0 | 0.0 |
| refcount_multi (4) | 0.0 | 0.0 | 0.0 | 0.0 |

- Abstention: predicted 10/200 (coverage 0.05 over ALL selected), precision
  0.0 (0/10).
- REPAIRED wording: aggregate metric equality is NOT ranking identity —
  spectral/coverage target ranks differ from uniform on 8/200 rows;
  spectral and coverage target ranks coincide on all 200 val rows. Full
  candidate-order equality was not recorded, so this comparison concerns
  the reported target ranks and aggregate top-k metrics only.
- Missing references never shrank a pool or improved a denominator — all
  200 queries stay in every `all` denominator with full pools.
- `massresid` is at uniform level (0.05 top-1): same-formula isomers are
  mass-identical up to parser rounding, as expected. It is a replacement
  baseline, not evidence about the prior learned ranker.

## Positive control (TRAIN fold, labelled CONTROL, not a val result)

50 train spectra as queries vs remaining train references (query identifier
AND query content fingerprint excluded):

| Pool (n) | uniform t1 | spectral t1 | coverage t1 | massresid t1 |
|---|---|---|---|---|
| all (50) | 0.00 | 0.94 | 0.72 | 0.00 |
| refcount_1 (33) | 0.00 | 0.97 | 0.97 | 0.00 |
| refcount_multi (15) | 0.00 | 1.00 | 0.27 | 0.00 |

- REPAIRED claim (the first draft's "49/49 proves discrimination" was an
  overclaim): 33/50 control pools hold exactly ONE referenced candidate, so
  spectral == coverage there by construction (32/33 both; the shared miss
  is a pool whose sole referenced candidate is not the target). The honest
  evidence for the cosine VALUES is the multi-reference subgroup: 15/15
  top-1 vs 4/15 (0.27) for coverage-only. Representation quality is
  therefore established only in this narrow sense — 15 pools where several
  referenced candidates compete — not as a general ranking result, and the
  val gap remains a coverage gap first: with 0/200 targets referenced, no
  scorer of any quality can rank them by reference similarity.

## Costs (REPAIRED — timer now covers hashing + row writes)

12.8 s wall / 12.7 s CPU single-threaded (200 val + 50 control, incl. TSV
scans, sha256 verification, and row-CSV writes; only the summary-JSON
write itself, ms-scale, falls outside); peak RSS 293 MB; persisted outputs
46,062 + 11,337 bytes CSV. No truncation or search budgets apply to this
method (no combinatorial search); all 250 rows `complete`.

## Tests (REPAIRED — 90 total across all seven suites, exit 0)

35 in `tools/ms2_spectral_rank.py` (parsing/edge cases, bin edges, cosine
identity/orthogonality/scale-invariance, zero-vector 0.0,
max-aggregation, missing-refs-retained, seeded tie-breaking,
target-absent Nones, exact-string identity, leak audit + file-backed
exclusion/duplicate-id tests, NEW: content-fingerprint guards,
same-content/different-ID exclusion, content dedup counting,
overlap-abort + missing-key-exclusion enforcement, coverage-only ordering
and no-score-use, denominator/coverage/subgroup accounting, strip_stereo
approximation wording, and a file-backed integration test with malformed
queries in FIRST and later positions running full output writing), plus
all pre-existing suites: database_retrieval 11, msgym_corpus 22,
chebi_corpus 13, oop_enumerate 1, completion_ambiguity 3, rank_abstain 5
(the earlier "74" omitted the rank_abstain suite; pre-review total was 79).

CPU stdlib reference: no GPU tensors and no Rust port exist for this
experiment, so GPU execution and Rust/Python parity checks are explicitly
NOT APPLICABLE (recorded as limitations, not passing checks). No
performance optimization was performed (none needed; no profiling trigger).

## Caveats

- Coverage is an exact-string lower bound; the no-stereo count is a rough
  approximation in neither direction, not a tight bound.
- One spectrum per structure (first in file order) ignores
  instrument/collision-energy variation in the query; refs pool all
  instruments without stratification (subgroup counts in summary).
- MassSpecGym intensities are used as given (already max-normalized);
  no peak weighting was tried — any such choice must be declared a priori
  and validated on train-only controls, never on val.
- Control queries reuse the same formula pools with abundant refs; the
  multi-ref subgroup (n=15) is small — representation evidence, not proof.

## Decision

Measured spectral NN retrieval against TRAIN references does not rank val
formula pools: coverage 5% of queries, 0% of targets, 0/10 abstained
predictions correct — while the identical pipeline scores 15/15 on
multi-reference control pools (4/15 for coverage-only) and 32/33 where a
single reference exists. The gap is reference coverage, not scoring. Next
step is a small trained spectrum→fingerprint predictor (the benchmark's
own direction) evaluated on these IDENTICAL pools with the same honest
denominators — or external spectra (MassBank/GNPS) joined by explicit
identity. No generation work is justified by this slice: retrieval has not
yet had any evidence to fail on. Test fold stays sealed.

## Reproduction

- `python3 tools/ms2_spectral_rank.py --max-queries 200 --max-control 50
  --out-dir experiments/molecular_completion/20261003_spectral`
  (exit 0; wall ~13 s; see `RUN_COMMANDS.txt`, `run.stdout.log`).
- `python3 -m unittest tools.ms2_spectral_rank tools.ms2_database_retrieval
  tools.ms2_msgym_corpus tools.ms2_chebi_corpus tools.ms2_oop_enumerate
  tools.ms2_completion_ambiguity tools.ms2_rank_abstain` (90 tests, exit 0).
- Outputs: `spectral_rows.csv` (200 per-query val rows),
  `spectral_control_rows.csv` (50 control rows),
  `spectral_summary.json` (parameters, hashes, seeds, metrics, costs,
  provenance, output byte counts).
