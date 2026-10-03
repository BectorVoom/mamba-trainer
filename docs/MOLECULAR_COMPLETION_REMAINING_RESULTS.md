# Remaining protocol: measured results (2026-10-03, second-review repairs)

Implementation: resumed protocol per `experiments/molecular_completion/20261003_remaining/PROMPT.md`,
`RESUME_PROMPT.md`, `GO_REPAIR_PROMPT.md`; second-review defects in
`SUPERVISOR_SECOND_REVIEW.md` repaired in code and re-measured. All numbers below
are executed code with pinned inputs. No commits or repository-file deletions; `data/pinned/PIN.json`
untouched; prior logs/archives preserved. Frozen 200 ranks and model never retuned.

## A. MassBank external validation — COMPLETED (bounded, repaired)

- Source: `MassBank-data-2026.03.zip` (239,602,156 B); release tag `2026.03` =
  commit `705afb7bccc3b2c42410a744eef73674716a60ef`; Zenodo 19073053,
  DOI `10.5281/zenodo.19073053`.
- Scan: 139,240 record files; archive-order scan capped at 40,000 (`truncated_cap`,
  declared). Parser now rejects peak-count mismatches and unterminated records.
  Per-record LICENSE retained. Adduct ion masses electron-corrected
  (Na+/K+ minus 0.000548579909 Da); MassBank ION_MODE polarity cross-checked.
  Precursor text persists rounding half-width uncertainty (e.g. 3 decimals =
  500 uDa); assumed 10 ppm tolerance declared separately, never inferred.
- Selection: first 150 distinct molecules by canonical connectivity + frozen
  spectrum-content semantics; frozen USED fit/calib excluded by canonical
  identity (684 excluded — measured via actual `group` column) AND by content
  (same spectrum, different ID excluded); cross-source dedup vs GNPS.
- Prefix-pool arm (retained, separately labelled): formula pools from
  `msgym_candidates_formula_prefix64.json`; strict canonical identity (no raw
  fallback); massresid baseline from MEASURED precursor neutral (theoretical
  annotation is cross-check only); predictor ranks gated on query capability
  AND finite true-candidate score; frozen abstain-all gate applied and measured
  (accepted 0). Results (n=150, Wilson 95%): pool recall **54/150 = 0.360
  [0.288, 0.439]**; predictor top1 5/150, top25 **21/150 = 0.140** (present-only
  top25 21/54 = 0.389); paired predictor−uniform top25 +0.040. Tiers:
  all 150 / present 54 / present+parseable 54 / capability 74 /
  predictor-applicable 54.
- ChEBI full-coverage arm (required stage, pinned 59 MB SDF: 52,956 scanned,
  26,659 in-domain indexed, domain exclusions counted; 88 provider InChIKey
  disagreements disclosed, none matches a selected query under the stated
  canonical-SMILES identity policy): nested measured
  mass→formula→oracle stages (nesting verified all queries): stage-1 presence
  **76/150 = 0.507**, stage-2|stage-1 76/76, stage-3|stage-2 76/76;
  formula-pool ranking (separately labelled arm): recall **77/150 = 0.513**
  (pools tiny: mostly ≤10 members — conditional accuracy read with pool sizes).
  Stage-4 precursor `not_evaluated` (whole-parent redundancy). Per-query
  formula-pool ranks are saved in `chebi_rank_rows.csv`, with qids joining
  the source-provenance rows.
- Reference-NN arm (redesigned): separate query spectra vs 3,000 remaining
  MassBank + 126 GNPS same-source references; exact content/self/fit-calib
  excluded: eligibility **112/152**, conditional accuracy **51/112 = 0.455
  [0.366, 0.548]** — REFERENCELOOKUP only.
- Exact Murcko scaffolds (RDKit, versioned policy recorded): MB vs fit
  intersection 24.
- Cost: CPU 149.8 s, wall 150.4 s (≤10 min); per-1000 985.5 CPU-s; 0 API calls.

## B. GNPS second check — COMPLETED with measured limitation

- `GNPS-FAULKNERLEGACY.mgf` (216,958 B): 126 complete blocks;
  **124/126 `SMILES=N/A`** (counted, never scored). Whole-token adduct
  validation rejects water-loss annotations and CHARGE sign/magnitude
  conflicts (B26A11-style CHARGE=2 fixture covered); MGF NaN/negative peaks
  rejected; count metadata enforced when present. 2 structurized queries
  selected (cap 100, cross-source deduped): pool recall 0/2 (absent).
  Per-row provenance (license/contributor/accession/CE/canon/keys/adduct/
  ionmode/precision) persisted in `external_rows.csv`. License: per-block
  attribution retained; FAULKNERLEGACY import status UNVERIFIED — CC0 not
  assumed.
- val200 library lookup: MassBank/GNPS structure coverage — LIBRARYLOOKUP only.

## C. QM9 restricted control — COMPLETED (exact identity, exclusions enforced)

- Pinned archives only (0 new bytes this run; all cached). Rerun reproduces the
  independent replay exactly: eligible 239; allowed exact **81**; excluded
  geometry **60** (3,054 exclusion members visited, enforced by full-GDB9 join);
  GDB-only 1; no-match 97; full archive 133,885 complete.
- Provider `*^`/Fortran numeric notation normalized (record 212 parses).
- Linearity VALIDATED from coordinates (SVD, documented tolerance; planar
  degeneracy handled): exact expected count 3N−5/3N−6 enforced. Full archive:
  133,448 complete-count, 437 incomplete. Stationary-point evidence: **80
  members, 3,312 positive frequencies, 0 zero/negative** — restricted evidence
  only, never kinetics. Subset rows contribute ZERO positive evidence
  (exclusions unresolvable in subset scope). Actual record-58 (excluded) and
  record-212 (numeric) read-only tests pass.

## D. Quality gates — COMPLETED

- eligible_parseable = complete AND present AND parseable AND valid applicable
  ranks (all four arm cells); paired differences qid-matched explicitly.
  Frozen 200: all_selected 200 / present 200 / eligible 156 / capability 188.
- Exact Murcko scaffolds COMPLETE with versioned algorithm + stereo/acyclic
  policy recorded.
- Full-run cost incl. fit+calib; frozen top-k unchanged (ranks reused verbatim).

## E. Official MassSpecGym baseline — BLOCKED (concrete evidence)

- No retrieval checkpoint found in inspected official locations: releases carry zero assets; HF dataset
  listing (26 files, sha-pinned) holds only DreaMS/MIST simulation/embedding
  `.ckpt` files; README documents train-yourself DeepSets only.
- Bounded execution attempt with installed torch 2.13 + project-local RDKit
  against cached massspecgym==1.3.1 (file hashes recorded, unaltered): import
  fails at `massspecgym/utils.py:351` (`pulp.listSolvers(onlyAvailable=True)[0]`
  → IndexError; zero LP-solver binaries on system). No stubbing/substitution
  performed. No training undertaken.
- Prior pip incident reported as unknown/exceeded ingress, never <1GB.

## F. PubChem conditional — TRIGGERED, cache-replay only

- 0 internal + 45 canonical-unresolved misses (absent from BOTH prefix and
  ChEBI pools, parseable). No live requests: cache replay only (api_calls 0);
  four of the first five selected formulas lack cached responses and remain
  unavailable. Only C13H16O4 can be replayed: 100 capped candidates evaluate
  **1 of 45 unresolved queries, with 0/1 recovered**. This does not establish
  absence for the other 44 queries or beyond the truncated list. Frozen
  metrics never rescored. The separate supervisor schema audit recovered
  four of the *earlier* prefix-pool misses; those historical queries are not
  the corrected final selection and are not mixed into this denominator.

## G. Confidence — COMPLETED

- Wilson 95% on every rate incl. stage presence (all-selected + eligible
  conditionals); paired predictor-vs-baseline on fixed ranks; frozen n=200 kept.

## Tests

- 11 unittest modules, **212 tests, all exit 0** (`test_exits.json` with actual
  exit codes): 8 preexisting suites unchanged plus `ms2_quality_audit` (22),
  `ms2_qm9_control` (29), `ms2_external_validation` (40) with regression tests
  for every confirmed defect (excluded-58, `*^`-212, linearity, nested stages,
  rank enforcement, gate measurement, adduct/charge/count/precision, leakage
  fixtures, cache replay). Seven additional supervisor regressions pass.
