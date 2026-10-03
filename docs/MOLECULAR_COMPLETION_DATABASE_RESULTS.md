# Database-first retrieval: first synthetic run

Run date: 2026-10-03. Source: `tools/ms2_database_retrieval.py`.
Commands:
`python3 tools/ms2_database_retrieval.py > retrieval.csv`
`python3 -m unittest tools.ms2_database_retrieval`
Plan: [database experiments](MOLECULAR_COMPLETION_DATABASE_EXPERIMENTS.md).
Parent design: [molecular completion](MOLECULAR_COMPLETION_DESIGN.md).

This is a harness + controlled-fixture run, not a corpus measurement.
The 12-molecule fixture corpus stands in for a pinned structure export; no
ChEBI, MassSpecGym, MassBank, GNPS, PubChem, or QM9 data was downloaded.
Nothing below measures real database coverage, real spectral ranking, or
physical stability.

## What was built

- Standardization: V0 neutral vocab (C/N/O/F types), rejects charge,
  isotopes, salts/disconnected graphs, open valences. Stereo stripped with
  connectivity-only identity; tautomer not resolved (recorded limitation).
- Integer neutral-mass arithmetic and three-valued verdict
  (accept / reject / boundary-ambiguous / unavailable), same semantics as
  the bounded-grammar pilot.
- Staged filtering per query: mass window, exact formula + charge, typed
  subgraph containment with joint overlap consistency, optional precursor
  (heavy-atom conservation, provisional target-fragment reading).
- Fingerprint (atom-type + typed-edge counters) used only as a sound
  rejection screen; all survivors go through exact typed matching.
  Soundness is unit-tested (no false rejection on the fixture matrix).
- Unknown-overlap is the default condition; the known-overlap oracle is a
  separately labeled arm that never leaks mappings into the unknown path.
- Ranking baselines at fixed pools: uniform, provenance, typed-edge
  fingerprint similarity. Spectral similarity is `unavailable_no_spectrum`
  (no spectra fabricated). Top-k is emitted only when the target is in the
  reported pool.
- Ingestion stubs for ChEBI SDF, MassSpecGym TSV/candidates, MGF archives,
  PubChem cache (throttleniosk: <=5 req/s, cache by formula+release), and
  QM9 positive-control matching. All raise a pinning error until the
  operator supplies a pinned local file with sha256.

## Fixture results (12 in-domain + 2 excluded records)

| Query | s1 mass | s2 formula | s3 +subgraphs | s4 +precursor | Recall s3 | Status |
|---|---:|---:|---:|---:|---|---|
| C2H6O mass only | 2 | 2 | 2 | 2 | True | nonempty |
| C2H6O + CH3 | 2 | 2 | 2 | 2 | True | nonempty |
| C2H6O + CH2–OH | 2 | 2 | 1 | 1 | True | nonempty |
| C2H6O two overlapping (unknown / known) | 2 | 2 | 1 | 1 | True | nonempty |
| C3H8O mass only | 3 | 3 | 3 | 3 | True | nonempty |
| C3H8O two overlapping | 3 | 3 | 1 | 1 | True | nonempty |
| amine + C–N | 1 | 1 | 1 | 1 | True | nonempty |
| amide + C1–N | 1 | 1 | 1 | 1 | True | nonempty |
| amide + C3–N (incompatible H count) | 1 | 1 | 0 | 0 | False | target_filtered |
| ring mass only | 1 | 1 | 1 | 1 | True | nonempty |
| fragment + precursor FIX-003 | 2 | 2 | 2 | 2 | True | nonempty |
| out-of-pool slice (target removed) | 0 | 0 | 0 | 0 | None | target_absent |

The C2H6O (2: ethanol / dimethyl ether) and C3H8O (3) mass-only counts
agree with the independent bounded-grammar pilot. The CH2–OH typed edge
resolves the C2 ether/alcohol pair; an isolated CH3 does not. The
amide C3–N row shows parent-relative hydrogen strictness: a pattern with
the wrong H count on carbon correctly filters out its own target
(`target_filtered`, top-k blank). The out-of-pool slice yields zero
candidates with `target_absent` and blank recall/top-k: reported as pool
miss, not as constraint inconsistency.

## Denominators and resource gates

- Fraction parseable: 12/14 (2 synthetic exclusions: charged, salt-like).
- Fraction in domain: 12/12 of parseable.
- Target-in-pool recall after stage 3: 11/12 (the miss is the deliberate
  incompatible-pattern probe).
- Boundary-ambiguous hits: 0 at 10 ppm / 50 uDa on fixtures (expected;
  real corpora will produce nonzero ambiguous/missing/truncated classes).
- Scaffolds: 9 distinct, 2 shared (ether/alcohol isomer families).
- Physical-label coverage: 0 (QM9 index not loaded; no new DFT per plan).
- Downloads / API calls: 0.
- Wall ~3.5 ms for 12 queries single-threaded CPU (~0.3 s per 1,000
  queries at fixture scale; real-corpus indexing costs still unmeasured).
- Peak-RAM accounting via `resource` is not yet wired; harness reports
  wall/process time and exact-match work counters per query only.

## Tests

8 unit tests pass: mass trichotomy + unavailable precision,
charge/salt rejection, fingerprint soundness over the fixture matrix,
stage monotonicity, unknown-overlap isolation from the oracle object,
target-absent vs target-filtered statuses, top-k gating on pool
membership, joint overlap consistency.

CPU-only stdlib reference. GPU execution is not applicable (no tensor
ops); Rust/Python parity for a future Rust port is pending, as is any
device-kernel work. Recorded as limitations, not passing checks.

## Decision

No generation decision follows from fixtures. Next step per plan: pin one
ChEBI 3-star export and the MassSpecGym candidate pools, re-run stages
1–3 unmodified on those pools, and publish the same denominators before
any ranker or Mamba-3 work. Stop after the database stage only if formula
+ subgraph filtering leaves few candidates at high pool recall on real
pools; otherwise test a cheap ranker with calibrated abstention first,
and reserve generation for the measured out-of-pool slice.

---

# Second run: MassSpecGym 1.5 capped val slice (real pools)

Run date: 2026-10-03. Source: `tools/ms2_msgym_corpus.py` (imports the
synthetic harness unmodified for stages 1–4).
Command: `python3 tools/ms2_msgym_corpus.py --prefix
data/pinned/msgym_candidates_formula_prefix64.json --tsv
data/pinned/MassSpecGym1.5.tsv --max-queries 400`.
Pin, hashes, release: `data/pinned/PIN.json` (dataset rev `d2e86d0c`,
2026-08-07; TSV sha256 `50cfdd1d…06a`; 64 MiB prefix sha256
`6066de3b…a9a6864`; 0 API calls).

Sample: 6,234 prefix entries scanned in file order; 349 join to ≥1
val-fold TSV row (test fold untouched); first 400-cap yields 102 queries
whose TRUE structures are inside the current typed-graph domain.
Supplied formula candidate pools used verbatim (15,680 supplied entries,
per-query pools capped at 256 by the provider); the target is never
injected. Query arms per target: mass_only, single_bond, bonded_pair
(one/two bonded open subgraphs extracted from the true structure,
parent-relative H counts, unknown overlap). Precursor arm not_evaluated
(whole-target request; precursor measurement redundant per design).
No ChEBI/MassBank/GNPS/PubChem/QM9 data in this run.

## Denominators

- TSV rows: 231,104; val structures: 3,386 (dedup by SMILES string).
- Query targets: 102 in-domain of 349 joined (29%). Excluded: aromatic
  220 (63%), unsupported element 21, unsupported charge 5. Aromaticity is
  a syntactic domain boundary (no silent kekulization); most MassSpecGym
  chemistry is outside the current V0 closed-molecule domain.
- Pool members: 11,533 in-domain of 15,680 supplied (74%). Excluded:
  aromatic 3,685, disconnected/salt 426, element 23, charge 13.
- Mass diagnostic: graph-derived theoretical mass vs provider
  `parent_mass` (a MEASURED m/z-derived value, not theory): 98/102
  queries within 10 ppm (observed sub-ppm Orbitrap errors 0.2–0.8 ppm on
  spot checks); 4 queries at 10.8–12.4 ppm flagged for provenance
  review. The mass column never filters (stages use formula +
  subgraphs); parser H-count logic is independently validated by exact
  formula agreement on all 102 targets.
- Metadata retained per query: adduct ([M+H]+ 76, [M+Na]+ 26; neutral
  columns used directly, no adduct conversion performed), instrument
  (QTOF 66, Orbitrap 30, missing 6), collision energy.
- Physical-label coverage: 0 (QM9 index not loaded; no new DFT per plan).

## Ambiguity on real pools (in-domain candidate counts)

| Arm (102 queries) | median | min–max | s3 recall |
|---|---:|---:|---|
| mass_only (formula pool) | 88 | 2–256 | 102/102 |
| + single bonded substructure | 54 | 1–256 | 102/102 |
| + overlapping bonded pair | 32 | 1–226 | 102/102 |

Supplied-pool recall is 102/102 by exact SMILES string (the benchmark
ships the true molecule in its own formula pool; a pool property, not a
method result). Stage recall 102/102 on every arm is a self-consistency
check: the true structure always survives its own mass, formula and
substructure filters. Counts are of distinct supplied entries.
Even formula + two bonded substructures leaves a median of 32
same-formula candidates on drug-like chemistry: exact cheap constraints
narrow but do not resolve identity here. 19 bonded-pair arms hit the
100k-node budget and report explicit `search_budget_exhausted` (all with
recall intact; their counts are lower bounds, medians included).

## Ranking baselines

- Uniform order top-1 is 102/102 because each supplied list places the
  query molecule first: contaminated by construction, not a method
  result. Reported only to document the contamination.
- Oracle fingerprint similarity (TRUE-structure typed-edge Tanimoto, an
  upper-bound reference, not a spectrum method): top-1 on 31/102.
  Same-formula isomers are largely indistinguishable to count-based
  fingerprints; a learned spectral ranker remains untested.
- Spectral similarity: unavailable (candidates have no spectra and no
  predictor was run; measured spectra retained as metadata only).

## Resource gates

6.1 s wall / 6.1 s CPU single-threaded for 102 queries × 3 arms
(~60 ms/query, ~1,000 queries/min); 0 download bytes beyond the pinned
files (278 MB total ingress), 0 API calls. Peak-RAM accounting still
open. No GPU involved (stdlib CPU reference); Rust parity pending.

## Decision (amended after the domain extension below)

Retrieval-before-generation is confirmed as the right order. At the
time of the original run, median 32 candidates remained after formula +
subgraph constraints at full pool recall, and 63% of val targets fell
outside the chemistry domain (aromatics) — motivating (1) the cheap
ranker below, (2) the aromatics/element extension (now done: 301/349
val targets in-domain, pools 92% in-domain; remaining exclusions are
higher-valent sulfur, charges, P/Si/metals), and (3) generation reserved
for out-of-pool cases. The uniform/fingerprint top-1 numbers from the
unshuffled run were order artifacts (see the fourth run) and are
retained here only as a cautionary record: uniform 102/102 (self-first
supplier order), oracle fingerprint 31/102. ChEBI 3-star SDF ingestion
followed (third run).

---

# Third run: ChEBI 3-star full-file export (curated database)

Run date: 2026-10-03. Source: `tools/ms2_chebi_corpus.py` (same staged
filters as the other runs; V2000 molfile front end with stdlib-only
parsing).
Command: `python3 tools/ms2_chebi_corpus.py --sdf
data/pinned/chebi_3_stars.sdf.gz --max-records 60000 --max-queries 100`.
Pin: `data/pinned/PIN.json` (Sep 2026 release, 59,525,636 bytes, sha256
`5ad2c762…0982a40929b`, 52,956 records, CC BY 4.0 source-data note).

Sample: the ENTIRE 3-star export (no row cap needed; ingestion is linear
and fast). 100 queries are the first 100 in-domain structures in file
order, each with mass_only / single_bond / bonded_pair arms built from
its own bonded open subgraphs (parent-relative H, unknown overlap).
Index pool is the full in-domain set (leave-one-in). A 20-query
out-of-pool slice reruns mass_only + bonded_pair with the target
removed. Precursor arm not_evaluated (no precursor evidence in a
structure export).

Numbers below are the second ingestion pass (v2) with the extended
domain (aromatic orders, S/Cl/Br/I, max-hydride types, hydrogen
halides); the v1 pass (18,691 index, medians 8/7/3) is superseded but
consistent.

## Parsing notes (all verified against file contents)

- V2000 fields are fixed-width: 3-digit atom numbers run together
  (`97100` = atoms 97 and 100); split-based parsing misreads them.
  Fixed-width with split fallback is unit-tested.
- Molfiles carry partial explicit H (e.g. stereo-relevant H); they are
  folded into heavy-atom valence with validation (order-1, exactly one
  heavy neighbor, uncharged). Nothing is dropped silently: dangling or
  H–H-bonded H raises `explicit_h_mismatch` (9 records).
- No V3000 records in this release (checked: 0).
- Property names used: `ChEBI ID`, `ChEBI NAME`, `STAR` (verified `3`),
  `FORMULA`, `MASS`, `MONOISOTOPIC_MASS`, `SMILES`, `INCHIKEY`.

## Denominators (v2, extended domain)

- Scanned: 52,956. In-domain index: 21,720 (41%, up from 35% / 18,691
  in v1). InChIKey-connectivity duplicates removed: ~7,500.
- Excluded: heavier/main-group elements 11,502 (S/Cl/Br/I now admitted;
  P/Si/metals remain out); nonzero charge 8,414; hypervalent 1,497
  (almost all higher-valent sulfur — sulfones/sulfonamides beyond the
  documented S(II) bound); multi-component dot-formulas 106 (salts,
  co-crystals, hydrates); disconnected/salt 439; above the
  100-heavy-atom database-pass bound 560; genuine formula-vs-graph
  mismatches 98 (sampled: all atomic [Br]/[Cl]/[I]/[S] records, correctly
  rejected radicals — same mechanism as atomic fluorine); explicit-H
  anomalies 9; isotopes 172; monoisotopic-mass property mismatch beyond
  500 uDa display slack: 1; SDF aromatic bond type 4: 6 (still rejected —
  V2000 carries no endpoint aromaticity flags); unsupported bond types
  11.
- Mass used for filtering is computed from each validated graph with
  the repository integer arithmetic, never the source display mass; the
  property column serves only as the ≤500 uDa diagnostic above.

## Ambiguity over the whole index (100 queries, target always present)

| Arm | median | min–max | stage recall |
|---|---:|---:|---|
| mass window only (10 ppm + formula) | 7 | 1–31 | 100/100 |
| + single bonded substructure | 5 | 1–28 | 100/100 |
| + overlapping bonded pair | 3 | 1–21 | 100/100 |

Zero truncation on all 300 arms; zero `target_filtered`/`target_absent`
in-pool. Whole-database mass windows discriminate far better than
same-formula pools: ChEBI pair-median 3 vs MassSpecGym formula-pool
median 32. The two measurements answer different questions (curated
coverage vs hard-isomer discrimination) and are not interchangeable.

## Out-of-pool control (20 queries, target removed, 40 arms)

35 arms `nonempty` (other same-window structures remain — correctly not
reported as constraint inconsistency), 5 `target_absent`
(2 mass_only, 3 bonded_pair) where the target was the sole occupant of
its window. A zero count with a missing target is a pool miss, never a
proof of inconsistent constraints, exactly per plan.

## Ranking baselines

Uniform top-1: 81–87/100 depending on arm (file order here, no
self-first contamination — a legitimate weak baseline). Oracle
true-structure fingerprint similarity top-1: 88/100 (diverse-mass pools
are easier for count fingerprints than same-formula pools at 31/102).
Spectral: unavailable. No learned ranker run.

## Resource gates

19.1 s wall / 19.1 s CPU single-threaded for full-file ingestion
(52,956 records), 100 queries × 3 arms and the 40-arm out-of-pool slice
(~5 queries/s including ingestion; steady-state retrieval ≈
15 ms/arm). 59.5 MB ingress, 0 API calls. No GPU involved (stdlib CPU
reference); Rust parity pending.

---

# Fourth run: cheap ranker with calibrated abstention

Run date: 2026-10-03. Source: `tools/ms2_rank_abstain.py` (linear score
on cheap exact features; coordinate-ascent fit on file-order train
half; gate fit for ≥90% train precision; evaluated on the held-out
half). Features never include the true structure (no oracle leakage;
asserted by unit test). Candidate pools are query-seeded shuffled, so
index order (ChEBI IDs) and self-first supplier order (MassSpecGym)
cannot leak into top-1 — an earlier unshuffled pass measured 74–100%
"top-1" that was pure order artifact and is discarded, not reported.

## ChEBI (100 queries, 50/50 split, 463 candidate rows)

Learned weights: residual 1.0, log-synonyms 0.5, log-xrefs 0.5,
neg-embedding-count −1.0. Unshuffled top-1: train 0.50, test 0.54
(uniform-at-shuffled-order would be ~0.2–0.3 on these pool sizes).
Gate (margin ≥ 0.59, pool ≤ 3): train coverage 0.40 at 0.90 precision;
test coverage 0.50 at 0.88 precision — calibration transfers within
noise. Risk–coverage curve (test): (0.60, 0.80) → (0.50, 0.88) →
(0.36, 1.00). A few exact counts plus literature frequency halve the
error rate at roughly half coverage. A post-review rerun with fixed
embedding counts and truncation exclusion reproduced these numbers
exactly (463 rows, 0 truncated queries). Spectral evidence would be
needed to go further; nothing here justifies a generative model.

## MassSpecGym formula pools (235 queries, 117/118 split, 9,455 rows)

No feature separates same-formula isomers without spectra or
provenance metadata: fitted weights (residual 1.0, embedding-count
−2.0) reach top-1 0.14/0.06 train/test — uniform level, with the
embedding-count sign likely fitting noise at this signal strength. The
fitted gate abstains almost everywhere (pool ≤ 1 only): coverage 3%/2%
at 1.00 precision on train/test. Eighteen queries whose matching hit
the node budget are excluded from calibration (partial pools must not
pose as exact) and counted separately. This is the correct outcome for
a cheap metadata-free ranker — calibrated refusal instead of confident
wrong answers — and it locates the remaining gap precisely: spectral
ranking (the benchmark's own fingerprint-prediction direction), not
structure generation, is the next experiment. The out-of-pool slice for
generation stays empty in formula pools by construction (pool recall
100%).

## Out-of-pool generation demo (synthetic domain only)

`tools/ms2_oop_enumerate.py`: pool holds only dimethyl ether; query
targets ethanol with hydroxyl evidence. Retrieval reports
`target_absent` with zero candidates (honest miss); the independent
bounded enumerator then recovers exactly 1 graph (ethanol), status
`complete`. This is the plan's "generation evaluated separately on pool
misses" at pilot scale — a protocol demonstration, not a general
generation result. Unit-tested (miss-then-recover assertion).

---

# Codex review (2026-10-03, `codex exec review --uncommitted`)

An independent review pass over all five tool modules returned six
findings, each with a concrete reproduction. All six reproduced on
inspection and are fixed, each with a regression test (52 tests green):

1. Embedding counter overcounted (triangle path-pattern: 9 instead of
   6) — leaf bulk-reset let later branches reuse ancestor atoms.
   Existence matching was never affected (returns on first hit).
2. Failed calibration fell back to predict-everything — now reject-all.
3. Ranker pools silently kept truncated results — incomplete pools are
   now excluded from calibration and counted (18 MassSpecGym queries).
4. Isotope brackets silently normalized (`[13CH4]` → 12C-methane) —
   now rejected as `unsupported_isotope`.
5. Unknown mass precision read as rejection — now passes through with a
   separate `n_unavailable` count, per the design doc.
6. Prefix ending mid-key crashed the candidate scanner — now keeps
   complete entries like mid-array truncation.

Review-prompt friction noted: this Codex version rejects any prompt
combined with `review --uncommitted`, so the review ran with default
scope (which proved sufficient).
