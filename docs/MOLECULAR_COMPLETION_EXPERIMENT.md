# Bounded molecular-completion ambiguity experiment

Status: bounded reference audit completed on 2026-10-04. Of 26 synthetic
queries, 21 completed with resolved mass evidence, two had unresolved mass
evidence, one had unsupported oracle input, and two exhausted their search
budgets. Bounded database, ranking and external-validation experiments also exist;
representative corpus expansion and a validated precursor comparison remain pending.
See
[ambiguity results 2026-10-04](MOLECULAR_COMPLETION_AMBIGUITY_RESULTS.md)
and the earlier first-restricted pilot run (2026-10-03):
[pilot results](MOLECULAR_COMPLETION_PILOT_RESULTS.md).
Parent design: [molecular completion](MOLECULAR_COMPLETION_DESIGN.md).

## Purpose and answer to the model-accuracy question

The broader experimental purpose is to determine whether the model proposed in
[MOLECULAR_COMPLETION_DESIGN.md](MOLECULAR_COMPLETION_DESIGN.md) can achieve high
prediction accuracy. The user has clarified that **top-25 exact-identity
recovery is the primary accuracy goal**; top-1 performance is secondary. On
2026-10-05 the user set the number: **95% top-25 recovery counts as high**. It
was set after the first synthetic measurements below and applies to every later
evaluation.
The bounded ambiguity audit is one feasibility check within
that program; it tests input information and search correctness, not learned
prediction accuracy.

**Current conclusion: high accuracy of the designed completion-conditioned
Mamba model has not been verified.** A first native version of that model was
trained and evaluated on 2026-10-05 in a synthetic setting (exact composition
plus substructures cut from the target): at a matched 28,000-step budget its
top-25 recovery is 3.4% at 64 samples per query and 5.0% at 256, against 0.5%
for a formula-only control and 0% untrained; trained on to 64,000 steps it
reaches 5.8% and 8.4%, with validation loss still falling (section "2026-10-05:
native completion-conditioned model" below). The existing trained spectrum-to-fingerprint model is Ridge, a
separate retrieval-ranking baseline with different inputs and output task. Its
200-query val top-1 is 6% (12/200), versus 4% (8/200) for uniform ranking;
top-10 is 24.5% (49/200), versus 21.5% (43/200). Its 90%-precision calibration
gate fails closed and accepts zero val predictions. These findings do not show
high accuracy, and do not prove that the Mamba design cannot achieve it.

A direct model feasibility test remains to be specified and run: freeze input
semantics and identity, keep the 95% top-25 target and predeclare
coverage/calibration targets, train the proposed model on disjoint molecules/scaffolds, and compare
with matched baselines on held-out data. Report search/domain exclusions and
confidence intervals. The 21/26 (80.8%) audit completion rate and successful
reference recovery must not be presented as the designed model's accuracy.

### Top-25 reanalysis of saved baseline predictions

After the user accepted top-25 recovery as the objective, the frozen Ridge val
predictions were reaggregated at K=25. No model, candidate ordering, calibration
threshold or test-fold outcome was changed. The result uses the original exact
provider-SMILES identity and one-based ranks; retained seeded tails for unscored
candidates remain part of the raw ranking.

| Baseline | Top-25 hits / all 200 selected val queries | Recovery |
|---|---:|---:|
| Uniform seeded order | 68/200 | 34.0% |
| Mass-residual order | 72/200 | 36.0% |
| TRAIN fingerprint prior | 74/200 | 37.0% |
| Ridge predictor | 91/200 | 45.5% |

For the separate present-and-target-parseable subgroup, predictor recovery is
72/156 (46.2%) versus uniform 46/156 (29.5%). On the 188 inference-capable queries,
it is 79/188 (42.0%) versus uniform 56/188 (29.8%). These subgroups do not replace
the all-query denominator. All 200 targets occur in the supplied pools, and
35 pools contain at most 25 candidates, so small-pool inclusion contributes to
all methods. The old selective gate still accepts zero requests; this top-25
reanalysis measures raw shortlist recovery separately from acceptance.

This observed top-25 difference is more encouraging than the earlier top-1 result,
but is a post-hoc metric on a fixed val slice, not proof of high accuracy or of a
Mamba advantage. The designed completion model uses different inputs and remains
untested. Source ranks and their SHA-256, subgroup counts and metric definition
are saved in `experiments/molecular_completion/20261003_predictor/top25_reanalysis.json`.
Existing top-1/top-3/top-10 hit columns were verified against the saved ranks
before computing the new metric.

### Top-50 failure analysis of the same baseline

A subsequent reaggregation gives predictor recovery of 113/200 (56.5%) at K=50,
versus uniform 95/200 (47.5%), mass residual 99/200 (49.5%) and prior 95/200
(47.5%). These are frozen raw full-pool ranks, not accepted predictions or Mamba
results. All 200 targets are in their candidate pools; pool absence therefore
explains none of the 87 predictor misses.

| Mutually exclusive miss category | Queries | Direct observation |
|---|---:|---|
| Parseable target with unique fingerprint in its pool | 61 | Target ranked below 50 despite no exact target fingerprint collision |
| Parseable target with colliding fingerprint | 5 | Target ranked below 50; tied fingerprint classes contain 34, 30, 3, 7 and 29 members |
| Unsupported target fingerprint | 21 | Ten hypervalent and eleven unsupported-charge targets receive no finite fingerprint score and remain in the seeded fallback tail |

Eleven of the 21 unsupported-target misses have at least 50 finitely scored
candidates preceding the fallback tail. All 87 misses occur on queries that can
score some candidates, so a usable query prediction does not mean the target
itself is supported. Seventy misses occur in pools larger than 200. The maximum
exact target fingerprint equivalence class among all parseable targets is 34;
no class exceeds 50. Exact collisions alone therefore do not explain the top-50
shortfall, and there is no target-absence explanation on this slice.

The measured bottlenecks are poor target ordering and unsupported target scoring.
The baseline's 1-Da spectrum bins, local 256-dimensional hashed fingerprint,
linear Ridge model and 3,269-molecule fit sample are possible contributors to
poor ordering. This failure audit does not isolate their causal effects. It also
does not establish an information-theoretic ceiling or a failure of the proposed
mass/substructure-conditioned Mamba architecture, which was not evaluated here.
Controlled representation/model/data ablations are needed before naming one of
those choices as the dominant cause.

Source hash, affected query IDs and the exact decomposition are saved in
`experiments/molecular_completion/20261003_predictor/top50_failure_analysis.json`.

## Progress overview

Status as of 2026-10-04: the synthetic reference experiments, bounded
implementation validation and earlier bounded database/ranking studies are recorded. The full experimental program is **not
complete**. Completing a bounded run includes recording truncation correctly;
it does not mean every query yielded an exact count.

| Workstream | Current status | Evidence / outstanding requirement |
|---|---|---|
| Independent C/O pilot | Completed | Five synthetic queries; three unit tests passed |
| Rust C/N/O bounded audit | Completed with two censored searches | 26 queries; 21 resolved and complete, two mass-unresolved, one unsupported, two graph-budget-exhausted |
| Independent small-query cross-check | Completed within its declared scope | Python enumerator agrees on 24 fixtures; 5/6-atom fixtures excluded |
| Rust/Python API and CPU/GPU checks | Completed for the bounded implementation | 34 integration tests per backend feature configuration; seven internal CPU tests; seven Python API tests |
| Exact full-domain 5/6-atom counts | Unresolved | C5H12 and C6H14 searches need completed enumeration |
| Precursor-constraint comparison | Not evaluated | Requires independently validated compatible parent/target pairs |
| Database/corpus ambiguity study | Bounded runs completed; expansion remains open | ChEBI and MassSpecGym retrieval measurements exist; sampling/domain exclusions limit generalization |
| Cheap learned ranking and abstention | Bounded baselines completed | Linear metadata ranker and Ridge spectrum-to-fingerprint predictor evaluated; predictor gate abstains on all 200 val queries |
| Completion-conditioned Mamba generation | First native version implemented and measured in a synthetic setting; recovery low | `completion_model.rs`, grammar `completion-exact-v2`; top-25 13/379 at 64 samples, 19/379 at 256 at the matched budget, 22/379 and 32/379 after 64,000 steps (oracle composition, target-derived substructures). Mass-derived formulas, matched Transformer/retrieval baselines, calibration and an untouched test set are outstanding |
| External validation and restricted physical evidence | Completed with limitations | MassBank/GNPS checks and QM9 controls exist; no asynchronous verification service or new physical calculations |
| Publication of implementation and audit artifacts | Pending | Only the experiment document was committed and pushed in `5f962d2`; implementation, detailed report and dated artifacts remain local |

The checked acceptance items at the end apply to the bounded reference audit.
They are not a completion checklist for the precursor or completion-conditioned Mamba
workstreams. Earlier corpus/ranking work has its own denominators and domains. The pilot report is historical: its then-pending nitrogen, input
uncertainty and integration checks were subsequently covered by the Rust audit.

## Experiments conducted to date

### 2026-10-03: independent Python pilot

The standalone reference in `tools/ms2_completion_ambiguity.py` enumerated
connected, neutral C/H/O typed graphs with two to four heavy atoms, closed valence
and at most one ring. It used synthetic neutral monoisotopic masses, 10 ppm
tolerance and 50 microdaltons observation uncertainty. It independently performed
integer mass decisions, exhaustive bond assignments and canonical permutation
identity checks; it did not invoke the Rust grammar.

| Pilot query | Accepted formulas | Compatible graphs | Search status |
|---|---:|---:|---|
| C2H6O, mass only | 1 | 2 | Complete |
| C2H6O, singleton CH3 | 1 | 2 | Complete |
| C2H6O, CH2–OH bond | 1 | 1 | Complete |
| C2H6O, overlapping CH3–CH2 and CH2–OH bonds | 1 | 1 | Complete |
| C3H8O, mass only | 1 | 3 | Complete |

Each query visited 204 formula vectors. C2H6O queries examined 21 edge assignments;
the C3H8O query examined 175. All five completed within the pilot bounds and its
three unit tests passed. The manually known alternatives were ethanol/dimethyl
ether and 1-propanol/2-propanol/methoxyethane. No precursor constraint, instrument
data, model or physical-stability validation was used. See the
[pilot report](MOLECULAR_COMPLETION_PILOT_RESULTS.md).

### 2026-10-03: database, ranking and external-validation studies

These studies predate the Rust audit and are separate experiments. Their Python
retrieval domain includes extensions beyond the bounded Rust C/N/O domain; do
not combine their counts or validation claims without a common-domain check.
The detailed results and caveats are retained in the linked reports.

| Experiment | Recorded outcome | Limits on interpretation |
|---|---|---|
| Synthetic staged database retrieval | 12 in-domain molecules plus two excluded records; target absence separated from filtering failure | Harness check, not real corpus coverage; synthetic precursor conservation probe is not validated fragmentation evidence |
| MassSpecGym initial capped val retrieval | 102 in-domain targets; candidate medians 88 → 54 → 32 with subgraph evidence; 19 bonded-pair searches censored | Provider formula pools include the target; self-first ordering contaminates the initial ranking figures; counts include lower bounds |
| ChEBI full-export retrieval | 52,956 records scanned; historical v2 index 21,720; 100 queries with candidate medians 7 → 5 → 3; 20 target-removed controls | Curated database membership is not exhaustive chemistry; later external-validation indexing reports 26,659 under its own policy |
| Cheap metadata ranker | ChEBI held-out top-1 0.54; gate coverage 0.50 at precision 0.88. MassSpecGym train/test top-1 0.14/0.06 | Small fixed slices; 18 truncated matching queries excluded from calibration; no generative-model benefit established |
| Spectral nearest-neighbor baseline | 200 val queries: reference-backed candidates in 10 queries, target reference in zero; 0/10 accepted predictions correct | Exact-string coverage and file-order selection limit generalization; TRAIN control is separate from val results |
| Ridge spectrum-to-fingerprint predictor | 200 val queries: top-1 12/200 vs uniform 8/200; 90%-precision calibration gate fails closed, 0/200 accepted | Four extra hits are exploratory, not demonstrated improvement; 156 target-parseable queries and 188 capability queries are separate denominators |
| MassBank external frozen-model check | 150 selected queries: prefix-pool recall 54/150; predictor top-25 21/150 (21/54 among present targets); ChEBI nested mass-stage presence 76/150 | Capped archive scan and fixed identity/domain policy; whole-parent precursor stage remains `not_evaluated` |
| GNPS check | 126 blocks, 124 lacking structure; two structurized queries, both absent from pools | Very limited structure coverage; imported-library license status not assumed |
| QM9 restricted physical control | 239 eligible fit molecules; 81 allowed exact matches, 60 excluded, one GDB-only, 97 no-match; 80 stationary-point evidence members | Dataset computational evidence only, not kinetic stability, a live verification service or new calculations |
| Official MassSpecGym baseline / PubChem expansion | Official baseline blocked by unavailable retrieval checkpoint and LP solver import; cache-only PubChem replay covers one of 45 unresolved queries, 0 recovered | Four of five selected formulas lack cache; lists truncated; no absence proof and no live expansion |

Sources: [database results](MOLECULAR_COMPLETION_DATABASE_RESULTS.md),
[spectral results](MOLECULAR_COMPLETION_SPECTRAL_RESULTS.md),
[predictor results](MOLECULAR_COMPLETION_PREDICTOR_RESULTS.md),
[external/remaining results](MOLECULAR_COMPLETION_REMAINING_RESULTS.md), and
[protocol status matrix](MOLECULAR_COMPLETION_PROTOCOL_STATUS.md).
The associated saved summaries are under `20261003_spectral/`,
`20261003_predictor/` and `20261003_remaining/` in
`experiments/molecular_completion/`. These are prior recorded results, not
experiments rerun during this update. The sealed test fold was not used by the
reported predictor fit or val evaluation.

### 2026-10-04: Rust bounded audit and integration validation

OpenCode implemented the Rust reference search, PyO3 API, independent small-fixture
Python audit and initial tests. Local review completed corrections, validation
and reporting after its provider returned HTTP 429 (endpoint unavailable).
The Rust enumerator starts at START under an exact formula budget and filters
completed graphs by composition and residual valence before typed matching.
The domain extends the pilot to supported neutral C/N/O types and two to six heavy
atoms, with at most one independent cycle.

The 26 hand-authored queries cover chains, branching, unsaturation, rings,
nitrogen-containing structures, symmetry, insufficient and incompatible evidence,
unknown overlap, known overlap/disjointness, rejected mass, boundary ambiguity,
unavailable precision and full-domain budget exhaustion. Manually pinned graph
sets check ethanol/dimethyl ether and ethylamine/dimethylamine independently of
the Rust output. Reference metadata is used for recovery auditing, not to restrict
the candidate search.

The independent Python audit enumerates type multisets and bond assignments and
uses permutation identities. It agrees with the 24 affordable fixtures on semantic
results; its counters count different operations from Rust and are not compared.
The two larger fixtures are explicitly outside this independent cross-check.
The installed Python API calls the authoritative Rust implementation, so API
parity and independent-enumerator agreement are separate validation checks.

Final integration validation covers exact composition, residual valence, typed
matching, overlap classes, graph identity, deterministic reruns, uncertain mass,
invalid inputs and every declared resource limit. The counts and resource
measurement are recorded below. No additional experiment was run for this
progress-document update.

### 2026-10-05: native completion-conditioned model (synthetic, oracle-formula setting)

**What was built.** The exact-completion grammar `completion-exact-v2` (host
reference, device kernels, host twins); training examples cut from known
molecules (`completion_data.rs`); a substructure-set encoder feeding the
existing Mamba-3 action decoder, with trainer, sampler and host acceptance
(`completion_model.rs`); recovery metrics (`completion_eval.rs`); the experiment
driver (`completion_experiment.rs`, `examples/molecular_completion_experiment.rs`);
a JSON generation protocol with a Python binding (`completion_api.rs`,
`mamba3_rl.MolecularCompletionModel`, [API](MOLECULAR_COMPLETION_API.md)).
Implementation by OpenCode from written task specifications; plan and code
reviewed by Codex (four reviews; no blocker; findings applied).

**Setting — read this before the numbers.** A query is a held-out molecule's
**exact composition** (an oracle input: no mass, no formula search) and one to
four **substructures cut from that same molecule**, each two to eight atoms,
carrying the parent's hydrogen counts (also an oracle annotation). On the
evaluation queries they average 2.46 patterns of 5.05 atoms covering 43% of the
target's atoms. A returned candidate must have that composition, be a closed
connected molecule and contain every substructure; identity is the typed
kekulized graph (a bond-order-free "skeleton" identity is reported as a relaxed
secondary). Ranking is by sample frequency. This measures whether the model can
complete a molecule from correct partial structure. It is not a test on
measured spectra, mass-derived formulas or predicted substructures.

**Data.** CASMI 2026 structures, identity folds: training folds 2–4, validation
= the 379 fold-1 molecules of the existing export; fold 0 untouched. Domain 32
heavy atoms and six ring closures: 354 of the 379 validation molecules are
eligible, the 25 larger ones are counted as misses. Two training sets: 17,876
molecules (the earlier export) and 150,425 molecules
(`tools/ms2/export_casmi_molecules.py`; 28 more skipped: 26 over the closure
limit, 2 canonicalization budget). None of the 354 eligible validation
molecules has the canonical trace (or the bond-order-free skeleton trace) of a
retained training molecule, in either set. By Murcko scaffold, relative to the
150k export, 171 validation molecules have a scaffold absent from training, 201
a seen one, 7 are acyclic (relative to the small set: 278 / 94 / 7). The data is
CC BY-NC: exports, checkpoints and per-query predictions stay outside the
repository; only aggregates are recorded here.

**Model and training.** `d_model` 128, four decoder blocks, three message
rounds; batch 16, AdamW at 3e-4, gradient clip 1.0, one seed; fresh patterns
every epoch; checkpoint chosen by teacher-forced validation loss — so the
validation fold is a selection set and no untouched test set was used. Apple
M1 / Metal.

**Results** (all 379 validation molecules in the denominator; 95% intervals from
1,000 bootstrap resamples of molecules):

| Grammar | Training molecules (steps of the chosen checkpoint) | Arm | Samples per query | Top-1 | Top-10 | Top-25 | Finished / dead end | Completed but lacking a substructure |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v1 | — | Untrained | 64 | 0 | 0 | 0/379 | 0.3% / 99.7% | 0.3% |
| v1 | 17,876 (20,000) | Full | 64 | 7 | 7 | 7/379 = 1.8% (0.5–3.2) | 39.7% / 60.3% | 34.2% |
| v1 | 17,876 (20,000) | Full | 256 | 10 | 11 | 12/379 = 3.2% (1.6–5.0) | 39.1% / 60.9% | 33.7% |
| v2 | — | Untrained | 64 | 0 | 0 | 0/379 | 43.3% / 56.7% | 42.3% |
| v2 | 150,425 (26,000) | Formula only | 64 | 2 | 2 | 2/379 = 0.5% (0.0–1.3) | 73.0% / 27.0% | 69.3% |
| v2 | 150,425 (24,000) | **Full** | 64 | 13 | 13 | **13/379 = 3.4% (1.6–5.5)** | 70.0% / 30.0% | 58.7% |
| v2 | 150,425 (24,000) | **Full** | 256 | 15 | 18 | **19/379 = 5.0% (2.9–7.4)** | 70.1% / 29.9% | 59.0% |
| v2 | 150,425 (64,000, continued) | Full, longer training | 64 | 17 | 21 | 22/379 = 5.8% (3.4–8.2) | 74.3% / 25.7% | 59.6% |
| v2 | 150,425 (64,000, continued) | Full, longer training | 256 | 21 | 30 | 32/379 = 8.4% (5.8–11.4) | 74.4% / 25.6% | 59.6% |

"v1" is the exact grammar before the feasibility lookahead (STOP only when
complete, closed-prefix attachment); "v2" adds the lookahead. The formula-only
arm never shows the substructures to the model but is judged by the same
acceptance filter. The two "continued" rows resume the full model from its
step-24,000 checkpoint for 40,000 more steps (the optimizer's moments restart
at the resume); they have no formula-only counterpart at that budget, so the
matched comparison is the pair of rows at 24,000 / 26,000 steps. Percentages in the last two columns are of all sampled
trajectories. Skeleton top-25 differs from strict by at most one molecule in
every row. For the full v2 model at 256 samples, by scaffold relative to its own
training set: scaffold-novel 1/171 (0.6%), scaffold-seen 15/201 (7.5%), acyclic
3/7 (recomputed from the saved per-query predictions; the run's own report used
groups built against the small training set and is not the figure to quote).
After 64,000 steps: scaffold-novel 7/171 (4.1%), scaffold-seen 22/201 (10.9%),
acyclic 3/7.
Teacher-forced validation loss at the chosen checkpoints, on the first 128
eligible validation molecules: 0.613 nats per token with substructures, 0.708
formula-only (about 17 and 20 nats per molecule): the target trace is still an
improbable sample, which is what the recovery numbers show. Both arms had a 28,000-step
budget; the full run was cut at step 28,000 by a full disk while saving, so its
last saved best checkpoint (step 24,000) was evaluated.

**What the measurements say.**

1. *Substructures help, recovery is low.* 13 against 2 of 379 at a matched
   budget, with non-overlapping intervals (1.6–5.5% against 0.0–1.3%), in this
   one seed and setting; 3–5% is not high accuracy. The intervals resample the
   fixed query outcomes: they do not include training-seed, sampling-seed or
   checkpoint-selection variation.
2. *When the target is found it is usually ranked first* (top-1 13, top-25 13 at
   64 samples; 17 of 22 after longer training): the limit is coverage — whether
   the target is sampled at all — more than ordering. Four times the samples
   adds six molecules at 24,000 steps and ten at 64,000.
3. *The dominant loss is now containment.* In the full v2 run 59% of all
   trajectories (84% of the finished ones) complete a valid molecule of the
   right formula that lacks a required substructure; about half of those miss
   every pattern. The decoder is not using the patterns enough. The design's
   per-step containment input (architecture item 3) is not implemented yet.
4. *The feasibility lookahead removes most dead ends.* Under v1, 60% of a
   trained model's samples died; every one of 13,672 classified dead ends
   violated one of four cheap necessary conditions, on average 1.4 steps before
   dying. With those conditions part of legality (v2) an untrained sampler goes
   from 0.3% to 43% finished — the one comparison in the table where only the
   grammar changed. The trained rows changed grammar, training set and
   checkpoint together, so their 40% to 70% is not attributable to the grammar
   alone. The remaining 30% are dead ends none of the four conditions explains.
5. *The small training set overfits early; the large one keeps improving.* With
   17,876 molecules validation loss turned up after 20,000 steps while training
   loss kept falling. With 150,425 (and the v2 grammar) it fell from 0.613 nats
   per token at 24,000 steps to 0.533 at 64,000 (about 15 nats per molecule),
   its last and best evaluation — the model is not converged. Recovery rose
   with it: 13 to 22 of 379 at 64 samples, 19 to 32 at 256. The share of
   samples that lack a required substructure did not move (59.0% to 59.6%): more
   training made the model better at finishing and at the molecules it does
   reach, not at honouring the patterns.
6. *Little transfers to new scaffolds*: at 256 samples, 1 of 171 scaffold-novel
   molecules at 24,000 steps and 7 of 171 (4.1%) at 64,000, against 15 and 22 of
   201 with a scaffold seen in training (7.5% and 10.9%).

**Distance to the 95% target.** The best recorded result, 32 of 379 (8.4%) at
256 samples after 64,000 steps, is a factor of eleven short. Broken down over
the 354 eligible queries of that run (from the saved per-query predictions):

| Heavy atoms | Queries | Top-25 hits |
|---|---:|---:|
| up to 12 | 8 | 6 (75%) |
| 13–16 | 5 | 2 (40%) |
| 17–20 | 61 | 9 (15%) |
| 21–24 | 161 | 12 (7.5%) |
| 25–28 | 93 | 3 (3.2%) |
| 29–32 | 26 | 0 |

| Supplied substructure atoms ÷ molecule atoms (overlaps counted twice) | Queries | Top-25 hits |
|---|---:|---:|
| under 0.25 | 61 | 0 |
| 0.25–0.5 | 105 | 4 (3.8%) |
| 0.5–0.75 | 93 | 6 (6.5%) |
| 0.75–1.0 | 58 | 7 (12%) |
| 1.0 or more | 37 | 15 (41%) |

In every one of the 32 hits the target was inside the top 25 as soon as it was
sampled at all; no query sampled its target and ranked it below 25. So ranking
is not what separates this model from the target: the target molecule is simply
never drawn for 322 queries. Recovery tracks how much of the molecule the query
specifies and how small the molecule is. Two readings follow, neither proven
here. First, a decoder that honoured the supplied substructures (59% of samples
do not) would recover more at every evidence level. Second, with a formula and
substructures covering under half of a 24-atom molecule, many distinct molecules
satisfy the query, and no model can place the right one in 25 slots for 95% of
such queries; the bounded audit above already shows several compatible graphs
at two to four heavy atoms. The 95% target needs a query definition under which
the answer is nearly determined — near-complete substructure coverage, a
candidate database to choose from, or measured-spectrum evidence — and that
definition has to be fixed before the next evaluation.

**Validation of the implementation.** Suites run on the CPU backend and on
Metal (wgpu): grammar 19 tests, model 18, generation 15, experiment driver 5,
generation API 10 (its exact-fixture comparison is defined on CPU only), plus the
existing MS2 generation (35) and fused-step (2) suites unchanged; host-only
suites (CPU): data 11, diagnostics 7. Python: 11 tests of the model binding (20
with the existing request-layer tests) over a rebuilt CPU wheel sharing the Rust
fixture (`tests/fixtures/ms2/completion_tiny.*`), each mirroring a Rust
assertion. Exhaustive searches on six small formulas (at most six
atoms and one ring closure) confirm the exact grammar removes no molecule and no
completing trace there; at 32 atoms the evidence is the necessity argument for
each rule plus the legal replay of one canonical trace per retained molecule.
Device and host legality masks agree on every token of the canonical traces of
17,876 real molecules (495,285 tokens) on Metal, and of the 354 validation
molecules (9,929 tokens) on both backends. Sampled trace log-probabilities
matched the teacher-forced ones to about 3e-6 in the test that compares them
(2.9e-6 on CPU, 3.4e-6 on Metal; the asserted tolerance is 1e-3). A training
step performs no device read unless a loss report is requested (then one); a
generation call performs one.

**Limits of this result.** One seed; oracle composition and oracle
substructure hydrogens; substructures sampled from the answer rather than
predicted from data; identity-fold validation used for checkpoint selection, no
untouched test set; no retrieval or matched-Transformer baseline, so nothing
here attributes anything to Mamba-3; no calibration or abstention; an early
formula-only run on the small set is not reported because its checkpoint was
selected with a driver bug (fixed, with a regression test, before the runs in
the table). Known technical debt: the autograd tape-merge hazard worked around
in the encoder (`completion_model.rs`), the sampler's 32-bit draw key, slower
training under v2 (about 8 steps per second against 20 under v1 on M1; the
cause is not yet profiled), and the unresolved 30% of dead ends.

### 2026-10-06: functional-group queries, formula from a mass, stereochemistry, physical verification

**Query definition changed.** The user specified that the supplied substructures
are **functional groups**. Extraction follows Ertl's algorithm
(`functional_groups.rs`; every heteroatom, carbons in non-aromatic multiple
bonds, acetal carbons and three-membered heterocycles, merged when bonded),
each group given as its own atoms with the molecule's hydrogen counts.
Aromaticity is an approximation (`aromatic-ring-v2`); against RDKit's reference
implementation the groups agree exactly on 34 of 34 named molecules and 4,992
of 5,000 real ones. On the 354 eligible validation queries a query carries on
average 4.3 groups of 1.8 atoms, covering 34% of the molecule's atoms; every
molecule has at least one group. The random-patch results of the previous
section belong to the earlier query definition and are not comparable.

**Results** (150,425 training molecules, 40,000 steps, one seed, all 379
validation molecules in the denominator; the same caveats as the previous
section apply: oracle inputs, validation fold used for checkpoint selection):

| Arm | Acceptance rule | Formula | Samples | Top-1 | Top-10 | Top-25 |
|---|---|---|---:|---:|---:|---:|
| Untrained | contained | given | 64 | 0 | 0 | 0/379 |
| Formula only (26,000 steps) | contained | given | 64 | 0 | 2 | 2/379 = 0.5% |
| Full | contained | given | 64 | 4 | 14 | 14/379 = 3.7% (1.9–5.5) |
| Full | contained | given | 256 | 3 | 13 | 18/379 = 4.7% (2.9–7.1) |
| Full | separate occurrences | given | 64 | 4 | 14 | 14/379 |
| Full | complete list | given | 64 | 4 | 14 | 14/379 |
| Full | complete list | given | 256 | 3 | 15 | 21/379 = 5.5% (3.4–7.9) |
| Full | complete list | given | 1,024 | 4 | 20 | 32/379 = 8.4% (5.8–11.4) |
| Full | complete list | from mass | 64 | 4 | 10 | 11/379 = 2.9% |
| Full | complete list | from mass | 256 | 5 | 19 | 24/379 = 6.3% (4.0–8.7) |
| Full, retrained with `aromatic-ring-v2` groups | complete list | given | 256 | 3 | 18 | 21/379 = 5.5% |
| Full, retrained | complete list | from mass | 256 | 4 | 13 | 14/379 = 3.7% |

Differences of a few molecules between rows are within sampling noise (the
intervals overlap; different rows draw different samples). Scaffold-novel
molecules: 0–2 of 171 in every row.

1. *The model honours functional groups; that was not the case for random
   patches.* 86% of finished samples pass the atom-type test for every group and
   2% miss every group (random patches: about half missed every pattern).
2. *The answer space is the limit.* At 1,024 samples a query has on average 134
   distinct molecules that satisfy the formula and the complete functional-group
   list. In every run, a target that was sampled at all was inside the top 25;
   it was sampled for 32 of 354 queries at 1,024 samples.
3. *Stricter acceptance shrinks the lists, not the misses.* The complete-list
   rule cuts distinct candidates from 26 to 10 per query at 64 samples and
   leaves the 14 hits unchanged.
4. *A formula from a mass costs nothing here.* With a synthetic exact neutral
   mass at 5 ppm the mass alone admits a median of 47 formulas (maximum 1,127);
   the complete functional-group list fixes every heteroatom count and exactly
   one formula survives for all 354 queries, with and without the training-fit
   pruning. That is a property of an exact mass and a complete, correct group
   list; a measured mass or an incomplete list would leave more.

**What evidence reaches 90%** (`tools/ms2/completion_evidence_ambiguity.py`;
model-free). For 18,173 validation molecules, the number of known molecules that
share the stated evidence, counted in the 219,358 distinct structures of the
competition's training folds 1–4 (the query included; fold 0 never read):

| Evidence | Unique | At most 5 | At most 25 | Median / 99th percentile / largest pool |
|---|---:|---:|---:|---|
| Neutral mass (5 ppm) | 5.0% | 17.8% | 47.1% | 28 / 279 / 387 |
| Formula | 11.2% | 30.3% | 59.8% | 16 / 278 / 383 |
| Mass or formula + functional groups | 56.4% | 90.1% | **99.7%** | 1 / 17 / 44 |
| + aromatic ring systems | 60.5% | 91.9% | 99.8% | 1 / 15 / 44 |
| + ring counts and sizes | 67.9% | 95.7% | 99.96% | 1 / 11 / 28 |
| + what each group is attached to | 72.2% | 97.4% | 100% | 1 / 8 / 18 |
| + all ring systems | 73.5% | 97.0% | 99.99% | 1 / 9 / 28 |
| + carbon-type counts (hybridisation, hydrogens) | 78.1% | 98.1% | 100% | 1 / 7 / 18 |
| + Murcko scaffold | 85.9% | 98.5% | 99.99% | 1 / 7 / 28 |
| + distances between functional groups | 89.6% | 99.8% | 100% | 1 / 4 / 15 |
| + all atom environments of radius 1 / 2 / 3 | 97.4% / 98.7% / 99.2% | 99.8%+ | 100% | 1 / 2 / 10 |
| + connectivity without bond orders | 98.9% | 99.9% | 100% | 1 / 2 / 10 |
| ring counts, carbon types, ring systems, group attachments and group distances together | 96.9% | 99.95% | 100% | 1 / 2 / 10 |

Three readings, in decreasing order of certainty:

1. *Against a table of known molecules that contains the answer, a mass and the
   functional groups already suffice for 90%.* 99.7% of queries have at most 25
   known molecules with the same formula and groups, 90.1% at most five. The
   missing factor for top-25 is therefore not more chemistry but **a candidate
   list**: the model generates freely and meets more than a hundred valid
   molecules per query that no table contains.
2. *For a single answer (top-1) at 90%, the skeleton has to be pinned down.*
   Mass and groups identify 56% uniquely. Adding how far apart the groups are
   reaches 90%; ring information, carbon types and group attachments together
   with those distances reach 97%; complete local atom environments — what a
   perfect fingerprint predictor would supply — reach 97–99%.
3. *Against a much larger table, the "unique" column is the one to read.* These
   counts are lower bounds: only 219k molecules are counted. A query that has a
   twin in this table will have many in a table hundreds of times larger, so for
   a database of that size the fraction still within 25 can be at most about the
   unique fraction here: 56% for mass and groups, 90% with group distances, 97%
   and more with local environments or connectivity. For generation with no
   table at all the same holds with more force. This reading is an inference
   from the lower-bound nature of the counts, not a measurement on a larger
   database.

**Formula from a mass, stereochemistry, physical verification** (implemented;
details in [API](MOLECULAR_COMPLETION_API.md) and
`tools/ms2/COMPLETION_PHYSICAL_VERIFY.md`):

- *Formula from a mass.* A request may give a neutral mass, or a precursor m/z
  with `[M+H]+` / `[M-H]-`, instead of a composition. Mass evidence is reported
  as accepted, boundary-ambiguous, rejected, search-incomplete, unavailable or
  overflow, separately from why nothing was sampled; training-fit pruning is
  labelled empirical and can be switched off.
- *Stereochemistry.* Each candidate carries its stereo elements and the exact
  number of distinct stereoisomers, computed from the graph's automorphisms, and
  each stereoisomer on request. The model does not choose among them: the data
  has no stereo labels. Checked against RDKit: 20 of 20 named molecules agree;
  of 379 real molecules, 295 have only supported kinds and a resolved count, and
  294 of those agree (in the remaining one RDKit conflates two constitutionally
  different ligands); 79 carry a kind this version does not model and say so,
  5 exceed the element limit and are reported unresolved.
- *Physical verification.* A separate Python tool on RDKit: structural alerts on
  the graph, 3D embedding, MMFF94/UFF relaxation, geometry diagnostics, and the
  requested stereoisomer re-read from the coordinates. Its statuses are
  `force_field_optimization_converged`, `calculation_failed`, `unsupported`,
  `error`; every result states that stability is not evaluated and that no
  electronic-structure program is installed. It never reorders candidates.

**Validation.** CPU and Metal: grammar, model, generation, experiment driver,
generation API and acceptance-semantics suites; host-only: data, diagnostics,
functional groups, stereo, formula enumeration; Python: binding tests over a
rebuilt CPU wheel and the tool tests. Three further Codex reviews (plan, code of
the mass/stereo/functional-group work, code of the acceptance rules and the
verification tool) found one blocker and twelve major issues in the code, each
fixed with a test that reproduces it first.

## Question

Within a declared small chemical domain, how many distinct complete molecular
graphs agree with target mass and required open substructures? Measure the effect
of adding precursor constraints separately. This is an input-information audit,
not a claim about a trained model's achievable accuracy or physical stability.

## Initial domain and budgets

Start with two to six heavy atoms, C/N/O atom types drawn from the supported V0
vocabulary, connected closed-shell graphs and at most one independent cycle.
Use the current bond-order rules and count hydrogens through atom types. Exclude
unsupported stereo/charge states explicitly. Report results for this restricted
domain; a single-ring restriction excludes chemically legitimate larger domains.

Provisional per-query limits:

| Resource | Initial limit |
|---|---:|
| Formula enumeration work | 100,000 count-vector visits |
| Graph action extensions | 100,000 |
| Embedding-match search nodes | 100,000 |
| Canonicalization extensions, cumulative | 100,000 |
| Retained unique graphs | 10,000 |
| Modeled retained identities + frontier memory | 64 MiB |
| Wall-clock watchdog | 30 seconds |

These are safety bounds for a first measurement, not measured throughput targets.
Deterministic work limits define reproducible experiments. The wall-clock watchdog
is a separate failure/truncation status, because it can differ across machines.
Record whole-process peak RSS separately. The modeled memory limit excludes
parser, caller and temporary allocations; it is not an allocator-level RSS cap.
Input limits are eight substructures, 24 total pattern atoms and 16 correspondence
pairs.

## Query construction

1. Build a fixed, provenance-recorded set of small reference molecules covering
   chains, branching, unsaturation and a ring. Check each reference is inside the
   exact domain and representable by the existing grammar.
2. Extract one or more open subgraphs while preserving parent hydrogen counts.
   Include overlapping, disjoint, symmetric and insufficiently informative sets.
   Do not include ground-truth cross-substructure atom correspondence in the main
   unknown-overlap condition. Known correspondence is a separately labeled oracle.
3. Derive neutral target masses using the repository arithmetic and explicit
   precision metadata. Include accepted, rejected, boundary-ambiguous and unknown
   precision fixtures. Distinguish synthetic measurements from instrument data.
4. For precursor-condition comparisons, use independently validated compatible
   parent/target pairs and explicit conventions. If none are available, report
   that arm as not evaluated; do not fabricate a precursor by adding an arbitrary
   mass and call it fragmentation evidence.

## Reference search

Enumerate bounded formulas over the declared domain. For accepted mass candidates,
enumerate legal BFS graph traces from START using `TraceState` and the formula
budget. Do not seed the state with an arbitrary input subgraph: the existing grammar
does not expose unrestricted partial-graph completion.

At each possible STOP, require exact composition and zero residual valences before
performing typed subgraph-containment checks. Required embeddings are injective
within each supplied substructure; distinct substructures may share target atoms.
Unknown cross-substructure overlap is inferred by these mappings. When complete
correspondence is supplied, enforce its equivalence classes across all maps:
different classes must map to different target atoms. An empty supplied relation
means all pattern atoms are disjoint; an absent relation means overlap is unknown.
Count a target graph once, not once per embedding or traversal. Canonical traces
give identities within the stated graph representation, not stereochemical
identity beyond it.

Do not prune a partial target merely because a required substructure is not yet
contained. Initially perform containment only on completed graphs; add pruning
later only with a sound reference argument and tests. Keep all formula branches
needed for a completeness claim. Canonicalization or matching failures invalidate
that claim just as graph-search truncation does.

Mass-ambiguous candidates are a separate unresolved class. Unknown precision cannot
yield a mass-constrained zero count; optionally run a separately labeled structural
audit with mass filtering disabled.

## Output and interpretation

For every query record domain, input hash, seed, candidate formulas, numeric mass
status, unique accepted graphs, unresolved hypotheses, ground-truth recovery,
every work counter, elapsed time and termination reason.

Use distinct statuses:

- `complete`: all relevant formula, trace, match and identity searches completed.
- `search_budget_exhausted`: candidate count is a lower bound, not an exact count.
- `mass_evidence_unresolved`: boundary ambiguity or unavailable precision prevents
  a definitive mass-constrained count.
- `unsupported_input`: chemistry or semantics outside the declared domain.
- `reference_error`: arithmetic, canonicalization or other validation failure.

Record multiple reasons if applicable. Only a complete search with resolved mass
evidence may certify zero compatible graphs, and only within its declared domain.
Do not equate zero candidates in the restricted domain with global inconsistency.

Report exact-count distributions only for completed, resolved queries, alongside
the completion rate and censored lower-bound distribution. Report counts as
constraints are added: mass only, one substructure, multiple substructures, known
overlap oracle, and precursor evidence if available. Count monotonicity is a useful
correctness check when the domains and evidence are fixed.

More candidates means weaker identifiability under the chosen constraints. It does
not imply a `1 / count` bound on accuracy unless candidates are conditionally
equiprobable. Learned chemical priors may be strongly nonuniform.

## Measured audit results (2026-10-04)

The resolved completion rate was 21/26 (80.8%). Exact-count distributions include
only these completed queries; unresolved, unsupported and truncated queries are
reported separately.

| Exact compatible graphs | Completed queries |
|---:|---:|
| 0 | 2 |
| 1 | 10 |
| 2 | 5 |
| 3 | 3 |
| 5 | 1 |

The full-domain C5H12 and C6H14 searches retained three and five graphs,
respectively, before reaching the graph-extension limit. These are lower bounds,
not established exact counts. All supplied in-domain references in completed
searches were recovered; an unrelated reference supplied as a negative control
was not recovered. Recovery is unknown when its shared resource budget is exhausted.

For two singleton methyl patterns at C2H6O mass, unknown overlap admits ethanol
and dimethyl ether (two graphs). The known-disjoint oracle admits only dimethyl
ether (one graph), demonstrating that absent and empty correspondence differ.

Final validation passed 34 completion integration tests on CPU and 34 with wgpu
features, seven internal completion tests on CPU, and seven Python API tests
against a rebuilt and installed CPU wheel. The independent Python enumerator
matched all 24 fixtures within its four-heavy-atom limit; the two larger fixtures
were explicitly excluded from that cross-check. Earlier chemistry, contract,
decoder and formula regressions also passed on CPU and GPU. The completion
reference search itself runs on the host; shared device mass and grammar behavior
was checked on Apple M1 / Metal.

The final release audit took 1.581 seconds with whole-process peak RSS of
7,585,792 bytes (7.23 MiB), measured after compilation with `/usr/bin/time -l`.
This is a bounded audit measurement, not an optimization or throughput claim.
No performance optimization or model training was performed in this Rust audit.
Earlier Ridge/linear ranking studies are separate experiments. Precursor evidence
remains `not_evaluated` because no independently validated compatible pairs were
provided.

Artifacts are recorded under
`experiments/molecular_completion/20261004_completion_ambiguity/`:
`completion_report.json` contains per-query results; `supervisor_distribution.json`
separates exact counts from censored lower bounds; `supervisor_experiment_metadata.json`
records commands, input/source hashes, platform and peak RSS; `final_validation.json`
records final validation commands and exit codes. Earlier failed environment and
compile attempts are preserved alongside successful reruns.

### Per-query audit outcomes

| Query | Status | Mass status | Compatible graphs | Recovery |
|---|---|---|---:|---|
| c2h6o_mass_only | complete | accepted | 2 | recovered |
| c2h6o_ch2oh_bond | complete | accepted | 1 | recovered |
| c2h6o_ch3_symmetry | complete | accepted | 2 | not evaluated / not supplied |
| c2h6o_incompatible_n_atom | complete | accepted | 0 | not evaluated / not supplied |
| c3h8o_mass_only | complete | accepted | 3 | recovered |
| c3h8o_other_target_not_recovered | complete | accepted | 3 | not recovered |
| c3h8o_disjoint_ch3_and_ch2oh | complete | accepted | 1 | recovered |
| c2h7n_mass_only | complete | accepted | 2 | recovered |
| c2h7n_nh2_selects_ethylamine | complete | accepted | 1 | recovered |
| c3h4_mass_only_unsaturation | complete | accepted | 3 | recovered |
| c3h4_triple_bond_selects_propyne | complete | accepted | 1 | recovered |
| c3h6_ring_cyclopropane | complete | accepted | 1 | recovered |
| c3h6_disjoint_double_and_methyl | complete | accepted | 1 | recovered |
| c4h8_mass_only | complete | accepted | 5 | recovered |
| c4h8_square_ring_selects_cyclobutane | complete | accepted | 1 | recovered |
| c2h6o_oracle_overlap_shares_ch2 | complete | accepted | 1 | recovered |
| oracle_requires_incompatible_types | unsupported_input | not_evaluated | 0 (not a zero certificate) | not evaluated / not supplied |
| c2h6o_mass_far_off | complete | rejected | 0 | not evaluated / not supplied |
| c2h6o_mass_boundary_ambiguous | mass_evidence_unresolved | ambiguous | 0 (not a zero certificate) | not evaluated / not supplied |
| c2h6o_precision_unavailable | mass_evidence_unresolved | unavailable | 0 (not a zero certificate) | not evaluated / not supplied |
| c2h6o_overlap_unknown | complete | accepted | 1 | recovered |
| c2h6o_full_domain_cn_o | complete | accepted | 2 | not evaluated / not supplied |
| c5h12_full_domain_cn_o | search_budget_exhausted | accepted | 3 (lower bound) | not evaluated / not supplied |
| c6h14_full_domain_cn_o | search_budget_exhausted | accepted | 5 (lower bound) | not evaluated / not supplied |
| c2h6o_two_methyls_unknown_overlap | complete | accepted | 2 | recovered |
| c2h6o_two_methyls_known_disjoint | complete | accepted | 1 | recovered |

Both truncated searches terminated with `graph_extension_limit`. The unsupported
oracle equates incompatible atom types and was rejected before search. The
boundary-ambiguous and missing-precision queries do not certify zero, despite
retaining no accepted graphs. Missing recovery values are not recovery failures.
The unrelated reference in `c3h8o_other_target_not_recovered` is a deliberate
negative control, not a failed in-domain target-recovery test.

## Current key findings

1. **Mass alone leaves structural ambiguity in these synthetic domains.**
   Completed mass-only searches retain two C2H6O graphs, three C3H8O graphs,
   two C2H7N graphs, three C3H4 graphs and five C4H8 graphs. The independent
   pilot confirms the small C/O alternatives.
2. **The information in a substructure matters more than its mere presence.**
   An isolated methyl atom leaves the C2H6O count at two; a typed CH2–OH bond
   reduces it to one. An NH2 pattern selects ethylamine from the two C2H7N
   alternatives, and a square-ring pattern selects cyclobutane from five C4H8
   graphs. Constraint counts were checked for monotonicity at fixed evidence
   and domain.
3. **Known correspondence is a stronger evidence condition.** Two singleton
   methyl patterns admit two C2H6O graphs under unknown overlap and one under
   known disjointness. An absent relation and an explicitly empty oracle
   relation therefore cannot be treated interchangeably.
4. **The bounded search and input handling pass the recorded checks.**
   Completed in-domain references are recovered, identities are deduplicated,
   and incomplete or unresolved searches do not certify zero. This supports
   correctness on the tested fixtures; it is not a proof for arbitrary inputs.
5. **Full-domain 5/6-atom completeness is still unestablished.** Retaining the
   familiar three pentanes and five hexanes does not establish exact counts:
   both runs stopped at their work budget. The 80.8% resolved completion rate
   measures query disposition, not model accuracy or recovery accuracy.
6. **The Rust audit does not establish corpus or physical conclusions.**
   Its synthetic neutral masses and typed graph constraints do not measure
   instrument noise, physical stability, precursor improvement or model accuracy.
   Earlier database and ranking studies do provide bounded corpus and held-out
   baseline evidence, but they do not establish a general ambiguity rate or a
   trained completion-conditioned Mamba result. QM9 supplies restricted
   computational controls, not a stability service.
7. **The existing ranking gap does not yet establish a benefit from generation.**
   Formula pools retain many isomers, spectral reference coverage is sparse, and
   the Ridge predictor's small exploratory top-1 gain fails its acceptance gate.
   These results motivate further baselines and calibrated abstention; a learned
   graph generator must be compared on matched pools and compute budgets.

## Remaining unresolved issues and completion criteria

| Issue | Current evidence / impact | Work needed to resolve it |
|---|---|---|
| Search censoring at 5/6 heavy atoms | C5H12 and C6H14 terminate at 100,000 graph extensions; retained counts are lower bounds | Rerun with declared larger budgets, or explicitly narrow the domain; obtain complete formula, graph, match and identity searches before reporting exact counts |
| Limited independent validation of larger targets | The separate Python enumerator excludes the two 5/6-atom queries | Supply an independent reference or manually established typed identity sets for larger completed searches; do not count PyO3 parity as independent enumeration |
| Unresolved mass evidence | Boundary and unavailable-precision fixtures cannot certify zero | Obtain adequate precision/convention metadata for real inputs, or retain unresolved status; a deliberately uncertain control need not become a resolved query |
| Contradictory oracle input | Incompatible shared types are rejected as unsupported input | Correct invalid correspondence in real queries; retain this fixture as a negative control rather than forcing it to succeed |
| Precursor comparison | No validated compatible parent/target pairs; arm is `not_evaluated` | Define mass/charge and fragmentation conventions, validate pairs independently, and compare fixed queries with and without precursor evidence |
| Corpus generalization | Bounded ChEBI/MassSpecGym and external slices exist; fixed file-order sampling and chemistry exclusions limit coverage | Expand representative samples and report confidence, exclusions and matched query/evidence definitions; retain existing bounded results |
| Chemical scope and measurement realism | Neutral V0 C/N/O, one-cycle restriction, no stereo or stability evidence | Either keep these limits explicit or validate additional chemistry and instrument-derived evidence before expanding claims |
| Completion-conditioned model and ranking improvement | A first native model exists; in the synthetic oracle-formula setting its top-25 recovery is 3.4–5.0% at a matched budget (formula-only 0.5%) and 5.8–8.4% after 64,000 steps, not converged, with 59% of samples completing molecules that lack a required substructure | Give the decoder per-step containment progress (design architecture item 3) or containment-aware legality; train to convergence on the 150k set; then add the mass-derived formula stage, retrieval and matched-Transformer baselines, an untouched scaffold-disjoint test set and calibration |
| Official baseline and PubChem coverage | Official checkpoint/LP solver unavailable; four selected PubChem formula caches unavailable | Resolve actual dependency/checkpoint availability and declare any data-ingress budget before new acquisition; preserve capped/unavailable statuses |
| Memory and performance generalization | One whole-process RSS observation; modeled memory excludes temporary/parser/caller allocations | Measure representative workloads and actual memory if stronger resource claims are needed; use profiling before performance optimization |
| Repository publication and reproducibility | Prior push included only this status document; implementation, detailed report, fixtures and logs remain local | Review and commit/push the implementation and reproducibility artifacts; exclude generated wheel/venv output |

The recorded corrupt build artifact, full disk and non-Clone test compilation
failures were resolved, and their reruns passed. They remain in the validation
history for traceability; they are not current experimental blockers. Unknown
recovery on exhausted resource budgets is reported as unknown rather than a
false negative.

## Recommended next sequence

1. Publish the implementation and audit artifacts so the documented experiments
   can be reproduced from the repository.
2. Resolve or deliberately retain the 5/6-atom domain restriction, and strengthen
   independent validation before claiming full-domain exact counts.
3. Extend the existing corpus baselines and collect validated precursor pairs;
   run those evidence arms separately with explicit conventions and uncertainty
   metadata, keeping the prior frozen ranking results separate.
4. The first completion-conditioned model is measured (3.4–5.0% top-25 in the
   synthetic setting at a matched budget, 5.8–8.4% after longer training).
   Before scaling: make the decoder use the substructures
   (per-step containment progress, or a sound containment-aware mask), finish
   training on the 150k set, and decide which evidence a query must carry for
   the 95% target to be reachable at all (see "Distance to the 95% target");
   then evaluate with mass-derived formulas and matched baselines, using validity,
   diversity, abstention and search accounting to interpret recovery.

## Bounded-audit acceptance and next decision

- [x] Independent tiny fixtures have manually established completion sets
      (`experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json`
      with hand-authored `expected` counts; two identity sets hand-authored
      for `c2h6o`/`c2h7n`; the pilot and fixture-mirror cross-check both
      independently reproduce the small C/O counts).
- [x] Tests cover composition exhaustion, residual valence, overlap double
      counting, incompatible shared atom types, symmetry, ambiguous masses,
      missing precision, invalid input and each resource limit
      (`tests/ms2_completion.rs`: 34 tests covering
      `formula_visit_limit`, `graph_extension_limit`,
      `embedding_node_limit`, `canonicalization_limit`,
      `retained_graph_limit`, `memory_bound`, watchdog (zero and overflow),
      and the irresolvable-no-zero rule for ambiguous/unknown precision).
- [x] Complete searches recover all in-domain fixture targets and never
      certify a truncated count as exact. Repeated deterministic runs agree
      (`fixture_expectations_hold`, `results_are_identical_across_runs`,
      deterministic parity vs the independent Python mirror).
- [x] CPU/GPU checks cover shared mass and legality behavior; exposed
      Rust/Python semantics agree. `grammar_replay` device-host parity and
      `formula_top` mass-table twin parity are executed on both CPU and
      `wgpu` (Metal) exit 0 (`supervisor_validation.json`,
      `supervisor_gpu.log`, and final checks in `final_validation.json`).
      Python mirror parity is run in
      `tests/ms2_completion.rs`; the PyO3 binding tests
      (`bindings/python/tests/test_ms2_completion.py`) run green over the
      installed wheel (`experiments/molecular_completion/20261004_completion_ambiguity/pkg-venv`
      is build output, not a source artifact).
- [x] The report includes domain exclusions, every incomplete search
      (`docs/MOLECULAR_COMPLETION_AMBIGUITY_RESULTS.md` per-fixture table),
      input provenance and exact configuration. No accuracy or performance
      claim is made from an unmeasured estimate. The measurement protocol
      separates identity/structural workloads (modeled storage counters)
      from the one whole-process peak RSS sampled by `/usr/bin/time -l`.

Next decision: the recorded correctness checks pass and the small-fixture audit
finishes within the stated limits. The 5/6-atom full-domain fixtures
genuinely truncate at default budgets. Do not scale the model: raise the
trace-enumerator budget bound or shrink the domain before drawing
identifiability conclusions. If ambiguity remains high, evaluate
ranking and calibrated abstention or seek additional typed-evidence, not a
larger unsupervised model. The precursor arm remains explicitly
`not_evaluated` pending genuinely independently validated parent/target
pairs.
