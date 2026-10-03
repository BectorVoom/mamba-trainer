# MS2 contracts and host reference: second codex review and disposition

Date: 2026-10-03. Reviewer: `codex exec` (codex-cli 0.160.0, read-only sandbox, reasoning effort high), reviewing
revision 2 of [MS2_CONTRACTS.md](../MS2_CONTRACTS.md), the Python reference and scripts in `tools/ms2/`, and the
Rust host reference `src/models/ms2/{chem,graph,grammar,targets,formula,contract}.rs` with its tests. No blocker
was found; the major findings were checked against the code and are dispositioned below. The review is
preserved as received.

## Disposition

| Finding | Checked | Action |
|---|---|---|
| Python identity kept stereochemistry (33 SMILES classes against 32 traces on a stereo diol) | Yes: `isomericSmiles=True` on the parent molecule | `fragment_smiles` strips stereo marks before tagging; the stereo diol is a fixture molecule; pilot and reports rerun |
| `CandidateBatch::validate` accepted illegal finished candidates (`[START, STOP]`) | Yes | P1-D: replay of every emitted prefix, record order, status relations, V0 constants, residual valences |
| Schema version missing from four schemas; no field for retained formula mass | Yes | Contract §3 (revision 3) and P1-D: versions on every schema; `formula_support_complete`, `formula_mass_retained` |
| Formula-search semantics lived only in code; exhaustion dropped counters; conversion failure looked like absence | Yes: `formula.rs` returned an idle result on mid-scan exhaustion | Contract §9 now states the algorithm; P1-D keeps partial counters, adds `mass_overflow` and a `complete` flag |
| `f64` weight accumulation breaks exact ties | Yes: the cut is tied in 14 of 300 pilot spectra | Recipe step 5 uses integer weights; both references changed; a tie fixture and a 45-graph cut fixture added |
| Unchecked integer conversions (`GenerationConfig`, `composition`, `Limits`) | Yes | P1-D: checked conversions, a 4,096-atom host limit, validated `Limits` |
| Merge policy, structure-prior control and the V1 domain were under-specified | Yes | Merged inputs unsupported in V0 (FPNet's merge recorded as FPNet's); the prior blinds the encoder only; V1 labelled provisional |
| Pinned-row export could label a test row as training data | Yes | `export_casmi.py` refuses a pinned row outside the train subset |
| Report comparison could pass on missing fields | Yes | Strict comparison: identical unique rows and fields, per-graph embeddings, integer weights and anchors |
| Python reference boundary behaviour (bounds, `ion` range, positional anchors, sentinel, invalid intensities) | Yes | All changed in `ms2_reference.py`; Rust follows in P1-D |
| Request validation gaps (`peak_id`, counts) | Yes | Contract §3.1 and P1-D |
| Test gaps: decimal ion intervals, 4-closure prefixes, differing-boundary embeddings, traces above 9 atoms | Yes | Fixture schema 2: `ion_cases`, `whole_trace` (pyrene: 16 atoms, 4 closures, 22 tokens), `special_spectra` |
| Fold-1 parts not in the audit JSON; median shown as 16 | Yes | Audit writes the parts; the table shows 16.5 |

Not adopted: independent canonical minima for 10 to 16 atoms from a second exhaustive search (too slow in Python
without the same pruning, which would no longer be independent). Identity at those sizes is checked instead by
the RDKit partition of every fixture subgraph, atom permutations, and the 300-spectrum report comparison.

## Review as received

## Findings

No blocker established. Major findings follow. Paths are repository-relative; **C** means `docs/MS2_CONTRACTS.md`.

**Major — Python identity disagrees with the frozen graph identity.**  
`tools/ms2/ms2_reference.py:176` uses `isomericSmiles=True` and preserves stereochemistry; C:355 and C:476 define identity solely by atom type and bond order. Reproduced on the in-domain parent `CC[C@H](O)CCCC[C@@H](O)CC`: Python produces **33 SMILES classes**, but canonical traces produce **32 classes**. Embeddings `(1,2,3,4)` and `(7,8,9,10)` have identical traces but opposite stereochemical SMILES. This affects pilot weights and identity cross-checks. **Fix:** remove stereochemical annotations from the tagged molecule before generating identity keys, while preserving the type tags.

**Major — `CandidateBatch::validate` accepts illegal “finished” candidates.**  
`src/models/ms2/contract.rs:646` checks padding, lengths, STOP presence, and NaNs, but never replays the grammar. Its positive test explicitly accepts `[START, STOP]`, with no atom (`tests/ms2_contract.rs:372`, `:385`), contrary to C:311 and C:324. It also omits checks for:

- Non-zero unused action fields, invalid token fields, interior PAD/STOP, composition and atom/closure limits.
- `(spectrum, trajectory)` ordering and provenance.
- `request_failed ⇒ length=0`, and consistency with fatal request status.
- V0’s constant attachment/evidence/identity fields.
- Oracle formula probability being zero, STOP-time residual valences, and search-counter consistency.
- Valid retained-intensity fractions and mathematical log-probability ranges.

**Fix:** replay emitted prefixes without truncating their `u32` fields; validate record ordering, status relationships, constants, residuals, and counter invariants. Replace the positive `[START, STOP]` test with a legal trace.

**Major — four schemas omit the promised schema version.**  
C:106 says “Every schema starts with `schema_version: u32`”; only `SpectrumBatch` contains it. It is absent from `GenerationConfig` (`contract.rs:415`), `CandidateBatch` (`:503`), `ModelConfig` (`:717`), and `ChemistryDomain` (`:864`). Their semantic version strings do not implement this rule. C:522 also promises retained formula probability mass, but C:194 and `CandidateBatch` have no field for it. **Fix:** implement version validation for every serialized schema and add an explicit optional retained-formula probability field.

**Major — formula-search semantics are partly defined only in code, and exhaustion loses counters.**

- C:520 does not freeze the widened window, inclusion of ambiguous formulas, binary-search read counting, or table-order scoring truncation. These decisions first appear in `src/models/ms2/formula.rs:216`.
- `formula.rs:299` returns an idle result after partial scanning, resetting already observed joins to zero. Example: rows `C6` and `C3SH4`, a 100-ppm midpoint query, and visit cap **4** perform three binary-search reads and join the first scanned row; the result reports `rows_visited=4, rows_joined=0`.
- `formula.rs:250` converts parent-mass errors into absence with `.ok()`. For precursor `0`, adduct `1`, an arithmetic underflow becomes `absent=true`, rather than a distinguishable mass error.
- C:183 says **reaching** a cap sets exhaustion; code sets it only when additional work would exceed the cap (`formula.rs:266`, `:325`). A one-row search completing in three visits at cap three is not exhausted.
- C:522’s `rows_scored == rows_joined` condition can hold after visited exhaustion resets both counters to zero. Equality does not prove complete enumeration.
- The “whole table” default is replaced by the row count (`contract.rs:441`), although visiting the whole table additionally requires binary-search reads.

**Fix:** freeze these algorithms in the contract; preserve partial counters; distinguish conversion failure, absence, and exhaustion; require completed enumeration before reporting probabilities; include search overhead in a whole-table visit budget.

**Major — Rust floating-point accumulation can violate exact weight ties.**  
Python uses rational weights (`ms2_reference.py:442`); Rust accumulates `f64` and compares those values exactly (`targets.rs:366`, `:378`). Six unit peaks, each shared among six graphs, give each graph exact weight **1**. Rust’s sequential sum of six `1/6` shares is **0.9999999999999999**; a seventh graph receiving one exclusive unit peak gets **1.0**, bypassing the canonical-trace tie rule in C:431. **Fix:** use an exact comparison representation for retention weights, and add a tie-at-retention-boundary fixture.

**Major — checked integer handling is incomplete outside ordinary V0 sizes.**

- `GenerationConfig::validate` truncates `usize` limits and adds unchecked (`contract.rs:483`). On a 64-bit host, `max_atoms=2^32+16` becomes 16; `max_ring_closures=u32::MAX` causes debug overflow or a wrapped step floor. **Fix:** checked conversions/additions before range validation.
- `MolGraph::composition` accumulates unchecked `u16` counts (`graph.rs:258`). A chemically valid 32,767-carbon chain has **65,536 hydrogens**, causing panic/wrap before checked mass computation. **Fix:** return a checked composition result or reject unsupported size/count bounds first.
- Public `Limits` are unrestricted (`grammar.rs:49`, `:102`): `max_atoms=0` still permits a root (`:359`), while sufficiently large limits can exceed the 32-bit pointer masks (`:153`). **Fix:** validate representable limits at construction.

**Major — some frozen preprocessing/control decisions remain missing.**

- **Merge:** C:400 says “peaks united” and cites FPNet’s merge. The external `v4g/v4b__code__casmi__spectra.py:307` instead normalizes each spectrum, merges near duplicates within **0.005 Da**, keeps the stronger peak, and returns **float32** at :334. Adduct-mode ties and averaging individual energies versus per-spectrum means are also unspecified. These choices change masses, uncertainty, intensities, and energy features. **Fix:** choose the actual merge algorithm, deterministic ties, averaging unit, and uncertainty propagation.
- **Structure-prior control:** C:565 bypasses empty-spectrum abstention, but C:566 sets all metadata unknown. Unknown adduct still causes fatal `insufficient_metadata` (`contract.rs:348`), and ordinary formula search cannot proceed. **Fix:** define which inputs are hidden from the encoder while retaining required chemistry, or explicitly define unconditioned diagnostic generation.
- **V1:** C:358 calls the domain frozen, but C:364 defers its vocabulary and ion rules. The audit accepts any otherwise clean atom type (`audit_casmi.py:103`), rather than enforcing the stated non-test vocabulary. **Fix:** freeze those mappings or label the coverage as a provisional broader-domain estimate.

**Major — pinned-row export can put test data into training.**  
Normal export applies `subset_of` (`tools/ms2/export_casmi.py:39`). The `--rows-from` branch instead assigns every selected molecule `"subset": "train"` without checking its fold (`:77–84`). A report containing a fold-0 row bypasses the exclusion. **Fix:** validate every pinned row’s fold and subset before exporting it.

**Major — the report comparison can falsely pass.**  
`tools/ms2/label_report.py:91` ignores missing fields; :83–101 ignores extra comparison rows and overwrites duplicate row IDs. Reproduced: a comparison row containing only `{"row":1}` passes against `{"row":1,"targets":2}`. Counts and sorted probabilities also cannot establish equal graph partitions or graph-to-anchor associations. **Fix:** require identical unique row sets and required fields, and compare identities, weights, and anchors per graph.

**Minor — Python reference boundary behavior is incomplete.**

| Function | Disagreement and minimal example | Fix |
|---|---|---|
| `all_bfs_traces` / `canonical_trace` | No size/closure validation or 200,000-expansion failure path (`ms2_reference.py:192`, `:221`), unlike C:321 and C:483. A 17-atom chain returns a trace. | Add a bounded production reference; retain exhaustive enumeration separately for small tests. |
| `ion` | Unchecked range: `ion(Counter(C=400,H=0),1,0)` returns **4,801,007,276**, exceeding `u32` (`:390`), contrary to C:370. Unknown adduct produces an unclassified `KeyError` at :386. | Checked range and explicit unsupported-adduct errors. |
| `targets` | Anchors use filtered positional indices (`:443`), not original `peak_id` as C:433 requires. A supplied peak with original ID 7 becomes anchor `[0,s]`. | Accept/preserve original IDs. |
| `targets` / Rust `pseudo_labels` | Unknown uncertainty still performs decisions and counts hypotheses as ambiguous (`Python:434`; `targets.rs:349`), although C:122 disables exact-mass decisions. An exact-mass butane peak with sentinel uncertainty yields ambiguity rather than a skipped decision. | Handle the sentinel before matching. |
| `filter_peaks` | Invalid intensities can silently disappear (`Python:467–471`; Rust `targets.rs:209–218`). `[1, NaN]` or `[1,-1]` can yield a seemingly usable spectrum, whereas C:120 requires explicit defects. | Enforce validation at the reference entry point. |

**Minor — request metadata validation has gaps.**  
`SpectrumBatch::validate` checks ID ordering but accepts a valid `peak_id=u32::MAX`, IDs outside the original list, and `raw_peak_count < peak_count` (`contract.rs:307`, `:363`). Unknown precursor precision returns only `exact_mass_unavailable` (`:366`); C:506 additionally requires `formula_absent` outside oracle mode. The test currently expects warning-only behavior (`tests/ms2_contract.rs:177`). **Fix:** validate count/ID relationships and compose final precision/formula status with oracle context.

## Verified correct

**1. Revision-1 dispositions**

| Finding | Revision-2 disposition | Resolving evidence / remainder |
|---|---|---|
| 1 — mass-error interval | **Fully resolved** | C:237: residuals rounded “**up**”; C:375: ion’s “**own hydrogen count**”; C:380: `E = E_arith + U`; C:384 separates tolerance from storage uncertainty. |
| 2 — schemas | **Partly resolved** | Sidecar, IDs, separate tolerances, energy scaling, capacities and status bits now exist (C:89, :114, :170, :492). Universal schema-version handling and retained formula probability remain missing. |
| 3 — domain audit | **Fully resolved** | Shared vocabulary check at `ms2_reference.py:93`; audit calls it at `audit_casmi.py:93`. JSON: `spectra.v0_request_domain.in_domain=1750154`; `structures.in_v0_structure_domain=272609`. |
| 4 — pilot population/ion/retention/overlap bugs | **Fully resolved** | Fixed population and denominator at `pilot_targets.py:130`; cyclic unrelated parent :144; retention :146; overlap :149. Shift cap and non-negative H at `ms2_reference.py:409`, :388. C:453 explicitly says it is not a false-label rate. |
| 5 — identity/evidence semantics | **Partly resolved** | C:417 specifies linear intensity; :424 drops failures before matching; :428 unions embeddings; :433 stores accepted pairs. Type tags fix sulfur identity, but retained stereochemistry and positional anchors still disagree. |
| 6 — evaluation/splits/controls | **Partly resolved** | Disjoint subdivisions C:45–47; instrument holdout :54; containment :467; baselines :567; aggregation :558. V1 mappings, structure-prior execution, and pinned export remain incomplete. |
| 7 — feasibility/configuration | **Fully resolved** | C:164 freezes four groups and no convolution; C:545 explicitly excludes uncounted allocations. Rust factory agrees (`contract.rs:757`). |
| 8 — numeric/provenance discrepancies | **Fully resolved** | Other instruments **4.2%** (C:56); target sizes **3–16** (:457); outside elements **80** plus explicit-H **18** (:346); RDKit comparison now in `casmi_audit.mass_table_check`. |
| 9 — caller counts/serialization | **Fully resolved** | Caller-specific counts C:82–85; fixed record packing :191; actual emitted length :199; PAD :198; formula sentinel :200; attachment field :204. Validator shortcomings are listed separately above. |

**2. Measured figures**

No numerical disagreement remains **at the displayed rounding** across the three reports.

One exact value deserves preservation: C:449 displays the three-cut median as **16**, while `target_pilot.grid[7].targets_per_spectrum.p50 = 16.5`. That is compatible with half-even integer rounding; showing **16.5** would avoid ambiguity.

The measured fold-1 subdivisions at C:51—**18,055 / 18,186 / 18,265**—are absent from all three reports. I independently reproduced them from `structures.parquet` using the frozen predicate, but the cited JSON evidence should contain them. The same source confirms **274,184 identity groups**, **140,160 scaffold groups**, and **18 explicit-H structures**.

The pilot table, retained-target histograms, precursor percentages, energy statistics, vocabulary frequencies, bond proportions, peak statistics, RDKit differences, formula-table **37,859 rows / 908,616 bytes**, window occupancy and **0.8669687740799178** validation formula coverage agree.

**3. Python reference**

For valid supported inputs, these agree with the contract:

- `classify`: neutral/type/isotope/radical/connectivity restrictions (`ms2_reference.py:81–96`).
- `enumerate_subgraphs`: qualifying cut sets, induced components, boundary counts, sizes and closures (:137–164).
- `Replay.masks`: all eight §4.4 rules, including empty-budget root masking, unused fields, STOP and post-STOP behavior (:243–300). `apply` itself assumes the caller has checked legality (:302).
- BFS traversal and exhaustive lexicographic minimum (:197–222), subject to the missing bounds/work limit above.
- Ion algebra and ion-H error bound (:388–393); negative-H hypotheses are absent.
- `decide` and observed-mass tolerance (:380–402).
- Shift enumeration, union of embedding evidence, equal graph sharing and rational retention (:409–459), subject to identity/ID issues above.
- Valid-input peak filtering and decimal uncertainty formulas (:464–478).

`stored_decimals` (:481) measures digits needed by numeric values; it cannot establish original storage precision or recover float32 provenance. Its use must not replace the caller’s precision metadata.

**4. Rust reference**

- **Mass arithmetic:** element constants, checked neutral mass and signed parent/ion transformations agree (`chem.rs:372`, :398, :425). `parse_decimal` correctly implements half-even rounding, including fractional carry (:339–368).
- **Ion-H cast:** `hydrogens as u16` at `chem.rs:445` is safe for a successful ion: non-negative heavy mass and checked m/z bound imply at most **4,261 ion hydrogens**.
- **`tolerance_u32`:** bounds are correct: `hi*t ≤ 429,496,000`, second numerator `≤ 19,989,000`; values above 1000 are rejected (`chem.rs:463–477`). Decision sums use `u64` (:496).
- **Grammar at `Limits::V0`:** masks and legality agree with Python, including root, STOP, budget, unused fields, closure pointers and both caps (`grammar.rs:192`, :223, :252, :346).
- **Canonical minimum:** no non-minimal-return bug found. Pruning is strictly `tokens > best[..len]` (:679); equal prefixes continue. Every traversal has the same final token count, so the slice is safe. Candidate sorting changes visitation order only (:765); head advancement preserves FIFO discovery (:718). Roots and child extensions count expansions (:604, :767); exceeded budgets return an error, never an incumbent trace (:634).
- **Recipe:** cut enumeration, graph grouping, dropping canonicalization failures before matching, accepted-shift union, anchors and dropped-weight formula agree (`targets.rs:116`, :293, :343, :373, :408), except the tie and sentinel issues above.
- **Formula window:** widened bounds form a safe superset; neutral-row error plus precursor/adduct bound is conservative (`formula.rs:258–260`, :305–321). Saturation does not create false acceptance within supported tolerance bounds.
- **Spectrum fields:** shapes, schema version, bucket catalogue, duplicate spectrum IDs, intensity/energy defects, polarity/adduct rules, precursor range, over-capacity, tolerance defaults, raw truncation and ignored padding are correctly checked (`contract.rs:194–390`). Invalid intensity scale, known flag, energy count, tolerance and instrument IDs produce configuration errors.
- **Generation ranges:** K, F, temperature, normal step limits and unsupported Beam handling are correct (`contract.rs:465–494`). Memory preflight is not implemented by this validator.

**5. Tests and P1 coverage**

The fixture comparisons are substantial. Read-only regeneration reproduced all **25 molecule records**, **48 tolerance cases**, **8 decisions**, and **18 invalid traces**.

Assertions comparing production calculations with themselves include:

- Permutation baseline derived by the same canonicalizer (`ms2_chemistry.rs:444–455`).
- Hydrogen expectation derived from production atom types (:649–654).
- Formula-row mass checked using the production mass function (`ms2_targets.rs:414–417`).
- Formula-window inputs and recount derived from production mass/tolerance/error helpers (:485, :607–641).
- Loss-edge inputs derived from production loss masses (:678, :710).

These are useful consistency checks, but do not independently establish the chemistry/numerical expectations.

| Plan item | Coverage | Exact remaining gap |
|---|---|---|
| P1.1 | **Partly** — decimal parent masses, composition, tolerance and decisions have fixture comparisons. | Direct decimal ion masses/error intervals; explicit signed overflow/underflow and transformation/conservation cases. |
| P1.2 | **Partly** — grammar masks, invalid traces and RDKit identity partitions checked. | Independent canonical minima for 10–16 atoms; stereo-ignored identity; correct final-candidate validation. |
| P1.3 | **Partly** — constants, residuals and full-range tolerance sweep. | Independent unrounded ion intervals and overflow tests. The exact-mass comparison rounds the decimal first (`ms2_chemistry.rs:261`), weakening the bound check. |
| P1.4 | **Partly** — both V0 adducts, intrinsic-charge exclusions and shifts −2…2. | Named unsupported multi-charge/multimer/in-source-loss cases and direct independent transfer-ion expectations. |
| P1.5 | **Mostly** — hand-built request failures cover the main statuses. | Sentinel/out-of-range peak IDs, inconsistent counts, unknown-energy poison, and correct unknown-precursor/oracle status composition. |
| P1.6 | **Partly** — tables, absence/exhaustion, scoring caps and deterministic loss edges. | Independent complete joined-set/window expectations, saturation and mid-scan exhaustion counters; independent expectations for all supported losses. |
| P1.7 | **Partly** — rings, symmetries, reorderings, duplicate bonds, disconnection, valence errors, incomplete traces and isomers. | Legal four-closure prefixes and fifth-closure mask rejection; additional unused-field failures and independent large-graph traces. |
| P1.8 | **Partly** — residuals, incomplete prefixes and residual-two attachments. | Independent fragment hydrogen/composition expectation; candidate residual validation rather than self-derived H sums. |
| P1.9 | **Partly** — fixture masks, weights, anchors and retention checked. | Exact retention ties; a genuine top-16 overflow case; budget-failure removal before sharing; original IDs; identical embeddings with differing boundaries. Current fixture has neither differing-boundary classes nor multiple accepted shifts for one graph/peak. |

The fixture’s independent canonical traces stop at **9 atoms** (`make_fixtures.py:85`); its maximum traced closure count is **2**. Fixture targets never exceed **16**, and the retention test deliberately chooses unequal cutoff weights (`ms2_targets.rs:223–234`).

**6. P0 acceptance**

| Item | Status | Implementation-changing gap |
|---|---|---|
| P0.1 | **Satisfied** | FPNet preprocessing and caller-specific metadata are documented accurately (C:64–102; external `fpnet6.py:28–44`, :142–159). |
| P0.2 | **Partly** | Universal schema versions, retained formula probability representation, and count/ID/status invariants remain incomplete. |
| P0.3 | **Satisfied for V0** | Defined decimal masses, atom types, adduct algebra, H bookkeeping, attachments and unsupported chemistry (C:217–341). |
| P0.4 | **Partly** | Merge clustering, precision propagation, energy averaging and deterministic adduct ties remain unspecified (C:400). |
| P0.5 | **Satisfied** | Disjoint identity subsets, scaffold protocol, instrument holdout and pretraining exclusions are defined (C:40–60). |
| P0.6 | **Partly** | The all-unknown-metadata structure-prior control conflicts with fatal request validation (C:565–566). |
| P0.7 | **Partly** | V0 recipe is explicit; V1 vocabulary/adduct-ion mappings remain deferred despite the frozen-domain claim (C:358–365). |
| P0.8 | **Partly** | Formula-window membership, work-counter semantics, partial-search output and probability availability require choices currently supplied only by code. |

## Not checked

- No cargo, Rust tests, GPU execution, allocation profiling or training was run. No files were edited.
- Full audit/pilot regeneration and current source-file SHA-256 verification.
- Authoritative isotope-table attribution, competition permissions, and the provider’s energy conversion.