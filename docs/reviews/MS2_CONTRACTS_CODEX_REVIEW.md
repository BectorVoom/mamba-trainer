# MS2 contracts: codex review and disposition

Date: 2026-10-02. Reviewer: `codex exec` (codex-cli 0.160.0, read-only sandbox, reasoning effort high), reviewing
revision 1 of [MS2_CONTRACTS.md](../MS2_CONTRACTS.md) with the two JSON reports and the audit and pilot scripts as
they were then. The findings were checked against the code before they were applied; the review is preserved
below as received. This is a review of documents and scripts, not of an implementation.

## Disposition

| # | Finding | Checked | Action in revision 2 |
|---|---|---|---|
| 1 | Mass-error bound understated and incomplete | Yes: displayed 3-decimal residuals are not upper bounds | Residuals in nano-dalton rounded up; the bound counts the ion's own composition; the observation uncertainty `U` is a schema field and part of `E` (§5) |
| 2 | Schemas cannot represent the promised inputs | Yes | §3 rewritten: typed fields, peak ids, precision fields, two tolerances, instrument class, defaults and ranges, padding and sentinels, numeric status bits (§8) |
| 3 | Audit measured a broader domain than the frozen vocabulary | Yes: `in_domain` never applied the atom-type list | One predicate (`ms2_reference.classify`) shared by audit, pilot and fixtures; audit rerun: 1,750,154 spectra, 272,609 structures (the reviewer's recount) |
| 4 | Pilot denominators, shift cap, hydrogen check, top-16 cut, overlap claim | Yes | Pilot rewritten on the shared reference: fixed sample, common denominators, shift cap, non-negative hydrogens, retention, per-peak overlap with the unrelated molecule; the "noise floor" claim is replaced by the measured shared fraction and called a diagnostic |
| 5 | Embedding aggregation, anchors, failure handling, intensity scale, untagged SMILES | Yes: untagged fragment SMILES collapses S v2 and S v6 | §7.2 steps 3 to 7; linear intensity; identity cross-check uses type-tagged SMILES |
| 6 | Split, vocabulary and control contracts | Yes: vocabulary and pilot had used all folds | Fold 1 divided into validation, ranking and calibration by identity group; vocabulary, recipe and formula table chosen from non-test folds; instrument holdout protocol frozen; V1 domain stated with measured coverage; controls and containment defined (§1, §4.6, §7.3, §10) |
| 7 | Feasibility table conditional | Yes: `d_inner` is derived, not validated, and the defaults carry a convolution | `SsmConfig` frozen in §3.3 (4 groups, no convolution); the claim about memory limits removed |
| 8 | Minor numeric discrepancies | Yes | Corrected from the rerun reports; the RDKit mass comparison is now in the audit JSON |
| 9 | FPNet merged-count variants; candidate serialization | Yes | §2 states the per-caller counts; §3.5 defines length, padding, sentinels and the attachment field |

Not adopted: none. Left open and listed in §11 of the contracts: the provenance of the element-mass decimals and
the data provider's NCE-to-eV conversion.

## Review as received

## Findings

**P0 is partly complete; the “frozen” status is premature.** No files were edited. References below use repository-relative paths; FPNet filenames refer to the external `data/v4g/` directory named in the request.

1. **Blocker — incomplete mass-error interval (questions 3, 8).**  
   [MS2_CONTRACTS.md:291](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:291) defines `E` using rounded residuals and adds `0.42` for the electron. Several magnitudes are understated:

   | Element | Exact residual from the stated decimal, µDa |
   |---|---:|
   | H | +0.03223 |
   | N | +0.00443 |
   | O | −0.38043 |
   | F | +0.16273 |
   | P | −0.00158 |
   | S | +0.1744 |
   | Cl | −0.318 |
   | Br | −0.4 |
   | I | −0.1 |
   | electron | −0.420091 |

   The table’s three-decimal display is reasonable, but those displayed values are unsafe as upper bounds. For protonated `C6H12`, the actual arithmetic error is `13 × 0.03223 + 0.420091 = 0.839081 µDa`; even counting all 13 ion hydrogens, the displayed bound gives only `0.836`.

   `n_e` must count the **ion composition**, including `h_a+s`; precursor-to-parent conversion needs its adduct error too. Observed-mass quantization and upstream uncertainty are stated at lines 294–295 but omitted from the decision interval. This contradicts [design:175](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_DESIGN.md:175).

   **Fix:** use exact residual constants or outward-rounded bounds; include ion/adduct and observation conversion errors. Define whether upstream uncertainty participates in acceptance. The accept/reject/ambiguous inequalities are sound only with a complete bound and a clearly defined tolerance.

2. **Blocker — the schemas cannot represent all promised inputs and behavior (questions 6, 8).**

   - Section 2 accepts float32-only requests with mass filtering disabled ([contracts:76](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:76)), but `SpectrumBatch` supplies only integer m/z fields ([contracts:100](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:100)). There is no separate neural m/z input or defined float32 adapter path.
   - `exact_mass` means “at least four decimals” ([contracts:110](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:110)). Decimal count does not establish precision provenance. There is no observation uncertainty or separate precursor precision.
   - Original peak IDs disappear after host selection of 512 peaks ([contracts:112](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:112)); the design explicitly requires their preservation ([design:39](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_DESIGN.md:39)).
   - A single mass-tolerance field defaults to section 6, which specifies **10 ppm fragment** and **20 ppm precursor** tolerances. Override scope is unspecified.
   - Instrument metadata, collision-energy training statistics/scaling, invalid flag values, unsupported IDs, enum/bit encodings, and version-compatibility behavior are absent or incomplete. `N_raw` and the bucket catalogue are not schema fields.
   - Generation limits lack concrete defaults/ranges; formula-table byte/workspace limits are incomplete. Sections 3.2–3.5 also omit types for several fields.
   - `canonicalization_budget_exceeded` is promised at line 382 but absent from section 8. Warning statuses such as `raw_truncated` and `exact_mass_unavailable` are not distinguished from fatal request errors.

   **Fix:** freeze a fully typed schema, adapter contract, peak-ID mapping, precision metadata, distinct tolerance semantics, numeric status encodings, and validation/error rules.

3. **Major — the audit measures a broader domain than the frozen vocabulary (questions 2, 8).**  
   [audit_casmi.py:229](/Users/ods/Documents/mamba-trainer/tools/ms2/audit_casmi.py:229) checks element/charge/isotope/radical restrictions, but never excludes the five dropped joint atom types before setting `in_domain` at line 250. `pilot_targets.py:55–58` has the same omission.

   Consequently, these document figures match their JSON reports but **do not describe the frozen 17-type domain**:

   | Claim | Report evidence | Corrected full-data recount enforcing the 17 types |
   |---|---|---|
   | Request coverage | `casmi_audit.spectra.v0_request_domain.in_v0_domain = 1750499` | **1,750,154**, or **68.914336%** |
   | Structure coverage | `casmi_audit.structures.in_v0_structure_domain = 272638` | **272,609** |
   | Identity-fold counts | `structures.folds.fold_identity["0"…"4"].in_v0_domain` | **54,532 / 54,506 / 54,521 / 54,519 / 54,531** |

   The omission affects 29 structures and 345 otherwise eligible requests. Precursor-within-ppm percentages also require regeneration for the strict domain.

   Exclusion order is **adduct → polarity → structure → precursor**, explicitly first-match-only (`audit_casmi.py:82–92`). If the document’s listed order is intended as **adduct → structure → polarity → precursor**, 83 overlapping rows move: structure exclusions become **33,861**, polarity exclusions **5,850**, before applying the vocabulary correction. Accepted coverage is unchanged by that reorder.

   **Fix:** share the frozen domain predicate, state exclusion priority, and regenerate coverage reports.

4. **Major — pilot bugs and omissions undermine the measured recipe comparison (question 2).**

   - **Missing denominator rows:** `pilot_targets.py:211–215` skips peak counters whenever the current molecule has no candidate. Thus `target_pilot.grid[1].peaks = 13323`, versus `grid[4].peaks = 13656`. The one-cut/10-ppm claim of **4.8%** uses a conditional denominator. Its 644 matches over all 13,656 peaks give **4.715876%**, rounding to **4.7%**.
   - **Wrong-parent null is conditional:** lines 211–222 exclude rows with no current candidates, the first sampled molecule, and rows whose previous molecule has no candidates. Its rates therefore use different peak/spectrum populations from the true-parent rates.
   - **Hydrogen shifts are uncapped:** `match():134–136` permits `|s| <= c`, so the three-cut row allows `s=±3`, contradicting `|s| <= min(c,2)` ([contracts:218](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:218)).
   - **Negative ion hydrogen counts are allowed:** graphs contain no hydrogen-total field and `match()` never checks `H_parent+h_a+s >= 0`. A synthetic zero-H graph is accepted with `[M-H]-`, `s=-2`.
   - **No top-16 retention or dropped-weight measurement:** lines 225–238 count every matching graph. This additional difference from `q-cut-v1` is omitted from the approximation disclosure at contracts:358–360.
   - **The null is not a measured false-label floor:** aggregate wrong-parent/true-parent rate ratios do not measure the intersection of matched peak sets. The “one explained peak in four … would also be explained” claim at contracts:350–351 does not follow from these counters.

   **Fix:** use a fixed sample and common denominators, include zero-hit rows, enforce hydrogen rules, implement retention, and measure matched-set intersections if claiming overlap. Treat unrelated-parent matches as a diagnostic, not a demonstrated noise floor.

5. **Major — target identity and evidence semantics remain ambiguous (questions 4, 5).**  
   [Contracts:326–334](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:326) leaves unresolved:

   - How to combine different **embeddings** of one identical labeled graph when their boundary-bond counts differ: minimum `c`, union of permitted shifts, or another policy?
   - Whether evidence stores every matching shift/embedding or selects one.
   - Whether canonicalization failures discard weight before peak sharing, or require recomputation afterward.
   - Whether recipe intensity is linear relative intensity or the square-root output obtained by literally applying section 2.

   Further, the planned RDKit fragment-SMILES cross-check is insufficient for the stated atom labels ([contracts:384](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:384)). Using the pilot’s `describe()`:

   - Whole `CSC`: sulfur type `S H0 v2`.
   - C–S–C fragment of `CS(=O)(=O)C`: sulfur type `S H0 v6`.
   - Both serialize as **`[CH3][S][CH3]`** (`pilot_targets.py:117–118`).

   They are distinct graphs under section 7.4 but share that SMILES.

   **Fix:** define embedding aggregation and anchor selection; specify linear weights; cross-check identity using all atom-type labels, including parent valence.

6. **Major — evaluation/split contracts contradict or defer design requirements (question 8).**

   - The same fold is assigned validation, reranker fitting, and calibration ([contracts:40](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:40)), without disjoint subdivisions. [Design:150](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_DESIGN.md:150) explicitly separates fitting and calibration evaluation.
   - Vocabulary selection uses all audited structures (`audit_casmi.py:198–201, 255–278`), and the pilot has no fold restriction (`pilot_targets.py:160–165`). This conflicts with “test untouched until P9” and the design’s training/validation-only dictionary policy (design:280).
   - Instrument classes are named, but the holdout protocol is deferred to P9.1 (contracts:45–47); P0.5 requires frozen instrument splits.
   - V1 chemistry/domain rules are absent despite P0.7.
   - The metadata-only control removes peaks (contracts:438–439), triggering mandatory empty-spectrum abstention (contracts:85). Its diagnostic execution path is undefined.
   - FPNet, set-encoder, and parameter-matched Transformer comparisons from design:284 are absent. Metric aggregation and containment’s induced/non-induced matching convention need explicit definitions.

   **Fix:** freeze disjoint evaluation subsets, restrict recipe/vocabulary selection appropriately, define executable controls and baseline comparisons, and specify containment and metric aggregation.

7. **Major — feasibility is conditional, not a complete carry budget (question 7).**  
   The recurrent-state arithmetic is correct, but [contracts:125](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:125) incorrectly attributes `n_heads*head_dim=d_inner` checking to existing validation:

   - `d_inner()` is **derived** as `heads × head_dim × rank` (`src/ssm/config.rs:164–166`).
   - Validation checks group divisibility and other constraints, not that claimed equality (`config.rs:203–249`).
   - Default `n_groups=24` and `conv_kernel=Some(4)` (`config.rs:138,143`). Merely changing heads to four leaves invalid groups, and convolution requires additional carries.

   **Fix:** freeze SISO, rotational dynamics, valid `n_groups`, and either `conv_kernel=None` or its carry budget. Remove the assertion that these partial byte counts establish feasibility under “any memory limit” (contracts:424–426).

8. **Minor — other numerical/provenance discrepancies (question 2).**

   - **“Other” instruments 4.3% → 4.2%.** Full grouping of `raw/train.parquet.instrument_type` under contracts:45–47 yields `106203 / 2539608 = 4.181866%`. The JSON retains only the top 40 categories (`audit_casmi.py:41–45`), so it cannot establish the full class total alone.
   - **Two-cut specificity ratio 4.3 → 4.2.** `target_pilot.grid[4].matched_peak_fraction = 0.15355887521968364` divided by `.wrong_parent_matched_peak_fraction = 0.03648618946442963` is **4.208685**.
   - **Pilot target-size range is 2–16**, not exclusively 3–16: `targets_at_cuts2_ppm10.size_histogram["2"] = 3`. The later disclosure acknowledges those three exceptions.
   - **98 “elements outside §4.1” is mislabeled.** The audit excludes explicit H atom vertices too, although H is in §4.1. A full recount finds **80** structures containing elements outside that table and **18** additional structures whose only element outside the heavy-atom set is H (`audit_casmi.py:229–230`).
   - The RDKit mass differences at contracts:159–160 are independently correct, but neither cited JSON contains that measurement, contrary to contracts:6–7.

   **Fix:** correct the values/labels and store complete instrument and mass-comparison evidence.

9. **Minor — FPNet metadata and failed-candidate serialization need qualifications (questions 1, 6).**

   The peak-processing/collation table itself has **no mismatch**. However, section 2’s universal merged-count description omits training’s lower clamp: `traindata.py:76,81,95` uses `max(1,count)` before capping; inference singles use `min(count,8)` (`engine.py:345`), while merged inference sums `max(1,count)` (`engine.py:359`).

   `CandidateBatch.length` promises START and STOP (contracts:144), although `truncated` explicitly means no STOP (contracts:394–395). Padding outside `length`, failed formula-row sentinels, and output packing after abstention/deduplication are unspecified. The promised `attachment_partition=unknown` also has no corresponding schema field.

   **Fix:** document caller-specific counts and define actual emitted length, padding, failure sentinels, and constant/explicit attachment semantics.

## Verified correct

- **1 — `prep_peaks` and `collate`:** float64 conversion; `0<mz<=precursor+2`; empty/max≤0 handling; normalization **before** floor; intensity top-160; ascending m/z; square root **before** float32 cast; float32 empty arrays; zero padding with `True` meaning padded; empty spectrum’s slot 0 unmasked. Evidence: `fpnet6.py:28–44,142–159`. Deterministic project ties, cap 128, and empty abstention are declared project differences.
- **2 — remaining measured figures:** dataset/hash/fold counts match the reports as generated; peak statistics, precursor accuracy percentages, energy percentages/counts, vocabulary frequencies, bond proportions, and the remaining pilot table entries agree at displayed precision. Pilot p95 “24” is rounded from `grid[4].targets_per_spectrum.p95 = 24.100000000000023`.
- **2 — boundary counter:** `enumerate_subgraphs():88–102` is correct. Every cut edge crosses components; each increments both endpoint components. For a fixed atom set, the boundary count is invariant across admissible cut sets. No boundary-count bug found.
- **3 — masses/adducts:** every integer mass is the correct rounding of its stated decimal. `[M+H]+`: `M=mz−m_H+m_e`; `[M-H]-`: `M=mz+m_H−m_e`. The signed general adduct formula is correct (contracts:206–210).
- **3 — u32 tolerance:** computable without overflow for the requested bounds. With `q=m/10000`, `r=m%10000`, `a=q*p`:

  ```text
  tol = a/1000 + (10000*(a%1000) + r*p)/10000000
  ```

  Integer divisions are floors. `a≤429496000`; the second numerator is ≤19,989,000; maximum tolerance is 429,496 µDa. Splitting alone is insufficient if an implementation reconstructs `a*10000` in u32. The corrected 16-heavy-atom ion arithmetic bound remains below 9 µDa.

- **4 — BFS/grammar:** all three statements hold for the specified discovery-order, FIFO BFS. Any lower-numbered neighbor than the parent would already have discovered the new atom; queue processing makes parent pointers non-decreasing; other earlier neighbors therefore exceed the parent. **No counterexample applies to that traversal.** Exhaustive minimum traces are a complete invariant because they losslessly encode the labeled graph and isomorphisms preserve the traversal set. The work cap makes this a partial canonicalizer that can abstain. `T=2+A+R_max=22` is correct. Rules 5–8, with valid atom types and the structural rules, preserve nonnegative residual valence and forbid duplicate bonds.
- **5 — already resolved:** candidates are logically **induced** because every removed edge must cross components. Ring closures mean internal non-tree edges, `|E|−|V|+1` (`pilot_targets.py:116`; design:138). The zero-cut set admits the whole molecule when it has **3–16 atoms** and ≤4 closures. Equal-weight target ties are explicitly broken by canonical trace (contracts:333).
- **7 — memory arithmetic:** angle is `[batch,heads,d_state/2]` (`scan.rs:369,413–414`). Carry totals are **16,448 elements / 65,792 bytes per trajectory/layer**, **2,105,344 elements / 8.03125 MiB** overall; encoder output plus stated K/V is **2.5 MiB**. Four heads ×64 channels with `d_model=128`, `d_state=32` is valid when the remaining configuration is valid.

| P0 task | Assessment | Gap |
|---|---|---|
| P0.1 | **Partly** | Peak contract correct; caller-specific count handling needs qualification. |
| P0.2 | **Partly** | Precision/ID paths, typed fields, capacities, encodings and errors incomplete. |
| P0.3 | **Partly** | Chemistry tables largely defined; embedding/mapping and unsupported-case behavior incomplete. |
| P0.4 | **Partly** | Energy conventions defined; tolerance override and merge-count semantics incomplete. |
| P0.5 | **Partly** | Inventory exists; instrument split and fitting/calibration separation unfrozen. |
| P0.6 | **Partly** | Targets/devices/controls named; control execution, baselines and aggregation incomplete. |
| P0.7 | **Partly** | V1 absent; recipe ambiguities, pilot defects and identity cross-check gap remain. |
| P0.8 | **Partly** | Core counts/order/T correct; table bytes deferred and complete carries/configuration unfrozen. |

Acceptance evidence: [tasks:41–50](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:41).

## Not checked

- A fully corrected pilot rerun; corrected null rates and recipe coverage cannot be recovered from aggregate JSON.
- GPU execution, allocations, latency, and training feasibility: no implementation changes or performance claims were tested.
- Authoritative isotope-table attribution and underlying competition licence permissions. The local source notice identifies component-specific terms rather than a blanket licence (`SOURCE-LICENSES.md:3–12`).