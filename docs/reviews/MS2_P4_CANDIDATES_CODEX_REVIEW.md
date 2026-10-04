# Codex review: candidate compositions and schema version 2 (B1)

Reviewer: codex exec (read-only), 2026-10-04, on the diff against the previously reviewed state. Verdict: reject; the six findings are the follow-up task.

B1 has correctness and test-coverage gaps that prevent acceptance. This was a static review of the patch, post-change files, and CodeGraph caller paths. I did not modify files or run cargo.

1. **Major — `formula_gather` reads outside an empty table.**  
   [`src/tensor/ops/ms2.rs:3725`](src/tensor/ops/ms2.rs:3725)

   **Scenario:** `B=1`, `M=32`, table shape `[0,2]`, counts shape `[0,10]`, and a fully padded window. The wrapper accepts these shapes. Although every slot has `ok=false`, the kernel unconditionally executes `table[safe * 2]` with `safe=0`. There is no valid fallback index in an empty array. The host twin correctly produces padding without accessing the table.

   Empty tables are permitted by `FormulaTable::default`, `from_compositions([])`, and table upload.

   **Fix:** Handle `rows==0` without any table load while still writing all 13 words of every candidate. Add a poisoned-output twin test for an empty table on CPU and GPU.

2. **Major — the memory estimate omits allocated formula buffers.**  
   [`src/models/ms2/workspace.rs:469`](src/models/ms2/workspace.rs:469), [`:491`](src/models/ms2/workspace.rs:491), [`:1112`](src/models/ms2/workspace.rs:1112)

   `top_lp` is calculated but never included in `items`. Neither estimate accounts explicitly for `window`, `counters`, or `top_count`; the training estimate also omits the allocated `top`, `top_log_prob`, and `top_counts`. The new gold feature/embedding path is absent from the training additions.

   **Scenario:** At `B=8, M=2048`, the omitted window alone occupies **131,072 bytes**. Increasing M therefore allocates additional memory that the estimate does not count. The generation readout estimate also remains unchanged despite packing `top_counts` into the read.

   The required `cand`, `cand_feat`, three generation head activations, six training head activation/gradient equivalents, and scores **are included**. Generation checks the estimate before request upload and workspace allocation, but incomplete accounting weakens that safeguard.

   **Fix:** Account for the complete formula workspace, gold path, and packed readout, using checked arithmetic. Test exact M-dependent item sizes and refusal around a meaningful boundary.

3. **Major — Composition conditioning’s required bit equality is not tested.**  
   [`tests/ms2_formula.rs:2233`](tests/ms2_formula.rs:2233), [`src/models/ms2/train.rs:753`](src/models/ms2/train.rs:753)

   **Scenario:** Gold is present in the scored window. The test compares the separately computed Composition embedding with another separately computed embedding. It never compares that embedding with the corresponding row of `scored.embedding`.

   Sharing `embed_rows` and feature bits is good, but the scored projection operates on `[B,M,10]`, while gold operates on `[B,10]`. Matmul tuning is keyed by problem shape, so calling the same row-network method does not itself prove identical device arithmetic.

   The test also reconstructs conditioning outside the trainer and applies the absent-row mask on the host. The finite-difference test similarly reconstructs the forward path. A regression in the trainer’s mode selection could escape both.

   **Fix:** Compare every in-window gold embedding element by `to_bits()` against the scored row, across supported capacities and CPU/GPU. Exercise the actual trainer conditioning path, including an out-of-window gold whose graph loss drives row-network gradients. I found no proven numerical mismatch; this is an unfulfilled regression gate.

4. **Major — the universal kernel twin/poison requirement is incomplete.**  
   [`src/tensor/ops/ms2.rs:6693`](src/tensor/ops/ms2.rs:6693), [`:4177`](src/tensor/ops/ms2.rs:4177), [`tests/ms2_generation.rs:1014`](tests/ms2_generation.rs:1014)

   The changed `init_trajectories` has no host twin. Its direct test allocates uninitialized outputs and checks only ten budget words per trajectory; it never compares all `traj_meta`, `state`, and `actions` elements.

   The additional new `cand_mask` has a host twin, but no direct poisoned-output, every-element comparison test.

   **Scenario:** An initialization regression leaves an action/status/state word unwritten, or a mask regression mishandles an ambiguous flag outside the few asserted positions. The current direct tests do not enforce the stated requirement.

   **Fix:** Add complete host twins/comparisons and poison all outputs. Include padded tops, no formulas, empty peaks, metadata bypass, and modulo assignment. Add synthetic duplicate candidates to the `gold_slot` test to verify first-match selection; the present randomized table is explicitly deduplicated.

5. **Minor — `CandidateBatch::validate` accepts inconsistent formula provenance.**  
   [`src/models/ms2/contract.rs:945`](src/models/ms2/contract.rs:945), [`tests/ms2_contract.rs:884`](tests/ms2_contract.rs:884)

   **Scenario:** Set nonzero counts, `formula_rank=5`, and `formula_row=5` while `rows_scored=0`. The new test explicitly accepts this. A table-source record with nonzero counts, a real rank, and `formula_row=u32::MAX` also passes the new checks.

   These contradict the field meanings: the rank is a slot in scored support, and a table hypothesis has a resident-table row.

   **Fix:** Require every real rank to be below that spectrum’s `rows_scored`; require a real table row for a table-source hypothesis. Update the new positive fixture’s counters and add negative consistency tests.

6. **Minor — the M=2048 refusal test exercises generation at M=32.**  
   [`tests/ms2_generation.rs:2021`](tests/ms2_generation.rs:2021)

   **Scenario:** Break propagation of `config.formula_window` into generation’s estimate. This test still passes: its standalone estimate uses 2048, but its generation request uses `tiny_generation()` unchanged, with M=32 and a one-byte limit.

   **Fix:** Set `cfg.formula_window=2048`, choose a limit separating the M=32 and M=2048 estimates, and assert no request allocation or launch occurs.

The existing-test expectation audit found these changes:

| Post-change location | Change | Assessment |
|---|---|---|
| `tests/ms2_contract.rs:330` | ChemistryDomain expects `SPECTRUM_SCHEMA_VERSION` instead of `SCHEMA_VERSION` | Legitimate; numerical expectation stays **1**. |
| `tests/ms2_contract.rs:374` | GenerationConfig rejection message expects version **3**, formerly **2** | Legitimate schema migration; rejection of 2 would now be incorrect. |
| `tests/ms2_contract.rs:383` | Same change for CandidateBatch | Legitimate schema migration. |
| `tests/ms2_contract.rs:392` | Same change for ModelConfig | Legitimate schema migration. |
| `tests/ms2_workspace.rs:71` | Table bytes: `37_859 * 48` → `37_859 * 88 + 1024 * 4` | Legitimate residency correction/addition: exact counts plus log table. |

These are exceptions to the literal “only launch expectations may change” restriction, but they follow the required schema/residency changes. **No existing tolerance was loosened, and no existing assertion was removed or weakened.** Other existing-test edits update inputs or call sites.

The kernel audit is:

| Kernel | Logical lane | Arrays | Result |
|---|---|---:|---|
| `formula_gather` | `(b,m)` | 4 | Writes all 13 words; empty-table load defect above. |
| `count_features` | `(record,e)` | 3 | Safe 1024 bound; writes every element, including invalid counts. |
| `formula_top` | spectrum | 5 | Comparison ranking; dense prefix; both padding sentinels and zero probabilities written. |
| `formula_top_counts` | `(b,f)` | 3 | MAX/out-of-range slots produce ten zeros without indexing that slot. |
| `gold_slot` | spectrum | 3 | First flagged composition match; literal MAX initialization avoids the specified CPU scalar-copy pitfall. |
| `cand_mask` | output element | 2 | Comparison selection and complete writes; direct twin test missing. |
| `init_trajectories` | trajectory | 6 | Initializes all outputs and reads retained budgets; full twin verification missing. |

All use `launch_1d_spans`; the inspected kernels select through comparisons rather than multiplication by masks. Gather, count-features, top, top-counts, and gold-slot tests really launch kernels, check launch errors, and compare complete poisoned outputs.

Other requested checks:

- **Gold semantics:** Flags restrict `gold_slot` to the scored prefix produced by Table search. It returns the first duplicate match by inspection. No new explicit device read appears on non-report training steps; the scored-gold count is read only in the packed four-scalar report at `train.rs:923`.
- **Generation:** No gold input reaches `generate`. Retained mass sums exactly `top_count` entries. Fatal requests clear counts and rank along with the existing fields.
- **Compatibility:** Version-1 defaults/loading and version-3 rejection follow the specified rules. TrainConfig defaults to `ScoredRowOrZero`. Formula-head parameter names, shapes, initialization order, and visitor order remain unchanged, supporting V0 checkpoint compatibility.
- **Upload bound:** Counts above 1023 are rejected before device uploads; tests cover 1023 and 1024.
- **Launch counts:** **No numeric old→new stage expectations changed in the footprint tests.** By source accounting, generation search gains one net launch: gather/features replace the old ID-slice/lookup pair, and top-counts adds one. Other generation stages are unchanged. Training additionally launches gold-slot, gold features/network, and scored-count operations; the gold network runs even in `ScoredRowOrZero`. The tests measure counts rather than assert numeric stage budgets, so absolute old→new counts cannot be reported as verified.
- **Contracts document:** The changed schema, defaults, field definitions, and scored-cap text agree with the code, subject to the provenance validation gap above.

**Verdict: reject.**

---

# Second review: the fixes

Reviewer: codex exec (read-only), 2026-10-04. Findings 1 and 5 resolved; 2, 3, 4 and 6 partly; the residual items are the next follow-up.

Static review only: no files modified and no cargo commands run. **Findings 1 and 5 are resolved; 2, 3, 4 and 6 are partly resolved.**

1. **Resolved — empty-table gather.**  
   [`ms2.rs:3816`](src/tensor/ops/ms2.rs:3816) dispatches `rows == 0` to a separate kernel binding **only `cand`**. That kernel writes ten counts, mass, flag and source—every one of the 13 words—without any table access ([`:3688`](src/tensor/ops/ms2.rs:3688)). This structurally avoids empty-array loads on every backend.  
   The poisoned-output twin test covers both padded and misleadingly live windows against `[0,2]`/`[0,10]` tables and compares the complete output ([`ms2_formula.rs:1248`](tests/ms2_formula.rs:1248)). CPU/GPU execution was not verified in this review.

2. **Partly resolved — memory accounting.**  
   Generation now includes every listed formula workspace buffer, including mask; training additionally includes retained tops and the gold counts/features/network/slot path. These additions and the total use checked arithmetic ([`workspace.rs:478`](src/models/ms2/workspace.rs:478), [`:1175`](src/models/ms2/workspace.rs:1175), [`:1287`](src/models/ms2/workspace.rs:1287)). Tests pin the exact M-dependent sizes at **32/128/512/2048** ([`ms2_workspace.rs:273`](tests/ms2_workspace.rs:273)).

   **Remaining defect:** the readout adds `top_counts` but retains an incorrect base formula ([`workspace.rs:435`](src/models/ms2/workspace.rs:435)). The actual read includes actions, top, top_count, counters, summary, top_counts, top_log_prob and stats ([`generate.rs:686`](src/models/ms2/generate.rs:686)). For F32:
   - Actual: `actions + 4B(13F + 11)`.
   - Estimate: `actions + 4B(K + 16 + 10F)`.

   **Failing scenario:** valid `K=1, F=8` undercounts readout by **72 bytes per spectrum**. The new test uses `K=8,F=4` and pins the estimator’s expression instead of deriving the actual read layout ([`ms2_workspace.rs:315`](tests/ms2_workspace.rs:315)).

3. **Partly resolved — trainer conditioning coverage.**  
   The new test does compare every in-window embedding element using `to_bits()` at **M=32 and 128** ([`ms2_formula.rs:2417`](tests/ms2_formula.rs:2417)). Both production and hook call the shared `condition_embeddings`.

   **But the hook is still a parallel reconstruction:** production performs preparation/encoding/search/scoring/gold-slot setup at [`train.rs:753`](src/models/ms2/train.rs:753); the hook repeats that setup with a separate formula buffer at [`:910`](src/models/ms2/train.rs:910). It does not invoke the forward path used by `step`, and it does not observe the embedding handed to the decoder.

   **Escaping regression:** production passes `ScoredRowOrZero` instead of the configured mode at its conditioning call, or passes a different embedding to `decoder.teacher`; the hook’s Composition bit-equality tests still pass. Extract the common forward prefix or expose the conditioning result from production forward.

   The out-of-window gradient test **is meaningful**: it disables decay, steps **only spectrum 1**, whose gold is absent, and checks row-network parameter movement ([`ms2_formula.rs:2561`](tests/ms2_formula.rs:2561), [`:2587`](tests/ms2_formula.rs:2587)). The formula loss masks absent gold, so this isolates graph-loss influence under finite arithmetic ([`formula_head.rs:380`](src/models/ms2/formula_head.rs:380)).

   `ScoredRowOrZero` **really skips the gold network**: its branch gathers and masks scored embeddings without calling gold `count_features` or `embed_rows` ([`train.rs:137`](src/models/ms2/train.rs:137)).

   The zero-row value comparison is justified. Negative gathered components multiplied by `+0.0` produce `-0.0` ([`:150`](src/models/ms2/train.rs:150)); this masking operation also exists in HEAD’s V0 path. The decoder adds the conditioning embedding to token embeddings ([`decoder.rs:329`](src/models/ms2/decoder.rs:329)). Accepting either zero sign does not hide nonzero values or NaNs and introduces no identified V0 numerical change. It does cease enforcing canonical **positive-zero bits**, which is separate from the required in-window bit equality.

4. **Partly resolved — twins and poisoned outputs.**  
   The `init_trajectories` twin mirrors the kernel’s initialization, slot validity, modulo assignment, budgets and failure fields ([`twin.rs:743`](src/models/ms2/twin.rs:743)). Its test poisons and compares **every element** of all three outputs, covering padded tops, no formulas, empty peaks, metadata bypass and modulo assignment ([`ms2_generation.rs:1083`](tests/ms2_generation.rs:1083), [`:1154`](tests/ms2_generation.rs:1154)).

   The direct duplicate `gold_slot` test is non-vacuous: two flagged matching candidates must choose slot 1; an unflagged earlier match must lose to slot 4 ([`ms2_formula.rs:1824`](tests/ms2_formula.rs:1824)).

   **Remaining gap:** `cand_mask` compares every element and exercises ambiguous flags, but **does not poison output** ([`ms2_formula.rs:1867`](tests/ms2_formula.rs:1867)). The wrapper allocates `Tensor::empty` ([`ms2.rs:4265`](src/tensor/ops/ms2.rs:4265)).  
   **Escaping regression:** omit writes for zero-flag lanes; zero-filled or reused memory can satisfy those assertions. The comment claiming allocator contents cannot match expected zeros is incorrect. Provide a poisoned-output launch path.

5. **Resolved — provenance validation.**  
   Real ranks must be below that spectrum’s `rows_scored`, and table-source formulas require a non-MAX row ([`contract.rs:965`](src/models/ms2/contract.rs:965)). Negative tests independently cover zero scored rows, rank equal to the boundary, and missing table-row provenance; positive fixtures now carry consistent counters ([`ms2_contract.rs:884`](tests/ms2_contract.rs:884), [`:907`](tests/ms2_contract.rs:907)).

6. **Partly resolved — M=2048 refusal.**  
   The real request now sets `formula_window=2048`, uses a limit strictly between matching M=32/M=2048 estimates, and checks no workspace bucket exists ([`ms2_generation.rs:2153`](tests/ms2_generation.rs:2153)). Production checks the estimate before upload/allocation ([`generate.rs:423`](src/models/ms2/generate.rs:423)).

   **Remaining test weakness:** counter changes are tolerated until one clean attempt occurs ([`ms2_generation.rs:2177`](tests/ms2_generation.rs:2177)).  
   **Escaping regression:** the first refused call allocates a temporary buffer or launches one lazy initialization kernel; the second call does neither. The test passes. Isolate counters and require the **first** refused call to leave both unchanged.

Additional observations:

- The new launch comparison resets **process-global counters inside the multi-test formula suite** ([`ms2_formula.rs:2501`](tests/ms2_formula.rs:2501)). Concurrent tests can contaminate its comparison or have their measurements reset. Also, “Composition launches more” alone does not prove zero gold-network launches; source inspection supplies that proof.
- Public test support adds `TrainerConditioning` and two hooks. The broader HEAD diff also changes `FormulaHead::score` and memory-estimator signatures. Those broader changes were already part of B1; no saved B1 code snapshot was provided to attribute them specifically to these fixes.
- Removing the unconditional gold path changes V0-mode launch counts relative to reviewed B1. The current branch restores scored-row gather/mask behavior. No numeric old→new launch expectations are pinned.
- The pre-existing expectation changes remain the schema/residency changes listed in the earlier review: schema rejection moves to 3 for migrated schemas; table bytes become `37_859*88 + 1024*4`. I found no additional removed assertion or loosened pre-existing tolerance.

**Verdict: reject.** The readout accounting defect and the remaining production-path/poison/refusal regression gates prevent accepting all six fixes.