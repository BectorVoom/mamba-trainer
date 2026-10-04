# Codex review: main-tree work of 2026-10-04 (profile harness third pass, candidate follow-ups, decoder capacities and validation, enumeration integration, ranking and packing)

Reviewer: codex exec (read-only), 2026-10-04. Verdicts: A reject, B accept, C accept-with-fixes, D reject, E reject; the findings are the follow-up tasks.

Static review only. I inspected status/diffs, traced callers with CodeGraph, and read the pinned CubeCL sources. No files changed, no cargo commands run, and the four excluded enumerator implementation/test files were not reviewed. Test conclusions below concern assertion adequacy, not verified execution.

**PART A — profile harness and driver**

1. **Major — WGPU `profile_ms` does not cover the whole stage.**  
   [examples/profile_ms2_substructure.rs:1537](examples/profile_ms2_substructure.rs:1537), [src/backend.rs:527](src/backend.rs:527).

   The emitted scope incorrectly describes the duration as covering the stage’s submitted passes. In pinned `cubecl-wgpu-0.10.0`:

   - `stream.rs:239` flushes previous work and registers the token.
   - `stream.rs:545` asks for timestamp writes when opening a compute pass.
   - `timings.rs:347` drains **newly initialized tokens** and creates their query set. With no new token, subsequent calls return no query set.
   - Automatic flushing at `stream.rs:457` ends that first pass. Later passes consequently have no timestamp writes.
   - `timings.rs:207` assigns the token’s end from `current`, which ordinarily still identifies its initial query set.

   Thus an ordinary, unnested profile returns **the first timestamped compute pass’s beginning-to-end duration**, not the sum or elapsed span of every pass in the closure. The reported approximately 0.4 ms for 3,255 launches is consistent with this implementation; attributing the discrepancy solely to host dispatch is incorrect.

   `device_span_plausible` warns but retains the misleading duration and scope. Its wall-time ratio cannot establish timestamp coverage.

   **Fix:** mark whole-stage device duration unavailable for this multi-pass WGPU path and correct the scope/documentation. A single public `client.profile` around the unchanged stage cannot provide it with this pinned implementation. A demonstrably single-pass configuration could, but would require controlling and proving pass boundaries; summing separately profiled operations changes the measurement and is not whole-stage elapsed time. Otherwise fix/update the runtime’s timestamp bookkeeping.

2. **Major — Enumerate profiling uses the table-only memory estimate.**  
   [examples/profile_ms2_substructure.rs:655](examples/profile_ms2_substructure.rs:655), [examples/profile_ms2_substructure.rs:854](examples/profile_ms2_substructure.rs:854).

   **Scenario:** `--formula-source enumerate` with a limit between the table-only estimate and the enumeration-inclusive estimate. The driver admits the configuration, uploads/initializes resources, then production `generate_preflight` refuses it; the cold call’s `expect` panics. The T+1 preflight has the same discrepancy.

   **Fix:** fit/size the artifacts on the host first and use `generation_with_enum` consistently for base, slope, stability, and reported estimates.

3. **Minor — synchronization regression coverage remains incomplete.**  
   [src/backend.rs:331](src/backend.rs:331), [tests/ms2_profile.rs:605](tests/ms2_profile.rs:605).

   **Escaping regression:** delete `device.synchronize()` inside `counted_synchronize`, retaining the atomic increment. The new counter test passes; the later synchronizing read masks the missing synchronization in the value test.

   **Fix:** instrument the actual synchronization operation, rather than a wrapper’s independent bookkeeping, or test a deferred failure surfacing before any subsequent sync/read.

The seven remaining problems from the second review:

| Earlier problem | Third-pass assessment |
|---|---|
| 1. Replica device workload | **Resolved for Table:** production calls the same workspace stages at `generate.rs:1332`; the stages hold `no_grad`, and device-session warmup uses production generation at driver `:1829`. Per-stage counter/output equivalence is tested at `tests/ms2_profile.rs:998`. Enumerate device mode remains explicitly unsupported at driver `:1799`. |
| 2. Callback panic destroys state/token | **Resolved:** type checked before removal; restoration guards and callback `catch_unwind` exist at `backend.rs:430,450,475,543,564,581`. The span catch is inside CubeCL’s closure, allowing `end_profile` to run. Recovery tests start at `tests/ms2_profile.rs:662,697,719`. |
| 3. Missing session identity | **Resolved:** identity checks precede removal at `backend.rs:424,538`; same/different-type nested-session tests at `tests/ms2_profile.rs:748`. |
| 4. Memory-limit propagation | **Resolved for Table; partly overall:** configuration carries the CLI limit at driver `:728`, and T+1 has a preflight at `:854`. Enumeration estimates remain wrong, as finding 2 explains. |
| 5. Launch attribution tests | **Resolved for pinned CPU/WGPU:** ordered boundaries, readout interval, independent slope, and individual stage pins at `tests/ms2_launch_budget.rs:225,245,280,332,342`. Other backends deliberately remain unpinned. |
| 6. Synchronization regression | **Partly:** removing the entire counted wrapper is detected, but removing its actual synchronization is not. |
| 7. Endpoint called peak | **Resolved:** `reserved_bytes_after` and explicitly sampled `peak_reserved_bytes_sampled` at driver `:1050` and `:1302`. |

Training boundaries now measure actual forward/backward/optimizer work through `step_with_boundaries` at [train.rs:1173](src/models/ms2/train.rs:1173). Allocation-free decoder acceptance remains correctly open.

**PART B — candidate-composition follow-up**

1. **Finding 2 — resolved: readout accounting matches the actual read.**  
   [workspace.rs:226](src/models/ms2/workspace.rs:226), [generate.rs:879](src/models/ms2/generate.rs:879).

   For F32 the expression is now `4BK(4T+A+4) + 4B(13F+11)`. Actual download bytes are compared with the estimate, including the previously failing `K=1,F=8` shape, at [tests/ms2_generation_footprint.rs:386](tests/ms2_generation_footprint.rs:386).

2. **Finding 3 — resolved: conditioning setup comes from the production prefix.**  
   [train.rs:829](src/models/ms2/train.rs:829), [train.rs:1017](src/models/ms2/train.rs:1017), [train.rs:1126](src/models/ms2/train.rs:1126).

   Production and inspection now share `ForwardPrefix`; production passes its `e_cond` to `decoder.teacher` at `:1072`. The out-of-window mode test at `tests/ms2_formula.rs:2523` and the absent-gold production gradient test provide meaningful coverage. The inspection hook itself stops at the prefix, so its bit comparison alone does not verify the later decoder argument.

3. **Finding 4 — resolved: poisoned `cand_mask_into` output.**  
   [tests/ms2_formula.rs:1869](tests/ms2_formula.rs:1869).

   Every lane starts as NaN, including lanes expected to become zero; every output is compared bitwise. Omitting zero-lane writes now fails.

4. **Finding 6 — resolved: first-call refusal is asserted without retries.**  
   [tests/ms2_generation_footprint.rs:455](tests/ms2_generation_footprint.rs:455).

   One refused call on a fresh workspace must preserve allocation/launch counters. Counter-sensitive verification has moved out of the concurrent formula/generation suites into serialized footprint tests.

5. **Tally-scope observation — resolved.**  
   [train.rs:136](src/models/ms2/train.rs:136), [tests/ms2_footprint.rs:405](tests/ms2_footprint.rs:405).

   The dedicated `ms2.gold_embed` tally asserts zero launches for `ScoredRowOrZero` and positive launches for Composition. This is stronger than comparing total launches.

6. **Backend launch-pin observation — resolved for CPU/WGPU.**  
   [tests/ms2_footprint.rs:500](tests/ms2_footprint.rs:500), [tests/ms2_generation_footprint.rs:271](tests/ms2_generation_footprint.rs:271).

   Training and generation have separate backend expectations. Their WGPU branches use a feature check; using `device.name() == "wgpu"` consistently would avoid misclassification in a future mixed-backend execution, but the present preceding CPU branch handles CPU.

No additional correctness defect identified in these specific fixes.

**PART C — capacities, validation, work, failure outcomes**

1. **Minor — the failure-probability normalization assertion does not exercise failure leaves.**  
   [tests/ms2_generation.rs:885](tests/ms2_generation.rs:885), [tests/ms2_generation.rs:994](tests/ms2_generation.rs:994).

   The test asserting “finished and failed probabilities sum to 1” simultaneously asserts that its failure vector is empty. The separate unsatisfiable-budget case correctly enumerates `[START]` and verifies absorbing failure, but assigns `p_fail = 1` directly at `:1030`.

   **Escaping regression:** probability accounting accidentally adds a token factor to failure leaves; neither normalization assertion exercises that calculation.

   **Fix:** run the same outcome-probability evaluator over the unsatisfiable-budget leaves and assert their summed probability. A mixed finished/failure domain need not be invented: under the current grammar STOP remains legal after a root exists.

   The enumeration is independent of the sampler twin: it uses `TraceState` masks and its own field-probability calculation. Its narrow `A=2,R_max=0` domain makes omitting CLOSE_RING continuations legitimate.

2. **Capacity and packing checks — conformant in the reviewed paths.**  
   [contract.rs:1485](src/models/ms2/contract.rs:1485), [contract.rs:644](src/models/ms2/contract.rs:644), [decoder.rs:190](src/models/ms2/decoder.rs:190).

   A/R/block ranges are configurable; generation accepts the derived minimum through T=64. Decoder layers and pointer widths use configuration, and mixer initialization receives the decoder SSM configuration, preserving independent `d_inner`. The 32-atom/four-block test configuration explicitly uses inner width 32 versus residual width 16 at `tests/ms2_decoder.rs:1212`.

   I found no remaining compiled V0 capacity in these paths. The fixed vocabulary widths and 64-row step embedding are contract constants. Atom-mask shifts stay below 32; pointer fields can represent all configured indices.

3. **STOP check and work accounting — conformant for well-formed records.**  
   [ms2.rs:7027](src/tensor/ops/ms2.rs:7027), [twin.rs:1235](src/models/ms2/twin.rs:1235), [contract.rs:842](src/models/ms2/contract.rs:842).

   A claimed finished record without terminal STOP is invalid; legal truncated records are unaffected. Replay also rejects illegal STOP fields and tokens after STOP. The paired negative test at `tests/ms2_generation.rs:3043` covers both cases.

   `work()` matches §3.3: `length > t` counts emitted-token invocations; `NO_VALID_ACTION && length == t` counts the failed invocation; length-zero never-started records are excluded.

**PART D — enumeration callers and integration**

1. **Major — table counter inequalities reject legal enumeration results.**  
   [contract.rs:1301](src/models/ms2/contract.rs:1301), [pack.rs:587](src/models/ms2/pack.rs:587).

   Both validators require `rows_joined <= rows_visited`. Enumeration visits count heavy vectors, while each vector can join several hydrogen compositions.

   **Concrete scenario:** one rare lane containing S₂, C/N/O caps zero, hydrogen range admitting H=0 and H=2, and a sufficiently broad allowed uncertainty around 65 Da. One heavy-vector visit can join both parity-valid formulas. The contract permits `visited=1, joined=2, scored=2`; host validation rejects it, causing generation to return an error.

   **Fix:** make counter validation source-specific. Preserve `scored <= joined`; for enumeration use its hydrogen-range bound and saturation semantics instead of the table inequality.

2. **Major — duplicate detection conflates different enumerated formulas.**  
   [ms2.rs:7048](src/tensor/ops/ms2.rs:7048), [twin.rs:1257](src/models/ms2/twin.rs:1257).

   Every enumerated formula has `formula_row = u32::MAX`, but duplicate detection still compares that row plus the trace.

   **Scenario:** two different enumerated parent compositions both permit the same one-carbon fragment trace. The later trajectory becomes `duplicate_trace` despite having different conditioning formula provenance.

   **Fix:** compare composition or scored-support identity for enumeration; retain table-row comparison for Table. Add a same-trace/different-formula enumeration regression.

3. **Major — training refuses excessive lanes after allocations and encoder launches.**  
   [train.rs:850](src/models/ms2/train.rs:850), [train.rs:862](src/models/ms2/train.rs:862), [train.rs:912](src/models/ms2/train.rs:912).

   **Scenario:** `B=17,P=16,384`; lanes exceed 262,144. Training uploads spectra, creates the oversized bucket, and runs the encoder before checking the lane limit.

   **Fix:** resolve/check artifacts and lane products before upload, bucket allocation, or encoding. Add a first-refusal training footprint test.

   Fixed default training limits are a reproducibility limitation rather than silently ignored `TrainConfig` fields: that configuration currently exposes no lane-limit fields. If configurable training limits are intended, add and checkpoint them explicitly.

4. **Major — exhausted empty support is incorrectly reported as `formula_absent`.**  
   [generate.rs:975](src/models/ms2/generate.rs:975), [generate.rs:1449](src/models/ms2/generate.rs:1449).

   **Scenario:** known precursor precision but an uncertainty producing `half > DEVICE_HALF_MAX`. No formulas are searched; readout adds fatal `formula_absent`, then reconciliation adds exhausted and clears complete. Contracts §9 reserves absence for a **completed** empty search, apart from the explicit unknown-precision rule.

   **Fix:** distinguish “no scored conditioning hypothesis” from completed chemical absence. Carry truthful device-side status into trajectory initialization and define the corresponding abstention representation without manufacturing `formula_absent`.

   Unknown precision should retain unavailable+absent with complete=0; arithmetic overflow should retain mass_overflow with complete=0. The host patches occur after `generate_readout` validates and constructs records, and workspace-stage readout omits them. Centralize reconciliation before validation—or preferably produce the contract counters/statuses on the device—so profiling, future resident output, and generation consume identical semantics.

5. **Major — the extra exhaustion probe breaks valid shuffled-spectrum experiments.**  
   [examples/ms2_experiment.rs:556](examples/ms2_experiment.rs:556), [examples/ms2_experiment.rs:582](examples/ms2_experiment.rs:582).

   **Scenario:** Enumerate + ShuffledSpectrum with a one-spectrum final evaluation chunk. The extra report probe passes the original batch and shuffled control directly to `model.generate`; it refuses batches below two. Larger chunks use simple rotation instead of the trainer’s molecule-aware donors.

   **Fix:** obtain exhaustion statistics through the established evaluation/donor path, or reuse request statuses from that evaluation.

6. **Major — `--enum-fit` does not enforce train-only fitting.**  
   [examples/ms2_experiment.rs:223](examples/ms2_experiment.rs:223).

   Default fitting correctly finishes before validation loading at `:258`, and uses no validation compositions. But `--enum-fit validation.json` is accepted and loads/fits those validation molecules at `:228`; filename ordering does not enforce the train-only contract.

   **Fix:** verify fitting-set split provenance and exclude validation molecule identities; record the fitting source identity. Reject a validation export supplied as `--enum-fit`.

7. **Minor — additional enumeration metadata is missing from the memory estimate.**  
   [workspace.rs:648](src/models/ms2/workspace.rs:648), [generate.rs:624](src/models/ms2/generate.rs:624), [train.rs:925](src/models/ms2/train.rs:925).

   Rare/bounds/lane_stats/offsets are correctly priced, but the additional `[B,8]` enumeration metadata upload coexists with `DeviceSpectra::meta` and adds another `32B` bytes.

   **Fix:** include that buffer in both enumeration estimates, preferably as reusable workspace metadata.

8. **Minor — checkpoints can load successfully without required artifacts.**  
   [train.rs:1549](src/models/ms2/train.rs:1549).

   **Scenario:** checkpoint training source is Enumerate, but both optional artifact JSON fields are missing. Load succeeds and the first forward fails for missing resident artifacts.

   **Fix:** require both artifacts whenever the source/reference requires them. Keeping outer schema 1 with defaulted optional fields is compatible with old table checkpoints; absence is legitimate only for those checkpoints.

9. **Minor — unavailable gold-not-scored reporting becomes a false zero.**  
   [examples/ms2_experiment.rs:606](examples/ms2_experiment.rs:606).

   **Scenario:** `--eval-only` has no training loss curve. The report emits `gold_not_scored_rate=0.0`, even if evaluation gold is entirely outside scored support.

   **Fix:** emit unavailable/null or calculate the metric from evaluation gold slots, with its denominator.

Normal-path source/rank/count construction uses source 1 and MAX table rows at `generate.rs:901,950,958`. Gold-slot lookup operates on enumerated `cand` compositions at `train.rs:959`; report counts travel in the existing packed read. The one-read warmed-generation and zero-read non-report-training tests are relevant, but do not address the defects above.

**PART E — ranking and packing**

1. **Major — packed validation accepts invalid scores and provenance.**  
   [pack.rs:450](src/models/ms2/pack.rs:450), [pack.rs:484](src/models/ms2/pack.rs:484).

   **Concrete inputs:** a one-filled-slot result with `score=NaN` passes: NaN is checked only during comparison with a previous slot. Infinite scores can also pass ordered comparisons. A legal stopped trace with zero formula counts, MAX rank/row, or a rank beyond `rows_scored` likewise lacks the CandidateBatch provenance checks.

   **Fix:** validate every filled score independently; enforce source/row/rank/count relationships, identity-resolution range, unique original trajectories, and composition-aware replay. Add independent corruption tests. Validation also cannot establish that omitted candidates had lower scores without the original candidate set; that limitation should be explicit.

2. **Major — u32 address arithmetic is not checked before launch.**  
   [ms2_pack.rs:323](src/tensor/ops/ms2_pack.rs:323), [ms2_pack.rs:525](src/tensor/ops/ms2_pack.rs:525), [ms2_pack.rs:849](src/tensor/ops/ms2_pack.rs:849), [pack.rs:983](src/models/ms2/pack.rs:983).

   Wrappers check shapes but narrow strides/counts to u32 and kernels multiply addresses in u32. Bounds checks happen **after** address arithmetic wraps.

   **Scenario:** a sufficiently large, shape-consistent buffer with `rows * record_width > u32::MAX`; later records address earlier records and may acquire multiple writers. The host twin has analogous narrowing and u32 loops.

   **Fix:** checked conversion/product validation for every input/output address domain before allocation or launch. Reject unsupported sizes rather than saturating/narrowing.

3. **Major — finite eligible scores are discarded by an undocumented stricter bound.**  
   [pack.rs:685](src/models/ms2/pack.rs:685), [ms2_pack.rs:169](src/tensor/ops/ms2_pack.rs:169).

   **Scenario:** a valid candidate with reranker score `3.1e38`, or finite raw log-probability `-3.1e38`, is excluded. §4.4 requires finite scores, not strict membership in `(-3e38,3e38)`.

   **Fix:** implement exact finiteness classification, including extreme finite F32 values, without NaN self-comparison. Add finite-extreme, infinity, NaN, and overflowing-sum cases to both host and kernel tests.

4. **Minor — device packing accepts `R > K`.**  
   [ms2_pack.rs:806](src/tensor/ops/ms2_pack.rs:806).

   **Scenario:** `B=1,K=1,R=2` with matching output shapes launches successfully, unlike the host API and specified `1 <= R <= K`.

   **Fix:** reject `returned > per_spectrum`; test the refusal separately from `R > eligible`, which remains legal.

5. **Minor — malformed formula-rank lengths can panic in the host API.**  
   [pack.rs:1233](src/models/ms2/pack.rs:1233), [pack.rs:1340](src/models/ms2/pack.rs:1340).

   **Scenario:** otherwise shape-consistent one-record input with empty `formula_rank`. The preliminary checks omit that field, then index it directly.

   **Fix:** check every consumed field’s length using checked products before constructing device-layout buffers.

For bounded, consistent inputs, kernel and twin ranking/gather/packing statements agree: descending score, ties by original trajectory index; MAX rank for excluded records; complete zero/MAX unfilled pattern; and `returned_count` has exactly one writer, output slot zero, at `ms2_pack.rs:736`. Actual array bindings are rank **5**, integer gather **5**, float gather **3**, and pack **6**, within the seven-binding capability requirement. Twin agreement does not detect shared contract mistakes such as the score cutoff.

**Verdicts:** Part A **reject**; Part B **accept**; Part C **accept-with-fixes**; Part D **reject**; Part E **reject**.