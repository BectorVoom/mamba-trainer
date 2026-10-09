# Codex review: the enumeration cache (task T6) and checkpoints, resume and step safety (task T1A)

Reviewer: codex exec (read-only), 2026-10-05, uncommitted tree over d39ec34. Verdicts: Part A (enumeration cache) reject — the attached cache header is not checked when a batch is served, a corrupted payload is accepted, concurrent saves share a temporary file; no counterexample to the key or the value with a matching header. Part B (checkpoints and step safety) reject — resume accepts a different export and keeps stale provenance, a non-resume load shuffles with the command-line seed, the guard counter is neither checkpointed nor an integer, and loss scaling can overflow the clip norm before unscaling. Fix task: F7.

Both exactness claims need fixes. **Part A: reject. Part B: reject.** This is a source review of the changing working tree; I ignored unfinished evidence additions, modified nothing, and did not run cargo.

**Part A — enumeration cache findings**

1. **P1 — Attached headers are not checked on cache hits.** [generate.rs:1300](src/models/ms2/generate.rs:1300), also [train.rs:1843](src/models/ms2/train.rs:1843).
   
   Concrete scenario: use the same domain, window, visit budget and precursor, but bounds A fitted to ethane and bounds B fitted to ethene. A cache built with A can contain an ethane candidate; enumeration under B rejects it. Attach cache A to the model carrying B through `set_enum_cache`: the identical meta key hits, because `expand_batch` checks only `window_m`. Cached generation/training scores A’s candidates instead of B’s device result.
   
   Minimal fix: check `cache.header().check_compatible(&current_header)` before expansion in both consumers. Attachment-time validation alone cannot cover generation configurations supplied later.

2. **P2 — Payload corruption can silently change enumeration results.** [enum_cache.rs:818](src/models/ms2/enum_cache.rs:818).
   
   Concrete scenario: save a productive, complete entry, then flip its stored `complete` word from `1` to `0`. Loading accepts it; cached readout reports incomplete support while device enumeration reports complete support. Changing a small heavy count can likewise pass mass representability checks because mass is recomputed.
   
   Minimal fix: add and verify a payload checksum, with a format-version bump. Structural validation does not establish payload integrity.

3. **P2 — Concurrent saves share a temporary file.** [enum_cache.rs:721](src/models/ms2/enum_cache.rs:721).
   
   Concrete scenario: two threads save different caches to the same path. Both use `<path>.tmp-<pid>`. After A finishes writing, B truncates that temporary file; A can rename it while B is still writing. The destination then exposes a partial file, and a crash can preserve it.
   
   Minimal fix: create a unique temporary file per invocation using exclusive creation, then rename that completed file.

The remaining enumeration assessment:

- **Key:** With an actually matching header and unchanged enumeration semantics, I found no equal-key/equal-header device-result counterexample. Artifact hashes cover domain, bounds and derived rare-table content. Domain error affects the meta window. Scored cap is bounded by `M` in the meta row. Enumeration uses integer tensors, so neural dtype and formula-table contents do not affect its result. `enum_lanes_max` controls refusal; `enum_dispatch_visits_max` controls chunking.
- **Value:** Padding matches `cand_pad`: twelve zeros followed by `u32::MAX`. Flags `1` and `2` are preserved. Heavy counts above 255, hydrogen above 1023, wrong source IDs and mismatched masses are refused.
- **Scratch state:** Hits leave `lane_stats` and `offsets` stale/uninitialized. I found no downstream consumer after a hit. A later miss rewrites count statistics and offsets before fill; readout and diagnostics consume candidates/counters. `bench_ms2_enum` performs device enumeration directly.
- **Statuses/refusals:** Counters are copied intact, and host reconciliation still runs against the current batch. Unknown precision and invalid-parent requests can share the sentinel key without losing their distinct host statuses. Lane ceilings, missing artifacts and configuration checks remain enforced for generation/training. Header enforcement is the exception above.
- **Concurrency:** The enumeration map has no interior mutability. Shared `Arc` ownership prevents mutation during expansion/upload; stats use atomics. Partial hits fall back for the entire batch.
- **File loading:** Integer encoding is little-endian; lengths use checked slicing; versions and header mismatches are refused. I found no malformed-slice panic. Resource handling is weaker: the entire file is read first, header compatibility is checked after parsing, and a tiny truncated file advertising one million entries triggers substantial map preallocation before rejection.
- **Jitter/reads:** `V=0` retains the previous path. `V>0` deterministically selects the fixed pool using seed, step and spectrum index; training tags are positive and evaluation uses tag zero. Driver precomputation uses the same jitter function and includes evaluation draws. Cache hits add uploads, no device reads.

**Part A test coverage and gaps**

The existing tests cover raw candidate/counter equality, padding and compact bounds, ordinary generation/training equivalence, partial misses, precursor changes, lane refusal, header comparison, basic malformed files, fixed-pool repeatability, and warmed read/launch counts. GPU downstream floats use tolerance comparisons, so those tests do not prove universal bit identity.

Missing cases by requested item:

1. Attaching a cache from different artifacts/configurations; dispatch-size invariance.
2. Explicit flag-2 round-trip; hit → miss on the same bucket; downstream training weights/moments.
3. Invalid parent mass, genuine lane-budget exhaustion, saturated counters, and full cached readout reconciliation for exceptional requests.
4. Consumer header mismatches; cached missing-artifact/configuration refusals beyond lane limits.
5. Concurrent shared-cache lookups and stats.
6. Valid-looking payload corruption, concurrent saves, allocation bounds, duplicate/trailing/invalid-entry cases and version refusal.
7. Batch reordering/subsetting independence, `V=0` regression, evaluation-tag behavior, and driver-built pool coverage.
8. Existing warmed read tests cover the ordinary path; exceptional-hit and hit → miss transitions remain untested.

**Part B — checkpoint and step-safety findings**

1. **P1 — Resume accepts a different export and preserves stale provenance.** [ms2_experiment.rs:540](examples/ms2_experiment.rs:540).
   
   Concrete scenario: checkpoint training on export A, then invoke `--load checkpoint --resume --train B` with the same table/configuration. The driver never compares B with recorded provenance. It applies A’s cursor to B and diverges at the first resumed batch. Without `--resume`, further training on B still saves provenance naming only A. Continuing a schema-1 checkpoint can save another checkpoint with no provenance.
   
   Minimal fix: compute current provenance for loaded runs too; require a matching export/subset for resume. For non-resume continuation, record the new training export while retaining previous training exposure.

2. **P2 — Non-resume loads shuffle with the CLI seed instead of the effective checkpoint seed.** [ms2_experiment.rs:1096](examples/ms2_experiment.rs:1096).
   
   Concrete scenario: load a checkpoint whose training seed is `41`, omit `--seed` so the CLI uses `1`, train and save. Batches shuffle with `1`, and the updated cursor stores `1`, while `train_config.seed` remains `41`. Subsequent `--resume` refuses the checkpoint at the cursor-seed check.
   
   Minimal fix: initialize `shuffle_seed` from `effective_train.seed`.

3. **P2 — Guard history is not checkpointed.** [train.rs:3274](src/models/ms2/train.rs:3274).
   
   Concrete scenario: skip one step, recover to finite weights, save, reload, then report a healthy step. Uninterrupted execution reports `skipped_steps_total=1`; resumed execution reports `0`, because construction initializes the counter to zero and loading never restores it.
   
   Minimal fix: serialize/restore the guard counter. Weights and AdamW clocks need not diverge here, but reported trainer state does.

4. **P1 — Loss scaling can overflow the norm before unscaling, disabling clipping.** [train.rs:2471](src/models/ms2/train.rs:2471).
   
   Concrete scenario: guard off, `loss_scale=256`, `grad_clip=1`, and one unscaled gradient `g=1e17`. The scaled gradient is finite, but squaring it produces approximately `6.55e38`, overflowing f32. `clip_factor` rejects the resulting zero clip factor and leaves only `1/256`; AdamW receives `1e17` instead of approximately `1`, producing vastly different moments.
   
   Minimal fix: unscale inside the norm reduction **before squaring**, then derive clipping from that finite unscaled norm. Rescaling an already-overflowed sum cannot repair it.

5. **P2 — The skipped counter stops counting in supported bf16 training.** [fused.rs:3873](src/tensor/ops/fused.rs:3873).
   
   Concrete scenario: CPU bf16 training skips 257 steps. The counter reaches `256`, where storing `256 + 1` rounds back to `256`; the report undercounts. F32 similarly stalls at `2²⁴`. This is precision loss, not explicit saturation or wrapping.
   
   Minimal fix: use a dtype-independent integer counter with defined saturation, and include it in the existing batched report read and checkpoint.

The remaining checkpoint/safety assessment:

- Weights, named AdamW moment pairs, optimizer clock, trainer steps, cursor and enumeration artifacts are restored. MS2 holds no EMA or persistent training RNG stream; initialization RNG is replaced by restored weights, and subsequent sampling/jitter keys derive from restored configuration and steps.
- Dtype, chemistry, recipe, grammar, traversal, spectrum schema and unknown checkpoint schema are checked with field-naming `Error::Config`. Schema 1 restores weights with a fresh optimizer and exposes `optimizer_restored=false`.
- The guard predicate is computed on-device without a read. Single, wide-multi and narrow-multi gated kernels preserve parameters and moments on skip; weight decay is also skipped. `last_step_skipped` describes the reported step, while the cumulative counter captures earlier unreported skips.
- Bias-correction advances on skipped steps and is explicitly documented and checkpointed.
- For the same explicit configuration, guard off and scale one retain the previous optimizer branches and arithmetic. Other trainers pass a one-element factor, so I found no regression in their shared AdamW path.
- Validation permits powers of two in `[1,65536]`; reported losses remain unscaled. The guard tests **scaled** loss, so finite unscaled loss is skipped when `|L| ≥ 3e38 / loss_scale`. It also tests scaled gradient squares, tightening the effective norm threshold to approximately `sqrt(3e38) / loss_scale`.

**Part B test coverage and gaps**

1. `exact_resume` compares losses, weights and moments with decay and clipping configured. It repeats one batch, manually inserts a cursor, and disables jitter. Missing: actual shuffle continuation across multiple epoch boundaries, jitter on, resume after skips, bf16, and continuation after artifact/head restoration.
2. Refusal tests cover dtype, chemistry, recipe and unknown schema. Missing: grammar, traversal, spectrum schema, missing required fields, and malformed optimizer state.
3. Provenance round-trip is covered. Missing: different-export/subset resume refusal, stale-provenance continuation and schema-1 continuation metadata.
4. Safety tests cover a nonfinite skip, established moments surviving it, and a later healthy report counting an unreported skip. Kernel tests cover all three variants. Missing: counter precision limits, strict predicate boundaries, resume after skip, nonfinite inputs across every kernel variant, and narrow-path decay on skip.
5. Existing shared-optimizer tests and unchanged branches support ordinary behavior. Missing: explicit guard-off/scale-one launch equivalence against HEAD across MS2 and RL/entity/graph/LM callers.
6. Scale validation and ordinary scale-256 comparisons are covered. Missing: reduction-overflow clipping, scaled-loss guard boundaries, supported bf16, no-clip combinations, and direct post-update moment/weight comparisons for scaled versus unscaled training.