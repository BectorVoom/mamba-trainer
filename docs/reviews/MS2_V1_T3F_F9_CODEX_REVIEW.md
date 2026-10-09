# Codex re-review: scratch arena fixes (task T3F) and cache budget / top-F fixes (task F9)

Reviewer: codex exec (read-only, on a snapshot of the tree), 2026-10-06. Verdict T3F: request changes — a scratch-backed tensor that escapes to another thread and is used there can still be recycled ahead of that thread's queued work; three carry banks do not bound the composed freeze; the execution mode can change between preflight and the decode loop through a hook; the profiling example estimates mode-blind. Verdict F9: request changes — map tombstones break the prospective capacity bound and the per-slot byte charge is not an upper bound; chunk scratch below M = 256 is allocated but not estimated; the load size check can race a concurrent replacement. Everything else listed in the earlier reviews is resolved. Fix task: F10.

**Part A — T3F: request changes.** The tally fix and scope flag look correct, but stream safety and functional memory budgeting remain incomplete.

1. **P1 — Escaped tensors can still be recycled ahead of another thread’s work.** [src/backend.rs:2164](src/backend.rs:2164)

   **Scenario:** A creates a retained scratch tensor, moves it to B, and B queues `fill(2)` and drops it. A then checks out that buffer, queues `fill(3)` and flushes A; B subsequently flushes. Expected: A’s live tensor stays `3`. Actual: B’s delayed write can overwrite it with `2`.

   `handle.stream` correctly records the creation stream. It does not record subsequent use on B. CubeCL bindings preserve that creation stream, and queued bindings do not prevent `can_mut()`. Consequently, A’s next launch does not discover B’s pending write.

   **Minimal fix:** enforce stream confinement for scratch-backed tensors, or track foreign-stream use and prevent recycling until that work is drained.

   **Reachability:** Rust’s staged MS2 API exposes scratch-backed decoder tensors, including `DecoderState.prev_h`, which can escape or move between threads. Moving a resident lease alone does not expose its private scratch tensors. The Python classes are `unsendable`, and `detached` runs on the calling thread, so ordinary Python calls cannot produce this migration.

2. **P1 — Three carry banks do not bound the composed freeze.** [src/models/ms2/workspace.rs:595](src/models/ms2/workspace.rs:595), [src/models/ms2/generate.rs:1052](src/models/ms2/generate.rs:1052)

   **Scenario:** one decoder layer, no convolution, `composed_step=true`, `capture_carry_trace=true`, and large recurrent-state dimensions. Let one bank be `C=H+U+angle`. While freezing `last_u`, both complete old/new banks remain live, the frozen `h` is retained, and `kept_new`, `kept_old`, and the addition output coexist.

   Expected: the functional estimate covers this peak. Actual: the peak includes **`2C + H + 3U`**, which exceeds `3C` for the usual `H=U` and smaller angle tensor. Expanded masks add further storage. `decode_scratch_bytes` prices the fused output chain, not this freeze peak.

   **Minimal fix:** use a single selection kernel—or the existing in-place freeze—for the composed path; alternatively budget the actual freeze temporaries and masks.

3. **P1 — Execution mode can change after preflight.** [src/models/ms2/generate.rs:3347](src/models/ms2/generate.rs:3347)

   **Scenario:** warm a bucket with fused stepping enabled. Set the limit between the in-place and functional estimates, then call `generate_with_hook`; its `AfterEncoder` hook calls `set_fused_step(false)`. Expected: refuse the functional execution or keep the preflighted mode. Actual: preflight accepts one-bank execution, but `decoder_init` re-evaluates the mutable switch and builds functional caches.

   The same gap exists in packed/resident hook entry points. Changing workspace flags **between** complete calls is handled correctly: preflight runs again, and I found no cached estimate keyed without the mode.

   **Minimal fix:** latch the execution mode across preflight, initialization and stepping, or revalidate changed mode before allocating its state.

4. **P2 — Some estimate callers still assume in-place execution.** [examples/profile_ms2_substructure.rs:755](examples/profile_ms2_substructure.rs:755)

   **Scenario:** run the profiling driver with `MAMBA3_FUSED_STEP=0` and a limit between its one-bank estimate and the functional estimate. Expected: its initial feasibility check reports refusal. Actual: its mode-blind estimate accepts; the later production preflight refuses and `.expect("cold generate runs")` panics. The slope and stability estimates use the same mode-blind APIs.

   **Minimal fix:** pass the actual decoder mode to these estimates, or use a conservative functional estimate before constructing the decoder.

   Ordinary `generate`, `generate_packed`, and `generate_resident` now pass their intended flags correctly; resident correctly ignores capture because it never captures carries. Python bindings and the experiment driver delegate to these entry points.

The inactive flag correctly follows installed activation across nesting, normal return, early return and unwinding. I found no callable path with mismatched flag/arena state.

Earlier findings:

| Earlier finding | Disposition |
|---|---|
| Cross-thread reuse | Partially resolved: foreign-stream checkout is refused; escaped foreign-stream use remains unsafe. |
| One-bank functional estimate | Partially resolved: mode selection exists, but freeze peaks and mode changes remain uncovered. |
| Tally counts two calls | Resolved: one `generate_with_hook` supplies both measurements. |

New-test sensitivity, assessed from source:

| Test | Would reverting the corresponding fix fail it? |
|---|---|
| Cross-thread fallthrough | Yes for removing the checkout stream guard; it never moves an existing tensor to B. |
| Escaped clones/views | Yes for removing live-handle protection. |
| Unwind/early-return restoration | Yes for leaving activation installed; removing only the fast-path optimization would still pass. |
| Device mismatch/per-size cap | Yes for removing their respective guards. |
| Functional workspace arithmetic | Yes for reverting the numeric allowance; it does not verify execution. |
| Carry estimate per mode | Detects reverting the allowance, but measures post-step retained banks, not freeze temporaries; it independently supplies the state’s mode to the estimate and therefore misses broken production preflight selection. |

**Part B — F9: request changes.** Refusal fallback, lazy variants, evidence invalidation and routing proofs improve substantially, but admission and scratch accounting still have failures.

1. **P1 — Tombstones invalidate the prospective map-capacity bound.** [src/models/ms2/enum_cache.rs:237](src/models/ms2/enum_cache.rs:237), [src/models/ms2/enum_cache.rs:1651](src/models/ms2/enum_cache.rs:1651)

   **Concrete sequence:** fill an evidence table with 28 keys in 32 buckets, with an occupied run long enough that removing an interior key produces a `DELETED` control byte. Replace that key’s enumeration entry, evicting its evidence. Now evidence `len=capacity=27`, although the physical table remains 32 buckets. Insert a new evidence key whose probe chooses an `EMPTY` slot rather than the tombstone.

   Expected: predict the next table’s usable capacity, 56, and refuse when necessary. Actual: prediction is **`2×27+1=55`**; hashbrown grows to 64 buckets with capacity **56**. Set the budget to the prospective footprint using 55 slots: admission passes, but committed `resident_bytes()` exceeds it by 73 bytes. Debug builds panic at the assertion; release builds retain an over-budget accepted entry.

   **Minimal fix:** track physical table capacity, or a conservative capacity high-water mark, through deletion and growth; use it consistently in admission and residency accounting.

   The randomized test does not deliberately couple evidence metas to replaced enumeration metas, so it does not reliably exercise this deletion state.

   **Allocation accounting:** 81/73 bytes per *usable* slot are not table-allocation upper bounds. On x86-64, capacity 3 means four buckets: enumeration storage is `4×80+4+16=340` bytes versus a 243-byte map charge; evidence storage is 308 versus 219. Tombstone removal can also reduce reported `capacity()` without releasing table memory. Payload/bucket eviction credit is correct, but physical map storage needs bucket and control-padding accounting.

2. **P2 — Always-allocated chunk scratch is omitted below `M=256`.** [src/models/ms2/workspace.rs:746](src/models/ms2/workspace.rs:746), [src/models/ms2/workspace.rs:1716](src/models/ms2/workspace.rs:1716)

   **Scenario:** f32, `B=8`, `M=32`, either formula layout. Both constructors allocate `[8,1]` score and slot buffers. Expected: estimates include their 64 payload bytes. Actual: generation and training charge zero because routing cannot select chunked at this window size.

   **Minimal fix:** charge `B×ceil(M/64)×(elem+4)` unconditionally, matching allocation, or omit those allocations below the threshold.

   The new workspace test explicitly expects zero below 256 and therefore preserves this error.

3. **P2 — Metadata checking can race the subsequent unbounded file read.** [src/models/ms2/enum_cache.rs:2097](src/models/ms2/enum_cache.rs:2097)

   **Scenario:** budget 1 MiB; metadata observes a 512 KiB cache. A concurrent save atomically replaces the path with an 8 MiB cache before `std::fs::read` opens it. Expected: reject before allocating the oversized file. Actual: read allocates the replacement file before parsing rejects it.

   **Minimal fix:** open once, check metadata on that handle, and bound the subsequent read.

   For an unchanged file, metadata rejection occurs before reading. A successfully loaded cache preserves budget X through attachment and subsequent inserts, subject to finding 1.

Other requested checks:

- **A1:** generation and training skip evidence construction on enumeration misses. Later batch misses run both stages uncached; partial evidence is not uploaded. Lookup/hit accounting remains consistent.
- **A5:** replacement evicts every evidence bucket sharing the meta, including different jitter/content hashes. Identical candidates are also evicted: harmless, with recomputation cost.
- **A6:** feature-off removes the hook type, static, mutex use and save-path calls. `required-features` gates the test; it does not automatically enable the feature. Explicitly enabling `test-support` also enables it for examples in that build. Normal Python builds do not request it.
- **B1/B3:** buckets key B and M; larger shapes obtain appropriately sized buffers. Selection rechecks exact scratch shapes. Non-production chunk sizes allocate separate scratch, preventing retained-buffer overruns.
- **B3:** environment sampling uses `OnceLock`, with first-use semantics documented.
- **B2/B5:** isolated routing tests measure only selection launches. The plane pin compares identically configured, warmed search calls without an attached enumeration cache.

Earlier findings:

| Earlier finding | Disposition |
|---|---|
| A1: Evidence build aborts on refusal | Resolved in generation and training. |
| A2: Accepted insert exceeds budget | Partially resolved; tombstone growth still breaks the invariant. |
| A3: Variants materialized together | Resolved by lazy iteration and incremental driver consumption. |
| A4: Load ignores budget | Partially resolved; ordinary loads enforce it, concurrent replacement can bypass the initial size bound. |
| A5: Replacement leaves stale evidence | Resolved, including all hashes sharing the meta. |
| Save-hook production overhead | Resolved with `test-support` off. |
| B1: Missing chunk scratch estimate | Partially resolved; below-threshold allocations remain unpriced. |
| B2: Routing unproved/override interference | Resolved by isolated launch assertions and restoring guards. |
| Switch sampled repeatedly | Resolved. |
| Missing plane-route launch pin | Resolved. |
| Missing productive/incomplete fixture | Resolved by the asserted productive incomplete companion test. |
| Missing attached bf16 execution check | Added, with an explicit capability-refusal alternative. |

The new refusal tests catch reverting the **training** fix; they do not directly guard reverting generation’s separate branch. Replacement, lazy-variant, routing and scratch-reuse tests are sensitive to their respective reversions. The load test guards ordinary size/budget rejection, but does not measure pre-read allocation or exercise replacement races.

Static review only: no files modified, Cargo or tests run, or functional-group files reviewed.