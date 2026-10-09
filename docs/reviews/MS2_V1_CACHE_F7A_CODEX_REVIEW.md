# Codex re-review: enumeration/evidence cache after F7A (format v3)

Reviewer: codex exec (read-only, on a snapshot of the tree), 2026-10-06. Verdict: the three Part A findings are resolved; the combined cache is rejected for two evidence-identity omissions (kept-peak capacity `n_peaks`, element dtype), an unverified hash hit, no host memory budget, and regression tests that do not guard what they name. Fix task: F8.

The three original Part A findings are resolved, but the evidence cache still has exactness defects. **I would not accept the combined enumeration/evidence cache yet.**

This was a read-only source review of this snapshot. I modified nothing, ran no cargo commands, and excluded Part B, scratch/arena work, and functional-group files. Test assessments below are from source, not execution.

### Disposition of the original Part A findings

| Original finding | Disposition | Evidence |
|---|---|---|
| A1: attached headers unchecked at use | **Resolved** | Generation checks at [generate.rs:1545](src/models/ms2/generate.rs:1545); training checks at [train.rs:1944](src/models/ms2/train.rs:1944). Attachment also checks artifact fields. |
| A2: valid-looking payload corruption accepted | **Resolved for accidental corruption** | Format v3 checks every preceding byte before parsing the header or entry lengths, at [enum_cache.rs:1090](src/models/ms2/enum_cache.rs:1090). |
| A3: concurrent saves share a temporary file | **Resolved** | PID/sequence/time naming plus `create_new(true)`, at [enum_cache.rs:990](src/models/ms2/enum_cache.rs:990). |
| Additional loading concern: tiny file causes huge reservation | **Resolved in implementation; regression test insufficient** | Reservations use available bytes as a bound. The test checks rejection but does not check allocation. |

### Remaining findings

**1. P1 — Evidence identity omits the model’s kept-peak capacity.**  
[enum_cache.rs:540](src/models/ms2/enum_cache.rs:540), [generate.rs:936](src/models/ms2/generate.rs:936), [enum_cache.rs:1533](src/models/ms2/enum_cache.rs:1533).

Concrete scenario: build an Evidence cache with `n_peaks=16`, then attach it to a model with `n_peaks=32`, keeping artifacts, enumeration budgets, window, dtype and spectrum identical. Supply 17 eligible peaks. Uncached evidence sees 16 valid peaks in the first model and 17 in the second; the seventeenth can also explain an additional fragment. The header and evidence key agree, so the second model instead receives the first model’s `n_ev`, explained count and weight.

Minimal fix: include `n_peaks` in evidence identity, enforce it during build and lookup, and version the persisted representation.

**2. P1 — Evidence identity omits the floating element dtype.**  
[enum_cache.rs:135](src/models/ms2/enum_cache.rs:135), [enum_cache.rs:561](src/models/ms2/enum_cache.rs:561), [generate.rs:1718](src/models/ms2/generate.rs:1718).

Concrete scenario: build evidence with `E=f32`, then reuse the file with `E=bf16` on a supported CPU backend. The same host intensity words produce the same key, although upload, relative-intensity arithmetic, peak ordering and evidence-weight accumulation run in `E`.

For example, two eligible peaks with linear intensities `1.0` and `0.33333334`, with only the second explained, involve intermediate rounding in bf16. An uncached bf16 calculation need not equal the final bf16 conversion of the cached f32 calculation. Near an intensity tie or selection threshold, explained counts can differ too. Attachment accepts this reuse.

Minimal fix: stamp and check the evidence computation dtype. Integer enumeration entries can remain dtype-independent.

**3. P2 — A peak-row hash collision is not fully detected.**  
[enum_cache.rs:827](src/models/ms2/enum_cache.rs:827).

Concrete failure state: two rows have identical enumeration meta, peak count and m/z sum, but different intensities or peak arrangements, and collide on both FNV hashes. The lookup accepts the stored entry because its only secondary comparisons are count and m/z sum. Expected: compare the actual canonical inputs and miss on inequality. Actual: silently serve the other row’s evidence.

I have not constructed a genuine colliding byte pair; this is a conditional exactness defect, not evidence of frequent practical collisions. The implementation cannot support an unconditional “exact memo” claim with these checks.

Minimal fix: retain canonical evidence inputs and compare them on a hash hit, with collision buckets or an equivalent full-equality check. A stronger hash alone still gives probabilistic identity.

**4. P2 — Cache construction has no memory budget, and the reported byte count understates resident memory.**  
[enum_cache.rs:623](src/models/ms2/enum_cache.rs:623), [enum_cache.rs:692](src/models/ms2/enum_cache.rs:692), [examples/ms2_experiment.rs:638](examples/ms2_experiment.rs:638), [enum_cache.rs:852](src/models/ms2/enum_cache.rs:852).

Concrete scenario: build 100,000 distinct spectra with eight distinct jitter variants and 2,048 scored candidates per entry. Candidate/evidence vector payload alone is approximately **29.5 GB**, before maps, vector metadata, allocator overhead, retained input batches and serialization buffers. The driver first retains all variant batches, then builds both unbounded maps; `save` allocates another complete serialized buffer. The process can exhaust host memory while device-memory preflight succeeds.

`bytes()` explicitly measures serialized size. The driver reports that value; the footprint estimate does not account for these host maps and construction buffers.

Minimal fix: enforce a host cache-byte budget with safe refusal or eviction, build variants incrementally, and report resident host memory separately from file bytes.

### Exactness, integrity and concurrency assessment

For **ordinary production-built integer enumeration entries**, I found no new key omission:

- Jitter changes the actual precursor used to build meta; variant identity itself is unnecessary.
- `P`, `M`, scored cap, lane visit budget, artifact identity and chemistry version are checked or encoded.
- Lane ceilings still produce refusal on hits.
- Formula-table contents do not affect enumeration; downstream features are recomputed.
- Artifact replacement through the provided setters is caught by the per-use check.
- Evidence keys include host peak contents, intensity scale, fragment tolerance, m/z uncertainty, adduct, evidence work limit, evidence peak limit and hydrogen bound. Their omissions are findings 1–2.

**Dispatch bounds are scheduling inputs.** Enumeration and evidence kernels use absolute lane indices and preserve each lane’s work budget. Replaying an incomplete enumeration under another nonzero dispatch bound appears valid by source inspection.

Also, **candidate flag `2` means ambiguous**, not partially completed enumeration. Enumeration completeness is `counters[4]`; evidence has its own complete bit. Both are stored, but the new tests do not jointly exercise a productive exhausted enumeration across genuinely different chunk layouts.

**Checksum:** I found no tamper-authentication claim in the inspected implementation/documentation. FNV provides corruption detection, not protection against someone changing data and recomputing the trailer. Verification precedes parsing every file-supplied length. `std::fs::read` nevertheless allocates for the entire actual file before verification. Ordinary truncation is rejected by checksum or structural checks; a truncated payload with a recomputed checksum is still rejected when required fields are missing.

**Threads:** maps have no internal locks or interior mutation. Lookups and saves borrow `&self`; inserts require `&mut self`. Safe shared `Arc` use therefore permits concurrent readers, but not simultaneous insertion into that same object. External synchronization would be needed for mutable sharing. Atomic counters preserve totals after workers finish; reading the four counters is not a coherent simultaneous snapshot.

**Processes:** separate temporary files prevent partial destination exposure. Two writers still replace the entire destination: last rename wins, with no merge. I found no documentation of that lost-update behavior.

**Crashes:** returned save errors attempt temporary-file cleanup. A crash between creation and rename leaves an orphan temporary file; subsequent save/load neither cleans nor reports it. The destination remains separate from that incomplete file. The containing directory is not fsynced, so atomic visibility should not be interpreted as guaranteed rename durability after power loss.

### F7A speed assessment

The per-use header is **rebuilt**, but artifacts are **not rehashed** and JSON is **not regenerated**. Generation and training clone the two precomputed SHA strings and allocate the chemistry-version string on each cached batch.

That is avoidable allocation: compare borrowed artifact/version fields and current scalar configuration directly. It is a small new cost, not a demonstrated major regression.

I found **no new device readback on the hit path**. Existing hit costs include candidate/counter expansion, evidence-output allocation, lookup vectors, two per-spectrum allocations for evidence hashing, uploads, and counter cloning. These largely predate F7A.

The two benchmarks exercise device dispatch paths, not cached per-use header checks, so they do not measure this F7A overhead.

### New `f7a_*` tests: would reverting the guarded behavior fail?

All locations below are in [tests/ms2_enum_cache.rs](tests/ms2_enum_cache.rs).

| Test | Reversion sensitivity |
|---|---|
| `checksum_rejects_single_bit_flips`:1945 | **Yes.** Valid-looking payload changes must specifically fail checksum verification. |
| `old_versions_refused_by_name`:1982 | **Yes** for removing named version refusal. |
| `bounded_loading_million_entries`:2005 | **No** for reverting bounded reservation: allocating one million slots and then rejecting satisfies every assertion. **Fix:** measure/capture reservation size or allocation. |
| `duplicate_trailing_and_invalid_entries_rejected`:2038 | **Yes** for the enum duplicate, trailing-byte and invalid-flag checks exercised. Evidence duplicates are untested. |
| `concurrent_saves_same_path`:2105 | **Probabilistic.** The old shared temporary path can fail, but the barrier does not force overlap inside save. **Fix:** coordinate creation/write/rename phases. |
| `concurrent_lookups_exact_totals`:2157 | **No** for reverting production counter updates: it increments a test-local atomic. It checks only `Some`, not output equality. **Fix:** assert production totals and returned buffers. |
| `attach_rejects_foreign_bounds_generation_and_training`:2204 | **Yes** for reverting attachment-time bounds checks. |
| `per_use_header_mismatch_is_config_error`:2257 | **Yes** for reverting generation’s per-use header check. |
| `training_per_use_header_mismatch_is_config_error`:2299 | **Yes** for reverting training’s per-use header check. |
| `dispatch_invariance`:2344 | **No** for a multi-chunk indexing defect: the CHNO fixture has `P=1`, and both bounds fit its two lanes in one chunk. **Fix:** force different chunk counts and compare raw buffers, including exhausted entries. |
| `flag2_round_trip_and_invalid_flags_refused`:2380 | **Yes** for in-memory flag preservation and the tested insert refusals. It never saves/loads and never tests flag `0`. **Fix:** add those cases. |
| `hit_miss_hit_matches_three_uncached_calls`:2410 | **Yes** for basic hit/miss behavior; **no** for a defect confined to reuse of one unchanged bucket. `B` changes `1→2→1`. **Fix:** keep batch shape fixed while changing an uncached precursor. |
| `cached_step_weights_and_moments_bit_identical`:2448 | **Yes on CPU** for divergence in the exercised step. Explicitly skipped on GPU; uses f32 and the Counts layout. |
| `exceptional_readout_cached`:2503 | Checks output/status parity, but **would pass if cache serving were disabled**. **Fix:** assert a hit and exercise genuine exhaustion/saturation explicitly. |
| `cached_missing_artifact_and_config_refusals`:2550 | **Yes** for missing-artifact attachment refusal and the selected schema refusal. |
| `batch_reorder_and_subset_keep_jitter_hits`:2602 | **Yes** for pool membership/hits; **no** for selecting the wrong member of the same spectrum’s pool. **Fix:** assert the exact step/index-selected variant. |
| `v0_unchanged`:2668 | Guards the selected V=0 miss and CPU report parity, but lacks an independent legacy draw oracle. **Fix:** assert the prepared precursor against that oracle. |
| `eval_tag_never_training_tag`:2714 | **No** for reverting the actual evaluation caller’s tag: it calls only the jitter helper. **Fix:** exercise evaluation preparation. |
| `driver_pool_covers_training_two_epochs`:2735 | **No** for reverting driver pool construction: it builds its own pool instead of invoking the example. **Fix:** exercise shared production pool construction or the driver. |
| `evidence_header_mismatch_refused`:2780 | **Yes** for the exercised attachment and generation per-use checks. |

**Verdict: reject the combined enumeration/evidence cache.** The original three Part A blockers are fixed. Evidence reuse across kept-peak capacities and dtypes remains incorrect, collision handling remains probabilistic, and several claimed regression guards do not test the behavior they name.