# Codex review: scratch arena for the MS2 decode loop (task T3)

Reviewer: codex exec (read-only, on a snapshot of the tree), 2026-10-06. Verdict: request changes — same-stream reuse in the warmed decode loop looks sound; reuse across threads can reorder queued writes, the one-bank carry estimate undercounts the functional decode paths, and the tally example counts two calls as one. Fix task: T3F.

Found three issues. This was a static, read-only review; I did not run Cargo or independently reproduce the supervisor’s measurements.

1. **[P1] Cross-thread arena reuse can reorder writes.** [src/backend.rs:2096](src/backend.rs:2096)

   `can_mut()` excludes queued kernel bindings from its reference count. That is safe for consecutive launches on one stream, but the arena checks only device identity and size. CubeCL 0.10 assigns streams per thread; its scheduler orders dependencies using each binding’s recorded creation stream.

   **Scenario:** Thread A allocates and releases a scratch buffer. Thread B activates the same cloned arena, reuses that buffer, queues `fill(2)`, and drops its tensor without flushing. A reuses it again, queues `fill(3)`, synchronizes, and keeps its tensor alive. B then synchronizes. Expected: A’s live tensor remains `3`. Actual: B’s previously queued write can execute afterward and overwrite it with `2`; the handle still names A’s creation stream, so A’s later launch does not drain B’s queue.

   **Minimal fix:** Bind arena reuse to the allocation stream as well as the device, falling through on another stream. Also enforce that scratch tensors remain on that stream, or track and drain their last-use streams before recycling escaped tensors used elsewhere. A mutex around checkout alone does not order device work.

2. **[P1] The unconditional one-bank estimate undercounts functional decode paths.** [src/models/ms2/workspace.rs:516](src/models/ms2/workspace.rs:516)

   **Scenario:** Set `workspace.composed_step = true`, enable carry capture, or use a mixer configuration/backend that fails `step_in_place_supported`. `generate_decode_step` preserves `old_caches` while constructing new caches. Expected: the estimate covers concurrently live carry storage. Actual: `decoder_carries` charges one bank while at least two banks remain live. The composed freeze additionally constructs replacement tensors. `decode_scratch_bytes` budgets the in-place fused output chain, not these state-sized allocations; reaching its retention cap merely falls through to fresh allocations.

   Consequently, preflight can accept a memory limit that does not cover the selected execution path.

   **Minimal fix:** Select the estimate using the actual decoder execution mode and budget functional carry/freeze intermediates separately. Until that selection exists, retain the conservative functional allowance. CPU is not inherently a two-bank path, and `MAMBA3_MS2_SCRATCH=0` alone does not disable in-place carries.

3. **[P2] The tally example counts two calls as one.** [examples/ms2_launch_tally.rs:449](examples/ms2_launch_tally.rs:449)

   **Scenario:** Run the ordinary generate tally, with either scratch setting. The first `generate` executes after `start_launch_tally`; the added allocation probe executes another `generate_with_hook` before `report`, without stopping/resetting the launch tally or timer.

   Expected: launch counts and timings describe one warmed call. Actual: they include both calls, while per-step counts divide by one call’s decode-step count. The whole-call allocation number describes only the first call.

   **Minimal fix:** Use the allocation hook on the original tallied call and remove the second call.

The remaining audit results:

- **Lifetime and aliases:** The arena retains a clone immediately after allocation. Reuse becomes possible after the last external tensor handle disappears, rather than at scope exit. Tensor clones and reshapes preserve handle references, so escaping `prev_h`, `prev_heads`, functional caches, or outputs prevent checkout while alive. CubeCL’s pool reference plus the arena reference explain the `strong_count <= 2` threshold.
- **Same-thread queued launches:** Reuse within a step is ordered safely on the same stream, even before flushing. Earlier kernel bindings remain valid, and later overwrites execute afterward.
- **Uninitialized contents:** I found no stale-content dependency in the traced decode outputs. Embedding, norm, projection, attention context, residual, head, and mixer kernels cover their logical outputs. Vector widths divide the relevant extents; surplus lanes guard stores. Optional placeholders have compile-time-gated reads. Partial atom-memory updates target persistent buffers initialized to zero during decoder initialization, outside the step scope.
- **Exact-size exchange:** Different shapes and dtypes can exchange equal-byte buffers. That is valid for fully overwritten outputs. These constructors allocate through wgpu’s main storage pool; uniform and readback staging buffers do not enter this arena.
- **Scopes:** The drop guard clears activation on panic and `?` return. Same-arena nesting preserves activation; different-arena nesting deliberately executes under the outer arena. A worker thread does not inherit activation. Device identity prevents reuse across devices, including separately constructed device identities.
- **Growth and release:** Retention is capped by `max_bytes` and 64 buffers per size. The workspace caches at most four buckets; changing shapes eventually evicts buckets. `clear`, eviction, and bucket drop release arena handles to the runtime pool, whose reserved pages need not immediately shrink. Resident leases retain their own buckets until released.
- **Accounting:** Reuse correctly avoids `count_allocation` and `note_alloc`; retained handles remain resident in runtime memory accounting. `peak_alloc_bytes` measures the largest new allocation, not total residency. Switching scratch off after warming leaves existing retained buffers resident.
- **Stopped rows:** Their carries are read by subsequent recurrent steps, but those computations remain row-local and their logits are ignored by the stopped sampler. Final validation, identity, and readout consume trajectory records rather than decoder carries. Each generation initializes a fresh decoder state; resident bucket reuse does not revive old carries.
- **Outside MS2:** Without activation, constructors retain the old allocation behavior. They nevertheless add two thread-local accesses per nonempty allocation—reuse lookup and retention lookup. No arena mutex is acquired outside a scope.

The new tests’ sensitivity, assessed from source:

| Test | What removal would it detect? |
|---|---|
| `buffer_recycled_only_after_drop` | Removing recycling, retention, or hit/miss accounting fails. |
| `live_tensor_buffer_never_handed_out` | Removing the live-handle check fails; removing recycling entirely passes. |
| `id_tensors_recycle_too` | Removing ID recycling or live-handle protection fails. |
| `sizes_do_not_mix` | Removing exact-size selection or recycling fails. |
| `retention_is_bounded_and_clearable` | Removing the byte cap or clearing fails. |
| `nested_scope_same_arena_is_noop_and_other_is_refused` | Switching to the inner arena or losing nested reuse fails. |
| `outside_scope_allocations_behave_as_today` | Leaving activation installed after normal return fails. |
| `warmed_decode_loop_allocates_zero` | Removing recycling from any tested generation entry point fails; the OFF control is meaningful. |
| `decode_scratch_estimate_matches_arena_within_ten_percent` | Removing retention fails. Its upper comparison is constrained by that same estimate being the arena cap, so it cannot independently prove adequacy. |
| `scratch_on_equals_off` | Observable corruption fails; removing recycling entirely passes. |
| `scratch_alternating_buckets_stay_correct_and_flat` | Observable corruption/growth can fail; removing recycling need not fail. |
| `in_place_carries_match_functional_every_step` | Removing in-place state fails the explicit carry-presence assertion; incorrect live-row state fails comparison. It does not verify allocation count or the memory estimate. |
| Updated workspace estimate test | Restoring two-bank arithmetic fails its numeric assertion; removing actual in-place execution would still pass. |

Missing coverage includes cross-thread ordering, escaped clones/views, unwind restoration, device mismatch, the per-size cap, and functional-path memory accounting.

**Verdict: request changes. The same-stream warmed decode reuse looks sound, but cross-thread reuse, fallback memory budgeting, and tally contamination need correction.**