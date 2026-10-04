# Codex review: profile driver / V0.6 tests (A1) and host formula enumeration (H1)

Reviewer: codex exec (read-only), 2026-10-03. Scope: uncommitted changes for P2.6, V0.6, P2.3 measurement, and the P4.9 host reference.

Reviewed statically against the contracts and pinned CubeCL source. No files changed; no cargo commands or build/test executables run.

Findings, most severe first:

1. **[src/models/ms2/formula_enum.rs:667](src/models/ms2/formula_enum.rs:667) — major: capacity truncation selects the wrong canonical prefix.**  
   Rows are retained in DFS order, then sorted at line 758. Sorting the retained subset cannot recover lower-mass rows discarded after capacity was reached.

   Concrete input: heavy caps `C=2, O=1`, other caps zero, heavy total ≤2, hydrogen fixed at zero; parent mass `20_000_000`, protonated precursor `21_007_276`, uncertainty `10_000_000`, ppm zero, capacity 2. DFS joins `C`, `C2`, `O`, `CO`. The output keeps masses **12 and 24 Da**, whereas the canonical first two are **12 and 15.994915 Da**. These compositions also pass the enabled filters.

   **Fix:** maintain a bounded collection of the smallest `(mass, composition)` rows while continuing to count joins. Add a capped test comparing the actual output with the uncapped canonical prefix. The current carbon-only capacity test misses this ordering disagreement.

2. **[examples/profile_ms2_substructure.rs:968](examples/profile_ms2_substructure.rs:968) — major: the training stage subtraction does not isolate backward work.**  
   “Forward” calls `teacher_eval`, which performs evaluation-only gold-selection operations and a device read at `train.rs:755`, under `no_grad`. `step` performs a training forward and different reporting setup. Consequently, `step − teacher_eval` subtracts work absent from the training forward, including readback time, and produces incorrect backward launch/allocation accounting. Clamping negative differences hides the disagreement.

   **Fix:** measure the actual training forward, backward, and optimizer boundaries within one training step. Until those boundaries are available, report the combined step and label isolated values unavailable. The optimizer’s current `launches: 0` and `allocation_calls: 0` at lines 999–1001 are particularly misleading: its work is included elsewhere, rather than measured as zero.

3. **[src/backend.rs:288](src/backend.rs:288), [examples/profile_ms2_substructure.rs:635](examples/profile_ms2_substructure.rs:635) — major: stage wall-time windows include previously queued work.**  
   `sync_span` synchronizes only after the body. The driver launches another encoder pass at line 635, then starts the search window without draining that pass. On an asynchronous runtime, search’s final synchronization waits for this earlier encoder work, although the search launch counter excludes it.

   Separately, `generate_total − standalone prefix times` at line 760 includes decoder initialization, validation, readback, and host candidate construction. It is not an isolated decoder-loop duration, and separately synchronized runs need not have additive timings.

   **Fix:** drain prerequisite work before resetting counters and starting isolated timers. Measure decoder boundaries directly, or label the residual as a combined tail rather than decoder time.

4. **[examples/profile_ms2_substructure.rs:316](examples/profile_ms2_substructure.rs:316) — major: timer metadata labels host timing as device timestamps.**  
   On timestamp-capable WGPU, `sync_wall_ms` comes from `std::time::Instant`, but `timer` and `timing_method` say `DeviceTimestamps`. The empty `client.profile` probe establishes runtime capability; it does not determine the clock used by `sync_span`.

   **Fix:** label these measurements `SystemTime` or `SynchronizedHostWallClock`, with a separate field for the runtime’s profiling capability. The test at `tests/ms2_profile.rs:149` checks the capability probe, not the emitted metadata.

5. **[examples/profile_ms2_substructure.rs:433](examples/profile_ms2_substructure.rs:433) — major: stability measurement bypasses the requested memory limit.**  
   For `--max-device-bytes 1 --stability 1`, normal profile records correctly refuse execution, but `profile_stability` still uploads the table, initializes the model, and generates using `GenerationConfig`’s default limit. It never receives the user’s limit.

   **Fix:** pass the limit into stability measurement, preflight its tested configurations before allocations, and record refusals consistently.

6. **[examples/ms2_formula_report.rs:360](examples/ms2_formula_report.rs:360) — major: unknown precursor precision produces fictitious recall before scoring.**  
   With `precursor_uncertainty_udalton = u32::MAX`, both search implementations correctly evaluate nothing. The report nevertheless computes a gold verdict using a saturated enormous error and constructs a nearly universal window. A valid gold formula can therefore contribute to `window`, `accepted_or_ambiguous`, and subsequent chemical-filter recall while actual search returns no rows.

   **Fix:** gate these stage predicates on known precision and valid parent-mass derivation. Keep the spectrum in the denominator and report unavailable precision separately. Add a report-level sentinel regression test.

7. **[examples/profile_ms2_substructure.rs:785](examples/profile_ms2_substructure.rs:785) — major acceptance gap: this does not establish allocation outside the decoder hot loop.**  
   The reported **4,074 allocation calls per request** alone cannot locate allocations. However, a positive warmed `T+1 − T` allocation slope, together with `step_logits`’ allocating tensor operations and `generate.rs:562–569`’s freeze operations, establishes allocation *inside* the loop. Flat reserved memory demonstrates pool reuse, not absence of allocation churn.

   **Fix:** keep P2’s allocation acceptance explicitly open, and preserve the per-step allocation evidence. Closing that acceptance requires reusable outputs for these operations and a meaningful zero-allocation decoder-window check. Architecture §5 already acknowledges this departure; the measurement cannot reverse that conclusion.

8. **[examples/profile_ms2_substructure.rs:754](examples/profile_ms2_substructure.rs:754) — minor: the launch-budget equation omits decoder initialization.**  
   The reported difference of 12 is credible for the reviewed V0 path: four K/V projections, six rotational-cache zero fills, and two zero fills for atom memory/previous hidden state. These are outside the listed stages.

   **Fix:** charge initialization to an existing stage or add `L_init`, so the declared budget actually equals the measured total. Test reconciliation rather than treating an unexplained remainder as acceptable across configurations.

9. **[examples/profile_ms2_substructure.rs:1168](examples/profile_ms2_substructure.rs:1168) — minor: the fixed-shape bucket count is sampled after alternating measurement.**  
   With `--b 1,8`, `identical.bucket_count` reports two buckets because the second bucket has already been created, even though the identical-call phase used one.

   **Fix:** capture bucket counts at each phase boundary. Also distinguish endpoint reserved-byte measurements from a measured peak or full stability series; `peak_reserved_bytes` currently samples one endpoint.

10. **[tests/ms2_generation.rs:1736](tests/ms2_generation.rs:1736) — minor: record-preservation assertions are weaker than their claim.**  
    The forced duplicate test genuinely exercises the device kernel, but `statuses.len() == 3` follows from constructing a vector over the input row count. A faulty kernel that cleared trace contents or lengths could pass while destroying the records.

    **Fix:** inspect returned action words, lengths, formula rows, and `FINISHED` bits for all three records. Preserve the current duplicate-bit assertions.

11. **[tests/ms2_profile.rs:178](tests/ms2_profile.rs:178) — minor: the tests do not establish synchronization.**  
    Removing `device.synchronize()` from `sync_span` would leave the host-only tests passing. The generation test also reads its output internally, completing the relevant device work before the span ends.

    **Fix:** add a synchronization-sensitive test with queued device work and no read inside the span, or verify propagation of a deferred runtime failure at the span boundary.

12. **[examples/ms2_formula_report.rs:120](examples/ms2_formula_report.rs:120) — minor: CLI narrowing silently changes the requested tolerance.**  
    `--ppm-tenths 4294967296` becomes zero through `as u32` and passes validation. Capacity similarly truncates on a 32-bit target.

    **Fix:** use checked conversions before range validation.

Claims in comments/docs that the code does not support:

- **“Per-stage device timestamps are unavailable from safe code”** at `backend.rs:280`, the driver header, and architecture §6.5 is too broad. The caller-thread `Rc` borrowing problem is real, but it does not prove that a runner-local profiling harness is impossible.
- **“The recorded timer … says which clock the wall time is”** is false on a device-timestamp runtime; the measurements still use host `Instant`.
- **“Forward is the report-free teacher pass”** at driver line 963 is contradicted by `teacher_eval`’s readback.
- The complete launch equation lacks the initialization term described above.
- The duplicate test’s claim that both complete records remain is stronger than its assertions.
- The brute-force reference is independent for composition enumeration, masses, residuals, and verdict logic, but **not fully independent end to end**: expected parent mass and tolerance at `tests/ms2_formula_enum.rs:347–348` use production helpers.

The nested-profiling alternative is viable, with qualifications. Pinned `channel.rs:151` executes submissions inline when already on the same device runner, so nested `exclusive`/`profile` calls avoid another thread hop. CPU profiling stores separate tokens in a map; WGPU profiling likewise tracks separate tokens and reference-counted starting query sets, supporting nesting.

However, **inline execution does not remove the public `Send` bounds**. Simply enclosing existing `Rc`-borrowing inner closures will still fail those bounds. A safe arrangement can create and destroy the model on the runner, keep it in runner-local storage, and use `Send` callbacks that access that storage without capturing `Rc`. Return only `Send` results. Preserve stream identity too: `profile` captures its stream before dispatch, whereas ordinary operations default to the current thread’s stream. `StreamId::executes` provides a scoped mechanism for keeping them aligned.

Correct, non-obvious points:

- The hydrogen ceiling/floor division handles negative intermediates correctly. The domain-wide residual bound is conservative, and positive-mass pruning plus inclusive caps does not miss representable accepted or ambiguous compositions when work limits do not intervene.
- The chemical filters are sound **necessary conditions for complete, connected V0 parent molecules**. S(v2) versus S(v6) changes the DBE numerator by four, preserving parity and only loosening the maximum-valence bound. They are not sufficient realizability tests or filters for arbitrary open-valence proposals.
- Exhaustion, absence, and support completeness otherwise follow §9 correctly, including continuing join counting beyond capacity.
- The launch/allocation slope method is appropriate for this fixed-dispatch warmed loop. Production counters are collected before later profiling windows.
- The forced duplicate test reaches `validate_trajectories → ms2_validate_kernel` with equal trace/formula pairs; the duplicate-bit assertion is non-vacuous. The graph-equivalent, trace-distinct pair is also meaningful.
- Gold derivation counts heavy elements and parent hydrogens correctly. Out-of-domain gold remains in the exported-spectrum denominator, and JSON contains aggregates rather than per-molecule records.

**A1: reject** — stage attribution and metadata need correction; P2 allocation acceptance remains unmet.  
**H1: accept-with-fixes** — fix canonical capacity selection, unknown-precision reporting, and checked CLI conversions.

---

# Second review: the fix of A1 and the runner-local profiling harness

Reviewer: codex exec (read-only), 2026-10-03, on a frozen patch. Verdict: reject; the remaining problems are the follow-up task A3.

Reviewed the frozen patch against HEAD, the two authorized test files, and installed CubeCL 0.10.0 sources. No files modified; no cargo commands run. References below use reconstructed post-patch line numbers, except the two test files.

| Earlier finding | Status and evidence |
|---|---|
| 2 — training subtraction | **Resolved:** actual forward, backward, and optimizer boundaries replace subtraction, with measured optimizer counters (`train.rs:752`; driver:1060–1104). |
| 3 — queued work / decoder residual | **Resolved:** `open_window` drains before measurement, and decoder durations come from direct boundaries (`driver:366,734–781`). |
| 4 — clock metadata | **Resolved:** host timings use `SynchronizedHostWallClock`, while device spans report their returned timing method (`backend.rs:271`; driver:400,1372). |
| 5 — stability memory limit | **Resolved:** stability receives the limit, preflights both shapes before uploads, and supplies it to generation (`driver:1175,1220,1252`). |
| 7 — allocation acceptance | **Partly:** the patch explicitly keeps acceptance open and preserves allocation evidence, but the decoder loop still allocates (`driver:826–830,940`; architecture §6.5). |
| 8 — launch initialization | **Partly:** `L_init` and exact driver reconciliation are added, but the regression test does not independently establish stage attribution or the launch slope (`driver:819–825`; `tests/ms2_launch_budget.rs:175–193`). |
| 9 — stability reporting | **Partly:** phase-specific bucket counts and reserved-memory series are fixed, but production endpoints remain named `peak_reserved_bytes` (`driver:1270,1285,1351–1356,934,1147`). |
| 10 — duplicate preservation | **Resolved:** the test now checks action words, lengths, formula rows, and `FINISHED` bits (`tests/ms2_generation.rs:1781–1812`, frozen). |
| 11 — synchronization test | **Not resolved:** the new test explicitly synchronizes before observing output, so removing the span’s synchronization still passes (`tests/ms2_profile.rs:307–308`). |

Remaining problems, most severe first:

1. **Major — device generation profiling measures a different workload.**  
   **`examples/profile_ms2_substructure.rs:1433,1453,1574,1813`**  
   Production generation disables autograd at `generate.rs:406` and reuses warmed bucket buffers. The replica stages run with tracking enabled and allocate fresh peak, formula, trajectory, logits, and token buffers. Their warmup warms `workspace`, but the measured stages never use those buffers. Consequently, `profile_ms` measures fresh allocations and graph construction/retention alongside the kernels, rather than the reported warmed inference workload; `SystemTime` directly includes that additional host work.  
   **Fix:** profile shared production stage implementations using warmed buffers and a runner-local `no_grad` guard. Verify replica/production outputs and stage counters if any replica remains.

2. **Major — a caught callback panic destroys session state and can strand a profiling token.**  
   **`src/backend.rs:356–365,392–401`**  
   Both accessors remove the box before invoking user code and restore it only on normal return. A panic drops the state and leaves the slot empty. If `run` catches that panic, its next accessor fails. A type mismatch also consumes and drops the state. For `span`, the panic additionally skips CubeCL’s `end_profile` (`client.rs:944–949`). The claim at `backend.rs:451` that panic necessarily breaks the whole device channel is inaccurate: installed `channel.rs` catches task panics.  
   **Fix:** use an unwind-safe restoration guard, validate the type before consuming state, and catch callback panics inside the profiling closure so CubeCL can close its token before propagating failure. Add recovery tests.

3. **Medium — profiler handles have no session identity.**  
   **`src/backend.rs:328,355,392,477–484`**  
   Nested sessions normally save and restore either the same or a different state type correctly. However, calling the **outer** profiler while an inner session is active accesses whichever box currently occupies the single slot: with the same type it silently mutates/profiles the inner state; with a different type it destroys that state through the failing downcast.  
   **Fix:** associate each handle and slot entry with a session identifier, reject mismatched access without consuming state, and test outer-handle access during nested sessions of both types.

4. **Medium — generation does not receive the requested memory limit.**  
   **`examples/profile_ms2_substructure.rs:673–676,791–801`**  
   The driver preflights against `--max-device-bytes`, but constructs `GenerationConfig` without assigning that value; its default is 2 GiB. A configuration admitted by a larger CLI limit can therefore fail at the cold call’s `expect`. The extra `T+1` configuration also lacks its own CLI-limit preflight. Stability now behaves differently because it correctly supplies the limit.  
   **Fix:** propagate `max_device_bytes` into generation configuration and preflight the slope configuration before allocating its bucket, recording refusal consistently.

5. **Minor — launch-budget tests verify totals more strongly than attribution.**  
   **`tests/ms2_launch_budget.rs:175–193`**  
   The sum telescopes across boundaries, and `L_step` is computed from the same loop total subsequently reconstructed. Moving initialization launches into search still passes; arbitrary boundary movement outside the loop can also pass. The test ignores stage identities and does not exercise the driver’s independent `T+1 − T` slope. It also checks only work **after** `AfterReadout` at line 182, rather than the readout interval itself.  
   **Fix:** assert the complete stage sequence, compare independently measured `T`/`T+1` slope against every step delta, require meaningful positive counts, and check `AfterReadout − AfterValidate` separately. The current test does provide a useful independent plain-versus-hooked total check, so it is not wholly vacuous.

6. **Minor — synchronization regression test remains ineffective.**  
   **`tests/ms2_profile.rs:278–308`**  
   Removing `device.synchronize()` from `sync_span` still passes: the test calls `try_synchronize()` afterward, and the subsequent tensor read also waits. The comment claiming asynchronous values would otherwise be undefined is incorrect for a synchronizing read.  
   **Fix:** observe boundary synchronization directly through an instrumented synchronization seam, or assert that a deliberately deferred failure surfaces at the span boundary before any later sync/read.

7. **Minor — endpoint reserved memory is still reported as a peak.**  
   **`examples/profile_ms2_substructure.rs:724,934,1050,1147`**  
   A transient high reservation during a call can exceed the sampled endpoint, while JSON presents that endpoint as `peak_reserved_bytes`.  
   **Fix:** rename it to an endpoint measurement or collect an actual high-water mark.

The normal harness path is memory-safe: non-Send state is constructed, accessed, and dropped on the runner; `span` results and session results require `Send`. Returning an ordinary `Err` value restores the slot, and runtime start/end-profile errors do not themselves leave it empty. The problems above concern recovery and session ownership, not an unsafe transfer or aliasing of `Rc`.

Stream identity is preserved for the driver’s ordinary clients: `client.rs:906` captures the stream, `channel.rs:104` propagates it through `StreamId::executes`, and same-runner dispatch executes inline. WGPU’s `ticks.duration()` **does** derive from GPU timestamps: `compute/timings.rs:260–273` converts query ticks using the queue timestamp period into synthetic instants whose difference is the device duration.

Hook-free generation preserves the original device operations and order; its added branches perform no launch, read, or allocation. Training likewise preserves the forward/report-pack → backward/clip → optimizer sequence, although splitting state changes some object lifetimes. The host generation boundaries correctly charge validation separately and attribute the single `read_all` to readout (`driver:776–781`).

The earlier unsupported claims about safe timestamps, host clock identity, training forward subtraction, the missing initialization term, and duplicate preservation are corrected. The fully independent formula-reference claim is **unaddressed by this snapshot**: it contains no corresponding test changes, and I did not inspect the concurrently edited formula work.

**Verdict: reject.**