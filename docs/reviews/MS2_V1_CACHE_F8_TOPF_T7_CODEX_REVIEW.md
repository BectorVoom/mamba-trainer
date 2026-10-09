# Codex review: cache identity fixes (task F8, format v4) and chunked top-F selection (task T7)

Reviewer: codex exec (read-only, on a snapshot of the tree), 2026-10-06. Verdict, cache: the three evidence-identity fixes hold on the production paths; the host memory budget is rejected as incomplete (a refused enumeration insert aborts Evidence cache construction; admission can exceed the budget; variants still materialised together; the budget is ignored on load; replacing an enumeration entry leaves its old evidence reusable). Verdict, top-F: the selection algorithm and kernels pass source review; scratch buffers are missing from the memory preflight and the switch test does not prove routing. Fix task: F9.

**Part A: reject F8 as complete.** The three evidence identity fixes work on the production lookup path, but budget enforcement and refusal handling remain defective.

**Part B: reject the integration as complete.** The selection algorithm passes source review; scratch-memory accounting and routing verification need fixes.

Read-only review: no files changed, no Cargo or tests run. Scratch-arena implementation and functional-group files were excluded. Test conclusions below concern source-level sensitivity, not observed execution.

### Part A — findings

**A1 — P1: an enumeration budget refusal aborts Evidence cache construction.**  
Locations: [generate.rs:1003](src/models/ms2/generate.rs:1003), [train.rs:1452](src/models/ms2/train.rs:1452).

- **Scenario:** use `FormulaFeatures::Evidence`, a nonempty batch, and a zero budget—or a budget that refuses one enumeration row. Enumeration construction correctly returns success after refusing insertion. The evidence phase then calls `expand_batch_core`, gets `None`, and returns “enumeration entry missing.” The example exits through `fail`.
- **Expected:** construction succeeds with those rows uncached; their subsequent step equals uncached execution.
- **Actual:** no subsequent step occurs through the normal driver.
- **Minimal fix:** skip evidence precomputation for a batch whose enumeration cannot be expanded, or retain the enumeration computation’s buffers independently of successful insertion. Add zero/partial-budget tests using the Evidence layout; the current budget test uses Counts.

An evidence-only insertion refusal does fall back correctly later. An attached incomplete cache also takes the full uncached stage on a batch miss.

**A2 — P1: insertion can exceed even the reported resident budget.**  
Locations: [enum_cache.rs:1254](src/models/ms2/enum_cache.rs:1254), [enum_cache.rs:1402](src/models/ms2/enum_cache.rs:1402), [enum_cache.rs:1440](src/models/ms2/enum_cache.rs:1440).

- **Scenario:** give an empty cache a 124-byte budget and insert a zero-scored enumeration entry. The admission calculation is exactly `32 + 20 + 72 = 124`, so it accepts. Insertion grows the map; `resident_bytes()` then adds a positive map-capacity charge and exceeds 124.
- **Additional scenario:** pass valid evidence vectors with short lengths but large reserved capacities. Admission uses lengths; committed accounting uses capacities. An accepted insert can exceed the budget by an arbitrarily large reserved payload.
- **Expected:** every accepted insertion remains within the enforced estimate.
- **Actual:** admission omits upcoming map growth and uses smaller evidence length terms.
- **Minimal fix:** calculate the prospective capacity-based footprint, including map and bucket growth, before committing. Subtract the replaced entry when checking replacements.

The accounting also omits unused `Vec<EvidenceEntry>` bucket capacity. A singleton bucket reserves space for multiple entry structs; growth doubles that allocation without any corresponding capacity term. `16 * map.capacity()` is an estimate, not the actual allocation of these much larger key/value slots. Header allocations and rehash overlap are also outside the formula. Canonical and payload vector capacities themselves are included in committed accounting.

`budget_refusals` increments on the explicit rejection branches, but an insert incorrectly accepted beyond budget is not counted.

**A3 — P2: jitter variants are still materialized together.**  
Locations: [experiment.rs:625](src/models/ms2/experiment.rs:625), [ms2_experiment.rs:681](examples/ms2_experiment.rs:681).

- **Scenario:** a large batch with many fixed variants and a small cache budget. `jitter_variants_of_batch` clones all `V` batches before the driver begins consuming the returned vector.
- **Expected:** one variant’s temporary memory at a time.
- **Actual:** temporary memory grows with `V`, independently of the cache budget. The comment claiming variants are never retained together is incorrect.
- **Minimal fix:** return a lazy iterator, or construct/build/drop each variant inside the driver’s loop.

The driver no longer retains variants for the entire dataset, which is an improvement. `save` still allocates a complete serialized buffer plus sorting vectors; a resident-cache budget does not bound that transient peak.

**A4 — P2: the driver’s budget is ignored when loading an existing cache.**  
Locations: [ms2_experiment.rs:637](examples/ms2_experiment.rs:637), [enum_cache.rs:1799](src/models/ms2/enum_cache.rs:1799), [enum_cache.rs:2163](src/models/ms2/enum_cache.rs:2163).

- **Scenario:** an existing large cache, invoked with `--enum-cache-max-mb 1`.
- **Expected:** the requested host limit governs cache acceptance.
- **Actual:** the entire file and parsed cache are allocated; loading reinstates the 4096-MiB default, and the driver attaches it without checking the requested limit.
- **Minimal fix:** pass the budget into loading and enforce it during parsing. Bound the initial file allocation too if this flag promises a host-memory limit. Setting the limit only after loading does not protect the load-time peak.

**A5 — P2: replacing an enumeration entry leaves its old evidence reusable.**  
Location: [enum_cache.rs:1272](src/models/ms2/enum_cache.rs:1272).

- **Scenario:** build enumeration and evidence for key `K`; then use the public, documented overwrite API to replace `K` with different valid candidates having the same scored count. Keep the spectrum unchanged.
- **Expected:** evidence is recomputed for the replacement candidates.
- **Actual:** evidence identity contains `K` and the spectrum inputs, but not the candidate contents. Old evidence passes comparison and is paired with the replacement candidates.
- **Minimal fix:** invalidate evidence buckets with `key.meta == K` whenever an enumeration entry is successfully replaced.

The current production builders skip existing enumeration keys, so they do not themselves trigger this scenario.

### Part A — disposition and exactness

| Previous finding | Disposition |
|---|---|
| `n_peaks` omitted | **Resolved.** Header/build/use checks enforce it; attachment rejects mismatches. |
| Dtype omitted | **Resolved on production serving paths.** Enumeration remains reusable; evidence requires the exact dtype name. |
| Hash hit unverified | **Resolved for spectrum inputs.** Full canonical equality and collision buckets replace probabilistic acceptance. Enumeration overwrite caveat: A5. |
| No host budget | **Partially resolved.** A budget and reporting exist, but A1–A4 prevent the promised behavior. |

**Canonical inputs:** `evidence_canonical_bytes` contains nine scalar words and uploaded valid-prefix peak pairs: count, intensity scale, precursor, adduct, resolved fragment tolerance, fragment uncertainty, evidence work limit, evidence peak limit and hydrogen bound; then m/z and intensity bits.

It does **not literally contain every dependency**: enumeration meta is in `EvidenceKey`; `n_peaks`, dtype, artifact identity and enumeration configuration are checked separately. Together these cover the ordinary deterministic production enumeration/evidence path. Candidate contents themselves are not compared—A5.

The uncached path is upload → `peak_select` → `evidence_peaks` → `formula_evidence`. Fatal rows contribute uploaded count zero. Padding beyond that count is irrelevant. Kept/evidence valid counts follow deterministically from the encoded inputs and checked capacity/dtype. Precursor uncertainty and precursor tolerance affect enumeration/features, not the memoized evidence computation independently of the selected candidates.

Intensity comparison uses `f32::to_bits()`, with **no rounding or NaN normalization**. Different NaN payloads and `−0.0`/`+0.0` remain distinct. Under narrower dtypes this is conservative: different host words may upload identically, but cannot falsely alias through canonical equality.

**Dtype/misses:** generation, training and diagnostics pass `E::DTYPE.name()`. I found no production route serving f32 evidence under bf16. Batch expansion builds a private vector and returns `None` on any miss; callers upload only `Some`. Partial results cannot escape.

**Save hook:** [enum_cache.rs:998](src/models/ms2/enum_cache.rs:998) takes one global mutex per phase—three acquisitions per successful save—even with no hook. The callback runs after releasing the mutex, so callback replacement does not deadlock that mutex. However, the public setter leaves a hook installed until explicitly cleared; recursive saves can recurse indefinitely, and an installed rendezvous hook can block an unrelated save. The supplied test resets it through a drop guard and serializes its other tests. A test-support feature gate would remove this production exposure and cost.

**Hit cost:** for `K = min(uploaded_peak_count, n_raw)`, the content hash processes **`36 + 8K` bytes** through two FNV passes. At 512 peaks that is 4132 bytes. It also computes an m/z sum; verification recomputes that sum and compares the canonical bytes. `EvidenceKey` hashes its eight meta words plus two hash words. Full row comparison occurs **only after a map-key match**, potentially once per collision-bucket member. Header checks borrow strings; canonical lookup allocates no per-spectrum byte buffer. Batch query/output allocations remain.

### Part A — previously weak test rows

| Previous “No”, “Probabilistic”, or qualified row | Current disposition |
|---|---|
| Bounded loading, `ms2_enum_cache.rs:2117` | Original test still only checks rejection. **Companion allocation test fixes the guard** at `ms2_enum_cache_alloc.rs:294`: reverting reservation bounds should exceed its allocation limit. |
| Concurrent saves, `:2315` | **Deterministic overlap now enforced** at `Written`, with overlap asserted. An early worker failure can leave its peer blocked at the barrier, so regressions may hang rather than terminate cleanly. |
| Concurrent lookups, `:2390` | **Fixed:** production counters and returned buffers are asserted. |
| Dispatch invariance, `:2613` | **Fixed for distinct chunk layouts:** launch counts and raw buffers differ/agree as appropriate. Still lacks an asserted **productive, incomplete** enumeration fixture; its far-precursor row has zero scored candidates. |
| Flag-2 round trip, `:2699` | **Fixed:** save/load and flag-zero rejection added. |
| Hit–miss–hit, `:2756` | **Fixed:** batch shape stays constant. |
| Exceptional cached readout, `:2858` | **Fixed:** actual hit and exhaustion/incomplete status asserted. |
| Reorder/subset jitter, `:2977` | **Improved:** exact first-step variant asserted. Later steps still check only pool membership. |
| V=0, `:3063` | **Improved:** first prepared draw/tag asserted. Expected values reuse the production draw helper, so this is an orchestration guard, not an independent RNG implementation oracle. |
| Evaluation tag, `:3130` | **Fixed for shared evaluation preparation:** calls `jittered_set_for_eval` and evaluates its output. |
| Driver pool, `:3196` | **Fixed for shared pool construction.** Still would not catch the example omitting/truncating consumption of that helper’s output. |

The previously “Yes” guards retain their stated coverage. The new dtype test checks string-policy misses and bf16 build refusal; it does not execute an attached bf16 model’s full cached-versus-uncached evidence step.

### Part B — findings

**B1 — P2: chunk scratch buffers are missing from memory preflight.**  
Locations: [ms2.rs:4349](src/tensor/ops/ms2.rs:4349), [workspace.rs:648](src/models/ms2/workspace.rs:648).

- **Scenario:** plane device, Enumerate source, `B=16`, `M=2048`, `F=4`, f32. Selection allocates score and slot buffers of `[16,32]`: **4096 additional payload bytes**, before allocator rounding.
- **Expected:** the memory estimate includes the enabled selection workspace.
- **Actual:** neither buffer appears in the estimate. A preflight limit set at the reported total accepts work with these additional allocations.
- **Minimal fix:** account for `B * ceil(M/64) * (sizeof(E)+4)` when routing chunked, or conservatively whenever chunked is possible. Retaining these buffers in the formula workspace would also avoid per-call allocation requests.

Normal generation search runs outside the decode scratch scope. These are temporary device allocation requests per call; backend pooling may recycle their storage.

**B2 — P2: the switch test does not prove routing and its overrides interfere.**  
Locations: [ms2_formula.rs:3451](tests/ms2_formula.rs:3451), [ms2_formula.rs:3496](tests/ms2_formula.rs:3496), [ms2_formula.rs:3501](tests/ms2_formula.rs:3501).

- **Scenario:** change routing to always use the old kernel. The switch test’s old, routed and direct-chunked outputs still agree.
- **Expected:** forced-on routing must be verified.
- **Actual:** all assertions can pass without routing to chunked.
- **Minimal fix:** use an isolated counter-owning test binary and assert one old launch versus `2F` chunked launches.

The kernel-comparison test also sets a process-global override without a drop guard. Panic leaves it installed; concurrent switch tests can overwrite or clear each other’s setting. The existing guard clears rather than restores a previous setting. Isolation or properly scoped, serialized overrides are needed for reliable routing tests.

### Part B — algorithm, kernels and coverage

**Equivalence:** strict succession is equivalent to repeatedly choosing the best remaining eligible slot. Numeric equality handles signed zeros as ties; slot order chooses the same original score bits. Contiguous chunks and increasing combine order preserve ties across chunks.

NaN, infinities and exact `±3e38` fail eligibility. Empty support stays padded; once a previous pick is padding, all subsequent picks remain empty. Ragged chunks mask out-of-range slots after safe loads. `F=1`, support smaller than `F`, and ragged dispatch lanes are consistent by inspection.

Neither interface receives `rows_scored`. Both depend on the production invariant that flags beyond scored support are zero; arbitrary nonzero flags beyond an external `rows_scored` value would be selected by both.

**Kernel rules:** each new kernel has six array bindings. One chunk lane owns each scratch entry; one combine lane owns each spectrum’s current pick and count. Combine pass `ff` initializes its pick’s padding, then optionally replaces it; every pass rewrites `top_count`, with the last pass leaving the final count. I found no prohibited scalar-argument initialization of loop-carried state or branch-contained global load in the new scans.

The chunk launch reads prior outputs and writes separate scratch buffers. The combine launch reads scratch and writes the current pick; its same-lane output reloads introduce no cross-lane race. Prior-pick reads and current-pick writes are separated by launches.

**Switch cost/CPU:** routing uses sequentially consistent atomic loads and may read the environment twice per eligible call. Those reads form no single coherent configuration snapshot, although the atomics themselves have no data race. Plane detection is `plane_size_max > 1`; local CubeCL CPU 0.10 reports `1`, so the **default** retains the old CPU kernel. Explicit-on overrides permit CPU chunking.

**Launch accounting:** chunked replaces one launch with `2F`, so search and total counts increase by **`2F−1`**: seven for `F=4`, fifteen for `F=8`. Existing numeric pins use `M=32` and should not change. Existing enumeration footprint tests also use `M=32`; they do not guard the newly enabled plane-device route. Add an Enumerate, plane-device, `M≥256` search-stage pin.

| New test | Device chunked path? | Detects broken previous-slot tie-break? |
|---|---|---|
| Random windows/twin/oracle, `:3242` | **No**, host-only | No device-kernel guard; catches the corresponding twin mutation. |
| Kernel versus old, `:3346` | **Yes**, direct calls on its selected backend | **Yes:** all-equal flagged spectrum forces successive slots. |
| Switch, `:3501` | Direct chunked call: **yes**; routed call unproved | Quantized ties provide sensitivity, including through the direct comparison; route selection itself remains unguarded. |

Device fixtures cover `F=4/8`, empty/small support, invalid scores and scored-prefix padding. They lack an explicit `F=1`, signed-zero pair, and non-chunk-multiple `M` fixture. The sort oracle correctly uses numeric comparison followed by slot order. The supplied 10.7→2.5-ms measurement was not rerun.