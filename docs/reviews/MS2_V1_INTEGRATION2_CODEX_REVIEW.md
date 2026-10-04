# Codex review: top-F selection, bounded enumeration dispatch, allocation/identity/packed integration, dtype guard

Reviewer: codex exec (read-only, frozen source snapshot of 2026-10-04 14:03, before task I3b). Verdicts: A accept-with-fixes, B accept-with-fixes, C reject, D reject.

Static review only. No files modified and no cargo commands run. Runtime conclusions below come from source inspection and arithmetic reconstruction, not executed Rust tests.

**Part A — linear top-F selection**

1. **Minor — the stated complexity omits the taken-slot scan.** [ms2.rs:3588](src/tensor/ops/ms2.rs:3588), [twin.rs:490](src/models/ms2/twin.rs:490).  
   Each of F picks scans M candidates and checks previous picks. The kernel checks all F output slots per candidate; the twins check the preceding picks. Consequently, complexity is **O(F²M)**, although it is linear in M under the contractual `F <= 8`. For `M=2048,F=8`, the kernel performs 131,072 prior-slot loads, beyond the 16,384 candidate visits. This does not recreate the old quadratic-in-M defect.  
   **Fix:** state the actual complexity and bounded-F qualification, or implement constant-time taken membership.

The selection itself is correct for the requested f32 domain: descending score, ascending-slot ties, no repeated selections, dense prefix, and MAX/MAX/zero padding. NaN and infinities fail the predicate. Every top, log-probability and count element is written, including empty support. Loads are outside conditional branches; selection’s loop-carried counters start from literals.

The twins are **arithmetically equivalent for f32, not literal copies**: the kernel reloads its current best from outputs and checks the lower bound using `0 - score`; the twins carry the best in host variables and compare against `-FINITE_MAX`. Neither difference changes the requested ordering.

The poisoned tests and independent sort oracle at `tests/ms2_formula.rs:2961,2994` provide meaningful correctness coverage. Downstream consumers use the retained count and original window slot; `gold_slot` searches scored composition support independently. I found no downstream dependency requiring non-finite entries to retain their old ranks. Bit-identical V0 compatibility necessarily excludes the deliberately changed non-finite/out-of-domain behavior.

**Part B — bounded enumeration dispatch**

1. **Minor — comments still assert a disproven reset cause.** [ms2_enum.rs:148](src/tensor/ops/ms2_enum.rs:148).  
   The concrete counterexample is the measured production run described in `docs/MS2_SUBSTRUCTURE_TASKS.md:338`: bounded enumeration submissions did not prevent the reset; replacing top-F did.

   Every remaining sentence asserting or invoking the wrong causal explanation is:

   - `src/tensor/ops/ms2_enum.rs:148–151`: chunk flushing is described as measured prevention of a timeout caused by batching enumeration.
   - `src/tensor/ops/ms2_enum.rs:178–180`: fill flushing is justified by the same supposedly measured timeout.
   - `src/tensor/ops/ms2_enum.rs:2086–2091`: batching is said to trip the driver timeout, while one job per chunk does not.
   - `src/tensor/ops/ms2_enum.rs:2189–2192`: “Without this” flushing, the runtime is said to create a command buffer that times out at production scale.
   - `src/tensor/ops/ms2_enum.rs:2315–2318`: fill flushing invokes the erroneous timeout explanation on `enum_count`.
   - `tests/ms2_enum_kernels.rs:2052–2055`: the test comment repeats the supposedly measured batching/timeout claim.

   `ms2_enum.rs:2430–2431` additionally points readers back to that explanation. The architecture and task-document reset narratives have already been corrected.  
   **Fix:** describe submission as a precautionary work bound; remove claims that enumeration batching was the measured cause or that this change cured it.

2. **Minor — chunk-equivalence tests barely exercise fill.** [ms2_enum_kernels.rs:2010](tests/ms2_enum_kernels.rs:2010), [ms2_enum_kernels.rs:2068](tests/ms2_enum_kernels.rs:2068).  
   Both chunk-specific tests use `B=1`, identical rare rows and an empty search. Fill writes no candidates. A regression using the local lane index in **fill alone** can therefore pass these tests. The second test exercises a non-dividing chunk size, but still only with empty fill.  
   **Fix:** use distinct productive lanes across at least two spectra, compare poisoned candidate outputs, and include chunk sizes 1, 2 and a size crossing a spectrum boundary. Assert actual launch deltas in a serialized counter test.

The implementation’s absolute lane index is correct in count and fill. Contiguous chunks cover every lane once, including tails. With consistent production metadata, chunking preserves lane statistics, offsets and candidate order. Count and fill each launch  
`ceil(B*P / max(1, dispatch_visits_max/lane_visits_max))` times; offsets and padding add two launches. Flushing adds no crate-level launch or read.

Both callers propagate their configured budgets, and configuration validation rejects zero fields. Memory estimates include enumeration metadata and need no chunk-sized additional buffer. Existing stage pins correctly cover **Table** search at 33 launches; they do not establish enumeration’s chunk-dependent launch count.

**Part C — allocation, identity and packed integration**

1. **Major — evidence is silently omitted, and switching evidence off can expose stale evidence.** [generate.rs:2515](src/models/ms2/generate.rs:2515), [pack.rs:1498](src/models/ms2/pack.rs:1498).  
   For a supported `assignment` configuration with `evidence=true`, unpacked generation runs assignment/evidence, while packed and resident generation skip those stages and zero evidence. Host `pack` copies only evidence status; `assemble` initializes every evidence-detail field to zero. Packing an eligible candidate with nonzero evidence status can therefore fail validation rather than preserve its evidence.

   There is also a reuse failure: call `generate(evidence=true)` with actual evidence, then `generate_packed(evidence=false)` on the same bucket. The clearing branch does not run, leaving nonzero evidence status with zero packed details. All integration fixtures use `evidence=false`.  
   **Fix:** implement the shared evidence stages and complete evidence gathering in both pack paths. Ensure OFF ignores or resets previous evidence. Add ON equality and ON→OFF reuse tests.

2. **Major — packed validation still rejects legal enumeration counters.** [pack.rs:844](src/models/ms2/pack.rs:844).  
   The earlier S₂ scenario remains failing: one visited heavy vector can join both H=0 and H=2, giving `visited=1, joined=2, scored=2`. `CandidateBatch::validate` now accepts the source-specific bound; `PackedCandidateBatch::validate` still requires `joined <= visited`. Thus `generate` can succeed while `generate_packed`, resident `read`, and host `pack` fail.  
   **Fix:** share source-specific counter validation, including saturation/exhaustion rules. Test the same productive enumeration request through all three modes.

3. **Major — bf16 proportional allocation violates the specified f32 arithmetic.** [ms2_identity.rs:1113](src/tensor/ops/ms2_identity.rs:1113), [ms2_identity.rs:1480](src/tensor/ops/ms2_identity.rs:1480).  
   The wrapper now accepts bf16, but the kernel’s maximum, exponentials, sum, quotas and fractions remain type `F`. The host twin and §3.2 specify f32.

   A concrete arithmetic counterexample is `F=3,K=15`, retained log-probabilities  
   `[-0.6796875, -1.375, -3.21875]`. F32 produces assignments `[9,5,1]`; bf16 arithmetic produces `[8,5,2]`. These are representable bf16 inputs, and the f32 quotas are away from the stated integer/tie tolerance.  
   **Fix:** widen loaded probabilities and perform allocation arithmetic in f32. Add bf16-input tests against the f32 twin. Existing allocation-kernel comparisons upload f32 only.

4. **Major — bf16 packed scoring breaks host/device equality and can reverse ranking.** [ms2_pack.rs:178](src/tensor/ops/ms2_pack.rs:178), [ms2_pack.rs:778](src/tensor/ops/ms2_pack.rs:778).  
   Device ranking and `record_pack_f` add the terms in `F`; host packing adds downloaded terms in f32.

   For representable bf16 terms `formula=-0.69140625, trace=-1.0078125`, host score is `-1.69921875`, whereas bf16 packed score is `-1.703125`. An earlier trajectory scoring exactly `-1.703125` becomes tied on device and wins by trajectory index, although the host ranks the later candidate higher. Integration equality tests use f32 only.  
   **Fix:** rank and store the raw sum in f32 across neural dtypes, and test bf16 equality for every field and near-tie ordering.

5. **Major — finite extreme scores remain incorrectly excluded.** [pack.rs:153](src/models/ms2/pack.rs:153), [ms2_pack.rs:183](src/tensor/ops/ms2_pack.rs:183).  
   A finished, valid candidate with finite reranker score `3.1e38` remains excluded. The top-F domain restriction does not establish this additional ranking restriction in §4.4. New tests explicitly assert exclusion, so they encode the previous defect rather than resolve it.  
   **Fix:** use exact finiteness classification for ranking terms and score, including finite f32 extremes, NaN, infinities and overflowing sums.

6. **Major — packed validation still accepts a finished graph without any conditioning formula.** [pack.rs:635](src/models/ms2/pack.rs:635), [pack.rs:785](src/models/ms2/pack.rs:785).  
   Construct one filled slot with legal `START, ADD_ATOM, STOP`, matching open valence, `FINISHED`, zero formula counts, MAX row/rank, zero search counters and nonfatal request status. Provenance checks permit the absent rank, and replay receives `None`, so validation accepts it. The generation contract requires no trajectory to start without a formula.  
   **Fix:** require real formula provenance for every filled packed candidate. Add this corruption independently of counts/rank-mismatch tests.

7. **Minor — the public workspace readout silently drops graph identity and can bypass enumeration reconciliation.** [generate.rs:2154](src/models/ms2/generate.rs:2154).  
   Run workspace stages with `identity=Graph`, then call `generate_readout_ws`: it passes `use_graph=false`, losing the computed bits and resolutions. With Enumerate and unknown precision, it also passes no host batch, leaving device `complete=1` instead of reconciled `complete=0`. Its comment describes a restricted profiler path, but its API does not enforce that restriction.  
   **Fix:** reject unsupported modes explicitly or route through the complete batch-aware readout.

8. **Minor — host packing checks some unsupported address domains after substantial staging allocation.** [pack.rs:1789](src/models/ms2/pack.rs:1789), [pack.rs:1322](src/models/ms2/pack.rs:1322).  
   With `T=22,A=16`, input record stride is 108 words and packed width is 123. Choose a shape where `rows*108 <= u32::MAX` but `rows*123 > u32::MAX`. Host `pack` allocates and fills its staging buffers before the later helper rejects the output domain; it can exhaust memory before returning the promised shape error.  
   **Fix:** preflight every staging/output product before constructing any staging buffer.

For ordinary warmed f32 calls with carry capture disabled, the read paths provide one read for `generate`/`generate_packed` and zero for `generate_resident`. Resident ownership prevents subsequent workspace calls from overwriting its leased buffers. Cold autotuning and explicitly enabled carry capture are outside that read-count guarantee.

Graph hashing is synchronous and permutation invariant; hash equality leads to bounded exact comparison rather than automatic deduplication. The search checks injectivity, labels and both present/absent bonds. Budget exhaustion sets unresolved rather than duplicate. F32 allocation writes exactly K records, handles ties deterministically, and preserves the specified baseline assignment even when a retained formula’s exponent underflows to zero.

Prior review **Part D** findings:

| Finding | Status | Current evidence |
|---|---|---|
| D1 Counter inequalities | **Partly** | `contract.rs:1655` fixed; `pack.rs:844` remains wrong. |
| D2 Enumeration duplicate identity | **Resolved** | `ms2.rs:7089`, `twin.rs:1239`: conditioning counts distinguish formulas. |
| D3 Late training lane refusal | **Resolved** | `train.rs:1051`: check precedes upload/bucket/encoder; footprint regression at `tests/ms2_footprint.rs:516`. |
| D4 Incomplete search reported absent | **Partly** | Production reconciliation at `generate.rs:1357` fixed; unrestricted legacy stage readout at `:2154` bypasses it. |
| D5 Shuffled-spectrum extra probe | **Resolved** | `ms2_experiment.rs:547`, `train.rs:2004`: use donor-path evaluation statuses. |
| D6 Train-only fitting | **Resolved** | `ms2_experiment.rs:270,356`, `experiment.rs:776`: subset and molecule-overlap checks. |
| D7 Missing enumeration metadata estimate | **Resolved** | `workspace.rs:925,1896`: generation and training include `32B`. |
| D8 Missing checkpoint artifacts | **Resolved** | `train.rs:2167`: required artifact JSONs checked at load. |
| D9 False zero gold-not-scored | **Resolved** | `ms2_experiment.rs:491,705`: evaluation gold slots and explicit denominator/null. |

Prior review **Part E** findings:

| Finding | Status | Current evidence |
|---|---|---|
| E1 Scores/provenance validation | **Partly** | Independent scores, rank bounds, uniqueness and composition replay added at `pack.rs:583,635,705,785`; finished/no-formula case remains accepted. |
| E2 u32 addressing | **Partly** | Device wrappers and lane domains are checked, e.g. `ms2_pack.rs:1052`; host output-domain refusal still follows staging allocations. |
| E3 Finite extreme score exclusion | **Not resolved** | `pack.rs:153`, `ms2_pack.rs:183`; tests now pin the exclusion. |
| E4 Device accepts R>K | **Resolved** | `ms2_pack.rs:1017`; refusal test at `tests/ms2_pack_kernels.rs:894`. |
| E5 Malformed formula-rank length | **Resolved** | `pack.rs:1692`: consumed rank length checked before indexing. |

**Part D — dtype guard and matrix**

1. **Major — the gate checks configuration dtype instead of the actual generic element type.** [generate.rs:489](src/models/ms2/generate.rs:489), [workspace.rs:183](src/models/ms2/workspace.rs:183).  
   On CPU, call `Ms2Model::<R, half::f16>::init` with a valid config whose `dtype=F32`. The gate approves f32; parameter initialization uses `E=f16`. It therefore performs unsupported MS2 dtype work rather than refusing before allocation/upload. Conversely, actual f32 with config bf16 makes memory estimates price two-byte elements while allocating four-byte elements.

   The tests consistently set `config.dtype=E::DTYPE`, so they miss the mismatch.  
   **Fix:** validate `E::DTYPE` and require configuration/type equality at constructors and preflight entry points. Test mismatches with allocation/upload/launch counters unchanged.

2. **Major — bf16 is admitted on any capable backend, rather than CPU only.** [workspace.rs:161](src/models/ms2/workspace.rs:161), [ms2_dtype_report.rs:507](examples/ms2_dtype_report.rs:507).  
   `check_dtype("cuda", BF16, true, true)` returns success. CUDA is an enabled repository backend, and pinned CubeCL exposes bf16 support. This admits an unvalidated combination; the report instead expects every non-CPU bf16 configuration to fail and can panic at `expect_err`.  
   **Fix:** enforce the validated backend/dtype allowlist independently of hardware support, using the same policy in production, tests and reporting.

3. **Minor — the committed report predates the refusal policy and is not reproducible by the current generator.** [dtype_matrix_cpu.json:25](bench/results/ms2/dtype_matrix_cpu.json:25), [ms2_dtype_report.rs:539](examples/ms2_dtype_report.rs:539).  
   The JSON reports f16 as `supported=true`, with measured step-zero NaN and attempted generation. The current generator refuses f16 and emits no measurements for it. These may be legitimate historical measurements, but the artifact provides no revision or historical-policy distinction.  
   **Fix:** identify the historical run and distinguish hardware support from MS2 validation, or regenerate under the current policy. I found no evidence that an unmeasured GPU combination was fabricated as measured.

**Verdicts:** Part A **accept-with-fixes**; Part B **accept-with-fixes**; Part C **reject**; Part D **reject**.