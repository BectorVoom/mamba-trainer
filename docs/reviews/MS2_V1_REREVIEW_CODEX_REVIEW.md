# Codex re-review of fix rounds: graph identity, ion assignment, enumeration, profile harness

Reviewer: codex exec (read-only, frozen source snapshot of 2026-10-04 14:03). Verdicts: A accept-with-fixes, B accept-with-fixes, C reject, D accept-with-fixes.

The core fixes are substantially better, but **Part C still has a production correctness failure**. Parts A, B and D retain narrower API, kernel-rule or verification gaps.

This was a static, read-only review. No files were changed and no cargo commands were run. The snapshot’s ion review file contains only one review; I also assessed the ion follow-ups recorded in the identity review. Allocation-only findings are outside the requested scope. The `*_scored` / `*_features` ion kernels were excluded.

**PART A — graph identity**

| Earlier finding | Disposition | Current evidence |
|---|---|---|
| First review #1: wrong refinement bank/round | **resolved** | `src/models/ms2/identity.rs:308–354`: typed initialization, four synchronous alternating rounds, final bank 0. Independent labels/hashes at `tests/ms2_identity.rs:558`. |
| First review #2, identity portion: nonportable addressing/type storage | **partly** | Full-buffer addressing and local type decoding are fixed, but the new extent checks perform `usize` addition inside both twin and kernel. See new defect A1 below. |
| First review #3: inferred incompatible scratch layout | **resolved** | Explicit capacities and scratch/stack strides are checked at `identity.rs:433–444`; regression at `tests/ms2_identity.rs:1086`. |
| First review #5: adjacency regression exits on unequal counts | **resolved** | Equal-count topology and bond-order cases at `tests/ms2_identity.rs:974` reach exact adjacency comparison. |
| Second review: record-local identity slices | **resolved** | Full buffers and explicit record/base addressing at `identity.rs:223`, `:416`, `:602`. |
| Latest review #1: ordinary debug hash overflow | **resolved** | Explicit wrapping at `identity.rs:335`, `:337`, `:354`; independent regression at `tests/ms2_identity.rs:609`. This wrapping is required by §4.2. |
| Latest review #3, identity portion: unchecked device address domain | **resolved** | Binding lengths and narrowed scalars are checked at `src/tensor/ops/ms2_identity.rs:425–430`, `:850–859`. |
| Latest review #7: incomplete public-helper layouts can yield equality | **partly** | Original empty/truncated-scratch counterexamples are fixed by `identity.rs:455–474`, tested at `tests/ms2_identity.rs:1200`. The action stride itself remains unvalidated. |
| Latest review #9, identity statuses | **resolved** | Resolution values at `docs/MS2_CONTRACTS.md:235`; bits 7/8 at `:574–576`. |

**Remaining scenario for latest #7:** call `graph_equal_lane` with `A=32`, bonds=39, steps=42, records 1 and 0, `record_stride=0`, empty actions, hashes `[0,0]`, 362 scratch words, scratch stride 181, and a complete 96-word stack. The extent checks accept the empty action ranges, missing lengths decode as zero, and `identity.rs:496–497` returns **equal**, rather than unresolved. Also, record-base multiplication at `:446–449` precedes validation; an oversized record index can panic in debug builds before the promised unresolved result. These are public-helper failures; correctly shaped launchers prevent them.

**New defect A1 — Minor: the extent fix violates the u32-only arithmetic rule.**  
[identity.rs:455](src/models/ms2/identity.rs:455) and [ms2_identity.rs:537](src/tensor/ops/ms2_identity.rs:537) add values after casting them to `usize`; the scratch and stack checks repeat this. An ordinary comparison of two valid identical graphs executes these additions. This is arithmetic beyond indexing/bounds conversion. Use guarded u32 extent comparisons after validating the buffer lengths.

The hash and exact DFS otherwise retain matching decisions: complete injective mappings, types, labels, bond orders and non-bonds; deterministic budget consumption; and independent duplicate/unresolved accumulation. Bindings are hash **3**, identity **5**. I found no prohibited loop-carried scalar-argument initialization.

---

**PART B — ion assignment, label mask and unscored evidence**

| Earlier finding | Disposition | Current evidence |
|---|---|---|
| Initial #1: implementation not u32-only | **partly** | Radix, mass and verdict arithmetic are u32 in `ion.rs:334`, `:365`, `:427`. Full-buffer twins still calculate addresses and loop counters in `usize`, e.g. `:564–576`, whereas kernels calculate addresses in u32. |
| Initial #2: missing report denominator | **resolved** | Inspected/conditional denominator reporting at `examples/ms2_ion_report.rs:362–377`. |
| Initial #3: label-upload losses disappear | **resolved** | Uncapped and uploaded-label metrics and explicit lost-label counts at `examples/ms2_ion_report.rs:378–397`. |
| Initial #4 and follow-up: helper categories can escape the superset test | **resolved** | Independent expected mappings now check `embedding_ion` itself, including unexpected `None`, at `tests/ms2_ion.rs:643`, `:676`; category coverage remains asserted. |
| Initial #5: absent-anchor assertion vacuous | **resolved** | Deliberately absent anchor and complete baseline comparison at `tests/ms2_ion.rs:938`; negative-adduct/shift cases follow. |
| Follow-up: inactive hydrogen slots form overflowing masses | **resolved** | Candidate mass is formed inside the active-slot guard at `ion.rs:498`; corresponding kernel guard at `ms2_ion.rs:613`. |
| Latest shared review #2: unsupported shifts counted/emitted | **resolved** | `base > 0` gates status, count and emission at `ion.rs:1274`, `ms2_ion.rs:1314`. Butanol regression at `tests/ms2_ion_kernels.rs:1176`. |
| Latest shared review #3: device address wrapping | **resolved** | Complete binding/scalar checks at `ms2_ion.rs:813–824`, `:1046–1057`, `:1451–1460`. |
| Latest shared review #8: mask expectations discarded | **resolved** | Complete comparisons at `tests/ms2_ion_kernels.rs:766–767`; hypothesis-1 and multispectrum positive cases at `:848`, `:865`. |
| Latest shared review #9, evidence statuses | **resolved** | Status 1/2 and incomplete-support bit documented at `docs/MS2_CONTRACTS.md:578–580`. |

**Remaining scenario for initial #1:** an ordinary `B=1,F=1,N=1,J=4` assignment executes `usize` multiplication/addition and an incrementing `usize` zeroing loop at [ion.rs:564](src/models/ms2/ion.rs:564). The mask/evidence twins have analogous addressing. The new launcher limits make those addresses equivalent on supported layouts, but the twins still fail the strict u32-only arithmetic requirement.

**New defects introduced by these fixes:** none confirmed in the scoped kernels.

Bindings are assignment **6**, mask **6**, evidence **5**. Outputs have disjoint lane ownership. I found no remaining reachable mass overflow for validated counts, or prohibited loop-carried scalar-argument initialization. Probability ranking belongs to the explicitly excluded integration kernels.

---

**PART C — enumeration**

The latest enumeration review’s seven findings:

| Finding | Disposition | Current evidence |
|---|---|---|
| #1: fill ignores row scored cap | **resolved** | Effective cap at `ms2_enum.rs:1129–1134`; wrapper clamps to M at `:2376–2377`. Independent fill-only/pad-only cap cases at `tests/ms2_enum_kernels.rs:1493`. |
| #2: unsafe DBE packing/comparison endpoints | **resolved** | Safe widened biasing at `formula_enum.rs:1937`; seen-bucket endpoint validation at `:2164–2183`. Original counterexamples rejected at `tests/ms2_formula_enum.rs:3471`. |
| #3: P_max checked before unrepresentable tuple exclusion | **resolved** | Mass exclusion precedes retained-row limit at `formula_enum.rs:2278–2294`; exact 16,384-row regression at `tests/ms2_formula_enum.rs:3554`. |
| #4: missing fill capacity exits | **resolved** | Initial exclusion at `ms2_enum.rs:1206`; traversal stop at `:1647`. Twin counterparts at `formula_enum.rs:3623`, `:4064`. |
| #5: missing lane ceiling/standalone offset address check | **resolved** | Dispatch validation at `ms2_enum.rs:2133`, `:2371`; offset length check at `:2383`. |
| #6: ppm rejection bypassed by early status exits | **resolved** | Validation now precedes exits at `formula_enum.rs:4355`; regression at `tests/ms2_formula_enum.rs:3571`. |
| #7: inadequate ownership/boundary tests | **resolved** | M>scored ownership at `tests/ms2_enum_kernels.rs:1493`; device saturation `:1760`; CH2/CH4 `:1806`; 2²⁰ ratios `:1847`; actual failed-row metadata `:1891`, plus downstream failure assertions in `tests/ms2_enum_integration.rs`. |

The latest review also carried eight second-review findings. Their current dispositions are:

| Carried finding | Disposition | Evidence |
|---|---|---|
| Portable u32 twin | **resolved** | `formula_enum.rs:3499` and shared helpers. |
| Counter saturation status | **resolved** | `formula_enum.rs:4165`; device saturation regression at `tests/ms2_enum_kernels.rs:1760`. |
| No-carbon margin oracle | **resolved** | `tests/ms2_formula_enum.rs:2784`. |
| Oversized tolerance/early ppm exits | **resolved** | `formula_enum.rs:4355`, `:4410`. |
| Product before division guard | **resolved** | Guarded products, e.g. `ms2_enum.rs:1379–1390`. |
| Rare representation/P policy | **resolved** | Documented representable-row policy and `formula_enum.rs:2287–2294`. |
| Missing cap/alignment checks | **resolved** | `formula_enum.rs:2074–2183`. |
| Weak boundary coverage | **resolved** | The latest seven-finding regressions listed above complete the concrete missing cases. |

Integration review Part D:

| Finding | Disposition | Current evidence / remaining failure |
|---|---|---|
| #1: table counter inequality rejects enumeration | **partly** | `CandidateBatch` is source-specific at `contract.rs:1669–1699`; `PackedCandidateBatch` still requires joined ≤ visited at `pack.rs:844–849`. |
| #2: MAX row conflates different formulas | **resolved** | Ten conditioning counts compared at `twin.rs:1239–1247`, `ms2.rs:7096` onward. |
| #3: training refuses excessive lanes after device work | **resolved** | Preflight precedes upload/bucket/encoder at `train.rs:1054–1091`. |
| #4: incomplete empty search reported absent | **resolved in production readout** | Shared reconciliation before validation at `generate.rs:1327–1409`; generation uses the batch-aware adapter at `:2472`. The older workspace adapter is explicitly documented as Table/profiler-only. |
| #5: extra shuffled exhaustion probe fails | **resolved** | Same donor-path evaluation supplies statuses at `examples/ms2_experiment.rs:546`, `train.rs:2006`. |
| #6: enum-fit permits validation leakage | **resolved** | Export subset check at `ms2_experiment.rs:270`, molecule overlap check at `:356`. |
| #7: missing 32B metadata estimate | **resolved** | Generation `workspace.rs:925–930`; training `:1896–1901`. |
| #8: checkpoint loads without required artifacts | **resolved** | `train.rs:2167–2175`. |
| #9: unavailable gold metric reported zero | **resolved** | Evaluation gold slots collected at `ms2_experiment.rs:489–492`; denominator/null handling at `:705`. |

**Remaining integration #1 — Major:** the earlier S₂ example can produce `visited=1, joined=2, scored=2`. Unpacked validation now accepts it, but [pack.rs:844](src/models/ms2/pack.rs:844) rejects the packed result. Thus legal enumeration output still fails `CandidateBatch::pack`/packed generation validation.

**New defect C1 — Major: chunk sizing trusts a budget argument that the kernel does not enforce.**  
[ms2_enum.rs:2149](src/tensor/ops/ms2_enum.rs:2149) sizes chunks from `lane_visits_max`, but the launched lane receives no such bound and instead reads its budget from metadata at `:1119–1123`. Fill repeats the discrepancy at `:2387`.

**Failing scenario:** supply valid metadata with budget 4,096, but call the wrapper with `lane_visits_max=1`, `dispatch_visits_max=1`. A spectrum admitting multiple heavy-vector visits—bounds fitted from C₂H₆ and N₂H₂, with a 50,000-unit uncertainty around C₂H₆—performs more than one visit in the supposedly one-visit dispatch. At production-sized chunks, the discrepancy can multiply job work substantially. Pass the advertised bound into the lane and enforce it, or enforce metadata consistency.

The lane, offsets, pad and **20 shared helper bodies match** after removing comments and normalizing indexing casts. Bindings are **4/4/5/2**. Effective-cap fixes restore fill/pad ownership for consistent pipeline inputs. I found no prohibited loop-carried initialization or unintentional overflow in their validated chemistry arithmetic.

---

**PART D — profile harness**

| Earlier finding | Disposition | Current evidence |
|---|---|---|
| Third-pass #1: misleading whole-stage WGPU time | **resolved for identified multi-pass stages** | `profile_ms` becomes unavailable when launches exceed the pass limit or uploads/reads occur, at `profile_ms2_substructure.rs:1887–1910`; scope is corrected. |
| Third-pass #2: table-only Enumerate estimate | **resolved** | Host artifact sizing `:389`; base generation `:755`; T+1 `:969`; training `:1334`; stability `:1633` use enumeration-inclusive estimates. |
| Third-pass #3: synchronization regression coverage | **partly** | Counter moved into `Device::try_synchronize`, but remains independent of the actual runtime synchronization. |
| Carried #1: replica workload | **resolved for supported Table profiling** | Shared production stages at `profile_ms2_substructure.rs:1995`, `:2011`, `:2055`; equivalence test at `tests/ms2_profile.rs:998`. Enumerate device profiling remains explicitly unsupported. |
| Carried #2: callback panic destroys state/token | **resolved** | Restoration and caught callback panic inside the profiling closure at `backend.rs:598–629`. |
| Carried #3: missing session identity | **resolved** | Session checks precede state removal in `Profiler`; recovery/nested-session tests remain present. |
| Carried #4: memory-limit propagation | **resolved** | Enumeration-inclusive estimates now cover base, slope and stability paths. |
| Carried #5: launch attribution tests | **resolved for the pinned CPU/WGPU paths** | Existing ordered-boundary/slope/stage tests remain applicable. |
| Carried #6: synchronization regression | **partly** | Same remaining issue as third-pass #3. |
| Carried #7: endpoint called peak | **resolved** | `reserved_bytes_after` and explicitly sampled peak reporting remain distinct. |

**Remaining synchronization scenario:** replace the runtime drain at [backend.rs:220](src/backend.rs:220) with `Ok(())`, retaining the increment at `:219`. The counter tests at `tests/ms2_profile.rs:627–644` still pass; later reads still mask missing completion. Moving independent bookkeeping into the synchronization method does not establish that synchronization occurred. Instrument the runtime operation or assert a deferred failure before any subsequent synchronization/read.

**New defects introduced by the profile fixes:** none confirmed.

| Part | Verdict |
|---|---|
| A — graph identity | **accept-with-fixes** |
| B — ion assignment/mask/evidence | **accept-with-fixes** |
| C — enumeration and integration | **reject** |
| D — profile harness | **accept-with-fixes** |