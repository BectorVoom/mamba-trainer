# MS2 V0-A (formula window and head) and V0-B (decoder specification): codex reviews and disposition

Date: 2026-10-03. Reviewer: `codex exec` (codex-cli 0.160.0, read-only, reasoning effort high).

1. **V0-B specification review**, run before implementation on the task prompt (kept in the session's work
   directory) against the architecture, the contracts and `grammar.rs`.
2. **V0-A code review** of the formula window kernel, `formula_top`, `nonzero_mask`, `formula_head.rs`, the twins
   and tests, and the kernel-review fixes.

## Disposition

| Review | Finding | Checked | Action |
|---|---|---|---|
| V0-B spec | Last teacher position would read a token row `T` | Yes | Boundary rule: unscored positions get id 0, mask "index 0 only", `use = 0`; no unsigned `length − 1` |
| V0-B spec | Truncating targets to `G` slots would break q's normalisation | Yes | `TargetBatch::build` rejects more targets than slots (`G = 16` equals the recipe's cap) |
| V0-B spec | `take_along_last` indexes inactive ids directly | Yes: no sentinel handling | Inactive target ids replaced by 0 before the gather |
| V0-B spec | Causality test too weak; no stepped parity | Yes | Test on `h` and unconditioned logits; a supplied-prefix stepped parity test through a new `step_logits` API |
| V0-B spec | GPU verification missing | Yes | The supervisor runs every new test on wgpu |
| V0-B spec | Adjacency wording; `DeviceTargets` incomplete; tables via `ms2_lookup` | Yes | Architecture §3.5 states why no adjacency is needed; spec completed |
| V0-B spec | Prerequisite tests did not compile at review time | Yes: V0-A was mid-edit | Verified after V0-A finished (CPU and wgpu green) |
| V0-A code | `gold_slots_host` ignored the window capacity `M` (blocker) | Yes | Fix scheduled in V0-B section 0, with the `M = 1` test |
| V0-A code | A NaN could make `formula_top` index with `u32::MAX` | Yes | Guarded writes; a `[NaN, 0]` test |
| V0-A code | The table hash covered row bytes, not the JSON the checkpoint names | Yes | Hash `FormulaTable::to_json()`; `DeviceFormulaTable::check` against `ModelConfig` |
| V0-A code | Parity outside validated metadata (adduct above `u16`, tolerance above 1000) | Yes | Inputs are validated before upload; parity is specified for validated inputs |
| V0-A code | Ranks read before shape checks in `score` / `loss` | Yes | Rank checks first |

V0-A verification (isolated copy, see the MS2 plan's progress log): `ms2_formula` 7 tests, footprint 41 launches
and no read per warmed call, all earlier MS2 suites green on CPU and wgpu. The CPU encode uploads a 16-byte
causal mask per call from the library's chunked scan (`Tensor::strict_causal_mask` in `ssd_chunked`); the GPU
path uploads nothing. That is library behaviour outside MS2 and is recorded, not changed.

## V0-B specification review as received

**Findings**

1. **Blocker — the claimed prerequisite baseline is not ready.** [v0b.md:5](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:5) says “present and verified,” but [tests/ms2_kernels.rs:700](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:700) closes the test immediately, leaving `let` statements at module scope. The regression command at [v0b.md:83](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:83) also names `ms2_formula`, whose test file is absent. The allowlist excludes repairing `ms2_kernels.rs`.
   **Fix:** require a repaired, verified prerequisite revision before handoff; correct the regression target list.

2. **Major — the final teacher position needs an explicit boundary rule.** [v0b.md:52](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:52) runs `T` positions, but position `T−1` requests token/replay row `T`. Neither exists. Masking the loss afterward does not prevent that access.
   **Fix:** “For `i < T−1`, shift tokens/replay by one. At `i=T−1`, synthesize target IDs 0, effective masks containing index 0 only, residuals 0, and use flags 0; never read row `T`. Stored `length` includes START and STOP. Scored positions satisfy `i+1 < length`.” Avoid unsigned `length−1` for empty slots.

3. **Major — truncating to `slots` does not preserve normalized q automatically.** [v0b.md:34](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:34) says take up to `slots` and cast q. Existing q is normalized over **all previously retained targets**, before this additional truncation: [targets.rs:532](/Users/ods/Documents/mamba-trainer/src/models/ms2/targets.rs:532), [targets.rs:550](/Users/ods/Documents/mamba-trainer/src/models/ms2/targets.rs:550). Thus q can sum below 1, contradicting the proposed test.
   **Fix:** preferably reject insufficient `slots`, preserving the frozen retained-target objective. If secondary truncation is intentional, explicitly renormalize selected integer weights in f64 before casting and document that this changes the training target distribution. Contracts retain the top 16: [MS2_CONTRACTS.md:443](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:443).

4. **Major — inactive masks need safe gather IDs too.** [v0b.md:56](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:56) changes masks and multiplies by use, but does not explicitly replace inactive target IDs. `take_along_last` directly indexes the supplied ID, without sentinel handling: [index.rs:619](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:619). Arbitrarily changed empty-slot tokens can therefore access outside a row before multiplication.
   **Fix:** “Where use=0, set the gathered target ID to 0 before gathering. Normalize ignored input padding to safe lookup IDs. Do not reinterpret an empty legal set for a **used** field as padding.” With safe IDs and finite logits, index-0-only plus use=0 gives exact zero contribution and gradient.

5. **Major — the tests do not establish decoder parity or the strongest causality boundary.** [v0b.md:65](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:65) only checks positions `<t` after changing token `t+1`; it misses leakage into `h[t]`, the output predicting that token. Conditional distributions and gathered probabilities at `t` may legitimately change when the target fields change.
   **Fix:** compare `h` and unconditioned head logits through position `t`; compare conditional heads with conditioning fields held fixed. Add a supplied-prefix stepped reference using `Mamba3Block::step`, without a sampler, and compare attention, atom memory, conditional logits and log-probabilities. This is explicitly required by [MS2_V0_ARCHITECTURE.md:345](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:345).

   Also add explicit tests for:
   - exact zero **gradients**, including poisoned padding and unused fields;
   - nonuniform q and the `B` denominator with unlabeled spectra;
   - atom/closure caps, full-length traces, repeated/parent/self closures, and malformed unused fields;
   - batch permutation and spectrum-alone versus batched equivalence.

   The 25%-of-initial, four-spectrum test at [v0b.md:68](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:68) is only a smoke test. Explicitly distinguish it from the architecture’s 128-spectrum, 10% NLL plus sampled-coverage experiment: [architecture:322](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:322).

6. **Major — verification conflicts with the required backend matrix.** [v0b.md:76](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:76) forbids wgpu, and verification is CPU-only. Architecture requires CPU and wgpu for replay, gradients and decoder parity: [architecture:338](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:338).
   **Fix:** provide an authorized GPU verification command, or explicitly record GPU verification as deferred and do not claim full architecture acceptance.

7. **Minor — adjacency omission is valid, but contradicts the architecture’s wording.** [v0b.md:25](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:25) excludes adjacency; [architecture:123](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:123) explicitly includes it, despite allocating the same `3A+16` words.
   **Fix:** reconcile the architecture text, documenting the redundancy proof below. Also specify initialization and residual values after the first illegal token; the prompt only explicitly zeros subsequent masks.

8. **Minor — complete the batch/API contract.** [v0b.md:35](/Users/ods/.claude/projects/-Users-ods-Documents-mamba-trainer/ms2_work/prompts/v0b.md:35) promises three uploads, but line 56 adds a fourth tensor. `DeviceTargets` lacks a declaration and consistent float generic.
   **Fix:** declare `DeviceTargets<R,E>`, include the use tensor, specify four uploads, `T=limits.max_steps()`, and matching decoder/replay capacities. Require nonempty targets to end in STOP: legal replay alone accepts prefixes. Explicitly mandate `Var::ms2_lookup` for every learned table; ordinary embedding backward reads IDs to the host.

**Verified correct**

- **The interior indexing is correct and leak-free.** Input token `i` produces `h[i]`, predicting token `i+1`; replay row `i+1` describes precisely that prefix. Atom created at step `s` uses `h[s−1]`; legal pointers at output `i` address only atoms with `s<=i`. This matches [architecture:219](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:219).

  Concrete example: budget `C₂H₆`, trace  
  `START, ADD(type=4,bond=0,pointer=0), ADD(type=4,bond=1,pointer=0), STOP`.  
  Type 4 is `CH₃`; types 1–4 are the budget-permitted carbon types: [contracts:256](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:256). Stored length is **4**, with **3 predicted tokens**.

  Masks below are bitsets ordered `(kind,type,bond,pointer)`; use flags have the same order.

  | Output i / input | Predicts / replay row | Raw masks | Residuals before target | Use |
  |---|---|---|---|---|
  | 0 / START | root / 1 | `(4,30,0,0)` | `(0,0)` | `1100` |
  | 1 / root | second atom / 2 | `(20,30,2,1)` | `(1,0)` | `1111` |
  | 2 / second atom | STOP / 3 | `(16,0,0,0)` | `(0,0)` | `1000` |
  | ≥3 / STOP or padding | padding | `(0,0,0,0)` | `(0,0)` | `0000` |

  Replace each unused field’s mask with bitset `1`. Atom steps are `[1,2,MAX,…]`; gather IDs are `[0,1,MAX,…]`, giving memory `[h[0],h[1],zero,…]`. At output 1 only atom 0 is pointer-legal. Output 2 has both atoms temporally available, but STOP uses no pointer. Replay row 0 is `(2,0,0,0)` and is never scored.

- **Replay equivalence is correctly specified by reference**, with these details essential to implementation:
  - Step 0: START only. Step 1: budget-permitted root types, **even if the supplied token kind is wrong**; bond/pointer masks zero: [grammar.rs:293](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:293).
  - STOP after any atom, including open valence or unspent budget; nothing legal afterward: [grammar.rs:255](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:255).
  - ADD types/bonds require a legal completion; budget counts element and fixed hydrogens: [grammar.rs:153](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:153), [grammar.rs:310](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:310).
  - CLOSE pointers satisfy `max(parent+1,last_close+1) <= p < newest`, both residual checks, and closure cap: [grammar.rs:224](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:224).
  - Unused fields must actually equal zero, not merely have zero masks: [grammar.rs:378](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:378).

  **No duplicate bond can be legal.** ADD creates a fresh endpoint. While that endpoint is newest, its existing earlier neighbours are its parent and previous closure targets. The closure lower bound excludes all of them; after another ADD, closures cannot return to the old newest atom. `last_close` resets on ADD: [grammar.rs:446](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:446). Thus `3A + 6 + 10` words suffice for legality, with budget and limits supplied externally; they do not retain the complete graph.

- **The loss formula is correct:** sum used-field NLL per trace, START excluded, STOP included; then `sum(q*nll)/B`, including unlabeled spectra in `B`: [architecture:243](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:243). Existing Labels q is normalized over its kept targets, subject to finding 3.

- **Conditional heads match the sampler algebra.** Target type for ADD, `c=18` for CLOSE, and target bond condition the appropriate heads before masking. Distributivity makes `(Q+E_type+E_bond)·k/√d` equal the sampler’s three packed pointer terms: [architecture:140](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:140), [architecture:234](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:234). Residual keys must remain position-specific. V0 residuals never exceed 6, so clamping at 7 changes nothing.

- **Existing primitives are sufficient**, with these exact interfaces:

  | Step | Existing function / semantics |
  |---|---|
  | Extract IDs | [`index::slice_ids_along`](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:342); device-side slicing |
  | Expand masks | [`ms2::bits_to_mask`](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1080); rank-1 u32 IDs → `[rows,width]`, width ≤32; zero bits remain all-zero |
  | Mask logits | [`Var::mask_logits`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:799); float mask, broadcast supported underneath; illegal values become finite minimum, gradients zero |
  | Normalize | [`Var::log_softmax(axis)`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:2609), [`Var::softmax(axis)`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:2599) |
  | Pick target | [`Var::take_along_last`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:1203); IDs shaped like leading dimensions; backward uses device `one_hot`, **no host read** |
  | Atom memory | [`Var::gather_tokens(&h,&ids,A)`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:1855); `[BG,T,d]` plus `BG*A` IDs → `[BG,A,d]`; MAX gives zero; strip the error column first |
  | Learned tables | [`Var::ms2_lookup`](/Users/ods/Documents/mamba-trainer/src/autograd/ms2.rs:36); flat IDs, out-of-range gives zero, device-only backward |
  | Products/movement | [`Var::matmul`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:300), `matmul_nt`; [`reshape`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:932), [`permute`](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:957) |

  `safe_ids` only replaces MAX; it neither decrements atom steps nor sanitizes arbitrary inactive IDs: [ms2.rs:1278](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1278). The decrement/shift adapters must be added. Ordinary `autograd::embedding` backward calls host-sorting `scatter_add_rows`: [index.rs:515](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:515).

**Not checked**

No files edited; no cargo, runtime tests, numerical gradients, GPU execution or profiling run. Parity and zero-read conclusions above are source-level assessments, not execution results. Git status was inaccessible under the sandbox.
## V0-A code review as received

## Findings

- **Blocker — host gold slots can exceed the device window.** The kernel caps scored rows at `min(rows_scored_max, M)` ([ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1629)); `gold_slots_host` calls the host search without `M` ([formula_head.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula_head.rs:365)). With the test’s three-row cluster, `M=1`, and gold row 1, the helper returns slot 1 while the device scores only slot 0. Pass `M` to the helper and cap its query before searching. Add that case to the test; its current claim that it applies `M` is incorrect ([ms2_formula.rs](/Users/ods/Documents/mamba-trainer/tests/ms2_formula.rs:817)).

- **Major — a NaN can make `formula_top` index with `u32::MAX`.** Ranking assumes every flagged slot has a unique rank, then indexes `best` without checking it ([ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:3482)). For two flagged slots with log probabilities `[NaN, 0]` and `F=2`, both rank zero; rank one is missing. Define and implement a total ordering for nonfinite values, or reject nonfinite inputs before launch.

- **Major — the uploaded hash cannot verify the checkpoint’s table reference.** `FormulaTableRef.sha256` specifies the table **JSON** ([contract.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/contract.rs:992)); upload hashes binary row bytes and makes no comparison ([formula_head.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula_head.rs:139)). Hash and compare the specified JSON at load, using one shared table identity check. The SHA-256 implementation itself has the standard padding, big-endian length and digest encoding by inspection ([formula_head.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula_head.rs:37)); the separate binary hash does not establish checkpoint identity.

- **Minor — “every `u32` input” parity fails outside validated metadata.** The twin narrows adduct metadata to `u16` ([twin.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/twin.rs:389)), while the kernel compares all 32 bits ([ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1643)). Minimal case: empty table, precursor `1_007_276`, uncertainty `0`, adduct `65_537`, tolerance `0`: the host reports completed/absent via adduct 1; the kernel reports `MASS_OVERFLOW`. Also, the kernel tolerance arithmetic is proved only for `ppm_tenths <= 1000` ([ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1540)). Validate these fields at the public boundary, or specify parity only for validated inputs.

- **Minor — malformed head shapes can panic before shape validation.** `score` reads dimensions before checking window rank, and `loss` reads `dims()[0]` before checking log-probability rank ([formula_head.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula_head.rs:224), [formula_head.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula_head.rs:301)). Check rank first.

## Verified correct

- For validated V0 metadata and finite scores, the window matches the host’s parent-mass handling, sentinel, tolerance, saturated bounds, row verdict and scored cap. Both searches contain steps 0–39; 40 exceeds the steps needed for a `u32` row count. The guarded reads preserve visit counts and partial-search exhaustion; counters, completion and status follow the host ([ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1520), [ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:3284), [formula.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula.rs:251)).
- For finite log probabilities, `formula_top` breaks ties by smaller slot, counts fewer than `F` joins and writes padding sentinels; `nonzero_mask` maps flags `0/1/2` to `0/1/1` ([ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:3448), [ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:3600)).
- With finite head values and valid slots, `score` applies the join mask and empty-window slot-zero rule. `loss` reduces the valid count on device and masks absent gold contributions, giving them zero gradient ([formula_head.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula_head.rs:248), [formula_head.rs](/Users/ods/Documents/mamba-trainer/src/models/ms2/formula_head.rs:309)).
- The requested peak fixes are present: count clamps in five kernels, full peak shape checks, poisoned stats, ordered subtraction, resident constants, and the repeated-ID test ([ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:108), [ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:630), [ms2.rs](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:889), [ms2_kernels.rs](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:708)). The three new kernels use 4, 5 and 2 array bindings, span loops, and no kernel-side `u64`, `f64`, infinity literal or fast-math-unsafe self comparison.

## Not checked

No cargo, generated WGSL, CPU/GPU execution or profiling was run. The tests do not cover the `M`-limited host gold slot, nonfinite top scores, SHA-256 padding boundaries, or malformed head ranks.