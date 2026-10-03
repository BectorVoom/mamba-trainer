# MS2 V0 architecture: codex review and disposition

Date: 2026-10-02. Reviewer: `codex exec` (codex-cli 0.160.0, read-only sandbox, reasoning effort high), reviewing
revision 1 of [MS2_V0_ARCHITECTURE.md](../MS2_V0_ARCHITECTURE.md) against the repository sources, the pinned
CubeCL 0.10 sources and the CubeCL manual. The cited source lines for the blockers were read before the
findings were applied. This is a review of a specification; nothing in it was built or run.

## Disposition

| Finding | Checked | Action in revision 2 |
|---|---|---|
| Atom memory in the decoder input makes the parallel pass circular | Yes, by construction of the spec | Decoder inputs are static lookups; atom memory is used by the pointer head only and gathered after the pass; mask row `i + 1` governs output `i` (§4.3) |
| Padding argument fails for NaN: the chunked scan multiplies by a zero band | Yes: `src/ssm/scan.rs:260-267` | `ms2_select_valid` selects padding to exact zeros before every block and after it; the multiplication masks are gone (§3.8, §4.1) |
| `embedding` has no sentinel handling and reads ids on the host in its adjoint; `mask_logits` wants float masks; all-masked rows give a uniform distribution | Yes: `src/tensor/ops/index.rs:502-515`, `elemwise.rs:550` | Adapters `ms2_lookup`, `ms2_bits_to_mask`, `ms2_safe_ids` and the explicit all-masked rule (§3.8) |
| Too many bindings for an 8-buffer wgpu adapter | Yes: `cubecl-wgpu-0.10.0/src/runtime.rs:308-313` | At most 6 arrays per kernel, packed buffers, preflight on `max_bindings` (§1, §3, §6.1) |
| Root ADD_ATOM reported legal with an empty type set | Yes, in both references | Both references and the contract changed; fixture cases and a Rust test added |
| Peak preprocessing under-specified (ties, zero maximum, NaN, squaring overflow, totals) | Yes | §3.1 rewritten; `peak_id` must increase within a spectrum (contracts §3.1) |
| `read_count` skips tuning reads | Yes: `src/backend.rs:741-762`, `matmul.rs:2892` | `runtime_read_count` and a warmed-bucket protocol (§5, §6.3) |
| Identical sampling across backends is not achievable | Accepted | The sampler test asserts exact RNG words and masks, log-probabilities within tolerance, frequencies and same-backend repeatability (§8) |
| Loss denominator differs from the design | Yes | Divide by `B`; unlabeled spectra contribute zero (§4.4) |
| Fourier "full precision" wording | Yes | Exact remainder, `f32` phase; ordered subtraction (§3.2) |
| Task coverage gaps (P2 infrastructure, experiments, candidate construction) | Yes | §5 to §7 added |
| `T * L_step` versus `T − 1` iterations | Yes | `(T − 1) * L_step` (§5) |

Left open on purpose: the functional mixer step allocates per step, which the design's zero-allocation
acceptance forbids; V0 measures it and P8.2 owns the fix (§5).

## Review as received

## Findings

**Blocker — Q7(a): atom memory makes the proposed parallel decoder circular.** Input embeddings consume `atom_memory`, while that memory is gathered from the final decoder output `h` ([architecture:165–170](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:165)). Consequently, building the inputs requires first computing the outputs.

The shift itself is correct: an atom emitted at token step `s` stores `h[s−1]`. For predicting token `t`, only atoms with `atom_step < t` may be visible; residuals and legality masks must describe the prefix **before token `t`**, hence use replay row `i+1` for output `h[i]`. Replay supplies those prefix residuals ([architecture:93–98](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:93)), but the spec gives no executable dependency order.

**Fix:** use static pointer/type embeddings in decoder inputs and creation-state memory only in the pointer head, permitting a parallel decoder followed by gathering memory. Alternatively, specify differentiable sequential teacher forcing and abandon the parallel-pass claim. Preserve `u32::MAX` when generating `atom_step−1`; subtracting from the sentinel produces an invalid, non-IGNORE index.

**Blocker — Q2: the padding argument does not support NaN poison.** The mathematical recurrence is causal: the apparent next-position trapezoid coefficient contributes only when `j<t`, therefore `j+1≤t` ([scan.rs:8–24](/Users/ods/Documents/mamba-trainer/src/ssm/scan.rs:8)). However:

- Chunked intra-scan computes dense `C Bᵀ`, multiplies it by a zero upper-triangular band, then multiplies by **all** source inputs. Future NaNs survive either multiplication: `NaN*0=NaN` ([scan.rs:256–267](/Users/ods/Documents/mamba-trainer/src/ssm/scan.rs:256)).
- Inter-chunk scan similarly multiplies masked transfer coefficients by all chunk summaries; a future NaN summary can contaminate an earlier chunk ([scan.rs:274–307](/Users/ods/Documents/mamba-trainer/src/ssm/scan.rs:274)).
- The fused scan emits `y[t]` from the incoming state and current `g`, then retains the `w[t]` update for later outputs. Its current output does not use future `w[t]` ([ssd_scan.rs:231–246](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ssd_scan.rs:231)).

Thus bounded finite suffixes with finite intermediates have no mathematical influence on earlier outputs. **Finite but arbitrary inputs are insufficient:** overflow in projections or dense products can create infinities/NaNs that reach masked terms. CPU defaults to the chunked path; GPU normally uses the fused path ([scan.rs:123–128](/Users/ods/Documents/mamba-trainer/src/ssm/scan.rs:123)).

**Fix:** select padding to exact finite zeros **before** any projection/scan and select outputs afterward. Require finite weights and intermediates. Replace the multiplication masks in §4.1; the design explicitly requires selection and preservation of every carry ([design:91–93](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_DESIGN.md:91)). Literal poisoned inputs passed into the existing chunked mixer are not safe.

**Blocker — Q1/Q7(b): several proposed gathers and masks lack necessary adapters.**

- `autograd::embedding` calls unchecked `gather_rows`, which has **no IGNORE handling** ([ops.rs:2644](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:2644), [index.rs:434–438](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:434)). Formula-window padding is `u32::MAX`, yet §4.2 gathers every slot ([architecture:78](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:78), [154](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:154)).
- `mask_logits` requires a **floating 0/1 tensor**, whereas replay produces packed `u32` bitsets ([ops.rs:786–803](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:786), [architecture:95](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:95)). Passing flag `2` as a float mask also scales its backward gradient by two.
- All-masked logits become identical finite minima; `log_softmax` consequently returns a uniform distribution, not an empty support ([elemwise.rs:550–553](/Users/ods/Documents/mamba-trainer/src/tensor/ops/elemwise.rs:550), [ops.rs:2609–2612](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:2609)).

**Fix:** specify sentinel-safe gathers, bitset-to-float-mask expansion, and explicit empty-row/unused-field handling. Mask only after constructing the complete conditional additive logits.

**Major — Q3: the separate-buffer sampler is not portable to low-limit wgpu adapters.** Section 3.6 names **seven float input arrays**, before grammar state, IDs, actions, lengths, status, or score outputs ([architecture:107–119](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:107)). WGSL also adds a storage binding for scalar/shape information ([shader.rs:134–158](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/shader.rs:134)).

There is no universal limit of 10. wgpu’s default is **8 storage buffers per shader stage** ([limits.rs:152](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/wgpu-types-29.0.4/src/limits.rs:152)); CubeCL requests the adapter’s limits ([wgsl.rs:30–38](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/backend/wgsl.rs:30)) and advertises `max_bindings = limit−1` ([runtime.rs:308–313](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/runtime.rs:308)). Separate-buffer peak gathering also needs a binding audit.

**Fix:** pack into four typed bindings: float logits/tables; integer IDs/budgets/tables; mutable integer grammar/actions/status; mutable float scores. Address fields by offsets inside each kernel, leaving room for metadata. The manual supports internal slicing of one physical argument ([buffer-slicing manual:20–28](/Users/ods/Documents/cubecl_manual/manual/Cubecl/Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md:20)). Check every kernel’s complete binding count before launch.

**Major — Q8: literal reference parity conflicts with root completion legality.** Both references return ADD as legal at step 1 even if no type fits:

- [Python:282–285](/Users/ods/Documents/mamba-trainer/tools/ms2/ms2_reference.py:282).
- [Rust:241–248](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:241).

A zero composition budget therefore produces a nonempty kind set followed by an empty type set. This contradicts “a kind is legal only if some completion exists” ([contracts:327](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:327)). Rust’s separate `has_legal_action` already detects this root failure ([grammar.rs:434–435](/Users/ods/Documents/mamba-trainer/src/models/ms2/grammar.rs:434)), but the sampler specifies checking only legal kinds.

**Fix:** reconcile both references and the contract: root ADD is legal iff its type mask is nonempty; otherwise emit `no_valid_action`. Check every conditional support before sampling. Specify root ADD uses only kind/type, CLOSE uses kind/bond/pointer with serialized type zero, and STOP uses kind only.

**Major — Q5: sorting is sound, but preprocessing is not fully defined or contract-consistent.**

- Ties use raw array index, while contracts require original `peak_id`, which can differ after host preselection ([architecture:48–51](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:48), [contracts:118,134–136](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:118)).
- With `max=0`, `intensity >= floor*max` keeps zero-intensity peaks; relative intensity and retained fraction involve zero denominators. The reference returns empty when `top≤0` ([Python:461–470](/Users/ods/Documents/mamba-trainer/tools/ms2/ms2_reference.py:461)).
- NaN/negative inputs are silently excluded, although contracts require request status bits ([contracts:120](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:120)).
- Squaring a finite large scale-1 intensity can overflow; eligibility before squaring does not prevent this ([architecture:43–45](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:43)).
- The transformed intensity used by all four kernels, summation order, numerator of `retained`, and empty-spectrum result are unspecified.

**Fix:** pass `peak_id`; use the same transformed intensity everywhere; require positive finite maximum; define `total=Σ eligible-above-floor relative`, `retained=Σ selected relative/total`, with both zero for an empty spectrum. Validate transformed values, capacities, precursor arithmetic, and emit required diagnostics.

**Major — Q9: zero `read_count` does not prove zero reads.** Matmul tuning performs a real batched device read under `uncounted_reads` ([matmul.rs:2892–2895](/Users/ods/Documents/mamba-trainer/src/tensor/ops/matmul.rs:2892)); that helper explicitly suppresses the counter while retaining synchronization ([backend.rs:741–762](/Users/ods/Documents/mamba-trainer/src/backend.rs:741)). Cold block/attention matmuls can therefore read invisibly. Embedding backward additionally reads IDs to build host buckets ([index.rs:513–515](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:513)).

**Fix:** add an unsuppressed runtime-read counter; isolate counter tests; distinguish cold setup, warmed production, backward, and profiling. Warm every bucket before claiming a production zero-read boundary.

**Major — Q4: matching seeds does not guarantee matching sampled actions.** The WGSL backend emits builtin `exp` ([WGSL instructions:644](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:644)); CPU lowers it to an LLVM intrinsic ([CPU arithmetic:228](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/operation/arithmetic.rs:228)). Small exp/reduction differences can move a CDF boundary across the identical draw and select a different integer token. “Within tolerance” cannot describe those token differences ([architecture:216](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:216)).

**Fix:** assert exact RNG words, masks and deterministic integer outputs; tolerance on probabilities/log-probabilities for the **same supplied prefix**; same-backend seed repeatability and batch independence; distributional agreement/frequencies. Define the host twin with wrapping-u32 hashes, exact 24-bit draws, explicit f32 rounding and fixed legal-index accumulation order. Require identical actions only when the draw is separated from every CDF boundary by the numerical error bound. Universal identical sampling requires deliberately standardized arithmetic.

**Major — Q7(c): the loss denominator differs from the design.** The architecture divides by `B_labeled` ([architecture:184–187](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:184)); the design says `mean_b` ([design:269](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_DESIGN.md:269)). They agree only for batches containing exclusively labeled spectra. Contracts exclude unlabeled graph contributions but do not establish `B_labeled` normalization ([contracts:464–465](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:464)).

**Fix:** define the averaging population explicitly across all three documents and microbatch accumulation. To retain the stated design objective, divide by `B`, with unlabeled terms zero. Also define formula-loss averaging and zero-denominator behavior.

**Minor — Q6: integer modulo preserves the remainder, not full f32 phase precision.** `%` is integer modulo on both backends ([WGSL instructions:600–604](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:600), [CPU arithmetic:327–332](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/operation/arithmetic.rs:327)). For nonzero `w≤2e9`, `v%w` is safe for every u32 `v`. But f32 has 24 significant bits ([random.rs:148–156](/Users/ods/Documents/mamba-trainer/src/tensor/ops/random.rs:148)); near `2e9`, spacing is 128 integer units, so adjacent remainders can collapse and the ratio can round to 1.

**Fix:** change “full precision” to “exact integer remainder followed by f32 phase approximation”; freeze the integer wavelength table. Compute `|c−m|` by ordered integer subtraction, since peaks above precursor are allowed. The sin/cos pair already removes the mathematical wrap discontinuity; only small numerical error remains.

**Major — Q10: required task coverage remains incomplete.** Naming files or tests does not specify these requirements:

| Task | Missing or contradicted; concrete addition |
|---|---|
| [P2.1:68](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:68) | API/allocation/read audit, optimizer and current MIMO paths. Add an audited dependency/capability table. |
| [P2.2:69](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:69) | Complete checked estimator, 129/258 MiB reproduction, functional-step temporaries, training state and movement bytes. Specify formulas and overflow/OOM checks. |
| [P2.3:70](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:70) | Reusable outputs, bounded bucket retention and explicit stream ownership. Functional step allocation ([architecture:199](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:199)) contradicts the no-hot-loop-allocation acceptance ([design:259](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_DESIGN.md:259)). Add reusable outputs/banks or explicitly revise acceptance. |
| [P2.4:71](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:71) | Capability checks, including binding limits and timer availability. Specify preflight behavior. |
| [P2.5:72](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:72) | Single-u32 proof for mass, signed adduct adjustments, tolerance/intermediate arithmetic and sentinels. Add independent decimal boundary tests. |
| [P2.6:73](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:73) | Stage spans, `client.profile`, synchronized timing, warmup/cold metadata and machine-readable report schema. |
| [P2.7:74](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:74) | Allocation calls, transfers, live/peak bytes and direct/suppressed runtime reads. Existing counters are insufficient. |
| [P2.8:75](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:75) | Alternating buckets, OOM preflight, alias violations and launch-error checks. Define complete writes after early failure, not just output poisoning. |
| [P2.9:76](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:76) | Counter isolation, warmup settling, profiling separation and alternating-bucket stability; only 200 repeated calls are mentioned. |
| [P3.1:82](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:82) | Original-ID mapping and correct empty/invalid preprocessing; fix Q5. |
| [P3.2:83](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:83) | Integer sidecar/FP32 separation is stated; add explicit boundary and dtype acceptance tests. |
| [P3.3:84](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:84) | Specify the full contracted SSM configuration, including both directional mixers being internally unidirectional. |
| [P3.4:85](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:85) | Select masks, full-cache freeze, independent reset, missing metadata and short/long cases. Fix Q2 and freeze finished decoder `h/last_u/angle`, not merely sampler state. |
| [P3.5:86](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:86) | Proposed tests cover this, but need exact-config block parity, explicit carry comparisons and tolerances. |
| [P3.6:87](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:87) | Required N/B profile grid and recorded memory exclusions are absent. |
| [V0.1:93](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:93) | Shapes/shared K/V match; connect A/R/T to coverage and specify direct composed attention. |
| [V0.2:94](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:94) | Oracle path/provenance and absent-gold versus search-exhaustion semantics. Partial windows must not claim complete-support probabilities ([contracts:519–522](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:519)). |
| [V0.3:95](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:95) | Root empty-support behavior, START/state initialization, owned/frozen carries and complete failed-request initialization. |
| [V0.4:96](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:96) | Resolve memory circularity, prefix indexing, mask conversion, denominator and embedding-backward reads. |
| [V0.5:97](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:97) | Overfit protocol, molecule-disjoint pilot, shuffled control, structure-prior baseline and uncertainty reporting are absent. |
| [V0.6:98](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:98) | Specify construction of every CandidateBatch field/status, including failed requests and unresolved identity/evidence fields. |
| [V0.7:99](/Users/ods/Documents/mamba-trainer/docs/MS2_SUBSTRUCTURE_TASKS.md:99) | Full forward/backward/generation profiling and predeclared pilot/hardware acceptance targets are absent. |

## Verified correct

**1. Existing primitives**

| API | Actual semantics |
|---|---|
| `Var::gather_tokens` | Per-batch IDs, `[B,S,d]→[B,R,d]`; IGNORE yields zero; backward sums all matching gathered gradients without atomics ([autograd:1852](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:1852), [kernel:605](/Users/ods/Documents/mamba-trainer/src/tensor/ops/entity_model.rs:605), [adjoint:679](/Users/ods/Documents/mamba-trainer/src/tensor/ops/entity_model.rs:679)). Other IDs require bounds validation. |
| `Mamba3Block::{apply,step,empty_cache}` | Exist; pre-norm/residual wrappers, step consumes `[B,1,d]`, returns a fresh cache; zero cache includes full SSM carries ([mamba3.rs:885](/Users/ods/Documents/mamba-trainer/src/models/mamba3.rs:885), [922](/Users/ods/Documents/mamba-trainer/src/models/mamba3.rs:922), [947](/Users/ods/Documents/mamba-trainer/src/models/mamba3.rs:947), [713](/Users/ods/Documents/mamba-trainer/src/models/mamba3.rs:713)). |
| `Var::mask_logits` | Exists; float 0/1 mask, finite minimum for illegal logits, gradient multiplied by mask ([ops.rs:786](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:786)). |
| `Var::log_softmax` | Exists; max-shift, exp, sum, log along requested axis ([ops.rs:2609](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:2609)). No empty-support handling. |
| `Var::take_along_last` | Exists; one ID per row, removes last dimension; no IGNORE/bounds handling; adjoint uses one-hot expansion ([ops.rs:1202](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:1202), [index.rs:607](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:607)). |
| `autograd::embedding` | Exists; output `ids.shape+[width]`, table-gradient scatter-add; backward performs a host ID read ([ops.rs:2644](/Users/ods/Documents/mamba-trainer/src/autograd/ops.rs:2644), [index.rs:445](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:445), [514](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:514)). |
| `IdTensor`, `read_all` | Dense u32 storage; batched ID/float read exists ([index.rs:15](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:15), [167](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:167)). One read requires one stream; mixed streams fall back to multiple reads ([backend.rs:309](/Users/ods/Documents/mamba-trainer/src/backend.rs:309)). |
| `launch_1d`, `launch_1d_spans` | Crate-private, available to proposed internal kernels; both increment launch count. Spans require looping over the returned span on CPU ([backend.rs:917](/Users/ods/Documents/mamba-trainer/src/backend.rs:917), [969](/Users/ods/Documents/mamba-trainer/src/backend.rs:969)). |
| `hash_u32`, `hash_unit_f32` | Exist as `#[cube]` helpers; wrapping integer hash and exact top-24-bit draw in `[0,1)` ([random.rs:135](/Users/ods/Documents/mamba-trainer/src/tensor/ops/random.rs:135), [155](/Users/ods/Documents/mamba-trainer/src/tensor/ops/random.rs:155)). Host twin must use wrapping multiplication explicitly. |
| `Tensor::from_f32` | Exists; shape-checked upload after conversion to element type ([base.rs:86–102](/Users/ods/Documents/mamba-trainer/src/tensor/base.rs:86)). |

The reversal IDs are a per-spectrum permutation: reverse `[0,len)` and leave `[len,N)` fixed. Applying it twice returns the original tensor, including padding; **yes, gather implements it and it is its own inverse** ([architecture:54–55](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:54)).

**No existing test was found for block-level apply/step parity with the exact no-convolution configuration.** Related evidence exists: rotational SISO scan/step parity ([tests/ssm.rs:332–398](/Users/ods/Documents/mamba-trainer/tests/ssm.rs:332)) and learned-trapezoid rotational SISO LM parity ([tests/model.rs:153–200](/Users/ods/Documents/mamba-trainer/tests/model.rs:153)); the LM test retains default `conv_kernel=Some(4)` ([config.rs:143](/Users/ods/Documents/mamba-trainer/src/ssm/config.rs:143)).

**2. Causality:** per-position projections and rotational prefix sums are causal ([mamba3.rs:389](/Users/ods/Documents/mamba-trainer/src/models/mamba3.rs:389), [scan.rs:597–604](/Users/ods/Documents/mamba-trainer/src/ssm/scan.rs:597)). The trapezoid lookahead does not create mathematical future dependence. Numerical qualifications are in Findings.

**3. Kernel language feasibility:** the proposed constructs have lowering support on both backends:

- Mixed u32/float arrays and `u32::MAX`: WGSL bindings use each argument’s own element type and typed constants; CPU supports u32 and floating types ([WGSL shader:310](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/shader.rs:310), [constants:352–369](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/base.rs:352), [CPU types:25](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/elem.rs:25)).
- U32 division/modulo: direct WGSL `/` and `%`, CPU unsigned division/remainder ([WGSL:600](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:600), [634](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:634), [CPU:190](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/operation/arithmetic.rs:190), [327](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/operation/arithmetic.rs:327)). Divisors must be nonzero.
- Dynamic/nested loops and `break`: recursive WGSL loop-body emission and CPU loop/break control-flow lowering ([WGSL:800–825](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:800), [884](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:884), [CPU:118–150](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/block.rs:118)). No fixed trip-count prohibition appears here; actual runtime cost remains unmeasured.
- Multiple outputs and lane-owned read/write scratch: supported indexed loads/stores. Initialize before reading, keep lane rows disjoint, and avoid overlapping independent bindings ([WGSL:570–584](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:570), [CPU stores:192](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/operation/operator.rs:192), [manual:28](/Users/ods/Documents/cubecl_manual/manual/Cubecl/Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md:28)).
- Bond bitsets: OR/AND/XOR/shifts are supported; one u32 adjacency word per atom suffices for A=16 ([WGSL:912–950](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.10.0/src/compiler/wgsl/instructions.rs:912), [CPU:78–100](/Users/ods/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpu-0.10.0/src/compiler/visitor/operation/bitwise.rs:78)).

**4. RNG identity:** the specified hash depends on stable spectrum ID, trajectory, step, field and seed, not batch position ([architecture:122–125](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:122)). This supports same-backend batch independence, subject to numerical/model invariance.

**5. Rank/order/reversal:** on positive finite transformed intensities, total-order tie-breaking gives unique ranks and positions, handles duplicate masses, and selects exactly `min(N, kept_count)` peaks. `len−1−p` has no off-by-one when evaluated only for `p<len`. Empty handling needs the fixes above ([architecture:47–55](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:47)).

**6. Fourier wraps:** integer modulo is sound; the sin/cos representation is periodic and needs no additional wrap-discontinuity treatment. The full-precision claim needs qualification.

**7. Decoder normalization and attention:** adding bond/type offsets and the three pointer score components **before** masked log-softmax gives correctly normalized conditional distributions over nonempty supports ([architecture:174–178](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:174)). The token NLL correctly includes STOP and excludes START/padding ([184–185](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:184)).

Flattening `[B*G,T,d]` into `[B,G*T,d]` for **cross-attention only** is mathematically equivalent: each query independently attends over its spectrum’s memory, with softmax over memory positions. Restore target layout before Mamba scanning; do not introduce query-axis normalization, causal query masks, or target-dependent position transforms. Numerical summation plans may differ.

**8. Non-root legality:** the references implement formula **upper bounds**, STOP after any atom, BFS parent ordering, valence constraints, and completion-aware ADD/type/bond/pointer and CLOSE/bond/pointer supports ([Python:236–298](/Users/ods/Documents/mamba-trainer/tools/ms2/ms2_reference.py:236), [contracts:311–328](/Users/ods/Documents/mamba-trainer/docs/MS2_CONTRACTS.md:311)). With those supports, later fields cannot be empty after a legal non-root selection. The root exception is the identified mismatch.

**9. Uploads/reads:** `IdTensor::from_slice` and `Tensor::from_f32` upload via `create_from_slice` and **do not increment read_count** ([index.rs:60](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:60), [base.rs:96–102](/Users/ods/Documents/mamba-trainer/src/tensor/base.rs:96)). `mask_logits` and `log_softmax` have no direct host reads; block step and composed attention have no explicit operand read, but their matmuls can trigger the hidden tuning reads described above.

**10. Scope:** the proposed shapes, direct shared spectrum memory, factor order, bounded formula hypotheses, fixed-step generation and trace-only duplicate flags broadly match V0. The task-specific gaps are listed in Findings.

## Not checked

- No files edited; no cargo commands or kernels run.
- Actual CPU/WGSL compilation, adapter limits, performance, gradients, allocation stability and sampling frequencies remain unverified.
- `CrossAttention`, `FormulaTable::window`, and the proposed MS2 kernels are specification-level dependencies here; existing self-attention supplies only a composed-operation example ([attention.rs:328–348](/Users/ods/Documents/mamba-trainer/src/nn/attention.rs:328)).
- The launch formula uses `T*L_step`, while the loop schedules `T−1` iterations; reconcile the counter definition before pinning it ([architecture:192–196](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:192)).