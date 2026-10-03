# MS2 kernels, counters and estimates: codex review and disposition

Date: 2026-10-03. Reviewer: `codex exec` (codex-cli 0.160.0, read-only, reasoning effort high), reviewing the first
implementation of `src/tensor/ops/ms2.rs`, `src/autograd/ms2.rs`, `src/models/ms2/{twin,workspace}.rs`, the
counter additions to `src/backend.rs`, `src/tensor/base.rs`, `src/tensor/ops/index.rs`, and their tests. The
review ran while the fast-math finiteness fix was being applied (it notes the files changed under it). No code
was run by the reviewer; the GPU result below is from the supervisor's wgpu run on the M1.

## Disposition

| Finding | Checked | Action |
|---|---|---|
| Kernels index past the input when `peak_count > n_raw` (invalid metadata the public function accepts) | Yes | Kernels clamp the count to `n_raw`; the device batch uploads 0 for fatal spectra (contracts §3.1) |
| Incomplete shape checks in `peak_features` / `peak_select` | Yes | Full shape validation before any dimension is read |
| `FINITE_MAX = 3e38` makes finite intensities at or above it ineligible | Yes; the range test replaced a fast-math-unsafe `t - t == 0` that failed on Metal | Kept the range test (fast-math-safe) and moved the rule to request validation: such values are `nonfinite_input` (contracts §3.1), so nothing valid is silently dropped |
| `bits_to_mask` does not turn formula flags 1 and 2 into 1 | Yes | A separate nonzero-to-mask adapter lands with the formula window task (V0.2) |
| Unchecked width arithmetic in the estimate | Yes | `ModelConfig::validate` bounds every dimension at 65,536, so no derived width can overflow |
| Memory inventory omits peak stats/summary and training atom memory and head scratch | Yes | Added; the training test expects them |
| `c - m` evaluated before the `m > c` branch (wrapping, result still correct) | Yes | Subtraction moved inside the branches |
| Capability probe lacks the timing method | Yes | With the profile driver (P2.6) |
| `PeakBuffers::poisoned` leaves `stats` unpoisoned; no repeated-id lookup gradient test | Yes | Poison `stats`; a repeated-id backward test |

GPU: all kernel tests pass on wgpu/Metal and CPU after the finiteness fix
(`cargo test --release --no-default-features --features wgpu --test ms2_kernels --test ms2_kernel_launches
--test ms2_counters --test ms2_workspace`).

## Review as received

## Findings

Line references use the latest contents read; `ms2.rs` and its twin changed during this review.

- **Blocker — unchecked reads when `peak_count > n_raw` (2).** Stats loops directly over metadata’s count, indexing both input arrays; rank, order, gather and summary repeat this assumption. Example: `B=1, n_raw=64, peak_count=65` reads past the buffers. This is invalid contract metadata, but the public safe wrapper accepts it. Evidence: [ms2.rs:127](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:127), [ms2.rs:585](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:585). **Fix:** enforce validated metadata at upload and guard oversized counts before unchecked indexing; fatal spectra should use count zero as specified.

- **Blocker — incomplete shape validation permits another unchecked read (2).** `peak_features` checks kept’s rank, but never its last dimension. `kept=[0]` shaped `[1,1,1]`, with otherwise matching inputs, reaches `kept[1]`. Evidence: [ms2.rs:848](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:848), [ms2.rs:801](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:801). Separately, a rank-one `out.kept` panics while reading dimension 1 before returning a shape error: [ms2.rs:601](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:601). **Fix:** validate complete shapes before extracting dimensions.

- **Major — eligibility excludes valid finite intensities; twin shares the bug (1, 3).** `FINITE_MAX=3e38` and strict `<` reject every finite transformed intensity at or above that value. Minimal spectrum: one peak, `mz=100_000_000`, precursor `200_000_000`, intensity `3e38`, scale 0. Specification requires `max=3e38, total=1, len=1`; implementation returns empty. Evidence: [ms2.rs:44](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:44), [ms2.rs:154](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:154), [twin.rs:47](/Users/ods/Documents/mamba-trainer/src/models/ms2/twin.rs:47). **Fix:** use a portable positive-finite test covering the entire finite `f32` range, and independently test its boundaries.

- **Major — formula flag conversion is missing (1, 3).** `bits_to_mask([2], 1)` returns `[0]`; architecture §3.8 requires flags 1 and 2 both to become 1. The twin implements the same bit extraction. Evidence: [ms2.rs:990](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:990), [twin.rs:266](/Users/ods/Documents/mamba-trainer/src/models/ms2/twin.rs:266), [architecture:168](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:168). **Fix:** provide an explicit nonzero-flags conversion, distinct from bit-set expansion.

- **Major — memory arithmetic is not checked end to end (6).** Widths are calculated with unchecked `usize` arithmetic before conversion to `u64`: [workspace.rs:161](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:161), [workspace.rs:634](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:634), [workspace.rs:583](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:583). Underlying products/sums are unchecked at [config.rs:165](/Users/ods/Documents/mamba-trainer/src/ssm/config.rs:165) and [config.rs:192](/Users/ods/Documents/mamba-trainer/src/ssm/config.rs:192). Example on this 64-bit host: V0 encoder with `head_dim=1<<62` overflows `4*head_dim`, panicking in debug or wrapping in release. **Fix:** derive every width with checked arithmetic before using it.

- **Major — memory inventory is incomplete (6).** Both `peak_selection` formulas omit stats and summary: **`B*(3*elem_bytes+8)`**, despite their allocations at [ms2.rs:71](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:71). Training also omits atom memory and head scratch from its inventory, although teacher-forced decoding gathers atom memory and computes those heads: [workspace.rs:530](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:530), [architecture:223](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:223). Its test explicitly expects those items absent: [ms2_workspace.rs:118](/Users/ods/Documents/mamba-trainer/tests/ms2_workspace.rs:118). **Fix:** account for training’s corresponding buffers and gradients, and reconcile actual peak allocations before treating the estimate as a memory admission bound.

- **Minor — feature difference performs an unsigned underflow (2, 3).** `v=c-m` executes before the `m>c` correction. Example: `c=50_000_000, m=50_000_001`, an eligible peak. Final features are correct under wrapping arithmetic, but arithmetic differs from the twin’s `abs_diff`. Evidence: [ms2.rs:824](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:824), [twin.rs:244](/Users/ods/Documents/mamba-trainer/src/models/ms2/twin.rs:244). **Fix:** initialise `v=0` and perform subtraction only inside the appropriate comparison branch.

- **Minor — capability probe omits timing method (6).** No field reports device timestamps versus system timing, required by §6.1. Evidence: [workspace.rs:33](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:33), [architecture:277](/Users/ods/Documents/mamba-trainer/docs/MS2_V0_ARCHITECTURE.md:277). **Fix:** report the runtime timing method and profiling availability.

- **Minor — claimed poison coverage excludes stats (7).** `PeakBuffers::poisoned` allocates stats uninitialised instead of filling it with NaNs: [ms2.rs:89](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:89). **Fix:** poison stats too.

## Verified correct

1. **Kernel results, except findings above.**

   - **Stats:** eligible maximum and index-ordered thresholded sum; all three columns written: [ms2.rs:130](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:130), [ms2.rs:202](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:202).
   - **Rank:** decreasing transformed intensity, smaller-index ties, sentinel otherwise: [ms2.rs:282](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:282).
   - **Order:** increasing mass, smaller-index ties: [ms2.rs:364](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:364).
   - **Gather:** raw index, mass, valid-prefix reversal and explicit padding writes: [ms2.rs:469](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:469).
   - **Summary:** capped length, bit **19** iff qualifying count exceeds `N`, retained fraction summed in raw-index order: [ms2.rs:543](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:543).
   - **Features:** seven scalars, then mass’s 16 sin/cos pairs, then absolute-loss pairs; integer modulo precedes conversion. All padding features are zero. Wavelengths match the stated rounding formula: [ms2.rs:33](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:33), [ms2.rs:795](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:795).
   - **Adapters:** selection branches rather than multiplying; lookup zeroes all `ids>=V`; safe_ids replaces only `u32::MAX`: [ms2.rs:921](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:921), [ms2.rs:1053](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1053), [ms2.rs:1187](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1187).

2. **Safety within validated V0 inputs.** `n_keep>n_raw` is safe; excess kept slots receive padding. Batch zero returns before launching: [ms2.rs:652](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:652). Scalar kernels impose no vector-width divisibility requirement. Every kernel loops over its assigned span; geometry partitions those spans: [backend.rs:1088](/Users/ods/Documents/mamba-trainer/src/backend.rs:1088). No cross-lane output ownership violation or uninitialised-output read found. Stats column 2 is initialised by kernel 1 and overwritten by kernel 5, which reads only columns 0–1: [ms2.rs:204](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:204), [ms2.rs:540](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:540).

   Array binding counts, in kernel order, are **4, 5, 4, 6, 6, 5, 3, 2, 3, 3, 2**. No kernel `continue`, explicit 64-bit integer, `f64`, infinity literal, or unsupported cast found. Arbitrarily large shapes are not protected against index/product overflow; V0 bounds avoid it apart from the subtraction noted above.

3. **Twins and circular comparisons.** Twins preserve each result’s arithmetic and accumulation order; summary combines two independent passes, which does not change either accumulation. They are mirrors rather than independent oracles, demonstrably sharing the eligibility and flag bugs.

   Kernel-derived expectations occur in poisoned-versus-clean padding comparisons and batch permutations: [ms2_kernels.rs:244](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:244), [ms2_kernels.rs:362](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:362). Feature expectations use twin-produced selection and shared wavelength constants: [ms2_kernels.rs:431](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:431). The separate `filter_peaks` plus host-sort comparison supplies useful independent selection coverage, but only scale 0 and two fixtures: [ms2_kernels.rs:275](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:275).

4. **Adjoints are correct by inspection.** Selection reselects the upstream gradient; lookup sums every matching row in increasing order, correctly accumulating repeated ids and ignoring sentinel/out-of-range rows: [autograd/ms2.rs:24](/Users/ods/Documents/mamba-trainer/src/autograd/ms2.rs:24), [autograd/ms2.rs:40](/Users/ods/Documents/mamba-trainer/src/autograd/ms2.rs:40), [tensor/ops/ms2.rs:1118](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:1118).

   Finite differences perturb every table and multiplier element and use nonuniform weights, so they catch many wrong adjoints. **They cannot catch overwrite-instead-of-accumulate:** valid ids are unique in both backward tests. Evidence: [ms2_kernels.rs:563](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:563), [ms2_kernels.rs:588](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:588). Add repeated ids with distinct upstream gradients.

5. **Counters cover every direct data-moving client call found in `src`, exactly once.** Allocation/upload sites: [base.rs:80](/Users/ods/Documents/mamba-trainer/src/tensor/base.rs:80), [base.rs:97](/Users/ods/Documents/mamba-trainer/src/tensor/base.rs:97), [index.rs:42](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:42), [index.rs:60](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:60), [index.rs:544](/Users/ods/Documents/mamba-trainer/src/tensor/ops/index.rs:544), [backend.rs:926](/Users/ods/Documents/mamba-trainer/src/backend.rs:926), [backend.rs:938](/Users/ods/Documents/mamba-trainer/src/backend.rs:938), [ms2.rs:879](/Users/ods/Documents/mamba-trainer/src/tensor/ops/ms2.rs:879).

   Mixed-stream `read_handles` **does not double count**: it delegates each runtime count to `read_handle`; the outer path only counts step reads: [backend.rs:287](/Users/ods/Documents/mamba-trainer/src/backend.rs:287), [backend.rs:320](/Users/ods/Documents/mamba-trainer/src/backend.rs:320).

   `ms2_counters` contains one test in its own integration binary. Upload/allocation/read deltas checked there are exact; download bytes use a lower bound, and several deltas are omitted: [ms2_counters.rs:25](/Users/ods/Documents/mamba-trainer/tests/ms2_counters.rs:25), [ms2_counters.rs:57](/Users/ods/Documents/mamba-trainer/tests/ms2_counters.rs:57).

6. **Core memory formulas and block counts match.** Formula table `48R`, encoder ping-pong/projection terms, spectrum memory/KV, **two carry banks**, graph state, actions, atom memory and generation head scratch follow §6.2: [workspace.rs:167](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:167), [workspace.rs:206](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:206), [workspace.rs:234](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:234), [workspace.rs:264](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:264). Carry arithmetic reproduces 129/258 MiB.

   `block_parameter_count` matches actual projection, convolution, per-head parameters, B/C biases and norms allocated by mixer init: [workspace.rs:629](/Users/ods/Documents/mamba-trainer/src/models/ms2/workspace.rs:629), [mamba3.rs:259](/Users/ods/Documents/mamba-trainer/src/models/mamba3.rs:259). The real-block test is meaningful: parameter count traverses owned parameters, not this formula. It covers only V0’s configuration: [ms2_workspace.rs:137](/Users/ods/Documents/mamba-trainer/tests/ms2_workspace.rs:137), [module.rs:142](/Users/ods/Documents/mamba-trainer/src/nn/module.rs:142).

7. **Important test omissions.** No targeted coverage for oversized counts, malformed output shapes, zero-dimensional buckets, `n_keep>n_raw`, maximum finite intensities, exact intensity-floor equality, repeated lookup ids, poisoned upstream gradients, nonbinary validity values, or formula flags `[0,1,2]`. Fourier checks lack independent nonzero-phase expectations across all wavelengths: [ms2_kernels.rs:466](/Users/ods/Documents/mamba-trainer/tests/ms2_kernels.rs:466). Counters omit mixed-stream reads, metadata cache hit/miss deltas and direct embedding-backward uploads: [ms2_counters.rs:31](/Users/ods/Documents/mamba-trainer/tests/ms2_counters.rs:31). Workspace tests lack allocation reconciliation, optional block configurations and derived-width overflow cases: [ms2_workspace.rs:45](/Users/ods/Documents/mamba-trainer/tests/ms2_workspace.rs:45), [ms2_workspace.rs:91](/Users/ods/Documents/mamba-trainer/tests/ms2_workspace.rs:91).

## Not checked

- No cargo, compilation, kernel execution, CPU/GPU tests or profiling was run.
- Generated WGSL and backend numerical behaviour remain unverified.
- Exclusive-memory/WebGPU portability needs checking: summary binds stats twice. Native CubeCL defaults both bindings to read-write, but exclusive-memory mode preserves read/read-write visibility; using one mutable stats binding would avoid that distinction.
- Estimates were not reconciled with runtime allocations; full-model parameter counts cannot yet be checked against a complete MS2 model.