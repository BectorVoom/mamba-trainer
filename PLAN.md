# Training-speed implementation plan

> **Status (2026-08-16, same session):** Phase 1.1, 1.2, and part of Phase 2 (2.5,
> 2.1) are implemented, tested, and merged into the working tree (not yet
> committed to git). Phase 1.3, 1.4, 1.5 were investigated and deliberately
> skipped — see "What shipped" below for why. Phases 2.2–2.4, 2.6, and 3–4 remain
> as designed below; 3–4 need GPU hardware this session did not have.
>
> **What shipped:**
> - **1.1 — `Var::sum` adjoint.** Replaced the blocking `g.to_f32()[0]` readback
>   with `elemwise::expand(g, &shape)`, mirroring `sum_dim`'s adjoint. Verified via
>   a new `backend::read_count()`/`reset_read_count()` counter (mirrors
>   `launch_count()`): backward now does exactly the one documented embedding
>   read, full step does exactly the two intended reads (loss, grad-norm scale).
> - **1.2 — skip discarded scan state in training.** `ScanInputs` gained
>   `want_state: bool` (default `true`, `.with_state_output()`); `ssd_chunked`
>   and `mamba3_scan` return `Option` state, skipping `final_state`,
>   `last_outer_product`, and the rotation's `end_angle` when unwanted.
>   `Mamba3Mixer::apply_with_state` derives `want_state = cache.is_some()` — exact
>   and safe because the only two call sites are `apply()` (`cache: None`,
>   discards state) and `HybridLayer::apply_cached` (`cache` always `Some` via
>   `empty_cache()`). Cascaded to `Option<MixerCache>` through
>   `Mamba3Block::apply_with_state` and `HybridLayer::apply_cached`. Saved ~44
>   launches/step on the Phase-0 benchmark shape.
> - **1.3 (embedding backward), 1.5 (ignore_index)** — investigated and **skipped**.
>   Both host round-trips are covered by explicit code comments defending them as
>   deliberate (exact reproducibility / no float-atomic dependency for embedding;
>   similar reasoning for ignore_index). Overriding a documented trade-off needs a
>   stronger reason than "fewer launches."
> - **1.4 (causal-mask hoisting)** — investigated and **skipped**. The
>   "recomputed per rank² call" waste only bites MIMO (rank > 1) configs; the
>   benchmarked/default SISO config already builds it once per layer per step.
>   A correct fix needs a per-device cache (real new plumbing) or duplicated
>   padding arithmetic in the scan's most correctness-sensitive file — poor
>   risk/reward for the actually-exercised config.
> - **2.5 — `Var::matmul_nt`.** Added with adjoint `dA = matmul(g, b)`,
>   `dB = matmul_tn(g, a)`, built on the existing (already-tested) `mm::matmul_nt`/
>   `mm::matmul_tn` kernels. Verified against composed `a.matmul(&b.transpose()?)`
>   plus central differences (`tests/autograd.rs::matmul_nt_matches_transpose_then_matmul`).
>   Wired into 4 call sites: the scan's intra-chunk `cb` product, `last_outer_product`,
>   `mamba3_step`'s `u` computation, and the tied-embedding output head (this last
>   one was materialising a full `[vocab, d_model]` transpose every forward pass).
> - **2.1 — fused softplus.** New `fused::softplus`/`softplus_backward` kernel
>   (numerically stable `max(x,0) + ln(1+exp(-|x|))` forward, `sigmoid(x)`
>   backward) replacing 7 forward + ~8 backward composed launches with 1 each.
>   `Var::softplus_composed` kept as the test oracle
>   (`tests/autograd.rs::fused_silu_and_state_update_gradients`, extended):
>   verified equal to the composed form at ±40 (no overflow) and against central
>   differences. **A/B'd on the CPU benchmark and found wall-clock-neutral** (the
>   machine's noise floor here is ~±15%, larger than this fusion's effect) — the
>   real payoff is the launch-count reduction, which matters most on GPU where
>   each dispatch has a fixed ~9–13µs cost per the README's own methodology.
>
> **Verification:** all 76 tests green (`cargo test --release`); `train_lm`
> reproduces the README's exact trajectory (loss 3.19→0.0335, 97.7% held-out
> accuracy, identical greedy continuation); `generate` confirms parallel vs.
> incremental decode agree to 6e-8 (exercises the `Option<MixerCache>` refactor);
> `finetune_lora` and `vs_transformer` run correctly. CPU launch count on the
> Phase-0 shape (`SEQ=256 BATCH=2 LAYERS=4`): 1197 → 1093 (−8.7%), forward
> 405→337, backward 686→650.


Goal: reduce wall-clock time per optimizer step of `Mamba3Lm` training, measured by
`examples/bench_train.rs` (tokens/s, best and median), without changing what the
model computes in the default configuration — and, for the mixed-precision phases,
without losing convergence (`train_lm` must still reach ~97.7% held-out accuracy).

Ground truth this plan is built on (verified against source on 2026-08-16):

- Matmuls are ~60% of a realistic training step at 630–890 GFLOP/s in `f32`
  (README, "What that came to"). The stated next lever is bf16 + matrix cores.
- The launch map of one training step found waste that is *not* matmuls:
  a blocking readback inside every backward pass, per-layer work computed and
  discarded, host round-trips, and unfused elementwise chains in the scan.
- Validation environments: CPU runtime locally, CUDA via
  `notebooks/mamba3_cuda_colab.ipynb`, ROCm/wgpu on the benchmark machine only.

Every phase lands as an independent commit with a `bench_train` before/after and a
green `cargo test --release` (CPU). Phases are ordered so each is useful alone;
later phases do not depend on earlier ones except where noted.

---

## Phase 0 — Pin the baseline (half a day)

1. Record the reference numbers on the shapes that matter, so every later diff has
   a denominator:
   - CPU: `SEQ=256 BATCH=2 LAYERS=4 ITERS=5 cargo run --release --example bench_train`
     (today: 1197 launches/step — forward 405, backward 686; best 1.38 s).
   - CUDA (Colab): run the notebook's bench cell at the default 25.8 M shape.
   - Keep the per-phase launch counts printed by `bench_train`; they are the
     backend-independent regression signal for Phases 1–2.
2. Confirm `MAMBA3_TUNE_CHECK=1 cargo run --release --example bench_train` passes —
   it is the harness Phases 3–4 extend.

No code changes.

---

## Phase 1 — Remove per-step stalls and dead work (1–2 days)

These change *when* and *whether* work runs, not what is computed. All CPU-testable.

### 1.1 Device-side adjoint for `Var::sum` — kill the mid-backward sync

- Where: `src/autograd/ops.rs:666–676`. The rule does `g.to_f32()[0]` — a blocking
  device→host read inside **every backward pass** (the loss path is
  `mean()` → `sum()`, `src/train/loss.rs`), defeating the crate's own
  one-sync-per-step design (`src/train/trainer.rs`, "A step that never stops to look").
- Change: the gradient of `sum` is `g` broadcast to the input shape. Replace the
  readback + `Tensor::full` with `elemwise::expand(&g_r, &shape)` where
  `g_r = g.reshape(vec![1; shape.rank()])`. Same single full-size write as the
  fill it replaces, zero readbacks. Apply the same treatment to `mean()` if it
  carries its own scale.
- Tests: existing `tests/autograd.rs` central-difference coverage already exercises
  `sum`; add an assertion-free smoke check that `backward()` issues no reads
  (extend `backend.rs` with a read counter next to `launch_count` — cheap, and it
  makes this class of regression visible forever).

### 1.2 Don't compute the carry-out state that training throws away

- Where: `src/ssm/scan.rs` — `ssd_chunked` computes `final_state` (lines 266–275:
  slice, clamp, exp, mul, slice, add per call) and `mamba3_scan` computes
  `last_outer_product` (line 577; two slices + a **matmul against a materialised
  transpose**) plus the `end_angle` slice (line 484). `Mamba3Mixer::apply_with_state`
  (`src/models/mamba3.rs:441`) returns these in a `MixerCache` that the training
  forward (`mamba3.rs:433`) immediately discards — every layer, every step.
- Change:
  - Add `want_state: bool` to `ScanInputs` (builder: `.with_state_output(bool)`,
    default `true` so existing callers are untouched).
  - `ssd_chunked` gains the flag and returns `(Var, Option<Var>)`; when `false`,
    skip the boundary-state block. Blast radius: 6 callers
    (`src/ssm/scan.rs`, `src/ssm/mod.rs`), tests in `tests/ssm.rs`.
  - `mamba3_scan`: when `false`, skip `last_outer_product`, the `total_state`
    accumulation in the rank loop, and `end_angle`; `ScanOutput.state` becomes
    `Option<SsmState>`.
  - `apply_with_state` passes `want_state = cache.is_some()`; `forward` (training)
    passes `None` cache today, so training gets the fast path automatically.
    Decode/prefill (`forward_cached`) pass `Some` and are unchanged.
- Expected: ~10 launches + one matmul + one contiguous-axis (scalar-kernel)
  transpose per layer per step.
- Tests: `tests/ssm.rs` equality of `y` with the flag on/off; existing state tests
  keep covering the `true` path.

### 1.3 Embedding backward without the host round-trip

- Where: `src/tensor/ops/index.rs:191–253` (`scatter_add_rows`): reads all
  `batch*seq` ids to the host, sorts them, uploads 3 buffers — a hidden sync in
  every backward pass.
- Change: one device kernel that atomically adds each position's gradient row into
  the weight gradient (`atomic_add` per element of the row; contention is per
  token id and rows are wide, so conflicts are rare in practice — see manual
  `08_atomic_contention.md`). Gate on
  `client.properties()` float-atomics support; keep the host path as fallback for
  runtimes without it (CubeCL CPU runtime), selected the same way `launch_1d`
  already branches on plane support.
- Tests: `tests/autograd.rs` embedding gradient vs the host path on ids with heavy
  duplication (all-same, all-distinct, and a Zipf-ish mix) — determinism caveat:
  atomic float addition reorders; compare with a tolerance, and keep the host path
  as the reference oracle.

### 1.4 Hoist the causal chunk mask

- Where: `src/ssm/scan.rs:216` — `Tensor::strict_causal_mask(chunks)` is built on
  the host and uploaded on **every** `ssd_chunked` call (`rank²` per layer per step).
- Change: build it once in `mamba3_scan` before the rank loop and pass
  `&Var<R, E>` into `ssd_chunked` (parameter, not cache — no generics-hostile
  global state). Rank-1 models save the per-layer rebuild; MIMO saves `rank²−1`.
- Tests: covered by existing `tests/ssm.rs` equivalence.

### 1.5 `ignore_index` mask on device

- Where: `src/train/loss.rs:95–108` — with `ignore_index` set, all targets are read
  back per micro-batch and a mask is rebuilt on the host.
- Change: a small kernel `mask_ne_scalar(ids, ignore) -> Tensor` producing the 0/1
  mask on device; the count of kept tokens stays on device via the existing
  `sum`. Only matters for users of `ignore_index`, but it is another hidden sync.

**Phase 1 acceptance:** CPU launch count drops by ≥ 40/step on the Phase-0 shape;
`backward()` performs zero device reads; full test suite green; Colab notebook
suite green.

---

## Phase 2 — Fuse the measured chains (2–3 days)

Follow the crate's fusion discipline (README "Fused ops are the crate's one
deliberate exception…"): every fused op keeps a `_composed` twin, a fused-vs-composed
test, and a central-difference adjoint test.

### 2.1 Fused `softplus` (7 fwd + ~8 bwd launches → 1 + 1)

- Where: `src/autograd/ops.rs:1011` — expanded today as
  `abs, neg, exp, add_scalar, log, relu, add`; used per layer on `dt`
  (`src/models/mamba3.rs:385–387`).
- Change: `fused::softplus` forward (numerically stable form
  `max(x,0) + log1p(exp(-|x|))`) and adjoint `g * sigmoid(x)` as one kernel each,
  wired as `Var::softplus` with `Var::softplus_composed` kept public for the test.
  Tensors are `[b, t, heads]` — small, so this is a launch-count win (decode
  benefits too), not a bandwidth one.

### 2.2 Fused `exp(clamp(sub))` — the decay pattern (3–4 passes → 1)

- Where: the pattern `sub → clamp(LOG_DECAY_FLOOR, 0) → exp` (sometimes `→ mul`)
  appears four times in `ssd_chunked` (`src/ssm/scan.rs:200–204, 220–225, 250,
  266–270`) on `[b, chunks, chunk, heads]`-sized tensors, and `clamp`'s adjoint
  alone is 4 launches + 3 full-size intermediates (`src/autograd/ops.rs:601–612`).
- Change: `Var::exp_decay(a, b) = exp(clamp(a - b, FLOOR, 0.0))` with broadcasting
  on the inputs (the sites subtract broadcast pairs), one forward kernel, one
  adjoint kernel (`d/da = y * in_range`, `d/db = -y * in_range`, reduced to each
  parent's shape via the existing `reduce_grad_to`). This is the same move that
  turned the intra-chunk band into `Var::ssd_band`, applied to the three
  remaining decay sites (the fourth is inside the band already).
- Fold the adjacent multiply where profitable: site `:225` multiplies by the causal
  mask — after 1.4 the mask is an argument, so the kernel can take it as an
  optional third operand.

### 2.3 Fused cross-entropy forward/backward — delete the dense one-hot

- Where: `src/train/loss.rs:59` composes `log_softmax` (6+ vocab-sized passes,
  `src/autograd/ops.rs:1035`) with `take_along_last`, whose adjoint materialises a
  dense `[positions, vocab]` one-hot and multiplies it (`ops.rs:868–874`). At the
  bench shape that is several extra 64 MiB passes — the largest transient traffic
  in the step.
- Change:
  - Forward: one kernel per row — plane-strided max, plane-strided
    sum-of-exp (the plane-reduction shape already used by the fused norms,
    `src/tensor/ops/fused.rs`), then `loss[row] = log_sum_exp - logit[target]`.
    Per-unit fallback for the CPU runtime, mirroring `rms_norm`'s two variants.
  - Backward: `dlogits[row, v] = g * (softmax(row, v) - [v == target])` in one
    kernel; recompute softmax from saved `max` and `log_sum_exp` (two
    `[positions]` vectors) instead of saving a vocab-sized activation.
  - Wire as `Var::cross_entropy_fused(logits, ids)` used by
    `cross_entropy_with`; keep the composed path as the test oracle. `ignore_index`
    folds in as a per-row multiplier (combines with 1.5).
- Expected: removes ~4–6 full `[positions, vocab]` passes plus that allocation;
  also shrinks backward peak memory by one logits-sized buffer.

### 2.4 Fused trapezoid weights (8 launches → 1)

- Where: `src/ssm/scan.rs:452–455` — `g = λ·dt`, `next = (1−λ)·dt`,
  `shift_left` (fill + slice + 2 cat-writes), `w = g + f`.
- Change: one kernel over `[b, t, heads]` producing `g` and `w` packed as
  `[2, b*t*heads]` (the packed-output trick `ssd_band`'s coefficients already
  use), reading `dt[t+1], λ[t+1]` directly for the shift. Adjoint is one kernel
  scattering into `dλ`, `ddt`.

### 2.5 Transposed matmul at the `Var` level

- Where: forward code still materialises transposes that the kernels can read in
  place: `src/ssm/scan.rs:191` (`b_flat.transpose()` — a contiguous-axis swap, i.e.
  the *scalar* copy kernel), `scan.rs:605`, and the tied-embeddings head
  (`src/models/lm.rs:305–314`). `matmul_nt`/`matmul_tn` exist
  (`src/tensor/ops/matmul.rs:1518,1528`) but are reachable only from backward rules.
- Change: `Var::matmul_nt(&self, rhs)` (and `_tn`) recording the right adjoints
  (`dA = G B`, `dB = Gᵀ A` for `nt` — note the adjoint of a transposed product needs
  no new kernels, only the existing four entry points). Switch the three sites.
- Expected: deletes one scalar-kernel strided copy per `ssd_chunked` call — the
  worst copy shape in the step.

### 2.6 Optional, measure first: meta-buffer cache for broadcast ops

- Where: `src/tensor/ops/elemwise.rs:440`, `src/tensor/ops/movement.rs:154` — every
  broadcasting binary op / strided copy uploads a fresh shape-metadata buffer
  (hundreds of tiny H2D copies per step).
- Only do this if a Colab profile shows submission overhead worth having: cache
  `(shape, strides) → Handle` in a small LRU on the `Device`. Skip if the profile
  says it is noise.

**Phase 2 acceptance:** each fused op has fused-vs-composed and central-difference
tests; CPU launch count for the Phase-0 shape drops below ~950/step; `train_lm`
end-to-end output unchanged (same losses to f32 tolerance, same 97.7%).

---

## Phase 3 — bf16 mixed-precision matmul mode (3–5 days)

The big lever. Design constraints discovered in the code:

- The crate promises "every kernel computes the same product to within
  floating-point associativity" (`src/tensor/ops/matmul.rs:56`). bf16 staging
  breaks that contract, so mixed precision must be an **explicit mode**, not a
  tuner candidate that can silently win: `set_matmul_precision(Precision::F32 |
  Bf16)` beside `set_default_kernel`, plus a `MAMBA3_MATMUL_PRECISION` env read in
  the examples. Default stays `F32`.
- Weights and gradients stay `f32` end-to-end (master weights are what the
  optimizer already holds; AdamW kernel untouched). Only matmul *inputs* are
  rounded to bf16; accumulation is f32. This is the PyTorch-autocast recipe and
  needs no loss scaling.

### 3.1 Probe and plumbing

- `elemwise::cast<E1, E2>` kernel (`Tensor<R, E1> -> Tensor<R, E2>`); a
  10-line test instantiating `half::bf16` on the CPU runtime answers the open
  question of whether the MLIR backend compiles bf16 at all. If it does not:
  CPU tests emulate rounding with a `round_to_bf16` f32 kernel (bit-mask the
  mantissa), and real-bf16 correctness moves to the Colab notebook. Do this probe
  **first**; it decides the test strategy for the whole phase.

### 3.2 The mixed kernel

- New variant of `matmul_block_tiled_kernel` (and `_t_kernel`) generic over a
  storage element `ES` and accumulator `EA = f32`
  (`src/tensor/ops/matmul.rs:296–650`): global loads and shared tiles in `ES`
  (halving both global traffic and shared-memory footprint — the shape analysis in
  the README says the scan's 64³ batched products are memory-bound at ~10.7
  FLOP/byte, so bandwidth is exactly their ceiling), `acc`, `a_reg`, `b_reg`
  widened to f32 at the FMA. The prefetch/double-buffer structure is untouched.
- Entry path: in `Plan`/`launch_plan` (`matmul.rs:947,1176`), when the mode is
  `Bf16` and both operands are f32, cast each operand once (3.1's kernel) and
  launch the mixed kernel. The tuner (`tuned_plan`, `matmul.rs:1252`) tunes
  *within* the mode as it does today.
- `MAMBA3_TUNE_CHECK` learns a per-mode tolerance: bf16 candidates verify against
  the *bf16-rounded* simple product (cast inputs, run `Simple`), not the f32 one,
  keeping the check meaningful instead of loosening it to uselessness.

### 3.3 Avoid re-casting weights every matmul

- Activations are cast at use (they are consumed once). Weights are used twice per
  step (forward + weight-adjoint), and are unchanged within a step: cache the bf16
  copy on the `Param` (a `RefCell<Option<Tensor<R, bf16>>>` beside the value,
  invalidated by the optimizer's write — `Param` is already the shared-handle
  interior-mutable cell, so the hook point exists). Measure; if the win is small,
  drop the cache and keep the code simple.

### 3.4 Numerics gate

- `train_lm` from the same seed with the mode on: loss curve overlay and held-out
  accuracy must stay ≥ 97% (CPU if 3.1 says bf16 compiles there; otherwise the
  emulated-rounding build locally + real run in the Colab notebook).
- `tests/model.rs` gets a bf16-mode variant asserting relative error of one
  forward pass vs f32 stays within bf16 bounds (~1e-2 relative on logits).

**Phase 3 acceptance:** mode off → bit-identical behaviour to Phase 2; mode on →
Colab bench shows the matmul row improve (expectation: 1.2–1.5× on the memory-bound
shapes; compute-bound projections move less until Phase 4), `train_lm` converges.

---

## Phase 4 — Matrix cores: CMMA tile path (3–5 days, GPU-validated)

- Add a `BlockTiled`-shaped kernel whose inner tile is `cubecl::cmma`
  (16×16×16 fragments, `ES ∈ {f16, bf16}` inputs, f32 accumulate), as a tuner
  candidate **within** the bf16/f16 mode only.
- Gate on `client.properties()` CMMA feature support for the exact
  (in, out, shape) combination; runtimes without it (CPU, wgpu today) never see
  the candidate. Support both `f16` and `bf16` element types through the same
  generic: Colab's T4 only has f16 tensor cores, the ROCm target (gfx1151 WMMA)
  and A100/L4 have bf16 — so f16 is what makes the path *testable* on free Colab.
- The manual's staging guidance applies unchanged
  (`Staging_Mitigation_via_Asynchronous_Loads_and_Double_Buffering_in_CubeCL.md`);
  the fragment-store shared-memory bounce described in
  "State-of-the-Art Multiplatform Matrix Multiplication Kernels.md" is the known
  cost on the output side.
- Validation: `MAMBA3_TUNE_CHECK` (per-mode tolerance from 3.2) on Colab across
  the full `bench_train` shape repertoire; `bench_matmul` before/after per shape.
- This is where the README's "3.4 TFLOP/s is where the real ceiling is" gets
  attacked; a 2× on the compute-bound 60% of the step is the realistic prize.

---

## Phase 5 — Wrap-up

1. Re-run the full matrix: CPU suite, Colab CUDA suite + bench, and hand the
   ROCm/wgpu runs to the benchmark machine (`bench/compare.sh` vs PyTorch —
   including `--dtype bf16`, which is the honest comparison once Phase 3 lands).
2. Update the README's performance chronicle with the measured tables, in its
   established before/after format.
3. `cargo run --release --example train_lm` end-to-end check: same seed, ≥ 97%.

## Sequencing and risk register

| Risk | Phase | Mitigation |
|---|---|---|
| CubeCL CPU runtime lacks bf16 | 3 | 3.1 probe first; emulated-rounding fallback for local tests, Colab for real bf16 |
| Float atomics unavailable/nondeterministic | 1.3 | capability gate + host fallback kept; tolerance-based test |
| bf16 hurts convergence | 3 | f32 accumulate + f32 master weights; gate on train_lm curve; mode stays opt-in |
| CMMA API mismatch on HIP (WMMA) vs CUDA | 4 | candidate is per-device tuner-selected; TUNE_CHECK rejects wrong results before they can win |
| Fusion adjoint bugs | 2 | composed twins + central differences, per crate convention |
| Tuner picks a stale plan after kernel changes | 3–4 | tuner cache is per-process; no persistence to invalidate |

Estimated end state: Phases 1–2 mostly serve latency/small-model and memory
traffic (and decode gets the softplus/CE fusions free); Phase 3–4 target the 60%
of the step that is matmul, where a 1.5–2× overall step-time improvement on the
ROCm/CUDA targets is the realistic goal.
