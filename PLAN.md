# Training-speed implementation plan

Goal: reduce wall-clock time per optimizer step of `Mamba3Lm` training, measured by
`examples/bench_train.rs` (tokens/s, best and median), without changing what the
model computes in the default configuration — and, for the mixed-precision work,
without losing convergence (`train_lm` must still reach ~97.7% held-out accuracy).

> **Status (2026-08-17):** Phases 0–3 are done. Phases 1.1/1.2/2.1/2.5 are on
> `main` (commit `15cdbd6`); Phases 2.2/2.3/2.4 and 3.1/3.2/3.4 are implemented,
> tested and sitting in the working tree, uncommitted. This file now records the
> completed work compactly and plans the remainder — GPU validation, the CMMA
> tile path, the measure-first follow-ups, and the wrap-up — in detail.

---

## Part I — record of what is done

| Item | What | Where it lives | Evidence |
|---|---|---|---|
| 1.1 | `Var::sum` adjoint without the mid-backward readback | `src/autograd/ops.rs` | `read_count()` = exactly 2 reads/step |
| 1.2 | `want_state` skips the discarded boundary state | `src/ssm/scan.rs`, `Option<MixerCache>` | −44 launches/step; decode agrees to 6e-8 |
| 2.1 | fused `softplus` | `fused.rs`, `Var::softplus` | fused-vs-composed + central differences |
| 2.5 | `Var::matmul_nt` at 4 sites incl. tied head | `ops.rs`, callers | `matmul_nt_matches_transpose_then_matmul` |
| 2.2 | `Var::exp_decay` — `exp(clamp(a−b,floor,0))·m` fused, 3-operand broadcast, optional `b`/`m`, wired into all 5 decay sites of `ssd_chunked` | `fused.rs`, `ops.rs`, `scan.rs` | `fused_exp_decay_matches_composed_and_differentiates` |
| 2.4 | `Var::trapezoid_weights` — one launch, two outputs via the `split()` sink; one-launch adjoint | `fused.rs`, `ops.rs`, `scan.rs` | `fused_trapezoid_weights_match_composed_and_differentiate` |
| 2.3 | `Var::cross_entropy_rows` — plane-per-row fused CE, label smoothing folded in, backward rebuilds softmax from a saved `[rows]` log-sum-exp; the dense one-hot is gone | `fused.rs`, `ops.rs`, `loss.rs` (+ `cross_entropy_per_token_composed` kept as oracle) | `fused_cross_entropy_matches_composed_and_differentiates` |
| 3.1 | probe: the CubeCL CPU (MLIR) runtime compiles and runs **real bf16**, so the whole precision phase is correctness-testable locally | `tests/bf16.rs` | 3 tests green |
| 3.2 | bf16 mixed-precision matmul mode: all 7 kernels generic over storage `ES` + accumulator `F` (operands and shared tiles in `ES`, widened at the FMA; pure path `ES = F` is bit-identical); `set_matmul_precision(Bf16)` / `MAMBA3_MATMUL_PRECISION=bf16`; one cast per operand in `matmul_3d_t`; tune cache keyed by `(ES, E)` dtypes; `MAMBA3_TUNE_CHECK` compares within the mode | `matmul.rs`, `elemwise::cast`, examples | `bf16_mixed_precision_mode` forces every reachable kernel against a host bf16-rounded reference |
| 3.4 | numerics gate: `train_lm` under the mode reproduces the f32 trajectory to 4 decimals (3.1907 → 0.0335, 97.7%, identical continuation) | — | run log |

Deliberately skipped, with reasons that still hold: **1.3/1.5** (embedding backward
and `ignore_index` host round-trips are defended by explicit code comments —
reproducibility, no float-atomic dependency); **1.4** (causal-mask hoisting only
bites MIMO configs the benchmark never runs).

**Measured** (CPU, `SEQ=256 BATCH=2 LAYERS=4`): launches **1197 → 1093 → 911**/step
(forward 337→272, backward 650→533) — past the Phase-2 target of <950. f32
wall-clock unchanged with the mode off; bf16 mode measured ~501 vs 390 tokens/s
best-of-5 — plausible (halved operand bytes on memory-bound products) but near
this machine's ±15% noise floor. 82 tests green; `train_lm` / `generate` /
`finetune_lora` reproduce their README numbers in both modes.

Also in the working tree from a parallel session, kept: `weight_grad` reuse in
`matmul_nt`'s adjoint (tied head no longer materialises per-sequence gradients),
a decode-cache fix in `hybrid.rs` (zeroed cache instead of `None`, so decode
always gets its state back), and doc touch-ups.

---

## Part II — remaining work, in detail

Ordering: A (land) → B (GPU validation) → C (CMMA) → D (measure-first
follow-ups, informed by B) → E (comparison + docs). B before C because C's
tuner-candidate design assumes the mixed mode is confirmed sound on CUDA, and
because B's profile decides whether D is worth doing at all.

### Phase A — land the working tree (when the user asks to commit)

Nothing here runs; it is the commit split so each change stays reviewable and
revertable. Suggested order and messages:

1. *(parallel session's work, first — it is independent)*
   `Sum the tied head's gradient over the batch and resume decode from a zeroed cache`
   — `src/autograd/ops.rs` (matmul_nt adjoint via `weight_grad`),
   `src/models/hybrid.rs`, `src/backend.rs` doc fix, the matching
   `tests/autograd.rs` shared-rhs test, README/PLAN doc touches.
2. `Fuse the scan's decay chains, the trapezoid weights, and cross-entropy`
   — `fused.rs` (three new sections), `ops.rs` (`exp_decay`,
   `trapezoid_weights`, `cross_entropy_rows` + composed twins), `scan.rs`
   rewiring, `loss.rs`/`train/mod.rs`, the three new `tests/autograd.rs` tests.
   Body cites the launch counts: 1093 → 911/step.
3. `Add an opt-in bf16 mixed-precision matmul mode`
   — `matmul.rs` (kernel genericization, `MatmulPrecision`, dtype-keyed tuner),
   `elemwise::cast`, example env wiring, `tests/bf16.rs`.
4. `Colab: exercise the bf16 mode on CUDA` — `notebooks/mamba3_cuda_colab.ipynb`.

`.codegraph/` stays untracked (add to `.gitignore` if it bothers `git status`).

### Phase B — GPU validation on Colab (T4) — the next session's first move

Run `notebooks/mamba3_cuda_colab.ipynb` top to bottom (it clones the pushed
repo, so Phase A must be pushed first). What each cell must show, and what to
do if it does not:

1. **Full suite** — all 8 binaries `ok`, 82 tests. This is the first time the
   *plane* variants run at all: `cross_entropy_rows_plane_kernel` and
   `plane_per_row` are unreachable on CPU, as is `Plan::PlaneDot`'s mixed
   instantiation. A CE failure here with CPU green points at the plane kernel's
   lane/guard logic, not at the maths — compare against the unit variant by
   temporarily forcing `plane_per_row` to `None`.
2. **train_lm / generate / finetune_lora / vision** — same acceptance as ever:
   97.7%, ~1e-7 parallel-vs-incremental agreement, LoRA merge behaviour.
3. **Tuner cell, f32** — `MAMBA3_TUNE_CHECK=1` must not abort. Record the
   `tune (…) -> Plan` lines and the steady-state tokens/s row; this is the
   "before" column for every later comparison.
4. **New bf16 cell** — three gates in one cell: TUNE_CHECK under the mode
   (every mixed candidate vs the mixed Simple kernel — tolerance is unchanged
   because both read the same rounded operands), steady-state tokens/s under
   the mode, and `train_lm` convergence under the mode (~97.7%). Record the
   f32-vs-bf16 tokens/s delta — **this is the number the whole phase was
   for.** Expectation from the shape analysis: memory-bound scan matmuls
   (~10.7 FLOP/byte at 64³) improve toward 1.5–2×; compute-bound projections
   move less until Phase C. If the delta is ~0, profile whether the per-call
   `cast` launches are eating the win — that promotes Phase D.1 from
   "measure-first" to "do".
5. Failure triage for the mixed kernels: a wrong-product abort on CUDA with
   CPU green most likely means a vector-width mismatch between the `bf16`
   input arrays and the `f32` output at some `line` the CPU runtime never
   picks — check `line_for` in `launch_matmul` against the T4's supported
   widths for both dtypes before suspecting the kernels.

Deliverables: the numbers pasted into this file under a "Measured on T4"
heading, and the decision for Phase D (do / skip).

### Phase C — Phase 4: the CMMA/WMMA tile path (3–5 days, GPU-validated)

The compute-bound 60% of the step is matmul at 630–890 GFLOP/s f32; the
device's matrix cores run ~3.4 TFLOP/s. This phase makes the tensor cores a
*tuner candidate inside the reduced-precision modes only*.

**C.1 — an f16 mode arm (half a day).** The T4 — the only free Colab GPU — has
f16 tensor cores but no bf16 ones, so f16 is what makes this phase *testable*.
Add `MatmulPrecision::F16` as a third arm: the dispatcher in `matmul_3d_t`
gains one `cast::<R, E, half::f16>` branch, the env accepts `f16`, and
everything downstream is already generic over `ES`. Gate on convergence the
same way (`train_lm` under f16 on Colab; f16's narrow exponent is the risk —
if it diverges, the mode still serves as the CMMA test vehicle and bf16 stays
the training recommendation on Ampere+/RDNA).

**C.2 — capability gate.** CubeCL exposes exactly what is needed:
`client.properties().features.cmma` is a `BTreeSet<MmaConfig>` with per-combination
`(a_type, b_type, cd_type, m, n, k)` entries, registered per-arch by the CUDA/HIP
runtimes. The candidate is offered only when the mode's `ES` is `f16`/`bf16` **and**
`features.cmma` contains `(ES, ES, f32, 16, 16, 16)`. CPU and wgpu never see it;
no runtime probing, no `#[cfg]`.

**C.3 — the kernel.** A new `Plan::Cmma(BlockShape)` variant whose kernel keeps
the block-tiled skeleton (same staging assignment, same prefetch double-buffer,
same bounds-guarded zero-fill — which is also what keeps fragments full at the
tails) and replaces the register inner loop with fragments:

- One *plane* owns one 16×16 output tile: a `bm×bn` block holds
  `(bm/16)·(bn/16)` planes; `CubeDim` is that times the plane width. Start with
  `bm=64, bn=64, bk=16` (16 planes ≈ 512 units on CUDA) and let the tuner also
  try `128×64×16`.
- Stage `sa` **untransposed** for this kernel (`[row][kk]`, row-major) — the
  transposed layout the scalar kernel wants defeats `cmma::load`'s stride
  contract. `sb` stays `[kk][col]` row-major. Shared-memory padding is dropped
  here: fragment loads are cooperative and the bank-conflict analysis that
  motivated the pad does not apply to them; measure before re-adding.
- Inner loop per `bk` step: `cmma::load` A and B fragments from shared slices
  with the tile stride, `cmma::execute::<ES, ES, f32, f32>` into a
  per-plane accumulator fragment kept across the whole `k` walk.
- Output: `cmma::store` to a shared scratch, then units re-read it and write
  vectors to global — the "fragment-store bounce" the manual documents as the
  unavoidable cost on the way out (State-of-the-Art Multiplatform Matrix
  Multiplication Kernels.md).
- No `_t` variant in the first cut: transposed problems keep the existing
  `BlockT` candidates; the tuner decides per shape. (The scan's `cb` product is
  `matmul_nt`, so if the tuner tables show it stuck on `BlockT`, a follow-up
  `CmmaT` that loads B fragments column-major is the designed extension —
  `cmma::load_with_layout` exists for exactly this.)

**C.4 — validation.** `MAMBA3_TUNE_CHECK=1` on Colab under `f16` covers every
shape `bench_train` issues with real operands (the reference is the mixed
Simple kernel — same rounded inputs, so the tolerance still means something).
Add one capability-gated test to `tests/bf16.rs` that no-ops when
`features.cmma` is empty (CPU) and on a GPU forces the CMMA plan against the
host-rounded reference — same pattern as the existing forced-kernel loop.
`bench_matmul` before/after per shape for the README table.

**Acceptance:** tune tables show `Cmma` winning the compute-bound projection
shapes; the matmul row of the step moves toward the TFLOP/s range; nothing
regresses on shapes where `Cmma` loses (the tuner simply keeps the old winner);
`train_lm` still converges under the mode used.

### Phase D — measure-first follow-ups (only what Phase B's profile justifies)

**D.1 — bf16 weight cache on `Param` (skip unless the cast launches show up).**
The mode casts every matmul operand per call; weights are unchanged within a
step and appear in two products each (forward, and `dA = G·Wᵀ`). Design:
`Param` (`src/nn/param.rs` — already the shared-handle interior-mutable cell)
gains a `Rc<RefCell<Option<Tensor<R, half::bf16>>>>` beside `value`;
`Param::bf16_view()` fills it lazily via `elemwise::cast` and every mutation of
`value` (the optimizer's `set`, LoRA merges, quantizer writes — audit all
`borrow_mut` sites) clears it. Consumers are the `Var`-level call sites that
read weights into matmuls (`Linear`, the tied head), not `matmul_3d_t` — the
matmul layer sees raw tensors and cannot know what is a weight. Numerics note:
this changes *nothing* (the cast is deterministic), so the only test needed is
cache-invalidation-on-update. Only build it if Phase B measured the casts as
real cost; on the CPU bench shape they were +174 launches/step.

**D.2 — meta-buffer LRU for broadcast ops (same bar).** Every broadcasting
binary op and strided copy uploads a fresh `[rank, shape, strides]` buffer
(`elemwise.rs`, `movement.rs`, now also `exp_decay`). If a Colab profile shows
submission overhead worth having: a small `RefCell<HashMap<Vec<u32>, Handle>>`
on the `Device` (per-device, capped at a few hundred entries, keyed by the
packed meta contents). Skip if it is noise, exactly as originally planned.

### Phase E — comparison and documentation wrap-up (Phase 5, updated)

1. **PyTorch comparison, honest edition.** `bench/compare.sh` alternates this
   crate and `bench/torch_mamba3.py` on the same shape. Run it on the CUDA
   Colab (`PYTHON` pointed at Colab's torch, `RUST_BIN` at the cuda-feature
   build) in three configurations: f32 vs f32, **bf16 mode vs
   `TORCH_ARGS=--dtype bf16`** (the honest comparison the plan promised), and —
   once C lands — the reduced mode with CMMA in the tuner. The ROCm benchmark
   machine rerun is optional and owner-driven; nothing in this plan depends on
   it.
2. **README performance chronicle.** One new section in the established
   before/after voice: the Phase-2 fusion round (launch table 1197→1093→911,
   what each fusion deleted), then "the bf16 mode" with the measured T4 table
   and the tuner-winner tables per mode, then CMMA when it lands. Update the
   backends/test-count lines (82 tests), and document
   `MAMBA3_MATMUL_PRECISION` next to `MAMBA3_TUNE_CHECK`/`MAMBA3_TUNE_LOG`.
3. **Final gates re-run** (the same three every phase used): full CPU suite;
   Colab suite + bench; `train_lm` ≥97% from the same seed in every shipped
   mode. Update this file's status block and the project memory.

---

## Sequencing and risk register

| Risk | Phase | Mitigation |
|---|---|---|
| Plane kernels (CE, PlaneDot) wrong on CUDA — first real exercise | B | unit-variant fallback comparison; suite runs them under real training shapes |
| Mixed-kernel vector width unsupported for one dtype on some backend | B | `line_for` already takes the min across both dtypes; T4 run confirms |
| casts eat the bf16 win on GPU | B→D.1 | measured decision; weight cache design ready |
| f16 hurts convergence (no loss scaling) | C.1 | f16 stays a test vehicle; bf16 is the training mode; gate on train_lm |
| CMMA layout/stride mismatch (opaque fragment mapping) | C.3 | untransposed `sa` staging; TUNE_CHECK rejects wrong results before the tuner can pick them |
| WMMA (HIP) diverges from CUDA cmma semantics | C | candidate is per-device tuner-selected; capability set is per-arch; ROCm run optional |
| Fragment-store bounce erases the gain on small `n` | C | it is one candidate among several; the tuner keeps `BlockV`/`BlockT` where they win |
| Param cache serves a stale bf16 weight | D.1 | invalidate in every `Param` mutator; test asserts post-update freshness |

Estimated end state, unchanged from the original plan but now two phases
closer: Phases 1–2 bought latency and memory traffic (911 launches/step, the
one-hot gone); Phase 3 halves matmul operand bytes on the memory-bound 40%;
Phase C attacks the compute-bound 60% with the matrix cores, where a 1.5–2×
overall step-time improvement on the CUDA/ROCm targets remains the realistic
prize.
