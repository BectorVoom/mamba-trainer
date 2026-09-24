# Entity kernel plan

This is an execution document. It is written so that it can be worked start to finish without reading the session that
produced it. Each task is self-contained: goal, the measurement that justifies it, files and line anchors, the kernel to
write, its adjoint, the traps, the test, and the command that proves it. Work the tasks **in the order given**. Each one
lands on its own and leaves the suite green.

**Subject.** The structured-observation path from `ENTITY_ENCODER_PLAN.md` (E1–E5, landed):
- the observation split (`src/rl/spec.rs`);
- the shared entity encoder and masked pooling (`src/nn/entity.rs`);
- the pointer head (`src/rl/heads.rs`).

Today every piece is composed from existing kernels: no new kernel and no hand-written backward. It is correct and
never reads back, but it spends launches on intermediate tensors that a fused on-device kernel would keep in registers.
The trainer is host-bound (each launch costs more than its arithmetic; see "Why launches"), so those launches are the
cost.

This plan replaces the four hot compositions with CubeCL kernels. Each one:
- has a hand-written adjoint;
- sits behind one runtime switch that restores the composed path;
- is held to that composed path by tests.

---

## Why: the measurement

`examples/profile_entity.rs` attributes every launch to the source line that issued it. Kaggriculture destination
shape: 4 globals, 100 tiles × 60 features, d_model 64, 2 layers, encoder `[64] → 48`, pointer hidden 48.
Backend wgpu (Metal), 64 envs; the behaviour-cloning (BC) step is 64 × 16.

```bash
cargo run --release --no-default-features --features wgpu --example profile_entity
```

| policy | rollout step | BC optimizer step |
|---|---|---|
| flat (`Linear(6004 → 64)`) | 55 | 391 |
| structured, additive pointer | **81** (+26) | **474** (+83) |
| structured, dot pointer | 77 (+22) | 456 (+65) |

Where the +26 of a rollout step goes (additive; counted from the composed code, and they agree with the tally deltas):

| stage | ops today | launches |
|---|---|---|
| split | obs split (1), per-set features/presence split (1) | 2 |
| zero empty slots | `features × presence` (broadcast) | 1 |
| encoder MLP | per layer: matmul, bias add, ReLU; last layer no ReLU | 5 |
| presence stats | `sum_dim`, `clamp_max`, `clamp_min`, `div` | 4 |
| mean pool | batched matmul `[B,T,1,N] @ [B,T,N,d]` | 1 |
| max pool | `mask_logits`, `max_dim`, `× any` | 3 |
| join | `cat` of globals and pools, then matmul + bias | 3 |
| pointer (additive) | `w_h` (2), `w_e` (1), broadcast add, ReLU, `v` matmul, `mask_logits` | 7 |
| minus the flat head and encoder it replaces | | −2 … |

In the BC step the composed adjoints cost more than the forward:
- `max_dim`'s backward alone is 7 launches (expand, sub, `eq_scalar`, `sum_dim`, expand, div, mul);
- the broadcast add in the pointer reduces its gradient back to `[B,T,1,H]` with a `reduce_grad_to`.

The tally shows this as +12 `reduce_op`, +17 flat elementwise, +7 `cat`, +5 `strided_copy` over flat.

### Why launches, and not arithmetic

See the memory note `mamba-trainer-is-host-bound`: a PPO round on wgpu was ~118 ms of host submission against 1–3 ms
of device time. A launch costs ~10 µs of host time whatever it computes; at these sizes nothing here is
bandwidth-bound. Broadcast metadata uploads are already cached (`backend::set_meta_cache`), so the launch count is the
right unit.

---

## 0. Before you start

```bash
cd /Users/ods/Documents/mamba-trainer
cargo test --release --no-default-features --features cpu --no-fail-fast > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|^error|FAILED" /tmp/t.log
```

The rules of `ROLLOUT_FUSION_PLAN.md` §0 apply:
- check the exit code, not grep;
- never build wgpu while cpu tests run (shared `target/`);
- never benchmark while anything else runs.

Two additions:
- **`--no-fail-fast`**, or the first failing binary hides every later one.
- **Watch the disk.** A release build with examples is ~16 GB in `target/`; the disk filled once during this work and
  the failure looked like a link error (`ld: write() failed, errno=28`). Check `df -h .` before a full run.

CubeCL references: `/Users/ods/Documents/cubecl_manual/manual/Cubecl/` (the path in `CLAUDE.md` is wrong).
- `03_kernel_fusion.md`
- `11_launch_overhead_and_transfers.md`
- `07_memory_coalescing.md`
- `08_atomic_contention.md` — why every adjoint below is written without atomics.

---

## Repo map for these tasks

| Path | What lives there |
|---|---|
| `src/rl/spec.rs:162` | `ObsSpec::split`: obs → globals + per-set `(features [B,T,N,F], presence [B,T,N,1])` |
| `src/nn/entity.rs:168` | `EntityEncoder::apply`: `features × presence`, then the MLP, then the slot embedding |
| `src/nn/entity.rs:280` | `Presence::new`: mean weights `[B,T,1,N]`, `any` `[B,T,1,1]` (off the tape) |
| `src/nn/entity.rs:305` | `pool_parts`: mean (matmul) and max (`mask_logits` → `max_dim` → `× any`) |
| `src/rl/policy.rs:430` | `InputStage::apply`: split, encode, pool, `cat`, `pool.proj` |
| `src/rl/heads.rs:205` | `PointerHead::apply`: additive/dot scores, presence mask, extras `cat` |
| `src/tensor/ops/fused.rs:2205` | `ssm_coefficients_kernel` + `_backward_kernel` + wrappers: **the template** (comptime flag, `launch_1d`, packed outputs) |
| `src/tensor/ops/fused.rs:1586` | `silu` / `silu_backward`: the smallest unary forward/adjoint pair |
| `src/autograd/ops.rs:1243` | `Var::ssm_coefficients`: how a fused op is recorded on the tape (`record`, `rule!`, `reduce_grad_to`) |
| `src/tensor/ops/movement.rs:501,522` | `fused_split_enabled` / `set_fused_split`: **the toggle pattern to copy** |
| `src/backend.rs:446-480` | `start_launch_tally` / `launch_tally`: per-line launch attribution |
| `examples/profile_entity.rs` | the baseline above; re-run after every task |
| `examples/bench_split.rs` | the interleaved in-process wall-clock A/B pattern |
| `tests/rl_entity.rs` | behaviour tests the fused path must keep passing unchanged |
| `tests/rl_entity_footprint.rs` | reads = 0 and a flat launch count; gets the exact counts pinned in K5 |

---

## Invariants that must survive every task

1. **The composed path stays and stays correct.** `set_fused_entity(false)` (K0) restores today's code exactly. Every
   test in `tests/rl_entity.rs` runs green in both modes.
2. **Forward parity.** Fused and composed forwards agree within 1e-6 relative error (f32), on CPU and wgpu. The masked
   positions (`f32::MIN`) must be identical.
3. **Gradient parity.** Every fused adjoint passes `check_grad` (central differences, tolerance `2e-2·(1+|n|)`). It also
   matches the composed adjoint within 1e-5, on tie-free data (see the max-pool trap).
4. **No host reads and no atomics** in any new kernel. Every adjoint below is written as a gather ("each output element
   loops over what feeds it"), never a scatter.
5. **Numerics in f32 inside the kernel.** Accumulate sums and maxima in `f32` whatever `E` is, so bf16/f16 storage
   (`set_matmul_precision`) does not change the reduction order's error.
6. **`numpy_ref` unchanged.** Semantics do not move, so `bindings/python/tests/test_entity.py` passes without edits.
7. **Flat policies untouched.** `train_ema_parity`'s golden fingerprints and the flat launch counts (55 / 391) do not
   change.

---

## Status

| ID | Task | Launches saved (rollout / BC step) | Status |
|---|---|---|---|
| K0 | Switch, A/B bench, parity harness | 0 / 0 | todo |
| K1 | `entity_prepare`: split + zeroing + presence stats | ~5 / ~6 | todo |
| K2 | `entity_pool`: mean + max + join, with adjoint | ~5 per set / ~15 per set | todo |
| K3 | `pointer_scores`: additive and dot scorer + mask + extras, with adjoint | ~4 / ~8 | todo |
| K4 | `bias_relu`: the encoder's hidden layers | 1 per hidden layer / ~2 | todo |
| K5 | Measure, pin the counts, write it down | – | todo |

Target after K1–K4 (additive, one set): rollout step **81 → ~66**, BC step **474 → ~440**. The structured step then
costs about 11 more launches than flat, for an encoder with 8× fewer parameters. The estimates are from reading the
compositions; K5 replaces them with the tally.

---

## K0 — switch, A/B bench, parity harness

- **Goal:** make every later task measurable and reversible before any kernel exists.
- **Do:**
  - `src/nn/entity.rs`: add `pub fn set_fused_entity(on: bool)` and a private `fused_entity_enabled()`.
    - Copy `movement.rs:501-530`: an atomic `i8` with `-1` = read the env var `MAMBA3_FUSED_ENTITY` once (`"0"` = off).
    - The default is on, but every call site still takes the composed path until its task lands.
  - Re-export it from `mamba3::nn`. Add `m3.set_fused_entity(on)` to the Python module next to `set_matmul_kernel`,
    plus its `.pyi` stub; Python/Rust parity is part of `docs/test_guidline.md`.
  - `examples/bench_entity.rs`: an interleaved in-process A/B, following `bench_split.rs`.
    - Alternate `set_fused_entity(true/false)` every iteration, over N=200 iterations each of a rollout step and a
      BC step at the profile shape.
    - Report the median per mode and the ratio.
    - Wall noise across runs is ±20%, which is why the two modes must be interleaved in one process.
  - `tests/rl_entity_parity.rs`: runs one structured policy (additive and dot, with extras and a slot embedding) in
    both modes, compares logits, values and every parameter's gradient (invariants 2, 3), on tie-free random data.
- **Test:** passes trivially (both modes are the same code for now). The harness is what K1–K4 plug into.

## K1 — `entity_prepare`: split, zeroing and presence statistics in one launch

- **Goal:** replace the per-set split, the `features × presence` zeroing and `Presence::new`'s four launches with one
  kernel per set. That is 6 launches per set down to 1.
- **Kernel** (`src/tensor/ops/entity.rs`, new module, registered in `tensor/ops/mod.rs`):
  - Inputs: the set's band of the observation as `[rows, N·(F+1)]` (`rows = B·T`), read in place from `obs` with a
    column offset. The obs split disappears too when every consumer takes offsets.
  - Outputs:
    - `features [rows, N, F]`, zeroed where presence is 0;
    - `mean_w [rows, N]` = `p / max(1, Σp)`;
    - `any [rows]` = `min(1, Σp)`;
    - `legal [rows, N]` = `p` (a contiguous copy the pointer's mask and the pool read).
  - One thread per `(row, n)` for the features, with the row's sum computed per row.
    - Simplest correct layout: one thread per row loops over `N` for `Σp` (N ≤ a few hundred), then writes the row.
    - Coalescing is irrelevant at this size; the launch is the cost.
- **Adjoint:** needed only when `obs` is traced (never in RL; `check_grad` does it).
  - `d_obs[band] = d_features × p` at the feature columns, 0 at the presence column.
  - One elementwise kernel. Presence gets no gradient, matching the composed path, where presence is a constant by
    construction (`Presence::new` is off the tape).
- **Wire:** `ObsSpec::split` returns the prepared parts when fused. `EntityEncoder::apply` then skips its multiply;
  add an `apply_prepared` that takes zeroed features. `Presence` gains a constructor from the kernel's outputs.
- **Traps:**
  - Globals still need to reach the join. Leave them as a view (`slice`) until K2's join writes them directly.
  - `presence` that is not exactly 0/1: the composed path treats any non-zero as present for masks and as a weight for
    the mean. Keep exactly that; the parity test must include a 0.5.
- **Test:** parity (K0 harness), `check_grad` through a traced `obs`, and an all-absent row (mean weights 0, any 0).

## K2 — `entity_pool`: mean and max into the joined buffer, with a gather adjoint

- **Goal:** replace, per set:
  - the mean matmul, `mask_logits`, `max_dim` and `× any`;
  - that set's share of the `cat`;
  - the composed adjoint (7 launches for `max_dim` alone, plus the mean matmul's two).

  That is about 5 forward and ~10 backward launches per set down to 1 + 1.
- **Kernel `entity_pool`:**
  - Inputs:
    - `e [rows, N, d]`;
    - `mean_w [rows, N]` and `any [rows]` (from K1);
    - the `joined [rows, W]` output buffer, a column offset and a comptime `kinds` mask (mean, max).
  - One thread per `(row, j)`, `j < d`. It loops over `n`, accumulating `Σ w_n e_nj` and the running max over present
    `n` together with its index, then writes:
    - `joined[row, off + j] = mean`;
    - `joined[row, off + d + j] = any ? max : 0`;
    - `argmax [rows, d]` as `u32`, saved for the adjoint (0 when the set is empty).
  - The first set's launch also copies the globals into `joined[:, 0..G]` (comptime flag). That removes the `cat` as
    well: K2 plus K1 turn "split, pool, cat" into one prepare and one pool launch per set.
- **Adjoint `entity_pool_backward`:**
  - One thread per `(row, n, j)`:
    `d_e = g_mean[row, j] · w[row, n] + (argmax[row, j] == n) · any[row] · g_max[row, j]`.
  - It is a gather: no atomics, one launch.
  - `g_mean`/`g_max` are read from the joined gradient at the set's offsets; the joined node owns the gradient buffer,
    so no split launch is needed.
- **Tape:** the joined buffer is one `Var` with one node whose rule calls every set's backward kernel and slices the
  globals' gradient. Mirror how `Var::split` stashes bands in a sink (`autograd/ops.rs:955`), but in reverse:
  one output, many inputs.
- **Traps:**
  - **Ties.** The composed `max_dim` shares the gradient among tied maxima; the argmax adjoint sends it all to the first.
    Both are valid subgradients, but they are not equal. The parity test must use tie-free data. Record the difference
    in the kernel's doc comment; do not emulate the sharing, which would cost a second pass.
  - **Empty set:** `argmax` must be a valid index (write 0) and `any = 0` must zero both the value and the gradient.
  - **Masked value:** do not reintroduce a finite `BIG`. Initialise the running max from the first *present* entry,
    so no sentinel can be outranked.
- **Test:**
  - parity forward and backward;
  - `check_grad` on `e`;
  - the existing permutation-invariance, mask-invariance and empty-set tests in both modes;
  - an entity set with `N = 1`.

## K3 — `pointer_scores`: the scorer, the mask and the extras in one launch

- **Goal:** replace, for the additive head, the broadcast add, ReLU, `v` matmul, reshape, `mask_logits` and the extras
  `cat`: 4–5 forward launches down to 1. It also stops storing the `[B,T,N,H]` pre-activation for the backward; the
  adjoint recomputes it.
- **Kernel `pointer_additive`:**
  - Inputs: `k = W_e e` `[rows, N, H]`, `q = W_h h + b` `[rows, H]`, `v [H]`, `legal [rows, N]`, and optional
    `extra [rows, K]`.
  - One thread per `(row, n)`:
    `logit = legal ? Σ_h v_h · relu(k[row,n,h] + q[row,h]) : F::min_value()`, written to `logits[row, n]`.
  - Threads `n ≥ N` copy `extra[row, n − N]`, so the output is `[rows, N + K]` with no `cat`.
- **Adjoint** (three launches, no atomics):
  1. `d_k[row, n, h] = g[row, n] · legal · v_h · [k + q > 0]`: one thread per `(row, n, h)`, which recomputes the
     pre-activation.
  2. `d_q[row, h] = Σ_n d_k[row, n, h]` and `dv_partial[row, h] = Σ_n g · legal · relu(k + q)`: one thread per
     `(row, h)` looping over `n` (a gather).
  3. `d_v = Σ_row dv_partial` (the existing `sum_dim`); `d_extra` is a view of `g[:, N..]`.
- **Dot variant `pointer_dot`:** forward `logit = legal ? Σ_d e[row,n,d] · qd[row,d] : MIN`, one thread per `(row, n)`.
  The adjoint is two gathers:
  - `d_e[row,n,d] = g·legal·qd[row,d]`;
  - `d_qd[row,d] = Σ_n g·legal·e[row,n,d]`.

  This replaces the batched matmul with `[.., d, 1]` operands, which is a poor shape for the tiled kernel anyway.
- **Traps:**
  - A masked position must produce exactly `f32::MIN` and exactly zero gradient into `k`, `q`, `v`, since the composed
    `mask_logits` rule multiplies by `legal`.
  - `legal` comes from K1's contiguous copy, not a reshape of a strided presence view.
- **Test:**
  - parity;
  - `check_grad` on `k`, `q`, `v` (and `e`, `qd` for dot);
  - the equivariance and empty-slot tests in both modes;
  - the BC "pick the largest" test (`tests/rl_entity.rs`) still reaches ≥ 0.99 in both modes.

## K4 — `bias_relu`: the encoder's hidden layers

- **Goal:** fold each hidden layer's bias add into its ReLU: 2 launches down to 1 per hidden layer, forward. The
  adjoint is `g · [y > 0]`, with the bias gradient summed by the existing reduction, so the backward count is
  unchanged but no pre-activation is stored.
- **Do:** `fused::bias_relu(x, bias)` forward + `bias_relu_backward`, recorded as one tape node over `(x, bias)`.
  Use it in `EntityEncoder::apply` for every layer but the last.
  - Do **not** change `Linear::apply` globally. The Mamba blocks' bias folding is `ROLLOUT_FUSION_PLAN.md`'s subject,
    with its own measurements.
- **Test:** parity, `check_grad` on `x` and `bias`, and a layer whose pre-activation has exact zeros. Match the
  composed `relu`'s gradient at 0, which is 0; check `Var::relu`'s rule and copy it.

## K5 — measure, pin, write down

- Re-run `examples/profile_entity.rs` and `examples/bench_entity.rs` on an idle machine (wgpu), and record both tables
  in this file:
  - launches per rollout and BC step, fused and composed;
  - median wall time per mode.
- Pin the fused rollout step's exact launch count in `tests/rl_entity_footprint.rs`, the way `rl_footprint.rs` pins
  its own. A later change that adds a launch should fail a test, not a benchmark someone may not run.
- Update `ENTITY_ENCODER_PLAN.md` ("Speed") and the memory note on rollout launch attribution with the new
  attribution.
- **Acceptance:**
  - every invariant holds on CPU and wgpu;
  - the fused rollout step is at most 70 launches at the profile shape;
  - the in-process A/B shows the fused BC step no slower than the composed one.

  If a kernel does not pay for itself in the A/B, leave it switched off by default and say so here, with the numbers.

---

## What is deliberately not in this plan

- **Multi-tensor AdamW.** The structured policy has 6 more parameter tensors than the flat one, and each costs an
  `adamw_step` and a `sum_squares_into` per optimizer step (+12 of the +83). That is a property of the optimizer, not
  of the entity path, and it helps every model. It needs its own plan.
- **Fusing the matmuls** (`W_e e` into the pool, `W_h h` into the pointer). They go through the tuned matmul kernels.
  A hand-written GEMM would lose on the GPU what it saves in launches.
- **Attention over entities.** That is `ENTITY_ENCODER_PLAN.md`'s phase 2; its kernels are a separate plan once it
  exists.
- **CUDA/ROCm-specific paths** (shared-memory tiling, warp shuffles for the row sums). Every kernel here is a
  plain `#[cube]` kernel that runs on the CPU, wgpu, CUDA and HIP runtimes alike, and at these sizes the launch, not the
  inner loop, is the cost.
