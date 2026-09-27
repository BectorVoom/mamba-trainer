# GPU kernel optimisation plan: fewer memory passes per training step

This is an execution document. It is written so that it can be worked start to finish **without reading the session
that produced it**, by an implementer who follows instructions literally. Every task gives the goal, the files, the
exact API (with code skeletons), the traps, the test, the command that proves it, and a "done when" line. Work the
tasks **in the order given**. Each task lands on its own and leaves the suite green (`cargo test --features cpu` and
`cargo test --features vulkan`). Do not skip a measurement gate. Do not start a task whose gate number you cannot
measure.

Written against commit `91df9ea` ("fix entity model bug"). Line anchors (`file:line`) are from that commit; if a line
moved, search for the quoted symbol.

---

## 0. Why this plan exists

### 0.1 The measurement (2026-09-27, AMD Radeon 860M, gfx1151, an integrated GPU with no memory of its own)

The Kaggriculture entity model (`d_model` 128, 3 context + 3 decoder Mamba-3 layers, 100 context entities + 20
queries, K = 3 steps) was profiled with `examples/profile_entity_model` built with `--features vulkan`
(`MAMBA3_ENTITY_STEPS=4`, see trap T1) and from Python (`mamba3_rl` wheel, same GPU).

| One training step (forward + backward + AdamW) | launches / step | batch 8 | batch 32 | batch 128 |
|---|---|---|---|---|
| composed path (`MAMBA3_FUSED_ENTITY_MODEL=0`) | ≈ 2,300 | 0.38 s | – | 3.33 s |
| **fused path (default)** | ≈ 2,200 | 0.31 s | 1.05 s | **3.32 s** |
| Python `queue_train_step`, same step | 2,462 | – | 1.08 s | 5.5 s |
| PyTorch 2.13 ROCm, a 4.1 M-parameter Transformer doing the same job, bf16 | ≈ 300 | – | – | **0.56–0.85 s** |

Facts that decide what to optimise:

1. **Step time is proportional to the batch** (16× batch → 10.7× time) with a floor of ≈ 0.15 s. So the step is bound
   by kernel *execution*, not by dispatch overhead and not by host reads (reads/step = 0.1).
2. **Halving `d_model` (128 → 64) does not change the step time** (5.3 s vs 5.4 s at batch 128 from Python), and f16
   matrix products do not help (11.0 s vs 10.5 s at batch 256). **Matrix products are not the cost.**
3. The launches per step are dominated by small kernels that each stream a whole activation tensor through memory.
   Per composed step at batch 32: `flat_launch` elementwise 964 (`src/tensor/ops/elemwise.rs:86`), reductions 720
   (`src/tensor/ops/reduce.rs:217`, the two-pass split), `strided_copy` 570 (`src/tensor/ops/movement.rs:155`, behind
   `permute`, `reshape` of non-contiguous tensors, `slice`, `split`), `cat` 190 (`movement.rs:672`), `clamp` 180
   (`elemwise.rs:506`), `expand` 80 (`elemwise.rs:637`), per-parameter AdamW 258 (`src/tensor/ops/fused.rs:87`) and
   per-parameter grad-norm partial sums 258 (`fused.rs:2561`), matmuls 234 + 134, scan 144.
4. The existing "fused entity model" switch removes only ≈ 5% of the launches: the K-kernels of
   `ENTITY_MODEL_PLAN.md` fused the batch gather, the loss and the embedding join; **the Mamba-3 mixer and the chunked
   scan (`src/ssm/scan.rs:141` `ssd_chunked`) still run one kernel per tensor op.**
5. On an integrated GPU every pass over an activation tensor costs DDR bandwidth shared with the CPU. At batch 128 an
   activation is 128 × 120 × 128 × 4 B ≈ 7.9 MB, the scan's per-chunk tensors are larger; ≈ 2,200 kernels × 2–3 tensors
   each is tens of GB per step, which at the 15–50 GB/s this iGPU sees is the 1–3 s measured.

Full notes and the raw profiler output: the Kaggriculture repo,
`experiments/kobayashi/exp-planner053_dsm_task_planner/runs/20260927_dagger/mamba_trainer_speed.md` and
`profile_entity_model_b32.log`.

### 0.2 The target

| Gate | Fused step, batch 128, vulkan, this machine | Launches / step |
|---|---|---|
| G0 (now) | 3.32 s | ≈ 2,200 |
| G1 after M0 + K7 (optimizer) | ≤ 3.0 s | ≤ 1,700 |
| G2 after K8 (mixer chains) | ≤ 2.4 s | ≤ 1,300 |
| G3 after K9 (reductions) | ≤ 2.0 s | ≤ 1,000 |
| G4 after K10 (bf16 activations) | ≤ 1.3 s | ≤ 1,000 |
| G5 after K11 (Python path) | Python step ≤ 1.1 × Rust step | – |
| G6 after K12 (scan layouts, optional) | ≤ 1.0 s | ≤ 700 |

Every gate is measured with `bench/entity_step.sh` (M0.3) and written into the status table in §4. A task that does
not reach its gate is **kept only if it reaches at least half of the promised gain and its tests pass**; otherwise it
is reverted (`git revert`), the number is recorded in the status table, and the plan moves on.

### 0.3 Rules

- Every kernel change ships with: a parity test against the composed (unfused) computation on **both** `cpu` and
  `vulkan` (`docs/test_guidline.md`), a gradient check where there is a backward, a launch-count pin, and a note in the
  status table. Python and Rust must stay at feature parity (`docs/test_guidline.md`).
- No device → host read inside a training step (`read_count()` stays 0 across `queue_step`).
- Never `unwrap()` a device error; return `Result`.
- One task per commit. Commit messages start with the task id (`K7: ...`).
- Numbers come from the profiler, never from reasoning about the code. If a measurement contradicts this document,
  the measurement wins; write it down.

---

## 1. Vocabulary

- **launch**: one kernel dispatch, counted by `mamba3::backend::launch_count()` (`src/backend.rs:525`).
- **tally**: launches per source site, `start_launch_tally()` / `launch_tally()` (`src/backend.rs:446-472`).
- **step**: `Trainer::queue_step` for one batch (`src/train/trainer.rs`), i.e. forward + backward + optimizer.
- **fused path**: `set_fused_entity_model(true)` (the default); **composed path**: the same computation with one
  primitive per tensor op, used as the oracle in parity tests.
- **pass**: one read or write of a whole tensor by a kernel. The quantity this plan minimises.

---

## 2. Measurement harness (M0) — do this first

### M0.1 Make the profiler's timing honest

**Goal.** `profile_entity_model` reports 261 ms/step with `MAMBA3_ENTITY_STEPS=2` and 1,047 ms/step with `=4` for the
same work: with 2 steps the wgpu queue absorbs the launches and `device.synchronize()` returns before they have all
executed. The tool must report the same ms/step for 2, 4 and 8 steps within 15%.

**Files.** `examples/profile_entity_model.rs:156-188` (`profile_stage`), `src/backend.rs` (`Device::synchronize`).

**Do.**
1. In `profile_stage`, after the warm-up loop and before starting the timer, drain the queue completely: call
   `device.synchronize()` **and** read back one scalar (a 1-element tensor created with `Tensor::zeros` (`src/tensor/base.rs:255`) → `to_data()` (`base.rs:112`)),
   because a read is the only operation guaranteed to wait for every queued kernel. Do the same read after the timed
   loop, before taking `elapsed()`. Exclude the read's own cost by timing a bare read once and subtracting it (print it).
2. Set the default `MAMBA3_ENTITY_STEPS` to 8.
3. Add to the printed header: batch, steps, `CUBECL_WGPU_MAX_TASKS` if set, and `fused_entity_model()`.

**Test.** `tests/profile_timing.rs` (new, `#[ignore]`d on `cpu`): run the fused optimizer stage with steps = 2, 4, 8
at batch 8 and assert the three ms/step values are within 15% of their mean.

**Command.** `cargo run --release --features vulkan --example profile_entity_model` with `MAMBA3_ENTITY_STEPS=2`, then
`=8`; the two "optimizer step" lines agree within 15%.

**Done when.** The test passes on `vulkan`; the README of `examples/` documents the read-based drain.

### M0.2 Attribute launches to the tensor op and to the model region

**Goal.** The tally names the helper (`elemwise.rs:86` is `flat_launch`, a function every flat kernel calls), so it
cannot say which op or which part of the model launched. Make it print three columns: model region, public op, site.

**Files.** `src/backend.rs:440-480`, `src/tensor/ops/elemwise.rs:86` (`flat_launch`), every `pub fn` in
`src/tensor/ops/*.rs` that launches, `src/models/mamba3.rs`, `src/ssm/scan.rs`, `src/tensor/ops/entity_model.rs`.

**Do.**
1. In `src/backend.rs` add a thread-local label stack:
   ```rust
   thread_local! { static TALLY_LABELS: core::cell::RefCell<Vec<&'static str>> = const { core::cell::RefCell::new(Vec::new()) }; }
   pub struct TallyScope;                       // pops on drop
   pub fn tally_scope(label: &'static str) -> TallyScope {
       TALLY_LABELS.with(|s| s.borrow_mut().push(label)); TallyScope
   }
   impl Drop for TallyScope { fn drop(&mut self) { TALLY_LABELS.with(|s| { s.borrow_mut().pop(); }); } }
   ```
   When a launch is charged (the function below `/// Charge one launch to `site``, `src/backend.rs:480`), read the top
   label (or `"-"`) and key the tally by `(label, op, file, line)`. `launch_tally()` returns rows
   `("label / op / file:line", count)`.
2. Give every public launching op a name: add `#[track_caller]` is **not** enough (it names the caller's line, not the
   op). Instead pass the op name explicitly: change `flat_launch` (`elemwise.rs:86`) and the equivalent helpers in
   `reduce.rs`, `movement.rs`, `matmul.rs`, `scan.rs` to take `op: &'static str` and forward it to the charge. Every
   `pub fn` that calls a helper passes its own name (`"silu"`, `"strided_copy"`, `"sum_dim"`, …). Do this mechanically:
   one op at a time, compile, next op.
3. Put scopes in the model: in `Mamba3Mixer::apply` (`src/models/mamba3.rs:556`) wrap the projection
   (`self.project`, `:368`) in `tally_scope("mixer.project")`, the convolution + `silu` in `"mixer.conv"`, the SSM
   coefficient path in `"mixer.coef"`, the scan call in `"mixer.scan"`, the gate + out projection in `"mixer.out"`; in
   `ssd_chunked` (`src/ssm/scan.rs:141`) wrap the four stages (`"scan.intra"`, `"scan.summary"`, `"scan.inter"`,
   `"scan.out"`); in the entity model forward wrap `"encoder"`, `"queries"`, `"decoder"`, `"heads"`, `"loss"`; in the
   trainer wrap `"backward"` and `"optimizer"`. A scope costs one Vec push/pop; leave them in.
4. Extend `profile_entity_model` to print the top 25 rows and, per label, the sum.

**Test.** `tests/launch_tally.rs` (new): on `cpu`, run one fused step at batch 2 with the Kaggriculture spec, and
assert (a) every row's op field is non-empty, (b) the labels seen include exactly the set above, (c) the sum over rows
equals `launch_count()`.

**Command.** `MAMBA3_ENTITY_BATCH=32 cargo run --release --features vulkan --example profile_entity_model`.

**Done when.** The tally shows, for the fused optimizer stage at batch 32, launches per label. Copy that table into
§4 as the **G0 attribution**. Everything after this task is chosen from that table, not from this document's guesses.

### M0.3 One command for the gate numbers

**Goal.** A script that produces the status-table row.

**Files.** `bench/entity_step.sh` (new), `bench/results/` (new, committed).

**Do.** The script builds the vulkan profiler, runs it with `MAMBA3_ENTITY_STEPS=8` and batch 8, 32, 128 (fused only),
and appends one Markdown row per batch to `bench/results/entity_step.md` with: date, commit, batch, ms/step,
launches/step, reads/step, the top 5 tally labels with their counts. It refuses to run if `git status --porcelain` is
non-empty (numbers must belong to a commit).

**Done when.** `bash bench/entity_step.sh` appends three rows for `91df9ea` + M0 that reproduce §0.1 within 10%.

---

## 3. Kernel tasks (K7–K12)

Order is by (expected gain) / (risk). Re-order only if the G0 attribution table says a later task's region is larger
than an earlier one's; write the reason in §4.

### K7. Multi-tensor AdamW and grad-norm (516 → ≤ 8 launches per step)

**Goal.** The optimizer launches `adamw_step` once per parameter tensor (`src/train/optim.rs:320-360`) and
`sum_squares_into` once per gradient (`optim.rs:647`); with ≈ 258 parameter tensors that is 516 launches of tiny
kernels, all in the ≈ 0.15 s floor. Process up to 8 tensors per launch.

**Files.** `src/tensor/ops/fused.rs` (add `adamw_step_multi`, `sum_squares_multi`), `src/train/optim.rs`,
`tests/train.rs`, `tests/rl_update_footprint.rs` (launch pins).

**Design (follow exactly).** CubeCL kernels take a fixed number of buffers, so use 8 slots, like `gather_rows_multi`
(`src/tensor/ops/entity_model.rs`, K1 of the entity plan) which takes up to 8 tables in one launch.

```rust
/// Up to 8 parameter tensors updated in one launch. `lens[i]` is the element count of
/// slot i (0 for an unused slot); `decay[i]` its weight decay. One thread per element of
/// the longest slot; each thread loops over the 8 slots and skips slots it is past.
#[cube(launch_unchecked)]
fn adamw_multi_kernel<F: Float + CubeElement>(
    p0: &mut Array<F>, g0: &Array<F>, m0: &mut Array<F>, v0: &mut Array<F>,
    /* … p1..p7, g1..g7, m1..m7, v1..v7 … */
    lens: &Array<u32>,          // [8]
    decay: &Array<F>,           // [8]
    scale: &Array<F>,           // [1] the loss-scale inverse, as adamw_step takes it
    lr: F, beta1: F, beta2: F, eps: F, bias1: F, bias2: F,
) { /* for slot in 0..8 { if ABSOLUTE_POS < lens[slot] { …the body of adamw_kernel on slot's buffers… } } */ }
```
Write the slot body once as a `#[cube] fn adamw_elem(...)` and call it 8 times; do not copy-paste the arithmetic.
Unused slots are bound to a shared 1-element dummy tensor with `lens[i] = 0`.

`sum_squares_multi`: the same slot layout; each slot's partials go to their own region of the shared `partials`
buffer that `sum_squares_into` already uses (`fused.rs:2556` `sum_squares_groups`); the final `clip_factor`
(`fused.rs:2522`) is unchanged.

In `optim.rs`, replace the per-parameter loop with: collect `(param, grad, m, v, decay)` for every parameter that has
a gradient into a `Vec`, then `for chunk in list.chunks(8) { adamw_step_multi(chunk, …) }`. Keep the old
per-parameter function for the composed path and for parameters on a different device.

**Traps.** Slot buffers must be the same element type; `E` is generic, so instantiate once. `param.set(next)` is
gone: the multi kernel updates in place, so `Param` must expose a mutable value or the kernel writes into a fresh
buffer per slot that is then `set` (do the in-place version; `adamw_step` already returns a new tensor only because
the single kernel was written that way — `Param::set` is at `src/nn/param.rs:68`). Bias correction constants are per
step, not per slot.

**Test.** `tests/train.rs`: train 20 steps of the Kaggriculture spec on random data with the per-parameter optimizer
and with the multi-tensor one from the same seed; every parameter agrees to 1e-6 relative on `cpu` and `vulkan`.
Launch pin: optimizer launches per step ≤ 2 × ceil(258 / 8) + 3.

**Gate G1.** batch 8 fused step ≤ 0.22 s (from 0.31), batch 128 ≤ 3.0 s.

### K8. Fuse the mixer's elementwise chains (from the G0 attribution)

**Goal.** Remove the passes between the mixer's projection and the scan, and between the scan and the output.

**Files.** `src/models/mamba3.rs:368` (`project`), `:556` (`apply`), `src/tensor/ops/fused.rs`,
`src/autograd/ops.rs` (the `Var` wrappers, pattern at `:1180` `silu` / `:1195` `swiglu`), `tests/ssm.rs`,
`tests/entity_kernels.rs`.

**Method.** Take the G0 attribution rows under `mixer.*` and `scan.*` sorted by count. For each row, apply the first
pattern that fits, in this order. Stop when the remaining rows are each < 2% of the step's launches.

- **P1 — chain into one flat kernel.** A sequence of unary / binary elementwise ops on tensors of the same shape
  (`add`, `mul`, `exp`, `clamp`, `silu`, a scalar scale…) becomes one kernel with one adjoint. Template: `silu` /
  `silu_backward` (`fused.rs:1586-1600`) and its `Var` wrapper (`autograd/ops.rs:1180`); keep the `*_composed`
  twin as the oracle. Example already present: `swiglu` replaces `mul(silu(other))`.
- **P2 — write the pieces directly.** `split` / `slice` on the last axis followed by an elementwise op (the
  projection: `projected.split(&widths, 2)` then `xbc.silu()` then `xbc.split(...)`, `mamba3.rs:368-400`) becomes one
  kernel that reads `projected` once and writes each piece (with `silu` applied to the pieces that need it): inputs
  `[B, L, W]`, `widths[8]`, `apply_silu[8]`; outputs up to 8 tensors. Adjoint: the gradient of each piece is written
  back into one `[B, L, W]` buffer (one kernel, the inverse layout). This replaces 1 `silu` + up to 6 strided copies
  per layer.
- **P3 — reduce with an epilogue.** A reduction followed by a scalar op (`sum` then scale, `mean`) uses the existing
  `sum_dim_scaled` (`reduce.rs`, "one launch, not two"); a reduction followed by `exp` / `clamp` gets an epilogue
  parameter on the reduce kernel (add `epilogue: u32` = 0 none, 1 exp, 2 clamp…).
- **P4 — a `permute` + `reshape` pair is one copy; make it zero.** `heads_first` and `rows` in `ssd_chunked`
  (`scan.rs:193-200`) each materialise a strided copy so that the following `matmul` sees a contiguous operand. Give
  `matmul` and `matmul_nt` a batched-strided variant that takes the operand's strides (K12 does this; in K8 only count
  these copies and leave them).

For every kernel: forward formula, adjoint formula, `Var` wrapper behind the existing fused switch
(`fused_entity()`), composed twin, parity test (`cpu` + `vulkan`, relative 1e-5), `check_grad`, launch pin.

**Gate G2.** batch 128 fused step ≤ 2.4 s; `mixer.*` launches per layer ≤ 12 excluding matmuls and the scan.

### K9. Single-pass reductions where the axis is short

**Goal.** `reduce_op!` (`reduce.rs:75`, the split at `:156-167`) splits a reduction into `groups` partials and a fold whenever
`split_factor` (`reduce.rs:217`) says so — two launches. For the shapes in this model (axis ≤ 512, many outputs) one
launch is faster on this GPU.

**Files.** `src/tensor/ops/reduce.rs`, `tests/tensor.rs`.

**Do.** Measure first: time `sum_dim` on `[128·120, 128]` axis 1, `[128, 120, 128]` axis 1, `[128·20·3, 101]` axis 1
(the loss) with the split on and off (`bench/reduce.sh`, new, prints µs per call, 200 calls after warm-up). Then change
`split_factor` to return `None` when `outputs ≥ 4096` or `axis_len ≤ 512`, **using the numbers you measured**; put
the thresholds in one `const` each with the measurement in the comment.

**Test.** Existing reduction parity tests keep passing; add a launch pin: `sum_dim` on the three shapes above is one
launch.

**Gate G3.** batch 128 fused step ≤ 2.0 s; reduction launches per step ≤ 40% of G0's.

### K10. bf16 storage for activations

**Goal.** Half the bytes per pass. `FloatElem` is implemented for `half::bf16` (`src/backend.rs:101`) and `half::f16`
(`:87`); the entity model is generic in `E`, so this is a switch plus a precision audit, not new kernels.

**Files.** `bindings/python/src/entity_model.rs` (a `dtype` argument), `src/models/entity/model.rs`, `tests/entity_f16.rs`.

**Do.**
1. Run the existing `tests/entity_f16.rs` with `E = bf16`; list every kernel that accumulates in `E` (reductions, the
   scan's cumulative sums, `sum_squares`, the loss) — those must accumulate in `f32` and store `E`. Fix each (the
   reduce kernels already have a `Float` accumulator type parameter? check; if not, add one).
2. Keep master weights and optimizer moments in `f32` (K7's kernel then takes `p, m, v` as `f32` and `g` as `E`).
3. Python: `EntityModel(spec, ..., dtype="bf16")`; `train_entity.py --dtype bf16` in the Kaggriculture repo.

**Test.** Kaggriculture spec, 200 steps from the same seed in `f32` and `bf16` on `vulkan`: loss curves within 2% at
step 200, no non-finite step. Accuracy A/B on the real data (the Kaggriculture repo's `train_entity.py`, 1 epoch,
`--max-train 20000`): dev next-visit top-1 within 0.3 pp.

**Gate G4.** batch 128 fused step ≤ 1.3 s.

### K11. The Python path

**Goal.** Python's step is 1.6× the Rust profiler's for the same launches (5.5 s vs 3.3 s at batch 128).

**Files.** `bindings/python/src/entity_model.rs:545-569` (`queue_train_step`), `src/models/entity/batch.rs:1091`
(`EntityBatch::from_ids`), `src/train/trainer.rs`.

**Do.** Wrap the four parts of `queue_train_step` in tally scopes (`"py.ids"`, `"py.gather"`, `"py.step"`,
`"py.components"`) and time each with `std::time::Instant` printed under `MAMBA3_PROFILE_PY=1`. Then fix the largest:
the likely ones are (a) `EntityTask::new` per step rebuilding the loss segment table (build once, keep in the model),
(b) `take_components` keeping every head's loss tensor per step (keep only the totals unless `read_losses` asked for
heads), (c) the host `Vec<u32>` id conversion and upload (one small upload per step is fine; two or more is not).

**Test.** `bindings/python/tests/test_entity_speed.py` (`@pytest.mark.slow`): batch 128, 8 steps, Python ms/step ≤
1.1 × the Rust profiler's number read from `bench/results/entity_step.md`.

**Gate G5.** As the test.

### K12 (optional). Strided operands for the scan's matmuls

**Goal.** Remove the ≈ 10 strided copies per layer that `heads_first` / `rows` / the `permute(...).reshape(...)`
pairs in `ssd_chunked` (`scan.rs:193-300`) create for `matmul`, `matmul_nt` and `mul`.

**Files.** `src/tensor/ops/matmul.rs:1576-1620` (`launch_matmul`), `src/ssm/scan.rs`.

**Do.** Add `matmul_strided(a, a_strides, b, b_strides)` that reads operands through their strides (batch dims may be
permuted; the two inner dims must each be contiguous or the kernel falls back to a copy). Replace each
`permute + reshape + matmul` in `ssd_chunked` with the strided call; keep the copies for `mul` unless the tally says they
matter. Parity test against the copying version on both backends; `check_grad`.

**Gate G6.** batch 128 fused step ≤ 1.0 s.

---

## 4. Status table (fill in as you go)

| Task | Commit | batch 8 | batch 32 | batch 128 | launches / step | Gate | Kept? | Note |
|---|---|---|---|---|---|---|---|---|
| G0 (`91df9ea`) | – | 0.31 s | 1.05 s | 3.32 s | ≈ 2,200 | – | – | vulkan, Radeon 860M |
| M0 | | | | | | – | | attribution table below |
| K7 | | | | | | G1 ≤ 3.0 s | | |
| K8 | | | | | | G2 ≤ 2.4 s | | |
| K9 | | | | | | G3 ≤ 2.0 s | | |
| K10 | | | | | | G4 ≤ 1.3 s | | |
| K11 | | | | | | G5 py ≤ 1.1× | | |
| K12 | | | | | | G6 ≤ 1.0 s | | |

**G0 attribution (fill from M0.2):**

| label | launches / step | share |
|---|---|---|
| | | |

---

## 5. Traps, collected

- **T1 Profiler under-reports with few steps.** `MAMBA3_ENTITY_STEPS=2` gave 261 ms/step, `=4` gave 1,047 ms/step for
  the same work (the queue absorbs the first submissions). Until M0.1 lands, always use ≥ 8 steps and compare two
  step counts.
- **T2 Batch scaling is the diagnostic.** A launch-bound step is flat in the batch; a bandwidth-bound step is linear.
  Before and after every task, run batch 8 and 128 and say which one moved.
- **T3 Kernel cache.** `cubecl.toml` (`[compilation] cache = "target"`) is read from the process's working directory.
  Running Python from another directory recompiles every kernel at start (2.5 min at batch 128 in the first run) and
  reruns autotune. Not a per-step cost, but it ruins short benchmarks: warm up ≥ 4 steps per shape.
- **T4 "Fused" is a switch, not a promise.** `fused_entity_model()` true only routes the K1–K5 kernels; the mixer and
  the scan are untouched. Check the tally, not the flag.
- **T5 Slots.** Multi-tensor kernels (K7) bind unused slots to a dummy tensor with length 0; never bind the same
  mutable buffer to two slots.
- **T6 bf16 accumulation.** A reduction that accumulates in bf16 loses 3 digits; every reduce / cumsum / norm kernel
  must accumulate in f32 (K10). The existing f16 loss-scale path (`loss_scale=1024`) stays for f16; bf16 needs none.
- **T7 Two queries anchored at the same entity** (from `ENTITY_MODEL_PLAN.md`): any scatter-style adjoint must use
  gather-style accumulation; test that case.
- **T8 Reproducibility.** `set_matmul_kernel("auto")` times candidates per process; pin one kernel in parity tests.
- **T9 Memory on the iGPU.** Datasets and models live in system memory. Do not run a profiler while another large
  process (a 4-worker game collection at 3 GB each) is running; the machine has 30 GB.

---

## 6. Out of scope

- A graph compiler / lazy tensors (a general fusion engine). This plan fuses the chains the tally names, by hand.
- Attention, LoRA, the RL (PPO) kernels: not in the entity training step.
- Discrete-GPU tuning (workgroup sizes, tensor cores): measure again on such a machine before touching them.

## 7. Definition of done

- M0.1–M0.3 landed; `bench/results/entity_step.md` has the G0 row and one row per kept task.
- K7–K11 each either kept with their gate met (or ≥ half the gain, noted) or reverted with the number noted.
- batch 128 fused step ≤ 1.3 s on this machine (G4), Python within 1.1× of Rust (G5), all tests green on `cpu` and
  `vulkan`, Python / Rust parity tests green.
- The Kaggriculture repo's `train_entity.py` runs an epoch of its data at ≤ 1.3 s/step with `--dtype bf16`, and its
  dev next-visit top-1 is within 0.3 pp of the f32 run.
