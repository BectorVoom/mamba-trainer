# bf16 activations for the entity model, exposed in Python

Execution document for opencode tasks; each task lands on its own and leaves the cpu suite green. Tests run on cpu and on
the local Mac GPU (`--features wgpu`), Python and Rust stay at feature parity (`docs/test_guidline.md`).

## 0. Why

On the Radeon 860M the Transformer planner trains 2.3x faster in bf16 autocast than in fp32 (`bench/results/planner_step.md`),
because that iGPU is bandwidth-bound and bf16 halves the bytes of every pass. The entity model's step is ~1,100 kernels, most
of them memory passes over activations (`bench/results/entity_step_t4.md`), so storing activations in 16 bits is the largest
remaining lever. `KERNEL_OPTIMIZATION_PLAN.md` K10 made the entity model run with `E = half::bf16` (tests/entity_bf16.rs:
200-step loss parity with f32 on cpu), but deferred three parts, which this plan does:
(a) f32 master weights and optimizer moments, (b) f32 accumulation in reductions (trap T6), (c) the Python `dtype`.

## 1. Design (decided)

- **Element type = activation storage.** A model built with `E = bf16` (or `f16`) stores activations, parameters as the
  forward reads them, and gradients in `E`. Every kernel accumulates in f32 and rounds once when it stores (T6).
- **Master weights.** The optimizer keeps an f32 master copy of every parameter plus f32 moments. A step reads the `E`
  gradient, updates the f32 master and moments, and writes `E` parameters rounded from the master. Updates smaller than
  bf16's resolution (8 bits of mantissa: a weight of 1.0 cannot move by 3e-4) accumulate in the master instead of being
  lost. Gradient clipping computes the global norm in f32. For `E = f32` nothing changes (no master copy, same kernels,
  same launch counts).
- **Checkpoints** store the f32 master weights, so a bf16 run's checkpoint loads into an f32 model and vice versa.
- **Backends.** `bf16` needs a backend with the type: cpu, CUDA (sm_80+ for arithmetic; check sm_75), Vulkan / SPIR-V where
  the device reports it. WGSL (the default Mac wgpu backend) has no bf16; `f16` runs there. Every entry point checks
  `backend::supports_dtype` and refuses an unsupported dtype with a message naming the backend and the alternative.
- **f16** needs loss scaling (5 exponent bits); the existing `loss_scale` / `EntityTask::with_loss_scale` applies. bf16 has
  f32's exponent range and needs none.
- **Python:** `dtype="f32" | "bf16" | "f16"` on `EntityModel`, `EntityDataset` and `EntityPolicy` (default `"f32"`,
  unchanged behaviour). A dataset and the model training on it must share a dtype (clear error otherwise).
  `m3.supports_dtype(name) -> bool`.

## 2. Tasks

### B1. f32 master weights and moments for reduced-precision parameters
Files: `src/train/optim.rs` (AdamW, `grad_scale`, the multi-tensor path), `src/tensor/ops/fused.rs` (`adamw_step_multi`,
`sum_squares_multi`, `sumsq_slot`), `src/nn/param.rs` / module state dicts as needed, `src/models/entity/model.rs` save/load.
- AdamW over `Param<R, E>` with `E != f32` keeps, per parameter, an f32 master (initialised from the parameter at its first
  step) and f32 `m`, `v`. A mixed multi-tensor kernel updates up to N parameters per launch: reads `g: E`, updates
  `p_master, m, v: f32`, writes `p: E`. Choose N from `max_bindings` like the existing K7 kernel does (it has 8- and 2-slot
  widths); fall back per parameter where needed. `E = f32` keeps the existing kernels and launch counts exactly.
- The gradient-norm partial sums accumulate in f32 whatever `E` is (`sumsq_slot` accumulates in `F` today), and the partial
  buffer and its final reduction are f32.
- `save` writes the master weights as f32; `load` rounds into the model's `E`.
- Tests (`tests/entity_bf16.rs`, extend): (1) 200 steps, bf16 with master weights vs f32: loss within 2% at step 200, no
  non-finite loss; (2) a small learning rate (1e-5) moves bf16-with-master parameters over 50 steps where pure bf16 storage
  would not (assert the master changed and the rounded parameter changed at least once); (3) checkpoint written from bf16
  loads into an f32 model with the same weights up to bf16 rounding; (4) f32 launch counts unchanged
  (`tests/entity_footprint.rs`, `tests/adamw_multi.rs` pass unchanged).

### B2. f32 accumulation audit (T6)
Every kernel on the entity training step that sums or scans must accumulate in f32 for `E = bf16 / f16`: `reduce.rs`
(sum_dim, sum_all, mean, max, the split/two-pass paths), `scan.rs` (cumsum), `fused.rs` (rms_norm forward/backward,
softmax / log_softmax / cross-entropy rows, rotate backward table sums, causal conv weight grad, swiglu backward if it
reduces), `entity_model.rs` (the segmented / planner loss kernels, pooling), `ssd_scan.rs` (already f32 — confirm), matmul
(accumulates in `F` = f32 for mixed plans — confirm the `E = bf16` plans accumulate in f32 too). For each: fix it if it
accumulates in `E`, and add a parity test at the entity model's shapes: bf16 input, f32 reference computed on the host from
the same bf16-rounded inputs, tolerance of a few bf16 ulps of the output (a 20,000-long sum in bf16 accumulation fails this;
f32 accumulation passes).

### B3. Python `dtype`
`bindings/python/src/{lib.rs, entity_model.rs, entity_rl.rs}`, `python/mamba3_rl/{__init__.py, _mamba3_rl.pyi}`:
- `EntityModel(spec, ..., dtype="f32")`, `EntityDataset(spec, arrays, dtype="f32")`, `EntityPolicy(spec, ..., dtype="f32")`,
  `EntityPolicy.from_model` / `to_model` keep the dtype, `save` / `load` round-trip (checkpoints are f32 master weights, so
  `load(path, dtype=...)` picks the element type), `m3.supports_dtype(name)`. Refuse unsupported dtypes up front.
- The binding's `type E = f32` becomes a per-object choice: hold the model as an enum over the three monomorphisations
  (`EntityModel<R, f32>`, `<R, half::bf16>`, `<R, half::f16>`) and dispatch each method with a macro, so no method is
  written three times. Observations and labels still arrive as numpy f32 / f16 / ints and are converted on upload.
- f16: the constructor's `loss_scale` (default 1024 for f16, 1 otherwise) is applied, as `EntityModel` already allows.
- Tests `bindings/python/tests/test_entity_dtype.py`: for each dtype `supports_dtype` reports true on the current backend:
  train the tiny spec 50 steps, losses track f32 within 5%; predict / act work; save in bf16, load as f32 and back;
  mismatched dataset / model dtype raises; unsupported dtype raises with the backend's name; read budgets unchanged
  (`act` 1 read, `update` 1 read, `read_losses` as before).

### B4. Measure
Step time f32 vs 16-bit on the local Mac GPU (wgpu: f16), and bf16 on a T4 or the Radeon when available, with
`examples/profile_entity_model` (add `MAMBA3_ENTITY_DTYPE=f32|bf16|f16`) and `bindings/python/examples/bench_entity_training.py`
(add `--dtype`). Record in `bench/results/entity_step_t4.md` / a new `bench/results/entity_dtype.md`.

## 3. Status

| Task | Commit | cpu | wgpu (Mac) | Note |
|---|---|---|---|---|
| B1 | – | – | – | |
| B2 | – | – | – | |
| B3 | – | – | – | |
| B4 | – | – | – | |
