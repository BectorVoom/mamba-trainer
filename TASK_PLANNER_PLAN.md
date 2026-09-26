# Task planner plan (Mamba-3 replacement for the Kaggriculture Transformer planner)

This is an execution document. It is written so that it can be worked start to finish **without reading the session that
produced it**, by an implementer who follows instructions literally. Every task gives: the goal, the files, the exact API to
add (with code skeletons), the traps, the test, the command that proves it, and a "done when" line. Work the tasks **in the
order given**. Each task lands on its own and leaves the suite green. Do not skip a measurement gate.

**Primary path: on-device GPU kernels.** T1–T5 first build the model from existing ops; that composed version is kept only
as the correctness oracle. K0–K6 then move the batch assembly, the gathers, the loss and the small joins onto the GPU as
CubeCL kernels with hand-written, atomic-free adjoints, behind one switch. Training, profiling and the final run use the
kernel path. Order: **T1–T5 → K0–K6 → T6–T8 → T10**.

Written against commit `d2351f2` of this repo. Line anchors (`file:line`) are from that commit; if a line moved, search for the
quoted symbol instead.

---

## 0. Before you start

### 0.1 Where things are

| What | Path |
|---|---|
| This repo (the "crate") | `/home/user/Documents/workspace/mamba-trainer` (on the teammate's Mac: `/Users/ods/Documents/mamba-trainer`) |
| The Kaggriculture repo (data, the PyTorch reference model, evaluation) | `/home/user/Documents/workspace/Kaggriculture` |
| PyTorch reference model (what we reproduce) | `Kaggriculture/experiments/kobayashi/exp-planner053_dsm_task_planner/src/model.py` |
| Features and label definitions | `.../exp-planner053_dsm_task_planner/src/features.py`, `src/extract.py` |
| Training data (npz) | `.../exp-planner053_dsm_task_planner/runs/20260926_task_planner/data/{train,dev16,test40}.npz` |
| Evaluation script (reused unchanged) | `.../exp-planner053_dsm_task_planner/src/evaluate.py` |
| Wheel build helper | `Kaggriculture/tools/setup_mamba3.sh <backend>` (calls this repo's `tools/build_wheel.sh`) |

### 0.2 Rules (from `ENTITY_ENCODER_PLAN.md` §0 and `AGENTS.md`; they still apply)

- Check **exit codes**, not grep output: `cmd > /tmp/t.log 2>&1; echo "exit=$?"`.
- Never build `wgpu`/`vulkan` while CPU tests run: they share `target/`.
- Never benchmark while anything else runs (no other build, no training, no browser video).
- Test on **both CPU and GPU** (`cpu` and `vulkan` features).
- Keep **Python and Rust APIs in parity**: every Rust entry point added here has a Python binding and a stub in
  `bindings/python/python/mamba3_rl/_mamba3_rl.pyi`, and vice versa.
- Performance work uses the profile tools (`examples/profile_*.rs`, `launch_count()`, `read_count()`); CubeCL manual:
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl`.

### 0.3 The working tree is not clean

At the time of writing, `git status` shows uncommitted work that is **not part of this plan**:
`Cargo.toml`, `src/backend.rs`, `src/tensor/ops/entity.rs` (modified) and `examples/bench_entity_kernels.rs`,
`examples/recall_probe.rs`, `tests/entity_kernels_wide.rs` (untracked). Ask the owner whether to land or stash it first.
**Never commit it as part of this plan.** Commit only the files this plan names.

### 0.4 Baseline suite

```bash
cd /home/user/Documents/workspace/mamba-trainer
cargo test --release --no-default-features --features cpu > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|^error|FAILED" /tmp/t.log
```

It must be green before you touch anything. If it is not, stop and report.

### 0.5 GPU backend on this machine

The GPU is an AMD Radeon 860M (gfx1151, 16 GB shared). **Use the `vulkan` feature.** The `hip` feature builds but crashes at
the first compute (`Rollout(policy, 2)` segfaults in the CubeCL-HIP device path; see
`Kaggriculture/experiments/kobayashi/exp-planner052_dsm_mamba_il/runs/20260926_mamba_il/REPORT.en.md`). Do not debug HIP as
part of this plan.

---

## 1. What we are building and why

### 1.1 The task

Kaggriculture is a 10 × 10 farm game. Each turn, a crew of up to 20 units (a farmer + hired hands) acts. The **task planner**
predicts, for every unit, its next **K = 3 work visits** that day: which tile, which op first, which ops in total, which crop
if it plants, and how many turns until the first op. It is trained by imitation on the replays of the leaderboard leader.

A PyTorch Transformer already does this (`model.py`, ≈ 4.07 M parameters). This plan rebuilds it on this crate with **Mamba-3
mixers instead of attention**, so that it trains on the AMD GPU through Vulkan and can later share code with the RL policies.

### 1.2 Inputs and labels (fixed by the npz files; do not change them)

Per sample (one turn):

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `tiles` | float16 | `[100, 48]` | per-tile state, row-major `index = y*10 + x`; feature names in `features.TILE_FEATURES` |
| `glob` | float32 | `[114]` | day, hour, special days, economy (`features.global_features`) |
| `units` | float16 | `[20, 36]` | per unit: slot one-hot, position, inventory (`features.unit_features`) |
| `upos` | int16 | `[20]` | tile index of each unit, `-1` = no unit in this slot (padding) |
| `tgt` | int16 | `[20, 3]` | target of visit j: `0..99` tile, `100` = NONE (no more work today), `-100` = ignore (padding) |
| `op` | int16 | `[20, 3]` | first op of visit j, `0..12`, `-100` = ignore |
| `crop` | int16 | `[20, 3]` | crop planted in visit j, `0..4`, `-100` = ignore |
| `opset` | uint8 | `[20, 3, 13]` | multi-hot of the ops in visit j |
| `eta` | int16 | `[20]` | turns until visit 1's first op, `-1` = none |

Sizes: train 53,925 turns (562,929 unit-turns), dev16 11,504, test40 28,760. Constants: `NT = 100`, `U = 20`, `K = 3`,
`C_TILE = 48`, `C_GLOB = 114`, `C_UNIT = 36`, `N_OPS = 13`, `N_CROPS = 5`.

### 1.3 Reference numbers

The PyTorch model's dev16 and test40 results are in
`Kaggriculture/experiments/kobayashi/exp-planner053_dsm_task_planner/runs/20260926_task_planner/` (`eval_dev16.md`,
`eval_test40.md`, and `train_s0.log`). After one epoch it reached dev next-visit top-1 **77.2%**, op-at-true-target 88.5%.
Its speed on the same GPU (PyTorch on ROCm with torch's own bf16 autocast, batch 128) is **≈ 0.56 s per optimizer step** (235 s per epoch of 422
steps). These are the numbers to match (accuracy) and to beat (speed).

---

## 2. The design

### 2.1 Architecture (all Mamba, no attention)

```
tiles [B,100,48] ─ Linear→GELU→Linear ─┐
             + tile position (Param [100,d])     │
             + global (Linear→GELU→Linear on glob [B,114], broadcast to every tile)
                                                 ▼
                       TileMixer × n_tile_layers   (bidirectional Mamba-3 over the 100 tiles,
                                                    alternating row-major / column-major order)
                                                 ▼ T [B,100,d]
units [B,20,36] ─ Linear→GELU→Linear ─┐
 + one_hot(upos) [B,20,100] @ T  (the unit's own tile token)
 + step embedding (Param [3,d]) → queries Q [B,60,d]   (unit-major: q = u*3 + j)
                                                 ▼
             joint sequence S = [T ; Q]  [B,160,d]
                       JointMixer × n_joint_layers (bidirectional Mamba-3 over all 160 tokens)
                                                 ▼
                 H = S[:, 100:160]  [B,60,d]     T' = S[:, 0:100]  [B,100,d]
                                                 ▼
 target logits  = (H Wq) · [T' Wk ; none_key]ᵀ / √d   → [B,60,101]   (ONE batched matmul_nt)
 target token   = one_hot(target) [B,60,101] @ [T' ; none_key]      → [B,60,d]
 aux heads      = Linear(concat(H, target token)) → [B,60, 13 op | 13 opset | 5 crop | 1 eta]  (ONE Linear, then split)
```

Why each piece:
- **Tile mixer = bidirectional Mamba-3** (`Mamba3MixerConfig::with_bidirectional(true)`, as `VisionMamba3` does). The farm has
  no causal order, so both directions are needed. The fused bidirectional mixer launches each kernel once at double width.
- **Row / column alternation.** A 1-D scan over row-major tiles puts vertical neighbours 10 tokens apart. Every second tile
  layer reads the grid transposed (`reshape [B,10,10,d] → permute [0,2,1,3] → reshape [B,100,d]`), so vertical neighbours
  become adjacent. This costs 2 permute copies per alternated layer; §2.3 gates it on measured accuracy.
- **Global features broadcast-added to every tile** instead of a separate global token: no token bookkeeping, the sequence
  stays exactly 100 tiles, and one add replaces a `cat`.
- **Joint sequence instead of cross-attention.** The crate has no cross-attention and no key-padding mask
  (`src/nn/attention.rs:272` attends a sequence to itself only). Concatenating the queries after the tiles and running a
  bidirectional mixer over the 160 tokens lets queries read tiles (forward direction), tiles read queries (backward
  direction), and queries read each other (crew coordination). Pointer scoring then uses the updated tile tokens directly.
- **NONE as a learned key** appended to the tile keys: the 101 target logits come out of a single `matmul_nt` with no `cat` of
  logits and no second head.
- **Gathers.** In the composed oracle (T3): `one_hot(ids)` (`index.rs:497`) and one batched `matmul`. In the kernel path:
  index gathers with gather-style adjoints (K2), so no one-hot tensor exists at all.
- **Everything around the mixers is an on-device kernel in the training path**: batch assembly from a device-resident
  dataset (K1), the query build and token gathers (K2), the whole loss (K3), the grid transpose (K4), the embedding join and
  the NONE key (K5). The Mamba mixers, `Linear`s and `matmul_nt` are the crate's existing (already fused) kernels.
- **One fused aux Linear** for op / opset / crop / eta instead of four MLPs: one matmul instead of eight.

### 2.2 Shapes and defaults

| Symbol | Default | Note |
|---|---|---|
| `d_model` | 128 | try 192 only if dev top-1 is > 1 pp below the reference |
| `n_tile_layers` | 3 | |
| `n_joint_layers` | 3 | |
| SSM | `SsmConfig { d_model: 128, n_heads: 4, head_dim: 64, d_state: 32, n_groups: 1, chunk_size: 32, ..SsmConfig::default() }` before doubling | `n_heads × head_dim = 256 = 2 × d_model` (the usual expansion, `config.rs:134`). `d_state` 32 instead of the default 64 halves the scan work; T8 sweeps 32 / 64. `n_heads % n_groups == 0` is required (`config.rs:212`); the bidirectional block doubles both (see `vision.rs:181-195`) |
| `chunk_size` | **32** | sequences are 100 and 160 tokens. 160 = 5 × 32 needs no padding. 100 is padded to 128 inside `ssd_chunked` (`scan.rs:168-171`); try `chunk_size = 20` or 25 for the tile mixer if padding shows up in the profile (both divide 100) |
| batch | 128 turns | = 7,680 query tokens, 20,480 joint tokens per step |
| `alternate_axes` | true | gated in T8 |

### 2.3 Speed design (why this should be faster than the PyTorch reference)

At these sizes (`d = 128`, 160 tokens, batch 128) the GPU is **dispatch-bound**: each kernel does little work, and a launch on
wgpu/Vulkan costs tens of microseconds, and a device → host read about **1.4 ms** (`trainer.rs:309`). So speed comes from
**few launches and no reads inside the step**. The rules below are part of the design, not optional polish:

| # | Rule | Where it matters |
|---|---|---|
| S1 | **No device → host read inside `loss()`.** Do **not** use `cross_entropy_with(..., ignore_index)`: it reads the targets back to build its mask (`loss.rs:84`, `try_to_vec`). The composed oracle builds keep-masks on the host (T4); the training path builds them **on the device** (K1) and normalises by device sums inside the loss kernel (K3). | T4, K1, K3 |
| S2 | **No per-step uploads at all.** Each split is uploaded once (`PlannerData`, K1); a step sends only the `[B]` turn ids. (The host constructor `PlannerBatch::from_host` is for tests and the oracle.) | K1, T6 |
| S3 | Use `Trainer::queue_step` and read losses only every `log_every` steps (`trainer.rs:302`, `read_steps`), never after every step. | T6 |
| S4 | One `matmul_nt` for all 101 target logits (NONE key appended by `keys_with_none`, K5); index gathers instead of one-hot matmuls (K2); one Linear for all aux heads. | T3, K2, K5 |
| S5 | Chunk sizes that divide the sequence (160 / 32) so `ssd_chunked` does not pad and slice. | T2, T3 |
| S6 | **f16** matmul operands (`MatmulPrecision::F16`, `matmul.rs:95-112`), set through the checked `try_set_matmul_precision` (`matmul.rs:191`). **bf16 is not used: it cannot be tested on the target machine.** Master weights, gradients and accumulation stay f32; only what the matmul kernels read is rounded. f16 needs a **static loss scale** (§2.5). Gated by an accuracy A/B in T8. | T4, T6, T8 |
| S7 | Evaluation under `no_grad`, batch 512, logits read back **once per batch**. | T6 |
| S8 | **Device-resident dataset** (train ≈ 1.2 GB as f32, fits in 16 GB): batches are gathered on the device (K1). | K1 |
| S9 | **Fused on-device kernels** for everything that is not a mixer, a `Linear` or a matmul: one launch forward and at most one backward each, gather-style adjoints, no atomics, f32 accumulation (K0–K6 invariants). | K0–K6 |

The measurement gate (T8) checks these with `launch_count()` / `read_count()` and wall time. **Target: ≤ 0.25 s per optimizer
step at batch 128 on Vulkan** (≥ 2× faster than the PyTorch reference) and **zero reads per step** except the queued loss
readback. If the step is slower than 0.56 s, stop and report the profile before training for real.

### 2.4 Speed review of this plan (done while writing it)

A back-of-envelope check that the design can meet the T8 target, so the implementer knows what "normal" looks like.

**Compute per optimizer step (batch 128, defaults).** Token-layers per forward: tile mixer 128 × 100 × 3 = 38.4 k, joint
mixer 128 × 160 × 3 = 61.4 k, total ≈ 100 k. The bidirectional block doubles the heads, so per block `H = 8`, `G = 2`,
`d_inner = 8 × 64 = 512`, `N = d_state = 32`. Its projections cost about
`2 × d × (2·d_inner + 2·G·N + 3·H) + 2 × d_inner × d ≈ 2·128·1,176 + 2·512·128 ≈ 0.43 MFLOP` per token, plus the chunked
scan (small at `d_state 32`, chunk 32). Forward ≈ 45 GFLOP, forward + backward ≈ 130 GFLOP. The Radeon 860M sustains on the
order of 1–3 TFLOP/s in practice for these matmul shapes, so **arithmetic is ≈ 45–130 ms per step**. (With `d_state 64` the
scan roughly doubles; with `head_dim 32` the projections roughly halve — both are T8 sweep knobs if the gate is missed.)

**Launches per step.** A bidirectional mixer is one fused mixer (not two), so a block is one mixer's launches plus the
norm and the residual add. Expect a few hundred launches per step forward and roughly twice that backward. Around the
mixers the composed path spends ≈ 60 more (the loss ≈ 40, gathers and joins ≈ 20) plus the per-step uploads; the on-device
kernels K1–K5 bring that to ≈ 12 (K1: 2, K2: 4, K3: 3, K5: 4, K4: 2 per alternated layer) and remove the uploads. At tens of
µs per launch on Vulkan (≈ 10 µs of host time per launch was measured on Metal, `ENTITY_KERNEL_PLAN.md`), the mixers cost
≈ 20–60 ms and the kernel path saves a further ≈ 1–3 ms plus the upload and host time. `profile_vision.rs` is the yardstick
for the mixer blocks: compare launches per block with its bidirectional numbers.

**Reads per step: 0** by construction (rules S1, S3). Each avoided read saves ≈ 1.4 ms of fixed wait (`trainer.rs:309`).
The PyTorch reference reads nothing per step either, but pays Python dispatch for ≈ 4.07 M parameters' worth of eager ops.

**Expected step time: ≈ 0.1–0.25 s**, so the T8 gate of 0.25 s is realistic but not loose: meet it by the rules above, not by luck. The places where the plan could still be slow,
in the order to check:

| Risk | Why | Check / fix |
|---|---|---|
| A hidden device read | one `to_vec`/`scalar()` per step costs 1.4 ms and serialises the queue | `read_count()` delta must be 0 (T4 test 2, T8) |
| `ssd_chunked` padding the 100-token tile sequence to 128 | +28% tile-mixer work and a pad + slice | T8 sweeps `chunk_size` 20 / 25 / 32 / 64 for the tile mixer |
| Permute copies of `alternate_axes` | 2 copies of `[B,100,d]` per alternated layer | T8 measures on / off; keep only if it pays in accuracy |
| Per-step host work and uploads | numpy slicing, float16 → float32 conversion, one-hot building | removed by design: K1 (device dataset + on-device batch) |
| Composed loss and gathers | ≈ 40 loss launches and ≈ 20 gather/join launches per step | removed by design: K2, K3, K5 |
| Eval calling predict twice | two forwards | device argmax in `predict` (T5): one forward |

What the plan deliberately does **not** do for speed: no custom kernels (the fused bidirectional mixer, `matmul_nt` and
`cross_entropy_rows` already cover the hot paths), and no attention (no `[B,H,160,160]` score tensors).

### 2.5 f16 and the static loss scale

The target machine can test **f16 but not bf16**, so the plan uses `MatmulPrecision::F16` for speed and never bf16.

What F16 does in this crate (`matmul.rs:95-112`): a matmul rounds its `f32` operands to `f16` once per call and accumulates
in `f32`; parameters, gradients and optimizer state stay `f32`. Two failure modes follow from f16's range:
- **Overflow**: f16's largest value is 65504. The activations here are RMS-normalised and the logits are scaled by `1/√d`, so
  forward overflow is unlikely, but it must be detected, not assumed.
- **Underflow**: the backward matmuls read the upstream gradient as f16. The loss is a mean over ≈ 7,680 query rows, so
  per-element gradients are small (≈ 1e-5 to 1e-8) and fall into or below f16's subnormal range (smallest normal ≈ 6.1e-5,
  smallest subnormal ≈ 6e-8). They lose precision or become 0.

The crate has no loss scaler, and none is needed: use a **static loss scale S** (default 1024):
1. multiply the loss by `S` in `PlannerTask::loss` (T4), so every gradient is `S` times larger when it is rounded to f16;
2. build AdamW with `eps = 1e-8 × S` and the trainer with `max_grad_norm = 1.0 × S`. AdamW's update `m / (√v + eps)` is then
   exactly the unscaled update (m and √v both scale by S), the clip triggers at the same unscaled norm, and the decoupled
   weight decay does not see the gradient at all. So **S changes nothing but the f16 rounding**;
3. report `loss / S` and `grad_norm / S`.

If a step's loss or grad norm is not finite: stop, halve `S`, and restart from the last epoch checkpoint. If the loss is not
finite even at `S = 1`, the forward overflows: run in f32 and report which activation grows (print `max |x|` after each block
in a debug run).

**f16 is adopted only if T8's A/B passes**: 1 epoch in f16 vs f32 on the same seed, dev next-visit top-1 within 0.3 pp and no
non-finite step, and f16 at least 15% faster per step. Otherwise train in f32.

Before T8, confirm f16 works on the Vulkan backend at all:
`cargo test --release --no-default-features --features vulkan --test mixed_precision > /tmp/mp.log 2>&1; echo "exit=$?"`
(`the_capability_query_matches_what_the_kernels_do` checks the capability answer against the kernels). If it fails, use f32.

---

## 3. Repo map for these tasks

| Path | What you use from it |
|---|---|
| `src/models/vision.rs:166-236` | how a bidirectional stack is built: double `n_heads`/`n_groups`, `Mamba3MixerConfig::new(ssm).with_depth(n).with_bidirectional(true).init(device, rng)?`, `RmsNormConfig::new(d).with_eps(eps).init(device, rng)` |
| `src/models/vision.rs:328-346` | `VisionBlock`: `x + mixer(norm(x))`, and its `Module::visit` — copy this pattern |
| `src/models/vision.rs:382-398` | `Param::var(&anchor)`, `.expand(...)`, `cat(...)`, `PositionalEmbedding::add_to` |
| `src/models/mamba3.rs:120-212`, `:556` | `Mamba3MixerConfig`, `Mamba3Mixer::apply(&Var [B,T,d])` |
| `src/nn/linear.rs:54,115,222` | `LinearConfig::new(i, o).init(device, rng)`, `Linear::apply` (works on `[..., i]`) |
| `src/nn/param.rs:49,102` | `Param::new(Tensor)`, `Param::var(&anchor)` |
| `src/tensor/base.rs:101,255` | `Tensor::from_f32(&data, shape, device)`, `Tensor::zeros` |
| `src/tensor/ops/random.rs:25,62` | `Rng::seeded`, `rng.normal_vec(n, mean, std)` |
| `src/tensor/ops/index.rs:51,361,497` | `IdTensor::from_slice`, `gather_rows`, `one_hot(&ids, classes)` |
| `src/autograd/ops.rs` | `matmul` :298, `matmul_nt` :331, `gelu` :1907, `softplus` :1828, `log_softmax` :1924, `cross_entropy_rows` :1068, `take_along_last` :1145, `slice` :956, `split` :987, `permute` :899, `reshape` :874, `mul`/`add`/`sub` :161-231, `sum` :857, `mul_scalar` :743; free fn `cat` :1932 |
| `src/autograd/var.rs:60,87` | `Var::constant(Tensor)`, `var.tensor()` |
| `src/train/trainer.rs:20-33` | `trait TrainStep { type Batch; fn parameters(); fn loss(&Batch) -> Result<Var>; fn set_training(bool) }` |
| `src/train/trainer.rs:208,282,302,379` | `Trainer::new(config, optimizer)`, `step`, `queue_step`, `read_steps` |
| `src/train/tasks.rs:41-107` | `LmTask`: the template for a `TrainStep` implementation |
| `src/train/optim.rs:268` | `AdamW::new(lr)` (and `AdamWConfig` for weight decay) |
| `src/train/checkpoint.rs:182,255,274,496` | `Checkpoint::capture(&model, step).save(path)`, `Checkpoint::load(path)?.restore(&model)` |
| `src/models/mod.rs:12-20` | where to register the new module |
| `bindings/python/src/lib.rs:216-244` | `#[pymodule] fn _mamba3_rl`: `module.add_class::<...>()?` |
| `bindings/python/src/policy.rs:50-205` | template for a `#[pyclass(unsendable)]` wrapper with `#[new]`, `save`, `load` |
| `bindings/python/src/array.rs:66,143` | `tensor_2d`, `ids_1d`: host-side validation helpers |
| `bindings/python/src/err.rs:24` | `to_py(err)` |
| `examples/profile_vision.rs`, `examples/profile_entity.rs` | templates for the launch / time profile (K0, T8) |
| `src/tensor/ops/entity.rs`, `ENTITY_KERNEL_PLAN.md` | **templates for every on-device kernel in K0–K6** (see the table at the start of the K section) |
| `tests/autograd.rs:18` | `check_grad` (finite differences) |

---

## 4. Tasks

### T1. Module skeleton and config

**Goal:** a new module `src/models/planner.rs` with the config type, registered and compiling.

**Files:** create `src/models/planner.rs`; edit `src/models/mod.rs` (add `pub mod planner;` and
`pub use planner::{TaskPlanner, TaskPlannerConfig, PlannerBatch, PlannerOutput, PlannerTask};`).

**Code:**

```rust
//! Task planner: next K work visits per unit over a tile grid, all Mamba-3 (see TASK_PLANNER_PLAN.md).

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TaskPlannerConfig {
    pub grid: usize,          // 10 (tiles = grid * grid)
    pub c_tile: usize,        // 48
    pub c_glob: usize,        // 114
    pub c_unit: usize,        // 36
    pub max_units: usize,     // 20
    pub k: usize,             // 3
    pub n_ops: usize,         // 13
    pub n_crops: usize,       // 5
    pub d_model: usize,       // 128
    pub n_tile_layers: usize, // 3
    pub n_joint_layers: usize,// 3
    pub alternate_axes: bool, // true
    pub ssm: SsmConfig,       // SsmConfig { d_model: 128, n_heads: 4, head_dim: 64, d_state: 32, n_groups: 1,
                              //             chunk_size: 32, ..SsmConfig::default() }  (serde: config.rs:83)
    pub norm_eps: f32,        // 1e-5
    pub seed: u64,            // 0
}

impl Default for TaskPlannerConfig { /* the values in the comments above */ }

impl TaskPlannerConfig {
    pub fn tiles(&self) -> usize { self.grid * self.grid }
    pub fn queries(&self) -> usize { self.max_units * self.k }
    pub fn aux_width(&self) -> usize { self.n_ops + self.n_ops + self.n_crops + 1 }
    pub fn validate(&self) -> Result<()> {
        // every size > 0; d_model divisible by ssm.n_heads * 2 after doubling is not needed —
        // copy the checks VisionMamba3Config::validate (vision.rs:150) makes on d_model and the ssm
    }
}
```

`SsmConfig` is already `Clone + PartialEq + Serialize + Deserialize` (`src/ssm/config.rs:83`), so it can be a field.
`validate()` must also call `let mut ssm = self.ssm.clone(); ssm.d_model = self.d_model; ssm.validate()` as
`VisionMamba3Config::validate` does (`vision.rs:160-162`).

**Test** (`tests/planner.rs`, new file, start it now):

```rust
#[test]
fn default_config_validates_and_round_trips_json() {
    let c = TaskPlannerConfig::default();
    c.validate().unwrap();
    let s = serde_json::to_string(&c).unwrap();
    assert_eq!(serde_json::from_str::<TaskPlannerConfig>(&s).unwrap(), c);
}
```

**Command:** `cargo test --release --no-default-features --features cpu --test planner > /tmp/t.log 2>&1; echo "exit=$?"`

**Done when:** exit 0, and the full suite (§0.4) is still green.

---

### T2. The mixer block and the tile mixer

**Goal:** a residual bidirectional Mamba-3 block and the row / column alternation.

**Code** (in `planner.rs`):

```rust
struct BiBlock<R: Runtime, E: FloatElem> { norm: RmsNorm<R, E>, mixer: Mamba3Mixer<R, E> }

impl<R: Runtime, E: FloatElem> BiBlock<R, E> {
    fn new(cfg: &TaskPlannerConfig, depth: usize, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        let mut ssm = cfg.ssm.clone();
        ssm.d_model = cfg.d_model;
        ssm.n_heads *= 2;      // bidirectional: each direction keeps the configured capacity
        ssm.n_groups *= 2;
        Ok(Self {
            norm: RmsNormConfig::new(cfg.d_model).with_eps(cfg.norm_eps).init(device, rng),
            mixer: Mamba3MixerConfig::new(ssm).with_depth(depth).with_bidirectional(true).init(device, rng)?,
        })
    }
    fn apply(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        x.add(&self.mixer.apply(&self.norm.apply(x)?)?)
    }
}
// Module::visit: visitor.child("norm", ..); visitor.child("mixer", ..)   (copy vision.rs:342)

/// [B, g*g, d] row-major <-> column-major. The same function converts both ways.
fn transpose_grid<R: Runtime, E: FloatElem>(x: &Var<R, E>, g: usize) -> Result<Var<R, E>> {
    let (b, d) = (x.shape().dim(0), x.shape().dim(2));
    x.reshape(vec![b, g, g, d])?.permute(&[0, 2, 1, 3])?.reshape(vec![b, g * g, d])
}
```

Tile-mixer loop (used in T3): for layer `i` in `0..n_tile_layers`: if `alternate_axes && i % 2 == 1`, run
`transpose_grid → block.apply → transpose_grid`; else `block.apply`.

**Traps:**
- `permute` then `reshape` may need a contiguous copy; that is the cost §2.1 mentions. Do not try to avoid it by hand.
- Do not change `Mamba3Mixer` itself.

**Tests** (`tests/planner.rs`):
1. `transpose_grid` twice is the identity (compare `to_f32()` vectors exactly) and once moves element `(y, x)` to `(x, y)`.
2. A `BiBlock` output has the input's shape and is finite.
3. **Bidirectionality:** change only the **last** token of the input; the output at token 0 must change
   (|Δ| > 1e-6). With a forward-only mixer it would not. This catches a wrongly built mixer.

**Done when:** tests pass on `cpu`.

---

### T3. The model forward

**Goal:** `TaskPlanner` with `forward` producing target logits and a `heads` function for the aux outputs.

**Fields:**

```rust
pub struct TaskPlanner<R: Runtime, E: FloatElem> {
    tile_in1: Linear<R, E>, tile_in2: Linear<R, E>,    // c_tile -> d -> d
    glob_in1: Linear<R, E>, glob_in2: Linear<R, E>,    // c_glob -> d -> d
    unit_in1: Linear<R, E>, unit_in2: Linear<R, E>,    // c_unit -> d -> d
    tile_pos: Param<R, E>,                              // [tiles, d], init normal(0, 0.02)
    step_emb: Param<R, E>,                              // [k, d],     init normal(0, 0.02)
    none_key: Param<R, E>,                              // [1, d],     init normal(0, 0.02)
    tile_blocks: Vec<BiBlock<R, E>>,
    joint_blocks: Vec<BiBlock<R, E>>,
    norm: RmsNorm<R, E>,
    q_proj: Linear<R, E>, k_proj: Linear<R, E>,         // d -> d
    aux: Linear<R, E>,                                  // 2d -> aux_width (op | opset | crop | eta)
    config: TaskPlannerConfig,
}

pub struct PlannerOutput<R: Runtime, E: FloatElem> {
    pub target_logits: Var<R, E>,  // [B, Q, tiles + 1]
    pub queries: Var<R, E>,        // [B, Q, d]   (H)
    pub keys_full: Var<R, E>,      // [B, tiles + 1, d]  ([T' ; none_key], before k_proj)
}
```

`Param` init: `Param::new(Tensor::from_f32(&rng.normal_vec(n * d, 0.0, 0.02), vec![n, d], device)?)`.

**Forward** (`pub fn forward(&self, tiles: &Var, glob: &Var, units: &Var, unit_onehot: &Tensor) -> Result<PlannerOutput>`),
exactly in this order:

```text
B = tiles.dim(0); d = d_model; N = tiles(); U = max_units; K = k; Q = U*K
1  t  = tile_in2(gelu(tile_in1(tiles)))                         [B,N,d]
2  t  = t + tile_pos.var(&t)            (broadcast [N,d] -> [B,N,d] via expand)
3  g  = glob_in2(gelu(glob_in1(glob)))                          [B,d]
4  t  = t + g.reshape([B,1,d]).expand([B,N,d])
5  tile mixer (T2 loop)                                          [B,N,d]
6  oh = Var::constant(unit_onehot)                               [B,U,N]   (padding rows are all zero)
7  here = oh.matmul(&t)                                          [B,U,d]
8  u  = unit_in2(gelu(unit_in1(units))) + here                   [B,U,d]
9  q  = u.reshape([B,U,1,d]).expand([B,U,K,d]) + step_emb.var(&u).reshape([1,1,K,d]).expand([B,U,K,d])
       .reshape([B,Q,d])
10 s  = cat(&[t, q], 1)                                          [B,N+Q,d]
11 for block in joint_blocks: s = block.apply(&s)
12 s  = norm(s)
13 t2 = s.slice(1, 0, N);  h = s.slice(1, N, Q)
14 keys_full = cat(&[t2, none_key.var(&t2).reshape([1,1,d]).expand([B,1,d])], 1)     [B,N+1,d]
15 logits = q_proj(h).matmul_nt(&k_proj(keys_full)).mul_scalar(1/sqrt(d))           [B,Q,N+1]
```

**Aux heads** (`pub fn heads(&self, out: &PlannerOutput, target_onehot: &Tensor) -> Result<Var>`):

```text
tt  = Var::constant(target_onehot).matmul(&out.keys_full)       [B,Q,d]   (target_onehot is [B,Q,N+1])
aux = self.aux.apply(&cat(&[out.queries.clone(), tt], 2)?)       [B,Q,aux_width]
```

Split `aux` along axis 2 with `split(&[n_ops, n_ops, n_crops, 1], 2)` **only in the loss / on the host**, not in the model.

**Traps:**
- Padding units (upos = -1) have an all-zero one-hot row, so `here` is 0 for them; they still enter the joint sequence. The
  bidirectional scan is **not** invariant to padding tokens (a zero input still advances the SSM state). Therefore **always
  pad to exactly `max_units = 20`** in training and in inference. Never trim the batch to the units present.
- `matmul_nt` is the fast `A @ Bᵀ` (`ops.rs:331`); use it instead of `transpose()` + `matmul`.
- Do not add dropout in this plan (the reference uses 0.1; add it only if T7 shows over-fitting).

**Tests:**
1. Shapes: `target_logits [2, 60, 101]`, `heads(...) [2, 60, 32]` for a random batch of 2.
2. **Gradient reaches every parameter**: build a scalar `logits.sum() + aux.sum()`, backward, and assert every entry of
   `model.named_parameters()` has a gradient with a non-zero norm (copy the pattern of
   `tests/rl.rs:420 a_trajectory_pass_is_differentiable_and_reaches_every_parameter`).
3. `check_grad` (`tests/autograd.rs:18`) on a **tiny** config (`d_model 8`, 1 + 1 layers, grid 2, max_units 2, k 2) for
   `tile_pos`, `none_key` and one `Linear` weight.
4. Unit one-hot gather equals direct indexing: for random `t`, `oh.matmul(t)[b,u]` equals `t[b, upos[b,u]]` (tolerance 1e-5).

**Done when:** all pass on `cpu`.

---

### T4. The batch and the loss (no host reads)

**Goal:** `PlannerBatch` and `PlannerTask: TrainStep`, with every mask built on the host (rule S1).

```rust
pub struct PlannerBatch<R: Runtime, E: FloatElem> {
    pub tiles: Tensor<R, E>,           // [B, N, c_tile]
    pub glob: Tensor<R, E>,            // [B, c_glob]
    pub units: Tensor<R, E>,           // [B, U, c_unit]
    pub unit_onehot: Tensor<R, E>,     // [B, U, N]         built on the host (see below)
    pub target_ids: IdTensor<R>,       // [B*Q]             ignore -> 0
    pub target_keep: Tensor<R, E>,     // [B*Q]             1 keep, 0 ignore; pre-multiplied by the step weight (1, .5, .5)
    pub target_onehot: Tensor<R, E>,   // [B, Q, N+1]       one-hot of the TRUE target (teacher forcing); zero row if ignored
    pub op_ids: IdTensor<R>, pub op_keep: Tensor<R, E>,        // [B*Q]; keep where 0 <= tgt < N
    pub crop_ids: IdTensor<R>, pub crop_keep: Tensor<R, E>,    // [B*Q]; keep where crop >= 0
    pub opset: Tensor<R, E>, pub opset_keep: Tensor<R, E>,     // [B*Q, n_ops] targets; [B*Q] keep = op_keep
    pub eta_log: Tensor<R, E>, pub eta_keep: Tensor<R, E>,     // [B*Q] log1p(eta) on step 0, keep only step 0 with eta >= 0
    pub weights: [f32; 5],             // divisors precomputed on the host: sum of each keep mask (clamped to >= 1)
}
```

Build it with **one host function** `PlannerBatch::from_host(cfg, &HostBatch, device)` where `HostBatch` holds plain
`Vec<f32>` / `Vec<i32>` slices in the npz layout of §1.2. Compute every keep mask and every divisor on the host.
`unit_onehot` and `target_onehot` are built on the host here (1 MB and 3 MB per batch of 128). **This host batch is the
composed oracle only** (tests, parity checks, inference on a single live observation). The training path never builds a
one-hot or a host mask: K1 assembles the batch on the device and K2 gathers by index.

**Loss** (`impl TrainStep for PlannerTask { fn loss(&self, b) }`):

```text
out    = model.forward(tiles, glob, units, unit_onehot)
ce_t   = out.target_logits.reshape([B*Q, N+1]).cross_entropy_rows(&target_ids, 0.0)    [B*Q]
L_t    = (ce_t * target_keep).sum() * (1 / weights[0])
aux    = model.heads(&out, &target_onehot).reshape([B*Q, aux_width])
[op, os, cr, eta] = aux.split(&[n_ops, n_ops, n_crops, 1], 1)
L_op   = (op.cross_entropy_rows(&op_ids, 0.0)   * op_keep).sum()   * (1 / weights[1])
L_cr   = (cr.cross_entropy_rows(&crop_ids, 0.0) * crop_keep).sum() * (1 / weights[2])
bce    = os.softplus() - os * opset                         (BCE with logits, elementwise)  [B*Q, n_ops]
L_os   = (bce.sum_dim(1).reshape([B*Q]) * opset_keep).sum() * (1 / (weights[1] * n_ops))
         (sum_dim KEEPS the axis with size 1, ops.rs:815, hence the reshape)
diff   = eta.reshape([B*Q]) - eta_log
L_eta  = (diff * diff * eta_keep).sum() * (1 / weights[4])  (MSE on log1p; the reference uses smooth-L1, MSE is fine)
loss   = (L_t + L_op + 0.3 * L_os + 0.3 * L_cr + 0.1 * L_eta) * loss_scale      (loss_scale from the task, §2.5; 1.0 in f32)
```

Multiplying by `1 / weights[i]` uses `mul_scalar` with a host `f32`: no device read.

**Traps:**
- `cross_entropy_rows` must never see an id ≥ number of classes: ignored rows get id **0** and keep 0.
- `sum_dim(axis)` keeps the axis with size 1 (`ops.rs:815`): always reshape after it.
- Do **not** call `cross_entropy_with` / `ignore_index` (host read, rule S1).

**Tests:**
1. On a hand-made batch of 1 turn, 2 units, compare each partial loss with a straightforward host computation in `f64`
   (tolerance 1e-4).
2. `read_count()` before and after one `loss()` + backward: **the difference must be 0**.
3. 200 optimizer steps on one fixed random batch drive the loss below 10% of its start (the model can over-fit).
4. **Loss-scale invariance (f32):** two identical models, one trained 20 steps with `loss_scale 1` (eps 1e-8, clip 1.0), one
   with `loss_scale 1024` (eps 1.024e-5, clip 1024). All parameters must agree within 1e-5 relative. This proves §2.5's
   scaling of eps and the clip is right before f16 is involved.
5. **f16 smoke (run on `vulkan` only, skip on backends where `supports_matmul_precision(F16)` is false):** 50 steps with
   F16 and `loss_scale 1024` on the fixed batch give finite losses that decrease; reset the precision to F32 at the end of the
   test (it is a process-global mode).

**Done when:** all pass on `cpu`.

---

### T5. Checkpoint and inference entry points

**Goal:** save / load, and a no-grad prediction call.

- `pub fn save(&self, path) -> Result<()>`: `Checkpoint::capture(self, step).with_metadata(serde_json::to_value(&self.config)?)
  .save(path)`.
- `pub fn load(path, device) -> Result<Self>`: read the checkpoint, rebuild the config from its metadata, `init`, `restore`.
- `pub fn predict(&self, batch) -> Result<(Tensor /*logits*/, Tensor /*aux at the argmax target*/)>`: under
  `crate::autograd::no_grad()`; runs `forward`, then **on the device**:
  `let ids = crate::tensor::ops::reduce::argmax(out.target_logits.tensor(), 2)?;` (`reduce.rs:378`, returns an `IdTensor`
  `[B,Q]`). Composed path: `let oh = crate::tensor::ops::index::one_hot::<R, E>(&ids, N + 1)?;` (`index.rs:497`), then
  `heads(&out, &oh)`. Fused path (after K2): `gather_tokens(keys_full, ids)` instead of the one-hot — `heads` takes either an
  id tensor or a one-hot, chosen by `fused_planner_enabled()`. One forward, no host round trip, no second call. Also provide `predict_aux(batch, target_onehot)` for teacher-forced
  aux outputs at given targets (the evaluator's "op at the true target").

**Test:** save, load, and `predict` gives bit-identical logits on `cpu`.

---

---

### On-device kernels (K0–K6): the primary path

T1–T5 build the model **composed** from existing ops. That path stays forever as the **reference (oracle)** that the tests
compare against, but it is not the path that trains. K0–K6 move every hot or host-side piece onto the GPU as CubeCL kernels:
the batch is assembled on the device from a device-resident dataset, the gathers and the loss are single kernels with
hand-written adjoints, and nothing is built on the host per step.

Read first: `ENTITY_KERNEL_PLAN.md` (the same approach, landed for the entity path), and in the CubeCL manual
(`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/`): `03_kernel_fusion.md`, `11_launch_overhead_and_transfers.md`,
`07_memory_coalescing.md`, `08_atomic_contention.md`.

#### Kernel templates to copy (do not invent a new style)

| Need | Copy from |
|---|---|
| forward kernel with vector lanes, `launch_unchecked`, `ABSOLUTE_POS * span` loop | `pointer_dot_kernel`, `src/tensor/ops/entity.rs:1283-1330` |
| host wrapper: shape checks, `Tensor::empty`, `line_dividing`, `launch_1d_spans`, `unsafe { ..::launch_unchecked }` | `pointer_dot`, `entity.rs:1336-1420` |
| gather-style adjoint (no atomics) | `pointer_dot_de_kernel` / `pointer_dot_backward_de`, `entity.rs:1426-1530` |
| `line_dividing` (vector width that divides every extent) | `entity.rs:46` |
| recording a fused op on the tape (`Var::record` / `record_with_mask`, `rule!`) | `Var::pointer_dot`, `src/autograd/ops.rs:1618-1690` |
| process-global on/off switch read once from an env var | `fused_split_enabled` / `set_fused_split`, `src/tensor/ops/movement.rs:495-525` |
| launch attribution per source line | `start_launch_tally` / `launch_tally`, `src/backend.rs:446-480`; `examples/profile_entity.rs` |
| parity tests in their own binary (the switch is process-global) | `tests/entity_kernels.rs`, `tests/rl_entity_parity.rs` |
| launch-count / read-count pinning | `tests/rl_entity_footprint.rs` |

Put all planner kernels in **one new module** `src/tensor/ops/planner.rs` (register it in `src/tensor/ops/mod.rs` next to
`entity`), and their `Var` wrappers in `src/autograd/ops.rs` next to `pointer_dot`.

#### Invariants (every K task must keep all of them)

1. **The composed path stays and stays correct.** `set_fused_planner(false)` (K0) restores the T1–T5 code exactly.
2. **Forward parity:** fused vs composed within 1e-6 relative (f32), on `cpu` and `vulkan`.
3. **Gradient parity:** every adjoint passes `check_grad` (`tests/autograd.rs:18`) and matches the composed gradient within
   1e-5.
4. **No host reads, no atomics.** Every adjoint is a *gather*: each output element loops over what feeds it. Never a scatter.
5. **Accumulate in f32 inside the kernel** (`f32::cast_from`, as `pointer_dot_kernel` does), whatever the storage type.
   f16 affects only matmul operands (§2.5), never these kernels.
6. **Ids are `u32` with one sentinel**, `IGNORE = u32::MAX`. Kernels test for it; they never index with it.
7. **Padding stays fixed:** always `U = 20`, `K = 3`, `N = 100` (T3 trap). Kernels may assume these are runtime `usize`
   arguments, not comptime constants.

#### Status table (fill in as you go)

| ID | Kernel(s) | Replaces | Expected launches saved per train step | Status |
|---|---|---|---|---|
| K0 | switch, parity harness, profile | – | 0 | |
| K1 | `planner_gather_inputs`, `planner_labels` | host batch build + ~9 uploads per step | all per-step uploads | |
| K2 | `planner_queries` (+ adjoint), `gather_tokens` (+ adjoint) | one-hot upload, 2 matmuls, expands and adds (fwd + bwd) | ≈ 15–20 | |
| K3 | `planner_loss_rows`, `planner_loss_reduce`, `planner_loss_backward` | ≈ 40 composed loss launches (fwd + bwd) | ≈ 35 | |
| K4 | `grid_transpose` | reshape/permute/reshape copies in alternated tile layers | ≈ 2 per alternated layer (fwd + bwd) | |
| K5 | `tile_embed_join` (+ adjoint), `keys_with_none` (+ adjoint) | broadcast adds, expands, `cat` | ≈ 10 | |
| K6 | measure, pin counts, write down | – | – | |

The counts are estimates from reading the compositions. K6 replaces them with the launch tally.

---

### K0. Switch, parity harness, profile

**Goal:** make every later K task measurable and reversible before any kernel exists.

**Do:**
- In `src/models/planner.rs`: `pub fn set_fused_planner(on: bool)` and `pub(crate) fn fused_planner_enabled() -> bool`,
  copied from `movement.rs:495-525` with env var `MAMBA3_FUSED_PLANNER` (`"0"` = off; default on). Until a K task lands,
  every call site still takes the composed path.
- Re-export `set_fused_planner` from `mamba3::models`. Python: `mamba3_rl.set_fused_planner(on)` next to
  `set_fused_entity` in `bindings/python/src/lib.rs`, plus the `.pyi` stub.
- `tests/planner_kernels.rs` (new binary): for each kernel, run the op fused and composed on the same tie-free random data
  and compare forward values and every input's gradient (invariants 2, 3). Start with an empty test list.
- `examples/profile_planner.rs` (this is also T8's tool): one training step of the default config at batch 128, printing
  `launch_count()`, `read_count()` and the `launch_tally()` per source line, **in both modes**, forward / backward / update
  separately. Register it in `Cargo.toml` as an `[[example]]` with `required-features = ["backend"]`.
- `examples/bench_planner.rs`: interleaved in-process A/B (copy `examples/bench_split.rs`): alternate
  `set_fused_planner(true/false)` every iteration, 100 iterations each, report the median step time per mode and the ratio.
  Interleaving in one process is required: run-to-run wall noise is ±20%.

**Commands:**
```bash
cargo test --release --no-default-features --features cpu --test planner_kernels > /tmp/k.log 2>&1; echo "exit=$?"
cargo run  --release --no-default-features --features cpu --example profile_planner > /tmp/p.log 2>&1; echo "exit=$?"
```

**Done when:** both exit 0 and `/tmp/p.log` shows the composed baseline (launches, reads = 0). Save that table in the status
table above as "before".

---

### K1. Device-resident dataset and on-device batch assembly

**Goal:** after one upload per split, a training step moves **no data from the host**. The batch is gathered and its labels,
keep masks and ids are built by two kernels.

**Rust types** (in `src/models/planner.rs`):

```rust
/// One split of the data, uploaded once.
pub struct PlannerData<R: Runtime, E: FloatElem> {
    pub turns: usize,
    pub floats: Tensor<R, E>,   // [turns, F]  F = N*c_tile + c_glob + U*c_unit, row = tiles | glob | units (flattened)
    pub labels: IdTensor<R>,    // [turns, L]  L = U + 3*Q + U + Q   (see the layout below), u32
}
```

Label row layout (`u32`, `Q = U*K`), written on the host **once** when the split is uploaded:

| Offset | Length | Content | Sentinel |
|---|---|---|---|
| 0 | U | `upos` (tile 0..99) | `IGNORE` for padding |
| U | Q | `tgt` (0..99 tile, 100 NONE) | `IGNORE` for -100 |
| U + Q | Q | `op` (0..12) | `IGNORE` |
| U + 2Q | Q | `crop` (0..4) | `IGNORE` |
| U + 3Q | U | `eta` (turns ≥ 0) | `IGNORE` for -1 |
| 2U + 3Q | Q | `opset` as a 13-bit mask (bit i = op i present) | 0 |

`PlannerData::from_host(cfg, floats: &[f32], labels: &[u32], turns, device)`: validates lengths and every id range on the
host (T6's rules), then two uploads. Memory: train = 53,925 × (4,800 + 114 + 720) × 4 B ≈ 1.2 GB + labels 53,925 × 280 × 4 B
≈ 60 MB. Fits the 16 GB device.

**Kernel `planner_gather_inputs`** (one launch): inputs `floats [turns, F]`, `turn_ids [B]` (IdTensor). Output one contiguous
buffer `[B, F]`, then split **without copies** into `tiles [B,N,c_tile]`, `glob [B,c_glob]`, `units [B,U,c_unit]` by
`slice` + `reshape` views if the tensor layer allows zero-copy slices; otherwise make the kernel write three outputs
(one launch, three `&mut Array` arguments, lanes = `B*F`, each lane decides which output it writes by its column).
Vectorised with `line_dividing(&[F, N*c_tile, c_glob, U*c_unit])`; if that width is 1, it is still one launch.
No adjoint (data are constants).

**Kernel `planner_labels`** (one launch): inputs `labels [turns, L]`, `turn_ids [B]`, `step_w [K]` (host constant, uploaded
once at construction: `[1.0, 0.5, 0.5]`). One thread per `(b, q)` for `q < Q`, plus one per `(b, u)` for `u < U`. Outputs:

| Output | Shape | Value |
|---|---|---|
| `unit_ids` | `[B*U]` u32 | `upos`, or `IGNORE` |
| `target_ids` | `[B*Q]` u32 | `tgt`, or 0 where ignored |
| `target_w` | `[B*Q]` f32 | `step_w[q % K]` where `tgt != IGNORE`, else 0 |
| `op_ids`, `op_w` | `[B*Q]` | `op` (0 if ignored), weight 1 where `tgt < N` and `op != IGNORE`, else 0 |
| `crop_ids`, `crop_w` | `[B*Q]` | `crop` (0 if ignored), weight 1 where `crop != IGNORE` |
| `opset` | `[B*Q, 13]` f32 | bit i of the mask as 0/1 (13 writes per thread) |
| `eta_log`, `eta_w` | `[B*Q]` f32 | for `q % K == 0`: `ln(1 + eta)` and 1 where `eta != IGNORE`; else 0 and 0 |

Denominators are **not** computed here: K3 sums the weights on the device as part of the loss.

**`PlannerBatch` changes:** add a constructor `PlannerBatch::gather(&data, &turn_ids, &model_consts) -> Result<Self>` that
runs the two kernels. The host constructor `from_host` (T4) stays for tests and for the composed path.

**Tests** (`tests/planner_kernels.rs`):
1. For 3 random turn-id batches, `PlannerBatch::gather` equals `PlannerBatch::from_host` on the same turns **element by
   element** (floats exactly equal; ids equal; weights equal).
2. Ids never exceed their class count (read back in the test only).
3. `read_count()` delta over 10 × `gather` = 0; `launch_count()` delta = 20 (2 per gather).

**Done when:** tests pass on `cpu` and `vulkan`.

---

### K2. Token gathers and the query build

**Goal:** replace the one-hot matmul gathers (T3 steps 6–9 and the aux-head target gather) with index gathers on the device,
forward and backward, without atomics.

**Kernel `planner_queries`** — forward (one launch):
```text
inputs : um [B,U,d] (unit MLP output), t [B,N,d] (tile tokens), step [K,d] (Param), unit_ids [B*U] u32
output : q [B,U*K,d]
q[b, u*K + k, :] = um[b,u,:] + (unit_ids[b*U+u] == IGNORE ? 0 : t[b, unit_ids[b*U+u], :]) + step[k,:]
threads: one per (b, u, k, d-vector)
```
Adjoint — **three gathers in one launch** (lanes split into three regions, as `pointer_dot_de_kernel` splits `d_e` / `d_extra`):
```text
d_um[b,u,:]  = Σ_k g[b, u*K+k, :]                                   region 1: one thread per (b,u,dvec), loop k < K
d_t[b,n,:]   = Σ_{u : unit_ids[b*U+u] == n} Σ_k g[b, u*K+k, :]      region 2: one thread per (b,n,dvec), loop u < U (=20)
d_step[k,:]  = Σ_{b,u} g[b, u*K+k, :]                               region 3: one thread per (k,dvec), loop b,u (B*U = 2,560)
```
Region 3 is a long serial loop per thread (2,560 iterations); that is fine at `K*d/line = 3*128/4 = 96` threads, and it keeps
the kernel atomic-free. If K6 shows region 3 dominating the kernel, split it into a two-pass reduction (partial sums over
blocks of 64 `b`, then a second tiny kernel), still without atomics.

**Kernel `gather_tokens`** — generic, used for the aux-head target token (T3 "target token"):
```text
forward : out[b, r, :] = ids[b*R + r] == IGNORE ? 0 : src[b, ids[b*R + r], :]       src [B,S,d], ids [B*R], out [B,R,d]
adjoint : d_src[b, s, :] = Σ_{r : ids[b*R + r] == s} g[b, r, :]                       one thread per (b,s,dvec), loop r < R
```
Here `src = keys_full [B, N+1, d]`, `R = Q = 60`, ids = `target_ids` in training (teacher forcing) and the device argmax in
`predict` (`reduce::argmax`, `reduce.rs:378`, already returns an `IdTensor`). This removes the `[B,Q,N+1]` one-hot entirely.

**`Var` wrappers** (`src/autograd/ops.rs`): `Var::planner_queries(um, t, step, unit_ids) -> Var` recording one rule with three
parent gradients; `Var::gather_tokens(src, ids, r) -> Var`. Follow `Var::pointer_dot` (`ops.rs:1618`), including
`record_with_mask` so an unneeded gradient region is not computed.

**Wire:** in `TaskPlanner::forward` / `heads`, `if fused_planner_enabled()` call these; else the composed T3 code.

**Tests:** forward parity, gradient parity and `check_grad` for both ops, with some `IGNORE` ids and with two units on the same
tile (the case a scatter would get wrong). Launch delta: `planner_queries` 1 forward + 1 backward; `gather_tokens` 1 + 1.

---

### K3. The fused loss

**Goal:** the whole T4 loss — three masked cross-entropies, the masked BCE, the masked eta MSE, the five normalisations, the
weighted sum and the loss scale — in **two forward launches and one backward launch**, with no host involvement.

**Kernel `planner_loss_rows`** (forward, pass 1): one thread per row `r` of `B*Q`:
```text
lt   = target_logits[r, 0..N+1];  ax = aux[r, 0..32]  (op 0..13 | opset 13..26 | crop 26..31 | eta 31)
ce_t = logsumexp(lt) - lt[target_ids[r]]                 (max-shifted, f32)
ce_o = logsumexp(ax[0..13]) - ax[op_ids[r]]
ce_c = logsumexp(ax[26..31]) - ax[26 + crop_ids[r]]
bce  = Σ_i softplus(ax[13+i]) - ax[13+i]*opset[r,i]      (softplus(x) = max(x,0) + ln(1 + e^-|x|))
se   = (ax[31] - eta_log[r])²
rows_out[r, 0..10] = [target_w*ce_t, op_w*ce_o, crop_w*ce_c, op_w*bce, eta_w*se,
                      target_w, op_w, crop_w, op_w, eta_w]
lse_out[r, 0..3]   = [lse_t, lse_o, lse_c]               (saved for the backward pass)
```
**Kernel `planner_loss_reduce`** (forward, pass 2): a single cube reduces `rows_out [B*Q, 10]` over rows into `sums [10]`
(each of 10 threads loops over 7,680 rows serially — or a plane reduction if you know CubeCL's plane ops; serial is correct
and fast enough here), then writes
```text
den_i  = max(sums[5+i], 1)                    i = 0..4   (the opset term also divides by n_ops = 13)
loss   = S * ( sums[0]/den_0 + sums[1]/den_1 + 0.3*sums[3]/(den_3*13) + 0.3*sums[2]/den_2 + 0.1*sums[4]/den_4 )
out    = [loss, 5 unscaled component losses, 5 denominators]   ([11] f32; the first element is the scalar loss)
```
The components are what `read_losses()` reports (they are read with the queued step, so no extra read).

**Kernel `planner_loss_backward`** (one launch): one thread per `(r, column)` over the `N+1` target columns and the 32 aux
columns (two regions), using the saved `lse` and `den`, the upstream scalar gradient `g`:
```text
d_lt[r,c]     = g*S * target_w[r]/den_0 * (exp(lt[r,c] - lse_t[r]) - [c == target_ids[r]])
d_ax[r,i<13]  = g*S * op_w[r]/den_1     * (exp(ax[r,i] - lse_o[r]) - [i == op_ids[r]])
d_ax[r,13+i]  = g*S * 0.3*op_w[r]/(den_3*13) * (sigmoid(ax[r,13+i]) - opset[r,i])
d_ax[r,26+j]  = g*S * 0.3*crop_w[r]/den_2 * (exp(ax[r,26+j] - lse_c[r]) - [j == crop_ids[r]])
d_ax[r,31]    = g*S * 0.1*eta_w[r]/den_4 * 2*(ax[r,31] - eta_log[r])
```
Every output element depends only on its own row and the saved scalars: a pure gather, no atomics.

**`Var` wrapper:** `Var::planner_loss(target_logits, aux, &labels, loss_scale) -> Result<(Var /*scalar*/, Tensor /*[11]*/)>`;
the second value is off the tape (for logging). `PlannerTask::loss` returns the scalar when fused.

**Tests:**
1. Forward: fused scalar and all five components equal the composed T4 loss within 1e-6 relative (f32), with ignored rows,
   rows where only the target is kept, and all-ignored crop (den clamps to 1).
2. Gradient parity and `check_grad` for `target_logits` and `aux`.
3. Numerical safety: logits of ±80 (no overflow in exp: max-shifted), a whole batch with `op_w = 0` (no division by 0).
4. Launches: forward 2, backward 1; reads 0.

---

### K4. Grid transpose

**Goal:** the alternated tile layers (T2) transpose `[B, g*g, d]` with one launch each way instead of reshape + permute +
reshape copies.

**Kernel `grid_transpose`**: `out[b, x*g + y, :] = in[b, y*g + x, :]`, one thread per `(b, cell, dvec)`. The transpose is its
own inverse, so the adjoint is the same kernel applied to the upstream gradient (one launch). `Var::grid_transpose(x, g)`.

**Tests:** parity with `transpose_grid` (T2), applied twice is the identity (exact), `check_grad`, 1 launch forward and
1 backward.

---

### K5. Tile embedding join and the NONE key

**Kernel `tile_embed_join`** (one launch forward): `t[b,n,:] = x[b,n,:] + pos[n,:] + g[b,:]` (x = tile MLP output, pos =
`tile_pos`, g = global MLP output). Adjoint in one launch with two regions (gathers):
`d_pos[n,:] = Σ_b grad[b,n,:]` (loop over b < B) and `d_g[b,:] = Σ_n grad[b,n,:]` (loop over n < N); `d_x = grad` needs no
launch (return the upstream gradient tensor itself).

**Kernel `keys_with_none`** (one launch forward): `keys[b, n, :] = n < N ? t2[b,n,:] : none[0,:]` → `[B, N+1, d]`. Adjoint in
one launch: `d_t2[b,n,:] = grad[b,n,:]` for n < N (copy region) and `d_none[:] = Σ_b grad[b,N,:]` (gather region).

**Tests:** parity, `check_grad`, launch counts (1 + 1 each).

---

### K6. Measure, pin, write down

1. Run `examples/profile_planner.rs` in both modes on `cpu` and `vulkan`; fill the status table's "after" column with the
   measured launches per step (forward / backward / update) and reads (must be 0).
2. Run `examples/bench_planner.rs` on `vulkan`: median step time per mode and the ratio.
3. `tests/planner_footprint.rs`: pin the fused launch count per train step and reads = 0 (copy
   `tests/rl_entity_footprint.rs`), so a later change that adds launches fails a test.
4. If a kernel shows up as the single most expensive line in the tally, note it; do **not** optimise it further in this plan.

**Done when:** the fused path is the default, all parity tests pass on both backends, and the table is filled in.

---

### T6. Python bindings

**Goal:** a Python API with parity to the Rust one.

**Files:** new `bindings/python/src/planner.rs`; register in `bindings/python/src/lib.rs` (`mod planner;` and
`module.add_class::<planner::PyTaskPlannerConfig>()?; module.add_class::<planner::PyTaskPlanner>()?;`); stubs in
`_mamba3_rl.pyi`; tests in `bindings/python/tests/test_planner.py`.

**Python API** (all arrays are numpy, shapes as in §1.2 with a leading batch axis):

```python
cfg = mamba3_rl.TaskPlannerConfig(d_model=128, n_tile_layers=3, n_joint_layers=3, alternate_axes=True, chunk_size=32,
                                  seed=0)                       # other fields default to §2.2
model = mamba3_rl.TaskPlanner(cfg, learning_rate=3e-4, weight_decay=0.05, max_grad_norm=1.0,
                              lr_schedule=mamba3_rl.LrSchedule.cosine(...), matmul_precision="f16", loss_scale=1024.0)
train = mamba3_rl.PlannerData(tiles, glob, units, upos, tgt, op, crop, opset, eta)   # a whole split, uploaded ONCE (K1)
model.queue_train_step(train, turn_ids)   # turn_ids: int array [B]; the batch is gathered on the device; nothing read back
losses = model.read_losses()      # [(loss, grad_norm, [5 component losses])] for the steps queued since the last read
model.queue_train_step_host(tiles, glob, units, upos, tgt, op, crop, opset, eta)   # host batch; tests / oracle only
out = model.predict(tiles, glob, units, upos)   # dict: "target_logits" [B,20,3,101] float32,
                                                #       aux at the argmax target: "op" [B,20,3,13], "opset" [B,20,3,13],
                                                #       "crop" [B,20,3,5], "eta" [B,20]
aux = model.predict_aux(tiles, glob, units, upos, tgt)   # same aux dict, at the given targets (teacher forcing)
model.save(path); model2 = mamba3_rl.TaskPlanner.load(path)
```

Implementation notes:
- `#[pyclass(module = "mamba3_rl", name = "TaskPlanner", unsendable)]`, holding the model, the `Trainer<R, E, AdamW>` and a
  `Vec<QueuedStep>`; copy the structure of `policy.rs:50-205`.
- `PlannerData.__init__` converts the numpy split once (float16 → float32, labels → the K1 `u32` layout with `IGNORE`),
  validates shapes and every id range on the host with clear `ValueError`s (see `array.rs:143`) **before** anything reaches
  the device (an out-of-range id has no bounds check on the device), then uploads two buffers.
- `queue_train_step(data, turn_ids)` validates `0 <= id < data.turns`, uploads the `[B]` ids (the only per-step transfer)
  and runs `PlannerBatch::gather` (K1) + the fused forward / loss (K2–K5).
- `predict` / `predict_aux` also accept a `PlannerData` + turn ids (evaluation on dev16 / test40 without re-uploading), in
  addition to host arrays (inference on a live observation).
- `set_fused_planner(on)` is exposed as a module function (K0); the Python tests run both modes.
- `queue_train_step` calls `Trainer::queue_step`; `read_losses` calls `read_steps` on everything queued (rule S3).
- `matmul_precision` accepts `"f32"` (default) or `"f16"`; anything else (including `"bf16"`) raises `ValueError`
  ("bf16 is not supported by this binding"). Call `try_set_matmul_precision`; on refusal raise `ValueError` with the backend
  name. Note the precision is a **process-global** mode (`matmul.rs:129`): set it once in `__init__` and document that two
  planners with different precisions in one process share the last setting.
- `loss_scale` (float, default `1.0`; use `1024.0` with f16) is applied as in §2.5: the task multiplies the loss by it, the
  AdamW `eps` and the trainer's `max_grad_norm` are multiplied by it at construction, and `read_losses()` divides the loss and
  the gradient norm by it before returning them. `read_losses()` returns `(loss, grad_norm)` pairs; if either is not finite,
  it raises `FloatingPointError` naming the step, so the caller can stop (see §2.5).
- `predict` returns float32 numpy arrays reshaped to `[B, U, K, ...]` (the query axis is unit-major, `q = u*K + j`).

**Python tests** (`test_planner.py`, run with the CPU wheel):
1. Construct, `predict` on random inputs: shapes and finiteness.
2. 100 queued steps on a fixed batch, `read_losses()` returns 100 floats, last < first / 2.
3. `save` / `load` round trip gives identical `predict` output.
4. Bad input (wrong shape, `tgt` = 101, `op` = 13) raises `ValueError` and does not crash.

**Commands:**

```bash
# CPU wheel for tests (Kaggriculture helper builds and installs into its venv)
cd /home/user/Documents/workspace/Kaggriculture && bash tools/setup_mamba3.sh cpu
cd /home/user/Documents/workspace/mamba-trainer/bindings/python && <kaggriculture python> -m pytest tests/test_planner.py -q
```

**Done when:** Rust suite green, Python tests green, `.pyi` updated.

---

### T7. Kaggriculture training script (CPU smoke, then Vulkan)

**Goal:** train on the real data and score with the existing evaluator.

**Files (Kaggriculture repo):** `experiments/kobayashi/exp-planner053_dsm_task_planner/src/train_mamba.py` and
`src/mamba_adapter.py`.

- `train_mamba.py`: load `train.npz` / `dev16.npz` with numpy, build `PlannerData` **once per split** (K1), shuffle turn
  indices each epoch on the host (a `[turns]` int array), batch 128, `queue_train_step(train, ids)` per batch,
  `read_losses()` every 50 steps, dev evaluation after each epoch with `predict(dev, ids)` (batch 512), keep the best dev
  next-visit top-1 checkpoint, log JSON lines like `train.py`. No per-batch numpy work besides slicing the id array.
- `mamba_adapter.py`: a class with the same `predict(model, d)` contract as `train.predict` in `src/train.py` (returns
  `{"logits", "op", "op_pred", "eta"}` as torch CPU tensors or numpy), so that `evaluate.py` can score the Mamba model by
  swapping one import. `predict` already returns aux at the **predicted** target (device argmax, T5); `op` at the **true**
  target comes from `predict_aux` with the labels' one-hot.

**Commands:**

```bash
cd /home/user/Documents/workspace/Kaggriculture && bash tools/setup_mamba3.sh vulkan
cd experiments/kobayashi/exp-planner053_dsm_task_planner
python src/train_mamba.py --data runs/20260926_task_planner/data --out runs/<date>_mamba_planner/model_s0 --epochs 1 --limit 2000   # smoke
python src/train_mamba.py --data runs/20260926_task_planner/data --out runs/<date>_mamba_planner/model_s0 --epochs 14
python src/evaluate.py ... --set dev16   # with the adapter
```

**Done when:** the smoke run finishes without error on Vulkan and dev top-1 after 1 epoch is ≥ 70% (the reference reached
77.2%). If < 60%, stop: something is wired wrong (check T3 test 4 and the unit-major query order first).

---

### T8. Measurement gate: speed and launches (do this before the 14-epoch run)

**Goal:** prove the speed design of §2.3.

**Files:** `examples/profile_planner.rs`, `examples/bench_planner.rs` (from K0).

`examples/profile_planner.rs` and `examples/bench_planner.rs` were created in K0. For T8 they must print, for batch 128, the
default config and **both** `set_fused_planner` modes, averaged over 20 steps after 5 warm-up steps:
- ms per optimizer step (forward + backward + update), and split forward / backward / update;
- `launch_count()` per step, split the same way;
- `read_count()` per step (**must be 0**);
- the same with `alternate_axes = false`, with `chunk_size` 20 / 25 / 32 / 64, and with `n_tile_layers` 0 (joint only).
- the same with `MatmulPrecision::F32` vs `F16` (f16 with `loss_scale` 1024), and the max `|x|` of the activations and
  gradients in f16 (overflow margin).

**Commands:**

```bash
cargo run --release --no-default-features --features cpu    --example profile_planner > /tmp/p_cpu.log 2>&1; echo "exit=$?"
cargo run --release --no-default-features --features vulkan --example profile_planner > /tmp/p_vk.log  2>&1; echo "exit=$?"
```

**Pass criteria (Vulkan):**
- ≤ 0.25 s per step at batch 128 **on the fused path** (reference: 0.56 s), and the fused path faster than the composed
  path in the interleaved A/B (`bench_planner`);
- 0 reads per step;
- f16 vs f32: the §2.5 A/B (accuracy within 0.3 pp, no non-finite step, ≥ 15% faster). If it fails, train in f32 and
  re-check the 0.25 s criterion in f32;
- `alternate_axes` adds ≤ 10% time. If more, keep it only if it gains ≥ 1 pp dev top-1 in a 3-epoch A/B (T7 script with
  `--alternate-axes 0/1`).

**If a criterion fails:** report the profile table to the owner before changing kernels. Likely causes, in order:
(1) a hidden read (search for `to_vec`, `to_f32`, `scalar()` in the new code); (2) a K kernel that is slower than its
composed version (compare per-line times in the tally; a long serial loop, e.g. K2 region 3, is the usual suspect);
(3) padding in `ssd_chunked` (change `chunk_size`); (4) the mixers themselves (sweep `d_state`, `head_dim`).

---

### T9. (Merged into K1)

The device-resident dataset and on-device batch assembly are no longer optional: they are K1.

---

### T10. Full run and report

1. 14 epochs on Vulkan, seed 0 (T7 command). Then seeds 1 and 2 if seed 0 is within 2 pp of the reference.
2. Score dev16 with `evaluate.py` through the adapter; **score test40 once**, with the checkpoint chosen on dev16.
3. Write `Kaggriculture/experiments/kobayashi/exp-planner053_dsm_task_planner/runs/<date>_mamba_planner/REPORT.en.md` with:
   dev/test next-visit top-1 / top-3, walk top-1, op at true target, plan steps 2–3, by-status table, permutation
   importance, **ms per step and epoch time vs the PyTorch reference**, launches per step, and the parameter count.
4. Commit in this repo only the files this plan names (see §0.3).

---

## 5. Things that are deliberately out of scope

- Attention layers, cross-attention and key-padding masks. If the all-Mamba model is > 2 pp below the reference after T10,
  the next plan adds `MultiHeadAttention::apply_cross(query, memory, key_bias)` (a copy of `apply_cached` at
  `attention.rs:272` with separate query and key/value sources and an additive `[B,1,1,Tk]` bias) and interleaves one
  cross-attention layer per joint block. Not in this plan.
- Recurrence across turns (carrying the SSM state from turn to turn). The planner reads one observation. A sequence version
  (turns as time, `Mamba3Mixer::apply_with_state`) is a later plan.
- HIP debugging and changes to `rl::policy`.
- Custom kernels for the Mamba mixer, `Linear` or matmul (they are already fused in the crate), and any kernel beyond K1–K5.
  If K6 shows a remaining hot spot, write it down for a follow-up plan.

## 6. Traps, collected

| Trap | Symptom | Fix |
|---|---|---|
| Using `cross_entropy_with(.., ignore_index)` | 1 read per call, step time + ~1.4 ms × calls | host keep masks (T4) |
| Ignored ids left at -100 / 101 | garbage or out-of-bounds on the device | map to 0 with keep 0 (T4) |
| Trimming padding units | train/inference mismatch; worse accuracy | always pad to 20 (T3) |
| Query order mixed up (step-major vs unit-major) | top-1 near chance for steps 2–3, T7 < 60% | `q = u*K + j` everywhere |
| Forgetting to double `n_heads`/`n_groups` for bidirectional | init error ("requires even n_heads") or half capacity | T2 `BiBlock::new` |
| Reading losses every step | step time dominated by reads | `queue_train_step` + `read_losses` every 50 steps |
| Asking for bf16 | cannot be tested here; WGSL has no bf16 | only `f32` / `f16` are accepted (T6) |
| f16 without a loss scale | tiny gradients round to 0 in the f16 operand copy; training stalls or diverges from the f32 run | `loss_scale = 1024` with scaled `eps` and clip (§2.5) |
| f16 overflow (> 65504) | `inf` / `NaN` loss or grad norm | halve `loss_scale`; if activations overflow (loss is `inf` at scale 1), fall back to f32 and report |
| An adjoint written as a scatter / with atomics | nondeterministic gradients, contention on vulkan | every adjoint is a gather loop (K invariants 4) |
| Indexing with the `IGNORE` sentinel | out-of-bounds read, garbage on the device | test `id == IGNORE` before every indexed read (K invariant 6) |
| Two units on the same tile | a naive `d_t[upos] = g` overwrites instead of summing | K2 region 2 loops over all units (tested) |
| Toggling `set_fused_planner` inside a shared test binary | other tests flip modes mid-run | parity tests live in their own binary (`tests/planner_kernels.rs`) |
| exp overflow in the loss kernel | `inf` loss at large logits | max-shifted logsumexp; K3 test 3 |
| Building vulkan while cpu tests run | corrupted `target/`, random failures | one cargo job at a time |
| Committing the unrelated working-tree changes | noisy history | §0.3 |

## 7. Definition of done

- Rust `cpu` suite green, including `tests/planner.rs` (T1–T5 tests), `tests/planner_kernels.rs` (K1–K5 parity, both
  `cpu` and `vulkan`) and `tests/planner_footprint.rs` (pinned launch count, 0 reads).
- Python tests green (`test_planner.py`), `.pyi` updated, Rust/Python parity.
- The fused on-device path is the default; `examples/profile_planner.rs` shows ≤ 0.25 s/step and 0 reads/step on Vulkan,
  and `bench_planner` shows it faster than the composed path (or a reported profile explaining why not).
- A 14-epoch Vulkan run scored on dev16 and once on test40, with a report comparing accuracy and speed to the PyTorch
  Transformer reference.
