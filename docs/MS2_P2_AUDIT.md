# MS2 P2.1: audit of the reusable APIs

Status: read from the sources on 2026-10-03; file and line references are to the working tree of that day.
Nothing here is measured. The allocation and read behaviour below is what the code does, not a timing.

Used by: [MS2_V0_ARCHITECTURE.md](MS2_V0_ARCHITECTURE.md). Plan item: P2.1 of
[MS2_SUBSTRUCTURE_TASKS.md](MS2_SUBSTRUCTURE_TASKS.md).

## Reuse table

| Need | API | Allocation behaviour | Device reads | Notes for MS2 |
|---|---|---|---|---|
| Residual SSM block, sequence form | `Mamba3Block::apply` (`src/models/mamba3.rs:885`) | Functional: every operation returns a new tensor | None | Pre-norm, residual. One fused input projection of width `SsmConfig::in_proj_width()` |
| Residual SSM block, one step | `Mamba3Block::step`, `empty_cache` (`mamba3.rs:922`, `947`) | Functional: returns a fresh `MixerCache` each step, so two cache banks are live across a step | None | Takes `[B, 1, d]`. Carries are `h`, `last_u` (same shape, `[B, heads, head_dim, d_state]`), `angle` (`[B, heads, d_state / 2]`) and a convolution history when `conv_kernel` is set (`src/ssm/scan.rs:363`) |
| Sequence scan | `ssd_chunked` (`src/ssm/scan.rs`) | Chunked form builds a `[.., chunk, chunk]` band per chunk; fused form (`Var::ssd_scan`) keeps only the output | None | Default: fused on GPU-like devices, chunked on the CPU runtime (`scan.rs:118-128`). The chunked form multiplies by a zero band (`scan.rs:260-267`), so a NaN in a later position reaches earlier outputs: inputs must be finite everywhere |
| MIMO | `SsmMode::Mimo` | — | — | Reference path only: `R²` SISO scans sharing the decay (`scan.rs:664-707`). No rank-aware kernel exists. V0 is SISO |
| Bidirectional mixer | `Mamba3MixerConfig::with_bidirectional` | — | — | Reverses the whole padded sequence, not the valid prefix. Not used; MS2 reverses by valid length with `gather_tokens` |
| Per-batch gather | `Var::gather_tokens` (`src/autograd/ops.rs:1855`) | One output | None, forward or backward | `u32::MAX` id gives a zero row. Adjoint is a gather loop, no atomics |
| Embedding lookup | `autograd::embedding` (`ops.rs:2644`) | One output | **Backward reads the ids to the host** (`src/tensor/ops/index.rs:513`) | No sentinel handling. Not used; MS2 adds `ms2_lookup` |
| Logit masking | `Var::mask_logits` (`ops.rs:799`) | One output | None | Wants a float 0/1 mask; illegal logits become the finite minimum, never `-inf` |
| Softmax | `Var::softmax`, `log_softmax` (`ops.rs:2599`, `2609`) | Composed: max, exp, sum, divide | None | No empty-support handling: an all-masked row is uniform |
| Row select | `Var::take_along_last` (`ops.rs:1203`) | One output; the adjoint builds a one-hot | None | No bounds or sentinel check |
| Attention | `MultiHeadAttention` (`src/nn/attention.rs`) | Composed matmuls | None | Self-attention only. MS2 writes its cross-attention from the same operations |
| Normalisation | `RmsNorm`, `LayerNorm` (`src/nn/norm.rs`) | One output | None | — |
| Linear, MLP | `Linear`, `Mlp` (`src/nn/linear.rs`, `mlp.rs`) | One output per layer | None | Matmul shapes are tuned on first use: see below |
| Matmul | `Var::matmul`, `matmul_nt` (`src/tensor/ops/matmul.rs`) | One output | **One-time tuning read per new shape**, hidden from `read_count` by `uncounted_reads` (`matmul.rs:2894`, `src/backend.rs:741`) | A cold call reads; a warmed shape does not. Counter tests must warm first and use `runtime_read_count` |
| Integer buffers | `IdTensor` (`src/tensor/ops/index.rs:16`) | `empty`, `from_slice` | `to_vec` reads | `u32` only |
| Batched readout | `read_all`, `read_all_mixed` (`index.rs:167`, `245`) | — | One read for handles of one stream, one per handle otherwise | The single readout of a generation call |
| Device RNG | `hash_u32`, `hash_unit_f32` (`src/tensor/ops/random.rs:135`, `155`) | None | None | Stateless hash of `(index, seed)`: 24-bit draws |
| Host RNG | `Rng` (`random.rs:19`) | — | — | Weight initialisation only |
| Optimizer | `AdamW` (`src/train/optim.rs:273`), `clip_grad_norm`, `grad_scale` | Moments allocated once per parameter | `grad_norm` reads one scalar (`optim.rs:995`); `Trainer::queue_step` + `read_steps` batch the loss and norm scalars into one read (`src/train/trainer.rs:327-411`) | Use the queued path so a training step has no read of its own |
| Counters | `launch_count`, `read_count`, `reserved_bytes` (`src/backend.rs`) | — | — | Process-global: a test that pins them lives alone in its test binary. P2.7 adds `runtime_read_count`, transfer bytes and allocation calls |
| Element types | `supports_dtype`, `ensure_dtype` (`backend.rs:215`, `237`) | — | — | WGSL has no `bf16`; `f16` needs the device feature |
| Launch geometry | `launch_1d`, `launch_1d_spans` (`backend.rs:917`, `969`) | — | — | Both count the launch. On the CPU runtime a kernel walks a span of lanes per unit |

## Consequences already applied to the architecture

- Padding is selected to exact zeros before and after every block (chunked-scan finding).
- Embeddings use `ms2_lookup`, whose adjoint stays on the device.
- All-masked rows get an explicit single-index mask.
- The zero-read claim is tested on warmed shapes with a counter that the tuner cannot hide from.
- Two decoder cache banks are budgeted while the step is functional.

## Not audited

Performance of any of these at MS2 shapes, and the CUDA and HIP backends. P3.6 and V0.7 measure the first on CPU
and wgpu/Metal.
