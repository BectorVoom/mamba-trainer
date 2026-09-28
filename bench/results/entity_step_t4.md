# Entity training step on a Tesla T4 (Google Colab, via the `colab` CLI)

Kaggriculture spec (`bench/entity_planner_spec.json`), batch 128, random batch, forward + backward + AdamW
(weight decay 0.05, clip 1.0). Rust: `MAMBA3_ENTITY_SPEC=bench/entity_planner_spec.json MAMBA3_PROFILE_QUICK=1
MAMBA3_ENTITY_STEPS=10 ./target/release/examples/profile_entity_model` built with `--features cuda`; "drained min" is the
best of 10 single steps each followed by a device drain, "pipelined" is 10 queued steps with the launch tally off.
PyTorch: `bench/torch_entity_step.py`, the same architecture composed from ordinary torch ops (1.57 M parameters, the
Rust model has 1.58 M), torch 2.11+cu128, best of 10 steps.

| date | side | variant | ms/step |
|---|---|---|---|
| 2026-09-28 | PyTorch | eager fp32 | 345 |
| 2026-09-28 | PyTorch | eager fp16 autocast | 331 |
| 2026-09-28 | PyTorch | `torch.compile` fp32 (530 s compile) | 217 |
| 2026-09-28 | PyTorch | `torch.compile` fp16 autocast (420 s compile) | 146 |
| 2026-09-28 | EntityModel | `3a7bed6`, f32, drained min / median | 145-149 / 154-156 |
| 2026-09-28 | EntityModel | + grad-norm partials 32 elements per unit (`sum_squares_groups`), drained min | 126-132 |
| 2026-09-28 | EntityModel | + `broadcast_join_backward` in two parallel launches, f32 | 134 (drained min), 140 pipelined |
| 2026-09-28 | EntityModel | + warp-tiled `matmul_cmma_kernel`, `MAMBA3_MATMUL_PRECISION=f16` (tensor cores, 3-6 TFLOP/s) | 127-128 (drained min), 133-172 pipelined |

Per-kernel GPU time of the `3a7bed6` step (`ncu --metrics gpu__time_duration.sum`, one step, shares; ncu locks the
base clock so absolute times are ~1.5x real): scan backward (chunked) 21%, f32 block-tiled matmuls 34% (~3.5-4 TFLOP/s
on the large shapes), scan forward 6%, `sum_dim` 5% (246 launches), grad-norm partials 3%, swiglu / split / cat /
reverse_bands / rms_norm / conv / rotate 1-3% each. The matrix-core kernel (`matmul_cmma_kernel`) measured 7 GFLOP/s
before its rewrite and never won a tuner slot; bf16 fragments do not compile on sm_75.

The rewritten matrix-core kernel wins the tuner on most f16 shapes (`MAMBA3_TUNE_LOG=1`: `Cmma(64, 64, ..)` at
2.5-6 TFLOP/s against 3-5 for the f32 register kernels) and passes the seven `mixed_precision` GPU tests plus
`MAMBA3_TUNE_CHECK=1` on the T4, but it is still a single-stage kernel (two barriers per 32-deep k step, scalar
16-bit shared stores), a fraction of what the T4's tensor cores do; double buffering and wider per-plane tiles are the
next step. bf16 fragments do not compile on sm_75 (`nvcc: incomplete type nvcuda::wmma::fragment<..., __nv_bfloat16>`),
so on CUDA the bf16 fragment is now behind `MAMBA3_CMMA_BF16=1` and `MatmulPrecision::Bf16` falls back to the register
kernels instead of aborting.

The T4 step is not 3-5x the PyTorch step. Against eager fp32 it is 2.6x (345 / 134); against `torch.compile` fp16,
which is PyTorch's honest best here, it is 1.1x. The remaining GPU time is the chunked scan backward (one cube per
(batch, head), 38 KB of shared memory, so one cube per T4 SM), the matmuls, and ~1,000 bandwidth passes; the pipelined
step being no faster than the drained step says host submission of ~1,100-1,250 launches is as long as the GPU work.

Boost-clock kernel shares (`ncu --clock-control none --metrics gpu__time_duration.sum`, one step, after the changes):

| kernel | f32 step | f16 step |
|---|---|---|
| matmuls (register kernels / `matmul_cmma_kernel`) | 36% | 23% (+ 6% operand casts, 200 launches) |
| `ssd_scan_backward_chunked_kernel` | 20% | 21% |
| `ssd_scan_kernel` | 6% | 6% |
| `sum_dim` (broadcast-gradient reductions, ~240 launches) | 5% | 6% |
| swiglu, split / cat / reverse bands, rms_norm backward, conv weight grad | 2-3% each | 2-3% each |

Next levers, in order: cache the f16 copies of the weights instead of casting both operands on every call (the 6%
eats most of the matrix-core gain); a scan backward that fits more than one cube per SM; double-buffered staging and
wider per-plane tiles in `matmul_cmma_kernel`; folding the `sum_dim` adjoint reductions into their producers.

`tests/entity_footprint.rs`'s two launch pins are CPU-backend numbers and fail on CUDA both before and after these
changes (904 against 1847 at `3a7bed6`, 906 against 1849 now); every other GPU suite run here passes on the T4
(`mixed_precision` 7/7, `matmul_paths` 2/2, `adamw_multi` 3/3, `entity_model_kernels` 11/11, `ssd_scan` 7/7).

CPU suite (`cargo test --release --features cpu --no-fail-fast`, Mac mini M1) after the changes: every suite passes
except `train_ema_parity::rust_python_parity`, which fails identically at `3a7bed6` without these changes (its
recorded fingerprint is one ulp off in the first values). `sum_dim_single_pass_pins` moved from `tests/tensor.rs` to
its own binary, `tests/reduce_launch_pins.rs`: it read the process-wide launch counter while the other tensor tests
were launching kernels, and failed in the full run (4 launches read for 1) though it passed alone.

## Scan backward: vectorised shared loads (2026-09-28, uncommitted)

`ssd_scan_backward_chunked_kernel` now reads shared memory four floats per instruction (`Vector<f32, 4>` rows padded by one
vector), gives each unit wider output tiles, and runs the dC and dB products on separate halves of the cube at once: about
300 shared-load instructions per unit per chunk instead of ~980, for the same multiply-adds (the Nsight profile of the old
kernel on the T4: load/store pipe 86.5% busy, FMA pipe 19%, 57% of stalls on the memory-instruction queue). Shared memory at
head_dim 64 / d_state 32: 38.6 KB for chunk 16, 28.3 KB for chunk 8. Correctness: `tests/ssd_scan.rs` 7/7 and
`tests/ssd_scan_chunk8.rs` 4/4 on cpu and on wgpu (Metal, M1).

On the M1 (wgpu, 32 KB of shared memory per workgroup) the chunked backward does not fit at chunk 16, so the per-step
backward runs, as before; forcing chunk 8 makes the chunked kernel fit but it is 2.4x slower there (113 vs 47 ms per call at
b=128, t=100, h=8; 88 vs 38 ms at t=160, h=4), so the default is right for that GPU. The rewrite matters where the chunked
kernel runs (the T4: 2-3x faster than the per-step backward before this change). T4 A/B against the previous kernel: pending
(Colab had no T4 capacity at 20:20).
