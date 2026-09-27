# Task-planner training step: EntityModel vs the PyTorch reference (`bench/torch_planner_step.py`)

Same data (Kaggriculture `runs/20260926_task_planner/data/train.npz`, first 5,120 turns), batch 128, 1 epoch = 40
steps, Radeon 860M (integrated, shared DDR), torch 2.13 + ROCm 7.1, mamba3_rl vulkan wheel from `tools/setup_mamba3.sh`.
Training steps only (no dev scoring). Raw traces: Kaggriculture `experiments/kobayashi/exp-planner053_dsm_task_planner/runs/20260927_dagger/bench_speed2.log`.

| date | trainer | commit / variant | s/step mean | s/step steady (21-40) | epoch s |
|---|---|---|---|---|---|
| 2026-09-27 | PyTorch ROCm (this script) | torch 2.13 | **0.568** | **0.543** | 22.7 |
| 2026-09-27 | EntityModel, `train_entity.py` | 91df9ea, defaults | 2.565 | 2.18 | 102.6 |
| 2026-09-27 | EntityModel | 094a936, defaults (K7 multi-AdamW, K8 fused split, K9 reduce, K11 seg cache) | 2.298 | 1.88 | 91.9 |
| 2026-09-27 | EntityModel | 094a936, `MAMBA3_ADAMW_MULTI=0 MAMBA3_FUSED_SPLIT=0 MAMBA3_REDUCE_SPLIT=0` | 2.690 | 2.37 | 107.6 |
| 2026-09-27 | EntityModel | 094a936, `--f16 1` | 2.431 | 2.08 | 97.2 |
| 2026-09-28 | PyTorch ROCm (this script) | torch 2.13, bf16 autocast (paired with the rows below) | **0.605-0.615** | **0.56-0.59** | 24.2-24.6 |
| 2026-09-28 | PyTorch ROCm | torch 2.13, autocast off (fp32, same precision as EntityModel) | 1.398 | 1.363 | 55.9 |
| 2026-09-28 | EntityModel | 467eb49 + fused scan / conv / split / matmul work (uncommitted), defaults, warm caches | 0.893-0.935 | ~0.89 | 35.7-37.4 |

Per-step losses are bit-identical between 91df9ea and 094a936 (same computation, faster kernels). K10 bf16 activation
storage is not reachable from the Python `EntityModel` yet (no dtype argument). Gate: the EntityModel is 3.3-4× the
PyTorch step; the planner is trained with PyTorch until this table says otherwise.

2026-09-28 rows: each pair measured back to back in one quiet window (other GPU jobs change these numbers by up to
2×; compare rows of the same date only). "Warm caches" means a second run of `train_entity.py`: the first run in a
fresh environment also compiles the kernels (now cached in the user cache directory) and tunes the matmuls (now
remembered in `~/.cache/mamba3`), which adds ~6 s to the first steps once. At equal precision the EntityModel step
is 1.5× faster than the PyTorch step (0.89 vs 1.36 s steady); PyTorch's 0.58 s is its bf16 autocast, which halves
the bytes every pass moves on this bandwidth-bound iGPU. Epoch-1 losses of the EntityModel are unchanged
(train_loss 5.4224 in both the 094a936 and the new runs, last-step loss 4.3873).
