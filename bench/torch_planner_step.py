"""PyTorch reference step for the Kaggriculture task planner: the number `train_entity.py` (EntityModel) is measured
against.

This times only the training step of `exp-planner053_dsm_task_planner/src/train.py`'s model (TaskPlanner: tile
Transformer encoder + 3-step autoregressive plan decoder, 4.1 M parameters) on the planner's npz data — the same
data, batch size, turn limit, bf16 autocast, AdamW + OneCycle + grad clipping and per-step host sync as train.py's
loop, without the dev scoring that train.py's per-epoch `secs` includes. It imports train.py / model.py / features.py
from the Kaggriculture checkout and modifies nothing there.

Pairing (both sides read the same file and the same first LIMIT turns):
    # PyTorch ROCm (this script), from the Kaggriculture torch venv
    KAGG_EXP=~/Documents/workspace/Kaggriculture/experiments/kobayashi/exp-planner053_dsm_task_planner \\
      ~/Documents/workspace/torch-rocm-venv/bin/python bench/torch_planner_step.py
    # mamba-trainer EntityModel (vulkan wheel from tools/setup_mamba3.sh), from the Kaggriculture rust venv
    cd $KAGG_EXP && ~/.venvs/kaggriculture-rust/bin/python src/train_entity.py --data runs/20260926_task_planner/data \\
      --out /tmp/bench_entity --max-train 5120 --epochs 1 --log-every 5
Knobs: KAGG_EXP (experiment dir), KAGG_DATA (train.npz; default <KAGG_EXP>/runs/20260926_task_planner/data/train.npz),
LIMIT (5120), BS (128), EPOCHS (1). Results: bench/results/planner_step.md.

Measured 2026-09-27 (Radeon 860M, ROCm 7.1, torch 2.13; small data: 5,120 turns = 40 steps at batch 128, 1 epoch):
    PyTorch      0.568 s/step mean, 0.543 steady (steps 21-40)
    EntityModel  094a936 defaults 2.298 mean / 1.88 steady; 91df9ea 2.565 / 2.18; K7/K8/K9 off 2.690 / 2.37; --f16 1 2.431 / 2.08
Measured 2026-09-28, paired runs in one window (see bench/results/planner_step.md):
    PyTorch      0.605-0.615 mean / 0.56-0.59 steady (bf16 autocast); 1.398 / 1.363 with autocast off (fp32)
    EntityModel  fused scan + kernel work, warm caches: 0.89-0.93 mean / ~0.89 steady (fp32)
"""
import math, os, sys, time
from pathlib import Path

import numpy as np
import torch

EXP = Path(os.environ.get("KAGG_EXP", Path.home() / "Documents/workspace/Kaggriculture/experiments/kobayashi/exp-planner053_dsm_task_planner")).expanduser()
DATA = Path(os.environ.get("KAGG_DATA", EXP / "runs/20260926_task_planner/data/train.npz")).expanduser()
LIMIT, BS, EPOCHS = int(os.environ.get("LIMIT", 5120)), int(os.environ.get("BS", 128)), int(os.environ.get("EPOCHS", 1))
sys.path.insert(0, str(EXP / "src"))
import train as T  # noqa: E402
import features as F  # noqa: E402
from model import TaskPlanner, losses  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda")
print("device", torch.cuda.get_device_name(0), torch.__version__, "data", DATA)
tr = T.load(DATA, dev)
tr = {k: v[:LIMIT] for k, v in tr.items()}
cfg = dict(c_tile=tr["tiles"].shape[-1], c_glob=F.C_GLOB, c_unit=tr["units"].shape[-1], u_max=F.U_MAX, k=3, d=192, heads=6,
           enc_layers=4, dec_layers=3, autoregressive=True, pairwise=False)
model = TaskPlanner(**cfg).to(dev)
print("params", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
n = tr["tgt"].shape[0]; steps = EPOCHS * math.ceil(n / BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
dt = T.amp_dtype(dev); scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)
print("autocast", dt, "turns", n, "steps", steps, "batch", BS)
model.train(); times = []
torch.cuda.synchronize(); t_all = time.time()
i = 0
for ep in range(EPOCHS):
    perm = torch.randperm(n, device=dev)
    for s in range(0, n, BS):
        t0 = time.time()
        b = T.batch_of(tr, perm[s:s + BS])
        with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
            L, _, _ = losses(model, b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(L["total"]).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        loss = L["total"].item()          # the same host sync per step as train.py's loss aggregation
        torch.cuda.synchronize(); times.append(time.time() - t0); i += 1
        if i % 5 == 0:
            print(f"  step {i}/{steps} loss {loss:.4f} ({np.mean(times):.3f} s/step cum, {np.mean(times[-5:]):.3f} s/step last 5)", flush=True)
torch.cuda.synchronize(); total = time.time() - t_all
steady = np.mean(times[20:]) if len(times) > 20 else float("nan")
print(f"train-only: {steps} steps, {total:.1f} s, {total / steps:.3f} s/step mean; steady (steps 21+) {steady:.3f} s/step; "
      f"first 5 {np.mean(times[:5]):.3f} s/step")
