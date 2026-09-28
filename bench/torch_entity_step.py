"""The entity model (`src/models/entity`) in PyTorch, as the reference its CubeCL training step is timed against.

`bench/torch_planner_step.py` times Kaggriculture's Transformer planner, which needs that checkout and its data. This
script needs only torch: it is the *same architecture* as `EntityModel` on the Kaggriculture spec
(`examples/profile_entity_model.rs::kaggriculture_spec`, `bench/entity_planner_spec.json`) — the tile / unit / global
MLPs, three bidirectional Mamba-3 context blocks with the grid transposes between them, the teacher-forced query
tokens (anchor tile, step embedding, two lag projections of the previous choices), three step-causal decoder layers
each with the crew-symmetric reversed second scan, the pointer head over `[tiles ; extra ; none]`, the conditioned and
unconditioned shared Linears, and the same masked losses — composed from ordinary tensor ops exactly as
`bench/torch_mamba3.py` composes the mixer, so the comparison is between two implementations of one algorithm.

Random data with `profile_entity_model`'s label rates; forward + backward + AdamW (weight decay 0.05, clip 1.0, the
Python binding's trainer settings) per step. Pair it with

    MAMBA3_ENTITY_SPEC=bench/entity_planner_spec.json MAMBA3_PROFILE_QUICK=1 \\
        ./target/release/examples/profile_entity_model

Run:
    python bench/torch_entity_step.py                      # eager fp32
    python bench/torch_entity_step.py --dtype fp16         # autocast fp16 + GradScaler (a T4 has no bf16 units)
    python bench/torch_entity_step.py --compile            # torch.compile
Knobs: BATCH (128), ITERS (10).
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_mamba3 import RmsNorm, rotate_halves, ssd_chunked  # noqa: E402

# Kaggriculture spec.
N_TILES, TILE_F, GLOB_F, N_UNITS, UNIT_F, K_STEPS = 100, 48, 114, 20, 36, 3
GRID = 10
D, H, P, N_STATE, CONV = 128, 4, 64, 32, 4
CTX_LAYERS, DEC_LAYERS = 3, 3
N_OP, N_OPSET, N_CROP = 13, 13, 5
EXTRA = 1  # pointer head's learned extra action ("none" target)


def chunk_for(t):
    for c in (64, 50, 48, 40, 32, 25, 20, 16):
        if t % c == 0:
            return c
    return 32


class Mamba3Mixer(nn.Module):
    """`src/models/mamba3.rs` with `n_groups` (B/C shared by `heads // groups` heads) and the fused bidirectional
    form: the second half of the heads (and groups) scans the sequence right to left."""

    def __init__(self, d_model, n_heads, head_dim, d_state, n_groups, bidirectional):
        super().__init__()
        if bidirectional:
            n_heads, n_groups = 2 * n_heads, 2 * n_groups
        self.h, self.p, self.n, self.g = n_heads, head_dim, d_state, n_groups
        self.bidirectional = bidirectional
        self.d_inner = n_heads * head_dim
        self.bc = n_groups * d_state
        width = 2 * self.d_inner + 2 * self.bc + n_heads + n_heads + n_heads * d_state // 2
        self.in_proj = nn.Linear(d_model, width, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        conv_ch = self.d_inner + 2 * self.bc
        self.conv = nn.Conv1d(conv_ch, conv_ch, CONV, groups=conv_ch, padding=CONV - 1)
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.a_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, n_heads)))
        self.d_skip = nn.Parameter(torch.ones(n_heads))
        self.b_bias = nn.Parameter(torch.zeros(n_heads, d_state))
        self.c_bias = nn.Parameter(torch.zeros(n_heads, d_state))
        self.bc_norm = RmsNorm(d_state)

    def _scan(self, x, b, c, dt, lam, theta, a_log, chunk):
        """One direction: x [B,T,H,P], b/c [B,T,H,N], dt/lam [B,T,H], theta [B,T,H,N/2], a_log [H] -> y [B,T,H,P]."""
        a = dt * (-torch.exp(a_log))
        g = lam * dt
        nxt = (1.0 - lam) * dt
        f = torch.cat([nxt[:, 1:], torch.zeros_like(nxt[:, :1])], dim=1)
        w = g + f
        phi = (dt.unsqueeze(3) * theta).cumsum(1)
        two_pi = 2.0 * math.pi
        phi = phi - (torch.round(phi / two_pi) * two_pi).detach()
        b = rotate_halves(b, phi)
        c = rotate_halves(c, phi)
        return ssd_chunked(x, b, c, a, g, w, chunk)

    def forward(self, u):
        B, T, _ = u.shape
        H, P, N, G = self.h, self.p, self.n, self.g
        proj = self.in_proj(u)
        z, xbc, dt_raw, lam_raw, theta_raw = torch.split(
            proj, [self.d_inner, self.d_inner + 2 * self.bc, H, H, H * N // 2], dim=-1
        )
        if self.bidirectional:
            # Backward heads see reversed time through the convolution and the scan.
            half = lambda t, w: torch.cat([t[..., : w // 2], t[..., w // 2 :].flip(1)], dim=-1)  # noqa: E731
            xbc = torch.cat(
                [
                    half(xbc[..., : self.d_inner], self.d_inner),
                    half(xbc[..., self.d_inner : self.d_inner + self.bc], self.bc),
                    half(xbc[..., self.d_inner + self.bc :], self.bc),
                ],
                dim=-1,
            )
            dt_raw, lam_raw = half(dt_raw, H), half(lam_raw, H)
            theta_raw = half(theta_raw, H * N // 2)
        xbc = self.conv(xbc.transpose(1, 2))[..., :T].transpose(1, 2)
        xbc = F.silu(xbc)
        x, b_flat, c_flat = torch.split(xbc, [self.d_inner, self.bc, self.bc], dim=-1)
        per_group = H // G
        b = b_flat.reshape(B, T, G, 1, N).expand(B, T, G, per_group, N).reshape(B, T, H, N) + self.b_bias
        c = c_flat.reshape(B, T, G, 1, N).expand(B, T, G, per_group, N).reshape(B, T, H, N) + self.c_bias
        b, c = self.bc_norm(b), self.bc_norm(c)
        x = x.reshape(B, T, H, P)
        dt = F.softplus(dt_raw + self.dt_bias)
        lam = torch.sigmoid(lam_raw)
        theta = theta_raw.reshape(B, T, H, N // 2)
        chunk = chunk_for(T)
        if self.bidirectional:
            hh = H // 2
            y_f = self._scan(x[:, :, :hh], b[:, :, :hh], c[:, :, :hh], dt[..., :hh], lam[..., :hh], theta[:, :, :hh],
                             self.a_log[:hh], chunk)
            y_b = self._scan(x[:, :, hh:], b[:, :, hh:], c[:, :, hh:], dt[..., hh:], lam[..., hh:], theta[:, :, hh:],
                             self.a_log[hh:], chunk)
            y = torch.cat([y_f, y_b.flip(1)], dim=2)
            x = torch.cat([x[:, :, :hh], x[:, :, hh:].flip(1)], dim=2)
        else:
            y = self._scan(x, b, c, dt, lam, theta, self.a_log, chunk)
        y = y + x * self.d_skip.view(1, 1, H, 1)
        y = y.reshape(B, T, self.d_inner) * F.silu(z)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.norm = RmsNorm(D)
        self.mixer = Mamba3Mixer(D, H, P, N_STATE, 1, bidirectional)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


def mlp(f_in):
    return nn.Sequential(nn.Linear(f_in, D), nn.GELU(), nn.Linear(D, D))


def grid_transpose_index(n_ctx, offset=0, h=GRID, w=GRID):
    fwd = torch.arange(n_ctx)
    for r in range(h):
        for c in range(w):
            fwd[offset + c * h + r] = offset + r * w + c
    return fwd


def reverse_blocks_index(n, offset, block, blocks):
    fwd = torch.arange(n)
    for k in range(blocks):
        for i in range(block):
            fwd[offset + k * block + i] = offset + k * block + (block - 1 - i)
    return fwd


class EntityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile_mlp, self.glob_mlp, self.unit_mlp = mlp(TILE_F), mlp(GLOB_F), mlp(UNIT_F)
        self.pos = nn.Parameter(torch.randn(N_TILES, D) * 0.02)
        self.typ = nn.Parameter(torch.randn(1, D) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(K_STEPS, D) * 0.02)
        self.none_prev = nn.Parameter(torch.randn(1, D) * 0.02)
        self.extra_emb = nn.Parameter(torch.randn(EXTRA, D) * 0.02)
        self.ctx_blocks = nn.ModuleList([Block(True) for _ in range(CTX_LAYERS)])
        self.lags = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.dec_main = nn.ModuleList([Block(False) for _ in range(DEC_LAYERS)])
        self.dec_rev = nn.ModuleList([Block(False) for _ in range(DEC_LAYERS)])
        self.norm = RmsNorm(D)
        self.q_proj, self.k_proj = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.cond = nn.Linear(2 * D, N_OP + N_OPSET + N_CROP)
        self.uncond = nn.Linear(D, 1)
        n_dec = N_TILES + N_UNITS * K_STEPS
        self.register_buffer("grid_perm", grid_transpose_index(N_TILES))
        self.register_buffer("rev_perm", reverse_blocks_index(n_dec, N_TILES, N_UNITS, K_STEPS))

    def forward(self, batch):
        tiles, glob, units = batch["tiles"], batch["globals"], batch["units"]
        anchor, tgt = batch["units.anchor"], batch["label.target"]
        B = tiles.shape[0]
        # --- encoder ---
        present = (tiles.abs().sum(-1, keepdim=True) > 0).to(tiles.dtype)
        c = self.tile_mlp(tiles) * present + self.pos + self.typ
        g = self.glob_mlp(glob)
        c = c + g.unsqueeze(1)
        for l, blk in enumerate(self.ctx_blocks):
            if l % 2 == 1:  # alternate_axes: odd layers scan the grid column-major
                c = blk(c[:, self.grid_perm])[:, self.grid_perm]
            else:
                c = blk(c)
        # --- queries (teacher forced) ---
        upres = (anchor >= 0).to(tiles.dtype).unsqueeze(-1)
        u = self.unit_mlp(units) * upres
        table = torch.cat([c, self.extra_emb.expand(B, EXTRA, D), self.none_prev.expand(B, 1, D)], dim=1)
        none_id = table.shape[1] - 1
        anchor_tok = table[torch.arange(B, device=c.device).unsqueeze(1), anchor.clamp(min=0)] * upres
        base = u + anchor_tok + g.unsqueeze(1)                                      # [B, M, d]
        choice = torch.where(tgt >= 0, tgt, torch.full_like(tgt, none_id))         # [B, M, K]
        extra = 0.0
        bidx = torch.arange(B, device=c.device).view(B, 1, 1)
        for l, lag in enumerate(self.lags):
            shifted = torch.full_like(choice, none_id)
            shifted[:, :, l + 1 :] = choice[:, :, : K_STEPS - l - 1]
            extra = extra + lag(table[bidx, shifted])                             # [B, M, K, d]
        q = base.unsqueeze(2) + self.step_emb.view(1, 1, K_STEPS, D) + extra       # [B, M, K, d]
        q = q.permute(0, 2, 1, 3).reshape(B, K_STEPS * N_UNITS, D)                 # step-major (StepCausal)
        # --- decoder ---
        s = torch.cat([c, q], dim=1)
        for main, rev in zip(self.dec_main, self.dec_rev):
            y = main(s)
            r = rev(s[:, self.rev_perm])[:, self.rev_perm]
            s = y + (r - s)
        s = self.norm(s)
        ctx_out, h = s[:, :N_TILES], s[:, N_TILES:]
        h = h.reshape(B, K_STEPS, N_UNITS, D).permute(0, 2, 1, 3)                  # [B, M, K, d]
        # --- heads ---
        keys = torch.cat([ctx_out, self.extra_emb.expand(B, EXTRA, D)], dim=1)     # [B, 101, d]
        logit = self.q_proj(h.reshape(B, -1, D)) @ self.k_proj(keys).transpose(1, 2) / math.sqrt(D)
        logit = logit.reshape(B, N_UNITS, K_STEPS, N_TILES + EXTRA)
        mask = torch.cat([(1.0 - present.squeeze(-1)) * -1e4, torch.zeros(B, EXTRA, device=c.device, dtype=logit.dtype)], 1)
        logit = logit + mask.view(B, 1, 1, -1)
        tok = torch.cat([ctx_out, self.extra_emb.expand(B, EXTRA, D), self.none_prev.expand(B, 1, D)], 1)[bidx, choice]
        cond = self.cond(torch.cat([h, tok], -1))
        op_l, opset_l, crop_l = torch.split(cond, [N_OP, N_OPSET, N_CROP], dim=-1)
        eta_l = self.uncond(h[:, :, 0])
        return logit, op_l, opset_l, crop_l, eta_l

    def loss(self, batch):
        logit, op_l, opset_l, crop_l, eta_l = self(batch)
        tgt, op, crop, opset, eta = (batch[k] for k in ("label.target", "label.op", "label.crop", "label.opset", "label.eta"))
        w = torch.tensor([1.0, 0.5, 0.5], device=tgt.device)
        ce = F.cross_entropy(logit.reshape(-1, logit.shape[-1]).float(), tgt.reshape(-1), ignore_index=-1, reduction="none")
        kept = (tgt.reshape(-1) >= 0).float()
        l_tgt = (ce * w.repeat(tgt.shape[0] * N_UNITS) * kept).sum() / kept.sum().clamp(min=1)
        l_op = F.cross_entropy(op_l.reshape(-1, N_OP).float(), op.reshape(-1), ignore_index=-1)
        l_crop = F.cross_entropy(crop_l.reshape(-1, N_CROP).float(), crop.reshape(-1), ignore_index=-1)
        m = (op >= 0).float().unsqueeze(-1)
        l_opset = (F.binary_cross_entropy_with_logits(opset_l.float(), opset, reduction="none") * m).sum() / (m.sum() * N_OPSET).clamp(min=1)
        em = (~torch.isnan(eta)).float()
        l_eta = (((eta_l.float() - torch.nan_to_num(eta)) ** 2) * em).sum() / em.sum().clamp(min=1)
        return l_tgt + l_op + 0.3 * l_opset + 0.3 * l_crop + 0.1 * l_eta


def random_batch(b, dev, seed=99):
    g = torch.Generator().manual_seed(seed)
    anchor = torch.where(torch.rand(b, N_UNITS, generator=g) < 0.8, torch.randint(0, N_TILES, (b, N_UNITS), generator=g), -1)
    r = torch.rand(b, N_UNITS, K_STEPS, generator=g)
    tgt = torch.where(r < 0.7, torch.randint(0, N_TILES, (b, N_UNITS, K_STEPS), generator=g), torch.where(r < 0.8, N_TILES, -1))
    lab = r < 0.7
    op = torch.where(lab, torch.randint(0, N_OP, tgt.shape, generator=g), -1)
    crop = torch.where(lab, torch.randint(0, N_CROP, tgt.shape, generator=g), -1)
    opset = F.one_hot(torch.randint(0, N_OPSET, tgt.shape, generator=g), N_OPSET).float() * lab.unsqueeze(-1)
    eta = torch.where(anchor >= 0, torch.randint(0, 20, anchor.shape, generator=g).float(), torch.nan).unsqueeze(-1)
    batch = {
        "tiles": torch.rand(b, N_TILES, TILE_F, generator=g) * 2 - 1,
        "globals": torch.rand(b, GLOB_F, generator=g) * 2 - 1,
        "units": torch.rand(b, N_UNITS, UNIT_F, generator=g) * 2 - 1,
        "units.anchor": anchor, "label.target": tgt, "label.op": op, "label.crop": crop, "label.opset": opset, "label.eta": eta,
    }
    return {k: v.to(dev) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()
    batch_size, iters = int(os.environ.get("BATCH", 128)), int(os.environ.get("ITERS", 10))
    dev = "cuda"
    torch.manual_seed(0)
    model = EntityModel().to(dev)
    print(f"torch   {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(f"params  {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M   batch {batch_size}   dtype {args.dtype}"
          f"{'  compile' if args.compile else ''}")
    loss_fn = torch.compile(model.loss) if args.compile else model.loss
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    batch = random_batch(batch_size, dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            loss = loss_fn(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return loss

    t = time.perf_counter()
    for _ in range(3):
        loss = step()
    torch.cuda.synchronize()
    print(f"warmup  {time.perf_counter() - t:9.2f}s  loss {loss.item():.4f}")
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    print(f"step    best {times[0] * 1e3:9.2f} ms  median {times[len(times) // 2] * 1e3:9.2f} ms"
          f"   peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
