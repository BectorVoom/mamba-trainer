"""Mamba-3 in PyTorch, as the reference the Rust/CubeCL implementation is measured against.

This is a direct port of `src/models/mamba3.rs` and `src/ssm/scan.rs`: the same fused
input projection, the same depthwise causal convolution, the same B/C bias + RMS norm,
the same learned-trapezoidal weights, the same rotating frame, and the same chunked SSD
scan written out of the same primitives. Nothing here is fused by hand, which is exactly
the point — both sides compose the scan from ordinary tensor ops, so the comparison is
between two implementations of one algorithm rather than between two algorithms.

Run:
    python bench/torch_mamba3.py                       # eager
    python bench/torch_mamba3.py --compile             # torch.compile
Shape knobs mirror examples/bench_train.rs and come from the same environment variables.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_DECAY_FLOOR = -60.0


def env(name, default):
    return int(os.environ.get(name, default))


class RmsNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def cumsum_exclusive(x, dim):
    return x.cumsum(dim) - x


def ssd_chunked(x, b, c, a, g, w, chunk):
    """The scan of `src/ssm/scan.rs::ssd_chunked`.

    x [B,T,H,P]; b, c [B,T,H,N]; a, g, w [B,T,H]. Returns y [B,T,H,P].
    """
    B, T, H, P = x.shape
    N = b.shape[-1]
    pad = (chunk - T % chunk) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, 0, 0, pad))
        c = F.pad(c, (0, 0, 0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        w = F.pad(w, (0, 0, 0, pad))
    L = T + pad
    C = L // chunk

    x = x.reshape(B, C, chunk, H, P)
    b = b.reshape(B, C, chunk, H, N)
    c = c.reshape(B, C, chunk, H, N)
    a = a.reshape(B, C, chunk, H)
    g = g.reshape(B, C, chunk, H)
    w = w.reshape(B, C, chunk, H)

    acum = a.cumsum(2)

    # ---- 1. intra-chunk ----
    diff = acum.unsqueeze(3) - acum.unsqueeze(2)              # [B,C,t,s,H]
    causal = torch.tril(torch.ones(chunk, chunk, device=x.device, dtype=x.dtype))
    decay = diff.clamp(LOG_DECAY_FLOOR, 0.0).exp() * causal.view(1, 1, chunk, chunk, 1)
    strict = torch.tril(
        torch.ones(chunk, chunk, device=x.device, dtype=x.dtype), diagonal=-1
    ).view(1, 1, chunk, chunk, 1)
    eye = torch.eye(chunk, device=x.device, dtype=x.dtype).view(1, 1, chunk, chunk, 1)
    weight = w.unsqueeze(2) * strict + g.unsqueeze(2) * eye

    def heads_first(v, last):
        return v.permute(0, 1, 3, 2, 4).reshape(B * C * H, chunk, last)

    c_flat = heads_first(c, N)
    b_flat = heads_first(b, N)
    x_flat = heads_first(x, P)

    cb = (c_flat @ b_flat.transpose(-1, -2)).reshape(B, C, H, chunk, chunk).permute(
        0, 1, 3, 4, 2
    )
    mixing = cb * decay * weight
    mixing_flat = mixing.permute(0, 1, 4, 2, 3).reshape(B * C * H, chunk, chunk)
    y_diag = (mixing_flat @ x_flat).reshape(B, C, H, chunk, P).permute(0, 1, 3, 2, 4)

    # ---- 2. chunk summaries ----
    last_acum = acum[:, :, chunk - 1 : chunk]
    decay_to_end = (last_acum - acum).clamp(LOG_DECAY_FLOOR, 0.0).exp()
    weighted_x = x * (decay_to_end * w).unsqueeze(4)
    chunk_state = (
        weighted_x.permute(0, 1, 3, 4, 2).reshape(B * C * H, P, chunk) @ b_flat
    ).reshape(B, C, H, P, N)

    # ---- 3. inter-chunk recurrence ----
    chunk_decay = last_acum.reshape(B, C, H)
    before = cumsum_exclusive(chunk_decay, 1)
    through = chunk_decay.cumsum(1)
    strict_chunks = torch.tril(
        torch.ones(C, C, device=x.device, dtype=x.dtype), diagonal=-1
    ).view(1, C, C, 1)
    transfer = (before.unsqueeze(2) - through.unsqueeze(1)).clamp(
        LOG_DECAY_FLOOR, 0.0
    ).exp() * strict_chunks
    carry_in = (
        transfer.permute(0, 3, 1, 2).reshape(B * H, C, C)
        @ chunk_state.permute(0, 2, 1, 3, 4).reshape(B * H, C, P * N)
    ).reshape(B, H, C, P, N).permute(0, 2, 1, 3, 4)

    # ---- 4. carry-in contribution ----
    c_decayed = c * acum.clamp(LOG_DECAY_FLOOR, 0.0).exp().unsqueeze(4)
    y_off = (
        heads_first(c_decayed, N)
        @ carry_in.permute(0, 1, 2, 4, 3).reshape(B * C * H, N, P)
    ).reshape(B, C, H, chunk, P).permute(0, 1, 3, 2, 4)

    y = (y_diag + y_off).reshape(B, L, H, P)
    return y[:, :T] if pad else y


def rotate_halves(v, phi):
    """Rotate the two halves of the trailing axis by `-phi`, matching `rotate_by_angle`."""
    cos, sin = torch.cos(phi), torch.sin(phi)
    d = v.shape[-1] // 2
    lo, hi = v[..., :d], v[..., d:]
    return torch.cat([lo * cos + hi * sin, hi * cos - lo * sin], dim=-1)


class Mamba3Mixer(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_state, chunk, conv_kernel=4):
        super().__init__()
        self.d_model = d_model
        self.h = n_heads
        self.p = head_dim
        self.n = d_state
        self.chunk = chunk
        self.d_inner = n_heads * head_dim
        self.bc = n_heads * d_state          # n_groups == n_heads
        width = 2 * self.d_inner + 2 * self.bc + n_heads + n_heads + n_heads * d_state // 2
        self.in_proj = nn.Linear(d_model, width, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        conv_ch = self.d_inner + 2 * self.bc
        self.conv = nn.Conv1d(
            conv_ch, conv_ch, conv_kernel, groups=conv_ch, padding=conv_kernel - 1
        )
        self.conv_kernel = conv_kernel
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.a_log = nn.Parameter(torch.zeros(n_heads))
        self.d_skip = nn.Parameter(torch.ones(n_heads))
        self.b_bias = nn.Parameter(torch.zeros(n_heads, d_state))
        self.c_bias = nn.Parameter(torch.zeros(n_heads, d_state))
        self.bc_norm = RmsNorm(d_state)

    def forward(self, u):
        B, T, _ = u.shape
        H, P, N = self.h, self.p, self.n
        proj = self.in_proj(u)
        z, xbc, dt_raw, lam_raw, theta_raw = torch.split(
            proj,
            [self.d_inner, self.d_inner + 2 * self.bc, H, H, H * N // 2],
            dim=-1,
        )
        # Depthwise causal convolution over x, B and C together.
        xbc = self.conv(xbc.transpose(1, 2))[..., :T].transpose(1, 2)
        xbc = F.silu(xbc)
        x, b_flat, c_flat = torch.split(
            xbc, [self.d_inner, self.bc, self.bc], dim=-1
        )

        b = b_flat.reshape(B, T, H, N) + self.b_bias
        c = c_flat.reshape(B, T, H, N) + self.c_bias
        b = self.bc_norm(b)
        c = self.bc_norm(c)
        x = x.reshape(B, T, H, P)

        dt = F.softplus(dt_raw + self.dt_bias)
        lam = torch.sigmoid(lam_raw)
        theta = theta_raw.reshape(B, T, H, N // 2)

        a = dt * (-torch.exp(self.a_log))
        g = lam * dt
        nxt = (1.0 - lam) * dt
        f = torch.cat([nxt[:, 1:], torch.zeros_like(nxt[:, :1])], dim=1)
        w = g + f

        # Rotating frame: cumulative angle, wrapped into [-pi, pi].
        phi = (dt.unsqueeze(3) * theta).cumsum(1)
        two_pi = 2.0 * math.pi
        phi = phi - (torch.round(phi / two_pi) * two_pi).detach()
        b = rotate_halves(b, phi)
        c = rotate_halves(c, phi)

        y = ssd_chunked(x, b, c, a, g, w, self.chunk)
        y = y + x * self.d_skip.view(1, 1, H, 1)
        y = y.reshape(B, T, self.d_inner) * F.silu(z)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.norm = RmsNorm(kw["d_model"])
        self.mixer = Mamba3Mixer(**kw)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Mamba3Lm(nn.Module):
    def __init__(self, vocab, d_model, n_layers, **kw):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(
            [Block(d_model=d_model, **kw) for _ in range(n_layers)]
        )
        self.norm = RmsNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()

    vocab = env("VOCAB", 8192)
    d_model = env("D_MODEL", 512)
    n_layers = env("LAYERS", 8)
    n_heads = env("HEADS", 16)
    head_dim = env("HEAD_DIM", 64)
    d_state = env("D_STATE", 64)
    chunk = env("CHUNK", 64)
    seq = env("SEQ", 512)
    batch = env("BATCH", 4)
    iters = env("ITERS", 10)

    dev = "cuda"
    torch.manual_seed(1234)
    model = Mamba3Lm(
        vocab,
        d_model,
        n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        d_state=d_state,
        chunk=chunk,
    ).to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f"torch   {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(
        f"model   d_model {d_model} layers {n_layers} heads {n_heads}x{head_dim} "
        f"state {d_state} chunk {chunk} vocab {vocab}"
    )
    print(f"batch   {batch} x {seq} tokens = {batch * seq} tokens/step")
    print(f"params  {params / 1e6:.2f} M")

    fwd = model
    if args.compile:
        fwd = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)
    ids = torch.randint(0, vocab, (batch, seq), device=dev)
    targets = torch.randint(0, vocab, (batch, seq), device=dev)

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16"
        else torch.autocast("cuda", enabled=False)
    )

    def step():
        opt.zero_grad(set_to_none=True)
        with autocast:
            logits = fwd(ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss

    t = time.perf_counter()
    step()
    torch.cuda.synchronize()
    print(f"warmup  {time.perf_counter() - t:9.2f}s")

    # Phase split, each phase synchronised.
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    with autocast:
        logits = fwd(ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
    torch.cuda.synchronize()
    f_ms = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    b_ms = (time.perf_counter() - t) * 1e3
    print(f"forward {f_ms:9.2f}ms")
    print(f"backward{b_ms:9.2f}ms")

    # Best and median rather than mean, for the same reason the Rust benchmark uses
    # them: this GPU shares its bandwidth with the rest of the machine.
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    best, median = times[0], times[len(times) // 2]
    print(f"step    best {best * 1e3:9.2f}ms  median {median * 1e3:9.2f}ms")
    print(
        f"tokens/s best {batch * seq / best:7.0f}  median {batch * seq / median:7.0f}"
        f"   peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
    )


if __name__ == "__main__":
    main()
