# Scan cost at Graph Mamba's shapes (GRAPH_MAMBA_PLAN.md §2.7, GS0)

Measured 2026-10-01 on the Apple M1 (wgpu, WGSL, plane 32), working tree at `477789a` plus the uncommitted bf16 work,
with `examples/bench_ssd_scan` built `--release --no-default-features --features wgpu`:

```bash
MAMBA3_SCAN_SHAPE=b,t,h,p,n MAMBA3_SCAN_CHUNKED=1 ./target/release/examples/bench_ssd_scan
```

`b` sequences of `t` steps, `h` heads, `p = head_dim`, `n = d_state`. Times are ms per call, pipelined (a run of calls,
one drain), mean of three rounds; the three rounds of every row agreed within 2%. "fused" is the recurrence kernel
(`ssd_scan_saving` + `ssd_scan_backward` with the saved checkpoints), the default on GPU-like devices. "chunked" is
`ssd_chunked` forward + backward with chunk 40 and the boundary state computed. Scan only: no projections, no gate.

## Stage 1: one row per node, `t = L = m·s + 1` tokens (4,800 nodes)

| t | h | n | fused fwd | fused bwd | fused fwd+bwd | chunked fwd+bwd |
|---|---|---|---|---|---|---|
| 17 | 4 | 16 | 7.4 | 360.8 | 368.3 | 190.0 |
| 17 | 4 | 12 | 7.6 | 50.8 | 58.4 | 174.5 |
| 17 | 4 | 8 | 6.2 | 39.7 | 45.9 | 158.7 |
| 17 | 2 | 16 | 3.8 | 180.4 | 184.2 | 95.0 |
| 17 | 2 | 8 | 3.2 | 19.9 | 23.1 | 79.6 |
| 17 | 1 | 8 | 1.6 | 9.8 | 11.5 | 35.5 |
| 16 | 4 | 16 | 5.9 | 187.7 | 193.6 | 178.9 |
| 16 | 2 | 8 | 2.7 | 17.3 | 20.0 | 77.1 |
| 33 | 2 | 8 | 5.7 | 36.4 | 42.0 | 154.7 |
| 65 | 2 | 8 | 10.8 | 61.7 | 72.4 | 413.9 |

## Stage 2: one row per graph, `t` = nodes

| b × t | h | n | fused fwd | fused bwd | fused fwd+bwd | chunked fwd+bwd |
|---|---|---|---|---|---|---|
| 32 × 256 | 4 | 16 | 0.71 | 20.7 | 21.4 | 25.4 |
| 32 × 256 | 4 | 8 | 0.64 | 3.06 | 3.7 | 24.6 |
| 1 × 5,632 | 4 | 16 | 2.9 | 19.1 | 22.0 | 17.4 |
| 1 × 5,632 | 4 | 8 | 2.5 | 8.8 | 11.4 | 17.2 |
| 1 × 22,528 | 4 | 8 | 9.6 | 34.8 | 44.4 | 59.7 |

## Which backward kernel ran (added after review)

`ssd_scan_backward` has two kernels. `chunked_backward` (`src/tensor/ops/ssd_scan.rs:1471`) picks the chunked one when
`chunk·head_dim`, `chunk·d_state` and `head_dim·d_state` are multiples of 256 (chunk 16 by default) and its shared
memory fits; otherwise the per-step (recurrent) one runs. At `p = 64` that means **`n = 16` took the chunked kernel and
`n = 8`, `n = 12` took the per-step kernel**: the rows above compare two kernels, not two state sizes.
`MAMBA3_SCAN_BACKWARD=recurrent` forces the per-step kernel:

| shape | h | n | fwd | bwd, default dispatch (chunked kernel) | bwd, forced per-step |
|---|---|---|---|---|---|
| 4,800 × 17 | 4 | 16 | 8.0 | 360.8 | 79.5 |
| 4,800 × 17 | 1 | 16 | 2.1 | – | 19.8 |
| 32 × 256 | 4 | 16 | 0.78 | 20.7 | 4.7 |
| 1 × 5,632 | 4 | 16 | 2.9 | 19.1 | 11.2 |
| 128 × 160 (entity decoder) | 4 | 32 | 10.8 | 37.3 | 37.4 (same kernel: `n = 32` is not eligible for the chunked one) |

`MAMBA3_SCAN_CHUNK=8` at 4,800 × 17, `n = 16` gives 79.6 ms: `8·16` is not a multiple of 256, so the per-step kernel
runs.

## What it says

1. **At `d_state = 16`, `head_dim = 64` the default dispatch picks the chunked backward kernel, and on this GPU it is
   1.7x to 4.5x slower than the per-step kernel** (360.8 vs 79.5, 20.7 vs 4.7, 19.1 vs 11.2 ms). The `t` 16 → 17
   doubling (187.7 → 360.8 ms) is a second chunk of 16. This is a dispatch choice, not a property of the state size.
2. **With the per-step kernel the backward is about proportional to `d_state`**: 79.5 ms at 16, 50.8 at 12, 39.7 at 8.
3. **At `n = 8` the cost scales roughly with heads** (9.8 / 19.9 / 39.7 ms backward for 1 / 2 / 4) **and with `t`**
   (23.1 / 42.0 / 72.4 ms forward + backward for 17 / 33 / 65: 0.28 / 0.27 / 0.23 µs per node-token).
4. **One long sequence is not a problem**: 1 × 22,528 runs in 44 ms although only `h·p = 256` units walk it.
5. With the per-step backward the fused scan beats the composed (`ssd_chunked`) one at every shape measured.

## What it does not say

- The "fused fwd+bwd" column is the **sum of two separately timed calls**. The "chunked fwd+bwd" column is one
  autograd pass (`y.mul(dy).sum().backward()`) that also computes the boundary state, with chunk
  `min(40, t)`, timed after the fused rows rather than interleaved with them. They are not the same work.
- The benchmark passes a materialised log decay and no skip. The real mixer uses the rate form and `d_skip`, whose
  per-head parameter gradients are extra folds in the backward. Numbers here are a lower bound for the mixer's scan.
- One machine; each shape in its own process; only one shape was measured at `n = 12`.
- Nothing here measures model quality: a smaller state is a capacity trade.
