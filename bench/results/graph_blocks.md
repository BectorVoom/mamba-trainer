# Mixer blocks at Graph Mamba's shapes: time and device memory (GRAPH_MAMBA_PLAN.md §2.7)

Measured 2026-10-01 on the Apple M1 (wgpu, WGSL), working tree at `477789a` plus the uncommitted bf16 work, with
`examples/bench_graph_blocks`:

```bash
ROWS=4800 SEQ=17 HEADS=1 STATE=8 BIDIR=0 ./target/release/examples/bench_graph_blocks
```

One residual block (`ForwardBlock` or `BiBlock`, `src/models/entity/blocks.rs`), `d_model = head_dim = 64`, the input
requiring a gradient, loss = sum of the output; forward + backward, drained, minimum of 6 steps after 3 warm-up steps
(median within 9% of the minimum in every row). `HEADS` is per direction. `W` is the width of the
block's fused input projection (`SsmConfig::in_proj_width`). **tape** = bytes in use right after the forward pass
minus the bytes in use before it: the incremental live memory when the backward starts. It does not include the
block's input (19.9 MiB at 4,800 × 17 × 64), which exists before the baseline, nor the backward's own peak.
**reserved** = the pool's reservation after the run (`backend::reserved_bytes`): pages the default pool holds and
does not give back, including whatever the matmul tuner allocated while probing cold shapes. Memory is in MiB.

## Stage 1: `rows × 17` tokens

| block | heads/dir, `d_state` | `W` | rows | ms | launches | tape MiB | reserved MiB |
|---|---|---|---|---|---|---|---|
| forward | 1, 8 | 150 | 2,400 | 25.7 | 82 | – | 352 |
| forward | 1, 8 | 150 | 4,800 | 47.5 | 83 | 197 | 592 |
| forward | 1, 8 | 150 | 6,000 | 60.7 | 83 | 247 | 656 |
| forward | 1, 8 | 150 | 6,600 | 69.5 | 83 | 272 | 1,616 |
| forward | 1, 8 | 150 | 9,600 | 95.0 | 83 | – | 1,936 |
| forward, 2 layers | 1, 8 | 150 | 4,800 | 90.7 | 161 | – | 848 |
| bidirectional | 1, 8 | 300 | 4,800 | 97.4 | 87 | 375 | 1,680 |
| bidirectional, per-step scan backward forced | 2, 16 | 680 | 4,800 | 241.9 | 87 | 951 | 2,384 |
| bidirectional, default scan dispatch | 2, 16 | 680 | 4,800 | 528.1 | 87 | – | 2,896 |

## Stage 2: `B × T` nodes, bidirectional

| heads/dir, `d_state` | `W` | `B × T` | ms | tape MiB | reserved MiB |
|---|---|---|---|---|---|
| 2, 8 | 600 | 32 × 256 | 19.9 | 70 | 400 |
| 1, 8 | 300 | 32 × 256 | 11.2 | 36 | 336 |
| 2, 8 | 600 | 1 × 5,632 | 26.3 | 48 | 176 |
| 1, 8 | 300 | 1 × 5,632 | 22.3 | 25 | 80 |
| 1, 8 | 300 | 4 × 1,408 | 11.4 | 25 | 80 |
| 1, 8 | 300 | 8 × 704 | 10.1 | 25 | 80 |
| 1, 8 | 300 | 1 × 22,656 | 81.8 | 99 | 384 |
| 1, 8 | 300 | 4 × 5,664 | 36.9 | 99 | 384 |
| 2, 8, 2 layers | 600 | 32 × 256 | 38.1 | – | 416 |

## What it says

Empirical fits for these rows only (f32, `d_model = head_dim = 64`, this GPU); each row ran in its own process, so
the comparisons between rows are not interleaved.

1. **Time is close to proportional to positions × `W`: 3.9–4.6 ns per (position, projection column).** 47.5 ms for
   81,600 × 150 (3.9 ns); 97.4 ms for 81,600 × 300 (4.0); 241.9 ms for 81,600 × 680 (4.4); 19.9 ms for
   8,192 × 600 (4.0); 11.2 ms for 8,192 × 300 (4.6). Linear in rows: 10.7, 9.9, 9.9 ms per 1,000 rows at 2,400,
   4,800 and 9,600 rows.
2. **One long sequence does not follow that fit: it is bound by the scan walking it serially.** 1 × 5,632 at
   `W = 300` takes 22.3 ms where the fit gives 6.8; cutting the same nodes into 4 or 8 sequences gives 11.4 and
   10.1 ms. At 22,656 nodes: 81.8 ms as one sequence, 36.9 ms as four.
3. **The tape is 15–18 bytes per (position, projection column)**: 197 MiB for 81,600 × 150 (16.9 B), 375 MiB for
   × 300 (16.1 B), 951 MiB for × 680 (18.0 B), 36 MiB for 8,192 × 300 (15.4 B). That is about four f32 buffers as
   wide as the input projection.
4. **The pool's reservation is 3x to 9x the tape, and it jumps when one allocation passes 64 MiB.** Forward blocks:
   6,000 rows reserve 656 MiB, 6,600 rows reserve 1,616 MiB; the input projection's output
   (`rows · 17 · 150 · 4` bytes) goes from 61.2 MB to 67.3 MB between them, and 64 MiB is 67.1 MB (6,579 rows).
   Mechanism (`cubecl-runtime-0.10.0/src/memory_management/memory_manage.rs:203-257`, `memory_pool/sliced_pool.rs`):
   with this adapter's 512 MiB `max_page_size` the default `SubSlices` configuration has pools of (page, largest
   slice) = (8, 1), (32, 8), (128, 64), (512, 512) MiB, and an allocation goes to the first pool that accepts it.
   So a 61 MB tensor is a slice of a 128 MiB page and a 67 MB tensor a slice of a 512 MiB page. Pages are shared
   and reused, not owned by one tensor; the jump is the coarser page, not a wasted page per tensor. The same step
   shows at 8 MiB: 32 × 256 at `W = 300` has 9.8 MB tensors and reserves 336 MiB (9.3x its tape); 1 × 5,632 has
   6.8 MB tensors and reserves 80 MiB.
5. **A second forward layer adds its tape, not another pool**: 592 → 848 MiB.

## What it does not say

- One block alone: no embedding, no local encoder, no message passing, no optimizer.
- `apply_last` and the last-row-only gate (the plan's S3) are not implemented, so the forward rows are an upper
  bound for the tail layer.
- Whether a step is host- or GPU-bound: that needs host-submit time against drained time, not ms per launch.
- The backward's peak, and total device memory: neither number here is one of them.
- Other `d_model`, other dtypes, the CPU runtime (which may take the composed scan), other GPUs.
