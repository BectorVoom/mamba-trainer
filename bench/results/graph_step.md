# A Graph Mamba training step, measured (GRAPH_MAMBA_PLAN.md GM11)

Measured 2026-10-01 with `examples/profile_graph` on the **cpu** backend (AMD Ryzen AI 7 350, 16 threads, Linux),
release build, `f32`. The two reference workloads of §2.7, both at `d = 64`, `m = 4`, `s = 4`, `M = 8` (17 tokens per
node), one forward token layer, two bidirectional node layers, GINE, `d_state = 8`, one head per direction:

- **W-B** — 320 graphs of about 150 nodes (46,737 nodes), whole-graph batches of 4,864 rows, 10 steps an epoch.
- **W-A** — one graph of 22,662 nodes with 300 features, in 4 node parts of 5,666 rows.

**Every number here is the cpu runtime.** The plan's budget (§GM11) and its levers are written for the Mac GPU
through wgpu, where a step is bound by launches and by bandwidth; on the cpu runtime the calling thread computes the
kernels, so the same step is bound by arithmetic. The wgpu rows are still owed, and none of the conclusions below
about *time* should be carried over to a GPU. The counts (launches, uploads, reads, shapes, bytes) do carry over.

## The step

| | W-B | W-A |
|---|---|---|
| launches per step, warm | 748 | 757 |
| drained step (queue, then one read), 6 steps of one batch | 673–728 ms | 908–980 ms |
| host submit (time for the step to be queued) | 100% of the drained step | 100% |
| step inside a training epoch (10 / 4 steps, one read at the end) | 785–789 ms | 947–954 ms |
| uploads per epoch: first epoch → later epochs | 189 → **1** | 88 → **0** |
| reads while an epoch is queued | **0** | **0** |
| metadata-cache misses per epoch: first → later | 183 → 0 | 86 → 0 |
| matmul tuner misses | 0 (the tuner is GPU-only) | 0 |
| matrix products per step / distinct shapes (tuner keys on a GPU) | 102 / 60 | 99 / 54 |
| distinct row counts among them | 14 | 12 |
| live bytes after the forward pass: measured (estimated) | 408.5 MiB (396.9) | 533.1 MiB (566.7) |
| largest single allocation, forward and with the backward pass | 27.1 MiB | 110.2 MiB |
| pool reservation after 10, 20, 30 steps (W-A: 4, 8, 12) | 1,011 MiB, flat | 1,041 MiB, flat |

The first epoch's uploads are the shape tables and constants of every kernel shape of the run, each uploaded once
and then held; after that an epoch uploads its batch table and nothing else (node parts have no batch table: 0).
`MemoryEstimate` is within 3% (W-B) and 6% (W-A) of the measured live bytes. The pool reserves 2.0–2.5x the live
bytes and does not grow.

"Host submit is 100% of the drained step" is what the cpu runtime looks like: there is no queue for the host to run
ahead of. It is **not** evidence that the step is host-bound on a GPU. That question needs the wgpu run.

## Where the launches and the time are

One step, with `MAMBA3_TIME_LAUNCHES=1` (the queue drained after every launch, so read the split, not the total):

| region | W-B launches | W-B time | W-A launches | W-A time |
|---|---|---|---|---|
| backward pass (every rule) | 497 (66.5%) | 468 ms (65.0%) | 492 (65.1%) | 628 ms (66.0%) |
| `mixer.project` | 12 | 69.8 ms (9.7%) | 12 | 63.1 ms (6.6%) |
| `mixer.scan` + `scan.*` | 117 | 95.6 ms (13.3%) | 143 | 100.7 ms (10.6%) |
| `mixer.coef`, `mixer.conv`, `mixer.out` | 24 | 21.1 ms (2.9%) | 24 | 15.1 ms (1.6%) |
| `graph.data` (batch layout, walks, token features) | 3 | 28.4 ms (3.9%) | 3 | 66.7 ms (7.0%) |
| `graph.local` (token encoder) | 10 | 21.3 ms (3.0%) | 10 | 59.5 ms (6.3%) |
| `graph.node` (pad, unpad, residuals, FFN) | 34 | 7.8 ms (1.1%) | 30 | 7.5 ms (0.8%) |
| `graph.mpnn` | 12 | 2.5 ms (0.3%) | 12 | 2.7 ms (0.3%) |
| `graph.token`, `graph.embed`, `graph.head` | 17 | 2.9 ms (0.4%) | 9 | 5.0 ms (0.5%) |
| optimizer (AdamW and the gradient norm) | 21 | 2.7 ms (0.4%) | 21 | 2.6 ms (0.3%) |

Two thirds of the launches and of the time are the backward pass, which the tally does not split by region. Of the
forward pass, the mixer is 74% (W-B) and 55% (W-A); the graph model's own kernels — the whole data path, the token
encoder, message passing, pooling and the heads — are 63 ms of 720 (W-B) and 141 ms of 950 (W-A). The data path's
three launches replace the host loader: 28 ms for 82,688 token rows from 16 features, 67 ms for 96,322 rows from 300.
The optimizer is 21 launches (the plan estimated about 60).

## The switches of §2.7, A/B

W-B, one process per candidate, the candidates alternated over three rounds; each cell is the drained step over 6
steps of one batch. The baseline's own spread over the three rounds is ±2.5%, so differences below 5% are noise.

| candidate | min of three rounds | median of the medians | against the baseline | what else changes |
|---|---|---|---|---|
| baseline | 672.7 ms | 702.4 ms | 1.00 | |
| `TokenTail::Bidirectional` | 1,294.0 ms | 1,384.1 ms | **1.97x** | live 720.8 MiB (1.76x), largest allocation 94.6 MiB (3.5x), pool reservation 3,369 MiB (3.3x) |
| `d_state = 16` | 785.5 ms | 826.7 ms | **1.18x** | live 476.3 MiB (1.17x) |
| `LocalEncoder::Mean` instead of `Sgc` | 654.5 ms | 679.4 ms | 0.97 (noise) | same launches, same bytes |
| size buckets off | 704.8 ms | 737.1 ms | 1.05 | padded length 256 instead of 160; **33 uploads and 31 metadata misses per epoch** instead of 1 and 0 |
| `TokenSampling::Static` instead of `PerStep` | 683.9 ms | 703.6 ms | 1.00 | none: the walks are sampled either way, with a fixed counter |
| no message passing | 670.3 ms | 689.1 ms | 0.98 (noise) | 692 launches instead of 748 |

What the table says, on this backend:

1. **The forward-only tail (S3) is the largest single decision**: making the last stage-1 layer bidirectional doubles
   the step and more than triples the largest allocation and the pool. It stays the default.
2. **`d_state` 8 → 16 costs 18%** in time and 17% in live bytes (S1).
3. **Size buckets (S6) are worth keeping for a reason the plan did not list**: without them the padded length of the
   stage-2 rectangle differs from batch to batch, so every epoch meets new shapes — 31 metadata tables uploaded per
   epoch that the bucketed run uploads once. On a GPU each of those is also a tuner key.
4. **Resampling the walks every step is free** next to sampling them once (S10): the kernel runs in both cases.
5. **`Sgc` against `Mean`, and GINE against no message passing, are within the noise** in time. GINE is 56 of the
   748 launches and 2.5 ms of the forward pass.

Not measured: capacities against exact sizes (the cold-tuning cost of S5), which only exists where there is a tuner.

## What the indirection costs (protocol item 6)

`examples/bench_graph_kernels`: each gathering kernel against a baseline of matched work with no index in it, and
on a *banded* graph (a node's neighbours are the next rows in memory) beside a *random* one of the same degree.
64 features, 17 tokens per row, candidates interleaved, 7 rounds of 10 launches and one drain; minimum (maximum).

| | 1,200 rows | 4,800 rows |
|---|---|---|
| `token_features`, random graph (13.6 nodes per token) | 0.342 ms (0.530) | 1.515 ms (1.881) |
| `token_features`, banded graph (8.8 nodes per token) | 0.257 ms (0.341) | 1.158 ms (1.307) |
| `mean_dim` over a contiguous `[rows · 17, 14, 64]` tensor: the same source rows, the same output | 2.409 ms (2.856) | 12.258 ms (13.485) |
| `gine_aggregate`, random graph | 0.044 ms (0.064) | 0.128 ms (0.165) |
| `gine_aggregate`, banded graph | 0.036 ms (0.212) | 0.065 ms (0.102) |
| `sum_dim` over a contiguous `[rows, 6, 64]` tensor: the same `nnz` | 0.039 ms (0.092) | 0.096 ms (0.180) |

1. **Building token features by gathering is 7–8x faster than reducing the same rows laid out contiguously**
   (0.12–0.14x). The reason is what is read, not how: the gather reads the node feature table (1.2 MB at 4,800
   rows) over and over, and it stays in cache; the contiguous form has to exist first — 292 MB at 4,800 rows — and
   streaming it is bound by memory. The index is free next to that. This is the design of §2.2 (tokens as
   constants gathered from the store, never materialised per slot) measured rather than argued.
2. **Locality is worth 1.3x in the token gather**, but the banded graph's tokens also hold fewer nodes (its walks
   revisit), so most of that is less work, not better access.
3. **GINE's gather is within noise of a contiguous sum of the same size on a random graph (1.1–1.3x)** and faster
   on a banded one. These launches take about 0.1 ms, and the maxima show the noise at that scale.

The comparison is a cpu one: caches and memory bandwidth decide it. On a GPU the same benchmark answers a
different question (random access against sequential in device memory) and has to be run there.

## The levers of the plan, against their triggers

| lever | trigger | measured on cpu | pulled |
|---|---|---|---|
| fuse the walk sampler and the token features | `graph.data` > 10% of the drained step | 3.9% (W-B), 7.0% (W-A) | no |
| long-row reductions for pooling | graphs above ~1,000 nodes and `graph.head` > 5% | `graph.head` 0.1% | no |
| skip `pad_ragged` / `unpad_ragged` without message passing | their launches > 5% of `graph.node` | not separated by the tally; `graph.node` as a whole is 1.1% of the step | no |
| dense short-sequence scan | `mixer.scan` > 35% of the drained step | 13.3% of the step in the forward pass; its share of the backward pass is not attributed | no |
| elementwise fusion inside the mixer | `mixer.project` + `mixer.coef` > 40% | 10.9% of the step in the forward pass; backward not attributed | no |
| the others (recompute stage 1, release saved tensors early, checkpoint spacing, pool classes, scan-backward dispatch) | memory or GPU-specific conditions | not reached on this backend | no |

No lever's trigger is met on the cpu runtime, and the remaining time is in the mixer's own scopes and in the backward
pass, which is where the plan says to stop. The triggers are to be read again on the wgpu run: they were written for
it.

## What this does not say

- Anything about a GPU: no wgpu, CUDA or Metal number. Host submit against drained step, device time per kernel
  (CubeCL's profiler), `client.empty` time and tuner misses are meaningful only there.
- Where the backward pass's two thirds go: `profile_graph` attributes forward regions only.
- One machine, with the spreads given; nothing here was run with a quiet-machine guarantee beyond "nothing else
  started by this session".
