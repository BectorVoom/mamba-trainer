# Graph Mamba plan: Graph Mamba Networks (GMN) on Mamba-3

Implements *Graph Mamba: Towards Learning on Graphs with State Space Models* (Ali Behrouz, Farnoosh Hashemi; KDD 2024;
arXiv:2402.08678v2) as a new model family `mamba3::models::graph`, with Rust and Python APIs in parity.

This is an execution document. It is written so that it can be worked start to finish **without reading the session that
produced it**, by an implementer who follows instructions literally (opencode tasks, one task per run; see
`docs/opencode_manual.md`). Every task gives the goal, the files, the API, the traps, the tests, the commands and a
"done when" line. Work the tasks **in the order given**. Each task lands on its own and leaves the cpu suite green.

Written against commit `477789a` plus the uncommitted bf16 work in the tree (`BF16_ACTIVATIONS_PLAN.md`). Line anchors
(`file:line`) are from that state; if a line moved, search for the quoted symbol.

---

## 0. What is being built, and what is not

A graph model in two stages, both made of the crate's existing bidirectional Mamba-3 block:

1. **Per-node stage.** Each node gets a short sequence of *subgraph tokens*: samples of its 1-hop, 2-hop, … m-hop
   neighbourhood drawn with random walks, each encoded to one vector by a small local encoder. A bidirectional Mamba
   scans that sequence from the farthest neighbourhood inwards; the last position, which is the node itself, is the
   node's new encoding.
2. **Node stage.** The node encodings of one graph, ordered by degree, form one long sequence. Bidirectional Mamba
   scans it, optionally summed with a message-passing layer over the real edges.

One parameter, the maximum walk length `m`, switches between the two tokenisations: `m >= 1` uses both stages;
`m = 0` skips stage 1 and is "GPS with the Transformer replaced by bidirectional Mamba".

**In scope:** node classification, graph classification, graph regression and multi-label graph classification; node
features (float or categorical), optional edge features; RWSE and Laplacian positional encodings; GINE and GatedGCN
message passing; mini-batches of many small graphs and partitions of one large graph; f32 first, 16-bit dtypes last;
Python bindings; synthetic learning tests; one real-data reproduction run.

**How it runs:** the dataset is uploaded once; after that, sampling the walks, building the tokens, assembling the
batch and computing the loss are device kernels. A training step uploads nothing and reads nothing (§2.1, §2.2).

**Out of scope (say so in the README section, do not build):** link prediction (PCQM-Contact), class-weighted losses
(COCO-SP, PascalVOC-SP use them), a nonlinear local encoder and the CRaWl walk-feature encoder (optional task GM15
only), message passing on directed graphs, datasets that do not fit on the device, BatchNorm, virtual nodes, any
dataset downloader (examples take a file path).

---

## 1. The paper, exactly

Source of truth: the arXiv v2 TeX source, §4 and Algorithms 1–2. Where the prose and the algorithm box disagree, this
section says which one the plan follows.

### 1.1 Neighbourhood sampling (§4.1)

Hyperparameters: `M` walks per token, `m` maximum walk length, `s` repetitions. For a node `v` and each walk length
`m̂ ∈ {1, …, m}`, sample `M` random walks of length `m̂` (that many edges) starting at `v`. The token is the **induced
subgraph** on the union of the nodes visited by those `M` walks:

```
T_m̂(v) = ∪_{i=1..M} nodes(walk_i),        token = G[T_m̂(v)]
```

Repeat `s` times per walk length, giving `T_m̂^1(v), …, T_m̂^s(v)`. `T_0(v) = {v}`. The paper's Eq. 5 writes the union
over `i = 0..M`; Algorithm 1 loops `M̂ = 1..M`. **Follow the algorithm: `M` walks.**

### 1.2 Token encoding and order (§4.1)

Each token is encoded by a local encoder `φ` (an MPNN such as GatedGCN, or CRaWl's walk features):

```
x_v^{(i-1)s + j} = φ( G[T_i^j(v)],  X_{T_i^j(v)} ‖ P_{T_i^j(v)} ),     1 ≤ i ≤ m,  1 ≤ j ≤ s
```

`φ` reads the **input** features (with the optional positional encoding `P` concatenated), not hidden states.

Order: reversed, so the largest neighbourhood comes first and the node itself comes last:

```
Φ_v = [ x_v^{sm}, x_v^{sm-1}, …, x_v^{2}, x_v^{1}, x_v^{0} ]          length  L = m·s + 1
```

Algorithm 1 builds `Φ_v` from `x^1..x^{sm}` only, but §4.2 states "the last state of the output corresponds to the
walk with length m̂ = 0, i.e., the node itself". **Follow the prose: `x_v^0` is the last token.** Tokens with the same
walk length are randomly shuffled; since the `s` samples of one walk length are i.i.d., resampling the walks (§2.3
`TokenSampling`, every step by default) has the same effect and no separate shuffle is implemented.

### 1.3 Bidirectional Mamba (§4.2, Algorithm 2)

Two Mamba blocks with separate weights; the second reads the rows in reverse and its output is reversed back:

```
Y_f = W_f1 ( SSM_f(σ(Conv(W_in,f · LN(Φ))))  ⊙ σ(W_f2 · LN(Φ)) )
Y_b = W_b1 ( SSM_b(σ(Conv(W_in,b · LN(rev Φ)))) ⊙ σ(W_b2 · LN(rev Φ)) )
y_out = W_out ( Y_f + rev(Y_b) )
```

### 1.4 The whole layer (Algorithm 1)

```
for v:  y(v) = last_row( BiMamba^{×L_t}(Φ_v) )            # stage 1; skipped when m = 0
Y = rows y(v), nodes ordered by degree
Y_out = BiMamba(Y) + Ψ(G, X ‖ P)                            # stage 2; Ψ is the optional MPNN
```

"Last layer(s)" of stage 2 may be stacked. Node order for `m = 0` and for stage 2: by degree (the paper's choice);
PPR, k-core and other centralities are named as alternatives (ablation row "PPR ordering").

### 1.5 Paper → crate mapping, and the deliberate deviations

| Paper | Here | Note |
|---|---|---|
| Two Mamba-1 blocks, summed, then `W_out` | one fused bidirectional `Mamba3Mixer` (`Mamba3MixerConfig::with_bidirectional`, `src/models/mamba3.rs:161`) inside `BiBlock` (`src/models/entity/blocks.rs`) | Same function class: `W_out(W_f1·a + W_b1·b) = [W_out W_f1, W_out W_b1]·[a; b]`, which is the fused mixer's single `out_proj` over direction-major channels. One mixer's launches instead of two. |
| Mamba-1 selective SSM | Mamba-3 SSM (`SsmConfig`) | **Deviation.** This crate is Mamba-3 (trapezoidal discretisation, rotational state). Reported numbers will not be bit-comparable with the paper. |
| `LayerNorm` | `RmsNorm` (what `BiBlock` uses) | **Deviation**, consistent with every other model in the crate. |
| Residual connections | pre-norm residual `x + mixer(norm(x))` | The paper does not write the residuals; the reference GPS layer has them. |
| `φ` = GatedGCN or RWF on the induced subgraph | `LocalEncoder::{Mean, Sgc}` (§2.3, §2.4) | **Deviation.** The linear members of that family, with token statistics, because they let tokens be built on the device from constants (§2.2). The nonlinear encoder is the optional GM15. |
| BatchNorm inside GatedGCN | `RmsNorm` | **Deviation**: the crate has no BatchNorm. |
| Walks sampled once before training; shuffle within a walk length | walks resampled on the device every step (`TokenSampling::PerStep`) | Equivalent in distribution to the shuffle (§1.2), and more samples than the paper's. `Static` reproduces sample-once. |
| Full-batch on one large graph | jittered stratified node partitions (§2.2) | `parts = 1` is full batch. |
| Stage 1 with `m = 0` "becomes a simple projection" | stage 1 is skipped | A length-1 scan adds nothing the embedding MLP does not. |
| Every stage-1 layer bidirectional | the last stage-1 layer is forward-only (`TokenTail::Forward`) | **Deviation, for speed** (§2.7 S3): only the last position is read, and there the backward scan has seen only that token. `TokenTail::Bidirectional` is the literal form. |
| Widths unspecified | lean stage-1 mixer, `d_state` 8 | §2.7 S1–S2, from measurement. |

Paper results to compare against (not pass/fail gates, because of the deviations): Roman-empire 0.8769 accuracy,
Amazon-ratings 0.5407, Minesweeper 0.9101 ROC AUC; Peptides-func 0.7071 AP, Peptides-struct 0.2473 MAE. Ablation on
Roman-empire: without bidirectional 0.8327, without MPNN 0.8620, PPR ordering 0.8612, without PE 0.8591. Search space:
`M ∈ {1,2,4,8,16,32}`, `s ∈ {0,1,2,4,8,16}`, 3–6 layers, lr 1e-3, 300 epochs; typically `M·s·(m+1) ≤ 200`.

---

## 2. Design (decided)

### 2.1 Rules

- **The on-device rule.** A dataset is uploaded once. After that a training step does on the host only: integer
  arithmetic on graph sizes (shapes and launch geometry need them), scalar kernel arguments, and queueing. No
  per-step table is built on the host, nothing is uploaded per step, nothing is read per step. Walk sampling, token
  construction, batch assembly, padding, pooling, the loss and the countable metrics are kernels. One small table
  per **epoch** (the batch list, §2.5) is the only recurring upload. One-time preprocessing at dataset build
  (ordering, symmetrising, PE/SE) stays on the host: it runs once and its result is uploaded once.
  Two precise limits: "upload" means a **buffer this crate creates from host data** (launch scalars travel as
  uniforms and are not counted); and "nothing is read" holds in **steady state**, once every matmul shape of the run
  has been tuned — a cold shape makes the tuner synchronise and read (uncounted by `read_count`), so tuner misses
  are counted separately (GM11).
- `index::scatter_add_rows` (`src/tensor/ops/index.rs:502`) reads its ids back and sorts them on the host on every
  call (~1.4 ms per read on Metal): it and `autograd::ops::embedding` (whose backward calls it) **must not appear on
  the graph model's training path**.
- Check **exit codes**: `cmd > /tmp/t.log 2>&1; echo "exit=$?"`. Use `--no-fail-fast` for full runs.
- Test on **cpu** and on the **local Mac GPU**: `cargo test --release --test <suite>` and
  `cargo test --release --no-default-features --features wgpu --test <suite>` (`docs/test_guidline.md`). Never build
  wgpu while cpu tests run (shared `target/`); never benchmark while anything else runs.
- **Python and Rust APIs in parity**; every Python entry point has a `.pyi` stub and a test. The Python side is the
  module `mamba3_graph` of the existing bindings (§2.6).
- Reductions accumulate in f32 whatever `E` is (`BF16_ACTIVATIONS_PLAN.md` B2): the kernel loads `E`, sums in an f32
  register and casts once on store. Not cast-launch, reduce, cast-launch.
- Performance claims come from `launch_count()`, `read_count()`, the new `upload_count()`, `launch_tally()` and
  interleaved A/B runs inside one process (`AGENTS.md`; manual at `/Users/ods/Documents/cubecl_manual/manual`, not
  the path `CLAUDE.md` gives).
- The crate has pre-existing clippy findings; do not fix them in these tasks. Watch disk (`df -h .`; a wgpu test
  build adds several GB; `target/release/examples` holds stale binaries that are safe to delete).
- wgpu results that look random: suspect a dropped launch first (poison outputs with NaN), then delete
  `~/.cache/mamba3`. `cargo build` does not verify a kernel: every kernel is launched in a test on cpu **and** wgpu.

### 2.2 The on-device data path — `src/models/graph/store.rs`, `src/tensor/ops/graph.rs`

**The device dataset** (`GraphStore<R, E>`), uploaded once by `GraphDataset::new`. Nodes are in canonical order
(§2.3), the nodes and edges of a graph are contiguous, ids are dataset-wide:

| table | shape | content |
|---|---|---|
| `x` | `[N, F]` `E` | float node features; or |
| `feat_ids`, `field_offset` | `[N, fields]`, `[fields + 1]` u32 | categorical features; multi-hot width `V = Σ vocab ≤ 1024` |
| `pe` | `[N, pe_dim]` `E` | PE/SE columns, for either kind of features (absent when `pe_dim = 0`) |
| `adj_off`, `adj_col` | `[N + 1]`, `[E]` u32 | the symmetrised, deduplicated graph as CSR, each row sorted ascending. Walks **and** message passing use it |
| `adj_rev` | `[E]` u32 | position of the reverse edge (`v → u` for `u → v`), so every adjoint is a gather over the same CSR. Built only with edge features or GatedGCN (§2.7 S12) |
| `edge_x` / `edge_ids` | `[E, Fe]` | optional edge features (categorical ones become a multi-hot, like nodes) |
| `y_node`, `split` | `[N]` u32 | class (`IGNORE = u32::MAX` unlabelled) and train / val / test bit flags |
| `y_graph`, `graph_split` | `[G, T]` `E` or `[G]` u32; `[G]` u32 | graph targets (NaN = missing) and split flags |

CSR orientation: row `u` of `adj` lists the edges **into** `u` (`adj_col` holds their sources); with a symmetric
graph the same row is also `u`'s out-neighbours, and `adj_rev` maps one to the other. Features of an edge and of its
reverse are separate rows and need not be equal.
The host keeps only `graph_ptr [G + 1]`, `edge_ptr [G + 1]`, the number of labelled items per split (counted at
build, so an empty split is an error then), the canonical → original permutation and the spec.
Message passing on a directed graph is out of scope: a spec with `mpnn` requires `symmetrize = true`.
`GraphDataset::new` estimates the store's bytes and refuses above `MAMBA3_GRAPH_MAX_BYTES` (default 2 GiB) with the
number and the remedy (a subset of the dataset; sharded residency is not built).

**A batch is a descriptor.** Shapes are host arithmetic; contents are written by one kernel:

```rust
pub struct GraphBatch<R: Runtime, E: FloatElem> {
    pub store: Rc<GraphStore<R, E>>,  // the dataset this batch indexes; `forward` reads it through the batch
    pub mode: BatchMode,              // Graphs | NodeSubset
    pub rows: usize,                  // Nb: row capacity (below); trailing rows are absent
    pub edges: usize,                 // Eb: edge capacity (Graphs mode); trailing edges are absent
    pub graphs: usize,                // B: graph capacity, a multiple of 8; trailing graphs are absent
    pub lengths: RaggedLengths<R>,    // device [B] true lengths (kernel G1) + host `max` (Nmax) and `all_full`
    pub gid: IdTensor<R>,             // [Nb] dataset node id of each row, IGNORE where absent      (kernel G1)
    pub row_graph: IdTensor<R>,       // [Nb] batch slot of each row, IGNORE where absent            (kernel G1)
    pub row_of: Option<IdTensor<R>>,  // NodeSubset only: [N] node id → row, IGNORE if not in batch  (kernel G1)
    pub epoch: Rc<EpochTable<R>>, pub index: usize,   // §2.5
    pub seed: (u32, u32), pub counter: u32,           // token sampling
}
```

- **Graphs mode** (many small graphs). Row `r` belongs to batch slot `g` (binary search of `r` in `node_off [B+1]`,
  at most `log2 B` steps) and is node `graph_start[g] + r − node_off[g]`. A neighbour `c` of that node is row
  `c − graph_start[g] + node_off[g]`. Edges translate the same way through `edge_start`, `edge_off`. The slot's
  dataset graph id (`graph_id[g]`, for `y_graph`, `graph_split`, PE signs and the order of predictions) is in the
  epoch table too.
- **NodeSubset mode** (one large graph, `P` parts): the canonical order is cut into blocks of `P` consecutive nodes;
  batch `j` of an epoch takes from block `r` the node `r·P + ((hash(r, epoch seed) mod P) + j) mod P` (reduce the
  hash first: adding `j` to a full-range u32 can wrap). The `P` batches of an epoch **partition the nodes exactly**;
  every batch is in canonical (degree) order and is stratified by degree; `⌈N / P⌉` rows in every batch; a row
  whose node id is `≥ N` (last block only) is absent. No table, no compaction, no read. A neighbour `c` is in the
  batch iff `row_of[c] != IGNORE`. A batch in which no row carries a label of the training split is an ordinary step
  with zero gradient (the loss denominator is `max(count, 1)`); with stratified parts and a split of more than a few
  percent it does not occur.
- **Capacities, not sizes.** Graphs mode fills a batch to a **row budget** (graphs are added while they fit), so
  `rows` is the budget rounded up to `row_quantum` (256) — the same number for every batch of a run. `edges` is the
  epoch's largest batch edge count rounded up to 4,096; `graphs` is rounded up to a multiple of 8; `lengths.max`
  (`Nmax`) to a multiple of 32. Absent rows have `gid = IGNORE`, absent graphs have length 0. Their activations are
  **ignored, not guaranteed zero** (a biased `Linear` turns a zero row into its bias): every kernel that crosses a
  row boundary — neighbours, padding, pooling, the loss, and their adjoints — skips them by the `IGNORE` test.
  Because a batch is arithmetic, capacities cost nothing to build, and they keep the shapes of a run few (§2.7 S5).

**Kernels** (`src/tensor/ops/graph.rs`). All are gathers: one unit owns an output element and loops over its
sources. None scatters, none uses atomics, none needs a transposed index.

| | kernel | output | differentiated |
|---|---|---|---|
| G1 | `batch_rows` | `gid`, `row_graph` (and `row_of`) from the epoch table | no |
| G2 | `walk_tokens` | `tok_node [Nb·L·C]` u32, `tok_w [Nb·L·C]` f32, `tok_stats [Nb·L, 3]` f32 | no |
| G3 | `token_features` / `token_counts` | `[Nb·L, F + pe_dim]`: `Σ_c w_c · (x ‖ pe)[node_c]`; or `[Nb·L, V + pe_dim]` with weighted multi-hot counts in the first `V` columns | no |
| G4 | `gather_rows_or_zero`, `multi_hot`, `safe_targets` | `[Nb, F + pe_dim]` / `[Nb, V + pe_dim]` node inputs; edge inputs likewise; `IGNORE` rows give zeros (the existing `index::gather_rows` has no such guard). `safe_targets`: class ids with 0 in place of `IGNORE`, float targets with 0 in place of NaN, and the f32 mask that says which were real | no |
| G5 | `pad_ragged`, `unpad_ragged` | `[Nb, d] ↔ [B, Nmax, d]` by arithmetic on `node_off` | yes, each is the other's adjoint |
| G6 | `segment_pool`, `segment_broadcast` | `[Nb, d] → [B, d]` (mean or sum) and its adjoint via `row_graph` | yes |
| G7 | `gine_aggregate` (+ `_dh`, `_de`), `gated_edge`, `gated_node` (+ adjoints) | §2.4 | yes |
| G8 | `confusion` | `[classes, classes]` counts from `argmax` ids, labels and the split flag | no |

**Why the token path needs no gradient.** `Mean` and `Sgc` are linear in the node embeddings, the node embedding
is one `Linear`, and the weights of a token sum to 1, so

```
Σ_c w_c · (x[node_c] W + b)  =  (Σ_c w_c · x[node_c]) W + b
```

G3 aggregates the **input features**, which are constants, and the model applies its embedding `Linear` to the
result. No gradient flows through a gather, so there is no adjoint and nothing to transpose; the embedding's own
backward is a dense product (and has no input gradient to compute). The same holds for categorical features with
weighted counts in place of `x`, and for the PE columns with their per-graph signs. Limits: the identity is exact in
exact arithmetic and to rounding in f32 (looser in 16 bits: use dtype-specific tolerances); it needs the embedding
to be one affine map with nothing between it and the aggregation — so no dropout, activation or activation
quantisation on `embed`'s input or output before the head's `gelu` — and it does not apply to absent rows, which are ignored.

### 2.3 Host data, canonical order, tokens, encodings — `src/models/graph/{data,tokenize,encoding}.rs`

**`GraphData`** (host, plain `Vec`s, validated on construction with messages that name the field):

```rust
pub struct GraphData {
    pub n_nodes: usize,
    pub edge_src: Vec<u32>, pub edge_dst: Vec<u32>,     // directed pairs; `symmetrize()` adds reverses and dedups
    pub x: NodeFeatures,                                 // Float { dim, data: Vec<f32> } | Categorical { fields, vocab, ids }
    pub edge_attr: Option<EdgeFeatures>,                 // Float { dim, data } | Categorical { .. }
    pub pe: Option<(usize, Vec<f32>)>,                   // [n_nodes, pe_dim], filled by encoding.rs or by the caller
    pub y: Labels,                                       // Node(Vec<i64>, -1 = unlabelled) | Graph(Vec<f32>) | GraphClass(Vec<u32>) | None
    pub graph_ptr: Vec<u32>,                             // [n_graphs + 1] node offsets; one graph = [0, n_nodes]
    pub masks: Option<Splits>,                           // train / val / test node masks (single-graph tasks)
}
```

**Canonical order** (once, on the host). `GraphData::canonicalize(order) -> (GraphData, Vec<u32> /* new → original */)`
sorts the nodes of every graph by the ordering key and relabels everything. After this, **storage order is sequence
order**: stage 2 never permutes on the device. `NodeOrder::{Degree { descending: false }, Ppr { alpha, iters },
KCore, Given}`; default ascending degree (important nodes late, §4.3 of the paper), ties by original id.
Predictions are returned in original order through the permutation.

**Tokens are sampled on the device, every step** (kernel G2). One unit per token `(row r, position t)`:

- Position `t = (m − i)·s + (j − 1)` holds repetition `j` of walk length `i`, `i` descending from `m` to 1;
  `t = L − 1` is the node itself. `L = m·s + 1`. Capacity per token `C = 1 + M·m` slots (validated `C ≤ 128`).
- `M` walks of `i` steps from `v = gid[r]`. Randomness is counter-based, from the crate's `hash_u32`
  (`src/tensor/ops/random.rs:135`), chained so that **every component passes through the avalanche**
  (`⊞`, `⊠` are wrapping u32 add and multiply):
  `k0 = hash_u32(v, seed_lo, seed_hi)`; `k1 = hash_u32(k0 ⊞ counter ⊠ 0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35)`;
  `k2 = hash_u32(k1 ⊞ (t·M + k) ⊠ 0x27D4EB2F, 0x165667B1, 0x9E3779B9)` for walk `k`; step `q` draws
  `hash_u32(k2 ⊞ q ⊠ 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F)` and moves to neighbour `adj_col[adj_off[u] + draw % deg(u)]`.
  Do **not** fold a component into `hash_u32`'s seed words by XOR: `seed_hi` enters after the multiplies, so
  `seed_hi ^ counter` only flips low bits of the result, and the next stage then sees walk `k` as walk `k ^ 1` — the
  same token set at every counter. A node without neighbours stays put. Tokens of a node depend on
  `(seed, counter, node)` only — not on the batch it is in.
- The unit writes visited nodes into **its own `C` output slots** in visit order (`v` first), skipping a node
  already there (linear scan of its own slots; the output is the scratch space, no local array), and pads with
  `IGNORE`. Then it writes the weights: `Mean`: `1/|T|`; `Sgc { hops: 1 }`: `(1 + deg_T(u)) / (|T| + 2|E_T|)` —
  the normalised column sums of `A_T + I`, where `A_T` is the induced adjacency **without self-loops** (a
  self-loop of the original graph is skipped here), `deg_T(u)` counts the token's *other* nodes found by binary
  search in `u`'s sorted adjacency row, and `|E_T|` counts unordered pairs. And the statistics
  `(ln(1 + |T|), ln(1 + |E_T|), i / m)` (`|E_T|` is 0 for `Mean`, which does no adjacency tests; a singleton token
  has `|T| = 1`, `|E_T| = 0`, weight 1).
- Kernel form: guards are **statements** (`if deg > 0 { … % deg … }`, `if id != IGNORE { … table[id] … }`):
  `select` evaluates both arms and does not protect a modulo by zero or an out-of-range index. `break` is
  available, `continue` is not (manual, *Loop Control* §1–2). A unit reads back only slots it has already written
  and owns.
- `TokenSampling::{PerStep (default), PerEpoch, Static}` only chooses `counter` (step, epoch, 0). Evaluation uses
  `Static`. Per-step sampling is what the paper's "randomly shuffle" asks for, and here it is free: nothing is
  stored and nothing is uploaded.
- In `NodeSubset` mode walks run on the **whole** graph (they read the store), so tokens may leave the batch.

**The host tokeniser is the test oracle.** `tokenize::tokens_host(store data, nodes, cfg, seed, counter)` implements
the same hash and the same visit order with ordinary loops; `hash_u32` gets a host twin tested bit-for-bit against
the kernel. It is not on any training path.

**Encodings** (`encoding.rs`, host, once, written into `GraphData::pe`):
- `rwse(g, k) -> [n, k]`: diagonal of `(D⁻¹A)^j`, `j = 1..k`, exact, by propagating a sparse indicator vector from
  each node (parallel over nodes with `std::thread::scope`). Refuse with a clear error if a `k`-hop ball exceeds
  `max_ball` (default 200,000 entries) and name the remedy (smaller `k`).
- `laplacian_pe(g, k) -> [n, k]`: the `k` eigenvectors of the symmetric normalised Laplacian with the smallest
  non-zero eigenvalues, per graph, by a dense cyclic-Jacobi eigensolver written in the crate (~60 lines, no new
  dependency). Refuse above `max_dense_nodes` (default 2048) with an error that points to RWSE; the paper's own §3
  says PE is the bottleneck on large graphs. Sign flips during training: a per-graph, per-epoch sign from
  `hash_u32`, applied inside G3 / G4 to the PE columns (a scalar flag per column range), so the stored table never
  changes.
- Graphs smaller than `k + 1` nodes: zero columns.

### 2.4 The model — `src/models/graph/{spec,layers,model}.rs`

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphMambaSpec {
    pub node_features: FeatureSpec,        // Float { dim } | Categorical { fields, vocab: Vec<usize> }
    pub edge_features: Option<FeatureSpec>,
    pub pe_dim: usize,                     // 0 = no PE/SE
    pub d_model: usize,                    // default 64
    pub tokens: WalkTokens,                // { max_hops: m, walks: M, repeats: s }; max_hops = 0 → node tokens only
    pub token_sampling: TokenSampling,     // PerStep (default) | PerEpoch | Static
    pub local: LocalEncoder,               // Mean | Sgc { hops: 1 } (default); Mpnn is reserved for GM15
    pub token_layers: usize,               // stage-1 layers (L_t), default 1; ≥ 1 iff max_hops ≥ 1
    pub token_tail: TokenTail,             // last stage-1 layer: Forward (default, §2.7 S3) | Bidirectional (paper-literal)
    pub node_layers: usize,                // stage-2 layers (L_n ≥ 1)
    pub mpnn: Option<MpnnKind>,            // Ψ: Gine | GatedGcn; None = "GMN−" when pe_dim == 0 too
    pub direction: ScanDirection,          // stage 2; reuse models::vision::ScanDirection; Forward = ablation
    pub node_sequences: usize,             // K: a single graph's nodes as K interleaved sequences (§2.7 S13); 1
    pub order: NodeOrder,
    pub task: GraphTaskSpec,               // NodeClass { classes } | GraphClass { classes } | GraphRegression { targets }
                                           // | GraphMultiLabel { labels };  + pool: Mean | Sum for graph tasks
    pub token_ssm: SsmConfig,              // stage-1 mixer (lean, §2.7 S2)
    pub node_ssm: SsmConfig,               // stage-2 mixer
    pub row_quantum: usize,                // 256
    pub dropout: f32, pub norm_eps: f32, pub seed: u64,
}
```

`validate()` rejects, naming the field: `d_model == 0`; `token_layers > 0` with `max_hops == 0` and the converse;
`walks == 0` or `repeats == 0` with `max_hops ≥ 1`; `1 + walks·max_hops > 128`; `Sgc { hops }` with `hops > 1`
and `LocalEncoder::Mpnn` (not built: GM15); `edge_features` without `mpnn`; `GatedGcn` or `edge_features` in `NodeSubset` training
(v1: per-edge tensors need an edge count the host does not know there; use `Gine` without edge features);
`node_sequences` that is 0, or above 1 in a spec trained on whole-graph batches; multi-hot width above 1024;
an `SsmConfig` that `SsmConfig::validate` rejects (rotational dynamics need an even `d_state`). JSON round-trips.

Mixer defaults (§2.7 S1, S2), everything not listed from `SsmConfig::default()`; `BiBlock` doubles heads and groups
for the two directions:

| | `n_heads` | `head_dim` | `d_state` | `n_groups` | input projection width (bidirectional / forward-only) |
|---|---|---|---|---|---|
| `token_ssm` | 1 | `d_model` | 8 | 1 | 300 / 150 at `d = 64` |
| `node_ssm` | 1 | `d_model` | 8 | 1 | 300 / – |

**Forward, with shapes** (`Nb` rows, `B` graphs, `Nmax` longest graph, `C` token capacity; launches of the data
path in brackets):

```
gid, row_graph           = batch_rows(epoch table, index)                       [1]   constants
--- stage 1 (max_hops ≥ 1) ---
tok_node, tok_w, stats   = walk_tokens(store, gid, seed, counter)               [1]   constants
tf                       = token_features(tok_node, tok_w, store.x ‖ pe)        [1]   [Nb·L, F + pe_dim] constant
tok                      = gelu( embed(tf) + stats_proj(stats) )                      [Nb·L, d]; embed is the model's one
                                                                                      Linear(F + pe_dim → d, one bias),
                                                                                      shared with nodes
Φ                        = tok.reshape([Nb, L, d])
Φ                        = BiBlock_t(Φ) × (token_layers − 1)                          [Nb, L, d]
y                        = Tail_t.apply_last(Φ)                                       [Nb, d]  node = last token
                                                                                      (TokenTail::Forward: §2.7 S3)
--- stage 2 ---
h0                       = embed( gather_rows_or_zero(store.x ‖ pe, gid) )      [1]   [Nb, d]; only if mpnn or max_hops = 0
h                        = y            (or h0 when max_hops = 0)
for layer ℓ in 0..node_layers:
    g = unpad_ragged( BiBlock_n.branch_ragged( pad_ragged(h), lengths ) )       [2]   mixer(norm(·)), no residual
    l = Mpnn_Ψ( u )   where u = h0 (ℓ = 0, Algorithm 1: Ψ(G, X‖P)) or h (ℓ > 0, GPS stacking); Ψ returns its
                      branch only — the residual is the `h +` of the next line
    h = h + g + l                                   (l absent when mpnn = None)
    h = h + MLP_ffn( RmsNorm(h) )                   d → 2d → d, GELU, dropout
out  = head( RmsNorm(h) )                           node tasks: Linear(d, classes) per row
       head( segment_pool(RmsNorm(h)) )             graph tasks: mean or sum over a graph's rows, then MLP
```

`pad_ragged` / `unpad_ragged` are skipped only when the layouts are identical: `graphs == 1`, in which case the
sequence **is** the batch — `[Nb, d]` reshaped to `[1, Nb, d]` with `lengths.max = Nb` and the true length on the
device (so `Nmax` rounding applies only when `graphs > 1`). With `node_sequences = K > 1` the single graph becomes
`[K, Nb/K, d]`: reshape to `[Nb/K, K, d]` and swap the first two axes (one existing `permute` each way, a copy),
sequence `k` holding rows `k, k + K, …`. `Nb` is rounded up to a multiple of `K`; the `K` true lengths come from
G1 on the device (which rows are absent depends on the epoch's hash), and the interleaving is undone before the
residual, `Ψ`, pooling and the loss (§2.7 S13). `all_full` alone is not enough: capacity rows follow
the last graph. Absent rows never reach a real row: they are last in every sequence, in no graph, and masked out of
pooling and the loss.

**`BiBlock::branch`, `apply_last`, and the ragged scan** (`src/models/entity/blocks.rs`, `src/models/mamba3.rs`):

- `BiBlock::branch(x) = mixer(norm(x))` (no residual); `apply` becomes `x + branch(x)`.
- **Ragged bidirectional scan.** In a padded batch the backward direction would read the pads first. Fix: reverse
  each row **within its own length**, so pads are last in *both* directions; a causal scan then never lets a pad
  reach a real position. `movement::reverse_bands_ragged(input, axis = 1, bands, lengths)`: as `reverse_bands`
  (`src/tensor/ops/movement.rs:363`) but position `d < len_b` reads `len_b − 1 − d` and `d ≥ len_b` reads `d`;
  an involution, so its own adjoint. `Mamba3Mixer::apply_ragged(input, &RaggedLengths)` is the bidirectional path
  of `project` / `apply_with_state_masked` (`src/models/mamba3.rs:374,608`) with both reversals in the ragged form.
  `RaggedLengths { ids: IdTensor [B], max: usize, all_full: bool }` carries what the host already knows, so
  choosing the plain `reverse_bands` path when `all_full` needs no read; construction validates `len ≤ max`.
  Verified against the source: the causal convolution, the rotation prefixes, `bc_norm`, the block norm, the gate
  and the projections are all causal or row-local, so real positions are exactly independent of pads.
  **Exception: activation quantisation** computes ranges across time and batch (`src/nn/quant.rs`);
  `apply_ragged` returns `Error::config` on a mixer built with it.
- **`ForwardBlock::apply_last(x) -> [B, d]`** (forward-only mixer): normalise **all** positions; project the
  non-gate bands for all positions and the gate band `z` for the **last row only**; scan all positions; slice the
  scan output `y` to `[B, 1, ·]`; call `finish` on that row; add the last row of the **original** input; reshape.
  The projection is one `Linear` whose first `d_inner` output columns are `z` (`project`,
  `src/models/mamba3.rs:374`): the split is two products with two column slices of the same weight (and bias),
  `x_last · W[:, ..d_inner]` and `x_all · W[:, d_inner..]`; the weight slices copy, and their adjoints zero-fill a
  weight-sized buffer. Weight quantisation and LoRA without dropout keep this form (LoRA: project all columns for
  all rows and slice `z`). With **activation quantisation**, whose ranges span the whole tensor on both
  projections, `apply_last` is `apply` followed by a slice: finishing one row would change what `out_proj`'s
  quantiser sees.
  (`d_skip` is already inside `y`; the autograd slices zero-fill the discarded positions in the adjoint.) The
  all-rows projection is then `W − d_inner` columns wide, and `out_proj` with both of its adjoint products runs on
  `B` rows, not `B·T`.

**Local encoder `φ`.** On the device path (§2.2): `Mean` — `gelu(embed(mean_u x_u) + stats_proj(stats))` — and
`Sgc { hops: 1 }` — the same with the degree-in-token weights of §2.3. `stats_proj` is a `Linear(3 → d)`; there is
no further MLP, because the stage-1 block's input projection follows immediately (§2.7 S11). Neither is in the paper; they are the linear
members of its "MPNN on the induced subgraph" family, chosen because they keep stage 1's input a constant. The
paper's nonlinear `φ` (`Mpnn`) needs per-slot activations (`Nb·L·C` rows) and a way to send gradients back to nodes;
it is designed in GM15 and built only if GM13 shows it is needed.

**Message passing `Ψ`** (`layers.rs`; kernels G7). Both run directly on the store's CSR, translating neighbours by
the batch descriptor; both adjoints are gathers over the **same** CSR, using `adj_rev` (the graph is symmetric, so
the out-edges of `u` are the reverses of its in-edges):

- `Gine` (`ε` fixed at 0, GIN-0, so there is no `dε`): the branch is `MLP(a)` with
  `a[r] = u[r] + Σ_{e ∈ in(r)} relu(u[src(e)] + ee[e])` in one launch, no `[E, d]` tensor (the `u[r]` term is GIN's
  self term, not the layer's residual);
  `du[r] = g[r] + Σ_{e ∈ in(r)} g[src(e)] ⊙ [u[r] + ee[rev(e)] > 0]` in one launch;
  `dee[e] = g[dst(e)] ⊙ [u[src(e)] + ee[e] > 0]` in one launch, only with edge features (`ee = Linear(edge input)`
  on the batch's edges, `[Eb, d]`; dataset edge ids and `adj_rev` are translated to batch edge rows by
  `edge_start` / `edge_off`). Without edge features `ee` is absent, the adjoint's test is `[u[r] > 0]`, and
  `adj_rev` is not read. Neighbours outside the batch are skipped in the forward **and** in both adjoints.
- `GatedGcn` (Graphs mode only in v1), edge states `e` `[Eb, d]` carried across node layers; `e⁰ = Linear(edge
  input)`, or one learned vector broadcast to every edge when the spec has no edge features. With `b_j = B u_j`:
  `ê_ij = C e_ij + D u_i + E u_j`; `D_i = Σ_{j'→i} σ(ê_ij') + 1e-6`; `z_i = Σ_{j→i} σ(ê_ij) ⊙ b_j / D_i`;
  branch `= relu(RmsNorm(A u_i + z_i))` (no `u_i +`: the residual is the model's); `e_ij' = e_ij + relu(RmsNorm(ê_ij))`,
  **not computed in the last node layer** (nothing reads it, and its norm would be a parameter without gradient).
  `gated_edge` forms `ê` from the three products in one launch; `gated_node` accumulates numerator and denominator
  in registers in one launch and saves `z` and `D`. Adjoints, one launch each, with `g` the gradient of `z`:
  `dê_ij = g_i ⊙ (b_j − z_i) / D_i ⊙ σ(ê_ij)(1 − σ(ê_ij))` plus the edge-update branch's contribution;
  `db_j = Σ_{i: j→i} g_i ⊙ σ(ê_ij) / D_i` and `d(E u)_j = Σ_{i: j→i} dê_ij` through `adj_rev`;
  `d(D u)_i = Σ_{j→i} dê_ij` over the contiguous in-edges; `d(C e)_ij = dê_ij`; `A`, the norms and the edge residual
  are ordinary autograd. BatchNorm is replaced by `RmsNorm`.

**Losses** (`loss.rs`). Masking comes **before** any indexing or arithmetic on a missing target: `safe_targets`
(G4) writes class 0 in place of `IGNORE` and 0 in place of NaN and returns the f32 mask (label present **and**
split flag matches **and** row / graph not absent). Then: node and graph cross-entropy = `cross_entropy_rows` on the
safe ids, times the mask; multi-label = `softplus(x) − x·y` on the safe targets, times the mask; regression = L1
(`mse` selectable) likewise. Every mean is `Var::masked_mean`, whose denominator is the count of real targets
(`max(count, 1)`). `cross_entropy_with(.., ignore_index)` is **not** used: it builds its mask on the host
(`src/train/loss.rs:85`). `masked_mean` multiplies by the mask, so a NaN that reached it would stay NaN — hence the
safe targets. `GraphTask<'a, R, E>` implements `TrainStep` with `Batch = GraphBatch<R>` (pattern: `EntityTask`,
`src/models/entity/loss.rs:31`), keeps the unscaled loss on the device for `read_losses`, and has
`with_loss_scale` for f16.

### 2.5 Epochs and batches — `src/models/graph/batch.rs`

```rust
impl<R: Runtime, E: FloatElem> GraphDataset<R, E> {
    pub fn new(spec: &GraphMambaSpec, data: GraphData, device: &Device<R>) -> Result<Self>;  // canonicalise, validate, upload once
    pub fn epoch_graphs(&self, batch_rows: Option<usize>, epoch: u64, split: Split) -> Result<Epoch<R, E>>;  // None = auto (§2.7 S9)
    pub fn epoch_nodes(&self, parts: Option<usize>, epoch: u64, split: Split) -> Result<Epoch<R, E>>;  // None = auto; Some(1): full batch
    pub fn memory_estimate(&self, batch_rows: Option<usize>, parts: Option<usize>) -> MemoryEstimate;  // §2.7 S10
}
impl<R: Runtime, E: FloatElem> Epoch<R, E> {
    pub fn len(&self) -> usize;
    pub fn batch(&self, i: usize) -> Result<GraphBatch<R, E>>;
}
```

- `epoch_graphs` takes the graphs of the split, shuffles on the host (a permutation of integers), walks **size
  buckets** (graphs sorted by node count, cut into runs of ~256 graphs, shuffled inside a run, runs shuffled) and
  fills each batch up to `batch_rows` rows. It writes the **epoch table**: for every batch its `graph_id`,
  `graph_start`, `node_off`, `edge_start`, `edge_off` — `(5B + 2)` u32 per batch at the epoch's graph capacity —
  uploaded **once per epoch** (tens of KB). `batch(i)` launches G1 and returns a descriptor. Filling by rows makes
  every step cost about the same and makes `rows` a constant of the run.
- `epoch_nodes` uploads nothing: a batch is `(parts, j, epoch seed)`.
- Bucketing reduces the pads stage 2 scans (the scan still runs over `Nmax` per row); it does not remove them.
- Everything a step needs from the host is in `graph_ptr`, `edge_ptr` and the permutation.

### 2.6 Python — the module `mamba3_graph` (`bindings/python/src/graph.rs`)

The graph model is a **module of this repository's existing Python bindings**, imported as `mamba3_graph`. It is
not a separate package or crate:

- Rust: `bindings/python/src/graph.rs`, in the same extension crate and the same compiled library as the RL and
  entity bindings; its classes are registered in the existing `#[pymodule]` (`bindings/python/src/lib.rs:248`) and
  declared `#[pyclass(module = "mamba3_graph", ..)]`.
- Python: `bindings/python/python/mamba3_graph/{__init__.py, __init__.pyi, py.typed}`, next to
  `python/mamba3_rl/`. `__init__.py` re-exports the graph classes and functions from the shared extension
  (`from mamba3_rl._mamba3_rl import GraphMambaSpec, GraphDataset, GraphMamba, ...`) together with what the graph
  API uses from it: `LrSchedule`, `read_count` / `reset_read_count`, `upload_count` / `reset_upload_count`,
  `launch_count`, `backend`, `supports_dtype`, `synchronize`, `build_info`. `pyproject.toml` lists the module so the
  wheel ships it (`[tool.maturin] python-packages = ["mamba3_graph"]`).
- One wheel, one copy of the core, one device client, one set of counters. Error mapping, dtype parsing and the
  schedule class are the ones already in `bindings/python/src/`; nothing is duplicated.

```python
import mamba3_graph as mg

spec = mg.GraphMambaSpec(
    node_features=300,                 # int = float features; mg.Categorical([vocab, ...]) for id fields
    edge_features=None, pe_dim=16, d_model=64,
    max_hops=4, walks=8, repeats=4,    # m, M, s;  max_hops=0 → node tokens only
    token_sampling="step",             # "step" | "epoch" | "static"
    local="sgc",                       # "mean" | "sgc"
    token_layers=1, token_tail="forward",              # or "bidirectional" (paper-literal, §2.7 S3)
    node_layers=2, mpnn="gine",                        # "gine" | "gated_gcn" | None
    d_state=8, token_heads=1, node_heads=1,            # §2.7 S1–S2; heads are per direction
    node_sequences=1,                                  # §2.7 S13
    bidirectional=True, order="degree",                # "degree" | "degree_desc" | "ppr" | "kcore" | "given"
    task=mg.NodeClassification(18),    # GraphClassification(c) | GraphRegression(t) | GraphMultiLabel(l); pool="mean"
    dropout=0.0, seed=0)
spec.to_json(); mg.GraphMambaSpec.from_json(s)

data = mg.GraphDataset(spec, dict(
    edge_index=int[2, E], x=float[N, F] | int[N, fields], edge_attr=None, y=..., graph_ptr=None,
    train_mask=bool[N], val_mask=..., test_mask=..., pe=None), symmetrize=True, dtype="f32")
    # one pass over each array, no device read, one upload (P4); float16 / 32 / 64 and any integer width accepted
mg.rwse(edge_index, num_nodes, k) -> float32[N, k];  mg.laplacian_pe(edge_index, num_nodes, k, graph_ptr=None)
data.num_nodes;  data.num_graphs

model = mg.GraphMamba(spec, learning_rate=1e-3, weight_decay=0.0, max_grad_norm=1.0, lr_schedule=None,
                      dtype="f32", loss_scale=None)
model.train_epoch(data, epoch, batch_rows=None)      # queues every step of the epoch; no read, one small upload;
                                                     # None = auto (§2.7 S9), or a row budget. GIL released while it
                                                     # waits; Ctrl-C stops it after a completed step (P2, P3)
model.train_epoch(data, epoch, parts=None)           # one large graph in node partitions; None = auto, 1 = full batch
model.memory_estimate(data, batch_rows=None, parts=None) -> dict(store, parameters, live, largest_allocation)
model.read_losses() -> list[dict(step, loss, grad_norm, learning_rate)]      # one read for everything queued
model.evaluate(data, split="val", metric="accuracy", batch_rows=9600) -> dict   # "accuracy"|"f1_macro"|"mae": one
                                                     # small read; "ap"|"roc_auc": one read of the scores
model.predict(data, split=None) -> float32 array, original node / graph order      # one read (P5)
model.save(path, step); mg.GraphMamba.load(path, dtype="f32"); model.num_parameters
mg.build_info() -> dict(profile="release" | "debug", backend, version)              # P7
```

Mirrors `EntityModel` (`bindings/python/src/entity_model.rs`) in its errors: unknown dict keys, wrong shapes and
out-of-range ids raise `ValueError` naming the key; the dataset and the model must share a dtype. Rust offers the
same surface through `GraphDataset`, `Epoch`, `GraphMamba`, `GraphTask`, `metrics::*`, `encoding::*`.

**The boundary, measured** on the existing entity bindings (`bench/results/python_boundary.md`,
`bindings/python/examples/bench_boundary.py`, release wgpu wheel, entity model, batch 32):

| | measured | consequence here |
|---|---|---|
| a trivial bound call | 29–46 ns | the number of Python calls is not a cost |
| `queue_train_step` | 255 ms wall per step, 37 ms of it CPU on the calling thread | the call mostly **waits**, holding the GIL |
| a second Python thread during training | 0.5% of its idle wake-up rate | P2 |
| `EntityDataset(...)` on 56 MB | 0.30–0.33 GB/s against 27.7 GB/s for a NumPy copy; 28 device reads | P4 |
| `predict` | 9 device reads per call | P5 |

**Rules for `bindings/python/src/graph.rs`.**

- **P1. One call per unit of work.** `train_epoch`, `evaluate` and `predict` each run their whole loop in Rust. Not
  to save call overhead (there is none to save) but so that Rust owns the loop: it can release the GIL across it,
  check for interrupts at safe points and batch the reads.
- **P2. The GIL is released whenever the call waits on the device or computes on the host without touching Python
  objects:** the body of `train_epoch`, the drain and read of `read_losses`, `evaluate` and `predict`, and the host
  preprocessing of `GraphDataset(...)` after the arrays have been read. The classes are `unsendable` (they hold
  `Rc`), and `Python::detach` (pyo3 0.29) wants a `Send` closure, so the borrowed Rust state goes through a small
  `struct Detached<T>(T)` with `unsafe impl Send`. SAFETY argument, to be written at the one place it is used, and
  checked against pyo3 0.29.2 (`src/marker.rs:562`, `src/impl_/pyclass.rs:1082-1110`): `detach` runs the closure on
  the calling thread, so nothing moves between threads; the closure captures no `Bound` / `Py` and returns none;
  an `unsendable` object panics on access from any other thread (`ThreadCheckerImpl::ensure`) and is leaked, not
  dropped, if another thread releases the last reference (`can_drop`); the `PyRefMut` the method holds refuses
  re-entry from this thread. So nothing can reach the `Rc`s concurrently — **provided every Python-visible object
  that shares state with the model (the dataset, which shares the store) is `unsendable` too.** No class here
  implements `__traverse__`. A single-
  threaded script gains nothing from this; a script with a data, logging or UI thread gets the ~85% of the wall
  time the call spends waiting.
- **P3. Interruptible.** `train_epoch` re-attaches (`Python::attach`) and calls `py.check_signals()` between steps
  when at least 50 ms have passed since the last check (off the main thread it does nothing and returns `Ok`;
  on it, a pending signal runs its Python handler, which is why the thread must be attached). On
  `KeyboardInterrupt` it stops queueing, synchronises, and raises: the model is left after a completed optimizer
  step and stays usable. The Rust API takes the check as a closure — `GraphMamba::train_epoch_with(&epoch,
  |step| ControlFlow)` — so the library does not depend on pyo3.
- **P4. Ingest is one pass per array, makes no device read, and uploads by value.**
  - Arrays are **borrowed** with their own dtype (`f32`, `f64`, `f16`, `i64`, `i32`, `u32`, `u8`, `bool`), not
    extracted with `AllowTypeChange`, which makes NumPy allocate a converted copy when the dtype differs.
  - **Read in logical order.** `PyReadonlyArray::as_slice` succeeds for C-contiguous **and** Fortran-contiguous
    arrays and returns memory order (numpy 0.29.0, `src/array.rs:759`, `src/untyped_array.rs:160`). Use the slice
    only when `is_c_contiguous()`; otherwise copy the array once, in logical order through `as_array()`, into a
    contiguous buffer and view that. The existing bindings
    take the slice unconditionally (`bindings/python/src/array.rs:46`, `entity_model.rs:139`) and so read a
    Fortran-ordered array scrambled: the same data gives a first-step loss of 21.5246 instead of 21.4098
    (`bench/results/python_boundary.md`). That is a bug to fix there, and a trap not to copy here.
  - The core takes views: `GraphDataView<'a>` (typed slices) is what `GraphDataset::new` consumes; the owned
    `GraphData` of §2.3 is a view over its own vectors. Canonicalisation writes each table **directly in canonical
    order and in the store's element type** into the buffer that will be uploaded: reading `x[perm[i]]`, converting
    and range-checking in the same pass. Edges: one pass to convert, check and count degrees, one to fill the CSR.
  - Everything is validated on the host slices before the upload; nothing is read back to derive a table
    (`EntityDataset::from_arrays` does that: `src/models/entity/batch.rs:929`).
  - Buffers go to the device by value through `Tensor::from_vec` / `IdTensor::from_vec`, i.e.
    `client.create(Bytes::from_elems(vec))`, which takes the typed `Vec` without copying it
    (`cubecl-common-0.10.0/src/bytes/base.rs:268`); `create_from_slice` copies twice on the host (manual,
    *Hardware-Adaptive Launch Geometry* §2.4). Peak host memory is one canonical copy of the dataset.
  - The passes that read NumPy memory run with the GIL held (another thread could otherwise write to the arrays);
    CSR construction, PE/SE and the upload run with it released.
  - Target, checked by GM9: `mg.read_count() == 0` across construction, and at least 1 GB/s of input on this
    machine (0.3 today for the entity dataset; the target is a first estimate, the read count is not).
- **P5. One read, one pass on the way out.** `predict` reads every batch's output in **one** `read_all`, then
  writes rows in original order straight into the result array's buffer (a `Vec` handed to NumPy with
  `into_pyarray`, no further copy). 16-bit models widen in that same pass. `evaluate` reads the device confusion
  matrix or sums once; `"ap"` / `"roc_auc"` read scores, labels and masks once and sort with the GIL released.
  `read_losses` is one `read_steps`.
- **P6. No Python in the loop.** No per-step callbacks, no Python-side metric, schedule or sampler; those are Rust
  objects configured from Python. A user who wants per-epoch logic writes a Python loop over `train_epoch`.
- **P7. A debug build says so.** `mg.build_info()` reports the profile, and constructing a `GraphMamba` from a
  debug build raises a `RuntimeWarning` naming `maturin develop --release`. (The two debug `.so` files in
  `bindings/python/python/mamba3_rl/` are 167 MB against the release wheel's 30 MB.)

Status of this section: the rules were checked against the pyo3 0.29.2, numpy 0.29.0 and cubecl 0.10 sources;
the codex review of it did not run (usage limit, 2026-10-01) and is still owed.

### 2.7 Speed and memory design

Sources: the CubeCL manual (`/Users/ods/Documents/cubecl_manual/manual/Cubecl`, chapters cited by number), the
repo's measurements (`bench/results/*.md`), two measurements made for this plan —
`bench/results/graph_blocks.md` (whole mixer blocks: time and device memory, `examples/bench_graph_blocks`) and
`bench/results/graph_scan_shapes.md` (the scan alone) — and two codex reviews, whose corrections are applied. The
manual's governing rule (ch. 11 §6, ch. 16 §5.6): **re-measure after every change, warm, one variable at a time,
candidates interleaved inside one process.** Numbers marked *estimate* are arithmetic; GM11 replaces them.

**Reference workloads** (`d = 64`, `m = 4, s = 4, M = 8`, so `L = 17`, `C = 33`):

| | W-A: one large graph | W-B: many small graphs |
|---|---|---|
| batch | 22,662 nodes, 4 parts: `Nb = 5,666`, `B = 1` | row budget 4,800 (≈ 32 graphs × 150 nodes), `Nmax ≈ 256` |
| stage-1 positions `Nb · L` | 96,300 | 81,600 |
| stage-2 positions `B · Nmax` | 5,666 | 8,192 |

**Two empirical fits** (M1, wgpu, f32, `d_model = head_dim = 64`, one residual block, forward + backward; `W` is
the width of the block's fused input projection, `2·d_inner + 2·n_groups·d_state + n_heads·(2 + d_state/2)` with
the default discretisation). They hold for the rows measured; other widths, dtypes, backends and the CPU runtime's
composed scan are not covered:

- **Time ≈ 4 ns × positions × `W`** (3.9–4.6 over the table). 81,600 positions: 47.5 ms at `W = 150`, 97.4 ms at
  300, 241.9 ms at 680. 8,192 positions: 11.2 ms at 300, 19.9 ms at 600. Linear in rows where measured: 10.7, 9.9
  and 9.9 ms per 1,000 rows at 2,400, 4,800 and 9,600 rows.
  Exception: **one long sequence is bound by the scan walking it serially** — 1 × 5,632 at `W = 300` takes
  22.3 ms where the fit gives 6.8, and the same nodes as 4 sequences take 11.4 ms.
- **Live memory after the forward pass ≈ 16 bytes × positions × `W`** (15–18 over the table; "tape" below: the
  bytes in use when the backward starts, not counting the block's input). **Pool reservation is larger and lumpy:**
  3x to 9x the tape in these runs, with a jump when a single allocation passes 64 MiB — a forward block reserves
  656 MiB at 6,000 rows and 1,616 MiB at 6,600. With the adapter's 512 MiB `max_page_size`, CubeCL's default
  `SubSlices` pools are (page, largest slice) = (8, 1), (32, 8), (128, 64), (512, 512) MiB; an allocation goes to
  the first class that accepts it, so 61 MB is sliced from 128 MiB pages and 67 MB from 512 MiB pages. Pages are
  shared and reused, so staying under 64 MiB lowers the reservation but does not bound it.

| block (one layer) | positions × `W` | ms | tape MiB | pool reservation MiB |
|---|---|---|---|---|
| stage 1, forward, 1 head, `d_state` 8 | 81,600 × 150 | 47.5 | 197 | 592 |
| stage 1, bidirectional, 1 head/direction, `d_state` 8 | 81,600 × 300 | 97.4 | 375 | 1,680 |
| stage 1, the first draft's block (2 heads/direction, `d_state` 16), per-step scan backward forced | 81,600 × 680 | 241.9 | 951 | 2,384 |
| the same with the default scan dispatch | 81,600 × 680 | 528.1 | – | 2,896 |
| stage 2, bidirectional, 2 heads/direction, 32 × 256 | 8,192 × 600 | 19.9 | 70 | 400 |
| stage 2, bidirectional, 1 head/direction, 32 × 256 | 8,192 × 300 | 11.2 | 36 | 336 |
| stage 2, 1 head/direction, 1 × 5,632 / as 4 × 1,408 | 5,632 × 300 | 22.3 / 11.4 | 25 | 80 |

**Where the work is.** With the defaults below, W-B is 47.5 ms of stage 1 and 2 × 11.2 ms of stage 2; W-A is
about 56 ms of stage 1 (by the fit) and 2 × 22 ms of stage 2. Neither stage is negligible, and everything outside
the mixer blocks (data path, embedding, `Ψ`, heads, optimizer) is not measured yet; whether a whole step is host-
or GPU-bound is for GM11's three clocks to say.

**Decisions.**

- **S1. `d_state = 8` in both stages.** `d_state` enters `W` (150 → 170 for the forward block at 16) and the scan
  is about proportional to it with the per-step backward (79.5 → 39.7 ms from 16 to 8, scan alone). It also avoids
  a dispatch problem: at `d_state = 16`, `head_dim = 64` the default rule picks the chunked backward kernel, which
  on this GPU is 1.7–4.5x slower than the per-step one — the whole first-draft block is 528 ms that way and 242 ms
  with `MAMBA3_SCAN_BACKWARD=recurrent`. Fixing the dispatch rule per device is a lever in GM11, outside this
  model. The smaller state is a capacity trade, priced by one row of GM13.
- **S2. One head per direction with `head_dim = d` in both stages** (`W = 150` forward, 300 bidirectional). By the
  fit, width is paid linearly: stage 2 at two heads per direction (`W = 600`) measured 19.9 ms and 70 MiB of tape
  per layer against 11.2 ms and 36 MiB. This is expansion 1 per direction where Mamba's usual choice is 2; a
  capacity trade, priced in GM13.
- **S3. The last stage-1 layer is forward-only and finishes one row** (`TokenTail::Forward`, default). Only position
  `L − 1` is read. The backward bands are reversed before the causal convolution and the scan, so at that position
  the backward direction has consumed nothing but the last **layer-input** token: it adds a function of that one
  input and doubles `W`. `ForwardBlock::apply_last` (§2.4) goes further: the gate band `z` is `d_inner` of the
  projection's `W` columns and is only needed where the output is read, so it is projected **for the last row
  only** — the all-rows projection is 86 columns wide instead of 150 — and the gate and `out_proj` run on `Nb`
  rows instead of `Nb·L`. How much that saves is **not known**: the norm, the convolution and the scan still cover
  every position, and the split adds two weight slices (copies) and their adjoints. GM4 measures time and tape,
  interleaved. With `token_layers = 1` stage 1 is one forward scan. Every extra
  (bidirectional) token layer costs 97 ms and 375 MiB of tape at W-B. This is a capacity change as well as a speed
  one; `TokenTail::Bidirectional` is the literal form of the paper.
- **S4. Tokens on the device, aggregated in feature space** (§2.2, §2.3). Two launches make the whole stage-1
  input; nothing is stored between steps, nothing has an adjoint. The cost that remains is the embedding `Linear`
  on `[Nb·L, F]`: for `F = 300` (W-A) two products of 3.7 GFLOP, *estimate* ~15 ms; negligible for small `F`.
- **S5. Few shapes, repeated.** The matmul tuner keys on the exact `(batch, m, n, k)`, the two transpose flags and
  the two dtypes, and the first time it sees a key on a GPU it times every candidate `PROBES = 5` times with a
  synchronisation each, plus one read to check the winner (`src/tensor/ops/matmul.rs:2714-2790`, `:2894`). A
  `Linear` on `[R, I]` or `[B, T, I]` is the key `(1, R, O, I)`; its input gradient `(1, R, I, O)`; its weight
  gradient `(1, I, O, R)` or a split of it. The row counts `R` of this model are: `rows·L` (stage-1 projections,
  `embed` on tokens — or its chunk sizes when the token features are chunked (S9) — and `stats_proj`), `rows` (the tail's `out_proj`, `embed` on nodes, `Ψ`'s MLP, the FFN, the node
  head), `graphs·Nmax` (stage-2 projections), `edges` (edge projections, GatedGCN's `C e`), `graphs` (the graph
  head). With sizes instead of capacities every one of them would be new on most steps of every epoch. With the
  capacities of §2.2, `rows` is one value per run, `edges` one or two, `graphs·Nmax` one per size bucket; how many
  keys that makes in a run is **measured** in GM11, not promised here. The alternative — classing the tuner's key
  — was reviewed and dropped: block candidates are only eligible when `n` and `k` are multiples of 4, so a class
  would need those predicates in its key.
- **S6. Nothing per step from the host** (§2.1). Steps are queued (`Trainer::queue_step`), the host is free while
  the device drains (manual ch. 5), and losses come back in one `read_steps` every `log_every` steps (a read is a
  fixed ~1.4 ms wait on Metal). This needs one fix outside the model: the multi-tensor AdamW and the gradient-norm
  kernel upload their small tables (lengths, decays, group offsets) **on every step, per chunk**
  (`src/tensor/ops/fused.rs:615`, `:1469`, `:4908`, `:5291`), about 21 µs each. Task GP1 keeps them on the device.
  `tests/graph_footprint.rs` then pins counted reads **and uploads** per steady-state step at zero.
- **S7. Kernel rules** (manual ch. 6–8, 10, 11, *Hardware-Adaptive Launch Geometry* §2.2b and pitfalls 1–4):
  - `launch_unchecked`, with every indexed access proved on the host: the store's tables are validated when built
    (`adj_off` non-decreasing from 0 to `E`, every `adj_col < N`, `adj_rev` — when built — an involution, lengths consistent) and
    are private fields; each launcher checks its operands' ranks, row counts and device and carries a `// SAFETY:`
    comment naming those checks.
  - Geometry from `backend::launch_1d_spans`. **Its span must be used**: on the CPU runtime it launches one cube in
    which unit `pos` owns lanes `pos·span .. min((pos + 1)·span, lanes)`, so a kernel body is the span loop of
    `gather_tokens_kernel` (`src/tensor/ops/entity_model.rs:585`), not `if pos < lanes`. Index from the flattened
    `ABSOLUTE_POS`; per-axis builtins only. `work_per_lane` = mean sources per output × vector width.
  - A lane is `(row, vector)` with the vector index innermost (`Vector<F, N>`, `line_dividing(&[d])`): writes are
    unit-stride and the lanes of an output row read a source row contiguously.
  - One f32 accumulator per lane in registers; cast on store. Every output element is written (outputs come from
    `Tensor::empty`), including zeros for absent rows.
  - Integer kernels (G1, G2, G8) are scalar: a hash per lane must not be shared by the lanes of a vector.
  - Counts, seeds, offsets and capacities are **runtime** arguments; `#[comptime]` is for what is fixed for a model
    (`C`, `M`, `m`, `s`, the encoder, the batch mode), so no kernel is recompiled as batches change.
- **S8. Not done, and why.** Host tokenisation, host-built index tables, a device cache of built batches and
  per-step one-copy uploads (earlier revisions): the device path makes them unnecessary; the one-time store upload
  does use the one-copy constructors (§2.6 P4). Tuner key classing: S5.
  Scatter-add with atomics: float atomics are non-deterministic in order and do not exist on cubecl-cpu (manual
  ch. 8). Recomputing stage 1 to save its tape (run it without a tape, backpropagate stage 2 to get the gradient
  of each node encoding, then re-run stage 1 in row chunks with a tape): it would allow full-graph stage-2 context
  on a large graph at one extra stage-1 forward, but node partitions already bound memory; a lever in GM11.
- **S9. A batch is sized so that no single allocation passes the pool's 64 MiB class.** `batch_rows = None`
  (auto) picks the largest multiple of `row_quantum` for which the **largest allocation of the step, found by
  enumeration**, stays under the threshold. The enumeration covers, per row: the stage-1 projection output
  (`L · W_all · 4` B, `W_all` = 86 with S3's split, 150 without, 300 for a bidirectional token layer), the scan's
  f32 checkpoints (`⌈L / 8⌉ · heads · head_dim · d_state · 4` B: three checkpoints at `L = 17`), the token tables
  (`L · C · 4` B each), the embedding output and the gate/norm buffers (`L · d · 4` B), the convolution's
  `L · (d_inner + 2·bc) · 4` B, the expanded `B` / `C`, and the padded stage-2 tensors and edge tensors of the
  batch. At the defaults and `L = 17` the checkpoints (6,144 B per row) are larger than the split projection
  (5,848 B), giving 10,752 rows; without the split 6,400; with a bidirectional token layer 3,072. The threshold is
  `device.client().properties().memory.max_page_size / 8` **when the default `SubSlices` pool is in use** (the
  class layout is not exposed, so it is an assumption stated in the log), overridable by
  `MAMBA3_GRAPH_TENSOR_MAX_BYTES`. `epoch_nodes` picks `parts` the same way for `parts = None`. Position-sized
  **constants** wider than that — the token features when `F + pe_dim` is large (316 columns for W-A) — are
  produced and embedded in row chunks, each under the threshold, and the chunks' outputs concatenated; all chunks
  stay alive for the embedding's weight gradient, so this removes the oversized allocation, not the bytes.
- **S10. A memory estimate before the first step.** `GraphDataset::memory_estimate` returns, separately: the store;
  parameters and optimizer state; the live bytes of a step, `≈ 16 B × Σ_layers positions × W` plus the enumerated
  constants (token tables, token features, checkpoints); and the largest single allocation. `epoch_*` refuses when
  store + parameters + live bytes exceed `MAMBA3_GRAPH_MAX_BYTES`, with the numbers and the remedies (smaller
  budget, more parts, fewer token layers, a 16-bit dtype). The pool's reservation is **not** estimated — it was 3x
  to 9x the live bytes in the block runs — and is reported by GM11 instead.
- **S11. The local encoder's head is one nonlinearity, not an MLP:** `tok = gelu(embed(tf) + stats_proj(stats))`.
  A second `Linear` on `[Nb·L, d]` would be two more products with their adjoints and two more position-sized
  activations. Dropping it is a capacity choice (the block's `RmsNorm` sits between the head and the block's
  projection, so the two are not algebraically one), made for speed and memory and priced by an ablation in GM13.
  Adding the statistics through their own small `Linear(3 → d)` also removes the concatenation.
- **S12. `adj_rev` is built only when something reads it** (edge features or GatedGCN): GINE without edge features
  needs no reverse-edge index — its adjoint is `g[r] + [u[r] > 0] ⊙ Σ g[neighbour]` over the same symmetric row.
  That is 4 bytes per edge of the store for the common case.
- **S13. `node_sequences` (default 1).** A single graph's node sequence can be cut into `K` interleaved sequences
  (row `r` goes to sequence `r mod K`, so each stays in degree order and stratified): measured 22.3 → 11.4 ms per
  block at 5,632 nodes and 81.8 → 36.9 ms at 22,656 with `K = 4`, same memory. Contract: the batch's row capacity
  is rounded up to a multiple of `K` (extra rows `IGNORE`); G1 writes the `K` true lengths on the device (a
  sequence can be empty), and both reversals of the ragged scan use them; the interleaving is undone before the
  residual addition, `Ψ`, pooling and the loss. In the **mixer** a node then sees `1/K` of the batch's nodes; `Ψ`
  still connects the sequences through real edges. A modelling trade, off by default; GM13 prices it.
- **S14. 16-bit activations are a memory setting here, not a speed one.** f16 / bf16 halve the `E`-typed buffers
  only: scan checkpoints, token weights and statistics stay f32, token ids u32, and matmul and norms allocate f32
  temporaries. The automatic row budget is recomputed by enumeration, not doubled. On this GPU f16 measured 2–3.5%
  faster at 1.9x the launches (`bench/results/entity_dtype.md`). GM12 opens them.

**Per step, from the measured blocks** (mixer blocks only; sums and the W-A stage-1 entry are *estimates* built
from single-block measurements in separate processes; the first-draft row is the configuration this plan started
from):

| | W-B: time, tape | W-A, per part: time, tape |
|---|---|---|
| stage 1: one forward layer, `W = 150` (before S3's split) | 47.5 ms, 197 MiB | ~56 ms, ~233 MiB (by the fit) |
| stage 2: two bidirectional layers, `W = 300` | 22.4 ms, 72 MiB | 44.6 ms, 50 MiB (22.8 ms with `node_sequences = 4`) |
| **blocks** | **~70 ms, ~270 MiB** | **~100 ms, ~285 MiB** |
| first draft: two bidirectional token layers at `W = 680`, default scan dispatch; two node layers at `W ≥ 600` | ≥ 1,096 ms, ≥ 2 GiB | – |

---

## 3. Tasks

Dependencies: `GM1 → GM3`; `GM2` (store and kernels) needs GM1; `GM4` (mixer) is independent; `GM5` needs GM2;
`GM6` needs GM1–GM2; `GP1` (optimizer tables) is independent; `GM7` needs everything before it; then in order. GS0 (the scan
measurement behind §2.7) is done.

### GM0. Skeleton
Files: `src/models/graph/mod.rs` (new, module doc summarising §0 and §1.5), `src/models/mod.rs` (declare `pub mod
graph;`, extend the module-list doc comment), `src/tensor/ops/mod.rs` (declare `pub mod graph; pub use graph::*;`),
empty `src/tensor/ops/graph.rs`. No behaviour. Done when `cargo build --release` and `cargo doc --no-deps` are clean
of new warnings (`#![warn(missing_docs)]` is on).

### GM1. Host graph data — `src/models/graph/data.rs`
`GraphData` (owned), `GraphDataView<'a>` (typed borrowed slices: the form everything below consumes, §2.6 P4),
`NodeFeatures`, `EdgeFeatures`, `Labels`, `Splits`, `NodeOrder`, `canonicalize`, `symmetrize`, the CSR builder
(`adj_off`, `adj_col` sorted per row, `adj_rev`), validation. `canonicalize` takes a view and writes each table once,
in canonical order and in the requested element type, into a fresh buffer; it never builds an intermediate
`GraphData`.
- Traps: self-loops are kept once (their reverse is themselves); duplicate edges are removed (their features: keep
  the first, say so); `graph_ptr` must be non-decreasing, start at 0, end at `n_nodes`; an edge that crosses two
  graphs is an error; `u32` overflow of `n_nodes` / edge count is an error, not a wrap.
- PPR order: power iteration (`alpha` 0.15, 50 iterations, per graph); k-core: the linear bucket algorithm.
- Tests `tests/graph_data.rs`: canonicalise a 6-node graph with known degrees → expected order and relabelled CSR;
  ties keep original order; the permutation maps predictions back; symmetrise is idempotent; `adj_rev[adj_rev[e]]
  == e` and it points at the reverse pair; each validation error names its field; two-graph `graph_ptr` keeps
  graphs separate; PPR on a star ranks the centre last (ascending); k-core of a triangle with a pendant node =
  `[2, 2, 2, 1]`; a view over `f64` features and `i64` edges gives the same canonical tables as the `f32` / `u32`
  one; an edge id above `u32::MAX` or below 0 is an error naming `edge_index`.
- Done when: `cargo test --release --test graph_data` passes (host-only; no device).

### GM2. Device store and data-path kernels — `src/models/graph/{store,tokenize}.rs`, `src/tensor/ops/graph.rs`, `src/backend.rs`
`GraphStore` (§2.2), kernels G1–G6 and G8 under the rules of §2.7 S7, the host oracle `tokenize::tokens_host` and
the host twin of `hash_u32` (§2.3), `Var::pad_ragged` / `unpad_ragged` / `segment_pool`, and
`backend::upload_count()` / `reset_upload_count()`: every buffer this crate creates from host data, counted at the
creation sites — `Tensor::from_data` (`src/tensor/base.rs:96`), `IdTensor::from_slice`
(`src/tensor/ops/index.rs:60`), both branches of `meta_handle` (`src/backend.rs:817`, `:828`; expose
`meta_miss_count()` too) and the three direct uploads of `scatter_add_rows` (`src/tensor/ops/index.rs:542`). Fills
(`zeros` / `full` / `ones`) are kernels and launch scalars are uniforms: neither is an upload. Add the by-value
constructors `Tensor::from_vec` and `IdTensor::from_vec` (`client.create(Bytes::from_elems(vec))`, counted like
the others) and build the store with them (§2.6 P4). Read the manual first: `INDEX.md`,
`07_memory_coalescing.md`, `10_grid_stride_occupancy.md`, `11_launch_overhead_and_transfers.md`,
`Hardware-Adaptive_Launch_Geometry.md`.
- Traps: G2 is the first kernel in the crate that uses its **output as scratch** (it re-reads slots it wrote):
  check on wgpu with NaN / sentinel poisoning that every slot of every token is written, and that a launch is not
  silently dropped (the Metal failure of `wgpu-metal-dropped-launch`). `% deg` with `deg == 0` must not be
  evaluated. Binary search bounds are data-dependent loops: precedent `bucket_scatter_add_kernel`
  (`src/tensor/ops/index.rs:476`). For `IGNORE` rows G3 and G4 write zeros; G5 and G6 skip them. G6 with one graph of thousands of rows
  has few lanes doing long sums: correct, and a lever in GM11, not a bug.
- Tests `tests/graph_kernels.rs` (cpu and wgpu):
  - `hash_u32` host twin (wrapping arithmetic) equals the kernel on 10,000 inputs.
  - Resampling is real: over 200 nodes, the token **sets** (not the visit order) at counters 0, 1, 2 differ for
    most tokens of walk length ≥ 2; the first step of 20,000 walks from a degree-5 node picks each neighbour
    within ±5% of uniform.
  - G1: both modes against host arithmetic; in `NodeSubset` mode the `P` batches of an epoch cover every node
    exactly once, each batch ascending, absent rows only at the end; quantised rows are `IGNORE`.
  - G2 equals `tokens_host` exactly (ids) and to 1e-6 (weights, stats) on a random 300-node graph, for `Mean` and
    `Sgc`; token `L − 1` is `{v}`; every token contains `v`; every token node is within `i` hops (BFS);
    `|T| ≤ 1 + M·i`; an isolated node gives `{v}` everywhere; weights of a token sum to 1; tokens of a node are the
    same in two different batches of the **same dataset** (tokens are keyed by dataset node id); on a path graph
    with large `M` a token is an interval around `v`; a graph with self-loops gives the same `Sgc` weights and
    `|E_T|` as the same graph without them.
  - G3 / `token_counts` against host loops, including bf16 / f16 inputs where the dtype exists (accumulation in
    f32: compare with `E::from_f32(reference)`; skip with a printed reason otherwise).
  - G5: `unpad(pad(h)) == h` on real rows; adjoint identity `⟨pad x, y⟩ = ⟨x, unpad y⟩` (both are linear); pad
    positions are written as zero; absent rows and absent graphs are skipped. G6: against host means; adjoint
    identity; finite-difference check; absent rows filled with large values do not change any pooled row.
  - G4 `safe_targets` and G8: against host loops, with ignored labels, NaN targets, split flags and absent rows.
  - Every kernel: forward launches as listed in §2.4, `read_count() == 0`, `upload_count() == 0`; a launch with
    more than 65,535 × 256 lanes runs on wgpu (the cube count spreads over Y).
- Done when: both backends pass twice in a row.

### GM3. Encodings — `src/models/graph/encoding.rs`
`rwse`, `laplacian_pe`, `jacobi_eigh` (private).
- Tests `tests/graph_encoding.rs`: RWSE on a triangle `= [0, 1/2, 1/4, …]` and on a 4-cycle (odd powers 0) against a
  dense matrix-power reference; LapPE: `‖L v − λ v‖ < 1e-4` for every returned pair, eigenvalues ascending, the
  trivial eigenvector excluded per connected graph of a two-graph input, columns orthonormal; a graph smaller than
  `k + 1` is zero-padded; both size limits raise the documented errors.

### GM4. Mixer additions — `src/tensor/ops/movement.rs`, `src/autograd/ops.rs`, `src/models/mamba3.rs`, `src/models/entity/blocks.rs`
As §2.4: `reverse_bands_ragged`, `RaggedLengths`, `Mamba3Mixer::apply_ragged`, `BiBlock::{branch, apply_ragged,
branch_ragged}`, `Mamba3Mixer::apply_last`, `ForwardBlock::{branch, apply_last, apply_ragged}` (a forward-only
mixer ignores lengths: pads are already last).
- Tests `tests/graph_ragged.rs`: the kernel against a host reference; applying it twice is the identity;
  **padding invariance**: a `[1, n, d]` sequence through `BiBlock::apply` equals rows `0..n` of the same sequence
  padded to `[2, Nmax, d]` (with a second, longer row) through `apply_ragged`, to 1e-5 on cpu and 1e-4 on wgpu, with
  the pad rows filled with large random values to prove they cannot leak; gradients w.r.t. the real rows match too;
  `all_full` lengths reproduce `apply` bit-for-bit with the same launches; a mixer with activation quantisation is
  refused; `ForwardBlock::apply_last(x)` equals row `T − 1` of `ForwardBlock::apply(x)` (1e-5) with equal parameter
  gradients (including the projection's weight and bias, assembled from the two column slices), its all-rows
  projection has `W − d_inner` columns and its `out_proj` product `B` rows, not `B·T` (check through
  `MAMBA3_TRACE=1`); the gradient w.r.t. the block's input matches too; the same holds with `post_gate_norm`, with
  weight quantisation and with LoRA (dropout off); with activation quantisation `apply_last` equals `apply` +
  slice exactly; existing suites `ssm`, `model`, `entity_blocks`, `entity_model`, `entity_footprint` unchanged.
- Measure (interleaved, one process, `examples/bench_graph_blocks` extended with `APPLY_LAST=1`): `apply_last`
  against `apply` + slice at 4,800 × 17 — time, tape and reserved memory — and add the rows to
  `bench/results/graph_blocks.md`. §2.7 S3 makes no estimate; if the split is not faster, keep the last-row
  `finish` and drop the split.

### GM5. Message-passing kernels and layers — `src/tensor/ops/graph.rs`, `src/models/graph/{spec,layers}.rs`
`GraphMambaSpec` (+ `with_*` setters, `validate`, JSON), `FeatureSpec`, `LocalEncoder`, `MpnnKind`, `GraphTaskSpec`,
kernels G7 with their `Var` ops, `Gine`, `GatedGcn` (both implement `Module`), `LocalEncoderModule`.
- Tests `tests/graph_layers.rs` (cpu and wgpu): `Gine` and `GatedGcn` on a 5-node graph and on a batch of two
  graphs against a host implementation of the formulas of §2.4 that reads the layer's own weights
  (`named_parameters`); a node with no neighbour keeps a finite, correct value; a graph with self-loops; for each
  nonlinear kernel a **vector-Jacobian check at fixed inputs** — `⟨J δ, g⟩` from a finite difference along a random
  `δ` against `⟨δ, Jᵀ g⟩` from the adjoint kernels, for `u`, edge inputs, `b`, `ê` — and a finite-difference check on
  one weight of each layer, with and without edge features; GatedGCN's last layer computes no edge update and
  every remaining parameter has a gradient; `NodeSubset` mode: edges to nodes outside the batch contribute
  nothing, forward and backward; `Sgc` on a token
  equals the host formula followed by the head (`gelu(embed(·) + stats_proj(stats))`); with constant features two tokens of different size get different
  encodings (the `stats` columns); the commutation of §2.2: `embed(token_features(x))` equals the weighted mean of
  `embed(x)` rows to 1e-5; spec JSON round trip; every `validate` error names its field.

### GM6. Dataset, epochs, batches — `src/models/graph/batch.rs`
As §2.5: `GraphDataset`, `Epoch`, `EpochTable`, `GraphBatch`, `Split`, the store size guard, the automatic row
budget and parts (§2.7 S9), the memory estimate (§2.7 S10, `GraphDataset::memory_estimate`), and the chunked token
features.
- Traps: an empty split is an error naming the split; labels of unlabelled nodes never reach the loss as a class
  id; the epoch table is the only upload of an epoch — build it with one `IdTensor::from_slice`.
- Tests `tests/graph_batch.rs`: a batch's `gid` / `row_graph` / lengths reproduce a brute-force concatenation of
  its graphs, and `graph_id` names them; every graph of the split appears exactly once per epoch; two epochs give
  different orders; with bucketing the mean of `Nmax / mean nodes` over the batches of a dataset with sizes 10..400
  is below 1.3 (and above 1.8 without); `rows` is the same for every batch, `edges` the same within an epoch,
  `graphs` a multiple of 8, `Nmax` a multiple of 32; a graph larger than the row budget is an error naming it;
  `upload_count()` is 1 for an `epoch_graphs` and 0 for an `epoch_nodes` and for every `batch(i)`; an empty split is
  an error at `epoch_*`; the size guard raises the documented error; with the automatic budget **no allocation of
  a training step exceeds the threshold** (instrument `Tensor::empty` in the test build, or read CubeCL's memory
  log) for a forward tail, a bidirectional token layer, `L = 17` and `L = 65`, and the budget follows
  `MAMBA3_GRAPH_TENSOR_MAX_BYTES`; with 316 feature columns the token features come in chunks, each under the
  threshold, and the embedded result equals the unchunked one; `memory_estimate`'s live bytes are within a factor
  1.5 of the measured ones (`bytes_in_use` after a forward pass) for the two specs of the footprint test; with
  `node_sequences = 4` and a row count that is not a multiple of 4 the outputs and gradients of real rows equal a
  host-permuted reference, and an absent last row changes nothing.

### GP1. Keep the optimizer's tables on the device — `src/tensor/ops/fused.rs`, `src/train/optim.rs`
A prerequisite of the zero-upload step, and a gain for every model in the crate. Today `adamw_step_multi` (both
the plain and the master-weight path) uploads a `lens` id table and a `decay` table per chunk on every step, and
`sum_squares_multi` uploads a `[3 · width]` meta table per chunk (`src/tensor/ops/fused.rs:615`, `:1469`, `:4908`,
`:5291`).
- Make them residents: the u32 tables through `backend::meta_handle` (content-keyed, so a changed parameter list
  is simply a new entry), and the decay table through an f32 twin of it or a small cache on the optimizer keyed by
  the same content. No invalidation logic beyond the content key.
- Tests: `tests/adamw_multi.rs` unchanged (bit-identical updates); a new assertion in `tests/train.rs`: after two
  warm-up steps of the LM task, a step makes `upload_count() == 0`; changing a parameter's weight-decay flag
  between steps still gives the right update; `rl_update_footprint` and `entity_footprint` launch pins unchanged.
- Report in §5 the uploads per step before and after for the entity model's step (from `upload_count()`).

### GM7. Model, task, losses, metrics — `src/models/graph/{model,loss,metrics}.rs`
`GraphMamba<R, E>`: `GraphMambaSpec::init`, `forward(&GraphBatch) -> Var`, `Module::visit` (stable parameter names:
`embed.*`, `local.*`, `token.{i}.*`, `node.{i}.{mixer,mpnn,ffn}.*`, `head.*`), `save` / `load` (as
`EntityModel::save`, `src/models/entity/model.rs:2253`), `predict`, `evaluate`, and
`train_epoch_with(&epoch, control)` — queue every step of an epoch, calling `control(step) -> ControlFlow<()>`
between steps and, on `Break`, synchronising and returning after the last completed step (§2.6 P3). `predict`
reads all batches in one `read_all`. `GraphTask` and the losses of §2.4.
`metrics`: `accuracy` and `f1_macro` from the device confusion matrix (G8) accumulated over the batches of the
split and read once; `mae` from a device sum; `average_precision` and `roc_auc` (rank-based, ties averaged) on the
host from one batched read of the scores together with their labels and masks (the host does not keep labels).
`predict` / `evaluate` run under `no_grad` **and in evaluation mode** (`set_training(false)`: `no_grad` alone does
not switch dropout off), with `TokenSampling::Static` and without PE sign flips, and restore the previous mode.
Evaluation runs the model over **all** nodes (every part, every graph of the split) and applies the split only to
the metric: filtering the context by split would change what a node sees. Results come back in original node /
graph order.
- Tests `tests/graph_model.rs`: output shapes for each task; `max_hops = 0, mpnn = None` equals an explicit
  `embed → BiBlock → ffn → head` composition on the degree-sorted sequence; **batch independence**: with
  `TokenSampling::Static` and one dataset, a graph's outputs are the same in a batch of its own and in a batch
  with two other graphs of different sizes (1e-4), and unchanged by the row budget and by filling the absent rows'
  inputs with large values; dropout is off in `evaluate` after a training step; with `max_hops = 0`, relabelling the input nodes of a graph with distinct
  degrees by a random permutation leaves each node's prediction unchanged; gradients reach every parameter (no
  `None`, none all-zero) for every `LocalEncoder` × `mpnn` × `TokenTail` combination; `ScanDirection::Forward`
  builds forward-only node blocks and changes the output; with `token_layers = 1` and `TokenTail::Forward` stage 1
  launches no `reverse_bands`; save / load round trip gives identical predictions; metric functions against
  hand-computed values (AUC with ties; a single-class label column is skipped in AP).
- **Expressivity** (the paper's Theorem 4.4 in miniature): graph A = one 6-cycle, graph B = two triangles (one
  6-node graph), constant node features, no PE, graph classification with mean pooling. With `max_hops = 0,
  mpnn = Gine` the two graph embeddings are equal — 1-WL cannot separate them: assert equality to 1e-5. With
  `max_hops = 2, walks = 8, local = Sgc` they differ (through `|T|` and the induced edge count in `stats`): assert
  a gap, and that 100 training steps classify the pair.
- Tests `tests/graph_learn.rs` (each under a minute in release on cpu; also run on wgpu). The thresholds are first
  targets: if one is missed, report the measured value and the curve; do not loosen it silently.
  1. **Neighbour majority (needs stage 1):** random graph, 300 nodes, mean degree 6; feature = one random bit per
     node; label = whether more than half of the node's neighbours have bit 1 (nodes with a tie are unlabelled);
     60 / 20 / 20 split. With `max_hops = 1, walks = 16, repeats = 2, local = Mean, mpnn = None`: test accuracy
     > 0.85 within 300 steps. The same spec with `max_hops = 0` stays below 0.65: assert the gap.
  2. **Long range (needs stage 2 and both directions):** path graphs of 40 nodes where every node must predict the
     colour (one of 4) written in the features of node 0 only; `max_hops = 0, node_layers = 2, mpnn = None`;
     accuracy > 0.95. With `ScanDirection::Forward` and `Degree { descending: true }` (node 0, an endpoint, is then
     scanned near the end), accuracy over the nodes scanned before node 0 stays below 0.5: assert the gap. This is
     the one-directional failure the paper's appendix describes.
  3. **Graph regression:** predict the triangle count (divided by its dataset standard deviation) of random 20-node
     graphs with `max_hops = 2, local = Sgc, mpnn = Gine`; held-out MAE below 0.7 × the MAE of predicting the
     training mean, within 300 steps.
  4. **Multi-label** smoke: three binary graph labels, one of them NaN on half the graphs; loss decreases and stays
     finite.
  5. **Node partitions:** task 1 trained with `epoch_nodes(parts = 3)` reaches the same threshold.
- Footprint `tests/graph_footprint.rs` (alone in its binary, `COUNT_LOCK`, as `tests/entity_footprint.rs`): after
  two warm-up epochs over the same batches (so every shape is tuned), one `Trainer::queue_step` →
  `read_count() == 0` and `upload_count() == 0`, in both batch modes (needs GP1); a whole `epoch_graphs` of 10
  batches → `upload_count() == 1`; pin `launch_count()` for two specs
  (`max_hops = 0` + GatedGCN; `max_hops = 2` + `Sgc` + `Gine`) **on cpu only** (wgpu counts differ, as the entity
  pins do); record the numbers in §5.

### GM8. Examples
`examples/train_graph.rs` (the neighbour-majority and triangle-count tasks; prints loss, metric, ms/step,
launches/step; `MAMBA3_GRAPH_*` env overrides for `m`, `M`, `s`, layers, local, mpnn, parts), and
`examples/profile_graph.rs` (pattern: `examples/profile_entity_model.rs`: warm-up, then `launch_tally()` by scope,
host submit time and drained step time; add `tally_scope("graph.data" | "graph.embed" | "graph.local" |
"graph.token" | "graph.node" | "graph.mpnn" | "graph.head")` in the model). Register both in `Cargo.toml` with
`required-features = ["backend"]`; list them in `examples/README.md`.

### GM9. Python module `mamba3_graph`
`bindings/python/src/graph.rs` (classes registered in the existing `#[pymodule]` of `lib.rs`, each
`#[pyclass(module = "mamba3_graph")]`), the new `bindings/python/python/mamba3_graph/{__init__.py, __init__.pyi,
py.typed}`, `python-packages = ["mamba3_graph"]` under `[tool.maturin]` in `bindings/python/pyproject.toml`, and
`bindings/python/README.md`. The graph classes are also listed in `_mamba3_rl.pyi` (they live in that extension)
but are **not** re-exported from `mamba3_rl/__init__.py`: the import name is `mamba3_graph`. API and rules P1–P7
of §2.6. Hold `f32` only in this task (`dtype` is accepted; anything
but `"f32"` raises `NotImplementedError` until GM12). Expose `mg.upload_count()` / `mg.reset_upload_count()` next to
the read counter, and `mg.build_info()`.
- Traps: `Python::detach` needs a `Send` closure — use the one `Detached` wrapper with its SAFETY comment, do not
  sprinkle `unsafe`; no `Bound` / `Py` value may be captured by a detached closure (copy scalars and borrow slices
  out first); `check_signals` needs the GIL, so re-attach for it; an error raised inside a detached section must
  be carried out as a Rust `Result` and converted after re-attaching; NumPy arrays are read only while the GIL is
  held, and through `as_slice` only when C-contiguous (P4); every class that shares the store is `unsendable`.
- Tests `bindings/python/tests/test_graph.py`: constructor and dataset validation errors name the key; `rwse` /
  `laplacian_pe` against NumPy (`numpy.linalg.matrix_power`, `numpy.linalg.eigh`, compared up to sign);
  predictions come back in original node order (shuffle the input numbering, compare per node at `max_hops = 0`);
  the neighbour-majority task learns (> 0.85); save / load round trip; `evaluate`'s accuracy equals the accuracy
  computed in NumPy from `predict`; `train_epoch` over 10 batches → `mg.read_count() == 0`, `mg.upload_count() == 1`;
  every public name of `graph.rs` appears in `mamba3_graph/__init__.py` and its `.pyi`; `import mamba3_graph`
  works from the installed wheel (not only from the source tree), `mg.GraphMamba.__module__ == "mamba3_graph"`,
  and `mg.read_count` is the same function object as `mamba3_rl.read_count`.
- Boundary tests (same file):
  - Ingest: `mg.read_count() == 0` across `GraphDataset(...)`; `float64` features, `int32` edges and a
    Fortran-ordered or sliced (non-contiguous) feature array give the same predictions as the `float32` / `int64`
    C-ordered ones; the input arrays are unchanged afterwards.
  - Outputs: `predict` over 5 batches makes exactly 1 read; `evaluate("accuracy")` makes 1.
  - GIL: while `train_epoch` runs on the main thread, a second thread that sleeps 1 ms in a loop wakes at more
    than 30% of its idle rate (wgpu only; skip on cpu, where the calling thread computes); calling a method of the
    same model from that second thread raises instead of racing.
  - Interrupt: a timer thread sends `SIGINT` 0.3 s into a long `train_epoch`; `KeyboardInterrupt` is raised
    within 1 s; the model then trains and predicts normally, and its step counter equals the number of steps whose
    losses `read_losses` returns.
  - `mg.build_info()["profile"]` is `"release"` in the wheel the tests run on; a debug build's warning is tested
    by calling the warning helper directly.
- Measure: extend `bindings/python/examples/bench_boundary.py` with a graph section — ingest GB/s and reads for a
  synthetic 200 MB dataset, `train_epoch` wall against calling-thread CPU, the second thread's share, reads per
  `predict` — and add the rows to `bench/results/python_boundary.md` next to the entity rows. Report the ingest
  rate reached; the 1 GB/s of P4 is a target to check, not a gate.
- **Parity:** `tests/graph_model.rs` writes `tests/golden/graph_tiny.json` (spec JSON, a 12-node graph, five losses
  and the final predictions) when run with `MAMBA3_WRITE_GOLDEN=1`, and otherwise checks itself against the file;
  `test_graph.py` loads the same file, builds the same model through Python and matches losses and predictions to
  1e-4. Commit the golden file (written on cpu; wgpu compares at 1e-3). Device sampling makes this exact: the
  tokens are a function of the seed.
- Example `bindings/python/examples/graph_node_classification.py`: loads a heterophilic-benchmark `.npz`
  (`node_features`, `node_labels`, `edges`, `train_masks`, `val_masks`, `test_masks`, as distributed by Platonov et
  al.) from a path argument, trains with `parts`, prints the validation and test metric per epoch.
- Build and run: `cd bindings/python && maturin develop --release && pytest tests/test_graph.py`, then the same with
  `--no-default-features --features wgpu`.

### GM10. README
A "Graph Mamba" subsection under "The seven extension points" (now eight: fix the heading and the list in
`src/lib.rs`'s crate doc): what it is, the two stages, the on-device data path (one upload, then kernels), a Rust
and a Python snippet (`import mamba3_graph as mg`), the §1.5 deviation table, the out-of-scope list. Add the paper
to "References" and `models/graph` to "Layout of the source"; in the README's "Python" section say that the
bindings now provide two import modules, `mamba3_rl` and `mamba3_graph`, from one wheel.

### GM11. Measure, attribute, then pull only the levers the profile names
The design of §2.7 is already in the code by this task. This task checks that it worked and decides what is next.

**Protocol** (manual ch. 13, ch. 16 §1.3, §2.4, §5.5–5.6, ch. 11 §6; `AGENTS.md`):
1. Correctness first: every suite green on the build being timed.
2. Warm: run one full epoch first, so every shape of the run has been compiled, tuned and allocated. Delete
   `~/.cache/mamba3` once before the session, not between runs.
3. Candidates **interleaved in one process**, at least 5 rounds, report the minimum and the spread; run-to-run noise
   on this machine is ±20%. Nothing else running.
4. One change per measurement; re-attribute after each.
5. Separate the three clocks, and never mix them in one comparison: **host submit** (time for `queue_step` to
   return), **drained step** (queue, then one read), and **device time per kernel** (CubeCL's profiler: a
   `cubecl.toml` with `[profiling] logger = { file = "...", level = "basic" }` in the working directory). Host-bound
   means host submit ≈ drained step; GPU-bound means drained ≫ host submit. A launch count divided into a step
   time does not tell them apart.
6. Baselines of matched work for the new kernels: `token_features` against a contiguous weighted sum over the same
   number of source rows and the same output; `gine_aggregate` against a contiguous neighbour sum of the same
   `nnz`. The ratio is the cost of the indirection.
7. Allocation: profile `client.empty` time on the step (manual ch. 13), and record `reserved_bytes` after 1, 10
   and 100 steps; a flat high-water mark shows no growth, not no cost.

**What to record** in `bench/results/graph_step.md`, for W-A and W-B of §2.7, cpu and Mac wgpu: launches/step,
the three clocks, uploads/epoch, **tuner misses** (add a counter next to the tuner's cache; each miss is probes,
synchronisations and a read that `read_count` does not see) and `meta_miss_count()` per epoch over three epochs,
the distinct values of each row count of §2.7 S5 and the resulting number of tuner keys, the tally by
scope (`graph.*` and the mixer's `mixer.project` / `mixer.conv` / `mixer.coef` / `mixer.scan`), and the A/B of each
decision of §2.7 that has a switch: `TokenTail`, `d_state` 8 vs 16 (with `MAMBA3_SCAN_BACKWARD=recurrent`), `Sgc` vs
`Mean`, bucketing on/off, capacities vs exact sizes (the cold-tuning cost of S5), `TokenSampling::PerStep` vs `Static`.

**Budget to check** (the block rows are measured, one block alone, `bench/results/graph_blocks.md`; the rest are
*estimates*; replace each with the whole model's measurement):

| W-B, per step | ms | tape MiB | basis |
|---|---|---|---|
| data path: G1, G2, G3 | ~3–6 | 22 (token tables) + `81,600·(F + pe)·4` B | *estimate*: 3 launches of integer and gather work |
| embedding and `φ` head on 81,600 positions | ~3–15 | ~40 | *estimate*: one product `[81,600, F + pe] → 64` with its weight adjoint (no input adjoint), a `gelu`; grows with `F` |
| stage 1: one forward block, `W = 150` | 47.5 | 197 | measured; S3's split tail should take 15–30% off both |
| stage 2: two bidirectional blocks, `W = 300`, 32 × 256 | 22.4 | 72 | measured |
| `Ψ` (two layers), FFN, heads | ~5–10 | ~20 | *estimate*: 4,800-row tensors, launch-bound |
| AdamW and the gradient norm | ~2–4 | – | *estimate*: ~60 launches after GP1 |
| **step** | **~85–105** | **~350–400** | sums of separate measurements and estimates; the pool's reservation is measured, not predicted |

W-A per part (5,666 rows, one sequence): stage 1 ~56 ms by the fit, stage 2 2 × 22.3 ms measured at 5,632 nodes;
`F = 300` puts the embedding at the top of its range and the token features at 122 MB, in chunks (S9).

**Levers, each with the measurement that triggers it.** Do not pull one whose trigger is not met.

| Lever | Trigger | Expected | Note |
|---|---|---|---|
| Fuse G2 and G3 into one kernel per token row | `graph.data` > 10% of the drained step | removes the 22–25 MB token tables and one launch | each `(token, vector)` lane would redo the walk, so only if G2 is cheap next to G3 |
| Long-row reductions: split `segment_pool` (and any row with thousands of sources) into deterministic partial sums, then fold | graph tasks with graphs above ~1,000 nodes **and** `graph.head` > 5% | occupancy (manual ch. 10 §3–4) | no atomics; keep the simple kernel for short rows |
| Recompute stage 1 instead of keeping its tape (§2.7 S8) | full-graph stage-2 context is wanted on a graph whose stage-1 tape does not fit (W-A unpartitioned: ~0.9 GiB by the fit, and allocations far above 64 MiB) | stage-1 live bytes divided by the number of chunks; costs one extra stage-1 forward, regenerated token features, and the stored encodings and their gradients | needs a design addendum: stage 1 under `no_grad` in chunks; the encodings enter stage 2 through a temporary `Param` kept out of the optimizer, so `backward()` returns their gradient; each chunk is recomputed with the same token counter and backpropagated through `sum(y_chunk · constant(saved gradient))`; the chunks' parameter gradients are **added** to stage 2's with `Grads::merge` (`src/autograd/graph.rs:224`), not averaged, and the optimizer steps once. `TrainerConfig::grad_accumulation` averages micro-batch losses and does not do this |
| Release each backward rule's saved tensors as soon as it has run (`Var::backward`, `src/autograd/var.rs:218`) | backward peak (CubeCL memory log) well above the live bytes after forward | lower peak for every model in the crate | crate-wide |
| Do not store the scan's first checkpoint (it is the zero initial state) and re-examine the checkpoint spacing | checkpoints are the largest allocation (they are at the default tail: S9) | 9.4 MiB at 4,800 rows, and a larger automatic budget | `src/tensor/ops/ssd_scan.rs:125`, `:220` |
| A custom pool configuration (`RuntimeOptions::memory_config`) with a class between 64 and 512 MiB | pool reservation above 4x the live bytes on the whole model, measured warm after an explicit cleanup, separately from cold tuning | less reservation, same live bytes | measure before and after; the default is shared by every model |
| Skip `pad_ragged` / `unpad_ragged` between node layers when `mpnn` is `None` | their launches > 5% of `graph.node` | 2 launches and 2 passes per layer | |
| Scan-backward dispatch per device: make `chunked_backward` (`src/tensor/ops/ssd_scan.rs:1471`) choose by a measured rule instead of alignment alone | anyone needs `d_state = 16` at `head_dim = 64` on this GPU | 1.7–4.5x on that backward (`bench/results/graph_scan_shapes.md`) | crate-wide; A/B the entity model too (its `d_state = 32` is not eligible for the chunked kernel, so it should not move) |
| Dense short-sequence scan (`L ≤ 32` as an `L × L` kernel, reductions inside a unit, no barriers) | `mixer.scan` > 35% of the drained step after S1–S3 | unknown; prototype in `examples/bench_ssd_scan` first, proceed only at ≥ 2x | unlikely to trigger: the scan is ~11.5 ms of an estimated 60–85 |
| Elementwise fusion inside the mixer | `mixer.project` + `mixer.coef` > 40% of the drained step | shared with every model in the crate | belongs in `KERNEL_OPTIMIZATION_PLAN.md`; report the number there |
| A paired scan benchmark: forward + backward of the mixer's real scan (rate form, skip, no boundary state), candidates interleaved | before acting on any scan number | replaces the microbenchmark's transferred estimate | `examples/bench_ssd_scan.rs` times the two halves separately today |

Stop when the remaining time is in the mixer's own scopes.

### GM12. 16-bit dtypes
Generic `E` already flows through; this task proves it and opens the Python `dtype`. One fix first:
`Var::masked_mean` (`src/autograd/ops.rs:903`) divides in f32 in the forward but casts the saved denominator to `E`
for the backward, so a count above f16's range (65,504) becomes infinity and the gradient vanishes — and stage 1
has more rows than that. Keep the denominator in f32 in the rule and cast the finished gradient once; test with a
mask of 100,000 ones in f16. Add `tests/graph_dtype.rs`:
bf16 and f16 (with `loss_scale`) track f32 within 5% over 50 steps of the neighbour-majority task where
`backend::supports_dtype` says the dtype exists, and skip with a printed reason otherwise. Python: the
enum-over-monomorphisations macro of `entity_model.rs`; `test_graph.py` gains the dtype cases of
`test_entity_dtype.py`.

### GM13. Real-data run (a measurement, not a gate)
With the example of GM9 on Roman-empire and Minesweeper (files supplied by the owner), 4 seeds each, Mac wgpu:
`GMN−` (`pe_dim = 0`, `mpnn = None`), full GMN (RWSE 16 + `Gine`), and the ablations: forward-only stage 2, no
MPNN, PPR order, no PE, plus the prices of §6: `token_tail = Bidirectional`, `d_state = 16` (with
`MAMBA3_SCAN_BACKWARD=recurrent`), two heads per direction, `Mean` vs `Sgc`, a second `Linear` in the local
encoder's head, no convolution in the token mixer (`conv_kernel = None`), `node_sequences = 4`, and the number of
parts (2, 4, 8).
Record mean ± std, epochs, s/epoch and peak memory next to the paper's numbers (§1.5) in
`bench/results/graph_accuracy.md`, with the hyperparameters used. A gap is reported, not hidden; §1.5 lists the
reasons to expect one.

### GM14. Review pass
Run every suite touched above on cpu and wgpu from a cleared tuning cache (`rm -rf ~/.cache/mamba3`), twice; run
`/code-review` on the diff; update §5.

### GM15 (optional; only if GM13 shows the linear local encoder is the accuracy bottleneck). Nonlinear `φ` on the device
`LocalEncoder::Mpnn { layers }`: G2 additionally writes, per slot, a bit row of its neighbours inside the token
(`C ≤ 64`, two u32 words). A dense-in-token GIN layer is then one kernel — lane `(token, slot, vector)` loops the
`C` slots with a bit test — and it is its own adjoint, because the induced adjacency is symmetric. Slot inputs are
gathered in feature space (constants) and embedded at slot level, so the cost is a `[Nb·L·C, F] × [F, d]` product:
affordable only when `F·C` is small (W-B at `F = 300`: ~100 GFLOP per step, not affordable). Write the design as a
short addendum, with that cost for the target dataset, before starting. CRaWl walk features (the paper's other
option) would extend G2 to keep the walks; same rule.

---

## 4. Commands

```bash
# Rust, one suite, cpu then Mac GPU
cargo test --release --test graph_kernels > /tmp/t.log 2>&1; echo "exit=$?"
cargo test --release --no-default-features --features wgpu --test graph_kernels > /tmp/t.log 2>&1; echo "exit=$?"
# all graph suites
for t in graph_data graph_kernels graph_encoding graph_ragged graph_layers graph_batch graph_model graph_learn graph_footprint; do
  cargo test --release --test $t > /tmp/$t.log 2>&1; echo "$t exit=$?"; done
# suites that must stay green because GM4 touches the mixer
cargo test --release --test ssm --test model --test entity_blocks --test entity_model --test entity_footprint
# Python
cd bindings/python && maturin develop --release && pytest tests/test_graph.py -q
python examples/bench_boundary.py                    # the Python boundary: ingest, GIL, reads (§2.6)
# profile
cargo run --release --no-default-features --features wgpu --example profile_graph
# the scan measurement behind §2.7
MAMBA3_SCAN_SHAPE=4800,17,1,64,8 MAMBA3_SCAN_CHUNKED=1 ./target/release/examples/bench_ssd_scan
ROWS=4800 SEQ=17 HEADS=1 STATE=8 BIDIR=0 ./target/release/examples/bench_graph_blocks   # one block: ms, tape, reserved
ROWS=32 SEQ=256 HEADS=1 STATE=8 BIDIR=1 ./target/release/examples/bench_graph_blocks
```

---

## 5. Status

| Task | Commit | cpu | wgpu (Mac) | Note |
|---|---|---|---|---|
| GS0 | – | – | blocks, scan shapes and the Python boundary measured 2026-10-01 | `bench/results/graph_blocks.md` (`examples/bench_graph_blocks`, added with this plan), `bench/results/graph_scan_shapes.md`, `bench/results/python_boundary.md` (`bindings/python/examples/bench_boundary.py`, existing entity bindings); re-run if the mixer or the scan kernels change |
| GM0 | – | done 2026-10-01 | owed | `upload_count`, `meta_miss_count`, `peak_alloc_bytes`, tuner-miss and matmul-shape logs; the composed scan's causal mask is now a cached constant (it was uploaded on every call) |
| GM1 | – | `graph_data` 18 | owed | host-only |
| GM2 | – | `graph_kernels` 22 | owed | **cubecl-cpu trap found and documented** (top of `src/tensor/ops/graph.rs`): a kernel whose loop-carried variable is initialised by copying a scalar argument is dropped silently; the batch-slot search is a fixed-step bit descent because of it |
| GM3 | – | `graph_encoding` 6 | owed | host-only; `rwse_csr` / `laplacian_pe_csr` added for the bindings |
| GM4 | – | `graph_ragged` 7; `ssm`, `model`, `entity_blocks`, `entity_model`, `entity_footprint` unchanged | owed | `apply_last` on cpu: 0.66–0.68x the time of `apply` + slice at 4,800 × 17, tape −24% (`bench/results/graph_blocks.md`) |
| GM5 | – | `graph_layers` 12 | owed | `pe_sign_flip` added to the spec; `node_ssm.chunk_size` defaults to 32 and the token chunk is balanced (`token_chunk()`): a 17-token sequence was being scanned as 64 positions |
| GM6 | – | `graph_batch` 9 | owed | size buckets: padding 1.26x against 1.90x unbucketed (2,000 graphs); `PreparedDataset` splits host preparation from the upload |
| GP1 | – | `train_uploads` 3, `adamw_multi` 3, `train`, `rl_update_footprint`, `entity_footprint` | owed | uploads/step before → after: **25 → 5** on the entity model's step (the 5 are its own host tables); pure optimizer step 0; the LM step keeps 3 (its embedding's `scatter_add_rows`), so "zero uploads" is asserted for the graph and optimizer steps, not the LM one |
| GM7 | – | `graph_model` 15, `graph_learn` 5, `graph_footprint` 4 | owed | launch pins (cpu): 589 (node tokens + GatedGCN, whole graphs; 587 before the review's masked edge adjoint), 704 (walk tokens + Sgc + GINE, whole graphs), 529 (node tokens + GINE, parts), 728 (walk tokens + GINE, parts), 736 (four node sequences). Steady-state step: 0 reads, 0 uploads, both modes; 10-batch epoch: 1 upload, 1 read. Learning (after the review's change to the walk counters): neighbour majority 0.892 with tokens vs 0.541 without (37 test nodes; 0.946 on three parts); long range 1.000 bidirectional vs 0.250 forward; triangles 0.26 of the constant predictor's MAE |
| GM8 | – | examples run | owed | `train_graph`, `profile_graph`, `bench_graph_kernels`; listed in `examples/README.md` |
| GM9 | – | `test_graph.py` 21 | owed | cpu wheel; both modules import from a clean wheel install; `tools/build_wheel.sh --smoke` imports both. Deviations: `dtype="bf16"` is open already (GM12 done in the same pass), `predict(split=...)` filters by the caller's masks in the binding (Rust `predict` has no split), the GIL test is not skipped on cpu (it passes there: 94%) |
| GM10 | – | – | – | README "Graph Mamba" (eight extension points), crate doc, layout, references, `bindings/python/README.md`; the README's Rust snippet was compiled and run |
| GM11 | – | `bench/results/graph_step.md` | owed | cpu only: W-B 748 launches, 673–728 ms; W-A 757 launches, 908–980 ms; estimate within 3–6% of measured live bytes; **no lever's trigger met on cpu**. The plan's budget is a GPU budget and is not checked by these numbers |
| GM12 | – | `graph_dtype` 3 + 1 ignored | owed | **bf16 tracks f32** (ten-step loss means within 2.3% over 50 steps). **f16 does not train in this tree**: the mixer's backward pass returns NaN gradients in f16 at `477789a` already, with or without the graph model; it needs `BF16_ACTIVATIONS_PLAN.md`, which is not in this tree. The f16 50-step test is `#[ignore]`d and the bindings refuse `dtype="f16"`. `masked_mean` keeps its denominator in f32 (tested with 100,000 ones) |
| GM13 | – | not run | not run | needs the benchmark files from the owner; `bindings/python/examples/graph_node_classification.py` is ready for them (smoke-run on a synthetic file in the benchmark's layout) |
| GM14 | – | every Rust suite twice, the Python suite twice | owed | four read-only reviews of the diff; fixes and their regressions in `tests/graph_review.rs` 14, `tests/train_uploads.rs` 5 and `test_graph.py`. The reviews' follow-ups are fixed too: the per-device table caches drop the tables of devices that no longer exist and the optimizer's tables have their own budget (`src/backend.rs`); the launchers check the reverse index, the graph's symmetry before an adjoint, the dataset a batch belongs to, the target tables and the epoch offsets; `gated_edge`'s adjoint masks absent edge rows; masks without targets are an error. `~/.cache/mamba3` was **not** cleared: it holds this machine's GPU tuning tables, and the cpu backend does not read it. Red before and after this work, identically, on this machine: `train_ema_parity` (Rust) and four tests of the Python suite (EMA parity, two entity-generality tests, entity speed) |

---

## 6. Decisions the owner may want to overrule

Each has a default above; none blocks starting.

1. **Mamba-3 and RmsNorm instead of Mamba-1 and LayerNorm** (§1.5). A Mamba-1 mixer for paper parity is a separate,
   large piece of work and is not planned.
2. **The local encoder is linear** (`Sgc` or `Mean`, with token statistics) so that tokens can be built on the
   device from constants (§2.2). The paper's nonlinear MPNN encoder is not built unless GM13 shows the need (GM15),
   and would be expensive for wide features.
3. **Tokens are resampled every step.** The paper samples before training and shuffles; `token_sampling = "static"`
   reproduces that.
4. **One large graph is trained on jittered stratified node partitions** (§2.2) rather than full-batch only
   (`parts = 1` is full batch).
5. **Message passing uses the symmetrised graph only**, and **GatedGCN only with whole-graph batches** in v1.
6. **LapPE only up to 2048 nodes per graph** (dense eigensolver, no new dependency); larger graphs use RWSE or none.
7. **Ascending degree order** by default; the paper says "by degree" without a direction.
8. **`d_state = 8` and one head per direction in both stages** (§2.7 S1–S2). Time and tape are proportional to the
   projection width: two heads per direction cost 1.8x in stage 2 (measured), and the first draft's stage-1 block
   cost 5x this one. Capacity is the trade; GM13 prices it.
9. **The last stage-1 layer is forward-only** (§2.7 S3), so with the default `token_layers = 1` stage 1 has no
   backward scan. `token_tail = Bidirectional` is the paper-literal alternative at about twice the cost.
10. **Python is the module `mamba3_graph`** (owner's decision, 2026-10-01): a second import module of the existing
    bindings in `bindings/python/`, in the same wheel and the same extension library as `mamba3_rl` — not a
    separate package or crate. The distribution is still named `mamba3-rl`.
11. **A dataset must fit on the device** (2 GiB guard). Sharded residency is not built.
12. **Batches are filled to a row budget**, not to a number of graphs (§2.2): equal work per step and one stage-1
    shape per run. The number of graphs per step then varies with their size. **The budget defaults to what keeps
    every allocation under 64 MiB** (§2.7 S9), because that is where this device's memory pool changes page class;
    time per row was flat from 2,400 to 9,600 rows in the block measurement.
13. **GP1 changes the optimizer** (it stops uploading its small tables every step). It is outside the graph model
    but the zero-upload step depends on it.
14. **The local encoder's head is a single nonlinearity** (§2.7 S11), not an MLP: a capacity choice, priced in GM13.
15. **The Python bindings release the GIL with one `unsafe` wrapper** (§2.6 P2), because the model types are not
    `Send`. The alternative is to keep the GIL for the whole of `train_epoch`, as the existing bindings do.

---

## 7. References

- A. Behrouz, F. Hashemi. *Graph Mamba: Towards Learning on Graphs with State Space Models.* KDD 2024.
  arXiv:2402.08678v2 (§4 and Algorithms 1–2 are the specification used here). The repository the paper names,
  `github.com/GraphMamba/GMN`, returned 404 on 2026-10-01, so no reference code was consulted.
- L. Rampášek et al. *Recipe for a General, Powerful, Scalable Graph Transformer* (GPS): the layer GMN reduces to at
  `m = 0`; source of the RWSE / LapPE conventions.
- X. Bresson, T. Laurent. *Residual Gated Graph ConvNets* (GatedGCN). W. Hu et al. *Strategies for Pre-training GNNs*
  (GINE). J. Tönshoff et al. *Walking Out of the Weisfeiler Leman Hierarchy* (CRaWl; GM15).
- O. Platonov et al. *A critical look at the evaluation of GNNs under heterophily* (GM13 datasets).
- The CubeCL manual, `/Users/ods/Documents/cubecl_manual/manual/Cubecl`: ch. 5 (lazy execution), 7 (coalescing), 8
  (atomics), 10 (occupancy), 11 (launch overhead and transfers), 13 (preallocation), 16 (profiling),
  *Hardware-Adaptive Launch Geometry*.
- In this repo: `ENTITY_MODEL_PLAN.md` (plan format, `BiBlock`, dataset pattern), `BF16_ACTIVATIONS_PLAN.md` (f32
  accumulation, dtype in Python), `bench/results/graph_blocks.md`, `bench/results/graph_scan_shapes.md`,
  `bench/results/python_boundary.md`, `docs/test_guidline.md`, `AGENTS.md`.
