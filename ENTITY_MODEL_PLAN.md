# Entity model plan: a general set-to-plan model on Mamba-3 (supersedes TASK_PLANNER_PLAN.md)

This is an execution document. It is written so that it can be worked start to finish **without reading the session that
produced it**, by an implementer who follows instructions literally. Every task gives the goal, the files, the exact API
(with code skeletons), the traps, the test, the command that proves it, and a "done when" line. Work the tasks **in the
order given**. Each task lands on its own and leaves the suite green. Do not skip a measurement gate.

Written against commit `761365c` ("Task planner T1-T6a and K0 harness"). Line anchors (`file:line`) are from that commit;
if a line moved, search for the quoted symbol.

---

## 0. Why this plan exists, and what it replaces

`TASK_PLANNER_PLAN.md` built a model for **one** problem: Kaggriculture's crew planner. Its public API speaks that problem's
language — `TaskPlannerConfig { grid, c_tile, c_unit, max_units, n_ops, n_crops, … }` (`src/models/planner.rs:57`),
`HostBatch { tiles, upos, tgt, op, crop, opset, eta }` (`planner.rs:487`), a fixed loss with fixed step weights
(`STEP_W`, `planner.rs:511`) and five hard-wired heads. Nothing in it can be reused for another problem without editing the
library.

The underlying pattern is general, and common:

> A **context** of entities (tiles, customers, cards, jobs, enemy units — one or more named sets, some laid out on a grid),
> plus global features. A set of **queries** (agents, vehicles, workers, hands of cards), each optionally anchored to one
> context entity. For each query, a **plan of K steps**; at each step one or more **heads** predict: which context entity
> to act on (a pointer, plus a few extra actions such as "none"), and attributes of that action (a class, a set of labels,
> a number), possibly depending on the chosen entity. Steps may depend on the previous steps' choices.

Examples this plan must serve without code changes (only a different spec):

| Problem | Context | Queries | Heads |
|---|---|---|---|
| Kaggriculture crew planner (the reference) | 100 tiles on a 10×10 grid | 20 units, anchored at their tile, K = 3 | pointer(tiles + NONE), op (13), op set (13 labels), crop (5), eta (1 number, step 1) |
| Vehicle dispatch / routing | customers, depots | vehicles, anchored at their location, K = next stops | pointer(customers + RETURN), load (number) |
| Multi-agent target selection (RTS, MOBA) | enemy units, map cells | own units, K = 1 | pointer(enemies + HOLD), ability (class) |
| Job-shop / task scheduling | jobs | machines, K = queue length | pointer(jobs + IDLE), duration (number) |
| Card or slate selection | cards / items | K = 1 query (the hand / the slate), K = slate size | pointer(cards + PASS) |
| Entity classification | objects | 0 queries (pure encoder) — heads on context | categorical per entity (see "context heads", optional, §8) |

This plan therefore replaces `TaskPlanner` with a domain-free **`EntityModel`**. The Kaggriculture planner becomes a
**spec** (a configuration), written in an example and in the Kaggriculture repo — never in the library.

### 0.1 What already exists (commit `761365c`) and what happens to it

| Existing | Fate |
|---|---|
| `BiBlock`, `transpose_grid` (`planner.rs:227,271`) | **reused**, moved to `src/models/entity/blocks.rs` (G2) |
| Joint-sequence decoder, NONE key, one-matmul pointer, fused aux Linear (`planner.rs:335-406`) | **generalised** (G3, G4) |
| Host batch + on-host keep masks + loss with static loss scale (`planner.rs:487-830`) | **generalised** into `EntityBatch` / `EntityLoss` (G4) |
| Tests `tests/planner.rs`, `tests/planner_f16.rs`, `tests/planner_footprint.rs`, `tests/planner_kernels.rs` | **ported** to the generic model with the Kaggriculture spec (G6), then the old files are deleted |
| K0 switch `set_fused_planner` / `MAMBA3_FUSED_PLANNER` | **renamed** `set_fused_entity_model` / `MAMBA3_FUSED_ENTITY_MODEL` (K0) |
| Python `TaskPlannerConfig`, `TaskPlanner` (`bindings/python/src/planner.rs`) | **replaced** by the generic bindings (P1); removed in G6 after the port |
| `examples/profile_planner.rs`, `bench_planner.rs` | **ported** to the generic model (K0) |
| `TASK_PLANNER_PLAN.md` | kept for history, with a "superseded by ENTITY_MODEL_PLAN.md" banner (G0) |

The old classes were committed on the same day and have no users outside this repo and the Kaggriculture experiment (which
has not called them yet), so they are **removed**, not deprecated. If the owner objects, keep them one release as thin
wrappers that build the Kaggriculture spec (G6 says how) — ask before deleting.

### 0.2 Rules (unchanged from `TASK_PLANNER_PLAN.md` §0.2 and `AGENTS.md`)

- Check **exit codes**: `cmd > /tmp/t.log 2>&1; echo "exit=$?"`. Use `--no-fail-fast` for full runs.
- Never build `wgpu`/`vulkan` while CPU tests run (shared `target/`); never benchmark while anything else runs.
- Test on **both** `cpu` and `vulkan`. On this machine use **`vulkan`**, not `hip` (HIP crashes at the first compute).
- **Python and Rust APIs in parity**; every Python entry point has a `.pyi` stub.
- Performance claims come from `launch_count()` / `read_count()` / `launch_tally()` and interleaved A/B runs.
- **bf16 is not supported** on the target machine: precision options are `f32` and `f16` only (f16 with a static loss scale,
  `TASK_PLANNER_PLAN.md` §2.5, which stays the reference for the numerics).
- Watch disk space (`df -h .`): a release build with examples is ~16 GB of `target/`.

---

## 1. The API

### 1.1 Vocabulary

| Term | Meaning |
|---|---|
| **sample** | one training example (one game turn, one dispatch decision, …); batch axis `B` |
| **context set** | a named set of up to `count` entities with `features` floats each and a presence flag; optional grid layout |
| **globals** | `G` floats per sample that belong to no entity |
| **query set** | up to `count` queries (agents) with `features` floats each, a presence flag, and an optional **anchor**: the index of one entity of a named context set |
| **plan length `K`** | number of steps predicted per query |
| **head** | one prediction per (query, step): `Pointer`, `Categorical`, `MultiLabel` or `Regression` |
| **plan head** | the one pointer head whose choice at step `j` conditions step `j+1` (autoregressive decoding) |
| **extra actions** | `E` learned choices appended to a pointer's entities (e.g. NONE, PASS, RETURN) |
| **IGNORE** | the label sentinel: `-1` in every integer label array (Python), `u32::MAX` inside the crate; `NaN` in float labels |

### 1.2 Rust types (new module `src/models/entity/`, re-exported from `mamba3::models`)

```rust
// src/models/entity/spec.rs — all #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]

pub enum SetLayout {
    /// Entities in the order given.
    Sequence,
    /// Row-major `height × width` grid (count must equal height*width). With `alternate_axes`, every second
    /// encoder layer scans the grid column-major, so vertical neighbours are adjacent in the scan.
    Grid { height: usize, width: usize, alternate_axes: bool },
}

pub struct ContextSetSpec {
    pub name: String,
    pub count: usize,
    pub features: usize,
    pub layout: SetLayout,          // default Sequence
    pub position_embedding: bool,   // default true: learned [count, d] per slot
}

pub struct QuerySetSpec {
    pub name: String,
    pub count: usize,
    pub features: usize,
    /// Name of a context set; each query then carries an index into it (IGNORE = no anchor).
    pub anchor: Option<String>,
    /// Plan length K (>= 1).
    pub steps: usize,
    /// Name of the pointer head whose previous choices condition later steps; None = steps independent.
    pub autoregressive_on: Option<String>,
    /// How many previous steps condition a step (1 = only j-1). Ignored when autoregressive_on is None.
    pub lags: usize,                // default: steps - 1
}

pub enum HeadKind {
    /// Scores the entities of `set` plus `extra_actions` learned extras. Absent entities are masked.
    Pointer { set: String, extra_actions: usize },
    /// `classes` logits.
    Categorical { classes: usize },
    /// `labels` independent Bernoulli logits (BCE).
    MultiLabel { labels: usize },
    /// `outputs` real numbers (MSE on the given targets).
    Regression { outputs: usize },
}

pub enum StepSelection { All, First }   // First = only step 0 has a label and an output (e.g. eta)

pub struct HeadSpec {
    pub name: String,
    pub kind: HeadKind,
    /// Name of a pointer head: this head also sees the token of the entity chosen there
    /// (the true entity in training, the decoded one at inference). None = sees only the query state.
    pub condition_on: Option<String>,
    pub steps: StepSelection,       // default All
    pub loss_weight: f32,           // default 1.0
    /// Per-step weights, length = steps (default all 1.0). Kaggriculture uses [1.0, 0.5, 0.5] on its pointer.
    pub step_weights: Option<Vec<f32>>,
}

pub enum DecoderMode {
    /// All queries and steps in one bidirectional scan after the context. Only valid without autoregressive_on.
    Joint,
    /// Step-major sequence [context ; step 0 queries ; step 1 queries ; …] with FORWARD-only scans, so a step-j query
    /// sees the context and steps <= j only. With `crew_symmetric`, a second forward scan runs with the queries
    /// reversed inside each step block and the two are summed, so every query also sees all queries of its own step.
    StepCausal { crew_symmetric: bool },
}

pub struct EntityModelSpec {
    pub globals: usize,                  // may be 0
    pub context: Vec<ContextSetSpec>,    // >= 1
    pub queries: Option<QuerySetSpec>,   // None only with context heads (§8, not in this plan)
    pub heads: Vec<HeadSpec>,            // >= 1
    pub d_model: usize,                  // default 128
    pub context_layers: usize,           // default 3
    pub decoder_layers: usize,           // default 3
    pub decoder: DecoderMode,            // default StepCausal { crew_symmetric: true } if autoregressive_on, else Joint
    pub ssm: SsmConfig,                  // default as TASK_PLANNER_PLAN §2.2: n_heads 4, head_dim 64, d_state 32, n_groups 1
    pub chunk_size: Option<usize>,       // None = auto (§2.4)
    pub norm_eps: f32,                   // 1e-5
    pub seed: u64,
}
```

`EntityModelSpec::validate()` must reject, with a message naming the field:
- duplicate set or head names; a `Pointer.set`, `anchor` or `condition_on` / `autoregressive_on` naming nothing, or naming a
  head that is not a pointer;
- `Grid` with `height*width != count`;
- `autoregressive_on` with `DecoderMode::Joint` (it would leak the future, §2.3);
- `steps == 0`, `step_weights.len() != steps`, `StepSelection::First` on the plan head;
- zero sizes; plus `ssm.validate()` with `d_model` filled in (as `VisionMamba3Config::validate`, `vision.rs:160-162`).

### 1.3 Data contract

Everything is named, so one dictionary describes a batch or a whole dataset. Shapes use `S` for samples (a dataset) or `B`
(a batch); the model code sees `B`.

| Key | dtype (Python) | Shape | Required | Meaning |
|---|---|---|---|---|
| `globals` | float | `[S, G]` | if `G > 0` | |
| `<ctx>` | float | `[S, count, features]` | yes | one per context set |
| `<ctx>.presence` | bool / 0-1 | `[S, count]` | no (default all 1) | absent entities are zeroed and never pointed at |
| `<q>` | float | `[S, count, features]` | yes | the query set |
| `<q>.presence` | bool / 0-1 | `[S, count]` | no (default all 1) | absent queries produce no loss and no output |
| `<q>.anchor` | int | `[S, count]` | if `anchor` set | index into the anchor set, `-1` = none |
| `legal.<pointer head>` | bool / 0-1 | `[S, count_ctx + E]` or `[S, count_q, count_ctx + E]` | no | extra action masking; combined with presence |
| `label.<pointer or categorical head>` | int | `[S, count_q, K]` (or `[S, count_q]` for `First`) | for training | `-1` = IGNORE; pointer ids `0..N-1` entities, `N..N+E-1` extras |
| `label.<multilabel head>` | float 0/1, NaN | `[S, count_q, K, labels]` | for training | a NaN anywhere in a row ignores the row |
| `label.<regression head>` | float, NaN | `[S, count_q, K, outputs]` | for training | NaN = IGNORE |

The **padding convention is fixed per spec**: queries and entities are always padded to `count`. The Mamba scans are not
invariant to padding tokens (`TASK_PLANNER_PLAN.md` T3 trap), so trimming is never allowed.

### 1.4 Python API (bindings `bindings/python/src/entity_model.rs`, stubs in `_mamba3_rl.pyi`)

```python
import mamba3_rl as m3

spec = m3.EntityModelSpec(
    globals=114,
    context=[m3.ContextSet("tiles", count=100, features=48, layout=m3.Grid(10, 10, alternate_axes=True))],
    queries=m3.QuerySet("units", count=20, features=36, anchor="tiles", steps=3, autoregressive_on="target"),
    heads=[
        m3.Head.pointer("target", set="tiles", extra_actions=1, step_weights=[1.0, 0.5, 0.5]),
        m3.Head.categorical("op", classes=13, condition_on="target"),
        m3.Head.multilabel("opset", labels=13, condition_on="target", loss_weight=0.3),
        m3.Head.categorical("crop", classes=5, condition_on="target", loss_weight=0.3),
        m3.Head.regression("eta", outputs=1, steps="first", loss_weight=0.1),
    ],
    d_model=128, context_layers=3, decoder_layers=3, seed=0)        # decoder defaults to StepCausal(crew_symmetric=True)

spec.to_json(); m3.EntityModelSpec.from_json(s)                        # round trip

train = m3.EntityDataset(spec, arrays)          # dict of numpy arrays keyed as in §1.3; validated, uploaded ONCE
model = m3.EntityModel(spec, learning_rate=3e-4, weight_decay=0.05, max_grad_norm=1.0,
                       lr_schedule=m3.LrSchedule.cosine(...), matmul_precision="f16", loss_scale=1024.0)
model.queue_train_step(train, ids)             # ids: int array [B] of sample indices; nothing is read back
logs = model.read_losses()                     # [{"loss": f, "grad_norm": f, "heads": {"target": f, "op": f, ...}}]
metrics = model.evaluate(dev, ids, batch=512)  # {"target": {"top1": f, "top3": f}, "op": {"top1": f}, "opset": {"bce": f},
                                               #  "eta": {"mse": f}}  computed on the device, one read per call
out = model.predict(dev, ids)                  # greedy decoding; or model.predict(inputs=dict_of_arrays)
# out["target"]: {"logits": [B,20,3,101] float32, "choice": [B,20,3] int}; out["op"]: {"logits": [B,20,3,13]}; ...
out = model.predict(dev, ids, decode="teacher_forced")        # conditions on the labels (for diagnostics)
out = model.predict(inputs=obs_arrays, chooser=my_chooser)    # my_chooser(step, logits [B,M,N+E]) -> ints [B,M]
model.save(path); m3.EntityModel.load(path)
m3.set_fused_entity_model(True)                               # K0 switch
```

`chooser` lets a caller impose constraints at inference (for example Kaggriculture's "no two units on one tile" crew
assignment) and have the constrained choice feed the next step. It costs one device read per step, so it is for inference
only; training never calls it.

### 1.5 The Kaggriculture spec lives outside the library

- `examples/entity_kaggriculture.rs`: builds the spec above in Rust, trains a few steps on random data, prints launches. It is
  the Rust-side smoke test and the profile subject (K6).
- The Kaggriculture repo builds the same spec in Python (P3). Nothing named tile, unit, op, crop or eta appears in `src/`.

---

## 2. The design

### 2.1 Encoder (context)

```
for each context set s:   x_s = MLP_s(features_s) * presence_s          [B, N_s, d]   (Linear → GELU → Linear)
                          x_s += pos_s[slot] (if position_embedding) + type_s (learned [d], one per set)
g = MLP_g(globals)                                                          [B, d]  (skipped if G = 0)
c = concat_s(x_s) + g broadcast over all context tokens                     [B, N_ctx, d],  N_ctx = Σ N_s
for layer l in 0..context_layers:   c = unpermute_l( BiBlock_l( permute_l(c) ) )
```

`permute_l` is a **precomputed index permutation of the whole context sequence**: identity, except that on odd layers the
token range of every `Grid { alternate_axes: true }` set is transposed (row-major ↔ column-major). This generalises
`transpose_grid` (`planner.rs:271`) to several sets and to any grid shape. The permutation and its inverse are computed once
in `init` as `Vec<u32>`; the composed path applies them with `gather`-based indexing, the fused path with K4.

`BiBlock` (`planner.rs:227`) is reused unchanged: `x + mixer(norm(x))` with a **bidirectional** fused mixer (context has no
causal order).

### 2.2 Queries

```
u    = MLP_q(query features) * presence_q                                  [B, M, d]
u   += anchor token: c[b, anchor[b,m]] (0 where anchor = IGNORE)
u   += g                                                                   (globals, broadcast)
q_j  = u + step_emb[j] + Σ_{lag=1..lags, j-lag>=0} P_lag( tok(choice[b,m,j-lag]) )    for j in 0..K
```

`tok(i)` is the final context token of entity `i` of the plan head's set, or the learned extra-action embedding
`extra_emb[i - N]` for an extra action, or a learned `none_prev` vector when there is no previous step. `P_lag` is a
`Linear(d, d)` per lag. In training `choice` = the labels (teacher forcing, IGNORE → `none_prev`); at inference it is the
decoded choice. Without `autoregressive_on`, the `P_lag` terms are absent.

### 2.3 Decoder and the leakage rule

- `DecoderMode::Joint` (non-autoregressive): sequence `[c ; q_0 … q_{K-1}]` (query-major, as today), `decoder_layers`
  **bidirectional** BiBlocks — exactly the committed `TaskPlanner` decoder.
- `DecoderMode::StepCausal` (autoregressive): sequence `[c ; Q_0 ; Q_1 ; … ; Q_{K-1}]` where `Q_j` holds all `M` queries
  of step `j` (**step-major**), `decoder_layers` **forward-only** blocks (`Mamba3MixerConfig::with_bidirectional(false)`,
  no head doubling). A forward scan lets a token see only what precedes it, so step-`j` queries see the context and steps
  `≤ j`, never a later step's teacher-forced choice.
  - `crew_symmetric: true` adds, per layer, a second forward-only mixer applied to the same sequence with the query tokens
    **reversed inside each step block** (another precomputed permutation, K4), un-permuted and added. Step order is kept,
    so the rule still holds, and each query sees every query of its own step through one of the two scans.
  - **Never** use a bidirectional mixer over the query tokens in this mode: the backward direction carries step `j+1`'s
    teacher-forced choice into step `j`. (This exact bug was found in the PyTorch reference by the leakage test below: a
    reversed causal mask let step-1 queries read step-2's true input.)
- After the decoder: `RmsNorm`; split into context tokens `c'` and query states `h [B, M, K, d]`.

**The leakage test (mandatory, G3):** with teacher forcing, change the step-`j` labels of every query to random valid ids;
the outputs (logits of every head) of all steps `≤ j` must be **exactly** unchanged, and those of step `j+1` must change.
And: greedy `generate` must equal a teacher-forced forward fed with its own choices, exactly.

### 2.4 Heads, loss, decoding

- **Pointer**: keys `[c'_set ; extra_emb]` → `[B, N+E, d]`; logits `= (h W_q) · (keys W_k)ᵀ / √d` in **one** `matmul_nt`
  (as `planner.rs:396-400`); then absent entities and `legal` = 0 get `-1e4` (a constant mask added, no host read).
- **Conditioned heads** (`condition_on = p`): input `concat(h, tok_p(choice))` where `choice` is the label (training) or
  the decoded choice; **all conditioned heads share one Linear** whose output is split by head (the committed fused aux
  Linear, generalised). Unconditioned heads share a second Linear on `h`.
- **Loss**: per head, the mean over kept (query, step) rows of CE (pointer, categorical), BCE averaged over labels
  (multilabel) or squared error averaged over outputs (regression), rows weighted by `step_weights[j]`; total
  `= loss_scale × Σ_h loss_weight_h × L_h`. Kept = query present and label not IGNORE / NaN.
- **Chunk size (auto)**: the encoder scans `N_ctx` tokens, the decoder `N_ctx + M·K`. Pick, per mixer, the largest
  `c ∈ {64, 50, 48, 40, 32, 25, 20, 16}` that divides the length; if none, 32. `ssd_chunked` pads otherwise
  (`scan.rs:168-171`). Kaggriculture: 100 → 50, 160 → 40 (Joint) / 160 → 40 (StepCausal, 100 + 60).
- **Greedy decoding** (`generate`): encoder once; for `j in 0..K`: run the decoder with `choice[<j]` filled, take step `j`'s
  pointer logits, apply `chooser` (default argmax on the device, `reduce::argmax`, `reduce.rs:378`), store. `K` decoder
  passes. (An incremental version with `Mamba3Mixer::apply_with_state`, `mamba3.rs:567`, that continues the scan instead of
  re-running it, is a follow-up; not in this plan.)

### 2.5 Speed rules (carried over, generalised)

| # | Rule |
|---|---|
| S1 | No device → host read inside `loss()` or a training step. Keep-masks come from the dataset on the device (K1) or, in the composed oracle, from the host batch builder. Never `cross_entropy_with(.., ignore_index)` (`loss.rs:84` reads back). |
| S2 | Each dataset is uploaded once (`EntityDataset`); a step sends only `[B]` sample ids. |
| S3 | `Trainer::queue_step`; losses read every `log_every` steps (`read_losses`), never per step. |
| S4 | One `matmul_nt` per pointer head (extras appended as keys); one shared Linear for conditioned heads, one for unconditioned. |
| S5 | Auto chunk size dividing the sequence (§2.4). |
| S6 | f16 matmul operands with a static loss scale (`TASK_PLANNER_PLAN.md` §2.5); bf16 refused. Gated by an A/B (K6). |
| S7 | `evaluate` / `predict` under `no_grad`, large batches, one read per call (plus one per step with a Python `chooser`). |
| S8 | On-device kernels for everything that is not a mixer, a Linear or a matmul (K1–K5), gather-style adjoints, no atomics. |

---

## 3. Repo map

All anchors from `TASK_PLANNER_PLAN.md` §3 still hold (mixer, Linear, Param, Tensor, IdTensor, autograd ops, Trainer,
AdamW, Checkpoint, bindings helpers, kernel templates). Additional anchors for this plan:

| Path | What you use from it |
|---|---|
| `src/models/planner.rs:227-278` | `BiBlock`, `transpose_grid` (move to `entity/blocks.rs`) |
| `src/models/planner.rs:335-460` | the committed forward, heads, predict — the template for G3–G5 |
| `src/models/planner.rs:487-830` | host batch, keep masks, loss with loss scale — the template for G4 |
| `src/models/planner.rs:31-55` | the K0 switch to rename |
| `src/rl/spec.rs:30-62` | `EntitySet`, `ObsSpec`: the crate's existing entity vocabulary. Provide `impl From<&EntitySet> for ContextSetSpec` so RL users can reuse their specs |
| `src/rl/heads.rs:81` | `PointerHeadConfig` (RL): naming precedent (`set`, extras) — keep the same words |
| `src/models/mamba3.rs:155` | `with_bidirectional(false)` for forward-only decoder blocks |
| `src/tensor/ops/reduce.rs:378` | device `argmax` → `IdTensor` |
| `bindings/python/src/planner.rs` | the committed bindings: template for P1 (array reading, loss-scale handling, read_losses) |
| `tests/planner.rs` | the committed tests: port them in G6 |

---

## 4. Tasks: the generic model (composed path)

### G0. Banner, branch, baseline

1. The banner at the top of `TASK_PLANNER_PLAN.md` ("Superseded by ENTITY_MODEL_PLAN.md …") was added together with this
   plan. Commit both plan files together, alone.
2. Baseline suite green (`cargo test --release --no-default-features --features cpu --no-fail-fast`). If not, stop.
3. `git status`: commit nothing that is not named in this plan.

**Done when:** the two plan files are committed alone; suite green.

### G1. Spec types

**Files:** `src/models/entity/mod.rs` (`pub mod spec; pub mod blocks; pub mod model; pub mod batch; pub mod loss;`),
`src/models/entity/spec.rs`, `src/models/mod.rs` (`pub mod entity; pub use entity::{…};`).

**Do:** the types of §1.2 with `Default` impls where §1.2 names a default, builder-style helpers
(`HeadSpec::pointer(name, set, extra)`, `::categorical`, `::multilabel`, `::regression`, `.condition_on(..)`,
`.loss_weight(..)`, `.step_weights(..)`, `.first_step_only()`), `validate()` (§1.2 list), and derived helpers:
`n_ctx()`, `set_offset(name)`, `query_tokens() = count*steps`, `head(name)`, `plan_head()`, `chunk_for(len)` (§2.4).
`impl From<&crate::rl::spec::EntitySet> for ContextSetSpec`.

**Tests** (`tests/entity_spec.rs`): JSON round trip of the Kaggriculture spec; each validation error (one test per bullet
of §1.2, asserting the message names the field); `chunk_for(100) == 50`, `chunk_for(160) == 40`, `chunk_for(7) == 32`.

**Done when:** tests pass on `cpu`.

### G2. Encoder

**Files:** `src/models/entity/blocks.rs` (move `BiBlock` and `transpose_grid` here; add `ForwardBlock` = same as `BiBlock`
with `with_bidirectional(false)` and no head doubling; add `Permutation { fwd: Vec<u32>, inv: Vec<u32> }` with
`identity(n)`, `grid_transpose(offset, h, w, n)`, `reverse_blocks(offset, block, blocks, n)`, `apply(&Var) -> Var`
(composed: `gather` of rows by index via `IdTensor` + `embedding`-style gather on the token axis, reshaped
`[B*n, d]`), `inverse`), `src/models/entity/model.rs` (`EntityModel` struct, `encode`).

`planner.rs` keeps compiling by importing `BiBlock` / `transpose_grid` from the new place until G6 deletes it.

**Tests:** `Permutation::grid_transpose` equals `transpose_grid` on a `[2, 100, 8]` tensor; `inverse ∘ apply` = identity
(exact); encoder output shape `[B, N_ctx, d]` for a spec with **two** context sets (a 4×4 grid and a 5-entity sequence),
finite; zero-presence entities give the same output as zero features (presence gating works).

**Done when:** tests pass on `cpu`.

### G3. Queries, decoder, leakage test

**Do:** in `model.rs`: `build_queries` (§2.2, composed: anchor gather by one-hot matmul as in the committed code, prev-choice
tokens by a one-hot matmul over `[c'_set ; extra_emb ; none_prev]`), `decode` for both `DecoderMode`s (§2.3), returning
`DecoderOut { ctx: Var [B,N_ctx,d], h: Var [B,M,K,d] }`. Query token order: **step-major** (`t = j*M + m`) in
`StepCausal`, query-major (`t = m*K + j`) in `Joint`; `h` is always returned as `[B, M, K, d]`.

**Tests** (`tests/entity_model.rs`):
1. **Leakage test** (§2.3) for `StepCausal { crew_symmetric: false }` and `{ true }`: exactly unchanged outputs for steps
   `≤ j`, changed for `j+1`.
2. `crew_symmetric: true`: changing query `m2`'s features changes query `m1`'s step-0 output for `m1 < m2` **and**
   `m1 > m2` (both directions reached).
3. `Joint` with `autoregressive_on` set is refused by `validate` (a unit test on the spec is enough).

**Done when:** tests pass on `cpu`.

### G4. Heads, batch, loss (composed oracle)

**Files:** `src/models/entity/batch.rs`, `src/models/entity/loss.rs`.

- `HostBatch` becomes generic: `pub struct HostArrays { pub f32s: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
  pub ints: BTreeMap<String, (Vec<usize>, Vec<i64>)> }` keyed exactly as §1.3; `EntityBatch::from_host(spec, &HostArrays,
  device)` validates every key, shape and id range (message names the key) and builds: input tensors, presence tensors,
  anchor `IdTensor`, per-head label `IdTensor`s / float tensors, per-head keep-weight tensors (step weights folded in) and
  per-head host divisors — all on the host, as the committed `PlannerBatch::from_host` does (`planner.rs:578`).
- `EntityModel::heads(&DecoderOut, &choices) -> HeadOutputs` (a `BTreeMap<String, Var>`, logits shaped
  `[B, M, K, width]`, `width` = N+E / classes / labels / outputs; `First` heads `[B, M, 1, width]`).
- `EntityTask: TrainStep` with `with_loss_scale(s)`; `component_losses(&batch) -> BTreeMap<String, Var>`.

**Tests:**
1. Each head type's loss equals a straightforward host `f64` computation on a hand-made batch (1 sample, 2 queries, K = 2,
   one head of each kind, some IGNORE / NaN rows) within 1e-4.
2. `read_count()` delta over one loss + backward = 0.
3. Over-fit: 200 steps on one fixed batch drive the loss below 10% of its start.
4. Loss-scale invariance (f32): scale 1 vs 1024 (eps and clip scaled, `TASK_PLANNER_PLAN.md` §2.5) → parameters within
   1e-5 relative after 20 steps.
5. Absent entities are never chosen: with presence 0 on entity 3, its pointer logit is ≤ -1e4 + ε for every query.

**Done when:** tests pass on `cpu`.

### G5. Predict, generate, save / load

- `predict(batch, Decode::{Greedy, TeacherForced}, chooser: Option<&mut dyn FnMut(usize, &Tensor) -> Result<IdTensor>>)`
  under `no_grad` → `HeadOutputs` + `choices: IdTensor [B, M, K]`.
- `evaluate(batch) -> BTreeMap<String, Tensor>` of per-head metrics on the device (top-1 / top-3 for pointer, top-1 for
  categorical, mean BCE, mean squared error), returned as small tensors so the binding reads them in one go.
- `save(path, step)` stores the spec JSON in the checkpoint metadata; `load(path, device)` rebuilds from it.

**Tests:** generate == teacher-forced-on-own-choices (exact, from G3, now through `predict`); save → load → identical
`predict` output; a chooser that forbids entity 0 is honoured at every step and its choices condition the next step.

**Done when:** tests pass on `cpu`.

### G6. Port the committed planner and delete it

1. `examples/entity_kaggriculture.rs`: the Kaggriculture spec (§1.4) in Rust, random data, 20 steps, prints losses and
   launch / read counts.
2. Port every test of `tests/planner.rs`, `tests/planner_f16.rs`, `tests/planner_footprint.rs` to the generic model built
   from that spec (`tests/entity_model.rs`, `tests/entity_f16.rs`, `tests/entity_footprint.rs`). Same assertions.
3. Behavioural equivalence with the committed model in `Joint` mode (no `autoregressive_on`): same spec shapes, same data,
   both models over-fit the same fixed batch to < 10% of the start loss in 200 steps (weights differ, so no bit parity).
4. Delete `src/models/planner.rs`, `bindings/python/src/planner.rs`, the old tests and examples, the old `.pyi` entries and
   `__init__.py` exports; rename the switch (K0). Grep: `rg -n "TaskPlanner|planner::" src bindings tests examples` must
   return nothing.

**Done when:** full `cpu` suite green with the old files gone.

---

## 5. Tasks: on-device kernels (K0–K6)

Same method, templates and invariants as `TASK_PLANNER_PLAN.md` "On-device kernels" (templates: `src/tensor/ops/entity.rs`,
`Var::pointer_dot` at `ops.rs:1618`, switch pattern `movement.rs:495-525`, parity binaries as `tests/entity_kernels.rs`).
The kernels are now **generic** — no Kaggriculture constant anywhere; every size is a runtime `usize`. New module
`src/tensor/ops/entity_model.rs`; `Var` wrappers in `src/autograd/ops.rs`.

**Invariants** (every K task): composed path kept and correct behind `set_fused_entity_model(false)`; forward parity 1e-6
relative and gradient parity 1e-5 + `check_grad`, on `cpu` and `vulkan`; no host reads, no atomics (gather-style
adjoints); f32 accumulation; `IGNORE = u32::MAX` tested before every indexed read.

| ID | Kernel(s) | Generic contract | Replaces |
|---|---|---|---|
| K0 | switch `set_fused_entity_model` (env `MAMBA3_FUSED_ENTITY_MODEL`), `tests/entity_model_kernels.rs`, `examples/profile_entity_model.rs`, `examples/bench_entity_model.rs` (ported from the planner ones) | – | – |
| K1 | `gather_rows_multi` | up to 8 row-major device tables `[S, w_i]` (f32 or u32) and `ids [B]` → 8 outputs `[B, w_i]`, **one launch**; no adjoint | per-step host batch build and uploads (`EntityDataset` + batch-from-ids) |
| K2 | `gather_tokens` (+ adjoint), `build_queries` (+ adjoint) | `gather_tokens(src [B,S,d], ids [B·R]) → [B,R,d]`, IGNORE → 0; adjoint `d_src[b,s] = Σ_{r: ids=s} g[b,r]` (gather over r). `build_queries(u [B,M,d], anchor tok, step_emb [K,d], prev toks [lags][B,M,K,d] or none, order)` → `[B, M·K, d]` in the decoder's token order; adjoint regions: `d_u` (Σ over K), `d_step` (Σ over B·M), `d_prev` (copy) | one-hot uploads + matmuls; expands, adds, reshapes |
| K3 | `segmented_loss_rows`, `segmented_loss_reduce`, `segmented_loss_backward` | one "segment table" per spec (built once at init: for each head, column offset, width, kind, weight, step selection) drives **one** row kernel over all heads' logits packed per row, one reduction into `[2·H + 1]` (weighted sums and weight sums per head, total), one backward kernel; CE uses max-shifted logsumexp; BCE via softplus; MSE; step weights and IGNORE/NaN handling inside | ≈ 10 launches per head in the composed loss (fwd + bwd) |
| K4 | `permute_tokens` (+ adjoint = inverse permutation, same kernel) | `x [B, n, d]`, `perm [n] u32` → `x[:, perm, :]`; covers grid transposes, within-block reversal and any future ordering | reshape/permute/reshape copies and gathers of the composed permutations |
| K5 | `broadcast_join` (+ adjoint) | `out[b,n] = x[b,n] + pos[slot(n)] + type[set(n)] + g[b]` for the concatenated context, and the same with no pos/type for queries; adjoint regions `d_pos` (Σ_b), `d_type` (Σ over the set's tokens), `d_g` (Σ_n); `d_x = grad` needs no launch | expands, broadcast adds and `cat` of the embedding stage |
| K6 | measure and pin | `profile_entity_model` in both modes on `cpu` and `vulkan`; fill a status table; `tests/entity_footprint.rs` pins the fused launch count per train step for the Kaggriculture spec **and** for one synthetic spec with two context sets; reads = 0; f16 vs f32 A/B (accuracy within 0.3 pp, no non-finite step, ≥ 15% faster, else f32) | – |

Write each K task in the same shape as `TASK_PLANNER_PLAN.md` K1–K5 (inputs/outputs table, thread layout "one thread per
output element (plus regions)", forward formula, adjoint formula, `Var` wrapper, wiring behind the switch, tests: parity,
`check_grad`, IGNORE cases, **two queries anchored at the same entity** (the case a scatter gets wrong), launch counts).
Kernel-specific traps:
- K3: the segment table is data, not code: never generate a kernel per head. Rows with every weight 0 must not divide by
  zero (clamp each head's weight sum to ≥ 1 in the reduce kernel).
- K4: the permutation buffer is uploaded once at `init` and kept on the device.
- K1: validate `ids < S` on the host before upload (the only per-step transfer).

---

## 6. Tasks: Python, generality, Kaggriculture

### P1. Python bindings

**Files:** `bindings/python/src/entity_model.rs`, registered in `bindings/python/src/lib.rs`; `.pyi`; `__init__.py`.
Classes: `EntityModelSpec`, `ContextSet`, `Grid`, `QuerySet`, `Head` (static constructors `pointer`, `categorical`,
`multilabel`, `regression`), `EntityDataset`, `EntityModel`; function `set_fused_entity_model`. Behaviour exactly as §1.4.
Reuse the committed binding's array readers (`planner.rs:32-60` in the bindings) and loss-scale handling.

- `EntityDataset(spec, arrays)`: validates every key against the spec (unknown keys → `ValueError` listing the expected
  keys), converts float16 → float32, ints → the internal `u32` with IGNORE, NaN rules for float labels; uploads once.
- `matmul_precision` in `{"f32", "f16"}`; `"bf16"` → `ValueError("bf16 is not supported")`.
- `read_losses` raises `FloatingPointError` on a non-finite loss or grad norm (naming the step).

**Tests** (`bindings/python/tests/test_entity_model.py`): spec JSON round trip; a bad key / shape / id raises `ValueError`
before anything reaches the device; train 100 steps on a fixed dataset (loss halves); save / load round trip; greedy vs
teacher-forced-on-own-choices equal; chooser honoured; both fused modes give equal `predict` outputs (within 1e-5).

### P2. Generality tests (non-Kaggriculture)

Two small synthetic problems in `bindings/python/tests/test_entity_generality.py`, each trained on CPU in < 60 s:
1. **Nearest free item** (pointer + extras, K = 1): 8 agents on a 1-D line of 16 items (Sequence layout), presence masks
   random; label = the nearest present item not taken by a lower-index agent, else extra action 0 ("none"). Must reach
   ≥ 95% top-1. Exercises presence masking, extras, `crew_symmetric` coordination.
2. **Ordered visits on a grid** (autoregressive, K = 3, conditioned categorical): 1 query on a 6×6 grid with 3 marked cells;
   label = the three marked cells in nearest-neighbour order from the anchor, plus a categorical head = the cell's colour
   (conditioned on the pointer). Must reach ≥ 95% on step 1 and ≥ 90% on step 3 with greedy decoding, and 0 repeated
   cells. Exercises `Grid` + `alternate_axes`, anchors, autoregression, conditioned heads.

These tests are the proof that the API is general. If either needs a library change to be expressible, the API is wrong:
fix the API, not the test.

### P3. Kaggriculture reproduction (acceptance)

In the Kaggriculture repo (`experiments/kobayashi/exp-planner053_dsm_task_planner/`), a script `src/train_entity.py` builds
the §1.4 spec, converts the existing npz files (`runs/20260926_task_planner/data/*.npz`) to the §1.3 keys
(`tiles`, `units`, `units.presence` = `upos >= 0`, `units.anchor` = `upos`, `label.target` = `tgt` with -100 → -1, …),
trains 14 epochs on Vulkan, and scores with its `evaluate.py` through an adapter. Acceptance, test40 scored once:
- next-visit top-1 ≥ 88.8% (within 2 pp of the autoregressive PyTorch v2 reference, 90.83%; v1 was 90.64%), and steps
  2 / 3 ≥ 74.4% / 68.2% (v2: 76.36% / 70.20%);
- consecutive-step repeated tiles (non-shed) ≤ 0.5% of unit-turns (DSM 0.02%; PyTorch v2 0.35%; v1 8.9%, the failure
  this requirement exists for);
- step time ≤ 0.25 s at batch 128 on the fused path, 0 reads per step; report the f16 A/B.

---

## 7. Traps, collected

| Trap | Symptom | Fix |
|---|---|---|
| Domain words in the library | the next problem needs a library edit | nothing named tile/unit/op/crop/eta in `src/` or `bindings/` (G6 grep) |
| Bidirectional mixer over queries with autoregression | step j reads step j+1's teacher-forced input; dev accuracy looks great, inference is worse | `StepCausal` = forward-only; `validate` refuses `Joint` + autoregression; leakage test (G3) |
| Query-major order in `StepCausal` | causality across steps broken | step-major tokens (`t = j*M + m`) |
| Trimming padding | train / inference mismatch | pad to `count` always |
| Pointer to an absent entity | nonsense actions | presence and `legal` masks added as `-1e4` constants |
| IGNORE as an index | out-of-bounds on the device | test before every indexed read; host validation of ids |
| `cross_entropy_with(.., ignore_index)` | a host read per call | S1 |
| A per-head kernel | launches grow with the number of heads | K3 segment table |
| Scatter / atomics in an adjoint | nondeterminism, contention | gather-style adjoints |
| bf16 | cannot be tested here | refused in the binding |
| f16 without loss scale / overflow | stalls / NaN | static loss scale, `FloatingPointError`, f32 fallback |
| Toggling the fused switch in a shared test binary | flaky tests | parity tests in their own binary |

## 8. Out of scope

- **Context heads** (predictions per context entity with no queries, e.g. entity classification): the spec reserves
  `queries: None` for it; implement in a follow-up (a head on `c'` instead of `h`).
- Incremental decoding with `apply_with_state`; attention / cross-attention layers; recurrence across samples; HIP
  debugging; kernels for the mixer, Linear or matmul.
- Merging with the RL `Mamba3Policy`'s entity encoder and pointer head (`src/rl/heads.rs`). The spec types are compatible
  (`From<&EntitySet>`), but sharing code is a separate plan.

## 9. Definition of done

- The generic `EntityModel` (Rust + Python, parity) replaces `TaskPlanner`; nothing domain-specific in `src/` or
  `bindings/`.
- Tests green on `cpu` (and the kernel parity + f16 tests on `vulkan`): `entity_spec`, `entity_model` (incl. the leakage
  test), `entity_model_kernels`, `entity_footprint`, `entity_f16`, Python `test_entity_model`, `test_entity_generality`.
- K6 table filled; fused path the default; 0 reads per step.
- P3 report in the Kaggriculture repo meeting the acceptance numbers, or explaining which one is missed and why.
