# Entity encoder and pointer head plan

This is an execution document. It is written so that it can be worked start to finish without reading the session that produced it.
Each task is self-contained: goal, the measurement that justifies it, files and line anchors, the API, the traps, the test, and the
command that proves it. Work the tasks **in the order given**. Each one lands on its own and leaves the suite green.

**Subject.** A general way to feed a Mamba-3 policy a *set of entities* (tiles, units, cards, items):
- one encoder shared across the entities of a set;
- pooling into the recurrent backbone;
- a pointer action head that scores the entities themselves.

The wire format and the learners stay unchanged.

---

## Why: the measurement

From the Kaggriculture repo (`experiments/kobayashi/exp038_majkel_fast_ranker/runs/20260924_mlp_scorer/REPORT.md`).
The task: predict which of up to 100 tiles a worker walks to next. Same data, same held-out set (reg70, 152,745 decisions).

| Model | Held-out top-1 | Final train loss |
|---|---|---|
| Linear conditional logit, shared across tiles | 73.0% | – |
| MLP scorer, **one scorer shared across tiles** (48 hidden) | **82.5%** | 0.62 |
| `mamba3_rl` policy, **flat** 6,004-dim obs (100 tiles × 60 features + 4), d_model 64, 2 layers | 75.5% | 0.47 |

The Mamba policy fits its training data better and generalises worse. Its `encoder: Linear(obs_dim → d_model)` (`src/rl/policy.rs:202`)
learns separate weights for every tile position, and `actor: Linear(d_model → action_dim)` (:205) learns a separate output per tile.
Nothing is shared between tile 17 and tile 18, so the model must see every pattern at every position.

The shared scorer has one set of weights for "a dry strawberry two steps away", wherever the tile is. The recurrence (the worker's day
history) is worth having, but the flat encoder throws away more than the recurrence adds.

The fix is structural, and useful beyond this task: any environment whose observation is a list of like things wants this.

---

## 0. Before you start

```bash
cd /Users/ods/Documents/mamba-trainer
cargo test --release --no-default-features --features cpu > /tmp/t.log 2>&1; echo "exit=$?"
grep -E "^test result|^error|FAILED" /tmp/t.log
```

The same rules as `ROLLOUT_FUSION_PLAN.md` §0 apply:
- check the exit code, not grep;
- never build wgpu while cpu tests run (shared `target/`);
- never benchmark while anything else runs.

**The working tree carries uncommitted genetic-algorithm work** (`src/evo/`, `bindings/python/src/ga.rs`, edits to `src/lib.rs`,
`bindings/python/src/lib.rs`, `_mamba3_rl.pyi`, READMEs). This plan touches `src/lib.rs`, `bindings/python/src/lib.rs` and
`_mamba3_rl.pyi` too. Land or stash that work first, and never commit it as part of this plan.

---

## Repo map for these tasks

| Path | What lives there |
|---|---|
| `src/rl/policy.rs:84` | `Mamba3PolicyConfig { obs_dim, action_dim, n_layers, ssm, norm_eps, seed }`; `validate()` :138 |
| `src/rl/policy.rs:159-209` | init and fields: `encoder: Linear`, `blocks`, `norm: RmsNorm`, `actor: Linear`, `critic: Linear` |
| `src/rl/policy.rs:251` | `check_obs`: rank 3, `dims[2] == obs_dim` |
| `src/rl/policy.rs:265` | `heads()` → `PolicyOutput { logits [B,T,A], value [B,T] }` (:42) |
| `src/rl/policy.rs:282` / `:337` | `step()` (rollout, `[envs,1,obs_dim]`, no_grad) / `forward()` (parallel scan, TBPTT `initial`) |
| `src/rl/policy.rs:377-387` | `visit`, the parameter path names (`encoder.*`, `blocks.{i}.*`, `norm.*`, `actor.*`, `critic.*`) |
| `src/nn/mlp.rs:45` / `:140` / `:164` | `MlpConfig` / `Mlp` / `Mlp::apply`, rank-N input (applies to the last axis) |
| `src/nn/linear.rs:222` | `Linear::apply`, broadcasts over leading axes, so `[B,T,N,F]` already works |
| `src/nn/attention.rs` | multi-head attention (GQA), for the optional attention pooling in "not in this plan" |
| `src/models/vision.rs:61,404` | `Pooling::{Mean, ClassToken, Last}`, prior art for pooling over a token axis |
| `src/autograd/ops.rs:765` | `Var::mask_logits(legal)` |
| `src/autograd/ops.rs:784/793/806` | `sum_dim` / `mean_dim` / `max_dim` |
| `src/autograd/ops.rs:1113` | `take_along_last` |
| `src/rl/buffer.rs:122` | `TrajectoryBuffer` `[envs,steps,obs_dim]`, flat and unchanged by this plan |
| `src/rl/collect.rs:345-430` | `Collector::run_window`: mask → sample; unchanged |
| `src/rl/ppo.rs:483` / `src/rl/imitation.rs:276` | PPO objective / behaviour-cloning loss; both consume `logits [B,T,A]`; unchanged |
| `src/rl/game.rs:117` | fused `GameSpec` (comptime obs/action dims); structured policies are refused there (see invariants) |
| `bindings/python/src/config.rs:65-317` | `PyPolicyConfig`; `as_json` :278, `from_json` :290 (optional-with-default keys) |
| `bindings/python/src/env.rs:557` | `check_against_policy(env, obs_dim, action_dim)` |
| `bindings/python/python/mamba3_rl/_mamba3_rl.pyi` | the hand-written stubs; update with every API change |
| `tests/autograd.rs:18` | `check_grad`, central differences, tolerance `2e-2*(1+|n|)` |
| `tests/rl.rs:170,420` | `rollout_matches_the_parallel_scan`; `a_trajectory_pass_is_differentiable_and_reaches_every_parameter` |

---

## The design

### Wire format: unchanged

The observation stays one flat `float32[obs_dim]` per environment, so `VecEnv`, `TrajectoryBuffer`, `Collector`, the Python adapter,
checkpoints of rollouts and every existing environment are untouched. A new **`ObsSpec`** tells the *policy* how to read the flat
vector:

```
flat obs = [ globals (G) | set_1: N_1 × (F_1 + 1) | set_2: N_2 × (F_2 + 1) | ... ]
                                  └ per entity: F features, then 1 presence flag (1 = the entity exists)
obs_dim  = G + Σ_k N_k · (F_k + 1)
```

A fixed maximum `N_k` with a presence flag handles a variable number of entities without ragged tensors. For example, Kaggriculture's
candidate tiles are always 100 slots, and the tiles that are not candidates have presence 0.

### The model

```
globals ─────────────────────────────────────────────┐
set_k: [B,T,N_k,F_k] ─ EntityEncoder_k (shared MLP) ─ e_k [B,T,N_k,d_e]
                     └ optional slot embedding (learned, per index)          │
                        masked pooling (mean, max, and/or attention later) ──┤
                                                                   concat ── Linear → [B,T,d_model]
                                                                              │
                                              Mamba3Block × n_layers → RmsNorm → h [B,T,d_model]
                                                                              │
     actor: Flat  = Linear(h) → [B,T,A]                                      │ (today, default)
            Pointer(set_k) = score(h_t, e_k,i) for i in 0..N_k → [B,T,N_k] ───┤
            Hybrid = [ Pointer logits | Linear(h) → K extra actions ]          │
     critic: Linear(h) → [B,T] (unchanged)
```

- **Shared entity encoder:** `Mlp` applied to `[B,T,N,F]`. It is the same weights for every entity of a set, so it is permutation
  equivariant. An optional **slot embedding** (`[N,d_e]`, added) lets a model use position when it matters: tile coordinates in a
  fixed grid, for example. It is off by default, because it reintroduces per-slot parameters.
- **Masked pooling:**
  - mean = `sum_dim(e · p) / max(1, sum_dim(p))`;
  - max = `max_dim(e + (p − 1)·BIG)` (absent entities pushed to −BIG);
  - both are composed from existing `Var` ops, so they need no new kernels and no hand-written backward.
- **Pointer head (additive scoring):** `logit_i = v · relu(W_h h_t + W_e e_i + b)`. It is `[B,T,N]` from broadcasting
  `[B,T,1,H] + [B,T,N,H]`. `mask_logits` with the presence flags is applied inside the head, and the learner's `action_mask` is applied
  afterwards as today.
  - The action id *is* the entity index, so `action_dim == N_set` (+ K for `Hybrid`).
  - `ImitationLearner` expert ids, `Rollout` masks and PPO log-probs need no change.
  - A dot-product variant (`h·W e_i`) is cheaper but measured worse on scorer tasks. Keep additive as the default and offer `"dot"` as
    an option.

### The Python API

```python
import mamba3_rl as m3

spec = m3.ObsSpec(
    globals=4,
    sets=[m3.EntitySet("tiles", count=100, features=60)],   # more sets: units, items, ...
)
cfg = m3.PolicyConfig(
    obs_dim=spec.obs_dim, action_dim=100, d_model=64, n_layers=2,
    obs_spec=spec,                                              # None (default) = today's flat policy
    entity_encoders={"tiles": m3.EntityEncoderConfig(hidden=[64], d_entity=48, slot_embedding=False)},
    pooling=m3.PoolingConfig(kinds=("mean", "max")),
    action_head=m3.PointerHead("tiles", hidden=48, scoring="additive", extra_actions=0),
)
policy = m3.Policy(cfg)

# numpy helpers that build and read the flat wire format, so environments never hand-compute offsets
obs = spec.pack(globals=g, tiles=(features[100, 60], present[100]))   # -> float32[obs_dim]
parts = spec.unpack(obs)                                              # -> {"globals": ..., "tiles": (F, present)}
batch = spec.pack_batch(...)                                          # [num_envs, obs_dim]
```

**Rules:**
- keyword-only;
- every new knob is off by default;
- `PolicyConfig(obs_dim, action_dim)` alone still builds the flat policy, byte-identical to today;
- `PolicyConfig.validate()` refuses:
  - `obs_dim != spec.obs_dim`;
  - a pointer head over a set whose `count + extra_actions != action_dim`;
  - an encoder for an unknown set.

**Parameter paths** (for freeze, fingerprint, checkpoint and EMA):
- `entity.{set}.mlp.*`, `entity.{set}.slot`;
- `pool.proj.*`;
- `actor.pointer.{w_h,w_e,b,v}`, `actor.extra.*`.

The flat policy's names are unchanged.

**Checkpoints:**
- `metadata.policy` gains the optional keys `obs_spec`, `entity_encoders`, `pooling`, `action_head`;
- `from_json` treats absence as the flat policy, following the pattern at `config.rs:290`;
- policy files need no format version bump; learner checkpoints are unaffected;
- architecture stays non-adoptable (`resume.rs:71`).

**Export for runtime without the trainer:**
- `Policy.export_numpy(path)` writes an `.npz` with the config and weights;
- `mamba3_rl.numpy_ref` (pure numpy, no compiled code) provides `Policy.step(obs, state)`;
- this lets a Kaggle agent run the trained model. The existing Kaggriculture port (`exp013 mamba3_numpy.py`, reused by exp017) is the
  template for the recurrent part; this plan adds the entity encoder, pooling and pointer head to it and moves it upstream.

---

## Invariants that must survive every task

1. **Defaults unchanged.** Flat policies are byte-identical: `Policy.fingerprint()` of `PolicyConfig(obs_dim, action_dim, seed=s)` and
   all golden files in `tests/golden/` do not change.
2. **No silent fallback.** Structured policies are refused with a clear error by anything that cannot handle them. The fused rollout
   (`src/rl/fused.rs`, `GameSpec`) refuses a structured policy up front; it does not quietly run the flat path.
3. **Replay ratio = 1** under masks (`tests/rl_masking.rs`), and rollout == parallel scan (`tests/rl.rs:170`) for structured policies.
4. **Zero per-step host reads** in `step()` for structured policies (footprint suites `tests/rl_*footprint.rs`).
5. **CPU and GPU parity; Python and Rust feature parity** (`docs/test_guidline.md`).

---

## Status

| ID | Task | Status |
|---|---|---|
| E0 | Fixtures and baselines | todo (needs the Kaggriculture export) |
| E1 | `ObsSpec` (Rust + Python), pack/unpack, validation | done |
| E2 | `EntityEncoder` + masked pooling | done |
| E3 | Pointer head (additive / dot, hybrid extra actions) | done |
| E4 | Wire into `Mamba3Policy`, config, checkpoints, stubs | done |
| E5 | `export_numpy` + `numpy_ref` + parity | done (in-repo; the Kaggriculture port is not replaced yet) |
| E6 | Kaggriculture validation (acceptance) | todo |

### Where it landed

| Piece | Files | Tests |
|---|---|---|
| E1 | `src/rl/spec.rs`; `bindings/python/src/entity.rs` (`ObsSpec`, `EntitySet`) | `tests/rl_entity.rs`, `bindings/python/tests/test_entity.py` |
| E2 | `src/nn/entity.rs` | `tests/rl_entity.rs` (check_grad, permutation, mask invariance, empty set) |
| E3 | `src/rl/heads.rs` | `tests/rl_entity.rs` (equivariance, empty slot = `f32::MIN`, BC "pick the largest": pointer ≥ 0.99, flat < 0.5) |
| E4 | `src/rl/policy.rs`, `bindings/python/src/config.rs`, `_mamba3_rl.pyi` | rollout == scan across resets, every parameter gets a gradient, checkpoint round trip, pre-structure checkpoint loads flat, `tests/rl_entity_footprint.rs` (0 reads, flat launches), `tests/rl_fused.rs` (structured fused == unfused) |
| E5 | `Policy.export_numpy`, `python/mamba3_rl/numpy_ref.py` | 64 envs × 200 steps with resets, structured and flat, max rel. error < 1e-4 |

### Speed: what the implementation does, and what it costs

The trainer is host-bound (each launch costs more than its arithmetic), so every piece was built to minimise
launches and to never read back:

- **Split:** one fused `split` along the observation axis for globals and all sets, then one split per set to peel
  off the presence column. Reshapes are free.
- **Presence statistics** (mean weights `p / max(1, Σp)` and "any present") are plain tensors computed once per set
  per call, off the tape.
- **Mean pooling** is one batched matmul `[B,T,1,N] @ [B,T,N,d]` instead of multiply + reduce + divide.
- **Max pooling** reuses the fused `mask_logits` kernel. Absent slots become the most negative finite float,
  replacing the plan's `BIG = 1e4`, so a large embedding can never outrank the mask. Then one reduction and one
  multiply zero an empty set.
- **One concatenation and one projection** for globals plus every pool of every set (`pool.proj`), rather than one
  projection per part summed afterwards.
- **Pointer:** the bias `b` rides on `W_h` (stored as `actor.pointer.w_h.bias`), which saves an add over `[B,T,N,H]`.
  The additive head is 7 launches, the dot head 3.

Measured with `m3.launch_count()` for a 64-environment `Rollout.evaluate` step, with Kaggriculture shapes
(4 globals, 100 tiles × 60 features, d_model 64, 2 layers, encoder `[64] → 48`):

| architecture | parameters | launches / step | host reads / step |
|---|---|---|---|
| flat (today) | 428,491 | 55 | 1 (the evaluate readback) |
| structured, flat head | 51,259 | 76 | 1 |
| structured, additive pointer | 50,231 | 81 | 1 |
| structured, dot pointer | 47,831 | 77 | 1 |

The entity stage adds about 21 launches per step and has 8× fewer parameters than the flat encoder. The next levers,
if a profile says the stage matters, are:
- a single fused presence-statistics kernel (4 launches → 1);
- a fused bias + ReLU inside the encoder MLP.

### Deviations from the design above

- **Fused rollout accepts structured policies** instead of refusing them. `collect_fused` fuses the steps *around*
  `Mamba3Policy::step`, not the policy itself, so there is no flat path to fall back to silently.
  `tests/rl_fused.rs::a_structured_policy_collects_the_same_window_fused_and_unfused` holds the two paths to identical
  bytes.
- **Pointer parameter paths:** `actor.pointer.{w_h.weight, w_h.bias (= b), w_e.weight, v.weight}` and, for `dot`,
  `actor.pointer.w_q.weight`. The encoder MLP is `entity.{set}.mlp.{i}.{weight,bias}`.
- **Empty-slot features are zeroed** (`features × presence`) before the encoder, so arbitrary (even non-finite)
  values in an empty slot never reach anything.
- **`n_layers = 0` is still refused**, so the E6 "no recurrence" ablation needs that relaxed first.

---

## E0 — fixtures and baselines

- **Goal:** fix the measurements that later tasks must beat, and a small dataset to test on.
- **Do:**
  - Export the Kaggriculture destination data as `.npz`:
    - per decision `globals[4]`, `tiles[100,60]`, `present[100]`, `label`;
    - `sequence_id` and `step` for the worker-day sequences.
  - Export 3,000 train decisions (fixture) and the full train/eval sets (outside the repo).
  - Record the three baseline numbers above in this file with their commands.
- **Test:** none; the fixture's sha256 goes into this file.

## E1 — `ObsSpec`

- **Rust:** `src/rl/spec.rs`:
  - `ObsSpec { globals: usize, sets: Vec<EntitySet { name, count, features }> }`, `obs_dim()`, `validate()`;
  - `split(&Var[B,T,obs_dim]) -> (globals [B,T,G], Vec<(features [B,T,N,F], present [B,T,N])>)`, built from narrow/reshape (no new ops).
- **Python:** `m3.ObsSpec`, `m3.EntitySet`, `pack` / `unpack` / `pack_batch` in pure numpy.
- **Tests:**
  - Python `pack`→`unpack` round trip;
  - Rust `split` of a packed vector equals the parts;
  - a spec with no sets is the flat policy's `obs_dim`;
  - validation errors name the field.

## E2 — `EntityEncoder` and masked pooling

- **Rust:** `src/nn/entity.rs`:
  - `EntityEncoderConfig { hidden, d_entity, slot_embedding }` → shared `Mlp` over the last axis, plus an optional `Param [N,d_e]`;
  - `masked_pool(e, present, kinds) -> [B,T,k·d_e]`.
- **Traps:**
  - Max over an all-absent set must not produce −BIG: return 0 where `sum(present) == 0`.
  - Keep `BIG` finite (1e4, not `f32::MAX`), or `mask_logits`' gradient path overflows in f16.
- **Tests:**
  - `check_grad` on encoder and pooling;
  - **permutation invariance**: permuting entities (with their presence) leaves the pooled vector unchanged within 1e-6;
  - **mask invariance**: changing features of absent entities leaves the output unchanged;
  - CPU and GPU parity.

## E3 — pointer head

- **Rust:** `src/rl/heads.rs`:
  - `enum ActionHead { Flat(Linear), Pointer(PointerHead), Hybrid(PointerHead, Linear) }`;
  - `PointerHead { w_h, w_e, b, v, scoring }` returns `[B,T,N]` (+K).
- **Tests:**
  - `check_grad`;
  - **permutation equivariance**: permuting entities permutes the logits the same way;
  - absent entities get probability exactly 0;
  - behaviour cloning on a synthetic "pick the entity with the largest feature 3" task reaches ≥ 99% in < 200 rounds, where the flat
    head on the same task stays near chance at N = 100.

## E4 — wire into the policy

- **Config and forward:**
  - `Mamba3PolicyConfig` gains `obs_spec: Option<ObsSpec>`, `entity_encoders`, `pooling`, `action_head`, all `#[serde(default)]`.
  - `Mamba3Policy` holds `input: InputStage { Flat(Linear) | Entities { encoders, pool_proj } }` and `actor: ActionHead`.
  - `heads()`, `step()` and `forward()` call the stage; `check_obs` is unchanged (it still checks the flat `obs_dim`).
- **Surface:**
  - `visit` emits the new paths;
  - `PyPolicyConfig` gets the keyword arguments and `as_json`/`from_json`;
  - `.pyi` stubs are updated with the semantics.
- **Tests:**
  - the flat fingerprint is unchanged (invariant 1);
  - `rollout_matches_the_parallel_scan` and `…reaches_every_parameter` pass for a structured policy;
  - the checkpoint round trip; an old checkpoint loads as flat;
  - `check_against_policy` still compares flat dims;
  - the fused rollout refuses a structured policy with a clear message.

## E5 — numpy export and parity

- **Do:** `Policy.export_numpy(path)` and `mamba3_rl/numpy_ref.py`:
  - pure numpy `Policy.step(obs[N,obs_dim], state) -> (logits, value, state)`;
  - it covers flat and structured policies.
- **Test:** 64 random environments × 200 steps of a structured policy; `numpy_ref` logits equal Rust `Rollout.evaluate` within 1e-4
  (f32), with and without resets.
- **Downstream:** replace the Kaggriculture repo's copied port with this module.

## E6 — Kaggriculture validation (acceptance)

- **Do:**
  - Retrain the destination model: `ObsSpec(globals=4, tiles 100×60)`, shared encoder, mean+max pooling, `PointerHead("tiles")`,
    `ImitationLearner` with `DaggerSchedule.fixed(1.0)`, the same 30 train files.
  - Evaluate on reg70.
- **Acceptance:** held-out top-1 ≥ **82.5%** (the shared MLP scorer) for the model to be considered for the agent, at CPU training
  time ≤ the MLP's (~15 min).
- **Report either way.**
  - If it wins, the agent integration follows the usual sealed plan in the Kaggriculture repo: same-state P2 agreement, closed-loop
    production, confirmation, with the numpy export at runtime.
  - Also report whether the recurrence adds anything over the shared scorer. Ablation: the same model with `n_layers=0` (pooling →
    head only), if the config allows it.

---

## What is deliberately not in this plan

- **Attention between entities** (a transformer layer over the set, or attention pooling with a learned query). `src/nn/attention.rs`
  makes it possible; it is phase 2, after E6 shows whether pooling plus pointer already closes the gap.
- **Ragged / truly variable-length sets.** A fixed maximum with presence flags covers the use cases seen so far.
- **Multi-discrete or factored actions** (e.g. "which tile" × "which op"). `distributions::Independent` (`combinator.rs:39`) is the
  starting point if needed.
- **Per-entity value heads.** The critic stays a single scalar from `h`.
- **Fused-rollout support for structured policies.** It is refused, not emulated; the fused path's gains were measured small for this
  workload (`ROLLOUT_FUSION_PLAN.md`, and exp010's 3.1% environment share).
