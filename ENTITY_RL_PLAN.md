# Reinforcement learning for the entity model (PPO) and its Python bindings

This is an execution document, written so each task can be implemented without reading the session that produced it.
Work the tasks in order; each lands on its own and leaves `cargo test --features cpu` green. Tests run on both CPU and
GPU, and Python and Rust stay at feature parity (`docs/test_guidline.md`).

## 0. What exists and what is missing

`EntityModel` (`src/models/entity/model.rs`) is a supervised planner: context entity sets (e.g. a 10x10 tile grid) plus
globals are encoded by bidirectional Mamba-3 blocks; `M` query entities (units) each get a `K`-step plan from a decoder;
heads read the decoder states — pointer heads over a context set (+ learned extra actions), categorical, multilabel and
regression heads, optionally conditioned on a pointer head's choice at the same step. It is trained by behaviour cloning
(`EntityTask`, `src/models/entity/loss.rs`) and decodes greedily step by step (`EntityModel::predict`, `Decode::Greedy`,
with an optional `chooser` that picks the pointer ids). Python: `EntityModelSpec`, `EntityDataset`, `EntityModel`
(`bindings/python/src/entity_model.rs`).

The crate's RL stack (`src/rl`: `Mamba3Policy`, `Collector`, `ppo_objective`, `PpoConfig`, `generalized_advantage` in
`src/tensor/ops/rl.rs`, `sample_categorical`) serves a *flat recurrent* policy with one categorical action per step. There
is no way to train the entity model from rewards. This plan adds that: the entity model becomes the actor of an
actor-critic, plans are sampled instead of taken greedily, and PPO updates it from rewards an external environment
(usually Python) supplies.

## 1. Design (decided)

- **Actions.** Every pointer head and every categorical head is an action, at every (query, step) cell it covers
  (`StepSelection::All`: all `K` steps; `First`: step 0 only). Multilabel and regression heads are not actions: `act`
  returns their outputs for the environment to use as it likes, and they get no policy gradient. Cells of absent queries
  (`presence = 0`) take no action (id `-1`). Pointer logits already carry the presence and `legal.<head>` masks, so a
  sampled pointer never picks an absent or illegal entity.
- **Sampling, fully on the device.** `EntityPolicy` is device-resident: between the upload of an observation batch
  and the single read that hands the actions to the caller, nothing leaves the GPU. `act` does *not* reuse
  `EntityModel::predict`, which reads the chosen ids back to the host after every step (`ids.to_vec()`) and rebuilds the
  next step's tokens from host ids. Instead it runs its own decode loop on device tensors only: the encoder stem once
  (anchor tokens from the device anchor ids, `anchor_tokens_ids`), then for step `j in 0..K` build the queries from the
  device choice table (`prev_tokens_ids`), run the decoder, take step `j`'s pointer logits `[B*M, N+E]` (they already
  carry the presence and `legal.<head>` masks), draw ids and their tempered log-probabilities with `sample_categorical`
  (a per-call, per-step, per-head seed), set absent queries to `IGNORE` on the device, write the ids into column `j` of
  the `[B*M, K]` device choice table with `write_step_ids`, then compute the conditioned heads for step `j` with
  `ChoiceIds::Device` and draw the categorical heads the same way. Per-cell log-probabilities and entropies accumulate in
  device tensors. At the end, one `read_together` returns every action id table, the per-cell log-probabilities, the
  values and the logits the caller asked for. `T = 0` is greedy (argmax on the device). Requires the fused entity path
  (`fused_entity_model()`, the default); the RL entry points refuse the composed path with a clear error.
- **Read budget (tested).** `act`: exactly 1 device-to-host read per call. `update`: exactly 1 read per call (the
  statistics), with the rollout uploaded once, GAE, advantage normalisation and minibatch gathering on the device.
  `backend::read_count()` pins both.
- **Re-scoring by teacher forcing.** PPO needs `log π(a|s)` of stored actions under new weights. Instead of re-decoding,
  run the training forward with the stored actions as the teacher-forced choices (`label.<pointer head>` = sampled ids):
  for a decoder in which step `j` cannot see steps `> j` this reproduces the step-by-step logits exactly. That holds for
  `DecoderMode::StepCausal` and for any decoder when the spec has no `autoregressive_on`. `DecoderMode::Joint` with
  `autoregressive_on` is **refused** by the RL entry points with a clear error (step `j` would see later steps' choices
  that did not exist when it was sampled). `QueryCausal` (query-major decoding, one pass per query and step) is refused in this version: the device loop decodes
  one step for all queries at a time.
- **Log-probability granularity.** One log-probability per cell: the sum over action heads of `log softmax(logits / T)`
  at the chosen id. PPO takes one ratio per cell (`exp(new - old)`), the sample's advantage is shared by its cells, and
  the surrogate is the mean over acted cells — per-token PPO, as in LLM fine-tuning. A per-sample joint ratio over up to
  `M * K` cells would be the product of ~60 ratios and leave the trust region on every update.
- **Critic.** A new value head, `EntityValueHead`: `Linear(d, d) -> GELU -> Linear(d, 1)` over the presence-weighted
  mean of the *encoder* output (the context tokens after the context blocks, before the decoder) plus the embedded
  globals. It depends only on the observation, never on sampled actions, in every decoder mode.
- **Actor-critic.** `EntityActorCritic { model: EntityModel, value: EntityValueHead }`, parameters `model.*` and
  `value.*`. Built from a spec (fresh weights) or from an existing `EntityModel` (a behaviour-cloned planner; fresh
  value head). Saved as one checkpoint that also holds the spec.
- **Advantages.** The caller supplies time-major rollouts: `reward[t, e]`, `done[t, e]`, the stored `value[t, e]` and
  `last_value[e]` (the critic's estimate of the observation after the last step). GAE is `generalized_advantage` on the
  transposed `[envs, steps]` layout. Advantages are normalised per update (`PpoConfig::normalize_advantages`).
- **Update.** `epochs` passes over the rollout in `minibatches` shuffled minibatches, one optimizer step each (AdamW,
  gradient clipping), the loss `-surrogate + c_v * value_loss - c_H * entropy` with `PpoConfig`'s clip, value clip and
  coefficients. Nothing is read back until the end of the update: statistics are read once (`Trainer::read_steps`).

## 2. Tasks

### R1. Rust core — `src/models/entity/rl.rs`

Add the module (export it from `src/models/entity/mod.rs`) with:

- `EntityValueHead<R, E>` (Module; `visit` names `lin1`, `lin2`), `forward(ctx: &Var [B, N_ctx, d], presence: [B, N_ctx]
  0/1, globals: Option<&Var [B, d]>) -> Var [B]`.
- `EntityModel` additions in `model.rs` (small, public): `encode_with_globals(batch, traced: bool) -> (ctx, g)` or
  equivalent so the value head can reuse the encoder output of the same forward (do not encode twice in the PPO loss:
  the teacher-forced forward and the value head share one `encode`). If `train_decode_with` does not expose the encoder
  output, add a variant that returns it.
- `EntityActorCritic<R, E>`: `init(spec, device)`, `from_model(model)`, `model()`, `value_head()`, `Module` impl,
  `save(path, step)` / `load(path, device)` (one checkpoint with the spec, like `EntityModel::save`), and
  `check_rl_support(&spec) -> Result<()>` implementing the decoder rule of §1.
- `Acted` (device tensors) and `ActedHost` (`actions: BTreeMap<String, Vec<i64>>` `[B*M*K]` with `-1` = no action,
  `cell_log_prob`, `value`, `outputs` as host vectors with shapes).
- `act(&self, batch: &EntityBatch, temperature: f32, seed: u64) -> Result<Acted>` (no grad): the device-resident
  decode loop of §1 (not `predict`). `Acted` holds device tensors (`actions: BTreeMap<String, IdTensor> [B, M, K]`,
  `cell_log_prob [B, M, K]`, `cell_entropy [B, M, K]`, `value [B]`, `outputs`: per-head logits); a separate
  `Acted::read(&self) -> Result<ActedHost>` does the single `read_together`. The value comes from the value head on
  the same stem's encoder output. Make `infer_stem` usable here (`pub(crate)`) or add a device-id variant of it.
- `evaluate_actions(&self, batch_with_actions: &EntityBatch, temperature) -> Result<(Var cell_log_prob [B,M,K], Var
  entropy [B,M,K], Var value [B])`: one teacher-forced forward (`forward_train`) with the actions as `label.<head>`,
  per-head `-cross_entropy_rows(ids)` on `logits / T` (`T = 0` treated as 1 for scoring), summed per cell over action
  heads; cells whose id is `-1` contribute 0; entropy of each action head's tempered distribution, summed per cell.
- `pub fn actions_to_arrays(acted: &ActedHost, spec) -> HostArrays` (the `label.<head>` int arrays) so a batch for the update is
  built with the ordinary `EntityBatch::from_host` / dataset path.
- `EntityPpoBatch<R, E> { batch: EntityBatch /* with label.<head> = actions */, old_log_prob: Tensor [B,M,K],
  cell_mask: Tensor [B,M,K] /* 1 where any action head acted */, advantages: Tensor [B], returns: Tensor [B],
  old_values: Tensor [B] }`.
- `entity_ppo_objective(ac, batch, config: &PpoConfig, temperature) -> Result<PpoLoss>`: per-cell surrogate with
  `Var::ppo_surrogate` (advantages broadcast to cells), masked mean over cells; value loss per sample with
  `Var::ppo_value_loss`; entropy masked mean; `approx_kl` / `clip_fraction` from `fused::ppo_diagnostics` over acted
  cells. Reuse `PpoLoss`.
- `EntityPpoTask` implementing `TrainStep` over `EntityPpoBatch` (like `PpoTask` in `src/rl/ppo.rs`, including
  recorded stats for a single read).
- `pub fn entity_gae(rewards: &[f32] [T*E time-major], values, dones, last_value: &[f32] [E], t, e, gamma, lambda,
  device) -> Result<(Tensor advantages [T*E], Tensor returns [T*E])>` in time-major sample order, via
  `generalized_advantage` on the transposed layout.

Tests `tests/entity_rl.rs` (CPU; must also pass on CUDA): a tiny spec (4-9 context entities, 2-3 queries, K = 2,
pointer + conditioned categorical + a multilabel head, StepCausal, d_model 16):
1. **Teacher forcing equals decoding:** sample with `act` at T = 1; the per-cell log-probabilities accumulated during
   the device decode equal the teacher-forced `evaluate_actions` values for the same actions within 1e-4 (this is what
   makes the PPO ratio exactly 1 before the first update); at T = 0 `act`'s pointer ids equal greedy `predict`'s. Same for a spec without `autoregressive_on`. `Joint` + autoregressive
   is refused with an error mentioning the decoder mode. `QueryCausal` + autoregressive is refused the same way.
2. Sampling respects masks: over many seeds no pointer id ever names an absent entity or a `legal = 0` one; absent
   queries get `-1` everywhere; `T = 0` is deterministic and equals greedy `predict`.
3. Gradients reach every parameter (model and value head) from `entity_ppo_objective`.
4. **Learning:** a contextual bandit — each sample has one "target" context entity marked by feature 0 = 1 at a random
   index; reward = mean over present queries of `[pointer at step 0 == target]`. Batch 64, PPO (clip 0.2, entropy 0.01,
   lr 3e-3, 4 epochs x 2 minibatches, one-step episodes: `done = 1`, `last_value = 0`). Mean reward rises from about
   `1/N` to > 0.8 within 60 updates on CPU. Keep the test under a minute in release.
5. Save / load round trip: same seed, same actions and log-probabilities.
6. `entity_gae` matches a host reference on a random 3 x 5 rollout with a done in the middle.
7. **On-device pins:** after a warm-up, `reset_read_count()`, one `act` + `Acted::read` → `read_count() == 1`; one PPO
   update of several minibatches through the trainer (queue the steps, read the statistics once) → `read_count() == 1`.

### R2. Python bindings — `bindings/python/src/entity_rl.rs`

`mamba3_rl.EntityPolicy` (registered in `lib.rs`, exported from `python/mamba3_rl/__init__.py`, typed in
`_mamba3_rl.pyi`):

```python
policy = m3.EntityPolicy(spec, *, learning_rate=3e-4, weight_decay=0.0, max_grad_norm=0.5, gamma=0.99, lam=0.95,
                         clip=0.2, value_coef=0.5, entropy_coef=0.01, temperature=1.0, seed=0)
policy = m3.EntityPolicy.from_model(model, **same_kwargs)      # start from a behaviour-cloned EntityModel
out = policy.act(obs, *, greedy=False)
# one upload, one device-to-host read per call
# out = {"actions": {head: int64 [B, M, K]} (-1 = no action), "log_prob": float32 [B, M, K],
#        "value": float32 [B], "outputs": {head: float32 logits as EntityModel.predict returns}}
stats = policy.update(obs, actions, log_prob, value, reward, done, last_value, *, epochs=4, minibatches=4)
# obs: dict of arrays with T*E samples, time-major (sample t*E + e); actions: {head: [T*E, M, K]};
# log_prob: [T*E, M, K]; value: [T*E] or [T, E]; reward, done: [T, E]; last_value: [E]
# stats = {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"}
policy.value(obs) -> float32 [B]
policy.save(path); m3.EntityPolicy.load(path)
policy.to_model() -> m3.EntityModel        # the actor's weights as a supervised model (predict / evaluate / fine-tune)
```

`update` uploads the rollout once (an `EntityDataset` built from `obs` plus the `label.<head>` action arrays), computes
GAE, and gathers minibatches by id on the device, like `EntityModel.queue_train_step`. One host read per `update`,
one per `act` (the Python tests pin both with `m3.read_count()` if the module exposes it; check `lib.rs`).

Tests `bindings/python/tests/test_entity_rl.py`, mirroring R1: shapes and `-1` for absent queries; masked entities never
chosen; greedy deterministic and equal to `EntityModel.predict` of `to_model()`; the contextual bandit learns (> 0.8);
save / load round trip; `from_model` keeps the model's greedy predictions; errors for a `Joint` + autoregressive spec and
for mismatched rollout shapes. Example `bindings/python/examples/entity_ppo_bandit.py` (the bandit, printing mean reward
per update).

### R3. Docs

A section in `README.md` (RL for the entity model: the loop, the API, the decoder rule), and the status table below.

## 3. Status

| Task | Commit | CPU | GPU (T4) | Note |
|---|---|---|---|---|
| R1 | uncommitted | tests/entity_rl.rs 8/8, tests/entity_rl_reads.rs 2/2 (act = 1 read, update = 1 read; act launches pinned: tiny 296, two-ptr 289, down from 350/395); entity_footprint unchanged; entity_model_kernels 12/12; entity_model 15/15; entity_loss green | wgpu (Metal, M1): entity_rl 8/8, entity_rl_reads 2/2, entity_model_kernels 12/12, entity_model 15/15, ssd_scan 7/7, ssd_scan_chunk8 4/4, matmul_paths 3/3 (after the tuner fix, §4). CUDA (T4, before the act fix): entity_rl 8/8, entity_rl_reads 1/1 | read-count pin moved to its own binary (the counter is process-wide); `QueryCausal` + autoregressive refused |
| R2 | uncommitted | bindings/python/tests/test_entity_rl.py 13/13, test_entity_model.py 8/8 (cpu wheel; act = 1 read, update = 1 read pinned with m3.read_count; bandit 0.16 -> >0.8 in 5 updates) | wgpu wheel (Metal, M1): test_entity_rl.py + test_entity_model.py 21/21 | observations re-upload per minibatch: `EntityDataset::from_arrays` reads every tensor back, which would break the 1-read update; a read-free dataset builder would restore upload-once |
| R3 | uncommitted | README.md "Reinforcement learning for the entity model" + shorter bindings/python/README.md subsection; README example run on CPU (`.venv-rl`): bandit 0.19 -> 0.89 in 6 updates, save/load/to_model OK; `help(m3.EntityPolicy)` OK | – | docs only, no code changed |


## 4. Open issues

- **wgpu (Metal, Apple M1) forward pass was not deterministic — root cause found.** At commit `3a7bed6` (before this plan)
  `tests/entity_model_kernels.rs` failed 5 tests on `--features wgpu`, and one `BiBlock` applied twice to the same input differed
  by up to 11.5. Cause: the matmul tuner could pick `BlockV(bm 128, bn 128, bk 16, tm 8, tn 4)` (512 units per cube), which Metal
  silently drops for these kernels (a pipeline's threads per threadgroup are limited by its register use, below the adapter's 1024,
  and nothing reports it). A dropped launch times as the fastest candidate, so it won, and every later product of that shape
  returned stale memory. `MAMBA3_TUNE_CHECK` missed it because each candidate ran into an output already holding the reference
  answer; with the output poisoned to NaN first, the check fails on exactly that candidate. Fixed: the tuner drops any
  candidate whose output does not match the simple kernel (two device-side sums, one read per newly tuned shape), and the on-disk
  plan cache version is bumped (FORMAT 3) so plans chosen before the fix are ignored; `tests/matmul_paths.rs` `tuned_projection_shapes_match_host` pins it. After the fix every entity, RL and scan suite passes on wgpu.
