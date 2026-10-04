# mamba3-rl

Reinforcement learning on [Mamba-3](../../README.md) state space models, from
Python, on the GPU.

A recurrent policy is the awkward case for reinforcement learning: acting wants
the smallest possible step and learning wants the largest possible batch, and most
architectures are cheap at only one of them. A transformer acts by carrying a
cache that grows with every action it takes. A state space model does not — its
state is a fixed-size buffer, overwritten in place — and the same window of
experience replays through a parallel scan whose depth is logarithmic in its
length.

| | entry point | cost per step | state |
|---|---|---|---|
| acting | `Rollout.step`, inside `PpoLearner.collect` | `O(1)` | fixed, `[layers, envs, heads, head_dim, d_state]` |
| learning | the replay inside `PpoLearner.update` | `O(T)`, `O(log T)` depth | the tape |

Both honour the same episode mask, and they agree numerically — which is what
makes the pair usable, because a policy gradient estimated from the second has to
be a gradient of what the first actually did.

---

## Install

Pre-built wheels are separate PyPI projects, one per backend — pip has no way
to pick a compile-time feature under a single distribution name, and each
backend is a compile-time choice:

```bash
pip install mamba3-rl          # CPU, portable, no GPU toolchain needed
pip install mamba3-rl-wgpu     # wgpu, WGSL; picks Vulkan/Metal/DX12 underneath
pip install mamba3-rl-vulkan   # wgpu, shaders compiled to SPIR-V (Linux/Windows)
pip install mamba3-rl-msl      # wgpu, shaders compiled to MSL (macOS/Metal)
pip install mamba3-rl-cuda     # NVIDIA
pip install mamba3-rl-rocm     # AMD ROCm
```

Every one of them provides the same two import modules, `mamba3_rl` and
[`mamba3_graph`](#graph-mamba-mamba3_graph); installing more than one into the same
environment is unsupported, since they share those names. `.github/workflows/publish.yml`
builds and publishes them from `tools/build_wheel.sh`.

To build from source instead:

```bash
pip install maturin
cd bindings/python
maturin develop --release              # CPU
```

The backend is a build-time choice, because a Python extension module is one
compiled artifact and its classes name one runtime:

```bash
maturin develop --release --no-default-features --features cuda    # NVIDIA
maturin develop --release --no-default-features --features hip     # AMD
maturin develop --release --no-default-features --features wgpu    # Vulkan/Metal/DX12
maturin develop --release --no-default-features --features msl     # Metal, via MSL
```

To build a wheel for another machine, use the script, which vendors the shared
libraries a CPU wheel links and can check the result:

```bash
tools/build_wheel.sh cpu    target/wheels-cpu    --smoke    # from the repository root
tools/build_wheel.sh wgpu   target/wheels-wgpu   --smoke
tools/build_wheel.sh vulkan target/wheels-vulkan             # wgpu, via SPIR-V
tools/build_wheel.sh msl    target/wheels-msl                # wgpu, via MSL (macOS)
tools/build_wheel.sh cuda   target/wheels-cuda               # NVIDIA; needs the CUDA toolkit
tools/build_wheel.sh hip    target/wheels-rocm               # AMD ROCm; needs the ROCm toolkit
```

The CPU runtime's code generator links `libzstd`, which on macOS resolves to
Homebrew's copy; a plain `maturin build` wheel therefore only works where that
same library is installed. The script builds CPU wheels with
`--auditwheel=repair`, which copies it (BSD-licensed, ~650 KB) into
`mamba3_rl.dylibs/` and relinks against it; the wgpu build links no such library.
`--smoke` installs the wheel into a fresh virtual environment, fails if the
extension links anything outside the system and the wheel itself, and imports it
with the dynamic-library search paths cleared. `tools/check.sh` runs it, and the
Python suite against the result.

`mamba3_rl.backend()` reports what the wheel actually got — for wgpu, with the
shader language: `wgpu<wgsl>`, `wgpu<msl>`, `wgpu<spirv>`. Weights, activations
and observations are `f32`; `set_matmul_precision("bf16")` rounds what the product
kernels *read* while keeping `f32` master weights, gradients and accumulation,
which needs no loss scaling.

---

## A run, end to end

```python
import mamba3_rl as m3

# A task that cannot be solved without memory: a symbol is shown once, at the
# first step of an episode, and the only reward comes from naming it at the last.
env = m3.RecallEnv(num_envs=32, symbols=4, horizon=4)
print(env.chance_return, env.optimal_return)     # 0.25 guessing, 1.0 perfect

policy = m3.Policy(m3.PolicyConfig(
    env.obs_dim, env.action_dim, d_model=64, n_layers=2,
    n_heads=4, head_dim=16, d_state=8, chunk_size=8, seed=7,
))

learner = m3.PpoLearner(policy, env, steps=16, learning_rate=1e-3, seed=5)
for stats in learner.run(rounds=80, epochs=4):
    if stats.round % 10 == 0:
        # `episode_return` is the mean over episodes *completed* in the window,
        # and None when none completed.
        print(f"{stats.round:>4}  return {stats.episode_return}  "
              f"entropy {stats.entropy:.3f}  kl {stats.approx_kl:.5f}")

print("greedy:", m3.evaluate(policy, m3.RecallEnv(32, 4, 4, seed=99), steps=16))
policy.save("recall.json")
```

`learner.policy` is the same weights, not a copy: training through the learner and
saving through the policy object refer to one stack.

### Starting from an expert instead

If the environment can say what an expert would do, cloning it reaches a competent
policy in a handful of updates — and then stops, because nothing in cross entropy
mentions reward. Running the two in order is the standard recipe, and the handover
costs nothing: it is the same policy object.

```python
cloner = m3.ImitationLearner(policy, env, steps=16,
                             schedule=m3.DaggerSchedule.exponential(0.7))
for stats in cloner.run(rounds=12):
    print(stats.round, f"beta={stats.beta:.2f}", f"agreement={stats.agreement:.3f}")

m3.PpoLearner(policy, env, steps=16, learning_rate=1e-3).run(rounds=40)
```

`beta` is the expert's share of the *acting*, and it decays: the states being
labelled drift from the expert's own distribution towards the ones the learner
actually reaches, which is what stops a cloned policy from being good only where
the expert already went.

---

## Your own environment

Anything with `num_envs`, `obs_dim` and `action_dim` and these two methods is an
environment here — duck typing, no base class to inherit:

```python
class Corridor:
    """Walk left or right; the reward is at one end, and which end was shown once."""

    num_envs, obs_dim, action_dim = 64, 3, 2

    def reset(self) -> np.ndarray: ...
        # [num_envs, obs_dim] float

    def step(self, actions: np.ndarray):
        # actions: [num_envs] int64
        return observation, reward, done      # [envs, obs_dim], [envs], [envs]

    def expert_actions(self) -> np.ndarray | None:
        # optional; only ImitationLearner asks
        ...

    def action_mask(self) -> np.ndarray | None:
        # optional; [num_envs, action_dim], 1/True where legal, None = all legal
        ...

    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...
        # optional, both or neither; what learner.save(path, level="full") needs
```

Two conventions decide whether a run is correct, and neither of them complains
when it is wrong:

* **Environments auto-reset.** Where `done[i]` is `1`, the observation returned
  beside it is already the first observation of environment `i`'s next episode.
  That is what lets a window of fixed length hold environments whose episodes end
  at different moments.
* **`done` marks the transition that ended an episode**, not the observation that
  begins one. The library derives the second from the first — shifting by one step
  and carrying the last flag of a window into the next — and cuts both the
  recurrence and the short causal convolution there, so nothing before a reset is
  visible after it.

`mamba3_rl.VecEnv` is that protocol, as a `typing.Protocol`, for type checkers and
for reading.

### Legal-action masks

`action_mask()` is asked once per step, *before* the action for the observation
the environment just returned is drawn. The mask is applied identically to the
draw, the recorded log-probability, the PPO replay, the entropy bonus, the
reference's score and imitation's cross entropy, and `Rollout.step(...,
action_mask=...)` and `evaluate()` honour it the same way.

* `None` means every action is legal **on that step**. A mask may come and go:
  steps without one are recorded as all-legal.
* A mask is checked on the host when it is returned: a value other than 0/1 or a
  row with no legal action raises `ValueError` before anything is drawn, and an
  exception from `action_mask()` itself propagates the same way. The environment
  is not stepped past a failed mask, and the next collection starts from `reset()`.
* For imitation, every expert label must be legal under the mask; a batch that
  breaks that is refused when it is built.
* Entropy is over the legal actions, so it drops when masking turns on. Do not
  compare `Stats.entropy` across that change.

`examples/custom_env.py` is a complete one in numpy.

### Compiled device games

An environment written in Python costs two copies and a device synchronisation a
step. A game written as device code costs nothing, and the learner can go further
and fold the action draw, the trajectory writes and the transition into one
kernel per step:

```python
world = m3.game("recall", 64, symbols=4, seed=0, masked=False)
learner = m3.PpoLearner(policy, world, steps=16)
learner.collection_path      # "fused"; PpoLearner(..., fused=False) steps it from the host
```

The fused window is byte-identical to the host-path window (`tests/test_game.py`),
and `m3.launch_count()` shows the dispatches it saves. A game's constants are
compiled into its kernel — `recall`'s horizon is 8 — so a parameter it cannot
honour raises `ValueError`, as does an unknown name, `fused=True` over a Python
environment, or an `ImitationLearner` over a game (a game has no expert). Games
save and load for `learner.save(path, level="full")`.

Adding one is Rust work, in this crate and these bindings:

1. Implement `mamba3::rl::GameLogic` (`reset`, `transition`, `legal`) for a unit
   struct and write its `GameSpec`, as `src/rl/games.rs` does for `Recall`.
   `GameWorld` then gives it `VecEnv`, the fused rollout and `save_state`.
2. In `bindings/python/src/game.rs`, add a `World` variant — the compiler lists
   every `match` that needs an arm — its name in `GAMES`, and its parameters in
   `build`.
3. Rebuild the wheel (`tools/build_wheel.sh`) and extend `tests/test_game.py`: the
   fused-against-host window comparison, masked if the game masks, and the
   launch-count comparison.

### What it costs

The built-in `RecallEnv` is a kernel: a rollout over it never touches the host, and
`read_count()` is flat however long it runs. An environment written in Python
cannot be that — its observations are host arrays — so each step copies them onto
the device and copies the actions back, and `read_count()` grows once per step.
That is the honest price of reaching an existing simulator, and for anything whose
own step takes more than a few microseconds it is noise. For a task written to be
trained *fast*, write it as a kernel instead, in Rust, beside `RecallEnv`.

---

## API

| | what it is |
|---|---|
| `PolicyConfig` | the architecture: `obs_dim`, `action_dim`, `d_model`, `n_layers`, and the mixer underneath |
| `Policy` | the weights, plus `save` / `load` / `freeze` / `fingerprint` / `export_numpy` |
| `ObsSpec`, `EntitySet` | read the flat observation as entity sets; `pack` / `pack_batch` / `unpack` |
| `EntityEncoderConfig`, `PoolingConfig`, `PointerHead` | the shared entity encoder, its pooling, and an actor that scores entities |
| `numpy_ref.Policy` | the recurrent step in numpy alone, from an `export_numpy` file |
| `Rollout` | the recurrent state of `num_envs` environments and the `O(1)` step that advances it |
| `RecallEnv` | a memory task with a known chance floor and ceiling, as a device kernel |
| `PpoConfig` | `gamma`, `gae_lambda`, `clip_coeff`, `value_coeff`, `entropy_coeff`, … |
| `PpoLearner` | `collect` / `update` / `round` / `run`, `policy`, `save` / `load_checkpoint` / `from_checkpoint` |
| `ImitationLearner` | behaviour cloning and DAgger, with `DaggerSchedule`; the same checkpoint methods |
| `LrSchedule` | the optimizer's learning-rate schedule (`lr_schedule=`), per optimizer step |
| `EmaConfig` | a moving average of the weights (`ema=`), kept on the device; `ema_policy`, `reset_ema()` |
| `Stats`, `CloneStats` | what one round reports |
| `evaluate(policy, env, steps)` | the greedy return, from a fresh state |
| `backend()`, `read_count()`, `synchronize()` | what the wheel got, and what it is doing |

### Observations that are sets of things

When an observation is a list of like things (tiles, units, cards), a flat
`Linear(obs_dim → d_model)` learns separate weights for every slot. An `ObsSpec`
tells the policy how to read the same flat vector instead:

```
[ globals | set_1: N_1 × (F_1 + 1) | set_2: ... ]     each entity: F features, then presence (1/0)
```

Each set then gets one MLP shared by all of its entities. The results are pooled
over the present entities (mean and/or max) and projected to `d_model`.
`PointerHead` scores the entities themselves, so action `i` is entity `i`:

```python
spec = m3.ObsSpec(globals=4, sets=[m3.EntitySet("tiles", count=100, features=60)])
cfg = m3.PolicyConfig(
    spec.obs_dim, 100, d_model=64, n_layers=2,
    obs_spec=spec,
    entity_encoders={"tiles": m3.EntityEncoderConfig(hidden=[64], d_entity=48)},
    pooling=m3.PoolingConfig(kinds=("mean", "max")),
    action_head=m3.PointerHead("tiles", hidden=48),   # extra_actions=K appends K flat actions
)
obs = spec.pack_batch(globals=g, tiles=(features, present))   # [num_envs, obs_dim]
```

Environments, buffers, learners and checkpoints are unchanged, because the wire
format is still flat. Empty slots never get an action (probability exactly 0) and
never influence the pools. Everything defaults off: `PolicyConfig(obs_dim,
action_dim)` is the flat policy, bit for bit.

To act without the trainer, `policy.export_numpy("p.npz")` writes the weights and
architecture, and `mamba3_rl.numpy_ref.Policy.load("p.npz")` runs the recurrent step
with numpy alone. `numpy_ref.py` has no compiled dependencies, so it can be copied
next to an agent.

### Reinforcement learning for the entity model

The entity-to-plan model (`EntityModelSpec`, `EntityDataset`, `EntityModel`)
trains from rewards through `EntityPolicy`: the model as a PPO actor with a
value head (`Linear(d, d) -> GELU -> Linear(d, 1)` over the presence-weighted
mean of the encoder output plus the embedded globals). Every pointer head and
every categorical head is an action, at every (query, step) cell it covers;
multilabel and regression heads are not actions (their logits are still
returned in `outputs` for the environment to use). Absent queries take no
action (`-1`).

```python
policy = m3.EntityPolicy(spec, learning_rate=3e-3)
# or: m3.EntityPolicy.from_model(model)  # a behaviour-cloned planner as the actor
out = policy.act(obs)   # {"actions": {head: int64 [B, M, K]}, "log_prob": [B, M, K],
                        #  "value": [B], "outputs": {head: logits}}; greedy=True for argmax
stats = policy.update(obs, out["actions"], out["log_prob"], out["value"],
                      reward, done, last_value, epochs=4, minibatches=2)
policy.save("p.m3ck"); loaded = m3.EntityPolicy.load("p.m3ck")
model = policy.to_model()   # the actor back as an EntityModel (predict / evaluate)
```

Rollouts are time-major: `obs` holds `S = T*E` samples (sample `t*E + e`);
`actions` maps every pointer and categorical head to int `[S, M, K]` (`-1` = no
action; `first`-step heads also accept `[S, M]`); `log_prob` is `[S, M, K]`;
`value` is `[S]` or `[T, E]`; `reward`/`done` are `[T, E]`; `last_value` is
`[E]`; `minibatches` must divide `S`. `act` is one upload and one
device-to-host read per call, `update` one read (the statistics) — both pinned
with `read_count()`. Stored plans are re-scored by teacher forcing, so
`StepCausal` (or no `autoregressive_on`) is required: `Joint` with
`autoregressive_on` and `QueryCausal` are refused, as is the composed
entity-model path. The PPO ratio is per cell, the advantage shared by a
sample's cells, the surrogate the mean over acted cells. The main README has
the full section; `examples/entity_ppo_bandit.py` is the runnable loop.

### Driving the policy yourself

`Rollout` is the policy without a learning loop around it — for serving a trained
policy, or for a loop you would rather write in Python:

```python
rollout = m3.Rollout(policy, num_envs=env.num_envs, temperature=0.0)
obs, done = env.reset(), None
for _ in range(steps):
    actions, values, log_probs = rollout.step(obs, reset=done)
    obs, reward, done = env.step(actions)
```

`reset` is the previous step's `done`: pass it, or every episode will be
remembered by the one after it. `action_mask=` restricts the draw exactly as a
learner's collection does, so a masked policy is evaluated on the distribution it
was trained on.

A step is one device synchronisation: its three arrays come back in a single
read (`evaluate`'s two likewise), and so is `RecallEnv.step` or a `Game`'s. On a
GPU that read's wait, not the step's kernels, is most of the cost; wgpu on Apple
silicon, 32 environments, a two-layer `d_model=64` policy: about 1.9 ms a policy
step and 1.4 ms an environment step, of which about 1.4 ms each is the read.

### Schedules: `lr_schedule` and DAgger's `schedule`

Two different clocks. `lr_schedule=LrSchedule.cosine(...)` (both learners) is the
optimizer's learning rate, advanced once per *optimizer step* — `epochs *
minibatches` of them per PPO round. `ImitationLearner(schedule=DaggerSchedule...)`
is the expert's share of the acting, advanced once per *round*.

### An anchor: the reference policy

`PpoLearner(..., reference=policy, ppo=PpoConfig(reference_coeff=c))` prices
drift from a frozen policy. The reference is a **copy taken at construction**:
passing the policy being trained is safe, and later training or reloading that
object does not move the anchor. It keeps its own recurrent history across
windows, cut at the same episode boundaries as the actor's, and `reset()` clears
both. `reference_coeff` without a `reference` is an error.

### A moving average of the weights

`PpoLearner(..., ema=m3.EmaConfig(0.99))` (and `ImitationLearner`) keeps an
exponential moving average of the policy's weights on the device. After every
optimizer step, `ema <- ema + (1 - decay) * (theta - ema)`; `warmup="tf"` uses
`min(decay, (1 + t) / (10 + t))` at optimizer step `t` instead, so early averages
are not dominated by the initial weights. The clock is the optimizer step, not the
round: a PPO round of `epochs * minibatches` steps moves the average that many
times, and the half-life is `ln 2 / -ln(decay)` steps (69 at 0.99, 138 at 0.995,
693 at 0.999).

`learner.ema_policy` is the average as a `Policy`: evaluate it, roll it out, save
it. It is a handle, not a snapshot — it keeps moving as the learner trains — and it
is never trained by the optimizer; passing it to another learner as the policy to
train is unsupported. Frozen parameters are not averaged: the average holds the
trained policy's. `learner.reset_ema()` restarts the average from the current
weights, for example when a critic-only warm-up ends (between rounds only, for a
`PpoLearner`). `ema_updates` counts the average's steps, `ema_config` returns the
configuration.

Attaching an average changes nothing about training: losses, statistics and
weights are the same to the bit. It costs one copy of the trainable weights on the
device (3.70 MiB for the 971,125 parameters of `d_model=256`, 4 layers over
`RecallEnv`) and one kernel launch per trainable parameter per optimizer step:
+0.10% of an `update()` on the CPU runtime and +0.17% on wgpu at that size
(`bench/ema_overhead.py`). The average is `f32`, like the weights;
`set_matmul_precision` rounds compute operands only and does not change it.

### Saving and resuming

Three different things, from weakest to strongest:

| | what it restores | use |
|---|---|---|
| `Policy.save` / `Policy.load` / `load_weights` | weights | a **warm start**: optimizer moments and counters begin again |
| `learner.save(path)` / `load_checkpoint` / `from_checkpoint` | weights, AdamW moments, `rounds`, optimizer step, and the training configuration | resuming the **optimizer and schedule** exactly |
| `learner.save(path, level="full")` | also the collector's observation, flags and episode accounting, every layer's recurrent state, the action-draw schedule, the reference's carried cache, and the environment's own `save_state()` bytes | continuing **the run** exactly, in any process |

A learner with a moving average saves its weights, counter and configuration at
either level; `load_checkpoint` restores it under the rules below, and
`from_checkpoint` rebuilds it. A `PpoLearner` with a reference saves the
reference's weights and architecture at either level. `from_checkpoint` rebuilds the reference from them when none is
passed (one that is passed must have the same weights), and
`load_checkpoint(config="checkpoint")` adopts them in place of a different
reference, or where the learner had none.

`load_checkpoint` is all or nothing — a load that raises changes nothing, the
environment included — and clears the last collected window. It compares the
saved configuration (base rate, `lr_schedule`, AdamW settings, `max_grad_norm`,
PPO or DAgger settings, architecture, reference-weights fingerprint, `EmaConfig`, and
for a full checkpoint the sampling seed and temperature) with the learner's:
`config="verify"` (the default) raises listing every difference, `"checkpoint"`
adopts the saved settings, and `"live"` keeps the learner's. For the moving
average, `"checkpoint"` adopts the saved one (configuration, weights, counter);
`"live"` re-seeds the learner's average from the loaded weights when the checkpoint
has none, marking the load `"warm"`, and ignores, with a note, an average the
learner does not keep. A full checkpoint
restores the rollout too, calling the environment's `load_state()` last, after
everything else has been checked; `level="optimizer"` ignores that part and
`level="full"` insists on it. `load_checkpoint(policy_file, strict=False)` is an
explicit warm start. Use `.m3ck` (binary) paths: smaller, bit-exact, and the only
format a full checkpoint can be written in.

A full save needs an environment with both `save_state()` and `load_state()`
(`NotImplementedError` otherwise) and is taken between rounds — after `update()`,
not between `collect()` and `update()`. Built-in `RecallEnv` supports it.

`learner.continuation` says how exactly the learner's history continues one run:
`"full"` for a fresh learner or one only ever restored from full checkpoints,
`"optimizer"` after a load that restored training but not the rollout, `"warm"`
after a warm start, a legacy checkpoint, or a load that kept different live
settings. It only moves down, and `save` records it. Checkpoints written before
levels existed read `{"exact": true}` as `"optimizer"`.

A full continuation reproduces the run exactly: every sampled action, reward,
mask, reference score, learning rate, counter, loss and final weight, bit for bit,
also when the restore happens in another process (`tests/test_continuation.py`).
On a GPU, pin the matrix-product kernel in both processes —
`m3.set_matmul_kernel("block_tiled")` or `MAMBA3_MATMUL_KERNEL=block_tiled` — because
the default, `"auto"`, picks the fastest kernel per shape by timing in each
process, and kernels agree only to a few ulp. Actions and rewards match either
way; losses and weights then match to the bit as well.

### What crosses the boundary

One round is one call. Inside it the rollout, the action draws, the trajectory
writes, the advantage estimate and every gradient step are queued device work that
the host never waits on — with two deliberate exceptions, both of them numbers a
human asked for:

* the end of `update()`, which reads every optimizer step's loss and gradient
  norm and the diagnostics it returns in **one** read, however many `epochs` and
  `minibatches` it took (it used to be two reads per step plus six — 54 for four
  epochs over four minibatches, about 23% of that update on wgpu,
  `examples/bench_update_reads.rs`);
* `episode_return()`, which reads the completed-episode mean and count together
  and returns `None` when no episode completed. `round()` folds it into the
  update's read, so a round on a device environment is one read in all.

`ImitationLearner.round()` is one read too: the agreement replay is queued behind
the optimizer step and its sums come back with the step's loss and gradient norm.
It used to be three reads — the step, then the replay's predictions and the labels
compared on the host — and taking them together is ~15-20% of a round on wgpu
(`examples/bench_imitation_round.rs`).

A moving average of the weights (`ema=`) adds none: it is seeded, updated after
every optimizer step and `reset_ema()`'d on the device. Its bytes cross only when a
checkpoint is saved (once per parameter), and `Policy.fingerprint()` on
`ema_policy` reads it like any policy.

Masking adds one more, only when a mask is present: the whole window's mask is
validated once when it becomes a batch (and, for imitation, the labels with it).
A mask that comes from Python is checked on the host before upload, which costs no
device read.

Everything else that looks like a number — `Stats.loss`, `CloneStats.agreement` —
is one of those reads, not another one.

---

## Graph Mamba: `mamba3_graph`

The wheel holds a second import module. `mamba3_graph` is
[Graph Mamba](../../README.md#graph-mamba) — a graph model built from the same
bidirectional Mamba-3 block — and it is the same extension library as
`mamba3_rl`: one device, one set of counters, one `LrSchedule` class. Nothing
else needs installing, and the two can be used in one process.

```python
import numpy as np
import mamba3_graph as mg

spec = mg.GraphMambaSpec(
    node_features=300,                      # floats per node; mg.Categorical([...]) for ids
    task=mg.NodeClassification(18),
    pe_dim=16,                              # width of the `pe` array below
    max_hops=4, walks=8, repeats=4,         # the paper's m, M, s; max_hops=0: node tokens only
    node_layers=2, mpnn="gine",
)
data = mg.GraphDataset(spec, dict(
    edge_index=edge_index,                  # int[2, E]
    x=x, y=y,                               # float[N, 300], int[N] (-1 = unlabelled)
    train_mask=train, val_mask=val, test_mask=test,
    pe=mg.rwse(edge_index, x.shape[0], 16),
))                                          # validated, reordered, uploaded once
model = mg.GraphMamba(spec, learning_rate=1e-3)

for epoch in range(300):
    model.train_epoch(data, epoch)          # every step of the epoch, queued
    losses = model.read_losses()            # one read: [{step, loss, grad_norm, learning_rate}]
    val = model.evaluate(data, split="val", metric="accuracy")     # one read

logits = model.predict(data)                # float32[N, 18], in your node order; one read
test = model.predict(data, split="test")    # only the rows of the test mask, same order
model.save("graph.m3ck")
```

| | what it is |
|---|---|
| `GraphMambaSpec` | the model, completely; validates itself, `to_json` / `from_json` |
| `NodeClassification`, `GraphClassification`, `GraphRegression`, `GraphMultiLabel` | the task: what is predicted, how graphs are pooled, which loss |
| `Categorical(vocab)` | integer-id features (atom and bond types) in place of floats |
| `GraphDataset(spec, arrays)` | one graph, or many with `graph_ptr`, on the device |
| `rwse`, `laplacian_pe` | structural and positional encodings to pass as `pe`, computed once on the host |
| `GraphMamba` | `train_epoch` / `read_losses` / `evaluate` / `predict` / `memory_estimate` / `save` / `load` |
| `upload_count()`, `read_count()`, `build_info()` | what crossed the boundary, and how the wheel was built |

A dataset is one large graph (trained on `parts` node partitions per epoch;
`parts=1` is full batch) or many graphs (trained in batches filled to
`batch_rows` nodes). With neither argument both are sized from the device's
memory. `examples/graph_node_classification.py` runs the first on a
heterophilic-benchmark `.npz`.

**What crosses the boundary.** The rules are the ones the measurements in
[`bench/results/python_boundary.md`](../../bench/results/python_boundary.md)
asked for, and `tests/test_graph.py` pins each of them:

* *Arrays go in as NumPy holds them.* `float16` / `float32` / `float64`, any
  integer width, C-ordered, Fortran-ordered or sliced; none is modified. The pass
  that puts the graph in its canonical order also converts, so the dtype costs
  nothing. A Fortran-ordered, sliced or unaligned array is copied once by
  NumPy into C order first.
  Building a dataset reads nothing back (`read_count() == 0`).
* *A training epoch is one call that reads nothing* and uploads one small table
  (`upload_count() == 1`); walk sampling, token construction, batch layout and the
  loss are device kernels. `read_losses()` is one read for every step queued
  since the last one, `predict` and `evaluate` one each.
* *The interpreter lock is released* for the whole of `train_epoch`,
  `read_losses`, `evaluate`, `predict` and the upload, so other Python threads
  run (94% of their idle rate, measured). The classes are bound to the thread
  that made them: using a model from another thread raises.
* *Ctrl-C works.* `train_epoch` checks for signals between steps and raises
  `KeyboardInterrupt` after a completed step; the model is intact, its step
  counter matches the losses `read_losses()` returns, and it can go on training.
* *Errors name the argument or the array key* (`ValueError`); a non-finite loss
  raises `FloatingPointError` from `read_losses()`.
* *A debug build says so*: `build_info()["profile"]`, and a `RuntimeWarning` when
  a model is built.

`dtype="bf16"` on both the dataset and the model stores weights, optimizer
moments and activations in 16 bits where the backend has the type
(`mg.supports_dtype("bf16")`: the CPU runtime and CUDA; WGSL does not); pass a
`loss_scale` with it (256 is what the tests use). It is a memory setting: over
50 steps its loss stays within 3% of the `f32` run's. `dtype="f16"` raises
`NotImplementedError`: the Mamba-3 mixer's backward pass returns NaN gradients
in f16 in this build, with or without the graph model around it.

## MS2-to-substructure: `mamba3_ms2`

The wheel holds a third import module. `mamba3_ms2` is the MS2 model — a
spectrum encoder, a formula head over a resident formula table and a
graph-action decoder — and it is the same extension library as `mamba3_rl`:
one device and one set of counters. Nothing else needs installing.

```python
import mamba3_ms2 as ms2

table = ms2.FormulaTable.from_json("formula_table.json")
model = ms2.Ms2Model(ms2.ModelConfig(), table, seed=0)
batch = ms2.SpectrumBatch(
    n_raw=64,
    spectrum_id=spectrum_id,              # uint64[B]
    raw_peak_count=raw, peak_count=count,  # uint32[B]
    peak_id=peak_id,                       # uint32[B, n_raw]
    mz_udalton=mz,                         # uint32[B, n_raw], 1e-6 Da units
    intensity=intensity,                   # float32[B, n_raw]
    mz_uncertainty_udalton=..., precursor_mz_udalton=...,
    precursor_uncertainty_udalton=..., adduct=..., polarity=...,
    collision_energy_ev=..., collision_energy_known=...,
    energy_count=..., fragment_tolerance_ppm_tenths=...,
    precursor_tolerance_ppm_tenths=..., instrument_class=...,
)
out = model.generate(batch, ms2.GenerationConfig(trajectories=8, seed=1))
out.validate()                            # every invariant of the contracts
print(out.actions.shape, out.actions.dtype)   # (B*K, T, 4), uint32
print(out.distinct_traces())              # finished, valid, non-duplicate rows
```

| | what it is |
|---|---|
| `SpectrumBatch` | `B` spectra, row-major `[B, n_raw]` per-peak fields; keyword arguments named exactly as the contract's fields, `validate()`, `to_json` / `from_json` |
| `ModelConfig`, `GenerationConfig`, `ChemistryDomain`, `TrainConfig` | constructible with keyword arguments defaulting to the Rust defaults (`ModelConfig.v0()`), attribute access, `validate()`, `to_json` / `from_json` |
| `FormulaTable` | `from_json(path_or_text)`: the V0 formula table, uploaded once and reused |
| `Ms2Model` | `Ms2Model(config, table, seed)`, `generate(batch, config) -> CandidateBatch` |
| `CandidateBatch` | every field of contracts §3.5 as a NumPy array with the contract's dtype and shape, `validate()`, `to_json`, `distinct_traces()` |
| `ExperimentSet` | `from_export(path, table)`: spectra, `labeled_count`, `take_labeled(n)` |
| `Ms2Trainer` | `Ms2Trainer(model_config, train_config, table, seed)`, `step(set, indices)`, `request_report()`, `teacher_eval(set, indices)`, `save` / `load` |
| `request_status_names`, `candidate_status_names` | names of the set status bits |

Training runs through the trainer, one optimizer step per call:

```python
trainer = ms2.Ms2Trainer(ms2.ModelConfig(), ms2.TrainConfig(batch=8, seed=1), table)
data = ms2.ExperimentSet.from_export("train.json", table).take_labeled(128)
indices = list(range(128))
for _ in range(steps):
    trainer.request_report()          # the next step reads its losses back
    report = trainer.step(data, indices)   # None without a requested report
eval_out = trainer.teacher_eval(data, indices)  # dict of arrays, one read
trainer.save("ms2.m3ck")
```

**What crosses the boundary.** The graph module's rules apply here too:

* *Arrays go in as NumPy holds them* — C-ordered, Fortran-ordered or
  non-contiguous (copied once to contiguous buffers, never modified).
  Integer fields take any integer width, float fields any float width; a
  dtype of the wrong kind is a `TypeError` naming the field, a wrong shape
  or out-of-range value a `ValueError` naming it.
* *`generate`, `step` and `teacher_eval` each run their whole pipeline in
  Rust* with the interpreter lock released (no Python object is borrowed
  across the device wait). A training step without a requested report reads
  nothing; a reported step and an evaluation read once each.
* *Errors use the same mapping* as the rest of the bindings
  (`src/err.rs`) with the Rust message unchanged: `Error::Config` and
  `Error::Json` are `ValueError`, `Error::Io` is `OSError`,
  `Error::Unsupported` is `NotImplementedError`, the rest `RuntimeError`.
* *FP32 only*; the backend is the one the wheel was built for.

`Ms2Model` has no `save` / `load`: the crate checkpoints the trainer
(weights plus the configs and table reference that bind them), so
checkpoints go through `Ms2Trainer.save` / `Ms2Trainer.load(path, table)`.
There is no packed, resident or beam output yet: `generate` returns the
uncompacted `B * K` records and `mode="beam"` is refused until P5 builds
it.

---

## What is not bound

`mamba3_rl` is the reinforcement learning stack and the policy it needs, and
`mamba3_graph` the graph model; neither is the whole crate. Three things are
deliberately on the other side of the line:

* **The rest of the model zoo** — language modelling, vision, hybrids, LoRA and
  quantization. They are Rust APIs with Rust examples, and binding them would be a
  second, much larger surface with nothing reinforcement-specific about it.
* **`MultiSyncCollector`**, which runs environments on worker threads. A Python
  environment cannot be driven from a thread that does not hold the interpreter
  lock, so the pool is only worth having for environments written in Rust — where
  it already is.
* **Tensors.** Nothing here hands out a device buffer, because a half-bound tensor
  type is worse than none: it invites a loop that reads one value per step and
  gives back everything the design is for. Observations come in as numpy arrays and
  scalars come out; what happens in between stays on the device.

## Development

```bash
maturin develop                                      # debug build, into the active venv
pytest                                               # the test suite
python examples/train_recall.py                      # PPO, then imitation, side by side
python examples/custom_env.py                        # an environment written in numpy
python examples/ppo_anchored_masked.py --smoke       # reference + masks + lr_schedule + resume
python examples/imitation_schedules.py --smoke       # DAgger schedule and lr_schedule together
python examples/graph_node_classification.py g.npz   # Graph Mamba on a heterophilic-benchmark file
python examples/bench_boundary.py                    # what crossing into Rust costs: ingest, reads, GIL
```

Use `maturin develop --release` for anything that is timed or trained: the debug
build is many times slower, and `mamba3_graph` warns when a model is built on one.

The tests run against whichever backend the module was built with, and they are
sized to finish on a CPU. Run them on every backend you ship: a kernel that does
not compile on one shader language (WGSL has no infinity literal, for one) works
everywhere else. Such a kernel raises `RuntimeError`, naming it, at the next value
read back — it used to compute zeros silently — but only a run on that backend
reaches it. `tools/check.sh --wgpu` runs both suites on both backends.

---

## License

MIT OR Apache-2.0, as the crate it binds.
