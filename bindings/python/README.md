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
tools/build_wheel.sh cpu  target/wheels-cpu  --smoke    # from the repository root
tools/build_wheel.sh wgpu target/wheels-wgpu --smoke
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
| `Policy` | the weights, plus `save` / `load` / `freeze` / `fingerprint` |
| `Rollout` | the recurrent state of `num_envs` environments and the `O(1)` step that advances it |
| `RecallEnv` | a memory task with a known chance floor and ceiling, as a device kernel |
| `PpoConfig` | `gamma`, `gae_lambda`, `clip_coeff`, `value_coeff`, `entropy_coeff`, … |
| `PpoLearner` | `collect` / `update` / `round` / `run`, `policy`, `save` / `load_checkpoint` / `from_checkpoint` |
| `ImitationLearner` | behaviour cloning and DAgger, with `DaggerSchedule`; the same checkpoint methods |
| `LrSchedule` | the optimizer's learning-rate schedule (`lr_schedule=`), per optimizer step |
| `Stats`, `CloneStats` | what one round reports |
| `evaluate(policy, env, steps)` | the greedy return, from a fresh state |
| `backend()`, `read_count()`, `synchronize()` | what the wheel got, and what it is doing |

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

### Saving and resuming

Three different things, from weakest to strongest:

| | what it restores | use |
|---|---|---|
| `Policy.save` / `Policy.load` / `load_weights` | weights | a **warm start**: optimizer moments and counters begin again |
| `learner.save(path)` / `load_checkpoint` / `from_checkpoint` | weights, AdamW moments, `rounds`, optimizer step, and the training configuration | resuming the **optimizer and schedule** exactly |
| `learner.save(path, level="full")` | also the collector's observation, flags and episode accounting, every layer's recurrent state, the action-draw schedule, the reference's carried cache, and the environment's own `save_state()` bytes | continuing **the run** exactly, in any process |

A `PpoLearner` with a reference saves the reference's weights and architecture at
either level. `from_checkpoint` rebuilds the reference from them when none is
passed (one that is passed must have the same weights), and
`load_checkpoint(config="checkpoint")` adopts them in place of a different
reference, or where the learner had none.

`load_checkpoint` is all or nothing — a load that raises changes nothing, the
environment included — and clears the last collected window. It compares the
saved configuration (base rate, `lr_schedule`, AdamW settings, `max_grad_norm`,
PPO or DAgger settings, architecture, reference-weights fingerprint, and for a full
checkpoint the sampling seed and temperature) with the learner's:
`config="verify"` (the default) raises listing every difference, `"checkpoint"`
adopts the saved settings, and `"live"` keeps the learner's. A full checkpoint
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

* the end of `update()`, which reads the five diagnostics it returns;
* `episode_return()`, which reads the completed-episode mean and count together
  and returns `None` when no episode completed.

Masking adds one more, only when a mask is present: the whole window's mask is
validated once when it becomes a batch (and, for imitation, the labels with it).
A mask that comes from Python is checked on the host before upload, which costs no
device read.

Everything else that looks like a number — `Stats.loss`, `CloneStats.agreement` —
is one of those reads, not another one.

---

## What is not bound

The module is the reinforcement learning stack and the policy it needs, not the
whole crate. Three things are deliberately on the other side of the line:

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
```

The tests run against whichever backend the module was built with, and they are
sized to finish on a CPU. Run them on every backend you ship: a kernel that does
not compile on one shader language (WGSL has no infinity literal, for one) works
everywhere else. Such a kernel raises `RuntimeError`, naming it, at the next value
read back — it used to compute zeros silently — but only a run on that backend
reaches it. `tools/check.sh --wgpu` runs both suites on both backends.

---

## License

MIT OR Apache-2.0, as the crate it binds.
