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

`mamba3_rl.backend()` reports what the wheel actually got. Weights, activations
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
        print(f"{stats.round:>4}  return {stats.episode_return:.3f}  "
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

`examples/custom_env.py` is a complete one in numpy.

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
| `Policy` | the weights, plus `save` / `load` / `freeze` |
| `Rollout` | the recurrent state of `num_envs` environments and the `O(1)` step that advances it |
| `RecallEnv` | a memory task with a known chance floor and ceiling, as a device kernel |
| `PpoConfig` | `gamma`, `gae_lambda`, `clip_coeff`, `value_coeff`, `entropy_coeff`, … |
| `PpoLearner` | `collect` / `update` / `round` / `run`, and `policy` |
| `ImitationLearner` | behaviour cloning and DAgger, with `DaggerSchedule` |
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
remembered by the one after it.

### What crosses the boundary

One round is one call. Inside it the rollout, the action draws, the trajectory
writes, the advantage estimate and every gradient step are queued device work that
the host never waits on — with two deliberate exceptions, both of them numbers a
human asked for:

* the end of `update()`, which reads the five diagnostics it returns;
* `episode_return()`, which reads one scalar.

Everything else that looks like a number — `Stats.loss`, `CloneStats.agreement` —
is one of those two reads, not another one.

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
```

The tests run against whichever backend the module was built with, and they are
sized to finish on a CPU.

---

## License

MIT OR Apache-2.0, as the crate it binds.
