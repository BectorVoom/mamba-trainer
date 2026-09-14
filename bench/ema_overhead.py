"""What a moving average of the weights (T1, `ema=`) costs a PPO update.

Two learners of the same seed at the exp010 shape (208 environments, 120 steps,
d_model=256, 4 layers, 2 epochs, 1 minibatch), one with `ema=EmaConfig(0.99)`,
alternate rounds so drift in the machine's load hits both alike. Each `update()`
is timed to a synchronisation; the report is the median over the timed rounds.

Run against an installed wheel (never the Kaggriculture venv):

    python bench/ema_overhead.py                   # 20 timed rounds, 2 warm-up
    ROUNDS=5 python bench/ema_overhead.py          # fewer, e.g. on the CPU runtime

The matmul kernel is pinned (`KERNEL`, default `row_tiled`) so both learners run
the same kernels. The added device bytes are not measured here — Python has no
allocator probe — but computed from the trainable parameter count, which
`tests/train_ema_footprint.rs` shows is what attaching an average holds, up to
each buffer's alignment.
"""

import os
import statistics
import time

import mamba3_rl as m3

ENVS = int(os.environ.get("ENVS", 208))
STEPS = int(os.environ.get("STEPS", 120))
D_MODEL = int(os.environ.get("D_MODEL", 256))
LAYERS = int(os.environ.get("LAYERS", 4))
EPOCHS = int(os.environ.get("EPOCHS", 2))
ROUNDS = int(os.environ.get("ROUNDS", 20))
WARMUP = int(os.environ.get("WARMUP", 2))
KERNEL = os.environ.get("KERNEL", "row_tiled")


def build(ema):
    env = m3.RecallEnv(num_envs=ENVS, symbols=4, horizon=8, seed=1)
    policy = m3.Policy(m3.PolicyConfig(env.obs_dim, env.action_dim, d_model=D_MODEL,
                                       n_layers=LAYERS, seed=7))
    return m3.PpoLearner(policy, env, steps=STEPS, learning_rate=3e-4, seed=3, ema=ema)


def timed_update(learner):
    learner.collect()
    m3.synchronize()
    start = time.perf_counter()
    learner.update(epochs=EPOCHS)
    m3.synchronize()
    return time.perf_counter() - start


def launches_per_step(learner):
    learner.collect()
    m3.synchronize()
    m3.reset_launch_count()
    learner.update(epochs=1)
    m3.synchronize()
    return m3.launch_count()


def main():
    m3.set_matmul_kernel(KERNEL)
    plain, averaged = build(None), build(m3.EmaConfig(0.99))
    for _ in range(WARMUP):
        timed_update(plain)
        timed_update(averaged)
    times = {"off": [], "on": []}
    for _ in range(ROUNDS):
        times["off"].append(timed_update(plain))
        times["on"].append(timed_update(averaged))
    launches = {"off": launches_per_step(plain), "on": launches_per_step(averaged)}

    policy = averaged.policy
    off, on = statistics.median(times["off"]), statistics.median(times["on"])
    print(f"backend {m3.backend()}, matmul kernel {m3.matmul_kernel()}")
    print(f"shape: {ENVS} envs x {STEPS} steps, d_model={D_MODEL} x {LAYERS} layers, "
          f"{EPOCHS} epochs, 1 minibatch; {policy.num_parameters:,} parameters "
          f"({policy.num_trainable_parameters:,} trainable)")
    print(f"median update(): EMA off {off:.3f} s, on {on:.3f} s, "
          f"{100 * (on - off) / off:+.2f}% over {ROUNDS} rounds "
          f"(off min/max {min(times['off']):.3f}/{max(times['off']):.3f}, "
          f"on {min(times['on']):.3f}/{max(times['on']):.3f})")
    print(f"launches per optimizer step: off {launches['off']}, on {launches['on']} "
          f"(+{launches['on'] - launches['off']})")
    added = policy.num_trainable_parameters * 4
    print(f"added device bytes (computed): {added:,} ({added / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
