"""Reinforcement learning on Mamba-3 state space models, on the GPU.

A Mamba-3 policy suits reinforcement learning for one structural reason: acting
and learning want opposite shapes, and a state space model is cheap at both.
Acting wants the smallest possible step -- one observation, ``num_envs``
environments, latency that does not grow with the episode. Learning wants the
largest possible batch -- a whole ``[envs, steps]`` window through one wide
parallel pass. A transformer can only do the first by carrying a cache that grows
with every action it takes; here the recurrent state is a fixed-size buffer
overwritten in place, and the window replays through a scan whose depth is
logarithmic in its length.

Both halves honour the same episode mask, so a policy gradient taken from the
replay is a gradient of what the rollout actually did.

A run, end to end::

    import mamba3_rl as m3

    env = m3.RecallEnv(num_envs=32, symbols=4, horizon=4)
    policy = m3.Policy(m3.PolicyConfig(env.obs_dim, env.action_dim, d_model=64, n_layers=2))
    learner = m3.PpoLearner(policy, env, steps=16, learning_rate=1e-3)

    for stats in learner.run(rounds=80, epochs=4):
        print(stats.round, stats.episode_return)

    print("greedy return:", m3.evaluate(policy, env, steps=16))
    policy.save("recall.json")

Your own environment goes in the same place ``RecallEnv`` does: any object with
``num_envs``, ``obs_dim`` and ``action_dim`` and the two methods
:class:`~mamba3_rl.VecEnv` describes.
"""

from ._mamba3_rl import (
    CloneStats,
    DaggerSchedule,
    Game,
    ImitationLearner,
    LrSchedule,
    Policy,
    PolicyConfig,
    PpoConfig,
    PpoLearner,
    RecallEnv,
    Rollout,
    Stats,
    __version__,
    backend,
    evaluate,
    game,
    launch_count,
    matmul_kernel,
    matmul_precision,
    read_count,
    reset_launch_count,
    reset_read_count,
    set_matmul_kernel,
    set_matmul_precision,
    synchronize,
)
from .protocol import VecEnv

__all__ = [
    "CloneStats",
    "DaggerSchedule",
    "Game",
    "ImitationLearner",
    "LrSchedule",
    "Policy",
    "PolicyConfig",
    "PpoConfig",
    "PpoLearner",
    "RecallEnv",
    "Rollout",
    "Stats",
    "VecEnv",
    "__version__",
    "backend",
    "evaluate",
    "game",
    "launch_count",
    "matmul_kernel",
    "matmul_precision",
    "read_count",
    "reset_launch_count",
    "reset_read_count",
    "set_matmul_kernel",
    "set_matmul_precision",
    "synchronize",
]
