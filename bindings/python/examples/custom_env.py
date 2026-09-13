"""An environment written in numpy, trained by the same loops.

Nothing about the learners is specific to the built-in task: an environment is any
object with ``num_envs``, ``obs_dim`` and ``action_dim`` that can ``reset`` and
``step``. This one is a light switch with a delay -- at the first step of an
episode a lamp shows one of ``ACTIONS`` colours, and the agent is asked to name it
``HORIZON`` steps later, when the lamp is dark. It is deliberately the same *shape*
of problem as the built-in ``RecallEnv``, so the two can be compared: the built-in
one is a kernel and never touches the host, and this one pays two copies per step.

    python examples/custom_env.py
"""

import numpy as np

import mamba3_rl as m3

ENVS = 32
ACTIONS = 3
HORIZON = 5
WINDOW = HORIZON * 3


class LampRecall:
    """Name the colour the lamp showed at the start of the episode.

    The observation is a one-hot colour -- all zeros once the lamp goes dark --
    followed by the clock. The reward is ``1`` for naming the colour on the last
    step of the episode and ``0`` everywhere else, so a memoryless policy can do no
    better than ``1 / ACTIONS``.
    """

    def __init__(self, num_envs=ENVS, action_dim=ACTIONS, horizon=HORIZON, seed=0):
        self.num_envs = num_envs
        self.action_dim = action_dim
        # One channel per colour, plus the clock and a lamp-lit flag.
        self.obs_dim = action_dim + 2
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.colour = np.zeros(num_envs, dtype=np.int64)
        self.clock = np.zeros(num_envs, dtype=np.int64)

    def _observation(self):
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        lit = self.clock == 0
        obs[lit, self.colour[lit]] = 1.0
        obs[:, self.action_dim] = self.clock / self.horizon
        obs[:, self.action_dim + 1] = lit
        return obs

    def reset(self):
        self.colour = self.rng.integers(0, self.action_dim, self.num_envs)
        self.clock[:] = 0
        return self._observation()

    def step(self, actions):
        terminal = self.clock + 1 == self.horizon
        reward = np.where(terminal & (actions == self.colour), 1.0, 0.0).astype(np.float32)
        done = terminal.astype(np.float32)

        # Auto-reset: a finished environment draws a new colour and restarts its
        # clock, so the rollout stays rectangular however the episodes fall, and
        # the observation returned beside `done` is already the next episode's.
        fresh = self.rng.integers(0, self.action_dim, self.num_envs)
        self.colour = np.where(terminal, fresh, self.colour)
        self.clock = np.where(terminal, 0, self.clock + 1)
        return self._observation(), reward, done

    def expert_actions(self):
        """The optimal action is always the colour, which this environment knows."""
        return self.colour.copy()

    @property
    def chance_return(self):
        return 1.0 / self.action_dim


def main():
    env = LampRecall(seed=4)
    print(f"backend: {m3.backend()}")
    print(f"guessing earns {env.chance_return:.2f} per episode, a perfect memory 1.00\n")

    policy = m3.Policy(
        m3.PolicyConfig(
            env.obs_dim, env.action_dim,
            d_model=64, n_layers=2, n_heads=4, head_dim=16, d_state=8,
            chunk_size=8, seed=7,
        )
    )

    # Cloning first: the environment knows its own optimal action, so it can label
    # any state the learner wanders into.
    print("== cloning the expert ==")
    cloner = m3.ImitationLearner(
        policy, env, steps=WINDOW, learning_rate=3e-3,
        schedule=m3.DaggerSchedule.exponential(0.7), seed=3,
    )
    for stats in cloner.run(rounds=12):
        print(f"{stats.round:>4}  beta {stats.beta:.2f}  loss {stats.loss:.4f}  "
              f"agreement {stats.agreement:.3f}")

    # Then the reward itself, from where cloning left the policy.
    print("\n== PPO from there ==")
    learner = m3.PpoLearner(policy, env, steps=WINDOW, learning_rate=1e-3, seed=5)
    for stats in learner.run(rounds=20, epochs=4):
        if stats.round % 5 == 0:
            print(f"{stats.round:>4}  return {stats.episode_return:.3f}  "
                  f"entropy {stats.entropy:.3f}  kl {stats.approx_kl:.5f}")

    greedy = m3.evaluate(policy, LampRecall(seed=99), steps=WINDOW)
    print(f"\ngreedy return: {greedy:.3f}  (chance {env.chance_return:.3f})")

    # Reading a host array per step is what an environment in Python costs, and the
    # counter says so. The built-in RecallEnv, being a kernel, leaves it flat.
    m3.reset_read_count()
    learner.collect()
    print(f"host reads for one {WINDOW}-step window: {m3.read_count()}")


if __name__ == "__main__":
    main()
