"""Shared fixtures.

Everything here is sized to run on a CPU in a few seconds: four symbols, a
four-step horizon, two layers of 32 channels. The point of the suite is that the
plumbing is right, not that the policy is good — except where a test says so in
its name, and those are marked ``slow``.
"""

import numpy as np
import pytest

import mamba3_rl as m3

ENVS = 8
SYMBOLS = 4
HORIZON = 4
WINDOW = HORIZON * 2


@pytest.fixture
def env():
    return m3.RecallEnv(num_envs=ENVS, symbols=SYMBOLS, horizon=HORIZON, seed=1)


def config_for(env, **overrides):
    """A small architecture sized to whatever environment it will be driven with."""
    settings = dict(
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        d_state=8,
        chunk_size=4,
        seed=7,
    )
    settings.update(overrides)
    return m3.PolicyConfig(env.obs_dim, env.action_dim, **settings)


@pytest.fixture
def config(env):
    return config_for(env)


@pytest.fixture
def policy(config):
    return m3.Policy(config)


@pytest.fixture
def policy_for():
    """Build a policy for an environment other than the `env` fixture's."""

    def build(env, **overrides):
        return m3.Policy(config_for(env, **overrides))

    return build


class CountingEnv:
    """A vectorised environment written in numpy, with a known optimal action.

    Each environment holds a target symbol drawn at reset. The reward is ``1`` for
    naming it and the episode ends after ``horizon`` steps, so it is solvable
    without memory — which is what makes it a test of the *boundary* rather than of
    the recurrence.
    """

    def __init__(self, num_envs=ENVS, action_dim=SYMBOLS, horizon=HORIZON, seed=0):
        self.num_envs = num_envs
        self.obs_dim = action_dim + 1
        self.action_dim = action_dim
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.targets = np.zeros(num_envs, dtype=np.int64)
        self.clock = np.zeros(num_envs, dtype=np.int64)
        self.steps_taken = 0
        self.resets = 0

    def _observation(self):
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        obs[np.arange(self.num_envs), self.targets] = 1.0
        obs[:, -1] = self.clock / self.horizon
        return obs

    def reset(self):
        self.resets += 1
        self.targets = self.rng.integers(0, self.action_dim, self.num_envs)
        self.clock[:] = 0
        return self._observation()

    def step(self, actions):
        assert actions.dtype == np.int64
        assert actions.shape == (self.num_envs,)
        self.steps_taken += 1
        reward = (actions == self.targets).astype(np.float32)
        self.clock += 1
        done = (self.clock >= self.horizon).astype(np.float32)
        ending = done.astype(bool)
        # Auto-reset: a finished environment draws a new target and restarts, and
        # the observation returned beside its `done` is already the new episode's.
        self.targets[ending] = self.rng.integers(0, self.action_dim, int(ending.sum()))
        self.clock[ending] = 0
        return self._observation(), reward, done

    def expert_actions(self):
        return self.targets.copy()


@pytest.fixture
def numpy_env():
    return CountingEnv()
