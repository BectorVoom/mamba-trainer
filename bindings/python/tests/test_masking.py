"""A1: an environment can declare which actions are legal per step.

The environments here are sized to match the shared `policy`/`env` fixtures
(``obs_dim=6``, ``action_dim=4``, from `conftest.py`'s `SYMBOLS=4`), so they
can be swapped in wherever those are used without a bespoke policy.
"""

import numpy as np
import pytest

import mamba3_rl as m3

NUM_ENVS = 8
ACTION_DIM = 4
OBS_DIM = ACTION_DIM + 2
HORIZON = 4


class _BaseCountingEnv:
    """A target symbol per environment; naming it earns the reward. Not memory
    dependent -- this is a test of the masking boundary, not of the recurrence."""

    def __init__(self, num_envs=NUM_ENVS, action_dim=ACTION_DIM, horizon=HORIZON, seed=0):
        self.num_envs = num_envs
        self.obs_dim = action_dim + 2
        self.action_dim = action_dim
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.targets = np.zeros(num_envs, dtype=np.int64)
        self.clock = np.zeros(num_envs, dtype=np.int64)

    def _observation(self):
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        obs[np.arange(self.num_envs), self.targets] = 1.0
        obs[:, -1] = self.clock / self.horizon
        return obs

    def reset(self):
        self.targets = self.rng.integers(0, self.action_dim, self.num_envs)
        self.clock[:] = 0
        return self._observation()

    def step(self, actions):
        reward = (actions == self.targets).astype(np.float32)
        self.clock += 1
        done = (self.clock >= self.horizon).astype(np.float32)
        ending = done.astype(bool)
        self.targets[ending] = self.rng.integers(0, self.action_dim, int(ending.sum()))
        self.clock[ending] = 0
        return self._observation(), reward, done

    def expert_actions(self):
        return self.targets.copy()


class HalfMaskedCountingEnv(_BaseCountingEnv):
    """Only even-numbered actions are ever legal. The target may be either, so
    this is for PPO -- which does not need its own label to be legal -- and
    not for imitation."""

    def action_mask(self):
        mask = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        mask[:, 0::2] = 1.0
        return mask


class MaskedCountingEnvWithLegalExpert(_BaseCountingEnv):
    """Like above, but the target is snapped to a legal action, so the expert
    it hands imitation learning never names an illegal one."""

    def _snap(self):
        self.targets = (self.targets // 2) * 2

    def reset(self):
        super().reset()
        self._snap()
        return self._observation()

    def step(self, actions):
        super().step(actions)
        self._snap()
        return self._observation(), (actions == self.targets).astype(np.float32), np.zeros(
            self.num_envs, dtype=np.float32
        )

    def action_mask(self):
        mask = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        mask[:, 0::2] = 1.0
        return mask


class AlwaysIllegalExpertEnv(_BaseCountingEnv):
    """The mask never permits the target it labels the expert with -- every
    window this drives should be refused."""

    def action_mask(self):
        mask = np.ones((self.num_envs, self.action_dim), dtype=np.float32)
        mask[np.arange(self.num_envs), self.targets] = 0.0
        return mask


class EmptyLegalSetEnv(_BaseCountingEnv):
    """No action is ever legal -- an invalid environment contract."""

    def action_mask(self):
        return np.zeros((self.num_envs, self.action_dim), dtype=np.float32)


@pytest.fixture
def masked_env():
    return HalfMaskedCountingEnv()


def test_ppo_trains_under_masking_without_error(policy, masked_env):
    learner = m3.PpoLearner(policy, masked_env, steps=8, learning_rate=1e-3, seed=5)
    stats = learner.round(epochs=2)
    assert np.isfinite([stats.loss, stats.entropy, stats.approx_kl]).all()
    # First-epoch replay must reproduce the masked draw: no clipping and ~0 KL
    # the very first time a freshly collected window is scored.
    assert stats.approx_kl < 1e-3


def test_a_plain_environment_without_action_mask_is_unaffected(policy, env):
    # `env` (the shared fixture) implements no `action_mask` at all; masking
    # must stay fully absent rather than silently switching on.
    learner = m3.PpoLearner(policy, env, steps=8, learning_rate=1e-3, seed=5)
    stats = learner.round(epochs=2)
    assert np.isfinite(stats.loss)


def test_an_empty_legal_set_is_refused(policy):
    env = EmptyLegalSetEnv()
    learner = m3.PpoLearner(policy, env, steps=4, seed=5)
    with pytest.raises(ValueError, match="no legal action"):
        learner.collect()


def test_imitation_trains_under_masking_with_a_legal_expert(policy):
    env = MaskedCountingEnvWithLegalExpert()
    cloner = m3.ImitationLearner(policy, env, steps=8, seed=3)
    stats = cloner.round()
    assert np.isfinite(stats.loss)
    assert 0.0 <= stats.agreement <= 1.0


def test_an_expert_label_naming_an_illegal_action_is_refused(policy):
    env = AlwaysIllegalExpertEnv()
    cloner = m3.ImitationLearner(policy, env, steps=4, seed=3)
    with pytest.raises(ValueError, match="illegal"):
        cloner.round()
