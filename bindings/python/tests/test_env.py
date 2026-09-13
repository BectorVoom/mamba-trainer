"""The built-in task, and what makes an object acceptable as an environment."""

import numpy as np
import pytest

import mamba3_rl as m3


def test_recall_env_reports_its_own_floor_and_ceiling(env):
    assert env.chance_return == pytest.approx(1.0 / env.action_dim)
    assert env.optimal_return == 1.0
    # One channel per symbol, plus a clock and a cue-present flag.
    assert env.obs_dim == env.action_dim + 2


def test_reset_and_step_have_the_promised_shapes(env):
    obs = env.reset()
    assert obs.shape == (env.num_envs, env.obs_dim)
    assert obs.dtype == np.float32

    actions = np.zeros(env.num_envs, dtype=np.int64)
    obs, reward, done = env.step(actions)
    assert obs.shape == (env.num_envs, env.obs_dim)
    assert reward.shape == (env.num_envs,)
    assert done.shape == (env.num_envs,)


def test_the_episode_ends_exactly_at_the_horizon(env):
    env.reset()
    actions = np.zeros(env.num_envs, dtype=np.int64)
    for step in range(env.horizon):
        _, _, done = env.step(actions)
        expected = 1.0 if step + 1 == env.horizon else 0.0
        assert np.allclose(done, expected)


def test_naming_the_cue_at_the_end_is_the_only_thing_that_pays(env):
    """The expert's own action earns the ceiling, and only on the last step."""
    env.reset()
    total = np.zeros(env.num_envs, dtype=np.float32)
    for _ in range(env.horizon):
        expert = env.expert_actions()
        assert expert.shape == (env.num_envs,)
        _, reward, _ = env.step(expert)
        total += reward
    assert np.allclose(total, env.optimal_return)


def test_an_action_outside_the_space_is_refused(env):
    env.reset()
    with pytest.raises(ValueError, match="outside the 4 actions"):
        env.step(np.full(env.num_envs, env.action_dim, dtype=np.int64))


def test_an_object_missing_the_protocol_is_refused(policy):
    class Empty:
        pass

    with pytest.raises(ValueError, match="no reset"):
        m3.PpoLearner(policy, Empty(), steps=4)


def test_an_object_missing_its_dimensions_is_refused(policy):
    class NoDimensions:
        def reset(self):
            ...

        def step(self, actions):
            ...

    with pytest.raises(ValueError, match="num_envs or envs"):
        m3.PpoLearner(policy, NoDimensions(), steps=4)


def test_an_environment_of_the_wrong_width_is_refused(policy, numpy_env):
    """The policy is sized for the recall task; this environment is narrower."""
    assert numpy_env.obs_dim != policy.obs_dim
    with pytest.raises(ValueError, match="observation channels"):
        m3.PpoLearner(policy, numpy_env, steps=4)
