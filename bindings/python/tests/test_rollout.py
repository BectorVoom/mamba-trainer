"""Acting: one observation in, one action out, in constant state."""

import numpy as np
import pytest

import mamba3_rl as m3


def test_a_step_returns_an_action_a_value_and_a_density(policy, env):
    rollout = m3.Rollout(policy, num_envs=env.num_envs, seed=2)
    actions, values, log_probs = rollout.step(env.reset())

    assert actions.shape == (env.num_envs,)
    assert actions.dtype == np.int64
    assert np.all((actions >= 0) & (actions < env.action_dim))
    assert values.shape == (env.num_envs,)
    assert np.all(log_probs <= 0.0)


def test_the_state_is_fixed_size(policy, env):
    rollout = m3.Rollout(policy, num_envs=env.num_envs)
    before = rollout.state_bytes
    obs = env.reset()
    for _ in range(16):
        actions, _, _ = rollout.step(obs)
        obs, _, _ = env.step(actions)
    assert rollout.state_bytes == before


def test_the_seed_decides_the_draws(policy, env):
    obs = env.reset()
    first = m3.Rollout(policy, num_envs=env.num_envs, seed=5).step(obs)[0]
    same = m3.Rollout(policy, num_envs=env.num_envs, seed=5).step(obs)[0]
    assert np.array_equal(first, same)


def test_temperature_zero_acts_greedily(policy, env):
    obs = env.reset()
    greedy = m3.Rollout(policy, num_envs=env.num_envs, temperature=0.0)
    logits, _ = m3.Rollout(policy, num_envs=env.num_envs).evaluate(obs)
    assert np.array_equal(greedy.step(obs)[0], logits.argmax(axis=1))


def test_evaluate_reports_the_whole_distribution(policy, env):
    rollout = m3.Rollout(policy, num_envs=env.num_envs)
    logits, values = rollout.evaluate(env.reset())
    assert logits.shape == (env.num_envs, env.action_dim)
    assert values.shape == (env.num_envs,)


def test_resetting_the_state_makes_the_next_step_the_first_one(policy, env):
    obs = env.reset()
    rollout = m3.Rollout(policy, num_envs=env.num_envs, temperature=0.0)
    first, _ = rollout.evaluate(obs)
    rollout.step(obs)
    rollout.reset()
    again, _ = rollout.evaluate(obs)
    assert np.allclose(first, again, atol=1e-5)


def test_the_reset_mask_cuts_the_recurrence(policy, env):
    """An observation marked as beginning an episode cannot see what came before."""
    obs = env.reset()
    rollout = m3.Rollout(policy, num_envs=env.num_envs, temperature=0.0)
    fresh, _ = rollout.evaluate(obs)

    rollout.reset()
    rollout.step(obs)
    rollout.step(obs)
    cut, _ = rollout.evaluate(obs, reset=np.ones(env.num_envs, dtype=np.float32))
    assert np.allclose(fresh, cut, atol=1e-5)


def test_a_step_synchronises_once(policy, env):
    """Actions, values and log-probabilities come back in one read, not three.

    On a GPU each read is a fixed wait for the device that costs more than the
    step's launches, so a second read on this path is a slowdown, not a detail.
    """
    obs = env.reset()
    done = np.zeros(env.num_envs, dtype=np.float32)
    rollout = m3.Rollout(policy, num_envs=env.num_envs, seed=4)
    rollout.step(obs, reset=done)

    m3.reset_read_count()
    rollout.step(obs, reset=done)
    assert m3.read_count() == 1

    m3.reset_read_count()
    rollout.evaluate(obs, reset=done)
    assert m3.read_count() == 1


def test_a_misshapen_observation_says_so(policy, env):
    rollout = m3.Rollout(policy, num_envs=env.num_envs)
    with pytest.raises(ValueError, match=r"obs must be \[8, 6\]"):
        rollout.step(np.zeros((env.num_envs, env.obs_dim + 1), dtype=np.float32))


def test_a_negative_temperature_is_refused(policy):
    with pytest.raises(ValueError, match="non-negative"):
        m3.Rollout(policy, num_envs=2, temperature=-1.0)
