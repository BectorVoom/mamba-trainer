"""An environment written in Python, driven by the same loops.

The boundary is the interesting part here: observations arrive as host arrays and
are copied onto the device, actions are copied back, and an exception raised in
someone's `step()` has to reach them as their own exception with their own
traceback — not as a string inside a Rust error.
"""

import numpy as np
import pytest

import mamba3_rl as m3


@pytest.fixture
def policy(policy_for, numpy_env):
    return policy_for(numpy_env)


@pytest.fixture
def learner(policy, numpy_env):
    return m3.PpoLearner(policy, numpy_env, steps=8, learning_rate=1e-3, seed=5)


def test_a_numpy_environment_drives_a_whole_round(learner, numpy_env):
    stats = learner.round(epochs=2)
    assert numpy_env.resets == 1
    assert numpy_env.steps_taken == learner.steps
    assert stats.episode_return is not None


def test_the_environment_object_is_handed_back_unchanged(learner, numpy_env):
    assert learner.env is numpy_env


def test_the_environment_is_reset_once_and_then_carried_across_windows(learner, numpy_env):
    learner.run(rounds=3, epochs=1)
    assert numpy_env.resets == 1
    assert numpy_env.steps_taken == 3 * learner.steps


def test_resetting_the_learner_starts_the_environment_over(learner, numpy_env):
    learner.round(epochs=1)
    learner.reset()
    learner.round(epochs=1)
    assert numpy_env.resets == 2


def test_a_float64_observation_is_accepted(policy, numpy_env):
    original = numpy_env._observation

    def as_float64():
        return original().astype(np.float64)

    numpy_env._observation = as_float64
    learner = m3.PpoLearner(policy, numpy_env, steps=4, seed=1)
    assert learner.round(epochs=1).episode_return is not None


def test_a_misshapen_observation_names_the_method_that_returned_it(policy, numpy_env):
    numpy_env._observation = lambda: np.zeros((numpy_env.num_envs,), dtype=np.float32)
    learner = m3.PpoLearner(policy, numpy_env, steps=4)
    with pytest.raises(ValueError, match="observation returned by reset"):
        learner.collect()


def test_a_step_that_returns_the_wrong_thing_says_what_was_wanted(policy, numpy_env):
    numpy_env.step = lambda actions: numpy_env._observation()
    learner = m3.PpoLearner(policy, numpy_env, steps=4)
    with pytest.raises(ValueError, match="observation, reward, done"):
        learner.collect()


def test_an_exception_in_the_environment_reaches_the_caller(policy, numpy_env):
    class Boom(Exception):
        pass

    def explode(actions):
        raise Boom("the simulator fell over")

    numpy_env.step = explode
    learner = m3.PpoLearner(policy, numpy_env, steps=4)
    with pytest.raises(Boom, match="the simulator fell over"):
        learner.collect()


def test_an_exception_in_the_expert_reaches_the_caller(policy, numpy_env):
    class Boom(Exception):
        pass

    def explode():
        raise Boom("the expert fell over")

    numpy_env.expert_actions = explode
    cloner = m3.ImitationLearner(policy, numpy_env, steps=4)
    with pytest.raises(Boom, match="the expert fell over"):
        cloner.round()


def test_an_action_the_environment_cannot_accept_is_caught_before_it_is_sent(
    policy, numpy_env
):
    """An expert label outside the action space is refused, not silently indexed."""
    numpy_env.expert_actions = lambda: np.full(numpy_env.num_envs, 99, dtype=np.int64)
    cloner = m3.ImitationLearner(policy, numpy_env, steps=4)
    with pytest.raises(ValueError, match="outside the 4 actions"):
        cloner.round()


def test_the_learner_and_a_rollout_agree_on_the_same_environment(policy, numpy_env):
    """Driving the policy by hand reaches the same shapes the learner does."""
    rollout = m3.Rollout(policy, num_envs=numpy_env.num_envs, temperature=0.0)
    obs, done = numpy_env.reset(), None
    for _ in range(4):
        actions, values, log_probs = rollout.step(obs, reset=done)
        assert actions.shape == values.shape == log_probs.shape == (numpy_env.num_envs,)
        obs, reward, done = numpy_env.step(actions)
        assert reward.shape == (numpy_env.num_envs,)
