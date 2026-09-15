"""The PPO loop: collect a window, replay it, take a step."""

import numpy as np
import pytest

import mamba3_rl as m3


@pytest.fixture
def learner(policy, env):
    return m3.PpoLearner(policy, env, steps=8, learning_rate=1e-3, seed=5)


def test_a_learner_shares_the_policy_it_was_given(learner, policy):
    assert learner.policy.config == policy.config
    assert learner.num_envs == learner.env.num_envs
    assert learner.steps == 8


def test_an_update_before_a_collection_says_so(learner):
    with pytest.raises(ValueError, match="nothing has been collected"):
        learner.update()


def test_a_round_reports_every_diagnostic(learner):
    stats = learner.round(epochs=2)
    assert stats.round == 0
    assert stats.steps == learner.steps
    assert stats.optimizer_steps == 2
    assert stats.episode_return is not None
    assert np.isfinite([stats.loss, stats.policy_loss, stats.value_loss]).all()
    assert stats.entropy > 0.0
    assert stats.approx_kl >= 0.0
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.learning_rate == pytest.approx(1e-3)
    assert learner.rounds == 1


def test_a_bare_update_does_not_look_at_the_environment(learner):
    learner.collect()
    assert learner.update(epochs=1).episode_return is None


def test_a_sub_episode_window_reports_no_return_at_all(policy, env):
    """A window shorter than the task's horizon cannot complete an episode, and
    must report that honestly rather than silently reporting its total reward as
    if it were a mean over however many episodes happened to finish."""
    short = m3.PpoLearner(policy, env, steps=1, seed=5)
    assert short.round(epochs=1).episode_return is None


def test_evaluate_reports_no_return_when_nothing_completes(policy, env):
    assert m3.evaluate(policy, env, steps=1) is None


# ---------------------------------------------------------------------------
# A3: lr_schedule
# ---------------------------------------------------------------------------


def test_no_schedule_leaves_the_rate_unchanged_across_rounds(policy, env):
    plain = m3.PpoLearner(policy, env, steps=8, learning_rate=1e-3, seed=5)
    rates = [plain.round(epochs=2).learning_rate for _ in range(3)]
    assert rates == [pytest.approx(1e-3)] * 3


def test_a_schedule_advances_on_optimizer_steps_not_rounds(policy, env):
    # `epochs=1` makes one round exactly one optimizer step, so `cosine(10)`'s
    # ten steps line up with ten rounds and the rate should strictly decrease
    # round over round -- not stay flat, which is what it would do if the
    # schedule were keyed off `rounds` instead of `Trainer::step_count()`.
    scheduled = m3.PpoLearner(
        policy,
        env,
        steps=8,
        learning_rate=1.0,
        lr_schedule=m3.LrSchedule.cosine(10),
        seed=5,
    )
    rates = [scheduled.round(epochs=1).learning_rate for _ in range(10)]
    assert all(b < a for a, b in zip(rates, rates[1:])), rates


def test_the_buffer_does_not_grow_with_the_run(learner):
    before = learner.buffer_bytes
    learner.run(rounds=3, epochs=1)
    assert learner.buffer_bytes == before


def test_minibatches_split_the_environment_axis(learner):
    learner.collect()
    stats = learner.update(epochs=2, minibatches=2)
    assert stats.optimizer_steps == 4


@pytest.mark.parametrize("epochs,minibatches", [(1, 1), (3, 1), (2, 4)])
def test_an_update_synchronises_once_however_many_steps_it_takes(learner, epochs, minibatches):
    """Every step's loss and gradient norm, and the diagnostics, come back in one
    read — not two per step, one per minibatch cut and six at the end."""
    learner.collect()
    m3.reset_read_count()
    learner.update(epochs=epochs, minibatches=minibatches)
    assert m3.read_count() == 1


def test_a_round_on_a_device_environment_synchronises_once(learner):
    learner.round(epochs=1)  # compile
    m3.reset_read_count()
    learner.round(epochs=2, minibatches=2)
    assert m3.read_count() == 1


def test_a_round_reports_what_collect_update_and_episode_return_do(policy_for, env):
    """`round()` reads the episode return with the update's numbers; the result
    must be the one the three separate calls give, to the bit."""
    def learner():
        world = m3.RecallEnv(num_envs=env.num_envs, symbols=4, horizon=4, seed=3)
        return m3.PpoLearner(policy_for(world), world, steps=8, seed=5)

    together, apart = learner(), learner()
    for _ in range(2):
        a = together.round(epochs=2, minibatches=2)
        apart.collect()
        b = apart.update(epochs=2, minibatches=2)
        episode_return = apart.episode_return()
        for field in ("steps", "optimizer_steps", "loss", "policy_loss", "value_loss",
                      "entropy", "approx_kl", "clip_fraction", "reference_kl",
                      "grad_norm", "learning_rate"):
            assert getattr(a, field) == getattr(b, field), field
        assert a.episode_return == episode_return
        assert b.episode_return is None
    assert together.policy.fingerprint() == apart.policy.fingerprint()


def test_minibatches_that_do_not_divide_the_environments_are_refused(learner):
    learner.collect()
    with pytest.raises(ValueError, match="must divide the 8 environments"):
        learner.update(minibatches=3)


def test_zero_epochs_are_refused(learner):
    learner.collect()
    with pytest.raises(ValueError, match="at least one epoch"):
        learner.update(epochs=0)


def test_run_reports_every_round_in_order(learner):
    history = learner.run(rounds=3, epochs=1)
    assert [stats.round for stats in history] == [0, 1, 2]
    assert learner.rounds == 3


def test_a_callback_sees_each_round_and_can_stop_the_run(learner):
    seen = []

    def watch(stats):
        seen.append(stats.round)
        return stats.round < 1

    history = learner.run(rounds=5, epochs=1, callback=watch)
    assert seen == [0, 1]
    assert len(history) == 2


def test_a_callback_that_returns_nothing_does_not_stop_the_run(learner):
    history = learner.run(rounds=2, epochs=1, callback=lambda stats: None)
    assert len(history) == 2


def test_an_exception_in_a_callback_reaches_the_caller(learner):
    class Stop(Exception):
        pass

    def explode(stats):
        raise Stop("from the callback")

    with pytest.raises(Stop, match="from the callback"):
        learner.run(rounds=2, epochs=1, callback=explode)


def test_resetting_forgets_the_window_but_not_the_weights(learner):
    learner.round(epochs=1)
    trained = learner.policy.describe()
    learner.reset()
    with pytest.raises(ValueError, match="nothing has been collected"):
        learner.update()
    assert learner.policy.describe() == trained


def test_evaluating_does_not_disturb_a_learner(learner, policy):
    learner.round(epochs=1)
    fresh = m3.RecallEnv(num_envs=learner.num_envs, symbols=4, horizon=4, seed=99)
    value = m3.evaluate(policy, fresh, steps=8)
    assert 0.0 <= value <= 1.0
    assert learner.round(epochs=1).episode_return is not None
