"""Cloning an expert, and then DAgger."""

import pytest

import mamba3_rl as m3


def test_a_schedule_decays_the_expert_s_share():
    exponential = m3.DaggerSchedule.exponential(0.5)
    assert exponential.beta(0) == pytest.approx(1.0)
    assert exponential.beta(2) == pytest.approx(0.25)

    assert m3.DaggerSchedule.linear(4).beta(2) == pytest.approx(0.5)
    assert m3.DaggerSchedule.linear(4).beta(9) == 0.0
    assert m3.DaggerSchedule.only_first().beta(0) == 1.0
    assert m3.DaggerSchedule.only_first().beta(1) == 0.0
    assert m3.DaggerSchedule.fixed(0.3).beta(7) == pytest.approx(0.3)


def test_an_environment_that_cannot_label_a_state_is_refused(policy):
    class Blind:
        """Speaks the protocol, but cannot say what an expert would do."""

        num_envs, obs_dim, action_dim = 8, 6, 4

        def reset(self):
            raise AssertionError("the learner should refuse before stepping")

        def step(self, actions):
            raise AssertionError("the learner should refuse before stepping")

    with pytest.raises(ValueError, match="expert_actions"):
        m3.ImitationLearner(policy, Blind(), steps=4)


def test_a_round_reports_its_beta_and_its_agreement(policy, env):
    cloner = m3.ImitationLearner(policy, env, steps=8, seed=3)
    stats = cloner.round()
    assert stats.round == 0
    assert stats.beta == pytest.approx(1.0)
    assert 0.0 <= stats.agreement <= 1.0
    assert stats.loss > 0.0
    assert cloner.rounds == 1


def test_agreement_can_be_skipped(policy, env):
    cloner = m3.ImitationLearner(policy, env, steps=8, seed=3)
    assert cloner.round(agreement=False).agreement is None


@pytest.mark.parametrize("agreement", [True, False])
def test_a_round_synchronises_once(policy, env, agreement):
    """The step's loss and gradient norm and the agreement come back in one read,
    not one for the step and two more for the replay's predictions and labels."""
    cloner = m3.ImitationLearner(policy, env, steps=8, seed=3)
    cloner.round(agreement=agreement)  # compile
    m3.reset_read_count()
    stats = cloner.round(agreement=agreement)
    assert m3.read_count() == 1
    assert (stats.agreement is not None) == agreement


def test_daggers_own_schedule_and_the_lr_schedule_coexist(policy, env):
    """`schedule=` (the expert-mixing beta) and `lr_schedule=` (the optimizer's
    rate) are two different knobs on two different clocks; each must move on
    its own terms without the other's presence changing that."""
    cloner = m3.ImitationLearner(
        policy,
        env,
        steps=8,
        schedule=m3.DaggerSchedule.linear(rounds=4),
        learning_rate=1.0,
        lr_schedule=m3.LrSchedule.step(every=1, gamma=0.5),
        seed=3,
    )
    betas = []
    rates = []
    for _ in range(4):
        stats = cloner.round()
        betas.append(stats.beta)
        rates.append(stats.learning_rate)

    assert betas == [pytest.approx(1.0), pytest.approx(0.75), pytest.approx(0.5), pytest.approx(0.25)]
    assert rates == [pytest.approx(0.5), pytest.approx(0.25), pytest.approx(0.125), pytest.approx(0.0625)]


def test_beta_outside_the_unit_interval_is_refused(policy, env):
    cloner = m3.ImitationLearner(policy, env, steps=8)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cloner.round(beta=1.5)


@pytest.mark.slow
def test_cloning_the_expert_solves_the_recall_task(config):
    """The end-to-end claim: an expert is cloned, and the clone carries the cue.

    A memoryless policy cannot exceed `1 / symbols` here, because the cue is on
    screen only at the first step of an episode and the reward is only at the
    last. Reaching the ceiling is the recurrent state doing its job.
    """
    policy = m3.Policy(config)
    env = m3.RecallEnv(num_envs=16, symbols=4, horizon=4, seed=11)
    cloner = m3.ImitationLearner(
        policy,
        env,
        steps=16,
        learning_rate=3e-3,
        schedule=m3.DaggerSchedule.exponential(0.7),
        seed=3,
    )
    history = cloner.run(rounds=14)

    assert history[-1].agreement > 0.9
    assert history[-1].loss < history[0].loss

    held_out = m3.RecallEnv(num_envs=16, symbols=4, horizon=4, seed=99)
    assert m3.evaluate(policy, held_out, steps=16) > held_out.chance_return
