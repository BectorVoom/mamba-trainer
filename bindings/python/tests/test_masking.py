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


# ---------------------------------------------------------------------------
# K4: `None` from action_mask() means every action legal on that step
# ---------------------------------------------------------------------------


class _SometimesMasked(HalfMaskedCountingEnv):
    """The review probes' shapes: a mask on some steps and `None` on others."""

    def __init__(self, masked_steps, **kwargs):
        super().__init__(**kwargs)
        self.masked_steps = masked_steps
        self.t = 0
        self.actions = []

    def step(self, actions):
        self.actions.append((self.t, np.asarray(actions).copy()))
        self.t += 1
        return super().step(actions)

    def action_mask(self):
        return super().action_mask() if self.masked_steps(self.t) else None


@pytest.mark.parametrize(
    "name, masked_steps",
    [("mask then none", lambda t: t == 0), ("none then mask", lambda t: t > 0)],
)
def test_a_mask_may_come_and_go_within_a_window(policy, name, masked_steps):
    env = _SometimesMasked(masked_steps)
    learner = m3.PpoLearner(policy, env, steps=4, learning_rate=1e-3, seed=5)
    stats = learner.round(epochs=1)
    assert np.isfinite(stats.loss), name
    assert stats.approx_kl < 1e-3, name
    for t, actions in env.actions:
        if masked_steps(t):
            assert (actions % 2 == 0).all(), f"{name}: step {t} drew an illegal action"


def test_a_boolean_mask_is_accepted(policy):
    class BoolMasked(HalfMaskedCountingEnv):
        def action_mask(self):
            return super().action_mask().astype(bool)

    learner = m3.PpoLearner(policy, BoolMasked(), steps=4, seed=5)
    assert np.isfinite(learner.round(epochs=1).loss)


class _StepCounting(HalfMaskedCountingEnv):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.stepped = 0

    def step(self, actions):
        self.stepped += 1
        return super().step(actions)

    def action_mask(self):
        return self.mask(self)


def test_a_mask_that_is_not_zero_or_one_is_refused_before_any_step(policy):
    env = _StepCounting(lambda e: np.full((e.num_envs, e.action_dim), 0.5, dtype=np.float32))
    learner = m3.PpoLearner(policy, env, steps=4, seed=5)
    with pytest.raises(ValueError, match="other than 0 or 1"):
        learner.collect()
    assert env.stepped == 0


def test_an_empty_row_is_refused_before_the_environment_is_stepped(policy):
    env = _StepCounting(lambda e: np.zeros((e.num_envs, e.action_dim), dtype=np.float32))
    learner = m3.PpoLearner(policy, env, steps=4, seed=5)
    with pytest.raises(ValueError, match="no legal action"):
        learner.collect()
    assert env.stepped == 0


def test_an_exception_in_action_mask_reaches_the_caller_before_the_draw(policy):
    class Broken(Exception):
        pass

    def mask(env):
        if env.stepped == 2:
            raise Broken("no mask today")
        return HalfMaskedCountingEnv.action_mask(env)

    env = _StepCounting(mask)
    learner = m3.PpoLearner(policy, env, steps=4, seed=5)
    with pytest.raises(Broken):
        learner.collect()
    assert env.stepped == 2, "the environment was stepped past the failing mask"


def test_evaluate_draws_only_legal_actions_from_a_masked_environment(policy):
    env = _SometimesMasked(lambda t: True)
    m3.evaluate(policy, env, steps=6, temperature=1.0, seed=3)
    assert all((actions % 2 == 0).all() for _, actions in env.actions)


# ---------------------------------------------------------------------------
# K5: Rollout.step / Rollout.evaluate take the same mask
# ---------------------------------------------------------------------------


EVEN = np.tile(np.array([1, 0, 1, 0], dtype=np.float32), (NUM_ENVS, 1))


def _obs(seed):
    return np.random.default_rng(seed).standard_normal((NUM_ENVS, OBS_DIM)).astype(np.float32)


@pytest.mark.parametrize("temperature", [0.0, 1.0, 5.0])
def test_a_masked_rollout_never_draws_an_illegal_action(policy, temperature):
    rollout = m3.Rollout(policy, NUM_ENVS, temperature=temperature, seed=1)
    for i in range(50):
        actions, _, log_probs = rollout.step(_obs(i), action_mask=EVEN)
        assert (actions % 2 == 0).all()
        assert np.isfinite(log_probs).all()


def test_masked_log_probs_are_the_masked_distributions(policy):
    stepper = m3.Rollout(policy, NUM_ENVS, temperature=1.0, seed=2)
    evaluator = m3.Rollout(policy, NUM_ENVS, seed=2)
    obs = _obs(7)
    actions, _, log_probs = stepper.step(obs, action_mask=EVEN)
    logits, _ = evaluator.evaluate(obs, action_mask=EVEN)
    assert (logits[:, 1::2] == np.finfo(np.float32).min).all()
    legal = logits[:, 0::2].astype(np.float64)
    top = legal.max(axis=1, keepdims=True)
    log_softmax = logits.astype(np.float64) - (top + np.log(np.exp(legal - top).sum(axis=1, keepdims=True)))
    expected = log_softmax[np.arange(NUM_ENVS), actions]
    np.testing.assert_allclose(log_probs, expected, atol=1e-5)


def test_an_all_legal_mask_is_identical_to_no_mask(policy):
    plain = m3.Rollout(policy, NUM_ENVS, temperature=1.0, seed=4)
    masked = m3.Rollout(policy, NUM_ENVS, temperature=1.0, seed=4)
    ones = np.ones((NUM_ENVS, ACTION_DIM), dtype=np.float32)
    for i in range(5):
        a = plain.step(_obs(i))
        b = masked.step(_obs(i), action_mask=ones)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


@pytest.mark.parametrize(
    "mask, message",
    [
        (np.zeros((NUM_ENVS, ACTION_DIM), dtype=np.float32), "no legal action"),
        (np.full((NUM_ENVS, ACTION_DIM), np.nan, dtype=np.float32), "non-finite"),
        (np.full((NUM_ENVS, ACTION_DIM), 2.0, dtype=np.float32), "other than 0 or 1"),
        (np.ones((NUM_ENVS, ACTION_DIM + 1), dtype=np.float32), "must be"),
    ],
)
def test_a_bad_rollout_mask_raises_and_does_not_advance(policy, mask, message):
    probed = m3.Rollout(policy, NUM_ENVS, temperature=0.0)
    untouched = m3.Rollout(policy, NUM_ENVS, temperature=0.0)
    with pytest.raises(ValueError, match=message):
        probed.step(_obs(0), action_mask=mask)
    with pytest.raises(ValueError, match=message):
        probed.evaluate(_obs(0), action_mask=mask)
    # Neither failed call moved the recurrent state.
    assert np.array_equal(probed.evaluate(_obs(1))[0], untouched.evaluate(_obs(1))[0])


def test_rollout_and_learner_collection_draw_the_same_masked_actions(policy):
    class Logged(_SometimesMasked):
        def __init__(self):
            super().__init__(lambda t: t % 2 == 0)
            self.observations = []
            self.masks = []
            self.dones = [np.zeros(self.num_envs, dtype=np.float32)]

        def reset(self):
            obs = super().reset()
            self.observations.append(obs)
            return obs

        def step(self, actions):
            obs, reward, done = super().step(actions)
            self.observations.append(obs)
            self.dones.append(done)
            return obs, reward, done

        def action_mask(self):
            mask = super().action_mask()
            self.masks.append(mask)
            return mask

    env = Logged()
    learner = m3.PpoLearner(policy, env, steps=6, temperature=1.0, seed=9)
    learner.collect()

    rollout = m3.Rollout(policy, NUM_ENVS, temperature=1.0, seed=9)
    for t, (step, taken) in enumerate(env.actions):
        actions, _, _ = rollout.step(env.observations[t], reset=env.dones[t], action_mask=env.masks[t])
        assert np.array_equal(actions, taken), f"step {step}: Rollout and the learner disagree"
