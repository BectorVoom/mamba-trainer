"""Configurations validate on construction, not at the first kernel launch."""

import pytest

import mamba3_rl as m3


def test_defaults_are_consistent():
    config = m3.PolicyConfig(6, 4)
    assert config.obs_dim == 6
    assert config.action_dim == 4
    assert config.d_model == 64
    assert config.n_heads * config.head_dim == config.d_model
    assert config.conv_kernel == 4


def test_head_dim_implies_a_head_count():
    config = m3.PolicyConfig(6, 4, d_model=64, head_dim=16)
    assert config.n_heads == 4
    assert config.n_groups == 4


def test_a_zero_conv_kernel_disables_the_convolution():
    assert m3.PolicyConfig(6, 4, conv_kernel=0).conv_kernel is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"obs_dim": 0, "action_dim": 4}, "positive observation"),
        ({"obs_dim": 6, "action_dim": 0}, "positive observation"),
        ({"obs_dim": 6, "action_dim": 4, "n_layers": 0}, "at least one layer"),
        ({"obs_dim": 6, "action_dim": 4, "discretization": "runge_kutta"}, "unknown discretization"),
        ({"obs_dim": 6, "action_dim": 4, "dynamics": "complex"}, "unknown dynamics"),
        ({"obs_dim": 6, "action_dim": 4, "n_groups": 0}, "must all be positive"),
    ],
)
def test_a_bad_architecture_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        m3.PolicyConfig(**kwargs)


def test_config_round_trips_through_a_dict():
    config = m3.PolicyConfig(6, 4, d_model=32, n_layers=3, d_state=8, seed=11)
    restored = m3.PolicyConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.seed == 11
    assert restored.d_state == 8


def test_from_dict_defaults_a_missing_seed_and_norm_eps():
    payload = m3.PolicyConfig(6, 4).to_dict()
    del payload["seed"]
    del payload["norm_eps"]
    restored = m3.PolicyConfig.from_dict(payload)
    assert restored.seed == 0
    assert restored.norm_eps == pytest.approx(1e-5)


def test_ppo_defaults_are_the_usual_ones():
    config = m3.PpoConfig()
    assert config.gamma == pytest.approx(0.99)
    assert config.gae_lambda == pytest.approx(0.95)
    assert config.clip_coeff == pytest.approx(0.2)
    assert config.normalize_advantages


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"gamma": 1.5}, "discount factors"),
        ({"gae_lambda": -0.1}, "discount factors"),
        ({"clip_coeff": 0.0}, "clip range must be positive"),
    ],
)
def test_bad_ppo_hyperparameters_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        m3.PpoConfig(**kwargs)


# ---------------------------------------------------------------------------
# A3: m3.LrSchedule
# ---------------------------------------------------------------------------


def test_constant_never_changes_the_rate():
    schedule = m3.LrSchedule.constant()
    assert schedule.rate_at(1.0, 1) == schedule.rate_at(1.0, 1000)


def test_cosine_lies_strictly_between_its_endpoints_after_warmup():
    # A 1-step warmup so `rate_at(.., 1)` is already the peak of the curve
    # (`cosine(100)`'s *default* 2-step warmup would still be ramping up at
    # step 1, which is lower than the mid-run point on the decay that follows
    # it -- correctly so, but not what this test means to exercise).
    schedule = m3.LrSchedule.cosine(100, warmup_steps=1)
    start = schedule.rate_at(1.0, 1)
    mid = schedule.rate_at(1.0, 50)
    end = schedule.rate_at(1.0, 100)
    assert end < mid < start


def test_cosine_default_warmup_is_two_percent():
    # `warmup_steps=None` should match the Rust convenience constructor's
    # default of `(total_steps / 50).max(1)`.
    with_default = m3.LrSchedule.cosine(200)
    explicit = m3.LrSchedule.cosine(200, warmup_steps=4)
    for step in (1, 2, 4, 50, 200):
        assert with_default.rate_at(1.0, step) == pytest.approx(explicit.rate_at(1.0, step))


def test_linear_decays_to_min_ratio():
    schedule = m3.LrSchedule.linear(100, warmup_steps=1, min_ratio=0.2)
    assert schedule.rate_at(1.0, 100) == pytest.approx(0.2, abs=1e-3)


def test_step_halves_on_schedule():
    schedule = m3.LrSchedule.step(every=10, gamma=0.5)
    assert schedule.rate_at(1.0, 9) == pytest.approx(1.0)
    assert schedule.rate_at(1.0, 10) == pytest.approx(0.5)
    assert schedule.rate_at(1.0, 20) == pytest.approx(0.25)


def test_inverse_sqrt_decays_past_warmup():
    schedule = m3.LrSchedule.inverse_sqrt(10)
    assert schedule.rate_at(1.0, 10) == pytest.approx(1.0)
    assert schedule.rate_at(1.0, 40) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "call",
    [
        lambda: m3.LrSchedule.cosine(10, warmup_steps=20),
        lambda: m3.LrSchedule.cosine(0),
        lambda: m3.LrSchedule.linear(10, min_ratio=-0.1),
        lambda: m3.LrSchedule.step(every=0, gamma=0.5),
        lambda: m3.LrSchedule.step(every=10, gamma=-1.0),
        lambda: m3.LrSchedule.inverse_sqrt(0),
    ],
)
def test_invalid_schedules_are_refused(call):
    with pytest.raises(ValueError):
        call()
