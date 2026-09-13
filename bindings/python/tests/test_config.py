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
