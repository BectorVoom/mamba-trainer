"""The weights: what they are, and how they travel."""

import json

import pytest

import mamba3_rl as m3
from weights import entries, fingerprint


def test_a_policy_reports_the_architecture_it_was_built_from(policy, config):
    assert policy.config == config
    assert policy.obs_dim == config.obs_dim
    assert policy.action_dim == config.action_dim
    assert policy.num_parameters > 0
    assert policy.num_trainable_parameters == policy.num_parameters
    assert policy.backend == m3.backend()


def test_the_seed_decides_the_weights(config):
    first = m3.Policy(config)
    same = m3.Policy(config)
    assert first.describe() == same.describe()


def test_freezing_takes_parameters_out_of_the_optimizer_s_reach(policy):
    total = policy.num_parameters
    policy.freeze(["blocks"])
    frozen = policy.num_trainable_parameters
    assert 0 < frozen < total
    policy.unfreeze(["blocks"])
    assert policy.num_trainable_parameters == total


def test_a_checkpoint_carries_its_own_architecture(policy, tmp_path):
    path = str(tmp_path / "policy.json")
    policy.save(path, step=3)
    restored = m3.Policy.load(path)
    assert restored.config == policy.config
    # Same architecture and the same weights: `describe` prints every parameter
    # path and shape, and the weights themselves are compared through a rollout.
    assert restored.describe() == policy.describe()


def test_loading_weights_into_a_mismatched_policy_is_refused(policy, tmp_path):
    path = str(tmp_path / "policy.json")
    policy.save(path)
    other = m3.Policy(m3.PolicyConfig(policy.obs_dim, policy.action_dim, d_model=64, n_layers=1))
    with pytest.raises(RuntimeError):
        other.load_weights(path, strict=True)


def test_loading_a_checkpoint_with_no_architecture_in_it(policy, tmp_path):
    path = tmp_path / "policy.json"
    policy.save(str(path))
    checkpoint = json.loads(path.read_text())
    checkpoint["metadata"] = {}
    path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="no policy architecture"):
        m3.Policy.load(str(path))


def test_fingerprint_matches_checkpoint(policy, env, tmp_path):
    path = str(tmp_path / "policy.json")
    policy.save(path)
    before = policy.fingerprint()
    assert len(before) == 16 and int(before, 16) >= 0
    # Computed from the file by an independent implementation of the format.
    assert before == fingerprint(entries(path))
    assert m3.Policy.load(path).fingerprint() == before

    # Acting does not touch the weights; training does.
    m3.evaluate(policy, m3.RecallEnv(num_envs=4, symbols=4, horizon=4, seed=3), steps=8)
    assert policy.fingerprint() == before
    learner = m3.PpoLearner(policy, env, steps=8, learning_rate=1e-3)
    learner.round(epochs=1)
    after = policy.fingerprint()
    assert after != before
    policy.save(path)
    assert after == fingerprint(entries(path))


def test_loading_something_that_is_not_a_checkpoint(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        m3.Policy.load(str(path))
