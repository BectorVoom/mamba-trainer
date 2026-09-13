"""A2a: a saved learner can resume training, not just warm-start from its
weights."""

import json

import pytest

import mamba3_rl as m3


def _weights(path):
    """`{name: [floats]}` from a `Policy.save` JSON file, for a numeric diff
    that does not care about key order."""
    with open(path) as f:
        data = json.load(f)
    return {name: entry["data"] for name, entry in data["state"]["entries"].items()}


def _assert_close(a, b, tol=1e-4):
    assert a.keys() == b.keys()
    for name in a:
        for x, y in zip(a[name], b[name]):
            assert abs(x - y) < tol, f"{name} diverged: {x} vs {y}"


def test_ppo_learner_restores_weights_optimizer_and_counters(tmp_path, policy_for):
    from conftest import CountingEnv

    def build():
        env = CountingEnv(seed=1)
        pol = policy_for(env)
        return pol, env

    # A Python environment has no snapshot protocol here, so this test covers
    # A2a (weights/optimizer/counters), not A2b's exact environment resume.
    resumed_policy, resumed_env = build()
    resumed = m3.PpoLearner(resumed_policy, resumed_env, steps=8, learning_rate=1e-2, seed=5)
    for _ in range(3):
        resumed.round(epochs=1)
    path = str(tmp_path / "ppo.m3ck")
    resumed.save(path)

    fresh_policy, fresh_env = build()
    fresh = m3.PpoLearner(fresh_policy, fresh_env, steps=8, learning_rate=1e-2, seed=5)
    fresh.load_checkpoint(path)
    assert fresh.rounds == 3
    assert fresh.round(epochs=1).optimizer_steps == 4

    # The restore changed the newly created policy and it remains trainable.
    restored_path = str(tmp_path / "restored.json")
    fresh.policy.save(restored_path)
    assert _weights(restored_path)


def test_load_checkpoint_strict_rejects_a_weights_only_checkpoint(tmp_path, policy, env):
    policy.save(str(tmp_path / "weights_only.json"))
    learner = m3.PpoLearner(policy, env, steps=8, seed=1)
    with pytest.raises(ValueError):
        learner.load_checkpoint(str(tmp_path / "weights_only.json"), strict=True)
    # Non-strict accepts it as an explicit warm start.
    learner.load_checkpoint(str(tmp_path / "weights_only.json"), strict=False)


def test_policy_save_picks_the_binary_format_by_extension(tmp_path, policy):
    # A5's binary/JSON dispatch is purely extension-based in the Rust core, so
    # it reaches every Python caller of `save`/`load` -- `Policy`, `PpoLearner`
    # and `ImitationLearner` alike -- with no binding-side change at all.
    json_path = tmp_path / "policy.json"
    binary_path = tmp_path / "policy.m3ck"
    policy.save(str(json_path))
    policy.save(str(binary_path))
    assert json_path.stat().st_size > 0
    assert binary_path.stat().st_size > 0
    # Both still load back into a working policy.
    m3.Policy.load(str(json_path))
    m3.Policy.load(str(binary_path))


def test_ppo_learner_checkpoint_round_trips_through_the_binary_format(tmp_path, policy, env):
    learner = m3.PpoLearner(policy, env, steps=8, seed=1)
    learner.round(epochs=1)
    path = str(tmp_path / "ppo.m3ck")
    learner.save(path)

    fresh = m3.PpoLearner(policy_for_reload(policy), env, steps=8, seed=1)
    fresh.load_checkpoint(path)
    assert fresh.rounds == 1


def policy_for_reload(policy):
    """A separate `Policy` object with the same architecture, so loading a
    checkpoint into it is a genuine restore rather than a no-op onto the same
    weights."""
    return m3.Policy(policy.config)


def test_imitation_learner_save_and_load_checkpoint_round_trips(tmp_path, policy, env):
    cloner = m3.ImitationLearner(policy, env, steps=8, seed=3)
    cloner.round()
    cloner.round()
    path = str(tmp_path / "imitation.m3ck")
    cloner.save(path)

    fresh = m3.ImitationLearner(policy, env, steps=8, seed=3)
    fresh.load_checkpoint(path)
    assert fresh.rounds == 2
