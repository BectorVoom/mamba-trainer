"""A6: compiled-in device games from Python, collected through the fused rollout.

A learner given `mamba3_rl.game(...)` collects each window with one kernel per
step. The same game collected through the host path (`fused=False`) must produce
a byte-identical window: the two paths share one sampler and one transition, and
only the number of launches may differ.
"""

import numpy as np
import pytest

import mamba3_rl as m3

ENVS = 8
STEPS = 6


def policy(game, seed=7):
    return m3.Policy(m3.PolicyConfig(game.obs_dim, game.action_dim, d_model=16, n_layers=1,
                                     n_heads=1, head_dim=16, d_state=4, chunk_size=2, seed=seed))


def learner(fused, masked=False, policy_seed=7, **overrides):
    world = m3.game("recall", ENVS, symbols=4, seed=3, masked=masked)
    settings = dict(steps=STEPS, temperature=1.0, seed=11)
    settings.update(overrides)
    return m3.PpoLearner(policy(world, policy_seed), world, fused=fused, **settings)


@pytest.mark.parametrize("masked", [False, True])
def test_the_fused_window_is_the_host_window_byte_for_byte(masked):
    fused, host = learner(None, masked), learner(False, masked)
    assert fused.collection_path == "fused"
    assert host.collection_path == "host"
    for window in range(3):
        fused.collect()
        host.collect()
        a, b = fused.window(), host.window()
        assert a.keys() == b.keys()
        assert ("action_mask" in a) == masked
        for key in a:
            assert np.array_equal(a[key], b[key]), f"window {window}: {key} differs"
        fused.update(epochs=1)
        host.update(epochs=1)
    if masked:
        # Recall under a mask never allows the symbol after the cue.
        assert (a["action_mask"] == 0).any()


def test_fusing_cuts_the_launches_per_step():
    fused, host = learner(None), learner(False)
    for run in (fused, host):
        run.collect()  # compile and allocate
    m3.reset_launch_count()
    fused.collect()
    fused_launches = m3.launch_count()
    m3.reset_launch_count()
    host.collect()
    host_launches = m3.launch_count()
    assert fused_launches < host_launches, (fused_launches, host_launches)
    assert m3.read_count() >= 0


def test_unknown_games_and_parameters_are_refused():
    with pytest.raises(ValueError, match="no game named"):
        m3.game("pong", ENVS)
    with pytest.raises(ValueError, match="horizon"):
        m3.game("recall", ENVS, horizon=5)
    with pytest.raises(ValueError, match="symbols"):
        m3.game("recall", ENVS, symbols=1)
    with pytest.raises(ValueError, match="environment"):
        m3.game("recall", 0)


def test_unsupported_combinations_are_refused_rather_than_run_another_way():
    env = m3.RecallEnv(ENVS, symbols=4, horizon=8)
    with pytest.raises(ValueError, match="fused=True"):
        m3.PpoLearner(policy(env), env, steps=STEPS, fused=True)
    world = m3.game("recall", ENVS)
    with pytest.raises(ValueError, match="expert"):
        m3.ImitationLearner(policy(world), world, steps=STEPS)


def test_a_game_is_also_an_ordinary_environment():
    world = m3.game("recall", ENVS, symbols=4, masked=True)
    assert (world.name, world.num_envs, world.obs_dim, world.action_dim, world.masked) == \
        ("recall", ENVS, 6, 4, True)
    obs = world.reset()
    assert obs.shape == (ENVS, 6)
    mask = world.action_mask()
    assert mask.shape == (ENVS, 4) and (mask.sum(axis=1) == 3).all()
    obs, reward, done = world.step(np.zeros(ENVS, dtype=np.int64))
    assert obs.shape == (ENVS, 6) and reward.shape == (ENVS,) and done.shape == (ENVS,)
    assert "recall" in repr(world)


def test_a_game_step_synchronises_once():
    """Observation, reward and done come back in one read, not three."""
    world = m3.game("recall", ENVS, symbols=4)
    world.reset()
    actions = np.zeros(ENVS, dtype=np.int64)
    world.step(actions)

    m3.reset_read_count()
    world.step(actions)
    assert m3.read_count() == 1


def test_a_game_run_continues_exactly_from_a_full_checkpoint(tmp_path):
    def rounds(run, n):
        return [(s.loss, s.entropy, s.episode_return) for s in (run.round(epochs=1) for _ in range(n))]

    continuous = learner(None, masked=True)
    rounds(continuous, 2)
    continuous.collect()
    expected = continuous.window()
    continuous.update(epochs=1)

    first = learner(None, masked=True)
    rounds(first, 2)
    path = tmp_path / "game.m3ck"
    first.save(str(path), level="full")

    # Other initial weights; the game itself must be built the same way, or its
    # load_state refuses the bytes.
    second = learner(None, masked=True, policy_seed=99)
    assert second.load_checkpoint(str(path))["level"] == "full"
    assert second.collection_path == "fused"
    second.collect()
    got = second.window()
    for key in expected:
        assert np.array_equal(got[key], expected[key]), key
