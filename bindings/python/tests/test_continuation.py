"""A2b: exact continuation.

N rounds, `save(level="full")`, a *new process* that restores and runs M rounds,
against N + M rounds that never stopped. Sampling is stochastic (temperature 1),
the environment resets asynchronously from its own random generator, masks
change with the clock, PPO carries a reference and an LR schedule, and DAgger
mixes the expert on a decaying schedule — every source of state the continuation
has to carry.

What is compared exactly: every sampled action (the environment logs them),
counters, learning rates, the DAgger mixture. Losses, returns and weights are
compared to 1e-5, because the CPU runtime's parallel reductions are not
bit-reproducible from one process to the next (two identical runs differ by
about 1e-7; a different trajectory differs by orders of magnitude more).
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

import mamba3_rl as m3
from continuation_env import BUILDERS, StatefulLaneEnv, one_round, weights

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST, SECOND = 3, 3
TOL = 1e-5


def close(a, b):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


RESTORE = textwrap.dedent("""
    import json, sys, pathlib
    sys.path.insert(0, {here!r})
    import mamba3_rl as m3
    from continuation_env import BUILDERS, StatefulLaneEnv, one_round, weights

    kind, checkpoint, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    learner = BUILDERS[kind](env=StatefulLaneEnv(seed=99))
    summary = learner.load_checkpoint(checkpoint)
    rounds = [one_round(learner) for _ in range({second})]
    out.write_text(json.dumps({{
        "summary": summary,
        "continuation": learner.continuation,
        "rounds": rounds,
        "log": learner.env.log,
        "weights": weights(learner, out.with_suffix(".weights.json")),
    }}))
""")


@pytest.mark.parametrize("kind", ["ppo", "imitation"])
def test_a_full_checkpoint_continues_exactly_in_another_process(tmp_path, kind):
    build = BUILDERS[kind]

    continuous = build()
    for _ in range(FIRST):
        one_round(continuous)
    logged = len(continuous.env.log)
    expected = [one_round(continuous) for _ in range(SECOND)]
    expected_log = continuous.env.log[logged:]
    expected_weights = weights(continuous, tmp_path / "continuous.json")

    first = build()
    for _ in range(FIRST):
        one_round(first)
    checkpoint = tmp_path / f"{kind}.m3ck"
    first.save(str(checkpoint), level="full")

    script = tmp_path / "restore.py"
    script.write_text(RESTORE.format(here=HERE, second=SECOND))
    result = tmp_path / "restored.json"
    subprocess.run([sys.executable, str(script), kind, str(checkpoint), str(result)],
                   check=True, capture_output=True, text=True)
    restored = json.loads(result.read_text())

    assert restored["summary"]["level"] == "full", restored["summary"]
    assert restored["continuation"] == {"level": "full", "notes": []}
    assert restored["log"][logged:] == expected_log, "sampled actions diverged"
    for got, want in zip(restored["rounds"], expected):
        for key, value in want.items():
            if isinstance(value, float) or value is None:
                assert close(got[key], value), f"round {want['round']} {key}: {got[key]} vs {value}"
            else:
                assert got[key] == value, f"round {want['round']} {key}: {got[key]} vs {value}"
    for name, values in expected_weights.items():
        worst = max(abs(a - b) for a, b in zip(values, restored["weights"][name]))
        assert worst <= TOL, f"{name} differs by {worst}"


def test_without_the_restore_the_run_diverges(tmp_path):
    """The comparison above is not vacuous: restoring only the optimizer level
    samples different actions from the very next round."""
    continuous = BUILDERS["ppo"]()
    for _ in range(FIRST):
        one_round(continuous)
    logged = len(continuous.env.log)
    one_round(continuous)

    first = BUILDERS["ppo"]()
    for _ in range(FIRST):
        one_round(first)
    path = tmp_path / "full.m3ck"
    first.save(str(path), level="full")
    warm = BUILDERS["ppo"](env=StatefulLaneEnv(seed=99))
    assert warm.load_checkpoint(str(path), level="optimizer")["level"] == "optimizer"
    one_round(warm)
    assert warm.env.log != continuous.env.log[logged:]


def test_a_full_save_needs_the_environment_protocol(tmp_path):
    class Bare:
        num_envs, obs_dim, action_dim = 4, 6, 4

        def __init__(self):
            self.inner = StatefulLaneEnv()

        def reset(self):
            return self.inner.reset()

        def step(self, actions):
            return self.inner.step(actions)

    learner = BUILDERS["ppo"](env=Bare())
    one_round(learner)
    with pytest.raises(NotImplementedError, match="save_state"):
        learner.save(str(tmp_path / "x.m3ck"), level="full")
    learner.save(str(tmp_path / "x.m3ck"))  # the optimizer level still works


def test_a_full_save_between_collect_and_update_is_refused(tmp_path):
    learner = BUILDERS["ppo"]()
    learner.collect()
    with pytest.raises(ValueError, match="update"):
        learner.save(str(tmp_path / "pending.m3ck"), level="full")
    learner.update(epochs=1)
    learner.save(str(tmp_path / "done.m3ck"), level="full")


def test_full_state_needs_the_binary_format(tmp_path):
    learner = BUILDERS["imitation"]()
    one_round(learner)
    with pytest.raises(NotImplementedError, match="m3ck"):
        learner.save(str(tmp_path / "full.json"), level="full")


def test_an_unknown_level_is_refused(tmp_path):
    learner = BUILDERS["imitation"]()
    with pytest.raises(ValueError, match="level"):
        learner.save(str(tmp_path / "x.m3ck"), level="everything")


@pytest.mark.parametrize("kind", ["ppo", "imitation"])
def test_environment_bytes_it_refuses_leave_the_learner_unchanged(tmp_path, kind):
    build = BUILDERS[kind]
    source = build()
    for _ in range(2):
        one_round(source)
    path = tmp_path / "source.m3ck"
    source.save(str(path), level="full")

    class Refusing(StatefulLaneEnv):
        def load_state(self, data):
            raise ValueError("these bytes are not mine")

    target = build(env=Refusing(seed=3))
    twin = build(env=StatefulLaneEnv(seed=3))
    one_round(target)
    one_round(twin)
    with pytest.raises(ValueError, match="not mine"):
        target.load_checkpoint(str(path))
    assert target.rounds == twin.rounds == 1
    # Weights, optimizer, rollout state and environment all as they were: the
    # next round is the one the twin, which never tried to load, takes.
    a, b = one_round(target), one_round(twin)
    assert target.env.log == twin.env.log
    for key, value in b.items():
        assert a[key] == value or close(a[key], value), key


def test_a_corrupt_built_in_environment_state_is_refused(tmp_path):
    env = m3.RecallEnv(4, symbols=4, horizon=3, seed=1)
    policy = m3.Policy(m3.PolicyConfig(env.obs_dim, 4, d_model=16, n_layers=1, n_heads=1,
                                       head_dim=16, d_state=4, chunk_size=2, seed=7))
    source = m3.PpoLearner(policy, env, steps=4, temperature=1.0, seed=2)
    source.round(epochs=1)
    path = tmp_path / "recall.m3ck"
    source.save(str(path), level="full")
    data = bytearray(path.read_bytes())
    tag = data.index(b"M3RECALL")
    data[tag + 8] ^= 0xFF  # the layout version
    corrupt = tmp_path / "corrupt.m3ck"
    corrupt.write_bytes(bytes(data))

    target = m3.PpoLearner(m3.Policy(policy.config), m3.RecallEnv(4, symbols=4, horizon=3, seed=1),
                           steps=4, temperature=1.0, seed=2)
    with pytest.raises(ValueError, match="version"):
        target.load_checkpoint(str(corrupt))
    assert target.rounds == 0
    assert target.load_checkpoint(str(path))["level"] == "full"


def test_the_built_in_environment_continues_exactly(tmp_path):
    def build():
        env = m3.RecallEnv(8, symbols=4, horizon=5, seed=1)
        policy = m3.Policy(m3.PolicyConfig(env.obs_dim, 4, d_model=16, n_layers=1, n_heads=1,
                                           head_dim=16, d_state=4, chunk_size=2, seed=7))
        return m3.PpoLearner(policy, env, steps=3, temperature=1.0, seed=2)

    continuous = build()
    for _ in range(2):
        continuous.round(epochs=1)
    expected = [continuous.round(epochs=1) for _ in range(2)]

    first = build()
    for _ in range(2):
        first.round(epochs=1)
    path = tmp_path / "recall.m3ck"
    first.save(str(path), level="full")
    second = build()
    second.load_checkpoint(str(path))
    got = [second.round(epochs=1) for _ in range(2)]
    for a, b in zip(got, expected):
        assert close(a.loss, b.loss) and close(a.entropy, b.entropy)
        assert a.episode_return == b.episode_return or close(a.episode_return, b.episode_return)


def test_checkpoints_from_before_levels_read_as_the_optimizer_level(tmp_path):
    source = BUILDERS["imitation"]()
    one_round(source)
    path = tmp_path / "old.json"
    source.save(str(path))
    data = json.loads(path.read_text())
    data["metadata"].pop("contents", None)
    data["metadata"]["continuation"] = {"exact": True, "notes": []}
    path.write_text(json.dumps(data))

    learner = BUILDERS["imitation"]()
    summary = learner.load_checkpoint(str(path))
    assert summary["level"] == "optimizer"
    assert learner.continuation == {"level": "optimizer", "notes": []}

    data["metadata"]["continuation"] = {"exact": False, "notes": ["kept live lr_schedule"]}
    path.write_text(json.dumps(data))
    learner = BUILDERS["imitation"]()
    learner.load_checkpoint(str(path))
    assert learner.continuation["level"] == "warm"


def test_a_fresh_learner_is_a_full_continuation():
    assert BUILDERS["ppo"]().continuation == {"level": "full", "notes": []}


def test_level_full_on_a_checkpoint_without_rollout_state_is_refused(tmp_path):
    source = BUILDERS["imitation"]()
    one_round(source)
    path = tmp_path / "optimizer.m3ck"
    source.save(str(path))
    learner = BUILDERS["imitation"]()
    with pytest.raises(ValueError, match="rollout"):
        learner.load_checkpoint(str(path), level="full")
    assert learner.load_checkpoint(str(path))["level"] == "optimizer"
