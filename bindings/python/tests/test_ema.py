"""T1: an exponential moving average of the policy's weights.

`learner.ema_policy` is a second policy the learner averages the trained weights
into after every optimizer step, on the device. These tests hold it to the
documented recurrence, to the invariant that it changes nothing about training,
and to the checkpoint semantics every other piece of learner state follows.

Comparisons with a NumPy recomputation are exact on the CPU wheel. On a GPU the
shader compiler may contract `e + c * (p - e)` into a fused multiply-add, so there
they are held to `|delta| <= 4 * eps * max(1, |x|)` per value, and a test that
needed the budget says so. Comparisons of the device with itself are exact on
every backend, with the matmul kernel pinned where a run crosses processes.
"""

import json
import math
import os

import numpy as np
import pytest

import mamba3_rl as m3
from continuation_env import BUILDERS, StatefulLaneEnv, one_round
from weights import entries, fingerprint, section

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "..", "..", "..", "tests", "golden", "ema_parity.json")
PINNED_KERNEL = None if m3.backend() == "cpu" else "block_tiled"
KINDS = ["ppo", "imitation"]


@pytest.fixture(autouse=True)
def pinned_kernel():
    if PINNED_KERNEL is None:
        yield
        return
    m3.set_matmul_kernel(PINNED_KERNEL)
    try:
        yield
    finally:
        m3.set_matmul_kernel("auto")


def learner(kind, env=None, **overrides):
    return BUILDERS[kind](env=env, **overrides)


def advance(learner_):
    """One round, reduced to plain data; for PPO, two optimizer steps."""
    return one_round(learner_)


def shadow(learner_, path):
    learner_.ema_policy.save(str(path))
    return entries(str(path))


def assert_twin(got, want, what):
    """Exact on the CPU wheel; within 4 * eps * max(1, |x|) elsewhere."""
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    assert got.shape == want.shape, what
    differ = got.view(np.uint32) != want.view(np.uint32)
    if m3.backend() == "cpu":
        assert not differ.any(), f"{what}: {int(differ.sum())} values differ from NumPy"
        return 0
    budget = 4 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(want))
    assert (np.abs(got - want) <= budget).all(), f"{what}: over the budget"
    return int(differ.sum())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_ema_config_validation():
    for bad in [math.nan, math.inf, -math.inf, -0.1, 1.1]:
        with pytest.raises(ValueError, match="decay"):
            m3.EmaConfig(bad)
    with pytest.raises(TypeError):
        m3.EmaConfig("0.9")
    with pytest.raises(ValueError, match="warm-up"):
        m3.EmaConfig(0.9, warmup="x")
    for config in [m3.EmaConfig(0.0), m3.EmaConfig(1.0), m3.EmaConfig(0.9),
                   m3.EmaConfig(0.999, warmup="tf")]:
        assert eval(repr(config), {"EmaConfig": m3.EmaConfig}) == config
    config = m3.EmaConfig(0.99, warmup="tf")
    assert (config.decay, config.warmup) == (pytest.approx(0.99), "tf")
    assert m3.EmaConfig(0.99) != config
    assert m3.EmaConfig(0.99).warmup == "none"


@pytest.mark.parametrize("kind", KINDS)
def test_no_ema_is_the_default(kind, tmp_path):
    plain = learner(kind)
    assert plain.ema_policy is None
    assert plain.ema_config is None
    assert plain.ema_updates == 0
    with pytest.raises(ValueError, match="no EMA"):
        plain.reset_ema()
    advance(plain)
    path = tmp_path / "plain.json"
    plain.save(str(path))
    data = json.loads(path.read_text())
    assert "ema" not in data and "ema_updates" not in data
    assert data["metadata"]["trainer_config"]["ema"] is None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_ema_does_not_change_training(kind):
    plain = learner(kind)
    averaged = learner(kind, ema=m3.EmaConfig(0.9))
    for _ in range(3):
        with_average = advance(averaged)
        assert with_average.pop("ema_updates") > 0
        with_average.pop("ema_fingerprint")
        assert with_average == advance(plain)
    assert averaged.policy.fingerprint() == plain.policy.fingerprint()


@pytest.mark.parametrize("warmup", ["none", "tf"])
@pytest.mark.parametrize("kind", KINDS)
def test_recurrence_matches_numpy(kind, warmup, tmp_path):
    config = m3.EmaConfig(0.9, warmup=warmup)
    run = learner(kind, ema=config)
    ema = shadow(run, tmp_path / "ema.json")
    run.policy.save(str(tmp_path / "theta.json"))
    assert fingerprint(ema) == fingerprint(entries(str(tmp_path / "theta.json")))
    inexact = 0
    for _ in range(3):
        if kind == "ppo":
            run.collect()
            updates = [lambda: run.update(epochs=1), lambda: run.update(epochs=1)]
        else:
            updates = [lambda: run.round()]
        for update in updates:
            before = run.ema_updates
            stats = update()
            assert run.ema_updates == before + 1
            step = stats.optimizer_steps
            decay = np.float32(config.decay)
            if warmup == "tf":
                decay = min(decay, np.float32((1 + step) / (10 + step)))
            c = np.float32(1) - np.float32(decay)
            run.policy.save(str(tmp_path / "theta.json"))
            theta = entries(str(tmp_path / "theta.json"))
            got = shadow(run, tmp_path / "ema.json")
            for name, e in ema.items():
                want = e + c * (theta[name] - e)
                inexact += assert_twin(got[name], want, f"{name} at step {step}")
            # One-step twin: the next step starts from what the device holds.
            ema = got
    if inexact:
        print(f"test_recurrence_matches_numpy: {inexact} values within budget, not exact, "
              f"on {m3.backend()}")
    assert fingerprint(ema) != run.policy.fingerprint()


def test_rust_python_parity(tmp_path):
    """The scenario `tests/train_ema_parity.rs` runs in Rust, from Python."""
    with open(GOLDEN) as f:
        golden = json.load(f)
    want = golden["backends"].get(m3.backend())
    if want is None:
        pytest.skip(f"no Rust parity record for {m3.backend()}; see {GOLDEN}")
    scenario = golden["scenario"]
    env = m3.RecallEnv(num_envs=scenario["envs"], symbols=scenario["symbols"],
                       horizon=scenario["horizon"], seed=scenario["env_seed"])
    policy = m3.Policy(m3.PolicyConfig(
        env.obs_dim, env.action_dim, d_model=scenario["d_model"], n_layers=scenario["n_layers"],
        n_heads=scenario["n_heads"], head_dim=scenario["head_dim"], d_state=scenario["d_state"],
        chunk_size=scenario["chunk_size"], seed=scenario["policy_seed"]))
    run = m3.PpoLearner(policy, env, steps=scenario["steps"],
                        learning_rate=scenario["learning_rate"],
                        ema=m3.EmaConfig(scenario["decay"]))
    for _ in range(scenario["rounds"]):
        run.round(epochs=scenario["epochs"])
    ema = shadow(run, tmp_path / "ema.json")
    assert run.ema_updates == want["ema_updates"]
    assert run.policy.fingerprint() == want["policy_fingerprint"]
    assert run.ema_policy.fingerprint() == want["ema_fingerprint"] == fingerprint(ema)
    values = np.concatenate([ema[name].reshape(-1) for name in sorted(ema, key=str.encode)])
    assert values[:8].view(np.uint32).tolist() == want["first_values_bits"]


@pytest.mark.parametrize("kind", KINDS)
def test_ema_policy_is_usable_and_untrained(kind, tmp_path):
    run = learner(kind, ema=m3.EmaConfig(0.9))
    for _ in range(2):
        advance(run)
    averaged = run.ema_policy
    before = averaged.fingerprint()
    score = m3.evaluate(averaged, StatefulLaneEnv(seed=3), steps=8)
    assert score is None or np.isfinite(score)
    env = StatefulLaneEnv(seed=4)
    rollout = m3.Rollout(averaged, num_envs=env.num_envs, temperature=0.0)
    rollout.step(env.reset(), action_mask=env.action_mask())
    assert averaged.fingerprint() == before
    assert run.policy.fingerprint() != before
    path = tmp_path / "ema_policy.json"
    averaged.save(str(path))
    assert m3.Policy.load(str(path)).fingerprint() == before
    # The handle follows the learner: it is the average, not a copy of it.
    advance(run)
    assert averaged.fingerprint() != before
    assert averaged.fingerprint() == run.ema_policy.fingerprint()


@pytest.mark.parametrize("kind", KINDS)
def test_reset_ema(kind):
    run = learner(kind, ema=m3.EmaConfig(0.9))
    for _ in range(2):
        advance(run)
    assert run.ema_updates > 0
    assert run.ema_policy.fingerprint() != run.policy.fingerprint()
    run.reset_ema()
    assert run.ema_policy.fingerprint() == run.policy.fingerprint()
    assert run.ema_updates == 0
    if kind == "ppo":
        advance(run)
        run.collect()
        averaged, updates = run.ema_policy.fingerprint(), run.ema_updates
        with pytest.raises(ValueError, match="update"):
            run.reset_ema()
        assert (run.ema_policy.fingerprint(), run.ema_updates) == (averaged, updates)
        run.update(epochs=1)
        run.reset_ema()
        assert run.ema_updates == 0


def test_freeze_keeps_frozen_shadow_equal(tmp_path):
    run = learner("ppo", ema=m3.EmaConfig(0.9))
    run.policy.freeze(["critic"])
    for _ in range(2):
        advance(run)
    ema = shadow(run, tmp_path / "ema.json")
    run.policy.save(str(tmp_path / "theta.json"))
    theta = entries(str(tmp_path / "theta.json"))
    critic = [n for n in theta if n.startswith("critic")]
    assert critic
    for name in theta:
        same = ema[name].view(np.uint32).tolist() == theta[name].view(np.uint32).tolist()
        assert same == (name in critic), name


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["optimizer", "full"])
@pytest.mark.parametrize("kind", KINDS)
def test_ema_saved_at_both_levels(kind, level, tmp_path):
    run = learner(kind, ema=m3.EmaConfig(0.9, warmup="tf"))
    for _ in range(2):
        advance(run)
    path = tmp_path / "run.m3ck"
    run.save(str(path), level=level)
    restored = learner(kind, ema=m3.EmaConfig(0.9, warmup="tf"))
    summary = restored.load_checkpoint(str(path))
    assert summary["level"] == level
    assert restored.ema_policy.fingerprint() == run.ema_policy.fingerprint()
    assert restored.ema_updates == run.ema_updates > 0
    assert restored.ema_config == run.ema_config
    # JSON holds it too, at the optimizer level.
    if level == "optimizer":
        json_path = tmp_path / "run.json"
        run.save(str(json_path))
        data = json.loads(json_path.read_text())
        assert data["ema_updates"] == run.ema_updates
        assert fingerprint(section(str(json_path), "ema")) == run.ema_policy.fingerprint()
        assert data["metadata"]["trainer_config"]["ema"] == {"decay": pytest.approx(0.9),
                                                               "warmup": "tf"}


@pytest.mark.parametrize("kind", KINDS)
def test_verify_rejects_ema_mismatch(kind, tmp_path):
    saved = learner(kind, ema=m3.EmaConfig(0.9))
    advance(saved)
    with_ema = tmp_path / "with.m3ck"
    saved.save(str(with_ema))
    plain = learner(kind)
    advance(plain)
    without_ema = tmp_path / "without.m3ck"
    plain.save(str(without_ema))

    cases = [
        (learner(kind, ema=m3.EmaConfig(0.95)), with_ema, "ema.decay"),
        (learner(kind, ema=m3.EmaConfig(0.9, warmup="tf")), with_ema, "ema.warmup"),
        (learner(kind), with_ema, "ema"),
        (learner(kind, ema=m3.EmaConfig(0.9)), without_ema, "ema"),
    ]
    for target, path, listed in cases:
        before = (target.policy.fingerprint(), target.rounds, target.ema_updates,
                  target.ema_policy.fingerprint() if target.ema_policy else None)
        with pytest.raises(ValueError, match=listed.replace(".", r"\.")):
            target.load_checkpoint(str(path))
        after = (target.policy.fingerprint(), target.rounds, target.ema_updates,
                 target.ema_policy.fingerprint() if target.ema_policy else None)
        assert after == before


@pytest.mark.parametrize("kind", KINDS)
def test_checkpoint_mode_adopts_ema(kind, tmp_path):
    saved = learner(kind, ema=m3.EmaConfig(0.95, warmup="tf"))
    for _ in range(2):
        advance(saved)
    path = tmp_path / "saved.m3ck"
    saved.save(str(path))
    for target in [learner(kind, ema=m3.EmaConfig(0.5)), learner(kind)]:
        summary = target.load_checkpoint(str(path), config="checkpoint")
        assert summary["config"] == "adopted"
        assert any(note.startswith("adopted ema") for note in summary["notes"]), summary
        assert target.ema_config == m3.EmaConfig(0.95, warmup="tf")
        assert target.ema_policy.fingerprint() == saved.ema_policy.fingerprint()
        assert target.ema_updates == saved.ema_updates
    # A learner with an EMA cannot adopt "none".
    plain = learner(kind)
    advance(plain)
    plain.save(str(tmp_path / "plain.m3ck"))
    target = learner(kind, ema=m3.EmaConfig(0.5))
    with pytest.raises(ValueError, match="ema"):
        target.load_checkpoint(str(tmp_path / "plain.m3ck"), config="checkpoint")


@pytest.mark.parametrize("kind", KINDS)
def test_live_mode_reseeds_and_marks_warm(kind, tmp_path):
    plain = learner(kind)
    for _ in range(2):
        advance(plain)
    path = tmp_path / "plain.m3ck"
    plain.save(str(path))
    target = learner(kind, ema=m3.EmaConfig(0.9))
    advance(target)
    summary = target.load_checkpoint(str(path), config="live")
    assert summary["level"] == "warm"
    assert "ema re-seeded from loaded weights" in summary["notes"]
    assert target.continuation["level"] == "warm"
    assert target.ema_policy.fingerprint() == plain.policy.fingerprint()
    assert target.ema_updates == 0

    averaged = learner(kind, ema=m3.EmaConfig(0.9))
    advance(averaged)
    path = tmp_path / "averaged.m3ck"
    averaged.save(str(path))
    target = learner(kind)
    summary = target.load_checkpoint(str(path), config="live")
    assert summary["level"] == "warm"
    assert any("ema ignored" in note for note in summary["notes"]), summary
    assert target.ema_policy is None


@pytest.mark.parametrize("kind", KINDS)
def test_from_checkpoint_rebuilds_ema(kind, tmp_path):
    run = learner(kind, ema=m3.EmaConfig(0.97, warmup="tf"))
    for _ in range(2):
        advance(run)
    path = tmp_path / "run.m3ck"
    run.save(str(path), level="full")
    cls = m3.PpoLearner if kind == "ppo" else m3.ImitationLearner
    rebuilt = cls.from_checkpoint(str(path), StatefulLaneEnv(seed=99), steps=5)
    assert rebuilt.ema_config == m3.EmaConfig(0.97, warmup="tf")
    assert rebuilt.ema_policy.fingerprint() == run.ema_policy.fingerprint()
    assert rebuilt.ema_updates == run.ema_updates
    assert rebuilt.continuation["level"] == "full"
