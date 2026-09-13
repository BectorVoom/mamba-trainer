"""K1/K2: a learner checkpoint restores all or nothing, and knows the
configuration it was trained under.

Every learner here acts greedily over a deterministic environment. Restored
state (weights, moments, counters) is compared exactly; state produced by
further *training* is compared to 1e-5, because the CPU backend's parallel
reductions are not bit-reproducible run to run (two identical 20-update runs
differ by ~1e-7, while a genuinely different trajectory differs by ~7e-2).
"""

import json

import numpy as np
import pytest

import mamba3_rl as m3

NUM_ENVS = 4
ACTIONS = 4


class LaneEnv:
    """Deterministic: each lane has its own horizon, the observation is a fixed
    function of lane, clock and episode, and `reset` really starts over."""

    num_envs = NUM_ENVS
    obs_dim = 6
    action_dim = ACTIONS

    def __init__(self):
        self.reset()

    def _observation(self):
        lanes = np.arange(self.num_envs)[:, None]
        k = np.arange(self.obs_dim)[None, :]
        phase = lanes * 7 + self.clock[:, None] * 3 + self.episode[:, None] * 5 + k * 11
        return np.sin(phase * 0.61).astype(np.float32)

    def reset(self):
        self.clock = np.zeros(self.num_envs, dtype=np.int64)
        self.episode = np.zeros(self.num_envs, dtype=np.int64)
        return self._observation()

    def expert_actions(self):
        return (np.arange(self.num_envs) + self.clock) % self.action_dim

    def step(self, actions):
        reward = (actions == self.expert_actions()).astype(np.float32)
        self.clock += 1
        done = self.clock >= 3 + np.arange(self.num_envs) % 3
        self.episode[done] += 1
        self.clock[done] = 0
        return self._observation(), reward, done.astype(np.float32)


def small_policy(seed=7):
    return m3.Policy(
        m3.PolicyConfig(LaneEnv.obs_dim, ACTIONS, d_model=16, n_layers=1, n_heads=1,
                        head_dim=16, d_state=4, chunk_size=2, seed=seed)
    )


def ppo(policy=None, **overrides):
    settings = dict(steps=4, learning_rate=0.008, temperature=0.0, seed=5)
    settings.update(overrides)
    return m3.PpoLearner(policy or small_policy(), LaneEnv(), **settings)


def imitation(policy=None, **overrides):
    settings = dict(steps=4, learning_rate=0.008, temperature=0.0, seed=5)
    settings.update(overrides)
    return m3.ImitationLearner(policy or small_policy(), LaneEnv(), **settings)


BUILDERS = {"ppo": ppo, "imitation": imitation}


def advance(learner, rounds=1):
    for _ in range(rounds):
        if isinstance(learner, m3.PpoLearner):
            learner.round(epochs=1)
        else:
            learner.round(agreement=False)


def snapshot(learner, path):
    """Everything a learner checkpoint holds, as plain data: weights, optimizer
    moments, counters, configuration."""
    learner.save(str(path))
    return json.loads(path.read_text())


def next_rate(learner):
    return learner.round(epochs=1).learning_rate if isinstance(learner, m3.PpoLearner) \
        else learner.round(agreement=False).learning_rate


def assert_close(a, b, what, tol=1e-5):
    """Two `{"entries": {name: {"shape", "data"}}}` dicts agree to `tol`."""
    assert a["entries"].keys() == b["entries"].keys(), what
    for name, entry in a["entries"].items():
        other = b["entries"][name]
        assert entry["shape"] == other["shape"], f"{what}: {name}"
        worst = max((abs(x - y) for x, y in zip(entry["data"], other["data"])), default=0.0)
        assert worst < tol, f"{what}: {name} differs by {worst}"


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# K1: all or nothing
# ---------------------------------------------------------------------------


def _corruptions(tmp_path, source):
    good = tmp_path / "good.json"
    source.save(str(good))

    def optimizer_entries(data):
        return data["optimizer"]["entries"]

    def short(data):
        entry = next(iter(optimizer_entries(data).values()))
        entry["data"] = entry["data"][:-1]

    def reshaped(data):
        entry = next(iter(optimizer_entries(data).values()))
        entry["shape"] = entry["shape"] + [1]

    def unknown(data):
        optimizer_entries(data)["no.such.parameter.m"] = {"shape": [1], "data": [0.0]}

    def weights_short(data):
        entry = list(data["state"]["entries"].values())[-1]
        entry["data"] = entry["data"][:-1]

    def rounds_negative(data):
        data["metadata"]["rounds"] = -1

    weights_only = tmp_path / "weights_only.json"
    source.policy.save(str(weights_only))
    cases = {"weights only": weights_only}
    for name, change in [("short optimizer entry", short), ("wrong optimizer shape", reshaped),
                         ("unexpected parameter path", unknown), ("short weights", weights_short),
                         ("negative rounds", rounds_negative)]:
        target = tmp_path / f"{name.replace(' ', '_')}.json"
        target.write_text(good.read_text())
        cases[name] = edit(target, change)
    return good, cases


@pytest.mark.parametrize("kind", ["ppo", "imitation"])
def test_a_failed_strict_load_changes_nothing(tmp_path, kind):
    build = BUILDERS[kind]
    schedule = m3.LrSchedule.step(1, 0.5)
    source = build(policy=small_policy(1), lr_schedule=schedule)
    advance(source, 3)
    good, cases = _corruptions(tmp_path, source)

    target = build(policy=small_policy(2), lr_schedule=schedule)
    advance(target, 2)
    before = snapshot(target, tmp_path / "before.json")
    for name, path in cases.items():
        with pytest.raises(ValueError):
            target.load_checkpoint(str(path), strict=True)
        after = snapshot(target, tmp_path / "after.json")
        assert after == before, f"{name}: a failed load changed the learner"

    # The next update is exactly the one it would have taken anyway.
    assert target.rounds == 2
    assert next_rate(target) == pytest.approx(schedule.rate_at(0.008, 3), abs=0)

    # And a good checkpoint still loads after all those failures. Binary, so the
    # comparison below is bit-exact rather than through decimal text.
    binary = tmp_path / "good.m3ck"
    source.save(str(binary))
    summary = target.load_checkpoint(str(binary))
    assert summary == {"weights": True, "optimizer": True, "counters": True,
                       "config": "verified", "exact": True, "notes": []}
    assert target.rounds == 3
    assert snapshot(target, tmp_path / "loaded.json")["state"] == json.loads(good.read_text())["state"]


@pytest.mark.parametrize("kind", ["ppo", "imitation"])
def test_a_weights_only_warm_start_restarts_the_optimizer_and_counters(tmp_path, kind):
    build = BUILDERS[kind]
    weights = tmp_path / "weights.json"
    small_policy(3).save(str(weights))
    learner = build()
    advance(learner, 2)
    summary = learner.load_checkpoint(str(weights), strict=False)
    assert summary["weights"] and not summary["optimizer"] and not summary["counters"]
    assert summary["config"] == "warm_start" and not summary["exact"]
    assert learner.rounds == 0
    assert learner.continuation["exact"] is False
    stats = learner.round(epochs=1) if kind == "ppo" else learner.round(agreement=False)
    assert stats.optimizer_steps == 1


def test_loading_drops_the_window_collected_under_the_old_weights(tmp_path):
    source = ppo()
    advance(source)
    path = tmp_path / "ppo.json"
    source.save(str(path))
    learner = ppo()
    learner.collect()
    learner.load_checkpoint(str(path))
    with pytest.raises(ValueError, match="nothing has been collected"):
        learner.update()


def test_an_unreadable_checkpoint_is_an_os_error(tmp_path):
    with pytest.raises(OSError):
        ppo().load_checkpoint(str(tmp_path / "missing.m3ck"))


def test_an_unknown_config_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="config must be"):
        ppo().load_checkpoint(str(tmp_path / "x.json"), config="maybe")


# ---------------------------------------------------------------------------
# K2: the training configuration travels with the checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def decayed(tmp_path):
    """The review probe: base rate 0.008 halving every step, one round in."""
    source = ppo(lr_schedule=m3.LrSchedule.step(1, 0.5))
    source.round(epochs=1)
    path = tmp_path / "scheduled.m3ck"
    source.save(str(path))
    expected = source.round(epochs=1).learning_rate
    assert expected == pytest.approx(0.002)
    return path, expected


def test_a_matching_learner_resumes_on_schedule(decayed):
    path, expected = decayed
    restored = ppo(lr_schedule=m3.LrSchedule.step(1, 0.5))
    assert restored.load_checkpoint(str(path))["config"] == "verified"
    assert restored.round(epochs=1).learning_rate == expected


def test_a_different_configuration_is_refused_listing_every_difference(decayed):
    path, _ = decayed
    restored = ppo(betas=(0.8, 0.999), ppo=m3.PpoConfig(clip_coeff=0.3))
    with pytest.raises(ValueError) as caught:
        restored.load_checkpoint(str(path))
    message = str(caught.value)
    for field in ("lr_schedule", "optimizer.beta1", "algorithm.clip_coeff"):
        assert field in message, f"{field} missing from: {message}"
    assert restored.rounds == 0


def test_config_checkpoint_adopts_the_saved_settings(decayed):
    path, expected = decayed
    restored = ppo(betas=(0.8, 0.999), ppo=m3.PpoConfig(clip_coeff=0.3))
    summary = restored.load_checkpoint(str(path), config="checkpoint")
    assert summary["config"] == "adopted" and summary["exact"]
    assert any("lr_schedule" in note for note in summary["notes"])
    assert restored.config.clip_coeff == pytest.approx(m3.PpoConfig().clip_coeff)
    assert restored.round(epochs=1).learning_rate == expected


def test_config_checkpoint_never_adopts_the_architecture(decayed):
    path, _ = decayed
    wider = m3.Policy(m3.PolicyConfig(LaneEnv.obs_dim, ACTIONS, d_model=16, n_layers=1, n_heads=1,
                                      head_dim=16, d_state=4, chunk_size=4, seed=7))
    with pytest.raises(ValueError, match="policy"):
        ppo(policy=wider).load_checkpoint(str(path), config="checkpoint")


def test_config_live_keeps_live_settings_and_says_so(decayed, tmp_path):
    path, _ = decayed
    restored = ppo()
    summary = restored.load_checkpoint(str(path), config="live")
    assert summary["config"] == "live" and not summary["exact"]
    assert restored.round(epochs=1).learning_rate == pytest.approx(0.008)
    assert restored.continuation["exact"] is False

    # The verdict travels: a later save records it, and a matching load keeps it.
    again = tmp_path / "again.json"
    restored.save(str(again))
    follower = ppo()
    assert follower.load_checkpoint(str(again))["config"] == "verified"
    assert follower.continuation["exact"] is False
    assert follower.continuation["notes"]


def test_a_legacy_learner_checkpoint_needs_an_explicit_non_exact_mode(tmp_path):
    source = ppo()
    advance(source, 2)
    path = tmp_path / "legacy.json"
    source.save(str(path))
    edit(path, lambda data: data.__setitem__("metadata", {"rounds": 2}))

    with pytest.raises(ValueError, match="config='live'"):
        ppo().load_checkpoint(str(path))
    with pytest.raises(ValueError, match="config='live'"):
        ppo().load_checkpoint(str(path), config="checkpoint")
    learner = ppo()
    summary = learner.load_checkpoint(str(path), config="live")
    assert summary["config"] == "legacy" and not summary["exact"]
    assert learner.rounds == 2


def test_a_different_reference_is_a_configuration_difference(tmp_path):
    config = m3.PpoConfig(reference_coeff=0.5)
    source = ppo(ppo=config, reference=small_policy(11))
    advance(source)
    path = tmp_path / "anchored.json"
    source.save(str(path))

    with pytest.raises(ValueError, match="reference.weights_fnv1a64"):
        ppo(ppo=config, reference=small_policy(12)).load_checkpoint(str(path))
    with pytest.raises(ValueError, match="reference"):
        ppo(ppo=config, reference=small_policy(12)).load_checkpoint(str(path), config="checkpoint")
    assert ppo(ppo=config, reference=small_policy(11)).load_checkpoint(str(path))["exact"]


@pytest.mark.parametrize("kind", ["ppo", "imitation"])
def test_halfway_save_and_load_continues_exactly_like_one_run(tmp_path, kind):
    build = BUILDERS[kind]
    schedule = m3.LrSchedule.cosine(40, warmup_steps=2)
    # The sampling RNG is not part of a checkpoint (A2b), so every draw here is
    # made deterministic instead: greedy actions, and for DAgger a schedule
    # whose mixture is a certainty from round 1 on.
    extra = {} if kind == "ppo" else dict(schedule=m3.DaggerSchedule.only_first())

    continuous = build(lr_schedule=schedule, **extra)
    advance(continuous, 10)
    continuous.reset()
    advance(continuous, 10)

    first = build(lr_schedule=schedule, **extra)
    advance(first, 10)
    path = tmp_path / f"{kind}.m3ck"
    first.save(str(path))
    second = build(policy=small_policy(99), lr_schedule=schedule, **extra)
    second.load_checkpoint(str(path))
    second.reset()
    advance(second, 10)

    a = snapshot(continuous, tmp_path / "continuous.json")
    b = snapshot(second, tmp_path / "second.json")
    assert a["step"] == b["step"] == 20
    assert a["optimizer_steps"] == b["optimizer_steps"] == 20
    assert_close(a["state"], b["state"], "weights across the save/load boundary")
    assert_close(a["optimizer"], b["optimizer"], "optimizer moments across the save/load boundary")


@pytest.mark.parametrize("kind", ["ppo", "imitation"])
def test_from_checkpoint_rebuilds_the_learner_it_saved(tmp_path, kind):
    build = BUILDERS[kind]
    schedule = m3.LrSchedule.step(2, 0.5)
    extra = dict(ppo=m3.PpoConfig(clip_coeff=0.25)) if kind == "ppo" else \
        dict(schedule=m3.DaggerSchedule.fixed(0.25), entropy_bonus=0.02)
    source = build(lr_schedule=schedule, betas=(0.8, 0.99), eps=1e-6, max_grad_norm=0.7, **extra)
    advance(source, 3)
    path = tmp_path / f"{kind}.m3ck"
    source.save(str(path))

    cls = m3.PpoLearner if kind == "ppo" else m3.ImitationLearner
    rebuilt = cls.from_checkpoint(str(path), LaneEnv(), steps=4, temperature=0.0, seed=5)
    assert rebuilt.rounds == 3
    assert rebuilt.continuation["exact"]
    assert snapshot(rebuilt, tmp_path / "rebuilt.json")["metadata"] == \
        snapshot(source, tmp_path / "source.json")["metadata"]
    assert next_rate(rebuilt) == next_rate(source)


def test_from_checkpoint_refuses_the_other_learners_checkpoint(tmp_path):
    source = imitation()
    advance(source)
    path = tmp_path / "imitation.json"
    source.save(str(path))
    with pytest.raises(ValueError, match="imitation"):
        m3.PpoLearner.from_checkpoint(str(path), LaneEnv())


# ---------------------------------------------------------------------------
# K7 from Python: reset() forgets the actor's and the reference's histories
# ---------------------------------------------------------------------------


def test_reset_starts_both_recurrent_histories_over(tmp_path):
    config = m3.PpoConfig(reference_coeff=0.5)
    reference = small_policy(21)
    trained = ppo(ppo=config, reference=reference)
    advance(trained, 3)
    trained.reset()

    weights = tmp_path / "weights.m3ck"
    trained.policy.save(str(weights))
    fresh = ppo(policy=m3.Policy.load(str(weights)), ppo=config, reference=reference)

    a, b = trained.round(epochs=1), fresh.round(epochs=1)
    assert (a.reference_kl, a.approx_kl, a.loss) == (b.reference_kl, b.approx_kl, b.loss)
    assert a.reference_kl > 0.0
