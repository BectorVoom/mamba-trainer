"""PPO for the entity model (ENTITY_RL_PLAN.md R2), mirroring tests/entity_rl.rs."""

import numpy as np
import pytest

import mamba3_rl as m3


def tiny_spec():
    # 6 context entities, 2 queries, K = 2: a pointer plus a conditioned
    # categorical (both actions) plus a multilabel head (not an action).
    return m3.EntityModelSpec(
        globals=0,
        context=[m3.ContextSet("cells", count=6, features=4)],
        queries=m3.QuerySet("agents", count=2, features=3, steps=2, autoregressive_on="pick"),
        heads=[
            m3.Head.pointer("pick", set="cells"),
            m3.Head.categorical("cfg", classes=3, condition_on="pick"),
            m3.Head.multilabel("flags", labels=2),
        ],
        d_model=16,
        context_layers=1,
        decoder_layers=1,
        seed=11,
    )


def plain_obs(b, seed=31):
    rng = np.random.default_rng(seed)
    return {
        "cells": rng.normal(size=(b, 6, 4)).astype(np.float32),
        "agents": rng.normal(size=(b, 2, 3)).astype(np.float32),
    }


def test_act_shapes_and_absent_minus_one():
    spec = tiny_spec()
    policy = m3.EntityPolicy(spec)
    obs = plain_obs(3)
    obs["agents.presence"] = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.float32)
    out = policy.act(obs)
    assert set(out["actions"]) == {"pick", "cfg"}
    for head, ids in out["actions"].items():
        assert ids.shape == (3, 2, 2), head
        assert ids.dtype == np.int64, head
    assert out["log_prob"].shape == (3, 2, 2)
    assert out["log_prob"].dtype == np.float32
    assert out["value"].shape == (3,)
    # Query 1 of sample 1 is absent: it never acts, anywhere.
    for head in ("pick", "cfg"):
        assert np.all(out["actions"][head][1, 1] == -1)
    assert np.all(out["log_prob"][1, 1] == 0.0)
    # Present cells hold real ids.
    assert np.all(out["actions"]["pick"][:, 0] >= 0)
    assert np.all((out["actions"]["cfg"][:, 0] >= 0) & (out["actions"]["cfg"][:, 0] < 3))
    # Non-action heads still report logits, shaped as predict returns them.
    assert set(out["outputs"]) == {"pick", "cfg", "flags"}
    assert out["outputs"]["pick"].shape == (3, 2, 2, 6)
    assert out["outputs"]["cfg"].shape == (3, 2, 2, 3)
    assert out["outputs"]["flags"].shape == (3, 2, 2, 2)


def test_masked_entities_never_chosen():
    spec = tiny_spec()
    policy = m3.EntityPolicy(spec)
    b, n, mm = 3, 6, 2
    rng = np.random.default_rng(41)
    presence = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    legal = rng.integers(0, 2, size=(b, mm, n)).astype(np.float32)
    for bi in range(b):
        for mi in range(mm):
            if not legal[bi, mi].any():
                legal[bi, mi, 0] = 1.0
    obs = {
        "cells": rng.normal(size=(b, n, 4)).astype(np.float32),
        "cells.presence": presence,
        "agents": rng.normal(size=(b, mm, 3)).astype(np.float32),
        "agents.presence": np.array([[1, 1], [1, 0], [1, 1]], dtype=np.float32),
        "legal.pick": legal,
    }
    for _ in range(30):
        out = policy.act(obs)
        pick = out["actions"]["pick"]
        for bi in range(b):
            for mi in range(mm):
                for j in range(2):
                    ident = pick[bi, mi, j]
                    if bi == 1 and mi == 1:
                        assert ident == -1
                        continue
                    assert 0 <= ident < n
                    assert presence[bi, ident] == 1.0, f"absent entity {ident}"
                    assert legal[bi, mi, ident] == 1.0, f"illegal entity {ident}"


def test_greedy_deterministic_and_matches_predict():
    spec = tiny_spec()
    policy = m3.EntityPolicy(spec)
    obs = plain_obs(2)
    first = policy.act(obs, greedy=True)
    second = policy.act(obs, greedy=True)
    assert np.array_equal(first["actions"]["pick"], second["actions"]["pick"])
    assert np.array_equal(first["log_prob"], second["log_prob"])
    greedy = policy.to_model().predict(obs)
    np.testing.assert_array_equal(first["actions"]["pick"], greedy["pick"]["choice"])
    np.testing.assert_allclose(
        first["outputs"]["pick"], greedy["pick"]["logits"], rtol=1e-5, atol=1e-5
    )


def test_value_shape_and_finite():
    policy = m3.EntityPolicy(tiny_spec())
    values = policy.value(plain_obs(4))
    assert values.shape == (4,)
    assert np.all(np.isfinite(values))


# ---------------------------------------------------------------------------
# Contextual bandit: one target entity per sample is marked by feature 0 = 1;
# reward is the share of queries whose step-0 pointer names it.
# ---------------------------------------------------------------------------

BANDIT_N = 6


def bandit_spec():
    return m3.EntityModelSpec(
        globals=0,
        context=[m3.ContextSet("field", count=BANDIT_N, features=2)],
        queries=m3.QuerySet("q", count=2, features=1, steps=1),
        heads=[m3.Head.pointer("pick", set="field")],
        d_model=16,
        context_layers=1,
        decoder_layers=1,
        seed=7,
    )


def bandit_obs(rng, b):
    targets = rng.integers(0, BANDIT_N, b)
    ctx = np.zeros((b, BANDIT_N, 2), dtype=np.float32)
    for bi, target in enumerate(targets):
        ctx[bi, target, 0] = 1.0
    ctx[:, :, 1] = rng.normal(size=(b, BANDIT_N)).astype(np.float32)
    return {"field": ctx, "q": np.zeros((b, 2, 1), dtype=np.float32)}, targets


def test_bandit_learns():
    policy = m3.EntityPolicy(bandit_spec(), learning_rate=3e-3)
    rng = np.random.default_rng(1234)
    b = 64
    first, done_at = None, None
    for update in range(60):
        obs, targets = bandit_obs(rng, b)
        out = policy.act(obs)
        pick = out["actions"]["pick"].reshape(b, 2)
        rewards = (pick == targets[:, None]).mean(axis=1).astype(np.float32)
        mean = float(rewards.mean())
        if first is None:
            first = mean
        if mean > 0.8:
            done_at = update
            break
        policy.update(
            obs,
            out["actions"],
            out["log_prob"],
            out["value"],
            rewards.reshape(1, b),
            np.ones((1, b), dtype=np.float32),
            np.zeros(b, dtype=np.float32),
            epochs=4,
            minibatches=2,
        )
    assert first < 0.5, f"bandit started at {first:.3f}, too good for chance"
    assert done_at is not None, f"bandit never passed 0.8 within 60 updates (from {first:.3f})"


def test_save_load_round_trip(tmp_path):
    policy = m3.EntityPolicy(tiny_spec())
    obs = plain_obs(2)
    before = policy.act(obs)
    path = str(tmp_path / "entity_policy.m3ck")
    policy.save(path)
    loaded = m3.EntityPolicy.load(path)
    after = loaded.act(obs)
    # Both took their first sample from the same stream position.
    assert np.array_equal(before["actions"]["pick"], after["actions"]["pick"])
    assert np.array_equal(before["log_prob"], after["log_prob"])
    assert np.array_equal(before["value"], after["value"])
    # Weights travel: keep training the loaded policy and stay greedy-equal.
    rollout = plain_obs(8)
    out = loaded.act(rollout)
    loaded.update(
        rollout,
        out["actions"],
        out["log_prob"],
        out["value"],
        np.zeros((2, 4), dtype=np.float32),
        np.ones((2, 4), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        epochs=1,
        minibatches=2,
    )
    greedy = loaded.act(obs, greedy=True)["actions"]["pick"]
    predict = loaded.to_model().predict(obs)["pick"]["choice"]
    np.testing.assert_array_equal(greedy, predict)


def test_from_model_keeps_greedy_predictions():
    spec = tiny_spec()
    model = m3.EntityModel(spec)
    policy = m3.EntityPolicy.from_model(model)
    obs = plain_obs(2)
    want = model.predict(obs)
    got = policy.to_model().predict(obs)
    for head in want:
        np.testing.assert_array_equal(got[head]["logits"], want[head]["logits"])


def test_unsupported_specs_raise_value_error():
    # Joint with autoregression cannot even be built: the spec refuses it.
    with pytest.raises(ValueError, match="autoregressive"):
        m3.EntityModelSpec(
            globals=0,
            context=[m3.ContextSet("cells", count=6, features=4)],
            queries=m3.QuerySet(
                "agents", count=2, features=3, steps=2, autoregressive_on="pick"
            ),
            heads=[m3.Head.pointer("pick", set="cells")],
            d_model=16,
            context_layers=1,
            decoder_layers=1,
            decoder="joint",
            seed=1,
        )
    # QueryCausal builds but the RL policy refuses it by name.
    qc = m3.EntityModelSpec(
        globals=0,
        context=[m3.ContextSet("cells", count=6, features=4)],
        queries=m3.QuerySet(
            "agents", count=2, features=3, steps=2, autoregressive_on="pick"
        ),
        heads=[m3.Head.pointer("pick", set="cells")],
        d_model=16,
        context_layers=1,
        decoder_layers=1,
        decoder="query_causal",
        seed=1,
    )
    with pytest.raises(ValueError, match="QueryCausal"):
        m3.EntityPolicy(qc)


def tiny_rollout(policy, b=8):
    """One T=2, E=4 rollout worth of validated arguments."""
    rng = np.random.default_rng(99)
    t, e, s = 2, 4, 8
    obs = {
        "cells": rng.normal(size=(s, 6, 4)).astype(np.float32),
        "agents": rng.normal(size=(s, 2, 3)).astype(np.float32),
    }
    out = policy.act(obs)
    args = dict(
        obs=obs,
        actions=out["actions"],
        log_prob=out["log_prob"],
        value=out["value"],
        reward=np.zeros((t, e), dtype=np.float32),
        done=np.ones((t, e), dtype=np.float32),
        last_value=np.zeros(e, dtype=np.float32),
    )
    return args


def test_update_validates_rollout():
    policy = m3.EntityPolicy(tiny_spec())
    good = tiny_rollout(policy)

    bad = dict(
        good,
        reward=np.zeros((3, 4), dtype=np.float32),
        done=np.ones((3, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="T\\*E"):
        policy.update(**bad, epochs=1, minibatches=2)

    bad = dict(good, done=np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="\\[T, E\\]"):
        policy.update(**bad, epochs=1, minibatches=2)

    bad = dict(good, actions={"pick": good["actions"]["pick"], "nope": good["actions"]["pick"]})
    with pytest.raises(ValueError, match="unknown head"):
        policy.update(**bad, epochs=1, minibatches=2)

    bad = dict(good, actions={"pick": good["actions"]["pick"]})
    with pytest.raises(ValueError, match="cfg"):
        policy.update(**bad, epochs=1, minibatches=2)

    bad = dict(
        good,
        actions=dict(good["actions"], flags=np.zeros((8, 2, 2), dtype=np.int64)),
    )
    with pytest.raises(ValueError, match="not an action head"):
        policy.update(**bad, epochs=1, minibatches=2)

    bad = dict(good, log_prob=np.zeros((8, 2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="log_prob"):
        policy.update(**bad, epochs=1, minibatches=2)

    bad = dict(good, value=np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError, match="value"):
        policy.update(**bad, epochs=1, minibatches=2)

    bad = dict(good, last_value=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="last_value"):
        policy.update(**bad, epochs=1, minibatches=2)

    with pytest.raises(ValueError, match="minibatches"):
        policy.update(**good, epochs=1, minibatches=3)
    with pytest.raises(ValueError, match="epochs"):
        policy.update(**good, epochs=0, minibatches=2)


def test_update_accepts_first_step_and_flat_value():
    spec = m3.EntityModelSpec(
        globals=0,
        context=[m3.ContextSet("cells", count=6, features=4)],
        queries=m3.QuerySet("agents", count=2, features=3, steps=2),
        heads=[
            m3.Head.pointer("pick", set="cells"),
            m3.Head.categorical("eta", classes=3, steps="first"),
        ],
        d_model=16,
        context_layers=1,
        decoder_layers=1,
        seed=5,
    )
    policy = m3.EntityPolicy(spec)
    rng = np.random.default_rng(3)
    s, t, e = 8, 2, 4
    obs = {
        "cells": rng.normal(size=(s, 6, 4)).astype(np.float32),
        "agents": rng.normal(size=(s, 2, 3)).astype(np.float32),
    }
    out = policy.act(obs)
    actions = {"pick": out["actions"]["pick"], "eta": out["actions"]["eta"][:, :, :1].reshape(s, 2)}
    stats = policy.update(
        obs,
        actions,
        out["log_prob"],
        out["value"].reshape(t, e),
        np.zeros((t, e), dtype=np.float32),
        np.ones((t, e), dtype=np.float32),
        np.zeros(e, dtype=np.float32),
        epochs=1,
        minibatches=2,
    )
    assert set(stats) == {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
    }
    assert all(np.isfinite(v) for v in stats.values())


def test_constructor_validates_hyperparameters():
    spec = tiny_spec()
    with pytest.raises(ValueError, match="learning_rate"):
        m3.EntityPolicy(spec, learning_rate=-1.0)
    with pytest.raises(ValueError, match="gamma"):
        m3.EntityPolicy(spec, gamma=2.0)
    with pytest.raises(ValueError, match="clip"):
        m3.EntityPolicy(spec, clip=0.0)
    with pytest.raises(ValueError, match="temperature"):
        m3.EntityPolicy(spec, temperature=-1.0)
    with pytest.raises(ValueError, match="seed"):
        m3.EntityPolicy(spec, seed=-1)


def test_act_reads_once():
    policy = m3.EntityPolicy(tiny_spec())
    obs = plain_obs(4)
    policy.act(obs)  # warm up lazy backend init
    m3.reset_read_count()
    policy.act(obs)
    assert m3.read_count() == 1


def test_update_reads_once():
    policy = m3.EntityPolicy(tiny_spec())
    args = tiny_rollout(policy)
    policy.update(**args, epochs=1, minibatches=2)  # warm up
    m3.reset_read_count()
    policy.update(**args, epochs=1, minibatches=2)
    assert m3.read_count() == 1
