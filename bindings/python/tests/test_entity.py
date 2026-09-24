"""Structured observations: the spec, the structured policy, and the numpy export."""

import json

import numpy as np
import pytest

import mamba3_rl as m3
from mamba3_rl import numpy_ref

N_TILES = 5
F_TILES = 3


def two_set_spec():
    return m3.ObsSpec(
        globals=2,
        sets=[m3.EntitySet("tiles", count=3, features=2), m3.EntitySet("units", count=2, features=1)],
    )


def structured_spec():
    return m3.ObsSpec(
        globals=2,
        sets=[
            m3.EntitySet("tiles", count=N_TILES, features=F_TILES),
            m3.EntitySet("units", count=2, features=2),
        ],
    )


def structured_config(scoring="additive", extra=1, **overrides):
    spec = structured_spec()
    settings = dict(
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        d_state=8,
        chunk_size=4,
        seed=7,
        obs_spec=spec,
        entity_encoders={
            "tiles": m3.EntityEncoderConfig(hidden=[16], d_entity=12),
            "units": m3.EntityEncoderConfig(hidden=[], d_entity=8, slot_embedding=True),
        },
        pooling=m3.PoolingConfig(kinds=("mean", "max")),
        action_head=m3.PointerHead("tiles", hidden=16, scoring=scoring, extra_actions=extra),
    )
    settings.update(overrides)
    return m3.PolicyConfig(spec.obs_dim, N_TILES + extra, **settings)


def random_observations(spec, envs, rng):
    """``[envs, obs_dim]`` with random features and about a third of the slots empty."""
    parts = {}
    for s in spec.sets:
        features = rng.standard_normal((envs, s.count, s.features)).astype(np.float32)
        presence = (rng.random((envs, s.count)) > 0.3).astype(np.float32)
        presence[:, 0] = 1.0  # never an empty set, so there is always a legal tile
        parts[s.name] = (features, presence)
    globals_ = rng.standard_normal((envs, spec.globals)).astype(np.float32)
    return spec.pack_batch(globals=globals_, **parts), parts


# -- ObsSpec -------------------------------------------------------------------


def test_the_wire_layout_is_the_one_the_rust_crate_packs():
    # The same vector `tests/rl_entity.rs::pack_then_unpack_is_the_identity`
    # asserts, so the two languages cannot drift apart.
    spec = two_set_spec()
    assert spec.obs_dim == 15
    assert spec.offsets() == [("tiles", 2), ("units", 11)]
    flat = spec.pack(
        globals=[0.5, -1.0],
        tiles=(np.arange(1, 7, dtype=np.float32).reshape(3, 2), [1, 0, 1]),
        units=(np.array([[7.0], [8.0]]), [0, 1]),
    )
    assert flat.dtype == np.float32
    assert flat.tolist() == [0.5, -1.0, 1, 2, 1, 3, 4, 0, 5, 6, 1, 7, 0, 8, 1]


def test_pack_and_unpack_round_trip_single_and_batched():
    spec = two_set_spec()
    rng = np.random.default_rng(0)
    batch, parts = random_observations(spec, 4, rng)
    assert batch.shape == (4, spec.obs_dim)

    out = spec.unpack(batch)
    np.testing.assert_array_equal(out["tiles"][0], parts["tiles"][0])
    np.testing.assert_array_equal(out["tiles"][1], parts["tiles"][1])
    np.testing.assert_array_equal(out["units"][1], parts["units"][1])
    assert out["globals"].shape == (4, 2)

    for row in range(4):
        one = spec.unpack(batch[row])
        again = spec.pack(globals=one["globals"], tiles=one["tiles"], units=one["units"])
        np.testing.assert_array_equal(again, batch[row])


def test_features_alone_mean_every_slot_is_present():
    spec = m3.ObsSpec(sets=[m3.EntitySet("a", count=2, features=1)])
    assert spec.pack(a=np.array([[3.0], [4.0]])).tolist() == [3.0, 1.0, 4.0, 1.0]


def test_pack_names_what_is_wrong():
    spec = two_set_spec()
    tiles = (np.zeros((3, 2)), np.ones(3))
    units = (np.zeros((2, 1)), np.ones(2))
    with pytest.raises(ValueError, match="globals is missing"):
        spec.pack(tiles=tiles, units=units)
    with pytest.raises(ValueError, match="units.*missing"):
        spec.pack(globals=[0, 0], tiles=tiles)
    with pytest.raises(ValueError, match="cards"):
        spec.pack(globals=[0, 0], tiles=tiles, units=units, cards=tiles)
    with pytest.raises(ValueError, match=r"tiles features must be shaped \[3, 2\]"):
        spec.pack(globals=[0, 0], tiles=(np.zeros((2, 3)), np.ones(3)), units=units)
    with pytest.raises(ValueError, match="obs must be"):
        spec.unpack(np.zeros(7))


def test_a_spec_validates_on_construction():
    with pytest.raises(ValueError, match="sets"):
        m3.ObsSpec(globals=2, sets=[])
    with pytest.raises(ValueError, match=r"sets\[1\].name .* used twice"):
        m3.ObsSpec(sets=[m3.EntitySet("a", count=1, features=1)] * 2)
    with pytest.raises(ValueError, match=r"sets\[0\].count"):
        m3.ObsSpec(sets=[m3.EntitySet("a", count=0, features=1)])


def test_a_spec_round_trips_through_a_dict():
    spec = two_set_spec()
    assert m3.ObsSpec.from_dict(spec.to_dict()) == spec
    assert json.loads(json.dumps(spec.to_dict()))["sets"][0]["name"] == "tiles"


# -- PolicyConfig ----------------------------------------------------------------


def test_defaults_stay_flat_and_structure_is_reported():
    flat = m3.PolicyConfig(6, 3)
    assert flat.obs_spec is None
    assert flat.action_head is None
    assert not flat.is_structured
    assert "obs_spec" not in flat.to_dict()

    cfg = structured_config()
    assert cfg.is_structured
    assert cfg.obs_spec == structured_spec()
    assert cfg.action_head == m3.PointerHead("tiles", hidden=16, extra_actions=1)
    assert cfg.entity_encoders["units"].slot_embedding
    assert cfg.pooling.kinds == ["mean", "max"]
    assert m3.PolicyConfig.from_dict(cfg.to_dict()) == cfg


def test_inconsistent_structure_is_refused():
    spec = structured_spec()
    with pytest.raises(ValueError, match="obs_dim"):
        m3.PolicyConfig(spec.obs_dim + 1, 5, obs_spec=spec)
    with pytest.raises(ValueError, match="cards"):
        m3.PolicyConfig(spec.obs_dim, 5, obs_spec=spec,
                        entity_encoders={"cards": m3.EntityEncoderConfig()})
    with pytest.raises(ValueError, match="action_dim is 4"):
        m3.PolicyConfig(spec.obs_dim, 4, obs_spec=spec, action_head=m3.PointerHead("tiles"))
    with pytest.raises(ValueError, match="no obs_spec"):
        m3.PolicyConfig(6, 3, action_head=m3.PointerHead("tiles"))
    with pytest.raises(ValueError, match="scoring"):
        m3.PointerHead("tiles", scoring="cosine")
    with pytest.raises(ValueError, match="pooling kind"):
        m3.PoolingConfig(kinds=("sum",))


# -- the structured policy ---------------------------------------------------------


def test_parameter_paths_follow_the_structure():
    described = m3.Policy(structured_config()).describe()
    for path in ("entity.tiles.mlp.0.weight", "entity.units.slot", "pool.proj.weight",
                 "actor.pointer.w_h.bias", "actor.pointer.v.weight", "actor.extra.weight"):
        assert path in described
    assert "encoder.weight" not in described


def test_a_structured_checkpoint_carries_its_structure(tmp_path):
    policy = m3.Policy(structured_config(scoring="dot"))
    path = str(tmp_path / "structured.json")
    policy.save(path)
    restored = m3.Policy.load(path)
    assert restored.config == policy.config
    assert restored.fingerprint() == policy.fingerprint()


def test_a_checkpoint_from_before_structure_loads_as_flat(tmp_path):
    policy = m3.Policy(m3.PolicyConfig(6, 3, d_model=32, n_layers=1))
    path = tmp_path / "flat.json"
    policy.save(str(path))
    checkpoint = json.loads(path.read_text())
    assert set(checkpoint["metadata"]["policy"]) == {
        "obs_dim", "action_dim", "n_layers", "norm_eps", "seed", "ssm"}
    restored = m3.Policy.load(str(path))
    assert not restored.config.is_structured
    assert restored.fingerprint() == policy.fingerprint()


def test_a_rollout_never_picks_an_empty_slot():
    cfg = structured_config(extra=0)
    policy = m3.Policy(cfg)
    rng = np.random.default_rng(3)
    obs, parts = random_observations(cfg.obs_spec, 64, rng)
    rollout = m3.Rollout(policy, num_envs=64, seed=1)
    for _ in range(5):
        actions, _, log_probs = rollout.step(obs)
        present = parts["tiles"][1][np.arange(64), actions]
        assert np.all(present == 1.0)
        assert np.all(np.isfinite(log_probs))
    logits, _ = rollout.evaluate(obs)
    empty = parts["tiles"][1] == 0.0
    assert np.all(logits[empty] == np.finfo(np.float32).min)


# -- numpy export ------------------------------------------------------------------


@pytest.mark.parametrize("structured", [True, False])
def test_the_numpy_reference_matches_the_rollout(tmp_path, structured):
    envs, steps = 64, 200
    if structured:
        cfg = structured_config()
    else:
        cfg = m3.PolicyConfig(10, 4, d_model=32, n_layers=2, n_heads=2, head_dim=16,
                              d_state=8, chunk_size=4, seed=3)
    policy = m3.Policy(cfg)
    path = str(tmp_path / "policy.npz")
    policy.export_numpy(path)
    reference = numpy_ref.Policy.load(path)
    assert reference.action_dim == cfg.action_dim

    rng = np.random.default_rng(11)
    rollout = m3.Rollout(policy, num_envs=envs)
    state = reference.initial_state(envs)
    worst = 0.0
    for step in range(steps):
        if structured:
            obs, _ = random_observations(cfg.obs_spec, envs, rng)
        else:
            obs = rng.standard_normal((envs, cfg.obs_dim)).astype(np.float32)
        # Every environment restarts now and then, never all at once.
        reset = (rng.random(envs) < 0.05).astype(np.float32) if step else None
        want_logits, want_values = rollout.evaluate(obs, reset=reset)
        logits, values, state = reference.step(obs, state, reset=reset)
        masked = want_logits == np.finfo(np.float32).min
        np.testing.assert_array_equal(logits == np.finfo(np.float32).min, masked)
        err = np.abs(logits[~masked] - want_logits[~masked]) / (1 + np.abs(want_logits[~masked]))
        verr = np.abs(values - want_values) / (1 + np.abs(want_values))
        worst = max(worst, float(err.max()), float(verr.max()))
    assert worst < 1e-4, f"numpy_ref drifted from the rollout by {worst}"


def test_the_export_is_self_describing(tmp_path):
    policy = m3.Policy(structured_config())
    path = str(tmp_path / "policy.npz")
    policy.export_numpy(path)
    with np.load(path) as packed:
        config = json.loads(str(packed["__config__"]))
        assert config == policy.config.to_dict()
        assert packed["pool.proj.weight"].dtype == np.float32
