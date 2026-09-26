"""Task planner bindings (TASK_PLANNER_PLAN.md T6, host path).

The device-resident ``PlannerData`` tests (upload once, ``queue_train_step``
on turn ids) arrive with K1; everything here runs through
``queue_train_step_host`` / ``predict`` on plain NumPy arrays.
"""

import numpy as np
import pytest

import mamba3_rl as m3

B = 2
U = 20
K = 3
Q = U * K
N = 100


def random_batch(seed=0, turns=B, active=3):
    rng = np.random.default_rng(seed)
    tiles = rng.standard_normal((turns, N, 48), dtype=np.float32)
    glob = rng.standard_normal((turns, 114), dtype=np.float32)
    units = rng.standard_normal((turns, U, 36), dtype=np.float32)
    upos = np.full((turns, U), -1, dtype=np.int16)
    tgt = np.full((turns, U, K), -100, dtype=np.int16)
    op = np.full((turns, U, K), -100, dtype=np.int16)
    crop = np.full((turns, U, K), -100, dtype=np.int16)
    opset = np.zeros((turns, U, K, 13), dtype=np.uint8)
    eta = np.full((turns, U), -1, dtype=np.int16)
    for bi in range(turns):
        for uu in range(active):
            upos[bi, uu] = rng.integers(0, N)
            eta[bi, uu] = int(rng.integers(0, 6))
            for j in range(K):
                t = int(rng.integers(0, N)) if rng.random() < 0.8 else N
                tgt[bi, uu, j] = t
                if t < N:
                    op[bi, uu, j] = int(rng.integers(0, 13))
                    if rng.random() < 0.5:
                        crop[bi, uu, j] = int(rng.integers(0, 5))
                    opset[bi, uu, j, rng.choice(13, size=2, replace=False)] = 1
    return tiles, glob, units, upos, tgt, op, crop, opset, eta


def small_model(**overrides):
    model_kwargs = overrides.pop("model_kwargs", {})
    cfg = m3.TaskPlannerConfig(
        d_model=32, n_tile_layers=1, n_joint_layers=1, seed=0, **overrides
    )
    return m3.TaskPlanner(cfg, **model_kwargs)


def test_predict_shapes_and_finiteness():
    model = small_model()
    tiles, glob, units, upos, *_ = random_batch()
    out = model.predict(tiles, glob, units, upos)
    assert out["target_logits"].shape == (B, U, K, N + 1)
    assert out["op"].shape == (B, U, K, 13)
    assert out["opset"].shape == (B, U, K, 13)
    assert out["crop"].shape == (B, U, K, 5)
    assert out["eta"].shape == (B, U)
    for v in out.values():
        assert np.all(np.isfinite(v)), "non-finite prediction"


def test_predict_aux_at_given_targets():
    model = small_model()
    tiles, glob, units, upos, tgt, *_ = random_batch()
    out = model.predict_aux(tiles, glob, units, upos, tgt)
    assert "target_logits" not in out
    assert out["op"].shape == (B, U, K, 13)
    assert np.all(np.isfinite(out["op"]))


def test_queued_host_steps_decrease_and_report():
    model = small_model()
    arrays = random_batch()
    for _ in range(100):
        model.queue_train_step_host(*arrays)
    losses = model.read_losses()
    assert len(losses) == 100
    assert all(len(entry) == 3 for entry in losses)
    first = [entry[0] for entry in losses]
    assert all(np.isfinite(first)), "non-finite queued loss"
    assert all(len(entry[2]) == 5 for entry in losses)
    # Same fixed batch: the last queued loss is below half the first.
    assert first[-1] < first[0] / 2, f"{first[0]} -> {first[-1]}"
    assert model.read_losses() == [], "queue not drained by read"


def test_save_load_round_trip_identical():
    model = small_model()
    tiles, glob, units, upos, *_ = random_batch()
    before = model.predict(tiles, glob, units, upos)
    path = "/tmp/mamba3_planner_py_roundtrip.m3ck"
    model.save(path)
    loaded = m3.TaskPlanner.load(path)
    after = loaded.predict(tiles, glob, units, upos)
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_bad_inputs_raise_value_error():
    model = small_model()
    tiles, glob, units, upos, tgt, op, crop, opset, eta = random_batch()
    with pytest.raises(ValueError):
        model.predict(tiles[:, :, :10], glob, units, upos)
    bad_tgt = tgt.copy()
    bad_tgt[0, 0, 0] = 101
    with pytest.raises(ValueError):
        model.queue_train_step_host(
            tiles, glob, units, upos, bad_tgt, op, crop, opset, eta
        )
    bad_op = op.copy()
    bad_op[0, 0, 0] = 13
    with pytest.raises(ValueError):
        model.queue_train_step_host(
            tiles, glob, units, upos, tgt, bad_op, crop, opset, eta
        )
    with pytest.raises(ValueError):
        m3.TaskPlanner(
            m3.TaskPlannerConfig(d_model=32, n_tile_layers=1, n_joint_layers=1),
            matmul_precision="bf16",
        )


def test_set_fused_planner_round_trips():
    assert m3.fused_planner() in (True, False)
    m3.set_fused_planner(False)
    assert m3.fused_planner() is False
    m3.set_fused_planner(True)
    assert m3.fused_planner() is True
