"""Generic entity-model bindings (ENTITY_MODEL_PLAN.md P1)."""

import numpy as np
import pytest

import mamba3_rl as m3


def tiny_spec():
    return m3.EntityModelSpec(
        globals=2,
        context=[m3.ContextSet("cells", count=4, features=3)],
        queries=m3.QuerySet("agents", count=2, features=2, steps=2),
        heads=[
            m3.Head.pointer("p", set="cells", extra_actions=1),
            m3.Head.categorical("c", classes=3, condition_on="p"),
            m3.Head.multilabel("ml", labels=2, loss_weight=0.5),
            m3.Head.regression("r", outputs=1, steps="first", loss_weight=0.25),
        ],
        d_model=8,
        context_layers=1,
        decoder_layers=1,
        seed=3,
    )


def tiny_arrays():
    rng = np.random.default_rng(7)
    s = 6
    d = {
        "cells": rng.normal(size=(s, 4, 3)).astype(np.float32),
        "globals": rng.normal(size=(s, 2)).astype(np.float32),
        "agents": rng.normal(size=(s, 2, 2)).astype(np.float32),
        "label.p": np.array([[1, 4, -1, 2]] * s, dtype=np.int64).reshape(s, 2, 2),
        "label.c": np.array([[2, 0, -1, 1]] * s, dtype=np.int64).reshape(s, 2, 2),
        "label.ml": np.array(
            [[1.0, 0.0, np.nan, np.nan, 0.0, 1.0, 1.0, 1.0]] * s, dtype=np.float32
        ).reshape(s, 2, 2, 2),
        "label.r": np.array([[0.5, np.nan]] * s, dtype=np.float32).reshape(s, 2, 1),
    }
    return d


def test_spec_json_round_trip():
    spec = tiny_spec()
    back = m3.EntityModelSpec.from_json(spec.to_json())
    assert back.to_json() == spec.to_json()
    with pytest.raises(Exception):
        m3.EntityModelSpec(
            globals=0,
            context=[m3.ContextSet("cells", count=0, features=3)],
            heads=[m3.Head.categorical("c", classes=2)],
        )


def test_bad_key_shape_id_raise_value_error():
    spec = tiny_spec()
    bad_key = tiny_arrays()
    bad_key["typo"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match="typo"):
        m3.EntityDataset(spec, bad_key)
    bad_shape = tiny_arrays()
    bad_shape["cells"] = np.zeros((6, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="cells"):
        m3.EntityDataset(spec, bad_shape)
    bad_id = tiny_arrays()
    bad_id["label.p"] = np.full((6, 2, 2), 9, dtype=np.int64)
    with pytest.raises(ValueError, match="label.p"):
        m3.EntityDataset(spec, bad_id)


def test_train_100_steps_halves_loss():
    spec = tiny_spec()
    arrays = {k: v[:2] for k, v in tiny_arrays().items()}
    data = m3.EntityDataset(spec, arrays)
    model = m3.EntityModel(spec, learning_rate=1e-2, weight_decay=0.0)
    ids = np.arange(2)
    for _ in range(100):
        model.queue_train_step(data, ids)
    losses = [entry["loss"] for entry in model.read_losses()]
    assert len(losses) == 100
    assert losses[-1] < 0.5 * losses[0], f"{losses[0]} -> {losses[-1]}"
    # Per-head entries exist.
    model.queue_train_step(data, ids)
    entry = model.read_losses()[0]
    assert set(entry["heads"]) == {"p", "c", "ml", "r"}
    assert np.isfinite(entry["loss"]) and np.isfinite(entry["grad_norm"])


def mamba3_dataset(spec):
    return m3.EntityDataset(spec, tiny_arrays())


def test_save_load_round_trip(tmp_path):
    spec = tiny_spec()
    data = mamba3_dataset(spec)
    model = m3.EntityModel(spec)
    ids = np.arange(2)
    before = model.predict(data, ids)
    path = str(tmp_path / "entity.m3ck")
    model.save(path)
    loaded = m3.EntityModel.load(path)
    after = loaded.predict(data, ids)
    for head in before:
        np.testing.assert_array_equal(before[head]["logits"], after[head]["logits"])


def test_greedy_equals_teacher_forced_on_own_choices():
    spec = tiny_spec()
    arrays = {k: v[:2] for k, v in tiny_arrays().items()}
    data = m3.EntityDataset(spec, arrays)
    model = m3.EntityModel(spec)
    ids = np.arange(2)
    greedy = model.predict(data, ids)
    arrays["label.p"] = greedy["p"]["choice"].astype(np.int64)
    forced = model.predict(arrays, decode="teacher_forced")
    for head in greedy:
        np.testing.assert_array_equal(greedy[head]["logits"], forced[head]["logits"])


def test_chooser_is_honoured():
    spec = tiny_spec()
    data = mamba3_dataset(spec)
    model = m3.EntityModel(spec)
    ids = np.arange(2)
    calls = []

    def chooser(step, logits):
        calls.append(step)
        out = np.argmax(logits, axis=-1)
        out[:] = 1  # force entity 1 everywhere
        return out

    out = model.predict(data, ids, chooser=chooser)
    assert calls == [0, 1]
    assert np.all(out["p"]["choice"] == 1)


def test_fused_modes_agree():
    spec = tiny_spec()
    data = mamba3_dataset(spec)
    ids = np.arange(2)
    outs = {}
    for fused in (False, True):
        m3.set_fused_entity_model(fused)
        model = m3.EntityModel(spec)
        outs[fused] = model.predict(data, ids)
    m3.set_fused_entity_model(True)
    for head in outs[False]:
        np.testing.assert_allclose(
            outs[False][head]["logits"], outs[True][head]["logits"], rtol=1e-5, atol=1e-5
        )


def test_set_fused_entity_model_round_trips():
    assert m3.fused_entity_model() in (True, False)
    m3.set_fused_entity_model(False)
    assert m3.fused_entity_model() is False
    m3.set_fused_entity_model(True)
    assert m3.fused_entity_model() is True
