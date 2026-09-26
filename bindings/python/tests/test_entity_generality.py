"""Generality proofs (ENTITY_MODEL_PLAN.md P2): two non-Kaggriculture problems,
each training on CPU in < 60 s. If either needs a library change to be
expressible, the API is wrong: fix the API, not the test."""

import numpy as np

import mamba3_rl as m3


def test_nearest_free_item():
    """Pointer + extras, K = 1: 8 agents on a 1-D line of 16 items.

    The label is the nearest present item not taken by a lower-index agent,
    else extra action 0 ("none"). Must reach ≥ 95% top-1.
    """
    rng = np.random.default_rng(11)
    n_agents, n_items = 8, 16
    spec = m3.EntityModelSpec(
        context=[m3.ContextSet("items", count=n_items, features=2)],
        queries=m3.QuerySet("agents", count=n_agents, features=1, steps=1),
        heads=[m3.Head.pointer("pick", set="items", extra_actions=1)],
        d_model=32,
        context_layers=2,
        decoder_layers=2,
        seed=1,
    )

    def sample(batch):
        pos = rng.uniform(size=(batch, n_items))
        present = (rng.uniform(size=(batch, n_items)) < 0.7).astype(np.float64)
        agents = rng.uniform(size=(batch, n_agents))
        items = np.stack([pos, present], axis=-1).astype(np.float32)
        agents_f = agents[..., None].astype(np.float32)
        labels = np.full((batch, n_agents, 1), -1, dtype=np.int64)
        for b in range(batch):
            taken = set()
            for a in range(n_agents):
                best, best_d = -1, np.inf
                for i in range(n_items):
                    if present[b, i] == 0 or i in taken:
                        continue
                    d = abs(pos[b, i] - agents[b, a])
                    if d < best_d:
                        best, best_d = i, d
                if best < 0:
                    labels[b, a, 0] = n_items  # extra action 0 ("none")
                else:
                    labels[b, a, 0] = best
                    taken.add(best)
        return {
            "items": items,
            "agents": agents_f,
            "label.pick": labels,
        }

    train = m3.EntityDataset(spec, sample(256))
    dev = m3.EntityDataset(spec, sample(64))
    model = m3.EntityModel(spec, learning_rate=3e-3, weight_decay=0.0)
    ids = np.arange(256)
    for _ in range(60):
        rng.shuffle(ids)
        for start in range(0, 256, 32):
            model.queue_train_step(train, ids[start : start + 32])
    model.read_losses()
    metrics = model.evaluate(dev, np.arange(64), batch=64)
    top1 = metrics["pick"]["top1"]
    assert top1 >= 0.95, f"nearest-free-item top-1 {top1}"


def test_ordered_visits_on_grid():
    """Autoregressive K = 3 plus a conditioned categorical: 1 query on a 6x6
    grid with 3 marked cells. Labels are the marked cells in nearest-neighbour
    order from the anchor, plus each cell's colour. Must reach ≥ 95% on step 1
    and ≥ 90% on step 3 with greedy decoding, and 0 repeated cells.
    """
    rng = np.random.default_rng(23)
    spec = m3.EntityModelSpec(
        context=[
            m3.ContextSet(
                "cells",
                count=36,
                features=4,
                layout=m3.Grid(6, 6, alternate_axes=True),
            )
        ],
        queries=m3.QuerySet(
            "probe", count=1, features=0 + 2, anchor="cells", steps=3,
            autoregressive_on="visit",
        ),
        heads=[
            m3.Head.pointer("visit", set="cells", extra_actions=0),
            m3.Head.categorical("colour", classes=3, condition_on="visit"),
        ],
        d_model=32,
        context_layers=2,
        decoder_layers=2,
        seed=2,
    )

    def order_from(start, cells):
        order, at, todo = [], start, set(cells)
        while todo:
            nxt = min(todo, key=lambda c: (abs(c // 6 - at // 6) + abs(c % 6 - at % 6), c))
            order.append(nxt)
            todo.remove(nxt)
            at = nxt
        return order

    def sample(batch):
        cells = np.zeros((batch, 36, 4), dtype=np.float32)
        probe = np.zeros((batch, 1, 2), dtype=np.float32)
        anchor = np.zeros((batch, 1), dtype=np.int64)
        visit = np.zeros((batch, 1, 3), dtype=np.int64)
        colour = np.zeros((batch, 1, 3), dtype=np.int64)
        for b in range(batch):
            anchor[b, 0] = rng.integers(36)
            marked = rng.choice(36, size=3, replace=False)
            colours = rng.integers(3, size=3)
            for i, c in enumerate(marked):
                cells[b, c, 0] = 1.0
                cells[b, c, 1 + colours[i]] = 1.0
            probe[b, 0, 0] = (anchor[b, 0] // 6) / 5.0
            probe[b, 0, 1] = (anchor[b, 0] % 6) / 5.0
            for j, c in enumerate(order_from(int(anchor[b, 0]), marked)):
                visit[b, 0, j] = c
                colour[b, 0, j] = colours[list(marked).index(c)]
        return {
            "cells": cells,
            "probe": probe,
            "probe.anchor": anchor,
            "label.visit": visit,
            "label.colour": colour,
        }

    train = m3.EntityDataset(spec, sample(256))
    dev_arrays = sample(64)
    dev = m3.EntityDataset(spec, dev_arrays)
    model = m3.EntityModel(spec, learning_rate=3e-3, weight_decay=0.0)
    ids = np.arange(256)
    for _ in range(80):
        rng.shuffle(ids)
        for start in range(0, 256, 32):
            model.queue_train_step(train, ids[start : start + 32])
    model.read_losses()
    out = model.predict(dev, np.arange(64))
    pred = out["visit"]["choice"][:, 0, :]  # [64, 3]
    true = dev_arrays["label.visit"][:, 0, :]
    step1 = float(np.mean(pred[:, 0] == true[:, 0]))
    step3 = float(np.mean(pred[:, 2] == true[:, 2]))
    repeats = int(np.sum((pred[:, 1] == pred[:, 0]) | (pred[:, 2] == pred[:, 1]) | (pred[:, 2] == pred[:, 0])))
    assert step1 >= 0.95, f"step-1 top-1 {step1}"
    assert step3 >= 0.90, f"step-3 top-1 {step3}"
    assert repeats == 0, f"{repeats} repeated cells"
