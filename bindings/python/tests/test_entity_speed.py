"""K11: the Python training step stays within 1.1x of the Rust profiler.

Slow by design (batch 128, 8 timed steps of the Kaggriculture-scale spec):
run with ``pytest -m slow``. The Rust reference number comes from
``bench/results/entity_step.md`` (written by ``bench/entity_step.sh``);
without a batch-128 row the test skips.
"""

import re
import time
from pathlib import Path

import numpy as np
import pytest

import mamba3_rl as m3

pytestmark = pytest.mark.slow


def kaggriculture_spec():
    return m3.EntityModelSpec(
        globals=114,
        context=[m3.ContextSet("tiles", count=100, features=48)],
        queries=m3.QuerySet(
            "units", count=20, features=36, anchor="tiles", steps=3, autoregressive_on="target"
        ),
        heads=[
            m3.Head.pointer("target", set="tiles", step_weights=[1.0, 0.5, 0.5]),
            m3.Head.categorical("op", classes=13, condition_on="target"),
            m3.Head.multilabel("opset", labels=13, condition_on="target", loss_weight=0.3),
            m3.Head.categorical("crop", classes=5, condition_on="target", loss_weight=0.3),
            m3.Head.regression("eta", outputs=1, steps="first", loss_weight=0.1),
        ],
        d_model=128,
        context_layers=3,
        decoder_layers=3,
        seed=0,
    )


def kaggriculture_arrays(samples, seed=99):
    rng = np.random.default_rng(seed)
    n, u, k = 100, 20, 3
    anchor = np.full((samples, u), -1, dtype=np.int64)
    tgt = np.full((samples, u, k), -1, dtype=np.int64)
    op = np.full((samples, u, k), -1, dtype=np.int64)
    crop = np.full((samples, u, k), -1, dtype=np.int64)
    opset = np.zeros((samples, u, k, 13), dtype=np.float32)
    eta = np.full((samples, u, 1), np.nan, dtype=np.float32)
    for bi in range(samples):
        for uu in range(u):
            if rng.integers(5) > 0:
                anchor[bi, uu] = rng.integers(n)
                eta[bi, uu, 0] = float(rng.integers(20))
            for j in range(k):
                r = rng.integers(10)
                if r < 7:
                    tgt[bi, uu, j] = rng.integers(n)
                    op[bi, uu, j] = rng.integers(13)
                    crop[bi, uu, j] = rng.integers(5)
                    opset[bi, uu, j, rng.integers(13)] = 1.0
                elif r < 8:
                    tgt[bi, uu, j] = n
    return {
        "tiles": rng.normal(size=(samples, n, 48)).astype(np.float32),
        "globals": rng.normal(size=(samples, 114)).astype(np.float32),
        "units": rng.normal(size=(samples, u, 36)).astype(np.float32),
        "units.anchor": anchor,
        "label.target": tgt,
        "label.op": op,
        "label.opset": opset,
        "label.crop": crop,
        "label.eta": eta,
    }


def rust_ms_per_step():
    md = Path(__file__).resolve().parents[3] / "bench" / "results" / "entity_step.md"
    if not md.exists():
        pytest.skip("no bench/results/entity_step.md; run bench/entity_step.sh first")
    rows = md.read_text().splitlines()
    for line in reversed(rows):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 4 and cells[2] == "128":
            m = re.search(r"([\d.]+)\s*ms", cells[3])
            if m:
                return float(m.group(1))
    pytest.skip("no batch-128 row in bench/results/entity_step.md")


def test_python_step_within_1_1x_of_rust():
    batch = 128
    spec = kaggriculture_spec()
    data = m3.EntityDataset(spec, kaggriculture_arrays(256))
    model = m3.EntityModel(spec)
    ids = np.arange(batch, dtype=np.int64)
    for _ in range(4):  # warm up (kernels, allocator, autotune)
        model.queue_train_step(data, ids)
    model.read_losses()
    started = time.perf_counter()
    for _ in range(8):
        model.queue_train_step(data, ids)
    py_ms = (time.perf_counter() - started) * 1000.0 / 8
    model.read_losses()
    rust_ms = rust_ms_per_step()
    assert py_ms <= 1.1 * rust_ms, f"python {py_ms:.0f}ms/step vs rust {rust_ms:.0f}ms/step"
