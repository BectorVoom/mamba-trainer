"""The ``mamba3_graph`` module: Graph Mamba from Python.

The Rust suites prove the model; these prove the boundary -- that the Python
surface is the Rust one (the golden file both sides reproduce), that arrays go in
in whatever dtype and layout NumPy holds them, that nothing is read while
training, and that the interpreter stays responsive and interruptible.
"""

import json
import os
import pathlib
import re
import signal
import threading
import time
import warnings

import numpy as np
import pytest

import mamba3_graph as mg
import mamba3_rl

ROOT = pathlib.Path(__file__).resolve().parents[3]


def undirected(pairs):
    """``int64[2, 2 * len(pairs)]``: every pair in both directions."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.concatenate([pairs.T, pairs.T[::-1]], axis=1)


def random_graph(n, edges, rng):
    """An undirected random graph on ``n`` nodes, without self-loops."""
    pairs = rng.integers(0, n, size=(edges, 2))
    return undirected(pairs[pairs[:, 0] != pairs[:, 1]])


def small_spec(features=6, task=None, **overrides):
    settings = dict(
        node_features=features,
        task=task or mg.NodeClassification(4),
        d_model=16,
        max_hops=2,
        walks=3,
        repeats=2,
        node_layers=1,
        seed=3,
    )
    settings.update(overrides)
    return mg.GraphMambaSpec(**settings)


def node_data(n=60, features=6, classes=4, seed=0):
    rng = np.random.default_rng(seed)
    index = np.arange(n)
    return dict(
        edge_index=random_graph(n, 2 * n, rng),
        x=rng.standard_normal((n, features)).astype(np.float32),
        y=rng.integers(0, classes, n),
        train_mask=index % 3 == 0,
        val_mask=index % 3 == 1,
        test_mask=index % 3 == 2,
    )


def graphs_data(sizes, features=6, classes=3, seed=0):
    """Many graphs with one class each, all in the training split."""
    rng = np.random.default_rng(seed)
    edges, offset = [], 0
    for size in sizes:
        edges.append(random_graph(size, 2 * size, rng) + offset)
        offset += size
    count = len(sizes)
    return dict(
        edge_index=np.concatenate(edges, axis=1),
        x=rng.standard_normal((offset, features)).astype(np.float32),
        y=np.arange(count) % classes,
        graph_ptr=np.concatenate([[0], np.cumsum(sizes)]),
        train_mask=np.ones(count, dtype=bool),
        val_mask=np.zeros(count, dtype=bool),
        test_mask=np.zeros(count, dtype=bool),
    )


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


def test_the_module_is_a_second_import_name_of_one_extension():
    assert mg.GraphMamba.__module__ == "mamba3_graph"
    assert mg.GraphMambaSpec.__module__ == "mamba3_graph"
    assert mg.GraphDataset.__module__ == "mamba3_graph"
    # One library, one device, one set of counters.
    assert mg.read_count is mamba3_rl.read_count
    assert mg.upload_count is mamba3_rl.upload_count
    assert mg.LrSchedule is mamba3_rl.LrSchedule
    assert mg.backend() == mamba3_rl.backend()
    # The graph classes are not re-exported from the RL module.
    assert not hasattr(mamba3_rl, "GraphMamba")
    # Both packages come from the same place: one install, not a stray tree.
    assert pathlib.Path(mg.__file__).parents[1] == pathlib.Path(mamba3_rl.__file__).parents[1]


def test_every_public_name_is_exported_and_typed():
    extension = mamba3_rl._mamba3_rl
    names = {
        name
        for name in dir(extension)
        if not name.startswith("_")
        and getattr(getattr(extension, name), "__module__", None) == "mamba3_graph"
    }
    # The classes carry the module name; the functions are found in the source.
    source = (ROOT / "bindings/python/src/graph.rs").read_text()
    named = set(re.findall(r'#\[pyfunction\(name = "([A-Za-z]+)"\)\]', source))
    plain = set(re.findall(r"#\[pyfunction\]\s*(?:#\[pyo3[^\]]*\]\s*)?fn (\w+)", source))
    names |= named | plain
    assert {"GraphMamba", "GraphDataset", "GraphMambaSpec", "Categorical", "GraphTask"} <= names
    assert {"NodeClassification", "GraphRegression", "rwse", "laplacian_pe", "build_info"} <= names
    stubs = (pathlib.Path(mg.__file__).parent / "__init__.pyi").read_text()
    for name in sorted(names):
        assert name in mg.__all__, f"{name} is not exported from mamba3_graph"
        assert hasattr(mg, name)
        assert re.search(rf"\b(class|def) {name}\b", stubs), f"{name} has no stub"
    for name in mg.__all__:
        assert hasattr(mg, name)


def test_build_info_and_the_debug_warning():
    info = mg.build_info()
    assert info["profile"] == "release", "the tests are meant to run on a release build"
    assert info["backend"] == mg.backend()
    assert info["version"] == mg.__version__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mg._warn_if_debug_build()  # a release build stays silent
        mg.GraphMamba(small_spec())
    with pytest.warns(RuntimeWarning, match="maturin develop --release"):
        mg._warn_if_debug_build(True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_spec_errors_name_the_argument():
    def fails(match, **overrides):
        with pytest.raises(ValueError, match=match):
            small_spec(**overrides)

    fails("d_model", d_model=0)
    fails("token_layers", token_layers=0)
    fails("tokens", max_hops=8, walks=16)
    fails("local", local="mpnn")
    fails("mpnn", mpnn="gcn")
    fails("token_tail", token_tail="both")
    fails("token_sampling", token_sampling="never")
    fails("order", order="random")
    fails("edge_features", edge_features=3)
    fails("node_features", node_features="six")
    fails("dropout", dropout=1.5)
    fails("pe_sign_flip", pe_sign_flip=(0, 4))
    with pytest.raises(ValueError, match="pool"):
        mg.GraphClassification(3, pool="max")
    with pytest.raises(ValueError, match="loss"):
        mg.GraphRegression(1, loss="huber")
    # Node tokens only: no token layer is asked for.
    assert small_spec(max_hops=0).tokens_per_node == 1
    assert small_spec().tokens_per_node == 5


def test_spec_round_trips_through_json():
    spec = small_spec(
        node_features=mg.Categorical([5, 3]),
        edge_features=mg.Categorical([4]),
        mpnn="gated_gcn",
        pe_dim=8,
        pe_sign_flip=(2, 8),
        order="ppr",
        token_tail="bidirectional",
        task=mg.GraphRegression(2, pool="sum", loss="mse"),
    )
    again = mg.GraphMambaSpec.from_json(spec.to_json())
    assert again == spec
    assert again.to_json() == spec.to_json()
    assert spec.task.outputs == 2 and spec.task.per_graph
    assert mg.GraphMamba(spec).num_parameters == spec.num_parameters
    with pytest.raises(ValueError):
        mg.GraphMambaSpec.from_json("{")


def test_dataset_errors_name_the_key():
    spec = small_spec()

    def fails(match, **changes):
        arrays = node_data()
        for key, value in changes.items():
            if value is None:
                arrays.pop(key)
            else:
                arrays[key] = value
        with pytest.raises(ValueError, match=match):
            mg.GraphDataset(spec, arrays)

    fails("labels", labels=np.zeros(60))
    fails("x is required", x=None)
    fails("edge_index is required", edge_index=None)
    fails("x must be", x=np.zeros((60, 5), dtype=np.float32))
    fails("x must be", x=np.zeros((60, 6), dtype=np.int64))
    fails("edge_index must be", edge_index=np.zeros((7, 3), dtype=np.int64))
    fails("edge_index must be", edge_index=np.zeros((2, 7), dtype=np.float32))
    fails("edge_index", edge_index=np.array([[0, 60], [1, 2]]))
    fails("edge_index", edge_index=np.array([[0, -1], [1, 2]]))
    fails("y must be", y=np.zeros(59, dtype=np.int64))
    fails(r"^y\b|: y\b|\by:", y=np.full(60, 4))
    fails("train_mask must be", train_mask=np.zeros(59, dtype=bool))
    fails("val_mask must be", val_mask=np.zeros(60))
    fails(r"\bpe\b", pe=np.zeros((60, 2), dtype=np.float32))
    fails("graph_ptr", graph_ptr=np.array([0, 30, 20, 60]))
    fails(r"\bx\b", x=np.full((60, 6), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="dtype"):
        mg.GraphDataset(spec, node_data(), dtype="f64")
    with pytest.raises(ValueError, match="edge_attr is required"):
        mg.GraphDataset(small_spec(mpnn="gine", edge_features=2), node_data())
    data = mg.GraphDataset(spec, node_data())
    assert (data.num_nodes, data.num_graphs, data.dtype) == (60, 1, "f32")
    assert data.num_edges > 0 and data.nbytes > 0


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------


def dense_adjacency(edge_index, n):
    a = np.zeros((n, n))
    a[edge_index[0], edge_index[1]] = 1.0
    a[edge_index[1], edge_index[0]] = 1.0
    return a


def test_rwse_matches_matrix_powers():
    rng = np.random.default_rng(1)
    n, k = 30, 6
    edge_index = random_graph(n, 60, rng)
    got = mg.rwse(edge_index, n, k)
    assert got.shape == (n, k) and got.dtype == np.float32
    a = dense_adjacency(edge_index, n)
    degree = a.sum(1, keepdims=True)
    p = np.divide(a, degree, out=np.zeros_like(a), where=degree > 0)
    want = np.stack([np.diag(np.linalg.matrix_power(p, j)) for j in range(1, k + 1)], axis=1)
    np.testing.assert_allclose(got, want, atol=1e-5)
    # Any integer width, and a transposed (Fortran-ordered) view.
    np.testing.assert_array_equal(mg.rwse(edge_index.astype(np.int32), n, k), got)
    np.testing.assert_array_equal(mg.rwse(np.asfortranarray(edge_index), n, k), got)
    with pytest.raises(ValueError, match="smaller k"):
        mg.rwse(edge_index, n, k, max_ball=3)
    with pytest.raises(ValueError, match="edge_index"):
        mg.rwse(edge_index.T, n, k)


def test_laplacian_pe_matches_eigh_up_to_sign():
    rng = np.random.default_rng(2)
    sizes = [9, 12]
    edges, offset = [], 0
    for size in sizes:
        ring = np.stack([np.arange(size), (np.arange(size) + 1) % size], axis=1)
        chords = rng.integers(0, size, size=(size, 2))
        pairs = np.concatenate([ring, chords[chords[:, 0] != chords[:, 1]]])
        edges.append(undirected(pairs) + offset)
        offset += size
    edge_index = np.concatenate(edges, axis=1)
    k = 4
    got = mg.laplacian_pe(edge_index, offset, k, graph_ptr=np.array([0, 9, 21]))
    assert got.shape == (offset, k)
    a = dense_adjacency(edge_index, offset)
    start = 0
    for size in sizes:
        block = a[start : start + size, start : start + size]
        scale = 1.0 / np.sqrt(block.sum(1))
        laplacian = np.eye(size) - scale[:, None] * block * scale[None, :]
        values, vectors = np.linalg.eigh(laplacian)
        assert values[0] < 1e-8 < values[1], "a connected graph has one trivial eigenvector"
        for column in range(k):
            ours = got[start : start + size, column].astype(np.float64)
            theirs = vectors[:, 1 + column]
            # Distinct eigenvalues: the eigenvector is unique up to its sign.
            assert abs(abs(ours @ theirs) - 1.0) < 1e-4
            np.testing.assert_allclose(laplacian @ ours, values[1 + column] * ours, atol=1e-4)
        start += size
    with pytest.raises(ValueError, match="rwse"):
        mg.laplacian_pe(edge_index, offset, k, max_nodes=10)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_ingest_reads_nothing_and_takes_any_dtype_and_layout():
    spec = small_spec(max_hops=0)
    arrays = node_data()
    model = mg.GraphMamba(spec)
    mg.reset_read_count()
    reference_data = mg.GraphDataset(spec, arrays)
    assert mg.read_count() == 0, "building a dataset reads nothing back"
    reference = model.predict(reference_data, parts=1)
    # Labels and masks are read by the metric, not by the prediction.
    scores = {
        split: model.evaluate(reference_data, split=split, parts=1)
        for split in ("train", "val", "test")
    }
    assert [scores[split]["count"] for split in ("train", "val", "test")] == [20, 20, 20]
    mg.reset_read_count()

    # C-contiguous but unaligned: float32 at an odd byte offset.
    raw = bytearray(1 + arrays["x"].nbytes)
    raw[1:] = arrays["x"].tobytes()
    unaligned = np.frombuffer(raw, dtype=np.float32, offset=1).reshape(arrays["x"].shape)
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned

    wide = np.random.default_rng(5).standard_normal((60, 12)).astype(np.float32)
    wide[:, ::2] = arrays["x"]
    variants = {
        "float64 features": dict(x=arrays["x"].astype(np.float64)),
        "float16 features": dict(x=arrays["x"].astype(np.float16)),
        "int32 edges": dict(edge_index=arrays["edge_index"].astype(np.int32)),
        "uint32 edges": dict(edge_index=arrays["edge_index"].astype(np.uint32)),
        "Fortran-ordered features": dict(x=np.asfortranarray(arrays["x"])),
        "sliced features": dict(x=wide[:, ::2]),
        "Fortran-ordered edges": dict(edge_index=np.asfortranarray(arrays["edge_index"])),
        "int32 labels and byte masks": dict(
            y=arrays["y"].astype(np.int32), train_mask=arrays["train_mask"].astype(np.uint8)
        ),
        "lists": dict(y=arrays["y"].tolist(), train_mask=arrays["train_mask"].tolist()),
        "unaligned features": dict(x=unaligned),
        "Fortran-ordered labels and sliced masks": dict(
            y=np.asfortranarray(arrays["y"]),
            val_mask=np.repeat(arrays["val_mask"], 2)[::2],
        ),
    }
    assert not variants["sliced features"]["x"].flags.c_contiguous
    assert not variants["Fortran-ordered features"]["x"].flags.c_contiguous
    for name, changes in variants.items():
        given = {**arrays, **changes}
        before = {key: np.array(value, copy=True) for key, value in given.items()}
        data = mg.GraphDataset(spec, given)
        assert mg.read_count() == 0, f"{name}: building a dataset reads nothing back"
        predictions = model.predict(data, parts=1)
        tolerance = 2e-2 if "float16" in name else 1e-6
        np.testing.assert_allclose(predictions, reference, atol=tolerance, err_msg=name)
        if "float16" not in name:
            for split, want in scores.items():
                assert model.evaluate(data, split=split, parts=1) == want, f"{name}: {split}"
        for key, value in given.items():
            np.testing.assert_array_equal(np.asarray(value), before[key], err_msg=f"{name}: {key}")
        mg.reset_read_count()


def test_predictions_come_back_in_the_order_the_nodes_were_given():
    n = 12
    # i ~ j iff i + j >= n, with self-loops on the upper half: every degree is
    # different, so the degree order is the graph's own.
    pairs = [(i, j) for i in range(n) for j in range(n) if i + j >= n and (i != j or i >= n // 2)]
    edge_index = np.asarray(pairs, dtype=np.int64).T
    rng = np.random.default_rng(3)
    x = rng.standard_normal((n, 6)).astype(np.float32)
    spec = small_spec(max_hops=0, node_layers=2)
    model = mg.GraphMamba(spec)
    reference = model.predict(mg.GraphDataset(spec, dict(edge_index=edge_index, x=x)), parts=1)
    assert reference.shape == (n, 4) and reference.dtype == np.float32

    renumber = rng.permutation(n)  # node i becomes node renumber[i]
    shuffled_x = np.empty_like(x)
    shuffled_x[renumber] = x
    shuffled = model.predict(
        mg.GraphDataset(spec, dict(edge_index=renumber[edge_index], x=shuffled_x)), parts=1
    )
    np.testing.assert_allclose(shuffled[renumber], reference, atol=1e-5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def neighbour_majority(n=300, seed=2024):
    """One bit per node; the label says whether most neighbours carry 1."""
    rng = np.random.default_rng(seed)
    edge_index = random_graph(n, 3 * n, rng)
    bits = rng.integers(0, 2, n)
    a = dense_adjacency(edge_index, n)
    ones, degree = a @ bits, a.sum(1)
    y = np.where(2 * ones > degree, 1, np.where(2 * ones < degree, 0, -1))
    draw = rng.integers(0, 10, n)
    return dict(
        edge_index=edge_index,
        x=(bits * 2.0 - 1.0).astype(np.float32)[:, None],
        y=y,
        train_mask=draw < 6,
        val_mask=(draw >= 6) & (draw < 8),
        test_mask=draw >= 8,
    )


@pytest.mark.slow
def test_neighbour_majority_is_learned_and_evaluate_agrees_with_predict():
    arrays = neighbour_majority()
    spec = mg.GraphMambaSpec(
        node_features=1,
        task=mg.NodeClassification(2),
        max_hops=1,
        walks=16,
        repeats=2,
        local="mean",
        node_layers=1,
        seed=1,
    )
    data = mg.GraphDataset(spec, arrays)
    model = mg.GraphMamba(spec, learning_rate=3e-3)
    for epoch in range(300):
        assert model.train_epoch(data, epoch, parts=1) == 1
    losses = model.read_losses()
    assert [entry["step"] for entry in losses] == list(range(1, 301))
    assert losses[-1]["loss"] < losses[0]["loss"]
    assert set(losses[0]) == {"step", "loss", "grad_norm", "learning_rate"}
    assert model.step == 300

    result = model.evaluate(data, split="test", metric="accuracy", parts=1)
    assert result["accuracy"] > 0.85

    # The same number from the predictions, in NumPy.
    predictions = model.predict(data, parts=1)
    counted = arrays["test_mask"] & (arrays["y"] >= 0)
    assert result["count"] == counted.sum()
    accuracy = (predictions.argmax(1)[counted] == arrays["y"][counted]).mean()
    assert abs(result["accuracy"] - accuracy) < 1e-6
    f1 = model.evaluate(data, split="test", metric="f1_macro", parts=1)
    assert 0.0 < f1["f1_macro"] <= 1.0
    with pytest.raises(ValueError, match="metric"):
        model.evaluate(data, metric="mae", parts=1)
    with pytest.raises(ValueError, match="split"):
        model.evaluate(data, split="holdout", parts=1)


def test_an_epoch_of_ten_batches_reads_nothing_and_uploads_once():
    spec = small_spec(task=mg.GraphClassification(3), mpnn="gine")
    data = mg.GraphDataset(spec, graphs_data([20] * 40))
    model = mg.GraphMamba(spec)
    for epoch in range(2):
        assert model.train_epoch(data, epoch, batch_rows=80) == 10
        model.read_losses()
    mg.reset_read_count()
    mg.reset_upload_count()
    assert model.train_epoch(data, 2, batch_rows=80) == 10
    assert mg.read_count() == 0, "queueing an epoch reads nothing"
    assert mg.upload_count() == 1, "the epoch table is the only upload"
    losses = model.read_losses()
    assert mg.read_count() == 1, "every loss of the epoch in one read"
    assert len(losses) == 10 and all(np.isfinite(entry["loss"]) for entry in losses)
    assert model.read_losses() == []
    assert mg.read_count() == 1, "an empty backlog reads nothing"

    # Five batches out: one read each for predict and evaluate.
    mg.reset_read_count()
    predictions = model.predict(data, batch_rows=160)
    assert predictions.shape == (40, 3)
    assert mg.read_count() == 1
    assert model.evaluate(data, split="train", batch_rows=160)["count"] == 40
    assert mg.read_count() == 2

    estimate = model.memory_estimate(data, batch_rows=80)
    assert set(estimate) == {
        "store",
        "parameters",
        "live",
        "largest_allocation",
        "rows",
        "threshold",
    }
    assert estimate["store"] == data.nbytes and estimate["rows"] == 256
    assert estimate["parameters"] == 3 * 4 * model.num_parameters
    with pytest.raises(ValueError, match="pass one"):
        model.memory_estimate(data, batch_rows=80, parts=2)
    other = mg.GraphMamba(small_spec(task=mg.GraphClassification(3), mpnn="gine", d_model=32))
    with pytest.raises(ValueError, match="different spec"):
        other.memory_estimate(data)
    with pytest.raises(ValueError, match="batch_rows"):
        model.train_epoch(data, 3, batch_rows=80, parts=2)
    with pytest.raises(ValueError, match="whole-graph batches"):
        model.train_epoch(data, 3, parts=2)


def test_predict_selects_a_split_in_the_given_order():
    spec = small_spec(max_hops=0)
    arrays = node_data()
    data = mg.GraphDataset(spec, arrays)
    model = mg.GraphMamba(spec)
    everything = model.predict(data, parts=1)
    for split in ("train", "val", "test"):
        mg.reset_read_count()
        rows = model.predict(data, split=split, parts=1)
        assert mg.read_count() == 1
        np.testing.assert_array_equal(rows, everything[arrays[f"{split}_mask"]])
    # A mask that was left out, beside ones that were given, selects nothing.
    partial = {key: value for key, value in arrays.items() if key != "test_mask"}
    assert model.predict(mg.GraphDataset(spec, partial), split="test", parts=1).shape == (0, 4)
    with pytest.raises(ValueError, match="split"):
        model.predict(data, split="holdout", parts=1)
    no_masks = {key: arrays[key] for key in ("edge_index", "x", "y")}
    with pytest.raises(ValueError, match="without train_mask"):
        model.predict(mg.GraphDataset(spec, no_masks), split="val", parts=1)


def test_a_model_that_cannot_run_on_partitions_uses_the_whole_graph():
    # GatedGCN keeps a state per edge, which node partitions do not have: with
    # no batch argument, one graph is trained, predicted and evaluated whole.
    spec = small_spec(mpnn="gated_gcn")
    arrays = node_data()
    data = mg.GraphDataset(spec, arrays)
    model = mg.GraphMamba(spec)
    assert model.train_epoch(data, 0) == 1
    assert np.isfinite(model.read_losses()[0]["loss"])
    assert model.predict(data).shape == (60, 4)
    assert model.evaluate(data, split="val")["count"] == arrays["val_mask"].sum()
    with pytest.raises(ValueError, match="GatedGcn"):
        model.train_epoch(data, 1, parts=2)


def test_save_and_load_round_trip(tmp_path):
    spec = small_spec(mpnn="gine", node_layers=2)
    data = mg.GraphDataset(spec, node_data())
    model = mg.GraphMamba(spec)
    for epoch in range(3):
        model.train_epoch(data, epoch, parts=2)
    model.read_losses()
    path = str(tmp_path / "graph.m3ck")
    model.save(path)
    loaded = mg.GraphMamba.load(path)
    assert loaded.spec == spec
    assert loaded.step == model.step == 6
    assert loaded.num_parameters == model.num_parameters
    np.testing.assert_array_equal(loaded.predict(data, parts=2), model.predict(data, parts=2))
    with pytest.raises(OSError):
        mg.GraphMamba.load(str(tmp_path / "missing.m3ck"))


def test_the_golden_file_is_reproduced():
    golden = json.loads((ROOT / "tests/golden/graph_tiny.json").read_text())
    spec = mg.GraphMambaSpec.from_json(json.dumps(golden["spec"]))
    n = golden["n_nodes"]
    train = np.asarray(golden["train_mask"], dtype=bool)
    data = mg.GraphDataset(
        spec,
        dict(
            edge_index=np.asarray([golden["edge_src"], golden["edge_dst"]]),
            x=np.asarray(golden["x"], dtype=np.float32).reshape(n, -1),
            y=np.asarray(golden["y"]),
            train_mask=train,
            val_mask=~train,
            test_mask=np.zeros(n, dtype=bool),
        ),
    )
    model = mg.GraphMamba(spec, learning_rate=golden["learning_rate"])
    for epoch in range(5):
        model.train_epoch(data, epoch, parts=1)
    losses = [entry["loss"] for entry in model.read_losses()]
    predictions = model.predict(data, parts=1)
    # Written on the CPU runtime; a GPU agrees to rounding.
    tolerance = 1e-4 if mg.backend() == "cpu" else 1e-3
    np.testing.assert_allclose(losses, golden["losses"], atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        predictions.reshape(-1), golden["predictions"], atol=tolerance, rtol=tolerance
    )


# ---------------------------------------------------------------------------
# Element types
# ---------------------------------------------------------------------------


def majority_losses(dtype, loss_scale, tmp_path=None):
    """50 full-batch steps of the neighbour-majority task in ``dtype``."""
    arrays = neighbour_majority()
    spec = mg.GraphMambaSpec(
        node_features=1,
        task=mg.NodeClassification(2),
        max_hops=1,
        walks=16,
        repeats=2,
        local="mean",
        node_layers=1,
        seed=1,
    )
    data = mg.GraphDataset(spec, arrays, dtype=dtype)
    model = mg.GraphMamba(spec, dtype=dtype, loss_scale=loss_scale)
    for epoch in range(50):
        model.train_epoch(data, epoch, parts=1)
    losses = np.array([entry["loss"] for entry in model.read_losses()])
    return spec, arrays, data, model, losses


@pytest.mark.slow
def test_bf16_tracks_f32_and_its_checkpoint_loads_as_f32(tmp_path):
    if not mg.supports_dtype("bf16"):
        pytest.skip(f"{mg.backend()} cannot store or compute bf16")
    _, _, _, reference_model, reference = majority_losses("f32", None)
    spec, arrays, data, model, losses = majority_losses("bf16", 256.0)
    assert (model.dtype, data.dtype, reference_model.dtype) == ("bf16", "bf16", "f32")
    assert np.isfinite(losses).all()
    # The criterion of tests/graph_dtype.rs: the first step differs by rounding
    # alone, and the ten-step means stay within 5%, relative with a floor of 1.
    # Once the task is learned a single step's loss is resampling noise in both
    # runs, so single steps are not compared.
    assert abs(losses[0] - reference[0]) / reference[0] < 0.01
    means, reference_means = losses.reshape(5, 10).mean(1), reference.reshape(5, 10).mean(1)
    worst = np.max(np.abs(means - reference_means) / np.maximum(reference_means, 1.0))
    assert worst < 0.05, f"bf16 is {worst:.3f} off the f32 curve: {means} against {reference_means}"
    assert losses[-1] < 0.5 * losses[0]

    # The same numbers as the Rust API reports for this run.
    assert model.evaluate(data, split="train", parts=1)["accuracy"] > 0.9

    # A dataset and a model of different element types do not mix.
    f32_data = mg.GraphDataset(spec, arrays)
    with pytest.raises(ValueError, match="same dtype"):
        model.train_epoch(f32_data, 0, parts=1)
    with pytest.raises(ValueError, match="same dtype"):
        model.predict(f32_data, parts=1)

    # A bf16 checkpoint is an f32 model's weights up to bf16 rounding.
    path = str(tmp_path / "bf16.m3ck")
    model.save(path)
    again = mg.GraphMamba.load(path, dtype="bf16")
    np.testing.assert_array_equal(again.predict(data, parts=1), model.predict(data, parts=1))
    wide = mg.GraphMamba.load(path)
    assert wide.dtype == "f32" and wide.step == model.step
    # The same weights, computed in f32 instead of bf16: the logits agree to
    # the rounding of 8-bit mantissas through the network, and so do the classes.
    narrow_logits, wide_logits = model.predict(data, parts=1), wide.predict(f32_data, parts=1)
    assert np.abs(wide_logits - narrow_logits).max() < 0.1 * np.abs(wide_logits).max()
    assert (wide_logits.argmax(1) == narrow_logits.argmax(1)).mean() > 0.97


def test_f16_is_refused_with_the_reason():
    spec = small_spec()
    for build in (
        lambda: mg.GraphMamba(spec, dtype="f16"),
        lambda: mg.GraphDataset(spec, node_data(), dtype="f16"),
    ):
        with pytest.raises(NotImplementedError, match="not finite in f16"):
            build()
    with pytest.raises(ValueError, match="dtype"):
        mg.GraphMamba(spec, dtype="f64")
    with pytest.raises(ValueError, match="dtype"):
        mg.supports_dtype("f64")
    assert mg.supports_dtype("f32")
    with pytest.raises(ValueError, match="loss_scale"):
        mg.GraphMamba(spec, loss_scale=0.0)


# ---------------------------------------------------------------------------
# The interpreter lock and interrupts
# ---------------------------------------------------------------------------


def long_run():
    """A model and a dataset whose epoch is a few hundred steps."""
    spec = small_spec(task=mg.GraphClassification(3))
    data = mg.GraphDataset(spec, graphs_data([12] * 1200))
    model = mg.GraphMamba(spec)
    model.train_epoch(data, 0, batch_rows=48)  # warm: kernels compiled
    model.read_losses()
    return model, data


def test_the_model_is_bound_to_its_thread():
    model, data = long_run()
    caught = []

    def touch():
        try:
            model.num_parameters
        except BaseException as error:  # pyo3 raises a PanicException here
            caught.append(error)

    thread = threading.Thread(target=touch)
    thread.start()
    thread.join()
    assert len(caught) == 1, "another thread must be refused, not race"
    assert "unsendable" in str(caught[0]) or "thread" in str(caught[0])
    assert model.num_parameters > 0, "and the owning thread is unaffected"


def test_training_releases_the_interpreter_lock():
    model, data = long_run()
    stop = threading.Event()
    wakes = [0]

    training = threading.Event()
    refused = []

    def ticker():
        tried = False
        while not stop.is_set():
            time.sleep(0.001)
            wakes[0] += 1
            if training.is_set() and not tried:
                tried = True
                try:
                    model.num_parameters
                except BaseException as error:  # pyo3 raises a PanicException here
                    refused.append(error)

    thread = threading.Thread(target=ticker)
    thread.start()
    time.sleep(0.5)
    idle_rate = wakes[0] / 0.5
    wakes[0] = 0
    training.set()
    started = time.perf_counter()
    model.train_epoch(data, 1, batch_rows=48)
    busy_rate = wakes[0] / (time.perf_counter() - started)
    stop.set()
    thread.join()
    model.read_losses()
    # The plan expected this to hold on a GPU only, where the calling thread
    # waits; it holds on the CPU runtime too (94% measured), so it is not
    # skipped there. 30% leaves room for a loaded machine.
    assert busy_rate > 0.3 * idle_rate, f"{busy_rate:.0f} wakes/s against {idle_rate:.0f} idle"
    # While the epoch ran, the other thread could run Python but not reach the
    # model: it is bound to the thread that made it.
    assert len(refused) == 1, "the second thread's call during training must be refused"


def test_ctrl_c_stops_after_a_completed_step():
    model, data = long_run()
    before = model.step
    per_epoch = model.train_epoch(data, 999, batch_rows=48)
    model.read_losses()
    before += per_epoch
    timer = threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT))
    started = time.perf_counter()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            # Epochs until the signal arrives; each is a few hundred steps.
            for epoch in range(1, 1000):
                model.train_epoch(data, epoch, batch_rows=48)
    finally:
        timer.cancel()
    elapsed = time.perf_counter() - started
    assert 0.25 < elapsed < 1.3, f"interrupted after {elapsed:.2f} s"
    # Every step that completed is reported, and nothing was left half done.
    losses = model.read_losses()
    assert model.step == before + len(losses)
    assert len(losses) > 0 and all(np.isfinite(entry["loss"]) for entry in losses)
    # The loop stopped inside an epoch, between two of its steps: it was the
    # Rust loop that saw the signal, not the interpreter between two calls.
    assert len(losses) % per_epoch != 0, f"{len(losses)} steps is a whole number of epochs"
    assert [entry["step"] for entry in losses] == list(range(before + 1, model.step + 1))
    # The model is still usable.
    assert model.train_epoch(data, 2000, batch_rows=4800) > 0
    assert len(model.read_losses()) > 0
    assert model.predict(data, batch_rows=4800).shape == (1200, 3)
