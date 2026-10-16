"""Graph matching from Python: ``MatchQueries``, ``MatchGroups`` and
``GraphMatchTrainer``. The Rust suite (``tests/graph_match.rs``) proves the
loss, the gradients and the learning; these prove the boundary."""

import numpy as np
import pytest

import mamba3_graph as mg

N_BITS = 64
D_OUT = 8


def dataset(sizes, seed=0):
    """Small connected graphs with two categorical node fields and one edge
    field."""
    rng = np.random.default_rng(seed)
    src, dst, offset = [], [], 0
    for size in sizes:
        for v in range(1, size):
            src += [offset + v - 1, offset + v]
            dst += [offset + v, offset + v - 1]
        offset += size
    edge_index = np.asarray([src, dst], dtype=np.int64)
    x = np.stack([rng.integers(0, 5, offset), rng.integers(0, 3, offset)], axis=1).astype(np.uint8)
    edge_attr = (np.arange(edge_index.shape[1]) % 2).astype(np.uint8)[:, None]
    graph_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    return dict(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_ptr=graph_ptr)


def spec():
    return mg.GraphMambaSpec(
        node_features=mg.Categorical([5, 3]),
        task=mg.GraphRegression(D_OUT, pool="mean"),
        edge_features=mg.Categorical([2]),
        d_model=16,
        max_hops=1,
        walks=2,
        repeats=1,
        node_layers=1,
        mpnn="gine",
        seed=1,
    )


@pytest.fixture
def setup():
    sizes = [4, 6, 3, 5, 4, 7, 5, 6]
    data = mg.GraphDataset(spec(), dataset(sizes), symmetrize=True)
    bits = np.array([[0, 5, 9, 0], [1, 2, 63, 0], [7, 7, 0, 0]], dtype=np.uint32)
    weights = np.array([[1, 0.5, 2, 0], [1, 1, 0.25, 0], [1, 1, 0, 0]], dtype=np.float32)
    queries = mg.MatchQueries(bits=bits, weights=weights, n_bits=N_BITS)
    groups = mg.MatchGroups(
        query=np.array([0, 1, 2, 0], dtype=np.uint32),
        member_ptr=np.array([0, 3, 5, 9, 10], dtype=np.uint32),
        members=np.array([0, 1, 2, 1, 3, 4, 5, 6, 4, 7], dtype=np.uint32),
    )
    return data, queries, groups


def test_queries_and_groups(setup):
    _, queries, groups = setup
    assert (queries.num_queries, queries.k, queries.n_bits) == (3, 4, N_BITS)
    assert len(groups) == groups.num_groups == 4
    np.testing.assert_array_equal(groups.members(2), [4, 5, 6])
    np.testing.assert_array_equal(groups.trivial(), [3])
    with pytest.raises(ValueError, match="bit 64"):
        mg.MatchQueries(np.array([[64]], dtype=np.uint32), np.ones((1, 1), np.float32), n_bits=64)
    with pytest.raises(ValueError):
        mg.MatchQueries(np.zeros((1, 257), np.uint32), np.ones((1, 257), np.float32))
    with pytest.raises(ValueError):
        mg.MatchGroups([0], [0, 2], [0])
    # a pointer beyond the members in the middle of the sequence is an error, not a panic
    with pytest.raises(ValueError, match="beyond"):
        mg.MatchGroups([0, 1], [0, 3, 2], [0, 1])


def test_train_score_embed_and_reload(setup, tmp_path):
    data, queries, groups = setup
    trainer = mg.GraphMatchTrainer(
        spec(), n_bits=N_BITS, query_hidden=32, learning_rate=1e-3, weight_decay=0.01, seed=2
    )
    assert trainer.d_out == D_OUT and trainer.n_bits == N_BITS
    assert abs(trainer.scale - 1 / 0.07) < 1e-3
    steps = sum(trainer.train_epoch(data, queries, groups, batches=[[0, 1], [2]], epoch=e)
                for e in range(3))
    assert steps == 6 and trainer.step == 6
    losses = trainer.read_losses()
    assert len(losses) == 6 and all(np.isfinite(losses))
    assert trainer.read_losses() == []

    with pytest.raises(ValueError, match="no negative"):
        trainer.train_epoch(data, queries, groups, batches=[[3]])

    scores = trainer.score_groups(data, queries, groups, [[0, 1, 2], [3]])
    assert [s.shape for s in scores] == [(3,), (2,), (3,), (1,)]
    assert all(s.dtype == np.float32 for s in scores)
    again = trainer.score_groups(data, queries, groups, [[0, 1, 2], [3]])
    for a, b in zip(scores, again):
        np.testing.assert_array_equal(a, b)

    z = trainer.embed_graphs(data)
    assert z.shape == (8, D_OUT) and z.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, rtol=1e-4)
    some = trainer.embed_graphs(data, graph_ids=np.array([7, 0]), batch_graphs=1)
    np.testing.assert_allclose(some, z[[7, 0]], atol=1e-5)
    q = trainer.embed_queries(queries)
    assert q.shape == (3, D_OUT)
    # The scores are the scaled cosines of the embeddings.
    for g, s in enumerate(scores):
        want = trainer.scale * z[groups.members(g)] @ q[groups.query(g)]
        np.testing.assert_allclose(s, want, atol=1e-4)

    path = str(tmp_path / "matcher.ckpt")
    trainer.save(path)
    loaded = mg.GraphMatchTrainer.load(path)
    assert loaded.step == 6
    assert loaded.spec == trainer.spec
    for a, b in zip(loaded.score_groups(data, queries, groups, [[0, 1, 2], [3]]), scores):
        np.testing.assert_array_equal(a, b)


def test_residual_offsets(setup, tmp_path):
    data, queries, _ = setup
    offsets = np.array([0.5, -1.0, 2.0, 0.25, 0.0, -0.5, 1.5, 3.0, 9.0, 0.0], dtype=np.float32)
    query, ptr = np.array([0, 1, 2, 0]), np.array([0, 3, 5, 9, 10])
    members = np.array([0, 1, 2, 1, 3, 4, 5, 6, 4, 7])
    groups = mg.MatchGroups(query, ptr, members, offsets)
    assert groups.has_offsets
    np.testing.assert_array_equal(groups.offsets(2), [-0.5, 1.5, 3.0])     # the repeated member's offset is dropped
    with pytest.raises(ValueError, match="aligned"):
        mg.MatchGroups(query, ptr, members, offsets[:-1])
    with pytest.raises(ValueError, match="non-finite"):
        mg.MatchGroups(query, ptr, members, np.where(np.arange(10) == 3, np.nan, offsets))
    plain = mg.GraphMatchTrainer(spec(), n_bits=N_BITS, query_hidden=32, seed=2)
    assert plain.offset_weight is None and not plain.residual
    with pytest.raises(ValueError, match="not residual"):
        plain.score_groups(data, queries, groups, [[0, 1]])
    trainer = mg.GraphMatchTrainer(spec(), n_bits=N_BITS, query_hidden=32, seed=2, residual=True)
    assert trainer.residual and trainer.offset_weight == 1.0
    with pytest.raises(ValueError, match="needs member offsets"):
        trainer.score_groups(data, queries, mg.MatchGroups(query, ptr, members), [[0, 1]])
    scores = trainer.score_groups(data, queries, groups, [[0, 1, 2], [3]])
    z, q = trainer.embed_graphs(data), trainer.embed_queries(queries)
    for g, s in enumerate(scores):
        want = trainer.scale * z[groups.members(g)] @ q[groups.query(g)] + groups.offsets(g)
        np.testing.assert_allclose(s, want, atol=1e-4)
    trainer.train_epoch(data, queries, groups, batches=[[0, 1, 2]], epoch=0)
    assert all(np.isfinite(trainer.read_losses()))
    path = str(tmp_path / "residual.ckpt")
    trainer.save(path)
    loaded = mg.GraphMatchTrainer.load(path)
    assert loaded.residual and loaded.offset_weight == trainer.offset_weight
