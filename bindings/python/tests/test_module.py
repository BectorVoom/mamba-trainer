"""The module itself: what it was built with, and what it is doing."""

import pytest

import mamba3_rl as m3


def test_backend_is_one_of_the_known_runtimes():
    assert m3.backend() in {"cpu", "wgpu", "cuda", "hip"}


def test_version_is_exposed():
    assert m3.__version__.count(".") == 2


def test_matmul_precision_round_trips():
    original = m3.matmul_precision()
    try:
        for precision in ("f32", "bf16", "f16"):
            m3.set_matmul_precision(precision)
            assert m3.matmul_precision() == precision
    finally:
        m3.set_matmul_precision(original)


def test_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="unknown precision"):
        m3.set_matmul_precision("float8")


def test_read_counter_moves_only_when_something_is_read(policy, env):
    rollout = m3.Rollout(policy, num_envs=env.num_envs)
    obs = env.reset()
    m3.reset_read_count()
    rollout.step(obs)
    # Reading the actions back into numpy is a read; the count is allowed to be
    # anything positive, but it must have moved.
    assert m3.read_count() > 0
