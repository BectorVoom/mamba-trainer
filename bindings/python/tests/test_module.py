"""The module itself: what it was built with, and what it is doing."""

import os
import subprocess
import sys

import pytest

import mamba3_rl as m3


# `backend()` names the runtime and, for wgpu, the shader language it compiles
# to: "wgpu<wgsl>", "wgpu<msl>", "wgpu<spirv>".
WGSL = "wgsl" in m3.backend()


def test_backend_is_one_of_the_known_runtimes():
    assert m3.backend().split("<")[0] in {"cpu", "wgpu", "cuda", "hip"}


def test_version_is_exposed():
    assert m3.__version__.count(".") == 2


def test_matmul_precision_round_trips():
    original = m3.matmul_precision()
    # WGSL has no bf16 type; that refusal is its own test below.
    precisions = ("f32", "f16") if WGSL else ("f32", "bf16", "f16")
    try:
        for precision in precisions:
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


# ---------------------------------------------------------------------------
# A7: MAMBA3_MATMUL_PRECISION, checked at import time.
#
# Every case below needs a *fresh interpreter*: once `mamba3_rl` is imported
# in this process the environment variable has already been read, and the
# process-global precision it may have set cannot be un-set by re-importing.
# ---------------------------------------------------------------------------


def _import_with_env(value):
    """Import the module in a subprocess with `MAMBA3_MATMUL_PRECISION` set to
    `value` (or removed if `None`), and report what happened."""
    env = os.environ.copy()
    if value is None:
        env.pop("MAMBA3_MATMUL_PRECISION", None)
    else:
        env["MAMBA3_MATMUL_PRECISION"] = value
    return subprocess.run(
        [sys.executable, "-c", "import mamba3_rl as m3; print(m3.matmul_precision())"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_an_unset_precision_env_var_leaves_f32_and_touches_no_device():
    result = _import_with_env(None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "f32"


_NO_BF16 = pytest.mark.skipif(WGSL, reason="WGSL has no bf16; see the WGSL refusal test")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("f32", "f32"),
        pytest.param("bf16", "bf16", marks=_NO_BF16),
        ("f16", "f16"),
        # Case-insensitive, and agreeing with `set_matmul_precision`'s own policy.
        ("F32", "f32"),
        pytest.param("BF16", "bf16", marks=_NO_BF16),
        ("F16", "f16"),
    ],
)
def test_a_supported_precision_env_var_sets_the_mode(value, expected):
    result = _import_with_env(value)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_an_unrecognised_precision_env_var_fails_import_with_an_actionable_message():
    result = _import_with_env("float8")
    assert result.returncode != 0
    assert "MAMBA3_MATMUL_PRECISION" in result.stderr
    assert "float8" in result.stderr
    # Not a silent fallback: the process never got as far as printing a mode.
    assert result.stdout == ""


@pytest.mark.skipif(
    "wgsl" not in m3.backend(),
    reason="bf16 is only unsupported on a WGSL build; this wheel is not one",
)
def test_unsupported_wgsl_bf16_is_refused_without_a_worker_panic():
    result = _import_with_env("bf16")
    assert result.returncode != 0
    assert "cannot compile" in result.stderr
    # A worker-thread abort inside the shader compiler would not come back as an
    # orderly Python exception at all -- this asserts the failure is one.
    assert "PyValueError" in result.stderr or "ValueError" in result.stderr


def test_a_failed_import_does_not_poison_a_later_supported_value():
    bad = _import_with_env("nonsense")
    assert bad.returncode != 0

    good = _import_with_env("f16")
    assert good.returncode == 0, good.stderr
    assert good.stdout.strip() == "f16"


def test_an_explicit_call_overrides_the_environment_default():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mamba3_rl as m3; m3.set_matmul_precision('f32'); print(m3.matmul_precision())",
        ],
        env={**os.environ, "MAMBA3_MATMUL_PRECISION": "f16"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "f32"
