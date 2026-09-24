"""A numpy-only reader for a policy ``Policy.export_numpy`` wrote.

Training needs the compiled extension; acting does not have to. A game agent
that asks for one action per turn wants a plain Python file with no toolchain,
and the recurrence that makes a state space model worth using is also what makes
that step cheap: a fixed-size state, no history to re-read. This module is the
recurrent path (``Mamba3Policy::step``) transcribed into numpy, flat and
structured policies alike, and imports nothing from the extension::

    from mamba3_rl import numpy_ref

    policy = numpy_ref.Policy.load("policy.npz")
    state = policy.initial_state(num_envs=1)
    logits, value, state = policy.step(obs[None, :], state)

The transcription follows the *composed* form of every kernel, the readable
statement the crate's fused kernels are tested against, and the test suite
holds it to ``Rollout.evaluate`` on the same weights. Keep it importable without
the compiled module: copy this one file next to an agent and it runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Policy", "MixerCache", "State"]

TWO_PI = 2.0 * math.pi
# What `mask_logits` writes where an action is illegal: the most negative finite
# float, a probability of exactly zero after a softmax.
MASKED = np.finfo(np.float32).min


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    # log1p(exp(x)) without overflowing for large x.
    return np.logaddexp(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rms_norm(x: np.ndarray, gain: np.ndarray | None, eps: float) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    out = x * scale
    return out if gain is None else out * gain


class _Params:
    """Weights keyed by parameter path, viewed from a prefix."""

    def __init__(self, arrays: dict[str, np.ndarray], prefix: str = ""):
        self._arrays = arrays
        self._prefix = prefix

    def child(self, name: str) -> "_Params":
        return _Params(self._arrays, f"{self._prefix}{name}.")

    def has(self, name: str) -> bool:
        return f"{self._prefix}{name}" in self._arrays

    def get(self, name: str) -> np.ndarray:
        key = f"{self._prefix}{name}"
        array = self._arrays.get(key)
        if array is None:
            raise KeyError(f"the export has no parameter `{key}`")
        return array

    def maybe(self, name: str) -> np.ndarray | None:
        return self.get(name) if self.has(name) else None


class _Linear:
    """``x @ weight + bias``, matching ``nn::linear::Linear``."""

    def __init__(self, params: _Params):
        self.weight = params.get("weight")
        self.bias = params.maybe("bias")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.weight
        return out if self.bias is None else out + self.bias


class _CausalConv1d:
    """Depthwise causal convolution, one position at a time.

    ``weight`` is ``[taps, channels]`` with the newest tap last.
    """

    def __init__(self, params: _Params):
        self.weight = params.get("weight")
        self.bias = params.maybe("bias")
        self.taps, self.channels = self.weight.shape

    def empty_history(self, batch: int) -> np.ndarray:
        return np.zeros((batch, self.taps - 1, self.channels), dtype=np.float32)

    def step(self, x: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        window = np.concatenate([history, x[:, None, :]], axis=1)
        out = np.einsum("bkc,kc->bc", window, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out, window[:, 1:, :]


@dataclass
class MixerCache:
    """One layer's recurrent state: ``h``, the previous input product, the
    rotating frame's angle and the convolution taps."""

    h: np.ndarray
    last_u: np.ndarray
    angle: np.ndarray | None
    conv: np.ndarray | None


State = list[MixerCache]


def _rank(ssm: dict) -> int:
    mode = ssm.get("mode", "Siso")
    if isinstance(mode, dict) and "Mimo" in mode:
        return int(mode["Mimo"]["rank"])
    return 1


class _Mixer:
    """``models::mamba3::Mamba3Mixer``, recurrent path only."""

    def __init__(self, params: _Params, ssm: dict):
        self.heads = ssm["n_heads"]
        self.head_dim = ssm["head_dim"]
        self.d_state = ssm["d_state"]
        self.groups = ssm["n_groups"]
        self.rank = _rank(ssm)
        self.rotational = ssm["dynamics"] == "Rotational"
        self.learned_lambda = ssm["discretization"] == "LearnedTrapezoid"
        self.fixed_lambda = {"Euler": 1.0, "Trapezoid": 0.5}.get(ssm["discretization"])

        self.in_proj = _Linear(params.child("in_proj"))
        self.out_proj = _Linear(params.child("out_proj"))
        self.conv = _CausalConv1d(params.child("conv")) if params.has("conv.weight") else None
        self.dt_bias = params.get("dt_bias")
        self.a_log = params.get("a_log")
        self.d_skip = params.maybe("d")
        self.b_bias = params.maybe("b_bias")
        self.c_bias = params.maybe("c_bias")
        self.bc_norm = params.maybe("bc_norm.weight")
        self.post_gate_norm = params.maybe("post_gate_norm.weight")
        # Both inner norms are plain `RmsNormConfig::new`, which keeps the 1e-5
        # default rather than the block's `norm_eps`.
        self.inner_eps = 1e-5

    @property
    def d_inner(self) -> int:
        return self.heads * self.head_dim * self.rank

    @property
    def bc_width(self) -> int:
        return self.groups * self.d_state * self.rank

    def empty_cache(self, batch: int) -> MixerCache:
        h = np.zeros((batch, self.heads, self.head_dim, self.d_state), dtype=np.float32)
        angle = (
            np.zeros((batch, self.heads, self.d_state // 2), dtype=np.float32)
            if self.rotational
            else None
        )
        conv = self.conv.empty_history(batch) if self.conv else None
        return MixerCache(h, np.zeros_like(h), angle, conv)

    def _to_heads(self, flat: np.ndarray, batch: int) -> np.ndarray:
        """``[batch, groups * rank * state]`` -> ``[batch, heads, rank, state]``."""
        grouped = flat.reshape(batch, self.groups, self.rank, self.d_state)
        if self.groups == self.heads:
            return grouped
        return np.repeat(grouped, self.heads // self.groups, axis=1)

    def _rotate(self, v: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """Rotate the two halves of the state axis of ``[batch, heads, state, rank]``
        by ``-angle``, as ``Var::rotate_by_angle`` does."""
        cos = np.cos(angle)[:, :, :, None]
        sin = np.sin(angle)[:, :, :, None]
        half = v.shape[2] // 2
        x1, x2 = v[:, :, :half, :], v[:, :, half:, :]
        return np.concatenate([x1 * cos + x2 * sin, x2 * cos - x1 * sin], axis=2)

    def step(
        self, x_in: np.ndarray, cache: MixerCache, reset: np.ndarray | None
    ) -> tuple[np.ndarray, MixerCache]:
        batch = x_in.shape[0]
        heads, p, n, rk = self.heads, self.head_dim, self.d_state, self.rank
        d_inner, bc = self.d_inner, self.bc_width

        widths = [d_inner, d_inner + 2 * bc, heads]
        if self.learned_lambda:
            widths.append(heads)
        if self.rotational:
            widths.append(heads * n // 2)
        pieces = np.split(self.in_proj(x_in), np.cumsum(widths)[:-1], axis=-1)
        z, xbc, dt_raw = pieces[0], pieces[1], pieces[2]
        rest = list(pieces[3:])
        lambda_raw = rest.pop(0) if self.learned_lambda else None
        theta_raw = rest.pop(0) if self.rotational else None

        # A reset drops the convolution taps that reach into the previous
        # episode, which for one step is a zero history.
        conv_history = cache.conv
        if self.conv is not None:
            if reset is not None:
                conv_history = conv_history * (1.0 - reset)[:, None, None]
            xbc, conv_history = self.conv.step(xbc, conv_history)
        xbc = silu(xbc)

        x = xbc[:, :d_inner].reshape(batch, heads, p, rk)
        b = self._to_heads(xbc[:, d_inner:d_inner + bc], batch)
        c = self._to_heads(xbc[:, d_inner + bc:], batch)
        if self.b_bias is not None:
            b = b + self.b_bias[None, :, None, :]
        if self.c_bias is not None:
            c = c + self.c_bias[None, :, None, :]
        if self.bc_norm is not None:
            b = rms_norm(b, self.bc_norm, self.inner_eps)
            c = rms_norm(c, self.bc_norm, self.inner_eps)
        # The scan wants the rank axis last: [batch, heads, state, rank].
        b = np.swapaxes(b, 2, 3)
        c = np.swapaxes(c, 2, 3)

        dt = softplus(dt_raw + self.dt_bias)
        lam = (
            sigmoid(lambda_raw)
            if lambda_raw is not None
            else np.full((batch, heads), self.fixed_lambda, dtype=np.float32)
        )
        alpha = np.exp(dt * -np.exp(self.a_log))
        beta = (1.0 - lam) * dt * alpha
        g = lam * dt
        if reset is not None:
            # The two coefficients that carry the previous state are cut.
            keep = (1.0 - reset)[:, None]
            alpha = alpha * keep
            beta = beta * keep

        angle = cache.angle
        if self.rotational:
            # The angle is not reset: a common offset rotates B and C alike and
            # cancels in C·h, so a fresh episode computes the same outputs.
            raw = theta_raw.reshape(batch, heads, n // 2) * dt[:, :, None] + cache.angle
            angle = raw - np.round(raw / TWO_PI) * TWO_PI
            b = self._rotate(b, angle)
            c = self._rotate(c, angle)

        u = np.einsum("bhpr,bhnr->bhpn", x, b)
        h = (alpha[:, :, None, None] * cache.h
             + beta[:, :, None, None] * cache.last_u
             + g[:, :, None, None] * u)
        y = np.einsum("bhnr,bhpn->bhpr", c, h)
        if self.d_skip is not None:
            y = y + x * self.d_skip[None, :, None, None]

        flat = y.reshape(batch, d_inner)
        if self.post_gate_norm is not None:
            flat = rms_norm(flat, self.post_gate_norm, self.inner_eps)
        out = self.out_proj(flat * silu(z))
        return out, MixerCache(h, u, angle, conv_history)


class _Block:
    """``x + mixer(norm(x))``."""

    def __init__(self, params: _Params, ssm: dict, eps: float):
        self.norm = params.get("norm.weight")
        self.mixer = _Mixer(params.child("mixer"), ssm)
        self.eps = eps

    def step(
        self, x: np.ndarray, cache: MixerCache, reset: np.ndarray | None
    ) -> tuple[np.ndarray, MixerCache]:
        out, cache = self.mixer.step(rms_norm(x, self.norm, self.eps), cache, reset)
        return x + out, cache


class _EntityEncoder:
    """``nn::entity::EntityEncoder``: one MLP shared across a set's entities."""

    def __init__(self, params: _Params):
        self.layers = []
        while params.has(f"mlp.{len(self.layers)}.weight"):
            self.layers.append(_Linear(params.child(f"mlp.{len(self.layers)}")))
        if not self.layers:
            raise KeyError("an entity encoder with no layers")
        self.slot = params.maybe("slot")

    def __call__(self, features: np.ndarray, presence: np.ndarray) -> np.ndarray:
        x = features * presence
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x if self.slot is None else x + self.slot


def _pool(e: np.ndarray, presence: np.ndarray, kind: str) -> np.ndarray:
    """``[B, N, d]`` under ``[B, N, 1]`` presence -> ``[B, d]``; zero for an empty set."""
    count = presence.sum(axis=1, keepdims=True)
    if kind == "mean":
        weights = presence / np.maximum(count, 1.0)
        return (weights * e).sum(axis=1)
    if kind == "max":
        masked = np.where(presence == 0.0, MASKED, e)
        return masked.max(axis=1) * np.minimum(count, 1.0)[:, 0, :]
    raise ValueError(f"unknown pooling kind {kind!r}")


class _Pointer:
    """``rl::heads::PointerHead``: score the entities of one set."""

    def __init__(self, params: _Params, head: dict):
        pointer = params.child("pointer")
        self.scoring = head.get("scoring", "additive")
        if self.scoring == "additive":
            self.w_h = _Linear(pointer.child("w_h"))
            self.w_e = _Linear(pointer.child("w_e"))
            self.v = _Linear(pointer.child("v"))
        elif self.scoring == "dot":
            self.w_q = _Linear(pointer.child("w_q"))
        else:
            raise ValueError(f"unknown pointer scoring {self.scoring!r}")
        self.extra = _Linear(params.child("extra")) if params.has("extra.weight") else None

    def __call__(self, h: np.ndarray, e: np.ndarray, presence: np.ndarray) -> np.ndarray:
        if self.scoring == "additive":
            hidden = np.maximum(self.w_e(e) + self.w_h(h)[:, None, :], 0.0)
            scores = self.v(hidden)[..., 0]
        else:
            scores = np.einsum("bnd,bd->bn", e, self.w_q(h))
        logits = np.where(presence[..., 0] == 0.0, MASKED, scores)
        if self.extra is not None:
            logits = np.concatenate([logits, self.extra(h)], axis=-1)
        return logits


class Policy:
    """``rl::policy::Mamba3Policy``, the recurrent step, in numpy."""

    def __init__(self, arrays: dict[str, np.ndarray], config: dict[str, Any]):
        root = _Params(arrays)
        self.config = config
        self.obs_dim = int(config["obs_dim"])
        self.action_dim = int(config["action_dim"])
        eps = float(config.get("norm_eps", 1e-5))
        self.eps = eps
        ssm = config["ssm"]

        self.spec = config.get("obs_spec")
        if self.spec is None:
            self.encoder = _Linear(root.child("encoder"))
        else:
            self.globals = int(self.spec["globals"])
            self.sets = self.spec["sets"]
            self.encoders = [_EntityEncoder(root.child(f"entity.{s['name']}")) for s in self.sets]
            self.kinds = list(config.get("pooling", {"kinds": ["mean", "max"]})["kinds"])
            self.proj = _Linear(root.child("pool.proj"))

        self.blocks = [
            _Block(root.child(f"blocks.{i}"), ssm, eps) for i in range(int(config["n_layers"]))
        ]
        self.norm = root.get("norm.weight")
        head = config.get("action_head") or {"kind": "flat"}
        if head.get("kind", "flat") == "pointer":
            self.pointer_set = [s["name"] for s in self.sets].index(head["set"])
            self.actor = _Pointer(root.child("actor"), head)
        else:
            self.pointer_set = None
            self.actor = _Linear(root.child("actor"))
        self.critic = _Linear(root.child("critic"))

    @classmethod
    def load(cls, path: str) -> "Policy":
        """Read an ``.npz`` written by ``mamba3_rl.Policy.export_numpy``."""
        with np.load(path, allow_pickle=False) as packed:
            config = json.loads(str(packed["__config__"]))
            arrays = {
                key: np.asarray(packed[key], dtype=np.float32)
                for key in packed.files
                if key != "__config__"
            }
        return cls(arrays, config)

    def initial_state(self, num_envs: int) -> State:
        """A zeroed recurrent state for ``num_envs`` environments."""
        return [block.mixer.empty_cache(num_envs) for block in self.blocks]

    def _encode(self, obs: np.ndarray) -> tuple[np.ndarray, tuple | None]:
        if self.spec is None:
            return self.encoder(obs), None
        batch = obs.shape[0]
        parts = [obs[:, : self.globals]] if self.globals else []
        pointed = None
        at = self.globals
        for i, (s, encoder) in enumerate(zip(self.sets, self.encoders)):
            count, features = int(s["count"]), int(s["features"])
            band = obs[:, at: at + count * (features + 1)].reshape(batch, count, features + 1)
            at += count * (features + 1)
            presence = band[:, :, features:]
            e = encoder(band[:, :, :features], presence)
            parts.extend(_pool(e, presence, kind) for kind in self.kinds)
            if i == self.pointer_set:
                pointed = (e, presence)
        return self.proj(np.concatenate(parts, axis=-1)), pointed

    def step(
        self,
        obs: np.ndarray,
        state: State,
        reset: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, State]:
        """Advance ``[num_envs, obs_dim]`` observations by one step.

        ``reset`` is an optional ``[num_envs]`` mask, ``1`` where the previous
        episode ended, exactly as ``Rollout.step`` takes it. Returns
        ``(logits [num_envs, action_dim], value [num_envs], new_state)``; the
        state passed in is not modified.
        """
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"obs must be [num_envs, {self.obs_dim}], got {obs.shape}")
        if len(state) != len(self.blocks):
            raise ValueError(f"state has {len(state)} layers, the policy {len(self.blocks)}")
        if reset is not None:
            reset = np.asarray(reset, dtype=np.float32).reshape(obs.shape[0])

        x, pointed = self._encode(obs)
        new_state = []
        for block, cache in zip(self.blocks, state):
            x, cache = block.step(x, cache, reset)
            new_state.append(cache)
        hidden = rms_norm(x, self.norm, self.eps)
        logits = self.actor(hidden) if pointed is None else self.actor(hidden, *pointed)
        value = self.critic(hidden)[:, 0]
        return logits.astype(np.float32), value.astype(np.float32), new_state
