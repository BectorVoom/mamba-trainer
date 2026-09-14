"""Type stubs for the compiled extension module."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

__version__: str

def backend() -> str:
    """The backend this wheel was compiled against, e.g. ``"cpu"`` or ``"cuda"``."""

def matmul_precision() -> str:
    """The storage precision matrix products read their operands at."""

def set_matmul_precision(precision: str) -> None:
    """Set it to ``"f32"``, ``"bf16"`` or ``"f16"``. Master weights stay ``f32``."""

def read_count() -> int:
    """Host reads of a device buffer since the counter was last reset."""

def reset_read_count() -> None:
    """Reset the counter :func:`read_count` reports."""

def matmul_kernel() -> Literal["auto", "simple", "row_tiled", "tiled", "block_tiled", "cmma"]: ...
def set_matmul_kernel(kernel: str) -> None:
    """Pin the matrix-product kernel (or `"auto"`). On a GPU `"auto"` picks per
    shape by timing, per process, so pin the same kernel in every process whose
    runs must agree to the bit. `MAMBA3_MATMUL_KERNEL` sets it at import."""
def launch_count() -> int:
    """Kernels launched since `reset_launch_count()`: the dispatch count a
    fused rollout over a `game()` cuts."""
def reset_launch_count() -> None: ...
def synchronize() -> None:
    """Block until every queued kernel has completed. Only needed for timing."""

def evaluate(
    policy: Policy,
    env: Any,
    steps: int = 64,
    *,
    temperature: float = 0.0,
    seed: int = 0,
) -> Optional[float]:
    """Mean reward per episode completed within one window, from a fresh
    recurrent state, or ``None`` if none completed."""

class PolicyConfig:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        n_layers: int = 2,
        *,
        n_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        d_state: Optional[int] = None,
        n_groups: Optional[int] = None,
        chunk_size: Optional[int] = None,
        conv_kernel: Optional[int] = None,
        discretization: str = "learned_trapezoid",
        dynamics: str = "rotational",
        norm_eps: float = 1e-5,
        seed: int = 0,
    ) -> None: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def d_model(self) -> int: ...
    @property
    def n_layers(self) -> int: ...
    @property
    def n_heads(self) -> int: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def d_state(self) -> int: ...
    @property
    def n_groups(self) -> int: ...
    @property
    def chunk_size(self) -> int: ...
    @property
    def conv_kernel(self) -> Optional[int]: ...
    @property
    def discretization(self) -> str: ...
    @property
    def dynamics(self) -> str: ...
    @property
    def norm_eps(self) -> float: ...
    @property
    def seed(self) -> int: ...
    def to_dict(self) -> dict: ...
    @staticmethod
    def from_dict(mapping: dict) -> PolicyConfig: ...

class PpoConfig:
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coeff: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        clip_value_loss: bool = True,
        normalize_advantages: bool = True,
        reference_coeff: float = 0.0,
    ) -> None: ...
    @property
    def gamma(self) -> float: ...
    @property
    def gae_lambda(self) -> float: ...
    @property
    def clip_coeff(self) -> float: ...
    @property
    def value_coeff(self) -> float: ...
    @property
    def entropy_coeff(self) -> float: ...
    @property
    def clip_value_loss(self) -> bool: ...
    @property
    def normalize_advantages(self) -> bool: ...
    @property
    def reference_coeff(self) -> float:
        """Weight of the penalty on drifting away from `PpoLearner`'s `reference`."""

class LrSchedule:
    """A learning-rate schedule, evaluated once per optimizer step (not per
    round, epoch or minibatch). `PpoLearner`/`ImitationLearner`'s
    `lr_schedule=None` means `LrSchedule.constant()`."""

    @staticmethod
    def constant() -> LrSchedule: ...
    @staticmethod
    def cosine(
        total_steps: int, warmup_steps: Optional[int] = None, min_ratio: float = 0.1
    ) -> LrSchedule: ...
    @staticmethod
    def linear(
        total_steps: int, warmup_steps: Optional[int] = None, min_ratio: float = 0.0
    ) -> LrSchedule: ...
    @staticmethod
    def inverse_sqrt(warmup_steps: int) -> LrSchedule: ...
    @staticmethod
    def step(every: int, gamma: float) -> LrSchedule: ...
    def rate_at(self, base: float, step: int) -> float: ...

class EmaConfig:
    """A moving average of the policy's weights, for a learner's `ema=`. After
    every optimizer step `ema <- ema + (1 - d) * (theta - ema)`, with
    `d = decay`, or with `warmup="tf"` `d = min(decay, (1 + t) / (10 + t))` at
    the learner's optimizer step `t` (so an average reset late in a run is past
    its warm-up). The half-life is `ln 2 / -ln(decay)` optimizer steps: 69 at
    0.99, 693 at 0.999; a PPO round takes `epochs * minibatches` of them.
    `decay=0` tracks the weights, `decay=1` keeps the first ones. A decay that
    is not a finite number in [0, 1], or an unknown warm-up, raises
    `ValueError`; a decay that is not a number, `TypeError`."""

    def __init__(self, decay: float, warmup: Literal["none", "tf"] = "none") -> None: ...
    @property
    def decay(self) -> float: ...
    @property
    def warmup(self) -> Literal["none", "tf"]: ...
    def decay_at(self, step: int) -> float:
        """The decay applied after optimizer step `step` (one-based)."""

class Policy:
    def __init__(self, config: PolicyConfig) -> None: ...
    @property
    def config(self) -> PolicyConfig: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def num_parameters(self) -> int: ...
    @property
    def num_trainable_parameters(self) -> int: ...
    @property
    def backend(self) -> str: ...
    def describe(self) -> str: ...
    def train(self) -> None: ...
    def eval(self) -> None: ...
    def freeze(self, patterns: Sequence[str]) -> None: ...
    def unfreeze(self, patterns: Sequence[str]) -> None: ...
    def fingerprint(self) -> str:
        """Sixteen hex digits of FNV-1a over every parameter's path, shape and
        `f32` bits, in path order: equal exactly when every weight bit is, and
        equal to the fingerprint of the same weights in a checkpoint. One host
        read per parameter. Not a cryptographic digest."""
    def save(self, path: str, step: int = 0) -> None: ...
    @staticmethod
    def load(path: str) -> Policy: ...
    def load_weights(self, path: str, strict: bool = True) -> None: ...

class Rollout:
    def __init__(
        self,
        policy: Policy,
        num_envs: int,
        *,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def state_bytes(self) -> int: ...
    temperature: float
    def reset(self) -> None: ...
    def step(
        self,
        obs: np.ndarray,
        reset: Optional[np.ndarray] = None,
        temperature: Optional[float] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`action_mask` is `[num_envs, action_dim]`, 1/True where legal: the draw
        (sampled or greedy) never picks an illegal action and `log_probs` are the
        masked distribution's -- the same a learner's replay scores. A mask with
        a value other than 0/1 or a row with no legal action raises
        `ValueError` before the state advances."""
    def evaluate(
        self,
        obs: np.ndarray,
        reset: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """With `action_mask`, illegal actions' logits are
        `numpy.finfo(numpy.float32).min`: probability exactly 0 under softmax."""

class RecallEnv:
    def __init__(
        self, num_envs: int, symbols: int = 4, horizon: int = 8, seed: int = 0
    ) -> None: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def horizon(self) -> int: ...
    @property
    def chance_return(self) -> float: ...
    @property
    def optimal_return(self) -> float: ...
    def reset(self) -> np.ndarray: ...
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def expert_actions(self) -> Optional[np.ndarray]: ...

class Stats:
    round: int
    steps: int
    optimizer_steps: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    reference_kl: float
    grad_norm: float
    learning_rate: float
    episode_return: Optional[float]

class CloneStats:
    round: int
    steps: int
    optimizer_steps: int
    beta: float
    loss: float
    agreement: Optional[float]
    grad_norm: float
    learning_rate: float

class DaggerSchedule:
    @staticmethod
    def exponential(decay: float = 0.5) -> DaggerSchedule: ...
    @staticmethod
    def linear(rounds: int) -> DaggerSchedule: ...
    @staticmethod
    def only_first() -> DaggerSchedule: ...
    @staticmethod
    def fixed(beta: float) -> DaggerSchedule: ...
    def beta(self, round: int) -> float: ...

LoadSummary = Dict[str, Union[bool, str, List[str]]]
"""`{"weights": bool, "optimizer": bool, "counters": bool, "config":
"verified" | "adopted" | "live" | "legacy" | "warm_start", "level":
"full" | "optimizer" | "warm", "notes": [str]}`. `level` is what this load
restored."""

Continuation = Dict[str, Union[str, List[str]]]
"""`{"level": "full" | "optimizer" | "warm", "notes": [str]}`: how exactly the
learner's history continues one run. Only ever moves down that list."""

class Game:
    """A device game compiled into this extension; see `game()`. Learners given
    one collect through the fused rollout. Also an ordinary environment for
    inspection (`reset`, `step`, `action_mask`, a device read each)."""
    @property
    def name(self) -> str: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def masked(self) -> bool: ...
    def reset(self) -> np.ndarray: ...
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def action_mask(self) -> Optional[np.ndarray]: ...

def game(
    name: str,
    num_envs: int,
    *,
    symbols: int = 4,
    horizon: int = 8,
    seed: int = 0,
    masked: bool = False,
) -> Game:
    """A compiled-in device game by name. `"recall"` is the device twin of
    `RecallEnv`, with its horizon compiled in (8). Unknown names and parameters
    a game cannot honour raise `ValueError`."""

class PpoLearner:
    def __init__(
        self,
        policy: Policy,
        env: Any,
        steps: int = 128,
        *,
        ppo: Optional[PpoConfig] = None,
        learning_rate: float = 3e-4,
        lr_schedule: Optional[LrSchedule] = None,
        max_grad_norm: float = 0.5,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        temperature: float = 1.0,
        seed: int = 0,
        reference: Optional[Policy] = None,
        fused: Optional[bool] = None,
        ema: Optional[EmaConfig] = None,
    ) -> None:
        """`reference` freezes a policy to price the run against; see
        `PpoConfig.reference_coeff`. Without both, PPO is unchanged.

        `fused=None` collects a `game()` through the fused rollout and any other
        environment from the host; `fused=False` drives a game from the host too;
        `fused=True` with an environment that is not a game raises `ValueError`.

        `lr_schedule=None` means `LrSchedule.constant()`: `learning_rate` never
        changes. A schedule advances once per optimizer step, i.e. once per
        `update()` epoch times minibatch, not once per round.

        `ema=EmaConfig(decay)` keeps a moving average of the weights on the
        device, updated after every optimizer step (`ema_policy`). `None` keeps
        none, and attaching one changes nothing about training."""
    @property
    def policy(self) -> Policy: ...
    @property
    def env(self) -> Any: ...
    @property
    def config(self) -> PpoConfig: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def steps(self) -> int: ...
    @property
    def rounds(self) -> int: ...
    @property
    def buffer_bytes(self) -> int: ...
    @property
    def collection_path(self) -> Literal["fused", "host"]: ...
    @property
    def ema_policy(self) -> Optional[Policy]:
        """The moving average of the weights (`ema=`), as a policy; `None`
        without one. A handle onto the average, not a copy: it keeps moving as
        the learner trains, so `save` it (or note its `fingerprint`) to keep a
        moment. Usable with `evaluate`, `Rollout` and `save`; passing it to
        another learner as the policy to train is unsupported. Frozen
        parameters are not averaged: the average holds the trained policy's."""
    @property
    def ema_updates(self) -> int:
        """Optimizer steps the average has taken since it was built, reset or
        restored with its counter; 0 without one."""
    @property
    def ema_config(self) -> Optional[EmaConfig]: ...
    def reset_ema(self) -> None:
        """Restart the average from the current weights and its counter from
        zero (after a critic-only warm-up, say). A device copy, no host read.
        `ValueError` without an average, and between `collect()` and `update()`."""

    def window(self) -> Dict[str, np.ndarray]:
        """The window last collected, read back: `observations`, `actions`,
        `log_probs`, `values`, `rewards`, `dones` (and `action_mask`), shaped
        `[num_envs, steps, ...]`. A synchronisation."""
    def collect(self) -> int: ...
    def update(self, epochs: int = 4, minibatches: int = 1) -> Stats: ...
    def episode_return(self) -> Optional[float]: ...
    def round(self, epochs: int = 4, minibatches: int = 1) -> Stats: ...
    def run(
        self,
        rounds: int,
        epochs: int = 4,
        minibatches: int = 1,
        callback: Optional[Callable[[Stats], Any]] = None,
    ) -> List[Stats]: ...
    def reset(self) -> None: ...
    def save(self, path: str, level: Literal["optimizer", "full"] = "optimizer") -> None:
        """Weights, optimizer state, counters and the training configuration
        (base rate, `lr_schedule`, AdamW settings, `max_grad_norm`, `PpoConfig`,
        architecture, reference-weights fingerprint, `EmaConfig`), a reference's
        weights and architecture, and the moving average's weights and counter.
        Round-trip through `load_checkpoint` or `from_checkpoint`.

        `level="full"` adds the rollout: the collector's observation, flags and
        episode accounting, every layer's recurrent state, the action-draw
        schedule, the reference's carried cache, and the environment's own
        `save_state()` bytes — so a restore in any process continues the run
        exactly. Needs a `.m3ck` path and an environment implementing
        `save_state()`/`load_state()` (`NotImplementedError` otherwise), and
        raises `ValueError` between `collect()` and `update()`."""
    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        config: Literal["verify", "checkpoint", "live"] = "verify",
        level: Optional[Literal["optimizer", "full"]] = None,
    ) -> LoadSummary:
        """Restore what `save` wrote, all or nothing: a load that raises leaves
        the learner — and its environment — exactly as they were. `strict=False`
        with a weights-only file is a warm start (optimizer and counters
        restart). `config="verify"` raises on any configuration difference
        (including a full checkpoint's seed and temperature), `"checkpoint"`
        adopts the saved settings (and the saved reference policy, when the
        checkpoint carries its weights), `"live"` keeps these and marks the run
        `"warm"`. A full checkpoint also restores the rollout and calls the
        environment's `load_state()`; `level="optimizer"` ignores that state,
        `level="full"` requires it. Clears the last collected window. Raises
        `OSError` for an unreadable file and `ValueError` for anything wrong
        inside it.

        The moving average follows `config` too: `"verify"` raises on any
        difference in `ema.decay`/`ema.warmup` or in whether there is one,
        `"checkpoint"` adopts the saved average (configuration, weights,
        counter), and `"live"` keeps this learner's — re-seeding it from the
        loaded weights when the checkpoint has none (a `"warm"` load, noted),
        and ignoring, with a note, one this learner does not keep."""
    @staticmethod
    def from_checkpoint(
        path: str,
        env: Any,
        steps: int = 128,
        *,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        reference: Optional[Policy] = None,
        strict: bool = True,
    ) -> PpoLearner:
        """A learner built with the architecture and settings `path` recorded,
        then loaded from it with `config="verify"`. `reference` defaults to the
        reference policy the checkpoint carries; one passed in must have the
        same weights. `temperature` and `seed` default to those a full
        checkpoint recorded, else `1.0` and `0`. A moving average is rebuilt
        from the checkpoint."""
    @property
    def continuation(self) -> Continuation:
        """How exactly this learner's history continues one run."""

class ImitationLearner:
    def __init__(
        self,
        policy: Policy,
        env: Any,
        steps: int = 128,
        *,
        schedule: Optional[DaggerSchedule] = None,
        entropy_bonus: float = 0.01,
        learning_rate: float = 3e-3,
        lr_schedule: Optional[LrSchedule] = None,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        temperature: float = 1.0,
        seed: int = 0,
        ema: Optional[EmaConfig] = None,
    ) -> None:
        """`schedule` is DAgger's own expert-mixing schedule; `lr_schedule` is
        the optimizer's learning-rate schedule, `None` meaning
        `LrSchedule.constant()`. Distinct knobs, distinct clocks. `ema` as for
        `PpoLearner`: one optimizer step, and one average update, per round."""
    @property
    def policy(self) -> Policy: ...
    @property
    def env(self) -> Any: ...
    @property
    def schedule(self) -> DaggerSchedule: ...
    @property
    def rounds(self) -> int: ...
    @property
    def ema_policy(self) -> Optional[Policy]:
        """The moving average of the weights (`ema=`), as a policy; `None`
        without one. A handle onto the average, not a copy: it keeps moving as
        the learner trains, so `save` it (or note its `fingerprint`) to keep a
        moment. Usable with `evaluate`, `Rollout` and `save`; passing it to
        another learner as the policy to train is unsupported. Frozen
        parameters are not averaged: the average holds the trained policy's."""
    @property
    def ema_updates(self) -> int:
        """Optimizer steps the average has taken since it was built, reset or
        restored with its counter; 0 without one."""
    @property
    def ema_config(self) -> Optional[EmaConfig]: ...
    def reset_ema(self) -> None:
        """Restart the average from the current weights and its counter from
        zero (after a critic-only warm-up, say). A device copy, no host read.
        `ValueError` without an average."""
    def round(
        self, beta: Optional[float] = None, agreement: bool = True
    ) -> CloneStats: ...
    def run(
        self,
        rounds: int,
        agreement: bool = True,
        callback: Optional[Callable[[CloneStats], Any]] = None,
    ) -> List[CloneStats]: ...
    def reset(self) -> None: ...
    def save(self, path: str, level: Literal["optimizer", "full"] = "optimizer") -> None:
        """See `PpoLearner.save`; the algorithm settings are the DAgger
        `schedule` and `entropy_bonus`, and a full checkpoint carries the
        expert-mixing draws with the policy's."""
    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        config: Literal["verify", "checkpoint", "live"] = "verify",
        level: Optional[Literal["optimizer", "full"]] = None,
    ) -> LoadSummary:
        """See `PpoLearner.load_checkpoint`."""
    @staticmethod
    def from_checkpoint(
        path: str,
        env: Any,
        steps: int = 128,
        *,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        strict: bool = True,
    ) -> ImitationLearner:
        """See `PpoLearner.from_checkpoint`."""
    @property
    def continuation(self) -> Continuation:
        """See `PpoLearner.continuation`."""
