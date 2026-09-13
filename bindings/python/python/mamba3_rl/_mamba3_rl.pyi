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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def evaluate(
        self, obs: np.ndarray, reset: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

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
"verified" | "adopted" | "live" | "legacy" | "warm_start", "exact": bool,
"notes": [str]}`."""

Continuation = Dict[str, Union[bool, List[str]]]
"""`{"exact": bool, "notes": [str]}`."""

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
    ) -> None:
        """`reference` freezes a policy to price the run against; see
        `PpoConfig.reference_coeff`. Without both, PPO is unchanged.

        `lr_schedule=None` means `LrSchedule.constant()`: `learning_rate` never
        changes. A schedule advances once per optimizer step, i.e. once per
        `update()` epoch times minibatch, not once per round."""
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
    def save(self, path: str) -> None:
        """Weights, optimizer state, counters and the training configuration
        (base rate, `lr_schedule`, AdamW settings, `max_grad_norm`, `PpoConfig`,
        architecture, reference-weights fingerprint). Round-trip through
        `load_checkpoint` or `from_checkpoint`. Does not cover the environment's
        own state, the collector's or reference's recurrent state, or the
        sampling RNG (A2b)."""
    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        config: Literal["verify", "checkpoint", "live"] = "verify",
    ) -> LoadSummary:
        """Restore what `save` wrote, all or nothing: a load that raises leaves
        the learner exactly as it was. `strict=False` with a weights-only file
        is a warm start (optimizer and counters restart). `config="verify"`
        raises on any configuration difference, `"checkpoint"` adopts the
        saved optimizer/schedule/PPO settings, `"live"` keeps these and marks
        the run non-exact. Clears the last collected window. Raises `OSError`
        for an unreadable file and `ValueError` for anything wrong inside it."""
    @staticmethod
    def from_checkpoint(
        path: str,
        env: Any,
        steps: int = 128,
        *,
        temperature: float = 1.0,
        seed: int = 0,
        reference: Optional[Policy] = None,
        strict: bool = True,
    ) -> PpoLearner:
        """A learner built with the architecture and settings `path` recorded,
        then loaded from it with `config="verify"`."""
    @property
    def continuation(self) -> Continuation:
        """Whether this learner's history is one run under one configuration."""

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
    ) -> None:
        """`schedule` is DAgger's own expert-mixing schedule; `lr_schedule` is
        the optimizer's learning-rate schedule, `None` meaning
        `LrSchedule.constant()`. Distinct knobs, distinct clocks."""
    @property
    def policy(self) -> Policy: ...
    @property
    def env(self) -> Any: ...
    @property
    def schedule(self) -> DaggerSchedule: ...
    @property
    def rounds(self) -> int: ...
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
    def save(self, path: str) -> None:
        """See `PpoLearner.save`; the algorithm settings are the DAgger
        `schedule` and `entropy_bonus`."""
    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        config: Literal["verify", "checkpoint", "live"] = "verify",
    ) -> LoadSummary:
        """See `PpoLearner.load_checkpoint`."""
    @staticmethod
    def from_checkpoint(
        path: str,
        env: Any,
        steps: int = 128,
        *,
        temperature: float = 1.0,
        seed: int = 0,
        strict: bool = True,
    ) -> ImitationLearner:
        """See `PpoLearner.from_checkpoint`."""
    @property
    def continuation(self) -> Continuation:
        """See `PpoLearner.continuation`."""
