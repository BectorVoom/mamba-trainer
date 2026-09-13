"""What an environment has to look like.

The learners drive a *vectorised* environment: ``num_envs`` copies of a task
advancing together, one action each per step. That is not a convenience -- it is
what keeps the device busy, because one step of 512 environments is one batch of
kernel launches and 512 steps of one environment is 512 of them.

Two conventions are worth reading twice, because getting either wrong produces a
run that trains without ever complaining:

**Auto-reset.** Where ``done[i]`` is ``1``, the observation returned beside it is
already the first observation of environment ``i``'s *next* episode. Episodes
therefore straddle window boundaries and a rollout stays rectangular however the
episodes fall.

**``done`` marks the transition that ended an episode**, not the observation that
begins one. The library derives the second from the first -- shifting it by one
step and carrying the last flag of each window into the next -- and uses it to cut
both the recurrence and the short causal convolution, so nothing before a reset is
visible after it.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

__all__ = ["VecEnv"]


@runtime_checkable
class VecEnv(Protocol):
    """A batch of environments advancing together.

    Only the attributes and methods below are used, and duck typing is enough:
    an object does not have to inherit from this to be accepted. Arrays may be
    any dtype numpy can cast to ``float32`` (or ``int64`` for actions), and
    ``num_envs`` may also be spelled ``envs``.
    """

    #: How many environments run in parallel.
    num_envs: int
    #: Width of one observation vector.
    obs_dim: int
    #: Number of discrete actions.
    action_dim: int

    def reset(self) -> np.ndarray:
        """Start every environment.

        Returns the first ``[num_envs, obs_dim]`` observation. Called once, by
        the first collection a learner makes, and again after
        :meth:`PpoLearner.reset`.
        """
        ...

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply one action per environment.

        ``actions`` is ``[num_envs]`` of ``int64``. Returns
        ``(observation, reward, done)`` shaped ``[num_envs, obs_dim]``,
        ``[num_envs]`` and ``[num_envs]``, where ``done`` is ``1`` for a
        transition that ended an episode and the observation beside it is
        already the next episode's first.
        """
        ...

    def expert_actions(self) -> Optional[np.ndarray]:
        """What an expert would do on the observation most recently returned.

        Optional, and only :class:`ImitationLearner` needs it: an environment
        that knows its own optimal action can label any state the learner
        wanders into, which is exactly what DAgger asks for and what a recorded
        demonstration set cannot provide. Returns ``[num_envs]`` action ids, or
        ``None`` where it cannot say.
        """
        ...
