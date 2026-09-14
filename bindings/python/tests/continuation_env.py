"""A stateful environment and learner builders for the exact-continuation tests.

Kept out of the test module so the restoring half of a test can run in a separate
Python process and build exactly the same objects.
"""

import json

import numpy as np

import mamba3_rl as m3

NUM_ENVS = 4
ACTIONS = 4


class StatefulLaneEnv:
    """Everything that makes an exact resume hard, in a few lines.

    * Asynchronous resets: each lane's episode length is drawn from the
      environment's own random number generator when the episode starts.
    * Stochastic rewards from the same generator, so its position matters.
    * A legal-action mask that changes with the clock, and an expert that only
      names legal actions.
    * A log of every action taken, which is part of the state and is how a test
      compares the actions two runs sampled.
    * Episode-return accounting — each lane's running return, and the total
      and count of completed episodes — so a test can compare those directly.

    `save_state`/`load_state` implement the optional protocol: opaque bytes,
    validated completely before anything changes.
    """

    num_envs = NUM_ENVS
    obs_dim = 6
    action_dim = ACTIONS

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.clock = np.zeros(self.num_envs, dtype=np.int64)
        self.horizon = np.full(self.num_envs, 3, dtype=np.int64)
        self.episode = np.zeros(self.num_envs, dtype=np.int64)
        self.log = []
        self.running_return = np.zeros(self.num_envs, dtype=np.float64)
        self.completed_return = 0.0
        self.completed = 0

    def _observation(self):
        lanes = np.arange(self.num_envs)[:, None]
        k = np.arange(self.obs_dim)[None, :]
        phase = lanes * 7 + self.clock[:, None] * 3 + self.episode[:, None] * 5 + k * 11
        return np.sin(phase * 0.61).astype(np.float32)

    def reset(self):
        self.clock[:] = 0
        self.episode[:] = 0
        self.horizon = self.rng.integers(2, 6, size=self.num_envs)
        return self._observation()

    def action_mask(self):
        mask = np.ones((self.num_envs, self.action_dim), dtype=np.float32)
        starting = self.clock == 0
        mask[starting, np.arange(self.num_envs)[starting] % self.action_dim] = 0.0
        return mask

    def expert_actions(self):
        return (np.arange(self.num_envs) + self.clock + 1) % self.action_dim

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        self.log.extend(int(a) for a in actions)
        reward = (actions == self.expert_actions()).astype(np.float32)
        reward += self.rng.normal(0.0, 0.1, size=self.num_envs).astype(np.float32)
        self.clock += 1
        done = self.clock >= self.horizon
        self.running_return += reward
        self.completed_return += float(self.running_return[done].sum())
        self.completed += int(done.sum())
        self.running_return[done] = 0.0
        self.episode[done] += 1
        self.clock[done] = 0
        self.horizon[done] = self.rng.integers(2, 6, size=int(done.sum()))
        return self._observation(), reward, done.astype(np.float32)

    def save_state(self):
        return json.dumps({
            "layout": "StatefulLaneEnv/2",
            "rng": self.rng.bit_generator.state,
            "clock": self.clock.tolist(),
            "horizon": self.horizon.tolist(),
            "episode": self.episode.tolist(),
            "log": self.log,
            "running_return": self.running_return.tolist(),
            "completed_return": self.completed_return,
            "completed": self.completed,
        }).encode()

    def load_state(self, data):
        state = json.loads(bytes(data).decode())
        if state.get("layout") != "StatefulLaneEnv/2":
            raise ValueError(f"not a StatefulLaneEnv state: {state.get('layout')!r}")
        for key in ("clock", "horizon", "episode", "running_return"):
            if len(state[key]) != self.num_envs:
                raise ValueError(f"saved {key} has {len(state[key])} lanes, not {self.num_envs}")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        # Validated; nothing below can fail.
        self.rng = rng
        self.clock = np.array(state["clock"], dtype=np.int64)
        self.horizon = np.array(state["horizon"], dtype=np.int64)
        self.episode = np.array(state["episode"], dtype=np.int64)
        self.log = list(state["log"])
        self.running_return = np.array(state["running_return"], dtype=np.float64)
        self.completed_return = float(state["completed_return"])
        self.completed = int(state["completed"])


def small_policy(seed=7):
    return m3.Policy(
        m3.PolicyConfig(StatefulLaneEnv.obs_dim, ACTIONS, d_model=16, n_layers=1, n_heads=1,
                        head_dim=16, d_state=4, chunk_size=2, seed=seed)
    )


def ppo(env=None, policy=None, **overrides):
    settings = dict(
        steps=5,
        learning_rate=0.004,
        lr_schedule=m3.LrSchedule.cosine(40, warmup_steps=2),
        ppo=m3.PpoConfig(reference_coeff=0.3),
        reference=small_policy(11),
        temperature=1.0,
        seed=5,
    )
    settings.update(overrides)
    return m3.PpoLearner(policy or small_policy(), env or StatefulLaneEnv(), **settings)


def imitation(env=None, policy=None, **overrides):
    settings = dict(
        steps=5,
        learning_rate=0.004,
        lr_schedule=m3.LrSchedule.cosine(40, warmup_steps=2),
        schedule=m3.DaggerSchedule.exponential(0.7),
        temperature=1.0,
        seed=5,
    )
    settings.update(overrides)
    return m3.ImitationLearner(policy or small_policy(), env or StatefulLaneEnv(), **settings)


def with_ema(build):
    """`build`, with a moving average of the weights (T1)."""

    def build_with_ema(env=None, policy=None, **overrides):
        return build(env=env, policy=policy, ema=m3.EmaConfig(0.9, warmup="tf"), **overrides)

    return build_with_ema


BUILDERS = {"ppo": ppo, "imitation": imitation,
            "ppo_ema": with_ema(ppo), "imitation_ema": with_ema(imitation)}


def episodes(env):
    """What the environment has seen of its episodes, and the observation the
    learner's next window starts from."""
    return {"next_observation": env._observation().tolist(),
            "episode_return_total": env.completed_return, "episodes_completed": env.completed,
            "running_return": env.running_return.tolist()}


def averaged(learner):
    """The moving average's fingerprint and counter, when the learner keeps one."""
    if learner.ema_policy is None:
        return {}
    return {"ema_fingerprint": learner.ema_policy.fingerprint(), "ema_updates": learner.ema_updates}


def one_round(learner):
    """One round, reduced to plain data."""
    if isinstance(learner, m3.PpoLearner):
        s = learner.round(epochs=2)
        return {**averaged(learner), "round": s.round, "optimizer_steps": s.optimizer_steps,
                "learning_rate": s.learning_rate, "loss": s.loss, "entropy": s.entropy,
                "approx_kl": s.approx_kl, "reference_kl": s.reference_kl,
                "episode_return": s.episode_return, "grad_norm": s.grad_norm,
                "observations": learner.window()["observations"].tolist(),
                **episodes(learner.env)}
    s = learner.round(agreement=True)
    return {**averaged(learner), "round": s.round, "optimizer_steps": s.optimizer_steps,
            "learning_rate": s.learning_rate, "beta": s.beta, "loss": s.loss,
            "agreement": s.agreement, "grad_norm": s.grad_norm, **episodes(learner.env)}


def weights(learner, path):
    """The policy's weights as `{name: [values]}`, through the JSON writer."""
    learner.policy.save(str(path))
    return {name: entry["data"] for name, entry in json.loads(path.read_text())["state"]["entries"].items()}
