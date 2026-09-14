"""PPO with every reliability feature on at once: a reference anchor, a legal-action
mask, a learning-rate schedule, and a checkpoint that resumes the optimizer.

The environment is a numpy recall task whose odd-numbered colours are *out of
stock*: they are never legal, and the environment says so through
``action_mask()``. The cue itself is always a legal colour, so masking removes
only actions that could never be right.

The run clones the environment's expert briefly, freezes that clone as the
reference, trains PPO anchored to it under a cosine schedule, saves half way,
rebuilds a learner from the checkpoint, and finishes there.

    python examples/ppo_anchored_masked.py                # a real run
    python examples/ppo_anchored_masked.py --smoke        # CI-sized, seconds
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

import mamba3_rl as m3

ACTIONS = 6
HORIZON = 4


class StockedRecall:
    """Name the colour shown at the start of the episode; odd colours are never legal."""

    def __init__(self, num_envs, seed=0):
        self.num_envs = num_envs
        self.action_dim = ACTIONS
        self.obs_dim = ACTIONS + 1
        self.rng = np.random.default_rng(seed)
        self.colour = np.zeros(num_envs, dtype=np.int64)
        self.clock = np.zeros(num_envs, dtype=np.int64)

    def _observation(self):
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        lit = self.clock == 0
        obs[lit, self.colour[lit]] = 1.0
        obs[:, -1] = self.clock / HORIZON
        return obs

    def _draw(self, count):
        return self.rng.integers(0, ACTIONS // 2, count) * 2  # even colours only

    def reset(self):
        self.colour = self._draw(self.num_envs)
        self.clock[:] = 0
        return self._observation()

    def step(self, actions):
        terminal = self.clock + 1 == HORIZON
        reward = np.where(terminal & (actions == self.colour), 1.0, 0.0).astype(np.float32)
        self.colour = np.where(terminal, self._draw(self.num_envs), self.colour)
        self.clock = np.where(terminal, 0, self.clock + 1)
        return self._observation(), reward, terminal.astype(np.float32)

    def expert_actions(self):
        return self.colour.copy()

    def action_mask(self):
        mask = np.zeros((self.num_envs, ACTIONS), dtype=bool)
        mask[:, 0::2] = True
        return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a few rounds, to check it runs")
    args = parser.parse_args()
    envs, clone_rounds, ppo_rounds = (8, 2, 4) if args.smoke else (64, 10, 120)
    steps, epochs = HORIZON * 3, 2

    policy = m3.Policy(m3.PolicyConfig(ACTIONS + 1, ACTIONS, d_model=32, n_layers=2, n_heads=2,
                                       head_dim=16, d_state=8, chunk_size=4, seed=3))
    print(f"backend: {m3.backend()}")

    cloner = m3.ImitationLearner(policy, StockedRecall(envs, seed=1), steps=steps)
    for stats in cloner.run(rounds=clone_rounds):
        print(f"clone {stats.round:>3}  agreement {stats.agreement:.3f}")

    scratch = tempfile.TemporaryDirectory()
    # The anchor: the clone, frozen. A learner snapshots its reference at
    # construction anyway, but a resumed learner needs the *same* weights handed
    # to it again, and `policy` itself is about to be trained -- so keep a copy.
    anchor_path = str(Path(scratch.name) / "clone.m3ck")
    policy.save(anchor_path)
    anchor = m3.Policy.load(anchor_path)

    ppo = m3.PpoConfig(reference_coeff=0.1)
    schedule = m3.LrSchedule.cosine(ppo_rounds * epochs, warmup_steps=2)
    settings = dict(steps=steps, ppo=ppo, learning_rate=1e-3, lr_schedule=schedule,
                    reference=anchor, seed=5)
    learner = m3.PpoLearner(policy, StockedRecall(envs, seed=2), **settings)

    def report(stats):
        print(f"ppo {stats.round:>4}  return {stats.episode_return}  lr {stats.learning_rate:.2e}  "
              f"reference_kl {stats.reference_kl:.4f}")

    half = ppo_rounds // 2
    for stats in learner.run(rounds=half, epochs=epochs):
        report(stats)

    path = str(Path(scratch.name) / "ppo.m3ck")
    learner.save(path)
    # A new process would do exactly this: rebuild from the checkpoint, handing
    # over what a checkpoint cannot carry -- the environment. The reference is
    # rebuilt from the weights the checkpoint carries; passing one instead checks
    # it against the fingerprint the checkpoint recorded.
    resumed = m3.PpoLearner.from_checkpoint(path, StockedRecall(envs, seed=2), steps=steps,
                                            seed=5)
    scratch.cleanup()
    print(f"resumed at round {resumed.rounds}, continuation {resumed.continuation}")
    assert resumed.rounds == half

    for stats in resumed.run(rounds=ppo_rounds - half, epochs=epochs):
        report(stats)

    # Evaluate on the masked distribution the policy was trained on.
    env = StockedRecall(envs, seed=9)
    rollout = m3.Rollout(resumed.policy, envs, temperature=0.0)
    obs, done, total = env.reset(), None, 0.0
    for _ in range(steps):
        actions, _, _ = rollout.step(obs, reset=done, action_mask=env.action_mask())
        assert (actions % 2 == 0).all(), "a masked rollout drew an out-of-stock colour"
        obs, reward, done = env.step(actions)
        total += float(reward.sum())
    print(f"greedy return per episode: {total / (envs * steps / HORIZON):.3f}")


if __name__ == "__main__":
    main()
