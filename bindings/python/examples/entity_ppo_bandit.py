"""PPO on a contextual bandit with the entity policy.

Each sample hides one "target" entity marked by feature 0 = 1; the reward is
the share of queries whose step-0 pointer names it. One-step episodes, so
``done = 1`` and ``last_value = 0``. Deterministic (fixed seeds); the mean
reward climbs from chance (1/6) past 0.8 within a few updates on CPU.
"""

import numpy as np

import mamba3_rl as m3

N = 6
BATCH = 64
QUERIES = 2


def main() -> None:
    spec = m3.EntityModelSpec(
        globals=0,
        context=[m3.ContextSet("field", count=N, features=2)],
        queries=m3.QuerySet("q", count=QUERIES, features=1, steps=1),
        heads=[m3.Head.pointer("pick", set="field")],
        d_model=16,
        context_layers=1,
        decoder_layers=1,
        seed=7,
    )
    policy = m3.EntityPolicy(spec, learning_rate=3e-3)
    rng = np.random.default_rng(1234)
    for update in range(60):
        targets = rng.integers(0, N, BATCH)
        ctx = np.zeros((BATCH, N, 2), dtype=np.float32)
        for bi, target in enumerate(targets):
            ctx[bi, target, 0] = 1.0
        ctx[:, :, 1] = rng.normal(size=(BATCH, N)).astype(np.float32)
        obs = {"field": ctx, "q": np.zeros((BATCH, QUERIES, 1), dtype=np.float32)}
        out = policy.act(obs)
        pick = out["actions"]["pick"].reshape(BATCH, QUERIES)
        rewards = (pick == targets[:, None]).mean(axis=1).astype(np.float32)
        print(f"update {update:2d}  mean reward {rewards.mean():.3f}")
        if rewards.mean() > 0.8:
            break
        policy.update(
            obs,
            out["actions"],
            out["log_prob"],
            out["value"],
            rewards.reshape(1, BATCH),
            np.ones((1, BATCH), dtype=np.float32),
            np.zeros(BATCH, dtype=np.float32),
            epochs=4,
            minibatches=2,
        )


if __name__ == "__main__":
    main()
