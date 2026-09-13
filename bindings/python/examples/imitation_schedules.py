"""Imitation with both of its schedules, which are easy to confuse.

``schedule=DaggerSchedule...`` decides how much of the *acting* the expert does,
once per round. ``lr_schedule=LrSchedule...`` decides the optimizer's learning rate,
once per optimizer step. They are separate clocks and this run prints both.

    python examples/imitation_schedules.py            # a real run
    python examples/imitation_schedules.py --smoke    # CI-sized, seconds
"""

import argparse
import tempfile
from pathlib import Path

import mamba3_rl as m3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a few rounds, to check it runs")
    args = parser.parse_args()
    envs, rounds = (8, 4) if args.smoke else (32, 40)

    env = m3.RecallEnv(num_envs=envs, symbols=4, horizon=4, seed=1)
    policy = m3.Policy(m3.PolicyConfig(env.obs_dim, env.action_dim, d_model=32, n_layers=2,
                                       n_heads=2, head_dim=16, d_state=8, chunk_size=4, seed=7))
    cloner = m3.ImitationLearner(
        policy,
        env,
        steps=16,
        schedule=m3.DaggerSchedule.exponential(0.7),
        learning_rate=3e-3,
        lr_schedule=m3.LrSchedule.linear(rounds, warmup_steps=1, min_ratio=0.1),
    )
    for stats in cloner.run(rounds=rounds):
        print(f"round {stats.round:>3}  beta {stats.beta:.3f} (DAgger)  "
              f"lr {stats.learning_rate:.2e} (optimizer)  agreement {stats.agreement:.3f}")

    with tempfile.TemporaryDirectory() as scratch:
        path = str(Path(scratch) / "clone.m3ck")
        cloner.save(path)
        # Both schedules are part of the saved configuration; a learner built with
        # a different one is refused unless told how to treat the difference.
        other = m3.ImitationLearner(m3.Policy(policy.config), env, steps=16)
        try:
            other.load_checkpoint(path)
        except ValueError as refusal:
            print("refused, as it should be:", str(refusal).splitlines()[-1].strip())
        summary = other.load_checkpoint(path, config="checkpoint")
        print(f"adopted: {summary['notes']}")
        print(f"next beta {other.schedule.beta(other.rounds):.3f}, rounds {other.rounds}")


if __name__ == "__main__":
    main()
