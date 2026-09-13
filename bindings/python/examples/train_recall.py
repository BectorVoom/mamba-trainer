"""Reinforcement learning and imitation learning, on a task that needs memory.

The task is ``RecallEnv``: a symbol is shown once, at the first step of an episode,
and the only reward comes from naming it at the last. A policy with no memory
cannot do better than guessing -- ``1 / symbols`` -- so every point above that line
is the recurrent state carrying the cue across the gap, and nothing else.

The two families run side by side on identical architectures, which is the
comparison worth seeing:

1. **PPO from scratch.** Nothing but the reward signal. It has to discover that the
   cue matters before it can learn to carry it, and the reward is sparse, so this
   is the slow road.
2. **Imitation, then PPO.** The environment knows its own optimal action, so it is
   an expert that can label any state. Cloning it is cross entropy and it arrives
   in a handful of updates -- but it never mentions reward and can never exceed the
   expert, so PPO takes over from where it lands.

The same policy, the same replay and the same episode mask serve both; the handover
is a change of learner and nothing else.

    python examples/train_recall.py --rounds 60
"""

import argparse

import mamba3_rl as m3

ENVS = 32
SYMBOLS = 4
HORIZON = 4
# Steps per window: four whole episodes per environment, so every window holds
# several episode boundaries and the reset masking is exercised continuously.
WINDOW = HORIZON * 4


def build_policy(env, seed=7):
    return m3.Policy(
        m3.PolicyConfig(
            env.obs_dim,
            env.action_dim,
            d_model=64,
            n_layers=2,
            n_heads=4,
            head_dim=16,
            d_state=8,
            chunk_size=8,
            conv_kernel=4,
            seed=seed,
        )
    )


def clone_expert(policy, rounds):
    """Clone the environment's own expert into `policy`, DAgger-style.

    `beta` is the expert's share of the *acting* and it decays, so the states being
    labelled drift from the expert's own distribution towards the ones the learner
    actually reaches. Round 0, where `beta` is 1, is plain behaviour cloning.
    """
    env = m3.RecallEnv(num_envs=ENVS, symbols=SYMBOLS, horizon=HORIZON, seed=11)
    cloner = m3.ImitationLearner(
        policy,
        env,
        steps=WINDOW,
        learning_rate=3e-3,
        max_grad_norm=1.0,
        entropy_bonus=0.01,
        schedule=m3.DaggerSchedule.exponential(0.7),
        seed=3,
    )

    print(f"{'round':>6}  {'beta':>5}  {'loss':>8}  {'agreement':>9}")
    for stats in cloner.run(rounds=rounds):
        if stats.round % 3 == 0 or stats.round + 1 == rounds:
            print(
                f"{stats.round:>6}  {stats.beta:>5.2f}  "
                f"{stats.loss:>8.4f}  {stats.agreement:>9.3f}"
            )


def optimise_return(policy, rounds):
    """Optimise the return itself, reporting it every few rounds."""
    env = m3.RecallEnv(num_envs=ENVS, symbols=SYMBOLS, horizon=HORIZON, seed=23)
    learner = m3.PpoLearner(
        policy,
        env,
        steps=WINDOW,
        ppo=m3.PpoConfig(gamma=0.99, gae_lambda=0.95, clip_coeff=0.2,
                         value_coeff=0.5, entropy_coeff=0.01),
        learning_rate=1e-3,
        max_grad_norm=0.5,
        seed=5,
    )

    print(f"{'round':>6}  {'return':>8}  {'entropy':>8}  {'kl':>8}  {'clipped':>8}")

    def report(stats):
        if stats.round % 10 == 0 or stats.round + 1 == rounds:
            print(
                f"{stats.round:>6}  {stats.episode_return:>8.3f}  {stats.entropy:>8.3f}  "
                f"{stats.approx_kl:>8.4f}  {stats.clip_fraction:>8.2f}"
            )

    # Four epochs over each window. This is what the importance ratio is for:
    # without it, reusing the data would optimise against a policy that has already
    # moved away from the one that collected it.
    history = learner.run(rounds=rounds, epochs=4, callback=report)
    return [stats.episode_return for stats in history]


def greedy_return(policy):
    """What the policy *believes*, rather than what its exploration produces."""
    env = m3.RecallEnv(num_envs=ENVS, symbols=SYMBOLS, horizon=HORIZON, seed=99)
    return m3.evaluate(policy, env, steps=WINDOW, temperature=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=60, help="PPO rounds from scratch")
    parser.add_argument("--clone-rounds", type=int, default=12)
    args = parser.parse_args()

    probe = m3.RecallEnv(num_envs=ENVS, symbols=SYMBOLS, horizon=HORIZON)
    print(f"backend: {m3.backend()}")
    print(
        f"{probe!r}: guessing earns {probe.chance_return:.2f} per episode, "
        f"a perfect memory {probe.optimal_return:.2f}\n"
    )

    print("== PPO from scratch ==")
    from_scratch = build_policy(probe)
    scratch_history = optimise_return(from_scratch, args.rounds)
    scratch_greedy = greedy_return(from_scratch)

    print("\n== imitation first, then PPO ==")
    cloned = build_policy(probe)
    clone_expert(cloned, args.clone_rounds)
    print(f"greedy return after cloning alone: {greedy_return(cloned):.3f}\n")
    cloned_history = optimise_return(cloned, args.rounds // 2)
    cloned_greedy = greedy_return(cloned)

    # How many rounds of *reward* each run needed to get most of the way there. The
    # gap is what an expert is worth when you happen to have one.
    def reached(history, target=0.9):
        for index, value in enumerate(history):
            if value >= target:
                return str(index)
        return "never"

    print("\n== summary ==")
    print(f"{'run':<24}  {'greedy':>8}  {'rounds to 0.9':>16}")
    print(f"{'PPO from scratch':<24}  {scratch_greedy:>8.3f}  {reached(scratch_history):>16}")
    print(f"{'cloned, then PPO':<24}  {cloned_greedy:>8.3f}  {reached(cloned_history):>16}")
    print(
        f"\nguessing would earn {probe.chance_return:.3f}; "
        f"a perfect memory {probe.optimal_return:.3f}"
    )


if __name__ == "__main__":
    main()
