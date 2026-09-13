//! The learning half of reinforcement learning.
//!
//! `tests/rl.rs` holds the two *engines* to each other — that `T` rollout steps and
//! one scan over the same window compute the same numbers. This file holds the
//! things built on top of that agreement to their definitions: the advantage
//! estimator against a host recurrence written from the paper, the sampler against
//! the distribution it claims to draw from, PPO against the vanilla policy gradient
//! it degenerates to, and both algorithms against the only test that finally
//! matters — whether the policy gets better at something it has to remember to do.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::rl as kernels;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn tensor(data: &[f32], dims: Vec<usize>) -> Tensor<R, f32> {
    Tensor::from_f32(data, dims, &dev()).expect("data fills the shape")
}

/// Deterministic pseudo-random values in `[-1, 1)`.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let err = (a - e).abs() / (1.0 + e.abs());
        assert!(
            err <= eps,
            "{what}: index {i} is {a}, expected {e} (relative error {err})"
        );
    }
}

// ---------------------------------------------------------------------------
// Generalized advantage estimation
// ---------------------------------------------------------------------------

/// A `[envs, steps]` rollout, on the host, as the advantage estimator sees it.
struct Rollout {
    rewards: Vec<f32>,
    values: Vec<f32>,
    dones: Vec<f32>,
    bootstrap: Vec<f32>,
    envs: usize,
    steps: usize,
}

impl Rollout {
    fn at(&self, env: usize, step: usize) -> usize {
        env * self.steps + step
    }

    /// GAE written the way the paper states it, straight down the recurrence.
    ///
    /// Deliberately *not* the kernel's structure: this is the definition the kernel
    /// is supposed to implement, not a transcription of how it implements it.
    fn advantages(&self, gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
        let mut advantages = vec![0.0; self.envs * self.steps];
        let mut returns = vec![0.0; self.envs * self.steps];
        for (e, bootstrap) in self.bootstrap.iter().enumerate() {
            let mut carry = 0.0;
            for t in (0..self.steps).rev() {
                let i = self.at(e, t);
                let next_value = if t + 1 == self.steps {
                    *bootstrap
                } else {
                    self.values[i + 1]
                };
                let alive = 1.0 - self.dones[i];
                let delta = self.rewards[i] + gamma * next_value * alive - self.values[i];
                carry = delta + gamma * lambda * alive * carry;
                advantages[i] = carry;
                returns[i] = carry + self.values[i];
            }
        }
        (advantages, returns)
    }
}

#[test]
fn advantages_match_the_recurrence_they_are_defined_by() {
    let (envs, steps) = (4usize, 7usize);
    let n = envs * steps;
    let mut rollout = Rollout {
        rewards: noise(n, 1),
        values: noise(n, 2),
        dones: vec![0.0; n],
        bootstrap: noise(4, 3),
        envs,
        steps,
    };
    // Terminations placed to cover every case that changes the recurrence: one
    // inside the window, one exactly on the last step (where the bootstrap must be
    // ignored), one on the first, and an environment that never ends.
    for (env, step) in [(0, 3), (1, steps - 1), (2, 0)] {
        let at = rollout.at(env, step);
        rollout.dones[at] = 1.0;
    }

    for (gamma, lambda) in [(0.99, 0.95), (1.0, 1.0), (0.0, 0.0), (0.9, 0.0)] {
        let out = kernels::generalized_advantage(
            &tensor(&rollout.rewards, vec![envs, steps]),
            &tensor(&rollout.values, vec![envs, steps]),
            &tensor(&rollout.dones, vec![envs, steps]),
            &tensor(&rollout.bootstrap, vec![envs]),
            gamma,
            lambda,
        )
        .unwrap();

        let (want_adv, want_ret) = rollout.advantages(gamma, lambda);
        let what = format!("gae(gamma={gamma}, lambda={lambda})");
        assert_close(&out.advantages.to_f32(), &want_adv, 1e-5, &what);
        assert_close(&out.returns.to_f32(), &want_ret, 1e-5, &what);
    }
}

#[test]
fn a_termination_stops_credit_from_crossing_it() {
    // One environment, a reward of 1 on the last step only, no discounting and a
    // full trace: without a boundary every earlier step shares the credit, and with
    // one, nothing before it does.
    let steps = 5;
    let rewards = vec![0.0, 0.0, 0.0, 0.0, 1.0];
    let values = vec![0.0; steps];
    let bootstrap = vec![0.0];

    let open = kernels::generalized_advantage(
        &tensor(&rewards, vec![1, steps]),
        &tensor(&values, vec![1, steps]),
        &tensor(&vec![0.0; steps], vec![1, steps]),
        &tensor(&bootstrap, vec![1]),
        1.0,
        1.0,
    )
    .unwrap();
    assert_close(
        &open.advantages.to_f32(),
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        1e-6,
        "an undiscounted trace carries the final reward to every step",
    );

    let mut dones = vec![0.0f32; steps];
    dones[2] = 1.0;
    let cut = kernels::generalized_advantage(
        &tensor(&rewards, vec![1, steps]),
        &tensor(&values, vec![1, steps]),
        &tensor(&dones, vec![1, steps]),
        &tensor(&bootstrap, vec![1]),
        1.0,
        1.0,
    )
    .unwrap();
    assert_close(
        &cut.advantages.to_f32(),
        &[0.0, 0.0, 0.0, 1.0, 1.0],
        1e-6,
        "a termination at step 2 keeps the reward at step 4 out of steps 0..=2",
    );
}

#[test]
fn the_bootstrap_carries_the_value_past_the_window() {
    // No rewards, no terminations, γ = 1, λ = 1: every advantage is the bootstrap
    // minus the step's own value, because the only return in sight is what the
    // critic says lies beyond the window.
    let steps = 3;
    let values = vec![0.5, -0.25, 2.0];
    let out = kernels::generalized_advantage(
        &tensor(&vec![0.0; steps], vec![1, steps]),
        &tensor(&values, vec![1, steps]),
        &tensor(&vec![0.0; steps], vec![1, steps]),
        &tensor(&[3.0], vec![1]),
        1.0,
        1.0,
    )
    .unwrap();
    assert_close(
        &out.advantages.to_f32(),
        &[3.0 - 0.5, 3.0 - (-0.25), 3.0 - 2.0],
        1e-6,
        "bootstrapped advantage",
    );
    assert_close(
        &out.returns.to_f32(),
        &[3.0, 3.0, 3.0],
        1e-6,
        "every λ-return is the bootstrapped value",
    );
}

#[test]
fn advantage_normalisation_survives_a_constant_rollout() {
    // Every advantage identical: the variance is zero, and the answer must be zeros
    // rather than the infinities `1/sqrt(var)` would produce.
    let flat = kernels::normalize(&tensor(&[2.5; 12], vec![3, 4]), 1e-8).unwrap();
    assert_close(&flat.to_f32(), &[0.0; 12], 1e-5, "a constant rollout");

    let values = noise(24, 9);
    let out = kernels::normalize(&tensor(&values, vec![4, 6]), 1e-8)
        .unwrap()
        .to_f32();
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    let variance: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-5, "normalised advantages have mean {mean}");
    assert!(
        (variance - 1.0).abs() < 1e-4,
        "normalised advantages have variance {variance}"
    );
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

#[test]
fn greedy_sampling_takes_the_argmax() {
    let logits = tensor(&[0.1, 0.9, -2.0, 0.3, -1.0, -3.0, 4.0, 0.0], vec![2, 4]);
    let (actions, logprobs) = kernels::sample_categorical(&logits, 0.0, 0).unwrap();
    assert_eq!(actions.to_vec(), vec![1, 2], "greedy actions");

    // And the log-probability that comes back is the log-softmax of the row.
    let rows = [[0.1f32, 0.9, -2.0, 0.3], [-1.0, -3.0, 4.0, 0.0]];
    let want: Vec<f32> = rows
        .iter()
        .zip([1usize, 2])
        .map(|(row, pick)| {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let total: f32 = row.iter().map(|v| (v - max).exp()).sum();
            row[pick] - max - total.ln()
        })
        .collect();
    assert_close(&logprobs.to_f32(), &want, 1e-5, "greedy log-probabilities");
}

#[test]
fn sampled_log_probabilities_are_the_log_softmax_of_what_was_drawn() {
    let rows = 64usize;
    let classes = 5usize;
    let values = noise(rows * classes, 11);
    let logits = tensor(&values, vec![rows, classes]);

    for temperature in [1.0f32, 0.5, 2.0] {
        let (actions, logprobs) = kernels::sample_categorical(&logits, temperature, 7).unwrap();
        let picked = actions.to_vec();
        let want: Vec<f32> = (0..rows)
            .map(|r| {
                let row: Vec<f32> = values[r * classes..(r + 1) * classes]
                    .iter()
                    .map(|v| v / temperature)
                    .collect();
                let max = row.iter().cloned().fold(f32::MIN, f32::max);
                let total: f32 = row.iter().map(|v| (v - max).exp()).sum();
                row[picked[r] as usize] - max - total.ln()
            })
            .collect();
        assert_close(
            &logprobs.to_f32(),
            &want,
            1e-5,
            &format!("sampled log-probabilities at temperature {temperature}"),
        );
        assert!(
            picked.iter().all(|a| (*a as usize) < classes),
            "sampling produced an action outside the action space"
        );
    }
}

#[test]
fn sampling_follows_the_distribution_it_is_given() {
    // One row of logits, replicated many times, so a single launch is a batch of
    // independent draws from the same distribution — each row hashes its own
    // position, so they do not share a coin.
    let row = [2.0f32, 0.0, -1.0, 1.0];
    let classes = row.len();
    let rows = 20_000usize;
    let logits = tensor(
        &row.iter()
            .cycle()
            .take(rows * classes)
            .cloned()
            .collect::<Vec<_>>(),
        vec![rows, classes],
    );

    let max = row.iter().cloned().fold(f32::MIN, f32::max);
    let weights: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = weights.iter().sum();
    let expected: Vec<f32> = weights.iter().map(|w| w / total).collect();

    let actions = kernels::sample_categorical(&logits, 1.0, 12345)
        .unwrap()
        .0
        .to_vec();
    let mut counts = vec![0usize; classes];
    for a in &actions {
        counts[*a as usize] += 1;
    }

    for (class, (count, want)) in counts.iter().zip(&expected).enumerate() {
        let observed = *count as f32 / rows as f32;
        // Three standard errors of a binomial proportion, which a correct sampler
        // clears essentially always and a wrong one misses by a mile.
        let tolerance = 3.0 * (want * (1.0 - want) / rows as f32).sqrt();
        assert!(
            (observed - want).abs() <= tolerance,
            "class {class} was drawn {observed} of the time, expected {want} (± {tolerance})"
        );
    }
}

#[test]
fn a_colder_temperature_concentrates_the_draw() {
    let row = [1.0f32, 0.0, 0.0, 0.0];
    let rows = 4_000usize;
    let logits = tensor(
        &row.iter()
            .cycle()
            .take(rows * 4)
            .cloned()
            .collect::<Vec<_>>(),
        vec![rows, 4],
    );
    let share = |temperature: f32| {
        let actions = kernels::sample_categorical(&logits, temperature, 99)
            .unwrap()
            .0
            .to_vec();
        actions.iter().filter(|a| **a == 0).count() as f32 / rows as f32
    };
    let (hot, cold) = (share(2.0), share(0.25));
    assert!(
        cold > hot + 0.1,
        "temperature 0.25 picked the best action {cold} of the time and temperature 2.0 {hot}"
    );
}

// ---------------------------------------------------------------------------
// Trajectory writes and DAgger's mixture
// ---------------------------------------------------------------------------

#[test]
fn a_step_lands_in_its_own_column() {
    let (envs, steps, width) = (3usize, 4usize, 2usize);
    let buffer = Tensor::<R, f32>::zeros(vec![envs, steps, width], &dev());
    for t in 0..steps {
        let column: Vec<f32> = (0..envs * width).map(|i| (t * 100 + i) as f32).collect();
        kernels::write_step(&buffer, &tensor(&column, vec![envs, width]), t).unwrap();
    }

    let got = buffer.to_f32();
    for env in 0..envs {
        for t in 0..steps {
            for w in 0..width {
                let want = (t * 100 + env * width + w) as f32;
                let at = (env * steps + t) * width + w;
                assert_eq!(got[at], want, "buffer[{env}, {t}, {w}]");
            }
        }
    }

    let ids = IdTensor::<R>::empty(vec![envs, steps], &dev());
    for t in 0..steps {
        let column: Vec<u32> = (0..envs).map(|e| (10 * t + e) as u32).collect();
        kernels::write_step_ids(
            &ids,
            &IdTensor::from_slice(&column, vec![envs], &dev()).unwrap(),
            t,
        )
        .unwrap();
    }
    let got = ids.to_vec();
    for env in 0..envs {
        for t in 0..steps {
            assert_eq!(
                got[env * steps + t],
                (10 * t + env) as u32,
                "ids[{env}, {t}]"
            );
        }
    }
}

#[test]
fn writing_past_the_end_of_a_buffer_is_an_error() {
    let buffer = Tensor::<R, f32>::zeros(vec![2, 3, 1], &dev());
    let step = Tensor::<R, f32>::zeros(vec![2, 1], &dev());
    assert!(kernels::write_step(&buffer, &step, 3).is_err());
    // And a step of the wrong width does not silently scatter.
    let wrong = Tensor::<R, f32>::zeros(vec![2, 5], &dev());
    assert!(kernels::write_step(&buffer, &wrong, 0).is_err());
}

#[test]
fn dagger_mixes_at_the_rate_it_is_asked_to() {
    let n = 20_000usize;
    let learner = IdTensor::<R>::from_slice(&vec![0u32; n], vec![n], &dev()).unwrap();
    let expert = IdTensor::<R>::from_slice(&vec![1u32; n], vec![n], &dev()).unwrap();

    // The endpoints must be exact, not approximate: DAgger's first round is pure
    // expert and its limit is pure learner.
    let all_expert = kernels::mix_actions(&learner, &expert, 1.0, 5)
        .unwrap()
        .to_vec();
    assert!(
        all_expert.iter().all(|a| *a == 1),
        "beta=1 must be all expert"
    );
    let all_learner = kernels::mix_actions(&learner, &expert, 0.0, 5)
        .unwrap()
        .to_vec();
    assert!(
        all_learner.iter().all(|a| *a == 0),
        "beta=0 must be all learner"
    );

    for beta in [0.25f32, 0.5, 0.75] {
        let mixed = kernels::mix_actions(&learner, &expert, beta, 5)
            .unwrap()
            .to_vec();
        let share = mixed.iter().filter(|a| **a == 1).count() as f32 / n as f32;
        let tolerance = 3.0 * (beta * (1.0 - beta) / n as f32).sqrt();
        assert!(
            (share - beta).abs() <= tolerance,
            "beta={beta} took the expert {share} of the time (± {tolerance})"
        );
    }
}

// ---------------------------------------------------------------------------
// The environment
// ---------------------------------------------------------------------------

mod task {
    use super::*;
    use mamba3::rl::{RecallEnv, VecEnv};

    fn env(envs: usize, symbols: usize, horizon: usize) -> RecallEnv<R, f32> {
        RecallEnv::new(envs, symbols, horizon, 4, &dev()).unwrap()
    }

    #[test]
    fn the_cue_is_visible_only_on_the_first_step_of_an_episode() {
        let (envs, symbols, horizon) = (4usize, 3usize, 4usize);
        let mut env = env(envs, symbols, horizon);
        let obs_dim = env.obs_dim();

        let first = env.reset().unwrap().to_f32();
        for e in 0..envs {
            let row = &first[e * obs_dim..(e + 1) * obs_dim];
            let shown: f32 = row[..symbols].iter().sum();
            assert_eq!(
                shown, 1.0,
                "env {e} was not shown exactly one cue at step 0"
            );
            assert_eq!(row[symbols], 0.0, "the clock should read zero at step 0");
            assert_eq!(row[symbols + 1], 1.0, "the cue-present flag should be set");
        }

        // Every later step of the episode shows nothing.
        let expert = env.expert_actions().unwrap();
        for step in 1..horizon {
            let out = env.step(&expert).unwrap();
            if step == horizon - 1 {
                break;
            }
            let obs = out.observation.to_f32();
            for e in 0..envs {
                let row = &obs[e * obs_dim..(e + 1) * obs_dim];
                let shown: f32 = row[..symbols].iter().sum();
                assert_eq!(shown, 0.0, "env {e} was shown a cue at step {step}");
                assert_eq!(
                    row[symbols + 1],
                    0.0,
                    "the cue-present flag leaked at step {step}"
                );
            }
        }
    }

    #[test]
    fn only_naming_the_remembered_cue_at_the_end_pays() {
        let (envs, symbols, horizon) = (8usize, 4usize, 3usize);
        let mut env = env(envs, symbols, horizon);
        env.reset().unwrap();

        for step in 0..horizon {
            let out = env.step(&env.expert_actions().unwrap()).unwrap();
            let reward = out.reward.to_f32();
            let done = out.done.to_f32();
            if step + 1 == horizon {
                assert!(reward.iter().all(|r| *r == 1.0), "the expert was not paid");
                assert!(done.iter().all(|d| *d == 1.0), "the episode did not end");
            } else {
                assert!(reward.iter().all(|r| *r == 0.0), "reward arrived early");
                assert!(done.iter().all(|d| *d == 0.0), "the episode ended early");
            }
        }

        // A policy that names the wrong symbol earns nothing, however confidently.
        // The wrong answer is derived from the *current* cue each step: the episode
        // just ended, so the environment has already drawn a new one, and an answer
        // stale by one episode would be wrong only by luck.
        for step in 0..horizon {
            let cue = env.expert_actions().unwrap().to_vec();
            let wrong: Vec<u32> = cue.iter().map(|c| (c + 1) % symbols as u32).collect();
            let wrong = IdTensor::from_slice(&wrong, vec![envs], &dev()).unwrap();
            let out = env.step(&wrong).unwrap();
            assert!(
                out.reward.to_f32().iter().all(|r| *r == 0.0),
                "a wrong answer was rewarded at step {step}"
            );
        }
    }

    #[test]
    fn every_symbol_is_cued_about_equally_often() {
        // The property the whole task rests on. If the cue were fixed, or even
        // skewed, a policy could beat `1/symbols` by learning the prior and never
        // looking at anything — and the learning tests below, which measure exactly
        // that margin, would be proving nothing.
        let (envs, symbols, horizon) = (16usize, 4usize, 2usize);
        let mut env = env(envs, symbols, horizon);
        env.reset().unwrap();

        let episodes = 300;
        let mut counts = vec![0usize; symbols];
        for _ in 0..episodes {
            for cue in env.expert_actions().unwrap().to_vec() {
                counts[cue as usize] += 1;
            }
            let actions = env.expert_actions().unwrap();
            for _ in 0..horizon {
                env.step(&actions).unwrap();
            }
        }

        let total: usize = counts.iter().sum();
        let uniform = 1.0 / symbols as f32;
        // Three standard errors of a binomial proportion over `total` draws.
        let tolerance = 3.0 * (uniform * (1.0 - uniform) / total as f32).sqrt();
        for (symbol, count) in counts.iter().enumerate() {
            let share = *count as f32 / total as f32;
            assert!(
                (share - uniform).abs() <= tolerance,
                "symbol {symbol} was cued {share:.4} of the time, expected \
                 {uniform:.4} (± {tolerance:.4})"
            );
        }
    }

    #[test]
    fn a_degenerate_configuration_is_refused() {
        assert!(RecallEnv::<R, f32>::new(0, 4, 3, 0, &dev()).is_err());
        assert!(RecallEnv::<R, f32>::new(4, 1, 3, 0, &dev()).is_err());
        assert!(RecallEnv::<R, f32>::new(4, 4, 0, 0, &dev()).is_err());
    }
}

// ---------------------------------------------------------------------------
// Collection, and the agreement PPO rests on
// ---------------------------------------------------------------------------

mod collection {
    use super::*;
    use mamba3::autograd::Var;
    use mamba3::rl::{Collector, Mamba3Policy, Mamba3PolicyConfig, RecallEnv, VecEnv};
    use mamba3::ssm::config::Discretization;

    /// Small enough for the CPU runtime, wide enough that a broken mask shows.
    pub fn policy(obs_dim: usize, actions: usize, seed: u64) -> Mamba3Policy<R, f32> {
        Mamba3PolicyConfig::new(obs_dim, actions, 16, 2)
            .with_seed(seed)
            .with_ssm(|s| {
                s.n_heads = 2;
                s.head_dim = 8;
                s.n_groups = 2;
                s.d_state = 4;
                s.chunk_size = 4;
                s.conv_kernel = Some(3);
                s.discretization = Discretization::LearnedTrapezoid;
            })
            .init::<R, f32>(&dev())
            .unwrap()
    }

    #[test]
    fn a_replayed_window_reproduces_what_the_rollout_did() {
        // The property every policy gradient in this module rests on. PPO scores a
        // recomputed pass against log-probabilities recorded step by step during the
        // rollout; if the two passes disagree, the importance ratio is not 1 at the
        // first epoch and every update after it is measured against a policy that
        // never acted.
        let (envs, steps, symbols, horizon) = (4usize, 8usize, 3usize, 3usize);
        let mut env = RecallEnv::<R, f32>::new(envs, symbols, horizon, 11, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let policy = policy(obs_dim, symbols, 3);

        let mut collector = Collector::new(&policy, envs, steps, obs_dim, &dev())
            .unwrap()
            .with_seed(5);
        let report = collector.collect(&mut env).unwrap();
        let buffer = collector.buffer();

        // Replay the same window through the scan, under the same episode mask and
        // from the same starting state.
        let (out, _) = policy
            .forward(
                &Var::constant(buffer.observations().clone()),
                Some(&buffer.reset_mask().unwrap()),
                Some(&report.initial),
            )
            .unwrap();

        let rows = envs * steps;
        let replayed = out
            .logits
            .reshape(vec![rows, symbols])
            .unwrap()
            .log_softmax(1)
            .unwrap()
            .take_along_last(&buffer.actions().reshape(vec![rows]).unwrap())
            .unwrap()
            .to_f32();

        assert_close(
            &replayed,
            &buffer.log_probs().to_f32(),
            1e-4,
            "the replayed log-probabilities of the recorded actions",
        );
    }

    #[test]
    fn a_mixture_records_the_probability_of_what_it_actually_did() {
        // Under DAgger the expert overrides the draw, so the action stored is not
        // the action sampled. The log-probability stored beside it has to belong to
        // the one that was *executed* — it is the denominator of any importance
        // ratio taken later, and scoring the discarded draw instead would record the
        // density of something that never happened, silently.
        let (envs, steps, symbols, horizon) = (4usize, 6usize, 3usize, 3usize);
        let mut env = RecallEnv::<R, f32>::new(envs, symbols, horizon, 31, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let policy = policy(obs_dim, symbols, 8);

        let mut collector = Collector::new(&policy, envs, steps, obs_dim, &dev())
            .unwrap()
            .with_seed(9)
            .recording_expert_labels();
        // Pure expert: every step overrides the draw, so a bug here is unmissable.
        let report = collector.collect_with_expert(&mut env, 1.0).unwrap();
        let buffer = collector.buffer();

        let (out, _) = policy
            .forward(
                &Var::constant(buffer.observations().clone()),
                Some(&buffer.reset_mask().unwrap()),
                Some(&report.initial),
            )
            .unwrap();

        let rows = envs * steps;
        let replayed = out
            .logits
            .reshape(vec![rows, symbols])
            .unwrap()
            .log_softmax(1)
            .unwrap()
            .take_along_last(&buffer.actions().reshape(vec![rows]).unwrap())
            .unwrap()
            .to_f32();
        assert_close(
            &replayed,
            &buffer.log_probs().to_f32(),
            1e-4,
            "the log-probability of the action the mixture executed",
        );
    }

    #[test]
    fn the_reset_mask_is_the_termination_mask_one_step_later() {
        let (envs, steps, symbols, horizon) = (3usize, 7usize, 3usize, 2usize);
        let mut env = RecallEnv::<R, f32>::new(envs, symbols, horizon, 2, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let policy = policy(obs_dim, symbols, 1);
        let mut collector = Collector::new(&policy, envs, steps, obs_dim, &dev()).unwrap();

        collector.collect(&mut env).unwrap();
        let buffer = collector.buffer();
        let dones = buffer.dones().to_f32();
        let resets = buffer.reset_mask().unwrap().to_f32();

        for e in 0..envs {
            // The first window starts fresh, so nothing had ended before it.
            assert_eq!(resets[e * steps], 0.0, "env {e} opened the run mid-episode");
            for t in 1..steps {
                assert_eq!(
                    resets[e * steps + t],
                    dones[e * steps + t - 1],
                    "reset[{e}, {t}] does not follow done[{e}, {t}-1]"
                );
            }
        }

        // A second window must carry the last termination of the first across.
        let last_done = buffer.last_done().unwrap().to_f32();
        collector.collect(&mut env).unwrap();
        let carried = collector.buffer().reset_mask().unwrap().to_f32();
        for e in 0..envs {
            assert_eq!(
                carried[e * steps],
                last_done[e],
                "env {e} lost the termination that straddled the window boundary"
            );
        }
    }

    #[test]
    fn an_environment_reached_through_a_reference_collects_the_same_window() {
        // `VecEnv` is implemented for `&mut V` as well, so a caller holding its
        // environment behind a trait object — a boxed simulator, a binding to
        // another language — can drive a collector without re-wrapping it. The two
        // paths must be the same collection, not merely both legal.
        let (envs, steps, symbols, horizon) = (3usize, 6usize, 3usize, 2usize);
        let obs_dim = RecallEnv::<R, f32>::new(envs, symbols, horizon, 9, &dev())
            .unwrap()
            .obs_dim();
        let policy = policy(obs_dim, symbols, 4);

        let collect = |through_a_reference: bool| {
            let mut env = RecallEnv::<R, f32>::new(envs, symbols, horizon, 9, &dev()).unwrap();
            let mut collector = Collector::new(&policy, envs, steps, obs_dim, &dev())
                .unwrap()
                .with_seed(7);
            if through_a_reference {
                let mut indirect: &mut dyn VecEnv<R, f32> = &mut env;
                collector.collect(&mut indirect).unwrap();
            } else {
                collector.collect(&mut env).unwrap();
            }
            (
                collector.buffer().actions().to_vec(),
                collector.buffer().rewards().to_f32(),
            )
        };

        let (direct_actions, direct_rewards) = collect(false);
        let (indirect_actions, indirect_rewards) = collect(true);
        assert_eq!(direct_actions, indirect_actions);
        assert_close(
            &direct_rewards,
            &indirect_rewards,
            0.0,
            "a window collected through a reference",
        );
    }

    #[test]
    fn a_buffer_refuses_to_overflow_and_reports_its_own_size() {
        use mamba3::rl::TrajectoryBuffer;

        let mut buffer = TrajectoryBuffer::<R, f32>::new(2, 3, 4, &dev()).unwrap();
        assert!(buffer.is_empty());
        assert!(TrajectoryBuffer::<R, f32>::new(0, 3, 4, &dev()).is_err());

        let ones = Tensor::<R, f32>::ones(vec![2], &dev());
        let obs = Tensor::<R, f32>::ones(vec![2, 4], &dev());
        let ids = IdTensor::from_slice(&[0u32, 1], vec![2], &dev()).unwrap();
        let push = |b: &mut TrajectoryBuffer<R, f32>| {
            b.push(mamba3::rl::Transition {
                observation: &obs,
                action: &ids,
                log_prob: &ones,
                value: &ones,
                reward: &ones,
                done: &ones,
                action_mask: None,
            })
        };
        for _ in 0..3 {
            push(&mut buffer).unwrap();
        }
        assert!(buffer.is_full());
        assert!(
            push(&mut buffer).is_err(),
            "a full buffer accepted a fourth step"
        );

        let before = buffer.bytes();
        buffer.rewind(None).unwrap();
        assert_eq!(before, buffer.bytes(), "rewinding changed the footprint");
        assert!(buffer.is_empty());
    }
}

// ---------------------------------------------------------------------------
// A4: completed-episode returns, not window fragments
// ---------------------------------------------------------------------------

mod episode_returns {
    use super::*;
    use mamba3::error::Result;
    use mamba3::rl::{Collector, EnvStep, RecallEnv, VecEnv};
    use mamba3::tensor::ops::index::IdTensor;

    use super::collection::policy;

    /// An environment whose rewards and terminations are scripted per step, so a
    /// window's completed-episode return can be checked against a number worked
    /// out by hand rather than trusted to whatever a real task happens to do.
    struct ScriptedEnv {
        envs: usize,
        obs_dim: usize,
        t: usize,
        rewards: Vec<Vec<f32>>,
        dones: Vec<Vec<f32>>,
        device: Device<R>,
    }

    impl ScriptedEnv {
        fn new(
            rewards: Vec<Vec<f32>>,
            dones: Vec<Vec<f32>>,
            obs_dim: usize,
            device: &Device<R>,
        ) -> Self {
            Self {
                envs: rewards.len(),
                obs_dim,
                t: 0,
                rewards,
                dones,
                device: device.clone(),
            }
        }
    }

    impl VecEnv<R, f32> for ScriptedEnv {
        fn envs(&self) -> usize {
            self.envs
        }

        fn obs_dim(&self) -> usize {
            self.obs_dim
        }

        fn action_dim(&self) -> usize {
            2
        }

        fn reset(&mut self) -> Result<Tensor<R, f32>> {
            self.t = 0;
            Ok(Tensor::zeros(vec![self.envs, self.obs_dim], &self.device))
        }

        fn step(&mut self, _actions: &IdTensor<R>) -> Result<EnvStep<R, f32>> {
            let t = self.t;
            let reward: Vec<f32> = (0..self.envs).map(|e| self.rewards[e][t]).collect();
            let done: Vec<f32> = (0..self.envs).map(|e| self.dones[e][t]).collect();
            self.t += 1;
            Ok(EnvStep {
                observation: Tensor::zeros(vec![self.envs, self.obs_dim], &self.device),
                reward: tensor(&reward, vec![self.envs]),
                done: tensor(&done, vec![self.envs]),
            })
        }
    }

    /// `(mean, count)`, read back for a hand-checkable assertion.
    fn read(collector: &Collector<'_, R, f32>) -> (f32, f32) {
        let (mean, count) = collector.episode_return().unwrap();
        (mean.to_f32()[0], count.to_f32()[0])
    }

    #[test]
    fn a_window_shorter_than_one_episode_completes_nothing() {
        // The bug this whole item exists to fix: dividing a fragment's total
        // reward by `max(dones, 1)` reports that total as if it were a mean,
        // silently changing units the moment nothing finishes. The fix reports
        // zero episodes instead.
        let (envs, symbols, horizon) = (4usize, 3usize, 4usize);
        let mut env = RecallEnv::<R, f32>::new(envs, symbols, horizon, 17, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let policy = policy(obs_dim, symbols, 21);
        let mut collector = Collector::new(&policy, envs, 1, obs_dim, &dev())
            .unwrap()
            .with_seed(1);
        collector.collect(&mut env).unwrap();
        let (_, count) = read(&collector);
        assert_eq!(
            count, 0.0,
            "one step of a 4-step horizon cannot complete an episode"
        );
    }

    #[test]
    fn completed_returns_are_correct_across_a_window_boundary() {
        // Two lanes, three steps per window, six steps scripted by hand. Lane 0
        // completes a short episode inside window 1 and a longer one that starts
        // in window 1 and ends in window 2; lane 1's only episode starts in
        // window 1 and ends in window 2. Both properties A4 exists for are here
        // at once: an episode that spans the boundary is not dropped (lane 1,
        // and lane 0's second episode), and one that does not is not confused
        // with the window's total (lane 0's first episode).
        let rewards = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ];
        let dones = vec![
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ];
        let obs_dim = 2;
        let policy = policy(obs_dim, 2, 5);
        let mut env = ScriptedEnv::new(rewards, dones, obs_dim, &dev());
        let mut collector = Collector::new(&policy, 2, 3, obs_dim, &dev())
            .unwrap()
            .with_seed(1);

        collector.collect(&mut env).unwrap();
        let (mean1, count1) = read(&collector);
        assert_eq!(
            count1, 1.0,
            "only lane 0's first episode finishes in window 1"
        );
        assert!(
            (mean1 - 3.0).abs() < 1e-5,
            "lane 0's first episode earned 1 + 2 = 3, got mean {mean1}"
        );

        collector.collect(&mut env).unwrap();
        let (mean2, count2) = read(&collector);
        assert_eq!(
            count2, 2.0,
            "lane 0's second episode and lane 1's only episode both finish in window 2"
        );
        // Lane 0: 3 (carried in) + 4 + 5 = 12. Lane 1: 10 + 20 + 30 (carried in) + 40 = 100.
        assert!(
            (mean2 - 56.0).abs() < 1e-4,
            "expected (12 + 100) / 2 = 56, got {mean2}"
        );

        // Reading twice must not change the answer.
        let (mean_again, count_again) = read(&collector);
        assert_eq!(
            (mean2, count2),
            (mean_again, count_again),
            "reading consumed the counters"
        );
    }

    #[test]
    fn a_negative_return_is_reported_as_negative() {
        let rewards = vec![vec![-5.0, -3.0]];
        let dones = vec![vec![0.0, 1.0]];
        let obs_dim = 2;
        let policy = policy(obs_dim, 2, 6);
        let mut env = ScriptedEnv::new(rewards, dones, obs_dim, &dev());
        let mut collector = Collector::new(&policy, 1, 2, obs_dim, &dev())
            .unwrap()
            .with_seed(1);
        collector.collect(&mut env).unwrap();
        let (mean, count) = read(&collector);
        assert_eq!(count, 1.0);
        assert!(
            (mean - (-8.0)).abs() < 1e-5,
            "expected -5 + -3 = -8, got {mean}"
        );
    }

    #[test]
    fn resetting_the_collector_forgets_an_episode_in_progress() {
        // Lane 0 earns 1 + 2 = 3 without finishing; resetting must not let that
        // leftover leak into whatever the next episode happens to complete with.
        let rewards = vec![vec![1.0, 2.0]];
        let dones = vec![vec![0.0, 0.0]];
        let obs_dim = 2;
        let policy = policy(obs_dim, 2, 7);
        let mut env = ScriptedEnv::new(rewards, dones, obs_dim, &dev());
        let mut collector = Collector::new(&policy, 1, 2, obs_dim, &dev())
            .unwrap()
            .with_seed(1);

        collector.collect(&mut env).unwrap();
        let (_, count1) = read(&collector);
        assert_eq!(count1, 0.0, "1 + 2 has not finished an episode yet");

        // `Collector::reset` clears the observation, so the next `collect` calls
        // `env.reset()` itself and zeroes `env.t` through it.
        collector.reset();
        // Re-script so the next window starts a fresh, single-step episode; if
        // the 3 in progress before the reset leaked in, this would report 101
        // instead of 1.
        env.rewards = vec![vec![1.0, 1.0]];
        env.dones = vec![vec![1.0, 0.0]];
        collector.collect(&mut env).unwrap();
        let (mean2, count2) = read(&collector);
        assert_eq!(count2, 1.0);
        assert!(
            (mean2 - 1.0).abs() < 1e-5,
            "the pre-reset running return leaked across the reset: got mean {mean2}"
        );
    }
}

// ---------------------------------------------------------------------------
// What the clipped surrogate actually does
// ---------------------------------------------------------------------------

mod objective {
    use super::*;
    use mamba3::autograd::Var;
    use mamba3::rl::{PolicyOutput, PpoBatch, PpoConfig, ppo_objective};

    type V = Var<R, f32>;

    struct Fixture {
        envs: usize,
        steps: usize,
        actions: usize,
    }

    impl Fixture {
        fn small() -> Self {
            Self {
                envs: 3,
                steps: 4,
                actions: 5,
            }
        }

        fn rows(&self) -> usize {
            self.envs * self.steps
        }

        /// Logits as a tracked leaf, so `d loss / d logits` can be read directly.
        fn logits(&self, seed: u64) -> V {
            V::traced(tensor(
                &noise(self.rows() * self.actions, seed),
                vec![self.envs, self.steps, self.actions],
            ))
        }

        fn actions_taken(&self, seed: u64) -> IdTensor<R> {
            let ids: Vec<u32> = (0..self.rows())
                .map(|i| ((i as u64 * seed + 3) % self.actions as u64) as u32)
                .collect();
            IdTensor::from_slice(&ids, vec![self.envs, self.steps], &dev()).unwrap()
        }

        /// The log-probability the policy currently assigns to each taken action.
        fn current_log_probs(&self, logits: &V, actions: &IdTensor<R>) -> V {
            logits
                .reshape(vec![self.rows(), self.actions])
                .unwrap()
                .log_softmax(1)
                .unwrap()
                .take_along_last(&actions.reshape(vec![self.rows()]).unwrap())
                .unwrap()
        }

        fn batch(
            &self,
            actions: IdTensor<R>,
            old_log_probs: Tensor<R, f32>,
            advantages: Vec<f32>,
        ) -> PpoBatch<R, f32> {
            let shape = vec![self.envs, self.steps];
            PpoBatch {
                observations: Tensor::zeros(vec![self.envs, self.steps, 1], &dev()),
                actions,
                log_probs: old_log_probs.reshape(shape.clone()).unwrap(),
                advantages: tensor(&advantages, shape.clone()),
                returns: Tensor::zeros(shape.clone(), &dev()),
                values: Tensor::zeros(shape, &dev()),
                reset: None,
                initial: None,
                mask: None,
                reference_log_probs: None,
                action_mask: None,
            }
        }

        /// Only the surrogate: no critic, no entropy, and a trust region wide
        /// enough that nothing is ever clipped.
        fn surrogate_only(&self, clip: f32) -> PpoConfig {
            PpoConfig::new()
                .with_clip(clip)
                .with_coefficients(0.0, 0.0)
                .with_normalized_advantages(false)
        }

        fn value(&self) -> V {
            V::constant(Tensor::zeros(vec![self.envs, self.steps], &dev()))
        }
    }

    /// `d loss / d leaf` for a leaf that is not a parameter.
    ///
    /// `backward_retain` rather than `backward`: the plain one keeps only the
    /// parameter gradients an optimizer will ask for and drops the intermediate
    /// nodes, and the logits here are an intermediate by construction.
    fn gradient_of(loss: &V, leaf: &V) -> Vec<f32> {
        let grads = loss.backward_retain().unwrap();
        grads
            .node(leaf.node().expect("the leaf is tracked"))
            .expect("the loss depends on the leaf")
            .to_f32()
    }

    #[test]
    fn the_first_epoch_compares_the_policy_with_itself() {
        // Before any update, the recomputed policy *is* the behaviour policy, so
        // every ratio is 1: no KL, nothing clipped. A run that does not start here
        // has a bug between the rollout and the replay, and everything after it is
        // measured against a policy that never acted.
        let f = Fixture::small();
        let logits = f.logits(1);
        let actions = f.actions_taken(7);
        let old = f.current_log_probs(&logits, &actions).tensor().clone();
        let batch = f.batch(actions, old, noise(f.rows(), 4));

        let out = PolicyOutput {
            logits,
            value: f.value(),
        };
        let loss = ppo_objective(&out, &batch, &f.surrogate_only(0.2)).unwrap();

        assert!(
            loss.approx_kl.to_f32()[0].abs() < 1e-6,
            "the first epoch reported a KL of {}",
            loss.approx_kl.to_f32()[0]
        );
        assert_eq!(
            loss.clip_fraction.to_f32()[0],
            0.0,
            "the first epoch clipped something"
        );
        // And with every ratio at 1 the surrogate is just the mean advantage.
        let want = -noise(f.rows(), 4).iter().sum::<f32>() / f.rows() as f32;
        assert_close(
            &loss.policy.to_f32(),
            &[want],
            1e-5,
            "the surrogate at ratio 1",
        );
    }

    #[test]
    fn without_a_clip_ppo_is_the_vanilla_policy_gradient() {
        // At ratio 1 and with an unreachable trust region, `d/dθ -E[r A]` is
        // `-E[A ∇log π]` — the score-function estimator PPO is a refinement of. If
        // these two gradients differ, the refinement has changed the thing it was
        // supposed to refine.
        let f = Fixture::small();
        let advantages = noise(f.rows(), 5);
        let actions = f.actions_taken(11);

        let logits = f.logits(2);
        let old = f.current_log_probs(&logits, &actions).tensor().clone();
        let batch = f.batch(actions.clone(), old, advantages.clone());
        let clipped = ppo_objective(
            &PolicyOutput {
                logits: logits.clone(),
                value: f.value(),
            },
            &batch,
            &f.surrogate_only(1e6),
        )
        .unwrap();
        let ppo_grad = gradient_of(&clipped.total, &logits);

        // The same objective written the textbook way, on its own tape.
        let plain = f.logits(2);
        let log_probs = f.current_log_probs(&plain, &actions);
        let weights = V::constant(tensor(&advantages, vec![f.rows()]));
        let vanilla = log_probs.mul(&weights).unwrap().mean().unwrap().neg();
        let vanilla_grad = gradient_of(&vanilla, &plain);

        assert_close(
            &ppo_grad,
            &vanilla_grad,
            1e-5,
            "PPO without a clip against the policy gradient",
        );
    }

    #[test]
    fn the_clip_stops_an_overshoot_but_not_a_recovery() {
        // The asymmetry that makes the bound *pessimistic* rather than symmetric.
        // Both cases put the ratio far outside the trust region; what differs is the
        // sign of the advantage, and so whether moving further is a step in the
        // direction the data supports or away from it.
        let f = Fixture::small();
        let actions = f.actions_taken(13);
        let config = f.surrogate_only(0.2);

        let magnitude = |advantage: f32| {
            let logits = f.logits(3);
            // An old log-probability far *below* the current one makes the ratio
            // enormous: the policy has already moved much further than this data
            // licenses.
            let old = f.current_log_probs(&logits, &actions).tensor().clone();
            let old = crate::tensor(
                &old.to_f32().iter().map(|v| v - 3.0).collect::<Vec<_>>(),
                vec![f.rows()],
            );
            let batch = f.batch(actions.clone(), old, vec![advantage; f.rows()]);
            let loss = ppo_objective(
                &PolicyOutput {
                    logits: logits.clone(),
                    value: f.value(),
                },
                &batch,
                &config,
            )
            .unwrap();
            assert_eq!(
                loss.clip_fraction.to_f32()[0],
                1.0,
                "the fixture was supposed to put every ratio outside the trust region"
            );
            gradient_of(&loss.total, &logits)
                .iter()
                .map(|g| g.abs())
                .fold(0.0f32, f32::max)
        };

        assert!(
            magnitude(1.0) < 1e-7,
            "an action that already looks too good kept pushing: {}",
            magnitude(1.0)
        );
        assert!(
            magnitude(-1.0) > 1e-3,
            "an action that turned out badly was not pulled back: {}",
            magnitude(-1.0)
        );
    }

    #[test]
    fn a_mask_excludes_a_position_from_every_term() {
        // A padded or unlabelled position must not shift the loss, and must not
        // receive a gradient either.
        let f = Fixture::small();
        let actions = f.actions_taken(17);
        let logits = f.logits(4);
        let old = f.current_log_probs(&logits, &actions).tensor().clone();
        let mut batch = f.batch(actions, old, noise(f.rows(), 6));

        let mut mask = vec![1.0f32; f.rows()];
        mask[2] = 0.0;
        mask[f.rows() - 1] = 0.0;
        batch.mask = Some(tensor(&mask, vec![f.envs, f.steps]));

        let loss = ppo_objective(
            &PolicyOutput {
                logits: logits.clone(),
                value: f.value(),
            },
            &batch,
            &f.surrogate_only(0.2),
        )
        .unwrap();
        let grad = gradient_of(&loss.total, &logits);

        for row in [2usize, f.rows() - 1] {
            let slice = &grad[row * f.actions..(row + 1) * f.actions];
            assert!(
                slice.iter().all(|g| g.abs() < 1e-8),
                "a masked position received a gradient: {slice:?}"
            );
        }
        assert!(
            grad.iter().any(|g| g.abs() > 1e-6),
            "masking removed every gradient, so the test proves nothing"
        );
    }

    #[test]
    fn an_impossible_configuration_is_refused() {
        assert!(PpoConfig::new().with_clip(0.0).validate().is_err());
        assert!(
            PpoConfig::new()
                .with_discount(1.5, 0.95)
                .validate()
                .is_err()
        );
        assert!(PpoConfig::default().validate().is_ok());
    }
}

// ---------------------------------------------------------------------------
// The only test that finally matters: does it learn?
// ---------------------------------------------------------------------------
//
// Every test above checks a piece against its definition. These two check the
// whole stack against the world: a policy that starts out guessing has to end up
// solving a task it cannot solve without remembering something. Deliberately small
// — one layer, sixteen environments, a three-step episode — so they run in seconds
// while still being unsolvable by a memoryless policy.

mod learning {
    use super::*;
    use mamba3::nn::module::Module;
    use mamba3::rl::{
        BehaviourCloningTask, Collector, Mamba3Policy, Mamba3PolicyConfig, PpoConfig, PpoTask,
        RecallEnv, VecEnv,
    };
    use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

    const ENVS: usize = 16;
    const SYMBOLS: usize = 3;
    const HORIZON: usize = 3;
    const WINDOW: usize = HORIZON * 3;

    fn policy(obs_dim: usize) -> Mamba3Policy<R, f32> {
        Mamba3PolicyConfig::new(obs_dim, SYMBOLS, 32, 1)
            .with_seed(7)
            .with_ssm(|s| {
                s.n_heads = 2;
                s.head_dim = 16;
                s.n_groups = 2;
                s.d_state = 8;
                s.chunk_size = 8;
                s.conv_kernel = Some(3);
            })
            .init::<R, f32>(&dev())
            .unwrap()
    }

    fn trainer(lr: f32) -> Trainer<R, f32, mamba3::train::AdamW<R, f32>> {
        Trainer::new(
            TrainerConfig::builder()
                .learning_rate(lr)
                .max_grad_norm(0.5)
                .build()
                .unwrap(),
            AdamWConfig::builder()
                .learning_rate(lr)
                .build()
                .init::<R, f32>(),
        )
    }

    /// Greedy return per episode: what the policy believes, not what its
    /// exploration noise produced.
    fn greedy_return(policy: &Mamba3Policy<R, f32>, seed: u64) -> f32 {
        let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, seed, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let mut collector = Collector::new(policy, ENVS, WINDOW, obs_dim, &dev())
            .unwrap()
            .with_temperature(0.0);
        collector.collect(&mut env).unwrap();
        let (mean, _count) = collector.episode_return().unwrap();
        mean.to_f32()[0]
    }

    #[test]
    fn behaviour_cloning_learns_to_carry_the_cue() {
        let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, 11, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let policy = policy(obs_dim);
        let chance = env.chance_return();

        let mut collector = Collector::new(&policy, ENVS, WINDOW, obs_dim, &dev())
            .unwrap()
            .with_seed(3)
            .recording_expert_labels();
        let task = BehaviourCloningTask::new(&policy);
        let mut trainer = trainer(5e-3);

        // Pure expert acting, which is behaviour cloning proper: the states are the
        // expert's own, and the labels are what it did on them.
        collector.collect_with_expert(&mut env, 1.0).unwrap();
        let before = task
            .agreement(&collector.imitation_batch().unwrap())
            .unwrap();

        for _ in 0..14 {
            collector.collect_with_expert(&mut env, 1.0).unwrap();
            let batch = collector.imitation_batch().unwrap();
            trainer.step(&task, &[batch]).unwrap();
        }

        collector.collect_with_expert(&mut env, 1.0).unwrap();
        let after = task
            .agreement(&collector.imitation_batch().unwrap())
            .unwrap();
        assert!(
            after > 0.9,
            "cloning reached only {after:.3} agreement with the expert (was {before:.3})"
        );

        // Agreement is measured against a label the policy could in principle have
        // guessed. The return is not: it is only earned by naming the cue two steps
        // after it disappeared.
        let earned = greedy_return(&policy, 91);
        assert!(
            earned > 0.9,
            "the cloned policy agrees with the expert but earns only {earned:.3} \
             per episode, against {chance:.3} for guessing"
        );
    }

    #[test]
    fn ppo_learns_to_carry_the_cue_from_reward_alone() {
        // No expert anywhere in this test. The only signal is a reward that arrives
        // once per episode, two steps after the information needed to earn it.
        let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, 23, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let policy = policy(obs_dim);
        let chance = env.chance_return();

        let before = greedy_return(&policy, 91);
        let mut collector = Collector::new(&policy, ENVS, WINDOW, obs_dim, &dev())
            .unwrap()
            .with_seed(5);
        let config = PpoConfig::default()
            .with_discount(0.99, 0.95)
            .with_clip(0.2)
            .with_coefficients(0.5, 0.01);
        let task = PpoTask::new(&policy, config);
        let mut trainer = trainer(3e-3);

        for _ in 0..40 {
            let report = collector.collect(&mut env).unwrap();
            let batch = collector.ppo_batch(&report, &config).unwrap();
            for _ in 0..3 {
                trainer.step(&task, std::slice::from_ref(&batch)).unwrap();
            }
        }

        let after = greedy_return(&policy, 91);
        assert!(
            after > 0.75,
            "PPO reached {after:.3} per episode, up from {before:.3}; guessing earns \
             {chance:.3} and a policy that remembers the cue earns 1.0"
        );

        // The diagnostics have to be finite and in range, because a run whose KL or
        // clip fraction is nonsense can still stumble into a good return by luck.
        let stats = task.stats().expect("forty rounds of updates were run");
        assert!(
            stats.approx_kl.is_finite() && stats.approx_kl >= 0.0,
            "approximate KL was {}",
            stats.approx_kl
        );
        assert!(
            (0.0..=1.0).contains(&stats.clip_fraction),
            "clip fraction was {}",
            stats.clip_fraction
        );
        assert!(stats.entropy > 0.0, "the policy collapsed to a point mass");
    }

    #[test]
    fn only_the_named_parameters_move() {
        // A PPO run restricted to part of the policy must leave the rest exactly
        // where it was — the property a LoRA or head-only fine-tune depends on.
        let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, 5, &dev()).unwrap();
        let obs_dim = env.obs_dim();
        let policy = policy(obs_dim);
        let config = PpoConfig::default();

        let frozen: Vec<(String, Vec<f32>)> = policy
            .named_parameters()
            .into_iter()
            .filter(|(name, _)| !name.contains("actor") && !name.contains("critic"))
            .map(|(name, p)| (name, p.value().to_f32()))
            .collect();

        let task = PpoTask::new(&policy, config).only(&["actor", "critic"]);
        let mut collector = Collector::new(&policy, ENVS, WINDOW, obs_dim, &dev()).unwrap();
        let mut trainer = trainer(1e-2);
        for _ in 0..3 {
            let report = collector.collect(&mut env).unwrap();
            let batch = collector.ppo_batch(&report, &config).unwrap();
            trainer.step(&task, &[batch]).unwrap();
        }

        let after: std::collections::HashMap<String, Vec<f32>> = policy
            .named_parameters()
            .into_iter()
            .map(|(name, p)| (name, p.value().to_f32()))
            .collect();
        for (name, before) in frozen {
            assert_eq!(
                after[&name], before,
                "{name} moved, but only the heads were being trained"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Collecting from environments on worker threads
// ---------------------------------------------------------------------------

mod parallel {
    use super::collection::policy;
    use super::*;
    use mamba3::error::Result as MResult;
    use mamba3::rl::{Collector, EnvStep, MultiSyncCollector, ParallelEnvs, RecallEnv, VecEnv};
    use std::time::{Duration, Instant};

    const SYMBOLS: usize = 3;
    const HORIZON: usize = 3;

    fn recall(envs: usize, seed: u64) -> RecallEnv<R, f32> {
        RecallEnv::new(envs, SYMBOLS, HORIZON, seed, &dev()).unwrap()
    }

    /// Rows `[env, ..]` of a `[envs, steps, width]` buffer, flattened.
    fn rows(data: &[f32], envs: usize, first: usize, count: usize) -> Vec<f32> {
        let stride = data.len() / envs;
        data[first * stride..(first + count) * stride].to_vec()
    }

    #[test]
    fn fanning_out_does_not_change_the_data() {
        // The property that makes the parallel collector a drop-in: worker `w`'s
        // rows of the joined batch are exactly what that worker's environment would
        // have produced on its own. Greedy actions, so the comparison is exact
        // rather than statistical — a sampled action would draw from a different
        // row index in the wider batch and the two runs would legitimately diverge.
        let (per_worker, steps) = (3usize, 6usize);
        let seeds = [101u64, 202];
        let obs_dim = recall(per_worker, 0).obs_dim();
        let policy = policy(obs_dim, SYMBOLS, 4);

        let mut together = MultiSyncCollector::new(
            &policy,
            seeds.iter().map(|s| recall(per_worker, *s)).collect(),
            steps,
            &dev(),
        )
        .unwrap()
        .with_temperature(0.0);
        together.collect().unwrap();
        let joined = together.buffer();
        let total = per_worker * seeds.len();

        assert_eq!(together.workers(), 2);
        assert_eq!(together.envs(), total);
        assert_eq!(together.widths(), vec![per_worker, per_worker]);

        for (worker, seed) in seeds.iter().enumerate() {
            let mut alone = Collector::new(&policy, per_worker, steps, obs_dim, &dev())
                .unwrap()
                .with_temperature(0.0);
            alone.collect(&mut recall(per_worker, *seed)).unwrap();
            let one = alone.buffer();

            let first = worker * per_worker;
            for (what, joined, alone) in [
                (
                    "observations",
                    joined.observations().to_f32(),
                    one.observations().to_f32(),
                ),
                (
                    "log_probs",
                    joined.log_probs().to_f32(),
                    one.log_probs().to_f32(),
                ),
                ("values", joined.values().to_f32(), one.values().to_f32()),
                ("rewards", joined.rewards().to_f32(), one.rewards().to_f32()),
                ("dones", joined.dones().to_f32(), one.dones().to_f32()),
            ] {
                assert_close(
                    &rows(&joined, total, first, per_worker),
                    &alone,
                    1e-6,
                    &format!("worker {worker}'s {what}"),
                );
            }
            let joined_actions = joined.actions().to_vec();
            let stride = joined_actions.len() / total;
            assert_eq!(
                joined_actions[first * stride..(first + per_worker) * stride].to_vec(),
                one.actions().to_vec(),
                "worker {worker}'s actions"
            );
        }
    }

    /// An environment that takes its time, the way a real simulator does.
    ///
    /// It sleeps rather than spinning: the point being measured is that `W` workers
    /// wait *concurrently*, and a sleep makes that visible without the result
    /// depending on how many cores happen to be idle.
    struct Slow<V> {
        inner: V,
        delay: Duration,
    }

    impl<V: VecEnv<R, f32>> VecEnv<R, f32> for Slow<V> {
        fn envs(&self) -> usize {
            self.inner.envs()
        }
        fn obs_dim(&self) -> usize {
            self.inner.obs_dim()
        }
        fn action_dim(&self) -> usize {
            self.inner.action_dim()
        }
        fn reset(&mut self) -> MResult<Tensor<R, f32>> {
            self.inner.reset()
        }
        fn step(&mut self, actions: &IdTensor<R>) -> MResult<EnvStep<R, f32>> {
            std::thread::sleep(self.delay);
            self.inner.step(actions)
        }
        fn expert_actions(&self) -> Option<IdTensor<R>> {
            self.inner.expert_actions()
        }
    }

    #[test]
    fn slow_environments_run_at_the_same_time() {
        if std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            < 2
        {
            // One core cannot demonstrate concurrency; the correctness tests above
            // still cover the mechanism.
            return;
        }
        let (workers, per_worker, steps) = (4usize, 2usize, 5usize);
        let delay = Duration::from_millis(20);
        let obs_dim = recall(per_worker, 0).obs_dim();
        let policy = policy(obs_dim, SYMBOLS, 4);
        let slow = |seed: u64| Slow {
            inner: recall(per_worker, seed),
            delay,
        };

        // The baseline is the same four environments collected one after another,
        // which is what a single-threaded collector over four pools would cost.
        let sequential = Instant::now();
        for w in 0..workers {
            let mut alone = Collector::new(&policy, per_worker, steps, obs_dim, &dev()).unwrap();
            alone.collect(&mut slow(w as u64)).unwrap();
        }
        let sequential = sequential.elapsed();

        let mut together = MultiSyncCollector::new(
            &policy,
            (0..workers).map(|w| slow(w as u64)).collect(),
            steps,
            &dev(),
        )
        .unwrap();
        // Warm up: the first window compiles kernels, which would otherwise be
        // counted as if it were environment time.
        together.collect().unwrap();
        let parallel = Instant::now();
        together.collect().unwrap();
        let parallel = parallel.elapsed();

        // Four workers should take about a quarter of the time. Asserting only that
        // it beats 60% leaves room for a loaded machine while still failing loudly
        // if the workers are secretly running in series.
        assert!(
            parallel.as_secs_f64() < 0.6 * sequential.as_secs_f64(),
            "four workers took {parallel:?} against {sequential:?} for the same work \
             done one at a time; they are not overlapping"
        );
    }

    /// An environment that fails on its second step.
    struct Broken {
        inner: RecallEnv<R, f32>,
        steps: usize,
    }

    impl VecEnv<R, f32> for Broken {
        fn envs(&self) -> usize {
            self.inner.envs()
        }
        fn obs_dim(&self) -> usize {
            self.inner.obs_dim()
        }
        fn action_dim(&self) -> usize {
            self.inner.action_dim()
        }
        fn reset(&mut self) -> MResult<Tensor<R, f32>> {
            self.inner.reset()
        }
        fn step(&mut self, actions: &IdTensor<R>) -> MResult<EnvStep<R, f32>> {
            self.steps += 1;
            if self.steps == 2 {
                return Err(mamba3::error::Error::config("the simulator fell over"));
            }
            self.inner.step(actions)
        }
    }

    #[test]
    fn a_failing_worker_surfaces_instead_of_hanging() {
        // A worker that errors must come back as an error on the owning thread. The
        // failure mode this guards against is the barrier waiting forever for an
        // answer that will never come.
        let obs_dim = recall(2, 0).obs_dim();
        let policy = policy(obs_dim, SYMBOLS, 4);
        let mut collector = MultiSyncCollector::new(
            &policy,
            vec![
                Broken {
                    inner: recall(2, 1),
                    steps: 0,
                },
                Broken {
                    inner: recall(2, 2),
                    steps: 0,
                },
            ],
            6,
            &dev(),
        )
        .unwrap();

        let failure = collector.collect();
        assert!(
            failure.is_err(),
            "a worker whose environment failed reported success"
        );
        assert!(
            format!("{}", failure.unwrap_err()).contains("fell over"),
            "the worker's own error was not the one that came back"
        );
    }

    #[test]
    fn a_ragged_pool_is_refused_and_an_uneven_one_is_not() {
        let policy_env = recall(2, 0);
        let obs_dim = policy_env.obs_dim();

        // Different observation widths cannot be concatenated into one batch.
        let ragged = ParallelEnvs::<R, f32>::new(
            vec![
                RecallEnv::new(2, SYMBOLS, HORIZON, 1, &dev()).unwrap(),
                RecallEnv::new(2, SYMBOLS + 1, HORIZON, 2, &dev()).unwrap(),
            ],
            &dev(),
        );
        assert!(
            ragged.is_err(),
            "a pool with mismatched observations was accepted"
        );
        assert!(ParallelEnvs::<R, f32>::new(Vec::<RecallEnv<R, f32>>::new(), &dev()).is_err());

        // Differing *counts* are fine: that is how a pool of unequal machines is
        // expressed, and the rows simply line up end to end.
        let uneven =
            ParallelEnvs::new(vec![recall(1, 1), recall(3, 2), recall(2, 3)], &dev()).unwrap();
        assert_eq!(uneven.envs(), 6);
        assert_eq!(uneven.widths(), vec![1, 3, 2]);
        assert_eq!(uneven.obs_dim(), obs_dim);
    }

    #[test]
    fn a_parallel_window_trains_like_any_other() {
        // The joined batch has to be usable by the same PPO and imitation paths as a
        // single-threaded one, episode masking and all.
        use mamba3::rl::PpoConfig;

        let obs_dim = recall(2, 0).obs_dim();
        let policy = policy(obs_dim, SYMBOLS, 4);
        let mut collector = MultiSyncCollector::new(
            &policy,
            vec![recall(2, 7), recall(2, 8), recall(2, 9)],
            9,
            &dev(),
        )
        .unwrap()
        .recording_expert_labels();

        let report = collector.collect().unwrap();
        let config = PpoConfig::default();
        let batch = collector.ppo_batch(&report, &config).unwrap();
        assert_eq!(batch.envs(), 6);
        assert_eq!(batch.steps(), 9);

        let imitation = collector.imitation_batch().unwrap();
        assert_eq!(imitation.envs(), 6);
        assert_eq!(imitation.expert_actions.len(), 6 * 9);

        // And the joined window replays to what the rollout did, exactly as a
        // single-threaded one does.
        let (out, _) = policy
            .forward(
                &mamba3::autograd::Var::constant(collector.buffer().observations().clone()),
                Some(&collector.buffer().reset_mask().unwrap()),
                Some(&report.initial),
            )
            .unwrap();
        let replayed = out
            .logits
            .reshape(vec![54, SYMBOLS])
            .unwrap()
            .log_softmax(1)
            .unwrap()
            .take_along_last(&collector.buffer().actions().reshape(vec![54]).unwrap())
            .unwrap()
            .to_f32();
        assert_close(
            &replayed,
            &collector.buffer().log_probs().to_f32(),
            1e-4,
            "a joined window replayed through the scan",
        );
    }
}
