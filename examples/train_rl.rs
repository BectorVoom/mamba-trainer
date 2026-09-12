//! Reinforcement learning and imitation learning, on a task that needs memory.
//!
//! The task is [`RecallEnv`]: a symbol is shown once, at the first step of an
//! episode, and the only reward comes from naming it at the last. A policy with no
//! memory cannot do better than guessing — `1/SYMBOLS` — so every point above that
//! line is the recurrent state carrying the cue across the gap, and nothing else.
//!
//! The example runs the two families side by side on identical architectures, which
//! is the comparison worth seeing:
//!
//! 1. **PPO from scratch.** Nothing but the reward signal. It has to discover that
//!    the cue matters before it can learn to carry it, and the reward is sparse, so
//!    this is the slow road.
//! 2. **Imitation, then PPO.** The environment knows its own optimal action, so it
//!    is an expert that can label any state. Cloning it is cross entropy and it
//!    arrives in a handful of updates — but it never mentions reward and can never
//!    exceed the expert, so PPO takes over from where it lands.
//!
//! The same policy, the same replay through the scan and the same episode mask serve
//! both; the handover is a change of `TrainStep` and nothing else.
//!
//! ```text
//! cargo run --release --example train_rl
//! ```

use mamba3::prelude::*;
use mamba3::rl::{BehaviourCloningTask, DaggerSchedule, Mamba3Policy, PpoTask, RecallEnv};
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

const ENVS: usize = 32;
const SYMBOLS: usize = 4;
const HORIZON: usize = 4;
/// Steps per window: four whole episodes per environment, so every window holds
/// several episode boundaries and the reset masking is exercised continuously.
const WINDOW: usize = HORIZON * 4;

const CLONE_ROUNDS: usize = 12;
const PPO_ROUNDS: usize = 80;
const PPO_EPOCHS: usize = 4;

fn policy(obs_dim: usize, device: &Device<R>) -> Result<Mamba3Policy<R, f32>> {
    mamba3::rl::Mamba3PolicyConfig::new(obs_dim, SYMBOLS, 64, 2)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.head_dim = 16;
            s.n_groups = 4;
            s.d_state = 8;
            s.chunk_size = 8;
            s.conv_kernel = Some(4);
        })
        .init::<R, f32>(device)
}

fn env(seed: u64, device: &Device<R>) -> Result<RecallEnv<R, f32>> {
    RecallEnv::new(ENVS, SYMBOLS, HORIZON, seed, device)
}

fn trainer(lr: f32, clip: f32) -> Result<Trainer<R, f32, mamba3::train::AdamW<R, f32>>> {
    Ok(Trainer::new(
        TrainerConfig::builder()
            .learning_rate(lr)
            .max_grad_norm(clip)
            .build()?,
        AdamWConfig::builder()
            .learning_rate(lr)
            .build()
            .init::<R, f32>(),
    ))
}

/// Clone the environment's own expert into `policy`.
///
/// DAgger rather than plain behaviour cloning: `beta` is the expert's share of the
/// *acting* and it decays, so the states being labelled drift from the expert's own
/// distribution towards the ones the learner actually reaches. Round 0, where `beta`
/// is 1, is exactly behaviour cloning.
fn clone_expert(policy: &Mamba3Policy<R, f32>, device: &Device<R>) -> Result<()> {
    let mut env = env(11, device)?;
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(policy, ENVS, WINDOW, obs_dim, device)?
        .with_seed(3)
        .recording_expert_labels();
    let task = BehaviourCloningTask::new(policy).with_entropy_bonus(0.01);
    let schedule = DaggerSchedule::Exponential { decay: 0.7 };
    let mut trainer = trainer(3e-3, 1.0)?;

    println!(
        "{:>6}  {:>5}  {:>8}  {:>9}",
        "round", "beta", "loss", "agreement"
    );
    for round in 0..CLONE_ROUNDS {
        let beta = schedule.beta(round as u32);
        collector.collect_with_expert(&mut env, beta)?;
        let batch = collector.imitation_batch()?;
        let info = trainer.step(&task, std::slice::from_ref(&batch))?;
        if round % 3 == 0 || round + 1 == CLONE_ROUNDS {
            println!(
                "{round:>6}  {beta:>5.2}  {:>8.4}  {:>9.3}",
                info.loss,
                task.agreement(&batch)?,
            );
        }
    }
    Ok(())
}

/// Optimise the return itself, reporting it every few rounds.
fn optimise_return(
    policy: &Mamba3Policy<R, f32>,
    rounds: usize,
    device: &Device<R>,
) -> Result<Vec<f32>> {
    let mut env = env(23, device)?;
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(policy, ENVS, WINDOW, obs_dim, device)?.with_seed(5);
    let config = PpoConfig::default()
        .with_discount(0.99, 0.95)
        .with_clip(0.2)
        .with_coefficients(0.5, 0.01);
    let task = PpoTask::new(policy, config);
    let mut trainer = trainer(1e-3, 0.5)?;

    println!(
        "{:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
        "round", "return", "entropy", "kl", "clipped"
    );
    let mut history = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let report = collector.collect(&mut env)?;
        let batch = collector.ppo_batch(&report, &config)?;
        // Several epochs over the same window. This is what the importance ratio is
        // for: without it, reusing the data would optimise against a policy that has
        // already moved away from the one that collected it.
        for _ in 0..PPO_EPOCHS {
            trainer.step(&task, std::slice::from_ref(&batch))?;
        }
        // The one synchronisation in the loop, and it is here because a human is
        // reading the number — not because the algorithm needs it.
        let ret = collector.episode_return()?.to_f32()[0];
        history.push(ret);
        if round % 10 == 0 || round + 1 == rounds {
            let stats = task.stats().unwrap_or_default();
            println!(
                "{round:>6}  {ret:>8.3}  {:>8.3}  {:>8.4}  {:>8.2}",
                stats.entropy, stats.approx_kl, stats.clip_fraction,
            );
        }
    }
    Ok(history)
}

/// The return of a greedy rollout: what the policy *believes*, rather than what its
/// exploration noise happens to produce.
fn evaluate(policy: &Mamba3Policy<R, f32>, device: &Device<R>) -> Result<f32> {
    let mut env = env(99, device)?;
    let obs_dim = env.obs_dim();
    let mut collector =
        Collector::new(policy, ENVS, WINDOW, obs_dim, device)?.with_temperature(0.0);
    collector.collect(&mut env)?;
    Ok(collector.episode_return()?.to_f32()[0])
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::set_precision_from_env();
    let device = Device::<R>::default();
    let probe = env(0, &device)?;
    println!("backend: {}", device.name());
    println!(
        "{probe:?}: guessing earns {:.2} per episode, a perfect memory {:.2}\n",
        probe.chance_return(),
        probe.optimal_return(),
    );

    println!("== PPO from scratch ==");
    let from_scratch = policy(probe.obs_dim(), &device)?;
    let scratch_history = optimise_return(&from_scratch, PPO_ROUNDS, &device)?;
    let scratch_greedy = evaluate(&from_scratch, &device)?;

    println!("\n== imitation first, then PPO ==");
    let cloned = policy(probe.obs_dim(), &device)?;
    clone_expert(&cloned, &device)?;
    let after_cloning = evaluate(&cloned, &device)?;
    println!("greedy return after cloning alone: {after_cloning:.3}\n");
    let cloned_history = optimise_return(&cloned, PPO_ROUNDS / 2, &device)?;
    let cloned_greedy = evaluate(&cloned, &device)?;

    // How many rounds of *reward* each run needed to get most of the way there. The
    // gap is what an expert is worth when you happen to have one.
    let target = 0.9;
    let reached = |history: &[f32]| {
        history
            .iter()
            .position(|r| *r >= target)
            .map(|i| i.to_string())
            .unwrap_or_else(|| "never".to_string())
    };

    println!("\n== summary ==");
    println!("{:<24}  {:>8}  {:>16}", "run", "greedy", "rounds to 0.9");
    println!(
        "{:<24}  {scratch_greedy:>8.3}  {:>16}",
        "PPO from scratch",
        reached(&scratch_history)
    );
    println!(
        "{:<24}  {cloned_greedy:>8.3}  {:>16}",
        "cloned, then PPO",
        reached(&cloned_history)
    );
    println!(
        "\nguessing would earn {:.3}; a perfect memory {:.3}",
        probe.chance_return(),
        probe.optimal_return(),
    );
    Ok(())
}
