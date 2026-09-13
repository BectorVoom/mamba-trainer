//! The anchor: PPO priced against a policy that does not move.
//!
//! The clip bounds how far one update may take the policy from the weights that
//! collected the data. Nothing in it bounds where two hundred such updates end up,
//! so a run that starts from a good cloned policy can walk away from it a fraction
//! of a nat at a time and never once trip the trust region. `reference_coeff` and
//! `PpoBatch::reference_log_probs` are what price that distance, and these are the
//! properties they have to have:
//!
//! * off by default, and inert without a reference to compare against;
//! * zero, exactly, when the reference *is* the policy;
//! * positive otherwise, and added to the total at its stated weight;
//! * and — the one that matters — an update under it moves the policy *towards*
//!   the reference rather than away.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::prelude::*;
use mamba3::rl::{
    Collector, Mamba3Policy, Mamba3PolicyConfig, PpoBatch, PpoConfig, PpoTask, RecallEnv,
    reference_log_probs,
};
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = Auto;

const ENVS: usize = 8;
const SYMBOLS: usize = 4;
const HORIZON: usize = 4;
const WINDOW: usize = 8;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// The recall task's observation is wider than its action space; ask it rather
/// than assuming.
fn obs_dim() -> usize {
    RecallEnv::<R, f32>::new(1, SYMBOLS, HORIZON, 0, &dev())
        .expect("the recall task")
        .obs_dim()
}

fn policy(seed: u64) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(obs_dim(), SYMBOLS, 32, 1)
        .with_seed(seed)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 16;
            s.n_groups = 2;
            s.d_state = 8;
            s.chunk_size = 4;
            s.conv_kernel = Some(4);
        })
        .init::<R, f32>(&dev())
        .expect("a small policy")
}

/// One collected window, plus the collector that owns its buffer.
fn window(actor: &Mamba3Policy<R, f32>, config: &PpoConfig) -> PpoBatch<R, f32> {
    let device = dev();
    let mut env = RecallEnv::new(ENVS, SYMBOLS, HORIZON, 11, &device).expect("the recall task");
    let obs_dim = env.obs_dim();
    let mut collector =
        Collector::new(actor, ENVS, WINDOW, obs_dim, &device).expect("a collector").with_seed(3);
    let report = collector.collect(&mut env).expect("a window");
    collector.ppo_batch(&report, config).expect("a batch")
}

/// Mean log-probability the policy gives the actions the window recorded.
fn agreement(policy: &Mamba3Policy<R, f32>, batch: &PpoBatch<R, f32>) -> f32 {
    let scores = reference_log_probs(policy, batch).expect("scores");
    let values = scores.to_f32();
    values.iter().sum::<f32>() / values.len() as f32
}

#[test]
fn the_anchor_is_off_by_default() {
    assert_eq!(PpoConfig::default().reference_coeff, 0.0);

    let actor = policy(1);
    let batch = window(&actor, &PpoConfig::default());
    let task = PpoTask::new(&actor, PpoConfig::default());
    let loss = task.evaluate(&batch).expect("a loss");
    assert_eq!(
        loss.reference_kl.to_f32()[0],
        0.0,
        "with no reference there is no distance to report"
    );
}

#[test]
fn a_coefficient_without_a_reference_changes_nothing() {
    let actor = policy(2);
    let plain = PpoConfig::default();
    let batch = window(&actor, &plain);

    let without = PpoTask::new(&actor, plain).evaluate(&batch).expect("a loss");
    let anchored = PpoConfig::default().with_reference_penalty(10.0);
    let with = PpoTask::new(&actor, anchored).evaluate(&batch).expect("a loss");

    // The coefficient prices a distance; with nothing to be distant from it is inert.
    assert!(
        (without.total.tensor().to_f32()[0] - with.total.tensor().to_f32()[0]).abs() < 1e-6,
        "a reference penalty with no reference must not move the loss"
    );
}

#[test]
fn anchoring_a_policy_to_itself_costs_nothing() {
    let actor = policy(3);
    let config = PpoConfig::default().with_reference_penalty(1.0);
    let batch = window(&actor, &config);
    let scores = reference_log_probs(&actor, &batch).expect("scores");
    let anchored = batch.clone().with_reference_log_probs(scores);

    let task = PpoTask::new(&actor, config);
    let plain = task.evaluate(&batch).expect("a loss");
    let with_self = task.evaluate(&anchored).expect("a loss");

    let kl = with_self.reference_kl.to_f32()[0];
    assert!(kl.abs() < 1e-5, "a policy is not distant from itself, got {kl}");
    assert!(
        (plain.total.tensor().to_f32()[0] - with_self.total.tensor().to_f32()[0]).abs() < 1e-5,
        "and so the total is unchanged"
    );
}

#[test]
fn a_different_reference_costs_its_weight() {
    let actor = policy(4);
    let other = policy(5);
    let plain = PpoConfig::default();
    let batch = window(&actor, &plain);
    let scores = reference_log_probs(&other, &batch).expect("scores");
    let anchored = batch.clone().with_reference_log_probs(scores);

    let base = PpoTask::new(&actor, plain).evaluate(&anchored).expect("a loss");
    let coeff = 2.0;
    let priced = PpoTask::new(&actor, PpoConfig::default().with_reference_penalty(coeff))
        .evaluate(&anchored)
        .expect("a loss");

    let kl = priced.reference_kl.to_f32()[0];
    assert!(kl > 0.0, "two different policies are at a positive distance, got {kl}");
    let expected = base.total.tensor().to_f32()[0] + coeff * kl;
    let actual = priced.total.tensor().to_f32()[0];
    assert!(
        (expected - actual).abs() < 1e-4,
        "the anchor enters the total at its weight: expected {expected}, got {actual}"
    );
}

#[test]
fn an_anchored_update_moves_towards_the_reference() {
    // The learner starts where `reference` is and is pulled elsewhere by the
    // advantages; the question is whether the anchor pulls back.
    let reference = policy(6);
    let learner = policy(7);

    let plain = PpoConfig::default();
    let batch = window(&learner, &plain);
    let scores = reference_log_probs(&reference, &batch).expect("scores");
    let anchored = batch.clone().with_reference_log_probs(scores);

    let before = agreement(&learner, &anchored);

    // A weight large enough that the anchor, not the advantage, decides the step.
    let config = PpoConfig::default().with_reference_penalty(50.0);
    let task = PpoTask::new(&learner, config);
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(3e-3)
            .max_grad_norm(1.0)
            .build()
            .expect("a trainer config"),
        AdamWConfig::builder().learning_rate(3e-3).build().init::<R, f32>(),
    );
    for _ in 0..20 {
        trainer
            .step(&task, std::slice::from_ref(&anchored))
            .expect("a step");
    }

    let after = agreement(&learner, &anchored);
    assert!(
        after > before,
        "an anchored update should raise the policy's agreement with the reference: \
         {before} -> {after}"
    );
}
