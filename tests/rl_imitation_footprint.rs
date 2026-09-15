//! What a DAgger round synchronises and launches.
//!
//! The only test in its binary on purpose, like `rl_update_footprint.rs`: it reads
//! the process-wide read and launch counters, and any test running beside it would
//! add to them.
//!
//! `ImitationLearner.round()` used to read three times — the step's loss, then the
//! agreement replay's predictions and its labels separately — and a read is a fixed
//! wait for the device (~1.4 ms on wgpu). It now reads once. The mixture's own
//! scoring used to cost seven launches per collected step and now costs one; this
//! pins both.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, launch_count, read_count, reset_launch_count, reset_read_count};
use mamba3::backends::Auto;
use mamba3::rl::{
    BehaviourCloningTask, Collector, Mamba3Policy, Mamba3PolicyConfig, RecallEnv, VecEnv,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::read_all;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = Auto;

const ENVS: usize = 4;
const SYMBOLS: usize = 4;
const STEPS: usize = 6;

#[test]
fn a_dagger_round_synchronises_once_and_mixing_costs_two_launches_a_step() {
    let device = Device::<R>::default();
    let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, 3, 5, &device).unwrap();
    let obs_dim = env.obs_dim();
    let policy: Mamba3Policy<R, f32> = Mamba3PolicyConfig::new(obs_dim, SYMBOLS, 16, 1)
        .with_seed(4)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .init::<R, f32>(&device)
        .unwrap();
    let mut collector = Collector::new(&policy, ENVS, STEPS, obs_dim, &device)
        .unwrap()
        .with_seed(3)
        .recording_expert_labels();
    let task = BehaviourCloningTask::new(&policy).with_entropy_bonus(0.01);
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(1e-2)
            .build()
            .unwrap(),
        AdamWConfig::builder().learning_rate(1e-2).build().init(),
    );
    // The first window resets the environment, which later ones do not.
    collector.collect_with_expert(&mut env, 0.5).unwrap();

    // Mixing adds the coin flip and the executed action's score, one launch each,
    // to what the same window costs without an expert.
    reset_launch_count();
    collector.collect(&mut env).unwrap();
    let plain = launch_count();
    reset_launch_count();
    collector.collect_with_expert(&mut env, 0.5).unwrap();
    assert_eq!(
        launch_count(),
        plain + 2 * STEPS,
        "mixing with the expert cost {} launches over {STEPS} steps, not {}",
        launch_count() - plain,
        2 * STEPS
    );

    reset_read_count();
    let batch = collector.imitation_batch().unwrap();
    let step = trainer
        .queue_step(&task, std::slice::from_ref(&batch))
        .unwrap();
    let agreement = task.queue_agreement(&batch).unwrap();
    assert_eq!(
        read_count(),
        0,
        "batching, queueing the step or queueing the agreement read back"
    );
    let scalars: Vec<&Tensor<R, f32>> = step.scalars().chain(agreement.scalars()).collect();
    let (_, values) = read_all(&[], &scalars).unwrap();
    let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
    let (step_values, agreed) = values.split_at(step.scalars().count());
    let info = trainer.report_steps(std::slice::from_ref(&step), step_values)[0];
    let agreement = agreement.fraction(agreed);
    assert_eq!(
        read_count(),
        1,
        "the round's numbers took more than one read"
    );
    assert!(info.loss.is_finite() && info.grad_norm > 0.0);
    assert!((0.0..=1.0).contains(&agreement));

    // And on its own, agreement is one read, whether or not a mask weighs it.
    reset_read_count();
    task.agreement(&batch).unwrap();
    assert_eq!(read_count(), 1, "agreement took more than one read");
    let mask =
        Tensor::<R, f32>::from_f32(&[1.0; ENVS * STEPS], vec![ENVS, STEPS], &device).unwrap();
    reset_read_count();
    task.agreement(&batch.with_mask(mask)).unwrap();
    assert_eq!(
        read_count(),
        1,
        "a masked agreement took more than one read"
    );
}
