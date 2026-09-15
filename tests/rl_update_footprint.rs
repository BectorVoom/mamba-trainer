//! What a PPO update synchronises.
//!
//! The only test in its binary on purpose, like `rl_footprint.rs`: it reads the
//! process-wide read counter, and any test running beside it would add to it.
//!
//! A read is a fixed wait for the device (~1.4 ms on wgpu, whatever its size), and
//! an update of `epochs * minibatches` optimizer steps used to pay two per step, one
//! more per minibatch cut and six for the diagnostics — 54 for four epochs over four
//! minibatches. `PpoLearner.update()` now pays one, and this pins the pieces it is
//! built from.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, read_count, reset_read_count};
use mamba3::backends::Auto;
use mamba3::rl::{
    Collector, Mamba3Policy, Mamba3PolicyConfig, PpoConfig, PpoTask, RecallEnv, VecEnv,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::read_all;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = Auto;

const ENVS: usize = 4;
const SYMBOLS: usize = 4;
const EPOCHS: usize = 3;
const MINIBATCHES: usize = 2;

#[test]
fn a_ppo_update_synchronises_once_however_many_steps_it_takes() {
    let device = Device::<R>::default();
    let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, 3, 5, &device).unwrap();
    let policy: Mamba3Policy<R, f32> = Mamba3PolicyConfig::new(env.obs_dim(), SYMBOLS, 16, 1)
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
    let mut collector = Collector::new(&policy, ENVS, 6, env.obs_dim(), &device)
        .unwrap()
        .with_seed(3);
    let config = PpoConfig::default();
    let task = PpoTask::new(&policy, config);
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(1e-2)
            .build()
            .unwrap(),
        AdamWConfig::builder().learning_rate(1e-2).build().init(),
    );

    reset_read_count();
    let report = collector.collect(&mut env).unwrap();
    let batch = collector.ppo_batch(&report, &config).unwrap();
    let per = ENVS / MINIBATCHES;
    let micros: Vec<_> = (0..MINIBATCHES)
        .map(|i| batch.minibatch(i * per, per).unwrap())
        .collect();
    let mut queued = Vec::new();
    for _ in 0..EPOCHS {
        for micro in &micros {
            queued.push(
                trainer
                    .queue_step(&task, std::slice::from_ref(micro))
                    .unwrap(),
            );
        }
    }
    assert_eq!(
        read_count(),
        0,
        "collecting, batching, cutting minibatches or queueing steps read back"
    );

    let diagnostics = task.stat_tensors().expect("a loss was taken");
    let mut scalars: Vec<&Tensor<R, f32>> = queued.iter().flat_map(|q| q.scalars()).collect();
    let steps = scalars.len();
    scalars.extend(&diagnostics);
    let (_, values) = read_all(&[], &scalars).unwrap();
    let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
    let infos = trainer.report_steps(&queued, &values[..steps]);
    assert_eq!(
        read_count(),
        1,
        "the update's numbers took more than one read"
    );
    assert_eq!(infos.len(), EPOCHS * MINIBATCHES);
    assert_eq!(infos.last().unwrap().step, (EPOCHS * MINIBATCHES) as u64);
    assert!(
        infos
            .iter()
            .all(|i| i.loss.is_finite() && i.grad_norm > 0.0)
    );

    reset_read_count();
    let stats = task.stats().expect("a loss was taken");
    assert_eq!(
        read_count(),
        1,
        "the six diagnostics took more than one read"
    );
    assert_eq!(stats.policy_loss.to_bits(), values[steps].to_bits());
    assert_eq!(stats.reference_kl.to_bits(), values[steps + 5].to_bits());

    reset_read_count();
    let queued = trainer
        .queue_step(&task, std::slice::from_ref(&batch))
        .unwrap();
    trainer.read_steps(std::slice::from_ref(&queued)).unwrap();
    assert_eq!(
        read_count(),
        1,
        "a step's loss and norm took more than one read"
    );
}
