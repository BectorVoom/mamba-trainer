//! Data-parallel training: replicas on several devices against one trainer.
//!
//! The machine running the suite may have one device. Two `Device::new` handles
//! of the same runtime device are two replicas all the same — separate models,
//! separate optimizers, gradients that meet on the host — so every property here
//! is checked without a second GPU. `MAMBA3_TEST_DEVICES=n` runs the replicas on
//! `n` real devices instead.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::nn::Module;
use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{DataParallel, LmBatch, LmTask, StepInfo, TrainStep};

type R = Auto;

/// Two replica devices: real ones under `MAMBA3_TEST_DEVICES`, else two handles
/// of the default device.
fn devices() -> Vec<Device<R>> {
    match std::env::var("MAMBA3_TEST_DEVICES") {
        Ok(n) => Device::<R>::first_n(n.parse().expect("MAMBA3_TEST_DEVICES is a count")).unwrap(),
        Err(_) => (0..2)
            .map(|_| Device::<R>::new(&Default::default()))
            .collect(),
    }
}

fn tiny_lm(seed: u64, device: &Device<R>) -> Mamba3Lm<R, f32> {
    Mamba3LmConfig::builder()
        .vocab_size(8)
        .d_model(16)
        .n_layers(1)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 4;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .seed(seed)
        .build()
        .unwrap()
        .init::<R, f32>(device)
        .unwrap()
}

fn batch(inputs: &[u32], targets: &[u32], device: &Device<R>) -> LmBatch<R> {
    let n = inputs.len();
    LmBatch {
        inputs: IdTensor::from_slice(inputs, vec![1, n], device).unwrap(),
        targets: IdTensor::from_slice(targets, vec![1, n], device).unwrap(),
    }
}

const DATA: [([u32; 4], [u32; 4]); 4] = [
    ([1, 2, 3, 4], [2, 3, 4, 5]),
    ([5, 6, 7, 0], [6, 7, 0, 1]),
    ([3, 1, 4, 1], [1, 4, 1, 5]),
    ([2, 7, 1, 6], [7, 1, 6, 2]),
];

fn data(i: usize, device: &Device<R>) -> LmBatch<R> {
    let (x, y) = DATA[i % DATA.len()];
    batch(&x, &y, device)
}

fn weight_bits(model: &Mamba3Lm<R, f32>) -> Vec<Vec<u32>> {
    model
        .parameters()
        .iter()
        .map(|p| p.value().to_f32().iter().map(|v| v.to_bits()).collect())
        .collect()
}

fn config(max_grad_norm: f32) -> TrainerConfig {
    TrainerConfig::builder()
        .learning_rate(1e-2)
        .max_grad_norm(max_grad_norm)
        .build()
        .unwrap()
}

fn data_parallel(n: usize, max_grad_norm: f32) -> DataParallel<R, f32, AdamW<R, f32>> {
    DataParallel::new(
        config(max_grad_norm),
        (0..n).map(|_| AdamW::<R, f32>::new(1e-2)).collect(),
    )
    .unwrap()
}

fn assert_same_info(a: &StepInfo, b: &StepInfo) {
    assert_eq!(a.step, b.step);
    assert_eq!(
        a.loss.to_bits(),
        b.loss.to_bits(),
        "loss {} vs {}",
        a.loss,
        b.loss
    );
    assert_eq!(
        a.grad_norm.to_bits(),
        b.grad_norm.to_bits(),
        "grad norm {} vs {}",
        a.grad_norm,
        b.grad_norm
    );
    assert_eq!(a.learning_rate.to_bits(), b.learning_rate.to_bits());
}

/// Replicas taking `shards` micro-batch indices for three steps, against one
/// trainer taking every shard concatenated: the same reports and weights, to
/// the bit, on every replica.
fn matches_one_trainer(shards: &[&[usize]], max_grad_norm: f32) {
    let devices = devices();
    let devices = &devices[..shards.len().min(devices.len())];
    assert_eq!(
        devices.len(),
        shards.len(),
        "not enough devices for the shards"
    );

    let single_device = Device::<R>::new(&Default::default());
    let single = tiny_lm(21, &single_device);
    let single_task = LmTask::new(&single);
    let mut trainer = Trainer::new(config(max_grad_norm), AdamW::<R, f32>::new(1e-2));

    let models: Vec<_> = devices.iter().map(|d| tiny_lm(21, d)).collect();
    let tasks: Vec<_> = models.iter().map(LmTask::new).collect();
    let mut parallel = data_parallel(devices.len(), max_grad_norm);

    for step in 0..3 {
        let all: Vec<_> = shards
            .iter()
            .flat_map(|s| s.iter())
            .map(|&i| data(i + step, &single_device))
            .collect();
        let expected = trainer.step(&single_task, &all).unwrap();

        let replica_shards: Vec<Vec<_>> = shards
            .iter()
            .zip(devices)
            .map(|(s, d)| s.iter().map(|&i| data(i + step, d)).collect())
            .collect();
        let got = parallel.step(&tasks, &replica_shards).unwrap();
        assert_same_info(&got, &expected);
    }
    let want = weight_bits(&single);
    for (r, model) in models.iter().enumerate() {
        assert!(
            weight_bits(model) == want,
            "replica {r}'s weights differ from the single trainer's"
        );
    }
}

#[test]
fn two_replicas_are_one_trainer_accumulating_both_shards() {
    matches_one_trainer(&[&[0], &[1]], 0.0);
}

#[test]
fn clipped_steps_clip_the_global_norm() {
    // A clip far below the norm: the factor must come from both shards' summed
    // gradients, or the update differs from the single trainer's.
    matches_one_trainer(&[&[0], &[1]], 0.05);
}

#[test]
fn uneven_shards_average_over_every_micro_batch() {
    matches_one_trainer(&[&[0, 2], &[1]], 0.5);
}

#[test]
fn one_replica_is_the_trainer() {
    matches_one_trainer(&[&[0, 1]], 0.5);
}

#[test]
fn broadcast_makes_differently_seeded_replicas_one_model() {
    let devices = devices();
    let models: Vec<_> = devices
        .iter()
        .enumerate()
        .map(|(r, d)| tiny_lm(r as u64 + 1, d))
        .collect();
    let tasks: Vec<_> = models.iter().map(LmTask::new).collect();
    assert!(weight_bits(&models[0]) != weight_bits(&models[1]));

    let mut parallel = data_parallel(devices.len(), 0.5);
    parallel.broadcast(&tasks).unwrap();
    for model in &models[1..] {
        assert!(weight_bits(model) == weight_bits(&models[0]));
    }
    // Each replica sees different data; the weights stay one model.
    for step in 0..3 {
        let shards: Vec<Vec<_>> = devices
            .iter()
            .enumerate()
            .map(|(r, d)| vec![data(step + r, d)])
            .collect();
        let info = parallel.step(&tasks, &shards).unwrap();
        assert!(info.loss.is_finite() && info.grad_norm > 0.0);
    }
    for model in &models[1..] {
        assert!(
            weight_bits(model) == weight_bits(&models[0]),
            "replicas drifted apart"
        );
    }
}

#[test]
fn queued_steps_report_as_stepping_does() {
    fn run(queued: bool) -> (Vec<StepInfo>, Vec<u64>, Vec<Vec<u32>>) {
        let devices = devices();
        let models: Vec<_> = devices.iter().map(|d| tiny_lm(5, d)).collect();
        let tasks: Vec<_> = models.iter().map(LmTask::new).collect();
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let log = seen.clone();
        let config = TrainerConfig::builder()
            .learning_rate(1e-2)
            .max_grad_norm(0.5)
            .log_every(2)
            .build()
            .unwrap();
        let mut parallel = DataParallel::new(
            config,
            devices.iter().map(|_| AdamW::<R, f32>::new(1e-2)).collect(),
        )
        .unwrap()
        .on_step(move |info| log.borrow_mut().push(info.step));
        let shards = |step: usize| -> Vec<Vec<LmBatch<R>>> {
            devices
                .iter()
                .enumerate()
                .map(|(r, d)| vec![data(step + r, d)])
                .collect()
        };
        let infos = if queued {
            let steps: Vec<_> = (0..4)
                .map(|s| parallel.queue_step(&tasks, &shards(s)).unwrap())
                .collect();
            assert!(seen.borrow().is_empty(), "a callback ran before the read");
            parallel.read_steps(&steps).unwrap()
        } else {
            (0..4)
                .map(|s| parallel.step(&tasks, &shards(s)).unwrap())
                .collect()
        };
        let calls = seen.borrow().clone();
        (infos, calls, weight_bits(&models[0]))
    }
    let (stepped, stepped_calls, stepped_weights) = run(false);
    let (queued, queued_calls, queued_weights) = run(true);
    for (a, b) in stepped.iter().zip(&queued) {
        assert_same_info(a, b);
    }
    assert_eq!(stepped_calls, [2, 4]);
    assert_eq!(queued_calls, stepped_calls);
    assert!(stepped_weights == queued_weights);
}

#[test]
fn fit_accumulates_rounds_and_stops_at_max_steps() {
    let devices = devices();
    let models: Vec<_> = devices.iter().map(|d| tiny_lm(8, d)).collect();
    let tasks: Vec<_> = models.iter().map(LmTask::new).collect();
    let config = TrainerConfig::builder()
        .learning_rate(1e-2)
        .grad_accumulation(2)
        .max_steps(3)
        .build()
        .unwrap();
    let mut parallel = DataParallel::new(
        config,
        devices.iter().map(|_| AdamW::<R, f32>::new(1e-2)).collect(),
    )
    .unwrap();
    let rounds = (0..20).map(|i| {
        devices
            .iter()
            .enumerate()
            .map(|(r, d)| data(i + r, d))
            .collect()
    });
    let report = parallel.fit(&tasks, rounds).unwrap();
    assert_eq!(report.steps, 3);
    assert_eq!(report.losses.len(), 3);
    assert_eq!(parallel.step_count(), 3);
    assert_eq!(parallel.optimizer(0).step_count(), 3);
    assert!(weight_bits(&models[0]) == weight_bits(&models[1]));
}

#[test]
fn refusals_leave_the_weights_alone() {
    let devices = devices();
    let models: Vec<_> = devices.iter().map(|d| tiny_lm(3, d)).collect();
    let tasks: Vec<_> = models.iter().map(LmTask::new).collect();
    let before = weight_bits(&models[0]);
    let mut parallel = data_parallel(devices.len(), 0.5);
    let shards: Vec<Vec<_>> = devices.iter().map(|d| vec![data(0, d)]).collect();

    // Fewer tasks than replicas.
    assert!(parallel.step(&tasks[..1], &shards[..1]).is_err());
    // Fewer shards than replicas.
    assert!(parallel.step(&tasks, &shards[..1]).is_err());
    // An empty shard.
    let mut empty = shards.clone();
    empty[1].clear();
    assert!(parallel.step(&tasks, &empty).is_err());
    // The same model twice is not two replicas.
    let same = [LmTask::new(&models[0]), LmTask::new(&models[0])];
    let two: Vec<Vec<_>> = (0..2).map(|_| vec![data(0, &devices[0])]).collect();
    assert!(data_parallel(2, 0.5).step(&same, &two).is_err());
    // A replica of another shape.
    let wide = Mamba3LmConfig::builder()
        .vocab_size(8)
        .d_model(32)
        .n_layers(1)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 4;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .seed(3)
        .build()
        .unwrap()
        .init::<R, f32>(&devices[1])
        .unwrap();
    let mixed = [LmTask::new(&models[0]), LmTask::new(&wide)];
    assert!(data_parallel(2, 0.5).step(&mixed, &shards[..2]).is_err());
    // No optimizers at all.
    assert!(DataParallel::<R, f32, AdamW<R, f32>>::new(config(0.5), Vec::new()).is_err());

    assert_eq!(parallel.step_count(), 0);
    assert!(weight_bits(&models[0]) == before);
    assert_eq!(tasks[0].parameters().len(), tasks[1].parameters().len());
}

#[test]
fn first_n_opens_what_the_runtime_has() {
    let available = Device::<R>::enumerate();
    assert!(!available.is_empty());
    assert!(Device::<R>::first_n(0).is_err());
    assert_eq!(Device::<R>::first_n(1).unwrap().len(), 1);
    assert_eq!(
        Device::<R>::first_n(available.len()).unwrap().len(),
        available.len()
    );
    assert!(Device::<R>::first_n(available.len() + 1).is_err());
}
