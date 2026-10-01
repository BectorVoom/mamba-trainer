//! GP1: the optimizer's small tables stay on the device.
//!
//! Alone in its binary, and its tests share a lock: the upload counter is
//! process-wide.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, clear_meta_cache, meta_cache_devices, reset_upload_count, upload_count,
};
use mamba3::backends::Auto;
use mamba3::models::lm::{Mamba3Lm, Mamba3LmConfig};
use mamba3::nn::Param;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{AdamW, AdamWConfig, LmBatch, LmTask, Optimizer, Trainer, TrainerConfig};

type R = Auto;

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Parameters of several shapes, so the multi-tensor update has a table to
/// describe, and a gradient for each.
fn problem(device: &Device<R>) -> (Vec<Param<R, f32>>, mamba3::autograd::Grads<R, f32>) {
    let shapes: [&[usize]; 5] = [&[4], &[3, 5], &[7], &[2, 2], &[6, 4]];
    let params: Vec<Param<R, f32>> = shapes
        .iter()
        .map(|shape| Param::new(Tensor::<R, f32>::ones(shape.to_vec(), device)))
        .collect();
    let mut grads = mamba3::autograd::Grads::<R, f32>::default();
    for param in &params {
        grads
            .accumulate(param.id(), Tensor::full(param.shape(), 0.25, device))
            .unwrap();
    }
    (params, grads)
}

#[test]
fn an_optimizer_step_uploads_nothing_once_warm() {
    let _guard = lock();
    let device = dev();
    let (params, grads) = problem(&device);
    let mut optimizer = AdamWConfig::builder()
        .learning_rate(0.1)
        .weight_decay(0.5)
        .build()
        .init::<R, f32>();
    for _ in 0..2 {
        optimizer.step(&params, &grads).unwrap();
    }
    reset_upload_count();
    for _ in 0..3 {
        optimizer.step(&params, &grads).unwrap();
    }
    assert_eq!(upload_count(), 0, "the length and decay tables are kept on the device");
}

#[test]
fn a_changed_decay_policy_is_a_different_table() {
    let _guard = lock();
    let device = dev();
    // The same shapes under two decay policies: the length tables are equal,
    // the decay tables are not, and each step must read its own.
    let run = |matrices_only: bool| -> Vec<Vec<f32>> {
        let (params, _) = problem(&device);
        let mut grads = mamba3::autograd::Grads::<R, f32>::default();
        for param in &params {
            grads
                .accumulate(param.id(), Tensor::zeros(param.shape(), &device))
                .unwrap();
        }
        let mut optimizer = AdamWConfig::builder()
            .learning_rate(0.1)
            .weight_decay(1.0)
            .decay_matrices_only(matrices_only)
            .build()
            .init::<R, f32>();
        optimizer.step(&params, &grads).unwrap();
        params.iter().map(|p| p.value().to_f32()).collect()
    };
    for round in 0..2 {
        let selective = run(true);
        let everything = run(false);
        // Zero gradients: only the decay moves a weight, by lr · decay = 0.1.
        for (index, values) in selective.iter().enumerate() {
            let vector = index == 0 || index == 2;
            let want = if vector { 1.0 } else { 0.9 };
            assert!(
                values.iter().all(|v| (v - want).abs() < 1e-6),
                "round {round}, matrices only, parameter {index}: {values:?}"
            );
        }
        for (index, values) in everything.iter().enumerate() {
            assert!(
                values.iter().all(|v| (v - 0.9).abs() < 1e-6),
                "round {round}, every parameter decayed, parameter {index}: {values:?}"
            );
        }
    }
}

#[test]
fn a_language_model_step_uploads_only_its_embedding_index() {
    let _guard = lock();
    let device = dev();
    let model: Mamba3Lm<R, f32> = Mamba3LmConfig::builder()
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
        .seed(1)
        .build()
        .unwrap()
        .init(&device)
        .unwrap();
    let batch = LmBatch {
        inputs: IdTensor::from_slice(&[1, 2, 3, 4, 5, 6, 7, 0], vec![1, 8], &device).unwrap(),
        targets: IdTensor::from_slice(&[2, 3, 4, 5, 6, 7, 0, 1], vec![1, 8], &device).unwrap(),
    };
    let task = LmTask::new(&model);
    let mut trainer = Trainer::new(TrainerConfig::default(), AdamW::<R, f32>::new(1e-3));
    for _ in 0..2 {
        let queued = trainer.queue_step(&task, std::slice::from_ref(&batch)).unwrap();
        trainer.read_steps(&[queued]).unwrap();
    }
    reset_upload_count();
    let queued = trainer.queue_step(&task, std::slice::from_ref(&batch)).unwrap();
    // The embedding's gradient is accumulated through a bucket index built on
    // the host (`scatter_add_rows`): its three tables are the only uploads
    // left in the step. The optimizer and the scan add none.
    assert_eq!(upload_count(), 3, "only the embedding's scatter index is uploaded");
    trainer.read_steps(&[queued]).unwrap();
}

/// A broadcast whose shape table is a new entry of the shape-table cache.
fn broadcast(columns: usize, device: &Device<R>) -> Vec<f32> {
    let wide = Tensor::<R, f32>::ones(vec![2, columns], device);
    let row = Tensor::<R, f32>::ones(vec![columns], device);
    mamba3::tensor::ops::elemwise::add(&wide, &row).unwrap().to_f32()
}

/// The optimizer's tables have a budget of their own: a run that meets more
/// shapes than the shape-table cache holds does not evict them.
#[test]
fn shape_tables_do_not_evict_the_optimizers() {
    let _guard = lock();
    let device = dev();
    let (params, grads) = problem(&device);
    let mut optimizer = AdamW::<R, f32>::new(1e-3);
    for _ in 0..2 {
        optimizer.step(&params, &grads).unwrap();
    }
    // More distinct shapes than the cache's limit of 512: it is cleared on the
    // way, at least once.
    for columns in 1..=600 {
        assert_eq!(broadcast(columns, &device), vec![2.0; 2 * columns]);
    }
    reset_upload_count();
    optimizer.step(&params, &grads).unwrap();
    assert_eq!(upload_count(), 0, "the optimizer's tables survived the shape tables");
}

/// A device's cached tables go when the device has: opening many devices in
/// one thread does not keep every one's tables for the life of the thread.
#[test]
fn tables_of_dropped_devices_are_released() {
    let _guard = lock();
    clear_meta_cache();
    for _ in 0..6 {
        let device = dev();
        assert_eq!(broadcast(3, &device), vec![2.0; 6]);
        // This device's entry, and at most the previous one's: it is swept
        // when the next new device first caches a table.
        assert!(meta_cache_devices() <= 2, "{} devices cached", meta_cache_devices());
    }
    // A device that is still alive keeps its tables across other devices.
    let kept = dev();
    broadcast(3, &kept);
    for _ in 0..3 {
        broadcast(3, &dev());
    }
    reset_upload_count();
    broadcast(3, &kept);
    assert_eq!(upload_count(), 0, "a live device's shape table is still on the device");
}
