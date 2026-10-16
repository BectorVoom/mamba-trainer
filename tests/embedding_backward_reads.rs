//! The embedding backward with ids that carry their host copy: no device read.
//!
//! Its own test binary because the read counter is process-wide, and tests of
//! one binary run on parallel threads.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{LmBatch, LmTask, TrainStep};

type R = Auto;

const DATA: [([u32; 4], [u32; 4]); 2] =
    [([1, 2, 3, 4], [2, 3, 4, 5]), ([5, 6, 7, 0], [6, 7, 0, 1])];

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

#[test]
fn host_ids_spare_the_backward_its_read_and_change_nothing() {
    // Replicas only run at the same time when a backward pass never waits for
    // its device; the embedding backward's id read did, unless the ids carry
    // their host copy. Same gradients either way, to the bit.
    let device = Device::<R>::new(&Default::default());
    let model = tiny_lm(4, &device);
    let task = LmTask::new(&model);
    let (x, y) = DATA[1];
    let grads = |inputs: IdTensor<R>| {
        let batch = LmBatch {
            inputs,
            targets: IdTensor::from_slice(&y, vec![1, 4], &device).unwrap(),
        };
        let grads = task.loss(&batch).unwrap().backward().unwrap();
        task.parameters()
            .iter()
            .map(|p| {
                grads
                    .get(p.id())
                    .map(|g| g.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>()
    };
    let from_device = grads(IdTensor::from_slice(&x, vec![1, 4], &device).unwrap());
    mamba3::backend::reset_read_count();
    let batch = LmBatch {
        inputs: IdTensor::from_host(x.to_vec(), vec![1, 4], &device).unwrap(),
        targets: IdTensor::from_slice(&y, vec![1, 4], &device).unwrap(),
    };
    let loss = task.loss(&batch).unwrap();
    let _ = loss.backward().unwrap();
    assert_eq!(
        mamba3::backend::read_count(),
        0,
        "the backward pass read the device"
    );
    let from_host = grads(IdTensor::from_host(x.to_vec(), vec![1, 4], &device).unwrap());
    assert!(from_device == from_host);
}
