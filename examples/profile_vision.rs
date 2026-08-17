//! Kernel launches and time per training step for the vision model.
//!
//! The number that matters at small patch counts is launches/step: at 64 patches
//! the work per kernel is tiny and dispatch dominates, so the fused bidirectional
//! path lives or dies by this count. Forward-only is printed as the floor — a
//! bidirectional block should cost barely more launches than one direction.
//!
//! ```text
//! cargo run --release --example profile_vision
//! ```

use std::time::Instant;

use mamba3::models::vision::{Pooling, ScanDirection};
use mamba3::prelude::*;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{ClassificationTask, ImageBatch, TrainStep};

type R = mamba3::backends::Auto;

const IMAGE: usize = 32;
const PATCH: usize = 4;

fn batch(device: &Device<R>, b: usize) -> ImageBatch<R, f32> {
    let mut rng = Rng::seeded(5);
    let n = b * 3 * IMAGE * IMAGE;
    let pixels: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let labels: Vec<u32> = (0..b).map(|_| rng.next_index(10) as u32).collect();
    ImageBatch {
        images: Tensor::from_f32(&pixels, vec![b, 3, IMAGE, IMAGE], device).unwrap(),
        labels: IdTensor::from_slice(&labels, vec![b], device).unwrap(),
    }
}

fn config(direction: ScanDirection, chunk: usize) -> VisionMamba3Config {
    VisionMamba3Config::builder()
        .image_size(IMAGE)
        .patch_size(PATCH)
        .in_channels(3)
        .num_classes(10)
        .d_model(128)
        .n_layers(4)
        .direction(direction)
        .pooling(Pooling::Mean)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.n_groups = 4;
            s.head_dim = 32;
            s.d_state = 32;
            s.chunk_size = chunk;
        })
        .seed(1234)
        .build()
        .unwrap()
}

fn profile<T: TrainStep<R, f32, Batch = ImageBatch<R, f32>>>(
    name: &str,
    task: &T,
    device: &Device<R>,
) {
    let data = batch(device, 2);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-4).build().unwrap(),
        AdamW::<R, f32>::new(1e-4),
    );
    trainer.step(task, std::slice::from_ref(&data)).unwrap();
    device.synchronize();
    mamba3::backend::reset_launch_count();
    let started = Instant::now();
    let iters = 5;
    for _ in 0..iters {
        trainer.step(task, std::slice::from_ref(&data)).unwrap();
    }
    device.synchronize();
    let elapsed = started.elapsed() / iters;
    let launches = mamba3::backend::launch_count() / iters as usize;
    println!("{name:<28} {elapsed:>10.2?}/step  {launches:>5} launches/step");
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    for (label, dir, chunk) in [
        ("bidir chunk16", ScanDirection::Bidirectional, 16),
        ("bidir chunk32", ScanDirection::Bidirectional, 32),
        ("bidir chunk64", ScanDirection::Bidirectional, 64),
        ("forward-only chunk16", ScanDirection::Forward, 16),
    ] {
        let model = config(dir, chunk).init::<R, f32>(&device)?;
        let task = ClassificationTask::new(&model);
        profile(label, &task, &device);
    }
    Ok(())
}
