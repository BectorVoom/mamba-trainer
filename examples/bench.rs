//! Where the time goes on the CPU reference backend.

use std::time::Instant;

use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{LmBatch, LmTask, TrainStep};

type R = mamba3::backends::Auto;

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let t0 = Instant::now();
    let model = Mamba3LmConfig::builder()
        .vocab_size(24)
        .d_model(32)
        .n_layers(1)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 16;
            s.d_state = 16;
            s.chunk_size = 16;
        })
        .seed(1)
        .build()?
        .init::<R, f32>(&device)?;
    println!("build            {:>10.2?}", t0.elapsed());

    let ids = IdTensor::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8], vec![1, 8], &device)?;
    let t = Instant::now();
    let _ = model.forward(&ids, false)?.to_f32();
    println!("forward #1       {:>10.2?}", t.elapsed());
    let t = Instant::now();
    for _ in 0..5 {
        let _ = model.forward(&ids, false)?.to_f32();
    }
    println!("forward (avg/5)  {:>10.2?}", t.elapsed() / 5);

    let task = LmTask::new(&model);
    let batch = LmBatch {
        inputs: ids.clone(),
        targets: ids.clone(),
    };
    let t = Instant::now();
    let loss = task.loss(&batch)?;
    let g = loss.backward()?;
    println!("fwd+bwd #1       {:>10.2?} ({} grads)", t.elapsed(), g.len());

    let cfg = TrainerConfig::builder().learning_rate(1e-3).build()?;
    let mut tr = Trainer::new(cfg, AdamW::<R, f32>::new(1e-3));
    tr.step(&task, &[batch.clone()])?;
    let t = Instant::now();
    for _ in 0..10 {
        tr.step(&task, &[batch.clone()])?;
    }
    println!("train (avg/10)   {:>10.2?}", t.elapsed() / 10);

    // Decoding cost at two context lengths, with the dispatch count that explains
    // it: at these sizes every backend is bound by launches, not arithmetic.
    model.eval();
    for context in [8usize, 128] {
        let prompt: Vec<u32> = (0..context).map(|i| (i % 24) as u32).collect();
        let mut cache = model.empty_cache(1, &device);
        let prefill = IdTensor::from_slice(&prompt, vec![1, context], &device)?;
        model.forward_cached(&prefill, &mut cache)?;
        device.synchronize();
        mamba3::backend::reset_launch_count();
        let t = Instant::now();
        for i in 0..8 {
            let step = IdTensor::from_slice(&[(i % 24) as u32], vec![1, 1], &device)?;
            model.forward_cached(&step, &mut cache)?;
        }
        device.synchronize();
        println!(
            "decode @ ctx {context:<4} {:>10.2?} per token ({} dispatches)",
            t.elapsed() / 8,
            mamba3::backend::launch_count() / 8
        );
    }
    Ok(())
}
