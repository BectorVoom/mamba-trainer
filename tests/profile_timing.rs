//! M0.1: the profiler's timing is honest across step counts.
//!
//! Runs the fused optimizer stage with 2, 4 and 8 timed steps at batch 8 and
//! asserts the three ms/step values agree within 15% of their mean. Needs a
//! real asynchronous backend; on `cpu` the queue drains synchronously, so the
//! test is ignored there.
//!
//! ```text
//! cargo test --release --features vulkan --test profile_timing -- --ignored
//! ```

#![cfg_attr(feature = "cpu", allow(dead_code))]

use std::time::Instant;

use mamba3::models::entity::{
    ContextSetSpec, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, set_fused_entity_model,
};
use mamba3::prelude::*;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

fn small_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 8,
        context: vec![ContextSetSpec::new("tiles", 16, 8)],
        queries: Some(QuerySetSpec::new("units", 4, 8, 2).with_anchor("tiles")),
        heads: vec![HeadSpec::pointer("target", "tiles", 0)],
        d_model: 32,
        context_layers: 1,
        decoder_layers: 1,
        decoder: mamba3::models::entity::DecoderMode::StepCausal {
            crew_symmetric: false,
        },
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            d_state: 8,
            n_groups: 1,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 0,
    }
}

fn frand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32) / (u64::MAX >> 11) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn drain(device: &Device<R>) {
    device.synchronize();
    let probe = Tensor::<R, f32>::zeros(vec![1], device);
    let _ = probe.to_data();
}

fn stage_ms(
    device: &Device<R>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    task: &EntityTask<R, f32>,
    batch: &EntityBatch<R, f32>,
    steps: usize,
) -> mamba3::error::Result<f64> {
    for _ in 0..steps.min(2) {
        trainer.step(task, std::slice::from_ref(batch))?;
    }
    drain(device);
    let started = Instant::now();
    for _ in 0..steps {
        trainer.step(task, std::slice::from_ref(batch))?;
    }
    drain(device);
    Ok(started.elapsed().as_secs_f64() * 1000.0 / steps as f64)
}

#[cfg_attr(feature = "cpu", ignore)]
#[test]
fn optimizer_ms_per_step_agrees_across_step_counts() -> Result<()> {
    set_fused_entity_model(true);
    let device = Device::<R>::default();
    let spec = small_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device)?;
    let b = 8usize;
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![b, 16, 8], frand(b * 16 * 8, 1));
    a.insert_f32("globals", vec![b, 8], frand(b * 8, 2));
    a.insert_f32("units", vec![b, 4, 8], frand(b * 4 * 8, 3));
    a.insert_int("units.anchor", vec![b, 4], vec![0i64; b * 4]);
    a.insert_int("label.target", vec![b, 4, 2], vec![0i64; b * 4 * 2]);
    let batch = EntityBatch::<R, f32>::from_host(&spec, &a, &device)?;
    let task = EntityTask::new(&model);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(3e-4).build()?,
        AdamWConfig::builder()
            .learning_rate(3e-4)
            .build()
            .init::<R, f32>(),
    );
    for _ in 0..3 {
        trainer.step(&task, std::slice::from_ref(&batch))?;
    }
    let ms: Vec<f64> = [2, 4, 8]
        .into_iter()
        .map(|s| stage_ms(&device, &mut trainer, &task, &batch, s))
        .collect::<mamba3::error::Result<Vec<_>>>()?;
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    for m in &ms {
        let rel = (m - mean).abs() / mean;
        assert!(
            rel <= 0.15,
            "ms/step depends on step count: {ms:?} (mean {mean:.1})"
        );
    }
    Ok(())
}
