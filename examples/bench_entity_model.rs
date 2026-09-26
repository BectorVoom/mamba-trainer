//! Interleaved fused-vs-composed A/B for one entity-model optimizer step.
//!
//! Alternates `set_fused_entity_model(true/false)` every iteration and reports
//! the median step time per mode and the ratio. Interleaving in one process is
//! required: run-to-run wall noise is ±20%. Until a K kernel lands both modes
//! run the composed oracle and must print a ratio of ~1.
//!
//! ```text
//! cargo run --release --no-default-features --features vulkan --example bench_entity_model
//! ```
//!
//! `MAMBA3_ENTITY_BATCH` (default 128) and `MAMBA3_ENTITY_ITERS` (default
//! 100) override the batch and the iterations per mode.

use std::time::{Duration, Instant};

use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout, set_fused_entity_model,
};
use mamba3::prelude::*;

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
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

fn kaggriculture_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 114,
        context: vec![
            ContextSetSpec::new("tiles", 100, 48).with_layout(SetLayout::Grid {
                height: 10,
                width: 10,
                alternate_axes: true,
            }),
        ],
        queries: Some(
            QuerySetSpec::new("units", 20, 36, 3)
                .with_anchor("tiles")
                .with_autoregressive("target"),
        ),
        heads: vec![
            HeadSpec::pointer("target", "tiles", 1).step_weights(vec![1.0, 0.5, 0.5]),
            HeadSpec::categorical("op", 13).condition_on("target"),
            HeadSpec::multilabel("opset", 13)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::categorical("crop", 5)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::regression("eta", 1)
                .first_step_only()
                .loss_weight(0.1),
        ],
        d_model: 128,
        context_layers: 3,
        decoder_layers: 3,
        decoder: DecoderMode::StepCausal {
            crew_symmetric: true,
        },
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 128,
            n_heads: 4,
            head_dim: 64,
            d_state: 32,
            n_groups: 1,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 0,
    }
}

fn main() -> Result<()> {
    let batch_size = env_usize("MAMBA3_ENTITY_BATCH", 128);
    let iters = env_usize("MAMBA3_ENTITY_ITERS", 100);
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    let spec = kaggriculture_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device)?;
    let (n, u, k, q) = (100usize, 20, 3, 60);
    let mut a = HostArrays::new();
    a.insert_f32(
        "tiles",
        vec![batch_size, n, 48],
        frand(batch_size * n * 48, 1),
    );
    a.insert_f32("globals", vec![batch_size, 114], frand(batch_size * 114, 2));
    a.insert_f32(
        "units",
        vec![batch_size, u, 36],
        frand(batch_size * u * 36, 3),
    );
    a.insert_int("units.anchor", vec![batch_size, u], vec![0; batch_size * u]);
    a.insert_int(
        "label.target",
        vec![batch_size, u, k],
        vec![0; batch_size * q],
    );
    a.insert_int("label.op", vec![batch_size, u, k], vec![0; batch_size * q]);
    a.insert_f32(
        "label.opset",
        vec![batch_size, u, k, 13],
        vec![0.0; batch_size * q * 13],
    );
    a.insert_int(
        "label.crop",
        vec![batch_size, u, k],
        vec![0; batch_size * q],
    );
    a.insert_f32(
        "label.eta",
        vec![batch_size, u, 1],
        vec![3.0; batch_size * u],
    );
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

    let mut fused_times: Vec<Duration> = Vec::with_capacity(iters);
    let mut composed_times: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        for (fused, sink) in [(false, &mut composed_times), (true, &mut fused_times)] {
            set_fused_entity_model(fused);
            let started = Instant::now();
            trainer.step(&task, std::slice::from_ref(&batch))?;
            device.synchronize();
            sink.push(started.elapsed());
        }
    }
    let median = |mut v: Vec<Duration>| {
        v.sort_unstable();
        v[v.len() / 2].as_secs_f64() * 1000.0
    };
    let (fused, composed) = (median(fused_times), median(composed_times));
    println!("batch {batch_size}: fused {fused:.1} ms/step, composed {composed:.1} ms/step");
    println!("ratio fused/composed: {:.3}", fused / composed);
    Ok(())
}
