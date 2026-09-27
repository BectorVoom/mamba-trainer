//! Entity-model training-step profile: launches, reads and time per stage.
//!
//! One training step of the Kaggriculture spec at batch 128, in both
//! `set_fused_entity_model` modes, forward / backward / update separately,
//! with the launch tally per source line. Until a K kernel lands both modes
//! run the composed oracle and must print the same counts.
//!
//! ```text
//! cargo run --release --no-default-features --features cpu --example profile_entity_model
//! ```
//!
//! `MAMBA3_ENTITY_BATCH` overrides the batch (samples); `MAMBA3_ENTITY_STEPS`
//! overrides the timed steps per stage; `MAMBA3_ENTITY_DECODER=query_causal`
//! swaps the StepCausal decoder for QueryCausal; `MAMBA3_ENTITY_CHUNK` pins
//! the scan chunk (default: per-scan auto, `EntityModelSpec::chunk_for`).

use std::time::Instant;

use mamba3::backend::{
    launch_count, launch_tally, read_count, reset_launch_count, reset_launch_tally,
    reset_read_count, start_launch_tally, stop_launch_tally,
};
use mamba3::models::entity::model::Decode;
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout, set_fused_entity_model,
};
use mamba3::prelude::*;
use mamba3::train::TrainStep;

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// The ENTITY_MODEL_PLAN.md §1.4 reference spec.
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
        decoder: match std::env::var("MAMBA3_ENTITY_DECODER").as_deref() {
            Ok("query_causal") => DecoderMode::QueryCausal,
            _ => DecoderMode::StepCausal {
                crew_symmetric: true,
            },
        },
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 128,
            n_heads: 4,
            head_dim: 64,
            d_state: 32,
            n_groups: 1,
            ..Default::default()
        },
        chunk_size: std::env::var("MAMBA3_ENTITY_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok()),
        norm_eps: 1e-5,
        seed: 0,
    }
}

/// Deterministic pseudo-random floats in [-1, 1].
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

/// Random host arrays with mostly-valid labels.
fn random_arrays(b: usize, seed: u64) -> HostArrays {
    let (n, u, k, q) = (100usize, 20, 3, 60);
    let mut s = seed.max(1);
    let mut ri = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    let mut anchor = vec![-1i64; b * u];
    let mut tgt = vec![-1i64; b * q];
    let mut op = vec![-1i64; b * q];
    let mut crop = vec![-1i64; b * q];
    let mut opset = vec![0.0f32; b * q * 13];
    let mut eta = vec![f32::NAN; b * u];
    for bi in 0..b {
        for uu in 0..u {
            if ri(5) > 0 {
                anchor[bi * u + uu] = ri(n) as i64;
                eta[bi * u + uu] = ri(20) as f32;
            }
            for j in 0..k {
                let f = bi * q + uu * k + j;
                let r = ri(10);
                if r < 7 {
                    tgt[f] = ri(n) as i64;
                    op[f] = ri(13) as i64;
                    crop[f] = ri(5) as i64;
                    opset[f * 13 + ri(13)] = 1.0;
                } else if r < 8 {
                    tgt[f] = n as i64;
                }
            }
        }
    }
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![b, n, 48], frand(b * n * 48, seed + 1));
    a.insert_f32("globals", vec![b, 114], frand(b * 114, seed + 2));
    a.insert_f32("units", vec![b, u, 36], frand(b * u * 36, seed + 3));
    a.insert_int("units.anchor", vec![b, u], anchor);
    a.insert_int("label.target", vec![b, u, k], tgt);
    a.insert_int("label.op", vec![b, u, k], op);
    a.insert_f32("label.opset", vec![b, u, k, 13], opset);
    a.insert_int("label.crop", vec![b, u, k], crop);
    a.insert_f32("label.eta", vec![b, u, 1], eta);
    a
}

fn profile_stage(
    label: &str,
    steps: usize,
    device: &Device<R>,
    mut body: impl FnMut() -> Result<()>,
) -> Result<()> {
    for _ in 0..steps.min(2) {
        body()?;
    }
    device.synchronize();
    reset_launch_count();
    reset_read_count();
    reset_launch_tally();
    start_launch_tally();
    let started = Instant::now();
    for _ in 0..steps {
        body()?;
    }
    device.synchronize();
    let ms = started.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    stop_launch_tally();
    println!(
        "{label:<28} {:>8} launches {:>4} reads {:>10.1} ms/step",
        launch_count(),
        read_count(),
        ms
    );
    for (site, count) in launch_tally().into_iter().take(15) {
        println!("    {count:>8}  {site}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let batch_size = env_usize("MAMBA3_ENTITY_BATCH", 128);
    let steps = env_usize("MAMBA3_ENTITY_STEPS", 5);
    let device = Device::<R>::default();
    println!("backend: {}", device.name());
    mamba3::tensor::ops::matmul::try_set_precision_from_env::<R>()?;

    let spec = kaggriculture_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device)?;
    let arrays = random_arrays(batch_size, 99);
    let batch = EntityBatch::<R, f32>::from_host(&spec, &arrays, &device)?;
    let task = EntityTask::new(&model);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(3e-4).build()?,
        AdamWConfig::builder()
            .learning_rate(3e-4)
            .build()
            .init::<R, f32>(),
    );
    // Warm up (compiles kernels, settles the allocator).
    for _ in 0..3 {
        trainer.step(&task, std::slice::from_ref(&batch))?;
    }
    device.synchronize();

    for fused in [false, true] {
        set_fused_entity_model(fused);
        println!(
            "\n== fused={fused} (batch {batch_size}, d={} {}+{} layers) ==",
            spec.d_model, spec.context_layers, spec.decoder_layers
        );
        profile_stage("forward", steps, &device, || {
            task.loss(&batch)?;
            Ok(())
        })?;
        profile_stage("forward+backward", steps, &device, || {
            task.loss(&batch)?.backward()?;
            Ok(())
        })?;
        profile_stage("optimizer step", steps, &device, || {
            trainer.step(&task, std::slice::from_ref(&batch))?;
            Ok(())
        })?;
        profile_stage("predict greedy", steps, &device, || {
            model.predict(&batch, Decode::Greedy, None)?;
            Ok(())
        })?;
        profile_stage("predict teacher-forced", steps, &device, || {
            model.predict(&batch, Decode::TeacherForced, None)?;
            Ok(())
        })?;
    }
    Ok(())
}
