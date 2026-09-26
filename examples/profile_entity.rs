//! Where a structured policy's dispatches go: the entity stage and pointer head.
//!
//! The baseline for `ENTITY_KERNEL_PLAN.md`. It attributes every launch to the
//! source line that issued it, for one rollout step and for one behaviour-cloning
//! optimizer step, flat and structured, at the Kaggriculture destination shape
//! (4 globals, 100 tiles × 60 features, d_model 64, 2 layers). Launch counts rather
//! than wall time, because they are deterministic and the loop is host-bound.
//!
//! ```text
//! cargo run --release --no-default-features --features wgpu --example profile_entity
//! ```
//!
//! `MAMBA3_ENTITY_ENVS` and `MAMBA3_ENTITY_STEPS` override the batch.

use mamba3::backend::{
    launch_count, launch_tally, reset_launch_count, reset_launch_tally, start_launch_tally,
    stop_launch_tally,
};
use mamba3::nn::entity::EntityEncoderConfig;
use mamba3::prelude::*;
use mamba3::rl::{
    ActionHeadConfig, BehaviourCloningTask, EntitySet, ImitationBatch, Mamba3Policy,
    Mamba3PolicyConfig, ObsSpec, PointerHeadConfig, RolloutEngine, Scoring,
};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn attribute(label: &str, body: impl FnOnce() -> Result<()>) -> Result<usize> {
    reset_launch_count();
    reset_launch_tally();
    start_launch_tally();
    body()?;
    stop_launch_tally();
    let total = launch_count();
    println!("\n{label}: {total} launches\n");
    println!("{:>8}  site", "launches");
    for (site, count) in launch_tally().into_iter().take(20) {
        println!("{count:>8}  {site}");
    }
    Ok(total)
}

fn profile(name: &str, config: Mamba3PolicyConfig, envs: usize, steps: usize) -> Result<()> {
    let device = Device::<R>::default();
    let obs_dim = config.obs_dim;
    let policy: Mamba3Policy<R, f32> = config.init::<R, f32>(&device)?;

    let obs = Var::constant(Tensor::<R, f32>::ones(vec![envs, 1, obs_dim], &device));
    let done = Tensor::<R, f32>::zeros(vec![envs], &device);
    let mut engine = RolloutEngine::new(&policy, envs, &device);
    for _ in 0..8 {
        engine.step(&obs, Some(&done))?;
    }
    device.synchronize();
    let step = attribute(&format!("[{name}] rollout step, {envs} envs"), || {
        engine.step(&obs, Some(&done))?;
        Ok(())
    })?;

    let task = BehaviourCloningTask::new(&policy);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-3).build()?,
        AdamWConfig::builder()
            .learning_rate(1e-3)
            .build()
            .init::<R, f32>(),
    );
    let labels = vec![0u32; envs * steps];
    let batch = ImitationBatch {
        observations: Tensor::ones(vec![envs, steps, obs_dim], &device),
        expert_actions: IdTensor::from_slice(&labels, vec![envs, steps], &device)?,
        reset: None,
        initial: None,
        mask: None,
        action_mask: None,
    };
    for _ in 0..3 {
        trainer.step(&task, std::slice::from_ref(&batch))?;
    }
    device.synchronize();
    let update = attribute(
        &format!("[{name}] behaviour-cloning step, {envs} × {steps}"),
        || {
            trainer.step(&task, std::slice::from_ref(&batch))?;
            Ok(())
        },
    )?;
    println!("\n[{name}] summary: rollout step {step}, optimizer step {update}");
    Ok(())
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::try_set_precision_from_env::<R>()?;
    let envs = env_usize("MAMBA3_ENTITY_ENVS", 64);
    let steps = env_usize("MAMBA3_ENTITY_STEPS", 16);
    println!("backend: {}", Device::<R>::default().name());

    let spec = ObsSpec::new(4, vec![EntitySet::new("tiles", 100, 60)]);
    let base = || Mamba3PolicyConfig::new(spec.obs_dim(), 100, 64, 2).with_seed(0);
    let structured = |scoring| {
        base()
            .with_obs_spec(spec.clone())
            .with_entity_encoder("tiles", EntityEncoderConfig::new(vec![64], 48))
            .with_action_head(ActionHeadConfig::Pointer(
                PointerHeadConfig::new("tiles")
                    .with_hidden(48)
                    .with_scoring(scoring),
            ))
    };
    profile("flat", base(), envs, steps)?;
    profile(
        "pointer additive",
        structured(Scoring::Additive),
        envs,
        steps,
    )?;
    profile("pointer dot", structured(Scoring::Dot), envs, steps)?;
    Ok(())
}
