//! A structured policy's rollout step and behaviour-cloning step, fused against
//! composed, at the Kaggriculture destination shape.
//!
//! Both modes run the same code until the fused kernels land, so today this
//! measures nothing but noise — which is the point. Later tasks plug kernels
//! in behind the switch, and the harness that judges them is already here.
//!
//! A training round is too noisy to answer whether a kernel paid for itself —
//! wall time swings ±20% run to run for identical work — so this measures the
//! two steps alone, with the modes interleaved inside one process and the
//! median taken over the repetitions. Interleaving is what makes the
//! comparison survive thermal drift; the median is what makes it survive
//! everything else running on the machine.
//!
//! ```text
//! cargo run --release --no-default-features --features wgpu --example bench_entity
//! ```
//!
//! `MAMBA3_ENTITY_ENVS` overrides the environment count (default 64),
//! `MAMBA3_BENCH_ITERS` the iterations per mode (default 200). The
//! behaviour-cloning window is `envs × 16`.

use std::time::{Duration, Instant};

use mamba3::nn::entity::{EntityEncoderConfig, set_fused_entity};
use mamba3::prelude::*;
use mamba3::rl::{
    ActionHeadConfig, BehaviourCloningTask, EntitySet, ImitationBatch, Mamba3Policy,
    Mamba3PolicyConfig, ObsSpec, PointerHeadConfig, RolloutEngine, Scoring,
};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{AdamW, AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Time one rollout step followed by one behaviour-cloning step, draining the
/// device queue on both sides so the wall time includes the device work.
///
/// The synchronisation before the clock starts matters as much as the one
/// after it: without the first, this sample would be charged for the previous
/// sample's queue.
fn sample(
    engine: &mut RolloutEngine<'_, R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    task: &BehaviourCloningTask<'_, R, f32>,
    obs: &Var<R, f32>,
    done: &Tensor<R, f32>,
    batch: &ImitationBatch<R, f32>,
    device: &Device<R>,
) -> Result<(Duration, Duration)> {
    device.synchronize();
    let started = Instant::now();
    engine.step(obs, Some(done))?;
    device.synchronize();
    let rollout = started.elapsed();

    device.synchronize();
    let started = Instant::now();
    trainer.step(task, std::slice::from_ref(batch))?;
    device.synchronize();
    let bc = started.elapsed();
    Ok((rollout, bc))
}

fn median(mut times: Vec<Duration>) -> Duration {
    times.sort_unstable();
    times[times.len() / 2]
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    let envs = env_usize("MAMBA3_ENTITY_ENVS", 64);
    let steps = 16;
    let iters = env_usize("MAMBA3_BENCH_ITERS", 200);

    let spec = ObsSpec::new(4, vec![EntitySet::new("tiles", 100, 60)]);
    let config = Mamba3PolicyConfig::new(spec.obs_dim(), 100, 64, 2)
        .with_seed(0)
        .with_obs_spec(spec)
        .with_entity_encoder("tiles", EntityEncoderConfig::new(vec![64], 48))
        .with_action_head(ActionHeadConfig::Pointer(
            PointerHeadConfig::new("tiles")
                .with_hidden(48)
                .with_scoring(Scoring::Additive),
        ));
    let obs_dim = config.obs_dim;
    let policy: Mamba3Policy<R, f32> = config.init::<R, f32>(&device)?;

    let obs = Var::constant(Tensor::<R, f32>::ones(vec![envs, 1, obs_dim], &device));
    let done = Tensor::<R, f32>::zeros(vec![envs], &device);
    let mut engine = RolloutEngine::new(&policy, envs, &device);

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

    // Warm-up compiles the kernels and settles the allocator, so neither mode
    // pays for first runs inside a timed sample.
    for _ in 0..5 {
        set_fused_entity(true);
        sample(&mut engine, &mut trainer, &task, &obs, &done, &batch, &device)?;
        set_fused_entity(false);
        sample(&mut engine, &mut trainer, &task, &obs, &done, &batch, &device)?;
    }

    let mut fused_rollout = Vec::with_capacity(iters);
    let mut fused_bc = Vec::with_capacity(iters);
    let mut composed_rollout = Vec::with_capacity(iters);
    let mut composed_bc = Vec::with_capacity(iters);
    for _ in 0..iters {
        set_fused_entity(true);
        let (r, b) = sample(&mut engine, &mut trainer, &task, &obs, &done, &batch, &device)?;
        fused_rollout.push(r);
        fused_bc.push(b);
        set_fused_entity(false);
        let (r, b) = sample(&mut engine, &mut trainer, &task, &obs, &done, &batch, &device)?;
        composed_rollout.push(r);
        composed_bc.push(b);
    }
    let fused_rollout = median(fused_rollout);
    let fused_bc = median(fused_bc);
    let composed_rollout = median(composed_rollout);
    let composed_bc = median(composed_bc);
    let each = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "rollout step ({envs} envs): fused {:>8.3}ms, composed {:>8.3}ms, ratio {:>6.3}",
        each(fused_rollout),
        each(composed_rollout),
        each(fused_rollout) / each(composed_rollout),
    );
    println!(
        "bc step ({envs} x {steps}):      fused {:>8.3}ms, composed {:>8.3}ms, ratio {:>6.3}",
        each(fused_bc),
        each(composed_bc),
        each(fused_bc) / each(composed_bc),
    );
    Ok(())
}
