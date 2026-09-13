//! Measurements behind the reliability work (FIX_PLAN K8), on whichever backend
//! this is built for. Every number printed is measured in this run; nothing is an
//! estimate.
//!
//! * `ReferencePolicy::score` per window at the exp010 shape — 208 lanes x 120
//!   steps, `d_model = 256`, 4 layers — time and device memory;
//! * the legal-action mask column's bytes and its one validation read;
//! * binary (`.m3ck`) against JSON checkpoints of that policy: size, save, load.
//!
//! ```text
//! cargo run --release --no-default-features --features cpu  --example measure_reliability
//! cargo run --release --no-default-features --features wgpu --example measure_reliability
//! ```

use std::time::{Duration, Instant};

use mamba3::backend::{Device, reserved_bytes};
use mamba3::nn::Module;
use mamba3::prelude::*;
use mamba3::rl::{
    Collector, Mamba3PolicyConfig, PpoConfig, RecallEnv, ReferencePolicy, VecEnv,
    validate_action_mask,
};
use mamba3::tensor::Tensor;
use mamba3::train::{AdamW, Checkpoint};

type R = mamba3::backends::Auto;

const LANES: usize = 208;
const STEPS: usize = 120;
const ACTIONS: usize = 37;
const REPEATS: usize = 5;

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    // --- 1. Reference scoring at the exp010 shape ------------------------------
    let mut env = RecallEnv::<R, f32>::new(LANES, ACTIONS, 16, 1, &device)?;
    let obs_dim = env.obs_dim();
    let policy = Mamba3PolicyConfig::new(obs_dim, ACTIONS, 256, 4)
        .with_seed(1)
        .init::<R, f32>(&device)?;
    let parameters = Module::<R, f32>::num_parameters(&policy);
    println!("policy: obs_dim={obs_dim} actions={ACTIONS} d_model=256 layers=4 parameters={parameters}");

    let config = PpoConfig::default();
    let mut collector = Collector::new(&policy, LANES, STEPS, obs_dim, &device)?.with_seed(2);
    let before_snapshot = reserved_bytes(&device);
    let mut reference = ReferencePolicy::snapshot(&policy, &device)?;
    device.synchronize();
    let after_snapshot = reserved_bytes(&device);

    // Warm up: compile kernels, grow the pools, and give the reference a cache.
    for _ in 0..2 {
        let report = collector.collect(&mut env)?;
        let batch = collector.ppo_batch(&report, &config)?;
        reference.score(&batch)?;
        device.synchronize();
    }
    let cache_bytes: usize = reference
        .cache()
        .map(|layers| layers.iter().map(|c| c.num_elements()).sum::<usize>())
        .unwrap_or(0)
        * core::mem::size_of::<f32>();

    let steady_before = reserved_bytes(&device);
    let mut score_times = Vec::with_capacity(REPEATS);
    let mut collect_times = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let started = Instant::now();
        let report = collector.collect(&mut env)?;
        let batch = collector.ppo_batch(&report, &config)?;
        device.synchronize();
        collect_times.push(started.elapsed());

        let started = Instant::now();
        let scores = reference.score(&batch)?;
        device.synchronize();
        score_times.push(started.elapsed());
        drop(scores);
    }
    let steady_after = reserved_bytes(&device);
    println!(
        "reference score per window ({LANES}x{STEPS}): median {:?} over {REPEATS} \
         (collect+batch median {:?})",
        median(score_times),
        median(collect_times)
    );
    match (before_snapshot, after_snapshot, steady_before, steady_after) {
        (Some(a), Some(b), Some(c), Some(d)) => println!(
            "reference memory: snapshot reserved {:+.2} MiB; carried cache {:.3} MiB; \
             reserved across {REPEATS} scored windows {:+.2} MiB (pool total {:.1} MiB)",
            mib(b) as f64 - mib(a),
            mib(cache_bytes as u64),
            mib(d) - mib(c),
            mib(d),
        ),
        _ => println!(
            "reference memory: this runtime reports no reserved bytes; carried cache {:.3} MiB",
            mib(cache_bytes as u64)
        ),
    }

    // --- 2. The mask column and its validation read ----------------------------
    let mask = Tensor::<R, f32>::ones(vec![LANES, STEPS, ACTIONS], &device);
    device.synchronize();
    validate_action_mask(&mask)?; // compile
    let mut validation = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        device.synchronize();
        let started = Instant::now();
        validate_action_mask(&mask)?;
        validation.push(started.elapsed());
    }
    println!(
        "mask column [{LANES}, {STEPS}, {ACTIONS}]: {} bytes ({:.2} MiB); validation read median {:?}",
        mask.len() * core::mem::size_of::<f32>(),
        mib((mask.len() * core::mem::size_of::<f32>()) as u64),
        median(validation)
    );

    // --- 3. Checkpoints: binary against JSON ------------------------------------
    let optimizer = AdamW::<R, f32>::new(1e-3);
    let dir = std::env::temp_dir().join(format!("mamba3-measure-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let checkpoint = Checkpoint::capture(&policy, 0).with_optimizer(&policy, &optimizer);
    for extension in ["json", "m3ck"] {
        let path = dir.join(format!("policy.{extension}"));
        let mut saves = Vec::new();
        let mut loads = Vec::new();
        for _ in 0..3 {
            let started = Instant::now();
            checkpoint.save(&path)?;
            saves.push(started.elapsed());
            let started = Instant::now();
            let loaded = Checkpoint::load(&path)?;
            loads.push(started.elapsed());
            assert_eq!(loaded.state.entries.len(), checkpoint.state.entries.len());
        }
        let size = std::fs::metadata(&path)?.len();
        println!(
            "checkpoint .{extension}: {size} bytes ({:.2} MiB) for {parameters} parameters; \
             save median {:?}, load median {:?} (weights only; this optimizer has taken no step)",
            mib(size),
            median(saves),
            median(loads)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
