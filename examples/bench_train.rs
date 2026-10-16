//! Training throughput at a realistic model size.
//!
//! `examples/bench.rs` measures a toy model, where every backend is dispatch-bound.
//! This one measures a model big enough that arithmetic matters, and reports the
//! numbers a training run is actually judged on: milliseconds per optimizer step and
//! tokens per second. It also splits the step into forward, backward and update so
//! it is visible which of the three is worth attacking.
//!
//! Shape knobs come from the environment so a sweep is a shell loop rather than a
//! recompile:
//!
//! ```text
//! cargo run --release --no-default-features --features hip --example bench_train
//! SEQ=1024 BATCH=2 cargo run --release ... --example bench_train
//! DEVICES=2 cargo run --release --no-default-features --features cuda --example bench_train
//! ```
//!
//! `DEVICES=n` (n > 1) then times a data-parallel step over the first `n` GPUs,
//! each replica taking its own `BATCH x SEQ` batch, and reports the tokens/s
//! against the single-device step above.

use std::time::Instant;

use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{DataParallel, LmBatch, LmTask, TrainStep};

type R = mamba3::backends::Auto;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let vocab = env("VOCAB", 8192);
    let d_model = env("D_MODEL", 512);
    let n_layers = env("LAYERS", 8);
    let n_heads = env("HEADS", 16);
    let head_dim = env("HEAD_DIM", 64);
    let d_state = env("D_STATE", 64);
    let chunk = env("CHUNK", 64);
    let seq = env("SEQ", 512);
    let batch = env("BATCH", 4);
    let iters = env("ITERS", 10);
    let replicas = env("DEVICES", 1);

    mamba3::tensor::ops::matmul::try_set_precision_from_env::<R>()?;
    let device = Device::<R>::default();
    println!("backend {}", device.name());
    println!(
        "matmul  precision {:?}",
        mamba3::tensor::ops::matmul::matmul_precision()
    );
    println!(
        "model   d_model {d_model} layers {n_layers} heads {n_heads}x{head_dim} \
         state {d_state} chunk {chunk} vocab {vocab}"
    );
    println!(
        "batch   {batch} x {seq} tokens = {} tokens/step",
        batch * seq
    );

    let config = Mamba3LmConfig::builder()
        .vocab_size(vocab)
        .d_model(d_model)
        .n_layers(n_layers)
        .with_ssm(|s| {
            s.n_heads = n_heads;
            s.n_groups = n_heads;
            s.head_dim = head_dim;
            s.d_state = d_state;
            s.chunk_size = chunk;
        })
        .seed(1234)
        .build()?;
    let model = config.init::<R, f32>(&device)?;

    let params: usize = model
        .parameters()
        .iter()
        .map(|p| p.value().shape().num_elements())
        .sum();
    println!("params  {:.2} M", params as f64 / 1e6);

    let mut rng = Rng::seeded(7);
    let ids: Vec<u32> = (0..batch * seq)
        .map(|_| rng.next_index(vocab) as u32)
        .collect();
    let targets: Vec<u32> = (0..batch * seq)
        .map(|_| rng.next_index(vocab) as u32)
        .collect();
    let batch_data = LmBatch {
        // A host copy of the ids spares the embedding backward a device read.
        inputs: IdTensor::from_host(ids.clone(), vec![batch, seq], &device)?,
        targets: IdTensor::from_slice(&targets, vec![batch, seq], &device)?,
    };

    let task = LmTask::new(&model);
    let cfg = TrainerConfig::builder()
        .learning_rate(1e-3)
        .max_grad_norm(1.0)
        .build()?;
    let mut trainer = Trainer::new(cfg, AdamW::<R, f32>::new(1e-3));

    // Warm up: JIT, autotune and allocator pools all settle on the first step.
    let t = Instant::now();
    trainer.step(&task, std::slice::from_ref(&batch_data))?;
    device.synchronize();
    println!("warmup  {:>9.2?}", t.elapsed());

    // Phase split. Each phase syncs, so the sum is longer than a pipelined step;
    // it is a breakdown, not a budget.
    mamba3::backend::reset_launch_count();
    mamba3::backend::reset_read_count();
    let t = Instant::now();
    let loss = task.loss(&batch_data)?;
    device.synchronize();
    let fwd = t.elapsed();
    let fwd_launches = mamba3::backend::launch_count();
    let fwd_reads = mamba3::backend::read_count();

    mamba3::backend::reset_launch_count();
    mamba3::backend::reset_read_count();
    let t = Instant::now();
    let grads = loss.backward()?;
    device.synchronize();
    let bwd = t.elapsed();
    let bwd_launches = mamba3::backend::launch_count();
    let bwd_reads = mamba3::backend::read_count();
    drop(grads);

    println!("forward {fwd:>9.2?}  ({fwd_launches} launches, {fwd_reads} reads)");
    println!("backward{bwd:>9.2?}  ({bwd_launches} launches, {bwd_reads} reads)");

    // The number that matters. Timed per step and reported as the best and the
    // median rather than the mean: this is an integrated GPU that shares its memory
    // bandwidth with everything else on the machine, so a mean over ten steps mostly
    // measures what else was running. The best step is the one the hardware is
    // capable of; the median says how often it gets there.
    mamba3::backend::reset_launch_count();
    mamba3::backend::reset_read_count();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        trainer.step(&task, std::slice::from_ref(&batch_data))?;
        device.synchronize();
        times.push(t.elapsed());
    }
    let launches = mamba3::backend::launch_count() / iters;
    let reads = mamba3::backend::read_count() / iters;
    times.sort();
    let best = times[0];
    let median = times[times.len() / 2];
    let tokens = (batch * seq) as f64;
    println!(
        "step    best {:>9.2?}  median {:>9.2?}  ({launches} launches, {reads} reads)",
        best, median
    );
    println!(
        "tokens/s best {:>7.0}  median {:>7.0}",
        tokens / best.as_secs_f64(),
        tokens / median.as_secs_f64()
    );
    if replicas <= 1 {
        return Ok(());
    }

    // Data parallel: one replica per device, one batch each, so a step trains on
    // `replicas` times the tokens of the single-device step. The single-device
    // run's model, optimizer state and batch are freed first: replica 0 shares
    // its device.
    drop(trainer);
    drop(batch_data);
    drop(task);
    drop(model);
    let devices = Device::<R>::first_n(replicas)?;
    println!("data parallel over {devices:?}");
    let models = devices
        .iter()
        .map(|d| config.init::<R, f32>(d))
        .collect::<Result<Vec<_>>>()?;
    let tasks: Vec<_> = models.iter().map(LmTask::new).collect();
    let shards: Vec<Vec<LmBatch<R>>> = devices
        .iter()
        .map(|d| {
            Ok(vec![LmBatch {
                inputs: IdTensor::from_host(ids.clone(), vec![batch, seq], d)?,
                targets: IdTensor::from_slice(&targets, vec![batch, seq], d)?,
            }])
        })
        .collect::<Result<_>>()?;
    let cfg = TrainerConfig::builder()
        .learning_rate(1e-3)
        .max_grad_norm(1.0)
        .build()?;
    let mut parallel = DataParallel::new(
        cfg,
        (0..replicas).map(|_| AdamW::<R, f32>::new(1e-3)).collect(),
    )?;
    parallel.broadcast(&tasks)?;
    let t = Instant::now();
    parallel.step(&tasks, &shards)?;
    devices.iter().for_each(Device::synchronize);
    println!("warmup  {:>9.2?}", t.elapsed());
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        parallel.step(&tasks, &shards)?;
        devices.iter().for_each(Device::synchronize);
        times.push(t.elapsed());
    }
    times.sort();
    let (dp_best, dp_median) = (times[0], times[times.len() / 2]);
    let dp_tokens = tokens * replicas as f64;
    println!("step    best {dp_best:>9.2?}  median {dp_median:>9.2?}  ({replicas} replicas)");
    println!(
        "tokens/s best {:>7.0}  median {:>7.0}  ({:.2}x the single device's median)",
        dp_tokens / dp_best.as_secs_f64(),
        dp_tokens / dp_median.as_secs_f64(),
        (dp_tokens / dp_median.as_secs_f64()) / (tokens / median.as_secs_f64())
    );
    Ok(())
}
