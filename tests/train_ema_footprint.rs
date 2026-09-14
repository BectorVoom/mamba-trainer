//! T1: what an exponential moving average of the weights costs a training step.
//!
//! The only test in its binary on purpose, like `rl_footprint.rs`: it reads the
//! process-wide launch and read counters, and any test running beside it would
//! add to them.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, launch_count, read_count, reserved_bytes, reset_launch_count, reset_read_count,
};
use mamba3::backends::Auto;
use mamba3::nn::Module;
use mamba3::rl::{
    Collector, Mamba3Policy, Mamba3PolicyConfig, PpoBatch, PpoConfig, PpoTask, RecallEnv, VecEnv,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::fused::ema_step;
use mamba3::train::{AdamW, AdamWConfig, Ema, EmaConfig, Trainer, TrainerConfig};

type R = Auto;

const ENVS: usize = 4;
const SYMBOLS: usize = 4;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn env() -> RecallEnv<R, f32> {
    RecallEnv::new(ENVS, SYMBOLS, 3, 5, &dev()).unwrap()
}

fn policy(seed: u64) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(env().obs_dim(), SYMBOLS, 16, 2)
        .with_seed(seed)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
            s.conv_kernel = Some(3);
        })
        .init::<R, f32>(&dev())
        .unwrap()
}

fn window(policy: &Mamba3Policy<R, f32>) -> PpoBatch<R, f32> {
    let mut env = env();
    let mut collector = Collector::new(policy, ENVS, 6, env.obs_dim(), &dev())
        .unwrap()
        .with_seed(3);
    let report = collector.collect(&mut env).unwrap();
    collector.ppo_batch(&report, &PpoConfig::default()).unwrap()
}

fn trainer() -> Trainer<R, f32, AdamW<R, f32>> {
    Trainer::new(
        TrainerConfig::builder()
            .learning_rate(1e-2)
            .build()
            .unwrap(),
        AdamWConfig::builder().learning_rate(1e-2).build().init(),
    )
}

/// Bytes held by live buffers, once the queue has drained.
fn bytes_in_use(device: &Device<R>) -> u64 {
    device.synchronize();
    device
        .client()
        .memory_usage()
        .expect("the runtime reports its memory")
        .bytes_in_use
}

/// A source, an optional average over it, and the batch it trains on.
struct Run {
    source: Mamba3Policy<R, f32>,
    shadow: Option<Mamba3Policy<R, f32>>,
    batch: PpoBatch<R, f32>,
    trainer: Trainer<R, f32, AdamW<R, f32>>,
}

impl Run {
    fn new(freeze_critic: bool) -> Self {
        let source = policy(7);
        if freeze_critic {
            source.freeze_matching(&["critic"]);
        }
        let batch = window(&source);
        Self {
            source,
            shadow: None,
            batch,
            trainer: trainer(),
        }
    }

    /// Build the shadow and attach an average over the source.
    fn attach(&mut self) {
        let shadow = self.shadow.insert(policy(8));
        let ema = Ema::new(&self.source, shadow, EmaConfig::new(0.9)).unwrap();
        let trainer = std::mem::replace(&mut self.trainer, trainer());
        self.trainer = trainer.with_ema(ema);
    }

    fn step(&mut self) {
        let task = PpoTask::new(&self.source, PpoConfig::default());
        self.trainer
            .step(&task, std::slice::from_ref(&self.batch))
            .unwrap();
    }

    /// Launches and reads of each of `steps` steps.
    fn per_step(&mut self, steps: usize) -> Vec<(usize, usize)> {
        (0..steps)
            .map(|_| {
                reset_launch_count();
                reset_read_count();
                self.step();
                (launch_count(), read_count())
            })
            .collect()
    }

    /// Trainable tensors, and the scalars in them.
    fn trainable(&self) -> (usize, usize) {
        let params = self.source.trainable_parameters();
        (params.len(), params.iter().map(|p| p.numel()).sum())
    }
}

#[test]
fn ema_footprint() {
    let device = dev();

    // R2: an empty average costs no launch.
    let empty = Tensor::<R, f32>::zeros(vec![0], &device);
    reset_launch_count();
    let out = ema_step(&empty, &empty, 0.25).unwrap();
    assert_eq!(launch_count(), 0, "an empty EMA step launched a kernel");
    assert!(out.is_empty());
    assert_eq!(out.shape().dims(), &[0]);

    // The allocator's buffer alignment, measured: the bytes one four-byte
    // tensor holds (8 on the CPU runtime, 256 on wgpu<wgsl>).
    let align = {
        let before = bytes_in_use(&device);
        let probe = Tensor::<R, f32>::zeros(vec![1], &device);
        let align = (bytes_in_use(&device) - before) as usize;
        drop(probe);
        assert!(align >= 4 && align.is_power_of_two(), "alignment {align}");
        align
    };

    // F1: building the shadow and attaching the average leaves one copy of the
    // trainable weights held, and nothing else: the shadow's own initial weights
    // are replaced, and its frozen parameters share the source's buffers.
    let attach_bytes = |freeze_critic: bool| {
        let mut run = Run::new(freeze_critic);
        for _ in 0..3 {
            run.step();
        }
        let before = bytes_in_use(&device);
        run.attach();
        let added = bytes_in_use(&device) as i64 - before as i64;
        let (tensors, scalars) = run.trainable();
        let copy = scalars * core::mem::size_of::<f32>();
        let held: usize = run
            .source
            .trainable_parameters()
            .iter()
            .map(|p| (p.numel() * core::mem::size_of::<f32>()).next_multiple_of(align))
            .sum();
        println!(
            "ema_footprint: attaching added {added} bytes for {tensors} trainable tensors of \
             {scalars} scalars ({copy} bytes of f32, {held} in {align}-byte buffers; critic \
             frozen: {freeze_critic})"
        );
        assert_eq!(
            added, held as i64,
            "attaching an EMA held other than one copy of the trainable weights"
        );
        run
    };
    attach_bytes(true);
    let mut run = attach_bytes(false);
    let (tensors, _) = run.trainable();

    // F2: the average neither grows nor synchronises. Warm up first: the first
    // steps compile kernels and grow the allocator's pools.
    for _ in 0..10 {
        run.step();
    }
    let reserved = reserved_bytes(&device);
    let with_ema = run.per_step(200);
    if let (Some(before), Some(after)) = (reserved, reserved_bytes(&device)) {
        assert_eq!(
            before,
            after,
            "200 steps with an EMA reserved {} more bytes",
            after - before
        );
    }
    let mut plain = Run::new(false);
    for _ in 0..13 {
        plain.step();
    }
    let without = plain.per_step(20);
    let reads = |steps: &[(usize, usize)]| steps.iter().map(|s| s.1).collect::<Vec<_>>();
    assert!(
        reads(&with_ema).iter().all(|&r| r == without[0].1),
        "reads per step with an EMA {:?} against {} without",
        &reads(&with_ema)[..5],
        without[0].1
    );
    assert!(reads(&without).iter().all(|&r| r == without[0].1));

    // F3: one launch per trainable parameter per step, the same every step, and
    // freezing the critic takes its parameters out of both the optimizer's and
    // the average's launches.
    let launches = |steps: &[(usize, usize)]| {
        let first = steps[0].0;
        assert!(
            steps.iter().all(|s| s.0 == first),
            "launches per step drifted: {:?}",
            steps.iter().map(|s| s.0).collect::<Vec<_>>()
        );
        first
    };
    let added = launches(&with_ema) - launches(&without);
    assert_eq!(added, tensors, "launches an EMA adds per step");

    let mut frozen_plain = Run::new(true);
    let mut frozen_ema = Run::new(true);
    frozen_ema.attach();
    for _ in 0..13 {
        frozen_plain.step();
        frozen_ema.step();
    }
    let frozen_added = launches(&frozen_ema.per_step(20)) - launches(&frozen_plain.per_step(20));
    let (frozen_tensors, _) = frozen_ema.trainable();
    assert_eq!(frozen_added, frozen_tensors);
    let critic = run
        .source
        .named_parameters()
        .iter()
        .filter(|(n, _)| n.starts_with("critic"))
        .count();
    assert!(critic > 0);
    assert_eq!(added - frozen_added, critic, "freezing the critic");
    println!(
        "ema_footprint: {added} launches added per step ({tensors} trainable tensors), \
         {frozen_added} with the critic frozen; reads per step {} either way",
        without[0].1
    );
}
