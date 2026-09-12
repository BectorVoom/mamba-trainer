//! A collection loop is flat, and it never synchronises.
//!
//! `tests/rl_footprint.rs` makes this claim about the rollout *step*. This file
//! makes it about the whole loop that wraps one: the environment transition, the
//! action draw, the log-probability, the write into the trajectory buffer and the
//! advantage estimate on top. Those are the four places a reinforcement learning
//! implementation normally reaches for the host, and if any of them did, the
//! environment and the device could never overlap — the loop would spend its life
//! draining the queue instead of filling it.
//!
//! Alone in its binary for the same reason as its sibling: the launch and read
//! counters are process-wide, and any test running beside this one would contribute
//! to them.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, launch_count, read_count, reserved_bytes, reset_launch_count, reset_read_count,
};
use mamba3::backends::Auto;
use mamba3::rl::{Collector, Mamba3PolicyConfig, MultiSyncCollector, PpoConfig, RecallEnv, VecEnv};

type R = Auto;

#[test]
fn a_collection_loop_neither_grows_nor_synchronises() {
    let device = Device::<R>::default();
    let (envs, steps, symbols, horizon) = (8usize, 8usize, 4usize, 4usize);
    let mut env = RecallEnv::<R, f32>::new(envs, symbols, horizon, 3, &device).unwrap();
    let obs_dim = env.obs_dim();
    let policy = Mamba3PolicyConfig::new(obs_dim, symbols, 16, 2)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
            s.conv_kernel = Some(3);
        })
        .init::<R, f32>(&device)
        .unwrap();

    let mut collector = Collector::new(&policy, envs, steps, obs_dim, &device)
        .unwrap()
        .with_seed(1)
        .recording_expert_labels();
    let config = PpoConfig::default();

    // Warm up: the first windows compile kernels and grow the allocator's pools to
    // the size one window needs. What matters is that they then stop growing.
    for _ in 0..4 {
        let report = collector.collect(&mut env).unwrap();
        let _ = collector.ppo_batch(&report, &config).unwrap();
    }

    let footprint = reserved_bytes(&device);
    let buffer_bytes = collector.buffer().bytes();
    reset_read_count();
    reset_launch_count();

    const WINDOWS: usize = 12;
    let collect_windows = |collector: &mut Collector<'_, R, f32>, env: &mut RecallEnv<R, f32>| {
        for _ in 0..WINDOWS {
            let report = collector.collect(env).unwrap();
            // Building the learning batch is part of the loop: the advantage
            // estimate, the reset mask and the λ-returns are all on the device too.
            let _ = collector.ppo_batch(&report, &config).unwrap();
            let _ = collector.imitation_batch().unwrap();
        }
    };

    collect_windows(&mut collector, &mut env);
    let per_window = launch_count() / WINDOWS;

    assert_eq!(
        read_count(),
        0,
        "collecting read back to the host; sampling an action, scoring it, recording \
         it and estimating its advantage are all supposed to stay on the device"
    );
    assert_eq!(
        buffer_bytes,
        collector.buffer().bytes(),
        "the trajectory buffer changed size mid-run"
    );

    if let (Some(before), Some(after)) = (footprint, reserved_bytes(&device)) {
        assert_eq!(
            before,
            after,
            "{WINDOWS} windows reserved {} more bytes; at that rate a 12,000-window \
             run would reserve {} more",
            after - before,
            (after - before) * 1_000,
        );
    }

    // Every window costs the same, so a long run costs its length times one.
    reset_launch_count();
    collect_windows(&mut collector, &mut env);
    assert_eq!(
        launch_count() / WINDOWS,
        per_window,
        "per-window launch count drifted from {per_window}"
    );

    // DAgger's mixture rides inside the same loop rather than beside it: one extra
    // kernel for the coin flip, and no read to decide it.
    collector.collect_with_expert(&mut env, 0.5).unwrap();
    assert_eq!(read_count(), 0, "a DAgger rollout synchronised");

    // Moving the environments onto worker threads must not change any of that. The
    // fan-out splits actions and joins observations with kernels, not with a round
    // trip through the host, so the joined loop is as quiet as the single one.
    let workers: Vec<RecallEnv<R, f32>> = (0..4)
        .map(|w| RecallEnv::new(envs / 4, symbols, horizon, 40 + w, &device).unwrap())
        .collect();
    let mut parallel = MultiSyncCollector::new(&policy, workers, steps, &device)
        .unwrap()
        .with_seed(2)
        .recording_expert_labels();
    for _ in 0..4 {
        let report = parallel.collect().unwrap();
        let _ = parallel.ppo_batch(&report, &config).unwrap();
    }

    let footprint = reserved_bytes(&device);
    reset_read_count();
    reset_launch_count();
    for _ in 0..WINDOWS {
        let report = parallel.collect().unwrap();
        let _ = parallel.ppo_batch(&report, &config).unwrap();
        let _ = parallel.imitation_batch().unwrap();
    }
    let per_window = launch_count() / WINDOWS;

    assert_eq!(
        read_count(),
        0,
        "collecting across worker threads read back to the host"
    );
    if let (Some(before), Some(after)) = (footprint, reserved_bytes(&device)) {
        assert_eq!(
            before,
            after,
            "{WINDOWS} parallel windows reserved {} more bytes",
            after - before,
        );
    }

    reset_launch_count();
    for _ in 0..WINDOWS {
        let report = parallel.collect().unwrap();
        let _ = parallel.ppo_batch(&report, &config).unwrap();
        let _ = parallel.imitation_batch().unwrap();
    }
    assert_eq!(
        launch_count() / WINDOWS,
        per_window,
        "per-window launch count drifted from {per_window} once the environments          moved onto threads"
    );
}
