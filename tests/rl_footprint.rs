//! Acceptance criterion 2: a rollout loop is flat.
//!
//! This is the only test in its binary on purpose. It reads the process-wide
//! launch and device-read counters from [`mamba3::backend`], and any test running
//! beside it would add to them — a false failure that says nothing about the
//! rollout. Cargo gives each integration test file its own process, so keeping
//! this one alone is what makes the numbers mean what they say.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{
    Device, launch_count, read_count, reserved_bytes, reset_launch_count, reset_read_count,
};
use mamba3::backends::Auto;
use mamba3::rl::{Mamba3PolicyConfig, RolloutEngine};
use mamba3::tensor::Tensor;

type R = Auto;

#[test]
fn a_rollout_neither_grows_nor_synchronises() {
    let device = Device::<R>::default();
    let (envs, obs_dim) = (8usize, 6usize);
    let policy = Mamba3PolicyConfig::new(obs_dim, 4, 16, 2)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.conv_kernel = Some(3);
        })
        .init::<R, f32>(&device)
        .unwrap();

    let mut engine = RolloutEngine::new(&policy, envs, &device);
    let obs = Var::constant(Tensor::zeros(vec![envs, 1, obs_dim], &device));
    let done = Tensor::<R, f32>::zeros(vec![envs], &device);

    // Warm up. The first steps compile kernels and grow the allocator's pools to
    // the size one step needs; what matters is that they stop growing.
    for _ in 0..25 {
        engine.step(&obs, Some(&done)).unwrap();
    }

    let footprint = reserved_bytes(&device);
    let state_bytes = engine.state().bytes();
    reset_read_count();
    reset_launch_count();

    for _ in 0..200 {
        engine.step(&obs, Some(&done)).unwrap();
    }
    let per_step = launch_count() / 200;

    // A rollout that synchronises cannot overlap with the environment, whatever
    // its kernel time says. Nothing here reads back.
    assert_eq!(read_count(), 0, "a rollout step read back to the host");

    // The state is the only thing that persists, and it is the size the
    // configuration fixes — not a function of how many steps have been taken.
    assert_eq!(
        state_bytes,
        engine.state().bytes(),
        "the state buffer changed size mid-rollout"
    );

    if let (Some(before), Some(after)) = (footprint, reserved_bytes(&device)) {
        assert_eq!(
            before,
            after,
            "200 steps reserved {} more bytes on the device; at that rate \
             1,000,000 steps would reserve {} more",
            after - before,
            (after - before) * 5_000,
        );
    }

    // Every step issues the same launches, so a million of them cost a million
    // times one. This is the property the whole design is for.
    reset_launch_count();
    for _ in 0..50 {
        engine.step(&obs, Some(&done)).unwrap();
    }
    assert_eq!(
        launch_count() / 50,
        per_step,
        "per-step launch count drifted from {per_step}"
    );

    // Terminations are free: flagging every environment as done on every step
    // must not add a launch, because the reset rides inside kernels that already
    // run rather than taking passes of its own.
    let all_done = Tensor::<R, f32>::ones(vec![envs], &device);
    reset_launch_count();
    for _ in 0..50 {
        engine.step(&obs, Some(&all_done)).unwrap();
    }
    assert_eq!(
        launch_count() / 50,
        per_step,
        "resetting every environment cost extra launches"
    );
}
