//! A structured policy's rollout step is flat and reads nothing back.
//!
//! Alone in its binary for the reason `rl_footprint.rs` gives: the launch and
//! read counters are process-wide, and a test running beside this one would add
//! to them.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, launch_count, read_count, reset_launch_count, reset_read_count};
use mamba3::backends::Auto;
use mamba3::nn::entity::EntityEncoderConfig;
use mamba3::rl::{
    ActionHeadConfig, EntitySet, Mamba3PolicyConfig, ObsSpec, PointerHeadConfig, RolloutEngine,
};
use mamba3::tensor::Tensor;

type R = Auto;

#[test]
fn a_structured_rollout_step_neither_reads_back_nor_grows() {
    let device = Device::<R>::default();
    let envs = 8;
    let spec = ObsSpec::new(3, vec![EntitySet::new("tiles", 10, 4)]);
    let obs_dim = spec.obs_dim();
    let policy = Mamba3PolicyConfig::new(obs_dim, 11, 16, 2)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.conv_kernel = Some(3);
        })
        .with_obs_spec(spec)
        .with_entity_encoder("tiles", EntityEncoderConfig::new(vec![8], 8))
        .with_action_head(ActionHeadConfig::Pointer(
            PointerHeadConfig::new("tiles").with_extra_actions(1),
        ))
        .init::<R, f32>(&device)
        .unwrap();

    let mut engine = RolloutEngine::new(&policy, envs, &device);
    let obs = Var::constant(Tensor::ones(vec![envs, 1, obs_dim], &device));
    let done = Tensor::<R, f32>::zeros(vec![envs], &device);
    for _ in 0..10 {
        engine.step(&obs, Some(&done)).unwrap();
    }

    reset_read_count();
    reset_launch_count();
    engine.step(&obs, Some(&done)).unwrap();
    let first = launch_count();
    for _ in 0..49 {
        engine.step(&obs, Some(&done)).unwrap();
    }
    assert_eq!(read_count(), 0, "a structured rollout step read back to the host");
    assert_eq!(
        launch_count(),
        first * 50,
        "the launch count of a step drifted: {first} on the first, {} over 50",
        launch_count()
    );
}
