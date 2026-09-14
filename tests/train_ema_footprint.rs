//! T1: what an exponential moving average of the weights costs a training step.
//!
//! The only test in its binary on purpose, like `rl_footprint.rs`: it reads the
//! process-wide launch and read counters, and any test running beside it would
//! add to them.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::fused::ema_step;

type R = Auto;

#[test]
fn ema_footprint() {
    let device = Device::<R>::default();

    // R2: an empty average costs no launch.
    let empty = Tensor::<R, f32>::zeros(vec![0], &device);
    reset_launch_count();
    let out = ema_step(&empty, &empty, 0.25).unwrap();
    assert_eq!(launch_count(), 0, "an empty EMA step launched a kernel");
    assert!(out.is_empty());
    assert_eq!(out.shape().dims(), &[0]);
}
