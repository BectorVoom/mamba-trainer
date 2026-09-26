//! Parity harness for the planner on-device kernels (K1–K5).
//!
//! Each kernel task adds forward-parity, gradient-parity and `check_grad`
//! tests here, run fused against composed on the same tie-free random data.
//! Alone in its binary: the fused switch is process-global, so toggling it
//! here must not flip modes mid-run for any other test.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::planner::{PlannerBatch, TaskPlannerConfig, fused_planner};
use mamba3::models::set_fused_planner;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

#[test]
fn fused_switch_round_trips_and_keeps_the_composed_value() {
    // Until a K task lands, both modes run the composed path and must agree.
    let cfg = TaskPlannerConfig::default();
    assert!(fused_planner(), "fused planner path is the default");
    let device = dev();
    let model = cfg.init::<R, f32>(&device).unwrap();
    let host = mamba3::models::planner::HostBatch {
        turns: 1,
        tiles: vec![0.1; 100 * 48],
        glob: vec![0.2; 114],
        units: vec![0.3; 20 * 36],
        upos: vec![-1; 20],
        tgt: vec![-100; 60],
        op: vec![-100; 60],
        crop: vec![-100; 60],
        opset: vec![0; 60 * 13],
        eta: vec![-1; 20],
    };
    let batch = PlannerBatch::from_host(&cfg, &host, &device).unwrap();
    set_fused_planner(false);
    let off = model.predict(&batch).unwrap().0.to_f32();
    set_fused_planner(true);
    let on = model.predict(&batch).unwrap().0.to_f32();
    assert!(fused_planner());
    assert_eq!(off, on);
}
