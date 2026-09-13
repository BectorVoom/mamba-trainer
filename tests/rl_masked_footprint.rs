//! A masked collection loop is still flat, still never synchronises, and a
//! fused masked step is still one launch.
//!
//! `tests/rl_collect_footprint.rs` and `tests/rl_fused_footprint.rs` make those
//! claims for unmasked rollouts; this file makes them with a legal-action mask on
//! both paths. The one host read masking adds is the per-window validation when
//! the window becomes a batch, and it is counted here as exactly one.
//!
//! Alone in its binary, like its siblings: the launch and read counters are
//! process-wide.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, launch_count, read_count, reserved_bytes, reset_launch_count, reset_read_count,
};
use mamba3::rl::{Collector, GameWorld, Mamba3PolicyConfig, PpoConfig, Recall, recall_spec};

type R = mamba3::backends::Auto;

const SYMBOLS: usize = 4;
const OBS_DIM: usize = SYMBOLS + 2;
/// The unfused step's tail (`tests/rl_fused_footprint.rs`) plus what masking adds
/// to it: the game's mask kernel, `mask_logits`, and the mask column's write.
const UNFUSED_MASKED_TAIL: usize = 8 + 3;

#[test]
fn a_masked_rollout_is_flat_read_free_and_fuses_to_one_launch() {
    let device = Device::<R>::default();
    let (envs, steps) = (8usize, 8usize);
    let spec = recall_spec(SYMBOLS).with_action_mask();
    let policy = Mamba3PolicyConfig::new(OBS_DIM, SYMBOLS, 16, 2)
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
    let config = PpoConfig::default();
    let mut plain_world: GameWorld<R, f32, Recall> = GameWorld::new(envs, spec, 5, &device).unwrap();
    let mut fused_world: GameWorld<R, f32, Recall> = GameWorld::new(envs, spec, 5, &device).unwrap();
    let mut plain = Collector::new(&policy, envs, steps, OBS_DIM, &device).unwrap().with_seed(9);
    let mut fused = Collector::new(&policy, envs, steps, OBS_DIM, &device).unwrap().with_seed(9);
    let unmasked_bytes = fused.buffer().bytes();

    for _ in 0..4 {
        let report = plain.collect(&mut plain_world).unwrap();
        let _ = plain.ppo_batch(&report, &config).unwrap();
        let report = fused.collect_fused(&mut fused_world).unwrap();
        let _ = fused.ppo_batch(&report, &config).unwrap();
    }

    // The mask column is the only thing masking adds to the buffer.
    let mask_column = envs * steps * SYMBOLS * core::mem::size_of::<f32>();
    assert_eq!(fused.buffer().bytes(), unmasked_bytes + mask_column);
    assert_eq!(plain.buffer().bytes(), unmasked_bytes + mask_column);
    println!(
        "mask column: {mask_column} bytes ({envs} envs x {steps} steps x {SYMBOLS} actions), \
         on a {unmasked_bytes}-byte unmasked buffer"
    );

    const WINDOWS: usize = 8;
    let footprint = reserved_bytes(&device);

    reset_read_count();
    reset_launch_count();
    for _ in 0..WINDOWS {
        plain.collect(&mut plain_world).unwrap();
    }
    let unfused_per_window = launch_count() / WINDOWS;
    reset_launch_count();
    for _ in 0..WINDOWS {
        fused.collect_fused(&mut fused_world).unwrap();
    }
    let fused_per_window = launch_count() / WINDOWS;
    assert_eq!(read_count(), 0, "a masked collection read back to the host");
    assert_eq!(
        unfused_per_window - fused_per_window,
        (UNFUSED_MASKED_TAIL - 1) * steps,
        "a fused masked step should replace {UNFUSED_MASKED_TAIL} launches with one"
    );
    println!(
        "masked launches per {steps}-step window: {unfused_per_window} unfused, \
         {fused_per_window} fused"
    );

    // Validation is one read per window, when the window becomes a batch.
    for (what, collector) in [("unfused", &plain), ("fused", &fused)] {
        let report_steps = collector.buffer().len();
        assert_eq!(report_steps, steps);
        reset_read_count();
        let started = std::time::Instant::now();
        let batch = collector.buffer().action_mask().cloned().unwrap();
        mamba3::rl::validate_action_mask(&batch).unwrap();
        println!("{what}: mask validation read took {:?} (measured, CPU)", started.elapsed());
        assert_eq!(read_count(), 1, "{what}: mask validation is one read");
    }

    if let (Some(before), Some(after)) = (footprint, reserved_bytes(&device)) {
        assert_eq!(before, after, "masked windows reserved {} more bytes", after - before);
    }
}
