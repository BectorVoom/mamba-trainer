//! K8 launch pins for `silu_split`.
//!
//! Alone in its binary: the launch counter is process-wide, and a test running
//! beside these would add to it. Both tests hold the lock for their whole
//! body so they do not interleave with each other either.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::prelude::*;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type V = Var<R, f32>;

/// The launch counter is process-wide: tests that measure it hold this for
/// their whole body.
static COUNT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn dev() -> Device<R> {
    Device::<R>::default()
}

#[test]
fn silu_split_round_trip_is_two_op_launches() {
    let _guard = COUNT_LOCK.lock().unwrap();
    let data: Vec<f32> = (0..60).map(|i| (i as f32) * 0.23 - 6.0).collect();
    let shape = vec![2, 3, 10];
    let widths = [4usize, 3, 3];

    let round_trip = |fused: bool| -> usize {
        let x = V::traced(Tensor::from_f32(&data, shape.clone(), &dev()).unwrap());
        reset_launch_count();
        let pieces: Vec<V> = if fused {
            x.silu_split(&widths).unwrap()
        } else {
            x.silu_split_composed(&widths).unwrap()
        };
        if fused {
            assert_eq!(launch_count(), 1, "silu_split forward must be one launch");
        }
        let loss = pieces
            .iter()
            .map(|p| p.sum().unwrap())
            .reduce(|a, b| a.add(&b).unwrap())
            .unwrap();
        let _ = loss.backward_retain().unwrap();
        launch_count()
    };

    let fused_total = round_trip(true);
    let composed_total = round_trip(false);
    assert!(
        fused_total < composed_total,
        "fused round trip ({fused_total}) must beat composed ({composed_total})"
    );
}

#[test]
fn fused_mixer_saves_silu_split_launches() {
    use mamba3::models::mamba3::set_fused_silu_split;
    use mamba3::models::{Mamba3Mixer, Mamba3MixerConfig};
    use mamba3::ssm::config::SsmConfig;

    let _guard = COUNT_LOCK.lock().unwrap();
    let ssm = SsmConfig {
        d_model: 16,
        n_heads: 2,
        n_groups: 1,
        head_dim: 8,
        d_state: 8,
        chunk_size: 8,
        ..SsmConfig::default()
    };
    let run = |fused: bool| -> usize {
        set_fused_silu_split(fused);
        let device = dev();
        let mut rng = Rng::seeded(11);
        let mixer: Mamba3Mixer<R, f32> =
            Mamba3MixerConfig::new(ssm.clone()).init(&device, &mut rng).unwrap();
        let data: Vec<f32> = (0..2 * 5 * 16).map(|i| (i as f32 * 0.11).sin()).collect();
        let input = V::traced(Tensor::from_f32(&data, vec![2, 5, 16], &device).unwrap());
        reset_launch_count();
        let loss = mixer.apply(&input).unwrap().sum().unwrap();
        let _ = loss.backward_retain().unwrap();
        launch_count()
    };
    let fused_launches = run(true);
    let composed_launches = run(false);
    set_fused_silu_split(true);
    assert!(
        fused_launches + 2 <= composed_launches,
        "fused mixer ({fused_launches}) must save at least silu+split forward and back vs composed ({composed_launches})"
    );
}
