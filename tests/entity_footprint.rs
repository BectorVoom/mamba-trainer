//! The entity loss reads nothing back.
//!
//! Alone in its binary for the reason `rl_footprint.rs` gives: the launch and
//! read counters are process-wide, and a test running beside this one would add
//! to them. K6 extends this file with fused launch-count pins.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout,
};
use mamba3::train::TrainStep;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn frand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32) / (u64::MAX >> 11) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn kag_joint_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 8,
        context: vec![ContextSetSpec::new("tiles", 16, 6).with_layout(SetLayout::Grid {
            height: 4,
            width: 4,
            alternate_axes: true,
        })],
        queries: Some(QuerySetSpec::new("units", 4, 5, 2).with_anchor("tiles")),
        heads: vec![
            HeadSpec::pointer("target", "tiles", 1),
            HeadSpec::categorical("op", 4).condition_on("target"),
        ],
        d_model: 32,
        context_layers: 1,
        decoder_layers: 1,
        decoder: DecoderMode::Joint,
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 32,
            d_state: 8,
            n_groups: 1,
            chunk_size: 32,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 0,
    }
}

#[test]
fn loss_and_backward_do_no_host_reads() {
    let device = dev();
    let spec = kag_joint_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let (n, u, k, q) = (16usize, 4, 2, 8);
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![1, n, 6], frand(n * 6, 1));
    a.insert_f32("globals", vec![1, 8], frand(8, 2));
    a.insert_f32("units", vec![1, u, 5], frand(u * 5, 3));
    a.insert_int("units.anchor", vec![1, u], vec![-1; u]);
    a.insert_int("label.target", vec![1, u, k], vec![-1; q]);
    a.insert_int("label.op", vec![1, u, k], vec![-1; q]);
    let b = EntityBatch::<R, f32>::from_host(&spec, &a, &device).unwrap();
    let task = EntityTask::new(&model);
    // Warm up once (lazy init), then measure.
    task.loss(&b).unwrap().backward().unwrap();
    let before = mamba3::backend::read_count();
    let loss = task.loss(&b).unwrap();
    loss.backward().unwrap();
    assert_eq!(mamba3::backend::read_count() - before, 0);
}
