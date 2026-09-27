//! The entity loss reads nothing back, and the fused launch count per train
//! step is pinned.
//!
//! Alone in its binary for the reason `rl_footprint.rs` gives: the launch and
//! read counters are process-wide, and a test running beside this one would add
//! to them. All tests here run the fused path (the default); K6 pins the
//! fused counts.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, launch_count, read_count, reset_launch_count, reset_read_count};
use mamba3::backends::Auto;
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout, set_fused_entity_model,
};
use mamba3::train::TrainStep;

type R = Auto;

/// Launch/read counters are process-wide: tests that measure them hold this
/// for their whole body (parallel test threads would otherwise interleave).
static COUNT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        context: vec![
            ContextSetSpec::new("tiles", 16, 6).with_layout(SetLayout::Grid {
                height: 4,
                width: 4,
                alternate_axes: true,
            }),
        ],
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
    let _guard = COUNT_LOCK.lock().unwrap();
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

/// Full Kaggriculture §1.4 spec (K6 pin subject).
fn kaggriculture_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 114,
        context: vec![
            ContextSetSpec::new("tiles", 100, 48).with_layout(SetLayout::Grid {
                height: 10,
                width: 10,
                alternate_axes: true,
            }),
        ],
        queries: Some(
            QuerySetSpec::new("units", 20, 36, 3)
                .with_anchor("tiles")
                .with_autoregressive("target"),
        ),
        heads: vec![
            HeadSpec::pointer("target", "tiles", 1).step_weights(vec![1.0, 0.5, 0.5]),
            HeadSpec::categorical("op", 13).condition_on("target"),
            HeadSpec::multilabel("opset", 13)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::categorical("crop", 5)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::regression("eta", 1)
                .first_step_only()
                .loss_weight(0.1),
        ],
        d_model: 128,
        context_layers: 3,
        decoder_layers: 3,
        decoder: DecoderMode::StepCausal {
            crew_symmetric: true,
        },
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 128,
            n_heads: 4,
            head_dim: 64,
            d_state: 32,
            n_groups: 1,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 0,
    }
}

/// Synthetic two-context-set spec (K6 pin subject): pointers over different
/// sets, conditioned + multi-output heads.
fn two_set_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 3,
        context: vec![
            ContextSetSpec::new("cells", 16, 6).with_layout(SetLayout::Grid {
                height: 4,
                width: 4,
                alternate_axes: true,
            }),
            ContextSetSpec::new("items", 5, 2),
        ],
        queries: Some(
            QuerySetSpec::new("agents", 4, 5, 2)
                .with_anchor("cells")
                .with_autoregressive("tgt"),
        ),
        heads: vec![
            HeadSpec::pointer("tgt", "cells", 1),
            HeadSpec::pointer("gift", "items", 0),
            HeadSpec::categorical("kind", 3).condition_on("tgt"),
            HeadSpec::regression("amount", 2).loss_weight(0.5),
        ],
        d_model: 32,
        context_layers: 2,
        decoder_layers: 2,
        decoder: DecoderMode::StepCausal {
            crew_symmetric: true,
        },
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

fn pin_fused_launches(spec: &EntityModelSpec, arrays: &HostArrays, launches: usize) {
    let _guard = COUNT_LOCK.lock().unwrap();
    let device = dev();
    set_fused_entity_model(true);
    let model = EntityModel::<R, f32>::init(spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(spec, arrays, &device).unwrap();
    let task = EntityTask::new(&model);
    // Warm up once (lazy init), then measure one train step.
    task.loss(&batch).unwrap().backward().unwrap();
    reset_launch_count();
    reset_read_count();
    task.loss(&batch).unwrap().backward().unwrap();
    assert_eq!(launch_count(), launches, "fused launches per train step");
    assert_eq!(read_count(), 0, "reads per train step");
}

#[test]
fn pin_kaggriculture_fused_launches() {
    let (n, u, k, q) = (100usize, 20, 3, 60);
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![1, n, 48], frand(n * 48, 1));
    a.insert_f32("globals", vec![1, 114], frand(114, 2));
    a.insert_f32("units", vec![1, u, 36], frand(u * 36, 3));
    a.insert_int("units.anchor", vec![1, u], vec![0; u]);
    a.insert_int("label.target", vec![1, u, k], vec![0; q]);
    a.insert_int("label.op", vec![1, u, k], vec![0; q]);
    a.insert_f32("label.opset", vec![1, u, k, 13], vec![0.0; q * 13]);
    a.insert_int("label.crop", vec![1, u, k], vec![0; q]);
    a.insert_f32("label.eta", vec![1, u, 1], vec![1.0; u]);
    pin_fused_launches(&kaggriculture_spec(), &a, 2002);
}

#[test]
fn pin_two_set_fused_launches() {
    let mut a = HostArrays::new();
    a.insert_f32("cells", vec![1, 16, 6], frand(96, 1));
    a.insert_f32("items", vec![1, 5, 2], frand(10, 2));
    a.insert_f32("globals", vec![1, 3], frand(3, 3));
    a.insert_f32("agents", vec![1, 4, 5], frand(20, 4));
    a.insert_int("agents.anchor", vec![1, 4], vec![0, 1, 2, 3]);
    a.insert_int("label.tgt", vec![1, 4, 2], vec![0, 1, 2, 3, 4, 5, 6, 16]);
    a.insert_int("label.gift", vec![1, 4, 2], vec![0, 1, 2, 3, 4, 0, 1, 2]);
    a.insert_int("label.kind", vec![1, 4, 2], vec![0, 1, 2, 0, 1, 2, 0, 1]);
    a.insert_f32("label.amount", vec![1, 4, 2, 2], frand(16, 5));
    pin_fused_launches(&two_set_spec(), &a, 1351);
}
