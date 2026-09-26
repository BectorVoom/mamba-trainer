//! f16 smoke for the entity loss.
//!
//! Alone in its binary: the matmul precision is a process-global mode, and a
//! test running beside this one would compute its own matmuls in f16.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout,
};
use mamba3::prelude::*;
use mamba3::train::{Optimizer, TrainStep};

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

#[test]
fn f16_smoke_on_capable_backends() {
    use mamba3::tensor::ops::matmul::{MatmulPrecision, supports_matmul_precision};
    let device = dev();
    if !supports_matmul_precision(&device, MatmulPrecision::F16) {
        return;
    }
    let spec = EntityModelSpec {
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
        seed: 51,
    };
    // Random but valid labels: 2 anchored units, real-tile-or-NONE targets.
    let (n, u, k, q) = (16usize, 4, 2, 8);
    let mut anchor = vec![-1i64; u];
    let mut tgt = vec![-1i64; q];
    let mut op = vec![-1i64; q];
    let mut s = 53u64;
    let mut ri = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    for uu in 0..2 {
        anchor[uu] = ri(n) as i64;
        for j in 0..k {
            let t = if ri(5) < 4 { ri(n) as i64 } else { n as i64 };
            tgt[uu * k + j] = t;
            if t < n as i64 {
                op[uu * k + j] = ri(4) as i64;
            }
        }
    }
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![1, n, 6], frand(n * 6, 54));
    a.insert_f32("globals", vec![1, 8], frand(8, 55));
    a.insert_f32("units", vec![1, u, 5], frand(u * 5, 56));
    a.insert_int("units.anchor", vec![1, u], anchor);
    a.insert_int("label.target", vec![1, u, k], tgt);
    a.insert_int("label.op", vec![1, u, k], op);
    mamba3::tensor::ops::matmul::set_matmul_precision(MatmulPrecision::F16);
    let result = (|| -> mamba3::error::Result<Vec<f32>> {
        let model = EntityModel::<R, f32>::init(&spec, &device)?;
        let b = EntityBatch::<R, f32>::from_host(&spec, &a, &device)?;
        let task = EntityTask::new(&model).with_loss_scale(1024.0);
        let mut opt = AdamWConfig::builder()
            .learning_rate(1e-3)
            .eps(1e-8 * 1024.0)
            .weight_decay(0.0)
            .build()
            .init::<R, f32>();
        let params = task.parameters();
        let mut losses = Vec::new();
        for _ in 0..50 {
            let loss = task.loss(&b)?;
            losses.push(loss.to_f32()[0] / 1024.0);
            let grads = loss.backward()?;
            opt.step(&params, &grads)?;
        }
        Ok(losses)
    })();
    mamba3::tensor::ops::matmul::set_matmul_precision(MatmulPrecision::F32);
    let losses = result.unwrap();
    assert!(losses.iter().all(|v| v.is_finite()), "non-finite f16 loss");
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "f16 losses did not decrease: {} -> {}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}
