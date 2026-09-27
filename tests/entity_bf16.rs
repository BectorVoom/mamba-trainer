//! K10: bf16 storage for activations tracks f32.
//!
//! Alone in its binary: 200 training steps take a while, and the loss
//! comparison needs no interference.
//!
//! ```text
//! cargo test --release --features cpu --test entity_bf16
//! ```

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout,
};
use mamba3::prelude::*;
use mamba3::train::{AdamWConfig, TrainStep, Trainer, TrainerConfig};

type R = Auto;

fn spec() -> EntityModelSpec {
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
        seed: 51,
    }
}

fn arrays() -> HostArrays {
    let mut s = 53u64;
    let mut ri = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    let (n, u, k, q) = (16usize, 4, 2, 8);
    let mut anchor = vec![-1i64; u];
    let mut tgt = vec![-1i64; q];
    let mut op = vec![-1i64; q];
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
    let mut rf = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f32) / (u64::MAX >> 11) as f32 * 2.0 - 1.0
            })
            .collect()
    };
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![1, n, 6], rf(n * 6, 54));
    a.insert_f32("globals", vec![1, 8], rf(8, 55));
    a.insert_f32("units", vec![1, u, 5], rf(u * 5, 56));
    a.insert_int("units.anchor", vec![1, u], anchor);
    a.insert_int("label.target", vec![1, u, k], tgt);
    a.insert_int("label.op", vec![1, u, k], op);
    a
}

fn run<E: mamba3::backend::FloatElem>(steps: usize) -> Result<Vec<f32>> {
    let device = Device::<R>::default();
    let spec = spec();
    let model = EntityModel::<R, E>::init(&spec, &device)?;
    let b = EntityBatch::<R, E>::from_host(&spec, &arrays(), &device)?;
    let task = EntityTask::new(&model);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-3).build()?,
        AdamWConfig::builder()
            .learning_rate(1e-3)
            .weight_decay(0.0)
            .build()
            .init::<R, E>(),
    );
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        out.push(trainer.step(&task, std::slice::from_ref(&b))?.loss);
    }
    Ok(out)
}

#[test]
fn bf16_tracks_f32_for_200_steps() -> Result<()> {
    let f32 = run::<f32>(200)?;
    let bf16 = run::<half::bf16>(200)?;
    assert!(
        bf16.iter().all(|v| v.is_finite()),
        "non-finite bf16 step"
    );
    assert!(
        bf16.last().unwrap() < bf16.first().unwrap(),
        "bf16 losses did not decrease: {} -> {}",
        bf16.first().unwrap(),
        bf16.last().unwrap()
    );
    let (a, b) = (*f32.last().unwrap(), *bf16.last().unwrap());
    // Relative with a floor: on this toy spec both runs converge to ~zero
    // (bf16 rounds the last 1e-4 to exactly 0), so a pure relative check
    // divides by ~zero. The plan's 2% criterion is the relative part.
    let rel = (a - b).abs() / a.abs().max(1.0);
    assert!(
        rel <= 0.02,
        "bf16 final loss {b} drifted more than 2% from f32 {a}"
    );
    Ok(())
}
