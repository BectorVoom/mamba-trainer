//! The Kaggriculture spec (ENTITY_MODEL_PLAN.md §1.4) in Rust: random data,
//! 20 training steps, losses and launch / read counts.
//!
//! This is the Rust-side smoke test and the profile subject (K6). The spec
//! itself is the only place that names tiles, units, ops, crops or eta.
//!
//! ```text
//! cargo run --release --no-default-features --features cpu --example entity_kaggriculture
//! ```

use mamba3::backend::{launch_count, read_count, reset_launch_count, reset_read_count};
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout,
};
use mamba3::prelude::*;
use mamba3::train::TrainStep;

type R = mamba3::backends::Auto;

/// The §1.4 reference spec.
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

/// Deterministic pseudo-random floats in [-1, 1].
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

/// Random host arrays with mostly-valid labels.
fn random_arrays(b: usize, seed: u64) -> HostArrays {
    let (n, u, k, q) = (100usize, 20, 3, 60);
    let mut s = seed.max(1);
    let mut ri = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    let mut anchor = vec![-1i64; b * u];
    let mut tgt = vec![-1i64; b * q];
    let mut op = vec![-1i64; b * q];
    let mut crop = vec![-1i64; b * q];
    let mut opset = vec![0.0f32; b * q * 13];
    let mut eta = vec![f32::NAN; b * u];
    for bi in 0..b {
        for uu in 0..u {
            if ri(5) > 0 {
                anchor[bi * u + uu] = ri(n) as i64;
                eta[bi * u + uu] = ri(20) as f32;
            }
            for j in 0..k {
                let f = bi * q + uu * k + j;
                let r = ri(10);
                if r < 7 {
                    tgt[f] = ri(n) as i64;
                    op[f] = ri(13) as i64;
                    crop[f] = ri(5) as i64;
                    opset[f * 13 + ri(13)] = 1.0;
                } else if r < 8 {
                    tgt[f] = n as i64;
                }
            }
        }
    }
    // eta is step-0 only.
    let mut eta_first = vec![f32::NAN; b * u * 1];
    for bi in 0..b {
        for uu in 0..u {
            eta_first[bi * u + uu] = eta[bi * u + uu];
        }
    }
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![b, n, 48], frand(b * n * 48, seed + 1));
    a.insert_f32("globals", vec![b, 114], frand(b * 114, seed + 2));
    a.insert_f32("units", vec![b, u, 36], frand(b * u * 36, seed + 3));
    a.insert_int("units.anchor", vec![b, u], anchor);
    a.insert_int("label.target", vec![b, u, k], tgt);
    a.insert_int("label.op", vec![b, u, k], op);
    a.insert_f32("label.opset", vec![b, u, k, 13], opset);
    a.insert_int("label.crop", vec![b, u, k], crop);
    a.insert_f32("label.eta", vec![b, u, 1], eta_first);
    a
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend: {}", device.name());
    let spec = kaggriculture_spec();
    spec.validate()?;
    println!(
        "spec: {} ctx tokens, {} query tokens, {} heads",
        spec.n_ctx(),
        spec.query_tokens(),
        spec.heads.len()
    );
    let model = EntityModel::<R, f32>::init(&spec, &device)?;
    let arrays = random_arrays(4, 99);
    let batch = EntityBatch::<R, f32>::from_host(&spec, &arrays, &device)?;
    let task = EntityTask::new(&model);
    let mut opt = AdamWConfig::builder()
        .learning_rate(3e-4)
        .weight_decay(0.05)
        .build()
        .init::<R, f32>();
    let params = task.parameters();
    reset_launch_count();
    reset_read_count();
    for step in 0..20 {
        let loss = task.loss(&batch)?;
        let grads = loss.backward()?;
        opt.step(&params, &grads)?;
        if step % 5 == 0 {
            println!("step {step:>2}: loss {}", loss.to_f32()[0]);
        }
    }
    device.synchronize();
    println!(
        "20 steps: {} launches, {} reads",
        launch_count(),
        read_count()
    );
    Ok(())
}
