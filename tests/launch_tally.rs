//! M0.2: the launch tally names the model region and the public op.
//!
//! ```text
//! cargo test --features cpu --test launch_tally
//! ```

use mamba3::backend::{
    launch_count, launch_tally_detailed, reset_launch_count, reset_launch_tally,
    start_launch_tally, stop_launch_tally,
};
use mamba3::models::entity::{
    ContextSetSpec, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout, set_fused_entity_model,
};
use mamba3::prelude::*;
use mamba3::train::{AdamWConfig, TrainStep, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

fn kaggriculture_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 114,
        context: vec![ContextSetSpec::new("tiles", 100, 48).with_layout(
            SetLayout::Grid {
                height: 10,
                width: 10,
                alternate_axes: true,
            },
        )],
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
        decoder: mamba3::models::entity::DecoderMode::StepCausal {
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
    let mut a = HostArrays::new();
    a.insert_f32("tiles", vec![b, n, 48], frand(b * n * 48, seed + 1));
    a.insert_f32("globals", vec![b, 114], frand(b * 114, seed + 2));
    a.insert_f32("units", vec![b, u, 36], frand(b * u * 36, seed + 3));
    a.insert_int("units.anchor", vec![b, u], anchor);
    a.insert_int("label.target", vec![b, u, k], tgt);
    a.insert_int("label.op", vec![b, u, k], op);
    a.insert_f32("label.opset", vec![b, u, k, 13], opset);
    a.insert_int("label.crop", vec![b, u, k], crop);
    a.insert_f32("label.eta", vec![b, u, 1], eta);
    a
}

#[test]
fn fused_step_tally_has_labels_and_ops() -> Result<()> {
    set_fused_entity_model(true);
    let device = Device::<R>::default();
    let spec = kaggriculture_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device)?;
    let arrays = random_arrays(2, 99);
    let batch = EntityBatch::<R, f32>::from_host(&spec, &arrays, &device)?;
    let task = EntityTask::new(&model);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(3e-4).build()?,
        AdamWConfig::builder()
            .learning_rate(3e-4)
            .build()
            .init::<R, f32>(),
    );
    // Warm up outside the tally (compiles kernels).
    trainer.step(&task, std::slice::from_ref(&batch))?;

    reset_launch_count();
    reset_launch_tally();
    start_launch_tally();
    trainer.step(&task, std::slice::from_ref(&batch))?;
    stop_launch_tally();

    let rows = launch_tally_detailed();
    assert!(!rows.is_empty(), "a training step launches kernels");
    for row in &rows {
        assert!(!row.op.is_empty(), "op named at {}", row.site);
        assert!(!row.label.is_empty(), "label set at {}", row.site);
    }
    let total: usize = rows.iter().map(|r| r.count).sum();
    assert_eq!(total, launch_count(), "tally rows sum to launch_count");

    let expected = [
        "mixer.project",
        "mixer.conv",
        "mixer.coef",
        "mixer.scan",
        "mixer.out",
        "scan.intra",
        "scan.summary",
        "scan.inter",
        "scan.out",
        "encoder",
        "queries",
        "decoder",
        "heads",
        "loss",
        "backward",
        "optimizer",
    ];
    let seen: std::collections::HashSet<&str> =
        rows.iter().map(|r| r.label.as_str()).collect();
    for label in expected {
        assert!(seen.contains(label), "tally sees label {label}; got {seen:?}");
    }
    Ok(())
}
