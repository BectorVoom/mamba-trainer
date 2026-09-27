//! K7: the multi-tensor optimizer matches the per-parameter one, in fewer launches.
//!
//! ```text
//! cargo test --features cpu --test adamw_multi
//! ```

#![cfg(feature = "backend")]

use std::collections::HashSet;

use mamba3::backend::{
    launch_count, launch_tally_detailed, reset_launch_count, reset_launch_tally,
    start_launch_tally, stop_launch_tally,
};
use mamba3::models::entity::{
    ContextSetSpec, EntityBatch, EntityModel, EntityModelSpec, EntityTask, HeadSpec,
    HostArrays, QuerySetSpec, SetLayout, set_fused_entity_model,
};
use mamba3::prelude::*;
use mamba3::tensor::ops::fused::{SUM_SQUARES_MULTI_SLOTS, set_adamw_multi};
use mamba3::train::{AdamWConfig, TrainStep, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

/// Launch/read counters are process-wide: tests that measure them hold this
/// for their whole body (parallel test threads would otherwise interleave).
static COUNT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn run_steps(multi: bool, steps: usize) -> Result<Vec<(String, Vec<f32>)>> {
    let _guard = COUNT_LOCK.lock().unwrap();
    let prev = mamba3::tensor::ops::fused::adamw_multi_enabled();
    set_adamw_multi(multi);
    let out = (|| -> Result<Vec<(String, Vec<f32>)>> {
        set_fused_entity_model(true);
        let device = Device::<R>::default();
        let spec = kaggriculture_spec();
        let model = EntityModel::<R, f32>::init(&spec, &device)?;
        let arrays = random_arrays(2, 99);
        let batch = EntityBatch::<R, f32>::from_host(&spec, &arrays, &device)?;
        let task = EntityTask::new(&model);
        // The clip is disabled so the gradient scale is the micro-batch
        // average bit-for-bit on both paths: both reductions then reorder
        // nothing observable, and the comparison below checks the optimizer
        // math itself. (With clipping engaged the two reductions reassociate
        // the norm — as every parallel reduction in reduce.rs may — and agree
        // only to the last bits; the norm path's launches are pinned below.)
        let mut trainer = Trainer::new(
            TrainerConfig::builder()
                .learning_rate(3e-4)
                .max_grad_norm(0.0)
                .build()?,
            AdamWConfig::builder()
                .learning_rate(3e-4)
                .build()
                .init::<R, f32>(),
        );
        for _ in 0..steps {
            trainer.step(&task, std::slice::from_ref(&batch))?;
        }
        // Names are not stable across runs (ParamIds are fresh), so key by
        // shape + index within shape.
        let mut params = task.parameters();
        params.sort_by_key(|p| (p.numel(), p.shape().dims().to_vec()));
        let mut seen: std::collections::HashMap<(usize, Vec<usize>), usize> =
            std::collections::HashMap::new();
        let mut out = Vec::new();
        for p in &params {
            let key = (p.numel(), p.shape().dims().to_vec());
            let n = seen.entry(key.clone()).or_insert(0);
            *n += 1;
            out.push((format!("{}x{}#{n}", key.0, key.1.len()), p.value().to_f32()));
        }
        Ok(out)
    })();
    set_adamw_multi(prev);
    out
}

#[test]
fn multi_tensor_optimizer_matches_per_parameter() -> Result<()> {
    let single = run_steps(false, 20)?;
    let multi = run_steps(true, 20)?;
    assert_eq!(single.len(), multi.len(), "same parameter count");
    for ((name_a, a), (name_b, b)) in single.iter().zip(multi.iter()) {
        assert_eq!(name_a, name_b, "same parameter order");
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            let tol = 1e-6 * x.abs().max(y.abs()).max(1.0);
            assert!(
                (x - y).abs() <= tol,
                "param {name_a} drifts: {x} vs {y}"
            );
        }
    }
    Ok(())
}

#[test]
fn optimizer_launches_fit_in_chunks() -> Result<()> {    let _guard = COUNT_LOCK.lock().unwrap();
    let prev = mamba3::tensor::ops::fused::adamw_multi_enabled();
    set_adamw_multi(true);
    let out = (|| -> Result<()> {
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
        trainer.step(&task, std::slice::from_ref(&batch))?;
        let n_params = task.parameters().iter().filter(|p| p.requires_grad()).count();

        reset_launch_count();
        reset_launch_tally();
        start_launch_tally();
        trainer.step(&task, std::slice::from_ref(&batch))?;
        stop_launch_tally();

        // Optimizer-region launches: one adamw chunk + one norm chunk per
        // group of slots, plus the clip factor, the partials buffer fill and
        // the final fold.
        let opt: usize = launch_tally_detailed()
            .iter()
            .filter(|r| r.label == "optimizer")
            .map(|r| r.count)
            .sum();
        let ops: HashSet<String> = launch_tally_detailed()
            .iter()
            .filter(|r| r.label == "optimizer")
            .map(|r| r.op.clone())
            .collect();
        let _ = launch_count();
        let bound = 2 * n_params.div_ceil(SUM_SQUARES_MULTI_SLOTS) + 6;
        assert!(
            opt <= bound,
            "optimizer launches {opt} exceed 2*ceil({n_params}/8)+3={bound} (ops: {ops:?})"
        );
        Ok(())
    })();
    set_adamw_multi(prev);
    out
}

#[test]
fn multi_norm_matches_per_gradient_norm() -> Result<()> {
    let _guard = COUNT_LOCK.lock().unwrap();
    let prev = mamba3::tensor::ops::fused::adamw_multi_enabled();
    let device = Device::<R>::default();
    // Uneven sizes, including an empty gradient and one needing many groups.
    let sizes = [3usize, 0, 17, 1000, 65536, 7, 129, 4096, 5];
    let grads_single: Vec<Tensor<R, f32>> = sizes
        .iter()
        .map(|n| {
            Tensor::from_f32(&frand(*n.max(&1), (*n as u64) + 7)[..*n], vec![*n], &device)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut map_single = mamba3::autograd::Grads::default();
    let mut map_multi = mamba3::autograd::Grads::default();
    for g in grads_single.iter() {
        let id = mamba3::autograd::ParamId::fresh();
        map_single.accumulate(id, g.clone())?;
        map_multi.accumulate(id, g.clone())?;
    }
    set_adamw_multi(false);
    let norm_single = mamba3::train::optim::grad_norm(&map_single)?;
    set_adamw_multi(true);
    let norm_multi = mamba3::train::optim::grad_norm(&map_multi)?;
    set_adamw_multi(prev);
    let tol = 1e-5 * norm_single.abs().max(1.0);
    assert!(
        (norm_single - norm_multi).abs() <= tol,
        "norm {norm_single} vs multi {norm_multi}"
    );
    Ok(())
}
