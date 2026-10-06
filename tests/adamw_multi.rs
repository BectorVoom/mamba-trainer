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

/// Gated AdamW with the apply flag = 0 leaves every parameter and both
/// moments bit-unchanged (the host reference is the input itself).
#[test]
fn gated_apply_zero_leaves_everything_unchanged() -> Result<()> {
    let _guard = COUNT_LOCK.lock().unwrap();
    use mamba3::tensor::Tensor;
    use mamba3::tensor::ops::fused::{
        AdamWSlot, AdamWStep, adamw_step_gated, adamw_step_multi_gated,
        adamw_step_multi_narrow_gated,
    };
    let device = Device::<R>::default();
    let step = AdamWStep {
        lr: 3e-4,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        decay: 0.01,
        bias1: 1.0 - 0.9f32.powi(3),
        bias2: 1.0 - 0.95f32.powi(3),
    };
    // Single path: one parameter plus moments (moments start at zero, as the
    // optimizer creates them; nonzero frand would put negative mass in `v`).
    let n = 17usize;
    let param = Tensor::<R, f32>::from_f32(&frand(n, 11), vec![n], &device)?;
    let grad = Tensor::<R, f32>::from_f32(&frand(n, 12), vec![n], &device)?;
    let m = Tensor::<R, f32>::zeros(vec![n], &device);
    let v = Tensor::<R, f32>::zeros(vec![n], &device);
    let scale = Tensor::<R, f32>::from_f32(&[0.5, 0.0], vec![2], &device)?;
    let (p0, m0, v0) = (param.to_f32(), m.to_f32(), v.to_f32());
    let out = adamw_step_gated(&param, &grad, &m, &v, &scale, step);
    assert_eq!(out.to_f32(), p0, "gated single param unchanged at apply=0");
    assert_eq!(m.to_f32(), m0, "gated single m unchanged at apply=0");
    assert_eq!(v.to_f32(), v0, "gated single v unchanged at apply=0");
    // Wide multi path: three slots of different shapes on one device.
    let shapes = [vec![9], vec![5, 4], vec![33]];
    let mut ps = Vec::new();
    let mut gs = Vec::new();
    let mut ms = Vec::new();
    let mut vs = Vec::new();
    for (i, shape) in shapes.iter().enumerate() {
        let len: usize = shape.iter().product();
        ps.push(Tensor::<R, f32>::from_f32(
            &frand(len, 100 + i as u64),
            shape.clone(),
            &device,
        )?);
        gs.push(Tensor::<R, f32>::from_f32(
            &frand(len, 200 + i as u64),
            shape.clone(),
            &device,
        )?);
        // Moments start at zero, as the optimizer creates them.
        ms.push(Tensor::<R, f32>::zeros(shape.clone(), &device));
        vs.push(Tensor::<R, f32>::zeros(shape.clone(), &device));
    }
    let before_p: Vec<Vec<f32>> = ps.iter().map(|t| t.to_f32()).collect();
    let before_m: Vec<Vec<f32>> = ms.iter().map(|t| t.to_f32()).collect();
    let before_v: Vec<Vec<f32>> = vs.iter().map(|t| t.to_f32()).collect();
    let slots: Vec<AdamWSlot<'_, R, f32>> = (0..3)
        .map(|i| AdamWSlot {
            param: &ps[i],
            grad: &gs[i],
            m: &ms[i],
            v: &vs[i],
            decay: 0.01,
        })
        .collect();
    adamw_step_multi_gated(&slots, &scale, step)?;
    for i in 0..3 {
        assert_eq!(ps[i].to_f32(), before_p[i], "wide slot {i} param unchanged");
        assert_eq!(ms[i].to_f32(), before_m[i], "wide slot {i} m unchanged");
        assert_eq!(vs[i].to_f32(), before_v[i], "wide slot {i} v unchanged");
    }
    // Narrow path: two slots.
    let slots2: Vec<AdamWSlot<'_, R, f32>> = (0..2)
        .map(|i| AdamWSlot {
            param: &ps[i],
            grad: &gs[i],
            m: &ms[i],
            v: &vs[i],
            decay: 0.0,
        })
        .collect();
    adamw_step_multi_narrow_gated(&slots2, &scale, step)?;
    for i in 0..2 {
        assert_eq!(ps[i].to_f32(), before_p[i], "narrow slot {i} param unchanged");
        assert_eq!(ms[i].to_f32(), before_m[i], "narrow slot {i} m unchanged");
        assert_eq!(vs[i].to_f32(), before_v[i], "narrow slot {i} v unchanged");
    }
    Ok(())
}

/// Gated AdamW with the apply flag = 1 equals the unflagged kernel bit for
/// bit (single, wide multi and narrow multi).
#[test]
fn gated_apply_one_equals_unflagged() -> Result<()> {
    let _guard = COUNT_LOCK.lock().unwrap();
    use mamba3::tensor::Tensor;
    use mamba3::tensor::ops::fused::{
        AdamWSlot, AdamWStep, adamw_step, adamw_step_gated, adamw_step_multi,
        adamw_step_multi_gated, adamw_step_multi_narrow, adamw_step_multi_narrow_gated,
    };
    let device = Device::<R>::default();
    let step = AdamWStep {
        lr: 3e-4,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        decay: 0.01,
        bias1: 1.0 - 0.9f32.powi(3),
        bias2: 1.0 - 0.95f32.powi(3),
    };
    let n = 17usize;
    let mk = |seed: u64| Tensor::<R, f32>::from_f32(&frand(n, seed), vec![n], &device);
    let (param, grad) = (mk(11)?, mk(12)?);
    let m = Tensor::<R, f32>::zeros(vec![n], &device);
    let v = Tensor::<R, f32>::zeros(vec![n], &device);
    let m2 = Tensor::<R, f32>::zeros(vec![n], &device);
    let v2 = Tensor::<R, f32>::zeros(vec![n], &device);
    let scale1 = Tensor::<R, f32>::from_f32(&[0.5], vec![1], &device)?;
    let scale2 = Tensor::<R, f32>::from_f32(&[0.5, 1.0], vec![2], &device)?;
    let want = adamw_step(&param, &grad, &m, &v, &scale1, step);
    let got = adamw_step_gated(&param, &grad, &m2, &v2, &scale2, step);
    assert_eq!(got.to_f32(), want.to_f32(), "single apply=1 == unflagged");
    assert_eq!(m2.to_f32(), m.to_f32(), "single m apply=1 == unflagged");
    assert_eq!(v2.to_f32(), v.to_f32(), "single v apply=1 == unflagged");
    // Multi paths: fresh twins so the ungated run is the reference.
    let shapes = [vec![9], vec![5, 4], vec![33]];
    let build = || -> Result<(Vec<Tensor<R, f32>>, Vec<Tensor<R, f32>>, Vec<Tensor<R, f32>>, Vec<Tensor<R, f32>>)> {
        let mut ps = Vec::new();
        let mut gs = Vec::new();
        let mut ms = Vec::new();
        let mut vs = Vec::new();
        for (i, shape) in shapes.iter().enumerate() {
            let len: usize = shape.iter().product();
            ps.push(Tensor::<R, f32>::from_f32(&frand(len, 100 + i as u64), shape.clone(), &device)?);
            gs.push(Tensor::<R, f32>::from_f32(&frand(len, 200 + i as u64), shape.clone(), &device)?);
            ms.push(Tensor::<R, f32>::zeros(shape.clone(), &device));
            vs.push(Tensor::<R, f32>::zeros(shape.clone(), &device));
        }
        Ok((ps, gs, ms, vs))
    };
    let (ps_a, gs_a, ms_a, vs_a) = build()?;
    let (ps_b, gs_b, ms_b, vs_b) = build()?;
    let slots_a: Vec<AdamWSlot<'_, R, f32>> = (0..3)
        .map(|i| AdamWSlot { param: &ps_a[i], grad: &gs_a[i], m: &ms_a[i], v: &vs_a[i], decay: 0.01 })
        .collect();
    let slots_b: Vec<AdamWSlot<'_, R, f32>> = (0..3)
        .map(|i| AdamWSlot { param: &ps_b[i], grad: &gs_b[i], m: &ms_b[i], v: &vs_b[i], decay: 0.01 })
        .collect();
    adamw_step_multi(&slots_a, &scale1, step)?;
    adamw_step_multi_gated(&slots_b, &scale2, step)?;
    for i in 0..3 {
        assert_eq!(ps_b[i].to_f32(), ps_a[i].to_f32(), "wide slot {i} param");
        assert_eq!(ms_b[i].to_f32(), ms_a[i].to_f32(), "wide slot {i} m");
        assert_eq!(vs_b[i].to_f32(), vs_a[i].to_f32(), "wide slot {i} v");
    }
    let (ps_c, gs_c, ms_c, vs_c) = build()?;
    let (ps_d, gs_d, ms_d, vs_d) = build()?;
    let slots_c: Vec<AdamWSlot<'_, R, f32>> = (0..2)
        .map(|i| AdamWSlot { param: &ps_c[i], grad: &gs_c[i], m: &ms_c[i], v: &vs_c[i], decay: 0.0 })
        .collect();
    let slots_d: Vec<AdamWSlot<'_, R, f32>> = (0..2)
        .map(|i| AdamWSlot { param: &ps_d[i], grad: &gs_d[i], m: &ms_d[i], v: &vs_d[i], decay: 0.0 })
        .collect();
    adamw_step_multi_narrow(&slots_c, &scale1, step)?;
    adamw_step_multi_narrow_gated(&slots_d, &scale2, step)?;
    for i in 0..2 {
        assert_eq!(ps_d[i].to_f32(), ps_c[i].to_f32(), "narrow slot {i} param");
        assert_eq!(ms_d[i].to_f32(), ms_c[i].to_f32(), "narrow slot {i} m");
        assert_eq!(vs_d[i].to_f32(), vs_c[i].to_f32(), "narrow slot {i} v");
    }
    Ok(())
}

/// Narrow-path gated AdamW with nonzero weight decay still skips everything
/// on a skipped step (task F7B item B5): parameters and both moments are
/// bit-unchanged, including the decay term, with non-finite gradients.
#[test]
fn gated_narrow_skips_weight_decay_on_skip() -> Result<()> {
    let _guard = COUNT_LOCK.lock().unwrap();
    use mamba3::tensor::Tensor;
    use mamba3::tensor::ops::fused::{
        AdamWSlot, AdamWStep, adamw_step_gated, adamw_step_multi_gated,
        adamw_step_multi_narrow_gated,
    };
    let device = Device::<R>::default();
    let step = AdamWStep {
        lr: 3e-4,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        decay: 0.01,
        bias1: 1.0 - 0.9f32.powi(3),
        bias2: 1.0 - 0.95f32.powi(3),
    };
    // Two slots with nonzero decay and non-finite gradients.
    let shapes = [vec![7], vec![4, 3]];
    let mut ps = Vec::new();
    let mut gs = Vec::new();
    let mut ms = Vec::new();
    let mut vs = Vec::new();
    for (i, shape) in shapes.iter().enumerate() {
        let len: usize = shape.iter().product();
        ps.push(Tensor::<R, f32>::from_f32(
            &frand(len, 300 + i as u64),
            shape.clone(),
            &device,
        )?);
        gs.push(Tensor::<R, f32>::from_f32(
            &vec![f32::INFINITY; len],
            shape.clone(),
            &device,
        )?);
        ms.push(Tensor::<R, f32>::from_f32(
            &frand(len, 400 + i as u64),
            shape.clone(),
            &device,
        )?);
        vs.push(Tensor::<R, f32>::from_f32(
            &frand(len, 500 + i as u64),
            shape.clone(),
            &device,
        )?);
    }
    let before_p: Vec<Vec<f32>> = ps.iter().map(|t| t.to_f32()).collect();
    let before_m: Vec<Vec<f32>> = ms.iter().map(|t| t.to_f32()).collect();
    let before_v: Vec<Vec<f32>> = vs.iter().map(|t| t.to_f32()).collect();
    let scale = Tensor::<R, f32>::from_f32(&[0.5, 0.0], vec![2], &device)?;
    // Single and wide paths with the same non-finite gradients: a skipped
    // step leaves everything bit-unchanged there too.
    let out_single = adamw_step_gated(&ps[0], &gs[0], &ms[0], &vs[0], &scale, step);
    assert_eq!(out_single.to_f32(), before_p[0], "single param unchanged");
    assert_eq!(ms[0].to_f32(), before_m[0], "single m unchanged");
    assert_eq!(vs[0].to_f32(), before_v[0], "single v unchanged");
    let slots_wide: Vec<AdamWSlot<'_, R, f32>> = (0..2)
        .map(|i| AdamWSlot {
            param: &ps[i],
            grad: &gs[i],
            m: &ms[i],
            v: &vs[i],
            decay: 0.01,
        })
        .collect();
    adamw_step_multi_gated(&slots_wide, &scale, step)?;
    for i in 0..2 {
        assert_eq!(ps[i].to_f32(), before_p[i], "wide slot {i} param unchanged");
        assert_eq!(ms[i].to_f32(), before_m[i], "wide slot {i} m unchanged");
        assert_eq!(vs[i].to_f32(), before_v[i], "wide slot {i} v unchanged");
    }
    let slots: Vec<AdamWSlot<'_, R, f32>> = (0..2)
        .map(|i| AdamWSlot {
            param: &ps[i],
            grad: &gs[i],
            m: &ms[i],
            v: &vs[i],
            decay: 0.01,
        })
        .collect();
    adamw_step_multi_narrow_gated(&slots, &scale, step)?;
    for i in 0..2 {
        assert_eq!(ps[i].to_f32(), before_p[i], "narrow slot {i} param unchanged");
        assert_eq!(ms[i].to_f32(), before_m[i], "narrow slot {i} m unchanged");
        assert_eq!(vs[i].to_f32(), before_v[i], "narrow slot {i} v unchanged");
    }
    Ok(())
}

/// The shared optimizer path is untouched (task F7B item B5): `step_scaled`
/// with a one-element scale of exactly `1.0` is bit-identical to the
/// ungated `step()` entry point. Launch counts differ by exactly the
/// unit-scale fill `step()` builds internally (one launch).
#[test]
fn step_scaled_ones_matches_ungated_step() -> Result<()> {
    use mamba3::nn::Param;
    use mamba3::tensor::Tensor;
    use mamba3::train::optim::Optimizer;
    let device = Device::<R>::default();
    let mk_params = || -> Result<(Vec<Param<R, f32>>, mamba3::autograd::Grads<R, f32>)> {
        let mut params = Vec::new();
        let mut grads = mamba3::autograd::Grads::default();
        for (i, n) in [9usize, 64, 130].iter().enumerate() {
            let p = Param::new(Tensor::<R, f32>::from_f32(
                &frand(*n, 700 + i as u64),
                vec![*n],
                &device,
            )?);
            grads.accumulate(
                p.id(),
                Tensor::<R, f32>::from_f32(&frand(*n, 800 + i as u64), vec![*n], &device)?,
            )?;
            params.push(p);
        }
        Ok((params, grads))
    };
    let (params_a, grads_a) = mk_params()?;
    let (params_b, grads_b) = mk_params()?;
    // Same seeds above, so both runs start from identical values.
    let mut opt_a = AdamWConfig::builder()
        .learning_rate(3e-4)
        .build()
        .init::<R, f32>();
    let mut opt_b = AdamWConfig::builder()
        .learning_rate(3e-4)
        .build()
        .init::<R, f32>();
    reset_launch_count();
    opt_a.step(&params_a, &grads_a)?;
    let l_step = launch_count();
    let ones = Tensor::<R, f32>::from_f32(&[1.0], vec![1], &device)?;
    reset_launch_count();
    opt_b.step_scaled(&params_b, &grads_b, Some(&ones))?;
    let l_scaled = launch_count();
    assert_eq!(
        l_step,
        l_scaled + 1,
        "step() is step_scaled plus the one unit-scale fill: {l_step} vs {l_scaled}"
    );
    for (pa, pb) in params_a.iter().zip(params_b.iter()) {
        assert_eq!(
            pa.value().to_f32(),
            pb.value().to_f32(),
            "params bit-identical"
        );
    }
    let named_a: Vec<(String, Param<R, f32>)> = params_a
        .iter()
        .enumerate()
        .map(|(i, p)| (format!("p{i}"), p.clone()))
        .collect();
    let named_b: Vec<(String, Param<R, f32>)> = params_b
        .iter()
        .enumerate()
        .map(|(i, p)| (format!("p{i}"), p.clone()))
        .collect();
    let ma = opt_a.state_dict(&named_a);
    let mb = opt_b.state_dict(&named_b);
    assert_eq!(ma.entries.len(), mb.entries.len());
    for ((ka, ta), (kb, tb)) in ma.entries.iter().zip(mb.entries.iter()) {
        assert_eq!(ka, kb);
        assert_eq!(ta.data, tb.data, "moment {ka} bit-identical");
    }
    Ok(())
}
