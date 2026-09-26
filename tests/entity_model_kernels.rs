//! Fused-vs-composed parity for the entity-model kernels (K0–K5).
//!
//! Alone in its binary: the fused switch is process-global, and toggling it
//! in a shared binary makes flaky tests. Invariants (every K task): forward
//! parity 1e-6 relative, gradient parity 1e-5, no host reads, no atomics,
//! f32 accumulation, `IGNORE = u32::MAX` tested before every indexed read.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::model::{Decode, EntityModel};
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityModelSpec, EntityTask, HeadSpec, HostArrays,
    QuerySetSpec, SetLayout,
};
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::entity_model::IGNORE;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::TrainStep;

type R = Auto;

/// The fused switch is process-global: tests that toggle it hold this for
/// their whole body (parallel test threads would otherwise flake).
static SWITCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (1.0 + x.abs().max(y.abs())))
        .fold(0.0f32, f32::max)
}

fn tiny_spec(crew: bool) -> EntityModelSpec {
    EntityModelSpec {
        globals: 3,
        context: vec![
            ContextSetSpec::new("cells", 9, 4).with_layout(SetLayout::Grid {
                height: 3,
                width: 3,
                alternate_axes: true,
            }),
            ContextSetSpec::new("items", 4, 2),
        ],
        queries: Some(
            QuerySetSpec::new("agents", 2, 3, 2)
                .with_anchor("cells")
                .with_autoregressive("tgt"),
        ),
        heads: vec![
            HeadSpec::pointer("tgt", "cells", 1),
            HeadSpec::categorical("kind", 2).condition_on("tgt"),
            HeadSpec::regression("eta", 1)
                .first_step_only()
                .loss_weight(0.5),
        ],
        d_model: 8,
        context_layers: 2,
        decoder_layers: 1,
        decoder: DecoderMode::StepCausal {
            crew_symmetric: crew,
        },
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 8,
            n_heads: 2,
            head_dim: 4,
            d_state: 4,
            n_groups: 2,
            chunk_size: 4,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 9,
    }
}

fn tiny_arrays() -> HostArrays {
    let (b, m, k, q) = (1usize, 2, 2, 4);
    let mut a = HostArrays::new();
    a.insert_f32("cells", vec![b, 9, 4], frand(b * 9 * 4, 501));
    let mut presence = vec![1.0f32; b * 9];
    presence[5] = 0.0; // one absent cell: exercises the presence mask.
    a.insert_f32("cells.presence", vec![b, 9], presence);
    a.insert_f32("items", vec![b, 4, 2], frand(b * 4 * 2, 502));
    a.insert_f32("globals", vec![b, 3], frand(b * 3, 503));
    a.insert_f32("agents", vec![b, m, 3], frand(b * m * 3, 504));
    a.insert_int("agents.anchor", vec![b, m], vec![1, 1]);
    a.insert_int("label.tgt", vec![b, m, k], vec![2, 9, -1, 0]);
    a.insert_int("label.kind", vec![b, m, k], vec![1, 0, -1, 1]);
    a.insert_f32("label.eta", vec![b, m, 1], vec![0.25, f32::NAN]);
    let _ = q;
    a
}

#[test]
fn k4_permute_matches_composed() {
    use mamba3::models::entity::blocks::Permutation;
    let device = dev();
    let (b, n, d) = (2usize, 21, 8);
    let data = frand(b * n * d, 601);
    let p = Permutation::grid_transpose(5, 4, 4, n);
    let fwd = IdTensor::from_slice(&p.fwd, vec![n], &device).unwrap();
    let inv = IdTensor::from_slice(&p.inv, vec![n], &device).unwrap();
    let run = |fused: bool| {
        let x = Var::traced(Tensor::<R, f32>::from_f32(&data, vec![b, n, d], &device).unwrap());
        let y = if fused {
            Var::permute_tokens(&x, &fwd, &inv).unwrap()
        } else {
            p.apply(&x).unwrap()
        };
        let loss = y.sum().unwrap();
        let grads = loss.backward_retain().unwrap();
        let g = grads.node(x.node().unwrap()).unwrap().to_f32();
        (y.to_f32(), g)
    };
    let (yf, gf) = run(true);
    let (yc, gc) = run(false);
    assert!(rel_diff(&yf, &yc) < 1e-6, "k4 forward");
    assert!(rel_diff(&gf, &gc) < 1e-5, "k4 backward");
    assert_eq!(yf, yc, "k4 permutation is exact");
}

#[test]
fn k5_join_matches_composed() {
    let device = dev();
    let (b, n, d) = (2usize, 13, 8);
    let xd = frand(b * n * d, 611);
    let pd = frand(9 * d, 612);
    let td = frand(2 * d, 613);
    let gd = frand(b * d, 614);
    // Set 0 (9 slots) embeds rows 0..9, set 1 (4 slots) has none.
    let pos_row: Vec<u32> = (0..9).chain(std::iter::repeat(IGNORE).take(4)).collect();
    let set_of: Vec<u32> = std::iter::repeat(0)
        .take(9)
        .chain(std::iter::repeat(1).take(4))
        .collect();
    let run = |fused: bool| {
        let x = Var::traced(Tensor::<R, f32>::from_f32(&xd, vec![b, n, d], &device).unwrap());
        let pos = Var::traced(Tensor::<R, f32>::from_f32(&pd, vec![9, d], &device).unwrap());
        let typ = Var::traced(Tensor::<R, f32>::from_f32(&td, vec![2, d], &device).unwrap());
        let g = Var::traced(Tensor::<R, f32>::from_f32(&gd, vec![b, d], &device).unwrap());
        let y = if fused {
            let pr = IdTensor::from_slice(&pos_row, vec![n], &device).unwrap();
            let so = IdTensor::from_slice(&set_of, vec![n], &device).unwrap();
            Var::broadcast_join(&x, &pos, &pr, &typ, &so, Some(&g)).unwrap()
        } else {
            // Composed: per-slot pos + per-set type + broadcast globals.
            let mut rows = Vec::with_capacity(n);
            for slot in 0..n {
                let mut r = x.slice(1, slot, 1).unwrap();
                if pos_row[slot] != IGNORE {
                    let pr = pos_row[slot] as usize;
                    let pv = pos
                        .slice(0, pr, 1)
                        .unwrap()
                        .reshape(vec![1, 1, d])
                        .unwrap()
                        .expand(vec![b, 1, d])
                        .unwrap();
                    r = r.add(&pv).unwrap();
                }
                let tr = set_of[slot] as usize;
                let tv = typ
                    .slice(0, tr, 1)
                    .unwrap()
                    .reshape(vec![1, 1, d])
                    .unwrap()
                    .expand(vec![b, 1, d])
                    .unwrap();
                r = r.add(&tv).unwrap();
                rows.push(r);
            }
            let mut y = mamba3::autograd::cat(&rows, 1).unwrap();
            y = y
                .add(
                    &g.reshape(vec![b, 1, d])
                        .unwrap()
                        .expand(vec![b, n, d])
                        .unwrap(),
                )
                .unwrap();
            y
        };
        let loss = y.sum().unwrap();
        let grads = loss.backward_retain().unwrap();
        let gx = grads.node(x.node().unwrap()).unwrap().to_f32();
        let gp = grads.node(pos.node().unwrap()).unwrap().to_f32();
        let gt = grads.node(typ.node().unwrap()).unwrap().to_f32();
        let gg = grads.node(g.node().unwrap()).unwrap().to_f32();
        (y.to_f32(), gx, gp, gt, gg)
    };
    let (yf, gxf, gpf, gtf, ggf) = run(true);
    let (yc, gxc, gpc, gtc, ggc) = run(false);
    assert!(rel_diff(&yf, &yc) < 1e-6, "k5 forward");
    assert!(rel_diff(&gxf, &gxc) < 1e-5, "k5 d_x");
    assert!(rel_diff(&gpf, &gpc) < 1e-5, "k5 d_pos");
    assert!(rel_diff(&gtf, &gtc) < 1e-5, "k5 d_typ");
    assert!(rel_diff(&ggf, &ggc) < 1e-5, "k5 d_g");
}

fn model_parity(crew: bool) {
    use mamba3::models::entity::set_fused_entity_model;
    let _guard = SWITCH_LOCK.lock().unwrap();
    let device = dev();
    let spec = tiny_spec(crew);
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &tiny_arrays(), &device).unwrap();
    let task = EntityTask::new(&model);
    set_fused_entity_model(false);
    let lc = task.loss(&batch).unwrap();
    let gc = lc.backward().unwrap();
    let pc = model.predict(&batch, Decode::Greedy, None).unwrap();
    set_fused_entity_model(true);
    let lf = task.loss(&batch).unwrap();
    let gf = lf.backward().unwrap();
    let pf = model.predict(&batch, Decode::Greedy, None).unwrap();
    set_fused_entity_model(false);
    assert!(
        rel_diff(&lc.to_f32(), &lf.to_f32()) < 1e-6,
        "crew={crew}: loss {} vs {}",
        lc.to_f32()[0],
        lf.to_f32()[0]
    );
    for (name, p) in model.named_parameters() {
        let a = gc.get(p.id()).unwrap().to_f32();
        let b = gf.get(p.id()).unwrap().to_f32();
        assert!(rel_diff(&a, &b) < 1e-5, "crew={crew}: grad {name}");
    }
    for (name, t) in &pc.logits {
        assert!(
            rel_diff(&t.to_f32(), &pf.logits[name].to_f32()) < 1e-5,
            "crew={crew}: predict {name}"
        );
    }
    for (name, t) in &pc.choices {
        assert_eq!(
            t.to_vec(),
            pf.choices[name].to_vec(),
            "crew={crew}: choices {name}"
        );
    }
}

#[test]
fn model_fused_composed_parity() {
    model_parity(false);
    model_parity(true);
}

#[test]
fn k2_assemble_matches_composed() {
    let device = dev();
    let (b, m, k, d) = (2usize, 3, 2, 8);
    let base_d = frand(b * m * d, 701);
    let step_d = frand(k * d, 702);
    let extra_d = frand(b * m * k * d, 703);
    let run = |fused: bool| {
        let base =
            Var::traced(Tensor::<R, f32>::from_f32(&base_d, vec![b, m, d], &device).unwrap());
        let step = Var::traced(Tensor::<R, f32>::from_f32(&step_d, vec![k, d], &device).unwrap());
        let extra =
            Var::traced(Tensor::<R, f32>::from_f32(&extra_d, vec![b, m, k, d], &device).unwrap());
        let y = if fused {
            Var::assemble_queries(&base, &step, &extra, true).unwrap()
        } else {
            // Composed step-major assembly.
            let mut blocks = Vec::with_capacity(k);
            for j in 0..k {
                let mut rows = Vec::with_capacity(m);
                for mi in 0..m {
                    let bm = base
                        .slice(1, mi, 1)
                        .unwrap()
                        .reshape(vec![b, 1, d])
                        .unwrap();
                    let sj = step.slice(0, j, 1).unwrap().reshape(vec![1, 1, d]).unwrap();
                    let emj = extra
                        .slice(1, mi, 1)
                        .unwrap()
                        .slice(2, j, 1)
                        .unwrap()
                        .reshape(vec![b, 1, d])
                        .unwrap();
                    rows.push(bm.add(&sj).unwrap().add(&emj).unwrap());
                }
                blocks.push(mamba3::autograd::cat(&rows, 1).unwrap());
            }
            mamba3::autograd::cat(&blocks, 1).unwrap()
        };
        assert_eq!(y.shape().dims(), &[b, k * m, d]);
        let loss = y.sum().unwrap();
        let grads = loss.backward_retain().unwrap();
        let gb = grads.node(base.node().unwrap()).unwrap().to_f32();
        let gs = grads.node(step.node().unwrap()).unwrap().to_f32();
        let ge = grads.node(extra.node().unwrap()).unwrap().to_f32();
        (y.to_f32(), gb, gs, ge)
    };
    let (yf, gbf, gsf, gef) = run(true);
    let (yc, gbc, gsc, gec) = run(false);
    assert!(rel_diff(&yf, &yc) < 1e-6, "k2 assemble forward");
    assert!(rel_diff(&gbf, &gbc) < 1e-5, "k2 d_base");
    assert!(rel_diff(&gsf, &gsc) < 1e-5, "k2 d_step");
    assert!(rel_diff(&gef, &gec) < 1e-5, "k2 d_extra");
}

#[test]
fn k2_choice_tokens_match_host_mapping() {
    let device = dev();
    let (b, s, d, r) = (1usize, 6, 8, 4);
    let table_d = frand(b * s * d, 711);
    // Ids with IGNORE: must fall back to the none slot (s - 1).
    let ids: Vec<u32> = vec![0, IGNORE, 5, 2];
    let run = |fused: bool| {
        let table =
            Var::traced(Tensor::<R, f32>::from_f32(&table_d, vec![b, s, d], &device).unwrap());
        let y = if fused {
            let dev = IdTensor::from_slice(&ids, vec![b * r], &device).unwrap();
            Var::gather_choice(&table, &dev, r, s - 1).unwrap()
        } else {
            // Composed host mapping (what choice_tokens does).
            let mapped: Vec<u32> = ids
                .iter()
                .map(|&id| if id == IGNORE { (s - 1) as u32 } else { id })
                .collect();
            let dev = IdTensor::from_slice(&mapped, vec![b * r], &device).unwrap();
            Var::gather_tokens(&table, &dev, r).unwrap()
        };
        let loss = y.sum().unwrap();
        let grads = loss.backward_retain().unwrap();
        (
            y.to_f32(),
            grads.node(table.node().unwrap()).unwrap().to_f32(),
        )
    };
    let (yf, gf) = run(true);
    let (yc, gc) = run(false);
    assert_eq!(yf, yc, "k2 gather_choice forward is exact");
    assert_eq!(gf, gc, "k2 gather_choice backward is exact");
    // The IGNORE row really is the none slot.
    assert_eq!(&yf[d..2 * d], &table_d[(s - 1) * d..s * d]);
}

#[test]
fn k2_prev_tokens_match_host_shifting() {
    let device = dev();
    let (b, m, k, s, d) = (1usize, 2, 3, 7, 8);
    let table_d = frand(b * s * d, 721);
    let ids: Vec<u32> = vec![1, 2, 3, IGNORE, 0, 4];
    let lag = 2;
    let run = |fused: bool| {
        let table =
            Var::traced(Tensor::<R, f32>::from_f32(&table_d, vec![b, s, d], &device).unwrap());
        let dev = IdTensor::from_slice(&ids, vec![b * m * k], &device).unwrap();
        let y = if fused {
            Var::prev_choice(&table, &dev, m, k, lag, s - 1).unwrap()
        } else {
            // Composed host shifting (what prev_tokens does).
            let mut shifted = vec![IGNORE; b * m * k];
            for mi in 0..m {
                for j in lag..k {
                    shifted[mi * k + j] = ids[mi * k + (j - lag)];
                }
            }
            let mapped: Vec<u32> = shifted
                .iter()
                .map(|&id| if id == IGNORE { (s - 1) as u32 } else { id })
                .collect();
            let sh = IdTensor::from_slice(&mapped, vec![b * m * k], &device).unwrap();
            Var::gather_tokens(&table, &sh, m * k)
                .unwrap()
                .reshape(vec![b, m, k, d])
                .unwrap()
        };
        let loss = y.sum().unwrap();
        let grads = loss.backward_retain().unwrap();
        (
            y.to_f32(),
            grads.node(table.node().unwrap()).unwrap().to_f32(),
        )
    };
    let (yf, gf) = run(true);
    let (yc, gc) = run(false);
    assert_eq!(yf, yc, "k2 prev_choice forward is exact");
    assert_eq!(gf, gc, "k2 prev_choice backward is exact");
}

fn k1_arrays(s: usize) -> HostArrays {
    let (m, k, q) = (2usize, 2, 4);
    let mut a = HostArrays::new();
    a.insert_f32("cells", vec![s, 9, 4], frand(s * 9 * 4, 801));
    a.insert_f32("items", vec![s, 4, 2], frand(s * 4 * 2, 802));
    a.insert_f32("globals", vec![s, 3], frand(s * 3, 803));
    a.insert_f32("agents", vec![s, m, 3], frand(s * m * 3, 804));
    let mut rng = 805u64;
    let mut ri = |n: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % n as u64) as i64
    };
    let mut anchor = Vec::with_capacity(s * m);
    let mut tgt = Vec::with_capacity(s * q);
    let mut kind = Vec::with_capacity(s * q);
    let mut eta = Vec::with_capacity(s * m);
    for _ in 0..s {
        for _ in 0..m {
            anchor.push(if ri(5) == 0 { -1 } else { ri(9) });
        }
        for _ in 0..q {
            let r = ri(10);
            tgt.push(if r < 8 { ri(10) } else { -1 });
            kind.push(if ri(4) == 0 { -1 } else { ri(2) });
        }
        for _ in 0..m {
            eta.push(if ri(3) == 0 {
                f32::NAN
            } else {
                (ri(10) as f32) * 0.1
            });
        }
    }
    a.insert_int("agents.anchor", vec![s, m], anchor);
    a.insert_int("label.tgt", vec![s, m, k], tgt);
    a.insert_int("label.kind", vec![s, m, k], kind);
    a.insert_f32("label.eta", vec![s, m, 1], eta);
    a
}

fn assert_batches_equal(spec: &EntityModelSpec, a: &EntityBatch<R, f32>, b: &EntityBatch<R, f32>) {
    use mamba3::models::entity::HeadLabels;
    assert_eq!(a.b, b.b);
    for (x, y) in a.ctx_feats.iter().zip(b.ctx_feats.iter()) {
        assert_eq!(x.to_f32(), y.to_f32(), "ctx feats");
    }
    for (x, y) in a.ctx_presence.iter().zip(b.ctx_presence.iter()) {
        assert_eq!(x.to_f32(), y.to_f32(), "ctx presence");
    }
    assert_eq!(
        a.globals.as_ref().map(|t| t.to_f32()),
        b.globals.as_ref().map(|t| t.to_f32())
    );
    assert_eq!(
        a.q_feats.as_ref().map(|t| t.to_f32()),
        b.q_feats.as_ref().map(|t| t.to_f32())
    );
    for head in &spec.heads {
        match (a.labels[&head.name].as_ref(), b.labels[&head.name].as_ref()) {
            (
                Some(HeadLabels::Class { ids, keep, div }),
                Some(HeadLabels::Class {
                    ids: ids2,
                    keep: keep2,
                    div: div2,
                }),
            ) => {
                assert_eq!(ids.to_vec(), ids2.to_vec(), "class ids {}", head.name);
                assert_eq!(keep.to_f32(), keep2.to_f32(), "keep {}", head.name);
                assert!(
                    (div - div2).abs() < 1e-6,
                    "div {}: {div} vs {div2}",
                    head.name
                );
            }
            (
                Some(HeadLabels::Multi { targets, keep, div }),
                Some(HeadLabels::Multi {
                    targets: t2,
                    keep: k2,
                    div: d2,
                }),
            ) => {
                assert_eq!(targets.to_f32(), t2.to_f32(), "targets {}", head.name);
                assert_eq!(keep.to_f32(), k2.to_f32(), "keep {}", head.name);
                assert!((div - d2).abs() < 1e-6, "div {}", head.name);
            }
            (
                Some(HeadLabels::Reg { targets, keep, div }),
                Some(HeadLabels::Reg {
                    targets: t2,
                    keep: k2,
                    div: d2,
                }),
            ) => {
                assert_eq!(targets.to_f32(), t2.to_f32(), "targets {}", head.name);
                assert_eq!(keep.to_f32(), k2.to_f32(), "keep {}", head.name);
                assert!((div - d2).abs() < 1e-6, "div {}", head.name);
            }
            (None, None) => {}
            _ => panic!("label shape mismatch for {}", head.name),
        }
    }
    for (name, ids) in &a.choice_dev {
        assert_eq!(ids.to_vec(), b.choice_dev[name].to_vec(), "choice {name}");
    }
}

#[test]
fn k1_dataset_matches_host_and_reads_nothing() {
    use mamba3::models::entity::{EntityDataset, set_fused_entity_model};
    let _guard = SWITCH_LOCK.lock().unwrap();
    let device = dev();
    let spec = tiny_spec(true);
    let arrays = k1_arrays(4);
    let data = EntityDataset::<R, f32>::from_arrays(&spec, &arrays, &device).unwrap();
    let host = EntityBatch::<R, f32>::from_host(&spec, &arrays, &device).unwrap();
    set_fused_entity_model(true);
    // Warm up once (lazy init), then measure: gather + loss + backward.
    let all: Vec<u32> = vec![0, 1, 2, 3];
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let task = EntityTask::new(&model);
    let warm = EntityBatch::<R, f32>::from_ids(&spec, &data, &all).unwrap();
    task.loss(&warm).unwrap().backward().unwrap();
    let before = mamba3::backend::read_count();
    let res = EntityBatch::<R, f32>::from_ids(&spec, &data, &all).unwrap();
    let loss = task.loss(&res).unwrap();
    loss.backward().unwrap();
    assert_eq!(
        mamba3::backend::read_count() - before,
        0,
        "resident step reads"
    );
    // Comparisons read back; measure nothing after this point.
    assert_batches_equal(&spec, &host, &res);
    set_fused_entity_model(false);
    // Same loss both ways (divisors may differ by summation order: 1e-6).
    let lh = task.loss(&host).unwrap().to_f32()[0];
    set_fused_entity_model(true);
    let lr = task.loss(&res).unwrap().to_f32()[0];
    set_fused_entity_model(false);
    assert!(
        (lh - lr).abs() / (1.0 + lh.abs()) < 1e-6,
        "loss {lh} vs {lr}"
    );
    // A shuffled subset gathers the right rows.
    let sub = EntityBatch::<R, f32>::from_ids(&spec, &data, &[3, 1]).unwrap();
    assert_eq!(sub.b, 2);
    let sub_host_arrays = k1_sub(&arrays, &[3, 1]);
    let sub_host = EntityBatch::<R, f32>::from_host(&spec, &sub_host_arrays, &device).unwrap();
    assert_batches_equal(&spec, &sub_host, &sub);
}

/// Slice samples out of host arrays (test helper).
fn k1_sub(a: &HostArrays, keep: &[usize]) -> HostArrays {
    let mut out = HostArrays::new();
    for (k, (shape, data)) in &a.f32s {
        let b = shape[0];
        let row: usize = shape[1..].iter().product();
        let _ = b;
        let mut v = Vec::with_capacity(keep.len() * row);
        for &s in keep {
            v.extend_from_slice(&data[s * row..(s + 1) * row]);
        }
        let mut ns = vec![keep.len()];
        ns.extend_from_slice(&shape[1..]);
        out.insert_f32(k, ns, v);
    }
    for (k, (shape, data)) in &a.ints {
        let row: usize = shape[1..].iter().product();
        let mut v = Vec::with_capacity(keep.len() * row);
        for &s in keep {
            v.extend_from_slice(&data[s * row..(s + 1) * row]);
        }
        let mut ns = vec![keep.len()];
        ns.extend_from_slice(&shape[1..]);
        out.insert_int(k, ns, v);
    }
    out
}

#[test]
fn k3_rows_match_host_reference() {
    // One head of each fused kind, computed by hand on the host.
    let device = dev();
    let (r, wc, wu, wp, wf, h) = (3usize, 2, 3, 4, 5, 3);
    // Heads: CE over uncond[0..3], BCE over cond[0..2], MSE over ptr[1..3].
    let seg = vec![0u32, 3, 1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 2, 1, 2];
    let cond = frand(r * wc, 901);
    let uncond = frand(r * wu, 902);
    let ptr = frand(r * wp, 903);
    let class_ids: Vec<u32> = vec![1, 0, 2, 0, 0, 0, 0, 0, 0];
    let keep: Vec<f32> = vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let ft: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, -0.3, 0.0,
    ];
    let inv_width: Vec<f32> = vec![1.0 / 3.0, 1.0 / 2.0, 1.0 / 2.0];
    let rows = mamba3::tensor::ops::entity_model::seg_loss_rows(
        &Tensor::<R, f32>::from_f32(&cond, vec![r, wc], &device).unwrap(),
        &Tensor::<R, f32>::from_f32(&uncond, vec![r, wu], &device).unwrap(),
        &Tensor::<R, f32>::from_f32(&ptr, vec![r, wp], &device).unwrap(),
        &IdTensor::from_slice(&class_ids, vec![r, h], &device).unwrap(),
        &Tensor::<R, f32>::from_f32(&keep, vec![r, h], &device).unwrap(),
        &Tensor::<R, f32>::from_f32(&ft, vec![r, wf], &device).unwrap(),
        &IdTensor::from_slice(&seg, vec![h, 5], &device).unwrap(),
        &Tensor::<R, f32>::from_f32(&inv_width, vec![h], &device).unwrap(),
    )
    .unwrap()
    .to_f32();
    // Host reference in f64.
    let lse = |xs: &[f64]| {
        let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
    };
    let mut want = vec![0.0f64; r * h];
    for rr in 0..r {
        // CE head 0 over uncond[0..3], id class_ids[rr*3].
        if keep[rr * h] != 0.0 {
            let row: Vec<f64> = (0..3).map(|c| uncond[rr * wu + c] as f64).collect();
            want[rr * h] = lse(&row) - row[class_ids[rr * h] as usize];
        }
        // BCE head 1 over cond[0..2], targets ft[0..2].
        if keep[rr * h + 1] != 0.0 {
            let mut bce = 0.0;
            for c in 0..2 {
                let x = cond[rr * wc + c] as f64;
                let y = ft[rr * wf + c] as f64;
                bce += x.max(0.0) + (-x.abs()).exp().ln_1p() - x * y;
            }
            want[rr * h + 1] = bce / 2.0;
        }
        // MSE head 2 over ptr[1..3], targets ft[2..4].
        if keep[rr * h + 2] != 0.0 {
            let mut mse = 0.0;
            for c in 0..2 {
                let d = ptr[rr * wp + 1 + c] as f64 - ft[rr * wf + 2 + c] as f64;
                mse += d * d;
            }
            want[rr * h + 2] = mse / 2.0;
        }
    }
    for (i, (g, w)) in rows.iter().zip(want.iter()).enumerate() {
        assert!(
            (*g as f64 - *w).abs() < 1e-4,
            "row {i}: device {g} != host {w}"
        );
    }
}

#[test]
fn k3_fused_loss_matches_composed() {
    use mamba3::models::entity::{EntityDataset, set_fused_entity_model};
    let _guard = SWITCH_LOCK.lock().unwrap();
    let device = dev();
    // All four head kinds, one First head, IGNORE/NaN rows, absent entity.
    let spec = tiny_spec(true);
    let arrays = k1_arrays(2);
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let task = EntityTask::new(&model);
    let host = EntityBatch::<R, f32>::from_host(&spec, &arrays, &device).unwrap();
    let data = EntityDataset::<R, f32>::from_arrays(&spec, &arrays, &device).unwrap();
    let res = EntityBatch::<R, f32>::from_ids(&spec, &data, &[0, 1]).unwrap();

    set_fused_entity_model(false);
    let lc = task.loss(&host).unwrap();
    let gc = lc.backward().unwrap();
    let comps = task.component_losses(&host).unwrap();
    let comp_val = |name: &str| comps[name].to_f32()[0];

    set_fused_entity_model(true);
    for (name, batch) in [("host", &host), ("resident", &res)] {
        let lf = task.loss(batch).unwrap();
        assert!(
            rel_diff(&lc.to_f32(), &lf.to_f32()) < 1e-6,
            "{name}: loss {} vs {}",
            lc.to_f32()[0],
            lf.to_f32()[0]
        );
        let gf = lf.backward().unwrap();
        for (pname, p) in model.named_parameters() {
            let a = gc.get(p.id()).unwrap().to_f32();
            let b = gf.get(p.id()).unwrap().to_f32();
            assert!(rel_diff(&a, &b) < 1e-5, "{name}: grad {pname}");
        }
        // The fused report matches the composed components (report holds
        // keep-weighted sums; divide by the batch divisors for the means).
        let report = task.fused_report(batch).unwrap().unwrap().to_f32();
        let divs = batch.seg.as_ref().unwrap().divs.clone();
        assert_eq!(report.len(), comps.len() - 1, "{name}: report width");
        let mut ci = 0;
        for run in model.head_runs() {
            use mamba3::models::entity::StepSelection;
            if matches!(run.steps, StepSelection::First) {
                continue;
            }
            let mean = report[ci] / divs[ci];
            let want = comp_val(&run.name);
            assert!(
                (mean - want).abs() < 1e-4,
                "{name}: head {} report {mean} vs {want}",
                run.name,
            );
            ci += 1;
        }
    }
    set_fused_entity_model(false);
}

#[test]
fn k3_multisource_falls_back_to_composed() {
    use mamba3::models::entity::set_fused_entity_model;
    let _guard = SWITCH_LOCK.lock().unwrap();
    let device = dev();
    // Two conditioned heads on DIFFERENT pointers: fused loss must fall back
    // to the composed path and still agree exactly.
    let mut spec = tiny_spec(false);
    spec.heads.push(
        HeadSpec::pointer("tgt2", "items", 0),
    );
    spec.heads.push(
        HeadSpec::categorical("kind2", 2).condition_on("tgt2"),
    );
    spec.validate().unwrap();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let mut arrays = tiny_arrays();
    arrays.ints.insert(
        "label.tgt2".to_string(),
        (vec![1, 2, 2], vec![0, 1, 2, 3]),
    );
    arrays.ints.insert(
        "label.kind2".to_string(),
        (vec![1, 2, 2], vec![0, 1, 1, 0]),
    );
    let batch = EntityBatch::<R, f32>::from_host(&spec, &arrays, &device).unwrap();
    let task = EntityTask::new(&model);
    set_fused_entity_model(false);
    let lc = task.loss(&batch).unwrap().to_f32()[0];
    set_fused_entity_model(true);
    let lf = task.loss(&batch).unwrap().to_f32()[0];
    set_fused_entity_model(false);
    assert!(
        (lc - lf).abs() / (1.0 + lc.abs()) < 1e-6,
        "multisource fallback {lc} vs {lf}"
    );
}
