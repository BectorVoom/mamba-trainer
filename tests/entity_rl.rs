//! R1 tests (ENTITY_RL_PLAN.md §2 R1): device-resident sampling,
//! teacher-forced rescoring, PPO gradients, bandit learning, save/load, GAE
//! and the on-device pins. CPU and CUDA share the file (`Auto` backend).

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::model::{Decode, EntityModel};
use mamba3::models::entity::rl::{
    EntityActorCritic, EntityPpoBatch, EntityPpoTask, actions_to_arrays, check_rl_support,
    entity_gae, entity_ppo_objective,
};
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityDataset, EntityModelSpec, HeadSpec,
    HostArrays, QuerySetSpec,
};
use mamba3::nn::Module;
use mamba3::rl::PpoConfig;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::entity_model::{IGNORE, mask_absent_ids};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

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

fn small_ssm() -> mamba3::ssm::config::SsmConfig {
    mamba3::ssm::config::SsmConfig {
        d_model: 16,
        n_heads: 2,
        head_dim: 8,
        d_state: 8,
        n_groups: 2,
        chunk_size: 4,
        ..Default::default()
    }
}

/// Tiny RL spec: 6 context entities, 2 queries, K = 2, a pointer plus a
/// conditioned categorical (both actions) plus a multilabel head (not an
/// action), StepCausal, d_model 16.
fn tiny_spec(decoder: DecoderMode, autoregressive: bool, seed: u64) -> EntityModelSpec {
    let queries = if autoregressive {
        QuerySetSpec::new("agents", 2, 3, 2).with_autoregressive("pick")
    } else {
        QuerySetSpec::new("agents", 2, 3, 2)
    };
    EntityModelSpec {
        globals: 0,
        context: vec![ContextSetSpec::new("cells", 6, 4)],
        queries: Some(queries),
        heads: vec![
            HeadSpec::pointer("pick", "cells", 0),
            HeadSpec::categorical("cfg", 3).condition_on("pick"),
            HeadSpec::multilabel("flags", 2),
        ],
        d_model: 16,
        context_layers: 1,
        decoder_layers: 1,
        decoder,
        ssm: small_ssm(),
        chunk_size: None,
        norm_eps: 1e-5,
        seed,
    }
}

fn step_causal_spec() -> EntityModelSpec {
    tiny_spec(
        DecoderMode::StepCausal {
            crew_symmetric: false,
        },
        true,
        11,
    )
}

/// Plain observations: everything present, no legal masks.
fn plain_obs(b: usize, seed: u64) -> HostArrays {
    let mut a = HostArrays::new();
    a.insert_f32("cells", vec![b, 6, 4], frand(b * 24, seed));
    a.insert_f32("agents", vec![b, 2, 3], frand(b * 6, seed + 1));
    a
}

/// `act` at T = 1, read back, re-uploaded as labels, re-scored: the per-cell
/// log-probabilities and entropies must agree within 1e-4 (this is what makes
/// the PPO ratio exactly 1 before the first update).
fn rescore_gap(spec: &EntityModelSpec, b: usize, seed: u64) -> (f32, f32) {
    let device = dev();
    let ac = EntityActorCritic::<R, f32>::init(spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(spec, &plain_obs(b, seed), &device).unwrap();
    let acted = ac.act(&batch, 1.0, seed + 100).unwrap().read().unwrap();
    let mut full = plain_obs(b, seed);
    for (key, (shape, data)) in actions_to_arrays(&acted, spec).ints {
        full.insert_int(&key, shape, data);
    }
    let scored = EntityBatch::<R, f32>::from_host(spec, &full, &device).unwrap();
    let (lp, ent, _) = ac.evaluate_actions(&scored, 1.0).unwrap();
    let (lp, ent) = (lp.to_f32(), ent.to_f32());
    let gap = |a: &[f32], c: &[f32]| {
        a.iter()
            .zip(c.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    (gap(&lp, &acted.cell_log_prob), gap(&ent, &acted.cell_entropy))
}

#[test]
fn teacher_forcing_equals_decoding() {
    // StepCausal with autoregression: the planned case.
    let (lp_gap, ent_gap) = rescore_gap(
        &tiny_spec(
            DecoderMode::StepCausal {
                crew_symmetric: false,
            },
            true,
            21,
        ),
        4,
        31,
    );
    assert!(lp_gap < 1e-4, "step-causal log-prob gap {lp_gap}");
    assert!(ent_gap < 1e-4, "step-causal entropy gap {ent_gap}");
    // No autoregression (Joint): steps are independent, rescoring is exact.
    let (lp_gap, ent_gap) = rescore_gap(&tiny_spec(DecoderMode::Joint, false, 22), 4, 33);
    assert!(lp_gap < 1e-4, "joint log-prob gap {lp_gap}");
    assert!(ent_gap < 1e-4, "joint entropy gap {ent_gap}");
    // At T = 0 act's pointer ids equal greedy predict's on present cells.
    let device = dev();
    let spec = step_causal_spec();
    let ac = EntityActorCritic::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &plain_obs(2, 37), &device).unwrap();
    let acted = ac.act(&batch, 0.0, 5).unwrap().read().unwrap();
    let greedy = ac.model().predict(&batch, Decode::Greedy, None).unwrap();
    assert_eq!(
        acted.actions["pick"],
        greedy.choices["pick"]
            .to_vec()
            .into_iter()
            .map(|v| v as i64)
            .collect::<Vec<_>>()
    );
    // Joint + autoregressive is refused, naming the decoder mode (the spec
    // itself cannot even be built, so the check runs on the spec directly).
    let bad = tiny_spec(DecoderMode::Joint, true, 23);
    let err = check_rl_support(&bad)
        .expect_err("Joint + autoregressive must be refused")
        .to_string();
    assert!(err.contains("Joint"), "refusal names the decoder mode: {err}");
    // QueryCausal + autoregressive is refused the same way.
    let qc = tiny_spec(DecoderMode::QueryCausal, true, 24);
    let err = check_rl_support(&qc)
        .expect_err("QueryCausal must be refused")
        .to_string();
    assert!(
        err.contains("QueryCausal"),
        "refusal names the decoder mode: {err}"
    );
}

#[test]
fn sampling_respects_masks() {
    let device = dev();
    let spec = step_causal_spec();
    let ac = EntityActorCritic::<R, f32>::init(&spec, &device).unwrap();
    let (b, n, m) = (3usize, 6usize, 2usize);
    let mut a = HostArrays::new();
    a.insert_f32("cells", vec![b, n, 4], frand(b * n * 4, 41));
    a.insert_f32(
        "cells.presence",
        vec![b, n],
        vec![
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, //
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    a.insert_f32("agents", vec![b, m, 3], frand(b * m * 3, 42));
    // Query 1 of sample 1 is absent: it must never act.
    a.insert_f32(
        "agents.presence",
        vec![b, m],
        vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    );
    // Per-query legal masks (every row keeps at least one entity legal).
    let mut legal = vec![0.0f32; b * m * n];
    let mut s = 43u64;
    for v in legal.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = (s % 2) as f32;
    }
    for bi in 0..b {
        for mi in 0..m {
            let row = (bi * m + mi) * n;
            if legal[row..row + n].iter().all(|&v| v == 0.0) {
                legal[row] = 1.0;
            }
        }
    }
    a.insert_f32("legal.pick", vec![b, m, n], legal.clone());
    let batch = EntityBatch::<R, f32>::from_host(&spec, &a, &device).unwrap();
    let presence: Vec<f32> = vec![
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    for seed in 0..30u64 {
        let acted = ac.act(&batch, 1.0, seed).unwrap().read().unwrap();
        for bi in 0..b {
            for mi in 0..m {
                let absent_query = bi == 1 && mi == 1;
                for j in 0..2 {
                    let cell = (bi * m + mi) * 2 + j;
                    for head in ["pick", "cfg"] {
                        let id = acted.actions[head][cell];
                        if absent_query {
                            assert_eq!(id, -1, "absent query acted at seed {seed}");
                            continue;
                        }
                        if head == "pick" {
                            assert!(
                                (0..n as i64).contains(&id),
                                "pointer id {id} out of range at seed {seed}"
                            );
                            let id = id as usize;
                            assert_eq!(
                                presence[bi * n + id],
                                1.0,
                                "sampled absent entity {id} at seed {seed}"
                            );
                            assert_eq!(
                                legal[(bi * m + mi) * n + id],
                                1.0,
                                "sampled illegal entity {id} at seed {seed}"
                            );
                        } else {
                            assert!(
                                (0..3).contains(&id),
                                "categorical id {id} out of range at seed {seed}"
                            );
                        }
                    }
                }
            }
        }
        // Absent queries accumulate no log-probability and stay masked out.
        // Sample 1, query 1 is the absent one.
        let (absent_b, absent_q) = (1usize, 1usize);
        for j in 0..2 {
            let cell = (absent_b * m + absent_q) * 2 + j;
            assert_eq!(acted.cell_log_prob[cell], 0.0, "absent cell scored at seed {seed}");
            assert_eq!(acted.cell_mask[cell], 0.0, "absent cell masked in at seed {seed}");
        }
    }
    // T = 0 is deterministic and equals greedy predict on present cells.
    let g0 = ac.act(&batch, 0.0, 7).unwrap().read().unwrap();
    let g1 = ac.act(&batch, 0.0, 999).unwrap().read().unwrap();
    assert_eq!(g0.actions, g1.actions, "T = 0 depends on the seed");
    let greedy = ac.model().predict(&batch, Decode::Greedy, None).unwrap();
    let got = greedy.choices["pick"].to_vec();
    for bi in 0..b {
        for mi in 0..m {
            for j in 0..2 {
                let cell = (bi * m + mi) * 2 + j;
                if bi == 1 && mi == 1 {
                    assert_eq!(g0.actions["pick"][cell], -1);
                } else {
                    assert_eq!(
                        g0.actions["pick"][cell],
                        got[cell] as i64,
                        "T = 0 differs from greedy at cell {cell}"
                    );
                }
            }
        }
    }
}

#[test]
fn gradients_reach_every_parameter() {
    // An all-action spec: every parameter serves an action head or the
    // critic. (Multilabel/regression heads get no policy gradient by design,
    // so a spec whose only unconditioned head is one of those would leave its
    // shared Linear without a gradient.)
    let device = dev();
    let spec = EntityModelSpec {
        globals: 2,
        context: vec![ContextSetSpec::new("cells", 6, 4)],
        queries: Some(QuerySetSpec::new("agents", 2, 3, 2).with_autoregressive("pick")),
        heads: vec![
            HeadSpec::pointer("pick", "cells", 0),
            HeadSpec::categorical("cfg", 3).condition_on("pick"),
        ],
        d_model: 16,
        context_layers: 1,
        decoder_layers: 1,
        decoder: DecoderMode::StepCausal {
            crew_symmetric: false,
        },
        ssm: small_ssm(),
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 50,
    };
    let ac = EntityActorCritic::<R, f32>::init(&spec, &device).unwrap();
    let (b, m, k) = (2usize, 2usize, 2usize);
    // Labels with some IGNORE cells (exercising none_prev) and one absent
    // query whose labels are all IGNORE.
    let mut a = HostArrays::new();
    a.insert_f32("cells", vec![b, 6, 4], frand(b * 24, 51));
    a.insert_f32("globals", vec![b, 2], frand(b * 2, 52));
    a.insert_f32("agents", vec![b, m, 3], frand(b * m * 3, 53));
    a.insert_f32("agents.presence", vec![b, m], vec![1.0, 1.0, 1.0, 0.0]);
    let mut s = 54u64;
    let mut ri = |modulus: i64| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % (modulus as u64 + 1)) as i64 - 1
    };
    let mut pick: Vec<i64> = (0..b * m * k).map(|_| ri(6)).collect();
    let mut cfg: Vec<i64> = (0..b * m * k).map(|_| ri(3)).collect();
    for cell in m * k..b * m * k {
        pick[cell] = -1;
        cfg[cell] = -1;
    }
    a.insert_int("label.pick", vec![b, m, k], pick);
    a.insert_int("label.cfg", vec![b, m, k], cfg);
    let batch = EntityBatch::<R, f32>::from_host(&spec, &a, &device).unwrap();
    let up =
        |v: Vec<f32>, shape: Vec<usize>| Tensor::<R, f32>::from_f32(&v, shape, &device).unwrap();
    let ppo = EntityPpoBatch {
        batch,
        old_log_prob: up(vec![0.0; b * m * k], vec![b, m, k]),
        cell_mask: up(vec![1.0; b * m * k], vec![b, m, k]),
        advantages: up(frand(b, 55), vec![b]),
        returns: up(frand(b, 56), vec![b]),
        old_values: up(vec![0.0; b], vec![b]),
    };
    let loss = entity_ppo_objective(&ac, &ppo, &PpoConfig::default(), 1.0).unwrap();
    let grads = loss.total.backward().unwrap();
    for (name, p) in ac.named_parameters() {
        let g = grads
            .get(p.id())
            .unwrap_or_else(|| panic!("no gradient reached {name}"));
        let norm: f32 = g.to_f32().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "zero gradient at {name}");
    }
}

// ---------------------------------------------------------------------------
// Contextual bandit: one target entity per sample is marked by feature 0 = 1;
// reward is the share of queries whose step-0 pointer names it.
// ---------------------------------------------------------------------------

const BANDIT_N: usize = 6;

fn bandit_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 0,
        context: vec![ContextSetSpec::new("field", BANDIT_N, 2)],
        queries: Some(QuerySetSpec::new("q", 2, 1, 1)),
        heads: vec![HeadSpec::pointer("pick", "field", 0)],
        d_model: 16,
        context_layers: 1,
        decoder_layers: 1,
        decoder: DecoderMode::Joint,
        ssm: small_ssm(),
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 7,
    }
}

fn bandit_arrays(b: usize, seed: u64) -> (HostArrays, Vec<usize>) {
    let mut s = seed.max(1);
    let mut next = |modulus: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % modulus as u64) as usize
    };
    let targets: Vec<usize> = (0..b).map(|_| next(BANDIT_N)).collect();
    let mut ctx = vec![0.0f32; b * BANDIT_N * 2];
    for (bi, &target) in targets.iter().enumerate() {
        for e in 0..BANDIT_N {
            let at = (bi * BANDIT_N + e) * 2;
            ctx[at] = if e == target { 1.0 } else { 0.0 };
            ctx[at + 1] = (next(1000) as f32 / 500.0) - 1.0;
        }
    }
    let mut a = HostArrays::new();
    a.insert_f32("field", vec![b, BANDIT_N, 2], ctx);
    a.insert_f32("q", vec![b, 2, 1], vec![0.0; b * 2]);
    (a, targets)
}

fn bandit_rewards(
    actions: &std::collections::BTreeMap<String, Vec<i64>>,
    targets: &[usize],
) -> Vec<f32> {
    let pick = &actions["pick"];
    targets
        .iter()
        .enumerate()
        .map(|(bi, &target)| {
            let hits = (0..2).filter(|&mi| pick[bi * 2 + mi] == target as i64).count();
            hits as f32 / 2.0
        })
        .collect()
}

#[test]
fn ppo_learns_the_bandit() {
    let device = dev();
    let spec = bandit_spec();
    let ac = EntityActorCritic::<R, f32>::init(&spec, &device).unwrap();
    let config = PpoConfig::default().with_clip(0.2).with_coefficients(0.5, 0.01);
    let task = EntityPpoTask::new(&ac, config, 1.0);
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(3e-3)
            .max_grad_norm(0.5)
            .build()
            .unwrap(),
        AdamWConfig::builder().learning_rate(3e-3).build().init::<R, f32>(),
    );
    let (b, m, k) = (64usize, 2usize, 1usize);
    let mut first = 0.0f32;
    let mut done_at = 60usize;
    for update in 0..60usize {
        let (obs, targets) = bandit_arrays(b, 1000 + update as u64);
        let batch0 = EntityBatch::<R, f32>::from_host(&spec, &obs, &device).unwrap();
        let acted = ac.act(&batch0, 1.0, 5000 + update as u64).unwrap().read().unwrap();
        let rewards = bandit_rewards(&acted.actions, &targets);
        let mean: f32 = rewards.iter().sum::<f32>() / b as f32;
        if update == 0 {
            first = mean;
        }
        if mean > 0.8 {
            done_at = update;
            break;
        }
        // One-step episodes: done = 1, last_value = 0.
        let (adv, ret): (Tensor<R, f32>, Tensor<R, f32>) = entity_gae(
            &rewards,
            &acted.value,
            &vec![1.0f32; b],
            &vec![0.0f32; b],
            1,
            b,
            0.99,
            0.95,
            &device,
        )
        .unwrap();
        // The rollout is uploaded once; minibatches gather on the device.
        let mut full = obs.clone();
        for (key, (shape, data)) in actions_to_arrays(&acted, &spec).ints {
            full.insert_int(&key, shape, data);
        }
        let data = EntityDataset::<R, f32>::from_arrays(&spec, &full, &device).unwrap();
        let up =
            |v: Vec<f32>, shape: Vec<usize>| Tensor::<R, f32>::from_f32(&v, shape, &device).unwrap();
        let full_lp = up(acted.cell_log_prob.clone(), vec![b, m, k]);
        let full_mask = up(acted.cell_mask.clone(), vec![b, m, k]);
        let full_val = up(acted.value.clone(), vec![b]);
        // 4 epochs x 2 shuffled minibatches, one optimizer step each.
        let mut idx: Vec<usize> = (0..b).collect();
        for epoch in 0..4usize {
            let mut s = (7000 + update as u64 * 10 + epoch as u64).max(1);
            for i in (1..idx.len()).rev() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                idx.swap(i, (s % (i as u64 + 1)) as usize);
            }
            for mb in idx
                .chunks(32)
                .map(|c| c.iter().map(|&i| i as u32).collect::<Vec<_>>())
                .collect::<Vec<_>>() {
                let ppo = EntityPpoBatch::from_rollout(
                    &spec, &data, &full_lp, &full_mask, &adv, &ret, &full_val, &mb,
                )
                .unwrap();
                trainer.step(&task, std::slice::from_ref(&ppo)).unwrap();
            }
        }
    }
    assert!(
        first < 0.5,
        "bandit started at {first:.3}, too good for chance (1/{BANDIT_N})"
    );
    assert!(
        done_at < 60,
        "bandit never passed 0.8 within 60 updates (started at {first:.3})"
    );
}

#[test]
fn save_load_round_trip() {
    let device = dev();
    let spec = step_causal_spec();
    let ac = EntityActorCritic::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &plain_obs(2, 61), &device).unwrap();
    let before = ac.act(&batch, 1.0, 77).unwrap().read().unwrap();
    let path = std::env::temp_dir().join("mamba3_entity_rl_roundtrip.m3ck");
    ac.save(&path, 3).unwrap();
    let loaded = EntityActorCritic::<R, f32>::load(&path, &device).unwrap();
    assert_eq!(loaded.model().spec(), &spec);
    let after = loaded.act(&batch, 1.0, 77).unwrap().read().unwrap();
    assert_eq!(before.actions, after.actions);
    assert_eq!(before.cell_log_prob, after.cell_log_prob);
    assert_eq!(before.value, after.value);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gae_matches_host_reference() {
    let device = dev();
    let (t, e) = (3usize, 5usize);
    let rewards = frand(t * e, 71);
    let values = frand(t * e, 72);
    let mut dones = vec![0.0f32; t * e];
    dones[e + 2] = 1.0; // an episode ends mid-rollout
    dones[2 * e + 4] = 1.0;
    let last_value = frand(e, 73);
    let (gamma, lambda) = (0.99f32, 0.9f32);
    let (adv, ret): (Tensor<R, f32>, Tensor<R, f32>) =
        entity_gae(&rewards, &values, &dones, &last_value, t, e, gamma, lambda, &device).unwrap();
    // Host reference in f64.
    let at = |v: &[f32], tt: usize, ei: usize| v[tt * e + ei] as f64;
    let mut want_adv = vec![0.0f64; t * e];
    let mut want_ret = vec![0.0f64; t * e];
    for ei in 0..e {
        let mut carry = 0.0f64;
        let mut next_v = last_value[ei] as f64;
        for tt in (0..t).rev() {
            let alive = 1.0 - dones[tt * e + ei] as f64;
            let delta = at(&rewards, tt, ei) + gamma as f64 * next_v * alive - at(&values, tt, ei);
            carry = delta + gamma as f64 * lambda as f64 * alive * carry;
            want_adv[tt * e + ei] = carry;
            want_ret[tt * e + ei] = carry + at(&values, tt, ei);
            next_v = at(&values, tt, ei);
        }
    }
    for (got, want) in adv
        .to_f32()
        .iter()
        .zip(want_adv.iter())
        .chain(ret.to_f32().iter().zip(want_ret.iter()))
    {
        assert!(
            (*got as f64 - *want).abs() < 1e-4,
            "gae mismatch: got {got}, want {want}"
        );
    }
}

#[test]
fn mask_absent_ids_matches_host() {
    // Parity for the one new kernel: absent rows become IGNORE with a zero
    // acted flag, present rows pass through (IGNORE stays IGNORE).
    let device = dev();
    let ids = vec![3u32, IGNORE, 0, 5, IGNORE, 1, 2, 4];
    let presence = vec![1.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let idt = IdTensor::<R>::from_slice(&ids, vec![8], &device).unwrap();
    let prest = Tensor::<R, f32>::from_f32(&presence, vec![8], &device).unwrap();
    let (masked, acted) = mask_absent_ids(&idt, &prest).unwrap();
    assert_eq!(
        masked.to_vec(),
        vec![3, IGNORE, IGNORE, 5, IGNORE, 1, 2, IGNORE]
    );
    assert_eq!(acted.to_f32(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn from_model_keeps_greedy_predictions() {
    // A behaviour-cloned model becomes the actor unchanged: greedy decoding
    // before and after `from_model` agrees.
    let device = dev();
    let spec = step_causal_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let batch = EntityBatch::<R, f32>::from_host(&spec, &plain_obs(2, 81), &device).unwrap();
    let before = model.predict(&batch, Decode::Greedy, None).unwrap();
    let ac = EntityActorCritic::<R, f32>::from_model(model).unwrap();
    let after = ac.model().predict(&batch, Decode::Greedy, None).unwrap();
    for (name, b) in &before.logits {
        assert_eq!(b.to_f32(), after.logits[name].to_f32(), "head {name}");
    }
}
