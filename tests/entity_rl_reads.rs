//! R1's on-device pins (ENTITY_RL_PLAN.md §1 "Read budget"): one `act` plus
//! its read is one device-to-host read, and a several-minibatch PPO update
//! whose numbers are read together is one read.
//!
//! Alone in its binary: `read_count` is process-wide, so any test running
//! beside this one in the same binary adds its own reads between the reset
//! and the check (inside tests/entity_rl.rs it read 63 for 1).

#![cfg(feature = "backend")]

use mamba3::backend::{Device, launch_count, read_count, reset_launch_count, reset_read_count};
use mamba3::backends::Auto;
use mamba3::models::entity::rl::{EntityActorCritic, EntityPpoBatch, EntityPpoTask, actions_to_arrays};
use mamba3::models::entity::{
    ContextSetSpec, DecoderMode, EntityBatch, EntityDataset, EntityModelSpec, HeadSpec,
    HostArrays, QuerySetSpec,
};
use mamba3::rl::PpoConfig;
use mamba3::tensor::Tensor;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

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


/// Second spec for the launch pin: 2 pointer heads each conditioning a
/// categorical head, K = 3, no autoregression (one decode pass for all steps).
fn two_ptr_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 0,
        context: vec![ContextSetSpec::new("cells", 6, 4)],
        queries: Some(QuerySetSpec::new("agents", 2, 3, 3)),
        heads: vec![
            HeadSpec::pointer("pick", "cells", 0),
            HeadSpec::pointer("pick2", "cells", 0),
            HeadSpec::categorical("cfg", 3).condition_on("pick"),
            HeadSpec::categorical("cfg2", 4).condition_on("pick2"),
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
        seed: 12,
    }
}

/// Launches per `act` (the two inefficiencies removed: pointer logits once
/// per step with the masks built once per call, and one `accumulate_draw`
/// launch per draw; before, 350 and 395). The counts are the cpu backend's:
/// other backends pick other kernels for the same ops (the tuned matmuls, the
/// fused scan), so there they are printed, not pinned, as in
/// tests/entity_footprint.rs's reasoning.
#[test]
fn act_launch_count() {
    let _guard = COUNT_LOCK.lock().unwrap();
    let device = dev();
    let pin = device.name() == "cpu";
    let b = 8usize;
    let measure = |spec: &EntityModelSpec, seed: u64| {
        let ac = EntityActorCritic::<R, f32>::init(spec, &device).unwrap();
        let batch = EntityBatch::<R, f32>::from_host(spec, &plain_obs(b, seed), &device).unwrap();
        ac.act(&batch, 1.0, 1).unwrap();
        reset_launch_count();
        ac.act(&batch, 1.0, 2).unwrap();
        launch_count()
    };
    let tiny = measure(&step_causal_spec(), 91);
    let two_ptr = measure(&two_ptr_spec(), 92);
    println!("{}: act launches per call: tiny {tiny}, two-pointer {two_ptr}", device.name());
    if pin {
        assert_eq!(tiny, 296, "tiny act launches per call");
        assert_eq!(two_ptr, 289, "two-ptr act launches per call");
    }
}

#[test]
fn act_reads_once_and_update_reads_once() {
    let _guard = COUNT_LOCK.lock().unwrap();
    let device = dev();
    let spec = step_causal_spec();
    let ac = EntityActorCritic::<R, f32>::init(&spec, &device).unwrap();
    let (b, m, k) = (8usize, 2usize, 2usize);
    // Warm up (lazy backend init), then pin one act + read to one read.
    let batch0 = EntityBatch::<R, f32>::from_host(&spec, &plain_obs(b, 91), &device).unwrap();
    ac.act(&batch0, 1.0, 1).unwrap().read().unwrap();
    reset_read_count();
    ac.act(&batch0, 1.0, 2).unwrap().read().unwrap();
    assert_eq!(read_count(), 1, "act + read took more than one read");
    // A rollout uploaded once, then a several-minibatch update whose numbers
    // are read together: still one read.
    let obs = plain_obs(b, 95);
    let rollout = EntityBatch::<R, f32>::from_host(&spec, &obs, &device).unwrap();
    let acted = ac.act(&rollout, 1.0, 3).unwrap().read().unwrap();
    let mut full = obs.clone();
    for (key, (shape, data)) in actions_to_arrays(&acted, &spec).ints {
        full.insert_int(&key, shape, data);
    }
    // Labels for the categorical head ride along (present queries act it).
    let cfg: Vec<i64> = acted.actions["pick"]
        .iter()
        .map(|&v| if v == -1 { -1 } else { v % 3 })
        .collect();
    full.insert_int("label.cfg", vec![b, m, k], cfg);
    let flags: Vec<f32> = vec![0.0; b * m * k * 2];
    full.insert_f32("label.flags", vec![b, m, k, 2], flags);
    let data = EntityDataset::<R, f32>::from_arrays(&spec, &full, &device).unwrap();
    let up =
        |v: Vec<f32>, shape: Vec<usize>| Tensor::<R, f32>::from_f32(&v, shape, &device).unwrap();
    let full_lp = up(acted.cell_log_prob.clone(), vec![b, m, k]);
    let full_mask = up(acted.cell_mask.clone(), vec![b, m, k]);
    let full_adv = up(frand(b, 96), vec![b]);
    let full_ret = up(frand(b, 97), vec![b]);
    let full_val = up(acted.value.clone(), vec![b]);
    let config = PpoConfig::default();
    let task = EntityPpoTask::new(&ac, config, 1.0);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-3).build().unwrap(),
        AdamWConfig::builder().learning_rate(1e-3).build().init::<R, f32>(),
    );
    // Warm up the whole path once.
    let warm = EntityPpoBatch::from_rollout(
        &spec, &data, &full_lp, &full_mask, &full_adv, &full_ret, &full_val, &[0, 1, 2, 3],
    )
    .unwrap();
    let queued = trainer.step(&task, std::slice::from_ref(&warm)).unwrap();
    let _ = (queued, task.stats());
    // The pinned update: 2 epochs x 2 minibatches queued, one shared read.
    reset_read_count();
    let mut queued = Vec::new();
    for _ in 0..2 {
        for mb in [[0u32, 1, 2, 3], [4, 5, 6, 7]] {
            let ppo = EntityPpoBatch::from_rollout(
                &spec, &data, &full_lp, &full_mask, &full_adv, &full_ret, &full_val, &mb,
            )
            .unwrap();
            queued.push(trainer.queue_step(&task, std::slice::from_ref(&ppo)).unwrap());
        }
    }
    assert_eq!(read_count(), 0, "queueing PPO steps read back");
    let stats = task.stat_tensors().expect("a loss was taken");
    let mut scalars: Vec<&Tensor<R, f32>> = queued.iter().flat_map(|q| q.scalars()).collect();
    let steps = scalars.len();
    scalars.extend(&stats);
    let (_, values) = mamba3::tensor::ops::index::read_all(&[], &scalars).unwrap();
    assert_eq!(read_count(), 1, "the update's numbers took more than one read");
    let infos = trainer.report_steps(
        &queued,
        &values[..steps].iter().map(|v| v[0]).collect::<Vec<_>>(),
    );
    assert_eq!(infos.len(), 4);
    assert!(infos.iter().all(|i| i.loss.is_finite()));
}
