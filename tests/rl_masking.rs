//! A1: the same legal-action distribution everywhere.
//!
//! Masking has to hold in five different places at once — the draw, the
//! behaviour log-probability, the PPO replay, the entropy bonus and imitation's
//! cross entropy — and disagreeing between any two of them is exactly the class
//! of bug that trains without ever raising an error. This file checks the
//! mechanism directly (never drawing a masked action, a singleton legal action
//! having zero entropy, an empty row being refused) and the invariants that
//! already had names before this file existed: the first-epoch PPO ratio is 1,
//! and the reference agrees with the actor about which actions exist at all.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::prelude::*;
use mamba3::rl::{
    Collector, EnvStep, Mamba3Policy, Mamba3PolicyConfig, PpoConfig, PpoTask, RecallEnv,
    ReferencePolicy, VecEnv, reference_log_probs, validate_action_mask,
};
use mamba3::tensor::ops::index::IdTensor;

type R = Auto;

const ENVS: usize = 6;
const SYMBOLS: usize = 4;
const HORIZON: usize = 1000; // long enough that no episode completes in these windows
const WINDOW: usize = 6;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn tensor(data: &[f32], dims: Vec<usize>) -> Tensor<R, f32> {
    Tensor::from_f32(data, dims, &dev()).expect("data fills the shape")
}

fn policy(seed: u64) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(SYMBOLS + 2, SYMBOLS, 32, 1)
        .with_seed(seed)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 16;
            s.n_groups = 2;
            s.d_state = 8;
            s.chunk_size = 4;
            s.conv_kernel = Some(4);
        })
        .init::<R, f32>(&dev())
        .expect("a small policy")
}

/// [`RecallEnv`] with every odd-numbered action permanently illegal. The cue it
/// draws is unconstrained, so an even-cue episode is still winnable and an
/// odd-cue one is not -- this exists to exercise the masking mechanism under a
/// real collector and a real recurrent policy, not to prove the task is
/// learnable under it.
struct HalfMaskedEnv {
    inner: RecallEnv<R, f32>,
    envs: usize,
    action_dim: usize,
}

impl HalfMaskedEnv {
    fn new(envs: usize, symbols: usize, seed: u64) -> Self {
        let inner = RecallEnv::new(envs, symbols, HORIZON, seed, &dev()).expect("the recall task");
        Self {
            inner,
            envs,
            action_dim: symbols,
        }
    }

    fn legal_mask(&self) -> Tensor<R, f32> {
        let row: Vec<f32> = (0..self.action_dim)
            .map(|a| if a % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let flat: Vec<f32> = (0..self.envs).flat_map(|_| row.clone()).collect();
        tensor(&flat, vec![self.envs, self.action_dim])
    }
}

impl VecEnv<R, f32> for HalfMaskedEnv {
    fn envs(&self) -> usize {
        self.inner.envs()
    }

    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }

    fn action_dim(&self) -> usize {
        self.inner.action_dim()
    }

    fn reset(&mut self) -> Result<Tensor<R, f32>> {
        self.inner.reset()
    }

    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, f32>> {
        self.inner.step(actions)
    }

    fn action_mask(&self) -> Option<Tensor<R, f32>> {
        Some(self.legal_mask())
    }
}

#[test]
fn masked_sampling_never_draws_an_illegal_action() {
    let actor = policy(1);
    let mut env = HalfMaskedEnv::new(ENVS, SYMBOLS, 11);
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(&actor, ENVS, WINDOW, obs_dim, &dev())
        .expect("a collector")
        .with_seed(3);

    // Ten windows, so every lane draws sixty actions total: enough that a
    // masking bug (say, an off-by-one on which action is "even") would show
    // up almost immediately rather than by chance never firing.
    for _ in 0..10 {
        let report = collector.collect(&mut env).expect("a window");
        let ids = collector.buffer().actions().to_vec();
        for id in ids {
            assert_eq!(id % 2, 0, "an odd (masked) action was drawn: {id}");
        }
        let _ = report;
    }
}

#[test]
fn a_replayed_masked_window_reproduces_the_actors_log_probabilities() {
    // Invariant 1, under masking: the first-epoch PPO ratio must be 1, which
    // means the replay has to score the *masked* distribution the same way the
    // draw did, not the raw one.
    let actor = policy(2);
    let mut env = HalfMaskedEnv::new(ENVS, SYMBOLS, 12);
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(&actor, ENVS, WINDOW, obs_dim, &dev())
        .expect("a collector")
        .with_seed(4);
    let report = collector.collect(&mut env).expect("a window");
    let config = PpoConfig::default();
    let batch = collector.ppo_batch(&report, &config).expect("a batch");
    assert!(batch.action_mask.is_some(), "the mask should have been recorded");

    let task = PpoTask::new(&actor, config);
    let loss = task.evaluate(&batch).expect("a loss");
    let approx_kl = loss.approx_kl.to_f32()[0];
    assert!(
        approx_kl.abs() < 1e-4,
        "an unmoved policy replaying its own masked window should show ~0 KL, got {approx_kl}"
    );
    let clip_fraction = loss.clip_fraction.to_f32()[0];
    assert_eq!(clip_fraction, 0.0, "nothing should be clipped on the very first replay");
}

#[test]
fn a_singleton_legal_action_has_zero_entropy_and_finite_gradients() {
    use mamba3::distributions::{Categorical, Distribution};
    use mamba3::nn::param::Param;

    let envs = 3;
    let classes = 4;
    let logits = tensor(
        &(0..envs * classes).map(|i| (i as f32) * 0.37 - 1.0).collect::<Vec<_>>(),
        vec![envs, classes],
    );
    // Only action 0 is legal, for every lane.
    let legal_row = [1.0f32, 0.0, 0.0, 0.0];
    let legal: Vec<f32> = (0..envs).flat_map(|_| legal_row).collect();
    let legal = tensor(&legal, vec![envs, classes]);

    let param = Param::new(logits);
    let masked = param.var_standalone().mask_logits(&legal).expect("masked logits");
    let dist = Categorical::from_logits(masked).expect("a categorical");
    let entropy = dist.entropy().expect("entropy");
    for e in entropy.tensor().to_f32() {
        assert!(e.abs() < 1e-5, "a singleton legal action must have ~0 entropy, got {e}");
    }

    let grads = entropy.sum().expect("sum").backward().expect("finite gradients");
    let grad = grads.get(param.id()).expect("a gradient").to_f32();
    for g in grad {
        assert!(g.is_finite(), "the entropy gradient under a singleton mask must be finite");
    }
}

#[test]
fn an_all_legal_mask_matches_the_unmasked_objective_exactly() {
    // Disabled-by-default compatibility: a mask that permits everything must
    // not move the objective at all. Both batches share one replay pass, so
    // the mask is the only thing that can possibly differ between them.
    use mamba3::rl::ppo_objective;

    let actor = policy(3);
    let mut env =
        RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, 13, &dev()).expect("the recall task");
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(&actor, ENVS, WINDOW, obs_dim, &dev())
        .expect("a collector")
        .with_seed(5);
    let report = collector.collect(&mut env).expect("a window");
    let config = PpoConfig::default();
    let plain_batch = collector.ppo_batch(&report, &config).expect("a batch");

    let mut masked_batch = plain_batch.clone();
    masked_batch.action_mask = Some(tensor(
        &vec![1.0f32; ENVS * WINDOW * SYMBOLS],
        vec![ENVS, WINDOW, SYMBOLS],
    ));

    let (output, _) = actor
        .forward(
            &mamba3::autograd::Var::constant(plain_batch.observations.clone()),
            plain_batch.reset.as_ref(),
            plain_batch.initial.as_deref(),
        )
        .expect("a forward pass");

    let plain = ppo_objective(&output, &plain_batch, &config).expect("a loss");
    let masked = ppo_objective(&output, &masked_batch, &config).expect("a loss");
    let (a, b) = (
        plain.total.tensor().to_f32()[0],
        masked.total.tensor().to_f32()[0],
    );
    assert!(
        (a - b).abs() < 1e-4,
        "an all-legal mask must not move the objective: {a} vs {b}"
    );
}

#[test]
fn an_empty_legal_set_is_refused_with_no_special_casing() {
    let mask = tensor(&[1.0, 0.0, 0.0, /* env 1: */ 0.0, 0.0, 0.0], vec![2, 3]);
    let err = validate_action_mask(&mask).expect_err("a row with no legal action must be refused");
    assert!(format!("{err}").contains("no legal action"), "{err}");
}

#[test]
fn a_non_finite_mask_value_is_refused() {
    let mask = tensor(&[1.0, 0.0, f32::NAN, 1.0], vec![2, 2]);
    let err = validate_action_mask(&mask).expect_err("a NaN mask value must be refused");
    assert!(format!("{err}").contains("non-finite"), "{err}");
}

#[test]
fn a_fully_legal_mask_passes_validation() {
    let mask = tensor(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    validate_action_mask(&mask).expect("every action legal must be accepted");
}

#[test]
fn reference_scoring_respects_the_same_mask_the_actor_used() {
    // "Reference support agreement": the reference must be scored over exactly
    // the actions the mask left legal, using the same mechanism as the actor's
    // own replay -- both go through `PpoBatch::action_mask`.
    let actor = policy(4);
    let reference_source = policy(5);
    let mut env = HalfMaskedEnv::new(ENVS, SYMBOLS, 14);
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(&actor, ENVS, WINDOW, obs_dim, &dev())
        .expect("a collector")
        .with_seed(6);
    let report = collector.collect(&mut env).expect("a window");
    let config = PpoConfig::default();
    let batch = collector.ppo_batch(&report, &config).expect("a batch");

    let mut reference = ReferencePolicy::snapshot(&reference_source, &dev()).expect("a snapshot");
    let scores = reference.score(&batch).expect("masked reference scores");
    for s in scores.to_f32() {
        assert!(s.is_finite(), "a masked reference score must be finite, got {s}");
    }

    // The compatibility wrapper, scoring the same masked window from scratch,
    // must agree: both go through the same `batch.action_mask`.
    let via_compat =
        reference_log_probs(reference.policy(), &batch).expect("the compat wrapper");
    let a = scores.to_f32();
    let b = via_compat.to_f32();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-4, "the two scoring paths disagreed: {x} vs {y}");
    }
}

#[test]
fn minibatch_slicing_keeps_each_lanes_own_mask() {
    let actor = policy(6);
    let mut env = HalfMaskedEnv::new(4, SYMBOLS, 15);
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(&actor, 4, WINDOW, obs_dim, &dev())
        .expect("a collector")
        .with_seed(7);
    let report = collector.collect(&mut env).expect("a window");
    let config = PpoConfig::default();
    let batch = collector.ppo_batch(&report, &config).expect("a batch");
    let full_mask = batch.action_mask.clone().expect("a mask");

    let half = batch.minibatch(0, 2).expect("a minibatch");
    let sliced_mask = half.action_mask.expect("the minibatch keeps its mask");
    assert_eq!(sliced_mask.shape().dims(), &[2, WINDOW, SYMBOLS]);
    let (full, sliced) = (full_mask.to_f32(), sliced_mask.to_f32());
    let per_env = WINDOW * SYMBOLS;
    assert_eq!(&full[..2 * per_env], &sliced[..]);
}

mod imitation {
    use super::*;
    use mamba3::rl::{ImitationBatch, behaviour_cloning_loss};

    #[test]
    fn an_expert_label_naming_an_illegal_action_is_refused() {
        let logits = mamba3::autograd::Var::constant(tensor(
            &[0.1, 0.2, 0.3, 0.1, -0.2, 0.4],
            vec![1, 2, 3],
        ));
        // Action 1 is illegal at both positions, and the expert names it at
        // the second.
        let mask = tensor(&[1.0, 0.0, 1.0, 1.0, 0.0, 1.0], vec![1, 2, 3]);
        let expert = IdTensor::from_slice(&[0u32, 1], vec![1, 2], &dev()).expect("expert ids");
        let err = behaviour_cloning_loss(&logits, &expert, Some(&mask), None, 0.0)
            .expect_err("an illegal expert label must be refused");
        assert!(format!("{err}").contains("illegal"), "{err}");
    }

    #[test]
    fn a_legal_expert_label_trains_without_error_under_masking() {
        let logits = mamba3::autograd::Var::traced(tensor(
            &[0.1, 0.2, 0.3, 0.1, -0.2, 0.4],
            vec![1, 2, 3],
        ));
        let mask = tensor(&[1.0, 0.0, 1.0, 1.0, 0.0, 1.0], vec![1, 2, 3]);
        let expert = IdTensor::from_slice(&[0u32, 2], vec![1, 2], &dev()).expect("expert ids");
        let loss = behaviour_cloning_loss(&logits, &expert, Some(&mask), None, 0.01)
            .expect("a legal-label batch must train");
        let value = loss.tensor().to_f32()[0];
        assert!(value.is_finite(), "the masked imitation loss must be finite, got {value}");
        let grads = loss.backward().expect("a backward pass");
        let _ = grads;
    }

    #[test]
    fn masking_is_absent_when_the_batch_carries_none() {
        let batch = ImitationBatch::new(
            tensor(&[0.0; 2 * 3], vec![1, 2, 3]),
            IdTensor::from_slice(&[0u32, 1], vec![1, 2], &dev()).unwrap(),
        );
        assert!(batch.action_mask.is_none());
    }
}
