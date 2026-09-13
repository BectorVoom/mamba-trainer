//! The anchor: PPO priced against a policy that does not move.
//!
//! The clip bounds how far one update may take the policy from the weights that
//! collected the data. Nothing in it bounds where two hundred such updates end up,
//! so a run that starts from a good cloned policy can walk away from it a fraction
//! of a nat at a time and never once trip the trust region. `reference_coeff` and
//! `PpoBatch::reference_log_probs` are what price that distance, and these are the
//! properties they have to have:
//!
//! * off by default, and inert without a reference to compare against;
//! * zero, exactly, when the reference *is* the policy;
//! * positive otherwise, and added to the total at its stated weight;
//! * and — the one that matters — an update under it moves the policy *towards*
//!   the reference rather than away.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::prelude::*;
use mamba3::rl::{
    Collector, Mamba3Policy, Mamba3PolicyConfig, PpoBatch, PpoConfig, PpoTask, ReferencePolicy,
    RecallEnv, reference_log_probs,
};
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = Auto;

const ENVS: usize = 8;
const SYMBOLS: usize = 4;
const HORIZON: usize = 4;
const WINDOW: usize = 8;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// The recall task's observation is wider than its action space; ask it rather
/// than assuming.
fn obs_dim() -> usize {
    RecallEnv::<R, f32>::new(1, SYMBOLS, HORIZON, 0, &dev())
        .expect("the recall task")
        .obs_dim()
}

fn policy(seed: u64) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(obs_dim(), SYMBOLS, 32, 1)
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

/// One collected window, plus the collector that owns its buffer.
fn window(actor: &Mamba3Policy<R, f32>, config: &PpoConfig) -> PpoBatch<R, f32> {
    let device = dev();
    let mut env = RecallEnv::new(ENVS, SYMBOLS, HORIZON, 11, &device).expect("the recall task");
    let obs_dim = env.obs_dim();
    let mut collector =
        Collector::new(actor, ENVS, WINDOW, obs_dim, &device).expect("a collector").with_seed(3);
    let report = collector.collect(&mut env).expect("a window");
    collector.ppo_batch(&report, config).expect("a batch")
}

/// Mean log-probability the policy gives the actions the window recorded.
fn agreement(policy: &Mamba3Policy<R, f32>, batch: &PpoBatch<R, f32>) -> f32 {
    let scores = reference_log_probs(policy, batch).expect("scores");
    let values = scores.to_f32();
    values.iter().sum::<f32>() / values.len() as f32
}

#[test]
fn the_anchor_is_off_by_default() {
    assert_eq!(PpoConfig::default().reference_coeff, 0.0);

    let actor = policy(1);
    let batch = window(&actor, &PpoConfig::default());
    let task = PpoTask::new(&actor, PpoConfig::default());
    let loss = task.evaluate(&batch).expect("a loss");
    assert_eq!(
        loss.reference_kl.to_f32()[0],
        0.0,
        "with no reference there is no distance to report"
    );
}

#[test]
fn a_coefficient_without_a_reference_changes_nothing() {
    let actor = policy(2);
    let plain = PpoConfig::default();
    let batch = window(&actor, &plain);

    let without = PpoTask::new(&actor, plain).evaluate(&batch).expect("a loss");
    let anchored = PpoConfig::default().with_reference_penalty(10.0);
    let with = PpoTask::new(&actor, anchored).evaluate(&batch).expect("a loss");

    // The coefficient prices a distance; with nothing to be distant from it is inert.
    assert!(
        (without.total.tensor().to_f32()[0] - with.total.tensor().to_f32()[0]).abs() < 1e-6,
        "a reference penalty with no reference must not move the loss"
    );
}

#[test]
fn anchoring_a_policy_to_itself_costs_nothing() {
    let actor = policy(3);
    let config = PpoConfig::default().with_reference_penalty(1.0);
    let batch = window(&actor, &config);
    let scores = reference_log_probs(&actor, &batch).expect("scores");
    let anchored = batch.clone().with_reference_log_probs(scores);

    let task = PpoTask::new(&actor, config);
    let plain = task.evaluate(&batch).expect("a loss");
    let with_self = task.evaluate(&anchored).expect("a loss");

    let kl = with_self.reference_kl.to_f32()[0];
    assert!(kl.abs() < 1e-5, "a policy is not distant from itself, got {kl}");
    assert!(
        (plain.total.tensor().to_f32()[0] - with_self.total.tensor().to_f32()[0]).abs() < 1e-5,
        "and so the total is unchanged"
    );
}

#[test]
fn a_different_reference_costs_its_weight() {
    let actor = policy(4);
    let other = policy(5);
    let plain = PpoConfig::default();
    let batch = window(&actor, &plain);
    let scores = reference_log_probs(&other, &batch).expect("scores");
    let anchored = batch.clone().with_reference_log_probs(scores);

    let base = PpoTask::new(&actor, plain).evaluate(&anchored).expect("a loss");
    let coeff = 2.0;
    let priced = PpoTask::new(&actor, PpoConfig::default().with_reference_penalty(coeff))
        .evaluate(&anchored)
        .expect("a loss");

    let kl = priced.reference_kl.to_f32()[0];
    assert!(kl > 0.0, "two different policies are at a positive distance, got {kl}");
    let expected = base.total.tensor().to_f32()[0] + coeff * kl;
    let actual = priced.total.tensor().to_f32()[0];
    assert!(
        (expected - actual).abs() < 1e-4,
        "the anchor enters the total at its weight: expected {expected}, got {actual}"
    );
}

#[test]
fn an_anchored_update_moves_towards_the_reference() {
    // The learner starts where `reference` is and is pulled elsewhere by the
    // advantages; the question is whether the anchor pulls back.
    let reference = policy(6);
    let learner = policy(7);

    let plain = PpoConfig::default();
    let batch = window(&learner, &plain);
    let scores = reference_log_probs(&reference, &batch).expect("scores");
    let anchored = batch.clone().with_reference_log_probs(scores);

    let before = agreement(&learner, &anchored);

    // A weight large enough that the anchor, not the advantage, decides the step.
    let config = PpoConfig::default().with_reference_penalty(50.0);
    let task = PpoTask::new(&learner, config);
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(3e-3)
            .max_grad_norm(1.0)
            .build()
            .expect("a trainer config"),
        AdamWConfig::builder().learning_rate(3e-3).build().init::<R, f32>(),
    );
    for _ in 0..20 {
        trainer
            .step(&task, std::slice::from_ref(&anchored))
            .expect("a step");
    }

    let after = agreement(&learner, &anchored);
    assert!(
        after > before,
        "an anchored update should raise the policy's agreement with the reference: \
         {before} -> {after}"
    );
}

// ---------------------------------------------------------------------------
// R1: the reference keeps its own recurrent history, not the actor's.
// ---------------------------------------------------------------------------

fn mean_abs_diff(a: &mamba3::tensor::Tensor<R, f32>, b: &mamba3::tensor::Tensor<R, f32>) -> f32 {
    let (a, b) = (a.to_f32(), b.to_f32());
    assert_eq!(a.len(), b.len(), "compared tensors must be the same shape");
    a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

fn a_trainer() -> Trainer<R, f32, mamba3::train::AdamW<R, f32>> {
    Trainer::new(
        TrainerConfig::builder()
            .learning_rate(1e-2)
            .max_grad_norm(1.0)
            .build()
            .expect("a trainer config"),
        AdamWConfig::builder().learning_rate(1e-2).build().init::<R, f32>(),
    )
}

/// The bug R1 fixes: `reference_log_probs(reference, batch)` forwards
/// `batch.initial`, the *actor's* own recurrent snapshot, into the reference. That
/// is harmless at the very first window, where the actor's cache is still zero —
/// which is exactly what the reference would start from too — but wrong from the
/// second window on, once the actor has been trained: its cache then reflects
/// weights the reference never had, and scoring the reference from it is scoring
/// a policy as if it had lived a history that was never really its own.
///
/// [`ReferencePolicy`] fixes this by keeping its own cache, continued by
/// [`ReferencePolicy::score`] across windows independently of the actor's.
#[test]
fn reference_scoring_continues_the_references_own_history_not_the_actors() {
    let device = dev();
    let actor = policy(30);
    let reference_source = policy(31);
    let plain = PpoConfig::default();

    // A horizon far longer than the three windows collected below, so no lane
    // ever resets: every step of window 2 and window 3 depends on the cache
    // carried in from the window before. A horizon dividing the window would
    // instead reset every lane in lockstep exactly on the boundary, masking
    // away *any* incoming cache -- actor's or reference's alike -- and making
    // this test pass by accident regardless of which cache scoring actually used.
    let window = 8;
    let horizon = 1000;
    let mut env = RecallEnv::new(ENVS, SYMBOLS, horizon, 41, &device).expect("the recall task");
    let obs_dim = env.obs_dim();
    let mut collector = Collector::new(&actor, ENVS, window, obs_dim, &device)
        .expect("a collector")
        .with_seed(9);
    let mut tracked = ReferencePolicy::snapshot(&reference_source, &device).expect("a snapshot");

    // Window 1: the actor's cache (`batch.initial`) is still the collector's
    // zero-initialised starting state, which is exactly what a reference with no
    // history of its own also starts from -- so the buggy and corrected paths
    // agree here. This is the case that made the bug easy to miss.
    let report1 = collector.collect(&mut env).expect("window 1");
    let batch1 = collector.ppo_batch(&report1, &plain).expect("batch 1");
    let corrected1 = tracked.score(&batch1).expect("scored 1");
    let naive1 = reference_log_probs(&reference_source, &batch1).expect("naive 1");
    assert!(
        mean_abs_diff(&corrected1, &naive1) < 1e-5,
        "at the first window both paths start from a zeroed cache and must agree"
    );

    // Move the actor away from where it started, with two real gradient updates
    // over nonzero recurrent state, so its own recurrent snapshot for the next
    // windows reflects weights the reference was never given.
    let task = PpoTask::new(&actor, plain);
    let mut trainer = a_trainer();
    for _ in 0..2 {
        trainer
            .step(&task, std::slice::from_ref(&batch1))
            .expect("an update");
    }

    // At this tiny width (`d_model=32`, one layer) and one window's depth, an
    // untrained recurrent state barely accumulates any magnitude at all, so the
    // *size* of the divergence below is small even though it is completely real
    // — the point is not how big it is but that it is not exactly zero. If
    // `ReferencePolicy::score` regressed to forwarding `batch.initial` (the bug
    // this test exists to catch), `corrected2` and `naive2` would be the result
    // of the literal same function call on the literal same arguments and would
    // therefore agree to the bit, not just approximately: `diff2` would be
    // exactly `0.0`, not merely small. A threshold well above float rounding
    // noise (~1e-7 here) but far below the smallest divergence actually observed
    // tells the two cases apart.
    let report2 = collector.collect(&mut env).expect("window 2");
    let batch2 = collector.ppo_batch(&report2, &plain).expect("batch 2");
    let corrected2 = tracked.score(&batch2).expect("scored 2");
    let naive2 = reference_log_probs(&reference_source, &batch2).expect("naive 2");
    let diff2 = mean_abs_diff(&corrected2, &naive2);
    assert!(
        diff2 > 1e-6,
        "the actor's snapshot and the reference's own cache should have diverged \
         by the second window once the actor has been trained; got a difference \
         of only {diff2}"
    );

    // Window 3: the correction keeps holding as the shared history grows and a
    // second update widens the gap further.
    for _ in 0..2 {
        trainer
            .step(&task, std::slice::from_ref(&batch2))
            .expect("a second update");
    }
    let report3 = collector.collect(&mut env).expect("window 3");
    let batch3 = collector.ppo_batch(&report3, &plain).expect("batch 3");
    let corrected3 = tracked.score(&batch3).expect("scored 3");
    let naive3 = reference_log_probs(&reference_source, &batch3).expect("naive 3");
    let diff3 = mean_abs_diff(&corrected3, &naive3);
    assert!(
        diff3 > 1e-6,
        "the divergence should persist into a third window; got {diff3}"
    );
}

/// A snapshot is a deep copy: training (or otherwise mutating) the object it was
/// taken from must not move it, including when that object is the very one the
/// snapshot was built from -- the case a shared `Rc`, or a `requires_grad` flip,
/// would get wrong.
#[test]
fn a_snapshot_is_unaffected_by_later_mutation_of_its_source() {
    let device = dev();
    let source = policy(32);
    let mut reference = ReferencePolicy::snapshot(&source, &device).expect("a snapshot");

    let plain = PpoConfig::default();
    let batch = window(&source, &plain);
    let before = reference.score(&batch).expect("scored before training");

    // Train the very object the snapshot was taken from -- the same-object case
    // R1 has to be safe against, since `PpoLearner(policy, env, reference=policy)`
    // is exactly this shape from Python.
    let task = PpoTask::new(&source, plain);
    let mut trainer = a_trainer();
    for _ in 0..5 {
        trainer
            .step(&task, std::slice::from_ref(&batch))
            .expect("a step");
    }

    // Isolate weight drift as the only variable by rescoring from the same
    // zeroed starting cache the first call used.
    reference.reset();
    let after = reference.score(&batch).expect("scored after training");

    assert!(
        mean_abs_diff(&before, &after) < 1e-6,
        "training the source policy must not move an already-taken snapshot"
    );
}
