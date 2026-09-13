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
    Collector, Mamba3Policy, Mamba3PolicyConfig, PpoBatch, PpoConfig, PpoTask, RecallEnv,
    ReferencePolicy, reference_log_probs,
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
    let mut collector = Collector::new(actor, ENVS, WINDOW, obs_dim, &device)
        .expect("a collector")
        .with_seed(3);
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

    let without = PpoTask::new(&actor, plain)
        .evaluate(&batch)
        .expect("a loss");
    let anchored = PpoConfig::default().with_reference_penalty(10.0);
    let with = PpoTask::new(&actor, anchored)
        .evaluate(&batch)
        .expect("a loss");

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
    assert!(
        kl.abs() < 1e-5,
        "a policy is not distant from itself, got {kl}"
    );
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

    let base = PpoTask::new(&actor, plain)
        .evaluate(&anchored)
        .expect("a loss");
    let coeff = 2.0;
    let priced = PpoTask::new(&actor, PpoConfig::default().with_reference_penalty(coeff))
        .evaluate(&anchored)
        .expect("a loss");

    let kl = priced.reference_kl.to_f32()[0];
    assert!(
        kl > 0.0,
        "two different policies are at a positive distance, got {kl}"
    );
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
        AdamWConfig::builder()
            .learning_rate(3e-3)
            .build()
            .init::<R, f32>(),
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
        AdamWConfig::builder()
            .learning_rate(1e-2)
            .build()
            .init::<R, f32>(),
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

// ---------------------------------------------------------------------------
// K7: the corrected score is not just different from the naive one, it is right.
// ---------------------------------------------------------------------------

mod oracle {
    use super::*;
    use mamba3::autograd::Var;
    use mamba3::rl::{EnvStep, RolloutEngine, VecEnv};
    use mamba3::ssm::{Discretization, SsmMode, StateDynamics};
    use mamba3::tensor::ops::index::IdTensor;

    const LANES: usize = 6;
    const ACTIONS: usize = 4;
    const OBS: usize = 5;
    const STEPS: usize = 6;
    const WINDOWS: usize = 3;
    /// Scan (`forward`) against step-by-step decoding (`RolloutEngine::step`) in
    /// CPU `f32`: the two evaluate the same recurrence in a different order, so
    /// they agree to rounding, not to the bit — measured at ~1.2e-7 here. The
    /// naive (actor-cache) path and both local mutations this test exists to
    /// catch (scoring from `batch.initial`; a reset mask shifted one step) move
    /// scores by 5e-4 or more.
    const TOLERANCE: f32 = 1e-5;

    /// Per-lane episode lengths, chosen against `STEPS = 6` so that lanes end
    /// episodes at *different* steps: lane 0 ends exactly on the last step of
    /// window 1 (a boundary reset carried into window 2's first observation),
    /// lanes 1, 2 and 5 end inside windows, lane 3 straddles a boundary, and
    /// lane 4 never resets at all.
    const HORIZONS: [usize; LANES] = [6, 4, 5, 7, 1000, 3];

    /// A host-driven environment whose every lane has its own horizon, with a
    /// deterministic observation that depends on the lane, the clock, the episode
    /// and the last action — so the recurrent state has real structure to carry.
    /// It logs the flags it hands out, which is what the oracle derives its resets
    /// from: *not* from `PpoBatch::reset`, so a shifted reset mask on the scoring
    /// side cannot be mirrored into the oracle.
    struct LaneEnv {
        clock: [usize; LANES],
        episode: [usize; LANES],
        last_action: [u32; LANES],
        masked: bool,
        dones: Vec<[bool; LANES]>,
        /// `[LANES * ACTIONS]` per step, logged when the collector asks for it.
        masks: std::cell::RefCell<Vec<Vec<f32>>>,
    }

    impl LaneEnv {
        fn new(masked: bool) -> Self {
            Self {
                clock: [0; LANES],
                episode: [0; LANES],
                last_action: [0; LANES],
                masked,
                dones: Vec::new(),
                masks: std::cell::RefCell::new(Vec::new()),
            }
        }

        fn observation(&self) -> Tensor<R, f32> {
            let mut data = Vec::with_capacity(LANES * OBS);
            for lane in 0..LANES {
                for k in 0..OBS {
                    let phase = (lane * 7 + self.clock[lane] * 3 + self.episode[lane] * 5 + k * 11)
                        as f32
                        + self.last_action[lane] as f32 * 0.5;
                    data.push((phase * 0.61).sin());
                }
            }
            Tensor::from_f32(&data, vec![LANES, OBS], &dev()).expect("an observation")
        }

        /// Legal actions for the current observation: always at least one, and a
        /// different subset per lane and clock so the mask genuinely varies.
        fn mask_rows(&self) -> Vec<f32> {
            let mut rows = Vec::with_capacity(LANES * ACTIONS);
            for lane in 0..LANES {
                let keep = (lane + self.clock[lane]) % ACTIONS;
                for a in 0..ACTIONS {
                    let legal = a == keep || (a + lane + self.clock[lane]).is_multiple_of(3);
                    rows.push(if legal { 1.0 } else { 0.0 });
                }
            }
            rows
        }
    }

    impl VecEnv<R, f32> for LaneEnv {
        fn envs(&self) -> usize {
            LANES
        }

        fn obs_dim(&self) -> usize {
            OBS
        }

        fn action_dim(&self) -> usize {
            ACTIONS
        }

        fn reset(&mut self) -> Result<Tensor<R, f32>> {
            self.clock = [0; LANES];
            self.last_action = [0; LANES];
            Ok(self.observation())
        }

        fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, f32>> {
            let ids = actions.to_vec();
            let mut rewards = [0.0f32; LANES];
            let mut done = [false; LANES];
            for lane in 0..LANES {
                rewards[lane] = if ids[lane] as usize == lane % ACTIONS {
                    1.0
                } else {
                    0.0
                };
                self.last_action[lane] = ids[lane];
                self.clock[lane] += 1;
                if self.clock[lane] == HORIZONS[lane] {
                    done[lane] = true;
                    self.clock[lane] = 0;
                    self.episode[lane] += 1;
                    self.last_action[lane] = 0;
                }
            }
            self.dones.push(done);
            let flags: Vec<f32> = done.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
            Ok(EnvStep {
                observation: self.observation(),
                reward: Tensor::from_f32(&rewards, vec![LANES], &dev())?,
                done: Tensor::from_f32(&flags, vec![LANES], &dev())?,
            })
        }

        fn action_mask(&self) -> Result<Option<Tensor<R, f32>>> {
            if !self.masked {
                return Ok(None);
            }
            // Called once per step before the draw; logging here records the mask
            // for exactly the observation the action is drawn on, independently of
            // what the buffer later stores.
            let rows = self.mask_rows();
            let tensor = Tensor::from_f32(&rows, vec![LANES, ACTIONS], &dev()).expect("a mask");
            self.masks.borrow_mut().push(rows);
            Ok(Some(tensor))
        }
    }

    fn variant_policy(
        seed: u64,
        discretization: Discretization,
        dynamics: StateDynamics,
        mode: SsmMode,
        conv_kernel: Option<usize>,
    ) -> Mamba3Policy<R, f32> {
        Mamba3PolicyConfig::new(OBS, ACTIONS, 16, 2)
            .with_seed(seed)
            .with_ssm(|s| {
                s.n_heads = 2;
                s.head_dim = 8;
                s.n_groups = 2;
                s.d_state = 4;
                s.chunk_size = 4;
                s.conv_kernel = conv_kernel;
                s.discretization = discretization;
                s.dynamics = dynamics;
                s.mode = mode;
                // Large steps and slow decay, so the recurrent state is dominated by
                // history rather than washed out within a step or two: a cache
                // carried from the wrong place, or cut a step late, has to move the
                // scores by far more than rounding for this test to mean anything.
                s.dt_min = 0.5;
                s.dt_max = 1.0;
                s.a_init_min = 0.01;
                s.a_init_max = 0.05;
            })
            .init::<R, f32>(&dev())
            .expect("a variant policy")
    }

    /// The independent reference: a separately snapshotted copy stepped one
    /// observation at a time through the decode path `Rollout` uses, with its own
    /// persistent state, resetting a lane exactly when that lane's episode ended.
    struct Oracle<'a> {
        engine: RolloutEngine<'a, R, f32>,
        /// `[LANES]` flags: the previous step (possibly in the previous window)
        /// ended that lane's episode.
        carry: [bool; LANES],
    }

    impl<'a> Oracle<'a> {
        fn new(policy: &'a Mamba3Policy<R, f32>) -> Self {
            Self {
                engine: RolloutEngine::new(policy, LANES, &dev()),
                carry: [false; LANES],
            }
        }

        /// `log π_ref(a_t | own history)` for one window, `[LANES * STEPS]` row-major
        /// by lane, computed in `f64` on the host from the decoded logits.
        fn score(
            &mut self,
            batch: &PpoBatch<R, f32>,
            dones: &[[bool; LANES]],
            masks: Option<&[Vec<f32>]>,
        ) -> Vec<f32> {
            assert_eq!(dones.len(), STEPS, "the oracle scores exactly one window");
            let observations = batch.observations.to_f32();
            let actions = batch.actions.to_vec();
            let mut scores = vec![0.0f32; LANES * STEPS];
            for t in 0..STEPS {
                let mut obs = Vec::with_capacity(LANES * OBS);
                for lane in 0..LANES {
                    let base = (lane * STEPS + t) * OBS;
                    obs.extend_from_slice(&observations[base..base + OBS]);
                }
                let obs = Tensor::from_f32(&obs, vec![LANES, 1, OBS], &dev()).expect("obs");
                let reset: Vec<f32> = self
                    .carry
                    .iter()
                    .map(|&c| if c { 1.0 } else { 0.0 })
                    .collect();
                let reset = Tensor::from_f32(&reset, vec![LANES], &dev()).expect("reset");
                let out = self
                    .engine
                    .step(&Var::constant(obs), Some(&reset))
                    .expect("a decode step");
                let logits = out.logits.tensor().to_f32();
                for lane in 0..LANES {
                    let row = &logits[lane * ACTIONS..(lane + 1) * ACTIONS];
                    let legal = |a: usize| masks.is_none_or(|m| m[t][lane * ACTIONS + a] != 0.0);
                    let max = (0..ACTIONS)
                        .filter(|&a| legal(a))
                        .map(|a| row[a] as f64)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let lse = max
                        + (0..ACTIONS)
                            .filter(|&a| legal(a))
                            .map(|a| (row[a] as f64 - max).exp())
                            .sum::<f64>()
                            .ln();
                    let chosen = actions[lane * STEPS + t] as usize;
                    assert!(legal(chosen), "the actor drew an illegal action");
                    scores[lane * STEPS + t] = (row[chosen] as f64 - lse) as f32;
                }
                self.carry = dones[t];
            }
            scores
        }
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Every cache component the variant has must hold nonzero state, or a
    /// component that `score` forgot to carry would go unnoticed.
    fn assert_cache_is_live(
        reference: &ReferencePolicy<R, f32>,
        conv: bool,
        rotational: bool,
        trapezoid: bool,
    ) {
        let cache = reference.cache().expect("a cache after scoring");
        let nonzero = |values: Vec<f32>, what: &str| {
            let peak = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(peak > 1e-6, "{what} carried no state (peak {peak})");
        };
        for (layer, c) in cache.iter().enumerate() {
            nonzero(
                c.ssm.h.tensor().to_f32(),
                &format!("layer {layer} SSM state"),
            );
            if trapezoid {
                nonzero(
                    c.ssm.last_u.tensor().to_f32(),
                    &format!("layer {layer} trapezoid input"),
                );
            }
            match (&c.ssm.angle, rotational) {
                (Some(angle), true) => nonzero(
                    angle.tensor().to_f32(),
                    &format!("layer {layer} rotation angle"),
                ),
                (None, false) => {}
                (angle, _) => panic!(
                    "layer {layer}: angle presence {} unexpected",
                    angle.is_some()
                ),
            }
            match (&c.conv, conv) {
                (Some(history), true) => nonzero(
                    history.tensor().to_f32(),
                    &format!("layer {layer} conv history"),
                ),
                (None, false) => {}
                (history, _) => {
                    panic!(
                        "layer {layer}: conv presence {} unexpected",
                        history.is_some()
                    )
                }
            }
        }
    }

    /// The environment's own log, `[STEPS]` of `[LANES * ACTIONS]`, laid out
    /// `[LANES * STEPS * ACTIONS]` the way the batch stores it.
    fn as_batch_layout(steps: &[Vec<f32>]) -> Vec<f32> {
        let mut out = vec![0.0f32; LANES * STEPS * ACTIONS];
        for (t, row) in steps.iter().enumerate() {
            for lane in 0..LANES {
                let at = (lane * STEPS + t) * ACTIONS;
                out[at..at + ACTIONS].copy_from_slice(&row[lane * ACTIONS..(lane + 1) * ACTIONS]);
            }
        }
        out
    }

    fn run(
        discretization: Discretization,
        dynamics: StateDynamics,
        mode: SsmMode,
        conv_kernel: Option<usize>,
        masked: bool,
    ) {
        let label = format!(
            "{discretization:?}/{dynamics:?}/{mode:?}/conv={conv_kernel:?}/masked={masked}"
        );
        let device = dev();
        let actor = variant_policy(40, discretization, dynamics, mode, conv_kernel);
        let source = variant_policy(41, discretization, dynamics, mode, conv_kernel);
        let mut tracked = ReferencePolicy::snapshot(&source, &device).expect("a snapshot");
        // A second, independent snapshot drives the oracle, so nothing the scored
        // reference does to its own weights or cache can leak into it.
        let oracle_weights = ReferencePolicy::snapshot(&source, &device).expect("a snapshot");
        let mut oracle = Oracle::new(oracle_weights.policy());

        let config = PpoConfig::default();
        let mut env = LaneEnv::new(masked);
        let mut collector = Collector::new(&actor, LANES, STEPS, OBS, &device)
            .expect("a collector")
            .with_seed(17);
        let task = PpoTask::new(&actor, config);
        let mut trainer = a_trainer();

        let mut saw_boundary_reset = false;
        let mut saw_inner_reset = false;
        for w in 0..WINDOWS {
            let first_log = env.dones.len();
            let first_mask = env.masks.borrow().len();
            let report = collector.collect(&mut env).expect("a window");
            let batch = collector.ppo_batch(&report, &config).expect("a batch");
            let dones = &env.dones[first_log..];
            saw_boundary_reset |= dones[STEPS - 1].iter().any(|&d| d);
            saw_inner_reset |= dones[..STEPS - 1].iter().any(|row| row.iter().any(|&d| d));
            let logged: Vec<Vec<f32>> = env.masks.borrow()[first_mask..].to_vec();
            let masks = masked.then_some(logged);
            match (&batch.action_mask, &masks) {
                (Some(stored), Some(logged)) => assert_eq!(
                    stored.to_f32(),
                    as_batch_layout(logged),
                    "{label}: the batch must store the mask the environment gave"
                ),
                (None, None) => {}
                (stored, _) => panic!(
                    "{label}: mask stored = {}, provided = {masked}",
                    stored.is_some()
                ),
            }

            let corrected = tracked.score(&batch).expect("scored").to_f32();
            let expected = oracle.score(&batch, dones, masks.as_deref());
            let diff = max_abs(&corrected, &expected);
            eprintln!("{label}: window {w}: |score - oracle| = {diff}");
            assert!(
                diff < TOLERANCE,
                "{label}: window {w}: ReferencePolicy::score disagrees with an independently \
                 stepped reference by {diff}"
            );
            if w == 0 {
                assert_cache_is_live(
                    &tracked,
                    conv_kernel.is_some(),
                    dynamics == StateDynamics::Rotational,
                    discretization != Discretization::Euler,
                );
            } else {
                // Once the actor has moved, its snapshot is not the reference's
                // history: the naive path must be measurably wrong.
                let naive = reference_log_probs(&source, &batch)
                    .expect("naive")
                    .to_f32();
                let naive_diff = max_abs(&naive, &expected);
                eprintln!("{label}: window {w}: |naive - oracle| = {naive_diff}");
                assert!(
                    naive_diff > 10.0 * TOLERANCE,
                    "{label}: window {w}: scoring from the actor's cache should be wrong by \
                     more than rounding, got {naive_diff}"
                );
            }

            // Two updates, several epochs each: nothing in an update may advance
            // the reference's history. The next window's comparison is what checks it.
            for _ in 0..2 {
                for _ in 0..3 {
                    trainer
                        .step(&task, std::slice::from_ref(&batch))
                        .expect("an update");
                }
            }
        }
        assert!(
            saw_boundary_reset,
            "{label}: the fixture must end an episode on a window's last step"
        );
        assert!(
            saw_inner_reset,
            "{label}: the fixture must end an episode inside a window"
        );
    }

    #[test]
    fn siso_rotational_trapezoid_with_conv_matches_an_online_reference() {
        run(
            Discretization::LearnedTrapezoid,
            StateDynamics::Rotational,
            SsmMode::Siso,
            Some(4),
            false,
        );
    }

    #[test]
    fn mimo_rotational_trapezoid_with_conv_matches_an_online_reference() {
        run(
            Discretization::LearnedTrapezoid,
            StateDynamics::Rotational,
            SsmMode::Mimo { rank: 2 },
            Some(3),
            false,
        );
    }

    #[test]
    fn real_euler_without_conv_matches_an_online_reference() {
        run(
            Discretization::Euler,
            StateDynamics::Real,
            SsmMode::Siso,
            None,
            false,
        );
    }

    #[test]
    fn a_masked_window_matches_an_online_reference_under_the_stored_mask() {
        run(
            Discretization::LearnedTrapezoid,
            StateDynamics::Rotational,
            SsmMode::Siso,
            Some(4),
            true,
        );
    }
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
