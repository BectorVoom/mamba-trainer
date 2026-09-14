//! A2b: an exact continuation.
//!
//! N rounds, a checkpoint written to disk, fresh objects built from nothing but
//! that file, M more rounds — against N + M rounds that never stopped. The two
//! must see the same observations (every one acted on, and the one the next
//! window starts from), take the same actions, see the same rewards and masks,
//! reach the same episode-return totals, score the same reference
//! log-probabilities, apply the same learning rates, reach the same counters and
//! end on the same weights. The reference itself is rebuilt from the file. The environment resets lanes at
//! different moments from its own random generator, so a continuation that drops
//! any of the state `mamba3::rl::snapshot` lists diverges within a window.
//! With a moving average of the weights attached, its bits and update counter
//! continue exactly too (T1).

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::error::{Error, Result};
use mamba3::rl::{
    BehaviourCloningTask, CollectReport, Collector, DaggerSchedule, EnvStep, GameWorld,
    Mamba3Policy, Mamba3PolicyConfig, MultiSyncCollector, ParallelEnvs, PpoConfig, PpoTask, Recall,
    RecallEnv, ReferencePolicy, RolloutSnapshot, StateReader, StateWriter, VecEnv, recall_spec,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{
    AdamW, AdamWConfig, Checkpoint, Ema, EmaConfig, EmaWarmup, LrSchedule, StepInfo, Trainer,
    TrainerConfig,
};

type R = Auto;

const LANES: usize = 6;
const OBS: usize = 5;
const ACTIONS: usize = 4;
const WINDOW: usize = 5;
const FIRST: usize = 3;
const SECOND: usize = 3;

fn dev() -> Device<R> {
    Device::<R>::default()
}

// ---------------------------------------------------------------------------
// An environment with every kind of state
// ---------------------------------------------------------------------------

/// Host-side lanes whose episodes end on horizons drawn from the environment's
/// own generator, with rewards that depend on it too, a mask that changes with
/// the clock, and an expert that only names legal actions.
#[derive(Clone)]
struct ResumableEnv {
    rng: u64,
    clock: [u32; LANES],
    horizon: [u32; LANES],
    episode: [u32; LANES],
    /// Every action taken, in order: part of the state, and the record a test
    /// compares.
    log: Vec<u32>,
}

impl ResumableEnv {
    fn new(seed: u64) -> Self {
        Self {
            rng: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
            clock: [0; LANES],
            horizon: [3; LANES],
            episode: [0; LANES],
            log: Vec::new(),
        }
    }

    fn next(&mut self) -> u64 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn observation(&self) -> Result<Tensor<R, f32>> {
        let mut data = Vec::with_capacity(LANES * OBS);
        for lane in 0..LANES {
            for k in 0..OBS {
                let phase = (lane * 7
                    + self.clock[lane] as usize * 3
                    + self.episode[lane] as usize * 5
                    + k * 11) as f32;
                data.push((phase * 0.61).sin());
            }
        }
        Tensor::from_f32(&data, vec![LANES, OBS], &dev())
    }

    fn expert(&self) -> Vec<u32> {
        (0..LANES)
            .map(|lane| (lane as u32 + self.clock[lane] + 1) % ACTIONS as u32)
            .collect()
    }
}

const ENV_TAG: &[u8; 8] = b"TESTLANE";

impl VecEnv<R, f32> for ResumableEnv {
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
        self.episode = [0; LANES];
        for lane in 0..LANES {
            self.horizon[lane] = 2 + (self.next() % 4) as u32;
        }
        self.observation()
    }

    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, f32>> {
        let ids = actions.try_to_vec()?;
        let expert = self.expert();
        self.log.extend_from_slice(&ids);
        let mut reward = [0.0f32; LANES];
        let mut done = [0.0f32; LANES];
        for lane in 0..LANES {
            let noise = (self.next() % 7) as f32 * 0.01;
            reward[lane] = f32::from(ids[lane] == expert[lane]) + noise;
            self.clock[lane] += 1;
            if self.clock[lane] >= self.horizon[lane] {
                done[lane] = 1.0;
                self.clock[lane] = 0;
                self.episode[lane] += 1;
                self.horizon[lane] = 2 + (self.next() % 4) as u32;
            }
        }
        Ok(EnvStep {
            observation: self.observation()?,
            reward: Tensor::from_f32(&reward, vec![LANES], &dev())?,
            done: Tensor::from_f32(&done, vec![LANES], &dev())?,
        })
    }

    fn expert_actions(&self) -> Option<IdTensor<R>> {
        IdTensor::from_slice(&self.expert(), vec![LANES], &dev()).ok()
    }

    fn action_mask(&self) -> Result<Option<Tensor<R, f32>>> {
        let mut rows = vec![1.0f32; LANES * ACTIONS];
        for lane in 0..LANES {
            if self.clock[lane] == 0 {
                rows[lane * ACTIONS + lane % ACTIONS] = 0.0;
            }
        }
        Ok(Some(Tensor::from_f32(&rows, vec![LANES, ACTIONS], &dev())?))
    }

    fn save_state(&self) -> Result<Option<Vec<u8>>> {
        Ok(Some(
            StateWriter::new(ENV_TAG, 1)
                .u64(self.rng)
                .u32s(&self.clock)
                .u32s(&self.horizon)
                .u32s(&self.episode)
                .u32s(&self.log)
                .finish(),
        ))
    }

    fn load_state(&mut self, bytes: &[u8]) -> Result<()> {
        let mut input = StateReader::open(bytes, ENV_TAG, 1, "ResumableEnv")?;
        let rng = input.u64()?;
        let lanes = |v: Vec<u32>| -> Result<[u32; LANES]> {
            v.try_into()
                .map_err(|_| Error::StateDict("wrong lane count".to_string()))
        };
        let clock = lanes(input.u32s()?)?;
        let horizon = lanes(input.u32s()?)?;
        let episode = lanes(input.u32s()?)?;
        let log = input.u32s()?;
        input.finish()?;
        *self = Self {
            rng,
            clock,
            horizon,
            episode,
            log,
        };
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Two learners, spelled out
// ---------------------------------------------------------------------------

fn policy(seed: u64) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(OBS, ACTIONS, 16, 1)
        .with_seed(seed)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
            s.conv_kernel = Some(3);
        })
        .init::<R, f32>(&dev())
        .expect("a policy")
}

/// [`trainer`], with a moving average of `source` in `shadow` when `ema` is set.
fn trainer_with(
    source: &Mamba3Policy<R, f32>,
    shadow: &Mamba3Policy<R, f32>,
    ema: Option<EmaConfig>,
) -> Trainer<R, f32, AdamW<R, f32>> {
    match ema {
        Some(config) => trainer().with_ema(Ema::new(source, shadow, config).expect("an EMA")),
        None => trainer(),
    }
}

/// The average's fingerprint and update counter, when there is one.
fn ema_state(trainer: &Trainer<R, f32, AdamW<R, f32>>) -> Option<(String, u64)> {
    trainer
        .ema()
        .map(|ema| (ema.state_dict().fingerprint(), ema.updates()))
}

fn trainer() -> Trainer<R, f32, AdamW<R, f32>> {
    Trainer::new(
        TrainerConfig::builder()
            .learning_rate(4e-3)
            .max_grad_norm(0.5)
            .schedule(LrSchedule::cosine(40))
            .build()
            .expect("a trainer config"),
        AdamWConfig::builder()
            .learning_rate(4e-3)
            .build()
            .init::<R, f32>(),
    )
}

/// Everything one round produced that a continuation has to reproduce. Floats
/// are kept as bit patterns.
#[derive(Debug, Clone)]
struct Round {
    /// `[envs, steps, obs_dim]`: every observation the window acted on.
    observations: Vec<u32>,
    /// `[envs, obs_dim]`: the observation the next window starts from.
    next_observation: Vec<u32>,
    actions: Vec<u32>,
    rewards: Vec<u32>,
    masks: Vec<u32>,
    /// Reward summed over the episodes the window completed, and how many.
    episode_return_sum: Vec<u32>,
    episode_return_count: Vec<u32>,
    /// Each lane's return so far in the episode still running.
    running_return: Vec<u32>,
    /// What [`Collector::episode_return`] reports.
    episode_return_mean: Vec<u32>,
    reference: Vec<u32>,
    learning_rate: u32,
    step: u64,
    loss: f32,
    /// The moving average's fingerprint and update counter, if one is attached.
    ema: Option<(String, u64)>,
}

impl Round {
    /// Every field bit for bit, the loss included.
    ///
    /// The loss was once held to a few ulp: `Grads` kept its gradients in a
    /// `HashMap`, whose per-instance random order changed the order the global
    /// norm was summed in, so two identical runs drifted apart from the first
    /// clipped update. With ordered gradients they agree to the bit.
    fn assert_matches(&self, other: &Round, what: &str) {
        assert_eq!(
            self.observations, other.observations,
            "{what}: observations"
        );
        assert_eq!(
            self.next_observation, other.next_observation,
            "{what}: next observation"
        );
        assert_eq!(self.actions, other.actions, "{what}: actions");
        assert_eq!(self.rewards, other.rewards, "{what}: rewards");
        assert_eq!(self.masks, other.masks, "{what}: masks");
        assert_eq!(
            self.episode_return_sum, other.episode_return_sum,
            "{what}: completed episodes' total return"
        );
        assert_eq!(
            self.episode_return_count, other.episode_return_count,
            "{what}: completed episodes"
        );
        assert_eq!(
            self.running_return, other.running_return,
            "{what}: running returns"
        );
        assert_eq!(
            self.episode_return_mean, other.episode_return_mean,
            "{what}: mean episode return"
        );
        assert_eq!(self.reference, other.reference, "{what}: reference scores");
        assert_eq!(
            self.learning_rate, other.learning_rate,
            "{what}: learning rate"
        );
        assert_eq!(self.step, other.step, "{what}: optimizer step");
        assert_eq!(
            self.loss.to_bits(),
            other.loss.to_bits(),
            "{what}: loss {} against {}",
            self.loss,
            other.loss
        );
        assert_eq!(self.ema, other.ema, "{what}: moving average");
    }
}

fn bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

fn ppo_config() -> PpoConfig {
    PpoConfig {
        reference_coeff: 0.3,
        ..PpoConfig::default()
    }
}

impl Round {
    /// Read what the window just collected and trained on left behind.
    fn observe(
        collector: &Collector<'_, R, f32>,
        reference: Vec<u32>,
        info: &StepInfo,
        ema: Option<(String, u64)>,
    ) -> Self {
        let buffer = collector.buffer();
        let state = collector.export_state().expect("the collector's state");
        let saved = |name: &str| bits(state.tensors.entries[name].data.clone());
        let (mean, _) = collector.episode_return().expect("episode returns");
        Round {
            observations: bits(buffer.observations().to_f32()),
            next_observation: saved("observation"),
            actions: buffer.actions().to_vec(),
            rewards: bits(buffer.rewards().to_f32()),
            masks: bits(buffer.action_mask().expect("masked").to_f32()),
            episode_return_sum: saved("episode_return_sum"),
            episode_return_count: saved("episode_return_count"),
            running_return: saved("running_return"),
            episode_return_mean: bits(mean.to_f32()),
            reference,
            learning_rate: info.learning_rate.to_bits(),
            step: info.step,
            loss: info.loss,
            ema,
        }
    }
}

/// Where a round's window comes from: one environment beside its collector, or
/// a pool of worker threads bundled with one.
trait Rollouts<'a> {
    fn collect(&mut self) -> Result<CollectReport<R, f32>>;
    fn collect_with_expert(&mut self, beta: f32) -> Result<CollectReport<R, f32>>;
    fn collector(&self) -> &Collector<'a, R, f32>;
}

struct Single<'c, 'a> {
    collector: &'c mut Collector<'a, R, f32>,
    env: &'c mut ResumableEnv,
}

impl<'a> Rollouts<'a> for Single<'_, 'a> {
    fn collect(&mut self) -> Result<CollectReport<R, f32>> {
        self.collector.collect(self.env)
    }
    fn collect_with_expert(&mut self, beta: f32) -> Result<CollectReport<R, f32>> {
        self.collector.collect_with_expert(self.env, beta)
    }
    fn collector(&self) -> &Collector<'a, R, f32> {
        self.collector
    }
}

impl<'a> Rollouts<'a> for MultiSyncCollector<'a, R, f32> {
    fn collect(&mut self) -> Result<CollectReport<R, f32>> {
        MultiSyncCollector::collect(self)
    }
    fn collect_with_expert(&mut self, beta: f32) -> Result<CollectReport<R, f32>> {
        MultiSyncCollector::collect_with_expert(self, beta)
    }
    fn collector(&self) -> &Collector<'a, R, f32> {
        MultiSyncCollector::collector(self)
    }
}

fn ppo_round(
    policy: &Mamba3Policy<R, f32>,
    collector: &mut Collector<'_, R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    reference: &mut ReferencePolicy<R, f32>,
    env: &mut ResumableEnv,
) -> Round {
    ppo_round_on(policy, &mut Single { collector, env }, trainer, reference)
}

fn ppo_round_on<'a>(
    policy: &Mamba3Policy<R, f32>,
    source: &mut impl Rollouts<'a>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    reference: &mut ReferencePolicy<R, f32>,
) -> Round {
    let config = ppo_config();
    let report = source.collect().expect("a window");
    let batch = source
        .collector()
        .ppo_batch(&report, &config)
        .expect("a batch");
    let scores = reference.score(&batch).expect("reference scores");
    let batch = batch.with_reference_log_probs(scores.clone());
    let task = PpoTask::new(policy, config);
    let mut info = None;
    for _ in 0..2 {
        info = Some(
            trainer
                .step(&task, std::slice::from_ref(&batch))
                .expect("a step"),
        );
    }
    let info = info.expect("two epochs");
    Round::observe(
        source.collector(),
        bits(scores.to_f32()),
        &info,
        ema_state(trainer),
    )
}

fn imitation_round(
    policy: &Mamba3Policy<R, f32>,
    collector: &mut Collector<'_, R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    env: &mut ResumableEnv,
    round: usize,
) -> Round {
    imitation_round_on(policy, &mut Single { collector, env }, trainer, round)
}

fn imitation_round_on<'a>(
    policy: &Mamba3Policy<R, f32>,
    source: &mut impl Rollouts<'a>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    round: usize,
) -> Round {
    let beta = DaggerSchedule::Exponential { decay: 0.7 }.beta(round as u32);
    source.collect_with_expert(beta).expect("a DAgger window");
    let batch = source
        .collector()
        .imitation_batch()
        .expect("a labelled batch");
    let task = BehaviourCloningTask::new(policy).with_entropy_bonus(0.01);
    let info = trainer
        .step(&task, std::slice::from_ref(&batch))
        .expect("a step");
    Round::observe(source.collector(), Vec::new(), &info, ema_state(trainer))
}

fn weights(policy: &Mamba3Policy<R, f32>) -> Vec<(String, Vec<u32>)> {
    Checkpoint::capture(policy, 0)
        .state
        .entries
        .into_iter()
        .map(|(name, t)| (name, bits(t.data)))
        .collect()
}

/// Final weights, bit for bit (see [`Round::assert_matches`] on why this once
/// needed a tolerance and no longer does).
fn assert_weights_equal(got: &Mamba3Policy<R, f32>, want: &Mamba3Policy<R, f32>) {
    for ((name, got), (_, want)) in weights(got).iter().zip(weights(want).iter()) {
        let differ = got.iter().zip(want).filter(|(g, w)| g != w).count();
        assert_eq!(differ, 0, "{differ} values of {name} differ");
    }
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mamba3-rl-resume-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir.join(name)
}

/// Write a training checkpoint plus the rollout, as a learner would.
fn save(
    path: &std::path::Path,
    policy: &Mamba3Policy<R, f32>,
    trainer: &Trainer<R, f32, AdamW<R, f32>>,
    collector: &Collector<'_, R, f32>,
    reference: Option<&ReferencePolicy<R, f32>>,
    env: &dyn VecEnv<R, f32>,
) {
    let snapshot =
        RolloutSnapshot::capture(collector, reference, env).expect("the environment saves");
    save_snapshot(path, policy, trainer, snapshot);
}

fn save_snapshot(
    path: &std::path::Path,
    policy: &Mamba3Policy<R, f32>,
    trainer: &Trainer<R, f32, AdamW<R, f32>>,
    snapshot: RolloutSnapshot,
) {
    let mut checkpoint = Checkpoint::capture(policy, trainer.step_count())
        .with_optimizer(policy, trainer.optimizer())
        .with_metadata(serde_json::json!({"note": "rl_resume"}));
    if let Some(ema) = trainer.ema() {
        checkpoint = checkpoint.with_ema(ema);
    }
    snapshot
        .attach(checkpoint)
        .expect("metadata is an object")
        .save(path)
        .expect("a written checkpoint");
}

/// The reference a checkpoint was saved with, rebuilt from the file alone onto
/// `policy`'s architecture.
fn saved_reference(
    path: &std::path::Path,
    policy: &Mamba3Policy<R, f32>,
) -> ReferencePolicy<R, f32> {
    let checkpoint = Checkpoint::load(path).expect("a readable checkpoint");
    let weights = checkpoint
        .reference
        .as_ref()
        .expect("the reference's weights");
    ReferencePolicy::from_weights(policy.config(), weights, &dev()).expect("a reference")
}

/// Restore everything from `path` into fresh objects, all or nothing.
fn restore(
    path: &std::path::Path,
    policy: &Mamba3Policy<R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    collector: &mut Collector<'_, R, f32>,
    reference: Option<&mut ReferencePolicy<R, f32>>,
    env: &mut dyn VecEnv<R, f32>,
) {
    try_restore(path, policy, trainer, collector, reference, env).expect("a full restore");
}

/// [`restore`], returning what refused it. Everything is staged — weights,
/// optimizer, rollout, the trainer's moving average — before the environment
/// is handed its bytes, and nothing is applied until all of it has succeeded.
fn try_restore(
    path: &std::path::Path,
    policy: &Mamba3Policy<R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    collector: &mut Collector<'_, R, f32>,
    reference: Option<&mut ReferencePolicy<R, f32>>,
    env: &mut dyn VecEnv<R, f32>,
) -> Result<()> {
    let checkpoint = Checkpoint::load(path)?;
    let mut fresh = self::trainer();
    let (weights, _) = checkpoint.stage_training(policy, fresh.optimizer_mut(), true)?;
    fresh.set_step_count(checkpoint.step);
    let snapshot = RolloutSnapshot::from_checkpoint(&checkpoint)?
        .ok_or_else(|| Error::StateDict("not a full checkpoint".to_string()))?;
    let staged = snapshot.stage(collector, reference.as_deref())?;
    let staged_ema = trainer
        .ema()
        .map(|ema| checkpoint.stage_ema(ema, true))
        .transpose()?;
    staged.apply(collector, reference, env)?;
    weights.apply();
    if let (Some(staged_ema), Some(mut ema)) = (staged_ema, trainer.take_ema()) {
        staged_ema.apply(&mut ema);
        fresh = fresh.with_ema(ema);
    }
    *trainer = fresh;
    Ok(())
}

#[test]
fn ppo_continues_exactly_from_a_full_checkpoint() {
    ppo_continues_exactly(None, false);
}

/// R12: the same, with a moving average of the weights, which continues to the
/// bit — and at the optimizer level too, for training on a given window.
#[test]
fn ppo_continues_exactly_with_ema() {
    ppo_continues_exactly(Some(EmaConfig::new(0.8).with_warmup(EmaWarmup::Tf)), false);
}

/// R14: restore everything but re-seed the average from the loaded weights.
/// The weights still continue exactly and the average does not, which is what
/// shows R12 and R13 can fail.
#[test]
fn without_the_ema_state_the_continuation_diverges() {
    ppo_continues_exactly(Some(EmaConfig::new(0.8)), true);
}

/// N rounds, a full checkpoint, fresh objects, M rounds, against N + M. With
/// `reseed_ema`, the restored average is re-seeded from the loaded weights
/// instead, and only it is expected to differ.
fn ppo_continues_exactly(ema: Option<EmaConfig>, reseed_ema: bool) {
    let device = dev();
    let frozen = policy(11);

    // The run that never stops.
    let a_policy = policy(7);
    let a_shadow = policy(21);
    let mut a_env = ResumableEnv::new(1);
    let mut a_collector = Collector::new(&a_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_temperature(1.0)
        .with_seed(5);
    let mut a_trainer = trainer_with(&a_policy, &a_shadow, ema);
    let mut a_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    let mut expected = Vec::new();
    let mut first = Vec::new();
    for round in 0..FIRST + SECOND {
        let r = ppo_round(
            &a_policy,
            &mut a_collector,
            &mut a_trainer,
            &mut a_reference,
            &mut a_env,
        );
        if round >= FIRST {
            expected.push(r);
        } else {
            first.push(r);
        }
    }

    // The run that stops after FIRST rounds...
    let b_policy = policy(7);
    let b_shadow = policy(22);
    let mut b_env = ResumableEnv::new(1);
    let mut b_collector = Collector::new(&b_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_temperature(1.0)
        .with_seed(5);
    let mut b_trainer = trainer_with(&b_policy, &b_shadow, ema);
    let mut b_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    for (round, first) in first.iter().enumerate() {
        let r = ppo_round(
            &b_policy,
            &mut b_collector,
            &mut b_trainer,
            &mut b_reference,
            &mut b_env,
        );
        // The comparison below means nothing unless the uninterrupted run is
        // itself reproducible.
        r.assert_matches(first, &format!("two uninterrupted runs, round {round}"));
    }
    // One file per variant: the tests run in parallel in one process.
    let path = scratch(&format!(
        "ppo-{}-{reseed_ema}.m3ck",
        if ema.is_some() { "ema" } else { "plain" }
    ));
    save(
        &path,
        &b_policy,
        &b_trainer,
        &b_collector,
        Some(&b_reference),
        &b_env,
    );

    // ...and objects that know nothing but the file: other weights, another
    // environment seed, another sampling seed, and the reference rebuilt from
    // the weights the checkpoint carries rather than from `frozen`.
    let c_policy = policy(99);
    let c_shadow = policy(23);
    let mut c_env = ResumableEnv::new(42);
    let mut c_collector = Collector::new(&c_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_temperature(0.5)
        .with_seed(17);
    let mut c_trainer = trainer_with(&c_policy, &c_shadow, ema);
    let mut c_reference = saved_reference(&path, &c_policy);
    assert_eq!(c_reference.fingerprint(), b_reference.fingerprint());
    restore(
        &path,
        &c_policy,
        &mut c_trainer,
        &mut c_collector,
        Some(&mut c_reference),
        &mut c_env,
    );
    assert_eq!(c_trainer.step_count(), b_trainer.step_count());
    assert_eq!(ema_state(&c_trainer), ema_state(&b_trainer));
    if reseed_ema {
        let loaded = Checkpoint::load(&path).unwrap().state;
        c_trainer
            .ema_mut()
            .expect("an EMA")
            .load_state_dict(&loaded, 0, true)
            .unwrap();
    }
    let got: Vec<Round> = (0..SECOND)
        .map(|_| {
            ppo_round(
                &c_policy,
                &mut c_collector,
                &mut c_trainer,
                &mut c_reference,
                &mut c_env,
            )
        })
        .collect();

    for (index, (got, want)) in got.iter().zip(&expected).enumerate() {
        let what = format!("round {} after the restore", FIRST + index);
        if reseed_ema {
            assert_ne!(
                got.ema, want.ema,
                "{what}: a re-seeded average continued exactly"
            );
            Round {
                ema: None,
                ..got.clone()
            }
            .assert_matches(
                &Round {
                    ema: None,
                    ..want.clone()
                },
                &what,
            );
        } else {
            got.assert_matches(want, &what);
        }
    }
    assert_eq!(c_env.log, a_env.log, "the environments' action logs differ");
    assert_weights_equal(&c_policy, &a_policy);

    // The optimizer level: weights, optimizer and average restored without the
    // rollout train on a given window exactly as the saved learner does.
    if ema.is_some() && !reseed_ema {
        let d_policy = policy(98);
        let d_shadow = policy(24);
        let mut d_trainer = trainer_with(&d_policy, &d_shadow, ema);
        let checkpoint = Checkpoint::load(&path).unwrap();
        let mut d_fresh = trainer();
        checkpoint
            .restore_training(&d_policy, d_fresh.optimizer_mut(), true)
            .unwrap();
        d_fresh.set_step_count(checkpoint.step);
        let mut d_ema = d_trainer.take_ema().unwrap();
        checkpoint.restore_ema(&mut d_ema, true).unwrap();
        d_trainer = d_fresh.with_ema(d_ema);

        let config = ppo_config();
        let report = b_collector.collect(&mut b_env).unwrap();
        let batch = b_collector.ppo_batch(&report, &config).unwrap();
        let scores = b_reference.score(&batch).unwrap();
        let batch = batch.with_reference_log_probs(scores);
        for (policy, trainer) in [(&b_policy, &mut b_trainer), (&d_policy, &mut d_trainer)] {
            let task = PpoTask::new(policy, config);
            for _ in 0..2 {
                trainer.step(&task, std::slice::from_ref(&batch)).unwrap();
            }
        }
        assert_weights_equal(&d_policy, &b_policy);
        assert_eq!(ema_state(&d_trainer), ema_state(&b_trainer));
        assert!(ema_state(&d_trainer).is_some_and(|(_, updates)| updates > 0));
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn imitation_continues_exactly_from_a_full_checkpoint() {
    imitation_continues_exactly(None);
}

/// R13: the imitation continuation with a moving average of the weights.
#[test]
fn imitation_continues_exactly_with_ema() {
    imitation_continues_exactly(Some(EmaConfig::new(0.6)));
}

fn imitation_continues_exactly(ema: Option<EmaConfig>) {
    let device = dev();

    let a_policy = policy(7);
    let a_shadow = policy(31);
    let mut a_env = ResumableEnv::new(3);
    let mut a_collector = Collector::new(&a_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(9)
        .recording_expert_labels();
    let mut a_trainer = trainer_with(&a_policy, &a_shadow, ema);
    let mut expected = Vec::new();
    for round in 0..FIRST + SECOND {
        let r = imitation_round(
            &a_policy,
            &mut a_collector,
            &mut a_trainer,
            &mut a_env,
            round,
        );
        if round >= FIRST {
            expected.push(r);
        }
    }

    let b_policy = policy(7);
    let b_shadow = policy(32);
    let mut b_env = ResumableEnv::new(3);
    let mut b_collector = Collector::new(&b_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(9)
        .recording_expert_labels();
    let mut b_trainer = trainer_with(&b_policy, &b_shadow, ema);
    for round in 0..FIRST {
        imitation_round(
            &b_policy,
            &mut b_collector,
            &mut b_trainer,
            &mut b_env,
            round,
        );
    }
    let path = scratch("imitation.m3ck");
    save(&path, &b_policy, &b_trainer, &b_collector, None, &b_env);

    let c_policy = policy(98);
    let c_shadow = policy(33);
    let mut c_env = ResumableEnv::new(77);
    let mut c_collector = Collector::new(&c_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(1)
        .recording_expert_labels();
    let mut c_trainer = trainer_with(&c_policy, &c_shadow, ema);
    restore(
        &path,
        &c_policy,
        &mut c_trainer,
        &mut c_collector,
        None,
        &mut c_env,
    );
    let got: Vec<Round> = (FIRST..FIRST + SECOND)
        .map(|round| {
            imitation_round(
                &c_policy,
                &mut c_collector,
                &mut c_trainer,
                &mut c_env,
                round,
            )
        })
        .collect();

    for (index, (got, want)) in got.iter().zip(&expected).enumerate() {
        got.assert_matches(want, &format!("round {} after the restore", FIRST + index));
    }
    assert_eq!(c_env.log, a_env.log);
    assert_weights_equal(&c_policy, &a_policy);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn without_the_rollout_state_the_continuation_diverges() {
    // The positive tests above are only worth something if skipping the rollout
    // state would fail them: restore the training state alone and the actions
    // differ from the first window on.
    let device = dev();
    let frozen = policy(11);
    let a_policy = policy(7);
    let mut a_env = ResumableEnv::new(1);
    let mut a_collector = Collector::new(&a_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(5);
    let mut a_trainer = trainer();
    let mut a_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    for _ in 0..FIRST {
        ppo_round(
            &a_policy,
            &mut a_collector,
            &mut a_trainer,
            &mut a_reference,
            &mut a_env,
        );
    }
    let checkpoint = Checkpoint::capture(&a_policy, a_trainer.step_count())
        .with_optimizer(&a_policy, a_trainer.optimizer());
    let want = ppo_round(
        &a_policy,
        &mut a_collector,
        &mut a_trainer,
        &mut a_reference,
        &mut a_env,
    );

    let c_policy = policy(99);
    let mut c_env = ResumableEnv::new(1);
    let mut c_collector = Collector::new(&c_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(5);
    let mut c_trainer = trainer();
    checkpoint
        .restore_training(&c_policy, c_trainer.optimizer_mut(), true)
        .unwrap();
    c_trainer.set_step_count(checkpoint.step);
    let mut c_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    let got = ppo_round(
        &c_policy,
        &mut c_collector,
        &mut c_trainer,
        &mut c_reference,
        &mut c_env,
    );
    assert_ne!(got.actions, want.actions);
}

#[test]
fn a_refused_restore_changes_nothing() {
    let device = dev();
    let frozen = policy(11);
    let b_policy = policy(7);
    let mut b_env = ResumableEnv::new(1);
    let mut b_collector = Collector::new(&b_policy, LANES, WINDOW, OBS, &device).unwrap();
    let mut b_trainer = trainer();
    let mut b_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    ppo_round(
        &b_policy,
        &mut b_collector,
        &mut b_trainer,
        &mut b_reference,
        &mut b_env,
    );
    let path = scratch("refused.m3ck");
    save(
        &path,
        &b_policy,
        &b_trainer,
        &b_collector,
        Some(&b_reference),
        &b_env,
    );
    let snapshot = RolloutSnapshot::from_checkpoint(&Checkpoint::load(&path).unwrap())
        .unwrap()
        .unwrap();

    // A collector of another window length: refused at staging.
    let other = Collector::new(&b_policy, LANES, WINDOW + 1, OBS, &device).unwrap();
    let err = snapshot
        .stage(&other, Some(&b_reference))
        .err()
        .expect("refused");
    assert!(err.to_string().contains("windows"), "{err}");
    // No reference where the snapshot was saved with one.
    assert!(snapshot.stage(&b_collector, None).is_err());
    // A reference with other weights: its scores would not continue the run.
    let impostor = ReferencePolicy::snapshot(&policy(12), &device).unwrap();
    let err = snapshot
        .stage(&b_collector, Some(&impostor))
        .err()
        .expect("refused");
    assert!(err.to_string().contains("different weights"), "{err}");
    assert!(snapshot.stage(&b_collector, Some(&b_reference)).is_ok());

    // An environment that refuses its bytes: the collector and reference are
    // left exactly as they were, which the next round shows.
    struct Refusing(ResumableEnv);
    impl VecEnv<R, f32> for Refusing {
        fn envs(&self) -> usize {
            self.0.envs()
        }
        fn obs_dim(&self) -> usize {
            self.0.obs_dim()
        }
        fn action_dim(&self) -> usize {
            self.0.action_dim()
        }
        fn reset(&mut self) -> Result<Tensor<R, f32>> {
            self.0.reset()
        }
        fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, f32>> {
            self.0.step(actions)
        }
        fn action_mask(&self) -> Result<Option<Tensor<R, f32>>> {
            self.0.action_mask()
        }
        fn load_state(&mut self, _: &[u8]) -> Result<()> {
            Err(Error::StateDict("not these bytes".to_string()))
        }
    }
    let twin_policy = policy(7);
    let mut twin_collector = Collector::new(&twin_policy, LANES, WINDOW, OBS, &device).unwrap();
    let mut twin_env = ResumableEnv::new(2);
    let target_policy = policy(7);
    let mut target_collector = Collector::new(&target_policy, LANES, WINDOW, OBS, &device).unwrap();
    let mut target_env = Refusing(ResumableEnv::new(2));
    twin_collector.collect(&mut twin_env).unwrap();
    target_collector.collect(&mut target_env).unwrap();
    let staged = snapshot.stage(&target_collector, None);
    assert!(staged.is_err(), "saved with a reference, staged without");
    let no_reference = RolloutSnapshot {
        reference_cache: None,
        reference_weights: None,
        ..snapshot.clone()
    };
    let staged = no_reference.stage(&target_collector, None).unwrap();
    let err = staged
        .apply(&mut target_collector, None, &mut target_env)
        .expect_err("the environment refuses");
    assert!(err.to_string().contains("not these bytes"));
    twin_collector.collect(&mut twin_env).unwrap();
    target_collector.collect(&mut target_env).unwrap();
    assert_eq!(
        target_collector.buffer().actions().to_vec(),
        twin_collector.buffer().actions().to_vec()
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_malformed_ema_is_refused_and_nothing_changes() {
    let device = dev();
    let frozen = policy(11);
    let ema = Some(EmaConfig::new(0.9));
    let b_policy = policy(7);
    let b_shadow = policy(21);
    let mut b_env = ResumableEnv::new(1);
    let mut b_collector = Collector::new(&b_policy, LANES, WINDOW, OBS, &device).unwrap();
    let mut b_trainer = trainer_with(&b_policy, &b_shadow, ema);
    let mut b_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    ppo_round(
        &b_policy,
        &mut b_collector,
        &mut b_trainer,
        &mut b_reference,
        &mut b_env,
    );
    let path = scratch("malformed-ema.m3ck");
    save(
        &path,
        &b_policy,
        &b_trainer,
        &b_collector,
        Some(&b_reference),
        &b_env,
    );
    // Everything in the file is valid but one entry of the average's shape.
    let mut checkpoint = Checkpoint::load(&path).unwrap();
    let entry = checkpoint
        .ema
        .as_mut()
        .unwrap()
        .entries
        .values_mut()
        .next()
        .unwrap();
    entry.shape.push(1);
    checkpoint.save(&path).unwrap();

    // A target and its twin, which never tries to load: after the refusal the
    // target's next round — weights, moments, rollout, reference cache, average —
    // must be the twin's.
    let build = |seed: u64| {
        let policy = policy(seed);
        let shadow = policy_shadow(seed);
        (policy, shadow)
    };
    let (t_policy, t_shadow) = build(50);
    let (w_policy, w_shadow) = build(50);
    let mut t_env = ResumableEnv::new(8);
    let mut w_env = ResumableEnv::new(8);
    let mut t_collector = Collector::new(&t_policy, LANES, WINDOW, OBS, &device).unwrap();
    let mut w_collector = Collector::new(&w_policy, LANES, WINDOW, OBS, &device).unwrap();
    let mut t_trainer = trainer_with(&t_policy, &t_shadow, ema);
    let mut w_trainer = trainer_with(&w_policy, &w_shadow, ema);
    let mut t_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    let mut w_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    ppo_round(
        &t_policy,
        &mut t_collector,
        &mut t_trainer,
        &mut t_reference,
        &mut t_env,
    )
    .assert_matches(
        &ppo_round(
            &w_policy,
            &mut w_collector,
            &mut w_trainer,
            &mut w_reference,
            &mut w_env,
        ),
        "before the load",
    );
    let err = try_restore(
        &path,
        &t_policy,
        &mut t_trainer,
        &mut t_collector,
        Some(&mut t_reference),
        &mut t_env,
    )
    .expect_err("a malformed average is refused");
    assert!(err.to_string().contains("EMA"), "{err}");
    assert!(
        t_trainer.ema().is_some(),
        "the refused load took the average away"
    );
    ppo_round(
        &t_policy,
        &mut t_collector,
        &mut t_trainer,
        &mut t_reference,
        &mut t_env,
    )
    .assert_matches(
        &ppo_round(
            &w_policy,
            &mut w_collector,
            &mut w_trainer,
            &mut w_reference,
            &mut w_env,
        ),
        "after the refused load",
    );
    assert_weights_equal(&t_policy, &w_policy);
    let _ = std::fs::remove_file(&path);
}

/// A shadow for [`policy`]`(seed)`'s average.
fn policy_shadow(seed: u64) -> Mamba3Policy<R, f32> {
    policy(seed + 1000)
}

// ---------------------------------------------------------------------------
// The built-in environments
// ---------------------------------------------------------------------------

fn same_steps<V: VecEnv<R, f32>>(a: &mut V, b: &mut V, steps: usize) {
    for t in 0..steps {
        let ids: Vec<u32> = (0..a.envs())
            .map(|i| ((i + t) % a.action_dim()) as u32)
            .collect();
        let ids = IdTensor::from_slice(&ids, vec![a.envs()], &dev()).unwrap();
        let x = a.step(&ids).unwrap();
        let y = b.step(&ids).unwrap();
        assert_eq!(
            x.observation.to_f32(),
            y.observation.to_f32(),
            "observation at {t}"
        );
        assert_eq!(x.reward.to_f32(), y.reward.to_f32(), "reward at {t}");
        assert_eq!(x.done.to_f32(), y.done.to_f32(), "done at {t}");
        assert_eq!(
            a.expert_actions().map(|e| e.to_vec()),
            b.expert_actions().map(|e| e.to_vec())
        );
    }
}

#[test]
fn recall_env_state_round_trips_and_refuses_what_is_not_its_own() {
    let device = dev();
    let mut a = RecallEnv::<R, f32>::new(8, 4, 5, 3, &device).unwrap();
    a.reset().unwrap();
    let ids = IdTensor::from_slice(&[1, 2, 3, 0, 1, 2, 3, 0], vec![8], &device).unwrap();
    for _ in 0..7 {
        a.step(&ids).unwrap();
    }
    let bytes = a.save_state().unwrap().expect("RecallEnv saves");

    let mut b = RecallEnv::<R, f32>::new(8, 4, 5, 3, &device).unwrap();
    b.load_state(&bytes).unwrap();
    same_steps(&mut a, &mut b, 12);

    // Before a reset there is nothing to carry but the counter.
    let fresh = RecallEnv::<R, f32>::new(8, 4, 5, 3, &device).unwrap();
    let mut loaded = RecallEnv::<R, f32>::new(8, 4, 5, 3, &device).unwrap();
    loaded
        .load_state(&fresh.save_state().unwrap().unwrap())
        .unwrap();

    // Refusals, each leaving the environment as it was.
    let mut target = RecallEnv::<R, f32>::new(8, 4, 5, 3, &device).unwrap();
    target.reset().unwrap();
    let mut twin = RecallEnv::<R, f32>::new(8, 4, 5, 3, &device).unwrap();
    twin.reset().unwrap();
    let other_seed = RecallEnv::<R, f32>::new(8, 4, 5, 4, &device).unwrap();
    for (what, bad) in [
        ("truncated", bytes[..bytes.len() - 3].to_vec()),
        ("another seed", other_seed.save_state().unwrap().unwrap()),
        ("trailing", [bytes.clone(), vec![0]].concat()),
        ("foreign", b"not a recall state at all".to_vec()),
    ] {
        assert!(target.load_state(&bad).is_err(), "{what} was accepted");
    }
    same_steps(&mut target, &mut twin, 6);
}

#[test]
fn a_game_world_continues_exactly_through_the_fused_rollout() {
    let device = dev();
    let run = |world: &mut GameWorld<R, f32, Recall>, collector: &mut Collector<'_, R, f32>| {
        collector.collect_fused(world).unwrap();
        (
            collector.buffer().actions().to_vec(),
            bits(collector.buffer().rewards().to_f32()),
            bits(collector.buffer().observations().to_f32()),
        )
    };
    let spec = recall_spec(4);
    let net = policy_for(spec.obs_dim, spec.action_dim, 7);

    let mut a_world = GameWorld::<R, f32, Recall>::new(8, spec, 2, &device).unwrap();
    let mut a_collector = Collector::new(&net, 8, 6, spec.obs_dim, &device)
        .unwrap()
        .with_seed(4);
    for _ in 0..2 {
        run(&mut a_world, &mut a_collector);
    }
    let bytes = a_world.save_state().unwrap().unwrap();
    let state = a_collector.export_state().unwrap();
    let want: Vec<_> = (0..2)
        .map(|_| run(&mut a_world, &mut a_collector))
        .collect();

    let mut b_world = GameWorld::<R, f32, Recall>::new(8, spec, 2, &device).unwrap();
    let mut b_collector = Collector::new(&net, 8, 6, spec.obs_dim, &device)
        .unwrap()
        .with_seed(11);
    b_world.load_state(&bytes).unwrap();
    b_collector
        .stage_state(&state)
        .unwrap()
        .apply(&mut b_collector);
    let got: Vec<_> = (0..2)
        .map(|_| run(&mut b_world, &mut b_collector))
        .collect();
    let first = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(x, y)| x != y);
    for (round, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.0, w.0, "actions of window {round}");
        assert_eq!(
            g.1,
            w.1,
            "rewards of window {round}, from {:?}",
            first(&g.1, &w.1)
        );
        assert_eq!(
            g.2,
            w.2,
            "observations of window {round}, from {:?}",
            first(&g.2, &w.2)
        );
    }

    // A world of the same game with another seed refuses the bytes.
    let mut other = GameWorld::<R, f32, Recall>::new(8, spec, 3, &device).unwrap();
    assert!(other.load_state(&bytes).is_err());
}

fn policy_for(obs_dim: usize, actions: usize, seed: u64) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(obs_dim, actions, 16, 1)
        .with_seed(seed)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .init::<R, f32>(&dev())
        .unwrap()
}

#[test]
fn parallel_workers_continue_exactly() {
    // Host environments that read their actions back on their worker threads,
    // which is also the regression test for reading a buffer on a thread other
    // than the one that allocated it (`backend::read_handle`).
    let device = dev();
    let net = policy(7);
    let build = |seed: u64| {
        ParallelEnvs::new(
            vec![ResumableEnv::new(seed), ResumableEnv::new(seed + 1)],
            &device,
        )
        .unwrap()
    };
    let run = |envs: &mut ParallelEnvs<R, f32>, collector: &mut Collector<'_, R, f32>| {
        collector.collect(envs).unwrap();
        (
            collector.buffer().actions().to_vec(),
            bits(collector.buffer().rewards().to_f32()),
            bits(collector.buffer().action_mask().unwrap().to_f32()),
            bits(collector.buffer().observations().to_f32()),
            envs.expert_actions().map(|e| e.to_vec()),
        )
    };

    let mut a_envs = build(5);
    let mut a_collector = Collector::new(&net, 2 * LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(2);
    run(&mut a_envs, &mut a_collector);
    let bytes = a_envs.save_state().unwrap().expect("every worker saves");
    let state = a_collector.export_state().unwrap();
    let want: Vec<_> = (0..2).map(|_| run(&mut a_envs, &mut a_collector)).collect();

    let mut b_envs = build(50);
    let mut b_collector = Collector::new(&net, 2 * LANES, WINDOW, OBS, &device).unwrap();
    b_envs.load_state(&bytes).unwrap();
    b_collector
        .stage_state(&state)
        .unwrap()
        .apply(&mut b_collector);
    let got: Vec<_> = (0..2).map(|_| run(&mut b_envs, &mut b_collector)).collect();
    assert_eq!(got, want);

    // A pool of another shape refuses the bytes.
    let mut narrow = ParallelEnvs::new(vec![ResumableEnv::new(1)], &device).unwrap();
    assert!(narrow.load_state(&bytes).is_err());
}

fn multi_sync<'a>(
    policy: &'a Mamba3Policy<R, f32>,
    dagger: bool,
    env_seed: u64,
    draw_seed: u64,
    temperature: f32,
) -> MultiSyncCollector<'a, R, f32> {
    let envs = vec![ResumableEnv::new(env_seed), ResumableEnv::new(env_seed + 1)];
    let collector = MultiSyncCollector::new(policy, envs, WINDOW, &dev())
        .unwrap()
        .with_temperature(temperature)
        .with_seed(draw_seed);
    if dagger {
        collector.recording_expert_labels()
    } else {
        collector
    }
}

/// A PPO round when there is a reference, a DAgger round when there is not.
fn pooled_round(
    policy: &Mamba3Policy<R, f32>,
    collector: &mut MultiSyncCollector<'_, R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    reference: Option<&mut ReferencePolicy<R, f32>>,
    round: usize,
) -> Round {
    match reference {
        Some(reference) => ppo_round_on(policy, collector, trainer, reference),
        None => imitation_round_on(policy, collector, trainer, round),
    }
}

/// [`ppo_continues_exactly_from_a_full_checkpoint`] and its imitation twin, on
/// a pool of worker threads, saved and restored through the
/// [`MultiSyncCollector`] itself.
fn multi_sync_continues_exactly(dagger: bool) {
    let device = dev();
    let frozen = policy(11);
    let reference = || (!dagger).then(|| ReferencePolicy::snapshot(&frozen, &device).unwrap());
    let pooled_state = |collector: &MultiSyncCollector<'_, R, f32>| {
        collector
            .environments()
            .save_state()
            .unwrap()
            .expect("every worker saves")
    };

    let a_policy = policy(7);
    let mut a_collector = multi_sync(&a_policy, dagger, 1, 5, 1.0);
    let mut a_trainer = trainer();
    let mut a_reference = reference();
    let mut first = Vec::new();
    let mut expected = Vec::new();
    for round in 0..FIRST + SECOND {
        let r = pooled_round(
            &a_policy,
            &mut a_collector,
            &mut a_trainer,
            a_reference.as_mut(),
            round,
        );
        if round >= FIRST {
            expected.push(r);
        } else {
            first.push(r);
        }
    }

    let b_policy = policy(7);
    let mut b_collector = multi_sync(&b_policy, dagger, 1, 5, 1.0);
    let mut b_trainer = trainer();
    let mut b_reference = reference();
    for (round, want) in first.iter().enumerate() {
        pooled_round(
            &b_policy,
            &mut b_collector,
            &mut b_trainer,
            b_reference.as_mut(),
            round,
        )
        .assert_matches(want, &format!("two uninterrupted pools, round {round}"));
    }
    let path = scratch(if dagger {
        "multi-sync-dagger.m3ck"
    } else {
        "multi-sync-ppo.m3ck"
    });
    let snapshot = b_collector
        .capture_rollout(b_reference.as_ref())
        .expect("every worker saves");
    save_snapshot(&path, &b_policy, &b_trainer, snapshot);

    // Other weights, other worker seeds, another draw seed and temperature, and
    // any reference rebuilt from the file.
    let c_policy = policy(99);
    let mut c_collector = multi_sync(&c_policy, dagger, 40, 17, 0.5);
    let mut c_reference = (!dagger).then(|| saved_reference(&path, &c_policy));
    let checkpoint = Checkpoint::load(&path).unwrap();
    assert_eq!(checkpoint.reference.is_some(), !dagger);
    let mut c_trainer = trainer();
    let (weights, _) = checkpoint
        .stage_training(&c_policy, c_trainer.optimizer_mut(), true)
        .expect("weights and optimizer stage");
    c_trainer.set_step_count(checkpoint.step);
    let snapshot = RolloutSnapshot::from_checkpoint(&checkpoint)
        .unwrap()
        .expect("a full checkpoint");

    // Refusals change nothing: a reference with other weights...
    let before = pooled_state(&c_collector);
    if !dagger {
        let mut impostor = ReferencePolicy::snapshot(&policy(12), &device).unwrap();
        let err = c_collector
            .restore_rollout(&snapshot, Some(&mut impostor))
            .expect_err("refused");
        assert!(err.to_string().contains("different weights"), "{err}");
    }
    // ...a missing or unexpected reference...
    let mut spare = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    let wrong = if dagger { Some(&mut spare) } else { None };
    assert!(c_collector.restore_rollout(&snapshot, wrong).is_err());
    assert_eq!(pooled_state(&c_collector), before);
    // ...and a pool of another width.
    let mut narrow =
        MultiSyncCollector::new(&c_policy, vec![ResumableEnv::new(3)], WINDOW, &device).unwrap();
    assert!(
        narrow
            .restore_rollout(&snapshot, c_reference.as_mut())
            .is_err()
    );

    c_collector
        .restore_rollout(&snapshot, c_reference.as_mut())
        .expect("the pool restores");
    weights.apply();
    assert_eq!(pooled_state(&c_collector), pooled_state(&b_collector));
    for (index, want) in expected.iter().enumerate() {
        let round = FIRST + index;
        pooled_round(
            &c_policy,
            &mut c_collector,
            &mut c_trainer,
            c_reference.as_mut(),
            round,
        )
        .assert_matches(want, &format!("round {round} after the restore"));
    }
    // Every worker's generator, clocks and action log, and the expert labels the
    // pool cached for the next observation.
    assert_eq!(pooled_state(&c_collector), pooled_state(&a_collector));
    assert_eq!(
        c_collector
            .environments()
            .expert_actions()
            .map(|e| e.to_vec()),
        a_collector
            .environments()
            .expert_actions()
            .map(|e| e.to_vec())
    );
    assert_weights_equal(&c_policy, &a_policy);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_multi_sync_collector_continues_ppo_exactly_from_a_full_checkpoint() {
    multi_sync_continues_exactly(false);
}

#[test]
fn a_multi_sync_collector_continues_dagger_exactly_from_a_full_checkpoint() {
    multi_sync_continues_exactly(true);
}

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

#[test]
fn reference_weights_round_trip_in_both_formats() {
    let device = dev();
    let net = policy(7);
    let frozen = ReferencePolicy::snapshot(&policy(11), &device).unwrap();
    let checkpoint = Checkpoint::capture(&net, 2).with_reference(frozen.weights());

    let binary = scratch("reference.m3ck");
    checkpoint.save(&binary).unwrap();
    let bytes = std::fs::read(&binary).unwrap();
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 3);
    let json = scratch("reference.json");
    checkpoint.save(&json).unwrap();
    for path in [&binary, &json] {
        let loaded = Checkpoint::load(path).unwrap();
        let weights = loaded.reference.as_ref().expect("the reference's weights");
        let rebuilt =
            ReferencePolicy::<R, f32>::from_weights(net.config(), weights, &device).unwrap();
        assert_eq!(rebuilt.fingerprint(), frozen.fingerprint(), "{path:?}");
        assert!(rebuilt.cache().is_none());
        // Weights for another architecture are refused.
        let other = policy_for(OBS, ACTIONS, 7);
        assert!(ReferencePolicy::<R, f32>::from_weights(other.config(), weights, &device).is_err());
    }
    // A different reference fingerprints differently.
    let impostor = ReferencePolicy::snapshot(&policy(12), &device).unwrap();
    assert_ne!(impostor.fingerprint(), frozen.fingerprint());
    for path in [binary, json] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn rollout_state_lives_in_the_binary_format_only() {
    let device = dev();
    let net = policy(7);
    let mut env = ResumableEnv::new(1);
    let mut collector = Collector::new(&net, LANES, WINDOW, OBS, &device).unwrap();
    collector.collect(&mut env).unwrap();
    let with_rollout = RolloutSnapshot::capture(&collector, None, &env)
        .unwrap()
        .attach(Checkpoint::capture(&net, 3))
        .unwrap();

    let json = scratch("rollout.json");
    let err = with_rollout.save(&json).expect_err("JSON cannot hold it");
    assert!(err.to_string().contains(".m3ck"), "{err}");

    let binary = scratch("rollout.m3ck");
    with_rollout.save(&binary).unwrap();
    let bytes = std::fs::read(&binary).unwrap();
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 3);
    let loaded = Checkpoint::load(&binary).unwrap();
    assert_eq!(loaded.blobs, with_rollout.blobs);
    assert_eq!(
        loaded.rollout.as_ref().map(|r| r.entries.len()),
        with_rollout.rollout.as_ref().map(|r| r.entries.len())
    );
    let snapshot = RolloutSnapshot::from_checkpoint(&loaded).unwrap().unwrap();
    assert_eq!(
        snapshot.collector.draws,
        collector.export_state().unwrap().draws
    );
    assert_eq!(snapshot.env, env.save_state().unwrap().unwrap());

    // A checkpoint without rollout state is still written as version 2, which
    // builds from before version 3 can read.
    let plain = scratch("plain.m3ck");
    Checkpoint::capture(&net, 3).save(&plain).unwrap();
    let bytes = std::fs::read(&plain).unwrap();
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
    assert!(
        RolloutSnapshot::from_checkpoint(&Checkpoint::load(&plain).unwrap())
            .unwrap()
            .is_none()
    );

    // An environment that cannot save is refused when the snapshot is taken.
    struct Stateless(ResumableEnv);
    impl VecEnv<R, f32> for Stateless {
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
            self.0.reset()
        }
        fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, f32>> {
            self.0.step(actions)
        }
    }
    let err = RolloutSnapshot::capture(&collector, None, &Stateless(ResumableEnv::new(1)))
        .expect_err("refused");
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    for path in [json, binary, plain] {
        let _ = std::fs::remove_file(path);
    }
}
