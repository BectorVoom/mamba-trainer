//! A2b: an exact continuation.
//!
//! N rounds, a checkpoint written to disk, fresh objects built from nothing but
//! that file, M more rounds — against N + M rounds that never stopped. The two
//! must take the same actions, see the same rewards and masks, score the same
//! reference log-probabilities, apply the same learning rates, reach the same
//! counters and end on the same weights. The environment resets lanes at
//! different moments from its own random generator, so a continuation that drops
//! any of the state `mamba3::rl::snapshot` lists diverges within a window.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::error::{Error, Result};
use mamba3::rl::{
    BehaviourCloningTask, Collector, DaggerSchedule, EnvStep, GameWorld, Mamba3Policy,
    Mamba3PolicyConfig, ParallelEnvs, PpoConfig, PpoTask, Recall, RecallEnv, ReferencePolicy,
    RolloutSnapshot, StateReader, StateWriter, VecEnv, recall_spec,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{AdamW, AdamWConfig, Checkpoint, LrSchedule, Trainer, TrainerConfig};

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

/// Everything one round produced that a continuation has to reproduce.
#[derive(Debug)]
struct Round {
    actions: Vec<u32>,
    rewards: Vec<u32>,
    masks: Vec<u32>,
    reference: Vec<u32>,
    learning_rate: u32,
    step: u64,
    loss: f32,
}

impl Round {
    /// Every field bit for bit, except the reported loss.
    ///
    /// The PPO loss *scalar* is not bit-reproducible on the CPU runtime even
    /// between two uninterrupted runs in one process: measured, it differs by one
    /// ulp now and then (0.11835045 against 0.11835046), while the actions,
    /// rewards, reference scores, learning rates and final weights that depend on
    /// the same forward pass agree exactly. So it is held to a few ulp, and
    /// everything the run's future depends on to none.
    fn assert_matches(&self, other: &Round, what: &str) {
        assert_eq!(self.actions, other.actions, "{what}: actions");
        assert_eq!(self.rewards, other.rewards, "{what}: rewards");
        assert_eq!(self.masks, other.masks, "{what}: masks");
        assert_eq!(self.reference, other.reference, "{what}: reference scores");
        assert_eq!(
            self.learning_rate, other.learning_rate,
            "{what}: learning rate"
        );
        assert_eq!(self.step, other.step, "{what}: optimizer step");
        let tolerance = 4.0 * f32::EPSILON * self.loss.abs().max(1.0);
        assert!(
            (self.loss - other.loss).abs() <= tolerance,
            "{what}: loss {} against {}",
            self.loss,
            other.loss
        );
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

fn ppo_round(
    policy: &Mamba3Policy<R, f32>,
    collector: &mut Collector<'_, R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    reference: &mut ReferencePolicy<R, f32>,
    env: &mut ResumableEnv,
) -> Round {
    let config = ppo_config();
    let report = collector.collect(env).expect("a window");
    let batch = collector.ppo_batch(&report, &config).expect("a batch");
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
    let buffer = collector.buffer();
    Round {
        actions: buffer.actions().to_vec(),
        rewards: bits(buffer.rewards().to_f32()),
        masks: bits(buffer.action_mask().expect("masked").to_f32()),
        reference: bits(scores.to_f32()),
        learning_rate: info.learning_rate.to_bits(),
        step: info.step,
        loss: info.loss,
    }
}

fn imitation_round(
    policy: &Mamba3Policy<R, f32>,
    collector: &mut Collector<'_, R, f32>,
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    env: &mut ResumableEnv,
    round: usize,
) -> Round {
    let beta = DaggerSchedule::Exponential { decay: 0.7 }.beta(round as u32);
    collector
        .collect_with_expert(env, beta)
        .expect("a DAgger window");
    let batch = collector.imitation_batch().expect("a labelled batch");
    let task = BehaviourCloningTask::new(policy).with_entropy_bonus(0.01);
    let info = trainer
        .step(&task, std::slice::from_ref(&batch))
        .expect("a step");
    let buffer = collector.buffer();
    Round {
        actions: buffer.actions().to_vec(),
        rewards: bits(buffer.rewards().to_f32()),
        masks: bits(buffer.action_mask().expect("masked").to_f32()),
        reference: Vec::new(),
        learning_rate: info.learning_rate.to_bits(),
        step: info.step,
        loss: info.loss,
    }
}

fn weights(policy: &Mamba3Policy<R, f32>) -> Vec<(String, Vec<u32>)> {
    Checkpoint::capture(policy, 0)
        .state
        .entries
        .into_iter()
        .map(|(name, t)| (name, bits(t.data)))
        .collect()
}

/// Final weights to within a few ulp.
///
/// As for the loss (see [`Round::assert_matches`]): weights trained on the CPU
/// runtime agree between two uninterrupted runs in one process to a few ulp, not
/// always to the bit (up to 8 ulp in the runs measured here), so that is
/// the bound. A missing piece of rollout state moves them by orders of magnitude
/// more, and changes the sampled actions the rounds compare exactly.
fn assert_weights_close(got: &Mamba3Policy<R, f32>, want: &Mamba3Policy<R, f32>) {
    for ((name, got), (_, want)) in weights(got).iter().zip(weights(want).iter()) {
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, w)| (f32::from_bits(*g) - f32::from_bits(*w)).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-6, "{name} differs by {worst}");
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
    let checkpoint = Checkpoint::capture(policy, trainer.step_count())
        .with_optimizer(policy, trainer.optimizer())
        .with_metadata(serde_json::json!({"note": "rl_resume"}));
    RolloutSnapshot::capture(collector, reference, env)
        .expect("the environment saves")
        .attach(checkpoint)
        .expect("metadata is an object")
        .save(path)
        .expect("a written checkpoint");
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
    let checkpoint = Checkpoint::load(path).expect("a readable checkpoint");
    let mut fresh = self::trainer();
    let (weights, _) = checkpoint
        .stage_training(policy, fresh.optimizer_mut(), true)
        .expect("weights and optimizer stage");
    fresh.set_step_count(checkpoint.step);
    let snapshot = RolloutSnapshot::from_checkpoint(&checkpoint)
        .expect("a valid rollout section")
        .expect("a full checkpoint");
    let staged = snapshot
        .stage(collector, reference.as_deref())
        .expect("the rollout matches this collector");
    staged
        .apply(collector, reference, env)
        .expect("the environment accepts its bytes");
    weights.apply();
    *trainer = fresh;
}

#[test]
fn ppo_continues_exactly_from_a_full_checkpoint() {
    let device = dev();
    let frozen = policy(11);

    // The run that never stops.
    let a_policy = policy(7);
    let mut a_env = ResumableEnv::new(1);
    let mut a_collector = Collector::new(&a_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_temperature(1.0)
        .with_seed(5);
    let mut a_trainer = trainer();
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
    let mut b_env = ResumableEnv::new(1);
    let mut b_collector = Collector::new(&b_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_temperature(1.0)
        .with_seed(5);
    let mut b_trainer = trainer();
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
    let path = scratch("ppo.m3ck");
    save(
        &path,
        &b_policy,
        &b_trainer,
        &b_collector,
        Some(&b_reference),
        &b_env,
    );

    // ...and objects that know nothing but the file: other weights, another
    // environment seed, another sampling seed, a reference with no history.
    let c_policy = policy(99);
    let mut c_env = ResumableEnv::new(42);
    let mut c_collector = Collector::new(&c_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_temperature(0.5)
        .with_seed(17);
    let mut c_trainer = trainer();
    let mut c_reference = ReferencePolicy::snapshot(&frozen, &device).unwrap();
    restore(
        &path,
        &c_policy,
        &mut c_trainer,
        &mut c_collector,
        Some(&mut c_reference),
        &mut c_env,
    );
    assert_eq!(c_trainer.step_count(), b_trainer.step_count());
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
        got.assert_matches(want, &format!("round {} after the restore", FIRST + index));
    }
    assert_eq!(c_env.log, a_env.log, "the environments' action logs differ");
    assert_weights_close(&c_policy, &a_policy);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn imitation_continues_exactly_from_a_full_checkpoint() {
    let device = dev();

    let a_policy = policy(7);
    let mut a_env = ResumableEnv::new(3);
    let mut a_collector = Collector::new(&a_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(9)
        .recording_expert_labels();
    let mut a_trainer = trainer();
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
    let mut b_env = ResumableEnv::new(3);
    let mut b_collector = Collector::new(&b_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(9)
        .recording_expert_labels();
    let mut b_trainer = trainer();
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
    let mut c_env = ResumableEnv::new(77);
    let mut c_collector = Collector::new(&c_policy, LANES, WINDOW, OBS, &device)
        .unwrap()
        .with_seed(1)
        .recording_expert_labels();
    let mut c_trainer = trainer();
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
    assert_weights_close(&c_policy, &a_policy);
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
    // A reference the snapshot was not saved with, and the other way round.
    assert!(snapshot.stage(&b_collector, None).is_err());

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

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

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
