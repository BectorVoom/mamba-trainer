//! T1: an exponential moving average of the weights.
//!
//! The kernel against its host twin, then [`Ema`] and [`Trainer::with_ema`]
//! against a host recomputation of the documented recurrence.
//!
//! # CPU and GPU
//!
//! Every assertion that compares the device with a **host** recomputation is
//! exact on the CPU runtime, the reference backend. Elsewhere it is exact when
//! the device agrees bit for bit, and otherwise held to `|Δ| ≤ 4·ε·max(1, |x|)`
//! per element, printing a `skipped:` line that says so: the recurrence is only
//! `+ − ×`, so the realistic source of a difference is a shader compiler
//! contracting `e + c·(p − e)` into a fused multiply-add, which the budget covers.
//! Assertions that compare the device with **itself** (with and without the
//! average, two identical runs) are exact everywhere.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, FloatElem};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::nn::module::StateDict;
use mamba3::nn::{Module, ModuleVisitor, Param};
use mamba3::rl::{
    Collector, Mamba3Policy, Mamba3PolicyConfig, PpoBatch, PpoConfig, PpoTask, RecallEnv, VecEnv,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::fused::ema_step;
use mamba3::train::{
    AdamW, AdamWConfig, Ema, EmaConfig, EmaWarmup, Optimizer, StepInfo, TrainStep, Trainer,
    TrainerConfig,
};

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// The documented step, in the documented order.
fn ema_host(ema: &[f32], param: &[f32], one_minus_decay: f32) -> Vec<f32> {
    ema.iter()
        .zip(param)
        .map(|(&e, &p)| e + one_minus_decay * (p - e))
        .collect()
}

/// Deterministic values of mixed sign and magnitude, zeros of both signs among them.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|i| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            match i % 17 {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    let unit = (bits >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    let scale = [1e-3f32, 0.1, 1.0, 30.0][(bits & 3) as usize];
                    unit * scale
                }
            }
        })
        .collect()
}

/// Values that differ from the host twin: none on the CPU runtime, and within
/// the budget elsewhere (see the module docs), or the test fails.
fn twin_differences(what: &str, got: &[f32], want: &[f32]) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let differ: Vec<usize> = (0..got.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    if let Some(&first) = differ.first() {
        assert!(
            dev().name() != "cpu",
            "{what}: {} of {} values differ from the host twin on the CPU runtime, from index \
             {first}: {} against {}",
            differ.len(),
            got.len(),
            got[first],
            want[first]
        );
    }
    for &i in &differ {
        let budget = 4.0 * f32::EPSILON * want[i].abs().max(1.0);
        assert!(
            (got[i] - want[i]).abs() <= budget,
            "{what}: index {i} is {} against the twin's {}, over the budget {budget:e}",
            got[i],
            want[i]
        );
    }
    differ.len()
}

/// Tallies [`twin_differences`] over a test and reports once, as a skip of the
/// bit-exact claim, when any value was only within budget.
#[derive(Default)]
struct Twin {
    differ: usize,
    total: usize,
}

impl Twin {
    fn check(&mut self, what: &str, got: &[f32], want: &[f32]) {
        self.differ += twin_differences(what, got, want);
        self.total += got.len();
    }

    fn report(&self, test: &str) {
        if self.differ > 0 {
            println!(
                "skipped: {test} bit-exact on {} ({} of {} values differ from the host twin); \
                 held to 4·ε·max(1, |x|) instead",
                dev().name(),
                self.differ,
                self.total
            );
        }
    }
}

#[test]
fn ema_step_matches_its_host_twin() {
    let device = dev();
    // Lengths around every vector width a device offers (1 to 64) and their
    // tails, and one long enough for the CPU runtime to split across threads.
    let lengths = [
        0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 49, 63, 64, 65, 97, 193, 1000, 100_003,
    ];
    let mut twin = Twin::default();
    for (k, &n) in lengths.iter().enumerate() {
        let ema = values(n, 1 + k as u64);
        let param = values(n, 101 + k as u64);
        let e = Tensor::<R, f32>::from_f32(&ema, vec![n], &device).unwrap();
        let p = Tensor::<R, f32>::from_f32(&param, vec![n], &device).unwrap();
        for decay in [0.0f32, 0.5, 0.9, 0.999, 1.0] {
            let c = 1.0 - decay;
            let out = ema_step(&e, &p, c).unwrap();
            assert_eq!(out.shape().dims(), &[n]);
            twin.check(
                &format!("n={n}, decay={decay}"),
                &out.try_to_f32().unwrap(),
                &ema_host(&ema, &param, c),
            );
        }
        // Neither input is written.
        assert_eq!(
            e.try_to_f32().unwrap(),
            ema,
            "the average was written in place"
        );
        assert_eq!(
            p.try_to_f32().unwrap(),
            param,
            "the weight was written in place"
        );
    }

    twin.report("ema_step_matches_its_host_twin");

    // Shapes that disagree are refused rather than read out of bounds.
    let a = Tensor::<R, f32>::zeros(vec![2, 3], &device);
    let b = Tensor::<R, f32>::zeros(vec![3, 2], &device);
    assert!(ema_step(&a, &b, 0.1).is_err());
}

// ---------------------------------------------------------------------------
// Ema and Trainer::with_ema
// ---------------------------------------------------------------------------

const ENVS: usize = 4;
const SYMBOLS: usize = 4;
const HORIZON: usize = 3;
const WINDOW: usize = 6;

fn obs_dim() -> usize {
    RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, 5, &dev())
        .unwrap()
        .obs_dim()
}

fn policy_with(seed: u64, d_model: usize, n_layers: usize) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(obs_dim(), SYMBOLS, d_model, n_layers)
        .with_seed(seed)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = d_model / 2;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
            s.conv_kernel = Some(3);
        })
        .init::<R, f32>(&dev())
        .expect("a policy")
}

fn policy(seed: u64) -> Mamba3Policy<R, f32> {
    policy_with(seed, 16, 1)
}

/// One window collected by `policy`, trained on repeatedly.
fn window(policy: &Mamba3Policy<R, f32>) -> PpoBatch<R, f32> {
    let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, 5, &dev()).unwrap();
    let mut collector = Collector::new(policy, ENVS, WINDOW, env.obs_dim(), &dev())
        .unwrap()
        .with_temperature(1.0)
        .with_seed(3);
    let report = collector.collect(&mut env).unwrap();
    collector.ppo_batch(&report, &PpoConfig::default()).unwrap()
}

fn trainer() -> Trainer<R, f32, AdamW<R, f32>> {
    Trainer::new(
        TrainerConfig::builder()
            .learning_rate(1e-2)
            .max_grad_norm(0.5)
            .build()
            .unwrap(),
        AdamWConfig::builder()
            .learning_rate(1e-2)
            .build()
            .init::<R, f32>(),
    )
}

/// Every weight of `model`, by path.
fn weights(model: &Mamba3Policy<R, f32>) -> StateDict {
    model.state_dict()
}

fn bits_of(dict: &StateDict) -> Vec<(String, Vec<u32>)> {
    dict.entries
        .iter()
        .map(|(name, t)| (name.clone(), t.data.iter().map(|v| v.to_bits()).collect()))
        .collect()
}

fn step(
    trainer: &mut Trainer<R, f32, AdamW<R, f32>>,
    task: &impl TrainStep<R, f32, Batch = PpoBatch<R, f32>>,
    batch: &PpoBatch<R, f32>,
) -> StepInfo {
    trainer.step(task, std::slice::from_ref(batch)).unwrap()
}

/// Five steps, each checked against the documented step applied on the host
/// to the average the device held after the step before. (A one-step twin
/// rather than a five-step one, so the budget off the CPU runtime is per step;
/// on the CPU runtime every step is exact, so the two are the same claim.)
fn recurrence_matches_host(config: EmaConfig, test: &str) {
    let source = policy(7);
    let shadow = policy(8);
    let batch = window(&source);
    let task = PpoTask::new(&source, PpoConfig::default());
    let mut trainer = trainer().with_ema(Ema::new(&source, &shadow, config).unwrap());
    assert_eq!(
        bits_of(&weights(&shadow)),
        bits_of(&weights(&source)),
        "ema_0 is not θ_0"
    );
    let first = weights(&source);
    let mut twin = Twin::default();
    for t in 1..=5u64 {
        let before = weights(&shadow);
        let info = step(&mut trainer, &task, &batch);
        assert_eq!(info.step, t);
        let theta = weights(&source);
        let after = trainer.ema().unwrap().state_dict();
        let d = config.decay_at(t);
        for (name, e) in &before.entries {
            let want = ema_host(&e.data, &theta.entries[name].data, 1.0 - d);
            twin.check(
                &format!("{name} at step {t}"),
                &after.entries[name].data,
                &want,
            );
        }
        assert_eq!(bits_of(&after), bits_of(&weights(&shadow)));
        assert_eq!(trainer.ema().unwrap().updates(), t);
    }
    // Not vacuous: the average is neither the weights nor where it started.
    let last = weights(&shadow);
    assert_ne!(bits_of(&last), bits_of(&weights(&source)));
    assert_ne!(bits_of(&last), bits_of(&first));
    twin.report(test);
}

#[test]
fn recurrence_matches_host_over_five_steps() {
    recurrence_matches_host(
        EmaConfig::new(0.9),
        "recurrence_matches_host_over_five_steps",
    );
}

#[test]
fn decay_zero_tracks_weights_and_decay_one_keeps_the_start() {
    for decay in [0.0f32, 1.0] {
        let source = policy(7);
        let shadow = policy(8);
        let batch = window(&source);
        let task = PpoTask::new(&source, PpoConfig::default());
        let start = weights(&source);
        let mut trainer =
            trainer().with_ema(Ema::new(&source, &shadow, EmaConfig::new(decay)).unwrap());
        for _ in 0..5 {
            step(&mut trainer, &task, &batch);
            if decay == 0.0 {
                assert_eq!(bits_of(&weights(&shadow)), bits_of(&weights(&source)));
            }
        }
        assert_ne!(
            bits_of(&weights(&source)),
            bits_of(&start),
            "training moved nothing"
        );
        if decay == 1.0 {
            assert_eq!(bits_of(&weights(&shadow)), bits_of(&start));
        }
        assert_eq!(trainer.ema().unwrap().updates(), 5);
    }
}

#[test]
fn tf_warmup_uses_the_documented_decay() {
    let decay = 0.99f32;
    let tf = EmaConfig::new(decay).with_warmup(EmaWarmup::Tf);
    let none = EmaConfig::new(decay);
    for t in [1u64, 2, 3, 4, 5, 100, 1_000_000] {
        let ramp = ((1 + t) as f64 / (10 + t) as f64) as f32;
        assert_eq!(tf.decay_at(t).to_bits(), decay.min(ramp).to_bits(), "t={t}");
        assert_eq!(none.decay_at(t).to_bits(), decay.to_bits(), "t={t}");
    }
    assert_eq!(tf.decay_at(1), 2.0 / 11.0);
    assert_eq!(tf.decay_at(1_000_000), decay);
    let source = policy(1);
    let shadow = policy(2);
    let ema = Ema::new(&source, &shadow, tf).unwrap();
    assert_eq!(ema.decay_at(3), tf.decay_at(3));
    recurrence_matches_host(tf, "tf_warmup_uses_the_documented_decay");
}

#[test]
fn invalid_configs_are_refused() {
    let source = policy(7);
    let shadow = policy(8);
    for decay in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.0001] {
        let config = EmaConfig::new(decay);
        assert!(
            matches!(config.validate(), Err(Error::Config(_))),
            "{decay}"
        );
        let err = Ema::new(&source, &shadow, config).err().expect("refused");
        assert!(matches!(err, Error::Config(_)), "{decay}: {err:?}");
    }
    assert!(EmaConfig::new(0.0).validate().is_ok());
    assert!(EmaConfig::new(1.0).validate().is_ok());

    let source_before = bits_of(&weights(&source));
    let shadow_before = bits_of(&weights(&shadow));
    // A shape different, a path missing, and the source itself as its shadow.
    let narrower = policy_with(8, 8, 1);
    let deeper = policy_with(8, 16, 2);
    for (what, other) in [
        ("a shape", &narrower),
        ("a path", &deeper),
        ("itself", &source),
    ] {
        let other_before = bits_of(&weights(other));
        assert!(
            Ema::new(&source, other, EmaConfig::new(0.9)).is_err(),
            "{what} was accepted"
        );
        assert_eq!(
            bits_of(&weights(other)),
            other_before,
            "{what}: the shadow changed"
        );
    }
    let err = Ema::new(&source, &deeper, EmaConfig::new(0.9))
        .err()
        .unwrap();
    assert!(err.to_string().contains("parameters"), "{err}");
    let err = Ema::new(&source, &source, EmaConfig::new(0.9))
        .err()
        .unwrap();
    assert!(err.to_string().contains("shares"), "{err}");
    assert_eq!(bits_of(&weights(&source)), source_before);
    assert_eq!(bits_of(&weights(&shadow)), shadow_before);
}

/// The invariant every default rests on: attaching an average changes nothing
/// about training.
#[test]
fn ema_does_not_change_training() {
    let run = |with_ema: bool| {
        let source = policy(7);
        let shadow = policy(8);
        let batch = window(&source);
        let task = PpoTask::new(&source, PpoConfig::default());
        let mut trainer = trainer();
        if with_ema {
            trainer = trainer.with_ema(Ema::new(&source, &shadow, EmaConfig::new(0.9)).unwrap());
        }
        let infos: Vec<_> = (0..10)
            .map(|_| {
                let i = step(&mut trainer, &task, &batch);
                (
                    i.step,
                    i.loss.to_bits(),
                    i.learning_rate.to_bits(),
                    i.grad_norm.to_bits(),
                )
            })
            .collect();
        let moments = trainer.optimizer().state_dict(&source.named_parameters());
        (
            infos,
            bits_of(&weights(&source)),
            bits_of(&moments),
            trainer.optimizer().step_count(),
        )
    };
    let without = run(false);
    let with = run(true);
    assert_eq!(with.0, without.0, "StepInfo");
    assert_eq!(with.1, without.1, "weights");
    assert_eq!(with.2, without.2, "AdamW moments");
    assert_eq!(with.3, without.3, "AdamW step count");
    assert!(!without.2.is_empty());
}

#[test]
fn shadow_is_independent_of_its_source() {
    let source = policy(7);
    let shadow = policy(8);
    let batch = window(&source);
    let task = PpoTask::new(&source, PpoConfig::default());
    let mut with_ema = trainer().with_ema(Ema::new(&source, &shadow, EmaConfig::new(0.9)).unwrap());
    step(&mut with_ema, &task, &batch);
    for ((name, s), (_, h)) in source
        .named_parameters()
        .iter()
        .zip(shadow.named_parameters())
    {
        assert_ne!(s.id(), h.id(), "{name} is shared");
    }
    let averaged = bits_of(&weights(&shadow));

    // Reloading the source...
    source.load_state_dict(&weights(&policy(9)), true).unwrap();
    assert_eq!(
        bits_of(&weights(&shadow)),
        averaged,
        "reloading the source moved the average"
    );
    // ...acting with either model...
    for model in [&source, &shadow] {
        let mut env = RecallEnv::<R, f32>::new(ENVS, SYMBOLS, HORIZON, 6, &dev()).unwrap();
        Collector::new(model, ENVS, WINDOW, env.obs_dim(), &dev())
            .unwrap()
            .collect(&mut env)
            .unwrap();
    }
    assert_eq!(
        bits_of(&weights(&shadow)),
        averaged,
        "a rollout moved the average"
    );
    assert_eq!(bits_of(&with_ema.ema().unwrap().state_dict()), averaged);
    // ...and training the source through a trainer without the average.
    let mut plain = trainer();
    step(&mut plain, &task, &batch);
    assert_eq!(
        bits_of(&weights(&shadow)),
        averaged,
        "training moved the average"
    );
    // Only an update does.
    step(&mut with_ema, &task, &batch);
    assert_ne!(bits_of(&weights(&shadow)), averaged);
}

#[test]
fn frozen_parameters_follow_the_source() {
    let source = policy(7);
    let shadow = policy(8);
    source.freeze_matching(&["critic"]);
    let critic: Vec<String> = source
        .named_parameters()
        .into_iter()
        .filter(|(_, p)| !p.requires_grad())
        .map(|(n, _)| n)
        .collect();
    assert!(!critic.is_empty() && critic.iter().all(|n| n.starts_with("critic")));
    let batch = window(&source);
    let task = PpoTask::new(&source, PpoConfig::default());
    let config = EmaConfig::new(0.9);
    let mut trainer = trainer().with_ema(Ema::new(&source, &shadow, config).unwrap());
    let mut twin = Twin::default();
    for t in 1..=3u64 {
        let before = weights(&shadow);
        step(&mut trainer, &task, &batch);
        let theta = weights(&source);
        let after = weights(&shadow);
        for (name, e) in &before.entries {
            if critic.contains(name) {
                assert_eq!(
                    after.entries[name].data, theta.entries[name].data,
                    "{name} at step {t}"
                );
            } else {
                let want = ema_host(&e.data, &theta.entries[name].data, 1.0 - config.decay_at(t));
                twin.check(
                    &format!("{name} at step {t}"),
                    &after.entries[name].data,
                    &want,
                );
            }
        }
    }

    // A reload changes the frozen source; the next step's shadow follows it,
    // and the tensor the shadow was sharing is untouched by either.
    let held: Vec<(String, Tensor<R, f32>, Vec<u32>)> = shadow
        .named_parameters()
        .into_iter()
        .filter(|(n, _)| critic.contains(n))
        .map(|(n, p)| {
            let value = p.value();
            let bits = value.to_f32().iter().map(|v| v.to_bits()).collect();
            (n, value, bits)
        })
        .collect();
    // Shifted, so zero-initialised biases change too.
    let mut reloaded = weights(&policy(9));
    for entry in reloaded.entries.values_mut() {
        entry.data.iter_mut().for_each(|v| *v += 0.25);
    }
    source.load_state_dict(&reloaded, true).unwrap();
    step(&mut trainer, &task, &batch);
    let theta = weights(&source);
    let after = weights(&shadow);
    for (name, old, old_bits) in &held {
        assert_eq!(
            after.entries[name].data, theta.entries[name].data,
            "{name} after a reload"
        );
        assert_ne!(
            after.entries[name]
                .data
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            *old_bits,
            "{name}: the reload did not change the frozen weight, so this proves nothing"
        );
        let now: Vec<u32> = old.to_f32().iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            &now, old_bits,
            "{name}: a buffer the shadow shared was written"
        );
    }
    twin.report("frozen_parameters_follow_the_source");
}

/// A task whose loss is always refused.
struct Refusing<'a>(PpoTask<'a, R, f32>);

impl TrainStep<R, f32> for Refusing<'_> {
    type Batch = PpoBatch<R, f32>;
    fn parameters(&self) -> Vec<mamba3::nn::Param<R, f32>> {
        self.0.parameters()
    }
    fn loss(&self, _: &Self::Batch) -> mamba3::error::Result<mamba3::autograd::Var<R, f32>> {
        Err(Error::config("this task refuses every batch"))
    }
}

#[test]
fn a_step_that_fails_before_the_update_changes_nothing() {
    let source = policy(7);
    let shadow = policy(8);
    let batch = window(&source);
    let task = PpoTask::new(&source, PpoConfig::default());
    let mut trainer = trainer().with_ema(Ema::new(&source, &shadow, EmaConfig::new(0.9)).unwrap());
    step(&mut trainer, &task, &batch);
    let state = |trainer: &Trainer<R, f32, AdamW<R, f32>>| {
        (
            bits_of(&weights(&source)),
            bits_of(&trainer.ema().unwrap().state_dict()),
            trainer.ema().unwrap().updates(),
            trainer.step_count(),
        )
    };
    let before = state(&trainer);
    assert!(trainer.step(&task, &[]).is_err());
    assert_eq!(state(&trainer), before, "an empty step changed something");
    let refusing = Refusing(PpoTask::new(&source, PpoConfig::default()));
    let err = trainer
        .step(&refusing, std::slice::from_ref(&batch))
        .expect_err("refused");
    assert!(err.to_string().contains("refuses"), "{err}");
    assert_eq!(state(&trainer), before, "a refused loss changed something");
}

#[test]
fn identical_runs_agree_to_the_bit() {
    let run = |shadow_seed: u64| {
        let source = policy(7);
        let shadow = policy(shadow_seed);
        let batch = window(&source);
        let task = PpoTask::new(&source, PpoConfig::default());
        let config = EmaConfig::new(0.95).with_warmup(EmaWarmup::Tf);
        let mut trainer = trainer().with_ema(Ema::new(&source, &shadow, config).unwrap());
        for _ in 0..5 {
            step(&mut trainer, &task, &batch);
        }
        trainer.ema().unwrap().state_dict().fingerprint()
    };
    // The shadow's own initial weights are overwritten, so its seed is irrelevant.
    assert_eq!(run(8), run(9));
}

/// A model of one parameter, in any element type.
struct Tiny<E: FloatElem> {
    w: Param<R, E>,
}

impl<E: FloatElem> Module<R, E> for Tiny<E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("w", &self.w);
    }
}

fn tiny<E: FloatElem>() -> mamba3::error::Result<Tiny<E>> {
    Ok(Tiny {
        w: Param::new(Tensor::<R, E>::from_f32(&[1.0, 2.0], vec![2], &dev())?),
    })
}

#[test]
fn f16_weights_are_refused() {
    fn refused<E: FloatElem>(name: &str) {
        let (Ok(source), Ok(shadow)) = (tiny::<E>(), tiny::<E>()) else {
            println!(
                "skipped: f16_weights_are_refused ({name}): this device holds no {name} buffers"
            );
            return;
        };
        let err = Ema::new(&source, &shadow, EmaConfig::new(0.9))
            .err()
            .expect("refused");
        assert!(matches!(err, Error::Unsupported(_)), "{name}: {err:?}");
        assert!(err.to_string().contains("f32 or wider"), "{err}");
    }
    refused::<half::f16>("f16");
    refused::<half::bf16>("bf16");
    // The same model in f32 is accepted.
    let (source, shadow) = (tiny::<f32>().unwrap(), tiny::<f32>().unwrap());
    assert!(Ema::new(&source, &shadow, EmaConfig::new(0.9)).is_ok());
}
