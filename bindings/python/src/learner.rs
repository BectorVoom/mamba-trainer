//! The two learning loops, and the numbers they report.
//!
//! Both loops are the same two halves in the same order — collect a window by
//! acting, then replay that window through the parallel scan and take a gradient
//! from it — and they differ only in what the replay is scored against:
//!
//! | | what it optimises | what it needs from the environment |
//! |---|---|---|
//! | [`PyPpoLearner`] | the return | a reward |
//! | [`PyImitationLearner`] | agreement with an expert | `expert_actions()` |
//!
//! Running the second and then the first is the standard recipe, and it is the
//! same policy object throughout: cloning an expert reaches a competent policy in
//! a handful of updates but can never exceed it, and PPO can exceed it but spends
//! most of its sample budget getting to where cloning arrives.
//!
//! # What crosses the boundary
//!
//! One round is one call. Inside it, the rollout, the action draws, the trajectory
//! writes, the advantage estimate and every gradient step are queued device work
//! that the host never waits on — with one deliberate exception: a PPO round ends
//! in a single read of every step's loss and gradient norm, the diagnostics and
//! the episode return, all numbers a human asked for. An environment written in
//! Python adds its own two copies per step, which is the price of that choice and
//! is visible in `mamba3::backend::read_count()`. A moving average of the weights
//! (`ema=`) adds no read at all: its update, its seeding and `reset_ema()` are
//! device work, and only `save` reads it back.

use mamba3::backend::Device;
use mamba3::rl::{
    BehaviourCloningTask, Collector, DaggerSchedule, PpoBatch, PpoConfig, PpoStats, PpoTask,
    QueuedAgreement, ReferencePolicy, RolloutSnapshot, StagedRollout,
};
use std::rc::Rc;

use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::{read_all, read_together};
use mamba3::train::{AdamW, Checkpoint, Ema, EmaConfig, Trainer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{PyClass, PyClassInitializer};

use crate::config::{PyEmaConfig, PyLrSchedule, PyPpoConfig};
use crate::env::{EnvHandle, check_against_policy, refuse_without_expert};
use crate::err::IntoPyResult;
use crate::policy::PyPolicy;
use crate::resume::{self, ConfigMode, Continuation, Level, LiveConfig, OptimSettings};
use crate::session::Session;
use crate::{E, R};

/// What one PPO round or update left behind.
///
/// `approx_kl` is the number to watch: it is what the clip is a proxy for, and a
/// run whose KL climbs across epochs is one whose ratio has left the region its
/// data supports — reduce `epochs` or the learning rate rather than the clip.
/// `episode_return` is `None` for a bare [`PyPpoLearner::update`], which does not
/// look at the environment.
#[pyclass(module = "mamba3_rl", name = "Stats", get_all, skip_from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct Stats {
    /// Rounds completed by this learner, counting from zero.
    pub round: u64,
    /// Steps in the window the update was taken from.
    pub steps: usize,
    /// Optimizer steps taken in total.
    pub optimizer_steps: u64,
    /// Mean total loss across the epochs of this update.
    pub loss: f32,
    /// Clipped surrogate objective, negated so lower is better.
    pub policy_loss: f32,
    /// Mean squared error of the critic against the lambda-returns.
    pub value_loss: f32,
    /// Mean policy entropy. Larger means it is still exploring.
    pub entropy: f32,
    /// Schulman's low-variance estimate of `KL(pi_old || pi_theta)`.
    pub approx_kl: f32,
    /// Fraction of positions whose probability ratio was clipped.
    pub clip_fraction: f32,
    /// Estimated `KL(policy || reference)`. Zero unless a reference anchors the run.
    pub reference_kl: f32,
    /// Mean global gradient norm before clipping.
    pub grad_norm: f32,
    /// Learning rate applied by the *last* of this update's optimizer steps —
    /// not a mean over them, unlike `loss` and `grad_norm` above. `epochs *
    /// minibatches` steps were taken to produce it; `optimizer_steps` is how
    /// many the learner has taken over its whole life, which is the argument a
    /// schedule (`lr_schedule=`) actually advances on.
    pub learning_rate: f32,
    /// Mean reward per completed episode in the collected window.
    pub episode_return: Option<f32>,
}

#[pymethods]
impl Stats {
    fn __repr__(&self) -> String {
        let episode_return = match self.episode_return {
            Some(value) => format!("{value:.4}"),
            None => "None".to_string(),
        };
        format!(
            "Stats(round={}, episode_return={episode_return}, loss={:.4}, \
             entropy={:.4}, approx_kl={:.5}, clip_fraction={:.3}, \
             reference_kl={:.5})",
            self.round,
            self.loss,
            self.entropy,
            self.approx_kl,
            self.clip_fraction,
            self.reference_kl,
        )
    }
}

/// What one imitation round left behind.
#[pyclass(
    module = "mamba3_rl",
    name = "CloneStats",
    get_all,
    skip_from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct CloneStats {
    /// Rounds completed, counting from zero.
    pub round: u64,
    /// Steps in the window that was labelled.
    pub steps: usize,
    /// Optimizer steps taken in total.
    pub optimizer_steps: u64,
    /// The expert's share of the acting in this round.
    pub beta: f32,
    /// Cross entropy against the expert's actions, with the entropy bonus.
    pub loss: f32,
    /// Fraction of positions where the policy's most likely action is the
    /// expert's, or `None` when it was not asked for.
    pub agreement: Option<f32>,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
    /// Learning rate applied.
    pub learning_rate: f32,
}

#[pymethods]
impl CloneStats {
    fn __repr__(&self) -> String {
        let agreement = match self.agreement {
            Some(value) => format!("{value:.3}"),
            None => "None".to_string(),
        };
        format!(
            "CloneStats(round={}, beta={:.3}, loss={:.4}, agreement={agreement})",
            self.round, self.beta, self.loss,
        )
    }
}

/// How the expert's share of the acting decays over DAgger's rounds.
///
/// Round `0` is always the expert's own trajectories under every schedule that
/// starts at `1`: a policy that has learned nothing visits nowhere worth labelling.
#[pyclass(module = "mamba3_rl", name = "DaggerSchedule", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyDaggerSchedule {
    pub(crate) inner: DaggerSchedule,
}

#[pymethods]
impl PyDaggerSchedule {
    /// `beta = decay ** round`. The usual choice.
    #[staticmethod]
    #[pyo3(signature = (decay = 0.5))]
    fn exponential(decay: f32) -> Self {
        Self {
            inner: DaggerSchedule::Exponential { decay },
        }
    }

    /// `beta = 1 - round / rounds`, reaching zero at `rounds`.
    #[staticmethod]
    fn linear(rounds: u32) -> Self {
        Self {
            inner: DaggerSchedule::Linear { rounds },
        }
    }

    /// Only round `0` uses the expert: behaviour cloning, then training on the
    /// learner's own states.
    #[staticmethod]
    fn only_first() -> Self {
        Self {
            inner: DaggerSchedule::OnlyFirst,
        }
    }

    /// A constant mixture, for ablations.
    #[staticmethod]
    fn fixed(beta: f32) -> Self {
        Self {
            inner: DaggerSchedule::Fixed { beta },
        }
    }

    /// The expert's share of the acting at `round`, counted from zero.
    fn beta(&self, round: u32) -> f32 {
        self.inner.beta(round)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            DaggerSchedule::Exponential { decay } => {
                format!("DaggerSchedule.exponential(decay={decay})")
            }
            DaggerSchedule::Linear { rounds } => format!("DaggerSchedule.linear(rounds={rounds})"),
            DaggerSchedule::OnlyFirst => "DaggerSchedule.only_first()".to_string(),
            DaggerSchedule::Fixed { beta } => format!("DaggerSchedule.fixed(beta={beta})"),
        }
    }
}

/// Validate `steps`, adopt `env`, check it against the policy's shape, and build
/// the [`Session`] that collects over it — the setup every learner and
/// [`evaluate`] share, differing only in whether an expert is required and
/// whether expert labels are recorded.
#[allow(clippy::too_many_arguments)]
fn setup_session(
    policy: &PyPolicy,
    env: &Bound<'_, PyAny>,
    steps: usize,
    temperature: f32,
    seed: u64,
    expert_labels: bool,
    require_expert: bool,
    zero_steps_msg: &'static str,
) -> PyResult<(EnvHandle, Session)> {
    if steps == 0 {
        return Err(PyValueError::new_err(zero_steps_msg));
    }
    let handle = EnvHandle::adopt(env, &policy.device)?;
    let shape = policy.inner.config();
    check_against_policy(&handle, shape.obs_dim, shape.action_dim)?;
    if require_expert {
        refuse_without_expert(&handle)?;
    }
    let session = Session::new(
        policy.share(),
        handle.envs(),
        steps,
        handle.obs_dim(),
        temperature,
        seed,
        expert_labels,
        &policy.device,
    )
    .py()?;
    Ok((handle, session))
}

/// The trainer both learners drive.
type LearnerTrainer = Trainer<R, E, AdamW<R, E>>;

/// `trainer` with a moving average of `policy`'s weights attached when `config`
/// is set, and the shadow policy that holds it: a separate policy of the same
/// architecture, seeded on the device with the current weights.
fn attach_ema(
    trainer: LearnerTrainer,
    policy: &PyPolicy,
    config: Option<EmaConfig>,
) -> PyResult<(LearnerTrainer, Option<resume::Shadow>)> {
    let Some(config) = config else {
        return Ok((trainer, None));
    };
    let shadow = Rc::new(policy.inner.config().init::<R, E>(&policy.device).py()?);
    let ema = Ema::new(&*policy.inner, &*shadow, config).py()?;
    Ok((trainer.with_ema(ema), Some(shadow)))
}

/// The average `from_checkpoint` rebuilds, from `trainer_config.ema`.
fn saved_ema(saved: &serde_json::Value) -> PyResult<Option<PyEmaConfig>> {
    Ok(
        PyEmaConfig::from_json(saved.get("ema").unwrap_or(&serde_json::Value::Null))?
            .map(|inner| PyEmaConfig { inner }),
    )
}

/// Call a per-round callback, and report whether the loop should continue.
///
/// Returning `False` from the callback stops the run; anything else, `None`
/// included, carries on — so the common case of a callback that only prints does
/// not have to return anything.
fn keep_going<T: PyClass + Into<PyClassInitializer<T>>>(
    py: Python<'_>,
    callback: Option<&Py<PyAny>>,
    stats: T,
) -> PyResult<(Py<T>, bool)> {
    let stats = Py::new(py, stats)?;
    let Some(callback) = callback else {
        return Ok((stats, true));
    };
    let returned = callback.call1(py, (stats.clone_ref(py),))?;
    let stop = returned.bind(py).is_instance_of::<pyo3::types::PyBool>()
        && !returned.bind(py).is_truthy()?;
    Ok((stats, !stop))
}

/// Proximal policy optimization over a recurrent policy.
///
/// One round is [`PyPpoLearner::collect`] — a window of `steps` steps over every
/// environment, acting entirely on the policy — followed by
/// [`PyPpoLearner::update`], which replays that window through the parallel scan
/// and takes `epochs` gradient steps against the behaviour policy's own
/// log-probabilities. Reusing one window for several epochs is what the importance
/// ratio is for, and what makes PPO sample-efficient enough to be worth running.
#[pyclass(module = "mamba3_rl", name = "PpoLearner", unsendable)]
pub struct PyPpoLearner {
    session: Session,
    env: EnvHandle,
    trainer: Trainer<R, E, AdamW<R, E>>,
    config: PpoConfig,
    batch: Option<PpoBatch<R, E>>,
    /// Whether `batch` was collected and not yet trained on — the one moment a
    /// full checkpoint cannot be taken, because the window it would have to
    /// carry is not part of one.
    pending: bool,
    /// Whether windows are collected through the fused rollout — one kernel per
    /// step over a compiled-in `game()` — rather than by stepping an environment.
    fused: bool,
    /// A frozen, independently snapshotted policy the run is priced against, for
    /// `PpoConfig.reference_coeff`. Scored once per window rather than once per
    /// epoch — it does not move — and keeps its own recurrent cache across
    /// windows, separate from the behaviour policy's.
    reference: Option<mamba3::rl::ReferencePolicy<R, E>>,
    /// The reference's weight fingerprint, computed on first save or load: one
    /// host read of weights that never change.
    reference_fingerprint: std::cell::OnceCell<String>,
    optim: OptimSettings,
    /// The policy the trainer's moving average lives in, when `ema=` was given.
    ema_policy: Option<resume::Shadow>,
    continuation: Continuation,
    steps: usize,
    rounds: u64,
    device: Device<R>,
}

#[pymethods]
impl PyPpoLearner {
    #[new]
    #[pyo3(signature = (
        policy,
        env,
        steps = 128,
        *,
        ppo = None,
        learning_rate = 3e-4,
        lr_schedule = None,
        max_grad_norm = 0.5,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
        temperature = 1.0,
        seed = 0,
        reference = None,
        fused = None,
        ema = None,
    ))]
    #[allow(clippy::too_many_arguments)] // Every one of them is a hyper-parameter.
    fn new(
        policy: &PyPolicy,
        env: &Bound<'_, PyAny>,
        steps: usize,
        ppo: Option<PyPpoConfig>,
        learning_rate: f32,
        lr_schedule: Option<PyLrSchedule>,
        max_grad_norm: f32,
        weight_decay: f32,
        betas: (f32, f32),
        eps: f32,
        temperature: f32,
        seed: u64,
        reference: Option<&PyPolicy>,
        fused: Option<bool>,
        ema: Option<PyEmaConfig>,
    ) -> PyResult<Self> {
        let config = ppo.unwrap_or_default().inner;
        config.validate().py()?;
        if config.reference_coeff != 0.0 && reference.is_none() {
            return Err(PyValueError::new_err(
                "ppo.reference_coeff is nonzero but no reference policy was given; \
                 pass PpoLearner(..., reference=some_policy) or leave reference_coeff at 0",
            ));
        }
        let (handle, session) = setup_session(
            policy,
            env,
            steps,
            temperature,
            seed,
            false,
            false,
            "a window needs at least one step",
        )?;
        // `ReferencePolicy::snapshot` deep-copies weights, so this is safe even
        // when `reference` is `policy` itself or shares its underlying `Rc`.
        let fused = match (fused, handle.is_game()) {
            (None, game) => game,
            (Some(true), false) => {
                return Err(PyValueError::new_err(
                    "fused=True needs a compiled-in device game from mamba3_rl.game(); this \
                     environment is stepped from the host",
                ));
            }
            (Some(choice), _) => choice,
        };
        let reference = reference
            .map(|p| mamba3::rl::ReferencePolicy::snapshot(&p.inner, &policy.device))
            .transpose()
            .py()?;
        let optim = OptimSettings {
            learning_rate,
            schedule: lr_schedule.unwrap_or_default().inner,
            max_grad_norm,
            weight_decay,
            betas,
            eps,
        };
        let (trainer, ema_policy) = attach_ema(optim.trainer()?, policy, ema.map(|e| e.inner))?;
        Ok(Self {
            session,
            env: handle,
            trainer,
            config,
            batch: None,
            pending: false,
            fused,
            reference,
            reference_fingerprint: std::cell::OnceCell::new(),
            optim,
            ema_policy,
            continuation: Continuation::fresh(),
            steps,
            rounds: 0,
            device: policy.device.clone(),
        })
    }

    /// The policy being optimised. The same weights, not a copy.
    #[getter]
    fn policy(&self) -> PyPolicy {
        PyPolicy::from_shared(self.session.policy(), self.device.clone())
    }

    /// The environment this learner drives.
    #[getter]
    fn env(&self, py: Python<'_>) -> Py<PyAny> {
        self.env.object(py)
    }

    /// Hyperparameters of the update.
    #[getter]
    fn config(&self) -> PyPpoConfig {
        PyPpoConfig { inner: self.config }
    }

    /// How many environments run in parallel.
    #[getter]
    fn num_envs(&self) -> usize {
        self.env.envs()
    }

    /// Steps in one window.
    #[getter]
    fn steps(&self) -> usize {
        self.steps
    }

    /// Rounds completed.
    #[getter]
    fn rounds(&self) -> u64 {
        self.rounds
    }

    /// `"fused"` when windows are collected by one kernel per step over a
    /// compiled-in `game()`, `"host"` when the environment is stepped between
    /// policy steps. A game collects fused unless built with `fused=False`.
    #[getter]
    fn collection_path(&self) -> &'static str {
        if self.fused { "fused" } else { "host" }
    }

    /// The window last collected, read back to numpy: `{"observations",
    /// "actions", "log_probs", "values", "rewards", "dones"}` shaped
    /// `[num_envs, steps, ...]`, plus `"action_mask"` when the window carried
    /// one. A synchronisation, for inspection and tests; the learning loop never
    /// needs it.
    fn window<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use numpy::PyArrayMethods;
        let buffer = self.session.collector().buffer();
        if buffer.is_empty() {
            return Err(PyValueError::new_err(
                "nothing has been collected yet; call collect() first",
            ));
        }
        let (envs, steps) = (buffer.envs(), buffer.len());
        let dict = pyo3::types::PyDict::new(py);
        let floats =
            |tensor: &mamba3::tensor::Tensor<R, E>, width: usize| -> PyResult<Bound<'py, PyAny>> {
                let full = tensor.try_to_f32().py()?;
                let capacity = buffer.steps();
                let mut kept = Vec::with_capacity(envs * steps * width);
                for env in 0..envs {
                    let start = env * capacity * width;
                    kept.extend_from_slice(&full[start..start + steps * width]);
                }
                let array = numpy::PyArray1::from_vec(py, kept);
                Ok(if width == 1 {
                    array.reshape((envs, steps))?.into_any()
                } else {
                    array.reshape((envs, steps, width))?.into_any()
                })
            };
        dict.set_item(
            "observations",
            floats(buffer.observations(), buffer.obs_dim())?,
        )?;
        dict.set_item("log_probs", floats(buffer.log_probs(), 1)?)?;
        dict.set_item("values", floats(buffer.values(), 1)?)?;
        dict.set_item("rewards", floats(buffer.rewards(), 1)?)?;
        dict.set_item("dones", floats(buffer.dones(), 1)?)?;
        if let Some(mask) = buffer.action_mask() {
            dict.set_item("action_mask", floats(mask, mask.shape().dim(2))?)?;
        }
        let ids = buffer.actions().try_to_vec().py()?;
        let capacity = buffer.steps();
        let mut kept = Vec::with_capacity(envs * steps);
        for env in 0..envs {
            kept.extend(
                ids[env * capacity..env * capacity + steps]
                    .iter()
                    .map(|&a| i64::from(a)),
            );
        }
        dict.set_item(
            "actions",
            numpy::PyArray1::from_vec(py, kept).reshape((envs, steps))?,
        )?;
        Ok(dict)
    }

    /// The moving average of the weights, as a policy, or `None` without
    /// `ema=`. A handle onto the average itself, not a copy: it keeps moving as
    /// the learner trains, and `save` (or its `fingerprint`) is how to keep a
    /// moment of it. Usable anywhere a policy is — `evaluate`, `Rollout`,
    /// `save` — except as the policy another learner trains.
    #[getter]
    fn ema_policy(&self) -> Option<PyPolicy> {
        self.ema_policy
            .as_ref()
            .map(|shadow| PyPolicy::from_shared(Rc::clone(shadow), self.device.clone()))
    }

    /// Optimizer steps the average has taken since it was built, reset or
    /// restored with its counter; `0` without one.
    #[getter]
    fn ema_updates(&self) -> u64 {
        self.trainer.ema().map_or(0, Ema::updates)
    }

    /// The average's configuration, or `None` without one.
    #[getter]
    fn ema_config(&self) -> Option<PyEmaConfig> {
        self.trainer.ema().map(|ema| PyEmaConfig {
            inner: *ema.config(),
        })
    }

    /// Restart the moving average from the current weights, and its counter from
    /// zero — after a critic-only warm-up, say. A device copy; no host read.
    /// Refused without an average, and between `collect()` and `update()`.
    fn reset_ema(&mut self) -> PyResult<()> {
        if self.pending {
            return Err(pending_window_error("an EMA reset"));
        }
        reset_trainer_ema(&mut self.trainer)
    }

    /// Bytes the trajectory buffer holds, fixed for the life of the learner.
    #[getter]
    fn buffer_bytes(&self) -> usize {
        self.session.collector().buffer().bytes()
    }

    /// Collect one window and prepare it as a PPO batch.
    ///
    /// Returns the number of steps collected. Nothing is read back: the advantage
    /// estimate, the action draws and the trajectory writes are all device kernels.
    fn collect(&mut self, py: Python<'_>) -> PyResult<usize> {
        let Self {
            session,
            env,
            config,
            reference,
            batch,
            ..
        } = self;
        let report = if self.fused {
            env.with_game(py, |world| world.collect_fused(session.collector_mut()))
                .expect("a fused learner is only ever built over a game")?
        } else {
            env.with(py, |mut vec_env| {
                session.collector_mut().collect(&mut vec_env)
            })?
        };
        let mut prepared = session.collector().ppo_batch(&report, config).py()?;
        // Scored here, once, rather than inside every epoch's loss: the reference
        // is frozen, so its answer for this window never changes. `score` continues
        // the reference's own cache from the previous window, not the actor's.
        if config.reference_coeff != 0.0
            && let Some(frozen) = reference.as_mut()
        {
            let scores = frozen.score(&prepared).py()?;
            prepared = prepared.with_reference_log_probs(scores);
        }
        *batch = Some(prepared);
        self.pending = true;
        Ok(report.steps)
    }

    /// Take `epochs` gradient steps over the window last collected.
    ///
    /// `minibatches` splits the *environment* axis, which is the only axis a
    /// recurrent policy can be split along: cutting across time would ask the
    /// replay to start mid-episode with no state. It must divide `num_envs`.
    ///
    /// `loss` and `grad_norm` in the result are means over every gradient step
    /// taken here; the five PPO diagnostics belong to the last one, which is the
    /// one whose `approx_kl` says whether the update went too far.
    ///
    /// Ends with one synchronisation, however many steps were taken: every
    /// step's loss and gradient norm and the diagnostics are read together.
    #[pyo3(signature = (epochs = 4, minibatches = 1))]
    fn update(&mut self, epochs: usize, minibatches: usize) -> PyResult<Stats> {
        self.update_reading(epochs, minibatches, false)
    }

    /// Mean reward per episode *completed* in the window last collected, or
    /// `None` if none completed — a window that ends mid-episode has nothing to
    /// average and must not be confused with one reporting a genuine mean.
    ///
    /// A synchronisation, and the one number worth paying it for: the mean and
    /// the count come back in a single read rather than one per value. `round()`
    /// reads it with the update's numbers instead, at no extra read.
    fn episode_return(&self) -> PyResult<Option<f32>> {
        read_episode_return(self.session.collector().episode_return().py()?)
    }

    /// One collection and one update: the loop body.
    #[pyo3(signature = (epochs = 4, minibatches = 1))]
    fn round(&mut self, py: Python<'_>, epochs: usize, minibatches: usize) -> PyResult<Stats> {
        self.collect(py)?;
        let stats = self.update_reading(epochs, minibatches, true)?;
        self.rounds += 1;
        Ok(stats)
    }

    /// Run `rounds` rounds, reporting each one.
    ///
    /// `callback` is called with the round's [`Stats`] as they are produced;
    /// returning `False` from it stops the run early.
    #[pyo3(signature = (rounds, epochs = 4, minibatches = 1, callback = None))]
    fn run(
        &mut self,
        py: Python<'_>,
        rounds: usize,
        epochs: usize,
        minibatches: usize,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<Vec<Py<Stats>>> {
        let mut history = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let stats = self.round(py, epochs, minibatches)?;
            let (stats, keep) = keep_going(py, callback.as_ref(), stats)?;
            history.push(stats);
            if !keep {
                break;
            }
        }
        Ok(history)
    }

    /// Forget everything: zero the recurrent state and start the environment over
    /// on the next collection. The weights are untouched.
    ///
    /// Also forgets the reference policy's own accumulated cache, if there is
    /// one, so a fresh collection after this scores the reference from a zeroed
    /// history too — matching the actor's.
    fn reset(&mut self) {
        self.session.collector_mut().reset();
        self.batch = None;
        self.pending = false;
        if let Some(reference) = self.reference.as_mut() {
            reference.reset();
        }
    }

    /// Save weights, optimizer state, the round/step counters and the training
    /// configuration, so training resumes with its optimizer and its settings
    /// rather than merely warm-starting weights.
    ///
    /// Distinct from [`PyPolicy::save`][crate::policy::PyPolicy::save]:
    /// that one is weights-only, by design, and loading it back is always a
    /// warm start regardless of which method reads it. Round-trip *this*
    /// through [`PyPpoLearner::load_checkpoint`] or
    /// [`PyPpoLearner::from_checkpoint`], not `Policy.load`.
    ///
    /// The configuration saved is the base learning rate, `lr_schedule`, the
    /// AdamW hyperparameters, `max_grad_norm`, the `PpoConfig`, the policy
    /// architecture, the `EmaConfig` and — when there is a reference — a
    /// fingerprint of its weights. The reference's weights and architecture, and
    /// the moving average's weights and counter, are saved too, at either level,
    /// so a restore can rebuild them.
    ///
    /// `level="optimizer"` (the default) stops there: a learner restored from it
    /// continues training exactly, but not the run — its next window starts from
    /// wherever its own environment and recurrent state are.
    ///
    /// `level="full"` adds everything the next round depends on: the collector's
    /// observation, termination flags and episode accounting, every layer's
    /// recurrent state, the action-draw schedule, the reference policy's carried
    /// cache, and the environment's own state through its optional
    /// `save_state()`. A learner restored from it in any process takes exactly the
    /// rounds this one would have taken. It needs a `.m3ck` path, an environment
    /// implementing `save_state()`/`load_state()` (`NotImplementedError`
    /// otherwise), and a learner between rounds: saving after `collect()` and
    /// before `update()` raises `ValueError`.
    #[pyo3(signature = (path, level = "optimizer"))]
    fn save(&self, py: Python<'_>, path: &str, level: &str) -> PyResult<()> {
        let level = Level::parse_save(level)?;
        let rollout = if level == Level::Full {
            if self.pending {
                return Err(pending_window_error("a full checkpoint"));
            }
            let (session, reference) = (&self.session, &self.reference);
            Some(self.env.with(py, |env| {
                RolloutSnapshot::capture(session.collector(), reference.as_ref(), env)
            })?)
        } else {
            None
        };
        let policy = self.session.policy();
        resume::save(
            path,
            &policy,
            &self.trainer,
            self.rounds,
            &self.live_config(),
            self.reference.as_ref(),
            &self.continuation,
            rollout,
        )
    }

    /// Restore weights, optimizer state and counters saved by
    /// [`PyPpoLearner::save`], onto this already-constructed learner.
    ///
    /// All or nothing: every part of the checkpoint is validated before any of
    /// the learner changes, so a load that raises leaves the weights, the
    /// optimizer's moments, `rounds`, the optimizer step and the next learning
    /// rate exactly as they were.
    ///
    /// `strict` requires the weights to match the policy exactly and the
    /// checkpoint to carry optimizer state and counters. `strict=False` with a
    /// weights-only checkpoint (`Policy.save`) is an explicit warm start: the
    /// optimizer and every counter restart from zero.
    ///
    /// `config` decides what happens when the saved training configuration
    /// differs from this learner's: `"verify"` (the default) raises, listing
    /// every difference; `"checkpoint"` adopts the saved optimizer, schedule and
    /// PPO settings, and the saved reference policy when the checkpoint carries
    /// its weights (never the architecture), and the saved moving average;
    /// `"live"` keeps this learner's and records the run as a non-exact
    /// continuation — see `continuation`. A checkpoint written before
    /// configurations were saved loads only with `"live"`.
    ///
    /// The moving average follows the same rules: it is restored with its
    /// counter when both sides keep one under the same configuration. Under
    /// `"live"`, a learner whose checkpoint has no average re-seeds its own from
    /// the loaded weights (a warm continuation, noted), and a checkpoint's
    /// average this learner does not keep is ignored, with a note.
    ///
    /// A full checkpoint (`save(level="full")`) also restores the rollout and
    /// the environment — through its `load_state()`, called after everything
    /// else has been validated and before anything is changed, so an
    /// environment that refuses its bytes leaves the learner untouched. Its
    /// sampling seed and temperature are settled by `config` like the rest of
    /// the configuration. `level="optimizer"` ignores the rollout state;
    /// `level="full"` requires it.
    ///
    /// Clears the window last collected, whose behaviour log-probabilities
    /// belong to the weights being replaced. Returns what was restored:
    /// `{"weights", "optimizer", "counters", "config", "level", "notes"}`,
    /// where `level` is what this load restored: `"full"`, `"optimizer"` or
    /// `"warm"`.
    #[pyo3(signature = (path, strict = true, config = "verify", level = None))]
    fn load_checkpoint<'py>(
        &mut self,
        py: Python<'py>,
        path: &str,
        strict: bool,
        config: &str,
        level: Option<&str>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let mode = ConfigMode::parse(config)?;
        let level = Level::parse_load(level)?;
        let checkpoint = Checkpoint::load(path).map_err(resume::load_error)?;
        self.load_from(py, &checkpoint, strict, mode, level)
    }

    /// Build a learner from a checkpoint written by [`PyPpoLearner::save`],
    /// with the architecture, optimizer, schedule and PPO settings it recorded,
    /// then load it with `config="verify"`.
    ///
    /// What a checkpoint cannot carry is still passed in: the environment and
    /// the window length. `reference` defaults to the reference policy the
    /// checkpoint carries; one passed in must have the same weights.
    /// `temperature` and `seed` default to the ones a full checkpoint recorded,
    /// and otherwise to `1.0` and `0`. A moving average is rebuilt with the
    /// configuration, weights and counter the checkpoint recorded.
    #[staticmethod]
    #[pyo3(signature = (path, env, steps = 128, *, temperature = None, seed = None, reference = None, strict = true))]
    #[allow(clippy::too_many_arguments)]
    fn from_checkpoint(
        py: Python<'_>,
        path: &str,
        env: &Bound<'_, PyAny>,
        steps: usize,
        temperature: Option<f32>,
        seed: Option<u64>,
        reference: Option<&PyPolicy>,
        strict: bool,
    ) -> PyResult<Self> {
        let checkpoint = Checkpoint::load(path).map_err(resume::load_error)?;
        let (temperature, seed) = saved_sampling(&checkpoint, temperature, seed);
        let (saved, optim) = resume::saved_trainer_config(&checkpoint, "ppo")?;
        let ppo: PpoConfig = serde_json::from_value(saved["algorithm"].clone()).map_err(|e| {
            PyValueError::new_err(format!("the checkpoint's PPO settings are unusable: {e}"))
        })?;
        let policy = PyPolicy::new(&crate::config::PyPolicyConfig::from_json(
            &checkpoint.metadata["policy"],
        )?)?;
        let saved_reference = match reference {
            Some(_) => None,
            None => resume::saved_reference(&checkpoint, &policy.device)?.map(|saved| {
                PyPolicy::from_shared(std::rc::Rc::new(saved.into_policy()), policy.device.clone())
            }),
        };
        let reference = reference.or(saved_reference.as_ref());
        let mut learner = Self::new(
            &policy,
            env,
            steps,
            Some(PyPpoConfig { inner: ppo }),
            optim.learning_rate,
            Some(PyLrSchedule {
                inner: optim.schedule,
            }),
            optim.max_grad_norm,
            optim.weight_decay,
            optim.betas,
            optim.eps,
            temperature,
            seed,
            reference,
            None,
            saved_ema(&saved)?,
        )?;
        learner.load_from(py, &checkpoint, strict, ConfigMode::Verify, None)?;
        Ok(learner)
    }

    /// How exactly this learner's history continues one run:
    /// `{"level": "full" | "optimizer" | "warm", "notes": [str]}`. A fresh
    /// learner is `"full"`; a load that restored training but not the rollout
    /// lowers it to `"optimizer"`; a warm start, a legacy checkpoint or a load
    /// that kept different live settings to `"warm"`. It never rises again, and
    /// a later `save` records it.
    #[getter]
    fn continuation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.continuation.to_dict(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "PpoLearner(num_envs={}, steps={}, rounds={}, collection={}, buffer={} KiB)",
            self.env.envs(),
            self.steps,
            self.rounds,
            self.collection_path(),
            self.session.collector().buffer().bytes() / 1024,
        )
    }
}

impl PyPpoLearner {
    /// [`PyPpoLearner::update`], also reading the episode return when
    /// `with_return` — under the same synchronisation, which is what lets a
    /// round cost one read rather than two.
    fn update_reading(
        &mut self,
        epochs: usize,
        minibatches: usize,
        with_return: bool,
    ) -> PyResult<Stats> {
        let batch = self.batch.clone().ok_or_else(|| {
            PyValueError::new_err("nothing has been collected yet; call collect() first")
        })?;
        if epochs == 0 {
            return Err(PyValueError::new_err("an update needs at least one epoch"));
        }
        let envs = batch.envs();
        if minibatches == 0 || envs % minibatches != 0 {
            return Err(PyValueError::new_err(format!(
                "minibatches must divide the {envs} environments, got {minibatches}; \
                 a recurrent policy can only be split along whole sequences"
            )));
        }
        let per = envs / minibatches;
        let policy = self.session.policy();
        let task = PpoTask::new(&policy, self.config);

        // Built once, not once per step: the minibatches of a window are the same
        // every epoch.
        let micros = if minibatches == 1 {
            vec![batch.clone()]
        } else {
            (0..minibatches)
                .map(|index| batch.minibatch(index * per, per))
                .collect::<mamba3::error::Result<Vec<_>>>()
                .py()?
        };
        // Every step is queued before anything is read. A read is a fixed wait for
        // the device, and reading two scalars per step used to be most of an
        // update's reads — two per optimizer step, then six for the diagnostics.
        let mut queued = Vec::with_capacity(epochs * minibatches);
        for _ in 0..epochs {
            for micro in &micros {
                queued.push(
                    self.trainer
                        .queue_step(&task, std::slice::from_ref(micro))
                        .py()?,
                );
            }
        }

        let diagnostics = task
            .stat_tensors()
            .expect("an update queues at least one step");
        let episode_return = with_return
            .then(|| self.session.collector().episode_return())
            .transpose()
            .py()?;
        let mut scalars: Vec<&Tensor<R, E>> = queued.iter().flat_map(|q| q.scalars()).collect();
        let step_scalars = scalars.len();
        scalars.extend(&diagnostics);
        if let Some((mean, count)) = &episode_return {
            scalars.extend([mean, count]);
        }
        let (_, values) = read_all(&[], &scalars).py()?;
        let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
        let (step_values, rest) = values.split_at(step_scalars);
        let (diagnostic_values, return_values) = rest.split_at(diagnostics.len());

        let infos = self.trainer.report_steps(&queued, step_values);
        let taken = infos.len() as f32;
        let last = infos.last().expect("an update queues at least one step");
        let diagnostics = PpoStats::from_values(
            diagnostic_values
                .try_into()
                .expect("six diagnostics were read"),
        );
        self.pending = false;
        Ok(Stats {
            round: self.rounds,
            steps: batch.steps(),
            optimizer_steps: last.step,
            loss: infos.iter().map(|i| i.loss).sum::<f32>() / taken,
            policy_loss: diagnostics.policy_loss,
            value_loss: diagnostics.value_loss,
            entropy: diagnostics.entropy,
            approx_kl: diagnostics.approx_kl,
            clip_fraction: diagnostics.clip_fraction,
            reference_kl: diagnostics.reference_kl,
            grad_norm: infos.iter().map(|i| i.grad_norm).sum::<f32>() / taken,
            learning_rate: last.learning_rate,
            episode_return: match return_values {
                [mean, count] => episode_mean(*mean, *count),
                _ => None,
            },
        })
    }

    fn live_config(&self) -> LiveConfig {
        let reference = match &self.reference {
            Some(reference) => {
                let fingerprint = self
                    .reference_fingerprint
                    .get_or_init(|| reference.fingerprint());
                serde_json::json!({ "weights_fnv1a64": fingerprint })
            }
            None => serde_json::Value::Null,
        };
        LiveConfig {
            kind: "ppo",
            optim: self.optim,
            algorithm: serde_json::to_value(self.config).expect("PpoConfig serialises"),
            policy: crate::config::PyPolicyConfig::from_inner(
                self.session.policy().config().clone(),
            )
            .as_json(),
            reference,
            ema: self.trainer.ema().map(|ema| *ema.config()),
        }
    }

    fn load_from<'py>(
        &mut self,
        py: Python<'py>,
        checkpoint: &Checkpoint,
        strict: bool,
        mode: ConfigMode,
        level: Option<Level>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        // `config="checkpoint"` brings the checkpoint's reference with it.
        let has_reference = self.reference.is_some()
            || (mode == ConfigMode::Checkpoint && checkpoint.reference.is_some());
        let adopt = |value: &serde_json::Value| -> PyResult<PpoConfig> {
            let config: PpoConfig = serde_json::from_value(value.clone()).map_err(|e| {
                PyValueError::new_err(format!("the checkpoint's PPO settings are unusable: {e}"))
            })?;
            config.validate().py()?;
            if config.reference_coeff != 0.0 && !has_reference {
                return Err(PyValueError::new_err(
                    "the checkpoint's reference_coeff is nonzero but this learner has no \
                     reference policy to price it against",
                ));
            }
            Ok(config)
        };
        let policy = self.session.policy();
        let mut loaded = resume::load(
            checkpoint,
            &policy,
            &self.live_config(),
            strict,
            mode,
            &|v| adopt(v).map(|_| ()),
        )?;
        let adopted = loaded.adopted_algorithm.as_ref().map(adopt).transpose()?;
        let staged_ema = resume::stage_ema(
            checkpoint,
            &mut loaded,
            self.trainer.ema(),
            &policy,
            &self.device,
            strict,
        )?;
        let adopted_reference = if loaded.adopt_reference {
            resume::saved_reference(checkpoint, &self.device)?
        } else {
            None
        };
        let staged = stage_rollout(
            checkpoint,
            level,
            &mut loaded,
            self.session.collector(),
            adopted_reference.as_ref().or(self.reference.as_ref()),
        )?;
        if let Some(staged) = &staged {
            self.env.with_mapped(
                py,
                |env| env.load_state(staged.env_bytes()),
                resume::load_error,
            )?;
        }
        // Everything below is infallible or already validated: the learner
        // changes all at once or not at all.
        let rollout = staged.is_some();
        let summary = loaded.summary(py, rollout)?;
        self.continuation = loaded.continuation(rollout);
        let (trainer, optim, rounds) = loaded.commit_weights();
        let trainer = commit_ema(trainer, &mut self.trainer, &mut self.ema_policy, staged_ema);
        if let Some(config) = adopted {
            self.config = config;
        }
        if let Some(reference) = adopted_reference {
            self.reference = Some(reference);
            self.reference_fingerprint = std::cell::OnceCell::new();
        }
        self.trainer = trainer;
        self.optim = optim;
        self.rounds = rounds;
        self.batch = None;
        self.pending = false;
        if let Some(staged) = staged {
            staged.apply_without_env(self.session.collector_mut(), self.reference.as_mut());
        }
        Ok(summary)
    }
}

/// The refusal a full save or an EMA reset gets between `collect()` and
/// `update()`; `what` is which of the two.
fn pending_window_error(what: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{what} is taken between rounds, and a window has been collected but not yet \
         trained on; call update() first, or reset() to discard the window"
    ))
}

/// `reset_ema()` for either learner.
fn reset_trainer_ema(trainer: &mut Trainer<R, E, AdamW<R, E>>) -> PyResult<()> {
    trainer
        .ema_mut()
        .ok_or_else(|| {
            PyValueError::new_err(
                "this learner has no EMA to reset; construct it with ema=EmaConfig(...)",
            )
        })?
        .reset()
        .py()
}

/// Move the learner's moving average onto the trainer a load built, applying
/// what the load staged for it. Infallible.
fn commit_ema(
    mut restored: Trainer<R, E, AdamW<R, E>>,
    current: &mut Trainer<R, E, AdamW<R, E>>,
    ema_policy: &mut Option<resume::Shadow>,
    staged: Option<resume::StagedLearnerEma>,
) -> Trainer<R, E, AdamW<R, E>> {
    if let Some(staged) = staged {
        let live = current.take_ema().zip(ema_policy.clone());
        let (ema, shadow) = staged.commit(live);
        restored = restored.with_ema(ema);
        *ema_policy = Some(shadow);
    }
    restored
}

/// The sampling settings a learner built from `checkpoint` uses: the caller's,
/// else the ones a full checkpoint recorded, else the constructors' defaults.
fn saved_sampling(
    checkpoint: &Checkpoint,
    temperature: Option<f32>,
    seed: Option<u64>,
) -> (f32, u64) {
    let rollout = checkpoint.metadata.get("rollout");
    let temperature = temperature
        .or_else(|| {
            rollout
                .and_then(|r| r.get("temperature_bits"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|bits| u32::try_from(bits).ok())
                .map(f32::from_bits)
        })
        .unwrap_or(1.0);
    let seed = seed
        .or_else(|| {
            rollout
                .and_then(|r| r.get("seed"))
                .and_then(serde_json::Value::as_u64)
        })
        .unwrap_or(0);
    (temperature, seed)
}

/// Validate and upload a checkpoint's rollout state for a learner, without
/// changing the learner, or `None` when there is none to restore.
///
/// The seed and temperature it was sampled with are settled the way `mode`
/// settled the training configuration: `"verify"` refuses a difference,
/// `"checkpoint"` adopts the saved values, `"live"` keeps the learner's and marks
/// the load as a warm one.
fn stage_rollout(
    checkpoint: &Checkpoint,
    level: Option<Level>,
    loaded: &mut resume::Loaded,
    collector: &Collector<'static, R, E>,
    reference: Option<&ReferencePolicy<R, E>>,
) -> PyResult<Option<StagedRollout<R, E>>> {
    if level == Some(Level::Optimizer) {
        return Ok(None);
    }
    let Some(mut snapshot) =
        RolloutSnapshot::from_checkpoint(checkpoint).map_err(resume::load_error)?
    else {
        if level == Some(Level::Full) {
            return Err(PyValueError::new_err(
                "this checkpoint carries no rollout state to restore; it was saved at the \
                 'optimizer' level (learner.save(path, level='full') writes one that does)",
            ));
        }
        return Ok(None);
    };
    if loaded.mode == ConfigMode::Live {
        // The configuration comparison has already settled the reference: a
        // different live one was kept and the load recorded as warm.
        snapshot.reference_weights = None;
    }
    let mut staged = snapshot
        .stage(collector, reference)
        .map_err(resume::load_error)?;
    let saved = &snapshot.collector;
    let mut differences = Vec::new();
    if saved.seed != collector.seed() {
        differences.push(format!(
            "seed: checkpoint {}, live {}",
            saved.seed,
            collector.seed()
        ));
    }
    if saved.temperature.to_bits() != collector.temperature().to_bits() {
        differences.push(format!(
            "temperature: checkpoint {}, live {}",
            saved.temperature,
            collector.temperature()
        ));
    }
    if !differences.is_empty() {
        match loaded.mode {
            ConfigMode::Verify => {
                return Err(PyValueError::new_err(format!(
                    "the checkpoint's rollout was sampled with different settings; pass \
                     config='checkpoint' to adopt them, or config='live' to keep these as a \
                     non-exact continuation. Differences:\n  {}",
                    differences.join("\n  ")
                )));
            }
            ConfigMode::Checkpoint => {
                loaded
                    .notes
                    .extend(differences.iter().map(|d| format!("adopted {d}")));
            }
            ConfigMode::Live => {
                staged.keep_sampling(collector.seed(), collector.temperature());
                for d in differences {
                    loaded.downgrade(format!("kept live {d}"));
                }
            }
        }
    }
    Ok(Some(staged))
}

/// Imitation learning: behaviour cloning, and DAgger.
///
/// Each round rolls out a mixture of the environment's expert and the policy —
/// `beta` is the expert's share — and fits the policy to what the expert would
/// have done at every state the mixture reached. As `beta` decays the states drift
/// from the expert's own distribution to the learner's, which is what stops a
/// cloned policy from being good only where the expert already went.
///
/// The critic is deliberately left untrained: nothing in cross entropy mentions
/// reward, and a critic fitted to an expert's states would be confidently wrong
/// about the states the policy actually visits. A PPO run that follows fits it
/// against real returns.
#[pyclass(module = "mamba3_rl", name = "ImitationLearner", unsendable)]
pub struct PyImitationLearner {
    session: Session,
    env: EnvHandle,
    trainer: Trainer<R, E, AdamW<R, E>>,
    schedule: DaggerSchedule,
    entropy_bonus: f32,
    optim: OptimSettings,
    /// The policy the trainer's moving average lives in, when `ema=` was given.
    ema_policy: Option<resume::Shadow>,
    continuation: Continuation,
    steps: usize,
    rounds: u64,
    device: Device<R>,
}

#[pymethods]
impl PyImitationLearner {
    #[new]
    #[pyo3(signature = (
        policy,
        env,
        steps = 128,
        *,
        schedule = None,
        entropy_bonus = 0.01,
        learning_rate = 3e-3,
        lr_schedule = None,
        max_grad_norm = 1.0,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
        temperature = 1.0,
        seed = 0,
        ema = None,
    ))]
    fn new(
        policy: &PyPolicy,
        env: &Bound<'_, PyAny>,
        steps: usize,
        schedule: Option<PyDaggerSchedule>,
        entropy_bonus: f32,
        learning_rate: f32,
        lr_schedule: Option<PyLrSchedule>,
        max_grad_norm: f32,
        weight_decay: f32,
        betas: (f32, f32),
        eps: f32,
        temperature: f32,
        seed: u64,
        ema: Option<PyEmaConfig>,
    ) -> PyResult<Self> {
        let (handle, session) = setup_session(
            policy,
            env,
            steps,
            temperature,
            seed,
            true,
            true,
            "a window needs at least one step",
        )?;
        let optim = OptimSettings {
            learning_rate,
            schedule: lr_schedule.unwrap_or_default().inner,
            max_grad_norm,
            weight_decay,
            betas,
            eps,
        };
        let (trainer, ema_policy) = attach_ema(optim.trainer()?, policy, ema.map(|e| e.inner))?;
        Ok(Self {
            session,
            env: handle,
            trainer,
            schedule: schedule.unwrap_or_default().inner,
            entropy_bonus,
            optim,
            ema_policy,
            continuation: Continuation::fresh(),
            steps,
            rounds: 0,
            device: policy.device.clone(),
        })
    }

    /// The policy being trained. The same weights, not a copy.
    #[getter]
    fn policy(&self) -> PyPolicy {
        PyPolicy::from_shared(self.session.policy(), self.device.clone())
    }

    /// The environment this learner drives.
    #[getter]
    fn env(&self, py: Python<'_>) -> Py<PyAny> {
        self.env.object(py)
    }

    /// The schedule the expert's share follows.
    #[getter]
    fn schedule(&self) -> PyDaggerSchedule {
        PyDaggerSchedule {
            inner: self.schedule,
        }
    }

    /// Rounds completed.
    #[getter]
    fn rounds(&self) -> u64 {
        self.rounds
    }

    /// The moving average of the weights, as a policy, or `None` without
    /// `ema=`. A handle onto the average itself, not a copy: it keeps moving as
    /// the learner trains, and `save` (or its `fingerprint`) is how to keep a
    /// moment of it. Usable anywhere a policy is — `evaluate`, `Rollout`,
    /// `save` — except as the policy another learner trains.
    #[getter]
    fn ema_policy(&self) -> Option<PyPolicy> {
        self.ema_policy
            .as_ref()
            .map(|shadow| PyPolicy::from_shared(Rc::clone(shadow), self.device.clone()))
    }

    /// Optimizer steps the average has taken since it was built, reset or
    /// restored with its counter; `0` without one.
    #[getter]
    fn ema_updates(&self) -> u64 {
        self.trainer.ema().map_or(0, Ema::updates)
    }

    /// The average's configuration, or `None` without one.
    #[getter]
    fn ema_config(&self) -> Option<PyEmaConfig> {
        self.trainer.ema().map(|ema| PyEmaConfig {
            inner: *ema.config(),
        })
    }

    /// Restart the moving average from the current weights and its counter from
    /// zero. See [`PyPpoLearner::reset_ema`]; a round is one call here, so
    /// there is never a pending window to refuse.
    fn reset_ema(&mut self) -> PyResult<()> {
        reset_trainer_ema(&mut self.trainer)
    }

    /// One DAgger round: roll out the mixture, label it, take one gradient step.
    ///
    /// `beta` overrides the schedule for this round. `agreement` costs one extra
    /// replay, and is the number that says whether the cloning is working. It is
    /// read together with the step's loss, so a round synchronises once either
    /// way.
    #[pyo3(signature = (beta = None, agreement = true))]
    fn round(
        &mut self,
        py: Python<'_>,
        beta: Option<f32>,
        agreement: bool,
    ) -> PyResult<CloneStats> {
        let beta = beta.unwrap_or_else(|| self.schedule.beta(self.rounds as u32));
        if !(0.0..=1.0).contains(&beta) {
            return Err(PyValueError::new_err(format!(
                "beta is the expert's share of the acting and lies in [0, 1], got {beta}"
            )));
        }
        let Self { session, env, .. } = self;
        let report = env.with(py, |mut vec_env| {
            session
                .collector_mut()
                .collect_with_expert(&mut vec_env, beta)
        })?;
        let batch = self.session.collector().imitation_batch().py()?;

        let policy = self.session.policy();
        let task = BehaviourCloningTask::new(&policy).with_entropy_bonus(self.entropy_bonus);
        // The replay is queued behind the step, not after reading it: it scores
        // the updated weights either way, but the host submits it while the device
        // is still running the backward pass, and the round waits for the device
        // once instead of once per read.
        let queued = self
            .trainer
            .queue_step(&task, std::slice::from_ref(&batch))
            .py()?;
        let agreement = agreement
            .then(|| task.queue_agreement(&batch))
            .transpose()
            .py()?;
        let scalars: Vec<&Tensor<R, E>> = queued
            .scalars()
            .chain(agreement.iter().flat_map(QueuedAgreement::scalars))
            .collect();
        let (_, values) = read_all(&[], &scalars).py()?;
        let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
        let (step_values, agreement_values) = values.split_at(queued.scalars().count());
        let info = self
            .trainer
            .report_steps(std::slice::from_ref(&queued), step_values)[0];
        let agreement = agreement.map(|queued| queued.fraction(agreement_values));

        self.rounds += 1;
        Ok(CloneStats {
            round: self.rounds - 1,
            steps: report.steps,
            optimizer_steps: info.step,
            beta,
            loss: info.loss,
            agreement,
            grad_norm: info.grad_norm,
            learning_rate: info.learning_rate,
        })
    }

    /// Run `rounds` rounds, following the schedule.
    ///
    /// `callback` is called with each round's [`CloneStats`]; returning `False`
    /// from it stops the run early.
    #[pyo3(signature = (rounds, agreement = true, callback = None))]
    fn run(
        &mut self,
        py: Python<'_>,
        rounds: usize,
        agreement: bool,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<Vec<Py<CloneStats>>> {
        let mut history = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let stats = self.round(py, None, agreement)?;
            let (stats, keep) = keep_going(py, callback.as_ref(), stats)?;
            history.push(stats);
            if !keep {
                break;
            }
        }
        Ok(history)
    }

    /// Forget everything: zero the recurrent state and start the environment over.
    fn reset(&mut self) {
        self.session.collector_mut().reset();
    }

    /// Save weights, optimizer state, counters and the training configuration.
    ///
    /// See [`PyPpoLearner::save`], including `level`. The algorithm settings
    /// saved are the DAgger `schedule` and `entropy_bonus`; DAgger's position
    /// in its schedule is `rounds`, and its expert mixing draws from the same
    /// schedule as the policy's actions, so a full checkpoint carries both.
    #[pyo3(signature = (path, level = "optimizer"))]
    fn save(&self, py: Python<'_>, path: &str, level: &str) -> PyResult<()> {
        let level = Level::parse_save(level)?;
        let rollout = if level == Level::Full {
            let session = &self.session;
            Some(self.env.with(py, |env| {
                RolloutSnapshot::capture(session.collector(), None, env)
            })?)
        } else {
            None
        };
        let policy = self.session.policy();
        resume::save(
            path,
            &policy,
            &self.trainer,
            self.rounds,
            &self.live_config(),
            None,
            &self.continuation,
            rollout,
        )
    }

    /// Restore a checkpoint written by [`PyImitationLearner::save`]. All or
    /// nothing, with the same `strict` and `config` semantics as
    /// [`PyPpoLearner::load_checkpoint`]; `config="checkpoint"` adopts the
    /// DAgger schedule and entropy bonus along with the optimizer settings.
    #[pyo3(signature = (path, strict = true, config = "verify", level = None))]
    fn load_checkpoint<'py>(
        &mut self,
        py: Python<'py>,
        path: &str,
        strict: bool,
        config: &str,
        level: Option<&str>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let mode = ConfigMode::parse(config)?;
        let level = Level::parse_load(level)?;
        let checkpoint = Checkpoint::load(path).map_err(resume::load_error)?;
        self.load_from(py, &checkpoint, strict, mode, level)
    }

    /// Build a learner from a checkpoint written by
    /// [`PyImitationLearner::save`]; see [`PyPpoLearner::from_checkpoint`].
    #[staticmethod]
    #[pyo3(signature = (path, env, steps = 128, *, temperature = None, seed = None, strict = true))]
    fn from_checkpoint(
        py: Python<'_>,
        path: &str,
        env: &Bound<'_, PyAny>,
        steps: usize,
        temperature: Option<f32>,
        seed: Option<u64>,
        strict: bool,
    ) -> PyResult<Self> {
        let checkpoint = Checkpoint::load(path).map_err(resume::load_error)?;
        let (temperature, seed) = saved_sampling(&checkpoint, temperature, seed);
        let (saved, optim) = resume::saved_trainer_config(&checkpoint, "imitation")?;
        let (schedule, entropy_bonus) = imitation_algorithm(&saved["algorithm"])?;
        let policy = PyPolicy::new(&crate::config::PyPolicyConfig::from_json(
            &checkpoint.metadata["policy"],
        )?)?;
        let mut learner = Self::new(
            &policy,
            env,
            steps,
            Some(PyDaggerSchedule { inner: schedule }),
            entropy_bonus,
            optim.learning_rate,
            Some(PyLrSchedule {
                inner: optim.schedule,
            }),
            optim.max_grad_norm,
            optim.weight_decay,
            optim.betas,
            optim.eps,
            temperature,
            seed,
            saved_ema(&saved)?,
        )?;
        learner.load_from(py, &checkpoint, strict, ConfigMode::Verify, None)?;
        Ok(learner)
    }

    /// Whether this learner's history is one uninterrupted run under one
    /// configuration. See [`PyPpoLearner::continuation`].
    #[getter]
    fn continuation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.continuation.to_dict(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ImitationLearner(num_envs={}, steps={}, rounds={})",
            self.env.envs(),
            self.steps,
            self.rounds,
        )
    }
}

/// An imitation learner's algorithm settings as saved under
/// `trainer_config.algorithm`.
fn imitation_algorithm(value: &serde_json::Value) -> PyResult<(DaggerSchedule, f32)> {
    let schedule: DaggerSchedule =
        serde_json::from_value(value.get("dagger_schedule").cloned().unwrap_or_default()).map_err(
            |e| PyValueError::new_err(format!("the checkpoint's DAgger schedule is unusable: {e}")),
        )?;
    let entropy_bonus = value
        .get("entropy_bonus")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| PyValueError::new_err("the checkpoint's entropy_bonus is not a number"))?
        as f32;
    Ok((schedule, entropy_bonus))
}

impl PyImitationLearner {
    fn live_config(&self) -> LiveConfig {
        LiveConfig {
            kind: "imitation",
            optim: self.optim,
            algorithm: serde_json::json!({
                "dagger_schedule": self.schedule,
                "entropy_bonus": self.entropy_bonus,
            }),
            policy: crate::config::PyPolicyConfig::from_inner(
                self.session.policy().config().clone(),
            )
            .as_json(),
            reference: serde_json::Value::Null,
            ema: self.trainer.ema().map(|ema| *ema.config()),
        }
    }

    fn load_from<'py>(
        &mut self,
        py: Python<'py>,
        checkpoint: &Checkpoint,
        strict: bool,
        mode: ConfigMode,
        level: Option<Level>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let policy = self.session.policy();
        let mut loaded = resume::load(
            checkpoint,
            &policy,
            &self.live_config(),
            strict,
            mode,
            &|v| imitation_algorithm(v).map(|_| ()),
        )?;
        let adopted = loaded
            .adopted_algorithm
            .as_ref()
            .map(imitation_algorithm)
            .transpose()?;
        let staged_ema = resume::stage_ema(
            checkpoint,
            &mut loaded,
            self.trainer.ema(),
            &policy,
            &self.device,
            strict,
        )?;
        let staged = stage_rollout(
            checkpoint,
            level,
            &mut loaded,
            self.session.collector(),
            None,
        )?;
        if let Some(staged) = &staged {
            self.env.with_mapped(
                py,
                |env| env.load_state(staged.env_bytes()),
                resume::load_error,
            )?;
        }
        let rollout = staged.is_some();
        let summary = loaded.summary(py, rollout)?;
        self.continuation = loaded.continuation(rollout);
        let (trainer, optim, rounds) = loaded.commit_weights();
        let trainer = commit_ema(trainer, &mut self.trainer, &mut self.ema_policy, staged_ema);
        if let Some(algorithm) = adopted {
            (self.schedule, self.entropy_bonus) = algorithm;
        }
        self.trainer = trainer;
        self.optim = optim;
        self.rounds = rounds;
        if let Some(staged) = staged {
            staged.apply_without_env(self.session.collector_mut(), None);
        }
        Ok(summary)
    }
}

/// Read `episode_return()`'s `(mean, count)` in one host read — no launch to
/// pack them first — and see [`episode_mean`].
fn read_episode_return(parts: (Tensor<R, E>, Tensor<R, E>)) -> PyResult<Option<f32>> {
    let (mean, count) = parts;
    let ([], [mean, count]) = read_together([], [&mean, &count]).py()?;
    Ok(episode_mean(mean[0], count[0]))
}

/// A zero count as `None`, so a window that ended mid-episode is never mistaken
/// for one reporting a genuine mean.
fn episode_mean(mean: f32, count: f32) -> Option<f32> {
    (count > 0.0).then_some(mean)
}

/// The return a policy earns on `env`, without training on it, or `None` if no
/// episode completed within `steps` — a short evaluation window has nothing to
/// average and must not be confused with one reporting a genuine mean.
///
/// Collects one window of `steps` steps from a *fresh* recurrent state and reports
/// the mean reward per completed episode. The default `temperature=0` acts
/// greedily, which is what the policy believes rather than what its exploration
/// noise happens to produce — and why this takes an environment of its own rather
/// than borrowing a learner's: a greedy rollout would consume observations the
/// training loop has to start its next window from.
#[pyfunction]
#[pyo3(signature = (policy, env, steps = 64, *, temperature = 0.0, seed = 0))]
pub fn evaluate(
    py: Python<'_>,
    policy: &PyPolicy,
    env: &Bound<'_, PyAny>,
    steps: usize,
    temperature: f32,
    seed: u64,
) -> PyResult<Option<f32>> {
    let (handle, mut session) = setup_session(
        policy,
        env,
        steps,
        temperature,
        seed,
        false,
        false,
        "an evaluation needs at least one step",
    )?;
    handle.with(py, |mut vec_env| {
        session.collector_mut().collect(&mut vec_env).map(|_| ())
    })?;
    read_episode_return(session.collector().episode_return().py()?)
}
