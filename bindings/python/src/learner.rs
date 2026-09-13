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
//! that the host never waits on — with two exceptions, both deliberate: reading
//! the diagnostics at the end of an update, and reading the episode return at the
//! end of a round. Both are numbers a human asked for. An environment written in
//! Python adds its own two copies per step, which is the price of that choice and
//! is visible in `mamba3::backend::read_count()`.

use mamba3::backend::Device;
use mamba3::rl::{BehaviourCloningTask, DaggerSchedule, PpoBatch, PpoConfig, PpoTask};
use mamba3::train::{AdamW, AdamWConfig, Checkpoint, StepInfo, Trainer, TrainerConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{PyClass, PyClassInitializer};

use crate::config::{PyLrSchedule, PyPpoConfig};
use crate::env::{EnvHandle, check_against_policy, refuse_without_expert};
use crate::err::IntoPyResult;
use crate::policy::PyPolicy;
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

/// The optimizer half of a learner, which both loops configure the same way.
///
/// `schedule` advances on optimizer updates, i.e. `Trainer::step_count()` — PPO
/// epochs and minibatches each take one, so a window collected once and trained
/// over `epochs * minibatches` times advances the schedule that many steps, not
/// one. `None` means [`LrSchedule::Constant`], today's unscheduled behaviour.
fn trainer(
    learning_rate: f32,
    max_grad_norm: f32,
    weight_decay: f32,
    betas: (f32, f32),
    schedule: Option<PyLrSchedule>,
) -> PyResult<Trainer<R, E, AdamW<R, E>>> {
    let config = TrainerConfig::builder()
        .learning_rate(learning_rate)
        .max_grad_norm(max_grad_norm)
        .schedule(schedule.unwrap_or_default().inner)
        .build()
        .py()?;
    let optimizer = AdamWConfig::builder()
        .learning_rate(learning_rate)
        .betas(betas.0, betas.1)
        .weight_decay(weight_decay)
        .build()
        .init::<R, E>();
    Ok(Trainer::new(config, optimizer))
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
    /// A frozen, independently snapshotted policy the run is priced against, for
    /// `PpoConfig.reference_coeff`. Scored once per window rather than once per
    /// epoch — it does not move — and keeps its own recurrent cache across
    /// windows, separate from the behaviour policy's.
    reference: Option<mamba3::rl::ReferencePolicy<R, E>>,
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
        temperature = 1.0,
        seed = 0,
        reference = None,
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
        temperature: f32,
        seed: u64,
        reference: Option<&PyPolicy>,
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
        let reference = reference
            .map(|p| mamba3::rl::ReferencePolicy::snapshot(&p.inner, &policy.device))
            .transpose()
            .py()?;
        Ok(Self {
            session,
            env: handle,
            trainer: trainer(
                learning_rate,
                max_grad_norm,
                weight_decay,
                betas,
                lr_schedule,
            )?,
            config,
            batch: None,
            reference,
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
        let report = env.with(py, |mut vec_env| {
            session.collector_mut().collect(&mut vec_env)
        })?;
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
    /// Ends with one synchronisation, to read those diagnostics.
    #[pyo3(signature = (epochs = 4, minibatches = 1))]
    fn update(&mut self, epochs: usize, minibatches: usize) -> PyResult<Stats> {
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

        let mut total = StepInfo {
            step: 0,
            loss: 0.0,
            learning_rate: 0.0,
            grad_norm: 0.0,
        };
        let mut taken = 0.0f32;
        for _ in 0..epochs {
            for index in 0..minibatches {
                let micro = if minibatches == 1 {
                    batch.clone()
                } else {
                    batch.minibatch(index * per, per).py()?
                };
                let info = self
                    .trainer
                    .step(&task, std::slice::from_ref(&micro))
                    .py()?;
                total.loss += info.loss;
                total.grad_norm += info.grad_norm;
                total.learning_rate = info.learning_rate;
                total.step = info.step;
                taken += 1.0;
            }
        }

        let diagnostics = task.stats().unwrap_or_default();
        Ok(Stats {
            round: self.rounds,
            steps: batch.steps(),
            optimizer_steps: total.step,
            loss: total.loss / taken,
            policy_loss: diagnostics.policy_loss,
            value_loss: diagnostics.value_loss,
            entropy: diagnostics.entropy,
            approx_kl: diagnostics.approx_kl,
            clip_fraction: diagnostics.clip_fraction,
            reference_kl: diagnostics.reference_kl,
            grad_norm: total.grad_norm / taken,
            learning_rate: total.learning_rate,
            episode_return: None,
        })
    }

    /// Mean reward per episode *completed* in the window last collected, or
    /// `None` if none completed — a window that ends mid-episode has nothing to
    /// average and must not be confused with one reporting a genuine mean.
    ///
    /// A synchronisation, and the one number worth paying it for: the mean and
    /// the count are packed into one two-element tensor first, so this is a
    /// single read rather than one per value.
    fn episode_return(&self) -> PyResult<Option<f32>> {
        packed_episode_return(self.session.collector().episode_return().py()?)
    }

    /// One collection and one update: the loop body.
    #[pyo3(signature = (epochs = 4, minibatches = 1))]
    fn round(&mut self, py: Python<'_>, epochs: usize, minibatches: usize) -> PyResult<Stats> {
        self.collect(py)?;
        let mut stats = self.update(epochs, minibatches)?;
        stats.episode_return = self.episode_return()?;
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
        if let Some(reference) = self.reference.as_mut() {
            reference.reset();
        }
    }

    /// Save weights, optimizer state and the round/step counters, so training
    /// resumes with its optimizer rather than merely warm-starting weights.
    ///
    /// Distinct from [`PyPolicy::save`][crate::policy::PyPolicy::save]:
    /// that one is weights-only, by design, and loading it back is always a
    /// warm start regardless of which method reads it. Round-trip *this*
    /// through [`PyPpoLearner::load_checkpoint`], not `Policy.load`.
    ///
    /// This is not an exact RL continuation: it does not yet include the
    /// environment's own state, the collector's recurrent state, or the
    /// action-sampling RNG. Call `learner.reset()` after loading for a clean
    /// window, or manage the environment's own state yourself.
    fn save(&self, path: &str) -> PyResult<()> {
        let policy = self.session.policy();
        let metadata = serde_json::json!({ "rounds": self.rounds });
        Checkpoint::capture::<R, E, _>(&*policy, self.trainer.step_count())
            .with_optimizer(&*policy, self.trainer.optimizer())
            .with_metadata(metadata)
            .save(path)
            .py()
    }

    /// Restore weights, optimizer state and counters saved by
    /// [`PyPpoLearner::save`], onto this already-constructed learner.
    ///
    /// `strict` requires every parameter's optimizer state to be present in
    /// the checkpoint; without it, a checkpoint missing some (or all) of the
    /// optimizer's state still loads the weights and warm-starts the rest,
    /// rather than failing outright. A checkpoint written by `Policy.save`
    /// alone has no optimizer state at all, so `strict=True` against one
    /// reports that plainly instead of silently warm-starting.
    #[pyo3(signature = (path, strict = true))]
    fn load_checkpoint(&mut self, path: &str, strict: bool) -> PyResult<()> {
        let checkpoint = Checkpoint::load(path).py()?;
        let policy = self.session.policy();
        checkpoint.restore::<R, E, _>(&*policy, strict).py()?;
        checkpoint
            .restore_optimizer::<R, E, _, _>(&*policy, self.trainer.optimizer_mut(), strict)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.trainer.set_step_count(checkpoint.step);
        if let Some(rounds) = checkpoint.metadata.get("rounds").and_then(|v| v.as_u64()) {
            self.rounds = rounds;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "PpoLearner(num_envs={}, steps={}, rounds={}, buffer={} KiB)",
            self.env.envs(),
            self.steps,
            self.rounds,
            self.session.collector().buffer().bytes() / 1024,
        )
    }
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
        temperature = 1.0,
        seed = 0,
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
        temperature: f32,
        seed: u64,
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
        Ok(Self {
            session,
            env: handle,
            trainer: trainer(
                learning_rate,
                max_grad_norm,
                weight_decay,
                betas,
                lr_schedule,
            )?,
            schedule: schedule.unwrap_or_default().inner,
            entropy_bonus,
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

    /// One DAgger round: roll out the mixture, label it, take one gradient step.
    ///
    /// `beta` overrides the schedule for this round. `agreement` costs one extra
    /// replay and one synchronisation, and is the number that says whether the
    /// cloning is working.
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
        let info = self
            .trainer
            .step(&task, std::slice::from_ref(&batch))
            .py()?;
        let agreement = agreement.then(|| task.agreement(&batch).py()).transpose()?;

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

    /// Save weights, optimizer state and the round/step counters, so training
    /// resumes with its optimizer rather than merely warm-starting weights.
    ///
    /// See [`PyPpoLearner::save`] for what this does and does not cover; the
    /// same limitations apply here.
    fn save(&self, path: &str) -> PyResult<()> {
        let policy = self.session.policy();
        let metadata = serde_json::json!({ "rounds": self.rounds });
        Checkpoint::capture::<R, E, _>(&*policy, self.trainer.step_count())
            .with_optimizer(&*policy, self.trainer.optimizer())
            .with_metadata(metadata)
            .save(path)
            .py()
    }

    /// Restore weights, optimizer state and counters saved by
    /// [`PyImitationLearner::save`]. See [`PyPpoLearner::load_checkpoint`].
    #[pyo3(signature = (path, strict = true))]
    fn load_checkpoint(&mut self, path: &str, strict: bool) -> PyResult<()> {
        let checkpoint = Checkpoint::load(path).py()?;
        let policy = self.session.policy();
        checkpoint.restore::<R, E, _>(&*policy, strict).py()?;
        checkpoint
            .restore_optimizer::<R, E, _, _>(&*policy, self.trainer.optimizer_mut(), strict)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.trainer.set_step_count(checkpoint.step);
        if let Some(rounds) = checkpoint.metadata.get("rounds").and_then(|v| v.as_u64()) {
            self.rounds = rounds;
        }
        Ok(())
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

/// Pack `episode_return()`'s `(mean, count)` into one host read and turn a zero
/// count into `None`, so a window that ended mid-episode is never mistaken for
/// one reporting a genuine mean.
fn packed_episode_return(
    parts: (mamba3::tensor::Tensor<R, E>, mamba3::tensor::Tensor<R, E>),
) -> PyResult<Option<f32>> {
    let (mean, count) = parts;
    let packed = mamba3::tensor::ops::movement::cat(&[mean, count], 0).py()?;
    let values = packed.to_f32();
    Ok((values[1] > 0.0).then_some(values[0]))
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
    packed_episode_return(session.collector().episode_return().py()?)
}
