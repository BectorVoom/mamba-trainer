//! Environments, whichever side of the boundary they live on.
//!
//! The crate's [`VecEnv`] is deliberately a *device* interface: observations are
//! device tensors, actions are device ids, and a collection loop that stays inside
//! it never synchronises. An environment written in Python cannot be that, and
//! pretending otherwise would be the wrong trade — so [`PyEnvAdapter`] pays the
//! copy honestly, twice per step, and everything above it is unchanged.
//!
//! | | where it runs | cost per step |
//! |---|---|---|
//! | [`RecallEnv`] | a kernel on the device | nothing leaves the device |
//! | a Python object | the interpreter | two copies and a synchronisation |
//!
//! The second is still the right way to reach an existing simulator, and for
//! anything whose own step is slower than a few microseconds the copy is noise.
//! For a task written to be trained *fast*, write it as a kernel instead — as
//! [`RecallEnv`] is — and the loop never touches the host at all.
//!
//! # The protocol
//!
//! An object is a vectorised environment here if it has `num_envs`, `obs_dim` and
//! `action_dim`, and:
//!
//! * `reset() -> [num_envs, obs_dim] float array`
//! * `step(actions: [num_envs] int64) -> (observation, reward, done)`, shaped
//!   `[num_envs, obs_dim]`, `[num_envs]`, `[num_envs]`
//! * optionally `expert_actions() -> [num_envs] int64 | None`, which is what
//!   imitation learning labels a state with.
//!
//! Environments **auto-reset**: where `done` is `1`, the observation returned
//! alongside it is already the first observation of the next episode. That is what
//! lets a rollout of fixed length hold environments whose episodes end at
//! different moments, and the recurrence is cut at exactly those points.

use mamba3::backend::Device;
use mamba3::error::Result;
use mamba3::rl::{EnvStep, RecallEnv, VecEnv};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array;
use crate::err::{ErrorSlot, IntoPyResult};
use crate::{E, R};

/// What one environment step hands back: `(observation, reward, done)`.
type StepArrays<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);

/// A cue-recall task: see a symbol once, name it `horizon` steps later.
///
/// The reward is sparse and it is at the end, so a policy with no memory cannot do
/// better than guessing — `1 / symbols` — and a perfect one earns `1`. The gap
/// between those two numbers is the recurrent state doing its job, which is what
/// makes this the task to sanity-check a run on. The whole transition function is
/// one kernel, so a rollout over it never touches the host.
#[pyclass(module = "mamba3_rl", name = "RecallEnv", unsendable)]
pub struct PyRecallEnv {
    pub(crate) inner: RecallEnv<R, E>,
    device: Device<R>,
}

#[pymethods]
impl PyRecallEnv {
    #[new]
    #[pyo3(signature = (num_envs, symbols = 4, horizon = 8, seed = 0))]
    fn new(num_envs: usize, symbols: usize, horizon: usize, seed: u64) -> PyResult<Self> {
        let device = Device::<R>::default();
        Ok(Self {
            inner: RecallEnv::new(num_envs, symbols, horizon, seed, &device).py()?,
            device,
        })
    }

    /// How many environments run in parallel.
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.envs()
    }

    /// Width of one observation: one channel per symbol, plus a clock and a
    /// cue-present flag.
    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }

    /// Number of symbols, which is also the number of actions.
    #[getter]
    fn action_dim(&self) -> usize {
        self.inner.action_dim()
    }

    /// Steps in one episode.
    #[getter]
    fn horizon(&self) -> usize {
        self.inner.horizon()
    }

    /// What a policy that guesses uniformly earns per episode.
    #[getter]
    fn chance_return(&self) -> f32 {
        self.inner.chance_return()
    }

    /// What a policy with a perfect memory earns per episode.
    #[getter]
    fn optimal_return(&self) -> f32 {
        self.inner.optimal_return()
    }

    /// Start every environment and return the first observation.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let obs = self.inner.reset().py()?;
        array::to_2d(py, &obs, self.inner.envs(), self.inner.obs_dim())
    }

    /// Apply one action per environment.
    ///
    /// Returns `(observation, reward, done)`. Reading them is a synchronisation —
    /// the learners in this module never do it, which is the point of them.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyAny>,
    ) -> PyResult<StepArrays<'py>> {
        let envs = self.inner.envs();
        let ids = array::ids_1d(
            actions,
            envs,
            self.inner.action_dim(),
            "actions",
            &self.device,
        )?;
        let step = self.inner.step(&ids).py()?;
        Ok((
            array::to_2d(py, &step.observation, envs, self.inner.obs_dim())?,
            array::to_1d(py, &step.reward)?,
            array::to_1d(py, &step.done)?,
        ))
    }

    /// What the expert would do on the observation most recently returned.
    fn expert_actions<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<i64>>>> {
        self.inner
            .expert_actions()
            .map(|ids| array::ids_to_1d(py, &ids))
            .transpose()
    }

    fn __repr__(&self) -> String {
        format!(
            "RecallEnv(num_envs={}, symbols={}, horizon={})",
            self.inner.envs(),
            self.inner.action_dim(),
            self.inner.horizon(),
        )
    }
}

/// A [`VecEnv`] that is really a Python object.
///
/// Built for the duration of one call into Rust and thrown away again, so it holds
/// a borrowed handle rather than a reference count. Every failure — a raised
/// exception, a wrong shape, an action id outside the space the object declares —
/// is parked in [`ErrorSlot`] and re-raised in Python with its traceback intact.
pub struct PyEnvAdapter<'py> {
    obj: Bound<'py, PyAny>,
    envs: usize,
    obs_dim: usize,
    action_dim: usize,
    expert: bool,
    masked: bool,
    device: Device<R>,
    slot: ErrorSlot,
}

impl<'py> PyEnvAdapter<'py> {
    /// Run `f` against the Python object, parking whatever it raises.
    fn attempt<T>(&self, context: &str, f: impl FnOnce(&Self) -> PyResult<T>) -> Result<T> {
        f(self).map_err(|err| self.slot.store(context, err))
    }
}

impl VecEnv<R, E> for PyEnvAdapter<'_> {
    fn envs(&self) -> usize {
        self.envs
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn reset(&mut self) -> Result<Tensor<R, E>> {
        self.attempt("reset()", |this| {
            let observation = this.obj.call_method0("reset")?;
            array::tensor_2d(
                &observation,
                this.envs,
                this.obs_dim,
                "the observation returned by reset()",
                &this.device,
            )
        })
    }

    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, E>> {
        self.attempt("step()", |this| {
            let py = this.obj.py();
            let ids = array::ids_to_1d(py, actions)?;
            let returned = this.obj.call_method1("step", (ids,))?;
            let (observation, reward, done) = returned
                .extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>)>()
                .map_err(|_| {
                    PyValueError::new_err(
                        "step() must return a (observation, reward, done) tuple; \
                         a terminated environment auto-resets, so the observation \
                         beside a done flag is the first one of the next episode",
                    )
                })?;
            Ok(EnvStep {
                observation: array::tensor_2d(
                    &observation,
                    this.envs,
                    this.obs_dim,
                    "the observation returned by step()",
                    &this.device,
                )?,
                reward: array::tensor_1d(
                    &reward,
                    this.envs,
                    "the reward returned by step()",
                    &this.device,
                )?,
                done: array::tensor_1d(
                    &done,
                    this.envs,
                    "the done flags returned by step()",
                    &this.device,
                )?,
            })
        })
    }

    fn expert_actions(&self) -> Option<IdTensor<R>> {
        if !self.expert {
            return None;
        }
        // No error channel here, so a failure parks its exception and reports "no
        // expert". The caller consults the slot before believing that.
        self.attempt("expert_actions()", |this| {
            let returned = this.obj.call_method0("expert_actions")?;
            if returned.is_none() {
                return Ok(None);
            }
            array::ids_1d(
                &returned,
                this.envs,
                this.action_dim,
                "the actions returned by expert_actions()",
                &this.device,
            )
            .map(Some)
        })
        .ok()
        .flatten()
    }

    fn action_mask(&self) -> mamba3::error::Result<Option<Tensor<R, E>>> {
        if !self.masked {
            return Ok(None);
        }
        // Unlike `expert_actions`, a failure here stops the collection at once:
        // the collector has not drawn this step's action yet, and drawing it from
        // an unmasked row would step the environment with an action it may have
        // just said was illegal. The exception itself is parked and re-raised.
        self.attempt("action_mask()", |this| {
            let returned = this.obj.call_method0("action_mask")?;
            if returned.is_none() {
                return Ok(None);
            }
            array::action_mask_2d(
                &returned,
                this.envs,
                this.action_dim,
                "the mask returned by action_mask()",
                &this.device,
            )
            .map(Some)
        })
    }
}

/// Either kind of environment, held by reference count so Python keeps its handle.
enum EnvKind {
    /// The built-in task, stepped through its own kernels.
    Recall(Py<PyRecallEnv>),
    /// Anything else that speaks the protocol.
    Python(Py<PyAny>),
}

/// What a learner holds instead of an environment.
///
/// The environment object stays a Python object — visible, inspectable and
/// steppable from Python between rounds — and is borrowed for the length of one
/// collection. Its dimensions are read once, at construction, because a learner
/// that discovered them again every window would be asking the interpreter for
/// something that cannot change.
pub struct EnvHandle {
    kind: EnvKind,
    envs: usize,
    obs_dim: usize,
    action_dim: usize,
    expert: bool,
    masked: bool,
    device: Device<R>,
}

/// The first attribute of `names` the object has, as a positive integer.
fn dimension(obj: &Bound<'_, PyAny>, names: &[&str]) -> PyResult<usize> {
    for name in names {
        if obj.hasattr(*name)? {
            let value = obj.getattr(*name)?;
            let value: usize = value.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "an environment's {name} must be a non-negative integer attribute, \
                     not a method"
                ))
            })?;
            if value == 0 {
                return Err(PyValueError::new_err(format!(
                    "an environment's {name} must be positive"
                )));
            }
            return Ok(value);
        }
    }
    Err(PyValueError::new_err(format!(
        "the environment has no {} attribute; a vectorised environment declares \
         num_envs, obs_dim and action_dim, and implements reset() and step()",
        names.join(" or ")
    )))
}

impl EnvHandle {
    /// Adopt whatever Python passed in, checking it can be driven before a run
    /// starts rather than at the first step of the first window.
    pub fn adopt(obj: &Bound<'_, PyAny>, device: &Device<R>) -> PyResult<Self> {
        if let Ok(recall) = obj.cast::<PyRecallEnv>() {
            let env = recall.borrow();
            return Ok(Self {
                envs: env.inner.envs(),
                obs_dim: env.inner.obs_dim(),
                action_dim: env.inner.action_dim(),
                expert: true,
                masked: false,
                device: device.clone(),
                kind: EnvKind::Recall(recall.clone().unbind()),
            });
        }
        for method in ["reset", "step"] {
            if !obj.hasattr(method)? {
                return Err(PyValueError::new_err(format!(
                    "the environment has no {method}(); a vectorised environment \
                     declares num_envs, obs_dim and action_dim, and implements \
                     reset() and step()"
                )));
            }
        }
        Ok(Self {
            envs: dimension(obj, &["num_envs", "envs"])?,
            obs_dim: dimension(obj, &["obs_dim"])?,
            action_dim: dimension(obj, &["action_dim"])?,
            expert: obj.hasattr("expert_actions")?,
            masked: obj.hasattr("action_mask")?,
            device: device.clone(),
            kind: EnvKind::Python(obj.clone().unbind()),
        })
    }

    /// How many environments run in parallel.
    pub fn envs(&self) -> usize {
        self.envs
    }

    /// Width of one observation.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Number of discrete actions.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Whether the environment can label a state with an expert's action.
    pub fn has_expert(&self) -> bool {
        self.expert
    }

    /// The Python object, for handing back to whoever passed it in.
    pub fn object(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.kind {
            EnvKind::Recall(env) => env.clone_ref(py).into_any(),
            EnvKind::Python(obj) => obj.clone_ref(py),
        }
    }

    /// Borrow the environment for the length of one call.
    ///
    /// The borrow is exclusive: Python code that steps the same environment from
    /// inside a callback gets a clear "already borrowed" error rather than two
    /// loops interleaved over one simulator.
    pub fn with<T>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut dyn VecEnv<R, E>) -> Result<T>,
    ) -> PyResult<T> {
        match &self.kind {
            EnvKind::Recall(env) => {
                let mut env = env.bind(py).try_borrow_mut()?;
                f(&mut env.inner).py()
            }
            EnvKind::Python(obj) => {
                let mut adapter = PyEnvAdapter {
                    obj: obj.bind(py).clone(),
                    envs: self.envs,
                    obs_dim: self.obs_dim,
                    action_dim: self.action_dim,
                    expert: self.expert,
                    masked: self.masked,
                    device: self.device.clone(),
                    slot: ErrorSlot::default(),
                };
                let result = f(&mut adapter);
                adapter.slot.resolve(result)
            }
        }
    }
}

/// The error a learner raises when it is handed an environment whose shape does
/// not match the policy it is meant to drive.
pub fn check_against_policy(env: &EnvHandle, obs_dim: usize, action_dim: usize) -> PyResult<()> {
    if env.obs_dim() != obs_dim || env.action_dim() != action_dim {
        return Err(PyValueError::new_err(format!(
            "the policy reads {obs_dim} observation channels and writes {action_dim} \
             actions; the environment offers {} and {}",
            env.obs_dim(),
            env.action_dim(),
        )));
    }
    Ok(())
}

/// Refuse an environment that cannot label a state, before a run starts.
pub fn refuse_without_expert(env: &EnvHandle) -> PyResult<()> {
    if env.has_expert() {
        return Ok(());
    }
    Err(PyValueError::new_err(
        "imitation learning needs an environment that can say what an expert would \
         do: implement expert_actions() returning [num_envs] action ids",
    ))
}
