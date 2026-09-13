//! The policy, and the `O(1)` way to act with it.
//!
//! [`PyPolicy`] is a handle onto a recurrent actor-critic stack: an observation
//! encoder, `n_layers` Mamba-3 blocks, and the actor and critic heads. It is held
//! behind a reference count, so a learner and the Python object that built it name
//! the same weights — training through one is visible through the other, which is
//! what makes `learner.policy.save(...)` mean what it looks like it means.
//!
//! [`PyRollout`] is the other half: the fixed-size recurrent state, plus the step
//! that advances it. Its cost per step does not grow with how long an episode has
//! run, which is the property the whole crate exists for — a transformer serving
//! the same loop carries a cache that grows with every action it takes.
//!
//! ```python
//! rollout = mamba3_rl.Rollout(policy, num_envs=8, temperature=0.0)
//! obs = env.reset()
//! for _ in range(steps):
//!     actions, values, log_probs = rollout.step(obs, reset=done)
//!     obs, reward, done = env.step(actions)
//! ```

use std::rc::Rc;

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::nn::Module;
use mamba3::rl::{Mamba3Policy, Mamba3StateBuffer, sample_categorical};
use mamba3::train::Checkpoint;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array;
use crate::config::PyPolicyConfig;
use crate::err::IntoPyResult;
use crate::{E, R};

/// What one rollout step hands back: `(actions, values, log_probs)`.
type ActionArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);

/// The same step without the draw: `(logits, values)`.
type DistributionArrays<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>);

/// A recurrent actor-critic policy over a Mamba-3 stack.
#[pyclass(module = "mamba3_rl", name = "Policy", unsendable)]
pub struct PyPolicy {
    pub(crate) inner: Rc<Mamba3Policy<R, E>>,
    pub(crate) device: Device<R>,
}

#[pymethods]
impl PyPolicy {
    #[new]
    fn new(config: &PyPolicyConfig) -> PyResult<Self> {
        let device = Device::<R>::default();
        let inner = config.inner.init::<R, E>(&device).py()?;
        Ok(Self {
            inner: Rc::new(inner),
            device,
        })
    }

    /// The architecture this policy was built from.
    #[getter]
    fn config(&self) -> PyPolicyConfig {
        PyPolicyConfig::from_inner(self.inner.config().clone())
    }

    /// Width of one observation vector.
    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.config().obs_dim
    }

    /// Number of discrete actions.
    #[getter]
    fn action_dim(&self) -> usize {
        self.inner.config().action_dim
    }

    /// Scalars in the stack, frozen ones included.
    #[getter]
    fn num_parameters(&self) -> usize {
        Module::<R, E>::num_parameters(&*self.inner)
    }

    /// Scalars the optimizer would move.
    #[getter]
    fn num_trainable_parameters(&self) -> usize {
        Module::<R, E>::num_trainable_parameters(&*self.inner)
    }

    /// The backend the weights live on.
    #[getter]
    fn backend(&self) -> &'static str {
        self.device.name()
    }

    /// Every parameter, by path and shape.
    fn describe(&self) -> String {
        Module::<R, E>::describe(&*self.inner)
    }

    /// Put the stack in training mode (dropout on, observers recording).
    fn train(&self) {
        Module::<R, E>::train(&*self.inner);
    }

    /// Put the stack in evaluation mode.
    fn eval(&self) {
        Module::<R, E>::eval(&*self.inner);
    }

    /// Freeze every parameter whose path contains one of `patterns`.
    ///
    /// A frozen parameter enters the graph as a constant: no gradient is computed
    /// and no activation is kept for it. `policy.freeze(["blocks"])` leaves only
    /// the heads trainable.
    fn freeze(&self, patterns: Vec<String>) {
        let patterns: Vec<&str> = patterns.iter().map(String::as_str).collect();
        Module::<R, E>::freeze_matching(&*self.inner, &patterns);
    }

    /// Unfreeze every parameter whose path contains one of `patterns`.
    fn unfreeze(&self, patterns: Vec<String>) {
        let patterns: Vec<&str> = patterns.iter().map(String::as_str).collect();
        Module::<R, E>::unfreeze_matching(&*self.inner, &patterns);
    }

    /// Write the weights and the architecture to a JSON checkpoint.
    ///
    /// The architecture travels with the weights, so [`PyPolicy::load`] needs
    /// nothing else. Weights are stored as `f32` whatever the compute element is.
    #[pyo3(signature = (path, step = 0))]
    fn save(&self, path: &str, step: u64) -> PyResult<()> {
        let metadata = serde_json::json!({ "policy": self.config().as_json() });
        Checkpoint::capture::<R, E, _>(&*self.inner, step)
            .with_metadata(metadata)
            .save(path)
            .py()
    }

    /// Read back what [`PyPolicy::save`] wrote.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let checkpoint = Checkpoint::load(path).py()?;
        let stored = checkpoint.metadata.get("policy").ok_or_else(|| {
            PyValueError::new_err(format!(
                "{path} carries no policy architecture; it was not written by \
                 Policy.save()"
            ))
        })?;
        let config = PyPolicyConfig::from_json(stored)?;
        let policy = Self::new(&config)?;
        checkpoint.restore::<R, E, _>(&*policy.inner, true).py()?;
        Ok(policy)
    }

    /// Load weights from a checkpoint into this policy.
    ///
    /// `strict=False` allows a partial checkpoint, such as adapters alone, onto a
    /// policy that already holds a base.
    #[pyo3(signature = (path, strict = true))]
    fn load_weights(&self, path: &str, strict: bool) -> PyResult<()> {
        Checkpoint::load(path)
            .py()?
            .restore::<R, E, _>(&*self.inner, strict)
            .py()
    }

    fn __repr__(&self) -> String {
        let config = self.inner.config();
        format!(
            "Policy(obs_dim={}, action_dim={}, d_model={}, n_layers={}, \
             parameters={}, backend={})",
            config.obs_dim,
            config.action_dim,
            config.ssm.d_model,
            config.n_layers,
            Module::<R, E>::num_parameters(&*self.inner),
            self.device.name(),
        )
    }
}

impl PyPolicy {
    /// Another handle onto the same weights.
    pub fn share(&self) -> Rc<Mamba3Policy<R, E>> {
        self.inner.clone()
    }

    /// Wrap a policy a learner already holds.
    pub fn from_shared(inner: Rc<Mamba3Policy<R, E>>, device: Device<R>) -> Self {
        Self { inner, device }
    }
}

/// The recurrent state of `num_envs` environments, and the step that advances it.
///
/// One step is `O(1)` in the length of the episode: the state is
/// `[layers, envs, heads, head_dim, d_state]` and is overwritten in place, so a
/// loop of any length allocates nothing after this object is built.
#[pyclass(module = "mamba3_rl", name = "Rollout", unsendable)]
pub struct PyRollout {
    policy: Rc<Mamba3Policy<R, E>>,
    state: Mamba3StateBuffer<R, E>,
    envs: usize,
    obs_dim: usize,
    action_dim: usize,
    temperature: f32,
    seed: u64,
    draws: u64,
    device: Device<R>,
}

#[pymethods]
impl PyRollout {
    #[new]
    #[pyo3(signature = (policy, num_envs, *, temperature = 1.0, seed = 0))]
    fn new(policy: &PyPolicy, num_envs: usize, temperature: f32, seed: u64) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("a rollout needs at least one environment"));
        }
        if temperature < 0.0 {
            return Err(PyValueError::new_err(
                "temperature must be non-negative; 0 acts greedily",
            ));
        }
        let inner = policy.share();
        let state = inner.empty_state(num_envs, &policy.device);
        Ok(Self {
            envs: num_envs,
            obs_dim: inner.config().obs_dim,
            action_dim: inner.config().action_dim,
            policy: inner,
            state,
            temperature,
            seed,
            draws: 0,
            device: policy.device.clone(),
        })
    }

    /// How many environments this state covers.
    #[getter]
    fn num_envs(&self) -> usize {
        self.envs
    }

    /// Bytes of recurrent state, fixed for the life of the object.
    #[getter]
    fn state_bytes(&self) -> usize {
        self.state.bytes()
    }

    /// Sampling temperature. `0` acts greedily.
    #[getter]
    fn temperature(&self) -> f32 {
        self.temperature
    }

    #[setter]
    fn set_temperature(&mut self, temperature: f32) -> PyResult<()> {
        if temperature < 0.0 {
            return Err(PyValueError::new_err(
                "temperature must be non-negative; 0 acts greedily",
            ));
        }
        self.temperature = temperature;
        Ok(())
    }

    /// Forget everything: zero the recurrent state of every environment.
    fn reset(&mut self) {
        self.state.reset_all();
    }

    /// Advance every environment by one observation and draw an action.
    ///
    /// `obs` is `[num_envs, obs_dim]`. `reset` is an optional `[num_envs]` mask
    /// holding `1` for an environment whose previous episode has ended — pass the
    /// `done` flags the environment returned, and the recurrence and the short
    /// convolution are both cut there, so the new episode starts from nothing.
    ///
    /// Returns `(actions, values, log_probs)`, each `[num_envs]`.
    #[pyo3(signature = (obs, reset = None, temperature = None))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'py, PyAny>,
        reset: Option<&Bound<'py, PyAny>>,
        temperature: Option<f32>,
    ) -> PyResult<ActionArrays<'py>> {
        let temperature = temperature.unwrap_or(self.temperature);
        if temperature < 0.0 {
            return Err(PyValueError::new_err(
                "temperature must be non-negative; 0 acts greedily",
            ));
        }
        let (logits, values) = self.advance(obs, reset)?;
        self.draws = self.draws.wrapping_add(1);
        let seed = self.seed ^ self.draws.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let (actions, log_probs) = sample_categorical(&logits, temperature, seed).py()?;
        Ok((
            array::ids_to_1d(py, &actions),
            array::to_1d(py, &values),
            array::to_1d(py, &log_probs),
        ))
    }

    /// The same step, reporting the whole distribution instead of a draw from it.
    ///
    /// Advances the state exactly as [`PyRollout::step`] does. Returns
    /// `(logits, values)`, shaped `[num_envs, action_dim]` and `[num_envs]`.
    #[pyo3(signature = (obs, reset = None))]
    fn evaluate<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'py, PyAny>,
        reset: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<DistributionArrays<'py>> {
        let (logits, values) = self.advance(obs, reset)?;
        Ok((
            array::to_2d(py, &logits, self.envs, self.action_dim)?,
            array::to_1d(py, &values),
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Rollout(num_envs={}, temperature={}, state={} KiB)",
            self.envs,
            self.temperature,
            self.state.bytes() / 1024,
        )
    }
}

impl PyRollout {
    /// One `O(1)` step: `[envs, action_dim]` logits and `[envs]` values.
    fn advance(
        &mut self,
        obs: &Bound<'_, PyAny>,
        reset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(mamba3::tensor::Tensor<R, E>, mamba3::tensor::Tensor<R, E>)> {
        let observation = array::tensor_2d(obs, self.envs, self.obs_dim, "obs", &self.device)?;
        let reset = reset
            .map(|mask| array::tensor_1d(mask, self.envs, "reset", &self.device))
            .transpose()?;
        let windowed = Var::constant(
            observation
                .reshape(vec![self.envs, 1, self.obs_dim])
                .py()?,
        );
        let out = self
            .policy
            .step(&windowed, &mut self.state, reset.as_ref())
            .py()?;
        let logits = out
            .logits
            .tensor()
            .reshape(vec![self.envs, self.action_dim])
            .py()?;
        let values = out.value.tensor().reshape(vec![self.envs]).py()?;
        Ok((logits, values))
    }
}
