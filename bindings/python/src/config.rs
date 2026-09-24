//! The configurations a run is shaped by: the policy's architecture, PPO's
//! hyperparameters, the learning-rate schedule and the moving average of the
//! weights.
//!
//! Both are plain data, both validate on construction rather than at the first
//! kernel launch, and both round-trip through JSON — which is what lets a
//! checkpoint carry the architecture that produced it, so
//! [`crate::policy::Policy::load`] needs nothing but the file.

use mamba3::rl::{ActionHeadConfig, Mamba3PolicyConfig, PpoConfig};
use mamba3::ssm::config::{Discretization, SsmConfig, StateDynamics};
use mamba3::train::{EmaConfig, EmaWarmup, LrSchedule};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::entity::{PyEntityEncoderConfig, PyObsSpec, PyPointerHead, PyPoolingConfig};
use crate::err::IntoPyResult;

/// An optional key of a stored policy config; absent or `null` is `None`.
///
/// A macro rather than a generic function so the bindings need no direct
/// dependency on `serde` for the `DeserializeOwned` bound.
macro_rules! optional {
    ($value:expr, $name:literal) => {
        match $value.get($name) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(v) => serde_json::from_value(v.clone()).map(Some).map_err(|e| {
                PyValueError::new_err(format!(
                    concat!("unusable ", $name, " in the policy config: {}"),
                    e
                ))
            }),
        }
    };
}

/// Parse the discretization rule, naming the alternatives when it is not one.
fn discretization(name: &str) -> PyResult<Discretization> {
    match name {
        "euler" => Ok(Discretization::Euler),
        "trapezoid" => Ok(Discretization::Trapezoid),
        "learned_trapezoid" => Ok(Discretization::LearnedTrapezoid),
        other => Err(PyValueError::new_err(format!(
            "unknown discretization {other:?}; \
             expected 'euler', 'trapezoid' or 'learned_trapezoid'"
        ))),
    }
}

/// The name [`discretization`] accepts for a rule.
fn discretization_name(rule: Discretization) -> &'static str {
    match rule {
        Discretization::Euler => "euler",
        Discretization::Trapezoid => "trapezoid",
        Discretization::LearnedTrapezoid => "learned_trapezoid",
    }
}

/// Parse the state transition structure.
fn dynamics(name: &str) -> PyResult<StateDynamics> {
    match name {
        "real" => Ok(StateDynamics::Real),
        "rotational" => Ok(StateDynamics::Rotational),
        other => Err(PyValueError::new_err(format!(
            "unknown dynamics {other:?}; expected 'real' or 'rotational'"
        ))),
    }
}

/// The name [`dynamics`] accepts for a transition.
fn dynamics_name(kind: StateDynamics) -> &'static str {
    match kind {
        StateDynamics::Real => "real",
        StateDynamics::Rotational => "rotational",
    }
}

/// The architecture of a recurrent actor-critic policy.
///
/// The four positional arguments are the ones a run is actually chosen by; the
/// keyword arguments open up the mixer underneath, and each defaults to what
/// `d_model` implies. `conv_kernel=0` removes the short causal convolution
/// entirely, which Mamba-3 permits.
///
/// `obs_spec` reads the flat observation as entity sets: each set gets a shared
/// encoder (`entity_encoders`, by set name, default `EntityEncoderConfig()`),
/// pooled under the presence flags (`pooling`, default mean and max) and
/// projected to `d_model`. `action_head=PointerHead(set)` scores that set's
/// entities instead of a flat linear head. All default off, and
/// `PolicyConfig(obs_dim, action_dim)` alone is today's flat policy.
#[pyclass(module = "mamba3_rl", name = "PolicyConfig", from_py_object)]
#[derive(Clone)]
pub struct PyPolicyConfig {
    pub(crate) inner: Mamba3PolicyConfig,
}

#[pymethods]
impl PyPolicyConfig {
    #[new]
    #[pyo3(signature = (
        obs_dim,
        action_dim,
        d_model = 64,
        n_layers = 2,
        *,
        n_heads = None,
        head_dim = None,
        d_state = None,
        n_groups = None,
        chunk_size = None,
        conv_kernel = None,
        discretization = "learned_trapezoid",
        dynamics = "rotational",
        norm_eps = 1e-5,
        seed = 0,
        obs_spec = None,
        entity_encoders = None,
        pooling = None,
        action_head = None,
    ))]
    fn new(
        obs_dim: usize,
        action_dim: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: Option<usize>,
        head_dim: Option<usize>,
        d_state: Option<usize>,
        n_groups: Option<usize>,
        chunk_size: Option<usize>,
        conv_kernel: Option<usize>,
        discretization: &str,
        dynamics: &str,
        norm_eps: f32,
        seed: u64,
        obs_spec: Option<PyObsSpec>,
        entity_encoders: Option<std::collections::BTreeMap<String, PyEntityEncoderConfig>>,
        pooling: Option<PyPoolingConfig>,
        action_head: Option<PyPointerHead>,
    ) -> PyResult<Self> {
        let mut inner = Mamba3PolicyConfig::new(obs_dim, action_dim, d_model, n_layers);
        inner.norm_eps = norm_eps;
        inner.seed = seed;
        inner.obs_spec = obs_spec.map(|s| s.inner);
        inner.entity_encoders = entity_encoders
            .unwrap_or_default()
            .into_iter()
            .map(|(name, c)| (name, c.inner))
            .collect();
        if let Some(pooling) = pooling {
            inner.pooling = pooling.inner;
        }
        if let Some(head) = action_head {
            inner.action_head = ActionHeadConfig::Pointer(head.inner);
        }
        let ssm = &mut inner.ssm;
        // `head_dim` first: `n_heads` defaults to `d_model / head_dim`, so setting
        // one without the other should still describe a consistent stack.
        if let Some(head_dim) = head_dim {
            ssm.head_dim = head_dim;
            ssm.n_heads = (d_model / head_dim.max(1)).max(1);
            ssm.n_groups = ssm.n_heads;
        }
        if let Some(n_heads) = n_heads {
            ssm.n_heads = n_heads;
            ssm.n_groups = n_heads;
        }
        if let Some(d_state) = d_state {
            ssm.d_state = d_state;
        }
        if let Some(n_groups) = n_groups {
            ssm.n_groups = n_groups;
        }
        if let Some(chunk_size) = chunk_size {
            ssm.chunk_size = chunk_size;
        }
        if let Some(kernel) = conv_kernel {
            ssm.conv_kernel = (kernel > 0).then_some(kernel);
        }
        ssm.discretization = self::discretization(discretization)?;
        ssm.dynamics = self::dynamics(dynamics)?;
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// Width of one observation vector.
    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim
    }

    /// Number of discrete actions.
    #[getter]
    fn action_dim(&self) -> usize {
        self.inner.action_dim
    }

    /// Residual stream width.
    #[getter]
    fn d_model(&self) -> usize {
        self.inner.ssm.d_model
    }

    /// Number of Mamba-3 blocks.
    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers
    }

    /// Number of state space heads.
    #[getter]
    fn n_heads(&self) -> usize {
        self.inner.ssm.n_heads
    }

    /// Channels per head.
    #[getter]
    fn head_dim(&self) -> usize {
        self.inner.ssm.head_dim
    }

    /// State size per head.
    #[getter]
    fn d_state(&self) -> usize {
        self.inner.ssm.d_state
    }

    /// Number of `B`/`C` groups.
    #[getter]
    fn n_groups(&self) -> usize {
        self.inner.ssm.n_groups
    }

    /// Sequence chunk length used by the parallel scan.
    #[getter]
    fn chunk_size(&self) -> usize {
        self.inner.ssm.chunk_size
    }

    /// Width of the short causal convolution, or `None` when it is disabled.
    #[getter]
    fn conv_kernel(&self) -> Option<usize> {
        self.inner.ssm.conv_kernel
    }

    /// Discretization rule.
    #[getter]
    fn discretization(&self) -> &'static str {
        discretization_name(self.inner.ssm.discretization)
    }

    /// Real or rotational state transition.
    #[getter]
    fn dynamics(&self) -> &'static str {
        dynamics_name(self.inner.ssm.dynamics)
    }

    /// Normalisation epsilon.
    #[getter]
    fn norm_eps(&self) -> f32 {
        self.inner.norm_eps
    }

    /// Initialisation seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// How the observation is read as entity sets, or `None` for the flat policy.
    #[getter]
    fn obs_spec(&self) -> Option<PyObsSpec> {
        self.inner
            .obs_spec
            .clone()
            .map(|inner| PyObsSpec { inner })
    }

    /// The encoder of every entity set, defaults filled in; empty when flat.
    #[getter]
    fn entity_encoders(&self) -> std::collections::BTreeMap<String, PyEntityEncoderConfig> {
        self.inner
            .obs_spec
            .iter()
            .flat_map(|spec| &spec.sets)
            .map(|set| {
                (
                    set.name.clone(),
                    PyEntityEncoderConfig {
                        inner: self.inner.entity_encoder(&set.name),
                    },
                )
            })
            .collect()
    }

    /// How each entity set is pooled.
    #[getter]
    fn pooling(&self) -> PyPoolingConfig {
        PyPoolingConfig {
            inner: self.inner.pooling.clone(),
        }
    }

    /// The pointer head, or `None` for the flat linear head.
    #[getter]
    fn action_head(&self) -> Option<PyPointerHead> {
        match &self.inner.action_head {
            ActionHeadConfig::Flat => None,
            ActionHeadConfig::Pointer(head) => Some(PyPointerHead {
                inner: head.clone(),
            }),
        }
    }

    /// Whether the observation is read as entity sets.
    #[getter]
    fn is_structured(&self) -> bool {
        self.inner.is_structured()
    }

    /// The configuration as a plain dictionary, as a checkpoint stores it.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json = serde_json::to_string(&self.as_json()).map_err(|e| {
            PyValueError::new_err(format!("could not serialise the policy config: {e}"))
        })?;
        py.import("json")?.call_method1("loads", (json,))
    }

    /// Rebuild a configuration from [`PyPolicyConfig::to_dict`].
    #[staticmethod]
    fn from_dict(mapping: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = mapping.py();
        let text: String = py
            .import("json")?
            .call_method1("dumps", (mapping,))?
            .extract()?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| PyValueError::new_err(format!("not a policy config: {e}")))?;
        Self::from_json(&value)
    }

    fn __repr__(&self) -> String {
        let structure = match &self.inner.obs_spec {
            None => String::new(),
            Some(spec) => format!(
                ", obs_spec=ObsSpec(globals={}, sets={:?}), action_head={}",
                spec.globals,
                spec.sets.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
                match &self.inner.action_head {
                    ActionHeadConfig::Flat => "flat".to_string(),
                    ActionHeadConfig::Pointer(h) => format!("pointer({:?})", h.set),
                }
            ),
        };
        format!(
            "PolicyConfig(obs_dim={}, action_dim={}, d_model={}, n_layers={}, \
             n_heads={}, head_dim={}, d_state={}, seed={}{structure})",
            self.inner.obs_dim,
            self.inner.action_dim,
            self.inner.ssm.d_model,
            self.inner.n_layers,
            self.inner.ssm.n_heads,
            self.inner.ssm.head_dim,
            self.inner.ssm.d_state,
            self.inner.seed,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl PyPolicyConfig {
    /// Wrap a configuration built in Rust.
    pub fn from_inner(inner: Mamba3PolicyConfig) -> Self {
        Self { inner }
    }

    /// The JSON a checkpoint carries. `SsmConfig` serialises itself, so a mixer
    /// knob added upstream travels with no change here.
    ///
    /// The structured keys (`obs_spec`, `entity_encoders`, `pooling`,
    /// `action_head`) are written only for a structured policy, so a flat
    /// policy's metadata is exactly what it was before they existed.
    pub fn as_json(&self) -> serde_json::Value {
        let mut value = serde_json::json!({
            "obs_dim": self.inner.obs_dim,
            "action_dim": self.inner.action_dim,
            "n_layers": self.inner.n_layers,
            "norm_eps": self.inner.norm_eps,
            "seed": self.inner.seed,
            "ssm": self.inner.ssm,
        });
        if let (Some(spec), Some(map)) = (&self.inner.obs_spec, value.as_object_mut()) {
            map.insert("obs_spec".into(), serde_json::json!(spec));
            map.insert(
                "entity_encoders".into(),
                serde_json::json!(self.inner.entity_encoders),
            );
            map.insert("pooling".into(), serde_json::json!(self.inner.pooling));
            map.insert(
                "action_head".into(),
                serde_json::json!(self.inner.action_head),
            );
        }
        value
    }

    /// The inverse of [`Self::as_json`].
    pub fn from_json(value: &serde_json::Value) -> PyResult<Self> {
        let field = |name: &str| -> PyResult<&serde_json::Value> {
            value.get(name).ok_or_else(|| {
                PyValueError::new_err(format!("the policy config has no {name:?} field"))
            })
        };
        let usize_field = |name: &str| -> PyResult<usize> {
            field(name)?.as_u64().map(|v| v as usize).ok_or_else(|| {
                PyValueError::new_err(format!("the policy config's {name:?} is not an integer"))
            })
        };
        let ssm: SsmConfig = serde_json::from_value(field("ssm")?.clone())
            .map_err(|e| PyValueError::new_err(format!("unusable mixer config: {e}")))?;
        let inner = Mamba3PolicyConfig {
            obs_dim: usize_field("obs_dim")?,
            action_dim: usize_field("action_dim")?,
            n_layers: usize_field("n_layers")?,
            ssm,
            norm_eps: value
                .get("norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5) as f32,
            seed: value.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
            // Absent in every checkpoint written before structured policies, and
            // in every flat one since: absence is the flat policy.
            obs_spec: optional!(value, "obs_spec")?,
            entity_encoders: optional!(value, "entity_encoders")?.unwrap_or_default(),
            pooling: optional!(value, "pooling")?.unwrap_or_default(),
            action_head: optional!(value, "action_head")?.unwrap_or_default(),
        };
        inner.validate().py()?;
        Ok(Self { inner })
    }
}

/// Hyperparameters of a PPO update.
///
/// The defaults are the ones PPO is usually reported with. `gae_lambda` is the
/// `lambda` of the advantage estimator, spelled out because `lambda` is a Python
/// keyword.
#[pyclass(module = "mamba3_rl", name = "PpoConfig", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyPpoConfig {
    pub(crate) inner: PpoConfig,
}

#[pymethods]
impl PyPpoConfig {
    #[new]
    #[pyo3(signature = (
        *,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_coeff = 0.2,
        value_coeff = 0.5,
        entropy_coeff = 0.01,
        clip_value_loss = true,
        normalize_advantages = true,
        reference_coeff = 0.0,
    ))]
    fn new(
        gamma: f32,
        gae_lambda: f32,
        clip_coeff: f32,
        value_coeff: f32,
        entropy_coeff: f32,
        clip_value_loss: bool,
        normalize_advantages: bool,
        reference_coeff: f32,
    ) -> PyResult<Self> {
        let inner = PpoConfig {
            gamma,
            lambda: gae_lambda,
            clip_coeff,
            value_coeff,
            entropy_coeff,
            clip_value_loss,
            normalize_advantages,
            reference_coeff,
        };
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// Discount factor.
    #[getter]
    fn gamma(&self) -> f32 {
        self.inner.gamma
    }

    /// Eligibility trace decay of the advantage estimator.
    #[getter]
    fn gae_lambda(&self) -> f32 {
        self.inner.lambda
    }

    /// Trust region half-width on the probability ratio.
    #[getter]
    fn clip_coeff(&self) -> f32 {
        self.inner.clip_coeff
    }

    /// Weight of the value loss.
    #[getter]
    fn value_coeff(&self) -> f32 {
        self.inner.value_coeff
    }

    /// Weight of the entropy bonus.
    #[getter]
    fn entropy_coeff(&self) -> f32 {
        self.inner.entropy_coeff
    }

    /// Whether the critic's own update is clipped to the same range.
    #[getter]
    fn clip_value_loss(&self) -> bool {
        self.inner.clip_value_loss
    }

    /// Whether advantages are centred and rescaled before weighting the gradient.
    #[getter]
    fn normalize_advantages(&self) -> bool {
        self.inner.normalize_advantages
    }

    /// Weight of the penalty on moving away from a fixed reference policy.
    ///
    /// The clip bounds one update; nothing in it bounds where two hundred of them
    /// end up. `0` leaves PPO as it was, and the coefficient is inert unless
    /// `PpoLearner` was given a `reference`.
    #[getter]
    fn reference_coeff(&self) -> f32 {
        self.inner.reference_coeff
    }

    fn __repr__(&self) -> String {
        format!(
            "PpoConfig(gamma={}, gae_lambda={}, clip_coeff={}, value_coeff={}, \
             entropy_coeff={}, clip_value_loss={}, normalize_advantages={}, \
             reference_coeff={})",
            self.inner.gamma,
            self.inner.lambda,
            self.inner.clip_coeff,
            self.inner.value_coeff,
            self.inner.entropy_coeff,
            if self.inner.clip_value_loss {
                "True"
            } else {
                "False"
            },
            if self.inner.normalize_advantages {
                "True"
            } else {
                "False"
            },
            self.inner.reference_coeff,
        )
    }
}

/// A learning-rate schedule, evaluated once per optimizer step — not per
/// rollout round, and not per PPO epoch or minibatch, both of which take
/// several optimizer steps from one collected window.
///
/// `PpoLearner(..., lr_schedule=None)` and `ImitationLearner(..., lr_schedule=None)`
/// (the default) mean [`PyLrSchedule::constant`], i.e. today's unscheduled
/// behaviour: the base `learning_rate` never changes.
#[pyclass(module = "mamba3_rl", name = "LrSchedule", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyLrSchedule {
    pub(crate) inner: LrSchedule,
}

#[pymethods]
impl PyLrSchedule {
    /// Hold the base rate for the whole run. The default.
    #[staticmethod]
    fn constant() -> Self {
        Self {
            inner: LrSchedule::Constant,
        }
    }

    /// Linear warmup, then cosine decay to `min_ratio` of the base rate.
    ///
    /// `warmup_steps` defaults to 2% of `total_steps` (at least one step), the
    /// same convenience the Rust `LrSchedule::cosine` constructor uses.
    #[staticmethod]
    #[pyo3(signature = (total_steps, warmup_steps = None, min_ratio = 0.1))]
    fn cosine(total_steps: u64, warmup_steps: Option<u64>, min_ratio: f32) -> PyResult<Self> {
        let inner = LrSchedule::CosineWithWarmup {
            warmup_steps: warmup_steps.unwrap_or_else(|| (total_steps / 50).max(1)),
            total_steps,
            min_ratio,
        };
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// Linear warmup, then linear decay to `min_ratio` of the base rate.
    #[staticmethod]
    #[pyo3(signature = (total_steps, warmup_steps = None, min_ratio = 0.0))]
    fn linear(total_steps: u64, warmup_steps: Option<u64>, min_ratio: f32) -> PyResult<Self> {
        let inner = LrSchedule::LinearWithWarmup {
            warmup_steps: warmup_steps.unwrap_or_else(|| (total_steps / 50).max(1)),
            total_steps,
            min_ratio,
        };
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// `base / sqrt(max(step, warmup_steps))`, the Transformer schedule.
    #[staticmethod]
    fn inverse_sqrt(warmup_steps: u64) -> PyResult<Self> {
        let inner = LrSchedule::InverseSqrt { warmup_steps };
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// Multiply by `gamma` every `every` optimizer steps.
    #[staticmethod]
    fn step(every: u64, gamma: f32) -> PyResult<Self> {
        let inner = LrSchedule::Step { every, gamma };
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// The rate this schedule gives at optimizer step `step` (one-based),
    /// given a base rate — the same evaluation `PpoLearner`/`ImitationLearner`
    /// apply internally, exposed here so a schedule can be inspected or tested
    /// without running a learner over it.
    fn rate_at(&self, base: f32, step: u64) -> f32 {
        self.inner.at(base, step)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            LrSchedule::Constant => "LrSchedule.constant()".to_string(),
            LrSchedule::CosineWithWarmup {
                warmup_steps,
                total_steps,
                min_ratio,
            } => format!(
                "LrSchedule.cosine(total_steps={total_steps}, warmup_steps={warmup_steps}, \
                 min_ratio={min_ratio})"
            ),
            LrSchedule::LinearWithWarmup {
                warmup_steps,
                total_steps,
                min_ratio,
            } => format!(
                "LrSchedule.linear(total_steps={total_steps}, warmup_steps={warmup_steps}, \
                 min_ratio={min_ratio})"
            ),
            LrSchedule::Step { every, gamma } => {
                format!("LrSchedule.step(every={every}, gamma={gamma})")
            }
            LrSchedule::InverseSqrt { warmup_steps } => {
                format!("LrSchedule.inverse_sqrt(warmup_steps={warmup_steps})")
            }
        }
    }
}

/// An exponential moving average of the policy's weights, for a learner's
/// `ema=`.
///
/// After every optimizer step the learner moves `ema_policy` towards the
/// trained weights: `ema ← ema + (1 − d)·(θ − ema)`, with `d = decay`, or with
/// `warmup="tf"` `d = min(decay, (1 + t) / (10 + t))` at optimizer step `t`.
/// The half-life is `ln 2 / −ln(decay)` optimizer steps — 69 at `0.99`, 693 at
/// `0.999` — and a PPO round takes `epochs × minibatches` of them. `decay=0`
/// makes the average the current weights; `decay=1` keeps the first ones.
#[pyclass(module = "mamba3_rl", name = "EmaConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyEmaConfig {
    pub(crate) inner: EmaConfig,
}

#[pymethods]
impl PyEmaConfig {
    #[new]
    #[pyo3(signature = (decay, warmup = "none"))]
    fn new(decay: f32, warmup: &str) -> PyResult<Self> {
        let inner = EmaConfig::new(decay).with_warmup(EmaWarmup::parse(warmup).py()?);
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// The weight the average keeps each optimizer step.
    #[getter]
    fn decay(&self) -> f32 {
        self.inner.decay
    }

    /// `"none"` or `"tf"`.
    #[getter]
    fn warmup(&self) -> &'static str {
        self.inner.warmup.name()
    }

    /// The decay applied after optimizer step `step` (one-based).
    fn decay_at(&self, step: u64) -> f32 {
        self.inner.decay_at(step)
    }

    fn __repr__(&self) -> String {
        format!(
            "EmaConfig({}, warmup={:?})",
            self.inner.decay,
            self.inner.warmup.name()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner.decay.to_bits() == other.inner.decay.to_bits()
            && self.inner.warmup == other.inner.warmup
    }
}

impl PyEmaConfig {
    /// `{"decay", "warmup"}`, as `trainer_config.ema` stores it.
    pub fn as_json(config: &EmaConfig) -> serde_json::Value {
        serde_json::json!({"decay": config.decay, "warmup": config.warmup.name()})
    }

    /// The inverse of [`PyEmaConfig::as_json`]; `null` is no average.
    pub fn from_json(value: &serde_json::Value) -> PyResult<Option<EmaConfig>> {
        if value.is_null() {
            return Ok(None);
        }
        let decay = value
            .get("decay")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| PyValueError::new_err("the checkpoint's ema.decay is not a number"))?;
        let warmup = value
            .get("warmup")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| PyValueError::new_err("the checkpoint's ema.warmup is not a string"))?;
        let config = EmaConfig::new(decay as f32).with_warmup(EmaWarmup::parse(warmup).py()?);
        config.validate().py()?;
        Ok(Some(config))
    }
}
