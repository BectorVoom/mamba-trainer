//! The two configurations a run is shaped by: the policy's architecture and PPO's
//! hyperparameters.
//!
//! Both are plain data, both validate on construction rather than at the first
//! kernel launch, and both round-trip through JSON — which is what lets a
//! checkpoint carry the architecture that produced it, so
//! [`crate::policy::Policy::load`] needs nothing but the file.

use mamba3::rl::{Mamba3PolicyConfig, PpoConfig};
use mamba3::ssm::config::{Discretization, SsmConfig, StateDynamics};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::err::IntoPyResult;

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
    ) -> PyResult<Self> {
        let mut inner = Mamba3PolicyConfig::new(obs_dim, action_dim, d_model, n_layers);
        inner.norm_eps = norm_eps;
        inner.seed = seed;
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
        format!(
            "PolicyConfig(obs_dim={}, action_dim={}, d_model={}, n_layers={}, \
             n_heads={}, head_dim={}, d_state={}, seed={})",
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
        self.inner.obs_dim == other.inner.obs_dim
            && self.inner.action_dim == other.inner.action_dim
            && self.inner.n_layers == other.inner.n_layers
            && self.inner.norm_eps == other.inner.norm_eps
            && self.inner.seed == other.inner.seed
            && self.inner.ssm == other.inner.ssm
    }
}

impl PyPolicyConfig {
    /// Wrap a configuration built in Rust.
    pub fn from_inner(inner: Mamba3PolicyConfig) -> Self {
        Self { inner }
    }

    /// The JSON a checkpoint carries. `SsmConfig` serialises itself, so a mixer
    /// knob added upstream travels with no change here.
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "obs_dim": self.inner.obs_dim,
            "action_dim": self.inner.action_dim,
            "n_layers": self.inner.n_layers,
            "norm_eps": self.inner.norm_eps,
            "seed": self.inner.seed,
            "ssm": self.inner.ssm,
        })
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
            norm_eps: field("norm_eps")?.as_f64().unwrap_or(1e-5) as f32,
            seed: field("seed")?.as_u64().unwrap_or(0),
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
            if self.inner.clip_value_loss { "True" } else { "False" },
            if self.inner.normalize_advantages { "True" } else { "False" },
            self.inner.reference_coeff,
        )
    }
}
