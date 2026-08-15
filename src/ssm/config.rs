//! Configuration for the Mamba-3 state space mixer.

use crate::error::{Error, Result};

/// How the continuous-time system is turned into a recurrence.
///
/// Mamba-2 uses the Euler (zero-order hold) rule, a first-order approximation that
/// only looks at the *right* endpoint of each interval. Mamba-3 replaces it with a
/// generalised trapezoidal rule, which is second order:
///
/// ```text
/// h_t = a_t h_{t-1} + b_t B_{t-1} x_{t-1} + g_t B_t x_t
/// a_t = exp(dt_t A_t)     b_t = (1 - l_t) dt_t exp(dt_t A_t)     g_t = l_t dt_t
/// ```
///
/// `l = 1` recovers Euler, `l = 1/2` is the classical trapezoid, and letting the
/// model predict `l` per token is the Mamba-3 default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Discretization {
    /// First-order, `lambda = 1`. Equivalent to Mamba-2.
    Euler,
    /// Second-order with a fixed `lambda = 1/2`.
    Trapezoid,
    /// Second-order with a per-token, per-head `lambda` predicted from the input.
    LearnedTrapezoid,
}

impl Discretization {
    /// The constant `lambda`, when it is not learned.
    pub fn fixed_lambda(&self) -> Option<f32> {
        match self {
            Discretization::Euler => Some(1.0),
            Discretization::Trapezoid => Some(0.5),
            Discretization::LearnedTrapezoid => None,
        }
    }

    /// Whether a `lambda` projection is needed.
    pub fn needs_lambda_projection(&self) -> bool {
        matches!(self, Discretization::LearnedTrapezoid)
    }
}

/// The eigenvalue structure of the state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StateDynamics {
    /// Real, positive decay only — a scalar `exp(dt * A)` per head, as in Mamba-2.
    /// Such a system can only forget; it cannot track periodic structure.
    Real,
    /// Decay times a rotation. The state dimension is split into `N/2` planes and
    /// each is rotated by a data-dependent angle `dt * theta`, which is exactly a
    /// complex eigenvalue `exp(dt(A + i*theta))`. This is what restores state
    /// tracking (parity, modular counting) to a linear recurrence.
    Rotational,
}

/// Single- or multi-input/output structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SsmMode {
    /// One input and one output channel per head; `B` and `C` are vectors and the
    /// state update is a rank-1 outer product.
    Siso,
    /// `rank` input and output channels per head. The state update becomes a
    /// rank-`R` matrix product, which raises arithmetic intensity at decode time
    /// without growing the state.
    Mimo {
        /// Number of input/output channels, `R` in the paper.
        rank: usize,
    },
}

impl SsmMode {
    /// The rank `R`. SISO is rank 1.
    pub fn rank(&self) -> usize {
        match self {
            SsmMode::Siso => 1,
            SsmMode::Mimo { rank } => *rank,
        }
    }
}

/// Everything that defines the state space mixer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SsmConfig {
    /// Residual stream width.
    pub d_model: usize,
    /// Number of SSM heads.
    pub n_heads: usize,
    /// Channels per head, `P`.
    pub head_dim: usize,
    /// State size per head, `N`.
    pub d_state: usize,
    /// Number of `B`/`C` groups; heads within a group share them. `n_heads` gives
    /// per-head `B`/`C`, `1` gives fully shared (multi-query style).
    pub n_groups: usize,
    /// Discretization rule.
    pub discretization: Discretization,
    /// Real or rotational state transition.
    pub dynamics: StateDynamics,
    /// SISO or MIMO.
    pub mode: SsmMode,
    /// Sequence chunk length used by the parallel scan.
    pub chunk_size: usize,
    /// Short causal convolution width applied to `x`, `B` and `C`. `None` disables
    /// it — Mamba-3 shows the convolution is optional once `B`/`C` carry biases and
    /// the trapezoidal rule is in play.
    pub conv_kernel: Option<usize>,
    /// RMS-normalise `B` and `C` (the paper's analogue of QK-norm).
    pub bc_norm: bool,
    /// Learnable per-head channel-wise biases on `B` and `C`.
    pub bc_bias: bool,
    /// Keep the Mamba-2 post-gate RMS norm. Off by default: `bc_norm` replaces it.
    pub post_gate_norm: bool,
    /// Include the direct `D * x` skip path.
    pub skip_connection: bool,
    /// Lower bound of the initial `dt` range.
    pub dt_min: f32,
    /// Upper bound of the initial `dt` range.
    pub dt_max: f32,
    /// Floor applied to the initialised `dt` bias.
    pub dt_init_floor: f32,
    /// `A` is initialised as `-exp(u)` with `u ~ log(U(a_init_min, a_init_max))`.
    pub a_init_min: f32,
    /// Upper end of the `A` initialisation range.
    pub a_init_max: f32,
    /// Bias on the input and output projections.
    pub bias: bool,
}

impl Default for SsmConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            // 24 heads x 64 channels = 1536 = 2 * d_model, the usual expansion.
            n_heads: 24,
            head_dim: 64,
            d_state: 64,
            n_groups: 24,
            discretization: Discretization::LearnedTrapezoid,
            dynamics: StateDynamics::Rotational,
            mode: SsmMode::Siso,
            chunk_size: 64,
            conv_kernel: Some(4),
            bc_norm: true,
            bc_bias: true,
            post_gate_norm: false,
            skip_connection: true,
            dt_min: 0.001,
            dt_max: 0.1,
            dt_init_floor: 1e-4,
            a_init_min: 1.0,
            a_init_max: 16.0,
            bias: false,
        }
    }
}

impl SsmConfig {
    /// Start a builder.
    pub fn builder() -> SsmConfigBuilder {
        SsmConfigBuilder::default()
    }

    /// Total inner width: `n_heads * head_dim * rank`.
    pub fn d_inner(&self) -> usize {
        self.n_heads * self.head_dim * self.mode.rank()
    }

    /// Width of the `B` (or `C`) projection: `n_groups * d_state * rank`.
    pub fn bc_width(&self) -> usize {
        self.n_groups * self.d_state * self.mode.rank()
    }

    /// Width of the `theta` projection, zero when dynamics are real.
    pub fn theta_width(&self) -> usize {
        match self.dynamics {
            StateDynamics::Real => 0,
            StateDynamics::Rotational => self.n_heads * self.d_state / 2,
        }
    }

    /// Width of the `lambda` projection, zero unless it is learned.
    pub fn lambda_width(&self) -> usize {
        if self.discretization.needs_lambda_projection() {
            self.n_heads
        } else {
            0
        }
    }

    /// Total width produced by the single fused input projection.
    pub fn in_proj_width(&self) -> usize {
        // z gate, x, B, C, dt, lambda, theta
        self.d_inner()
            + self.d_inner()
            + 2 * self.bc_width()
            + self.n_heads
            + self.lambda_width()
            + self.theta_width()
    }

    /// Check internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.n_heads == 0 || self.head_dim == 0 || self.d_state == 0 {
            return Err(Error::config("SSM dimensions must all be positive"));
        }
        if self.n_heads % self.n_groups != 0 {
            return Err(Error::config(format!(
                "n_heads ({}) must be a multiple of n_groups ({})",
                self.n_heads, self.n_groups
            )));
        }
        if self.dynamics == StateDynamics::Rotational && self.d_state % 2 != 0 {
            return Err(Error::config(format!(
                "rotational dynamics pair up state dimensions, so d_state must be \
                 even, got {}",
                self.d_state
            )));
        }
        if self.chunk_size == 0 {
            return Err(Error::config("chunk_size must be positive"));
        }
        if self.mode.rank() == 0 {
            return Err(Error::config("MIMO rank must be positive"));
        }
        if self.dt_min <= 0.0 || self.dt_max <= self.dt_min {
            return Err(Error::config(format!(
                "expected 0 < dt_min < dt_max, got {} and {}",
                self.dt_min, self.dt_max
            )));
        }
        if self.a_init_min <= 0.0 || self.a_init_max < self.a_init_min {
            return Err(Error::config(
                "expected 0 < a_init_min <= a_init_max".to_string(),
            ));
        }
        if let Some(k) = self.conv_kernel
            && k < 2
        {
            return Err(Error::config(
                "conv_kernel must be at least 2; use None to disable".to_string(),
            ));
        }
        Ok(())
    }
}

/// Builder for [`SsmConfig`].
#[derive(Debug, Clone, Default)]
pub struct SsmConfigBuilder {
    config: Option<SsmConfig>,
}

macro_rules! setter {
    ($(#[$meta:meta])* $name:ident, $ty:ty) => {
        $(#[$meta])*
        pub fn $name(mut self, value: $ty) -> Self {
            let mut config = self.config.take().unwrap_or_default();
            config.$name = value;
            self.config = Some(config);
            self
        }
    };
}

impl SsmConfigBuilder {
    setter!(
        /// Residual stream width.
        d_model, usize);
    setter!(
        /// Number of heads.
        n_heads, usize);
    setter!(
        /// Channels per head.
        head_dim, usize);
    setter!(
        /// State size per head.
        d_state, usize);
    setter!(
        /// Number of `B`/`C` groups.
        n_groups, usize);
    setter!(
        /// Discretization rule.
        discretization, Discretization);
    setter!(
        /// Real or rotational transition.
        dynamics, StateDynamics);
    setter!(
        /// SISO or MIMO.
        mode, SsmMode);
    setter!(
        /// Scan chunk length.
        chunk_size, usize);
    setter!(
        /// Short convolution width, or `None`.
        conv_kernel, Option<usize>);
    setter!(
        /// RMS-normalise `B` and `C`.
        bc_norm, bool);
    setter!(
        /// Learnable `B`/`C` biases.
        bc_bias, bool);
    setter!(
        /// Keep the Mamba-2 post-gate norm.
        post_gate_norm, bool);
    setter!(
        /// Include the `D * x` skip.
        skip_connection, bool);
    setter!(
        /// Projection biases.
        bias, bool);

    /// Set the initial `dt` sampling range.
    pub fn dt_range(mut self, min: f32, max: f32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.dt_min = min;
        config.dt_max = max;
        self.config = Some(config);
        self
    }

    /// Set the initial `A` magnitude range.
    pub fn a_init_range(mut self, min: f32, max: f32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.a_init_min = min;
        config.a_init_max = max;
        self.config = Some(config);
        self
    }

    /// Derive `n_heads` from `d_model`, an expansion factor and `head_dim`.
    pub fn expand(mut self, expansion: f32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        let inner = (config.d_model as f32 * expansion) as usize;
        let per_head = config.head_dim * config.mode.rank();
        config.n_heads = (inner / per_head.max(1)).max(1);
        config.n_groups = config.n_heads;
        self.config = Some(config);
        self
    }

    /// Validate and build.
    pub fn build(self) -> Result<SsmConfig> {
        let config = self.config.unwrap_or_default();
        config.validate()?;
        Ok(config)
    }
}
