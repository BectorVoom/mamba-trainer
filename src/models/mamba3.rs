//! The Mamba-3 token mixer and residual block.
//!
//! One fused input projection produces everything the layer needs:
//!
//! ```text
//! u -> in_proj -> [ z | x | B | C | dt | lambda | theta ]
//! ```
//!
//! `x`, `B` and `C` optionally pass through a short causal convolution; `B` and `C`
//! then get per-head channel biases and an RMS normalisation (the paper's analogue
//! of QK-norm, and the reason the post-gate norm can be dropped). `dt` becomes
//! positive through a softplus, `lambda` is squashed into `[0, 1]`, and `theta`
//! feeds the rotational state transition. The scan runs, the output is gated by
//! `silu(z)`, and one more projection returns to the residual stream.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::conv::{CausalConv1d, CausalConv1dConfig};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::lora::LoraConfig;
use crate::nn::module::{Layer, Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::nn::quant::QuantConfig;
use crate::nn::init::Initializer;
use crate::ssm::config::{SsmConfig, StateDynamics};
use crate::ssm::scan::{ScanInputs, SsmState, mamba3_scan, mamba3_step};
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

/// Per-layer state carried between decoding steps.
pub struct MixerCache<R: Runtime, E: FloatElem> {
    /// Recurrent SSM state.
    pub ssm: SsmState<R, E>,
    /// The last `kernel_size - 1` convolution inputs, if the mixer has a convolution.
    pub conv: Option<Var<R, E>>,
}

impl<R: Runtime, E: FloatElem> MixerCache<R, E> {
    /// Drop every tape link, so a cached state cannot keep a graph alive across
    /// decoding steps.
    pub fn detach(&self) -> Self {
        Self {
            ssm: self.ssm.detach(),
            conv: self.conv.as_ref().map(|c| c.detach()),
        }
    }

    /// Elements held on the device by this cache.
    pub fn num_elements(&self) -> usize {
        self.ssm.num_elements()
            + self
                .conv
                .as_ref()
                .map(|c| c.shape().num_elements())
                .unwrap_or(0)
    }
}

impl<R: Runtime, E: FloatElem> Clone for MixerCache<R, E> {
    fn clone(&self) -> Self {
        Self {
            ssm: self.ssm.clone(),
            conv: self.conv.clone(),
        }
    }
}

/// Builder-friendly configuration for [`Mamba3Mixer`].
#[derive(Debug, Clone)]
pub struct Mamba3MixerConfig {
    ssm: SsmConfig,
    lora: Option<LoraConfig>,
    weight_quant: Option<QuantConfig>,
    activation_quant: Option<QuantConfig>,
    /// Residual-branch rescaling for the output projection.
    n_layers: usize,
}

impl Mamba3MixerConfig {
    /// Wrap an [`SsmConfig`].
    pub fn new(ssm: SsmConfig) -> Self {
        Self {
            ssm,
            lora: None,
            weight_quant: None,
            activation_quant: None,
            n_layers: 1,
        }
    }

    /// Attach LoRA adapters to the input and output projections.
    pub fn with_lora(mut self, lora: LoraConfig) -> Self {
        self.lora = Some(lora);
        self
    }

    /// Fake-quantize projection weights.
    pub fn with_weight_quant(mut self, config: QuantConfig) -> Self {
        self.weight_quant = Some(config);
        self
    }

    /// Fake-quantize projection inputs.
    pub fn with_activation_quant(mut self, config: QuantConfig) -> Self {
        self.activation_quant = Some(config);
        self
    }

    /// Number of blocks in the surrounding stack; used to scale the output
    /// projection's initialisation.
    pub fn with_depth(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// The underlying SSM configuration.
    pub fn ssm(&self) -> &SsmConfig {
        &self.ssm
    }

    fn linear(&self, d_in: usize, d_out: usize, init: Initializer) -> LinearConfig {
        let mut cfg = LinearConfig::new(d_in, d_out)
            .with_bias(self.ssm.bias)
            .with_initializer(init);
        if let Some(lora) = self.lora {
            cfg = cfg.with_lora(lora);
        }
        if let Some(q) = self.weight_quant {
            cfg = cfg.with_weight_quant(q);
        }
        if let Some(q) = self.activation_quant {
            cfg = cfg.with_activation_quant(q);
        }
        cfg
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Mamba3Mixer<R, E>> {
        let cfg = &self.ssm;
        cfg.validate()?;

        let heads = cfg.n_heads;
        let state = cfg.d_state;

        // dt_bias is set so that softplus(dt_bias) is log-uniform over
        // [dt_min, dt_max], the standard Mamba initialisation.
        let dt_bias: Vec<f32> = rng
            .uniform_vec(heads, cfg.dt_min.ln(), cfg.dt_max.ln())
            .into_iter()
            .map(|log_dt| {
                let dt = log_dt.exp().max(cfg.dt_init_floor);
                // inverse softplus
                dt + (-(-dt).exp_m1()).ln()
            })
            .collect();

        // A = -exp(a_log) with a_log ~ log(U(a_min, a_max)).
        let a_log: Vec<f32> = rng
            .uniform_vec(heads, cfg.a_init_min, cfg.a_init_max)
            .into_iter()
            .map(|v| v.max(1e-4).ln())
            .collect();

        let conv_channels = cfg.d_inner() + 2 * cfg.bc_width();

        Ok(Mamba3Mixer {
            in_proj: self
                .linear(
                    cfg.d_model,
                    cfg.in_proj_width(),
                    Initializer::Normal {
                        mean: 0.0,
                        std: 0.02,
                    },
                )
                .init(device, rng),
            out_proj: self
                .linear(
                    cfg.d_inner(),
                    cfg.d_model,
                    Initializer::ResidualScaled {
                        std: 0.02,
                        n_layers: self.n_layers,
                    },
                )
                .init(device, rng),
            conv: cfg
                .conv_kernel
                .map(|k| CausalConv1dConfig::new(conv_channels, k).init(device, rng)),
            dt_bias: Param::new(Tensor::from_f32(&dt_bias, vec![heads], device)?),
            a_log: Param::new(Tensor::from_f32(&a_log, vec![heads], device)?),
            d_skip: cfg
                .skip_connection
                .then(|| Param::new(Tensor::ones(vec![heads], device))),
            b_bias: cfg
                .bc_bias
                .then(|| Param::new(Tensor::zeros(vec![heads, state], device))),
            c_bias: cfg
                .bc_bias
                .then(|| Param::new(Tensor::zeros(vec![heads, state], device))),
            bc_norm: cfg
                .bc_norm
                .then(|| RmsNormConfig::new(state).init(device, rng)),
            post_gate_norm: cfg
                .post_gate_norm
                .then(|| RmsNormConfig::new(cfg.d_inner()).init(device, rng)),
            config: cfg.clone(),
        })
    }
}

/// The Mamba-3 token mixer.
pub struct Mamba3Mixer<R: Runtime, E: FloatElem> {
    in_proj: Linear<R, E>,
    out_proj: Linear<R, E>,
    conv: Option<CausalConv1d<R, E>>,
    dt_bias: Param<R, E>,
    a_log: Param<R, E>,
    d_skip: Option<Param<R, E>>,
    b_bias: Option<Param<R, E>>,
    c_bias: Option<Param<R, E>>,
    bc_norm: Option<RmsNorm<R, E>>,
    post_gate_norm: Option<RmsNorm<R, E>>,
    config: SsmConfig,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Mamba3Mixer<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mamba3Mixer(d_model={}, heads={}, head_dim={}, state={}, {:?}, {:?}, {:?})",
            self.config.d_model,
            self.config.n_heads,
            self.config.head_dim,
            self.config.d_state,
            self.config.discretization,
            self.config.dynamics,
            self.config.mode,
        )
    }
}

/// The pieces the fused projection produces, already shaped for the scan.
struct Projected<R: Runtime, E: FloatElem> {
    z: Var<R, E>,
    x: Var<R, E>,
    b: Var<R, E>,
    c: Var<R, E>,
    dt: Var<R, E>,
    lambda: Var<R, E>,
    theta: Option<Var<R, E>>,
}

impl<R: Runtime, E: FloatElem> Mamba3Mixer<R, E> {
    /// The configuration this mixer was built from.
    pub fn config(&self) -> &SsmConfig {
        &self.config
    }

    /// A zeroed cache for `batch` sequences.
    pub fn empty_cache(&self, batch: usize, device: &Device<R>) -> MixerCache<R, E> {
        MixerCache {
            ssm: SsmState::zeros(
                batch,
                self.config.n_heads,
                self.config.head_dim,
                self.config.d_state,
                self.config.dynamics == StateDynamics::Rotational,
                device,
            ),
            conv: self.conv.as_ref().map(|c| c.empty_history(batch, device)),
        }
    }

    /// Run the fused projection and shape every piece.
    ///
    /// `conv_history` is only used on the incremental path; when it is `Some`, the
    /// updated history is written back through it.
    fn project(
        &self,
        input: &Var<R, E>,
        conv_history: Option<&mut Option<Var<R, E>>>,
    ) -> Result<Projected<R, E>> {
        let cfg = &self.config;
        let dims = input.dims().to_vec();
        let (batch, seq) = (dims[0], dims[1]);
        let (heads, head_dim, state, rank) = (
            cfg.n_heads,
            cfg.head_dim,
            cfg.d_state,
            cfg.mode.rank(),
        );
        let groups = cfg.n_groups;
        let per_group = heads / groups;

        let projected = self.in_proj.apply(input)?;

        let d_inner = cfg.d_inner();
        let bc = cfg.bc_width();

        // One split rather than a run of slices. The pieces tile the axis, so the
        // whole projection's gradient is their concatenation — one buffer written
        // once in bands — where independent slices would each produce a full-size
        // buffer that is zero outside its own band, to be added together afterwards.
        //
        // `x`, `B` and `C` stay together here: they are adjacent in the projection
        // and the convolution runs over all three at once, so cutting them apart and
        // concatenating them back would be launches that cancel out.
        let mut widths = vec![d_inner, d_inner + 2 * bc, heads];
        if cfg.lambda_width() > 0 {
            widths.push(cfg.lambda_width());
        }
        if cfg.theta_width() > 0 {
            widths.push(cfg.theta_width());
        }
        let mut pieces = projected.split(&widths, 2)?.into_iter();
        let z = pieces.next().expect("z piece");
        let xbc_raw = pieces.next().expect("xbc piece");
        let dt_raw = pieces.next().expect("dt piece");
        let lambda_raw = (cfg.lambda_width() > 0).then(|| pieces.next().expect("lambda piece"));
        let theta_raw = (cfg.theta_width() > 0).then(|| pieces.next().expect("theta piece"));

        // Short causal convolution over x, B and C together, then the activation.
        let mut xbc = xbc_raw;
        if let Some(conv) = &self.conv {
            match conv_history {
                Some(slot) => {
                    let history = slot.take().unwrap_or_else(|| {
                        conv.empty_history(batch, input.device())
                    });
                    let (out, updated) = conv.apply_with_history(&xbc, &history)?;
                    *slot = Some(updated);
                    xbc = out;
                }
                None => xbc = conv.apply(&xbc)?,
            }
        }
        let xbc = xbc.silu()?;

        let mut parts = xbc.split(&[d_inner, bc, bc], 2)?.into_iter();
        let x = parts.next().expect("x piece");
        let b_flat = parts.next().expect("B piece");
        let c_flat = parts.next().expect("C piece");

        // [b, t, groups, rank, state] -> [b, t, heads, rank, state]
        let to_heads = |v: &Var<R, E>| -> Result<Var<R, E>> {
            let grouped = v.reshape(vec![batch, seq, groups, rank, state])?;
            if per_group == 1 {
                return grouped.reshape(vec![batch, seq, heads, rank, state]);
            }
            grouped
                .reshape(vec![batch, seq, groups, 1, rank, state])?
                .expand(vec![batch, seq, groups, per_group, rank, state])?
                .reshape(vec![batch, seq, heads, rank, state])
        };

        let mut b = to_heads(&b_flat)?;
        let mut c = to_heads(&c_flat)?;

        // Head-specific, channel-wise biases, then the QK-norm analogue.
        if let Some(bias) = &self.b_bias {
            b = b.add(&bias.var(input).reshape(vec![1, 1, heads, 1, state])?)?;
        }
        if let Some(bias) = &self.c_bias {
            c = c.add(&bias.var(input).reshape(vec![1, 1, heads, 1, state])?)?;
        }
        if let Some(norm) = &self.bc_norm {
            b = norm.apply(&b)?;
            c = norm.apply(&c)?;
        }
        // The scan wants the rank axis last.
        let b = b.permute(&[0, 1, 2, 4, 3])?;
        let c = c.permute(&[0, 1, 2, 4, 3])?;

        let x = x.reshape(vec![batch, seq, heads, head_dim, rank])?;

        let dt = dt_raw
            .add(&self.dt_bias.var(input).reshape(vec![1, 1, heads])?)?
            .softplus()?;

        let lambda = match (&lambda_raw, cfg.discretization.fixed_lambda()) {
            (Some(raw), _) => raw.sigmoid(),
            (None, Some(value)) => Var::constant(Tensor::full(
                vec![batch, seq, heads],
                value,
                input.device(),
            )),
            (None, None) => {
                return Err(Error::config(
                    "learned trapezoidal discretization needs a lambda projection"
                        .to_string(),
                ));
            }
        };

        let theta = theta_raw
            .map(|raw| raw.reshape(vec![batch, seq, heads, state / 2]))
            .transpose()?;

        Ok(Projected {
            z,
            x,
            b,
            c,
            dt,
            lambda,
            theta,
        })
    }

    /// Combine the scan output with the gate and project back.
    fn finish(&self, y: &Var<R, E>, z: &Var<R, E>) -> Result<Var<R, E>> {
        let dims = y.dims().to_vec();
        let flat = y.reshape(vec![dims[0], dims[1], self.config.d_inner()])?;
        let gated = match &self.post_gate_norm {
            Some(norm) => norm.apply(&flat)?,
            None => flat,
        };
        let gated = gated.mul(&z.silu()?)?;
        self.out_proj.apply(&gated)
    }

    /// Apply the mixer to `[batch, seq, d_model]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        Ok(self.apply_with_state(input, None)?.0)
    }

    /// Apply the mixer over a window, optionally continuing from a cache.
    ///
    /// Returns the output and the state to carry forward. Training uses
    /// `state = None`; long-context and streaming inference pass the previous
    /// window's state.
    pub fn apply_with_state(
        &self,
        input: &Var<R, E>,
        cache: Option<&MixerCache<R, E>>,
    ) -> Result<(Var<R, E>, MixerCache<R, E>)> {
        input.shape().expect_rank(3)?;
        if input.shape().dim(2) != self.config.d_model {
            return Err(Error::shape(format!(
                "Mamba3Mixer expects d_model={}, got {}",
                self.config.d_model,
                input.shape()
            )));
        }

        let mut conv_slot = cache.map(|c| c.conv.clone());
        let projected = self.project(input, conv_slot.as_mut())?;

        let a_log = self.a_log.var(input);
        let d_skip = self.d_skip.as_ref().map(|d| d.var(input));

        let mut scan = ScanInputs::new(
            &projected.x,
            &projected.b,
            &projected.c,
            &projected.dt,
            &a_log,
            &projected.lambda,
        )
        .with_chunk_size(self.config.chunk_size);
        if let Some(theta) = &projected.theta {
            scan = scan.with_theta(theta);
        }
        if let Some(d) = &d_skip {
            scan = scan.with_skip(d);
        }
        if let Some(c) = cache {
            scan = scan.with_state(&c.ssm);
        }

        let out = mamba3_scan(scan)?;
        let y = self.finish(&out.y, &projected.z)?;
        Ok((
            y,
            MixerCache {
                ssm: out.state,
                conv: conv_slot.flatten(),
            },
        ))
    }

    /// One decoding step for `[batch, 1, d_model]`.
    ///
    /// Uses the direct recurrence rather than the chunked scan, so the cost is
    /// independent of how much context has already been consumed.
    pub fn step(
        &self,
        input: &Var<R, E>,
        cache: &MixerCache<R, E>,
    ) -> Result<(Var<R, E>, MixerCache<R, E>)> {
        input.shape().expect_rank(3)?;
        if input.shape().dim(1) != 1 {
            return Err(Error::shape(
                "step() expects a single position; use apply_with_state for a window"
                    .to_string(),
            ));
        }
        let cfg = &self.config;
        let batch = input.shape().dim(0);
        let (heads, head_dim, state, rank) =
            (cfg.n_heads, cfg.head_dim, cfg.d_state, cfg.mode.rank());

        let mut conv_slot = Some(cache.conv.clone());
        let projected = self.project(input, conv_slot.as_mut())?;

        let squeeze = |v: &Var<R, E>, trailing: Vec<usize>| -> Result<Var<R, E>> {
            let mut dims = vec![batch, heads];
            dims.extend(trailing);
            v.reshape(dims)
        };

        let a_log = self.a_log.var(input);
        let d_skip = self.d_skip.as_ref().map(|d| d.var(input));
        let theta = projected
            .theta
            .as_ref()
            .map(|t| squeeze(t, vec![state / 2]))
            .transpose()?;

        let (y, ssm) = mamba3_step(
            &squeeze(&projected.x, vec![head_dim, rank])?,
            &squeeze(&projected.b, vec![state, rank])?,
            &squeeze(&projected.c, vec![state, rank])?,
            &squeeze(&projected.dt, vec![])?,
            &squeeze(&projected.lambda, vec![])?,
            &a_log,
            theta.as_ref(),
            d_skip.as_ref(),
            &cache.ssm,
        )?;

        let y = y.reshape(vec![batch, 1, heads, head_dim, rank])?;
        let out = self.finish(&y, &projected.z)?;
        Ok((
            out,
            MixerCache {
                ssm,
                conv: conv_slot.flatten(),
            },
        ))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Mamba3Mixer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("in_proj", &self.in_proj);
        visitor.child("out_proj", &self.out_proj);
        if let Some(conv) = &self.conv {
            visitor.child("conv", conv);
        }
        visitor.param("dt_bias", &self.dt_bias);
        visitor.param("a_log", &self.a_log);
        if let Some(d) = &self.d_skip {
            visitor.param("d", d);
        }
        if let Some(b) = &self.b_bias {
            visitor.param("b_bias", b);
        }
        if let Some(c) = &self.c_bias {
            visitor.param("c_bias", c);
        }
        if let Some(n) = &self.bc_norm {
            visitor.child("bc_norm", n);
        }
        if let Some(n) = &self.post_gate_norm {
            visitor.child("post_gate_norm", n);
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for Mamba3Mixer<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}

/// Configuration for [`Mamba3Block`].
#[derive(Debug, Clone)]
pub struct Mamba3BlockConfig {
    mixer: Mamba3MixerConfig,
    norm_eps: f32,
    residual: bool,
}

impl Mamba3BlockConfig {
    /// A pre-norm residual block around a Mamba-3 mixer.
    pub fn new(ssm: SsmConfig) -> Self {
        Self {
            mixer: Mamba3MixerConfig::new(ssm),
            norm_eps: 1e-5,
            residual: true,
        }
    }

    /// Access the mixer configuration for further customisation.
    pub fn mixer(mut self, f: impl FnOnce(Mamba3MixerConfig) -> Mamba3MixerConfig) -> Self {
        self.mixer = f(self.mixer);
        self
    }

    /// Normalisation epsilon.
    pub fn with_norm_eps(mut self, eps: f32) -> Self {
        self.norm_eps = eps;
        self
    }

    /// Disable the residual connection (used when an outer stack owns it).
    pub fn with_residual(mut self, residual: bool) -> Self {
        self.residual = residual;
        self
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Mamba3Block<R, E>> {
        Ok(Mamba3Block {
            norm: RmsNormConfig::new(self.mixer.ssm().d_model)
                .with_eps(self.norm_eps)
                .init(device, rng),
            mixer: self.mixer.init(device, rng)?,
            residual: self.residual,
        })
    }
}

/// `x + mixer(norm(x))`.
pub struct Mamba3Block<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    mixer: Mamba3Mixer<R, E>,
    residual: bool,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Mamba3Block<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mamba3Block({:?})", self.mixer)
    }
}

impl<R: Runtime, E: FloatElem> Mamba3Block<R, E> {
    /// The mixer inside.
    pub fn mixer(&self) -> &Mamba3Mixer<R, E> {
        &self.mixer
    }

    /// Apply the block.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        let out = self.mixer.apply(&self.norm.apply(input)?)?;
        if self.residual { input.add(&out) } else { Ok(out) }
    }

    /// Apply over a window, carrying state.
    pub fn apply_with_state(
        &self,
        input: &Var<R, E>,
        cache: Option<&MixerCache<R, E>>,
    ) -> Result<(Var<R, E>, MixerCache<R, E>)> {
        let (out, state) = self
            .mixer
            .apply_with_state(&self.norm.apply(input)?, cache)?;
        let out = if self.residual { input.add(&out)? } else { out };
        Ok((out, state))
    }

    /// One decoding step.
    pub fn step(
        &self,
        input: &Var<R, E>,
        cache: &MixerCache<R, E>,
    ) -> Result<(Var<R, E>, MixerCache<R, E>)> {
        let (out, state) = self.mixer.step(&self.norm.apply(input)?, cache)?;
        let out = if self.residual { input.add(&out)? } else { out };
        Ok((out, state))
    }

    /// A zeroed cache.
    pub fn empty_cache(&self, batch: usize, device: &Device<R>) -> MixerCache<R, E> {
        self.mixer.empty_cache(batch, device)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Mamba3Block<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm", &self.norm);
        visitor.child("mixer", &self.mixer);
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for Mamba3Block<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}
