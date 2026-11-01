//! Mamba-1 time-series encoder (selective SSM blocks, causal throughout).
//!
//! Block (pre-norm, residual), as in Gu & Dao (2023):
//!
//! ```text
//! u, z  = split(in_proj(rmsnorm(x)))          [B, T, 2*Di]
//! u     = silu(causal_depthwise_conv(u))
//! dtr, Bm, Cm = split(x_proj(u))              dt_rank, N, N
//! dt    = softplus(dt_proj(dtr) + dt_bias)    [B, T, Di]
//! y     = selective_scan(u, dt, -exp(A_log), Bm, Cm, D)
//! out   = x + out_proj(y * silu(z))
//! ```
//!
//! The scan is the CubeCL kernel `Var::selective_scan` (forward and backward
//! kernels; see `src/tensor/ops/selective_scan.rs` in the core crate).

use mamba3::autograd::Var;
use mamba3::backend::{Device, FloatElem};
use mamba3::cubecl::prelude::Runtime;
use mamba3::error::Result;
use mamba3::nn::{
    CausalConv1d, CausalConv1dConfig, Initializer, Linear, LinearConfig, Module, ModuleVisitor,
    Param, RmsNorm, RmsNormConfig,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::tensor::ops::selective_scan::SelectiveScanMode;
use serde::{Deserialize, Serialize};

/// How a sequence of encoder states becomes one state vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Pooling {
    /// The last time step (the decision day). Default.
    #[default]
    Last,
    /// Mean over the window.
    Mean,
}

/// Mamba-1 encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MambaConfig {
    /// Input features per time step.
    pub input_dim: usize,
    /// Residual width.
    pub d_model: usize,
    /// State columns per channel (`N`).
    pub d_state: usize,
    /// Causal convolution taps.
    pub d_conv: usize,
    /// Inner width = `expand * d_model`.
    pub expand: usize,
    /// Blocks.
    pub num_layers: usize,
    /// Rank of the `dt` projection; 0 means `ceil(d_model / 16)`.
    pub dt_rank: usize,
    /// Initial `dt` range (log-uniform).
    pub dt_min: f32,
    /// Initial `dt` range (log-uniform).
    pub dt_max: f32,
    /// Pooling of the final states.
    pub pooling: Pooling,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            d_model: 128,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            num_layers: 4,
            dt_rank: 0,
            dt_min: 1e-3,
            dt_max: 1e-1,
            pooling: Pooling::Last,
        }
    }
}

impl MambaConfig {
    /// Inner width.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// Effective `dt` rank.
    pub fn rank(&self) -> usize {
        if self.dt_rank == 0 {
            self.d_model.div_ceil(16)
        } else {
            self.dt_rank
        }
    }
}

/// One Mamba-1 block.
pub struct MambaBlock<R: Runtime, E: FloatElem> {
    pub(crate) norm: RmsNorm<R, E>,
    pub(crate) in_proj: Linear<R, E>,
    pub(crate) conv: CausalConv1d<R, E>,
    pub(crate) x_proj: Linear<R, E>,
    pub(crate) dt_proj: Linear<R, E>,
    pub(crate) dt_bias: Param<R, E>,
    pub(crate) a_log: Param<R, E>,
    pub(crate) d_skip: Param<R, E>,
    pub(crate) out_proj: Linear<R, E>,
    pub(crate) d_inner: usize,
    pub(crate) d_state: usize,
    pub(crate) rank: usize,
}

/// `log(exp(y) - 1)`, the inverse of softplus.
fn inv_softplus(y: f32) -> f32 {
    (y as f64).exp_m1().ln() as f32
}

impl<R: Runtime, E: FloatElem> MambaBlock<R, E> {
    /// A freshly initialised block.
    pub fn new(cfg: &MambaConfig, device: &Device<R>, rng: &mut Rng) -> Self {
        let (dm, di, n, rank) = (cfg.d_model, cfg.d_inner(), cfg.d_state, cfg.rank());
        let in_proj = LinearConfig::new(dm, 2 * di).with_bias(false).init(device, rng);
        let conv = CausalConv1dConfig::new(di, cfg.d_conv).init(device, rng);
        let x_proj = LinearConfig::new(di, rank + 2 * n).with_bias(false).init(device, rng);
        let bound = (rank as f32).powf(-0.5);
        let dt_proj = LinearConfig::new(rank, di)
            .with_bias(false)
            .with_initializer(Initializer::Uniform {
                lo: -bound,
                hi: bound,
            })
            .init(device, rng);
        // dt = exp(U(log dt_min, log dt_max)) at init, stored through softplus.
        let (lo, hi) = (cfg.dt_min.ln(), cfg.dt_max.ln());
        let bias: Vec<f32> = (0..di)
            .map(|_| {
                let u = rng.next_f32();
                inv_softplus((lo + (hi - lo) * u).exp().max(1e-4))
            })
            .collect();
        let dt_bias = Param::new(Tensor::<R, E>::from_f32(&bias, vec![di], device).expect("dt bias"));
        // S4D-real initialisation: A[d, n] = -(n + 1).
        let a_log: Vec<f32> = (0..di)
            .flat_map(|_| (0..n).map(|k| ((k + 1) as f32).ln()))
            .collect();
        let a_log = Param::new(Tensor::<R, E>::from_f32(&a_log, vec![di, n], device).expect("A_log"));
        let d_skip = Param::new(Tensor::ones(vec![di], device));
        let out_proj = LinearConfig::new(di, dm).with_bias(false).init(device, rng);
        Self {
            norm: RmsNormConfig::new(dm).init(device, rng),
            in_proj,
            conv,
            x_proj,
            dt_proj,
            dt_bias,
            a_log,
            d_skip,
            out_proj,
            d_inner: di,
            d_state: n,
            rank,
        }
    }

    /// `[B, T, d_model] -> [B, T, d_model]`.
    pub fn apply(&self, x: &Var<R, E>, mode: SelectiveScanMode) -> Result<Var<R, E>> {
        let xn = self.norm.apply(x)?;
        let uz = self.in_proj.apply(&xn)?.split(&[self.d_inner, self.d_inner], 2)?;
        let u = self.conv.apply(&uz[0])?.silu()?;
        let parts = self
            .x_proj
            .apply(&u)?
            .split(&[self.rank, self.d_state, self.d_state], 2)?;
        let dt = self
            .dt_proj
            .apply(&parts[0])?
            .bias_softplus(&self.dt_bias.var(x))?;
        let a = self.a_log.var(x).exp().neg();
        let y = Var::selective_scan(
            &u,
            &dt,
            &a,
            &parts[1],
            &parts[2],
            Some(&self.d_skip.var(x)),
            mode,
        )?;
        let gated = y.mul(&uz[1].silu()?)?;
        x.add(&self.out_proj.apply(&gated)?)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for MambaBlock<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        v.child("norm", &self.norm);
        v.child("in_proj", &self.in_proj);
        v.child("conv", &self.conv);
        v.child("x_proj", &self.x_proj);
        v.child("dt_proj", &self.dt_proj);
        v.param("dt_bias", &self.dt_bias);
        v.param("A_log", &self.a_log);
        v.param("D", &self.d_skip);
        v.child("out_proj", &self.out_proj);
    }
}

/// Input projection, Mamba-1 blocks, final norm, pooling.
pub struct MambaEncoder<R: Runtime, E: FloatElem> {
    pub(crate) input: Linear<R, E>,
    pub(crate) blocks: Vec<MambaBlock<R, E>>,
    pub(crate) norm: RmsNorm<R, E>,
    pub(crate) cfg: MambaConfig,
    /// Forward schedule of the scans.
    pub scan_mode: SelectiveScanMode,
}

impl<R: Runtime, E: FloatElem> MambaEncoder<R, E> {
    /// A freshly initialised encoder.
    pub fn new(cfg: &MambaConfig, device: &Device<R>, rng: &mut Rng) -> Self {
        Self {
            input: LinearConfig::new(cfg.input_dim, cfg.d_model).init(device, rng),
            blocks: (0..cfg.num_layers)
                .map(|_| MambaBlock::new(cfg, device, rng))
                .collect(),
            norm: RmsNormConfig::new(cfg.d_model).init(device, rng),
            cfg: cfg.clone(),
            scan_mode: SelectiveScanMode::Auto,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &MambaConfig {
        &self.cfg
    }

    /// Output width.
    pub fn out_dim(&self) -> usize {
        self.cfg.d_model
    }

    /// Every time step's state, `[B, T, F] -> [B, T, d_model]`.
    pub fn sequence(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        let mut h = self.input.apply(x)?;
        for block in &self.blocks {
            h = block.apply(&h, self.scan_mode)?;
        }
        self.norm.apply(&h)
    }

    /// The pooled state, `[B, T, F] -> [B, d_model]`.
    pub fn apply(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        let h = self.sequence(x)?;
        pool(&h, self.cfg.pooling)
    }
}

/// Pool `[B, T, D]` to `[B, D]`.
pub fn pool<R: Runtime, E: FloatElem>(h: &Var<R, E>, pooling: Pooling) -> Result<Var<R, E>> {
    let t = h.shape().dim(1);
    match pooling {
        Pooling::Last => h.slice(1, t - 1, 1)?.squeeze(1),
        Pooling::Mean => h.mean_dim(1)?.squeeze(1),
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for MambaEncoder<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        v.child("input", &self.input);
        for (i, b) in self.blocks.iter().enumerate() {
            v.child_at("blocks", i, b);
        }
        v.child("norm", &self.norm);
    }
}
