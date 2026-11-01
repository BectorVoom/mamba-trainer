//! Decision heads: the contrastive state/action pair (CLM), a supervised MLP
//! classifier, a linear classifier, and the auxiliary return/risk heads.

use std::cell::RefCell;

use mamba3::autograd::Var;
use mamba3::backend::{Device, FloatElem};
use mamba3::cubecl::prelude::Runtime;
use mamba3::error::Result;
use mamba3::nn::{Linear, LinearConfig, Module, ModuleVisitor, Param};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;
use serde::{Deserialize, Serialize};

/// `x / max(|x|, eps)` along the last axis of `[B, E]`.
pub fn l2_normalize<R: Runtime, E: FloatElem>(x: &Var<R, E>) -> Result<Var<R, E>> {
    let sq = x.mul(x)?.sum_dim(1)?;
    x.mul(&sq.add_scalar(1e-12).rsqrt())
}

/// `Linear -> SiLU -> Linear`.
pub struct TwoLayer<R: Runtime, E: FloatElem> {
    a: Linear<R, E>,
    b: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> TwoLayer<R, E> {
    /// `in -> hidden -> out`.
    pub fn new(input: usize, hidden: usize, out: usize, device: &Device<R>, rng: &mut Rng) -> Self {
        Self {
            a: LinearConfig::new(input, hidden).init(device, rng),
            b: LinearConfig::new(hidden, out).init(device, rng),
        }
    }

    /// Apply.
    pub fn apply(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        self.b.apply(&self.a.apply(x)?.silu()?)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for TwoLayer<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        v.child("a", &self.a);
        v.child("b", &self.b);
    }
}

/// Temperature of the contrastive scores.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Temperature {
    /// `score = s.a / tau` with `tau` fixed.
    Fixed(f32),
    /// `tau` learned (as `log(1/tau)`, clamped to `1/tau` in `[1, 100]`), starting here.
    Learned(f32),
}

/// How a state embedding and an action embedding are scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scoring {
    /// Cosine similarity (both embeddings are unit vectors), over the temperature.
    Cosine,
    /// `s^T W a` over the temperature, `W` initialised to the identity.
    Bilinear,
}

/// Contrastive head configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    /// Embedding width shared by states and actions.
    pub embed_dim: usize,
    /// Hidden width of the state projection and the action encoder.
    pub hidden: usize,
    /// Temperature.
    pub temperature: Temperature,
    /// Scoring.
    pub scoring: Scoring,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            embed_dim: 128,
            hidden: 128,
            temperature: Temperature::Learned(0.1),
            scoring: Scoring::Cosine,
        }
    }
}

/// Describes each candidate action to the action encoder: `[NUM_ACTIONS, ATTR_DIM]`
/// rows of `[one-hot type (3), horizon / 20, cost floor of the threshold (2 kappa)
/// in %, one-way cost in 10 bp]`. The volatility part of the label threshold
/// (`m * sigma_h`) varies per sample and is not an action attribute; with fixed
/// horizon and cost the three rows differ only in the type.
pub const ATTR_DIM: usize = 6;

/// The attribute rows for the three actions of one decision problem.
pub fn action_attributes(horizon: usize, threshold: f32, kappa: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * ATTR_DIM);
    for a in 0..3 {
        for k in 0..3 {
            out.push(if k == a { 1.0 } else { 0.0 });
        }
        out.push(horizon as f32 / 20.0);
        out.push(threshold * 100.0);
        out.push(kappa * 1000.0);
    }
    out
}

/// The CLM head: a state projection and an independent action encoder whose
/// unit-norm embeddings are scored against each other.
pub struct ContrastiveHead<R: Runtime, E: FloatElem> {
    state_proj: TwoLayer<R, E>,
    action_enc: TwoLayer<R, E>,
    attrs: Tensor<R, E>,
    log_scale: Param<R, E>,
    bilinear: Option<Param<R, E>>,
    cfg: ContrastiveConfig,
    /// Action embeddings cached for inference; cleared by [`Self::invalidate`].
    cache: RefCell<Option<Tensor<R, E>>>,
}

impl<R: Runtime, E: FloatElem> ContrastiveHead<R, E> {
    /// A head over `input`-wide encoder states; `attrs` from [`action_attributes`].
    pub fn new(
        input: usize,
        cfg: &ContrastiveConfig,
        attrs: &[f32],
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Self {
        let tau = match cfg.temperature {
            Temperature::Fixed(t) | Temperature::Learned(t) => t,
        };
        let log_scale = Param::new(Tensor::<R, E>::from_f32(&[(1.0 / tau).ln()], vec![1], device).expect("scale"));
        if matches!(cfg.temperature, Temperature::Fixed(_)) {
            log_scale.freeze();
        }
        let e = cfg.embed_dim;
        let bilinear = (cfg.scoring == Scoring::Bilinear).then(|| {
            let eye: Vec<f32> = (0..e * e).map(|i| if i / e == i % e { 1.0 } else { 0.0 }).collect();
            Param::new(Tensor::<R, E>::from_f32(&eye, vec![e, e], device).expect("bilinear"))
        });
        Self {
            state_proj: TwoLayer::new(input, cfg.hidden, e, device, rng),
            action_enc: TwoLayer::new(ATTR_DIM, cfg.hidden, e, device, rng),
            attrs: Tensor::<R, E>::from_f32(attrs, vec![attrs.len() / ATTR_DIM, ATTR_DIM], device)
                .expect("attrs"),
            log_scale,
            bilinear,
            cfg: cfg.clone(),
            cache: RefCell::new(None),
        }
    }

    /// Drop the cached action embeddings (call after any weight change).
    pub fn invalidate(&self) {
        self.cache.borrow_mut().take();
    }

    /// Unit-norm state embedding `[B, E]`.
    pub fn state_embedding(&self, h: &Var<R, E>) -> Result<Var<R, E>> {
        l2_normalize(&self.state_proj.apply(h)?)
    }

    /// Unit-norm action embeddings `[A, E]`. Without gradient tracking they are
    /// computed once and reused.
    pub fn action_embeddings(&self) -> Result<Var<R, E>> {
        // With gradients on, the encoder's parameters start their own tape here
        // (the attributes are constants), merged with the state's when scored.
        let caching = !mamba3::autograd::grad_mode::is_enabled();
        if caching && let Some(t) = self.cache.borrow().as_ref() {
            return Ok(Var::constant(t.clone()));
        }
        let z = l2_normalize(&self.action_enc.apply(&Var::constant(self.attrs.clone()))?)?;
        if caching {
            *self.cache.borrow_mut() = Some(z.tensor().clone());
        }
        Ok(z)
    }

    /// The current `1 / tau`.
    pub fn scale_var(&self, anchor: &Var<R, E>) -> Var<R, E> {
        self.log_scale.var(anchor).clamp(0.0, 100f32.ln()).exp()
    }

    /// Scores `[B, A]` and the state embedding.
    pub fn scores(&self, h: &Var<R, E>) -> Result<(Var<R, E>, Var<R, E>)> {
        let zs = self.state_embedding(h)?;
        let za = self.action_embeddings()?;
        let raw = match &self.bilinear {
            Some(w) => zs.matmul(&w.var(&zs))?.matmul_nt(&za)?,
            None => zs.matmul_nt(&za)?,
        };
        Ok((raw.mul(&self.scale_var(&zs))?, zs))
    }

    /// Configuration.
    pub fn config(&self) -> &ContrastiveConfig {
        &self.cfg
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for ContrastiveHead<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        v.child("state_proj", &self.state_proj);
        v.child("action_enc", &self.action_enc);
        v.param("log_scale", &self.log_scale);
        if let Some(w) = &self.bilinear {
            v.param("bilinear", w);
        }
    }
}

/// Auxiliary heads on the encoder state: a vol-scaled return regression and a
/// stop-loss-hit logit.
pub struct AuxHeads<R: Runtime, E: FloatElem> {
    /// Return head.
    pub ret: Option<TwoLayer<R, E>>,
    /// Risk head.
    pub risk: Option<TwoLayer<R, E>>,
}

impl<R: Runtime, E: FloatElem> AuxHeads<R, E> {
    /// Build the requested heads.
    pub fn new(input: usize, hidden: usize, ret: bool, risk: bool, device: &Device<R>, rng: &mut Rng) -> Self {
        Self {
            ret: ret.then(|| TwoLayer::new(input, hidden, 1, device, rng)),
            risk: risk.then(|| TwoLayer::new(input, hidden, 1, device, rng)),
        }
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for AuxHeads<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        v.child_opt("ret", &self.ret);
        v.child_opt("risk", &self.risk);
    }
}
