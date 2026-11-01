//! The decision model: an encoder of the input window, a decision head producing
//! one score per action, and optional auxiliary heads. Every compared model
//! (ablations A–F, the linear and MLP baselines) is one [`ModelConfig`].

use mamba3::autograd::Var;
use mamba3::backend::{Device, FloatElem};
use mamba3::cubecl::prelude::Runtime;
use mamba3::error::{Error, Result};
use mamba3::nn::{Linear, LinearConfig, Module, ModuleVisitor};
use mamba3::tensor::ops::random::Rng;
use serde::{Deserialize, Serialize};

use crate::heads::{AuxHeads, ContrastiveConfig, ContrastiveHead, TwoLayer};
use crate::mamba::{MambaConfig, MambaEncoder};
use crate::tcn::{TcnConfig, TcnEncoder};

/// What turns the `[B, W, F]` window into a state vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EncoderConfig {
    /// Mamba-1 encoder over the window.
    Mamba(MambaConfig),
    /// Temporal convolutional encoder over the window.
    Tcn(TcnConfig),
    /// Only the decision day's feature row, through an MLP body of this width
    /// (0: the raw row, for the linear baseline).
    LastRow {
        /// Input features.
        input_dim: usize,
        /// Hidden width, 0 for none.
        hidden: usize,
    },
}

/// What turns the state vector into action scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HeadConfig {
    /// One linear layer to 3 logits.
    Linear,
    /// `Linear -> SiLU -> Linear` to 3 logits.
    Mlp {
        /// Hidden width.
        hidden: usize,
    },
    /// The CLM state/action embeddings and their scores.
    Contrastive(ContrastiveConfig),
}

/// A complete model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Encoder.
    pub encoder: EncoderConfig,
    /// Decision head.
    pub head: HeadConfig,
    /// Auxiliary return head.
    pub aux_return: bool,
    /// Auxiliary stop-loss head.
    pub aux_risk: bool,
    /// Hidden width of the auxiliary heads.
    pub aux_hidden: usize,
    /// Action attribute rows (see [`crate::heads::action_attributes`]).
    pub action_attrs: Vec<f32>,
    /// Initialisation seed.
    pub seed: u64,
}

enum Encoder<R: Runtime, E: FloatElem> {
    Mamba(MambaEncoder<R, E>),
    Tcn(TcnEncoder<R, E>),
    LastRow(Option<Linear<R, E>>, usize),
}

enum Head<R: Runtime, E: FloatElem> {
    Linear(Linear<R, E>),
    Mlp(TwoLayer<R, E>),
    Contrastive(ContrastiveHead<R, E>),
}

/// Everything a forward pass produces.
pub struct Outputs<R: Runtime, E: FloatElem> {
    /// Action scores `[B, 3]` (logits for the supervised heads, scaled
    /// similarities for the contrastive one). Not calibrated probabilities.
    pub scores: Var<R, E>,
    /// Unit-norm state embedding `[B, E]` (contrastive head only).
    pub state_embedding: Option<Var<R, E>>,
    /// Predicted vol-scaled return `[B]`.
    pub ret: Option<Var<R, E>>,
    /// Stop-loss logit `[B]`.
    pub risk: Option<Var<R, E>>,
}

/// The model.
pub struct DecisionModel<R: Runtime, E: FloatElem> {
    encoder: Encoder<R, E>,
    head: Head<R, E>,
    aux: AuxHeads<R, E>,
    cfg: ModelConfig,
}

impl<R: Runtime, E: FloatElem> DecisionModel<R, E> {
    /// Build and initialise from `cfg.seed`.
    pub fn new(cfg: &ModelConfig, device: &Device<R>) -> Result<Self> {
        let mut rng = Rng::seeded(cfg.seed);
        let encoder = match &cfg.encoder {
            EncoderConfig::Mamba(m) => Encoder::Mamba(MambaEncoder::new(m, device, &mut rng)),
            EncoderConfig::Tcn(t) => Encoder::Tcn(TcnEncoder::new(t, device, &mut rng)),
            EncoderConfig::LastRow { input_dim, hidden } => {
                if *hidden == 0 {
                    Encoder::LastRow(None, *input_dim)
                } else {
                    Encoder::LastRow(
                        Some(LinearConfig::new(*input_dim, *hidden).init(device, &mut rng)),
                        *hidden,
                    )
                }
            }
        };
        let width = match &encoder {
            Encoder::Mamba(m) => m.out_dim(),
            Encoder::Tcn(t) => t.out_dim(),
            Encoder::LastRow(_, w) => *w,
        };
        if cfg.action_attrs.len() != 3 * crate::heads::ATTR_DIM {
            return Err(Error::Shape(format!(
                "action_attrs must hold 3 x {} values, got {}",
                crate::heads::ATTR_DIM,
                cfg.action_attrs.len()
            )));
        }
        let head = match &cfg.head {
            HeadConfig::Linear => Head::Linear(LinearConfig::new(width, 3).init(device, &mut rng)),
            HeadConfig::Mlp { hidden } => Head::Mlp(TwoLayer::new(width, *hidden, 3, device, &mut rng)),
            HeadConfig::Contrastive(c) => Head::Contrastive(ContrastiveHead::new(
                width,
                c,
                &cfg.action_attrs,
                device,
                &mut rng,
            )),
        };
        let aux = AuxHeads::new(width, cfg.aux_hidden.max(1), cfg.aux_return, cfg.aux_risk, device, &mut rng);
        Ok(Self {
            encoder,
            head,
            aux,
            cfg: cfg.clone(),
        })
    }

    /// The configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Set the Mamba scans' forward schedule (no effect on other encoders).
    pub fn set_scan_mode(&mut self, mode: mamba3::tensor::ops::selective_scan::SelectiveScanMode) {
        if let Encoder::Mamba(m) = &mut self.encoder {
            m.scan_mode = mode;
        }
    }

    /// Forget cached inference values (action embeddings); call after an update.
    pub fn invalidate(&self) {
        if let Head::Contrastive(c) = &self.head {
            c.invalidate();
        }
    }

    /// The encoder state `[B, width]` of a `[B, W, F]` window.
    pub fn encode(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        match &self.encoder {
            Encoder::Mamba(m) => m.apply(x),
            Encoder::Tcn(t) => t.apply(x),
            Encoder::LastRow(body, _) => {
                let w = x.shape().dim(1);
                let last = x.slice(1, w - 1, 1)?.squeeze(1)?;
                match body {
                    Some(l) => l.apply(&last)?.silu(),
                    None => Ok(last),
                }
            }
        }
    }

    /// Forward pass over a `[B, W, F]` window.
    pub fn forward(&self, x: &Var<R, E>) -> Result<Outputs<R, E>> {
        let h = self.encode(x)?;
        let (scores, state_embedding) = match &self.head {
            Head::Linear(l) => (l.apply(&h)?, None),
            Head::Mlp(m) => (m.apply(&h)?, None),
            Head::Contrastive(c) => {
                let (s, z) = c.scores(&h)?;
                (s, Some(z))
            }
        };
        let squeeze = |v: Var<R, E>| v.squeeze(1);
        let ret = self.aux.ret.as_ref().map(|r| r.apply(&h).and_then(squeeze)).transpose()?;
        let risk = self.aux.risk.as_ref().map(|r| r.apply(&h).and_then(squeeze)).transpose()?;
        Ok(Outputs {
            scores,
            state_embedding,
            ret,
            risk,
        })
    }

    /// Number of scalar parameters.
    pub fn num_params(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for DecisionModel<R, E> {
    fn visit(&self, v: &mut ModuleVisitor<'_, R, E>) {
        match &self.encoder {
            Encoder::Mamba(m) => v.child("encoder", m),
            Encoder::Tcn(t) => v.child("encoder", t),
            Encoder::LastRow(Some(l), _) => v.child("encoder", l),
            Encoder::LastRow(None, _) => {}
        }
        match &self.head {
            Head::Linear(l) => v.child("head", l),
            Head::Mlp(m) => v.child("head", m),
            Head::Contrastive(c) => v.child("head", c),
        }
        v.child("aux", &self.aux);
    }
}
