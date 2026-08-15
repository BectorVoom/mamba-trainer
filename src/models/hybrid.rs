//! Hybrid stacks: Mamba-3 mixers interleaved with attention.
//!
//! The pattern is declarative. A stack is described by how many layers it has and
//! which of them use attention; everything else — parameter naming, caching,
//! LoRA, quantization — is identical either way, because both mixers are ordinary
//! modules built from the same [`crate::nn`] pieces.
//!
//! ```no_run
//! # use mamba3::prelude::*;
//! # use mamba3::ssm::SsmConfig;
//! let stack = HybridConfig::builder()
//!     .d_model(768)
//!     .n_layers(24)
//!     .pattern(LayerPattern::AttentionEvery { period: 6 })  // 4 attention layers
//!     .build()
//!     .unwrap();
//! ```

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::attention::{AttentionCache, AttentionConfig, MultiHeadAttention};
use crate::nn::lora::LoraConfig;
use crate::nn::mlp::{Mlp, MlpConfig};
use crate::nn::module::{Layer, Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::quant::QuantConfig;
use crate::models::mamba3::{Mamba3Mixer, Mamba3MixerConfig, MixerCache};
use crate::ssm::config::SsmConfig;
use crate::tensor::ops::random::Rng;

/// Which token mixer a layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LayerKind {
    /// A Mamba-3 state space mixer.
    Mamba,
    /// Multi-head self-attention.
    Attention,
}

/// How mixers are distributed through the stack.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LayerPattern {
    /// Every layer is a Mamba-3 mixer.
    AllMamba,
    /// Every layer is attention. Useful as a control when ablating.
    AllAttention,
    /// Every `period`-th layer (counting from the end of each period) is attention.
    AttentionEvery {
        /// Period between attention layers.
        period: usize,
    },
    /// Attention at exactly these layer indices.
    AttentionAt(Vec<usize>),
    /// An explicit sequence, cycled if shorter than the stack.
    Explicit(Vec<LayerKind>),
}

impl Default for LayerPattern {
    fn default() -> Self {
        LayerPattern::AllMamba
    }
}

impl LayerPattern {
    /// Expand into one entry per layer.
    pub fn expand(&self, n_layers: usize) -> Result<Vec<LayerKind>> {
        let kinds = match self {
            LayerPattern::AllMamba => vec![LayerKind::Mamba; n_layers],
            LayerPattern::AllAttention => vec![LayerKind::Attention; n_layers],
            LayerPattern::AttentionEvery { period } => {
                if *period == 0 {
                    return Err(Error::config("attention period must be positive"));
                }
                (0..n_layers)
                    .map(|i| {
                        if (i + 1) % period == 0 {
                            LayerKind::Attention
                        } else {
                            LayerKind::Mamba
                        }
                    })
                    .collect()
            }
            LayerPattern::AttentionAt(indices) => {
                for &i in indices {
                    if i >= n_layers {
                        return Err(Error::config(format!(
                            "attention index {i} is past the end of a {n_layers}-layer stack"
                        )));
                    }
                }
                (0..n_layers)
                    .map(|i| {
                        if indices.contains(&i) {
                            LayerKind::Attention
                        } else {
                            LayerKind::Mamba
                        }
                    })
                    .collect()
            }
            LayerPattern::Explicit(kinds) => {
                if kinds.is_empty() {
                    return Err(Error::config("an explicit pattern cannot be empty"));
                }
                (0..n_layers).map(|i| kinds[i % kinds.len()]).collect()
            }
        };
        Ok(kinds)
    }
}

/// Configuration for a [`HybridStack`].
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Residual stream width.
    pub d_model: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Which layers use which mixer.
    pub pattern: LayerPattern,
    /// Settings for the Mamba-3 layers. `d_model` is overwritten to match.
    pub ssm: SsmConfig,
    /// Settings for the attention layers. `d_model` is overwritten to match.
    pub attention: AttentionConfig,
    /// Feed-forward block after each mixer. `None` gives a pure mixer stack, which
    /// is what plain Mamba models use.
    pub mlp: Option<MlpConfig>,
    /// Normalisation epsilon.
    pub norm_eps: f32,
    /// LoRA adapters on every projection in the stack.
    pub lora: Option<LoraConfig>,
    /// Fake quantization for every projection weight.
    pub weight_quant: Option<QuantConfig>,
    /// Fake quantization for every projection input.
    pub activation_quant: Option<QuantConfig>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            n_layers: 12,
            pattern: LayerPattern::AllMamba,
            ssm: SsmConfig::default(),
            attention: AttentionConfig::new(768, 12),
            mlp: None,
            norm_eps: 1e-5,
            lora: None,
            weight_quant: None,
            activation_quant: None,
        }
    }
}

impl HybridConfig {
    /// Start a builder.
    pub fn builder() -> HybridConfigBuilder {
        HybridConfigBuilder::default()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.n_layers == 0 {
            return Err(Error::config("d_model and n_layers must be positive"));
        }
        self.pattern.expand(self.n_layers)?;
        let mut ssm = self.ssm.clone();
        ssm.d_model = self.d_model;
        ssm.validate()
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<HybridStack<R, E>> {
        self.validate()?;
        let kinds = self.pattern.expand(self.n_layers)?;

        let mut ssm = self.ssm.clone();
        ssm.d_model = self.d_model;

        let attention = {
            let mut cfg = self.attention.clone().with_d_model(self.d_model);
            if let Some(lora) = self.lora {
                cfg = cfg.with_lora(lora);
            }
            if let Some(q) = self.weight_quant {
                cfg = cfg.with_weight_quant(q);
            }
            cfg
        };

        let mut layers = Vec::with_capacity(self.n_layers);
        for kind in kinds {
            let mixer = match kind {
                LayerKind::Mamba => {
                    let mut cfg = Mamba3MixerConfig::new(ssm.clone()).with_depth(self.n_layers);
                    if let Some(lora) = self.lora {
                        cfg = cfg.with_lora(lora);
                    }
                    if let Some(q) = self.weight_quant {
                        cfg = cfg.with_weight_quant(q);
                    }
                    if let Some(q) = self.activation_quant {
                        cfg = cfg.with_activation_quant(q);
                    }
                    Mixer::Mamba(Box::new(cfg.init(device, rng)?))
                }
                LayerKind::Attention => {
                    Mixer::Attention(Box::new(attention.init(device, rng)?))
                }
            };
            let mlp = match &self.mlp {
                Some(cfg) => {
                    let mut cfg = cfg.clone();
                    if let Some(lora) = self.lora {
                        cfg = cfg.with_lora(lora);
                    }
                    if let Some(q) = self.weight_quant {
                        cfg = cfg.with_weight_quant(q);
                    }
                    Some((
                        RmsNormConfig::new(self.d_model)
                            .with_eps(self.norm_eps)
                            .init(device, rng),
                        cfg.init(device, rng),
                    ))
                }
                None => None,
            };
            layers.push(HybridLayer {
                norm: RmsNormConfig::new(self.d_model)
                    .with_eps(self.norm_eps)
                    .init(device, rng),
                mixer,
                mlp,
            });
        }
        Ok(HybridStack { layers })
    }
}

/// Builder for [`HybridConfig`].
#[derive(Debug, Clone, Default)]
pub struct HybridConfigBuilder {
    config: Option<HybridConfig>,
}

impl HybridConfigBuilder {
    fn edit(mut self, f: impl FnOnce(&mut HybridConfig)) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        f(&mut config);
        self.config = Some(config);
        self
    }

    /// Residual stream width.
    pub fn d_model(self, d_model: usize) -> Self {
        self.edit(|c| {
            c.d_model = d_model;
            c.ssm.d_model = d_model;
        })
    }

    /// Number of layers.
    pub fn n_layers(self, n_layers: usize) -> Self {
        self.edit(|c| c.n_layers = n_layers)
    }

    /// Size the SSM so that `d_inner = expansion * d_model`, by choosing the head
    /// count. Call after `d_model`.
    pub fn expansion(self, expansion: f32) -> Self {
        self.edit(|c| {
            let per_head = c.ssm.head_dim * c.ssm.mode.rank();
            c.ssm.n_heads = (((c.d_model as f32 * expansion) as usize) / per_head.max(1)).max(1);
            c.ssm.n_groups = c.ssm.n_heads;
        })
    }

    /// Mixer pattern.
    pub fn pattern(self, pattern: LayerPattern) -> Self {
        self.edit(|c| c.pattern = pattern)
    }

    /// Replace the SSM configuration.
    pub fn ssm(self, ssm: SsmConfig) -> Self {
        self.edit(|c| c.ssm = ssm)
    }

    /// Adjust the SSM configuration in place.
    pub fn with_ssm(self, f: impl FnOnce(&mut SsmConfig)) -> Self {
        self.edit(|c| f(&mut c.ssm))
    }

    /// Replace the attention configuration.
    pub fn attention(self, attention: AttentionConfig) -> Self {
        self.edit(|c| c.attention = attention)
    }

    /// Add a feed-forward block after every mixer.
    pub fn mlp(self, mlp: MlpConfig) -> Self {
        self.edit(|c| c.mlp = Some(mlp))
    }

    /// Normalisation epsilon.
    pub fn norm_eps(self, eps: f32) -> Self {
        self.edit(|c| c.norm_eps = eps)
    }

    /// Attach LoRA adapters throughout.
    pub fn lora(self, lora: LoraConfig) -> Self {
        self.edit(|c| c.lora = Some(lora))
    }

    /// Fake-quantize weights throughout.
    pub fn weight_quant(self, quant: QuantConfig) -> Self {
        self.edit(|c| c.weight_quant = Some(quant))
    }

    /// Fake-quantize activations throughout.
    pub fn activation_quant(self, quant: QuantConfig) -> Self {
        self.edit(|c| c.activation_quant = Some(quant))
    }

    /// Validate and build.
    pub fn build(self) -> Result<HybridConfig> {
        let config = self.config.unwrap_or_default();
        config.validate()?;
        Ok(config)
    }
}

/// One mixer, either kind.
enum Mixer<R: Runtime, E: FloatElem> {
    Mamba(Box<Mamba3Mixer<R, E>>),
    Attention(Box<MultiHeadAttention<R, E>>),
}

/// Per-layer decoding state.
pub struct LayerCache<R: Runtime, E: FloatElem> {
    /// Mamba recurrent state, when the layer is a Mamba layer.
    pub mamba: Option<MixerCache<R, E>>,
    /// Key/value cache, when the layer is an attention layer.
    pub attention: AttentionCache<R, E>,
}

/// A stack of pre-norm residual layers.
pub struct HybridStack<R: Runtime, E: FloatElem> {
    layers: Vec<HybridLayer<R, E>>,
}

struct HybridLayer<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    mixer: Mixer<R, E>,
    mlp: Option<(RmsNorm<R, E>, Mlp<R, E>)>,
}

impl<R: Runtime, E: FloatElem> HybridStack<R, E> {
    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The mixer kind of each layer.
    pub fn kinds(&self) -> Vec<LayerKind> {
        self.layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Mamba(_) => LayerKind::Mamba,
                Mixer::Attention(_) => LayerKind::Attention,
            })
            .collect()
    }

    /// A zeroed cache for every layer.
    pub fn empty_cache(&self, batch: usize, device: &Device<R>) -> Vec<LayerCache<R, E>> {
        self.layers
            .iter()
            .map(|l| LayerCache {
                mamba: match &l.mixer {
                    Mixer::Mamba(m) => Some(m.empty_cache(batch, device)),
                    Mixer::Attention(_) => None,
                },
                attention: AttentionCache::new(),
            })
            .collect()
    }

    /// Run the whole stack over `[batch, seq, d_model]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.apply(&current)?;
        }
        Ok(current)
    }

    /// Run the stack incrementally, updating `cache` in place.
    ///
    /// Works for a single token or for a whole prefill window.
    pub fn apply_cached(
        &self,
        input: &Var<R, E>,
        cache: &mut [LayerCache<R, E>],
    ) -> Result<Var<R, E>> {
        if cache.len() != self.layers.len() {
            return Err(Error::config(format!(
                "cache has {} entries but the stack has {} layers",
                cache.len(),
                self.layers.len()
            )));
        }
        let single = input.shape().dim(1) == 1;
        let mut current = input.clone();
        for (layer, slot) in self.layers.iter().zip(cache.iter_mut()) {
            current = layer.apply_cached(&current, slot, single)?;
        }
        Ok(current)
    }
}

impl<R: Runtime, E: FloatElem> HybridLayer<R, E> {
    fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        let normed = self.norm.apply(input)?;
        let mixed = match &self.mixer {
            Mixer::Mamba(m) => m.apply(&normed)?,
            Mixer::Attention(a) => a.apply(&normed)?,
        };
        let mut out = input.add(&mixed)?;
        if let Some((norm, mlp)) = &self.mlp {
            out = out.add(&mlp.apply(&norm.apply(&out)?)?)?;
        }
        Ok(out)
    }

    fn apply_cached(
        &self,
        input: &Var<R, E>,
        cache: &mut LayerCache<R, E>,
        single: bool,
    ) -> Result<Var<R, E>> {
        let normed = self.norm.apply(input)?;
        let mixed = match &self.mixer {
            Mixer::Mamba(m) => {
                let previous = cache.mamba.clone();
                let (out, state) = if single && previous.is_some() {
                    m.step(&normed, previous.as_ref().unwrap())?
                } else {
                    m.apply_with_state(&normed, previous.as_ref())?
                };
                cache.mamba = Some(state.detach());
                out
            }
            Mixer::Attention(a) => a.apply_cached(&normed, Some(&mut cache.attention))?,
        };
        let mut out = input.add(&mixed)?;
        if let Some((norm, mlp)) = &self.mlp {
            out = out.add(&mlp.apply(&norm.apply(&out)?)?)?;
        }
        Ok(out)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for HybridStack<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (i, layer) in self.layers.iter().enumerate() {
            visitor.child_at("layers", i, layer);
        }
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for HybridLayer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm", &self.norm);
        match &self.mixer {
            Mixer::Mamba(m) => visitor.child("mixer", m.as_ref()),
            Mixer::Attention(a) => visitor.child("attention", a.as_ref()),
        }
        if let Some((norm, mlp)) = &self.mlp {
            visitor.child("mlp_norm", norm);
            visitor.child("mlp", mlp);
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for HybridStack<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}
