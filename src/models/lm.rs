//! Causal language model built on a [`HybridStack`].
//!
//! One configuration covers the whole family: pure Mamba-3, Mamba-3 with periodic
//! attention, or pure attention, with or without feed-forward blocks.
//!
//! ```no_run
//! # use mamba3::prelude::*;
//! # type R = mamba3::backends::Auto;
//! let device = Device::<R>::default();
//! let config = Mamba3LmConfig::builder()
//!     .vocab_size(32_000)
//!     .d_model(768)
//!     .n_layers(24)
//!     .pattern(LayerPattern::AttentionEvery { period: 8 })
//!     .build()
//!     .unwrap();
//! let model = config.init::<R, f32>(&device).unwrap();
//! ```

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::hybrid::{HybridConfig, HybridStack, LayerCache, LayerKind, LayerPattern};
use crate::nn::dropout::Dropout;
use crate::nn::embedding::{Embedding, EmbeddingConfig};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::lora::LoraConfig;
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::quant::QuantConfig;
use crate::ssm::config::SsmConfig;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::Rng;

/// Configuration for [`Mamba3Lm`].
#[derive(Debug, Clone)]
pub struct Mamba3LmConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// The layer stack.
    pub stack: HybridConfig,
    /// Dropout on the embedding output.
    pub embedding_dropout: f32,
    /// Share the embedding table with the output head.
    pub tie_embeddings: bool,
    /// Seed used by [`Mamba3LmConfig::init`].
    pub seed: u64,
}

impl Default for Mamba3LmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            stack: HybridConfig::default(),
            embedding_dropout: 0.0,
            tie_embeddings: false,
            seed: 0,
        }
    }
}

impl Mamba3LmConfig {
    /// Start a builder.
    pub fn builder() -> Mamba3LmConfigBuilder {
        Mamba3LmConfigBuilder::default()
    }

    /// Residual stream width.
    pub fn d_model(&self) -> usize {
        self.stack.d_model
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.stack.n_layers
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(Error::config("vocab_size must be positive"));
        }
        self.stack.validate()
    }

    /// Instantiate with the configured seed.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
    ) -> Result<Mamba3Lm<R, E>> {
        let mut rng = Rng::seeded(self.seed);
        self.init_with_rng(device, &mut rng)
    }

    /// Instantiate with an explicit RNG.
    pub fn init_with_rng<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Mamba3Lm<R, E>> {
        self.validate()?;
        let d_model = self.stack.d_model;
        let embed = EmbeddingConfig::new(self.vocab_size, d_model).init(device, rng);
        let head = if self.tie_embeddings {
            None
        } else {
            let mut cfg = LinearConfig::new(d_model, self.vocab_size)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                });
            if let Some(lora) = self.stack.lora {
                cfg = cfg.with_lora(lora);
            }
            Some(cfg.init(device, rng))
        };
        Ok(Mamba3Lm {
            embed,
            dropout: (self.embedding_dropout > 0.0)
                .then(|| Dropout::new(self.embedding_dropout)),
            stack: self.stack.init(device, rng)?,
            norm_f: RmsNormConfig::new(d_model)
                .with_eps(self.stack.norm_eps)
                .init(device, rng),
            head,
            config: self.clone(),
        })
    }
}

/// Builder for [`Mamba3LmConfig`].
#[derive(Debug, Clone, Default)]
pub struct Mamba3LmConfigBuilder {
    config: Option<Mamba3LmConfig>,
}

impl Mamba3LmConfigBuilder {
    fn edit(mut self, f: impl FnOnce(&mut Mamba3LmConfig)) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        f(&mut config);
        self.config = Some(config);
        self
    }

    /// Vocabulary size.
    pub fn vocab_size(self, vocab_size: usize) -> Self {
        self.edit(|c| c.vocab_size = vocab_size)
    }

    /// Residual stream width.
    pub fn d_model(self, d_model: usize) -> Self {
        self.edit(|c| {
            c.stack.d_model = d_model;
            c.stack.ssm.d_model = d_model;
            c.stack.attention = c.stack.attention.clone().with_d_model(d_model);
        })
    }

    /// Number of layers.
    pub fn n_layers(self, n_layers: usize) -> Self {
        self.edit(|c| c.stack.n_layers = n_layers)
    }

    /// Size the SSM so that `d_inner = expansion * d_model`, by choosing the head
    /// count. Call after `d_model`.
    pub fn expansion(self, expansion: f32) -> Self {
        self.edit(|c| {
            let ssm = &mut c.stack.ssm;
            let per_head = ssm.head_dim * ssm.mode.rank();
            ssm.n_heads =
                (((c.stack.d_model as f32 * expansion) as usize) / per_head.max(1)).max(1);
            ssm.n_groups = ssm.n_heads;
        })
    }

    /// Mixer pattern.
    pub fn pattern(self, pattern: LayerPattern) -> Self {
        self.edit(|c| c.stack.pattern = pattern)
    }

    /// Replace the SSM configuration wholesale.
    pub fn ssm(self, ssm: SsmConfig) -> Self {
        self.edit(|c| {
            let d_model = c.stack.d_model;
            c.stack.ssm = ssm;
            c.stack.ssm.d_model = d_model;
        })
    }

    /// Adjust the SSM configuration in place.
    pub fn with_ssm(self, f: impl FnOnce(&mut SsmConfig)) -> Self {
        self.edit(|c| f(&mut c.stack.ssm))
    }

    /// Add a feed-forward block after every mixer.
    pub fn mlp(self, mlp: crate::nn::mlp::MlpConfig) -> Self {
        self.edit(|c| c.stack.mlp = Some(mlp))
    }

    /// Attention settings for hybrid layers.
    pub fn attention(self, attention: crate::nn::attention::AttentionConfig) -> Self {
        self.edit(|c| c.stack.attention = attention)
    }

    /// Dropout on the embedding output.
    pub fn embedding_dropout(self, p: f32) -> Self {
        self.edit(|c| c.embedding_dropout = p)
    }

    /// Share the embedding table with the output head.
    pub fn tie_embeddings(self, tie: bool) -> Self {
        self.edit(|c| c.tie_embeddings = tie)
    }

    /// Attach LoRA adapters throughout.
    pub fn lora(self, lora: LoraConfig) -> Self {
        self.edit(|c| c.stack.lora = Some(lora))
    }

    /// Fake-quantize weights throughout.
    pub fn weight_quant(self, quant: QuantConfig) -> Self {
        self.edit(|c| c.stack.weight_quant = Some(quant))
    }

    /// Fake-quantize activations throughout.
    pub fn activation_quant(self, quant: QuantConfig) -> Self {
        self.edit(|c| c.stack.activation_quant = Some(quant))
    }

    /// Initialisation seed.
    pub fn seed(self, seed: u64) -> Self {
        self.edit(|c| c.seed = seed)
    }

    /// Validate and build.
    pub fn build(self) -> Result<Mamba3LmConfig> {
        let config = self.config.unwrap_or_default();
        config.validate()?;
        Ok(config)
    }
}

/// A causal language model.
pub struct Mamba3Lm<R: Runtime, E: FloatElem> {
    embed: Embedding<R, E>,
    dropout: Option<Dropout>,
    stack: HybridStack<R, E>,
    norm_f: RmsNorm<R, E>,
    /// `None` when the head is tied to the embedding table.
    head: Option<Linear<R, E>>,
    config: Mamba3LmConfig,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Mamba3Lm<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mamba3Lm(vocab={}, d_model={}, layers={}, tied={})",
            self.config.vocab_size,
            self.config.d_model(),
            self.config.n_layers(),
            self.head.is_none()
        )
    }
}

impl<R: Runtime, E: FloatElem> Mamba3Lm<R, E> {
    /// The configuration this model was built from.
    pub fn config(&self) -> &Mamba3LmConfig {
        &self.config
    }

    /// The mixer kind of each layer.
    pub fn layer_kinds(&self) -> Vec<LayerKind> {
        self.stack.kinds()
    }

    /// The layer stack.
    pub fn stack(&self) -> &HybridStack<R, E> {
        &self.stack
    }

    /// The embedding table.
    pub fn embedding(&self) -> &Embedding<R, E> {
        &self.embed
    }

    /// Hidden states for `[batch, seq]` ids, before the output head.
    ///
    /// `train` decides whether the forward pass is recorded for autodiff.
    pub fn hidden(&self, ids: &IdTensor<R>, train: bool) -> Result<Var<R, E>> {
        let mut x = self.embed.lookup(ids, train)?;
        if let Some(d) = &self.dropout {
            x = d.apply(&x)?;
        }
        let x = self.stack.apply(&x)?;
        self.norm_f.apply(&x)
    }

    /// Project hidden states to vocabulary logits.
    pub fn logits_from(&self, hidden: &Var<R, E>) -> Result<Var<R, E>> {
        match &self.head {
            Some(head) => head.apply(hidden),
            // Tied head: reuse the embedding table transposed. The transpose is a
            // real kernel launch, which is the price of sharing the parameter.
            None => {
                let table = self.embed.weight().var(hidden);
                hidden.matmul(&table.transpose()?)
            }
        }
    }

    /// Logits for `[batch, seq]` ids.
    pub fn forward(&self, ids: &IdTensor<R>, train: bool) -> Result<Var<R, E>> {
        if !train {
            let _guard = crate::autograd::no_grad();
            let hidden = self.hidden(ids, false)?;
            return self.logits_from(&hidden);
        }
        let hidden = self.hidden(ids, true)?;
        self.logits_from(&hidden)
    }

    /// A zeroed decoding cache.
    pub fn empty_cache(&self, batch: usize, device: &Device<R>) -> Vec<LayerCache<R, E>> {
        self.stack.empty_cache(batch, device)
    }

    /// Logits for the next token, advancing `cache`.
    ///
    /// Accepts either a single token or a whole prefill window.
    pub fn forward_cached(
        &self,
        ids: &IdTensor<R>,
        cache: &mut [LayerCache<R, E>],
    ) -> Result<Var<R, E>> {
        // Decoding never needs a tape, and a cache must not keep one alive.
        let _guard = crate::autograd::no_grad();
        let mut x = self.embed.lookup(ids, false)?;
        if let Some(d) = &self.dropout {
            x = d.apply(&x)?;
        }
        let x = self.stack.apply_cached(&x, cache)?;
        let x = self.norm_f.apply(&x)?;
        self.logits_from(&x)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Mamba3Lm<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("embed", &self.embed);
        visitor.child("stack", &self.stack);
        visitor.child("norm_f", &self.norm_f);
        if let Some(head) = &self.head {
            visitor.child("head", head);
        }
    }
}
