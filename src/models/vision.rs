//! Vision Mamba-3.
//!
//! Images are cut into non-overlapping patches, embedded, and fed through a stack
//! of Mamba-3 mixers. The one thing an image needs that a sentence does not is a
//! **non-causal** view of the sequence, so each block can run in both directions
//! and add the results — the standard bidirectional-SSM trick for vision. It is
//! expressed here as one fused mixer per block whose second half of heads scans
//! right-to-left: both directions keep their own parameters, but every kernel
//! launches once at double width instead of twice.
//!
//! ```no_run
//! # use mamba3::prelude::*;
//! # use mamba3::models::vision::{ScanDirection, Pooling};
//! # type R = mamba3::backends::Auto;
//! let device = Device::<R>::default();
//! let model = VisionMamba3Config::builder()
//!     .image_size(224)
//!     .patch_size(16)
//!     .num_classes(1000)
//!     .d_model(384)
//!     .n_layers(24)
//!     .direction(ScanDirection::Bidirectional)
//!     .pooling(Pooling::Mean)
//!     .build()
//!     .unwrap()
//!     .init::<R, f32>(&device)
//!     .unwrap();
//! ```

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Mixer, Mamba3MixerConfig};
use crate::nn::conv::{PatchEmbed, PatchEmbedConfig};
use crate::nn::dropout::Dropout;
use crate::nn::embedding::PositionalEmbedding;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::lora::LoraConfig;
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::nn::quant::QuantConfig;
use crate::ssm::config::SsmConfig;
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

/// How each block traverses the patch sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScanDirection {
    /// Left to right only, as in a language model.
    Forward,
    /// Left to right and right to left, summed. Each direction has its own half
    /// of a single fused mixer's heads, groups and channels.
    Bidirectional,
}

/// How patch features become a single vector for the classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Pooling {
    /// Average over patches.
    Mean,
    /// Read a prepended learnable class token.
    ClassToken,
    /// Use the last position, which a causal scan has seen everything through.
    Last,
}

/// Configuration for [`VisionMamba3`].
#[derive(Debug, Clone)]
pub struct VisionMamba3Config {
    /// Input image edge length.
    pub image_size: usize,
    /// Patch edge length.
    pub patch_size: usize,
    /// Input channels.
    pub in_channels: usize,
    /// Number of output classes. `0` builds a feature extractor with no head.
    pub num_classes: usize,
    /// Embedding width.
    pub d_model: usize,
    /// Number of blocks.
    pub n_layers: usize,
    /// Traversal direction.
    pub direction: ScanDirection,
    /// How patch features are pooled.
    pub pooling: Pooling,
    /// Settings for the mixers.
    pub ssm: SsmConfig,
    /// Dropout after the patch embedding.
    pub dropout: f32,
    /// Normalisation epsilon.
    pub norm_eps: f32,
    /// LoRA adapters on every projection.
    pub lora: Option<LoraConfig>,
    /// Fake quantization for weights.
    pub weight_quant: Option<QuantConfig>,
    /// Initialisation seed.
    pub seed: u64,
}

impl Default for VisionMamba3Config {
    fn default() -> Self {
        let mut ssm = SsmConfig::default();
        ssm.d_model = 384;
        ssm.n_heads = 6;
        ssm.n_groups = 6;
        ssm.head_dim = 64;
        ssm.d_state = 64;
        Self {
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            num_classes: 1000,
            d_model: 384,
            n_layers: 12,
            direction: ScanDirection::Bidirectional,
            pooling: Pooling::Mean,
            ssm,
            dropout: 0.0,
            norm_eps: 1e-5,
            lora: None,
            weight_quant: None,
            seed: 0,
        }
    }
}

impl VisionMamba3Config {
    /// Start a builder.
    pub fn builder() -> VisionMamba3ConfigBuilder {
        VisionMamba3ConfigBuilder::default()
    }

    /// Number of patches an image produces.
    pub fn num_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Length of the sequence the mixers see, including any class token.
    pub fn sequence_len(&self) -> usize {
        self.num_patches() + usize::from(self.pooling == Pooling::ClassToken)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 || self.image_size % self.patch_size != 0 {
            return Err(Error::config(format!(
                "image size {} is not divisible by patch size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.d_model == 0 || self.n_layers == 0 {
            return Err(Error::config("d_model and n_layers must be positive"));
        }
        let mut ssm = self.ssm.clone();
        ssm.d_model = self.d_model;
        ssm.validate()
    }

    /// Instantiate with the configured seed.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
    ) -> Result<VisionMamba3<R, E>> {
        let mut rng = Rng::seeded(self.seed);
        self.init_with_rng(device, &mut rng)
    }

    /// Instantiate with an explicit RNG.
    pub fn init_with_rng<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<VisionMamba3<R, E>> {
        self.validate()?;
        let mut ssm = self.ssm.clone();
        ssm.d_model = self.d_model;

        // A bidirectional block is one fused mixer with both directions'
        // parameters, direction-major: doubled heads and groups keep the
        // per-direction capacity of the configured SSM, and the mixer reverses
        // the backward half's time axis internally. One mixer's worth of kernel
        // launches instead of two.
        let mixer_config = || {
            let mut ssm = ssm.clone();
            let bidirectional = self.direction == ScanDirection::Bidirectional;
            if bidirectional {
                ssm.n_heads *= 2;
                ssm.n_groups *= 2;
            }
            let mut cfg = Mamba3MixerConfig::new(ssm)
                .with_depth(self.n_layers)
                .with_bidirectional(bidirectional);
            if let Some(lora) = self.lora {
                cfg = cfg.with_lora(lora);
            }
            if let Some(q) = self.weight_quant {
                cfg = cfg.with_weight_quant(q);
            }
            cfg
        };

        let mut blocks = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            blocks.push(VisionBlock {
                norm: RmsNormConfig::new(self.d_model)
                    .with_eps(self.norm_eps)
                    .init(device, rng),
                mixer: mixer_config().init(device, rng)?,
            });
        }

        Ok(VisionMamba3 {
            patch_embed: PatchEmbedConfig::new(
                self.image_size,
                self.patch_size,
                self.in_channels,
                self.d_model,
            )
            .init(device, rng)?,
            class_token: (self.pooling == Pooling::ClassToken).then(|| {
                Param::new(Tensor::zeros(vec![1, 1, self.d_model], device))
            }),
            pos_embed: PositionalEmbedding::new(
                self.sequence_len(),
                self.d_model,
                device,
                rng,
            ),
            dropout: (self.dropout > 0.0).then(|| Dropout::new(self.dropout)),
            blocks,
            norm: RmsNormConfig::new(self.d_model)
                .with_eps(self.norm_eps)
                .init(device, rng),
            head: (self.num_classes > 0).then(|| {
                LinearConfig::new(self.d_model, self.num_classes).init(device, rng)
            }),
            config: self.clone(),
        })
    }
}

/// Builder for [`VisionMamba3Config`].
#[derive(Debug, Clone, Default)]
pub struct VisionMamba3ConfigBuilder {
    config: Option<VisionMamba3Config>,
}

impl VisionMamba3ConfigBuilder {
    fn edit(mut self, f: impl FnOnce(&mut VisionMamba3Config)) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        f(&mut config);
        self.config = Some(config);
        self
    }

    /// Input image edge length.
    pub fn image_size(self, size: usize) -> Self {
        self.edit(|c| c.image_size = size)
    }

    /// Patch edge length.
    pub fn patch_size(self, size: usize) -> Self {
        self.edit(|c| c.patch_size = size)
    }

    /// Input channels.
    pub fn in_channels(self, channels: usize) -> Self {
        self.edit(|c| c.in_channels = channels)
    }

    /// Number of classes; `0` omits the head.
    pub fn num_classes(self, classes: usize) -> Self {
        self.edit(|c| c.num_classes = classes)
    }

    /// Embedding width.
    pub fn d_model(self, d_model: usize) -> Self {
        self.edit(|c| {
            c.d_model = d_model;
            c.ssm.d_model = d_model;
        })
    }

    /// Number of blocks.
    pub fn n_layers(self, n_layers: usize) -> Self {
        self.edit(|c| c.n_layers = n_layers)
    }

    /// Traversal direction.
    pub fn direction(self, direction: ScanDirection) -> Self {
        self.edit(|c| c.direction = direction)
    }

    /// Pooling strategy.
    pub fn pooling(self, pooling: Pooling) -> Self {
        self.edit(|c| c.pooling = pooling)
    }

    /// Adjust the SSM configuration in place.
    pub fn with_ssm(self, f: impl FnOnce(&mut SsmConfig)) -> Self {
        self.edit(|c| f(&mut c.ssm))
    }

    /// Dropout after the patch embedding.
    pub fn dropout(self, p: f32) -> Self {
        self.edit(|c| c.dropout = p)
    }

    /// Attach LoRA adapters throughout.
    pub fn lora(self, lora: LoraConfig) -> Self {
        self.edit(|c| c.lora = Some(lora))
    }

    /// Fake-quantize weights throughout.
    pub fn weight_quant(self, quant: QuantConfig) -> Self {
        self.edit(|c| c.weight_quant = Some(quant))
    }

    /// Initialisation seed.
    pub fn seed(self, seed: u64) -> Self {
        self.edit(|c| c.seed = seed)
    }

    /// Validate and build.
    pub fn build(self) -> Result<VisionMamba3Config> {
        let config = self.config.unwrap_or_default();
        config.validate()?;
        Ok(config)
    }
}

/// One Mamba-3 block; a bidirectional one holds a single fused mixer whose
/// second half of heads runs right-to-left.
struct VisionBlock<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    mixer: Mamba3Mixer<R, E>,
}

impl<R: Runtime, E: FloatElem> VisionBlock<R, E> {
    fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        input.add(&self.mixer.apply(&self.norm.apply(input)?)?)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for VisionBlock<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm", &self.norm);
        visitor.child("mixer", &self.mixer);
    }
}

/// An image classifier built from Mamba-3 mixers.
pub struct VisionMamba3<R: Runtime, E: FloatElem> {
    patch_embed: PatchEmbed<R, E>,
    class_token: Option<Param<R, E>>,
    pos_embed: PositionalEmbedding<R, E>,
    dropout: Option<Dropout>,
    blocks: Vec<VisionBlock<R, E>>,
    norm: RmsNorm<R, E>,
    head: Option<Linear<R, E>>,
    config: VisionMamba3Config,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for VisionMamba3<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "VisionMamba3({}x{} image, {} patches, d_model={}, layers={}, {:?})",
            self.config.image_size,
            self.config.image_size,
            self.config.num_patches(),
            self.config.d_model,
            self.config.n_layers,
            self.config.direction,
        )
    }
}

impl<R: Runtime, E: FloatElem> VisionMamba3<R, E> {
    /// The configuration this model was built from.
    pub fn config(&self) -> &VisionMamba3Config {
        &self.config
    }

    /// Patch features for `[batch, channels, height, width]` images.
    pub fn features(&self, images: &Var<R, E>) -> Result<Var<R, E>> {
        let mut x = self.patch_embed.apply(images)?;
        if let Some(token) = &self.class_token {
            let batch = x.shape().dim(0);
            let cls = token
                .var(images)
                .expand(vec![batch, 1, self.config.d_model])?;
            x = crate::autograd::cat(&[cls, x], 1)?;
        }
        x = self.pos_embed.add_to(&x, 0)?;
        if let Some(d) = &self.dropout {
            x = d.apply(&x)?;
        }
        for block in &self.blocks {
            x = block.apply(&x)?;
        }
        self.norm.apply(&x)
    }

    /// Pool patch features into one vector per image.
    pub fn pool(&self, features: &Var<R, E>) -> Result<Var<R, E>> {
        let seq = features.shape().dim(1);
        match self.config.pooling {
            Pooling::Mean => features.mean_dim(1)?.squeeze(1),
            Pooling::ClassToken => features.slice(1, 0, 1)?.squeeze(1),
            Pooling::Last => features.slice(1, seq - 1, 1)?.squeeze(1),
        }
    }

    /// Class logits for `[batch, channels, height, width]` images.
    pub fn forward(&self, images: &Var<R, E>) -> Result<Var<R, E>> {
        let pooled = self.pool(&self.features(images)?)?;
        match &self.head {
            Some(head) => head.apply(&pooled),
            None => Ok(pooled),
        }
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for VisionMamba3<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("patch_embed", &self.patch_embed);
        if let Some(token) = &self.class_token {
            visitor.param("class_token", token);
        }
        visitor.child("pos_embed", &self.pos_embed);
        for (i, block) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, block);
        }
        visitor.child("norm", &self.norm);
        if let Some(head) = &self.head {
            visitor.child("head", head);
        }
    }
}
