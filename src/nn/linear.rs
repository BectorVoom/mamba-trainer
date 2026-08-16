//! Dense projection — the layer every other feature hangs off.
//!
//! A single `Linear` carries the optional pieces the rest of the crate needs:
//! a bias, a LoRA adapter, and weight/activation fake quantizers. Building them
//! into one layer (rather than wrapping it three times) means Mamba mixers,
//! attention blocks, MLPs and vision heads all inherit QAT and LoRA for free.
//!
//! Weights are stored as `[in_features, out_features]` so the forward pass is a
//! plain `x @ W` with no transpose, and so a LoRA delta `A @ B` has exactly the
//! weight's shape.
//!
//! ```no_run
//! # use mamba3::prelude::*;
//! # use mamba3::nn::quant::QuantConfig;
//! # use mamba3::tensor::ops::random::Rng;
//! # type R = mamba3::backends::Auto;
//! # let device = Device::<R>::default();
//! # let mut rng = Rng::seeded(0);
//! let proj = LinearConfig::new(512, 2048)
//!     .with_bias(false)
//!     .with_lora(LoraConfig::builder().rank(16).build().unwrap())
//!     .with_weight_quant(QuantConfig::int8_weights())
//!     .init::<R, f32>(&device, &mut rng);
//! ```

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::init::Initializer;
use crate::nn::lora::{LoraConfig, LoraLinear};
use crate::nn::module::{Layer, Module, ModuleVisitor};
use crate::nn::param::Param;
use crate::nn::quant::{QuantConfig, Quantizer};
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

/// Configuration for [`Linear`].
#[derive(Debug, Clone)]
pub struct LinearConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    initializer: Initializer,
    bias_initializer: Initializer,
    lora: Option<LoraConfig>,
    weight_quant: Option<QuantConfig>,
    activation_quant: Option<QuantConfig>,
}

impl LinearConfig {
    /// A projection from `in_features` to `out_features`, with bias.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
            initializer: Initializer::KaimingUniform { gain: 1.0 },
            bias_initializer: Initializer::Zeros,
            lora: None,
            weight_quant: None,
            activation_quant: None,
        }
    }

    /// Enable or disable the bias term.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Choose the weight initialiser.
    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    /// Choose the bias initialiser.
    pub fn with_bias_initializer(mut self, initializer: Initializer) -> Self {
        self.bias_initializer = initializer;
        self
    }

    /// Attach a LoRA adapter. The base weight is **not** frozen automatically; call
    /// [`Linear::freeze_base`] or [`Module::freeze_matching`] to do that.
    pub fn with_lora(mut self, lora: LoraConfig) -> Self {
        self.lora = Some(lora);
        self
    }

    /// Fake-quantize the weight before the matmul.
    pub fn with_weight_quant(mut self, config: QuantConfig) -> Self {
        self.weight_quant = Some(config);
        self
    }

    /// Fake-quantize the input activation before the matmul.
    pub fn with_activation_quant(mut self, config: QuantConfig) -> Self {
        self.activation_quant = Some(config);
        self
    }

    /// Input width.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output width.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Linear<R, E> {
        let weight = Param::new(self.initializer.init(
            vec![self.in_features, self.out_features],
            device,
            rng,
        ));
        let bias = self.bias.then(|| {
            Param::new(
                self.bias_initializer
                    .init(vec![self.out_features], device, rng),
            )
        });
        Linear {
            weight,
            bias,
            lora: self
                .lora
                .map(|cfg| cfg.init(self.in_features, self.out_features, device, rng)),
            weight_quant: self.weight_quant.map(Quantizer::new),
            activation_quant: self.activation_quant.map(Quantizer::new),
        }
    }
}

/// `y = x @ W (+ b)`, optionally low-rank adapted and fake-quantized.
pub struct Linear<R: Runtime, E: FloatElem> {
    weight: Param<R, E>,
    bias: Option<Param<R, E>>,
    lora: Option<LoraLinear<R, E>>,
    weight_quant: Option<Quantizer>,
    activation_quant: Option<Quantizer>,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Linear<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let shape = self.weight.shape();
        write!(
            f,
            "Linear({} -> {}, bias={}, lora={}, wquant={}, aquant={})",
            shape.dim(0),
            shape.dim(1),
            self.bias.is_some(),
            self.lora.is_some(),
            self.weight_quant.is_some(),
            self.activation_quant.is_some(),
        )
    }
}

impl<R: Runtime, E: FloatElem> Linear<R, E> {
    /// Input width.
    pub fn in_features(&self) -> usize {
        self.weight.shape().dim(0)
    }

    /// Output width.
    pub fn out_features(&self) -> usize {
        self.weight.shape().dim(1)
    }

    /// The `[in, out]` weight.
    pub fn weight(&self) -> &Param<R, E> {
        &self.weight
    }

    /// The bias, if present.
    pub fn bias(&self) -> Option<&Param<R, E>> {
        self.bias.as_ref()
    }

    /// The LoRA adapter, if attached.
    pub fn lora(&self) -> Option<&LoraLinear<R, E>> {
        self.lora.as_ref()
    }

    /// The weight quantizer, if attached.
    pub fn weight_quantizer(&self) -> Option<&Quantizer> {
        self.weight_quant.as_ref()
    }

    /// The activation quantizer, if attached.
    pub fn activation_quantizer(&self) -> Option<&Quantizer> {
        self.activation_quant.as_ref()
    }

    /// Freeze the base weight and bias, leaving only the adapter trainable.
    pub fn freeze_base(&self) {
        self.weight.freeze();
        if let Some(b) = &self.bias {
            b.freeze();
        }
    }

    /// Fold the LoRA adapter into the base weight. The adapter stays attached but
    /// contributes nothing until it is trained again.
    pub fn merge_lora(&self) -> Result<()> {
        match &self.lora {
            Some(lora) => lora.merge_into(&self.weight),
            None => Err(Error::config("this Linear has no LoRA adapter to merge")),
        }
    }

    /// Apply the projection to `[..., in_features]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        if input.shape().dim_from_end(0) != self.in_features() {
            return Err(Error::shape(format!(
                "Linear expects a trailing dimension of {}, got {}",
                self.in_features(),
                input.shape()
            )));
        }

        let x = match &self.activation_quant {
            Some(q) => q.quantize(input)?,
            None => input.clone(),
        };

        let weight = self.weight.var(input);
        let weight = match &self.weight_quant {
            Some(q) => q.quantize(&weight)?,
            None => weight,
        };

        let mut out = x.matmul(&weight)?;
        if let Some(lora) = &self.lora {
            out = out.add(&lora.delta(input)?)?;
        }
        if let Some(bias) = &self.bias {
            out = out.add(&bias.var(input))?;
        }
        Ok(out)
    }

    /// Replace the weight with an arbitrary tensor, e.g. when tying embeddings.
    pub fn set_weight(&self, weight: Tensor<R, E>) {
        self.weight.set(weight);
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Linear<R, E> {
    fn on_merge_lora(&self) -> Result<()> {
        match &self.lora {
            Some(lora) => lora.merge_into(&self.weight),
            None => Ok(()),
        }
    }

    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("weight", &self.weight);
        if let Some(bias) = &self.bias {
            visitor.param("bias", bias);
        }
        if let Some(lora) = &self.lora {
            visitor.child("lora", lora);
        }
        if let Some(q) = &self.weight_quant {
            visitor.child("weight_quant", q);
        }
        if let Some(q) = &self.activation_quant {
            visitor.child("activation_quant", q);
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for Linear<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}
