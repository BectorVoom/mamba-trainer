//! Feed-forward blocks.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::Result;
use crate::nn::dropout::Dropout;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::lora::LoraConfig;
use crate::nn::module::{Layer, Module, ModuleVisitor};
use crate::nn::quant::QuantConfig;
use crate::tensor::ops::random::Rng;

/// Nonlinearity used inside an MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    /// `x * sigmoid(x)`.
    Silu,
    /// Exact GELU.
    Gelu,
    /// `max(x, 0)`.
    Relu,
    /// `tanh(x)`.
    Tanh,
    /// Identity.
    Identity,
}

impl Activation {
    /// Apply the activation.
    pub fn apply<R: Runtime, E: FloatElem>(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        match self {
            Activation::Silu => x.silu(),
            Activation::Gelu => x.gelu(),
            Activation::Relu => Ok(x.relu()),
            Activation::Tanh => Ok(x.tanh()),
            Activation::Identity => Ok(x.clone()),
        }
    }
}

/// Configuration for [`Mlp`].
#[derive(Debug, Clone)]
pub struct MlpConfig {
    d_model: usize,
    hidden: usize,
    gated: bool,
    activation: Activation,
    bias: bool,
    dropout: f32,
    lora: Option<LoraConfig>,
    weight_quant: Option<QuantConfig>,
}

impl MlpConfig {
    /// A feed-forward block with the given widths. Defaults to a gated (SwiGLU)
    /// design, which is what the Mamba-3 hybrid stacks use.
    pub fn new(d_model: usize, hidden: usize) -> Self {
        Self {
            d_model,
            hidden,
            gated: true,
            activation: Activation::Silu,
            bias: false,
            dropout: 0.0,
            lora: None,
            weight_quant: None,
        }
    }

    /// Derive the hidden width from a multiplier, rounded to a multiple of 64.
    pub fn with_expansion(d_model: usize, expansion: f32) -> Self {
        let hidden = (((d_model as f32 * expansion) as usize).div_ceil(64)) * 64;
        Self::new(d_model, hidden)
    }

    /// Use a gated (SwiGLU-style) block instead of a plain two-layer MLP.
    pub fn with_gated(mut self, gated: bool) -> Self {
        self.gated = gated;
        self
    }

    /// Choose the nonlinearity.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Enable biases.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Dropout on the block output.
    pub fn with_dropout(mut self, p: f32) -> Self {
        self.dropout = p;
        self
    }

    /// Attach LoRA adapters to every projection.
    pub fn with_lora(mut self, lora: LoraConfig) -> Self {
        self.lora = Some(lora);
        self
    }

    /// Fake-quantize every weight.
    pub fn with_weight_quant(mut self, config: QuantConfig) -> Self {
        self.weight_quant = Some(config);
        self
    }

    fn linear(&self, d_in: usize, d_out: usize) -> LinearConfig {
        let mut cfg = LinearConfig::new(d_in, d_out).with_bias(self.bias);
        if let Some(lora) = self.lora {
            cfg = cfg.with_lora(lora);
        }
        if let Some(q) = self.weight_quant {
            cfg = cfg.with_weight_quant(q);
        }
        cfg
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(&self, device: &Device<R>, rng: &mut Rng) -> Mlp<R, E> {
        Mlp {
            up: self.linear(self.d_model, self.hidden).init(device, rng),
            gate: self
                .gated
                .then(|| self.linear(self.d_model, self.hidden).init(device, rng)),
            down: self.linear(self.hidden, self.d_model).init(device, rng),
            activation: self.activation,
            dropout: (self.dropout > 0.0).then(|| Dropout::new(self.dropout)),
        }
    }
}

/// A gated or plain feed-forward block.
pub struct Mlp<R: Runtime, E: FloatElem> {
    up: Linear<R, E>,
    gate: Option<Linear<R, E>>,
    down: Linear<R, E>,
    activation: Activation,
    dropout: Option<Dropout>,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Mlp<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mlp({} -> {} -> {}, gated={}, {:?})",
            self.up.in_features(),
            self.up.out_features(),
            self.down.out_features(),
            self.gate.is_some(),
            self.activation
        )
    }
}

impl<R: Runtime, E: FloatElem> Mlp<R, E> {
    /// Apply the block to `[..., d_model]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        let hidden = match &self.gate {
            Some(gate) => {
                let g = self.activation.apply(&gate.apply(input)?)?;
                g.mul(&self.up.apply(input)?)?
            }
            None => self.activation.apply(&self.up.apply(input)?)?,
        };
        let out = self.down.apply(&hidden)?;
        match &self.dropout {
            Some(d) => d.apply(&out),
            None => Ok(out),
        }
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Mlp<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("up", &self.up);
        if let Some(g) = &self.gate {
            visitor.child("gate", g);
        }
        visitor.child("down", &self.down);
        if let Some(d) = &self.dropout {
            visitor.child("dropout", d);
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for Mlp<R, E> {
    fn forward(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply(input)
    }
}
