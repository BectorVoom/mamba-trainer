//! Neural-network building blocks.
//!
//! Every layer follows the same three conventions, which is what makes the model
//! zoo above it small:
//!
//! * a **`XxxConfig` builder** describes the layer; `Config::init(&device, &mut rng)`
//!   materialises it. Configs are plain data and can be serialised.
//! * **[`Module::visit`]** is the single place a layer declares its parameters and
//!   children; named parameters, freezing, mode switching and checkpoints all fall
//!   out of it.
//! * **[`Param`]** is a shared handle, so optimizers, LoRA merging and weight tying
//!   never need mutable access to the model tree.

pub mod attention;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod init;
pub mod linear;
pub mod lora;
pub mod mlp;
pub mod module;
pub mod norm;
pub mod param;
pub mod quant;
pub mod rope;

pub use attention::{AttentionCache, AttentionConfig, MultiHeadAttention};
pub use conv::{CausalConv1d, CausalConv1dConfig, PatchEmbed, PatchEmbedConfig, shift_by};
pub use dropout::Dropout;
pub use embedding::{Embedding, EmbeddingConfig, PositionalEmbedding};
pub use init::Initializer;
pub use linear::{Linear, LinearConfig};
pub use lora::{LoraConfig, LoraLinear};
pub use mlp::{Activation, Mlp, MlpConfig};
pub use module::{Layer, Module, ModuleVisitor, Sequential, StateDict, TensorData};
pub use norm::{LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig};
pub use param::Param;
pub use quant::{QuantConfig, QuantScheme, Quantizer};
pub use rope::{RotaryEmbedding, rotate_halves};

pub use crate::autograd::ParamId;
