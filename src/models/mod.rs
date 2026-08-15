//! The model zoo.
//!
//! Every model here is assembled from the same [`crate::nn`] blocks and the same
//! [`crate::ssm`] scan, so a feature added at the layer level (LoRA, fake
//! quantization, a new backend) reaches all of them at once.
//!
//! * [`mamba3`] — the token mixer and its residual block.
//! * [`hybrid`] — stacks that interleave Mamba-3 mixers with attention.
//! * [`lm`] — a causal language model over a hybrid stack.
//! * [`vision`] — bidirectional Mamba-3 for images.

pub mod hybrid;
pub mod lm;
pub mod mamba3;
pub mod vision;

pub use hybrid::{HybridConfig, HybridStack, LayerCache, LayerKind, LayerPattern};
pub use lm::{Mamba3Lm, Mamba3LmConfig};
pub use mamba3::{Mamba3Block, Mamba3BlockConfig, Mamba3Mixer, Mamba3MixerConfig, MixerCache};
pub use vision::{Pooling, ScanDirection, VisionMamba3, VisionMamba3Config};
