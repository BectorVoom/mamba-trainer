//! Device tensors and the raw (non-differentiable) kernel layer.
//!
//! Everything here is deliberately gradient-free: [`crate::autograd`] wraps these
//! primitives with backward rules. Keeping the split sharp means an inference-only
//! build can drop the autodiff module entirely, and new kernels only have to be
//! written once.

pub mod base;
pub mod ops;
pub mod shape;

pub use base::Tensor;
pub use shape::{MAX_RANK, Shape};
