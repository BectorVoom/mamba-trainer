//! Reverse-mode automatic differentiation.
//!
//! Three ideas keep this layer small:
//!
//! 1. **Globally increasing node ids.** A node is always created after its inputs,
//!    so walking ids downwards is already a reverse topological order.
//! 2. **Optional tracking.** A [`Var`] without a tape is a bare tensor, so
//!    inference runs the same module code with no graph allocation at all.
//! 3. **Composition over fusion.** Only a couple of dozen primitives carry an
//!    adjoint; every higher-level op (softmax, RMS norm, the SSD scan, fake
//!    quantization) is expressed in terms of them and is therefore differentiable
//!    by construction.

pub mod grad_mode;
pub mod graph;
pub mod ops;
pub mod var;

pub use grad_mode::{NoGradGuard, enable_grad, no_grad, with_no_grad};
pub use graph::{Grads, Graph, NodeId, ParamId};
pub use ops::{cat, embedding, sum_all_vars};
pub use var::Var;
