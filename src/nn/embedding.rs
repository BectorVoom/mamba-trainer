//! Token and positional embeddings.

use cubecl::prelude::Runtime;

use crate::autograd::{Var, embedding as gather};
use crate::backend::{Device, FloatElem};
use crate::error::Result;
use crate::nn::init::Initializer;
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::Rng;

/// Configuration for [`Embedding`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingConfig {
    num_embeddings: usize,
    dim: usize,
    init: Initializer,
}

impl EmbeddingConfig {
    /// A `[num_embeddings, dim]` lookup table.
    pub fn new(num_embeddings: usize, dim: usize) -> Self {
        Self {
            num_embeddings,
            dim,
            init: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            },
        }
    }

    /// Choose the initialiser.
    pub fn with_initializer(mut self, init: Initializer) -> Self {
        self.init = init;
        self
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Embedding<R, E> {
        Embedding {
            weight: Param::new(self.init.init(
                vec![self.num_embeddings, self.dim],
                device,
                rng,
            )),
        }
    }
}

/// A learnable lookup table.
pub struct Embedding<R: Runtime, E: FloatElem> {
    weight: Param<R, E>,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Embedding<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = self.weight.shape();
        write!(f, "Embedding({} x {})", s.dim(0), s.dim(1))
    }
}

impl<R: Runtime, E: FloatElem> Embedding<R, E> {
    /// Number of rows.
    pub fn num_embeddings(&self) -> usize {
        self.weight.shape().dim(0)
    }

    /// Row width.
    pub fn dim(&self) -> usize {
        self.weight.shape().dim(1)
    }

    /// The table.
    pub fn weight(&self) -> &Param<R, E> {
        &self.weight
    }

    /// Look up ids of shape `[...]`, producing `[..., dim]`.
    ///
    /// `anchor` decides whether the result is tracked; pass any value already on
    /// the tape, or [`Var::traced`] on a dummy when embeddings are the graph root.
    pub fn apply(&self, ids: &IdTensor<R>, anchor: &Var<R, E>) -> Result<Var<R, E>> {
        gather(&self.weight.var(anchor), ids)
    }

    /// Look up ids, starting a fresh tape rooted at the table.
    pub fn lookup(&self, ids: &IdTensor<R>, train: bool) -> Result<Var<R, E>> {
        let anchor = if train {
            Var::traced(Tensor::<R, E>::zeros(vec![1], ids.device()))
        } else {
            Var::constant(Tensor::<R, E>::zeros(vec![1], ids.device()))
        };
        self.apply(ids, &anchor)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Embedding<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("weight", &self.weight);
    }
}

/// A learnable absolute position table, used by the vision models.
pub struct PositionalEmbedding<R: Runtime, E: FloatElem> {
    weight: Param<R, E>,
}

impl<R: Runtime, E: FloatElem> PositionalEmbedding<R, E> {
    /// A `[max_len, dim]` table initialised with small noise.
    pub fn new(max_len: usize, dim: usize, device: &Device<R>, rng: &mut Rng) -> Self {
        Self {
            weight: Param::new(
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init(vec![max_len, dim], device, rng),
            ),
        }
    }

    /// The table.
    pub fn weight(&self) -> &Param<R, E> {
        &self.weight
    }

    /// Add positions `0..len` to `[batch, len, dim]`.
    pub fn add_to(&self, input: &Var<R, E>, offset: usize) -> Result<Var<R, E>> {
        let len = input.shape().dim(1);
        let slice = self
            .weight
            .var(input)
            .slice(0, offset, len)?
            .unsqueeze(0)?;
        input.add(&slice)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for PositionalEmbedding<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("weight", &self.weight);
    }
}
