//! A shared encoder over a set of entities, and pooling that respects presence.
//!
//! An [`EntityEncoder`] is one small MLP applied to every entity of a set, so
//! "a dry strawberry two steps away" is learned once rather than once per slot:
//! the encoder is permutation equivariant, and [`masked_pool`] makes the pooled
//! summary permutation invariant. Empty slots — presence `0` — contribute
//! nothing to either, whatever their features hold.
//!
//! # Launch budget
//!
//! The trainer is bound by host submission, not arithmetic, so every piece here
//! is built to cost as few launches as possible and no host reads:
//!
//! * the presence statistics (mean weights, "any entity present") are plain
//!   tensors computed once per set, off the tape — they are constants;
//! * mean pooling is **one batched matrix product** `[B,T,1,N] @ [B,T,N,d]` of the
//!   normalised presence weights against the embeddings, instead of a multiply,
//!   a reduction and a divide;
//! * max pooling masks with the same fused `mask_logits` kernel the action
//!   masks use (absent entries become the most negative finite value, so no
//!   `BIG` constant can be out-ranked by a large embedding), then one reduction
//!   and one multiply that zeroes a set with no entity at all.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;
use crate::tensor::ops::{elemwise, reduce};

/// Configuration for an [`EntityEncoder`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EntityEncoderConfig {
    /// Hidden widths of the shared MLP, each followed by a ReLU. Empty means a
    /// single linear projection.
    #[serde(default = "EntityEncoderConfig::default_hidden")]
    pub hidden: Vec<usize>,
    /// Width of one entity's embedding.
    #[serde(default = "EntityEncoderConfig::default_d_entity")]
    pub d_entity: usize,
    /// Add a learned embedding per slot index. Off by default: it reintroduces
    /// per-slot parameters, which is what a shared encoder exists to avoid, but
    /// it lets position matter when it does (a fixed grid, say).
    #[serde(default)]
    pub slot_embedding: bool,
}

impl Default for EntityEncoderConfig {
    fn default() -> Self {
        Self {
            hidden: Self::default_hidden(),
            d_entity: Self::default_d_entity(),
            slot_embedding: false,
        }
    }
}

impl EntityEncoderConfig {
    fn default_hidden() -> Vec<usize> {
        vec![64]
    }

    fn default_d_entity() -> usize {
        64
    }

    /// An encoder with the given hidden widths and embedding width.
    pub fn new(hidden: Vec<usize>, d_entity: usize) -> Self {
        Self {
            hidden,
            d_entity,
            slot_embedding: false,
        }
    }

    /// Add a learned per-slot embedding.
    pub fn with_slot_embedding(mut self, on: bool) -> Self {
        self.slot_embedding = on;
        self
    }

    /// Check the widths are usable.
    pub fn validate(&self) -> Result<()> {
        if self.d_entity == 0 {
            return Err(Error::config(
                "entity encoder d_entity must be positive".to_string(),
            ));
        }
        if let Some(i) = self.hidden.iter().position(|&h| h == 0) {
            return Err(Error::config(format!(
                "entity encoder hidden[{i}] must be positive"
            )));
        }
        Ok(())
    }

    /// Instantiate for a set of `count` entities with `features` features each.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        features: usize,
        count: usize,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<EntityEncoder<R, E>> {
        self.validate()?;
        let mut widths = Vec::with_capacity(self.hidden.len() + 2);
        widths.push(features);
        widths.extend_from_slice(&self.hidden);
        widths.push(self.d_entity);
        let layers = widths
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device, rng))
            .collect();
        let slot = self.slot_embedding.then(|| {
            Param::new(
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init(vec![count, self.d_entity], device, rng),
            )
        });
        Ok(EntityEncoder {
            layers,
            slot,
            count,
            d_entity: self.d_entity,
        })
    }
}

/// One MLP shared by every entity of a set.
pub struct EntityEncoder<R: Runtime, E: FloatElem> {
    layers: Vec<Linear<R, E>>,
    slot: Option<Param<R, E>>,
    count: usize,
    d_entity: usize,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for EntityEncoder<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "EntityEncoder(layers={}, d_entity={}, slot={})",
            self.layers.len(),
            self.d_entity,
            self.slot.is_some()
        )
    }
}

impl<R: Runtime, E: FloatElem> EntityEncoder<R, E> {
    /// Width of one embedding.
    pub fn d_entity(&self) -> usize {
        self.d_entity
    }

    /// Embed `[..., N, F]` features to `[..., N, d_entity]`.
    ///
    /// `presence` (`[..., N, 1]`) zeroes an empty slot's features first, so what
    /// an environment leaves in an empty slot can never reach the output.
    pub fn apply(&self, features: &Var<R, E>, presence: &Var<R, E>) -> Result<Var<R, E>> {
        if features.shape().dim_from_end(1) != self.count {
            return Err(Error::shape(format!(
                "EntityEncoder expects {} entities, got {}",
                self.count,
                features.shape()
            )));
        }
        let mut x = features.mul(presence)?;
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.apply(&x)?;
            if i < last {
                x = x.relu();
            }
        }
        match &self.slot {
            Some(slot) => x.add(&slot.var(&x)),
            None => Ok(x),
        }
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EntityEncoder<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (i, layer) in self.layers.iter().enumerate() {
            visitor.child_at("mlp", i, layer);
        }
        if let Some(slot) = &self.slot {
            visitor.param("slot", slot);
        }
    }
}

/// A way of summarising a set of embeddings into one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolKind {
    /// The mean over present entities; zero when there are none.
    Mean,
    /// The elementwise maximum over present entities; zero when there are none.
    Max,
}

impl PoolKind {
    /// The name the configuration uses.
    pub fn name(self) -> &'static str {
        match self {
            PoolKind::Mean => "mean",
            PoolKind::Max => "max",
        }
    }

    /// Parse [`PoolKind::name`].
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "mean" => Ok(PoolKind::Mean),
            "max" => Ok(PoolKind::Max),
            other => Err(Error::config(format!(
                "unknown pooling kind {other:?}; expected 'mean' or 'max'"
            ))),
        }
    }
}

/// Which pools summarise each entity set.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PoolingConfig {
    /// The pools, concatenated in this order.
    pub kinds: Vec<PoolKind>,
}

impl Default for PoolingConfig {
    fn default() -> Self {
        Self {
            kinds: vec![PoolKind::Mean, PoolKind::Max],
        }
    }
}

impl PoolingConfig {
    /// Check the pools are usable.
    pub fn validate(&self) -> Result<()> {
        if self.kinds.is_empty() {
            return Err(Error::config(
                "pooling.kinds is empty; name at least one of 'mean' and 'max'".to_string(),
            ));
        }
        if let Some(i) = (1..self.kinds.len()).find(|&i| self.kinds[..i].contains(&self.kinds[i]))
        {
            return Err(Error::config(format!(
                "pooling.kinds[{i}] ({}) is listed twice",
                self.kinds[i].name()
            )));
        }
        Ok(())
    }
}

/// Presence statistics of one set, computed once and shared by every pool.
///
/// Constants: presence comes from the observation, which carries no gradient
/// worth having, so none of this is recorded on the tape.
pub struct Presence<R: Runtime, E: FloatElem> {
    /// `[..., N, 1]`, the flags as given.
    pub flags: Tensor<R, E>,
    /// `[..., 1, N]`, `flag / max(1, Σ flags)`: the mean-pooling weights.
    pub mean_weights: Tensor<R, E>,
    /// `[..., 1, 1]`, `1` when the set holds at least one entity.
    pub any: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> Presence<R, E> {
    /// Statistics of `[..., N, 1]` presence flags.
    pub fn new(presence: &Var<R, E>) -> Result<Self> {
        let flags = presence.tensor().clone();
        let rank = flags.rank();
        if rank < 2 || flags.shape().dim(rank - 1) != 1 {
            return Err(Error::shape(format!(
                "presence must be [..., N, 1], got {}",
                flags.shape()
            )));
        }
        let count = reduce::sum_dim(&flags, rank - 2)?;
        let any = elemwise::clamp_max(&count, 1.0);
        let weights = elemwise::div(&flags, &elemwise::clamp_min(&count, 1.0))?;
        let mut dims = flags.dims().to_vec();
        dims.swap(rank - 2, rank - 1);
        Ok(Self {
            mean_weights: weights.reshape(dims)?,
            flags,
            any,
        })
    }
}

/// Pool `[..., N, d]` embeddings into one `[..., d]` vector per pool kind.
pub fn pool_parts<R: Runtime, E: FloatElem>(
    embeddings: &Var<R, E>,
    presence: &Presence<R, E>,
    kinds: &[PoolKind],
) -> Result<Vec<Var<R, E>>> {
    let rank = embeddings.rank();
    let mut out_dims = embeddings.dims().to_vec();
    out_dims.remove(rank - 2);
    kinds
        .iter()
        .map(|kind| {
            let pooled = match kind {
                PoolKind::Mean => Var::constant(presence.mean_weights.clone()).matmul(embeddings)?,
                PoolKind::Max => embeddings
                    .mask_logits(&presence.flags)?
                    .max_dim(rank - 2)?
                    .mul(&Var::constant(presence.any.clone()))?,
            };
            pooled.reshape(out_dims.clone())
        })
        .collect()
}

/// Pool `[..., N, d]` embeddings under `[..., N, 1]` presence into
/// `[..., kinds.len() · d]`, the pools concatenated in `kinds` order.
pub fn masked_pool<R: Runtime, E: FloatElem>(
    embeddings: &Var<R, E>,
    presence: &Var<R, E>,
    kinds: &[PoolKind],
) -> Result<Var<R, E>> {
    if kinds.is_empty() {
        return Err(Error::config("masked_pool needs at least one pool kind"));
    }
    let parts = pool_parts(embeddings, &Presence::new(presence)?, kinds)?;
    let last = parts[0].rank() - 1;
    crate::autograd::cat(&parts, last)
}
