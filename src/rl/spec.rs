//! How a policy reads a flat observation that is really a *set of entities*.
//!
//! The wire format stays one `float32[obs_dim]` per environment, so buffers,
//! collectors, environments and checkpoints never learn about structure. An
//! [`ObsSpec`] is the policy's reading of that vector:
//!
//! ```text
//! flat obs = [ globals (G) | set_1: N_1 × (F_1 + 1) | set_2: N_2 × (F_2 + 1) | ... ]
//!                                  └ per entity: F features, then 1 presence flag
//! obs_dim  = G + Σ_k N_k · (F_k + 1)
//! ```
//!
//! A fixed maximum `N_k` with a presence flag (`1` = the entity exists, `0` = an
//! empty slot) handles a variable number of entities without ragged tensors.
//!
//! [`ObsSpec::split`] cuts a `[B, T, obs_dim]` window into those parts on the
//! device in at most `1 + sets` launches — one fused split along the observation
//! axis, then one per set to peel the presence flag off — and never reads back.
//! [`ObsSpec::pack`] and [`ObsSpec::unpack`] are the host-side inverse, so an
//! environment never computes an offset by hand.

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use cubecl::prelude::Runtime;

/// One kind of entity in an observation: `count` slots of `features` values each,
/// every slot followed by its presence flag.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EntitySet {
    /// The set's name: the key encoders, pointer heads and parameter paths use.
    pub name: String,
    /// Slots in the set, the maximum number of entities it can hold.
    pub count: usize,
    /// Features per entity, not counting the presence flag.
    pub features: usize,
}

impl EntitySet {
    /// A set of `count` entities with `features` features each.
    pub fn new(name: impl Into<String>, count: usize, features: usize) -> Self {
        Self {
            name: name.into(),
            count,
            features,
        }
    }

    /// Width of the set on the wire: `count × (features + 1)`.
    pub fn width(&self) -> usize {
        self.count * (self.features + 1)
    }
}

/// The layout of a flat observation: `globals` leading values, then each set.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ObsSpec {
    /// Values that belong to no entity, at the front of the observation.
    pub globals: usize,
    /// The entity sets, in wire order.
    pub sets: Vec<EntitySet>,
}

/// One set of a split observation window.
pub struct EntityParts<R: Runtime, E: FloatElem> {
    /// `[B, T, N, F]`, the entities' features.
    pub features: Var<R, E>,
    /// `[B, T, N, 1]`, `1` where the slot holds an entity and `0` where it is empty.
    pub presence: Var<R, E>,
}

/// A `[B, T, obs_dim]` window cut into the parts an [`ObsSpec`] names.
pub struct SplitObs<R: Runtime, E: FloatElem> {
    /// `[B, T, G]`, or `None` when the spec has no globals.
    pub globals: Option<Var<R, E>>,
    /// One entry per set, in the spec's order.
    pub sets: Vec<EntityParts<R, E>>,
}

impl ObsSpec {
    /// A spec with `globals` leading values and the given sets.
    pub fn new(globals: usize, sets: Vec<EntitySet>) -> Self {
        Self { globals, sets }
    }

    /// Width of the flat observation this spec describes.
    pub fn obs_dim(&self) -> usize {
        self.globals + self.sets.iter().map(EntitySet::width).sum::<usize>()
    }

    /// The index of the set called `name`.
    pub fn set_index(&self, name: &str) -> Option<usize> {
        self.sets.iter().position(|s| s.name == name)
    }

    /// The set called `name`.
    pub fn set(&self, name: &str) -> Option<&EntitySet> {
        self.sets.iter().find(|s| s.name == name)
    }

    /// Where each set starts in the flat observation.
    pub fn offsets(&self) -> Vec<usize> {
        let mut at = self.globals;
        self.sets
            .iter()
            .map(|s| {
                let start = at;
                at += s.width();
                start
            })
            .collect()
    }

    /// Check the spec is usable, naming the offending field when it is not.
    pub fn validate(&self) -> Result<()> {
        if self.sets.is_empty() {
            return Err(Error::config(
                "obs_spec.sets is empty; a spec without entity sets is the flat policy, \
                 so leave obs_spec unset instead"
                    .to_string(),
            ));
        }
        for (i, set) in self.sets.iter().enumerate() {
            if set.name.is_empty() {
                return Err(Error::config(format!(
                    "obs_spec.sets[{i}].name is empty; every set needs a name"
                )));
            }
            if set.name.contains('.') {
                return Err(Error::config(format!(
                    "obs_spec.sets[{i}].name {:?} contains '.', which would break the \
                     parameter path entity.{{name}}.*",
                    set.name
                )));
            }
            if set.count == 0 {
                return Err(Error::config(format!(
                    "obs_spec.sets[{i}].count ({:?}) must be positive",
                    set.name
                )));
            }
            if set.features == 0 {
                return Err(Error::config(format!(
                    "obs_spec.sets[{i}].features ({:?}) must be positive",
                    set.name
                )));
            }
            if self.sets[..i].iter().any(|s| s.name == set.name) {
                return Err(Error::config(format!(
                    "obs_spec.sets[{i}].name {:?} is used twice",
                    set.name
                )));
            }
        }
        Ok(())
    }

    /// Cut a `[B, T, obs_dim]` window into globals and entity sets.
    ///
    /// Nothing is read back, and a traced input keeps its trace: the parts are
    /// differentiable views of `obs`.
    pub fn split<R: Runtime, E: FloatElem>(&self, obs: &Var<R, E>) -> Result<SplitObs<R, E>> {
        obs.shape().expect_rank(3)?;
        let dims = obs.dims();
        let (batch, seq) = (dims[0], dims[1]);
        if dims[2] != self.obs_dim() {
            return Err(Error::shape(format!(
                "obs_spec describes obs_dim={}, got {}",
                self.obs_dim(),
                obs.shape()
            )));
        }

        let mut sizes = Vec::with_capacity(self.sets.len() + 1);
        if self.globals > 0 {
            sizes.push(self.globals);
        }
        sizes.extend(self.sets.iter().map(EntitySet::width));
        // One band is the whole observation; `split` would drop the trace there.
        let mut bands = if sizes.len() == 1 {
            vec![obs.clone()]
        } else {
            obs.split(&sizes, 2)?
        }
        .into_iter();

        let globals = if self.globals > 0 { bands.next() } else { None };
        let sets = self
            .sets
            .iter()
            .zip(bands)
            .map(|(set, band)| {
                let slots = band.reshape(vec![batch, seq, set.count, set.features + 1])?;
                let mut parts = slots.split(&[set.features, 1], 3)?.into_iter();
                let (features, presence) = (parts.next(), parts.next());
                match (features, presence) {
                    (Some(features), Some(presence)) => Ok(EntityParts { features, presence }),
                    _ => Err(Error::shape(format!(
                        "splitting set {:?} did not produce features and presence",
                        set.name
                    ))),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SplitObs { globals, sets })
    }

    /// Build one flat observation from its parts, on the host.
    ///
    /// `sets[k]` is `(features, presence)` for the `k`-th set: `count × features`
    /// values row-major and `count` flags.
    pub fn pack(&self, globals: &[f32], sets: &[(&[f32], &[f32])]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.obs_dim()];
        self.pack_into(globals, sets, &mut out)?;
        Ok(out)
    }

    /// [`ObsSpec::pack`] into a caller's buffer of exactly `obs_dim` values.
    pub fn pack_into(
        &self,
        globals: &[f32],
        sets: &[(&[f32], &[f32])],
        out: &mut [f32],
    ) -> Result<()> {
        if out.len() != self.obs_dim() {
            return Err(Error::shape(format!(
                "pack needs a buffer of obs_dim={} values, got {}",
                self.obs_dim(),
                out.len()
            )));
        }
        if globals.len() != self.globals {
            return Err(Error::shape(format!(
                "pack: globals has {} values, the spec wants {}",
                globals.len(),
                self.globals
            )));
        }
        if sets.len() != self.sets.len() {
            return Err(Error::shape(format!(
                "pack: {} sets given, the spec has {}",
                sets.len(),
                self.sets.len()
            )));
        }
        out[..self.globals].copy_from_slice(globals);
        for ((set, (features, presence)), start) in self.sets.iter().zip(sets).zip(self.offsets()) {
            if features.len() != set.count * set.features {
                return Err(Error::shape(format!(
                    "pack: set {:?} has {} feature values, the spec wants {} × {}",
                    set.name,
                    features.len(),
                    set.count,
                    set.features
                )));
            }
            if presence.len() != set.count {
                return Err(Error::shape(format!(
                    "pack: set {:?} has {} presence flags, the spec wants {}",
                    set.name,
                    presence.len(),
                    set.count
                )));
            }
            let row = set.features + 1;
            let band = &mut out[start..start + set.width()];
            for (i, slot) in band.chunks_exact_mut(row).enumerate() {
                slot[..set.features]
                    .copy_from_slice(&features[i * set.features..(i + 1) * set.features]);
                slot[set.features] = presence[i];
            }
        }
        Ok(())
    }

    /// The inverse of [`ObsSpec::pack`]: `(globals, [(features, presence)])`.
    #[allow(clippy::type_complexity)] // Globals plus one (features, presence) pair per set.
    pub fn unpack(&self, obs: &[f32]) -> Result<(Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
        if obs.len() != self.obs_dim() {
            return Err(Error::shape(format!(
                "unpack needs obs_dim={} values, got {}",
                self.obs_dim(),
                obs.len()
            )));
        }
        let sets = self
            .sets
            .iter()
            .zip(self.offsets())
            .map(|(set, start)| {
                let band = &obs[start..start + set.width()];
                let mut features = Vec::with_capacity(set.count * set.features);
                let mut presence = Vec::with_capacity(set.count);
                for slot in band.chunks_exact(set.features + 1) {
                    features.extend_from_slice(&slot[..set.features]);
                    presence.push(slot[set.features]);
                }
                (features, presence)
            })
            .collect();
        Ok((obs[..self.globals].to_vec(), sets))
    }
}
