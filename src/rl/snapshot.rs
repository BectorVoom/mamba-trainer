//! Exact continuation: everything a collection loop carries between windows.
//!
//! A learner checkpoint that holds weights, optimizer moments and counters
//! restores *training* exactly, but not the run: the next window would start from
//! a fresh environment, a zeroed recurrent state and a restarted action-draw
//! schedule, so the rounds after a restore differ from the rounds that would have
//! followed without one. This module is the rest of the state:
//!
//! | namespace | what | where it is saved |
//! |---|---|---|
//! | collector | the observation the next window starts from, the last termination flags, each lane's in-progress return, the last window's completed-episode totals, the draw seed, the draw counter, the temperature, the mask column's width | [`CollectorState`] |
//! | engine | every layer's recurrent state (hidden, trapezoidal carry, rotation angle, convolution history) and the step counter | [`CollectorState`] |
//! | reference | the frozen reference's own carried cache | [`RolloutSnapshot::reference_cache`] |
//! | reference weights | the frozen reference's parameters, so the reference can be rebuilt from the file ([`ReferencePolicy::from_weights`]) and a different one is refused | [`RolloutSnapshot::reference_weights`], written to [`Checkpoint::reference`] |
//! | environment | whatever [`VecEnv::save_state`] returns, opaque | [`RolloutSnapshot::env`] |
//!
//! # The boundary
//!
//! A snapshot is taken *between* windows. What a window leaves behind for its own
//! update — the trajectory buffer's contents, a prepared batch — is not part of it:
//! a caller saves after updating on a window, never between collecting it and
//! updating on it. The Python learners enforce that.
//!
//! # Atomicity
//!
//! [`RolloutSnapshot::stage`] validates every tensor against the live collector
//! (and the live reference's weights against the saved ones) and uploads it
//! without touching the collector; [`StagedRollout::apply`] then calls
//! the environment's [`VecEnv::load_state`] — the one fallible step with an effect,
//! and one every environment in this crate performs all or nothing — and only if
//! that succeeds swaps the staged state in, which cannot fail.

use std::collections::BTreeMap;

use cubecl::prelude::Runtime;
use serde_json::{Value, json};

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::MixerCache;
use crate::nn::module::{StateDict, TensorData};
use crate::ssm::scan::SsmState;
use crate::tensor::Tensor;
use crate::train::Checkpoint;

use super::collect::{Collector, StagedCollector};
use super::env::VecEnv;
use super::ppo::ReferencePolicy;

/// `metadata.rollout.version` this build writes and reads.
pub const ROLLOUT_VERSION: u64 = 1;

/// Key prefix of the collector's tensors in [`Checkpoint::rollout`].
const COLLECTOR_PREFIX: &str = "collector.";
/// Key prefix of the reference cache's tensors in [`Checkpoint::rollout`].
const REFERENCE_PREFIX: &str = "reference.";
/// Key of the environment's bytes in [`Checkpoint::blobs`].
const ENV_BLOB: &str = "env";

// ---------------------------------------------------------------------------
// Versioned byte layouts for environment state
// ---------------------------------------------------------------------------

/// Builds the byte layout an environment's [`VecEnv::save_state`] returns: an
/// eight-byte tag naming the environment type, a `u32` layout version, then
/// fields in a fixed order, every one little-endian and every slice
/// length-prefixed.
///
/// A layout is opaque to everything but the environment that wrote it; the tag
/// and version are what let that environment refuse bytes that are not its own,
/// before it has changed anything.
pub struct StateWriter {
    bytes: Vec<u8>,
}

impl StateWriter {
    /// Start a layout tagged `tag`, version `version`.
    pub fn new(tag: &[u8; 8], version: u32) -> Self {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(tag);
        bytes.extend_from_slice(&version.to_le_bytes());
        Self { bytes }
    }

    /// Append a `u32`.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a `u64`.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a length-prefixed slice of `u32`s.
    pub fn u32s(&mut self, values: &[u32]) -> &mut Self {
        self.u64(values.len() as u64);
        for v in values {
            self.u32(*v);
        }
        self
    }

    /// Append a length-prefixed slice of `f32`s, bit for bit.
    pub fn f32s(&mut self, values: &[f32]) -> &mut Self {
        self.u64(values.len() as u64);
        for v in values {
            self.bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self
    }

    /// Append a length-prefixed byte string.
    pub fn bytes(&mut self, values: &[u8]) -> &mut Self {
        self.u64(values.len() as u64);
        self.bytes.extend_from_slice(values);
        self
    }

    /// The finished layout.
    pub fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }
}

/// Reads a layout written by [`StateWriter`], refusing — with a message naming
/// what was wrong — bytes with another tag, another version, a truncated field or
/// anything left over.
pub struct StateReader<'a> {
    bytes: &'a [u8],
    at: usize,
    what: &'static str,
}

impl<'a> StateReader<'a> {
    /// Check the tag and version and position after them. `what` names the
    /// environment in error messages.
    pub fn open(bytes: &'a [u8], tag: &[u8; 8], version: u32, what: &'static str) -> Result<Self> {
        let mut reader = Self { bytes, at: 0, what };
        let found = reader.take(8)?;
        if found != tag {
            return Err(Error::StateDict(format!(
                "these bytes are not a saved {what} state (tag {:?}, expected {:?})",
                String::from_utf8_lossy(found),
                String::from_utf8_lossy(tag)
            )));
        }
        let found = reader.u32()?;
        if found != version {
            return Err(Error::StateDict(format!(
                "saved {what} state has layout version {found}; this build reads {version}"
            )));
        }
        Ok(reader)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len());
        let Some(end) = end else {
            return Err(Error::StateDict(format!(
                "saved {} state is truncated: {} bytes, needed more than {}",
                self.what,
                self.bytes.len(),
                self.at
            )));
        };
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    /// Read a `u32`.
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    /// Read a `u64`.
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn len(&mut self, width: usize) -> Result<usize> {
        let len = self.u64()?;
        let bytes = len.checked_mul(width as u64);
        match bytes {
            Some(bytes) if bytes <= (self.bytes.len() - self.at) as u64 => Ok(len as usize),
            _ => Err(Error::StateDict(format!(
                "saved {} state claims a field of {len} elements, more than it holds",
                self.what
            ))),
        }
    }

    /// Read a length-prefixed slice of `u32`s.
    pub fn u32s(&mut self) -> Result<Vec<u32>> {
        let len = self.len(4)?;
        (0..len).map(|_| self.u32()).collect()
    }

    /// Read a length-prefixed slice of `f32`s.
    pub fn f32s(&mut self) -> Result<Vec<f32>> {
        let len = self.len(4)?;
        (0..len).map(|_| self.u32().map(f32::from_bits)).collect()
    }

    /// Read a length-prefixed byte string.
    pub fn bytes(&mut self) -> Result<Vec<u8>> {
        let len = self.len(1)?;
        Ok(self.take(len)?.to_vec())
    }

    /// Check that a field read back is the one this environment was built with.
    pub fn expect(&self, field: &str, saved: u64, live: u64) -> Result<()> {
        if saved == live {
            return Ok(());
        }
        Err(Error::StateDict(format!(
            "saved {} state has {field} = {saved}, but this one has {live}",
            self.what
        )))
    }

    /// Refuse trailing bytes.
    pub fn finish(self) -> Result<()> {
        if self.at == self.bytes.len() {
            return Ok(());
        }
        Err(Error::StateDict(format!(
            "saved {} state has {} unread trailing bytes",
            self.what,
            self.bytes.len() - self.at
        )))
    }
}

/// The refusal [`VecEnv::load_state`]'s default returns.
pub fn unsupported_env_state() -> Error {
    Error::Unsupported(
        "this environment does not support exact resume: implement save_state() and \
         load_state() on it, or save at the 'optimizer' level"
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Recurrent caches as host tensors
// ---------------------------------------------------------------------------

fn host<R: Runtime, E: FloatElem>(
    prefix: &str,
    name: &str,
    var: &Var<R, E>,
) -> Result<(String, TensorData)> {
    Ok((
        format!("{prefix}{name}"),
        TensorData {
            shape: var.dims().to_vec(),
            data: var.try_to_f32()?,
        },
    ))
}

/// Every layer of `cache` as `f32` host tensors under `prefix`, one read each.
pub(crate) fn cache_to_entries<R: Runtime, E: FloatElem>(
    prefix: &str,
    cache: &[MixerCache<R, E>],
) -> Result<Vec<(String, TensorData)>> {
    let mut out = Vec::new();
    for (index, layer) in cache.iter().enumerate() {
        out.push(host(prefix, &format!("{index}.h"), &layer.ssm.h)?);
        out.push(host(prefix, &format!("{index}.last_u"), &layer.ssm.last_u)?);
        if let Some(angle) = &layer.ssm.angle {
            out.push(host(prefix, &format!("{index}.angle"), angle)?);
        }
        if let Some(conv) = &layer.conv {
            out.push(host(prefix, &format!("{index}.conv"), conv)?);
        }
    }
    Ok(out)
}

/// Upload one saved tensor, refusing a missing entry or a shape other than
/// `template`'s.
pub(crate) fn upload_like<R: Runtime, E: FloatElem>(
    entries: &StateDict,
    key: &str,
    template: &[usize],
    device: &Device<R>,
) -> Result<Tensor<R, E>> {
    let entry = entries
        .entries
        .get(key)
        .ok_or_else(|| Error::StateDict(format!("the saved rollout state has no `{key}`")))?;
    if entry.shape != template || entry.data.len() != template.iter().product::<usize>() {
        return Err(Error::StateDict(format!(
            "the saved rollout state's `{key}` is shaped {:?} with {} values, but this \
             learner's is {template:?}",
            entry.shape,
            entry.data.len()
        )));
    }
    Tensor::from_f32(&entry.data, template.to_vec(), device)
}

/// Rebuild a cache shaped like `template` from entries under `prefix`, uploading
/// fresh tensors. Refuses a missing, extra-optional or mis-shaped layer before
/// returning anything.
pub(crate) fn cache_from_entries<R: Runtime, E: FloatElem>(
    prefix: &str,
    entries: &StateDict,
    template: &[MixerCache<R, E>],
    device: &Device<R>,
) -> Result<Vec<MixerCache<R, E>>> {
    let layers = entries
        .entries
        .keys()
        .filter_map(|k| k.strip_prefix(prefix))
        .filter_map(|rest| rest.split('.').next()?.parse::<usize>().ok())
        .map(|i| i + 1)
        .max()
        .unwrap_or(0);
    if layers != template.len() {
        return Err(Error::StateDict(format!(
            "the saved rollout state holds {layers} {}layers of recurrent state, but this \
             policy has {}",
            prefix.trim_end_matches('.').to_string() + " ",
            template.len()
        )));
    }
    let optional =
        |index: usize, name: &str, like: Option<&Var<R, E>>| -> Result<Option<Var<R, E>>> {
            let key = format!("{prefix}{index}.{name}");
            match like {
                Some(like) => Ok(Some(Var::constant(upload_like(
                    entries,
                    &key,
                    like.dims(),
                    device,
                )?))),
                None if entries.entries.contains_key(&key) => Err(Error::StateDict(format!(
                    "the saved rollout state has `{key}`, which this policy's layers do not carry"
                ))),
                None => Ok(None),
            }
        };
    template
        .iter()
        .enumerate()
        .map(|(index, like)| {
            Ok(MixerCache {
                ssm: SsmState {
                    h: Var::constant(upload_like(
                        entries,
                        &format!("{prefix}{index}.h"),
                        like.ssm.h.dims(),
                        device,
                    )?),
                    last_u: Var::constant(upload_like(
                        entries,
                        &format!("{prefix}{index}.last_u"),
                        like.ssm.last_u.dims(),
                        device,
                    )?),
                    angle: optional(index, "angle", like.ssm.angle.as_ref())?,
                },
                conv: optional(index, "conv", like.conv.as_ref())?,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The snapshot itself
// ---------------------------------------------------------------------------

/// The integers and tensors a [`Collector`] carries between windows.
///
/// Produced by [`Collector::export_state`]; see the module docs for the list.
#[derive(Debug, Clone)]
pub struct CollectorState {
    /// `f32` tensors: `observation` (absent before the first window),
    /// `last_done`, `running_return`, `episode_return_sum`,
    /// `episode_return_count`, and `engine.<layer>.<h|last_u|angle|conv>`.
    pub tensors: StateDict,
    /// Environments, steps per window and observation width it was saved from.
    pub envs: usize,
    /// See [`CollectorState::envs`].
    pub steps: usize,
    /// See [`CollectorState::envs`].
    pub obs_dim: usize,
    /// The action-draw seed.
    pub seed: u64,
    /// Draws taken so far: with the seed, the whole draw schedule.
    pub draws: u64,
    /// Sampling temperature, bit for bit.
    pub temperature: f32,
    /// Policy steps the rollout engine has taken.
    pub engine_steps: u64,
    /// The trajectory buffer's legal-action mask width, if it has one.
    pub mask_width: Option<usize>,
}

/// Everything [`crate::rl`] needs to continue a run exactly, apart from the
/// weights and optimizer a [`Checkpoint`] already carries.
#[derive(Debug, Clone)]
pub struct RolloutSnapshot {
    /// The collector and its rollout engine.
    pub collector: CollectorState,
    /// The reference policy's carried cache: `None` when the run has no
    /// reference, `Some(None)` when it has one that has scored nothing since its
    /// last reset.
    pub reference_cache: Option<Option<StateDict>>,
    /// The reference policy's weights, present whenever `reference_cache` is:
    /// what [`RolloutSnapshot::stage`] checks a live reference against, and
    /// what [`ReferencePolicy::from_weights`] rebuilds one from. `None` also in
    /// a checkpoint written before weights were saved, which is staged without
    /// that check.
    pub reference_weights: Option<StateDict>,
    /// The environment's own bytes, from [`VecEnv::save_state`].
    pub env: Vec<u8>,
}

impl RolloutSnapshot {
    /// Capture a collector, its reference and its environment. Host reads: this
    /// is a synchronisation, and belongs between windows.
    ///
    /// Fails with [`Error::Unsupported`] when the environment has no
    /// [`VecEnv::save_state`].
    pub fn capture<R: Runtime, E: FloatElem, V: VecEnv<R, E> + ?Sized>(
        collector: &Collector<'_, R, E>,
        reference: Option<&ReferencePolicy<R, E>>,
        env: &V,
    ) -> Result<Self> {
        let env = env.save_state()?.ok_or_else(unsupported_env_state)?;
        let reference_weights = reference.map(ReferencePolicy::weights);
        let reference_cache = reference
            .map(|reference| {
                reference
                    .cache()
                    .map(|cache| -> Result<StateDict> {
                        Ok(StateDict {
                            entries: cache_to_entries("", cache)?.into_iter().collect(),
                        })
                    })
                    .transpose()
            })
            .transpose()?;
        Ok(Self {
            collector: collector.export_state()?,
            reference_cache,
            reference_weights,
            env,
        })
    }

    /// Write this snapshot into `checkpoint`: tensors into
    /// [`Checkpoint::rollout`], the reference's weights into
    /// [`Checkpoint::reference`], the environment's bytes into
    /// [`Checkpoint::blobs`], and every integer, exactly, into
    /// `metadata.rollout`. `checkpoint.metadata` must be an object or null.
    pub fn attach(self, mut checkpoint: Checkpoint) -> Result<Checkpoint> {
        let c = &self.collector;
        let mut entries = BTreeMap::new();
        for (name, tensor) in &c.tensors.entries {
            entries.insert(format!("{COLLECTOR_PREFIX}{name}"), tensor.clone());
        }
        if let Some(Some(cache)) = &self.reference_cache {
            for (name, tensor) in &cache.entries {
                entries.insert(format!("{REFERENCE_PREFIX}{name}"), tensor.clone());
            }
        }
        let rollout = json!({
            "version": ROLLOUT_VERSION,
            "envs": c.envs,
            "steps": c.steps,
            "obs_dim": c.obs_dim,
            "seed": c.seed,
            "draws": c.draws,
            "temperature": c.temperature,
            "temperature_bits": c.temperature.to_bits(),
            "engine_steps": c.engine_steps,
            "mask_width": c.mask_width,
            "reference": match &self.reference_cache {
                None => Value::Null,
                Some(None) => json!("empty"),
                Some(Some(_)) => json!("cached"),
            },
        });
        match &mut checkpoint.metadata {
            Value::Object(map) => {
                map.insert("rollout".to_string(), rollout);
            }
            Value::Null => checkpoint.metadata = json!({ "rollout": rollout }),
            _ => {
                return Err(Error::StateDict(
                    "a checkpoint's metadata must be an object to carry rollout state".to_string(),
                ));
            }
        }
        checkpoint.rollout = Some(StateDict { entries });
        if let Some(weights) = self.reference_weights {
            checkpoint.reference = Some(weights);
        }
        checkpoint.blobs.insert(ENV_BLOB.to_string(), self.env);
        Ok(checkpoint)
    }

    /// Read a snapshot back out of `checkpoint`, or `None` if it carries none.
    /// Validates the layout and every integer; shapes are checked against a live
    /// collector by [`RolloutSnapshot::stage`].
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Option<Self>> {
        let Some(meta) = checkpoint.metadata.get("rollout") else {
            if checkpoint.rollout.is_some() || checkpoint.blobs.contains_key(ENV_BLOB) {
                return Err(Error::StateDict(
                    "the checkpoint carries rollout tensors but no metadata.rollout to \
                     describe them"
                        .to_string(),
                ));
            }
            return Ok(None);
        };
        let int = |field: &str| -> Result<u64> {
            meta.get(field).and_then(Value::as_u64).ok_or_else(|| {
                Error::StateDict(format!(
                    "the checkpoint's metadata.rollout.{field} is not a non-negative integer"
                ))
            })
        };
        let version = int("version")?;
        if version != ROLLOUT_VERSION {
            return Err(Error::Unsupported(format!(
                "rollout state version {version} is not supported; this build reads \
                 {ROLLOUT_VERSION}"
            )));
        }
        let bits = int("temperature_bits")?;
        let temperature_bits = u32::try_from(bits).map_err(|_| {
            Error::StateDict(format!(
                "metadata.rollout.temperature_bits {bits} is not a u32"
            ))
        })?;
        let mask_width = match meta.get("mask_width") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_u64().ok_or_else(|| {
                Error::StateDict(format!("metadata.rollout.mask_width {v} is not an integer"))
            })? as usize),
        };
        let tensors = checkpoint.rollout.as_ref().ok_or_else(|| {
            Error::StateDict("metadata.rollout is present but the rollout tensors are not".into())
        })?;
        let mut collector = BTreeMap::new();
        let mut reference = BTreeMap::new();
        for (name, tensor) in &tensors.entries {
            if let Some(rest) = name.strip_prefix(COLLECTOR_PREFIX) {
                collector.insert(rest.to_string(), tensor.clone());
            } else if let Some(rest) = name.strip_prefix(REFERENCE_PREFIX) {
                reference.insert(rest.to_string(), tensor.clone());
            } else {
                return Err(Error::StateDict(format!(
                    "unexpected rollout tensor `{name}`"
                )));
            }
        }
        let reference_cache = match meta.get("reference").and_then(Value::as_str) {
            None if meta.get("reference").is_none_or(Value::is_null) => {
                if !reference.is_empty() {
                    return Err(Error::StateDict(
                        "rollout tensors for a reference cache, but no reference recorded".into(),
                    ));
                }
                None
            }
            Some("empty") if reference.is_empty() => Some(None),
            Some("cached") if !reference.is_empty() => Some(Some(StateDict { entries: reference })),
            other => {
                return Err(Error::StateDict(format!(
                    "metadata.rollout.reference is {other:?}, which does not match the \
                     reference tensors present"
                )));
            }
        };
        // The weights belong to the rollout only when it was saved with a
        // reference; a checkpoint may carry them for its training state alone.
        let reference_weights = reference_cache
            .as_ref()
            .and_then(|_| checkpoint.reference.clone());
        let env = checkpoint.blobs.get(ENV_BLOB).cloned().ok_or_else(|| {
            Error::StateDict("the checkpoint's rollout state has no environment bytes".into())
        })?;
        Ok(Some(Self {
            collector: CollectorState {
                tensors: StateDict { entries: collector },
                envs: int("envs")? as usize,
                steps: int("steps")? as usize,
                obs_dim: int("obs_dim")? as usize,
                seed: int("seed")?,
                draws: int("draws")?,
                temperature: f32::from_bits(temperature_bits),
                engine_steps: int("engine_steps")?,
                mask_width,
            },
            reference_cache,
            reference_weights,
            env,
        }))
    }

    /// Validate this snapshot against a live collector (and reference) and upload
    /// it, changing neither.
    ///
    /// A live reference whose weights differ from
    /// [`RolloutSnapshot::reference_weights`] is refused: its scores would not be
    /// the ones the saved run went on to compute. Checking reads the live
    /// reference's weights back to the host.
    pub fn stage<R: Runtime, E: FloatElem>(
        &self,
        collector: &Collector<'_, R, E>,
        reference: Option<&ReferencePolicy<R, E>>,
    ) -> Result<StagedRollout<R, E>> {
        let staged_collector = collector.stage_state(&self.collector)?;
        if let (Some(live), Some(saved)) = (reference, &self.reference_weights)
            && live.fingerprint() != saved.fingerprint()
        {
            return Err(Error::StateDict(
                "this learner's reference policy has different weights from the one the \
                 rollout was saved with; rebuild it from the checkpoint with \
                 ReferencePolicy::from_weights"
                    .to_string(),
            ));
        }
        let reference = match (reference, &self.reference_cache) {
            (None, None) => None,
            (Some(_), None) => {
                return Err(Error::StateDict(
                    "this learner has a reference policy but the checkpoint's rollout state \
                     was saved without one"
                        .to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::StateDict(
                    "the checkpoint's rollout state carries a reference cache but this \
                     learner has no reference policy"
                        .to_string(),
                ));
            }
            (Some(_), Some(None)) => Some(None),
            (Some(live), Some(Some(cache))) => {
                let device = collector.buffer().device();
                let template = live
                    .policy()
                    .empty_state(self.collector.envs, device)
                    .snapshot();
                Some(Some(cache_from_entries("", cache, &template, device)?))
            }
        };
        Ok(StagedRollout {
            collector: staged_collector,
            reference,
            env: self.env.clone(),
        })
    }
}

/// A validated, uploaded [`RolloutSnapshot`], ready to swap in.
pub struct StagedRollout<R: Runtime, E: FloatElem> {
    collector: StagedCollector<R, E>,
    reference: Option<Option<Vec<MixerCache<R, E>>>>,
    env: Vec<u8>,
}

impl<R: Runtime, E: FloatElem> StagedRollout<R, E> {
    /// The environment's saved bytes, for a caller that restores the environment
    /// itself (the Python bindings, which hold it behind a borrow).
    pub fn env_bytes(&self) -> &[u8] {
        &self.env
    }

    /// Keep a live seed and temperature rather than the saved ones; see
    /// [`StagedCollector::keep_sampling`].
    pub fn keep_sampling(&mut self, seed: u64, temperature: f32) {
        self.collector.keep_sampling(seed, temperature);
    }

    /// Restore the environment, then — only if that succeeded — swap the staged
    /// collector and reference state in. The swap cannot fail.
    pub fn apply<V: VecEnv<R, E> + ?Sized>(
        self,
        collector: &mut Collector<'_, R, E>,
        reference: Option<&mut ReferencePolicy<R, E>>,
        env: &mut V,
    ) -> Result<()> {
        env.load_state(&self.env)?;
        self.apply_without_env(collector, reference);
        Ok(())
    }

    /// The swap half of [`StagedRollout::apply`], for a caller that has already
    /// restored the environment from [`StagedRollout::env_bytes`].
    pub fn apply_without_env(
        self,
        collector: &mut Collector<'_, R, E>,
        reference: Option<&mut ReferencePolicy<R, E>>,
    ) {
        self.collector.apply(collector);
        if let (Some(reference), Some(cache)) = (reference, self.reference) {
            reference.restore_cache(cache);
        }
    }
}
