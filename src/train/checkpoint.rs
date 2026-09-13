//! Checkpoints.
//!
//! A checkpoint is a [`StateDict`] plus enough metadata to resume: the step count
//! and a free-form JSON blob for the model configuration. Weights are stored as
//! `f32` regardless of the compute element type, so a run can switch precision
//! between sessions.
//!
//! # Binary format
//!
//! [`Checkpoint::save`] picks the format from the extension: `.json` writes the
//! plain JSON encoding this crate has always used (every existing caller and
//! fixture keeps working), anything else — `.m3ck` by convention — writes a
//! smaller binary container instead:
//!
//! ```text
//! magic      b"MAMBA3CK"      8 bytes
//! version    u32 LE           4 bytes, currently 2 (1 is still read)
//! header_len u32 LE           4 bytes
//! header     JSON             header_len bytes -- step, optimizer_steps,
//!                             metadata, and a {name, shape, offset, len}
//!                             descriptor per tensor, one list for the weights
//!                             and an optional second list for the optimizer
//!                             state
//! payload    f32 LE           tightly packed, in header order
//! ```
//!
//! # Counters
//!
//! Every counter — [`Checkpoint::step`], [`Checkpoint::optimizer_steps`], and
//! whatever integers a caller puts in [`Checkpoint::metadata`] — is a JSON
//! integer in both encodings, never a tensor value: the payload is `f32`, which
//! stops representing consecutive integers at `2^24`. Version 1 files (and JSON
//! written before this field existed) kept the optimizer's counter as a one-element
//! `f32` tensor under `__step_count__`; [`Checkpoint::restore_optimizer`] still
//! reads that, refusing a value that is not a non-negative integer.
//!
//! [`Checkpoint::load`] sniffs the magic rather than trusting the extension —
//! a `.m3ck` file handed the wrong bytes, or a `.json` file that happens to be
//! binary, is still read correctly, and a plain 8-byte mismatch falls back to
//! JSON rather than failing outright. The write itself goes through a temporary
//! sibling file and an atomic rename, so a process killed mid-write cannot
//! destroy the checkpoint that was already there.

use std::io::Write;
use std::path::{Path, PathBuf};

use cubecl::prelude::Runtime;

use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::nn::module::{Module, StateDict, TensorData};
use crate::train::optim::Optimizer;

const MAGIC: &[u8; 8] = b"MAMBA3CK";
const BINARY_VERSION: u32 = 2;
/// The oldest binary version [`Checkpoint::load`] still reads.
const OLDEST_BINARY_VERSION: u32 = 1;
/// Where version-1 checkpoints kept the optimizer's step counter, as an `f32`.
const LEGACY_STEP_COUNT_KEY: &str = "__step_count__";
const HEADER_PREFIX_LEN: usize = 8 + 4 + 4;

/// Where one tensor lives in the binary payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TensorSlot {
    name: String,
    shape: Vec<usize>,
    offset: u64,
    len: u64,
}

/// The binary container's header, everything but the raw floats.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BinaryHeader {
    step: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    optimizer_steps: Option<u64>,
    metadata: serde_json::Value,
    tensors: Vec<TensorSlot>,
    optimizer: Option<Vec<TensorSlot>>,
}

/// A saved training state.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    /// Optimizer steps completed.
    pub step: u64,
    /// Model weights.
    pub state: StateDict,
    /// Optimizer state (moment estimates, momentum, ...), keyed by the same
    /// parameter paths as `state`. `None` for a weights-only checkpoint —
    /// including every one written before this field existed, since serde
    /// defaults a missing key to `None` rather than failing to parse.
    ///
    /// Loading weights alone through [`Checkpoint::restore`] is a warm start:
    /// the optimizer re-warms its moment estimates from zero, which is a
    /// different (and worse, in the short run) trajectory than the one that
    /// would have continued without stopping. [`Checkpoint::restore_optimizer`]
    /// is what makes a resume indistinguishable from not having stopped.
    #[serde(default)]
    pub optimizer: Option<StateDict>,
    /// The optimizer's own step counter ([`Optimizer::step_count`]), exact.
    /// Its bias correction depends on it, and it is distinct from
    /// [`Checkpoint::step`], the trainer's (schedule) counter, although the two
    /// normally agree. `None` for a weights-only checkpoint, and for one written
    /// before this field existed — those carry an `f32` counter inside
    /// `optimizer` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_steps: Option<u64>,
    /// Anything the caller wants to record, typically the model config.
    pub metadata: serde_json::Value,
}

impl Checkpoint {
    /// Snapshot a model.
    pub fn capture<R: Runtime, E: FloatElem, M: Module<R, E>>(model: &M, step: u64) -> Self {
        Self {
            step,
            state: model.state_dict(),
            optimizer: None,
            optimizer_steps: None,
            metadata: serde_json::Value::Null,
        }
    }

    /// Attach optimizer state, keyed by the same parameter paths `model`'s
    /// own [`Module::named_parameters`] would give — which is why this takes
    /// `model` as well as `optimizer`, rather than the optimizer alone: the
    /// optimizer only knows parameters by their process-local id, and a
    /// checkpoint has to outlive the process that wrote it.
    pub fn with_optimizer<R: Runtime, E: FloatElem, M: Module<R, E>, O: Optimizer<R, E>>(
        mut self,
        model: &M,
        optimizer: &O,
    ) -> Self {
        self.optimizer = Some(optimizer.state_dict(&model.named_parameters()));
        self.optimizer_steps = Some(optimizer.step_count());
        self
    }

    /// Attach metadata, usually a serialised configuration.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Keep only weights whose path contains `pattern`.
    ///
    /// Shipping a LoRA-only checkpoint is `checkpoint.filtered("lora")`.
    pub fn filtered(&self, pattern: &str) -> Self {
        Self {
            step: self.step,
            state: self.state.filter(pattern),
            optimizer: self.optimizer.as_ref().map(|o| o.filter(pattern)),
            optimizer_steps: self.optimizer_steps,
            metadata: self.metadata.clone(),
        }
    }

    /// Write to `path`. A `.json` extension keeps the legacy plain-JSON
    /// encoding; anything else writes the smaller binary container (see the
    /// module docs). Either way the write is atomic: a temporary sibling file
    /// is renamed into place only once it is complete, so an interrupted write
    /// cannot destroy a valid checkpoint that was already there.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = if path.extension().and_then(|e| e.to_str()) == Some("json") {
            serde_json::to_vec(self)?
        } else {
            self.to_binary()?
        };
        write_atomically(path, &bytes)
    }

    /// Read from `path`, sniffing the format from its content rather than its
    /// extension: the binary magic decides, and anything else is parsed as
    /// JSON (which fails with its own clear error if it is neither).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() >= MAGIC.len() && bytes[..MAGIC.len()] == *MAGIC {
            Self::from_binary(&bytes)
        } else {
            Ok(serde_json::from_slice(&bytes)?)
        }
    }

    /// Encode as the binary container described in the module docs.
    fn to_binary(&self) -> Result<Vec<u8>> {
        let mut payload = Vec::new();
        let tensors = pack(&self.state, &mut payload);
        let optimizer = self.optimizer.as_ref().map(|o| pack(o, &mut payload));
        let header = BinaryHeader {
            step: self.step,
            optimizer_steps: self.optimizer_steps,
            metadata: self.metadata.clone(),
            tensors,
            optimizer,
        };
        let header_bytes = serde_json::to_vec(&header)?;
        let header_len: u32 = header_bytes.len().try_into().map_err(|_| {
            Error::StateDict("checkpoint header is too large to encode".to_string())
        })?;

        let mut out = Vec::with_capacity(HEADER_PREFIX_LEN + header_bytes.len() + payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&BINARY_VERSION.to_le_bytes());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Decode the binary container, validating the header and every tensor's
    /// bounds before any of it is trusted — a checkpoint is untrusted input
    /// the moment it has been read from a file rather than produced by
    /// [`Checkpoint::to_binary`] in this same process.
    fn from_binary(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_PREFIX_LEN {
            return Err(Error::StateDict("checkpoint is shorter than its own fixed header".to_string()));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if !(OLDEST_BINARY_VERSION..=BINARY_VERSION).contains(&version) {
            return Err(Error::Unsupported(format!(
                "checkpoint format version {version} is not supported by this build, \
                 which reads versions {OLDEST_BINARY_VERSION} to {BINARY_VERSION}"
            )));
        }
        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let header_end = HEADER_PREFIX_LEN.checked_add(header_len).ok_or_else(|| {
            Error::StateDict("checkpoint header length overflows".to_string())
        })?;
        if bytes.len() < header_end {
            return Err(Error::StateDict(
                "checkpoint is shorter than its own header claims".to_string(),
            ));
        }
        let header: BinaryHeader = serde_json::from_slice(&bytes[HEADER_PREFIX_LEN..header_end])?;
        let payload = &bytes[header_end..];

        // Every descriptor is checked for internal consistency and payload
        // bounds before any tensor is materialised, and all of them together
        // for overlap, so a truncated or adversarially crafted file is
        // rejected outright rather than read partway and then failing oddly.
        let mut spans: Vec<(u64, u64, &str)> = Vec::new();
        for slot in header.tensors.iter().chain(header.optimizer.iter().flatten()) {
            check_slot(slot, payload.len())?;
            spans.push((slot.offset, slot.len, &slot.name));
        }
        spans.sort_by_key(|&(offset, ..)| offset);
        for pair in spans.windows(2) {
            let (a_offset, a_len, a_name) = pair[0];
            let (b_offset, _, b_name) = pair[1];
            if a_offset + a_len > b_offset {
                return Err(Error::StateDict(format!(
                    "checkpoint tensors {a_name:?} and {b_name:?} overlap in the payload"
                )));
            }
        }
        let mut names = std::collections::HashSet::new();
        for slot in header.tensors.iter().chain(header.optimizer.iter().flatten()) {
            if !names.insert(slot.name.as_str()) {
                return Err(Error::StateDict(format!(
                    "checkpoint has more than one tensor named {:?}",
                    slot.name
                )));
            }
        }

        let state = StateDict {
            entries: header
                .tensors
                .iter()
                .map(|slot| (slot.name.clone(), read_slot(slot, payload)))
                .collect(),
        };
        let optimizer = header.optimizer.as_ref().map(|slots| StateDict {
            entries: slots.iter().map(|slot| (slot.name.clone(), read_slot(slot, payload))).collect(),
        });
        Ok(Self {
            step: header.step,
            state,
            optimizer,
            optimizer_steps: header.optimizer_steps,
            metadata: header.metadata,
        })
    }

    /// Restore weights into a model.
    ///
    /// `strict` requires the key sets to match exactly; pass `false` when loading a
    /// partial checkpoint such as LoRA adapters onto a base model. All or
    /// nothing: see [`Module::load_state_dict`].
    pub fn restore<R: Runtime, E: FloatElem, M: Module<R, E>>(
        &self,
        model: &M,
        strict: bool,
    ) -> Result<()> {
        model.load_state_dict(&self.state, strict)
    }

    /// Replace `optimizer`'s state with the state saved by
    /// [`Checkpoint::with_optimizer`] — see [`Optimizer::load_state_dict`] for
    /// the replacement and all-or-nothing guarantees, and for what `strict`
    /// refuses.
    ///
    /// An absent `self.optimizer` (a weights-only checkpoint) is an error under
    /// `strict`; without it, the optimizer is reset to a fresh one — no state, a
    /// zero step count — which is what a warm start from weights alone means.
    pub fn restore_optimizer<R: Runtime, E: FloatElem, M: Module<R, E>, O: Optimizer<R, E>>(
        &self,
        model: &M,
        optimizer: &mut O,
        strict: bool,
    ) -> Result<()> {
        let params = model.named_parameters();
        match &self.optimizer {
            Some(saved) => {
                let (tensors, steps) = self.optimizer_parts(saved)?;
                optimizer.load_state_dict(&params, &tensors, steps, strict)
            }
            None if strict => Err(Error::StateDict(
                "this checkpoint carries no optimizer state to restore; it was \
                 saved with Checkpoint::capture alone, or written before A2a"
                    .to_string(),
            )),
            None => optimizer.load_state_dict(&params, &StateDict::default(), None, false),
        }
    }

    /// Restore weights and optimizer state together, all or nothing: the
    /// weights are validated and staged, the optimizer is replaced (itself all
    /// or nothing), and only then are the staged weights written. An error at
    /// any point leaves both `model` and `optimizer` exactly as they were.
    ///
    /// `strict` applies to both halves, as in [`Checkpoint::restore`] and
    /// [`Checkpoint::restore_optimizer`]. Counters beyond the optimizer's own
    /// (the trainer's [`Checkpoint::step`], anything in the metadata) are the
    /// caller's to apply once this has succeeded.
    pub fn restore_training<R: Runtime, E: FloatElem, M: Module<R, E>, O: Optimizer<R, E>>(
        &self,
        model: &M,
        optimizer: &mut O,
        strict: bool,
    ) -> Result<RestoreReport> {
        let staged = model.stage_state_dict(&self.state, strict)?;
        self.restore_optimizer(model, optimizer, strict)?;
        let report = RestoreReport {
            weights: staged.is_complete(),
            optimizer: self.optimizer.is_some(),
        };
        staged.apply();
        Ok(report)
    }

    /// The optimizer's tensors and exact step counter, reading a version-1
    /// `f32` counter out of the tensors when that is all there is.
    fn optimizer_parts(&self, saved: &StateDict) -> Result<(StateDict, Option<u64>)> {
        let mut tensors = saved.clone();
        let legacy = tensors.entries.remove(LEGACY_STEP_COUNT_KEY);
        let steps = match (self.optimizer_steps, legacy) {
            (Some(exact), _) => Some(exact),
            (None, Some(counter)) => Some(legacy_counter(&counter)?),
            (None, None) => None,
        };
        Ok((tensors, steps))
    }

    /// Total number of scalars stored.    /// Total number of scalars stored.
    pub fn num_values(&self) -> usize {
        self.state.entries.values().map(|e| e.data.len()).sum()
    }
}

/// What [`Checkpoint::restore_training`] restored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreReport {
    /// Every one of the model's parameters had an entry (always true under
    /// `strict`).
    pub weights: bool,
    /// The checkpoint carried optimizer state and it was loaded. `false` means
    /// the optimizer was reset to a fresh one: a warm start.
    pub optimizer: bool,
}

/// A version-1 optimizer step counter: one `f32`, which must be a non-negative
/// integer to mean anything.
fn legacy_counter(counter: &TensorData) -> Result<u64> {
    let [value] = counter.data[..] else {
        return Err(Error::StateDict(format!(
            "the legacy optimizer step counter holds {} values, not one",
            counter.data.len()
        )));
    };
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(Error::StateDict(format!(
            "the legacy optimizer step counter is {value}, not a non-negative integer"
        )));
    }
    Ok(value as u64)
}

/// Append every entry of `dict` to `payload` as little-endian `f32`s, in its
/// existing (`BTreeMap`, so deterministic) key order, and describe where each
/// one landed.
fn pack(dict: &StateDict, payload: &mut Vec<u8>) -> Vec<TensorSlot> {
    dict.entries
        .iter()
        .map(|(name, tensor)| {
            let offset = (payload.len() / 4) as u64;
            for &v in &tensor.data {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            TensorSlot {
                name: name.clone(),
                shape: tensor.shape.clone(),
                offset,
                len: tensor.data.len() as u64,
            }
        })
        .collect()
}

/// Check one descriptor's internal consistency and payload bounds, before any
/// tensor is read from it.
fn check_slot(slot: &TensorSlot, payload_len: usize) -> Result<()> {
    let want: u64 = slot
        .shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
        .ok_or_else(|| Error::StateDict(format!("{:?}'s shape overflows", slot.name)))?;
    if want != slot.len {
        return Err(Error::StateDict(format!(
            "{:?} is shaped {:?} ({want} elements) but claims {} of them",
            slot.name, slot.shape, slot.len
        )));
    }
    let byte_len = slot
        .len
        .checked_mul(4)
        .ok_or_else(|| Error::StateDict(format!("{:?}'s length overflows", slot.name)))?;
    let byte_offset = slot
        .offset
        .checked_mul(4)
        .ok_or_else(|| Error::StateDict(format!("{:?}'s offset overflows", slot.name)))?;
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| Error::StateDict(format!("{:?}'s span overflows", slot.name)))?;
    if end > payload_len as u64 {
        return Err(Error::StateDict(format!(
            "{:?} extends past the end of the checkpoint's payload",
            slot.name
        )));
    }
    Ok(())
}

/// Read one already-validated descriptor's floats out of `payload`.
fn read_slot(slot: &TensorSlot, payload: &[u8]) -> TensorData {
    let start = (slot.offset * 4) as usize;
    let end = start + (slot.len * 4) as usize;
    let data = payload[start..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect();
    TensorData {
        shape: slot.shape.clone(),
        data,
    }
}

/// Write `bytes` to `path` through a temporary sibling file and an atomic
/// rename, so a process killed mid-write leaves whatever was at `path`
/// before untouched rather than a half-written checkpoint in its place.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let file_name = path.file_name().ok_or_else(|| {
        Error::StateDict(format!("{path:?} names no file to write"))
    })?;
    let tmp_name = {
        let mut n = std::ffi::OsString::from(".");
        n.push(file_name);
        n.push(format!(".{}.tmp", std::process::id()));
        n
    };
    let tmp_path: PathBuf = match dir {
        Some(dir) => dir.join(&tmp_name),
        None => PathBuf::from(&tmp_name),
    };

    let write_result = (|| -> Result<()> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    if let Err(err) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    Ok(())
}
