//! Checkpoints.
//!
//! A checkpoint is a [`StateDict`] plus enough metadata to resume: the step count
//! and a free-form JSON blob for the model configuration. Weights are stored as
//! `f32` regardless of the compute element type, so a run can switch precision
//! between sessions.

use std::path::Path;

use cubecl::prelude::Runtime;

use crate::backend::FloatElem;
use crate::error::Result;
use crate::nn::module::{Module, StateDict};

/// A saved training state.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    /// Optimizer steps completed.
    pub step: u64,
    /// Model weights.
    pub state: StateDict,
    /// Anything the caller wants to record, typically the model config.
    pub metadata: serde_json::Value,
}

impl Checkpoint {
    /// Snapshot a model.
    pub fn capture<R: Runtime, E: FloatElem, M: Module<R, E>>(model: &M, step: u64) -> Self {
        Self {
            step,
            state: model.state_dict(),
            metadata: serde_json::Value::Null,
        }
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
            metadata: self.metadata.clone(),
        }
    }

    /// Write as JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer(std::io::BufWriter::new(file), self)?;
        Ok(())
    }

    /// Read from JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(std::io::BufReader::new(file))?)
    }

    /// Restore weights into a model.
    ///
    /// `strict` requires the key sets to match exactly; pass `false` when loading a
    /// partial checkpoint such as LoRA adapters onto a base model.
    pub fn restore<R: Runtime, E: FloatElem, M: Module<R, E>>(
        &self,
        model: &M,
        strict: bool,
    ) -> Result<()> {
        model.load_state_dict(&self.state, strict)
    }

    /// Total number of scalars stored.
    pub fn num_values(&self) -> usize {
        self.state.entries.values().map(|e| e.data.len()).sum()
    }
}
