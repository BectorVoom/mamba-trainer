//! Error types shared across the crate.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Every failure mode that is not a programming bug.
#[derive(Debug, Error)]
pub enum Error {
    /// A model / layer configuration is internally inconsistent.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A tensor operation received shapes that cannot be reconciled.
    #[error("shape error: {0}")]
    Shape(String),

    /// A tensor holds an unexpected rank.
    #[error("expected rank {expected}, got {got} (shape {shape:?})")]
    Rank {
        /// Rank the operation requires.
        expected: usize,
        /// Rank that was supplied.
        got: usize,
        /// Offending shape.
        shape: Vec<usize>,
    },

    /// Autodiff was asked for a gradient that does not exist.
    #[error("autodiff error: {0}")]
    Autodiff(String),

    /// Checkpoint / state-dict problem.
    #[error("state dict error: {0}")]
    StateDict(String),

    /// I/O failure while reading or writing a checkpoint.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// The requested feature is recognised but not implemented for this path.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl Error {
    /// Convenience constructor for [`Error::Config`].
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    /// Convenience constructor for [`Error::Shape`].
    pub fn shape(msg: impl Into<String>) -> Self {
        Error::Shape(msg.into())
    }
}
