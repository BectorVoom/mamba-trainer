//! Error type of the crate.

use std::path::PathBuf;

/// Everything that can go wrong while loading, validating or exporting data.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// An I/O error, with the path involved.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A Parquet read/write error.
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// An Arrow error (casts, batch construction).
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// A JSON error (universe file).
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// A file has the wrong columns or types.
    #[error("schema: {0}")]
    Schema(String),
    /// The data violates a contract rule (e.g. dates not strictly increasing).
    #[error("invalid data: {0}")]
    Invalid(String),
}

impl DataError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Result alias.
pub type Result<T> = std::result::Result<T, DataError>;
