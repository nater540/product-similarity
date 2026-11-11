//! The crate-wide error type.
//!
//! Variants carry the path or field they are about, because the two ways this crate fails in practice are "a model
//! directory is not what the loader expects" and "a record does not fit the configured schema", and both are only
//! actionable when the message names the offending file or field.

use std::path::PathBuf;

/// `Result` with this crate's [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything that can go wrong while loading a model, embedding text, or scoring records.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// A file could not be read.
  #[error("i/o error at {path}: {source}")]
  Io {
    /// The file or directory being accessed.
    path: PathBuf,
    /// The underlying failure.
    #[source]
    source: std::io::Error,
  },

  /// A JSON file in the model directory did not parse.
  #[error("invalid JSON in {path}: {source}")]
  Json {
    /// The file being parsed.
    path: PathBuf,
    /// The underlying failure.
    #[source]
    source: serde_json::Error,
  },

  /// The model directory describes an architecture or pipeline this crate does not implement.
  #[error("unsupported model at {path}: {reason}")]
  UnsupportedModel {
    /// The file that declared the unsupported setting.
    path: PathBuf,
    /// What is unsupported, and what would be accepted instead.
    reason: String,
  },

  /// A checkpoint did not apply cleanly to the module tree.
  #[error("weight load failed for {path}: {reason}")]
  Weights {
    /// The `.safetensors` file.
    path: PathBuf,
    /// Which tensors failed, were missing, or found no home.
    reason: String,
  },

  /// The tokenizer could not be loaded or could not encode an input.
  #[error("tokenizer error: {0}")]
  Tokenizer(String),

  /// The backend returned tensor data in an unexpected shape or dtype.
  #[error("inference error: {0}")]
  Inference(String),

  /// The field schema is not usable (no fields, a non-positive weight, a threshold outside `[-1, 1]`, ...).
  #[error("invalid schema: {0}")]
  Schema(String),

  /// A record does not fit the schema.
  #[error("invalid record: {0}")]
  Record(String),

  /// Two vectors that must be comparable are not (different lengths, or produced by different models).
  #[error("incompatible vectors: {0}")]
  Incompatible(String),
}

impl Error {
  /// Wraps an I/O failure with the path it happened at.
  pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
    Self::Io {
      path: path.into(),
      source,
    }
  }

  /// Wraps a JSON failure with the file it happened in.
  pub(crate) fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
    Self::Json {
      path: path.into(),
      source,
    }
  }

  /// An unsupported architecture or pipeline setting found in `path`.
  pub(crate) fn unsupported(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
    Self::UnsupportedModel {
      path: path.into(),
      reason: reason.into(),
    }
  }

  /// A checkpoint that did not apply cleanly.
  pub(crate) fn weights(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
    Self::Weights {
      path: path.into(),
      reason: reason.into(),
    }
  }
}
