//! The crate-wide error type.

use std::path::PathBuf;

/// `Result` with this crate's [`TrainError`] as the default error type.
pub type Result<T, E = TrainError> = std::result::Result<T, E>;

/// Everything that can go wrong while preparing data, training, or saving a model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrainError {
  /// The model could not be loaded, tokenized with, or saved.
  #[error(transparent)]
  Model(#[from] similarity_core::Error),

  /// A data file could not be read.
  #[error("i/o error at {path}: {source}")]
  Io {
    /// The file being read.
    path: PathBuf,
    /// The underlying failure.
    #[source]
    source: std::io::Error,
  },

  /// A line of a data file is not what its format requires.
  #[error("{path}, line {line}: {reason}")]
  Data {
    /// The data file.
    path: PathBuf,
    /// The one-based line number.
    line: usize,
    /// What is wrong with it.
    reason: String,
  },

  /// The options ask for something that cannot be trained.
  #[error("invalid training options: {0}")]
  Options(String),

  /// There is nothing to learn from: no usable example in any input.
  #[error("nothing to train on: {0}")]
  NoExamples(String),

  /// The loss or the gradient stopped being a finite number. The step is not taken: applying it would turn the
  /// weights into NaN, and everything written afterwards (a checkpoint, the model) would be garbage.
  #[error("training diverged at step {step}: the {quantity} is {value}")]
  Diverged {
    /// The one-based step at which it happened.
    step: usize,
    /// Which quantity went wrong: `loss` or `gradient norm`.
    quantity: &'static str,
    /// The offending value.
    value: f64,
  },

  /// A checkpoint could not be written, could not be read, or does not belong to this run.
  #[error("checkpoint {path}: {reason}")]
  Checkpoint {
    /// The checkpoint directory or file.
    path: PathBuf,
    /// What is wrong with it.
    reason: String,
  },

  /// The backend returned tensor data in an unexpected shape or dtype.
  #[error("backend error: {0}")]
  Backend(String),
}
