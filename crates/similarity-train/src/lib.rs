//! Fine-tuning for the record-similarity embedding model, on [Burn](https://burn.dev).
//!
//! This is the only crate in the workspace built with automatic differentiation and an optimizer. It trains the
//! same [`similarity_core::model::SentenceEncoder`] the server runs, through the same tokenizer and the same
//! weight files, so there is no second implementation of the model for training to disagree with.
//!
//! Modules, in dependency order:
//!
//! - [`error`]: the crate-wide error type.
//! - [`data`]: labeled pairs and catalog groups as training examples, and their batching.
//! - [`schedule`]: the learning-rate schedule.
//! - [`loss`]: the contrastive and ranking losses at truncated widths.
//! - [`eval`]: a progress check on labeled pairs.
//! - [`trainer`]: the training loop.

// Tests compute expected values with ordinary arithmetic and compare exact floats where exactness is the point.
#![cfg_attr(
  test,
  allow(clippy::arithmetic_side_effects, clippy::float_cmp, clippy::indexing_slicing)
)]

pub mod data;
pub mod error;
pub mod eval;
pub mod loss;
pub mod schedule;
pub mod trainer;

pub use error::{Result, TrainError};
pub use trainer::{Examples, Progress, Session, StepReport, TextBatch, TrainOptions, fit};
