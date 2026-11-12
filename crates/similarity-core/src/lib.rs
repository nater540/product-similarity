//! Record embedding and similarity scoring.
//!
//! This crate is the whole decision path with no I/O beyond reading (and, after fine-tuning, writing) a model
//! directory: it turns the text fields of a record into unit vectors with a BERT-class sentence encoder running on
//! [Burn](https://burn.dev), and scores a proposed revision of a record against a stored baseline. It knows nothing
//! about gRPC or Qdrant; the server and store crates sit on top of it, and `similarity-train` fine-tunes the encoder
//! it defines.
//!
//! Modules, in dependency order:
//!
//! - [`error`]: the crate-wide error type.
//! - [`vector`]: dot product, norm, cosine on plain slices.
//! - [`model`]: the encoder architecture, strict weight loading, and the sentence-transformers pipeline.
//! - [`tokenize`]: text to token ids, token ids to padded batches.
//! - [`embed`]: [`Embedder`], text in and unit vectors out.
//! - [`schema`]: the fields of a record, their weights, and the composite-vector layout.
//! - [`quantity`]: the numeric fields of a record, which are compared as numbers and never embedded.
//! - [`record`]: validating a record, reusing unchanged fields, and packing the stored form.
//! - [`provenance`]: the note a model directory carries about where it came from.
//! - [`rng`]: a small seeded generator, for results that must be reproducible from a seed.
//! - [`score`]: comparing a revision against its baseline and deciding whether to accept it.
//! - [`matching`]: scoring a listing from elsewhere against the stored records it might describe.

// Tests compute expected values with ordinary arithmetic and compare exact floats where exactness is the point.
#![cfg_attr(test, allow(clippy::arithmetic_side_effects, clippy::float_cmp))]

pub mod embed;
pub mod error;
pub mod matching;
pub mod model;
#[cfg(test)]
mod properties;
pub mod provenance;
pub mod quantity;
pub mod record;
pub mod rng;
pub mod schema;
pub mod score;
pub mod separation;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod tokenize;
pub mod vector;

pub use embed::{Embedder, EmbedderOptions};
pub use error::{Error, Result};
pub use provenance::Provenance;
pub use quantity::{Quantities, Quantity, QuantityChange, QuantityScore};
pub use record::{Plan, PreparedRecord, RecordVectors};
pub use schema::{FieldSpec, QuantitySpec, Schema};
pub use score::{Comparison, Evaluation, FieldChange, FieldScore, Policy, Reason};
pub use tokenize::TextTokenizer;

// Re-exported so that callers pick a device without depending on Burn themselves.
pub use burn::tensor::Device;
