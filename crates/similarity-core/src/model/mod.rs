//! The embedding model, in dependency order:
//!
//! - [`bert`]: the encoder architecture and its Hugging Face `config.json`.
//! - [`loader`]: strict safetensors loading with a post-load audit.
//! - [`sentence`]: the sentence-transformers pipeline around the encoder (pooling, optional projection).
//! - `onnx` (with the `onnx` feature): the same pipeline run by ONNX Runtime from an exported graph.

pub mod bert;
pub mod loader;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod sentence;

pub use bert::{BertConfig, BertModel, TrainingDropout};
pub use sentence::{EncoderWeights, Pipeline, Pooling, SentenceEncoder};
