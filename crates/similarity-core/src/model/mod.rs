//! The embedding model, in dependency order:
//!
//! - [`bert`]: the encoder architecture and its Hugging Face `config.json`.
//! - [`loader`]: strict safetensors loading with a post-load audit.
//! - [`sentence`]: the sentence-transformers pipeline around the encoder (pooling, optional projection).

pub mod bert;
pub mod loader;
pub mod sentence;

pub use bert::{BertConfig, BertModel, TrainingDropout};
pub use sentence::{EncoderWeights, Pipeline, Pooling, SentenceEncoder};
