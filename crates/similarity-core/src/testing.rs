//! Tiny models for tests: meaningless vectors, but real tokenization, batching, loading, and math.
//!
//! Compiled for this crate's own tests and, behind the `testing` feature, for the tests of crates built on this one,
//! so that nothing outside an `#[ignore]`d test ever needs the real 90 MB checkpoint.

// A broken fixture should stop a test on the spot, and this module is never part of a shipped binary.
#![allow(
  clippy::unwrap_used,
  clippy::missing_panics_doc,
  reason = "test support: panicking on a broken fixture is the intended behavior"
)]

use std::path::Path;

use burn::nn::LinearConfig;
use burn::tensor::Device;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;

use crate::embed::{Embedder, EmbedderOptions};
pub use crate::model::bert::testing::tiny_config;
use crate::model::loader::{Projection, save_encoder, save_projection};
use crate::model::{Pooling, SentenceEncoder};

/// The words the tiny tokenizer knows; everything else maps to `[UNK]`.
pub const WORDS: [&str; 12] = [
  "blueberry",
  "gummies",
  "vape",
  "cartridge",
  "edibles",
  "flower",
  "indica",
  "sativa",
  "hybrid",
  "resin",
  "mg",
  "wyld",
];

/// A word-level tokenizer with BERT's special tokens, built in memory.
///
/// Ids: `[PAD]` 0, `[UNK]` 1, `[CLS]` 2, `[SEP]` 3, then [`WORDS`] from 4.
pub fn tiny_tokenizer() -> Tokenizer {
  let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"];
  let vocab = specials
    .iter()
    .chain(WORDS.iter())
    .enumerate()
    .map(|(id, token)| ((*token).to_owned(), u32::try_from(id).unwrap()))
    .collect();
  let model = WordLevel::builder()
    .vocab(vocab)
    .unk_token("[UNK]".to_owned())
    .build()
    .unwrap();
  let processor = TemplateProcessing::builder()
    .try_single("[CLS] $A [SEP]")
    .unwrap()
    .special_tokens(vec![("[CLS]", 2), ("[SEP]", 3)])
    .build()
    .unwrap();
  let mut tokenizer = Tokenizer::new(model);
  tokenizer
    .with_pre_tokenizer(Some(Whitespace))
    .with_post_processor(Some(processor));
  tokenizer
}

/// A randomly initialized two-layer embedder with no projection.
pub fn tiny_embedder(options: &EmbedderOptions) -> Embedder {
  let device = Device::flex();
  let config = tiny_config();
  let encoder = SentenceEncoder::from_parts(config.init(&device), config, Pooling::Mean);
  Embedder::from_parts(tiny_tokenizer(), encoder, device, options).unwrap()
}

/// Writes a complete sentence-transformers directory around a randomly initialized tiny encoder.
///
/// The result loads through [`SentenceEncoder::load`] and [`Embedder::load`] exactly as a real model does: mean
/// pooling, the tiny tokenizer, and, if `projection` gives an output width, a bias-free `Dense` stage as the default
/// model has.
pub fn write_tiny_model(dir: &Path, projection: Option<usize>) {
  let device = Device::flex();
  let config = tiny_config();
  let write = |relative: &str, value: &serde_json::Value| {
    let path = dir.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
  };

  write(
    "config.json",
    &serde_json::json!({
      "model_type": "bert",
      "hidden_act": "gelu",
      "position_embedding_type": "absolute",
      "vocab_size": config.vocab_size,
      "hidden_size": config.hidden_size,
      "num_hidden_layers": config.num_hidden_layers,
      "num_attention_heads": config.num_attention_heads,
      "intermediate_size": config.intermediate_size,
      "max_position_embeddings": config.max_position_embeddings,
      "type_vocab_size": config.type_vocab_size,
      "layer_norm_eps": config.layer_norm_eps,
      "hidden_dropout_prob": config.hidden_dropout_prob,
      "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
    }),
  );
  let mut modules = vec![
    serde_json::json!({"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}),
    serde_json::json!({"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}),
  ];
  write(
    "1_Pooling/config.json",
    &serde_json::json!({"word_embedding_dimension": config.hidden_size, "pooling_mode_mean_tokens": true}),
  );
  save_encoder(&config.init(&device), &dir.join("model.safetensors")).unwrap();
  tiny_tokenizer().save(dir.join("tokenizer.json"), false).unwrap();

  if let Some(width) = projection {
    modules.push(
      serde_json::json!({"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"}),
    );
    write(
      "2_Dense/config.json",
      &serde_json::json!({
        "in_features": config.hidden_size,
        "out_features": width,
        "bias": false,
        "activation_function": "torch.nn.modules.linear.Identity",
      }),
    );
    let dense = Projection {
      linear: LinearConfig::new(config.hidden_size, width)
        .with_bias(false)
        .init(&device),
    };
    save_projection(&dense, &dir.join("2_Dense/model.safetensors")).unwrap();
  }
  write("modules.json", &serde_json::Value::Array(modules));
}
