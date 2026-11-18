//! Parity with the reference PyTorch pipeline, on real weights.
//!
//! `tests/fixtures/golden_mdbr-leaf-mt.json` holds what sentence-transformers produces for a handful of texts
//! (`tools/golden.py` regenerates it). The Burn port must reproduce those vectors: this is the test that proves the
//! architecture, the key remapping, the pooling, the projection, and the tokenizer settings are all right at once,
//! which no shape test on random weights can.
//!
//! Ignored by default because it needs the 90 MB checkpoint: `just fetch-model && just test-parity`.
//!
//! The same check works for any other model directory, a fine-tuned one in particular. `just verify-model <dir>`
//! generates a fixture for it and points this test at both through `PARITY_MODEL_DIR` and `PARITY_FIXTURE`
//! (relative paths are resolved against the workspace root).

#![allow(
  clippy::unwrap_used,
  clippy::expect_used,
  clippy::panic,
  clippy::indexing_slicing,
  missing_docs
)]

use std::path::PathBuf;

use serde::Deserialize;
use similarity_core::vector::cosine;
use similarity_core::{Device, Embedder, EmbedderOptions};

#[derive(Deserialize)]
struct Golden {
  max_tokens: usize,
  dimension: usize,
  texts: Vec<String>,
  embeddings: Vec<Vec<f32>>,
}

fn workspace_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The path in `variable` if it is set (relative to the workspace root unless absolute), otherwise `default`.
fn path_from_env(variable: &str, default: PathBuf) -> PathBuf {
  std::env::var_os(variable).map_or(default, |path| workspace_root().join(path))
}

fn golden() -> Golden {
  let default = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_mdbr-leaf-mt.json");
  let path = path_from_env("PARITY_FIXTURE", default);
  serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn embedder(golden: &Golden, batch_size: usize, output_dim: Option<usize>) -> Embedder {
  let options = EmbedderOptions {
    max_tokens: golden.max_tokens,
    batch_size,
    output_dim,
    threads: None,
  };
  let dir = path_from_env("PARITY_MODEL_DIR", workspace_root().join("models/mdbr-leaf-mt"));
  Embedder::load(&dir, Device::flex(), &options).unwrap()
}

#[test]
#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-parity`"]
fn burn_reproduces_the_pytorch_embeddings() {
  let golden = golden();
  let embedder = embedder(&golden, 32, None);
  assert_eq!(embedder.dimension(), golden.dimension);

  let ours = embedder.embed(&golden.texts).unwrap();
  assert_eq!(ours.len(), golden.embeddings.len());
  for ((text, ours), theirs) in golden.texts.iter().zip(&ours).zip(&golden.embeddings) {
    let similarity = cosine(ours, theirs).unwrap();
    let worst = ours
      .iter()
      .zip(theirs)
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f32, f32::max);
    let shown: String = text.chars().take(40).collect();
    assert!(similarity > 0.99999, "cosine {similarity} for {shown:?}");
    assert!(worst < 2e-4, "max component error {worst} for {shown:?}");
  }
}

#[test]
#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-parity`"]
fn batch_composition_does_not_change_an_embedding() {
  let golden = golden();
  let together = embedder(&golden, 32, None).embed(&golden.texts).unwrap();
  let one_by_one = embedder(&golden, 1, None).embed(&golden.texts).unwrap();
  for (a, b) in together.iter().zip(&one_by_one) {
    assert!(cosine(a, b).unwrap() > 0.99999);
  }
}

#[test]
#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-parity`"]
fn truncated_embeddings_preserve_the_ranking_of_the_full_ones() {
  let golden = golden();
  let full = embedder(&golden, 32, None).embed(&golden.texts).unwrap();
  let short = embedder(&golden, 32, Some(256)).embed(&golden.texts).unwrap();
  assert_eq!(short[0].len(), 256);

  // "Blueberry Gummies 100mg" is nearer to the gummies category than to the live-resin category at either width.
  let nearer =
    |vectors: &[Vec<f32>]| cosine(&vectors[0], &vectors[2]).unwrap() > cosine(&vectors[0], &vectors[3]).unwrap();
  assert!(nearer(&full) && nearer(&short));
}

/// D21: a long text is tokenized a part at a time, and padding must not push the text out of what is read. Each of
/// these once did: two different products, padded alike, embedded to the same vector.
#[test]
#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-parity`"]
fn padding_a_text_does_not_hide_it_from_the_encoder() {
  let golden = golden();
  let embedder = embedder(&golden, 8, None);
  let first_part = golden.max_tokens * 64;
  let products = [
    "Blueberry Gummies 100mg (Indica)".to_owned(),
    "Sour Diesel Live Resin Cartridge 1g".to_owned(),
  ];
  let plain = embedder.embed(&products).unwrap();
  let apart = cosine(&plain[0], &plain[1]).unwrap();
  assert!(apart < 0.8, "the two products are not alike to begin with: {apart}");

  // Characters the tokenizer drops: the padded text is the text.
  for (name, unit) in [
    ("zero-width spaces", "\u{200b}"),
    ("word joiners", "\u{2060}"),
    ("combining accents", "\u{301}"),
  ] {
    let padding = unit.repeat(first_part);
    let padded: Vec<String> = products.iter().map(|product| format!("{padding}{product}")).collect();
    let vectors = embedder.embed(&padded).unwrap();
    for (padded, plain) in vectors.iter().zip(&plain) {
      let agreement = cosine(padded, plain).unwrap();
      assert!(agreement > 0.99999, "{name}: {agreement}");
    }
  }

  // Padding that can be seen is one unknown token in front of the text, and the text behind it still counts.
  let padding = format!("{} ", "a".repeat(first_part * 2));
  let padded: Vec<String> = products.iter().map(|product| format!("{padding}{product}")).collect();
  let vectors = embedder.embed(&padded).unwrap();
  let alike = cosine(&vectors[0], &vectors[1]).unwrap();
  assert!(alike < 0.9, "one unbroken word in front of both: {alike}");
}
