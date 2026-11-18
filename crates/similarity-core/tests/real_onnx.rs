//! The ONNX Runtime engine on real weights: it has to compute what PyTorch computes, and what Burn computes.
//!
//! Compiled only with the `onnx` feature, and ignored by default because it needs the 90 MB checkpoint and its
//! exported graph:
//!
//! ```text
//! just fetch-model && just export-onnx && just test-onnx
//! ```

#![cfg(feature = "onnx")]
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
use similarity_core::{Device, Embedder, EmbedderOptions, EngineChoice};

#[derive(Deserialize)]
struct Golden {
  max_tokens: usize,
  dimension: usize,
  texts: Vec<String>,
  embeddings: Vec<Vec<f32>>,
}

fn model_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/mdbr-leaf-mt")
}

fn golden() -> Golden {
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_mdbr-leaf-mt.json");
  serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn embedder(golden: &Golden, batch_size: usize, output_dim: Option<usize>, engine: EngineChoice) -> Embedder {
  let options = EmbedderOptions {
    max_tokens: golden.max_tokens,
    batch_size,
    output_dim,
    threads: None,
  };
  Embedder::load_with(&model_dir(), Device::flex(), &options, engine).unwrap()
}

fn worst(ours: &[f32], theirs: &[f32]) -> f32 {
  ours
    .iter()
    .zip(theirs)
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f32, f32::max)
}

/// The same fixture, and the same bounds, the Burn encoder is held to.
#[test]
#[ignore = "needs ./models/mdbr-leaf-mt and its onnx/model.onnx (`just export-onnx`); run with `just test-onnx`"]
fn onnx_runtime_reproduces_the_pytorch_embeddings() {
  let golden = golden();
  let embedder = embedder(&golden, 32, None, EngineChoice::Onnx);
  assert_eq!(embedder.engine(), EngineChoice::Onnx);
  assert_eq!(embedder.dimension(), golden.dimension);

  let ours = embedder.embed(&golden.texts).unwrap();
  for ((text, ours), theirs) in golden.texts.iter().zip(&ours).zip(&golden.embeddings) {
    let shown: String = text.chars().take(40).collect();
    let similarity = cosine(ours, theirs).unwrap();
    assert!(similarity > 0.99999, "cosine {similarity} for {shown:?}");
    assert!(
      worst(ours, theirs) < 2e-4,
      "max component error {} for {shown:?}",
      worst(ours, theirs)
    );
  }
}

/// Two engines, one model: the same fingerprint, because the same vectors. That is what lets a collection written
/// under one be read under the other.
#[test]
#[ignore = "needs ./models/mdbr-leaf-mt and its onnx/model.onnx (`just export-onnx`); run with `just test-onnx`"]
fn the_two_engines_give_the_same_vectors_under_the_same_fingerprint() {
  let golden = golden();
  for (batch_size, output_dim) in [(32, None), (1, None), (32, Some(256))] {
    let burn = embedder(&golden, batch_size, output_dim, EngineChoice::Burn);
    let onnx = embedder(&golden, batch_size, output_dim, EngineChoice::Onnx);
    assert_eq!(burn.fingerprint(), onnx.fingerprint());
    assert_eq!(
      (burn.dimension(), burn.max_tokens()),
      (onnx.dimension(), onnx.max_tokens())
    );
    let (ours, theirs) = (onnx.embed(&golden.texts).unwrap(), burn.embed(&golden.texts).unwrap());
    let mut largest = 0.0_f32;
    for (ours, theirs) in ours.iter().zip(&theirs) {
      assert_eq!(ours.len(), theirs.len());
      largest = largest.max(worst(ours, theirs));
    }
    // On the pinned model this is about 2e-7.
    assert!(
      largest < 1e-5,
      "batch {batch_size}, width {output_dim:?}: the engines differ by {largest:.1e}"
    );
  }
}

/// A graph exported from other weights is refused when the model is loaded, not discovered in production. The
/// case that matters is a directory whose weights changed after the export (a fine-tune written over it, a copy
/// of another model's `onnx/` folder). Here a copy of the model keeps its graph and has a stretch of its weights
/// replaced with another stretch of them: still a model Burn loads, and no longer the one the graph was made from.
#[test]
#[ignore = "needs ./models/mdbr-leaf-mt and its onnx/model.onnx (`just export-onnx`); run with `just test-onnx`"]
fn a_graph_that_is_not_this_models_is_refused_at_load() {
  let scratch = tempfile::tempdir().unwrap();
  let copy = scratch.path().join("model");
  std::fs::create_dir_all(&copy).unwrap();
  for entry in std::fs::read_dir(model_dir()).unwrap() {
    let entry = entry.unwrap();
    if entry.file_type().unwrap().is_file() {
      std::fs::copy(entry.path(), copy.join(entry.file_name())).unwrap();
    } else if entry.file_name() != "onnx" {
      let target = copy.join(entry.file_name());
      std::fs::create_dir_all(&target).unwrap();
      for inner in std::fs::read_dir(entry.path()).unwrap() {
        let inner = inner.unwrap();
        std::fs::copy(inner.path(), target.join(inner.file_name())).unwrap();
      }
    }
  }
  let options = EmbedderOptions::default();

  // No graph at all: the message says how to make one.
  let missing = Embedder::load_with(&copy, Device::flex(), &options, EngineChoice::Onnx).unwrap_err();
  assert!(missing.to_string().contains("just export-onnx"), "{missing}");

  // The real graph, beside the real weights: accepted.
  std::fs::create_dir_all(copy.join("onnx")).unwrap();
  std::fs::copy(model_dir().join("onnx/model.onnx"), copy.join("onnx/model.onnx")).unwrap();
  assert!(Embedder::load_with(&copy, Device::flex(), &options, EngineChoice::Onnx).is_ok());

  // The weights change and the graph does not. A megabyte of the encoder's last layers is overwritten with a
  // megabyte from earlier in the file: floats for floats, so the file is as valid as it was.
  let weights = copy.join("model.safetensors");
  let mut bytes = std::fs::read(&weights).unwrap();
  let (at, span) = (bytes.len() - 8 * 1024 * 1024, 1024 * 1024);
  let donor: Vec<u8> = bytes[at - 4 * span..at - 3 * span].to_vec();
  bytes[at..at + span].copy_from_slice(&donor);
  std::fs::write(&weights, &bytes).unwrap();
  assert!(
    Embedder::load_with(&copy, Device::flex(), &options, EngineChoice::Burn).is_ok(),
    "still a model"
  );
  let stale = Embedder::load_with(&copy, Device::flex(), &options, EngineChoice::Onnx).unwrap_err();
  assert!(
    stale
      .to_string()
      .contains("does not compute what the directory's weights compute"),
    "{stale}"
  );
}
