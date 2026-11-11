//! Parity with PyTorch for training, on real weights.
//!
//! `tests/fixtures/training_mdbr-leaf-mt.json` records a few optimizer steps taken by PyTorch on fixed batches
//! with dropout off (`tools/train_golden.py` regenerates it). Replaying them here must give:
//!
//! - the same loss at every step: the forward pass in training mode and both loss functions;
//! - the same global gradient norm at every step: the entire backward pass, since one wrong partial derivative
//!   anywhere in the encoder changes it;
//! - the same embeddings afterwards: the optimizer's update rule and the gradient clipping, since each later step
//!   starts from the weights the earlier ones produced.
//!
//! A training bug does not crash; it produces a model that is quietly worse. This test is what stands between
//! that and a fine-tuned model being trusted.
//!
//! Ignored by default because it needs the 90 MB checkpoint: `just fetch-model && just test-parity`.

#![allow(
  clippy::unwrap_used,
  clippy::expect_used,
  clippy::panic,
  clippy::indexing_slicing,
  // With `--nocapture` the test prints the measured agreement, which is what a tolerance should be judged against.
  clippy::print_stderr,
  missing_docs
)]

use std::path::PathBuf;

use serde::Deserialize;
use similarity_core::model::SentenceEncoder;
use similarity_core::tokenize::pad_batch;
use similarity_core::vector::cosine;
use similarity_core::{Device, TextTokenizer};
use similarity_train::data::{GroupExample, PairExample};
use similarity_train::{Session, TextBatch, TrainOptions};

#[derive(Deserialize)]
struct Pair {
  before: String,
  after: String,
  same: bool,
}

#[derive(Deserialize)]
struct Group {
  catalog: String,
  listing: String,
}

#[derive(Deserialize)]
struct Step {
  kind: String,
  loss: f64,
  grad_norm: f64,
}

#[derive(Deserialize)]
struct Reference {
  max_tokens: usize,
  dims: Vec<usize>,
  margin: f64,
  scale: f64,
  learning_rate: f64,
  max_grad_norm: f64,
  pairs: Vec<Pair>,
  groups: Vec<Group>,
  steps: Vec<Step>,
  probes: Vec<String>,
  probes_before: Vec<Vec<f32>>,
  probes_after: Vec<Vec<f32>>,
}

fn workspace_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference() -> Reference {
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/training_mdbr-leaf-mt.json");
  serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The raw (not normalized) encoder output for each text, one text per forward pass.
fn embed(encoder: &SentenceEncoder, tokenizer: &TextTokenizer, device: &Device, texts: &[String]) -> Vec<Vec<f32>> {
  texts
    .iter()
    .map(|text| {
      let ids = tokenizer.encode(text).unwrap();
      let (ids, padding) = pad_batch(&[ids.as_slice()], device).unwrap();
      encoder.forward(ids, padding).into_data().try_into_vec::<f32>().unwrap()
    })
    .collect()
}

fn difference(after: &[f32], before: &[f32]) -> Vec<f32> {
  after.iter().zip(before).map(|(a, b)| a - b).collect()
}

fn length(vector: &[f32]) -> f32 {
  vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

#[test]
#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-parity`"]
fn burn_reproduces_pytorch_training_step_for_step() {
  let reference = reference();
  let dir = workspace_root().join("models/mdbr-leaf-mt");
  let device = Device::flex();
  let encoder = SentenceEncoder::load(&dir, &device).unwrap();
  let tokenizer = TextTokenizer::load(&dir, reference.max_tokens).unwrap();

  let before = embed(&encoder, &tokenizer, &device, &reference.probes);
  for (found, expected) in before.iter().zip(&reference.probes_before) {
    assert!(cosine(found, expected).unwrap() > 0.99999, "the starting point differs");
  }

  let pairs: Vec<PairExample> = reference
    .pairs
    .iter()
    .map(|pair| PairExample {
      before: pair.before.clone(),
      after: pair.after.clone(),
      same: pair.same,
    })
    .collect();
  let groups: Vec<GroupExample> = reference
    .groups
    .iter()
    .map(|group| GroupExample {
      catalog: group.catalog.clone(),
      listing: group.listing.clone(),
    })
    .collect();

  let options = TrainOptions {
    widths: reference.dims.clone(),
    max_grad_norm: Some(reference.max_grad_norm),
    dropout: false,
    margin: reference.margin,
    scale: reference.scale,
    ..TrainOptions::default()
  };
  let (mut session, mut encoder) = Session::begin(encoder, tokenizer.clone(), device.clone(), &options).unwrap();

  let relative = |found: f64, expected: f64| (found - expected).abs() / expected.abs().max(1e-12);
  let (mut clipped, mut unclipped) = (0, 0);
  for (index, step) in reference.steps.iter().enumerate() {
    let batch = match step.kind.as_str() {
      "pairs" => TextBatch::Pairs(pairs.iter().collect()),
      "groups" => TextBatch::Groups(groups.iter().collect()),
      other => panic!("unknown step kind `{other}`"),
    };
    let (updated, report) = session.step(encoder, &batch, reference.learning_rate).unwrap();
    encoder = updated;
    eprintln!(
      "step {index} ({:6}): loss {:.6} vs {:.6}, gradient norm {:.6} vs {:.6}",
      step.kind, report.loss, step.loss, report.grad_norm, step.grad_norm
    );
    // Each step starts from the weights the previous ones left, so the tolerance has to absorb float rounding
    // compounded through the optimizer. Measured agreement is around one part in 100,000; a real defect is off by
    // tens of percent.
    assert!(relative(report.loss, step.loss) < 1e-3, "loss at step {index}");
    assert!(
      relative(report.grad_norm, step.grad_norm) < 1e-3,
      "gradient norm at step {index}"
    );
    if step.grad_norm > reference.max_grad_norm {
      clipped += 1;
    } else {
      unclipped += 1;
    }
  }
  assert!(
    clipped > 0 && unclipped > 0,
    "the fixture should exercise both sides of the clipping limit"
  );

  let encoder = Session::finish(encoder);
  let after = embed(&encoder, &tokenizer, &device, &reference.probes);
  for (index, text) in reference.probes.iter().enumerate() {
    let agreement = cosine(&after[index], &reference.probes_after[index]).unwrap();
    // Agreement on where the embedding ended up is not enough: it barely moved from where both started. The
    // displacement itself must point the same way and be the same size.
    let moved = difference(&after[index], &before[index]);
    let expected = difference(&reference.probes_after[index], &reference.probes_before[index]);
    let direction = cosine(&moved, &expected).unwrap();
    let size = length(&moved) / length(&expected);
    eprintln!("probe {text:?}: cosine {agreement:.6}, displacement cosine {direction:.4}, size ratio {size:.4}");
    assert!(agreement > 0.9999, "{text}: {agreement}");
    assert!(direction > 0.999, "{text}: displacement direction {direction}");
    assert!((size - 1.0).abs() < 0.01, "{text}: displacement size ratio {size}");
  }
}
