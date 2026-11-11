//! How well a model separates edits from recycles on labeled pairs, field text by field text.
//!
//! This is a training-time progress check, not the calibration: it scores single field texts at one width, with no
//! field weights, floors, or thresholds. `record-similarity calibrate` on held-out pairs remains the measurement
//! that decides whether a fine-tuned model is better.

use similarity_core::model::SentenceEncoder;
use similarity_core::tokenize::pad_batch;
use similarity_core::{Device, TextTokenizer};

use crate::data::PairExample;
use crate::error::{Result, TrainError};

/// The cosine similarities of one label's examples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spread {
  /// How many examples.
  pub count: usize,
  /// The lowest similarity.
  pub min: f32,
  /// The mean similarity.
  pub mean: f32,
  /// The highest similarity.
  pub max: f32,
}

impl Spread {
  fn of(values: &[f32]) -> Option<Self> {
    let min = values.iter().copied().reduce(f32::min)?;
    let max = values.iter().copied().reduce(f32::max)?;
    #[expect(clippy::cast_precision_loss, reason = "example counts are far below 2^24")]
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    Some(Self {
      count: values.len(),
      min,
      mean,
      max,
    })
  }
}

/// Where edits and recycles sit on the similarity scale at one embedding width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Separation {
  /// The embedding width the similarities were computed at.
  pub width: usize,
  /// Examples from ordinary edits, if there are any.
  pub edits: Option<Spread>,
  /// Examples from recycled listings, if there are any.
  pub recycles: Option<Spread>,
  /// How often an edit scores above a recycle, a tie counting half: 1.0 is perfectly separated, 0.5 is not at
  /// all. It needs no threshold, which is what makes it fit for choosing between models. `None` unless both
  /// labels are present.
  pub auc: Option<f64>,
}

impl Separation {
  /// The mean similarity of edits minus the mean similarity of recycles: higher means better separated.
  ///
  /// The means are used rather than the extremes because this is tracked across a training run, and a single odd
  /// example should not decide whether the run looks like progress.
  pub fn mean_gap(&self) -> Option<f32> {
    Some(self.edits?.mean - self.recycles?.mean)
  }
}

/// Embeds `texts` with a plain (not training) encoder at its native width, without normalizing.
fn embed(
  encoder: &SentenceEncoder,
  tokenizer: &TextTokenizer,
  device: &Device,
  texts: &[&str],
  batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
  let mut vectors = Vec::with_capacity(texts.len());
  for chunk in texts.chunks(batch_size.max(1)) {
    let rows = chunk
      .iter()
      .map(|text| tokenizer.encode(text))
      .collect::<similarity_core::Result<Vec<_>>>()?;
    let rows: Vec<&[u32]> = rows.iter().map(Vec::as_slice).collect();
    let (ids, padding) = pad_batch(&rows, device)
      .ok_or_else(|| TrainError::Backend("an evaluation batch has no tokens to encode".to_owned()))?;
    let flat = encoder
      .forward(ids, padding)
      .into_data()
      .convert::<f32>()
      .try_into_vec::<f32>()
      .map_err(|source| TrainError::Backend(format!("unreadable embedding data: {source:?}")))?;
    vectors.extend(flat.chunks(encoder.dimension().max(1)).map(<[f32]>::to_vec));
  }
  Ok(vectors)
}

/// The cosine of the first `width` components of two vectors, which is what the server computes after truncating.
fn truncated_cosine(left: &[f32], right: &[f32], width: usize) -> Result<f32> {
  let prefix = |vector: &[f32]| -> Vec<f32> { vector.iter().take(width).copied().collect() };
  Ok(similarity_core::vector::cosine(&prefix(left), &prefix(right))?)
}

/// Scores every example's two texts against each other at each of `widths`, embedding each text once.
///
/// `encoder` must not be in training mode (use it as loaded, or after [`crate::Session::finish`]), or every forward
/// pass here would needlessly record a gradient graph.
///
/// # Errors
///
/// Returns [`TrainError::Model`] if a text cannot be tokenized or an embedding is degenerate, and
/// [`TrainError::Backend`] if the backend returns unreadable data.
pub fn separation(
  encoder: &SentenceEncoder,
  tokenizer: &TextTokenizer,
  device: &Device,
  examples: &[PairExample],
  widths: &[usize],
  batch_size: usize,
) -> Result<Vec<Separation>> {
  let before: Vec<&str> = examples.iter().map(|example| example.before.as_str()).collect();
  let after: Vec<&str> = examples.iter().map(|example| example.after.as_str()).collect();
  let before = embed(encoder, tokenizer, device, &before, batch_size)?;
  let after = embed(encoder, tokenizer, device, &after, batch_size)?;

  widths
    .iter()
    .map(|width| {
      let (mut edits, mut recycles) = (Vec::new(), Vec::new());
      for ((example, left), right) in examples.iter().zip(&before).zip(&after) {
        let cosine = truncated_cosine(left, right, *width)?;
        if example.same {
          edits.push(cosine);
        } else {
          recycles.push(cosine);
        }
      }
      Ok(Separation {
        width: *width,
        edits: Spread::of(&edits),
        recycles: Spread::of(&recycles),
        auc: similarity_core::separation::area_under_curve(&edits, &recycles),
      })
    })
    .collect()
}
