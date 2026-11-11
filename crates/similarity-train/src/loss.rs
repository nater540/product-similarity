//! The training objective.
//!
//! Two losses, one per kind of example (see [`crate::data`]), both computed on unit vectors:
//!
//! - [`contrastive`], for labeled pairs: pull the two versions of an ordinary edit together, push the two sides of
//!   a recycled listing at least a margin apart.
//! - [`ranking`], for catalog groups: each catalog entry must pick out its own listing from every listing in the
//!   batch.
//!
//! Either is applied at several truncated widths and averaged ([`truncated`]). The server keeps only the first
//! `output_dim` components of each embedding, and training on the full width alone would let those leading
//! components stop being a good embedding on their own, so every width that must stay usable takes part in the
//! loss (the Matryoshka idea).

use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::activation::relu;
use burn::tensor::{Int, Tensor};

/// How far apart, in cosine distance, the two sides of a recycled listing are pushed before the loss stops caring.
pub const DEFAULT_MARGIN: f64 = 0.5;

/// The factor cosine similarities are multiplied by before the ranking loss's softmax (an inverse temperature).
/// Cosines span only `[-1, 1]`, which on its own is too flat a distribution to learn from.
pub const DEFAULT_SCALE: f64 = 20.0;

/// Which loss a batch is trained with.
#[derive(Debug, Clone)]
#[expect(
  clippy::large_enum_variant,
  reason = "one short-lived value per training step; boxing the tensor handle would buy nothing"
)]
pub enum Objective {
  /// [`contrastive`]; `same` holds one value per row, 1 for an edit and 0 for a recycle.
  Contrastive {
    /// The labels, `[rows]`.
    same: Tensor<1>,
    /// See [`DEFAULT_MARGIN`].
    margin: f64,
  },
  /// [`ranking`].
  Ranking {
    /// See [`DEFAULT_SCALE`].
    scale: f64,
  },
}

/// The first `width` components of each row, scaled to unit length.
///
/// This is exactly what the server does to an embedding before storing or comparing it.
pub fn unit_prefix(embeddings: Tensor<2>, width: usize) -> Tensor<2> {
  let prefix = embeddings.narrow(1, 0, width);
  // The floor keeps an all-zero row from dividing by zero and poisoning the batch with NaN.
  let norm = prefix.clone().square().sum_dim(1).sqrt().clamp_min(1e-12);
  prefix.div(norm)
}

/// The contrastive loss over rows of unit vectors.
///
/// With `d` the cosine distance (one minus the cosine) of a row: `d²` for an edit, `max(0, margin - d)²` for a
/// recycle, halved and averaged over the batch. An edit is pulled all the way to distance zero; a recycle only has
/// to clear the margin, since "how different" two different products are carries no information.
pub fn contrastive(before: Tensor<2>, after: Tensor<2>, same: Tensor<1>, margin: f64) -> Tensor<1> {
  let [rows, _] = before.dims();
  let distance = before.mul(after).sum_dim(1).reshape([rows]).neg().add_scalar(1.0);
  let pull = same.clone().mul(distance.clone().square());
  let short_of_margin = relu(distance.neg().add_scalar(margin));
  let push = same.neg().add_scalar(1.0).mul(short_of_margin.square());
  pull.add(push).mean().mul_scalar(0.5)
}

/// The in-batch ranking loss over rows of unit vectors (multiple negatives ranking, or `InfoNCE`).
///
/// Row `i` of `anchors` is scored against every row of `positives`; the loss is the cross-entropy of picking row
/// `i`. Every other row of the batch therefore acts as a negative, which is why a batch must never hold the same
/// text twice.
pub fn ranking(anchors: Tensor<2>, positives: Tensor<2>, scale: f64) -> Tensor<1> {
  let [rows, _] = anchors.dims();
  let device = anchors.device();
  let scores = anchors.matmul(positives.transpose()).mul_scalar(scale);
  let own = Tensor::<1, Int>::arange(0..i64::try_from(rows).unwrap_or(i64::MAX), &device);
  CrossEntropyLossConfig::new().init(&device).forward(scores, own)
}

/// `objective` applied to the two sides at each of `widths`, averaged.
///
/// `left` and `right` are raw encoder outputs, `[rows, native width]`; each width must not exceed the native one.
/// Returns a one-element tensor.
pub fn truncated(left: &Tensor<2>, right: &Tensor<2>, widths: &[usize], objective: &Objective) -> Tensor<1> {
  let mut total: Option<Tensor<1>> = None;
  for width in widths {
    let (a, b) = (unit_prefix(left.clone(), *width), unit_prefix(right.clone(), *width));
    let loss = match objective {
      Objective::Contrastive { same, margin } => contrastive(a, b, same.clone(), *margin),
      Objective::Ranking { scale } => ranking(a, b, *scale),
    };
    total = Some(match total {
      Some(sum) => sum.add(loss),
      None => loss,
    });
  }
  #[expect(clippy::cast_precision_loss, reason = "a handful of widths")]
  let count = widths.len().max(1) as f64;
  match total {
    Some(sum) => sum.div_scalar(count),
    None => Tensor::zeros([1], &left.device()),
  }
}

#[cfg(test)]
mod tests {
  use burn::tensor::{Device, TensorData};

  use super::*;

  fn matrix<const C: usize>(rows: &[[f32; C]], device: &Device) -> Tensor<2> {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Tensor::from_data(TensorData::new(flat, [rows.len(), C]), device)
  }

  fn labels(values: &[f32], device: &Device) -> Tensor<1> {
    Tensor::from_data(TensorData::new(values.to_vec(), [values.len()]), device)
  }

  fn scalar(tensor: Tensor<1>) -> f32 {
    tensor.into_scalar::<f32>()
  }

  fn close(found: f32, expected: f32) {
    assert!((found - expected).abs() < 1e-5, "{found} vs {expected}");
  }

  #[test]
  fn a_unit_prefix_is_the_leading_components_at_unit_length() {
    let device = Device::flex();
    let rows = unit_prefix(matrix(&[[3.0, 4.0, 12.0], [0.0, -2.0, 9.0]], &device), 2);
    let values = rows.into_data().try_into_vec::<f32>().unwrap();
    for (found, expected) in values.iter().zip([0.6, 0.8, 0.0, -1.0]) {
      close(*found, expected);
    }
  }

  #[test]
  fn an_all_zero_row_stays_finite() {
    let device = Device::flex();
    let rows = unit_prefix(matrix(&[[0.0, 0.0, 5.0]], &device), 2);
    assert!(
      rows
        .into_data()
        .try_into_vec::<f32>()
        .unwrap()
        .iter()
        .all(|value| value.is_finite())
    );
  }

  #[test]
  fn the_contrastive_loss_matches_values_worked_out_by_hand() {
    let device = Device::flex();
    let x = [1.0, 0.0];
    let y = [0.0, 1.0];
    let loss = |before: [f32; 2], after: [f32; 2], same: f32| {
      scalar(contrastive(
        matrix(&[before], &device),
        matrix(&[after], &device),
        labels(&[same], &device),
        0.5,
      ))
    };
    // An edit that did not move: nothing to pull.
    close(loss(x, x, 1.0), 0.0);
    // An edit at distance 1: 0.5 * 1^2.
    close(loss(x, y, 1.0), 0.5);
    // A recycle already past the margin: nothing to push.
    close(loss(x, y, 0.0), 0.0);
    // A recycle at distance 0: 0.5 * (0.5 - 0)^2.
    close(loss(x, x, 0.0), 0.125);
    // A recycle at distance 0.2 (cosine 0.8): 0.5 * (0.5 - 0.2)^2.
    close(loss(x, [0.8, 0.6], 0.0), 0.045);
  }

  #[test]
  fn the_contrastive_loss_is_averaged_over_the_batch() {
    let device = Device::flex();
    let before = matrix(&[[1.0, 0.0], [1.0, 0.0]], &device);
    let after = matrix(&[[0.0, 1.0], [1.0, 0.0]], &device);
    // Row 0 is an edit at distance 1 (0.5), row 1 a recycle at distance 0 (0.125).
    close(
      scalar(contrastive(before, after, labels(&[1.0, 0.0], &device), 0.5)),
      0.3125,
    );
  }

  #[test]
  fn the_ranking_loss_matches_values_worked_out_by_hand() {
    let device = Device::flex();
    let identity = || matrix(&[[1.0, 0.0], [0.0, 1.0]], &device);
    // Each anchor scores 2 against its own positive and 0 against the other: -ln(e^2 / (e^2 + 1)).
    close(
      scalar(ranking(identity(), identity(), 2.0)),
      (1.0 + (-2.0_f32).exp()).ln(),
    );
    // Every positive is the same vector, so nothing can be told apart: ln(2).
    let same = matrix(&[[1.0, 0.0], [1.0, 0.0]], &device);
    close(scalar(ranking(identity(), same, 20.0)), 2.0_f32.ln());
    // Each anchor matches the other row's positive: -ln(1 / (1 + e^2)).
    let swapped = matrix(&[[0.0, 1.0], [1.0, 0.0]], &device);
    close(scalar(ranking(identity(), swapped, 2.0)), (1.0 + 2.0_f32.exp()).ln());
  }

  #[test]
  fn a_truncated_loss_is_the_mean_over_the_widths_each_renormalized() {
    let device = Device::flex();
    // At width 1 both rows are [1], distance 0. At width 2 they are orthogonal after normalization, distance 1.
    let left = matrix(&[[2.0, 0.0]], &device);
    let right = matrix(&[[0.000_001, 5.0]], &device);
    let objective = Objective::Contrastive {
      same: labels(&[1.0], &device),
      margin: 0.5,
    };
    close(scalar(truncated(&left, &right, &[1], &objective)), 0.0);
    close(scalar(truncated(&left, &right, &[2], &objective)), 0.5);
    close(scalar(truncated(&left, &right, &[2, 1], &objective)), 0.25);
  }
}
