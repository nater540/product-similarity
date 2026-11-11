//! Plain-slice vector math.
//!
//! Everything downstream of the encoder works on `Vec<f32>` rather than tensors: the vectors are a few hundred
//! floats, they cross a gRPC boundary and a Qdrant round trip as `f32`, and a dot product that small is faster on
//! the stack than the cost of getting it onto a device. Sums accumulate in `f64` so that the result does not depend
//! on the order of a few hundred `f32` additions.

use crate::error::{Error, Result};

/// The dot product of two equal-length vectors.
///
/// # Errors
///
/// Returns [`Error::Incompatible`] if the lengths differ.
pub fn dot(left: &[f32], right: &[f32]) -> Result<f64> {
  if left.len() != right.len() {
    return Err(Error::Incompatible(format!(
      "vector lengths differ: {} vs {}",
      left.len(),
      right.len()
    )));
  }
  Ok(left.iter().zip(right).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum())
}

/// The Euclidean length of a vector.
pub fn norm(vector: &[f32]) -> f64 {
  vector.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>().sqrt()
}

/// Scales `vector` to unit length in place.
///
/// Returns `false`, leaving the vector untouched, if it has no direction (zero or non-finite length).
pub fn normalize(vector: &mut [f32]) -> bool {
  let length = norm(vector);
  if !length.is_finite() || length <= f64::from(f32::MIN_POSITIVE) {
    return false;
  }
  for value in vector.iter_mut() {
    #[expect(
      clippy::cast_possible_truncation,
      reason = "a unit-vector component always fits in f32"
    )]
    let scaled = (f64::from(*value) / length) as f32;
    *value = scaled;
  }
  true
}

/// The cosine similarity of two vectors, in `[-1, 1]`.
///
/// Does not assume unit inputs: stored vectors come back from Qdrant, and trusting them to still be exactly unit
/// length would turn any drift there into a silently biased score.
///
/// # Errors
///
/// Returns [`Error::Incompatible`] if the lengths differ or either vector has no direction.
pub fn cosine(left: &[f32], right: &[f32]) -> Result<f32> {
  let product = dot(left, right)?;
  let scale = norm(left) * norm(right);
  if !scale.is_finite() || scale <= f64::from(f32::MIN_POSITIVE) {
    return Err(Error::Incompatible("cannot compare a zero-length vector".to_owned()));
  }
  #[expect(clippy::cast_possible_truncation, reason = "clamped to [-1, 1] first")]
  let similarity = (product / scale).clamp(-1.0, 1.0) as f32;
  Ok(similarity)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cosine_of_a_vector_with_itself_is_one() {
    let vector = [0.3_f32, -1.2, 4.0];
    assert!((cosine(&vector, &vector).unwrap() - 1.0).abs() < 1e-6);
  }

  #[test]
  fn cosine_ignores_magnitude() {
    let left = [1.0_f32, 2.0, 3.0];
    let right = [10.0_f32, 20.0, 30.0];
    assert!((cosine(&left, &right).unwrap() - 1.0).abs() < 1e-6);
  }

  #[test]
  fn orthogonal_and_opposite_vectors_score_zero_and_minus_one() {
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-6);
    assert!((cosine(&[1.0, 0.0], &[-2.0, 0.0]).unwrap() + 1.0).abs() < 1e-6);
  }

  #[test]
  fn mismatched_lengths_are_an_error_not_a_truncation() {
    assert!(cosine(&[1.0, 0.0], &[1.0, 0.0, 0.0]).is_err());
    assert!(dot(&[1.0], &[]).is_err());
  }

  #[test]
  fn a_zero_vector_cannot_be_compared_or_normalized() {
    assert!(cosine(&[0.0, 0.0], &[1.0, 0.0]).is_err());
    let mut zero = [0.0_f32; 4];
    assert!(!normalize(&mut zero));
    assert_eq!(zero, [0.0; 4]);
  }

  #[test]
  fn normalize_produces_unit_length() {
    let mut vector = [3.0_f32, 4.0];
    assert!(normalize(&mut vector));
    assert!((norm(&vector) - 1.0).abs() < 1e-6);
    assert!((vector[0] - 0.6).abs() < 1e-6);
  }
}
