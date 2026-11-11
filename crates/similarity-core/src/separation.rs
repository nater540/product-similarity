//! How well two groups of scores are separated, as one number that needs no threshold.
//!
//! Choosing between two models, or two checkpoints of one, by "errors at the configured threshold" is a mistake:
//! a threshold belongs to the model it was measured on (D3), so it would judge every other model on a scale that
//! is not its own. The area under the ROC curve asks only about order: how often an edit scores above a recycle.

/// A count as a number to divide by. Pair counts are far below the 2^53 a `f64` holds exactly.
#[expect(clippy::cast_precision_loss, reason = "counts of scores are far below 2^53")]
fn number(count: usize) -> f64 {
  count as f64
}

/// The probability that a randomly chosen edit scores above a randomly chosen recycle, a tie counting as half.
///
/// 1.0 when every edit is above every recycle, 0.5 when the scores say nothing, 0.0 when the order is exactly
/// backwards. It depends only on the order of the scores, so it is the same at any threshold and under any
/// rescaling that keeps the order. `None` unless both groups have a score.
pub fn area_under_curve(edits: &[f32], recycles: &[f32]) -> Option<f64> {
  if edits.is_empty() || recycles.is_empty() {
    return None;
  }
  // Adding zero makes a negative zero a zero, so that the two compare as the tie they are.
  let mut sorted: Vec<f32> = recycles.iter().map(|score| score + 0.0).collect();
  sorted.sort_by(f32::total_cmp);

  let mut wins = 0.0_f64;
  for edit in edits {
    let edit = edit + 0.0;
    let below = sorted.partition_point(|recycle| recycle.total_cmp(&edit).is_lt());
    let not_above = sorted.partition_point(|recycle| recycle.total_cmp(&edit).is_le());
    wins += number(below) + 0.5 * number(not_above.saturating_sub(below));
  }
  Some(wins / (number(edits.len()) * number(recycles.len())))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn the_area_is_how_often_an_edit_outscores_a_recycle() {
    assert_eq!(
      area_under_curve(&[0.9, 0.95], &[0.2, 0.5]),
      Some(1.0),
      "every edit above every recycle"
    );
    assert_eq!(
      area_under_curve(&[0.2, 0.5], &[0.9, 0.95]),
      Some(0.0),
      "exactly backwards"
    );
    // Edits 0.9 and 0.4 against recycles 0.5 and 0.3: three of the four pairings have the edit on top.
    assert_eq!(area_under_curve(&[0.9, 0.4], &[0.5, 0.3]), Some(0.75));
  }

  #[test]
  fn a_tie_counts_as_half_and_scores_that_say_nothing_give_one_half() {
    assert_eq!(area_under_curve(&[0.5], &[0.5]), Some(0.5));
    assert_eq!(area_under_curve(&[0.7, 0.7, 0.7], &[0.7, 0.7]), Some(0.5));
    // One edit above, one tied: (1 + 0.5) of 2.
    assert_eq!(area_under_curve(&[0.9, 0.5], &[0.5]), Some(0.75));
    assert_eq!(
      area_under_curve(&[0.0], &[-0.0]),
      Some(0.5),
      "a zero is a zero, whatever its sign bit"
    );
  }

  #[test]
  fn there_is_no_area_without_both_groups() {
    assert_eq!(area_under_curve(&[], &[0.5]), None);
    assert_eq!(area_under_curve(&[0.5], &[]), None);
    assert_eq!(area_under_curve(&[], &[]), None);
  }
}
