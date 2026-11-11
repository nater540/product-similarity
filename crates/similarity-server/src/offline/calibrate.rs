//! Scoring a file of labeled pairs.
//!
//! A calibration file holds before/after pairs that someone has labeled as an ordinary edit or as a recycled
//! record (see [`super::pairs`]). Scoring them all with the configured schema and policy shows where the two
//! groups sit, which is where a threshold belongs, and how many pairs the configuration as it stands gets wrong.

use anyhow::Context;
use similarity_core::score::Evaluation;

use super::Offline;
use super::pairs::{Input, Label, Pair, Pairs};

/// A labeled pair with its score.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredPair {
  /// The pair.
  pub pair: Pair,
  /// How the configured schema and policy judge it.
  pub evaluation: Evaluation,
}

impl ScoredPair {
  /// The pair's overall similarity.
  pub fn similarity(&self) -> f32 {
    self.evaluation.comparison.similarity
  }

  /// Whether the configured policy gets this pair wrong: rejects an edit, or accepts a recycle. A pair with any
  /// other label cannot be wrong.
  pub fn wrong(&self) -> bool {
    match self.pair.label {
      Some(Label::Edit) => !self.evaluation.accepted(),
      Some(Label::Recycle) => self.evaluation.accepted(),
      None => false,
    }
  }
}

/// The overall threshold that best separates edits from recycles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Suggestion {
  /// The threshold.
  pub threshold: f32,
  /// Edits that would be wrongly rejected at this threshold.
  pub false_rejects: usize,
  /// Recycles that would be wrongly accepted at this threshold.
  pub false_accepts: usize,
  /// The distance from the threshold to the nearest score on either side. Small means fragile.
  pub margin: f32,
}

/// The result of scoring a calibration file.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
  /// Every pair, scored, in file order.
  pub pairs: Vec<ScoredPair>,
  /// The best overall threshold for this data, if there is at least one pair of each label.
  pub suggestion: Option<Suggestion>,
  /// Edits the configured policy (threshold and field floors together) rejects.
  pub false_rejects: usize,
  /// Recycles the configured policy accepts.
  pub false_accepts: usize,
  /// What the pairs were read from.
  pub input: Input,
}

impl Calibration {
  /// The scored pairs carrying `label`.
  pub fn labeled(&self, label: Label) -> impl Iterator<Item = &ScoredPair> + Clone {
    self.pairs.iter().filter(move |scored| scored.pair.label == Some(label))
  }

  /// The scored pairs with any label other than `edit` and `recycle`. They are shown and never counted.
  pub fn unlabeled(&self) -> impl Iterator<Item = &ScoredPair> + Clone {
    self.pairs.iter().filter(|scored| scored.pair.label.is_none())
  }
}

/// The thresholds worth trying between `sorted`, distinct scores in rising order: one for each pair of neighbors,
/// with half the gap it sits in.
///
/// A record is accepted when its score is at or above the threshold, so any value above the lower neighbor and at
/// or below the upper one puts the two on different sides. The midpoint is such a value, except between two
/// scores so close that it rounds to the lower one. There the upper score itself is the threshold; the midpoint
/// would leave both above it and report as inseparable two scores that a threshold does separate.
pub(crate) fn cuts(sorted: &[f32]) -> impl Iterator<Item = (f32, f32)> + '_ {
  sorted.windows(2).filter_map(|pair| match pair {
    [below, above] => {
      let middle = f32::midpoint(*below, *above);
      Some((if middle > *below { middle } else { *above }, (above - below) / 2.0))
    }
    _ => None,
  })
}

/// Finds the overall threshold that misclassifies the fewest pairs, preferring the widest margin among ties.
///
/// Candidate thresholds are the ones [`cuts`] gives: the midpoints between neighboring scores. Returns `None`
/// unless both labels are present.
pub fn suggest_threshold(scores: &[(Label, f32)]) -> Option<Suggestion> {
  let has = |wanted: fn(&Label) -> bool| scores.iter().any(|(label, _)| wanted(label));
  if !has(|label| matches!(label, Label::Edit)) || !has(|label| matches!(label, Label::Recycle)) {
    return None;
  }
  let mut sorted: Vec<f32> = scores.iter().map(|(_, score)| *score).collect();
  sorted.sort_by(f32::total_cmp);
  sorted.dedup();

  cuts(&sorted)
    .map(|(threshold, margin)| Suggestion {
      threshold,
      margin,
      false_rejects: scores
        .iter()
        .filter(|(label, score)| *label == Label::Edit && *score < threshold)
        .count(),
      false_accepts: scores
        .iter()
        .filter(|(label, score)| *label == Label::Recycle && *score >= threshold)
        .count(),
    })
    .min_by(|a, b| {
      let errors = |s: &Suggestion| s.false_rejects.saturating_add(s.false_accepts);
      errors(a).cmp(&errors(b)).then_with(|| b.margin.total_cmp(&a.margin))
    })
}

/// Scores every pair and summarizes how well edits and recycles separate.
///
/// # Errors
///
/// Returns an error naming the pair if a record does not fit the schema or the model fails.
pub fn calibrate(offline: &Offline, pairs: Pairs) -> anyhow::Result<Calibration> {
  let mut scored = Vec::with_capacity(pairs.pairs.len());
  for pair in pairs.pairs {
    let evaluation = offline
      .compare(&pair.before, &pair.after, pair.anchor.as_ref())
      .with_context(|| format!("pair `{}` ({})", pair.id, pair.note))?;
    scored.push(ScoredPair { pair, evaluation });
  }
  Ok(summarize(scored, pairs.input))
}

/// Summarizes pairs that have been scored: what the configured policy gets wrong, and the threshold that would
/// separate the two labels best. Pairs with any other label take no part in either.
pub fn summarize(pairs: Vec<ScoredPair>, input: Input) -> Calibration {
  let scores: Vec<(Label, f32)> = pairs
    .iter()
    .filter_map(|scored| Some((scored.pair.label?, scored.similarity())))
    .collect();
  let wrong = |label: Label| {
    pairs
      .iter()
      .filter(|scored| scored.pair.label == Some(label) && scored.wrong())
      .count()
  };
  Calibration {
    suggestion: suggest_threshold(&scores),
    false_rejects: wrong(Label::Edit),
    false_accepts: wrong(Label::Recycle),
    pairs,
    input,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cleanly_separated_groups_get_the_midpoint_of_the_gap() {
    let scores = [
      (Label::Edit, 0.95),
      (Label::Edit, 0.90),
      (Label::Recycle, 0.50),
      (Label::Recycle, 0.30),
      (Label::Edit, 0.99),
    ];
    let suggestion = suggest_threshold(&scores).unwrap();
    assert!((suggestion.threshold - 0.70).abs() < 1e-6);
    assert!((suggestion.margin - 0.20).abs() < 1e-6);
    assert_eq!((suggestion.false_rejects, suggestion.false_accepts), (0, 0));
  }

  #[test]
  fn overlapping_groups_get_the_threshold_with_the_fewest_errors() {
    // One recycle scores above two edits; the best cut sacrifices that one recycle rather than both edits.
    let scores = [
      (Label::Edit, 0.95),
      (Label::Edit, 0.80),
      (Label::Edit, 0.78),
      (Label::Recycle, 0.85),
      (Label::Recycle, 0.40),
      (Label::Recycle, 0.35),
    ];
    let suggestion = suggest_threshold(&scores).unwrap();
    assert!((0.40..0.78).contains(&suggestion.threshold), "{suggestion:?}");
    assert_eq!((suggestion.false_rejects, suggestion.false_accepts), (0, 1));
  }

  #[test]
  fn one_label_alone_suggests_nothing() {
    assert_eq!(suggest_threshold(&[(Label::Edit, 0.9), (Label::Edit, 0.8)]), None);
    assert_eq!(suggest_threshold(&[]), None);
  }
}
