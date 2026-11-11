//! Labeled pairs, and what scoring a file of them shows.
//!
//! A calibration file holds before/after pairs that someone has labeled as an ordinary edit or as a recycled
//! record. Scoring them all with the configured schema and policy shows where the two groups sit, which is where
//! a threshold belongs, and how many pairs the configuration as it stands gets wrong.

use anyhow::Context;
use serde::Deserialize;
use similarity_core::score::Evaluation;

use super::{Offline, RecordFields};

/// What a labeled pair is an example of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Label {
  /// The same record, legitimately edited. Should be accepted.
  Edit,
  /// A different record under the old id. Should be rejected.
  Recycle,
}

/// One line of a calibration file.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabeledPair {
  /// What this pair is an example of.
  pub label: Label,
  /// A short description, shown in the report.
  #[serde(default)]
  pub note: String,
  /// The record before the change.
  pub before: RecordFields,
  /// The record after the change.
  pub after: RecordFields,
}

/// Parses a calibration file: one JSON [`LabeledPair`] per line, blank lines and `#` comments ignored.
///
/// # Errors
///
/// Returns an error naming the line if a line is not a valid pair.
pub fn parse_pairs(jsonl: &str) -> anyhow::Result<Vec<LabeledPair>> {
  jsonl
    .lines()
    .enumerate()
    .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
    .map(|(index, line)| {
      serde_json::from_str(line).with_context(|| format!("line {} is not a valid pair", index.saturating_add(1)))
    })
    .collect()
}

/// A labeled pair with its score.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredPair {
  /// The pair.
  pub pair: LabeledPair,
  /// How the configured schema and policy judge it.
  pub evaluation: Evaluation,
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
}

/// Finds the overall threshold that misclassifies the fewest pairs, preferring the widest margin among ties.
///
/// A record is accepted when its score is at or above the threshold, so candidate thresholds are the midpoints
/// between neighboring scores. Returns `None` unless both labels are present.
pub fn suggest_threshold(scores: &[(Label, f32)]) -> Option<Suggestion> {
  let count = |label: Label| scores.iter().filter(|(found, _)| *found == label).count();
  if count(Label::Edit) == 0 || count(Label::Recycle) == 0 {
    return None;
  }
  let mut sorted: Vec<f32> = scores.iter().map(|(_, score)| *score).collect();
  sorted.sort_by(f32::total_cmp);
  sorted.dedup();

  sorted
    .windows(2)
    .filter_map(|pair| match pair {
      [below, above] => Some((f32::midpoint(*below, *above), (above - below) / 2.0)),
      _ => None,
    })
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
pub fn calibrate(offline: &Offline, pairs: Vec<LabeledPair>) -> anyhow::Result<Calibration> {
  let mut scored = Vec::with_capacity(pairs.len());
  for (index, pair) in pairs.into_iter().enumerate() {
    let evaluation = offline
      .compare(&pair.before, &pair.after)
      .with_context(|| format!("pair {} ({})", index.saturating_add(1), pair.note))?;
    scored.push(ScoredPair { pair, evaluation });
  }
  Ok(summarize(scored))
}

pub(super) fn summarize(pairs: Vec<ScoredPair>) -> Calibration {
  let scores: Vec<(Label, f32)> = pairs
    .iter()
    .map(|scored| (scored.pair.label, scored.evaluation.comparison.similarity))
    .collect();
  let wrong = |label: Label, accepted: bool| {
    pairs
      .iter()
      .filter(|scored| scored.pair.label == label && scored.evaluation.accepted() == accepted)
      .count()
  };
  Calibration {
    suggestion: suggest_threshold(&scores),
    false_rejects: wrong(Label::Edit, false),
    false_accepts: wrong(Label::Recycle, true),
    pairs,
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

  #[test]
  fn pairs_parse_from_jsonl_with_comments_and_blank_lines() {
    let text = r#"
      # ordinary edits
      {"label": "edit", "note": "potency added", "before": {"name": "Gummies"}, "after": {"name": "Gummies 100mg"}}

      {"label": "recycle", "before": {"name": "Gummies"}, "after": {"name": "Vape"}}
    "#;
    let pairs = parse_pairs(text).unwrap();
    assert_eq!(pairs.len(), 2);
    assert_eq!((pairs[0].label, pairs[0].note.as_str()), (Label::Edit, "potency added"));
    assert_eq!(pairs[1].after.texts["name"], "Vape");
    assert!(pairs[1].after.quantities.is_empty());
  }

  #[test]
  fn a_bad_line_is_reported_by_number() {
    let text = "{\"label\": \"edit\", \"before\": {}, \"after\": {}}\n{\"label\": \"sideways\"}";
    let error = parse_pairs(text).unwrap_err();
    assert!(format!("{error:#}").contains("line 2"), "{error:#}");
    assert!(parse_pairs(r#"{"label": "edit", "before": {}, "after": {}, "extra": 1}"#).is_err());
  }
}
