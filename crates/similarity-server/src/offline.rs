//! Scoring records from the command line, with no server and no store.
//!
//! Thresholds are the whole product here, and they cannot be chosen by intuition: cosine similarities from a
//! sentence encoder are not percentages, and "0.8" means something different for every model, every field, and
//! every weighting. This module is how a threshold is chosen with evidence instead:
//!
//! - [`Offline::compare`] scores one before/after pair field by field, to see why a verdict came out as it did.
//! - [`calibrate`] scores a file of labeled pairs (ordinary edits and recycled records) and reports where the two
//!   groups separate, which is the threshold to configure.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use anyhow::Context;
use serde::Deserialize;
use similarity_core::score::{self, Evaluation};
use similarity_core::{Embedder, FieldChange, Policy, PreparedRecord, RecordVectors, Schema};

/// A record's fields as they appear in a JSON file: name to text.
pub type FieldMap = BTreeMap<String, String>;

/// An embedder plus the schema and policy to score with.
#[derive(Debug)]
pub struct Offline {
  embedder: Embedder,
  schema: Schema,
  policy: Policy,
}

impl Offline {
  /// Bundles a loaded model with the configured schema and policy.
  pub fn new(embedder: Embedder, schema: Schema, policy: Policy) -> Self {
    Self {
      embedder,
      schema,
      policy,
    }
  }

  /// One line naming the model and the width it ran at, to head every report.
  ///
  /// Scores are only comparable between runs at the same width, and a configuration that truncates a model which
  /// was not trained for it (see `output_dim`) produces a normal-looking report. The report has to say what it
  /// measured; a log line can be filtered out, and `just calibrate` does filter it.
  pub fn model_line(&self) -> String {
    format!(
      "model     {} ({} dimensions per field, up to {} tokens)\n",
      self.embedder.fingerprint(),
      self.embedder.dimension(),
      self.embedder.max_tokens()
    )
  }

  fn vectors(&self, fields: &FieldMap, baseline: Option<&RecordVectors>) -> anyhow::Result<RecordVectors> {
    let prepared = PreparedRecord::new(&self.schema, fields)?;
    let dimension = self.embedder.dimension();
    let plan = prepared.plan(&self.schema, dimension, baseline)?;
    let embedded = self.embedder.embed(&plan.texts())?;
    Ok(plan.finish(&self.schema, dimension, &embedded)?)
  }

  /// Judges `after` as a revision of `before`, exactly as the server would for a stored baseline.
  ///
  /// # Errors
  ///
  /// Returns an error if either record does not fit the schema or the model fails.
  pub fn compare(&self, before: &FieldMap, after: &FieldMap) -> anyhow::Result<Evaluation> {
    let baseline = self.vectors(before, None)?;
    let candidate = self.vectors(after, Some(&baseline))?;
    Ok(score::evaluate(
      &self.schema,
      &self.policy,
      self.embedder.dimension(),
      &baseline,
      None,
      &candidate,
    )?)
  }
}

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
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabeledPair {
  /// What this pair is an example of.
  pub label: Label,
  /// A short description, shown in the report.
  #[serde(default)]
  pub note: String,
  /// The record before the change.
  pub before: FieldMap,
  /// The record after the change.
  pub after: FieldMap,
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

fn summarize(pairs: Vec<ScoredPair>) -> Calibration {
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

fn change_mark(change: FieldChange) -> &'static str {
  match change {
    FieldChange::Unchanged => "unchanged",
    FieldChange::Modified => "modified",
    FieldChange::Added => "added",
    FieldChange::Removed => "removed",
    FieldChange::Restored => "restored",
  }
}

/// Renders one comparison as a small table, for a terminal.
pub fn render_comparison(evaluation: &Evaluation, policy: &Policy) -> String {
  let mut out = String::new();
  // Writing to a `String` cannot fail, here and below.
  let _ = writeln!(out, "{:<16} {:>6} {:>10}  change", "field", "weight", "similarity");
  for field in &evaluation.comparison.fields {
    let similarity = field
      .similarity
      .map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"));
    let flag = if field.below_floor { "  << below floor" } else { "" };
    let _ = writeln!(
      out,
      "{:<16} {:>6.2} {:>10}  {}{flag}",
      field.field,
      field.weight,
      similarity,
      change_mark(field.change)
    );
  }
  let _ = writeln!(
    out,
    "\noverall similarity {:.3} (threshold {:.3})",
    evaluation.comparison.similarity, policy.threshold
  );
  if evaluation.accepted() {
    let _ = writeln!(out, "verdict: ACCEPTED");
  } else {
    let _ = writeln!(out, "verdict: REJECTED");
    for reason in &evaluation.reasons {
      let _ = writeln!(out, "  - {reason}");
    }
  }
  out
}

/// Appends, per field, the range of its similarity over the pairs where that field's text actually changed.
///
/// This is what a field's `reject_below` floor is chosen from: it belongs in the gap between the two ranges, if
/// there is one.
fn render_field_ranges(calibration: &Calibration, out: &mut String) {
  let field_range = |field: &str, label: Label| {
    let mut scores = calibration
      .pairs
      .iter()
      .filter(|scored| scored.pair.label == label)
      .flat_map(|scored| &scored.evaluation.comparison.fields)
      .filter(|score| score.field == field && score.change == FieldChange::Modified)
      .filter_map(|score| score.similarity);
    let first = scores.next()?;
    Some(scores.fold((1_usize, first, first), |(count, low, high), score| {
      (count.saturating_add(1), low.min(score), high.max(score))
    }))
  };
  let field_names: std::collections::BTreeSet<&str> = calibration
    .pairs
    .iter()
    .flat_map(|scored| &scored.evaluation.comparison.fields)
    .map(|score| score.field.as_str())
    .collect();
  let describe = |range: Option<(usize, f32, f32)>| match range {
    Some((count, low, high)) => format!("{low:.3} to {high:.3} ({count:>2})"),
    None => "-".to_owned(),
  };
  let _ = writeln!(
    out,
    "\n{:<20} {:<22} {:<22}",
    "field (if modified)", "in edits", "in recycles"
  );
  for field in field_names {
    let _ = writeln!(
      out,
      "{field:<20} {:<22} {:<22}",
      describe(field_range(field, Label::Edit)),
      describe(field_range(field, Label::Recycle))
    );
  }
}

/// Renders a calibration as a report, for a terminal.
pub fn render_calibration(calibration: &Calibration, policy: &Policy) -> String {
  let mut out = String::new();
  let _ = writeln!(out, "{:<8} {:>7}  {:<9} note", "label", "overall", "verdict");
  let mut ordered: Vec<&ScoredPair> = calibration.pairs.iter().collect();
  ordered.sort_by(|a, b| {
    b.evaluation
      .comparison
      .similarity
      .total_cmp(&a.evaluation.comparison.similarity)
  });
  for scored in ordered {
    let label = match scored.pair.label {
      Label::Edit => "edit",
      Label::Recycle => "recycle",
    };
    let accepted = scored.evaluation.accepted();
    let verdict = if accepted { "accepted" } else { "rejected" };
    let wrong = if accepted == (scored.pair.label == Label::Edit) {
      ""
    } else {
      "  << WRONG"
    };
    let _ = writeln!(
      out,
      "{label:<8} {:>7.3}  {verdict:<9} {}{wrong}",
      scored.evaluation.comparison.similarity, scored.pair.note
    );
  }

  let range = |label: Label| {
    let mut scores = calibration
      .pairs
      .iter()
      .filter(|scored| scored.pair.label == label)
      .map(|scored| scored.evaluation.comparison.similarity);
    let first = scores.next()?;
    Some(scores.fold((first, first), |(low, high), score| (low.min(score), high.max(score))))
  };
  let _ = writeln!(out);
  for (name, label) in [("edits", Label::Edit), ("recycles", Label::Recycle)] {
    if let Some((low, high)) = range(label) {
      let _ = writeln!(out, "{name:<9} overall similarity from {low:.3} to {high:.3}");
    }
  }
  render_field_ranges(calibration, &mut out);

  let _ = writeln!(
    out,
    "\nconfigured policy (threshold {:.3} plus field floors): {} edit(s) wrongly rejected, {} recycle(s) wrongly \
     accepted, of {} pairs",
    policy.threshold,
    calibration.false_rejects,
    calibration.false_accepts,
    calibration.pairs.len()
  );
  match calibration.suggestion {
    Some(suggestion) => {
      let _ = writeln!(
        out,
        "best overall threshold for this data: {:.3} (margin {:.3}; {} edit(s) rejected, {} recycle(s) accepted)",
        suggestion.threshold, suggestion.margin, suggestion.false_rejects, suggestion.false_accepts
      );
    }
    None => {
      let _ = writeln!(
        out,
        "no threshold suggested: the file needs at least one `edit` and one `recycle` pair"
      );
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use similarity_core::score::{Comparison, Reason};

  use super::*;

  fn scored(label: Label, similarity: f32, accepted: bool) -> ScoredPair {
    let reasons = if accepted {
      Vec::new()
    } else {
      vec![Reason::BelowThreshold {
        similarity,
        threshold: 0.8,
      }]
    };
    ScoredPair {
      pair: LabeledPair {
        label,
        note: format!("{label:?} at {similarity}"),
        before: FieldMap::new(),
        after: FieldMap::new(),
      },
      evaluation: Evaluation {
        comparison: Comparison {
          similarity,
          fields: Vec::new(),
        },
        anchor_similarity: None,
        reasons,
      },
    }
  }

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
  fn the_summary_counts_what_the_configured_policy_gets_wrong() {
    let calibration = summarize(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Edit, 0.70, false),
      scored(Label::Recycle, 0.85, true),
      scored(Label::Recycle, 0.20, false),
    ]);
    assert_eq!((calibration.false_rejects, calibration.false_accepts), (1, 1));

    let report = render_calibration(&calibration, &Policy::new(0.8, None).unwrap());
    assert_eq!(report.matches("<< WRONG").count(), 2, "{report}");
    assert!(
      report.contains("1 edit(s) wrongly rejected, 1 recycle(s) wrongly accepted, of 4 pairs"),
      "{report}"
    );
    assert!(report.contains("best overall threshold"), "{report}");
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
    assert_eq!(pairs[1].after["name"], "Vape");
  }

  #[test]
  fn a_bad_line_is_reported_by_number() {
    let text = "{\"label\": \"edit\", \"before\": {}, \"after\": {}}\n{\"label\": \"sideways\"}";
    let error = parse_pairs(text).unwrap_err();
    assert!(format!("{error:#}").contains("line 2"), "{error:#}");
    assert!(parse_pairs(r#"{"label": "edit", "before": {}, "after": {}, "extra": 1}"#).is_err());
  }

  #[test]
  fn a_comparison_renders_its_reasons() {
    let evaluation = scored(Label::Recycle, 0.2, false).evaluation;
    let report = render_comparison(&evaluation, &Policy::new(0.8, None).unwrap());
    assert!(
      report.contains("verdict: REJECTED") && report.contains("below the threshold"),
      "{report}"
    );
    let accepted = scored(Label::Edit, 0.9, true).evaluation;
    assert!(render_comparison(&accepted, &Policy::new(0.8, None).unwrap()).contains("verdict: ACCEPTED"));
  }
}
