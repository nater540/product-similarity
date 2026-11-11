//! Scoring records from the command line, with no server and no store.
//!
//! Thresholds are the whole product here, and they cannot be chosen by intuition: cosine similarities from a
//! sentence encoder are not percentages, and "0.8" means something different for every model, every field, and
//! every weighting. This module is how a threshold is chosen with evidence instead:
//!
//! - [`Offline::compare`] scores one before/after pair field by field, to see why a verdict came out as it did.
//! - [`calibrate`] scores a file of labeled pairs (ordinary edits and recycled records) and reports where the two
//!   groups separate, which is the threshold to configure.
//!
//! The same goes for a quantity's limits. Both reports show how each quantity changed, and the calibration report
//! sets the factors seen in edits beside those seen in recycles, which is where `reject_above_factor` is read from.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use anyhow::Context;
use serde::{Deserialize, Deserializer};
use similarity_core::score::{self, Evaluation};
use similarity_core::{
  Embedder, FieldChange, Policy, PreparedRecord, Quantities, Quantity, QuantityChange, QuantityScore, RecordVectors,
  Schema,
};

/// A record as it appears in a JSON file: one object holding its text fields and its quantities together.
///
/// ```json
/// {"name": "Blue Dream 3.5g", "weight_per_item": {"value": 3.5, "unit": "g"},
///  "thc_per_item": {"min": 18, "max": 22, "unit": "%"}}
/// ```
///
/// The type of a value says which kind it is: a string is a text field, an object is a quantity. Whether the key
/// is configured as that kind is for the schema to say, when the record is scored.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecordFields {
  /// Text field name to text.
  pub texts: BTreeMap<String, String>,
  /// Quantity name to value.
  pub quantities: Quantities,
}

impl<'de> Deserialize<'de> for RecordFields {
  fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use serde::de::Error as _;
    use serde_json::Value;

    let mut fields = Self::default();
    for (key, value) in BTreeMap::<String, Value>::deserialize(deserializer)? {
      match value {
        Value::String(text) => {
          fields.texts.insert(key, text);
        }
        Value::Object(_) => {
          let quantity: Quantity =
            serde_json::from_value(value).map_err(|source| D::Error::custom(format!("`{key}`: {source}")))?;
          fields.quantities.insert(key, quantity);
        }
        // A bare number is the likeliest mistake, and the one a guess would get wrong: 3.5 of what?
        _ => {
          return Err(D::Error::custom(format!(
            "`{key}` must be text (a string) or a quantity (an object such as {{\"value\": 3.5, \"unit\": \"g\"}})"
          )));
        }
      }
    }
    Ok(fields)
  }
}

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

  /// Embeds a record's text and validates its quantities, as the server does for a request.
  fn prepared(
    &self,
    record: &RecordFields,
    baseline: Option<&RecordVectors>,
  ) -> anyhow::Result<(RecordVectors, Quantities)> {
    let quantities = record
      .quantities
      .iter()
      .map(|(name, quantity)| (name, quantity.clone()));
    let prepared = PreparedRecord::new(&self.schema, &record.texts)?.with_quantities(&self.schema, quantities)?;
    let dimension = self.embedder.dimension();
    let plan = prepared.plan(&self.schema, dimension, baseline)?;
    let embedded = self.embedder.embed(&plan.texts())?;
    let vectors = plan.finish(&self.schema, dimension, &embedded)?;
    Ok((vectors, prepared.quantities().clone()))
  }

  /// Judges `after` as a revision of `before`, exactly as the server would for a baseline it had just stored.
  ///
  /// # Errors
  ///
  /// Returns an error if either record does not fit the schema or the model fails.
  pub fn compare(&self, before: &RecordFields, after: &RecordFields) -> anyhow::Result<Evaluation> {
    let (baseline, old_quantities) = self.prepared(before, None)?;
    let (candidate, new_quantities) = self.prepared(after, Some(&baseline))?;
    let mut evaluation = score::evaluate(
      &self.schema,
      &self.policy,
      self.embedder.dimension(),
      &baseline,
      None,
      &candidate,
    )?;
    // A pair has no history, so there is no anchor and nothing can be "restored".
    evaluation.judge_quantities(
      &self.schema,
      &self.policy,
      &old_quantities,
      &Quantities::new(),
      &new_quantities,
    );
    Ok(evaluation)
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

fn quantity_mark(change: QuantityChange) -> &'static str {
  match change {
    QuantityChange::Unchanged => "unchanged",
    QuantityChange::Modified => "modified",
    QuantityChange::Incomparable => "incomparable",
    QuantityChange::Added => "added",
    QuantityChange::Removed => "removed",
    QuantityChange::Restored => "restored",
  }
}

/// Appends how each quantity changed, if the record has any. Says nothing at all for a record without quantities,
/// so a report on text-only records reads as it always did.
fn render_quantities(evaluation: &Evaluation, out: &mut String) {
  if evaluation.quantities.is_empty() {
    return;
  }
  let _ = writeln!(out, "\n{:<20} {:>8}  {:<14} units", "quantity", "factor", "change");
  for quantity in &evaluation.quantities {
    let factor = match quantity.factor {
      Some(factor) => format!("{factor:.2}"),
      None if quantity.zero_crossing => "no limit".to_owned(),
      None => "-".to_owned(),
    };
    let unit = |unit: &Option<String>| unit.clone().unwrap_or_else(|| "-".to_owned());
    let flag = if quantity.beyond_limit { "  << beyond limit" } else { "" };
    let _ = writeln!(
      out,
      "{:<20} {factor:>8}  {:<14} {} to {}{flag}",
      quantity.field,
      quantity_mark(quantity.change),
      unit(&quantity.previous_unit),
      unit(&quantity.unit)
    );
  }
  if evaluation.measure_changed {
    let _ = writeln!(out, "the measures the record is described by changed");
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
  render_quantities(evaluation, &mut out);
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

/// What one quantity did across the pairs of one label.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct QuantityRange {
  /// How many pairs changed the amount by a finite factor, and the smallest and largest of those factors.
  factors: Option<(usize, f64, f64)>,
  /// How many pairs put it in a unit of another dimension.
  incomparable: usize,
  /// How many pairs moved it between zero and a value above zero.
  zero_crossings: usize,
}

impl QuantityRange {
  fn of<'a>(scores: impl Iterator<Item = &'a QuantityScore>) -> Self {
    let mut range = Self::default();
    for score in scores {
      range.incomparable = range.incomparable.saturating_add(usize::from(score.incomparable));
      range.zero_crossings = range.zero_crossings.saturating_add(usize::from(score.zero_crossing));
      // A factor of exactly 1 is "no change" (the same amount, perhaps in another unit), which no limit can
      // reject and which would only drag every range down to 1.
      if let Some(factor) = score.factor.filter(|factor| *factor > 1.0) {
        range.factors = Some(match range.factors {
          Some((count, low, high)) => (count.saturating_add(1), low.min(factor), high.max(factor)),
          None => (1, factor, factor),
        });
      }
    }
    range
  }

  fn describe(&self) -> String {
    let mut parts = Vec::new();
    if let Some((count, low, high)) = self.factors {
      parts.push(format!("{low:.2} to {high:.2} ({count:>2})"));
    }
    if self.incomparable > 0 {
      parts.push(format!("{} incomparable", self.incomparable));
    }
    if self.zero_crossings > 0 {
      parts.push(format!("{} to or from zero", self.zero_crossings));
    }
    if parts.is_empty() {
      "-".to_owned()
    } else {
      parts.join(", ")
    }
  }
}

/// Appends, per quantity, the factors it changed by among edits and among recycles, and how often the measures
/// changed under each label. Says nothing if no pair has a quantity.
///
/// This is what a quantity's `reject_above_factor` is chosen from: it belongs above the largest factor seen in
/// edits and below the smallest seen in recycles, if the two leave a gap. `reject_incomparable` and
/// `reject_on_measure_change` are safe to switch on when the counts under `in edits` are zero.
fn render_quantity_ranges(calibration: &Calibration, out: &mut String) {
  let labeled = |label: Label| {
    calibration
      .pairs
      .iter()
      .filter(move |scored| scored.pair.label == label)
  };
  let names: std::collections::BTreeSet<&str> = calibration
    .pairs
    .iter()
    .flat_map(|scored| &scored.evaluation.quantities)
    .map(|score| score.field.as_str())
    .collect();
  if names.is_empty() {
    return;
  }
  let _ = writeln!(out, "\n{:<20} {:<38} in recycles", "quantity (factor)", "in edits");
  for name in names {
    let range = |label: Label| {
      let scores = labeled(label)
        .flat_map(|scored| &scored.evaluation.quantities)
        .filter(|score| score.field == name);
      QuantityRange::of(scores).describe()
    };
    let _ = writeln!(out, "{name:<20} {:<38} {}", range(Label::Edit), range(Label::Recycle));
  }
  let measures = |label: Label| {
    let changed = labeled(label)
      .filter(|scored| scored.evaluation.measure_changed)
      .count();
    format!("{changed} of {}", labeled(label).count())
  };
  let _ = writeln!(
    out,
    "measures changed in {} edit(s) and {} recycle(s)",
    measures(Label::Edit),
    measures(Label::Recycle)
  );
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
  render_quantity_ranges(calibration, &mut out);

  let _ = writeln!(
    out,
    "\nconfigured policy (threshold {:.3} plus field floors and any quantity limits): {} edit(s) wrongly rejected, \
     {} recycle(s) wrongly accepted, of {} pairs",
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
        before: RecordFields::default(),
        after: RecordFields::default(),
      },
      evaluation: Evaluation {
        comparison: Comparison {
          similarity,
          fields: Vec::new(),
        },
        anchor_similarity: None,
        reasons,
        quantities: Vec::new(),
        measure_changed: false,
      },
    }
  }

  /// One quantity's change: `(name, before, after)`, either side possibly absent.
  type Change<'a> = (&'a str, Option<Quantity>, Option<Quantity>);

  /// An accepted pair whose quantities changed as given.
  fn measured(label: Label, changes: &[Change<'_>]) -> ScoredPair {
    let schema = Schema::new(vec![similarity_core::FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: false,
      reject_below: None,
    }])
    .unwrap()
    .with_quantities(
      changes
        .iter()
        .map(|(name, ..)| similarity_core::QuantitySpec {
          name: (*name).to_owned(),
          reject_above_factor: Some(4.0),
          reject_incomparable: false,
        })
        .collect(),
    )
    .unwrap();
    let side = |pick: fn(&Change<'_>) -> Option<Quantity>| -> Quantities {
      changes
        .iter()
        .filter_map(|change| Some((change.0.to_owned(), pick(change)?)))
        .collect()
    };
    let mut pair = scored(label, 0.95, true);
    pair.evaluation.judge_quantities(
      &schema,
      &Policy::new(0.8, None).unwrap(),
      &side(|change| change.1.clone()),
      &Quantities::new(),
      &side(|change| change.2.clone()),
    );
    pair
  }

  fn amount(value: f64, unit: &str) -> Quantity {
    Quantity::single(value, unit)
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
    assert_eq!(pairs[1].after.texts["name"], "Vape");
    assert!(pairs[1].after.quantities.is_empty());
  }

  #[test]
  fn a_record_file_holds_text_and_quantities_in_one_object_told_apart_by_type() {
    let record: RecordFields = serde_json::from_str(
      r#"{"name": "Blueberry Gummies 100mg", "category": "Edibles > Gummies",
          "weight_per_item": {"value": 3.5, "unit": "g"},
          "thc_per_item": {"min": 18, "max": 22, "unit": "%"}}"#,
    )
    .unwrap();
    assert_eq!(record.texts.len(), 2);
    assert_eq!(record.texts["category"], "Edibles > Gummies");
    assert_eq!(record.quantities["weight_per_item"], Quantity::single(3.5, "g"));
    assert_eq!(record.quantities["thc_per_item"], Quantity::range(18.0, 22.0, "%"));

    // The same shape inside a calibration pair, which is one line of a file.
    let pairs = parse_pairs(concat!(
      r#"{"label": "edit", "before": {"name": "Gummies", "weight_per_item": {"value": 1, "unit": "g"}}, "#,
      r#""after": {"name": "Gummies", "weight_per_item": {"value": 1000, "unit": "mg"}}}"#,
    ))
    .unwrap();
    assert_eq!(
      pairs[0].after.quantities["weight_per_item"],
      Quantity::single(1000.0, "mg")
    );
  }

  #[test]
  fn a_value_that_is_neither_text_nor_a_quantity_is_refused_by_key() {
    for (text, expected) in [
      (
        r#"{"name": "Gummies", "weight_per_item": 3.5}"#,
        "`weight_per_item` must be text",
      ),
      (r#"{"name": "Gummies", "tags": ["vegan"]}"#, "`tags` must be text"),
      (r#"{"name": "Gummies", "brand": null}"#, "`brand` must be text"),
      (
        r#"{"name": "Gummies", "weight_per_item": {"value": 3.5}}"#,
        "`weight_per_item`",
      ),
      (
        r#"{"name": "Gummies", "weight_per_item": {"min": 1, "unit": "g"}}"#,
        "either `value` and `unit`",
      ),
    ] {
      let error = serde_json::from_str::<RecordFields>(text).unwrap_err().to_string();
      assert!(error.contains(expected), "{text}: {error}");
    }
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
    let report = render_comparison(&accepted, &Policy::new(0.8, None).unwrap());
    assert!(report.contains("verdict: ACCEPTED"));
    assert!(
      !report.contains("quantity"),
      "a record without quantities says nothing about them: {report}"
    );
  }

  #[test]
  fn a_comparison_renders_each_quantity_and_what_exceeded_its_limit() {
    let pair = measured(
      Label::Recycle,
      &[
        ("weight", Some(amount(3.5, "g")), Some(amount(28.0, "g"))),
        ("size", Some(amount(1.0, "g")), Some(amount(1000.0, "mg"))),
        ("thc", Some(amount(22.0, "%")), Some(amount(22.0, "mg"))),
        ("cbd", Some(amount(0.0, "mg")), Some(amount(5.0, "mg"))),
        ("volume", None, Some(amount(355.0, "ml"))),
        ("count", Some(amount(10.0, "each")), None),
      ],
    );
    let report = render_comparison(&pair.evaluation, &Policy::new(0.8, None).unwrap());
    let line = |name: &str| {
      report
        .lines()
        .find(|line| line.starts_with(name))
        .unwrap_or_else(|| panic!("no line for {name}: {report}"))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
    };
    assert_eq!(line("weight"), "weight 8.00 modified g to g << beyond limit");
    assert_eq!(line("size"), "size 1.00 unchanged g to mg");
    assert_eq!(line("thc"), "thc - incomparable % to mg");
    assert_eq!(line("cbd"), "cbd no limit modified mg to mg << beyond limit");
    assert_eq!(line("volume"), "volume - added - to ml");
    assert_eq!(line("count"), "count - removed each to -");
    assert!(
      report.contains("the measures the record is described by changed"),
      "{report}"
    );
    assert!(
      report.contains("verdict: REJECTED") && report.contains("factor of 8.00"),
      "{report}"
    );
  }

  #[test]
  fn the_calibration_report_sets_each_quantitys_factors_in_edits_beside_those_in_recycles() {
    let calibration = summarize(vec![
      measured(
        Label::Edit,
        &[("thc", Some(amount(21.3, "%")), Some(amount(22.8, "%")))],
      ),
      measured(
        Label::Edit,
        &[("thc", Some(amount(20.0, "%")), Some(amount(25.0, "%")))],
      ),
      // The same amount in another unit is no change, and must not pull the range down to 1.
      measured(
        Label::Edit,
        &[("weight", Some(amount(1.0, "g")), Some(amount(1000.0, "mg")))],
      ),
      measured(
        Label::Recycle,
        &[("weight", Some(amount(3.5, "g")), Some(amount(28.0, "g")))],
      ),
      measured(
        Label::Recycle,
        &[("thc", Some(amount(22.0, "%")), Some(amount(22.0, "mg")))],
      ),
      measured(
        Label::Recycle,
        &[
          ("weight", Some(amount(3.5, "g")), None),
          ("volume", None, Some(amount(355.0, "ml"))),
          ("thc", Some(amount(0.0, "mg")), Some(amount(10.0, "mg"))),
        ],
      ),
      scored(Label::Edit, 0.97, true),
    ]);
    let report = render_calibration(&calibration, &Policy::new(0.8, None).unwrap());
    let line = |name: &str| {
      report
        .lines()
        .find(|line| line.starts_with(name))
        .unwrap_or_else(|| panic!("no line for {name}: {report}"))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
    };
    assert_eq!(line("thc"), "thc 1.07 to 1.25 ( 2) 1 incomparable, 1 to or from zero");
    assert_eq!(line("weight"), "weight - 8.00 to 8.00 ( 1)");
    assert_eq!(line("volume"), "volume - -");
    assert!(
      report.contains("measures changed in 0 of 4 edit(s) and 2 of 3 recycle(s)"),
      "{report}"
    );
    // Limits count toward what the configured policy gets wrong: the eightfold weight and the potency from zero
    // exceed this schema's limit of 4, so those two recycles are rejected and only the third is wrongly accepted.
    assert!(
      report.contains("0 edit(s) wrongly rejected, 1 recycle(s) wrongly accepted, of 7 pairs"),
      "{report}"
    );
  }

  #[test]
  fn a_calibration_without_quantities_reports_none() {
    let calibration = summarize(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.2, false),
    ]);
    let report = render_calibration(&calibration, &Policy::new(0.8, None).unwrap());
    assert!(
      !report.contains("quantity (factor)") && !report.contains("measures changed"),
      "{report}"
    );
  }
}
