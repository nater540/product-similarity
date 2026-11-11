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
//!
//! Modules:
//!
//! - this one: a record as a file holds it ([`RecordFields`]), and [`Offline`], which scores one pair.
//! - [`pairs`]: labeled pairs, as the strict and the full file formats hold them.
//! - [`calibrate`]: scoring a file of pairs, and the threshold that separates the two labels.
//! - [`stats`]: what the scored pairs show beyond their totals: ranges, slices, error rates with intervals.
//! - [`report`]: the reports a person reads at a terminal.
//! - [`events`]: the same calibration as a stream of result events, for tools.

pub mod calibrate;
pub mod events;
pub mod pairs;
pub mod report;
pub mod stats;
#[cfg(test)]
pub(crate) mod testing;

use std::collections::BTreeMap;

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use similarity_core::score::{self, Evaluation};
use similarity_core::{
  Embedder, FieldChange, Policy, PreparedRecord, Quantities, Quantity, QuantityChange, RecordVectors, Schema,
};

pub use calibrate::{Calibration, ScoredPair, Suggestion, calibrate, suggest_threshold};
pub use pairs::{Confidence, Label, Pair, Pairs, Selection, parse_full_pairs, parse_pairs};
pub use report::{render_calibration, render_comparison};

/// A field's change as the reports and events spell it.
pub(crate) fn change_name(change: FieldChange) -> &'static str {
  match change {
    FieldChange::Unchanged => "unchanged",
    FieldChange::Modified => "modified",
    FieldChange::Added => "added",
    FieldChange::Removed => "removed",
    FieldChange::Restored => "restored",
  }
}

/// A quantity's change as the reports and events spell it.
pub(crate) fn quantity_change_name(change: QuantityChange) -> &'static str {
  match change {
    QuantityChange::Unchanged => "unchanged",
    QuantityChange::Modified => "modified",
    QuantityChange::Incomparable => "incomparable",
    QuantityChange::Added => "added",
    QuantityChange::Removed => "removed",
    QuantityChange::Restored => "restored",
  }
}

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

// Written as it is read: one object, text fields as strings and quantities as objects, keys in order.
impl Serialize for RecordFields {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let mut map = serializer.serialize_map(Some(self.texts.len().saturating_add(self.quantities.len())))?;
    for (key, text) in &self.texts {
      map.serialize_entry(key, text)?;
    }
    for (key, quantity) in &self.quantities {
      map.serialize_entry(key, quantity)?;
    }
    map.end()
  }
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

  /// The fingerprint of the model in use.
  pub fn fingerprint(&self) -> &str {
    self.embedder.fingerprint()
  }

  /// The width of one field's vector.
  pub fn dimension(&self) -> usize {
    self.embedder.dimension()
  }

  /// The token limit texts are truncated to.
  pub fn max_tokens(&self) -> usize {
    self.embedder.max_tokens()
  }

  /// The schema records are scored under.
  pub fn schema(&self) -> &Schema {
    &self.schema
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

#[cfg(test)]
mod tests {
  use super::*;

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
      pairs.pairs[0].after.quantities["weight_per_item"],
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
}
