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
//! - [`events`]: the same calibration as a stream of result events, for tools, and reading one back.
//! - [`diff`]: comparing two calibrations, or one calibration under two configurations.

pub mod calibrate;
pub mod diff;
pub mod events;
pub mod pairs;
pub mod report;
pub mod stats;
#[cfg(test)]
pub(crate) mod testing;

use std::collections::BTreeMap;

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use similarity_core::quantity::first_values;
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

  /// The model itself, so that a test can put the same weights behind an engine and compare the two.
  #[cfg(test)]
  pub(crate) fn embedder(&self) -> &Embedder {
    &self.embedder
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
  /// Without an `anchor` the pair has no history: nothing can be "restored", and drift is not measured. With one,
  /// the pair is a record created as `anchor` whose stored version is now `before`: the anchor is built as an
  /// import with an original version builds it (D20), a field or quantity `before` lacks is compared with it, and
  /// the similarity to it is reported and held to the policy's `anchor_threshold`.
  ///
  /// # Errors
  ///
  /// Returns an error if a record does not fit the schema or the model fails.
  pub fn compare(
    &self,
    before: &RecordFields,
    after: &RecordFields,
    anchor: Option<&RecordFields>,
  ) -> anyhow::Result<Evaluation> {
    let dimension = self.embedder.dimension();
    let (baseline, old_quantities) = self.prepared(before, None)?;
    let (candidate, new_quantities) = self.prepared(after, Some(&baseline))?;
    let first = anchor
      .map(|anchor| -> anyhow::Result<(Vec<f32>, Quantities)> {
        let (original, original_quantities) = self.prepared(anchor, None)?;
        Ok((
          // The original's values, plus the stored value of anything the original did not have.
          score::extend_anchor(&self.schema, dimension, &original.composite, &baseline.composite)?,
          first_values(&original_quantities, &old_quantities),
        ))
      })
      .transpose()?;
    let (anchor_composite, anchor_quantities) = match first {
      Some((composite, quantities)) => (Some(composite), quantities),
      None => (None, Quantities::new()),
    };
    let mut evaluation = score::evaluate(
      &self.schema,
      &self.policy,
      dimension,
      &baseline,
      anchor_composite.as_deref(),
      &candidate,
    )?;
    evaluation.judge_quantities(
      &self.schema,
      &self.policy,
      &old_quantities,
      &anchor_quantities,
      &new_quantities,
    );
    Ok(evaluation)
  }
}

#[cfg(test)]
mod tests {
  use similarity_core::score::Reason;
  use similarity_core::testing::tiny_embedder;
  use similarity_core::{EmbedderOptions, FieldSpec, QuantitySpec};

  use super::*;

  /// A scorer on the tiny model: two optional text fields and a weight, with drift held to `anchor_threshold`.
  fn offline(anchor_threshold: Option<f32>) -> Offline {
    let field = |name: &str| FieldSpec {
      name: name.to_owned(),
      weight: 1.0,
      required: false,
      reject_below: None,
    };
    let schema = Schema::new(vec![field("name"), field("brand")])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: Some(4.0),
        reject_incomparable: false,
      }])
      .unwrap();
    let options = EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    let policy = Policy::new(0.0, anchor_threshold).unwrap();
    Offline::new(tiny_embedder(&options), schema, policy)
  }

  fn fields(json: &str) -> RecordFields {
    serde_json::from_str(json).unwrap()
  }

  /// D8's two-step swap, as one pair: the brand was blanked in an earlier, accepted step, and now comes back as
  /// something else. Without the anchor that is a field being filled in; with it, it is a field being replaced.
  #[test]
  fn a_field_blanked_and_refilled_is_scored_against_the_anchor() {
    let offline = offline(None);
    let anchor = fields(r#"{"name": "blueberry gummies", "brand": "indica"}"#);
    let before = fields(r#"{"name": "blueberry gummies"}"#);
    let after = fields(r#"{"name": "blueberry gummies", "brand": "vape cartridge"}"#);
    let brand = |evaluation: &Evaluation| {
      let score = evaluation
        .comparison
        .fields
        .iter()
        .find(|score| score.field == "brand")
        .unwrap();
      (score.change, score.similarity)
    };

    let unanchored = offline.compare(&before, &after, None).unwrap();
    assert_eq!(brand(&unanchored), (FieldChange::Added, None));
    assert_eq!(unanchored.anchor_similarity, None);

    let anchored = offline.compare(&before, &after, Some(&anchor)).unwrap();
    let (change, similarity) = brand(&anchored);
    assert_eq!(change, FieldChange::Restored);
    assert!(
      similarity.unwrap() < 0.999,
      "another brand is not the first one: {similarity:?}"
    );
    assert!(anchored.comparison.similarity < unanchored.comparison.similarity);
    assert!(anchored.anchor_similarity.is_some());

    // Refilled with what it first held, it is the record it always was.
    let restored = offline.compare(&before, &anchor, Some(&anchor)).unwrap();
    assert!((restored.comparison.similarity - 1.0).abs() < 1e-5);
  }

  #[test]
  fn a_quantity_removed_and_refilled_is_compared_with_the_anchor() {
    let offline = offline(None);
    let anchor = fields(r#"{"name": "flower", "weight": {"value": 3.5, "unit": "g"}}"#);
    let before = fields(r#"{"name": "flower"}"#);
    let after = fields(r#"{"name": "flower", "weight": {"value": 28, "unit": "g"}}"#);

    let unanchored = offline.compare(&before, &after, None).unwrap();
    assert_eq!(unanchored.quantities[0].change, QuantityChange::Added);
    assert!(unanchored.accepted(), "{:?}", unanchored.reasons);

    let anchored = offline.compare(&before, &after, Some(&anchor)).unwrap();
    let weight = &anchored.quantities[0];
    assert_eq!(weight.change, QuantityChange::Restored);
    assert!((weight.factor.unwrap() - 8.0).abs() < 1e-9 && weight.beyond_limit);
    assert!(
      matches!(&anchored.reasons[..], [Reason::QuantityBeyondFactor { field, .. }] if field == "weight"),
      "{:?}",
      anchored.reasons
    );
  }

  #[test]
  fn an_anchored_pair_is_held_to_the_anchor_threshold() {
    let anchor = fields(r#"{"name": "blueberry gummies"}"#);
    let now = fields(r#"{"name": "vape cartridge"}"#);

    // The stored version and the new one are identical; only the distance from the first version can reject.
    let strict = offline(Some(0.999)).compare(&now, &now, Some(&anchor)).unwrap();
    assert!(
      strict.anchor_similarity.unwrap() < 0.999,
      "{:?}",
      strict.anchor_similarity
    );
    assert!(
      matches!(&strict.reasons[..], [Reason::AnchorDrift { .. }]),
      "{:?}",
      strict.reasons
    );

    let reporting = offline(None).compare(&now, &now, Some(&anchor)).unwrap();
    assert!(reporting.anchor_similarity.is_some() && reporting.accepted());
    let unanchored = offline(Some(0.999)).compare(&now, &now, None).unwrap();
    assert!(unanchored.anchor_similarity.is_none() && unanchored.accepted());
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
