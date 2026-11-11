//! Scored pairs made by hand, for the tests of this module: no model, no files.

use similarity_core::score::{Comparison, Evaluation, Reason};
use similarity_core::{FieldSpec, Policy, Quantities, Quantity, QuantitySpec, Schema};

use super::RecordFields;
use super::calibrate::ScoredPair;
use super::pairs::{Label, Pair};

/// A pair labeled `label` that scored `similarity` and was accepted or rejected as said, with nothing else known
/// about it.
pub(crate) fn scored(label: Label, similarity: f32, accepted: bool) -> ScoredPair {
  scored_as(label, similarity, accepted, |_| {})
}

/// [`scored`], with `adjust` applied to the pair before it is returned.
pub(crate) fn scored_as(label: Label, similarity: f32, accepted: bool, adjust: impl FnOnce(&mut Pair)) -> ScoredPair {
  let reasons = if accepted {
    Vec::new()
  } else {
    vec![Reason::BelowThreshold {
      similarity,
      threshold: 0.8,
    }]
  };
  let mut pair = Pair {
    id: format!("{}-{similarity}", label.as_str()),
    label: Some(label),
    label_text: label.as_str().to_owned(),
    note: format!("{label:?} at {similarity}"),
    split: None,
    owner: None,
    variant_change: None,
    label_source: None,
    label_confidence: None,
    before: RecordFields::default(),
    after: RecordFields::default(),
    anchor: None,
  };
  adjust(&mut pair);
  ScoredPair {
    pair,
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
pub(crate) type Change<'a> = (&'a str, Option<Quantity>, Option<Quantity>);

/// The schema [`measured`] judges under: a `name` field, and each named quantity limited to a factor of 4.
pub(crate) fn measuring_schema(quantities: &[&str]) -> Schema {
  Schema::new(vec![FieldSpec {
    name: "name".to_owned(),
    weight: 1.0,
    required: false,
    reject_below: None,
  }])
  .unwrap()
  .with_quantities(
    quantities
      .iter()
      .map(|name| QuantitySpec {
        name: (*name).to_owned(),
        reject_above_factor: Some(4.0),
        reject_incomparable: false,
      })
      .collect(),
  )
  .unwrap()
}

/// A pair that scored 0.95 on its text and whose quantities changed as given, each held to a factor of 4.
pub(crate) fn measured(label: Label, changes: &[Change<'_>]) -> ScoredPair {
  let names: Vec<&str> = changes.iter().map(|(name, ..)| *name).collect();
  let side = |pick: fn(&Change<'_>) -> Option<Quantity>| -> Quantities {
    changes
      .iter()
      .filter_map(|change| Some((change.0.to_owned(), pick(change)?)))
      .collect()
  };
  let mut pair = scored(label, 0.95, true);
  pair.evaluation.judge_quantities(
    &measuring_schema(&names),
    &Policy::new(0.8, None).unwrap(),
    &side(|change| change.1.clone()),
    &Quantities::new(),
    &side(|change| change.2.clone()),
  );
  pair
}

pub(crate) fn amount(value: f64, unit: &str) -> Quantity {
  Quantity::single(value, unit)
}
