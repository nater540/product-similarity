//! Comparing a proposed revision of a record against what is stored, and deciding whether to accept it.
//!
//! The question this module answers is "is this still the same record?". A merchant rewording a description or
//! adding a potency to a name is an edit; a merchant turning a blueberry gummy into a vape cartridge under the same
//! id is a different record wearing the old one's history. Three checks, each of which can reject on its own:
//!
//! 1. **Overall similarity**: the weighted mean of per-field cosines against the current baseline.
//! 2. **Per-field floors**: a field that matters (the name, the category) may not change beyond its own limit, even
//!    when unchanged long fields keep the weighted mean high.
//! 3. **Anchor drift** (optional): similarity against the record as it was first stored. Without this, a record can
//!    be walked anywhere in a series of individually acceptable steps, because every accepted step moves the
//!    baseline.
//!
//! The anchor has a second job that is not optional. A field the baseline lacks cannot be compared with the
//! baseline, and leaving it out of the score entirely would let a field be replaced in two steps: blank it, then
//! fill it with something else. So a field that comes back is scored against the value it had in the anchor
//! ([`FieldChange::Restored`]). Only a field the record has never had a value for is taken on trust.

use std::fmt;

use crate::error::{Error, Result};
use crate::record::RecordVectors;
use crate::schema::Schema;
use crate::vector::{cosine, normalize};

/// What happened to one field between the baseline and the candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldChange {
  /// The normalized text is identical.
  Unchanged,
  /// Both sides have a value and the text differs.
  Modified,
  /// Only the candidate has a value, and the record has never had one for this field.
  Added,
  /// Only the baseline has a value.
  Removed,
  /// The baseline has no value, but the record had one when it was first stored, and the candidate has one again.
  /// The similarity is against that original value.
  Restored,
}

/// The comparison of one field.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldScore {
  /// The field's name.
  pub field: String,
  /// The field's weight in the schema.
  pub weight: f32,
  /// What happened to the field.
  pub change: FieldChange,
  /// Cosine similarity of the two values. `None` for [`FieldChange::Added`] and [`FieldChange::Removed`], where
  /// there is nothing to compare; exactly `1.0` when [`FieldChange::Unchanged`]; against the anchor's value when
  /// [`FieldChange::Restored`].
  pub similarity: Option<f32>,
  /// Whether [`Self::similarity`] is below the field's `reject_below` floor.
  pub below_floor: bool,
}

/// A field-by-field comparison of two versions of a record.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
  /// The weighted mean of the per-field similarities over the fields that could be compared. `0.0` if there are
  /// none.
  ///
  /// Fields that were added for the first time or removed are reported but excluded from the mean: filling in a
  /// description that never existed is ordinary maintenance, not evidence of a different product. A field that
  /// must not disappear should be marked `required` in the schema instead. A restored field is included, scored
  /// against its original value.
  pub similarity: f32,
  /// One entry per field that either side has a value for, in schema order.
  pub fields: Vec<FieldScore>,
}

/// Compares two versions of a record.
///
/// `anchor` is the composite holding the first value the record had for each field (see [`extend_anchor`]). With
/// it, a field the baseline lacks but the anchor has is scored against the anchor; without it, such a field is
/// reported as added and not scored.
///
/// # Errors
///
/// Returns [`Error::Incompatible`] if any composite does not have this schema's layout.
pub fn compare(
  schema: &Schema,
  dimension: usize,
  baseline: &RecordVectors,
  anchor: Option<&[f32]>,
  candidate: &RecordVectors,
) -> Result<Comparison> {
  let old_blocks = schema.blocks(dimension, &baseline.composite)?;
  let new_blocks = schema.blocks(dimension, &candidate.composite)?;
  let original_blocks = match anchor {
    Some(anchor) => schema.blocks(dimension, anchor)?,
    None => vec![None; schema.fields().len()],
  };

  let mut fields = Vec::with_capacity(schema.fields().len());
  let mut weighted = 0.0_f64;
  let mut total = 0.0_f64;
  for (((spec, old), new), original) in schema
    .fields()
    .iter()
    .zip(old_blocks)
    .zip(new_blocks)
    .zip(original_blocks)
  {
    let (change, similarity) = match (old, new) {
      (None, None) => continue,
      (None, Some(new)) => match original {
        Some(original) => (FieldChange::Restored, Some(cosine(original, new)?)),
        None => (FieldChange::Added, None),
      },
      (Some(_), None) => (FieldChange::Removed, None),
      (Some(old), Some(new)) => {
        let old_hash = baseline.hashes.get(&spec.name);
        if old_hash.is_some() && old_hash == candidate.hashes.get(&spec.name) {
          (FieldChange::Unchanged, Some(1.0))
        } else {
          (FieldChange::Modified, Some(cosine(old, new)?))
        }
      }
    };
    if let Some(similarity) = similarity {
      weighted += f64::from(spec.weight) * f64::from(similarity);
      total += f64::from(spec.weight);
    }
    fields.push(FieldScore {
      field: spec.name.clone(),
      weight: spec.weight,
      change,
      similarity,
      below_floor: matches!((similarity, spec.reject_below), (Some(found), Some(floor)) if found < floor),
    });
  }

  Ok(Comparison {
    similarity: weighted_mean(weighted, total),
    fields,
  })
}

/// The anchor after an accepted revision: unchanged, except that a field it has never had a value for takes the
/// candidate's.
///
/// The anchor therefore holds the *first* value the record had for every field it has ever had, which is what
/// [`compare`] scores a restored field against and what drift is measured from. Values the anchor already holds
/// are never replaced here; only a forced write does that, by starting a new anchor.
///
/// # Errors
///
/// Returns [`Error::Incompatible`] if either composite does not have this schema's layout.
pub fn extend_anchor(schema: &Schema, dimension: usize, anchor: &[f32], candidate: &[f32]) -> Result<Vec<f32>> {
  let anchor_blocks = schema.blocks(dimension, anchor)?;
  let candidate_blocks = schema.blocks(dimension, candidate)?;
  let gains_a_field = anchor_blocks
    .iter()
    .zip(&candidate_blocks)
    .any(|(original, new)| original.is_none() && new.is_some());
  if !gains_a_field {
    return Ok(anchor.to_vec());
  }

  // The blocks of two composites are scaled differently (each composite is normalized as a whole), so every
  // block goes back to unit length before the two are packed into one.
  let units: Vec<Option<Vec<f32>>> = anchor_blocks
    .into_iter()
    .zip(candidate_blocks)
    .map(|(original, new)| {
      original.or(new).map(|block| {
        let mut unit = block.to_vec();
        normalize(&mut unit);
        unit
      })
    })
    .collect();
  schema.compose(dimension, units.iter().map(Option::as_deref))
}

/// The weighted mean of per-field cosines between two bare composites, over the fields both have a value for.
///
/// This is [`compare`] without text fingerprints, for when only a vector is on hand (the anchor).
///
/// # Errors
///
/// Returns [`Error::Incompatible`] if either composite does not have this schema's layout.
pub fn weighted_similarity(schema: &Schema, dimension: usize, left: &[f32], right: &[f32]) -> Result<f32> {
  let left_blocks = schema.blocks(dimension, left)?;
  let right_blocks = schema.blocks(dimension, right)?;
  let mut weighted = 0.0_f64;
  let mut total = 0.0_f64;
  for ((spec, a), b) in schema.fields().iter().zip(left_blocks).zip(right_blocks) {
    if let (Some(a), Some(b)) = (a, b) {
      weighted += f64::from(spec.weight) * f64::from(cosine(a, b)?);
      total += f64::from(spec.weight);
    }
  }
  Ok(weighted_mean(weighted, total))
}

fn weighted_mean(weighted: f64, total: f64) -> f32 {
  if total <= 0.0 {
    return 0.0;
  }
  #[expect(clippy::cast_possible_truncation, reason = "clamped to the cosine range first")]
  let mean = (weighted / total).clamp(-1.0, 1.0) as f32;
  mean
}

/// The thresholds a revision is judged against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
  /// A revision whose overall similarity to the baseline is below this is rejected.
  pub threshold: f32,
  /// If set, a revision whose overall similarity to the anchor is below this is rejected.
  pub anchor_threshold: Option<f32>,
}

impl Policy {
  /// Validates the thresholds.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Schema`] if a threshold lies outside the cosine range `[-1, 1]`.
  pub fn new(threshold: f32, anchor_threshold: Option<f32>) -> Result<Self> {
    for (name, value) in [("threshold", Some(threshold)), ("anchor_threshold", anchor_threshold)] {
      if let Some(value) = value
        && !(-1.0..=1.0).contains(&value)
      {
        return Err(Error::Schema(format!(
          "`{name}` is {value}, outside the cosine range [-1, 1]"
        )));
      }
    }
    Ok(Self {
      threshold,
      anchor_threshold,
    })
  }
}

/// Why a revision was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum Reason {
  /// The overall similarity to the baseline is below the policy threshold.
  BelowThreshold {
    /// The overall similarity.
    similarity: f32,
    /// The threshold it fell below.
    threshold: f32,
  },
  /// One field changed beyond its own floor.
  FieldBelowFloor {
    /// The field.
    field: String,
    /// That field's similarity.
    similarity: f32,
    /// The floor it fell below.
    floor: f32,
  },
  /// The record has drifted too far from its original form.
  AnchorDrift {
    /// The overall similarity to the anchor.
    similarity: f32,
    /// The threshold it fell below.
    threshold: f32,
  },
}

impl fmt::Display for Reason {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::BelowThreshold { similarity, threshold } => {
        write!(
          f,
          "overall similarity {similarity:.3} is below the threshold {threshold:.3}"
        )
      }
      Self::FieldBelowFloor {
        field,
        similarity,
        floor,
      } => {
        write!(
          f,
          "field `{field}` similarity {similarity:.3} is below its floor {floor:.3}"
        )
      }
      Self::AnchorDrift { similarity, threshold } => {
        write!(
          f,
          "similarity to the original record {similarity:.3} is below the drift threshold {threshold:.3}"
        )
      }
    }
  }
}

/// The verdict on a proposed revision.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
  /// The comparison against the current baseline.
  pub comparison: Comparison,
  /// Overall similarity to the anchor, when one was supplied.
  pub anchor_similarity: Option<f32>,
  /// Why the revision is rejected; empty means accepted.
  pub reasons: Vec<Reason>,
}

impl Evaluation {
  /// Whether the revision passed every check.
  pub fn accepted(&self) -> bool {
    self.reasons.is_empty()
  }
}

/// Judges `candidate` as a revision of `baseline`.
///
/// `anchor` is the composite of the record as first stored, if the caller tracks one (see [`extend_anchor`]). Its
/// overall similarity is always reported when supplied, and only enforced when the policy sets an
/// `anchor_threshold`. Independently of that setting, a field the baseline lacks is scored against it.
///
/// # Errors
///
/// Returns [`Error::Incompatible`] if any composite does not have this schema's layout.
pub fn evaluate(
  schema: &Schema,
  policy: &Policy,
  dimension: usize,
  baseline: &RecordVectors,
  anchor: Option<&[f32]>,
  candidate: &RecordVectors,
) -> Result<Evaluation> {
  let comparison = compare(schema, dimension, baseline, anchor, candidate)?;
  let anchor_similarity = anchor
    .map(|anchor| weighted_similarity(schema, dimension, anchor, &candidate.composite))
    .transpose()?;

  let mut reasons = Vec::new();
  if comparison.similarity < policy.threshold {
    reasons.push(Reason::BelowThreshold {
      similarity: comparison.similarity,
      threshold: policy.threshold,
    });
  }
  for (score, spec) in comparison
    .fields
    .iter()
    .filter_map(|score| Some((score, schema.field(&score.field)?)))
  {
    if let (true, Some(similarity), Some(floor)) = (score.below_floor, score.similarity, spec.reject_below) {
      reasons.push(Reason::FieldBelowFloor {
        field: score.field.clone(),
        similarity,
        floor,
      });
    }
  }
  if let (Some(similarity), Some(threshold)) = (anchor_similarity, policy.anchor_threshold)
    && similarity < threshold
  {
    reasons.push(Reason::AnchorDrift { similarity, threshold });
  }

  Ok(Evaluation {
    comparison,
    anchor_similarity,
    reasons,
  })
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;

  use super::*;
  use crate::schema::FieldSpec;
  use crate::schema::testing::product_schema;
  use crate::vector::normalize;

  const DIM: usize = 2;

  /// A unit vector at `degrees` in the plane, so the cosine between two of them is the cosine of the difference.
  fn at(degrees: f32) -> Vec<f32> {
    let radians = degrees.to_radians();
    let mut vector = vec![radians.cos(), radians.sin()];
    assert!(normalize(&mut vector));
    vector
  }

  /// Builds stored vectors from `(field, angle, text-stand-in)` triples.
  fn record(schema: &Schema, fields: &[(&str, f32, &str)]) -> RecordVectors {
    let units: Vec<Option<Vec<f32>>> = schema
      .fields()
      .iter()
      .map(|spec| {
        fields
          .iter()
          .find(|(name, ..)| *name == spec.name)
          .map(|(_, angle, _)| at(*angle))
      })
      .collect();
    let composite = schema.compose(DIM, units.iter().map(Option::as_deref)).unwrap();
    let hashes: BTreeMap<String, String> = fields
      .iter()
      .map(|(name, _, text)| ((*name).to_owned(), (*text).to_owned()))
      .collect();
    RecordVectors { composite, hashes }
  }

  fn score<'a>(comparison: &'a Comparison, field: &str) -> &'a FieldScore {
    comparison.fields.iter().find(|score| score.field == field).unwrap()
  }

  #[test]
  fn an_identical_record_scores_exactly_one() {
    let schema = product_schema();
    let stored = record(
      &schema,
      &[
        ("name", 10.0, "n"),
        ("category", 80.0, "c"),
        ("description", 200.0, "d"),
      ],
    );
    let comparison = compare(&schema, DIM, &stored, None, &stored.clone()).unwrap();
    assert!((comparison.similarity - 1.0).abs() < f32::EPSILON);
    assert!(
      comparison
        .fields
        .iter()
        .all(|score| score.change == FieldChange::Unchanged)
    );
  }

  #[test]
  fn the_overall_score_is_the_weighted_mean_of_the_field_scores() {
    let schema = product_schema(); // category 2, description 1, name 4
    let old = record(
      &schema,
      &[("name", 0.0, "n1"), ("category", 0.0, "c"), ("description", 0.0, "d1")],
    );
    let new = record(
      &schema,
      &[
        ("name", 60.0, "n2"),
        ("category", 0.0, "c"),
        ("description", 90.0, "d2"),
      ],
    );

    let comparison = compare(&schema, DIM, &old, None, &new).unwrap();
    assert!((score(&comparison, "name").similarity.unwrap() - 0.5).abs() < 1e-5);
    assert!(score(&comparison, "description").similarity.unwrap().abs() < 1e-5);
    assert_eq!(score(&comparison, "category").change, FieldChange::Unchanged);
    // (4 * 0.5 + 1 * 0.0 + 2 * 1.0) / 7
    assert!((comparison.similarity - 4.0 / 7.0).abs() < 1e-5);
  }

  #[test]
  fn added_and_removed_fields_are_reported_but_do_not_move_the_score() {
    let schema = product_schema();
    let old = record(&schema, &[("name", 0.0, "n"), ("description", 45.0, "d")]);
    let new = record(&schema, &[("name", 0.0, "n"), ("category", 120.0, "c")]);

    let comparison = compare(&schema, DIM, &old, None, &new).unwrap();
    assert_eq!(score(&comparison, "category").change, FieldChange::Added);
    assert_eq!(score(&comparison, "description").change, FieldChange::Removed);
    assert_eq!(score(&comparison, "category").similarity, None);
    assert!(
      (comparison.similarity - 1.0).abs() < 1e-6,
      "only `name` is comparable, and it is unchanged"
    );
  }

  #[test]
  fn a_field_that_comes_back_is_scored_against_the_value_it_first_had() {
    let schema = product_schema();
    let original = record(&schema, &[("name", 0.0, "n"), ("category", 0.0, "c")]);
    // The baseline is the record after its category was blanked.
    let blanked = record(&schema, &[("name", 0.0, "n")]);
    let refilled = record(&schema, &[("name", 0.0, "n"), ("category", 60.0, "other")]);

    // Without the anchor there is nothing to compare the category with, and the swap scores a perfect 1.
    let blind = compare(&schema, DIM, &blanked, None, &refilled).unwrap();
    assert_eq!(score(&blind, "category").change, FieldChange::Added);
    assert!((blind.similarity - 1.0).abs() < 1e-6);

    let comparison = compare(&schema, DIM, &blanked, Some(&original.composite), &refilled).unwrap();
    let category = score(&comparison, "category");
    assert_eq!(category.change, FieldChange::Restored);
    assert!((category.similarity.unwrap() - 0.5).abs() < 1e-5, "cos 60 degrees");
    // Exactly what changing the category in one step would have scored.
    let direct = compare(&schema, DIM, &original, None, &refilled).unwrap();
    assert!((comparison.similarity - direct.similarity).abs() < 1e-6);
    assert!(comparison.similarity < 1.0);
  }

  #[test]
  fn a_restored_field_is_held_to_its_floor() {
    let mut category = FieldSpec {
      name: "category".to_owned(),
      weight: 1.0,
      required: false,
      reject_below: Some(0.8),
    };
    let name = FieldSpec {
      name: "name".to_owned(),
      reject_below: None,
      ..category.clone()
    };
    category.weight = 0.5;
    let schema = Schema::new(vec![name, category]).unwrap();
    let policy = Policy::new(0.5, None).unwrap();
    let original = record(&schema, &[("name", 0.0, "n"), ("category", 0.0, "c")]);
    let blanked = record(&schema, &[("name", 0.0, "n")]);
    let refilled = record(&schema, &[("name", 0.0, "n"), ("category", 60.0, "other")]);

    let evaluation = evaluate(&schema, &policy, DIM, &blanked, Some(&original.composite), &refilled).unwrap();
    assert!(
      matches!(evaluation.reasons.as_slice(), [Reason::FieldBelowFloor { field, .. }] if field == "category"),
      "{:?}",
      evaluation.reasons
    );
  }

  #[test]
  fn the_anchor_takes_the_first_value_of_a_field_and_never_a_later_one() {
    let schema = product_schema();
    let anchor = record(&schema, &[("name", 0.0, "n")]).composite;
    let with_category = record(&schema, &[("name", 30.0, "n2"), ("category", 90.0, "c")]).composite;

    let extended = extend_anchor(&schema, DIM, &anchor, &with_category).unwrap();
    let expected = record(&schema, &[("name", 0.0, "n"), ("category", 90.0, "c")]).composite;
    for (found, wanted) in extended.iter().zip(&expected) {
      assert!(
        (found - wanted).abs() < 1e-6,
        "the name stays, the category is taken: {extended:?}"
      );
    }

    // A later, different category does not displace the first, and a field going missing changes nothing.
    let later = record(&schema, &[("name", 0.0, "n"), ("category", 10.0, "c2")]).composite;
    assert_eq!(extend_anchor(&schema, DIM, &extended, &later).unwrap(), extended);
    assert_eq!(extend_anchor(&schema, DIM, &extended, &anchor).unwrap(), extended);
  }

  #[test]
  fn records_with_no_field_in_common_score_zero() {
    let schema = product_schema();
    let old = record(&schema, &[("name", 0.0, "n")]);
    let new = record(&schema, &[("description", 0.0, "d")]);
    assert!(compare(&schema, DIM, &old, None, &new).unwrap().similarity.abs() < f32::EPSILON);
  }

  #[test]
  fn a_revision_above_the_threshold_is_accepted() {
    let schema = product_schema();
    let policy = Policy::new(0.8, None).unwrap();
    let old = record(&schema, &[("name", 0.0, "n1"), ("category", 0.0, "c")]);
    let new = record(&schema, &[("name", 20.0, "n2"), ("category", 0.0, "c")]);
    let evaluation = evaluate(&schema, &policy, DIM, &old, None, &new).unwrap();
    assert!(evaluation.accepted(), "{:?}", evaluation.reasons);
    assert_eq!(evaluation.anchor_similarity, None);
  }

  #[test]
  fn a_revision_below_the_threshold_is_rejected_with_the_numbers() {
    let schema = product_schema();
    let policy = Policy::new(0.8, None).unwrap();
    let old = record(&schema, &[("name", 0.0, "n1"), ("category", 0.0, "c1")]);
    let new = record(&schema, &[("name", 90.0, "n2"), ("category", 90.0, "c2")]);
    let evaluation = evaluate(&schema, &policy, DIM, &old, None, &new).unwrap();
    assert!(!evaluation.accepted());
    assert!(
      matches!(evaluation.reasons[..], [Reason::BelowThreshold { threshold, .. }] if (threshold - 0.8).abs() < 1e-6)
    );
    assert!(evaluation.reasons[0].to_string().contains("below the threshold"));
  }

  #[test]
  fn a_field_floor_rejects_even_when_the_overall_score_passes() {
    let floored = |name: &str, weight: f32, floor: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below: floor,
    };
    // A heavy description props the mean up; the name has its own floor.
    let schema = Schema::new(vec![floored("name", 1.0, Some(0.5)), floored("description", 9.0, None)]).unwrap();
    let policy = Policy::new(0.8, None).unwrap();
    let old = record(&schema, &[("name", 0.0, "n1"), ("description", 0.0, "d")]);
    let new = record(&schema, &[("name", 90.0, "n2"), ("description", 0.0, "d")]);

    let evaluation = evaluate(&schema, &policy, DIM, &old, None, &new).unwrap();
    assert!(evaluation.comparison.similarity > 0.8);
    assert!(matches!(&evaluation.reasons[..], [Reason::FieldBelowFloor { field, .. }] if field == "name"));
    assert!(score(&evaluation.comparison, "name").below_floor);
  }

  #[test]
  fn anchor_drift_is_reported_always_and_enforced_only_when_configured() {
    let schema = product_schema();
    let original = record(&schema, &[("name", 0.0, "n0"), ("category", 0.0, "c")]);
    let current = record(&schema, &[("name", 40.0, "n1"), ("category", 0.0, "c")]);
    let proposed = record(&schema, &[("name", 80.0, "n2"), ("category", 0.0, "c")]);
    let anchor = Some(&original.composite[..]);

    // Each step is 40 degrees on the name (cos 0.766), which a 0.8 overall threshold lets through...
    let lenient = Policy::new(0.8, None).unwrap();
    let evaluation = evaluate(&schema, &lenient, DIM, &current, anchor, &proposed).unwrap();
    assert!(evaluation.accepted());
    let drift = evaluation.anchor_similarity.unwrap();
    assert!(
      drift < evaluation.comparison.similarity,
      "two steps away is further than one"
    );

    // ...but two steps from the original is too far once drift is enforced.
    let strict = Policy::new(0.8, Some(0.8)).unwrap();
    let evaluation = evaluate(&schema, &strict, DIM, &current, anchor, &proposed).unwrap();
    assert!(matches!(evaluation.reasons[..], [Reason::AnchorDrift { .. }]));
  }

  #[test]
  fn thresholds_outside_the_cosine_range_are_refused() {
    assert!(Policy::new(1.2, None).is_err());
    assert!(Policy::new(0.8, Some(-3.0)).is_err());
    assert!(Policy::new(f32::NAN, None).is_err());
    assert!(Policy::new(-1.0, Some(1.0)).is_ok());
  }

  #[test]
  fn weighted_similarity_matches_compare_when_every_field_changed() {
    let schema = product_schema();
    let old = record(
      &schema,
      &[
        ("name", 0.0, "n1"),
        ("category", 10.0, "c1"),
        ("description", 20.0, "d1"),
      ],
    );
    let new = record(
      &schema,
      &[
        ("name", 30.0, "n2"),
        ("category", 50.0, "c2"),
        ("description", 100.0, "d2"),
      ],
    );
    let compared = compare(&schema, DIM, &old, None, &new).unwrap().similarity;
    let bare = weighted_similarity(&schema, DIM, &old.composite, &new.composite).unwrap();
    assert!((compared - bare).abs() < 1e-6);
  }
}
