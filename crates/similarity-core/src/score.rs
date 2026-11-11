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
//!
//! A record's **quantities** (see [`crate::quantity`]) are judged beside all of this and never inside it. They do
//! not enter the weighted mean, so the thresholds above mean the same with or without them. Each is compared as a
//! number and reported; it rejects only where the schema sets a limit for it, or the policy asks for a change of
//! measure to reject ([`Evaluation::judge_quantities`]).

use std::fmt;

use crate::error::{Error, Result};
use crate::quantity::{self, Quantities, QuantityChange, QuantityScore};
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
    fields.push(FieldScore {
      field: spec.name.clone(),
      weight: spec.weight,
      change,
      similarity,
      below_floor: false,
    });
  }

  let mut comparison = Comparison {
    similarity: 0.0,
    fields,
  };
  comparison.weigh(schema);
  Ok(comparison)
}

impl Comparison {
  /// Applies `schema`'s weights and floors to the field similarities already measured: sets every field's
  /// `weight` and `below_floor`, and [`Self::similarity`] to the weighted mean.
  ///
  /// [`compare`] ends with this. It is public so that a comparison measured once can be weighed again under
  /// another configuration without embedding anything, which is all a change of weights or floors needs (the
  /// per-field cosines do not depend on either). A field the schema does not define keeps no weight and is left
  /// out of the mean.
  pub fn weigh(&mut self, schema: &Schema) {
    let mut weighted = 0.0_f64;
    let mut total = 0.0_f64;
    for score in &mut self.fields {
      let Some(spec) = schema.field(&score.field) else {
        score.weight = 0.0;
        score.below_floor = false;
        continue;
      };
      score.weight = spec.weight;
      score.below_floor = matches!((score.similarity, spec.reject_below), (Some(found), Some(floor)) if found < floor);
      if let Some(similarity) = score.similarity {
        weighted += f64::from(spec.weight) * f64::from(similarity);
        total += f64::from(spec.weight);
      }
    }
    self.similarity = weighted_mean(weighted, total);
  }
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
  /// Reject a revision that changes which measures the record is described by (see
  /// [`Evaluation::measure_changed`]). Off unless switched on, like every quantity check.
  pub reject_on_measure_change: bool,
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
      reject_on_measure_change: false,
    })
  }

  /// The same policy with another overall threshold, for a request that overrides the configured one.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Schema`] if the threshold lies outside the cosine range `[-1, 1]`.
  pub fn with_threshold(self, threshold: f32) -> Result<Self> {
    Ok(Self {
      threshold: Self::new(threshold, self.anchor_threshold)?.threshold,
      ..self
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
  /// One quantity changed by more than its own limit.
  QuantityBeyondFactor {
    /// The quantity.
    field: String,
    /// The factor it changed by, or `None` if it moved between zero and a value above zero, where there is none.
    factor: Option<f64>,
    /// The limit it exceeded.
    limit: f64,
  },
  /// One quantity is now in a unit that cannot be compared with the one it had, and its schema entry forbids that.
  QuantityIncomparable {
    /// The quantity.
    field: String,
    /// The unit it had.
    previous_unit: String,
    /// The unit it has now.
    unit: String,
  },
  /// The measures the record is described by changed, and the policy forbids that.
  MeasureChanged,
}

impl Reason {
  /// A short, stable name for the kind of reason, for logs, reports, and anything else that must not parse the
  /// sentence [`fmt::Display`] writes. These six strings are part of what the tools emit: do not rename one.
  pub fn code(&self) -> &'static str {
    match self {
      Self::BelowThreshold { .. } => "below_threshold",
      Self::FieldBelowFloor { .. } => "field_below_floor",
      Self::AnchorDrift { .. } => "anchor_drift",
      Self::QuantityBeyondFactor { .. } => "quantity_beyond_factor",
      Self::QuantityIncomparable { .. } => "quantity_incomparable",
      Self::MeasureChanged => "measure_changed",
    }
  }

  /// The field or quantity the reason is about, if it is about one.
  pub fn subject(&self) -> Option<&str> {
    match self {
      Self::FieldBelowFloor { field, .. }
      | Self::QuantityBeyondFactor { field, .. }
      | Self::QuantityIncomparable { field, .. } => Some(field),
      Self::BelowThreshold { .. } | Self::AnchorDrift { .. } | Self::MeasureChanged => None,
    }
  }

  /// What was measured: a similarity, or a quantity's factor. `None` where no number describes it.
  pub fn value(&self) -> Option<f64> {
    match self {
      Self::BelowThreshold { similarity, .. }
      | Self::FieldBelowFloor { similarity, .. }
      | Self::AnchorDrift { similarity, .. } => Some(f64::from(*similarity)),
      Self::QuantityBeyondFactor { factor, .. } => *factor,
      Self::QuantityIncomparable { .. } | Self::MeasureChanged => None,
    }
  }

  /// The configured limit the value was held to: a threshold, a floor, or a factor limit.
  pub fn limit(&self) -> Option<f64> {
    match self {
      Self::BelowThreshold { threshold, .. } | Self::AnchorDrift { threshold, .. } => Some(f64::from(*threshold)),
      Self::FieldBelowFloor { floor, .. } => Some(f64::from(*floor)),
      Self::QuantityBeyondFactor { limit, .. } => Some(*limit),
      Self::QuantityIncomparable { .. } | Self::MeasureChanged => None,
    }
  }
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
      Self::QuantityBeyondFactor {
        field,
        factor: Some(factor),
        limit,
      } => {
        write!(
          f,
          "quantity `{field}` changed by a factor of {factor:.2}, above its limit {limit:.2}"
        )
      }
      Self::QuantityBeyondFactor {
        field,
        factor: None,
        limit,
      } => {
        write!(
          f,
          "quantity `{field}` moved between zero and a value above zero, which no factor limit ({limit:.2}) allows"
        )
      }
      Self::QuantityIncomparable {
        field,
        previous_unit,
        unit,
      } => {
        write!(
          f,
          "quantity `{field}` went from `{previous_unit}` to `{unit}`, which cannot be compared"
        )
      }
      Self::MeasureChanged => {
        write!(f, "the measures the record is described by changed")
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
  /// One entry per quantity that either side has a value for, in schema order. Empty until
  /// [`Self::judge_quantities`] is called, and for a record without quantities.
  pub quantities: Vec<QuantityScore>,
  /// Whether the revision changed which measures the record is described by: a quantity moved to a unit of
  /// another dimension (a potency in `%` became one in `mg`), or one quantity went away as a different one
  /// arrived (a weight for a volume). Filling in a quantity that was simply missing does not count.
  pub measure_changed: bool,
}

impl Evaluation {
  /// Whether the revision passed every check.
  pub fn accepted(&self) -> bool {
    self.reasons.is_empty()
  }

  /// Compares the quantities of the two versions, records the result, and adds a reason for each configured limit
  /// that is exceeded: [`compare_quantities`] followed by [`judge_quantities`].
  ///
  /// `anchor` holds the first value the record had for each quantity (see [`quantity::first_values`]); pass an
  /// empty map when there is none. Quantities the schema does not define are ignored: the caller has validated
  /// the candidate, and a stored value whose quantity was since removed from the configuration has nothing to be
  /// judged by.
  ///
  /// This never touches the similarity scores. With no limit configured it adds no reason either, so calling it
  /// can only reject where someone has asked for that.
  pub fn judge_quantities(
    &mut self,
    schema: &Schema,
    policy: &Policy,
    baseline: &Quantities,
    anchor: &Quantities,
    candidate: &Quantities,
  ) {
    self.quantities = compare_quantities(schema, baseline, anchor, candidate);
    let (measure_changed, reasons) = judge_quantities(schema, policy, &mut self.quantities);
    self.measure_changed = measure_changed;
    self.reasons.extend(reasons);
  }

  /// Judges again, under `schema` and `policy`, what this evaluation already measured: the field similarities,
  /// the similarity to the anchor, and the quantity comparisons. Every reason is replaced.
  ///
  /// This is how a change of weights, floors, thresholds, or quantity limits is tried out on comparisons that
  /// were measured earlier, with no model loaded. It cannot account for a change of model or of the set of
  /// fields: those change the measurements themselves.
  pub fn rejudge(&mut self, schema: &Schema, policy: &Policy) {
    self.comparison.weigh(schema);
    self.reasons = judge(schema, policy, &self.comparison, self.anchor_similarity);
    let (measure_changed, reasons) = judge_quantities(schema, policy, &mut self.quantities);
    self.measure_changed = measure_changed;
    self.reasons.extend(reasons);
  }
}

/// Compares every quantity the schema defines across two versions of a record, in schema order.
///
/// Measuring only: no limit is consulted and no score is marked as beyond one (see [`judge_quantities`]).
/// Quantities the schema does not define are left out.
pub fn compare_quantities(
  schema: &Schema,
  baseline: &Quantities,
  anchor: &Quantities,
  candidate: &Quantities,
) -> Vec<QuantityScore> {
  schema
    .quantities()
    .iter()
    .filter_map(|spec| {
      let name = spec.name.as_str();
      quantity::compare(name, baseline.get(name), anchor.get(name), candidate.get(name))
    })
    .collect()
}

/// Holds quantity comparisons to the limits `schema` and `policy` configure.
///
/// Sets each score's `beyond_limit` (clearing whatever an earlier judgment left there) and returns whether the
/// record's measures changed, with a reason for every limit exceeded. A score for a quantity the schema does not
/// define is held to nothing.
pub fn judge_quantities(schema: &Schema, policy: &Policy, scores: &mut [QuantityScore]) -> (bool, Vec<Reason>) {
  let mut reasons = Vec::new();
  for score in scores.iter_mut() {
    score.beyond_limit = false;
    let Some(spec) = schema.quantity(&score.field) else {
      continue;
    };
    if let Some(limit) = spec.reject_above_factor
      && (score.zero_crossing || score.factor.is_some_and(|factor| factor > limit))
    {
      score.beyond_limit = true;
      reasons.push(Reason::QuantityBeyondFactor {
        field: score.field.clone(),
        factor: score.factor,
        limit,
      });
    }
    if spec.reject_incomparable && score.incomparable {
      score.beyond_limit = true;
      reasons.push(Reason::QuantityIncomparable {
        field: score.field.clone(),
        previous_unit: score.previous_unit.clone().unwrap_or_default(),
        unit: score.unit.clone().unwrap_or_default(),
      });
    }
  }

  let any = |wanted: &[QuantityChange]| scores.iter().any(|score| wanted.contains(&score.change));
  // Two different quantities are needed for the second case, and they necessarily are: one name has one change.
  let measure_changed = scores.iter().any(|score| score.incomparable)
    || (any(&[QuantityChange::Removed]) && any(&[QuantityChange::Added, QuantityChange::Restored]));
  if measure_changed && policy.reject_on_measure_change {
    reasons.push(Reason::MeasureChanged);
  }
  (measure_changed, reasons)
}

/// Holds a comparison to the thresholds and floors `schema` and `policy` configure, and returns why it fails
/// them; empty means it passes.
///
/// It reads the comparison as it stands. To judge under weights or floors other than the ones it was compared
/// with, call [`Comparison::weigh`] first, or use [`Evaluation::rejudge`], which does both.
pub fn judge(schema: &Schema, policy: &Policy, comparison: &Comparison, anchor_similarity: Option<f32>) -> Vec<Reason> {
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
  reasons
}

/// Judges `candidate` as a revision of `baseline`.
///
/// `anchor` is the composite of the record as first stored, if the caller tracks one (see [`extend_anchor`]). Its
/// overall similarity is always reported when supplied, and only enforced when the policy sets an
/// `anchor_threshold`. Independently of that setting, a field the baseline lacks is scored against it.
///
/// Quantities are not looked at here. A caller whose records have them follows this with
/// [`Evaluation::judge_quantities`].
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

  let reasons = judge(schema, policy, &comparison, anchor_similarity);

  Ok(Evaluation {
    comparison,
    anchor_similarity,
    reasons,
    quantities: Vec::new(),
    measure_changed: false,
  })
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;

  use super::*;
  use crate::quantity::Quantity;
  use crate::schema::testing::{product_schema, quantity as spec};
  use crate::schema::{FieldSpec, QuantitySpec};
  use crate::vector::normalize;

  const DIM: usize = 2;

  /// A unit vector at `degrees` in the plane, so the cosine between two of them is the cosine of the difference.
  /// "Below the threshold" means below it. A revision that scores exactly the threshold is accepted, for the
  /// overall similarity and for the similarity to the anchor alike: the documented rule, and the one a caller who
  /// sets a threshold from a measured score is relying on.
  #[test]
  fn a_score_exactly_at_a_threshold_is_accepted_and_one_step_below_it_is_not() {
    let schema = product_schema();
    let policy = Policy::new(0.9, Some(0.85)).unwrap();
    let scored = |similarity: f32| Comparison {
      similarity,
      fields: Vec::new(),
    };

    assert_eq!(judge(&schema, &policy, &scored(0.9), Some(0.85)), []);
    assert_eq!(
      judge(&schema, &policy, &scored(0.9_f32.next_down()), Some(0.85)),
      [Reason::BelowThreshold {
        similarity: 0.9_f32.next_down(),
        threshold: 0.9
      }]
    );
    assert_eq!(
      judge(&schema, &policy, &scored(0.9), Some(0.85_f32.next_down())),
      [Reason::AnchorDrift {
        similarity: 0.85_f32.next_down(),
        threshold: 0.85
      }]
    );
  }

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
  fn overriding_the_threshold_keeps_the_rest_of_the_policy() {
    let mut policy = Policy::new(0.8, Some(0.6)).unwrap();
    policy.reject_on_measure_change = true;
    let overridden = policy.with_threshold(0.95).unwrap();
    assert!((overridden.threshold - 0.95).abs() < f32::EPSILON);
    assert_eq!(overridden.anchor_threshold, Some(0.6));
    assert!(overridden.reject_on_measure_change);
    assert!(policy.with_threshold(7.0).is_err());
  }

  fn quantities(entries: &[(&str, Quantity)]) -> Quantities {
    entries
      .iter()
      .map(|(name, value)| ((*name).to_owned(), value.clone()))
      .collect()
  }

  /// An accepted, text-identical evaluation with the quantities judged: whatever it reports came from them.
  fn judged(
    schema: &Schema,
    policy: &Policy,
    before: &Quantities,
    anchor: &Quantities,
    after: &Quantities,
  ) -> Evaluation {
    let stored = record(schema, &[("name", 0.0, "n")]);
    let mut evaluation = evaluate(schema, policy, DIM, &stored, None, &stored.clone()).unwrap();
    assert!(evaluation.accepted());
    evaluation.judge_quantities(schema, policy, before, anchor, after);
    evaluation
  }

  fn measured_schema(specs: Vec<QuantitySpec>) -> Schema {
    product_schema().with_quantities(specs).unwrap()
  }

  fn grams(value: f64) -> Quantity {
    Quantity::single(value, "g")
  }

  #[test]
  fn a_quantity_is_reported_and_rejects_nothing_until_a_limit_is_set() {
    let schema = measured_schema(vec![spec("weight")]);
    let policy = Policy::new(0.8, None).unwrap();
    let before = quantities(&[("weight", grams(1.0))]);
    let after = quantities(&[("weight", grams(100.0))]);

    let evaluation = judged(&schema, &policy, &before, &Quantities::new(), &after);
    assert!(evaluation.accepted(), "{:?}", evaluation.reasons);
    let [weight] = evaluation.quantities.as_slice() else {
      panic!("one quantity expected: {:?}", evaluation.quantities)
    };
    assert_eq!(weight.change, QuantityChange::Modified);
    assert!((weight.factor.unwrap() - 100.0).abs() < 1e-9);
    assert!(!weight.beyond_limit && !evaluation.measure_changed);
    assert!(
      (evaluation.comparison.similarity - 1.0).abs() < f32::EPSILON,
      "a quantity never moves the similarity"
    );
  }

  #[test]
  fn a_factor_limit_rejects_what_exceeds_it_and_nothing_else() {
    let limited = QuantitySpec {
      reject_above_factor: Some(4.0),
      ..spec("weight")
    };
    let schema = measured_schema(vec![limited]);
    let policy = Policy::new(0.8, None).unwrap();
    let before = quantities(&[("weight", grams(3.5))]);
    let judge = |after: Quantity| {
      judged(
        &schema,
        &policy,
        &before,
        &Quantities::new(),
        &quantities(&[("weight", after)]),
      )
    };

    let eightfold = judge(grams(28.0));
    assert!(
      matches!(
        eightfold.reasons.as_slice(),
        [Reason::QuantityBeyondFactor { field, factor: Some(factor), limit }]
          if field == "weight" && (*factor - 8.0).abs() < 1e-9 && (*limit - 4.0).abs() < 1e-9
      ),
      "{:?}",
      eightfold.reasons
    );
    assert!(eightfold.quantities[0].beyond_limit);
    assert!(eightfold.reasons[0].to_string().contains("factor of 8.00"));

    assert!(judge(grams(14.0)).accepted(), "exactly at the limit is within it");
    assert!(judge(grams(7.0)).accepted());
    assert!(
      judge(Quantity::single(3500.0, "mg")).accepted(),
      "a unit rewrite is no change"
    );

    // Zero to something has no factor, and so is beyond every limit.
    let from_nothing = judged(
      &schema,
      &policy,
      &quantities(&[("weight", grams(0.0))]),
      &Quantities::new(),
      &quantities(&[("weight", grams(1.0))]),
    );
    assert!(
      matches!(
        from_nothing.reasons.as_slice(),
        [Reason::QuantityBeyondFactor { factor: None, .. }]
      ),
      "{:?}",
      from_nothing.reasons
    );
    assert!(from_nothing.reasons[0].to_string().contains("zero"));

    // A change of dimension has no factor either, but is a different check with its own switch.
    assert!(judge(Quantity::single(355.0, "ml")).accepted());
  }

  #[test]
  fn an_incomparable_quantity_rejects_only_when_its_schema_entry_says_so() {
    let policy = Policy::new(0.8, None).unwrap();
    let before = quantities(&[("thc", Quantity::single(22.0, "%"))]);
    let after = quantities(&[("thc", Quantity::single(22.0, "mg"))]);

    let lenient = judged(
      &measured_schema(vec![spec("thc")]),
      &policy,
      &before,
      &Quantities::new(),
      &after,
    );
    assert!(lenient.accepted());
    assert_eq!(lenient.quantities[0].change, QuantityChange::Incomparable);
    assert!(lenient.measure_changed, "reported either way");

    let strict_spec = QuantitySpec {
      reject_incomparable: true,
      ..spec("thc")
    };
    let strict = judged(
      &measured_schema(vec![strict_spec]),
      &policy,
      &before,
      &Quantities::new(),
      &after,
    );
    assert!(
      matches!(
        strict.reasons.as_slice(),
        [Reason::QuantityIncomparable { field, previous_unit, unit }]
          if field == "thc" && previous_unit == "%" && unit == "mg"
      ),
      "{:?}",
      strict.reasons
    );
    assert!(strict.quantities[0].beyond_limit);
    assert!(strict.reasons[0].to_string().contains("`%` to `mg`"));
  }

  #[test]
  fn a_restored_quantity_is_held_to_its_limits_against_the_original() {
    let limited = QuantitySpec {
      reject_above_factor: Some(4.0),
      reject_incomparable: true,
      ..spec("weight")
    };
    let schema = measured_schema(vec![limited]);
    let policy = Policy::new(0.8, None).unwrap();
    let anchor = quantities(&[("weight", grams(3.5))]);
    let refill = |after: Quantity| {
      judged(
        &schema,
        &policy,
        &Quantities::new(),
        &anchor,
        &quantities(&[("weight", after)]),
      )
    };

    let swapped = refill(grams(28.0));
    assert_eq!(swapped.quantities[0].change, QuantityChange::Restored);
    assert!(matches!(
      swapped.reasons.as_slice(),
      [Reason::QuantityBeyondFactor { .. }]
    ));

    let other_kind = refill(Quantity::single(355.0, "ml"));
    assert_eq!(other_kind.quantities[0].change, QuantityChange::Restored);
    assert!(matches!(
      other_kind.reasons.as_slice(),
      [Reason::QuantityIncomparable { .. }]
    ));
    assert!(other_kind.measure_changed);

    assert!(refill(grams(3.5)).accepted(), "putting the original back is no change");
  }

  #[test]
  fn the_measure_changes_when_a_quantity_is_incomparable_or_one_is_traded_for_another() {
    let schema = measured_schema(vec![spec("weight"), spec("volume"), spec("thc")]);
    let policy = Policy::new(0.8, None).unwrap();
    let none = Quantities::new();
    let weight = ("weight", grams(3.5));
    let volume = ("volume", Quantity::single(355.0, "ml"));
    let thc = ("thc", Quantity::single(22.0, "%"));
    let (only_weight, only_volume, only_thc) = (
      quantities(std::slice::from_ref(&weight)),
      quantities(std::slice::from_ref(&volume)),
      quantities(std::slice::from_ref(&thc)),
    );
    let flower = quantities(&[weight, thc.clone()]);
    let beverage = quantities(&[volume, thc]);
    let changed = |before: &Quantities, anchor: &Quantities, after: &Quantities| {
      judged(&schema, &policy, before, anchor, after).measure_changed
    };

    // A weight disappears as a volume appears: flower becoming a beverage.
    assert!(changed(&flower, &none, &beverage));
    // The same trade where the volume is one the record had long ago.
    assert!(changed(&only_weight, &only_volume, &only_volume));
    // A unit of another dimension.
    assert!(changed(
      &only_thc,
      &none,
      &quantities(&[("thc", Quantity::single(22.0, "mg"))])
    ));

    // Filling in a quantity that was simply missing is ordinary maintenance.
    assert!(!changed(&only_weight, &none, &flower));
    // So is dropping one, or changing one, however much.
    assert!(!changed(&flower, &none, &only_weight));
    assert!(!changed(&only_weight, &none, &quantities(&[("weight", grams(28.0))])));
    assert!(!changed(&none, &none, &none));
  }

  #[test]
  fn a_change_of_measure_rejects_only_when_the_policy_says_so() {
    let schema = measured_schema(vec![spec("weight"), spec("volume")]);
    let before = quantities(&[("weight", grams(3.5))]);
    let after = quantities(&[("volume", Quantity::single(355.0, "ml"))]);

    let lenient = Policy::new(0.8, None).unwrap();
    let reported = judged(&schema, &lenient, &before, &Quantities::new(), &after);
    assert!(reported.measure_changed && reported.accepted());

    let mut strict = lenient;
    strict.reject_on_measure_change = true;
    let rejected = judged(&schema, &strict, &before, &Quantities::new(), &after);
    assert_eq!(rejected.reasons, [Reason::MeasureChanged]);
    assert!(rejected.reasons[0].to_string().contains("measures"));
    assert!(
      rejected.quantities.iter().all(|score| !score.beyond_limit),
      "no single quantity is at fault"
    );
  }

  #[test]
  fn quantities_are_reported_in_schema_order_and_unconfigured_ones_are_ignored() {
    let schema = measured_schema(vec![spec("weight"), spec("thc")]);
    let policy = Policy::new(0.8, None).unwrap();
    let before = quantities(&[
      ("weight", grams(3.5)),
      ("thc", Quantity::single(22.0, "%")),
      ("retired", Quantity::single(1.0, "each")),
    ]);
    let evaluation = judged(&schema, &policy, &before, &Quantities::new(), &before.clone());
    let names: Vec<&str> = evaluation.quantities.iter().map(|score| score.field.as_str()).collect();
    assert_eq!(names, ["thc", "weight"]);
    assert!(
      evaluation
        .quantities
        .iter()
        .all(|score| score.change == QuantityChange::Unchanged)
    );
  }

  #[test]
  fn every_reason_has_a_stable_code_and_says_what_it_measured() {
    let reasons = [
      Reason::BelowThreshold {
        similarity: 0.5,
        threshold: 0.9,
      },
      Reason::FieldBelowFloor {
        field: "name".to_owned(),
        similarity: 0.25,
        floor: 0.75,
      },
      Reason::AnchorDrift {
        similarity: 0.5,
        threshold: 0.75,
      },
      Reason::QuantityBeyondFactor {
        field: "weight".to_owned(),
        factor: Some(8.0),
        limit: 4.0,
      },
      Reason::QuantityBeyondFactor {
        field: "weight".to_owned(),
        factor: None,
        limit: 4.0,
      },
      Reason::QuantityIncomparable {
        field: "thc".to_owned(),
        previous_unit: "%".to_owned(),
        unit: "mg".to_owned(),
      },
      Reason::MeasureChanged,
    ];
    let described: Vec<_> = reasons
      .iter()
      .map(|reason| (reason.code(), reason.subject(), reason.value(), reason.limit()))
      .collect();
    // Pinned on purpose: reports and logs carry these strings, and tools match on them.
    assert_eq!(
      described,
      [
        ("below_threshold", None, Some(0.5), Some(f64::from(0.9_f32))),
        ("field_below_floor", Some("name"), Some(0.25), Some(0.75)),
        ("anchor_drift", None, Some(0.5), Some(0.75)),
        ("quantity_beyond_factor", Some("weight"), Some(8.0), Some(4.0)),
        ("quantity_beyond_factor", Some("weight"), None, Some(4.0)),
        ("quantity_incomparable", Some("thc"), None, None),
        ("measure_changed", None, None, None),
      ]
    );
  }

  #[test]
  fn judging_again_under_the_same_configuration_changes_nothing() {
    let limited = QuantitySpec {
      reject_above_factor: Some(4.0),
      reject_incomparable: true,
      ..spec("weight")
    };
    let floored = |name: &str, weight: f32, floor: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below: floor,
    };
    let schema = Schema::new(vec![floored("name", 4.0, Some(0.8)), floored("category", 2.0, None)])
      .unwrap()
      .with_quantities(vec![limited, spec("thc")])
      .unwrap();
    let mut policy = Policy::new(0.9, Some(0.95)).unwrap();
    policy.reject_on_measure_change = true;

    let original = record(&schema, &[("name", 0.0, "n0"), ("category", 0.0, "c")]);
    let old = record(&schema, &[("name", 20.0, "n1"), ("category", 0.0, "c")]);
    let new = record(&schema, &[("name", 80.0, "n2"), ("category", 30.0, "c2")]);
    let mut evaluation = evaluate(&schema, &policy, DIM, &old, Some(&original.composite), &new).unwrap();
    evaluation.judge_quantities(
      &schema,
      &policy,
      &quantities(&[("weight", grams(3.5)), ("thc", Quantity::single(22.0, "%"))]),
      &Quantities::new(),
      &quantities(&[("weight", grams(28.0)), ("thc", Quantity::single(22.0, "mg"))]),
    );
    assert_eq!(evaluation.reasons.len(), 5, "{:?}", evaluation.reasons);

    let mut again = evaluation.clone();
    again.rejudge(&schema, &policy);
    assert_eq!(again, evaluation);
    again.rejudge(&schema, &policy);
    assert_eq!(again, evaluation);
  }

  #[test]
  fn judging_again_under_other_weights_and_limits_equals_evaluating_under_them() {
    let fields = |name_weight: f32, floor: Option<f32>| {
      vec![
        FieldSpec {
          name: "name".to_owned(),
          weight: name_weight,
          required: false,
          reject_below: floor,
        },
        FieldSpec {
          name: "description".to_owned(),
          weight: 1.0,
          required: false,
          reject_below: None,
        },
      ]
    };
    let weight = |limit: Option<f64>| QuantitySpec {
      reject_above_factor: limit,
      ..spec("weight")
    };
    let first = Schema::new(fields(1.0, None))
      .unwrap()
      .with_quantities(vec![weight(None)])
      .unwrap();
    let second = Schema::new(fields(9.0, Some(0.6)))
      .unwrap()
      .with_quantities(vec![weight(Some(4.0))])
      .unwrap();
    let lenient = Policy::new(0.2, None).unwrap();
    let strict = Policy::new(0.7, None).unwrap();

    let old = record(&first, &[("name", 0.0, "n1"), ("description", 0.0, "d")]);
    let new = record(&first, &[("name", 60.0, "n2"), ("description", 10.0, "d2")]);
    let before = quantities(&[("weight", grams(3.5))]);
    let after = quantities(&[("weight", grams(28.0))]);
    let evaluated = |schema: &Schema, policy: &Policy| {
      let mut evaluation = evaluate(schema, policy, DIM, &old, None, &new).unwrap();
      evaluation.judge_quantities(schema, policy, &before, &Quantities::new(), &after);
      evaluation
    };

    let mut rejudged = evaluated(&first, &lenient);
    assert!(rejudged.accepted(), "{:?}", rejudged.reasons);
    rejudged.rejudge(&second, &strict);

    let fresh = evaluated(&second, &strict);
    assert_eq!(rejudged, fresh, "nothing was embedded to get here");
    let codes: Vec<&str> = fresh.reasons.iter().map(Reason::code).collect();
    assert_eq!(
      codes,
      ["below_threshold", "field_below_floor", "quantity_beyond_factor"]
    );
  }

  #[test]
  fn a_score_for_something_the_schema_no_longer_defines_is_held_to_nothing() {
    let schema = measured_schema(vec![QuantitySpec {
      reject_above_factor: Some(2.0),
      ..spec("weight")
    }]);
    let policy = Policy::new(0.8, None).unwrap();
    let mut evaluation = judged(
      &schema,
      &policy,
      &quantities(&[("weight", grams(1.0))]),
      &Quantities::new(),
      &quantities(&[("weight", grams(10.0))]),
    );
    assert!(evaluation.quantities[0].beyond_limit);

    // The same measurements under a schema that has dropped the quantity and the `name` field's siblings.
    let bare = Schema::new(vec![FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: false,
      reject_below: None,
    }])
    .unwrap();
    evaluation.rejudge(&bare, &policy);
    assert!(evaluation.accepted(), "{:?}", evaluation.reasons);
    assert!(!evaluation.quantities[0].beyond_limit);
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
