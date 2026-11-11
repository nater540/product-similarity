//! Which fields a record has, how much each one matters, and where each one lives inside a stored vector.
//!
//! A record is embedded field by field, and the per-field unit vectors are packed into one **composite** vector:
//!
//! ```text
//! composite = normalize( sqrt(w_1) * u_1  ++  sqrt(w_2) * u_2  ++  ...  ++  sqrt(w_n) * u_n )
//! ```
//!
//! with a block of zeros where a record has no value for a field. Two properties make this the stored form:
//!
//! - The cosine of two composites is the **weighted mean of the per-field cosines** (exactly, when both records have
//!   the same fields). So the score a vector database computes when searching composites is the same number the
//!   accept/reject policy uses, with no second index and no query-time fusion.
//! - Each block is a scaled copy of its field's vector and cosine ignores scale, so the per-field similarities can
//!   always be recovered from two composites, even ones written under different weights. Re-weighting a schema
//!   therefore never invalidates stored comparisons; it only leaves search rankings stale until records are
//!   rewritten.
//!
//! What *does* invalidate stored composites is a change to the block layout: the set of field names, or the width of
//! a block. [`Schema::layout_id`] names the layout, and the store refuses to compare across layouts.

use crate::error::{Error, Result};
use crate::vector::{norm, normalize};

/// A block whose length is below this is treated as "this record has no value for the field".
///
/// A real block has length `sqrt(w / total)`, which for any sane weighting is orders of magnitude above this, while
/// an absent block is exactly zero before the `f32` round trip through storage and still far below this after it.
const ABSENT_BLOCK_NORM: f64 = 1e-6;

/// One field of a record.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSpec {
  /// The key callers use for this field (`name`, `category`, ...).
  pub name: String,
  /// Relative importance in the overall score. Only ratios between fields matter.
  pub weight: f32,
  /// A record without a non-blank value for a required field is refused outright.
  pub required: bool,
  /// If set, a revision whose similarity **for this field alone** falls below it is rejected regardless of the
  /// overall score. This is what catches a record whose name is replaced while a long boilerplate description is
  /// left alone to prop the weighted average up.
  pub reject_below: Option<f32>,
}

/// The ordered set of fields that make up a record.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
  fields: Vec<FieldSpec>,
}

impl Schema {
  /// Validates and orders a set of fields.
  ///
  /// Fields are sorted by name, so the block layout depends on *which* fields exist and never on the order a
  /// configuration file happened to list them in.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Schema`] if there are no fields, a name is empty, duplicated, or contains anything other than
  /// ASCII letters, digits, and underscores, a weight is not a positive finite number, or a floor lies outside
  /// `[-1, 1]`.
  pub fn new(mut fields: Vec<FieldSpec>) -> Result<Self> {
    if fields.is_empty() {
      return Err(Error::Schema("at least one field is required".to_owned()));
    }
    fields.sort_by(|a, b| a.name.cmp(&b.name));
    for pair in fields.windows(2) {
      if let [a, b] = pair
        && a.name == b.name
      {
        return Err(Error::Schema(format!("field `{}` is defined twice", a.name)));
      }
    }
    for field in &fields {
      let valid_name = !field.name.is_empty() && field.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
      if !valid_name {
        return Err(Error::Schema(format!(
          "field name `{}` must be non-empty ASCII letters, digits, and underscores",
          field.name
        )));
      }
      if !(field.weight.is_finite() && field.weight > 0.0) {
        return Err(Error::Schema(format!(
          "field `{}` has weight {}, which is not a positive number",
          field.name, field.weight
        )));
      }
      if let Some(floor) = field.reject_below
        && !(-1.0..=1.0).contains(&floor)
      {
        return Err(Error::Schema(format!(
          "field `{}` has `reject_below` {floor}, outside the cosine range [-1, 1]",
          field.name
        )));
      }
    }
    Ok(Self { fields })
  }

  /// The fields, sorted by name. Block `i` of a composite belongs to field `i` of this slice.
  pub fn fields(&self) -> &[FieldSpec] {
    &self.fields
  }

  /// Looks a field up by name.
  pub fn field(&self, name: &str) -> Option<&FieldSpec> {
    self.fields.iter().find(|field| field.name == name)
  }

  /// The length of a composite vector whose blocks are `dimension` wide.
  pub fn composite_dimension(&self, dimension: usize) -> usize {
    self.fields.len().saturating_mul(dimension)
  }

  /// Names the block layout: which model produced the blocks and which field each block belongs to.
  ///
  /// Weights are deliberately absent, see the module documentation.
  pub fn layout_id(&self, model_fingerprint: &str) -> String {
    let names: Vec<&str> = self.fields.iter().map(|field| field.name.as_str()).collect();
    format!("{model_fingerprint}|{}", names.join(","))
  }

  /// Packs per-field unit vectors into a composite.
  ///
  /// `blocks` yields one entry per schema field, in schema order: the field's vector, or `None` if the record has
  /// no value for it.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Incompatible`] if the number of entries does not match the schema, a vector is not
  /// `dimension` long, or every entry is `None` (a record with no content has no direction).
  pub fn compose<'a>(&self, dimension: usize, blocks: impl IntoIterator<Item = Option<&'a [f32]>>) -> Result<Vec<f32>> {
    let mut composite = Vec::with_capacity(self.composite_dimension(dimension));
    let mut supplied = 0_usize;
    let mut fields = self.fields.iter();
    for block in blocks {
      let Some(field) = fields.next() else {
        return Err(Error::Incompatible(format!(
          "more blocks than the schema's {} fields",
          self.fields.len()
        )));
      };
      supplied = supplied.saturating_add(1);
      match block {
        None => composite.extend(std::iter::repeat_n(0.0_f32, dimension)),
        Some(vector) if vector.len() == dimension => {
          let scale = field.weight.sqrt();
          composite.extend(vector.iter().map(|value| value * scale));
        }
        Some(vector) => {
          return Err(Error::Incompatible(format!(
            "field `{}` has a vector of length {}, expected {dimension}",
            field.name,
            vector.len()
          )));
        }
      }
    }
    if supplied != self.fields.len() {
      return Err(Error::Incompatible(format!(
        "{supplied} blocks for a schema of {} fields",
        self.fields.len()
      )));
    }
    if !normalize(&mut composite) {
      return Err(Error::Incompatible(
        "a record with no field values cannot be embedded".to_owned(),
      ));
    }
    Ok(composite)
  }

  /// Splits a composite back into its per-field blocks, in schema order.
  ///
  /// A block is `None` where the record had no value for the field. Present blocks are returned as stored (scaled),
  /// which is all cosine needs.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Incompatible`] if the composite is not exactly `fields * dimension` long.
  pub fn blocks<'a>(&self, dimension: usize, composite: &'a [f32]) -> Result<Vec<Option<&'a [f32]>>> {
    let expected = self.composite_dimension(dimension);
    if dimension == 0 || composite.len() != expected {
      return Err(Error::Incompatible(format!(
        "composite has {} components, but {} fields of width {dimension} need {expected}",
        composite.len(),
        self.fields.len()
      )));
    }
    Ok(
      composite
        .chunks_exact(dimension)
        .map(|block| (norm(block) > ABSENT_BLOCK_NORM).then_some(block))
        .collect(),
    )
  }
}

#[cfg(test)]
pub(crate) mod testing {
  use super::{FieldSpec, Schema};

  pub(crate) fn field(name: &str, weight: f32) -> FieldSpec {
    FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below: None,
    }
  }

  /// `category` (2), `description` (1), `name` (4), in that (sorted) order.
  pub(crate) fn product_schema() -> Schema {
    Schema::new(vec![
      field("name", 4.0),
      field("description", 1.0),
      field("category", 2.0),
    ])
    .unwrap()
  }
}

#[cfg(test)]
mod tests {
  use super::testing::{field, product_schema};
  use super::*;
  use crate::vector::cosine;

  fn unit(components: [f32; 3]) -> Vec<f32> {
    let mut vector = components.to_vec();
    assert!(normalize(&mut vector));
    vector
  }

  #[test]
  fn fields_are_sorted_by_name_whatever_order_they_were_given_in() {
    let schema = product_schema();
    let names: Vec<&str> = schema.fields().iter().map(|field| field.name.as_str()).collect();
    assert_eq!(names, ["category", "description", "name"]);
  }

  #[test]
  fn invalid_schemas_are_refused() {
    assert!(Schema::new(vec![]).is_err(), "empty");
    assert!(
      Schema::new(vec![field("name", 1.0), field("name", 2.0)]).is_err(),
      "duplicate"
    );
    assert!(Schema::new(vec![field("", 1.0)]).is_err(), "blank name");
    assert!(Schema::new(vec![field("a,b", 1.0)]).is_err(), "separator in name");
    assert!(Schema::new(vec![field("name", 0.0)]).is_err(), "zero weight");
    assert!(Schema::new(vec![field("name", -1.0)]).is_err(), "negative weight");
    assert!(Schema::new(vec![field("name", f32::NAN)]).is_err(), "NaN weight");
    let mut floored = field("name", 1.0);
    floored.reject_below = Some(1.5);
    assert!(Schema::new(vec![floored]).is_err(), "floor outside [-1, 1]");
  }

  #[test]
  fn the_cosine_of_two_composites_is_the_weighted_mean_of_the_field_cosines() {
    let schema = product_schema();
    let (cat_a, cat_b) = (unit([1.0, 0.0, 0.0]), unit([1.0, 1.0, 0.0]));
    let (desc_a, desc_b) = (unit([0.0, 1.0, 0.0]), unit([0.0, 1.0, 0.0]));
    let (name_a, name_b) = (unit([0.0, 0.0, 1.0]), unit([1.0, 0.0, 0.2]));

    let a = schema
      .compose(3, [Some(&cat_a[..]), Some(&desc_a[..]), Some(&name_a[..])])
      .unwrap();
    let b = schema
      .compose(3, [Some(&cat_b[..]), Some(&desc_b[..]), Some(&name_b[..])])
      .unwrap();

    let expected = (2.0 * cosine(&cat_a, &cat_b).unwrap()
      + 1.0 * cosine(&desc_a, &desc_b).unwrap()
      + 4.0 * cosine(&name_a, &name_b).unwrap())
      / 7.0;
    assert!((cosine(&a, &b).unwrap() - expected).abs() < 1e-6);
  }

  #[test]
  fn blocks_recover_each_field_direction_and_mark_absent_fields() {
    let schema = product_schema();
    let category = unit([1.0, 2.0, 3.0]);
    let name = unit([-1.0, 0.5, 0.0]);
    let composite = schema.compose(3, [Some(&category[..]), None, Some(&name[..])]).unwrap();
    assert_eq!(composite.len(), 9);

    let blocks = schema.blocks(3, &composite).unwrap();
    assert!((cosine(blocks[0].unwrap(), &category).unwrap() - 1.0).abs() < 1e-6);
    assert!(blocks[1].is_none());
    assert!((cosine(blocks[2].unwrap(), &name).unwrap() - 1.0).abs() < 1e-6);
  }

  #[test]
  fn field_cosines_survive_a_change_of_weights() {
    let old = product_schema();
    let new = Schema::new(vec![
      field("name", 1.0),
      field("description", 9.0),
      field("category", 0.5),
    ])
    .unwrap();
    let (cat, desc) = (unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0]));
    let (name_a, name_b) = (unit([0.0, 0.0, 1.0]), unit([0.6, 0.0, 0.8]));

    let written_under_old = old
      .compose(3, [Some(&cat[..]), Some(&desc[..]), Some(&name_a[..])])
      .unwrap();
    let written_under_new = new
      .compose(3, [Some(&cat[..]), Some(&desc[..]), Some(&name_b[..])])
      .unwrap();

    let a = new.blocks(3, &written_under_old).unwrap();
    let b = new.blocks(3, &written_under_new).unwrap();
    assert!((cosine(a[2].unwrap(), b[2].unwrap()).unwrap() - 0.8).abs() < 1e-6);
    assert_eq!(
      old.layout_id("m"),
      new.layout_id("m"),
      "weights are not part of the layout"
    );
  }

  #[test]
  fn the_layout_changes_with_the_model_or_the_field_set() {
    let schema = product_schema();
    let wider = Schema::new(vec![field("name", 4.0), field("brand", 1.0)]).unwrap();
    assert_ne!(schema.layout_id("model-a"), schema.layout_id("model-b"));
    assert_ne!(schema.layout_id("model-a"), wider.layout_id("model-a"));
  }

  #[test]
  fn malformed_inputs_to_compose_and_blocks_are_errors() {
    let schema = product_schema();
    let vector = unit([1.0, 0.0, 0.0]);
    assert!(schema.compose(3, [None, None, None]).is_err(), "no content");
    assert!(schema.compose(3, [Some(&vector[..]), None]).is_err(), "too few blocks");
    assert!(
      schema.compose(3, [None, None, None, Some(&vector[..])]).is_err(),
      "too many blocks"
    );
    assert!(
      schema.compose(4, [Some(&vector[..]), None, None]).is_err(),
      "wrong block width"
    );
    assert!(schema.blocks(3, &[0.0; 8]).is_err(), "wrong composite length");
  }
}
