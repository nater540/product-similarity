//! Matching a listing from elsewhere to the stored record it describes.
//!
//! A revision is compared with one stored record, the one that shares its id. A listing (a retailer's menu item,
//! say) names no stored record: the question is *which* one it describes, if any. A search around the listing's
//! composite finds candidates, and this module scores them.
//!
//! **The search score is skewed and the score here is not.** The cosine of two composites is the weighted mean of
//! the per-field similarities over the fields both sides have, multiplied by a factor that depends only on which
//! fields each side filled in: `W(both) / sqrt(W(listing) * W(record))`, with `W` the sum of the weights of a set
//! of fields. A listing with only a name is therefore ranked below its true strength against a record with every
//! field filled, and the more fields the record has, the further below. [`rank`] scores each candidate with
//! [`score::compare`], which reports a field only one side has and leaves it out of the mean, so the factor is
//! gone. The search only has to bring the right record into the candidates; how many to fetch for that is a cost
//! the caller chooses.
//!
//! **A match is a reading, not a revision.** Field floors, the anchor, and quantity limits are rules about what an
//! edit may do to a record, and none applies. A candidate is compared with the stored record as the baseline and
//! the listing as the candidate, without an anchor, and no field is ever reported below a floor. Quantities are
//! compared and reported the same way round and never enter the score or the decision (D16).

use crate::error::Result;
use crate::quantity::{Quantities, QuantityScore};
use crate::record::RecordVectors;
use crate::schema::Schema;
use crate::score::{self, Comparison};

/// One stored record as a search returned it, before it is scored.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
  /// The record's id.
  pub id: &'a str,
  /// The record's owner, if it has one.
  pub tenant: Option<&'a str>,
  /// The composite cosine the search ranked it by.
  pub search_similarity: f32,
  /// The record's current vectors.
  pub vectors: &'a RecordVectors,
  /// The record's current quantities.
  pub quantities: &'a Quantities,
}

/// A candidate, scored exactly against the listing.
#[derive(Debug, Clone, PartialEq)]
pub struct Ranked {
  /// The record's id.
  pub id: String,
  /// The record's owner, if it has one.
  pub tenant: Option<String>,
  /// The composite cosine the search ranked it by.
  pub search_similarity: f32,
  /// The comparison with the stored record as the baseline and the listing as the candidate, weighed under the
  /// schema. [`Comparison::similarity`] is the match score: the weighted mean over the fields both sides have. A
  /// field only the listing has is `Added`, one only the record has is `Removed`; neither is scored, and no field
  /// is ever below a floor.
  pub comparison: Comparison,
  /// The quantities, compared the same way round. Reported, and consulted by nothing.
  pub quantities: Vec<QuantityScore>,
}

/// What was decided about a ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
  /// There is no threshold to hold the best candidate to: the candidates are ranked and nothing is claimed.
  NotJudged,
  /// The best candidate reaches the threshold.
  Matched,
  /// No candidate reaches the threshold, or there is none.
  NoMatch,
}

/// Scores every candidate against the listing, puts the best first, and keeps `limit`.
///
/// The order is by match score, highest first. Equal scores are ordered by id, so that the result does not
/// depend on the order the candidates came in.
///
/// # Errors
///
/// Returns [`crate::Error::Incompatible`] if a composite does not have this schema's layout.
pub fn rank<'a>(
  schema: &Schema,
  dimension: usize,
  listing: &RecordVectors,
  listing_quantities: &Quantities,
  candidates: impl IntoIterator<Item = Candidate<'a>>,
  limit: usize,
) -> Result<Vec<Ranked>> {
  let nothing = Quantities::new();
  let mut ranked = Vec::new();
  for candidate in candidates {
    let mut comparison = score::compare(schema, dimension, candidate.vectors, None, listing)?;
    // A floor is a rule about what an edit may do to a field. A listing is not an edit.
    for field in &mut comparison.fields {
      field.below_floor = false;
    }
    ranked.push(Ranked {
      id: candidate.id.to_owned(),
      tenant: candidate.tenant.map(str::to_owned),
      search_similarity: candidate.search_similarity,
      comparison,
      quantities: score::compare_quantities(schema, candidate.quantities, &nothing, listing_quantities),
    });
  }
  ranked.sort_by(|a, b| {
    b.comparison
      .similarity
      .total_cmp(&a.comparison.similarity)
      .then_with(|| a.id.cmp(&b.id))
  });
  ranked.truncate(limit);
  Ok(ranked)
}

/// Whether the best of `ranked` is a match: [`Decision::NotJudged`] without a threshold, and otherwise whether
/// the first candidate's match score is at or above it. A score exactly at the threshold matches, as a revision
/// exactly at its threshold is accepted.
pub fn decide(threshold: Option<f32>, ranked: &[Ranked]) -> Decision {
  let Some(threshold) = threshold else {
    return Decision::NotJudged;
  };
  match ranked.first() {
    Some(best) if best.comparison.similarity >= threshold => Decision::Matched,
    _ => Decision::NoMatch,
  }
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;

  use super::*;
  use crate::quantity::{Quantity, QuantityChange};
  use crate::schema::{FieldSpec, QuantitySpec};
  use crate::score::FieldChange;
  use crate::vector::{cosine, normalize};

  const DIM: usize = 2;

  /// A name that counts four times a description and twice a category, with a floor that a revision would be
  /// held to and a quantity with a limit.
  fn schema() -> Schema {
    let field = |name: &str, weight: f32, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below,
    };
    Schema::new(vec![
      field("name", 4.0, Some(0.99)),
      field("description", 1.0, None),
      field("category", 2.0, None),
    ])
    .unwrap()
    .with_quantities(vec![QuantitySpec {
      name: "weight".to_owned(),
      reject_above_factor: Some(1.5),
      reject_incomparable: true,
    }])
    .unwrap()
  }

  /// A unit vector at `degrees` in the plane: the cosine of two of them is the cosine of the angle between.
  fn at(degrees: f32) -> Vec<f32> {
    let radians = degrees.to_radians();
    let mut vector = vec![radians.cos(), radians.sin()];
    assert!(normalize(&mut vector));
    vector
  }

  /// Stored vectors from `(field, angle, stand-in for the text's hash)`.
  fn record(fields: &[(&str, f32, &str)]) -> RecordVectors {
    let schema = schema();
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
    let hashes: BTreeMap<String, String> = fields
      .iter()
      .map(|(name, _, text)| ((*name).to_owned(), (*text).to_owned()))
      .collect();
    RecordVectors {
      composite: schema.compose(DIM, units.iter().map(Option::as_deref)).unwrap(),
      hashes,
    }
  }

  fn candidate<'a>(id: &'a str, vectors: &'a RecordVectors, quantities: &'a Quantities) -> Candidate<'a> {
    Candidate {
      id,
      tenant: None,
      search_similarity: 0.0,
      vectors,
      quantities,
    }
  }

  fn ranked(listing: &RecordVectors, products: &[(&str, &RecordVectors)], limit: usize) -> Vec<Ranked> {
    let none = Quantities::new();
    rank(
      &schema(),
      DIM,
      listing,
      &none,
      products.iter().map(|(id, vectors)| candidate(id, vectors, &none)),
      limit,
    )
    .unwrap()
  }

  /// The case the module exists for. A menu item has a name and nothing else; the product it is a listing of has
  /// every field. The search would score the pair at the name's similarity times `4 / sqrt(4 * 7)`; the match
  /// score is the name's similarity.
  #[test]
  fn a_name_only_listing_is_scored_on_its_name_against_a_fully_filled_product() {
    let listing = record(&[("name", 10.0, "wyld blueberry 10pk")]);
    let product = record(&[
      ("name", 0.0, "blueberry gummies"),
      ("description", 80.0, "ten vegan gummies"),
      ("category", 120.0, "edibles"),
    ]);
    let result = ranked(&listing, &[("p1", &product)], 5);
    let name = 10.0_f32.to_radians().cos();
    assert!((result[0].comparison.similarity - name).abs() < 1e-6);

    let search = cosine(&listing.composite, &product.composite).unwrap();
    assert!((search - name * 4.0 / 28.0_f32.sqrt()).abs() < 1e-6, "{search}");
    assert!(
      search < 0.76 && name > 0.98,
      "the skew is not small: {search} against {name}"
    );
  }

  #[test]
  fn a_field_only_one_side_has_is_reported_and_left_out_of_the_score() {
    let listing = record(&[("name", 0.0, "a"), ("category", 60.0, "listing's own")]);
    let product = record(&[("name", 0.0, "b"), ("description", 90.0, "c")]);
    let result = ranked(&listing, &[("p1", &product)], 5);
    let fields: Vec<_> = result[0]
      .comparison
      .fields
      .iter()
      .map(|field| (field.field.as_str(), field.change, field.similarity.is_some()))
      .collect();
    assert_eq!(
      fields,
      [
        ("category", FieldChange::Added, false),
        ("description", FieldChange::Removed, false),
        ("name", FieldChange::Modified, true),
      ],
      "read with the stored record as the baseline: `Removed` is a field only it has"
    );
    assert!(
      (result[0].comparison.similarity - 1.0).abs() < 1e-6,
      "the name alone, which agrees"
    );
  }

  #[test]
  fn identical_text_is_unchanged_at_exactly_one() {
    let listing = record(&[("name", 0.0, "same text")]);
    let product = record(&[("name", 0.0, "same text"), ("category", 30.0, "edibles")]);
    let result = ranked(&listing, &[("p1", &product)], 5);
    let fields = &result[0].comparison.fields;
    let name = fields.iter().find(|field| field.field == "name").unwrap();
    assert_eq!((name.change, name.similarity), (FieldChange::Unchanged, Some(1.0)));
    assert_eq!(result[0].comparison.similarity, 1.0);
  }

  /// Why the candidates are scored again: the wrong product, with only a name, beats the right one in the search
  /// because the right one has more fields filled in.
  #[test]
  fn candidates_are_ordered_by_match_score_not_by_search_score() {
    let listing = record(&[("name", 0.0, "listing")]);
    let right = record(&[
      ("name", 5.0, "right"),
      ("description", 90.0, "d"),
      ("category", 90.0, "c"),
    ]);
    let wrong = record(&[("name", 30.0, "wrong")]);
    let search = |product: &RecordVectors| cosine(&listing.composite, &product.composite).unwrap();
    assert!(search(&wrong) > search(&right), "the search prefers the wrong one");

    let none = Quantities::new();
    let candidates = [
      Candidate {
        search_similarity: search(&wrong),
        tenant: Some("brand-2"),
        ..candidate("wrong", &wrong, &none)
      },
      Candidate {
        search_similarity: search(&right),
        tenant: Some("brand-1"),
        ..candidate("right", &right, &none)
      },
    ];
    let result = rank(&schema(), DIM, &listing, &none, candidates, 5).unwrap();
    let order: Vec<_> = result
      .iter()
      .map(|ranked| (ranked.id.as_str(), ranked.tenant.as_deref()))
      .collect();
    assert_eq!(order, [("right", Some("brand-1")), ("wrong", Some("brand-2"))]);
    assert_eq!(
      result[0].search_similarity,
      search(&right),
      "what the search said is kept beside it"
    );
    assert!(result[0].comparison.similarity > result[1].comparison.similarity);
  }

  #[test]
  fn equal_scores_are_ordered_by_id_and_the_limit_keeps_the_best() {
    let listing = record(&[("name", 0.0, "listing")]);
    let near = record(&[("name", 10.0, "near")]);
    let far = record(&[("name", 50.0, "far")]);
    let result = ranked(&listing, &[("z", &near), ("far", &far), ("a", &near), ("m", &near)], 10);
    let ids: Vec<&str> = result.iter().map(|ranked| ranked.id.as_str()).collect();
    assert_eq!(ids, ["a", "m", "z", "far"]);
    let kept = ranked(&listing, &[("far", &far), ("z", &near), ("a", &near)], 2);
    let ids: Vec<&str> = kept.iter().map(|ranked| ranked.id.as_str()).collect();
    assert_eq!(ids, ["a", "z"]);
    assert!(ranked(&listing, &[("a", &near)], 0).is_empty());
  }

  /// A floor is a rule for revisions. The schema here has one on the name that this listing would be far below.
  #[test]
  fn a_match_never_reports_a_field_below_a_floor() {
    let listing = record(&[("name", 60.0, "listing")]);
    let product = record(&[("name", 0.0, "product")]);
    let result = ranked(&listing, &[("p1", &product)], 5);
    assert!(result[0].comparison.similarity < 0.99);
    assert!(result[0].comparison.fields.iter().all(|field| !field.below_floor));
  }

  #[test]
  fn quantities_are_reported_per_candidate_and_move_no_score() {
    let listing = record(&[("name", 10.0, "listing")]);
    let product = record(&[("name", 0.0, "product")]);
    let grams = |value: f64| -> Quantities {
      [("weight".to_owned(), Quantity::single(value, "g"))]
        .into_iter()
        .collect()
    };
    let (one, ten, none) = (grams(1.0), grams(10.0), Quantities::new());
    let score = |listing_quantities: &Quantities, product_quantities: &Quantities| {
      rank(
        &schema(),
        DIM,
        &listing,
        listing_quantities,
        [candidate("p1", &product, product_quantities)],
        5,
      )
      .unwrap()
      .remove(0)
    };
    let plain = score(&none, &none);
    assert!(plain.quantities.is_empty());

    let tenfold = score(&ten, &one);
    assert_eq!(tenfold.comparison, plain.comparison, "a quantity changes no score");
    let weight = &tenfold.quantities[0];
    assert_eq!(
      (weight.field.as_str(), weight.change, weight.factor),
      ("weight", QuantityChange::Modified, Some(10.0))
    );
    assert!(!weight.beyond_limit, "a limit is a rule for revisions too");

    // Read with the stored record as the baseline: a quantity only the listing has was added.
    assert_eq!(score(&one, &none).quantities[0].change, QuantityChange::Added);
    assert_eq!(score(&none, &one).quantities[0].change, QuantityChange::Removed);
  }

  #[test]
  fn the_best_candidate_matches_at_the_threshold_and_not_below_it() {
    let listing = record(&[("name", 0.0, "listing")]);
    let product = record(&[("name", 20.0, "product")]);
    let result = ranked(&listing, &[("p1", &product)], 5);
    let score = result[0].comparison.similarity;
    assert_eq!(decide(None, &result), Decision::NotJudged, "no threshold, no judgment");
    assert_eq!(
      decide(Some(score), &result),
      Decision::Matched,
      "exactly at the threshold"
    );
    assert_eq!(decide(Some(score.next_up()), &result), Decision::NoMatch);
    assert_eq!(decide(Some(-1.0), &result), Decision::Matched);

    assert_eq!(
      decide(Some(-1.0), &[]),
      Decision::NoMatch,
      "nothing to match, whatever the threshold"
    );
    assert_eq!(decide(None, &[]), Decision::NotJudged);
  }

  /// Only the best candidate is judged: a second one above the threshold does not make a match of a first below.
  #[test]
  fn the_decision_is_about_the_first_candidate_alone() {
    let listing = record(&[("name", 0.0, "listing")]);
    let near = record(&[("name", 10.0, "near")]);
    let far = record(&[("name", 60.0, "far")]);
    let mut result = ranked(&listing, &[("near", &near), ("far", &far)], 5);
    assert_eq!(decide(Some(0.9), &result), Decision::Matched);
    result.reverse();
    assert_eq!(decide(Some(0.9), &result), Decision::NoMatch);
  }

  #[test]
  fn a_candidate_of_another_layout_is_an_error() {
    let listing = record(&[("name", 0.0, "listing")]);
    let other = RecordVectors {
      composite: vec![1.0, 0.0],
      hashes: BTreeMap::new(),
    };
    let none = Quantities::new();
    assert!(rank(&schema(), DIM, &listing, &none, [candidate("p1", &other, &none)], 5).is_err());
  }
}
