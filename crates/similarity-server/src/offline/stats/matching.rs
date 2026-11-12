//! What a match calibration shows.
//!
//! Everything a report or an event says about a [`MatchCalibration`] is computed here, so the two cannot
//! disagree. Ranking is measured on the served ranking, which is what a caller of `Match` would get.
//!
//! **A match threshold separates two populations**, and they are the two a revision threshold separates, in
//! another guise: the score a listing gives its own product, which a threshold should be at or below, and the
//! score it gives the best other product, which stands in for a listing whose product is not in the catalog and
//! which a threshold should be above. [`populations`] writes them as the `edit` and `recycle` pairs of a
//! [`Calibration`], one of each per listing and both carrying the listing's owner, so that the suggested
//! threshold, the operating points, the area under the curve, and the owner-resampled intervals are the tested
//! ones and not a second copy of them.

use std::collections::BTreeMap;

use similarity_core::QuantityChange;
use similarity_core::score::{Comparison, Evaluation, Reason};

use super::super::RecordFields;
use super::super::calibrate::{Calibration, ScoredPair, Suggestion, summarize};
use super::super::matching::{MatchCalibration, ScoredListing};
use super::super::pairs::{Input, Label, Pair};
use super::{OperatingPoint, Range, Rate, area_under_curve, bootstrap, number, operating_points, score_range};

/// How many of the listings something is true of, with an interval where the owners allow one. The count is in
/// [`Rate::errors`], whatever is being counted.
fn share(calibration: &MatchCalibration, seed: u64, is: impl Fn(&ScoredListing) -> Option<bool>) -> Rate {
  let count = |listings: &mut dyn Iterator<Item = &ScoredListing>| {
    listings.fold((0_usize, 0_usize), |(hits, chances), listing| match is(listing) {
      Some(hit) => (hits.saturating_add(usize::from(hit)), chances.saturating_add(1)),
      None => (hits, chances),
    })
  };
  let (errors, of) = count(&mut calibration.listings.iter());
  let interval = owners(calibration).and_then(|clusters| {
    let per_owner: Vec<(usize, usize)> = clusters
      .iter()
      .map(|listings| count(&mut listings.iter().copied()))
      .collect();
    bootstrap(&per_owner, seed)
  });
  Rate {
    errors,
    of,
    rate: (of > 0).then(|| number(errors) / number(of)),
    interval,
  }
}

/// The listings grouped by the owner of their product, or `None` if they cannot be resampled by owner: some
/// listing names none, or they all name the same one.
fn owners(calibration: &MatchCalibration) -> Option<Vec<Vec<&ScoredListing>>> {
  let mut groups: BTreeMap<&str, Vec<&ScoredListing>> = BTreeMap::new();
  for listing in &calibration.listings {
    groups
      .entry(listing.listing.owner.as_deref()?)
      .or_default()
      .push(listing);
  }
  (groups.len() > 1).then(|| groups.into_values().collect())
}

/// The share of listings whose own product is among the first `k` of the served ranking.
pub fn recall(calibration: &MatchCalibration, k: usize, seed: u64) -> Rate {
  share(calibration, seed, |listing| {
    Some(listing.own_rank.is_some_and(|rank| rank <= k))
  })
}

/// The mean of one over the rank of each listing's own product in the served ranking, a product beyond the
/// depth adding nothing. `None` without listings.
pub fn mean_reciprocal_rank(calibration: &MatchCalibration) -> Option<f64> {
  let total: f64 = calibration
    .listings
    .iter()
    .map(|listing| listing.own_rank.map_or(0.0, |rank| 1.0 / number(rank)))
    .sum();
  (!calibration.listings.is_empty()).then(|| total / number(calibration.listings.len()))
}

fn judged(id: String, label: Label, owner: Option<&str>, similarity: f32, threshold: Option<f32>) -> ScoredPair {
  let reasons = match threshold {
    Some(threshold) if similarity < threshold => vec![Reason::BelowThreshold { similarity, threshold }],
    _ => Vec::new(),
  };
  ScoredPair {
    pair: Pair {
      id,
      label: Some(label),
      label_text: label.as_str().to_owned(),
      note: String::new(),
      split: None,
      owner: owner.map(str::to_owned),
      variant_change: None,
      label_source: None,
      label_confidence: None,
      before: RecordFields::default(),
      after: RecordFields::default(),
      anchor: None,
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

/// The two populations a match threshold separates, as a calibration: for each listing an `edit` pair scoring
/// what its own product scores, and, if the index has another product, a `recycle` pair scoring what the best
/// other one scores. A pair is "accepted" when its score reaches `threshold`; with no threshold every pair is.
///
/// Read through it, a false reject is a listing that would not be matched to its own product, and a false accept
/// is a listing that would be matched to something were its own product not in the catalog.
pub fn populations(calibration: &MatchCalibration, threshold: Option<f32>) -> Calibration {
  let mut pairs = Vec::new();
  for scored in &calibration.listings {
    let listing = &scored.listing;
    let owner = listing.owner.as_deref();
    pairs.push(judged(
      listing.id.clone(),
      Label::Edit,
      owner,
      scored.own.comparison.similarity,
      threshold,
    ));
    if let Some(other) = &scored.best_other {
      pairs.push(judged(
        format!("{}/other", listing.id),
        Label::Recycle,
        owner,
        other.ranked.comparison.similarity,
        threshold,
      ));
    }
  }
  summarize(
    pairs,
    Input {
      sha256: calibration.input.sha256.clone(),
      split: calibration.input.split.clone(),
      ..Input::default()
    },
  )
}

/// What a threshold does to the listings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtThreshold {
  /// The threshold.
  pub threshold: f32,
  /// Listings whose own product scores below it: they would go unmatched, or be matched to something else.
  pub own_below: Rate,
  /// Listings whose best other product reaches it: were their own product not in the catalog, they would be
  /// matched to that one.
  pub other_reaches: Rate,
  /// Listings that are matched to a product that is not theirs, with theirs in the catalog: the first of the
  /// served ranking is another product, and it reaches the threshold.
  pub wrong: Rate,
}

/// How one quantity of the listings compares with their own products'.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantityAgreement {
  /// The quantity.
  pub quantity: String,
  /// Listings where both the listing and its product have a value that can be compared.
  pub compared: usize,
  /// Of those, the ones that agree after unit conversion.
  pub same: usize,
  /// Listings whose value is in a unit of another dimension than their product's.
  pub incomparable: usize,
  /// Listings that have a value their product lacks.
  pub only_listing: usize,
  /// Listings that lack a value their product has.
  pub only_product: usize,
}

/// Everything the report and the events say about a match calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchSummary {
  /// How many listings were scored.
  pub listings: usize,
  /// How many owners the intervals were resampled from, when there are intervals.
  pub owners: Option<usize>,
  /// Own product first in the served ranking.
  pub recall_at_1: Rate,
  /// Own product among the first five.
  pub recall_at_5: Rate,
  /// Own product within the depth.
  pub recall_at_depth: Rate,
  /// Own product first when every product is scored exactly: what an unlimited over-fetch would give.
  pub exhaustive_recall_at_1: Rate,
  /// The mean reciprocal rank over the served ranking.
  pub mean_reciprocal_rank: Option<f64>,
  /// Listings whose served ranking is not the exhaustive one within the depth.
  pub overfetch_lost: Rate,
  /// The scores listings give their own products.
  pub own: Option<Range>,
  /// The scores listings give the best other product.
  pub best_other: Option<Range>,
  /// How often a listing's own product outscores a best other product, a tie counting half.
  pub auc: Option<f64>,
  /// Where a threshold would best separate the two.
  pub suggestion: Option<Suggestion>,
  /// What it costs in unmatched listings to keep wrong matches of absent products rare.
  pub operating_points: Vec<OperatingPoint>,
  /// Best other products that belong to the owner of the listing's own product.
  pub siblings: Rate,
  /// What the configured threshold does, if one is configured.
  pub at_threshold: Option<AtThreshold>,
  /// Each quantity, compared between the listings and their own products.
  pub quantities: Vec<QuantityAgreement>,
}

fn quantities(calibration: &MatchCalibration) -> Vec<QuantityAgreement> {
  let mut found: BTreeMap<&str, QuantityAgreement> = BTreeMap::new();
  for listing in &calibration.listings {
    for score in &listing.own.quantities {
      let entry = found.entry(score.field.as_str()).or_insert_with(|| QuantityAgreement {
        quantity: score.field.clone(),
        compared: 0,
        same: 0,
        incomparable: 0,
        only_listing: 0,
        only_product: 0,
      });
      let slot = match score.change {
        QuantityChange::Added | QuantityChange::Restored => &mut entry.only_listing,
        QuantityChange::Removed => &mut entry.only_product,
        _ if score.incomparable => &mut entry.incomparable,
        QuantityChange::Unchanged => {
          entry.same = entry.same.saturating_add(1);
          &mut entry.compared
        }
        _ => &mut entry.compared,
      };
      *slot = slot.saturating_add(1);
    }
  }
  found.into_values().collect()
}

/// Summarizes a match calibration. `threshold` is the configured match threshold, if there is one; `seed` fixes
/// the resampling.
pub fn summarize_match(calibration: &MatchCalibration, threshold: Option<f32>, seed: u64) -> MatchSummary {
  let two = populations(calibration, threshold);
  let at_threshold = threshold.map(|threshold| AtThreshold {
    threshold,
    own_below: share(calibration, seed, |listing| {
      Some(listing.own.comparison.similarity < threshold)
    }),
    other_reaches: share(calibration, seed, |listing| {
      listing
        .best_other
        .as_ref()
        .map(|other| other.ranked.comparison.similarity >= threshold)
    }),
    wrong: share(calibration, seed, |listing| {
      let beaten = listing.own_rank != Some(1);
      let top = listing
        .best_other
        .as_ref()
        .is_some_and(|other| other.ranked.comparison.similarity >= threshold);
      Some(beaten && top)
    }),
  });
  MatchSummary {
    listings: calibration.listings.len(),
    owners: owners(calibration).map(|clusters| clusters.len()),
    recall_at_1: recall(calibration, 1, seed),
    recall_at_5: recall(calibration, 5, seed),
    recall_at_depth: recall(calibration, calibration.depth, seed),
    exhaustive_recall_at_1: share(calibration, seed, |listing| Some(listing.exhaustive_own_rank == 1)),
    mean_reciprocal_rank: mean_reciprocal_rank(calibration),
    overfetch_lost: share(calibration, seed, |listing| Some(listing.overfetch_lost)),
    own: score_range(&two, Some(Label::Edit)),
    best_other: score_range(&two, Some(Label::Recycle)),
    auc: area_under_curve(&two),
    suggestion: two.suggestion,
    operating_points: operating_points(&two, seed),
    siblings: share(calibration, seed, |listing| {
      listing.best_other.as_ref().map(|other| other.same_owner)
    }),
    at_threshold,
    quantities: quantities(calibration),
  }
}

#[cfg(test)]
mod tests {
  use similarity_core::score::compare_quantities;
  use similarity_core::{Quantities, Quantity, QuantitySpec, Schema};

  use super::super::super::matching::testing::{calibration, scored};
  use super::*;

  /// Five listings of two owners. Own product first for three of them, third for one, and not served for one.
  fn mixed() -> MatchCalibration {
    calibration(
      vec![
        scored("a", Some("brand-1"), 0.95, Some(1), Some(("b", true, 0.70))),
        scored("b", Some("brand-1"), 0.90, Some(1), Some(("a", true, 0.80))),
        scored("c", Some("brand-2"), 0.85, Some(1), Some(("a", false, 0.60))),
        scored("d", Some("brand-2"), 0.75, Some(3), Some(("c", true, 0.88))),
        scored("e", Some("brand-2"), 0.65, None, Some(("c", true, 0.92))),
      ],
      5,
    )
  }

  #[test]
  fn recall_counts_listings_whose_own_product_is_within_the_first_k_served() {
    let calibration = mixed();
    let at = |k: usize| {
      let rate = recall(&calibration, k, 0);
      (rate.errors, rate.of, rate.rate)
    };
    assert_eq!(at(1), (3, 5, Some(0.6)));
    assert_eq!(at(2), (3, 5, Some(0.6)));
    assert_eq!(at(3), (4, 5, Some(0.8)));
    assert_eq!(
      at(5),
      (4, 5, Some(0.8)),
      "a product that was not served is in no first k"
    );
    assert_eq!(
      recall(
        &super::super::super::matching::testing::calibration(Vec::new(), 5),
        1,
        0
      )
      .rate,
      None
    );
  }

  #[test]
  fn a_product_beyond_the_depth_adds_nothing_to_the_mean_reciprocal_rank() {
    let expected = (1.0 + 1.0 + 1.0 + 1.0 / 3.0 + 0.0) / 5.0;
    assert!((mean_reciprocal_rank(&mixed()).unwrap() - expected).abs() < 1e-12);
    assert_eq!(mean_reciprocal_rank(&calibration(Vec::new(), 5)), None);
  }

  /// Intervals come from resampling owners, so they need every listing to name one and more than one owner.
  #[test]
  fn a_share_has_an_interval_only_when_the_owners_can_be_resampled() {
    let with = recall(&mixed(), 1, 0);
    let interval = with.interval.unwrap();
    assert!(interval.low <= 0.6 && interval.high >= 0.6, "{interval:?}");
    assert_eq!(
      recall(&mixed(), 1, 0),
      recall(&mixed(), 1, 0),
      "the same seed, the same interval"
    );

    let mut unowned = mixed();
    unowned.listings[0].listing.owner = None;
    assert_eq!(recall(&unowned, 1, 0).interval, None);
    let one_owner = calibration(vec![scored("a", Some("brand-1"), 0.9, Some(1), None)], 5);
    assert_eq!(recall(&one_owner, 1, 0).interval, None);
  }

  /// The two populations are what a threshold has to separate: each listing's own score, and its best other.
  #[test]
  fn the_two_populations_are_each_listings_own_score_and_its_best_other() {
    let two = populations(&mixed(), None);
    let scores = |label: Label| -> Vec<f32> { two.labeled(label).map(ScoredPair::similarity).collect() };
    assert_eq!(scores(Label::Edit), [0.95, 0.90, 0.85, 0.75, 0.65]);
    assert_eq!(scores(Label::Recycle), [0.70, 0.80, 0.60, 0.88, 0.92]);
    assert!(two.pairs.iter().all(|pair| pair.pair.owner.is_some()));
    assert_eq!(two.false_rejects, 0, "with no threshold nothing is below one");

    let at = populations(&mixed(), Some(0.8));
    assert_eq!(
      (at.false_rejects, at.false_accepts),
      (2, 3),
      "own below 0.8, and best other at or above it"
    );

    // It says which groups file and which split it was made from, as any calibration says of its pairs.
    let mut from = mixed();
    from.input.sha256 = "abc".to_owned();
    from.input.split = Some("holdout".to_owned());
    let two = populations(&from, None);
    assert_eq!(
      (two.input.sha256.as_str(), two.input.split.as_deref()),
      ("abc", Some("holdout"))
    );

    // A listing with no other product to be confused with adds no second score.
    let alone = calibration(vec![scored("a", Some("brand-1"), 0.9, Some(1), None)], 5);
    let two = populations(&alone, None);
    assert_eq!((two.pairs.len(), two.suggestion), (1, None));
  }

  #[test]
  fn a_summary_says_what_a_threshold_would_do_and_where_one_belongs() {
    let summary = summarize_match(&mixed(), Some(0.8), 0);
    assert_eq!((summary.listings, summary.owners), (5, Some(2)));
    assert_eq!(
      (
        summary.recall_at_1.errors,
        summary.recall_at_5.errors,
        summary.recall_at_depth.errors
      ),
      (3, 4, 4)
    );
    assert_eq!(summary.exhaustive_recall_at_1.errors, 3);
    assert_eq!(summary.overfetch_lost.errors, 0);
    let (own, other) = (summary.own.unwrap(), summary.best_other.unwrap());
    assert!((own.low - 0.65).abs() < 1e-6 && (own.high - 0.95).abs() < 1e-6 && own.count == 5);
    assert!((other.low - 0.60).abs() < 1e-6 && (other.high - 0.92).abs() < 1e-6);
    // Of the 25 pairings, own is above best other in 5 + 4 + 3 + 2 + 1.
    assert_eq!(summary.auc, Some(15.0 / 25.0));
    assert_eq!((summary.siblings.errors, summary.siblings.of), (4, 5));

    let at = summary.at_threshold.unwrap();
    assert_eq!((at.own_below.errors, at.own_below.of), (2, 5));
    assert_eq!((at.other_reaches.errors, at.other_reaches.of), (3, 5));
    assert_eq!(
      (at.wrong.errors, at.wrong.of),
      (2, 5),
      "the two whose own product is not first, and whose best other reaches the threshold"
    );
    let suggestion = summary.suggestion.unwrap();
    assert_eq!(
      suggestion.false_rejects + suggestion.false_accepts,
      4,
      "the two overlap, and no cut gets fewer than four of the ten scores wrong: {suggestion:?}"
    );
    assert!(!summary.operating_points.is_empty());

    // A score exactly at the threshold reaches it, on both sides: an own product at 0.75 is not below 0.75, and
    // a best other at 0.88 reaches 0.88.
    let at = |threshold: f32| summarize_match(&mixed(), Some(threshold), 0).at_threshold.unwrap();
    assert_eq!(
      (at(0.75).own_below.errors, at(0.75_f32.next_up()).own_below.errors),
      (1, 2)
    );
    assert_eq!(
      (
        at(0.88).other_reaches.errors,
        at(0.88_f32.next_up()).other_reaches.errors
      ),
      (2, 1)
    );
    assert_eq!((at(0.88).wrong.errors, at(0.88_f32.next_up()).wrong.errors), (2, 1));

    let unjudged = summarize_match(&mixed(), None, 0);
    assert!(unjudged.at_threshold.is_none());
    assert_eq!(
      unjudged.suggestion, summary.suggestion,
      "a suggestion does not depend on what is configured"
    );
    assert_eq!(unjudged.auc, summary.auc);
  }

  #[test]
  fn an_overfetch_loss_is_a_listing_whose_served_ranking_is_not_the_exhaustive_one() {
    let mut calibration = mixed();
    calibration.listings[4].overfetch_lost = true;
    calibration.listings[4].exhaustive_own_rank = 1;
    let summary = summarize_match(&calibration, None, 0);
    assert_eq!((summary.overfetch_lost.errors, summary.overfetch_lost.of), (1, 5));
    assert_eq!(
      (summary.recall_at_1.errors, summary.exhaustive_recall_at_1.errors),
      (3, 4),
      "scoring everything would have put it first"
    );
  }

  /// Quantities are compared between each listing and its own product, and only counted.
  #[test]
  fn quantities_are_counted_by_how_the_listing_compares_with_its_own_product() {
    let schema = Schema::new(vec![similarity_core::FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: false,
      reject_below: None,
    }])
    .unwrap()
    .with_quantities(
      ["thc", "weight"]
        .iter()
        .map(|name| QuantitySpec {
          name: (*name).to_owned(),
          reject_above_factor: None,
          reject_incomparable: false,
        })
        .collect(),
    )
    .unwrap();
    let of = |entries: &[(&str, f64, &str)]| -> Quantities {
      entries
        .iter()
        .map(|(name, value, unit)| ((*name).to_owned(), Quantity::single(*value, *unit)))
        .collect()
    };
    let compared =
      |product: &Quantities, listing: &Quantities| compare_quantities(&schema, product, &Quantities::new(), listing);
    let mut calibration = mixed();
    // Same weight in another unit; a weight of another size; a weight only the product has; a potency in a unit
    // of another dimension, and a weight only the listing has.
    calibration.listings[0].own.quantities = compared(&of(&[("weight", 1.0, "g")]), &of(&[("weight", 1000.0, "mg")]));
    calibration.listings[1].own.quantities = compared(&of(&[("weight", 1.0, "g")]), &of(&[("weight", 3.5, "g")]));
    calibration.listings[2].own.quantities = compared(&of(&[("weight", 1.0, "g")]), &of(&[]));
    calibration.listings[3].own.quantities = compared(
      &of(&[("thc", 20.0, "%")]),
      &of(&[("thc", 100.0, "mg"), ("weight", 1.0, "g")]),
    );
    let summary = summarize_match(&calibration, None, 0);
    assert_eq!(
      summary.quantities,
      [
        QuantityAgreement {
          quantity: "thc".to_owned(),
          compared: 0,
          same: 0,
          incomparable: 1,
          only_listing: 0,
          only_product: 0,
        },
        QuantityAgreement {
          quantity: "weight".to_owned(),
          compared: 2,
          same: 1,
          incomparable: 0,
          only_listing: 1,
          only_product: 1,
        },
      ]
    );
  }
}
