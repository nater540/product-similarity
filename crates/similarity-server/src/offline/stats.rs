//! What a scored set of pairs shows beyond its totals.
//!
//! Everything here is derived from a [`Calibration`] and nothing else, so the text report and the event stream
//! state the same numbers. Only pairs labeled `edit` or `recycle` take part, unless a function says otherwise.
//!
//! The intervals deserve a word. Pairs from one owner are written by the same people about the same product line
//! and are not independent of each other, so an interval that resampled pairs would be too narrow, and would get
//! narrower the more one brand contributed. The bootstrap here resamples **owners**, with all their pairs, which is
//! why it needs every pair to name one and reports no interval otherwise.

use std::collections::{BTreeMap, BTreeSet};

use similarity_core::FieldChange;
use similarity_core::rng::Rng;

use super::calibrate::{Calibration, ScoredPair};
use super::pairs::Label;

/// How many times the owners are resampled for an interval.
pub const RESAMPLES: usize = 1000;

/// The false-accept rates an operating point is reported for, in percent.
const TARGET_PERCENTS: [usize; 3] = [1, 2, 5];

/// How many owners the per-owner slice lists: the largest contributors, which is what it is there to show.
const OWNERS_SHOWN: usize = 10;

/// A count as a float. Counts here are numbers of pairs, far below where the conversion would lose anything.
fn number(count: usize) -> f64 {
  f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// The smallest and largest of some values, and how many there were.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
  /// How many values.
  pub count: usize,
  /// The smallest.
  pub low: f64,
  /// The largest.
  pub high: f64,
}

impl Range {
  /// The range of `values`, or `None` if there are none.
  pub fn of(values: impl IntoIterator<Item = f64>) -> Option<Self> {
    let mut values = values.into_iter();
    let first = values.next()?;
    Some(values.fold(
      Self {
        count: 1,
        low: first,
        high: first,
      },
      |range, value| Self {
        count: range.count.saturating_add(1),
        low: range.low.min(value),
        high: range.high.max(value),
      },
    ))
  }
}

/// The range of overall similarity among the pairs carrying `label`; with `None`, among the pairs with any other
/// label.
pub fn score_range(calibration: &Calibration, label: Option<Label>) -> Option<Range> {
  Range::of(
    calibration
      .pairs
      .iter()
      .filter(|scored| scored.pair.label == label)
      .map(|scored| f64::from(scored.similarity())),
  )
}

/// How often an edit's overall similarity is above a recycle's, a tie counting half: 1.0 when the two are
/// perfectly separated, 0.5 when the score says nothing.
///
/// It asks only about order, so it needs no threshold. That makes it the number to compare two models by: a
/// threshold belongs to the model it was measured on (D3), and "errors at the configured threshold" would judge
/// every other model on a scale that is not its own. `None` unless both labels are present.
pub fn area_under_curve(calibration: &Calibration) -> Option<f64> {
  let scores = |label: Label| -> Vec<f32> { calibration.labeled(label).map(ScoredPair::similarity).collect() };
  similarity_core::separation::area_under_curve(&scores(Label::Edit), &scores(Label::Recycle))
}

/// Every text field that some pair compared, in name order.
pub fn field_names(calibration: &Calibration) -> BTreeSet<&str> {
  calibration
    .pairs
    .iter()
    .flat_map(|scored| &scored.evaluation.comparison.fields)
    .map(|score| score.field.as_str())
    .collect()
}

/// The range of one field's similarity over the pairs of `label` whose text for that field actually changed.
///
/// This is what a field's `reject_below` floor is chosen from: it belongs in the gap between the two labels'
/// ranges, if there is one.
pub fn field_range(calibration: &Calibration, field: &str, label: Label) -> Option<Range> {
  Range::of(
    calibration
      .labeled(label)
      .flat_map(|scored| &scored.evaluation.comparison.fields)
      .filter(|score| score.field == field && score.change == FieldChange::Modified)
      .filter_map(|score| score.similarity.map(f64::from)),
  )
}

/// Every quantity that some pair compared, in name order.
pub fn quantity_names(calibration: &Calibration) -> BTreeSet<&str> {
  calibration
    .pairs
    .iter()
    .flat_map(|scored| &scored.evaluation.quantities)
    .map(|score| score.field.as_str())
    .collect()
}

/// What one quantity did across the pairs of one label.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct QuantityRange {
  /// The factors of the pairs that changed the amount by a finite factor.
  pub factors: Option<Range>,
  /// How many pairs put it in a unit of another dimension.
  pub incomparable: usize,
  /// How many pairs moved it between zero and a value above zero.
  pub zero_crossings: usize,
}

/// The factors `quantity` changed by among the pairs of `label`.
///
/// A factor of exactly 1 is "no change" (the same amount, perhaps in another unit). No limit can reject it, and
/// counting it would only drag every range down to 1, so it is left out.
pub fn quantity_range(calibration: &Calibration, quantity: &str, label: Label) -> QuantityRange {
  let scores = || {
    calibration
      .labeled(label)
      .flat_map(|scored| &scored.evaluation.quantities)
      .filter(move |score| score.field == quantity)
  };
  QuantityRange {
    factors: Range::of(scores().filter_map(|score| score.factor.filter(|factor| *factor > 1.0))),
    incomparable: scores().filter(|score| score.incomparable).count(),
    zero_crossings: scores().filter(|score| score.zero_crossing).count(),
  }
}

/// Where a quantity's factor limit could go, read from the pairs.
///
/// A quantity limit is an extra way to reject, beside the text checks, so what matters is that it rejects no
/// edit. The suggestion therefore sits just above the largest factor any edit shows, and says how many recycles
/// it would catch. Most recycles leave a quantity alone (another strain at the same weight), and those are for
/// the text checks to catch.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantitySuggestion {
  /// The quantity.
  pub quantity: String,
  /// The largest factor among edits, if any edit changed the amount.
  pub edits_high: Option<f64>,
  /// The smallest factor among recycles that is above every edit's, if there is one.
  pub recycles_next: Option<f64>,
  /// The suggested `reject_above_factor`: the geometric midpoint of the two values above (factors are ratios),
  /// which leaves the same room on both sides. `None` when either is missing.
  pub reject_above_factor: Option<f64>,
  /// Recycles the suggestion would reject on this quantity alone, counting those that cross zero.
  pub recycles_caught: usize,
  /// Recycles that changed the amount at all.
  pub recycles_changed: usize,
  /// Edits that moved the quantity between zero and a value above zero. Any factor limit rejects these, so a
  /// number here is a reason to look at them before setting one.
  pub edit_zero_crossings: usize,
}

/// A suggestion for every quantity that some pair compared.
pub fn quantity_suggestions(calibration: &Calibration) -> Vec<QuantitySuggestion> {
  quantity_names(calibration)
    .into_iter()
    .map(|quantity| {
      let edits = quantity_range(calibration, quantity, Label::Edit);
      let recycles = quantity_range(calibration, quantity, Label::Recycle);
      let edits_high = edits.factors.map(|range| range.high);
      // The recycles a limit could reject: those that change the amount by more than every edit does. A recycle
      // that changes it exactly as much as some edit cannot be told from that edit by any limit.
      let beyond: Vec<f64> = edits_high.map_or_else(Vec::new, |floor| {
        calibration
          .labeled(Label::Recycle)
          .flat_map(|scored| &scored.evaluation.quantities)
          .filter(|score| score.field == quantity)
          .filter_map(|score| score.factor.filter(|factor| *factor > floor))
          .collect()
      });
      let recycles_next = beyond.iter().copied().min_by(f64::total_cmp);
      let reject_above_factor = edits_high.zip(recycles_next).map(|(floor, next)| (floor * next).sqrt());
      // The limit lies below the nearest of them, so it rejects them all.
      let recycles_caught = reject_above_factor.map_or(0, |_| beyond.len().saturating_add(recycles.zero_crossings));
      QuantitySuggestion {
        quantity: quantity.to_owned(),
        edits_high,
        recycles_next,
        reject_above_factor,
        recycles_caught,
        recycles_changed: recycles
          .factors
          .map_or(0, |range| range.count)
          .saturating_add(recycles.zero_crossings),
        edit_zero_crossings: edits.zero_crossings,
      }
    })
    .collect()
}

/// The labeled pairs that share one value of some attribute, summarized.
#[derive(Debug, Clone, PartialEq)]
pub struct Slice {
  /// The attribute: `variant_change`, `category`, `label_source`, or `owner`.
  pub by: &'static str,
  /// The value the pairs share.
  pub value: String,
  /// How many edits.
  pub edits: usize,
  /// How many recycles.
  pub recycles: usize,
  /// Edits the configured policy rejects.
  pub false_rejects: usize,
  /// Recycles the configured policy accepts.
  pub false_accepts: usize,
  /// The overall similarity of the edits.
  pub edits_range: Option<Range>,
  /// The overall similarity of the recycles.
  pub recycles_range: Option<Range>,
}

fn slice_of(by: &'static str, value: &str, pairs: &[&ScoredPair]) -> Slice {
  let of = |label: Label| {
    pairs
      .iter()
      .copied()
      .filter(move |scored| scored.pair.label == Some(label))
  };
  let wrong = |label: Label| of(label).filter(|scored| scored.wrong()).count();
  let range = |label: Label| Range::of(of(label).map(|scored| f64::from(scored.similarity())));
  Slice {
    by,
    value: value.to_owned(),
    edits: of(Label::Edit).count(),
    recycles: of(Label::Recycle).count(),
    false_rejects: wrong(Label::Edit),
    false_accepts: wrong(Label::Recycle),
    edits_range: range(Label::Edit),
    recycles_range: range(Label::Recycle),
  }
}

/// The labeled pairs sliced by variant change, top-level category, label source, and owner.
///
/// A pair without a value for an attribute is in none of that attribute's slices. Slices are in name order within
/// an attribute, except `owner`, which lists only the largest contributors, largest first: it exists to show one
/// brand dominating the data.
pub fn slices(calibration: &Calibration) -> Vec<Slice> {
  type Key = for<'a> fn(&'a ScoredPair) -> Option<&'a str>;
  let attributes: [(&'static str, Key); 4] = [
    ("variant_change", |scored| scored.pair.variant_change.as_deref()),
    ("category", |scored| scored.pair.category()),
    ("label_source", |scored| scored.pair.label_source.as_deref()),
    ("owner", |scored| scored.pair.owner.as_deref()),
  ];

  let mut slices = Vec::new();
  for (by, key) in attributes {
    let mut groups: BTreeMap<&str, Vec<&ScoredPair>> = BTreeMap::new();
    for scored in calibration.pairs.iter().filter(|scored| scored.pair.label.is_some()) {
      if let Some(value) = key(scored) {
        groups.entry(value).or_default().push(scored);
      }
    }
    let mut of_attribute: Vec<Slice> = groups.iter().map(|(value, pairs)| slice_of(by, value, pairs)).collect();
    if by == "owner" {
      let size = |slice: &Slice| slice.edits.saturating_add(slice.recycles);
      // Stable, so owners of equal size stay in name order.
      of_attribute.sort_by_key(|slice| std::cmp::Reverse(size(slice)));
      of_attribute.truncate(OWNERS_SHOWN);
    }
    slices.extend(of_attribute);
  }
  slices
}

/// The middle 95% of a rate across resamples of the owners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
  /// The 2.5th percentile.
  pub low: f64,
  /// The 97.5th percentile.
  pub high: f64,
}

/// How often something goes wrong, out of how many chances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rate {
  /// How many times.
  pub errors: usize,
  /// Out of how many.
  pub of: usize,
  /// `errors / of`, or `None` when there were no chances.
  pub rate: Option<f64>,
  /// How much that rate moves when the owners are resampled. `None` when the pairs do not all name an owner, or
  /// name only one.
  pub interval: Option<Interval>,
}

/// The labeled pairs grouped by owner, or `None` if they cannot be resampled by owner: some pair names none, or
/// they all name the same one.
fn owners(calibration: &Calibration) -> Option<Vec<Vec<&ScoredPair>>> {
  let mut groups: BTreeMap<&str, Vec<&ScoredPair>> = BTreeMap::new();
  for scored in calibration.pairs.iter().filter(|scored| scored.pair.label.is_some()) {
    groups.entry(scored.pair.owner.as_deref()?).or_default().push(scored);
  }
  (groups.len() > 1).then(|| groups.into_values().collect())
}

/// Resamples owners with replacement and returns the middle 95% of `errors / chances` across the resamples.
///
/// `clusters` holds one `(errors, chances)` per owner. A resample with no chances at all has no rate and is left
/// out; if every one is, there is no interval.
fn bootstrap(clusters: &[(usize, usize)], seed: u64) -> Option<Interval> {
  let mut rng = Rng::new(seed);
  let mut rates = Vec::with_capacity(RESAMPLES);
  for _ in 0..RESAMPLES {
    let (mut errors, mut chances) = (0_usize, 0_usize);
    for _ in 0..clusters.len() {
      if let Some((cluster_errors, cluster_chances)) = clusters.get(rng.below(clusters.len())) {
        errors = errors.saturating_add(*cluster_errors);
        chances = chances.saturating_add(*cluster_chances);
      }
    }
    if chances > 0 {
      rates.push(number(errors) / number(chances));
    }
  }
  rates.sort_by(f64::total_cmp);
  let last = rates.len().checked_sub(1)?;
  // The 2.5th and 97.5th percentiles, rounding outward.
  let low = last.saturating_mul(25).checked_div(1000)?;
  let high = last.saturating_mul(975).saturating_add(999).checked_div(1000)?;
  Some(Interval {
    low: *rates.get(low)?,
    high: *rates.get(high.min(last))?,
  })
}

/// A rate over the pairs of `label`, where `is_error` says which of them count, with its interval if the owners
/// allow one.
fn rate_of(
  calibration: &Calibration,
  clusters: Option<&[Vec<&ScoredPair>]>,
  label: Label,
  is_error: impl Fn(&ScoredPair) -> bool,
  seed: u64,
) -> Rate {
  let count = |pairs: &mut dyn Iterator<Item = &ScoredPair>| {
    pairs.fold((0_usize, 0_usize), |(errors, chances), scored| {
      (
        errors.saturating_add(usize::from(is_error(scored))),
        chances.saturating_add(1),
      )
    })
  };
  let (errors, of) = count(&mut calibration.labeled(label));
  let interval = clusters.and_then(|clusters| {
    let per_owner: Vec<(usize, usize)> = clusters
      .iter()
      .map(|pairs| count(&mut pairs.iter().copied().filter(|scored| scored.pair.label == Some(label))))
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

/// How often the configured policy is wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorRates {
  /// Edits rejected, out of all edits.
  pub false_reject: Rate,
  /// Recycles accepted, out of all recycles.
  pub false_accept: Rate,
  /// How many owners the intervals were resampled from, when there are intervals.
  pub owners: Option<usize>,
}

/// The configured policy's two error rates. `seed` fixes the resampling, so the same pairs and the same seed give
/// the same intervals.
pub fn error_rates(calibration: &Calibration, seed: u64) -> ErrorRates {
  let clusters = owners(calibration);
  let wrong = |scored: &ScoredPair| scored.wrong();
  ErrorRates {
    false_reject: rate_of(calibration, clusters.as_deref(), Label::Edit, wrong, seed),
    false_accept: rate_of(calibration, clusters.as_deref(), Label::Recycle, wrong, seed),
    owners: clusters.as_ref().map(Vec::len),
  }
}

/// What it costs, in rejected edits, to let through at most a given share of recycles, on the overall score alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperatingPoint {
  /// The largest share of recycles that may be accepted.
  pub target: f64,
  /// The lowest threshold that keeps to it, or `None` if no threshold does (a recycle scores as high as anything
  /// else in the data).
  pub threshold: Option<f32>,
  /// Recycles accepted at that threshold.
  pub false_accept: Option<Rate>,
  /// Edits rejected at that threshold.
  pub false_reject: Option<Rate>,
}

/// The operating points at 1%, 2%, and 5% of recycles accepted.
///
/// These look at the overall similarity only, as a threshold does; field floors and quantity limits are not
/// applied. A target is left out when it allows no more recycles than the one before it, which is every target
/// after the first until there are a hundred recycles or so. Empty unless both labels are present.
pub fn operating_points(calibration: &Calibration, seed: u64) -> Vec<OperatingPoint> {
  let scores = |label: Label| calibration.labeled(label).map(ScoredPair::similarity);
  let recycles = scores(Label::Recycle).count();
  if recycles == 0 || scores(Label::Edit).count() == 0 {
    return Vec::new();
  }
  // The thresholds worth considering are the ones between neighboring scores, lowest first: the same candidates
  // the suggested threshold is chosen from.
  let mut sorted: Vec<f32> = scores(Label::Edit).chain(scores(Label::Recycle)).collect();
  sorted.sort_by(f32::total_cmp);
  sorted.dedup();
  let cuts: Vec<f32> = super::calibrate::cuts(&sorted).map(|(cut, _)| cut).collect();
  let clusters = owners(calibration);

  let mut points = Vec::new();
  let mut previous = None;
  for percent in TARGET_PERCENTS {
    let allowed = recycles.saturating_mul(percent).checked_div(100).unwrap_or(0);
    if previous == Some(allowed) {
      continue;
    }
    previous = Some(allowed);
    let threshold = cuts
      .iter()
      .copied()
      .find(|cut| scores(Label::Recycle).filter(|score| score >= cut).count() <= allowed);
    let rate = |label: Label, is_error: &dyn Fn(f32) -> bool| {
      rate_of(
        calibration,
        clusters.as_deref(),
        label,
        |scored| is_error(scored.similarity()),
        seed,
      )
    };
    points.push(OperatingPoint {
      target: number(percent) / 100.0,
      threshold,
      false_accept: threshold.map(|cut| rate(Label::Recycle, &|score| score >= cut)),
      false_reject: threshold.map(|cut| rate(Label::Edit, &|score| score < cut)),
    });
  }
  points
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::offline::testing::{amount, measured, scored, scored_as};

  fn summarized(pairs: Vec<ScoredPair>) -> Calibration {
    crate::offline::calibrate::summarize(pairs, crate::offline::pairs::Input::default())
  }

  #[test]
  fn a_field_range_covers_only_the_pairs_that_changed_the_field() {
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.2, false),
    ]);
    assert_eq!(field_range(&calibration, "name", Label::Edit), None);
    assert!(field_names(&calibration).is_empty());
    let overall = score_range(&calibration, Some(Label::Edit)).unwrap();
    assert_eq!(overall.count, 1);
    assert!((overall.low - f64::from(0.95_f32)).abs() < 1e-12);
    assert_eq!(score_range(&calibration, None), None);
  }

  fn with_fields(mut scored: ScoredPair, fields: &[(&str, FieldChange, Option<f32>)]) -> ScoredPair {
    scored.evaluation.comparison.fields = fields
      .iter()
      .map(|(field, change, similarity)| similarity_core::score::FieldScore {
        field: (*field).to_owned(),
        weight: 1.0,
        change: *change,
        similarity: *similarity,
        below_floor: false,
      })
      .collect();
    scored
  }

  /// A field's floor is read from this range, so what goes into it matters: this field, this label, and only
  /// where the text was really changed.
  #[test]
  fn a_field_range_is_of_one_field_in_one_label_where_its_text_was_modified() {
    let calibration = summarized(vec![
      with_fields(
        scored(Label::Edit, 0.95, true),
        &[
          ("name", FieldChange::Modified, Some(0.9)),
          ("brand", FieldChange::Modified, Some(0.3)),
        ],
      ),
      with_fields(
        scored(Label::Edit, 0.96, true),
        &[
          ("name", FieldChange::Modified, Some(0.8)),
          ("brand", FieldChange::Unchanged, Some(1.0)),
        ],
      ),
      with_fields(
        scored(Label::Edit, 0.97, true),
        &[
          ("name", FieldChange::Unchanged, Some(1.0)),
          ("brand", FieldChange::Added, None),
        ],
      ),
      with_fields(
        scored(Label::Recycle, 0.2, false),
        &[("name", FieldChange::Modified, Some(0.1))],
      ),
    ]);
    assert_eq!(
      field_names(&calibration).into_iter().collect::<Vec<_>>(),
      ["brand", "name"]
    );

    let name = field_range(&calibration, "name", Label::Edit).unwrap();
    assert_eq!(name.count, 2, "the unchanged name and the recycle's are not in it");
    assert!((name.low - f64::from(0.8_f32)).abs() < 1e-9 && (name.high - f64::from(0.9_f32)).abs() < 1e-9);
    let brand = field_range(&calibration, "brand", Label::Edit).unwrap();
    assert_eq!(
      brand.count, 1,
      "the brand that stayed and the brand that was added are not in it"
    );
    assert!(
      (brand.high - f64::from(0.3_f32)).abs() < 1e-9,
      "and it is the brand's own similarity"
    );
    let recycled = field_range(&calibration, "name", Label::Recycle).unwrap();
    assert!((recycled.high - f64::from(0.1_f32)).abs() < 1e-9);
    assert_eq!(field_range(&calibration, "brand", Label::Recycle), None);
    assert_eq!(field_range(&calibration, "description", Label::Edit), None);
  }

  /// A limit has to sit above every edit. A recycle that changes a quantity by exactly as much as the largest
  /// edit is not something a limit can catch, so it is neither the "next" recycle nor counted as caught.
  #[test]
  fn a_recycle_that_changes_a_quantity_exactly_as_an_edit_does_is_not_beyond_it() {
    let change =
      |label: Label, after: f64| measured(label, &[("weight", Some(amount(1.0, "g")), Some(amount(after, "g")))]);
    let calibration = summarized(vec![
      change(Label::Edit, 2.0),
      change(Label::Recycle, 2.0),
      change(Label::Recycle, 8.0),
      change(Label::Recycle, 16.0),
    ]);
    let [weight] = quantity_suggestions(&calibration).try_into().unwrap();
    assert!((weight.edits_high.unwrap() - 2.0).abs() < 1e-9);
    assert!(
      (weight.recycles_next.unwrap() - 8.0).abs() < 1e-9,
      "not the recycle at 2"
    );
    assert!(
      (weight.reject_above_factor.unwrap() - 4.0).abs() < 1e-9,
      "sqrt(2 * 8), which is not (2 + 8) / 2"
    );
    assert_eq!((weight.recycles_caught, weight.recycles_changed), (2, 3));
  }

  /// Two scores one float apart are still two scores, and a threshold separates them: the upper one itself.
  #[test]
  fn two_neighboring_scores_are_separated_by_the_upper_one() {
    let (recycle, edit) = (0.5_f32, 0.5_f32.next_up());
    assert_eq!(
      f32::midpoint(recycle, edit),
      recycle,
      "the midpoint cannot be told from the lower score"
    );
    let calibration = summarized(vec![
      scored(Label::Recycle, recycle, false),
      scored(Label::Edit, edit, true),
    ]);

    let suggestion = crate::offline::suggest_threshold(&[(Label::Recycle, recycle), (Label::Edit, edit)]).unwrap();
    assert_eq!(suggestion.threshold, edit);
    assert_eq!((suggestion.false_rejects, suggestion.false_accepts), (0, 0));

    let [point] = operating_points(&calibration, 0).try_into().unwrap();
    assert_eq!(point.threshold, Some(edit));
    assert_eq!(point.false_accept.unwrap().errors, 0, "the recycle is below it");
    assert_eq!(
      point.false_reject.unwrap().errors,
      0,
      "an edit at the threshold is accepted"
    );
  }

  #[test]
  fn the_area_under_the_curve_is_over_the_overall_similarity_of_labeled_pairs() {
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Edit, 0.60, false),
      scored(Label::Recycle, 0.70, false),
      scored(Label::Recycle, 0.20, false),
    ]);
    // Of the four pairings, the edit at 0.60 loses one.
    assert_eq!(area_under_curve(&calibration), Some(0.75));
    assert_eq!(
      area_under_curve(&summarized(vec![scored(Label::Edit, 0.9, true)])),
      None
    );
  }

  #[test]
  fn a_quantity_suggestion_sits_above_every_edit_and_below_the_next_recycle() {
    let change = |label: Label, before: f64, after: f64| {
      measured(
        label,
        &[("weight", Some(amount(before, "g")), Some(amount(after, "g")))],
      )
    };
    let calibration = summarized(vec![
      change(Label::Edit, 1.0, 1.0),
      change(Label::Edit, 1.0, 1.1),
      change(Label::Edit, 10.0, 8.0),
      // One recycle changes it less than an edit does. That one is for the text checks.
      change(Label::Recycle, 1.0, 1.2),
      change(Label::Recycle, 1.0, 5.0),
      change(Label::Recycle, 40.0, 2.0),
      change(Label::Recycle, 0.0, 3.0),
      change(Label::Recycle, 7.0, 7.0),
    ]);
    let [weight] = quantity_suggestions(&calibration).try_into().unwrap();
    assert_eq!(weight.quantity, "weight");
    assert!((weight.edits_high.unwrap() - 1.25).abs() < 1e-9);
    assert!((weight.recycles_next.unwrap() - 5.0).abs() < 1e-9);
    assert!(
      (weight.reject_above_factor.unwrap() - 2.5).abs() < 1e-9,
      "sqrt(1.25 * 5)"
    );
    assert_eq!(
      (
        weight.recycles_caught,
        weight.recycles_changed,
        weight.edit_zero_crossings
      ),
      (3, 4, 0),
      "the two big changes and the one from zero"
    );
  }

  #[test]
  fn a_quantity_no_edit_changes_or_no_recycle_exceeds_gets_no_suggestion() {
    let change = |label: Label, name: &str, before: f64, after: f64| {
      measured(label, &[(name, Some(amount(before, "g")), Some(amount(after, "g")))])
    };
    let calibration = summarized(vec![
      change(Label::Edit, "steady", 1.0, 1.0),
      change(Label::Recycle, "steady", 1.0, 9.0),
      change(Label::Edit, "noisy", 1.0, 9.0),
      change(Label::Edit, "noisy", 0.0, 9.0),
      change(Label::Recycle, "noisy", 1.0, 2.0),
    ]);
    let suggestions = quantity_suggestions(&calibration);
    let noisy = &suggestions[0];
    assert_eq!((noisy.quantity.as_str(), noisy.reject_above_factor), ("noisy", None));
    assert_eq!((noisy.recycles_caught, noisy.edit_zero_crossings), (0, 1));
    let steady = &suggestions[1];
    assert_eq!((steady.edits_high, steady.reject_above_factor), (None, None));
    assert_eq!(steady.recycles_changed, 1);
  }

  #[test]
  fn slices_count_each_attribute_separately_and_their_counts_add_up() {
    let pair = |label: Label, similarity: f32, accepted: bool, owner: &str, variant: Option<&str>, category: &str| {
      scored_as(label, similarity, accepted, |pair| {
        pair.owner = Some(owner.to_owned());
        pair.variant_change = variant.map(str::to_owned);
        pair.label_source = Some("moderator".to_owned());
        pair.before.texts.insert("category".to_owned(), category.to_owned());
      })
    };
    let calibration = summarized(vec![
      pair(Label::Edit, 0.95, true, "b1", None, "Edibles > Gummies"),
      pair(Label::Edit, 0.70, false, "b1", Some("flavor"), "Edibles > Chocolates"),
      pair(Label::Recycle, 0.85, true, "b2", Some("flavor"), "Flower"),
      pair(Label::Recycle, 0.20, false, "b1", Some("size"), "Flower > Sativa"),
    ]);
    let slices = slices(&calibration);
    let find = |by: &str, value: &str| {
      slices
        .iter()
        .find(|slice| slice.by == by && slice.value == value)
        .unwrap_or_else(|| panic!("no slice {by}={value}: {slices:?}"))
    };

    let flavor = find("variant_change", "flavor");
    assert_eq!(
      (
        flavor.edits,
        flavor.recycles,
        flavor.false_rejects,
        flavor.false_accepts
      ),
      (1, 1, 1, 1)
    );
    let edibles = find("category", "Edibles");
    assert_eq!((edibles.edits, edibles.recycles, edibles.false_rejects), (2, 0, 1));
    assert_eq!(edibles.edits_range.unwrap().count, 2);
    assert_eq!(edibles.recycles_range, None);
    assert_eq!(find("category", "Flower").recycles, 2);
    assert_eq!(find("label_source", "moderator").edits, 2);

    // Owners are listed largest first.
    let owners: Vec<&str> = slices
      .iter()
      .filter(|slice| slice.by == "owner")
      .map(|slice| slice.value.as_str())
      .collect();
    assert_eq!(owners, ["b1", "b2"]);

    // Every pair has a category and an owner here, so those attributes' slices cover the whole.
    for by in ["category", "owner", "label_source"] {
      let total: usize = slices
        .iter()
        .filter(|slice| slice.by == by)
        .map(|slice| slice.edits + slice.recycles)
        .sum();
      assert_eq!(total, 4, "{by}");
    }
    // And only two pairs have a variant change at all... three, in fact: one of each above, plus the size.
    let variants: usize = slices
      .iter()
      .filter(|slice| slice.by == "variant_change")
      .map(|slice| slice.edits + slice.recycles)
      .sum();
    assert_eq!(variants, 3);
  }

  #[test]
  fn only_the_ten_largest_owners_are_listed() {
    let mut pairs = Vec::new();
    for owner in 0..12 {
      for _ in 0..=owner {
        pairs.push(owned(Label::Edit, 0.9, true, &format!("b{owner:02}")));
      }
    }
    let slices = slices(&summarized(pairs));
    let owners: Vec<(&str, usize)> = slices
      .iter()
      .filter(|slice| slice.by == "owner")
      .map(|slice| (slice.value.as_str(), slice.edits))
      .collect();
    assert_eq!(owners.len(), 10);
    assert_eq!(owners[0], ("b11", 12));
    assert_eq!(owners[9], ("b02", 3));
  }

  fn owned(label: Label, similarity: f32, accepted: bool, owner: &str) -> ScoredPair {
    scored_as(label, similarity, accepted, |pair| {
      pair.owner = Some(owner.to_owned());
    })
  }

  #[test]
  fn the_error_rates_are_counts_over_chances() {
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Edit, 0.70, false),
      scored(Label::Edit, 0.92, true),
      scored(Label::Edit, 0.91, true),
      scored(Label::Recycle, 0.85, true),
      scored(Label::Recycle, 0.20, false),
    ]);
    let rates = error_rates(&calibration, 0);
    assert_eq!((rates.false_reject.errors, rates.false_reject.of), (1, 4));
    assert!((rates.false_reject.rate.unwrap() - 0.25).abs() < 1e-12);
    assert!((rates.false_accept.rate.unwrap() - 0.5).abs() < 1e-12);
    assert_eq!(
      (rates.false_reject.interval, rates.owners),
      (None, None),
      "no owners, no interval"
    );

    let no_recycles = error_rates(&summarized(vec![scored(Label::Edit, 0.9, true)]), 0);
    assert_eq!((no_recycles.false_accept.of, no_recycles.false_accept.rate), (0, None));
  }

  #[test]
  fn an_interval_comes_from_resampling_owners_and_is_fixed_by_the_seed() {
    // Brand b1 gets everything right; brand b2 gets half its edits wrong; b3 has one wrong edit.
    let mut pairs = Vec::new();
    for _ in 0..10 {
      pairs.push(owned(Label::Edit, 0.95, true, "b1"));
    }
    for index in 0..10 {
      pairs.push(owned(Label::Edit, 0.95, index % 2 == 0, "b2"));
    }
    pairs.push(owned(Label::Edit, 0.5, false, "b3"));
    pairs.push(owned(Label::Recycle, 0.2, false, "b1"));
    let calibration = summarized(pairs);

    let rates = error_rates(&calibration, 0);
    assert_eq!(rates.owners, Some(3));
    let rate = rates.false_reject.rate.unwrap();
    assert!((rate - 6.0 / 21.0).abs() < 1e-12);
    let interval = rates.false_reject.interval.unwrap();
    assert!(
      interval.low <= rate && rate <= interval.high,
      "{interval:?} around {rate}"
    );
    assert!(
      interval.low >= 0.0 && interval.high <= 1.0,
      "a rate is a share: {interval:?}"
    );
    assert!(
      interval.low < 0.01 && interval.high > 0.5,
      "a resample of only b1 is all right, one of only b2 and b3 is mostly wrong: {interval:?}"
    );

    assert_eq!(error_rates(&calibration, 0), rates, "the same seed, the same intervals");
    // Every resample has the same recycles (b1's one, or none), and it is never wrong.
    let accepts = rates.false_accept.interval.unwrap();
    assert!(accepts.low == 0.0 && accepts.high == 0.0, "{accepts:?}");
  }

  #[test]
  fn one_owner_or_a_pair_without_one_means_no_interval() {
    let one = summarized(vec![
      owned(Label::Edit, 0.9, true, "b1"),
      owned(Label::Recycle, 0.2, false, "b1"),
    ]);
    assert_eq!(error_rates(&one, 0).false_reject.interval, None);

    let partly = summarized(vec![
      owned(Label::Edit, 0.9, true, "b1"),
      owned(Label::Edit, 0.9, true, "b2"),
      scored(Label::Recycle, 0.2, false),
    ]);
    assert_eq!(error_rates(&partly, 0).owners, None);

    // An unlabeled pair without an owner is not in the way: it is not resampled.
    let unsure = scored_as(Label::Edit, 0.5, true, |pair| {
      pair.label = None;
      pair.label_text = "unsure".to_owned();
    });
    let with_unsure = summarized(vec![
      owned(Label::Edit, 0.9, true, "b1"),
      owned(Label::Recycle, 0.2, false, "b2"),
      unsure,
    ]);
    assert_eq!(error_rates(&with_unsure, 0).owners, Some(2));
    assert_eq!(
      error_rates(&with_unsure, 0).false_reject.of,
      1,
      "and it is not counted as an edit"
    );
  }

  #[test]
  fn an_operating_point_is_the_lowest_threshold_that_lets_few_enough_recycles_through() {
    // 100 recycles from 0.000 to 0.495, and one high one; edits from 0.40 to 0.99.
    let mut pairs: Vec<ScoredPair> = (0..100_u16)
      .map(|index| scored(Label::Recycle, f32::from(index) * 0.005, false))
      .collect();
    pairs.push(scored(Label::Recycle, 0.97, true));
    pairs.extend((0..60_u16).map(|index| scored(Label::Edit, 0.40 + f32::from(index) * 0.01, true)));
    let calibration = summarized(pairs);

    let points = operating_points(&calibration, 0);
    let targets: Vec<f64> = points.iter().map(|point| point.target).collect();
    assert_eq!(targets, [0.01, 0.02, 0.05], "101 recycles allow 1, 2, and 5 through");

    // At 1%, exactly one recycle may pass: the high one. The cut is just above the highest of the rest.
    let strict = &points[0];
    let threshold = strict.threshold.unwrap();
    assert!((0.495..0.505).contains(&threshold), "{threshold}");
    assert_eq!(strict.false_accept.unwrap().errors, 1);
    let rejected = strict.false_reject.unwrap();
    assert_eq!((rejected.errors, rejected.of), (10, 60), "edits from 0.40 to 0.49");

    // Letting more recycles through lowers the threshold and costs fewer edits.
    assert!(points[2].threshold.unwrap() < threshold);
    assert!(points[2].false_reject.unwrap().errors < rejected.errors);
  }

  #[test]
  fn few_recycles_give_one_operating_point_and_an_unreachable_target_says_so() {
    let few = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.50, false),
      scored(Label::Recycle, 0.30, false),
    ]);
    let points = operating_points(&few, 0);
    assert_eq!(points.len(), 1, "1%, 2%, and 5% of two recycles are all none");
    assert!((points[0].threshold.unwrap() - 0.725).abs() < 1e-6);
    assert_eq!(points[0].false_reject.unwrap().errors, 0);

    // A recycle that scores as high as the best edit cannot be kept out by any threshold.
    let tied = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.95, true),
    ]);
    let points = operating_points(&tied, 0);
    assert_eq!((points.len(), points[0].threshold), (1, None));

    assert!(operating_points(&summarized(vec![scored(Label::Edit, 0.9, true)]), 0).is_empty());
  }
}
