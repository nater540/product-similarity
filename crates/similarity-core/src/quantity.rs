//! The numeric fields of a record (a weight, a volume, a potency), compared as numbers.
//!
//! A quantity is never embedded. A sentence encoder reads `1 g` and `1000 mg` as different text and `THC 100 mg` and
//! `THC 1000 mg` as nearly the same text, which is backwards on both counts (docs/decisions.md, D16). Compared as
//! numbers, a quantity gives three signals the text model cannot:
//!
//! - **How much it changed**, after unit conversion, as a [`QuantityScore::factor`]: `3.5 g` to `28 g` is 8.
//! - **That it stopped being comparable**: a potency in `%` becoming a potency in `mg` is not a bigger or smaller
//!   number, it is a different kind of product.
//! - **Which measures a record has**: a weight disappearing as a volume appears.
//!
//! Nothing in this module decides anything. [`compare`] describes one quantity; `crate::score` holds the limits
//! that turn a description into a rejection, and they are off unless configured.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Two converted values closer than this, relative to the larger, are the same value.
///
/// Wide enough to absorb the rounding of a unit conversion (`0.125 oz` and its gram equivalent written to a few
/// places would still differ, and should), narrow enough that no real edit hides inside it.
const RELATIVE_TOLERANCE: f64 = 1e-6;

/// A record's quantities: name to value.
pub type Quantities = BTreeMap<String, Quantity>;

/// A measured amount with its unit. A single value is a range whose ends are equal.
///
/// Read from JSON in either spelling, `{"value": 3.5, "unit": "g"}` or `{"min": 18, "max": 22, "unit": "%"}`, and
/// always written as the second.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "QuantityInput")]
pub struct Quantity {
  /// The low end of the range.
  pub min: f64,
  /// The high end of the range.
  pub max: f64,
  /// The unit both ends are in, as the caller wrote it.
  pub unit: String,
}

/// The shapes a quantity is accepted in. Kept apart from [`Quantity`] so that a file written by hand may say
/// `value` while everything in memory and in storage has exactly one shape.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantityInput {
  #[serde(default)]
  value: Option<f64>,
  #[serde(default)]
  min: Option<f64>,
  #[serde(default)]
  max: Option<f64>,
  unit: String,
}

impl TryFrom<QuantityInput> for Quantity {
  type Error = String;

  fn try_from(input: QuantityInput) -> Result<Self, Self::Error> {
    let (min, max) = match (input.value, input.min, input.max) {
      (Some(value), None, None) => (value, value),
      (None, Some(min), Some(max)) => (min, max),
      // One end alone is refused rather than guessed at: "at least 18" and "exactly 18" are different claims, and
      // a caller that has only one number says which by sending it as both ends or as `value`.
      _ => return Err("a quantity is either `value` and `unit`, or `min`, `max`, and `unit`".to_owned()),
    };
    Ok(Self {
      min,
      max,
      unit: input.unit,
    })
  }
}

impl Quantity {
  /// A single value.
  pub fn single(value: f64, unit: impl Into<String>) -> Self {
    Self::range(value, value, unit)
  }

  /// A range.
  pub fn range(min: f64, max: f64, unit: impl Into<String>) -> Self {
    Self {
      min,
      max,
      unit: unit.into(),
    }
  }

  /// Checks that this is a usable measurement.
  ///
  /// # Errors
  ///
  /// Returns what is wrong, without naming the field (the caller knows it): an end that is not a finite number or
  /// is negative, a range that runs backwards, a blank unit, or a value too large to convert to its base unit.
  pub fn check(&self) -> Result<(), String> {
    for (end, value) in [("min", self.min), ("max", self.max)] {
      if !value.is_finite() {
        return Err(format!("`{end}` is {value}, which is not a finite number"));
      }
      if value < 0.0 {
        return Err(format!("`{end}` is {value}, and a quantity cannot be negative"));
      }
    }
    if self.min > self.max {
      return Err(format!("`min` {} is above `max` {}", self.min, self.max));
    }
    if self.unit.trim().is_empty() {
      return Err("`unit` is blank".to_owned());
    }
    // Comparison happens in the base unit. A value that overflows on the way there would compare as infinity
    // against infinity, which is not a number, and a factor that is not a number passes every limit.
    let (_, _, converted) = self.converted();
    if !converted.is_finite() {
      return Err(format!(
        "`max` is {} {}, which is too large to compare",
        self.max,
        self.unit.trim()
      ));
    }
    Ok(())
  }

  /// The same quantity with the whitespace around its unit removed, which is the form that is stored.
  #[must_use]
  pub fn trimmed(&self) -> Self {
    Self::range(self.min, self.max, self.unit.trim())
  }

  /// Both ends in the base unit of this quantity's dimension.
  fn converted(&self) -> (Dimension, f64, f64) {
    let (dimension, factor) = unit(&self.unit);
    (dimension, self.min * factor, self.max * factor)
  }

  /// What this quantity's unit measures.
  #[must_use]
  pub fn dimension(&self) -> Dimension {
    unit(&self.unit).0
  }

  /// The same amount written in the unit `to`, which is kept as the caller wrote it apart from the space around
  /// it. `to` in the spelling this quantity already has, or in another spelling of the same unit, changes no
  /// number.
  ///
  /// This is the conversion [`compare`] makes before it compares, offered on its own: two quantities that
  /// convert into one another are exactly the ones that are comparable.
  ///
  /// # Errors
  ///
  /// [`Unconvertible::OtherDimension`] if the two units measure different things, which includes a unit the table
  /// does not know against any unit but itself. [`Unconvertible::TooLarge`] if the amount does not fit in a
  /// number once it is written in `to`.
  pub fn in_unit(&self, to: &str) -> Result<Self, Unconvertible> {
    let (from_dimension, from_factor) = unit(&self.unit);
    let (to_dimension, to_factor) = unit(to);
    if from_dimension != to_dimension {
      return Err(Unconvertible::OtherDimension {
        from: self.unit.trim().to_owned(),
        to: to.trim().to_owned(),
      });
    }
    // One unit under two spellings is the same numbers, and not the numbers a multiplication and a division by
    // one factor happen to round to.
    let convert = |value: f64| {
      if from_factor.to_bits() == to_factor.to_bits() {
        value
      } else {
        value * from_factor / to_factor
      }
    };
    let (min, max) = (convert(self.min), convert(self.max));
    if !(min.is_finite() && max.is_finite()) {
      return Err(Unconvertible::TooLarge {
        from: self.unit.trim().to_owned(),
        to: to.trim().to_owned(),
      });
    }
    Ok(Self::range(min, max, to.trim()))
  }

  /// The same amount in the unit its dimension is compared in: `mg`, `ml`, or `%`. A quantity in a unit the
  /// table does not know comes back in that unit, normalized.
  ///
  /// # Errors
  ///
  /// [`Unconvertible::TooLarge`] if the amount does not fit in a number in that unit, which [`Self::check`]
  /// refuses first for a quantity it was asked about.
  pub fn in_base_unit(&self) -> Result<Self, Unconvertible> {
    self.in_unit(self.dimension().base_unit())
  }
}

/// Why a quantity cannot be written in another unit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Unconvertible {
  /// The units measure different things. Not a failure of the table: a potency in `%` has no weight in `mg`.
  #[error("`{from}` and `{to}` do not measure the same thing, so one cannot be written as the other")]
  OtherDimension {
    /// The unit the quantity is in.
    from: String,
    /// The unit that was asked for.
    to: String,
  },
  /// The amount is a number in its own unit and not in the one asked for.
  #[error("the amount is too large to write in `{to}` (it is in `{from}`)")]
  TooLarge {
    /// The unit the quantity is in.
    from: String,
    /// The unit that was asked for.
    to: String,
  },
}

/// What a unit measures. Quantities are only comparable within one dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dimension {
  /// Mass, in milligrams.
  Mass,
  /// Volume, in milliliters.
  Volume,
  /// A share of the whole, in percent.
  Fraction,
  /// A unit the table does not know, which is comparable only with the identical unit (after trimming, collapsing
  /// whitespace, and lowercasing). Holds that normalized unit.
  Other(String),
}

impl Dimension {
  /// The unit amounts of this dimension are compared in, in the spelling [`units`] lists first for it. A unit
  /// the table does not know is its own.
  #[must_use]
  pub fn base_unit(&self) -> &str {
    match self {
      Self::Mass => "mg",
      Self::Volume => "ml",
      Self::Fraction => "%",
      Self::Other(unit) => unit,
    }
  }
}

/// One unit of the table: every spelling it is recognized under, what it measures, and how many of its
/// dimension's base unit one of it is.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitEntry {
  /// The spellings, already lowercase and with single spaces: what a unit is matched against after the same is
  /// done to it.
  pub spellings: &'static [&'static str],
  /// What the unit measures. Never [`Dimension::Other`].
  pub dimension: Dimension,
  /// One of this unit in [`Dimension::base_unit`]: 1,000 for `g`, whose base unit is `mg`.
  pub factor: f64,
}

/// Every unit that is converted, in the table's order. A unit that is not here is still accepted everywhere a
/// quantity is: it is comparable with itself and with nothing else.
pub fn units() -> impl Iterator<Item = UnitEntry> {
  UNITS.iter().map(|(spellings, dimension, factor)| UnitEntry {
    spellings,
    dimension: dimension.clone(),
    factor: *factor,
  })
}

/// Every unit this crate can convert: the spellings, the dimension, and the factor to the dimension's base unit.
///
/// Extending it is one line. A unit that is missing is never an error, only a [`Dimension::Other`], so the table
/// can lag behind the data without refusing a record.
const UNITS: &[(&[&str], Dimension, f64)] = &[
  // "µ" is U+00B5 (micro sign) and "μ" is U+03BC (Greek mu). Keyboards and databases produce both.
  (&["mcg", "µg", "μg", "microgram", "micrograms"], Dimension::Mass, 0.001),
  (&["mg", "milligram", "milligrams"], Dimension::Mass, 1.0),
  (&["g", "gram", "grams"], Dimension::Mass, 1_000.0),
  (&["kg", "kilogram", "kilograms"], Dimension::Mass, 1_000_000.0),
  (&["oz", "ounce", "ounces"], Dimension::Mass, 28_349.523_125),
  (&["lb", "lbs", "pound", "pounds"], Dimension::Mass, 453_592.37),
  (
    &["ml", "milliliter", "milliliters", "millilitre", "millilitres"],
    Dimension::Volume,
    1.0,
  ),
  (&["cl", "centiliter", "centiliters"], Dimension::Volume, 10.0),
  (&["l", "liter", "liters", "litre", "litres"], Dimension::Volume, 1_000.0),
  (
    &["fl oz", "floz", "fl. oz", "fl. oz.", "fluid ounce", "fluid ounces"],
    Dimension::Volume,
    29.573_529_562_5,
  ),
  (&["%", "percent", "pct"], Dimension::Fraction, 1.0),
];

/// A unit in the spelling it is matched in: trimmed, inner whitespace collapsed, and in lower case. Two units
/// that are the same here are one unit everywhere, and it is what [`Dimension::Other`] holds.
#[must_use]
pub fn normalized(raw: &str) -> String {
  raw.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// The dimension of a unit and the factor that converts a value in it to the dimension's base unit.
///
/// Units are matched after trimming, collapsing inner whitespace, and lowercasing, so `G`, ` g `, and `g` are one
/// unit and `fl  oz` is `fl oz`.
pub fn unit(raw: &str) -> (Dimension, f64) {
  let normalized = normalized(raw);
  UNITS
    .iter()
    .find(|(spellings, ..)| spellings.contains(&normalized.as_str()))
    .map_or((Dimension::Other(normalized), 1.0), |(_, dimension, factor)| {
      (dimension.clone(), *factor)
    })
}

/// What happened to one quantity between the baseline and the candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantityChange {
  /// Equal after unit conversion.
  Unchanged,
  /// Both sides have a value in the same dimension, and the values differ.
  Modified,
  /// Both sides have a value, in different dimensions.
  Incomparable,
  /// Only the candidate has a value, and the record has never had one for this quantity.
  Added,
  /// Only the baseline has a value.
  Removed,
  /// The baseline has no value, but the record had one when it was first stored, and the candidate has one again.
  /// It is compared against that original value.
  Restored,
}

/// The comparison of one quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantityScore {
  /// The quantity's name.
  pub field: String,
  /// What happened to it.
  pub change: QuantityChange,
  /// How many times larger or smaller the value became, whichever end of the range moved more: always at least 1,
  /// so "eight times bigger" and "an eighth" both read 8, and exactly 1 when nothing changed. `None` when there is
  /// no number to give: nothing to compare against ([`QuantityChange::Added`], [`QuantityChange::Removed`]),
  /// [`Self::incomparable`], or [`Self::zero_crossing`].
  pub factor: Option<f64>,
  /// One end of the range moved between zero and a value above zero, so the factor is unbounded.
  pub zero_crossing: bool,
  /// The two values are in different dimensions. Always set for [`QuantityChange::Incomparable`]; also set for a
  /// [`QuantityChange::Restored`] quantity whose original value was in another dimension.
  pub incomparable: bool,
  /// The unit of the value compared against: the baseline's, or the original's when restored. `None` when added.
  pub previous_unit: Option<String>,
  /// The candidate's unit. `None` when removed.
  pub unit: Option<String>,
  /// This quantity alone rejects the revision. Set by the policy in `crate::score`, never by [`compare`].
  pub beyond_limit: bool,
}

/// How far apart two values in the same unit are: the larger over the smaller.
///
/// `Some(1.0)` when they are equal within tolerance (two zeros included), `None` when exactly one of them is zero,
/// since no finite factor takes zero to anything else.
///
/// Always finite. Two valid amounts can be so far apart that their quotient overflows (`1e-300 g` against
/// `1e300 g`), and an infinite factor cannot be written as a JSON number: an event carrying it would hold `null`
/// and the run could not be read back. Such a pair gets the largest number there is, which is beyond every limit.
fn ratio(left: f64, right: f64) -> Option<f64> {
  let (low, high) = if left <= right { (left, right) } else { (right, left) };
  if high - low <= RELATIVE_TOLERANCE * high {
    Some(1.0)
  } else if low <= 0.0 {
    None
  } else {
    Some((high / low).min(f64::MAX))
  }
}

/// Compares `candidate` against `reference`, which both exist.
fn measure(field: &str, change: QuantityChange, reference: &Quantity, candidate: &Quantity) -> QuantityScore {
  let (old_dimension, old_min, old_max) = reference.converted();
  let (new_dimension, new_min, new_max) = candidate.converted();
  let mut score = QuantityScore {
    field: field.to_owned(),
    change,
    factor: None,
    zero_crossing: false,
    incomparable: false,
    previous_unit: Some(reference.unit.trim().to_owned()),
    unit: Some(candidate.unit.trim().to_owned()),
    beyond_limit: false,
  };
  if old_dimension != new_dimension {
    score.incomparable = true;
    // A restored quantity stays "restored": that it came back is what happened to it, and the flag says the rest.
    if change != QuantityChange::Restored {
      score.change = QuantityChange::Incomparable;
    }
    return score;
  }
  match (ratio(old_min, new_min), ratio(old_max, new_max)) {
    (Some(low), Some(high)) => {
      let factor = low.max(high);
      score.factor = Some(factor);
      if factor <= 1.0 && change == QuantityChange::Modified {
        score.change = QuantityChange::Unchanged;
      }
    }
    _ => score.zero_crossing = true,
  }
  score
}

/// Compares a quantity with the first value the record ever had for it, as [`compare`] compares it with the
/// version before: how far it has come in all, however small each step was.
///
/// `Added` when the record has never had the quantity before this version, `Removed` when this version lacks it.
/// `None` when neither has it.
pub fn against_first(field: &str, first: Option<&Quantity>, candidate: Option<&Quantity>) -> Option<QuantityScore> {
  compare(field, first, first, candidate)
}

/// Compares one quantity across two versions of a record.
///
/// `anchor` is the first value the record ever had for this quantity. It matters only when the baseline has none:
/// a quantity that was blanked and comes back is then compared with what it originally was, for the reason a text
/// field is (docs/decisions.md, D8): otherwise any quantity could be swapped in two steps.
///
/// Returns `None` when neither the baseline nor the candidate has a value, since there is nothing to report.
pub fn compare(
  field: &str,
  baseline: Option<&Quantity>,
  anchor: Option<&Quantity>,
  candidate: Option<&Quantity>,
) -> Option<QuantityScore> {
  let one_sided = |change, previous: Option<&Quantity>, current: Option<&Quantity>| QuantityScore {
    field: field.to_owned(),
    change,
    factor: None,
    zero_crossing: false,
    incomparable: false,
    previous_unit: previous.map(|quantity| quantity.unit.trim().to_owned()),
    unit: current.map(|quantity| quantity.unit.trim().to_owned()),
    beyond_limit: false,
  };
  match (baseline, candidate) {
    (None, None) => None,
    (Some(old), Some(new)) => Some(measure(field, QuantityChange::Modified, old, new)),
    (Some(old), None) => Some(one_sided(QuantityChange::Removed, Some(old), None)),
    (None, Some(new)) => Some(match anchor {
      Some(original) => measure(field, QuantityChange::Restored, original, new),
      None => one_sided(QuantityChange::Added, None, Some(new)),
    }),
  }
}

/// The anchor's quantities after an accepted revision: unchanged, except that a quantity it has never had a value
/// for takes the candidate's.
///
/// The counterpart of `score::extend_anchor` for quantities, with the same rule: a value the anchor already holds
/// is never replaced here. Only a forced write does that, by starting a new anchor.
pub fn first_values(anchor: &Quantities, candidate: &Quantities) -> Quantities {
  let mut extended = anchor.clone();
  for (name, quantity) in candidate {
    extended.entry(name.clone()).or_insert_with(|| quantity.clone());
  }
  extended
}

#[cfg(test)]
mod tests {
  use super::*;

  fn compared(baseline: &Quantity, candidate: &Quantity) -> QuantityScore {
    compare("q", Some(baseline), None, Some(candidate)).unwrap()
  }

  fn grams(value: f64) -> Quantity {
    Quantity::single(value, "g")
  }

  /// Both amounts pass `check`, and their quotient overflows. An infinite factor would be written to an event as
  /// `null`, and the run could not be read back.
  #[test]
  fn a_factor_too_large_to_write_down_is_still_a_number() {
    let (tiny, huge) = (grams(1e-300), grams(1e300));
    assert!(tiny.check().is_ok() && huge.check().is_ok());
    for score in [compared(&tiny, &huge), compared(&huge, &tiny)] {
      let factor = score.factor.unwrap();
      assert!(factor.is_finite() && factor > 1e300, "{factor}");
      let written = serde_json::to_string(&factor).unwrap();
      assert_eq!(
        serde_json::from_str::<f64>(&written).unwrap().to_bits(),
        factor.to_bits()
      );
    }
  }

  /// Four doublings, each within a limit of two, are sixteen times the first value.
  #[test]
  fn a_quantity_is_also_measured_against_its_first_value() {
    let first = grams(3.5);
    let walked = against_first("q", Some(&first), Some(&grams(56.0))).unwrap();
    assert_eq!(walked.change, QuantityChange::Modified);
    assert!((walked.factor.unwrap() - 16.0).abs() < 1e-9);
    assert_eq!(
      against_first("q", None, Some(&first)).unwrap().change,
      QuantityChange::Added
    );
    assert_eq!(
      against_first("q", Some(&first), None).unwrap().change,
      QuantityChange::Removed
    );
    assert_eq!(against_first("q", None, None), None);
    let same = against_first("q", Some(&first), Some(&Quantity::single(3500.0, "mg"))).unwrap();
    assert_eq!(same.change, QuantityChange::Unchanged);
  }

  #[test]
  fn the_same_amount_in_another_unit_is_unchanged() {
    for (before, after) in [
      (grams(1.0), Quantity::single(1000.0, "mg")),
      (Quantity::single(1000.0, "mg"), grams(1.0)),
      (grams(3.5), Quantity::single(3.5, " Grams ")),
      (Quantity::single(1.0, "l"), Quantity::single(100.0, "cl")),
      (Quantity::single(22.0, "%"), Quantity::single(22.0, "percent")),
      (Quantity::single(1.0, "kg"), Quantity::single(1_000_000_000.0, "mcg")),
      (Quantity::single(5.0, "µg"), Quantity::single(5.0, "μg")),
    ] {
      let score = compared(&before, &after);
      assert_eq!(score.change, QuantityChange::Unchanged, "{before:?} to {after:?}");
      assert_eq!(score.factor, Some(1.0));
      assert!(!score.incomparable && !score.zero_crossing && !score.beyond_limit);
    }
  }

  #[test]
  fn a_changed_amount_reports_the_same_factor_in_both_directions() {
    let bigger = compared(&grams(3.5), &grams(28.0));
    let smaller = compared(&grams(28.0), &grams(3.5));
    assert_eq!(bigger.change, QuantityChange::Modified);
    assert!((bigger.factor.unwrap() - 8.0).abs() < 1e-9);
    assert!((smaller.factor.unwrap() - 8.0).abs() < 1e-9);
    assert_eq!(
      (bigger.previous_unit.as_deref(), bigger.unit.as_deref()),
      (Some("g"), Some("g"))
    );

    // Across units too: an eighth of an ounce is about 3.54 g, so this is a factor of 8 and a little.
    let ounces = compared(&grams(3.5), &Quantity::single(1.0, "oz"));
    assert!((ounces.factor.unwrap() - 28.349_523_125 / 3.5).abs() < 1e-9);
  }

  #[test]
  fn a_rounded_conversion_is_a_small_change_and_not_an_identical_value() {
    let score = compared(&grams(3.5), &Quantity::single(0.125, "oz"));
    assert_eq!(score.change, QuantityChange::Modified);
    assert!((1.0..1.02).contains(&score.factor.unwrap()), "{score:?}");
  }

  #[test]
  fn the_factor_of_a_range_is_that_of_the_end_that_moved_more() {
    let routine = compared(&Quantity::range(18.0, 22.0, "%"), &Quantity::range(19.0, 23.0, "%"));
    assert!((routine.factor.unwrap() - 19.0 / 18.0).abs() < 1e-9);

    let widened = compared(&Quantity::range(18.0, 22.0, "%"), &Quantity::range(9.0, 23.0, "%"));
    assert!((widened.factor.unwrap() - 2.0).abs() < 1e-9, "the low end halved");

    // A single value against a range is compared end by end as well.
    let spread = compared(&Quantity::single(20.0, "%"), &Quantity::range(10.0, 25.0, "%"));
    assert!((spread.factor.unwrap() - 2.0).abs() < 1e-9);
  }

  #[test]
  fn a_move_between_zero_and_something_has_no_factor() {
    for (before, after) in [(0.0, 5.0), (5.0, 0.0)] {
      let score = compared(&Quantity::single(before, "mg"), &Quantity::single(after, "mg"));
      assert_eq!(score.change, QuantityChange::Modified);
      assert_eq!(score.factor, None);
      assert!(score.zero_crossing && !score.incomparable);
    }

    // One end is enough.
    let partly = compared(&Quantity::range(0.0, 2.0, "%"), &Quantity::range(0.5, 2.0, "%"));
    assert!(partly.zero_crossing && partly.factor.is_none());

    let still_none = compared(&Quantity::single(0.0, "mg"), &Quantity::single(0.0, "g"));
    assert_eq!(
      (still_none.change, still_none.factor, still_none.zero_crossing),
      (QuantityChange::Unchanged, Some(1.0), false)
    );
  }

  #[test]
  fn another_dimension_is_incomparable() {
    let score = compared(&Quantity::single(22.0, "%"), &Quantity::single(22.0, "mg"));
    assert_eq!(score.change, QuantityChange::Incomparable);
    assert!(score.incomparable && score.factor.is_none() && !score.zero_crossing);
    assert_eq!(
      (score.previous_unit.as_deref(), score.unit.as_deref()),
      (Some("%"), Some("mg"))
    );

    let weight_to_volume = compared(&grams(3.5), &Quantity::single(355.0, "ml"));
    assert_eq!(weight_to_volume.change, QuantityChange::Incomparable);
  }

  #[test]
  fn an_unknown_unit_is_comparable_only_with_itself() {
    let same = compared(&Quantity::single(10.0, "mg/g"), &Quantity::single(20.0, " MG/G "));
    assert_eq!(same.change, QuantityChange::Modified);
    assert!((same.factor.unwrap() - 2.0).abs() < 1e-9);

    let other = compared(&Quantity::single(10.0, "mg/g"), &Quantity::single(10.0, "mg/ml"));
    assert_eq!(other.change, QuantityChange::Incomparable);
    let known = compared(&Quantity::single(10.0, "mg/g"), &Quantity::single(10.0, "mg"));
    assert_eq!(known.change, QuantityChange::Incomparable);

    assert_eq!(unit("Each"), (Dimension::Other("each".to_owned()), 1.0));
  }

  #[test]
  fn units_are_matched_without_regard_to_case_or_spacing() {
    assert_eq!(normalized("  Fl \t Oz "), "fl oz");
    assert_eq!(normalized("G"), "g");
    assert_eq!(normalized("  "), "");
    assert_eq!(unit(" G "), (Dimension::Mass, 1000.0));
    assert_eq!(unit("Fl  Oz"), unit("fl oz"));
    assert_eq!(unit("fl oz").0, Dimension::Volume);
    assert_eq!(unit("oz").0, Dimension::Mass);
    assert_eq!(unit("%"), (Dimension::Fraction, 1.0));
  }

  #[test]
  fn a_quantity_on_one_side_only_is_added_or_removed() {
    let weight = grams(3.5);
    let added = compare("q", None, None, Some(&weight)).unwrap();
    assert_eq!(added.change, QuantityChange::Added);
    assert_eq!(
      (added.factor, added.previous_unit.as_deref(), added.unit.as_deref()),
      (None, None, Some("g"))
    );

    let removed = compare("q", Some(&weight), None, None).unwrap();
    assert_eq!(removed.change, QuantityChange::Removed);
    assert_eq!(
      (removed.previous_unit.as_deref(), removed.unit.as_deref()),
      (Some("g"), None)
    );
    // The anchor is irrelevant while the baseline has a value, or when nothing comes back.
    assert_eq!(compare("q", Some(&weight), Some(&grams(1.0)), None).unwrap(), removed);
    assert_eq!(compare("q", None, Some(&weight), None), None);
    assert_eq!(compare("q", None, None, None), None);
  }

  #[test]
  fn a_quantity_that_comes_back_is_compared_with_the_value_it_first_had() {
    let original = grams(3.5);

    let same = compare("q", None, Some(&original), Some(&Quantity::single(3500.0, "mg"))).unwrap();
    assert_eq!((same.change, same.factor), (QuantityChange::Restored, Some(1.0)));

    let swapped = compare("q", None, Some(&original), Some(&grams(28.0))).unwrap();
    assert_eq!(swapped.change, QuantityChange::Restored);
    assert!((swapped.factor.unwrap() - 8.0).abs() < 1e-9);
    assert_eq!(swapped.previous_unit.as_deref(), Some("g"));

    let other_kind = compare("q", None, Some(&original), Some(&Quantity::single(355.0, "ml"))).unwrap();
    assert_eq!(other_kind.change, QuantityChange::Restored);
    assert!(other_kind.incomparable && other_kind.factor.is_none());
  }

  #[test]
  fn the_anchor_takes_the_first_value_of_a_quantity_and_never_a_later_one() {
    let anchor = Quantities::from([("weight".to_owned(), grams(3.5))]);
    let candidate = Quantities::from([
      ("weight".to_owned(), grams(28.0)),
      ("thc".to_owned(), Quantity::range(18.0, 22.0, "%")),
    ]);
    let extended = first_values(&anchor, &candidate);
    assert_eq!(extended["weight"], grams(3.5), "the first weight stays");
    assert_eq!(
      extended["thc"],
      Quantity::range(18.0, 22.0, "%"),
      "the new potency is taken"
    );

    // A quantity going missing changes nothing.
    assert_eq!(first_values(&extended, &Quantities::new()), extended);
  }

  #[test]
  fn invalid_quantities_are_refused_with_the_reason() {
    let reason = |quantity: Quantity| quantity.check().unwrap_err();
    assert!(reason(grams(f64::NAN)).contains("finite"));
    assert!(reason(grams(f64::INFINITY)).contains("finite"));
    assert!(reason(grams(-1.0)).contains("negative"));
    assert!(reason(Quantity::range(5.0, 2.0, "g")).contains("above"));
    assert!(reason(Quantity::single(1.0, "  ")).contains("blank"));
    assert!(reason(Quantity::single(1e305, "kg")).contains("too large"));
    assert!(Quantity::single(1e305, "mg").check().is_ok());
    assert!(grams(0.0).check().is_ok());
    assert!(Quantity::range(18.0, 22.0, "%").check().is_ok());
    assert_eq!(Quantity::single(1.0, " g ").trimmed().unit, "g");
  }

  #[test]
  fn a_quantity_reads_from_json_in_either_spelling_and_writes_one() {
    let single: Quantity = serde_json::from_str(r#"{"value": 3.5, "unit": "g"}"#).unwrap();
    assert_eq!(single, grams(3.5));
    let range: Quantity = serde_json::from_str(r#"{"min": 18, "max": 22, "unit": "%"}"#).unwrap();
    assert_eq!(range, Quantity::range(18.0, 22.0, "%"));

    assert_eq!(
      serde_json::to_value(&single).unwrap(),
      serde_json::json!({"min": 3.5, "max": 3.5, "unit": "g"})
    );
    let reread: Quantity = serde_json::from_value(serde_json::to_value(&range).unwrap()).unwrap();
    assert_eq!(reread, range);

    for malformed in [
      r#"{"value": 3.5}"#,
      r#"{"unit": "g"}"#,
      r#"{"min": 1, "unit": "g"}"#,
      r#"{"value": 1, "min": 1, "max": 2, "unit": "g"}"#,
      r#"{"value": 1, "unit": "g", "per": "item"}"#,
      r#"{"value": "3.5", "unit": "g"}"#,
    ] {
      assert!(serde_json::from_str::<Quantity>(malformed).is_err(), "{malformed}");
    }
  }

  #[test]
  fn a_quantity_is_written_in_another_unit_of_its_dimension() {
    assert_eq!(grams(3.5).in_unit("mg").unwrap(), Quantity::single(3500.0, "mg"));
    assert_eq!(Quantity::single(3500.0, "mg").in_unit("g").unwrap(), grams(3.5));
    assert_eq!(
      Quantity::range(0.5, 2.0, "L").in_unit(" ml ").unwrap(),
      Quantity::range(500.0, 2000.0, "ml")
    );
    // An ounce is 28.349523125 g exactly, by definition.
    assert_eq!(Quantity::single(1.0, "oz").in_unit("g").unwrap(), grams(28.349_523_125));
    let eighth = grams(3.5).in_unit("oz").unwrap();
    assert!((eighth.min - 0.123_458_86).abs() < 1e-8, "{}", eighth.min);
    assert_eq!(eighth.min.to_bits(), eighth.max.to_bits());
    assert_eq!(eighth.unit, "oz");
  }

  /// `28.349523125 * f / f` is not always `28.349523125`. One unit under another of its names is not arithmetic.
  #[test]
  fn another_spelling_of_the_same_unit_changes_no_number() {
    for amount in [0.1, 1.0 / 3.0, 28.349_523_125, 1e-9, 7e11] {
      for (from, to) in [
        ("oz", "ounces"),
        ("OZ", "oz"),
        ("fl oz", "fluid ounce"),
        ("lb", "pounds"),
        ("%", "pct"),
      ] {
        let rewritten = Quantity::single(amount, from).in_unit(to).unwrap();
        assert_eq!(rewritten.min.to_bits(), amount.to_bits(), "{amount} {from} in {to}");
        assert_eq!(rewritten.unit, to);
      }
    }
  }

  #[test]
  fn units_that_measure_different_things_do_not_convert() {
    let other = |from: &str, to: &str| Unconvertible::OtherDimension {
      from: from.to_owned(),
      to: to.to_owned(),
    };
    assert_eq!(grams(1.0).in_unit("ml"), Err(other("g", "ml")));
    assert_eq!(Quantity::single(20.0, " % ").in_unit("mg"), Err(other("%", "mg")));
    // A unit the table does not know is comparable with itself, in any case and spacing, and with nothing else.
    assert_eq!(
      Quantity::single(2.0, "Pack").in_unit(" pack"),
      Ok(Quantity::single(2.0, "pack"))
    );
    assert_eq!(
      Quantity::single(2.0, "pack").in_unit("each"),
      Err(other("pack", "each"))
    );
    assert_eq!(Quantity::single(2.0, "pack").in_unit("g"), Err(other("pack", "g")));
  }

  #[test]
  fn an_amount_that_is_not_a_number_in_the_other_unit_is_refused() {
    let too_large = Unconvertible::TooLarge {
      from: "kg".to_owned(),
      to: "mcg".to_owned(),
    };
    // 1e300 kg is a number of milligrams (`check` passes) and not a number of micrograms.
    let huge = Quantity::single(1e300, "kg");
    assert!(huge.check().is_ok());
    assert_eq!(huge.in_unit("mcg"), Err(too_large.clone()));
    assert_eq!(huge.in_unit("g").unwrap(), grams(1e303));
    // Either end is enough, also the low end of a range that was never checked.
    assert_eq!(Quantity::range(1e300, 1.0, "kg").in_unit("mcg"), Err(too_large.clone()));
    assert_eq!(Quantity::range(1.0, 1e300, "kg").in_unit("mcg"), Err(too_large));
  }

  #[test]
  fn the_base_unit_of_a_dimension_is_the_unit_it_is_compared_in() {
    assert_eq!(grams(3.5).in_base_unit().unwrap(), Quantity::single(3500.0, "mg"));
    assert_eq!(
      Quantity::single(1.0, "fl oz").in_base_unit().unwrap(),
      Quantity::single(29.573_529_562_5, "ml")
    );
    assert_eq!(
      Quantity::single(20.0, "percent").in_base_unit().unwrap(),
      Quantity::single(20.0, "%")
    );
    assert_eq!(
      Quantity::single(2.0, " Pre  Rolls ").in_base_unit().unwrap(),
      Quantity::single(2.0, "pre rolls")
    );
    assert_eq!(grams(1.0).dimension(), Dimension::Mass);
    assert_eq!(Quantity::single(1.0, "cl").dimension(), Dimension::Volume);
    assert_eq!(Quantity::single(1.0, "pct").dimension(), Dimension::Fraction);
    assert_eq!(
      Quantity::single(1.0, "Each").dimension(),
      Dimension::Other("each".to_owned())
    );
  }

  /// The listing is the table: every spelling in it is recognized as what the listing says, a base unit is listed
  /// with a factor of exactly one, and no spelling is listed twice.
  #[test]
  fn the_listed_units_are_the_ones_that_are_recognized() {
    let listed: Vec<UnitEntry> = units().collect();
    assert_eq!(listed.len(), UNITS.len());
    let mut seen = std::collections::BTreeSet::new();
    for entry in &listed {
      assert!(!entry.spellings.is_empty());
      assert!(entry.factor > 0.0 && entry.factor.is_finite());
      assert!(!matches!(entry.dimension, Dimension::Other(_)));
      for spelling in entry.spellings {
        assert!(seen.insert(*spelling), "`{spelling}` is listed twice");
        assert_eq!(unit(spelling), (entry.dimension.clone(), entry.factor), "{spelling}");
      }
    }
    for dimension in [Dimension::Mass, Dimension::Volume, Dimension::Fraction] {
      assert_eq!(unit(dimension.base_unit()), (dimension.clone(), 1.0));
      let first = listed
        .iter()
        .find(|entry| entry.factor == 1.0 && entry.dimension == dimension)
        .unwrap();
      assert_eq!(first.spellings.first(), Some(&dimension.base_unit()));
    }
  }
}
