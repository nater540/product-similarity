//! An amount written as text (`3.5g`, `1/8 oz`, `18-22%`), read into a [`Quantity`].
//!
//! Systems that hold a weight or a potency as a string each need this before they can send a record, and each
//! parser of their own is a place where `1,000 mg` becomes 1 and `1/8` becomes 1. This is the one place.
//!
//! **The whole string is the amount, or it is not an amount.** [`parse`] does not find an amount inside prose
//! (`Blueberry Gummies 100mg` is a name, and what a name says stays in the name: docs/decisions.md, D16), and it
//! never guesses. A string it cannot read exactly is refused with a reason, because a wrong number that looks
//! right is worse than no number: it becomes a baseline. Data that is wrong is refused and never rescued, by the
//! owner's decision (2026-11-22).
//!
//! What is read: a decimal (`3.5`, `.5`), thousands grouped by commas (`1,000`), a fraction (`1/8`), a mixed
//! number (`1 1/2`, `1-1/2`, `1½`), a range of any two of those (`18-22`, `18 to 22`), and then a unit. The unit
//! is kept as it was written and handed to [`quantity::unit`] like any other, so one the table does not know is
//! still not an error. `data/amounts.cases.jsonl` is every form and every refusal, one to a line, and the tests
//! here hold [`parse`] to all of it.
//!
//! Nothing here decides anything, and nothing is written back: there is no canonical spelling of an amount.

use crate::quantity::{self, Quantity};

/// Why a text is not an amount. [`Self::code`] is the stable name of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NotAnAmount {
  /// Nothing but whitespace.
  #[error("there is nothing to read")]
  Empty,
  /// It does not start with a number: `g`, `THC 22%`, `eighth`. A word in front makes it prose.
  #[error("it does not start with a number")]
  NoNumber,
  /// A number and nothing after it: `3.5`. A number is not a measurement.
  #[error("the number has no unit")]
  NoUnit,
  /// A comma that is not grouping thousands: `3,5 g` is three and a half, or thirty-five with a typo.
  #[error("a comma in the number is not grouping thousands, so the number could be read two ways")]
  AmbiguousSeparator,
  /// It starts as a number and is not one: `1.2.3 g`, `1/0 oz`.
  #[error("the number is not written in a way that can be read")]
  BadNumber,
  /// `-1 g`.
  #[error("an amount cannot be negative")]
  Negative,
  /// `22-18 %`.
  #[error("the range runs backwards")]
  BackwardsRange,
  /// `500 mg - 1 g`: a range in two units is somebody's mistake more often than their meaning.
  #[error("the two ends of the range are in different units")]
  MixedUnits,
  /// `10 x 10 mg`: 100 mg a package, or 10 mg an item. Only the caller knows which its quantity means.
  #[error("it is one amount times another, and which of the two is meant is not this reader's to say")]
  Multiple,
  /// `<1 %`, `~3.5 g`: "less than one" is a claim about a range that cannot be made for the caller.
  #[error("it is qualified (less than, more than, about), which a number and a unit cannot say")]
  Qualified,
  /// What follows the number is not made of what a unit is made of: `3.5 g (1/8 oz)`, `20 %+`.
  #[error("what follows the number cannot be read as a unit")]
  UnreadableUnit,
  /// A number and a unit, and too large to compare: [`Quantity::check`] refuses it.
  #[error("the amount is too large to compare")]
  TooLarge,
}

impl NotAnAmount {
  /// Every reason, in the order they are declared.
  pub const ALL: [Self; 12] = [
    Self::Empty,
    Self::NoNumber,
    Self::NoUnit,
    Self::AmbiguousSeparator,
    Self::BadNumber,
    Self::Negative,
    Self::BackwardsRange,
    Self::MixedUnits,
    Self::Multiple,
    Self::Qualified,
    Self::UnreadableUnit,
    Self::TooLarge,
  ];

  /// The stable name of this reason, for a file, an event, or a count. An interface in the sense
  /// `Reason::code` is (D17): a code is added, and never renamed or given another meaning.
  #[must_use]
  pub fn code(self) -> &'static str {
    match self {
      Self::Empty => "empty",
      Self::NoNumber => "no_number",
      Self::NoUnit => "no_unit",
      Self::AmbiguousSeparator => "ambiguous_separator",
      Self::BadNumber => "bad_number",
      Self::Negative => "negative",
      Self::BackwardsRange => "backwards_range",
      Self::MixedUnits => "mixed_units",
      Self::Multiple => "multiple",
      Self::Qualified => "qualified",
      Self::UnreadableUnit => "unreadable_unit",
      Self::TooLarge => "too_large",
    }
  }
}

/// What a text starts with when its number is qualified. Symbols only: a list of words (`up to`, `about`) would
/// never be complete, and a word in front is [`NotAnAmount::NoNumber`] like any other.
const QUALIFIERS: [char; 7] = ['<', '>', '≤', '≥', '~', '≈', '±'];

/// What joins the two ends of a range, besides the word `to`: a hyphen, an en dash, an em dash, a minus sign.
const DASHES: [char; 4] = ['-', '\u{2013}', '\u{2014}', '\u{2212}'];

/// The fractions that are one character, each as its numerator and denominator.
const VULGAR: [(char, f64, f64); 18] = [
  ('½', 1.0, 2.0),
  ('⅓', 1.0, 3.0),
  ('⅔', 2.0, 3.0),
  ('¼', 1.0, 4.0),
  ('¾', 3.0, 4.0),
  ('⅕', 1.0, 5.0),
  ('⅖', 2.0, 5.0),
  ('⅗', 3.0, 5.0),
  ('⅘', 4.0, 5.0),
  ('⅙', 1.0, 6.0),
  ('⅚', 5.0, 6.0),
  ('⅐', 1.0, 7.0),
  ('⅛', 1.0, 8.0),
  ('⅜', 3.0, 8.0),
  ('⅝', 5.0, 8.0),
  ('⅞', 7.0, 8.0),
  ('⅑', 1.0, 9.0),
  ('⅒', 1.0, 10.0),
];

/// Reads a string that is an amount, the whole of it.
///
/// Whitespace around the text and runs of it inside are not part of it. The unit comes back as it was written,
/// and what comes back is a quantity a record accepts ([`Quantity::check`] passes).
///
/// # Errors
///
/// [`NotAnAmount`] with the reason, for anything that is not exactly an amount. Never a guess.
pub fn parse(text: &str) -> Result<Quantity, NotAnAmount> {
  let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
  let Some(first) = text.chars().next() else {
    return Err(NotAnAmount::Empty);
  };
  if QUALIFIERS.contains(&first) {
    return Err(NotAnAmount::Qualified);
  }
  if let Some(rest) = text.strip_prefix(['-', '\u{2212}'])
    && starts_number(rest.trim_start())
  {
    return Err(NotAnAmount::Negative);
  }
  if !starts_number(&text) {
    return Err(NotAnAmount::NoNumber);
  }

  let (low, rest) = number(&text)?;
  let (first_unit, high, rest) = match range(rest) {
    Some((unit, second)) => {
      let (high, rest) = number(second)?;
      (unit.trim(), high, rest)
    }
    None => ("", low, rest),
  };
  let unit = rest.trim();
  if unit.is_empty() {
    return Err(NotAnAmount::NoUnit);
  }
  if multiplies(first_unit) || multiplies(unit) {
    return Err(NotAnAmount::Multiple);
  }
  if !(readable(unit) && readable(first_unit)) {
    return Err(NotAnAmount::UnreadableUnit);
  }
  // The first end of a range may say its unit too (`18% - 22%`), if it is the unit the last end says.
  if !(first_unit.is_empty() || same_unit(first_unit, unit)) {
    return Err(NotAnAmount::MixedUnits);
  }
  if low > high {
    return Err(NotAnAmount::BackwardsRange);
  }
  let quantity = Quantity::range(low, high, unit);
  // Negative, backwards, and blank are refused above with a reason of their own, so what `check` still refuses
  // is a number that is not finite, in its own unit or in the one it is compared in.
  if quantity.check().is_err() {
    return Err(NotAnAmount::TooLarge);
  }
  Ok(quantity)
}

/// Whether two spellings are one unit to the table: `g` and `grams`, and not `g` and `mg`.
fn same_unit(left: &str, right: &str) -> bool {
  let ((left_dimension, left_factor), (right_dimension, right_factor)) = (quantity::unit(left), quantity::unit(right));
  left_dimension == right_dimension && left_factor.to_bits() == right_factor.to_bits()
}

/// Whether `text` starts the way a number does: a digit, a point and a digit, or a one-character fraction.
fn starts_number(text: &str) -> bool {
  let mut chars = text.chars();
  match chars.next() {
    Some('.') => chars.next().is_some_and(|next| next.is_ascii_digit()),
    Some(first) => first.is_ascii_digit() || vulgar_value(first).is_some(),
    None => false,
  }
}

/// The digits `text` starts with, and what follows them.
fn digits(text: &str) -> (&str, &str) {
  let end = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
  text.split_at_checked(end).unwrap_or((text, ""))
}

/// The value of a one-character fraction.
fn vulgar_value(c: char) -> Option<f64> {
  VULGAR
    .iter()
    .find(|(fraction, ..)| *fraction == c)
    .map(|(_, over, under)| over / under)
}

/// A one-character fraction at the start of `text`, and what follows it.
fn vulgar(text: &str) -> Option<(f64, &str)> {
  let mut chars = text.chars();
  let value = vulgar_value(chars.next()?)?;
  Some((value, chars.as_str()))
}

/// `1/2` at the start of `text`: the two numbers, and what follows.
fn ratio(text: &str) -> Option<(f64, f64, &str)> {
  let (over, rest) = digits(text);
  let (under, rest) = digits(rest.strip_prefix('/')?);
  // No digits on either side is not a number, which is what reading nothing as one says.
  Some((over.parse().ok()?, under.parse().ok()?, rest))
}

/// The number `text` starts with, which [`starts_number`] has said is there, and what follows it.
fn number(text: &str) -> Result<(f64, &str), NotAnAmount> {
  let (value, rest) = if let Some(found) = vulgar(text) {
    found
  } else {
    let (value, whole, rest) = decimal(text)?;
    if whole { fraction(value, rest)? } else { (value, rest) }
  };
  // A number ends where something that is not part of a number begins. More of one is a number written wrongly,
  // and is never the start of a unit.
  let mut after = rest.chars();
  match after.next() {
    Some(',') => Err(NotAnAmount::AmbiguousSeparator),
    Some(next) if next == '.' || next.is_ascii_digit() || vulgar_value(next).is_some() => Err(NotAnAmount::BadNumber),
    Some('/') if after.next().is_some_and(|next| next.is_ascii_digit()) => Err(NotAnAmount::BadNumber),
    _ => Ok((value, rest)),
  }
}

/// A number in digits at the start of `text`: its value, whether it is a whole number, and what follows it.
///
/// A comma groups thousands and does nothing else: the first group is one to three digits and does not start
/// with a zero, and every later group is exactly three. So `0,500 g` is not five hundred grams.
fn decimal(text: &str) -> Result<(f64, bool, &str), NotAnAmount> {
  let (lead, mut rest) = digits(text);
  let mut written = lead.to_owned();
  if rest.starts_with(',') && (!(1..=3).contains(&lead.len()) || lead.starts_with('0')) {
    return Err(NotAnAmount::AmbiguousSeparator);
  }
  while let Some(after) = rest.strip_prefix(',') {
    let (group, beyond) = digits(after);
    if group.len() != 3 {
      return Err(NotAnAmount::AmbiguousSeparator);
    }
    written.push_str(group);
    rest = beyond;
  }
  let mut whole = true;
  if let Some(after) = rest.strip_prefix('.') {
    let (decimals, beyond) = digits(after);
    if !decimals.is_empty() {
      written.push('.');
      written.push_str(decimals);
      rest = beyond;
      whole = false;
    }
  }
  // The nearest number to what was written: `0.1` is the `0.1` every other reader of it gets.
  let value = written.parse::<f64>().map_err(|_| NotAnAmount::BadNumber)?;
  Ok((value, whole, rest))
}

/// What a whole number turns out to be the start of: a fraction (`1/8`), a mixed number (`1 1/2`, `1-1/2`,
/// `1½`), or only itself.
fn fraction(whole: f64, rest: &str) -> Result<(f64, &str), NotAnAmount> {
  if let Some(after) = rest.strip_prefix('/') {
    let (under, beyond) = digits(after);
    // A slash and no number is not this number's: it is left for the unit, which refuses it.
    let Ok(under) = under.parse::<f64>() else {
      return Ok((whole, rest));
    };
    if whole <= 0.0 || under <= 0.0 {
      return Err(NotAnAmount::BadNumber);
    }
    return Ok((whole / under, beyond));
  }
  let spaced = rest.strip_prefix(' ');
  if let Some((part, beyond)) = vulgar(spaced.unwrap_or(rest)) {
    return Ok((whole + part, beyond));
  }
  if let Some((over, under, beyond)) = spaced.and_then(ratio) {
    if over <= 0.0 || over >= under {
      return Err(NotAnAmount::BadNumber);
    }
    return Ok((whole + over / under, beyond));
  }
  // `1-1/2` is one and a half. A whole number of at least 1 and a proper fraction are exactly the pairs whose
  // reading as a range would run backwards, so no string has two readings: `0-1/2` is a range.
  if let Some((over, under, beyond)) = rest.strip_prefix('-').and_then(ratio)
    && whole >= 1.0
    && over > 0.0
    && over < under
  {
    return Ok((whole + over / under, beyond));
  }
  Ok((whole, rest))
}

/// Where a range's second number starts in what follows the first, and the unit the first end was given, which
/// is usually none. `None` when what follows is not a range.
fn range(rest: &str) -> Option<(&str, &str)> {
  let mut chars = rest.chars();
  loop {
    let here = chars.as_str();
    if let Some(second) = separator(here).filter(|second| starts_number(second)) {
      return Some((rest.strip_suffix(here)?, second));
    }
    chars.next()?;
  }
}

/// What follows a range's separator, if `here` starts with one: a dash with or without a space on either side,
/// or the word `to` with one on both.
fn separator(here: &str) -> Option<&str> {
  let spaced = here.strip_prefix(' ');
  if let Some(after) = spaced.unwrap_or(here).strip_prefix(DASHES) {
    return Some(after.strip_prefix(' ').unwrap_or(after));
  }
  let (word, after) = spaced?.split_at_checked(3)?;
  word.eq_ignore_ascii_case("to ").then_some(after)
}

/// Whether what follows a number says "times another number": `x 10 mg`, `mg x 10`, `× 0.5 g`. The letter counts
/// only where it is not the end of a word (`max`).
fn multiplies(unit: &str) -> bool {
  let mut chars = unit.chars();
  let mut previous: Option<char> = None;
  while let Some(c) = chars.next() {
    let sign = matches!(c, '×' | '*') || (matches!(c, 'x' | 'X') && !previous.is_some_and(char::is_alphabetic));
    if sign && starts_number(chars.as_str().trim_start()) {
      return true;
    }
    previous = Some(c);
  }
  false
}

/// Whether `unit` is made of what a unit is made of: letters, spaces, `%`, `.`, `/`, a hyphen between two
/// letters, and digits only as the amount of a denominator (`mg/100 g`, `mg per 2 servings`). It starts with a
/// letter or `%`.
///
/// Not whether the unit is known: `pack` is readable, and [`quantity::unit`] says what it measures.
fn readable(unit: &str) -> bool {
  let mut chars = unit.chars().peekable();
  let mut seen = String::new();
  let mut in_amount = false;
  while let Some(c) = chars.next() {
    let fine = if c.is_ascii_digit() {
      in_amount || follows_per(&seen)
    } else if c == '-' {
      seen.ends_with(char::is_alphabetic) && chars.peek().is_some_and(|next| next.is_alphabetic())
    } else if seen.is_empty() {
      c.is_alphabetic() || c == '%'
    } else {
      c.is_alphabetic() || matches!(c, ' ' | '%' | '.' | '/')
    };
    if !fine {
      return false;
    }
    in_amount = c.is_ascii_digit() || (c == '.' && in_amount);
    seen.push(c);
  }
  true
}

/// Whether a number may come next in a unit: after a `/`, or after the word `per`.
fn follows_per(seen: &str) -> bool {
  let before = seen.trim_end();
  before.ends_with('/')
    || before
      .to_lowercase()
      .strip_suffix("per")
      .is_some_and(|head| !head.ends_with(char::is_alphabetic))
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeSet;

  use serde::Deserialize;

  use super::*;

  /// One line of `data/amounts.cases.jsonl`.
  #[derive(Debug, Deserialize)]
  #[serde(deny_unknown_fields)]
  struct Case {
    text: String,
    min: Option<f64>,
    max: Option<f64>,
    unit: Option<String>,
    refused: Option<String>,
  }

  impl Case {
    fn expected(&self) -> Result<Quantity, &str> {
      match (self.min, self.max, &self.unit, &self.refused) {
        (Some(min), Some(max), Some(unit), None) => Ok(Quantity::range(min, max, unit.as_str())),
        (None, None, None, Some(code)) => Err(code),
        _ => panic!("a case is `min`, `max`, and `unit`, or `refused`: {self:?}"),
      }
    }
  }

  fn cases() -> Vec<Case> {
    include_str!("../../../data/amounts.cases.jsonl")
      .lines()
      .filter(|line| !(line.trim().is_empty() || line.starts_with('#')))
      .map(|line| serde_json::from_str(line).unwrap_or_else(|error| panic!("{line}: {error}")))
      .collect()
  }

  #[allow(
    clippy::unnecessary_wraps,
    reason = "shaped like what `parse` returns, so a test compares the two whole"
  )]
  fn single(value: f64, unit: &str) -> Result<Quantity, NotAnAmount> {
    Ok(Quantity::single(value, unit))
  }

  #[allow(
    clippy::unnecessary_wraps,
    reason = "shaped like what `parse` returns, so a test compares the two whole"
  )]
  fn range_of(min: f64, max: f64, unit: &str) -> Result<Quantity, NotAnAmount> {
    Ok(Quantity::range(min, max, unit))
  }

  #[test]
  fn every_case_in_the_file_is_read_as_it_says_or_refused_as_it_says() {
    let cases = cases();
    assert!(cases.len() > 150, "{} cases", cases.len());
    let mut failures = Vec::new();
    for case in &cases {
      let read = parse(&case.text);
      let agrees = match (&read, case.expected()) {
        (Ok(quantity), Ok(expected)) => {
          quantity.min.to_bits() == expected.min.to_bits()
            && quantity.max.to_bits() == expected.max.to_bits()
            && quantity.unit == expected.unit
        }
        (Err(problem), Err(code)) => problem.code() == code,
        _ => false,
      };
      if !agrees {
        failures.push(format!(
          "{:?}: read as {read:?}, the file says {:?}",
          case.text,
          case.expected()
        ));
      }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
  }

  /// A code nobody wrote a case for is a refusal nobody checked.
  #[test]
  fn the_file_has_a_case_for_every_refusal_and_no_text_twice() {
    let cases = cases();
    let refused: BTreeSet<&str> = cases.iter().filter_map(|case| case.refused.as_deref()).collect();
    let codes: BTreeSet<&str> = NotAnAmount::ALL.iter().map(|problem| problem.code()).collect();
    assert_eq!(refused, codes);
    let mut texts = BTreeSet::new();
    for case in &cases {
      assert!(texts.insert(case.text.as_str()), "{:?} is in the file twice", case.text);
    }
  }

  /// Other tools count these names. One is added, and never renamed.
  #[test]
  fn the_codes_are_the_ones_that_were_published() {
    let codes: Vec<&str> = NotAnAmount::ALL.iter().map(|problem| problem.code()).collect();
    assert_eq!(
      codes,
      [
        "empty",
        "no_number",
        "no_unit",
        "ambiguous_separator",
        "bad_number",
        "negative",
        "backwards_range",
        "mixed_units",
        "multiple",
        "qualified",
        "unreadable_unit",
        "too_large",
      ]
    );
    for problem in NotAnAmount::ALL {
      assert!(!problem.to_string().is_empty());
    }
  }

  #[test]
  fn a_number_and_a_unit_is_an_amount_with_or_without_a_space_between() {
    assert_eq!(parse("3.5 g"), single(3.5, "g"));
    assert_eq!(parse("3.5g"), single(3.5, "g"));
    assert_eq!(parse(" \t3.5 \n g "), single(3.5, "g"));
    assert_eq!(parse(".5 g"), single(0.5, "g"));
    assert_eq!(parse("0 mg"), single(0.0, "mg"));
    // The unit is kept as it was written, and is still the unit the table knows.
    let ounces = parse("12  Fl.   Oz.").unwrap();
    assert_eq!(ounces.unit, "Fl. Oz.");
    assert_eq!(ounces.dimension(), quantity::Dimension::Volume);
    // `0.1` is the nearest number to a tenth, the one every other reader of `0.1` gets.
    assert_eq!(parse("0.1 g").unwrap().min.to_bits(), 0.1_f64.to_bits());
  }

  #[test]
  fn a_comma_groups_thousands_and_does_nothing_else() {
    assert_eq!(parse("1,000 mg"), single(1000.0, "mg"));
    assert_eq!(parse("12,500 mg"), single(12_500.0, "mg"));
    assert_eq!(parse("999,999 mg"), single(999_999.0, "mg"));
    assert_eq!(parse("1,000,000.5 mcg"), single(1_000_000.5, "mcg"));
    for text in [
      "3,5 g",
      "1,00 mg",
      "1,0000 mg",
      "0,500 g",
      "1000,000 mg",
      "1,000,00 mg",
      "1, g",
      "3.5,2 g",
    ] {
      assert_eq!(parse(text), Err(NotAnAmount::AmbiguousSeparator), "{text}");
    }
  }

  #[test]
  fn a_fraction_is_two_whole_numbers_above_zero() {
    assert_eq!(parse("1/8 oz"), single(0.125, "oz"));
    assert_eq!(parse("3/2 oz"), single(1.5, "oz"));
    assert_eq!(parse("1/3 oz"), single(1.0 / 3.0, "oz"));
    assert_eq!(parse("⅞ oz"), single(0.875, "oz"));
    assert_eq!(parse("⅒ g"), single(0.1, "g"));
    for text in ["1/0 oz", "0/8 oz", "0/0 oz", "1.5/8 oz", "1/8/2 oz", "1/8.5 oz"] {
      assert_eq!(parse(text), Err(NotAnAmount::BadNumber), "{text}");
    }
    // A slash with no number after it belongs to no number.
    assert_eq!(parse("5/serving"), Err(NotAnAmount::UnreadableUnit));
  }

  #[test]
  fn a_whole_number_and_a_proper_fraction_are_a_mixed_number() {
    assert_eq!(parse("1 1/2 oz"), single(1.5, "oz"));
    assert_eq!(parse("1-1/2 oz"), single(1.5, "oz"));
    assert_eq!(parse("1½ oz"), single(1.5, "oz"));
    assert_eq!(parse("1 ½ oz"), single(1.5, "oz"));
    assert_eq!(parse("2 3/4 lb"), single(2.75, "lb"));
    assert_eq!(parse("0 1/2 oz"), single(0.5, "oz"));
    for text in ["1 3/2 oz", "1 2/2 oz", "1 0/2 oz", "1 1/0 oz", "1½½ oz", "1.5½ oz"] {
      assert_eq!(parse(text), Err(NotAnAmount::BadNumber), "{text}");
    }
  }

  /// `1-1/2` is a mixed number exactly when reading it as a range would run backwards, so neither reading can be
  /// taken for the other.
  #[test]
  fn a_hyphen_makes_a_mixed_number_only_where_a_range_would_run_backwards() {
    assert_eq!(parse("1-1/2 oz"), single(1.5, "oz"));
    assert_eq!(parse("0-1/2 oz"), range_of(0.0, 0.5, "oz"));
    assert_eq!(parse("1-3/2 oz"), range_of(1.0, 1.5, "oz"));
    assert_eq!(parse("1-2/2 oz"), range_of(1.0, 1.0, "oz"));
    assert_eq!(parse("1 - 1/2 oz"), Err(NotAnAmount::BackwardsRange));
    assert_eq!(parse("1/2-3/4 oz"), range_of(0.5, 0.75, "oz"));
    assert_eq!(parse("1-1/2-2 oz"), range_of(1.5, 2.0, "oz"));
    assert_eq!(parse("1-0/2 oz"), Err(NotAnAmount::BadNumber));
  }

  #[test]
  fn two_numbers_joined_by_a_dash_or_the_word_to_are_a_range() {
    for text in [
      "18-22 %",
      "18 - 22 %",
      "18 -22 %",
      "18- 22 %",
      "18\u{2013}22 %",
      "18 \u{2014} 22 %",
      "18\u{2212}22 %",
      "18 to 22 %",
      "18 To 22 %",
      "18% - 22%",
      "18 %-22 %",
      "18 percent to 22 %",
    ] {
      assert_eq!(parse(text), range_of(18.0, 22.0, "%"), "{text}");
    }
    assert_eq!(parse("22-22 %"), single(22.0, "%"));
    assert_eq!(parse("22-18 %"), Err(NotAnAmount::BackwardsRange));
    assert_eq!(parse("22.5-22 %"), Err(NotAnAmount::BackwardsRange));
    // `to` is a word: it joins nothing from inside another, or without a space on both sides.
    assert_eq!(parse("2 photo"), single(2.0, "photo"));
    assert_eq!(parse("18to 22 %"), Err(NotAnAmount::UnreadableUnit));
    assert_eq!(parse("18 to22 %"), Err(NotAnAmount::UnreadableUnit));
    assert_eq!(parse("18 tot 22 %"), Err(NotAnAmount::UnreadableUnit));
  }

  #[test]
  fn both_ends_of_a_range_may_say_the_unit_if_it_is_one_unit() {
    assert_eq!(parse("18 g - 22 grams"), range_of(18.0, 22.0, "grams"));
    assert_eq!(parse("1 OZ to 2 oz"), range_of(1.0, 2.0, "oz"));
    assert_eq!(parse("1 pack - 2 Pack"), range_of(1.0, 2.0, "Pack"));
    assert_eq!(parse("500 mg - 1 g"), Err(NotAnAmount::MixedUnits));
    assert_eq!(parse("1 g - 2 ml"), Err(NotAnAmount::MixedUnits));
    assert_eq!(parse("1 pack - 2 packs"), Err(NotAnAmount::MixedUnits));
    // The last end has to say it.
    assert_eq!(parse("18% - 22"), Err(NotAnAmount::NoUnit));
    assert_eq!(parse("18% (lab) - 22%"), Err(NotAnAmount::UnreadableUnit));
  }

  #[test]
  fn a_text_that_does_not_start_with_a_number_is_not_an_amount() {
    assert_eq!(parse(""), Err(NotAnAmount::Empty));
    assert_eq!(parse(" \n\t "), Err(NotAnAmount::Empty));
    for text in [
      "g",
      "THC 22%",
      "eighth",
      "up to 22 %",
      ". g",
      ".",
      "-",
      "- g",
      "+5 g",
      "x 10 mg",
    ] {
      assert_eq!(parse(text), Err(NotAnAmount::NoNumber), "{text}");
    }
    for text in ["3.5", "1/8", "½", "18-22", "1,000", "3.5 "] {
      assert_eq!(parse(text), Err(NotAnAmount::NoUnit), "{text}");
    }
    for text in ["-1 g", "- 1 g", "\u{2212}1 g", "-.5 g", "-½ oz"] {
      assert_eq!(parse(text), Err(NotAnAmount::Negative), "{text}");
    }
    for text in [
      "<1 %",
      ">20 %",
      "≤ 1 %",
      "≥ 20 %",
      "~3.5 g",
      "≈ 3.5 g",
      "± 2 %",
      "< LOQ",
    ] {
      assert_eq!(parse(text), Err(NotAnAmount::Qualified), "{text}");
    }
    for text in ["1.2.3 g", "5. g", "5.g", "½5 oz"] {
      assert_eq!(parse(text), Err(NotAnAmount::BadNumber), "{text}");
    }
  }

  /// Ten pieces of 10 mg is 100 mg a package or 10 mg an item, and the text does not say which was asked for.
  #[test]
  fn one_amount_times_another_is_refused_and_a_word_ending_in_x_is_not_one() {
    for text in [
      "10 x 10 mg",
      "10x10mg",
      "10 X 10 mg",
      "2 × 0.5 g",
      "2 * 5 g",
      "10 mg x 10",
      "10 mg x10",
      "10mg×10",
      "2 x .5 g",
      "2 x ½ g",
      "2 x 5-10 mg",
    ] {
      assert_eq!(parse(text), Err(NotAnAmount::Multiple), "{text}");
    }
    assert_eq!(parse("10 xl"), single(10.0, "xl"));
    assert_eq!(parse("10 x"), single(10.0, "x"));
    assert_eq!(parse("5 max2"), Err(NotAnAmount::UnreadableUnit));
    assert_eq!(parse("5 mg * g"), Err(NotAnAmount::UnreadableUnit));
  }

  #[test]
  fn a_unit_is_letters_and_a_few_signs_and_starts_with_a_letter_or_a_percent_sign() {
    for (text, unit) in [
      ("2 pack", "pack"),
      ("3.5 g net", "g net"),
      ("2 pre-rolls", "pre-rolls"),
      ("5 µg", "µg"),
      ("22 % thc", "% thc"),
      ("10 mg/ml", "mg/ml"),
      ("10 mg / ml", "mg / ml"),
      ("5 mg/", "mg/"),
    ] {
      assert_eq!(
        parse(text),
        single(text.split(' ').next().unwrap().parse().unwrap(), unit),
        "{text}"
      );
    }
    for text in [
      "3.5 g (approx)",
      "3.5 g (1/8 oz)",
      "20 %+",
      "3.5 g, 1/8 oz",
      "3.5 g!",
      "5 /serving",
      "2 .5 g",
      "2 -pack",
      "2 pack-",
      "2 pre--rolls",
      "2 pre - rolls",
      "2 %-thc",
      "2 a-% b",
      "1e3 mg",
      "5 mg 2",
      "1 - 2 - 3 g",
    ] {
      assert_eq!(parse(text), Err(NotAnAmount::UnreadableUnit), "{text}");
    }
  }

  /// The one place a unit holds a number: `25 mg/5 ml` is a strength, and `5 mg 2` is nothing.
  #[test]
  fn a_unit_holds_a_number_only_as_the_amount_of_what_it_is_per() {
    for (text, unit) in [
      ("25 mg/5 ml", "mg/5 ml"),
      ("25 mg / 5 ml", "mg / 5 ml"),
      ("25 mg/ 5 ml", "mg/ 5 ml"),
      ("2 g/100 g", "g/100 g"),
      ("25 mg/2.5 ml", "mg/2.5 ml"),
      ("5 mg per 2 servings", "mg per 2 servings"),
      ("5 mg PER 2 servings", "mg PER 2 servings"),
      ("5 mg per2 servings", "mg per2 servings"),
    ] {
      assert_eq!(
        parse(text),
        single(text.split(' ').next().unwrap().parse().unwrap(), unit),
        "{text}"
      );
    }
    assert_eq!(parse("5-10 mg/5 ml"), range_of(5.0, 10.0, "mg/5 ml"));
    assert_eq!(parse("5 mg/5 ml - 10 mg/5 ml"), range_of(5.0, 10.0, "mg/5 ml"));
    for text in [
      "5 mg/ml 2",
      "5 mg super 2 ml",
      "5 mg/5.5 ml5",
      "5 mg/5 ml 5",
      "5 mg/5. 5 ml",
      "5 mg pe 2 ml",
    ] {
      assert_eq!(parse(text), Err(NotAnAmount::UnreadableUnit), "{text}");
    }
  }

  #[test]
  fn an_amount_too_large_to_compare_is_refused_and_one_that_is_only_large_is_not() {
    let digits = |zeros: usize| format!("1{}", "0".repeat(zeros));
    assert_eq!(parse(&format!("{} g", digits(400))), Err(NotAnAmount::TooLarge));
    // A number of kilograms, and not a number of the milligrams it is compared in.
    assert_eq!(parse(&format!("{} kg", digits(305))), Err(NotAnAmount::TooLarge));
    assert_eq!(parse(&format!("1-{} kg", digits(305))), Err(NotAnAmount::TooLarge));
    assert_eq!(parse(&format!("{} mg", digits(305))), single(1e305, "mg"));
  }

  #[test]
  fn what_is_read_is_a_quantity_a_record_accepts() {
    for case in cases() {
      if let Ok(quantity) = parse(&case.text) {
        assert!(quantity.check().is_ok(), "{:?}", case.text);
        assert_eq!(quantity, quantity.trimmed(), "{:?}", case.text);
        assert!(!quantity.unit.contains("  "), "{:?}", case.text);
      }
    }
  }
}
