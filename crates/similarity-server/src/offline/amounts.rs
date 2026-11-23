//! A survey of amounts written as text: how many of a column's values are amounts, in which units, and what the
//! rest are refused for.
//!
//! Which unit strings a catalog really holds is a question the repository cannot answer, since no real data is in
//! it (docs/decisions.md, D16). This is how it is answered where the data is: a column of texts, one to a line,
//! goes in, and counts come out. The counts are what the unit table and the parser are extended from, and nobody
//! has to read the column.
//!
//! Every text is read by `similarity_core::amount::parse`, the function behind `Quantities.Parse`, so the survey
//! says what the service would. Nothing here reads an amount another way, and every number in the report and in
//! the events is a field of [`Survey`].
//!
//! What it writes is counts, the spellings of units, and the codes of refusals. Only when asked for
//! ([`survey`]'s `examples`) does it also write refused texts themselves, which are values of a quantity column
//! and no part of a product's name or description, and are still the data's owner's.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use similarity_core::amount;
use similarity_core::quantity::{self, Dimension};
use similarity_log::RESULT;

/// What a column of texts holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Survey {
  /// How many texts there were: every line, a blank one included.
  pub texts: usize,
  /// How many of them are amounts.
  pub amounts: usize,
  /// How many are not. With [`Self::amounts`], all of them.
  pub refused: usize,
  /// How many amounts are in a unit that is not converted, and so are comparable only with that unit's own
  /// spelling.
  pub unconverted: usize,
  /// Every unit an amount was in, the most used first, and by spelling among equals.
  pub units: Vec<UnitCount>,
  /// Every reason a text was refused for, the most frequent first, and by code among equals.
  pub problems: Vec<ProblemCount>,
}

/// One unit, as it is matched (lower case, single spaces), and how many amounts were in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitCount {
  /// The unit's spelling. Two spellings of one unit (`g`, `grams`) are counted apart: the spelling is the finding.
  pub unit: String,
  /// What it measures: `mass`, `volume`, `fraction`, or `other` for a unit that is not converted.
  pub dimension: &'static str,
  /// How many amounts were in it.
  pub amounts: usize,
}

/// One reason texts were refused for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProblemCount {
  /// The reason's code (`NotAnAmount::code`).
  pub code: &'static str,
  /// How many texts were refused for it.
  pub texts: usize,
  /// The refused texts themselves, the most frequent first: as many as were asked for, and none unless asked.
  pub examples: Vec<Example>,
}

/// One refused text, with single spaces, and how many lines held it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
  /// The text.
  pub text: String,
  /// How many times it occurred.
  pub texts: usize,
}

/// The name of a dimension in the report and the events. Exhaustive on purpose, and an interface: add, never
/// rename.
fn dimension_code(dimension: &Dimension) -> &'static str {
  match dimension {
    Dimension::Mass => "mass",
    Dimension::Volume => "volume",
    Dimension::Fraction => "fraction",
    Dimension::Other(_) => "other",
  }
}

/// Entries of a tally, the largest count first, and in the order of their keys among equals.
fn ranked<K: Ord, V>(tally: BTreeMap<K, V>, count: impl Fn(&V) -> usize) -> Vec<(K, V)> {
  let mut entries: Vec<(K, V)> = tally.into_iter().collect();
  // Stable, and the map came in order of key.
  entries.sort_by_key(|(_, value)| std::cmp::Reverse(count(value)));
  entries
}

/// Surveys `column`: one text to a line, each read as the whole of an amount.
///
/// A blank line is a text (an empty value in a column is a finding), and is refused as `empty`. `examples` is how
/// many of the refused texts to keep for each reason, the most frequent first; with 0, no text is kept at all.
pub fn survey(column: &str, examples: usize) -> Survey {
  // A file written by a spreadsheet may start with a byte-order mark, which is no part of the first value.
  let column = column.strip_prefix('\u{feff}').unwrap_or(column);
  let mut texts = 0_usize;
  let mut units: BTreeMap<String, (&'static str, usize)> = BTreeMap::new();
  let mut problems: BTreeMap<&'static str, (usize, BTreeMap<String, usize>)> = BTreeMap::new();
  for line in column.lines() {
    texts = texts.saturating_add(1);
    match amount::parse(line) {
      Ok(quantity) => {
        let entry = units
          .entry(quantity::normalized(&quantity.unit))
          .or_insert((dimension_code(&quantity.dimension()), 0));
        entry.1 = entry.1.saturating_add(1);
      }
      Err(refused) => {
        let entry = problems.entry(refused.code()).or_default();
        entry.0 = entry.0.saturating_add(1);
        if examples > 0 {
          let text = line.split_whitespace().collect::<Vec<_>>().join(" ");
          let seen = entry.1.entry(text).or_default();
          *seen = seen.saturating_add(1);
        }
      }
    }
  }

  let sum = |counts: &mut dyn Iterator<Item = usize>| counts.fold(0_usize, usize::saturating_add);
  let amounts = sum(&mut units.values().map(|(_, count)| *count));
  let unconverted = sum(
    &mut units
      .values()
      .filter(|(dimension, _)| *dimension == dimension_code(&Dimension::Other(String::new())))
      .map(|(_, count)| *count),
  );
  let refused = sum(&mut problems.values().map(|(count, _)| *count));
  Survey {
    texts,
    amounts,
    refused,
    unconverted,
    units: ranked(units, |(_, count)| *count)
      .into_iter()
      .map(|(unit, (dimension, amounts))| UnitCount {
        unit,
        dimension,
        amounts,
      })
      .collect(),
    problems: ranked(problems, |(count, _)| *count)
      .into_iter()
      .map(|(code, (texts, seen))| ProblemCount {
        code,
        texts,
        examples: ranked(seen, |count| *count)
          .into_iter()
          .take(examples)
          .map(|(text, texts)| Example { text, texts })
          .collect(),
      })
      .collect(),
  }
}

/// The report a person reads. It lays out [`Survey`] and computes nothing.
pub fn render(survey: &Survey) -> String {
  let mut out = String::new();
  // Writing to a `String` cannot fail.
  let _ = writeln!(
    out,
    "texts    {}\namounts  {}\nrefused  {}",
    survey.texts, survey.amounts, survey.refused
  );
  if !survey.units.is_empty() {
    let _ = writeln!(out, "\namounts  measures       unit");
    for unit in &survey.units {
      let measures = if unit.dimension == "other" {
        "not converted"
      } else {
        unit.dimension
      };
      let _ = writeln!(out, "{:>7}  {measures:<13}  {}", unit.amounts, unit.unit);
    }
  }
  if survey.unconverted > 0 {
    let _ = writeln!(
      out,
      "\nAmounts in a unit that is not converted: {}. Such a unit is comparable only with its own spelling:\n\
       look among those units for misspellings of one that is, and for units the table should learn.",
      survey.unconverted
    );
  }
  if !survey.problems.is_empty() {
    let _ = writeln!(out, "\n  texts  refused as");
    for problem in &survey.problems {
      let _ = writeln!(out, "{:>7}  {}", problem.texts, problem.code);
      for example in &problem.examples {
        let _ = writeln!(out, "         {:>7}  {}", example.texts, example.text);
      }
    }
  }
  out
}

/// Writes a survey as events: one `amount_survey` line, then one `amount_unit` line for each unit, one
/// `amount_problem` line for each reason, and one `amount_example` line for each refused text that was kept.
///
/// The names of the events and their fields are an interface, like the calibration ones (D17): add, never rename.
pub fn emit(survey: &Survey) {
  tracing::info!(
    target: RESULT,
    texts = survey.texts,
    amounts = survey.amounts,
    refused = survey.refused,
    units = survey.units.len(),
    unconverted = survey.unconverted,
    "amount_survey"
  );
  for unit in &survey.units {
    tracing::info!(
      target: RESULT,
      unit = unit.unit.as_str(),
      dimension = unit.dimension,
      amounts = unit.amounts,
      "amount_unit"
    );
  }
  for problem in &survey.problems {
    tracing::info!(target: RESULT, code = problem.code, texts = problem.texts, "amount_problem");
    for example in &problem.examples {
      tracing::info!(
        target: RESULT,
        code = problem.code,
        text = example.text.as_str(),
        texts = example.texts,
        "amount_example"
      );
    }
  }
}

#[cfg(test)]
mod tests {
  use serde_json::Value;

  use super::*;

  const COLUMN: &str = "3.5 g\n3.5g\n1/8 oz\n28 Grams\n100 MG\n2 pack\n2  Pack\n3.5 gr\n18-22%\n\n3,5 g\n3,5  g\n1,00 g\n\
                        n/a\n<1 %\n10 x 10 mg\n";

  fn unit(unit: &str, dimension: &'static str, amounts: usize) -> UnitCount {
    UnitCount {
      unit: unit.to_owned(),
      dimension,
      amounts,
    }
  }

  fn problem(code: &'static str, texts: usize, examples: &[(&str, usize)]) -> ProblemCount {
    ProblemCount {
      code,
      texts,
      examples: examples
        .iter()
        .map(|(text, texts)| Example {
          text: (*text).to_owned(),
          texts: *texts,
        })
        .collect(),
    }
  }

  #[test]
  fn a_column_is_counted_by_unit_and_by_the_reason_a_text_was_refused() {
    let made = survey(COLUMN, 0);
    assert_eq!((made.texts, made.amounts, made.refused), (16, 9, 7));
    assert_eq!(made.unconverted, 3);
    // The most used first, and by spelling among equals. `g` and `grams` are one unit and two findings; `pack`
    // in another case and spacing is one spelling.
    assert_eq!(
      made.units,
      [
        unit("g", "mass", 2),
        unit("pack", "other", 2),
        unit("%", "fraction", 1),
        unit("gr", "other", 1),
        unit("grams", "mass", 1),
        unit("mg", "mass", 1),
        unit("oz", "mass", 1),
      ]
    );
    assert_eq!(
      made.problems,
      [
        problem("ambiguous_separator", 3, &[]),
        problem("empty", 1, &[]),
        problem("multiple", 1, &[]),
        problem("no_number", 1, &[]),
        problem("qualified", 1, &[]),
      ]
    );
  }

  #[test]
  fn everything_looked_at_is_an_amount_or_refused_and_the_counts_say_so() {
    for column in [COLUMN, "", "\n", "3.5 g", "nothing\nhere\n", "1 g\r\n2 g\r\n"] {
      let made = survey(column, 3);
      assert_eq!(made.texts, column.lines().count(), "{column:?}");
      assert_eq!(made.amounts + made.refused, made.texts, "{column:?}");
      assert_eq!(made.units.iter().map(|unit| unit.amounts).sum::<usize>(), made.amounts);
      assert_eq!(
        made.problems.iter().map(|problem| problem.texts).sum::<usize>(),
        made.refused
      );
      assert!(made.unconverted <= made.amounts);
    }
    assert_eq!(survey("", 0).texts, 0);
    // A blank line is a value, and an empty one.
    assert_eq!(survey("\n", 0).problems, [problem("empty", 1, &[])]);
    assert_eq!(survey("1 g\r\n2 g\r\n", 0).units, [unit("g", "mass", 2)]);
  }

  #[test]
  fn refused_texts_are_kept_only_when_asked_for_and_the_most_frequent_first() {
    assert!(
      survey(COLUMN, 0)
        .problems
        .iter()
        .all(|problem| problem.examples.is_empty())
    );

    let made = survey(COLUMN, 1);
    // `3,5 g` twice, once with two spaces: one text.
    assert_eq!(made.problems[0], problem("ambiguous_separator", 3, &[("3,5 g", 2)]));
    assert_eq!(made.problems[3], problem("no_number", 1, &[("n/a", 1)]));

    let made = survey(COLUMN, 5);
    assert_eq!(
      made.problems[0],
      problem("ambiguous_separator", 3, &[("3,5 g", 2), ("1,00 g", 1)])
    );
    assert_eq!(made.problems[1], problem("empty", 1, &[("", 1)]));
    // Among texts as frequent as each other, in order of the text.
    let tied = survey("b\na\nc\nc\n", 2);
    assert_eq!(tied.problems, [problem("no_number", 4, &[("c", 2), ("a", 1)])]);
  }

  #[test]
  fn a_byte_order_mark_is_no_part_of_the_first_text() {
    assert_eq!(survey("\u{feff}3.5 g\n", 0).units, [unit("g", "mass", 1)]);
    // Only at the very start.
    assert_eq!(survey("1 g\n\u{feff}3.5 g\n", 0).refused, 1);
  }

  #[test]
  fn the_report_lays_out_the_survey() {
    assert_eq!(
      render(&survey(COLUMN, 1)),
      "texts    16
amounts  9
refused  7

amounts  measures       unit
      2  mass           g
      2  not converted  pack
      1  fraction       %
      1  not converted  gr
      1  mass           grams
      1  mass           mg
      1  mass           oz

Amounts in a unit that is not converted: 3. Such a unit is comparable only with its own spelling:
look among those units for misspellings of one that is, and for units the table should learn.

  texts  refused as
      3  ambiguous_separator
               2  3,5 g
      1  empty
               1  \n      1  multiple
               1  10 x 10 mg
      1  no_number
               1  n/a
      1  qualified
               1  <1 %
"
    );
    assert_eq!(render(&survey("", 0)), "texts    0\namounts  0\nrefused  0\n");
    assert_eq!(
      render(&survey("3.5 g\n", 0)),
      "texts    1\namounts  1\nrefused  0\n\namounts  measures       unit\n      1  mass           g\n"
    );
    assert_eq!(
      render(&survey("x\n", 0)),
      "texts    1\namounts  0\nrefused  1\n\n  texts  refused as\n      1  no_number\n"
    );
  }

  #[test]
  fn the_events_are_flat_and_carry_the_numbers_of_the_report() {
    let made = survey(COLUMN, 1);
    let captured = similarity_log::testing::capture("off", || emit(&made));
    let events: Vec<Value> = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    for event in &events {
      for (key, value) in event.as_object().unwrap() {
        assert!(
          !value.is_object() && !value.is_array() && !value.is_null(),
          "`{key}` in {event}"
        );
      }
    }
    let named = |name: &str| -> Vec<&Value> { events.iter().filter(|event| event["message"] == name).collect() };
    let summary = named("amount_survey");
    assert_eq!(summary.len(), 1);
    assert_eq!(events[0]["message"], "amount_survey");
    assert_eq!(
      (
        summary[0]["texts"].as_u64(),
        summary[0]["amounts"].as_u64(),
        summary[0]["refused"].as_u64(),
        summary[0]["units"].as_u64(),
        summary[0]["unconverted"].as_u64(),
      ),
      (Some(16), Some(9), Some(7), Some(7), Some(3))
    );
    let units = named("amount_unit");
    assert_eq!(units.len(), 7);
    assert_eq!(
      (
        units[1]["unit"].as_str(),
        units[1]["dimension"].as_str(),
        units[1]["amounts"].as_u64()
      ),
      (Some("pack"), Some("other"), Some(2))
    );
    let problems = named("amount_problem");
    assert_eq!(problems.len(), 5);
    assert_eq!(
      (problems[0]["code"].as_str(), problems[0]["texts"].as_u64()),
      (Some("ambiguous_separator"), Some(3))
    );
    let examples = named("amount_example");
    assert_eq!(examples.len(), 5);
    assert_eq!(
      (
        examples[0]["code"].as_str(),
        examples[0]["text"].as_str(),
        examples[0]["texts"].as_u64()
      ),
      (Some("ambiguous_separator"), Some("3,5 g"), Some(2))
    );
    // Without being asked, no text leaves.
    let quiet = similarity_log::testing::capture("off", || emit(&survey(COLUMN, 0)));
    assert!(quiet.results.iter().all(|line| !line.contains("amount_example")));
    assert!(quiet.results.iter().all(|line| !line.contains("n/a")));
  }

  /// The survey says what the service would: it is held to the file the parser and the service are held to.
  #[test]
  fn the_survey_counts_the_conformance_file_as_the_file_says() {
    let cases: Vec<Value> = include_str!("../../../../data/amounts.cases.jsonl")
      .lines()
      .filter(|line| !(line.trim().is_empty() || line.starts_with('#')))
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    // A text with a line break in it would be two lines of a column.
    let texts: Vec<&str> = cases
      .iter()
      .map(|case| case["text"].as_str().unwrap())
      .filter(|text| !text.contains(['\n', '\r']))
      .collect();
    let made = survey(&texts.join("\n"), 0);
    let expected = |refused: bool| {
      cases
        .iter()
        .filter(|case| !case["text"].as_str().unwrap().contains(['\n', '\r']))
        .filter(|case| case.get("refused").is_some() == refused)
        .count()
    };
    assert_eq!(made.texts, texts.len());
    assert_eq!((made.amounts, made.refused), (expected(false), expected(true)));
    let counted: Vec<&str> = made.problems.iter().map(|problem| problem.code).collect();
    for reason in similarity_core::amount::NotAnAmount::ALL {
      assert!(counted.contains(&reason.code()), "{} is in the file", reason.code());
    }
  }

  #[test]
  fn every_dimension_has_a_name_of_its_own() {
    let names = [
      dimension_code(&Dimension::Mass),
      dimension_code(&Dimension::Volume),
      dimension_code(&Dimension::Fraction),
      dimension_code(&Dimension::Other("pack".to_owned())),
    ];
    assert_eq!(names, ["mass", "volume", "fraction", "other"]);
  }
}
