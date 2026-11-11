//! The reports `compare` and `calibrate` print for a person at a terminal.
//!
//! Plain text on purpose (D15): these are read by whoever is choosing a threshold, and are not logs.

use std::fmt::Write as _;

use similarity_core::Policy;
use similarity_core::score::Evaluation;

use super::calibrate::{Calibration, ScoredPair};
use super::pairs::Label;
use super::stats::{self, Interval, QuantityRange, Range, Rate};
use super::{change_name as change_mark, quantity_change_name as quantity_mark};

/// Appends how each quantity changed, if the record has any. Says nothing at all for a record without quantities,
/// so a report on text-only records reads as it always did.
fn render_quantities(evaluation: &Evaluation, out: &mut String) {
  if evaluation.quantities.is_empty() {
    return;
  }
  let _ = writeln!(out, "\n{:<20} {:>8}  {:<14} units", "quantity", "factor", "change");
  for quantity in &evaluation.quantities {
    let factor = match quantity.factor {
      Some(factor) => format!("{factor:.2}"),
      None if quantity.zero_crossing => "no limit".to_owned(),
      None => "-".to_owned(),
    };
    let unit = |unit: &Option<String>| unit.clone().unwrap_or_else(|| "-".to_owned());
    let flag = if quantity.beyond_limit { "  << beyond limit" } else { "" };
    let _ = writeln!(
      out,
      "{:<20} {factor:>8}  {:<14} {} to {}{flag}",
      quantity.field,
      quantity_mark(quantity.change),
      unit(&quantity.previous_unit),
      unit(&quantity.unit)
    );
  }
  if evaluation.measure_changed {
    let _ = writeln!(out, "the measures the record is described by changed");
  }
}

/// Renders one comparison as a small table, for a terminal.
pub fn render_comparison(evaluation: &Evaluation, policy: &Policy) -> String {
  let mut out = String::new();
  // Writing to a `String` cannot fail, here and below.
  let _ = writeln!(out, "{:<16} {:>6} {:>10}  change", "field", "weight", "similarity");
  for field in &evaluation.comparison.fields {
    let similarity = field
      .similarity
      .map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"));
    let flag = if field.below_floor { "  << below floor" } else { "" };
    let _ = writeln!(
      out,
      "{:<16} {:>6.2} {:>10}  {}{flag}",
      field.field,
      field.weight,
      similarity,
      change_mark(field.change)
    );
  }
  render_quantities(evaluation, &mut out);
  let _ = writeln!(
    out,
    "\noverall similarity {:.3} (threshold {:.3})",
    evaluation.comparison.similarity, policy.threshold
  );
  if evaluation.accepted() {
    let _ = writeln!(out, "verdict: ACCEPTED");
  } else {
    let _ = writeln!(out, "verdict: REJECTED");
    for reason in &evaluation.reasons {
      let _ = writeln!(out, "  - {reason}");
    }
  }
  out
}

/// Appends, per field, the range of its similarity over the pairs where that field's text actually changed.
///
/// This is what a field's `reject_below` floor is chosen from: it belongs in the gap between the two ranges, if
/// there is one.
fn render_field_ranges(calibration: &Calibration, out: &mut String) {
  let describe = |range: Option<Range>| match range {
    Some(range) => format!("{:.3} to {:.3} ({:>2})", range.low, range.high, range.count),
    None => "-".to_owned(),
  };
  let _ = writeln!(
    out,
    "\n{:<20} {:<22} {:<22}",
    "field (if modified)", "in edits", "in recycles"
  );
  for field in stats::field_names(calibration) {
    let _ = writeln!(
      out,
      "{field:<20} {:<22} {:<22}",
      describe(stats::field_range(calibration, field, Label::Edit)),
      describe(stats::field_range(calibration, field, Label::Recycle))
    );
  }
}

fn describe_quantity_range(range: &QuantityRange) -> String {
  let mut parts = Vec::new();
  if let Some(factors) = range.factors {
    parts.push(format!(
      "{:.2} to {:.2} ({:>2})",
      factors.low, factors.high, factors.count
    ));
  }
  if range.incomparable > 0 {
    parts.push(format!("{} incomparable", range.incomparable));
  }
  if range.zero_crossings > 0 {
    parts.push(format!("{} to or from zero", range.zero_crossings));
  }
  if parts.is_empty() {
    "-".to_owned()
  } else {
    parts.join(", ")
  }
}

/// Appends, per quantity, the factors it changed by among edits and among recycles, how often the measures
/// changed under each label, and where a factor limit could go. Says nothing if no pair has a quantity.
///
/// `reject_incomparable` and `reject_on_measure_change` are safe to switch on when the counts under `in edits`
/// are zero. A suggested limit sits above every edit; see [`stats::QuantitySuggestion`].
fn render_quantity_ranges(calibration: &Calibration, out: &mut String) {
  let names = stats::quantity_names(calibration);
  if names.is_empty() {
    return;
  }
  let _ = writeln!(out, "\n{:<20} {:<38} in recycles", "quantity (factor)", "in edits");
  for name in names {
    let range = |label: Label| describe_quantity_range(&stats::quantity_range(calibration, name, label));
    let _ = writeln!(out, "{name:<20} {:<38} {}", range(Label::Edit), range(Label::Recycle));
  }
  let measures = |label: Label| {
    let changed = calibration
      .labeled(label)
      .filter(|scored| scored.evaluation.measure_changed)
      .count();
    format!("{changed} of {}", calibration.labeled(label).count())
  };
  let _ = writeln!(
    out,
    "measures changed in {} edit(s) and {} recycle(s)",
    measures(Label::Edit),
    measures(Label::Recycle)
  );
  for suggestion in stats::quantity_suggestions(calibration) {
    if let (Some(limit), Some(largest)) = (suggestion.reject_above_factor, suggestion.edits_high) {
      let _ = writeln!(
        out,
        "a limit of {limit:.2} on {} is above every edit (largest {largest:.2}) and rejects {} of the {} recycle(s) \
         that change it",
        suggestion.quantity, suggestion.recycles_caught, suggestion.recycles_changed
      );
    }
    if suggestion.edit_zero_crossings > 0 {
      let _ = writeln!(
        out,
        "{}: {} edit(s) move it to or from zero, which any factor limit rejects",
        suggestion.quantity, suggestion.edit_zero_crossings
      );
    }
  }
}

/// Appends one table per attribute the pairs can be sliced by. An attribute no pair has a value for is left out,
/// which for a file in the strict format is every one but the category.
fn render_slices(calibration: &Calibration, out: &mut String) {
  let slices = stats::slices(calibration);
  for (by, title) in [
    ("variant_change", "by variant change"),
    ("category", "by category"),
    ("label_source", "by label source"),
    ("owner", "by owner (largest)"),
  ] {
    let mut of_attribute = slices.iter().filter(|slice| slice.by == by).peekable();
    if of_attribute.peek().is_none() {
      continue;
    }
    let _ = writeln!(
      out,
      "\n{title:<24} {:>6} {:>9} {:>17} {:>17}",
      "edits", "recycles", "wrongly rejected", "wrongly accepted"
    );
    for slice in of_attribute {
      let _ = writeln!(
        out,
        "{:<24} {:>6} {:>9} {:>17} {:>17}",
        slice.value, slice.edits, slice.recycles, slice.false_rejects, slice.false_accepts
      );
    }
  }
}

fn percent(rate: f64) -> String {
  format!("{:.1}%", rate * 100.0)
}

fn percent_interval(interval: Option<Interval>) -> String {
  interval.map_or_else(String::new, |interval| {
    format!(" ({} to {})", percent(interval.low), percent(interval.high))
  })
}

fn describe_rate(rate: &Rate) -> String {
  match rate.rate {
    Some(value) => format!("{}{}", percent(value), percent_interval(rate.interval)),
    None => "-".to_owned(),
  }
}

/// Appends the two error rates with their intervals, when the pairs name enough owners to resample.
fn render_error_rates(calibration: &Calibration, seed: u64, out: &mut String) {
  let rates = stats::error_rates(calibration, seed);
  let Some(owners) = rates.owners else {
    return;
  };
  let _ = writeln!(
    out,
    "false-reject rate {}, false-accept rate {} (95% intervals from resampling {owners} owners)",
    describe_rate(&rates.false_reject),
    describe_rate(&rates.false_accept)
  );
}

/// Appends what it costs in rejected edits to let through at most 1%, 2%, and 5% of recycles.
fn render_operating_points(calibration: &Calibration, seed: u64, out: &mut String) {
  for point in stats::operating_points(calibration, seed) {
    let target = percent(point.target);
    match (point.threshold, point.false_reject) {
      (Some(threshold), Some(rejected)) => {
        let _ = writeln!(
          out,
          "to accept at most {target} of recycles on the overall score alone: threshold {threshold:.3}, rejecting {} \
           of edits",
          describe_rate(&rejected)
        );
      }
      _ => {
        let _ = writeln!(
          out,
          "to accept at most {target} of recycles: no threshold does it, because a recycle scores as high as \
           anything else here"
        );
      }
    }
  }
}

/// Appends what was read and not compared: record keys the schema could not use, and pairs that were not selected.
fn render_input(calibration: &Calibration, out: &mut String) {
  let input = &calibration.input;
  if !input.ignored.is_empty() {
    let keys: Vec<String> = input
      .ignored
      .iter()
      .map(|((key, reason), count)| format!("{key} ({count}, {reason})"))
      .collect();
    let _ = writeln!(out, "\nrecord keys that were not compared: {}", keys.join("; "));
  }
  if input.skipped_split > 0 || input.skipped_confidence > 0 {
    let _ = writeln!(
      out,
      "left out: {} pair(s) of another split, {} labeled pair(s) below the confidence asked for",
      input.skipped_split, input.skipped_confidence
    );
  }
}

/// Renders a calibration as a report, for a terminal.
///
/// `seed` fixes the resampling behind the intervals, so the same pairs give the same report.
pub fn render_calibration(calibration: &Calibration, policy: &Policy, seed: u64) -> String {
  let mut out = String::new();
  let _ = writeln!(out, "{:<8} {:>7}  {:<9} note", "label", "overall", "verdict");
  let mut ordered: Vec<&ScoredPair> = calibration.pairs.iter().collect();
  ordered.sort_by(|a, b| b.similarity().total_cmp(&a.similarity()));
  for scored in ordered {
    let verdict = if scored.evaluation.accepted() {
      "accepted"
    } else {
      "rejected"
    };
    let wrong = if scored.wrong() { "  << WRONG" } else { "" };
    // A pair that says what the record was first created as is one step of a history, judged against that too.
    let anchored = if scored.pair.anchor.is_some() {
      " (anchored)"
    } else {
      ""
    };
    let _ = writeln!(
      out,
      "{:<8} {:>7.3}  {verdict:<9} {}{anchored}{wrong}",
      scored.pair.label_text,
      scored.similarity(),
      scored.pair.note
    );
  }

  let _ = writeln!(out);
  for (name, label) in [("edits", Label::Edit), ("recycles", Label::Recycle)] {
    if let Some(range) = stats::score_range(calibration, Some(label)) {
      let _ = writeln!(
        out,
        "{name:<9} overall similarity from {:.3} to {:.3}",
        range.low, range.high
      );
    }
  }
  // Pairs with any other label (`unsure`) are scored so that they can be looked at, and counted nowhere below.
  if let Some(range) = stats::score_range(calibration, None) {
    let accepted = calibration
      .unlabeled()
      .filter(|scored| scored.evaluation.accepted())
      .count();
    let _ = writeln!(
      out,
      "{:<9} overall similarity from {:.3} to {:.3} ({} pair(s) with another label, {accepted} of which would be \
       accepted; counted nowhere below)",
      "others", range.low, range.high, range.count
    );
  }
  render_field_ranges(calibration, &mut out);
  render_quantity_ranges(calibration, &mut out);
  render_slices(calibration, &mut out);

  let labeled = calibration
    .pairs
    .iter()
    .filter(|scored| scored.pair.label.is_some())
    .count();
  let _ = writeln!(
    out,
    "\nconfigured policy (threshold {:.3} plus field floors and any quantity limits): {} edit(s) wrongly rejected, \
     {} recycle(s) wrongly accepted, of {labeled} pairs",
    policy.threshold, calibration.false_rejects, calibration.false_accepts
  );
  render_error_rates(calibration, seed, &mut out);
  match calibration.suggestion {
    Some(suggestion) => {
      let _ = writeln!(
        out,
        "best overall threshold for this data: {:.3} (margin {:.3}; {} edit(s) rejected, {} recycle(s) accepted)",
        suggestion.threshold, suggestion.margin, suggestion.false_rejects, suggestion.false_accepts
      );
    }
    None => {
      let _ = writeln!(
        out,
        "no threshold suggested: the file needs at least one `edit` and one `recycle` pair"
      );
    }
  }
  render_operating_points(calibration, seed, &mut out);
  render_input(calibration, &mut out);
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::offline::calibrate::summarize;
  use crate::offline::pairs::{Ignored, Input};
  use crate::offline::testing::{amount, measured, scored, scored_as};

  fn summarized(pairs: Vec<ScoredPair>) -> Calibration {
    summarize(pairs, Input::default())
  }

  fn policy() -> Policy {
    Policy::new(0.8, None).unwrap()
  }

  #[test]
  fn the_summary_counts_what_the_configured_policy_gets_wrong() {
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Edit, 0.70, false),
      scored(Label::Recycle, 0.85, true),
      scored(Label::Recycle, 0.20, false),
    ]);
    assert_eq!((calibration.false_rejects, calibration.false_accepts), (1, 1));

    let report = render_calibration(&calibration, &policy(), 0);
    assert_eq!(report.matches("<< WRONG").count(), 2, "{report}");
    assert!(
      report.contains("1 edit(s) wrongly rejected, 1 recycle(s) wrongly accepted, of 4 pairs"),
      "{report}"
    );
    assert!(report.contains("best overall threshold"), "{report}");
  }

  #[test]
  fn a_comparison_renders_its_reasons() {
    let evaluation = scored(Label::Recycle, 0.2, false).evaluation;
    let report = render_comparison(&evaluation, &policy());
    assert!(
      report.contains("verdict: REJECTED") && report.contains("below the threshold"),
      "{report}"
    );
    let accepted = scored(Label::Edit, 0.9, true).evaluation;
    let report = render_comparison(&accepted, &policy());
    assert!(report.contains("verdict: ACCEPTED"));
    assert!(
      !report.contains("quantity"),
      "a record without quantities says nothing about them: {report}"
    );
  }

  #[test]
  fn a_comparison_renders_each_quantity_and_what_exceeded_its_limit() {
    let pair = measured(
      Label::Recycle,
      &[
        ("weight", Some(amount(3.5, "g")), Some(amount(28.0, "g"))),
        ("size", Some(amount(1.0, "g")), Some(amount(1000.0, "mg"))),
        ("thc", Some(amount(22.0, "%")), Some(amount(22.0, "mg"))),
        ("cbd", Some(amount(0.0, "mg")), Some(amount(5.0, "mg"))),
        ("volume", None, Some(amount(355.0, "ml"))),
        ("count", Some(amount(10.0, "each")), None),
      ],
    );
    let report = render_comparison(&pair.evaluation, &policy());
    let line = |name: &str| {
      report
        .lines()
        .find(|line| line.starts_with(name))
        .unwrap_or_else(|| panic!("no line for {name}: {report}"))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
    };
    assert_eq!(line("weight"), "weight 8.00 modified g to g << beyond limit");
    assert_eq!(line("size"), "size 1.00 unchanged g to mg");
    assert_eq!(line("thc"), "thc - incomparable % to mg");
    assert_eq!(line("cbd"), "cbd no limit modified mg to mg << beyond limit");
    assert_eq!(line("volume"), "volume - added - to ml");
    assert_eq!(line("count"), "count - removed each to -");
    assert!(
      report.contains("the measures the record is described by changed"),
      "{report}"
    );
    assert!(
      report.contains("verdict: REJECTED") && report.contains("factor of 8.00"),
      "{report}"
    );
  }

  #[test]
  fn the_calibration_report_sets_each_quantitys_factors_in_edits_beside_those_in_recycles() {
    let calibration = summarized(vec![
      measured(
        Label::Edit,
        &[("thc", Some(amount(21.3, "%")), Some(amount(22.8, "%")))],
      ),
      measured(
        Label::Edit,
        &[("thc", Some(amount(20.0, "%")), Some(amount(25.0, "%")))],
      ),
      // The same amount in another unit is no change, and must not pull the range down to 1.
      measured(
        Label::Edit,
        &[("weight", Some(amount(1.0, "g")), Some(amount(1000.0, "mg")))],
      ),
      measured(
        Label::Recycle,
        &[("weight", Some(amount(3.5, "g")), Some(amount(28.0, "g")))],
      ),
      measured(
        Label::Recycle,
        &[("thc", Some(amount(22.0, "%")), Some(amount(22.0, "mg")))],
      ),
      measured(
        Label::Recycle,
        &[
          ("weight", Some(amount(3.5, "g")), None),
          ("volume", None, Some(amount(355.0, "ml"))),
          ("thc", Some(amount(0.0, "mg")), Some(amount(10.0, "mg"))),
        ],
      ),
      scored(Label::Edit, 0.97, true),
    ]);
    let report = render_calibration(&calibration, &policy(), 0);
    let line = |name: &str| {
      report
        .lines()
        .find(|line| line.starts_with(name))
        .unwrap_or_else(|| panic!("no line for {name}: {report}"))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
    };
    assert_eq!(line("thc"), "thc 1.07 to 1.25 ( 2) 1 incomparable, 1 to or from zero");
    assert_eq!(line("weight"), "weight - 8.00 to 8.00 ( 1)");
    assert_eq!(line("volume"), "volume - -");
    assert!(
      report.contains("measures changed in 0 of 4 edit(s) and 2 of 3 recycle(s)"),
      "{report}"
    );
    // Limits count toward what the configured policy gets wrong: the eightfold weight and the potency from zero
    // exceed this schema's limit of 4, so those two recycles are rejected and only the third is wrongly accepted.
    assert!(
      report.contains("0 edit(s) wrongly rejected, 1 recycle(s) wrongly accepted, of 7 pairs"),
      "{report}"
    );
  }

  #[test]
  fn a_calibration_without_quantities_reports_none() {
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.2, false),
    ]);
    let report = render_calibration(&calibration, &policy(), 0);
    assert!(
      !report.contains("quantity (factor)") && !report.contains("measures changed"),
      "{report}"
    );
  }

  #[test]
  fn a_limit_is_suggested_above_every_edit_with_what_it_would_catch() {
    let change = |label: Label, before: f64, after: f64| {
      measured(
        label,
        &[("weight", Some(amount(before, "g")), Some(amount(after, "g")))],
      )
    };
    let calibration = summarized(vec![
      change(Label::Edit, 1.0, 1.25),
      change(Label::Edit, 0.0, 1.0),
      change(Label::Recycle, 1.0, 1.1),
      change(Label::Recycle, 1.0, 5.0),
    ]);
    let report = render_calibration(&calibration, &policy(), 0);
    assert!(
      report.contains(
        "a limit of 2.50 on weight is above every edit (largest 1.25) and rejects 1 of the 2 recycle(s) that change it"
      ),
      "{report}"
    );
    assert!(
      report.contains("weight: 1 edit(s) move it to or from zero, which any factor limit rejects"),
      "{report}"
    );
  }

  #[test]
  fn pairs_with_another_label_are_shown_and_counted_nowhere() {
    let unsure = |similarity: f32, accepted: bool| {
      scored_as(Label::Edit, similarity, accepted, |pair| {
        pair.label = None;
        pair.label_text = "unsure".to_owned();
      })
    };
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.2, false),
      unsure(0.9, true),
      unsure(0.6, false),
    ]);
    assert_eq!((calibration.false_rejects, calibration.false_accepts), (0, 0));

    let report = render_calibration(&calibration, &policy(), 0);
    assert_eq!(
      report.lines().filter(|line| line.starts_with("unsure ")).count(),
      2,
      "{report}"
    );
    assert!(!report.contains("WRONG"), "{report}");
    assert!(
      report.contains("others    overall similarity from 0.600 to 0.900 (2 pair(s) with another label, 1 of which"),
      "{report}"
    );
    assert!(
      report.contains("0 edit(s) wrongly rejected, 0 recycle(s) wrongly accepted, of 2 pairs"),
      "{report}"
    );
    // The suggestion comes from the labeled pairs alone: the midpoint of 0.2 and 0.95.
    assert!(
      report.contains("best overall threshold for this data: 0.575"),
      "{report}"
    );
  }

  #[test]
  fn slices_rates_and_what_was_left_out_are_reported_when_the_pairs_carry_them() {
    let pair = |label: Label, similarity: f32, accepted: bool, owner: &str, variant: Option<&str>| {
      scored_as(label, similarity, accepted, |pair| {
        pair.owner = Some(owner.to_owned());
        pair.variant_change = variant.map(str::to_owned);
        pair
          .before
          .texts
          .insert("category".to_owned(), "Edibles > Gummies".to_owned());
      })
    };
    let mut input = Input {
      skipped_split: 12,
      skipped_confidence: 3,
      ..Input::default()
    };
    input.ignored.insert(("strain".to_owned(), Ignored::NotInSchema), 40);
    input
      .ignored
      .insert(("weight_per_item".to_owned(), Ignored::WrongKind), 2);
    let calibration = summarize(
      vec![
        pair(Label::Edit, 0.95, true, "b1", None),
        pair(Label::Edit, 0.70, false, "b2", Some("flavor")),
        pair(Label::Recycle, 0.85, true, "b2", Some("flavor")),
        pair(Label::Recycle, 0.20, false, "b1", None),
      ],
      input,
    );
    let report = render_calibration(&calibration, &policy(), 0);
    let squeezed = |line: &str| line.split_whitespace().collect::<Vec<_>>().join(" ");
    let lines: Vec<String> = report.lines().map(squeezed).collect();
    let has = |wanted: &str| lines.iter().any(|line| line == wanted);

    assert!(
      has("by variant change edits recycles wrongly rejected wrongly accepted"),
      "{report}"
    );
    assert!(has("flavor 1 1 1 1"), "{report}");
    assert!(has("Edibles 2 2 1 1"), "{report}");
    assert!(has("b1 1 1 0 0") && has("b2 1 1 1 1"), "{report}");
    assert!(!report.contains("by label source"), "no pair has one: {report}");

    assert!(
      report.contains("false-reject rate 50.0% (0.0% to 100.0%), false-accept rate 50.0% (0.0% to 100.0%) (95% intervals from resampling 2 owners)"),
      "{report}"
    );
    assert!(
      report.contains("record keys that were not compared: strain (40, not in the schema); weight_per_item (2, the wrong kind of value)"),
      "{report}"
    );
    assert!(
      report.contains("left out: 12 pair(s) of another split, 3 labeled pair(s) below the confidence asked for"),
      "{report}"
    );
  }

  #[test]
  fn a_strict_file_reports_no_rates_and_no_slices_but_the_category() {
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.2, false),
    ]);
    let report = render_calibration(&calibration, &policy(), 0);
    assert!(
      !report.contains("false-reject rate") && !report.contains("by ") && !report.contains("left out"),
      "{report}"
    );
  }

  #[test]
  fn operating_points_say_what_a_cap_on_accepted_recycles_costs() {
    let calibration = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Edit, 0.60, false),
      scored(Label::Recycle, 0.70, false),
      scored(Label::Recycle, 0.30, false),
    ]);
    let report = render_calibration(&calibration, &policy(), 0);
    assert!(
      report.contains(
        "to accept at most 1.0% of recycles on the overall score alone: threshold 0.825, rejecting 50.0% of edits"
      ),
      "{report}"
    );

    let tied = summarized(vec![
      scored(Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.95, true),
    ]);
    let report = render_calibration(&tied, &policy(), 0);
    assert!(report.contains("no threshold does it"), "{report}");
  }
}
