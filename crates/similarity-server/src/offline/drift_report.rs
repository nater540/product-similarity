//! A drift run for a person, and the same run as data.
//!
//! Both are drawn from [`stats::drift`](super::stats::drift), so the report and the events cannot disagree. The
//! events are flat, like a calibration's (D17): one `history` line, then one line per step, and per field,
//! quantity, and reason of a step, joined on `history_id` and `step`. Their names are an interface once written:
//! fields are added, never renamed.

use std::fmt::Write as _;

use similarity_core::{Policy, Schema};
use similarity_log::RESULT;

use super::drift::DriftRun;
use super::events::{emit_schema, six, six_wide};
use super::pairs::{Confidence, Label};
use super::stats::Rate;
use super::stats::drift::{self, HistorySummary};
use super::{change_name, quantity_change_name};

/// The model a run was replayed with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model<'a> {
  /// Its fingerprint.
  pub fingerprint: &'a str,
  /// The width of one field's vector.
  pub dimension: usize,
  /// The token limit.
  pub max_tokens: usize,
}

fn number(value: Option<f64>) -> String {
  value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"))
}

fn factor(value: Option<f64>) -> String {
  value.map_or_else(|| "-".to_owned(), |value| format!("{value:.2}"))
}

fn share(rate: &Rate) -> String {
  let interval = rate.interval.map_or_else(String::new, |interval| {
    format!(", {:.1}% to {:.1}%", interval.low * 100.0, interval.high * 100.0)
  });
  format!("{} of {}{interval}", rate.errors, rate.of)
}

fn render_histories(run: &DriftRun, summaries: &[HistorySummary], out: &mut String) {
  let _ = writeln!(
    out,
    "{:<8} {:>5}  {:>16}  {:<34} history",
    "label", "steps", "lowest to anchor", "first rejected"
  );
  for (replayed, summary) in run.histories.iter().zip(summaries) {
    let lowest = summary.min_anchor.map_or_else(
      || "-".to_owned(),
      |(similarity, step)| format!("{similarity:.3} at {step}"),
    );
    let rejected = drift::first_rejection(replayed).map_or_else(
      || "-".to_owned(),
      |(step, reasons)| {
        let codes: Vec<&str> = reasons.iter().map(similarity_core::Reason::code).collect();
        format!("step {} ({})", step.step, codes.join(", "))
      },
    );
    let history = &replayed.history;
    let recycled = history
      .recycled_at
      .map_or_else(String::new, |at| format!(", recycled at {at}"));
    let truncated = if history.truncated { ", truncated" } else { "" };
    let _ = writeln!(
      out,
      "{:<8} {:>5}  {lowest:>16}  {rejected:<34} {}{recycled}{truncated}",
      history.label_text, summary.steps, history.id
    );
  }
}

fn render_quantities(run: &DriftRun, out: &mut String) {
  let quantities = drift::quantity_drift(run);
  if quantities.is_empty() {
    return;
  }
  let _ = writeln!(
    out,
    "\n{:<20} {:<28} {:<28} walked",
    "quantity (factor)", "legitimate: step, vs first", "recycles: step, vs first"
  );
  for quantity in &quantities {
    let pair = |(step, first): (Option<f64>, Option<f64>)| format!("{}, {}", factor(step), factor(first));
    let _ = writeln!(
      out,
      "{:<20} {:<28} {:<28} {}",
      quantity.quantity,
      pair(quantity.edits),
      pair(quantity.recycles),
      quantity.walked
    );
  }
  let walked: usize = quantities.iter().map(|quantity| quantity.walked).sum();
  let _ = writeln!(
    out,
    "walked: recycles that end further from a quantity's first value than any legitimate history does, in steps no \
     larger than a legitimate history's largest. A limit on each step cannot catch these ({walked} here)."
  );
}

/// Appends what was left out of the run, and of its suggestion.
fn render_left_out(run: &DriftRun, truncated: usize, out: &mut String) {
  if truncated > 0 {
    let _ = writeln!(
      out,
      "\n{truncated} truncated history(ies) are left out of the ranges and the suggestion: they start after their creation, \
       so they drifted further than they show"
    );
  }
  let input = &run.input;
  if !input.ignored.is_empty() {
    let keys: Vec<String> = input
      .ignored
      .iter()
      .map(|((key, reason), count)| format!("{key} ({count}, {reason})"))
      .collect();
    let _ = writeln!(out, "\nrecord keys that were not compared: {}", keys.join("; "));
  }
  if input.skipped_confidence > 0 {
    let _ = writeln!(
      out,
      "left out: {} labeled history(ies) below the confidence asked for",
      input.skipped_confidence
    );
  }
}

/// Renders a drift run as a report, for a terminal. `seed` fixes the resampling behind the intervals.
pub fn render_drift(run: &DriftRun, policy: &Policy, seed: u64) -> String {
  let mut out = String::new();
  let totals = drift::totals(run);
  // Writing to a `String` cannot fail, here and below.
  let _ = writeln!(
    out,
    "histories {} ({} edit, {} recycle, {} other), {} steps\n",
    totals.histories, totals.edits, totals.recycles, totals.others, totals.steps
  );
  let summaries: Vec<HistorySummary> = run
    .histories
    .iter()
    .map(|replayed| drift::summarize(replayed, policy))
    .collect();
  render_histories(run, &summaries, &mut out);

  let _ = writeln!(
    out,
    "\n{:<34} {:>5} {:>7} {:>7} {:>7} {:>7}",
    "lowest similarity to the anchor", "count", "lowest", "p5", "median", "highest"
  );
  for (name, part) in [
    ("legitimate histories", Label::Edit),
    ("recycles, once recycled", Label::Recycle),
  ] {
    if let Some(range) = drift::drift_range(run, part) {
      let _ = writeln!(
        out,
        "{name:<34} {:>5} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
        range.count, range.lowest, range.p5, range.median, range.highest
      );
    }
  }

  let blocked = drift::blocked(run, seed);
  let recycles = drift::recycles(run, policy);
  let drift_threshold = policy.anchor_threshold.map_or_else(
    || "no anchor_threshold".to_owned(),
    |threshold| format!("anchor_threshold {threshold:.3}"),
  );
  let _ = writeln!(
    out,
    "\nconfigured policy (threshold {:.3}, {drift_threshold}, field floors and any quantity limits):",
    policy.threshold
  );
  let _ = writeln!(
    out,
    "  legitimate histories blocked: {} (by the pair rules {}, by the drift check alone {})",
    share(&blocked.by_policy),
    blocked.by_pair_rules.errors,
    blocked.by_drift_alone.errors
  );
  let _ = writeln!(
    out,
    "  recycles first rejected: at the recycled version {}, later {}, before it {}, somewhere {}, never {}; by the \
     drift check alone {}",
    recycles.at_recycle, recycles.later, recycles.early, recycles.somewhere, recycles.never, recycles.by_drift_alone
  );

  let suggestion = drift::anchor_suggestion(run, policy);
  let _ = match (suggestion.anchor_threshold, suggestion.legit_low) {
    (Some(threshold), Some(legit_low)) => writeln!(
      out,
      "\nanchor_threshold {threshold:.3} is below every legitimate history (lowest {legit_low:.3}) and rejects {} of \
       the {} recycle(s) the pair rules miss (margin {}){}",
      suggestion.recycles_caught,
      suggestion.uncaught,
      number(suggestion.margin),
      if suggestion.above_threshold {
        "; it is above the overall threshold, so it would be the stricter of the two on a record's first edit"
      } else {
        ""
      }
    ),
    (None, Some(legit_low)) => writeln!(
      out,
      "\nno anchor_threshold suggested: it must be below {legit_low:.3}, the lowest any legitimate history reaches, \
       and no recycle the pair rules miss ({} here) sinks below that",
      suggestion.uncaught
    ),
    _ => writeln!(
      out,
      "\nno anchor_threshold suggested: the file has no legitimate history to stay below"
    ),
  };
  render_quantities(run, &mut out);

  render_left_out(run, totals.truncated, &mut out);
  out
}

fn emit_steps(run: &DriftRun) {
  for replayed in &run.histories {
    let history_id = replayed.history.id.as_str();
    let recycled_at = replayed.history.recycled_at;
    for step in &replayed.steps {
      let evaluation = &step.evaluation;
      tracing::info!(
        target: RESULT,
        history_id,
        step = step.step,
        at = step.at.as_deref(),
        similarity = six(evaluation.comparison.similarity),
        anchor_similarity = evaluation.anchor_similarity.map(six),
        accepted = evaluation.accepted(),
        accepted_by_pair_rules = step.accepted_by_pair_rules,
        written = step.written,
        recycled = recycled_at.map(|at| step.step >= at),
        measure_changed = evaluation.measure_changed,
        "step"
      );
      for field in &evaluation.comparison.fields {
        let to_anchor = step.anchor_fields.iter().find(|(name, _)| *name == field.field);
        tracing::info!(
          target: RESULT,
          history_id,
          step = step.step,
          field = field.field.as_str(),
          change = change_name(field.change),
          similarity = field.similarity.map(six),
          anchor_similarity = to_anchor.and_then(|(_, similarity)| *similarity).map(six),
          weight = six(field.weight),
          below_floor = field.below_floor,
          "step_field"
        );
      }
      for quantity in &evaluation.quantities {
        let first = step
          .anchor_quantities
          .iter()
          .find(|score| score.field == quantity.field);
        tracing::info!(
          target: RESULT,
          history_id,
          step = step.step,
          quantity = quantity.field.as_str(),
          change = quantity_change_name(quantity.change),
          factor = quantity.factor,
          zero_crossing = quantity.zero_crossing,
          incomparable = quantity.incomparable,
          previous_unit = quantity.previous_unit.as_deref(),
          unit = quantity.unit.as_deref(),
          beyond_limit = quantity.beyond_limit,
          first_change = first.map(|score| quantity_change_name(score.change)),
          first_factor = first.and_then(|score| score.factor),
          first_zero_crossing = first.map(|score| score.zero_crossing),
          first_incomparable = first.map(|score| score.incomparable),
          "step_quantity"
        );
      }
      for reason in &evaluation.reasons {
        tracing::info!(
          target: RESULT,
          history_id,
          step = step.step,
          code = reason.code(),
          subject = reason.subject(),
          value = reason.value().map(six_wide),
          limit = reason.limit().map(six_wide),
          "step_reason"
        );
      }
    }
  }
}

fn emit_histories(run: &DriftRun, policy: &Policy) {
  for replayed in &run.histories {
    let (history, summary) = (&replayed.history, drift::summarize(replayed, policy));
    let history_id = history.id.as_str();
    tracing::info!(
      target: RESULT,
      history_id,
      label = history.label_text.as_str(),
      brand_id = history.brand_id.as_deref(),
      label_source = history.label_source.as_deref(),
      label_confidence = history.label_confidence.map(Confidence::as_str),
      recycled_at = history.recycled_at,
      truncated = history.truncated,
      seeded_from_original = history.original.is_some(),
      versions = history.versions.len(),
      first_at = history.versions.first().and_then(|version| version.at.as_deref()),
      last_at = history.versions.last().and_then(|version| version.at.as_deref()),
      min_anchor_similarity = summary.min_anchor.map(|(similarity, _)| six(similarity)),
      min_anchor_step = summary.min_anchor.map(|(_, step)| step),
      final_anchor_similarity = summary.final_anchor.map(six),
      first_rejected_step = summary.first_rejected,
      first_rejected_by_pair_rules_step = summary.first_rejected_by_pair_rules,
      first_anchor_drift_step = summary.first_anchor_drift,
      caught = summary.caught.map(drift::Caught::as_str),
      "history"
    );
    for quantity in &summary.quantities {
      tracing::info!(
        target: RESULT,
        history_id,
        quantity = quantity.quantity.as_str(),
        max_step_factor = quantity.max_step_factor,
        max_first_factor = quantity.max_first_factor,
        first_incomparable = quantity.first_incomparable,
        first_zero_crossing = quantity.first_zero_crossing,
        "history_quantity"
      );
    }
  }
}

fn emit_statistics(run: &DriftRun, policy: &Policy, seed: u64) {
  for (label, part) in [("edit", Label::Edit), ("recycle", Label::Recycle)] {
    if let Some(range) = drift::drift_range(run, part) {
      tracing::info!(
        target: RESULT,
        label,
        count = range.count,
        lowest = six_wide(range.lowest),
        p5 = six_wide(range.p5),
        median = six_wide(range.median),
        highest = six_wide(range.highest),
        "drift_range"
      );
    }
  }
  let blocked = drift::blocked(run, seed);
  for (what, rate) in [
    ("policy", blocked.by_policy),
    ("pair_rules", blocked.by_pair_rules),
    ("drift_only", blocked.by_drift_alone),
  ] {
    tracing::info!(
      target: RESULT,
      what,
      errors = rate.errors,
      of = rate.of,
      rate = rate.rate.map(six_wide),
      low = rate.interval.map(|interval| six_wide(interval.low)),
      high = rate.interval.map(|interval| six_wide(interval.high)),
      "drift_blocked"
    );
  }
  let suggestion = drift::anchor_suggestion(run, policy);
  tracing::info!(
    target: RESULT,
    anchor_threshold = suggestion.anchor_threshold.map(six_wide),
    margin = suggestion.margin.map(six_wide),
    legit_low = suggestion.legit_low.map(six_wide),
    next_below = suggestion.next_below.map(six_wide),
    recycles_caught = suggestion.recycles_caught,
    uncaught = suggestion.uncaught,
    above_threshold = suggestion.above_threshold,
    "drift_suggestion"
  );
  for quantity in drift::quantity_drift(run) {
    for (label, (step, first), incomparable, zero) in [
      (
        "edit",
        quantity.edits,
        quantity.first_incomparable.0,
        quantity.first_zero_crossings.0,
      ),
      (
        "recycle",
        quantity.recycles,
        quantity.first_incomparable.1,
        quantity.first_zero_crossings.1,
      ),
    ] {
      tracing::info!(
        target: RESULT,
        quantity = quantity.quantity.as_str(),
        label,
        max_step_factor = step,
        max_first_factor = first,
        walked = (label == "recycle").then_some(quantity.walked),
        first_incomparable = incomparable,
        first_zero_crossings = zero,
        "drift_quantity"
      );
    }
  }
}

/// Writes a drift run as result events: a `drift_run` line saying what was replayed, the schema, every history
/// and step, the statistics drawn from them, and a closing `drift_summary`.
pub fn emit_drift(model: Model<'_>, schema: &Schema, policy: &Policy, run: &DriftRun, seed: u64) {
  let input = &run.input;
  tracing::info!(
    target: RESULT,
    fingerprint = model.fingerprint,
    dimension = model.dimension,
    max_tokens = model.max_tokens,
    threshold = six(policy.threshold),
    anchor_threshold = policy.anchor_threshold.map(six),
    reject_on_measure_change = policy.reject_on_measure_change,
    histories_sha256 = input.sha256.as_str(),
    histories = run.histories.len(),
    min_confidence = input.min_confidence.map(Confidence::as_str),
    seed,
    "drift_run"
  );
  emit_schema(schema);
  emit_histories(run, policy);
  emit_steps(run);
  emit_statistics(run, policy, seed);
  for ((key, reason), count) in &input.ignored {
    tracing::info!(target: RESULT, key = key.as_str(), reason = %reason, count, "ignored_key");
  }
  let totals = drift::totals(run);
  let recycles = drift::recycles(run, policy);
  tracing::info!(
    target: RESULT,
    histories = totals.histories,
    edits = totals.edits,
    recycles = totals.recycles,
    others = totals.others,
    truncated = totals.truncated,
    steps = totals.steps,
    blocked = drift::blocked(run, seed).by_policy.errors,
    missed = recycles.never,
    skipped_confidence = input.skipped_confidence,
    "drift_summary"
  );
}

#[cfg(test)]
mod tests {
  use serde_json::Value;

  use super::*;
  use crate::offline::drift::testing::{line, policy, replayed, schema};

  fn run() -> DriftRun {
    let steady = [
      ("indica flower", None, Some(3.5)),
      ("indica flower premium", None, Some(7.0)),
    ];
    let walk = [
      ("blueberry gummies edibles sweet", None, Some(40.0)),
      ("blueberry gummies edibles sweet vegan", None, Some(40.0)),
      ("blueberry gummies edibles cartridge vegan", None, Some(40.0)),
      ("blueberry vape edibles cartridge vegan", None, Some(40.0)),
      ("sativa vape resin cartridge vegan", None, Some(40.0)),
    ];
    replayed(&[
      line("steady", r#""brand_id": "b1", "label": "edit", "#, &steady),
      line(
        "steady-2",
        r#""brand_id": "b2", "label": "edit", "label_confidence": "high", "#,
        &steady,
      ),
      line(
        "walk",
        r#""brand_id": "b3", "label": "recycle", "recycled_at": 2, "#,
        &walk,
      ),
    ])
  }

  #[test]
  fn the_report_says_how_far_histories_drift_and_where_the_threshold_belongs() {
    let report = render_drift(&run(), &policy(), 0);
    assert!(
      report.starts_with("histories 3 (2 edit, 1 recycle, 0 other), 6 steps\n"),
      "{report}"
    );
    assert!(report.contains("walk, recycled at 2"), "{report}");
    assert!(
      report.contains("(anchor_drift)"),
      "the walk is stopped by the drift check: {report}"
    );
    assert!(report.contains("legitimate histories") && report.contains("recycles, once recycled"));
    assert!(
      report.contains("configured policy (threshold 0.600, anchor_threshold 0.500"),
      "{report}"
    );
    assert!(report.contains("legitimate histories blocked: 0 of 2"), "{report}");
    assert!(
      report.contains("later 1") && report.contains("by the drift check alone 1"),
      "{report}"
    );
    assert!(report.contains("is below every legitimate history") && report.contains("rejects 1 of the 1 recycle(s)"));
    assert!(
      report.contains("weight") && report.contains("2.00, 2.00"),
      "the edits doubled a weight once: {report}"
    );

    // With nothing to stay above, it says so and suggests nothing.
    let only_edits = replayed(&[line(
      "steady",
      r#""label": "edit", "#,
      &[("a b", None, None), ("a b c", None, None)],
    )]);
    let report = render_drift(&only_edits, &policy(), 0);
    assert!(
      report.contains("no anchor_threshold suggested: it must be below"),
      "{report}"
    );
  }

  #[test]
  fn a_drift_run_is_written_as_flat_lines_joined_on_the_history_and_the_step() {
    let run = run();
    let model = Model {
      fingerprint: "word-bag-v1",
      dimension: 64,
      max_tokens: 16,
    };
    let captured = similarity_log::testing::capture("off", || emit_drift(model, &schema(), &policy(), &run, 0));
    let events: Vec<Value> = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    let named = |message: &str| -> Vec<&Value> { events.iter().filter(|event| event["message"] == message).collect() };
    assert_eq!(events.first().unwrap()["message"], "drift_run");
    assert_eq!(events.last().unwrap()["message"], "drift_summary");
    for event in &events {
      for (key, value) in event.as_object().unwrap() {
        assert!(
          !value.is_object() && !value.is_array() && !value.is_null(),
          "{key} in {event}"
        );
      }
    }
    assert_eq!((named("history").len(), named("step").len()), (3, 6));
    assert_eq!(named("schema_field").len(), 2);

    let walk = named("history")
      .into_iter()
      .find(|event| event["history_id"] == "walk")
      .unwrap();
    assert_eq!((&walk["recycled_at"], &walk["caught"]), (&2.into(), &"later".into()));
    assert!(
      walk.get("first_rejected_by_pair_rules_step").is_none(),
      "absent, not null"
    );
    let rejected = walk["first_rejected_step"].as_u64().unwrap();
    let reason = named("step_reason")
      .into_iter()
      .find(|event| event["history_id"] == "walk")
      .unwrap();
    assert_eq!(
      (&reason["code"], reason["step"].as_u64().unwrap()),
      (&"anchor_drift".into(), rejected)
    );

    // A step says whether it is part of the recycled stretch, and a field how far it is from the anchor.
    let steps: Vec<(u64, bool)> = named("step")
      .into_iter()
      .filter(|event| event["history_id"] == "walk")
      .map(|event| (event["step"].as_u64().unwrap(), event["recycled"].as_bool().unwrap()))
      .collect();
    assert_eq!(steps, [(1, false), (2, true), (3, true), (4, true)]);
    assert!(
      named("step_field")
        .iter()
        .all(|event| event["anchor_similarity"].is_number())
    );
    let weight = named("step_quantity")
      .into_iter()
      .find(|event| event["history_id"] == "steady")
      .unwrap();
    assert_eq!((&weight["factor"], &weight["first_factor"]), (&2.0.into(), &2.0.into()));

    assert_eq!(named("drift_range").len(), 2);
    assert_eq!(named("drift_blocked").len(), 3);
    assert!(named("drift_suggestion")[0]["anchor_threshold"].is_number());
    assert_eq!(named("drift_quantity").len(), 2, "one quantity, each label");
    let summary = events.last().unwrap();
    assert_eq!(
      (&summary["histories"], &summary["steps"], &summary["missed"]),
      (&3.into(), &6.into(), &0.into())
    );
  }
}
