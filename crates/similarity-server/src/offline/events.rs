//! A calibration as a stream of result events, for tools.
//!
//! The text report is for a person choosing a threshold. This is the same calibration for everything else: a
//! later run to be compared against, a chart, a table. It goes through `similarity-log` like every other result
//! (D15), one JSON object per line on stdout, never filtered.
//!
//! **The events are flat.** A pair is one `pair` line followed by one `pair_field`, `pair_quantity`, and
//! `pair_reason` line per item, all carrying the pair's `pair_id`. The logging layer writes scalars under fixed
//! names, so a map keyed by field name or a list of reasons could only have been written as a preformatted
//! string, which is exactly what D15 rules out. Flat lines are also the easier shape to filter with `jq` and to
//! load into a table.
//!
//! Similarities are rounded to six decimal places, which is below what a 32-bit cosine resolves, so re-judging
//! these lines under another configuration reproduces what scoring afresh would give. Factors are written as
//! they are. An optional value that is absent is left out, never written as `null`.

use similarity_core::Policy;
use similarity_log::RESULT;

use super::calibrate::Calibration;
use super::pairs::Label;
use super::stats::{self, Rate};
use super::{Offline, change_name, quantity_change_name};

/// A similarity, widened and rounded to six places. Without the rounding a 32-bit value prints its conversion
/// noise (`0.5670740008354187`).
fn six(value: f32) -> f64 {
  (f64::from(value) * 1e6).round() / 1e6
}

fn six_wide(value: f64) -> f64 {
  (value * 1e6).round() / 1e6
}

fn emit_pairs(calibration: &Calibration) {
  for scored in &calibration.pairs {
    let pair = &scored.pair;
    let pair_id = pair.id.as_str();
    tracing::info!(
      target: RESULT,
      pair_id,
      label = pair.label_text.as_str(),
      split = pair.split.as_deref(),
      owner = pair.owner.as_deref(),
      variant_change = pair.variant_change.as_deref(),
      label_source = pair.label_source.as_deref(),
      label_confidence = pair.label_confidence.as_deref(),
      category = pair.category(),
      note = pair.note.as_str(),
      similarity = six(scored.similarity()),
      accepted = scored.evaluation.accepted(),
      wrong = scored.wrong(),
      measure_changed = scored.evaluation.measure_changed,
      "pair"
    );
    for field in &scored.evaluation.comparison.fields {
      tracing::info!(
        target: RESULT,
        pair_id,
        field = field.field.as_str(),
        change = change_name(field.change),
        similarity = field.similarity.map(six),
        weight = six(field.weight),
        below_floor = field.below_floor,
        "pair_field"
      );
    }
    for quantity in &scored.evaluation.quantities {
      tracing::info!(
        target: RESULT,
        pair_id,
        quantity = quantity.field.as_str(),
        change = quantity_change_name(quantity.change),
        factor = quantity.factor,
        zero_crossing = quantity.zero_crossing,
        incomparable = quantity.incomparable,
        previous_unit = quantity.previous_unit.as_deref(),
        unit = quantity.unit.as_deref(),
        beyond_limit = quantity.beyond_limit,
        "pair_quantity"
      );
    }
    for reason in &scored.evaluation.reasons {
      tracing::info!(
        target: RESULT,
        pair_id,
        code = reason.code(),
        field = reason.subject(),
        value = reason.value().map(six_wide),
        limit = reason.limit().map(six_wide),
        "pair_reason"
      );
    }
  }
}

fn emit_ranges(calibration: &Calibration) {
  for label in [Label::Edit, Label::Recycle] {
    for field in stats::field_names(calibration) {
      if let Some(range) = stats::field_range(calibration, field, label) {
        tracing::info!(
          target: RESULT,
          field,
          label = label.as_str(),
          count = range.count,
          low = six_wide(range.low),
          high = six_wide(range.high),
          "field_range"
        );
      }
    }
    for quantity in stats::quantity_names(calibration) {
      let range = stats::quantity_range(calibration, quantity, label);
      tracing::info!(
        target: RESULT,
        quantity,
        label = label.as_str(),
        count = range.factors.map_or(0, |factors| factors.count),
        low = range.factors.map(|factors| factors.low),
        high = range.factors.map(|factors| factors.high),
        incomparable = range.incomparable,
        zero_crossings = range.zero_crossings,
        "quantity_range"
      );
    }
  }
}

fn emit_suggestions(calibration: &Calibration) {
  if let Some(suggestion) = calibration.suggestion {
    tracing::info!(
      target: RESULT,
      threshold = six(suggestion.threshold),
      margin = six(suggestion.margin),
      false_rejects = suggestion.false_rejects,
      false_accepts = suggestion.false_accepts,
      "suggestion"
    );
  }
  for suggestion in stats::quantity_suggestions(calibration) {
    tracing::info!(
      target: RESULT,
      quantity = suggestion.quantity.as_str(),
      reject_above_factor = suggestion.reject_above_factor,
      edits_high = suggestion.edits_high,
      recycles_next = suggestion.recycles_next,
      recycles_caught = suggestion.recycles_caught,
      recycles_changed = suggestion.recycles_changed,
      edit_zero_crossings = suggestion.edit_zero_crossings,
      "quantity_suggestion"
    );
  }
}

fn emit_slices(calibration: &Calibration) {
  for slice in stats::slices(calibration) {
    tracing::info!(
      target: RESULT,
      by = slice.by,
      value = slice.value.as_str(),
      edits = slice.edits,
      recycles = slice.recycles,
      false_rejects = slice.false_rejects,
      false_accepts = slice.false_accepts,
      edits_low = slice.edits_range.map(|range| six_wide(range.low)),
      edits_high = slice.edits_range.map(|range| six_wide(range.high)),
      recycles_low = slice.recycles_range.map(|range| six_wide(range.low)),
      recycles_high = slice.recycles_range.map(|range| six_wide(range.high)),
      "slice"
    );
  }
}

/// The parts of a rate that an event carries: the rate, and the two ends of its interval.
fn parts(rate: Option<&Rate>) -> (Option<f64>, Option<f64>, Option<f64>) {
  let interval = rate.and_then(|rate| rate.interval);
  (
    rate.and_then(|rate| rate.rate).map(six_wide),
    interval.map(|interval| six_wide(interval.low)),
    interval.map(|interval| six_wide(interval.high)),
  )
}

fn emit_operating_points(calibration: &Calibration, seed: u64) {
  for point in stats::operating_points(calibration, seed) {
    let (false_reject_rate, false_reject_low, false_reject_high) = parts(point.false_reject.as_ref());
    let (false_accept_rate, ..) = parts(point.false_accept.as_ref());
    tracing::info!(
      target: RESULT,
      false_accept_target = point.target,
      threshold = point.threshold.map(six),
      false_accept_rate,
      false_reject_rate,
      false_reject_low,
      false_reject_high,
      "operating_point"
    );
  }
}

/// Writes `calibration` as result events: a `calibration` line saying what was run, the pairs, the ranges and
/// suggestions drawn from them, and a closing `summary`.
///
/// `seed` fixes the resampling behind the intervals, as it does for the text report.
pub fn emit(offline: &Offline, policy: &Policy, calibration: &Calibration, seed: u64) {
  let input = &calibration.input;
  tracing::info!(
    target: RESULT,
    fingerprint = offline.fingerprint(),
    dimension = offline.dimension(),
    max_tokens = offline.max_tokens(),
    threshold = six(policy.threshold),
    anchor_threshold = policy.anchor_threshold.map(six),
    reject_on_measure_change = policy.reject_on_measure_change,
    pairs_sha256 = input.sha256.as_str(),
    pairs = calibration.pairs.len(),
    split = input.split.as_deref(),
    min_confidence = input.min_confidence.map(super::pairs::Confidence::as_str),
    seed,
    "calibration"
  );
  emit_pairs(calibration);
  emit_ranges(calibration);
  emit_suggestions(calibration);
  emit_slices(calibration);
  emit_operating_points(calibration, seed);
  for ((key, reason), count) in &input.ignored {
    tracing::info!(target: RESULT, key = key.as_str(), reason = %reason, count, "ignored_key");
  }

  let rates = stats::error_rates(calibration, seed);
  let (false_reject_rate, false_reject_low, false_reject_high) = parts(Some(&rates.false_reject));
  let (false_accept_rate, false_accept_low, false_accept_high) = parts(Some(&rates.false_accept));
  tracing::info!(
    target: RESULT,
    pairs = calibration.pairs.len(),
    edits = calibration.labeled(Label::Edit).count(),
    recycles = calibration.labeled(Label::Recycle).count(),
    unlabeled = calibration.unlabeled().count(),
    false_rejects = calibration.false_rejects,
    false_accepts = calibration.false_accepts,
    false_reject_rate,
    false_reject_low,
    false_reject_high,
    false_accept_rate,
    false_accept_low,
    false_accept_high,
    owners = rates.owners,
    skipped_split = input.skipped_split,
    skipped_confidence = input.skipped_confidence,
    "summary"
  );
}

#[cfg(test)]
mod tests {
  use serde_json::Value;
  use similarity_core::testing::tiny_embedder;
  use similarity_core::{EmbedderOptions, FieldSpec, QuantitySpec, Schema};

  use super::*;
  use crate::offline::{Selection, calibrate, parse_full_pairs};

  fn offline() -> (Offline, Policy) {
    let field = |name: &str, weight: f32, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below,
    };
    let schema = Schema::new(vec![field("name", 4.0, Some(0.999)), field("category", 2.0, None)])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: Some(4.0),
        reject_incomparable: false,
      }])
      .unwrap();
    let options = EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    let policy = Policy::new(0.5, None).unwrap();
    (Offline::new(tiny_embedder(&options), schema, policy), policy)
  }

  // Words the tiny tokenizer knows, so that different texts are different token sequences.
  const PAIRS: &str = concat!(
    r#"{"pair_id": "p1", "label": "edit", "owner": "b1", "split": "calibrate", "variant_change": null, "#,
    r#""label_source": "moderator", "label_confidence": "high", "note": "same gummies", "strain": "x", "#,
    r#""before": {"name": "blueberry gummies", "category": "edibles > gummies", "weight": {"value": 1, "unit": "g"}}, "#,
    r#""after": {"name": "blueberry gummies", "category": "edibles > gummies", "weight": {"value": 1000, "unit": "mg"}}}"#,
    "\n",
    r#"{"pair_id": "p2", "label": "recycle", "owner": "b2", "split": "calibrate", "variant_change": "size", "#,
    r#""before": {"name": "blueberry gummies", "category": "edibles", "weight": {"value": 1, "unit": "g"}, "strain": "x"}, "#,
    r#""after": {"name": "vape cartridge resin", "category": "flower", "weight": {"value": 8, "unit": "g"}}}"#,
    "\n",
    r#"{"pair_id": "p3", "label": "unsure", "before": {"name": "indica flower"}, "after": {"name": "sativa flower"}}"#,
  );

  fn events() -> Vec<Value> {
    let (offline, policy) = offline();
    let pairs = parse_full_pairs(PAIRS, offline.schema(), Selection::default()).unwrap();
    let calibration = calibrate(&offline, pairs).unwrap();
    let captured = similarity_log::testing::capture("warn", || emit(&offline, &policy, &calibration, 0));
    assert!(captured.logs.is_empty(), "results are not logs: {:?}", captured.logs);
    captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap_or_else(|error| panic!("{error}: {line}")))
      .collect()
  }

  fn named<'a>(events: &'a [Value], message: &str) -> Vec<&'a Value> {
    events.iter().filter(|event| event["message"] == message).collect()
  }

  #[test]
  fn a_calibration_is_written_as_flat_result_lines_even_when_logs_are_filtered() {
    let events = events();
    assert_eq!(events.first().unwrap()["message"], "calibration");
    assert_eq!(events.last().unwrap()["message"], "summary");
    for event in &events {
      assert_eq!(event["target"], "result", "{event}");
      let object = event.as_object().unwrap();
      assert!(
        object
          .values()
          .all(|value| !value.is_null() && !value.is_object() && !value.is_array()),
        "every value is a scalar and none is null: {event}"
      );
    }

    let header = &events[0];
    assert_eq!(header["pairs"], 3);
    assert_eq!(header["max_tokens"], 16);
    assert!(header["fingerprint"].is_string() && header["pairs_sha256"].as_str().unwrap().len() == 64);
    assert!((header["threshold"].as_f64().unwrap() - 0.5).abs() < 1e-9);
    assert!(
      header.get("anchor_threshold").is_none(),
      "an absent optional is left out"
    );

    assert_eq!(named(&events, "pair").len(), 3);
    assert_eq!(named(&events, "pair_field").len(), 5, "two, two, and one");
    assert_eq!(named(&events, "pair_quantity").len(), 2);
    let ignored = named(&events, "ignored_key");
    assert_eq!(ignored.len(), 1);
    assert_eq!(
      (&ignored[0]["key"], &ignored[0]["count"], &ignored[0]["reason"]),
      (
        &Value::from("strain"),
        &Value::from(1),
        &Value::from("not in the schema")
      ),
      "a key beside `before` is bookkeeping; a key inside it that the schema lacks is counted"
    );

    let summary = events.last().unwrap();
    assert_eq!(
      (
        &summary["pairs"],
        &summary["edits"],
        &summary["recycles"],
        &summary["unlabeled"]
      ),
      (&Value::from(3), &Value::from(1), &Value::from(1), &Value::from(1))
    );
    assert_eq!(summary["owners"], 2);
  }

  #[test]
  fn a_pair_line_carries_its_bookkeeping_and_its_items_carry_its_id() {
    let events = events();
    let pairs = named(&events, "pair");
    let find = |id: &str| *pairs.iter().find(|pair| pair["pair_id"] == id).unwrap();

    let edit = find("p1");
    assert_eq!(
      (&edit["label"], &edit["owner"], &edit["split"], &edit["category"]),
      (
        &Value::from("edit"),
        &Value::from("b1"),
        &Value::from("calibrate"),
        &Value::from("edibles")
      )
    );
    assert_eq!(edit["label_confidence"], "high");
    assert!(edit.get("variant_change").is_none(), "a null in the file is no value");
    assert!(
      (edit["similarity"].as_f64().unwrap() - 1.0).abs() < 1e-9,
      "identical text"
    );
    assert_eq!(
      (&edit["accepted"], &edit["wrong"]),
      (&Value::from(true), &Value::from(false))
    );

    let recycle = find("p2");
    assert_eq!(recycle["variant_change"], "size");
    assert_eq!(recycle["accepted"], false);
    let unsure = find("p3");
    assert_eq!(
      (&unsure["label"], &unsure["wrong"]),
      (&Value::from("unsure"), &Value::from(false))
    );
    assert!(unsure.get("owner").is_none());

    let quantity = |id: &str| {
      *named(&events, "pair_quantity")
        .iter()
        .find(|event| event["pair_id"] == id)
        .unwrap()
    };
    let same = quantity("p1");
    assert_eq!(
      (&same["quantity"], &same["change"]),
      (&Value::from("weight"), &Value::from("unchanged"))
    );
    assert_eq!(
      (&same["previous_unit"], &same["unit"]),
      (&Value::from("g"), &Value::from("mg"))
    );
    assert!((same["factor"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    let bigger = quantity("p2");
    assert!((bigger["factor"].as_f64().unwrap() - 8.0).abs() < 1e-12);
    assert_eq!(bigger["beyond_limit"], true);

    // The recycle is rejected for at least its quantity and its renamed product, each as a line of its own.
    let codes: Vec<&str> = named(&events, "pair_reason")
      .iter()
      .filter(|event| event["pair_id"] == "p2")
      .map(|event| event["code"].as_str().unwrap())
      .collect();
    assert!(
      codes.contains(&"quantity_beyond_factor") && codes.contains(&"field_below_floor"),
      "{codes:?}"
    );
    let beyond = *named(&events, "pair_reason")
      .iter()
      .find(|event| event["code"] == "quantity_beyond_factor")
      .unwrap();
    assert_eq!(beyond["field"], "weight");
    assert!((beyond["value"].as_f64().unwrap() - 8.0).abs() < 1e-9);
    assert!((beyond["limit"].as_f64().unwrap() - 4.0).abs() < 1e-9);
  }

  #[test]
  fn the_field_lines_of_a_pair_add_up_to_its_similarity_under_their_weights() {
    let events = events();
    for pair in named(&events, "pair") {
      let (mut weighted, mut total) = (0.0, 0.0);
      for field in named(&events, "pair_field")
        .iter()
        .filter(|field| field["pair_id"] == pair["pair_id"])
      {
        let weight = field["weight"].as_f64().unwrap();
        if let Some(similarity) = field["similarity"].as_f64() {
          weighted += weight * similarity;
          total += weight;
        }
      }
      let similarity = pair["similarity"].as_f64().unwrap();
      assert!(
        (weighted / total - similarity).abs() < 1e-5,
        "{}: {} against {similarity}",
        pair["pair_id"],
        weighted / total
      );
    }
  }

  #[test]
  fn ranges_slices_and_suggestions_follow_the_pairs() {
    let events = events();
    let by: Vec<(&str, &str)> = named(&events, "slice")
      .iter()
      .map(|slice| (slice["by"].as_str().unwrap(), slice["value"].as_str().unwrap()))
      .collect();
    assert_eq!(
      by,
      [
        ("variant_change", "size"),
        ("category", "edibles"),
        ("label_source", "moderator"),
        ("owner", "b1"),
        ("owner", "b2")
      ]
    );

    let ranges = named(&events, "quantity_range");
    assert_eq!(ranges.len(), 2, "one per label");
    let recycles = *ranges.iter().find(|range| range["label"] == "recycle").unwrap();
    assert_eq!(recycles["count"], 1);
    assert!((recycles["low"].as_f64().unwrap() - 8.0).abs() < 1e-12);
    let edits = *ranges.iter().find(|range| range["label"] == "edit").unwrap();
    assert_eq!(edits["count"], 0, "the same weight in another unit is no change");
    assert!(edits.get("low").is_none());

    // One pair of each label: a threshold between them, and one operating point.
    assert_eq!(named(&events, "suggestion").len(), 1);
    let points = named(&events, "operating_point");
    assert_eq!(points.len(), 1);
    assert!((points[0]["false_accept_target"].as_f64().unwrap() - 0.01).abs() < 1e-12);
    // No edit changes the weight, so there is nothing to place a limit above.
    let suggestion = named(&events, "quantity_suggestion");
    assert!(suggestion[0].get("reject_above_factor").is_none());
    assert_eq!(suggestion[0]["recycles_changed"], 1);
  }
}
