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

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use serde_json::Value;
use similarity_core::score::{Comparison, Evaluation, FieldScore, Reason};
use similarity_core::{FieldChange, Policy, QuantityChange, QuantityScore, Schema};
use similarity_log::RESULT;

use super::calibrate::{Calibration, ScoredPair, summarize};
use super::pairs::{Confidence, Ignored, Input, Label, Pair};
use super::stats::{self, Rate};
use super::{Offline, RecordFields, change_name, quantity_change_name};

/// A similarity, widened and rounded to six places. Without the rounding a 32-bit value prints its conversion
/// noise (`0.5670740008354187`).
pub(super) fn six(value: f32) -> f64 {
  (f64::from(value) * 1e6).round() / 1e6
}

fn six_wide(value: f64) -> f64 {
  (value * 1e6).round() / 1e6
}

/// The schema the run was scored under, one line per field and per quantity, so that a file of events says what
/// produced it and a later re-judging can check that it is being asked about the same fields.
fn emit_schema(schema: &Schema) {
  for field in schema.fields() {
    tracing::info!(
      target: RESULT,
      field = field.name.as_str(),
      weight = six(field.weight),
      required = field.required,
      reject_below = field.reject_below.map(six),
      "schema_field"
    );
  }
  for quantity in schema.quantities() {
    tracing::info!(
      target: RESULT,
      quantity = quantity.name.as_str(),
      reject_above_factor = quantity.reject_above_factor,
      reject_incomparable = quantity.reject_incomparable,
      "schema_quantity"
    );
  }
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
    min_confidence = input.min_confidence.map(Confidence::as_str),
    seed,
    "calibration"
  );
  emit_schema(offline.schema());
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

/// What a file of calibration events says about the run that wrote it.
#[derive(Debug, Clone, PartialEq)]
pub struct RunInfo {
  /// The model fingerprint the pairs were scored with.
  pub fingerprint: String,
  /// The overall threshold in force.
  pub threshold: f32,
  /// The drift threshold in force, if any.
  pub anchor_threshold: Option<f32>,
  /// Whether a change of measure rejected.
  pub reject_on_measure_change: bool,
  /// The names of the text fields of the schema, in order.
  pub fields: Vec<String>,
  /// The names of the quantities of the schema, in order.
  pub quantities: Vec<String>,
}

/// A calibration read back from the events `calibrate --ndjson` wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
  /// What the run was.
  pub info: RunInfo,
  /// The pairs as they were scored, summarized again. The records themselves are not in the events, so each
  /// pair's `before` and `after` are empty apart from the top-level category.
  pub calibration: Calibration,
}

#[expect(
  clippy::cast_possible_truncation,
  reason = "a similarity that was written from 32 bits, read back into them"
)]
fn narrow(value: f64) -> f32 {
  value as f32
}

/// One event, with the accessors a reader needs. A key that is absent is `None`; a key of the wrong type is an
/// error, because it means the file was not written by this tool.
struct Event<'a>(&'a serde_json::Map<String, Value>);

impl Event<'_> {
  fn text(&self, key: &str) -> anyhow::Result<Option<String>> {
    match self.0.get(key) {
      None => Ok(None),
      Some(Value::String(text)) => Ok(Some(text.clone())),
      Some(other) => bail!("`{key}` is {other}, not text"),
    }
  }

  fn number(&self, key: &str) -> anyhow::Result<Option<f64>> {
    match self.0.get(key) {
      None => Ok(None),
      Some(value) => match value.as_f64() {
        Some(number) => Ok(Some(number)),
        None => bail!("`{key}` is {value}, not a number"),
      },
    }
  }

  fn flag(&self, key: &str) -> anyhow::Result<bool> {
    match self.0.get(key) {
      None => Ok(false),
      Some(Value::Bool(flag)) => Ok(*flag),
      Some(other) => bail!("`{key}` is {other}, not true or false"),
    }
  }

  fn count(&self, key: &str) -> anyhow::Result<usize> {
    match self.0.get(key) {
      None => Ok(0),
      Some(value) => value
        .as_u64()
        .and_then(|count| usize::try_from(count).ok())
        .with_context(|| format!("`{key}` is {value}, not a count")),
    }
  }

  fn required(&self, key: &str) -> anyhow::Result<String> {
    self.text(key)?.with_context(|| format!("`{key}` is missing"))
  }
}

fn field_change(name: &str) -> anyhow::Result<FieldChange> {
  [
    FieldChange::Unchanged,
    FieldChange::Modified,
    FieldChange::Added,
    FieldChange::Removed,
    FieldChange::Restored,
  ]
  .into_iter()
  .find(|change| change_name(*change) == name)
  .with_context(|| format!("`{name}` is not a field change"))
}

fn quantity_change(name: &str) -> anyhow::Result<QuantityChange> {
  [
    QuantityChange::Unchanged,
    QuantityChange::Modified,
    QuantityChange::Incomparable,
    QuantityChange::Added,
    QuantityChange::Removed,
    QuantityChange::Restored,
  ]
  .into_iter()
  .find(|change| quantity_change_name(*change) == name)
  .with_context(|| format!("`{name}` is not a quantity change"))
}

/// Puts a `pair_reason` line back together. The units of an incomparable quantity are on its `pair_quantity`
/// line, which is why the quantities are needed.
fn reason(event: &Event<'_>, quantities: &[QuantityScore]) -> anyhow::Result<Reason> {
  let code = event.required("code")?;
  let value = event.number("value")?;
  let limit = event.number("limit")?;
  let similarity = || {
    value
      .map(narrow)
      .with_context(|| format!("a `{code}` reason has no `value`"))
  };
  let held_to = || limit.with_context(|| format!("a `{code}` reason has no `limit`"));
  let subject = || event.required("field");
  Ok(match code.as_str() {
    "below_threshold" => Reason::BelowThreshold {
      similarity: similarity()?,
      threshold: narrow(held_to()?),
    },
    "field_below_floor" => Reason::FieldBelowFloor {
      field: subject()?,
      similarity: similarity()?,
      floor: narrow(held_to()?),
    },
    "anchor_drift" => Reason::AnchorDrift {
      similarity: similarity()?,
      threshold: narrow(held_to()?),
    },
    "quantity_beyond_factor" => Reason::QuantityBeyondFactor {
      field: subject()?,
      factor: value,
      limit: held_to()?,
    },
    "quantity_incomparable" => {
      let field = subject()?;
      let score = quantities.iter().find(|score| score.field == field);
      Reason::QuantityIncomparable {
        previous_unit: score.and_then(|score| score.previous_unit.clone()).unwrap_or_default(),
        unit: score.and_then(|score| score.unit.clone()).unwrap_or_default(),
        field,
      }
    }
    "measure_changed" => Reason::MeasureChanged,
    // A code this build does not know means the file is from a newer one, and guessing at a verdict is worse
    // than saying so.
    other => bail!("`{other}` is not a reason this version knows"),
  })
}

/// Everything read about one pair before it is put together.
struct Collected {
  pair: Pair,
  similarity: f32,
  measure_changed: bool,
  fields: Vec<FieldScore>,
  quantities: Vec<QuantityScore>,
  reasons: Vec<serde_json::Map<String, Value>>,
}

fn collected_pair(event: &Event<'_>) -> anyhow::Result<Collected> {
  let label_text = event.required("label")?;
  let mut before = RecordFields::default();
  if let Some(category) = event.text("category")? {
    before.texts.insert("category".to_owned(), category);
  }
  Ok(Collected {
    pair: Pair {
      id: event.required("pair_id")?,
      label: match label_text.as_str() {
        "edit" => Some(Label::Edit),
        "recycle" => Some(Label::Recycle),
        _ => None,
      },
      label_text,
      note: event.text("note")?.unwrap_or_default(),
      split: event.text("split")?,
      owner: event.text("owner")?,
      variant_change: event.text("variant_change")?,
      label_source: event.text("label_source")?,
      label_confidence: event.text("label_confidence")?,
      before,
      after: RecordFields::default(),
    },
    similarity: narrow(event.number("similarity")?.context("`similarity` is missing")?),
    measure_changed: event.flag("measure_changed")?,
    fields: Vec::new(),
    quantities: Vec::new(),
    reasons: Vec::new(),
  })
}

/// The state of reading a run: what has been seen so far.
#[derive(Default)]
struct Reader {
  info: Option<RunInfo>,
  input: Input,
  fields: Vec<String>,
  quantities: Vec<String>,
  /// Pair ids in the order their `pair` lines came.
  order: Vec<String>,
  pairs: BTreeMap<String, Collected>,
}

impl Reader {
  /// The pair an item line belongs to, which must already have been seen.
  fn pair_of(&mut self, event: &Event<'_>) -> anyhow::Result<&mut Collected> {
    let id = event.required("pair_id")?;
    match self.pairs.get_mut(&id) {
      Some(collected) => Ok(collected),
      None => bail!("an item of pair `{id}` comes before the pair itself"),
    }
  }

  fn header(&mut self, event: &Event<'_>) -> anyhow::Result<()> {
    self.input.sha256 = event.required("pairs_sha256")?;
    self.input.split = event.text("split")?;
    self.input.min_confidence = event
      .text("min_confidence")?
      .map(|text| text.parse::<Confidence>())
      .transpose()
      .map_err(anyhow::Error::msg)?;
    self.info = Some(RunInfo {
      fingerprint: event.required("fingerprint")?,
      threshold: narrow(event.number("threshold")?.context("`threshold` is missing")?),
      anchor_threshold: event.number("anchor_threshold")?.map(narrow),
      reject_on_measure_change: event.flag("reject_on_measure_change")?,
      fields: Vec::new(),
      quantities: Vec::new(),
    });
    Ok(())
  }

  /// Takes in one result event. Events this version has no use for (`slice`, `field_range`, and the like) are
  /// passed over: everything they say is derived from the pairs, and is derived again when the run is used.
  fn take(&mut self, message: &str, object: &serde_json::Map<String, Value>) -> anyhow::Result<()> {
    let event = Event(object);
    match message {
      "calibration" => self.header(&event)?,
      "schema_field" => self.fields.push(event.required("field")?),
      "schema_quantity" => self.quantities.push(event.required("quantity")?),
      "pair" => {
        let collected = collected_pair(&event)?;
        let id = collected.pair.id.clone();
        if self.pairs.insert(id.clone(), collected).is_some() {
          bail!("pair `{id}` appears twice");
        }
        self.order.push(id);
      }
      "pair_field" => {
        let score = FieldScore {
          field: event.required("field")?,
          weight: narrow(event.number("weight")?.unwrap_or(0.0)),
          change: field_change(&event.required("change")?)?,
          similarity: event.number("similarity")?.map(narrow),
          below_floor: event.flag("below_floor")?,
        };
        self.pair_of(&event)?.fields.push(score);
      }
      "pair_quantity" => {
        let score = QuantityScore {
          field: event.required("quantity")?,
          change: quantity_change(&event.required("change")?)?,
          factor: event.number("factor")?,
          zero_crossing: event.flag("zero_crossing")?,
          incomparable: event.flag("incomparable")?,
          previous_unit: event.text("previous_unit")?,
          unit: event.text("unit")?,
          beyond_limit: event.flag("beyond_limit")?,
        };
        self.pair_of(&event)?.quantities.push(score);
      }
      "pair_reason" => self.pair_of(&event)?.reasons.push(object.clone()),
      "ignored_key" => {
        let reason = match event.required("reason")?.as_str() {
          "the wrong kind of value" => Ignored::WrongKind,
          _ => Ignored::NotInSchema,
        };
        self
          .input
          .ignored
          .insert((event.required("key")?, reason), event.count("count")?);
      }
      "summary" => {
        self.input.skipped_split = event.count("skipped_split")?;
        self.input.skipped_confidence = event.count("skipped_confidence")?;
      }
      _ => {}
    }
    Ok(())
  }

  fn finish(mut self) -> anyhow::Result<Run> {
    let Some(mut info) = self.info else {
      bail!("there is no `calibration` line: this is not the output of `calibrate --ndjson`");
    };
    info.fields = self.fields;
    info.quantities = self.quantities;

    let mut scored = Vec::with_capacity(self.order.len());
    for id in self.order {
      let Some(collected) = self.pairs.remove(&id) else {
        continue;
      };
      let reasons = collected
        .reasons
        .iter()
        .map(|object| reason(&Event(object), &collected.quantities))
        .collect::<anyhow::Result<Vec<Reason>>>()
        .with_context(|| format!("pair `{id}`"))?;
      scored.push(ScoredPair {
        pair: collected.pair,
        evaluation: Evaluation {
          comparison: Comparison {
            similarity: collected.similarity,
            fields: collected.fields,
          },
          anchor_similarity: None,
          reasons,
          quantities: collected.quantities,
          measure_changed: collected.measure_changed,
        },
      });
    }
    Ok(Run {
      info,
      calibration: summarize(scored, self.input),
    })
  }
}

/// Reads the events of one `calibrate --ndjson` run.
///
/// Lines that are not result events are skipped, so a file that captured the log stream as well still reads.
///
/// # Errors
///
/// Returns an error naming the line if a line is not a JSON object, if an event lacks what it must have or has
/// it in the wrong type, or if a pair's item arrives before its `pair` line; and an error if there is no
/// `calibration` line at all.
pub fn read(ndjson: &str) -> anyhow::Result<Run> {
  let mut reader = Reader::default();
  for (index, line) in ndjson.lines().enumerate().filter(|(_, line)| !line.trim().is_empty()) {
    let number = index.saturating_add(1);
    let value: Value = serde_json::from_str(line).with_context(|| format!("line {number} is not JSON"))?;
    let Some(object) = value.as_object() else {
      bail!("line {number} is not a JSON object");
    };
    if object.get("target").and_then(Value::as_str) != Some(RESULT) {
      continue;
    }
    if let Some(message) = object.get("message").and_then(Value::as_str) {
      reader.take(message, object).with_context(|| format!("line {number}"))?;
    }
  }
  reader.finish()
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

  fn written() -> (Calibration, String) {
    let (offline, policy) = offline();
    let pairs = parse_full_pairs(PAIRS, offline.schema(), Selection::default()).unwrap();
    let calibration = calibrate(&offline, pairs).unwrap();
    let captured = similarity_log::testing::capture("info", || {
      tracing::info!("model loaded");
      emit(&offline, &policy, &calibration, 0);
    });
    // Both streams in one file, as `2>&1` would capture them.
    let text = captured
      .logs
      .iter()
      .chain(&captured.results)
      .cloned()
      .collect::<Vec<_>>()
      .join("\n");
    (calibration, text)
  }

  #[test]
  fn a_run_reads_back_as_the_calibration_that_was_written() {
    let (calibration, text) = written();
    let run = read(&text).unwrap();

    assert_eq!(run.info.fields, ["category", "name"]);
    assert_eq!(run.info.quantities, ["weight"]);
    assert!((run.info.threshold - 0.5).abs() < 1e-6);
    assert_eq!(run.info.anchor_threshold, None);
    assert_eq!(run.calibration.input, calibration.input);
    assert_eq!(
      (run.calibration.false_rejects, run.calibration.false_accepts),
      (calibration.false_rejects, calibration.false_accepts)
    );
    assert_eq!(run.calibration.pairs.len(), 3);

    for (read, original) in run.calibration.pairs.iter().zip(&calibration.pairs) {
      // Everything about the pair but its records, which the events do not carry.
      assert_eq!(read.pair.id, original.pair.id);
      assert_eq!(
        (
          read.pair.label,
          &read.pair.owner,
          &read.pair.split,
          &read.pair.variant_change
        ),
        (
          original.pair.label,
          &original.pair.owner,
          &original.pair.split,
          &original.pair.variant_change
        )
      );
      assert_eq!(read.pair.category(), original.pair.category());
      assert!((read.similarity() - original.similarity()).abs() < 1e-6);
      assert_eq!(read.evaluation.accepted(), original.evaluation.accepted());
      assert_eq!(read.evaluation.quantities, original.evaluation.quantities);
      assert_eq!(read.evaluation.measure_changed, original.evaluation.measure_changed);
      let codes = |evaluation: &Evaluation| evaluation.reasons.iter().map(Reason::code).collect::<Vec<_>>();
      assert_eq!(codes(&read.evaluation), codes(&original.evaluation));
      for (field, wanted) in read
        .evaluation
        .comparison
        .fields
        .iter()
        .zip(&original.evaluation.comparison.fields)
      {
        assert_eq!(
          (&field.field, field.change, field.below_floor),
          (&wanted.field, wanted.change, wanted.below_floor)
        );
        assert!((field.weight - wanted.weight).abs() < 1e-6);
        match (field.similarity, wanted.similarity) {
          (Some(found), Some(wanted)) => assert!((found - wanted).abs() < 1e-6),
          (found, wanted) => assert_eq!(found, wanted),
        }
      }
    }
  }

  #[test]
  fn a_file_that_is_not_a_run_is_refused_with_the_line() {
    let (_, text) = written();
    let lines: Vec<&str> = text.lines().collect();

    let no_header: String = lines
      .iter()
      .filter(|line| !line.contains("\"message\":\"calibration\""))
      .copied()
      .collect::<Vec<_>>()
      .join("\n");
    let error = read(&no_header).unwrap_err().to_string();
    assert!(error.contains("no `calibration` line"), "{error}");

    assert!(read("not json at all").unwrap_err().to_string().contains("line 1"));
    assert!(read("[1, 2]").unwrap_err().to_string().contains("line 1"));

    // An item whose pair never appeared: the file was cut, or put together by hand.
    let orphan = r#"{"target":"result","message":"pair_field","pair_id":"zz","field":"name","change":"modified"}"#;
    let error = format!("{:#}", read(&format!("{}\n{orphan}", lines[1])).unwrap_err());
    assert!(
      error.contains("line 2") && error.contains("before the pair itself"),
      "{error}"
    );

    let from_the_future = text.replace("\"code\":\"field_below_floor\"", "\"code\":\"field_too_purple\"");
    let error = format!("{:#}", read(&from_the_future).unwrap_err());
    assert!(error.contains("field_too_purple"), "{error}");

    let wrong_type = text.replace("\"measure_changed\":false", "\"measure_changed\":\"no\"");
    assert!(
      read(&wrong_type).is_err(),
      "a flag written as text is not this tool's output"
    );
  }
}
