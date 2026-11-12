//! A policy sweep: one recorded calibration, judged again under every combination of a grid of settings.
//!
//! Weights, field floors, and quantity limits change what is made of a calibration's measurements, not the
//! measurements. So a grid of them costs nothing to try: no model is loaded, and each point is the calibration
//! judged again (`Evaluation::rejudge`), as `calibrate-diff --config-b` does for one.
//!
//! Two settings are not swept. The overall threshold is already searched exactly for every point: each point is
//! judged at the threshold that is best for it. The drift threshold cannot be chosen on pairs at all (D27).
//!
//! **A sweep fits its settings to the pairs it ranks on.** The errors it reports for its best points are
//! optimistic, the more so the larger the grid. `--evaluate` judges the best points on a second run, of other
//! pairs, at the threshold chosen on the first: that number, and only that one, is not fitted.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use anyhow::{Context, bail};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use similarity_core::record::hex_prefix;
use similarity_log::RESULT;

use super::Suggestion;
use super::diff;
use super::events::{Run, six, six_wide};
use super::pairs::Label;
use super::stats::{self, ErrorRates};
use crate::config::Config;

/// One value on an axis that may also be switched off: a floor, or a factor limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Setting {
  /// No floor, or no limit. Written `"off"`: TOML has no null.
  Off,
  /// The value.
  Value(f64),
}

impl<'de> Deserialize<'de> for Setting {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use serde::de::Error as _;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
      Number(f64),
      Text(String),
    }
    match Raw::deserialize(deserializer)? {
      Raw::Number(value) => Ok(Self::Value(value)),
      Raw::Text(text) if text == "off" => Ok(Self::Off),
      Raw::Text(text) => Err(D::Error::custom(format!(
        "`{text}` is not a number; the only word allowed here is \"off\""
      ))),
    }
  }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldAxes {
  #[serde(default)]
  weight: Vec<f64>,
  #[serde(default)]
  reject_below: Vec<Setting>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantityAxes {
  #[serde(default)]
  reject_above_factor: Vec<Setting>,
  #[serde(default)]
  reject_incomparable: Vec<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyAxes {
  #[serde(default)]
  reject_on_measure_change: Vec<bool>,
  // Named so that asking for them gets the reason and not "unknown field".
  #[serde(default)]
  threshold: Option<toml::Value>,
  #[serde(default)]
  anchor_threshold: Option<toml::Value>,
}

/// What a grid file says: for each setting it names, the values to try.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grid {
  #[serde(default)]
  fields: BTreeMap<String, FieldAxes>,
  #[serde(default)]
  quantities: BTreeMap<String, QuantityAxes>,
  #[serde(default)]
  policy: PolicyAxes,
}

/// One setting of one point.
#[derive(Debug, Clone, PartialEq)]
pub enum Change {
  /// A field's weight.
  Weight(String, f64),
  /// A field's `reject_below`.
  Floor(String, Setting),
  /// A quantity's `reject_above_factor`.
  FactorLimit(String, Setting),
  /// A quantity's `reject_incomparable`.
  Incomparable(String, bool),
  /// The policy's `reject_on_measure_change`.
  MeasureChange(bool),
}

impl Change {
  /// The setting's name as a config key, and what it is a setting of.
  fn key(&self) -> (&'static str, Option<&str>) {
    match self {
      Self::Weight(field, _) => ("weight", Some(field)),
      Self::Floor(field, _) => ("reject_below", Some(field)),
      Self::FactorLimit(quantity, _) => ("reject_above_factor", Some(quantity)),
      Self::Incomparable(quantity, _) => ("reject_incomparable", Some(quantity)),
      Self::MeasureChange(_) => ("reject_on_measure_change", None),
    }
  }

  /// Its value as a number, where it is one that is set.
  fn number(&self) -> Option<f64> {
    match self {
      Self::Weight(_, value) | Self::Floor(_, Setting::Value(value)) | Self::FactorLimit(_, Setting::Value(value)) => {
        Some(*value)
      }
      _ => None,
    }
  }

  /// Its value as a switch, where it is one.
  fn flag(&self) -> Option<bool> {
    match self {
      Self::Incomparable(_, value) | Self::MeasureChange(value) => Some(*value),
      _ => None,
    }
  }

  fn describe(&self) -> String {
    let (name, subject) = self.key();
    let value = match (self.number(), self.flag()) {
      (Some(number), _) => format!("{number}"),
      (None, Some(flag)) => flag.to_string(),
      (None, None) => "off".to_owned(),
    };
    match subject {
      Some(subject) => format!("{subject}.{name} = {value}"),
      None => format!("{name} = {value}"),
    }
  }
}

/// One combination of the grid's values.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
  /// The first twelve hex digits of a hash of the configuration the point amounts to, so that the same point has
  /// the same id in every sweep.
  pub id: String,
  /// What it sets, one entry per axis of the grid.
  pub changes: Vec<Change>,
}

/// Parses a grid file against the configuration it varies.
///
/// # Errors
///
/// Returns an error if the file is not a grid, if it names a field or quantity the configuration does not have,
/// or if it tries to sweep the overall or the drift threshold.
pub fn parse_grid(text: &str, base: &Config) -> anyhow::Result<Grid> {
  let grid: Grid = toml::from_str(text).context("the grid is not valid")?;
  if grid.policy.threshold.is_some() {
    bail!(
      "`policy.threshold` is not swept: every point is already judged at the threshold that is best for it, which \
       is an exact search"
    );
  }
  if grid.policy.anchor_threshold.is_some() {
    bail!("`policy.anchor_threshold` cannot be chosen on pairs, which have no history; see `calibrate-drift`");
  }
  if let Some(unknown) = grid.fields.keys().find(|name| !base.fields.contains_key(*name)) {
    bail!("the grid names a field `{unknown}` that the configuration does not have");
  }
  if let Some(unknown) = grid.quantities.keys().find(|name| !base.quantities.contains_key(*name)) {
    bail!("the grid names a quantity `{unknown}` that the configuration does not have");
  }
  Ok(grid)
}

/// The axes of a grid: for each, the alternatives a point picks one of. An axis with no values is not an axis.
fn axes(grid: &Grid) -> Vec<Vec<Change>> {
  let mut axes = Vec::new();
  for (field, values) in &grid.fields {
    axes.push(
      values
        .weight
        .iter()
        .map(|weight| Change::Weight(field.clone(), *weight))
        .collect(),
    );
    axes.push(
      values
        .reject_below
        .iter()
        .map(|floor| Change::Floor(field.clone(), *floor))
        .collect(),
    );
  }
  for (quantity, values) in &grid.quantities {
    axes.push(
      values
        .reject_above_factor
        .iter()
        .map(|limit| Change::FactorLimit(quantity.clone(), *limit))
        .collect(),
    );
    axes.push(
      values
        .reject_incomparable
        .iter()
        .map(|reject| Change::Incomparable(quantity.clone(), *reject))
        .collect(),
    );
  }
  axes.push(
    grid
      .policy
      .reject_on_measure_change
      .iter()
      .map(|reject| Change::MeasureChange(*reject))
      .collect(),
  );
  axes.retain(|axis: &Vec<Change>| !axis.is_empty());
  axes
}

fn off_or(setting: Setting) -> Option<f64> {
  match setting {
    Setting::Off => None,
    Setting::Value(value) => Some(value),
  }
}

/// `base` with a point's settings in place of its own.
#[expect(
  clippy::cast_possible_truncation,
  reason = "weights and floors are configured as 32-bit numbers"
)]
pub fn apply(base: &Config, changes: &[Change]) -> Config {
  let mut config = base.clone();
  for change in changes {
    match change {
      Change::Weight(field, weight) => {
        if let Some(field) = config.fields.get_mut(field) {
          field.weight = *weight as f32;
        }
      }
      Change::Floor(field, floor) => {
        if let Some(field) = config.fields.get_mut(field) {
          field.reject_below = off_or(*floor).map(|floor| floor as f32);
        }
      }
      Change::FactorLimit(quantity, limit) => {
        if let Some(quantity) = config.quantities.get_mut(quantity) {
          quantity.reject_above_factor = off_or(*limit);
        }
      }
      Change::Incomparable(quantity, reject) => {
        if let Some(quantity) = config.quantities.get_mut(quantity) {
          quantity.reject_incomparable = *reject;
        }
      }
      Change::MeasureChange(reject) => config.policy.reject_on_measure_change = *reject,
    }
  }
  config
}

/// Everything about a configuration that a sweep can change, written so that two configurations that judge
/// alike are written alike: weights only matter relative to each other, so they are divided by their sum.
fn canonical(config: &Config) -> String {
  let total: f64 = config.fields.values().map(|field| f64::from(field.weight)).sum();
  let mut text = String::new();
  // Writing to a `String` cannot fail.
  for (name, field) in &config.fields {
    let share = if total > 0.0 {
      f64::from(field.weight) / total
    } else {
      0.0
    };
    let _ = write!(text, "{name}:{share:.9}:{:?};", field.reject_below);
  }
  for (name, quantity) in &config.quantities {
    let _ = write!(
      text,
      "{name}:{:?}:{};",
      quantity.reject_above_factor, quantity.reject_incomparable
    );
  }
  let _ = write!(text, "{}", config.policy.reject_on_measure_change);
  text
}

/// Every point of the grid: each combination of one value per axis, in the order the axes and their values were
/// written. Points that amount to the same configuration (weights in the same proportions) are counted once.
///
/// Returns the points and how many combinations were left out as duplicates of an earlier one.
///
/// # Errors
///
/// Returns an error if the grid has more than `max` combinations, with the count, before anything is judged.
pub fn points(grid: &Grid, base: &Config, max: usize) -> anyhow::Result<(Vec<Point>, usize)> {
  let axes = axes(grid);
  let combinations = axes
    .iter()
    .try_fold(1_usize, |count, axis| count.checked_mul(axis.len()))
    .filter(|count| *count <= max);
  let Some(combinations) = combinations else {
    let sizes: Vec<String> = axes.iter().map(|axis| axis.len().to_string()).collect();
    bail!(
      "the grid has more than {max} points ({}); narrow an axis, or raise --max-points",
      sizes.join(" x ")
    );
  };
  let mut seen = BTreeSet::new();
  let (mut points, mut collapsed) = (Vec::new(), 0_usize);
  for number in 0..combinations {
    // The combination's digits, one per axis, the last axis turning fastest.
    let mut rest = number;
    let mut changes = Vec::with_capacity(axes.len());
    for axis in axes.iter().rev() {
      let size = axis.len().max(1);
      if let Some(change) = axis.get(rest.rem_euclid(size)) {
        changes.push(change.clone());
      }
      rest = rest.div_euclid(size);
    }
    changes.reverse();
    let key = canonical(&apply(base, &changes));
    if seen.insert(key.clone()) {
      points.push(Point {
        id: hex_prefix(&Sha256::digest(key.as_bytes()), 6),
        changes,
      });
    } else {
      collapsed = collapsed.saturating_add(1);
    }
  }
  Ok((points, collapsed))
}

/// A point, judged.
#[derive(Debug, Clone, PartialEq)]
pub struct Judged {
  /// The point.
  pub point: Point,
  /// The overall threshold that is best for it on the selection run, and how the two labels sit around it.
  pub suggestion: Option<Suggestion>,
  /// Edits wrongly rejected and recycles wrongly accepted under its whole policy at that threshold: floors and
  /// limits reject on their own, so this can be more than the suggestion's own count.
  pub errors: (usize, usize),
  /// The error rates with their intervals, for the points that are reported.
  pub rates: Option<ErrorRates>,
  /// On the evaluation run, at the threshold chosen on the selection run: wrongly rejected, wrongly accepted.
  pub evaluated: Option<(usize, usize)>,
}

impl Judged {
  fn total(&self) -> usize {
    self.errors.0.saturating_add(self.errors.1)
  }
}

/// A sweep's result.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicySweep {
  /// The configuration as it is, with no point applied: what the points are to beat.
  pub base: Judged,
  /// The best points, best first.
  pub top: Vec<Judged>,
  /// How many points were judged.
  pub points: usize,
  /// How many combinations were the same configuration as an earlier one.
  pub collapsed: usize,
  /// Points that are not a valid configuration (a weight of zero, a floor outside the cosine range), with why.
  pub invalid: Vec<(Point, String)>,
}

/// Judges one point on the selection run: at the threshold that is best for it, under its whole policy.
fn judge(select: &Run, base: &Config, point: &Point) -> anyhow::Result<(Judged, Config)> {
  let config = apply(base, &point.changes);
  let (schema, mut policy) = (config.schema()?, config.policy()?);
  // The first pass applies the weights, which is what moves the overall similarities and so the best threshold.
  let weighed = diff::rejudge(select, &schema, &policy)?;
  if let Some(suggestion) = weighed.suggestion {
    policy.threshold = suggestion.threshold;
  }
  let judged = diff::rejudge(select, &schema, &policy)?;
  let mut config = config;
  config.policy.threshold = policy.threshold;
  Ok((
    Judged {
      point: point.clone(),
      suggestion: weighed.suggestion,
      errors: (judged.false_rejects, judged.false_accepts),
      rates: None,
      evaluated: None,
    },
    config,
  ))
}

/// Fills in what is only worked out for a point that is reported: its intervals, and its errors on the
/// evaluation run at the threshold it was given on the selection run.
fn report_on(
  judged: &mut Judged,
  config: &Config,
  select: &Run,
  evaluate: Option<&Run>,
  seed: u64,
) -> anyhow::Result<()> {
  let (schema, policy) = (config.schema()?, config.policy()?);
  judged.rates = Some(stats::error_rates(&diff::rejudge(select, &schema, &policy)?, seed));
  if let Some(evaluate) = evaluate {
    let other = diff::rejudge(evaluate, &schema, &policy)?;
    judged.evaluated = Some((other.false_rejects, other.false_accepts));
  }
  Ok(())
}

/// Judges every point of `grid` on `select` and keeps the best `top`.
///
/// Points are ranked by how many pairs they get wrong, then by the margin around their threshold (wider first),
/// then by the order they were generated in, so that the ranking never depends on anything but the grid.
///
/// # Errors
///
/// Returns an error if the grid is too large, if the configuration's fields and quantities are not the ones the
/// runs were scored with, or if `evaluate` was scored with another model.
pub fn sweep(
  select: &Run,
  evaluate: Option<&Run>,
  base: &Config,
  grid: &Grid,
  top: usize,
  max_points: usize,
  seed: u64,
) -> anyhow::Result<PolicySweep> {
  if let Some(evaluate) = evaluate
    && evaluate.info.fingerprint != select.info.fingerprint
  {
    bail!(
      "the evaluation run was scored with model {} and the selection run with {}; a threshold does not carry from \
       one model to another",
      evaluate.info.fingerprint,
      select.info.fingerprint
    );
  }
  let (points, collapsed) = points(grid, base, max_points)?;
  let reference = Point {
    id: hex_prefix(&Sha256::digest(canonical(base).as_bytes()), 6),
    changes: Vec::new(),
  };
  let (mut base_judged, base_config) = judge(select, base, &reference)?;
  report_on(&mut base_judged, &base_config, select, evaluate, seed)?;

  let (mut judged, mut invalid) = (Vec::new(), Vec::new());
  let count = points.len();
  for point in points {
    match judge(select, base, &point) {
      Ok(found) => judged.push(found),
      Err(error) => invalid.push((point, format!("{error:#}"))),
    }
  }
  // A stable sort, so that equal points stay in the order they were generated.
  judged.sort_by(|(a, _), (b, _)| {
    let margin = |judged: &Judged| {
      judged
        .suggestion
        .map_or(f32::NEG_INFINITY, |suggestion| suggestion.margin)
    };
    a.total().cmp(&b.total()).then_with(|| margin(b).total_cmp(&margin(a)))
  });
  judged.truncate(top);
  let mut best = Vec::with_capacity(judged.len());
  for (mut found, config) in judged {
    report_on(&mut found, &config, select, evaluate, seed)?;
    best.push(found);
  }
  Ok(PolicySweep {
    base: base_judged,
    top: best,
    points: count,
    collapsed,
    invalid,
  })
}

fn render_row(name: &str, judged: &Judged, out: &mut String) {
  let (threshold, margin) = judged.suggestion.map_or_else(
    || ("-".to_owned(), "-".to_owned()),
    |suggestion| {
      (
        format!("{:.3}", suggestion.threshold),
        format!("{:.3}", suggestion.margin),
      )
    },
  );
  let evaluated = judged.evaluated.map_or_else(String::new, |(rejects, accepts)| {
    format!("   {rejects:>5} {accepts:>5}")
  });
  let changes: Vec<String> = judged.point.changes.iter().map(Change::describe).collect();
  // Writing to a `String` cannot fail.
  let _ = writeln!(
    out,
    "{name:<5} {threshold:>9} {margin:>7} {:>8} {:>8}{evaluated}   {}",
    judged.errors.0,
    judged.errors.1,
    if changes.is_empty() {
      "the configuration as it is".to_owned()
    } else {
      changes.join(", ")
    }
  );
}

/// Renders a sweep as a report, for a terminal.
pub fn render_sweep(sweep: &PolicySweep, evaluated: bool) -> String {
  let mut out = String::new();
  let _ = writeln!(
    out,
    "points {} judged ({} more were the same configuration as an earlier one, {} are not valid)\n",
    sweep.points,
    sweep.collapsed,
    sweep.invalid.len()
  );
  let other = if evaluated { "   on the evaluation run" } else { "" };
  let _ = writeln!(
    out,
    "{:<5} {:>9} {:>7} {:>8} {:>8}{other}",
    "", "threshold", "margin", "rejected", "accepted"
  );
  render_row("now", &sweep.base, &mut out);
  for (rank, judged) in sweep.top.iter().enumerate() {
    render_row(&format!("{}", rank.saturating_add(1)), judged, &mut out);
  }
  let _ = writeln!(
    out,
    "\nrejected: edits wrongly rejected. accepted: recycles wrongly accepted. Each point is judged at the overall \
     threshold that is best for it, with its floors and limits."
  );
  let _ = writeln!(
    out,
    "These settings were fitted to the pairs they are ranked on, so the counts above are optimistic{}.",
    if evaluated {
      "; the counts on the evaluation run, at the same threshold, are not"
    } else {
      ". Judge the best points on pairs that chose nothing (--evaluate) before believing them"
    }
  );
  if let Some(best) = sweep.top.first().filter(|best| best.total() < sweep.base.total()) {
    let _ = writeln!(
      out,
      "\nthe first point gets {} pair(s) wrong where the configuration as it is gets {}. To try, not to paste: put \
       these in a copy of the config, run `calibrate` under it, and compare with `calibrate-diff`.",
      best.total(),
      sweep.base.total()
    );
    for change in &best.point.changes {
      let _ = writeln!(out, "  {}", change.describe());
    }
    if let Some(suggestion) = best.suggestion {
      let _ = writeln!(out, "  policy.threshold = {:.3}", suggestion.threshold);
    }
  } else {
    let _ = writeln!(
      out,
      "\nno point gets fewer pairs wrong than the configuration as it is."
    );
    let margin = |judged: &Judged| judged.suggestion.map(|suggestion| suggestion.margin);
    if let Some(best) = sweep.top.first()
      && let (Some(wider), Some(now)) = (margin(best), margin(&sweep.base))
      && best.total() == sweep.base.total()
      && wider > now
    {
      let _ = writeln!(
        out,
        "The first point gets as many wrong with more room around its threshold (a margin of {wider:.3} against \
         {now:.3}). On this few pairs that is a hint, not a result."
      );
    }
  }
  for (point, reason) in sweep.invalid.iter().take(5) {
    let changes: Vec<String> = point.changes.iter().map(Change::describe).collect();
    let _ = writeln!(out, "not valid: {} ({reason})", changes.join(", "));
  }
  out
}

fn emit_point(rank: Option<usize>, judged: &Judged) {
  let point_id = judged.point.id.as_str();
  let rate = |pick: fn(&ErrorRates) -> &stats::Rate| judged.rates.as_ref().map(pick);
  let (rejects, accepts) = (rate(|rates| &rates.false_reject), rate(|rates| &rates.false_accept));
  let interval = |rate: Option<&stats::Rate>| rate.and_then(|rate| rate.interval);
  tracing::info!(
    target: RESULT,
    point_id,
    rank,
    threshold = judged.suggestion.map(|suggestion| six(suggestion.threshold)),
    margin = judged.suggestion.map(|suggestion| six(suggestion.margin)),
    false_rejects = judged.errors.0,
    false_accepts = judged.errors.1,
    false_reject_rate = rejects.and_then(|rate| rate.rate).map(six_wide),
    false_reject_low = interval(rejects).map(|interval| six_wide(interval.low)),
    false_reject_high = interval(rejects).map(|interval| six_wide(interval.high)),
    false_accept_rate = accepts.and_then(|rate| rate.rate).map(six_wide),
    false_accept_low = interval(accepts).map(|interval| six_wide(interval.low)),
    false_accept_high = interval(accepts).map(|interval| six_wide(interval.high)),
    evaluate_false_rejects = judged.evaluated.map(|errors| errors.0),
    evaluate_false_accepts = judged.evaluated.map(|errors| errors.1),
    "policy_point"
  );
  for change in &judged.point.changes {
    let (setting, subject) = change.key();
    tracing::info!(
      target: RESULT,
      point_id,
      setting,
      subject,
      value = change.number(),
      flag = change.flag(),
      "policy_point_setting"
    );
  }
}

/// Writes a sweep as result events: a `policy_sweep` line, then a `policy_point` line for the configuration as
/// it is (with no `rank`) and for each of the best points, each followed by one `policy_point_setting` line per
/// setting it changes. A setting that is switched off has neither a `value` nor a `flag`.
pub fn emit(sweep: &PolicySweep, select: &Run, evaluate: Option<&Run>, seed: u64) {
  let labeled = |run: &Run| {
    run
      .calibration
      .labeled(Label::Edit)
      .count()
      .saturating_add(run.calibration.labeled(Label::Recycle).count())
  };
  tracing::info!(
    target: RESULT,
    fingerprint = select.info.fingerprint.as_str(),
    pairs = labeled(select),
    evaluate_pairs = evaluate.map(labeled),
    points = sweep.points,
    collapsed = sweep.collapsed,
    invalid = sweep.invalid.len(),
    seed,
    "policy_sweep"
  );
  emit_point(None, &sweep.base);
  for (rank, judged) in sweep.top.iter().enumerate() {
    emit_point(Some(rank.saturating_add(1)), judged);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::offline::events;

  const CONFIG: &str = "[policy]\nthreshold = 0.6\n[fields.name]\nweight = 1.0\n[fields.category]\nweight = 1.0\n";

  fn config() -> Config {
    Config::parse(CONFIG).unwrap()
  }

  /// A calibration written by hand: each pair's similarity per field, so that the weights decide the order.
  /// Under equal weights the second edit scores below the recycle; with the name weighted up, both edits clear it.
  fn run(fingerprint: &str) -> Run {
    let pair = |id: &str, label: &str, name: f64, category: f64| {
      let field = |field: &str, similarity: f64| {
        format!(
          r#"{{"message":"pair_field","pair_id":"{id}","field":"{field}","change":"modified","similarity":{similarity},"weight":1.0,"below_floor":false,"target":"result"}}"#
        )
      };
      [
        format!(
          r#"{{"message":"pair","pair_id":"{id}","label":"{label}","similarity":{},"accepted":true,"wrong":false,"measure_changed":false,"target":"result"}}"#,
          f64::midpoint(name, category)
        ),
        field("category", category),
        field("name", name),
      ]
      .join("\n")
    };
    let lines = [
      format!(
        r#"{{"message":"calibration","fingerprint":"{fingerprint}","dimension":8,"max_tokens":16,"threshold":0.6,"pairs_sha256":"aa","pairs":3,"seed":0,"target":"result"}}"#
      ),
      r#"{"message":"schema_field","field":"category","weight":1.0,"required":false,"target":"result"}"#.to_owned(),
      r#"{"message":"schema_field","field":"name","weight":1.0,"required":false,"target":"result"}"#.to_owned(),
      pair("e1", "edit", 0.9, 0.5),
      pair("e2", "edit", 0.95, 0.2),
      pair("r1", "recycle", 0.4, 0.95),
    ];
    events::read(&lines.join("\n")).unwrap()
  }

  fn grid(text: &str) -> Grid {
    parse_grid(text, &config()).unwrap()
  }

  #[test]
  fn a_grid_is_every_combination_of_its_axes_in_the_order_written() {
    let grid = grid(
      "[fields.name]\nweight = [1.0, 4.0]\nreject_below = [\"off\", 0.5]\n[policy]\nreject_on_measure_change = [false, true]\n",
    );
    let (points, collapsed) = points(&grid, &config(), 100).unwrap();
    assert_eq!((points.len(), collapsed), (8, 0));
    assert_eq!(
      points[0].changes,
      [
        Change::Weight("name".to_owned(), 1.0),
        Change::Floor("name".to_owned(), Setting::Off),
        Change::MeasureChange(false)
      ]
    );
    assert_eq!(
      points[1].changes[2],
      Change::MeasureChange(true),
      "the last axis turns fastest"
    );
    assert_eq!(points[7].changes[0], Change::Weight("name".to_owned(), 4.0));
    let mut ids: Vec<&str> = points.iter().map(|point| point.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 8, "each point has an id of its own");
    assert_eq!(
      super::points(&grid, &config(), 100).unwrap().0,
      points,
      "and the same one every time"
    );

    let error = super::points(&grid, &config(), 7).unwrap_err().to_string();
    assert!(
      error.contains("more than 7 points") && error.contains("2 x 2 x 2"),
      "{error}"
    );
  }

  /// Weights only matter relative to each other: doubling every one of them is the same configuration.
  #[test]
  fn points_with_weights_in_the_same_proportions_are_one_point() {
    let grid = grid("[fields.name]\nweight = [1.0, 2.0]\n[fields.category]\nweight = [1.0, 2.0]\n");
    let (points, collapsed) = points(&grid, &config(), 100).unwrap();
    assert_eq!(
      (points.len(), collapsed),
      (3, 1),
      "1:1 and 2:2 are one; 1:2 and 2:1 are not"
    );
  }

  #[test]
  fn a_grid_that_is_not_one_is_refused_with_the_reason() {
    let refused = |text: &str| format!("{:#}", parse_grid(text, &config()).unwrap_err());
    assert!(refused("[policy]\nthreshold = [0.8]\n").contains("already judged at the threshold that is best"));
    assert!(refused("[policy]\nanchor_threshold = [0.8]\n").contains("calibrate-drift"));
    assert!(refused("[fields.nmae]\nweight = [1.0]\n").contains("field `nmae`"));
    assert!(refused("[quantities.weight]\nreject_above_factor = [2.0]\n").contains("quantity `weight`"));
    assert!(refused("[fields.name]\nwieght = [1.0]\n").contains("wieght"));
    assert!(refused("[fields.name]\nreject_below = [\"none\"]\n").contains("\"off\""));
  }

  #[test]
  fn points_are_ranked_by_errors_at_their_own_best_threshold_then_by_margin() {
    let select = run("m1");
    let grid = grid("[fields.name]\nweight = [1.0, 2.0, 4.0]\n");
    let sweep = sweep(&select, None, &config(), &grid, 10, 100, 0).unwrap();
    assert_eq!((sweep.points, sweep.collapsed, sweep.invalid.len()), (3, 0, 0));
    assert_eq!(
      sweep.base.point.changes,
      [],
      "the configuration as it is comes first, unranked"
    );
    assert_eq!(
      sweep.base.errors.0 + sweep.base.errors.1,
      1,
      "equal weights cannot separate all three"
    );

    let weights: Vec<f64> = sweep
      .top
      .iter()
      .filter_map(|judged| judged.point.changes[0].number())
      .collect();
    assert_eq!(weights, [4.0, 2.0, 1.0], "fewest errors first, then the wider margin");
    let best = &sweep.top[0];
    assert_eq!(best.errors, (0, 0));
    // (0.4 * 4 + 0.95) / 5 = 0.51 for the recycle and (0.95 * 4 + 0.2) / 5 = 0.8 for the lower edit.
    let suggestion = best.suggestion.unwrap();
    assert!(
      (suggestion.threshold - 0.655).abs() < 1e-4 && (suggestion.margin - 0.145).abs() < 1e-4,
      "{suggestion:?}"
    );
    assert!(best.rates.is_some() && best.evaluated.is_none());
    assert_eq!(
      sweep.top[2].point.id, sweep.base.point.id,
      "the point that changes nothing is the configuration as it is"
    );

    let only_one = super::sweep(&select, None, &config(), &grid, 1, 100, 0).unwrap();
    assert_eq!(
      (only_one.top.len(), only_one.points),
      (1, 3),
      "every point was judged; one is reported"
    );
  }

  /// The threshold chosen on the selection run is carried to the evaluation run, never chosen again there.
  #[test]
  fn the_best_points_are_judged_on_other_pairs_at_the_threshold_already_chosen() {
    let grid = grid("[fields.name]\nweight = [4.0]\n");
    let sweep = sweep(&run("m1"), Some(&run("m1")), &config(), &grid, 10, 100, 0).unwrap();
    assert_eq!(sweep.top[0].evaluated, Some((0, 0)));
    assert_eq!(sweep.base.evaluated.map(|errors| errors.0 + errors.1), Some(1));

    let error = super::sweep(&run("m1"), Some(&run("m2")), &config(), &grid, 10, 100, 0)
      .unwrap_err()
      .to_string();
    assert!(error.contains("does not carry from one model to another"), "{error}");
  }

  #[test]
  fn a_point_that_is_not_a_valid_configuration_is_reported_and_the_rest_are_judged() {
    let grid = grid("[fields.name]\nweight = [0.0, 4.0]\nreject_below = [\"off\", 2.0]\n");
    let sweep = sweep(&run("m1"), None, &config(), &grid, 10, 100, 0).unwrap();
    assert_eq!(sweep.points, 4);
    assert_eq!(sweep.invalid.len(), 3, "a weight of zero, a floor of two, and both");
    assert_eq!(sweep.top.len(), 1);
    assert!(sweep.invalid[0].1.contains("weight"), "{}", sweep.invalid[0].1);
  }

  #[test]
  fn the_report_names_the_best_point_and_says_what_its_counts_are_worth() {
    let grid = grid("[fields.name]\nweight = [1.0, 4.0]\nreject_below = [\"off\"]\n");
    let sweep = sweep(&run("m1"), None, &config(), &grid, 10, 100, 0).unwrap();
    let report = render_sweep(&sweep, false);
    assert!(
      report.starts_with("points 2 judged (0 more were the same configuration"),
      "{report}"
    );
    assert!(report.contains("the configuration as it is"), "{report}");
    assert!(report.contains("name.weight = 4, name.reject_below = off"), "{report}");
    assert!(
      report.contains("optimistic") && report.contains("--evaluate"),
      "{report}"
    );
    assert!(
      report.contains("gets 0 pair(s) wrong where the configuration as it is gets 1"),
      "{report}"
    );
    assert!(
      report.contains("  name.weight = 4") && report.contains("  policy.threshold = 0.655"),
      "{report}"
    );
    assert!(render_sweep(&sweep, true).contains("the counts on the evaluation run, at the same threshold, are not"));

    // The same sweep as flat lines: the reference point has no rank, and a setting that is off has no value.
    let captured = similarity_log::testing::capture("off", || emit(&sweep, &run("m1"), None, 0));
    let events: Vec<serde_json::Value> = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    assert_eq!(
      (&events[0]["message"], &events[0]["points"], &events[0]["pairs"]),
      (&"policy_sweep".into(), &2.into(), &3.into())
    );
    assert!(events[1].get("rank").is_none() && events[1]["message"] == "policy_point");
    assert_eq!(
      (&events[2]["rank"], &events[2]["false_rejects"]),
      (&1.into(), &0.into())
    );
    let settings: Vec<&serde_json::Value> = events
      .iter()
      .filter(|event| event["message"] == "policy_point_setting")
      .collect();
    assert_eq!(
      (&settings[0]["setting"], &settings[0]["subject"], &settings[0]["value"]),
      (&"weight".into(), &"name".into(), &4.0.into())
    );
    assert!(
      settings[1].get("value").is_none() && settings[1].get("flag").is_none(),
      "{}",
      settings[1]
    );
    for event in &events {
      let values = event.as_object().unwrap().values();
      assert!(
        values
          .into_iter()
          .all(|value| !value.is_object() && !value.is_array() && !value.is_null())
      );
    }
  }
}
