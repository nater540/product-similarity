//! Recorded runs as one HTML page: what went in, the training curves, and each calibration's scores.
//!
//! A single file with no script and nothing fetched: it opens from disk, attaches to an email, and prints. The
//! charts are inline SVG. A number under the pointer is read from a native tooltip, and every chart is followed
//! by a table of exactly what it draws, so nothing depends on color or on a pointer.
//!
//! The page computes nothing. Training numbers come from [`super::training`], calibration numbers from
//! [`crate::offline::stats`], and this module only lays them out.

use std::fmt::Write as _;

use similarity_runs::Status;

use super::Loaded;
use super::compare::{describe, inputs, outcomes};
use super::training::{StepLine, TrainingRun};
use crate::offline::pairs::Label;
use crate::offline::stats;

/// How many runs one page shows. Past this, the series of a chart cannot be told apart, and the answer is two
/// reports.
pub const MAX_RUNS: usize = 6;

/// How a series is dashed, by its place: color alone must not carry which run a line is.
const DASHES: [&str; MAX_RUNS] = ["", "6 3", "2 3", "8 3 2 3", "1 4", "10 4"];

const WIDTH: f64 = 680.0;
const HEIGHT: f64 = 230.0;
const LEFT: f64 = 58.0;
const RIGHT: f64 = 72.0;
const TOP: f64 = 14.0;
const BOTTOM: f64 = 34.0;

const STYLE: &str = r#"
:root { color-scheme: light dark; --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; }
@media (prefers-color-scheme: dark) { :root { --surface: #1a1a19; --plane: #0d0d0d; --ink: #f0efec;
  --ink-2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; } }
body { background: var(--plane); color: var(--ink); font: 14px/1.45 system-ui, sans-serif; margin: 0; padding: 16px; }
main { max-width: 760px; margin: 0 auto; }
h1 { font-size: 20px; margin: 8px 0 4px; } h2 { font-size: 16px; margin: 28px 0 8px; }
h3 { font-size: 14px; margin: 16px 0 4px; font-weight: 600; }
p, .note { color: var(--ink-2); margin: 4px 0 8px; }
figure { background: var(--surface); margin: 8px 0; padding: 8px 8px 4px; border-radius: 6px; overflow-x: auto; }
svg { display: block; max-width: 100%; height: auto; }
svg text { fill: var(--muted); font: 11px system-ui, sans-serif; }
svg .label { fill: var(--ink-2); }
.grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--axis); stroke-width: 1; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--surface); stroke-width: 2; } .hit { fill: transparent; }
.mark { stroke: var(--ink-2); stroke-width: 1; }
table { border-collapse: collapse; margin: 6px 0; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 2px 12px 2px 0; vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; } td.n, th.n { text-align: right; }
tr.differs td:first-child::before { content: "* "; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 4px 0; color: var(--ink-2); }
.key { display: inline-block; width: 22px; height: 0; border-top: 2px solid; vertical-align: middle; margin-right: 6px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
details { margin: 2px 0 10px; color: var(--ink-2); } summary { cursor: pointer; }
"#;

fn escape(text: &str) -> String {
  text
    .replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// The color of the series in place `index`, as the name of its custom property. Fixed by place, never by rank.
fn color(index: usize) -> String {
  format!("var(--s{})", index.min(MAX_RUNS.saturating_sub(1)).saturating_add(1))
}

/// The letter a run goes by on the page: `a`, `b`, ...
fn letter(index: usize) -> char {
  u8::try_from(index)
    .ok()
    .and_then(|index| b'a'.checked_add(index))
    .map_or('?', char::from)
}

/// Round values to mark an axis with, covering `low` to `high`: steps of 1, 2, or 5 times a power of ten.
fn ticks(low: f64, high: f64) -> Vec<f64> {
  let span = high - low;
  if !(span.is_finite() && span > 0.0) {
    return vec![low];
  }
  let rough = span / 4.0;
  let magnitude = 10_f64.powf(rough.log10().floor());
  let step = [1.0, 2.0, 5.0, 10.0]
    .into_iter()
    .map(|multiple| multiple * magnitude)
    .find(|step| *step >= rough)
    .unwrap_or(rough);
  let mut values = Vec::new();
  let mut value = (low / step).floor() * step;
  while value <= high + step * 1e-9 && values.len() < 12 {
    if value >= low - step * 1e-9 {
      values.push(value);
    }
    value += step;
  }
  values
}

/// A number as short as it can be read: no trailing zeros, and an exponent where the zeros would lead.
fn compact(value: f64) -> String {
  if value == 0.0 {
    return "0".to_owned();
  }
  if value.abs() < 1e-3 {
    return format!("{value:.0e}");
  }
  let text = format!("{value:.4}");
  text.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// One line of a chart.
struct Series {
  /// Its place among the runs, which fixes its color and dash.
  index: usize,
  points: Vec<(f64, f64)>,
}

/// Where a chart's data goes on the page: the ranges of its two axes and the rectangle they map onto.
struct Frame {
  x_range: (f64, f64),
  y_range: (f64, f64),
  /// The top of the plot: lower on the page where labels sit above it.
  top: f64,
}

impl Frame {
  fn plot_width() -> f64 {
    WIDTH - LEFT - RIGHT
  }

  fn base() -> f64 {
    HEIGHT - BOTTOM
  }

  fn x(&self, value: f64) -> f64 {
    let (low, high) = self.x_range;
    if high > low {
      LEFT + (value - low) / (high - low) * Self::plot_width()
    } else {
      LEFT + Self::plot_width() / 2.0
    }
  }

  fn y(&self, value: f64) -> f64 {
    let (low, high) = self.y_range;
    self.top + (1.0 - (value - low) / (high - low)) * (Self::base() - self.top)
  }

  /// Opens the drawing: a hairline and a number at each of `y_marks`, the baseline, and the horizontal axis's
  /// numbers and name. Everything here is recessive; the data drawn after it is the only loud thing.
  fn open(&self, y_marks: &[f64], x_name: &str) -> String {
    let mut svg = format!(r#"<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img">"#);
    let (right, base) = (LEFT + Self::plot_width(), Self::base());
    // Writing to a `String` cannot fail, here and below.
    for mark in y_marks {
      let at = self.y(*mark);
      let _ = write!(
        svg,
        r#"<line class="grid" x1="{LEFT}" x2="{right:.1}" y1="{at:.1}" y2="{at:.1}"/><text x="{:.1}" y="{:.1}" text-anchor="end">{}</text>"#,
        LEFT - 6.0,
        at + 4.0,
        compact(*mark)
      );
    }
    let _ = write!(
      svg,
      r#"<line class="axis" x1="{LEFT}" x2="{right:.1}" y1="{base:.1}" y2="{base:.1}"/>"#
    );
    for mark in ticks(self.x_range.0, self.x_range.1) {
      let _ = write!(
        svg,
        r#"<text x="{:.1}" y="{:.1}" text-anchor="middle">{}</text>"#,
        self.x(mark),
        base + 16.0,
        compact(mark)
      );
    }
    let _ = write!(
      svg,
      r#"<text x="{right:.1}" y="{:.1}" text-anchor="end">{x_name}</text>"#,
      base + 30.0
    );
    svg
  }
}

/// The frame that holds every point of `series`, and the round numbers its vertical axis is marked at. The axis
/// runs from a round number at or below the lowest point to one at or above the highest, never past `ceiling`:
/// a share cannot be drawn as if it could exceed one.
fn frame_of(series: &[Series], ceiling: Option<f64>) -> Option<(Frame, Vec<f64>)> {
  let all = || series.iter().flat_map(|series| series.points.iter().copied());
  let x_low = all().map(|point| point.0).min_by(f64::total_cmp)?;
  let x_high = all().map(|point| point.0).max_by(f64::total_cmp)?;
  let y_low = all().map(|point| point.1).min_by(f64::total_cmp)?;
  let y_high = all().map(|point| point.1).max_by(f64::total_cmp)?;
  // A flat series still needs a band to sit in.
  let pad = if y_high > y_low {
    0.0
  } else {
    y_high.abs().max(1e-6) * 0.1
  };
  let top = ceiling.map_or(y_high + pad, |ceiling| (y_high + pad).min(ceiling.max(y_high)));
  let rough = ticks(y_low - pad, top);
  let step = match (rough.first(), rough.get(1)) {
    (Some(first), Some(second)) => second - first,
    _ => (top - (y_low - pad)).max(1e-9),
  };
  // Out to the next round number on each side, so that the lowest and highest points sit inside marked lines.
  let low = ((y_low - pad) / step).floor() * step;
  let high = ceiling.map_or((top / step).ceil() * step, |ceiling| {
    ((top / step).ceil() * step).min(ceiling.max(y_high))
  });
  let marks = ticks(low, high);
  let frame = Frame {
    x_range: (x_low, x_high),
    y_range: (low, high.max(low + step * 1e-6)),
    top: TOP,
  };
  Some((frame, marks))
}

/// A line chart of one measure against the step, one line per run.
fn line_chart(series: &[Series], unit: &str, dots: bool, ceiling: Option<f64>) -> String {
  let Some((frame, marks)) = frame_of(series, ceiling) else {
    return String::new();
  };
  let mut svg = frame.open(&marks, "step");
  // Lines first, then the marks on top of them, then the invisible targets that carry the tooltips.
  for series in series {
    let path: Vec<String> = series
      .points
      .iter()
      .enumerate()
      .map(|(index, (px, py))| {
        format!(
          "{}{:.1} {:.1}",
          if index == 0 { "M" } else { "L" },
          frame.x(*px),
          frame.y(*py)
        )
      })
      .collect();
    let dash = DASHES.get(series.index).copied().unwrap_or("");
    let _ = write!(
      svg,
      r#"<path class="line" d="{}" style="stroke:{}" stroke-dasharray="{dash}"/>"#,
      path.join(" "),
      color(series.index)
    );
  }
  let ends: Vec<(f64, usize)> = series
    .iter()
    .filter_map(|series| series.points.last().map(|(_, py)| (frame.y(*py), series.index)))
    .collect();
  // End labels only where they do not sit on top of each other; the legend names every line either way.
  let apart = ends.iter().all(|(at, index)| {
    ends
      .iter()
      .all(|(other, place)| place == index || (at - other).abs() >= 13.0)
  });
  for series in series {
    let last = series.points.len().saturating_sub(1);
    for (index, (px, py)) in series.points.iter().enumerate() {
      let (cx, cy) = (frame.x(*px), frame.y(*py));
      if dots || index == last {
        let _ = write!(
          svg,
          r#"<circle class="dot" cx="{cx:.1}" cy="{cy:.1}" r="4" style="fill:{}"/>"#,
          color(series.index)
        );
      }
      let _ = write!(
        svg,
        r#"<circle class="hit" cx="{cx:.1}" cy="{cy:.1}" r="7"><title>{}: step {}, {} {unit}</title></circle>"#,
        letter(series.index),
        compact(*px),
        compact(*py)
      );
      if index == last && apart && ends.len() <= 4 {
        let _ = write!(
          svg,
          r#"<text class="label" x="{:.1}" y="{:.1}">{} {}</text>"#,
          cx + 8.0,
          cy + 4.0,
          letter(series.index),
          compact(*py)
        );
      }
    }
  }
  svg.push_str("</svg>");
  svg
}

/// The legend of a chart with more than one line: a key in the line's own color and dash, and the run's letter.
fn legend(series: &[Series]) -> String {
  if series.len() < 2 {
    return String::new();
  }
  let mut out = String::from(r#"<div class="legend">"#);
  for series in series {
    let dash = if DASHES.get(series.index).copied().unwrap_or("").is_empty() {
      "solid"
    } else {
      "dashed"
    };
    let _ = write!(
      out,
      r#"<span><span class="key" style="border-color:{};border-top-style:{dash}"></span>run {}</span>"#,
      color(series.index),
      letter(series.index)
    );
  }
  out.push_str("</div>");
  out
}

/// The points a chart draws, as a table: the same numbers for a reader who cannot or would rather not read a
/// chart.
fn points_table(series: &[Series], measure: &str) -> String {
  let mut out = format!(
    r#"<details><summary>The numbers</summary><table><tr><th>run</th><th class="n">step</th><th class="n">{}</th></tr>"#,
    escape(measure)
  );
  for series in series {
    for (step, value) in &series.points {
      let _ = write!(
        out,
        r#"<tr><td>{}</td><td class="n">{}</td><td class="n">{}</td></tr>"#,
        letter(series.index),
        compact(*step),
        compact(*value)
      );
    }
  }
  out.push_str("</table></details>");
  out
}

#[expect(
  clippy::cast_precision_loss,
  reason = "step counts and pair counts are far below 2^53"
)]
fn float(count: usize) -> f64 {
  count as f64
}

/// One measure of a logged step.
type Measure = fn(&StepLine) -> Option<f64>;

fn curve(run: &TrainingRun, measure: Measure) -> Vec<(f64, f64)> {
  run
    .steps
    .iter()
    .filter_map(|line| Some((float(line.step), measure(line)?)))
    .collect()
}

fn training_section(runs: &[(usize, &TrainingRun)], out: &mut String) {
  let measures: [(&str, &str, Measure); 4] = [
    ("Loss on labeled pairs", "pair loss", |line| line.pair_loss),
    ("Loss on catalog groups", "group loss", |line| line.group_loss),
    ("Gradient norm", "gradient norm", |line| line.grad_norm),
    ("Learning rate", "learning rate", |line| line.rate),
  ];
  let mut drawn = false;
  for (title, unit, measure) in measures {
    let series: Vec<Series> = runs
      .iter()
      .map(|(index, run)| Series {
        index: *index,
        points: curve(run, measure),
      })
      .filter(|series| !series.points.is_empty())
      .collect();
    if series.is_empty() {
      continue;
    }
    if !drawn {
      out.push_str(
        "<h2>Training</h2><p>Each point is the mean since the point before it, as the trainer logged it.</p>",
      );
      drawn = true;
    }
    let _ = write!(
      out,
      "<h3>{title}</h3>{}<figure>{}</figure>{}",
      legend(&series),
      line_chart(&series, unit, false, None),
      points_table(&series, unit)
    );
  }
  let without: Vec<char> = runs
    .iter()
    .filter(|(_, run)| run.steps.is_empty())
    .map(|(index, _)| letter(*index))
    .collect();
  if !without.is_empty() {
    let names: Vec<String> = without.iter().map(char::to_string).collect();
    let _ = write!(
      out,
      r#"<p class="note">No progress lines were kept for run {}.</p>"#,
      names.join(", ")
    );
  }

  // The validation score of each model a run scored, at the narrowest width the runs share.
  let width = runs.iter().filter_map(|(_, run)| run.widths().first().copied()).min();
  let Some(width) = width else { return };
  let series: Vec<Series> = runs
    .iter()
    .map(|(index, run)| Series {
      index: *index,
      points: run
        .validation_curve(width)
        .into_iter()
        .map(|(step, auc)| (float(step), auc))
        .collect(),
    })
    .filter(|series| !series.points.is_empty())
    .collect();
  if !series.is_empty() {
    let _ = write!(
      out,
      "<h3>Validation AUC of each model scored, at width {width}</h3><p>How often an edit scores above a recycle on \
       the validation pairs: the model the run started from at its first step, each checkpoint, and the final \
       model. The best one is the model the run names.</p>{}<figure>{}</figure>{}",
      legend(&series),
      line_chart(&series, "AUC", true, Some(1.0)),
      points_table(&series, "AUC")
    );
  }
}

/// How many bins a calibration's scores are shown in.
const BINS: usize = 20;

/// One group of pairs in a histogram: its place (which fixes its color), its name, and its counts.
type Group = (usize, &'static str, stats::Histogram);

/// The edits, the recycles, and the pairs with any other label, each counted into the same bins: from a round
/// number at or below the lowest labeled score and the configured threshold, up to 1.
fn groups_of(run: &crate::offline::events::Run) -> Option<(f64, Vec<Group>)> {
  let calibration = &run.calibration;
  let lowest = [Some(Label::Edit), Some(Label::Recycle), None]
    .into_iter()
    .filter_map(|label| stats::score_range(calibration, label))
    .map(|range| range.low)
    .min_by(f64::total_cmp)?;
  let low = ((lowest.min(f64::from(run.info.threshold)) * 20.0).floor() / 20.0).max(-1.0);
  let groups = [
    (Some(Label::Edit), 0_usize, "edits"),
    (Some(Label::Recycle), 1, "recycles"),
    (None, 2, "other labels"),
  ]
  .into_iter()
  .map(|(label, index, name)| (index, name, stats::histogram(calibration, label, low, 1.0, BINS)))
  .filter(|(_, _, histogram)| histogram.counts.iter().any(|count| *count > 0))
  .collect();
  Some((low, groups))
}

/// A calibration's overall similarities as grouped columns, edits beside recycles, with the configured and the
/// suggested threshold marked.
fn histogram_chart(run: &crate::offline::events::Run, low: f64, groups: &[Group]) -> String {
  let tallest = groups
    .iter()
    .flat_map(|(_, _, histogram)| histogram.counts.iter().copied())
    .max()
    .unwrap_or(1)
    .max(1);
  let marks: Vec<f64> = ticks(0.0, float(tallest))
    .into_iter()
    .filter(|mark| mark.fract() == 0.0)
    .collect();
  let frame = Frame {
    x_range: (low, 1.0),
    y_range: (0.0, marks.last().copied().unwrap_or(1.0).max(float(tallest))),
    // Room above the plot for the names of the two thresholds, one under the other.
    top: TOP + 30.0,
  };
  let mut svg = frame.open(&marks, "overall similarity");
  let base = Frame::base();
  // Columns of one bin stand side by side with a gap between them, never one over another.
  let bin = Frame::plot_width() / float(BINS);
  let column = ((bin - 2.0) / float(groups.len().max(1)) - 2.0).clamp(2.0, 24.0);
  let step = (1.0 - low) / float(BINS);
  for (place, (index, name, histogram)) in groups.iter().enumerate() {
    for (number, count) in histogram.counts.iter().enumerate().filter(|(_, count)| **count > 0) {
      let left = LEFT + float(number) * bin + 2.0 + float(place) * (column + 2.0);
      let top_at = frame.y(float(*count));
      let round = 4.0_f64.min(column / 2.0).min((base - top_at).max(0.0));
      let from = low + float(number) * step;
      // Rounded at the end that carries the value, square where it stands on the baseline.
      let _ = write!(
        svg,
        r#"<path d="M{left:.1} {base:.1} V{:.1} Q{left:.1} {top_at:.1} {:.1} {top_at:.1} H{:.1} Q{right:.1} {top_at:.1} {right:.1} {:.1} V{base:.1} Z" style="fill:{}"><title>{name}: {count} between {} and {}</title></path>"#,
        top_at + round,
        left + round,
        left + column - round,
        top_at + round,
        color(*index),
        compact(from),
        compact(from + step),
        right = left + column
      );
    }
  }
  let thresholds = [
    Some(("threshold", f64::from(run.info.threshold))),
    run
      .calibration
      .suggestion
      .map(|suggestion| ("suggested", f64::from(suggestion.threshold))),
  ];
  for (row, (name, value)) in thresholds.into_iter().flatten().enumerate() {
    let at = frame.x(value);
    let label_at = TOP + 10.0 + float(row) * 13.0;
    // The name goes on the side with room for it: left of a line in the right half, right of one in the left.
    let (anchor, text_at) = if at > LEFT + Frame::plot_width() / 2.0 {
      ("end", at - 5.0)
    } else {
      ("start", at + 5.0)
    };
    let _ = write!(
      svg,
      r#"<line class="mark" x1="{at:.1}" x2="{at:.1}" y1="{:.1}" y2="{base:.1}"/><text class="label" x="{text_at:.1}" y="{label_at:.1}" text-anchor="{anchor}">{name} {value:.3}</text>"#,
      label_at - 9.0
    );
  }
  svg.push_str("</svg>");
  svg
}

/// What a histogram draws, as its legend and as a table of the counts in every bin that has any.
fn histogram_legend_and_table(low: f64, groups: &[Group]) -> (String, String) {
  let mut legend = String::from(r#"<div class="legend">"#);
  let mut table =
    String::from(r#"<details><summary>The numbers</summary><table><tr><th class="n">from</th><th class="n">to</th>"#);
  for (index, name, histogram) in groups {
    let total: usize = histogram.counts.iter().sum();
    let _ = write!(
      legend,
      r#"<span><span class="swatch" style="background:{}"></span>{name} ({total})</span>"#,
      color(*index)
    );
    let _ = write!(table, r#"<th class="n">{name}</th>"#);
  }
  legend.push_str("</div>");
  table.push_str("</tr>");
  let step = (1.0 - low) / float(BINS);
  for number in 0..BINS {
    let cells: Vec<usize> = groups
      .iter()
      .map(|(_, _, histogram)| histogram.counts.get(number).copied().unwrap_or(0))
      .collect();
    if cells.iter().all(|count| *count == 0) {
      continue;
    }
    let from = low + float(number) * step;
    let _ = write!(
      table,
      r#"<tr><td class="n">{}</td><td class="n">{}</td>"#,
      compact(from),
      compact(from + step)
    );
    for count in cells {
      let _ = write!(table, r#"<td class="n">{count}</td>"#);
    }
    table.push_str("</tr>");
  }
  table.push_str("</table></details>");
  (legend, table)
}

fn calibration_section(index: usize, run: &crate::offline::events::Run, out: &mut String) {
  let calibration = &run.calibration;
  let _ = write!(
    out,
    "<h3>Run {}: model {}</h3>",
    letter(index),
    escape(&run.info.fingerprint)
  );
  if let Some((low, groups)) = groups_of(run) {
    let (legend, table) = histogram_legend_and_table(low, &groups);
    let _ = write!(
      out,
      "{legend}<figure>{}</figure>{table}",
      histogram_chart(run, low, &groups)
    );
  }
  let mut rows = vec![
    ("pairs scored", calibration.pairs.len().to_string()),
    ("edits wrongly rejected", calibration.false_rejects.to_string()),
    ("recycles wrongly accepted", calibration.false_accepts.to_string()),
    ("configured threshold", format!("{:.3}", run.info.threshold)),
  ];
  if let Some(suggestion) = calibration.suggestion {
    rows.push((
      "best threshold for these pairs",
      format!("{:.3} (margin {:.3})", suggestion.threshold, suggestion.margin),
    ));
  }
  if let Some(area) = stats::area_under_curve(calibration) {
    rows.push(("AUC", format!("{area:.4}")));
  }
  out.push_str("<table>");
  for (name, value) in rows {
    let _ = write!(out, r#"<tr><th>{name}</th><td class="n">{}</td></tr>"#, escape(&value));
  }
  out.push_str("</table>");
}

fn runs_table(runs: &[Loaded], out: &mut String) {
  out.push_str("<h2>Runs</h2><table><tr><th>run</th><th>kind</th><th>status</th><th>what it is</th></tr>");
  for (index, loaded) in runs.iter().enumerate() {
    let Some(listed) = loaded.listed() else { continue };
    let error = match (&listed.status, &listed.manifest.error) {
      (Status::Failed, Some(error)) => format!(": {}", escape(error)),
      _ => String::new(),
    };
    let chain = match loaded {
      Loaded::Training { chain, .. } if chain.len() > 1 => format!(" (resumed, {} processes)", chain.len()),
      _ => String::new(),
    };
    let _ = write!(
      out,
      "<tr><td>{}</td><td>{}</td><td>{}{error}</td><td>{}{chain}</td></tr>",
      letter(index),
      escape(&listed.manifest.kind),
      listed.status.as_str(),
      escape(&describe(listed))
    );
  }
  out.push_str("</table>");
}

/// The rows of a side-by-side table: every name any run has, in the order first seen, with each run's value.
fn side_by_side(columns: &[(usize, Vec<(String, String)>)], out: &mut String) {
  let mut names: Vec<&String> = columns
    .iter()
    .flat_map(|(_, rows)| rows.iter().map(|(name, _)| name))
    .collect();
  let mut seen = std::collections::BTreeSet::new();
  names.retain(|name| seen.insert(*name));
  out.push_str("<table><tr><th></th>");
  for (index, _) in columns {
    let _ = write!(out, "<th>run {}</th>", letter(*index));
  }
  out.push_str("</tr>");
  for name in names {
    let values: Vec<&str> = columns
      .iter()
      .map(|(_, rows)| {
        rows
          .iter()
          .find(|(found, _)| found == name)
          .map_or("-", |(_, value)| value.as_str())
      })
      .collect();
    let differs = values.windows(2).any(|pair| pair.first() != pair.get(1));
    let _ = write!(
      out,
      "<tr{}><td>{}</td>",
      if differs { r#" class="differs""# } else { "" },
      escape(name)
    );
    for value in values {
      let _ = write!(out, "<td>{}</td>", escape(value));
    }
    out.push_str("</tr>");
  }
  out.push_str("</table>");
}

/// One HTML page for `runs`: at most [`MAX_RUNS`] of them, training runs and calibrations in any mix.
pub fn report(runs: &[Loaded]) -> String {
  let mut out = format!(
    "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" \
     content=\"width=device-width, initial-scale=1\"><title>Recorded runs</title><style>{STYLE}</style></head><body><main>\
     <h1>Recorded runs</h1><p>{} run(s). Everything on this page was read from their run directories.</p>",
    runs.len()
  );
  runs_table(runs, &mut out);

  let training: Vec<(usize, &similarity_runs::Listed, &TrainingRun)> = runs
    .iter()
    .enumerate()
    .filter_map(|(index, loaded)| match loaded {
      Loaded::Training { chain, run } => Some((index, chain.last()?, run.as_ref())),
      Loaded::Calibration { .. } => None,
    })
    .collect();
  if !training.is_empty() {
    out.push_str("<h2>What went in, and what came out</h2>");
    if training.len() > 1 {
      out.push_str("<p>A row marked * is not the same for every run.</p>");
    }
    let went_in: Vec<(usize, Vec<(String, String)>)> = training
      .iter()
      .map(|(index, listed, _)| (*index, inputs(listed)))
      .collect();
    side_by_side(&went_in, &mut out);
    let came_out: Vec<(usize, Vec<(String, String)>)> =
      training.iter().map(|(index, _, run)| (*index, outcomes(run))).collect();
    side_by_side(&came_out, &mut out);
    let curves: Vec<(usize, &TrainingRun)> = training.iter().map(|(index, _, run)| (*index, *run)).collect();
    training_section(&curves, &mut out);
  }

  let calibrations: Vec<(usize, &crate::offline::events::Run)> = runs
    .iter()
    .enumerate()
    .filter_map(|(index, loaded)| match loaded {
      Loaded::Calibration { run, .. } => Some((index, &**run)),
      Loaded::Training { .. } => None,
    })
    .collect();
  if !calibrations.is_empty() {
    out.push_str(
      "<h2>Calibrations</h2><p>How many pairs scored where. A threshold accepts what is at or to the right of it. \
       The bins are for looking at; no threshold is to be read from where one ends.</p>",
    );
    for (index, run) in calibrations {
      calibration_section(index, run, &mut out);
    }
  }
  out.push_str("</main></body></html>\n");
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn an_axis_is_marked_at_round_numbers_that_cover_the_data() {
    assert_eq!(ticks(0.0, 40.0), [0.0, 10.0, 20.0, 30.0, 40.0]);
    assert_eq!(ticks(0.0, 7.0), [0.0, 2.0, 4.0, 6.0]);
    let small = ticks(0.016, 0.052);
    assert!(
      small.len() >= 3 && small.iter().all(|value| (0.016..=0.0521).contains(value)),
      "{small:?}"
    );
    assert_eq!(ticks(3.0, 3.0), [3.0], "no span, one mark");
  }

  #[test]
  fn a_number_is_written_as_short_as_it_can_be_read() {
    assert_eq!(compact(0.0), "0");
    assert_eq!(compact(40.0), "40");
    assert_eq!(compact(0.895), "0.895");
    assert_eq!(compact(0.050_04), "0.05");
    assert_eq!(compact(0.000_02), "2e-5");
  }

  #[test]
  fn text_from_a_run_cannot_become_markup() {
    assert_eq!(escape(r#"<b a="1">&"#), "&lt;b a=&quot;1&quot;&gt;&amp;");
  }

  #[test]
  fn a_line_chart_draws_every_run_in_its_own_color_and_dash_and_says_each_point() {
    let series = [
      Series {
        index: 0,
        points: vec![(10.0, 0.05), (20.0, 0.03)],
      },
      Series {
        index: 1,
        points: vec![(10.0, 0.06), (20.0, 0.02)],
      },
    ];
    let svg = line_chart(&series, "pair loss", false, None);
    assert_eq!(svg.matches(r#"class="line""#).count(), 2);
    assert!(svg.contains("var(--s1)") && svg.contains("var(--s2)"));
    assert!(
      svg.contains(r#"stroke-dasharray="6 3""#),
      "the second run is dashed as well as colored"
    );
    assert_eq!(svg.matches("<title>").count(), 4, "every point can be read");
    assert!(svg.contains("<title>b: step 20, 0.02 pair loss</title>"), "{svg}");
    assert_eq!(svg.matches(r#"class="dot""#).count(), 2, "only the ends are marked");
    assert!(legend(&series).contains("run a") && legend(&series).contains("run b"));
    assert_eq!(legend(&series[..1]), "", "one line needs no legend");
    assert!(points_table(&series, "pair loss").contains(r#"<td class="n">0.03</td>"#));
    assert_eq!(line_chart(&[], "x", false, None), "");

    // A share that is flat at one is drawn below one, not as if it could go above.
    let flat = [Series {
      index: 0,
      points: vec![(0.0, 1.0), (5.0, 1.0)],
    }];
    let (frame, marks) = frame_of(&flat, Some(1.0)).unwrap();
    assert!(
      frame.y_range.1 <= 1.0 + 1e-12 && frame.y_range.0 < 1.0,
      "{:?}",
      frame.y_range
    );
    assert!(marks.iter().all(|mark| *mark <= 1.0 + 1e-12), "{marks:?}");
    // And an axis reaches out to a round number on each side of what it holds.
    let (frame, _) = frame_of(&series, None).unwrap();
    assert!(
      frame.y_range.0 <= 0.02 && frame.y_range.1 >= 0.06,
      "{:?}",
      frame.y_range
    );
  }
}
