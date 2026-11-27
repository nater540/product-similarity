//! The few numbers the server keeps about itself (docs/decisions.md, D39).
//!
//! What one request did, and where its time went, is in its trace ([`crate::telemetry`]). These are the things
//! no single request's trace shows: how close the model is to its limit, how many requests are waiting for it,
//! how many were turned away or given up on, how many writes ended with nobody to tell, and what verdicts are
//! going out. There are few of them on purpose.
//!
//! They are OpenTelemetry instruments. Where they go (a collector, a page, both, nowhere) is decided by the
//! meter this is made from, and nothing here knows.
//!
//! **To add one:** a name, a field, a line in [`Metrics::new`], and a method that records it. Three rules:
//!
//! - **Nothing about a record is an attribute.** No id, no brand, no text, no score: an attribute value is one
//!   of a handful of words fixed in this file. Each distinct value is a time series somebody stores and pays
//!   for. The rest of the server calls a method and cannot spell an attribute wrong or invent one.
//! - **The names are an interface.** Dashboards and alerts are written against them. Add; never rename one or
//!   change what it counts.
//! - **A metric is for what a trace cannot say.** How long a kind of request takes, how often it fails, and
//!   what it waited for are read from spans.

use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram, Meter, MeterProvider as _};

use crate::engine::{Verdict, WriteMode};

/// How long forward passes of the model take. The rate of its sum is the share of the time the model is busy:
/// at 1 the server is at its limit.
pub const PASS_SECONDS: &str = "record_similarity_embed_pass_duration_seconds";
/// Requests waiting for the model right now. `model.queue_depth` is the most there can be.
pub const WAITING: &str = "record_similarity_embed_waiting";
/// Requests refused without being embedded, by `reason`: `queue_full` (`RESOURCE_EXHAUSTED`) or `too_late`
/// (`DEADLINE_EXCEEDED`).
pub const REFUSED: &str = "record_similarity_embed_refused_total";
/// Requests passed over because their caller had gone before the model reached them.
pub const ABANDONED: &str = "record_similarity_embed_abandoned_total";
/// Writes that finished, or failed, after the request that asked for them had gone. Each has a log line.
pub const UNANSWERED_WRITES: &str = "record_similarity_writes_unanswered_total";
/// `Evaluate` calls that were answered, by `verdict` and by the write `mode` the request asked for.
pub const EVALUATIONS: &str = "record_similarity_evaluations_total";

/// The upper bounds of the buckets a forward pass's duration is counted into, in seconds: from one millisecond
/// to two and a half seconds. A resolution, not a threshold: nothing is decided by them. One short text takes
/// about a millisecond and a full pass ten to twenty.
const SECONDS: [f64; 11] = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5];

/// Why a request was refused without being embedded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
  /// As many requests were waiting as may wait.
  QueueFull,
  /// Its deadline had passed, or could not be met.
  TooLate,
}

impl Refusal {
  const fn name(self) -> &'static str {
    match self {
      Self::QueueFull => "queue_full",
      Self::TooLate => "too_late",
    }
  }
}

const fn verdict_name(verdict: Verdict) -> &'static str {
  match verdict {
    Verdict::Created => "created",
    Verdict::Accepted => "accepted",
    Verdict::Rejected => "rejected",
    Verdict::Embedded => "embedded",
  }
}

const fn mode_name(mode: WriteMode) -> &'static str {
  match mode {
    WriteMode::Enforce => "enforce",
    WriteMode::DryRun => "dry_run",
    WriteMode::Force => "force",
  }
}

/// What a server counts about itself.
#[derive(Debug)]
pub struct Metrics {
  meter: Meter,
  pass_seconds: Histogram<f64>,
  refused: Counter<u64>,
  abandoned: Counter<u64>,
  unanswered_writes: Counter<u64>,
  evaluations: Counter<u64>,
}

impl Default for Metrics {
  fn default() -> Self {
    Self::discarded()
  }
}

impl Metrics {
  /// The server's metrics, kept wherever `meter` keeps them.
  ///
  /// Every count is there from the start, at zero, so that a rate over it exists from the first reading and its
  /// first occurrence is an increase and not a new series.
  #[must_use]
  pub fn new(meter: &Meter) -> Self {
    let metrics = Self {
      meter: meter.clone(),
      pass_seconds: meter
        .f64_histogram(PASS_SECONDS)
        .with_unit("s")
        .with_description("How long forward passes of the model took. The rate of its sum is how busy the model is.")
        .with_boundaries(SECONDS.to_vec())
        .build(),
      refused: meter
        .u64_counter(REFUSED)
        .with_description(
          "Requests refused without being embedded: the queue was full, or their deadline left no time.",
        )
        .build(),
      abandoned: meter
        .u64_counter(ABANDONED)
        .with_description("Requests passed over because their caller had gone before the model reached them.")
        .build(),
      unanswered_writes: meter
        .u64_counter(UNANSWERED_WRITES)
        .with_description("Writes that finished, or failed, after the request that asked for them had gone.")
        .build(),
      evaluations: meter
        .u64_counter(EVALUATIONS)
        .with_description("Evaluate calls that were answered, by verdict and by the write mode asked for.")
        .build(),
    };
    metrics
      .refused
      .add(0, &[KeyValue::new("reason", Refusal::QueueFull.name())]);
    metrics
      .refused
      .add(0, &[KeyValue::new("reason", Refusal::TooLate.name())]);
    metrics.abandoned.add(0, &[]);
    metrics.unanswered_writes.add(0, &[]);
    for verdict in [
      Verdict::Created,
      Verdict::Accepted,
      Verdict::Rejected,
      Verdict::Embedded,
    ] {
      for mode in [WriteMode::Enforce, WriteMode::DryRun, WriteMode::Force] {
        metrics.evaluations.add(0, &evaluation(verdict, mode));
      }
    }
    metrics
  }

  /// Metrics that are kept nowhere: for a server that was asked for none, a command, a test of something else.
  /// Every method works and costs next to nothing.
  #[must_use]
  pub fn discarded() -> Self {
    Self::new(&opentelemetry::metrics::noop::NoopMeterProvider::new().meter("discarded"))
  }

  /// A forward pass of the model took this long.
  pub fn passed(&self, elapsed: Duration) {
    self.pass_seconds.record(elapsed.as_secs_f64(), &[]);
  }

  /// A request was refused without being embedded.
  pub fn refused(&self, why: Refusal) {
    self.refused.add(1, &[KeyValue::new("reason", why.name())]);
  }

  /// A request was passed over: its caller had gone before the model reached it.
  pub fn abandoned(&self) {
    self.abandoned.add(1, &[]);
  }

  /// A write finished, or failed, after the request that asked for it had gone.
  pub fn write_unanswered(&self) {
    self.unanswered_writes.add(1, &[]);
  }

  /// An `Evaluate` was answered with this verdict, having asked for this mode.
  pub fn evaluated(&self, verdict: Verdict, mode: WriteMode) {
    self.evaluations.add(1, &evaluation(verdict, mode));
  }

  /// From now on, how many requests are waiting for the model is whatever `waiting` says when it is asked. A
  /// level is read, not counted: nothing happens here until somebody wants the number.
  pub fn watch_waiting(&self, waiting: impl Fn() -> u64 + Send + Sync + 'static) {
    // The instrument lives as long as its meter does; the handle is not needed to keep it.
    let _ = self
      .meter
      .u64_observable_gauge(WAITING)
      .with_description("Requests waiting for the model right now. model.queue_depth is the most there can be.")
      .with_callback(move |observer| observer.observe(waiting(), &[]))
      .build();
  }
}

fn evaluation(verdict: Verdict, mode: WriteMode) -> [KeyValue; 2] {
  [
    KeyValue::new("verdict", verdict_name(verdict)),
    KeyValue::new("mode", mode_name(mode)),
  ]
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeSet;
  use std::sync::Arc;
  use std::sync::atomic::{AtomicU64, Ordering};

  use super::*;
  use crate::telemetry::testing::{counted, paged};

  /// Every count is there before anything has happened, at zero: a rate over it exists from the first reading,
  /// and the first refusal is an increase and not a series nobody has seen. A duration has nothing to say until
  /// there has been one, and the number waiting is not there until something says how to read it.
  #[test]
  fn every_count_is_there_from_the_start_at_zero() {
    let telemetry = paged();
    let read = |series: &str| counted(&telemetry, series);
    assert_eq!(
      read("record_similarity_embed_refused_total{reason=\"queue_full\"}"),
      Some(0.0)
    );
    assert_eq!(
      read("record_similarity_embed_refused_total{reason=\"too_late\"}"),
      Some(0.0)
    );
    assert_eq!(read(ABANDONED), Some(0.0));
    assert_eq!(read(UNANSWERED_WRITES), Some(0.0));
    let page = telemetry.page().unwrap();
    let verdicts = page
      .lines()
      .filter(|line| line.starts_with(EVALUATIONS) && line.ends_with(" 0"))
      .count();
    assert_eq!(verdicts, 12, "four verdicts under three modes:\n{page}");
    assert!(!page.contains(PASS_SECONDS) && !page.contains(WAITING), "{page}");
  }

  #[test]
  fn each_thing_that_happens_is_counted_as_what_it_was() {
    let telemetry = paged();
    let metrics = telemetry.metrics();
    metrics.refused(Refusal::QueueFull);
    metrics.refused(Refusal::TooLate);
    metrics.refused(Refusal::TooLate);
    metrics.abandoned();
    metrics.write_unanswered();
    metrics.write_unanswered();
    metrics.write_unanswered();
    metrics.evaluated(Verdict::Accepted, WriteMode::Enforce);
    metrics.evaluated(Verdict::Accepted, WriteMode::Enforce);
    metrics.evaluated(Verdict::Rejected, WriteMode::DryRun);
    metrics.evaluated(Verdict::Rejected, WriteMode::Force);
    metrics.evaluated(Verdict::Created, WriteMode::Enforce);
    metrics.evaluated(Verdict::Embedded, WriteMode::Enforce);

    let read = |series: &str| counted(&telemetry, series);
    assert_eq!(
      read("record_similarity_embed_refused_total{reason=\"queue_full\"}"),
      Some(1.0)
    );
    assert_eq!(
      read("record_similarity_embed_refused_total{reason=\"too_late\"}"),
      Some(2.0)
    );
    assert_eq!(read(ABANDONED), Some(1.0));
    assert_eq!(read(UNANSWERED_WRITES), Some(3.0));
    let verdict = |verdict: &str, mode: &str| {
      read(&format!(
        "record_similarity_evaluations_total{{mode=\"{mode}\",verdict=\"{verdict}\"}}"
      ))
    };
    assert_eq!(verdict("accepted", "enforce"), Some(2.0));
    assert_eq!(verdict("rejected", "dry_run"), Some(1.0));
    assert_eq!(verdict("rejected", "force"), Some(1.0));
    assert_eq!(
      verdict("rejected", "enforce"),
      Some(0.0),
      "a preview's rejection is not a refusal"
    );
    assert_eq!(verdict("created", "enforce"), Some(1.0));
    assert_eq!(verdict("embedded", "enforce"), Some(1.0));
  }

  /// A pass's duration is counted into the buckets this file names, a value at a bound belonging to that bound,
  /// and their sum is the time the model was busy.
  #[test]
  fn a_pass_is_counted_into_the_buckets_this_file_names() {
    let telemetry = paged();
    let metrics = telemetry.metrics();
    for elapsed in [
      Duration::from_micros(999),
      Duration::from_millis(1),
      Duration::from_nanos(1_000_001),
      Duration::from_millis(12),
      Duration::from_mins(1),
    ] {
      metrics.passed(elapsed);
    }
    let bucket = |le: &str| {
      counted(
        &telemetry,
        &format!("record_similarity_embed_pass_duration_seconds_bucket{{le=\"{le}\"}}"),
      )
    };
    assert_eq!(bucket("0.001"), Some(2.0));
    assert_eq!(bucket("0.0025"), Some(3.0));
    assert_eq!(bucket("0.01"), Some(3.0));
    assert_eq!(bucket("0.025"), Some(4.0));
    assert_eq!(bucket("2.5"), Some(4.0));
    assert_eq!(bucket("+Inf"), Some(5.0), "longer than every bound is still counted");
    let page = telemetry.page().unwrap();
    let buckets = page
      .lines()
      .filter(|line| line.starts_with("record_similarity_embed_pass_duration_seconds_bucket"))
      .count();
    assert_eq!(buckets, SECONDS.len() + 1, "{page}");
    assert_eq!(
      counted(&telemetry, "record_similarity_embed_pass_duration_seconds_count"),
      Some(5.0)
    );
    let busy = counted(&telemetry, "record_similarity_embed_pass_duration_seconds_sum").unwrap();
    assert!((busy - 60.014_999_001).abs() < 1e-6, "{busy}");
  }

  /// How many are waiting is a level: it is whatever the server says when somebody asks, not a count of anything
  /// that happened here.
  #[test]
  fn the_number_waiting_is_read_when_it_is_asked_for() {
    let telemetry = paged();
    let waiting = Arc::new(AtomicU64::new(3));
    let read = Arc::clone(&waiting);
    telemetry.metrics().watch_waiting(move || read.load(Ordering::Relaxed));
    assert_eq!(counted(&telemetry, WAITING), Some(3.0));
    waiting.store(41, Ordering::Relaxed);
    assert_eq!(counted(&telemetry, WAITING), Some(41.0));
    waiting.store(0, Ordering::Relaxed);
    assert_eq!(counted(&telemetry, WAITING), Some(0.0));
  }

  /// Each metric says what it is and what kind of thing it is, under the name this file gives it and no other:
  /// what a dashboard is written against.
  #[test]
  fn each_metric_is_introduced_under_its_own_name_with_its_kind() {
    let telemetry = paged();
    let metrics = telemetry.metrics();
    metrics.passed(Duration::from_millis(2));
    metrics.watch_waiting(|| 0);
    let page = telemetry.page().unwrap();
    for (name, kind) in [
      (PASS_SECONDS, "histogram"),
      (WAITING, "gauge"),
      (REFUSED, "counter"),
      (ABANDONED, "counter"),
      (UNANSWERED_WRITES, "counter"),
      (EVALUATIONS, "counter"),
    ] {
      assert!(page.contains(&format!("# TYPE {name} {kind}\n")), "{name}:\n{page}");
      assert!(
        page.contains(&format!("# HELP {name} ")),
        "{name} says what it is:\n{page}"
      );
      assert_eq!(kind == "counter", name.ends_with("_total"), "{name}");
    }
    let introduced = page.lines().filter(|line| line.starts_with("# TYPE ")).count();
    assert_eq!(introduced, 6, "these and nothing else:\n{page}");
    assert!(!page.contains("otel_scope") && !page.contains("target_info"), "{page}");
  }

  /// Metrics that are kept nowhere cost nothing to make and break nothing.
  #[test]
  fn discarded_metrics_take_everything_and_keep_nothing() {
    let metrics = Metrics::default();
    metrics.passed(Duration::from_millis(1));
    metrics.refused(Refusal::QueueFull);
    metrics.abandoned();
    metrics.write_unanswered();
    metrics.evaluated(Verdict::Created, WriteMode::Enforce);
    metrics.watch_waiting(|| 7);
  }

  /// The dashboard that ships (`deploy/observability/grafana/`) is written against these names, which is what
  /// makes them an interface. A name it asks for that no server keeps is a panel that stays empty and says
  /// nothing, so every series in it is one of the six, and each of the six is on it.
  #[test]
  fn the_dashboard_asks_for_every_metric_and_for_no_other() {
    let dashboard = include_str!("../../../deploy/observability/grafana/dashboards/record-similarity.json");
    let asked: BTreeSet<&str> = dashboard
      .split(|c: char| !(c.is_ascii_lowercase() || c == '_'))
      .filter(|word| word.starts_with("record_similarity_"))
      // A histogram is read back as three series named after it.
      .map(|word| {
        ["_bucket", "_sum", "_count"]
          .iter()
          .find_map(|part| word.strip_suffix(part))
          .unwrap_or(word)
      })
      .collect();
    let kept = BTreeSet::from([
      PASS_SECONDS,
      WAITING,
      REFUSED,
      ABANDONED,
      UNANSWERED_WRITES,
      EVALUATIONS,
    ]);
    assert_eq!(asked, kept);
  }
}
