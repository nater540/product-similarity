//! Numbers about the process, for a monitoring system to read (docs/decisions.md, D39).
//!
//! A log line says what one request did. These say what the server is doing: how many requests of each kind it
//! answers and how, how long they take, what the one embedding thread is busy with, how the store behaves, and
//! what verdicts are going out. They are counted where they happen, in memory, and written out in the Prometheus
//! text format when something asks ([`serve`]). That format is the one thing a Datadog agent, an OpenTelemetry
//! collector, and Prometheus itself can all read, so nothing here knows which of them is asking.
//!
//! Three rules hold for everything in this module:
//!
//! - **Nothing about a record is a label.** No id, no brand, no text: a label value is one of a handful of
//!   words fixed in this file (a call, a status code, a verdict), or the model's fingerprint. Each distinct
//!   label value is a time series somebody stores and pays for, and a record's brand is nobody's business in a
//!   monitoring system.
//! - **Counting never fails a request and never waits.** Every count is an atomic add.
//! - **The names are an interface.** Dashboards and alerts are written against them. Add a metric or a label
//!   value; never rename one or change what it counts.

use std::fmt::Write as _;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;
use similarity_core::matching::Decision;
use similarity_store::{Expected, Found, Neighbor, SearchQuery, StoreError, StoredRecord, VectorStore, Version};
use tokio::net::TcpListener;
use tonic::{Code, Status};

use crate::engine::{Verdict, WriteMode};
use crate::worker::WorkerSnapshot;

/// The path the metrics are served at.
pub const PATH: &str = "/metrics";

/// The content type of the Prometheus text format, version 0.0.4.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// The upper bounds of the latency buckets: in nanoseconds for counting, and as the `le` label, in seconds.
///
/// From one millisecond, under which a save with no changes is answered, to two and a half seconds, by which any
/// caller has given up. They are not thresholds and nothing is decided by them: they are how finely a latency
/// can be read back, and each one is a time series for every call and every store operation that is used.
const BUCKETS: [(u64, &str); 11] = [
  (1_000_000, "0.001"),
  (2_500_000, "0.0025"),
  (5_000_000, "0.005"),
  (10_000_000, "0.01"),
  (25_000_000, "0.025"),
  (50_000_000, "0.05"),
  (100_000_000, "0.1"),
  (250_000_000, "0.25"),
  (500_000_000, "0.5"),
  (1_000_000_000, "1"),
  (2_500_000_000, "2.5"),
];

/// Every gRPC status code, in the order of its number, by the name the protocol gives it.
const CODES: [&str; 17] = [
  "OK",
  "CANCELLED",
  "UNKNOWN",
  "INVALID_ARGUMENT",
  "DEADLINE_EXCEEDED",
  "NOT_FOUND",
  "ALREADY_EXISTS",
  "PERMISSION_DENIED",
  "RESOURCE_EXHAUSTED",
  "FAILED_PRECONDITION",
  "ABORTED",
  "OUT_OF_RANGE",
  "UNIMPLEMENTED",
  "INTERNAL",
  "UNAVAILABLE",
  "DATA_LOSS",
  "UNAUTHENTICATED",
];

/// A call of the API, as the `call` label names it: the method's name in the `.proto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Call {
  /// `Describe`.
  Describe,
  /// `Evaluate`.
  Evaluate,
  /// `FindSimilar`.
  FindSimilar,
  /// `Match`.
  Match,
  /// `Delete`.
  Delete,
  /// `Import`: one stream is one call, however many records it carries.
  Import,
}

impl Call {
  const ALL: [Self; 6] = [
    Self::Describe,
    Self::Evaluate,
    Self::FindSimilar,
    Self::Match,
    Self::Delete,
    Self::Import,
  ];

  const fn name(self) -> &'static str {
    match self {
      Self::Describe => "Describe",
      Self::Evaluate => "Evaluate",
      Self::FindSimilar => "FindSimilar",
      Self::Match => "Match",
      Self::Delete => "Delete",
      Self::Import => "Import",
    }
  }

  const fn index(self) -> usize {
    match self {
      Self::Describe => 0,
      Self::Evaluate => 1,
      Self::FindSimilar => 2,
      Self::Match => 3,
      Self::Delete => 4,
      Self::Import => 5,
    }
  }
}

/// An operation on the store, as the `operation` label names it: the [`VectorStore`] method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
  /// [`VectorStore::check`].
  Check,
  /// [`VectorStore::get`].
  Get,
  /// [`VectorStore::put`].
  Put,
  /// [`VectorStore::put_if`].
  PutIf,
  /// [`VectorStore::delete`].
  Delete,
  /// [`VectorStore::delete_if`].
  DeleteIf,
  /// [`VectorStore::search`].
  Search,
  /// [`VectorStore::search_vectors`].
  SearchVectors,
}

impl Operation {
  const ALL: [Self; 8] = [
    Self::Check,
    Self::Get,
    Self::Put,
    Self::PutIf,
    Self::Delete,
    Self::DeleteIf,
    Self::Search,
    Self::SearchVectors,
  ];

  const fn name(self) -> &'static str {
    match self {
      Self::Check => "check",
      Self::Get => "get",
      Self::Put => "put",
      Self::PutIf => "put_if",
      Self::Delete => "delete",
      Self::DeleteIf => "delete_if",
      Self::Search => "search",
      Self::SearchVectors => "search_vectors",
    }
  }

  const fn index(self) -> usize {
    match self {
      Self::Check => 0,
      Self::Get => 1,
      Self::Put => 2,
      Self::PutIf => 3,
      Self::Delete => 4,
      Self::DeleteIf => 5,
      Self::Search => 6,
      Self::SearchVectors => 7,
    }
  }
}

const VERDICTS: [(Verdict, &str); 4] = [
  (Verdict::Created, "created"),
  (Verdict::Accepted, "accepted"),
  (Verdict::Rejected, "rejected"),
  (Verdict::Embedded, "embedded"),
];

const MODES: [(WriteMode, &str); 3] = [
  (WriteMode::Enforce, "enforce"),
  (WriteMode::DryRun, "dry_run"),
  (WriteMode::Force, "force"),
];

const DECISIONS: [(Decision, &str); 3] = [
  (Decision::Matched, "matched"),
  (Decision::NoMatch, "no_match"),
  (Decision::NotJudged, "not_judged"),
];

/// What became of one record of an import, as the `outcome` label names it.
const IMPORT_OUTCOMES: [&str; 3] = ["created", "exists", "failed"];

fn nanoseconds(elapsed: Duration) -> u64 {
  u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

#[expect(
  clippy::cast_precision_loss,
  reason = "a total of nanoseconds is far below 2^53 for any process that will ever run"
)]
fn seconds(nanoseconds: u64) -> f64 {
  nanoseconds as f64 / 1e9
}

fn add(counter: Option<&AtomicU64>, amount: u64) {
  if let Some(counter) = counter {
    counter.fetch_add(amount, Ordering::Relaxed);
  }
}

fn read(counter: &AtomicU64) -> u64 {
  counter.load(Ordering::Relaxed)
}

/// How long something took, counted into [`BUCKETS`].
#[derive(Debug, Default)]
struct Histogram {
  /// How many took at most each bound and more than the one before, and last, how many took longer than all.
  counts: [AtomicU64; 12],
  sum_ns: AtomicU64,
}

impl Histogram {
  fn observe(&self, elapsed: Duration) {
    let elapsed = nanoseconds(elapsed);
    // A value exactly at a bound belongs to that bound: `le` is "less than or equal".
    let bucket = BUCKETS
      .iter()
      .position(|(bound, _)| elapsed <= *bound)
      .unwrap_or(BUCKETS.len());
    add(self.counts.get(bucket), 1);
    self.sum_ns.fetch_add(elapsed, Ordering::Relaxed);
  }

  fn count(&self) -> u64 {
    self
      .counts
      .iter()
      .fold(0_u64, |total, count| total.saturating_add(read(count)))
  }

  /// Writes the buckets, the sum, and the count of the series `name{labels}`. `labels` is `key="value"` or
  /// nothing.
  fn render(&self, out: &mut String, name: &str, labels: &str) {
    let separator = if labels.is_empty() { "" } else { "," };
    let mut cumulative = 0_u64;
    for ((_, le), count) in BUCKETS.iter().zip(&self.counts) {
      cumulative = cumulative.saturating_add(read(count));
      let _ = writeln!(out, "{name}_bucket{{{labels}{separator}le=\"{le}\"}} {cumulative}");
    }
    let total = self.count();
    let _ = writeln!(out, "{name}_bucket{{{labels}{separator}le=\"+Inf\"}} {total}");
    let braces = if labels.is_empty() {
      String::new()
    } else {
      format!("{{{labels}}}")
    };
    let _ = writeln!(out, "{name}_sum{braces} {}", seconds(read(&self.sum_ns)));
    let _ = writeln!(out, "{name}_count{braces} {total}");
  }
}

/// What is read from the rest of the server each time the metrics are asked for: counts that are kept where
/// they happen, by the embedding thread and the engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Polled {
  /// What the embedding thread has done so far. `None` for an embedder that is not a thread.
  pub worker: Option<WorkerSnapshot>,
  /// Requests waiting for the model right now.
  pub waiting: u64,
  /// Writes on their way to the store right now.
  pub writes_in_flight: u64,
  /// Writes that finished, or failed, after the request that asked for them had gone (D38).
  pub writes_unanswered: u64,
}

/// The counts a server keeps about itself.
#[derive(Debug, Default)]
pub struct Metrics {
  fingerprint: String,
  ready: AtomicU64,
  requests: [[AtomicU64; 17]; 6],
  in_flight: [AtomicU64; 6],
  request_seconds: [Histogram; 6],
  evaluations: [[AtomicU64; 3]; 4],
  matches: [AtomicU64; 3],
  import_records: [AtomicU64; 3],
  store_seconds: [Histogram; 8],
  store_errors: [AtomicU64; 8],
}

impl Metrics {
  /// Counts for a server running the model with this fingerprint, all at zero.
  #[must_use]
  pub fn new(fingerprint: &str) -> Self {
    Self {
      fingerprint: fingerprint.to_owned(),
      ..Self::default()
    }
  }

  /// A request has arrived. What is returned counts it as in flight until it is answered, and as `CANCELLED` if
  /// it is dropped without an answer, which is what the transport does to a request whose caller has gone.
  #[must_use]
  pub fn begin(self: &Arc<Self>, call: Call) -> Request {
    add(self.in_flight.get(call.index()), 1);
    Request {
      metrics: Arc::clone(self),
      call,
      started: Instant::now(),
      answered: false,
    }
  }

  /// The server's readiness answer changed, or was given for the first time.
  pub fn set_ready(&self, ready: bool) {
    self.ready.store(u64::from(ready), Ordering::Relaxed);
  }

  /// An `Evaluate` was answered with this verdict, having asked for this mode.
  pub fn evaluated(&self, verdict: Verdict, mode: WriteMode) {
    let row = VERDICTS.iter().position(|(known, _)| *known == verdict);
    let column = MODES.iter().position(|(known, _)| *known == mode);
    if let (Some(row), Some(column)) = (row, column) {
      add(self.evaluations.get(row).and_then(|row| row.get(column)), 1);
    }
  }

  /// A `Match` was answered with this decision.
  pub fn matched(&self, decision: Decision) {
    let place = DECISIONS.iter().position(|(known, _)| *known == decision);
    add(place.and_then(|place| self.matches.get(place)), 1);
  }

  /// An `Import` stream ended, having created, found stored, and failed this many records.
  pub fn imported(&self, created: u64, existing: u64, failed: u64) {
    for (counter, amount) in self.import_records.iter().zip([created, existing, failed]) {
      counter.fetch_add(amount, Ordering::Relaxed);
    }
  }

  /// The store finished an operation, in this long, and with an error or without.
  pub fn stored(&self, operation: Operation, elapsed: Duration, failed: bool) {
    if let Some(histogram) = self.store_seconds.get(operation.index()) {
      histogram.observe(elapsed);
    }
    if failed {
      add(self.store_errors.get(operation.index()), 1);
    }
  }

  fn answered(&self, call: Call, code: Code, elapsed: Duration) {
    if let Some(in_flight) = self.in_flight.get(call.index()) {
      // Never below zero, whatever happens: a gauge that wrapped would read as eighteen quintillion requests.
      let _ = in_flight.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |now| now.checked_sub(1));
    }
    // A status code is its number, from 0 to 16, which is its place in `CODES`.
    let code = usize::try_from(i32::from(code)).unwrap_or(2);
    add(self.requests.get(call.index()).and_then(|codes| codes.get(code)), 1);
    if let Some(histogram) = self.request_seconds.get(call.index()) {
      histogram.observe(elapsed);
    }
  }

  /// Everything, in the Prometheus text format.
  ///
  /// A series that says nothing is left out: a status code no request has been answered with, and the latency of
  /// a call or a store operation that has never been used. `OK` is always there for every call, at zero if need
  /// be, so that a rate over it exists from the first scrape.
  #[must_use]
  pub fn render(&self, polled: &Polled) -> String {
    let mut out = String::new();
    self.render_identity(&mut out);
    self.render_requests(&mut out);
    self.render_outcomes(&mut out);
    if let Some(worker) = &polled.worker {
      render_worker(&mut out, worker, polled.waiting);
    }
    self.render_store(&mut out);
    family(
      &mut out,
      "record_similarity_writes_in_flight",
      "gauge",
      "Writes on their way to the store.",
    );
    let _ = writeln!(out, "record_similarity_writes_in_flight {}", polled.writes_in_flight);
    family(
      &mut out,
      "record_similarity_writes_unanswered_total",
      "counter",
      "Writes that finished, or failed, after the request that asked for them had gone. Each has a log line.",
    );
    let _ = writeln!(
      out,
      "record_similarity_writes_unanswered_total {}",
      polled.writes_unanswered
    );
    out
  }

  fn render_identity(&self, out: &mut String) {
    family(
      out,
      "record_similarity_build_info",
      "gauge",
      "Always 1. The version of the server is the label.",
    );
    let _ = writeln!(
      out,
      "record_similarity_build_info{{version=\"{}\"}} 1",
      escaped(env!("CARGO_PKG_VERSION"))
    );
    family(
      out,
      "record_similarity_model_info",
      "gauge",
      "Always 1. The fingerprint of the model the server is running is the label.",
    );
    let _ = writeln!(
      out,
      "record_similarity_model_info{{fingerprint=\"{}\"}} 1",
      escaped(&self.fingerprint)
    );
    family(
      out,
      "record_similarity_ready",
      "gauge",
      "1 while the server reports itself ready (the embedding thread is alive and the store answers), 0 otherwise.",
    );
    let _ = writeln!(out, "record_similarity_ready {}", read(&self.ready));
  }

  fn render_requests(&self, out: &mut String) {
    family(
      out,
      "record_similarity_requests_total",
      "counter",
      "Requests that ended, by call and gRPC status code. CANCELLED is a request whose caller went away.",
    );
    for (call, codes) in Call::ALL.iter().zip(&self.requests) {
      for (place, (code, count)) in CODES.iter().zip(codes).enumerate() {
        let count = read(count);
        if place == 0 || count > 0 {
          let _ = writeln!(
            out,
            "record_similarity_requests_total{{call=\"{}\",code=\"{code}\"}} {count}",
            call.name()
          );
        }
      }
    }
    family(
      out,
      "record_similarity_requests_in_flight",
      "gauge",
      "Requests being served right now, by call. An Import stream is one for as long as it is open.",
    );
    for (call, in_flight) in Call::ALL.iter().zip(&self.in_flight) {
      let _ = writeln!(
        out,
        "record_similarity_requests_in_flight{{call=\"{}\"}} {}",
        call.name(),
        read(in_flight)
      );
    }
    family(
      out,
      "record_similarity_request_duration_seconds",
      "histogram",
      "How long requests took, from the first line of the handler to the answer, by call, whatever the answer.",
    );
    for (call, histogram) in Call::ALL.iter().zip(&self.request_seconds) {
      if histogram.count() > 0 {
        let labels = format!("call=\"{}\"", call.name());
        histogram.render(out, "record_similarity_request_duration_seconds", &labels);
      }
    }
  }

  fn render_outcomes(&self, out: &mut String) {
    family(
      out,
      "record_similarity_evaluations_total",
      "counter",
      "Evaluate calls that were answered, by verdict and by the write mode the request asked for.",
    );
    for ((_, verdict), modes) in VERDICTS.iter().zip(&self.evaluations) {
      for ((_, mode), count) in MODES.iter().zip(modes) {
        let _ = writeln!(
          out,
          "record_similarity_evaluations_total{{verdict=\"{verdict}\",mode=\"{mode}\"}} {}",
          read(count)
        );
      }
    }
    family(
      out,
      "record_similarity_matches_total",
      "counter",
      "Match calls that were answered, by decision.",
    );
    for ((_, decision), count) in DECISIONS.iter().zip(&self.matches) {
      let _ = writeln!(
        out,
        "record_similarity_matches_total{{decision=\"{decision}\"}} {}",
        read(count)
      );
    }
    family(
      out,
      "record_similarity_import_records_total",
      "counter",
      "Records of Import streams that have ended, by what became of each.",
    );
    for (outcome, count) in IMPORT_OUTCOMES.iter().zip(&self.import_records) {
      let _ = writeln!(
        out,
        "record_similarity_import_records_total{{outcome=\"{outcome}\"}} {}",
        read(count)
      );
    }
  }

  fn render_store(&self, out: &mut String) {
    family(
      out,
      "record_similarity_store_duration_seconds",
      "histogram",
      "How long the store took, by operation, whether it succeeded or not.",
    );
    for (operation, histogram) in Operation::ALL.iter().zip(&self.store_seconds) {
      if histogram.count() > 0 {
        let labels = format!("operation=\"{}\"", operation.name());
        histogram.render(out, "record_similarity_store_duration_seconds", &labels);
      }
    }
    family(
      out,
      "record_similarity_store_errors_total",
      "counter",
      "Store operations that ended in an error, by operation.",
    );
    for (operation, errors) in Operation::ALL.iter().zip(&self.store_errors) {
      let _ = writeln!(
        out,
        "record_similarity_store_errors_total{{operation=\"{}\"}} {}",
        operation.name(),
        read(errors)
      );
    }
  }
}

/// Writes the two lines that introduce a metric.
fn family(out: &mut String, name: &str, kind: &str, help: &str) {
  let _ = writeln!(out, "# HELP {name} {help}");
  let _ = writeln!(out, "# TYPE {name} {kind}");
}

/// A label value as the text format wants it: backslash, quote, and newline escaped.
fn escaped(value: &str) -> String {
  value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn render_worker(out: &mut String, worker: &WorkerSnapshot, waiting: u64) {
  let mut counter = |name: &str, help: &str, value: String| {
    family(out, name, "counter", help);
    let _ = writeln!(out, "{name} {value}");
  };
  counter(
    "record_similarity_embed_passes_total",
    "Forward passes of the model. Texts over passes is how many texts share one.",
    worker.batches.to_string(),
  );
  counter(
    "record_similarity_embed_requests_total",
    "Requests the embedding thread served.",
    worker.jobs.to_string(),
  );
  counter(
    "record_similarity_embed_texts_total",
    "Texts embedded.",
    worker.texts.to_string(),
  );
  counter(
    "record_similarity_embed_busy_seconds_total",
    "Time spent inside the model. Its rate is the share of the time the model is busy: at 1 it is the limit.",
    seconds(worker.busy_ns).to_string(),
  );
  counter(
    "record_similarity_embed_queue_wait_seconds_total",
    "Time requests waited for the model, summed over the requests served.",
    seconds(worker.queue_wait_ns).to_string(),
  );
  counter(
    "record_similarity_embed_abandoned_total",
    "Requests passed over because their caller had gone before the model reached them.",
    worker.abandoned.to_string(),
  );
  family(
    out,
    "record_similarity_embed_refused_total",
    "counter",
    "Requests refused without being embedded: queue_full is RESOURCE_EXHAUSTED, too_late is DEADLINE_EXCEEDED.",
  );
  let _ = writeln!(
    out,
    "record_similarity_embed_refused_total{{reason=\"queue_full\"}} {}",
    worker.overloaded
  );
  let _ = writeln!(
    out,
    "record_similarity_embed_refused_total{{reason=\"too_late\"}} {}",
    worker.too_late
  );
  family(
    out,
    "record_similarity_embed_waiting",
    "gauge",
    "Requests waiting for the model right now. model.queue_depth is the most there can be.",
  );
  let _ = writeln!(out, "record_similarity_embed_waiting {waiting}");
}

/// One request, from its arrival to its answer (see [`Metrics::begin`]).
#[derive(Debug)]
pub struct Request {
  metrics: Arc<Metrics>,
  call: Call,
  started: Instant,
  answered: bool,
}

impl Request {
  /// Counts the request as answered the way `answer` says, and hands the answer on.
  ///
  /// # Errors
  ///
  /// Returns `answer`'s error, unchanged.
  pub fn answered<T>(self, answer: Result<T, Status>) -> Result<T, Status> {
    self.finished(answer.as_ref().map_or_else(Status::code, |_| Code::Ok));
    answer
  }

  /// Counts the request as ended with `code`.
  pub fn finished(mut self, code: Code) {
    self.answered = true;
    self.metrics.answered(self.call, code, self.started.elapsed());
  }
}

impl Drop for Request {
  fn drop(&mut self) {
    if !self.answered {
      // Dropped with the handler: the caller's deadline ran out, or it hung up. `CANCELLED` is what the transport
      // tells a caller that is still listening.
      self
        .metrics
        .answered(self.call, Code::Cancelled, self.started.elapsed());
    }
  }
}

/// A store whose every operation is timed and counted. It changes nothing about what the store does.
///
/// An operation is counted when it ends. One that is dropped half way, with the request that was waiting for it,
/// is not counted at all: there is no duration to give it.
#[derive(Debug)]
pub struct Measured<S> {
  inner: S,
  metrics: Arc<Metrics>,
}

impl<S> Measured<S> {
  /// Wraps `inner`, counting into `metrics`.
  pub const fn new(inner: S, metrics: Arc<Metrics>) -> Self {
    Self { inner, metrics }
  }

  async fn timed<T>(
    &self,
    operation: Operation,
    work: impl Future<Output = Result<T, StoreError>>,
  ) -> Result<T, StoreError> {
    let started = Instant::now();
    let result = work.await;
    self.metrics.stored(operation, started.elapsed(), result.is_err());
    result
  }
}

impl<S: VectorStore> VectorStore for Measured<S> {
  async fn check(&self) -> Result<(), StoreError> {
    self.timed(Operation::Check, self.inner.check()).await
  }

  async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
    self.timed(Operation::Get, self.inner.get(id)).await
  }

  async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
    self.timed(Operation::Put, self.inner.put(record)).await
  }

  async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
    self.timed(Operation::PutIf, self.inner.put_if(record, expected)).await
  }

  async fn delete(&self, id: &str) -> Result<bool, StoreError> {
    self.timed(Operation::Delete, self.inner.delete(id)).await
  }

  async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
    self.timed(Operation::DeleteIf, self.inner.delete_if(id, version)).await
  }

  async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
    self.timed(Operation::Search, self.inner.search(query)).await
  }

  async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
    self
      .timed(Operation::SearchVectors, self.inner.search_vectors(query))
      .await
  }
}

/// Serves what `render` returns at [`PATH`] over HTTP on `listener`, until `stopped` resolves.
///
/// One path, `GET` only, and nothing else: no index page, no health check (the gRPC health service is that), and
/// nothing that can change the server. It is plain HTTP with no authentication, like the gRPC port, and belongs
/// on the same private network.
///
/// # Errors
///
/// Returns an error if the listener fails.
pub async fn serve(
  listener: TcpListener,
  render: impl Fn() -> String + Send + Sync + 'static,
  stopped: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
  let render = Arc::new(render);
  let page = axum::routing::get(move || {
    let render = Arc::clone(&render);
    async move { ([(axum::http::header::CONTENT_TYPE, CONTENT_TYPE)], render()) }
  });
  axum::serve(listener, axum::Router::new().route(PATH, page))
    .with_graceful_shutdown(stopped)
    .await
    .context("the metrics endpoint stopped with an error")
}

#[cfg(test)]
mod tests {
  use similarity_store::MemoryStore;
  use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

  use super::*;

  fn metrics() -> Arc<Metrics> {
    Arc::new(Metrics::new("f46e-test"))
  }

  /// The value of the one series that is exactly `series`, or `None` if it is not there.
  fn value(page: &str, series: &str) -> Option<String> {
    page
      .lines()
      .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '))
      .map(str::to_owned)
  }

  fn lines_of<'a>(page: &'a str, metric: &str) -> Vec<&'a str> {
    page
      .lines()
      .filter(|line| !line.starts_with('#') && line.starts_with(metric))
      .collect()
  }

  /// A server that has done nothing still says who it is and that its counts are zero. `OK` is there for every
  /// call from the first scrape; no other status code is, and no latency of anything that has not happened.
  #[test]
  fn a_server_that_has_done_nothing_reports_zeros_and_no_series_that_says_nothing() {
    let page = metrics().render(&Polled::default());
    assert_eq!(
      value(&page, "record_similarity_model_info{fingerprint=\"f46e-test\"}").as_deref(),
      Some("1")
    );
    let version = format!(
      "record_similarity_build_info{{version=\"{}\"}}",
      env!("CARGO_PKG_VERSION")
    );
    assert_eq!(value(&page, &version).as_deref(), Some("1"));
    assert_eq!(value(&page, "record_similarity_ready").as_deref(), Some("0"));

    let requests = lines_of(&page, "record_similarity_requests_total");
    assert_eq!(requests.len(), 6, "{requests:?}");
    for call in ["Describe", "Evaluate", "FindSimilar", "Match", "Delete", "Import"] {
      let series = format!("record_similarity_requests_total{{call=\"{call}\",code=\"OK\"}}");
      assert_eq!(value(&page, &series).as_deref(), Some("0"), "{call}");
    }
    assert_eq!(lines_of(&page, "record_similarity_request_duration_seconds"), [""; 0]);
    assert_eq!(lines_of(&page, "record_similarity_store_duration_seconds"), [""; 0]);
    assert_eq!(lines_of(&page, "record_similarity_evaluations_total").len(), 12);
    assert_eq!(lines_of(&page, "record_similarity_store_errors_total").len(), 8);
    // Not a thread, so nothing about one.
    assert_eq!(lines_of(&page, "record_similarity_embed_"), [""; 0]);
    assert_eq!(value(&page, "record_similarity_writes_in_flight").as_deref(), Some("0"));
    assert_eq!(
      value(&page, "record_similarity_writes_unanswered_total").as_deref(),
      Some("0")
    );
  }

  /// Every metric is introduced by a HELP and a TYPE line, once, before its first sample, and every sample
  /// belongs to a metric that was introduced: what a parser of the format requires.
  #[test]
  fn every_sample_follows_the_help_and_type_of_its_metric() {
    let metrics = metrics();
    metrics.begin(Call::Evaluate).finished(Code::Ok);
    metrics.stored(Operation::Get, Duration::from_millis(3), false);
    let polled = Polled {
      worker: Some(WorkerSnapshot::default()),
      ..Polled::default()
    };
    let page = metrics.render(&polled);
    let mut introduced: Vec<(String, String)> = Vec::new();
    let mut helped = None;
    for line in page.lines() {
      if let Some(rest) = line.strip_prefix("# HELP ") {
        let name = rest.split(' ').next().unwrap();
        assert!(rest.len() > name.len() + 1, "no help text: {line}");
        helped = Some(name.to_owned());
      } else if let Some(rest) = line.strip_prefix("# TYPE ") {
        let (name, kind) = rest.split_once(' ').unwrap();
        assert_eq!(helped.as_deref(), Some(name), "TYPE without its HELP: {line}");
        assert!(["counter", "gauge", "histogram"].contains(&kind), "{line}");
        assert!(
          introduced.iter().all(|(known, _)| known != name),
          "introduced twice: {name}"
        );
        introduced.push((name.to_owned(), kind.to_owned()));
      } else {
        let name = line.split(['{', ' ']).next().unwrap();
        let (family, kind) = introduced.last().unwrap();
        let belongs = if kind == "histogram" {
          ["_bucket", "_sum", "_count"]
            .iter()
            .any(|suffix| name.strip_suffix(suffix) == Some(family.as_str()))
        } else {
          name == family
        };
        assert!(belongs, "`{line}` is not a sample of `{family}`");
        assert_eq!(kind == "counter", name.ends_with("_total"), "{line}");
        let sample = line.rsplit(' ').next().unwrap();
        assert!(sample.parse::<f64>().is_ok(), "not a number: {line}");
      }
    }
    assert!(page.ends_with('\n'));
    assert!(introduced.len() > 15, "{introduced:?}");
  }

  /// A request is in flight from its arrival to its answer, and is then counted once, under the status code it
  /// was answered with, with the time it took.
  #[test]
  fn a_request_is_counted_in_flight_and_then_by_how_it_was_answered() {
    let metrics = metrics();
    let in_flight = "record_similarity_requests_in_flight{call=\"Evaluate\"}";
    let request = metrics.begin(Call::Evaluate);
    assert_eq!(
      value(&metrics.render(&Polled::default()), in_flight).as_deref(),
      Some("1")
    );
    assert_eq!(request.answered(Ok(7)).unwrap(), 7);

    let refused: Result<(), Status> = Err(Status::resource_exhausted("the embedding queue is full"));
    let error = metrics.begin(Call::Evaluate).answered(refused).unwrap_err();
    assert_eq!(
      error.code(),
      Code::ResourceExhausted,
      "the answer is handed on as it was"
    );
    metrics.begin(Call::Match).finished(Code::InvalidArgument);

    let page = metrics.render(&Polled::default());
    assert_eq!(value(&page, in_flight).as_deref(), Some("0"));
    let count = |call: &str, code: &str| {
      value(
        &page,
        &format!("record_similarity_requests_total{{call=\"{call}\",code=\"{code}\"}}"),
      )
    };
    assert_eq!(count("Evaluate", "OK").as_deref(), Some("1"));
    assert_eq!(count("Evaluate", "RESOURCE_EXHAUSTED").as_deref(), Some("1"));
    assert_eq!(count("Match", "INVALID_ARGUMENT").as_deref(), Some("1"));
    assert_eq!(count("Match", "OK").as_deref(), Some("0"));
    assert_eq!(count("Delete", "INVALID_ARGUMENT"), None, "nothing to say, so not said");
    assert_eq!(
      value(
        &page,
        "record_similarity_request_duration_seconds_count{call=\"Evaluate\"}"
      )
      .as_deref(),
      Some("2")
    );
    assert_eq!(
      value(
        &page,
        "record_similarity_request_duration_seconds_count{call=\"Match\"}"
      )
      .as_deref(),
      Some("1")
    );
    assert_eq!(
      value(
        &page,
        "record_similarity_request_duration_seconds_count{call=\"Delete\"}"
      ),
      None
    );
  }

  /// The transport drops a handler whose caller has gone. That request ended too, and is counted as what the
  /// transport calls it.
  #[test]
  fn a_request_dropped_without_an_answer_is_counted_as_cancelled() {
    let metrics = metrics();
    drop(metrics.begin(Call::FindSimilar));
    let page = metrics.render(&Polled::default());
    assert_eq!(
      value(
        &page,
        "record_similarity_requests_total{call=\"FindSimilar\",code=\"CANCELLED\"}"
      )
      .as_deref(),
      Some("1")
    );
    assert_eq!(
      value(&page, "record_similarity_requests_in_flight{call=\"FindSimilar\"}").as_deref(),
      Some("0")
    );
    assert_eq!(
      value(
        &page,
        "record_similarity_request_duration_seconds_count{call=\"FindSimilar\"}"
      )
      .as_deref(),
      Some("1")
    );
  }

  /// Every status code is counted, under its protocol name: the names are in the order of the codes' numbers.
  #[test]
  fn every_status_code_is_counted_under_its_own_name() {
    let all = [
      (Code::Ok, "OK"),
      (Code::Cancelled, "CANCELLED"),
      (Code::Unknown, "UNKNOWN"),
      (Code::InvalidArgument, "INVALID_ARGUMENT"),
      (Code::DeadlineExceeded, "DEADLINE_EXCEEDED"),
      (Code::NotFound, "NOT_FOUND"),
      (Code::AlreadyExists, "ALREADY_EXISTS"),
      (Code::PermissionDenied, "PERMISSION_DENIED"),
      (Code::ResourceExhausted, "RESOURCE_EXHAUSTED"),
      (Code::FailedPrecondition, "FAILED_PRECONDITION"),
      (Code::Aborted, "ABORTED"),
      (Code::OutOfRange, "OUT_OF_RANGE"),
      (Code::Unimplemented, "UNIMPLEMENTED"),
      (Code::Internal, "INTERNAL"),
      (Code::Unavailable, "UNAVAILABLE"),
      (Code::DataLoss, "DATA_LOSS"),
      (Code::Unauthenticated, "UNAUTHENTICATED"),
    ];
    let metrics = metrics();
    for (times, (code, _)) in all.iter().enumerate() {
      for _ in 0..=times {
        metrics.begin(Call::Delete).finished(*code);
      }
    }
    let page = metrics.render(&Polled::default());
    for (times, (_, name)) in all.iter().enumerate() {
      let series = format!("record_similarity_requests_total{{call=\"Delete\",code=\"{name}\"}}");
      assert_eq!(value(&page, &series), Some((times + 1).to_string()), "{name}");
    }
  }

  /// A latency is counted into every bucket whose bound it does not exceed, the bound itself included, and the
  /// buckets a parser reads are cumulative and end at the count.
  #[test]
  fn a_latency_at_a_bound_belongs_to_that_bound() {
    let histogram = Histogram::default();
    histogram.observe(Duration::from_micros(999));
    histogram.observe(Duration::from_millis(1));
    histogram.observe(Duration::from_nanos(1_000_001));
    histogram.observe(Duration::from_millis(250));
    histogram.observe(Duration::from_mins(1));
    let mut page = String::new();
    histogram.render(&mut page, "latency", "call=\"x\"");
    let bucket = |le: &str| value(&page, &format!("latency_bucket{{call=\"x\",le=\"{le}\"}}")).unwrap();
    assert_eq!(bucket("0.001"), "2");
    assert_eq!(bucket("0.0025"), "3");
    assert_eq!(bucket("0.1"), "3");
    assert_eq!(bucket("0.25"), "4");
    assert_eq!(bucket("2.5"), "4");
    assert_eq!(bucket("+Inf"), "5", "longer than every bound is still counted");
    assert_eq!(value(&page, "latency_count{call=\"x\"}").as_deref(), Some("5"));
    let sum: f64 = value(&page, "latency_sum{call=\"x\"}").unwrap().parse().unwrap();
    assert!((sum - 60.252_999_001).abs() < 1e-9, "{sum}");
    assert_eq!(page.lines().count(), BUCKETS.len() + 3);

    // With no labels of its own, a series has only `le`, and its sum and count have no braces at all.
    let mut bare = String::new();
    histogram.render(&mut bare, "latency", "");
    assert_eq!(value(&bare, "latency_bucket{le=\"0.001\"}").as_deref(), Some("2"));
    assert_eq!(value(&bare, "latency_count").as_deref(), Some("5"));
  }

  #[test]
  fn verdicts_decisions_and_imported_records_are_counted_by_what_they_were() {
    let metrics = metrics();
    metrics.evaluated(Verdict::Accepted, WriteMode::Enforce);
    metrics.evaluated(Verdict::Accepted, WriteMode::Enforce);
    metrics.evaluated(Verdict::Rejected, WriteMode::DryRun);
    metrics.evaluated(Verdict::Rejected, WriteMode::Force);
    metrics.evaluated(Verdict::Created, WriteMode::Enforce);
    metrics.evaluated(Verdict::Embedded, WriteMode::Enforce);
    metrics.matched(Decision::Matched);
    metrics.matched(Decision::NotJudged);
    metrics.matched(Decision::NotJudged);
    metrics.matched(Decision::NoMatch);
    metrics.imported(5, 2, 1);
    metrics.imported(1, 0, 0);
    metrics.set_ready(true);

    let page = metrics.render(&Polled::default());
    let evaluations = |verdict: &str, mode: &str| {
      value(
        &page,
        &format!("record_similarity_evaluations_total{{verdict=\"{verdict}\",mode=\"{mode}\"}}"),
      )
      .unwrap()
    };
    assert_eq!(evaluations("accepted", "enforce"), "2");
    assert_eq!(evaluations("rejected", "dry_run"), "1");
    assert_eq!(evaluations("rejected", "force"), "1");
    assert_eq!(evaluations("rejected", "enforce"), "0");
    assert_eq!(evaluations("created", "enforce"), "1");
    assert_eq!(evaluations("embedded", "enforce"), "1");
    let matches = |decision: &str| {
      value(
        &page,
        &format!("record_similarity_matches_total{{decision=\"{decision}\"}}"),
      )
      .unwrap()
    };
    assert_eq!(
      (matches("matched"), matches("no_match"), matches("not_judged")),
      ("1".to_owned(), "1".to_owned(), "2".to_owned())
    );
    let imported = |outcome: &str| {
      value(
        &page,
        &format!("record_similarity_import_records_total{{outcome=\"{outcome}\"}}"),
      )
      .unwrap()
    };
    assert_eq!(
      (imported("created"), imported("exists"), imported("failed")),
      ("6".to_owned(), "2".to_owned(), "1".to_owned())
    );
    assert_eq!(value(&page, "record_similarity_ready").as_deref(), Some("1"));
    metrics.set_ready(false);
    assert_eq!(
      value(&metrics.render(&Polled::default()), "record_similarity_ready").as_deref(),
      Some("0")
    );
  }

  /// What the embedding thread and the engine count themselves is written as it is read, in seconds where it
  /// is a time.
  #[test]
  fn what_the_thread_and_the_engine_count_is_written_as_it_is_read() {
    let polled = Polled {
      worker: Some(WorkerSnapshot {
        batches: 4,
        jobs: 9,
        texts: 31,
        overloaded: 2,
        abandoned: 3,
        too_late: 5,
        busy_ns: 1_500_000_000,
        queue_wait_ns: 250_000_000,
      }),
      waiting: 7,
      writes_in_flight: 1,
      writes_unanswered: 6,
    };
    let page = metrics().render(&polled);
    let read = |series: &str| value(&page, series).unwrap();
    assert_eq!(read("record_similarity_embed_passes_total"), "4");
    assert_eq!(read("record_similarity_embed_requests_total"), "9");
    assert_eq!(read("record_similarity_embed_texts_total"), "31");
    assert_eq!(read("record_similarity_embed_busy_seconds_total"), "1.5");
    assert_eq!(read("record_similarity_embed_queue_wait_seconds_total"), "0.25");
    assert_eq!(read("record_similarity_embed_abandoned_total"), "3");
    assert_eq!(
      read("record_similarity_embed_refused_total{reason=\"queue_full\"}"),
      "2"
    );
    assert_eq!(read("record_similarity_embed_refused_total{reason=\"too_late\"}"), "5");
    assert_eq!(read("record_similarity_embed_waiting"), "7");
    assert_eq!(read("record_similarity_writes_in_flight"), "1");
    assert_eq!(read("record_similarity_writes_unanswered_total"), "6");
  }

  #[test]
  fn a_label_value_cannot_break_out_of_its_quotes() {
    assert_eq!(escaped("plain-1.0"), "plain-1.0");
    assert_eq!(escaped("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    let page = Metrics::new("odd\"one").render(&Polled::default());
    assert_eq!(
      value(&page, "record_similarity_model_info{fingerprint=\"odd\\\"one\"}").as_deref(),
      Some("1")
    );
  }

  fn record(id: &str) -> StoredRecord {
    let vectors = similarity_core::RecordVectors {
      composite: vec![1.0, 0.0],
      hashes: std::collections::BTreeMap::new(),
    };
    StoredRecord {
      id: id.to_owned(),
      brand_id: None,
      anchor: vectors.composite.clone(),
      vectors,
      quantities: similarity_core::Quantities::default(),
      anchor_quantities: similarity_core::Quantities::default(),
      revision: 1,
      updated_at: 0,
    }
  }

  /// A measured store is the store: every operation gives what the store gives. And each is counted once, under
  /// its own name, an error as an error.
  #[tokio::test]
  async fn a_measured_store_is_the_store_and_counts_each_operation_under_its_name() {
    let metrics = metrics();
    let store = Measured::new(MemoryStore::new("layout", 2), Arc::clone(&metrics));
    store.check().await.unwrap();
    assert_eq!(store.get("p1").await.unwrap(), None);
    assert!(store.put_if(&record("p1"), Expected::Absent).await.unwrap());
    assert!(!store.put_if(&record("p1"), Expected::Absent).await.unwrap());
    let stored = store.get("p1").await.unwrap().unwrap();
    assert_eq!(stored, record("p1"));
    store.put(&record("p2")).await.unwrap();
    let query = || SearchQuery {
      vector: &[1.0, 0.0],
      limit: 5,
      brand_id: None,
      exclude_id: None,
      min_similarity: None,
    };
    assert_eq!(store.search(query()).await.unwrap().len(), 2);
    assert_eq!(store.search_vectors(query()).await.unwrap().len(), 2);
    assert!(store.delete_if("p1", stored.version()).await.unwrap());
    assert!(store.delete("p2").await.unwrap());
    // A record of another layout is the store's error to give, and it is counted as one.
    let mut foreign = record("p3");
    foreign.vectors.composite = vec![1.0];
    assert!(store.put(&foreign).await.is_err());

    let page = metrics.render(&Polled::default());
    let count = |operation: &str| {
      value(
        &page,
        &format!("record_similarity_store_duration_seconds_count{{operation=\"{operation}\"}}"),
      )
    };
    let expected = [
      ("check", 1),
      ("get", 2),
      ("put", 2),
      ("put_if", 2),
      ("delete", 1),
      ("delete_if", 1),
      ("search", 1),
      ("search_vectors", 1),
    ];
    for (operation, times) in expected {
      assert_eq!(count(operation), Some(times.to_string()), "{operation}");
      let errors = value(
        &page,
        &format!("record_similarity_store_errors_total{{operation=\"{operation}\"}}"),
      );
      assert_eq!(errors, Some(u8::from(operation == "put").to_string()), "{operation}");
    }
  }

  /// What a scraper does: a plain HTTP `GET`, read to the end.
  async fn get(address: std::net::SocketAddr, path: &str) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    let request = format!("GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut answer = String::new();
    stream.read_to_string(&mut answer).await.unwrap();
    let (head, body) = answer.split_once("\r\n\r\n").unwrap();
    (head.to_lowercase(), body.to_owned())
  }

  /// The endpoint is one path. It answers a plain `GET` with what it is given to say, as the text format, each
  /// time anew, and knows no other path.
  #[tokio::test]
  async fn the_endpoint_serves_the_page_at_one_path_and_nothing_anywhere_else() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let scrapes = Arc::new(AtomicU64::new(0));
    let counted = Arc::clone(&scrapes);
    let render = move || format!("scrapes {}\n", counted.fetch_add(1, Ordering::Relaxed));
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve(listener, render, async {
      let _ = stopped.await;
    }));

    let (head, body) = get(address, PATH).await;
    assert!(head.starts_with("http/1.1 200"), "{head}");
    assert!(
      head.contains("content-type: text/plain; version=0.0.4; charset=utf-8"),
      "{head}"
    );
    assert!(body.contains("scrapes 0"), "{body}");
    let (_, body) = get(address, PATH).await;
    assert!(body.contains("scrapes 1"), "asked again, answered again: {body}");

    for elsewhere in ["/", "/metrics/", "/health", "/metricsx"] {
      let (head, _) = get(address, elsewhere).await;
      assert!(head.starts_with("http/1.1 404"), "{elsewhere}: {head}");
    }
    assert_eq!(
      scrapes.load(Ordering::Relaxed),
      2,
      "a path that is not there renders nothing"
    );

    stop.send(()).unwrap();
    server.await.unwrap().unwrap();
  }
}
