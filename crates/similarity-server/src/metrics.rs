//! Numbers about the process, for a monitoring system to read (docs/decisions.md, D39).
//!
//! A log line says what one request did. These say what the server is doing: how many requests of each kind it
//! answers and how, how long they take, what the one embedding thread is busy with, how the store behaves, and
//! what verdicts are going out.
//!
//! The counting is the [`metrics`] crate's, and so is the writing out: this module names what is counted and
//! says where, and the Prometheus exporter keeps the numbers and renders them in the text format that a Datadog
//! agent, an OpenTelemetry collector, and Prometheus all read ([`serve`]). Another exporter of that crate (one
//! that pushes to a Datadog agent, say) would take the same counts; nothing outside [`Metrics::exported`] knows
//! which is in use.
//!
//! The rest of the server does not use the crate's macros. It calls the methods of [`Metrics`], which is given
//! a recorder of its own rather than the process-wide one. That keeps three things true:
//!
//! - **Nothing about a record is a label.** No id, no brand, no text: a label value is one of a handful of
//!   words fixed in this file (a call, a status code, a verdict), or the model's fingerprint. Each distinct
//!   label value is a time series somebody stores and pays for, and a record's brand is nobody's business in a
//!   monitoring system. A call site cannot spell a label wrong, or invent one.
//! - **The names are an interface.** Dashboards and alerts are written against them. Add a metric or a label
//!   value; never rename one or change what it counts.
//! - **A server that is not asked for metrics keeps none** ([`Metrics::discarded`]), and two servers in one
//!   process, as in the tests, do not count into each other.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use metrics::{Counter, Gauge, Histogram, Key, KeyName, Label, Level, Metadata, NoopRecorder, Recorder, SharedString};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use similarity_core::matching::Decision;
use similarity_store::{Expected, Found, Neighbor, SearchQuery, StoreError, StoredRecord, VectorStore, Version};
use tokio::net::TcpListener;
use tonic::{Code, Status};
use tracing::Instrument as _;

use crate::engine::{Verdict, WriteMode};
use crate::worker::WorkerSnapshot;

/// The path the metrics are served at.
pub const PATH: &str = "/metrics";

/// The content type of the Prometheus text format, version 0.0.4.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// The upper bounds of the latency buckets, in seconds.
///
/// From one millisecond, under which a save with no changes is answered, to two and a half seconds, by which any
/// caller has given up. They are not thresholds and nothing is decided by them: they are how finely a latency
/// can be read back, and each one is a time series for every call and every store operation that is used.
const SECONDS: [f64; 11] = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5];

/// The upper bounds of the buckets for how many texts a forward pass held. A pass holds `model.batch_size` at
/// most, unless one request alone brings more.
const TEXTS: [f64; 8] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

const BUILD_INFO: &str = "record_similarity_build_info";
const MODEL_INFO: &str = "record_similarity_model_info";
const READY: &str = "record_similarity_ready";
const REQUESTS: &str = "record_similarity_requests_total";
const IN_FLIGHT: &str = "record_similarity_requests_in_flight";
const REQUEST_SECONDS: &str = "record_similarity_request_duration_seconds";
const EVALUATIONS: &str = "record_similarity_evaluations_total";
const MATCHES: &str = "record_similarity_matches_total";
const IMPORT_RECORDS: &str = "record_similarity_import_records_total";
const LOCK_WAIT: &str = "record_similarity_lock_wait_seconds";
const EMBED_WAIT: &str = "record_similarity_embed_wait_seconds";
const PASS_SECONDS: &str = "record_similarity_embed_pass_duration_seconds";
const PASS_TEXTS: &str = "record_similarity_embed_pass_texts";
const EMBED_REQUESTS: &str = "record_similarity_embed_requests_total";
const EMBED_ABANDONED: &str = "record_similarity_embed_abandoned_total";
const EMBED_REFUSED: &str = "record_similarity_embed_refused_total";
const EMBED_WAITING: &str = "record_similarity_embed_waiting";
const STORE_SECONDS: &str = "record_similarity_store_duration_seconds";
const STORE_ERRORS: &str = "record_similarity_store_errors_total";
const WRITES_IN_FLIGHT: &str = "record_similarity_writes_in_flight";
const WRITES_UNANSWERED: &str = "record_similarity_writes_unanswered_total";

/// What each metric is, as its `# HELP` line says it.
const COUNTERS: [(&str, &str); 8] = [
  (
    REQUESTS,
    "Requests that ended, by call and gRPC status code. CANCELLED is a request whose caller went away.",
  ),
  (
    EVALUATIONS,
    "Evaluate calls that were answered, by verdict and by the write mode the request asked for.",
  ),
  (MATCHES, "Match calls that were answered, by decision."),
  (
    IMPORT_RECORDS,
    "Records of Import streams that have ended, by what became of each.",
  ),
  (EMBED_REQUESTS, "Requests the embedding thread served."),
  (
    EMBED_ABANDONED,
    "Requests passed over because their caller had gone before the model reached them.",
  ),
  (
    EMBED_REFUSED,
    "Requests refused without being embedded: queue_full is RESOURCE_EXHAUSTED, too_late is DEADLINE_EXCEEDED.",
  ),
  (STORE_ERRORS, "Store operations that ended in an error, by operation."),
];

const GAUGES: [(&str, &str); 6] = [
  (BUILD_INFO, "Always 1. The version of the server is the label."),
  (
    MODEL_INFO,
    "Always 1. The fingerprint of the model the server is running is the label.",
  ),
  (
    READY,
    "1 while the server reports itself ready (the embedding thread is alive and the store answers), 0 otherwise.",
  ),
  (
    IN_FLIGHT,
    "Requests being served right now, by call. An Import stream is one for as long as it is open.",
  ),
  (
    EMBED_WAITING,
    "Requests waiting for the model right now. model.queue_depth is the most there can be.",
  ),
  (WRITES_IN_FLIGHT, "Writes on their way to the store."),
];

const HISTOGRAMS: [(&str, &str); 6] = [
  (
    REQUEST_SECONDS,
    "How long requests took, from the first line of the handler to the answer, by call, whatever the answer.",
  ),
  (
    LOCK_WAIT,
    "How long requests waited for their record's lock: another request in this process had it.",
  ),
  (
    EMBED_WAIT,
    "How long requests waited for the model, from joining the queue to being taken into a forward pass.",
  ),
  (
    PASS_SECONDS,
    "How long forward passes of the model took. Its sum, as a rate, is the share of the time the model is busy.",
  ),
  (PASS_TEXTS, "How many texts forward passes held."),
  (
    STORE_SECONDS,
    "How long the store took, by operation, whether it succeeded or not.",
  ),
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

  /// The call's name, as the `call` label and a request's span have it.
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::Describe => "Describe",
      Self::Evaluate => "Evaluate",
      Self::FindSimilar => "FindSimilar",
      Self::Match => "Match",
      Self::Delete => "Delete",
      Self::Import => "Import",
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

const fn decision_name(decision: Decision) -> &'static str {
  match decision {
    Decision::Matched => "matched",
    Decision::NoMatch => "no_match",
    Decision::NotJudged => "not_judged",
  }
}

/// A count of something as a gauge's value. Exact for any count below 2^53.
#[expect(
  clippy::cast_precision_loss,
  reason = "requests waiting and writes in flight are far below 2^53"
)]
const fn level(count: u64) -> f64 {
  count as f64
}

fn label(key: &'static str, value: &'static str) -> Label {
  Label::from_static_parts(key, value)
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

/// What a server counts about itself, and where the counts go.
pub struct Metrics {
  recorder: Box<dyn Recorder + Send + Sync>,
  /// The exporter's other half, which renders what the recorder holds. `None` when nothing is kept.
  exporter: Option<PrometheusHandle>,
}

impl std::fmt::Debug for Metrics {
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    formatter
      .debug_struct("Metrics")
      .field("exported", &self.exporter.is_some())
      .finish_non_exhaustive()
  }
}

impl Metrics {
  /// Counts that are kept nowhere: for a server nobody asks, a command, a test of something else. Every method
  /// works and costs next to nothing.
  #[must_use]
  pub fn discarded() -> Self {
    Self {
      recorder: Box::new(NoopRecorder),
      exporter: None,
    }
  }

  /// Counts for a server running the model with this fingerprint, kept by the Prometheus exporter and written
  /// out by [`Self::render`].
  ///
  /// Every metric is described, and every series whose zero means something is there from the start: `OK` for
  /// each call, so that a rate over it exists from the first scrape, and the verdicts, decisions, import
  /// outcomes, and store errors. A status code no request has ended with, and the latency of anything that has
  /// not happened, appear when they first do.
  ///
  /// # Errors
  ///
  /// Returns an error if the exporter refuses the buckets, which are fixed in this file.
  pub fn exported(fingerprint: &str) -> anyhow::Result<Self> {
    let recorder = PrometheusBuilder::new()
      .set_buckets(&SECONDS)
      .and_then(|builder| builder.set_buckets_for_metric(Matcher::Full(PASS_TEXTS.to_owned()), &TEXTS))
      .context("the metrics exporter refused its buckets")?
      .build_recorder();
    let metrics = Self {
      exporter: Some(recorder.handle()),
      recorder: Box::new(recorder),
    };
    for (name, help) in COUNTERS.into_iter().chain([(
      WRITES_UNANSWERED,
      "Writes that finished, or failed, after the request that asked for them had gone. Each has a log line.",
    )]) {
      let (name, help) = (KeyName::from_const_str(name), SharedString::const_str(help));
      metrics.recorder.describe_counter(name, None, help);
    }
    for (name, help) in GAUGES {
      let (name, help) = (KeyName::from_const_str(name), SharedString::const_str(help));
      metrics.recorder.describe_gauge(name, None, help);
    }
    for (name, help) in HISTOGRAMS {
      let (name, help) = (KeyName::from_const_str(name), SharedString::const_str(help));
      metrics.recorder.describe_histogram(name, None, help);
    }
    metrics
      .gauge(BUILD_INFO, vec![label("version", env!("CARGO_PKG_VERSION"))])
      .set(1.0);
    metrics
      .gauge(MODEL_INFO, vec![Label::new("fingerprint", fingerprint.to_owned())])
      .set(1.0);
    metrics.gauge(READY, Vec::new()).set(0.0);
    for call in Call::ALL {
      metrics.requests(call, 0).increment(0);
      metrics.in_flight(call).set(0.0);
    }
    for verdict in [
      Verdict::Created,
      Verdict::Accepted,
      Verdict::Rejected,
      Verdict::Embedded,
    ] {
      for mode in [WriteMode::Enforce, WriteMode::DryRun, WriteMode::Force] {
        metrics.evaluations(verdict, mode).increment(0);
      }
    }
    for decision in [Decision::Matched, Decision::NoMatch, Decision::NotJudged] {
      metrics.matches(decision).increment(0);
    }
    metrics.imported(0, 0, 0);
    for operation in Operation::ALL {
      metrics.store_errors(operation).increment(0);
    }
    Ok(metrics)
  }

  fn counter(&self, name: &'static str, labels: Vec<Label>) -> Counter {
    self
      .recorder
      .register_counter(&Key::from_parts(name, labels), &METADATA)
  }

  fn gauge(&self, name: &'static str, labels: Vec<Label>) -> Gauge {
    self.recorder.register_gauge(&Key::from_parts(name, labels), &METADATA)
  }

  fn histogram(&self, name: &'static str, labels: Vec<Label>) -> Histogram {
    self
      .recorder
      .register_histogram(&Key::from_parts(name, labels), &METADATA)
  }

  /// The count of requests of `call` that ended with the status code numbered `code`.
  fn requests(&self, call: Call, code: usize) -> Counter {
    // A number with no name in the table is counted as what it is to a caller: unknown.
    let code = CODES.get(code).copied().unwrap_or("UNKNOWN");
    self.counter(REQUESTS, vec![label("call", call.name()), label("code", code)])
  }

  fn in_flight(&self, call: Call) -> Gauge {
    self.gauge(IN_FLIGHT, vec![label("call", call.name())])
  }

  fn evaluations(&self, verdict: Verdict, mode: WriteMode) -> Counter {
    let labels = vec![label("verdict", verdict_name(verdict)), label("mode", mode_name(mode))];
    self.counter(EVALUATIONS, labels)
  }

  fn matches(&self, decision: Decision) -> Counter {
    self.counter(MATCHES, vec![label("decision", decision_name(decision))])
  }

  fn store_errors(&self, operation: Operation) -> Counter {
    self.counter(STORE_ERRORS, vec![label("operation", operation.name())])
  }

  /// A request has arrived. What is returned counts it as in flight until it is answered, and as `CANCELLED` if
  /// it is dropped without an answer, which is what the transport does to a request whose caller has gone.
  #[must_use]
  pub fn begin(self: &Arc<Self>, call: Call) -> Request {
    self.in_flight(call).increment(1.0);
    Request {
      metrics: Arc::clone(self),
      call,
      started: Instant::now(),
      answered: false,
    }
  }

  /// The server's readiness answer changed, or was given for the first time.
  pub fn set_ready(&self, ready: bool) {
    self.gauge(READY, Vec::new()).set(if ready { 1.0 } else { 0.0 });
  }

  /// An `Evaluate` was answered with this verdict, having asked for this mode.
  pub fn evaluated(&self, verdict: Verdict, mode: WriteMode) {
    self.evaluations(verdict, mode).increment(1);
  }

  /// A `Match` was answered with this decision.
  pub fn matched(&self, decision: Decision) {
    self.matches(decision).increment(1);
  }

  /// An `Import` stream ended, having created, found stored, and failed this many records.
  pub fn imported(&self, created: u64, existing: u64, failed: u64) {
    for (outcome, records) in [("created", created), ("exists", existing), ("failed", failed)] {
      self
        .counter(IMPORT_RECORDS, vec![label("outcome", outcome)])
        .increment(records);
    }
  }

  /// The store finished an operation, in this long, and with an error or without.
  pub fn stored(&self, operation: Operation, elapsed: Duration, failed: bool) {
    self
      .histogram(STORE_SECONDS, vec![label("operation", operation.name())])
      .record(elapsed.as_secs_f64());
    if failed {
      self.store_errors(operation).increment(1);
    }
  }

  /// A request waited this long for its record's lock.
  pub fn lock_waited(&self, elapsed: Duration) {
    self.histogram(LOCK_WAIT, Vec::new()).record(elapsed.as_secs_f64());
  }

  /// A request waited this long for the model before a forward pass took it.
  pub fn embed_waited(&self, elapsed: Duration) {
    self.histogram(EMBED_WAIT, Vec::new()).record(elapsed.as_secs_f64());
  }

  /// A forward pass of `texts` texts took this long.
  pub fn passed(&self, texts: u64, elapsed: Duration) {
    self.histogram(PASS_SECONDS, Vec::new()).record(elapsed.as_secs_f64());
    self.histogram(PASS_TEXTS, Vec::new()).record(level(texts));
  }

  fn answered(&self, call: Call, code: Code, elapsed: Duration) {
    self.in_flight(call).decrement(1.0);
    // A status code is its number, from 0 to 16, which is its place in `CODES`.
    self
      .requests(call, usize::try_from(i32::from(code)).unwrap_or(2))
      .increment(1);
    self
      .histogram(REQUEST_SECONDS, vec![label("call", call.name())])
      .record(elapsed.as_secs_f64());
  }

  /// Everything, in the Prometheus text format, with `polled` brought up to date first. Nothing at all for
  /// counts that are not kept.
  #[must_use]
  pub fn render(&self, polled: &Polled) -> String {
    let Some(exporter) = &self.exporter else {
      return String::new();
    };
    if let Some(worker) = &polled.worker {
      self.counter(EMBED_REQUESTS, Vec::new()).absolute(worker.jobs);
      self.counter(EMBED_ABANDONED, Vec::new()).absolute(worker.abandoned);
      self
        .counter(EMBED_REFUSED, vec![label("reason", "queue_full")])
        .absolute(worker.overloaded);
      self
        .counter(EMBED_REFUSED, vec![label("reason", "too_late")])
        .absolute(worker.too_late);
      self.gauge(EMBED_WAITING, Vec::new()).set(level(polled.waiting));
    }
    self
      .gauge(WRITES_IN_FLIGHT, Vec::new())
      .set(level(polled.writes_in_flight));
    self
      .counter(WRITES_UNANSWERED, Vec::new())
      .absolute(polled.writes_unanswered);
    exporter.render()
  }

  /// Folds what has been observed since the last time into the exporter's buckets. The exporter holds every
  /// observation until it is rendered or this is called, so a server that nobody scrapes has to call it now and
  /// then ([`UPKEEP_EVERY`]).
  pub fn upkeep(&self) {
    if let Some(exporter) = &self.exporter {
      exporter.run_upkeep();
    }
  }
}

/// How often a server that keeps metrics runs [`Metrics::upkeep`]: the exporter's own interval when it is left to
/// do it itself.
pub const UPKEEP_EVERY: Duration = Duration::from_secs(5);

const METADATA: Metadata<'static> = Metadata::new(module_path!(), Level::INFO, Some(module_path!()));

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
    // A span as well as a count: where a trace is collected, this is the store's part of a request.
    let span = tracing::debug_span!("store", operation = operation.name());
    let result = work.instrument(span).await;
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
  use std::sync::atomic::{AtomicU64, Ordering};
  use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

  use super::*;

  fn metrics() -> Arc<Metrics> {
    Arc::new(Metrics::exported("f46e-test").unwrap())
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
      let in_flight = format!("record_similarity_requests_in_flight{{call=\"{call}\"}}");
      assert_eq!(value(&page, &in_flight).as_deref(), Some("0"), "{call}");
    }
    for latency in [
      REQUEST_SECONDS,
      STORE_SECONDS,
      LOCK_WAIT,
      EMBED_WAIT,
      PASS_SECONDS,
      PASS_TEXTS,
    ] {
      assert_eq!(lines_of(&page, latency), [""; 0], "{latency}");
    }
    assert_eq!(lines_of(&page, EVALUATIONS).len(), 12);
    assert_eq!(lines_of(&page, MATCHES).len(), 3);
    assert_eq!(lines_of(&page, IMPORT_RECORDS).len(), 3);
    assert_eq!(lines_of(&page, STORE_ERRORS).len(), 8);
    // Not a thread, so nothing about one.
    assert_eq!(lines_of(&page, "record_similarity_embed_"), [""; 0]);
    assert_eq!(value(&page, WRITES_IN_FLIGHT).as_deref(), Some("0"));
    assert_eq!(value(&page, WRITES_UNANSWERED).as_deref(), Some("0"));
  }

  /// Every metric is introduced by a HELP and a TYPE line, once, before its samples, and every sample belongs to
  /// the metric introduced above it: what a parser of the format requires. And every name says what kind of
  /// thing it is, as the convention has it.
  #[test]
  fn every_sample_follows_the_help_and_type_of_its_metric() {
    let metrics = metrics();
    metrics.begin(Call::Evaluate).finished(Code::Ok);
    metrics.stored(Operation::Get, Duration::from_millis(3), false);
    metrics.lock_waited(Duration::from_millis(1));
    metrics.embed_waited(Duration::from_millis(1));
    metrics.passed(4, Duration::from_millis(2));
    let polled = Polled {
      worker: Some(WorkerSnapshot::default()),
      ..Polled::default()
    };
    let page = metrics.render(&polled);
    let mut introduced: Vec<(String, String)> = Vec::new();
    let mut helped = None;
    for line in page.lines().filter(|line| !line.is_empty()) {
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
        assert!(name.starts_with("record_similarity_"), "{name}");
        assert_eq!(kind == "counter", name.ends_with("_total"), "{line}");
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
        let sample = line.rsplit(' ').next().unwrap();
        assert!(sample.parse::<f64>().is_ok(), "not a number: {line}");
      }
    }
    // Everything this file names is there, and nothing it does not name.
    let named = COUNTERS.len() + 1 + GAUGES.len() + HISTOGRAMS.len();
    assert_eq!(introduced.len(), named, "{introduced:?}");
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
    let took = |call: &str| {
      value(
        &page,
        &format!("record_similarity_request_duration_seconds_count{{call=\"{call}\"}}"),
      )
    };
    assert_eq!(took("Evaluate").as_deref(), Some("2"));
    assert_eq!(took("Match").as_deref(), Some("1"));
    assert_eq!(took("Delete"), None);
  }

  /// The transport drops a handler whose caller has gone. That request ended too, and is counted as what the
  /// transport calls it.
  #[test]
  fn a_request_dropped_without_an_answer_is_counted_as_cancelled() {
    let metrics = metrics();
    drop(metrics.begin(Call::FindSimilar));
    let page = metrics.render(&Polled::default());
    let cancelled = "record_similarity_requests_total{call=\"FindSimilar\",code=\"CANCELLED\"}";
    assert_eq!(value(&page, cancelled).as_deref(), Some("1"));
    let in_flight = "record_similarity_requests_in_flight{call=\"FindSimilar\"}";
    assert_eq!(value(&page, in_flight).as_deref(), Some("0"));
    let took = "record_similarity_request_duration_seconds_count{call=\"FindSimilar\"}";
    assert_eq!(value(&page, took).as_deref(), Some("1"));
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
  /// buckets end at the count. The bounds are this file's, in seconds, and texts have bounds of their own.
  #[test]
  fn latencies_are_counted_into_the_buckets_this_file_names() {
    let metrics = metrics();
    for elapsed in [
      Duration::from_micros(999),
      Duration::from_millis(1),
      Duration::from_nanos(1_000_001),
      Duration::from_millis(250),
      Duration::from_mins(1),
    ] {
      metrics.stored(Operation::Get, elapsed, false);
    }
    metrics.passed(1, Duration::from_millis(2));
    metrics.passed(32, Duration::from_millis(12));
    metrics.passed(33, Duration::from_millis(12));
    let page = metrics.render(&Polled::default());
    let bucket = |le: &str| {
      let series = format!("record_similarity_store_duration_seconds_bucket{{operation=\"get\",le=\"{le}\"}}");
      value(&page, &series).unwrap()
    };
    assert_eq!(bucket("0.001"), "2");
    assert_eq!(bucket("0.0025"), "3");
    assert_eq!(bucket("0.1"), "3");
    assert_eq!(bucket("0.25"), "4");
    assert_eq!(bucket("2.5"), "4");
    assert_eq!(bucket("+Inf"), "5", "longer than every bound is still counted");
    let get = lines_of(&page, "record_similarity_store_duration_seconds");
    assert_eq!(get.len(), SECONDS.len() + 3, "{get:?}");
    let sum: f64 = value(&page, "record_similarity_store_duration_seconds_sum{operation=\"get\"}")
      .unwrap()
      .parse()
      .unwrap();
    assert!((sum - 60.252_999_001).abs() < 1e-6, "{sum}");

    let texts = |le: &str| {
      value(
        &page,
        &format!("record_similarity_embed_pass_texts_bucket{{le=\"{le}\"}}"),
      )
      .unwrap()
    };
    assert_eq!(
      (texts("1"), texts("16"), texts("32"), texts("64")),
      ("1".into(), "1".into(), "2".into(), "3".into())
    );
    assert_eq!(
      value(&page, "record_similarity_embed_pass_texts_sum").as_deref(),
      Some("66")
    );
    assert_eq!(
      value(&page, "record_similarity_embed_pass_duration_seconds_count").as_deref(),
      Some("3")
    );
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

  /// What the embedding thread and the engine count themselves is written as it is read, each time the page is
  /// asked for: these are totals kept elsewhere, not events counted here, so asking twice does not double them.
  #[test]
  fn what_the_thread_and_the_engine_count_is_written_as_it_is_read() {
    let metrics = metrics();
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
    let first = metrics.render(&polled);
    let page = metrics.render(&polled);
    assert_eq!(first.lines().count(), page.lines().count());
    let read = |series: &str| value(&page, series).unwrap();
    assert_eq!(read("record_similarity_embed_requests_total"), "9");
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

  /// The waits a request can have besides the store are each counted where they happen: for its record's lock,
  /// and for the model.
  #[test]
  fn the_waits_for_a_lock_and_for_the_model_are_counted() {
    let metrics = metrics();
    metrics.lock_waited(Duration::from_millis(3));
    metrics.lock_waited(Duration::ZERO);
    metrics.embed_waited(Duration::from_millis(40));
    let page = metrics.render(&Polled::default());
    assert_eq!(
      value(&page, "record_similarity_lock_wait_seconds_count").as_deref(),
      Some("2")
    );
    assert_eq!(
      value(&page, "record_similarity_lock_wait_seconds_sum").as_deref(),
      Some("0.003")
    );
    assert_eq!(
      value(&page, "record_similarity_embed_wait_seconds_count").as_deref(),
      Some("1")
    );
    assert_eq!(
      value(&page, "record_similarity_embed_wait_seconds_bucket{le=\"0.025\"}").as_deref(),
      Some("0")
    );
    assert_eq!(
      value(&page, "record_similarity_embed_wait_seconds_bucket{le=\"0.05\"}").as_deref(),
      Some("1")
    );
  }

  /// Counts that are kept nowhere cost nothing to make and say nothing when asked. And two servers that each
  /// keep their own do not count into each other.
  #[test]
  fn discarded_counts_say_nothing_and_two_sets_of_counts_stay_apart() {
    let nowhere = Arc::new(Metrics::discarded());
    nowhere.begin(Call::Evaluate).finished(Code::Ok);
    nowhere.stored(Operation::Get, Duration::from_millis(1), true);
    nowhere.evaluated(Verdict::Created, WriteMode::Enforce);
    nowhere.passed(3, Duration::from_millis(1));
    nowhere.upkeep();
    assert_eq!(nowhere.render(&Polled::default()), "");

    let (one, other) = (metrics(), metrics());
    one.begin(Call::Evaluate).finished(Code::Ok);
    one.upkeep();
    let series = "record_similarity_requests_total{call=\"Evaluate\",code=\"OK\"}";
    assert_eq!(value(&one.render(&Polled::default()), series).as_deref(), Some("1"));
    assert_eq!(value(&other.render(&Polled::default()), series).as_deref(), Some("0"));
  }

  /// A fingerprint is the one label value that is not a word fixed in this file. Whatever it holds, the line it
  /// is on is still one line with one value.
  #[test]
  fn a_label_value_cannot_break_out_of_its_line() {
    let page = Metrics::exported("odd\"one\nand\\more")
      .unwrap()
      .render(&Polled::default());
    let lines = lines_of(&page, MODEL_INFO);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].ends_with("} 1"), "{lines:?}");
    assert!(lines[0].contains("odd\\\"one"), "the quote is escaped: {lines:?}");
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
      let count = format!("record_similarity_store_duration_seconds_count{{operation=\"{operation}\"}}");
      assert_eq!(value(&page, &count), Some(times.to_string()), "{operation}");
      let errors = format!("record_similarity_store_errors_total{{operation=\"{operation}\"}}");
      assert_eq!(
        value(&page, &errors),
        Some(u8::from(operation == "put").to_string()),
        "{operation}"
      );
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
