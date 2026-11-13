//! Logging for the `record-similarity` binaries: one JSON object per line (NDJSON), and nothing else.
//!
//! Both binaries are run by schedulers and read by log pipelines far more often than by a person at a terminal,
//! so there is one format and no switch. A person reads it with `jq`.
//!
//! Three kinds of line, told apart by the event's target:
//!
//! - **Logs** go to stderr and are filtered by `RUST_LOG` (default `info`). Progress, warnings, what was loaded.
//! - **Results** (target [`RESULT`]) go to stdout, always. What a run produced is its output, not commentary on
//!   it, so no log level can hide it and it can be captured on its own with `> results.ndjson`.
//! - **Failures** (target [`FAILURE`]) go to stderr, always, panics included. A run that fails says so in the
//!   same stream and the same format as everything else, whatever `RUST_LOG` is set to.
//!
//! A process that is recording a run (see [`Init`]) also keeps one file with all three kinds in the order they
//! were emitted: every result, every failure, and logs at `info` whatever `RUST_LOG` says, since a run's loss curve
//! is made of logs and must not depend on how quiet someone wanted stderr. It is a third layer on the same
//! subscriber, built by the same function as the other two, so its lines are the lines the streams carry.
//!
//! Every line has `timestamp`, `level`, `message`, and `target`, with the event's own fields beside them at the
//! top level. Field values are logged as numbers and booleans where they are numbers and booleans, never as
//! preformatted text, so a consumer can chart or filter on them without parsing strings.
//!
//! Command-line usage errors are the one exception: they are written by the argument parser, as text, before
//! anything has started, for whoever typed the command.

use std::fmt::Display;
use std::fs::File;
use std::sync::Mutex;

use tracing::{Metadata, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::filter::{EnvFilter, FilterExt, filter_fn};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::fmt::format::{Format, Json, JsonFields};
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

/// The target of an event that is part of a run's output. Always written, to stdout.
pub const RESULT: &str = "result";

/// The target of an event that reports a failure. Always written, to stderr.
pub const FAILURE: &str = "failure";

fn is_result(metadata: &Metadata<'_>) -> bool {
  metadata.target() == RESULT
}

fn is_failure(metadata: &Metadata<'_>) -> bool {
  metadata.target() == FAILURE
}

/// The one shape of line there is, written to `writer`: fields at the top level of each object, which is what log
/// pipelines and `jq` expect, and no span context, since these programs have no spans.
///
/// Every layer is made here, so no stream can come to differ from another in format.
fn json_layer<S, W>(writer: W) -> tracing_subscriber::fmt::Layer<S, JsonFields, Format<Json>, W>
where
  S: Subscriber + for<'span> LookupSpan<'span>,
  W: for<'writer> MakeWriter<'writer> + 'static,
{
  tracing_subscriber::fmt::layer()
    .json()
    .flatten_event(true)
    .with_current_span(false)
    .with_span_list(false)
    .with_writer(writer)
}

/// Builds the subscriber: `logs` receives filtered logs and every failure, `results` receives every result.
///
/// Separate from [`init`] so that tests can capture both streams.
pub fn subscriber<L, R>(logs: L, results: R, filter: EnvFilter) -> impl Subscriber + Send + Sync + 'static
where
  L: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
  R: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
  subscriber_with(logs, Some(results), None::<fn() -> std::io::Sink>, filter)
}

/// A layer that is not a line of output: what sends a server's spans to a tracing backend. It is made by
/// whoever knows where the spans go, and added here, because there is one subscriber and this crate builds it.
/// It brings its own filter, so what it wants to see does not change what is logged.
pub type Tracing = Box<dyn Layer<Registry> + Send + Sync + 'static>;

/// Builds the subscriber of a process that may be recording a run.
///
/// `logs` receives filtered logs and every failure. `results`, if given, receives every result. `record`, if
/// given, receives every result, every failure, and logs at `info` and above whatever `filter` says, in the order
/// they are emitted.
pub fn subscriber_with<L, R, K>(
  logs: L,
  results: Option<R>,
  record: Option<K>,
  filter: EnvFilter,
) -> impl Subscriber + Send + Sync + 'static
where
  L: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
  R: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
  K: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
  subscriber_traced(logs, results, record, filter, None)
}

/// As [`subscriber_with`], and with `tracing`, if there is one, beside the lines.
pub fn subscriber_traced<L, R, K>(
  logs: L,
  results: Option<R>,
  record: Option<K>,
  filter: EnvFilter,
  tracing: Option<Tracing>,
) -> impl Subscriber + Send + Sync + 'static
where
  L: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
  R: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
  K: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
  let log_lines = json_layer(logs).with_filter(filter.and(filter_fn(is_result).not()).or(filter_fn(is_failure)));
  let result_lines = results.map(|results| json_layer(results).with_filter(filter_fn(is_result)));
  let record_lines = record.map(|record| {
    json_layer(record).with_filter(
      filter_fn(is_result)
        .or(filter_fn(is_failure))
        .or(EnvFilter::new("info")),
    )
  });
  tracing_subscriber::registry()
    .with(tracing)
    .with(log_lines)
    .with(result_lines)
    .with(record_lines)
}

/// How a process logs, beyond the two streams every process has.
pub struct Init {
  /// The file that keeps everything the run emits, if the process is recording a run. Each event is one write of
  /// one line, unbuffered, so a run that is killed keeps everything up to its last line.
  pub record: Option<File>,
  /// Whether results go to stdout. False only where stdout carries a report for a person and the same results
  /// are in the record: results on stdout are otherwise never switched off.
  pub results_on_stdout: bool,
  /// What sends the process's spans to a tracing backend, if it has one. `None` for every process but a server
  /// that was configured with a collector.
  pub tracing: Option<Tracing>,
}

impl std::fmt::Debug for Init {
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    formatter
      .debug_struct("Init")
      .field("record", &self.record)
      .field("results_on_stdout", &self.results_on_stdout)
      .field("tracing", &self.tracing.is_some())
      .finish()
  }
}

/// Starts logging for the process: logs and failures to stderr, results to stdout, panics reported as failures.
///
/// The log level comes from `RUST_LOG` and defaults to `info`. Call once, first thing in `main`.
pub fn init() {
  init_with(Init {
    record: None,
    results_on_stdout: true,
    tracing: None,
  });
}

/// Starts logging as [`init`] does, with a run's record and the choice about stdout that [`Init`] describes.
pub fn init_with(init: Init) {
  let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
  let results = init
    .results_on_stdout
    .then_some(std::io::stdout as fn() -> std::io::Stdout);
  subscriber_traced(
    std::io::stderr,
    results,
    init.record.map(Mutex::new),
    filter,
    init.tracing,
  )
  .init();

  // Rust's default panic message is several lines of plain text. In a stream of JSON lines that is the one entry
  // a reader most needs to parse, so it is reported as an event like any other failure.
  std::panic::set_hook(Box::new(|panic| {
    let payload = panic.payload();
    let reason = payload
      .downcast_ref::<&str>()
      .copied()
      .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
      .unwrap_or("panic with a non-text payload");
    let location = panic
      .location()
      .map(|location| format!("{}:{}", location.file(), location.line()));
    tracing::error!(target: FAILURE, error = reason, location, "panicked");
  }));
}

/// Reports that the run failed, with the reason. Never filtered out.
pub fn failed(error: &dyn Display) {
  tracing::error!(target: FAILURE, error = %error, "failed");
}

/// Captures what would be written, for tests here and, behind the `testing` feature, in the binaries.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
  // A poisoned lock or bytes that are not UTF-8 should stop a test on the spot.
  #![allow(
    clippy::unwrap_used,
    clippy::missing_panics_doc,
    reason = "test support: panicking on a broken fixture is the intended behavior"
  )]

  use std::io::Write;
  use std::sync::{Arc, Mutex, OnceLock};

  use tracing_subscriber::filter::EnvFilter;
  use tracing_subscriber::fmt::MakeWriter;

  /// A writer that keeps what is written, standing in for stderr or stdout.
  #[derive(Debug, Clone, Default)]
  struct Capture(Arc<Mutex<Vec<u8>>>);

  impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
      let mut written = self
        .0
        .lock()
        .map_err(|_| std::io::Error::other("capture lock poisoned"))?;
      written.extend_from_slice(bytes);
      Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
      Ok(())
    }
  }

  impl<'writer> MakeWriter<'writer> for Capture {
    type Writer = Self;

    fn make_writer(&'writer self) -> Self::Writer {
      self.clone()
    }
  }

  impl Capture {
    fn lines(&self) -> Vec<String> {
      let bytes = self.0.lock().unwrap().clone();
      String::from_utf8(bytes).unwrap().lines().map(str::to_owned).collect()
    }
  }

  /// The lines a run of `emit` writes, by stream.
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct Captured {
    /// What would have gone to stderr: logs that pass the level, and every failure.
    pub logs: Vec<String>,
    /// What would have gone to stdout: every result.
    pub results: Vec<String>,
    /// What a run's record would hold. Empty when nothing is being recorded.
    pub record: Vec<String>,
  }

  /// Keeps a capture from losing events to a test running beside it on another thread.
  ///
  /// `tracing` remembers, for each place an event is written, whether anybody wants it. While exactly one
  /// subscriber exists it takes a shortcut and asks whichever subscriber the *current* thread has, on the
  /// assumption that the one subscriber is the process's. A capture's subscriber belongs to one thread. So when a
  /// capture was the only one alive, and a test on another thread was the first to reach some event, the answer
  /// remembered was that other thread's (nobody), and the capturing thread's later event was dropped: a
  /// `finished` line missing from a run that wrote one. It happened whenever the tests ran on three threads, which
  /// is what a three-core CI runner gives them. A second subscriber that lives as long as the process ends the
  /// shortcut, and every subscriber that exists is asked. A process that installs one global subscriber, which is
  /// every real one (D15), is what the shortcut was made for and is not affected.
  fn never_the_only_subscriber() {
    static SECOND: OnceLock<tracing::Dispatch> = OnceLock::new();
    SECOND.get_or_init(|| tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default()));
  }

  /// Runs `emit` with logging set up as [`super::init`] sets it up, at the `RUST_LOG` value `level`.
  pub fn capture(level: &str, emit: impl FnOnce()) -> Captured {
    never_the_only_subscriber();
    let (logs, results) = (Capture::default(), Capture::default());
    let subscriber = super::subscriber(logs.clone(), results.clone(), EnvFilter::new(level));
    tracing::subscriber::with_default(subscriber, emit);
    Captured {
      logs: logs.lines(),
      results: results.lines(),
      record: Vec::new(),
    }
  }

  /// Runs `emit` with logging set up as [`super::init_with`] sets it up for a process that is recording a run.
  pub fn capture_recorded(level: &str, results_on_stdout: bool, emit: impl FnOnce()) -> Captured {
    never_the_only_subscriber();
    let (logs, results, record) = (Capture::default(), Capture::default(), Capture::default());
    let subscriber = super::subscriber_with(
      logs.clone(),
      results_on_stdout.then(|| results.clone()),
      Some(record.clone()),
      EnvFilter::new(level),
    );
    tracing::subscriber::with_default(subscriber, emit);
    Captured {
      logs: logs.lines(),
      results: results.lines(),
      record: record.lines(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Both streams of a run of `emit`, every line parsed: a line that is not valid JSON fails the test here.
  fn capture(level: &str, emit: impl FnOnce()) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let parse = |lines: Vec<String>| lines.iter().map(|line| serde_json::from_str(line).unwrap()).collect();
    let captured = testing::capture(level, emit);
    (parse(captured.logs), parse(captured.results))
  }

  #[test]
  fn a_log_line_is_one_json_object_with_its_fields_at_the_top_level_and_numbers_as_numbers() {
    let (logs, results) = capture("info", || {
      tracing::info!(step = 40, pair_loss = 0.0162, done = false, path = %"a/b", "training");
    });
    assert!(results.is_empty());
    let [line] = logs.as_slice() else {
      panic!("expected one line, got {logs:?}")
    };
    assert_eq!(line["message"], "training");
    assert_eq!(line["level"], "INFO");
    assert!(line["timestamp"].is_string() && line["target"].is_string());
    assert_eq!(line["step"], 40);
    assert!((line["pair_loss"].as_f64().unwrap() - 0.0162).abs() < 1e-12);
    assert_eq!(line["done"], false);
    assert_eq!(line["path"], "a/b");
    assert!(line.get("fields").is_none() && line.get("span").is_none(), "{line}");
  }

  #[test]
  fn an_absent_optional_field_is_left_out_rather_than_written_as_null() {
    let (logs, _) = capture("info", || {
      tracing::info!(pair_loss = Some(0.5), group_loss = None::<f64>, "training");
    });
    assert_eq!(logs[0]["pair_loss"], 0.5);
    assert!(logs[0].get("group_loss").is_none(), "{}", logs[0]);
  }

  #[test]
  fn the_log_level_filters_logs_and_only_logs() {
    let emit = || {
      tracing::info!("progress");
      tracing::warn!("careful");
      tracing::info!(target: RESULT, gap = 0.4, "separation");
      failed(&"disk full");
    };
    let messages = |lines: &[serde_json::Value]| -> Vec<String> {
      lines
        .iter()
        .map(|line| line["message"].as_str().unwrap().to_owned())
        .collect()
    };

    let (logs, results) = capture("info", emit);
    assert_eq!(messages(&logs), ["progress", "careful", "failed"]);
    assert_eq!(messages(&results), ["separation"]);

    let (logs, results) = capture("warn", emit);
    assert_eq!(messages(&logs), ["careful", "failed"]);
    assert_eq!(messages(&results), ["separation"]);

    // With logging switched off entirely, a run still reports what it produced and whether it failed.
    let (logs, results) = capture("off", emit);
    assert_eq!(messages(&logs), ["failed"]);
    assert_eq!(messages(&results), ["separation"]);
    assert_eq!(logs[0]["error"], "disk full");
    assert_eq!(logs[0]["level"], "ERROR");
    assert_eq!(results[0]["target"], RESULT);
  }

  fn emit_one_of_each() {
    tracing::info!(step = 1, "training");
    tracing::debug!("detail");
    tracing::info!(target: RESULT, gap = 0.4, "separation");
    tracing::warn!("careful");
    failed(&"disk full");
  }

  fn messages(lines: &[String]) -> Vec<String> {
    lines
      .iter()
      .map(|line| {
        let line: serde_json::Value = serde_json::from_str(line).unwrap();
        line["message"].as_str().unwrap().to_owned()
      })
      .collect()
  }

  #[test]
  fn a_recorded_run_gets_logs_results_and_failures_in_one_file_in_the_order_emitted() {
    let captured = testing::capture_recorded("info", true, emit_one_of_each);
    assert_eq!(
      messages(&captured.record),
      ["training", "separation", "careful", "failed"]
    );
    assert_eq!(messages(&captured.logs), ["training", "careful", "failed"]);
    assert_eq!(messages(&captured.results), ["separation"]);
  }

  /// A run's loss curve is made of `info` logs. Quietening stderr must not lose it.
  #[test]
  fn the_record_keeps_info_logs_that_rust_log_hides_from_stderr() {
    for level in ["warn", "off"] {
      let captured = testing::capture_recorded(level, true, emit_one_of_each);
      assert_eq!(
        messages(&captured.record),
        ["training", "separation", "careful", "failed"],
        "{level}"
      );
      assert!(!messages(&captured.logs).contains(&"training".to_owned()), "{level}");
    }
    // And a louder stderr does not make the record louder: it is `info`, whatever is asked of stderr.
    let captured = testing::capture_recorded("debug", true, emit_one_of_each);
    assert!(messages(&captured.logs).contains(&"detail".to_owned()));
    assert!(!messages(&captured.record).contains(&"detail".to_owned()));
  }

  #[test]
  fn results_kept_off_stdout_still_reach_the_record() {
    let captured = testing::capture_recorded("info", false, emit_one_of_each);
    assert!(captured.results.is_empty());
    assert!(messages(&captured.record).contains(&"separation".to_owned()));
  }

  #[test]
  fn a_line_in_the_record_is_the_line_the_streams_carry() {
    let captured = testing::capture_recorded("info", true, emit_one_of_each);
    let undated = |line: &String| {
      let mut object: serde_json::Map<String, serde_json::Value> = serde_json::from_str(line).unwrap();
      assert!(object.remove("timestamp").is_some());
      object
    };
    let record: Vec<_> = captured.record.iter().map(undated).collect();
    for line in captured.logs.iter().chain(&captured.results) {
      assert!(record.contains(&undated(line)), "{line}");
    }
    assert_eq!(record.len(), captured.logs.len() + captured.results.len());
  }

  #[test]
  fn without_a_record_logging_is_exactly_as_before() {
    let plain = testing::capture("info", emit_one_of_each);
    assert!(plain.record.is_empty());
    assert_eq!(messages(&plain.logs), ["training", "careful", "failed"]);
    assert_eq!(messages(&plain.results), ["separation"]);
  }
}
