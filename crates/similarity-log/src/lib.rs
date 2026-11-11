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
//! Every line has `timestamp`, `level`, `message`, and `target`, with the event's own fields beside them at the
//! top level. Field values are logged as numbers and booleans where they are numbers and booleans, never as
//! preformatted text, so a consumer can chart or filter on them without parsing strings.
//!
//! Command-line usage errors are the one exception: they are written by the argument parser, as text, before
//! anything has started, for whoever typed the command.

use std::fmt::Display;

use tracing::{Metadata, Subscriber};
use tracing_subscriber::filter::{EnvFilter, FilterExt, filter_fn};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::{Layer, SubscriberExt};
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

/// Builds the subscriber: `logs` receives filtered logs and every failure, `results` receives every result.
///
/// Separate from [`init`] so that tests can capture both streams.
pub fn subscriber<L, R>(logs: L, results: R, filter: EnvFilter) -> impl Subscriber + Send + Sync + 'static
where
  L: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
  R: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
  // Fields at the top level of each object, which is what log pipelines and `jq` expect, and no span context:
  // these programs have no spans.
  let log_lines = tracing_subscriber::fmt::layer()
    .json()
    .flatten_event(true)
    .with_current_span(false)
    .with_span_list(false)
    .with_writer(logs)
    .with_filter(filter.and(filter_fn(is_result).not()).or(filter_fn(is_failure)));
  let result_lines = tracing_subscriber::fmt::layer()
    .json()
    .flatten_event(true)
    .with_current_span(false)
    .with_span_list(false)
    .with_writer(results)
    .with_filter(filter_fn(is_result));
  tracing_subscriber::registry().with(log_lines).with(result_lines)
}

/// Starts logging for the process: logs and failures to stderr, results to stdout, panics reported as failures.
///
/// The log level comes from `RUST_LOG` and defaults to `info`. Call once, first thing in `main`.
pub fn init() {
  let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
  subscriber(std::io::stderr, std::io::stdout, filter).init();

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
  use std::sync::{Arc, Mutex};

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
  }

  /// Runs `emit` with logging set up as [`super::init`] sets it up, at the `RUST_LOG` value `level`.
  pub fn capture(level: &str, emit: impl FnOnce()) -> Captured {
    let (logs, results) = (Capture::default(), Capture::default());
    let subscriber = super::subscriber(logs.clone(), results.clone(), EnvFilter::new(level));
    tracing::subscriber::with_default(subscriber, emit);
    Captured {
      logs: logs.lines(),
      results: results.lines(),
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
}
