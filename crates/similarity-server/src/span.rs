//! Where one request's time went (docs/decisions.md, D40).
//!
//! A request's log line says how long it took. This says what it spent the time on: waiting for its record's
//! lock, reading the store, waiting for the model, the forward pass it was part of, writing, searching. The
//! difference between those and the whole is what the server itself did (validating, scoring, translating),
//! and what the runtime kept it waiting for.
//!
//! The times are kept on the request's own task ([`traced`]) and added to by the code that waits ([`timed`],
//! [`spent`]). Like a deadline ([`crate::worker::by`]) they are a fact about the request and not an argument of
//! every call in between, and code that runs outside a traced request adds to nothing and costs nothing. They
//! are read once, when the request is answered, for its log line and for the metrics. A request that is dropped
//! takes its times with it.

use std::cell::Cell;
use std::future::Future;
use std::time::{Duration, Instant};

/// Something a request waits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
  /// Its record's lock: another request for the same record, in this process, is being served.
  Lock,
  /// The store, reading: the record's baseline, or the record a search starts from.
  StoreRead,
  /// The model: from joining the queue to being taken into a forward pass.
  Queue,
  /// The forward pass it was part of, from the pass being put together to the answer.
  Model,
  /// The store, writing, to the write's end.
  StoreWrite,
  /// The store, searching.
  Search,
}

impl Stage {
  /// Every stage, in the order a request meets them.
  pub const ALL: [Self; 6] = [
    Self::Lock,
    Self::StoreRead,
    Self::Queue,
    Self::Model,
    Self::StoreWrite,
    Self::Search,
  ];

  /// The stage's name, as a metric's label has it.
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::Lock => "lock",
      Self::StoreRead => "store_read",
      Self::Queue => "queue",
      Self::Model => "model",
      Self::StoreWrite => "store_write",
      Self::Search => "search",
    }
  }

  const fn index(self) -> usize {
    match self {
      Self::Lock => 0,
      Self::StoreRead => 1,
      Self::Queue => 2,
      Self::Model => 3,
      Self::StoreWrite => 4,
      Self::Search => 5,
    }
  }
}

/// What one request spent on each stage. A stage it met more than once (a record read again after a lost race)
/// is the sum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timings {
  nanoseconds: [u64; 6],
  /// How many texts were in the forward pass the request was part of: its own and everybody else's. Zero if it
  /// was in none; the largest, if it was in several.
  pub pass_texts: u64,
}

impl Timings {
  /// The time spent on `stage`.
  #[must_use]
  pub fn spent(&self, stage: Stage) -> Duration {
    Duration::from_nanos(self.nanoseconds.get(stage.index()).copied().unwrap_or(0))
  }

  /// The time spent on `stage` in milliseconds, to three places: what a log line carries.
  #[must_use]
  pub fn milliseconds(&self, stage: Stage) -> f64 {
    (self.spent(stage).as_secs_f64() * 1_000_000.0).round() / 1000.0
  }

  /// The time spent on every stage together.
  #[must_use]
  pub fn accounted(&self) -> Duration {
    Duration::from_nanos(
      self
        .nanoseconds
        .iter()
        .fold(0_u64, |total, stage| total.saturating_add(*stage)),
    )
  }
}

/// A request's times while it runs. Cells and not atomics: they live on the request's task, and only the code
/// that task is running touches them.
#[derive(Debug, Default)]
struct Running {
  nanoseconds: [Cell<u64>; 6],
  pass_texts: Cell<u64>,
}

tokio::task_local! {
  static RUNNING: Running;
}

/// Runs `work` as a request whose time is accounted for, and returns what it spent on each stage beside what it
/// came to.
pub async fn traced<F: Future>(work: F) -> (F::Output, Timings) {
  RUNNING
    .scope(Running::default(), async {
      let output = work.await;
      let timings = RUNNING.with(|running| Timings {
        nanoseconds: std::array::from_fn(|stage| running.nanoseconds.get(stage).map_or(0, Cell::get)),
        pass_texts: running.pass_texts.get(),
      });
      (output, timings)
    })
    .await
}

/// Adds `elapsed` to what the request on this task has spent on `stage`. Outside a traced request it does
/// nothing.
pub fn spent(stage: Stage, elapsed: Duration) {
  let elapsed = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
  let _ = RUNNING.try_with(|running| {
    if let Some(total) = running.nanoseconds.get(stage.index()) {
      total.set(total.get().saturating_add(elapsed));
    }
  });
}

/// Notes that the request on this task was part of a forward pass of `texts` texts.
pub fn shared_pass(texts: u64) {
  let _ = RUNNING.try_with(|running| running.pass_texts.set(running.pass_texts.get().max(texts)));
}

/// Runs `work` and adds the time it took to `stage`. A wait that is dropped unfinished is not counted: the
/// request it belonged to is gone.
pub async fn timed<F: Future>(stage: Stage, work: F) -> F::Output {
  let started = Instant::now();
  let output = work.await;
  spent(stage, started.elapsed());
  output
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A request's times are the sum of what was added while it ran, stage by stage, and a second request starts
  /// from nothing.
  #[tokio::test]
  async fn a_traced_request_gets_back_what_it_spent_on_each_stage() {
    let ((), timings) = traced(async {
      spent(Stage::StoreRead, Duration::from_micros(300));
      spent(Stage::StoreRead, Duration::from_micros(200));
      spent(Stage::Model, Duration::from_millis(2));
      shared_pass(7);
      shared_pass(3);
    })
    .await;
    assert_eq!(timings.spent(Stage::StoreRead), Duration::from_micros(500));
    assert_eq!(timings.spent(Stage::Model), Duration::from_millis(2));
    assert_eq!(timings.spent(Stage::Lock), Duration::ZERO);
    assert_eq!(timings.accounted(), Duration::from_micros(2500));
    assert_eq!(timings.pass_texts, 7, "the largest pass it was part of");
    assert!((timings.milliseconds(Stage::StoreRead) - 0.5).abs() < 1e-12);

    let ((), fresh) = traced(async {}).await;
    assert_eq!(fresh, Timings::default());
  }

  /// A wait is timed around the thing waited for, and what it comes to is handed on unchanged.
  #[tokio::test]
  async fn a_timed_wait_is_added_to_its_stage_and_its_result_handed_on() {
    let (answer, timings) = traced(async {
      let first = timed(Stage::Search, async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        41
      })
      .await;
      first + timed(Stage::Lock, async { 1 }).await
    })
    .await;
    assert_eq!(answer, 42);
    assert!(timings.spent(Stage::Search) >= Duration::from_millis(20), "{timings:?}");
    assert!(timings.spent(Stage::Lock) < Duration::from_millis(20), "{timings:?}");
    assert_eq!(timings.spent(Stage::StoreWrite), Duration::ZERO);
  }

  /// Code that is not serving a traced request (an import stream, a test, a command) adds to nothing, and is not
  /// slowed or failed by trying.
  #[tokio::test]
  async fn outside_a_traced_request_nothing_is_kept_and_nothing_goes_wrong() {
    spent(Stage::Model, Duration::from_secs(1));
    shared_pass(9);
    assert_eq!(timed(Stage::Queue, async { 5 }).await, 5);
    let ((), timings) = traced(async {}).await;
    assert_eq!(timings, Timings::default(), "what was added outside did not leak in");
  }

  #[test]
  fn every_stage_has_a_name_and_a_place_of_its_own() {
    let mut names: Vec<_> = Stage::ALL.iter().map(|stage| stage.name()).collect();
    let mut places: Vec<_> = Stage::ALL.iter().map(|stage| stage.index()).collect();
    assert_eq!(places, [0, 1, 2, 3, 4, 5], "in the order they are listed");
    names.sort_unstable();
    names.dedup();
    places.dedup();
    assert_eq!((names.len(), places.len()), (6, 6));
  }
}
