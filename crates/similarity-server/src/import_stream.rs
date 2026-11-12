//! Creating many records over one connection: what is behind the streaming `Import` call.
//!
//! [`Engine::import`] already creates a batch of records (D20). This is the loop that feeds it from a stream:
//! take what has arrived, import it, answer each record in the order it came, and take the next. It knows nothing
//! of protobuf; `grpc.rs` hands it records and turns its answers into messages.
//!
//! **A stream is batched by its own backlog, in small batches.** The first record waiting is taken with however
//! many more are already there, up to the engine's `import_batch` (`server.import_batch_records`, 8 by default).
//! A caller that sends one record at a time gets each answered as it arrives; one that sends as fast as it can
//! gets full batches. The batches are small on purpose: the server has one embedding thread (D9), and a batch is
//! one job on it, during which a merchant's save has to wait. Eight records of four fields are about one forward
//! pass, and that is the trade the setting makes: larger batches load a catalog faster and make every other
//! request wait longer.
//!
//! **A busy model slows a stream down; it does not fail it.** The queue in front of the model is bounded, and a
//! single request that finds it full is refused at once so that it can back off. A stream is its own back-off:
//! it holds one batch, waits, and offers it again, and meanwhile reads nothing more, which the transport turns
//! into back-pressure on the sender. Only after [`PATIENCE`] is a batch given up, each of its records answered
//! as overloaded, and the stream goes on.
//!
//! **Cutting a stream off is safe.** Every record already imported stays imported; the rest were never taken.
//! Sending the whole stream again creates what is missing and leaves the rest exactly alone.

use std::time::Duration;

use similarity_store::VectorStore;
use tokio::sync::mpsc;

use crate::engine::{Engine, EngineError, ImportInput, Imported};
use crate::worker::TextEmbedder;

/// How long a batch waits for room in front of the model before its records are answered as overloaded.
pub const PATIENCE: Duration = Duration::from_secs(10);
/// The first wait after finding the model's queue full. Each later one is twice as long, up to [`LONGEST_WAIT`].
const FIRST_WAIT: Duration = Duration::from_millis(10);
/// The longest single wait.
const LONGEST_WAIT: Duration = Duration::from_millis(250);

/// What became of one record of a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
  /// The record's place in the stream, from 0.
  pub index: u64,
  /// The id it was sent with, if it had one.
  pub id: Option<String>,
  /// Created, already stored, or why not.
  pub outcome: Result<Imported, EngineError>,
}

/// What a stream came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Totals {
  /// Records taken from the stream.
  pub records: u64,
  /// Records created.
  pub created: u64,
  /// Records that were already stored, and left alone.
  pub existing: u64,
  /// Records that were not stored.
  pub failed: u64,
  /// Batches imported.
  pub batches: u64,
  /// Times a batch found the model's queue full and waited.
  pub waits: u64,
}

/// Imports a batch, waiting for room in front of the model for up to `patience` if there is none. A batch that
/// cannot be imported at all is answered record by record with why.
async fn import_patiently<S: VectorStore, E: TextEmbedder>(
  engine: &Engine<S, E>,
  batch: Vec<ImportInput>,
  patience: Duration,
  totals: &mut Totals,
) -> Vec<Result<Imported, EngineError>> {
  let mut wait = FIRST_WAIT;
  let mut waited = Duration::ZERO;
  loop {
    match engine.import(batch.clone()).await {
      Ok(outcomes) => return outcomes,
      Err(EngineError::Overloaded(_)) if waited < patience => {
        totals.waits = totals.waits.saturating_add(1);
        tokio::time::sleep(wait).await;
        waited = waited.saturating_add(wait);
        wait = wait.saturating_mul(2).min(LONGEST_WAIT);
      }
      Err(error) => return batch.iter().map(|_| Err(error.clone())).collect(),
    }
  }
}

/// Imports everything `incoming` yields, answering each record on `answers` in the order it came, until the
/// stream ends or nobody is listening for answers any more.
///
/// `patience` is how long a batch waits for room in front of the model ([`PATIENCE`] in service).
pub async fn run<S: VectorStore, E: TextEmbedder>(
  engine: &Engine<S, E>,
  mut incoming: mpsc::Receiver<ImportInput>,
  answers: mpsc::Sender<Answer>,
  patience: Duration,
) -> Totals {
  let mut totals = Totals::default();
  while let Some(first) = incoming.recv().await {
    let mut batch = vec![first];
    while batch.len() < engine.import_batch() {
      // Whatever has already arrived, and no waiting for more.
      let Ok(next) = incoming.try_recv() else { break };
      batch.push(next);
    }
    let ids: Vec<Option<String>> = batch.iter().map(|input| input.record.id.clone()).collect();
    let outcomes = import_patiently(engine, batch, patience, &mut totals).await;
    totals.batches = totals.batches.saturating_add(1);
    for (id, outcome) in ids.into_iter().zip(outcomes) {
      let index = totals.records;
      totals.records = totals.records.saturating_add(1);
      let count = match &outcome {
        Ok(Imported::Created) => &mut totals.created,
        Ok(Imported::Skipped) => &mut totals.existing,
        Err(_) => &mut totals.failed,
      };
      *count = count.saturating_add(1);
      if answers.send(Answer { index, id, outcome }).await.is_err() {
        // The caller has gone. What was imported stays; the rest of the stream is not taken.
        return totals;
      }
    }
  }
  totals
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::{AtomicUsize, Ordering};

  use similarity_core::{FieldSpec, Policy, Schema};
  use similarity_store::{MemoryStore, StoredRecord};

  use super::*;
  use crate::engine::{DEFAULT_IMPORT_BATCH, EvaluateInput, Original, RecordInput, store_layout};
  use crate::worker::EmbedError;
  use crate::worker::testing::WordBagEmbedder;

  /// A model whose queue is full for the first `refusals` calls, and which counts the calls that got through.
  struct Busy {
    inner: WordBagEmbedder,
    refusals: AtomicUsize,
    passes: AtomicUsize,
  }

  impl TextEmbedder for Busy {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
      let refuse = self
        .refusals
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| left.checked_sub(1));
      if refuse.is_ok() {
        return Err(EmbedError::Overloaded);
      }
      self.passes.fetch_add(1, Ordering::Relaxed);
      self.inner.embed(texts).await
    }

    fn dimension(&self) -> usize {
      self.inner.dimension()
    }

    fn fingerprint(&self) -> &str {
      self.inner.fingerprint()
    }
  }

  type TestEngine = Engine<MemoryStore, Busy>;

  fn engine(refusals: usize) -> TestEngine {
    let schema = Schema::new(vec![FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: true,
      reject_below: None,
    }])
    .unwrap();
    let embedder = Busy {
      inner: WordBagEmbedder::new("word-bag-v1"),
      refusals: AtomicUsize::new(refusals),
      passes: AtomicUsize::new(0),
    };
    let (layout, dimension) = store_layout(&schema, &embedder);
    Engine::new(
      MemoryStore::new(layout, dimension),
      embedder,
      schema,
      Policy::new(0.75, None).unwrap(),
    )
  }

  fn input(id: Option<&str>, name: &str) -> ImportInput {
    ImportInput {
      record: RecordInput {
        id: id.map(str::to_owned),
        tenant: None,
        fields: vec![("name".to_owned(), name.to_owned())],
        quantities: Vec::new(),
      },
      anchor: None,
    }
  }

  /// Sends every input before anything is imported, then imports: the stream arrives faster than it is read.
  async fn all_at_once(engine: &TestEngine, inputs: Vec<ImportInput>, patience: Duration) -> (Vec<Answer>, Totals) {
    let (sender, incoming) = mpsc::channel(inputs.len().max(1));
    let (answers, mut answered) = mpsc::channel(inputs.len().max(1));
    for input in inputs {
      sender.send(input).await.unwrap();
    }
    drop(sender);
    let totals = run(engine, incoming, answers, patience).await;
    let mut collected = Vec::new();
    while let Some(answer) = answered.recv().await {
      collected.push(answer);
    }
    (collected, totals)
  }

  async fn stored(engine: &TestEngine, id: &str) -> Option<StoredRecord> {
    engine.store().get(id).await.unwrap()
  }

  /// Every record gets one answer, in the order it was sent. A record that is already stored is left exactly as
  /// it is, and one that cannot be stored fails by itself.
  #[tokio::test]
  async fn every_record_is_answered_in_order_and_a_failure_or_a_stored_record_stops_nothing() {
    let engine = engine(0);
    engine
      .evaluate(EvaluateInput {
        record: input(Some("old"), "blueberry gummies").record,
        ..EvaluateInput::default()
      })
      .await
      .unwrap();
    let before = stored(&engine, "old").await;

    let inputs = vec![
      input(Some("a"), "sour diesel vape"),
      input(Some("old"), "something else entirely"),
      input(None, "no id"),
      input(Some("b"), "  "),
      ImportInput {
        anchor: Some(Original {
          fields: vec![("name".to_owned(), "indoor flower".to_owned())],
          quantities: Vec::new(),
        }),
        ..input(Some("c"), "indoor flower eighth")
      },
      input(Some("a"), "sent twice in one stream"),
    ];
    let (answers, totals) = all_at_once(&engine, inputs, PATIENCE).await;

    let seen: Vec<(u64, Option<&str>, Option<Imported>)> = answers
      .iter()
      .map(|answer| (answer.index, answer.id.as_deref(), answer.outcome.clone().ok()))
      .collect();
    assert_eq!(
      seen,
      [
        (0, Some("a"), Some(Imported::Created)),
        (1, Some("old"), Some(Imported::Skipped)),
        (2, None, None),
        (3, Some("b"), None),
        (4, Some("c"), Some(Imported::Created)),
        (5, Some("a"), Some(Imported::Skipped)),
      ]
    );
    for failed in [&answers[2], &answers[3]] {
      assert!(
        matches!(failed.outcome, Err(EngineError::InvalidArgument(_))),
        "{failed:?}"
      );
    }
    assert_eq!(
      totals,
      Totals {
        records: 6,
        created: 2,
        existing: 2,
        failed: 2,
        batches: 1,
        waits: 0,
      }
    );
    assert_eq!(
      stored(&engine, "old").await,
      before,
      "what was stored is exactly what it was"
    );
    assert_eq!(stored(&engine, "b").await, None);
    // The first of two records with one id is the one that is stored; the original version is the anchor.
    let a = stored(&engine, "a").await.unwrap();
    assert_eq!(a.revision, 1);
    let c = stored(&engine, "c").await.unwrap();
    assert_ne!(c.anchor, c.vectors.composite, "drift is measured from the original");
  }

  /// A stream is batched by what has already arrived, never by waiting: records that are all there go in batches
  /// of eight, and records that come one by one are imported one by one.
  #[tokio::test]
  async fn a_stream_is_batched_by_its_own_backlog() {
    let engine = engine(0);
    let inputs: Vec<ImportInput> = (0..19)
      .map(|index| input(Some(&format!("r{index}")), "indoor flower"))
      .collect();
    let (answers, totals) = all_at_once(&engine, inputs, PATIENCE).await;
    assert_eq!(
      (answers.len(), totals.created, totals.batches),
      (19, 19, 3),
      "eight, eight, and three"
    );
    assert_eq!(
      engine.embedder().passes.load(Ordering::Relaxed),
      3,
      "one job on the model for each batch"
    );
    assert_eq!(
      answers.iter().map(|answer| answer.index).collect::<Vec<_>>(),
      (0..19).collect::<Vec<_>>()
    );

    // One at a time, each answered before the next is sent.
    let engine = super::tests::engine(0);
    let (sender, incoming) = mpsc::channel(DEFAULT_IMPORT_BATCH);
    let (answers, mut answered) = mpsc::channel(DEFAULT_IMPORT_BATCH);
    let (totals, ()) = tokio::join!(run(&engine, incoming, answers, PATIENCE), async {
      for index in 0..5_u64 {
        sender
          .send(input(Some(&format!("r{index}")), "indoor flower"))
          .await
          .unwrap();
        assert_eq!(answered.recv().await.unwrap().index, index);
      }
      drop(sender);
    });
    assert_eq!((totals.records, totals.batches), (5, 5));

    // The size of a batch is the engine's setting, and never less than one record.
    for (setting, batches) in [(2, 10), (1, 19), (0, 19), (64, 1)] {
      let engine = super::tests::engine(0).with_import_batch(setting);
      let inputs: Vec<ImportInput> = (0..19)
        .map(|index| input(Some(&format!("r{index}")), "indoor flower"))
        .collect();
      let (_, totals) = all_at_once(&engine, inputs, PATIENCE).await;
      assert_eq!(
        (totals.created, totals.batches),
        (19, batches),
        "import_batch = {setting}"
      );
    }
  }

  /// A full queue in front of the model is waited out. The stream is its own back-off.
  #[tokio::test]
  async fn a_busy_model_slows_a_stream_down_and_does_not_fail_it() {
    let engine = engine(3);
    let inputs = vec![input(Some("a"), "indoor flower"), input(Some("b"), "sour diesel vape")];
    let (answers, totals) = all_at_once(&engine, inputs, PATIENCE).await;
    assert!(
      answers.iter().all(|answer| answer.outcome == Ok(Imported::Created)),
      "{answers:?}"
    );
    assert_eq!(
      (totals.created, totals.failed, totals.waits, totals.batches),
      (2, 0, 3, 1)
    );
    assert!(stored(&engine, "a").await.is_some() && stored(&engine, "b").await.is_some());
  }

  /// Patience runs out. The batch that could not get in is answered record by record, nothing of it is stored,
  /// and the stream goes on: the next batch gets its own chance.
  #[tokio::test]
  async fn a_batch_that_never_finds_room_is_answered_as_overloaded_and_the_stream_goes_on() {
    // Enough refusals for the whole of the first batch's patience, and none left for the second.
    let engine = engine(4);
    let inputs: Vec<ImportInput> = (0..=DEFAULT_IMPORT_BATCH)
      .map(|index| input(Some(&format!("r{index}")), "indoor flower"))
      .collect();
    let (answers, totals) = all_at_once(&engine, inputs, Duration::from_millis(60)).await;
    assert_eq!(answers.len(), DEFAULT_IMPORT_BATCH + 1);
    for answer in &answers[..DEFAULT_IMPORT_BATCH] {
      assert!(matches!(answer.outcome, Err(EngineError::Overloaded(_))), "{answer:?}");
    }
    assert_eq!(answers[DEFAULT_IMPORT_BATCH].outcome, Ok(Imported::Created));
    assert_eq!((totals.failed, totals.created, totals.batches), (8, 1, 2));
    assert_eq!(
      totals.waits, 3,
      "10 ms, 20 ms, and 40 ms, which is past the 60 ms it was given"
    );
    assert_eq!(stored(&engine, "r0").await, None);
    assert!(stored(&engine, "r8").await.is_some());
  }

  /// Nobody is listening for answers any more: the stream stops being read, and what was imported stays.
  #[tokio::test]
  async fn a_caller_that_has_gone_ends_the_stream_and_keeps_what_was_imported() {
    let engine = engine(0);
    let (sender, incoming) = mpsc::channel(32);
    let (answers, mut answered) = mpsc::channel(1);
    for index in 0..20 {
      sender
        .send(input(Some(&format!("r{index}")), "indoor flower"))
        .await
        .unwrap();
    }
    drop(sender);
    let (totals, ()) = tokio::join!(run(&engine, incoming, answers, PATIENCE), async {
      assert_eq!(answered.recv().await.unwrap().index, 0);
      drop(answered);
    });
    assert!(totals.records < 20, "the rest of the stream was not taken: {totals:?}");
    assert_eq!(totals.batches, 1);
    // The whole first batch was imported before its answers could be given, and it stays.
    for index in 0..DEFAULT_IMPORT_BATCH {
      assert!(stored(&engine, &format!("r{index}")).await.is_some());
    }
    assert_eq!(stored(&engine, "r8").await, None);

    // Sending the stream again from the start creates the rest and leaves the first eight alone.
    let again: Vec<ImportInput> = (0..20)
      .map(|index| input(Some(&format!("r{index}")), "indoor flower"))
      .collect();
    let (_, totals) = all_at_once(&engine, again, PATIENCE).await;
    assert_eq!((totals.existing, totals.created, totals.failed), (8, 12, 0));
  }
}
