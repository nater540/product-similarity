//! The service's behavior, free of any transport.
//!
//! [`Engine`] is what the gRPC layer calls and what the tests exercise: it validates a record, embeds what needs
//! embedding, compares against the stored baseline, applies the policy, and writes back only what the policy allows.
//! It is generic over the store and the embedder so that every rule below is tested in memory, in microseconds,
//! without Qdrant or model weights.
//!
//! Evaluating a record is read-compare-write, and two concurrent revisions of the same id must not both be
//! compared against the same baseline and then both be written. Two things see to that. Within one process,
//! requests for the same id are serialized by a set of locks, the one piece of state the engine owns; requests
//! for different ids run concurrently. Across processes, which share no lock, every write is conditional on the
//! store still holding what was read ([`VectorStore::put_if`]): a writer that finds the record changed under it
//! has written nothing, and reads and judges again (docs/decisions.md, D10).
//!
//! A request can be dropped at any point where it waits: its caller's deadline ran out, or the caller hung up.
//! Until it goes to write, that costs nothing and leaves nothing behind. A write is different: once the engine
//! has sent one to the store, nobody can say whether a write that is dropped half way happened. So a write is
//! never dropped. It runs on a task of its own to its end ([`Engine::committed`]), whoever is told is told, and
//! if the request is gone by then, the log says what it came to (D38).

use std::fmt;
use std::hash::{BuildHasher, RandomState};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use similarity_core::matching::{self, Candidate, Decision, Ranked};
use similarity_core::quantity::first_values;
use similarity_core::score::{self, Evaluation, FieldScore};
use similarity_core::{
  FieldChange, Policy, PreparedRecord, Quantities, Quantity, QuantityChange, QuantityScore, Reason, RecordVectors,
  Schema,
};
use similarity_store::{Expected, Neighbor, SearchQuery, StoreError, StoredRecord, VectorStore};
use tokio::sync::{Mutex, OwnedMutexGuard, watch};

use crate::worker::{EmbedError, TextEmbedder};

/// The most records of an import stream taken into one batch, unless the configuration says otherwise. With four
/// text fields that is about one forward pass of the default model.
pub const DEFAULT_IMPORT_BATCH: usize = 8;

/// How many candidates `match_menu_item` returns when the caller does not say.
pub const DEFAULT_MATCHES: usize = 5;
/// The most records one match fetches to score, whatever the limit and the over-fetch come to.
pub const MAX_MATCH_FETCH: usize = 1000;

/// How many neighbors `find_similar` returns when the caller does not say.
pub const DEFAULT_NEIGHBORS: usize = 10;
/// The most neighbors `find_similar` will return.
pub const MAX_NEIGHBORS: usize = 100;
/// How many independent locks guard record ids. A power of two well above the expected request concurrency, so that
/// two different ids rarely share a lock.
const LOCK_STRIPES: usize = 256;
/// How many times a request reads a record, judges, and tries to write before it gives up with
/// [`EngineError::Conflict`]. Each try after the first means another writer changed the record in the moment
/// between this one's read and its write, which takes another replica working on the same record at the same
/// instant: once is rare, and this many times in a row means something is rewriting the record in a loop.
pub const WRITE_ATTEMPTS: usize = 4;

/// A failed request, classified the way a caller needs it classified.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
  /// The request is malformed or does not fit the schema. Retrying it unchanged will fail again.
  #[error("{0}")]
  InvalidArgument(String),
  /// The record asked for does not exist.
  #[error("{0}")]
  NotFound(String),
  /// The request is fine but the stored state forbids it (another brand's record, another model's vectors).
  #[error("{0}")]
  FailedPrecondition(String),
  /// Another writer kept changing the record while this request was being judged, so nothing was written.
  /// Sending the request again is safe.
  #[error("{0}")]
  Conflict(String),
  /// The server is saturated. Retry with backoff.
  #[error("{0}")]
  Overloaded(String),
  /// The request said when its caller stops waiting, and it could not be served by then. Nothing was written.
  #[error("{0}")]
  DeadlineExceeded(String),
  /// A dependency is unreachable. Retry with backoff.
  #[error("{0}")]
  Unavailable(String),
  /// Something is wrong that neither the caller nor a retry can fix.
  #[error("{0}")]
  Internal(String),
}

impl From<similarity_core::Error> for EngineError {
  fn from(error: similarity_core::Error) -> Self {
    match error {
      similarity_core::Error::Record(_) | similarity_core::Error::Schema(_) => Self::InvalidArgument(error.to_string()),
      other => Self::Internal(other.to_string()),
    }
  }
}

impl From<StoreError> for EngineError {
  fn from(error: StoreError) -> Self {
    match error {
      StoreError::Unavailable(_) => Self::Unavailable(error.to_string()),
      StoreError::LayoutMismatch { .. } => Self::FailedPrecondition(error.to_string()),
      other => Self::Internal(other.to_string()),
    }
  }
}

impl From<EmbedError> for EngineError {
  fn from(error: EmbedError) -> Self {
    match error {
      EmbedError::Overloaded => Self::Overloaded(error.to_string()),
      EmbedError::WorkerGone => Self::Unavailable(error.to_string()),
      EmbedError::Failed(_) => Self::Internal(error.to_string()),
      EmbedError::TooLate => Self::DeadlineExceeded(error.to_string()),
    }
  }
}

/// A record as a caller supplies it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecordInput {
  /// The caller's identifier, if the record has one yet.
  pub id: Option<String>,
  /// The key of the brand that owns the record.
  pub brand_id: Option<String>,
  /// Field name and text pairs.
  pub fields: Vec<(String, String)>,
  /// Quantity name and value pairs.
  pub quantities: Vec<(String, Quantity)>,
}

/// What an evaluation may do to the stored baseline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WriteMode {
  /// Write if new or accepted; leave the baseline alone if rejected.
  #[default]
  Enforce,
  /// Never write.
  DryRun,
  /// Always write, and make this revision the new anchor.
  Force,
}

/// An `evaluate` request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvaluateInput {
  /// The record to judge.
  pub record: RecordInput,
  /// What may be written.
  pub mode: WriteMode,
  /// Overrides the configured overall threshold for this request.
  pub threshold: Option<f32>,
}

/// The outcome of an evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
  /// No id: embedded only.
  Embedded,
  /// No baseline: this record is the baseline now.
  Created,
  /// Similar enough to the baseline.
  Accepted,
  /// Too different from the baseline.
  Rejected,
}

/// An `evaluate` response.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluateOutput {
  /// The outcome.
  pub verdict: Verdict,
  /// Overall similarity to the baseline, when there was one.
  pub similarity: Option<f32>,
  /// The overall threshold that applied.
  pub threshold: f32,
  /// Per-field detail, when there was a baseline.
  pub fields: Vec<FieldScore>,
  /// Overall similarity to the anchor, when there was a baseline.
  pub anchor_similarity: Option<f32>,
  /// Whether this call changed what is stored.
  pub stored: bool,
  /// Why the record was rejected: empty unless the verdict is [`Verdict::Rejected`], in the order the checks
  /// found them. The values, not their sentences: what to show a person is for the caller to render.
  pub reasons: Vec<Reason>,
  /// The record's composite vector.
  pub vector: Vec<f32>,
  /// The record's revision after this call; zero if it is not stored.
  pub revision: u64,
  /// Per-quantity detail, when there was a baseline.
  pub quantities: Vec<QuantityScore>,
  /// Whether the revision changed which measures the record is described by.
  pub measure_changed: bool,
}

impl EvaluateOutput {
  /// The response when there was nothing to compare against: the record had no id, or no baseline yet.
  fn without_baseline(verdict: Verdict, policy: &Policy, stored: bool, vector: Vec<f32>) -> Self {
    Self {
      verdict,
      similarity: None,
      threshold: policy.threshold,
      fields: Vec::new(),
      anchor_similarity: None,
      stored,
      reasons: Vec::new(),
      vector,
      revision: u64::from(stored),
      quantities: Vec::new(),
      measure_changed: false,
    }
  }
}

/// One record to import: what it is now and, if the caller still has it, what it was when it was created.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportInput {
  /// The record as it is now. Its `id` is required.
  pub record: RecordInput,
  /// The record as it was first created: its text fields and quantities then. `None` if that is not known, in
  /// which case the record as it is now becomes its own point of reference, as for any new record.
  pub anchor: Option<Original>,
}

/// An earlier version of a record: fields and quantities, with no id or brand of its own.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Original {
  /// Field name and text pairs.
  pub fields: Vec<(String, String)>,
  /// Quantity name and value pairs.
  pub quantities: Vec<(String, Quantity)>,
}

/// What importing one record did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imported {
  /// The id was not stored. It is now, at revision 1.
  Created,
  /// The id was already stored, and was left exactly as it was.
  Skipped,
}

/// What to search around.
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarTo {
  /// A stored record.
  Id(String),
  /// An arbitrary record, embedded for the occasion.
  Record(RecordInput),
}

/// A `find_similar` request.
#[derive(Debug, Clone, PartialEq)]
pub struct FindSimilarInput {
  /// What to search around.
  pub query: SimilarTo,
  /// How many neighbors to return; zero means [`DEFAULT_NEIGHBORS`].
  pub limit: usize,
  /// Only return this brand's records.
  pub brand_id: Option<String>,
  /// Only return records at least this similar.
  pub min_similarity: Option<f32>,
}

/// How a menu item is matched (see [`Engine::match_menu_item`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matching {
  /// The similarity the best candidate has to reach to be called a match. `None`: candidates are ranked and
  /// nothing is decided.
  pub threshold: Option<f32>,
  /// How many candidates are fetched for each one returned, before they are scored exactly. At least 1.
  pub overfetch: usize,
}

impl Default for Matching {
  fn default() -> Self {
    Self {
      threshold: None,
      overfetch: 5,
    }
  }
}

/// A request to match a menu item.
#[derive(Debug, Clone, Default)]
pub struct MatchInput {
  /// The menu item, under the schema's field and quantity names. It must carry no id and no brand.
  pub menu_item: RecordInput,
  /// How many candidates to return; zero means [`DEFAULT_MATCHES`].
  pub limit: usize,
  /// Only consider records of this brand.
  pub brand_id: Option<String>,
  /// Holds the best candidate to this in place of the configured threshold.
  pub threshold: Option<f32>,
}

/// The answer to a [`MatchInput`].
#[derive(Debug, Clone, PartialEq)]
pub struct MatchOutput {
  /// The candidates, best first.
  pub candidates: Vec<Ranked>,
  /// What was decided about the first of them.
  pub decision: Decision,
  /// The threshold it was held to, if there was one.
  pub threshold: Option<f32>,
  /// How many records the search returned to be scored.
  pub fetched: usize,
}

/// What the server is running with.
#[derive(Debug, Clone, PartialEq)]
pub struct Description {
  /// The model fingerprint.
  pub model_fingerprint: String,
  /// Width of one field vector.
  pub dimension: usize,
  /// Width of a composite.
  pub composite_dimension: usize,
  /// The schema.
  pub schema: Schema,
  /// The thresholds.
  pub policy: Policy,
  /// How menu items are matched.
  pub matching: Matching,
}

/// The layout id and composite width a store must be opened with to serve `schema` with `embedder`.
pub fn store_layout(schema: &Schema, embedder: &impl TextEmbedder) -> (String, usize) {
  (
    schema.layout_id(embedder.fingerprint()),
    schema.composite_dimension(embedder.dimension()),
  )
}

/// A proposed revision of a stored record, embedded and validated: everything `revise` judges and may write.
#[derive(Debug)]
struct Revision {
  id: String,
  brand_id: Option<String>,
  vectors: RecordVectors,
  quantities: Quantities,
}

/// One record of an import, validated: everything needed to write it once its texts are embedded.
#[derive(Debug)]
struct PreparedImport {
  id: String,
  brand_id: Option<String>,
  record: PreparedRecord,
  anchor: Option<PreparedRecord>,
}

/// Something that is told about every request for an id once its outcome is final (see [`Engine::with_observer`]).
///
/// An observer is told; it is never asked. `observe` returns nothing, so nothing an observer concludes can reach
/// the response or the store, and it is called with the record's lock still held, so what it is told about one id
/// arrives in the order the engine judged it. For the same reason it must not wait on anything: a request for
/// that id is waiting on it.
pub trait Observer: fmt::Debug + Send + Sync + 'static {
  /// Called once per successful `evaluate` of a record that has an id, after anything it writes is written.
  fn observe(&self, observation: &Observation<'_>);
}

/// What an [`Observer`] is told about one request.
#[derive(Debug, Clone, Copy)]
pub struct Observation<'a> {
  /// The record's id.
  pub id: &'a str,
  /// The brand the request named, if it named one.
  pub brand_id: Option<&'a str>,
  /// What the request asked to be done with the verdict.
  pub mode: WriteMode,
  /// The threshold the request asked to be judged at in place of the configured one, if it did.
  pub threshold: Option<f32>,
  /// The policy the request was judged by: the configured one, at the request's threshold if it asked for one.
  pub policy: &'a Policy,
  /// Whether the record was written.
  pub stored: bool,
  /// The record's revision counter after the request.
  pub revision: u64,
  /// What was judged. `None` when nothing was stored under the id, so there was nothing to judge against.
  pub judged: Option<Judged<'a>>,
}

/// A revision and the verdict on it, as an [`Observer`] is shown them.
#[derive(Debug, Clone, Copy)]
pub struct Judged<'a> {
  /// The record as it was stored when the revision arrived.
  pub baseline: &'a StoredRecord,
  /// The revision's vectors.
  pub candidate: &'a RecordVectors,
  /// The revision's quantities.
  pub quantities: &'a Quantities,
  /// What the engine made of it, under the policy the request was judged by.
  pub evaluation: &'a Evaluation,
}

/// A record's lock, held by the request that took it and by any write that request has in flight.
///
/// Shared, so that a request that is dropped while its write is on its way does not let the next request for the
/// record in before that write has landed and been reported.
type Held = Arc<OwnedMutexGuard<()>>;

/// What a write did to the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Change {
  /// Nothing: the record was no longer what the writer had read.
  Nothing,
  /// The record is stored, at this revision.
  Stored(u64),
  /// The record is gone.
  Deleted,
}

/// The result of a write, on its way to the request that asked for it.
///
/// A request can be gone by the time its write is done, and then nobody hears what happened: the response has
/// nowhere to go and the request's own log line is never written. So the result says it itself. If it is dropped
/// without having been taken, wherever that happens (the task's output that nobody collected, or collected and
/// dropped unread), it logs what the write came to.
struct Delivery<T> {
  result: Option<Result<(T, Change), EngineError>>,
  call: &'static str,
  id: String,
  /// Counts every line this writes, for a monitoring system that does not read logs.
  unanswered: Arc<AtomicU64>,
}

impl<T> Delivery<T> {
  fn take(mut self) -> Option<Result<(T, Change), EngineError>> {
    self.result.take()
  }
}

impl<T> Drop for Delivery<T> {
  fn drop(&mut self) {
    let (call, id) = (self.call, self.id.as_str());
    match &self.result {
      // Taken by the request, or a write that lost its race: nothing changed that somebody was not told about.
      None | Some(Ok((_, Change::Nothing))) => return,
      Some(Ok((_, Change::Stored(revision)))) => tracing::warn!(
        call,
        id,
        stored = true,
        deleted = false,
        revision,
        "a write finished after the request that asked for it had gone"
      ),
      Some(Ok((_, Change::Deleted))) => tracing::warn!(
        call,
        id,
        stored = false,
        deleted = true,
        "a write finished after the request that asked for it had gone"
      ),
      // The store may or may not hold the write: that is exactly what the caller cannot find out.
      Some(Err(error)) => tracing::warn!(
        call,
        id,
        %error,
        "a write failed after the request that asked for it had gone, and may or may not have been applied"
      ),
    }
    self.unanswered.fetch_add(1, Ordering::Relaxed);
  }
}

/// The engine's writes, as a monitoring system is told about them ([`Engine::writes`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Writes {
  /// On their way to the store right now.
  pub in_flight: u64,
  /// Finished, or failed, after the request that asked for them had gone, since the engine was made. Each has a
  /// log line.
  pub unanswered: u64,
}

/// Counts a write as in flight for as long as it exists.
struct InFlight(watch::Sender<usize>);

impl InFlight {
  fn begin(writes: &watch::Sender<usize>) -> Self {
    writes.send_modify(|count| *count = count.saturating_add(1));
    Self(writes.clone())
  }
}

impl Drop for InFlight {
  fn drop(&mut self) {
    self.0.send_modify(|count| *count = count.saturating_sub(1));
  }
}

/// The service's behavior.
#[derive(Debug)]
pub struct Engine<S, E> {
  /// Shared with the writes in flight, which outlive the request that started them.
  store: Arc<S>,
  embedder: E,
  schema: Schema,
  policy: Policy,
  matching: Matching,
  import_batch: usize,
  observer: Option<Arc<dyn Observer>>,
  locks: Vec<Arc<Mutex<()>>>,
  lock_hasher: RandomState,
  /// How many writes are in flight.
  writes: watch::Sender<usize>,
  /// How many writes ended with nobody to tell.
  unanswered: Arc<AtomicU64>,
}

fn required_text(what: &str, value: Option<String>) -> Result<Option<String>, EngineError> {
  match value {
    // Present-but-blank is almost always a caller that forgot to populate the field. Treating a blank id as "no
    // id" would quietly turn a guarded update into an unguarded embed-only call.
    Some(text) if text.trim().is_empty() => Err(EngineError::InvalidArgument(format!(
      "`{what}` is present but blank; omit it or give it a value"
    ))),
    other => Ok(other),
  }
}

/// Refuses a request that does not name the brand `stored` belongs to.
///
/// Naming no brand is refused like naming the wrong one: a check that a caller can skip by leaving a field out
/// would not catch the mistake it exists for, a code path that lost track of whose record it is editing. A record
/// stored without a brand accepts any request, which is how it gains one.
fn same_brand(stored: &StoredRecord, given: Option<&str>) -> Result<(), EngineError> {
  match (stored.brand_id.as_deref(), given) {
    (None, _) => Ok(()),
    (Some(brand), Some(given)) if brand == given => Ok(()),
    (Some(brand), Some(given)) => Err(EngineError::FailedPrecondition(format!(
      "record `{}` belongs to brand `{brand}`, not `{given}`",
      stored.id
    ))),
    (Some(brand), None) => Err(EngineError::FailedPrecondition(format!(
      "record `{}` belongs to brand `{brand}`, and the request names no brand",
      stored.id
    ))),
  }
}

/// What a request is answered when every attempt it made found the record changed by another writer.
fn contended(id: &str) -> EngineError {
  EngineError::Conflict(format!(
    "record `{id}` was changed by another writer each of the {WRITE_ATTEMPTS} times this request read it and went \
     to write, so nothing was written; send the request again"
  ))
}

fn now() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_or(0, |elapsed| elapsed.as_secs())
}

impl<S: VectorStore, E: TextEmbedder> Engine<S, E> {
  /// Assembles an engine. `store` must have been opened with [`store_layout`] for the same schema and embedder.
  pub fn new(store: S, embedder: E, schema: Schema, policy: Policy) -> Self {
    Self {
      store: Arc::new(store),
      embedder,
      schema,
      policy,
      matching: Matching::default(),
      import_batch: DEFAULT_IMPORT_BATCH,
      observer: None,
      locks: (0..LOCK_STRIPES).map(|_| Arc::new(Mutex::new(()))).collect(),
      lock_hasher: RandomState::new(),
      writes: watch::Sender::new(0),
      unanswered: Arc::default(),
    }
  }

  /// The same engine, taking at most `records` records of an import stream into one batch (at least one).
  #[must_use]
  pub fn with_import_batch(mut self, records: usize) -> Self {
    self.import_batch = records.max(1);
    self
  }

  /// The most records of an import stream that are imported as one batch.
  pub fn import_batch(&self) -> usize {
    self.import_batch
  }

  /// The same engine, matching menu items by `matching` from now on.
  #[must_use]
  pub fn with_matching(mut self, matching: Matching) -> Self {
    self.matching = matching;
    self
  }

  /// The same engine, telling `observer` about every judged request from now on.
  ///
  /// This is how a candidate policy is tried on live requests without deciding any of them (see
  /// [`crate::shadow`]). There is one observer; a second call replaces the first.
  #[must_use]
  pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
    self.observer = Some(observer);
    self
  }

  /// The store, for the drift replay, which reads a record's anchor before each step, and for tests that check
  /// what a request left behind. Reading only: every write goes through a request.
  pub(crate) fn store(&self) -> &S {
    &self.store
  }

  /// The embedder, so that a test can ask its stand-in what it was called with.
  #[cfg(test)]
  pub(crate) fn embedder(&self) -> &E {
    &self.embedder
  }

  /// Reports the model, schema, and thresholds in force.
  pub fn describe(&self) -> Description {
    let (_, composite_dimension) = store_layout(&self.schema, &self.embedder);
    Description {
      model_fingerprint: self.embedder.fingerprint().to_owned(),
      dimension: self.embedder.dimension(),
      composite_dimension,
      schema: self.schema.clone(),
      policy: self.policy,
      matching: self.matching,
    }
  }

  fn lock_for(&self, id: &str) -> Result<&Arc<Mutex<()>>, EngineError> {
    let stripes = u64::try_from(self.locks.len()).unwrap_or(u64::MAX);
    let hash = usize::try_from(self.lock_hasher.hash_one(id).checked_rem(stripes).unwrap_or(0)).unwrap_or(0);
    self
      .locks
      .get(hash)
      .ok_or_else(|| EngineError::Internal("lock stripe out of range".to_owned()))
  }

  /// Waits for the record's lock.
  async fn hold(&self, id: &str) -> Result<Held, EngineError> {
    Ok(Arc::new(Arc::clone(self.lock_for(id)?).lock_owned().await))
  }

  /// Runs `write` to its end and returns what it returns, whether or not this request is still there to be told.
  ///
  /// Everything a request does before it writes can be dropped with the request and leave nothing behind. A
  /// write cannot: the store may have applied a write whose future was dropped, and then the baseline has moved
  /// for a caller that was told nothing, with no line anywhere saying so. So `write` runs on a task of its own,
  /// with the record's lock, and this only waits for it. If the wait is dropped, the write goes on, tells the
  /// observer what it would have told it, and its result logs itself ([`Delivery`]).
  ///
  /// `write` is the store call and everything that has to follow it: it must not wait on anything else.
  async fn committed<T, W>(&self, held: &Held, call: &'static str, id: &str, write: W) -> Result<T, EngineError>
  where
    T: Send + 'static,
    W: Future<Output = Result<(T, Change), EngineError>> + Send + 'static,
  {
    let (held, in_flight) = (Arc::clone(held), InFlight::begin(&self.writes));
    let (delivery_id, unanswered) = (id.to_owned(), Arc::clone(&self.unanswered));
    let task = tokio::spawn(async move {
      let result = write.await;
      // The lock goes before the count does, and both before the result can be seen: a request that waited for
      // this one finds the record as this write left it, and a server that waited for writes finds none.
      drop(held);
      drop(in_flight);
      Delivery {
        result: Some(result),
        call,
        id: delivery_id,
        unanswered,
      }
    });
    match task.await.map(Delivery::take) {
      Ok(Some(result)) => result.map(|(value, _)| value),
      Ok(None) => Err(EngineError::Internal(format!(
        "the write of record `{id}` finished without a result"
      ))),
      Err(stopped) => Err(EngineError::Internal(format!(
        "the write of record `{id}` did not run to its end, and may or may not have been applied: {stopped}"
      ))),
    }
  }

  /// How many writes are in flight, and how many have ended with nobody to tell.
  pub fn writes(&self) -> Writes {
    Writes {
      in_flight: u64::try_from(*self.writes.borrow()).unwrap_or(u64::MAX),
      unanswered: self.unanswered.load(Ordering::Relaxed),
    }
  }

  /// Resolves when no write is in flight. A server that is stopping waits for this after its last request, so
  /// that a write whose request has gone is not cut off with the process.
  pub async fn settled(&self) {
    let mut writes = self.writes.subscribe();
    // The sender is this engine's, so the wait cannot fail while the engine is there to be asked.
    let _ = writes.wait_for(|count| *count == 0).await;
  }

  fn prepare(&self, record: &RecordInput) -> Result<PreparedRecord, EngineError> {
    let fields = record.fields.iter().map(|(name, text)| (name.as_str(), text.as_str()));
    let quantities = record
      .quantities
      .iter()
      .map(|(name, quantity)| (name.as_str(), quantity.clone()));
    Ok(PreparedRecord::new(&self.schema, fields)?.with_quantities(&self.schema, quantities)?)
  }

  /// Embeds a prepared record, reusing `baseline`'s vectors for fields whose text has not changed.
  async fn embed(
    &self,
    prepared: &PreparedRecord,
    baseline: Option<&RecordVectors>,
  ) -> Result<RecordVectors, EngineError> {
    let dimension = self.embedder.dimension();
    let plan = prepared.plan(&self.schema, dimension, baseline)?;
    let texts: Vec<String> = plan.texts().into_iter().map(str::to_owned).collect();
    let embedded = self.embedder.embed(texts).await?;
    Ok(plan.finish(&self.schema, dimension, &embedded)?)
  }

  /// Whether a request sent now could be served: the embedder can still embed and the store answers. It embeds
  /// nothing and reads no record. This is what the server's readiness is made from.
  ///
  /// # Errors
  ///
  /// [`EngineError::Unavailable`] if the embedding thread has stopped or the store cannot be reached, and
  /// whatever else the store says is wrong with it.
  pub async fn check(&self) -> Result<(), EngineError> {
    if !self.embedder.alive() {
      return Err(EngineError::Unavailable(
        "the embedding thread has stopped: nothing can be embedded until the server is started again".to_owned(),
      ));
    }
    Ok(self.store.check().await?)
  }

  /// Embeds a record and, when it has an id, judges it against its stored baseline.
  ///
  /// # Errors
  ///
  /// See [`EngineError`]. Nothing is written if an error is returned.
  pub async fn evaluate(&self, input: EvaluateInput) -> Result<EvaluateOutput, EngineError> {
    let policy = match input.threshold {
      Some(threshold) => self.policy.with_threshold(threshold)?,
      None => self.policy,
    };
    let id = required_text("record.id", input.record.id.clone())?;
    let brand_id = required_text("record.brand_id", input.record.brand_id.clone())?;
    let prepared = self.prepare(&input.record)?;

    let Some(id) = id else {
      let vectors = self.embed(&prepared, None).await?;
      return Ok(EvaluateOutput::without_baseline(
        Verdict::Embedded,
        &policy,
        false,
        vectors.composite,
      ));
    };

    let held = self.hold(&id).await?;
    let asked = (input.mode, input.threshold);
    for _ in 0..WRITE_ATTEMPTS {
      let attempt = self.attempt(&held, (&id, brand_id.as_deref()), &prepared, asked, &policy);
      if let Some(output) = attempt.await? {
        return Ok(output);
      }
    }
    Err(contended(&id))
  }

  /// Reads the record, judges the request against it, and writes what that allows, provided the record is still
  /// what was read. `None` means it was not: another writer changed it in between, nothing was written, nobody
  /// was told, and the whole of this has to be done again against what is stored now.
  async fn attempt(
    &self,
    held: &Held,
    (id, brand_id): (&str, Option<&str>),
    prepared: &PreparedRecord,
    asked: (WriteMode, Option<f32>),
    policy: &Policy,
  ) -> Result<Option<EvaluateOutput>, EngineError> {
    let baseline = self.store.get(id).await?;
    if let Some(stored) = &baseline {
      same_brand(stored, brand_id)?;
    }
    let vectors = self
      .embed(prepared, baseline.as_ref().map(|record| &record.vectors))
      .await?;
    let revision = Revision {
      id: id.to_owned(),
      brand_id: brand_id.map(str::to_owned),
      vectors,
      quantities: prepared.quantities().clone(),
    };
    match baseline {
      Some(baseline) => self.revise(held, revision, asked, policy, baseline).await,
      None => self.create(held, revision, asked, policy).await,
    }
  }

  /// Stores the first version of a record, unless the mode forbids writing. `None` if another writer created the
  /// record first.
  async fn create(
    &self,
    held: &Held,
    first: Revision,
    asked: (WriteMode, Option<f32>),
    policy: &Policy,
  ) -> Result<Option<EvaluateOutput>, EngineError> {
    let (mode, threshold) = asked;
    let Revision {
      id,
      brand_id,
      vectors: candidate,
      quantities,
    } = first;
    let stored = mode != WriteMode::DryRun;
    let record = stored.then(|| StoredRecord {
      id: id.clone(),
      brand_id: brand_id.clone(),
      anchor: candidate.composite.clone(),
      vectors: candidate.clone(),
      // A new record is its own point of reference, for quantities as for text.
      anchor_quantities: quantities.clone(),
      quantities,
      revision: 1,
      updated_at: now(),
    });
    let (observer, policy, named) = (self.observer.clone(), *policy, id.clone());
    // Telling the observer and making the answer: what follows the write, if there is one, and is all there is
    // to do if there is not.
    let finish = move || {
      if let Some(observer) = &observer {
        observer.observe(&Observation {
          id: &id,
          brand_id: brand_id.as_deref(),
          mode,
          threshold,
          policy: &policy,
          stored,
          revision: u64::from(stored),
          judged: None,
        });
      }
      EvaluateOutput::without_baseline(Verdict::Created, &policy, stored, candidate.composite)
    };
    let Some(record) = record else {
      return Ok(Some(finish()));
    };
    let store = Arc::clone(&self.store);
    let write = async move {
      if !store.put_if(&record, Expected::Absent).await? {
        return Ok((None, Change::Nothing));
      }
      Ok((Some(finish()), Change::Stored(record.revision)))
    };
    self.committed(held, "evaluate", &named, write).await
  }

  /// The record a revision that is to be written leaves stored: `baseline`'s next revision.
  fn revised(
    &self,
    mode: WriteMode,
    baseline: &StoredRecord,
    (id, brand_id): (String, Option<String>),
    candidate: &RecordVectors,
    quantities: &Quantities,
  ) -> Result<StoredRecord, EngineError> {
    // A forced write is a reviewer saying "this is the record now", so it also becomes the point of reference
    // for drift. Every other write leaves the anchor's values where they were, and only lets it take the first
    // value of a field it has never had one for.
    let (anchor, anchor_quantities) = if mode == WriteMode::Force {
      (candidate.composite.clone(), quantities.clone())
    } else {
      let anchor = score::extend_anchor(
        &self.schema,
        self.embedder.dimension(),
        &baseline.anchor,
        &candidate.composite,
      )?;
      (anchor, first_values(&baseline.anchor_quantities, quantities))
    };
    Ok(StoredRecord {
      id,
      brand_id,
      anchor,
      vectors: candidate.clone(),
      quantities: quantities.clone(),
      anchor_quantities,
      revision: baseline.revision.saturating_add(1),
      updated_at: now(),
    })
  }

  /// Judges `candidate` against `baseline` and writes it back if the policy and the mode allow. `asked` is the
  /// request's mode and the threshold it asked for, if it asked for one; `policy` already has that threshold.
  /// `None` if there was something to write and `baseline` was no longer what is stored.
  async fn revise(
    &self,
    held: &Held,
    candidate: Revision,
    asked: (WriteMode, Option<f32>),
    policy: &Policy,
    baseline: StoredRecord,
  ) -> Result<Option<EvaluateOutput>, EngineError> {
    let (mode, threshold) = asked;
    let Revision {
      id,
      brand_id,
      vectors: candidate,
      quantities,
    } = candidate;
    let mut evaluation = score::evaluate(
      &self.schema,
      policy,
      self.embedder.dimension(),
      &baseline.vectors,
      Some(&baseline.anchor),
      &candidate,
    )?;
    evaluation.judge_quantities(
      &self.schema,
      policy,
      &baseline.quantities,
      &baseline.anchor_quantities,
      &quantities,
    );
    let accepted = evaluation.accepted();
    // Re-submitting exactly what is stored is common (a save with no edits, a retried request) and must not churn
    // the store or the revision counter. A quantity rewritten in another unit (`1 g` as `1000 mg`) is the same
    // quantity, as text re-saved with different whitespace is the same text. Any other difference in the
    // quantities is a change to the record, even when no text moved, and has to be written.
    let unchanged = |field: &FieldScore| field.change == FieldChange::Unchanged;
    let same_amount = |quantity: &QuantityScore| quantity.change == QuantityChange::Unchanged;
    let identical = evaluation.comparison.fields.iter().all(unchanged)
      && candidate.hashes == baseline.vectors.hashes
      && evaluation.quantities.iter().all(same_amount)
      && brand_id == baseline.brand_id;
    let write = match mode {
      WriteMode::DryRun => false,
      WriteMode::Enforce => accepted && !identical,
      WriteMode::Force => true,
    };

    let revision = baseline.revision.saturating_add(u64::from(write));
    let record = if write {
      let ids = (id.clone(), brand_id.clone());
      Some(self.revised(mode, &baseline, ids, &candidate, &quantities)?)
    } else {
      None
    };

    let (observer, policy, named) = (self.observer.clone(), *policy, id.clone());
    let read = baseline.version();
    // Telling the observer and making the answer. Only after the write, if there is one, is the outcome final:
    // a write that failed returns its error, a write that lost to another writer is judged again, and neither
    // is observed.
    let finish = move || {
      if let Some(observer) = &observer {
        observer.observe(&Observation {
          id: &id,
          brand_id: brand_id.as_deref(),
          mode,
          threshold,
          policy: &policy,
          stored: write,
          revision,
          judged: Some(Judged {
            baseline: &baseline,
            candidate: &candidate,
            quantities: &quantities,
            evaluation: &evaluation,
          }),
        });
      }
      EvaluateOutput {
        verdict: if accepted { Verdict::Accepted } else { Verdict::Rejected },
        similarity: Some(evaluation.comparison.similarity),
        threshold: policy.threshold,
        fields: evaluation.comparison.fields,
        anchor_similarity: evaluation.anchor_similarity,
        stored: write,
        reasons: evaluation.reasons,
        vector: candidate.composite,
        revision,
        quantities: evaluation.quantities,
        measure_changed: evaluation.measure_changed,
      }
    };
    let Some(record) = record else {
      return Ok(Some(finish()));
    };
    let store = Arc::clone(&self.store);
    let write = async move {
      // The verdict is about the baseline that was read. If that is no longer what is stored, the verdict is
      // about nothing, and writing would replace a record this request never saw: a forced write included,
      // whose revision number would be wrong.
      if !store.put_if(&record, Expected::Version(read)).await? {
        return Ok((None, Change::Nothing));
      }
      Ok((Some(finish()), Change::Stored(revision)))
    };
    self.committed(held, "evaluate", &named, write).await
  }

  /// Imports records that are not stored yet, embedding all of them in one call to the model.
  ///
  /// This is how a catalog is loaded, and how it is loaded again after a change of model (D12), without one
  /// request per record. It **only ever creates**. An id that is already stored is left exactly as it is and
  /// reported as [`Imported::Skipped`], so an import can never replace an anchor value (D8), running one twice is
  /// harmless, and an interrupted one is resumed by running it again.
  ///
  /// With an `anchor`, the record's point of reference for drift is its original version instead of its current
  /// one: the anchor holds the original's value for every field and quantity the original had, and the current
  /// value for anything it lacked, which is what the anchor would have become had the record lived here all along.
  /// That is what lets a re-index keep a record's drift history.
  ///
  /// Returns one outcome per input, in order. A record that cannot be imported (it does not fit the schema, or
  /// its id is stored under a brand it does not name) fails by itself and stops nothing else.
  ///
  /// # Errors
  ///
  /// Returns an error for the whole call only if the model fails: nothing is written in that case. Everything
  /// else is reported per record.
  pub async fn import(&self, inputs: Vec<ImportInput>) -> Result<Vec<Result<Imported, EngineError>>, EngineError> {
    let dimension = self.embedder.dimension();

    // For every record: validate it, see whether it is stored already, and if it is not, work out what to embed.
    // Records are embedded from scratch (there is no baseline to reuse).
    let mut ready = Vec::with_capacity(inputs.len());
    let mut texts: Vec<String> = Vec::new();
    for input in inputs {
      let prepared = match self.prepare_import(input) {
        Ok(prepared) => prepared,
        Err(error) => {
          ready.push(Err(error));
          continue;
        }
      };
      // A record that is already stored is left alone, so its text is not worth embedding: sending an export
      // again, to resume it or to add what is new, costs a read for each record that is there and no more. This
      // look is not under the record's lock and decides nothing by itself about a record that is absent; that is
      // decided again, under the lock, when it is written.
      match self.stored_already(&prepared).await {
        Ok(false) => {}
        Ok(true) => {
          ready.push(Ok(None));
          continue;
        }
        Err(error) => {
          ready.push(Err(error));
          continue;
        }
      }
      let wanted = |record: &PreparedRecord| -> Result<Vec<String>, EngineError> {
        let plan = record.plan(&self.schema, dimension, None)?;
        Ok(plan.texts().into_iter().map(str::to_owned).collect())
      };
      // Both lists are worked out before either is queued: the vectors that come back are told apart by
      // position alone, so a record must put in all of its texts or none.
      let counted = wanted(&prepared.record).and_then(|record_texts| {
        let anchor_texts = prepared.anchor.as_ref().map(wanted).transpose()?.unwrap_or_default();
        Ok((record_texts, anchor_texts))
      });
      ready.push(counted.map(|(record_texts, anchor_texts)| {
        let counts = (record_texts.len(), anchor_texts.len());
        texts.extend(record_texts);
        texts.extend(anchor_texts);
        Some((prepared, counts.0, counts.1))
      }));
    }
    // Nothing to embed (every record was refused, or is stored) is not a reason to wake the model.
    let embedded = if texts.is_empty() {
      Vec::new()
    } else {
      self.embedder.embed(texts).await?
    };

    let mut vectors = embedded.iter();
    let mut outcomes = Vec::with_capacity(ready.len());
    for prepared in ready {
      let outcome = match prepared {
        Err(error) => Err(error),
        Ok(None) => Ok(Imported::Skipped),
        Ok(Some((prepared, record_texts, anchor_texts))) => {
          // Each record's vectors are the next ones in line, whether or not the record ends up being written.
          let record: Vec<Vec<f32>> = vectors.by_ref().take(record_texts).cloned().collect();
          let anchor: Vec<Vec<f32>> = vectors.by_ref().take(anchor_texts).cloned().collect();
          self.store_import(prepared, &record, &anchor).await
        }
      };
      outcomes.push(outcome);
    }
    Ok(outcomes)
  }

  /// Whether a record to be imported is stored already, under a brand the import may name.
  async fn stored_already(&self, prepared: &PreparedImport) -> Result<bool, EngineError> {
    match self.store.get(&prepared.id).await? {
      Some(stored) => {
        // Naming another brand is the mistake the brand rule exists for, here as anywhere (D14).
        same_brand(&stored, prepared.brand_id.as_deref())?;
        Ok(true)
      }
      None => Ok(false),
    }
  }

  fn prepare_import(&self, input: ImportInput) -> Result<PreparedImport, EngineError> {
    let Some(id) = required_text("record.id", input.record.id.clone())? else {
      return Err(EngineError::InvalidArgument(
        "`record.id` is required: a record without an id cannot be stored".to_owned(),
      ));
    };
    let brand_id = required_text("record.brand_id", input.record.brand_id.clone())?;
    let record = self.prepare(&input.record)?;
    let anchor = input
      .anchor
      .map(|original| {
        self.prepare(&RecordInput {
          id: None,
          brand_id: None,
          fields: original.fields,
          quantities: original.quantities,
        })
      })
      .transpose()
      // The same message would otherwise read as a complaint about the record itself.
      .map_err(|error| match error {
        EngineError::InvalidArgument(message) => {
          EngineError::InvalidArgument(format!("the original version: {message}"))
        }
        other => other,
      })?;
    Ok(PreparedImport {
      id,
      brand_id,
      record,
      anchor,
    })
  }

  /// Writes one prepared import, unless its id is already stored.
  async fn store_import(
    &self,
    prepared: PreparedImport,
    record: &[Vec<f32>],
    anchor: &[Vec<f32>],
  ) -> Result<Imported, EngineError> {
    let dimension = self.embedder.dimension();
    let vectors = prepared
      .record
      .plan(&self.schema, dimension, None)?
      .finish(&self.schema, dimension, record)?;
    let quantities = prepared.record.quantities().clone();
    let (anchor, anchor_quantities) = match &prepared.anchor {
      Some(original) => {
        let original_vectors = original
          .plan(&self.schema, dimension, None)?
          .finish(&self.schema, dimension, anchor)?;
        (
          // The original's values, plus the current value of anything the original did not have.
          score::extend_anchor(&self.schema, dimension, &original_vectors.composite, &vectors.composite)?,
          first_values(original.quantities(), &quantities),
        )
      }
      None => (vectors.composite.clone(), quantities.clone()),
    };
    let record = StoredRecord {
      id: prepared.id,
      brand_id: prepared.brand_id,
      vectors,
      anchor,
      quantities,
      anchor_quantities,
      revision: 1,
      updated_at: now(),
    };

    let held = self.hold(&record.id).await?;
    let record = Arc::new(record);
    for _ in 0..WRITE_ATTEMPTS {
      if let Some(stored) = self.store.get(&record.id).await? {
        // Naming another brand is the mistake the brand rule exists for, here as anywhere (D14).
        same_brand(&stored, record.brand_id.as_deref())?;
        return Ok(Imported::Skipped);
      }
      // "Only creates" is the store's to keep, not this process's lock: a record another replica, or the import
      // command beside it, stored a moment ago is left alone like any other (D20).
      let (store, new) = (Arc::clone(&self.store), Arc::clone(&record));
      let write = async move {
        let created = store.put_if(&new, Expected::Absent).await?;
        Ok((
          created,
          if created {
            Change::Stored(new.revision)
          } else {
            Change::Nothing
          },
        ))
      };
      if self.committed(&held, "import", &record.id, write).await? {
        return Ok(Imported::Created);
      }
    }
    Err(contended(&record.id))
  }

  /// Finds stored records similar to a stored or ad hoc record, most similar first.
  ///
  /// # Errors
  ///
  /// See [`EngineError`]; in particular [`EngineError::NotFound`] when searching around an id that is not stored.
  pub async fn find_similar(&self, input: FindSimilarInput) -> Result<Vec<Neighbor>, EngineError> {
    let limit = match input.limit {
      0 => DEFAULT_NEIGHBORS,
      limit if limit <= MAX_NEIGHBORS => limit,
      limit => {
        return Err(EngineError::InvalidArgument(format!(
          "`limit` is {limit}; the most neighbors one request may return is {MAX_NEIGHBORS}"
        )));
      }
    };
    if let Some(minimum) = input.min_similarity
      && !(-1.0..=1.0).contains(&minimum)
    {
      return Err(EngineError::InvalidArgument(format!(
        "`min_similarity` is {minimum}, outside the cosine range [-1, 1]"
      )));
    }
    let brand_id = required_text("brand_id", input.brand_id)?;

    let (vector, exclude_id) = match input.query {
      SimilarTo::Id(id) => {
        let Some(id) = required_text("id", Some(id))? else {
          return Err(EngineError::InvalidArgument("`id` is required".to_owned()));
        };
        let Some(stored) = self.store.get(&id).await? else {
          return Err(EngineError::NotFound(format!("no record `{id}` is stored")));
        };
        (stored.vectors.composite, Some(id))
      }
      SimilarTo::Record(record) => {
        let exclude_id = required_text("record.id", record.id.clone())?;
        let prepared = self.prepare(&record)?;
        (self.embed(&prepared, None).await?.composite, exclude_id)
      }
    };

    let query = SearchQuery {
      vector: &vector,
      limit,
      brand_id: brand_id.as_deref(),
      exclude_id: exclude_id.as_deref(),
      min_similarity: input.min_similarity,
    };
    Ok(self.store.search(query).await?)
  }

  /// Finds the stored records `input.menu_item` describes, best first, and decides whether the best is a match.
  ///
  /// The menu item is embedded under the schema like any record and searched for like any record, among every
  /// brand's records unless `input.brand_id` narrows it. Then each of the records the search returned is scored
  /// exactly over the fields it shares with the menu item ([`matching::rank`]), which is what the search score
  /// cannot do, and the best is held to the threshold if there is one. More records are fetched than are
  /// returned ([`Matching::overfetch`]), because the search order is not the exact order.
  ///
  /// Nothing is written, whatever is decided.
  ///
  /// # Errors
  ///
  /// [`EngineError::InvalidArgument`] for a menu item that does not fit the schema or carries an id or a brand,
  /// a limit above [`MAX_NEIGHBORS`], or a threshold outside the cosine range; otherwise as [`Self::evaluate`].
  pub async fn match_menu_item(&self, input: MatchInput) -> Result<MatchOutput, EngineError> {
    if input.menu_item.id.is_some() || input.menu_item.brand_id.is_some() {
      return Err(EngineError::InvalidArgument(
        "a menu item carries no `id` and no `brand_id`: both belong to the system the menu item came from. To match \
         within one brand, name it in the request's `brand_id`"
          .to_owned(),
      ));
    }
    let limit = match input.limit {
      0 => DEFAULT_MATCHES,
      limit if limit <= MAX_NEIGHBORS => limit,
      limit => {
        return Err(EngineError::InvalidArgument(format!(
          "`limit` is {limit}; the most candidates one request may return is {MAX_NEIGHBORS}"
        )));
      }
    };
    let threshold = input.threshold.or(self.matching.threshold);
    if let Some(threshold) = input.threshold
      && !(-1.0..=1.0).contains(&threshold)
    {
      return Err(EngineError::InvalidArgument(format!(
        "`threshold` is {threshold}, outside the cosine range [-1, 1]"
      )));
    }
    let brand_id = required_text("brand_id", input.brand_id)?;
    let prepared = self.prepare(&input.menu_item)?;
    let menu_item = self.embed(&prepared, None).await?;

    let found = self
      .store
      .search_vectors(SearchQuery {
        vector: &menu_item.composite,
        limit: limit.saturating_mul(self.matching.overfetch).min(MAX_MATCH_FETCH),
        brand_id: brand_id.as_deref(),
        exclude_id: None,
        min_similarity: None,
      })
      .await?;
    let candidates = matching::rank(
      &self.schema,
      self.embedder.dimension(),
      &menu_item,
      prepared.quantities(),
      found.iter().map(|found| Candidate {
        id: &found.neighbor.id,
        brand_id: found.neighbor.brand_id.as_deref(),
        search_similarity: found.neighbor.similarity,
        vectors: &found.vectors,
        quantities: &found.quantities,
      }),
      limit,
    )?;
    Ok(MatchOutput {
      decision: matching::decide(threshold, &candidates),
      candidates,
      threshold,
      fetched: found.len(),
    })
  }

  /// Forgets a record. Returns whether there was one.
  ///
  /// # Errors
  ///
  /// See [`EngineError`].
  pub async fn delete(&self, id: String, brand_id: Option<String>) -> Result<bool, EngineError> {
    let Some(id) = required_text("id", Some(id))? else {
      return Err(EngineError::InvalidArgument("`id` is required".to_owned()));
    };
    let brand_id = required_text("brand_id", brand_id)?;
    let held = self.hold(&id).await?;
    for _ in 0..WRITE_ATTEMPTS {
      let Some(stored) = self.store.get(&id).await? else {
        return Ok(false);
      };
      same_brand(&stored, brand_id.as_deref())?;
      // The record that is deleted is the one whose brand was just checked. One that another writer has since
      // replaced, and may have given a brand, is looked at again.
      let (store, gone, read) = (Arc::clone(&self.store), id.clone(), stored.version());
      let write = async move {
        let deleted = store.delete_if(&gone, read).await?;
        Ok((deleted, if deleted { Change::Deleted } else { Change::Nothing }))
      };
      if self.committed(&held, "delete", &id, write).await? {
        return Ok(true);
      }
    }
    Err(contended(&id))
  }
}

/// A store that lets a test stand in for a second replica: shared with the engine tests and the properties.
#[cfg(test)]
pub(crate) mod racing {
  use std::collections::VecDeque;
  use std::pin::Pin;
  use std::sync::atomic::{AtomicUsize, Ordering};
  use std::sync::{Arc, Mutex};

  use similarity_store::{
    Expected, Found, MemoryStore, Neighbor, SearchQuery, StoreError, StoredRecord, VectorStore, Version,
  };

  /// What another replica does to the store while this engine is between a read and its write.
  type Rival = Pin<Box<dyn Future<Output = ()> + Send>>;

  /// A store another replica gets a request into just after one of this engine's reads. The read returns what was
  /// stored before the rival wrote, which is what a read from a store two processes share can be by the time it
  /// is acted on. One entry is taken for each read; `None` is a read with nobody beside it.
  pub(crate) struct Raced {
    store: Arc<MemoryStore>,
    rivals: Mutex<VecDeque<Option<Rival>>>,
    reads: AtomicUsize,
  }

  impl Raced {
    pub(crate) fn over(store: Arc<MemoryStore>) -> Self {
      Self {
        store,
        rivals: Mutex::default(),
        reads: AtomicUsize::new(0),
      }
    }

    /// Lets `skipped` reads through, and gets `rival` in after the next one.
    pub(crate) fn after_read(&self, skipped: usize, rival: impl Future<Output = ()> + Send + 'static) {
      let mut rivals = self.rivals.lock().unwrap();
      rivals.extend((0..skipped).map(|_| None));
      rivals.push_back(Some(Box::pin(rival)));
    }

    /// A rival that no read has let in yet, if there is one: the engine did not read as often as was planned.
    pub(crate) fn waiting(&self) -> Option<impl Future<Output = ()> + Send + use<>> {
      self.rivals.lock().unwrap().drain(..).flatten().next()
    }

    pub(crate) fn reads(&self) -> usize {
      self.reads.load(Ordering::Relaxed)
    }

    /// The store itself, as the other replica sees it.
    pub(crate) fn shared(&self) -> &Arc<MemoryStore> {
      &self.store
    }
  }

  impl VectorStore for Raced {
    async fn check(&self) -> Result<(), StoreError> {
      self.store.check().await
    }

    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      self.reads.fetch_add(1, Ordering::Relaxed);
      let found = self.store.get(id).await;
      let rival = self.rivals.lock().unwrap().pop_front().flatten();
      if let Some(rival) = rival {
        // As a task of its own, which is what another replica is, and so that its work is not stacked on top of
        // the request that is waiting here: the two together are deeper than a test thread's stack.
        tokio::spawn(rival).await.unwrap();
      }
      found
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      self.store.put(record).await
    }

    async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
      self.store.put_if(record, expected).await
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.store.delete(id).await
    }

    async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
      self.store.delete_if(id, version).await
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.store.search(query).await
    }

    async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
      self.store.search_vectors(query).await
    }
  }
}

#[cfg(test)]
mod tests {
  use std::pin::Pin;
  use std::sync::Arc;

  use similarity_core::{FieldSpec, QuantitySpec};
  use similarity_store::{Found, MemoryStore, Version};

  use super::racing::Raced;
  use super::*;
  use crate::worker::testing::WordBagEmbedder;

  type TestEngine = Engine<MemoryStore, WordBagEmbedder>;

  fn schema() -> Schema {
    let field = |name: &str, weight: f32, required: bool, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required,
      reject_below,
    };
    Schema::new(vec![
      field("name", 4.0, true, Some(0.3)),
      field("category", 2.0, false, None),
      field("description", 1.0, false, None),
    ])
    .unwrap()
  }

  fn engine_with(policy: Policy) -> TestEngine {
    engine_over(schema(), policy)
  }

  fn engine_over(schema: Schema, policy: Policy) -> TestEngine {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema, &embedder);
    Engine::new(MemoryStore::new(layout, dimension), embedder, schema, policy)
  }

  /// The usual schema plus three quantities, each with the limits given: `(factor limit, reject incomparable)`.
  fn measured_schema(limits: (Option<f64>, bool)) -> Schema {
    let quantity = |name: &str| QuantitySpec {
      name: name.to_owned(),
      reject_above_factor: limits.0,
      reject_incomparable: limits.1,
    };
    schema()
      .with_quantities(vec![quantity("weight"), quantity("volume"), quantity("thc")])
      .unwrap()
  }

  /// An engine whose quantities are compared and reported and never reject: the shipped default.
  fn measuring_engine() -> TestEngine {
    engine_over(measured_schema((None, false)), Policy::new(0.75, None).unwrap())
  }

  fn grams(value: f64) -> Quantity {
    Quantity::single(value, "g")
  }

  fn percent(min: f64, max: f64) -> Quantity {
    Quantity::range(min, max, "%")
  }

  /// The gummies under id `p1`, with these quantities.
  fn measured(quantities: &[(&str, Quantity)]) -> EvaluateInput {
    EvaluateInput {
      record: RecordInput {
        quantities: quantities
          .iter()
          .map(|(name, value)| ((*name).to_owned(), value.clone()))
          .collect(),
        ..record(Some("p1"), &GUMMIES)
      },
      ..EvaluateInput::default()
    }
  }

  fn quantity<'a>(output: &'a EvaluateOutput, name: &str) -> &'a QuantityScore {
    output.quantities.iter().find(|score| score.field == name).unwrap()
  }

  fn engine() -> TestEngine {
    engine_with(Policy::new(0.75, None).unwrap())
  }

  fn record(id: Option<&str>, fields: &[(&str, &str)]) -> RecordInput {
    RecordInput {
      id: id.map(str::to_owned),
      brand_id: None,
      fields: fields
        .iter()
        .map(|(name, text)| ((*name).to_owned(), (*text).to_owned()))
        .collect(),
      quantities: Vec::new(),
    }
  }

  fn evaluate_input(id: Option<&str>, fields: &[(&str, &str)]) -> EvaluateInput {
    EvaluateInput {
      record: record(id, fields),
      ..EvaluateInput::default()
    }
  }

  const GUMMIES: [(&str, &str); 3] = [
    ("name", "Blueberry Gummies"),
    ("category", "Edibles Gummies"),
    ("description", "Ten vegan gummies with real fruit"),
  ];
  const GUMMIES_EDITED: [(&str, &str); 3] = [
    ("name", "Blueberry Gummies 100mg"),
    ("category", "Edibles Gummies"),
    ("description", "Ten vegan gummies with real fruit"),
  ];
  const VAPE: [(&str, &str); 3] = [
    ("name", "Sour Diesel Vape Cartridge"),
    ("category", "Vape Pens"),
    ("description", "One gram of live resin"),
  ];

  async fn stored(engine: &TestEngine, id: &str) -> Option<StoredRecord> {
    engine.store.get(id).await.unwrap()
  }

  /// A store whose reads give way to other tasks, as a round trip to a real store does, and are counted.
  struct Slow(MemoryStore, std::sync::atomic::AtomicUsize);

  impl Slow {
    fn over(store: MemoryStore) -> Self {
      Self(store, std::sync::atomic::AtomicUsize::new(0))
    }

    fn reads(&self) -> usize {
      self.1.load(std::sync::atomic::Ordering::Relaxed)
    }
  }

  impl VectorStore for Slow {
    async fn check(&self) -> Result<(), StoreError> {
      self.0.check().await
    }

    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      self.1.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
      let found = self.0.get(id).await;
      tokio::task::yield_now().await;
      found
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      self.0.put(record).await
    }

    async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
      self.0.put_if(record, expected).await
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.0.delete(id).await
    }

    async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
      self.0.delete_if(id, version).await
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.0.search(query).await
    }

    async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
      self.0.search_vectors(query).await
    }
  }

  const RIGHT: [(&str, &str); 3] = [
    ("name", "blueberry gummies 100mg"),
    ("category", "edibles gummies"),
    ("description", "ten vegan gummies with real fruit"),
  ];
  /// Closer to nothing but the search: one field, so its composite is all name.
  const WRONG: [(&str, &str); 1] = [("name", "blueberry gummies 10pk tin")];
  const MENU_ITEM: [(&str, &str); 1] = [("name", "blueberry gummies 100mg 10pk")];

  fn menu_item(fields: &[(&str, &str)]) -> MatchInput {
    MatchInput {
      menu_item: record(None, fields),
      ..MatchInput::default()
    }
  }

  async fn catalog(matching: Matching) -> TestEngine {
    let engine = engine().with_matching(matching);
    for (id, brand_id, fields) in [
      ("right", "brand-1", RIGHT.as_slice()),
      ("wrong", "brand-2", WRONG.as_slice()),
      ("vape", "brand-2", VAPE.as_slice()),
    ] {
      let mut input = evaluate_input(Some(id), fields);
      input.record.brand_id = Some(brand_id.to_owned());
      engine.evaluate(input).await.unwrap();
    }
    engine
  }

  /// D5's skew, and what a match does about it. A menu item has a name and nothing else. The search puts a
  /// record that also has only a name ahead of the right product, which has every field filled in; scored over
  /// the field both sides have, the right product is first.
  #[tokio::test]
  async fn a_name_only_menu_item_finds_its_product_ahead_of_a_wrong_one_the_search_prefers() {
    let engine = catalog(Matching::default()).await;
    let searched = engine
      .find_similar(FindSimilarInput {
        query: SimilarTo::Record(record(None, &MENU_ITEM)),
        limit: 3,
        brand_id: None,
        min_similarity: None,
      })
      .await
      .unwrap();
    assert_eq!(
      searched[0].id, "wrong",
      "the search prefers the record with as few fields as the menu item"
    );

    let output = engine.match_menu_item(menu_item(&MENU_ITEM)).await.unwrap();
    let ids: Vec<&str> = output
      .candidates
      .iter()
      .map(|candidate| candidate.id.as_str())
      .collect();
    assert_eq!(ids, ["right", "wrong", "vape"]);
    assert_eq!(
      (output.decision, output.threshold, output.fetched),
      (Decision::NotJudged, None, 3)
    );

    let best = &output.candidates[0];
    assert_eq!(best.brand_id.as_deref(), Some("brand-1"));
    let name = best
      .comparison
      .fields
      .iter()
      .find(|field| field.field == "name")
      .unwrap();
    assert_eq!(
      Some(best.comparison.similarity),
      name.similarity,
      "the name is all they share"
    );
    assert!(best.comparison.similarity > best.search_similarity + 0.1, "{best:?}");
    let changes: Vec<_> = best
      .comparison
      .fields
      .iter()
      .map(|field| (field.field.as_str(), field.change, field.below_floor))
      .collect();
    assert_eq!(
      changes,
      [
        ("category", FieldChange::Removed, false),
        ("description", FieldChange::Removed, false),
        ("name", FieldChange::Modified, false),
      ]
    );
    let searched_right = searched.iter().find(|neighbor| neighbor.id == "right").unwrap();
    assert_eq!(
      best.search_similarity, searched_right.similarity,
      "what `FindSimilar` reports for it"
    );
  }

  /// The over-fetch is what the exact score costs: only what the search returns can be scored. Fetching one
  /// record for the one that is wanted returns the search's favorite.
  #[tokio::test]
  async fn only_what_the_search_fetched_can_be_ranked() {
    let narrow = catalog(Matching {
      threshold: None,
      overfetch: 1,
    })
    .await;
    let one = MatchInput {
      limit: 1,
      ..menu_item(&MENU_ITEM)
    };
    let output = narrow.match_menu_item(one.clone()).await.unwrap();
    assert_eq!((output.candidates[0].id.as_str(), output.fetched), ("wrong", 1));

    let wide = catalog(Matching {
      threshold: None,
      overfetch: 2,
    })
    .await;
    let output = wide.match_menu_item(one).await.unwrap();
    assert_eq!(
      (
        output.candidates[0].id.as_str(),
        output.candidates.len(),
        output.fetched
      ),
      ("right", 1, 2)
    );

    // Zero means five, and the fetch is the limit times the over-fetch, as far as there are records.
    let output = wide.match_menu_item(menu_item(&MENU_ITEM)).await.unwrap();
    assert_eq!((output.candidates.len(), output.fetched), (3, 3));
  }

  #[tokio::test]
  async fn the_best_candidate_is_held_to_the_configured_threshold_or_to_the_requests() {
    let plain = catalog(Matching::default()).await;
    let score = plain.match_menu_item(menu_item(&MENU_ITEM)).await.unwrap().candidates[0]
      .comparison
      .similarity;
    let with = |threshold: Option<f32>| MatchInput {
      threshold,
      ..menu_item(&MENU_ITEM)
    };
    assert_eq!(
      plain.match_menu_item(with(Some(score))).await.unwrap().decision,
      Decision::Matched
    );
    let above = plain.match_menu_item(with(Some(score.next_up()))).await.unwrap();
    assert_eq!(
      (above.decision, above.threshold),
      (Decision::NoMatch, Some(score.next_up()))
    );
    assert_eq!(
      above.candidates.len(),
      3,
      "the candidates are returned whatever is decided"
    );

    let configured = catalog(Matching {
      threshold: Some(0.99),
      overfetch: 5,
    })
    .await;
    let output = configured.match_menu_item(with(None)).await.unwrap();
    assert_eq!((output.decision, output.threshold), (Decision::NoMatch, Some(0.99)));
    let output = configured.match_menu_item(with(Some(0.5))).await.unwrap();
    assert_eq!((output.decision, output.threshold), (Decision::Matched, Some(0.5)));
    assert_eq!(configured.describe().matching.threshold, Some(0.99));

    for outside in [1.5, -1.5, f32::NAN] {
      let error = plain.match_menu_item(with(Some(outside))).await.unwrap_err();
      assert!(matches!(error, EngineError::InvalidArgument(_)), "{error:?}");
    }
  }

  #[tokio::test]
  async fn a_match_can_be_limited_to_one_brand() {
    let engine = catalog(Matching::default()).await;
    let within = |brand_id: &str| MatchInput {
      brand_id: Some(brand_id.to_owned()),
      ..menu_item(&MENU_ITEM)
    };
    let ids =
      |output: MatchOutput| -> Vec<String> { output.candidates.into_iter().map(|candidate| candidate.id).collect() };
    assert_eq!(ids(engine.match_menu_item(within("brand-1")).await.unwrap()), ["right"]);
    assert_eq!(
      ids(engine.match_menu_item(within("brand-2")).await.unwrap()),
      ["wrong", "vape"]
    );
    let nobody = engine.match_menu_item(within("brand-3")).await.unwrap();
    assert_eq!(
      (nobody.candidates.len(), nobody.fetched, nobody.decision),
      (0, 0, Decision::NotJudged)
    );
    assert!(
      engine.match_menu_item(within("  ")).await.is_err(),
      "a blank brand is not no brand"
    );
  }

  /// A menu item's id and its seller are another system's. Taking them would invite a caller to confuse the
  /// retailer with the brand whose records are being searched.
  #[tokio::test]
  async fn a_menu_item_that_does_not_fit_is_refused() {
    let engine = catalog(Matching::default()).await;
    let refused = |input: MatchInput| {
      let engine = &engine;
      async move {
        let error = engine.match_menu_item(input).await.unwrap_err();
        assert!(matches!(error, EngineError::InvalidArgument(_)), "{error:?}");
        error.to_string()
      }
    };
    let mut with_id = menu_item(&MENU_ITEM);
    with_id.menu_item.id = Some("menu-item-7".to_owned());
    assert!(refused(with_id).await.contains("no `id` and no `brand_id`"));
    let mut with_brand = menu_item(&MENU_ITEM);
    with_brand.menu_item.brand_id = Some("retailer-1".to_owned());
    refused(with_brand).await;

    refused(menu_item(&[("name", "x"), ("strain", "indica")])).await;
    refused(menu_item(&[("category", "edibles")])).await;
    refused(MatchInput {
      limit: MAX_NEIGHBORS + 1,
      ..menu_item(&MENU_ITEM)
    })
    .await;
    let at_most = MatchInput {
      limit: MAX_NEIGHBORS,
      ..menu_item(&MENU_ITEM)
    };
    assert_eq!(engine.match_menu_item(at_most).await.unwrap().candidates.len(), 3);
  }

  #[tokio::test]
  async fn matching_against_an_empty_store_finds_nothing_and_writes_nothing() {
    let engine = engine().with_matching(Matching {
      threshold: Some(0.5),
      overfetch: 5,
    });
    let output = engine.match_menu_item(menu_item(&MENU_ITEM)).await.unwrap();
    assert_eq!(
      (output.candidates.len(), output.decision, output.fetched),
      (0, Decision::NoMatch, 0)
    );
    assert_eq!(stored(&engine, "right").await, None);
  }

  /// A match is a read. Every stored record is exactly what it was, quantities and timestamps included.
  #[tokio::test]
  async fn matching_never_changes_what_is_stored() {
    let engine = catalog(Matching {
      threshold: Some(0.1),
      overfetch: 5,
    })
    .await;
    let mut before = Vec::new();
    for id in ["right", "wrong", "vape"] {
      before.push(stored(&engine, id).await);
    }
    for input in [menu_item(&MENU_ITEM), menu_item(&RIGHT), menu_item(&VAPE)] {
      let output = engine.match_menu_item(input).await.unwrap();
      assert_eq!(output.decision, Decision::Matched);
    }
    let mut after = Vec::new();
    for id in ["right", "wrong", "vape"] {
      after.push(stored(&engine, id).await);
    }
    assert_eq!(before, after);

    // The product's own text finds the product, unchanged in every field.
    let own = engine.match_menu_item(menu_item(&RIGHT)).await.unwrap();
    assert_eq!(own.candidates[0].id, "right");
    assert_eq!(own.candidates[0].comparison.similarity, 1.0);
    assert!(
      own.candidates[0]
        .comparison
        .fields
        .iter()
        .all(|field| field.change == FieldChange::Unchanged)
    );
  }

  /// Quantities are compared for each candidate, with the stored record as the baseline, and change no score.
  #[tokio::test]
  async fn a_match_reports_each_candidates_quantities_and_scores_none_of_them() {
    let engine = engine_over(measured_schema((Some(1.5), true)), Policy::new(0.75, None).unwrap());
    engine.evaluate(measured(&[("weight", grams(1.0))])).await.unwrap();
    let ask = |quantities: Vec<(String, Quantity)>| MatchInput {
      menu_item: RecordInput {
        quantities,
        ..record(None, &GUMMIES)
      },
      ..MatchInput::default()
    };
    let plain = engine.match_menu_item(ask(Vec::new())).await.unwrap();
    let tenfold = engine
      .match_menu_item(ask(vec![("weight".to_owned(), grams(10.0))]))
      .await
      .unwrap();
    assert_eq!(tenfold.candidates[0].comparison, plain.candidates[0].comparison);
    assert_eq!(plain.candidates[0].quantities[0].change, QuantityChange::Removed);
    let weight = &tenfold.candidates[0].quantities[0];
    assert_eq!(
      (weight.change, weight.factor, weight.beyond_limit),
      (QuantityChange::Modified, Some(10.0), false)
    );

    let unknown = engine
      .match_menu_item(ask(vec![("potency".to_owned(), grams(1.0))]))
      .await;
    assert!(matches!(unknown, Err(EngineError::InvalidArgument(_))));
  }

  /// What an observer was told, kept: an [`Observation`] only borrows.
  #[derive(Debug, Clone, PartialEq)]
  struct Told {
    id: String,
    brand_id: Option<String>,
    mode: WriteMode,
    threshold: Option<f32>,
    judged_at: f32,
    stored: bool,
    revision: u64,
    /// The baseline's revision, the candidate's composite, its quantities, and whether it was accepted.
    judged: Option<(u64, Vec<f32>, Quantities, bool)>,
  }

  #[derive(Debug, Default)]
  struct Recorder(std::sync::Mutex<Vec<Told>>);

  impl Recorder {
    fn told(&self) -> Vec<Told> {
      self.0.lock().unwrap().clone()
    }
  }

  impl Observer for Recorder {
    fn observe(&self, observation: &Observation<'_>) {
      self.0.lock().unwrap().push(Told {
        id: observation.id.to_owned(),
        brand_id: observation.brand_id.map(str::to_owned),
        mode: observation.mode,
        threshold: observation.threshold,
        judged_at: observation.policy.threshold,
        stored: observation.stored,
        revision: observation.revision,
        judged: observation.judged.map(|judged| {
          (
            judged.baseline.revision,
            judged.candidate.composite.clone(),
            judged.quantities.clone(),
            judged.evaluation.accepted(),
          )
        }),
      });
    }
  }

  fn observed() -> (TestEngine, Arc<Recorder>) {
    let recorder = Arc::new(Recorder::default());
    let engine = engine_over(measured_schema((None, false)), Policy::new(0.75, None).unwrap());
    (engine.with_observer(recorder.clone()), recorder)
  }

  /// An observer is how a candidate policy watches live requests (`shadow.rs`). It is told about each request
  /// for an id exactly once, with what was judged and what came of it.
  #[tokio::test]
  async fn an_observer_is_told_what_was_judged_and_what_came_of_it() {
    let (engine, recorder) = observed();
    let weighed = |fields: &[(&str, &str)], mode: WriteMode, threshold: Option<f32>| EvaluateInput {
      record: RecordInput {
        brand_id: Some("brand-1".to_owned()),
        quantities: vec![("weight".to_owned(), grams(3.5))],
        ..record(Some("p1"), fields)
      },
      mode,
      threshold,
    };

    let created = engine
      .evaluate(weighed(&GUMMIES, WriteMode::Enforce, None))
      .await
      .unwrap();
    let edited = engine
      .evaluate(weighed(&GUMMIES_EDITED, WriteMode::Enforce, Some(0.5)))
      .await
      .unwrap();
    let recycled = engine.evaluate(weighed(&VAPE, WriteMode::Enforce, None)).await.unwrap();
    let tried = engine.evaluate(weighed(&VAPE, WriteMode::DryRun, None)).await.unwrap();
    let forced = engine.evaluate(weighed(&VAPE, WriteMode::Force, None)).await.unwrap();
    assert_eq!(
      [edited.verdict, recycled.verdict, tried.verdict, forced.verdict],
      [
        Verdict::Accepted,
        Verdict::Rejected,
        Verdict::Rejected,
        Verdict::Rejected
      ]
    );

    let weight: Quantities = [("weight".to_owned(), grams(3.5))].into_iter().collect();
    let told = |mode, threshold: Option<f32>, stored, revision, judged| Told {
      id: "p1".to_owned(),
      brand_id: Some("brand-1".to_owned()),
      mode,
      threshold,
      judged_at: threshold.unwrap_or(0.75),
      stored,
      revision,
      judged,
    };
    let judged = |baseline: u64, output: &EvaluateOutput, accepted: bool| {
      Some((baseline, output.vector.clone(), weight.clone(), accepted))
    };
    assert_eq!(
      recorder.told(),
      [
        told(WriteMode::Enforce, None, true, 1, None),
        told(WriteMode::Enforce, Some(0.5), true, 2, judged(1, &edited, true)),
        told(WriteMode::Enforce, None, false, 2, judged(2, &recycled, false)),
        told(WriteMode::DryRun, None, false, 2, judged(2, &tried, false)),
        told(WriteMode::Force, None, true, 3, judged(2, &forced, false)),
      ]
    );
    assert_eq!(created.revision, 1);

    // A new record that a dry run does not store is observed as not stored, at revision zero.
    engine
      .evaluate(EvaluateInput {
        mode: WriteMode::DryRun,
        ..evaluate_input(Some("p2"), &GUMMIES)
      })
      .await
      .unwrap();
    let last = recorder.told().pop().unwrap();
    assert_eq!(
      (last.id.as_str(), last.brand_id, last.stored, last.revision, last.judged),
      ("p2", None, false, 0, None),
      "and a request that names no brand is observed with none"
    );
  }

  /// A store that can be told to refuse every write.
  struct Refusing {
    inner: MemoryStore,
    refuse: std::sync::atomic::AtomicBool,
  }

  impl VectorStore for Refusing {
    async fn check(&self) -> Result<(), StoreError> {
      self.inner.check().await
    }

    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      self.inner.get(id).await
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      self.inner.put(record).await
    }

    async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
      if self.refuse.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(StoreError::Unavailable("refusing writes".to_owned()));
      }
      self.inner.put_if(record, expected).await
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.inner.delete(id).await
    }

    async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
      self.inner.delete_if(id, version).await
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.inner.search(query).await
    }

    async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
      self.inner.search_vectors(query).await
    }
  }

  /// Only an outcome is observed. A request without an id has no verdict, a request that is refused has none,
  /// and a write that failed is an error the caller got: none of them is something a candidate could have
  /// decided differently.
  #[tokio::test]
  async fn a_request_without_an_id_and_a_request_that_failed_are_never_observed() {
    let (engine, recorder) = observed();
    engine.evaluate(evaluate_input(None, &GUMMIES)).await.unwrap();
    assert!(engine.evaluate(evaluate_input(Some("p1"), &[])).await.is_err());
    let invalid = EvaluateInput {
      threshold: Some(7.0),
      ..evaluate_input(Some("p1"), &GUMMIES)
    };
    assert!(engine.evaluate(invalid).await.is_err());
    assert_eq!(recorder.told(), []);

    // A write that fails, for a new record and for a revision of a stored one.
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let store = Refusing {
      inner: MemoryStore::new(layout, dimension),
      refuse: std::sync::atomic::AtomicBool::new(false),
    };
    let recorder = Arc::new(Recorder::default());
    let engine =
      Engine::new(store, embedder, schema(), Policy::new(0.75, None).unwrap()).with_observer(recorder.clone());
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    assert_eq!(recorder.told().len(), 1);

    engine.store.refuse.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(engine.evaluate(evaluate_input(Some("p2"), &GUMMIES)).await.is_err());
    assert!(
      engine
        .evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))
        .await
        .is_err()
    );
    assert_eq!(
      recorder.told().len(),
      1,
      "a write that failed has no outcome to observe"
    );
    // A request that writes nothing succeeds on the same store, and is observed.
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    assert_eq!(recorder.told().len(), 2);
  }

  /// The observer is called under the record's lock, so what it is told about one id is in the order the engine
  /// judged it: the revision counter it sees never goes back.
  #[tokio::test]
  async fn an_observer_is_told_about_one_record_in_the_order_it_was_judged() {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let recorder = Arc::new(Recorder::default());
    let engine = Engine::new(
      Slow::over(MemoryStore::new(layout, dimension)),
      embedder,
      schema(),
      Policy::new(0.0, None).unwrap(),
    )
    .with_observer(recorder.clone());
    let request = |fields: &[(&str, &str)]| engine.evaluate(evaluate_input(Some("p1"), fields));
    let (a, b, c, d) = tokio::join!(
      request(&GUMMIES),
      request(&GUMMIES_EDITED),
      request(&VAPE),
      request(&GUMMIES)
    );
    for outcome in [a, b, c, d] {
      outcome.unwrap();
    }
    // Which request got the lock first is not fixed. Whichever it was, each one after it was judged against
    // exactly what the one before it left.
    let told = recorder.told();
    assert_eq!(told.len(), 4);
    assert_eq!((told[0].revision, &told[0].judged), (1, &None));
    for pair in told.windows(2) {
      let baseline = pair[1].judged.as_ref().map(|judged| judged.0);
      assert_eq!(baseline, Some(pair[0].revision), "{told:?}");
      assert!(pair[1].revision >= pair[0].revision, "{told:?}");
    }
    assert!(told.iter().filter(|told| told.stored).count() >= 2, "{told:?}");
  }

  /// A verdict is a read, a judgment, and a write. Two requests for one id that interleave would both read "no
  /// such record", and the second must still be judged against the first: a recycle must not get through the gap
  /// between two requests. The store's conditional write sees to that wherever the two requests are (D35). Within
  /// one process the lock sees to more: each id is handled by one request at a time, so neither request ever
  /// finds the record changed under it and has to read and judge twice.
  #[tokio::test]
  async fn two_requests_for_one_record_are_judged_one_after_the_other() {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let store = Slow::over(MemoryStore::new(layout, dimension));
    let engine = Engine::new(store, embedder, schema(), Policy::new(0.75, None).unwrap());

    let (first, second) = tokio::join!(
      engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)),
      engine.evaluate(evaluate_input(Some("p1"), &VAPE)),
    );
    let verdicts = [first.unwrap().verdict, second.unwrap().verdict];
    let created = verdicts.iter().filter(|verdict| **verdict == Verdict::Created).count();
    assert_eq!(
      created, 1,
      "one created it, and the other was judged against that: {verdicts:?}"
    );
    assert!(
      verdicts.contains(&Verdict::Rejected),
      "gummies and a vape are not one record: {verdicts:?}"
    );
    assert_eq!(
      engine.store.0.get("p1").await.unwrap().unwrap().revision,
      1,
      "the rejected one wrote nothing"
    );
    assert_eq!(
      engine.store.reads(),
      2,
      "each read the record once: neither lost a race"
    );

    // Requests for different ids do not wait on each other, and both get their answer.
    let (a, b) = tokio::join!(
      engine.evaluate(evaluate_input(Some("p2"), &GUMMIES)),
      engine.evaluate(evaluate_input(Some("p3"), &VAPE)),
    );
    assert_eq!(
      (a.unwrap().verdict, b.unwrap().verdict),
      (Verdict::Created, Verdict::Created)
    );
  }

  #[tokio::test]
  async fn a_stored_record_carries_the_time_it_was_written() {
    let clock = || SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let engine = engine();
    let before = clock();
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let written = stored(&engine, "p1").await.unwrap().updated_at;
    assert!(
      (before..=clock()).contains(&written),
      "{written} is not between {before} and now"
    );
  }

  #[tokio::test]
  async fn a_record_without_an_id_is_embedded_and_nothing_is_stored() {
    let engine = engine();
    let output = engine.evaluate(evaluate_input(None, &GUMMIES)).await.unwrap();
    assert_eq!(output.verdict, Verdict::Embedded);
    assert!(!output.stored);
    assert_eq!(output.vector.len(), engine.describe().composite_dimension);
    assert_eq!((output.similarity, output.revision), (None, 0));
    assert!(
      engine
        .find_similar(similar_to_record(&GUMMIES))
        .await
        .unwrap()
        .is_empty()
    );
  }

  fn similar_to_record(fields: &[(&str, &str)]) -> FindSimilarInput {
    FindSimilarInput {
      query: SimilarTo::Record(record(None, fields)),
      limit: 0,
      brand_id: None,
      min_similarity: None,
    }
  }

  fn similar_to_id(id: &str) -> FindSimilarInput {
    FindSimilarInput {
      query: SimilarTo::Id(id.to_owned()),
      limit: 0,
      brand_id: None,
      min_similarity: None,
    }
  }

  #[tokio::test]
  async fn the_first_record_for_an_id_becomes_the_baseline() {
    let engine = engine();
    let output = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    assert_eq!(output.verdict, Verdict::Created);
    assert!(output.stored);
    assert_eq!(output.revision, 1);

    let record = stored(&engine, "p1").await.unwrap();
    assert_eq!(record.revision, 1);
    assert_eq!(record.vectors.composite, output.vector);
    assert_eq!(record.anchor, output.vector, "a new record is its own anchor");
    assert!(record.updated_at > 0);
  }

  #[tokio::test]
  async fn an_ordinary_edit_is_accepted_and_becomes_the_new_baseline() {
    let engine = engine();
    let created = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let edited = engine
      .evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))
      .await
      .unwrap();

    assert_eq!(edited.verdict, Verdict::Accepted);
    assert!(edited.stored && edited.reasons.is_empty());
    assert_eq!(edited.revision, 2);
    let similarity = edited.similarity.unwrap();
    assert!((0.75..1.0).contains(&similarity), "{similarity}");

    let name = edited.fields.iter().find(|field| field.field == "name").unwrap();
    assert_eq!(name.change, FieldChange::Modified);
    let untouched = edited
      .fields
      .iter()
      .filter(|field| field.change == FieldChange::Unchanged)
      .count();
    assert_eq!(untouched, 2);

    let record = stored(&engine, "p1").await.unwrap();
    assert_eq!(record.vectors.composite, edited.vector);
    assert_eq!(
      record.anchor, created.vector,
      "an accepted edit moves the baseline, never the anchor"
    );
  }

  #[tokio::test]
  async fn a_recycled_record_is_rejected_and_the_baseline_is_untouched() {
    let engine = engine();
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let before = stored(&engine, "p1").await.unwrap();

    let recycled = engine.evaluate(evaluate_input(Some("p1"), &VAPE)).await.unwrap();
    assert_eq!(recycled.verdict, Verdict::Rejected);
    assert!(!recycled.stored);
    assert!(recycled.similarity.unwrap() < 0.3, "{:?}", recycled.similarity);
    assert!(
      recycled
        .reasons
        .iter()
        .any(|reason| matches!(reason, Reason::BelowThreshold { .. })),
      "{:?}",
      recycled.reasons
    );
    assert!(
      recycled
        .reasons
        .iter()
        .any(|reason| matches!(reason, Reason::FieldBelowFloor { field, .. } if field == "name")),
      "{:?}",
      recycled.reasons
    );
    assert_eq!(recycled.revision, 1, "a rejected revision does not bump the counter");
    assert_eq!(stored(&engine, "p1").await.unwrap(), before);

    // And the original still compares as itself afterwards.
    let retry = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    assert_eq!(retry.verdict, Verdict::Accepted);
    assert!((retry.similarity.unwrap() - 1.0).abs() < f32::EPSILON);
  }

  #[tokio::test]
  async fn resubmitting_the_stored_record_touches_neither_the_model_nor_the_store() {
    let engine = engine();
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let embedded_after_create = engine.embedder.embedded();
    assert_eq!(embedded_after_create, 3);

    // Same text, different whitespace: still the same record.
    let respaced = [
      ("name", "  Blueberry   Gummies "),
      ("category", "Edibles Gummies"),
      ("description", "Ten vegan gummies\nwith real fruit"),
    ];
    let output = engine.evaluate(evaluate_input(Some("p1"), &respaced)).await.unwrap();
    assert_eq!(output.verdict, Verdict::Accepted);
    assert!(!output.stored);
    assert_eq!(output.revision, 1);
    assert_eq!(
      engine.embedder.embedded(),
      embedded_after_create,
      "nothing changed, so nothing was embedded"
    );
  }

  #[tokio::test]
  async fn only_the_changed_field_is_embedded() {
    let engine = engine();
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    engine
      .evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))
      .await
      .unwrap();
    assert_eq!(engine.embedder.embedded(), 3 + 1);
  }

  #[tokio::test]
  async fn a_dry_run_reports_the_verdict_and_never_writes() {
    let engine = engine();
    let preview = EvaluateInput {
      mode: WriteMode::DryRun,
      ..evaluate_input(Some("p1"), &GUMMIES)
    };
    let output = engine.evaluate(preview).await.unwrap();
    assert_eq!(
      (output.verdict, output.stored, output.revision),
      (Verdict::Created, false, 0)
    );
    assert_eq!(stored(&engine, "p1").await, None);

    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let before = stored(&engine, "p1").await.unwrap();
    for fields in [&GUMMIES_EDITED, &VAPE] {
      let preview = EvaluateInput {
        mode: WriteMode::DryRun,
        ..evaluate_input(Some("p1"), fields)
      };
      assert!(!engine.evaluate(preview).await.unwrap().stored);
    }
    assert_eq!(stored(&engine, "p1").await.unwrap(), before);
  }

  #[tokio::test]
  async fn force_writes_a_rejected_record_and_resets_its_anchor() {
    let engine = engine();
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();

    let forced = EvaluateInput {
      mode: WriteMode::Force,
      ..evaluate_input(Some("p1"), &VAPE)
    };
    let output = engine.evaluate(forced).await.unwrap();
    assert_eq!(
      output.verdict,
      Verdict::Rejected,
      "the verdict is still reported honestly"
    );
    assert!(output.stored);
    assert_eq!(output.revision, 2);

    let record = stored(&engine, "p1").await.unwrap();
    assert_eq!(record.vectors.composite, output.vector);
    assert_eq!(
      record.anchor, output.vector,
      "a reviewer's override is the new point of reference"
    );
  }

  #[tokio::test]
  async fn a_field_floor_rejects_a_renamed_record_whose_other_fields_are_intact() {
    // Without the floor this passes on the strength of the unchanged fields.
    let engine = engine_with(Policy::new(0.4, None).unwrap());
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let renamed = [("name", "Sour Diesel Vape Cartridge"), GUMMIES[1], GUMMIES[2]];
    let output = engine.evaluate(evaluate_input(Some("p1"), &renamed)).await.unwrap();

    assert!(output.similarity.unwrap() >= 0.4, "{:?}", output.similarity);
    assert_eq!(output.verdict, Verdict::Rejected);
    assert_eq!(output.reasons.len(), 1);
    assert!(
      matches!(&output.reasons[0], Reason::FieldBelowFloor { field, .. } if field == "name"),
      "{:?}",
      output.reasons
    );
    assert!(
      output
        .fields
        .iter()
        .any(|field| field.field == "name" && field.below_floor)
    );
  }

  #[tokio::test]
  async fn a_request_threshold_overrides_the_configured_one() {
    let engine = engine();
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();

    let strict = EvaluateInput {
      threshold: Some(0.99),
      mode: WriteMode::DryRun,
      ..evaluate_input(Some("p1"), &GUMMIES_EDITED)
    };
    let output = engine.evaluate(strict).await.unwrap();
    assert_eq!(output.verdict, Verdict::Rejected);
    assert!((output.threshold - 0.99).abs() < f32::EPSILON);

    let nonsense = EvaluateInput {
      threshold: Some(7.0),
      ..evaluate_input(Some("p1"), &GUMMIES_EDITED)
    };
    assert!(matches!(
      engine.evaluate(nonsense).await,
      Err(EngineError::InvalidArgument(_))
    ));
  }

  /// Six names, each sharing most of its words with the one before and none with the first by the end.
  const DRIFT: [&str; 6] = [
    "blueberry gummies vegan fruit chews",
    "blueberry gummies vegan fruit cartridge",
    "blueberry gummies vegan vape cartridge",
    "blueberry gummies resin vape cartridge",
    "blueberry diesel resin vape cartridge",
    "sour diesel resin vape cartridge",
  ];

  #[tokio::test]
  async fn small_steps_walk_a_record_anywhere_unless_drift_is_enforced() {
    let lenient = engine_with(Policy::new(0.75, None).unwrap());
    let mut drift = Vec::new();
    for name in DRIFT {
      let output = lenient
        .evaluate(evaluate_input(Some("p1"), &[("name", name)]))
        .await
        .unwrap();
      assert_ne!(
        output.verdict,
        Verdict::Rejected,
        "each step alone looks like an edit: {name}"
      );
      drift.extend(output.anchor_similarity);
    }
    assert!(
      drift.windows(2).all(|pair| pair[1] < pair[0]),
      "drift is visible in every response: {drift:?}"
    );
    assert!(
      drift.last().unwrap().abs() < 0.05,
      "and by the end nothing of the original is left: {drift:?}"
    );

    let strict = engine_with(Policy::new(0.75, Some(0.5)).unwrap());
    let mut verdicts = Vec::new();
    for name in DRIFT {
      verdicts.push(
        strict
          .evaluate(evaluate_input(Some("p1"), &[("name", name)]))
          .await
          .unwrap(),
      );
    }
    let rejected = verdicts
      .iter()
      .position(|output| output.verdict == Verdict::Rejected)
      .unwrap();
    assert!(
      verdicts[rejected]
        .reasons
        .iter()
        .any(|reason| matches!(reason, Reason::AnchorDrift { .. }))
    );
    assert!(
      verdicts[rejected].similarity.unwrap() >= 0.75,
      "the step itself was small"
    );
  }

  #[tokio::test]
  async fn a_record_can_gain_a_brand_and_from_then_on_every_request_must_name_it() {
    let engine = engine();
    let with_brand = |brand_id: Option<&str>, fields: &[(&str, &str)]| EvaluateInput {
      record: RecordInput {
        brand_id: brand_id.map(str::to_owned),
        ..record(Some("p1"), fields)
      },
      ..EvaluateInput::default()
    };

    engine.evaluate(with_brand(None, &GUMMIES)).await.unwrap();
    let adopted = engine.evaluate(with_brand(Some("brand-1"), &GUMMIES)).await.unwrap();
    assert!(adopted.stored, "gaining a brand is a change worth writing");
    assert_eq!(
      stored(&engine, "p1").await.unwrap().brand_id.as_deref(),
      Some("brand-1")
    );

    let stolen = engine.evaluate(with_brand(Some("brand-2"), &GUMMIES)).await;
    assert!(matches!(stolen, Err(EngineError::FailedPrecondition(message)) if message.contains("brand-1")));

    // Leaving the brand out is not a way around the check, in any mode.
    let before = stored(&engine, "p1").await.unwrap();
    for mode in [WriteMode::Enforce, WriteMode::DryRun, WriteMode::Force] {
      let unnamed = EvaluateInput {
        mode,
        ..with_brand(None, &VAPE)
      };
      let refused = engine.evaluate(unnamed).await;
      assert!(
        matches!(&refused, Err(EngineError::FailedPrecondition(message)) if message.contains("names no brand")),
        "{mode:?}: {refused:?}"
      );
    }
    assert_eq!(stored(&engine, "p1").await.unwrap(), before);

    let by_its_brand = engine
      .evaluate(with_brand(Some("brand-1"), &GUMMIES_EDITED))
      .await
      .unwrap();
    assert_eq!(by_its_brand.verdict, Verdict::Accepted);
  }

  #[tokio::test]
  async fn deleting_a_record_that_has_a_brand_takes_that_brand() {
    let engine = engine();
    let owned = EvaluateInput {
      record: RecordInput {
        brand_id: Some("brand-1".to_owned()),
        ..record(Some("p1"), &GUMMIES)
      },
      ..EvaluateInput::default()
    };
    engine.evaluate(owned).await.unwrap();

    for brand_id in [None, Some("brand-2".to_owned())] {
      let refused = engine.delete("p1".to_owned(), brand_id).await;
      assert!(
        matches!(refused, Err(EngineError::FailedPrecondition(_))),
        "{refused:?}"
      );
    }
    assert!(stored(&engine, "p1").await.is_some());
    assert!(
      engine
        .delete("p1".to_owned(), Some("brand-1".to_owned()))
        .await
        .unwrap()
    );
    assert!(stored(&engine, "p1").await.is_none());
  }

  #[tokio::test]
  async fn a_field_cannot_be_swapped_by_blanking_it_and_filling_it_in_again() {
    let engine = engine();
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let change = |output: &EvaluateOutput, name: &str| {
      let field = output.fields.iter().find(|field| field.field == name).unwrap();
      (field.change, field.similarity)
    };

    // Step one: the category is blanked. Nothing else changes, so this is accepted, as it should be.
    let without_category = [GUMMIES[0], GUMMIES[2]];
    let blanked = engine
      .evaluate(evaluate_input(Some("p1"), &without_category))
      .await
      .unwrap();
    assert_eq!(blanked.verdict, Verdict::Accepted);
    assert_eq!(change(&blanked, "category"), (FieldChange::Removed, None));

    // Step two: another product's category arrives. It is scored against the category the record started with,
    // exactly as if it had been changed in one step: (4 + 2*0 + 1) / 7.
    let swapped = [GUMMIES[0], ("category", "Vape Pens"), GUMMIES[2]];
    let refilled = engine.evaluate(evaluate_input(Some("p1"), &swapped)).await.unwrap();
    assert_eq!(refilled.verdict, Verdict::Rejected);
    assert_eq!(change(&refilled, "category").0, FieldChange::Restored);
    assert!(change(&refilled, "category").1.unwrap() < 0.01);
    assert!((refilled.similarity.unwrap() - 5.0 / 7.0).abs() < 1e-4);

    // Putting the original category back is not a change at all.
    let restored = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    assert_eq!(restored.verdict, Verdict::Accepted);
    assert_eq!(change(&restored, "category").0, FieldChange::Restored);
    assert!((restored.similarity.unwrap() - 1.0).abs() < 1e-5);
  }

  #[tokio::test]
  async fn a_field_added_later_is_remembered_from_its_first_value() {
    let engine = engine();
    let name = GUMMIES[0];
    engine.evaluate(evaluate_input(Some("p1"), &[name])).await.unwrap();
    let anchor_at_creation = stored(&engine, "p1").await.unwrap().anchor;

    // A description the record never had is new information, taken on trust, and becomes part of the anchor.
    let described = engine
      .evaluate(evaluate_input(Some("p1"), &[name, GUMMIES[2]]))
      .await
      .unwrap();
    assert_eq!(described.verdict, Verdict::Accepted);
    let field = described
      .fields
      .iter()
      .find(|field| field.field == "description")
      .unwrap();
    assert_eq!((field.change, field.similarity), (FieldChange::Added, None));
    let record = stored(&engine, "p1").await.unwrap();
    assert_ne!(record.anchor, anchor_at_creation, "the anchor gained the description");
    assert_eq!(
      record.anchor, record.vectors.composite,
      "and holds exactly the first value of each field"
    );

    // So blanking it and writing a different one is caught the same way as for an original field.
    engine.evaluate(evaluate_input(Some("p1"), &[name])).await.unwrap();
    let other = [name, ("description", "One gram of live resin")];
    let refilled = engine.evaluate(evaluate_input(Some("p1"), &other)).await.unwrap();
    let field = refilled
      .fields
      .iter()
      .find(|field| field.field == "description")
      .unwrap();
    assert_eq!(field.change, FieldChange::Restored);
    assert!(field.similarity.unwrap() < 0.01, "{field:?}");
    // (4 + 1*0) / 5: the name carries it over this engine's 0.75 threshold, and the score says what happened.
    assert!((refilled.similarity.unwrap() - 0.8).abs() < 1e-4);
  }

  #[tokio::test]
  async fn a_change_in_a_quantity_alone_is_stored_and_bumps_the_revision() {
    let engine = measuring_engine();
    let created = engine
      .evaluate(measured(&[("weight", grams(3.5)), ("thc", percent(18.0, 22.0))]))
      .await
      .unwrap();
    assert_eq!((created.verdict, created.revision), (Verdict::Created, 1));
    assert!(created.quantities.is_empty(), "nothing to compare a new record with");
    let record = stored(&engine, "p1").await.unwrap();
    assert_eq!(record.quantities["weight"], grams(3.5));
    assert_eq!(
      record.anchor_quantities, record.quantities,
      "a new record is its own anchor"
    );

    // Eight times the weight and not a word of text changed. Report-only, so it is accepted; and written.
    let resized = engine
      .evaluate(measured(&[("weight", grams(28.0)), ("thc", percent(18.0, 22.0))]))
      .await
      .unwrap();
    assert_eq!(resized.verdict, Verdict::Accepted);
    assert!(resized.stored && resized.reasons.is_empty());
    assert_eq!(resized.revision, 2);
    assert!((resized.similarity.unwrap() - 1.0).abs() < f32::EPSILON);
    let weight = quantity(&resized, "weight");
    assert_eq!(weight.change, QuantityChange::Modified);
    assert!((weight.factor.unwrap() - 8.0).abs() < 1e-9);
    assert!(!weight.beyond_limit && !resized.measure_changed);
    assert_eq!(quantity(&resized, "thc").change, QuantityChange::Unchanged);

    let record = stored(&engine, "p1").await.unwrap();
    assert_eq!(record.quantities["weight"], grams(28.0), "the baseline moved");
    assert_eq!(
      record.anchor_quantities["weight"],
      grams(3.5),
      "the anchor keeps the first value"
    );
    assert_eq!(engine.embedder.embedded(), 3, "and no text was embedded for it");
  }

  #[tokio::test]
  async fn resubmitting_the_same_quantities_is_not_a_write_even_in_other_units() {
    let engine = measuring_engine();
    let original = [("weight", grams(1.0)), ("thc", percent(18.0, 22.0))];
    engine.evaluate(measured(&original)).await.unwrap();
    let before = stored(&engine, "p1").await.unwrap();

    for same in [
      original.clone(),
      [("weight", Quantity::single(1000.0, "mg")), ("thc", percent(18.0, 22.0))],
      [
        ("weight", Quantity::single(1.0, " G ")),
        ("thc", Quantity::range(18.0, 22.0, "percent")),
      ],
    ] {
      let output = engine.evaluate(measured(&same)).await.unwrap();
      assert_eq!(output.verdict, Verdict::Accepted);
      assert!(!output.stored, "{same:?}");
      assert_eq!(output.revision, 1);
      assert_eq!(quantity(&output, "weight").change, QuantityChange::Unchanged);
      assert_eq!(quantity(&output, "weight").factor, Some(1.0));
    }
    assert_eq!(stored(&engine, "p1").await.unwrap(), before);
  }

  #[tokio::test]
  async fn a_quantity_cannot_be_swapped_by_removing_it_and_filling_it_in_again() {
    let engine = measuring_engine();
    engine.evaluate(measured(&[("weight", grams(3.5))])).await.unwrap();

    // Step one: the weight goes away. Reported, accepted, written.
    let removed = engine.evaluate(measured(&[])).await.unwrap();
    assert_eq!(quantity(&removed, "weight").change, QuantityChange::Removed);
    assert!(removed.stored && !removed.measure_changed);
    let record = stored(&engine, "p1").await.unwrap();
    assert!(record.quantities.is_empty());
    assert_eq!(
      record.anchor_quantities["weight"],
      grams(3.5),
      "the anchor remembers what it was"
    );

    // Step two: another weight arrives. It is compared with the weight the record started with, exactly as if it
    // had been changed in one step.
    let refilled = engine.evaluate(measured(&[("weight", grams(28.0))])).await.unwrap();
    let weight = quantity(&refilled, "weight");
    assert_eq!(weight.change, QuantityChange::Restored);
    assert!((weight.factor.unwrap() - 8.0).abs() < 1e-9);
    assert_eq!(weight.previous_unit.as_deref(), Some("g"));
    assert_eq!(
      stored(&engine, "p1").await.unwrap().anchor_quantities["weight"],
      grams(3.5),
      "an accepted write never replaces a value the anchor holds"
    );

    // With a limit, the two-step swap is rejected at the second step, as the one-step swap would be.
    let strict = engine_over(measured_schema((Some(4.0), false)), Policy::new(0.75, None).unwrap());
    strict.evaluate(measured(&[("weight", grams(3.5))])).await.unwrap();
    assert_eq!(strict.evaluate(measured(&[])).await.unwrap().verdict, Verdict::Accepted);
    let before = stored(&strict, "p1").await.unwrap();
    let refilled = strict.evaluate(measured(&[("weight", grams(28.0))])).await.unwrap();
    assert_eq!(refilled.verdict, Verdict::Rejected);
    assert!(quantity(&refilled, "weight").beyond_limit);
    assert_eq!(stored(&strict, "p1").await.unwrap(), before);
    // Putting the original weight back is not a change at all.
    let restored = strict.evaluate(measured(&[("weight", grams(3.5))])).await.unwrap();
    assert_eq!(restored.verdict, Verdict::Accepted);
    assert_eq!(quantity(&restored, "weight").factor, Some(1.0));
  }

  #[tokio::test]
  async fn a_quantity_beyond_its_limit_is_rejected_and_nothing_is_written() {
    let engine = engine_over(measured_schema((Some(4.0), true)), Policy::new(0.75, None).unwrap());
    engine
      .evaluate(measured(&[("weight", grams(3.5)), ("thc", percent(18.0, 22.0))]))
      .await
      .unwrap();
    let before = stored(&engine, "p1").await.unwrap();

    let resized = engine
      .evaluate(measured(&[("weight", grams(28.0)), ("thc", percent(18.0, 22.0))]))
      .await
      .unwrap();
    assert_eq!(resized.verdict, Verdict::Rejected);
    assert!(!resized.stored);
    assert_eq!(resized.revision, 1);
    assert!(
      (resized.similarity.unwrap() - 1.0).abs() < f32::EPSILON,
      "the text is identical"
    );
    assert_eq!(resized.reasons.len(), 1);
    assert!(
      matches!(
        &resized.reasons[0],
        Reason::QuantityBeyondFactor { field, factor: Some(factor), .. } if field == "weight" && (factor - 8.0).abs() < 1e-9
      ),
      "{:?}",
      resized.reasons
    );
    assert!(quantity(&resized, "weight").beyond_limit);
    assert!(!quantity(&resized, "thc").beyond_limit);

    // A potency in milligrams is not a bigger or smaller percentage.
    let other_kind = engine
      .evaluate(measured(&[
        ("weight", grams(3.5)),
        ("thc", Quantity::single(100.0, "mg")),
      ]))
      .await
      .unwrap();
    assert_eq!(other_kind.verdict, Verdict::Rejected);
    assert_eq!(quantity(&other_kind, "thc").change, QuantityChange::Incomparable);
    assert!(
      matches!(
        &other_kind.reasons[0],
        Reason::QuantityIncomparable { field, previous_unit, unit } if field == "thc" && previous_unit == "%" && unit == "mg"
      ),
      "{:?}",
      other_kind.reasons
    );
    assert!(other_kind.measure_changed);

    assert_eq!(stored(&engine, "p1").await.unwrap(), before, "a rejection never writes");

    // Within the limits, an edit goes through: a routine batch update.
    let batch = engine
      .evaluate(measured(&[("weight", grams(3.5)), ("thc", percent(19.0, 23.0))]))
      .await
      .unwrap();
    assert_eq!(batch.verdict, Verdict::Accepted);
    assert!(batch.stored);
  }

  #[tokio::test]
  async fn a_change_of_measure_is_reported_and_rejects_only_under_a_policy_that_says_so() {
    let flower = [("weight", grams(3.5)), ("thc", percent(18.0, 22.0))];
    let beverage = [("volume", Quantity::single(355.0, "ml")), ("thc", percent(18.0, 22.0))];

    let lenient = measuring_engine();
    lenient.evaluate(measured(&flower)).await.unwrap();
    // Filling in a quantity the record never had is not a change of measure.
    let filled = lenient
      .evaluate(measured(&[flower[0].clone(), flower[1].clone(), beverage[0].clone()]))
      .await
      .unwrap();
    assert_eq!(quantity(&filled, "volume").change, QuantityChange::Added);
    assert!(!filled.measure_changed);

    let lenient = measuring_engine();
    lenient.evaluate(measured(&flower)).await.unwrap();
    let traded = lenient.evaluate(measured(&beverage)).await.unwrap();
    assert!(traded.measure_changed);
    assert_eq!(traded.verdict, Verdict::Accepted, "reported, not enforced");
    assert!(traded.stored);

    let mut policy = Policy::new(0.75, None).unwrap();
    policy.reject_on_measure_change = true;
    let strict = engine_over(measured_schema((None, false)), policy);
    strict.evaluate(measured(&flower)).await.unwrap();
    let before = stored(&strict, "p1").await.unwrap();
    let traded = strict.evaluate(measured(&beverage)).await.unwrap();
    assert_eq!(traded.verdict, Verdict::Rejected);
    assert!(traded.measure_changed && !traded.stored);
    assert_eq!(traded.reasons, [Reason::MeasureChanged]);
    assert_eq!(stored(&strict, "p1").await.unwrap(), before);

    // A request that overrides the threshold is still held to the rest of the policy.
    let overridden = EvaluateInput {
      threshold: Some(0.1),
      ..measured(&beverage)
    };
    assert_eq!(strict.evaluate(overridden).await.unwrap().verdict, Verdict::Rejected);
  }

  #[tokio::test]
  async fn force_resets_the_anchors_quantities_and_a_dry_run_writes_none() {
    let engine = engine_over(measured_schema((Some(4.0), false)), Policy::new(0.75, None).unwrap());
    engine.evaluate(measured(&[("weight", grams(3.5))])).await.unwrap();
    let before = stored(&engine, "p1").await.unwrap();

    let preview = EvaluateInput {
      mode: WriteMode::DryRun,
      ..measured(&[("weight", grams(7.0))])
    };
    let previewed = engine.evaluate(preview).await.unwrap();
    assert_eq!((previewed.verdict, previewed.stored), (Verdict::Accepted, false));
    assert_eq!(stored(&engine, "p1").await.unwrap(), before);

    let forced = EvaluateInput {
      mode: WriteMode::Force,
      ..measured(&[("weight", grams(28.0))])
    };
    let output = engine.evaluate(forced).await.unwrap();
    assert_eq!(
      (output.verdict, output.stored),
      (Verdict::Rejected, true),
      "the verdict is still reported honestly"
    );
    let record = stored(&engine, "p1").await.unwrap();
    assert_eq!(record.quantities["weight"], grams(28.0));
    assert_eq!(
      record.anchor_quantities["weight"],
      grams(28.0),
      "a reviewer's override is the new point of reference"
    );

    // So a later blank-and-refill is measured from the forced value, not from the one before it.
    engine.evaluate(measured(&[])).await.unwrap();
    let refilled = engine.evaluate(measured(&[("weight", grams(14.0))])).await.unwrap();
    assert_eq!(quantity(&refilled, "weight").change, QuantityChange::Restored);
    assert!((quantity(&refilled, "weight").factor.unwrap() - 2.0).abs() < 1e-9);
  }

  #[tokio::test]
  async fn a_record_stored_before_quantities_existed_reports_them_as_added() {
    let old = engine();
    old.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    assert!(stored(&old, "p1").await.unwrap().quantities.is_empty());

    // The same store under a schema that has gained quantities: the layout id is the same, so nothing needs
    // re-indexing and the record is readable as it is.
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let schema = measured_schema((Some(4.0), true));
    let (layout, dimension) = store_layout(&schema, &embedder);
    assert_eq!(layout, store_layout(&self::schema(), &embedder).0);
    let upgraded = Engine::new(
      Arc::into_inner(old.store).unwrap().reopened(layout, dimension),
      embedder,
      schema,
      old.policy,
    );

    let output = upgraded.evaluate(measured(&[("weight", grams(3.5))])).await.unwrap();
    assert_eq!(output.verdict, Verdict::Accepted, "{:?}", output.reasons);
    assert_eq!(quantity(&output, "weight").change, QuantityChange::Added);
    assert!(output.stored && !output.measure_changed);
    let record = stored(&upgraded, "p1").await.unwrap();
    assert_eq!(record.quantities["weight"], grams(3.5));
    assert_eq!(
      record.anchor_quantities["weight"],
      grams(3.5),
      "the anchor takes the first value"
    );
  }

  #[tokio::test]
  async fn quantities_are_validated_wherever_a_record_is_given() {
    let engine = measuring_engine();
    let invalid = [
      measured(&[("height", grams(1.0))]),
      measured(&[("weight", grams(-1.0))]),
      measured(&[("weight", grams(f64::NAN))]),
      measured(&[("weight", Quantity::range(5.0, 1.0, "g"))]),
      measured(&[("weight", Quantity::single(1.0, " "))]),
      measured(&[("name", grams(1.0))]),
      measured(&[("weight", grams(1.0)), ("weight", grams(2.0))]),
    ];
    for input in invalid {
      let shown = format!("{input:?}");
      assert!(
        matches!(engine.evaluate(input).await, Err(EngineError::InvalidArgument(_))),
        "{shown}"
      );
    }
    assert_eq!(stored(&engine, "p1").await, None);
    assert_eq!(engine.embedder.embedded(), 0, "invalid records never reach the model");

    // An engine without quantities refuses them rather than dropping them silently.
    let refused = self::engine().evaluate(measured(&[("weight", grams(1.0))])).await;
    assert!(matches!(refused, Err(EngineError::InvalidArgument(message)) if message.contains("no quantities")));

    // A search validates the record it is given, and then ignores the quantities: it compares vectors.
    let mut search = similar_to_record(&GUMMIES);
    search.query = SimilarTo::Record(measured(&[("height", grams(1.0))]).record);
    assert!(matches!(
      engine.find_similar(search).await,
      Err(EngineError::InvalidArgument(_))
    ));
    engine.evaluate(measured(&[("weight", grams(3.5))])).await.unwrap();
    let mut search = similar_to_record(&GUMMIES);
    search.query = SimilarTo::Record(RecordInput {
      id: None,
      ..measured(&[("weight", grams(500.0))]).record
    });
    let neighbors = engine.find_similar(search).await.unwrap();
    assert!((neighbors[0].similarity - 1.0).abs() < 1e-5);
  }

  fn import_of(id: &str, fields: &[(&str, &str)], anchor: Option<&[(&str, &str)]>) -> ImportInput {
    let pairs = |fields: &[(&str, &str)]| -> Vec<(String, String)> {
      fields
        .iter()
        .map(|(name, text)| ((*name).to_owned(), (*text).to_owned()))
        .collect()
    };
    ImportInput {
      record: record(Some(id), fields),
      anchor: anchor.map(|fields| Original {
        fields: pairs(fields),
        quantities: Vec::new(),
      }),
    }
  }

  #[tokio::test]
  async fn an_import_creates_what_is_not_stored_and_leaves_what_is() {
    let engine = engine();
    engine.evaluate(evaluate_input(Some("kept"), &GUMMIES)).await.unwrap();
    let before = stored(&engine, "kept").await.unwrap();

    let outcomes = engine
      .import(vec![
        import_of("new-1", &GUMMIES, None),
        import_of("kept", &VAPE, None),
        import_of("new-2", &VAPE, None),
        // The same id twice in one import: the first one wins, as it would across two imports.
        import_of("new-1", &VAPE, None),
      ])
      .await
      .unwrap();
    assert_eq!(
      outcomes,
      [
        Ok(Imported::Created),
        Ok(Imported::Skipped),
        Ok(Imported::Created),
        Ok(Imported::Skipped)
      ]
    );
    assert_eq!(
      stored(&engine, "kept").await.unwrap(),
      before,
      "an import never updates: not even the timestamp moved"
    );

    let created = stored(&engine, "new-1").await.unwrap();
    assert_eq!(created.revision, 1);
    assert_eq!(
      created.anchor, created.vectors.composite,
      "with no original, a record is its own anchor"
    );
    // It is an ordinary baseline from here on.
    let edited = engine
      .evaluate(evaluate_input(Some("new-1"), &GUMMIES_EDITED))
      .await
      .unwrap();
    assert_eq!((edited.verdict, edited.revision), (Verdict::Accepted, 2));

    // Running the same import again changes nothing at all.
    let again = engine.import(vec![import_of("new-2", &VAPE, None)]).await.unwrap();
    assert_eq!(again, [Ok(Imported::Skipped)]);
  }

  #[tokio::test]
  async fn an_imported_original_version_is_the_anchor_drift_is_measured_from() {
    let engine = engine_with(Policy::new(0.75, Some(0.5)).unwrap());
    // What the record says today, and what it said when it was created, several edits ago.
    let today = [("name", DRIFT[3]), ("category", "Vape Pens")];
    let created_as = [("name", DRIFT[0])];
    let outcomes = engine
      .import(vec![import_of("p1", &today, Some(&created_as))])
      .await
      .unwrap();
    assert_eq!(outcomes, [Ok(Imported::Created)]);

    let record = stored(&engine, "p1").await.unwrap();
    assert_ne!(
      record.anchor, record.vectors.composite,
      "the anchor is not today's record"
    );
    let schema = schema();
    let anchor = schema.blocks(64, &record.anchor).unwrap();
    let current = schema.blocks(64, &record.vectors.composite).unwrap();
    // Schema order: category, description, name.
    assert!(
      similarity_core::vector::cosine(anchor[0].unwrap(), current[0].unwrap()).unwrap() > 0.9999,
      "a field the original lacked takes today's value, as it would have when it was first filled in"
    );
    assert!(anchor[1].is_none());
    assert!(
      similarity_core::vector::cosine(anchor[2].unwrap(), current[2].unwrap()).unwrap() < 0.7,
      "the name is the original name"
    );

    // Resubmitting today's record: nothing changed since the import, and it is already two steps from where it
    // began.
    let same = engine.evaluate(evaluate_input(Some("p1"), &today)).await.unwrap();
    assert!((same.similarity.unwrap() - 1.0).abs() < 1e-5);
    assert!(!same.stored);
    let drift = same.anchor_similarity.unwrap();
    assert!(drift < 0.9, "drift history survived the import: {drift}");

    // One more step in the same direction is small against the baseline and too far from the original. Imported
    // without the original, the same edit would have been measured from today and accepted.
    let next = [("name", DRIFT[5]), ("category", "Vape Pens")];
    let step = engine.evaluate(evaluate_input(Some("p1"), &next)).await.unwrap();
    assert!(
      step
        .reasons
        .iter()
        .any(|reason| matches!(reason, Reason::AnchorDrift { .. })),
      "{:?}",
      step.reasons
    );

    let forgetful = engine_with(Policy::new(0.75, Some(0.5)).unwrap());
    forgetful.import(vec![import_of("p1", &today, None)]).await.unwrap();
    let step = forgetful.evaluate(evaluate_input(Some("p1"), &next)).await.unwrap();
    assert!(
      !step
        .reasons
        .iter()
        .any(|reason| matches!(reason, Reason::AnchorDrift { .. })),
      "{:?}",
      step.reasons
    );
  }

  #[tokio::test]
  async fn an_imported_original_guards_fields_and_quantities_the_record_no_longer_has() {
    let engine = measuring_engine();
    let mut input = import_of("p1", &[GUMMIES[0]], Some(&[GUMMIES[0], GUMMIES[1]]));
    input.record.quantities = vec![("thc".to_owned(), percent(18.0, 22.0))];
    input.anchor.as_mut().unwrap().quantities = vec![("weight".to_owned(), grams(3.5))];
    assert_eq!(engine.import(vec![input]).await.unwrap(), [Ok(Imported::Created)]);

    let record = stored(&engine, "p1").await.unwrap();
    assert_eq!(record.quantities.keys().collect::<Vec<_>>(), ["thc"]);
    assert_eq!(
      record.anchor_quantities.keys().collect::<Vec<_>>(),
      ["thc", "weight"],
      "the original's weight, and the potency the original never had"
    );

    // The category and the weight were removed at some point before the import. Bringing them back as something
    // else is the two-step swap, and is judged against the original (D8, D16).
    let mut refill = measured(&[("thc", percent(18.0, 22.0)), ("weight", grams(28.0))]);
    refill.record.fields = vec![
      ("name".to_owned(), GUMMIES[0].1.to_owned()),
      ("category".to_owned(), "Vape Pens".to_owned()),
    ];
    let output = engine.evaluate(refill).await.unwrap();
    let category = output.fields.iter().find(|field| field.field == "category").unwrap();
    assert_eq!(category.change, FieldChange::Restored);
    assert!(category.similarity.unwrap() < 0.01);
    let weight = quantity(&output, "weight");
    assert_eq!(weight.change, QuantityChange::Restored);
    assert!((weight.factor.unwrap() - 8.0).abs() < 1e-9);
  }

  #[tokio::test]
  async fn a_record_that_cannot_be_imported_fails_alone() {
    let engine = measuring_engine();
    let owned = |id: &str, brand_id: &str, fields: &[(&str, &str)]| ImportInput {
      record: RecordInput {
        brand_id: Some(brand_id.to_owned()),
        ..record(Some(id), fields)
      },
      anchor: None,
    };
    engine.import(vec![owned("taken", "brand-1", &GUMMIES)]).await.unwrap();
    let embedded_before = engine.embedder.embedded();

    let mut negative = import_of("negative", &GUMMIES, None);
    negative.record.quantities = vec![("weight".to_owned(), grams(-1.0))];
    let outcomes = engine
      .import(vec![
        import_of("unknown-field", &[("name", "Gummies"), ("colour", "blue")], None),
        owned("good", "brand-1", &VAPE),
        import_of("no-name", &[("category", "Edibles")], None),
        negative,
        import_of("bad-original", &GUMMIES, Some(&[("category", "Edibles")])),
        owned("taken", "brand-2", &GUMMIES),
        ImportInput {
          record: record(None, &GUMMIES),
          anchor: None,
        },
        import_of("  ", &GUMMIES, None),
      ])
      .await
      .unwrap();

    let invalid = |outcome: &Result<Imported, EngineError>, wanted: &str| {
      assert!(
        matches!(outcome, Err(EngineError::InvalidArgument(message)) if message.contains(wanted)),
        "{outcome:?} should mention `{wanted}`"
      );
    };
    invalid(&outcomes[0], "colour");
    assert_eq!(
      outcomes[1],
      Ok(Imported::Created),
      "the good record between two bad ones is imported"
    );
    invalid(&outcomes[2], "name");
    invalid(&outcomes[3], "negative");
    invalid(&outcomes[4], "the original version");
    assert!(
      matches!(&outcomes[5], Err(EngineError::FailedPrecondition(message)) if message.contains("brand-1")),
      "{:?}",
      outcomes[5]
    );
    invalid(&outcomes[6], "`record.id` is required");
    invalid(&outcomes[7], "blank");

    for id in ["unknown-field", "no-name", "negative", "bad-original"] {
      assert_eq!(stored(&engine, id).await, None, "{id}");
    }
    assert_eq!(
      stored(&engine, "taken").await.unwrap().brand_id.as_deref(),
      Some("brand-1")
    );
    assert_eq!(
      stored(&engine, "good").await.unwrap().brand_id.as_deref(),
      Some("brand-1")
    );
    assert_eq!(
      engine.embedder.embedded() - embedded_before,
      3,
      "only a record that fits the schema and is not stored reaches the model: `good`, with its three fields"
    );
  }

  /// Sending an export again has to be cheap, because that is how an interrupted one is resumed and how new
  /// records are added to an old one. A record that is already stored costs a read: none of its text, and none
  /// of its original version's, is embedded, and it is left exactly as it was.
  #[tokio::test]
  async fn a_record_that_is_already_stored_costs_an_import_no_embedding() {
    let engine = engine();
    let export = || {
      vec![
        import_of("p1", &GUMMIES, Some(&GUMMIES_EDITED)),
        import_of("p2", &VAPE, None),
      ]
    };
    let first = engine.import(export()).await.unwrap();
    assert_eq!(first, [Ok(Imported::Created), Ok(Imported::Created)]);
    let (before, embedded) = (
      [stored(&engine, "p1").await, stored(&engine, "p2").await],
      engine.embedder.embedded(),
    );
    assert_eq!(
      embedded, 9,
      "three fields of each record, and three of the one original version"
    );

    let again = engine.import(export()).await.unwrap();
    assert_eq!(again, [Ok(Imported::Skipped), Ok(Imported::Skipped)]);
    assert_eq!(engine.embedder.embedded(), embedded, "nothing more reached the model");
    assert_eq!([stored(&engine, "p1").await, stored(&engine, "p2").await], before);

    // A batch of stored and new records embeds the new ones only, and each still gets its own answer in order.
    let mixed = engine
      .import(vec![
        import_of("p1", &VAPE, None),
        import_of("p3", &GUMMIES, None),
        import_of("p2", &GUMMIES, None),
      ])
      .await
      .unwrap();
    assert_eq!(
      mixed,
      [Ok(Imported::Skipped), Ok(Imported::Created), Ok(Imported::Skipped)]
    );
    assert_eq!(engine.embedder.embedded(), embedded + 3);
    assert_eq!(
      stored(&engine, "p1").await,
      before[0],
      "what the stream said about a stored record changed nothing"
    );
  }

  #[tokio::test]
  async fn a_model_failure_fails_the_import_and_writes_nothing() {
    let embedder = WordBagEmbedder {
      fail_with: Some(EmbedError::Overloaded),
      ..WordBagEmbedder::new("word-bag-v1")
    };
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let engine = Engine::new(
      MemoryStore::new(layout, dimension),
      embedder,
      schema(),
      Policy::new(0.75, None).unwrap(),
    );
    let outcome = engine.import(vec![import_of("p1", &GUMMIES, None)]).await;
    assert!(matches!(outcome, Err(EngineError::Overloaded(_))), "{outcome:?}");
    assert_eq!(stored(&engine, "p1").await, None);
  }

  #[tokio::test]
  async fn malformed_requests_are_invalid_arguments_and_store_nothing() {
    let engine = engine();
    let invalid = [
      evaluate_input(Some("p1"), &[("name", "Gummies"), ("colour", "blue")]),
      evaluate_input(Some("p1"), &[("category", "Edibles")]),
      evaluate_input(Some("p1"), &[("name", "   ")]),
      evaluate_input(Some("p1"), &[]),
      evaluate_input(Some("  "), &GUMMIES),
      EvaluateInput {
        record: RecordInput {
          brand_id: Some(String::new()),
          ..record(Some("p1"), &GUMMIES)
        },
        ..EvaluateInput::default()
      },
    ];
    for input in invalid {
      let shown = format!("{input:?}");
      assert!(
        matches!(engine.evaluate(input).await, Err(EngineError::InvalidArgument(_))),
        "{shown}"
      );
    }
    assert_eq!(stored(&engine, "p1").await, None);
    assert_eq!(engine.embedder.embedded(), 0, "invalid records never reach the model");
  }

  #[tokio::test]
  async fn a_baseline_from_another_model_is_a_failed_precondition() {
    let old = engine();
    old.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();

    let embedder = WordBagEmbedder::new("word-bag-v2");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let upgraded = Engine::new(
      Arc::into_inner(old.store).unwrap().reopened(layout, dimension),
      embedder,
      schema(),
      old.policy,
    );

    let output = upgraded.evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED)).await;
    assert!(matches!(output, Err(EngineError::FailedPrecondition(message)) if message.contains("re-index")));
  }

  #[tokio::test]
  async fn embedder_failures_keep_their_meaning() {
    type Kind = fn(&EngineError) -> bool;
    let cases: [(EmbedError, Kind); 4] = [
      (EmbedError::Overloaded, |error| {
        matches!(error, EngineError::Overloaded(_))
      }),
      (EmbedError::WorkerGone, |error| {
        matches!(error, EngineError::Unavailable(_))
      }),
      (EmbedError::TooLate, |error| {
        matches!(error, EngineError::DeadlineExceeded(_))
      }),
      (EmbedError::Failed("nan".to_owned()), |error| {
        matches!(error, EngineError::Internal(_))
      }),
    ];
    for (failure, expected) in cases {
      let embedder = WordBagEmbedder {
        fail_with: Some(failure.clone()),
        ..WordBagEmbedder::new("word-bag-v1")
      };
      let (layout, dimension) = store_layout(&schema(), &embedder);
      let engine = Engine::new(
        MemoryStore::new(layout, dimension),
        embedder,
        schema(),
        Policy::new(0.75, None).unwrap(),
      );
      let error = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap_err();
      assert!(expected(&error), "{failure:?} became {error:?}");
      assert_eq!(stored(&engine, "p1").await, None);
    }
  }

  /// What the per-record lock is for, now that the store refuses a write over a record that changed (D35).
  /// Without it, requests for one record in one process would all read it, one would write, and the rest would
  /// lose, read again, and judge again, some of them until they gave up. With it, every one of them reads once
  /// and writes once, in turn.
  #[tokio::test]
  async fn requests_for_one_record_in_one_process_take_turns_and_none_of_them_loses_a_race() {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let engine = Arc::new(Engine::new(
      Slow::over(MemoryStore::new(layout, dimension)),
      embedder,
      schema(),
      Policy::new(0.75, None).unwrap(),
    ));
    engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let before = engine.store.reads();

    // Every one of these writes, whatever it is judged against.
    let tasks: Vec<_> = (0..32)
      .map(|turn| {
        let engine = Arc::clone(&engine);
        tokio::spawn(async move {
          let fields = if turn % 2 == 0 { &VAPE } else { &GUMMIES };
          engine
            .evaluate(EvaluateInput {
              mode: WriteMode::Force,
              ..evaluate_input(Some("p1"), fields)
            })
            .await
        })
      })
      .collect();
    for task in tasks {
      let output = task.await.unwrap().expect("nobody gave up");
      assert!(output.stored);
    }
    assert_eq!(engine.store.0.get("p1").await.unwrap().unwrap().revision, 33);
    assert_eq!(engine.store.reads() - before, 32, "one read for each request");
  }

  #[tokio::test]
  async fn concurrent_revisions_of_one_id_are_serialized() {
    let engine = Arc::new(engine());
    let tasks: Vec<_> = (0..32)
      .map(|_| {
        let engine = Arc::clone(&engine);
        tokio::spawn(async move {
          engine
            .evaluate(evaluate_input(Some("p1"), &GUMMIES))
            .await
            .unwrap()
            .verdict
        })
      })
      .collect();
    let mut created = 0;
    for task in tasks {
      created += usize::from(task.await.unwrap() == Verdict::Created);
    }
    assert_eq!(created, 1, "exactly one request found no baseline");
    assert_eq!(stored(&engine, "p1").await.unwrap().revision, 1);
  }

  type Replica = Engine<Arc<MemoryStore>, WordBagEmbedder>;

  /// Two replicas on one store, which share nothing else: `here`, whose reads a test can get in behind, and
  /// `there`.
  fn replicas() -> (Engine<Raced, WordBagEmbedder>, Arc<Replica>) {
    let policy = Policy::new(0.75, None).unwrap();
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let store = Arc::new(MemoryStore::new(layout, dimension));
    let there = Engine::new(
      Arc::clone(&store),
      WordBagEmbedder::new("word-bag-v1"),
      schema(),
      policy,
    );
    (
      Engine::new(Raced::over(store), embedder, schema(), policy),
      Arc::new(there),
    )
  }

  /// `there` evaluating `input`, as something to get in behind a read.
  fn rival(there: &Arc<Replica>, input: EvaluateInput) -> impl Future<Output = ()> + Send + 'static {
    let there = Arc::clone(there);
    async move {
      there.evaluate(input).await.unwrap();
    }
  }

  const CHEWS: [(&str, &str); 3] = [
    ("name", "Blueberry Gummies Vegan Chews"),
    ("category", "Edibles Gummies"),
    ("description", "Ten vegan gummies with real fruit"),
  ];
  const RINGS: [(&str, &str); 3] = [
    ("name", "Blueberry Gummies Sour Rings"),
    ("category", "Edibles Gummies"),
    ("description", "Ten vegan gummies with real fruit"),
  ];

  /// What D10 could not promise until the store could be written to conditionally. Each of two edits is a small
  /// change to the record both replicas read, and neither is a small change to the other. Judged against that
  /// one baseline, both would be accepted, and the second write would replace the first unseen.
  #[tokio::test]
  async fn two_replicas_cannot_both_accept_against_one_baseline() {
    // One after the other, on one engine: the first is accepted and the second, held to the first, is not.
    let serial = engine();
    serial.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let first = serial.evaluate(evaluate_input(Some("p1"), &CHEWS)).await.unwrap();
    let second = serial.evaluate(evaluate_input(Some("p1"), &RINGS)).await.unwrap();
    assert_eq!((first.verdict, second.verdict), (Verdict::Accepted, Verdict::Rejected));
    // And the second alone would have been accepted: it is the order that decides.
    let alone = engine();
    alone.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let unopposed = alone.evaluate(evaluate_input(Some("p1"), &RINGS)).await.unwrap();
    assert_eq!(unopposed.verdict, Verdict::Accepted);

    // At once, on two replicas: `there` writes the first while `here` is between reading and writing.
    let (here, there) = replicas();
    there.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p1"), &CHEWS)));
    let raced = here.evaluate(evaluate_input(Some("p1"), &RINGS)).await.unwrap();
    assert_eq!(
      raced, second,
      "the answer the request gets when it simply arrives second"
    );
    assert_eq!(
      (raced.verdict, raced.stored, raced.revision),
      (Verdict::Rejected, false, 2)
    );
    let kept = here.store.shared().get("p1").await.unwrap().unwrap();
    let expected = stored(&serial, "p1").await.unwrap();
    assert_eq!(
      (kept.revision, &kept.vectors, &kept.anchor),
      (2, &expected.vectors, &expected.anchor)
    );
    assert_eq!(here.store.reads(), 2, "read, found it changed, read again");
  }

  /// The same race where both edits are fine in either order: neither is lost, and the revisions count both.
  #[tokio::test]
  async fn an_edit_that_lost_the_race_is_judged_against_the_edit_that_won() {
    let serial = engine();
    serial.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    serial
      .evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))
      .await
      .unwrap();
    let second = serial.evaluate(evaluate_input(Some("p1"), &CHEWS)).await.unwrap();

    let (here, there) = replicas();
    there.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p1"), &GUMMIES_EDITED)));
    let raced = here.evaluate(evaluate_input(Some("p1"), &CHEWS)).await.unwrap();
    assert_eq!(raced, second);
    assert_eq!(
      (raced.verdict, raced.stored, raced.revision),
      (Verdict::Accepted, true, 3)
    );
    let kept = here.store.shared().get("p1").await.unwrap().unwrap();
    let expected = stored(&serial, "p1").await.unwrap();
    assert_eq!(
      (kept.revision, &kept.vectors, &kept.anchor),
      (3, &expected.vectors, &expected.anchor)
    );

    // A forced write is conditional too: it replaces what is stored now, and its revision follows that.
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p1"), &GUMMIES)));
    let forced = here
      .evaluate(EvaluateInput {
        mode: WriteMode::Force,
        ..evaluate_input(Some("p1"), &VAPE)
      })
      .await
      .unwrap();
    assert_eq!(
      (forced.verdict, forced.stored, forced.revision),
      (Verdict::Rejected, true, 5)
    );
    let kept = here.store.shared().get("p1").await.unwrap().unwrap();
    assert_eq!((kept.revision, &kept.anchor), (5, &kept.vectors.composite));

    // A request that writes nothing has nothing to lose a race with, and is answered from what it read.
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p1"), &GUMMIES)));
    let dry = here
      .evaluate(EvaluateInput {
        mode: WriteMode::DryRun,
        ..evaluate_input(Some("p1"), &VAPE)
      })
      .await
      .unwrap();
    assert_eq!((dry.similarity, dry.stored, dry.revision), (Some(1.0), false, 5));
  }

  #[tokio::test]
  async fn two_replicas_that_both_find_no_record_do_not_both_create_it() {
    let (here, there) = replicas();
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p1"), &GUMMIES)));
    let second = here
      .evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))
      .await
      .unwrap();
    assert_eq!(
      (second.verdict, second.stored, second.revision),
      (Verdict::Accepted, true, 2),
      "it found a record there after all, and was judged as an edit of it"
    );
    assert_eq!(here.store.shared().get("p1").await.unwrap().unwrap().revision, 2);
    assert_eq!(here.store.reads(), 2);

    // The same record from both is created once, and the one that came second wrote nothing.
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p2"), &GUMMIES)));
    let same = here.evaluate(evaluate_input(Some("p2"), &GUMMIES)).await.unwrap();
    assert_eq!(
      (same.verdict, same.stored, same.revision),
      (Verdict::Accepted, false, 1)
    );

    // A record of another brand that turned up in between is another brand's record.
    let branded = |brand_id: &str| EvaluateInput {
      record: RecordInput {
        brand_id: Some(brand_id.to_owned()),
        ..record(Some("p3"), &GUMMIES)
      },
      ..EvaluateInput::default()
    };
    here.store.after_read(0, rival(&there, branded("brand-1")));
    let refused = here.evaluate(branded("brand-2")).await.unwrap_err();
    assert!(
      matches!(&refused, EngineError::FailedPrecondition(message) if message.contains("brand-1")),
      "{refused}"
    );
  }

  /// An import only creates (D20), and that holds against another replica as it holds against itself.
  #[tokio::test]
  async fn an_import_never_replaces_what_another_replica_stored_a_moment_before() {
    let (here, there) = replicas();
    // An import reads twice: once before it embeds, to save the work, and once when it goes to write.
    here
      .store
      .after_read(1, rival(&there, evaluate_input(Some("p1"), &GUMMIES)));
    let outcomes = here.import(vec![import_of("p1", &VAPE, None)]).await.unwrap();
    assert_eq!(outcomes, [Ok(Imported::Skipped)]);
    let reference = engine();
    reference.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let kept = here.store.shared().get("p1").await.unwrap().unwrap();
    assert_eq!(
      (kept.revision, &kept.vectors),
      (1, &stored(&reference, "p1").await.unwrap().vectors),
      "the gummies the other replica stored, not the vape this one was importing"
    );
    assert_eq!(here.store.reads(), 3);

    // Created under a brand the import does not name, it is refused like any import of that id.
    here.store.after_read(
      1,
      rival(
        &there,
        EvaluateInput {
          record: RecordInput {
            brand_id: Some("brand-1".to_owned()),
            ..record(Some("p2"), &GUMMIES)
          },
          ..EvaluateInput::default()
        },
      ),
    );
    let outcomes = here.import(vec![import_of("p2", &VAPE, None)]).await.unwrap();
    assert!(
      matches!(outcomes.as_slice(), [Err(EngineError::FailedPrecondition(message))] if message.contains("brand-1")),
      "{outcomes:?}"
    );
  }

  /// The brand a delete has to name is the brand of the record that is deleted (D14), not of a version of it
  /// that was read a moment earlier.
  #[tokio::test]
  async fn a_delete_is_held_to_the_record_as_it_is_when_it_is_deleted() {
    let (here, there) = replicas();
    there.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    // The record gains a brand while a delete that names none is on its way.
    here.store.after_read(
      0,
      rival(
        &there,
        EvaluateInput {
          record: RecordInput {
            brand_id: Some("brand-1".to_owned()),
            ..record(Some("p1"), &GUMMIES)
          },
          ..EvaluateInput::default()
        },
      ),
    );
    let refused = here.delete("p1".to_owned(), None).await.unwrap_err();
    assert!(
      matches!(&refused, EngineError::FailedPrecondition(message) if message.contains("brand-1")),
      "{refused}"
    );
    let kept = here.store.shared().get("p1").await.unwrap().unwrap();
    assert_eq!((kept.revision, kept.brand_id.as_deref()), (2, Some("brand-1")));

    // A record that was only edited in between is still the caller's to delete, and is deleted as it is now.
    let edit = EvaluateInput {
      record: RecordInput {
        brand_id: Some("brand-1".to_owned()),
        ..record(Some("p1"), &GUMMIES_EDITED)
      },
      ..EvaluateInput::default()
    };
    here.store.after_read(0, rival(&there, edit));
    let reads = here.store.reads();
    assert!(here.delete("p1".to_owned(), Some("brand-1".to_owned())).await.unwrap());
    assert_eq!(here.store.shared().get("p1").await.unwrap(), None);
    assert_eq!(here.store.reads() - reads, 2);
  }

  /// The case the property `two_replicas_asked_at_once_end_as_if_asked_one_after_the_other` first shrank to, and
  /// the one answer that is allowed to differ from any order of arrival. Two replicas delete one record at the
  /// same moment: the record is gone, and both are told there was one. The store can say that a record is gone
  /// and not which of two deletes removed it, and a caller that asked for a record to be gone has what it asked.
  #[tokio::test]
  async fn two_replicas_deleting_one_record_at_once_are_both_told_it_was_there() {
    let (here, there) = replicas();
    there.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    here.store.after_read(0, {
      let there = Arc::clone(&there);
      async move {
        assert!(there.delete("p1".to_owned(), None).await.unwrap());
      }
    });
    assert!(here.delete("p1".to_owned(), None).await.unwrap());
    assert_eq!(here.store.shared().get("p1").await.unwrap(), None);
    // One after the other, the second is told there was nothing to delete.
    assert!(!here.delete("p1".to_owned(), None).await.unwrap());
    assert!(!there.delete("p1".to_owned(), None).await.unwrap());
  }

  /// A lost attempt is not an outcome: the candidate policy hears about the request once, as it was finally
  /// judged, against the baseline it was finally judged against (D30).
  #[tokio::test]
  async fn an_observer_hears_nothing_of_an_attempt_that_lost_the_race() {
    let (here, there) = replicas();
    let recorder = Arc::new(Recorder::default());
    let here = here.with_observer(recorder.clone());
    there.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p1"), &GUMMIES_EDITED)));
    here.evaluate(evaluate_input(Some("p1"), &CHEWS)).await.unwrap();
    let told = recorder.told();
    assert_eq!(told.len(), 1, "{told:?}");
    assert_eq!((told[0].stored, told[0].revision), (true, 3));
    assert_eq!(
      told[0].judged.as_ref().unwrap().0,
      2,
      "the baseline that was there when it was judged"
    );

    // The same for a record that another replica created first.
    here
      .store
      .after_read(0, rival(&there, evaluate_input(Some("p2"), &GUMMIES)));
    here
      .evaluate(evaluate_input(Some("p2"), &GUMMIES_EDITED))
      .await
      .unwrap();
    let told = recorder.told();
    assert_eq!(told.len(), 2, "{told:?}");
    assert_eq!(told[1].judged.as_ref().unwrap().0, 1);
  }

  /// A store on which another writer always got there first.
  struct Contended {
    inner: MemoryStore,
    reads: std::sync::atomic::AtomicUsize,
  }

  impl VectorStore for Contended {
    async fn check(&self) -> Result<(), StoreError> {
      self.inner.check().await
    }

    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      self.reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
      self.inner.get(id).await
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      self.inner.put(record).await
    }

    async fn put_if(&self, _: &StoredRecord, _: Expected) -> Result<bool, StoreError> {
      Ok(false)
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.inner.delete(id).await
    }

    async fn delete_if(&self, _: &str, _: Version) -> Result<bool, StoreError> {
      Ok(false)
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.inner.search(query).await
    }

    async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
      self.inner.search_vectors(query).await
    }
  }

  /// Losing the race is tried again, a bounded number of times. After that the request has an answer of its own,
  /// which says that nothing was written and that sending it again is safe.
  #[tokio::test]
  async fn a_record_that_keeps_changing_under_a_request_is_a_conflict_and_nothing_is_written() {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let store = Contended {
      inner: MemoryStore::new(layout, dimension),
      reads: std::sync::atomic::AtomicUsize::new(0),
    };
    let recorder = Arc::new(Recorder::default());
    let engine =
      Engine::new(store, embedder, schema(), Policy::new(0.75, None).unwrap()).with_observer(recorder.clone());
    let reads =
      |engine: &Engine<Contended, WordBagEmbedder>| engine.store.reads.swap(0, std::sync::atomic::Ordering::Relaxed);
    let conflict = |error: &EngineError, id: &str| {
      assert!(
        matches!(error, EngineError::Conflict(message)
          if message.contains(&format!("`{id}`")) && message.contains("nothing was written") && message.contains("again")),
        "{error:?}"
      );
    };

    // A record nobody has: every attempt to create it finds someone else did.
    let error = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap_err();
    conflict(&error, "p1");
    assert_eq!(reads(&engine), WRITE_ATTEMPTS);
    let outcomes = engine.import(vec![import_of("p1", &GUMMIES, None)]).await.unwrap();
    let [Err(error)] = outcomes.as_slice() else {
      panic!("expected one failure, got {outcomes:?}")
    };
    conflict(error, "p1");
    // One look before embedding, and one for each attempt.
    assert_eq!(reads(&engine), WRITE_ATTEMPTS + 1);

    // A record that is there: an edit, a forced write, and a delete all give up the same way.
    let seed = super::tests::engine();
    seed.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    let gummies = stored(&seed, "p1").await.unwrap();
    engine.store.inner.put(&gummies).await.unwrap();
    let error = engine
      .evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))
      .await
      .unwrap_err();
    conflict(&error, "p1");
    assert_eq!(reads(&engine), WRITE_ATTEMPTS);
    let forced = EvaluateInput {
      mode: WriteMode::Force,
      ..evaluate_input(Some("p1"), &VAPE)
    };
    conflict(&engine.evaluate(forced).await.unwrap_err(), "p1");
    assert_eq!(reads(&engine), WRITE_ATTEMPTS);
    conflict(&engine.delete("p1".to_owned(), None).await.unwrap_err(), "p1");
    assert_eq!(reads(&engine), WRITE_ATTEMPTS);
    assert!(
      recorder.told().is_empty(),
      "a request that gave up had no outcome to tell of"
    );

    // What needs no write is answered at the first attempt: a dry run, a rejection, the same record again.
    let dry = EvaluateInput {
      mode: WriteMode::DryRun,
      ..evaluate_input(Some("p1"), &GUMMIES_EDITED)
    };
    assert_eq!(engine.evaluate(dry).await.unwrap().verdict, Verdict::Accepted);
    let rejected = engine.evaluate(evaluate_input(Some("p1"), &VAPE)).await.unwrap();
    assert_eq!(rejected.verdict, Verdict::Rejected);
    let same = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
    assert_eq!((same.verdict, same.stored), (Verdict::Accepted, false));
    assert_eq!(reads(&engine), 3);
    assert_eq!(recorder.told().len(), 3);
    assert_eq!(engine.store.inner.get("p1").await.unwrap().unwrap(), gummies);
  }

  /// Readiness is made from two things, and a request is served only when both hold: the embedder can embed and
  /// the store answers. Asking embeds nothing and reads nothing.
  #[tokio::test]
  async fn an_engine_is_ready_when_its_embedder_is_alive_and_its_store_answers() {
    struct Unreachable(MemoryStore);
    impl VectorStore for Unreachable {
      async fn check(&self) -> Result<(), StoreError> {
        Err(StoreError::Unavailable("connection refused".to_owned()))
      }
      async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
        self.0.get(id).await
      }
      async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
        self.0.put(record).await
      }
      async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
        self.0.put_if(record, expected).await
      }
      async fn delete(&self, id: &str) -> Result<bool, StoreError> {
        self.0.delete(id).await
      }
      async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
        self.0.delete_if(id, version).await
      }
      async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
        self.0.search(query).await
      }
      async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
        self.0.search_vectors(query).await
      }
    }
    struct Stopped(WordBagEmbedder);
    impl TextEmbedder for Stopped {
      async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0.embed(texts).await
      }
      fn dimension(&self) -> usize {
        self.0.dimension()
      }
      fn fingerprint(&self) -> &str {
        self.0.fingerprint()
      }
      fn alive(&self) -> bool {
        false
      }
    }

    let ready = engine();
    assert_eq!(ready.check().await, Ok(()));
    assert_eq!(ready.embedder.embedded(), 0, "nothing was embedded to find that out");

    let policy = Policy::new(0.75, None).unwrap();
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let no_store = Engine::new(
      Unreachable(MemoryStore::new(layout.clone(), dimension)),
      embedder,
      schema(),
      policy,
    );
    let error = no_store.check().await.unwrap_err();
    assert!(
      matches!(&error, EngineError::Unavailable(message) if message.contains("connection refused")),
      "{error:?}"
    );

    let no_thread = Engine::new(
      MemoryStore::new(layout, dimension),
      Stopped(WordBagEmbedder::new("word-bag-v1")),
      schema(),
      policy,
    );
    let error = no_thread.check().await.unwrap_err();
    assert!(
      matches!(&error, EngineError::Unavailable(message) if message.contains("embedding thread")),
      "{error:?}"
    );
  }

  async fn seeded() -> TestEngine {
    let engine = engine();
    let seed = |id: &str, brand_id: &str, fields: &[(&str, &str)]| EvaluateInput {
      record: RecordInput {
        brand_id: Some(brand_id.to_owned()),
        ..record(Some(id), fields)
      },
      ..EvaluateInput::default()
    };
    engine.evaluate(seed("gummies", "brand-1", &GUMMIES)).await.unwrap();
    engine
      .evaluate(seed("gummies-100", "brand-2", &GUMMIES_EDITED))
      .await
      .unwrap();
    engine.evaluate(seed("vape", "brand-1", &VAPE)).await.unwrap();
    engine
  }

  fn ids(neighbors: &[Neighbor]) -> Vec<&str> {
    neighbors.iter().map(|neighbor| neighbor.id.as_str()).collect()
  }

  #[tokio::test]
  async fn searching_around_a_stored_record_ranks_the_others_and_omits_itself() {
    let engine = seeded().await;
    let neighbors = engine.find_similar(similar_to_id("gummies")).await.unwrap();
    assert_eq!(ids(&neighbors), ["gummies-100", "vape"]);
    assert!(
      neighbors[0].similarity > 0.75 && neighbors[1].similarity < 0.3,
      "{neighbors:?}"
    );
    assert_eq!(neighbors[0].brand_id.as_deref(), Some("brand-2"));
  }

  #[tokio::test]
  async fn the_search_score_is_the_score_evaluate_would_give() {
    let engine = seeded().await;
    let neighbors = engine.find_similar(similar_to_id("gummies")).await.unwrap();
    let preview = EvaluateInput {
      mode: WriteMode::DryRun,
      record: RecordInput {
        brand_id: Some("brand-1".to_owned()),
        ..record(Some("gummies"), &GUMMIES_EDITED)
      },
      ..EvaluateInput::default()
    };
    let evaluated = engine.evaluate(preview).await.unwrap();
    assert!((neighbors[0].similarity - evaluated.similarity.unwrap()).abs() < 1e-5);
  }

  #[tokio::test]
  async fn searching_around_an_ad_hoc_record_embeds_it_without_storing_it() {
    let engine = seeded().await;
    let neighbors = engine.find_similar(similar_to_record(&VAPE)).await.unwrap();
    assert_eq!(ids(&neighbors)[0], "vape");
    assert!((neighbors[0].similarity - 1.0).abs() < 1e-5);

    // An ad hoc record that carries an id is searching for *other* records like it.
    let mut input = similar_to_record(&VAPE);
    input.query = SimilarTo::Record(record(Some("vape"), &VAPE));
    assert!(!ids(&engine.find_similar(input).await.unwrap()).contains(&"vape"));
  }

  #[tokio::test]
  async fn search_filters_and_limits_apply() {
    let engine = seeded().await;
    let around_gummies = || similar_to_id("gummies");

    let brand_id = FindSimilarInput {
      brand_id: Some("brand-1".to_owned()),
      ..around_gummies()
    };
    assert_eq!(ids(&engine.find_similar(brand_id).await.unwrap()), ["vape"]);

    let close = FindSimilarInput {
      min_similarity: Some(0.5),
      ..around_gummies()
    };
    assert_eq!(ids(&engine.find_similar(close).await.unwrap()), ["gummies-100"]);

    let one = FindSimilarInput {
      limit: 1,
      ..around_gummies()
    };
    assert_eq!(engine.find_similar(one).await.unwrap().len(), 1);

    let most = FindSimilarInput {
      limit: MAX_NEIGHBORS,
      ..around_gummies()
    };
    assert_eq!(engine.find_similar(most).await.unwrap().len(), 2);
  }

  #[tokio::test]
  async fn invalid_searches_are_refused() {
    let engine = seeded().await;
    let too_many = FindSimilarInput {
      limit: MAX_NEIGHBORS + 1,
      ..similar_to_id("gummies")
    };
    assert!(matches!(
      engine.find_similar(too_many).await,
      Err(EngineError::InvalidArgument(_))
    ));
    let off_scale = FindSimilarInput {
      min_similarity: Some(2.0),
      ..similar_to_id("gummies")
    };
    assert!(matches!(
      engine.find_similar(off_scale).await,
      Err(EngineError::InvalidArgument(_))
    ));
    assert!(matches!(
      engine.find_similar(similar_to_id(" ")).await,
      Err(EngineError::InvalidArgument(_))
    ));
    assert!(matches!(
      engine.find_similar(similar_to_id("nope")).await,
      Err(EngineError::NotFound(_))
    ));
    let unknown_field = similar_to_record(&[("colour", "blue")]);
    assert!(matches!(
      engine.find_similar(unknown_field).await,
      Err(EngineError::InvalidArgument(_))
    ));
  }

  #[tokio::test]
  async fn a_deleted_record_starts_over() {
    let engine = seeded().await;
    let brand = || Some("brand-1".to_owned());
    assert!(engine.delete("gummies".to_owned(), brand()).await.unwrap());
    assert!(!engine.delete("gummies".to_owned(), brand()).await.unwrap());
    assert!(
      !engine.delete("gummies".to_owned(), None).await.unwrap(),
      "nothing stored, nothing to check a brand against"
    );
    assert!(matches!(
      engine.delete(String::new(), None).await,
      Err(EngineError::InvalidArgument(_))
    ));

    // What would have been rejected as a revision is simply a new record now.
    let output = engine.evaluate(evaluate_input(Some("gummies"), &VAPE)).await.unwrap();
    assert_eq!((output.verdict, output.revision), (Verdict::Created, 1));
  }

  #[tokio::test]
  async fn describe_reports_what_the_engine_was_built_with() {
    let description = engine().describe();
    assert_eq!(description.model_fingerprint, "word-bag-v1");
    assert_eq!(description.composite_dimension, description.dimension * 3);
    assert_eq!(description.schema, schema());
    assert!((description.policy.threshold - 0.75).abs() < f32::EPSILON);
  }

  // ---- Requests that go away (D38) --------------------------------------------------------------------------

  /// A store whose writes wait at a gate until the test lets one through. For that while a write is neither done
  /// nor not done, which is where a request can be dropped and nobody can say what became of it. Reads give way
  /// once, as a round trip does, so that a request can also be dropped before it has written anything.
  struct Pausing {
    inner: Arc<MemoryStore>,
    gate: tokio::sync::Semaphore,
    arrived: std::sync::atomic::AtomicUsize,
    failing: std::sync::atomic::AtomicBool,
  }

  impl Pausing {
    /// Lets one write through.
    fn open(&self) {
      self.gate.add_permits(1);
    }

    /// Lets every write through from now on, for a test that is not about when they land.
    fn open_wide(&self) {
      self.gate.add_permits(1 << 20);
    }

    /// How many writes have reached the gate.
    fn arrived(&self) -> usize {
      self.arrived.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// From now on a write that is let through fails, and is not applied.
    fn fail(&self) {
      self.failing.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    async fn pass(&self) -> Result<(), StoreError> {
      self.arrived.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
      self.gate.acquire().await.unwrap().forget();
      if self.failing.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(StoreError::Unavailable("the connection was reset".to_owned()));
      }
      Ok(())
    }
  }

  impl VectorStore for Pausing {
    async fn check(&self) -> Result<(), StoreError> {
      self.inner.check().await
    }

    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      let found = self.inner.get(id).await;
      tokio::task::yield_now().await;
      found
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      self.inner.put(record).await
    }

    async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
      self.pass().await?;
      self.inner.put_if(record, expected).await
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.inner.delete(id).await
    }

    async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
      self.pass().await?;
      self.inner.delete_if(id, version).await
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.inner.search(query).await
    }

    async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
      self.inner.search_vectors(query).await
    }
  }

  type GatedEngine = Engine<Pausing, WordBagEmbedder>;

  fn gated() -> (GatedEngine, Arc<Recorder>) {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let store = Pausing {
      inner: Arc::new(MemoryStore::new(layout, dimension)),
      gate: tokio::sync::Semaphore::new(0),
      arrived: std::sync::atomic::AtomicUsize::new(0),
      failing: std::sync::atomic::AtomicBool::new(false),
    };
    let recorder = Arc::new(Recorder::default());
    let engine = Engine::new(store, embedder, schema(), Policy::new(0.75, None).unwrap());
    (engine.with_observer(recorder.clone()), recorder)
  }

  /// Runs `scene` on one thread and returns the warnings it logged. A write that outlives its request has
  /// nothing but the log to speak for it, so the log is what these tests read.
  fn warnings(scene: impl Future<Output = ()>) -> Vec<serde_json::Value> {
    let captured = similarity_log::testing::capture("warn", || {
      tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(scene);
    });
    captured
      .logs
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect()
  }

  /// Whether `waiting` is still not done after other tasks have had every chance to run.
  async fn still_waiting<T>(waiting: &mut Pin<Box<impl Future<Output = T>>>) -> bool {
    for _ in 0..50 {
      tokio::select! {
        biased;
        _ = &mut *waiting => return false,
        () = tokio::task::yield_now() => {}
      }
    }
    true
  }

  /// What `waiting` comes to, once other tasks have had every chance to run. One that never ends fails the test,
  /// and does not hang it.
  async fn ended<T>(waiting: &mut Pin<Box<impl Future<Output = T>>>) -> T {
    for _ in 0..50 {
      tokio::select! {
        biased;
        output = &mut *waiting => return output,
        () = tokio::task::yield_now() => {}
      }
    }
    panic!("it never ended");
  }

  /// Runs `request` until one more write has reached the gate, and drops it there: a caller that hung up, or
  /// whose deadline ran out, while its write was on its way to the store. A request that never gets that far
  /// fails the test, and does not hang it.
  async fn dropped_mid_write<T>(store: &Pausing, request: impl Future<Output = T>) {
    let before = store.arrived();
    let mut request = std::pin::pin!(request);
    for _ in 0..10_000 {
      if store.arrived() != before {
        return;
      }
      tokio::select! {
        biased;
        _ = &mut request => panic!("the request was answered before its write reached the store"),
        () = tokio::task::yield_now() => {}
      }
    }
    panic!("the request never got as far as its write");
  }

  /// Every write in flight has ended. One that never does fails the test, and does not hang it.
  async fn all_written(engine: &GatedEngine) {
    let mut settled = Box::pin(engine.settled());
    assert!(!still_waiting(&mut settled).await, "a write is still in flight");
  }

  const GONE: &str = "a write finished after the request that asked for it had gone";

  /// The case a caller cannot handle by itself. Its request was dropped while the write was on its way, so it
  /// was told nothing, and the write went through. The engine finishes what it started: the record is stored,
  /// the observer hears of it, the log says so with the id, and the record's next request waits for all of that
  /// and is judged against what the write left, so a retry of the same content is answered "unchanged".
  #[test]
  fn a_write_whose_request_has_gone_is_finished_observed_and_logged() {
    let (engine, recorder) = gated();
    let logged = warnings(async {
      let store = Arc::clone(&engine.store);
      store.open();
      engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
      all_written(&engine).await;

      dropped_mid_write(&store, engine.evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))).await;
      assert_eq!(store.inner.get("p1").await.unwrap().unwrap().revision, 1, "not yet");
      assert_eq!(
        recorder.told().len(),
        1,
        "and nobody has been told of a write that has not happened"
      );

      let mut retry = Box::pin(engine.evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED)));
      assert!(
        still_waiting(&mut retry).await,
        "the write in flight still holds the record"
      );
      let mut settled = Box::pin(engine.settled());
      assert!(still_waiting(&mut settled).await, "and it counts as in flight");
      assert_eq!(
        engine.writes(),
        Writes {
          in_flight: 1,
          unanswered: 0
        }
      );

      store.open();
      assert!(!still_waiting(&mut settled).await, "the write never ended");
      assert_eq!(store.inner.get("p1").await.unwrap().unwrap().revision, 2);
      let told = recorder.told();
      assert_eq!(told.len(), 2);
      assert_eq!((told[1].stored, told[1].revision), (true, 2));

      assert_eq!(
        engine.writes(),
        Writes {
          in_flight: 0,
          unanswered: 1
        },
        "counted where the line is written"
      );
      let retried = ended(&mut retry).await.unwrap();
      assert_eq!(
        (retried.verdict, retried.stored, retried.revision),
        (Verdict::Accepted, false, 2),
        "the retry finds its own content stored, and has nothing to write"
      );
    });
    assert_eq!(logged.len(), 1, "{logged:?}");
    let line = &logged[0];
    assert_eq!(line["message"], GONE);
    assert_eq!((&line["call"], &line["id"]), (&"evaluate".into(), &"p1".into()));
    assert_eq!((&line["stored"], &line["deleted"]), (&true.into(), &false.into()));
    assert_eq!(line["revision"], 2);
  }

  /// Every kind of write the engine makes is finished the same way: a record's first version, an import, and a
  /// delete.
  #[test]
  fn a_create_an_import_and_a_delete_each_outlive_their_request() {
    let (engine, recorder) = gated();
    let logged = warnings(async {
      let store = Arc::clone(&engine.store);
      dropped_mid_write(&store, engine.evaluate(evaluate_input(Some("p1"), &GUMMIES))).await;
      dropped_mid_write(&store, engine.import(vec![import_of("p2", &VAPE, None)])).await;
      for _ in 0..2 {
        store.open();
      }
      all_written(&engine).await;
      assert_eq!(store.inner.get("p1").await.unwrap().unwrap().revision, 1);
      assert_eq!(store.inner.get("p2").await.unwrap().unwrap().revision, 1);
      assert_eq!(
        recorder.told().len(),
        1,
        "the evaluate; an import is not a judged request"
      );

      dropped_mid_write(&store, engine.delete("p1".to_owned(), None)).await;
      assert!(store.inner.get("p1").await.unwrap().is_some());
      store.open();
      all_written(&engine).await;
      assert_eq!(store.inner.get("p1").await.unwrap(), None);
      assert_eq!(engine.writes().unanswered, 3);
    });
    let said: Vec<_> = logged
      .iter()
      .map(|line| {
        assert_eq!(line["message"], GONE);
        (
          line["call"].as_str().unwrap(),
          line["id"].as_str().unwrap(),
          line["stored"].as_bool().unwrap(),
          line["deleted"].as_bool().unwrap(),
          line["revision"].as_u64(),
        )
      })
      .collect();
    assert_eq!(
      said,
      [
        ("evaluate", "p1", true, false, Some(1)),
        ("import", "p2", true, false, Some(1)),
        ("delete", "p1", false, true, None),
      ]
    );
  }

  /// A request that is answered says what happened in its answer, and its write logs nothing of its own: the
  /// line is for the writes nobody was told about, and has to be rare enough to be read.
  #[test]
  fn a_write_whose_request_is_still_there_logs_nothing() {
    let (engine, _) = gated();
    let logged = warnings(async {
      engine.store.open_wide();
      engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
      engine
        .evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))
        .await
        .unwrap();
      assert!(engine.delete("p1".to_owned(), None).await.unwrap());
      all_written(&engine).await;
      assert_eq!(
        engine.writes(),
        Writes {
          in_flight: 0,
          unanswered: 0
        }
      );
    });
    assert_eq!(logged, Vec::<serde_json::Value>::new());
  }

  /// Until a request goes to write, dropping it is free: nothing was sent to the store, so there is nothing to
  /// finish, nobody to tell, and nothing to log.
  #[test]
  fn a_request_dropped_before_it_writes_leaves_nothing_behind() {
    let (engine, recorder) = gated();
    let logged = warnings(async {
      {
        let mut reading = Box::pin(engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)));
        tokio::select! {
          biased;
          _ = &mut reading => panic!("answered"),
          () = std::future::ready(()) => {}
        }
      }
      for _ in 0..50 {
        tokio::task::yield_now().await;
      }
      all_written(&engine).await;
      assert_eq!(engine.store.arrived(), 0, "no write was started");
      assert_eq!(engine.store.inner.get("p1").await.unwrap(), None);
      assert_eq!(recorder.told(), []);

      // The record's lock went with the request: the next one is not kept waiting.
      engine.store.open_wide();
      let mut next = Box::pin(engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)));
      let output = ended(&mut next).await.unwrap();
      assert_eq!(output.verdict, Verdict::Created);
    });
    assert_eq!(logged, Vec::<serde_json::Value>::new());
  }

  /// The one outcome nobody can know: the write failed on its way after the request had gone. The store may
  /// hold it or not, and the line says exactly that, so that whoever reads it knows which record to look at.
  #[test]
  fn a_write_that_fails_after_its_request_has_gone_is_logged_as_unknown() {
    let (engine, recorder) = gated();
    let logged = warnings(async {
      let store = Arc::clone(&engine.store);
      store.fail();
      dropped_mid_write(&store, engine.evaluate(evaluate_input(Some("p1"), &GUMMIES))).await;
      store.open();
      all_written(&engine).await;
      assert_eq!(recorder.told(), [], "a write that failed is not an outcome");
      assert_eq!(engine.writes().unanswered, 1, "nobody was told of this one either");
    });
    assert_eq!(logged.len(), 1, "{logged:?}");
    let line = &logged[0];
    assert_eq!(
      line["message"],
      "a write failed after the request that asked for it had gone, and may or may not have been applied"
    );
    assert_eq!((&line["call"], &line["id"]), (&"evaluate".into(), &"p1".into()));
    assert!(
      line["error"].as_str().unwrap().contains("the connection was reset"),
      "{line}"
    );
  }

  /// A write that lost its race changed nothing, whether or not anybody was waiting for it: with the request
  /// gone there is no second attempt, nobody to tell, and nothing worth a line.
  #[test]
  fn a_write_that_loses_its_race_after_its_request_has_gone_says_nothing() {
    let (engine, recorder) = gated();
    let logged = warnings(async {
      let store = Arc::clone(&engine.store);
      store.open();
      engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap();
      dropped_mid_write(&store, engine.evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED))).await;

      // Another replica writes the record while this one's write is on its way.
      let mut theirs = store.inner.get("p1").await.unwrap().unwrap();
      theirs.revision = 7;
      store.inner.put(&theirs).await.unwrap();

      store.open();
      all_written(&engine).await;
      assert_eq!(store.inner.get("p1").await.unwrap().unwrap(), theirs);
      assert_eq!(recorder.told().len(), 1, "only the creation");
      assert_eq!(
        engine.writes().unanswered,
        0,
        "nothing changed, so there was nothing to tell"
      );
    });
    assert_eq!(logged, Vec::<serde_json::Value>::new());
  }

  /// A request that is still there when its write fails gets the error, as it always did, and the log has no
  /// line about a request that had gone.
  #[tokio::test]
  async fn a_write_that_fails_is_the_requests_error_while_the_request_is_there() {
    let (engine, recorder) = gated();
    engine.store.fail();
    engine.store.open_wide();
    let error = engine.evaluate(evaluate_input(Some("p1"), &GUMMIES)).await.unwrap_err();
    assert!(matches!(error, EngineError::Unavailable(_)), "{error:?}");
    assert_eq!(recorder.told(), []);
    all_written(&engine).await;
  }
}
