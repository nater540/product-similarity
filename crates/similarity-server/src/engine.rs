//! The service's behavior, free of any transport.
//!
//! [`Engine`] is what the gRPC layer calls and what the tests exercise: it validates a record, embeds what needs
//! embedding, compares against the stored baseline, applies the policy, and writes back only what the policy allows.
//! It is generic over the store and the embedder so that every rule below is tested in memory, in microseconds,
//! without Qdrant or model weights.
//!
//! The one piece of state it owns is a set of locks. Evaluating a record is read-compare-write, and two concurrent
//! revisions of the same id must not both be compared against the same baseline and then both be written. Requests
//! for the same id are serialized here; requests for different ids run concurrently. This holds within one server
//! process. Several replicas writing the same id at the same instant can still interleave, which is acceptable for
//! a system whose callers already serialize edits to a single record, and is called out in the README.

use std::fmt;
use std::hash::{BuildHasher, RandomState};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use similarity_core::quantity::first_values;
use similarity_core::score::{self, Evaluation, FieldScore};
use similarity_core::{
  FieldChange, Policy, PreparedRecord, Quantities, Quantity, QuantityChange, QuantityScore, Reason, RecordVectors,
  Schema,
};
use similarity_store::{Neighbor, SearchQuery, StoreError, StoredRecord, VectorStore};
use tokio::sync::Mutex;

use crate::worker::{EmbedError, TextEmbedder};

/// How many neighbors `find_similar` returns when the caller does not say.
pub const DEFAULT_NEIGHBORS: usize = 10;
/// The most neighbors `find_similar` will return.
pub const MAX_NEIGHBORS: usize = 100;
/// How many independent locks guard record ids. A power of two well above the expected request concurrency, so that
/// two different ids rarely share a lock.
const LOCK_STRIPES: usize = 256;

/// A failed request, classified the way a caller needs it classified.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
  /// The request is malformed or does not fit the schema. Retrying it unchanged will fail again.
  #[error("{0}")]
  InvalidArgument(String),
  /// The record asked for does not exist.
  #[error("{0}")]
  NotFound(String),
  /// The request is fine but the stored state forbids it (another tenant's record, another model's vectors).
  #[error("{0}")]
  FailedPrecondition(String),
  /// The server is saturated. Retry with backoff.
  #[error("{0}")]
  Overloaded(String),
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
    }
  }
}

/// A record as a caller supplies it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecordInput {
  /// The caller's identifier, if the record has one yet.
  pub id: Option<String>,
  /// The record's owner.
  pub tenant: Option<String>,
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

/// An earlier version of a record: fields and quantities, with no id or tenant of its own.
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
  /// Only return this tenant's records.
  pub tenant: Option<String>,
  /// Only return records at least this similar.
  pub min_similarity: Option<f32>,
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
  tenant: Option<String>,
  vectors: RecordVectors,
  quantities: Quantities,
}

/// One record of an import, validated: everything needed to write it once its texts are embedded.
#[derive(Debug)]
struct PreparedImport {
  id: String,
  tenant: Option<String>,
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

/// The service's behavior.
#[derive(Debug)]
pub struct Engine<S, E> {
  store: S,
  embedder: E,
  schema: Schema,
  policy: Policy,
  observer: Option<Arc<dyn Observer>>,
  locks: Vec<Mutex<()>>,
  lock_hasher: RandomState,
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

/// Refuses a request that does not name the tenant `stored` belongs to.
///
/// Naming no tenant is refused like naming the wrong one: a check that a caller can skip by leaving a field out
/// would not catch the mistake it exists for, a code path that lost track of whose record it is editing. A record
/// stored without a tenant accepts any request, which is how it gains one.
fn same_tenant(stored: &StoredRecord, given: Option<&str>) -> Result<(), EngineError> {
  match (stored.tenant.as_deref(), given) {
    (None, _) => Ok(()),
    (Some(owner), Some(given)) if owner == given => Ok(()),
    (Some(owner), Some(given)) => Err(EngineError::FailedPrecondition(format!(
      "record `{}` belongs to tenant `{owner}`, not `{given}`",
      stored.id
    ))),
    (Some(owner), None) => Err(EngineError::FailedPrecondition(format!(
      "record `{}` belongs to tenant `{owner}`, and the request names no tenant",
      stored.id
    ))),
  }
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
      store,
      embedder,
      schema,
      policy,
      observer: None,
      locks: (0..LOCK_STRIPES).map(|_| Mutex::new(())).collect(),
      lock_hasher: RandomState::new(),
    }
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

  /// Reports the model, schema, and thresholds in force.
  pub fn describe(&self) -> Description {
    let (_, composite_dimension) = store_layout(&self.schema, &self.embedder);
    Description {
      model_fingerprint: self.embedder.fingerprint().to_owned(),
      dimension: self.embedder.dimension(),
      composite_dimension,
      schema: self.schema.clone(),
      policy: self.policy,
    }
  }

  fn lock_for(&self, id: &str) -> Result<&Mutex<()>, EngineError> {
    let stripes = u64::try_from(self.locks.len()).unwrap_or(u64::MAX);
    let hash = usize::try_from(self.lock_hasher.hash_one(id).checked_rem(stripes).unwrap_or(0)).unwrap_or(0);
    self
      .locks
      .get(hash)
      .ok_or_else(|| EngineError::Internal("lock stripe out of range".to_owned()))
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
    let tenant = required_text("record.tenant", input.record.tenant.clone())?;
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

    let _serialized = self.lock_for(&id)?.lock().await;
    let baseline = self.store.get(&id).await?;
    if let Some(stored) = &baseline {
      same_tenant(stored, tenant.as_deref())?;
    }
    let candidate = self
      .embed(&prepared, baseline.as_ref().map(|record| &record.vectors))
      .await?;

    let quantities = prepared.quantities().clone();
    if let Some(baseline) = baseline {
      let revision = Revision {
        id,
        tenant,
        vectors: candidate,
        quantities,
      };
      return self
        .revise(revision, (input.mode, input.threshold), &policy, baseline)
        .await;
    }
    let stored = input.mode != WriteMode::DryRun;
    if stored {
      let record = StoredRecord {
        id: id.clone(),
        tenant,
        anchor: candidate.composite.clone(),
        vectors: candidate.clone(),
        // A new record is its own point of reference, for quantities as for text.
        anchor_quantities: quantities.clone(),
        quantities,
        revision: 1,
        updated_at: now(),
      };
      self.store.put(&record).await?;
    }
    if let Some(observer) = &self.observer {
      observer.observe(&Observation {
        id: &id,
        mode: input.mode,
        threshold: input.threshold,
        policy: &policy,
        stored,
        revision: u64::from(stored),
        judged: None,
      });
    }
    Ok(EvaluateOutput::without_baseline(
      Verdict::Created,
      &policy,
      stored,
      candidate.composite,
    ))
  }

  /// Judges `candidate` against `baseline` and writes it back if the policy and the mode allow. `asked` is the
  /// request's mode and the threshold it asked for, if it asked for one; `policy` already has that threshold.
  async fn revise(
    &self,
    candidate: Revision,
    asked: (WriteMode, Option<f32>),
    policy: &Policy,
    baseline: StoredRecord,
  ) -> Result<EvaluateOutput, EngineError> {
    let (mode, threshold) = asked;
    let Revision {
      id,
      tenant,
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
      && tenant == baseline.tenant;
    let write = match mode {
      WriteMode::DryRun => false,
      WriteMode::Enforce => accepted && !identical,
      WriteMode::Force => true,
    };

    let mut revision = baseline.revision;
    if write {
      revision = revision.saturating_add(1);
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
        (anchor, first_values(&baseline.anchor_quantities, &quantities))
      };
      let record = StoredRecord {
        id: id.clone(),
        tenant,
        anchor,
        vectors: candidate.clone(),
        quantities: quantities.clone(),
        anchor_quantities,
        revision,
        updated_at: now(),
      };
      self.store.put(&record).await?;
    }

    // Only now is the outcome final: a write that failed has already returned its error, and is never observed.
    if let Some(observer) = &self.observer {
      observer.observe(&Observation {
        id: &id,
        mode,
        threshold,
        policy,
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

    Ok(EvaluateOutput {
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
    })
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
  /// its id is stored under a tenant it does not name) fails by itself and stops nothing else.
  ///
  /// # Errors
  ///
  /// Returns an error for the whole call only if the model fails: nothing is written in that case. Everything
  /// else is reported per record.
  pub async fn import(&self, inputs: Vec<ImportInput>) -> Result<Vec<Result<Imported, EngineError>>, EngineError> {
    let dimension = self.embedder.dimension();

    // Everything that does not need the store, for every record: validate, and work out what to embed. Records
    // are embedded from scratch (there is no baseline to reuse), so this does not depend on what is stored.
    let mut ready = Vec::with_capacity(inputs.len());
    let mut texts: Vec<String> = Vec::new();
    for input in inputs {
      ready.push(self.prepare_import(input).and_then(|prepared| {
        let wanted = |record: &PreparedRecord| -> Result<Vec<String>, EngineError> {
          let plan = record.plan(&self.schema, dimension, None)?;
          Ok(plan.texts().into_iter().map(str::to_owned).collect())
        };
        // Both lists are worked out before either is queued: the vectors that come back are told apart by
        // position alone, so a record must put in all of its texts or none.
        let record_texts = wanted(&prepared.record)?;
        let anchor_texts = prepared.anchor.as_ref().map(wanted).transpose()?.unwrap_or_default();
        let counts = (record_texts.len(), anchor_texts.len());
        texts.extend(record_texts);
        texts.extend(anchor_texts);
        Ok((prepared, counts.0, counts.1))
      }));
    }
    // Nothing to embed (every record was refused) is not a reason to wake the model.
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
        Ok((prepared, record_texts, anchor_texts)) => {
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

  fn prepare_import(&self, input: ImportInput) -> Result<PreparedImport, EngineError> {
    let Some(id) = required_text("record.id", input.record.id.clone())? else {
      return Err(EngineError::InvalidArgument(
        "`record.id` is required: a record without an id cannot be stored".to_owned(),
      ));
    };
    let tenant = required_text("record.tenant", input.record.tenant.clone())?;
    let record = self.prepare(&input.record)?;
    let anchor = input
      .anchor
      .map(|original| {
        self.prepare(&RecordInput {
          id: None,
          tenant: None,
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
      tenant,
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
    let _serialized = self.lock_for(&prepared.id)?.lock().await;
    if let Some(stored) = self.store.get(&prepared.id).await? {
      // Naming another tenant is the mistake the tenant rule exists for, here as anywhere (D14).
      same_tenant(&stored, prepared.tenant.as_deref())?;
      return Ok(Imported::Skipped);
    }

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
    self
      .store
      .put(&StoredRecord {
        id: prepared.id,
        tenant: prepared.tenant,
        vectors,
        anchor,
        quantities,
        anchor_quantities,
        revision: 1,
        updated_at: now(),
      })
      .await?;
    Ok(Imported::Created)
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
    let tenant = required_text("tenant", input.tenant)?;

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
      tenant: tenant.as_deref(),
      exclude_id: exclude_id.as_deref(),
      min_similarity: input.min_similarity,
    };
    Ok(self.store.search(query).await?)
  }

  /// Forgets a record. Returns whether there was one.
  ///
  /// # Errors
  ///
  /// See [`EngineError`].
  pub async fn delete(&self, id: String, tenant: Option<String>) -> Result<bool, EngineError> {
    let Some(id) = required_text("id", Some(id))? else {
      return Err(EngineError::InvalidArgument("`id` is required".to_owned()));
    };
    let tenant = required_text("tenant", tenant)?;
    let _serialized = self.lock_for(&id)?.lock().await;
    let Some(stored) = self.store.get(&id).await? else {
      return Ok(false);
    };
    same_tenant(&stored, tenant.as_deref())?;
    Ok(self.store.delete(&id).await?)
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use similarity_core::{FieldSpec, QuantitySpec};
  use similarity_store::MemoryStore;

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
      tenant: None,
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

  /// A store whose reads give way to other tasks, as a round trip to a real store does.
  struct Slow(MemoryStore);

  impl VectorStore for Slow {
    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      let found = self.0.get(id).await;
      tokio::task::yield_now().await;
      found
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      self.0.put(record).await
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.0.delete(id).await
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.0.search(query).await
    }
  }

  /// What an observer was told, kept: an [`Observation`] only borrows.
  #[derive(Debug, Clone, PartialEq)]
  struct Told {
    id: String,
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
      (last.id.as_str(), last.stored, last.revision, last.judged),
      ("p2", false, 0, None)
    );
  }

  /// A store that can be told to refuse every write.
  struct Refusing {
    inner: MemoryStore,
    refuse: std::sync::atomic::AtomicBool,
  }

  impl VectorStore for Refusing {
    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      self.inner.get(id).await
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      if self.refuse.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(StoreError::Unavailable("refusing writes".to_owned()));
      }
      self.inner.put(record).await
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.inner.delete(id).await
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.inner.search(query).await
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
      Slow(MemoryStore::new(layout, dimension)),
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
  /// such record" and both create it, and the second would never be judged against the first: a recycle through
  /// the gap between two requests. Each id is therefore handled by one request at a time.
  #[tokio::test]
  async fn two_requests_for_one_record_are_judged_one_after_the_other() {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let store = Slow(MemoryStore::new(layout, dimension));
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
      tenant: None,
      min_similarity: None,
    }
  }

  fn similar_to_id(id: &str) -> FindSimilarInput {
    FindSimilarInput {
      query: SimilarTo::Id(id.to_owned()),
      limit: 0,
      tenant: None,
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
  async fn a_record_can_gain_a_tenant_and_from_then_on_every_request_must_name_it() {
    let engine = engine();
    let with_tenant = |tenant: Option<&str>, fields: &[(&str, &str)]| EvaluateInput {
      record: RecordInput {
        tenant: tenant.map(str::to_owned),
        ..record(Some("p1"), fields)
      },
      ..EvaluateInput::default()
    };

    engine.evaluate(with_tenant(None, &GUMMIES)).await.unwrap();
    let adopted = engine
      .evaluate(with_tenant(Some("merchant-1"), &GUMMIES))
      .await
      .unwrap();
    assert!(adopted.stored, "gaining a tenant is a change worth writing");
    assert_eq!(
      stored(&engine, "p1").await.unwrap().tenant.as_deref(),
      Some("merchant-1")
    );

    let stolen = engine.evaluate(with_tenant(Some("merchant-2"), &GUMMIES)).await;
    assert!(matches!(stolen, Err(EngineError::FailedPrecondition(message)) if message.contains("merchant-1")));

    // Leaving the tenant out is not a way around the check, in any mode.
    let before = stored(&engine, "p1").await.unwrap();
    for mode in [WriteMode::Enforce, WriteMode::DryRun, WriteMode::Force] {
      let unnamed = EvaluateInput {
        mode,
        ..with_tenant(None, &VAPE)
      };
      let refused = engine.evaluate(unnamed).await;
      assert!(
        matches!(&refused, Err(EngineError::FailedPrecondition(message)) if message.contains("names no tenant")),
        "{mode:?}: {refused:?}"
      );
    }
    assert_eq!(stored(&engine, "p1").await.unwrap(), before);

    let owner = engine
      .evaluate(with_tenant(Some("merchant-1"), &GUMMIES_EDITED))
      .await
      .unwrap();
    assert_eq!(owner.verdict, Verdict::Accepted);
  }

  #[tokio::test]
  async fn deleting_a_record_that_has_a_tenant_takes_that_tenant() {
    let engine = engine();
    let owned = EvaluateInput {
      record: RecordInput {
        tenant: Some("merchant-1".to_owned()),
        ..record(Some("p1"), &GUMMIES)
      },
      ..EvaluateInput::default()
    };
    engine.evaluate(owned).await.unwrap();

    for tenant in [None, Some("merchant-2".to_owned())] {
      let refused = engine.delete("p1".to_owned(), tenant).await;
      assert!(
        matches!(refused, Err(EngineError::FailedPrecondition(_))),
        "{refused:?}"
      );
    }
    assert!(stored(&engine, "p1").await.is_some());
    assert!(
      engine
        .delete("p1".to_owned(), Some("merchant-1".to_owned()))
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
    let upgraded = Engine::new(old.store.reopened(layout, dimension), embedder, schema, old.policy);

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
    let owned = |id: &str, tenant: &str, fields: &[(&str, &str)]| ImportInput {
      record: RecordInput {
        tenant: Some(tenant.to_owned()),
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
      stored(&engine, "taken").await.unwrap().tenant.as_deref(),
      Some("brand-1")
    );
    assert_eq!(
      stored(&engine, "good").await.unwrap().tenant.as_deref(),
      Some("brand-1")
    );
    assert_eq!(
      engine.embedder.embedded() - embedded_before,
      6,
      "only records that fit the schema reach the model: `good` and the second `taken`, three fields each"
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
          tenant: Some(String::new()),
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
    let upgraded = Engine::new(old.store.reopened(layout, dimension), embedder, schema(), old.policy);

    let output = upgraded.evaluate(evaluate_input(Some("p1"), &GUMMIES_EDITED)).await;
    assert!(matches!(output, Err(EngineError::FailedPrecondition(message)) if message.contains("re-index")));
  }

  #[tokio::test]
  async fn embedder_failures_keep_their_meaning() {
    for (failure, expect_overloaded) in [(EmbedError::Overloaded, true), (EmbedError::WorkerGone, false)] {
      let embedder = WordBagEmbedder {
        fail_with: Some(failure),
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
      assert_eq!(
        matches!(error, EngineError::Overloaded(_)),
        expect_overloaded,
        "{error:?}"
      );
      assert_eq!(
        matches!(error, EngineError::Unavailable(_)),
        !expect_overloaded,
        "{error:?}"
      );
      assert_eq!(stored(&engine, "p1").await, None);
    }
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

  async fn seeded() -> TestEngine {
    let engine = engine();
    let seed = |id: &str, tenant: &str, fields: &[(&str, &str)]| EvaluateInput {
      record: RecordInput {
        tenant: Some(tenant.to_owned()),
        ..record(Some(id), fields)
      },
      ..EvaluateInput::default()
    };
    engine.evaluate(seed("gummies", "merchant-1", &GUMMIES)).await.unwrap();
    engine
      .evaluate(seed("gummies-100", "merchant-2", &GUMMIES_EDITED))
      .await
      .unwrap();
    engine.evaluate(seed("vape", "merchant-1", &VAPE)).await.unwrap();
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
    assert_eq!(neighbors[0].tenant.as_deref(), Some("merchant-2"));
  }

  #[tokio::test]
  async fn the_search_score_is_the_score_evaluate_would_give() {
    let engine = seeded().await;
    let neighbors = engine.find_similar(similar_to_id("gummies")).await.unwrap();
    let preview = EvaluateInput {
      mode: WriteMode::DryRun,
      record: RecordInput {
        tenant: Some("merchant-1".to_owned()),
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

    let tenant = FindSimilarInput {
      tenant: Some("merchant-1".to_owned()),
      ..around_gummies()
    };
    assert_eq!(ids(&engine.find_similar(tenant).await.unwrap()), ["vape"]);

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
    let owner = || Some("merchant-1".to_owned());
    assert!(engine.delete("gummies".to_owned(), owner()).await.unwrap());
    assert!(!engine.delete("gummies".to_owned(), owner()).await.unwrap());
    assert!(
      !engine.delete("gummies".to_owned(), None).await.unwrap(),
      "nothing stored, nothing to check a tenant against"
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
}
