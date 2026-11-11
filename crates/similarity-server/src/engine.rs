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

use std::hash::{BuildHasher, RandomState};
use std::time::{SystemTime, UNIX_EPOCH};

use similarity_core::score::{self, FieldScore};
use similarity_core::{FieldChange, Policy, PreparedRecord, RecordVectors, Schema};
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordInput {
  /// The caller's identifier, if the record has one yet.
  pub id: Option<String>,
  /// The record's owner.
  pub tenant: Option<String>,
  /// Field name and text pairs.
  pub fields: Vec<(String, String)>,
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
  /// Why the record was rejected.
  pub reasons: Vec<String>,
  /// The record's composite vector.
  pub vector: Vec<f32>,
  /// The record's revision after this call; zero if it is not stored.
  pub revision: u64,
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
    }
  }
}

/// What to search around.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// The service's behavior.
#[derive(Debug)]
pub struct Engine<S, E> {
  store: S,
  embedder: E,
  schema: Schema,
  policy: Policy,
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
      locks: (0..LOCK_STRIPES).map(|_| Mutex::new(())).collect(),
      lock_hasher: RandomState::new(),
    }
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
    Ok(PreparedRecord::new(&self.schema, fields)?)
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
      Some(threshold) => Policy::new(threshold, self.policy.anchor_threshold)?,
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

    if let Some(baseline) = baseline {
      return self.revise(id, tenant, input.mode, &policy, baseline, candidate).await;
    }
    let stored = input.mode != WriteMode::DryRun;
    if stored {
      let record = StoredRecord {
        id,
        tenant,
        anchor: candidate.composite.clone(),
        vectors: candidate.clone(),
        revision: 1,
        updated_at: now(),
      };
      self.store.put(&record).await?;
    }
    Ok(EvaluateOutput::without_baseline(
      Verdict::Created,
      &policy,
      stored,
      candidate.composite,
    ))
  }

  /// Judges `candidate` against `baseline` and writes it back if the policy and the mode allow.
  async fn revise(
    &self,
    id: String,
    tenant: Option<String>,
    mode: WriteMode,
    policy: &Policy,
    baseline: StoredRecord,
    candidate: RecordVectors,
  ) -> Result<EvaluateOutput, EngineError> {
    let evaluation = score::evaluate(
      &self.schema,
      policy,
      self.embedder.dimension(),
      &baseline.vectors,
      Some(&baseline.anchor),
      &candidate,
    )?;
    let accepted = evaluation.accepted();
    // Re-submitting exactly what is stored is common (a save with no edits, a retried request) and must not churn
    // the store or the revision counter.
    let unchanged = |field: &FieldScore| field.change == FieldChange::Unchanged;
    let identical = evaluation.comparison.fields.iter().all(unchanged)
      && candidate.hashes == baseline.vectors.hashes
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
      let anchor = if mode == WriteMode::Force {
        candidate.composite.clone()
      } else {
        score::extend_anchor(
          &self.schema,
          self.embedder.dimension(),
          &baseline.anchor,
          &candidate.composite,
        )?
      };
      let record = StoredRecord {
        id,
        tenant,
        anchor,
        vectors: candidate.clone(),
        revision,
        updated_at: now(),
      };
      self.store.put(&record).await?;
    }

    Ok(EvaluateOutput {
      verdict: if accepted { Verdict::Accepted } else { Verdict::Rejected },
      similarity: Some(evaluation.comparison.similarity),
      threshold: policy.threshold,
      fields: evaluation.comparison.fields,
      anchor_similarity: evaluation.anchor_similarity,
      stored: write,
      reasons: evaluation.reasons.iter().map(ToString::to_string).collect(),
      vector: candidate.composite,
      revision,
    })
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

  use similarity_core::FieldSpec;
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
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    Engine::new(MemoryStore::new(layout, dimension), embedder, schema(), policy)
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
        .any(|reason| reason.contains("below the threshold")),
      "{:?}",
      recycled.reasons
    );
    assert!(
      recycled.reasons.iter().any(|reason| reason.contains("`name`")),
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
    assert!(output.reasons[0].contains("`name`"), "{:?}", output.reasons);
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
        .any(|reason| reason.contains("original record"))
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
