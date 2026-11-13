//! One Qdrant collection, one point per record.
//!
//! Each point carries two named vectors of the same width and a small payload:
//!
//! - `current`: the composite of the accepted baseline. HNSW-indexed; this is what searches match.
//! - `anchor`: the composite of the record as first stored. Only ever read back by id, so it has no HNSW graph
//!   (`m = 0`) and lives on disk, costing no memory.
//! - payload: the caller's id, the brand, the layout id, the per-field text fingerprints, a revision counter, a
//!   timestamp, and the record's quantities (current and original). Quantities are compared as numbers and are in
//!   no vector, which is why a point written before they existed needs no migration: it reads back with none.
//!
//! A conditional write is an upsert with a filter on the point being replaced: Qdrant applies it only if the
//! stored payload still has the revision and time the writer read. Qdrant does not say whether it did (the
//! answer is `completed` either way), so each conditional write carries a mark of its own in the payload and the
//! point is read back to see whose mark it holds.
//!
//! A scan is Qdrant's scroll: pages in the order of the point ids, payloads only. It lists every point in the
//! collection, and says of each what [`QdrantStore::get`] would make of it.
//!
//! Qdrant point ids must be integers or UUIDs, and record ids are arbitrary strings, so the point id is the UUIDv5
//! of the record id under a fixed namespace. The mapping is deterministic, so a record is always one point no
//! matter which server instance writes it.

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, RandomState};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::vector_output::Vector as StoredVector;
use qdrant_client::qdrant::vectors_config::Config as VectorsLayout;
use qdrant_client::qdrant::{
  Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DeletePointsBuilder, Distance, FieldType,
  Filter, GetPointsBuilder, HnswConfigDiffBuilder, KeywordIndexParamsBuilder, NamedVectors, PointId, PointStruct,
  PointsIdsList, QueryPointsBuilder, RetrievedPoint, ScrollPointsBuilder, UpdateMode, UpsertPointsBuilder, Value,
  VectorParamsBuilder, VectorsConfigBuilder, VectorsOutput, VectorsSelector,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use serde::{Deserialize, Serialize};
use similarity_core::{Quantities, RecordVectors};
use uuid::Uuid;

use crate::{
  Cursor, Entry, Expected, Found, Neighbor, Page, Result, Scan, ScannedRecord, SearchQuery, StoreError, StoredRecord,
  VectorStore, Version,
};

const CURRENT: &str = "current";
const ANCHOR: &str = "anchor";
const BRAND_ID_KEY: &str = "brand_id";
const LAYOUT_KEY: &str = "layout";
// The two payload keys a conditional write filters on. They are the names `PointPayload` serializes under.
const REVISION_KEY: &str = "revision";
const UPDATED_AT_KEY: &str = "updated_at";
// Where a point says which record it is. Read by name only for a point that cannot be decoded.
const RECORD_ID_KEY: &str = "record_id";

/// The UUIDv5 namespace record ids are hashed under. Changing it orphans every stored point.
const POINT_NAMESPACE: Uuid = Uuid::from_u128(0x7d0c_3f5e_9a41_4b6e_8c1d_52a7_e3b9_604f);

/// How to reach Qdrant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QdrantSettings {
  /// The gRPC endpoint, for example `http://localhost:6334`.
  pub url: String,
  /// The API key, for Qdrant Cloud or a secured deployment.
  pub api_key: Option<String>,
  /// The collection to use. Created on first start if it does not exist.
  pub collection: String,
  /// The deadline for any single request.
  pub timeout: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
struct PointPayload {
  record_id: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  brand_id: Option<String>,
  layout: String,
  #[serde(default)]
  hashes: BTreeMap<String, String>,
  #[serde(default)]
  revision: u64,
  #[serde(default)]
  updated_at: u64,
  // Absent on every point written before quantities existed, and left out of a record that has none.
  #[serde(default, skip_serializing_if = "Quantities::is_empty")]
  quantities: Quantities,
  #[serde(default, skip_serializing_if = "Quantities::is_empty")]
  anchor_quantities: Quantities,
  // The mark of the conditional write that stored the point, by which that writer recognizes it when it reads
  // the point back. Absent on a point stored by a plain write, and never read for anything else.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  write: Option<String>,
}

/// Makes the marks conditional writes are recognized by: no two alike from one store, and none alike between
/// two processes short of a 64-bit accident. The keys are random for each store, and the counter never repeats.
#[derive(Debug)]
struct Marks {
  keys: RandomState,
  next: AtomicU64,
}

impl Marks {
  fn new() -> Self {
    Self {
      keys: RandomState::new(),
      next: AtomicU64::new(0),
    }
  }

  fn next(&self) -> String {
    format!("{:016x}", self.keys.hash_one(self.next.fetch_add(1, Ordering::Relaxed)))
  }
}

/// A [`VectorStore`] backed by a Qdrant collection.
#[derive(Clone)]
pub struct QdrantStore {
  client: Qdrant,
  collection: String,
  layout: String,
  dimension: usize,
  marks: Arc<Marks>,
}

impl std::fmt::Debug for QdrantStore {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("QdrantStore")
      .field("collection", &self.collection)
      .field("layout", &self.layout)
      .field("dimension", &self.dimension)
      .finish_non_exhaustive()
  }
}

fn unavailable(source: &QdrantError) -> StoreError {
  StoreError::Unavailable(source.to_string())
}

fn point_id(record_id: &str) -> PointId {
  PointId::from(Uuid::new_v5(&POINT_NAMESPACE, record_id.as_bytes()).to_string())
}

impl QdrantStore {
  /// Connects to Qdrant and makes sure the collection exists and can hold composites of `dimension` components.
  ///
  /// A missing collection is created. An existing one is verified, never altered: if its vectors are the wrong
  /// width the server refuses to start rather than write vectors Qdrant would reject one request at a time.
  ///
  /// # Errors
  ///
  /// [`StoreError::Unavailable`] if Qdrant cannot be reached, [`StoreError::Collection`] if the collection exists
  /// with an incompatible vector configuration, [`StoreError::Invalid`] if `dimension` is zero.
  pub async fn connect(settings: &QdrantSettings, layout: impl Into<String>, dimension: usize) -> Result<Self> {
    let store = Self::client(settings, layout, dimension)?;
    store.ensure_collection().await?;
    Ok(store)
  }

  /// Connects to a collection that is already there, and changes nothing about it: for looking at a store, not
  /// for serving from one. Where [`Self::connect`] creates a missing collection and its payload indexes, this
  /// refuses a collection that does not exist and creates no index. The collection is verified as `connect`
  /// verifies it.
  ///
  /// What comes back is a whole store and could be written to. Whoever opens one this way should hold it as a
  /// [`Scan`], which cannot.
  ///
  /// # Errors
  ///
  /// [`StoreError::Unavailable`] if Qdrant cannot be reached, [`StoreError::Collection`] if the collection does
  /// not exist or exists with an incompatible vector configuration, [`StoreError::Invalid`] if `dimension` is
  /// zero.
  pub async fn open(settings: &QdrantSettings, layout: impl Into<String>, dimension: usize) -> Result<Self> {
    let store = Self::client(settings, layout, dimension)?;
    if !store.collection_exists().await? {
      return Err(StoreError::Collection {
        collection: store.collection,
        reason: "it does not exist".to_owned(),
      });
    }
    store.verify_collection().await?;
    Ok(store)
  }

  /// A store for `settings` that has not spoken to Qdrant yet.
  fn client(settings: &QdrantSettings, layout: impl Into<String>, dimension: usize) -> Result<Self> {
    if dimension == 0 {
      return Err(StoreError::Invalid(
        "the composite dimension must be positive".to_owned(),
      ));
    }
    let mut builder = Qdrant::from_url(&settings.url)
      .timeout(settings.timeout)
      .connect_timeout(settings.timeout);
    if let Some(key) = &settings.api_key {
      builder = builder.api_key(key.as_str());
    }
    let client = builder.build().map_err(|source| unavailable(&source))?;
    Ok(Self {
      client,
      collection: settings.collection.clone(),
      layout: layout.into(),
      dimension,
      marks: Arc::new(Marks::new()),
    })
  }

  async fn ensure_collection(&self) -> Result<()> {
    if !self.collection_exists().await? {
      if let Err(source) = self.create_collection().await {
        // Two instances starting together both see "missing" and both create; the loser's error is only real if
        // the collection still is not there.
        if !self.collection_exists().await? {
          return Err(source);
        }
      } else {
        tracing::info!(collection = %self.collection, dimension = self.dimension, "created Qdrant collection");
      }
    }
    self.verify_collection().await?;
    self.ensure_indexes().await
  }

  async fn collection_exists(&self) -> Result<bool> {
    self
      .client
      .collection_exists(self.collection.as_str())
      .await
      .map_err(|source| unavailable(&source))
  }

  async fn create_collection(&self) -> Result<()> {
    let size = u64::try_from(self.dimension).map_err(|_| StoreError::Invalid("dimension overflows u64".to_owned()))?;
    let mut vectors = VectorsConfigBuilder::default();
    vectors.add_named_vector_params(CURRENT, VectorParamsBuilder::new(size, Distance::Cosine));
    vectors.add_named_vector_params(
      ANCHOR,
      VectorParamsBuilder::new(size, Distance::Cosine)
        .on_disk(true)
        .hnsw_config(HnswConfigDiffBuilder::default().m(0)),
    );
    self
      .client
      .create_collection(CreateCollectionBuilder::new(self.collection.as_str()).vectors_config(vectors))
      .await
      .map_err(|source| unavailable(&source))?;
    Ok(())
  }

  /// Creates the payload indexes every filter this store issues relies on, if they are not there yet.
  ///
  /// This runs on every connect rather than only when the collection is created. Creating an index that exists is
  /// a no-op for Qdrant, whereas tying it to collection creation would leave a collection whose first start failed
  /// between the two steps (or lost a creation race) without indexes for good, with nothing ever noticing: filters
  /// still work unindexed, only slowly.
  ///
  /// On a new collection this happens before any point is written, so the HNSW graph is built with the brand as
  /// its partition from the start (`is_tenant` is Qdrant's name for a key a collection is partitioned by).
  async fn ensure_indexes(&self) -> Result<()> {
    self
      .client
      .create_field_index(
        CreateFieldIndexCollectionBuilder::new(self.collection.as_str(), BRAND_ID_KEY, FieldType::Keyword)
          .field_index_params(KeywordIndexParamsBuilder::default().is_tenant(true))
          .wait(true),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    self
      .client
      .create_field_index(
        CreateFieldIndexCollectionBuilder::new(self.collection.as_str(), LAYOUT_KEY, FieldType::Keyword).wait(true),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    Ok(())
  }

  async fn verify_collection(&self) -> Result<()> {
    let unusable = |reason: String| StoreError::Collection {
      collection: self.collection.clone(),
      reason,
    };
    let info = self
      .client
      .collection_info(self.collection.as_str())
      .await
      .map_err(|source| unavailable(&source))?;
    let layout = info
      .result
      .and_then(|result| result.config)
      .and_then(|config| config.params)
      .and_then(|params| params.vectors_config)
      .and_then(|vectors| vectors.config);
    let Some(VectorsLayout::ParamsMap(named)) = layout else {
      return Err(unusable(format!(
        "it does not use named vectors; expected `{CURRENT}` and `{ANCHOR}`"
      )));
    };
    for name in [CURRENT, ANCHOR] {
      let Some(params) = named.map.get(name) else {
        return Err(unusable(format!("it has no vector named `{name}`")));
      };
      if usize::try_from(params.size).ok() != Some(self.dimension) {
        return Err(unusable(format!(
          "vector `{name}` has {} components, but this server's model and schema produce {}; point the server at \
           a new collection and re-index",
          params.size, self.dimension
        )));
      }
      if params.distance != i32::from(Distance::Cosine) {
        return Err(unusable(format!("vector `{name}` does not use cosine distance")));
      }
    }
    Ok(())
  }

  /// Drops the whole collection. For tests and for re-indexing from scratch.
  ///
  /// # Errors
  ///
  /// [`StoreError::Unavailable`] if the request fails.
  pub async fn delete_collection(&self) -> Result<()> {
    self
      .client
      .delete_collection(self.collection.as_str())
      .await
      .map(|_| ())
      .map_err(|source| unavailable(&source))
  }

  fn check_length(&self, what: &str, vector: &[f32]) -> Result<()> {
    if vector.len() == self.dimension {
      Ok(())
    } else {
      Err(StoreError::Invalid(format!(
        "{what} has {} components, but collection `{}` holds vectors of {}",
        vector.len(),
        self.collection,
        self.dimension
      )))
    }
  }
}

impl QdrantStore {
  /// The dense vector `name` of the point holding record `id`, checked to have this store's width.
  fn dense(&self, id: &str, vectors: Option<&VectorsOutput>, name: &str) -> Result<Vec<f32>> {
    match vectors.and_then(|vectors| vectors.get_vector_by_name(name)) {
      Some(StoredVector::Dense(dense)) if dense.data.len() == self.dimension => Ok(dense.data),
      Some(StoredVector::Dense(dense)) => Err(StoreError::Corrupt {
        id: id.to_owned(),
        reason: format!(
          "vector `{name}` has {} components, expected {}",
          dense.data.len(),
          self.dimension
        ),
      }),
      _ => Err(StoreError::Corrupt {
        id: id.to_owned(),
        reason: format!("dense vector `{name}` is missing"),
      }),
    }
  }

  /// The point a record is stored as. `write` is the mark of a conditional write, and `None` for a plain one.
  fn point(&self, record: &StoredRecord, write: Option<String>) -> Result<PointStruct> {
    self.check_length("the composite", &record.vectors.composite)?;
    self.check_length("the anchor", &record.anchor)?;

    let payload = serde_json::to_value(PointPayload {
      record_id: record.id.clone(),
      brand_id: record.brand_id.clone(),
      layout: self.layout.clone(),
      hashes: record.vectors.hashes.clone(),
      revision: record.revision,
      updated_at: record.updated_at,
      quantities: record.quantities.clone(),
      anchor_quantities: record.anchor_quantities.clone(),
      write,
    })
    .map_err(|source| StoreError::Invalid(format!("payload does not serialize: {source}")))
    .and_then(|value| {
      Payload::try_from(value).map_err(|source| StoreError::Invalid(format!("payload is not an object: {source}")))
    })?;
    let vectors = NamedVectors::default()
      .add_vector(CURRENT, record.vectors.composite.clone())
      .add_vector(ANCHOR, record.anchor.clone());
    Ok(PointStruct::new(point_id(&record.id), vectors, payload))
  }

  /// What a point's payload must say for it to be `version` of a record of this layout.
  fn at(&self, version: Version) -> Result<Vec<Condition>> {
    let stored = |what: &str, value: u64| {
      i64::try_from(value)
        .map_err(|_| StoreError::Invalid(format!("{what} {value} is beyond what a stored record can hold")))
    };
    Ok(vec![
      Condition::matches(LAYOUT_KEY, self.layout.clone()),
      Condition::matches(REVISION_KEY, stored("the revision", version.revision)?),
      Condition::matches(UPDATED_AT_KEY, stored("the time", version.updated_at)?),
    ])
  }

  /// A point's payload without its vectors, or `None` if there is no such point.
  async fn payload_of(&self, id: &str) -> Result<Option<PointPayload>> {
    let response = self
      .client
      .get_points(
        GetPointsBuilder::new(self.collection.as_str(), vec![point_id(id)])
          .with_payload(true)
          .with_vectors(false),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    response
      .result
      .into_iter()
      .next()
      .map(|point| decode_payload(id, point.payload))
      .transpose()
  }

  /// The request both kinds of search send, or `None` for a search that asks for nothing. One place, so that
  /// they cannot differ in what they match.
  fn query_request(&self, query: SearchQuery<'_>) -> Result<Option<QueryPointsBuilder>> {
    self.check_length("the query vector", query.vector)?;
    let limit = u64::try_from(query.limit).unwrap_or(u64::MAX);
    if limit == 0 {
      return Ok(None);
    }

    let mut must = vec![Condition::matches(LAYOUT_KEY, self.layout.clone())];
    if let Some(brand_id) = query.brand_id {
      must.push(Condition::matches(BRAND_ID_KEY, brand_id.to_owned()));
    }
    let mut filter = Filter::must(must);
    if let Some(excluded) = query.exclude_id {
      filter.must_not.push(Condition::has_id([point_id(excluded)]));
    }

    let mut request = QueryPointsBuilder::new(self.collection.as_str())
      .query(query.vector.to_vec())
      .using(CURRENT)
      .filter(filter)
      .limit(limit)
      .with_payload(true);
    if let Some(minimum) = query.min_similarity {
      request = request.score_threshold(minimum);
    }
    Ok(Some(request))
  }
}

fn decode_payload(id: &str, payload: HashMap<String, Value>) -> Result<PointPayload> {
  // A point from before 2026-11-12 keeps its brand under `tenant`. Read as it is, it would be a record that
  // belongs to nobody, and the rule that a request must name a record's brand would let anyone through.
  if payload.contains_key("tenant") {
    return Err(StoreError::Corrupt {
      id: id.to_owned(),
      reason: "it was stored when a record's brand was kept under `tenant`, which is `brand_id` now; import the \
               collection again"
        .to_owned(),
    });
  }
  Payload::from(payload)
    .deserialize()
    .map_err(|source| StoreError::Corrupt {
      id: id.to_owned(),
      reason: format!("payload: {source}"),
    })
}

/// A point's id as a cursor. The kind of id is kept: a collection can hold points this service did not write, and
/// theirs may be numbers.
fn cursor_of(point: &PointId) -> Option<Cursor> {
  match point.point_id_options.as_ref()? {
    PointIdOptions::Uuid(uuid) => Some(Cursor(format!("u{uuid}"))),
    PointIdOptions::Num(number) => Some(Cursor(format!("n{number}"))),
  }
}

/// The point a cursor of this store names, or `None` for a cursor some other store gave.
fn point_of(cursor: &Cursor) -> Option<PointId> {
  let Cursor(text) = cursor;
  if let Some(uuid) = text.strip_prefix('u') {
    Some(PointId::from(uuid.to_owned()))
  } else {
    text.strip_prefix('n')?.parse::<u64>().ok().map(PointId::from)
  }
}

/// What a scan says of one point, for a store of `layout`: what [`QdrantStore::get`] would make of it, without its
/// vectors and without failing.
fn entry(layout: &str, point: RetrievedPoint) -> Entry {
  // Looked up before decoding, so that a point that cannot be decoded can still say which record it is. It is
  // only believed if the point is the one that record is kept at: anything can be written into a payload, and a
  // point `get` would never look at must not be taken for the record it names.
  let named = match point.payload.get(RECORD_ID_KEY).and_then(|value| value.kind.as_ref()) {
    Some(Kind::StringValue(id)) => Some(id.clone()),
    _ => None,
  };
  let Some(id) = named.filter(|id| point.id == Some(point_id(id))) else {
    return Entry::Unreadable {
      id: None,
      reason: match point.payload.get(RECORD_ID_KEY).and_then(|value| value.kind.as_ref()) {
        Some(Kind::StringValue(id)) => format!("it says it is record `{id}`, which is kept at another point"),
        _ => "it does not say which record it is".to_owned(),
      },
    };
  };
  let payload = match decode_payload(&id, point.payload) {
    Ok(payload) => payload,
    Err(StoreError::Corrupt { reason, .. }) => return Entry::Unreadable { id: Some(id), reason },
    Err(other) => {
      return Entry::Unreadable {
        id: Some(id),
        reason: other.to_string(),
      };
    }
  };
  if payload.layout != layout {
    return Entry::OtherLayout {
      id,
      layout: payload.layout,
    };
  }
  Entry::Record(ScannedRecord {
    id,
    brand_id: payload.brand_id,
    hashes: payload.hashes,
    quantities: payload.quantities,
    anchor_quantities: payload.anchor_quantities,
    revision: payload.revision,
    updated_at: payload.updated_at,
  })
}

impl Scan for QdrantStore {
  /// In the order of the point ids, which is Qdrant's own and no order of the record ids.
  async fn scan(&self, from: Option<&Cursor>, limit: usize) -> Result<Page> {
    if limit == 0 {
      return Err(StoreError::Invalid(
        "a scan needs a page of at least one entry".to_owned(),
      ));
    }
    let mut request = ScrollPointsBuilder::new(self.collection.as_str())
      .limit(u32::try_from(limit).unwrap_or(u32::MAX))
      .with_payload(true)
      .with_vectors(false);
    if let Some(cursor) = from {
      let first = point_of(cursor)
        .ok_or_else(|| StoreError::Invalid("the cursor is not one a scan of this store gave".to_owned()))?;
      request = request.offset(first);
    }
    let response = self
      .client
      .scroll(request)
      .await
      .map_err(|source| unavailable(&source))?;
    let next = match response.next_page_offset {
      None => None,
      // Ending the scan here would report everything after this page as not stored.
      Some(point) => Some(
        cursor_of(&point)
          .ok_or_else(|| StoreError::Unavailable("Qdrant named a next page by a point with no id".to_owned()))?,
      ),
    };
    Ok(Page {
      entries: response
        .result
        .into_iter()
        .map(|point| entry(&self.layout, point))
        .collect(),
      next,
    })
  }
}

impl VectorStore for QdrantStore {
  async fn check(&self) -> Result<()> {
    // One small call that needs the server to answer and the collection to be there. A collection can be
    // deleted under a running server, and every request after that fails: that is not ready either.
    if self.collection_exists().await? {
      Ok(())
    } else {
      Err(StoreError::Collection {
        collection: self.collection.clone(),
        reason: "it no longer exists".to_owned(),
      })
    }
  }

  async fn get(&self, id: &str) -> Result<Option<StoredRecord>> {
    let response = self
      .client
      .get_points(
        GetPointsBuilder::new(self.collection.as_str(), vec![point_id(id)])
          .with_payload(true)
          .with_vectors(true),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    let Some(point) = response.result.into_iter().next() else {
      return Ok(None);
    };

    let payload = decode_payload(id, point.payload)?;
    if payload.record_id != id {
      // Only reachable through a UUIDv5 collision or a point written by something other than this service.
      return Err(StoreError::Corrupt {
        id: id.to_owned(),
        reason: format!("the point holds record `{}`", payload.record_id),
      });
    }
    if payload.layout != self.layout {
      return Err(StoreError::LayoutMismatch {
        id: id.to_owned(),
        stored: payload.layout,
        expected: self.layout.clone(),
      });
    }

    let vectors = point.vectors;
    let named = |name: &str| self.dense(id, vectors.as_ref(), name);
    Ok(Some(StoredRecord {
      id: payload.record_id,
      brand_id: payload.brand_id,
      vectors: RecordVectors {
        composite: named(CURRENT)?,
        hashes: payload.hashes,
      },
      anchor: named(ANCHOR)?,
      quantities: payload.quantities,
      anchor_quantities: payload.anchor_quantities,
      revision: payload.revision,
      updated_at: payload.updated_at,
    }))
  }

  async fn put(&self, record: &StoredRecord) -> Result<()> {
    let point = self.point(record, None)?;
    self
      .client
      .upsert_points(
        // `wait` makes the write visible to the very next read, which the compare-then-write cycle depends on.
        UpsertPointsBuilder::new(self.collection.as_str(), vec![point]).wait(true),
      )
      .await
      .map(|_| ())
      .map_err(|source| unavailable(&source))
  }

  async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool> {
    let mark = self.marks.next();
    let point = self.point(record, Some(mark.clone()))?;
    let request = UpsertPointsBuilder::new(self.collection.as_str(), vec![point]).wait(true);
    let request = match expected {
      // Never over a point that is there, whatever it holds.
      Expected::Absent => request.update_mode(UpdateMode::InsertOnly),
      // Only over the point that was read, and never as a new point: a record deleted since stays deleted.
      Expected::Version(version) => request
        .update_mode(UpdateMode::UpdateOnly)
        .update_filter(Filter::must(self.at(version)?)),
    };
    self
      .client
      .upsert_points(request)
      .await
      .map_err(|source| unavailable(&source))?;
    // Qdrant answers `completed` whether the condition held or not. The point says which: it carries this
    // write's mark only if this write is what stored it.
    let stored = self.payload_of(&record.id).await?;
    Ok(stored.and_then(|payload| payload.write).as_deref() == Some(mark.as_str()))
  }

  async fn delete(&self, id: &str) -> Result<bool> {
    let existing = self
      .client
      .get_points(
        GetPointsBuilder::new(self.collection.as_str(), vec![point_id(id)])
          .with_payload(false)
          .with_vectors(false),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    if existing.result.is_empty() {
      return Ok(false);
    }
    self
      .client
      .delete_points(
        DeletePointsBuilder::new(self.collection.as_str())
          .points(PointsIdsList {
            ids: vec![point_id(id)],
          })
          .wait(true),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    Ok(true)
  }

  async fn delete_if(&self, id: &str, version: Version) -> Result<bool> {
    let mut only = self.at(version)?;
    only.push(Condition::has_id([point_id(id)]));
    self
      .client
      .delete_points(
        DeletePointsBuilder::new(self.collection.as_str())
          .points(Filter::must(only))
          .wait(true),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    // As for a conditional write, the answer does not say whether anything matched: looking does.
    let remaining = self
      .client
      .get_points(
        GetPointsBuilder::new(self.collection.as_str(), vec![point_id(id)])
          .with_payload(false)
          .with_vectors(false),
      )
      .await
      .map_err(|source| unavailable(&source))?;
    Ok(remaining.result.is_empty())
  }

  async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>> {
    let Some(request) = self.query_request(query)? else {
      return Ok(Vec::new());
    };
    let response = self
      .client
      .query(request)
      .await
      .map_err(|source| unavailable(&source))?;

    response
      .result
      .into_iter()
      .map(|point| {
        let payload = decode_payload("<search result>", point.payload)?;
        Ok(Neighbor {
          id: payload.record_id,
          brand_id: payload.brand_id,
          similarity: point.score,
        })
      })
      .collect()
  }

  async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>> {
    let Some(request) = self.query_request(query)? else {
      return Ok(Vec::new());
    };
    // Only the current vector: the anchor is not needed to score a result, and stays where it is.
    let request = request.with_vectors(VectorsSelector {
      names: vec![CURRENT.to_owned()],
    });
    let response = self
      .client
      .query(request)
      .await
      .map_err(|source| unavailable(&source))?;

    response
      .result
      .into_iter()
      .map(|point| {
        let payload = decode_payload("<search result>", point.payload)?;
        let composite = self.dense(&payload.record_id, point.vectors.as_ref(), CURRENT)?;
        Ok(Found {
          neighbor: Neighbor {
            id: payload.record_id,
            brand_id: payload.brand_id,
            similarity: point.score,
          },
          vectors: RecordVectors {
            composite,
            hashes: payload.hashes,
          },
          quantities: payload.quantities,
        })
      })
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use similarity_core::Quantity;

  use super::*;

  #[test]
  fn a_record_id_always_maps_to_the_same_point() {
    assert_eq!(point_id("product-42"), point_id("product-42"));
    assert_ne!(point_id("product-42"), point_id("product-43"));
    assert_ne!(
      point_id("42"),
      point_id(" 42"),
      "ids are opaque: no trimming, no parsing"
    );
  }

  fn stored(record_id: &str, payload: serde_json::Value) -> RetrievedPoint {
    RetrievedPoint {
      id: Some(point_id(record_id)),
      payload: Payload::try_from(payload).unwrap().into(),
      ..RetrievedPoint::default()
    }
  }

  #[test]
  fn a_scanned_point_of_this_layout_is_its_record_without_its_vectors() {
    let point = stored(
      "product-42",
      serde_json::json!({
        "record_id": "product-42",
        "brand_id": "brand-7",
        "layout": "model-a|name",
        "hashes": {"name": "00ff"},
        "revision": 3,
        "updated_at": 1_700_000_000,
        "quantities": {"weight": {"min": 3.5, "max": 3.5, "unit": "g"}},
        "anchor_quantities": {"weight": {"min": 1.0, "max": 1.0, "unit": "g"}},
        "write": "00000000deadbeef",
      }),
    );
    assert_eq!(
      entry("model-a|name", point),
      Entry::Record(ScannedRecord {
        id: "product-42".to_owned(),
        brand_id: Some("brand-7".to_owned()),
        hashes: BTreeMap::from([("name".to_owned(), "00ff".to_owned())]),
        quantities: Quantities::from([("weight".to_owned(), Quantity::single(3.5, "g"))]),
        anchor_quantities: Quantities::from([("weight".to_owned(), Quantity::single(1.0, "g"))]),
        revision: 3,
        updated_at: 1_700_000_000,
      })
    );
  }

  #[test]
  fn a_scanned_point_of_another_layout_is_named_and_not_read() {
    let point = stored(
      "product-42",
      serde_json::json!({"record_id": "product-42", "layout": "model-b|name", "hashes": {"name": "00ff"}}),
    );
    assert_eq!(
      entry("model-a|name", point),
      Entry::OtherLayout {
        id: "product-42".to_owned(),
        layout: "model-b|name".to_owned()
      }
    );
  }

  #[test]
  fn a_point_that_keeps_its_brand_under_the_old_name_is_unreadable_and_still_says_which_record() {
    let point = stored(
      "product-42",
      serde_json::json!({"record_id": "product-42", "tenant": "brand-7", "layout": "model-a|name"}),
    );
    match entry("model-a|name", point) {
      Entry::Unreadable { id, reason } => {
        assert_eq!(id.as_deref(), Some("product-42"));
        assert!(reason.contains("`tenant`"), "{reason}");
      }
      other => panic!("expected an unreadable point, got {other:?}"),
    }
  }

  #[test]
  fn a_point_that_does_not_say_which_record_it_is_is_unreadable() {
    let mut point = stored("product-42", serde_json::json!({"note": "written by something else"}));
    point.id = Some(PointId::from(7));
    assert_eq!(
      entry("model-a|name", point),
      Entry::Unreadable {
        id: None,
        reason: "it does not say which record it is".to_owned()
      }
    );
  }

  #[test]
  fn a_point_that_is_not_where_its_record_id_puts_it_is_unreadable_and_is_not_taken_for_that_record() {
    // `get` looks for `product-42` at another point. Whatever is there is that record, and this is not.
    let mut point = stored(
      "product-42",
      serde_json::json!({"record_id": "product-42", "layout": "model-a|name"}),
    );
    point.id = Some(point_id("product-43"));
    assert_eq!(
      entry("model-a|name", point),
      Entry::Unreadable {
        id: None,
        reason: "it says it is record `product-42`, which is kept at another point".to_owned()
      }
    );
  }

  #[test]
  fn a_cursor_names_a_point_whichever_kind_of_id_it_has() {
    for point in [point_id("product-42"), PointId::from(7), PointId::from(u64::MAX)] {
      assert_eq!(point_of(&cursor_of(&point).unwrap()), Some(point));
    }
    assert_ne!(cursor_of(&PointId::from(7)), cursor_of(&PointId::from("7".to_owned())));
    // What the in-memory store gives, a record id, is not a point.
    assert_eq!(point_of(&Cursor("product-42".to_owned())), None);
    assert_eq!(point_of(&Cursor("n-7".to_owned())), None);
    assert_eq!(cursor_of(&PointId::default()), None);
  }

  #[test]
  fn a_payload_round_trips_through_qdrant_values() {
    let payload = PointPayload {
      record_id: "product-42".to_owned(),
      brand_id: Some("brand-7".to_owned()),
      layout: "abc-d256-t256|category,name".to_owned(),
      hashes: BTreeMap::from([("name".to_owned(), "00ff".to_owned())]),
      revision: 3,
      updated_at: 1_700_000_000,
      quantities: Quantities::from([
        ("weight_per_item".to_owned(), Quantity::single(3.5, "g")),
        // A whole number, which JSON and Qdrant would both happily turn into an integer.
        ("thc_per_item".to_owned(), Quantity::range(18.0, 22.0, "%")),
      ]),
      anchor_quantities: Quantities::from([("weight_per_item".to_owned(), Quantity::single(1.0, "g"))]),
      write: Some("00000000deadbeef".to_owned()),
    };
    let encoded = Payload::try_from(serde_json::to_value(&payload).unwrap()).unwrap();
    // The filter of a conditional write names these two keys: they have to be the ones the payload is written
    // under, as whole numbers.
    let written: HashMap<String, Value> = encoded.clone().into();
    assert_eq!(written[REVISION_KEY].as_integer(), Some(3));
    assert_eq!(written[UPDATED_AT_KEY].as_integer(), Some(1_700_000_000));
    let decoded = decode_payload("product-42", encoded.into()).unwrap();
    assert_eq!(decoded.write.as_deref(), Some("00000000deadbeef"));
    assert_eq!(decoded.record_id, "product-42");
    assert_eq!(decoded.brand_id.as_deref(), Some("brand-7"));
    assert_eq!(decoded.hashes["name"], "00ff");
    assert_eq!((decoded.revision, decoded.updated_at), (3, 1_700_000_000));
    assert_eq!(decoded.quantities, payload.quantities);
    assert_eq!(decoded.anchor_quantities, payload.anchor_quantities);
  }

  #[test]
  fn a_payload_written_before_quantities_existed_decodes_with_none() {
    let encoded = Payload::try_from(serde_json::json!({
      "record_id": "p", "layout": "l", "hashes": { "name": "00ff" }, "revision": 2, "updated_at": 1_700_000_000
    }))
    .unwrap();
    let decoded = decode_payload("p", encoded.into()).unwrap();
    assert!(decoded.quantities.is_empty() && decoded.anchor_quantities.is_empty());

    // And a record without quantities writes exactly the payload it always did.
    let written = serde_json::to_value(&decoded).unwrap();
    assert!(written.get("quantities").is_none() && written.get("anchor_quantities").is_none());
  }

  #[test]
  fn a_whole_number_stored_as_an_integer_still_reads_as_a_quantity() {
    let encoded = Payload::try_from(serde_json::json!({
      "record_id": "p", "layout": "l",
      "quantities": { "thc_per_item": { "min": 18, "max": 22, "unit": "%" } }
    }))
    .unwrap();
    let decoded = decode_payload("p", encoded.into()).unwrap();
    assert_eq!(decoded.quantities["thc_per_item"], Quantity::range(18.0, 22.0, "%"));
  }

  #[test]
  fn a_payload_without_a_brand_decodes_to_none() {
    let encoded = Payload::try_from(serde_json::json!({ "record_id": "p", "layout": "l" })).unwrap();
    let decoded = decode_payload("p", encoded.into()).unwrap();
    assert_eq!(decoded.brand_id, None);
    assert!(decoded.hashes.is_empty());
  }

  /// A point from before the brand's key was renamed still says `tenant`. Decoded as it is, it would be a record
  /// that belongs to nobody, which every request may touch.
  #[test]
  fn a_point_that_keeps_its_brand_under_the_name_from_before_is_refused() {
    let old = serde_json::json!({ "record_id": "p", "layout": "l", "tenant": "brand-7", "revision": 1 });
    let error = decode_payload("p", Payload::try_from(old).unwrap().into()).unwrap_err();
    assert!(
      matches!(&error, StoreError::Corrupt { id, reason }
        if id == "p" && reason.contains("`brand_id`") && reason.contains("import")),
      "{error}"
    );
  }

  #[test]
  fn a_payload_missing_its_identity_is_corrupt() {
    let encoded = Payload::try_from(serde_json::json!({ "brand_id": "brand-7" })).unwrap();
    assert!(matches!(
      decode_payload("p", encoded.into()),
      Err(StoreError::Corrupt { .. })
    ));
  }

  #[test]
  fn a_plain_write_carries_no_mark_and_a_point_without_one_decodes() {
    let payload = PointPayload {
      record_id: "p".to_owned(),
      brand_id: None,
      layout: "l".to_owned(),
      hashes: BTreeMap::new(),
      revision: 1,
      updated_at: 1,
      quantities: Quantities::new(),
      anchor_quantities: Quantities::new(),
      write: None,
    };
    let value = serde_json::to_value(&payload).unwrap();
    assert!(value.get("write").is_none(), "{value}");
    let decoded = decode_payload("p", Payload::try_from(value).unwrap().into()).unwrap();
    assert_eq!(decoded.write, None);
  }

  #[test]
  fn no_two_marks_of_a_store_are_alike_and_two_stores_do_not_share_theirs() {
    let (here, there) = (Marks::new(), Marks::new());
    let mut seen = std::collections::HashSet::new();
    for _ in 0..1000 {
      assert!(seen.insert(here.next()));
      assert!(seen.insert(there.next()), "another store's marks are other marks");
    }
    assert!(seen.iter().all(|mark| mark.len() == 16));
  }
}
