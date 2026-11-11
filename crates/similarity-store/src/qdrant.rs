//! One Qdrant collection, one point per record.
//!
//! Each point carries two named vectors of the same width and a small payload:
//!
//! - `current`: the composite of the accepted baseline. HNSW-indexed; this is what searches match.
//! - `anchor`: the composite of the record as first stored. Only ever read back by id, so it has no HNSW graph
//!   (`m = 0`) and lives on disk, costing no memory.
//! - payload: the caller's id, the tenant, the layout id, the per-field text fingerprints, a revision counter, and
//!   a timestamp.
//!
//! Qdrant point ids must be integers or UUIDs, and record ids are arbitrary strings, so the point id is the UUIDv5
//! of the record id under a fixed namespace. The mapping is deterministic, so a record is always one point no
//! matter which server instance writes it.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use qdrant_client::qdrant::vector_output::Vector as StoredVector;
use qdrant_client::qdrant::vectors_config::Config as VectorsLayout;
use qdrant_client::qdrant::{
  Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DeletePointsBuilder, Distance, FieldType,
  Filter, GetPointsBuilder, HnswConfigDiffBuilder, KeywordIndexParamsBuilder, NamedVectors, PointId, PointStruct,
  PointsIdsList, QueryPointsBuilder, UpsertPointsBuilder, Value, VectorParamsBuilder, VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use serde::{Deserialize, Serialize};
use similarity_core::RecordVectors;
use uuid::Uuid;

use crate::{Neighbor, Result, SearchQuery, StoreError, StoredRecord, VectorStore};

const CURRENT: &str = "current";
const ANCHOR: &str = "anchor";
const TENANT_KEY: &str = "tenant";
const LAYOUT_KEY: &str = "layout";

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
  tenant: Option<String>,
  layout: String,
  #[serde(default)]
  hashes: BTreeMap<String, String>,
  #[serde(default)]
  revision: u64,
  #[serde(default)]
  updated_at: u64,
}

/// A [`VectorStore`] backed by a Qdrant collection.
#[derive(Clone)]
pub struct QdrantStore {
  client: Qdrant,
  collection: String,
  layout: String,
  dimension: usize,
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
    let store = Self {
      client,
      collection: settings.collection.clone(),
      layout: layout.into(),
      dimension,
    };
    store.ensure_collection().await?;
    Ok(store)
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
  /// On a new collection this happens before any point is written, so the HNSW graph is built tenant-aware from
  /// the start.
  async fn ensure_indexes(&self) -> Result<()> {
    self
      .client
      .create_field_index(
        CreateFieldIndexCollectionBuilder::new(self.collection.as_str(), TENANT_KEY, FieldType::Keyword)
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

fn decode_payload(id: &str, payload: HashMap<String, Value>) -> Result<PointPayload> {
  Payload::from(payload)
    .deserialize()
    .map_err(|source| StoreError::Corrupt {
      id: id.to_owned(),
      reason: format!("payload: {source}"),
    })
}

impl VectorStore for QdrantStore {
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
    let named = |name: &str| -> Result<Vec<f32>> {
      match vectors.as_ref().and_then(|vectors| vectors.get_vector_by_name(name)) {
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
    };
    Ok(Some(StoredRecord {
      id: payload.record_id,
      tenant: payload.tenant,
      vectors: RecordVectors {
        composite: named(CURRENT)?,
        hashes: payload.hashes,
      },
      anchor: named(ANCHOR)?,
      revision: payload.revision,
      updated_at: payload.updated_at,
    }))
  }

  async fn put(&self, record: &StoredRecord) -> Result<()> {
    self.check_length("the composite", &record.vectors.composite)?;
    self.check_length("the anchor", &record.anchor)?;

    let payload = serde_json::to_value(PointPayload {
      record_id: record.id.clone(),
      tenant: record.tenant.clone(),
      layout: self.layout.clone(),
      hashes: record.vectors.hashes.clone(),
      revision: record.revision,
      updated_at: record.updated_at,
    })
    .map_err(|source| StoreError::Invalid(format!("payload does not serialize: {source}")))
    .and_then(|value| {
      Payload::try_from(value).map_err(|source| StoreError::Invalid(format!("payload is not an object: {source}")))
    })?;
    let vectors = NamedVectors::default()
      .add_vector(CURRENT, record.vectors.composite.clone())
      .add_vector(ANCHOR, record.anchor.clone());

    self
      .client
      .upsert_points(
        // `wait` makes the write visible to the very next read, which the compare-then-write cycle depends on.
        UpsertPointsBuilder::new(
          self.collection.as_str(),
          vec![PointStruct::new(point_id(&record.id), vectors, payload)],
        )
        .wait(true),
      )
      .await
      .map(|_| ())
      .map_err(|source| unavailable(&source))
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

  async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>> {
    self.check_length("the query vector", query.vector)?;
    let limit = u64::try_from(query.limit).unwrap_or(u64::MAX);
    if limit == 0 {
      return Ok(Vec::new());
    }

    let mut must = vec![Condition::matches(LAYOUT_KEY, self.layout.clone())];
    if let Some(tenant) = query.tenant {
      must.push(Condition::matches(TENANT_KEY, tenant.to_owned()));
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
          tenant: payload.tenant,
          similarity: point.score,
        })
      })
      .collect()
  }
}

#[cfg(test)]
mod tests {
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

  #[test]
  fn a_payload_round_trips_through_qdrant_values() {
    let payload = PointPayload {
      record_id: "product-42".to_owned(),
      tenant: Some("merchant-7".to_owned()),
      layout: "abc-d256-t256|category,name".to_owned(),
      hashes: BTreeMap::from([("name".to_owned(), "00ff".to_owned())]),
      revision: 3,
      updated_at: 1_700_000_000,
    };
    let encoded = Payload::try_from(serde_json::to_value(&payload).unwrap()).unwrap();
    let decoded = decode_payload("product-42", encoded.into()).unwrap();
    assert_eq!(decoded.record_id, "product-42");
    assert_eq!(decoded.tenant.as_deref(), Some("merchant-7"));
    assert_eq!(decoded.hashes["name"], "00ff");
    assert_eq!((decoded.revision, decoded.updated_at), (3, 1_700_000_000));
  }

  #[test]
  fn a_payload_without_a_tenant_decodes_to_none() {
    let encoded = Payload::try_from(serde_json::json!({ "record_id": "p", "layout": "l" })).unwrap();
    let decoded = decode_payload("p", encoded.into()).unwrap();
    assert_eq!(decoded.tenant, None);
    assert!(decoded.hashes.is_empty());
  }

  #[test]
  fn a_payload_missing_its_identity_is_corrupt() {
    let encoded = Payload::try_from(serde_json::json!({ "tenant": "merchant-7" })).unwrap();
    assert!(matches!(
      decode_payload("p", encoded.into()),
      Err(StoreError::Corrupt { .. })
    ));
  }
}
