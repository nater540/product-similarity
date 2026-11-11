//! The storage contract, run against every implementation.
//!
//! [`MemoryStore`] always runs. [`QdrantStore`] needs a server, so it is ignored by default:
//!
//! ```text
//! just qdrant-up && just test-qdrant
//! ```
//!
//! Running the *same* assertions against both is the point: the in-memory store is what every other test in the
//! workspace trusts, so it must behave like the real thing.

#![allow(
  clippy::unwrap_used,
  clippy::expect_used,
  clippy::panic,
  clippy::indexing_slicing,
  missing_docs
)]
// The contract is one long scenario on purpose: each step depends on the state the previous one left behind.
#![allow(clippy::arithmetic_side_effects, clippy::float_cmp, clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::time::Duration;

use similarity_core::vector::{cosine, normalize};
use similarity_core::{Quantities, Quantity, RecordVectors};
use similarity_store::{MemoryStore, QdrantSettings, QdrantStore, SearchQuery, StoreError, StoredRecord, VectorStore};

const DIM: usize = 6;
const LAYOUT: &str = "model-a|category,name";
const OTHER_LAYOUT: &str = "model-b|category,name";

fn unit(components: [f32; DIM]) -> Vec<f32> {
  let mut vector = components.to_vec();
  assert!(normalize(&mut vector));
  vector
}

fn record(id: &str, tenant: Option<&str>, composite: [f32; DIM]) -> StoredRecord {
  StoredRecord {
    id: id.to_owned(),
    tenant: tenant.map(str::to_owned),
    vectors: RecordVectors {
      composite: unit(composite),
      hashes: BTreeMap::from([("name".to_owned(), format!("hash-of-{id}"))]),
    },
    anchor: unit([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    quantities: Quantities::new(),
    anchor_quantities: Quantities::new(),
    revision: 1,
    updated_at: 1_700_000_000,
  }
}

fn quantities(entries: &[(&str, Quantity)]) -> Quantities {
  entries
    .iter()
    .map(|(name, value)| ((*name).to_owned(), value.clone()))
    .collect()
}

fn query(vector: &[f32]) -> SearchQuery<'_> {
  SearchQuery {
    vector,
    limit: 10,
    tenant: None,
    exclude_id: None,
    min_similarity: None,
  }
}

fn assert_same_direction(left: &[f32], right: &[f32]) {
  assert!(cosine(left, right).unwrap() > 0.999_999, "{left:?} vs {right:?}");
}

/// `store` and `other_layout` are two views of the same underlying data, opened under different layouts.
async fn contract<S: VectorStore>(store: &S, other_layout: &S) {
  // Nothing there yet.
  assert_eq!(store.get("gummies").await.unwrap(), None);
  assert!(!store.delete("gummies").await.unwrap());
  assert!(
    store
      .search(query(&unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
      .await
      .unwrap()
      .is_empty()
  );

  // A write is immediately readable, with everything intact.
  let gummies = record("gummies", Some("merchant-1"), [1.0, 0.1, 0.0, 0.0, 0.0, 0.0]);
  store.put(&gummies).await.unwrap();
  let read = store.get("gummies").await.unwrap().unwrap();
  assert_eq!(
    (read.id.as_str(), read.tenant.as_deref()),
    ("gummies", Some("merchant-1"))
  );
  assert_eq!((read.revision, read.updated_at), (1, 1_700_000_000));
  assert_eq!(read.vectors.hashes, gummies.vectors.hashes);
  assert_same_direction(&read.vectors.composite, &gummies.vectors.composite);
  assert_same_direction(&read.anchor, &gummies.anchor);
  assert!(
    read.quantities.is_empty() && read.anchor_quantities.is_empty(),
    "a record written without quantities has none"
  );

  // A second write to the same id replaces the first; the anchor is whatever the caller says it is.
  let mut revised = record("gummies", Some("merchant-1"), [1.0, 0.3, 0.0, 0.0, 0.0, 0.0]);
  revised.revision = 2;
  revised.anchor = gummies.vectors.composite.clone();
  // Quantities come back exactly as written: single values, ranges, whole numbers, fractions, and units the
  // store has no opinion about. The current and the original set are kept apart.
  revised.quantities = quantities(&[
    ("weight_per_item", Quantity::single(3.5, "g")),
    ("thc_per_item", Quantity::range(18.0, 22.0, "%")),
    ("strength", Quantity::single(0.1, "mg/g")),
  ]);
  revised.anchor_quantities = quantities(&[("weight_per_item", Quantity::single(1.0, "oz"))]);
  store.put(&revised).await.unwrap();
  let read = store.get("gummies").await.unwrap().unwrap();
  assert_eq!(read.revision, 2);
  assert_same_direction(&read.vectors.composite, &revised.vectors.composite);
  assert_same_direction(&read.anchor, &gummies.vectors.composite);
  assert_eq!(read.quantities, revised.quantities);
  assert_eq!(read.anchor_quantities, revised.anchor_quantities);

  // Writing the record again without them removes them; nothing lingers from the previous write.
  let mut bare = revised.clone();
  bare.quantities.clear();
  bare.anchor_quantities.clear();
  store.put(&bare).await.unwrap();
  let read = store.get("gummies").await.unwrap().unwrap();
  assert!(read.quantities.is_empty() && read.anchor_quantities.is_empty());
  store.put(&revised).await.unwrap();

  // Ids are opaque strings, not just slugs.
  let odd_id = "merchant 1 / SKU #42 – ブルーベリー";
  store
    .put(&record(odd_id, None, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    .await
    .unwrap();
  assert_eq!(store.get(odd_id).await.unwrap().unwrap().tenant, None);
  assert!(store.delete(odd_id).await.unwrap());

  // Search: most similar first, on the current composite.
  store
    .put(&record("chocolate", Some("merchant-1"), [0.8, 0.6, 0.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();
  store
    .put(&record("vape", Some("merchant-2"), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();
  let probe = unit([1.0, 0.3, 0.0, 0.0, 0.0, 0.0]);
  let found = store.search(query(&probe)).await.unwrap();
  let ids: Vec<&str> = found.iter().map(|neighbor| neighbor.id.as_str()).collect();
  assert_eq!(ids, ["gummies", "chocolate", "vape"]);
  assert!((found[0].similarity - 1.0).abs() < 1e-4);
  assert!(found[2].similarity.abs() < 1e-4);
  assert_eq!(found[1].tenant.as_deref(), Some("merchant-1"));
  let expected = cosine(&probe, &unit([0.8, 0.6, 0.0, 0.0, 0.0, 0.0])).unwrap();
  assert!(
    (found[1].similarity - expected).abs() < 1e-4,
    "the score is plain cosine similarity"
  );

  // Limit, tenant filter, self-exclusion, and minimum similarity each narrow the result.
  let limited = store
    .search(SearchQuery {
      limit: 1,
      ..query(&probe)
    })
    .await
    .unwrap();
  assert_eq!(limited.len(), 1);
  assert!(
    store
      .search(SearchQuery {
        limit: 0,
        ..query(&probe)
      })
      .await
      .unwrap()
      .is_empty()
  );

  let theirs = store
    .search(SearchQuery {
      tenant: Some("merchant-2"),
      ..query(&probe)
    })
    .await
    .unwrap();
  assert_eq!(
    theirs.iter().map(|neighbor| neighbor.id.as_str()).collect::<Vec<_>>(),
    ["vape"]
  );
  assert!(
    store
      .search(SearchQuery {
        tenant: Some("nobody"),
        ..query(&probe)
      })
      .await
      .unwrap()
      .is_empty()
  );

  let others = store
    .search(SearchQuery {
      exclude_id: Some("gummies"),
      ..query(&probe)
    })
    .await
    .unwrap();
  assert_eq!(
    others.iter().map(|neighbor| neighbor.id.as_str()).collect::<Vec<_>>(),
    ["chocolate", "vape"]
  );

  let close = store
    .search(SearchQuery {
      min_similarity: Some(0.5),
      ..query(&probe)
    })
    .await
    .unwrap();
  assert_eq!(
    close.iter().map(|neighbor| neighbor.id.as_str()).collect::<Vec<_>>(),
    ["gummies", "chocolate"]
  );

  // Vectors of the wrong width are refused before they reach storage.
  let mut short = record("short", None, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
  short.vectors.composite.pop();
  assert!(matches!(store.put(&short).await, Err(StoreError::Invalid(_))));
  let mut short_anchor = record("short", None, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
  short_anchor.anchor.pop();
  assert!(matches!(store.put(&short_anchor).await, Err(StoreError::Invalid(_))));
  assert!(matches!(
    store.search(query(&[1.0, 0.0])).await,
    Err(StoreError::Invalid(_))
  ));
  assert_eq!(store.get("short").await.unwrap(), None);

  // Under a different layout the same data is unreadable and unsearchable, but can still be cleaned up.
  match other_layout.get("gummies").await {
    Err(StoreError::LayoutMismatch { id, stored, expected }) => {
      assert_eq!(
        (id.as_str(), stored.as_str(), expected.as_str()),
        ("gummies", LAYOUT, OTHER_LAYOUT)
      );
    }
    other => panic!("expected a layout mismatch, got {other:?}"),
  }
  assert!(other_layout.search(query(&probe)).await.unwrap().is_empty());
  assert_eq!(other_layout.get("never-stored").await.unwrap(), None);
  assert!(other_layout.delete("vape").await.unwrap());
  assert_eq!(store.get("vape").await.unwrap(), None);

  // Delete reports whether there was something to delete.
  assert!(store.delete("gummies").await.unwrap());
  assert!(!store.delete("gummies").await.unwrap());
  assert_eq!(store.get("gummies").await.unwrap(), None);
  let remaining = store.search(query(&probe)).await.unwrap();
  assert_eq!(
    remaining
      .iter()
      .map(|neighbor| neighbor.id.as_str())
      .collect::<Vec<_>>(),
    ["chocolate"]
  );
}

/// Two handles onto one map, opened under different layouts.
#[derive(Debug)]
struct SharedMemory(std::sync::Arc<tokio::sync::Mutex<Option<MemoryStore>>>, &'static str);

impl SharedMemory {
  async fn with<T>(&self, run: impl AsyncFnOnce(&MemoryStore) -> T) -> T {
    let mut slot = self.0.lock().await;
    let store = slot.take().unwrap().reopened(self.1, DIM);
    let result = run(&store).await;
    *slot = Some(store);
    result
  }
}

impl VectorStore for SharedMemory {
  async fn get(&self, id: &str) -> similarity_store::Result<Option<StoredRecord>> {
    self.with(async |store| store.get(id).await).await
  }

  async fn put(&self, record: &StoredRecord) -> similarity_store::Result<()> {
    self.with(async |store| store.put(record).await).await
  }

  async fn delete(&self, id: &str) -> similarity_store::Result<bool> {
    self.with(async |store| store.delete(id).await).await
  }

  async fn search(&self, query: SearchQuery<'_>) -> similarity_store::Result<Vec<similarity_store::Neighbor>> {
    self.with(async |store| store.search(query).await).await
  }
}

#[tokio::test]
async fn the_memory_store_honors_the_contract() {
  let shared = std::sync::Arc::new(tokio::sync::Mutex::new(Some(MemoryStore::new(LAYOUT, DIM))));
  contract(
    &SharedMemory(shared.clone(), LAYOUT),
    &SharedMemory(shared, OTHER_LAYOUT),
  )
  .await;
}

fn qdrant_settings(collection: String) -> QdrantSettings {
  QdrantSettings {
    url: std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_owned()),
    api_key: std::env::var("QDRANT_API_KEY").ok(),
    collection,
    timeout: Duration::from_secs(10),
  }
}

fn unique_collection(prefix: &str) -> String {
  let nanos = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap()
    .as_nanos();
  format!("{prefix}_{}_{nanos}", std::process::id())
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn the_qdrant_store_honors_the_contract() {
  let settings = qdrant_settings(unique_collection("contract"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let other_layout = QdrantStore::connect(&settings, OTHER_LAYOUT, DIM).await.unwrap();

  let outcome = tokio::spawn({
    let (store, other_layout) = (store.clone(), other_layout);
    async move { contract(&store, &other_layout).await }
  })
  .await;
  // Clean up before reporting, so a failed assertion does not leave a collection behind.
  store.delete_collection().await.unwrap();
  outcome.unwrap();
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn an_existing_collection_of_the_wrong_width_is_refused_not_altered() {
  let settings = qdrant_settings(unique_collection("width"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  store
    .put(&record("gummies", None, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();

  let wider = QdrantStore::connect(&settings, LAYOUT, DIM * 2).await;
  let still_there = store.get("gummies").await;
  store.delete_collection().await.unwrap();

  match wider {
    Err(StoreError::Collection { reason, .. }) => assert!(reason.contains("re-index"), "{reason}"),
    other => panic!("expected the collection to be refused, got {other:?}"),
  }
  assert!(still_there.unwrap().is_some());
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn reconnecting_to_an_existing_collection_sees_the_same_records() {
  let settings = qdrant_settings(unique_collection("reconnect"));
  let first = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  first
    .put(&record("gummies", Some("merchant-1"), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();

  let second = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let read = second.get("gummies").await;
  first.delete_collection().await.unwrap();
  assert_eq!(read.unwrap().unwrap().tenant.as_deref(), Some("merchant-1"));
}

/// The names of the collection's payload indexes, read straight from Qdrant.
async fn indexed_fields(settings: &QdrantSettings) -> Vec<String> {
  let client = qdrant_client::Qdrant::from_url(&settings.url).build().unwrap();
  let info = client.collection_info(settings.collection.as_str()).await.unwrap();
  let mut fields: Vec<String> = info.result.unwrap().payload_schema.into_keys().collect();
  fields.sort();
  fields
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn a_collection_that_lost_its_payload_indexes_gets_them_back_on_connect() {
  use qdrant_client::qdrant::DeleteFieldIndexCollectionBuilder;

  let settings = qdrant_settings(unique_collection("indexes"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let created = indexed_fields(&settings).await;

  // What a first start that died between creating the collection and creating its indexes leaves behind.
  let client = qdrant_client::Qdrant::from_url(&settings.url).build().unwrap();
  for field in &created {
    client
      .delete_field_index(
        DeleteFieldIndexCollectionBuilder::new(settings.collection.as_str(), field.as_str()).wait(true),
      )
      .await
      .unwrap();
  }
  let stripped = indexed_fields(&settings).await;

  QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let repaired = indexed_fields(&settings).await;
  store.delete_collection().await.unwrap();

  assert_eq!(created, ["layout", "tenant"]);
  assert!(stripped.is_empty(), "{stripped:?}");
  assert_eq!(repaired, created);
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn a_point_written_before_quantities_existed_reads_back_with_none() {
  use qdrant_client::Payload;
  use qdrant_client::qdrant::{PointsIdsList, ScrollPointsBuilder, SetPayloadPointsBuilder};

  let settings = qdrant_settings(unique_collection("legacy"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let mut measured = record("gummies", Some("merchant-1"), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
  measured.quantities = quantities(&[("weight_per_item", Quantity::single(3.5, "g"))]);
  measured.anchor_quantities = measured.quantities.clone();
  store.put(&measured).await.unwrap();

  // Turn the point into what an older server wrote: the same payload without the two quantity keys. Overwriting
  // the payload through the raw client is the only way to get one, since this crate no longer writes them.
  let client = qdrant_client::Qdrant::from_url(&settings.url).build().unwrap();
  let scrolled = client
    .scroll(ScrollPointsBuilder::new(settings.collection.as_str()).with_payload(true))
    .await
    .unwrap();
  let point = scrolled.result.into_iter().next().unwrap();
  let mut legacy = serde_json::to_value(Payload::from(point.payload)).unwrap();
  let keys = legacy.as_object_mut().unwrap();
  assert!(keys.remove("quantities").is_some() && keys.remove("anchor_quantities").is_some());
  client
    .overwrite_payload(
      SetPayloadPointsBuilder::new(settings.collection.as_str(), Payload::try_from(legacy).unwrap())
        .points_selector(PointsIdsList {
          ids: vec![point.id.unwrap()],
        })
        .wait(true),
    )
    .await
    .unwrap();

  let read = store.get("gummies").await;
  store.delete_collection().await.unwrap();

  let read = read.unwrap().unwrap();
  assert!(read.quantities.is_empty() && read.anchor_quantities.is_empty());
  assert_eq!(
    (read.tenant.as_deref(), read.revision),
    (Some("merchant-1"), 1),
    "and the rest of the record is intact"
  );
}

#[tokio::test]
async fn an_unreachable_qdrant_is_reported_as_unavailable() {
  let settings = QdrantSettings {
    // A reserved port on localhost: connection refused, immediately.
    url: "http://127.0.0.1:1".to_owned(),
    api_key: None,
    collection: "unreachable".to_owned(),
    timeout: Duration::from_secs(2),
  };
  assert!(matches!(
    QdrantStore::connect(&settings, LAYOUT, DIM).await,
    Err(StoreError::Unavailable(_))
  ));
}
