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
use similarity_store::{
  Expected, MemoryStore, QdrantSettings, QdrantStore, SearchQuery, StoreError, StoredRecord, VectorStore, Version,
};

const DIM: usize = 6;
const LAYOUT: &str = "model-a|category,name";
const OTHER_LAYOUT: &str = "model-b|category,name";

fn unit(components: [f32; DIM]) -> Vec<f32> {
  let mut vector = components.to_vec();
  assert!(normalize(&mut vector));
  vector
}

fn record(id: &str, brand_id: Option<&str>, composite: [f32; DIM]) -> StoredRecord {
  StoredRecord {
    id: id.to_owned(),
    brand_id: brand_id.map(str::to_owned),
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
    brand_id: None,
    exclude_id: None,
    min_similarity: None,
  }
}

fn assert_same_direction(left: &[f32], right: &[f32]) {
  assert!(cosine(left, right).unwrap() > 0.999_999, "{left:?} vs {right:?}");
}

/// `store` and `other_layout` are two views of the same underlying data, opened under different layouts.
async fn contract<S: VectorStore>(store: &S, other_layout: &S) {
  // A store that can be used says so, with or without anything in it, under either layout.
  store.check().await.unwrap();
  other_layout.check().await.unwrap();

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
  let gummies = record("gummies", Some("brand-1"), [1.0, 0.1, 0.0, 0.0, 0.0, 0.0]);
  store.put(&gummies).await.unwrap();
  let read = store.get("gummies").await.unwrap().unwrap();
  assert_eq!(
    (read.id.as_str(), read.brand_id.as_deref()),
    ("gummies", Some("brand-1"))
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
  let mut revised = record("gummies", Some("brand-1"), [1.0, 0.3, 0.0, 0.0, 0.0, 0.0]);
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
  assert_eq!(store.get(odd_id).await.unwrap().unwrap().brand_id, None);
  assert!(store.delete(odd_id).await.unwrap());

  // Search: most similar first, on the current composite.
  store
    .put(&record("chocolate", Some("brand-1"), [0.8, 0.6, 0.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();
  store
    .put(&record("vape", Some("brand-2"), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();
  let probe = unit([1.0, 0.3, 0.0, 0.0, 0.0, 0.0]);
  let found = store.search(query(&probe)).await.unwrap();
  let ids: Vec<&str> = found.iter().map(|neighbor| neighbor.id.as_str()).collect();
  assert_eq!(ids, ["gummies", "chocolate", "vape"]);
  assert!((found[0].similarity - 1.0).abs() < 1e-4);
  assert!(found[2].similarity.abs() < 1e-4);
  assert_eq!(found[1].brand_id.as_deref(), Some("brand-1"));
  let expected = cosine(&probe, &unit([0.8, 0.6, 0.0, 0.0, 0.0, 0.0])).unwrap();
  assert!(
    (found[1].similarity - expected).abs() < 1e-4,
    "the score is plain cosine similarity"
  );

  // Limit, brand filter, self-exclusion, and minimum similarity each narrow the result.
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
      brand_id: Some("brand-2"),
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
        brand_id: Some("nobody"),
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

  // A search with vectors is the same search: the same results in the same order under every filter, each with
  // the record's current vectors and quantities as `get` returns them.
  let mut weighed = record("chocolate", Some("brand-1"), [0.8, 0.6, 0.0, 0.0, 0.0, 0.0]);
  weighed.quantities = quantities(&[("weight", Quantity::single(3.5, "g"))]);
  store.put(&weighed).await.unwrap();
  let variants = [
    query(&probe),
    SearchQuery {
      limit: 1,
      ..query(&probe)
    },
    SearchQuery {
      limit: 0,
      ..query(&probe)
    },
    SearchQuery {
      brand_id: Some("brand-2"),
      ..query(&probe)
    },
    SearchQuery {
      brand_id: Some("nobody"),
      ..query(&probe)
    },
    SearchQuery {
      exclude_id: Some("gummies"),
      ..query(&probe)
    },
    SearchQuery {
      min_similarity: Some(0.5),
      ..query(&probe)
    },
  ];
  for variant in variants {
    let plain = store.search(variant).await.unwrap();
    let with_vectors = store.search_vectors(variant).await.unwrap();
    assert_eq!(plain.len(), with_vectors.len(), "{variant:?}");
    for (neighbor, found) in plain.iter().zip(&with_vectors) {
      assert_eq!(
        (&neighbor.id, &neighbor.brand_id),
        (&found.neighbor.id, &found.neighbor.brand_id)
      );
      assert!((neighbor.similarity - found.neighbor.similarity).abs() < 1e-6);
      let stored = store.get(&neighbor.id).await.unwrap().unwrap();
      assert_same_direction(&found.vectors.composite, &stored.vectors.composite);
      assert_eq!(found.vectors.hashes, stored.vectors.hashes);
      assert_eq!(found.quantities, stored.quantities);
    }
  }
  let found = store.search_vectors(query(&probe)).await.unwrap();
  assert_eq!(
    found[1].quantities, weighed.quantities,
    "the chocolate, with its weight"
  );
  assert!(matches!(
    store.search_vectors(query(&[1.0, 0.0])).await,
    Err(StoreError::Invalid(_))
  ));

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
  assert!(other_layout.search_vectors(query(&probe)).await.unwrap().is_empty());
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

/// The record `id` as one of several writers would write it: told apart by the fingerprint it carries.
fn written_by(writer: &str, id: &str, revision: u64, updated_at: u64) -> StoredRecord {
  let mut record = record(id, Some("brand-1"), [1.0, 0.2, 0.0, 0.0, 0.0, 0.0]);
  record.vectors.hashes = BTreeMap::from([("name".to_owned(), format!("written-by-{writer}"))]);
  record.revision = revision;
  record.updated_at = updated_at;
  record
}

async fn writer_of<S: VectorStore>(store: &S, id: &str) -> Option<String> {
  let stored = store.get(id).await.unwrap()?;
  Some(
    stored.vectors.hashes["name"]
      .trim_start_matches("written-by-")
      .to_owned(),
  )
}

/// Conditional writes: what writers that cannot share a lock rely on. `store` and `other_layout` are two views of
/// the same underlying data, opened under different layouts.
async fn conditional<S: VectorStore>(store: &S, other_layout: &S) {
  const T0: u64 = 1_700_000_000;
  const T1: u64 = 1_700_000_050;

  // Creating: only when nothing is there.
  let first = written_by("first", "gummies", 1, T0);
  assert!(store.put_if(&first, Expected::Absent).await.unwrap());
  assert_eq!(store.get("gummies").await.unwrap().unwrap().version(), first.version());
  assert!(
    !store
      .put_if(&written_by("rival", "gummies", 1, T0), Expected::Absent)
      .await
      .unwrap(),
    "it is there now"
  );
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("first"));

  // Updating: only when what is stored is still the version that was read.
  let second = written_by("second", "gummies", 2, T1);
  assert!(store.put_if(&second, Expected::Version(first.version())).await.unwrap());
  // A writer that read the first version too has lost, even though it would write the same revision in the same
  // second as the one that won.
  let late = written_by("late", "gummies", 2, T1);
  assert!(!store.put_if(&late, Expected::Version(first.version())).await.unwrap());
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("second"));
  // A version is the revision and the time together: neither alone will do.
  for stale in [
    Version {
      revision: 2,
      updated_at: T0,
    },
    Version {
      revision: 1,
      updated_at: T1,
    },
    Version {
      revision: 3,
      updated_at: T1,
    },
  ] {
    assert!(
      !store.put_if(&late, Expected::Version(stale)).await.unwrap(),
      "{stale:?}"
    );
  }
  let read = store.get("gummies").await.unwrap().unwrap();
  assert_eq!(read.version(), second.version());
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("second"));
  // Everything a plain write stores, a conditional one stores.
  assert_eq!(read.brand_id.as_deref(), Some("brand-1"));
  assert_same_direction(&read.vectors.composite, &second.vectors.composite);
  assert_same_direction(&read.anchor, &second.anchor);
  // The write that won is what the next one names.
  let third = written_by("third", "gummies", 3, T1);
  assert!(store.put_if(&third, Expected::Version(second.version())).await.unwrap());
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("third"));

  // Another layout's record is not a store's to replace, to create over, or to delete by version.
  let foreign = written_by("another-layout", "gummies", 4, T1);
  assert!(
    !other_layout
      .put_if(&foreign, Expected::Version(third.version()))
      .await
      .unwrap()
  );
  assert!(!other_layout.put_if(&foreign, Expected::Absent).await.unwrap());
  assert!(!other_layout.delete_if("gummies", third.version()).await.unwrap());
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("third"));

  // A vector of the wrong length is refused, as by a plain write.
  let mut short = written_by("short", "gummies", 4, T1);
  short.anchor.pop();
  assert!(matches!(
    store.put_if(&short, Expected::Version(third.version())).await,
    Err(StoreError::Invalid(_))
  ));
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("third"));

  // Deleting: only the version that was read.
  assert!(!store.delete_if("gummies", second.version()).await.unwrap());
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("third"));
  assert!(store.delete_if("gummies", third.version()).await.unwrap());
  assert_eq!(store.get("gummies").await.unwrap(), None);
  // Gone is gone, whoever deleted it.
  assert!(store.delete_if("gummies", third.version()).await.unwrap());
  // A write that expected a version of a record that has since been deleted does not bring it back.
  assert!(
    !store
      .put_if(
        &written_by("ghost", "gummies", 4, T1),
        Expected::Version(third.version())
      )
      .await
      .unwrap()
  );
  assert_eq!(store.get("gummies").await.unwrap(), None);
  // It can be created again, and a plain write is still unconditional.
  assert!(store.put_if(&first, Expected::Absent).await.unwrap());
  store.put(&written_by("plain", "gummies", 9, T1)).await.unwrap();
  assert_eq!(writer_of(store, "gummies").await.as_deref(), Some("plain"));
  // A record a plain write stored is one a conditional write can follow.
  let after_plain = Version {
    revision: 9,
    updated_at: T1,
  };
  assert!(
    store
      .put_if(
        &written_by("after-plain", "gummies", 10, T1),
        Expected::Version(after_plain)
      )
      .await
      .unwrap()
  );
  assert!(store.delete("gummies").await.unwrap());

  // Eight writers at once that all saw nothing: one creates, and it is the one that was told so.
  let create = |writer: usize| {
    let record = written_by(&writer.to_string(), "race", 1, T0);
    async move { store.put_if(&record, Expected::Absent).await.unwrap() }
  };
  let created = tokio::join!(
    create(0),
    create(1),
    create(2),
    create(3),
    create(4),
    create(5),
    create(6),
    create(7)
  );
  let created = [
    created.0, created.1, created.2, created.3, created.4, created.5, created.6, created.7,
  ];
  let winners: Vec<String> = (0..8)
    .filter(|writer| created[*writer])
    .map(|writer| writer.to_string())
    .collect();
  assert_eq!(winners.len(), 1, "{created:?}");
  assert_eq!(writer_of(store, "race").await, Some(winners[0].clone()));

  // Eight writers at once that all read that record: one replaces it, and it is the one that was told so.
  let seen = store.get("race").await.unwrap().unwrap().version();
  let update = |writer: usize| {
    let record = written_by(&format!("update-{writer}"), "race", 2, T1);
    async move { store.put_if(&record, Expected::Version(seen)).await.unwrap() }
  };
  let updated = tokio::join!(
    update(0),
    update(1),
    update(2),
    update(3),
    update(4),
    update(5),
    update(6),
    update(7)
  );
  let updated = [
    updated.0, updated.1, updated.2, updated.3, updated.4, updated.5, updated.6, updated.7,
  ];
  let winners: Vec<String> = (0..8)
    .filter(|writer| updated[*writer])
    .map(|writer| format!("update-{writer}"))
    .collect();
  assert_eq!(winners.len(), 1, "{updated:?}");
  assert_eq!(writer_of(store, "race").await, Some(winners[0].clone()));
  assert!(store.delete("race").await.unwrap());
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
  async fn check(&self) -> similarity_store::Result<()> {
    self.with(async |store| store.check().await).await
  }

  async fn get(&self, id: &str) -> similarity_store::Result<Option<StoredRecord>> {
    self.with(async |store| store.get(id).await).await
  }

  async fn put(&self, record: &StoredRecord) -> similarity_store::Result<()> {
    self.with(async |store| store.put(record).await).await
  }

  async fn put_if(&self, record: &StoredRecord, expected: Expected) -> similarity_store::Result<bool> {
    self.with(async |store| store.put_if(record, expected).await).await
  }

  async fn delete(&self, id: &str) -> similarity_store::Result<bool> {
    self.with(async |store| store.delete(id).await).await
  }

  async fn delete_if(&self, id: &str, version: Version) -> similarity_store::Result<bool> {
    self.with(async |store| store.delete_if(id, version).await).await
  }

  async fn search_vectors(&self, query: SearchQuery<'_>) -> similarity_store::Result<Vec<similarity_store::Found>> {
    self.with(async |store| store.search_vectors(query).await).await
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

#[tokio::test]
async fn the_memory_store_writes_only_over_what_the_writer_read() {
  let shared = std::sync::Arc::new(tokio::sync::Mutex::new(Some(MemoryStore::new(LAYOUT, DIM))));
  conditional(
    &SharedMemory(shared.clone(), LAYOUT),
    &SharedMemory(shared, OTHER_LAYOUT),
  )
  .await;
  // And without the test's own lock around it, where the eight writers really are at once.
  let store = std::sync::Arc::new(MemoryStore::new(LAYOUT, DIM));
  let tasks: Vec<_> = (0..32)
    .map(|writer| {
      let store = std::sync::Arc::clone(&store);
      tokio::spawn(async move {
        let record = written_by(&writer.to_string(), "race", 1, 1_700_000_000);
        store.put_if(&record, Expected::Absent).await.unwrap()
      })
    })
    .collect();
  let mut created = 0;
  for task in tasks {
    created += usize::from(task.await.unwrap());
  }
  assert_eq!(created, 1);
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

  // A store whose collection was deleted under it can be reached and cannot be used, and says which.
  match store.check().await {
    Err(StoreError::Collection { collection, reason }) => {
      assert_eq!(collection, settings.collection);
      assert!(reason.contains("no longer exists"), "{reason}");
    }
    other => panic!("expected the collection to be reported gone, got {other:?}"),
  }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn the_qdrant_store_writes_only_over_what_the_writer_read() {
  let settings = qdrant_settings(unique_collection("conditional"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let other_layout = QdrantStore::connect(&settings, OTHER_LAYOUT, DIM).await.unwrap();
  // A third connection, as another replica is: its writes must lose and win by the same rule.
  let replica = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();

  let outcome = tokio::spawn({
    let (store, other_layout) = (store.clone(), other_layout);
    async move {
      Box::pin(conditional(&store, &other_layout)).await;

      // Two replicas that both read revision 1: one of them writes revision 2.
      let first = written_by("first", "shared", 1, 1_700_000_000);
      assert!(store.put_if(&first, Expected::Absent).await.unwrap());
      for round in 0..20_u64 {
        let seen = store.get("shared").await.unwrap().unwrap().version();
        assert_eq!(replica.get("shared").await.unwrap().unwrap().version(), seen);
        let here = written_by("here", "shared", seen.revision + 1, 1_700_000_100 + round);
        let there = written_by("there", "shared", seen.revision + 1, 1_700_000_100 + round);
        let (local, remote) = tokio::join!(
          store.put_if(&here, Expected::Version(seen)),
          replica.put_if(&there, Expected::Version(seen))
        );
        let (local, remote) = (local.unwrap(), remote.unwrap());
        assert!(local != remote, "round {round}: {local} and {remote}");
        let winner = if local { "here" } else { "there" };
        assert_eq!(writer_of(&store, "shared").await.as_deref(), Some(winner));
        assert_eq!(store.get("shared").await.unwrap().unwrap().revision, round + 2);
      }
    }
  })
  .await;
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
    .put(&record("gummies", Some("brand-1"), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();

  let second = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let read = second.get("gummies").await;
  first.delete_collection().await.unwrap();
  assert_eq!(read.unwrap().unwrap().brand_id.as_deref(), Some("brand-1"));
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

  assert_eq!(created, ["brand_id", "layout"]);
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
  let mut measured = record("gummies", Some("brand-1"), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
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
    (read.brand_id.as_deref(), read.revision),
    (Some("brand-1"), 1),
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
