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

use qdrant_client::qdrant::ScrollPointsBuilder;
use similarity_core::vector::{cosine, normalize};
use similarity_core::{Quantities, Quantity, RecordVectors};
use similarity_store::{
  Cursor, Entry, Expected, MemoryStore, Page, QdrantSettings, QdrantStore, Scan, ScannedRecord, SearchQuery,
  StoreError, StoredRecord, VectorStore, Version,
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

/// Everything a scan of `store` lists from `from` on, a page of `limit` at a time.
async fn scanned_from<S: Scan>(store: &S, mut from: Option<Cursor>, limit: usize) -> Vec<Entry> {
  let mut entries = Vec::new();
  loop {
    let Page { entries: page, next } = store.scan(from.as_ref(), limit).await.unwrap();
    assert!(page.len() <= limit, "a page of {} for a limit of {limit}", page.len());
    entries.extend(page);
    match next {
      Some(next) => from = Some(next),
      None => return entries,
    }
  }
}

fn id_of(entry: &Entry) -> &str {
  match entry {
    Entry::Record(record) => &record.id,
    Entry::OtherLayout { id, .. } => id,
    Entry::Unreadable { id, .. } => id.as_deref().expect("an entry that says which record it is"),
  }
}

/// The scan half of the contract: a store lists everything it holds, once, whatever the size of a page.
/// `store` and `other_layout` are two views of the same underlying data, which must be empty.
async fn scanning<S: VectorStore + Scan>(store: &S, other_layout: &S) {
  // Nothing stored is one empty page, and a page of nothing is not a page.
  assert_eq!(
    store.scan(None, 10).await.unwrap(),
    Page {
      entries: Vec::new(),
      next: None
    }
  );
  assert!(matches!(store.scan(None, 0).await, Err(StoreError::Invalid(_))));

  // Nine records of this layout, with and without a brand and quantities, and two of another.
  let mut stored = Vec::new();
  for index in 0..9_u8 {
    let mut written = record(
      &format!("sku-{index}"),
      (index % 3 != 0).then_some("brand-1"),
      [1.0, f32::from(index), 0.0, 0.0, 0.0, 0.0],
    );
    if index % 2 == 0 {
      written.quantities = quantities(&[("weight", Quantity::single(f64::from(index) + 1.0, "g"))]);
      written.anchor_quantities = quantities(&[("weight", Quantity::single(1.0, "g"))]);
    }
    written.revision = u64::from(index) + 1;
    store.put(&written).await.unwrap();
    stored.push(written);
  }
  for id in ["old-0", "old-1"] {
    other_layout
      .put(&record(id, Some("brand-2"), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
      .await
      .unwrap();
  }
  let mut expected: Vec<String> = stored.iter().map(|record| record.id.clone()).collect();
  expected.extend(["old-0".to_owned(), "old-1".to_owned()]);
  expected.sort();

  // Every id once and in one order, whether a page holds one entry, exactly all of them, or more.
  let whole = scanned_from(store, None, 100).await;
  let order: Vec<String> = whole.iter().map(|entry| id_of(entry).to_owned()).collect();
  let mut sorted = order.clone();
  sorted.sort();
  assert_eq!(sorted, expected);
  for limit in [1, 2, 3, 4, 9, 11] {
    assert_eq!(scanned_from(store, None, limit).await, whole, "pages of {limit}");
  }

  // A record of this layout is listed as it would be read, without its vectors. One of another layout is named
  // and not read, from whichever side it is looked at.
  for entry in &whole {
    match entry {
      Entry::Record(scanned) => {
        let read = store.get(&scanned.id).await.unwrap().unwrap();
        assert_eq!(*scanned, ScannedRecord::from(&read));
        assert_eq!(scanned.hashes["name"], format!("hash-of-{}", scanned.id));
      }
      Entry::OtherLayout { id, layout } => {
        assert!(id.starts_with("old-"), "{id}");
        assert_eq!(layout, OTHER_LAYOUT);
      }
      Entry::Unreadable { .. } => panic!("everything here can be read: {entry:?}"),
    }
  }
  let even = whole
    .iter()
    .find_map(|entry| match entry {
      Entry::Record(scanned) if scanned.id == "sku-4" => Some(scanned.clone()),
      _ => None,
    })
    .unwrap();
  assert_eq!(even.brand_id.as_deref(), Some("brand-1"));
  assert_eq!(even.quantities["weight"], Quantity::single(5.0, "g"));
  assert_eq!(even.anchor_quantities["weight"], Quantity::single(1.0, "g"));
  assert_eq!(even.revision, 5);
  let from_the_other_side = scanned_from(other_layout, None, 4).await;
  assert_eq!(
    from_the_other_side
      .iter()
      .filter(|entry| matches!(entry, Entry::Record(_)))
      .count(),
    2
  );
  assert!(from_the_other_side.iter().all(|entry| match entry {
    Entry::Record(scanned) => scanned.id.starts_with("old-"),
    Entry::OtherLayout { id, layout } => id.starts_with("sku-") && layout == LAYOUT,
    Entry::Unreadable { .. } => false,
  }));

  // A scan is not a snapshot, and the store is written to between its pages. Here: a record already listed is
  // deleted, the very record the next page starts at is deleted, one further on is rewritten, and one is
  // created.
  let first = store.scan(None, 4).await.unwrap();
  let seen: Vec<String> = first.entries.iter().map(|entry| id_of(entry).to_owned()).collect();
  assert_eq!(seen, order[..4]);
  let (gone_seen, gone_unseen) = (order[0].clone(), order[4].clone());
  assert!(store.delete(&gone_seen).await.unwrap());
  assert!(store.delete(&gone_unseen).await.unwrap());
  let rewritten = order[5..]
    .iter()
    .find(|id| id.starts_with("sku-"))
    .cloned()
    .expect("a record of this layout after the fifth entry");
  let mut again = store.get(&rewritten).await.unwrap().unwrap();
  again.revision = 77;
  again.brand_id = Some("brand-9".to_owned());
  store.put(&again).await.unwrap();
  store
    .put(&record("created", None, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();

  let rest = scanned_from(store, first.next, 4).await;
  let listed: Vec<&str> = seen.iter().map(String::as_str).chain(rest.iter().map(id_of)).collect();
  let times = |id: &str| listed.iter().filter(|listed| **listed == id).count();
  for id in &order {
    if *id == gone_unseen {
      assert_eq!(times(id), 0, "{id} was deleted before its page was read");
    } else {
      assert_eq!(times(id), 1, "{id}");
    }
  }
  assert!(times("created") <= 1);
  assert_eq!(listed.len(), order.len() - 1 + times("created"), "{listed:?}");
  let after = rest
    .iter()
    .find_map(|entry| match entry {
      Entry::Record(scanned) if scanned.id == rewritten => Some(scanned),
      _ => None,
    })
    .unwrap();
  assert_eq!((after.revision, after.brand_id.as_deref()), (77, Some("brand-9")));
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

impl Scan for SharedMemory {
  async fn scan(&self, from: Option<&Cursor>, limit: usize) -> similarity_store::Result<Page> {
    self.with(async |store| store.scan(from, limit).await).await
  }
}

#[tokio::test]
async fn the_memory_store_lists_what_it_holds() {
  let shared = std::sync::Arc::new(tokio::sync::Mutex::new(Some(MemoryStore::new(LAYOUT, DIM))));
  scanning(
    &SharedMemory(shared.clone(), LAYOUT),
    &SharedMemory(shared, OTHER_LAYOUT),
  )
  .await;
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
  use qdrant_client::qdrant::{PointsIdsList, SetPayloadPointsBuilder};

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
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn the_qdrant_store_lists_what_it_holds() {
  let settings = qdrant_settings(unique_collection("scan"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  let other_layout = QdrantStore::connect(&settings, OTHER_LAYOUT, DIM).await.unwrap();

  let outcome = tokio::spawn({
    let (store, other_layout) = (store.clone(), other_layout);
    async move { Box::pin(scanning(&store, &other_layout)).await }
  })
  .await;
  store.delete_collection().await.unwrap();
  outcome.unwrap();
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn a_point_that_cannot_be_read_is_listed_and_the_scan_goes_on() {
  use qdrant_client::Payload;
  use qdrant_client::qdrant::{NamedVectors, PointStruct, PointsIdsList, SetPayloadPointsBuilder, UpsertPointsBuilder};

  let settings = qdrant_settings(unique_collection("unreadable"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  for id in ["gummies", "vape", "flower"] {
    store
      .put(&record(id, Some("brand-1"), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
      .await
      .unwrap();
  }

  // Two things this service did not write as they are now. A point from before 2026-11-12, whose brand is under
  // `tenant`: the payload of `vape`, with the key renamed. And a point of something else altogether, under a
  // number, as this service never names a point.
  let client = qdrant_client::Qdrant::from_url(&settings.url).build().unwrap();
  let scrolled = client
    .scroll(ScrollPointsBuilder::new(settings.collection.as_str()).with_payload(true))
    .await
    .unwrap();
  let point = scrolled
    .result
    .into_iter()
    .find(|point| point.payload["record_id"].as_str().map(String::as_str) == Some("vape"))
    .unwrap();
  let mut old = serde_json::to_value(Payload::from(point.payload)).unwrap();
  let keys = old.as_object_mut().unwrap();
  let brand = keys.remove("brand_id").unwrap();
  keys.insert("tenant".to_owned(), brand);
  client
    .overwrite_payload(
      SetPayloadPointsBuilder::new(settings.collection.as_str(), Payload::try_from(old).unwrap())
        .points_selector(PointsIdsList {
          ids: vec![point.id.unwrap()],
        })
        .wait(true),
    )
    .await
    .unwrap();
  let direction = unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
  let foreign = PointStruct::new(
    7,
    NamedVectors::default()
      .add_vector("current", direction.clone())
      .add_vector("anchor", direction),
    Payload::try_from(serde_json::json!({"note": "not a record"})).unwrap(),
  );
  client
    .upsert_points(UpsertPointsBuilder::new(settings.collection.as_str(), vec![foreign]).wait(true))
    .await
    .unwrap();

  // One entry a page, so that the numbered point is a cursor on the way.
  let scan = tokio::spawn({
    let store = store.clone();
    async move { scanned_from(&store, None, 1).await }
  })
  .await;
  store.delete_collection().await.unwrap();
  let entries = scan.unwrap();

  assert_eq!(entries.len(), 4, "{entries:?}");
  let mut readable: Vec<&str> = entries
    .iter()
    .filter_map(|entry| match entry {
      Entry::Record(scanned) => Some(scanned.id.as_str()),
      _ => None,
    })
    .collect();
  readable.sort_unstable();
  assert_eq!(readable, ["flower", "gummies"]);
  assert!(entries.iter().any(|entry| matches!(
    entry,
    Entry::Unreadable { id: Some(id), reason } if id == "vape" && reason.contains("`tenant`")
  )));
  assert!(
    entries
      .iter()
      .any(|entry| matches!(entry, Entry::Unreadable { id: None, .. }))
  );
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn opening_a_collection_that_is_not_there_is_refused_and_creates_nothing() {
  let settings = qdrant_settings(unique_collection("absent"));
  match QdrantStore::open(&settings, LAYOUT, DIM).await {
    Err(StoreError::Collection { collection, reason }) => {
      assert_eq!(collection, settings.collection);
      assert!(reason.contains("does not exist"), "{reason}");
    }
    other => panic!("expected the collection to be reported missing, got {other:?}"),
  }
  let client = qdrant_client::Qdrant::from_url(&settings.url).build().unwrap();
  assert!(!client.collection_exists(settings.collection.as_str()).await.unwrap());
}

#[tokio::test]
#[ignore = "needs a Qdrant server at $QDRANT_URL (`just qdrant-up`); run with `just test-qdrant`"]
async fn opening_a_collection_changes_nothing_about_it_and_lists_what_it_holds() {
  use qdrant_client::qdrant::DeleteFieldIndexCollectionBuilder;

  let settings = qdrant_settings(unique_collection("opened"));
  let store = QdrantStore::connect(&settings, LAYOUT, DIM).await.unwrap();
  store
    .put(&record("gummies", None, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    .await
    .unwrap();
  let client = qdrant_client::Qdrant::from_url(&settings.url).build().unwrap();
  for field in indexed_fields(&settings).await {
    client
      .delete_field_index(
        DeleteFieldIndexCollectionBuilder::new(settings.collection.as_str(), field.as_str()).wait(true),
      )
      .await
      .unwrap();
  }

  // Where `connect` would put the indexes back, `open` leaves the collection as it found it, and still refuses
  // one of another width.
  let opened = QdrantStore::open(&settings, LAYOUT, DIM).await;
  let indexes = indexed_fields(&settings).await;
  let listed = match &opened {
    Ok(opened) => Some(opened.scan(None, 10).await),
    Err(_) => None,
  };
  let wrong_width = QdrantStore::open(&settings, LAYOUT, DIM + 1).await;
  store.delete_collection().await.unwrap();

  assert!(indexes.is_empty(), "{indexes:?}");
  let page = listed.unwrap().unwrap();
  assert_eq!(page.next, None);
  assert_eq!(page.entries.iter().map(id_of).collect::<Vec<_>>(), ["gummies"]);
  assert!(matches!(wrong_width, Err(StoreError::Collection { .. })));
}

#[tokio::test]
async fn an_unreachable_qdrant_cannot_be_opened() {
  let settings = QdrantSettings {
    url: "http://127.0.0.1:1".to_owned(),
    api_key: None,
    collection: "unreachable".to_owned(),
    timeout: Duration::from_secs(2),
  };
  assert!(matches!(
    QdrantStore::open(&settings, LAYOUT, DIM).await,
    Err(StoreError::Unavailable(_))
  ));
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
