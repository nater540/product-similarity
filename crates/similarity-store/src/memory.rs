//! An in-process store: a map behind a lock, searched by brute force.
//!
//! At the scale this service targets (tens of thousands of records) brute force over a few thousand floats per
//! record is a handful of milliseconds, so this is not only a test double: it is the honest baseline against which
//! "what does Qdrant buy us?" is measured. What it lacks is everything operational: persistence, replication, and
//! access from more than one server process.

use std::collections::HashMap;
use std::sync::RwLock;

use similarity_core::vector::cosine;

use crate::{
  Cursor, Entry, Expected, Found, Neighbor, Page, Read, Result, Scan, ScannedRecord, SearchQuery, StoreError,
  StoredRecord, VectorStore, Version,
};

/// A [`VectorStore`] that keeps everything in memory.
#[derive(Debug)]
pub struct MemoryStore {
  layout: String,
  dimension: usize,
  records: RwLock<HashMap<String, (String, StoredRecord)>>,
}

impl MemoryStore {
  /// Creates an empty store for composites of `dimension` components written under `layout`.
  pub fn new(layout: impl Into<String>, dimension: usize) -> Self {
    Self {
      layout: layout.into(),
      dimension,
      records: RwLock::new(HashMap::new()),
    }
  }

  /// Opens the same records under a different layout, as a server restarted with a new model or schema would.
  #[must_use]
  pub fn reopened(self, layout: impl Into<String>, dimension: usize) -> Self {
    Self {
      layout: layout.into(),
      dimension,
      records: self.records,
    }
  }

  fn check_length(&self, what: &str, vector: &[f32]) -> Result<()> {
    if vector.len() == self.dimension {
      Ok(())
    } else {
      Err(StoreError::Invalid(format!(
        "{what} has {} components, but this store holds vectors of {}",
        vector.len(),
        self.dimension
      )))
    }
  }

  fn poisoned() -> StoreError {
    StoreError::Unavailable("in-memory store lock was poisoned by a panicking writer".to_owned())
  }

  /// The records a query matches, most similar first, each turned into a result by `found`. One loop for both
  /// kinds of search, so that they cannot differ in what they match or in what order.
  fn hits<T>(&self, query: SearchQuery<'_>, found: impl Fn(&StoredRecord, f32) -> T) -> Result<Vec<T>> {
    self.check_length("the query vector", query.vector)?;
    let records = self.records.read().map_err(|_| Self::poisoned())?;

    let mut hits = Vec::new();
    for (layout, record) in records.values() {
      let excluded = query.exclude_id == Some(record.id.as_str());
      let other_brand = query.brand_id.is_some() && query.brand_id != record.brand_id.as_deref();
      if *layout != self.layout || excluded || other_brand {
        continue;
      }
      let similarity = cosine(query.vector, &record.vectors.composite).map_err(|source| StoreError::Corrupt {
        id: record.id.clone(),
        reason: source.to_string(),
      })?;
      if query.min_similarity.is_none_or(|minimum| similarity >= minimum) {
        hits.push((similarity, record));
      }
    }
    // Ties are broken by id so that results are stable across runs and hash-map iteration orders.
    hits.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
    hits.truncate(query.limit);
    Ok(
      hits
        .into_iter()
        .map(|(similarity, record)| found(record, similarity))
        .collect(),
    )
  }
}

fn neighbor(record: &StoredRecord, similarity: f32) -> Neighbor {
  Neighbor {
    id: record.id.clone(),
    brand_id: record.brand_id.clone(),
    similarity,
  }
}

impl VectorStore for MemoryStore {
  async fn check(&self) -> Result<()> {
    // There is nothing to reach. What can be wrong with it is a writer that panicked with the lock held.
    self.records.read().map(|_| ()).map_err(|_| Self::poisoned())
  }

  async fn get(&self, id: &str) -> Result<Option<StoredRecord>> {
    let records = self.records.read().map_err(|_| Self::poisoned())?;
    match records.get(id) {
      None => Ok(None),
      Some((layout, record)) if *layout == self.layout => Ok(Some(record.clone())),
      Some((layout, _)) => Err(StoreError::LayoutMismatch {
        id: id.to_owned(),
        stored: layout.clone(),
        expected: self.layout.clone(),
      }),
    }
  }

  async fn put(&self, record: &StoredRecord) -> Result<()> {
    self.check_length("the composite", &record.vectors.composite)?;
    self.check_length("the anchor", &record.anchor)?;
    let mut records = self.records.write().map_err(|_| Self::poisoned())?;
    records.insert(record.id.clone(), (self.layout.clone(), record.clone()));
    Ok(())
  }

  async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool> {
    self.check_length("the composite", &record.vectors.composite)?;
    self.check_length("the anchor", &record.anchor)?;
    // Looked at and written under one lock: that is all a compare-and-swap is.
    let mut records = self.records.write().map_err(|_| Self::poisoned())?;
    let as_expected = match (expected, records.get(&record.id)) {
      (Expected::Absent, stored) => stored.is_none(),
      (Expected::Version(version), Some((layout, stored))) => *layout == self.layout && stored.version() == version,
      (Expected::Version(_), None) => false,
    };
    if as_expected {
      records.insert(record.id.clone(), (self.layout.clone(), record.clone()));
    }
    Ok(as_expected)
  }

  async fn delete(&self, id: &str) -> Result<bool> {
    let mut records = self.records.write().map_err(|_| Self::poisoned())?;
    Ok(records.remove(id).is_some())
  }

  async fn delete_if(&self, id: &str, version: Version) -> Result<bool> {
    let mut records = self.records.write().map_err(|_| Self::poisoned())?;
    let Some((layout, stored)) = records.get(id) else {
      return Ok(true);
    };
    if *layout != self.layout || stored.version() != version {
      return Ok(false);
    }
    records.remove(id);
    Ok(true)
  }

  async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>> {
    self.hits(query, neighbor)
  }

  async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>> {
    self.hits(query, |record, similarity| Found {
      neighbor: neighbor(record, similarity),
      vectors: record.vectors.clone(),
      quantities: record.quantities.clone(),
    })
  }
}

impl Scan for MemoryStore {
  /// In the order of the records' ids. The map has no order of its own, so every page sorts the ids that are
  /// left: fine for what this store is for, and it leaves the map the requests use as it was.
  async fn scan(&self, from: Option<&Cursor>, limit: usize, read: Read) -> Result<Page> {
    if limit == 0 {
      return Err(StoreError::Invalid(
        "a scan needs a page of at least one entry".to_owned(),
      ));
    }
    let records = self.records.read().map_err(|_| Self::poisoned())?;
    let mut ids: Vec<&String> = records
      .keys()
      .filter(|id| from.is_none_or(|Cursor(first)| *id >= first))
      .collect();
    ids.sort_unstable();
    // The cursor is the id the next page starts at: a record deleted meanwhile is simply not there to start at,
    // and one created meanwhile is listed if it sorts after it.
    let next = ids.get(limit).map(|id| Cursor((*id).clone()));
    let entries = ids
      .into_iter()
      .take(limit)
      .filter_map(|id| records.get(id))
      .map(|(layout, record)| {
        if *layout == self.layout {
          Entry::Record(ScannedRecord {
            anchor: (read == Read::Anchor).then(|| record.anchor.clone()),
            ..ScannedRecord::from(record)
          })
        } else {
          Entry::OtherLayout {
            id: record.id.clone(),
            layout: layout.clone(),
          }
        }
      })
      .collect();
    Ok(Page { entries, next })
  }
}
