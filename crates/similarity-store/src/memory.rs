//! An in-process store: a map behind a lock, searched by brute force.
//!
//! At the scale this service targets (tens of thousands of records) brute force over a few thousand floats per
//! record is a handful of milliseconds, so this is not only a test double: it is the honest baseline against which
//! "what does Qdrant buy us?" is measured. What it lacks is everything operational: persistence, replication, and
//! access from more than one server process.

use std::collections::HashMap;
use std::sync::RwLock;

use similarity_core::vector::cosine;

use crate::{Neighbor, Result, SearchQuery, StoreError, StoredRecord, VectorStore};

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
}

impl VectorStore for MemoryStore {
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

  async fn delete(&self, id: &str) -> Result<bool> {
    let mut records = self.records.write().map_err(|_| Self::poisoned())?;
    Ok(records.remove(id).is_some())
  }

  async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>> {
    self.check_length("the query vector", query.vector)?;
    let records = self.records.read().map_err(|_| Self::poisoned())?;

    let mut neighbors = Vec::new();
    for (layout, record) in records.values() {
      let excluded = query.exclude_id == Some(record.id.as_str());
      let other_tenant = query.tenant.is_some() && query.tenant != record.tenant.as_deref();
      if *layout != self.layout || excluded || other_tenant {
        continue;
      }
      let similarity = cosine(query.vector, &record.vectors.composite).map_err(|source| StoreError::Corrupt {
        id: record.id.clone(),
        reason: source.to_string(),
      })?;
      if query.min_similarity.is_none_or(|minimum| similarity >= minimum) {
        neighbors.push(Neighbor {
          id: record.id.clone(),
          tenant: record.tenant.clone(),
          similarity,
        });
      }
    }
    // Ties are broken by id so that results are stable across runs and hash-map iteration orders.
    neighbors.sort_by(|a, b| b.similarity.total_cmp(&a.similarity).then_with(|| a.id.cmp(&b.id)));
    neighbors.truncate(query.limit);
    Ok(neighbors)
  }
}
