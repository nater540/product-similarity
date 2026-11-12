//! Where record vectors live.
//!
//! [`VectorStore`] is the whole contract the server needs: fetch a record's stored vectors by id, replace them
//! (always, or only if they are still what was read), delete them, and search for similar records. Two
//! implementations:
//!
//! - [`MemoryStore`]: a map behind a lock with brute-force search. Used by every test that is not about Qdrant, and
//!   a faithful reference for what the Qdrant implementation must do.
//! - [`QdrantStore`]: one Qdrant collection, one point per record.
//!
//! Both are constructed with a **layout id** (see `similarity_core::schema`). A record written under one layout is
//! never handed back, compared, or returned from a search under another: its vector blocks would be sliced at the
//! wrong offsets or have come from a different model, and the numbers would be wrong without looking wrong.

use std::future::Future;
use std::sync::Arc;

use similarity_core::{Quantities, RecordVectors};

mod memory;
mod qdrant;

pub use memory::MemoryStore;
pub use qdrant::{QdrantSettings, QdrantStore};

/// `Result` with this crate's [`StoreError`] as the default error type.
pub type Result<T, E = StoreError> = std::result::Result<T, E>;

/// Everything that can go wrong talking to a vector store.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
  /// The store could not be reached, or refused the request.
  #[error("vector store unavailable: {0}")]
  Unavailable(String),

  /// The record exists, but was written by a server with a different model or field set.
  #[error("record `{id}` was stored under layout `{stored}`, but this server uses `{expected}`; re-index it")]
  LayoutMismatch {
    /// The record's id.
    id: String,
    /// The layout the record was written under.
    stored: String,
    /// The layout this store was opened with.
    expected: String,
  },

  /// The backing collection exists but cannot hold this server's vectors.
  #[error("collection `{collection}` is not usable: {reason}")]
  Collection {
    /// The collection's name.
    collection: String,
    /// What is wrong with it.
    reason: String,
  },

  /// A stored record could not be decoded.
  #[error("stored record `{id}` is malformed: {reason}")]
  Corrupt {
    /// The record's id.
    id: String,
    /// What is wrong with it.
    reason: String,
  },

  /// The caller asked for something that can never succeed.
  #[error("invalid store request: {0}")]
  Invalid(String),
}

/// A record as stored.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRecord {
  /// The caller's identifier.
  pub id: String,
  /// The key of the brand that owns the record, if it has one. An identifier: the brand's name is a text field.
  pub brand_id: Option<String>,
  /// The current baseline: what the next revision is compared against, and what searches match.
  pub vectors: RecordVectors,
  /// The composite of the record as first stored (or last force-written), for drift detection.
  pub anchor: Vec<f32>,
  /// The record's current quantities: what the next revision's are compared against. They are stored beside the
  /// vectors and are part of neither, so a record written before quantities existed simply has none.
  pub quantities: Quantities,
  /// The first value the record had for each quantity since it was first stored (or last force-written): what a
  /// quantity that was removed and comes back is compared against.
  pub anchor_quantities: Quantities,
  /// How many times the record has been written.
  pub revision: u64,
  /// When the record was last written, in seconds since the Unix epoch.
  pub updated_at: u64,
}

impl StoredRecord {
  /// Which write of the record this is: what a conditional write or delete names to say "still this one".
  #[must_use]
  pub const fn version(&self) -> Version {
    Version {
      revision: self.revision,
      updated_at: self.updated_at,
    }
  }
}

/// One write of a record, as a later writer names it.
///
/// A record's revision goes up by one with every write, so within the life of one record a version names one
/// write. A record that is deleted and stored again starts over at revision 1, which is why the time is part of
/// it: the two lives are told apart unless both were written in the same second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
  /// [`StoredRecord::revision`] of that write.
  pub revision: u64,
  /// [`StoredRecord::updated_at`] of that write.
  pub updated_at: u64,
}

/// What a conditional write expects to find under the record's id. The write happens only if it is so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expected {
  /// Nothing: the write creates the record. A record with that id, under any layout, is left as it is.
  Absent,
  /// The record that was read, unchanged since: this store's layout, and this version.
  Version(Version),
}

/// A similarity search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchQuery<'a> {
  /// The composite vector to search around.
  pub vector: &'a [f32],
  /// The most neighbors to return.
  pub limit: usize,
  /// Only match records of this brand.
  pub brand_id: Option<&'a str>,
  /// Never return the record with this id (typically the record being searched around).
  pub exclude_id: Option<&'a str>,
  /// Only return records with at least this cosine similarity.
  pub min_similarity: Option<f32>,
}

/// One search result.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor {
  /// The matching record's id.
  pub id: String,
  /// The key of the brand that owns the matching record.
  pub brand_id: Option<String>,
  /// Cosine similarity between the query and the record's current composite.
  pub similarity: f32,
}

/// A search result with what scoring it exactly needs: the record's current vectors and quantities.
#[derive(Debug, Clone, PartialEq)]
pub struct Found {
  /// The result as [`VectorStore::search`] would give it.
  pub neighbor: Neighbor,
  /// The record's current composite and the fingerprints of its texts.
  pub vectors: RecordVectors,
  /// The record's current quantities.
  pub quantities: Quantities,
}

/// The storage contract.
///
/// Methods return `impl Future + Send` rather than using `async fn` so that a server generic over the store can
/// spawn the futures onto a multi-threaded runtime.
pub trait VectorStore: Send + Sync + 'static {
  /// Fetches a record by id. `Ok(None)` if there is none.
  ///
  /// # Errors
  ///
  /// [`StoreError::LayoutMismatch`] if the record exists under a different layout, [`StoreError::Unavailable`] or
  /// [`StoreError::Corrupt`] if it cannot be read.
  fn get(&self, id: &str) -> impl Future<Output = Result<Option<StoredRecord>>> + Send;

  /// Writes a record, replacing any record with the same id.
  ///
  /// The write is durable and visible to [`Self::get`] and [`Self::search`] when the future resolves.
  ///
  /// # Errors
  ///
  /// [`StoreError::Invalid`] if a vector has the wrong length, [`StoreError::Unavailable`] if the write fails.
  fn put(&self, record: &StoredRecord) -> impl Future<Output = Result<()>> + Send;

  /// Writes a record only if what is stored under its id is still what `expected` says: read, decide, and write
  /// as one step, for writers that cannot share a lock. Returns whether the record was written. `Ok(false)` means
  /// another writer got there first and nothing was changed: read again and decide again.
  ///
  /// A record that has been deleted is never brought back by a write that expected a version of it.
  ///
  /// # Errors
  ///
  /// As [`Self::put`]. An error says nothing about whether the write happened.
  fn put_if(&self, record: &StoredRecord, expected: Expected) -> impl Future<Output = Result<bool>> + Send;

  /// Deletes a record. Returns whether there was one, under any layout.
  ///
  /// # Errors
  ///
  /// [`StoreError::Unavailable`] if the delete fails.
  fn delete(&self, id: &str) -> impl Future<Output = Result<bool>> + Send;

  /// Deletes a record only if it is still at `version`, under this store's layout. Returns whether the record is
  /// gone: `Ok(false)` means a record is stored under that id that is not that version (another writer changed
  /// it, or it is another layout's), and it was left as it is. A record that was not there is gone.
  ///
  /// # Errors
  ///
  /// [`StoreError::Unavailable`] if the delete fails.
  fn delete_if(&self, id: &str, version: Version) -> impl Future<Output = Result<bool>> + Send;

  /// Finds the records most similar to a vector, most similar first. Only records of this store's layout match.
  ///
  /// # Errors
  ///
  /// [`StoreError::Invalid`] if the query vector has the wrong length, [`StoreError::Unavailable`] if the search
  /// fails.
  fn search(&self, query: SearchQuery<'_>) -> impl Future<Output = Result<Vec<Neighbor>>> + Send;

  /// [`Self::search`], with each result's current vectors and quantities: the same filters, the same order, the
  /// same limit. It is what a caller that scores the results again (a match) asks for in place of one `get` per
  /// result. The anchor is not returned.
  ///
  /// # Errors
  ///
  /// As [`Self::search`], and [`StoreError::Corrupt`] if a result's vectors cannot be read.
  fn search_vectors(&self, query: SearchQuery<'_>) -> impl Future<Output = Result<Vec<Found>>> + Send;
}

/// A shared store is a store: two engines in one process, or a test standing in for two replicas, hold one each.
impl<S: VectorStore> VectorStore for Arc<S> {
  fn get(&self, id: &str) -> impl Future<Output = Result<Option<StoredRecord>>> + Send {
    self.as_ref().get(id)
  }

  fn put(&self, record: &StoredRecord) -> impl Future<Output = Result<()>> + Send {
    self.as_ref().put(record)
  }

  fn put_if(&self, record: &StoredRecord, expected: Expected) -> impl Future<Output = Result<bool>> + Send {
    self.as_ref().put_if(record, expected)
  }

  fn delete(&self, id: &str) -> impl Future<Output = Result<bool>> + Send {
    self.as_ref().delete(id)
  }

  fn delete_if(&self, id: &str, version: Version) -> impl Future<Output = Result<bool>> + Send {
    self.as_ref().delete_if(id, version)
  }

  fn search(&self, query: SearchQuery<'_>) -> impl Future<Output = Result<Vec<Neighbor>>> + Send {
    self.as_ref().search(query)
  }

  fn search_vectors(&self, query: SearchQuery<'_>) -> impl Future<Output = Result<Vec<Found>>> + Send {
    self.as_ref().search_vectors(query)
  }
}
