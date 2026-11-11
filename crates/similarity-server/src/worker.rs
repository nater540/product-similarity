//! Running a synchronous, CPU-bound model under an async server.
//!
//! A forward pass takes milliseconds to tens of milliseconds of solid CPU and must never run on a tokio worker,
//! where it would stall every other request on that thread. [`EmbedWorker`] owns one dedicated OS thread that owns
//! the model; request handlers hand it texts over a bounded queue and await the reply.
//!
//! One thread, not a pool: the backend already spreads a forward pass across cores, so a second concurrent pass
//! would only fight the first for them. What a single thread gives instead is **batching for free**. While one pass
//! runs, requests pile up in the queue; when it finishes, the worker takes everything waiting and runs it as one
//! pass. Under light load a request is served alone with no added delay; under heavy load batches grow on their own.
//! There is no timer to tune.
//!
//! The queue is bounded. When it is full the server says so immediately (`RESOURCE_EXHAUSTED`) instead of accepting
//! work it will only finish after the caller has timed out.

use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use similarity_core::Embedder;
use tokio::sync::oneshot;

/// Why texts could not be embedded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmbedError {
  /// Too many requests are already waiting for the model.
  #[error("the embedding queue is full")]
  Overloaded,
  /// The worker thread is gone. The process should be restarted.
  #[error("the embedding worker has stopped")]
  WorkerGone,
  /// The model failed on this input.
  #[error("embedding failed: {0}")]
  Failed(String),
}

/// Something that turns texts into unit vectors, asynchronously.
///
/// The engine is generic over this so that its logic can be tested with a deterministic stand-in instead of 90 MB
/// of weights.
pub trait TextEmbedder: Send + Sync + 'static {
  /// Embeds each text, in order.
  fn embed(&self, texts: Vec<String>) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send;

  /// The number of components in every vector.
  fn dimension(&self) -> usize;

  /// Identifies the model behind the vectors, see [`Embedder::fingerprint`].
  fn fingerprint(&self) -> &str;
}

/// An embedder that runs the model on the calling thread, with no queue in front of it.
///
/// For a command that is the model's only user (`import`): everything it embeds is one large batch after another,
/// so there is nothing to merge and nobody to keep waiting. The server never uses this; a forward pass on a
/// runtime thread would stall every other request.
#[derive(Debug)]
pub struct DirectEmbedder(pub Embedder);

impl TextEmbedder for DirectEmbedder {
  async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
    self
      .0
      .embed(&texts)
      .map_err(|error| EmbedError::Failed(error.to_string()))
  }

  fn dimension(&self) -> usize {
    self.0.dimension()
  }

  fn fingerprint(&self) -> &str {
    self.0.fingerprint()
  }
}

struct Job {
  texts: Vec<String>,
  reply: oneshot::Sender<Result<Vec<Vec<f32>>, EmbedError>>,
}

/// A handle to the embedding thread. Cheap to clone; the thread stops when the last handle is dropped.
#[derive(Debug, Clone)]
pub struct EmbedWorker {
  queue: SyncSender<Job>,
  dimension: usize,
  fingerprint: Arc<str>,
}

impl EmbedWorker {
  /// Moves `embedder` onto a new thread.
  ///
  /// `queue_depth` is how many requests may wait; `max_batch_texts` caps how many texts are merged into one call to
  /// the model (the model splits further by its own batch size).
  ///
  /// # Errors
  ///
  /// Returns the OS error if the thread cannot be spawned.
  pub fn spawn(embedder: Embedder, queue_depth: usize, max_batch_texts: usize) -> std::io::Result<Self> {
    let (queue, jobs) = sync_channel(queue_depth.max(1));
    let handle = Self {
      queue,
      dimension: embedder.dimension(),
      fingerprint: Arc::from(embedder.fingerprint()),
    };
    std::thread::Builder::new()
      .name("embed-worker".to_owned())
      .spawn(move || run(&embedder, &jobs, max_batch_texts.max(1)))?;
    Ok(handle)
  }
}

impl TextEmbedder for EmbedWorker {
  async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }
    let (reply, response) = oneshot::channel();
    self
      .queue
      .try_send(Job { texts, reply })
      .map_err(|refused| match refused {
        TrySendError::Full(_) => EmbedError::Overloaded,
        TrySendError::Disconnected(_) => EmbedError::WorkerGone,
      })?;
    response.await.map_err(|_| EmbedError::WorkerGone)?
  }

  fn dimension(&self) -> usize {
    self.dimension
  }

  fn fingerprint(&self) -> &str {
    &self.fingerprint
  }
}

fn run(embedder: &Embedder, jobs: &Receiver<Job>, max_batch_texts: usize) {
  // `recv` fails only when every handle is gone, which is the signal to stop.
  while let Ok(first) = jobs.recv() {
    let mut batch = vec![first];
    let mut texts = batch.iter().map(|job| job.texts.len()).sum::<usize>();
    while texts < max_batch_texts {
      let Ok(job) = jobs.try_recv() else { break };
      texts = texts.saturating_add(job.texts.len());
      batch.push(job);
    }
    serve(embedder, batch);
  }
  tracing::debug!("embedding worker stopped: every handle was dropped");
}

/// One call into the model, with a backend panic turned into an error.
///
/// A panic must cost the texts it was computing, not the thread: with the thread gone, every later request would
/// fail until the process is restarted.
fn embed_guarded<F>(embed: &F, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>
where
  F: Fn(&[&str]) -> similarity_core::Result<Vec<Vec<f32>>>,
{
  match catch_unwind(AssertUnwindSafe(|| embed(texts))) {
    Ok(Ok(vectors)) if vectors.len() == texts.len() => Ok(vectors),
    Ok(Ok(vectors)) => Err(EmbedError::Failed(format!(
      "the model returned {} vectors for {} texts",
      vectors.len(),
      texts.len()
    ))),
    Ok(Err(source)) => Err(EmbedError::Failed(source.to_string())),
    Err(_) => {
      tracing::error!(texts = texts.len(), "the embedding backend panicked");
      Err(EmbedError::Failed("the embedding backend panicked".to_owned()))
    }
  }
}

fn serve(embedder: &Embedder, batch: Vec<Job>) {
  serve_with(&|texts: &[&str]| embedder.embed(texts), batch);
}

/// Answers every job of `batch`, embedding them in one call when that works.
///
/// If the merged call fails, each job is embedded again on its own, so that a text the model cannot handle fails
/// the request it came from and no other. The retry costs a second pass only on a path that was already failing.
fn serve_with<F>(embed: &F, batch: Vec<Job>)
where
  F: Fn(&[&str]) -> similarity_core::Result<Vec<Vec<f32>>>,
{
  let all: Vec<&str> = batch
    .iter()
    .flat_map(|job| job.texts.iter().map(String::as_str))
    .collect();
  match embed_guarded(embed, &all) {
    Ok(vectors) => {
      let mut vectors = vectors.into_iter();
      for job in batch {
        let mine: Vec<Vec<f32>> = vectors.by_ref().take(job.texts.len()).collect();
        // The caller may have gone away (cancelled request); that is not this thread's problem.
        let _ = job.reply.send(Ok(mine));
      }
    }
    Err(error) if batch.len() == 1 => {
      for job in batch {
        let _ = job.reply.send(Err(error.clone()));
      }
    }
    Err(error) => {
      tracing::warn!(jobs = batch.len(), %error, "a merged embedding batch failed; retrying each request alone");
      for job in batch {
        let texts: Vec<&str> = job.texts.iter().map(String::as_str).collect();
        let _ = job.reply.send(embed_guarded(embed, &texts));
      }
    }
  }
}

#[cfg(test)]
pub(crate) mod testing {
  use std::sync::atomic::{AtomicUsize, Ordering};

  use similarity_core::vector::normalize;

  use super::{EmbedError, TextEmbedder};

  /// A deterministic stand-in for the model: a bag of hashed words.
  ///
  /// Texts that share words get similar vectors, and texts that share none are orthogonal, which is all the engine's
  /// logic needs from an embedder. It also counts what it was asked to embed, so tests can assert that unchanged
  /// fields never reach the model.
  #[derive(Debug, Default)]
  pub(crate) struct WordBagEmbedder {
    pub(crate) fingerprint: String,
    pub(crate) texts_embedded: AtomicUsize,
    pub(crate) fail_with: Option<EmbedError>,
  }

  pub(crate) const WORD_BAG_DIMENSION: usize = 64;

  impl WordBagEmbedder {
    pub(crate) fn new(fingerprint: &str) -> Self {
      Self {
        fingerprint: fingerprint.to_owned(),
        ..Self::default()
      }
    }

    pub(crate) fn embedded(&self) -> usize {
      self.texts_embedded.load(Ordering::SeqCst)
    }
  }

  pub(crate) fn word_bag(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0_f32; WORD_BAG_DIMENSION];
    for word in text.to_lowercase().split_whitespace() {
      // FNV-1a: stable across runs and platforms, unlike the standard library's randomized hasher.
      let hash = word.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
      });
      let slot = usize::try_from(hash % (WORD_BAG_DIMENSION as u64)).unwrap();
      vector[slot] += 1.0;
    }
    assert!(normalize(&mut vector), "text must contain at least one word");
    vector
  }

  impl TextEmbedder for WordBagEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
      if let Some(error) = &self.fail_with {
        return Err(error.clone());
      }
      self.texts_embedded.fetch_add(texts.len(), Ordering::SeqCst);
      Ok(texts.iter().map(|text| word_bag(text)).collect())
    }

    fn dimension(&self) -> usize {
      WORD_BAG_DIMENSION
    }

    fn fingerprint(&self) -> &str {
      &self.fingerprint
    }
  }
}

#[cfg(test)]
mod tests {
  use similarity_core::vector::cosine;

  use super::testing::{WordBagEmbedder, word_bag};
  use super::*;

  #[test]
  fn the_word_bag_embedder_orders_texts_the_way_a_real_model_would() {
    let gummies = word_bag("Blueberry Gummies");
    let stronger = word_bag("Blueberry Gummies 100mg");
    let vape = word_bag("Vape Cartridge");
    assert!(cosine(&gummies, &stronger).unwrap() > 0.7);
    assert!(cosine(&gummies, &vape).unwrap() < 0.3);
    assert!((cosine(&gummies, &word_bag("blueberry   GUMMIES")).unwrap() - 1.0).abs() < 1e-6);
  }

  /// A model that fails, or panics, on any call that includes a marked text.
  fn fragile(texts: &[&str]) -> similarity_core::Result<Vec<Vec<f32>>> {
    assert!(!texts.contains(&"panic"), "the backend blew up");
    if texts.contains(&"poison") {
      return Err(similarity_core::Error::Inference("cannot embed `poison`".to_owned()));
    }
    Ok(texts.iter().map(|text| word_bag(text)).collect())
  }

  type Reply = oneshot::Receiver<Result<Vec<Vec<f32>>, EmbedError>>;

  fn jobs(requests: &[&[&str]]) -> (Vec<Job>, Vec<Reply>) {
    requests
      .iter()
      .map(|texts| {
        let (reply, receiver) = oneshot::channel();
        let texts = texts.iter().map(|text| (*text).to_owned()).collect();
        (Job { texts, reply }, receiver)
      })
      .unzip()
  }

  #[test]
  fn each_request_of_a_merged_batch_gets_its_own_vectors_back_in_order() {
    let (batch, mut replies) = jobs(&[&["a", "b c"], &["d"]]);
    serve_with(&fragile, batch);
    assert_eq!(
      replies[0].try_recv().unwrap().unwrap(),
      [word_bag("a"), word_bag("b c")]
    );
    assert_eq!(replies[1].try_recv().unwrap().unwrap(), [word_bag("d")]);
  }

  #[test]
  fn a_text_the_model_cannot_embed_fails_only_the_request_it_came_from() {
    let (batch, mut replies) = jobs(&[&["blueberry gummies"], &["vape", "poison"], &["flower"]]);
    serve_with(&fragile, batch);
    assert_eq!(replies[0].try_recv().unwrap().unwrap(), [word_bag("blueberry gummies")]);
    let failed = replies[1].try_recv().unwrap().unwrap_err();
    assert!(
      matches!(&failed, EmbedError::Failed(detail) if detail.contains("poison")),
      "{failed}"
    );
    assert_eq!(replies[2].try_recv().unwrap().unwrap(), [word_bag("flower")]);
  }

  #[test]
  fn a_backend_panic_fails_only_the_request_it_came_from() {
    let (batch, mut replies) = jobs(&[&["panic"], &["flower"]]);
    serve_with(&fragile, batch);
    assert!(matches!(
      replies[0].try_recv().unwrap(),
      Err(EmbedError::Failed(detail)) if detail.contains("panicked")
    ));
    assert_eq!(replies[1].try_recv().unwrap().unwrap(), [word_bag("flower")]);
  }

  #[tokio::test]
  async fn the_stand_in_counts_what_it_embeds_and_can_be_made_to_fail() {
    let embedder = WordBagEmbedder::new("test");
    let vectors = embedder.embed(vec!["a b".to_owned(), "c".to_owned()]).await.unwrap();
    assert_eq!((vectors.len(), embedder.embedded()), (2, 2));

    let failing = WordBagEmbedder {
      fail_with: Some(EmbedError::Overloaded),
      ..WordBagEmbedder::new("test")
    };
    assert_eq!(failing.embed(vec!["a".to_owned()]).await, Err(EmbedError::Overloaded));
  }
}
