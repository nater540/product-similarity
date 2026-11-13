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
//! The number of requests that may wait is bounded. When that many are waiting the server says so immediately
//! (`RESOURCE_EXHAUSTED`) instead of accepting work it will only finish after the caller has timed out.
//!
//! The thread works only for callers that are still waiting, and for as long as it can it works in the order
//! they came:
//!
//! - A request that was cancelled, or whose deadline ran out, drops its side of the reply channel, and the
//!   thread passes over it. It also gives up its place among the waiting the moment it goes, so a queue of
//!   requests nobody wants refuses nobody.
//! - A request that says when its caller stops waiting ([`by`]) is in a pass only if the pass can end before
//!   then, even at the fastest rate the model has shown ([`fit`]). One that is out of time, or that the pass
//!   about to run leaves no time for, is told so ([`EmbedError::TooLate`]) and not embedded.
//! - Once any request has been lost that way, the thread is behind, and waiting in order is what loses them:
//!   the request at the head of the queue is always the one closest to its deadline. So while it is behind it
//!   takes the newest requests first, which have the most time left, also for whatever the request still has to
//!   do after the model has answered. When a pass is put together and nobody has been lost, it is back to the
//!   order they came in.

use std::collections::VecDeque;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

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
  /// The caller's deadline passed while the request waited for the model, or the texts waiting ahead of it could
  /// not be embedded before it does. Nothing was embedded for it.
  #[error("the request's deadline leaves no time for the model to reach it")]
  TooLate,
}

tokio::task_local! {
  /// When the caller of the request this task is serving stops waiting for it.
  static DEADLINE: Instant;
}

/// Runs `work` as a request whose caller stops waiting at `deadline`; with `None`, as it is.
///
/// The deadline is a fact about the request, not an argument of any one call: it holds for everything the
/// request does on this task, and [`EmbedWorker`] reads it where it queues texts. Work that is moved to another
/// task does not carry it.
pub async fn by<F: Future>(deadline: Option<Instant>, work: F) -> F::Output {
  match deadline {
    Some(deadline) => DEADLINE.scope(deadline, work).await,
    None => work.await,
  }
}

/// The deadline of the request this task is serving, if it has one.
fn deadline() -> Option<Instant> {
  DEADLINE.try_with(|deadline| *deadline).ok()
}

/// The least time `texts` texts can take: at `fastest_ns_per_text`, the fastest rate any forward pass has
/// reached, or `None` before there has been one.
///
/// A lower bound on purpose. It is used to refuse a request, and a request must only be refused for a deadline
/// that cannot be met: one that might be is let in.
fn least_wait(texts: u64, fastest_ns_per_text: u64) -> Option<Duration> {
  (fastest_ns_per_text > 0).then(|| Duration::from_nanos(texts.saturating_mul(fastest_ns_per_text)))
}

/// A request's place among the requests waiting for the model.
///
/// Given up exactly once: by the thread when the request stops waiting (it is taken into a pass, or refused),
/// or by the request when it goes away first. The second is what keeps a queue of abandoned requests from
/// refusing new ones until the thread has got round to passing each of them over.
#[derive(Debug)]
struct Waiting {
  stats: Arc<WorkerStats>,
  left: AtomicBool,
}

impl Waiting {
  /// Takes a place, unless `depth` requests are waiting already.
  fn join(stats: &Arc<WorkerStats>, depth: u64) -> Option<Arc<Self>> {
    stats
      .waiting
      .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |waiting| {
        (waiting < depth).then(|| waiting.saturating_add(1))
      })
      .ok()?;
    Some(Arc::new(Self {
      stats: Arc::clone(stats),
      left: AtomicBool::new(false),
    }))
  }

  fn leave(&self) {
    if !self.left.swap(true, Ordering::Relaxed) {
      self.stats.waiting.fetch_sub(1, Ordering::Relaxed);
    }
  }
}

/// Gives up a request's place when the request ends, however it ends.
struct Leaving(Arc<Waiting>);

impl Drop for Leaving {
  fn drop(&mut self) {
    self.0.leave();
  }
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

  /// Whether this embedder can still embed at all. An embedder that runs the model where it is called always
  /// can; one that hands texts to a thread cannot once that thread is gone, and a server that is asked whether
  /// it is ready has to know.
  fn alive(&self) -> bool {
    true
  }
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
  /// When the job joined the queue, for the time it waited there.
  queued: Instant,
  /// When the caller stops waiting, if it said.
  deadline: Option<Instant>,
  waiting: Arc<Waiting>,
}

/// What the embedding thread has done since it started. Every count only grows, so two snapshots subtracted
/// ([`WorkerSnapshot::since`]) describe the time between them.
///
/// A client sees how long a request took and nothing about why. These say why: how many requests shared a
/// forward pass, how long they queued for one, and how much of the time the model was busy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerSnapshot {
  /// Calls into the model, each serving one or more requests.
  pub batches: u64,
  /// Requests served.
  pub jobs: u64,
  /// Texts embedded.
  pub texts: u64,
  /// Requests refused because the queue was full.
  pub overloaded: u64,
  /// Requests passed over because their caller had gone by the time the thread reached them.
  pub abandoned: u64,
  /// Requests refused because their deadline left no time: at once, or when it passed in the queue.
  pub too_late: u64,
  /// Nanoseconds spent inside the model.
  pub busy_ns: u64,
  /// Nanoseconds requests spent in the queue, summed over requests.
  pub queue_wait_ns: u64,
}

impl WorkerSnapshot {
  /// What happened between `earlier` and this snapshot.
  #[must_use]
  pub fn since(&self, earlier: &Self) -> Self {
    Self {
      batches: self.batches.saturating_sub(earlier.batches),
      jobs: self.jobs.saturating_sub(earlier.jobs),
      texts: self.texts.saturating_sub(earlier.texts),
      overloaded: self.overloaded.saturating_sub(earlier.overloaded),
      abandoned: self.abandoned.saturating_sub(earlier.abandoned),
      too_late: self.too_late.saturating_sub(earlier.too_late),
      busy_ns: self.busy_ns.saturating_sub(earlier.busy_ns),
      queue_wait_ns: self.queue_wait_ns.saturating_sub(earlier.queue_wait_ns),
    }
  }
}

/// The counters behind a [`WorkerSnapshot`], shared between the handles and the thread. Relaxed ordering
/// throughout: these are statistics, not synchronization.
#[derive(Debug, Default)]
struct WorkerStats {
  batches: AtomicU64,
  jobs: AtomicU64,
  texts: AtomicU64,
  overloaded: AtomicU64,
  abandoned: AtomicU64,
  too_late: AtomicU64,
  busy_ns: AtomicU64,
  queue_wait_ns: AtomicU64,
  /// Requests waiting for the model whose callers are still there. Not a count that only grows, and not in a
  /// snapshot.
  waiting: AtomicU64,
  /// The least time per text any forward pass that succeeded has taken, in nanoseconds; zero until there is one.
  fastest_ns_per_text: AtomicU64,
  /// Set when the thread ends, for whatever reason.
  stopped: AtomicBool,
}

fn count(value: usize) -> u64 {
  u64::try_from(value).unwrap_or(u64::MAX)
}

fn nanoseconds(elapsed: Duration) -> u64 {
  u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

/// A handle to the embedding thread. Cheap to clone; the thread stops when the last handle is dropped.
#[derive(Debug, Clone)]
pub struct EmbedWorker {
  queue: Sender<Job>,
  /// How many requests may wait.
  depth: u64,
  dimension: usize,
  fingerprint: Arc<str>,
  stats: Arc<WorkerStats>,
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
    // Not a bounded channel: the bound is on requests whose callers are waiting, wherever the thread has them.
    let (queue, jobs) = channel();
    let handle = Self {
      queue,
      depth: count(queue_depth.max(1)),
      dimension: embedder.dimension(),
      fingerprint: Arc::from(embedder.fingerprint()),
      stats: Arc::default(),
    };
    let stats = Arc::clone(&handle.stats);
    std::thread::Builder::new()
      .name("embed-worker".to_owned())
      .spawn(move || {
        // Dropped however the thread ends, a panic included: that is the one way to learn that it has.
        let _stopped = Stopped(Arc::clone(&stats));
        run(&embedder, &jobs, max_batch_texts.max(1), &stats);
      })?;
    Ok(handle)
  }

  /// How many requests are waiting for the model right now.
  pub fn waiting(&self) -> u64 {
    self.stats.waiting.load(Ordering::Relaxed)
  }

  /// What the embedding thread has done so far.
  pub fn stats(&self) -> WorkerSnapshot {
    let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    WorkerSnapshot {
      batches: read(&self.stats.batches),
      jobs: read(&self.stats.jobs),
      texts: read(&self.stats.texts),
      overloaded: read(&self.stats.overloaded),
      abandoned: read(&self.stats.abandoned),
      too_late: read(&self.stats.too_late),
      busy_ns: read(&self.stats.busy_ns),
      queue_wait_ns: read(&self.stats.queue_wait_ns),
    }
  }
}

/// Marks the embedding thread as gone when the thread lets go of it.
struct Stopped(Arc<WorkerStats>);

impl Drop for Stopped {
  fn drop(&mut self) {
    self.0.stopped.store(true, Ordering::Relaxed);
  }
}

impl TextEmbedder for EmbedWorker {
  fn alive(&self) -> bool {
    !self.stats.stopped.load(Ordering::Relaxed)
  }

  async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }
    let queued = Instant::now();
    let deadline = deadline();
    if deadline.is_some_and(|deadline| deadline <= queued) {
      self.stats.too_late.fetch_add(1, Ordering::Relaxed);
      return Err(EmbedError::TooLate);
    }
    let Some(waiting) = Waiting::join(&self.stats, self.depth) else {
      self.stats.overloaded.fetch_add(1, Ordering::Relaxed);
      return Err(EmbedError::Overloaded);
    };
    // Dropped when this request ends, which is also when it is dropped unfinished.
    let leaving = Leaving(waiting);
    let (reply, response) = oneshot::channel();
    self
      .queue
      .send(Job {
        texts,
        reply,
        queued,
        deadline,
        waiting: Arc::clone(&leaving.0),
      })
      .map_err(|_| EmbedError::WorkerGone)?;
    response.await.map_err(|_| EmbedError::WorkerGone)?
  }

  fn dimension(&self) -> usize {
    self.dimension
  }

  fn fingerprint(&self) -> &str {
    &self.fingerprint
  }
}

/// A job, if anybody still wants its answer and it is not already too late for one.
///
/// A job whose caller has gone is dropped: whatever was waiting on it was dropped first, so there is nobody to
/// tell. One whose deadline has passed is answered [`EmbedError::TooLate`]: its caller is still there (its own
/// timer has not fired yet, or it set none) and would otherwise get vectors it has no time left to use.
fn wanted(job: Job, now: Instant, stats: &WorkerStats) -> Option<Job> {
  if job.reply.is_closed() {
    job.waiting.leave();
    stats.abandoned.fetch_add(1, Ordering::Relaxed);
    return None;
  }
  if job.deadline.is_some_and(|deadline| deadline <= now) {
    return too_late(job, stats);
  }
  Some(job)
}

/// Tells a job's caller that its deadline leaves no time, and counts it.
fn too_late(job: Job, stats: &WorkerStats) -> Option<Job> {
  job.waiting.leave();
  stats.too_late.fetch_add(1, Ordering::Relaxed);
  let _ = job.reply.send(Err(EmbedError::TooLate));
  None
}

fn texts_in(batch: &[Job]) -> usize {
  batch.iter().map(|job| job.texts.len()).sum()
}

/// The jobs of `batch` that one pass can answer in time, in the order given. The rest are told they are too
/// late.
///
/// A pass answers every request in it when it ends, and it cannot end sooner than all of its texts take at the
/// fastest rate the model has shown. So a request is only worth its place if the pass, with it in, can still
/// end before its deadline. Requests are let in with the most time left first, which is what makes room for the
/// most of them: each one that is let in only has to leave time for those let in before it, which have more.
/// Requests with no deadline are always in. So is the first one considered when the pass is otherwise empty:
/// a request alone in a pass is never refused on a guess at how long its own texts take.
///
/// Before the first pass there is no rate, and everything is let in.
fn fit(batch: Vec<Job>, now: Instant, stats: &WorkerStats) -> Vec<Job> {
  let fastest = stats.fastest_ns_per_text.load(Ordering::Relaxed);
  let mut texts: u64 = batch
    .iter()
    .filter(|job| job.deadline.is_none())
    .map(|job| count(job.texts.len()))
    .sum();
  let mut by_time_left: Vec<(Instant, usize, u64)> = batch
    .iter()
    .enumerate()
    .filter_map(|(place, job)| Some((job.deadline?, place, count(job.texts.len()))))
    .collect();
  by_time_left.sort_unstable_by(|a, b| b.cmp(a));
  let mut late = std::collections::HashSet::new();
  for (deadline, place, own) in by_time_left {
    let with = texts.saturating_add(own);
    let ends = now.checked_add(least_wait(with, fastest).unwrap_or_default());
    if texts == 0 || ends.is_some_and(|ends| ends < deadline) {
      texts = with;
    } else {
      late.insert(place);
    }
  }
  batch
    .into_iter()
    .enumerate()
    .filter_map(|(place, job)| {
      if late.contains(&place) {
        too_late(job, stats)
      } else {
        Some(job)
      }
    })
    .collect()
}

/// Takes the next pass out of `backlog`, the requests waiting in the order they came, and says whether any
/// request was lost while doing it. May be empty, and then nothing is left waiting.
///
/// First everything nobody can use is dropped, wherever it stands ([`wanted`]). Then the pass is filled up to
/// its limit and held to what it can answer in time ([`fit`]); a place that frees is filled again. Last, a
/// request that is left waiting, and whose deadline will have passed before the pass about to run can end, is
/// told now and not when it has.
///
/// The pass is filled with the oldest requests first, unless the thread is `behind`: a request was lost the
/// last time, or has been this time. Then it is filled with the newest. In order, the request being served is
/// always the one that has waited longest, and once requests are being lost that is the one with least time
/// left: its answer arrives as its caller gives up, or leaves no time for what the request does next, and every
/// request behind it ages into the same place. The newest have the most time, and the old ones that cannot all
/// be served anyway are the ones that are lost.
fn gather(
  backlog: &mut VecDeque<Job>,
  now: Instant,
  max_batch_texts: usize,
  behind: bool,
  stats: &WorkerStats,
) -> (Vec<Job>, bool) {
  let came = backlog.len();
  let mut waiting: VecDeque<Job> = std::mem::take(backlog)
    .into_iter()
    .filter_map(|job| wanted(job, now, stats))
    .collect();
  let mut lost = waiting.len() < came;

  let mut batch: Vec<Job> = Vec::new();
  loop {
    let mut texts = texts_in(&batch);
    while texts < max_batch_texts {
      let next = if behind || lost {
        waiting.pop_back()
      } else {
        waiting.pop_front()
      };
      let Some(job) = next else { break };
      texts = texts.saturating_add(job.texts.len());
      batch.push(job);
    }
    let filled = batch.len();
    batch = fit(batch, now, stats);
    if batch.len() == filled {
      break;
    }
    lost = true;
    if waiting.is_empty() {
      break;
    }
  }

  let fastest = stats.fastest_ns_per_text.load(Ordering::Relaxed);
  if let Some(ends) = least_wait(count(texts_in(&batch)), fastest).and_then(|least| now.checked_add(least)) {
    let left = waiting.len();
    waiting = waiting
      .into_iter()
      .filter_map(|job| {
        if job.deadline.is_some_and(|deadline| deadline <= ends) {
          too_late(job, stats)
        } else {
          Some(job)
        }
      })
      .collect();
    lost |= waiting.len() < left;
  }
  for job in &batch {
    job.waiting.leave();
  }
  *backlog = waiting;
  (batch, lost)
}

/// Remembers a forward pass that succeeded, if it was the fastest so far per text.
fn remember(stats: &WorkerStats, texts: u64, elapsed: Duration) {
  // At least one, which is what tells a rate from "no pass yet".
  let Some(rate) = nanoseconds(elapsed).checked_div(texts).map(|rate| rate.max(1)) else {
    return;
  };
  let _ = stats
    .fastest_ns_per_text
    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |fastest| {
      (fastest == 0 || rate < fastest).then_some(rate)
    });
}

fn run(embedder: &Embedder, jobs: &Receiver<Job>, max_batch_texts: usize, stats: &WorkerStats) {
  // The requests waiting, in the order they came, and whether one was lost when the last pass was put together.
  let mut backlog = VecDeque::new();
  let mut behind = false;
  loop {
    if backlog.is_empty() {
      // `recv` fails only when every handle is gone, which is the signal to stop.
      let Ok(first) = jobs.recv() else { break };
      backlog.push_back(first);
    }
    backlog.extend(jobs.try_iter());
    let (batch, lost) = gather(&mut backlog, Instant::now(), max_batch_texts, behind, stats);
    behind = lost;
    if batch.is_empty() {
      continue;
    }
    let texts = count(texts_in(&batch));
    let taken = Instant::now();
    let waited: Duration = batch
      .iter()
      .map(|job| taken.saturating_duration_since(job.queued))
      .sum();
    stats.batches.fetch_add(1, Ordering::Relaxed);
    stats.jobs.fetch_add(count(batch.len()), Ordering::Relaxed);
    stats.texts.fetch_add(texts, Ordering::Relaxed);
    stats.queue_wait_ns.fetch_add(nanoseconds(waited), Ordering::Relaxed);
    let succeeded = serve(embedder, batch);
    let elapsed = taken.elapsed();
    if succeeded {
      // A pass that failed says nothing about how fast the model is: an error can come back in no time at all.
      remember(stats, texts, elapsed);
    }
    stats.busy_ns.fetch_add(nanoseconds(elapsed), Ordering::Relaxed);
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

fn serve(embedder: &Embedder, batch: Vec<Job>) -> bool {
  serve_with(&|texts: &[&str]| embedder.embed(texts), batch)
}

/// Answers every job of `batch`, embedding them in one call when that works, and says whether it did.
///
/// If the merged call fails, each job is embedded again on its own, so that a text the model cannot handle fails
/// the request it came from and no other. The retry costs a second pass only on a path that was already failing.
fn serve_with<F>(embed: &F, batch: Vec<Job>) -> bool
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
        // The caller may have gone away while the model ran; nothing can be done about a pass that is over.
        let _ = job.reply.send(Ok(mine));
      }
      true
    }
    Err(error) if batch.len() == 1 => {
      for job in batch {
        let _ = job.reply.send(Err(error.clone()));
      }
      false
    }
    Err(error) => {
      tracing::warn!(jobs = batch.len(), %error, "a merged embedding batch failed; retrying each request alone");
      for job in batch {
        let texts: Vec<&str> = job.texts.iter().map(String::as_str).collect();
        let _ = job.reply.send(embed_guarded(embed, &texts));
      }
      false
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
    jobs_counted(requests, &Arc::default())
  }

  /// Jobs as `embed` queues them: each has a place among the requests waiting in `stats`.
  fn jobs_counted(requests: &[&[&str]], stats: &Arc<WorkerStats>) -> (Vec<Job>, Vec<Reply>) {
    requests
      .iter()
      .map(|texts| {
        let (reply, receiver) = oneshot::channel();
        let texts: Vec<String> = texts.iter().map(|text| (*text).to_owned()).collect();
        (
          Job {
            waiting: Waiting::join(stats, u64::MAX).unwrap(),
            texts,
            reply,
            queued: Instant::now(),
            deadline: None,
          },
          receiver,
        )
      })
      .unzip()
  }

  fn texts_of(batch: &[Job]) -> Vec<&str> {
    batch
      .iter()
      .flat_map(|job| job.texts.iter().map(String::as_str))
      .collect()
  }

  fn waiting_in(stats: &WorkerStats) -> u64 {
    stats.waiting.load(Ordering::Relaxed)
  }

  /// One second a text: a rate at which every bound in a test is a whole number of seconds.
  fn slow() -> Arc<WorkerStats> {
    let stats = Arc::<WorkerStats>::default();
    stats.fastest_ns_per_text.store(1_000_000_000, Ordering::Relaxed);
    stats
  }

  /// The next pass out of `jobs`, for a thread that was not behind: the pass, what is left waiting, and whether
  /// a request was lost.
  fn pass(jobs: Vec<Job>, limit: usize, stats: &WorkerStats) -> (Vec<Job>, VecDeque<Job>, bool) {
    let mut backlog: VecDeque<Job> = jobs.into();
    let (batch, lost) = gather(&mut backlog, Instant::now(), limit, false, stats);
    (batch, backlog, lost)
  }

  /// The thread works for callers that are waiting. A request that was cancelled while it waited has dropped
  /// its side of the reply, and its texts never reach the model, wherever it stood.
  #[test]
  fn a_request_whose_caller_has_gone_is_passed_over_and_counted() {
    let stats = Arc::<WorkerStats>::default();
    let (jobs, mut replies) = jobs_counted(&[&["gone first"], &["a", "b"], &["gone", "too"], &["c"]], &stats);
    assert_eq!(waiting_in(&stats), 4);
    drop(replies.remove(2));
    drop(replies.remove(0));

    let (batch, left, lost) = pass(jobs, 64, &stats);
    // Requests were lost, so the newest of the rest comes first.
    assert_eq!(texts_of(&batch), ["c", "a", "b"]);
    assert!(left.is_empty() && lost);
    assert_eq!(stats.abandoned.load(Ordering::Relaxed), 2);
    assert_eq!(stats.too_late.load(Ordering::Relaxed), 0);
    assert_eq!(
      waiting_in(&stats),
      0,
      "a request in a pass is no longer waiting for one"
    );

    assert!(serve_with(&fragile, batch));
    assert_eq!(replies[0].try_recv().unwrap().unwrap(), [word_bag("a"), word_bag("b")]);
    assert_eq!(replies[1].try_recv().unwrap().unwrap(), [word_bag("c")]);
  }

  /// A queue of nothing but requests nobody is waiting for costs the model nothing at all: all of them go at
  /// once, however many a pass may hold.
  #[test]
  fn a_queue_of_abandoned_requests_is_emptied_without_a_forward_pass() {
    let stats = Arc::<WorkerStats>::default();
    let (jobs, replies) = jobs_counted(&[&["a"], &["b"], &["c"]], &stats);
    drop(replies);
    let (batch, left, lost) = pass(jobs, 1, &stats);
    assert!(batch.is_empty() && left.is_empty() && lost);
    assert_eq!(stats.abandoned.load(Ordering::Relaxed), 3);
  }

  /// A deadline that passed while the request waited is answered as one, not with vectors there is no time left
  /// to use. The caller is still there: its own timer has not fired, or it set none.
  #[test]
  fn a_request_whose_deadline_passed_while_it_waited_is_told_so_and_not_embedded() {
    let stats = Arc::<WorkerStats>::default();
    let (mut jobs, mut replies) = jobs_counted(&[&["late"], &["in time"], &["no deadline"]], &stats);
    let now = Instant::now();
    jobs[0].deadline = Some(now);
    jobs[1].deadline = Some(now + Duration::from_mins(1));

    let (batch, _, lost) = pass(jobs, 64, &stats);
    assert_eq!(texts_of(&batch), ["no deadline", "in time"]);
    assert!(lost);
    assert_eq!(replies[0].try_recv().unwrap(), Err(EmbedError::TooLate));
    assert_eq!(stats.too_late.load(Ordering::Relaxed), 1);
    assert_eq!(stats.abandoned.load(Ordering::Relaxed), 0);
    assert_eq!(waiting_in(&stats), 0);
  }

  /// While nobody is being lost, requests are served in the order they came, a pass at a time, and what does not
  /// fit waits where it was.
  #[test]
  fn requests_are_served_in_the_order_they_came_while_none_is_lost() {
    let stats = slow();
    let (mut jobs, _replies) = jobs_counted(&[&["a", "b"], &["c"], &["d"], &["e"]], &stats);
    for job in &mut jobs {
      job.deadline = Some(Instant::now() + Duration::from_mins(10));
    }
    let (batch, left, lost) = pass(jobs, 3, &stats);
    assert_eq!(texts_of(&batch), ["a", "b", "c"]);
    assert!(!lost);
    assert_eq!(
      left.iter().map(|job| job.texts[0].as_str()).collect::<Vec<_>>(),
      ["d", "e"]
    );
    assert_eq!(waiting_in(&stats), 2, "the two that are still waiting");

    // A first job that alone exceeds the limit is still served: the model splits it further.
    let (jobs, _replies) = jobs_counted(&[&["a", "b", "c"], &["d"]], &stats);
    let (batch, left, _) = pass(jobs, 2, &stats);
    assert_eq!(texts_of(&batch), ["a", "b", "c"]);
    assert_eq!(left.len(), 1);
  }

  /// A pass answers all of its requests when it ends, so a request is worth its place only if the pass, with it
  /// in, can still end before its deadline at the fastest rate the model has shown. The request that cannot be
  /// answered in time is told so, wherever it stood, and the ones that can are.
  #[test]
  fn a_pass_is_made_of_the_requests_it_can_answer_in_time() {
    let stats = slow();
    let requests: [&[&str]; 4] = [&["a"; 50], &["tight"], &["roomy"], &["unbounded"]];
    let (mut jobs, mut replies) = jobs_counted(&requests, &stats);
    let now = Instant::now();
    // Fifty seconds of texts and ten to do them in, at the head of the queue.
    jobs[0].deadline = Some(now + Duration::from_secs(10));
    jobs[1].deadline = Some(now + Duration::from_secs(40));
    jobs[2].deadline = Some(now + Duration::from_mins(10));

    let (batch, left, lost) = pass(jobs, 64, &stats);
    let mut answered = texts_of(&batch);
    answered.sort_unstable();
    assert_eq!(
      answered,
      ["roomy", "tight", "unbounded"],
      "three seconds of texts, and all of them can wait that long"
    );
    assert!(left.is_empty() && lost);
    assert_eq!(replies[0].try_recv().unwrap(), Err(EmbedError::TooLate));
    assert_eq!(stats.too_late.load(Ordering::Relaxed), 1);
    assert_eq!(waiting_in(&stats), 0);
  }

  /// Letting requests in with the most time left first is what lets the most in: one with little time left has
  /// to leave room only for itself and the ones that can wait longer.
  #[test]
  fn requests_are_fitted_into_a_pass_with_the_most_time_left_first() {
    let stats = slow();
    let requests: [&[&str]; 4] = [&["two", "texts"], &["one"], &["x"; 3], &["last"]];
    let (mut jobs, mut replies) = jobs_counted(&requests, &stats);
    let now = Instant::now();
    let within = |seconds: u64| Some(now + Duration::from_secs(seconds));
    // In the order of time left: 30 s (three texts), 20 s (one), 6 s (two), 5 s (one).
    jobs[0].deadline = within(6);
    jobs[1].deadline = within(20);
    jobs[2].deadline = within(30);
    jobs[3].deadline = within(5);

    let batch = fit(jobs, now, &stats);
    // Three texts, then four, are done within 30 and 20 seconds. Six texts are not done within six seconds, and
    // the two-text request is out. The last would make five texts, in exactly its five seconds: not before.
    assert_eq!(texts_of(&batch), ["one", "x", "x", "x"], "in the order given");
    assert_eq!(replies[0].try_recv().unwrap(), Err(EmbedError::TooLate));
    assert_eq!(replies[3].try_recv().unwrap(), Err(EmbedError::TooLate));
    assert_eq!(stats.too_late.load(Ordering::Relaxed), 2);
  }

  /// A request alone in a pass is never refused on a guess at its own texts: the rate is the fastest the model
  /// has shown, on other texts, and nobody else is waiting for the place. And before the first pass there is no
  /// rate at all, and only a deadline that has passed is too late.
  #[test]
  fn a_request_alone_in_a_pass_is_served_whatever_its_own_texts_may_take() {
    let stats = slow();
    let requests: [&[&str]; 1] = [&["a"; 50]];
    let (mut jobs, _replies) = jobs_counted(&requests, &stats);
    jobs[0].deadline = Some(Instant::now() + Duration::from_secs(10));
    let (batch, _, lost) = pass(jobs, 64, &stats);
    assert_eq!((batch.len(), lost), (1, false));

    // Beside a request with no deadline it is not alone: there are other texts to wait for.
    let requests: [&[&str]; 2] = [&["a"; 50], &["unbounded"]];
    let (mut jobs, mut replies) = jobs_counted(&requests, &stats);
    jobs[0].deadline = Some(Instant::now() + Duration::from_secs(10));
    let (batch, _, lost) = pass(jobs, 64, &stats);
    assert_eq!(texts_of(&batch), ["unbounded"]);
    assert!(lost);
    assert_eq!(replies[0].try_recv().unwrap(), Err(EmbedError::TooLate));

    let cold = Arc::<WorkerStats>::default();
    let requests: [&[&str]; 3] = [&["a"; 50], &["b"; 50], &["c"]];
    let (mut jobs, _replies) = jobs_counted(&requests, &cold);
    for job in &mut jobs {
      job.deadline = Some(Instant::now() + Duration::from_secs(10));
    }
    let (batch, _, lost) = pass(jobs, 200, &cold);
    assert_eq!((batch.len(), lost), (3, false), "no pass yet, no rate");
  }

  /// The moment a request is lost, the pass stops being filled from the head of the queue. The request at the
  /// head has waited longest and has the least time left; the place that was freed goes to the newest, which
  /// has the most, and so does every place of the next pass.
  #[test]
  fn once_a_request_is_lost_the_newest_are_served_first() {
    let stats = slow();
    let requests: [&[&str]; 8] = [
      &["oldest", "oldest"],
      &["old", "old"],
      &["b"],
      &["c"],
      &["d"],
      &["e"],
      &["f"],
      &["newest"],
    ];
    let (mut jobs, mut replies) = jobs_counted(&requests, &stats);
    let now = Instant::now();
    // The two at the head are four seconds of texts between them, and have two and three seconds left.
    jobs[0].deadline = Some(now + Duration::from_secs(2));
    jobs[1].deadline = Some(now + Duration::from_secs(3));
    for job in &mut jobs[2..] {
      job.deadline = Some(now + Duration::from_mins(1));
    }

    // A pass of four texts. In order it is the two at the head, and they cannot both be answered.
    let (batch, left, lost) = pass(jobs, 4, &stats);
    assert!(lost);
    assert_eq!(replies[0].try_recv().unwrap(), Err(EmbedError::TooLate));
    // The freed places go to the newest two, and with them in the pass is four seconds again: more than the
    // other old one has. Its place goes to the newest of the rest.
    assert_eq!(replies[1].try_recv().unwrap(), Err(EmbedError::TooLate));
    assert_eq!(
      texts_of(&batch),
      ["newest", "f", "e", "d"],
      "filled from the back of the queue, not with the next in line"
    );
    assert_eq!(
      left.iter().map(|job| job.texts[0].as_str()).collect::<Vec<_>>(),
      ["b", "c"]
    );

    // The next pass is put together by a thread that is behind: newest first from the start.
    let (mut jobs, _replies) = jobs_counted(&[&["a"], &["b"], &["c"], &["d"]], &stats);
    for job in &mut jobs {
      job.deadline = Some(Instant::now() + Duration::from_mins(1));
    }
    let mut backlog: VecDeque<Job> = jobs.into();
    let (batch, lost) = gather(&mut backlog, Instant::now(), 2, true, &stats);
    assert_eq!(texts_of(&batch), ["d", "c"]);
    assert!(!lost, "nobody was lost this time, so the next pass is in order again");
    let (batch, _) = gather(&mut backlog, Instant::now(), 1, lost, &stats);
    assert_eq!(texts_of(&batch), ["a"]);
  }

  /// A request that is left waiting, and whose deadline will have passed before the pass about to run can end,
  /// is told now. Waiting for the pass to end would only tell it later.
  #[test]
  fn a_request_the_pass_about_to_run_leaves_no_time_for_is_told_before_the_pass() {
    let stats = slow();
    let requests: [&[&str]; 4] = [&["a", "b"], &["c"], &["soon"], &["later"]];
    let (mut jobs, mut replies) = jobs_counted(&requests, &stats);
    let now = Instant::now();
    for job in &mut jobs {
      job.deadline = Some(now + Duration::from_mins(1));
    }
    // The pass will be three texts, and three seconds at the least.
    jobs[2].deadline = Some(now + Duration::from_secs(2));

    let (batch, left, lost) = pass(jobs, 3, &stats);
    assert_eq!(texts_of(&batch), ["a", "b", "c"]);
    assert!(lost);
    assert_eq!(replies[2].try_recv().unwrap(), Err(EmbedError::TooLate));
    assert_eq!(
      left.iter().map(|job| job.texts[0].as_str()).collect::<Vec<_>>(),
      ["later"]
    );
    assert_eq!(waiting_in(&stats), 1);
    assert_eq!(stats.too_late.load(Ordering::Relaxed), 1);
  }

  #[test]
  fn the_least_wait_is_the_texts_at_the_fastest_rate_seen_and_unknown_before_any_pass() {
    assert_eq!(least_wait(10, 0), None, "no pass yet: nothing is known");
    assert_eq!(least_wait(0, 250), Some(Duration::ZERO));
    assert_eq!(least_wait(10, 250), Some(Duration::from_nanos(2500)));
    assert_eq!(least_wait(u64::MAX, 2), Some(Duration::from_nanos(u64::MAX)));
  }

  #[test]
  fn only_a_faster_pass_replaces_the_fastest_rate() {
    let stats = WorkerStats::default();
    let fastest = || stats.fastest_ns_per_text.load(Ordering::Relaxed);
    remember(&stats, 0, Duration::from_millis(1));
    assert_eq!(fastest(), 0, "a pass of no texts has no rate");
    remember(&stats, 4, Duration::from_micros(4));
    assert_eq!(fastest(), 1000);
    remember(&stats, 4, Duration::from_micros(8));
    assert_eq!(fastest(), 1000, "a slower pass changes nothing");
    remember(&stats, 8, Duration::from_micros(4));
    assert_eq!(fastest(), 500);
    remember(&stats, 100, Duration::from_nanos(3));
    assert_eq!(fastest(), 1, "never zero, which means no pass yet");
  }

  /// A worker whose thread is never started: what is queued stays queued.
  fn stalled(depth: u64) -> (EmbedWorker, Receiver<Job>) {
    let (queue, jobs) = channel();
    let worker = EmbedWorker {
      queue,
      depth,
      dimension: 4,
      fingerprint: Arc::from("none"),
      stats: Arc::default(),
    };
    (worker, jobs)
  }

  fn texts(count: usize) -> Vec<String> {
    vec!["a".to_owned(); count]
  }

  /// On arrival the only deadline that is refused is one that has passed. How long a request will wait depends
  /// on who is served first, and that is decided when a pass is put together, not here: a request is never
  /// refused on a guess made at the door.
  #[tokio::test]
  async fn on_arrival_only_a_deadline_that_has_passed_is_refused() {
    let (worker, _jobs) = stalled(8);
    // Polled once and not awaited, here and below: nothing serves this worker, so a request that is let in
    // when it should not be would wait for ever.
    let passed = by(Some(Instant::now()), worker.embed(texts(1)));
    tokio::pin!(passed);
    assert_eq!(futures_poll_once(passed.as_mut()).await, Some(Err(EmbedError::TooLate)));
    assert_eq!(worker.stats().too_late, 1);
    assert_eq!(waiting_in(&worker.stats), 0, "it never took a place");

    // A hundred seconds of texts already waiting, at the fastest the model has been, and five to wait.
    worker.stats.fastest_ns_per_text.store(1_000_000_000, Ordering::Relaxed);
    let ahead = worker.embed(texts(100));
    tokio::pin!(ahead);
    assert!(futures_poll_once(ahead.as_mut()).await.is_none());
    let soon = by(Some(Instant::now() + Duration::from_secs(5)), worker.embed(texts(1)));
    tokio::pin!(soon);
    assert!(futures_poll_once(soon.as_mut()).await.is_none(), "it is let in");
    assert_eq!(worker.stats().too_late, 1);
    assert_eq!(waiting_in(&worker.stats), 2);
  }

  /// A request that gives up gives up its place at once, not when the thread gets round to passing it over: a
  /// queue full of requests nobody wants is not full.
  #[tokio::test]
  async fn a_request_that_goes_away_makes_room_for_another_at_once() {
    let (worker, _jobs) = stalled(2);
    {
      let first = worker.embed(texts(1));
      let second = worker.embed(texts(1));
      tokio::pin!(first, second);
      assert!(futures_poll_once(first.as_mut()).await.is_none());
      assert!(futures_poll_once(second.as_mut()).await.is_none());
      assert_eq!(waiting_in(&worker.stats), 2);
      let third = worker.embed(texts(1));
      tokio::pin!(third);
      assert_eq!(
        futures_poll_once(third.as_mut()).await,
        Some(Err(EmbedError::Overloaded))
      );
      assert_eq!(
        waiting_in(&worker.stats),
        2,
        "a request that was refused never had a place"
      );
    }
    assert_eq!(waiting_in(&worker.stats), 0);
    let next = worker.embed(texts(1));
    tokio::pin!(next);
    assert!(futures_poll_once(next.as_mut()).await.is_none(), "there is room again");
    assert_eq!(worker.stats().overloaded, 1);
  }

  /// The whole path on a real thread: a deadline reaches the thread from the task that set it, a request that
  /// meets its deadline is served like any other, and the thread learns its own rate from the passes it runs.
  #[tokio::test]
  async fn a_deadline_set_on_the_task_reaches_the_worker_thread() {
    let options = similarity_core::EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    let worker = EmbedWorker::spawn(similarity_core::testing::tiny_embedder(&options), 4, 8).unwrap();
    assert_eq!(worker.stats.fastest_ns_per_text.load(Ordering::Relaxed), 0);
    let minute = Some(Instant::now() + Duration::from_mins(1));
    let vectors = by(minute, worker.embed(vec!["blueberry gummies".to_owned()]))
      .await
      .unwrap();
    assert_eq!(vectors.len(), 1);
    busy_beyond(&worker, 0).await;
    assert!(worker.stats.fastest_ns_per_text.load(Ordering::Relaxed) > 0);

    let passed = Some(Instant::now().checked_sub(Duration::from_millis(1)).unwrap());
    assert_eq!(
      by(passed, worker.embed(vec!["vape cartridge".to_owned()])).await,
      Err(EmbedError::TooLate)
    );
    let stats = worker.stats();
    assert_eq!((stats.jobs, stats.too_late, stats.abandoned), (1, 1, 0));
    // Outside the scope there is no deadline, whatever was set inside it a moment ago.
    worker.embed(vec!["indica flower".to_owned()]).await.unwrap();
  }

  /// A pass that failed is no evidence of how fast the model is.
  #[test]
  fn a_failed_pass_is_not_reported_as_embedded() {
    let (batch, _replies) = jobs(&[&["poison"]]);
    assert!(!serve_with(&fragile, batch));
    let (batch, _replies) = jobs(&[&["fine"], &["poison"]]);
    assert!(!serve_with(&fragile, batch), "even when the retry saved one of them");
    let (batch, _replies) = jobs(&[&["fine"]]);
    assert!(serve_with(&fragile, batch));
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

  #[tokio::test]
  async fn the_worker_counts_what_it_did_and_how_long_requests_waited() {
    let options = similarity_core::EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    let worker = EmbedWorker::spawn(similarity_core::testing::tiny_embedder(&options), 4, 8).unwrap();
    assert_eq!(worker.stats(), WorkerSnapshot::default());

    worker.embed(vec!["blueberry gummies".to_owned()]).await.unwrap();
    let first = busy_beyond(&worker, 0).await;
    assert_eq!((first.batches, first.jobs, first.texts, first.overloaded), (1, 1, 1, 0));
    assert!(first.busy_ns > 0);

    worker
      .embed(vec!["vape cartridge".to_owned(), "indica flower".to_owned()])
      .await
      .unwrap();
    // An empty request never reaches the thread.
    worker.embed(Vec::new()).await.unwrap();
    let window = busy_beyond(&worker, first.busy_ns).await.since(&first);
    assert_eq!(
      (window.batches, window.jobs, window.texts),
      (1, 1, 2),
      "only what happened since"
    );
    assert!(
      window.busy_ns > 0,
      "the second batch's time is counted, and only its own"
    );
  }

  /// The worker's counters once its time in the model has passed `busy_ns`. The thread answers a batch and then
  /// adds the batch's time, so a caller that reads the counters the moment it is answered may read them first:
  /// the counts are there by then, and the time arrives a moment later. Gives up after two seconds and returns
  /// what is there, for the assertion to fail on.
  async fn busy_beyond(worker: &EmbedWorker, busy_ns: u64) -> WorkerSnapshot {
    for _ in 0..400 {
      let stats = worker.stats();
      if stats.busy_ns > busy_ns {
        return stats;
      }
      tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    worker.stats()
  }

  /// A worker says whether its thread is still there, which is what a readiness answer needs: a handle whose
  /// thread has ended can only ever answer `WorkerGone`.
  #[tokio::test]
  async fn a_worker_knows_when_its_thread_has_ended() {
    let options = similarity_core::EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    let worker = EmbedWorker::spawn(similarity_core::testing::tiny_embedder(&options), 4, 8).unwrap();
    assert!(worker.alive());
    worker.embed(vec!["blueberry gummies".to_owned()]).await.unwrap();
    assert!(worker.alive(), "serving a request does not end it");

    // A thread that has ended, however it ended: here by returning, and a panic unwinds through the same mark.
    let (queue, jobs) = channel();
    let stats = Arc::<WorkerStats>::default();
    let ended = Arc::clone(&stats);
    std::thread::spawn(move || {
      let _stopped = Stopped(ended);
      drop(jobs);
    })
    .join()
    .unwrap();
    let orphan = EmbedWorker {
      queue,
      depth: 1,
      dimension: 4,
      fingerprint: Arc::from("none"),
      stats,
    };
    assert!(!orphan.alive());
    assert_eq!(orphan.embed(vec!["a".to_owned()]).await, Err(EmbedError::WorkerGone));
    // An embedder that runs the model where it is called has no thread to lose.
    assert!(WordBagEmbedder::new("test").alive());
  }

  /// A full queue refuses a request at once, and that is counted where a load test can read it.
  #[tokio::test]
  async fn a_request_refused_by_a_full_queue_is_counted() {
    // A queue of one that nothing drains: the thread is never started.
    let (worker, _jobs) = stalled(1);
    let waiting = worker.embed(vec!["first".to_owned()]);
    tokio::pin!(waiting);
    // Polled once, the first request is in the queue and waiting for an answer that will not come.
    assert!(futures_poll_once(waiting.as_mut()).await.is_none());
    let second = worker.embed(vec!["second".to_owned()]);
    tokio::pin!(second);
    assert_eq!(
      futures_poll_once(second.as_mut()).await,
      Some(Err(EmbedError::Overloaded))
    );
    assert_eq!(worker.stats().overloaded, 1);
    assert_eq!(worker.stats().jobs, 0, "nothing was served");
  }

  /// Polls a future once: `Some` if it finished.
  async fn futures_poll_once<F: Future + Unpin>(mut future: F) -> Option<F::Output> {
    std::future::poll_fn(|context| match std::pin::Pin::new(&mut future).poll(context) {
      std::task::Poll::Ready(output) => std::task::Poll::Ready(Some(output)),
      std::task::Poll::Pending => std::task::Poll::Ready(None),
    })
    .await
  }
}
