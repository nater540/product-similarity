//! A load test: how long requests take, and how many a server answers, at a given number of concurrent callers.
//!
//! A forward-pass benchmark says what the model costs. It does not say what a caller sees, which also depends on
//! the queue in front of the one embedding thread, on how many requests share a forward pass, and on gRPC and the
//! store. This drives the real API with synthetic records and measures that.
//!
//! One level of concurrency runs three phases over the same records:
//!
//! - **create**: every record is new, so every field goes to the model. A caller loading a catalog through the API.
//! - **edit**: every record comes back with one field changed, so one text goes to the model. Edit-time traffic,
//!   which is what a replica is sized for.
//! - **resave**: the edited record again, unchanged. Nothing goes to the model; what is left is gRPC, the lock, and
//!   the store. The floor under the other two.
//!
//! The loop is closed: each of `concurrency` callers sends a request, waits for the answer, and sends the next.
//! That measures latency at a fixed number of callers. It does not measure latency at a fixed arrival rate, where
//! a slow server makes a queue grow; no figure from here should be read as one.
//!
//! No record text appears in any result, only counts and times. Every number in a result is computed in this
//! module's statistics and nowhere else.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;
use similarity_core::rng::Rng;
use similarity_log::RESULT;
use similarity_proto::v1 as pb;
use similarity_proto::v1::record_similarity_client::RecordSimilarityClient;
use tonic::Code;
use tonic::transport::Channel;

use crate::worker::WorkerSnapshot;

/// The tenant every record of a load test belongs to, so that nothing it writes can be taken for a real record.
pub const TENANT: &str = "load-test";

/// How long a caller waits before sending again a request the server refused as overloaded. A setting of this
/// tool, not a policy: a real caller's retry is its own business.
const RETRY: Duration = Duration::from_millis(5);

/// Requests sent before anything is measured, so that the first measured request does not pay for a cold start.
const WARM_UP: usize = 32;

/// The three things a caller does with a record, in the order they are run.
pub const PHASES: [&str; 3] = ["create", "edit", "resave"];
/// The phase that creates records over one `Import` stream, run once, after the levels.
pub const STREAM_PHASE: &str = "stream";

type Client = RecordSimilarityClient<Channel>;

/// What to send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
  /// The levels of concurrency, each run in turn over its own records.
  pub concurrency: Vec<usize>,
  /// Records per level.
  pub records: usize,
  /// Fixes the synthetic catalog.
  pub seed: u64,
  /// The field each edit changes.
  pub edit_field: String,
  /// Leave the records in the store afterwards.
  pub keep: bool,
}

const STRAINS: [&str; 12] = [
  "Blue Dream",
  "Northern Pine",
  "Coastal Haze",
  "Mango Kush",
  "Sour Diesel",
  "Granite Peak",
  "Velvet Sunset",
  "Lemon Thunder",
  "Cedar Mist",
  "Golden Hour",
  "Midnight Orchard",
  "Paper Kite",
];
const FORMS: [(&str, &str); 6] = [
  ("Gummies", "Edibles > Gummies"),
  ("Indoor Flower", "Flower > Indoor"),
  ("Live Resin Cartridge", "Vape Pens > Cartridges"),
  ("Pre-Rolls", "Pre-Rolls"),
  ("Tincture", "Tinctures"),
  ("Chocolate Bar", "Edibles > Chocolate"),
];
const BRANDS: [&str; 8] = [
  "Sunny Orchard",
  "High Ridge Farms",
  "Volt Labs",
  "Paper Crane",
  "Still Water",
  "Tall Pines",
  "Quiet Harbor",
  "Copper Kettle",
];
const SENTENCES: [&str; 8] = [
  "Made in small batches and tested by an independent lab.",
  "Bright, clean flavor with a smooth finish.",
  "Hand trimmed and slow cured for two weeks.",
  "Vegan, gluten free, and made with real fruit.",
  "A balanced effect for the end of the day.",
  "Packed in a resealable, child-resistant tin.",
  "Grown under full-spectrum lights with living soil.",
  "No additives, no fillers, nothing artificial.",
];
const STRENGTHS: [u32; 5] = [10, 50, 100, 250, 500];
const SUFFIXES: [&str; 4] = [" (Indica)", " - New Look", " Limited Batch", " 2 Pack"];

fn pick<'a, T>(items: &'a [T], rng: &mut Rng) -> Option<&'a T> {
  items.get(rng.below(items.len()))
}

/// A synthetic catalog of `count` records under `prefix`, the same for the same seed.
///
/// Only the fields and quantities in `fields` and `quantities` are filled: a server refuses a name its schema
/// does not define. The text fields this knows how to write are `name`, `category`, `brand`, and `description`,
/// and the quantities `weight_per_item` and `thc_per_item`.
///
/// # Errors
///
/// Returns an error if the schema requires a text field this cannot write, naming it.
pub fn catalog(
  fields: &[(String, bool)],
  quantities: &[String],
  count: usize,
  seed: u64,
  prefix: &str,
) -> anyhow::Result<Vec<pb::Record>> {
  const KNOWN: [&str; 4] = ["name", "category", "brand", "description"];
  if let Some((unknown, _)) = fields
    .iter()
    .find(|(name, required)| *required && !KNOWN.contains(&name.as_str()))
  {
    anyhow::bail!(
      "the schema requires a field `{unknown}`, which the synthetic catalog cannot write (it knows {})",
      KNOWN.join(", ")
    );
  }
  let wanted = |name: &str| fields.iter().any(|(field, _)| field == name);
  let measured = |name: &str| quantities.iter().any(|quantity| quantity == name);
  let mut rng = Rng::new(seed);
  let mut records = Vec::with_capacity(count);
  for index in 0..count {
    let (strain, brand) = (pick(&STRAINS, &mut rng), pick(&BRANDS, &mut rng));
    let (form, strength) = (pick(&FORMS, &mut rng), pick(&STRENGTHS, &mut rng));
    let (Some(strain), Some(brand), Some((form, category)), Some(strength)) = (strain, brand, form, strength) else {
      continue;
    };
    let mut record = pb::Record {
      id: Some(format!("{prefix}-{index}")),
      tenant: Some(TENANT.to_owned()),
      ..pb::Record::default()
    };
    let mut put = |name: &str, text: String| {
      if wanted(name) {
        record.fields.insert(name.to_owned(), text);
      }
    };
    // The index keeps two records from being the same record, which a catalog of twelve strains would give.
    put("name", format!("{strain} {form} {strength}mg No. {index}"));
    put("category", (*category).to_owned());
    put("brand", (*brand).to_owned());
    let sentences: Vec<&str> = (0..rng.below(4).saturating_add(2))
      .filter_map(|_| pick(&SENTENCES, &mut rng).copied())
      .collect();
    put("description", sentences.join(" "));
    if measured("weight_per_item") {
      let grams = f64::from(*strength) / 10.0;
      record.quantities.insert(
        "weight_per_item".to_owned(),
        pb::Quantity {
          min: grams,
          max: grams,
          unit: "g".to_owned(),
        },
      );
    }
    if measured("thc_per_item") {
      let milligrams = f64::from(*strength);
      record.quantities.insert(
        "thc_per_item".to_owned(),
        pb::Quantity {
          min: milligrams,
          max: milligrams,
          unit: "mg".to_owned(),
        },
      );
    }
    records.push(record);
  }
  Ok(records)
}

/// `record` with `field` changed the way an ordinary edit changes it: something appended.
pub fn edited(record: &pb::Record, field: &str, seed: u64) -> pb::Record {
  let mut rng = Rng::new(seed);
  let suffix = pick(&SUFFIXES, &mut rng).copied().unwrap_or(" (edited)");
  let mut edited = record.clone();
  if let Some(text) = edited.fields.get_mut(field) {
    text.push_str(suffix);
  }
  edited
}

/// What one caller saw.
#[derive(Debug, Default)]
struct Tally {
  latencies: Vec<Duration>,
  overloaded: u64,
  failed: u64,
  created: u64,
  accepted: u64,
  rejected: u64,
}

impl Tally {
  fn merge(&mut self, other: Self) {
    self.latencies.extend(other.latencies);
    self.overloaded = self.overloaded.saturating_add(other.overloaded);
    self.failed = self.failed.saturating_add(other.failed);
    self.created = self.created.saturating_add(other.created);
    self.accepted = self.accepted.saturating_add(other.accepted);
    self.rejected = self.rejected.saturating_add(other.rejected);
  }
}

/// One caller: takes the next record, sends it, waits, and takes the next, until there are none.
async fn caller(
  mut client: Client,
  records: Arc<Vec<pb::Record>>,
  next: Arc<AtomicUsize>,
  mode: pb::WriteMode,
) -> Tally {
  let mut tally = Tally::default();
  while let Some(record) = records.get(next.fetch_add(1, Ordering::Relaxed)) {
    // Timed from the first attempt: a refusal and a retry are part of what a caller waits for.
    let started = Instant::now();
    loop {
      let request = pb::EvaluateRequest {
        record: Some(record.clone()),
        mode: mode.into(),
        ..pb::EvaluateRequest::default()
      };
      match client.evaluate(request).await {
        Ok(response) => {
          tally.latencies.push(started.elapsed());
          match response.into_inner().verdict() {
            pb::Verdict::Created => tally.created = tally.created.saturating_add(1),
            pb::Verdict::Accepted => tally.accepted = tally.accepted.saturating_add(1),
            pb::Verdict::Rejected => tally.rejected = tally.rejected.saturating_add(1),
            pb::Verdict::Embedded | pb::Verdict::Unspecified => {}
          }
          break;
        }
        Err(status) if status.code() == Code::ResourceExhausted => {
          tally.overloaded = tally.overloaded.saturating_add(1);
          tokio::time::sleep(RETRY).await;
        }
        Err(_) => {
          tally.failed = tally.failed.saturating_add(1);
          break;
        }
      }
    }
  }
  tally
}

/// Sends every one of `records` once, from `concurrency` callers.
async fn send(
  client: &Client,
  records: Vec<pb::Record>,
  concurrency: usize,
  mode: pb::WriteMode,
) -> anyhow::Result<(Tally, Duration)> {
  let (records, next) = (Arc::new(records), Arc::new(AtomicUsize::new(0)));
  let started = Instant::now();
  let callers: Vec<_> = (0..concurrency.max(1))
    .map(|_| tokio::spawn(caller(client.clone(), Arc::clone(&records), Arc::clone(&next), mode)))
    .collect();
  let mut tally = Tally::default();
  for caller in callers {
    tally.merge(caller.await.context("a caller stopped")?);
  }
  Ok((tally, started.elapsed()))
}

fn milliseconds(duration: Duration) -> f64 {
  (duration.as_secs_f64() * 100_000.0).round() / 100.0
}

#[expect(
  clippy::cast_precision_loss,
  reason = "counts and nanosecond totals of one run are far below 2^53"
)]
fn number(count: u64) -> f64 {
  count as f64
}

fn count(value: usize) -> u64 {
  u64::try_from(value).unwrap_or(u64::MAX)
}

/// The `percent`th percentile of `sorted`, by nearest rank: the smallest value with at least that share of the
/// values at or below it. Exact, and needs no interpolation to explain. `None` for no values.
pub fn percentile(sorted: &[Duration], percent: u32) -> Option<Duration> {
  let rank = count(sorted.len())
    .saturating_mul(u64::from(percent))
    .div_ceil(100)
    .max(1);
  sorted.get(usize::try_from(rank.saturating_sub(1)).ok()?).copied()
}

/// What the embedding thread did while a phase ran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkerShare {
  /// Calls into the model.
  pub batches: u64,
  /// Texts embedded.
  pub texts: u64,
  /// Texts per call: how much sharing of forward passes the level of concurrency bought.
  pub texts_per_batch: Option<f64>,
  /// The share of the phase's wall time the model was busy.
  pub busy: f64,
  /// How long a request waited in the queue before the model took it, on average, in milliseconds.
  pub queue_wait_ms: Option<f64>,
}

/// One phase at one level of concurrency.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseStats {
  /// `create`, `edit`, or `resave`.
  pub phase: &'static str,
  /// How many callers.
  pub concurrency: usize,
  /// Records sent.
  pub requests: usize,
  /// Requests answered.
  pub ok: u64,
  /// Attempts refused as overloaded and sent again. Can exceed `requests`.
  pub overloaded: u64,
  /// Requests that failed for another reason, and were not sent again.
  pub failed: u64,
  /// Verdicts of the answered requests.
  pub verdicts: (u64, u64, u64),
  /// The phase's wall time, in seconds.
  pub seconds: f64,
  /// Answered requests per second of wall time.
  pub per_second: f64,
  /// Latency in milliseconds: the median, the 90th and 99th percentiles, the slowest, and the mean. `None` when
  /// nothing was answered.
  pub latency_ms: Option<[f64; 5]>,
  /// What the embedding thread did meanwhile, when the server is in this process.
  pub worker: Option<WorkerShare>,
}

impl PhaseStats {
  fn of(
    phase: &'static str,
    concurrency: usize,
    requests: usize,
    mut tally: Tally,
    wall: Duration,
    worker: Option<WorkerSnapshot>,
  ) -> Self {
    tally.latencies.sort_unstable();
    let ok = count(tally.latencies.len());
    let seconds = wall.as_secs_f64();
    let latency_ms = (|| {
      let total: Duration = tally.latencies.iter().sum();
      Some([
        milliseconds(percentile(&tally.latencies, 50)?),
        milliseconds(percentile(&tally.latencies, 90)?),
        milliseconds(percentile(&tally.latencies, 99)?),
        milliseconds(*tally.latencies.last()?),
        (total.as_secs_f64() * 100_000.0 / number(ok)).round() / 100.0,
      ])
    })();
    let worker = worker.map(|window| WorkerShare {
      batches: window.batches,
      texts: window.texts,
      texts_per_batch: (window.batches > 0)
        .then(|| (number(window.texts) / number(window.batches) * 100.0).round() / 100.0),
      busy: if seconds > 0.0 {
        (number(window.busy_ns) / 1e9 / seconds * 1000.0).round() / 1000.0
      } else {
        0.0
      },
      queue_wait_ms: (window.jobs > 0)
        .then(|| (number(window.queue_wait_ns) / 1e6 / number(window.jobs) * 100.0).round() / 100.0),
    });
    Self {
      phase,
      concurrency,
      requests,
      ok,
      overloaded: tally.overloaded,
      failed: tally.failed,
      verdicts: (tally.created, tally.accepted, tally.rejected),
      seconds: (seconds * 1000.0).round() / 1000.0,
      per_second: if seconds > 0.0 {
        (number(ok) / seconds * 10.0).round() / 10.0
      } else {
        0.0
      },
      latency_ms,
      worker,
    }
  }

  /// A phase in which `requests` records went over one `Import` stream. There is one caller and no latency of a
  /// single request to speak of: a record's answer comes with its batch's.
  fn streamed(requests: usize, outcomes: (u64, u64, u64), wall: Duration, worker: Option<WorkerSnapshot>) -> Self {
    let (created, existing, failed) = outcomes;
    let tally = Tally {
      failed,
      created,
      ..Tally::default()
    };
    let mut stats = Self::of(STREAM_PHASE, 1, requests, tally, wall, worker);
    stats.ok = created.saturating_add(existing);
    let seconds = wall.as_secs_f64();
    if seconds > 0.0 {
      stats.per_second = (number(stats.ok) / seconds * 10.0).round() / 10.0;
    }
    stats
  }

  /// Writes the phase as one `load_phase` result line.
  fn emit(&self, run_id: &str) {
    let latency = |index: usize| self.latency_ms.and_then(|values| values.get(index).copied());
    tracing::info!(
      target: RESULT,
      run_id,
      concurrency = self.concurrency,
      phase = self.phase,
      requests = self.requests,
      ok = self.ok,
      overloaded = self.overloaded,
      failed = self.failed,
      created = self.verdicts.0,
      accepted = self.verdicts.1,
      rejected = self.verdicts.2,
      seconds = self.seconds,
      per_second = self.per_second,
      p50_ms = latency(0),
      p90_ms = latency(1),
      p99_ms = latency(2),
      max_ms = latency(3),
      mean_ms = latency(4),
      batches = self.worker.map(|worker| worker.batches),
      texts = self.worker.map(|worker| worker.texts),
      texts_per_batch = self.worker.and_then(|worker| worker.texts_per_batch),
      model_busy = self.worker.map(|worker| worker.busy),
      queue_wait_ms = self.worker.and_then(|worker| worker.queue_wait_ms),
      "load_phase"
    );
  }
}

/// Runs the load test against `client`, and writes one `load_run` line and one `load_phase` line per level and
/// phase.
///
/// `worker` reads the embedding thread's counters when the server is in this process; with `None` the results
/// hold only what a client can see.
///
/// # Errors
///
/// Returns an error if the server cannot be described, if its schema requires a field the synthetic catalog
/// cannot write, or if a caller task stops.
pub async fn run(
  mut client: Client,
  options: &Options,
  worker: Option<&(dyn Fn() -> WorkerSnapshot + Sync)>,
) -> anyhow::Result<Vec<PhaseStats>> {
  let description = client
    .describe(pb::DescribeRequest {})
    .await
    .context("the server cannot be described")?
    .into_inner();
  let fields: Vec<(String, bool)> = description
    .fields
    .iter()
    .map(|field| (field.name.clone(), field.required))
    .collect();
  let quantities: Vec<String> = description
    .quantities
    .iter()
    .map(|quantity| quantity.name.clone())
    .collect();
  anyhow::ensure!(
    fields.iter().any(|(name, _)| *name == options.edit_field),
    "the field to edit, `{}`, is not in the server's schema",
    options.edit_field
  );
  let started_at = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .map_or(0, |elapsed| elapsed.as_secs());
  let run_id = format!("{}-{started_at}", options.seed);
  tracing::info!(
    target: RESULT,
    run_id = run_id.as_str(),
    target_server = if worker.is_some() { "in_process" } else { "external" },
    model_fingerprint = description.model_fingerprint.as_str(),
    dimension = description.dimension,
    records = options.records,
    seed = options.seed,
    edit_field = options.edit_field.as_str(),
    threads = std::thread::available_parallelism().ok().map(std::num::NonZero::get),
    "load_run"
  );

  // Nothing of the warm-up is stored: it is sent as a dry run.
  let warm_up = catalog(
    &fields,
    &quantities,
    WARM_UP,
    options.seed,
    &format!("load-{run_id}-warm"),
  )?;
  send(&client, warm_up, 4, pb::WriteMode::DryRun).await?;

  let mut all = Vec::new();
  for concurrency in &options.concurrency {
    let prefix = format!("load-{run_id}-c{concurrency}");
    let created = catalog(&fields, &quantities, options.records, options.seed, &prefix)?;
    let changed: Vec<pb::Record> = created
      .iter()
      .enumerate()
      .map(|(index, record)| edited(record, &options.edit_field, options.seed.wrapping_add(count(index))))
      .collect();
    let phases = [created.clone(), changed.clone(), changed];
    for (phase, records) in PHASES.into_iter().zip(phases) {
      let before = worker.map(|read| read());
      let requests = records.len();
      let (tally, wall) = send(&client, records, *concurrency, pb::WriteMode::Enforce).await?;
      let window = worker.zip(before).map(|(read, before)| read().since(&before));
      let stats = PhaseStats::of(phase, *concurrency, requests, tally, wall, window);
      stats.emit(&run_id);
      tracing::info!(
        concurrency,
        phase,
        per_second = stats.per_second,
        p50_ms = stats.latency_ms.and_then(|values| values.first().copied()),
        "phase done"
      );
      all.push(stats);
    }
    if !options.keep {
      delete(&mut client, created).await;
    }
  }
  if let Some(stats) = stream(&mut client, options, &run_id, (&fields, &quantities), worker).await? {
    stats.emit(&run_id);
    tracing::info!(phase = STREAM_PHASE, per_second = stats.per_second, "phase done");
    all.push(stats);
  }
  Ok(all)
}

/// Tidying up. A record that cannot be deleted is left, and the run's numbers stand.
async fn delete(client: &mut Client, records: Vec<pb::Record>) {
  for record in records {
    let request = pb::DeleteRequest {
      id: record.id.unwrap_or_default(),
      tenant: record.tenant,
    };
    let _ = client.delete(request).await;
  }
}

/// Creates `options.records` new records over one `Import` stream and says how fast they went. `None`, with a
/// warning, against a server too old to have the call.
async fn stream(
  client: &mut Client,
  options: &Options,
  run_id: &str,
  schema: (&[(String, bool)], &[String]),
  worker: Option<&(dyn Fn() -> WorkerSnapshot + Sync)>,
) -> anyhow::Result<Option<PhaseStats>> {
  let records = catalog(
    schema.0,
    schema.1,
    options.records,
    options.seed,
    &format!("load-{run_id}-stream"),
  )?;
  let requests: Vec<pb::ImportRequest> = records
    .iter()
    .cloned()
    .map(|record| pb::ImportRequest {
      record: Some(record),
      anchor: None,
    })
    .collect();
  let before = worker.map(|read| read());
  let started = Instant::now();
  let mut answers = match client.import(tokio_stream::iter(requests)).await {
    Ok(answers) => answers.into_inner(),
    Err(status) if status.code() == tonic::Code::Unimplemented => {
      tracing::warn!("the server has no `Import` call; the stream phase is left out");
      return Ok(None);
    }
    Err(status) => return Err(status).context("the import stream was refused"),
  };
  let (mut created, mut existing, mut failed) = (0_u64, 0_u64, 0_u64);
  while let Some(answer) = answers.message().await.context("the import stream broke off")? {
    let slot = match answer.outcome() {
      pb::ImportOutcome::Created => &mut created,
      pb::ImportOutcome::Exists => &mut existing,
      _ => &mut failed,
    };
    *slot = slot.saturating_add(1);
  }
  let wall = started.elapsed();
  let window = worker.zip(before).map(|(read, before)| read().since(&before));
  if !options.keep {
    delete(client, records).await;
  }
  Ok(Some(PhaseStats::streamed(
    options.records,
    (created, existing, failed),
    wall,
    window,
  )))
}

#[cfg(test)]
mod tests {
  use similarity_core::{FieldSpec, Policy, QuantitySpec, Schema};
  use similarity_store::MemoryStore;
  use tokio::net::TcpListener;
  use tokio::sync::oneshot;

  use super::*;
  use crate::engine::{Engine, store_layout};
  use crate::worker::testing::WordBagEmbedder;

  fn schema_names() -> (Vec<(String, bool)>, Vec<String>) {
    let fields = [("name", true), ("category", false), ("description", false)];
    (
      fields
        .iter()
        .map(|(name, required)| ((*name).to_owned(), *required))
        .collect(),
      vec!["weight_per_item".to_owned()],
    )
  }

  #[test]
  fn the_catalog_is_the_same_for_the_same_seed_and_holds_only_what_the_schema_has() {
    let (fields, quantities) = schema_names();
    let catalog_of = |seed: u64| catalog(&fields, &quantities, 50, seed, "load-x").unwrap();
    let records = catalog_of(1);
    assert_eq!(records, catalog_of(1));
    assert_ne!(records, catalog_of(2));
    assert_eq!(records.len(), 50);

    let mut ids: Vec<&str> = records.iter().map(|record| record.id.as_deref().unwrap()).collect();
    ids.dedup();
    assert_eq!(ids.len(), 50, "every record has its own id");
    for record in &records {
      assert_eq!(record.tenant.as_deref(), Some(TENANT));
      let mut names: Vec<&str> = record.fields.keys().map(String::as_str).collect();
      names.sort_unstable();
      assert_eq!(
        names,
        ["category", "description", "name"],
        "no brand: the schema has none"
      );
      assert_eq!(record.quantities.keys().collect::<Vec<_>>(), ["weight_per_item"]);
    }

    let needs_more = [("sku".to_owned(), true)];
    let error = catalog(&needs_more, &[], 1, 1, "x").unwrap_err().to_string();
    assert!(error.contains("`sku`"), "{error}");
  }

  #[test]
  fn an_edit_changes_one_field_and_nothing_else() {
    let (fields, quantities) = schema_names();
    let record = catalog(&fields, &quantities, 1, 1, "x").unwrap().remove(0);
    let changed = edited(&record, "name", 7);
    assert_ne!(changed.fields["name"], record.fields["name"]);
    assert!(changed.fields["name"].starts_with(&record.fields["name"]));
    assert_eq!(changed.fields["description"], record.fields["description"]);
    assert_eq!((&changed.id, &changed.quantities), (&record.id, &record.quantities));
    assert_eq!(edited(&record, "name", 7), changed, "the same edit for the same seed");
  }

  #[test]
  fn a_percentile_is_the_nearest_rank() {
    let values: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
    let at = |percent: u32| percentile(&values, percent).unwrap().as_millis();
    assert_eq!((at(50), at(90), at(99), at(100)), (50, 90, 99, 100));
    let few = [Duration::from_millis(5), Duration::from_millis(9)];
    assert_eq!(percentile(&few, 50).unwrap().as_millis(), 5);
    assert_eq!(
      percentile(&few, 99).unwrap().as_millis(),
      9,
      "rounded up to a value that exists"
    );
    assert_eq!(percentile(&[], 50), None);
  }

  #[test]
  fn a_phase_reports_rates_over_wall_time_and_no_latency_when_nothing_was_answered() {
    let tally = Tally {
      latencies: vec![
        Duration::from_millis(30),
        Duration::from_millis(10),
        Duration::from_millis(20),
      ],
      overloaded: 2,
      failed: 1,
      created: 3,
      ..Tally::default()
    };
    let window = WorkerSnapshot {
      batches: 2,
      jobs: 3,
      texts: 9,
      overloaded: 2,
      busy_ns: 500_000_000,
      queue_wait_ns: 6_000_000,
    };
    let stats = PhaseStats::of("create", 4, 4, tally, Duration::from_secs(2), Some(window));
    assert_eq!(
      (stats.ok, stats.overloaded, stats.failed, stats.verdicts),
      (3, 2, 1, (3, 0, 0))
    );
    assert!(
      (stats.per_second - 1.5).abs() < 1e-9,
      "three answers in two seconds, not three over their latencies"
    );
    assert_eq!(stats.latency_ms, Some([20.0, 30.0, 30.0, 30.0, 20.0]));
    let worker = stats.worker.unwrap();
    assert_eq!(
      (worker.batches, worker.texts, worker.texts_per_batch),
      (2, 9, Some(4.5))
    );
    assert!((worker.busy - 0.25).abs() < 1e-9 && worker.queue_wait_ms == Some(2.0));

    let nothing = PhaseStats::of("edit", 1, 5, Tally::default(), Duration::from_secs(1), None);
    assert_eq!((nothing.ok, nothing.latency_ms, nothing.worker), (0, None, None));
  }

  /// The whole tool against a server in this process: every record is created, edited, and saved again, and
  /// nothing is left behind.
  #[tokio::test]
  async fn a_run_creates_edits_and_resaves_every_record_and_cleans_up() {
    let field = |name: &str, required: bool| FieldSpec {
      name: name.to_owned(),
      weight: 1.0,
      required,
      reject_below: None,
    };
    let schema = Schema::new(vec![field("name", true), field("description", false)])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight_per_item".to_owned(),
        reject_above_factor: None,
        reject_incomparable: false,
      }])
      .unwrap();
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema, &embedder);
    let engine = Arc::new(Engine::new(
      MemoryStore::new(layout, dimension),
      embedder,
      schema,
      Policy::new(0.5, None).unwrap(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(crate::serve(Arc::clone(&engine), listener, async {
      let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{address}"))
      .unwrap()
      .connect()
      .await
      .unwrap();
    let client = RecordSimilarityClient::new(channel);

    let options = Options {
      concurrency: vec![1, 4],
      records: 12,
      seed: 3,
      edit_field: "name".to_owned(),
      keep: false,
    };
    let captured_stats = run(client.clone(), &options, None).await.unwrap();
    let shape: Vec<(&str, usize, u64)> = captured_stats
      .iter()
      .map(|stats| (stats.phase, stats.concurrency, stats.ok))
      .collect();
    assert_eq!(
      shape,
      [
        ("create", 1, 12),
        ("edit", 1, 12),
        ("resave", 1, 12),
        ("create", 4, 12),
        ("edit", 4, 12),
        ("resave", 4, 12),
        ("stream", 1, 12)
      ]
    );
    for stats in &captured_stats {
      assert_eq!((stats.failed, stats.overloaded), (0, 0));
      assert!(stats.worker.is_none());
      // One stream has no latency of a single request: a record's answer comes with its batch's.
      assert_eq!(stats.latency_ms.is_some(), stats.phase != STREAM_PHASE, "{stats:?}");
      assert!(stats.per_second > 0.0, "{stats:?}");
      match stats.phase {
        "create" | STREAM_PHASE => assert_eq!(stats.verdicts, (12, 0, 0)),
        // Whether an edit passes the threshold is the server's business; that it was judged is this tool's.
        _ => assert_eq!(stats.verdicts.1 + stats.verdicts.2, 12, "{stats:?}"),
      }
    }

    // Nothing is left in the store, and a field the schema lacks is refused before anything is sent.
    let mut probe = client.clone();
    let like_any_of_them = pb::Record {
      fields: [("name".to_owned(), "Blue Dream Gummies 100mg".to_owned())].into(),
      ..pb::Record::default()
    };
    let left = probe
      .find_similar(pb::FindSimilarRequest {
        query: Some(pb::find_similar_request::Query::Record(like_any_of_them)),
        limit: 10,
        tenant: Some(TENANT.to_owned()),
        ..pb::FindSimilarRequest::default()
      })
      .await
      .unwrap()
      .into_inner()
      .neighbors;
    assert!(left.is_empty(), "{left:?}");
    let wrong = Options {
      edit_field: "brand".to_owned(),
      ..options
    };
    let error = run(client, &wrong, None).await.unwrap_err().to_string();
    assert!(error.contains("`brand`"), "{error}");

    let _ = stop.send(());
    server.await.unwrap().unwrap();
  }
}
