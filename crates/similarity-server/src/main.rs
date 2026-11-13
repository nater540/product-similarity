//! `record-similarity`: the server and the tools around it.
//!
//! Argument parsing and wiring only. Each subcommand loads what it needs and hands off to the library:
//!
//! - `serve` runs the gRPC server.
//! - `model fetch` downloads and verifies a model directory.
//! - `compare` and `calibrate` score records offline, for choosing thresholds.
//! - `runs list` shows what was recorded: every `calibrate`, and every training run, leaves a directory behind.
//!
//! Only `serve` starts an async runtime; the other commands are plain synchronous programs.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::{Args, Parser, Subcommand, ValueEnum};
use sha2::{Digest as _, Sha256};
use similarity_core::provenance::Build;
use similarity_core::record::hex_prefix;
use similarity_core::{Device, Embedder, EngineChoice, Provenance};
use similarity_log::Init;
use similarity_proto::v1::record_similarity_client::RecordSimilarityClient;
use similarity_runs::{Input, ModelRef, Spec, Status};
use similarity_server::Stop;
use similarity_server::config::Config;
use similarity_server::engine::{Engine, store_layout};
use similarity_server::fetch::{self, FetchRequest};
use similarity_server::import::{self, Failure, Summary};
use similarity_server::load;
use similarity_server::metrics::{Measured, Metrics, Polled};
use similarity_server::offline::events::{self, Run};
use similarity_server::offline::{self, Confidence, Offline, RecordFields, Selection, diff};
use similarity_server::registry::{self, Registry};
use similarity_server::runs;
use similarity_server::shadow::{self, PolicyShadow};
use similarity_server::worker::{DirectEmbedder, EmbedWorker, TextEmbedder};
use similarity_store::{MemoryStore, QdrantStore, VectorStore};

/// What this binary knows about itself, for the provenance of a model directory it fetches.
const BUILD: Build = Build::new(
  "record-similarity model fetch",
  env!("CARGO_PKG_VERSION"),
  env!("RECORD_SIMILARITY_BUILD_COMMIT"),
  env!("RECORD_SIMILARITY_BUILD_DIRTY"),
);

/// Detects when an edit to a record turns it into a different record.
#[derive(Debug, Parser)]
#[command(name = "record-similarity", version, about)]
struct Cli {
  #[command(subcommand)]
  command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
  /// Run the gRPC server.
  Serve(ServeArgs),
  /// Manage model files.
  #[command(subcommand)]
  Model(ModelCommand),
  /// Score one before/after pair of records, field by field.
  Compare(CompareArgs),
  /// Score a file of labeled pairs and report the threshold that separates edits from recycled records.
  Calibrate(CalibrateArgs),
  /// Compare two calibrations written with `calibrate --ndjson`, or one of them under a second configuration.
  CalibrateDiff(CalibrateDiffArgs),
  /// Replay whole edit histories and report how far products drift, to choose `anchor_threshold`.
  CalibrateDrift(CalibrateDriftArgs),
  /// Say what a shadowed server wrote: how often its candidate policy disagreed with the live one, in which
  /// direction, for which brands, and why. Reads the server's stdout; no model, no store.
  ShadowReport(ShadowReportArgs),
  /// Score catalog groups (a product and the menu items known to describe it) as `Match` would, to see how often
  /// the right product comes first and where `matching.threshold` belongs.
  CalibrateMatch(CalibrateMatchArgs),
  /// Judge one recorded calibration under every combination of a grid of weights, floors, and limits. No model.
  CalibrateSweep(CalibrateSweepArgs),
  /// Load a whole catalog into the store: every record of an export that is not stored yet becomes a baseline.
  Import(ImportArgs),
  /// Look at the runs that were recorded.
  #[command(subcommand)]
  Runs(RunsCommand),
  /// Work on a file of labeled pairs.
  #[command(subcommand)]
  Pairs(PairsCommand),
  /// Measure latency and throughput by driving the API with synthetic records.
  Load(LoadArgs),
}

#[derive(Debug, Args)]
struct LoadArgs {
  /// The configuration file, for the model and schema of the server started in this process.
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// Drive a server that is already running, at this URL (`http://host:port`), instead of starting one in this
  /// process on an in-memory store. Its records are written under the brand `load-test` and deleted afterwards.
  #[arg(long)]
  server: Option<String>,
  /// How many callers at once: each level is run in turn, over records of its own.
  #[arg(long, value_delimiter = ',', default_value = "1,4,16")]
  concurrency: Vec<usize>,
  /// Records per level. Each is created, edited, and saved again.
  #[arg(long, default_value_t = 200)]
  records: usize,
  /// Fixes the synthetic catalog.
  #[arg(long, default_value_t = 1)]
  seed: u64,
  /// The field each edit changes.
  #[arg(long, default_value = "name")]
  edit_field: String,
  /// Leave the records in the store afterwards.
  #[arg(long)]
  keep: bool,
  /// How long each caller of the edit and resave phases waits for an answer, in milliseconds, before it gives up
  /// on that record and takes the next. With more callers than the server can answer in that time, this is a
  /// server past what it can do.
  #[arg(long)]
  deadline_ms: Option<u64>,
}

#[derive(Debug, Subcommand)]
enum PairsCommand {
  /// Give a share of one split's brands a split of their own, for choosing a model without using the holdout.
  Carve(CarveArgs),
}

#[derive(Debug, Args)]
struct CarveArgs {
  /// The pairs file, in the full format (each pair with `split` and `brand_id`). The result is written to stdout.
  pairs: PathBuf,
  /// The split to take brands from.
  #[arg(long, default_value = "calibrate")]
  from: String,
  /// The split to give them.
  #[arg(long, default_value = "validation")]
  into: String,
  /// The share of the brands to move, between 0 and 1.
  #[arg(long)]
  fraction: f64,
  /// Decides, with each brand's name, which brands move. The same salt always moves the same brands.
  #[arg(long, default_value = "validation")]
  salt: String,
}

#[derive(Debug, Subcommand)]
enum RunsCommand {
  /// List the recorded runs, newest first.
  List(RunsListArgs),
  /// Compare two recorded runs of one kind: two training runs side by side, or two calibrations pair by pair.
  Compare(RunsCompareArgs),
  /// Write one HTML page for up to six recorded runs: what went in, the training curves, each calibration.
  Report(RunsReportArgs),
}

#[derive(Debug, Args)]
struct RunsCompareArgs {
  /// The first run's directory.
  a: PathBuf,
  /// The second run's directory.
  b: PathBuf,
  /// For two calibrations: compare the pairs both have, where they were not made from the same selection of the
  /// same file.
  #[arg(long)]
  intersect: bool,
  /// For two calibrations: how far a similarity may differ before it is listed as moved.
  #[arg(long, default_value_t = diff::TOLERANCE)]
  tolerance: f32,
}

#[derive(Debug, Args)]
struct RunsReportArgs {
  /// The run directories, at most six. A resumed training run is named by its last directory.
  #[arg(required = true, num_args = 1..=similarity_server::runs::html::MAX_RUNS)]
  runs: Vec<PathBuf>,
  /// Where to write the page.
  #[arg(long, default_value = "report.html")]
  out: PathBuf,
}

#[derive(Debug, Args)]
struct RunsListArgs {
  /// Where runs are recorded.
  #[arg(long, env = "RECORD_SIMILARITY_RUNS", default_value = "runs")]
  runs: PathBuf,
  /// Only runs of this kind (`train`, `calibrate`).
  #[arg(long)]
  kind: Option<String>,
  /// At most this many.
  #[arg(long)]
  limit: Option<usize>,
}

#[derive(Debug, Args)]
struct ServeArgs {
  /// The configuration file.
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// Listen here instead of the configured address.
  #[arg(long)]
  listen: Option<std::net::SocketAddr>,
  /// Keep vectors in process memory instead of Qdrant. Nothing survives a restart; for local experiments only.
  #[arg(long)]
  in_memory: bool,
  /// The Qdrant API key.
  #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
  qdrant_api_key: Option<String>,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
  /// Download a sentence-transformers model from the Hugging Face Hub and verify it loads.
  Fetch(FetchArgs),
  /// Put a model into service under a configuration, if its holdout calibration lets it through the gate, and
  /// print the lines that tie the configuration to it.
  Promote(PromoteArgs),
  /// Make a model directory known to the registry, without promoting it.
  Register(RegisterArgs),
  /// Take a model out of service. If it is the one in service, the one promoted before it is again.
  Retire(RetireArgs),
  /// List the models the registry knows: where each is, its promotions, and which is in service.
  List(RegistryArgs),
}

/// Where the registry is.
#[derive(Debug, Args)]
struct RegistryArgs {
  /// The registry file. It names local paths, so it is kept beside the models and out of git.
  #[arg(long, env = "RECORD_SIMILARITY_REGISTRY", default_value = "models/registry.ndjson")]
  registry: PathBuf,
}

#[derive(Debug, Args)]
struct PromoteArgs {
  /// The candidate configuration: the model it loads is the candidate, and its `[policy]`, `[fields]`, and
  /// `[quantities]` are what the holdout is judged under.
  #[arg(short, long, default_value = "config.toml")]
  config: PathBuf,
  /// The candidate's holdout calibration: the output of `calibrate --full --split holdout --ndjson` under this
  /// configuration, or the directory of the run that recorded it.
  #[arg(long)]
  calibration: PathBuf,
  /// A calibration of the model in service on the same holdout pairs, for when the one it was promoted on
  /// cannot be used (the holdout has grown, or the file is gone).
  #[arg(long)]
  baseline: Option<PathBuf>,
  /// What makes the comparison with the model in service a refusal.
  #[arg(long, value_enum, default_value = "errors-up")]
  fail_on: FailOn,
  /// Promote a candidate the comparison refuses, or a model that was retired, and record this reason. It never
  /// passes a calibration of another model or of another split.
  #[arg(long = "override", value_name = "REASON")]
  override_reason: Option<String>,
  /// Fixes the resampling behind the report's intervals.
  #[arg(long, default_value_t = 0)]
  seed: u64,
  #[command(flatten)]
  registry: RegistryArgs,
}

#[derive(Debug, Args)]
struct RegisterArgs {
  /// The model directory.
  dir: PathBuf,
  #[command(flatten)]
  registry: RegistryArgs,
}

#[derive(Debug, Args)]
struct RetireArgs {
  /// The model's content digest, as `model list` prints it.
  digest: String,
  /// Why it is being retired. Recorded.
  #[arg(long)]
  reason: String,
  #[command(flatten)]
  registry: RegistryArgs,
}

#[derive(Debug, Args)]
struct FetchArgs {
  /// The Hub repository to download.
  #[arg(long, default_value = fetch::DEFAULT_REPO)]
  repo: String,
  /// The revision to download. Use a commit hash; checksums are only pinned for the default.
  #[arg(long, default_value = fetch::DEFAULT_REVISION)]
  revision: String,
  /// Where to put the files [default: models/<the repository's name>]
  #[arg(long)]
  dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct CompareArgs {
  /// The configuration file (for the model, schema, and thresholds).
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// The record before the change: a JSON object of field name to text, and quantity name to {value, unit} or
  /// {min, max, unit}.
  before: PathBuf,
  /// The record after the change.
  after: PathBuf,
}

#[derive(Debug, Args)]
struct CalibrateArgs {
  /// The configuration file (for the model, schema, and thresholds).
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// Labeled pairs, one JSON object per line: {"label": "edit" | "recycle", "note": ..., "before": {...}, "after": {...}}.
  pairs: PathBuf,
  /// Read the full format (`pairs.full.jsonl`): extra keys are allowed, each record is reduced to what the schema
  /// defines, and `pair_id`, `brand_id`, `split`, and `variant_change` are used in the report.
  #[arg(long)]
  full: bool,
  /// With --full, score only the pairs of this split (for example `holdout`).
  #[arg(long, requires = "full")]
  split: Option<String>,
  /// With --full, leave out labeled pairs whose `label_confidence` is below this: high, medium, or low.
  #[arg(long, requires = "full", default_value = "medium")]
  min_confidence: Confidence,
  /// Write the calibration as NDJSON events on stdout, for tools, in place of the report.
  #[arg(long)]
  ndjson: bool,
  /// Fixes the resampling behind the intervals, so that a run can be reproduced.
  #[arg(long, default_value_t = 0)]
  seed: u64,
  /// Score with the model in this directory in place of the config's `model.dir`: a checkpoint, typically.
  /// Everything else (the schema, the thresholds, the width and token limit) still comes from the config.
  #[arg(long)]
  model: Option<PathBuf>,
  /// Where runs are recorded: this calibration gets a directory there with its manifest, its events (with or
  /// without --ndjson), and its report.
  #[arg(long, env = "RECORD_SIMILARITY_RUNS", default_value = "runs")]
  runs: PathBuf,
  /// Do not record this run.
  #[arg(long)]
  no_run: bool,
}

#[derive(Debug, Args)]
struct CalibrateSweepArgs {
  /// The calibration to rank the points on: the output of `calibrate --ndjson`, or a run's `events.ndjson`.
  select: PathBuf,
  /// The configuration the grid varies. Every setting the grid does not name keeps its value.
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// The grid: a TOML file with, for each setting to vary, the values to try.
  #[arg(long)]
  grid: PathBuf,
  /// A second calibration, of the same model on other pairs. The best points are judged on it at the threshold
  /// chosen on the first, which is the one count here that the sweep was not fitted to.
  #[arg(long)]
  evaluate: Option<PathBuf>,
  /// How many of the best points to report.
  #[arg(long, default_value_t = 10)]
  top: usize,
  /// Refuse a grid with more combinations than this.
  #[arg(long, default_value_t = 100_000)]
  max_points: usize,
  /// Write the sweep as NDJSON events on stdout in place of the report.
  #[arg(long)]
  ndjson: bool,
  /// Fixes the resampling behind the intervals.
  #[arg(long, default_value_t = 0)]
  seed: u64,
}

#[derive(Debug, Args)]
struct CalibrateDriftArgs {
  /// The configuration file (for the model, schema, and thresholds).
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// Edit histories, one JSON object per line: {"product": ..., "label": "edit" | "recycle", "versions": [{"fields":
  /// {...}}, ...]}, with `brand_id`, `recycled_at`, `original`, and `truncated` where they are known.
  histories: PathBuf,
  /// Leave out labeled histories whose `label_confidence` is below this: high, medium, or low.
  #[arg(long, default_value = "medium")]
  min_confidence: Confidence,
  /// Write the run as NDJSON events on stdout, for tools, in place of the report.
  #[arg(long)]
  ndjson: bool,
  /// Fixes the resampling behind the intervals, so that a run can be reproduced.
  #[arg(long, default_value_t = 0)]
  seed: u64,
}

#[derive(Debug, Args)]
struct ImportArgs {
  /// The configuration file (for the model, schema, and collection to import into).
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// The export, one JSON object per line: `{"id": ..., "brand_id": ..., "record": {...}, "anchor": {...}}`.
  /// `brand_id` and `anchor` (the record as it was first created) are optional.
  export: PathBuf,
  /// How many records are embedded and written at a time.
  #[arg(long, default_value_t = 64)]
  batch_records: usize,
  /// Stop at the first record that cannot be imported, instead of reporting it and going on.
  #[arg(long)]
  stop_on_error: bool,
  /// Only check every record against the schema. Loads no model, connects to nothing, and writes nothing.
  #[arg(long)]
  check: bool,
  /// The Qdrant API key.
  #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
  qdrant_api_key: Option<String>,
}

#[derive(Debug, Args)]
struct CalibrateDiffArgs {
  /// The earlier run: the output of `calibrate --ndjson`.
  old: PathBuf,
  /// The later run, over the same pairs.
  #[arg(required_unless_present = "config_b", conflicts_with = "config_b")]
  new: Option<PathBuf>,
  /// In place of a second run: judge the first run's pairs again under this configuration. No model is loaded;
  /// only `[policy]`, `[fields]`, and `[quantities]` are read, and the field and quantity names must be the run's.
  #[arg(long)]
  config_b: Option<PathBuf>,
  /// Compare the pairs both runs have, where the runs were not made from the same selection of the same file.
  #[arg(long)]
  intersect: bool,
  /// Fail (exit status 1) if the comparison shows this. May be given more than once; any one of them fails it.
  #[arg(long, value_enum)]
  fail_on: Vec<FailOn>,
  /// How far a similarity may differ between the two runs before it is listed as moved.
  #[arg(long, default_value_t = diff::TOLERANCE)]
  tolerance: f32,
  /// Write the comparison as NDJSON events on stdout in place of the report.
  #[arg(long)]
  ndjson: bool,
}

/// What `calibrate-diff --fail-on` can gate on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FailOn {
  /// A recycle that was rejected is now accepted.
  NewFalseAccepts,
  /// Any verdict changed.
  AnyFlip,
  /// The new run gets more pairs wrong than the old.
  ErrorsUp,
  /// A score moved by more than `--tolerance`, or a kind of change or a reason changed, with or without a verdict.
  Moved,
}

impl From<FailOn> for diff::FailOn {
  fn from(on: FailOn) -> Self {
    match on {
      FailOn::NewFalseAccepts => Self::NewFalseAccepts,
      FailOn::AnyFlip => Self::AnyFlip,
      FailOn::ErrorsUp => Self::ErrorsUp,
      FailOn::Moved => Self::Moved,
    }
  }
}

/// What a command does when the configuration names another model than the one it loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tie {
  /// Refuse to go on: the command judges or writes for production (`serve`, `import`).
  Enforced,
  /// Say so and go on: measuring a new model under a copy of the old configuration is how one is evaluated.
  Reported,
  /// Say nothing: `model promote` is where the line that names the model comes from.
  Unchecked,
}

/// Holds the loaded model to `model.expected_fingerprint` (docs/decisions.md, D31).
fn check_tie(config: &Config, fingerprint: &str, tie: Tie) -> anyhow::Result<()> {
  match (config.model.check_fingerprint(fingerprint), tie) {
    (Ok(_) | Err(_), Tie::Unchecked) | (Ok(true), _) | (Ok(false), Tie::Reported) => {}
    (Ok(false), Tie::Enforced) => tracing::warn!(
      fingerprint,
      "the configuration does not name the model its thresholds were measured with \
       (`model.expected_fingerprint`), so nothing stops them being used with another"
    ),
    (Err(mismatch), Tie::Enforced) => return Err(mismatch.into()),
    (Err(mismatch), Tie::Reported) => tracing::warn!(
      expected = mismatch.expected.as_str(),
      actual = mismatch.actual.as_str(),
      "this is not the model the configuration's thresholds were measured with; what it reports at those \
       thresholds says nothing about this model"
    ),
  }
  Ok(())
}

/// The model every command embeds with: the directory's weights, read by Burn, with the forward pass run by ONNX
/// Runtime from the directory's exported graph, which the load holds to those weights (docs/decisions.md, D34).
/// There is no setting for this. The device is the one Burn makes that check on, and nothing is served from it.
fn load_embedder(config: &Config, tie: Tie) -> anyhow::Result<Embedder> {
  let started = std::time::Instant::now();
  let embedder = Embedder::load_with(
    &config.model.dir,
    Device::flex(),
    &config.model.embedder_options(),
    EngineChoice::Onnx,
  )
  .with_context(|| {
    format!(
      "cannot load the model from {}; `record-similarity model fetch` downloads it, and `just export-onnx` writes \
       the `onnx/model.onnx` it is run from",
      config.model.dir.display()
    )
  })?;
  // Where the model came from, when its directory says. A model without the note is served like any other, and
  // so is one whose note cannot be read: lineage is for people, and never a reason to refuse to start.
  let origin = Provenance::read(&config.model.dir).unwrap_or_else(|error| {
    tracing::warn!(%error, "the model's provenance file cannot be read and is ignored");
    None
  });
  let trained = origin.as_ref().and_then(|origin| origin.trained.as_ref());
  let fetched = origin.as_ref().and_then(|origin| origin.fetched.as_ref());
  tracing::info!(
    dir = %config.model.dir.display(),
    fingerprint = embedder.fingerprint(),
    dimension = embedder.dimension(),
    engine = embedder.engine().as_str(),
    elapsed = ?started.elapsed(),
    written_by = origin.as_ref().map(|origin| origin.tool.as_str()),
    written_at = origin.as_ref().map(|origin| origin.written_at),
    commit = origin.as_ref().map(|origin| origin.commit.as_str()),
    repo = fetched.map(|fetched| fetched.repo.as_str()),
    revision = fetched.map(|fetched| fetched.revision.as_str()),
    base_digest = trained.map(|trained| trained.base.digest.as_str()),
    trained_steps = trained.map(|trained| trained.step),
    "model loaded"
  );
  if let Some(origin) = &origin
    && origin.digest != embedder.content_digest()
  {
    // Someone replaced the weights and left the old note beside them. The note is wrong, not the model.
    tracing::warn!(
      recorded = origin.digest.as_str(),
      actual = embedder.content_digest(),
      "the model's provenance file describes other files than the ones in the directory"
    );
  }
  check_tie(config, embedder.fingerprint(), tie)?;
  Ok(embedder)
}

/// Passes every request to stop on to the server, for as long as the process lives: Ctrl-C as [`Stop::Now`] (a
/// person is waiting), `SIGTERM` as [`Stop::Drain`] (an orchestrator is taking the replica out). Every one is
/// sent, because a second one during a drain means "now".
async fn stop_signals(asked: tokio::sync::mpsc::Sender<Stop>) {
  #[cfg(unix)]
  let mut terminate = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
    Ok(terminate) => Some(terminate),
    Err(error) => {
      tracing::warn!(%error, "cannot listen for SIGTERM; only Ctrl-C will stop the server");
      None
    }
  };
  loop {
    #[cfg(unix)]
    let stop = if let Some(terminate) = terminate.as_mut() {
      tokio::select! {
        _ = tokio::signal::ctrl_c() => Stop::Now,
        _ = terminate.recv() => Stop::Drain,
      }
    } else {
      let _ = tokio::signal::ctrl_c().await;
      Stop::Now
    };
    #[cfg(not(unix))]
    let stop = {
      let _ = tokio::signal::ctrl_c().await;
      Stop::Now
    };
    if asked.send(stop).await.is_err() {
      // The server has stopped and is not listening any more.
      return;
    }
  }
}

/// Starts the metrics endpoint on `address`, and returns the task that serves it until it is aborted.
async fn serve_metrics<S: VectorStore>(
  address: std::net::SocketAddr,
  metrics: &Arc<Metrics>,
  engine: &Arc<Engine<Measured<S>, EmbedWorker>>,
  worker: EmbedWorker,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
  let listener = tokio::net::TcpListener::bind(address)
    .await
    .with_context(|| format!("cannot serve metrics on {address}"))?;
  tracing::info!(address = %listener.local_addr()?, path = similarity_server::metrics::PATH, "serving metrics");
  let (rendered, engine) = (Arc::clone(metrics), Arc::clone(engine));
  let page = move || {
    let writes = engine.writes();
    rendered.render(&Polled {
      worker: Some(worker.stats()),
      waiting: worker.waiting(),
      writes_in_flight: writes.in_flight,
      writes_unanswered: writes.unanswered,
    })
  };
  // The exporter keeps every observation until it is rendered or tidied, so a server nobody is scraping tidies.
  let kept = Arc::clone(metrics);
  let upkeep = async move {
    loop {
      tokio::time::sleep(similarity_server::metrics::UPKEEP_EVERY).await;
      kept.upkeep();
    }
  };
  Ok(tokio::spawn(async move {
    tokio::select! {
      served = similarity_server::metrics::serve(listener, page, std::future::pending()) => {
        if let Err(error) = served {
          // The server goes on serving its callers: metrics are for watching it, not part of what it does.
          tracing::error!(error = format!("{error:#}"), "the metrics endpoint has stopped");
        }
      }
      () = upkeep => {}
    }
  }))
}

async fn serve_with<S: VectorStore>(
  store: S,
  worker: EmbedWorker,
  metrics: Arc<Metrics>,
  config: &Config,
  listen: std::net::SocketAddr,
) -> anyhow::Result<()> {
  let (schema, policy) = (config.schema()?, config.policy()?);
  // A candidate policy, if the configuration names one. It is told about every judged request and decides none
  // (docs/decisions.md, D30).
  let shadow = PolicyShadow::from_config(config, worker.dimension())?.map(Arc::new);
  // The store is counted from the outside, so that no store has to know it is being measured.
  let store = Measured::new(store, Arc::clone(&metrics));
  let mut engine = Engine::new(store, worker.clone(), schema.clone(), policy)
    .with_metrics(Arc::clone(&metrics))
    .with_matching(config.matching.settings()?)
    .with_import_batch(config.server.import_batch_records);
  let mut summaries = None;
  if let (Some(shadow), Some(settings)) = (&shadow, &config.shadow) {
    shadow.announce(&schema, &policy);
    engine = engine.with_observer(shadow.clone());
    let every = Duration::from_secs(settings.summary_every_s);
    summaries = Some(tokio::spawn(shadow::summarize_every(shadow.clone(), every)));
    tracing::info!(
      shadow = settings.name.as_str(),
      "judging every revision under a shadow policy as well"
    );
  }
  let listener = tokio::net::TcpListener::bind(listen)
    .await
    .with_context(|| format!("cannot listen on {listen}"))?;
  tracing::info!(address = %listener.local_addr()?, "serving gRPC");
  let (asked, stops) = tokio::sync::mpsc::channel(4);
  let signals = tokio::spawn(stop_signals(asked));
  let engine = Arc::new(engine);
  // Up before the first request and until after the last: a scrape during a drain is how a deploy is watched.
  let endpoint = match config.metrics.listen {
    Some(address) => Some(serve_metrics(address, &metrics, &engine, worker).await?),
    None => None,
  };
  let served = similarity_server::serve_until(engine, listener, stops, config.server.lifecycle(), metrics).await;
  signals.abort();
  if let Some(endpoint) = endpoint {
    endpoint.abort();
  }
  if let Some(summaries) = summaries {
    summaries.abort();
  }
  // The counts as the server stops: the timer's last line is up to an interval old.
  if let Some(shadow) = shadow {
    shadow.emit_summary();
  }
  served
}

impl ServeArgs {
  fn run(self) -> anyhow::Result<()> {
    let config = Config::load(&self.config)?;
    let listen = self.listen.unwrap_or(config.server.listen);
    let embedder = load_embedder(&config, Tie::Enforced)?;
    // What the server counts about itself (docs/decisions.md, D39): kept only if there is an address to serve
    // it on, and by nothing at all otherwise.
    let metrics = Arc::new(match config.metrics.listen {
      Some(_) => Metrics::exported(embedder.fingerprint())?,
      None => Metrics::discarded(),
    });
    let (depth, batch) = (config.model.queue_depth, config.model.batch_size);
    let worker = EmbedWorker::spawn_measured(embedder, depth, batch, Arc::clone(&metrics))
      .context("cannot start the embedding thread")?;
    let (layout, dimension) = store_layout(&config.schema()?, &worker);

    let runtime = tokio::runtime::Builder::new_multi_thread()
      .enable_all()
      .build()
      .context("cannot start tokio")?;
    runtime.block_on(async {
      if self.in_memory {
        tracing::warn!("--in-memory: vectors are not persisted and are lost when the server stops");
        serve_with(MemoryStore::new(layout, dimension), worker, metrics, &config, listen).await
      } else {
        let settings = config.qdrant.settings(self.qdrant_api_key);
        let store = QdrantStore::connect(&settings, layout, dimension)
          .await
          .with_context(|| {
            format!(
              "cannot use Qdrant collection `{}` at {}",
              settings.collection, settings.url
            )
          })?;
        tracing::info!(url = %settings.url, collection = %settings.collection, dimension, "connected to Qdrant");
        serve_with(store, worker, metrics, &config, listen).await
      }
    })
  }
}

impl FetchArgs {
  fn run(self) -> anyhow::Result<()> {
    let mut out = std::io::stdout().lock();
    let dir = match self.dir {
      Some(dir) => dir,
      None => fetch::default_dir(&self.repo)?,
    };
    let request = FetchRequest {
      repo: self.repo,
      revision: self.revision,
      dir,
    };
    writeln!(
      out,
      "fetching {} @ {} into {}",
      request.repo,
      request.revision,
      request.dir.display()
    )?;

    let mut report = Vec::new();
    let files = fetch::fetch(&request, |file| {
      let how = match (file.cached, file.verified) {
        (true, _) => "already present, checksum ok",
        (false, true) => "downloaded, checksum ok",
        (false, false) => "downloaded, checksum not pinned",
      };
      report.push(format!("  {:<28} {how}", file.name));
    })?;
    for line in report {
      writeln!(out, "{line}")?;
    }
    if files.iter().any(|file| !file.verified) {
      writeln!(
        out,
        "\nno checksums are pinned for this repository and revision; SHA-256 of what was downloaded:"
      )?;
      for file in &files {
        writeln!(out, "  {}  {}", file.sha256, file.name)?;
      }
    }

    // Loading is the real acceptance test: it runs the same strict checks the server will run at start-up.
    let options = similarity_core::EmbedderOptions::default();
    let embedder = Embedder::load(&request.dir, Device::flex(), &options)
      .with_context(|| format!("{} was downloaded but does not load", request.dir.display()))?;
    writeln!(
      out,
      "\nmodel loads: {} dimensions natively, up to {} tokens",
      embedder.dimension(),
      embedder.max_tokens()
    )?;

    // Last, once the directory is known to hold a model that runs: where it came from. The file is not one the
    // encoder reads, so writing it (again, on a re-run) changes no fingerprint.
    let provenance = fetch::provenance(&request, &files, embedder.content_digest(), &BUILD);
    provenance
      .write(&request.dir)
      .with_context(|| format!("cannot record the model's origin in {}", request.dir.display()))?;
    writeln!(
      out,
      "recorded where it came from in {}",
      request.dir.join(similarity_core::provenance::PROVENANCE_FILE).display()
    )?;
    Ok(())
  }
}

fn read_fields(path: &Path) -> anyhow::Result<RecordFields> {
  let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
  serde_json::from_str(&text).with_context(|| {
    format!(
      "{} must be a JSON object of field name to text and quantity name to an object with a unit",
      path.display()
    )
  })
}

/// This build, as an entry of the registry is stamped with it.
fn registry_stamp(tool: &'static str) -> registry::Written {
  registry::Written::now(Build { tool, ..BUILD }.stamp())
}

impl PromoteArgs {
  fn run(self) -> anyhow::Result<()> {
    let config = Config::load(&self.config)?;
    // Loaded as the server would load it, so that the fingerprint is the one the server would state.
    let embedder = load_embedder(&config, Tie::Unchecked)?;
    let provenance = Provenance::read(&config.model.dir).unwrap_or(None);
    let report = registry::promote(&registry::Promotion {
      registry: &self.registry.registry,
      model: registry::Model {
        digest: embedder.content_digest(),
        dir: &config.model.dir.display().to_string(),
        provenance: provenance.as_ref(),
      },
      fingerprint: embedder.fingerprint(),
      config: &config,
      config_path: &self.config.display().to_string(),
      calibration: &self.calibration,
      baseline: self.baseline.as_deref(),
      options: registry::GateOptions {
        written: registry_stamp("record-similarity model promote"),
        fail_on: self.fail_on.into(),
        override_reason: self.override_reason.as_deref(),
        seed: self.seed,
      },
    })?;
    write!(std::io::stdout().lock(), "{report}")?;
    Ok(())
  }
}

impl RegisterArgs {
  fn run(self) -> anyhow::Result<()> {
    // Loading is what says the directory holds a model, and which one.
    let embedder = Embedder::load(&self.dir, Device::flex(), &similarity_core::EmbedderOptions::default())
      .with_context(|| format!("{} does not load as a model", self.dir.display()))?;
    let provenance = Provenance::read(&self.dir).unwrap_or(None);
    let model = registry::Model {
      digest: embedder.content_digest(),
      dir: &self.dir.display().to_string(),
      provenance: provenance.as_ref(),
    };
    let added = registry::register(
      &self.registry.registry,
      &model,
      &registry_stamp("record-similarity model register"),
    )?;
    let mut out = std::io::stdout().lock();
    if added {
      writeln!(out, "registered {}  {}", model.digest, model.dir)?;
    } else {
      writeln!(out, "already registered: {}  {}", model.digest, model.dir)?;
    }
    Ok(())
  }
}

impl RetireArgs {
  fn run(self) -> anyhow::Result<()> {
    let path = &self.registry.registry;
    registry::retire(
      path,
      &self.digest,
      &self.reason,
      &registry_stamp("record-similarity model retire"),
    )?;
    let mut out = std::io::stdout().lock();
    writeln!(out, "retired {}: {}", self.digest, self.reason)?;
    match Registry::read(path)?.current() {
      Some(promoted) => writeln!(out, "in service now: {} ({})", promoted.fingerprint, promoted.model_dir)?,
      None => writeln!(out, "no model is in service now")?,
    }
    Ok(())
  }
}

impl RegistryArgs {
  fn run(self) -> anyhow::Result<()> {
    let registry = Registry::read(&self.registry)?;
    write!(std::io::stdout().lock(), "{}", registry::render_list(&registry))?;
    Ok(())
  }
}

fn offline(config: &Config) -> anyhow::Result<(Offline, similarity_core::Policy)> {
  let policy = config.policy()?;
  Ok((
    Offline::new(load_embedder(config, Tie::Reported)?, config.schema()?, policy),
    policy,
  ))
}

/// The SHA-256 of a file a run read, for its manifest. A file that cannot be read again here is one the command
/// has already read or is about to fail on, so there is nothing more to say about it.
fn sha256_of(path: &Path) -> Option<String> {
  let bytes = std::fs::read(path).ok()?;
  Some(hex_prefix(&Sha256::digest(&bytes), 32))
}

impl CompareArgs {
  fn run(self) -> anyhow::Result<()> {
    let (offline, policy) = offline(&Config::load(&self.config)?)?;
    let evaluation = offline.compare(&read_fields(&self.before)?, &read_fields(&self.after)?, None)?;
    write!(
      std::io::stdout().lock(),
      "{}\n{}",
      offline.model_line(),
      offline::render_comparison(&evaluation, &policy)
    )?;
    Ok(())
  }
}

impl CalibrateArgs {
  /// What this calibration is, for the registry: named after the pairs file, and the split if one was asked for.
  fn spec(&self) -> Spec<'_> {
    let stem = self
      .pairs
      .file_stem()
      .map(|stem| stem.to_string_lossy())
      .unwrap_or_default();
    Spec {
      root: &self.runs,
      kind: similarity_runs::KIND_CALIBRATE,
      slug: match &self.split {
        Some(split) => format!("{stem}-{split}"),
        None => stem.into_owned(),
      },
      build: Build {
        tool: "record-similarity calibrate",
        ..BUILD
      }
      .stamp(),
      argv: std::env::args_os()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect(),
      continues: None,
    }
  }

  /// Calibrates. `run` is the record being kept of it, if one is.
  fn run(self, run: Option<&similarity_runs::Run>) -> anyhow::Result<()> {
    let mut config = Config::load(&self.config)?;
    if let Some(model) = &self.model {
      config.model.dir.clone_from(model);
    }
    let (offline, policy) = offline(&config)?;
    let text = std::fs::read_to_string(&self.pairs).with_context(|| format!("cannot read {}", self.pairs.display()))?;
    let pairs = if self.full {
      let selection = Selection {
        split: self.split.as_deref(),
        min_confidence: Some(self.min_confidence),
      };
      offline::parse_full_pairs(&text, offline.schema(), selection)
    } else {
      offline::parse_pairs(&text)
    }
    .with_context(|| format!("in {}", self.pairs.display()))?;
    let calibration = offline::calibrate(&offline, pairs)?;

    if let Some(run) = run {
      run.note_model(ModelRef {
        role: "scored".to_owned(),
        dir: config.model.dir.display().to_string(),
        digest: offline.content_digest().to_owned(),
        fingerprint: Some(offline.fingerprint().to_owned()),
      });
      run.note_input(Input {
        role: "pairs".to_owned(),
        path: self.pairs.display().to_string(),
        sha256: calibration.input.sha256.clone(),
        examples: Some(calibration.pairs.len()),
      });
      if let Some(sha256) = sha256_of(&self.config) {
        run.note_input(Input {
          role: "config".to_owned(),
          path: self.config.display().to_string(),
          sha256,
          examples: None,
        });
      }
    }

    if self.ndjson {
      offline::events::emit(&offline, &policy, &calibration, self.seed);
    } else {
      let report = format!(
        "{}\n{}",
        offline.model_line(),
        offline::render_calibration(&calibration, &policy, self.seed)
      );
      write!(std::io::stdout().lock(), "{report}")?;
      if let Some(run) = run {
        // A calibration read by a person is data all the same: the events go to the run alone (stdout has the
        // report), so that a later run can be compared with this one whatever mode either was in.
        run.note_report(&report);
        offline::events::emit(&offline, &policy, &calibration, self.seed);
      }
    }
    Ok(())
  }
}

/// How long a run took, for a list: `41.2s`, `12m05s`, `1h02m`.
fn duration(seconds: f64) -> String {
  if seconds < 60.0 {
    return format!("{seconds:.1}s");
  }
  // A run's length in whole seconds is far below what a `u64` holds, and never negative.
  #[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a duration, rounded"
  )]
  let whole = seconds.round() as u64;
  let (hours, minutes, seconds) = (
    whole.div_euclid(3600),
    whole.rem_euclid(3600).div_euclid(60),
    whole.rem_euclid(60),
  );
  if hours > 0 {
    format!("{hours}h{minutes:02}m")
  } else {
    format!("{minutes}m{seconds:02}s")
  }
}

impl CalibrateSweepArgs {
  fn run(self) -> anyhow::Result<()> {
    let config = Config::load(&self.config)?;
    let select = read_run(&self.select)?;
    let evaluate = self.evaluate.as_deref().map(read_run).transpose()?;
    let grid_text =
      std::fs::read_to_string(&self.grid).with_context(|| format!("cannot read {}", self.grid.display()))?;
    let grid =
      offline::sweep::parse_grid(&grid_text, &config).with_context(|| format!("in {}", self.grid.display()))?;
    let sweep = offline::sweep::sweep(
      &select,
      evaluate.as_ref(),
      &config,
      &grid,
      self.top,
      self.max_points,
      self.seed,
    )?;
    if self.ndjson {
      offline::sweep::emit(&sweep, &select, evaluate.as_ref(), self.seed);
    } else {
      write!(
        std::io::stdout().lock(),
        "{}",
        offline::sweep::render_sweep(&sweep, evaluate.is_some())
      )?;
    }
    Ok(())
  }
}

#[derive(Debug, Args)]
struct ShadowReportArgs {
  /// The stdout of a server with a `[shadow]` table. Lines that are not a shadow's are passed over.
  events: PathBuf,
  /// Which candidate, when the file has lines of more than one.
  #[arg(long)]
  shadow: Option<String>,
  /// Write the revisions the two policies disagree on to this file, one JSON object per line, to be looked up
  /// and labeled.
  #[arg(long)]
  ids: Option<PathBuf>,
  /// How many brands to list.
  #[arg(long, default_value_t = 10)]
  top: usize,
  /// Write the report as NDJSON events on stdout, for tools, in place of the text.
  #[arg(long)]
  ndjson: bool,
  /// Fixes the resampling behind the intervals, so that a report can be reproduced.
  #[arg(long, default_value_t = 0)]
  seed: u64,
}

impl ShadowReportArgs {
  fn run(self) -> anyhow::Result<()> {
    let text =
      std::fs::read_to_string(&self.events).with_context(|| format!("cannot read {}", self.events.display()))?;
    let logs = offline::shadow_report::read(&text).with_context(|| format!("in {}", self.events.display()))?;
    let log = offline::shadow_report::select(logs, self.shadow.as_deref())
      .with_context(|| format!("in {}", self.events.display()))?;
    let report = offline::stats::shadow::report(&log, self.top, self.seed);
    if let Some(path) = &self.ids {
      std::fs::write(path, offline::shadow_report::ids(&log)?)
        .with_context(|| format!("cannot write {}", path.display()))?;
    }
    if self.ndjson {
      offline::shadow_report::emit(&report);
    } else {
      write!(std::io::stdout().lock(), "{}", offline::shadow_report::render(&report))?;
    }
    Ok(())
  }
}

#[derive(Debug, Args)]
struct CalibrateMatchArgs {
  /// The configuration file (for the model, the schema, and `[matching]`).
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// Catalog groups, one JSON object per line: `{"group": ..., "brand_id": ..., "split": ..., "catalog": {...},
  /// "menu_items": [{...}, ...]}`. Every group's product is in the index.
  groups: PathBuf,
  /// A catalog export, in the format `import` reads: its records join the index as products with no menu items,
  /// so that a menu item has more to be confused with.
  #[arg(long)]
  catalog: Option<PathBuf>,
  /// Score only the menu items of groups in this split. Every product stays in the index.
  #[arg(long)]
  split: Option<String>,
  /// How many candidates a served ranking returns: the `limit` a caller of `Match` would ask for.
  #[arg(long, default_value_t = 20)]
  depth: usize,
  /// Write the calibration as NDJSON events on stdout, for tools, in place of the report.
  #[arg(long)]
  ndjson: bool,
  /// Fixes the resampling behind the intervals, so that a run can be reproduced.
  #[arg(long, default_value_t = 0)]
  seed: u64,
}

impl CalibrateMatchArgs {
  fn run(self) -> anyhow::Result<()> {
    let config = Config::load(&self.config)?;
    let matching = config.matching.settings()?;
    if self.depth == 0 {
      anyhow::bail!("`--depth` must be at least 1");
    }
    let read =
      |path: &PathBuf| std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()));
    let text = read(&self.groups)?;
    let catalog = self.catalog.as_ref().map(read).transpose()?;
    let groups = offline::matching::parse_groups(&text, catalog.as_deref(), &config.schema()?, self.split.as_deref())
      .with_context(|| format!("in {}", self.groups.display()))?;
    let (offline, _) = offline(&config)?;
    let calibration = offline::matching::calibrate_match(&offline, groups, self.depth, matching.overfetch)?;
    let summary = offline::stats::matching::summarize_match(&calibration, matching.threshold, self.seed);
    if self.ndjson {
      let model = offline::drift_report::Model {
        fingerprint: offline.fingerprint(),
        dimension: offline.dimension(),
        max_tokens: offline.max_tokens(),
      };
      offline::match_report::emit_match(
        model,
        offline.schema(),
        &calibration,
        &summary,
        matching.threshold,
        self.seed,
      );
    } else {
      write!(
        std::io::stdout().lock(),
        "{}{}",
        offline.model_line(),
        offline::match_report::render_match(&calibration, &summary)
      )?;
    }
    Ok(())
  }
}

impl CalibrateDriftArgs {
  fn run(self) -> anyhow::Result<()> {
    let config = Config::load(&self.config)?;
    let (schema, policy) = (config.schema()?, config.policy()?);
    let text =
      std::fs::read_to_string(&self.histories).with_context(|| format!("cannot read {}", self.histories.display()))?;
    let histories = offline::histories::parse_histories(&text, &schema, Some(self.min_confidence))
      .with_context(|| format!("in {}", self.histories.display()))?;
    let embedder = load_embedder(&config, Tie::Reported)?;
    let (fingerprint, dimension, max_tokens) = (
      embedder.fingerprint().to_owned(),
      embedder.dimension(),
      embedder.max_tokens(),
    );
    // One caller and nothing to wait behind, as for an import: the model runs on this thread.
    let runtime = tokio::runtime::Builder::new_current_thread()
      .build()
      .context("cannot start tokio")?;
    let run = runtime.block_on(offline::drift::replay(
      DirectEmbedder(embedder),
      &schema,
      &policy,
      histories,
    ))?;
    if self.ndjson {
      let model = offline::drift_report::Model {
        fingerprint: &fingerprint,
        dimension,
        max_tokens,
      };
      offline::drift_report::emit_drift(model, &schema, &policy, &run, self.seed);
    } else {
      write!(
        std::io::stdout().lock(),
        "model     {fingerprint} ({dimension} dimensions per field, up to {max_tokens} tokens)\n{}",
        offline::drift_report::render_drift(&run, &policy, self.seed)
      )?;
    }
    Ok(())
  }
}

impl CarveArgs {
  fn run(self) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(&self.pairs).with_context(|| format!("cannot read {}", self.pairs.display()))?;
    let carved = similarity_server::carve::carve(&text, &self.from, &self.into, self.fraction, &self.salt)
      .with_context(|| format!("in {}", self.pairs.display()))?;
    write!(std::io::stdout().lock(), "{}", carved.text)?;
    tracing::info!(
      from = self.from.as_str(),
      into = self.into.as_str(),
      brands = carved.brands,
      brands_moved = carved.brands_moved,
      pairs_moved = carved.pairs_moved,
      edits_moved = carved.edits_moved,
      recycles_moved = carved.recycles_moved,
      without_brand = carved.without_brand,
      "carved"
    );
    if carved.edits_moved == 0 || carved.recycles_moved == 0 {
      tracing::warn!(
        "the new split does not have both edits and recycles, so no model can be compared on it; take a larger \
         fraction or another salt"
      );
    }
    Ok(())
  }
}

impl LoadArgs {
  fn run(self) -> anyhow::Result<()> {
    let options = load::Options {
      concurrency: self.concurrency,
      records: self.records,
      seed: self.seed,
      edit_field: self.edit_field,
      keep: self.keep,
      deadline: self.deadline_ms.map(std::time::Duration::from_millis),
    };
    let connect = |url: String| async move {
      let channel = tonic::transport::Channel::from_shared(url.clone())
        .with_context(|| format!("`{url}` is not a server address"))?
        .connect()
        .await
        .with_context(|| format!("cannot connect to {url}"))?;
      Ok::<_, anyhow::Error>(RecordSimilarityClient::new(channel))
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
      .enable_all()
      .build()
      .context("cannot start tokio")?;
    if let Some(url) = self.server {
      return runtime.block_on(async {
        load::run(connect(url).await?, &options, None).await?;
        Ok(())
      });
    }

    // A server of its own, on a store in memory: nothing here can touch a real collection.
    let config = Config::load(&self.config)?;
    let worker = EmbedWorker::spawn(
      load_embedder(&config, Tie::Reported)?,
      config.model.queue_depth,
      config.model.batch_size,
    )
    .context("cannot start the embedding thread")?;
    let schema = config.schema()?;
    let (layout, dimension) = store_layout(&schema, &worker);
    // The server a caller would get from this configuration, on a store in memory.
    let engine = Arc::new(
      Engine::new(
        MemoryStore::new(layout, dimension),
        worker.clone(),
        schema,
        config.policy()?,
      )
      .with_matching(config.matching.settings()?)
      .with_import_batch(config.server.import_batch_records),
    );
    runtime.block_on(async {
      let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("cannot bind a local port")?;
      let address = listener.local_addr().context("the local port has no address")?;
      let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
      let server = tokio::spawn(similarity_server::serve(engine, listener, async {
        let _ = stopped.await;
      }));
      let outcome = async {
        let client = connect(format!("http://{address}")).await?;
        load::run(client, &options, Some(&|| worker.stats())).await
      }
      .await;
      let _ = stop.send(());
      server.await.context("the server task stopped")??;
      outcome.map(|_| ())
    })
  }
}

impl RunsCompareArgs {
  fn run(self) -> anyhow::Result<()> {
    let (a, b) = (runs::load(&self.a)?, runs::load(&self.b)?);
    write!(
      std::io::stdout().lock(),
      "{}",
      runs::compare::compare(&a, &b, self.intersect, self.tolerance)?
    )?;
    Ok(())
  }
}

impl RunsReportArgs {
  fn run(self) -> anyhow::Result<()> {
    let loaded = self
      .runs
      .iter()
      .map(|dir| runs::load(dir))
      .collect::<anyhow::Result<Vec<_>>>()?;
    std::fs::write(&self.out, runs::html::report(&loaded))
      .with_context(|| format!("cannot write {}", self.out.display()))?;
    tracing::info!(out = %self.out.display(), runs = loaded.len(), "report written");
    Ok(())
  }
}

impl RunsListArgs {
  fn run(self) -> anyhow::Result<()> {
    let (runs, unreadable) = similarity_runs::list(&self.runs)?;
    let mut out = std::io::stdout().lock();
    let wanted = runs
      .iter()
      .filter(|run| self.kind.as_ref().is_none_or(|kind| &run.manifest.kind == kind))
      .take(self.limit.unwrap_or(usize::MAX));
    for run in wanted {
      let manifest = &run.manifest;
      // A run that did not end has no length; one that was killed stopped at a time nobody wrote down.
      let took = match (run.status, manifest.elapsed_s) {
        (Status::Finished | Status::Failed, Some(seconds)) => duration(seconds),
        _ => "-".to_owned(),
      };
      let model = manifest
        .model
        .as_ref()
        .map_or("-", |model| model.fingerprint.as_deref().unwrap_or(&model.digest));
      let input = manifest.inputs.first().map_or("-", |input| input.path.as_str());
      writeln!(
        out,
        "{:<52} {:<10} {:<9} {took:>7}  {model:<28} {input}",
        manifest.id,
        manifest.kind,
        run.status.as_str()
      )?;
    }
    for directory in &unreadable {
      tracing::warn!(dir = %directory.dir.display(), reason = directory.reason.as_str(), "not a run");
    }
    if !unreadable.is_empty() {
      writeln!(
        out,
        "{} directory(ies) under {} are not runs",
        unreadable.len(),
        self.runs.display()
      )?;
    }
    Ok(())
  }
}

/// A duration in seconds to one decimal place, as a number: precise enough to compare two runs, and readable.
fn tenths(elapsed: std::time::Duration) -> f64 {
  (elapsed.as_secs_f64() * 10.0).round() / 10.0
}

fn report_failure(failure: &Failure) {
  tracing::warn!(
    line = failure.line,
    id = failure.id.as_deref(),
    reason = failure.reason.as_str(),
    "record not imported"
  );
}

impl ImportArgs {
  fn run(self) -> anyhow::Result<()> {
    let config = Config::load(&self.config)?;
    let schema = config.schema()?;
    let open = || -> anyhow::Result<std::io::BufReader<std::fs::File>> {
      let file = std::fs::File::open(&self.export).with_context(|| format!("cannot read {}", self.export.display()))?;
      Ok(std::io::BufReader::new(file))
    };
    let started = std::time::Instant::now();

    let summary = if self.check {
      import::check(&schema, open()?, report_failure)
    } else {
      let embedder = DirectEmbedder(load_embedder(&config, Tie::Enforced)?);
      let (layout, dimension) = store_layout(&schema, &embedder);
      let options = import::Options {
        batch_records: self.batch_records,
        stop_on_error: self.stop_on_error,
      };
      let progress = |summary: &Summary| {
        tracing::info!(
          records = summary.records,
          imported = summary.imported,
          skipped = summary.skipped,
          failed = summary.failed,
          elapsed_s = tenths(started.elapsed()),
          "importing"
        );
      };
      let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("cannot start tokio")?;
      runtime.block_on(async {
        let settings = config.qdrant.settings(self.qdrant_api_key.clone());
        let store = QdrantStore::connect(&settings, layout, dimension)
          .await
          .with_context(|| {
            format!(
              "cannot use Qdrant collection `{}` at {}",
              settings.collection, settings.url
            )
          })?;
        tracing::info!(url = %settings.url, collection = %settings.collection, dimension, "connected to Qdrant");
        let engine = Engine::new(store, embedder, schema, config.policy()?);
        import::run(&engine, open()?, options, report_failure, progress).await
      })?
    };

    // What the run did is its result, on stdout whatever the log level (D15).
    tracing::info!(
      target: similarity_log::RESULT,
      checked_only = self.check,
      records = summary.records,
      imported = summary.imported,
      skipped = summary.skipped,
      failed = summary.failed,
      elapsed_s = tenths(started.elapsed()),
      "finished"
    );
    anyhow::ensure!(
      summary.failed == 0,
      "{} of {} record(s) could not be imported; each is reported above with its line",
      summary.failed,
      summary.records
    );
    Ok(())
  }
}

fn read_run(path: &Path) -> anyhow::Result<Run> {
  let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
  events::read(&text).with_context(|| format!("{} is not a calibration run", path.display()))
}

fn describe_run(run: &Run) -> String {
  format!(
    "{}, threshold {:.3}, {} pairs",
    run.info.fingerprint,
    run.info.threshold,
    run.calibration.pairs.len()
  )
}

impl CalibrateDiffArgs {
  fn run(self) -> anyhow::Result<()> {
    let old = read_run(&self.old)?;
    let (new, new_line) = match (&self.new, &self.config_b) {
      (Some(path), _) => {
        let new = read_run(path)?;
        if !self.intersect && !diff::same_pairs(&old.calibration.input, &new.calibration.input) {
          anyhow::bail!(
            "the two runs were not made from the same selection of the same pairs file; score both from one file \
             with the same --split and --min-confidence, or pass --intersect to compare the pairs they share"
          );
        }
        let line = describe_run(&new);
        (new.calibration, line)
      }
      (None, Some(path)) => {
        let config = Config::load(path)?;
        let policy = config.policy()?;
        let judged = diff::rejudge(&old, &config.schema()?, &policy)?;
        let line = format!(
          "the old run judged again under {}: threshold {:.3}",
          path.display(),
          policy.threshold
        );
        (judged, line)
      }
      (None, None) => anyhow::bail!("give a second run, or --config-b"),
    };

    let difference = diff::diff(&old.calibration, &new, self.tolerance);
    if self.ndjson {
      diff::emit(&difference);
    } else {
      write!(
        std::io::stdout().lock(),
        "{}",
        diff::render_diff(&difference, &describe_run(&old), &new_line)
      )?;
    }
    let failures: Vec<String> = self
      .fail_on
      .iter()
      .filter_map(|on| diff::failure(&difference, (*on).into()))
      .collect();
    if !failures.is_empty() {
      anyhow::bail!("the comparison fails the gate: {}", failures.join("; "));
    }
    Ok(())
  }
}

fn main() -> ExitCode {
  let cli = Cli::parse();
  // A calibration is recorded as a run. The record is started first, so that whatever would stop it from being
  // kept is found out before the model is loaded and the pairs are scored.
  let run = match &cli.command {
    Command::Calibrate(args) if !args.no_run => match similarity_runs::Run::start(args.spec()) {
      Ok(run) => Some(run),
      Err(error) => {
        similarity_log::init();
        similarity_log::failed(&format!(
          "the run cannot be recorded: {error} (give --runs another directory, or --no-run to go without)"
        ));
        return ExitCode::FAILURE;
      }
    },
    _ => None,
  };
  // With a report on stdout, a recorded calibration's events go to the record alone.
  let results_on_stdout = match &cli.command {
    Command::Calibrate(args) => args.ndjson || run.is_none(),
    _ => true,
  };
  let (record, unopened) = match run.as_ref().map(similarity_runs::Run::events).transpose() {
    Ok(record) => (record, None),
    Err(error) => (None, Some(error)),
  };
  // Logs are one JSON object per line on stderr (see `similarity-log`). The reports that `model fetch`,
  // `compare`, and `calibrate` print for a person to read are not logs, and stay as text on stdout.
  similarity_log::init_with(Init {
    record,
    results_on_stdout,
  });
  match (&run, unopened) {
    (Some(run), None) => tracing::info!(run = run.id(), dir = %run.dir().display(), "run recorded"),
    (Some(run), Some(error)) => {
      tracing::warn!(run = run.id(), %error, "the run's events cannot be kept; its manifest still is");
    }
    (None, _) => {}
  }

  let outcome = match cli.command {
    Command::Serve(args) => args.run(),
    Command::Model(ModelCommand::Fetch(args)) => args.run(),
    Command::Model(ModelCommand::Promote(args)) => args.run(),
    Command::Model(ModelCommand::Register(args)) => args.run(),
    Command::Model(ModelCommand::Retire(args)) => args.run(),
    Command::Model(ModelCommand::List(args)) => args.run(),
    Command::Compare(args) => args.run(),
    Command::Calibrate(args) => args.run(run.as_ref()),
    Command::CalibrateDiff(args) => args.run(),
    Command::CalibrateDrift(args) => args.run(),
    Command::CalibrateMatch(args) => args.run(),
    Command::ShadowReport(args) => args.run(),
    Command::CalibrateSweep(args) => args.run(),
    Command::Import(args) => args.run(),
    Command::Runs(RunsCommand::List(args)) => args.run(),
    Command::Runs(RunsCommand::Compare(args)) => args.run(),
    Command::Runs(RunsCommand::Report(args)) => args.run(),
    Command::Pairs(PairsCommand::Carve(args)) => args.run(),
    Command::Load(args) => args.run(),
  }
  .map_err(|error| format!("{error:#}"));
  if let Err(error) = &outcome {
    // A failure is a line in the same stream and the same format as everything before it.
    similarity_log::failed(error);
  }
  if let Some(run) = run
    && let Err(error) = run.finish(outcome.as_ref().copied().map_err(String::as_str))
  {
    tracing::warn!(%error, "the end of the run could not be recorded");
  }
  if outcome.is_ok() {
    ExitCode::SUCCESS
  } else {
    ExitCode::FAILURE
  }
}

#[cfg(test)]
mod tests {
  use clap::CommandFactory;

  use super::*;

  #[test]
  fn a_calibration_is_recorded_under_the_name_of_its_pairs_and_split() {
    let cli = Cli::parse_from([
      "record-similarity",
      "calibrate",
      "data/Pairs.Full.jsonl",
      "--full",
      "--split",
      "holdout",
      "--runs",
      "elsewhere",
    ]);
    let Command::Calibrate(args) = cli.command else {
      panic!("expected calibrate")
    };
    let spec = args.spec();
    assert_eq!((spec.kind, spec.slug.as_str()), ("calibrate", "Pairs.Full-holdout"));
    assert_eq!(similarity_runs::slug(&spec.slug), "pairs-full-holdout");
    assert_eq!(spec.root, Path::new("elsewhere"));
    assert_eq!(spec.build.tool, "record-similarity calibrate");
    assert!(!args.no_run, "recording is the default");
  }

  #[test]
  fn the_length_of_a_run_is_written_the_way_a_person_would_say_it() {
    for (seconds, expected) in [(0.04, "0.0s"), (41.24, "41.2s"), (725.0, "12m05s"), (3725.0, "1h02m")] {
      assert_eq!(duration(seconds), expected);
    }
  }

  #[test]
  fn the_command_line_definition_is_internally_consistent() {
    Cli::command().debug_assert();
  }

  /// `serve` and `import` refuse a model the configuration does not name; the tools that measure say so and go
  /// on, because measuring another model under a copy of the configuration is how one is evaluated.
  #[test]
  fn a_configuration_tied_to_another_model_stops_serving_and_only_warns_a_measurement() {
    let config = |model: &str| {
      Config::parse(&format!(
        "{model}[policy]\nthreshold = 0.8\n[fields.name]\nweight = 1.0"
      ))
      .unwrap()
    };
    let tied = config("[model]\nexpected_fingerprint = \"abc-d256-t256\"\n");
    let untied = config("");

    for tie in [Tie::Enforced, Tie::Reported] {
      assert!(check_tie(&tied, "abc-d256-t256", tie).is_ok());
      assert!(check_tie(&untied, "anything", tie).is_ok(), "the key is opt-in");
    }
    let refused = check_tie(&tied, "def-d256-t256", Tie::Enforced).unwrap_err();
    assert!(refused.to_string().contains("def-d256-t256"), "{refused}");
    assert!(check_tie(&tied, "def-d256-t256", Tie::Reported).is_ok());
  }

  #[test]
  fn the_registry_commands_say_what_they_need_and_default_to_the_file_beside_the_models() {
    let parse = |args: &[&str]| Cli::try_parse_from([&["record-similarity", "model"], args].concat());
    assert!(parse(&["promote"]).is_err(), "a calibration is required");
    assert!(parse(&["retire", "abc"]).is_err(), "a reason is required");
    assert!(parse(&["register"]).is_err(), "a directory is required");

    let Command::Model(ModelCommand::Promote(args)) =
      parse(&["promote", "--calibration", "holdout.ndjson"]).unwrap().command
    else {
      panic!("expected promote")
    };
    assert_eq!(args.config, PathBuf::from("config.toml"));
    assert_eq!(args.fail_on, FailOn::ErrorsUp);
    assert_eq!((args.baseline, args.override_reason, args.seed), (None, None, 0));
    assert_eq!(args.registry.registry, PathBuf::from("models/registry.ndjson"));

    let Command::Model(ModelCommand::Promote(args)) = parse(&[
      "promote",
      "--calibration",
      "runs/x",
      "--override",
      "the vendor is gone",
      "--fail-on",
      "new-false-accepts",
      "--registry",
      "/shared/registry.ndjson",
    ])
    .unwrap()
    .command
    else {
      panic!("expected promote")
    };
    assert_eq!(args.override_reason.as_deref(), Some("the vendor is gone"));
    assert_eq!(args.fail_on, FailOn::NewFalseAccepts);
    assert_eq!(args.registry.registry, PathBuf::from("/shared/registry.ndjson"));
    assert!(matches!(
      parse(&["list"]).unwrap().command,
      Command::Model(ModelCommand::List(_))
    ));
  }

  #[test]
  fn serve_defaults_to_config_toml_and_qdrant() {
    let cli = Cli::parse_from(["record-similarity", "serve"]);
    let Command::Serve(args) = cli.command else {
      panic!("expected serve")
    };
    assert_eq!(args.config, PathBuf::from("config.toml"));
    assert!(!args.in_memory && args.listen.is_none());
  }

  #[test]
  fn import_defaults_to_batches_of_64_and_goes_on_past_bad_records() {
    let cli = Cli::parse_from(["record-similarity", "import", "export.jsonl"]);
    let Command::Import(args) = cli.command else {
      panic!("expected import")
    };
    assert_eq!(args.config, PathBuf::from("config.toml"));
    assert_eq!((args.batch_records, args.stop_on_error, args.check), (64, false, false));
    assert!(
      Cli::try_parse_from(["record-similarity", "import"]).is_err(),
      "an export is required"
    );
  }

  #[test]
  fn calibrate_diff_takes_a_second_run_or_a_second_configuration_and_not_both() {
    let parse =
      |arguments: &[&str]| Cli::try_parse_from([&["record-similarity", "calibrate-diff"], arguments].concat());
    assert!(parse(&["old.ndjson", "new.ndjson"]).is_ok());
    assert!(parse(&["old.ndjson", "--config-b", "b.toml", "--fail-on", "new-false-accepts"]).is_ok());
    assert!(parse(&["old.ndjson"]).is_err(), "nothing to compare with");
    assert!(parse(&["old.ndjson", "new.ndjson", "--config-b", "b.toml"]).is_err());
    assert!(parse(&["old.ndjson", "new.ndjson", "--fail-on", "anything"]).is_err());
  }

  #[test]
  fn calibrate_reads_the_strict_format_unless_told_otherwise() {
    let parse = |arguments: &[&str]| Cli::try_parse_from([&["record-similarity", "calibrate"], arguments].concat());
    let Command::Calibrate(args) = parse(&["pairs.jsonl"]).unwrap().command else {
      panic!("expected calibrate")
    };
    assert!(!args.full && !args.ndjson && args.split.is_none());
    assert_eq!((args.min_confidence, args.seed), (Confidence::Medium, 0));
    assert!(
      parse(&["pairs.jsonl", "--split", "holdout"]).is_err(),
      "--split needs --full"
    );
    assert!(parse(&["pairs.jsonl", "--full", "--min-confidence", "certain"]).is_err());
    assert!(
      parse(&[
        "pairs.jsonl",
        "--full",
        "--split",
        "holdout",
        "--min-confidence",
        "high",
        "--ndjson"
      ])
      .is_ok()
    );
  }

  #[test]
  fn model_fetch_defaults_to_the_pinned_model() {
    let cli = Cli::parse_from(["record-similarity", "model", "fetch"]);
    let Command::Model(ModelCommand::Fetch(args)) = cli.command else {
      panic!("expected model fetch")
    };
    assert_eq!(
      (args.repo.as_str(), args.revision.as_str()),
      (fetch::DEFAULT_REPO, fetch::DEFAULT_REVISION)
    );
    assert_eq!(args.dir, None, "the directory follows the repository unless given");
  }
}
