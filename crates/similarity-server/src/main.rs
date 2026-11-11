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

use anyhow::Context;
use clap::{Args, Parser, Subcommand, ValueEnum};
use sha2::{Digest as _, Sha256};
use similarity_core::provenance::Build;
use similarity_core::record::hex_prefix;
use similarity_core::{Device, Embedder, Provenance};
use similarity_log::Init;
use similarity_runs::{Input, ModelRef, Spec, Status};
use similarity_server::config::{Config, DeviceChoice};
use similarity_server::engine::{Engine, store_layout};
use similarity_server::fetch::{self, FetchRequest};
use similarity_server::import::{self, Failure, Summary};
use similarity_server::offline::events::{self, Run};
use similarity_server::offline::{self, Confidence, Offline, RecordFields, Selection, diff};
use similarity_server::worker::{DirectEmbedder, EmbedWorker};
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
  /// Load a whole catalog into the store: every record of an export that is not stored yet becomes a baseline.
  Import(ImportArgs),
  /// Look at the runs that were recorded.
  #[command(subcommand)]
  Runs(RunsCommand),
}

#[derive(Debug, Subcommand)]
enum RunsCommand {
  /// List the recorded runs, newest first.
  List(RunsListArgs),
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
  /// defines, and `pair_id`, `owner`, `split`, and `variant_change` are used in the report.
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
  /// Where runs are recorded: this calibration gets a directory there with its manifest, its events (with or
  /// without --ndjson), and its report.
  #[arg(long, env = "RECORD_SIMILARITY_RUNS", default_value = "runs")]
  runs: PathBuf,
  /// Do not record this run.
  #[arg(long)]
  no_run: bool,
}

#[derive(Debug, Args)]
struct ImportArgs {
  /// The configuration file (for the model, schema, and collection to import into).
  #[arg(long, short, default_value = "config.toml")]
  config: PathBuf,
  /// The export, one JSON object per line: {"id": ..., "tenant": ..., "record": {...}, "anchor": {...}}. `tenant`
  /// and `anchor` (the record as it was first created) are optional.
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

fn device(choice: DeviceChoice) -> anyhow::Result<Device> {
  match choice {
    DeviceChoice::Cpu => Ok(Device::flex()),
    DeviceChoice::Metal => metal_device(),
  }
}

#[cfg(feature = "metal")]
#[expect(
  clippy::unnecessary_wraps,
  reason = "must match the signature of the build without Metal, which fails"
)]
fn metal_device() -> anyhow::Result<Device> {
  Ok(Device::metal(burn::tensor::DeviceKind::DefaultDevice))
}

#[cfg(not(feature = "metal"))]
fn metal_device() -> anyhow::Result<Device> {
  anyhow::bail!("`device = \"metal\"` needs a binary built with `--features metal`; this one runs on the CPU only")
}

fn load_embedder(config: &Config) -> anyhow::Result<Embedder> {
  let started = std::time::Instant::now();
  let embedder = Embedder::load(
    &config.model.dir,
    device(config.model.device)?,
    &config.model.embedder_options(),
  )
  .with_context(|| {
    format!(
      "cannot load the model from {}; run `record-similarity model fetch` first",
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
    device = ?config.model.device,
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
  Ok(embedder)
}

async fn shutdown_signal() {
  let interrupt = tokio::signal::ctrl_c();
  #[cfg(unix)]
  {
    let terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate());
    match terminate {
      Ok(mut terminate) => {
        tokio::select! {
          _ = interrupt => {}
          _ = terminate.recv() => {}
        }
      }
      Err(error) => {
        tracing::warn!(%error, "cannot listen for SIGTERM; only Ctrl-C will stop the server");
        let _ = interrupt.await;
      }
    }
  }
  #[cfg(not(unix))]
  {
    let _ = interrupt.await;
  }
  tracing::info!("shutting down: finishing in-flight requests");
}

async fn serve_with<S: VectorStore>(
  store: S,
  worker: EmbedWorker,
  config: &Config,
  listen: std::net::SocketAddr,
) -> anyhow::Result<()> {
  let engine = Arc::new(Engine::new(store, worker, config.schema()?, config.policy()?));
  let listener = tokio::net::TcpListener::bind(listen)
    .await
    .with_context(|| format!("cannot listen on {listen}"))?;
  tracing::info!(address = %listener.local_addr()?, "serving gRPC");
  similarity_server::serve(engine, listener, shutdown_signal()).await
}

impl ServeArgs {
  fn run(self) -> anyhow::Result<()> {
    let config = Config::load(&self.config)?;
    let listen = self.listen.unwrap_or(config.server.listen);
    let embedder = load_embedder(&config)?;
    let worker = EmbedWorker::spawn(embedder, config.model.queue_depth, config.model.batch_size)
      .context("cannot start the embedding thread")?;
    let (layout, dimension) = store_layout(&config.schema()?, &worker);

    let runtime = tokio::runtime::Builder::new_multi_thread()
      .enable_all()
      .build()
      .context("cannot start tokio")?;
    runtime.block_on(async {
      if self.in_memory {
        tracing::warn!("--in-memory: vectors are not persisted and are lost when the server stops");
        serve_with(MemoryStore::new(layout, dimension), worker, &config, listen).await
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
        serve_with(store, worker, &config, listen).await
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

fn offline(config: &Config) -> anyhow::Result<(Offline, similarity_core::Policy)> {
  let policy = config.policy()?;
  Ok((Offline::new(load_embedder(config)?, config.schema()?, policy), policy))
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
    let config = Config::load(&self.config)?;
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

/// How long a run took, for a listing: `41.2s`, `12m05s`, `1h02m`.
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
      let embedder = DirectEmbedder(load_embedder(&config)?);
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
    Command::Compare(args) => args.run(),
    Command::Calibrate(args) => args.run(run.as_ref()),
    Command::CalibrateDiff(args) => args.run(),
    Command::Import(args) => args.run(),
    Command::Runs(RunsCommand::List(args)) => args.run(),
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

  #[test]
  fn metal_is_refused_with_a_useful_message_when_not_compiled_in() {
    if cfg!(feature = "metal") {
      assert!(device(DeviceChoice::Metal).is_ok());
    } else {
      let error = device(DeviceChoice::Metal).unwrap_err().to_string();
      assert!(error.contains("--features metal"), "{error}");
    }
    assert!(device(DeviceChoice::Cpu).is_ok());
  }
}
