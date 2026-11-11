//! `record-similarity`: the server and the tools around it.
//!
//! Argument parsing and wiring only. Each subcommand loads what it needs and hands off to the library:
//!
//! - `serve` runs the gRPC server.
//! - `model fetch` downloads and verifies a model directory.
//! - `compare` and `calibrate` score records offline, for choosing thresholds.
//!
//! Only `serve` starts an async runtime; the other commands are plain synchronous programs.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use similarity_core::{Device, Embedder};
use similarity_server::config::{Config, DeviceChoice};
use similarity_server::engine::{Engine, store_layout};
use similarity_server::fetch::{self, FetchRequest};
use similarity_server::offline::{self, FieldMap, Offline};
use similarity_server::worker::EmbedWorker;
use similarity_store::{MemoryStore, QdrantStore, VectorStore};

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
  /// The record before the change: a JSON object of field name to text.
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
  tracing::info!(
    dir = %config.model.dir.display(),
    fingerprint = embedder.fingerprint(),
    dimension = embedder.dimension(),
    device = ?config.model.device,
    elapsed = ?started.elapsed(),
    "model loaded"
  );
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
    Ok(())
  }
}

fn read_fields(path: &Path) -> anyhow::Result<FieldMap> {
  let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
  serde_json::from_str(&text).with_context(|| format!("{} must be a JSON object of field name to text", path.display()))
}

fn offline(config_path: &Path) -> anyhow::Result<(Offline, similarity_core::Policy)> {
  let config = Config::load(config_path)?;
  let policy = config.policy()?;
  Ok((Offline::new(load_embedder(&config)?, config.schema()?, policy), policy))
}

impl CompareArgs {
  fn run(self) -> anyhow::Result<()> {
    let (offline, policy) = offline(&self.config)?;
    let evaluation = offline.compare(&read_fields(&self.before)?, &read_fields(&self.after)?)?;
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
  fn run(self) -> anyhow::Result<()> {
    let (offline, policy) = offline(&self.config)?;
    let text = std::fs::read_to_string(&self.pairs).with_context(|| format!("cannot read {}", self.pairs.display()))?;
    let pairs = offline::parse_pairs(&text).with_context(|| format!("in {}", self.pairs.display()))?;
    let calibration = offline::calibrate(&offline, pairs)?;
    write!(
      std::io::stdout().lock(),
      "{}\n{}",
      offline.model_line(),
      offline::render_calibration(&calibration, &policy)
    )?;
    Ok(())
  }
}

fn main() -> ExitCode {
  let cli = Cli::parse();
  // Logs are one JSON object per line on stderr (see `similarity-log`). The reports that `model fetch`,
  // `compare`, and `calibrate` print for a person to read are not logs, and stay as text on stdout.
  similarity_log::init();

  let outcome = match cli.command {
    Command::Serve(args) => args.run(),
    Command::Model(ModelCommand::Fetch(args)) => args.run(),
    Command::Compare(args) => args.run(),
    Command::Calibrate(args) => args.run(),
  };
  match outcome {
    Ok(()) => ExitCode::SUCCESS,
    Err(error) => {
      // A failure is a line in the same stream and the same format as everything before it.
      similarity_log::failed(&format!("{error:#}"));
      ExitCode::FAILURE
    }
  }
}

#[cfg(test)]
mod tests {
  use clap::CommandFactory;

  use super::*;

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
