//! `record-similarity-train`: fine-tunes the embedding model on labeled listing data.
//!
//! Reads a sentence-transformers model directory, trains it on labeled before/after pairs and/or catalog groups,
//! and writes a new model directory that the server loads like any other. See "Fine-tuning the model" in the
//! README for when this is worth doing and what has to be re-done afterwards.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use similarity_core::model::SentenceEncoder;
use similarity_core::{Device, Embedder, EmbedderOptions, TextTokenizer};
use similarity_train::data::{self, PairExample, PairExamples, Split};
use similarity_train::eval::{Separation, separation};
use similarity_train::{Examples, Progress, TrainOptions, fit};

/// Where training runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DeviceChoice {
  /// The CPU. Always available; fine for a few thousand short texts, slow beyond that.
  Cpu,
  /// The GPU of an Apple-silicon Mac. Needs a binary built with `--features metal`.
  Metal,
  /// An NVIDIA GPU. Needs a binary built with `--features cuda`.
  Cuda,
}

/// Fine-tune the record-similarity embedding model on labeled listing data.
#[derive(Debug, Parser)]
#[command(name = "record-similarity-train", version, about)]
struct Cli {
  /// The model directory to start from
  #[arg(long)]
  base: PathBuf,

  /// Where to write the fine-tuned model directory; it must not exist yet, or be empty
  #[arg(long)]
  out: PathBuf,

  /// Labeled before/after pairs, JSONL (`pairs.full.jsonl` or `pairs.jsonl`)
  #[arg(long)]
  pairs: Option<PathBuf>,

  /// Listings grouped by catalog product, JSONL (`catalog-groups.jsonl`)
  #[arg(long)]
  catalog_groups: Option<PathBuf>,

  /// Fields to learn from. Free-text fields only: for a categorical field the same change (say, sativa to hybrid)
  /// appears in both edits and recycles, which teaches the model nothing but noise
  #[arg(long, value_delimiter = ',', default_value = "name,description")]
  fields: Vec<String>,

  /// Train on pairs whose `split` is this, and on pairs that have no `split`
  #[arg(long, default_value = "calibrate")]
  split: String,

  /// Report, but never train on, pairs whose `split` is this
  #[arg(long, default_value = "holdout")]
  holdout_split: String,

  /// Embedding widths to keep usable. Must include the server's `output_dim`; a width the model cannot produce is
  /// replaced by its native width
  #[arg(long, value_delimiter = ',', default_value = "1024,512,256")]
  dims: Vec<usize>,

  /// Passes over the data
  #[arg(long, default_value_t = 2)]
  epochs: usize,

  /// Examples per step
  #[arg(long, default_value_t = 32)]
  batch_size: usize,

  /// Peak learning rate
  #[arg(long, default_value_t = 2e-5)]
  learning_rate: f64,

  /// Truncate texts to this many tokens; match `model.max_tokens` in the server config
  #[arg(long, default_value_t = 256)]
  max_tokens: usize,

  /// Where to run
  #[arg(long, value_enum, default_value_t = DeviceChoice::Cpu)]
  device: DeviceChoice,

  /// Seeds the batch order and dropout
  #[arg(long, default_value_t = 0)]
  seed: u64,

  /// Train without dropout, which makes a CPU run exactly reproducible from its seed
  #[arg(long)]
  no_dropout: bool,

  /// Decoupled weight decay, applied to every parameter
  #[arg(long, default_value_t = 0.0)]
  weight_decay: f32,

  /// Clip gradients to this global norm; 0 disables clipping
  #[arg(long, default_value_t = 1.0)]
  max_grad_norm: f64,

  /// Log progress every this many steps
  #[arg(long, default_value_t = 10)]
  log_every: usize,
}

fn device(choice: DeviceChoice) -> anyhow::Result<Device> {
  match choice {
    DeviceChoice::Cpu => Ok(Device::flex()),
    DeviceChoice::Metal => metal_device(),
    DeviceChoice::Cuda => cuda_device(),
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
  anyhow::bail!("`--device metal` needs a binary built with `--features metal`; this one trains on the CPU only")
}

#[cfg(feature = "cuda")]
#[expect(
  clippy::unnecessary_wraps,
  reason = "must match the signature of the build without CUDA, which fails"
)]
fn cuda_device() -> anyhow::Result<Device> {
  Ok(Device::cuda(0))
}

#[cfg(not(feature = "cuda"))]
fn cuda_device() -> anyhow::Result<Device> {
  anyhow::bail!("`--device cuda` needs a binary built with `--features cuda`; this one trains on the CPU only")
}

/// The widths to train at: `asked`, with anything wider than the model replaced by the model's own width.
///
/// Replaced, not dropped. The native width is the one a model is served at unless it is truncated, so a default
/// such as `1024,512,256` applied to a 384-wide model must still train (and report on) 384, not only the 256
/// prefix.
fn trainable_widths(asked: &[usize], native: usize) -> Vec<usize> {
  let mut widths = Vec::with_capacity(asked.len());
  for width in asked {
    let width = (*width).min(native);
    if !widths.contains(&width) {
      widths.push(width);
    }
  }
  widths
}

/// Refuses an output directory that already holds something, before any time is spent training.
fn ensure_free(out: &Path) -> anyhow::Result<()> {
  let occupied = std::fs::read_dir(out).is_ok_and(|mut entries| entries.next().is_some());
  anyhow::ensure!(
    !occupied && !out.is_file(),
    "{} already exists and is not empty; choose a new directory, a model is never overwritten",
    out.display()
  );
  Ok(())
}

/// Running means of the loss since the last progress line, per kind of batch.
#[derive(Debug, Default)]
struct Window {
  pairs: (f64, usize),
  groups: (f64, usize),
  grad_norm: f64,
  steps: usize,
}

impl Window {
  fn add(&mut self, progress: &Progress) {
    let slot = if progress.ranking {
      &mut self.groups
    } else {
      &mut self.pairs
    };
    slot.0 += progress.report.loss;
    slot.1 = slot.1.saturating_add(1);
    self.grad_norm += progress.report.grad_norm;
    self.steps = self.steps.saturating_add(1);
  }

  fn mean((sum, count): (f64, usize)) -> Option<f64> {
    #[expect(clippy::cast_precision_loss, reason = "a handful of steps")]
    (count > 0).then(|| sum / count as f64)
  }

  fn log(&mut self, progress: &Progress, started: Instant) {
    let shown = |mean: Option<f64>| mean.map_or_else(|| "-".to_owned(), |mean| format!("{mean:.4}"));
    #[expect(clippy::cast_precision_loss, reason = "a handful of steps")]
    let grad_norm = self.grad_norm / self.steps.max(1) as f64;
    tracing::info!(
      step = progress.step,
      of = progress.total_steps,
      epoch = progress.epoch,
      pair_loss = %shown(Self::mean(self.pairs)),
      group_loss = %shown(Self::mean(self.groups)),
      grad_norm = %format!("{grad_norm:.3}"),
      rate = %format!("{:.2e}", progress.rate),
      elapsed = ?started.elapsed(),
      "training"
    );
    *self = Self::default();
  }
}

fn write_separation(
  out: &mut impl Write,
  title: &str,
  before: &[Separation],
  after: &[Separation],
) -> std::io::Result<()> {
  let spread = |spread: Option<similarity_train::eval::Spread>| {
    spread.map_or_else(
      || "       (none)       ".to_owned(),
      |spread| format!("{:.3} / {:.3} / {:.3}", spread.min, spread.mean, spread.max),
    )
  };
  let gap = |separation: &Separation| {
    separation
      .mean_gap()
      .map_or_else(|| "    -".to_owned(), |gap| format!("{gap:+.3}"))
  };
  let count = |pick: fn(&Separation) -> Option<similarity_train::eval::Spread>| {
    before.first().and_then(pick).map_or(0, |spread| spread.count)
  };
  writeln!(
    out,
    "\n{title}: {} edit and {} recycle field texts",
    count(|separation| separation.edits),
    count(|separation| separation.recycles)
  )?;
  writeln!(
    out,
    "  width          edits min / mean / max   recycles min / mean / max   gap of means"
  )?;
  for (before, after) in before.iter().zip(after) {
    for (stage, separation) in [("before", before), ("after ", after)] {
      writeln!(
        out,
        "  {:>5}  {stage}  {}    {}      {}",
        separation.width,
        spread(separation.edits),
        spread(separation.recycles),
        gap(separation)
      )?;
    }
  }
  Ok(())
}

impl Cli {
  fn options(&self, encoder: &SentenceEncoder) -> TrainOptions {
    let widths = trainable_widths(&self.dims, encoder.dimension());
    if widths.iter().ne(self.dims.iter()) {
      tracing::warn!(
        asked = ?self.dims,
        training = ?widths,
        native = encoder.dimension(),
        "widths the model cannot produce were replaced by its native width"
      );
    }
    TrainOptions {
      widths,
      epochs: self.epochs,
      batch_size: self.batch_size,
      learning_rate: self.learning_rate,
      weight_decay: self.weight_decay,
      max_grad_norm: (self.max_grad_norm > 0.0).then_some(self.max_grad_norm),
      dropout: !self.no_dropout,
      seed: self.seed,
      ..TrainOptions::default()
    }
  }

  /// The training pairs and the held-out pairs of `--pairs`.
  fn pairs(&self) -> anyhow::Result<(PairExamples, Vec<PairExample>)> {
    let Some(path) = &self.pairs else {
      return Ok((PairExamples::default(), Vec::new()));
    };
    let jsonl = data::read(path)?;
    let training = data::pair_examples(&jsonl, path, &self.fields, Split::Training(&self.split))?;
    let held_out = data::pair_examples(&jsonl, path, &self.fields, Split::HeldOut(&self.holdout_split))?;
    tracing::info!(
      file = %path.display(),
      examples = training.examples.len(),
      from_edits = training.edits(),
      from_recycles = training.recycles(),
      held_out = held_out.examples.len(),
      skipped_other_labels = training.skipped_label,
      "labeled pairs"
    );
    Ok((training, held_out.examples))
  }

  fn run(&self) -> anyhow::Result<()> {
    ensure_free(&self.out)?;
    anyhow::ensure!(
      self.pairs.is_some() || self.catalog_groups.is_some(),
      "nothing to train on: give --pairs and/or --catalog-groups"
    );

    let (training, held_out) = self.pairs()?;
    let groups = match &self.catalog_groups {
      Some(path) => {
        let groups = data::group_examples(&data::read(path)?, path, &self.fields)?;
        tracing::info!(file = %path.display(), examples = groups.len(), "catalog groups");
        groups
      }
      None => Vec::new(),
    };
    let examples = Examples {
      pairs: training.examples,
      groups,
    };

    let device = device(self.device)?;
    let encoder = SentenceEncoder::load(&self.base, &device)
      .with_context(|| format!("cannot load the base model from {}", self.base.display()))?;
    let tokenizer = TextTokenizer::load(&self.base, self.max_tokens)?;
    anyhow::ensure!(
      self.max_tokens <= encoder.max_positions(),
      "--max-tokens is {} but the model's position table only covers {}",
      self.max_tokens,
      encoder.max_positions()
    );
    let options = self.options(&encoder);
    tracing::info!(
      base = %self.base.display(),
      device = ?self.device,
      native_width = encoder.dimension(),
      widths = ?options.widths,
      dropout = options.dropout,
      "model loaded"
    );

    let score = |encoder: &SentenceEncoder, pairs: &[PairExample]| {
      separation(encoder, &tokenizer, &device, pairs, &options.widths, self.batch_size)
    };
    let before = (score(&encoder, &examples.pairs)?, score(&encoder, &held_out)?);

    let started = Instant::now();
    let mut window = Window::default();
    let trained = fit(
      encoder,
      tokenizer.clone(),
      device.clone(),
      &examples,
      &options,
      |progress| {
        window.add(progress);
        if progress.step.is_multiple_of(self.log_every.max(1)) || progress.step == progress.total_steps {
          window.log(progress, started);
        }
      },
    )?;
    let elapsed = started.elapsed();
    let after = (score(&trained, &examples.pairs)?, score(&trained, &held_out)?);

    trained
      .save(&self.out)
      .with_context(|| format!("cannot save the model to {}", self.out.display()))?;
    // Load what was just written through the path the server uses, including its warm-up forward pass, so that a
    // directory this command reports as saved is one the server is known to start with.
    let embedder_options = EmbedderOptions {
      max_tokens: self.max_tokens,
      ..EmbedderOptions::default()
    };
    Embedder::load(&self.out, Device::flex(), &embedder_options)
      .with_context(|| format!("the model saved to {} does not load back", self.out.display()))?;

    let mut out = std::io::stdout().lock();
    writeln!(
      out,
      "trained on {} pair and {} catalog examples in {elapsed:.1?}",
      examples.pairs.len(),
      examples.groups.len()
    )?;
    writeln!(
      out,
      "similarity of single field texts, not record verdicts: use `record-similarity calibrate` for those"
    )?;
    if !examples.pairs.is_empty() {
      write_separation(&mut out, "training pairs (seen during training)", &before.0, &after.0)?;
    }
    if held_out.is_empty() {
      writeln!(
        out,
        "\nno pair has `split: \"{}\"`, so there is no held-out measurement",
        self.holdout_split
      )?;
    } else {
      write_separation(&mut out, "held-out pairs (never trained on)", &before.1, &after.1)?;
    }
    writeln!(out, "\nsaved to {} and verified that it loads", self.out.display())?;
    writeln!(
      out,
      "next: point `model.dir` at it, re-calibrate the thresholds on held-out pairs, and re-index (README.md)"
    )?;
    Ok(())
  }
}

fn main() -> anyhow::Result<()> {
  let cli = Cli::parse();
  tracing_subscriber::fmt()
    .with_env_filter(
      tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    )
    .with_writer(std::io::stderr)
    .init();
  cli.run()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_width_the_model_cannot_produce_becomes_its_native_width() {
    assert_eq!(trainable_widths(&[1024, 512, 256], 1024), [1024, 512, 256]);
    // A 384-wide model served at 384 must be trained at 384, not only at the 256 prefix.
    assert_eq!(trainable_widths(&[1024, 512, 256], 384), [384, 256]);
    assert_eq!(trainable_widths(&[1024], 384), [384]);
    assert_eq!(trainable_widths(&[256, 256], 384), [256]);
    assert!(trainable_widths(&[], 384).is_empty());
  }
}
