//! `record-similarity-train`: fine-tunes the embedding model on labeled listing data.
//!
//! Reads a sentence-transformers model directory, trains it on labeled before/after pairs and/or catalog groups,
//! and writes a new model directory that the server loads like any other. See "Fine-tuning the model" in the
//! README for when this is worth doing and what has to be re-done afterwards.
//!
//! Everything it says is one JSON object per line (see `similarity-log`): progress on stderr, and what the run
//! produced on stdout. A run is checkpointed at the end of every pass, and an interrupted one continues with
//! `--resume`.

// Tests compare exact floats where exactness is the point (what a rounded number is rounded to).
#![cfg_attr(test, allow(clippy::float_cmp))]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use sha2::{Digest as _, Sha256};
use similarity_core::model::SentenceEncoder;
use similarity_core::provenance::{Base, Build, DataFile, Resumed, Trained};
use similarity_core::record::hex_prefix;
use similarity_core::{Device, Embedder, EmbedderOptions, Provenance, TextTokenizer};
use similarity_log::RESULT;
use similarity_train::checkpoint::{self, RunIdentity};
use similarity_train::data::{self, PairExample, PairExamples, Split};
use similarity_train::eval::{Separation, Spread, separation};
use similarity_train::{Examples, Progress, Step, TrainOptions, fit};

/// What this binary knows about itself, for the provenance of every model directory it writes.
const BUILD: Build = Build::new(
  "record-similarity-train",
  env!("CARGO_PKG_VERSION"),
  env!("RECORD_SIMILARITY_BUILD_COMMIT"),
  env!("RECORD_SIMILARITY_BUILD_DIRTY"),
);

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
  #[arg(long, required_unless_present = "resume", conflicts_with = "resume")]
  base: Option<PathBuf>,

  /// Continue an interrupted run from this checkpoint directory, in place of --base. Every other option, and the
  /// data, must be what the run was started with
  #[arg(long)]
  resume: Option<PathBuf>,

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

  /// Where to write checkpoints; must not be inside --out [default: <OUT>.checkpoints]
  #[arg(long)]
  checkpoint_dir: Option<PathBuf>,

  /// Also checkpoint every this many steps (at least 1). One is always written at the end of each pass
  #[arg(long)]
  checkpoint_every: Option<NonZeroUsize>,

  /// Keep only this many of the run's checkpoints, deleting the oldest; 0 keeps them all
  #[arg(long, default_value_t = 0)]
  keep_checkpoints: usize,

  /// Write no checkpoints
  #[arg(long, conflicts_with_all = ["checkpoint_dir", "checkpoint_every"])]
  no_checkpoints: bool,
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

/// `value` to `places` decimal places, so that a log line carries a number and not fifteen digits of noise.
///
/// Through the decimal text and back, which is exact about what "four decimal places" means; scaling by a power
/// of ten and dividing again leaves digits like `0.30000000000000004` behind.
fn rounded(value: f64, places: usize) -> f64 {
  format!("{value:.places$}").parse().unwrap_or(value)
}

/// `value` to three significant digits, for quantities such as a learning rate that span orders of magnitude.
fn significant(value: f64) -> f64 {
  format!("{value:.2e}").parse().unwrap_or(value)
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
    (count > 0).then(|| rounded(sum / count as f64, 4))
  }

  /// One progress event. Every value is a number, so that it can be charted without parsing strings; a loss is
  /// absent when no batch of its kind fell in the window. (None of them can be NaN or infinite: the trainer stops
  /// a run on the first loss or gradient that is.)
  fn log(&mut self, progress: &Progress, started: Instant) {
    #[expect(clippy::cast_precision_loss, reason = "a handful of steps")]
    let grad_norm = self.grad_norm / self.steps.max(1) as f64;
    tracing::info!(
      step = progress.step,
      of = progress.total_steps,
      epoch = progress.epoch,
      pair_loss = Self::mean(self.pairs),
      group_loss = Self::mean(self.groups),
      grad_norm = rounded(grad_norm, 3),
      rate = significant(progress.rate),
      elapsed_s = rounded(started.elapsed().as_secs_f64(), 1),
      "training"
    );
    *self = Self::default();
  }
}

/// When and where a run is checkpointed, and which checkpoints on disk are its own.
#[derive(Debug)]
struct Checkpoints {
  dir: PathBuf,
  every: Option<NonZeroUsize>,
  keep: usize,
  /// This run's checkpoints, oldest first: the ones `keep` counts and may delete.
  own: Vec<PathBuf>,
}

/// The step a checkpoint directory name (`step-000120`) stands for.
fn step_of(name: &str) -> Option<usize> {
  name.strip_prefix("step-")?.parse().ok()
}

impl Checkpoints {
  fn path(&self, step: usize) -> PathBuf {
    self.dir.join(format!("step-{step:06}"))
  }

  /// Makes sure, before any training, that checkpoints can be written where they are meant to go.
  ///
  /// Everything checked here would otherwise be found out at the first checkpoint: a run lost part-way, to the
  /// very mechanism meant to prevent that. `done` is the step the run starts from (0 unless resuming).
  ///
  /// It also takes stock of the run's earlier checkpoints, so that `--keep-checkpoints` bounds what is on disk
  /// across a resume and not only within one process. A checkpoint is counted, and so may later be deleted, only
  /// if its own state file says it was written by this same run, and never if it is `source`, the checkpoint
  /// this process was loaded from: the model keeps reading its configuration and tokenizer files from there
  /// every time it is saved, up to and including the final one.
  fn prepare(&mut self, out: &Path, source: Option<&Path>, done: usize, run: &RunIdentity) -> anyhow::Result<()> {
    let same = |left: &Path, right: &Path| match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
      (Ok(left), Ok(right)) => left == right,
      _ => left.components().eq(right.components()),
    };
    let normalized = |path: &Path| path.components().collect::<PathBuf>();
    anyhow::ensure!(
      !normalized(&self.dir).starts_with(normalized(out)),
      "the checkpoint directory {} is inside the output directory {}, which has to be empty when the model is \
       written; give another --checkpoint-dir",
      self.dir.display(),
      out.display()
    );
    std::fs::create_dir_all(&self.dir)
      .with_context(|| format!("cannot create the checkpoint directory {}", self.dir.display()))?;
    let entries = std::fs::read_dir(&self.dir)
      .with_context(|| format!("cannot read the checkpoint directory {}", self.dir.display()))?;

    let mut earlier = Vec::new();
    for entry in entries {
      let entry = entry.with_context(|| format!("cannot read the checkpoint directory {}", self.dir.display()))?;
      let Some(step) = entry.file_name().to_str().and_then(step_of) else {
        continue;
      };
      anyhow::ensure!(
        step <= done,
        "{} already holds a checkpoint for step {step}, which this run would write again; remove the checkpoints \
         after step {done}, or give another --checkpoint-dir",
        self.dir.display()
      );
      let in_use = source.is_some_and(|source| same(source, &entry.path()));
      if !in_use && checkpoint::read(&entry.path()).is_ok_and(|state| checkpoint::same_run(&state, run)) {
        earlier.push((step, entry.path()));
      }
    }
    earlier.sort();
    self.own = earlier.into_iter().map(|(_, path)| path).collect();
    Ok(())
  }

  fn due(&self, progress: &Progress) -> bool {
    // The end of the run is the model in --out, written as soon as training ends; a checkpoint of it would be a
    // second copy of the same weights.
    progress.step < progress.total_steps
      && (progress.epoch_complete
        || self
          .every
          .is_some_and(|every| progress.step.is_multiple_of(every.get())))
  }

  fn after(&mut self, step: &Step<'_>, lineage: &Provenance) -> similarity_train::Result<()> {
    if !self.due(&step.progress) {
      return Ok(());
    }
    let path = self.path(step.progress.step);
    step.checkpoint(&path, Some(lineage))?;
    tracing::info!(
      step = step.progress.step,
      epoch = step.progress.epoch,
      path = %path.display(),
      "checkpoint"
    );
    self.own.push(path);
    self.prune();
    Ok(())
  }

  /// Deletes the oldest of this run's checkpoints beyond the number to keep.
  ///
  /// Never an error. The new checkpoint is already safely written, and failing to tidy up behind it is no reason
  /// to throw the run away; a checkpoint someone has moved elsewhere to keep is simply no longer here to delete.
  fn prune(&mut self) {
    while self.keep > 0 && self.own.len() > self.keep {
      let oldest = self.own.remove(0);
      match std::fs::remove_dir_all(&oldest) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => tracing::warn!(
          path = %oldest.display(),
          %error,
          "could not delete an old checkpoint; training continues"
        ),
      }
    }
  }
}

/// What a run measured and produced, reported once at the end.
struct Outcome<'a> {
  out: &'a Path,
  examples: &'a Examples,
  held_out: usize,
  resumed_at: Option<usize>,
  steps: usize,
  elapsed_s: f64,
  /// Before and after, on the training pairs and on the held-out pairs.
  training: (&'a [Separation], &'a [Separation]),
  held: (&'a [Separation], &'a [Separation]),
}

/// The lowest, mean, and highest similarity of one label's examples, rounded for reporting.
fn spread(side: Option<Spread>) -> [Option<f64>; 3] {
  let number = |value: f32| rounded(f64::from(value), 4);
  match side {
    Some(spread) => [
      Some(number(spread.min)),
      Some(number(spread.mean)),
      Some(number(spread.max)),
    ],
    None => [None; 3],
  }
}

impl Outcome<'_> {
  /// Reports the results: one `separation` event per split, stage, and width that has something to report, then
  /// one `finished` event for the run. These are the run's output, so they go to stdout and no log level hides
  /// them.
  fn report(&self) {
    // When a run was resumed, its "before" is the checkpoint it resumed from, which is not where the run began.
    let baseline = if self.resumed_at.is_some() {
      "checkpoint"
    } else {
      "base model"
    };
    let splits = [
      ("training", self.examples.pairs.len(), self.training),
      ("held_out", self.held_out, self.held),
    ];
    for (split, examples, (before, after)) in splits {
      // No pairs in a split means no measurement, which is not the same as a measurement of zero examples.
      if examples == 0 {
        continue;
      }
      for (stage, rows) in [("before", before), ("after", after)] {
        for row in rows {
          let [edits_min, edits_mean, edits_max] = spread(row.edits);
          let [recycles_min, recycles_mean, recycles_max] = spread(row.recycles);
          tracing::info!(
            target: RESULT,
            split,
            stage,
            compared_with = (stage == "before").then_some(baseline),
            width = row.width,
            edits = row.edits.map_or(0, |spread| spread.count),
            edits_min,
            edits_mean,
            edits_max,
            recycles = row.recycles.map_or(0, |spread| spread.count),
            recycles_min,
            recycles_mean,
            recycles_max,
            gap_of_means = row.mean_gap().map(|gap| rounded(f64::from(gap), 4)),
            "separation"
          );
        }
      }
    }
    tracing::info!(
      target: RESULT,
      out = %self.out.display(),
      pair_examples = self.examples.pairs.len(),
      catalog_examples = self.examples.groups.len(),
      held_out_examples = self.held_out,
      steps = self.steps,
      resumed_at = self.resumed_at,
      elapsed_s = rounded(self.elapsed_s, 1),
      "finished"
    );
  }
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

  /// Everything to train on, and the held-out pairs to report on.
  fn examples(&self) -> anyhow::Result<(Examples, Vec<PairExample>)> {
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
    Ok((examples, held_out))
  }

  /// Where and when this run checkpoints, if it does.
  fn checkpoints(&self) -> Option<Checkpoints> {
    if self.no_checkpoints {
      return None;
    }
    let dir = self
      .checkpoint_dir
      .clone()
      .unwrap_or_else(|| checkpoint::sibling(&self.out, ".checkpoints"));
    Some(Checkpoints {
      dir,
      every: self.checkpoint_every,
      keep: self.keep_checkpoints,
      own: Vec::new(),
    })
  }

  /// Loads the model to train: the base model, or the checkpoint being resumed, which is a model directory too.
  fn load(&self, device: &Device) -> anyhow::Result<(SentenceEncoder, TextTokenizer, &Path)> {
    let Some(source) = self.resume.as_deref().or(self.base.as_deref()) else {
      anyhow::bail!("give --base, or --resume to continue an interrupted run");
    };
    let encoder = SentenceEncoder::load(source, device)
      .with_context(|| format!("cannot load the model from {}", source.display()))?;
    let tokenizer = TextTokenizer::load(source, self.max_tokens)?;
    anyhow::ensure!(
      self.max_tokens <= encoder.max_positions(),
      "--max-tokens is {} but the model's position table only covers {}",
      self.max_tokens,
      encoder.max_positions()
    );
    Ok((encoder, tokenizer, source))
  }

  /// Where the model this run writes will have come from: everything about that which is known before training.
  ///
  /// The digest of the weights and the step they are at are filled in when each directory is written.
  fn lineage(
    &self,
    encoder: &SentenceEncoder,
    source: &Path,
    examples: &Examples,
    options: &TrainOptions,
  ) -> anyhow::Result<Provenance> {
    let data_file = |role: &str, path: &Path, examples: usize| -> anyhow::Result<DataFile> {
      Ok(DataFile {
        role: role.to_owned(),
        path: path.display().to_string(),
        sha256: hex_prefix(&Sha256::digest(data::read(path)?.as_bytes()), 32),
        examples,
      })
    };
    let mut files = Vec::new();
    if let Some(path) = &self.pairs {
      files.push(data_file("pairs", path, examples.pairs.len())?);
    }
    if let Some(path) = &self.catalog_groups {
      files.push(data_file("catalog_groups", path, examples.groups.len())?);
    }

    // The provenance of the directory the weights were loaded from. A model without one is not refused: the
    // record simply starts here.
    let inherited = Provenance::read(source).unwrap_or_else(|error| {
      tracing::warn!(%error, "the provenance of the model being trained from cannot be read and is left out");
      None
    });
    let loaded_from = || -> anyhow::Result<Base> {
      Ok(Base {
        dir: source.display().to_string(),
        digest: encoder.files().digest()?,
        provenance: None,
      })
    };
    let (base, resumed) = match (&self.resume, inherited) {
      (Some(checkpoint), inherited) => {
        // A resumed run is still the run that started from the original base. The checkpoint is where it was
        // interrupted, and goes on the list of those.
        let earlier = inherited.and_then(|provenance| provenance.trained);
        let mut resumed = earlier
          .as_ref()
          .map_or_else(Vec::new, |trained| trained.resumed.clone());
        resumed.push(Resumed {
          from: checkpoint.display().to_string(),
          step: checkpoint::read(checkpoint)?.step,
        });
        let base = match earlier {
          Some(trained) => trained.base,
          None => loaded_from()?,
        };
        (base, resumed)
      }
      (None, inherited) => {
        let base = Base {
          provenance: inherited.map(Box::new),
          ..loaded_from()?
        };
        (base, Vec::new())
      }
    };

    let mut provenance = BUILD.provenance(String::new());
    provenance.trained = Some(Trained {
      base,
      data: files,
      data_fingerprint: checkpoint::data_digest(examples),
      options: serde_json::to_value(options).context("the training options do not serialize")?,
      max_tokens: self.max_tokens,
      device: format!("{:?}", self.device).to_lowercase(),
      resumed,
      step: 0,
      total_steps: 0,
    });
    Ok(provenance)
  }

  fn run(&self) -> anyhow::Result<()> {
    ensure_free(&self.out)?;
    anyhow::ensure!(
      self.pairs.is_some() || self.catalog_groups.is_some(),
      "nothing to train on: give --pairs and/or --catalog-groups"
    );
    let resumed_at = match &self.resume {
      Some(dir) => Some(checkpoint::read(dir)?.step),
      None => None,
    };

    let (examples, held_out) = self.examples()?;
    let device = device(self.device)?;
    let (encoder, tokenizer, source) = self.load(&device)?;
    let options = self.options(&encoder);
    let mut checkpoints = self.checkpoints();
    if let Some(checkpoints) = &mut checkpoints {
      let run = RunIdentity::of(&options, &examples, self.max_tokens);
      checkpoints.prepare(&self.out, self.resume.as_deref(), resumed_at.unwrap_or(0), &run)?;
    }
    tracing::info!(
      model = %source.display(),
      resumed_at,
      device = ?self.device,
      native_width = encoder.dimension(),
      widths = ?options.widths,
      dropout = options.dropout,
      checkpoints = checkpoints.as_ref().map(|checkpoints| checkpoints.dir.display().to_string()),
      "model loaded"
    );

    let lineage = self.lineage(&encoder, source, &examples, &options)?;

    let score = |encoder: &SentenceEncoder, pairs: &[PairExample]| {
      separation(encoder, &tokenizer, &device, pairs, &options.widths, self.batch_size)
    };
    let before = (score(&encoder, &examples.pairs)?, score(&encoder, &held_out)?);

    let started = Instant::now();
    let mut window = Window::default();
    let mut steps = 0;
    let trained = fit(
      encoder,
      tokenizer.clone(),
      device.clone(),
      &examples,
      &options,
      self.resume.as_deref(),
      |step| {
        let progress = &step.progress;
        steps = progress.total_steps;
        window.add(progress);
        if progress.step.is_multiple_of(self.log_every.max(1)) || progress.step == progress.total_steps {
          window.log(progress, started);
        }
        match &mut checkpoints {
          Some(checkpoints) => checkpoints.after(step, &lineage),
          None => Ok(()),
        }
      },
    )?;
    let elapsed_s = started.elapsed().as_secs_f64();

    // Saved before anything else is done with it. Until this returns, the last pass exists only in memory, and
    // the scoring below is minutes of forward passes on a large dataset.
    checkpoint::save_model(&trained, &self.out, Some((&lineage, steps)))?;
    // Loaded back through the path the server uses, including its warm-up forward pass, so that a directory this
    // command reports as saved is one the server is known to start with.
    let embedder_options = EmbedderOptions {
      max_tokens: self.max_tokens,
      ..EmbedderOptions::default()
    };
    Embedder::load(&self.out, Device::flex(), &embedder_options)
      .with_context(|| format!("the model saved to {} does not load back", self.out.display()))?;
    tracing::info!(out = %self.out.display(), "model saved and verified to load");

    let after = (score(&trained, &examples.pairs)?, score(&trained, &held_out)?);
    Outcome {
      out: &self.out,
      examples: &examples,
      held_out: held_out.len(),
      resumed_at,
      steps,
      elapsed_s,
      training: (&before.0, &after.0),
      held: (&before.1, &after.1),
    }
    .report();
    Ok(())
  }
}

fn main() -> ExitCode {
  let cli = Cli::parse();
  similarity_log::init();
  match cli.run() {
    Ok(()) => ExitCode::SUCCESS,
    Err(error) => {
      similarity_log::failed(&format!("{error:#}"));
      ExitCode::FAILURE
    }
  }
}

#[cfg(test)]
mod tests {
  use similarity_log::testing::capture;
  use similarity_train::StepReport;
  use similarity_train::data::GroupExample;

  use super::*;

  #[test]
  fn log_numbers_are_rounded_to_what_is_worth_reading() {
    assert_eq!(rounded(0.016_234_567, 4), 0.0162);
    assert_eq!(rounded(0.1 + 0.2, 4), 0.3);
    assert_eq!(significant(1.894_736_8e-5), 1.89e-5);
    assert_eq!(significant(2e-5), 2e-5);
    assert_eq!(significant(12_345.678), 12_300.0);
    assert_eq!(significant(0.0), 0.0);
    // Far outside anything a learning rate takes, and still a number.
    assert_eq!(significant(1.234_5e-310), 1.23e-310);
  }

  fn progress(step: usize, total_steps: usize, epoch_complete: bool) -> Progress {
    Progress {
      step,
      total_steps,
      epoch: 1,
      epoch_complete,
      examples: 4,
      ranking: false,
      rate: 2e-5,
      report: StepReport {
        loss: 0.1,
        grad_norm: 1.0,
      },
    }
  }

  fn checkpoints(dir: &Path, every: Option<usize>, keep: usize) -> Checkpoints {
    Checkpoints {
      dir: dir.to_path_buf(),
      every: every.and_then(NonZeroUsize::new),
      keep,
      own: Vec::new(),
    }
  }

  fn examples() -> Examples {
    Examples {
      pairs: vec![PairExample {
        before: "blueberry gummies".to_owned(),
        after: "blueberry gummies mg".to_owned(),
        same: true,
      }],
      groups: vec![GroupExample {
        catalog: "vape cartridge".to_owned(),
        listing: "sativa vape cartridge".to_owned(),
      }],
    }
  }

  fn identity(seed: u64) -> RunIdentity {
    let options = TrainOptions {
      seed,
      ..TrainOptions::default()
    };
    RunIdentity::of(&options, &examples(), 256)
  }

  /// A directory that reads as a checkpoint of `run` taken after `step` steps.
  fn fake_checkpoint(dir: &Path, step: usize, run: &RunIdentity) -> PathBuf {
    let path = dir.join(format!("step-{step:06}"));
    std::fs::create_dir_all(&path).unwrap();
    let state = serde_json::json!({"format": 1, "step": step, "total_steps": 99, "run": run, "param_ids": []});
    std::fs::write(path.join(checkpoint::STATE_FILE), state.to_string()).unwrap();
    path
  }

  #[test]
  fn a_checkpoint_is_due_at_the_end_of_each_pass_and_never_at_the_end_of_the_run() {
    let by_pass = checkpoints(Path::new("c"), None, 0);
    assert!(by_pass.due(&progress(10, 30, true)));
    assert!(!by_pass.due(&progress(11, 30, false)));
    assert!(
      !by_pass.due(&progress(30, 30, true)),
      "the final model is the output, not a checkpoint"
    );

    let by_step = checkpoints(Path::new("c"), Some(4), 0);
    assert!(by_step.due(&progress(8, 30, false)));
    assert!(by_step.due(&progress(10, 30, true)), "the end of a pass still counts");
    assert!(!by_step.due(&progress(9, 30, false)));
    assert_eq!(by_step.path(8), Path::new("c/step-000008"));
  }

  #[test]
  fn a_run_refuses_to_start_over_checkpoints_it_would_have_to_write_again() {
    let dir = tempfile::tempdir().unwrap();
    let run = identity(0);
    for step in [4, 8] {
      fake_checkpoint(dir.path(), step, &run);
    }
    std::fs::create_dir(dir.path().join("notes")).unwrap();
    let out = dir.path().join("out");

    // Resuming from step 8: nothing later exists, so there is nothing to collide with.
    assert!(checkpoints(dir.path(), None, 0).prepare(&out, None, 8, &run).is_ok());
    // Resuming from step 4, or starting over: step 8 is in the way.
    let error = checkpoints(dir.path(), None, 0)
      .prepare(&out, None, 4, &run)
      .unwrap_err();
    assert!(error.to_string().contains("step 8"), "{error}");
    assert!(checkpoints(dir.path(), None, 0).prepare(&out, None, 0, &run).is_err());
  }

  #[test]
  fn a_place_checkpoints_cannot_go_is_found_out_before_training_not_at_the_first_checkpoint() {
    let scratch = tempfile::tempdir().unwrap();
    let run = identity(0);
    let out = scratch.path().join("out");

    // Missing is fine: it is created now, so that the first checkpoint cannot fail on it later.
    let fresh = scratch.path().join("a/b/checkpoints");
    checkpoints(&fresh, None, 0).prepare(&out, None, 0, &run).unwrap();
    assert!(fresh.is_dir());

    // Under a regular file there can never be a directory.
    let file = scratch.path().join("file");
    std::fs::write(&file, b"x").unwrap();
    let error = checkpoints(&file.join("checkpoints"), None, 0)
      .prepare(&out, None, 0, &run)
      .unwrap_err();
    assert!(
      error.to_string().contains("cannot create the checkpoint directory"),
      "{error}"
    );

    // Inside the output directory, the final save would find that directory not empty and refuse.
    for inside in [
      out.join("checkpoints"),
      out.clone(),
      scratch.path().join("out/.checkpoints"),
    ] {
      let error = checkpoints(&inside, None, 0).prepare(&out, None, 0, &run).unwrap_err();
      assert!(error.to_string().contains("inside the output directory"), "{error}");
    }
    assert!(!out.exists(), "nothing was created inside the output directory");
  }

  #[test]
  fn the_default_checkpoint_directory_sits_beside_the_output_however_the_output_is_spelled() {
    for out in ["models/ft", "models/ft/"] {
      let cli = Cli::parse_from(["x", "--base", "b", "--out", out, "--pairs", "p"]);
      assert_eq!(
        cli.checkpoints().unwrap().dir,
        Path::new("models/ft.checkpoints"),
        "{out}"
      );
    }
  }

  #[test]
  fn keeping_the_newest_checkpoints_counts_the_runs_earlier_ones_and_only_those() {
    let dir = tempfile::tempdir().unwrap();
    let (run, other_run) = (identity(0), identity(1));
    let out = dir.path().join("out");
    let first = fake_checkpoint(dir.path(), 2, &run);
    let second = fake_checkpoint(dir.path(), 4, &run);
    // Same directory, another run's checkpoint, and something that is not a checkpoint at all.
    let foreign = fake_checkpoint(dir.path(), 3, &other_run);
    let stray = dir.path().join("step-000001");
    std::fs::create_dir(&stray).unwrap();

    // A resume from step 4 that keeps two: the earlier process's checkpoints count toward the two.
    let mut resumed = checkpoints(dir.path(), None, 2);
    resumed.prepare(&out, None, 4, &run).unwrap();
    assert_eq!(resumed.own, [first.clone(), second.clone()]);

    // The checkpoint a run was loaded from is never counted, so never deleted: the model reads its tokenizer
    // and configuration files from that directory again each time it is saved.
    let mut loaded_from_second = checkpoints(dir.path(), None, 1);
    loaded_from_second.prepare(&out, Some(&second), 4, &run).unwrap();
    assert_eq!(loaded_from_second.own, std::slice::from_ref(&first));

    let third = fake_checkpoint(dir.path(), 6, &run);
    resumed.own.push(third.clone());
    resumed.prune();
    assert!(!first.exists(), "the oldest of this run's three was deleted");
    assert!(second.exists() && third.exists());
    assert!(
      foreign.exists() && stray.exists(),
      "nothing that is not this run's is ever deleted"
    );

    // A checkpoint someone moved away to keep is not an error, and neither is anything else about tidying up.
    std::fs::remove_dir_all(&second).unwrap();
    resumed.own.push(fake_checkpoint(dir.path(), 8, &run));
    resumed.prune();
    assert_eq!(resumed.own.len(), 2);
    assert!(third.exists());
  }

  #[test]
  fn resume_stands_in_for_base_and_impossible_checkpoint_options_are_refused_at_once() {
    let fresh = Cli::parse_from(["x", "--base", "b", "--out", "o", "--pairs", "p"]);
    assert!(fresh.base.is_some() && fresh.resume.is_none());
    let resumed = Cli::parse_from(["x", "--resume", "c/step-000004", "--out", "o", "--pairs", "p"]);
    assert!(resumed.base.is_none());

    let refused = |arguments: &[&str]| Cli::try_parse_from(arguments).is_err();
    assert!(
      refused(&["x", "--out", "o", "--pairs", "p"]),
      "neither --base nor --resume"
    );
    assert!(refused(&["x", "--base", "b", "--resume", "c", "--out", "o"]));
    // Zero would mean "every step": 265 MB per step until the disk is full.
    assert!(refused(&["x", "--base", "b", "--out", "o", "--checkpoint-every", "0"]));
    assert!(
      refused(&["x", "--base", "b", "--out", "o", "--log-format", "json"]),
      "there is one format"
    );
    let none = Cli::parse_from(["x", "--base", "b", "--out", "o", "--pairs", "p", "--no-checkpoints"]);
    assert!(none.checkpoints().is_none());
  }

  fn row(width: usize, edits: Option<(usize, f32)>, recycles: Option<(usize, f32)>) -> Separation {
    let spread = |side: Option<(usize, f32)>| {
      side.map(|(count, mean)| Spread {
        count,
        min: mean - 0.1,
        mean,
        max: mean + 0.1,
      })
    };
    Separation {
      width,
      edits: spread(edits),
      recycles: spread(recycles),
    }
  }

  fn reported(outcome: &Outcome<'_>, level: &str) -> (Vec<serde_json::Value>, usize) {
    let captured = capture(level, || outcome.report());
    let results = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    (results, captured.logs.len())
  }

  #[test]
  fn results_are_json_lines_on_stdout_that_no_log_level_can_hide() {
    let examples = examples();
    let before = [row(256, Some((5, 0.9)), Some((8, 0.6)))];
    let after = [row(256, Some((5, 0.95)), Some((8, 0.5)))];
    let outcome = Outcome {
      out: Path::new("models/ft"),
      examples: &examples,
      held_out: 13,
      resumed_at: None,
      steps: 60,
      elapsed_s: 19.84,
      training: (&before, &after),
      held: (&before, &after),
    };

    for level in ["info", "warn", "off"] {
      let (results, log_lines) = reported(&outcome, level);
      assert_eq!(log_lines, 0, "{level}: results are not log lines");
      assert_eq!(
        results.len(),
        5,
        "{level}: two splits, two stages, one width, and the run"
      );

      let held_after = &results[3];
      assert_eq!(held_after["message"], "separation");
      assert_eq!(
        (&held_after["split"], &held_after["stage"]),
        (&"held_out".into(), &"after".into())
      );
      assert_eq!(held_after["width"], 256);
      assert_eq!((&held_after["edits"], &held_after["recycles"]), (&5.into(), &8.into()));
      // Numbers, rounded: something a chart can use as it is.
      assert_eq!(held_after["edits_mean"], 0.95);
      assert_eq!(held_after["recycles_max"], 0.6);
      assert_eq!(held_after["gap_of_means"], 0.45);
      assert!(
        held_after.get("compared_with").is_none(),
        "only a `before` row has a baseline"
      );
      assert_eq!(results[2]["compared_with"], "base model");

      let finished = &results[4];
      assert_eq!(finished["message"], "finished");
      assert_eq!(finished["out"], "models/ft");
      assert_eq!(
        (&finished["steps"], &finished["held_out_examples"]),
        (&60.into(), &13.into())
      );
      assert_eq!(finished["elapsed_s"], 19.8);
      assert!(finished.get("resumed_at").is_none());
    }
  }

  #[test]
  fn a_split_with_no_pairs_reports_no_measurement_and_a_resumed_run_says_what_before_means() {
    // Catalog groups only, nothing held out, resumed from step 10.
    let examples = Examples {
      pairs: Vec::new(),
      ..examples()
    };
    let empty = [row(256, None, None)];
    let outcome = Outcome {
      out: Path::new("o"),
      examples: &examples,
      held_out: 0,
      resumed_at: Some(10),
      steps: 60,
      elapsed_s: 1.0,
      training: (&empty, &empty),
      held: (&empty, &empty),
    };
    let (results, _) = reported(&outcome, "info");
    assert_eq!(
      results.len(),
      1,
      "no `separation` event claims a measurement that was never made"
    );
    assert_eq!(results[0]["message"], "finished");
    assert_eq!(results[0]["resumed_at"], 10);

    // With held-out pairs, the `before` rows of a resumed run say they describe the checkpoint.
    let measured = [row(256, Some((5, 0.9)), None)];
    let outcome = Outcome {
      held_out: 5,
      held: (&measured, &measured),
      ..outcome
    };
    let (results, _) = reported(&outcome, "info");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["compared_with"], "checkpoint");
    // No recycles among them: their statistics are left out, not written as zeros or nulls.
    assert_eq!(results[0]["recycles"], 0);
    assert!(results[0].get("recycles_mean").is_none() && results[0].get("gap_of_means").is_none());
  }

  #[test]
  fn progress_is_logged_as_numbers_and_an_absent_loss_is_left_out() {
    let mut window = Window::default();
    window.add(&progress(10, 60, false));
    let captured = capture("info", || window.log(&progress(10, 60, false), Instant::now()));
    assert!(captured.results.is_empty());
    let line: serde_json::Value = serde_json::from_str(&captured.logs[0]).unwrap();
    assert_eq!(line["message"], "training");
    assert_eq!((&line["step"], &line["of"]), (&10.into(), &60.into()));
    assert_eq!(line["pair_loss"], 0.1);
    assert_eq!(line["grad_norm"], 1.0);
    assert_eq!(line["rate"], 2e-5);
    assert!(line["elapsed_s"].is_number());
    assert!(
      line.get("group_loss").is_none(),
      "no catalog batch fell in this window: {line}"
    );
  }

  #[test]
  fn a_width_the_model_cannot_produce_becomes_its_native_width() {
    assert_eq!(trainable_widths(&[1024, 512, 256], 1024), [1024, 512, 256]);
    // A 384-wide model served at 384 must be trained at 384, not only at the 256 prefix.
    assert_eq!(trainable_widths(&[1024, 512, 256], 384), [384, 256]);
    assert_eq!(trainable_widths(&[1024], 384), [384]);
    assert_eq!(trainable_widths(&[256, 256], 384), [256]);
    assert!(trainable_widths(&[], 384).is_empty());
  }

  /// A tiny base model that says it was fetched, and four labeled pairs in words its tokenizer knows.
  fn workspace() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    std::fs::create_dir(&base).unwrap();
    similarity_core::testing::write_tiny_model(&base, Some(24));
    let digest = SentenceEncoder::load(&base, &Device::flex())
      .unwrap()
      .files()
      .digest()
      .unwrap();
    let mut fetched = Build::new("record-similarity model fetch", "0.1.0", "abc", "false").provenance(digest);
    fetched.fetched = Some(similarity_core::provenance::Fetched {
      repo: "example/tiny".to_owned(),
      revision: "r1".to_owned(),
      files: Vec::new(),
    });
    fetched.write(&base).unwrap();

    let pairs = dir.path().join("pairs.jsonl");
    let line = |label: &str, before: &str, after: &str| {
      format!(r#"{{"label": "{label}", "before": {{"name": "{before}"}}, "after": {{"name": "{after}"}}}}"#)
    };
    let text = [
      line("edit", "blueberry gummies", "blueberry gummies mg"),
      line("edit", "indica flower", "indica flower resin"),
      line("recycle", "blueberry gummies", "vape cartridge"),
      line("recycle", "sativa flower", "wyld edibles"),
    ]
    .join("\n");
    std::fs::write(&pairs, text).unwrap();
    (dir, base, pairs)
  }

  fn train(arguments: &[&str]) {
    let fixed = [
      "x",
      "--epochs",
      "2",
      "--batch-size",
      "2",
      "--max-tokens",
      "16",
      "--dims",
      "24",
      "--fields",
      "name",
      "--no-dropout",
    ];
    Cli::parse_from(fixed.iter().chain(arguments)).run().unwrap();
  }

  fn provenance(dir: &Path) -> (Provenance, Trained) {
    let provenance = Provenance::read(dir).unwrap().expect("a provenance file");
    let trained = provenance.trained.clone().expect("a trained model");
    (provenance, trained)
  }

  #[test]
  fn a_trained_model_and_its_checkpoints_record_where_they_came_from() {
    let (dir, base, pairs) = workspace();
    let out = dir.path().join("out");
    let path = |path: &Path| path.to_str().unwrap().to_owned();
    train(&["--base", &path(&base), "--out", &path(&out), "--pairs", &path(&pairs)]);

    let (model, trained) = provenance(&out);
    assert_eq!(model.tool, "record-similarity-train");
    assert_eq!(model.version, env!("CARGO_PKG_VERSION"));
    assert!(!model.commit.is_empty());
    let options = EmbedderOptions {
      max_tokens: 16,
      ..EmbedderOptions::default()
    };
    let loaded = Embedder::load(&out, Device::flex(), &options).unwrap();
    assert_eq!(
      model.digest,
      loaded.content_digest(),
      "the digest is that of the model the directory holds"
    );

    // What it was trained from, with that model's own origin nested inside.
    let base_digest = SentenceEncoder::load(&base, &Device::flex())
      .unwrap()
      .files()
      .digest()
      .unwrap();
    assert_eq!(
      (trained.base.dir.as_str(), &trained.base.digest),
      (base.to_str().unwrap(), &base_digest)
    );
    assert_ne!(model.digest, base_digest, "training moved the weights");
    let origin = trained.base.provenance.as_ref().expect("the base's provenance");
    assert_eq!(origin.fetched.as_ref().unwrap().repo, "example/tiny");

    // On what, and how.
    let [data] = trained.data.as_slice() else {
      panic!("one data file expected: {:?}", trained.data)
    };
    assert_eq!((data.role.as_str(), data.examples), ("pairs", 4));
    assert_eq!(data.path, pairs.to_str().unwrap());
    assert_eq!(
      data.sha256,
      hex_prefix(&Sha256::digest(std::fs::read(&pairs).unwrap()), 32)
    );
    assert_eq!(trained.options["epochs"], 2);
    assert_eq!(trained.options["dropout"], false);
    assert_eq!((trained.max_tokens, trained.device.as_str()), (16, "cpu"));
    assert_eq!((trained.step, trained.total_steps), (4, 4));
    assert!(trained.resumed.is_empty());

    // The checkpoint after the first pass says the same, for the weights it holds.
    let checkpoint = checkpoint::sibling(&out, ".checkpoints").join("step-000002");
    let (at_checkpoint, partway) = provenance(&checkpoint);
    assert_eq!((partway.step, partway.total_steps), (2, 4));
    assert_eq!(partway.base, trained.base);
    assert_ne!(at_checkpoint.digest, model.digest, "it holds other weights");
    assert_eq!(
      at_checkpoint.digest,
      SentenceEncoder::load(&checkpoint, &Device::flex())
        .unwrap()
        .files()
        .digest()
        .unwrap()
    );
  }

  #[test]
  fn a_resumed_run_keeps_its_original_base_and_lists_where_it_was_resumed() {
    let (dir, base, pairs) = workspace();
    let path = |path: &Path| path.to_str().unwrap().to_owned();
    let first = dir.path().join("first");
    train(&["--base", &path(&base), "--out", &path(&first), "--pairs", &path(&pairs)]);
    let checkpoint = checkpoint::sibling(&first, ".checkpoints").join("step-000002");

    let second = dir.path().join("second");
    train(&[
      "--resume",
      &path(&checkpoint),
      "--out",
      &path(&second),
      "--pairs",
      &path(&pairs),
      "--checkpoint-dir",
      &path(&dir.path().join("second-checkpoints")),
    ]);

    let (_, resumed) = provenance(&second);
    let (_, uninterrupted) = provenance(&first);
    assert_eq!(
      resumed.base, uninterrupted.base,
      "still the run that started from the base model, not from its own checkpoint"
    );
    assert_eq!(
      resumed.resumed,
      [Resumed {
        from: path(&checkpoint),
        step: 2
      }]
    );
    assert_eq!((resumed.step, resumed.total_steps), (4, 4));
    assert_eq!(resumed.data_fingerprint, uninterrupted.data_fingerprint);
  }

  #[test]
  fn a_base_model_without_provenance_is_trained_from_all_the_same() {
    let (dir, base, pairs) = workspace();
    std::fs::remove_file(base.join(similarity_core::provenance::PROVENANCE_FILE)).unwrap();
    let out = dir.path().join("out");
    let path = |path: &Path| path.to_str().unwrap().to_owned();
    train(&[
      "--base",
      &path(&base),
      "--out",
      &path(&out),
      "--pairs",
      &path(&pairs),
      "--no-checkpoints",
    ]);
    let (_, trained) = provenance(&out);
    assert_eq!(trained.base.provenance, None, "the record starts here");
    assert_eq!(trained.base.digest.len(), 16);
  }
}
