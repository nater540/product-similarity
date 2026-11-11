//! The training loop: tokenize a batch, encode both sides, take the loss, step the optimizer.
//!
//! [`fit`] is the whole procedure, including resuming from a [checkpoint](crate::checkpoint). [`Session`] is one
//! layer down, a single optimizer step at a time, and exists so that tests can drive fixed batches at a fixed
//! learning rate and compare the outcome with PyTorch's.
//!
//! The recipe is the one sentence-embedding models are conventionally fine-tuned with: `AdamW` with a linear
//! warm-up and decay ([`crate::schedule`]), gradients clipped to a global norm of 1, and the encoder's own dropout
//! active. Every one of those is an option, because a parity test needs them off and a small dataset may want them
//! different.

use std::path::Path;

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Gradients, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use similarity_core::model::{EncoderWeights, SentenceEncoder, TrainingDropout};
use similarity_core::tokenize::pad_batch;
use similarity_core::{Device, TextTokenizer};

use crate::checkpoint::{self, RunIdentity};
use crate::data::{self, Batch, GroupExample, PairExample, Rng};
use crate::error::{Result, TrainError};
use crate::loss::{DEFAULT_MARGIN, DEFAULT_SCALE, Objective, truncated};
use crate::schedule::LinearSchedule;

/// Everything that shapes a training run.
///
/// Serializable because a checkpoint records the options it was made under: resuming under different ones would
/// continue on a different batch order or learning-rate schedule without anything looking wrong.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainOptions {
  /// The embedding widths the loss is applied at. Must include the width the server truncates to (`output_dim`),
  /// and should include the model's native width so the full embedding stays usable too.
  pub widths: Vec<usize>,
  /// How many passes over the data.
  pub epochs: usize,
  /// Examples per step. The ranking loss uses the rest of the batch as negatives, so larger is a harder, more
  /// informative task, up to what fits in memory.
  pub batch_size: usize,
  /// The peak learning rate.
  pub learning_rate: f64,
  /// The fraction of all steps spent warming up to the peak.
  pub warmup_ratio: f64,
  /// Decoupled weight decay, applied to every parameter. Zero by default: fine-tuning starts from weights that
  /// are already good, and the data here is small enough that pulling them toward zero costs more than it buys.
  pub weight_decay: f32,
  /// Scale the gradients down whenever their global norm exceeds this. `None` disables clipping.
  pub max_grad_norm: Option<f64>,
  /// Apply the encoder's dropout. Off only for runs that must be exactly reproducible.
  pub dropout: bool,
  /// Seeds the batch order and, as far as the backend allows, dropout.
  pub seed: u64,
  /// See [`DEFAULT_MARGIN`].
  pub margin: f64,
  /// See [`DEFAULT_SCALE`].
  pub scale: f64,
}

impl Default for TrainOptions {
  fn default() -> Self {
    Self {
      widths: vec![1024, 512, 256],
      epochs: 2,
      batch_size: 32,
      learning_rate: 2e-5,
      warmup_ratio: 0.1,
      weight_decay: 0.0,
      max_grad_norm: Some(1.0),
      dropout: true,
      seed: 0,
      margin: DEFAULT_MARGIN,
      scale: DEFAULT_SCALE,
    }
  }
}

impl TrainOptions {
  fn validate(&self, encoder: &SentenceEncoder) -> Result<()> {
    let refuse = |reason: String| Err(TrainError::Options(reason));
    if self.widths.is_empty() {
      return refuse("no embedding width to train at".to_owned());
    }
    if let Some(width) = self
      .widths
      .iter()
      .find(|width| **width == 0 || **width > encoder.dimension())
    {
      return refuse(format!(
        "width {width} is outside 1..={}, the widths this model can produce",
        encoder.dimension()
      ));
    }
    if self.epochs == 0 {
      return refuse("`epochs` must be positive".to_owned());
    }
    if self.batch_size < 2 {
      return refuse(
        "`batch_size` must be at least 2: the ranking loss needs another example to rank against".to_owned(),
      );
    }
    if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
      return refuse(format!(
        "`learning_rate` must be a positive number, not {}",
        self.learning_rate
      ));
    }
    if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
      return refuse(format!(
        "`weight_decay` must not be negative, not {}",
        self.weight_decay
      ));
    }
    if self.max_grad_norm.is_some_and(|norm| !(norm.is_finite() && norm > 0.0)) {
      return refuse("`max_grad_norm` must be a positive number".to_owned());
    }
    Ok(())
  }
}

/// The examples to train on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Examples {
  /// From labeled pairs.
  pub pairs: Vec<PairExample>,
  /// From catalog groups.
  pub groups: Vec<GroupExample>,
}

/// The texts of one step.
#[derive(Debug, Clone)]
pub enum TextBatch<'a> {
  /// Trained with the contrastive loss.
  Pairs(Vec<&'a PairExample>),
  /// Trained with the ranking loss.
  Groups(Vec<&'a GroupExample>),
}

/// What one optimizer step did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
  /// The loss on the batch, before the update.
  pub loss: f64,
  /// The global gradient norm, before any clipping.
  pub grad_norm: f64,
}

/// A step's outcome and where in the run it happened, for progress reporting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
  /// The one-based step just completed.
  pub step: usize,
  /// How many steps the run has.
  pub total_steps: usize,
  /// The one-based pass over the data this step belongs to.
  pub epoch: usize,
  /// Whether this step was the last of its pass.
  pub epoch_complete: bool,
  /// How many examples the step trained on.
  pub examples: usize,
  /// Whether the step used the ranking loss (catalog groups) rather than the contrastive one (labeled pairs).
  pub ranking: bool,
  /// The learning rate the step was taken at.
  pub rate: f64,
  /// The loss and gradient norm.
  pub report: StepReport,
}

/// Adds up the squared entries of every parameter's gradient.
struct SquaredNorm<'a> {
  grads: &'a Gradients,
  total: Option<Tensor<1>>,
}

impl ModuleVisitor for SquaredNorm<'_> {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
    if let Some(grad) = param.val().grad(self.grads) {
      let squared = grad.square().sum();
      self.total = Some(match self.total.take() {
        Some(total) => total.add(squared),
        None => squared,
      });
    }
  }
}

/// Multiplies every parameter's gradient by one factor.
struct Scale<'a> {
  grads: &'a mut Gradients,
  factor: f64,
}

impl ModuleVisitor for Scale<'_> {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
    let tensor = param.val();
    if let Some(grad) = tensor.grad(self.grads) {
      tensor.grad_replace(self.grads, grad.mul_scalar(self.factor));
    }
  }
}

/// The norm of all of `weights`' gradients taken as one long vector.
fn global_norm(grads: &Gradients, weights: &EncoderWeights) -> f64 {
  let mut visitor = SquaredNorm { grads, total: None };
  weights.visit(&mut visitor);
  visitor
    .total
    .map_or(0.0, |total| f64::from(total.into_scalar::<f32>()).sqrt())
}

/// Scales the gradients so that their global norm is at most `max_norm`, and returns the norm they had.
///
/// One factor for every tensor, so the direction of the update is unchanged; only its size is limited. Burn's
/// built-in norm clipping works on each parameter tensor separately, which bends the direction, so it is not used.
fn clip_to_global_norm(grads: &mut Gradients, weights: &EncoderWeights, max_norm: Option<f64>) -> f64 {
  let norm = global_norm(grads, weights);
  // A norm that is not finite is left alone for the caller to refuse: scaling by `max / inf` would multiply
  // infinities by zero and hand back NaN gradients that look clipped.
  if let Some(max_norm) = max_norm
    && norm.is_finite()
    && norm > max_norm
  {
    // The small constant mirrors `torch.nn.utils.clip_grad_norm_`, so that both produce the same update.
    let factor = max_norm / (norm + 1e-6);
    weights.visit(&mut Scale { grads, factor });
  }
  norm
}

/// What [`fit`] hands its caller after every step: the progress so far, and the means to checkpoint the run there.
#[derive(Debug)]
pub struct Step<'a> {
  /// Where the run is and what the step did.
  pub progress: Progress,
  encoder: &'a SentenceEncoder,
  session: &'a Session,
  run: &'a RunIdentity,
}

impl Step<'_> {
  /// Writes the run as it stands after this step to `dir`, which must not exist yet.
  ///
  /// The result is a complete model directory (so it can be served or calibrated like any other) that also holds
  /// what [`fit`] needs to continue from here. See [`crate::checkpoint`].
  ///
  /// # Errors
  ///
  /// Returns [`TrainError::Checkpoint`] if `dir` exists or cannot be written.
  pub fn checkpoint(&self, dir: &Path) -> Result<()> {
    checkpoint::write(
      dir,
      self.encoder,
      self.session,
      self.progress.step,
      self.progress.total_steps,
      self.run,
    )
  }
}

/// The state of a training run between steps: everything except the model itself.
///
/// The encoder is passed through [`Session::step`] by value, because an optimizer step consumes the weights it
/// updates and returns new ones.
pub struct Session {
  tokenizer: TextTokenizer,
  device: Device,
  dropout: TrainingDropout,
  optimizer: ModuleOptimizer,
  widths: Vec<usize>,
  margin: f64,
  scale: f64,
  max_grad_norm: Option<f64>,
  steps: usize,
}

impl std::fmt::Debug for Session {
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    formatter
      .debug_struct("Session")
      .field("widths", &self.widths)
      .field("steps", &self.steps)
      .finish_non_exhaustive()
  }
}

impl Session {
  /// Starts a run: checks `options` against the model and switches the encoder's weights to tracking gradients.
  ///
  /// `encoder` must have been loaded on `device`, which is a plain device (`Device::flex()` or a GPU), not an
  /// autodiff one; training mode is entered here.
  ///
  /// # Errors
  ///
  /// Returns [`TrainError::Options`] if the options cannot be trained with this model.
  pub fn begin(
    encoder: SentenceEncoder,
    tokenizer: TextTokenizer,
    device: Device,
    options: &TrainOptions,
  ) -> Result<(Self, SentenceEncoder)> {
    Self::begin_from(encoder, tokenizer, device, options, None)
  }

  /// [`Self::begin`], continuing from the optimizer file of a checkpoint.
  ///
  /// Not public, because it is only correct after [`checkpoint::restore`] has given the encoder's parameters the
  /// ids that file is keyed by. Called on any other encoder it succeeds and then trains as if there were no saved
  /// state at all, which is precisely the failure a caller could not see. [`fit`] is the way to resume.
  pub(crate) fn begin_from(
    encoder: SentenceEncoder,
    tokenizer: TextTokenizer,
    device: Device,
    options: &TrainOptions,
    optimizer_state: Option<&Path>,
  ) -> Result<(Self, SentenceEncoder)> {
    options.validate(&encoder)?;
    device.seed(options.seed);
    let dropout = if options.dropout {
      encoder.dropout()
    } else {
      TrainingDropout::disabled()
    };
    let optimizer = AdamWConfig::new()
      // Burn's defaults (1e-5 and 1e-4) are not PyTorch's; these are the values the reference recipe uses.
      .with_epsilon(1e-8)
      .with_weight_decay(options.weight_decay)
      .init();
    let optimizer = match optimizer_state {
      Some(path) => optimizer.load(path).map_err(|source| TrainError::Checkpoint {
        path: path.to_path_buf(),
        reason: format!("cannot read the optimizer state: {source}"),
      })?,
      None => optimizer,
    };
    let session = Self {
      tokenizer,
      device,
      dropout,
      optimizer,
      widths: options.widths.clone(),
      margin: options.margin,
      scale: options.scale,
      max_grad_norm: options.max_grad_norm,
      steps: 0,
    };
    Ok((session, encoder.map_weights(Module::train)))
  }

  /// Writes the optimizer's state (for `AdamW`, the two running moments of every parameter) to `path`.
  ///
  /// # Errors
  ///
  /// Returns [`TrainError::Checkpoint`] if the file cannot be written.
  pub(crate) fn save_optimizer(&self, path: &Path) -> Result<()> {
    self.optimizer.save(path).map_err(|source| TrainError::Checkpoint {
      path: path.to_path_buf(),
      reason: format!("cannot write the optimizer state: {source}"),
    })
  }

  /// Ends a run: returns the encoder with plain weights, ready to embed or to be saved.
  pub fn finish(encoder: SentenceEncoder) -> SentenceEncoder {
    encoder.map_weights(|weights| weights.valid())
  }

  /// Encodes `texts` with gradients, `[texts, native width]`.
  fn encode(&self, encoder: &SentenceEncoder, texts: &[&str]) -> Result<Tensor<2>> {
    let rows = texts
      .iter()
      .map(|text| self.tokenizer.encode(text))
      .collect::<similarity_core::Result<Vec<_>>>()?;
    let rows: Vec<&[u32]> = rows.iter().map(Vec::as_slice).collect();
    let (ids, padding) = pad_batch(&rows, &self.device)
      .ok_or_else(|| TrainError::Backend("a training batch has no tokens to encode".to_owned()))?;
    Ok(encoder.forward_training(ids, padding, &self.dropout))
  }

  /// Takes one optimizer step on `batch` at learning rate `rate`.
  ///
  /// # Errors
  ///
  /// Returns [`TrainError::Model`] if a text cannot be tokenized and [`TrainError::Diverged`] if the loss or the
  /// gradient is not finite, in which case the step is not taken. The encoder is consumed either way; a failed
  /// run has no model worth keeping.
  pub fn step(
    &mut self,
    encoder: SentenceEncoder,
    batch: &TextBatch<'_>,
    rate: f64,
  ) -> Result<(SentenceEncoder, StepReport)> {
    self.steps = self.steps.saturating_add(1);
    let (left, right, objective): (Vec<&str>, Vec<&str>, Objective) = match batch {
      TextBatch::Pairs(examples) => {
        let same: Vec<f32> = examples
          .iter()
          .map(|example| if example.same { 1.0 } else { 0.0 })
          .collect();
        let rows = same.len();
        (
          examples.iter().map(|example| example.before.as_str()).collect(),
          examples.iter().map(|example| example.after.as_str()).collect(),
          Objective::Contrastive {
            same: Tensor::from_data(TensorData::new(same, [rows]), &self.device),
            margin: self.margin,
          },
        )
      }
      TextBatch::Groups(examples) => (
        examples.iter().map(|example| example.catalog.as_str()).collect(),
        examples.iter().map(|example| example.listing.as_str()).collect(),
        Objective::Ranking { scale: self.scale },
      ),
    };

    let left = self.encode(&encoder, &left)?;
    let right = self.encode(&encoder, &right)?;
    let loss = truncated(&left, &right, &self.widths, &objective);
    let value = f64::from(loss.clone().into_scalar::<f32>());
    if !value.is_finite() {
      return Err(TrainError::Diverged {
        step: self.steps,
        quantity: "loss",
        value,
      });
    }

    let mut grads = loss.backward();
    let grad_norm = clip_to_global_norm(&mut grads, encoder.weights(), self.max_grad_norm);
    // A finite loss does not guarantee a finite gradient. Applying one that is not would make the weights NaN
    // one step before the loss shows it, and a checkpoint taken in between would be written from them.
    if !grad_norm.is_finite() {
      return Err(TrainError::Diverged {
        step: self.steps,
        quantity: "gradient norm",
        value: grad_norm,
      });
    }
    let grads = GradientsParams::from_grads(grads, encoder.weights());
    let optimizer = &mut self.optimizer;
    let encoder = encoder.map_weights(|weights| optimizer.step(rate, weights, grads));
    Ok((encoder, StepReport { loss: value, grad_norm }))
  }
}

/// Fine-tunes `encoder` on `examples` and returns it with plain weights.
///
/// `after_step` is called after every step, with the progress and the means to write a checkpoint; an error from
/// it stops the run. The batch order of every pass is fixed before the first step, so the number of steps, and
/// with it the learning-rate schedule, is known exactly.
///
/// To continue an interrupted run, load `encoder` from a checkpoint directory and pass that directory as `resume`,
/// with the same `examples`, `options`, and tokenizer limit as the original run (an encoder loaded from anywhere
/// else is refused). The run then picks up at the checkpoint's step: the
/// same batches in the same order, the same position in the learning-rate schedule, and the optimizer state it
/// had. Without dropout the result is identical to that of an uninterrupted run; with dropout it differs only in
/// which units are dropped after the resume, since the random stream starts over.
///
/// # Errors
///
/// Returns [`TrainError::NoExamples`] if there is nothing to train on, [`TrainError::Options`] if the options do
/// not fit the model, [`TrainError::Checkpoint`] if `resume` is not a checkpoint of this same run, and whatever
/// [`Session::step`] or `after_step` returns.
pub fn fit(
  encoder: SentenceEncoder,
  tokenizer: TextTokenizer,
  device: Device,
  examples: &Examples,
  options: &TrainOptions,
  resume: Option<&Path>,
  mut after_step: impl FnMut(&Step<'_>) -> Result<()>,
) -> Result<SentenceEncoder> {
  if examples.pairs.is_empty() && examples.groups.is_empty() {
    return Err(TrainError::NoExamples(
      "no field of any pair or catalog group has two different texts to compare".to_owned(),
    ));
  }
  // Before the batches are planned: zero passes or a zero batch size would otherwise surface as "no examples".
  options.validate(&encoder)?;

  let mut rng = Rng::new(options.seed);
  let plan: Vec<Vec<Batch>> = (0..options.epochs)
    .map(|_| data::epoch(&examples.pairs, &examples.groups, options.batch_size, &mut rng))
    .collect();
  let total_steps = plan.iter().map(Vec::len).sum();
  if total_steps == 0 {
    return Err(TrainError::NoExamples(
      "every catalog example belongs to the same catalog entry, so there is nothing to rank any of them against"
        .to_owned(),
    ));
  }
  let scheduled = |batches: &Vec<Batch>| -> usize {
    batches
      .iter()
      .filter(|batch| matches!(batch, Batch::Groups(_)))
      .map(Batch::len)
      .sum()
  };
  if let Some(first) = plan.first()
    && scheduled(first) < examples.groups.len()
  {
    tracing::warn!(
      examples = examples.groups.len(),
      used_per_pass = scheduled(first),
      "some catalog examples are left out of each pass: nothing else in the data to rank them against"
    );
  }
  let schedule = LinearSchedule::new(options.learning_rate, total_steps, options.warmup_ratio);

  let run = RunIdentity::of(options, examples, tokenizer.max_tokens());
  let (encoder, done, optimizer_state) = match resume {
    Some(dir) => {
      let (encoder, done) = checkpoint::restore(dir, encoder, &run, total_steps)?;
      (encoder, done, Some(checkpoint::optimizer_file(dir)))
    }
    None => (encoder, 0, None),
  };
  let (mut session, mut encoder) =
    Session::begin_from(encoder, tokenizer, device, options, optimizer_state.as_deref())?;
  session.steps = done;

  let mut step = 0_usize;
  for (epoch, batches) in plan.iter().enumerate() {
    for (position, batch) in batches.iter().enumerate() {
      step = step.saturating_add(1);
      if step <= done {
        continue;
      }
      let texts = match batch {
        Batch::Pairs(indexes) => {
          TextBatch::Pairs(indexes.iter().filter_map(|index| examples.pairs.get(*index)).collect())
        }
        Batch::Groups(indexes) => {
          TextBatch::Groups(indexes.iter().filter_map(|index| examples.groups.get(*index)).collect())
        }
      };
      let rate = schedule.rate(step.saturating_sub(1));
      let (updated, report) = session.step(encoder, &texts, rate)?;
      encoder = updated;
      after_step(&Step {
        progress: Progress {
          step,
          total_steps,
          epoch: epoch.saturating_add(1),
          epoch_complete: position.saturating_add(1) == batches.len(),
          examples: batch.len(),
          ranking: matches!(batch, Batch::Groups(_)),
          rate,
          report,
        },
        encoder: &encoder,
        session: &session,
        run: &run,
      })?;
    }
  }
  Ok(Session::finish(encoder))
}

#[cfg(test)]
mod tests {
  use similarity_core::testing::write_tiny_model;

  use super::*;
  use crate::eval::separation;

  const MAX_TOKENS: usize = 16;

  struct Fixture {
    _dir: tempfile::TempDir,
    encoder: SentenceEncoder,
    tokenizer: TextTokenizer,
    device: Device,
  }

  /// A tiny random model with a 24-wide projection, loaded from disk the way a real one is.
  fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_model(dir.path(), Some(24));
    let device = Device::flex();
    Fixture {
      encoder: SentenceEncoder::load(dir.path(), &device).unwrap(),
      tokenizer: TextTokenizer::load(dir.path(), MAX_TOKENS).unwrap(),
      device,
      _dir: dir,
    }
  }

  fn options() -> TrainOptions {
    TrainOptions {
      widths: vec![24, 8],
      epochs: 30,
      batch_size: 4,
      learning_rate: 3e-3,
      dropout: false,
      ..TrainOptions::default()
    }
  }

  fn pair(before: &str, after: &str, same: bool) -> PairExample {
    PairExample {
      before: before.to_owned(),
      after: after.to_owned(),
      same,
    }
  }

  fn pairs() -> Vec<PairExample> {
    vec![
      pair("blueberry gummies", "blueberry gummies mg", true),
      pair("wyld gummies indica", "wyld gummies", true),
      pair("vape cartridge sativa", "sativa vape cartridge", true),
      pair("flower hybrid", "hybrid flower resin", true),
      pair("blueberry gummies", "vape cartridge", false),
      pair("wyld gummies indica", "flower sativa", false),
      pair("vape cartridge sativa", "edibles gummies", false),
      pair("flower hybrid", "blueberry edibles mg", false),
    ]
  }

  fn groups() -> Vec<GroupExample> {
    [
      ("blueberry gummies", "wyld blueberry gummies mg"),
      ("vape cartridge", "sativa vape cartridge"),
      ("flower", "hybrid flower"),
      ("resin", "indica resin mg"),
    ]
    .into_iter()
    .map(|(catalog, listing)| GroupExample {
      catalog: catalog.to_owned(),
      listing: listing.to_owned(),
    })
    .collect()
  }

  fn embeddings(encoder: &SentenceEncoder, fixture_tokenizer: &TextTokenizer, device: &Device) -> Vec<f32> {
    let rows = [fixture_tokenizer.encode("blueberry gummies vape").unwrap()];
    let rows: Vec<&[u32]> = rows.iter().map(Vec::as_slice).collect();
    let (ids, padding) = pad_batch(&rows, device).unwrap();
    encoder.forward(ids, padding).into_data().try_into_vec::<f32>().unwrap()
  }

  #[test]
  fn training_on_labeled_pairs_pulls_edits_together_and_pushes_recycles_apart() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let examples = Examples {
      pairs: pairs(),
      groups: Vec::new(),
    };
    let before = separation(&encoder, &tokenizer, &device, &examples.pairs, &[8], 4).unwrap()[0];

    let mut losses = Vec::new();
    let trained = fit(
      encoder,
      tokenizer.clone(),
      device.clone(),
      &examples,
      &options(),
      None,
      |step| {
        losses.push(step.progress.report.loss);
        Ok(())
      },
    )
    .unwrap();
    let after = separation(&trained, &tokenizer, &device, &examples.pairs, &[8], 4).unwrap()[0];

    assert_eq!(losses.len(), 60, "8 examples in batches of 4, for 30 passes");
    let (first, last) = (losses[..4].iter().sum::<f64>(), losses[56..].iter().sum::<f64>());
    assert!(last < first * 0.5, "the loss should fall: {first} -> {last}");
    // Where training must end up, whatever random model it started from: the loss pulls an edit's two texts
    // together without limit, and pushes a recycle's apart until their cosine distance reaches the margin of 0.5.
    // (Asserting an improvement over the start instead would fail whenever the random start happens to separate
    // the pairs already, which is about one run in 150. Over 300 random starts the trained values stayed above
    // 0.95 and below 0.52.)
    let (edits, recycles) = (after.edits.unwrap(), after.recycles.unwrap());
    assert!(edits.min > 0.9, "{before:?} -> {after:?}");
    assert!(recycles.max < 0.6, "{before:?} -> {after:?}");
  }

  #[test]
  fn training_on_catalog_groups_teaches_each_entry_to_pick_out_its_own_listing() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let examples = Examples {
      pairs: Vec::new(),
      groups: groups(),
    };
    let mut reports = Vec::new();
    fit(encoder, tokenizer, device, &examples, &options(), None, |step| {
      assert!(step.progress.ranking);
      reports.push(step.progress);
      Ok(())
    })
    .unwrap();

    assert_eq!(reports.len(), 30);
    let (first, last) = (reports[0].report.loss, reports[29].report.loss);
    // Four candidates, so a model that cannot tell them apart scores ln(4) = 1.39.
    assert!(last < 0.2 && last < first, "{first} -> {last}");
    assert!(reports.iter().all(|progress| progress.report.grad_norm > 0.0));
  }

  #[test]
  fn both_kinds_of_example_are_trained_in_one_run_on_the_scheduled_rate() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let examples = Examples {
      pairs: pairs(),
      groups: groups(),
    };
    let run_options = TrainOptions { epochs: 2, ..options() };
    let mut seen = Vec::new();
    fit(encoder, tokenizer, device, &examples, &run_options, None, |step| {
      seen.push(step.progress);
      Ok(())
    })
    .unwrap();

    // Per pass: two batches of pairs and one of groups.
    assert_eq!(seen.len(), 6);
    assert_eq!(seen.iter().filter(|progress| progress.ranking).count(), 2);
    assert_eq!(
      seen.iter().map(|progress| progress.examples).sum::<usize>(),
      2 * (8 + 4)
    );
    assert_eq!(seen.last().unwrap().epoch, 2);
    assert!(seen.iter().all(|progress| progress.total_steps == 6));
    // One warm-up step (10% of 6, rounded up), so the first step is already at the peak, then a linear decay.
    let rates: Vec<f64> = seen.iter().map(|progress| progress.rate).collect();
    assert!((rates[0] - 3e-3).abs() < 1e-12);
    assert!(rates.windows(2).skip(1).all(|pair| pair[1] < pair[0]), "{rates:?}");
  }

  #[test]
  fn without_dropout_a_run_is_reproducible_from_its_seed() {
    let run = || {
      let dir = tempfile::tempdir().unwrap();
      // The same initial weights for both runs: the tiny model is random, so it is written once and copied.
      write_tiny_model(dir.path(), Some(24));
      dir
    };
    let base = run();
    let train = |seed: u64| {
      let device = Device::flex();
      let encoder = SentenceEncoder::load(base.path(), &device).unwrap();
      let tokenizer = TextTokenizer::load(base.path(), MAX_TOKENS).unwrap();
      let examples = Examples {
        pairs: pairs(),
        groups: groups(),
      };
      let run_options = TrainOptions {
        epochs: 3,
        seed,
        ..options()
      };
      let trained = fit(
        encoder,
        tokenizer.clone(),
        device.clone(),
        &examples,
        &run_options,
        None,
        |_| Ok(()),
      )
      .unwrap();
      embeddings(&trained, &tokenizer, &device)
    };
    assert_eq!(train(7), train(7));
    assert_ne!(train(7), train(8), "another seed orders the batches differently");
  }

  #[test]
  fn dropout_perturbs_training_passes_and_never_inference() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let run_options = TrainOptions {
      dropout: true,
      ..options()
    };
    let plain = embeddings(&encoder, &tokenizer, &device);
    let (session, encoder) = Session::begin(encoder, tokenizer.clone(), device.clone(), &run_options).unwrap();

    let pass = || {
      let output = session.encode(&encoder, &["blueberry gummies vape"]).unwrap();
      output.into_data().try_into_vec::<f32>().unwrap()
    };
    assert_ne!(pass(), pass(), "two training passes should drop different units");

    // The ordinary forward pass ignores dropout even while the weights are in training mode, and again after.
    // (Tracking gradients reorders a few float operations, hence a tolerance rather than equality; dropout at 10%
    // moves these values in the first decimal place.)
    let unchanged = |found: Vec<f32>| plain.iter().zip(&found).all(|(a, b)| (a - b).abs() < 1e-5);
    assert!(unchanged(embeddings(&encoder, &tokenizer, &device)));
    let finished = Session::finish(encoder);
    assert!(unchanged(embeddings(&finished, &tokenizer, &device)));
  }

  #[test]
  fn a_trained_model_saves_and_reloads_to_the_same_embeddings() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let untrained = embeddings(&encoder, &tokenizer, &device);
    let examples = Examples {
      pairs: pairs(),
      groups: groups(),
    };
    let run_options = TrainOptions { epochs: 2, ..options() };
    let trained = fit(
      encoder,
      tokenizer.clone(),
      device.clone(),
      &examples,
      &run_options,
      None,
      |_| Ok(()),
    )
    .unwrap();
    let expected = embeddings(&trained, &tokenizer, &device);
    assert_ne!(untrained, expected, "training should have moved the weights");

    let out = tempfile::tempdir().unwrap();
    let saved = out.path().join("fine-tuned");
    trained.save(&saved).unwrap();
    let reloaded = SentenceEncoder::load(&saved, &device).unwrap();
    assert_eq!(expected, embeddings(&reloaded, &tokenizer, &device));
    // The tokenizer travels with the weights, so the saved directory is a complete model.
    assert!(TextTokenizer::load(&saved, MAX_TOKENS).is_ok());
  }

  /// Trains the fixture's data for three passes (nine steps), optionally checkpointing after `checkpoint_at`
  /// steps into `dir`, and optionally resuming from `resume`.
  fn run(base: &Path, checkpoint: Option<(usize, &Path)>, resume: Option<&Path>, seed: u64) -> Result<Vec<f32>> {
    let device = Device::flex();
    let encoder = SentenceEncoder::load(resume.unwrap_or(base), &device)?;
    let tokenizer = TextTokenizer::load(base, MAX_TOKENS)?;
    let examples = Examples {
      pairs: pairs(),
      groups: groups(),
    };
    let run_options = TrainOptions {
      epochs: 3,
      seed,
      ..options()
    };
    let trained = fit(
      encoder,
      tokenizer.clone(),
      device.clone(),
      &examples,
      &run_options,
      resume,
      |step| match checkpoint {
        Some((at, dir)) if step.progress.step == at => step.checkpoint(dir),
        _ => Ok(()),
      },
    )?;
    Ok(embeddings(&trained, &tokenizer, &device))
  }

  #[test]
  fn a_run_resumed_from_a_checkpoint_ends_exactly_where_an_uninterrupted_one_does() {
    let base = tempfile::tempdir().unwrap();
    write_tiny_model(base.path(), Some(24));
    let scratch = tempfile::tempdir().unwrap();
    let checkpoint = scratch.path().join("step-4");

    let uninterrupted = run(base.path(), Some((4, &checkpoint)), None, 7).unwrap();
    let resumed = run(base.path(), None, Some(&checkpoint), 7).unwrap();
    assert_eq!(uninterrupted, resumed);

    // The optimizer state is what makes that exact. Without it the weights still load, the remaining steps still
    // run, and the result is a different model: the failure a checkpoint that forgot the parameter ids would
    // produce without any error.
    std::fs::remove_file(checkpoint::optimizer_file(&checkpoint)).unwrap();
    let error = run(base.path(), None, Some(&checkpoint), 7).unwrap_err();
    assert!(matches!(error, TrainError::Checkpoint { .. }), "{error}");
  }

  /// Continues the run checkpointed after four steps by hand: `fit`'s loop written out, so that the one step
  /// under test, restoring the parameter ids, can be left out.
  fn continue_by_hand(base: &Path, checkpoint: &Path, restore_ids: bool) -> Vec<f32> {
    let device = Device::flex();
    let encoder = SentenceEncoder::load(checkpoint, &device).unwrap();
    let tokenizer = TextTokenizer::load(base, MAX_TOKENS).unwrap();
    let run_options = TrainOptions {
      epochs: 3,
      seed: 7,
      ..options()
    };
    let examples = Examples {
      pairs: pairs(),
      groups: groups(),
    };
    let mut rng = Rng::new(7);
    let plan: Vec<Batch> = (0..3)
      .flat_map(|_| data::epoch(&examples.pairs, &examples.groups, 4, &mut rng))
      .collect();

    let encoder = if restore_ids {
      let run = RunIdentity::of(&run_options, &examples, MAX_TOKENS);
      checkpoint::restore(checkpoint, encoder, &run, plan.len()).unwrap().0
    } else {
      // The parameters keep the fresh ids they were built with, and the optimizer state loaded below is filed
      // under ids that no longer exist.
      encoder
    };
    let optimizer = checkpoint::optimizer_file(checkpoint);
    let (mut session, mut encoder) = Session::begin_from(
      encoder,
      tokenizer.clone(),
      device.clone(),
      &run_options,
      Some(&optimizer),
    )
    .unwrap();
    let schedule = LinearSchedule::new(run_options.learning_rate, plan.len(), run_options.warmup_ratio);
    for (index, batch) in plan.iter().enumerate().skip(4) {
      let texts = match batch {
        Batch::Pairs(indexes) => TextBatch::Pairs(indexes.iter().map(|index| &examples.pairs[*index]).collect()),
        Batch::Groups(indexes) => TextBatch::Groups(indexes.iter().map(|index| &examples.groups[*index]).collect()),
      };
      let (updated, _) = session.step(encoder, &texts, schedule.rate(index)).unwrap();
      encoder = updated;
    }
    embeddings(&Session::finish(encoder), &tokenizer, &device)
  }

  #[test]
  fn resuming_without_the_recorded_parameter_ids_would_not_be_the_same_run() {
    let base = tempfile::tempdir().unwrap();
    write_tiny_model(base.path(), Some(24));
    let scratch = tempfile::tempdir().unwrap();
    let checkpoint = scratch.path().join("step-4");
    let uninterrupted = run(base.path(), Some((4, &checkpoint)), None, 7).unwrap();

    // The control: the hand-written loop with the ids restored is the uninterrupted run. Without this, the next
    // assertion would also pass if the hand-written loop had merely drifted from `fit`.
    assert_eq!(uninterrupted, continue_by_hand(base.path(), &checkpoint, true));
    // The same loop without the ids: everything loads, every step runs, and the result is a different model.
    assert_ne!(
      uninterrupted,
      continue_by_hand(base.path(), &checkpoint, false),
      "this is the silent failure the recorded ids exist to prevent"
    );
  }

  #[test]
  fn a_checkpoint_only_continues_the_model_loaded_from_it() {
    let base = tempfile::tempdir().unwrap();
    write_tiny_model(base.path(), Some(24));
    let scratch = tempfile::tempdir().unwrap();
    let checkpoint = scratch.path().join("step-4");
    run(base.path(), Some((4, &checkpoint)), None, 7).unwrap();

    // The base model's weights with the checkpoint's optimizer state and step count would pass every other check.
    let device = Device::flex();
    let encoder = SentenceEncoder::load(base.path(), &device).unwrap();
    let tokenizer = TextTokenizer::load(base.path(), MAX_TOKENS).unwrap();
    let examples = Examples {
      pairs: pairs(),
      groups: groups(),
    };
    let run_options = TrainOptions {
      epochs: 3,
      seed: 7,
      ..options()
    };
    let error = fit(
      encoder,
      tokenizer,
      device,
      &examples,
      &run_options,
      Some(&checkpoint),
      |_| Ok(()),
    )
    .unwrap_err()
    .to_string();
    assert!(
      error.contains("load the encoder from the checkpoint directory"),
      "{error}"
    );
  }

  #[test]
  fn a_checkpoint_is_not_continued_under_another_token_limit() {
    let base = tempfile::tempdir().unwrap();
    write_tiny_model(base.path(), Some(24));
    let scratch = tempfile::tempdir().unwrap();
    let checkpoint = scratch.path().join("step-4");
    run(base.path(), Some((4, &checkpoint)), None, 7).unwrap();

    let device = Device::flex();
    let encoder = SentenceEncoder::load(&checkpoint, &device).unwrap();
    // Every text would be cut shorter than in the steps already taken.
    let tokenizer = TextTokenizer::load(base.path(), 4).unwrap();
    let examples = Examples {
      pairs: pairs(),
      groups: groups(),
    };
    let run_options = TrainOptions {
      epochs: 3,
      seed: 7,
      ..options()
    };
    let error = fit(
      encoder,
      tokenizer,
      device,
      &examples,
      &run_options,
      Some(&checkpoint),
      |_| Ok(()),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("token limit of 16, not 4"), "{error}");
  }

  #[test]
  fn a_loss_that_is_not_finite_stops_the_run_before_the_step_is_taken() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let examples = Examples {
      pairs: Vec::new(),
      groups: groups(),
    };
    // A scale this large turns every score into an infinity, and the loss into NaN.
    let run_options = TrainOptions {
      scale: 1e300,
      ..options()
    };
    let mut steps = 0;
    let error = fit(encoder, tokenizer, device, &examples, &run_options, None, |_| {
      steps += 1;
      Ok(())
    })
    .unwrap_err();
    assert!(
      matches!(
        error,
        TrainError::Diverged {
          step: 1,
          quantity: "loss",
          ..
        }
      ),
      "{error}"
    );
    assert_eq!(
      steps, 0,
      "no step was completed, so nothing could have been checkpointed"
    );
  }

  #[test]
  fn a_gradient_that_is_not_finite_is_reported_and_never_scaled_into_nan() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let (session, encoder) = Session::begin(encoder, tokenizer, device, &options()).unwrap();
    let left = session.encode(&encoder, &["blueberry gummies", "vape"]).unwrap();
    let right = session.encode(&encoder, &["flower resin", "wyld edibles mg"]).unwrap();
    let objective = Objective::Ranking { scale: DEFAULT_SCALE };
    let mut grads = truncated(&left, &right, &[24], &objective).backward();
    // An overflow, simulated: every entry is still finite, but their squares no longer fit in an `f32`.
    encoder.weights().visit(&mut Scale {
      grads: &mut grads,
      factor: 1e30,
    });

    let norm = clip_to_global_norm(&mut grads, encoder.weights(), Some(1.0));
    assert!(norm.is_infinite(), "{norm}");
    // Left exactly as they were. "Clipping" to `1 / inf` would have multiplied every gradient by zero: a step
    // that changes nothing, reported as an ordinary one. `Session::step` refuses the step on the norm instead.
    assert!(global_norm(&grads, encoder.weights()).is_infinite());
  }

  #[test]
  fn a_checkpoint_is_a_complete_model_directory_that_is_never_overwritten() {
    let base = tempfile::tempdir().unwrap();
    write_tiny_model(base.path(), Some(24));
    let scratch = tempfile::tempdir().unwrap();
    let checkpoint = scratch.path().join("step-4");
    run(base.path(), Some((4, &checkpoint)), None, 7).unwrap();

    let state = checkpoint::read(&checkpoint).unwrap();
    assert_eq!((state.step, state.total_steps), (4, 9));
    assert_eq!((state.run.options.seed, state.run.max_tokens), (7, MAX_TOKENS));
    // Loadable as a model on its own, which is how one pass of a run is compared with another.
    let device = Device::flex();
    let encoder = SentenceEncoder::load(&checkpoint, &device).unwrap();
    assert_eq!(encoder.dimension(), 24);
    assert!(TextTokenizer::load(&checkpoint, MAX_TOKENS).is_ok());
    assert!(!scratch.path().join("step-4.partial").exists());

    let error = run(base.path(), Some((4, &checkpoint)), None, 7).unwrap_err();
    assert!(error.to_string().contains("ever overwritten"), "{error}");
  }

  #[test]
  fn a_checkpoint_is_only_resumed_by_the_run_that_wrote_it() {
    let base = tempfile::tempdir().unwrap();
    write_tiny_model(base.path(), Some(24));
    let scratch = tempfile::tempdir().unwrap();
    let checkpoint = scratch.path().join("step-4");
    run(base.path(), Some((4, &checkpoint)), None, 7).unwrap();

    // Another seed is another batch order: the first four steps of that run are not the ones this checkpoint took.
    let other_seed = run(base.path(), None, Some(&checkpoint), 8).unwrap_err().to_string();
    assert!(other_seed.contains("different options"), "{other_seed}");

    let resume_with = |examples: Examples, epochs: usize| {
      let device = Device::flex();
      let encoder = SentenceEncoder::load(&checkpoint, &device).unwrap();
      let tokenizer = TextTokenizer::load(base.path(), MAX_TOKENS).unwrap();
      let run_options = TrainOptions {
        epochs,
        seed: 7,
        ..options()
      };
      fit(
        encoder,
        tokenizer,
        device,
        &examples,
        &run_options,
        Some(&checkpoint),
        |_| Ok(()),
      )
      .unwrap_err()
      .to_string()
    };
    let fewer = Examples {
      pairs: pairs().into_iter().skip(1).collect(),
      groups: groups(),
    };
    assert!(resume_with(fewer, 3).contains("different training data"));
    let same = || Examples {
      pairs: pairs(),
      groups: groups(),
    };
    assert!(resume_with(same(), 5).contains("different options"));
    assert!(read_error(scratch.path()).contains("is not a checkpoint"));
  }

  fn read_error(dir: &Path) -> String {
    checkpoint::read(dir).unwrap_err().to_string()
  }

  #[test]
  fn the_last_step_of_each_pass_is_marked() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let examples = Examples {
      pairs: pairs(),
      groups: groups(),
    };
    let run_options = TrainOptions { epochs: 2, ..options() };
    let mut ends = Vec::new();
    fit(encoder, tokenizer, device, &examples, &run_options, None, |step| {
      if step.progress.epoch_complete {
        ends.push((step.progress.epoch, step.progress.step));
      }
      Ok(())
    })
    .unwrap();
    assert_eq!(ends, [(1, 3), (2, 6)]);
  }

  #[test]
  fn an_error_from_the_step_callback_stops_the_run() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let examples = Examples {
      pairs: pairs(),
      groups: Vec::new(),
    };
    let mut steps = 0;
    let result = fit(encoder, tokenizer, device, &examples, &options(), None, |_| {
      steps += 1;
      Err(TrainError::Backend("disk full".to_owned()))
    });
    assert!(matches!(result, Err(TrainError::Backend(_))));
    assert_eq!(steps, 1);
  }

  #[test]
  fn gradients_above_the_limit_are_scaled_down_to_it_as_a_whole() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let (session, encoder) = Session::begin(encoder, tokenizer, device, &options()).unwrap();
    let gradients = || {
      let left = session.encode(&encoder, &["blueberry gummies", "vape"]).unwrap();
      let right = session.encode(&encoder, &["flower resin", "wyld edibles mg"]).unwrap();
      let objective = Objective::Ranking { scale: DEFAULT_SCALE };
      truncated(&left, &right, &[24], &objective).backward()
    };

    let mut grads = gradients();
    let norm = clip_to_global_norm(&mut grads, encoder.weights(), None);
    assert!(norm > 0.01, "the fixture should produce a real gradient: {norm}");
    assert!(
      (global_norm(&grads, encoder.weights()) - norm).abs() < 1e-6,
      "no limit, no change"
    );

    let limit = norm / 4.0;
    let mut grads = gradients();
    let reported = clip_to_global_norm(&mut grads, encoder.weights(), Some(limit));
    assert!(
      (reported - norm).abs() < norm * 1e-4,
      "the norm before clipping is what is reported"
    );
    let clipped = global_norm(&grads, encoder.weights());
    assert!((clipped - limit).abs() < limit * 1e-3, "{clipped} vs {limit}");

    let mut grads = gradients();
    clip_to_global_norm(&mut grads, encoder.weights(), Some(norm * 4.0));
    assert!(
      (global_norm(&grads, encoder.weights()) - norm).abs() < norm * 1e-4,
      "under the limit, no change"
    );
  }

  #[test]
  fn options_the_model_cannot_be_trained_with_are_refused() {
    let refused = |change: fn(&mut TrainOptions)| {
      let Fixture {
        encoder,
        tokenizer,
        device,
        _dir,
      } = fixture();
      let mut run_options = options();
      change(&mut run_options);
      let examples = Examples {
        pairs: pairs(),
        groups: Vec::new(),
      };
      let error = fit(encoder, tokenizer, device, &examples, &run_options, None, |_| Ok(())).unwrap_err();
      assert!(matches!(error, TrainError::Options(_)), "{error}");
      error.to_string()
    };
    assert!(refused(|options| options.widths = vec![24, 25]).contains("width 25"));
    assert!(refused(|options| options.widths = Vec::new()).contains("no embedding width"));
    assert!(refused(|options| options.epochs = 0).contains("epochs"));
    assert!(refused(|options| options.batch_size = 1).contains("batch_size"));
    assert!(refused(|options| options.learning_rate = 0.0).contains("learning_rate"));
    assert!(refused(|options| options.max_grad_norm = Some(0.0)).contains("max_grad_norm"));
  }

  #[test]
  fn catalog_examples_that_cannot_be_ranked_are_an_error_not_an_untouched_model() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    // Three listings of a single catalog entry: no batch can hold two of them.
    let examples = Examples {
      pairs: Vec::new(),
      groups: [
        "wyld blueberry gummies",
        "blueberry gummies mg",
        "gummies blueberry indica",
      ]
      .into_iter()
      .map(|listing| GroupExample {
        catalog: "blueberry gummies".to_owned(),
        listing: listing.to_owned(),
      })
      .collect(),
    };
    let error = fit(encoder, tokenizer, device, &examples, &options(), None, |_| Ok(())).unwrap_err();
    assert!(
      matches!(&error, TrainError::NoExamples(reason) if reason.contains("same catalog entry")),
      "{error}"
    );
  }

  #[test]
  fn nothing_to_train_on_is_an_error_not_an_untouched_model() {
    let Fixture {
      encoder,
      tokenizer,
      device,
      _dir,
    } = fixture();
    let error = fit(
      encoder,
      tokenizer,
      device,
      &Examples::default(),
      &options(),
      None,
      |_| Ok(()),
    )
    .unwrap_err();
    assert!(matches!(error, TrainError::NoExamples(_)), "{error}");
  }
}
