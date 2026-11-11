//! The training loop: tokenize a batch, encode both sides, take the loss, step the optimizer.
//!
//! [`fit`] is the whole procedure. [`Session`] is one layer down, a single optimizer step at a time, and exists so
//! that tests can drive fixed batches at a fixed learning rate and compare the outcome with PyTorch's.
//!
//! The recipe is the one sentence-embedding models are conventionally fine-tuned with: `AdamW` with a linear
//! warm-up and decay ([`crate::schedule`]), gradients clipped to a global norm of 1, and the encoder's own dropout
//! active. Every one of those is an option, because a parity test needs them off and a small dataset may want them
//! different.

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Gradients, Tensor, TensorData};
use similarity_core::model::{EncoderWeights, SentenceEncoder, TrainingDropout};
use similarity_core::tokenize::pad_batch;
use similarity_core::{Device, TextTokenizer};

use crate::data::{self, Batch, GroupExample, PairExample, Rng};
use crate::error::{Result, TrainError};
use crate::loss::{DEFAULT_MARGIN, DEFAULT_SCALE, Objective, truncated};
use crate::schedule::LinearSchedule;

/// Everything that shapes a training run.
#[derive(Debug, Clone, PartialEq)]
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
  if let Some(max_norm) = max_norm
    && norm > max_norm
  {
    // The small constant mirrors `torch.nn.utils.clip_grad_norm_`, so that both produce the same update.
    let factor = max_norm / (norm + 1e-6);
    weights.visit(&mut Scale { grads, factor });
  }
  norm
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
  /// Returns [`TrainError::Model`] if a text cannot be tokenized and [`TrainError::Diverged`] if the loss is not
  /// finite. The encoder is consumed either way; a failed run has no model worth keeping.
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
        loss: value,
      });
    }

    let mut grads = loss.backward();
    let grad_norm = clip_to_global_norm(&mut grads, encoder.weights(), self.max_grad_norm);
    let grads = GradientsParams::from_grads(grads, encoder.weights());
    let optimizer = &mut self.optimizer;
    let encoder = encoder.map_weights(|weights| optimizer.step(rate, weights, grads));
    Ok((encoder, StepReport { loss: value, grad_norm }))
  }
}

/// Fine-tunes `encoder` on `examples` and returns it with plain weights.
///
/// `progress` is called after every step. The batch order of every pass is fixed before the first step, so the
/// number of steps, and with it the learning-rate schedule, is known exactly.
///
/// # Errors
///
/// Returns [`TrainError::NoExamples`] if there is nothing to train on, [`TrainError::Options`] if the options do
/// not fit the model, and whatever [`Session::step`] returns.
pub fn fit(
  encoder: SentenceEncoder,
  tokenizer: TextTokenizer,
  device: Device,
  examples: &Examples,
  options: &TrainOptions,
  mut progress: impl FnMut(&Progress),
) -> Result<SentenceEncoder> {
  if examples.pairs.is_empty() && examples.groups.is_empty() {
    return Err(TrainError::NoExamples(
      "no field of any pair or catalog group has two different texts to compare".to_owned(),
    ));
  }
  let (mut session, mut encoder) = Session::begin(encoder, tokenizer, device, options)?;

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

  let mut step = 0_usize;
  for (epoch, batches) in plan.iter().enumerate() {
    for batch in batches {
      let texts = match batch {
        Batch::Pairs(indexes) => {
          TextBatch::Pairs(indexes.iter().filter_map(|index| examples.pairs.get(*index)).collect())
        }
        Batch::Groups(indexes) => {
          TextBatch::Groups(indexes.iter().filter_map(|index| examples.groups.get(*index)).collect())
        }
      };
      let rate = schedule.rate(step);
      let (updated, report) = session.step(encoder, &texts, rate)?;
      encoder = updated;
      step = step.saturating_add(1);
      progress(&Progress {
        step,
        total_steps,
        epoch: epoch.saturating_add(1),
        examples: batch.len(),
        ranking: matches!(batch, Batch::Groups(_)),
        rate,
        report,
      });
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
      |progress| {
        losses.push(progress.report.loss);
      },
    )
    .unwrap();
    let after = separation(&trained, &tokenizer, &device, &examples.pairs, &[8], 4).unwrap()[0];

    assert_eq!(losses.len(), 60, "8 examples in batches of 4, for 30 passes");
    let (first, last) = (losses[..4].iter().sum::<f64>(), losses[56..].iter().sum::<f64>());
    assert!(last < first * 0.5, "the loss should fall: {first} -> {last}");
    assert!(
      after.mean_gap().unwrap() > before.mean_gap().unwrap() + 0.2,
      "{before:?} -> {after:?}"
    );
    assert!(after.edits.unwrap().min > after.recycles.unwrap().max, "{after:?}");
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
    fit(encoder, tokenizer, device, &examples, &options(), |progress| {
      assert!(progress.ranking);
      reports.push(*progress);
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
    fit(encoder, tokenizer, device, &examples, &run_options, |progress| {
      seen.push(*progress);
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
        |_| {},
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
      |_| {},
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
      let error = fit(encoder, tokenizer, device, &examples, &run_options, |_| {}).unwrap_err();
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
    let error = fit(encoder, tokenizer, device, &examples, &options(), |_| {}).unwrap_err();
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
    let error = fit(encoder, tokenizer, device, &Examples::default(), &options(), |_| {}).unwrap_err();
    assert!(matches!(error, TrainError::NoExamples(_)), "{error}");
  }
}
