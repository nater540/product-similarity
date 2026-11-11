//! Checkpoints: a training run written to disk part-way, to resume after an interruption or to evaluate.
//!
//! A checkpoint is a directory holding:
//!
//! - **a complete model directory**, exactly what the run would have produced had it stopped there. It can be
//!   served, calibrated, or compared like any other model, which is how the best pass of a long run is picked.
//! - **`optimizer.bpk`**: the optimizer's state. For `AdamW` that is two running moments per parameter; a run
//!   resumed without them takes its first steps at full size on estimates built from a single batch.
//! - **`training_state.json`**: how far the run had got, what identifies the run (its options, a fingerprint of
//!   its data, its token limit), and the id of every parameter.
//!
//! The last item needs explaining. Burn's optimizer files its state under each parameter's id, and ids are made up
//! afresh every time a module is built, including when a checkpoint is loaded. Without restoring the ids, a resumed
//! run would load the optimizer state successfully and then never find any of it, and train on as if it had none.
//! So the ids are recorded here in traversal order and put back before the optimizer is loaded.
//!
//! A checkpoint is only resumed by the run that wrote it ([`RunIdentity`]). Anything else would continue on a
//! different batch order, learning-rate schedule, or token limit while looking like a normal run.
//!
//! Everything this module writes appears whole or not at all: it is written to a sibling directory, flushed to
//! disk, and renamed into place. A directory with a checkpoint's name is therefore a complete checkpoint, after a
//! killed process and after a machine that lost power alike.

use std::path::{Path, PathBuf};

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use similarity_core::model::{EncoderWeights, SentenceEncoder};
use similarity_core::record::hex_prefix;

use crate::error::{Result, TrainError};
use crate::trainer::{Examples, Session, TrainOptions};

/// The file whose presence marks a directory as a complete checkpoint.
pub const STATE_FILE: &str = "training_state.json";

const OPTIMIZER_FILE: &str = "optimizer.bpk";

/// Bumped when the meaning of [`RunState`] changes, so that an old checkpoint is refused instead of misread.
const FORMAT: u32 = 1;

/// What makes two invocations the same training run.
///
/// Everything that decides which texts each step sees, in what form, and what is done with them. Two runs that
/// agree on all of it take identical steps (up to dropout), so one can continue where the other stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunIdentity {
  /// The options the run was started with.
  pub options: TrainOptions,
  /// A fingerprint of the examples the run was started with, see [`data_digest`].
  pub data: String,
  /// The token limit texts are truncated to. Not a training option, but it changes what every batch contains.
  pub max_tokens: usize,
}

impl RunIdentity {
  /// The identity of a run over `examples` with `options` and a tokenizer truncating to `max_tokens`.
  pub fn of(options: &TrainOptions, examples: &Examples, max_tokens: usize) -> Self {
    Self {
      options: options.clone(),
      data: data_digest(examples),
      max_tokens,
    }
  }

  /// What differs from `other`, in words, or `None` if they are the same run.
  fn difference(&self, other: &Self) -> Option<String> {
    if self.options != other.options {
      Some(format!("different options ({:?})", self.options))
    } else if self.data != other.data {
      Some("different training data".to_owned())
    } else if self.max_tokens != other.max_tokens {
      Some(format!(
        "a token limit of {}, not {}",
        self.max_tokens, other.max_tokens
      ))
    } else {
      None
    }
  }
}

/// What a checkpoint records about the run it belongs to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunState {
  format: u32,
  /// How many optimizer steps had been taken.
  pub step: usize,
  /// How many steps the whole run has.
  pub total_steps: usize,
  /// Which run this is.
  pub run: RunIdentity,
  /// The id of every parameter, in the order a module traversal visits them. Written as decimal strings: the ids
  /// are 64-bit, and a JSON number above 2^53 does not survive the many tools that read numbers as doubles.
  param_ids: Vec<String>,
}

/// The one field read before anything else, so that a checkpoint from another format is reported as that and not
/// as whichever field happens to be missing.
#[derive(Deserialize)]
struct Header {
  format: u32,
}

fn failure(path: &Path, reason: impl Into<String>) -> TrainError {
  TrainError::Checkpoint {
    path: path.to_path_buf(),
    reason: reason.into(),
  }
}

/// A fingerprint of the training examples: the same examples in the same order give the same value.
///
/// Order matters because the batches of a run are drawn from the examples by position.
pub fn data_digest(examples: &Examples) -> String {
  let mut hasher = Sha256::new();
  // Every text is length-prefixed, so that no two different lists of examples can produce the same byte stream.
  let text = |hasher: &mut Sha256, value: &str| {
    hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(value.as_bytes());
  };
  hasher.update(u64::try_from(examples.pairs.len()).unwrap_or(u64::MAX).to_le_bytes());
  for pair in &examples.pairs {
    text(&mut hasher, &pair.before);
    text(&mut hasher, &pair.after);
    hasher.update([u8::from(pair.same)]);
  }
  hasher.update(u64::try_from(examples.groups.len()).unwrap_or(u64::MAX).to_le_bytes());
  for group in &examples.groups {
    text(&mut hasher, &group.catalog);
    text(&mut hasher, &group.listing);
  }
  hex_prefix(&hasher.finalize(), 16)
}

/// `path` with `suffix` appended to its last component: `models/ft` and `.partial` give `models/ft.partial`.
///
/// A trailing separator is dropped first. Appending to the raw text would turn `models/ft/` into
/// `models/ft/.partial`, a directory *inside* the one it is meant to sit beside.
pub fn sibling(path: &Path, suffix: &str) -> PathBuf {
  let mut name = path.components().collect::<PathBuf>().into_os_string();
  name.push(suffix);
  PathBuf::from(name)
}

/// Flushes every file under `dir`, and `dir` itself, to disk.
fn sync_tree(dir: &Path) -> std::io::Result<()> {
  for entry in std::fs::read_dir(dir)? {
    let path = entry?.path();
    if path.is_dir() {
      sync_tree(&path)?;
    } else {
      std::fs::File::open(&path)?.sync_all()?;
    }
  }
  sync_dir(dir)
}

/// Flushes a directory's own entries (the names in it) to disk.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> std::io::Result<()> {
  std::fs::File::open(dir)?.sync_all()
}

/// Directories cannot be opened for syncing on this platform; the rename itself is as durable as it gets.
#[cfg(not(unix))]
#[expect(
  clippy::unnecessary_wraps,
  reason = "must match the signature of the Unix build, which can fail"
)]
fn sync_dir(_dir: &Path) -> std::io::Result<()> {
  Ok(())
}

/// Creates the directory `dir` with whatever `fill` writes into it, whole or not at all.
///
/// `fill` writes into a sibling directory, which is flushed to disk and only then renamed to `dir`. `dir` must
/// not exist, or be an empty directory.
fn write_whole(dir: &Path, fill: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
  let dir = &dir.components().collect::<PathBuf>();
  let occupied = std::fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_some());
  if occupied || dir.is_file() {
    return Err(failure(dir, "already exists; nothing is ever overwritten"));
  }
  let partial = sibling(dir, ".partial");
  let io = |path: &Path| {
    let path = path.to_path_buf();
    move |source: std::io::Error| failure(&path, source.to_string())
  };
  // Only ever the leftovers of this function being interrupted: nothing else creates this name.
  if partial.exists() {
    std::fs::remove_dir_all(&partial).map_err(io(&partial))?;
  }
  if let Some(parent) = partial.parent().filter(|parent| !parent.as_os_str().is_empty()) {
    std::fs::create_dir_all(parent).map_err(io(parent))?;
  }

  fill(&partial)?;
  sync_tree(&partial).map_err(io(&partial))?;
  // An empty directory under the final name is in the way of the rename on some platforms, and holds nothing.
  if dir.is_dir() {
    std::fs::remove_dir(dir).map_err(io(dir))?;
  }
  std::fs::rename(&partial, dir).map_err(io(dir))?;
  // The rename is an edit to the parent directory, so that is what makes it durable.
  if let Some(parent) = dir.parent().filter(|parent| !parent.as_os_str().is_empty()) {
    sync_dir(parent).map_err(io(parent))?;
  }
  Ok(())
}

/// Writes `encoder` to `out` as a model directory, whole or not at all.
///
/// # Errors
///
/// Returns [`TrainError::Checkpoint`] if `out` already holds something or cannot be written, and
/// [`TrainError::Model`] if the encoder cannot be saved.
pub fn save_model(encoder: &SentenceEncoder, out: &Path) -> Result<()> {
  write_whole(out, |partial| Ok(encoder.save(partial)?))
}

/// Collects the id of every float parameter, in traversal order.
#[derive(Default)]
struct CollectIds(Vec<String>);

impl ModuleVisitor for CollectIds {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
    self.0.push(param.id.val().to_string());
  }
}

/// Gives every float parameter the next recorded id, in traversal order.
struct AssignIds<'a> {
  ids: std::slice::Iter<'a, u64>,
  missing: usize,
}

impl ModuleMapper for AssignIds<'_> {
  fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
    let Some(id) = self.ids.next() else {
      self.missing = self.missing.saturating_add(1);
      return param;
    };
    let (_, tensor, mapper) = param.consume();
    Param::from_mapped_value(ParamId::from(*id), tensor, mapper)
  }
}

fn param_ids(weights: &EncoderWeights) -> Vec<String> {
  let mut ids = CollectIds::default();
  weights.visit(&mut ids);
  ids.0
}

/// Where a checkpoint keeps its optimizer state.
pub(crate) fn optimizer_file(dir: &Path) -> PathBuf {
  dir.join(OPTIMIZER_FILE)
}

/// Writes a checkpoint to `dir`, which must not exist.
pub(crate) fn write(
  dir: &Path,
  encoder: &SentenceEncoder,
  session: &Session,
  step: usize,
  total_steps: usize,
  run: &RunIdentity,
) -> Result<()> {
  write_whole(dir, |partial| {
    encoder.save(partial)?;
    session.save_optimizer(&optimizer_file(partial))?;
    let state = RunState {
      format: FORMAT,
      step,
      total_steps,
      run: run.clone(),
      param_ids: param_ids(encoder.weights()),
    };
    let state_path = partial.join(STATE_FILE);
    let json = serde_json::to_vec_pretty(&state).map_err(|source| failure(&state_path, source.to_string()))?;
    std::fs::write(&state_path, json).map_err(|source| failure(&state_path, source.to_string()))
  })
}

/// Reads what a checkpoint records about its run.
///
/// # Errors
///
/// Returns [`TrainError::Checkpoint`] if `dir` is not a complete checkpoint, or was written in a format this
/// version does not read.
pub fn read(dir: &Path) -> Result<RunState> {
  let path = dir.join(STATE_FILE);
  let bytes = std::fs::read(&path)
    .map_err(|source| failure(dir, format!("is not a checkpoint: cannot read {STATE_FILE} ({source})")))?;
  // The format first, on its own: a different format usually means different fields, and "missing field" would
  // then be reported in place of the actual reason.
  let header: Header = serde_json::from_slice(&bytes).map_err(|source| failure(&path, source.to_string()))?;
  if header.format != FORMAT {
    return Err(failure(
      dir,
      format!(
        "was written in checkpoint format {}, and this version reads format {FORMAT}",
        header.format
      ),
    ));
  }
  serde_json::from_slice(&bytes).map_err(|source| failure(&path, source.to_string()))
}

/// Whether `state` was written by the run `run`.
pub fn same_run(state: &RunState, run: &RunIdentity) -> bool {
  state.run.difference(run).is_none()
}

/// Whether two paths name the same directory, following symbolic links where both exist.
fn same_directory(left: &Path, right: &Path) -> bool {
  match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
    (Ok(left), Ok(right)) => left == right,
    _ => left.components().eq(right.components()),
  }
}

/// Prepares `encoder` to continue the run checkpointed at `dir`.
///
/// Checks that the encoder was loaded from that very directory and that the run being started is the run the
/// checkpoint belongs to, then gives the encoder's parameters the ids the checkpoint's optimizer state is filed
/// under. Returns the encoder and the number of steps already taken.
pub(crate) fn restore(
  dir: &Path,
  encoder: SentenceEncoder,
  run: &RunIdentity,
  total_steps: usize,
) -> Result<(SentenceEncoder, usize)> {
  let state = read(dir)?;
  // Otherwise the checkpoint's optimizer state and step count would be attached to some other model's weights,
  // and every check below would still pass.
  if !same_directory(&encoder.files().dir, dir) {
    return Err(failure(
      dir,
      format!(
        "cannot continue a model loaded from {}; load the encoder from the checkpoint directory itself",
        encoder.files().dir.display()
      ),
    ));
  }
  let different = |what: &str| {
    Err(failure(
      dir,
      format!("belongs to a run with {what}; resume with the options and data it was started with"),
    ))
  };
  if let Some(what) = state.run.difference(run) {
    return different(&what);
  }
  if state.total_steps != total_steps || state.step > total_steps {
    return different(&format!("{} steps, not {total_steps}", state.total_steps));
  }
  if state.step == total_steps {
    return Err(failure(dir, "is the end of its run; there is nothing left to train"));
  }

  let ids = state
    .param_ids
    .iter()
    .map(|id| id.parse::<u64>())
    .collect::<std::result::Result<Vec<_>, _>>()
    .map_err(|source| failure(dir, format!("records a parameter id that is not a number ({source})")))?;
  let mut assign = AssignIds {
    ids: ids.iter(),
    missing: 0,
  };
  let encoder = encoder.map_weights(|weights| weights.map(&mut assign));
  if assign.missing > 0 || assign.ids.next().is_some() {
    return Err(failure(
      dir,
      format!("records {} parameters, which is not how many this model has", ids.len()),
    ));
  }
  Ok((encoder, state.step))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::data::{GroupExample, PairExample};

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

  #[test]
  fn the_data_fingerprint_changes_with_any_text_label_or_order() {
    let base = examples();
    let digest = data_digest(&base);
    assert_eq!(digest, data_digest(&examples()));
    assert_eq!(digest.len(), 32);

    let mut relabeled = examples();
    relabeled.pairs[0].same = false;
    let mut reworded = examples();
    reworded.groups[0].listing.push('s');
    let mut reordered = examples();
    reordered.pairs.push(reordered.pairs[0].clone());
    reordered.pairs[0].after = "other".to_owned();
    // The same bytes split differently between two fields must not collide either.
    let mut shifted = examples();
    shifted.pairs[0].before = "blueberry".to_owned();
    shifted.pairs[0].after = " gummiesblueberry gummies mg".to_owned();
    for changed in [relabeled, reworded, reordered, shifted, Examples::default()] {
      assert_ne!(digest, data_digest(&changed));
    }
  }

  #[test]
  fn a_directory_that_is_not_a_checkpoint_is_refused_with_a_reason() {
    let dir = tempfile::tempdir().unwrap();
    let error = read(dir.path()).unwrap_err().to_string();
    assert!(error.contains("is not a checkpoint"), "{error}");
  }

  #[test]
  fn a_checkpoint_in_another_format_says_so_whatever_its_fields_are() {
    let dir = tempfile::tempdir().unwrap();
    // A future format with none of today's fields: the version is still what gets reported.
    std::fs::write(dir.path().join(STATE_FILE), br#"{"format": 2, "cursor": {"pass": 3}}"#).unwrap();
    let error = read(dir.path()).unwrap_err().to_string();
    assert!(
      error.contains("checkpoint format 2") && error.contains("reads format 1"),
      "{error}"
    );

    // The current format with a field missing is a damaged file, reported as such.
    std::fs::write(dir.path().join(STATE_FILE), br#"{"format": 1, "step": 4}"#).unwrap();
    let error = read(dir.path()).unwrap_err().to_string();
    assert!(error.contains("missing field"), "{error}");
  }

  #[test]
  fn a_sibling_sits_beside_a_directory_however_the_directory_was_spelled() {
    assert_eq!(
      sibling(Path::new("models/ft"), ".partial"),
      Path::new("models/ft.partial")
    );
    assert_eq!(
      sibling(Path::new("models/ft/"), ".partial"),
      Path::new("models/ft.partial")
    );
    assert_eq!(sibling(Path::new("ft"), ".checkpoints"), Path::new("ft.checkpoints"));
    assert_eq!(
      sibling(Path::new("/abs/ft//"), ".checkpoints"),
      Path::new("/abs/ft.checkpoints")
    );
  }

  #[test]
  fn a_directory_is_written_whole_and_only_where_nothing_is() {
    let scratch = tempfile::tempdir().unwrap();
    // Spelled with a trailing separator, as a shell's tab completion would leave it.
    let target = scratch.path().join("out/");
    std::fs::create_dir(scratch.path().join("out")).unwrap();
    write_whole(&target, |partial| {
      assert_eq!(partial, scratch.path().join("out.partial"));
      std::fs::create_dir_all(partial.join("nested")).unwrap();
      std::fs::write(partial.join("nested/file"), b"x").unwrap();
      Ok(())
    })
    .unwrap();
    assert!(scratch.path().join("out/nested/file").is_file());
    assert!(!scratch.path().join("out.partial").exists());

    let again = write_whole(&target, |_| panic!("must not be asked to write over something"));
    assert!(again.unwrap_err().to_string().contains("ever overwritten"));

    // A failure part-way leaves nothing under the final name, and the leftovers do not block the next attempt.
    let failing = scratch.path().join("second");
    let error = write_whole(&failing, |partial| {
      std::fs::create_dir_all(partial).unwrap();
      Err(TrainError::Backend("disk full".to_owned()))
    });
    assert!(error.is_err() && !failing.exists());
    write_whole(&failing, |partial| {
      assert!(!partial.exists(), "the earlier attempt's leftovers were cleared");
      std::fs::create_dir_all(partial).map_err(|source| failure(partial, source.to_string()))
    })
    .unwrap();
    assert!(failing.is_dir());
  }

  #[test]
  fn the_same_run_is_recognized_and_any_difference_is_named() {
    let options = TrainOptions::default();
    let run = RunIdentity::of(&options, &examples(), 256);
    assert_eq!(run.difference(&RunIdentity::of(&options, &examples(), 256)), None);

    let shorter = RunIdentity::of(&options, &examples(), 32);
    assert!(shorter.difference(&run).unwrap().contains("token limit of 32, not 256"));
    let other_data = RunIdentity::of(&options, &Examples::default(), 256);
    assert_eq!(other_data.difference(&run).unwrap(), "different training data");
    let other_options = TrainOptions {
      seed: 9,
      ..TrainOptions::default()
    };
    let reseeded = RunIdentity::of(&other_options, &examples(), 256);
    assert!(reseeded.difference(&run).unwrap().contains("different options"));
  }

  #[test]
  fn options_read_back_from_a_state_file_exactly_as_they_were_written() {
    // Values chosen because the shortest text that identifies them does not parse back to the same bits unless
    // the JSON parser is asked to be exact.
    let options = TrainOptions {
      learning_rate: 1e-4 / 7.0,
      warmup_ratio: 0.1 + 0.2,
      margin: 1.0 / 3.0,
      scale: 20.0 / 3.0,
      max_grad_norm: Some(2.0_f64.sqrt()),
      weight_decay: 0.1,
      ..TrainOptions::default()
    };
    let run = RunIdentity::of(&options, &examples(), 256);
    let state = RunState {
      format: FORMAT,
      step: 4,
      total_steps: 9,
      run: run.clone(),
      param_ids: vec![u64::MAX.to_string()],
    };
    let read_back: RunState = serde_json::from_slice(&serde_json::to_vec_pretty(&state).unwrap()).unwrap();
    assert_eq!(read_back.run.difference(&run), None);
    assert_eq!(read_back, state);
    assert_eq!(read_back.param_ids[0].parse::<u64>().unwrap(), u64::MAX);
  }
}
