//! The sentence-transformers pipeline around a BERT encoder.
//!
//! A sentence-transformers model directory is a `modules.json` listing stages, each in its own subdirectory:
//!
//! ```text
//! modules.json                 [Transformer, Pooling, (Dense), (Normalize)]
//! config.json                  encoder architecture
//! model.safetensors            encoder weights
//! tokenizer.json
//! 1_Pooling/config.json        which pooling mode
//! 2_Dense/config.json          optional projection after pooling
//! 2_Dense/model.safetensors
//! ```
//!
//! Reading the pipeline from the directory, rather than hard-coding it per model, is what lets the same binary serve
//! `mdbr-leaf-mt` (mean pooling plus a 384 to 1024 projection), `gte-small` and `all-MiniLM-L6-v2` (mean pooling),
//! or `bge-small-en-v1.5` (CLS pooling) by pointing it at a different directory. Anything in the pipeline this
//! module does not implement is an error, never a silent skip: a skipped stage would still produce vectors, just
//! the wrong ones.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::nn::LinearConfig;
use burn::tensor::{Bool, Device, Int, Tensor};
use serde::Deserialize;

use super::bert::{BertConfig, BertModel, TrainingDropout};
use super::loader::{Projection, load_encoder, load_projection, save_encoder, save_projection};
use crate::error::{Error, Result};

/// How token states are reduced to one vector per text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
  /// The mean of the non-padding token states.
  Mean,
  /// The state of the first (`[CLS]`) token.
  Cls,
}

#[derive(Debug, Deserialize)]
struct ModuleEntry {
  path: String,
  #[serde(rename = "type")]
  kind: String,
}

/// The file listing a pipeline's stages, relative to the model directory.
pub const MODULES_FILE: &str = "modules.json";

/// The encoder's architecture file, relative to the model directory.
const ENCODER_CONFIG: &str = "config.json";

/// The stages of a sentence-transformers pipeline, checked against what this crate implements.
///
/// This is the one definition of "a model this project can run", as far as `modules.json` can tell. The loader
/// builds the encoder from it and `model fetch` derives its download list from it, so the two cannot disagree
/// about which pipelines are supported or which files they consist of.
///
/// Accepted: a `Transformer` at the directory root, then one `Pooling`, then optionally one `Dense`, then
/// optionally a `Normalize`. Anything else is refused, never skipped: a skipped stage would still produce vectors,
/// just the wrong ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pipeline {
  pooling_dir: String,
  dense_dir: Option<String>,
}

impl Pipeline {
  /// Parses the contents of a `modules.json`. `origin` is only used to name the file in errors.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Json`] if `modules_json` is malformed and [`Error::UnsupportedModel`] if the pipeline has a
  /// stage this crate does not implement, has stages in an order it does not implement, or keeps a stage anywhere
  /// but in a subdirectory of the model directory.
  pub fn parse(modules_json: &[u8], origin: &Path) -> Result<Self> {
    let modules: Vec<ModuleEntry> =
      serde_json::from_slice(modules_json).map_err(|source| Error::json(origin, source))?;
    let unsupported = |reason: String| Error::unsupported(origin, reason);

    let mut stages = modules.iter();
    match stages.next() {
      Some(first) if first.kind.ends_with(".Transformer") && first.path.is_empty() => {}
      _ => {
        return Err(unsupported(
          "the first module must be a `Transformer` stored at the directory root".to_owned(),
        ));
      }
    }

    let mut pooling_dir = None;
    let mut dense_dir = None;
    let mut normalized = false;
    for stage in stages {
      let kind = stage.kind.rsplit('.').next().unwrap_or(stage.kind.as_str());
      match kind {
        // `Normalize` is only a no-op as the very last stage (every embedding leaves this crate L2-normalized
        // regardless). Followed by anything, it changes what that stage computes.
        _ if normalized => {
          return Err(unsupported(format!(
            "module `{kind}` at `{}` comes after `Normalize`, which is only implemented as the last stage",
            stage.path
          )));
        }
        "Pooling" if pooling_dir.is_none() && dense_dir.is_none() => {
          pooling_dir = Some(stage_dir(&stage.path, origin)?);
        }
        "Dense" if pooling_dir.is_some() && dense_dir.is_none() => {
          dense_dir = Some(stage_dir(&stage.path, origin)?);
        }
        "Normalize" if pooling_dir.is_some() => normalized = true,
        other => {
          return Err(unsupported(format!(
            "module `{other}` at `{}` is not implemented, or is out of order",
            stage.path
          )));
        }
      }
    }

    let Some(pooling_dir) = pooling_dir else {
      return Err(unsupported("the pipeline has no `Pooling` module".to_owned()));
    };
    Ok(Self { pooling_dir, dense_dir })
  }

  /// Every file this crate reads to run the pipeline, relative to the model directory, `/`-separated.
  pub fn files(&self) -> Vec<String> {
    let mut files = vec![
      MODULES_FILE.to_owned(),
      ENCODER_CONFIG.to_owned(),
      crate::tokenize::TOKENIZER_FILE.to_owned(),
      ENCODER_WEIGHTS.to_owned(),
      format!("{}/config.json", self.pooling_dir),
    ];
    if let Some(dense_dir) = &self.dense_dir {
      files.push(format!("{dense_dir}/config.json"));
      files.push(format!("{dense_dir}/{ENCODER_WEIGHTS}"));
    }
    files
  }
}

/// Checks that a stage's `path` names a subdirectory of the model directory, and returns it.
///
/// `modules.json` comes from wherever the model came from, and the path is joined onto the model directory for
/// reading, for downloading into, and for saving. Without this check `../..` or an absolute path would take all
/// three outside it.
fn stage_dir(path: &str, origin: &Path) -> Result<String> {
  let inside = !path.is_empty()
    && !path.contains('\\')
    && Path::new(path)
      .components()
      .all(|component| matches!(component, std::path::Component::Normal(_)));
  if inside {
    Ok(path.trim_end_matches('/').to_owned())
  } else {
    Err(Error::unsupported(
      origin,
      format!("module path `{path}` is not a subdirectory of the model directory"),
    ))
  }
}

/// `1_Pooling/config.json`, in either of the two formats sentence-transformers has written.
///
/// Releases up to 5.x store one boolean per mode (`pooling_mode_mean_tokens: true`). Later releases store a single
/// `pooling_mode` holding a mode name or a list of them. A directory fetched from the Hub usually has the first
/// form and one saved by a current release (after fine-tuning, say) has the second, so both must load.
#[derive(Debug, Default, Deserialize)]
#[expect(
  clippy::struct_excessive_bools,
  clippy::struct_field_names,
  reason = "mirrors the on-disk format sentence-transformers writes"
)]
struct PoolingConfig {
  #[serde(default)]
  pooling_mode: Option<PoolingModes>,
  #[serde(default)]
  pooling_mode_cls_token: bool,
  #[serde(default)]
  pooling_mode_mean_tokens: bool,
  #[serde(default)]
  pooling_mode_max_tokens: bool,
  #[serde(default)]
  pooling_mode_mean_sqrt_len_tokens: bool,
  #[serde(default)]
  pooling_mode_weightedmean_tokens: bool,
  #[serde(default)]
  pooling_mode_lasttoken: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PoolingModes {
  One(String),
  Many(Vec<String>),
}

impl PoolingConfig {
  /// The enabled modes, by their current names, whichever format the file used.
  fn modes(&self) -> Vec<&str> {
    match &self.pooling_mode {
      Some(PoolingModes::One(mode)) => vec![mode.as_str()],
      Some(PoolingModes::Many(modes)) => modes.iter().map(String::as_str).collect(),
      None => [
        ("cls", self.pooling_mode_cls_token),
        ("max", self.pooling_mode_max_tokens),
        ("mean", self.pooling_mode_mean_tokens),
        ("mean_sqrt_len_tokens", self.pooling_mode_mean_sqrt_len_tokens),
        ("weightedmean", self.pooling_mode_weightedmean_tokens),
        ("lasttoken", self.pooling_mode_lasttoken),
      ]
      .into_iter()
      .filter_map(|(mode, enabled)| enabled.then_some(mode))
      .collect(),
    }
  }
}

#[derive(Debug, Deserialize)]
struct DenseConfig {
  in_features: usize,
  out_features: usize,
  #[serde(default = "default_true")]
  bias: bool,
  #[serde(default)]
  activation_function: Option<String>,
}

fn default_true() -> bool {
  true
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
  let bytes = std::fs::read(path).map_err(|source| Error::io(path, source))?;
  serde_json::from_slice(&bytes).map_err(|source| Error::json(path, source))
}

/// The files of a sentence-transformers model directory that determine its output.
///
/// Exposed so a caller can fingerprint exactly the bytes the encoder was built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFiles {
  /// The model directory.
  pub dir: PathBuf,
  /// Every file read while building the pipeline, in a stable order.
  pub read: Vec<PathBuf>,
}

impl ModelFiles {
  /// The same files under another directory: where a copy of the model has been, or is being, written.
  #[must_use]
  pub fn rebased(&self, dir: &Path) -> Self {
    Self {
      dir: dir.to_path_buf(),
      read: self
        .read
        .iter()
        .map(|path| {
          path
            .strip_prefix(&self.dir)
            .map_or_else(|_| path.clone(), |relative| dir.join(relative))
        })
        .collect(),
    }
  }

  /// A digest of the model's content: the first 16 hex digits of the SHA-256 over every file the encoder was
  /// built from and the tokenizer.
  ///
  /// This is the part of a model fingerprint that says *which model*; [`crate::Embedder`] adds the output width
  /// and the token limit to it. Only the files that determine the output are hashed, so a note left beside them
  /// (a README, a `provenance.json`) does not make it a different model.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Io`] if a file cannot be read, or changes size while it is being read.
  pub fn digest(&self) -> Result<String> {
    use sha2::{Digest as _, Sha256};

    let tokenizer = self.dir.join(crate::tokenize::TOKENIZER_FILE);
    let mut hasher = Sha256::new();
    for path in self.read.iter().chain(std::iter::once(&tokenizer)) {
      // Streamed: the checkpoint is tens of megabytes.
      let mut file = std::fs::File::open(path).map_err(|source| Error::io(path, source))?;
      let length = file.metadata().map_err(|source| Error::io(path, source))?.len();
      // Length-prefix each file so that moving bytes between two files cannot collide.
      hasher.update(length.to_le_bytes());
      let streamed = std::io::copy(&mut file, &mut hasher).map_err(|source| Error::io(path, source))?;
      if streamed != length {
        return Err(Error::io(
          path,
          std::io::Error::other("the file changed size while it was being read"),
        ));
      }
    }
    Ok(crate::record::hex_prefix(&hasher.finalize(), 8))
  }
}

/// Every learned tensor of the pipeline as one module, so that an optimizer can update all of them in one step.
#[derive(Module, Debug)]
pub struct EncoderWeights {
  bert: BertModel,
  projection: Option<Projection>,
}

/// Files that sit next to the weights and configuration of a model and that other tools expect to find there.
///
/// This crate reads only `tokenizer.json` from this list; the rest is carried along by [`SentenceEncoder::save`]
/// so that a saved model is still a complete sentence-transformers directory.
const SIDE_FILES: [&str; 6] = [
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "vocab.txt",
  "config_sentence_transformers.json",
  "sentence_bert_config.json",
];

/// The encoder's checkpoint, relative to the model directory.
const ENCODER_WEIGHTS: &str = "model.safetensors";

/// A loaded sentence encoder: token ids in, one un-normalized vector per text out.
#[derive(Debug)]
pub struct SentenceEncoder {
  weights: EncoderWeights,
  pooling: Pooling,
  config: BertConfig,
  dimension: usize,
  files: ModelFiles,
}

impl SentenceEncoder {
  /// Loads the pipeline described by `dir/modules.json` onto `device`.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Io`] / [`Error::Json`] for unreadable or malformed files, [`Error::UnsupportedModel`] if the
  /// pipeline contains a stage, pooling mode, or activation this crate does not implement, and [`Error::Weights`]
  /// if a checkpoint does not apply completely.
  pub fn load(dir: &Path, device: &Device) -> Result<Self> {
    let modules_path = dir.join(MODULES_FILE);
    let modules = std::fs::read(&modules_path).map_err(|source| Error::io(&modules_path, source))?;
    let pipeline = Pipeline::parse(&modules, &modules_path)?;
    let mut read = vec![modules_path];

    let config_path = dir.join(ENCODER_CONFIG);
    let weights_path = dir.join(ENCODER_WEIGHTS);
    let config = BertConfig::from_file(&config_path)?;
    let mut bert = config.init(device);
    load_encoder(&mut bert, &config, &weights_path)?;
    read.extend([config_path, weights_path]);

    let pooling_path = dir.join(&pipeline.pooling_dir).join("config.json");
    let pooling = pooling_mode(&read_json(&pooling_path)?, &pooling_path)?;
    read.push(pooling_path);

    let mut projection = None;
    let mut dimension = config.hidden_size;
    if let Some(dense_dir) = &pipeline.dense_dir {
      let path = dir.join(dense_dir).join("config.json");
      let dense: DenseConfig = read_json(&path)?;
      if dense.in_features != dimension {
        return Err(Error::unsupported(
          &path,
          format!(
            "`in_features` is {} but the encoder produces {dimension}",
            dense.in_features
          ),
        ));
      }
      if let Some(activation) = dense.activation_function.as_deref()
        && !activation.ends_with(".Identity")
      {
        return Err(Error::unsupported(
          &path,
          format!("activation `{activation}` is not implemented; only `Identity` is"),
        ));
      }
      let mut loaded = Projection {
        linear: LinearConfig::new(dense.in_features, dense.out_features)
          .with_bias(dense.bias)
          .init(device),
      };
      let weights = dir.join(dense_dir).join(ENCODER_WEIGHTS);
      load_projection(&mut loaded, dense.bias, &weights)?;
      dimension = dense.out_features;
      projection = Some(loaded);
      read.extend([path, weights]);
    }

    Ok(Self {
      weights: EncoderWeights { bert, projection },
      pooling,
      config,
      dimension,
      files: ModelFiles {
        dir: dir.to_path_buf(),
        read,
      },
    })
  }

  /// Builds an encoder from already-constructed parts, for tests that must not touch the filesystem.
  #[cfg(any(test, feature = "testing"))]
  pub(crate) fn from_parts(bert: BertModel, config: BertConfig, pooling: Pooling) -> Self {
    let dimension = config.hidden_size;
    Self {
      weights: EncoderWeights { bert, projection: None },
      pooling,
      config,
      dimension,
      files: ModelFiles {
        dir: PathBuf::new(),
        read: Vec::new(),
      },
    }
  }

  /// Encodes a padded batch.
  ///
  /// `input_ids` is `[batch, seq]`; `padding` is `[batch, seq]` with `true` at padded positions. Returns
  /// `[batch, dimension]`, **not** normalized.
  pub fn forward(&self, input_ids: Tensor<2, Int>, padding: Tensor<2, Bool>) -> Tensor<2> {
    let hidden = self.weights.bert.forward(input_ids, padding.clone());
    self.pool(hidden, padding)
  }

  /// Encodes a padded batch as [`Self::forward`] does, with `dropout` applied inside the encoder.
  ///
  /// The output carries gradients when the weights do, which is the case after
  /// `map_weights(|weights| weights.train())` in a build of Burn with automatic differentiation.
  pub fn forward_training(
    &self,
    input_ids: Tensor<2, Int>,
    padding: Tensor<2, Bool>,
    dropout: &TrainingDropout,
  ) -> Tensor<2> {
    let hidden = self.weights.bert.forward_training(input_ids, padding.clone(), dropout);
    self.pool(hidden, padding)
  }

  fn pool(&self, hidden: Tensor<3>, padding: Tensor<2, Bool>) -> Tensor<2> {
    let [batch, _, _] = hidden.dims();
    let width = self.config.hidden_size;

    let pooled = match self.pooling {
      Pooling::Cls => hidden.slice_dim(1, 0..1).reshape([batch, width]),
      Pooling::Mean => {
        let keep = padding.bool_not().float().unsqueeze_dim::<3>(2);
        let total = hidden.mul(keep.clone()).sum_dim(1).reshape([batch, width]);
        // A row with no real tokens cannot happen (the tokenizer always emits [CLS] and [SEP]), but dividing by
        // zero would poison the whole batch with NaN, so the count is clamped anyway.
        let count = keep.sum_dim(1).reshape([batch, 1]).clamp_min(1.0);
        total.div(count)
      }
    };

    match &self.weights.projection {
      Some(projection) => projection.linear.forward(pooled),
      None => pooled,
    }
  }

  /// The dropout this model's architecture is trained with, for [`Self::forward_training`].
  pub fn dropout(&self) -> TrainingDropout {
    self.config.dropout()
  }

  /// The learned tensors.
  pub fn weights(&self) -> &EncoderWeights {
    &self.weights
  }

  /// Replaces the learned tensors with `update` of them, which is how an optimizer step is applied.
  #[must_use]
  pub fn map_weights(self, update: impl FnOnce(EncoderWeights) -> EncoderWeights) -> Self {
    Self {
      weights: update(self.weights),
      ..self
    }
  }

  /// Writes this encoder to `out` as a sentence-transformers directory that [`Self::load`] reads back.
  ///
  /// The weights are written from memory. Every configuration file that was read at load time, and the tokenizer
  /// files next to them, are copied unchanged from the directory this encoder was loaded from. `out` must be empty
  /// or absent: a model directory is never modified in place, because its content hash is what identifies the
  /// vectors computed with it.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Io`] if `out` is not empty or a file cannot be copied, and [`Error::Weights`] if a
  /// checkpoint cannot be written.
  pub fn save(&self, out: &Path) -> Result<()> {
    let occupied = std::fs::read_dir(out).is_ok_and(|mut entries| entries.next().is_some());
    if occupied {
      return Err(Error::io(
        out,
        std::io::Error::new(
          std::io::ErrorKind::AlreadyExists,
          "refusing to write a model into a directory that is not empty",
        ),
      ));
    }

    let source = &self.files.dir;
    let side_files = SIDE_FILES
      .iter()
      .map(|name| source.join(name))
      .filter(|path| path.is_file());
    for path in self.files.read.iter().cloned().chain(side_files) {
      let Ok(relative) = path.strip_prefix(source) else {
        continue;
      };
      let target = out.join(relative);
      if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|error| Error::io(parent, error))?;
      }
      if relative.extension().is_none_or(|extension| extension != "safetensors") {
        std::fs::copy(&path, &target).map_err(|error| Error::io(&target, error))?;
      } else if relative == Path::new(ENCODER_WEIGHTS) {
        save_encoder(&self.weights.bert, &target)?;
        make_readable(&target)?;
      } else if let Some(projection) = &self.weights.projection {
        // The only other checkpoint a supported pipeline has is its single `Dense` stage.
        save_projection(projection, &target)?;
        make_readable(&target)?;
      }
    }
    Ok(())
  }

  /// The width of the vectors [`Self::forward`] returns.
  pub fn dimension(&self) -> usize {
    self.dimension
  }

  /// The longest sequence, in tokens, the position table supports.
  pub fn max_positions(&self) -> usize {
    self.config.max_position_embeddings
  }

  /// The pooling mode in use.
  pub fn pooling(&self) -> Pooling {
    self.pooling
  }

  /// The files this encoder was built from.
  pub fn files(&self) -> &ModelFiles {
    &self.files
  }
}

/// Gives a freshly written checkpoint the permissions of an ordinary file.
///
/// The store writes through a private temporary file, which leaves the result readable by its owner only. A model
/// directory is routinely copied into a container image and read there by a different, unprivileged user, and a
/// model that loads on the machine that trained it and nowhere else is a bad surprise to leave for deployment.
#[cfg(unix)]
fn make_readable(path: &Path) -> Result<()> {
  use std::os::unix::fs::PermissionsExt;

  std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644)).map_err(|error| Error::io(path, error))
}

#[cfg(not(unix))]
#[expect(
  clippy::unnecessary_wraps,
  reason = "must match the signature of the Unix build, which can fail"
)]
fn make_readable(_path: &Path) -> Result<()> {
  Ok(())
}

fn pooling_mode(config: &PoolingConfig, path: &Path) -> Result<Pooling> {
  match config.modes().as_slice() {
    ["mean"] => Ok(Pooling::Mean),
    ["cls"] => Ok(Pooling::Cls),
    [other] => Err(Error::unsupported(
      path,
      format!("pooling mode `{other}` is not implemented; only `mean` and `cls` are"),
    )),
    [] => Err(Error::unsupported(path, "no pooling mode is enabled")),
    several => Err(Error::unsupported(
      path,
      format!(
        "{} pooling modes are enabled ({}); concatenated pooling is not implemented",
        several.len(),
        several.join(", ")
      ),
    )),
  }
}

#[cfg(test)]
mod tests {
  use burn::tensor::TensorData;

  use super::*;
  use crate::model::bert::testing::tiny_config;

  fn encoder(pooling: Pooling, device: &Device) -> SentenceEncoder {
    let config = tiny_config();
    SentenceEncoder::from_parts(config.init(device), config, pooling)
  }

  fn batch(ids: Vec<i64>, padding: Vec<bool>, shape: [usize; 2], device: &Device) -> (Tensor<2, Int>, Tensor<2, Bool>) {
    (
      Tensor::from_data(TensorData::new(ids, shape), device),
      Tensor::from_data(TensorData::new(padding, shape), device),
    )
  }

  fn rows(output: Tensor<2>) -> Vec<Vec<f32>> {
    let [_, width] = output.dims();
    let flat = output.into_data().try_into_vec::<f32>().unwrap();
    flat.chunks(width).map(<[f32]>::to_vec).collect()
  }

  fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
      assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
  }

  #[test]
  fn mean_pooling_ignores_padding() {
    let device = Device::flex();
    let encoder = encoder(Pooling::Mean, &device);

    let (ids, padding) = batch(vec![1, 2, 3], vec![false; 3], [1, 3], &device);
    let alone = rows(encoder.forward(ids, padding));

    let (ids, padding) = batch(
      vec![1, 2, 3, 0, 0, 4, 5, 6, 7, 8],
      vec![false, false, false, true, true, false, false, false, false, false],
      [2, 5],
      &device,
    );
    let padded = rows(encoder.forward(ids, padding));

    assert_eq!(padded.len(), 2);
    assert_close(&alone[0], &padded[0]);
  }

  #[test]
  fn cls_pooling_returns_the_first_token_state() {
    let device = Device::flex();
    let encoder = encoder(Pooling::Cls, &device);
    let (ids, padding) = batch(vec![1, 2, 3, 4], vec![false; 4], [1, 4], &device);
    let hidden = encoder.weights.bert.forward(ids.clone(), padding.clone());
    let first = hidden.slice_dim(1, 0..1).into_data().try_into_vec::<f32>().unwrap();
    assert_close(&rows(encoder.forward(ids, padding))[0], &first);
  }

  fn parsed(json: &str) -> PoolingConfig {
    serde_json::from_str(json).unwrap()
  }

  fn pooling_from(json: &str) -> Result<Pooling> {
    pooling_mode(&parsed(json), Path::new("1_Pooling/config.json"))
  }

  #[test]
  fn the_legacy_boolean_pooling_config_is_understood() {
    let mean = r#"{"word_embedding_dimension": 384, "pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true,
                   "pooling_mode_max_tokens": false, "pooling_mode_mean_sqrt_len_tokens": false}"#;
    assert_eq!(pooling_from(mean).unwrap(), Pooling::Mean);
    let cls = r#"{"word_embedding_dimension": 384, "pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}"#;
    assert_eq!(pooling_from(cls).unwrap(), Pooling::Cls);
  }

  #[test]
  fn the_current_pooling_config_is_understood() {
    // What sentence-transformers 5.7 writes when it saves a model, for example after fine-tuning.
    let mean = r#"{"embedding_dimension": 384, "pooling_mode": "mean", "include_prompt": true}"#;
    assert_eq!(pooling_from(mean).unwrap(), Pooling::Mean);
    assert_eq!(pooling_from(r#"{"pooling_mode": "cls"}"#).unwrap(), Pooling::Cls);
    assert_eq!(pooling_from(r#"{"pooling_mode": ["mean"]}"#).unwrap(), Pooling::Mean);
  }

  #[test]
  fn an_unimplemented_missing_or_combined_pooling_mode_is_refused() {
    let named = |json: &str| pooling_from(json).unwrap_err().to_string();
    assert!(named(r#"{"pooling_mode_max_tokens": true}"#).contains("`max`"));
    assert!(named(r#"{"pooling_mode": "lasttoken"}"#).contains("`lasttoken`"));
    assert!(named(r#"{"embedding_dimension": 384}"#).contains("no pooling mode"));
    assert!(named(r#"{"pooling_mode": ["mean", "cls"]}"#).contains("concatenated"));
    assert!(named(r#"{"pooling_mode_mean_tokens": true, "pooling_mode_cls_token": true}"#).contains("concatenated"));
  }

  #[test]
  fn a_saved_encoder_is_a_complete_directory_that_loads_back_identically() {
    let device = Device::flex();
    let base = tempfile::tempdir().unwrap();
    crate::testing::write_tiny_model(base.path(), Some(24));
    let encoder = SentenceEncoder::load(base.path(), &device).unwrap();
    assert_eq!(encoder.dimension(), 24);

    let out = tempfile::tempdir().unwrap();
    let saved = out.path().join("copy");
    encoder.save(&saved).unwrap();
    for file in [
      "modules.json",
      "config.json",
      "tokenizer.json",
      "model.safetensors",
      "1_Pooling/config.json",
      "2_Dense/config.json",
      "2_Dense/model.safetensors",
    ] {
      assert!(saved.join(file).is_file(), "{file} was not written");
    }

    #[cfg(unix)]
    for weights in ["model.safetensors", "2_Dense/model.safetensors"] {
      use std::os::unix::fs::PermissionsExt;
      let mode = std::fs::metadata(saved.join(weights)).unwrap().permissions().mode();
      assert_eq!(
        mode & 0o777,
        0o644,
        "{weights} must be readable by the user a container runs as"
      );
    }

    let reloaded = SentenceEncoder::load(&saved, &device).unwrap();
    let input = || {
      batch(
        vec![2, 4, 5, 3, 2, 6, 3, 0],
        vec![false, false, false, false, false, false, false, true],
        [2, 4],
        &device,
      )
    };
    let (ids, padding) = input();
    let expected = rows(encoder.forward(ids, padding));
    let (ids, padding) = input();
    assert_eq!(expected, rows(reloaded.forward(ids, padding)));
  }

  #[test]
  fn saving_into_a_directory_that_holds_anything_is_refused() {
    let device = Device::flex();
    let base = tempfile::tempdir().unwrap();
    crate::testing::write_tiny_model(base.path(), None);
    let encoder = SentenceEncoder::load(base.path(), &device).unwrap();

    // Its own source directory is the case that matters: overwriting it would destroy the base model.
    let error = encoder.save(base.path()).unwrap_err();
    assert!(error.to_string().contains("not empty"), "{error}");
  }

  #[test]
  fn training_dropout_does_nothing_to_a_model_that_tracks_no_gradients() {
    let device = Device::flex();
    let encoder = encoder(Pooling::Mean, &device);
    let input = || batch(vec![1, 2, 3, 4], vec![false; 4], [1, 4], &device);
    let (ids, padding) = input();
    let plain = rows(encoder.forward(ids, padding));
    let (ids, padding) = input();
    let regularized = rows(encoder.forward_training(ids, padding, &encoder.dropout()));
    assert!(encoder.dropout().is_enabled());
    assert_eq!(plain, regularized);
  }

  fn pipeline(stages: &[(&str, &str)]) -> Result<Pipeline> {
    let modules: Vec<serde_json::Value> = stages
      .iter()
      .map(|(kind, path)| serde_json::json!({"type": format!("sentence_transformers.models.{kind}"), "path": path}))
      .collect();
    let json = serde_json::Value::Array(modules).to_string();
    Pipeline::parse(json.as_bytes(), Path::new("modules.json"))
  }

  #[test]
  fn the_files_of_a_pipeline_follow_its_stages() {
    let plain = pipeline(&[
      ("Transformer", ""),
      ("Pooling", "1_Pooling"),
      ("Normalize", "2_Normalize"),
    ])
    .unwrap();
    assert_eq!(
      plain.files(),
      [
        "modules.json",
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "1_Pooling/config.json"
      ]
    );
    let projected = pipeline(&[("Transformer", ""), ("Pooling", "1_Pooling"), ("Dense", "2_Dense")]).unwrap();
    assert_eq!(
      projected.files()[5..],
      ["2_Dense/config.json", "2_Dense/model.safetensors"]
    );
  }

  #[test]
  fn a_pipeline_this_crate_does_not_implement_is_refused_not_approximated() {
    let refused = |stages: &[(&str, &str)]| pipeline(stages).unwrap_err().to_string();
    assert!(refused(&[("Pooling", "1_Pooling")]).contains("first module"));
    assert!(refused(&[("Transformer", "0_Transformer"), ("Pooling", "1_Pooling")]).contains("first module"));
    assert!(refused(&[("Transformer", "")]).contains("no `Pooling`"));
    assert!(refused(&[("Transformer", ""), ("Dense", "2_Dense"), ("Pooling", "1_Pooling")]).contains("out of order"));
    assert!(refused(&[("Transformer", ""), ("Pooling", "1"), ("Pooling", "2")]).contains("out of order"));
    assert!(refused(&[("Transformer", ""), ("Pooling", "1"), ("Dense", "2"), ("Dense", "3")]).contains("out of order"));
    assert!(refused(&[("Transformer", ""), ("Pooling", "1"), ("LSTM", "2_LSTM")]).contains("`LSTM`"));
    assert!(Pipeline::parse(b"not json", Path::new("modules.json")).is_err());
  }

  #[test]
  fn a_projection_after_normalize_is_refused_because_it_would_compute_something_else() {
    // PyTorch would compute W(x / |x|) + b here; skipping the normalization would silently compute Wx + b.
    let stages = [
      ("Transformer", ""),
      ("Pooling", "1_Pooling"),
      ("Normalize", "2_Normalize"),
      ("Dense", "3_Dense"),
    ];
    let error = pipeline(&stages).unwrap_err().to_string();
    assert!(error.contains("comes after `Normalize`"), "{error}");
  }

  #[test]
  fn a_stage_outside_the_model_directory_is_refused() {
    for path in ["../other", "1_Pooling/../../x", "/etc", "", ".", "a\\..\\b"] {
      let error = pipeline(&[("Transformer", ""), ("Pooling", path)])
        .unwrap_err()
        .to_string();
      assert!(error.contains("not a subdirectory"), "{path:?}: {error}");
      let error = pipeline(&[("Transformer", ""), ("Pooling", "1_Pooling"), ("Dense", path)]).unwrap_err();
      assert!(error.to_string().contains("not a subdirectory"), "{path:?}: {error}");
    }
    assert!(pipeline(&[("Transformer", ""), ("Pooling", "stages/pooling")]).is_ok());
  }

  #[test]
  fn a_directory_without_modules_json_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let error = SentenceEncoder::load(dir.path(), &Device::flex()).unwrap_err();
    assert!(error.to_string().contains("modules.json"), "{error}");
  }
}
