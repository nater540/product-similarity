//! Strict safetensors loading, and saving in the same layout.
//!
//! The load is deliberately paranoid, because the failure it guards against is invisible: a checkpoint that
//! half-applies leaves randomly initialized layers that still produce unit vectors, cosine similarities, and
//! confident accept/reject verdicts. `allow_partial(false)` only covers the `missing` set, so everything else is
//! audited here:
//!
//! - `errors` is empty: a shape or dtype mismatch means the checkpoint is for a different architecture.
//! - `missing` is empty: every parameter of the module got a value.
//! - `unused` holds only tensors this crate knowingly ignores, see [`IGNORED_PREFIXES`].
//! - `applied.len()` equals the count derived from the config, see [`BertConfig::expected_tensor_count`].
//!
//! Saving ([`save_encoder`], [`save_projection`]) writes Hugging Face tensor names and PyTorch tensor layouts, so a
//! fine-tuned model comes back through exactly the audited path above, and is equally a model PyTorch can open.

use std::path::Path;

use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::DType;
use burn_store::{
  ApplyResult, BurnToPyTorchAdapter, FloatCastAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot,
  PyTorchToBurnAdapter, SafetensorsStore,
};

use super::bert::{BertConfig, BertModel};
use crate::error::{Error, Result};

/// Checkpoint tensors a bare encoder legitimately leaves on the floor.
///
/// `pooler.` is BERT's tanh pooler (sentence encoders pool the token states themselves), `cls.` is the pretraining
/// heads, and `embeddings.position_ids` is a constant buffer older exports persisted. **Anything else** in `unused`
/// means the remapping table failed to find a home for a real encoder weight, and the load is refused.
pub const IGNORED_PREFIXES: [&str; 3] = ["pooler.", "cls.", "embeddings.position_ids"];

/// Hugging Face tensor names to this crate's module tree.
///
/// The tree in [`super::bert`] mirrors HF's on purpose, so only three things differ: some checkpoints nest the
/// encoder under `bert.`, `LayerNorm` is not snake case, and `self` is a Rust keyword.
const ENCODER_REMAPPINGS: [(&str, &str); 3] = [
  (r"^bert\.", ""),
  (r"\.LayerNorm\.", ".layer_norm."),
  (r"\.attention\.self\.", ".attention.self_attention."),
];

/// This crate's module tree back to Hugging Face tensor names: [`ENCODER_REMAPPINGS`] in reverse.
///
/// The optional `bert.` prefix is not restored; sentence-transformers checkpoints do not carry it.
const ENCODER_SAVE_REMAPPINGS: [(&str, &str); 2] = [
  (r"\.layer_norm\.", ".LayerNorm."),
  (r"\.attention\.self_attention\.", ".attention.self."),
];

/// The adapter every checkpoint is read through: PyTorch's layout conventions, then a cast to `f32`.
///
/// The cast matters because loaded parameters keep the checkpoint's dtype, not the module's, and several popular
/// models (`gte-small` among them) are published in half precision. Without it such a model loads cleanly, passes
/// the audit, and then fails inside the backend on the first forward pass, when an `f16` weight meets an `f32`
/// activation.
fn from_checkpoint() -> impl ModuleAdapter + 'static {
  PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32))
}

/// A sentence-transformers `Dense` module: one linear projection applied after pooling.
///
/// The field is named `linear` because that is the key prefix sentence-transformers saves it under.
#[derive(Module, Debug)]
pub struct Projection {
  pub(crate) linear: Linear,
}

/// Loads `weights` into a freshly initialized encoder, refusing anything short of a complete, exact application.
///
/// # Errors
///
/// Returns [`Error::Weights`] naming the tensors involved if any tensor fails to apply, any parameter is left
/// without a value, or any checkpoint tensor outside [`IGNORED_PREFIXES`] goes unused.
pub fn load_encoder(model: &mut BertModel, config: &BertConfig, weights: &Path) -> Result<()> {
  let remapper = KeyRemapper::from_patterns(ENCODER_REMAPPINGS.to_vec())
    .map_err(|source| Error::weights(weights, format!("key-remapping table does not compile: {source}")))?;
  let mut store = SafetensorsStore::from_file(weights)
    .with_from_adapter(from_checkpoint())
    .remap(remapper)
    .allow_partial(false);
  let result = model
    .load_from(&mut store)
    .map_err(|source| Error::weights(weights, source.to_string()))?;
  audit(&result, weights, config.expected_tensor_count(), &IGNORED_PREFIXES)
}

/// Loads a sentence-transformers `Dense` checkpoint (`linear.weight`, optionally `linear.bias`).
///
/// # Errors
///
/// Returns [`Error::Weights`] under the same conditions as [`load_encoder`], with nothing tolerated as unused.
pub fn load_projection(projection: &mut Projection, has_bias: bool, weights: &Path) -> Result<()> {
  let mut store = SafetensorsStore::from_file(weights)
    .with_from_adapter(from_checkpoint())
    .allow_partial(false);
  let result = projection
    .load_from(&mut store)
    .map_err(|source| Error::weights(weights, source.to_string()))?;
  audit(&result, weights, if has_bias { 2 } else { 1 }, &[])
}

/// Writes the encoder to `weights` under Hugging Face tensor names, refusing to replace an existing file.
///
/// The pooler and pretraining heads a published checkpoint may carry are not part of [`BertModel`] and are not
/// written. Nothing in a sentence-embedding pipeline reads them.
///
/// # Errors
///
/// Returns [`Error::Weights`] if the file exists already or cannot be written.
pub fn save_encoder(model: &BertModel, weights: &Path) -> Result<()> {
  let mut store = SafetensorsStore::from_file(weights).with_to_adapter(BurnToPyTorchAdapter);
  for (pattern, replacement) in ENCODER_SAVE_REMAPPINGS {
    store = store.with_key_remapping(pattern, replacement);
  }
  // `valid` strips gradient tracking, so a model that has just been trained is stored as plain tensors.
  model
    .valid()
    .save_into(&mut store)
    .map_err(|source| Error::weights(weights, source.to_string()))
}

/// Writes a sentence-transformers `Dense` checkpoint (`linear.weight`, optionally `linear.bias`).
///
/// # Errors
///
/// Returns [`Error::Weights`] if the file exists already or cannot be written.
pub fn save_projection(projection: &Projection, weights: &Path) -> Result<()> {
  let mut store = SafetensorsStore::from_file(weights).with_to_adapter(BurnToPyTorchAdapter);
  projection
    .valid()
    .save_into(&mut store)
    .map_err(|source| Error::weights(weights, source.to_string()))
}

fn audit(result: &ApplyResult, path: &Path, expected_applied: usize, ignored_prefixes: &[&str]) -> Result<()> {
  if !result.errors.is_empty() {
    let detail: Vec<String> = result.errors.iter().map(ToString::to_string).collect();
    return Err(Error::weights(
      path,
      format!("{} tensor(s) failed to apply: {}", detail.len(), detail.join("; ")),
    ));
  }

  if !result.missing.is_empty() {
    let names: Vec<&str> = result.missing.iter().map(|(name, _)| name.as_str()).collect();
    return Err(Error::weights(
      path,
      format!(
        "{} parameter(s) got no value and would stay randomly initialized: {}",
        names.len(),
        names.join(", ")
      ),
    ));
  }

  let stray: Vec<&str> = result
    .unused
    .iter()
    .map(String::as_str)
    .filter(|name| !ignored_prefixes.iter().any(|prefix| name.starts_with(prefix)))
    .collect();
  if !stray.is_empty() {
    return Err(Error::weights(
      path,
      format!(
        "{} checkpoint tensor(s) found no home in the module, so this is not a plain BERT encoder: {}",
        stray.len(),
        stray.join(", ")
      ),
    ));
  }

  if result.applied.len() != expected_applied {
    return Err(Error::weights(
      path,
      format!(
        "applied {} tensors but the architecture accounts for {expected_applied}",
        result.applied.len()
      ),
    ));
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use burn::nn::LinearConfig;
  use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
  use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore};

  use super::*;
  use crate::model::bert::testing::tiny_config;

  /// Saves `model` the way PyTorch would name it, so loading exercises the real remapping and adapter path.
  fn save_as_hugging_face(model: &BertModel, path: &Path, prefix: &str) {
    let mut store = SafetensorsStore::from_file(path)
      .with_to_adapter(BurnToPyTorchAdapter)
      .with_key_remapping(r"\.layer_norm\.", ".LayerNorm.")
      .with_key_remapping(r"\.attention\.self_attention\.", ".attention.self.")
      .with_key_remapping(r"^", prefix);
    model.save_into(&mut store).unwrap();
  }

  /// The encoder's output on a fixed input: equal outputs mean equal weights for every layer that matters.
  fn output_of(model: &BertModel, device: &Device) -> Vec<f32> {
    let ids = Tensor::<2, Int>::from_data(TensorData::new(vec![1_i64, 5, 9, 3], [1, 4]), device);
    let padding = Tensor::<2, Bool>::from_data(TensorData::new(vec![false; 4], [1, 4]), device);
    model.forward(ids, padding).into_data().try_into_vec::<f32>().unwrap()
  }

  #[test]
  fn a_hugging_face_checkpoint_round_trips_into_the_encoder() {
    let device = Device::flex();
    let config = tiny_config();
    let source = config.init(&device);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    save_as_hugging_face(&source, &path, "");

    let mut target = config.init(&device);
    assert_ne!(output_of(&source, &device), output_of(&target, &device));
    load_encoder(&mut target, &config, &path).unwrap();
    assert_eq!(output_of(&source, &device), output_of(&target, &device));
  }

  #[test]
  fn a_saved_encoder_loads_back_identically_and_is_never_overwritten() {
    let device = Device::flex();
    let config = tiny_config();
    let source = config.init(&device);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    save_encoder(&source, &path).unwrap();

    let mut target = config.init(&device);
    load_encoder(&mut target, &config, &path).unwrap();
    assert_eq!(output_of(&source, &device), output_of(&target, &device));

    let error = save_encoder(&target, &path).unwrap_err();
    assert!(matches!(error, Error::Weights { .. }), "{error}");
  }

  #[test]
  fn a_saved_projection_loads_back_identically() {
    let device = Device::flex();
    let build = |bias: bool| Projection {
      linear: LinearConfig::new(4, 6).with_bias(bias).init(&device),
    };
    let input = Tensor::<2>::from_data(TensorData::new(vec![0.5_f32, -1.0, 2.0, 0.25], [1, 4]), &device);
    let output = |projection: &Projection| {
      let projected = projection.linear.forward(input.clone());
      projected.into_data().try_into_vec::<f32>().unwrap()
    };

    for bias in [false, true] {
      let source = build(bias);
      let dir = tempfile::tempdir().unwrap();
      let path = dir.path().join("model.safetensors");
      save_projection(&source, &path).unwrap();
      let mut target = build(bias);
      load_projection(&mut target, bias, &path).unwrap();
      assert_eq!(output(&source), output(&target), "bias: {bias}");
    }
  }

  #[test]
  fn a_half_precision_checkpoint_is_cast_up_and_runs() {
    let device = Device::flex();
    let config = tiny_config();
    let source = config.init(&device);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let mut store = SafetensorsStore::from_file(&path)
      .with_to_adapter(BurnToPyTorchAdapter.chain(FloatCastAdapter::to(DType::F16)))
      .with_key_remapping(r"\.layer_norm\.", ".LayerNorm.")
      .with_key_remapping(r"\.attention\.self_attention\.", ".attention.self.");
    source.save_into(&mut store).unwrap();

    let mut target = config.init(&device);
    load_encoder(&mut target, &config, &path).unwrap();
    // The forward pass is the assertion that matters: with f16 parameters left in place it panics in the backend.
    let (expected, found) = (output_of(&source, &device), output_of(&target, &device));
    for (a, b) in expected.iter().zip(&found) {
      assert!(
        (a - b).abs() < 0.05,
        "{a} vs {b}: only half-precision rounding should separate them"
      );
    }
  }

  #[test]
  fn a_bert_prefixed_checkpoint_loads_too() {
    let device = Device::flex();
    let config = tiny_config();
    let source = config.init(&device);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    save_as_hugging_face(&source, &path, "bert.");

    let mut target = config.init(&device);
    load_encoder(&mut target, &config, &path).unwrap();
    assert_eq!(output_of(&source, &device), output_of(&target, &device));
  }

  #[test]
  fn a_checkpoint_for_a_deeper_model_is_refused() {
    let device = Device::flex();
    let config = tiny_config();
    let mut deeper = config.clone();
    deeper.num_hidden_layers = 3;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    save_as_hugging_face(&deeper.init(&device), &path, "");

    let error = load_encoder(&mut config.init(&device), &config, &path).unwrap_err();
    assert!(error.to_string().contains("encoder.layer.2"), "{error}");
  }

  #[test]
  fn a_checkpoint_for_a_shallower_model_is_refused() {
    let device = Device::flex();
    let config = tiny_config();
    let mut shallower = config.clone();
    shallower.num_hidden_layers = 1;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    save_as_hugging_face(&shallower.init(&device), &path, "");

    assert!(load_encoder(&mut config.init(&device), &config, &path).is_err());
  }
}
