//! A BERT encoder.
//!
//! The module tree deliberately mirrors Hugging Face's `BertModel` state dict (`embeddings.word_embeddings`,
//! `encoder.layer.N.attention.self.query`, ...) so that the key-remapping table in [`super::loader`] stays two
//! entries long and a reader can line a checkpoint's tensor names up against these structs by eye. Burn's stock
//! `TransformerEncoder` would need eleven remappings and hides the attention math that parity debugging needs.
//!
//! There is no dropout in the module tree, so [`BertModel::forward`] cannot produce a non-deterministic embedding
//! whatever state the module is in. Training needs dropout, and gets it by handing a [`TrainingDropout`] to
//! [`BertModel::forward_training`]: the regularizer is an argument of the one call that wants it, never a property
//! of the model that a forgotten mode switch could leave on.

use std::path::Path;

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Bool, Device, Int, Tensor};
use serde::Deserialize;

use crate::error::{Error, Result};

/// The subset of a Hugging Face `config.json` that defines a BERT encoder.
///
/// Unknown keys are ignored (real configs carry dozens that only matter for training), but the keys that would
/// change the forward pass are validated by [`BertConfig::from_file`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BertConfig {
  /// Number of rows in the word-embedding table.
  pub vocab_size: usize,
  /// Width of every hidden state.
  pub hidden_size: usize,
  /// Number of encoder layers.
  pub num_hidden_layers: usize,
  /// Number of attention heads per layer; must divide [`Self::hidden_size`].
  pub num_attention_heads: usize,
  /// Width of the feed-forward inner layer.
  pub intermediate_size: usize,
  /// Longest sequence the learned position table covers.
  pub max_position_embeddings: usize,
  /// Number of segment types (always 2 for BERT; only row 0 is ever used here).
  pub type_vocab_size: usize,
  /// Epsilon inside every layer normalization.
  #[serde(default = "default_layer_norm_eps")]
  pub layer_norm_eps: f64,
  /// Dropout probability on hidden states. Applied only by [`BertModel::forward_training`].
  #[serde(default = "default_dropout")]
  pub hidden_dropout_prob: f64,
  /// Dropout probability on attention weights. Applied only by [`BertModel::forward_training`].
  #[serde(default = "default_dropout")]
  pub attention_probs_dropout_prob: f64,
  #[serde(default)]
  model_type: Option<String>,
  #[serde(default)]
  hidden_act: Option<String>,
  #[serde(default)]
  position_embedding_type: Option<String>,
}

fn default_layer_norm_eps() -> f64 {
  1e-12
}

fn default_dropout() -> f64 {
  0.1
}

/// The dropout a BERT encoder is trained with.
///
/// Deliberately not part of [`BertModel`]: see the module documentation. Even when passed in, it only takes effect
/// on tensors that track gradients, which no tensor does outside a training build.
#[derive(Debug, Clone)]
pub struct TrainingDropout {
  hidden: Dropout,
  attention: Dropout,
}

impl TrainingDropout {
  /// A regularizer that changes nothing, for training runs that must be exactly reproducible.
  pub fn disabled() -> Self {
    Self {
      hidden: DropoutConfig::new(0.0).init(),
      attention: DropoutConfig::new(0.0).init(),
    }
  }

  /// Whether any dropout would be applied.
  pub fn is_enabled(&self) -> bool {
    self.hidden.prob > 0.0 || self.attention.prob > 0.0
  }
}

fn drop_hidden<const D: usize>(dropout: Option<&TrainingDropout>, tensor: Tensor<D>) -> Tensor<D> {
  match dropout {
    Some(dropout) => dropout.hidden.forward(tensor),
    None => tensor,
  }
}

fn drop_attention<const D: usize>(dropout: Option<&TrainingDropout>, tensor: Tensor<D>) -> Tensor<D> {
  match dropout {
    Some(dropout) => dropout.attention.forward(tensor),
    None => tensor,
  }
}

impl BertConfig {
  /// Reads and validates a Hugging Face `config.json`.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Io`] / [`Error::Json`] if the file cannot be read or parsed, and [`Error::UnsupportedModel`]
  /// if it describes anything other than a BERT encoder with GELU activations and absolute position embeddings.
  pub fn from_file(path: &Path) -> Result<Self> {
    let bytes = std::fs::read(path).map_err(|source| Error::io(path, source))?;
    let config: Self = serde_json::from_slice(&bytes).map_err(|source| Error::json(path, source))?;
    config.validate(path)?;
    Ok(config)
  }

  fn validate(&self, path: &Path) -> Result<()> {
    let expect = |key: &str, value: Option<&str>, wanted: &str| match value {
      Some(found) if found != wanted => Err(Error::unsupported(
        path,
        format!("`{key}` is `{found}`, but only `{wanted}` is implemented"),
      )),
      _ => Ok(()),
    };
    expect("model_type", self.model_type.as_deref(), "bert")?;
    expect("hidden_act", self.hidden_act.as_deref(), "gelu")?;
    expect(
      "position_embedding_type",
      self.position_embedding_type.as_deref(),
      "absolute",
    )?;

    let sizes = [
      ("vocab_size", self.vocab_size),
      ("hidden_size", self.hidden_size),
      ("num_hidden_layers", self.num_hidden_layers),
      ("num_attention_heads", self.num_attention_heads),
      ("intermediate_size", self.intermediate_size),
      ("max_position_embeddings", self.max_position_embeddings),
      ("type_vocab_size", self.type_vocab_size),
    ];
    if let Some((key, _)) = sizes.iter().find(|(_, value)| *value == 0) {
      return Err(Error::unsupported(path, format!("`{key}` must be positive")));
    }
    for (key, probability) in [
      ("hidden_dropout_prob", self.hidden_dropout_prob),
      ("attention_probs_dropout_prob", self.attention_probs_dropout_prob),
    ] {
      if !(0.0..1.0).contains(&probability) {
        return Err(Error::unsupported(
          path,
          format!("`{key}` is {probability}, which is not a probability below 1"),
        ));
      }
    }
    if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
      return Err(Error::unsupported(
        path,
        format!(
          "`hidden_size` ({}) is not divisible by `num_attention_heads` ({})",
          self.hidden_size, self.num_attention_heads
        ),
      ));
    }
    Ok(())
  }

  /// How many tensors a checkpoint must supply for this architecture.
  ///
  /// Five for the embeddings (three tables plus a normalization's scale and shift) and sixteen per layer (query,
  /// key, value, attention output, intermediate, and output projections at two tensors each, plus two
  /// normalizations at two each). The loader compares this against what actually landed, which is the only check
  /// that catches a remapping table quietly routing a whole family of tensors nowhere.
  pub fn expected_tensor_count(&self) -> usize {
    self.num_hidden_layers.saturating_mul(16).saturating_add(5)
  }

  /// The dropout this architecture was trained with, for [`BertModel::forward_training`].
  pub fn dropout(&self) -> TrainingDropout {
    // `validate` has already confined both probabilities to [0, 1); the clamp keeps a hand-built config from
    // reaching the panic inside Burn's constructor.
    let layer = |probability: f64| DropoutConfig::new(probability.clamp(0.0, 0.999)).init();
    TrainingDropout {
      hidden: layer(self.hidden_dropout_prob),
      attention: layer(self.attention_probs_dropout_prob),
    }
  }

  /// Builds the encoder with freshly initialized (random) weights on `device`.
  ///
  /// The result is only meaningful after [`super::loader::load_encoder`] has overwritten every parameter; shape
  /// tests are the one legitimate user of an unloaded model.
  pub fn init(&self, device: &Device) -> BertModel {
    let linear = |d_input: usize, d_output: usize| LinearConfig::new(d_input, d_output).init(device);
    let norm = || {
      LayerNormConfig::new(self.hidden_size)
        .with_epsilon(self.layer_norm_eps)
        .init(device)
    };
    let hidden = self.hidden_size;

    let embeddings = BertEmbeddings {
      word_embeddings: EmbeddingConfig::new(self.vocab_size, hidden).init(device),
      position_embeddings: EmbeddingConfig::new(self.max_position_embeddings, hidden).init(device),
      token_type_embeddings: EmbeddingConfig::new(self.type_vocab_size, hidden).init(device),
      layer_norm: norm(),
    };
    let layer = (0..self.num_hidden_layers)
      .map(|_| BertLayer {
        attention: BertAttention {
          self_attention: BertSelfAttention {
            query: linear(hidden, hidden),
            key: linear(hidden, hidden),
            value: linear(hidden, hidden),
            heads: self.num_attention_heads,
          },
          output: BertOutput {
            dense: linear(hidden, hidden),
            layer_norm: norm(),
          },
        },
        intermediate: BertIntermediate {
          dense: linear(hidden, self.intermediate_size),
        },
        output: BertOutput {
          dense: linear(self.intermediate_size, hidden),
          layer_norm: norm(),
        },
      })
      .collect();

    BertModel {
      embeddings,
      encoder: BertEncoder { layer },
    }
  }
}

/// Token, position, and segment embeddings, summed and normalized.
#[derive(Module, Debug)]
pub struct BertEmbeddings {
  word_embeddings: Embedding,
  position_embeddings: Embedding,
  token_type_embeddings: Embedding,
  layer_norm: LayerNorm,
}

impl BertEmbeddings {
  fn forward(&self, input_ids: Tensor<2, Int>, dropout: Option<&TrainingDropout>) -> Tensor<3> {
    let [batch, seq_len] = input_ids.dims();
    let device = input_ids.device();

    let positions = Tensor::<1, Int>::arange(0..i64::try_from(seq_len).unwrap_or(i64::MAX), &device)
      .unsqueeze_dim::<2>(0)
      .expand([batch, seq_len]);
    // Every input is a single segment, so the segment id is zero everywhere.
    let token_types = input_ids.zeros_like();

    let summed = self
      .word_embeddings
      .forward(input_ids)
      .add(self.position_embeddings.forward(positions))
      .add(self.token_type_embeddings.forward(token_types));
    drop_hidden(dropout, self.layer_norm.forward(summed))
  }
}

/// Multi-head scaled dot-product self-attention (HF's `BertSelfAttention`).
#[derive(Module, Debug)]
pub struct BertSelfAttention {
  query: Linear,
  key: Linear,
  value: Linear,
  heads: usize,
}

impl BertSelfAttention {
  /// `padding` is `[batch, 1, 1, seq]` with `true` at padded key positions.
  fn forward(&self, hidden: Tensor<3>, padding: Tensor<4, Bool>, dropout: Option<&TrainingDropout>) -> Tensor<3> {
    let [batch, seq_len, width] = hidden.dims();
    let head_width = width.checked_div(self.heads).unwrap_or(width);
    let split = |projected: Tensor<3>| {
      projected
        .reshape([batch, seq_len, self.heads, head_width])
        .swap_dims(1, 2)
    };

    let query = split(self.query.forward(hidden.clone()));
    let key = split(self.key.forward(hidden.clone()));
    let value = split(self.value.forward(hidden));

    #[expect(
      clippy::cast_precision_loss,
      reason = "head widths are tiny; the conversion is exact"
    )]
    let scale = (head_width as f64).sqrt();
    let scores = query.matmul(key.swap_dims(2, 3)).div_scalar(scale);
    // A large finite negative rather than -inf: a fully padded row would otherwise softmax to NaN.
    let scores = scores.mask_fill(padding.expand([batch, self.heads, seq_len, seq_len]), -1.0e9);
    let weights = drop_attention(dropout, softmax(scores, 3));

    weights.matmul(value).swap_dims(1, 2).reshape([batch, seq_len, width])
  }
}

/// A projection followed by a residual connection and layer normalization.
///
/// HF has two classes with this shape (`BertSelfOutput` after attention, `BertOutput` after the feed-forward);
/// they differ only in input width, so one struct serves both.
#[derive(Module, Debug)]
pub struct BertOutput {
  dense: Linear,
  layer_norm: LayerNorm,
}

impl BertOutput {
  fn forward(&self, hidden: Tensor<3>, residual: Tensor<3>, dropout: Option<&TrainingDropout>) -> Tensor<3> {
    let projected = drop_hidden(dropout, self.dense.forward(hidden));
    self.layer_norm.forward(projected.add(residual))
  }
}

/// Self-attention plus its output block (HF's `BertAttention`).
#[derive(Module, Debug)]
pub struct BertAttention {
  // HF calls this field `self`, which Rust reserves; the loader remaps `attention.self.` to this name.
  self_attention: BertSelfAttention,
  output: BertOutput,
}

/// The feed-forward expansion (HF's `BertIntermediate`).
#[derive(Module, Debug)]
pub struct BertIntermediate {
  dense: Linear,
}

/// One encoder block.
#[derive(Module, Debug)]
pub struct BertLayer {
  attention: BertAttention,
  intermediate: BertIntermediate,
  output: BertOutput,
}

impl BertLayer {
  fn forward(&self, hidden: Tensor<3>, padding: Tensor<4, Bool>, dropout: Option<&TrainingDropout>) -> Tensor<3> {
    let attended = self.attention.self_attention.forward(hidden.clone(), padding, dropout);
    let attended = self.attention.output.forward(attended, hidden, dropout);
    let expanded = gelu(self.intermediate.dense.forward(attended.clone()));
    self.output.forward(expanded, attended, dropout)
  }
}

/// The stack of encoder blocks.
#[derive(Module, Debug)]
pub struct BertEncoder {
  layer: Vec<BertLayer>,
}

/// A BERT encoder without the pooler or any pretraining head.
#[derive(Module, Debug)]
pub struct BertModel {
  embeddings: BertEmbeddings,
  encoder: BertEncoder,
}

impl BertModel {
  /// Runs the encoder.
  ///
  /// `input_ids` is `[batch, seq]`; `padding` is `[batch, seq]` with `true` where a position is padding. Returns
  /// the last hidden state, `[batch, seq, hidden]`. Padded positions hold garbage and must be masked by the
  /// caller's pooling.
  pub fn forward(&self, input_ids: Tensor<2, Int>, padding: Tensor<2, Bool>) -> Tensor<3> {
    self.run(input_ids, padding, None)
  }

  /// Runs the encoder as [`Self::forward`] does, with `dropout` applied where BERT applies it: to the embeddings,
  /// to the attention weights, and to each sub-layer's output before its residual connection.
  pub fn forward_training(
    &self,
    input_ids: Tensor<2, Int>,
    padding: Tensor<2, Bool>,
    dropout: &TrainingDropout,
  ) -> Tensor<3> {
    self.run(input_ids, padding, Some(dropout))
  }

  fn run(&self, input_ids: Tensor<2, Int>, padding: Tensor<2, Bool>, dropout: Option<&TrainingDropout>) -> Tensor<3> {
    let padding = padding.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
    let mut hidden = self.embeddings.forward(input_ids, dropout);
    for layer in &self.encoder.layer {
      hidden = layer.forward(hidden, padding.clone(), dropout);
    }
    hidden
  }
}

#[cfg(any(test, feature = "testing"))]
pub(crate) mod testing {
  use super::BertConfig;

  /// A two-layer, 16-wide encoder: big enough to exercise every code path, small enough to build instantly.
  pub fn tiny_config() -> BertConfig {
    BertConfig {
      vocab_size: 32,
      hidden_size: 16,
      num_hidden_layers: 2,
      num_attention_heads: 4,
      intermediate_size: 32,
      max_position_embeddings: 24,
      type_vocab_size: 2,
      layer_norm_eps: 1e-12,
      hidden_dropout_prob: 0.1,
      attention_probs_dropout_prob: 0.1,
      model_type: Some("bert".to_owned()),
      hidden_act: Some("gelu".to_owned()),
      position_embedding_type: Some("absolute".to_owned()),
    }
  }
}

#[cfg(test)]
mod tests {
  use burn::tensor::{Device, Int, Tensor, TensorData};

  use super::testing::tiny_config;
  use super::*;

  fn ids(rows: &[[i64; 5]], device: &Device) -> Tensor<2, Int> {
    let flat: Vec<i64> = rows.iter().flatten().copied().collect();
    Tensor::from_data(TensorData::new(flat, [rows.len(), 5]), device)
  }

  fn padding(rows: &[[bool; 5]], device: &Device) -> Tensor<2, Bool> {
    let flat: Vec<bool> = rows.iter().flatten().copied().collect();
    Tensor::from_data(TensorData::new(flat, [rows.len(), 5]), device)
  }

  fn row(hidden: &Tensor<3>, batch: usize, position: usize) -> Vec<f32> {
    hidden
      .clone()
      .slice_dim(0, batch..=batch)
      .slice_dim(1, position..=position)
      .into_data()
      .try_into_vec::<f32>()
      .unwrap()
  }

  fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
      assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
  }

  #[test]
  fn forward_returns_one_hidden_state_per_token() {
    let device = Device::flex();
    let model = tiny_config().init(&device);
    let hidden = model.forward(
      ids(&[[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]], &device),
      padding(&[[false; 5]; 2], &device),
    );
    assert_eq!(hidden.dims(), [2, 5, 16]);
  }

  #[test]
  fn padded_positions_do_not_influence_real_tokens() {
    let device = Device::flex();
    let model = tiny_config().init(&device);
    let mask = padding(&[[false, false, false, true, true]; 2], &device);
    // Same real tokens, different garbage in the padded tail.
    let hidden = model.forward(ids(&[[1, 2, 3, 0, 0], [1, 2, 3, 9, 11]], &device), mask);
    for position in 0..3 {
      assert_close(&row(&hidden, 0, position), &row(&hidden, 1, position));
    }
  }

  #[test]
  fn a_fully_padded_row_stays_finite() {
    let device = Device::flex();
    let model = tiny_config().init(&device);
    let hidden = model.forward(ids(&[[0; 5]], &device), padding(&[[true; 5]], &device));
    assert!(row(&hidden, 0, 0).iter().all(|value| value.is_finite()));
  }

  #[test]
  fn expected_tensor_count_matches_the_module_tree() {
    let config = tiny_config();
    assert_eq!(config.expected_tensor_count(), 5 + 16 * 2);
  }

  #[test]
  fn a_non_bert_config_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(
      &path,
      r#"{"model_type":"roberta","vocab_size":8,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
          "intermediate_size":8,"max_position_embeddings":8,"type_vocab_size":1}"#,
    )
    .unwrap();
    let error = BertConfig::from_file(&path).unwrap_err();
    assert!(error.to_string().contains("model_type"), "{error}");
  }

  #[test]
  fn heads_must_divide_the_hidden_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(
      &path,
      r#"{"vocab_size":8,"hidden_size":10,"num_hidden_layers":1,"num_attention_heads":4,
          "intermediate_size":8,"max_position_embeddings":8,"type_vocab_size":1}"#,
    )
    .unwrap();
    assert!(BertConfig::from_file(&path).is_err());
  }
}
