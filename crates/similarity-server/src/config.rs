//! The server's configuration file.
//!
//! One TOML file describes everything that changes what the server computes or where it stores it. Every table and
//! key is checked (`deny_unknown_fields`): a misspelled `reject_below` that silently fell back to "no floor" would
//! be a policy hole nobody could see.
//!
//! Secrets are not in the file. The Qdrant API key comes from the `QDRANT_API_KEY` environment variable.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use serde::Deserialize;
use similarity_core::{EmbedderOptions, FieldSpec, Policy, Schema};
use similarity_store::QdrantSettings;

/// The whole configuration file.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
  /// Network settings.
  #[serde(default)]
  pub server: ServerConfig,
  /// The embedding model and how it runs.
  #[serde(default)]
  pub model: ModelConfig,
  /// The vector store.
  #[serde(default)]
  pub qdrant: QdrantConfig,
  /// The accept/reject thresholds.
  pub policy: PolicyConfig,
  /// The record schema: field name to its settings. At least one field is required.
  pub fields: BTreeMap<String, FieldConfig>,
}

/// `[server]`
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
  /// The address the gRPC server listens on.
  pub listen: SocketAddr,
}

impl Default for ServerConfig {
  fn default() -> Self {
    Self {
      listen: SocketAddr::from(([127, 0, 0, 1], 50051)),
    }
  }
}

/// Which Burn backend runs the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceChoice {
  /// The pure-Rust CPU backend. Always available.
  #[default]
  Cpu,
  /// The GPU through Metal. Requires a binary built with `--features metal`.
  Metal,
}

/// `[model]`
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
  /// A sentence-transformers model directory, as written by `record-similarity model fetch`.
  pub dir: PathBuf,
  /// Where the encoder runs.
  pub device: DeviceChoice,
  /// Texts longer than this many tokens are truncated.
  pub max_tokens: usize,
  /// The most texts in one forward pass.
  pub batch_size: usize,
  /// Keep only the first N components of each field vector. Omit to keep the model's native width.
  ///
  /// Only safe for models trained to be truncated (Matryoshka models, such as the default one). For any other
  /// model the leading components are not a good embedding by themselves, which is why this has no default.
  pub output_dim: Option<usize>,
  /// How many embedding requests may wait for the encoder before new ones are refused with `RESOURCE_EXHAUSTED`.
  pub queue_depth: usize,
}

impl Default for ModelConfig {
  fn default() -> Self {
    let options = EmbedderOptions::default();
    Self {
      dir: PathBuf::from("models/mdbr-leaf-mt"),
      device: DeviceChoice::Cpu,
      max_tokens: options.max_tokens,
      batch_size: options.batch_size,
      // Deliberately not 256, even though that is what `config.example.toml` sets for the default model: a
      // default that truncates would silently damage any model that was not trained for it (docs/decisions.md, D4).
      output_dim: None,
      queue_depth: 256,
    }
  }
}

impl ModelConfig {
  /// The options the embedder is loaded with.
  pub fn embedder_options(&self) -> EmbedderOptions {
    EmbedderOptions {
      max_tokens: self.max_tokens,
      batch_size: self.batch_size,
      output_dim: self.output_dim,
    }
  }
}

/// `[qdrant]`
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct QdrantConfig {
  /// Qdrant's gRPC endpoint.
  pub url: String,
  /// The collection holding this server's records.
  pub collection: String,
  /// The deadline for any single Qdrant request, in milliseconds.
  pub timeout_ms: u64,
}

impl Default for QdrantConfig {
  fn default() -> Self {
    Self {
      url: "http://localhost:6334".to_owned(),
      collection: "records".to_owned(),
      timeout_ms: 5_000,
    }
  }
}

impl QdrantConfig {
  /// The connection settings, with the API key taken from the environment.
  pub fn settings(&self, api_key: Option<String>) -> QdrantSettings {
    QdrantSettings {
      url: self.url.clone(),
      api_key,
      collection: self.collection.clone(),
      timeout: Duration::from_millis(self.timeout_ms),
    }
  }
}

/// `[policy]`
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
  /// Reject a revision whose overall similarity to its baseline is below this.
  pub threshold: f32,
  /// Reject a revision whose overall similarity to the record's original form is below this. Omit to report drift
  /// without enforcing it.
  #[serde(default)]
  pub anchor_threshold: Option<f32>,
}

/// `[fields.<name>]`
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldConfig {
  /// Relative importance in the overall score.
  pub weight: f32,
  /// Refuse records without a value for this field.
  #[serde(default)]
  pub required: bool,
  /// Reject a revision whose similarity for this field alone is below this.
  #[serde(default)]
  pub reject_below: Option<f32>,
}

impl Config {
  /// Parses a configuration from TOML text.
  ///
  /// # Errors
  ///
  /// Returns an error naming the offending key if the text is not valid TOML, has an unknown or missing key, or
  /// describes an invalid schema or policy.
  pub fn parse(text: &str) -> anyhow::Result<Self> {
    let config: Self = toml::from_str(text)?;
    // Building these is the validation; the server builds them again from the same data when it starts.
    config.schema()?;
    config.policy()?;
    if config.model.queue_depth == 0 {
      bail!("`model.queue_depth` must be positive");
    }
    if config.qdrant.collection.trim().is_empty() {
      bail!("`qdrant.collection` must not be blank");
    }
    Ok(config)
  }

  /// Reads and parses a configuration file.
  ///
  /// # Errors
  ///
  /// Returns an error naming the file if it cannot be read, plus anything [`Self::parse`] returns.
  pub fn load(path: &Path) -> anyhow::Result<Self> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read config file {}", path.display()))?;
    Self::parse(&text).with_context(|| format!("invalid config file {}", path.display()))
  }

  /// The record schema described by `[fields]`.
  ///
  /// # Errors
  ///
  /// Returns an error if the schema is invalid (no fields, a bad name, a non-positive weight, ...).
  pub fn schema(&self) -> anyhow::Result<Schema> {
    let fields = self
      .fields
      .iter()
      .map(|(name, field)| FieldSpec {
        name: name.clone(),
        weight: field.weight,
        required: field.required,
        reject_below: field.reject_below,
      })
      .collect();
    Ok(Schema::new(fields)?)
  }

  /// The thresholds described by `[policy]`.
  ///
  /// # Errors
  ///
  /// Returns an error if a threshold lies outside the cosine range.
  pub fn policy(&self) -> anyhow::Result<Policy> {
    Ok(Policy::new(self.policy.threshold, self.policy.anchor_threshold)?)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const MINIMAL: &str = r"
    [policy]
    threshold = 0.8

    [fields.name]
    weight = 1.0
  ";

  #[test]
  fn a_minimal_file_gets_every_default() {
    let config = Config::parse(MINIMAL).unwrap();
    assert_eq!(config.server.listen.port(), 50051);
    assert_eq!(config.model.device, DeviceChoice::Cpu);
    assert_eq!(
      config.model.output_dim, None,
      "truncation is opt-in: it is only safe for some models"
    );
    assert_eq!(config.qdrant.collection, "records");
    assert_eq!(config.policy.anchor_threshold, None);
    assert_eq!(config.schema().unwrap().fields().len(), 1);
  }

  #[test]
  fn the_shipped_example_parses() {
    let text = include_str!("../../../config.example.toml");
    let config = Config::parse(text).unwrap();
    let schema = config.schema().unwrap();
    assert_eq!(config.model.output_dim, Some(256));
    assert!(schema.field("name").is_some_and(|field| field.required));
    assert!(config.policy().is_ok());
  }

  #[test]
  fn every_section_can_be_overridden() {
    let config = Config::parse(
      r#"
      [server]
      listen = "0.0.0.0:9000"

      [model]
      dir = "models/gte-small"
      device = "metal"
      max_tokens = 128
      batch_size = 8
      queue_depth = 16

      [qdrant]
      url = "https://example.cloud.qdrant.io:6334"
      collection = "products"
      timeout_ms = 250

      [policy]
      threshold = 0.7
      anchor_threshold = 0.5

      [fields.name]
      weight = 4
      required = true
      reject_below = 0.4

      [fields.description]
      weight = 1
      "#,
    )
    .unwrap();
    assert_eq!(config.server.listen.to_string(), "0.0.0.0:9000");
    assert_eq!(config.model.device, DeviceChoice::Metal);
    assert_eq!(config.model.embedder_options().max_tokens, 128);
    assert_eq!(
      config.qdrant.settings(Some("key".to_owned())).timeout,
      Duration::from_millis(250)
    );
    assert_eq!(config.policy().unwrap().anchor_threshold, Some(0.5));
    assert_eq!(config.schema().unwrap().field("name").unwrap().reject_below, Some(0.4));
  }

  #[test]
  fn a_misspelled_key_is_an_error_not_a_silent_default() {
    let typo = MINIMAL.replace("weight = 1.0", "weight = 1.0\nreject_bellow = 0.5");
    let error = Config::parse(&typo).unwrap_err();
    assert!(format!("{error:#}").contains("reject_bellow"), "{error:#}");

    assert!(Config::parse(&format!("{MINIMAL}\n[modle]\ndir = \"x\"")).is_err());
  }

  #[test]
  fn a_file_without_fields_or_policy_is_refused() {
    assert!(Config::parse("[policy]\nthreshold = 0.8").is_err());
    assert!(Config::parse("[fields.name]\nweight = 1.0").is_err());
    assert!(Config::parse("[policy]\nthreshold = 0.8\n[fields]").is_err());
  }

  #[test]
  fn invalid_values_are_refused_at_parse_time() {
    assert!(Config::parse(&MINIMAL.replace("threshold = 0.8", "threshold = 8")).is_err());
    assert!(Config::parse(&MINIMAL.replace("weight = 1.0", "weight = 0")).is_err());
    assert!(Config::parse(&format!("[model]\nqueue_depth = 0\n{MINIMAL}")).is_err());
    assert!(Config::parse(&format!("[model]\ndevice = \"cuda\"\n{MINIMAL}")).is_err());
  }
}
