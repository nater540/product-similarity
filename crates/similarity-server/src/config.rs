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
use serde::{Deserialize, Serialize};
use similarity_core::{EmbedderOptions, FieldSpec, Policy, QuantitySpec, Schema};
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
  /// The record's quantities: name to its limits. Optional; a quantity is defined by naming it, even with an
  /// empty table.
  #[serde(default)]
  pub quantities: BTreeMap<String, QuantityConfig>,
  /// A candidate policy to judge every request under as well, without deciding any. Optional; with no table
  /// nothing is shadowed.
  #[serde(default)]
  pub shadow: Option<ShadowConfig>,
  /// How `Match` decides and how far it looks.
  #[serde(default)]
  pub matching: MatchingConfig,
}

/// `[matching]`. The table is not called `match` because that is a keyword of the language this is written in.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MatchingConfig {
  /// Call the best candidate a match only at or above this similarity. A measurement, like `policy.threshold`,
  /// read from `calibrate-match` on held-out catalog groups. Without it `Match` ranks and decides nothing.
  pub threshold: Option<f32>,
  /// How many candidates to fetch for each one returned, before they are scored exactly. A cost, not a
  /// threshold: `calibrate-match` reports what a given value loses against scoring the whole catalog.
  pub overfetch: usize,
}

impl Default for MatchingConfig {
  fn default() -> Self {
    Self {
      threshold: None,
      overfetch: crate::engine::Matching::default().overfetch,
    }
  }
}

impl MatchingConfig {
  /// The settings the engine matches by.
  ///
  /// # Errors
  ///
  /// Returns an error if the threshold lies outside the cosine range or the over-fetch is zero.
  pub fn settings(&self) -> anyhow::Result<crate::engine::Matching> {
    if let Some(threshold) = self.threshold
      && !(-1.0..=1.0).contains(&threshold)
    {
      bail!("`matching.threshold` is {threshold}, outside the cosine range [-1, 1]");
    }
    if self.overfetch == 0 {
      bail!("`matching.overfetch` must be at least 1");
    }
    Ok(crate::engine::Matching {
      threshold: self.threshold,
      overfetch: self.overfetch,
    })
  }
}

/// `[server]`
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
  /// The address the gRPC server listens on.
  pub listen: SocketAddr,
  /// The most records of an `Import` stream that are embedded as one batch. A batch is one job on the one
  /// embedding thread, and every other request waits while it runs: larger batches load a catalog faster and
  /// make somebody's save wait longer.
  pub import_batch_records: usize,
}

impl Default for ServerConfig {
  fn default() -> Self {
    Self {
      listen: SocketAddr::from(([127, 0, 0, 1], 50051)),
      import_batch_records: crate::engine::DEFAULT_IMPORT_BATCH,
    }
  }
}

/// Settings `[model]` used to have, with what to tell a file that still has one. Both chose how Burn ran the forward
/// pass when Burn served; ONNX Runtime runs it now, on the CPU (docs/decisions.md, D34). They are refused by name
/// and not as an unknown key, so that a config from before says what to do about it.
const RETIRED_MODEL_KEYS: [(&str, &str); 2] = [
  (
    "engine",
    "every command runs the model through ONNX Runtime, from the `onnx/model.onnx` that `just export-onnx` writes \
     into the model directory",
  ),
  (
    "device",
    "ONNX Runtime runs the model, on the CPU; `--device` on `record-similarity-train` still chooses where a model \
     is trained",
  ),
];

/// `[model]`
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
  /// A sentence-transformers model directory, as written by `record-similarity model fetch`.
  pub dir: PathBuf,
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
  /// The fingerprint of the model the thresholds in this file were measured with. `serve` and `import` refuse
  /// to run with any other model: thresholds do not transfer between models. Omit, and nothing is checked.
  ///
  /// An assertion, not an input: it is no part of the fingerprint or of the layout, and changes no vector.
  pub expected_fingerprint: Option<String>,
}

/// The model that was loaded is not the one the configuration's thresholds were measured with.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
  "the configuration's thresholds were measured with the model `{expected}` (`model.expected_fingerprint`), and \
   the model that was loaded is `{actual}`. Thresholds do not transfer between models: run `calibrate` with this \
   model, set the thresholds from its report, and then name it in `model.expected_fingerprint`"
)]
pub struct FingerprintMismatch {
  /// The fingerprint the configuration names.
  pub expected: String,
  /// The fingerprint of the model that was loaded, at the configured width and token limit.
  pub actual: String,
}

impl Default for ModelConfig {
  fn default() -> Self {
    let options = EmbedderOptions::default();
    Self {
      dir: PathBuf::from("models/mdbr-leaf-mt"),
      max_tokens: options.max_tokens,
      batch_size: options.batch_size,
      // Deliberately not 256, even though that is what `config.example.toml` sets for the default model: a
      // default that truncates would silently damage any model that was not trained for it (docs/decisions.md, D4).
      output_dim: None,
      queue_depth: 256,
      expected_fingerprint: None,
    }
  }
}

impl ModelConfig {
  /// Whether the configuration is tied to the model whose fingerprint is `actual`: `true` if it names that
  /// model, `false` if it names none.
  ///
  /// # Errors
  ///
  /// Returns the mismatch if the configuration names another model.
  pub fn check_fingerprint(&self, actual: &str) -> Result<bool, FingerprintMismatch> {
    match &self.expected_fingerprint {
      None => Ok(false),
      Some(expected) if expected == actual => Ok(true),
      Some(expected) => Err(FingerprintMismatch {
        expected: expected.clone(),
        actual: actual.to_owned(),
      }),
    }
  }

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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
  /// Reject a revision whose overall similarity to its baseline is below this.
  pub threshold: f32,
  /// Reject a revision whose overall similarity to the record's original form is below this. Omit to report drift
  /// without enforcing it.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub anchor_threshold: Option<f32>,
  /// Reject a revision that changes which measures the record is described by: a quantity that became
  /// incomparable with its old value, or one quantity traded for another.
  #[serde(default)]
  pub reject_on_measure_change: bool,
}

/// `[fields.<name>]`
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldConfig {
  /// Relative importance in the overall score.
  pub weight: f32,
  /// Refuse records without a value for this field.
  #[serde(default)]
  pub required: bool,
  /// Reject a revision whose similarity for this field alone is below this.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub reject_below: Option<f32>,
}

/// `[quantities.<name>]`
///
/// Both keys are optional, and with neither the quantity is compared and reported and never rejects. That is the
/// default on purpose: a limit is a measurement, chosen from `calibrate` on labeled pairs, like a threshold.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantityConfig {
  /// Reject a revision that changes this quantity by more than this factor, in either direction.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub reject_above_factor: Option<f64>,
  /// Reject a revision that puts this quantity in a unit of another dimension.
  #[serde(default)]
  pub reject_incomparable: bool,
}

/// A setting that may also be switched off where the table it overrides switches it on: a floor, or a factor
/// limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Setting {
  /// No floor, or no limit. Written `"off"`: TOML has no null.
  Off,
  /// The value.
  Value(f64),
}

impl Setting {
  /// The value, or `None` when the setting is off.
  pub fn value(self) -> Option<f64> {
    match self {
      Self::Off => None,
      Self::Value(value) => Some(value),
    }
  }
}

impl<'de> Deserialize<'de> for Setting {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use serde::de::Error as _;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
      Number(f64),
      Text(String),
    }
    match Raw::deserialize(deserializer)? {
      Raw::Number(value) => Ok(Self::Value(value)),
      Raw::Text(text) if text == "off" => Ok(Self::Off),
      Raw::Text(text) => Err(D::Error::custom(format!(
        "`{text}` is not a number; the only word allowed here is \"off\""
      ))),
    }
  }
}

/// Which judgments a shadow writes a line for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadowEvents {
  /// Only the requests the candidate would have decided the other way. The summary still counts the rest.
  #[default]
  Disagreements,
  /// Every judged request. One more line on stdout per request, which is what it costs to see how the
  /// candidate scores the requests it agrees on.
  All,
}

/// `[shadow]`: a candidate policy, written as what it changes.
///
/// `[shadow.policy]` is a whole policy and inherits nothing: a candidate's threshold belongs to its own weights
/// and floors, so it has to be stated. `[shadow.fields.<name>]` and `[shadow.quantities.<name>]` are overrides
/// of the live tables, and can only name what those define, since a candidate with other fields would be judging
/// other requests.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowConfig {
  /// What the candidate is called on every line it writes.
  pub name: String,
  /// Which judgments get a line of their own.
  #[serde(default)]
  pub events: ShadowEvents,
  /// How often the running counts are written, in seconds.
  #[serde(default = "default_summary_every_s")]
  pub summary_every_s: u64,
  /// The candidate's thresholds.
  pub policy: PolicyConfig,
  /// What the candidate changes about a field.
  #[serde(default)]
  pub fields: BTreeMap<String, ShadowFieldConfig>,
  /// What the candidate changes about a quantity.
  #[serde(default)]
  pub quantities: BTreeMap<String, ShadowQuantityConfig>,
  // A candidate model is not something this server can shadow. Named so that asking for one gets the reason and
  // not "unknown field".
  #[serde(default)]
  model: Option<toml::Value>,
  #[serde(default)]
  qdrant: Option<toml::Value>,
}

fn default_summary_every_s() -> u64 {
  60
}

/// `[shadow.fields.<name>]`
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowFieldConfig {
  /// The candidate's weight for the field. Omit to keep the live one.
  #[serde(default)]
  pub weight: Option<f32>,
  /// The candidate's floor for the field, or `"off"` for none. Omit to keep the live one.
  #[serde(default)]
  pub reject_below: Option<Setting>,
}

/// `[shadow.quantities.<name>]`
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowQuantityConfig {
  /// The candidate's factor limit for the quantity, or `"off"` for none. Omit to keep the live one.
  #[serde(default)]
  pub reject_above_factor: Option<Setting>,
  /// Whether the candidate rejects a unit of another dimension. Omit to keep the live setting.
  #[serde(default)]
  pub reject_incomparable: Option<bool>,
}

impl Config {
  /// Parses a configuration from TOML text.
  ///
  /// # Errors
  ///
  /// Returns an error naming the offending key if the text is not valid TOML, has an unknown or missing key, or
  /// describes an invalid schema or policy.
  pub fn parse(text: &str) -> anyhow::Result<Self> {
    let table: toml::Table = toml::from_str(text)?;
    let model = table.get("model").and_then(toml::Value::as_table);
    for (key, why) in RETIRED_MODEL_KEYS {
      if model.is_some_and(|model| model.contains_key(key)) {
        bail!("`model.{key}` is no longer a setting: {why}. Remove the line");
      }
    }
    let config: Self = toml::from_str(text)?;
    // Building these is the validation; the server builds them again from the same data when it starts.
    config.schema()?;
    config.policy()?;
    if config.model.queue_depth == 0 {
      bail!("`model.queue_depth` must be positive");
    }
    if let Some(expected) = &config.model.expected_fingerprint
      && expected.trim() != expected
    {
      bail!(
        "`model.expected_fingerprint` must be a fingerprint exactly as `calibrate` prints it, with no spaces around it"
      );
    }
    if config.model.expected_fingerprint.as_deref() == Some("") {
      bail!("`model.expected_fingerprint` must not be empty; leave the key out to check nothing");
    }
    if config.qdrant.collection.trim().is_empty() {
      bail!("`qdrant.collection` must not be blank");
    }
    config.matching.settings()?;
    if config.server.import_batch_records == 0 {
      bail!("`server.import_batch_records` must be at least 1");
    }
    if let Some(shadow) = &config.shadow {
      if shadow.name.trim().is_empty() {
        bail!("`shadow.name` must not be blank");
      }
      if shadow.summary_every_s == 0 {
        bail!("`shadow.summary_every_s` must be positive");
      }
      if shadow.model.is_some() || shadow.qdrant.is_some() {
        bail!(
          "`[shadow.model]` and `[shadow.qdrant]`: this server shadows a policy, not a model. A candidate model \
           is compared with `calibrate --model` (docs/decisions.md, D30)"
        );
      }
      let candidate = config
        .shadowed()
        .context("`[shadow]` does not describe a valid policy")?;
      candidate
        .schema()
        .context("`[shadow]` does not describe a valid schema")?;
      candidate.policy().context("`[shadow.policy]` is not a valid policy")?;
    }
    Ok(config)
  }

  /// The configuration the shadow judges by: this one with `[shadow]`'s policy in place of `[policy]` and its
  /// overrides applied to `[fields]` and `[quantities]`. The names, and which fields are required, are the live
  /// ones by construction, so the candidate reads the same vectors under the same layout.
  ///
  /// # Errors
  ///
  /// Returns an error if there is no `[shadow]` table, or if it overrides a field or a quantity that the live
  /// tables do not define.
  #[expect(clippy::cast_possible_truncation, reason = "floors are configured as 32-bit numbers")]
  pub fn shadowed(&self) -> anyhow::Result<Self> {
    let Some(shadow) = &self.shadow else {
      bail!("the configuration has no `[shadow]` table");
    };
    let mut candidate = self.clone();
    candidate.shadow = None;
    candidate.policy = shadow.policy;
    for (name, change) in &shadow.fields {
      let Some(field) = candidate.fields.get_mut(name) else {
        bail!("`[shadow.fields.{name}]`: `[fields]` has no `{name}`");
      };
      if let Some(weight) = change.weight {
        field.weight = weight;
      }
      if let Some(floor) = change.reject_below {
        field.reject_below = floor.value().map(|floor| floor as f32);
      }
    }
    for (name, change) in &shadow.quantities {
      let Some(quantity) = candidate.quantities.get_mut(name) else {
        bail!("`[shadow.quantities.{name}]`: `[quantities]` has no `{name}`");
      };
      if let Some(limit) = change.reject_above_factor {
        quantity.reject_above_factor = limit.value();
      }
      if let Some(reject) = change.reject_incomparable {
        quantity.reject_incomparable = reject;
      }
    }
    Ok(candidate)
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

  /// The record schema described by `[fields]` and `[quantities]`.
  ///
  /// # Errors
  ///
  /// Returns an error if the schema is invalid (no fields, a bad name, a non-positive weight, a quantity named
  /// like a field, a factor limit below 1, ...).
  pub fn schema(&self) -> anyhow::Result<Schema> {
    schema_of(&self.fields, &self.quantities)
  }

  /// The thresholds described by `[policy]`.
  ///
  /// # Errors
  ///
  /// Returns an error if a threshold lies outside the cosine range.
  pub fn policy(&self) -> anyhow::Result<Policy> {
    policy_of(&self.policy)
  }
}

/// The schema that a `[fields]` and a `[quantities]` table describe.
///
/// # Errors
///
/// As [`Config::schema`].
pub fn schema_of(
  fields: &BTreeMap<String, FieldConfig>,
  quantities: &BTreeMap<String, QuantityConfig>,
) -> anyhow::Result<Schema> {
  let fields = fields
    .iter()
    .map(|(name, field)| FieldSpec {
      name: name.clone(),
      weight: field.weight,
      required: field.required,
      reject_below: field.reject_below,
    })
    .collect();
  let quantities = quantities
    .iter()
    .map(|(name, quantity)| QuantitySpec {
      name: name.clone(),
      reject_above_factor: quantity.reject_above_factor,
      reject_incomparable: quantity.reject_incomparable,
    })
    .collect();
  Ok(Schema::new(fields)?.with_quantities(quantities)?)
}

/// The policy that a `[policy]` table describes.
///
/// # Errors
///
/// As [`Config::policy`].
pub fn policy_of(table: &PolicyConfig) -> anyhow::Result<Policy> {
  let mut policy = Policy::new(table.threshold, table.anchor_threshold)?;
  policy.reject_on_measure_change = table.reject_on_measure_change;
  Ok(policy)
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
    assert_eq!(config.server.import_batch_records, 8);
    assert!(Config::parse(&format!("[server]\nimport_batch_records = 0\n{MINIMAL}")).is_err());
    let small = Config::parse(&format!("[server]\nimport_batch_records = 2\n{MINIMAL}")).unwrap();
    assert_eq!(small.server.import_batch_records, 2);
    assert_eq!(
      config.model.output_dim, None,
      "truncation is opt-in: it is only safe for some models"
    );
    assert_eq!(config.qdrant.collection, "records");
    assert_eq!(config.policy.anchor_threshold, None);
    assert_eq!(config.schema().unwrap().fields().len(), 1);
    assert!(config.schema().unwrap().quantities().is_empty());
    assert!(!config.policy().unwrap().reject_on_measure_change);
  }

  #[test]
  fn the_shipped_example_parses() {
    let text = include_str!("../../../config.example.toml");
    let config = Config::parse(text).unwrap();
    let schema = config.schema().unwrap();
    assert_eq!(config.model.output_dim, Some(256));
    assert!(schema.field("name").is_some_and(|field| field.required));
    assert!(config.policy().is_ok());

    // The shipped quantities are report-only: nobody has measured where a limit belongs (docs/decisions.md, D16).
    let names: Vec<&str> = schema.quantities().iter().map(|spec| spec.name.as_str()).collect();
    assert_eq!(
      names,
      ["cbd_per_item", "thc_per_item", "volume_per_item", "weight_per_item"]
    );
    assert!(
      schema
        .quantities()
        .iter()
        .all(|spec| spec.reject_above_factor.is_none() && !spec.reject_incomparable)
    );
    assert!(!config.policy().unwrap().reject_on_measure_change);
  }

  #[test]
  fn quantities_are_defined_by_naming_them_and_limited_only_when_asked() {
    let config = Config::parse(&format!(
      "{MINIMAL}
      [quantities.thc_per_item]

      [quantities.weight_per_item]
      reject_above_factor = 4.0
      reject_incomparable = true
      "
    ))
    .unwrap();
    let schema = config.schema().unwrap();
    let thc = schema.quantity("thc_per_item").unwrap();
    assert_eq!((thc.reject_above_factor, thc.reject_incomparable), (None, false));
    let weight = schema.quantity("weight_per_item").unwrap();
    assert_eq!(
      (weight.reject_above_factor, weight.reject_incomparable),
      (Some(4.0), true)
    );

    let strict = Config::parse(&MINIMAL.replace("threshold = 0.8", "threshold = 0.8\nreject_on_measure_change = true"));
    assert!(strict.unwrap().policy().unwrap().reject_on_measure_change);
  }

  #[test]
  fn invalid_quantities_are_refused_at_parse_time() {
    let with = |table: &str| Config::parse(&format!("{MINIMAL}\n{table}"));
    assert!(
      with("[quantities.weight]\nreject_above_factor = 0.5").is_err(),
      "below 1"
    );
    assert!(with("[quantities.name]").is_err(), "named like a text field");
    assert!(with("[quantities.\"per item\"]").is_err(), "a space in the name");
    let typo = with("[quantities.weight]\nreject_above = 4.0").unwrap_err();
    assert!(format!("{typo:#}").contains("reject_above"), "{typo:#}");
    assert!(with("[quantities.weight]\nreject_above_factor = 4.0").is_ok());
  }

  #[test]
  fn every_section_can_be_overridden() {
    let config = Config::parse(
      r#"
      [server]
      listen = "0.0.0.0:9000"

      [model]
      dir = "models/gte-small"
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
    assert_eq!(config.model.embedder_options().max_tokens, 128);
    assert_eq!(
      config.qdrant.settings(Some("key".to_owned())).timeout,
      Duration::from_millis(250)
    );
    assert_eq!(config.policy().unwrap().anchor_threshold, Some(0.5));
    assert_eq!(config.schema().unwrap().field("name").unwrap().reject_below, Some(0.4));
  }

  /// The key that ties thresholds to the model they were measured with (D31). It is opt-in: the shipped example
  /// does not set it, and a file without it checks nothing.
  #[test]
  fn a_config_names_the_model_its_thresholds_were_measured_with_only_if_it_wants_to() {
    let untied = Config::parse(MINIMAL).unwrap();
    assert_eq!(untied.model.expected_fingerprint, None);
    assert_eq!(untied.model.check_fingerprint("abc-d256-t256"), Ok(false));
    let shipped = Config::parse(include_str!("../../../config.example.toml")).unwrap();
    assert_eq!(shipped.model.expected_fingerprint, None);

    let tied = Config::parse(&format!("[model]\nexpected_fingerprint = \"abc-d256-t256\"\n{MINIMAL}")).unwrap();
    assert_eq!(tied.model.check_fingerprint("abc-d256-t256"), Ok(true));
    let mismatch = tied.model.check_fingerprint("abc-d384-t256").unwrap_err();
    assert_eq!(
      (mismatch.expected.as_str(), mismatch.actual.as_str()),
      ("abc-d256-t256", "abc-d384-t256"),
      "the same files at another width are another model"
    );
    let message = mismatch.to_string();
    assert!(
      message.contains("abc-d256-t256") && message.contains("abc-d384-t256"),
      "{message}"
    );
    assert!(message.contains("do not transfer"), "{message}");

    for bad in ["\"\"", "\" abc \"", "7"] {
      assert!(
        Config::parse(&format!("[model]\nexpected_fingerprint = {bad}\n{MINIMAL}")).is_err(),
        "{bad}"
      );
    }
  }

  /// A match threshold is a measurement, so none is shipped; the over-fetch is a cost with a default.
  #[test]
  fn matching_has_no_threshold_unless_one_was_measured_and_an_overfetch_of_at_least_one() {
    let default = Config::parse(MINIMAL).unwrap().matching.settings().unwrap();
    assert_eq!((default.threshold, default.overfetch), (None, 5));
    let shipped = Config::parse(include_str!("../../../config.example.toml")).unwrap();
    assert_eq!(shipped.matching.threshold, None, "the example ships no match threshold");

    let with = |table: &str| Config::parse(&format!("{MINIMAL}\n[matching]\n{table}"));
    let set = with("threshold = 0.8\noverfetch = 12")
      .unwrap()
      .matching
      .settings()
      .unwrap();
    assert_eq!((set.threshold, set.overfetch), (Some(0.8), 12));
    assert_eq!(with("overfetch = 1").unwrap().matching.overfetch, 1);
    assert!(with("overfetch = 0").is_err());
    assert!(with("threshold = 1.5").is_err());
    assert!(with("threshold = -1.5").is_err());
    assert!(with("threshhold = 0.8").is_err(), "a misspelled key");
  }

  const SHADOWED: &str = r#"
    [policy]
    threshold = 0.8
    anchor_threshold = 0.6
    reject_on_measure_change = true

    [fields.name]
    weight = 4.0
    required = true
    reject_below = 0.4

    [fields.description]
    weight = 1.0

    [quantities.weight]
    reject_above_factor = 4.0
    reject_incomparable = true

    [quantities.thc]

    [shadow]
    name = "candidate"
  "#;

  fn shadowed(tables: &str) -> anyhow::Result<Config> {
    Config::parse(&format!("{SHADOWED}\n{tables}"))
  }

  #[test]
  fn a_config_without_a_shadow_has_none() {
    assert_eq!(Config::parse(MINIMAL).unwrap().shadow, None);
    assert!(Config::parse(MINIMAL).unwrap().shadowed().is_err());
    let shipped = Config::parse(include_str!("../../../config.example.toml")).unwrap();
    assert_eq!(shipped.shadow, None, "a shadow is something an operator turns on");
  }

  /// The candidate's policy is its own, whole: nothing of `[policy]` leaks into it, least of all a threshold,
  /// which belongs to the weights and floors it was measured with.
  #[test]
  fn a_shadow_policy_is_stated_whole_and_inherits_nothing() {
    let missing = shadowed("").unwrap_err();
    assert!(format!("{missing:#}").contains("policy"), "{missing:#}");

    let config = shadowed("[shadow.policy]\nthreshold = 0.7").unwrap();
    let shadow = config.shadow.as_ref().unwrap();
    assert_eq!(
      (shadow.name.as_str(), shadow.events, shadow.summary_every_s),
      ("candidate", ShadowEvents::Disagreements, 60)
    );
    let policy = config.shadowed().unwrap().policy().unwrap();
    assert_eq!(policy.threshold, 0.7);
    assert_eq!(
      policy.anchor_threshold, None,
      "the live drift threshold is not inherited"
    );
    assert!(!policy.reject_on_measure_change);
    // The live policy is what it was.
    assert_eq!(config.policy().unwrap().threshold, 0.8);
    assert_eq!(config.policy().unwrap().anchor_threshold, Some(0.6));
  }

  #[test]
  fn a_shadow_overrides_only_what_it_names_and_can_switch_a_limit_off() {
    let base = SHADOWED
      .replace(
        "name = \"candidate\"",
        "name = \"candidate\"\nevents = \"all\"\nsummary_every_s = 5",
      )
      .replace("[quantities.thc]", "[quantities.thc]\nreject_incomparable = true");
    let config = Config::parse(&format!(
      r#"{base}
      [shadow.policy]
      threshold = 0.7

      [shadow.fields.name]
      weight = 2.5
      reject_below = "off"

      [shadow.fields.description]
      reject_below = 0.25

      [shadow.quantities.weight]
      reject_above_factor = "off"
      reject_incomparable = false

      [shadow.quantities.thc]
      reject_above_factor = 3.0
      "#
    ))
    .unwrap();
    let shadow = config.shadow.as_ref().unwrap();
    assert_eq!((shadow.events, shadow.summary_every_s), (ShadowEvents::All, 5));

    let candidate = config.shadowed().unwrap();
    assert_eq!(candidate.shadow, None);
    let schema = candidate.schema().unwrap();
    let name = schema.field("name").unwrap();
    assert_eq!((name.weight, name.required, name.reject_below), (2.5, true, None));
    let description = schema.field("description").unwrap();
    assert_eq!((description.weight, description.reject_below), (1.0, Some(0.25)));
    let weight = schema.quantity("weight").unwrap();
    assert_eq!((weight.reject_above_factor, weight.reject_incomparable), (None, false));
    let thc = schema.quantity("thc").unwrap();
    assert_eq!(
      (thc.reject_above_factor, thc.reject_incomparable),
      (Some(3.0), true),
      "what the shadow does not name stays as the live table has it"
    );

    // The live schema is untouched, and both describe the same vectors.
    let live = config.schema().unwrap();
    assert_eq!(live.field("name").unwrap().reject_below, Some(0.4));
    assert_eq!(live.quantity("weight").unwrap().reject_above_factor, Some(4.0));
    assert_eq!(live.layout_id("m"), schema.layout_id("m"));
  }

  #[test]
  fn a_shadow_that_names_what_the_live_tables_lack_is_refused() {
    let policy = "[shadow.policy]\nthreshold = 0.7";
    let field = shadowed(&format!("{policy}\n[shadow.fields.brand]\nweight = 1.0")).unwrap_err();
    assert!(format!("{field:#}").contains("`[fields]` has no `brand`"), "{field:#}");
    let quantity = shadowed(&format!(
      "{policy}\n[shadow.quantities.volume]\nreject_incomparable = true"
    ))
    .unwrap_err();
    assert!(
      format!("{quantity:#}").contains("`[quantities]` has no `volume`"),
      "{quantity:#}"
    );
  }

  #[test]
  fn an_invalid_shadow_is_refused_at_parse_time() {
    let policy = "[shadow.policy]\nthreshold = 0.7";
    assert!(
      shadowed("[shadow.policy]\nthreshold = 7").is_err(),
      "outside the cosine range"
    );
    assert!(shadowed(&format!("{policy}\n[shadow.fields.name]\nweight = 0")).is_err());
    assert!(shadowed(&format!("{policy}\n[shadow.fields.name]\nreject_below = \"none\"")).is_err());
    assert!(shadowed(&format!("{policy}\n[shadow.fields.name]\nrequired = false")).is_err());
    assert!(
      shadowed(&format!(
        "{policy}\n[shadow.quantities.weight]\nreject_above_factor = 0.5"
      ))
      .is_err()
    );
    assert!(Config::parse(&format!("{}\n{policy}", SHADOWED.replace("\"candidate\"", "\" \""))).is_err());
    let never = SHADOWED.replace("name = \"candidate\"", "name = \"candidate\"\nsummary_every_s = 0");
    assert!(Config::parse(&format!("{never}\n{policy}")).is_err());
    let events = SHADOWED.replace("name = \"candidate\"", "name = \"candidate\"\nevents = \"some\"");
    assert!(Config::parse(&format!("{events}\n{policy}")).is_err());
  }

  /// Asking for a shadow model gets the reason it is not there, not "unknown field".
  #[test]
  fn a_shadow_model_is_refused_with_the_reason() {
    let policy = "[shadow.policy]\nthreshold = 0.7";
    for table in [
      "[shadow.model]\ndir = \"models/candidate\"",
      "[shadow.qdrant]\ncollection = \"c\"",
    ] {
      let error = shadowed(&format!("{policy}\n{table}")).unwrap_err();
      assert!(
        format!("{error:#}").contains("shadows a policy, not a model"),
        "{error:#}"
      );
    }
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
  }

  #[test]
  fn a_setting_that_chose_how_burn_ran_the_model_is_refused_by_name_and_says_what_replaced_it() {
    for (line, expected) in [
      ("engine = \"onnx\"", "`model.engine` is no longer a setting"),
      ("engine = \"burn\"", "`model.engine` is no longer a setting"),
      ("device = \"cpu\"", "`model.device` is no longer a setting"),
      ("device = \"metal\"", "`model.device` is no longer a setting"),
    ] {
      let error = Config::parse(&format!("[model]\n{line}\n{MINIMAL}"))
        .unwrap_err()
        .to_string();
      assert!(error.contains(expected), "{line}: {error}");
      assert!(error.contains("ONNX Runtime"), "{line}: {error}");
      assert!(error.contains("Remove the line"), "{line}: {error}");
    }
    // Any other key the table does not have is still an unknown key, as before.
    let error = Config::parse(&format!("[model]\nengines = \"onnx\"\n{MINIMAL}"))
      .unwrap_err()
      .to_string();
    assert!(error.contains("unknown field `engines`"), "{error}");
    // And the keys are only retired under `[model]`: a field of a record may be called `engine`.
    assert!(Config::parse(&format!("{MINIMAL}\n[fields.engine]\nweight = 1.0\n")).is_ok());
  }
}
