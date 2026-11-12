//! Which model was put into service, on what evidence, and the gate a candidate passes to get there.
//!
//! Two mistakes are easy to make with a second model: serving it under thresholds measured for the first (D3), and
//! writing its vectors into the first one's collection (D12). `model.expected_fingerprint` stops the first at
//! start-up. This module is what tells an operator what to put there, and keeps the record of having done so:
//!
//! - **The registry** is one file of JSON lines beside the model directories, never inside one (D18):
//!   `registered`, `promoted`, and `retired` entries, appended and never rewritten. It is a data file this tool
//!   owns, like `provenance.json`, not a stream of results, so it is not written through `similarity-log` and an
//!   entry may nest. Its names are an interface all the same: fields are added, never renamed or repurposed.
//! - **The gate** ([`gate`]) is a pure function over a candidate's holdout calibration and the registry's record
//!   of the model in service. It judges the candidate under the candidate's own configuration, and the model in
//!   service under the configuration it was promoted with, on the same holdout pairs.
//!
//! **The gate measures; it chooses nothing.** It never suggests or writes a threshold. The holdout it reads is
//! the one measurement of what was already chosen on other pairs (D24), and it refuses a calibration that is
//! not of the holdout split, or not of the model the configuration loads, whatever else is said.
//!
//! **The server never reads the registry.** It is a local file that names local paths; a container does not
//! have it. What travels to production is the one line the gate prints for the configuration.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use similarity_core::provenance::{Provenance, Stamp};
use similarity_core::record::hex_prefix;
use similarity_core::{Policy, Schema};

use crate::config::{Config, FieldConfig, PolicyConfig, QuantityConfig, policy_of, schema_of};
use crate::offline::calibrate::Calibration;
use crate::offline::diff::{self, Diff, FailOn};
use crate::offline::events::Run;
use crate::offline::pairs::{Confidence, Label};
use crate::offline::report::describe_rate;
use crate::offline::stats;

/// The registry's file name, beside the model directories.
pub const REGISTRY_FILE: &str = "registry.ndjson";
/// The version of the entries this module writes, and the newest it reads.
pub const FORMAT: u32 = 1;
/// The split a promotion is measured on.
pub const HOLDOUT: &str = "holdout";
/// How many hex digits of a layout's hash name its collection.
const COLLECTION_HASH_BYTES: usize = 6;

/// When an entry was written, and by what.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Written {
  /// The version of the entry's layout.
  pub format: u32,
  /// When, in seconds since the Unix epoch.
  pub at: u64,
  /// The program that wrote it.
  #[serde(flatten)]
  pub build: Stamp,
}

impl Written {
  /// An entry written now by `build`.
  pub fn now(build: Stamp) -> Self {
    Self {
      format: FORMAT,
      at: SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs()),
      build,
    }
  }
}

/// One line of the registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "entry", rename_all = "snake_case")]
pub enum Entry {
  /// A model directory was made known.
  Registered(Box<Registered>),
  /// A model passed the gate under a configuration.
  Promoted(Box<Promoted>),
  /// A model was taken out of service.
  Retired(Retired),
}

impl Entry {
  fn written(&self) -> &Written {
    match self {
      Self::Registered(entry) => &entry.written,
      Self::Promoted(entry) => &entry.written,
      Self::Retired(entry) => &entry.written,
    }
  }
}

/// A model directory, by the digest of its content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Registered {
  /// When, and by what.
  #[serde(flatten)]
  pub written: Written,
  /// The content digest of the model: the first part of every fingerprint it has.
  pub digest: String,
  /// The directory, as it was given.
  pub dir: String,
  /// The model's own note about where it came from, copied, if it had one. The directory may not outlive this file.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub provenance: Option<Provenance>,
}

/// The policy, fields, and quantities a model was promoted under, kept so that a later gate can judge its
/// calibration again exactly as this one did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotedConfig {
  /// `[policy]`
  pub policy: PolicyConfig,
  /// `[fields]`
  pub fields: BTreeMap<String, FieldConfig>,
  /// `[quantities]`
  #[serde(default)]
  pub quantities: BTreeMap<String, QuantityConfig>,
}

impl PromotedConfig {
  /// The three tables of `config` that judge.
  pub fn of(config: &Config) -> Self {
    Self {
      policy: config.policy,
      fields: config.fields.clone(),
      quantities: config.quantities.clone(),
    }
  }

  /// The schema these tables describe.
  ///
  /// # Errors
  ///
  /// Returns an error if they no longer describe a valid schema, which a file written by this tool cannot hold.
  pub fn schema(&self) -> anyhow::Result<Schema> {
    schema_of(&self.fields, &self.quantities)
  }

  /// The policy these tables describe.
  ///
  /// # Errors
  ///
  /// As [`Self::schema`].
  pub fn policy(&self) -> anyhow::Result<Policy> {
    policy_of(&self.policy)
  }
}

/// The calibration a promotion rests on. It points at the file; the file is the record. The counts here are the
/// ones the gate compared, from the same [`Calibration`] the report was written from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRef {
  /// Where it was read from, as given.
  pub path: String,
  /// The SHA-256 of the file's text, to notice it changing later.
  pub sha256: String,
  /// The recorded run it came from, when it was given as a run directory.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub run: Option<String>,
  /// The SHA-256 of the pairs file it scored.
  pub pairs_sha256: String,
  /// The split it was restricted to.
  pub split: String,
  /// The least label confidence it required, if any.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub min_confidence: Option<String>,
  /// Labeled edits.
  pub edits: usize,
  /// Labeled recycles.
  pub recycles: usize,
  /// Edits the promoted configuration rejects.
  pub false_rejects: usize,
  /// Recycles the promoted configuration accepts.
  pub false_accepts: usize,
  /// How often an edit scores above a recycle, which no threshold enters into.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub auc: Option<f64>,
}

/// What a promoted model was compared with: the model in service at the time, on the same pairs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Comparison {
  /// The fingerprint of the model that was in service.
  pub fingerprint: String,
  /// The SHA-256 of its calibration file.
  pub calibration_sha256: String,
  /// What would have failed the comparison.
  pub fail_on: String,
  /// Edits it rejects, under the configuration it was promoted with.
  pub false_rejects: usize,
  /// Recycles it accepts.
  pub false_accepts: usize,
  /// Pairs the two decide differently.
  pub flips: usize,
}

/// A model that passed the gate under a configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Promoted {
  /// When, and by what.
  #[serde(flatten)]
  pub written: Written,
  /// The model's content digest.
  pub digest: String,
  /// Its fingerprint at the configuration's width and token limit: what `model.expected_fingerprint` names.
  pub fingerprint: String,
  /// The layout its vectors are stored under.
  pub layout: String,
  /// The model directory, as configured.
  pub model_dir: String,
  /// The collection the promotion told the operator to use.
  pub collection: String,
  /// The configuration it was judged under.
  pub config: PromotedConfig,
  /// The holdout calibration it was judged on.
  pub calibration: CalibrationRef,
  /// The comparison with the model in service. Absent for the first promotion.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub compared_with: Option<Comparison>,
  /// Why the gate was overridden, when it was.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub override_reason: Option<String>,
}

/// A model taken out of service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retired {
  /// When, and by what.
  #[serde(flatten)]
  pub written: Written,
  /// The model's content digest.
  pub digest: String,
  /// Why.
  pub reason: String,
}

/// The registry, read.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Registry {
  /// Every entry, oldest first.
  pub entries: Vec<Entry>,
}

impl Registry {
  /// Reads registry text.
  ///
  /// # Errors
  ///
  /// Returns an error naming the line if one is not an entry, or was written by a newer version of this tool:
  /// such a file is not guessed at.
  pub fn parse(text: &str) -> anyhow::Result<Self> {
    let mut entries = Vec::new();
    for (index, line) in text.lines().enumerate() {
      let number = index.saturating_add(1);
      if line.trim().is_empty() {
        continue;
      }
      let entry: Entry =
        serde_json::from_str(line).with_context(|| format!("line {number} is not a registry entry"))?;
      let format = entry.written().format;
      if format > FORMAT {
        bail!("line {number} has format {format} and this tool reads up to {FORMAT}; use the tool that wrote it");
      }
      entries.push(entry);
    }
    Ok(Self { entries })
  }

  /// Reads the registry at `path`. A file that does not exist is an empty registry.
  ///
  /// # Errors
  ///
  /// Returns an error naming the file if it cannot be read, plus anything [`Self::parse`] returns.
  pub fn read(path: &Path) -> anyhow::Result<Self> {
    match std::fs::read_to_string(path) {
      Ok(text) => Self::parse(&text).with_context(|| format!("in the registry {}", path.display())),
      Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
      Err(error) => Err(error).with_context(|| format!("cannot read the registry {}", path.display())),
    }
  }

  /// The model in service: the latest promotion of a model that has not been retired since.
  ///
  /// Retiring the model in service therefore puts the one before it back, which is what a rollback is.
  pub fn current(&self) -> Option<&Promoted> {
    self.entries.iter().enumerate().rev().find_map(|(index, entry)| {
      let Entry::Promoted(promoted) = entry else {
        return None;
      };
      let retired_since = self
        .entries
        .iter()
        .skip(index)
        .any(|later| matches!(later, Entry::Retired(retired) if retired.digest == promoted.digest));
      (!retired_since).then_some(promoted.as_ref())
    })
  }

  /// The retirement that still stands for the model `digest`: its latest one, unless it was promoted again after.
  pub fn retirement(&self, digest: &str) -> Option<&Retired> {
    self.entries.iter().rev().find_map(|entry| match entry {
      Entry::Retired(retired) if retired.digest == digest => Some(Some(retired)),
      Entry::Promoted(promoted) if promoted.digest == digest => Some(None),
      _ => None,
    })?
  }

  /// The latest registration of the model `digest`.
  pub fn registration(&self, digest: &str) -> Option<&Registered> {
    self.entries.iter().rev().find_map(|entry| match entry {
      Entry::Registered(registered) if registered.digest == digest => Some(registered.as_ref()),
      _ => None,
    })
  }

  /// Whether the model `digest` appears in any entry.
  pub fn knows(&self, digest: &str) -> bool {
    self.entries.iter().any(|entry| match entry {
      Entry::Registered(registered) => registered.digest == digest,
      Entry::Promoted(promoted) => promoted.digest == digest,
      Entry::Retired(retired) => retired.digest == digest,
    })
  }
}

/// Appends `entry` to the registry at `path` as one line, creating the file if it is not there.
///
/// The line is one write, flushed to the disk before this returns, and nothing already in the file is touched.
/// Two promotions at the same moment both compare against the same model in service; that is accepted, not
/// locked against.
///
/// # Errors
///
/// Returns an error naming the file if it cannot be opened or written.
pub fn append(path: &Path, entry: &Entry) -> anyhow::Result<()> {
  let mut line = serde_json::to_string(entry).context("cannot write a registry entry as JSON")?;
  line.push('\n');
  if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
    std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
  }
  let mut file = std::fs::OpenOptions::new()
    .create(true)
    .append(true)
    .open(path)
    .with_context(|| format!("cannot open the registry {}", path.display()))?;
  file
    .write_all(line.as_bytes())
    .and_then(|()| file.sync_all())
    .with_context(|| format!("cannot write to the registry {}", path.display()))
}

/// The SHA-256 of a file's text, in hex.
pub fn sha256(text: &str) -> String {
  hex_prefix(&Sha256::digest(text.as_bytes()), 32)
}

/// A calibration as the gate is given it: the run, and where it was read from.
#[derive(Debug, Clone, Copy)]
pub struct Evidence<'a> {
  /// The calibration, read back from its events.
  pub run: &'a Run,
  /// Where it was read from, as given.
  pub path: &'a str,
  /// The SHA-256 of the file's text.
  pub sha256: &'a str,
  /// The recorded run it came from, if it was given as a run directory.
  pub run_id: Option<&'a str>,
}

/// The model and configuration asking to be put into service.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
  /// The model's content digest.
  pub digest: &'a str,
  /// Its fingerprint at the configuration's width and token limit, as the loaded model states it.
  pub fingerprint: &'a str,
  /// The configuration: its policy, fields, and quantities are what is judged; its model directory and
  /// collection are recorded.
  pub config: &'a Config,
  /// Its holdout calibration.
  pub evidence: Evidence<'a>,
}

/// The model in service, and a calibration of it to compare against.
#[derive(Debug, Clone, Copy)]
pub struct Baseline<'a> {
  /// Its promotion.
  pub promoted: &'a Promoted,
  /// A calibration of it.
  pub evidence: Evidence<'a>,
  /// Whether that calibration was read from the path its promotion recorded, in which case it must be the file
  /// that was promoted on. One given by hand (`--baseline`) is a fresh measurement and is not held to that.
  pub recorded: bool,
}

/// What the gate is asked to hold a candidate to.
#[derive(Debug, Clone)]
pub struct GateOptions<'a> {
  /// When the gate is being asked, and by what: stamped on the entry of a candidate it lets through.
  pub written: Written,
  /// What makes the comparison with the model in service a failure.
  pub fail_on: FailOn,
  /// A reason to promote a candidate the comparison fails, or a model that was retired. It never passes a
  /// calibration of another model or of another split.
  pub override_reason: Option<&'a str>,
  /// Fixes the resampling behind the report's intervals.
  pub seed: u64,
}

/// Why a candidate is not promoted.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Refusal {
  /// The calibration is of another model than the configuration loads.
  #[error(
    "the calibration was made with the model `{calibrated}` and the configuration loads `{configured}`: calibrate \
     this model under this configuration (`calibrate --full --split holdout --ndjson`)"
  )]
  OtherModel {
    /// The fingerprint in the calibration.
    calibrated: String,
    /// The fingerprint of the model the configuration loads.
    configured: String,
  },
  /// The calibration is not restricted to the holdout split.
  #[error(
    "the calibration is of {}, and a model is promoted on its `{HOLDOUT}` split, which chose nothing: run \
     `calibrate --full --split {HOLDOUT} --ndjson`",
    .split.as_ref().map_or_else(|| "pairs of no particular split".to_owned(), |split| format!("the `{split}` split"))
  )]
  NotHoldout {
    /// The split the calibration was restricted to.
    split: Option<String>,
  },
  /// The holdout does not have both labels.
  #[error("the holdout has {edits} labeled edit(s) and {recycles} labeled recycle(s); it needs at least one of each")]
  MissingLabel {
    /// Labeled edits.
    edits: usize,
    /// Labeled recycles.
    recycles: usize,
  },
  /// The configuration cannot judge the calibration.
  #[error("{0}")]
  OtherSchema(String),
  /// The model was retired.
  #[error("the model `{digest}` was retired ({reason}); promote it again only with `--override` and the reason")]
  Retired {
    /// The model's digest.
    digest: String,
    /// Why it was retired.
    reason: String,
  },
  /// The baseline calibration is not of the model in service.
  #[error(
    "the baseline calibration was made with the model `{given}` and the model in service is `{recorded}`: \
     calibrate the model in service on the same holdout and pass it as `--baseline`"
  )]
  BaselineOtherModel {
    /// The fingerprint of the model in service.
    recorded: String,
    /// The fingerprint in the calibration that was given.
    given: String,
  },
  /// The file the model in service was promoted on is not what it was.
  #[error(
    "the calibration the model in service was promoted on, {path}, has changed since: calibrate the model in \
     service on the same holdout again and pass it as `--baseline`"
  )]
  BaselineChanged {
    /// The recorded path.
    path: String,
  },
  /// The two calibrations are not over the same selection of the same pairs.
  #[error(
    "the model in service was calibrated on another selection of pairs than the candidate, so the two cannot be \
     compared: calibrate the model in service on the candidate's holdout and pass it as `--baseline`"
  )]
  BaselineOtherPairs,
  /// The candidate does worse than the model in service.
  #[error(
    "the candidate does worse on the holdout than the model in service: {0} (the new run is the candidate, the \
     old one the model in service). Promote it anyway only with `--override` and the reason"
  )]
  Worse(String),
}

/// The comparison a promotion passed, or was overridden past.
#[derive(Debug, Clone, PartialEq)]
pub struct Compared {
  /// The model in service, judged under the configuration it was promoted with.
  pub baseline: Calibration,
  /// What differs, the model in service as the old run and the candidate as the new.
  pub diff: Diff,
  /// Why the comparison fails, when it does and was overridden.
  pub failed: Option<String>,
}

/// A candidate the gate let through.
#[derive(Debug, Clone, PartialEq)]
pub struct Approval {
  /// The candidate's holdout, judged under the candidate's configuration.
  pub calibration: Calibration,
  /// The comparison with the model in service, if there is one.
  pub compared: Option<Compared>,
  /// The layout the candidate's vectors are stored under.
  pub layout: String,
  /// The collection to use.
  pub collection: String,
  /// Whether that collection has to be filled before serving: the layout is not the one in service.
  pub new_layout: bool,
  /// The entry to append.
  pub entry: Promoted,
}

fn edits_and_recycles(calibration: &Calibration) -> (usize, usize) {
  (
    calibration.labeled(Label::Edit).count(),
    calibration.labeled(Label::Recycle).count(),
  )
}

/// A collection name with the `-<hash>` this module may have added taken off again.
fn stem(collection: &str) -> &str {
  let hex = COLLECTION_HASH_BYTES.saturating_mul(2);
  match collection.rsplit_once('-') {
    Some((stem, suffix))
      if !stem.is_empty()
        && suffix.len() == hex
        && suffix.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f')) =>
    {
      stem
    }
    _ => collection,
  }
}

/// The collection a candidate should be served from, and whether it is a new one.
///
/// A layout that is the one in service keeps the configured collection: nothing needs re-indexing (D12). A new
/// layout must not share a collection with the old one. If the configuration already names another collection
/// than the one in service, that name is kept; if it still names the one in service, the name becomes its stem
/// and twelve hex digits of the layout's hash, the same name for the same layout every time.
pub fn collection_for(configured: &str, layout: &str, in_service: Option<&Promoted>) -> (String, bool) {
  let Some(in_service) = in_service else {
    return (configured.to_owned(), false);
  };
  if in_service.layout == layout {
    return (configured.to_owned(), false);
  }
  if configured != in_service.collection {
    return (configured.to_owned(), true);
  }
  let hash = hex_prefix(&Sha256::digest(layout.as_bytes()), COLLECTION_HASH_BYTES);
  (format!("{}-{hash}", stem(configured)), true)
}

/// Decides whether `candidate` may be put into service, given the model that is (`baseline`) and any retirement
/// of the candidate that still stands.
///
/// In order, stopping at the first refusal: the calibration is of the candidate's model; it is of the holdout
/// split; it has both labels; the candidate's configuration can judge it; the model is not retired; and, with a
/// model in service, that model's calibration is of that model, is the file it was promoted on (when read from
/// the recorded path), covers the same pairs, and is not beaten by the candidate's in the way `fail_on` names.
/// `override_reason` passes only the last check and a retirement.
///
/// # Errors
///
/// Returns the [`Refusal`].
pub fn gate(
  candidate: &Candidate<'_>,
  baseline: Option<&Baseline<'_>>,
  retirement: Option<&Retired>,
  options: &GateOptions<'_>,
) -> Result<Approval, Refusal> {
  let run = candidate.evidence.run;
  if run.info.fingerprint != candidate.fingerprint {
    return Err(Refusal::OtherModel {
      calibrated: run.info.fingerprint.clone(),
      configured: candidate.fingerprint.to_owned(),
    });
  }
  let input = &run.calibration.input;
  if input.split.as_deref() != Some(HOLDOUT) {
    return Err(Refusal::NotHoldout {
      split: input.split.clone(),
    });
  }
  let (edits, recycles) = edits_and_recycles(&run.calibration);
  if edits == 0 || recycles == 0 {
    return Err(Refusal::MissingLabel { edits, recycles });
  }
  // What is promoted is the configuration. What the calibration's header says its thresholds were does not
  // matter: every pair is judged again under the candidate's own.
  let other_schema = |error: anyhow::Error| Refusal::OtherSchema(format!("{error:#}"));
  let schema = candidate.config.schema().map_err(other_schema)?;
  let policy = candidate.config.policy().map_err(other_schema)?;
  let calibration = diff::rejudge(run, &schema, &policy).map_err(other_schema)?;
  if let (Some(retired), None) = (retirement, options.override_reason) {
    return Err(Refusal::Retired {
      digest: retired.digest.clone(),
      reason: retired.reason.clone(),
    });
  }

  let compared = match baseline {
    None => None,
    Some(baseline) => Some(compare(&calibration, baseline, options)?),
  };
  let layout = schema.layout_id(candidate.fingerprint);
  let (collection, new_layout) = collection_for(
    &candidate.config.qdrant.collection,
    &layout,
    baseline.map(|baseline| baseline.promoted),
  );
  let entry = Promoted {
    written: options.written.clone(),
    digest: candidate.digest.to_owned(),
    fingerprint: candidate.fingerprint.to_owned(),
    layout: layout.clone(),
    model_dir: candidate.config.model.dir.display().to_string(),
    collection: collection.clone(),
    config: PromotedConfig::of(candidate.config),
    calibration: CalibrationRef {
      path: candidate.evidence.path.to_owned(),
      sha256: candidate.evidence.sha256.to_owned(),
      run: candidate.evidence.run_id.map(str::to_owned),
      pairs_sha256: input.sha256.clone(),
      split: HOLDOUT.to_owned(),
      min_confidence: input
        .min_confidence
        .map(|confidence| Confidence::as_str(confidence).to_owned()),
      edits,
      recycles,
      false_rejects: calibration.false_rejects,
      false_accepts: calibration.false_accepts,
      auc: stats::area_under_curve(&calibration),
    },
    compared_with: baseline.zip(compared.as_ref()).map(|(baseline, compared)| Comparison {
      fingerprint: baseline.promoted.fingerprint.clone(),
      calibration_sha256: baseline.evidence.sha256.to_owned(),
      fail_on: fail_on_name(options.fail_on).to_owned(),
      false_rejects: compared.baseline.false_rejects,
      false_accepts: compared.baseline.false_accepts,
      flips: compared.diff.flips.len(),
    }),
    override_reason: options.override_reason.map(str::to_owned),
  };
  Ok(Approval {
    calibration,
    compared,
    layout,
    collection,
    new_layout,
    entry,
  })
}

/// The comparison with the model in service: step six of [`gate`].
fn compare(candidate: &Calibration, baseline: &Baseline<'_>, options: &GateOptions<'_>) -> Result<Compared, Refusal> {
  let promoted = baseline.promoted;
  let run = baseline.evidence.run;
  if run.info.fingerprint != promoted.fingerprint {
    return Err(Refusal::BaselineOtherModel {
      recorded: promoted.fingerprint.clone(),
      given: run.info.fingerprint.clone(),
    });
  }
  if baseline.recorded && baseline.evidence.sha256 != promoted.calibration.sha256 {
    return Err(Refusal::BaselineChanged {
      path: promoted.calibration.path.clone(),
    });
  }
  if !diff::same_pairs(&run.calibration.input, &candidate.input) {
    return Err(Refusal::BaselineOtherPairs);
  }
  // The model in service is judged under the configuration it was promoted with: each model at its own
  // thresholds, which is the only way thresholds can be compared across models (D3).
  let other_schema = |error: anyhow::Error| Refusal::OtherSchema(format!("the model in service: {error:#}"));
  let schema = promoted.config.schema().map_err(other_schema)?;
  let policy = promoted.config.policy().map_err(other_schema)?;
  let judged = diff::rejudge(run, &schema, &policy).map_err(other_schema)?;
  let diff = diff::diff(&judged, candidate, diff::TOLERANCE);
  let failed = diff::failure(&diff, options.fail_on);
  if let (Some(failure), None) = (&failed, options.override_reason) {
    return Err(Refusal::Worse(failure.clone()));
  }
  Ok(Compared {
    baseline: judged,
    diff,
    failed,
  })
}

/// A gate's name, as the command line and the registry spell it.
pub fn fail_on_name(on: FailOn) -> &'static str {
  match on {
    FailOn::NewFalseAccepts => "new-false-accepts",
    FailOn::AnyFlip => "any-flip",
    FailOn::ErrorsUp => "errors-up",
    FailOn::Moved => "moved",
  }
}

fn describe_errors(calibration: &Calibration, seed: u64) -> String {
  let rates = stats::error_rates(calibration, seed);
  let auc = stats::area_under_curve(calibration).map_or_else(String::new, |auc| format!(", AUC {auc:.3}"));
  format!(
    "{} of {} edits rejected [{}], {} of {} recycles accepted [{}]{auc}",
    rates.false_reject.errors,
    rates.false_reject.of,
    describe_rate(&rates.false_reject),
    rates.false_accept.errors,
    rates.false_accept.of,
    describe_rate(&rates.false_accept),
  )
}

/// The report of a promotion, for a person: what was promoted on what evidence, what it was compared with, and
/// the lines to put in the configuration. Every number is from the judged calibrations and `offline::stats`.
pub fn render_promotion(approval: &Approval, config_path: &str, seed: u64) -> String {
  let entry = &approval.entry;
  let mut out = String::new();
  // Writing to a `String` cannot fail.
  let _ = writeln!(out, "promoted  {} ({})", entry.fingerprint, entry.model_dir);
  let _ = writeln!(out, "holdout   {}", describe_errors(&approval.calibration, seed));
  match (&approval.compared, &entry.compared_with) {
    (Some(compared), Some(recorded)) => {
      let _ = writeln!(out, "against   {} on the same pairs", recorded.fingerprint);
      let _ = writeln!(out, "          {}", describe_errors(&compared.baseline, seed));
      let _ = writeln!(
        out,
        "          {} verdict(s) differ between the two",
        compared.diff.flips.len()
      );
      if let (Some(failed), Some(reason)) = (&compared.failed, &entry.override_reason) {
        let _ = writeln!(out, "OVERRIDDEN  {failed}");
        let _ = writeln!(out, "            reason given: {reason}");
      }
    }
    _ => {
      let _ = writeln!(
        out,
        "against   nothing: no model is in service in this registry, so this is what later ones are compared with"
      );
    }
  }
  if approval
    .compared
    .as_ref()
    .is_none_or(|compared| compared.failed.is_none())
    && let Some(reason) = &entry.override_reason
  {
    let _ = writeln!(out, "override  given and recorded, though nothing needed it: {reason}");
  }
  let _ = writeln!(out, "\nAdd to {config_path}:\n");
  let _ = writeln!(out, "  [model]\n  expected_fingerprint = \"{}\"\n", entry.fingerprint);
  let note = if approval.new_layout {
    "   # a new layout: import into it before serving (README, Re-indexing)"
  } else {
    "   # unchanged: the stored vectors are the ones this model reads"
  };
  let _ = writeln!(out, "  [qdrant]\n  collection = \"{}\"{note}", entry.collection);
  out
}

/// A calibration read from disk: a file of `calibrate --ndjson`, or the directory of a recorded calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
  /// The calibration.
  pub run: Run,
  /// Where it was read from, as given.
  pub path: String,
  /// The SHA-256 of the events' text.
  pub sha256: String,
  /// The run's id, when it was read from a run directory.
  pub run_id: Option<String>,
}

impl Loaded {
  /// Reads `path`.
  ///
  /// # Errors
  ///
  /// Returns an error naming the path if it cannot be read, is a run of another kind, or does not hold a
  /// calibration's events.
  pub fn read(path: &Path) -> anyhow::Result<Self> {
    let (events, run_id) = if path.is_dir() {
      let listed = similarity_runs::open(path)?;
      if listed.manifest.kind != similarity_runs::KIND_CALIBRATE {
        bail!(
          "{} is a `{}` run; a promotion rests on a calibration",
          path.display(),
          listed.manifest.kind
        );
      }
      (path.join(similarity_runs::EVENTS_FILE), Some(listed.manifest.id))
    } else {
      (path.to_owned(), None)
    };
    let text = std::fs::read_to_string(&events).with_context(|| format!("cannot read {}", events.display()))?;
    let run = crate::offline::events::read(&text).with_context(|| format!("in {}", events.display()))?;
    Ok(Self {
      run,
      path: path.display().to_string(),
      sha256: sha256(&text),
      run_id,
    })
  }

  fn evidence(&self) -> Evidence<'_> {
    Evidence {
      run: &self.run,
      path: &self.path,
      sha256: &self.sha256,
      run_id: self.run_id.as_deref(),
    }
  }
}

/// A model, as the commands that write the registry are given it: by what loading it found.
#[derive(Debug, Clone, Copy)]
pub struct Model<'a> {
  /// Its content digest.
  pub digest: &'a str,
  /// Its directory, as given.
  pub dir: &'a str,
  /// Its own note about where it came from, if it has one.
  pub provenance: Option<&'a Provenance>,
}

/// Everything `model promote` was asked.
#[derive(Debug, Clone)]
pub struct Promotion<'a> {
  /// The registry file.
  pub registry: &'a Path,
  /// The candidate model, as loaded under `config`.
  pub model: Model<'a>,
  /// Its fingerprint under `config`.
  pub fingerprint: &'a str,
  /// The candidate configuration.
  pub config: &'a Config,
  /// The configuration's path, for the report.
  pub config_path: &'a str,
  /// The candidate's holdout calibration.
  pub calibration: &'a Path,
  /// A calibration of the model in service on the same holdout, when the one its promotion recorded will not do.
  pub baseline: Option<&'a Path>,
  /// What the gate holds the candidate to.
  pub options: GateOptions<'a>,
}

/// Puts a candidate through the gate and, if it passes, appends its promotion (and its registration, if the
/// model is new to the registry). Returns the report.
///
/// # Errors
///
/// Returns the gate's [`Refusal`], or an error naming whatever could not be read or written. Nothing is appended
/// on any error.
pub fn promote(promotion: &Promotion<'_>) -> anyhow::Result<String> {
  let registry = Registry::read(promotion.registry)?;
  let candidate = Loaded::read(promotion.calibration)?;
  let in_service = registry.current();
  let baseline = match (in_service, promotion.baseline) {
    (None, Some(path)) => bail!(
      "`--baseline {}` was given and no model is in service in {}: there is nothing for it to be a baseline of",
      path.display(),
      promotion.registry.display()
    ),
    (None, None) => None,
    (Some(promoted), Some(path)) => Some((promoted, Loaded::read(path)?, false)),
    (Some(promoted), None) => {
      let recorded = Loaded::read(Path::new(&promoted.calibration.path)).with_context(|| {
        format!(
          "the calibration the model in service ({}) was promoted on cannot be read; calibrate that model on the \
           same holdout and pass it as `--baseline`",
          promoted.fingerprint
        )
      })?;
      Some((promoted, recorded, true))
    }
  };
  let approval = gate(
    &Candidate {
      digest: promotion.model.digest,
      fingerprint: promotion.fingerprint,
      config: promotion.config,
      evidence: candidate.evidence(),
    },
    baseline
      .as_ref()
      .map(|(promoted, loaded, recorded)| Baseline {
        promoted,
        evidence: loaded.evidence(),
        recorded: *recorded,
      })
      .as_ref(),
    registry.retirement(promotion.model.digest),
    &promotion.options,
  )?;
  if registry.registration(promotion.model.digest).is_none() {
    register_entry(promotion.registry, &promotion.model, &promotion.options.written)?;
  }
  append(promotion.registry, &Entry::Promoted(Box::new(approval.entry.clone())))?;
  Ok(render_promotion(
    &approval,
    promotion.config_path,
    promotion.options.seed,
  ))
}

fn register_entry(path: &Path, model: &Model<'_>, written: &Written) -> anyhow::Result<()> {
  append(
    path,
    &Entry::Registered(Box::new(Registered {
      written: written.clone(),
      digest: model.digest.to_owned(),
      dir: model.dir.to_owned(),
      provenance: model.provenance.cloned(),
    })),
  )
}

/// Makes a model directory known to the registry. Returns whether an entry was appended: a model already
/// registered at the same path is not registered again.
///
/// Nothing is written into the model directory (D18).
///
/// # Errors
///
/// Returns an error naming the registry if it cannot be read or written.
pub fn register(path: &Path, model: &Model<'_>, written: &Written) -> anyhow::Result<bool> {
  let registry = Registry::read(path)?;
  if registry
    .registration(model.digest)
    .is_some_and(|registered| registered.dir == model.dir)
  {
    return Ok(false);
  }
  register_entry(path, model, written)?;
  Ok(true)
}

/// Takes the model `digest` out of service, with the reason. If it is the model in service, the one promoted
/// before it is in service again.
///
/// # Errors
///
/// Returns an error if the registry does not know the model, or cannot be read or written.
pub fn retire(path: &Path, digest: &str, reason: &str, written: &Written) -> anyhow::Result<()> {
  let registry = Registry::read(path)?;
  if !registry.knows(digest) {
    bail!(
      "the registry {} has no model `{digest}`; `model list` shows the ones it has",
      path.display()
    );
  }
  if reason.trim().is_empty() {
    bail!("a retirement needs a reason");
  }
  append(
    path,
    &Entry::Retired(Retired {
      written: written.clone(),
      digest: digest.to_owned(),
      reason: reason.to_owned(),
    }),
  )
}

/// One model of the registry, with everything recorded about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Listed<'a> {
  /// The model's content digest.
  pub digest: &'a str,
  /// Its latest registration, if it was ever registered by itself or by a promotion.
  pub registered: Option<&'a Registered>,
  /// Its promotions, oldest first.
  pub promotions: Vec<&'a Promoted>,
  /// The retirement that still stands, if one does.
  pub retired: Option<&'a Retired>,
  /// Whether it is the model in service.
  pub in_service: bool,
}

/// Every model the registry knows, in the order it first appears.
pub fn list(registry: &Registry) -> Vec<Listed<'_>> {
  let current = registry.current();
  let mut digests: Vec<&str> = Vec::new();
  for entry in &registry.entries {
    let digest = match entry {
      Entry::Registered(registered) => registered.digest.as_str(),
      Entry::Promoted(promoted) => promoted.digest.as_str(),
      Entry::Retired(retired) => retired.digest.as_str(),
    };
    if !digests.contains(&digest) {
      digests.push(digest);
    }
  }
  digests
    .into_iter()
    .map(|digest| Listed {
      digest,
      registered: registry.registration(digest),
      promotions: registry
        .entries
        .iter()
        .filter_map(|entry| match entry {
          Entry::Promoted(promoted) if promoted.digest == digest => Some(promoted.as_ref()),
          _ => None,
        })
        .collect(),
      retired: registry.retirement(digest),
      in_service: current.is_some_and(|current| current.digest == digest),
    })
    .collect()
}

/// The registry as a listing, for a person.
pub fn render_list(registry: &Registry) -> String {
  let mut out = String::new();
  let models = list(registry);
  if models.is_empty() {
    out.push_str("the registry is empty\n");
    return out;
  }
  let current = registry.current();
  for model in models {
    let dir = model.registered.map_or("-", |registered| registered.dir.as_str());
    let origin = model
      .registered
      .and_then(|registered| registered.provenance.as_ref())
      .map_or_else(String::new, |provenance| format!("  written by {}", provenance.tool));
    // Writing to a `String` cannot fail.
    let _ = writeln!(out, "{}  {dir}{origin}", model.digest);
    for promoted in &model.promotions {
      let in_service = current.is_some_and(|current| std::ptr::eq(current, *promoted));
      let _ = writeln!(
        out,
        "  promoted {} at {}  collection {}  holdout: {} of {} edits rejected, {} of {} recycles accepted{}{}",
        promoted.fingerprint,
        similarity_runs::utc_stamp(promoted.written.at),
        promoted.collection,
        promoted.calibration.false_rejects,
        promoted.calibration.edits,
        promoted.calibration.false_accepts,
        promoted.calibration.recycles,
        promoted
          .override_reason
          .as_ref()
          .map_or_else(String::new, |reason| format!("  overridden: {reason}")),
        if in_service { "  << in service" } else { "" },
      );
    }
    if let Some(retired) = model.retired {
      let _ = writeln!(
        out,
        "  retired at {}: {}",
        similarity_runs::utc_stamp(retired.written.at),
        retired.reason
      );
    }
  }
  out
}

#[cfg(test)]
pub(crate) mod testing {
  //! Calibrations and entries made by hand, for the tests here and the properties.

  use super::*;
  use crate::offline::events;

  /// A build that never existed.
  pub(crate) fn written(at: u64) -> Written {
    Written {
      format: FORMAT,
      at,
      build: Stamp {
        tool: "record-similarity model promote".to_owned(),
        version: "0.0.0".to_owned(),
        commit: "test".to_owned(),
        dirty: None,
      },
    }
  }

  /// A configuration of two equally weighted fields at `threshold`, serving from `collection`.
  pub(crate) fn config(threshold: f32, collection: &str) -> Config {
    Config::parse(&format!(
      "[qdrant]\ncollection = \"{collection}\"\n[policy]\nthreshold = {threshold}\n[fields.name]\nweight = 1.0\n\
       [fields.category]\nweight = 1.0\n"
    ))
    .unwrap()
  }

  /// The text of a calibration of the model `fingerprint` over `pairs`: each an id, a label, and the similarity
  /// both of its fields scored, so that under equal weights that is its overall similarity too.
  pub(crate) fn calibration(
    fingerprint: &str,
    split: Option<&str>,
    pairs_sha: &str,
    pairs: &[(&str, &str, f64)],
  ) -> String {
    let split = split.map_or_else(String::new, |split| format!(r#""split":"{split}","#));
    let mut lines = vec![
      format!(
        r#"{{"message":"calibration","fingerprint":"{fingerprint}","dimension":8,"max_tokens":16,"threshold":0.5,"pairs_sha256":"{pairs_sha}","pairs":{},{split}"seed":0,"target":"result"}}"#,
        pairs.len()
      ),
      r#"{"message":"schema_field","field":"category","weight":1.0,"required":false,"target":"result"}"#.to_owned(),
      r#"{"message":"schema_field","field":"name","weight":1.0,"required":false,"target":"result"}"#.to_owned(),
    ];
    for (id, label, similarity) in pairs {
      lines.push(format!(
        r#"{{"message":"pair","pair_id":"{id}","label":"{label}","similarity":{similarity},"accepted":true,"wrong":false,"measure_changed":false,"target":"result"}}"#
      ));
      for field in ["category", "name"] {
        lines.push(format!(
          r#"{{"message":"pair_field","pair_id":"{id}","field":"{field}","change":"modified","similarity":{similarity},"weight":1.0,"below_floor":false,"target":"result"}}"#
        ));
      }
    }
    lines.join("\n")
  }

  /// Four holdout pairs that a threshold between 0.7 and 0.8 gets all right.
  pub(crate) const PAIRS: [(&str, &str, f64); 4] = [
    ("e1", "edit", 0.9),
    ("e2", "edit", 0.8),
    ("r1", "recycle", 0.3),
    ("r2", "recycle", 0.7),
  ];

  /// A calibration held as the gate wants it.
  pub(crate) struct Held {
    pub(crate) run: Run,
    pub(crate) sha256: String,
    pub(crate) path: String,
  }

  impl Held {
    pub(crate) fn of(text: &str, path: &str) -> Self {
      Self {
        run: events::read(text).unwrap(),
        sha256: sha256(text),
        path: path.to_owned(),
      }
    }

    pub(crate) fn evidence(&self) -> Evidence<'_> {
      Evidence {
        run: &self.run,
        path: &self.path,
        sha256: &self.sha256,
        run_id: None,
      }
    }
  }

  pub(crate) fn options(override_reason: Option<&str>) -> GateOptions<'_> {
    GateOptions {
      written: written(100),
      fail_on: FailOn::ErrorsUp,
      override_reason,
      seed: 0,
    }
  }

  /// A promotion of the model `digest` under [`config`] at `threshold`, by the gate itself with nothing in
  /// service.
  pub(crate) fn promoted(digest: &str, threshold: f32, held: &Held) -> Promoted {
    let config = config(threshold, "records");
    let fingerprint = held.run.info.fingerprint.clone();
    let candidate = Candidate {
      digest,
      fingerprint: &fingerprint,
      config: &config,
      evidence: held.evidence(),
    };
    gate(&candidate, None, None, &options(None)).unwrap().entry
  }
}

#[cfg(test)]
mod tests {
  use super::testing::{Held, PAIRS, calibration, config, options, promoted, written};
  use super::*;

  fn holdout(fingerprint: &str, pairs: &[(&str, &str, f64)]) -> Held {
    Held::of(&calibration(fingerprint, Some(HOLDOUT), "aa", pairs), "holdout.ndjson")
  }

  fn candidate<'a>(config: &'a Config, held: &'a Held) -> Candidate<'a> {
    Candidate {
      digest: "new",
      fingerprint: "new-d8-t16",
      config,
      evidence: held.evidence(),
    }
  }

  fn retired(digest: &str, at: u64) -> Retired {
    Retired {
      written: written(at),
      digest: digest.to_owned(),
      reason: "worse in production".to_owned(),
    }
  }

  fn registered(digest: &str) -> Registered {
    Registered {
      written: written(1),
      digest: digest.to_owned(),
      dir: format!("models/{digest}"),
      provenance: None,
    }
  }

  #[test]
  fn a_missing_registry_is_empty_and_has_nothing_in_service() {
    let directory = tempfile::tempdir().unwrap();
    let registry = Registry::read(&directory.path().join("models").join(REGISTRY_FILE)).unwrap();
    assert_eq!(registry, Registry::default());
    assert!(registry.current().is_none());
    assert!(!registry.knows("anything"));
    assert_eq!(render_list(&registry), "the registry is empty\n");

    // Only a file that is not there is an empty registry. One that is there and cannot be read is an error: a
    // promotion that took it for empty would compare with nothing and record a first promotion.
    let unreadable = Registry::read(directory.path()).unwrap_err();
    assert!(
      format!("{unreadable:#}").contains("cannot read the registry"),
      "{unreadable:#}"
    );
  }

  #[test]
  fn entries_read_back_as_they_were_appended_and_earlier_lines_are_left_byte_for_byte() {
    let directory = tempfile::tempdir().unwrap();
    // The directory is created with the first entry.
    let path = directory.path().join("models").join(REGISTRY_FILE);
    let held = holdout("old-d8-t16", &PAIRS);
    let entries = [
      Entry::Registered(Box::new(registered("old"))),
      Entry::Promoted(Box::new(promoted("old", 0.75, &held))),
      Entry::Retired(retired("old", 200)),
    ];
    let mut before = String::new();
    for entry in &entries {
      append(&path, entry).unwrap();
      let now = std::fs::read_to_string(&path).unwrap();
      assert!(now.starts_with(&before), "an append rewrote what was there");
      assert_eq!(now.lines().count(), before.lines().count() + 1, "one entry, one line");
      assert!(now.ends_with('\n'));
      before = now;
    }
    assert_eq!(Registry::read(&path).unwrap().entries, entries);
    // The kind of entry is the first thing on its line, and what is absent is left out, not written as null.
    assert!(
      before.starts_with(r#"{"entry":"registered","format":1,"at":1,"tool":"#),
      "{before}"
    );
    assert!(!before.contains("null"), "{before}");
    for absent in [
      "provenance",
      "compared_with",
      "override_reason",
      "dirty",
      "min_confidence",
      "anchor_threshold",
    ] {
      assert!(
        !before.contains(absent),
        "`{absent}` was written though it is absent: {before}"
      );
    }
  }

  #[test]
  fn a_line_that_is_not_an_entry_is_an_error_naming_its_line_number() {
    let good = serde_json::to_string(&Entry::Retired(retired("old", 5))).unwrap();
    for bad in ["{\"entry\":\"deleted\"}", "not json", "{\"entry\":\"retired\"}"] {
      let error = Registry::parse(&format!("{good}\n\n{bad}\n")).unwrap_err();
      assert!(format!("{error:#}").contains("line 3"), "{error:#}");
    }
    assert_eq!(Registry::parse(&format!("{good}\n\n")).unwrap().entries.len(), 1);
  }

  #[test]
  fn an_entry_of_a_newer_format_is_refused_rather_than_guessed_at() {
    let newer = serde_json::to_string(&Entry::Retired(retired("old", 5)))
      .unwrap()
      .replace("\"format\":1", "\"format\":2");
    let error = Registry::parse(&newer).unwrap_err();
    assert!(format!("{error:#}").contains("format 2"), "{error:#}");
    // A field this version does not know is not a newer format: fields are added, never repurposed.
    let extended = serde_json::to_string(&Entry::Retired(retired("old", 5)))
      .unwrap()
      .replace("\"digest\"", "\"added_later\":true,\"digest\"");
    assert_eq!(Registry::parse(&extended).unwrap().entries.len(), 1);
  }

  /// What "in service" means, and that a retirement is how a rollback is made.
  #[test]
  fn the_model_in_service_is_the_latest_promotion_of_a_model_not_retired_since() {
    let promotion = |digest: &str| {
      let held = holdout(&format!("{digest}-d8-t16"), &PAIRS);
      Entry::Promoted(Box::new(promoted(digest, 0.75, &held)))
    };
    let mut registry = Registry::default();
    let in_service = |registry: &Registry| registry.current().map(|promoted| promoted.digest.clone());

    registry.entries.push(Entry::Registered(Box::new(registered("a"))));
    assert_eq!(in_service(&registry), None, "registering puts nothing into service");
    registry.entries.push(promotion("a"));
    assert_eq!(in_service(&registry).as_deref(), Some("a"));
    registry.entries.push(promotion("b"));
    assert_eq!(in_service(&registry).as_deref(), Some("b"));
    assert!(registry.retirement("b").is_none());

    registry.entries.push(Entry::Retired(retired("b", 300)));
    assert_eq!(in_service(&registry).as_deref(), Some("a"), "the one before is back");
    assert_eq!(registry.retirement("b").map(|retired| retired.written.at), Some(300));
    assert!(registry.retirement("a").is_none());

    registry.entries.push(Entry::Retired(retired("a", 400)));
    assert_eq!(in_service(&registry), None, "everything is retired");

    registry.entries.push(promotion("b"));
    assert_eq!(in_service(&registry).as_deref(), Some("b"));
    assert!(
      registry.retirement("b").is_none(),
      "promoted again, it is no longer retired"
    );
    assert!(registry.retirement("a").is_some());

    let listed = list(&registry);
    let summary: Vec<_> = listed
      .iter()
      .map(|model| {
        (
          model.digest,
          model.promotions.len(),
          model.retired.is_some(),
          model.in_service,
        )
      })
      .collect();
    assert_eq!(summary, [("a", 1, true, false), ("b", 2, false, true)]);
    assert!(listed[0].registered.is_some() && listed[1].registered.is_none());
    let text = render_list(&registry);
    assert_eq!(text.matches("<< in service").count(), 1, "{text}");
    assert!(
      text.contains("a  models/a") && text.contains("retired at 19700101T000640Z: worse in production"),
      "{text}"
    );
  }

  #[test]
  fn a_calibration_of_another_model_is_refused_whatever_else_is_true() {
    let config = config(0.75, "records");
    let held = holdout("other-d8-t16", &PAIRS);
    for reason in [None, Some("I know what I am doing")] {
      let refusal = gate(&candidate(&config, &held), None, None, &options(reason)).unwrap_err();
      assert_eq!(
        refusal,
        Refusal::OtherModel {
          calibrated: "other-d8-t16".to_owned(),
          configured: "new-d8-t16".to_owned(),
        }
      );
    }
  }

  #[test]
  fn a_calibration_that_is_not_of_the_holdout_split_is_refused_whatever_else_is_true() {
    let config = config(0.75, "records");
    for split in [Some("validation"), Some("calibrate"), None] {
      let held = Held::of(&calibration("new-d8-t16", split, "aa", &PAIRS), "pairs.ndjson");
      for reason in [None, Some("the holdout is too small")] {
        let refusal = gate(&candidate(&config, &held), None, None, &options(reason)).unwrap_err();
        assert_eq!(
          refusal,
          Refusal::NotHoldout {
            split: split.map(str::to_owned)
          }
        );
        assert!(refusal.to_string().contains("--split holdout"), "{refusal}");
      }
    }
  }

  #[test]
  fn a_holdout_without_both_labels_is_refused() {
    let config = config(0.75, "records");
    let edits_only = holdout("new-d8-t16", &PAIRS[..2]);
    assert_eq!(
      gate(&candidate(&config, &edits_only), None, None, &options(Some("anyway"))).unwrap_err(),
      Refusal::MissingLabel { edits: 2, recycles: 0 }
    );
    let recycles_only = holdout("new-d8-t16", &PAIRS[2..]);
    assert_eq!(
      gate(&candidate(&config, &recycles_only), None, None, &options(None)).unwrap_err(),
      Refusal::MissingLabel { edits: 0, recycles: 2 }
    );
  }

  /// What is promoted is the configuration. The calibration was run at a threshold of 0.5 and marked every pair
  /// accepted; the gate judges each pair again under the candidate's own threshold.
  #[test]
  fn the_candidate_is_judged_under_its_own_configuration_not_the_one_the_calibration_ran_with() {
    let held = holdout("new-d8-t16", &PAIRS);
    let errors = |threshold: f32| {
      let config = config(threshold, "records");
      let approval = gate(&candidate(&config, &held), None, None, &options(None)).unwrap();
      let entry = &approval.entry.calibration;
      assert_eq!(
        (entry.false_rejects, entry.false_accepts),
        (approval.calibration.false_rejects, approval.calibration.false_accepts)
      );
      assert_eq!((entry.edits, entry.recycles, entry.auc), (2, 2, Some(1.0)));
      (entry.false_rejects, entry.false_accepts)
    };
    assert_eq!(errors(0.75), (0, 0));
    assert_eq!(errors(0.85), (1, 0), "the edit at 0.8 is rejected");
    assert_eq!(errors(0.6), (0, 1), "the recycle at 0.7 is accepted");
  }

  #[test]
  fn a_configuration_with_other_fields_than_the_calibration_is_refused() {
    let held = holdout("new-d8-t16", &PAIRS);
    let config = Config::parse("[policy]\nthreshold = 0.75\n[fields.name]\nweight = 1.0\n").unwrap();
    let refusal = gate(&candidate(&config, &held), None, None, &options(None)).unwrap_err();
    assert!(
      matches!(&refusal, Refusal::OtherSchema(text) if text.contains("category")),
      "{refusal}"
    );
  }

  #[test]
  fn the_first_promotion_has_nothing_to_compare_with_and_records_what_it_was_promoted_on() {
    let held = holdout("new-d8-t16", &PAIRS);
    let config = config(0.75, "records");
    let approval = gate(&candidate(&config, &held), None, None, &options(None)).unwrap();
    assert!(approval.compared.is_none());
    assert_eq!((approval.collection.as_str(), approval.new_layout), ("records", false));
    let entry = &approval.entry;
    assert_eq!(entry.written, written(100));
    assert_eq!(
      (entry.digest.as_str(), entry.fingerprint.as_str()),
      ("new", "new-d8-t16")
    );
    assert_eq!(entry.layout, config.schema().unwrap().layout_id("new-d8-t16"));
    assert_eq!(entry.layout, approval.layout);
    assert_eq!(entry.model_dir, "models/mdbr-leaf-mt");
    assert_eq!(entry.config, PromotedConfig::of(&config));
    assert_eq!(entry.config.policy().unwrap(), config.policy().unwrap());
    assert_eq!(entry.config.schema().unwrap(), config.schema().unwrap());
    assert_eq!(
      (
        entry.calibration.path.as_str(),
        entry.calibration.sha256.as_str(),
        entry.calibration.pairs_sha256.as_str(),
        entry.calibration.split.as_str(),
        &entry.calibration.run,
      ),
      ("holdout.ndjson", held.sha256.as_str(), "aa", HOLDOUT, &None)
    );
    assert_eq!((&entry.compared_with, &entry.override_reason), (&None, &None));

    let report = render_promotion(&approval, "candidate.toml", 0);
    assert!(report.contains("against   nothing"), "{report}");
    assert!(
      report.contains("0 of 2 edits rejected") && report.contains("0 of 2 recycles accepted"),
      "{report}"
    );
  }

  /// The model in service: `old`, promoted at 0.75 on a holdout where it gets one edit wrong.
  fn in_service() -> (Held, Promoted) {
    let pairs = [PAIRS[0], ("e2", "edit", 0.7), PAIRS[2], ("r2", "recycle", 0.6)];
    let held = holdout("old-d8-t16", &pairs);
    let entry = promoted("old", 0.75, &held);
    assert_eq!(
      (entry.calibration.false_rejects, entry.calibration.false_accepts),
      (1, 0)
    );
    (held, entry)
  }

  fn baseline<'a>(held: &'a Held, promoted: &'a Promoted, recorded: bool) -> Baseline<'a> {
    Baseline {
      promoted,
      evidence: held.evidence(),
      recorded,
    }
  }

  #[test]
  fn a_candidate_that_is_no_worse_is_promoted_and_the_comparison_is_recorded() {
    let (old, promoted) = in_service();
    let held = holdout("new-d8-t16", &PAIRS);
    let config = config(0.75, "records");
    let approval = gate(
      &candidate(&config, &held),
      Some(&baseline(&old, &promoted, true)),
      None,
      &options(None),
    )
    .unwrap();
    let compared = approval.compared.as_ref().unwrap();
    assert_eq!((compared.baseline.false_rejects, compared.failed.as_ref()), (1, None));
    assert_eq!(
      approval.entry.compared_with,
      Some(Comparison {
        fingerprint: "old-d8-t16".to_owned(),
        calibration_sha256: old.sha256.clone(),
        fail_on: "errors-up".to_owned(),
        false_rejects: 1,
        false_accepts: 0,
        flips: 1,
      })
    );
    let report = render_promotion(&approval, "candidate.toml", 0);
    assert!(report.contains("against   old-d8-t16 on the same pairs"), "{report}");
    assert!(report.contains("1 of 2 edits rejected"), "{report}");
    assert!(report.contains("1 verdict(s) differ"), "{report}");
    assert!(
      !report.contains("OVERRIDDEN") && !report.contains("override"),
      "{report}"
    );
  }

  /// A candidate is judged at its own thresholds and the model in service at the ones it was promoted with.
  /// Here the candidate's are too strict for it: one more pair wrong, and it is refused unless someone says why.
  #[test]
  fn a_candidate_that_gets_more_pairs_wrong_is_refused_unless_overridden_with_a_reason() {
    let pairs = [PAIRS[0], PAIRS[1], PAIRS[2], ("r2", "recycle", 0.6)];
    let old = holdout("old-d8-t16", &pairs);
    let promoted = promoted("old", 0.75, &old);
    assert_eq!(promoted.calibration.false_rejects, 0);
    let held = holdout("new-d8-t16", &PAIRS);
    let strict = config(0.85, "records");

    let ask = |reason| {
      gate(
        &candidate(&strict, &held),
        Some(&baseline(&old, &promoted, true)),
        None,
        &options(reason),
      )
    };
    let refusal = ask(None).unwrap_err();
    assert!(
      matches!(&refusal, Refusal::Worse(text) if text.contains("1 pair(s) wrong and the old one 0")),
      "{refusal}"
    );
    assert!(refusal.to_string().contains("--override"), "{refusal}");

    let approval = ask(Some("the old model's vendor is gone")).unwrap();
    assert_eq!(
      approval.entry.override_reason.as_deref(),
      Some("the old model's vendor is gone")
    );
    assert!(approval.compared.as_ref().unwrap().failed.is_some());
    let report = render_promotion(&approval, "candidate.toml", 0);
    assert!(
      report.contains("OVERRIDDEN  the new run gets 1 pair(s) wrong"),
      "{report}"
    );
    assert!(
      report.contains("reason given: the old model's vendor is gone"),
      "{report}"
    );
  }

  #[test]
  fn a_baseline_that_cannot_be_compared_is_refused_and_no_reason_overrides_that() {
    let (old, promoted) = in_service();
    let held = holdout("new-d8-t16", &PAIRS);
    let config = config(0.75, "records");
    let ask = |baseline: &Baseline<'_>| {
      gate(
        &candidate(&config, &held),
        Some(baseline),
        None,
        &options(Some("anyway")),
      )
      .unwrap_err()
    };

    // Of another model than the one in service.
    let stranger = holdout("stranger-d8-t16", &PAIRS);
    assert_eq!(
      ask(&baseline(&stranger, &promoted, false)),
      Refusal::BaselineOtherModel {
        recorded: "old-d8-t16".to_owned(),
        given: "stranger-d8-t16".to_owned(),
      }
    );

    // The recorded file is not what it was when the model was promoted on it. The same file given by hand is a
    // fresh measurement, and is taken.
    let changed = holdout("old-d8-t16", &[PAIRS[0], PAIRS[1], PAIRS[2], ("r2", "recycle", 0.6)]);
    assert_ne!(changed.sha256, promoted.calibration.sha256);
    assert_eq!(
      ask(&baseline(&changed, &promoted, true)),
      Refusal::BaselineChanged {
        path: "holdout.ndjson".to_owned()
      }
    );
    assert!(
      gate(
        &candidate(&config, &held),
        Some(&baseline(&changed, &promoted, false)),
        None,
        &options(None)
      )
      .is_ok()
    );

    // Over another pairs file, or another selection of the same one.
    let grown = Held::of(&calibration("old-d8-t16", Some(HOLDOUT), "bb", &PAIRS), "later.ndjson");
    let refusal = ask(&baseline(&grown, &promoted, false));
    assert_eq!(refusal, Refusal::BaselineOtherPairs);
    assert!(refusal.to_string().contains("--baseline"), "{refusal}");
    let _ = old;
  }

  #[test]
  fn a_retired_model_is_refused_unless_overridden() {
    let held = holdout("new-d8-t16", &PAIRS);
    let config = config(0.75, "records");
    let retirement = retired("new", 50);
    let refusal = gate(&candidate(&config, &held), None, Some(&retirement), &options(None)).unwrap_err();
    assert_eq!(
      refusal,
      Refusal::Retired {
        digest: "new".to_owned(),
        reason: "worse in production".to_owned(),
      }
    );
    let approval = gate(
      &candidate(&config, &held),
      None,
      Some(&retirement),
      &options(Some("the regression was the importer's")),
    )
    .unwrap();
    let report = render_promotion(&approval, "candidate.toml", 0);
    assert!(report.contains("override  given and recorded"), "{report}");
    assert_eq!(
      approval.entry.override_reason.as_deref(),
      Some("the regression was the importer's")
    );
  }

  /// D12: a change of thresholds or weights needs no new collection, and a change of model always does.
  #[test]
  fn the_collection_is_kept_for_the_layout_in_service_and_is_a_new_one_for_any_other() {
    let (_, promoted) = in_service();
    assert_eq!(promoted.collection, "records");
    let new_layout = config(0.75, "records").schema().unwrap().layout_id("new-d8-t16");

    assert_eq!(
      collection_for("records", &promoted.layout, Some(&promoted)),
      ("records".to_owned(), false)
    );
    assert_eq!(
      collection_for("anything", "any layout", None),
      ("anything".to_owned(), false)
    );

    let (named, is_new) = collection_for("records", &new_layout, Some(&promoted));
    assert!(is_new);
    let hash = named.strip_prefix("records-").unwrap();
    assert_eq!(hash.len(), 12);
    assert!(
      hash
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
      "{named}"
    );
    assert_eq!(
      collection_for("records", &new_layout, Some(&promoted)).0,
      named,
      "the same layout, the same name"
    );
    assert_ne!(collection_for("records", "another layout", Some(&promoted)).0, named);

    // A name that already carries a hash does not grow a second one.
    let hashed = Promoted {
      collection: named.clone(),
      ..promoted.clone()
    };
    let (next, _) = collection_for(&named, "another layout", Some(&hashed));
    assert!(
      next.starts_with("records-") && next.len() == named.len() && next != named,
      "{next}"
    );
    assert_eq!(stem("records-v2"), "records-v2", "only twelve hex digits are a hash");
    assert_eq!(stem("-0123456789ab"), "-0123456789ab");

    // An operator who already named another collection keeps the name, and is still told to fill it.
    assert_eq!(
      collection_for("records-2027", &new_layout, Some(&promoted)),
      ("records-2027".to_owned(), true)
    );
  }

  /// The gate, with a model in service and a new layout: the collection it names is in the entry and in the
  /// lines it prints, and those lines are a configuration.
  #[test]
  fn a_promotion_prints_lines_that_parse_as_configuration_and_name_what_was_promoted() {
    let (old, promoted) = in_service();
    let held = holdout("new-d8-t16", &PAIRS);
    let config = config(0.75, "records");
    let approval = gate(
      &candidate(&config, &held),
      Some(&baseline(&old, &promoted, true)),
      None,
      &options(None),
    )
    .unwrap();
    assert!(approval.new_layout);
    assert_eq!(approval.entry.collection, approval.collection);
    assert_ne!(approval.collection, "records");

    let report = render_promotion(&approval, "candidate.toml", 0);
    let fragment = report.split("Add to candidate.toml:\n").nth(1).unwrap();
    let parsed = Config::parse(&format!(
      "{fragment}\n[policy]\nthreshold = 0.75\n[fields.name]\nweight = 1.0\n"
    ))
    .unwrap();
    assert_eq!(parsed.model.expected_fingerprint.as_deref(), Some("new-d8-t16"));
    assert_eq!(parsed.qdrant.collection, approval.collection);
    assert!(fragment.contains("import into it before serving"), "{fragment}");
    assert!(
      report.starts_with("promoted  new-d8-t16 (models/mdbr-leaf-mt)\n"),
      "{report}"
    );
  }

  #[test]
  fn a_gate_is_named_as_the_command_line_spells_it() {
    let names: Vec<&str> = [
      FailOn::ErrorsUp,
      FailOn::NewFalseAccepts,
      FailOn::AnyFlip,
      FailOn::Moved,
    ]
    .into_iter()
    .map(fail_on_name)
    .collect();
    assert_eq!(names, ["errors-up", "new-false-accepts", "any-flip", "moved"]);
  }

  /// The whole command, on files: a first promotion, a second model compared with it through the path the
  /// first one recorded, and a refusal that leaves the file exactly as it was.
  #[test]
  fn promoting_appends_a_registration_and_a_promotion_and_a_refusal_appends_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join(REGISTRY_FILE);
    let write = |name: &str, text: &str| {
      let file = directory.path().join(name);
      std::fs::write(&file, text).unwrap();
      file
    };
    let old_holdout = write("old.ndjson", &calibration("old-d8-t16", Some(HOLDOUT), "aa", &PAIRS));
    let new_holdout = write("new.ndjson", &calibration("new-d8-t16", Some(HOLDOUT), "aa", &PAIRS));
    let validation = write(
      "validation.ndjson",
      &calibration("new-d8-t16", Some("validation"), "aa", &PAIRS),
    );
    let config = config(0.75, "records");
    let strict = super::testing::config(0.85, "records");
    let ask = |digest: &'static str, fingerprint: &'static str, config: &Config, calibration: &Path| {
      let dir = format!("models/{digest}");
      promote(&Promotion {
        registry: &path,
        model: Model {
          digest,
          dir: &dir,
          provenance: None,
        },
        fingerprint,
        config,
        config_path: "candidate.toml",
        calibration,
        baseline: None,
        options: options(None),
      })
    };

    let first = ask("old", "old-d8-t16", &config, &old_holdout).unwrap();
    assert!(first.contains("against   nothing"), "{first}");
    let registry = Registry::read(&path).unwrap();
    assert!(matches!(
      registry.entries.as_slice(),
      [Entry::Registered(registered), Entry::Promoted(promoted)]
        if registered.dir == "models/old" && promoted.digest == "old"
    ));

    // Refused: not the holdout, and then worse than the model in service. Neither leaves a trace.
    let before = std::fs::read_to_string(&path).unwrap();
    let refused = ask("new", "new-d8-t16", &config, &validation).unwrap_err();
    assert!(refused.downcast_ref::<Refusal>().is_some(), "{refused:#}");
    let worse = ask("new", "new-d8-t16", &strict, &new_holdout).unwrap_err();
    assert!(matches!(worse.downcast_ref(), Some(Refusal::Worse(_))), "{worse:#}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);

    let second = ask("new", "new-d8-t16", &config, &new_holdout).unwrap();
    assert!(second.contains("against   old-d8-t16 on the same pairs"), "{second}");
    let registry = Registry::read(&path).unwrap();
    assert_eq!(registry.entries.len(), 4);
    assert_eq!(registry.current().map(|promoted| promoted.digest.as_str()), Some("new"));
    assert!(std::fs::read_to_string(&path).unwrap().starts_with(&before));

    // The same model again is not registered again, and its promotion is compared with itself in service.
    ask("new", "new-d8-t16", &config, &new_holdout).unwrap();
    assert_eq!(Registry::read(&path).unwrap().entries.len(), 5);

    // The recorded calibration is the baseline only while it is the file that was promoted on.
    std::fs::write(
      &new_holdout,
      calibration("new-d8-t16", Some(HOLDOUT), "aa", &PAIRS[..3]),
    )
    .unwrap();
    let changed = ask("old", "old-d8-t16", &config, &old_holdout).unwrap_err();
    assert!(
      matches!(changed.downcast_ref(), Some(Refusal::BaselineChanged { .. })),
      "{changed:#}"
    );
    std::fs::remove_file(&new_holdout).unwrap();
    let gone = ask("old", "old-d8-t16", &config, &old_holdout).unwrap_err();
    assert!(format!("{gone:#}").contains("--baseline"), "{gone:#}");
  }

  #[test]
  fn a_model_is_registered_once_at_a_path_and_retired_only_if_known_and_with_a_reason() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join(REGISTRY_FILE);
    let model = |dir: &'static str| Model {
      digest: "abc",
      dir,
      provenance: None,
    };
    assert!(register(&path, &model("models/abc"), &written(1)).unwrap());
    assert!(
      !register(&path, &model("models/abc"), &written(2)).unwrap(),
      "the same path again"
    );
    assert!(
      register(&path, &model("elsewhere/abc"), &written(3)).unwrap(),
      "it moved"
    );
    let registry = Registry::read(&path).unwrap();
    assert_eq!(registry.entries.len(), 2);
    assert_eq!(
      registry.registration("abc").map(|registered| registered.dir.as_str()),
      Some("elsewhere/abc")
    );

    assert!(retire(&path, "nobody", "a reason", &written(4)).is_err());
    assert!(retire(&path, "abc", "  ", &written(4)).is_err());
    assert_eq!(
      Registry::read(&path).unwrap().entries.len(),
      2,
      "a refused retirement appends nothing"
    );
    retire(&path, "abc", "superseded", &written(5)).unwrap();
    let registry = Registry::read(&path).unwrap();
    assert_eq!(
      registry.retirement("abc").map(|retired| retired.reason.as_str()),
      Some("superseded")
    );
  }

  /// A recorded calibration is given as its run directory, and the entry remembers which run it was.
  #[test]
  fn a_calibration_is_read_from_a_file_or_from_the_directory_of_a_recorded_run() {
    let directory = tempfile::tempdir().unwrap();
    let text = calibration("new-d8-t16", Some(HOLDOUT), "aa", &PAIRS);
    let file = directory.path().join("holdout.ndjson");
    std::fs::write(&file, &text).unwrap();
    let from_file = Loaded::read(&file).unwrap();
    assert_eq!(
      (from_file.run_id.as_deref(), from_file.sha256.as_str()),
      (None, sha256(&text).as_str())
    );
    assert_eq!(from_file.sha256.len(), 64);

    let run = similarity_runs::Run::start(similarity_runs::Spec {
      root: directory.path(),
      kind: similarity_runs::KIND_CALIBRATE,
      slug: "holdout".to_owned(),
      build: written(1).build,
      argv: vec!["record-similarity".to_owned(), "calibrate".to_owned()],
      continues: None,
    })
    .unwrap();
    let (id, dir) = (run.id().to_owned(), run.dir().to_owned());
    std::fs::write(dir.join(similarity_runs::EVENTS_FILE), &text).unwrap();
    run.finish(Ok(())).unwrap();
    let from_run = Loaded::read(&dir).unwrap();
    assert_eq!(from_run.run_id.as_deref(), Some(id.as_str()));
    assert_eq!((from_run.sha256, from_run.run), (from_file.sha256, from_file.run));

    assert!(Loaded::read(&directory.path().join("missing.ndjson")).is_err());
    assert!(Loaded::read(directory.path()).is_err(), "a directory that is not a run");
  }
}
