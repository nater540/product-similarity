//! Where a model directory came from: `provenance.json`.
//!
//! A model's fingerprint says *which* model produced a vector. It does not say how the model came to be: which
//! published model it was fetched as, or which base, data, and options it was trained from. After a few rounds of
//! fine-tuning nobody can reconstruct that, and it matters, because thresholds belong to one specific model (D3).
//! So whatever writes a model directory also writes this file into it, at the moment the model is written, which
//! is the only moment the answer is known.
//!
//! The file records **what produced the model**, never what the model scored: a model directory is written whole
//! and not touched again, and the scores come later.
//!
//! It is not part of the model. The content digest covers only the files the encoder is built from
//! ([`crate::model::sentence::ModelFiles::digest`]), so adding this file to a directory, or lacking it, changes
//! no fingerprint and invalidates no stored vector. Nothing refuses a model for not having one.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The file's name inside a model directory.
pub const PROVENANCE_FILE: &str = "provenance.json";

/// The version of the file's layout that this crate writes.
pub const FORMAT: u32 = 1;

/// What a model directory records about its origin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
  /// The layout version, [`FORMAT`] for files written by this crate.
  pub format: u32,
  /// When the directory was written, in seconds since the Unix epoch.
  pub written_at: u64,
  /// The program that wrote it (`record-similarity model fetch`, `record-similarity-train`).
  pub tool: String,
  /// That program's version.
  pub version: String,
  /// The source commit that program was built from, or `unknown`.
  pub commit: String,
  /// Whether the source tree had uncommitted changes when it was built. Absent when that is not known.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub dirty: Option<bool>,
  /// The content digest of the model in this directory: the first part of its fingerprint.
  pub digest: String,
  /// The id of the recorded run that wrote the directory, if the program was recording one. It names what
  /// produced the model, like everything else here, and a run's results are found under that id, not in this file.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub run: Option<String>,
  /// Set for a model that was downloaded.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub fetched: Option<Fetched>,
  /// Set for a model that was trained.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub trained: Option<Trained>,
}

/// The origin of a downloaded model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fetched {
  /// The Hub repository.
  pub repo: String,
  /// The revision that was downloaded.
  pub revision: String,
  /// Every file that was downloaded.
  pub files: Vec<FetchedFile>,
}

/// One downloaded file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchedFile {
  /// The file's path inside the model directory.
  pub name: String,
  /// The SHA-256 of what is on disk.
  pub sha256: String,
  /// Whether that was checked against a checksum pinned in the program.
  pub verified: bool,
}

/// The origin of a trained model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trained {
  /// The model training started from. For a run that was resumed, still the model the run first started from.
  pub base: Base,
  /// The files the examples were read from.
  pub data: Vec<DataFile>,
  /// A fingerprint of the examples themselves, in order: what a resumed run is checked against.
  pub data_fingerprint: String,
  /// The training options, as the trainer serializes them.
  pub options: serde_json::Value,
  /// The token limit texts were truncated to.
  pub max_tokens: usize,
  /// Where training ran (`cpu`, `metal`, `cuda`).
  pub device: String,
  /// The checkpoints the run was continued from, oldest first. Empty for a run that was never interrupted.
  #[serde(default)]
  pub resumed: Vec<Resumed>,
  /// How many optimizer steps the weights in this directory have had.
  pub step: usize,
  /// How many steps the whole run has. A directory where `step` is lower is a checkpoint.
  pub total_steps: usize,
}

/// The model a training run started from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Base {
  /// Where it was.
  pub dir: String,
  /// Its content digest.
  pub digest: String,
  /// Its own provenance, if it had one, so that the chain back to a published model is in this one file.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub provenance: Option<Box<Provenance>>,
}

/// One file of training data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataFile {
  /// What it was used as (`pairs`, `catalog_groups`).
  pub role: String,
  /// Where it was.
  pub path: String,
  /// The SHA-256 of its text.
  pub sha256: String,
  /// How many training examples came from it.
  pub examples: usize,
}

/// One point at which a run was continued from a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resumed {
  /// The checkpoint directory.
  pub from: String,
  /// The step it held.
  pub step: usize,
  /// The id of the recorded run that wrote that checkpoint, if one did.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub run: Option<String>,
}

/// A program's build, as it is written down: in a model's provenance (flat, as [`Provenance`]'s first fields) and
/// in a run's manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stamp {
  /// The program, as a person would name it.
  pub tool: String,
  /// Its version.
  pub version: String,
  /// The source commit it was built from, or `unknown`.
  pub commit: String,
  /// Whether the source tree had uncommitted changes then. Absent when that is not known.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub dirty: Option<bool>,
}

/// What a program knows about itself when it writes a model directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Build {
  /// The program, as a person would name it.
  pub tool: &'static str,
  /// Its version.
  pub version: &'static str,
  /// The source commit it was built from, or `unknown`.
  pub commit: &'static str,
  /// Whether the source tree had uncommitted changes then, if that is known.
  pub dirty: Option<bool>,
}

impl Build {
  /// A build from what the two binaries' build scripts hand them: the commit, and `true`, `false`, or nothing
  /// for whether the tree was dirty.
  ///
  /// A `const fn`, so that each binary can state its build once as a constant.
  pub const fn new(tool: &'static str, version: &'static str, commit: &'static str, dirty: &str) -> Self {
    Self {
      tool,
      version,
      commit,
      dirty: match dirty.as_bytes() {
        b"true" => Some(true),
        b"false" => Some(false),
        _ => None,
      },
    }
  }

  /// This build, in the form that is written down.
  pub fn stamp(&self) -> Stamp {
    Stamp {
      tool: self.tool.to_owned(),
      version: self.version.to_owned(),
      commit: self.commit.to_owned(),
      dirty: self.dirty,
    }
  }

  /// The provenance of a model with content digest `digest`, written by this build now. The caller adds how the
  /// model was made.
  pub fn provenance(&self, digest: impl Into<String>) -> Provenance {
    Provenance {
      format: FORMAT,
      written_at: std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs()),
      tool: self.tool.to_owned(),
      version: self.version.to_owned(),
      commit: self.commit.to_owned(),
      dirty: self.dirty,
      digest: digest.into(),
      run: None,
      fetched: None,
      trained: None,
    }
  }
}

impl Provenance {
  /// Reads the provenance of the model in `dir`. `Ok(None)` if the directory has none, which is the case for
  /// every model written before this file existed and for any model put together by hand.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Io`] if the file exists and cannot be read, and [`Error::Json`] if it is not a provenance
  /// file this crate understands.
  pub fn read(dir: &Path) -> Result<Option<Self>> {
    let path = dir.join(PROVENANCE_FILE);
    let bytes = match std::fs::read(&path) {
      Ok(bytes) => bytes,
      Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
      Err(error) => return Err(Error::io(&path, error)),
    };
    serde_json::from_slice(&bytes)
      .map(Some)
      .map_err(|source| Error::json(&path, source))
  }

  /// Writes this provenance into `dir`, replacing any that is there.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Json`] if it does not serialize and [`Error::Io`] if the file cannot be written.
  pub fn write(&self, dir: &Path) -> Result<()> {
    let path = dir.join(PROVENANCE_FILE);
    let mut json = serde_json::to_vec_pretty(self).map_err(|source| Error::json(&path, source))?;
    json.push(b'\n');
    std::fs::write(&path, json).map_err(|source| Error::io(&path, source))
  }
}

#[cfg(test)]
mod tests {
  use sha2::{Digest as _, Sha256};

  use super::*;
  use crate::embed::{Embedder, EmbedderOptions};
  use crate::model::SentenceEncoder;
  use crate::testing::write_tiny_model;

  fn options() -> EmbedderOptions {
    EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    }
  }

  fn fetched(digest: &str) -> Provenance {
    Provenance {
      format: FORMAT,
      written_at: 1_700_000_000,
      tool: "record-similarity model fetch".to_owned(),
      version: "0.1.0".to_owned(),
      commit: "0123456789ab".to_owned(),
      dirty: Some(false),
      digest: digest.to_owned(),
      run: None,
      fetched: Some(Fetched {
        repo: "MongoDB/mdbr-leaf-mt".to_owned(),
        revision: "1ed41b2".to_owned(),
        files: vec![FetchedFile {
          name: "model.safetensors".to_owned(),
          sha256: "ab".repeat(32),
          verified: true,
        }],
      }),
      trained: None,
    }
  }

  #[test]
  fn a_directory_without_the_file_has_no_provenance_and_that_is_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(Provenance::read(dir.path()).unwrap(), None);
  }

  #[test]
  fn provenance_reads_back_as_written_with_its_whole_chain() {
    let dir = tempfile::tempdir().unwrap();
    let trained = Provenance {
      tool: "record-similarity-train".to_owned(),
      dirty: None,
      digest: "fedcba9876543210".to_owned(),
      fetched: None,
      trained: Some(Trained {
        base: Base {
          dir: "models/mdbr-leaf-mt".to_owned(),
          digest: "0123456789abcdef".to_owned(),
          provenance: Some(Box::new(fetched("0123456789abcdef"))),
        },
        data: vec![DataFile {
          role: "pairs".to_owned(),
          path: "pairs.full.jsonl".to_owned(),
          sha256: "cd".repeat(32),
          examples: 4120,
        }],
        data_fingerprint: "00ff".to_owned(),
        options: serde_json::json!({"epochs": 2, "learning_rate": 2e-5}),
        max_tokens: 256,
        device: "metal".to_owned(),
        resumed: vec![Resumed {
          from: "models/x.checkpoints/step-000120".to_owned(),
          step: 120,
          run: Some("20261111T120002Z-train-x".to_owned()),
        }],
        step: 240,
        total_steps: 240,
      }),
      ..fetched("")
    };
    trained.write(dir.path()).unwrap();
    assert_eq!(Provenance::read(dir.path()).unwrap().as_ref(), Some(&trained));

    let text = std::fs::read_to_string(dir.path().join(PROVENANCE_FILE)).unwrap();
    assert!(text.ends_with("}\n"), "a text file ends with a newline");
    let written: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(
      written.get("dirty").is_none() && written.get("fetched").is_none(),
      "what is not known is left out, not written as null: {text}"
    );
    let base = &trained.trained.as_ref().unwrap().base;
    assert_eq!(
      base.provenance.as_ref().unwrap().fetched.as_ref().unwrap().repo,
      "MongoDB/mdbr-leaf-mt",
      "the chain back to the published model is in the one file"
    );
  }

  #[test]
  fn a_build_stamps_what_it_knows_about_itself() {
    let build = Build::new("record-similarity-train", "0.1.0", "0123456789ab", "true");
    let provenance = build.provenance("fedcba9876543210");
    assert_eq!(
      (provenance.tool.as_str(), provenance.commit.as_str(), provenance.dirty),
      ("record-similarity-train", "0123456789ab", Some(true))
    );
    assert_eq!(
      (provenance.format, provenance.digest.as_str()),
      (FORMAT, "fedcba9876543210")
    );
    assert!(provenance.written_at > 1_700_000_000);
    assert!(provenance.fetched.is_none() && provenance.trained.is_none());

    assert_eq!(Build::new("t", "v", "c", "false").dirty, Some(false));
    assert_eq!(
      Build::new("t", "v", "unknown", "").dirty,
      None,
      "a build that could not ask git does not claim a clean tree"
    );
  }

  /// Runs came later than the file. One written before them has no `run` key anywhere, and must still read.
  #[test]
  fn a_provenance_file_written_before_runs_existed_reads_with_no_run() {
    let dir = tempfile::tempdir().unwrap();
    let old = concat!(
      r#"{"format": 1, "written_at": 1700000000, "tool": "record-similarity-train", "version": "0.1.0", "#,
      r#""commit": "0123456789ab", "digest": "aaaa", "trained": {"base": {"dir": "m", "digest": "bbbb"}, "#,
      r#""data": [], "data_fingerprint": "cccc", "options": {}, "max_tokens": 256, "device": "cpu", "#,
      r#""resumed": [{"from": "m.checkpoints/step-000010", "step": 10}], "step": 20, "total_steps": 20}}"#,
    );
    std::fs::write(dir.path().join(PROVENANCE_FILE), old).unwrap();
    let read = Provenance::read(dir.path()).unwrap().unwrap();
    assert_eq!(read.run, None);
    assert_eq!(read.trained.unwrap().resumed[0].run, None);

    // And a file with no run is written without the key, so an older reader is not shown one either.
    let written = serde_json::to_string(&fetched("aaaa")).unwrap();
    assert!(!written.contains("\"run\""), "{written}");
  }

  #[test]
  fn a_build_stamp_carries_what_the_provenance_fields_carry() {
    let build = Build::new("record-similarity-train", "0.1.0", "0123456789ab", "true");
    let (stamp, provenance) = (build.stamp(), build.provenance("aaaa"));
    assert_eq!(
      (&stamp.tool, &stamp.version, &stamp.commit, stamp.dirty),
      (
        &provenance.tool,
        &provenance.version,
        &provenance.commit,
        provenance.dirty
      )
    );
    let unknown = Build::new("x", "0.1.0", "unknown", "").stamp();
    assert!(
      !serde_json::to_string(&unknown).unwrap().contains("dirty"),
      "an unknown is left out"
    );
  }

  #[test]
  fn a_file_that_is_not_provenance_is_an_error_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(PROVENANCE_FILE), "{\"format\": 1}").unwrap();
    let error = Provenance::read(dir.path()).unwrap_err().to_string();
    assert!(error.contains(PROVENANCE_FILE), "{error}");
  }

  #[test]
  fn the_digest_covers_the_files_the_encoder_reads_and_the_tokenizer_and_nothing_else() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_model(dir.path(), Some(24));
    let encoder = SentenceEncoder::load(dir.path(), &crate::Device::flex()).unwrap();
    let files = encoder.files();

    // Computed again here, independently: each file's length, then its bytes, in the order they were read.
    let mut hasher = Sha256::new();
    for path in files
      .read
      .iter()
      .cloned()
      .chain([dir.path().join(crate::tokenize::TOKENIZER_FILE)])
    {
      let bytes = std::fs::read(&path).unwrap();
      hasher.update(u64::try_from(bytes.len()).unwrap().to_le_bytes());
      hasher.update(&bytes);
    }
    let expected = crate::record::hex_prefix(&hasher.finalize(), 8);
    let digest = files.digest().unwrap();
    assert_eq!(digest, expected);
    assert_eq!(digest.len(), 16);

    // It is the first part of the fingerprint the embedder reports.
    let fingerprint = |dir: &Path| {
      Embedder::load(dir, crate::Device::flex(), &options())
        .unwrap()
        .fingerprint()
        .to_owned()
    };
    let before = fingerprint(dir.path());
    assert!(before.starts_with(&format!("{digest}-")), "{before}");

    // A provenance file, or any other note, is not the model.
    fetched(&digest).write(dir.path()).unwrap();
    std::fs::write(dir.path().join("README.md"), "notes").unwrap();
    assert_eq!(files.digest().unwrap(), digest);
    assert_eq!(fingerprint(dir.path()), before);

    // A byte of the tokenizer is.
    let tokenizer = dir.path().join(crate::tokenize::TOKENIZER_FILE);
    let mut text = std::fs::read_to_string(&tokenizer).unwrap();
    text.push(' ');
    std::fs::write(&tokenizer, text).unwrap();
    assert_ne!(files.digest().unwrap(), digest);
  }

  #[test]
  fn a_saved_copy_is_digested_where_it_was_written() {
    let base = tempfile::tempdir().unwrap();
    write_tiny_model(base.path(), None);
    let encoder = SentenceEncoder::load(base.path(), &crate::Device::flex()).unwrap();
    fetched("x").write(base.path()).unwrap();

    let copy = tempfile::tempdir().unwrap();
    encoder.save(copy.path()).unwrap();
    let digest = encoder.files().rebased(copy.path()).digest().unwrap();
    let loaded = Embedder::load(copy.path(), crate::Device::flex(), &options()).unwrap();
    assert!(loaded.fingerprint().starts_with(&format!("{digest}-")));
    assert_eq!(
      Provenance::read(copy.path()).unwrap(),
      None,
      "saving a model does not carry the base's provenance along as if it were the copy's"
    );
  }
}
