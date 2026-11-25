//! Downloading a model directory from the Hugging Face Hub.
//!
//! The server itself only ever takes a path: it has no network code on its model-loading path and starts the same
//! way in an air-gapped cluster as on a desk. This module is the one place a URL appears, and it runs only when a
//! human or an image build asks for it (`record-similarity model fetch`).
//!
//! Downloads are pinned to an immutable revision (a commit hash, never `main`), and for the default model every
//! file's SHA-256 is compiled into the binary. A model is 90 MB of numbers nobody can review by eye, so "what we
//! tested is what we run" has to be checked by a machine.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};
use similarity_core::Provenance;
use similarity_core::model::Pipeline;
use similarity_core::provenance::{self, Build, Fetched};

/// The default model's repository on the Hugging Face Hub.
pub const DEFAULT_REPO: &str = "MongoDB/mdbr-leaf-mt";
/// The commit of [`DEFAULT_REPO`] that the parity fixture and the calibration were produced with.
pub const DEFAULT_REVISION: &str = "1ed41b22ce166d66c24f88ebfc340e1f03adb20f";
/// Where `model fetch` puts the default model, and where the default configuration looks for it.
pub const DEFAULT_DIR: &str = "models/mdbr-leaf-mt";

/// Where `model fetch` puts `repo` when it is not told: `models/<the repository's name>`.
///
/// One directory per repository, so that fetching a second model cannot land on top of the first. A directory
/// holding the files of one model and the name of another would load without complaint, under a configuration and
/// thresholds that were made for the other. For [`DEFAULT_REPO`] this is [`DEFAULT_DIR`].
///
/// # Errors
///
/// Returns an error if `repo` does not end in a name that can be a directory.
pub fn default_dir(repo: &str) -> anyhow::Result<PathBuf> {
  let name = repo.rsplit('/').next().unwrap_or_default();
  let usable = !name.is_empty()
    && name != "."
    && name != ".."
    && name
      .chars()
      .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'));
  if !usable {
    bail!("cannot derive a directory from the repository `{repo}`; expected `owner/name`, or pass --dir");
  }
  Ok(Path::new("models").join(name))
}

/// SHA-256 of every file of [`DEFAULT_REPO`] at [`DEFAULT_REVISION`].
const DEFAULT_CHECKSUMS: [(&str, &str); 7] = [
  (
    "modules.json",
    "9e6e1d1856b2c2fbf3ba7005fb5fc215c49ff6f0a0f917ec9e2110aca9f30ed6",
  ),
  (
    "config.json",
    "258407cfa27c7e1e502a9c312abebb4f5241bf226972ffc366287f370a3567f1",
  ),
  (
    "tokenizer.json",
    "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
  ),
  (
    "model.safetensors",
    "08a4fce05c16baee8b6ca143d1ae11535ababb05412f506e7a89a8b56916a6a5",
  ),
  (
    "1_Pooling/config.json",
    "b4105bf48c5257a542a7e1968cef220b3ecd49ae24943143279650c4ac6e30a0",
  ),
  (
    "2_Dense/config.json",
    "5d4010b4ce519411f3d09a8eb2c757c18877a727c29a53be0c23f53ab3c951a1",
  ),
  (
    "2_Dense/model.safetensors",
    "dfe95933b75110ca0c1650dc0a78f06d0a05a028892ac74ffc5aa3644283f16f",
  ),
];

/// What to download and where to put it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchRequest {
  /// The Hub repository, `owner/name`.
  pub repo: String,
  /// The revision to download. Should be a commit hash.
  pub revision: String,
  /// The directory to write into. Created if missing.
  pub dir: PathBuf,
}

/// One downloaded (or already present and verified) file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedFile {
  /// The path relative to the model directory.
  pub name: String,
  /// Its SHA-256, in hex.
  pub sha256: String,
  /// Whether the digest was checked against a pinned value, or merely computed.
  pub verified: bool,
  /// Whether the file was already on disk with the right digest and the download was skipped.
  pub cached: bool,
}

/// The files a sentence-transformers pipeline needs, derived from its `modules.json`.
///
/// The pipeline is checked by the same code the encoder is loaded with ([`Pipeline`]), so a model is refused here,
/// before its weights are downloaded, exactly when the server would refuse to load it. That check also confines
/// every file to the model directory, which matters because `modules.json` is itself a downloaded file.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or describes a pipeline the encoder does not implement.
pub fn files_for(modules_json: &[u8]) -> anyhow::Result<Vec<String>> {
  Ok(Pipeline::parse(modules_json, Path::new("modules.json"))?.files())
}

/// What a fetched model directory records about its origin: the repository, the revision, and every file with
/// the digest of what is on disk and whether that was checked against a pinned checksum.
///
/// `digest` is the content digest of the model as it loads (see `Embedder::content_digest`).
pub fn provenance(request: &FetchRequest, files: &[FetchedFile], digest: &str, build: &Build) -> Provenance {
  let mut provenance = build.provenance(digest);
  provenance.fetched = Some(Fetched {
    repo: request.repo.clone(),
    revision: request.revision.clone(),
    files: files
      .iter()
      .map(|file| provenance::FetchedFile {
        name: file.name.clone(),
        sha256: file.sha256.clone(),
        verified: file.verified,
      })
      .collect(),
  });
  provenance
}

fn pinned_checksum(request: &FetchRequest, name: &str) -> Option<&'static str> {
  if request.repo != DEFAULT_REPO || request.revision != DEFAULT_REVISION {
    return None;
  }
  DEFAULT_CHECKSUMS
    .iter()
    .find(|(file, _)| *file == name)
    .map(|(_, digest)| *digest)
}

pub(crate) fn hex(digest: &[u8]) -> String {
  use std::fmt::Write as _;
  digest.iter().fold(String::new(), |mut hex, byte| {
    // Writing to a `String` cannot fail.
    let _ = write!(hex, "{byte:02x}");
    hex
  })
}

pub(crate) fn sha256_of_file(path: &Path) -> anyhow::Result<String> {
  let mut file = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
  let mut hasher = Sha256::new();
  std::io::copy(&mut file, &mut hasher).with_context(|| format!("cannot read {}", path.display()))?;
  Ok(hex(&hasher.finalize()))
}

/// Streams `url` into `destination` through a temporary file, returning the SHA-256 of what was written.
pub(crate) fn download(url: &str, destination: &Path) -> anyhow::Result<String> {
  if let Some(parent) = destination.parent() {
    std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
  }
  // Written next to the destination and renamed into place, so an interrupted download never leaves a truncated
  // file under the real name.
  let partial = destination.with_extension("part");
  let mut response = ureq::get(url).call().with_context(|| format!("GET {url} failed"))?;
  let mut reader = response.body_mut().as_reader();
  let mut file = std::fs::File::create(&partial).with_context(|| format!("cannot create {}", partial.display()))?;
  let mut hasher = Sha256::new();
  let mut buffer = vec![0_u8; 1 << 16];
  loop {
    let read = reader
      .read(&mut buffer)
      .with_context(|| format!("download of {url} was interrupted"))?;
    let Some(chunk) = buffer.get(..read).filter(|chunk| !chunk.is_empty()) else {
      break;
    };
    hasher.update(chunk);
    file
      .write_all(chunk)
      .with_context(|| format!("cannot write {}", partial.display()))?;
  }
  file
    .sync_all()
    .with_context(|| format!("cannot flush {}", partial.display()))?;
  std::fs::rename(&partial, destination).with_context(|| format!("cannot move {} into place", partial.display()))?;
  Ok(hex(&hasher.finalize()))
}

/// Downloads every file of a model into `request.dir`, verifying checksums where they are pinned.
///
/// Files already present with the pinned digest are not downloaded again, so re-running after an interruption only
/// fetches what is missing. `progress` is called once per file, after it is in place.
///
/// # Errors
///
/// Returns an error if a download fails, the model uses an unsupported module, or a pinned checksum does not match
/// (in which case the offending file is removed).
pub fn fetch(request: &FetchRequest, mut progress: impl FnMut(&FetchedFile)) -> anyhow::Result<Vec<FetchedFile>> {
  let base = format!("https://huggingface.co/{}/resolve/{}", request.repo, request.revision);
  let mut fetched = Vec::new();

  let mut fetch_one = |name: &str| -> anyhow::Result<()> {
    let destination = request.dir.join(name);
    let pinned = pinned_checksum(request, name);

    let cached = match pinned {
      Some(expected) if destination.is_file() => sha256_of_file(&destination)? == expected,
      _ => false,
    };
    let sha256 = if cached {
      pinned.unwrap_or_default().to_owned()
    } else {
      download(&format!("{base}/{name}"), &destination)?
    };
    if let Some(expected) = pinned
      && sha256 != expected
    {
      // Best effort: the mismatch is the error worth reporting, not a failure to clean up after it.
      let _ = std::fs::remove_file(&destination);
      bail!("checksum mismatch for {name}: expected {expected}, got {sha256}; the file was removed");
    }

    let file = FetchedFile {
      name: name.to_owned(),
      sha256,
      verified: pinned.is_some(),
      cached,
    };
    progress(&file);
    fetched.push(file);
    Ok(())
  };

  fetch_one("modules.json")?;
  let modules_path = request.dir.join("modules.json");
  let modules = std::fs::read(&modules_path).with_context(|| format!("cannot read {}", modules_path.display()))?;
  for name in files_for(&modules)?
    .iter()
    .filter(|name| name.as_str() != "modules.json")
  {
    fetch_one(name)?;
  }
  Ok(fetched)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_fetched_model_records_its_repository_revision_and_every_file() {
    let request = FetchRequest {
      repo: "MongoDB/mdbr-leaf-mt".to_owned(),
      revision: DEFAULT_REVISION.to_owned(),
      dir: PathBuf::from("models/mdbr-leaf-mt"),
    };
    let file = |name: &str, verified: bool| FetchedFile {
      name: name.to_owned(),
      sha256: "ab".repeat(32),
      verified,
      cached: !verified,
    };
    let files = [file("model.safetensors", true), file("tokenizer.json", false)];
    let build = Build::new("record-similarity model fetch", "0.1.0", "0123456789ab", "false");

    let written = provenance(&request, &files, "f46e48ee0ea2dd02", &build);
    assert_eq!(
      (written.tool.as_str(), written.digest.as_str(), written.dirty),
      ("record-similarity model fetch", "f46e48ee0ea2dd02", Some(false))
    );
    assert!(written.trained.is_none());
    let fetched = written.fetched.as_ref().unwrap();
    assert_eq!(
      (fetched.repo.as_str(), fetched.revision.as_str()),
      ("MongoDB/mdbr-leaf-mt", DEFAULT_REVISION)
    );
    let recorded: Vec<(&str, bool)> = fetched
      .files
      .iter()
      .map(|file| (file.name.as_str(), file.verified))
      .collect();
    assert_eq!(recorded, [("model.safetensors", true), ("tokenizer.json", false)]);

    // And it survives the trip to disk beside the model.
    let dir = tempfile::tempdir().unwrap();
    written.write(dir.path()).unwrap();
    assert_eq!(Provenance::read(dir.path()).unwrap(), Some(written));
  }

  const LEAF_MODULES: &str = r#"[
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"}
  ]"#;

  #[test]
  fn the_file_list_follows_the_pipeline() {
    let files = files_for(LEAF_MODULES.as_bytes()).unwrap();
    assert_eq!(
      files,
      [
        "modules.json",
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "1_Pooling/config.json",
        "2_Dense/config.json",
        "2_Dense/model.safetensors"
      ]
    );
  }

  #[test]
  fn every_file_of_the_default_model_has_a_pinned_checksum() {
    let request = FetchRequest {
      repo: DEFAULT_REPO.to_owned(),
      revision: DEFAULT_REVISION.to_owned(),
      dir: PathBuf::new(),
    };
    for name in files_for(LEAF_MODULES.as_bytes()).unwrap() {
      let digest = pinned_checksum(&request, &name).unwrap_or_else(|| panic!("{name} is not pinned"));
      assert_eq!(digest.len(), 64);
    }
  }

  #[test]
  fn checksums_apply_only_to_the_exact_pinned_revision() {
    let other_revision = FetchRequest {
      repo: DEFAULT_REPO.to_owned(),
      revision: "main".to_owned(),
      dir: PathBuf::new(),
    };
    assert_eq!(pinned_checksum(&other_revision, "config.json"), None);
    let other_repo = FetchRequest {
      repo: "thenlper/gte-small".to_owned(),
      revision: DEFAULT_REVISION.to_owned(),
      dir: PathBuf::new(),
    };
    assert_eq!(pinned_checksum(&other_repo, "config.json"), None);
  }

  #[test]
  fn a_normalize_module_needs_no_files_and_an_unknown_module_is_refused() {
    let with_normalize = r#"[
      {"path": "", "type": "sentence_transformers.models.Transformer"},
      {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
      {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
    ]"#;
    assert_eq!(files_for(with_normalize.as_bytes()).unwrap().len(), 5);

    let with_lstm = r#"[
      {"path": "", "type": "sentence_transformers.models.Transformer"},
      {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
      {"path": "2_LSTM", "type": "sentence_transformers.models.LSTM"}
    ]"#;
    assert!(
      files_for(with_lstm.as_bytes())
        .unwrap_err()
        .to_string()
        .contains("LSTM")
    );
    assert!(files_for(b"not json").is_err());
  }

  #[test]
  fn a_downloaded_module_list_cannot_name_files_outside_the_model_directory() {
    let escaping = r#"[
      {"path": "", "type": "sentence_transformers.models.Transformer"},
      {"path": "../../elsewhere", "type": "sentence_transformers.models.Pooling"}
    ]"#;
    let error = files_for(escaping.as_bytes()).unwrap_err().to_string();
    assert!(error.contains("not a subdirectory"), "{error}");
  }

  #[test]
  fn each_repository_gets_its_own_default_directory() {
    assert_eq!(default_dir(DEFAULT_REPO).unwrap(), Path::new(DEFAULT_DIR));
    assert_eq!(
      default_dir("sentence-transformers/all-MiniLM-L6-v2").unwrap(),
      Path::new("models/all-MiniLM-L6-v2")
    );
    for odd in ["", "owner/", "owner/..", "owner/a b"] {
      assert!(default_dir(odd).is_err(), "{odd:?}");
    }
  }

  #[test]
  fn a_file_digest_is_its_sha256() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("abc.txt");
    std::fs::write(&path, b"abc").unwrap();
    assert_eq!(
      sha256_of_file(&path).unwrap(),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
  }
}
