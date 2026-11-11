//! A local record of what the tools ran: one directory per run, as plain files.
//!
//! A training run or a calibration used to leave behind only what someone remembered to redirect. A recorded run
//! leaves a directory under the runs root (`runs/` unless told otherwise):
//!
//! ```text
//! runs/20261111T142301Z-train-mdbr-leaf-mt-ft/
//!   run.json        the manifest: what ran, with what, on what, and how it ended
//!   events.ndjson   every result and failure, and logs at `info`, in the order emitted
//!   report.txt      a calibration's text report, when that is what stdout got
//!   .lock           held by the process for as long as it runs
//! ```
//!
//! Nothing is sent anywhere. The directory is the whole record, and it is made of the same JSON lines the tools
//! already write, so the calibration reader and `calibrate-diff` read a run's `events.ndjson` as it is.
//!
//! Three rules shape the code:
//!
//! - **Everything that could stop a run from being recorded is found out in [`Run::start`]**, before any work.
//!   After that, noting something in the manifest never fails the work it describes: it warns and goes on.
//! - **A manifest is never seen half written.** It is written beside itself and renamed into place.
//! - **`killed` is never written, because a killed process writes nothing.** The stored status is `running`,
//!   `finished`, or `failed`. A reader concludes `killed` from a stored `running` whose lock nobody holds; the
//!   operating system releases the lock however the process ends.
//!
//! This crate records. It does not decide what a run means: comparing runs, reporting on them, and gating on them
//! belong to their callers.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use similarity_core::provenance::{DataFile, Stamp};

#[cfg(test)]
mod properties;

/// The manifest of a run, inside its directory.
pub const MANIFEST_FILE: &str = "run.json";
/// Everything the run emitted, inside its directory.
pub const EVENTS_FILE: &str = "events.ndjson";
/// A calibration's text report, inside its directory.
pub const REPORT_FILE: &str = "report.txt";
/// The file whose lock says the run's process is alive.
pub const LOCK_FILE: &str = ".lock";
/// The manifest layout this crate writes and the newest it reads.
pub const FORMAT: u32 = 1;
/// The kind of a fine-tuning run.
pub const KIND_TRAIN: &str = "train";
/// The kind of a calibration.
pub const KIND_CALIBRATE: &str = "calibrate";

/// How long a slug may be, in bytes. With the time and the kind in front, an id stays well inside a file name.
const SLUG_BYTES: usize = 40;
/// How many runs may start in one second with one kind and slug before that is taken for a mistake.
const SAME_SECOND: u32 = 99;

/// Why a run could not be recorded or read.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
  /// A file or directory could not be created, read, or written.
  #[error("{}: {source}", path.display())]
  Io {
    /// What was being touched.
    path: PathBuf,
    /// What went wrong.
    source: std::io::Error,
  },
  /// A manifest is not one this crate can read.
  #[error("{}: {reason}", path.display())]
  Manifest {
    /// The manifest file.
    path: PathBuf,
    /// What is wrong with it.
    reason: String,
  },
  /// Every directory name a run could take was already there.
  #[error("{} runs named `{id}` already started this second under {}", SAME_SECOND, root.display())]
  NoFreeId {
    /// The runs root.
    root: PathBuf,
    /// The id without a suffix.
    id: String,
  },
}

impl RunError {
  fn io(path: &Path, source: std::io::Error) -> Self {
    Self::Io {
      path: path.to_owned(),
      source,
    }
  }
}

/// A result of this crate.
pub type Result<T> = std::result::Result<T, RunError>;

/// What a manifest says about how its run stands. `killed` is never stored: see [`Status`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stored {
  /// Started and not yet ended, as far as the file knows.
  Running,
  /// Ended with its work done.
  Finished,
  /// Ended with an error, which the manifest carries.
  Failed,
}

/// How a run stands, as a reader concludes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
  /// Its process is alive.
  Running,
  /// It ended with its work done.
  Finished,
  /// It ended with an error.
  Failed,
  /// Its manifest says running and no process holds its lock: it was killed, or the machine went down.
  Killed,
}

impl Status {
  /// The word for it.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Running => "running",
      Self::Finished => "finished",
      Self::Failed => "failed",
      Self::Killed => "killed",
    }
  }
}

/// The machine a run was on. Everything but the operating system and architecture is best effort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Host {
  /// Its name, as `hostname` gives it.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub name: Option<String>,
  /// The operating system (`macos`, `linux`).
  pub os: String,
  /// The processor architecture (`aarch64`, `x86_64`).
  pub arch: String,
  /// How many threads it offers.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cpus: Option<usize>,
  /// The processor model.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub cpu: Option<String>,
}

/// The checkpoint a resumed run went on from. A resumed run is a run of its own that names the one it continues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Continues {
  /// The run that wrote the checkpoint, if it was recorded.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub run: Option<String>,
  /// The checkpoint directory.
  pub from: String,
  /// The step it held.
  pub step: usize,
}

/// The model a run worked on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
  /// What it was to the run: `base` for training, `scored` for a calibration.
  pub role: String,
  /// Where it was.
  pub dir: String,
  /// Its content digest.
  pub digest: String,
  /// Its fingerprint, where the run used it with embedder settings: the digest plus the width and token limit.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub fingerprint: Option<String>,
}

/// A file a run read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
  /// What it was used as (`pairs`, `catalog_groups`, `config`).
  pub role: String,
  /// Where it was.
  pub path: String,
  /// The SHA-256 of its text.
  pub sha256: String,
  /// How many examples or pairs came from it, where that means something.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub examples: Option<usize>,
}

impl From<&DataFile> for Input {
  fn from(file: &DataFile) -> Self {
    Self {
      role: file.role.clone(),
      path: file.path.clone(),
      sha256: file.sha256.clone(),
      examples: Some(file.examples),
    }
  }
}

/// Something a run wrote, outside its own directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
  /// What it is (`model`).
  pub role: String,
  /// Where it is.
  pub path: String,
  /// Its content digest, for a model.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub digest: Option<String>,
}

/// What ran, with what, on what, and how it ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
  /// The layout version, [`FORMAT`] for manifests written by this crate.
  pub format: u32,
  /// The run's id, which is its directory's name.
  pub id: String,
  /// What kind of run. Text, not an enum, so that a reader built before a new kind still lists it.
  pub kind: String,
  /// How the run stands, as far as the file knows.
  pub status: Stored,
  /// When it started, in seconds since the Unix epoch.
  pub started_at: u64,
  /// When it ended.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub ended_at: Option<u64>,
  /// How long it took, in seconds.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub elapsed_s: Option<f64>,
  /// The program that ran.
  pub build: Stamp,
  /// The machine it ran on.
  pub host: Host,
  /// Its process id.
  pub pid: u32,
  /// The directory it was started in.
  pub cwd: String,
  /// Its command line. Neither recorded command takes a secret; one that does must not be recorded as given.
  pub argv: Vec<String>,
  /// The checkpoint it went on from, for a resumed training run.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub continues: Option<Continues>,
  /// For training, the options as the trainer serializes them, so that a run that never wrote a model still says
  /// how it was set up.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub options: Option<serde_json::Value>,
  /// The model it worked on.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub model: Option<ModelRef>,
  /// The files it read.
  #[serde(default)]
  pub inputs: Vec<Input>,
  /// What it wrote outside its own directory.
  #[serde(default)]
  pub outputs: Vec<Output>,
  /// Why it failed, in the words of its `failed` event.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub error: Option<String>,
}

fn now() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_or(0, |elapsed| elapsed.as_secs())
}

/// `seconds` since the Unix epoch as UTC text, `YYYYMMDDTHHMMSSZ`: the front of a run's id, so that a listing of
/// the runs directory is in order of time.
///
/// The date is Howard Hinnant's days-to-civil conversion, on values that are far too small to overflow.
pub fn utc_stamp(seconds: u64) -> String {
  let days = i64::try_from(seconds.div_euclid(86_400)).unwrap_or(i64::MAX);
  let in_day = seconds.rem_euclid(86_400);
  let (hour, minute, second) = (
    in_day.div_euclid(3600),
    in_day.rem_euclid(3600).div_euclid(60),
    in_day.rem_euclid(60),
  );

  // Days are counted from 0000-03-01, so that a leap day is the last day of its year.
  let shifted = days.saturating_add(719_468);
  let era = shifted.div_euclid(146_097);
  let day_of_era = shifted.rem_euclid(146_097);
  let year_of_era = day_of_era
    .wrapping_sub(day_of_era.div_euclid(1_460))
    .wrapping_add(day_of_era.div_euclid(36_524))
    .wrapping_sub(day_of_era.div_euclid(146_096))
    .div_euclid(365);
  let day_of_year = day_of_era.wrapping_sub(
    year_of_era
      .wrapping_mul(365)
      .wrapping_add(year_of_era.div_euclid(4))
      .wrapping_sub(year_of_era.div_euclid(100)),
  );
  let shifted_month = day_of_year.wrapping_mul(5).wrapping_add(2).div_euclid(153);
  let day = day_of_year
    .wrapping_sub(shifted_month.wrapping_mul(153).wrapping_add(2).div_euclid(5))
    .wrapping_add(1);
  let month = if shifted_month < 10 {
    shifted_month.wrapping_add(3)
  } else {
    shifted_month.wrapping_sub(9)
  };
  let year = year_of_era
    .saturating_add(era.saturating_mul(400))
    .saturating_add(i64::from(month <= 2));
  format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

/// The part of an id that says what the run was of: `text` in lower case, every run of anything but ASCII letters
/// and digits turned into one `-`, trimmed, cut short, and `run` if nothing is left.
pub fn slug(text: &str) -> String {
  let mut slug = String::with_capacity(text.len().min(SLUG_BYTES));
  for character in text.chars() {
    if slug.len() >= SLUG_BYTES {
      break;
    }
    if character.is_ascii_alphanumeric() {
      slug.push(character.to_ascii_lowercase());
    } else if !slug.is_empty() && !slug.ends_with('-') {
      slug.push('-');
    }
  }
  let slug = slug.trim_end_matches('-');
  if slug.is_empty() {
    "run".to_owned()
  } else {
    slug.to_owned()
  }
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
  let output = std::process::Command::new(program).args(arguments).output().ok()?;
  let text = String::from_utf8(output.stdout).ok()?;
  let text = text.trim();
  (output.status.success() && !text.is_empty()).then(|| text.to_owned())
}

fn host() -> Host {
  let cpu = match std::env::consts::OS {
    "macos" => command_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
    "linux" => std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
      let line = text.lines().find(|line| line.starts_with("model name"))?;
      Some(line.split_once(':')?.1.trim().to_owned())
    }),
    _ => None,
  };
  Host {
    name: command_output("hostname", &[]),
    os: std::env::consts::OS.to_owned(),
    arch: std::env::consts::ARCH.to_owned(),
    cpus: std::thread::available_parallelism().ok().map(std::num::NonZero::get),
    cpu,
  }
}

/// Writes `manifest` into `dir` whole: beside itself, flushed, then renamed over the old one.
fn write_manifest(dir: &Path, manifest: &Manifest) -> Result<()> {
  let path = dir.join(MANIFEST_FILE);
  let partial = dir.join(format!("{MANIFEST_FILE}.partial"));
  let mut bytes = serde_json::to_vec_pretty(manifest).map_err(|source| RunError::Manifest {
    path: path.clone(),
    reason: source.to_string(),
  })?;
  bytes.push(b'\n');
  let mut file = File::create(&partial).map_err(|source| RunError::io(&partial, source))?;
  file
    .write_all(&bytes)
    .and_then(|()| file.sync_all())
    .map_err(|source| RunError::io(&partial, source))?;
  std::fs::rename(&partial, &path).map_err(|source| RunError::io(&path, source))?;
  // The rename is durable once the directory is flushed. Where a directory cannot be opened for that, the
  // manifest is still whole, which is the promise that matters here.
  if let Ok(directory) = File::open(dir) {
    let _ = directory.sync_all();
  }
  Ok(())
}

/// What a run is, before it has a directory.
#[derive(Debug)]
pub struct Spec<'a> {
  /// The runs root. Created if it is not there.
  pub root: &'a Path,
  /// The kind of run ([`KIND_TRAIN`], [`KIND_CALIBRATE`]).
  pub kind: &'static str,
  /// What the run is of, for its id: an output directory's name, a pairs file's stem. See [`slug`].
  pub slug: String,
  /// The program.
  pub build: Stamp,
  /// Its command line.
  pub argv: Vec<String>,
  /// The checkpoint it goes on from, for a resumed training run.
  pub continues: Option<Continues>,
}

/// A run that is being recorded. Dropping it without [`Run::finish`] records it as failed.
#[derive(Debug)]
pub struct Run {
  dir: PathBuf,
  id: String,
  // Held, not read: the lock on it is released when it closes, however that comes about.
  _lock: File,
  manifest: Mutex<Manifest>,
  started: Instant,
  done: bool,
}

impl Run {
  /// Claims a new directory under `spec.root` and records the run as running.
  ///
  /// Everything that could stop the run from being recorded is tried here: the root, the directory, the lock, the
  /// events file, and the manifest.
  ///
  /// # Errors
  ///
  /// Returns [`RunError::Io`] if the root or any file in the new directory cannot be created, and
  /// [`RunError::NoFreeId`] if every name the run could take is in use.
  pub fn start(spec: Spec<'_>) -> Result<Self> {
    std::fs::create_dir_all(spec.root).map_err(|source| RunError::io(spec.root, source))?;
    let started_at = now();
    let base = format!("{}-{}-{}", utc_stamp(started_at), spec.kind, slug(&spec.slug));
    let (id, dir) = claim(spec.root, &base)?;

    // The order matters to a reader: by the time a manifest says `running`, the lock is already held.
    let lock_path = dir.join(LOCK_FILE);
    let lock = File::create(&lock_path).map_err(|source| RunError::io(&lock_path, source))?;
    lock.try_lock().map_err(|error| {
      let source = match error {
        TryLockError::Error(source) => source,
        TryLockError::WouldBlock => std::io::Error::other("the lock of a run that was just created is held"),
      };
      RunError::io(&lock_path, source)
    })?;
    let events = dir.join(EVENTS_FILE);
    File::create_new(&events).map_err(|source| RunError::io(&events, source))?;

    let manifest = Manifest {
      format: FORMAT,
      id: id.clone(),
      kind: spec.kind.to_owned(),
      status: Stored::Running,
      started_at,
      ended_at: None,
      elapsed_s: None,
      build: spec.build,
      host: host(),
      pid: std::process::id(),
      cwd: std::env::current_dir().map_or_else(|_| String::new(), |dir| dir.display().to_string()),
      argv: spec.argv,
      continues: spec.continues,
      options: None,
      model: None,
      inputs: Vec::new(),
      outputs: Vec::new(),
      error: None,
    };
    write_manifest(&dir, &manifest)?;
    Ok(Self {
      dir,
      id,
      _lock: lock,
      manifest: Mutex::new(manifest),
      started: Instant::now(),
      done: false,
    })
  }

  /// The run's id.
  pub fn id(&self) -> &str {
    &self.id
  }

  /// The run's directory.
  pub fn dir(&self) -> &Path {
    &self.dir
  }

  /// A handle on the run's `events.ndjson` that appends, for the logging setup to write through.
  ///
  /// # Errors
  ///
  /// Returns [`RunError::Io`] if the file cannot be opened.
  pub fn events(&self) -> Result<File> {
    let path = self.dir.join(EVENTS_FILE);
    OpenOptions::new()
      .append(true)
      .open(&path)
      .map_err(|source| RunError::io(&path, source))
  }

  /// Changes the manifest and writes it. A failure is a warning: what the run is doing matters more than the note
  /// about it, and the place it cannot be written to was writable when the run started.
  fn note(&self, change: impl FnOnce(&mut Manifest)) {
    let Ok(mut manifest) = self.manifest.lock() else {
      tracing::warn!(
        run = self.id.as_str(),
        "the run's manifest cannot be updated: its lock is poisoned"
      );
      return;
    };
    change(&mut manifest);
    if let Err(error) = write_manifest(&self.dir, &manifest) {
      tracing::warn!(run = self.id.as_str(), %error, "the run's manifest could not be written");
    }
  }

  /// Records the options the run was set up with.
  pub fn note_options(&self, options: serde_json::Value) {
    self.note(|manifest| manifest.options = Some(options));
  }

  /// Records the model the run works on.
  pub fn note_model(&self, model: ModelRef) {
    self.note(|manifest| manifest.model = Some(model));
  }

  /// Records a file the run read.
  pub fn note_input(&self, input: Input) {
    self.note(|manifest| manifest.inputs.push(input));
  }

  /// Records something the run wrote.
  pub fn note_output(&self, output: Output) {
    self.note(|manifest| manifest.outputs.push(output));
  }

  /// Keeps a text report beside the events, as `report.txt`.
  pub fn note_report(&self, text: &str) {
    let path = self.dir.join(REPORT_FILE);
    if let Err(error) = std::fs::write(&path, text) {
      tracing::warn!(run = self.id.as_str(), %error, "the run's report could not be written");
    }
  }

  fn end(&mut self, outcome: std::result::Result<(), &str>) -> Result<()> {
    self.done = true;
    let elapsed = self.started.elapsed().as_secs_f64();
    let mut manifest = self.manifest.lock().map_err(|_| RunError::Manifest {
      path: self.dir.join(MANIFEST_FILE),
      reason: "its lock is poisoned".to_owned(),
    })?;
    manifest.ended_at = Some(now());
    // Tenths of a second: precise enough to compare two runs, and readable.
    manifest.elapsed_s = Some((elapsed * 10.0).round() / 10.0);
    match outcome {
      Ok(()) => manifest.status = Stored::Finished,
      Err(error) => {
        manifest.status = Stored::Failed;
        manifest.error = Some(error.to_owned());
      }
    }
    write_manifest(&self.dir, &manifest)
  }

  /// Records how the run ended, and releases its lock.
  ///
  /// # Errors
  ///
  /// Returns an error if the manifest cannot be written. The run's own outcome is the caller's to report first.
  pub fn finish(mut self, outcome: std::result::Result<(), &str>) -> Result<()> {
    self.end(outcome)
  }
}

impl Drop for Run {
  fn drop(&mut self) {
    // A run that unwinds past its `finish` (a panic) still ends on disk as what it is.
    if !self.done {
      let _ = self.end(Err("ended without reporting an outcome"));
    }
  }
}

/// Makes the run's directory under `root`: `base`, or `base-2`, `base-3`, ... if runs already started this second.
fn claim(root: &Path, base: &str) -> Result<(String, PathBuf)> {
  for attempt in 1..=SAME_SECOND {
    let id = if attempt == 1 {
      base.to_owned()
    } else {
      format!("{base}-{attempt}")
    };
    let dir = root.join(&id);
    // Not `create_dir_all`: a directory that is already there is someone else's run.
    match std::fs::create_dir(&dir) {
      Ok(()) => return Ok((id, dir)),
      Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
      Err(error) => return Err(RunError::io(&dir, error)),
    }
  }
  Err(RunError::NoFreeId {
    root: root.to_owned(),
    id: base.to_owned(),
  })
}

/// One run, as read.
#[derive(Debug, Clone, PartialEq)]
pub struct Listed {
  /// Its directory.
  pub dir: PathBuf,
  /// Its manifest.
  pub manifest: Manifest,
  /// How it stands.
  pub status: Status,
}

/// A directory under the runs root that could not be read as a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unreadable {
  /// The directory.
  pub dir: PathBuf,
  /// Why.
  pub reason: String,
}

/// How a run whose manifest says `running` really stands: alive if something holds its lock.
fn running_or_killed(dir: &Path) -> Status {
  let Ok(lock) = File::open(dir.join(LOCK_FILE)) else {
    // No lock file, so no process can be holding it.
    return Status::Killed;
  };
  match lock.try_lock() {
    // Nobody holds it. The reader's own lock goes when `lock` closes, and is released at once to be sure.
    Ok(()) => {
      let _ = lock.unlock();
      Status::Killed
    }
    // Held, or a filesystem that cannot say: alive cannot be ruled out.
    Err(TryLockError::WouldBlock | TryLockError::Error(_)) => Status::Running,
  }
}

/// Reads one run directory. Never writes: a killed run stays `running` on disk and is concluded killed each time.
///
/// # Errors
///
/// Returns [`RunError::Io`] if the manifest cannot be read, and [`RunError::Manifest`] if it is not a manifest or
/// is of a later format than this crate knows.
pub fn open(dir: &Path) -> Result<Listed> {
  let path = dir.join(MANIFEST_FILE);
  let bytes = std::fs::read(&path).map_err(|source| RunError::io(&path, source))?;
  let invalid = |reason: String| RunError::Manifest {
    path: path.clone(),
    reason,
  };
  let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|source| invalid(source.to_string()))?;
  let format = value.get("format").and_then(serde_json::Value::as_u64);
  if format.is_some_and(|format| format > u64::from(FORMAT)) {
    return Err(invalid(format!(
      "format {} is newer than this version reads ({FORMAT})",
      format.unwrap_or_default()
    )));
  }
  let manifest: Manifest = serde_json::from_value(value).map_err(|source| invalid(source.to_string()))?;
  let status = match manifest.status {
    Stored::Finished => Status::Finished,
    Stored::Failed => Status::Failed,
    Stored::Running => running_or_killed(dir),
  };
  Ok(Listed {
    dir: dir.to_owned(),
    manifest,
    status,
  })
}

/// Every run under `root`, newest first, and every directory there that is not a run. A root that does not exist
/// holds no runs.
///
/// # Errors
///
/// Returns [`RunError::Io`] if the root exists and cannot be listed.
pub fn list(root: &Path) -> Result<(Vec<Listed>, Vec<Unreadable>)> {
  let entries = match std::fs::read_dir(root) {
    Ok(entries) => entries,
    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), Vec::new())),
    Err(error) => return Err(RunError::io(root, error)),
  };
  let (mut runs, mut unreadable) = (Vec::new(), Vec::new());
  for entry in entries {
    let dir = entry.map_err(|source| RunError::io(root, source))?.path();
    if !dir.is_dir() {
      continue;
    }
    match open(&dir) {
      Ok(run) => runs.push(run),
      Err(error) => unreadable.push(Unreadable {
        dir,
        reason: error.to_string(),
      }),
    }
  }
  runs.sort_by(|a, b| (b.manifest.started_at, &b.manifest.id).cmp(&(a.manifest.started_at, &a.manifest.id)));
  unreadable.sort_by(|a, b| a.dir.cmp(&b.dir));
  Ok((runs, unreadable))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn stamp() -> Stamp {
    Stamp {
      tool: "record-similarity-train".to_owned(),
      version: "0.1.0".to_owned(),
      commit: "0123456789ab".to_owned(),
      dirty: Some(false),
    }
  }

  fn spec<'a>(root: &'a Path, kind: &'static str, slug: &'a str) -> Spec<'a> {
    Spec {
      root,
      kind,
      slug: slug.to_owned(),
      build: stamp(),
      argv: vec!["record-similarity-train".to_owned(), "--out".to_owned(), "m".to_owned()],
      continues: None,
    }
  }

  #[test]
  fn the_start_time_in_an_id_is_utc_to_the_second() {
    for (seconds, expected) in [
      (0, "19700101T000000Z"),
      (1_700_000_000, "20231114T221320Z"),
      // A leap day, and the seconds either side of a year's end.
      (1_709_164_800, "20240229T000000Z"),
      (1_735_689_599, "20241231T235959Z"),
      (1_735_689_600, "20250101T000000Z"),
      (951_782_400, "20000229T000000Z"),
      (4_102_444_800, "21000101T000000Z"),
    ] {
      assert_eq!(utc_stamp(seconds), expected, "{seconds}");
    }
  }

  #[test]
  fn a_slug_keeps_only_lowercase_letters_digits_and_single_dashes() {
    assert_eq!(slug("mdbr-leaf-mt-ft"), "mdbr-leaf-mt-ft");
    assert_eq!(slug("Pairs.Full  (holdout)"), "pairs-full-holdout");
    assert_eq!(slug("../../etc/passwd"), "etc-passwd");
    assert_eq!(slug("--"), "run");
    assert_eq!(slug(""), "run");
    assert_eq!(slug("café"), "caf");
    let long = slug(&"abc-".repeat(40));
    assert_eq!(
      long.len(),
      39,
      "cut at forty bytes, and not left ending in a dash: {long}"
    );
  }

  #[test]
  fn two_runs_started_in_the_same_second_get_different_directories() {
    let root = tempfile::tempdir().unwrap();
    let runs: Vec<Run> = (0..3)
      .map(|_| Run::start(spec(root.path(), KIND_TRAIN, "m")).unwrap())
      .collect();
    let mut ids: Vec<&str> = runs.iter().map(Run::id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 3, "{ids:?}");
    assert!(ids.iter().all(|id| id.contains("-train-m")));
  }

  #[test]
  fn a_started_run_reads_back_as_running_with_its_build_host_and_arguments() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), KIND_TRAIN, "models/Mdbr Leaf")).unwrap();
    assert!(run.id().ends_with("-train-models-mdbr-leaf"), "{}", run.id());
    assert_eq!(run.dir(), root.path().join(run.id()));

    let read = open(run.dir()).unwrap();
    assert_eq!(read.status, Status::Running, "its lock is held by this test");
    let manifest = read.manifest;
    assert_eq!((manifest.format, manifest.kind.as_str()), (FORMAT, KIND_TRAIN));
    assert_eq!(manifest.build, stamp());
    assert_eq!(manifest.argv.len(), 3);
    assert_eq!(manifest.pid, std::process::id());
    assert_eq!(manifest.host.os, std::env::consts::OS);
    assert!(manifest.started_at > 1_700_000_000 && manifest.ended_at.is_none());
    assert!(run.dir().join(EVENTS_FILE).is_file() && run.dir().join(LOCK_FILE).is_file());

    // What is not known is left out, never written as null.
    let text = std::fs::read_to_string(run.dir().join(MANIFEST_FILE)).unwrap();
    assert!(!text.contains("null"), "{text}");
  }

  #[test]
  fn a_finished_run_records_its_end_and_how_long_it_took() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), KIND_CALIBRATE, "pairs")).unwrap();
    let dir = run.dir().to_owned();
    run.note_options(serde_json::json!({"epochs": 2}));
    run.note_model(ModelRef {
      role: "scored".to_owned(),
      dir: "models/m".to_owned(),
      digest: "aaaa".to_owned(),
      fingerprint: Some("aaaa-d256-t256".to_owned()),
    });
    run.note_input(Input::from(&DataFile {
      role: "pairs".to_owned(),
      path: "pairs.jsonl".to_owned(),
      sha256: "bbbb".to_owned(),
      examples: 46,
    }));
    run.note_output(Output {
      role: "model".to_owned(),
      path: "models/out".to_owned(),
      digest: None,
    });
    run.note_report("the report\n");
    assert!(
      !dir.join(format!("{MANIFEST_FILE}.partial")).exists(),
      "never left half written"
    );
    run.finish(Ok(())).unwrap();

    let read = open(&dir).unwrap();
    assert_eq!(read.status, Status::Finished);
    let manifest = read.manifest;
    assert!(manifest.ended_at.unwrap() >= manifest.started_at && manifest.elapsed_s.is_some());
    assert_eq!(manifest.options.unwrap()["epochs"], 2);
    assert_eq!(manifest.model.unwrap().fingerprint.as_deref(), Some("aaaa-d256-t256"));
    assert_eq!((manifest.inputs[0].examples, manifest.outputs.len()), (Some(46), 1));
    assert_eq!(manifest.error, None);
    assert_eq!(std::fs::read_to_string(dir.join(REPORT_FILE)).unwrap(), "the report\n");
  }

  #[test]
  fn a_failed_run_records_the_error_it_ended_with() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), KIND_TRAIN, "m")).unwrap();
    let dir = run.dir().to_owned();
    run.finish(Err("the pairs file is empty")).unwrap();
    let read = open(&dir).unwrap();
    assert_eq!(read.status, Status::Failed);
    assert_eq!(read.manifest.error.as_deref(), Some("the pairs file is empty"));
  }

  #[test]
  fn a_run_dropped_without_an_outcome_is_recorded_as_failed() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), KIND_TRAIN, "m")).unwrap();
    let dir = run.dir().to_owned();
    drop(run);
    let read = open(&dir).unwrap();
    assert_eq!(read.status, Status::Failed);
    assert!(read.manifest.error.unwrap().contains("without reporting"));
  }

  /// What a killed process leaves: a manifest that says running, and a lock nobody holds.
  #[test]
  fn a_running_run_whose_lock_nobody_holds_reads_as_killed() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), KIND_TRAIN, "m")).unwrap();
    let dir = run.dir().to_owned();
    assert_eq!(open(&dir).unwrap().status, Status::Running, "while it is held");
    // Ending the process is the one thing a test cannot do to itself. Marking the run as done keeps its
    // destructor from writing, which leaves the manifest as a kill would; dropping it closes the lock file, which
    // is what the system then does.
    let mut run = run;
    run.done = true;
    drop(run);
    let read = open(&dir).unwrap();
    assert_eq!(read.manifest.status, Stored::Running, "nothing wrote to it");
    assert_eq!(read.status, Status::Killed);
    assert_eq!(open(&dir).unwrap().status, Status::Killed, "reading does not change it");

    // The same without a lock file at all, as in a run directory copied from elsewhere.
    std::fs::remove_file(dir.join(LOCK_FILE)).unwrap();
    assert_eq!(open(&dir).unwrap().status, Status::Killed);
  }

  #[test]
  fn a_note_that_cannot_be_written_warns_and_does_not_stop_the_run() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), KIND_TRAIN, "m")).unwrap();
    // The run's directory is gone: nothing can be written into it any more.
    std::fs::remove_dir_all(run.dir()).unwrap();
    run.note_options(serde_json::json!({"epochs": 2}));
    run.note_report("lost");
    assert!(run.finish(Ok(())).is_err(), "the last write does say that it failed");
  }

  #[test]
  fn listing_puts_the_newest_run_first_and_reports_a_directory_that_is_not_a_run() {
    let root = tempfile::tempdir().unwrap();
    assert_eq!(list(&root.path().join("missing")).unwrap(), (Vec::new(), Vec::new()));

    let first = Run::start(spec(root.path(), KIND_TRAIN, "a")).unwrap();
    let second = Run::start(spec(root.path(), KIND_CALIBRATE, "b")).unwrap();
    let (first_dir, second_id) = (first.dir().to_owned(), second.id().to_owned());
    first.finish(Ok(())).unwrap();
    // Make the order certain whatever the clock did between the two starts.
    let mut manifest = open(&first_dir).unwrap().manifest;
    manifest.started_at -= 10;
    write_manifest(&first_dir, &manifest).unwrap();

    std::fs::create_dir(root.path().join("notes")).unwrap();
    std::fs::write(root.path().join("a-file"), "not a directory").unwrap();

    let (runs, unreadable) = list(root.path()).unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].manifest.id, second_id, "newest first");
    assert_eq!((runs[0].status, runs[1].status), (Status::Running, Status::Finished));
    assert_eq!(unreadable.len(), 1);
    assert!(unreadable[0].dir.ends_with("notes"));
    drop(second);
  }

  #[test]
  fn a_manifest_from_a_later_format_is_refused_by_name() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), KIND_TRAIN, "m")).unwrap();
    let dir = run.dir().to_owned();
    run.finish(Ok(())).unwrap();
    let path = dir.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&path)
      .unwrap()
      .replace("\"format\": 1", "\"format\": 2");
    std::fs::write(&path, text).unwrap();
    let error = open(&dir).unwrap_err().to_string();
    assert!(
      error.contains("format 2 is newer") && error.contains(MANIFEST_FILE),
      "{error}"
    );
  }

  #[test]
  fn a_kind_this_version_does_not_know_is_still_listed() {
    let root = tempfile::tempdir().unwrap();
    let run = Run::start(spec(root.path(), "sweep", "grid")).unwrap();
    let dir = run.dir().to_owned();
    run.finish(Ok(())).unwrap();
    assert_eq!(open(&dir).unwrap().manifest.kind, "sweep");
    assert_eq!(list(root.path()).unwrap().0.len(), 1);
  }

  #[test]
  fn a_root_that_cannot_be_created_is_an_error_before_anything_is_written() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("file");
    std::fs::write(&file, "in the way").unwrap();
    let error = Run::start(spec(&file.join("runs"), KIND_TRAIN, "m")).unwrap_err();
    assert!(matches!(error, RunError::Io { .. }), "{error}");
    assert_eq!(
      std::fs::read_dir(root.path()).unwrap().count(),
      1,
      "nothing was left behind"
    );
  }

  #[test]
  fn a_resumed_run_names_the_run_it_continues() {
    let root = tempfile::tempdir().unwrap();
    let mut spec = spec(root.path(), KIND_TRAIN, "m");
    spec.continues = Some(Continues {
      run: Some("20261111T120002Z-train-m".to_owned()),
      from: "m.checkpoints/step-000120".to_owned(),
      step: 120,
    });
    let run = Run::start(spec).unwrap();
    let continues = open(run.dir()).unwrap().manifest.continues.unwrap();
    assert_eq!(
      (continues.run.as_deref(), continues.step),
      (Some("20261111T120002Z-train-m"), 120)
    );
  }
}
