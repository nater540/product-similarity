# Plan: a local run registry (feature 2)

A plan for keeping every training run and every calibration as a directory on disk, written so that a session with
no memory of the conversation that produced it can carry it out. Nothing here is built yet. Read `CLAUDE.md` first,
then section 2 of `docs/feature-suggestions.md`, then this file. Delete this file, or fold what is still true into
`docs/decisions.md`, when the work is done.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [What features 4 and 5 may rely on](#what-features-4-and-5-may-rely-on)
- [What can only be checked where the data lives](#what-can-only-be-checked-where-the-data-lives)
- [Decisions for the owner, with the default to use meanwhile](#decisions-for-the-owner-with-the-default-to-use-meanwhile)
- [Size](#size)
- [Out of scope](#out-of-scope)

## What it is for

Every `record-similarity-train` run and every `record-similarity calibrate` run leaves a directory under `runs/`
holding what it was asked to do (`run.json`) and everything it said (`events.ndjson`), whether or not anybody
redirected its output. Today a training run's loss curve and its `separation` results survive only if someone
remembered `> results.ndjson` and `2> logs.ndjson`, and a calibration only if someone remembered `--ndjson`. The
first real fine-tuning runs are about to happen (`docs/plan-features.md`, "What to build next", item 5), and their
results cannot be recovered afterwards.

It is what W&B calls a run, kept as local files. Nothing is sent anywhere, and W&B stays an optional after-the-fact
exporter that this feature does not build. Features 4 (model registry and promotion gate) and 5 (run report and
compare) read what this feature writes; their designs are not part of this plan.

## What already exists

Verified against the code on `initial-implementation` at `545ef20`.

- **The logging setup.** `similarity_log::subscriber<L, R>(logs, results, filter)` in
  `crates/similarity-log/src/lib.rs` builds one `tracing_subscriber::registry()` with two JSON `fmt` layers: one for
  logs and failures (filtered by `RUST_LOG`, failures always), one for results (target `similarity_log::RESULT`,
  never filtered). `init()` installs it with `stderr` and `stdout` and sets a panic hook that logs `panicked` at
  target `FAILURE`. `similarity_log::testing::capture` captures both streams for tests. Each layer is built with
  the same five builder calls (`json`, `flatten_event(true)`, no span context, a writer, a filter).
- **The build stamp.** `similarity_core::provenance::Build { tool, version, commit, dirty }` is a `Copy` struct of
  `&'static str`s, not serializable. Both binaries state one as a `const BUILD` from `build_commit.rs`. Its fields
  are copied flat into `Provenance` (`tool`, `version`, `commit`, `dirty`) by `Build::provenance`.
- **Provenance.** `Provenance`, `Trained`, `Base`, `DataFile { role, path, sha256, examples }`, and
  `Resumed { from, step }` in `crates/similarity-core/src/provenance.rs`. No `deny_unknown_fields`, so an added
  optional field is read by older code. `FORMAT` is `1`.
- **What the trainer emits** (`crates/similarity-train/src/main.rs`). Logs, on stderr and filtered: `model loaded`,
  `labeled pairs`, `catalog groups`, `training` (step, of, epoch, `pair_loss`, `group_loss`, `grad_norm`, `rate`,
  `elapsed_s`, every `--log-every` steps), `checkpoint` (step, epoch, path), `model saved and verified to load`.
  Results, on stdout: `separation` per split, stage, and width, then `finished`. `Cli::lineage` already hashes each
  data file into a `DataFile`. `checkpoint::save_model` and `Step::checkpoint` stamp `provenance.json` from it.
- **What calibrate emits** (`crates/similarity-server/src/main.rs`, `CalibrateArgs::run`). Without `--ndjson`, a text
  report on stdout through `offline::render_calibration`, and no events at all. With `--ndjson`,
  `offline::events::emit` writes `calibration`, `schema_*`, `pair`, `pair_field`, `pair_quantity`, `pair_reason`,
  ranges, suggestions, slices, operating points, `ignored_key`, and `summary`, all at target `RESULT`.
- **The calibration reader skips what is not a result.** `offline::events::read` passes over every line whose
  `target` is not `result`, and over result messages it has no use for. A file holding logs and results together
  already reads as a calibration, so `calibrate-diff runs/A/events.ndjson runs/B/events.ndjson` will work unchanged.
- **Whole-or-nothing writing** exists for directories only: `write_whole`, `sync_tree`, and `sync_dir` in
  `crates/similarity-train/src/checkpoint.rs`, all private to that crate.
- **A run's identity for resuming** is `checkpoint::RunIdentity` (options, data digest, token limit). CLAUDE.md
  requires anything that changes what a step does to be in it; a run id changes nothing a step does.
- **Workspace facts.** `rust-version = "1.96"`, so `std::fs::File::lock` and `try_lock` (stable since 1.89) are
  available with no dependency. `serde_json` has `float_roundtrip` workspace-wide. `.gitignore` and
  `.dockerignore` know nothing of `runs/`. No test runs a binary as a process (`CARGO_BIN_EXE` appears nowhere);
  the trainer's tests call `Cli::run` directly.

## What the suggestion got wrong or left open

1. **`throughput` is not a command.** It is a cargo example in `crates/similarity-core/examples/throughput.rs` that
   prints a text table with `println!` and uses no logging at all. Neither binary runs it. Recording it means first
   turning it into something that emits events. This plan leaves it out (owner decision 2).
2. **The metric history is mostly logs, not results.** `training` and `checkpoint` are info-level logs on stderr,
   so `RUST_LOG=warn` hides them. A file sink that simply tees the two streams, as the suggestion and the note in
   `plan-features.md` describe, would lose the loss curve whenever someone quietened stderr. The run file needs a
   level of its own. That makes it a third layer, built by the same function as the other two and added to the
   same subscriber, not a tee. It is still one subscriber and one format, which is what CLAUDE.md and D15 ask.
3. **`killed` cannot be written by a killed process.** Nothing runs after `SIGKILL`, and the trainer installs no
   signal handler, so `SIGINT` and `SIGTERM` end it without a word too. The stored status can only be `running`,
   `finished`, or `failed`; `killed` is derived when the registry is read, from an advisory lock the live process
   holds on a file in its run directory. The operating system releases it however the process ends.
4. **Appending a resumed run to its original directory is the wrong default.** A resume is a different process,
   possibly on another machine (checkpoints are copied to a GPU box), possibly from an earlier checkpoint than the
   last one (its steps would then appear twice in one file), and possibly started twice by mistake (two writers in
   one file). The original directory's status would also have to be rewritten from `killed` back to `running`.
   This plan gives every invocation its own directory and links a resume to the run it continues (owner decision
   1). A chain is reassembled by following the links, which feature 5 needs to do anyway.
5. **A calibration in text mode writes no events.** Without `--ndjson`, its run directory would hold only logs, and
   feature 4 (`model promote --calibration runs/<id>`) would have nothing to read. This plan records the events
   in the run directory in both modes and keeps stdout exactly as it is (owner decision 4).
6. **"Model fingerprint" fits a calibration, not a training run.** A fingerprint is the digest plus the embedder
   settings the server applies; the trainer has content digests (the base's and the output's), which `Provenance`
   already records. The manifest names a digest for training and a fingerprint for a calibration.
7. **`--run-dir` "moves it" is ambiguous** (the root, or this run's directory?). This plan uses `--runs <ROOT>`.
8. **"A new small crate with no Burn dependency"** conflicts with "the manifest and `provenance.json` should share
   types": `Build` and `DataFile` live in `similarity-core`, which is Burn. This plan lets `similarity-runs` depend
   on `similarity-core`; both binaries already link it, so nothing new is compiled for them (owner decision 9).
9. **Host and CPU** are not in `std`: there is no hostname call and no CPU model name. Both are best effort and
   left out of the manifest when unknown.
10. **Input hashes for training are already computed** in `Cli::lineage`; the manifest reuses them rather than
    reading the data files a third time.

Things it left undecided and this plan decides: the id format, what happens when the run directory cannot be
created (the command fails before doing any work), whether tests may write into `runs/` (never: recording starts
in `main`, not in `Cli::run`), and where `runs list` lives (owner decision 5).

## Design

### On disk

```text
runs/                                  default root; --runs or RECORD_SIMILARITY_RUNS to change it
  20261111T142301Z-train-mdbr-leaf-mt-ft/
    run.json                           the manifest, rewritten whole as it learns more
    events.ndjson                      every line both streams carried, plus info logs, in the order emitted
    .lock                              held (advisory, exclusive) by the process for as long as it runs
  20261111T150912Z-calibrate-pairs-full-holdout/
    run.json
    events.ndjson                      for a calibration: the same events `--ndjson` writes, whatever the mode
    report.txt                         the text report, when that is what stdout got
```

- **Id:** `<UTC start, %Y%m%dT%H%M%SZ>-<kind>-<slug>`. The slug is the `--out` directory's name for training, and
  the pairs file's stem plus `-<split>` (when `--split` is given) for a calibration, lowercased, with every run of
  other characters turned into one `-`, trimmed of `-`, cut to 40 bytes, and `run` if nothing is left. The
  directory is claimed with `std::fs::create_dir` (not `create_dir_all`), which fails if it exists; on a collision
  `-2`, `-3`, ... is appended, up to `-99`, then an error. No randomness, no clock crate: the UTC text comes from a
  twenty-line days-to-civil conversion of `SystemTime` seconds, tested against known instants.
- **Order of creation**, which the status logic relies on: create the directory, create and lock `.lock`, create
  `events.ndjson` (`create_new`), then write `run.json` with `status: "running"`. A reader that sees a `running`
  manifest therefore knows the lock was already taken.
- **`run.json` is written whole**: to `run.json.partial`, flushed, renamed over `run.json`, and the directory
  flushed (the same discipline as `write_whole`; a small private helper, since that one is private to the trainer
  and works on directories).
- **`events.ndjson` is unbuffered**: each event is one `write` of one line to the file through the subscriber's
  `Mutex<File>` writer, so a run killed at any point keeps everything up to its last line. No `BufWriter`.
- `runs/` joins `.gitignore` and `.dockerignore`. A run of real data holds pair ids, owners, and paths; it stays
  where it was made.

### `run.json`

```json
{
  "format": 1,
  "id": "20261111T142301Z-train-mdbr-leaf-mt-ft",
  "kind": "train",
  "status": "finished",
  "started_at": 1762870981,
  "ended_at": 1762871353,
  "elapsed_s": 372.4,
  "build": {"tool": "record-similarity-train", "version": "0.1.0", "commit": "545ef20c1a2b", "dirty": false},
  "host": {"name": "mac-mini", "os": "macos", "arch": "aarch64", "cpus": 12, "cpu": "Apple M4 Pro"},
  "pid": 41237,
  "cwd": "/Users/someone/record-similarity",
  "argv": ["record-similarity-train", "--base", "models/mdbr-leaf-mt", "--out", "models/mdbr-leaf-mt-ft", "..."],
  "continues": {"run": "20261111T120002Z-train-mdbr-leaf-mt-ft", "from": "models/mdbr-leaf-mt-ft.checkpoints/step-000120", "step": 120},
  "options": {"epochs": 2, "batch_size": 32, "learning_rate": 2e-5, "widths": [1024, 512, 256], "...": "..."},
  "model": {"role": "base", "dir": "models/mdbr-leaf-mt", "digest": "f46e48ee0ea2dd02"},
  "inputs": [{"role": "pairs", "path": "pairs.full.jsonl", "sha256": "…", "examples": 4120}],
  "outputs": [{"role": "model", "path": "models/mdbr-leaf-mt-ft", "digest": "9a1c…"}]
}
```

A failed run adds `"error": "<the same text as the failed event>"`. Absent values are left out, never `null`
(D15). For a calibration, `model` is `{"role": "scored", "dir": ..., "digest": ..., "fingerprint": ...}`, `inputs`
are the pairs file (with the number of pairs scored as `examples`) and the config file (no `examples`), and
`outputs` is empty. `continues.run` is absent when the checkpoint was written without a run.

In Rust, in `similarity-runs` (`crates/similarity-runs/src/manifest.rs`):

```rust
pub const MANIFEST_FILE: &str = "run.json";
pub const EVENTS_FILE: &str = "events.ndjson";
pub const REPORT_FILE: &str = "report.txt";
pub const LOCK_FILE: &str = ".lock";
pub const FORMAT: u32 = 1;
pub const KIND_TRAIN: &str = "train";
pub const KIND_CALIBRATE: &str = "calibrate";

pub struct Manifest {
  pub format: u32,
  pub id: String,
  /// A string, not an enum, so that a reader built before a new kind (`sweep`, `throughput`) still lists it.
  pub kind: String,
  pub status: Stored,
  pub started_at: u64,
  pub ended_at: Option<u64>,
  pub elapsed_s: Option<f64>,
  pub build: Stamp,
  pub host: Host,
  pub pid: u32,
  pub cwd: String,
  pub argv: Vec<String>,
  pub continues: Option<Continues>,
  /// For training, the `TrainOptions` as the trainer serializes them (the same value as `provenance.json`'s
  /// `trained.options`), so that a run that never wrote a model still says how it was set up. Absent otherwise.
  pub options: Option<serde_json::Value>,
  pub model: Option<ModelRef>,
  pub inputs: Vec<Input>,
  pub outputs: Vec<Output>,
  pub error: Option<String>,
}

/// What the file says. `Killed` is never stored: see [`Status`].
pub enum Stored { Running, Finished, Failed }

/// What a reader concludes: a stored `Running` whose lock nobody holds is `Killed`.
pub enum Status { Running, Finished, Failed, Killed }

pub struct Host { pub name: Option<String>, pub os: String, pub arch: String, pub cpus: Option<usize>, pub cpu: Option<String> }
pub struct Continues { pub run: Option<String>, pub from: String, pub step: usize }
pub struct ModelRef { pub role: String, pub dir: String, pub digest: String, pub fingerprint: Option<String> }
pub struct Input { pub role: String, pub path: String, pub sha256: String, pub examples: Option<usize> }
pub struct Output { pub role: String, pub path: String, pub digest: Option<String> }
```

All with `Serialize`, `Deserialize`, `skip_serializing_if = "Option::is_none"` on every option, and
`#[serde(rename_all = "snake_case")]` on the enums. `Input` has `From<&DataFile>`. A manifest whose `format` is
above `FORMAT` is refused by name, as `Provenance` would be.

`host.name` comes from running `hostname` (absent if that fails); `cpus` from `std::thread::available_parallelism`;
`cpu` from `sysctl -n machdep.cpu.brand_string` on macOS and the first `model name` in `/proc/cpuinfo` on Linux,
absent elsewhere or on failure. `argv` is recorded as given (lossy UTF-8). Neither recorded command takes a
secret; a later command that does (`import --qdrant-api-key`) must not be recorded without redaction, and the
module says so.

### Shared types in `similarity-core`

- `provenance::Stamp { tool: String, version: String, commit: String, dirty: Option<bool> }`, serializable, and
  `Build::stamp(&self) -> Stamp`. `Provenance` keeps its flat layout; `Build::provenance` fills it from the same
  four values. (Making `Provenance` `#[serde(flatten)]` a `Stamp` would read the same JSON but routes numbers
  through serde's buffered content, which is not worth the risk for no change on disk.)
- `Provenance` gains `run: Option<String>`: the id of the run that wrote the directory. It records what produced
  the model, so it belongs there under D18; it is written inside `write_whole` like the rest; it is not hashed by
  `ModelFiles::digest`. `Resumed` gains `run: Option<String>`, the run that wrote the checkpoint resumed from.
  Both are optional and additive, so `FORMAT` stays `1`.
- The run id is **not** part of `RunIdentity`. A resume is always a new run with a new id, and must be accepted.

### `similarity-runs`: writing a run

A new library crate, `crates/similarity-runs`, with `thiserror` errors (`RunError`). Dependencies: `similarity-core`
(for `Stamp` and `DataFile`), `serde`, `serde_json`, `thiserror`, `tracing`; `tempfile` for tests. No new external
crate, so `cargo deny` sees nothing new.

```rust
pub struct Spec<'a> {
  pub root: &'a Path,
  pub kind: &'static str,
  pub slug: &'a str,
  pub build: Stamp,
  pub argv: Vec<String>,
  pub continues: Option<Continues>,
}

pub struct Run { /* dir, id, lock: File, manifest: Mutex<Manifest>, started: Instant, done: bool */ }

impl Run {
  /// Claims a new directory under `spec.root` and records the run as running.
  /// # Errors: the root cannot be created, no free id, or a file cannot be written.
  pub fn start(spec: Spec<'_>) -> Result<Self, RunError>;
  pub fn id(&self) -> &str;
  pub fn dir(&self) -> &Path;
  /// A second handle on `events.ndjson`, for `similarity_log::init_with`.
  pub fn events(&self) -> Result<File, RunError>;
  pub fn note_options(&self, options: serde_json::Value);
  pub fn note_model(&self, model: ModelRef);
  pub fn note_input(&self, input: Input);
  pub fn note_output(&self, output: Output);
  /// Writes `report.txt`.
  pub fn note_report(&self, text: &str);
  /// Records the outcome, the end time, and releases the lock.
  pub fn finish(self, outcome: Result<(), &str>) -> Result<(), RunError>;
}

impl Drop for Run { /* not finished: status failed, error "ended without reporting an outcome" */ }
```

The `note_*` methods rewrite `run.json` and **never fail the run**: a failure is a `warn` log and the work goes on,
as checkpoint pruning does (D13). Everything that could stop a run from being recorded at all is found out in
`start`, before any work, which is the same rule as `Checkpoints::prepare`. `Drop` covers a panic (the release
profile unwinds), so a crash reads `failed` with the `panicked` event in `events.ndjson` beside it.

### `similarity-runs`: reading runs

```rust
pub struct Listed { pub dir: PathBuf, pub manifest: Manifest, pub status: Status }
pub enum Unreadable { /* dir and reason, for a directory under the root that is not a run */ }

/// Every run under `root`, newest first (by `started_at`, then id), and what could not be read.
pub fn list(root: &Path) -> Result<(Vec<Listed>, Vec<Unreadable>), RunError>;
/// One run directory.
pub fn open(dir: &Path) -> Result<Listed, RunError>;
```

Status: a stored `finished` or `failed` is taken as written. A stored `running` is checked by opening `.lock` and
calling `File::try_lock`: `WouldBlock` means the process is alive (`Running`); success means it is gone (`Killed`),
and the reader unlocks at once; any other error (a filesystem without locks) is reported as `Running`, since that
cannot be ruled out. Readers never write: a killed run stays `running` on disk and is concluded `killed` each time.

### `similarity-log`: a third layer

```rust
pub struct Init {
  /// The run's `events.ndjson`, if the process is recording a run.
  pub record: Option<std::fs::File>,
  /// False only for `calibrate` in text mode while recording: the report is stdout's result, and the events go
  /// to the run alone.
  pub results_on_stdout: bool,
}

pub fn init_with(init: Init);
pub fn init(); // init_with(Init { record: None, results_on_stdout: true }): unchanged behavior
```

- A private `fn json_layer<S, W>(writer: W) -> fmt::Layer<...>` holds the five builder calls, and all three
  layers are made by it, so the record cannot drift from the streams in format.
- The record layer's filter is `filter_fn(is_result).or(filter_fn(is_failure)).or(EnvFilter::new("info"))`:
  results and failures always, logs at `info` whatever `RUST_LOG` says (owner decision 8).
- The result layer is an `Option` (tracing-subscriber implements `Layer` for `Option<L>`), absent when
  `results_on_stdout` is false. This is the only exception to D15's "results go to stdout, always", and it is only
  taken when the same results are in the run and the report is on stdout.
- `subscriber(logs, results, filter)` stays as it is for existing callers; a `subscriber_with` takes the record
  writer and the flag. `testing::capture_recorded(level, results_on_stdout, emit)` returns a `Captured` with a
  third `record: Vec<String>`.

### The trainer

- Flags on `Cli`: `--runs <DIR>` (`env = "RECORD_SIMILARITY_RUNS"`, default `runs`) and `--no-run`
  (`conflicts_with = "runs"`).
- `main`: parse; unless `--no-run`, build a `Spec` (kind `train`, slug from `--out`, `BUILD.stamp()`, argv,
  `continues` from `--resume`: the checkpoint's `provenance.json` `run` and `trained.step`, read with
  `Provenance::read`, absent fields left out) and `Run::start` it. If that fails, `similarity_log::init()` and
  `failed` with the reason (naming `--no-run`), and exit 1 before any work. Otherwise `init_with` the record, log
  `run recorded` (id, dir) at info, call `cli.run(Some(&run))`, and `run.finish(...)` with the outcome.
- `Cli::run(&self, run: Option<&Run>)`. Tests pass `None` and so never write into the repository.
- In `run`: after `lineage` is computed, `note_options` (the serialized `TrainOptions`, `lineage.trained.options`),
  `note_model` (role `base`, the base dir and digest from
  `lineage.trained.base`) and `note_input` for each `DataFile`; set `lineage.run` to the run id before the first
  checkpoint so that every checkpoint and the final model name it; after `save_model`, `note_output` (role
  `model`, `--out`, the digest the embedder reports on the verifying load).
- `Resumed` entries get `run` from the checkpoint's own provenance, in `Cli::lineage`.

### `calibrate`

- The same two flags on `CalibrateArgs` only. `serve`, `import`, `compare`, `model fetch`, and `calibrate-diff` are
  not recorded (owner decision 2).
- `main` builds the spec (kind `calibrate`, slug from the pairs stem and split, a stamp from
  `Build { tool: "record-similarity calibrate", ..BUILD }`), and initializes logging with
  `results_on_stdout: self.ndjson`.
- `CalibrateArgs::run(self, run: Option<&Run>)`: after loading, `note_model` (role `scored`, dir, content digest,
  fingerprint) and `note_input` for the pairs file (`calibration.input.sha256`, pairs scored) and the config file.
  With `--ndjson`, unchanged. Without it, the text report goes to stdout as now and, when recording, also to
  `note_report`, and `events::emit` runs as well (its output reaches only the record). That repeats the bootstrap
  behind the intervals once, which is the price of a text-mode run being data too.

### `runs list`

`record-similarity runs list [--runs DIR] [--kind KIND] [--limit N]` (owner decision 5), a report for people
written with `writeln!` to a locked stdout like `calibrate`'s. One line per run, newest first:

```text
20261111T150912Z-calibrate-pairs-full-holdout  calibrate  finished      41.2s  f46e48ee0ea2dd02-…  pairs.full.jsonl
20261111T142301Z-train-mdbr-leaf-mt-ft         train      killed      1h02m   f46e48ee0ea2dd02 → (none)  pairs.full.jsonl
```

Directories that are not runs are reported in one trailing line with their count, and each as a `warn` log. No
`--ndjson`: the manifests are JSON already (`jq -s . runs/*/run.json`). A `just runs` recipe runs it.

## The work, in order

One commit each. `just ci` passes after every one.

1. **Shared types.** `Stamp`, `Build::stamp`, `Provenance::run`, `Resumed::run` in `similarity-core`, with tests.
   Nothing writes the new fields yet.
2. **The record layer.** `similarity-log`: `json_layer`, `Init`, `init_with`, `subscriber_with`,
   `capture_recorded`, with tests. `init()` behaves exactly as before.
3. **The registry crate.** `crates/similarity-runs`: id and slug, `Manifest` and friends, `Run`, `list`, `open`,
   with tests. Workspace member and workspace dependency. `runs/` in `.gitignore` and `.dockerignore`.
4. **Recording training.** Trainer flags, `main` wiring, `Cli::run(Option<&Run>)`, the run id in provenance,
   `continues`. Run `just test-real` once (CLAUDE.md asks for it after touching the trainer, though nothing here
   reaches a training step).
5. **Recording calibration, and listing.** Calibrate flags and wiring, text-mode events, `report.txt`,
   `runs list`, `just runs`.
6. **Documentation.** Everything under [Documentation to update](#documentation-to-update).

## Tests

None needs the model, the network, or Qdrant: the trainer's tests use `write_tiny_model` and the server's use
`offline::testing`, and every run root is a `tempfile::tempdir`.

`similarity-core` (`provenance.rs`):

- `a_provenance_file_written_before_runs_existed_reads_with_no_run`
- `a_build_stamp_carries_what_the_provenance_fields_carry`

`similarity-log`:

- `a_recorded_run_gets_logs_results_and_failures_in_one_file_in_the_order_emitted`
- `the_record_keeps_info_logs_that_rust_log_hides_from_stderr`
- `results_kept_off_stdout_still_reach_the_record`
- `a_line_in_the_record_is_the_line_the_streams_carry` (same keys and values, timestamp aside)
- `without_a_record_logging_is_exactly_as_before`

`similarity-runs`:

- `the_start_time_in_an_id_is_utc_to_the_second` (0, 1_700_000_000, a leap day, the last second of a year)
- `a_slug_keeps_only_lowercase_letters_digits_and_single_dashes`
- `two_runs_started_in_the_same_second_get_different_directories`
- `a_started_run_reads_back_as_running_with_its_build_host_and_arguments`
- `a_finished_run_records_its_end_and_how_long_it_took`
- `a_failed_run_records_the_error_it_ended_with`
- `a_run_dropped_without_an_outcome_is_recorded_as_failed`
- `a_running_run_whose_lock_nobody_holds_reads_as_killed`
- `a_running_run_whose_lock_is_held_reads_as_running`
- `a_manifest_is_never_seen_half_written` (no `run.json.partial` left after any `note_*`)
- `a_note_that_cannot_be_written_warns_and_does_not_stop_the_run` (the directory made read-only after `start`)
- `listing_puts_the_newest_run_first_and_reports_a_directory_that_is_not_a_run`
- `a_manifest_from_a_later_format_is_refused_by_name`
- `a_kind_this_version_does_not_know_is_still_listed`
- `a_root_that_cannot_be_created_is_an_error_before_anything_is_written`
- A `properties` module with one rule for every input: `an_id_is_always_one_safe_path_component` (any slug text,
  any time). It belongs there by CLAUDE.md's rule, though the crate is new.

`similarity-train` (`main.rs`):

- `a_recorded_training_run_keeps_its_progress_results_inputs_and_output`
- `the_model_and_its_checkpoints_name_the_run_that_wrote_them`
- `a_resumed_run_gets_its_own_directory_naming_the_run_and_step_it_continues`
- `a_resume_is_accepted_although_the_run_id_differs` (the id is not in `RunIdentity`)
- `a_resume_from_a_checkpoint_written_without_a_run_still_records_where_it_came_from`
- `no_run_and_runs_conflict_and_the_runs_root_can_come_from_the_environment`

`similarity-server` (`main.rs` and `offline`):

- `a_recorded_calibration_reads_back_as_the_same_run_calibrate_diff_reads` (`events::read` on the record equals
  `events::read` on the `--ndjson` stdout)
- `a_text_calibration_still_records_its_events_and_its_report_and_stdout_is_unchanged`
- `an_ndjson_calibration_writes_the_same_stdout_with_or_without_a_run`
- `runs_list_prints_one_line_per_run_newest_first_and_says_killed_where_the_process_is_gone`

Not run by `cargo test`: a real `SIGKILL` of a live process (a test would have to spawn the binary). Check it by
hand once: start `just finetune ...`, `kill -9` it, `just runs`.

## Documentation to update

**README.md**

- Commands table under "Running the server": `record-similarity runs list`.
- "Fine-tuning the model", "Output": every run is also kept in `runs/<id>/`; `> results.ndjson` becomes optional.
  Add `--runs` and `--no-run` to the options table.
- "Choosing thresholds": a calibration is kept in `runs/<id>/` in both modes, and `calibrate-diff` reads a run's
  `events.ndjson` directly. Re-run the example before quoting it (CLAUDE.md).
- A short "Where a run is kept" section: the layout, `run.json`, statuses and what `killed` means, `runs list`,
  that `runs/` is ignored by git and must stay where real data was used, and how to read it with `jq`.
- "Project layout": `similarity-runs`, and `runs/`. Also fix the stale lines there: `feature-suggestions.md` says
  "none built yet", and `plan-features.md` is described as a plan for the first five.

**CLAUDE.md**

- Workspace layout: `crates/similarity-runs`: the run registry (manifests, the run directory, reading runs). No
  network; the server and the trainer both use it.
- Code style, logging bullet: a run's `events.ndjson` is a third layer built by the same function in
  `similarity-log`, never a second subscriber or a tee with its own format.
- Things easy to get wrong: recording starts in `main`, never in `Cli::run`, so tests cannot write into `runs/`;
  a run directory is notes about a run and never an input to a model, a threshold, or a verdict; a resume is a new
  run linked by `continues`; the run id must never enter `RunIdentity`; `killed` is concluded by readers, never
  written; the run file and the streams must keep the same event names (D17 applies to a run's events as it does
  to stdout's).
- Commands: `just runs`.

**docs/decisions.md**: a new **D22, "A run is a directory of what it said"**: why local files and not a tracker;
why a third layer with its own level rather than a tee; the single D15 exception for text-mode calibration; why
`killed` is derived from a lock; why a resume is a new directory; why the run id is in `provenance.json` (D18) and
not in `RunIdentity` (D13). Amend D15's "Results go to stdout" paragraph to point at D22, and D18's list of what
a trained directory records to include the run id.

**docs/feature-suggestions.md** and **docs/plan-features.md**: mark feature 2 built and point at D22.

## What features 4 and 5 may rely on

- `similarity_runs::{list, open, Listed, Manifest, Status, MANIFEST_FILE, EVENTS_FILE, REPORT_FILE, KIND_*}`.
  There is no `Manifest::read`; `open(dir)` returns the manifest with the status a reader concludes.
- Names in `run.json` as written above: `argv` (not `args`), the commit and dirty flag under `build`, and the
  training options under `options`. Feature 5 (planned in parallel) first assumed otherwise; this list is what it
  should code against.
- `events.ndjson` holds every result and failure the process emitted and its logs at `info` and above, in emission
  order, each line exactly as the streams format it; `target` tells logs (`record_similarity_*`), results
  (`result`), and failures (`failure`) apart.
- A `calibrate` run's `events.ndjson` reads with `offline::events::read` in both modes. Its `calibration` event
  carries the fingerprint, `split`, and `pairs_sha256` that feature 4's checks need.
- A model directory's `provenance.json` `run` names the run that wrote it, and that run's `outputs` names the
  directory back. A training chain is followed through `continues.run`; the earlier run's events stop mattering
  at `continues.step`.
- `format` in `run.json` changes only when a reader built for `1` would misread the file; fields are added, never
  repurposed, as for events (D17).

## What can only be checked where the data lives

- Real runs on Metal and CUDA, hours long: that the record keeps up, and how large `events.ndjson` gets for a real
  calibration (several lines per pair; a 10,000-pair file is likely tens of megabytes).
- That advisory locks work on the work machine's filesystem. On NFS they may not, in which case a killed run lists
  as `running` and never as `killed`; the plan reports that rather than guessing.
- How the work environment's scheduler ends a job (`SIGTERM`, `SIGKILL`, out-of-memory), and that each reads as
  `killed`.
- Whether keeping hostnames, paths, pair ids, and owners in `runs/` on work machines is acceptable. The directory
  is never committed here, and real data never enters this repository.

## Decisions for the owner, with the default to use meanwhile

1. **A resume gets its own directory, linked by `continues`,** rather than appending to the original's.
   Default: its own directory, for the reasons in item 4 above.
2. **Which commands are recorded.** Default: `record-similarity-train` and `calibrate` only. Not `throughput` (an
   example printing text; converting it to events is its own change, best done with feature 12), not `import`
   (operations, whose logs belong to the scheduler, and which takes a secret), not `calibrate-diff` and `compare`
   (pure functions of their inputs).
3. **Recording is on by default,** and a run directory that cannot be created fails the command before it starts.
   Default: yes to both; `--no-run` turns it off. The alternative, warning and carrying on unrecorded, loses the
   very runs this is for.
4. **A text-mode calibration still records its events.** Default: yes, at the cost of computing the intervals
   twice, so that every calibration can be compared and promoted later.
5. **Where `runs list` lives.** Default: `record-similarity runs list`, beside `calibrate` and `calibrate-diff`. The
   alternative, a third binary `record-similarity-runs` as feature 5 imagines, changes CLAUDE.md's "anyhow only in
   the two binaries" and can wait for feature 5 to need it.
6. **The root and its flag.** Default: `./runs`, `--runs DIR`, `RECORD_SIMILARITY_RUNS`.
7. **Recording the hostname and CPU model.** Default: yes, best effort, local files only.
8. **The record's own log level.** Default: fixed at `info`, whatever `RUST_LOG` says, so that the loss curve is
   always kept. Debug logs never reach the record.
9. **`similarity-runs` depends on `similarity-core`** to share `Stamp` and `DataFile`. Default: yes. The
   alternative keeps Burn out of the registry's build by duplicating the stamp, which only matters if feature 5
   ships a binary that does not otherwise need Burn.
10. **No retention.** Default: runs are never deleted by any tool; a person deletes a directory. A `runs prune` can
    come later if `runs/` grows inconveniently.

## Size

**Medium.** No new dependency and no change to a verdict, a threshold, a layout, or a training step, so no
re-calibration and no parity fixture. The work is a new small crate (ids, an atomically rewritten manifest, a lock,
a lister; perhaps 600 lines with tests), a third layer in `similarity-log`, wiring in two `main`s, two optional
provenance fields, and documentation across four files. The care goes into the order of creation, the status
rules, and keeping tests out of `runs/`.

## Out of scope

- The W&B exporter (`docs/feature-suggestions.md`, "On Weights & Biases"): `events.ndjson` is what it would replay.
- Run reports, comparisons, charts (feature 5), the model registry and promotion gate (feature 4), evaluation at
  every checkpoint (feature 3; its events will land in the record with no change here), and sweeps (feature 6).
- Recording `throughput` (feature 12 is the natural place to turn it into events).
- Pruning old runs, and capturing GPU names or memory.
