# Plan: model registry and promotion gate (feature 4)

A plan for feature 4 of `docs/feature-suggestions.md`, written against the code on `initial-implementation` as of
2026-11-11, before any of it was built. Read `CLAUDE.md` first, then D3, D11, D12, D17, D18, and D20 in
`docs/decisions.md`. Delete this file, or fold what is still true into a decision record, when the work is done.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on unbuilt features](#dependencies-on-unbuilt-features)
- [What can only be validated at work](#what-can-only-be-validated-at-work)
- [Decisions for the owner, with the default to use meanwhile](#decisions-for-the-owner-with-the-default-to-use-meanwhile)
- [Size](#size)

## What it is for

Two mistakes `CLAUDE.md` calls easy to make depend today on someone having read the README: serving thresholds
measured for one model with another (D3), and changing the model without a new collection (D12). This feature turns
both into things a tool checks, and keeps a record of which model was put into service, on what evidence, and when.

It has three parts:

1. **A config that names the model its thresholds were measured for.** `model.expected_fingerprint`. `serve` and
   `import` refuse to start when the loaded model's fingerprint is another one. This is the part that protects
   production, and it needs no registry at run time.
2. **A gate.** `record-similarity model promote --config candidate.toml --calibration holdout.ndjson` refuses a
   candidate whose holdout calibration is of another model, is not a holdout, or is worse than the model in
   service on the same holdout pairs. On success it prints the two config lines that tie the config to the model
   and to the right collection.
3. **A registry.** An append-only file of `registered`, `promoted`, and `retired` entries, kept outside every model
   directory, which the gate reads to know what is in service and writes to record what it let through.

## What already exists

Verified in the code. Most of the feature is assembly of these.

- **What identifies a model.** `ModelFiles::digest` (`crates/similarity-core/src/model/sentence.rs`) is a 16-hex
  digest of the files the encoder reads plus `tokenizer.json`. `Embedder::load` (`crates/similarity-core/src/
  embed.rs`) builds the fingerprint as `format!("{digest}-d{dimension}-t{max_tokens}")`, exposed as
  `Embedder::fingerprint` and `Embedder::content_digest`. `Schema::layout_id(fingerprint)` (`schema.rs`) is
  `"{fingerprint}|{sorted field names}"`, and `engine::store_layout` pairs it with the composite width.
- **Where a model came from.** `similarity_core::provenance::{Provenance, Build, PROVENANCE_FILE}`.
  `Provenance::read(dir)` returns `Ok(None)` for a directory without the file. `Build` is the stamp each binary
  states once (`BUILD` in `crates/similarity-server/src/main.rs`, tool name `record-similarity model fetch`).
- **The server already reads provenance at start-up** (`load_embedder` in `main.rs`): it logs it on the
  `model loaded` line and warns if `Provenance::digest` differs from `Embedder::content_digest`. Nothing refuses.
- **A calibration is a file that names its model.** `offline::events::emit` (`offline/events.rs`) writes a
  `calibration` line with `fingerprint`, `dimension`, `max_tokens`, `threshold`, `anchor_threshold`,
  `reject_on_measure_change`, `pairs_sha256` (of the whole pairs file, before selection), `split` (the `--split`
  filter, absent when there was none), `min_confidence`, and `seed`; then `schema_field` and `schema_quantity`
  lines; then the pairs. `offline::events::read` returns `Run { info: RunInfo, calibration: Calibration }`;
  `RunInfo` keeps `fingerprint`, the policy, and the field and quantity names, and drops the weights and floors.
- **The comparison is built.** In `offline/diff.rs`: `same_pairs(&Input, &Input)` (same file hash, split, and
  confidence), `diff(&Calibration, &Calibration) -> Diff`, `rejudge(&Run, &Schema, &Policy)` (judges a run's
  measurements under another config, no model; refuses other field or quantity names), and
  `failure(&Diff, FailOn) -> Option<String>` with `FailOn::{NewFalseAccepts, AnyFlip, ErrorsUp}`. `calibrate-diff
  --fail-on` in `main.rs` is the existing gate.
- **The `model` subcommand group exists** (`ModelCommand` in `main.rs`) with one member, `Fetch`.
- **Config is strict.** Every config struct in `crates/similarity-server/src/config.rs` is `deny_unknown_fields`,
  and `config.example.toml` is parsed by a unit test (`config.rs`) and by `tests/real_e2e.rs`.
- **Splits.** The trainer's `--holdout-split` defaults to `holdout`; the data request assigns 80% `calibrate`, 20%
  `holdout` by brand hash. `data/calibration.sample.full.jsonl` has 42 `calibrate` pairs and 7 `holdout` pairs, so
  the whole flow can be exercised here on synthetic data.
- **No new dependency is needed.** `serde`, `serde_json`, `sha2`, and `toml` are already dependencies of
  `similarity-server`.

What does not exist: `similarity-runs` (feature 2, not built), any registry, any config key tying thresholds to a
model, and any check at start-up that a same-width collection holds vectors of the configured model (a mismatch is
found only when a stored record is read, as a `FAILED_PRECONDITION`, per D12).

## What the suggestion got wrong or left open

1. **"Keyed by model fingerprint" conflates two things.** A model directory has a *digest*. The *fingerprint* adds
   the width and token limit, which come from the config, so one directory has as many fingerprints as configs.
   The registry keys a model by digest and a promotion by fingerprint and layout id.
2. **"Prints the thresholds from that calibration" is wrong in two ways.** `calibrate` suggests only the overall
   threshold and quantity limits, never `anchor_threshold` or field floors, so there is no complete set to print.
   And a threshold read off the holdout makes the holdout a validation set (the caveat under feature 3). The gate
   should instead judge the candidate config's own thresholds on the holdout, which `diff::rejudge` does from the
   calibration's measurements without embedding anything. Thresholds stay measurements the operator chose (D3);
   the tool never writes one.
3. **The README already contradicts this.** "Deploy it", step 2, says "Set the thresholds from the holdout
   calibration of the new model". That has to change with this feature (see the decisions).
4. **The server cannot read a registry at start-up in production.** The registry is a local file next to
   `models/`, which `.gitignore` excludes and the Docker image does not contain. The check that matters at start-up
   is that the config and the model agree, and that can travel in the config (`model.expected_fingerprint`).
5. **A collection name cannot be "derived from the layout id" directly.** The layout id contains `|` and `,` and
   is long. Derive it from a hash of the layout id. And when the layout does not change (a threshold or weight
   change only), the collection must *not* change: D12 says those need no re-index.
6. **"Worse than the current one on the same holdout file" needs a baseline that may not exist.** The promoted
   model's calibration was made on whatever holdout existed then. Once the holdout grows, `diff::same_pairs` is
   false, and a fresh calibration of the model in service is required (`--baseline`).
7. **A `calibrated` entry would duplicate the calibration.** The calibration file (or, after feature 2, its run
   directory) is the record. Copying its numbers into the registry makes a second source; CLAUDE.md wants every
   number from `offline::stats`. Dropped: a `promoted` entry points at its calibration by path and SHA-256 and
   carries only the error counts the gate compared, computed by the same `Calibration` the report uses.
8. **"Fit: `similarity-runs`" does not work.** The gate needs `offline::{events, diff}`, which live in
   `similarity-server`, and the trainer must not link the server (D13). The registry belongs in
   `similarity-server`. Nothing in the trainer needs it: promotion registers a model implicitly.
9. **Left open by the suggestion:** what "currently promoted" means after a retirement, whether a worse model can
   ever be promoted (a rollback is exactly that), how small a holdout may be, and how concurrent promotions behave.
   Each is settled below or listed as a decision.

## Design

### 1. `model.expected_fingerprint`

```toml
[model]
dir = "models/mdbr-leaf-mt"
output_dim = 256
# The fingerprint the thresholds below were measured with. `serve` and `import` refuse any other model.
# `record-similarity model promote` prints this line.
expected_fingerprint = "f46e48ee0ea2dd02-d256-t256"
```

- `ModelConfig` gains `pub expected_fingerprint: Option<String>`, default `None`, so every existing config parses.
- `ModelConfig::check_fingerprint(&self, actual: &str) -> Result<(), FingerprintMismatch>` in `config.rs`, a pure
  function, with an error that names both fingerprints and says that thresholds do not transfer between models
  (D3) and how to measure new ones.
- `serve` and `import`: a mismatch is an error before anything listens or connects. An absent key is a `warn` on
  the `model loaded` line's heels (`thresholds_tied = false`), once.
- `compare`, `calibrate`: a mismatch is a `warn`, never an error. Measuring a new model with a copy of the old
  config is the documented way to evaluate it.
- It is an assertion, not an input: it is not part of the fingerprint or the layout id, and changes no vector
  (D12 untouched).
- The value quoted above is what the pinned model, `output_dim = 256`, and `max_tokens = 256` should produce
  (D18 quotes the digest); it must be read from a real run before it is written anywhere.

### 2. The registry file

Default path `models/registry.ndjson`, overridden by `--registry` or `RECORD_SIMILARITY_REGISTRY`. It sits beside
the model directories, never inside one (D18), and is ignored by git with them, because its entries name local
paths. One JSON object per line, written with `serde_json`, append-only.

It is a data file the tool owns, like `provenance.json`, and not a log or result stream: it is not written through
`similarity-log`, so D17's flatness rule (forced by `tracing`'s scalar fields) does not apply and entries may nest.
Like D17's events, the names are an interface: add fields, never rename or repurpose one. `format` is 1.

```rust
// crates/similarity-server/src/registry.rs

/// One line of the registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "entry", rename_all = "snake_case")]
pub enum Entry {
  Registered(Registered),
  Promoted(Box<Promoted>),
  Retired(Retired),
}

pub struct Stamp {            // flattened into every entry
  pub format: u32,
  pub at: u64,                // seconds since the epoch
  pub tool: String,           // "record-similarity model promote", ...
  pub version: String,
  pub commit: String,
  pub dirty: Option<bool>,
}

pub struct Registered {
  pub stamp: Stamp,
  pub digest: String,         // ModelFiles::digest
  pub dir: String,            // as given
  pub provenance: Option<Provenance>, // the model's own file, copied, if it had one
}

pub struct Promoted {
  pub stamp: Stamp,
  pub digest: String,
  pub fingerprint: String,
  pub layout: String,         // Schema::layout_id
  pub model_dir: String,
  pub collection: String,     // what the printed fragment names
  pub config: PromotedConfig, // [policy], [fields], [quantities] as judged, so a later gate can judge again
  pub calibration: CalibrationRef,
  pub compared_with: Option<Comparison>, // None for the first promotion
  pub override_reason: Option<String>,
}

pub struct CalibrationRef {
  pub path: String,
  pub sha256: String,         // of the file's bytes, to notice it changing later
  pub run_id: Option<String>, // feature 2, when the calibration came from a run directory
  pub pairs_sha256: String,
  pub split: String,
  pub min_confidence: Option<String>,
  pub edits: usize,
  pub recycles: usize,
  pub false_rejects: usize,   // under `config`, from the same `Calibration` the report renders
  pub false_accepts: usize,
}

pub struct Comparison {
  pub fingerprint: String,    // the model that was in service
  pub calibration_sha256: String,
  pub fail_on: String,        // "errors-up", ...
  pub false_rejects: usize,   // the baseline's, over the shared pairs
  pub false_accepts: usize,
  pub flips: usize,
}

pub struct Retired {
  pub stamp: Stamp,
  pub digest: String,
  pub reason: String,
}
```

`PromotedConfig` reuses `PolicyConfig`, `FieldConfig`, and `QuantityConfig` from `config.rs`, which gain
`Serialize`. Absent optional values are left out, never written as `null` (as in `provenance.json`).

**Reading** is strict: a line that is not an entry, or an entry with a `format` above 1, is an error naming the
line number (a newer tool's file is not guessed at, as `events::read` refuses an unknown reason code). A missing
file is an empty registry.

**Appending** opens with `append(true)`, writes one whole line with a single `write_all`, and calls `sync_all`.
Two promotions at the same moment both compare against the same model in service; that is accepted and
documented, not locked against. A torn last line after a crash is reported by line number and fixed by hand.

**What is in service** (`Registry::current`): the latest `promoted` entry whose digest has no `retired` entry after
it. Retiring the model in service therefore makes the previous one current again, which is what a rollback is;
`model list` shows which. A retired model can be promoted again only with `--override`.

### 3. The gate

A pure function, so that every refusal is tested without a model or a file:

```rust
pub struct Candidate<'a> {
  pub fingerprint: &'a str,      // of the model in the config, at the config's width and token limit
  pub layout: &'a str,
  pub schema: &'a Schema,
  pub policy: &'a Policy,
  pub run: &'a Run,              // the candidate's calibration
  pub run_sha256: &'a str,
}

pub struct Baseline<'a> {
  pub promoted: &'a Promoted,    // what is in service
  pub run: &'a Run,              // its calibration (from the registry's path, or --baseline)
  pub run_sha256: &'a str,
}

pub struct GateOptions<'a> {
  pub holdout_split: &'a str,    // "holdout"
  pub fail_on: FailOn,           // diff::FailOn, default ErrorsUp
  pub override_reason: Option<&'a str>,
}

pub enum Refusal {
  OtherModel { calibrated: String, configured: String },
  NotHoldout { split: Option<String>, wanted: String },
  MissingLabel { edits: usize, recycles: usize },
  OtherSchema(String),           // diff::rejudge's refusal, passed through
  Retired { digest: String, reason: String },
  BaselineOtherModel { recorded: String, given: String },
  BaselineChanged { path: String },          // file hash differs from the registry's
  BaselineOtherPairs,                        // !diff::same_pairs; says to pass --baseline
  Worse(String),                             // diff::failure's text
}

pub fn gate(candidate: &Candidate, baseline: Option<&Baseline>, retired: Option<&Retired>, options: &GateOptions)
  -> Result<Approval, Refusal>;
```

In order, stopping at the first refusal:

1. `run.info.fingerprint == candidate.fingerprint`, or `OtherModel`. Never overridable.
2. `run.calibration.input.split == Some(holdout_split)`, or `NotHoldout`. A strict-format run has no split and is
   refused. Never overridable.
3. At least one labeled edit and one labeled recycle, or `MissingLabel`. Never overridable. Fewer than 40
   recycles (the data request's own floor for a holdout) is a warning in the report, not a refusal.
4. Judge the candidate under its own config: `diff::rejudge(run, schema, policy)`. Its refusal is `OtherSchema`.
   What the calibration's header says the thresholds were does not matter; what is promoted is the config.
5. If the digest has a `retired` entry and no override, `Retired`.
6. With a baseline: its run must carry `promoted.fingerprint` (`BaselineOtherModel`), its bytes must hash to
   `promoted.calibration.sha256` when it was read from the registry's path (`BaselineChanged`), and
   `diff::same_pairs` must hold against the candidate's input (`BaselineOtherPairs`). Then rejudge it under
   `promoted.config`, `diff::diff(baseline, candidate)`, and `diff::failure(&diff, fail_on)`; a failure is `Worse`
   unless `override_reason` is given.
7. Without a baseline (the first promotion, or everything retired), skip 6 and say so in the report.

`Approval` carries the judged candidate `Calibration`, the `Diff` if any, and the collection name.

**The collection.** If the candidate's layout equals the promoted layout, the candidate config's
`qdrant.collection` is kept, since nothing needs re-indexing (D12). Otherwise the name is the candidate config's
collection with any previous `-<12 hex>` suffix stripped, plus `-` and the first 12 hex digits of the SHA-256 of
the layout id: `records-3fa9c1d2e0b4`. Only lowercase hex and `-` are added, which Qdrant accepts.

### 4. Commands

All in `crates/similarity-server/src/main.rs` under `ModelCommand`, wiring only; behavior in `registry.rs`.
Reports are text for a person on a locked stdout with `writeln!`, as `model fetch` does (D15). A refusal is an
`anyhow` error, so it goes through `similarity_log::failed` and exits 1, and appends nothing.

```sh
record-similarity model promote --config candidate.toml --calibration holdout.ndjson \
  [--baseline current-holdout.ndjson] [--fail-on errors-up|new-false-accepts|any-flip] \
  [--holdout-split holdout] [--override "reason"] [--registry models/registry.ndjson]
record-similarity model register models/mdbr-leaf-mt-ft [--registry ...]
record-similarity model retire <digest> --reason "..." [--registry ...]
record-similarity model list [--registry ...]
```

- **`promote`** loads the config, loads the embedder the way `offline()` does (so the fingerprint is computed by
  `Embedder::load`, not re-derived), reads the calibration with `events::read`, hashes it, reads the registry,
  finds the baseline (`--baseline`, else the path recorded in the current `promoted` entry), runs `gate`, appends
  a `registered` entry if the digest has none, then the `promoted` entry, and prints:

  ```text
  promoted f46e48ee0ea2dd02-d256-t256 (models/mdbr-leaf-mt-ft), holdout of 412 pairs: 3 false rejects, 1 false accept
  against   a3dde74c22049b9d-d256-t256 on the same pairs: 5 false rejects, 1 false accept, 4 verdicts changed

  Add to candidate.toml:

    [model]
    expected_fingerprint = "f46e48ee0ea2dd02-d256-t256"

    [qdrant]
    collection = "records-3fa9c1d2e0b4"   # new layout: import into it before serving (README, Re-indexing)
  ```

  The numbers in that text are illustrative of the format only. Every number in the real report comes from the
  judged `Calibration`, `offline::stats::error_rates` (with its intervals), and the `Diff`.
- **`register`** records a model directory before any promotion, for `list`. Registering a digest already
  registered at the same path appends nothing. It never writes into the model directory.
- **`retire`** appends a `retired` entry; a digest never registered is an error.
- **`list`** prints one line per registered digest: directory, when registered, the tool that wrote it (from
  provenance), each promotion with its fingerprint, whether it is the one in service, and any retirement.

`justfile` gains `promote *args` alongside `calibrate-diff`.

## The work, in order

Each step is one commit and leaves `just ci` green.

1. **`model.expected_fingerprint`.** The key, `check_fingerprint`, the refusal in `serve` and `import`, the warning
   in `compare` and `calibrate`, the README configuration row. If the owner agrees to ship it in
   `config.example.toml`, set it from a real `just calibrate` run's model line and run `just test-real` (CLAUDE.md
   requires it after touching that file).
2. **The registry file.** `registry.rs` with the entry types, strict reading, appending, `current`, and
   `Serialize` on the three config structs. `model register`, `model retire`, `model list`. `--registry` and the
   environment variable.
3. **The gate.** `gate`, `Refusal`, `Approval`, the collection name. Pure functions over `Run`, `Schema`, `Policy`,
   and registry entries; no model. A small accessor on `events::RunInfo` only if one turns out to be needed (the
   header already has everything).
4. **`model promote`.** Wiring, the report, the `justfile` recipe. Run it by hand on
   `data/calibration.sample.full.jsonl --full --split holdout --ndjson` with the real model: a first promotion
   into a scratch registry, then a second with a threshold moved, to see a comparison and a refusal.
5. **Documentation**, below, and a decision record (D22).
6. **After feature 2 lands:** accept a run directory for `--calibration` and `--baseline` and record its id.

## Tests

None needs the model or Qdrant except the one marked. The calibrations are built with the tiny encoder
(`similarity_core::testing`) through `offline::calibrate` and `events::emit` captured with
`similarity_log::testing::capture`, or by hand with `offline::testing::scored`, as `diff.rs`'s tests do.

`config.rs`:

- `a_config_without_an_expected_fingerprint_parses_as_before`
- `serving_is_refused_when_the_model_is_not_the_one_the_thresholds_were_measured_for`
- `a_mismatch_names_both_fingerprints_and_says_thresholds_do_not_transfer`
- `the_expected_fingerprint_is_part_of_neither_the_fingerprint_nor_the_layout`
- `the_example_config_names_the_fingerprint_of_the_pinned_model`: `#[ignore = "needs the model: just test-real"]`,
  only if the key is shipped.

`registry.rs`, the file:

- `a_missing_registry_is_empty_and_has_nothing_in_service`
- `entries_read_back_exactly_as_they_were_appended`
- `appending_leaves_every_earlier_line_byte_for_byte_as_it_was`
- `a_line_that_is_not_an_entry_is_an_error_naming_its_line_number`
- `an_entry_of_a_newer_format_is_refused_rather_than_guessed_at`
- `what_is_absent_is_left_out_and_not_written_as_null`
- `the_model_in_service_is_the_latest_promotion_not_retired_since`
- `retiring_the_model_in_service_puts_the_one_before_it_back_in_service`
- `registering_a_model_twice_at_the_same_path_records_it_once`
- `registering_a_model_writes_nothing_into_its_directory` (D18: the directory listing and digest are unchanged)

`registry.rs`, the gate:

- `a_calibration_of_another_model_is_refused_whatever_else_is_true`
- `a_calibration_not_restricted_to_the_holdout_split_is_refused`
- `a_strict_format_calibration_is_refused_because_it_has_no_split`
- `a_holdout_without_both_labels_is_refused`
- `the_candidate_is_judged_under_its_own_config_not_the_thresholds_the_calibration_ran_with`
- `a_config_with_other_fields_than_the_calibration_is_refused`
- `the_first_promotion_has_nothing_to_compare_against_and_says_so`
- `a_candidate_that_gets_more_holdout_pairs_wrong_than_the_model_in_service_is_refused`
- `a_candidate_that_is_no_worse_is_promoted_and_the_comparison_is_recorded`
- `a_baseline_over_another_selection_of_pairs_is_refused_with_how_to_make_one`
- `a_baseline_calibration_that_changed_since_its_promotion_is_refused`
- `a_baseline_of_another_model_than_the_one_in_service_is_refused`
- `an_override_promotes_a_worse_candidate_and_records_the_reason`
- `an_override_never_passes_another_model_or_another_split`
- `a_retired_model_is_refused_unless_overridden`
- `the_collection_is_kept_when_the_layout_is_unchanged`
- `a_new_layout_gets_a_collection_named_by_its_hash_and_the_same_layout_always_the_same_name`

`properties` (`similarity-server`, D19), since it is a rule for every history:

- `the_model_in_service_is_never_one_retired_after_its_last_promotion`, over generated sequences of entries.

`main.rs` and end to end, with a tiny model written to a temporary directory:

- `promote_prints_a_fragment_that_parses_as_config_and_names_the_promoted_fingerprint`
- `a_refused_promotion_appends_nothing_to_the_registry`
- `model_promote_requires_a_config_and_a_calibration` (clap)

After step 3, run `PROPTEST_CASES=5000 just test` once. Before declaring done, `just ci`, and `just test-real` if
`config.example.toml` changed.

## Documentation to update

- **README.md.** "Fine-tuning, 4. Deploy it": replace step 2 ("Set the thresholds from the holdout calibration")
  with: choose thresholds on the `calibrate` split, calibrate the candidate on the holdout with `--ndjson`, run
  `model promote`, and copy the printed lines into the config. "Using a different model": point at `model promote`
  for the new collection name and the fingerprint line. "Configuration reference": `model.expected_fingerprint`,
  and a row in "What a change costs". "Re-indexing", step 1: the collection name `promote` prints. "Development"
  or a new "Promoting a model" subsection: the registry, `list`, `retire`, rollback. Every example re-run before
  it is quoted.
- **CLAUDE.md.** Commands: `just promote`. "Things that are easy to get wrong": the registry is append-only and
  lives outside every model directory; `model.expected_fingerprint` is an assertion and never part of the
  fingerprint or layout; the gate's model and split checks are never overridable; promotion never writes a
  threshold. The thresholds bullet gains "update `expected_fingerprint` with them when the model changes".
- **docs/decisions.md.** A new D22, "A config names the model its thresholds were measured for, and a gate
  promotes it", covering the three parts, why the server does not read the registry, why there is no `calibrated`
  entry, and why entries may nest (not written through `tracing`). One-line cross-references from D3 and D12.
- **docs/feature-suggestions.md and docs/plan-features.md.** Mark feature 4 built and note what came out
  differently.
- **.gitignore.** Nothing: `/models/*` already covers `models/registry.ndjson`.

## Dependencies on unbuilt features

- **Feature 2 (local run registry).** Not required; the gate works on plain `calibrate --ndjson` files. Assumed
  interface for step 6: a run directory holds `run.json` with at least `id`, `kind` (`calibrate`), and `status`
  (`finished`), and `events.ndjson` holding the run's result lines exactly as `calibrate --ndjson` writes them. If
  feature 2 captures only the text report for a `calibrate` run without `--ndjson`, `promote` refuses that run and
  says to re-run with `--ndjson`.
- **Feature 3 (evaluation at every checkpoint).** Not required. Its `validation` fold is where thresholds for a
  fine-tuned model would best be chosen, since the `calibrate` split is the fine-tuned model's training data;
  until then the holdout gate is what catches thresholds chosen on data the model has seen.
- **Feature 9 (pinned snapshot)** and **feature 18 (shadow evaluation)** are independent. A shadow's candidate
  would be a natural thing to promote next, and its collection is the one `promote` names.

## What can only be validated at work

Everything builds and is tested here on synthetic data, and the sample's 7 holdout pairs exercise every path. What
needs the real holdout: whether `errors-up` on the real holdout is a sensible default or flips on noise, whether
the holdout reaches the 40 recycles the data request asks for, the first real promotion of the base model (which
records the baseline every later candidate is compared with), and the first comparison of a fine-tuned model
against it. The registry those runs write stays at work; it names local paths and nothing in it belongs here.

## Decisions for the owner, with the default to use meanwhile

1. **A config whose `expected_fingerprint` disagrees with its model: refuse, or warn?** Default: `serve` and
   `import` refuse, as the server already refuses a collection of the wrong width; the offline tools warn.
2. **Ship `expected_fingerprint` in `config.example.toml`?** Default: yes, with the value read from a real run.
   `just serve` then refuses a changed `model.dir` until the thresholds are re-measured, which is the point.
3. **Where thresholds are chosen.** Default: on the `calibrate` split (a `validation` fold once feature 3 exists),
   judged by the gate on the holdout. The README's "from the holdout" is changed. Choosing on the holdout remains
   possible, since the tool cannot tell, but the docs stop recommending it.
4. **What "worse" means.** Default: `--fail-on errors-up`, the existing `diff::FailOn`, over the shared holdout
   pairs. It compares counts, not intervals, and the report prints the intervals beside it so a person can see
   when one pair decided it.
5. **What may be overridden.** Default: only the comparison and a retirement, with a required reason recorded in
   the entry. Another model and another split never.
6. **Where the registry lives.** Default: `models/registry.ndjson`, ignored by git with the models it describes,
   with `--registry` and `RECORD_SIMILARITY_REGISTRY` to keep it elsewhere (a shared drive at work).
7. **Collection naming.** Default: the configured name stem plus 12 hex digits of the layout id's hash, kept
   unchanged when the layout is.
8. **Minimum holdout.** Default: refuse only without both labels; warn below 40 recycles, the data request's
   number. Making 40 a refusal is a one-line change if preferred.

## Size

**Medium**, a few days. The comparison, the rejudging, the run reader, the fingerprint, and the provenance all
exist; what is new is one config key, one module of serde types and pure functions with a long list of small
tests, four thin subcommands, and the documentation. No new dependency, no change to any vector, threshold, event,
or model directory, and nothing that needs Qdrant.
