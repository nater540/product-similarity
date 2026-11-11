# Plan: run report and compare (feature 5)

A plan for feature 5 of `docs/feature-suggestions.md`, written against the code on `initial-implementation` at
`545ef20`, so that a session with no memory of the conversation that produced it can carry it out. Nothing here is
built. Read `CLAUDE.md` first, then D13, D15, D17, and D18 in `docs/decisions.md`. It depends on feature 2 (the
local run registry), which is planned separately; the interface assumed from it is stated below and nothing more.

## Contents

- [What it is for](#what-it-is-for)
- [What exists to build on](#what-exists-to-build-on)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [What is assumed from feature 2](#what-is-assumed-from-feature-2)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)
- [Out of scope](#out-of-scope)

## What it is for

After two fine-tuning runs and a few calibrations, the questions are "how did these runs differ, and which is
better?" Today that is answered with `jq` over files somebody remembered to keep, and with `calibrate-diff` for
the calibration half. This feature reads the run directories feature 2 keeps and answers it two ways:

- `record-similarity runs compare A B`: a text table for a terminal, written to a locked stdout like every other
  report for people (D15).
- `record-similarity runs report A [B ...] --out report.html`: one self-contained HTML file with inline SVG
  charts (loss curves, score histograms, per-field ranges) and the lineage of each run side by side. No
  JavaScript, no network, opens offline, sits next to the data it describes.

It is a reader. It computes no new verdict, never embeds, loads no model, and needs no Qdrant.

## What exists to build on

Verified in the code as it is now.

**Training events** (`crates/similarity-train/src/main.rs`):

- `Window::log` writes a `training` line every `--log-every` steps (default 10) and on the last step: `step`,
  `of`, `epoch`, `pair_loss`, `group_loss` (means over the window, each absent if no batch of that kind fell in
  it), `grad_norm` (window mean, before clipping, per `StepReport::grad_norm`), `rate` (the learning rate, three
  significant figures), `elapsed_s` (since this process started). **It is a log, not a result**: target
  `record_similarity_train`, on stderr, filtered by `RUST_LOG`.
- `Checkpoints::after` writes a `checkpoint` log line: `step`, `epoch`, `path`.
- `Cli::run` writes a `model loaded` log line with `resumed_at`, `device`, `native_width`, `widths`, `dropout`.
- `Outcome::report` writes the results (target `similarity_log::RESULT`, stdout): one `separation` per split
  (`training`, `held_out`), stage (`before`, `after`), and width, with `edits`, `edits_min/mean/max`, `recycles`,
  `recycles_min/mean/max`, `gap_of_means`, and on a `before` line `compared_with` (`base model` or `checkpoint`);
  then one `finished` with `out`, example counts, `steps`, `resumed_at`, `elapsed_s`.
- The tests in that file already capture these streams with `similarity_log::testing::capture` and assert on
  `training` and `finished` lines.

**Lineage** (`crates/similarity-core/src/provenance.rs`): `Provenance` (`tool`, `version`, `commit`, `dirty`,
`digest`, `fetched`, `trained`), `Trained` (`base: Base`, `data: Vec<DataFile>`, `data_fingerprint`, `options` as
`serde_json::Value`, `max_tokens`, `device`, `resumed`, `step`, `total_steps`), `Provenance::read(dir)`. Every
model directory and checkpoint written since feature 1 has one. A fingerprint is `{digest}-d{dim}-t{max_tokens}`
(`Embedder::load`, `similarity-core/src/embed.rs`), so a calibration's `fingerprint` names the model's `digest`
as its prefix.

**Calibration events and their reader** (`crates/similarity-server/src/offline/`):

- `events::emit` writes `calibration` (fingerprint, dimension, max_tokens, threshold, anchor_threshold,
  pairs_sha256, split, seed), `schema_field`, `schema_quantity`, one `pair` per pair (with `similarity`, `label`,
  `accepted`, `wrong`) and its `pair_field`, `pair_quantity`, `pair_reason` lines, `field_range`,
  `quantity_range`, `suggestion`, `quantity_suggestion`, `slice`, `operating_point`, `ignored_key`, `summary`.
- `events::read(&str) -> anyhow::Result<Run>` reads them back, skipping non-result lines, into `Run { info:
  RunInfo, calibration: Calibration }`.
- `offline::stats` is the one source of every calibration number (`score_range`, `field_range`, `slices`,
  `error_rates`, `operating_points`, ...).
- `offline::diff` already **compares two calibrations**: `diff::diff` matches pairs by id and returns `Diff`
  (flips sorted by score change, totals, slice changes, field range shifts); `same_pairs` and `--intersect`;
  `rejudge` for `--config-b`; `render_diff` for the text report; `emit` for `flip` and `diff` events; `failure`
  for `--fail-on`. The command is `CalibrateDiffArgs` in `crates/similarity-server/src/main.rs`.

**Logging** (`crates/similarity-log/src/lib.rs`): every line has `timestamp`, `level`, `message`, `target`, and
its fields at the top level, numbers as numbers, absent optionals left out.

So for two calibrations, "compare" is already built. What does not exist is anything that reads a training run,
anything that puts a training run and the calibrations of the model it produced together, and any chart.

## What the suggestion got wrong or left open

1. **The curves it wants to draw are logs, not results.** Loss, gradient norm, and learning rate are `training`
   lines on stderr, under `RUST_LOG`. `RUST_LOG=warn` (which `just calibrate` already uses for its own command)
   gives a successful run with no curve, and nothing today keeps stderr. Feature 2 is what makes the curves exist
   at all: its `events.ndjson` keeps logs at `info` whatever `RUST_LOG` says. A run made before feature 2, or
   given by hand as a bare events file, can still lack them, and the report must say so rather than draw an empty
   axis. They are also window means every `--log-every` steps, not per-step values.
2. **"Separation over steps" does not exist.** Separation is measured twice, before and after training. A curve
   needs feature 3 (evaluation at every checkpoint). This plan draws one only if `separation` lines with a step
   are present, and otherwise shows the before/after table.
3. **A `similarity-runs` binary cannot reach the calibration reader or the diff** without depending on
   `similarity-server`, which brings tonic and the Qdrant client into a reporting tool. The commands belong in
   the `record-similarity` binary, beside `calibrate-diff`. The training half reads plain JSON lines and needs
   nothing from `similarity-train`, which keeps D13 intact (the server must never depend on the trainer).
4. **`runs compare` for two calibrations would duplicate `calibrate-diff`.** It delegates to `offline::diff` and
   adds only a header saying which model and which training run each side came from.
5. **"A side-by-side diff of options and lineage"** has a source now that it did not have when suggested:
   `provenance.json` (feature 1) records options, data hashes, base, commit, and resumes. The trainer's events do
   not carry the options at all.
6. **Nothing links a calibration to the training run that made its model**, other than the fingerprint prefix
   matching the `digest` in the model directory's `provenance.json`. That breaks once the directory is deleted
   (the README tells people to delete checkpoint directories). Adding `digest` to the `finished` event fixes it.
7. **Resumed runs** were not considered. Under feature 2 a resume is a run directory of its own that names the
   one it continues (`continues: {run, from, step}`). A resumed process restarts `elapsed_s` at zero, writes a
   second `before` that describes the checkpoint, and repeats the steps between its checkpoint and the point the
   earlier process died. The report follows the chain back, keeps the earlier run's lines below `continues.step`
   and the later run's from there on, and draws by `step`, never by time.
8. **HTML is a new kind of output here.** D15 says reports for people are text on stdout. A file written to
   `--out` is neither a report on stdout nor a log, and needs a decision record (D22 below).
9. **`throughput`** is an example in `similarity-core` (`just throughput`), not a command, and has no events.
   Reporting on it is out of scope.
10. The **W&B exporter** the suggestion would put "here too" is a different feature and stays out.

## What is assumed from feature 2

Stated as an interface, not designed here. It was checked against `docs/plans/02-run-registry.md` ("What features
4 and 5 may rely on"), whose author corrected this plan's first assumptions. If feature 2 lands differently, only
`runs::load` changes.

- A run is a directory under `runs/`. `similarity_runs::open(dir) -> Listed { dir, manifest, status }` reads one;
  `list(root)` lists them. There is no `Manifest::read`.
- `Listed.status` is `running`, `finished`, `failed`, or `killed`. The stored status is one of the first three;
  `killed` is concluded by the reader from a lock file.
- `run.json` (`Manifest`) fields used here: the kind (`KIND_*` constants, train and calibrate), start and end
  times, `argv`, `build.tool`, `build.version`, `build.commit`, `build.dirty`, and for a training run a top-level
  `options` (the same JSON as `provenance.json`'s `trained.options`).
- **A resume does not append.** Each process gets its own directory, and a resumed one carries
  `continues: {run, from, step}`. The earlier run's events stop mattering at `continues.step`.
- `events.ndjson` holds every result and failure in emission order, and every log at `info` and above whatever
  `RUST_LOG` says, each line as the streams format it. A calibrate run's file holds the calibration events in both
  text and `--ndjson` mode, and `offline::events::read` reads it as it is.
- A model directory's `provenance.json` gains `run`, naming the run that wrote it; that run's `outputs` names the
  directory back.
- `similarity-runs` depends on `similarity-core` (an owner decision in plan 2). That costs this feature nothing:
  `similarity-server` depends on `similarity-core` already.

This plan does not create, list, or write run directories, and does not add a file sink.

## Design

### Where it lives

A new module `crates/similarity-server/src/runs/`, public from the crate like `offline`:

| File | What |
|---|---|
| `runs/mod.rs` | `//!` header; `Loaded` (a run read from disk) and `load(dir)`. |
| `runs/training.rs` | Reading a training run's events into `TrainingRun`; every number derived from one. |
| `runs/compare.rs` | `compare` and its text rendering. |
| `runs/html.rs` | The HTML document: layout, tables, escaping. |
| `runs/svg.rs` | The three chart kinds as SVG strings. |

and a `Runs` subcommand group in `crates/similarity-server/src/main.rs` (shared with feature 2's `runs list` if it
puts that there). The rule in `CLAUDE.md` extends: every calibration number in the HTML comes from
`offline::stats`; every training number comes from `runs::training`. The renderers compute nothing.

### Types

```rust
/// One `training` line, as read back.
pub struct StepLine {
  pub step: usize,
  pub of: usize,
  pub epoch: usize,
  pub pair_loss: Option<f64>,
  pub group_loss: Option<f64>,
  pub grad_norm: f64,
  pub rate: f64,
}

/// One `separation` result line.
pub struct SeparationLine {
  pub split: String,
  pub stage: String,
  /// Present only on lines written at a checkpoint (feature 3); `None` for `before` and `after`.
  pub step: Option<usize>,
  pub width: usize,
  pub edits: usize,
  pub edits_mean: Option<f64>,
  pub recycles: usize,
  pub recycles_mean: Option<f64>,
  pub gap_of_means: Option<f64>,
}

/// A training run read back from its events, stitched across the run directories of its resume chain.
pub struct TrainingRun {
  /// One line per step, in step order: each directory's lines below the step the next one continues from.
  pub steps: Vec<StepLine>,
  /// Steps at which a process started from a checkpoint (`continues.step` of each later directory).
  pub resumed_at: Vec<usize>,
  pub checkpoints: Vec<usize>,
  /// `before` lines are kept only from the first process (`compared_with: "base model"`).
  pub separation: Vec<SeparationLine>,
  pub finished: Option<Finished>,
  /// The `error` of a `failed` or `panicked` line, if the run ended that way.
  pub failure: Option<String>,
}

pub struct Finished {
  pub out: String,
  /// The content digest of the saved model. Absent in events written before step 1 of the work.
  pub digest: Option<String>,
  pub steps: usize,
  pub pair_examples: usize,
  pub catalog_examples: usize,
  pub held_out_examples: usize,
}

/// A run directory, loaded. `chain` is the training run's directories, oldest first, ending with the one named.
pub enum Loaded {
  Training { chain: Vec<Listed>, run: TrainingRun, lineage: Option<Provenance> },
  Calibration { listed: Listed, run: offline::events::Run },
}
```

Naming any directory of a resume chain loads the whole chain up to it; a directory is followed back through
`continues.run` until one has no `continues`. A link to a directory that is gone is an error naming it.

`training::read(ndjson: &str) -> anyhow::Result<TrainingRun>` follows `events::read`: lines are parsed as JSON
objects, a line that is not one is an error naming its number, and lines it does not know are skipped. It reads
by `message`, and for the log lines it does not check `target` (the trainer's target is its crate name, which is
not an interface). A `training` line without `step` is an error; one with an unknown extra field is fine (D17:
fields are added, never repurposed). It reads one file; `runs::load` reads each directory of a chain and stitches
them (`training::stitch(parts: Vec<(TrainingRun, Option<usize>)>)`, the `Option` being `continues.step`), so the
stitching is tested without a filesystem. Within one file, a step written twice keeps the later line.

The options come from the run's `run.json` (`options`), so they survive the model directory being deleted.
`lineage` (base, data hashes, data fingerprint) is `Provenance::read(finished.out)` when that directory is still
there, else `None`, and the report says which.

Derived numbers, all in `runs::training`: the last logged `pair_loss` and `group_loss`, their minimum and the step
it was reached at, the final `rate`, and the separation table keyed by `(split, stage, width)`.

### `runs compare A B`

The kinds decide what is compared; mixing a training run with a calibration is refused with a message that says
which is which.

- **Two calibrations:** read both, apply the same `same_pairs` check and `--intersect` flag as `calibrate-diff`,
  and print `render_diff` under a header with each side's run id, status, and fingerprint. Nothing else is
  computed. `--fail-on` is not offered here: the gate is
  `calibrate-diff`, which CI and feature 4 already point at, and one gate is enough.
- **Two training runs:** one table, left and right, then the rows that differ marked with `*`:

  ```text
  a      runs/20261111T0316-ft-lr2e5   finished   commit 545ef20
  b      runs/20261112T0910-ft-lr1e5   finished   commit 545ef20 (dirty)

                                      a          b
  base digest              f46e48ee0ea2dd02 f46e48ee0ea2dd02
  data fingerprint         3a91c0d2e4b7...  3a91c0d2e4b7...
  pairs.full.jsonl         sha 9c1e..., 47  sha 9c1e..., 47
  learning_rate          *       0.00002    0.00001
  epochs                               2          2
  ...
  steps                               10         10
  last pair_loss                  0.0171     0.0204
  lowest pair_loss (step)    0.0162 (8)  0.0198 (10)

  separation, held_out, width 256    a before  a after  b before  b after
  edits mean                           0.9629   0.9617    0.9629   0.9622
  recycles mean                        0.6444   0.6108    0.6444   0.6301
  gap of means                         0.3185   0.3509    0.3185   0.3321
  ```

  The option rows are every key of `run.json`'s `options`, with lists (`widths`) printed joined by commas. When the
  lineage is missing for one side the rows say `-` and a line under the table says why. When the two `before`
  columns differ the table says so on a line of its own: the runs did not start from the same model and data,
  and their `after` columns are not a like-for-like comparison.

No `--ndjson` for the training comparison: its inputs are already NDJSON, and an option row cannot be written
without a preformatted value for a list. A calibration comparison with `--ndjson` is `calibrate-diff --ndjson`.

### `runs report A [B ...] --out report.html`

Up to six runs, any mix of kinds. Calibrations are grouped under the training run whose `finished.digest` (or,
failing that, `provenance.json` digest) is the prefix of their fingerprint, and listed on their own otherwise.
Sections, each present only when some run has data for it:

1. **Runs.** One row per run: id, kind, status, start, duration, commit, and for a failed run its error.
2. **Lineage and options.** The training comparison table above, for every training run, differences
   highlighted.
3. **Training curves.** Three small charts sharing a step axis: `pair_loss` and `group_loss`, `grad_norm`,
   `rate`. One series per run, told apart by color and by dash pattern (so it reads in grayscale and for
   color-blind readers), resume points as vertical ticks, checkpoints as dots on the axis. A run with no
   `training` lines (one made before feature 2 or kept by hand under `RUST_LOG=warn`) is named under the chart:
   "no progress lines were kept".
4. **Separation.** The before/after table per split and width. A line over steps only when checkpoint
   separation lines exist (feature 3).
5. **Calibrations.** Per calibration: a histogram of overall similarity, edits and recycles overlaid, `unsure`
   pairs as a third outline; the configured threshold and the suggested one as vertical lines, labeled with their
   values; the error rates with intervals and the operating points as a table, from `offline::stats`. When two or
   more calibrations are in the report, the histograms share one x range so they can be read against each other.
6. **Per-field ranges.** For each calibration, one horizontal bar per field and label from
   `stats::field_range`, edits above recycles, threshold-free (a field range has no line to draw).

The histogram counts come from a new function in `offline::stats`, so that a text renderer could show the same
numbers:

```rust
/// How many pairs of `label` fall in each of `bins` equal bins between `low` and `high` (`None`: unlabeled).
/// A score at `high` falls in the last bin. Scores outside the range are counted in `below` and `above`.
pub fn histogram(calibration: &Calibration, label: Option<Label>, low: f64, high: f64, bins: usize) -> Histogram
```

Bin edges are display, not a measurement: 40 bins over the combined range of every calibration in the report,
rounded out to the nearest 0.01. Nothing on the page suggests a threshold that `stats` did not.

**The HTML.** One file, written whole through a sibling and a rename like a model directory (no half-written
report after a crash), refused if it exists unless `--force`. Inline CSS only; no `<script>`, no external URL of
any kind, no web fonts (system font stack). Light and dark through `prefers-color-scheme`. Charts are inline SVG
with a `<title>` per series and a table of the plotted numbers in a `<details>` beneath each, so nothing is only
visible as a shape. Every string from a file (paths, run ids, owner and category names in slices, errors) goes
through one `escape` function covering `& < > " '`. **No per-pair text** goes in the page: no notes, no pair ids,
no record contents, only aggregates and slice names. Per-pair detail stays in `calibrate-diff`, at a terminal.
The output is deterministic for given inputs (no generation timestamp, fixed number formatting), so two reports
can themselves be diffed and a test can pin the structure.

A curve is drawn from every kept line. At `--log-every 10`, a 100,000-step run is 10,000 points per series,
a few hundred kilobytes of SVG; above 2,000 points a series is thinned to every n-th point plus the minimum and
the last, and the page says it was. The thinning lives in `runs::training`.

**No new dependency.** The SVG is string formatting into a `String` with `write!`. `plotters` with only its SVG
backend was considered: it is MIT and `cargo deny` would accept it, but it brings font and text-layout crates for
three chart kinds that are each a polyline or a set of rectangles on two linear axes, and its output is harder to
make deterministic and accessible than hand-written SVG.

### Changes to existing events

Additive only, as D17 requires:

- `finished` gains `digest`: the content digest of the model just saved (`checkpoint::save_model` already has
  what `ModelFiles::digest` needs; the lineage written into the directory carries it). This is what links a
  calibration to its training run after the model directory is gone.
- Nothing else. The `calibration` event already names the fingerprint; the `training` lines are left as logs
  (see decision 3).

### Commands

```sh
record-similarity runs compare runs/A runs/B [--intersect]
record-similarity runs report runs/A runs/B runs/C --out report.html [--force]
just runs-compare runs/A runs/B
just runs-report runs/A runs/B --out report.html
```

## The work, in order

Each step is one commit, `just ci` green.

1. **`finished` carries the model's digest.** `similarity-train/src/main.rs`: add `digest` to `Outcome` and the
   `finished` event; extend the existing test that asserts the `finished` line. README table row for `finished`.
2. **Read a training run.** `runs/mod.rs` and `runs/training.rs` with `training::read` and the derived numbers.
   Commit a fixture: the events of a real `just finetune` on the synthetic sample
   (`data/calibration.sample.full.jsonl`, two epochs, `--checkpoint-every 3`, interrupted once and resumed), as
   two files under `crates/similarity-server/tests/fixtures/train-resumed/` (the run before the interruption and
   the run that continues it), with their `run.json` once feature 2 writes one. Synthetic data only; none enters the
   repository. In `similarity-train`, a test that pins the names and fields the reader depends on, so renaming one
   fails in the crate that renamed it.
3. **Load a run directory.** `runs::load` on `similarity_runs::open`, following `continues.run`. If feature 2 is not
   merged yet, this step waits; steps 1, 2, 4, and 6 do not depend on it.
4. **`offline::stats::histogram`**, with its tests.
5. **`runs compare`**, both kinds, with the text renderer and the `Runs` subcommand group.
6. **`runs/svg.rs`**: axis, polyline, histogram bars, range bars, as pure functions of numbers to strings.
7. **`runs report`**: `runs/html.rs`, escaping, the atomic write, thinning. Run it on the fixture and a
   calibration of the sample, open the page in a browser, light and dark, and check it at phone width.
8. **Documentation**: README, `CLAUDE.md`, D22, justfile recipes, the feature list marks 5 as built.

Steps 6 and 7 are the HTML half. If the owner decides against HTML (decision 1), stop after step 5 and step 8.

## Tests

None needs the model, the network, or Qdrant: the report reads files. All run under `cargo test`.

`similarity-train` (`main.rs` tests):

- `the_finished_event_names_the_digest_of_the_model_it_saved`
- `the_events_a_run_report_reads_keep_their_names_and_fields`

`similarity-server`, `runs::training`:

- `a_training_run_is_read_back_from_both_streams_in_one_file`
- `a_resumed_run_keeps_the_earlier_runs_steps_only_below_the_step_it_continues_from`
- `a_resume_chain_is_followed_back_from_any_of_its_directories`
- `a_chain_whose_earlier_directory_is_gone_is_an_error_naming_it`
- `a_step_written_twice_in_one_file_keeps_the_later_line`
- `only_the_first_process_before_line_describes_the_base_model`
- `a_run_with_no_progress_lines_reads_and_says_it_has_no_curve`
- `a_failed_run_reads_with_its_error_and_the_steps_it_took`
- `a_line_that_is_not_json_is_an_error_naming_its_line_number`
- `an_unknown_field_on_a_known_event_is_ignored`
- `a_long_series_is_thinned_and_keeps_its_minimum_and_last_point`

`offline::stats`:

- `a_histogram_counts_every_pair_of_its_label_exactly_once`
- `a_score_at_the_top_of_the_range_falls_in_the_last_bin`
- `scores_outside_the_range_are_counted_below_and_above_not_dropped`

`runs::compare`:

- `two_calibrations_are_compared_by_the_same_diff_as_calibrate_diff`
- `two_calibrations_of_different_pairs_are_refused_without_intersect`
- `a_training_run_and_a_calibration_are_not_compared`
- `two_training_runs_mark_every_option_that_differs`
- `training_runs_from_different_starting_points_are_flagged`
- `a_missing_model_directory_leaves_lineage_rows_empty_and_says_why`

`runs::html` and `runs::svg`:

- `a_report_contains_no_script_and_no_external_url`
- `every_string_from_a_file_is_escaped` (a run id and a slice value of `<script>&"'`)
- `a_report_contains_no_pair_note_or_pair_id`
- `the_same_runs_give_byte_identical_reports`
- `a_calibration_is_grouped_under_the_training_run_that_made_its_model`
- `an_existing_report_is_not_overwritten_without_force`
- `every_chart_has_a_table_of_the_numbers_it_plots`
- `a_histogram_bar_is_as_tall_as_its_count_says`

A property, in `similarity-server`'s `properties` module (it must hold for every input): for any calibration and
any bin count, the histogram's bins plus `below` and `above` sum to the pairs of that label. Run
`PROPTEST_CASES=5000 just test` once after adding it.

`just test-real` is not needed: no encoder, loader, tokenizer, trainer step, or config value changes. Step 1
touches the trainer's output only, not a training step, so `just test-parity` is not needed either; run it anyway
if `Outcome` is reorganized beyond adding the field.

**What only the owner's work environment can show:** whether the charts are useful on a real run (thousands of
pairs, many owners, long curves, real `unsure` mass). Everything here can be built and tested on the synthetic
sample, which has 48 pairs and makes a sparse histogram. The thinning threshold and the bin count should be
looked at once on a real run there; they are display choices and can be changed without touching anything else.

## Documentation to update

- **README.md**: a section "Reports across runs" after "Comparing two calibrations", with a `runs compare`
  example on the sample (run, not invented), what the HTML contains and what it deliberately leaves out, and a
  pointer from "Fine-tuning, 3. Measure whether it helped". The `finished` row of the trainer's event table gains
  `digest`. "Project layout" gains `runs/`.
- **CLAUDE.md**: the commands list gains `just runs-compare` and `just runs-report`. Under "Things that are easy to
  get wrong": the trainer's `training`, `checkpoint`, `separation`, and `finished` names and fields are now an
  interface too (add fields, never rename); every number in a run report comes from `offline::stats` or
  `runs::training`; the HTML report has no script, no external URL, no per-pair text, and escapes every string.
- **docs/decisions.md**: D22, "A run report is a file, not a service": why HTML at all (charts are what a table
  cannot show), why no JavaScript or dependency, why no per-pair text, why it lives in the server binary and not
  in its own (the calibration reader and diff are there; D13 forbids the other direction), why the curves remain
  logs (D15), and the link through the model digest. D15 gains one sentence pointing to D22 for the one report
  that is a file.
- **docs/feature-suggestions.md**: mark 5 built, with what came out differently. **docs/plan-features.md**: the
  row for 5.

## Decisions for the owner

1. **HTML at all, or text only?** A text `runs compare` covers the comparison; the HTML adds curves and
   histograms, which are the reason people open W&B. **Recommended: build it**, as the last two commits, with no
   dependency and no JavaScript, so it can be dropped without touching anything else.
2. **Where the commands live.** **Recommended: `record-similarity runs ...`**, beside `calibrate-diff`, because the
   calibration reader and diff are in that crate. The alternative, a `record-similarity-runs` binary, needs either
   a dependency on `similarity-server` or moving `offline` into a shared crate, which is a large refactor for a
   name.
3. **Should `training` progress become result events?** It would make curves survive `RUST_LOG=warn` for a run
   kept by hand outside the registry. **Recommended: no.** D15 calls progress optional by nature, and feature 2
   keeps logs at `info` in `events.ndjson` whatever `RUST_LOG` says, so a registered run always has them.
4. **Per-pair text in the HTML.** **Recommended: none**, aggregates only. A report file is easy to attach to a
   ticket or a chat, and pair notes and ids from real data should not travel that way. Per-pair detail stays in
   `calibrate-diff`.
5. **`finished` gains `digest`.** **Recommended: yes.** Without it a calibration is linked to its training run only
   through the model directory's `provenance.json` (feature 2 adds `run` there), which is gone once someone deletes
   a checkpoint directory, as the README tells them to.
6. **How many runs in one report.** **Recommended: up to six**, the number of series that stay distinguishable by
   color and dash pattern together. More is an error naming the limit.

## Size

**Medium.** Reading training events and the text compare are small (a couple of days, mostly tests and the
resume edge cases). The HTML half is the larger part: three chart kinds, escaping, accessibility, and a layout
that holds in both color schemes are tedious rather than hard. It would be large only if feature 3's checkpoint
separation or a W&B exporter were pulled in, and both are kept out.

## Out of scope

- Creating, listing, or pruning run directories, and the file sink (feature 2).
- Separation at every checkpoint (feature 3). The report draws it when it appears.
- A promotion gate (feature 4) and any `--fail-on` here; `calibrate-diff --fail-on` is the gate.
- Exporting to W&B or any other tracker.
- `throughput` runs, and a terminal UI of any kind (D15).
