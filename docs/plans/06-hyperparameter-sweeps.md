# Plan: hyperparameter sweeps (feature 6)

A plan for trying many settings at once and saying which did best, written before any of it was built so that a
session with no memory of the planning can carry it out. Read `CLAUDE.md` first, then this file. `README.md`
("Fine-tuning the model", "The calibration as data", "Comparing two calibrations") and `docs/decisions.md` (D3,
D12, D13, D15, D17, D18) explain what it refers to. Delete this file, or fold what is still true into
`docs/decisions.md`, when the work is done.

## Contents

- [What it is for](#what-it-is-for)
- [Two kinds of sweep](#two-kinds-of-sweep)
- [What exists to build on](#what-exists-to-build-on)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [What can only be validated at work](#what-can-only-be-validated-at-work)
- [Dependencies on unbuilt features](#dependencies-on-unbuilt-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

The README's only tuning advice is "lower the learning rate if the result gets worse on the holdout". Once real
labeled pairs exist, two questions need more than that:

1. **Which training options produce a better model?** Learning rate, passes, batch size, widths, dropout. Each
   answer costs a training run, and at a few thousand examples the spread between seeds can be larger than the
   effect being measured, so every setting has to be run under several seeds and judged on pairs it never saw.
2. **Which weights, floors, and quantity limits suit a given model?** These cost nothing to try: a calibration
   already wrote down every per-field similarity and per-quantity factor, and judging them again is arithmetic
   (`Evaluation::rejudge`, D17).

A sweep answers either question for a grid of settings in one command and ends with a ranking. It does not
change any configuration. Its output is evidence for a person, who then runs `calibrate` and `calibrate-diff` as
today before a threshold or a weight changes (D3).

## Two kinds of sweep

| | Policy sweep | Training sweep |
|---|---|---|
| Sweeps | field weights, `reject_below`, quantity `reject_above_factor`, `reject_incomparable`, `reject_on_measure_change` | trainer options: `--learning-rate`, `--epochs`, `--batch-size`, `--dims`, `--no-dropout`, `--weight-decay`, `--max-grad-norm`, ... |
| Cost per point | microseconds (re-judging recorded measurements) | one training run and one calibration (minutes to hours) |
| Needs a model | no | yes, and a GPU in practice |
| Command | `record-similarity calibrate-sweep` | `record-similarity-train sweep` |
| Crate | `similarity-server` (`offline::sweep`) | `similarity-train` (`sweep` module) |
| Output | a text report, or NDJSON events with `--ndjson` | NDJSON result events (D15: the trainer prints no text) |

They compose: the training sweep ranks models by the best overall threshold each can reach, ignoring floors (see
"objective" below), and the winning model's calibration is the input to a policy sweep that then fits its floors
and weights.

The overall `threshold` is never an axis of either. `suggest_threshold` already tries every cut between
neighboring scores, and `stats::operating_points` already reads the threshold for each false-accept target.
Sweeping it on a grid could only find a worse version of either.

## What exists to build on

Verified against the tree at `545ef20`.

**Training (`crates/similarity-train`).**

- `src/main.rs`, `struct Cli`: a flat `clap` parser with no subcommands and no positional arguments. Every option a
  sweep would vary is a flag there: `--epochs`, `--batch-size`, `--learning-rate`, `--dims`, `--no-dropout`,
  `--weight-decay`, `--max-grad-norm`, `--seed`, plus `--checkpoint-dir`, `--keep-checkpoints`, `--resume`.
  `TrainOptions` (`src/trainer.rs`) also has `warmup_ratio`, `margin`, and `scale`, which have **no flag**
  (`..TrainOptions::default()` in `Cli::options`), so they cannot be swept today.
- `checkpoint::RunIdentity` (`src/checkpoint.rs`): options, data fingerprint, and `max_tokens`. A resume under any
  other value is refused. A sweep that resumes an interrupted point with the same argument list is therefore
  checked by the code that exists.
- `Checkpoints::prune` (`src/main.rs`) deletes only the run's own checkpoints, and `--keep-checkpoints` bounds them
  across resumes. A checkpoint is a whole model directory (265 MB for the pinned model).
- `checkpoint::save_model` and `write_whole`: `--out` is either absent or complete, never half written, so "does
  the point's model directory exist" is a reliable test of whether training finished.
- `similarity_core::rng::Rng` (`SplitMix64`, moved to core for D17): `--seed` seeds the batch order and dropout.
- `eval::separation` reports `separation` events per split, stage, and width: single field texts, no weights,
  floors, or thresholds. Its own doc comment says it is a progress check and not the measurement that decides
  whether a model is better.
- `data::pair_examples` reads `label`, `split`, `before`, and `after` and ignores every other key. It does **not
  read `owner`**, and `GroupLine` does not either.

**Calibration (`crates/similarity-server/src/offline`).**

- `calibrate::{calibrate, summarize, suggest_threshold}` and `stats::{error_rates, operating_points, ...}`: every
  number a ranking needs, with owner-resampled intervals.
- `events::{emit, read}`: `calibrate --ndjson` writes, and `read` reads back into a `Run` (`RunInfo` with the
  fingerprint, fields, and quantities, plus a `Calibration`).
- `diff::rejudge(run, schema, policy)`: re-judges every pair of a run under another schema and policy and
  re-summarizes, refusing a different set of fields or quantities. This is the whole inner loop of a policy sweep.
- `calibrate-diff --config-b` (`main.rs`, `CalibrateDiffArgs`): the one-point version of a policy sweep, already
  documented in the README.
- `Config` (`src/config.rs`): `fields: BTreeMap<String, FieldConfig>` and `quantities: BTreeMap<String,
  QuantityConfig>` are public, so a grid point is a cloned `Config` with values replaced, then `config.schema()`
  and `config.policy()`, which validate it.
- `CalibrateArgs` has `--config`, `--full`, `--split`, `--min-confidence`, `--ndjson`, `--seed`. There is no way to
  point it at a model other than the config's `model.dir`.

**Scoring (`crates/similarity-core/src/score.rs`).** `Comparison::weigh`, `score::judge`, `Evaluation::rejudge`,
`Policy`, stable `Reason::code`.

## What the suggestion got wrong or left open

1. **"The scoring half of `offline.rs` uses only core types and could move into `similarity-core`"**, and
   `docs/plan-features.md`'s "the scoring it needs is in `similarity-core` already". Half true. The primitives
   (`evaluate`, `weigh`, `judge`, `rejudge`) are in core. Everything a full-policy calibration also needs is in the
   server crate: the lenient `pairs.full.jsonl` reader and pair ids (`offline::pairs`), `Offline::compare`, the
   threshold search, owner-resampled intervals and operating points (`offline::stats`), the event writer and
   reader, and `Config`, which turns a TOML file into a `Schema` and `Policy`. That is several thousand lines
   written with `anyhow`; moving them into a library would also mean rewriting them on `thiserror` (CLAUDE.md).
   This plan does not move them. **The training sweep runs `record-similarity calibrate` as a child process**,
   which needs no refactor (see "Why child processes").
2. **"The trainer must not link the server (D13)."** D13 says the reverse: the server must never depend on the
   trainer. The trainer could link the server's library. It should not, for a different reason: that would pull
   `tonic`, the Qdrant client, and `ureq` into the training binary to reach a module that does file I/O.
3. **"Scored by full-policy calibration."** A new model invalidates the configured thresholds and floors (D3:
   thresholds do not transfer between models). Judging a candidate model by how many pairs the *configured*
   policy gets wrong measures how close it happens to land to the old model's numbers. The objective has to be
   what each model can reach at its own best threshold: `suggest_threshold`'s errors, which use the overall score
   only. Floors and weights are then fitted to the winner by a policy sweep.
4. **"On the validation fold."** There is no validation fold. The data request (`docs/data-request-prompt.md`)
   assigns `calibrate` (80%) and `holdout` (20%) by brand hash, and the trainer trains on `calibrate`. Selecting on
   `calibrate` scores models on their own training data; selecting on `holdout` turns the holdout into a
   validation set (the feature 3 caveat). A `validation` split carved out of `calibrate` by owner hash is needed
   first, honored by both the trainer and `calibrate`. Feature 3 plans exactly that
   (`docs/plans/03-checkpoint-evaluation.md`); this plan uses its interface, and builds that part of it as step 1
   if feature 3 has not been built.
5. **Epochs cannot be swept by picking checkpoints.** It is tempting to train the longest point once and calibrate
   each end-of-pass checkpoint. That is not the model `--epochs k` would give: the warm-up and linear decay span
   the total number of steps, so the rate at step `s` depends on `epochs`. Epochs are a training axis like any
   other. (Feature 3's per-checkpoint evaluation is still useful inside one run; it does not replace this.)
6. **Weights are scale-free.** The weighted score is `sum(w * s) / sum(w)`, so `name = 4, category = 3` and
   `name = 8, category = 6` are the same point. A weight grid has to collapse proportional points or it reports
   ties that are one setting counted twice. Floors and quantity limits are absolute and do not collapse.
7. **`anchor_threshold` cannot be swept on calibration pairs.** A pair has no history, so `Offline::compare`
   passes no anchor and `anchor_similarity` is always `None`. A grid naming it is refused. Feature 11 (drift
   calibration from histories) is where it can be measured.
8. **Margin, scale, and warm-up have no flag.** The contrastive margin and the ranking-loss scale are arguably the
   loss's most important settings and cannot be swept until `--margin`, `--scale`, and `--warmup-ratio` exist.
   They are already in `TrainOptions`, hence in `RunIdentity` and in `provenance.json`'s options, so adding flags
   is safe. Optional step 6; an owner decision.
9. **Left open by the suggestion:** what number ranks the points, how seeds are combined, what happens when one
   point diverges (an expected outcome at a high learning rate), how an interrupted sweep continues, what is
   deleted, and how the chosen point reaches the holdout. Each is decided below.
10. **"Keeps one checkpoint by default."** After a point finishes, its final model is in `--out`, written whole;
    its checkpoints were only ever for resuming that point. The sweep deletes the point's checkpoint directory once
    the point's calibration is written, which keeps less than the suggestion and loses nothing.

## Design

### The validation split, as feature 3 defines it (step 1, unless feature 3 has built it)

This plan uses feature 3's interface unchanged (`docs/plans/03-checkpoint-evaluation.md`, agreed with its
planner). The validation split lives **in the pairs file**, not in memory:

- **`record-similarity pairs carve pairs.full.jsonl --from calibrate --into validation --fraction 0.15 --salt S`**
  rewrites `split` for whole owners: an owner moves when the first eight bytes of SHA-256(salt, `0x00`, owner), as
  a big-endian `u64`, fall below `fraction * 2^64`. A pair without an owner stays where it is and is counted.
  The carved file is what both the trainer and `calibrate` read, so there is one copy of the hash.
- **Trainer:** `--validation-split validation` (on when that split has pairs, `--no-validation` turns it off) and
  `--validation-min-confidence`. Pairs of that split are read with `Split::HeldOut` and never trained on. Catalog
  groups whose owner has validation or holdout pairs are dropped. Feature 3 also measures every checkpoint on that
  split and adds `best_step`, `best_model`, and related fields to `finished`; the sweep does not use them
  (correction 5).
- **`calibrate`:** selects the same pairs with the existing `--full --split validation`. No new selection flag.
- **`calibrate --model <dir>`** replaces `model.dir` for that run; everything else (`output_dim`, `max_tokens`,
  `device`, schema, policy) comes from `--config`. The `calibration` event gains `model_dir` and `model_step`.

If feature 3 is not built when this starts, step 1 builds `pairs carve`, the trainer's `--validation-split` and
`--no-validation`, and `calibrate --model`, as feature 3's plan specifies, and leaves the per-checkpoint
measurement to feature 3.

**k folds** would replace `pairs carve` and the `validation` split value with an in-memory fold
(`similarity_core::fold::Fold { index, count }`, `--validation-fold 1/5` in the trainer, `calibrate --fold 1/5`).
That is the alternative under feature 3's decision 1 and decision 1 below. A sweep would then average each run
over folds as it averages over seeds. Nothing else here changes.

**One addition to `calibrate --model`** (step 2, small): since a sweep points `calibrate` at models it trained,
if the model directory has a `provenance.json` whose `trained.max_tokens` differs from the config's, or whose
`options.widths` lacks `output_dim`, `calibrate` logs a warning. It does not refuse (D18: nothing depends on the
note). The sweep repeats the second check as an error before training anything (see below).

### Policy sweep: `record-similarity calibrate-sweep` (steps 3 and 4)

```sh
record-similarity calibrate-sweep select.ndjson --config config.toml --grid grid.toml \
  [--evaluate validation.ndjson] [--top 10] [--max-points 100000] [--ndjson] [--seed 0]
```

- `select.ndjson`: a `calibrate --full --ndjson` run. Points are ranked on it.
- `--config`: the base configuration. Every setting the grid does not name keeps its value.
- `--evaluate`: optional second run of the **same model** (fingerprint equal, refused otherwise) on other pairs.
  The top points are judged on it at the threshold **suggested on the selection run**, carried over, never
  re-suggested: that is the out-of-sample number, and the only one in the report that is not optimistic.
- No model is loaded, as for `calibrate-diff --config-b`.

The grid file (`grid.toml`), parsed with `deny_unknown_fields`:

```toml
# Every list is an axis; the points are every combination of them.
[fields.name]
weight = [2.0, 4.0, 6.0]
reject_below = ["off", 0.75, 0.80, 0.85]

[fields.category]
weight = [1.0, 3.0]

[quantities.weight_per_item]
reject_above_factor = ["off", 1.5, 2.0]
reject_incomparable = [false, true]

[policy]
reject_on_measure_change = [false, true]
```

- Field and quantity names must exist in the base config; an unknown one is an error naming it (a typo here would
  otherwise be an axis that changes nothing).
- `"off"` is the only string allowed in a numeric list and means "no floor" or "no limit" (TOML has no null).
- `[policy] threshold` and `anchor_threshold` are refused with the reason (the first is searched exactly, the
  second needs histories).
- Points whose weights are proportional (equal after dividing by their sum) and whose other settings are equal are
  collapsed into the first one generated. The report says how many were collapsed.
- More points than `--max-points` is an error before any work, with the count.

`crates/similarity-server/src/offline/sweep.rs`:

```rust
/// One axis value: a number, or `"off"`.
pub enum Setting { Off, Value(f64) }

/// What a grid file says, before it is expanded.
pub struct Grid {
  fields: BTreeMap<String, FieldAxes>,
  quantities: BTreeMap<String, QuantityAxes>,
  policy: PolicyAxes,
}

/// One combination: the settings it changes, by name.
pub struct Point {
  pub id: String,
  pub fields: Vec<FieldSetting>,
  pub quantities: Vec<QuantitySetting>,
  pub measure_change: Option<bool>,
}

/// A point judged on the selection run, and on the evaluation run if there is one.
pub struct Judged {
  pub point: Point,
  pub suggestion: Option<Suggestion>,      // from calibrate::summarize, on the rejudged selection run
  pub configured: (usize, usize),          // false rejects and accepts under the point's floors and limits at
                                           // the suggested threshold (floors and limits do reject on their own)
  pub evaluated: Option<Evaluated>,        // errors and rates on --evaluate at the carried-over threshold
}

pub fn parse_grid(text: &str, base: &Config) -> anyhow::Result<Grid>;
pub fn points(grid: &Grid, max: usize) -> anyhow::Result<(Vec<Point>, usize /* collapsed */)>;
pub fn judge(run: &Run, base: &Config, point: &Point) -> anyhow::Result<Judged>;
pub fn sweep(select: &Run, evaluate: Option<&Run>, base: &Config, grid: &Grid, top: usize, seed: u64)
  -> anyhow::Result<PolicySweep>;
```

`judge` builds the point's `Config` (clone, replace), calls `schema()` and `policy()`, sets the policy's
threshold to the point's own suggestion, and calls `diff::rejudge`. A point that fails validation (a weight of 0
after an axis) is reported as invalid, not fatal.

**Ranking.** By total errors under the point's full policy at its own suggested threshold (floors and limits
included, since fitting them is the purpose), then by the suggestion's margin (wider first), then by generation
order, so the ranking is deterministic. Owner-resampled intervals (`stats::error_rates`, 1,000 resamples) are
computed only for the `--top` points, since they cost more than everything else together.

**Statistics.** One new function in `offline::stats`, used by both renderers (CLAUDE.md: every number comes from
there): `errors_at(calibration, threshold) -> (usize, usize)`, the false rejects and accepts at a given overall
threshold with the calibration's floors and limits. The rest already exists.

**Report** (`offline::report::render_policy_sweep`, text, locked stdout): the base config's own result first, as a
reference row; then the top points, one line each with what they change against the base, the suggested
threshold, errors, the interval, and the evaluation errors if any; then the winner as a TOML fragment headed
"to try, not to paste: run `calibrate` under it and compare with `calibrate-diff`".

**Events** (`--ndjson`, through `similarity-log`, `target: RESULT`, flat, D17):

| `message` | One per | Fields |
|---|---|---|
| `policy_sweep` | sweep, first | `fingerprint`, `pairs`, `evaluate_pairs`, `points`, `collapsed`, `invalid`, `seed` |
| `policy_point` | ranked point | `point_id`, `rank`, `threshold`, `margin`, `false_rejects`, `false_accepts`, `false_reject_rate`, `false_reject_low`, `false_reject_high`, `false_accept_rate`, `false_accept_low`, `false_accept_high`, `evaluate_false_rejects`, `evaluate_false_accepts` |
| `policy_point_field` | point and changed field | `point_id`, `field`, `weight`, `reject_below` |
| `policy_point_quantity` | point and changed quantity | `point_id`, `quantity`, `reject_above_factor`, `reject_incomparable` |
| `policy_point_policy` | point that changes it | `point_id`, `reject_on_measure_change` |

An absent setting (a floor switched off) is left out, as everywhere else. Interval fields are present only for the
top points. `point_id` is a 12-hex SHA-256 prefix of the point's settings, so the same point has the same id in
every sweep.

Only `--top` points are written by default; `--all` writes every point (a grid of 10,000 points is 10,000 lines).

### Training sweep: `record-similarity-train sweep` (steps 5 to 8)

```sh
just sweep sweep.toml                 # CPU
just sweep sweep.toml metal           # builds both binaries with that backend, then runs
```

**The sweep file**, parsed with `deny_unknown_fields`:

```toml
[sweep]
dir = "sweeps/lr-and-epochs"     # everything the sweep writes; created, or continued if it exists
seeds = [0, 1, 2]                # every point runs once per seed; at least 1, three recommended
config = "config.toml"           # the server config `calibrate` scores with (schema, policy, output_dim, max_tokens)
pairs = "pairs.carved.jsonl"     # scored by `calibrate --full`; the same carved file as `--pairs` below
select_split = "validation"      # never trained on, used to rank (feature 3 carves it)
holdout_split = "holdout"        # scored once, for the winner only (see below)
server_bin = "target/release/record-similarity"   # optional; default: next to this binary

[fixed]                          # trainer flags shared by every run, by their long names
base = "models/mdbr-leaf-mt"
pairs = "pairs.carved.jsonl"
catalog_groups = "catalog-groups.jsonl"
device = "metal"
max_tokens = 256
dims = "1024,512,256"

[axes]                           # trainer flags to vary; points are every combination
learning_rate = [1e-5, 2e-5, 5e-5]
epochs = [1, 2, 3]
no_dropout = [false, true]       # a boolean flag: true passes it, false leaves it out
```

**Validating before anything runs.** Each run's argument list is assembled as the command line it will be
(`--learning-rate 2e-05 --epochs 2 ...`) and parsed with `Cli::try_parse_from`. A misspelled key, a bad value, or a
combination `clap` refuses fails the sweep before the first minute of training, naming the point. These keys
belong to the sweep and are refused in `[fixed]` and `[axes]`: `out`, `resume`, `seed` (it is `seeds`),
`checkpoint_dir`, `keep_checkpoints`, `no_checkpoints`, `validation_split`, `no_validation`. Before training, the
sweep also checks:

- that `pairs` in `[fixed]` and in `[sweep]` name the same file, and that its `select_split` holds labeled pairs
  of both kinds (a sweep ranked on nothing is refused; an uncarved file is the likely cause, and the error says
  to run `pairs carve`);
- that `--dims` (after `trainable_widths`, so after loading the base once) includes the config's `output_dim`;
- that `max_tokens` equals the config's.

The last two read only `[model] output_dim` and `max_tokens` from the config, with a small lenient `serde` struct.
The trainer does not parse the rest of the server's config.

**Why child processes.** Each training run is `record-similarity-train` itself (`std::env::current_exe()`) with
the assembled flags, and each calibration is `record-similarity calibrate`. That:

- needs no refactor of the calibration code into a library (correction 1);
- frees the GPU's memory between runs; Burn's backends keep allocation pools for the life of a process;
- isolates failure: a run that diverges (D13 stops it on a non-finite loss or gradient norm) or runs out of
  memory is one failed run, recorded, and the sweep goes on;
- makes every run exactly a command a person could type, and the sweep records it.

No new dependency: `std::process::Command`.

**Layout.** Everything under `[sweep] dir`:

```text
sweeps/lr-and-epochs/
  sweep.json                        the sweep file as parsed, its SHA-256, the data files' SHA-256, the build
  base/calibration.ndjson           the base model on the validation split: the row to beat
  points/<point_id>/point.json      the point's flags (without seed and paths)
  points/<point_id>/seed-<n>/
    run.json                        argv, status, start and end, exit code
    model/                          the trained model (the run's --out)
    checkpoints/                    while running only; removed once calibration.ndjson is complete
    train.ndjson                    the run's stdout (results)
    train.log.ndjson                the run's stderr (logs)
    calibration.ndjson              `calibrate --full --split <select_split> --ndjson`
  holdout/<point_id>-seed-<n>.ndjson   the winner only, if run
```

`point_id` is a 12-hex SHA-256 prefix of the point's flags in a canonical order (sorted by name, values as their
argv spelling), excluding seeds and paths. Content-addressed points are what make a sweep continuable and
extensible: running the same file again skips finished runs; adding a value to an axis adds points and reruns
nothing.

**Each run, in order:**

1. If `calibration.ndjson` exists and contains a `summary` line, the run is done: skip it.
2. If `model/` exists, training finished (`write_whole`): go to 4.
3. Train: `--out <run>/model --checkpoint-dir <run>/checkpoints --keep-checkpoints 1
   --validation-split <select_split> --seed <n>` plus the point's flags. If `checkpoints/` holds a
   `step-NNNNNN`, pass `--resume` with the newest in place of `--base`: the argument list is the same, so
   `RunIdentity` accepts it, and refuses it if anything did change. Stdout and stderr go to the
   two files (appended on a resume). Feature 3's per-checkpoint measurement runs too; its `best_model` is ignored.
4. Calibrate: `<server_bin> calibrate --config <config> --model <run>/model --full
   --split <select_split> --ndjson <pairs>`, stdout to `calibration.ndjson` written to a sibling and renamed,
   so a partial file never looks complete.
5. Remove `checkpoints/`. Write `run.json` with the status.

A child that exits non-zero marks the run `failed` with its exit code and the last `failed` line of its stderr
(`similarity_log::failed` writes one), and the sweep continues. A sweep in which every run failed exits non-zero.

**Objective** (owner decision 2). For each run: the `suggestion` event of its calibration, that is the fewest
errors any overall threshold reaches on the validation split, using the overall score alone. Floors and the
configured threshold are deliberately not part of it (correction 3). Read from the event stream with a small
`serde` struct for each of `calibration`, `suggestion`, `operating_point`, and `summary`, ignoring other fields.
The trainer does not link the server, so it reads these events as any other tool does; that is what D17's "names
are an interface" is for.

**Combining seeds.** Per point: the mean, lowest, and highest of the objective over its finished seeds, and how
many seeds failed. Points are ranked by the mean, then by the highest (the worst seed), then by `point_id`. A point
is marked `indistinguishable` from the winner when its range over seeds overlaps the winner's. That flag is not a
significance test and the README says so; it says only that the seeds disagree more than the points do, which is
the situation the suggestion warns about. No statistical threshold is invented.

**The holdout, once.** When every run is done, the winning point's seed models are calibrated on
`--split <holdout_split>`, and only those. These are the numbers to quote; every validation number is
optimistic by selection. This is the feature 3 caveat applied: the holdout is touched once, by the chosen point.
`holdout = false` in `[sweep]` skips it.

**Pruning models** (owner decision 3). `keep_models = "all" | "best" | <k>` in `[sweep]`; default `"all"`. With
`"best"` or `k`, after the leaderboard, the `model/` directories of runs outside the top `k` points are deleted;
their `calibration.ndjson` and `train.ndjson` are kept, which is everything the leaderboard needs. Only directories
under the sweep's own `points/` whose `run.json` names them are ever deleted.

**Events.** The sweep's own stdout, `target: RESULT`, flat:

| `message` | One per | Fields |
|---|---|---|
| `sweep` | sweep, first | `dir`, `sweep_sha256`, `points`, `seeds`, `runs`, `runs_done` (finished in an earlier invocation), `select_split` |
| `sweep_option` | point and option | `point_id`, `option`, `number` or `text` (`number` when the argv value parses as a finite `f64`, else `text`; a boolean flag is `text: "true"`/`"false"`) |
| `sweep_run` | run, as it finishes | `point_id`, `seed`, `status` (`finished`, `failed`, `skipped`), `exit_code`, `steps`, `elapsed_s`, `threshold`, `false_rejects`, `false_accepts`, `margin`, `edits`, `recycles` |
| `sweep_point` | point, in rank order | `point_id`, `rank`, `seeds`, `failed`, `errors_mean`, `errors_low`, `errors_high`, `threshold_mean`, `indistinguishable` |
| `sweep_baseline` | sweep | the base model's `threshold`, `false_rejects`, `false_accepts`, `margin` on the validation split |
| `sweep_holdout` | winning run | `point_id`, `seed`, `threshold` (the run's validation suggestion), `false_rejects`, `false_accepts` at that carried-over threshold, plus the holdout's own `suggestion` numbers for reference |
| `sweep_finished` | sweep, last | `runs`, `failed`, `elapsed_s`, `winner` (point id), `beats_baseline` |

`errors_*` and `threshold_mean` are rounded to four places (D15). Progress (`starting run`, the child's argv) is
logged on stderr. A person reads the leaderboard with `jq`; the README gets the one-liner.

**Code.** The orchestration is a library module, so that it can be tested without processes:

```text
crates/similarity-train/src/sweep/
  mod.rs        //! header; `pub fn run(file: &SweepFile, runner: &mut dyn Runner) -> Result<Leaderboard>`
  file.rs       SweepFile, parsing and validation, canonical point ids, argv assembly
  runner.rs     `pub trait Runner { fn train(&mut self, argv: &[OsString], out: &RunFiles) -> Result<Exit>;
                fn calibrate(&mut self, argv: &[OsString], out: &Path) -> Result<Exit>; }` and `ProcessRunner`
  results.rs    the four calibration event structs, read leniently; `Exit` with the child's last failure line
  leaderboard.rs  combining seeds, ranking, `indistinguishable`, emitting the events
```

Errors are a `SweepError` (`thiserror`) added to `TrainError`. `main.rs` gains the subcommand with
`#[command(args_conflicts_with_subcommands = true)]` and `Option<Command>`, so `record-similarity-train --base ...`
keeps working exactly as today and `record-similarity-train sweep sweep.toml` is new. `argv` validation calls back
into `Cli::try_parse_from`, which lives in the binary; `sweep::run` therefore takes a `validate: &dyn
Fn(&[OsString]) -> Result<()>` from `main.rs`.

The trainer gains `toml.workspace = true`. `toml` is already in the workspace and the lock file (the server uses
it), so `cargo deny` sees no new crate or license.

## The work, in order

Each step is one commit and leaves `just ci` passing.

1. **The validation split** (skip what feature 3 has built): `pairs carve`, the trainer's `--validation-split`
   and `--no-validation`, `calibrate --model` with `model_dir` and `model_step`, all as
   `docs/plans/03-checkpoint-evaluation.md` specifies them, with that plan's tests. `just test-real` afterwards.
2. **The provenance warning in `calibrate --model`.** README note.
3. **Policy sweep core:** `offline::sweep` (grid parsing, expansion, collapsing, judging, ranking) and
   `stats::errors_at`. Tests only, no command yet.
4. **`calibrate-sweep` command:** text report, events, `--evaluate`, `just calibrate-sweep`. Run it on
   `data/calibration.sample.full.jsonl` and put that output in the README, labeled as the synthetic sample.
5. **Training sweep file and plan:** `sweep::file` (parsing, refused keys, canonical ids, argv assembly) and the
   subcommand skeleton that validates and prints the plan (`--dry-run`: points, runs, what is already done) without
   running anything.
6. *(Optional, owner decision 4)* `--margin`, `--scale`, `--warmup-ratio` flags on the trainer. Defaults unchanged,
   so `just test-parity` must still pass unchanged; run it.
7. **Running:** `Runner`, `ProcessRunner`, the per-run sequence, resume, failure isolation, checkpoint removal.
8. **Leaderboard, baseline, holdout, pruning, events**, and `just sweep`. Run a two-point, two-seed sweep on the
   synthetic sample with `max_tokens = 32` on the CPU end to end, interrupt it once mid-run and continue it, and
   put the commands (not the ranking, which is noise on 46 pairs) in the README.
9. **D22** in `docs/decisions.md`, CLAUDE.md, and the feature-suggestions status line.

## Tests

None needs the model, the network, or Qdrant unless marked.

The carve, the validation split, and `calibrate --model` are tested as `docs/plans/03-checkpoint-evaluation.md`
lists; step 1 brings those tests with it.

**`similarity-train`:**

- `a_sweep_file_naming_an_option_the_trainer_does_not_have_fails_before_any_run`
- `options_the_sweep_owns_are_refused_in_fixed_and_axes`
- `a_point_has_the_same_id_however_its_axes_are_ordered_and_whatever_its_seed`
- `adding_a_value_to_an_axis_adds_points_and_keeps_the_ids_of_the_old_ones`
- `the_widths_must_include_the_servers_output_dim_before_anything_trains`
- `a_sweep_whose_pairs_have_no_validation_split_is_refused_and_told_to_carve_them`
- `the_pairs_the_trainer_reads_and_the_pairs_calibrate_scores_must_be_one_file`
- with a fake `Runner` that records argv and writes canned outputs:
  - `every_point_runs_once_per_seed_with_the_validation_split_and_its_own_directories`
  - `a_finished_run_is_not_run_again_when_the_sweep_is_continued`
  - `an_interrupted_run_resumes_from_its_newest_checkpoint_with_the_same_flags`
  - `a_run_that_fails_is_recorded_and_the_sweep_goes_on`
  - `a_sweep_in_which_every_run_failed_fails`
  - `a_partial_calibration_file_is_not_taken_for_a_finished_one`
  - `checkpoints_are_removed_once_a_run_is_calibrated_and_not_before`
  - `only_the_winning_point_is_scored_on_the_holdout`
  - `keeping_the_best_models_deletes_only_model_directories_the_sweep_wrote`
- leaderboard:
  - `points_are_ranked_by_mean_errors_then_by_their_worst_seed`
  - `a_point_whose_seeds_overlap_the_winners_is_marked_indistinguishable`
  - `the_base_model_is_reported_beside_the_points_as_the_row_to_beat`
  - `sweep_results_are_flat_events_on_stdout_that_no_log_level_can_hide`
- `the_calibration_events_a_sweep_reads_are_parsed_from_a_real_calibrate_run`: reads a fixture
  (`crates/similarity-train/tests/fixtures/calibration.ndjson`) produced by the server's tiny testing encoder.
- `#[ignore = "needs both release binaries: just test-sweep"]`
  `a_two_point_sweep_trains_calibrates_and_ranks_end_to_end`: runs the real binaries on the tiny random model a
  test writes to disk (as `main.rs`'s `workspace()` helper does) and the synthetic sample. No real model.

**`similarity-server`:**

- `a_model_trained_at_another_token_limit_is_warned_about` (provenance check)
- `the_sweep_fixture_is_what_calibrate_writes_today`: regenerates the trainer's fixture with the testing encoder
  and compares the event names and field names the sweep reads, so a rename fails here and not at work.
- `offline::sweep`:
  - `a_grid_is_every_combination_of_its_axes`
  - `proportional_weights_are_one_point`
  - `a_grid_naming_a_field_the_config_lacks_is_refused_by_name`
  - `the_threshold_and_the_anchor_threshold_are_not_axes`
  - `off_turns_a_floor_or_a_limit_off_and_any_other_string_is_refused`
  - `a_point_that_fails_validation_is_reported_and_the_sweep_goes_on`
  - `points_are_ranked_by_errors_then_margin_then_order`
  - `an_evaluation_run_is_judged_at_the_threshold_suggested_on_the_selection_run`
  - `an_evaluation_run_of_another_model_is_refused`
  - `too_many_points_are_refused_before_any_work`
  - `the_report_and_the_events_give_the_same_numbers`
- properties (`similarity-server`'s `properties` module):
  - `a_point_that_changes_nothing_ranks_exactly_as_the_base_config_does`
  - `scaling_every_weight_of_a_point_changes_no_verdict`
  - `judging_a_point_equals_calibrate_diff_config_b_under_the_same_settings`

After steps 1, 6, and 7: `just test-real` (the trainer and `config.example.toml`'s neighbors changed);
`just test-parity` after step 6. `PROPTEST_CASES=5000 just test` once after step 3 (scoring is touched only
through `rejudge`, but the property is new).

## Documentation to update

**README.md**

- "Fine-tuning the model": a step between 2 (train) and 3 (calibrate), "Sweeping training options", with the
  sweep file, `just sweep`, the layout, what the objective is and why it ignores floors, why seeds, the meaning
  and the limits of `indistinguishable`, the holdout-once rule, continuing an interrupted sweep, disk use per run
  (a model directory each; checkpoints are removed), and the `jq` line for the leaderboard. Rows for
  `--margin`, `--scale`, `--warmup-ratio` if step 6 is taken. The run measured in
  step 8 on the synthetic sample, saying it shows the mechanics and not a result.
- "Choosing thresholds" / "Comparing two calibrations": "Trying many settings at once" (`calibrate-sweep`), the
  grid file, the `--evaluate` rule, the report on the synthetic sample, and the event table.
- `--model` and `pairs carve` come with feature 3's documentation; this plan adds the provenance warning to the
  `--model` row.
- Commands: `just sweep`, `just calibrate-sweep`, `just test-sweep`.

**CLAUDE.md**

- Commands: the three recipes.
- Things easy to get wrong:
  - "A sweep ranks on the `validation` split and touches the holdout once, for the winner. Never rank, filter, or
    choose anything on the holdout."
  - "A sweep's output is evidence, not configuration. Nothing it prints is pasted into `config.example.toml`
    without a `calibrate` run under it (D3)."
  - "Sweep point ids are content hashes of their flags: never put a path, a seed, or a timestamp into one, or a
    continued sweep reruns everything."
  - "The training sweep reads `calibration`, `suggestion`, `operating_point`, and `summary` events from
    `calibrate --ndjson`. Their names are an interface twice over (D17)."

**docs/decisions.md: D22. Sweeps judge policy again and run training as child processes.** Two kinds of sweep and
their costs; the threshold is searched, not swept; why the training objective ignores the configured policy (D3);
the validation split and holdout-once; child processes instead of moving the calibration code; seeds, and why
`indistinguishable` is not a test; content-addressed points; what is deleted and what never is. Amend D13's
"Pruning is conservative" with the sweep's checkpoint removal, and D17's interface note with the trainer as a
reader of the events.

**docs/feature-suggestions.md:** mark 6 built and correct its "Fit" paragraph. **docs/plan-features.md:** correct
the one-line note on 6.

## What can only be validated at work

Here, on the synthetic sample (46 strict pairs, a few dozen in the full file), the mechanics can be proved: grids
expand, runs resume, failures are isolated, ids are stable, events parse, the report and events agree. What
cannot be learned here, and must be done where the data lives:

- whether any training point beats the base model on the validation split, and by more than the seed spread;
- which axis ranges are worth sweeping (the defaults in the README example are placeholders, not advice);
- whether three seeds are enough: the seed spread on real data decides it;
- the validation split's size: carved at feature 3's suggested fraction of 0.15 from the 80% in `calibrate`, it
  holds about 12% of labeled pairs; whether that leaves enough recycles per owner for the intervals is a fact
  about the export;
- wall-clock and disk cost. From D13's 1.3 s per 32-pair step on Metal, a few thousand examples is on the order of
  a hundred steps a pass, so a 9-point, 3-seed, 2-pass sweep is a few hours. That is arithmetic, not a
  measurement; the README quotes only what step 8 measures.

## Dependencies on unbuilt features

- **Feature 3 (evaluation at every checkpoint).** Assumed to provide, as `docs/plans/03-checkpoint-evaluation.md`
  specifies and its planner confirmed: `record-similarity pairs carve` writing a `validation` split by owner hash;
  the trainer's `--validation-split` (on when the split has pairs) and `--no-validation`, never training on that
  split and dropping catalog groups of validation and holdout owners; `calibrate --full --split validation`; and
  `calibrate --model <dir>` with `model_dir` and `model_step` on the `calibration` event. If feature 3 is not
  built first, step 1 builds exactly those parts. Feature 3's `best_step` and `best_model` are not used: a sweep
  compares finished runs (correction 5).
- **Feature 2 (run registry).** Not required. The sweep directory is a self-contained record (`sweep.json`, each
  run's `run.json`, both of its output streams). If feature 2 is built, the assumed interface is: the trainer and
  `calibrate` accept `--run-dir <path>` and write `run.json` and `events.ndjson` there, and a manifest can name a
  `parent`. The sweep would then pass `--run-dir <run>` to each child and write its own manifest as the parent,
  and `train.ndjson`, `train.log.ndjson`, and `run.json` above become feature 2's files. Feature 2 should adopt
  this layout's per-run shape rather than the other way round.
- **Feature 5 (run report).** Would draw the leaderboard and per-point curves from these files. Nothing here waits
  for it.
- **Weights & Biases** is not involved. A sweep is files and NDJSON; feature 2's optional exporter could replay one
  later.

## Decisions for the owner

1. **One carved validation split, or k folds?** Recommended: one split carved into the file (`pairs carve`),
   which is feature 3's default and keeps one copy of the hash. k in-memory folds (`Fold { index, count }`) would
   let a sweep average each point over folds as well as seeds, which matters if one 12% split proves too small
   to separate points; it multiplies the cost by k. Decide it once, for features 3 and 6 together.
2. **What ranks a training point.** Recommended: errors at the model's own best overall threshold on the
   validation split (the `suggestion` event), mean over seeds. The alternative, the false-reject rate at a 2%
   false-accept operating point, is closer to how a threshold is finally chosen, but an operating point is omitted
   when it would repeat the one before (D17), so on a small split the objective could be missing for some runs.
   Both are reported either way.
3. **Model directories kept.** Recommended: `keep_models = "all"` by default (a model directory each, checkpoints
   removed), with `"best"` available. Deleting is not reversible, so it is opt-in.
4. **Add `--margin`, `--scale`, `--warmup-ratio` flags (step 6)?** Recommended: yes, as their own commit. They are
   the loss settings most likely to matter and the change is small and parity-checked; defaults do not change.
5. **Score the winner on the holdout automatically?** Recommended: yes (`holdout = true` by default). It is the
   step a person would take next, and doing it for the winner alone is what keeps the holdout clean.
6. **Policy sweep ranking includes floors and quantity limits at the suggested threshold.** Recommended as
   designed. The alternative, ranking by the overall-score suggestion alone, would make floor axes rank nothing.

## Size

**Large**, about a week and a half. The policy sweep (steps 3 and 4) is medium on its own, a few days, because
`rejudge`, `summarize`, and the stats already exist and the work is grid parsing, collapsing, the out-of-sample
rule, and the report. The training sweep is the larger part: a sweep file validated through the trainer's own
parser, a continuable, failure-isolating orchestrator over two binaries, and the end-to-end test
that needs both release binaries. The suggestion's "M to L" assumed a refactor this plan avoids, and did not count
the validation split (feature 3, or step 1 here) or the policy sweep.
