# Plan: evaluation at every checkpoint (feature 3)

A plan for feature 3 of `docs/feature-suggestions.md`, written against the code on `initial-implementation` as of
commit `545ef20`, so that a session with no memory of the conversation that produced it can carry it out. Nothing
here is built. Read `CLAUDE.md` first, then D13, D17, and D18 in `docs/decisions.md`, then this file.

## Contents

- [What it is for](#what-it-is-for)
- [What exists to build on](#what-exists-to-build-on)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Other features this assumes](#other-features-this-assumes)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

A fine-tuning run writes a checkpoint at the end of every pass (and every `--checkpoint-every` steps). Today the
trainer measures separation only before training and after the last step, so the README tells the owner to pick
the best pass by pointing a config at each checkpoint and calibrating it by hand. That is slow, and the README
says to do it on the `holdout` split, which turns the holdout into a validation set: the pass that looks best on
it is partly the pass that got lucky on it, and the holdout figure reported for it is then biased upward.

This feature makes the trainer measure every checkpoint on a **validation fold** that is neither trained on nor
the holdout, name the best one when the run ends, and report the holdout for that one model only. A full-policy
measurement of the candidates stays with `record-similarity calibrate`, which gains what it needs to be pointed at
a checkpoint without writing a config per checkpoint.

## What exists to build on

Verified in the code at `545ef20`:

- **The hook.** `Checkpoints::after` in `crates/similarity-train/src/main.rs` writes a checkpoint through
  `Step::checkpoint` when `Checkpoints::due` says so, logs a `checkpoint` line on stderr, pushes the path onto
  `own`, and calls `prune`. It is called from the `after_step` closure that `Cli::run` passes to `fit`.
- **The progress check.** `similarity_train::eval::separation` embeds each `PairExample`'s two texts with a plain
  encoder and returns one `Separation { width, edits, recycles }` per width, each side a `Spread { count, min,
  mean, max }`; `Separation::mean_gap` is the difference of means. It truncates by prefix and takes the cosine,
  as the server does. Its module doc already says it is "a training-time progress check, not the calibration".
- **The result events.** `Outcome::report` in `main.rs` emits `separation` with `target: RESULT` for `split`
  (`training`, `held_out`) by `stage` (`before`, `after`) by `width`, then one `finished`. A split with no pairs
  emits nothing. README "Fine-tuning the model" documents both events in a table.
- **Splits.** `data::Split::{Training, HeldOut}` and `data::pair_examples(jsonl, origin, fields, split)`.
  `Training("calibrate")` admits pairs of that split and pairs with none; `HeldOut(name)` admits only pairs that
  name it. A third split is therefore one more `HeldOut` call; no new mechanism is needed.
- **The data.** `docs/data-request-prompt.md` assigns `split` 80% `calibrate`, 20% `holdout` by a hash of the
  brand, and every line of `pairs.full.jsonl` and `catalog-groups.jsonl` carries `owner` (the brand, as a
  pseudonym). The catalog-group export already leaves out holdout brands.
- **Owner is not read by the trainer.** `PairLine` and `GroupLine` in `crates/similarity-train/src/data.rs`
  deserialize `label`, `split`, `before`, `after` and `catalog`, `listings`. `owner` is ignored, as
  `docs/plan-features.md` notes.
- **Checkpoints are model directories.** `checkpoint::write` (through `write_whole`) writes the weights,
  `provenance.json` (via `stamp`, with `trained.step`), `optimizer.bpk`, and `training_state.json`. A checkpoint
  loads with `SentenceEncoder::load` like any model.
- **Pruning.** `Checkpoints::prune` deletes the oldest of `own` beyond `keep`; `prepare` fills `own` with the
  run's earlier checkpoints (by `checkpoint::same_run`), never the one the process was loaded from.
- **Run identity.** `RunIdentity { options, data, max_tokens }`; `data` is `checkpoint::data_digest(examples)`,
  which hashes each example's texts and label in order. Nothing else about the data is in it.
- **Full-policy scoring.** `similarity_core::score::{evaluate, judge, Comparison::weigh, Evaluation::rejudge}`
  are in core. Everything around them is in the server: `offline::Offline::compare` (needs an `Embedder`, a
  `Schema`, a `Policy`), `offline::pairs::{parse_full_pairs, Selection, Pair, Label}`,
  `offline::calibrate::{calibrate, summarize, suggest_threshold}`, `offline::stats` (ranges, slices,
  `error_rates`, `operating_points` with owner-bootstrap intervals), `offline::events::emit`, and the TOML
  `Config` in `config.rs`. `CalibrateArgs` already has `--full --split <name> --min-confidence --ndjson --seed`.
  The model comes only from `config.model.dir` (`main.rs`, `offline()` then `load_embedder`).
- **Threshold-free statistics already exist.** `stats::operating_points` looks at the overall similarity only
  (weights, no floors or thresholds), and `suggest_threshold` finds the best cut. These are the only calibration
  outputs that compare two models fairly (see below).

## What the suggestion got wrong or left open

1. **"Writes `separation.json` into the checkpoint" contradicts D18.** A model directory records inputs, not
   results, and is written whole and never modified in place. A checkpoint is complete before anything could
   measure it (D13 saves first on purpose). Results go to the result stream only. The owner already took this
   decision for `provenance.json` ("Decisions that were taken", item 2, in `docs/plan-features.md`).
2. **"`--eval-config` runs the full-policy calibration at each checkpoint" has two problems.**
   - *Where the code lives.* Calibration needs the server's pair parser, config, `Offline`, and `stats`. D13 only
     forbids the server depending on the trainer, but the trainer depending on `similarity-server` would link
     tonic, tokio, the Qdrant client, and `ureq` into the training binary and invert the layering (the server is
     the top of the graph). Moving `offline` (minus the text report) and the config's schema and policy parts into
     core or a new crate is a refactor of roughly 3,000 lines for a convenience. The plan-features note that "the
     scoring it needs is in `similarity-core` already" is true of `evaluate`/`judge` and not of the statistics.
   - *What it would measure.* A config's `threshold`, `anchor_threshold`, and `reject_below` floors belong to one
     model (D3). Counting each checkpoint's false accepts and rejects at the base model's thresholds compares
     checkpoints on a scale that only fits one of them. Only threshold-free numbers (operating points, the
     suggested cut and its margin, the score ranges) compare models fairly.
   So the full-policy half is done after the fact by `calibrate`, pointed at each checkpoint (step 5).
3. **"Data.rs (a third split)"** is smaller than it sounds (one more `Split::HeldOut`), and larger in one way it
   does not mention: **catalog groups leak.** The validation brands' catalog groups would still be trained on,
   since `GroupLine` has no notion of owner. The fold is only brand-disjoint if groups of validation owners are
   dropped too.
4. **"Carved from `calibrate` by brand hash" does not say who carves.** If the trainer carves in memory,
   `calibrate` cannot select the same pairs without a second copy of the hash. This plan carves the file once,
   rewriting `split`, so every tool keeps selecting by `split` alone (step 4).
5. **The selection metric is not chosen.** `gap_of_means` is the only number the trainer has; a mean can rise
   while the overlap grows. This plan selects on the area under the ROC curve (AUC) of field similarities at the
   served width, and reports `gap_of_means` beside it (decision 3).
6. **The README's advice is itself the leak.** "Point a config's `model.dir` at a checkpoint and calibrate it ...
   That is how to pick the best pass" together with step 3's `--split holdout` is exactly the selection on the
   holdout the suggestion's caveat warns about. The README has to change whether or not this is built.
7. **Already true:** loss curves exist (`training` lines, every `--log-every` steps), though on stderr under
   `RUST_LOG`; keeping them is feature 2's job, not this one's.
8. **Noticed, not in scope:** the trainer ignores `label_confidence` and trains on `low`-confidence pairs, while
   `calibrate --full` drops them by default (`--min-confidence medium`). The validation fold below applies the
   calibrate default so that the two tiers measure the same pairs; whether training should too is a separate
   question for the owner.

## Design

Two tiers. Tier 1 is in the trainer, cheap, and runs at every checkpoint. Tier 2 is `calibrate`, run over the
surviving checkpoints when the owner wants the full policy.

### The validation fold

A pair is in the validation fold when its `split` is `validation` (`--validation-split`, default `validation`).
Nothing else decides it, in either binary.

- **Trainer.** `Split::Training("calibrate")` already excludes such a pair from training. The trainer reads it
  with `Split::HeldOut("validation")` and a new minimum confidence (`--validation-min-confidence`, default
  `medium`, the calibrate default; a pair with no confidence recorded is kept, as `calibrate` keeps it).
- **Validation is on when the fold has pairs**, off otherwise, with `--no-validation` to turn it off. A file
  without a `validation` split runs exactly as today, event for event.
- **Owner-disjoint groups.** `GroupLine` gains `owner: Option<String>`. `data::group_examples` gains an
  `exclude: &BTreeSet<String>` of owners and returns the count it left out. The trainer passes the owners of every
  pair in the validation or holdout split. A group without an owner is kept and counted in the `catalog groups`
  log line (`groups_without_owner`), since it cannot be checked.
- **`PairLine` gains `owner: Option<String>`**, used only to build that set. `PairExample` and `data_digest` do
  not change, so a checkpoint written before this change still resumes on the same data.

### Carving the fold out of an existing export

`record-similarity pairs carve` (a new `pairs` command group in `crates/similarity-server/src/main.rs`, code in
`crates/similarity-server/src/carve.rs`):

```sh
record-similarity pairs carve pairs.full.jsonl --from calibrate --into validation --fraction 0.15 \
  --salt validation-1 > pairs.carved.jsonl
```

- An owner goes to `--into` when the first eight bytes of `SHA-256(salt, 0x00, owner)`, read as a big-endian
  `u64`, are below `fraction * 2^64` (computed with checked arithmetic). Whole owners move, never single pairs.
- Only pairs whose `split` equals `--from` are candidates. Other lines, comments, and blank lines pass through
  byte for byte. A candidate without an `owner` stays and is counted.
- A moved line is parsed as a `serde_json::Map`, its `split` replaced, and written back. Key order may change
  (the workspace's `serde_json` has no `preserve_order`); values do not (`float_roundtrip` is on, D19).
- Output on stdout. One NDJSON log line on stderr through `similarity-log` with `owners_moved`, `pairs_moved`,
  `edits_moved`, `recycles_moved`, `without_owner`. A warning when fewer than 40 recycles moved: that is the
  data request's own minimum for the holdout, not a new number.
- Carving again with the same salt moves nothing more: the owners that hash below the fraction have already left
  `--from`.
- The fraction and salt are not measurements and not thresholds. They change which pairs are trained on, and so
  change `data_digest`, as any change of data does.

`docs/data-request-prompt.md` gains one sentence: a new export may assign `validation` directly (70/10/20 by the
same brand hash) instead of being carved afterwards.

### Tier 1: measuring each checkpoint in the trainer

**What is scored.** The checkpoint directory just written, loaded back with `SentenceEncoder::load(path,
device)` on the training device and scored with `eval::separation` over the validation examples at every
trained width. Loading it back (90 MB, against the 265 MB just written) scores exactly what would be served and
avoids exposing the training-mode encoder from `Step`, whose weights track gradients. `SentenceEncoder` is not
`Clone`, so the in-memory alternative would need a new core method; it is not worth one.

**When.** At step 0 (the model the run started from: the `before` line), after every checkpoint, and on the
final model after it is saved and verified (the `after` line). With `--no-checkpoints`, only `before` and
`after`. A checkpoint that cannot be scored is a warning and is not a candidate; training continues, since the
checkpoint itself is safely on disk.

**The score.** New in `similarity-core`, module `separation` (pure, no I/O, so D11 holds):

```rust
/// The probability that a randomly chosen edit scores above a randomly chosen recycle, ties counting half.
///
/// `None` unless both sides have at least one score.
pub fn area_under_curve(edits: &[f32], recycles: &[f32]) -> Option<f64>
```

Rank-based (sort the union, sum the edits' ranks with ties averaged, Mann-Whitney U), `O(n log n)`, all counts
through `saturating_*`. In core so that `calibrate` can report the same number over the overall similarity
(step 6), and so its rules go in core's `properties` module. `eval::Spread` gains nothing; `eval::Separation`
gains `auc: Option<f64>`, computed from the per-example cosines it already collects.

**Selection.** In a new `similarity-train` module `select` (pure, unit-tested):

```rust
/// One model that can be chosen: the one a run started from, a checkpoint, or the final model.
pub struct Candidate {
  pub step: usize,
  pub dir: PathBuf,
  pub auc: f64,
  pub gap_of_means: Option<f32>,
}

/// The best candidate so far: the highest AUC at the selection width, rounded to four places; on a tie, the
/// earlier step (less training for the same result).
pub struct Best(Option<Candidate>);

impl Best {
  pub fn offer(&mut self, candidate: Candidate) -> bool; // true if it became the best
  pub fn get(&self) -> Option<&Candidate>;
}
```

The selection width is `--select-width`, default the narrowest trained width (256 under the default `--dims`,
which is `output_dim` in `config.example.toml`), replaced by the native width when wider, as `trainable_widths`
does. It must be one of the trained widths, or the run is refused before training.

**Plumbing in `main.rs`.** A `Validation` struct (examples, tokenizer clone, device clone, widths, batch size,
selection width, `Best`) owned by `Cli::run`, borrowed mutably by the `after_step` closure next to
`checkpoints`. `Checkpoints::after` returns the path it wrote (or `None` when not due), and the closure scores
it. `Checkpoints::prune` takes the best path and never deletes it: `--keep-checkpoints N` keeps the newest N
**plus** the best.

**Resume.** Selection is not in `RunIdentity`: it changes no training step, and `CLAUDE.md` reserves
`RunIdentity` for what does. A resumed process does not know its predecessor's scores, so before training it
scores every candidate that still exists: the checkpoint it resumed from (its `before`), each of `own` (the
run's earlier checkpoints that `prepare` found), and the base model at `provenance.trained.base.dir` if it is
still there with `trained.base.digest`; otherwise it warns that the base was not re-scored. These lines carry
`rescored: true`, so that a reader concatenating the two processes' output can drop the duplicates. The best of
a resumed run is then the best of what is on disk, which is all that could be chosen anyway.

**Holdout.** With validation on, the holdout is measured once, for the selected model: after the final model
is scored, the best candidate is loaded (or the in-memory final model reused if it won) and scored on the
`held_out` pairs, emitted with `stage: "selected"`. The `held_out` `before` line stays (it is the base, not a
choice). The `held_out` `after` line for the final model is left out when the final model was not selected
(decision 5). With validation off, the held-out lines are exactly as today.

**`--out` still holds the final model.** It is saved the moment training ends (D13). The best model is named,
not copied.

### Events

Additions only; no field or event is renamed or repurposed.

`separation` (stdout, `target: RESULT`):

| Field | Change |
|---|---|
| `split` | New value `validation`. |
| `stage` | New values `checkpoint` (validation only) and `selected` (held-out only). |
| `step` | New, on every line: 0 or the resumed step for `before`, the step for `checkpoint`, the total for `after`, the selected step for `selected`. |
| `auc` | New, four places, when both labels are present. |
| `path` | New, on `checkpoint` and `selected`: the directory scored. |
| `rescored` | New, `true` on lines a resumed run re-measured; absent otherwise. |

```json
{"message":"separation","split":"validation","stage":"checkpoint","step":120,"path":"models/ft.checkpoints/step-000120","width":256,"edits":61,"edits_mean":0.91,"recycles":19,"recycles_mean":0.58,"gap_of_means":0.33,"auc":0.9412}
```

(Illustrative, not measured. The README example must be replaced by a real run on the synthetic sample.)

`finished` gains, when validation is on: `validation_examples`, `select_width`, `best_step`, `best_model` (the
directory: the base, a checkpoint, or `--out`), `best_auc`, and `best_is_final` (boolean). When no candidate
had both labels, the `best_*` fields are absent and a warning says why. Named `best_model`, not the suggestion's
`best_checkpoint`, because the winner can be the base or the final model, neither of which is a checkpoint.

New stderr log lines: `validation pairs` (count, edits, recycles, skipped for confidence, owners), and the
`catalog groups` line's new counts.

### Tier 2: the full policy over checkpoints, in `calibrate`

- **`calibrate --model <dir>`** replaces `config.model.dir` for that run. Everything else (schema, weights,
  `max_tokens`, `output_dim`, device) comes from `--config`. The `calibration` header event gains `model_dir`
  and, when the directory's `provenance.json` says so, `model_step` (from `trained.step`). D17 allows added
  fields.
- **`just calibrate-checkpoints <checkpoint-dir> <pairs> <config> *flags`** runs `calibrate --full --ndjson
  --split validation --model <each step-*>` and writes one NDJSON stream per checkpoint beside the checkpoint
  directory (`<checkpoint-dir>.calibration/step-000120.ndjson`), never inside a model directory (D18).
- **What to read:** the operating points and the suggested cut with its margin, which are threshold-free. Two
  checkpoints compare with `calibrate-diff` as two models already do. Then the holdout, once, for the model
  chosen: `just calibrate pairs.carved.jsonl config.toml --full --split holdout --model <chosen>`.
- **AUC in the calibration** (step 6, optional): `offline::stats::area_under_curve(calibration)` over the overall
  similarity of labeled pairs, calling the core function, emitted as a new `auc` field on the `summary` event
  and one line of the text report. That makes tier 2's ranking directly comparable with tier 1's.

### Cost

Per checkpoint: one model load and one forward pass over the validation texts (two per example, per field). A
forward pass without gradients costs a fraction of a training step on the same texts, so a validation fold of a
tenth of the pairs costs a few percent of a pass. This is an estimate; the README may only quote it once a run
has measured it (`elapsed_s` on the `checkpoint` and `separation` lines makes that possible: add
`eval_s` to the `separation` line of each checkpoint).

## The work, in order

Each step is one commit, leaves `just ci` green, and is written test first.

1. **Core: `separation::area_under_curve`.** The function, its unit tests, and two properties in
   `similarity-core`'s `properties` module. Run `PROPTEST_CASES=5000 just test similarity-core`.
2. **Trainer data: owner, the validation fold, owner-disjoint groups.** `PairLine.owner`, `GroupLine.owner`,
   `group_examples(..., exclude)`, the confidence filter for the validation fold, the log counts. No behavior
   change for a file without `validation` pairs, except that groups of holdout owners are now dropped
   (decision 9).
3. **Trainer: score, select, report.** `eval::Separation::auc`, the `select` module, `Validation` in `main.rs`,
   `Checkpoints::after` returning its path, pruning that keeps the best, resume re-scoring, the `selected`
   holdout line, the new event fields, `--validation-split`, `--validation-min-confidence`, `--select-width`,
   `--no-validation`. Then `just test-parity` (the training step must be untouched) and `just test-real`.
4. **Server: `pairs carve`.** The command, its tests, a property in the server's `properties` module.
5. **Server: `calibrate --model`** and the header fields; the `calibrate-checkpoints` recipe in the `justfile`.
6. **Server: AUC in the calibration summary and report** (optional; decision 7).
7. **Docs.** README, `CLAUDE.md`, `docs/decisions.md` (D22), `docs/data-request-prompt.md`,
   `docs/feature-suggestions.md` (mark feature 3 built, with what differs), `docs/plan-features.md` (its row).
   Re-run every changed README example on `data/calibration.sample.full.jsonl` carved with a fixed salt; the
   sample has seven owners, so pick a salt that leaves both labels in the fold and say so in the README.

## Tests

None needs the model, the network, or Qdrant unless marked. Trainer tests use the tiny model from
`similarity_core::testing::write_tiny_model`, as `main.rs`'s tests already do.

**`similarity-core`** (`separation.rs`, `properties.rs`):

- `the_area_under_the_curve_is_one_when_every_edit_outscores_every_recycle_and_zero_when_none_does`
- `tied_scores_count_half_so_identical_sides_score_one_half`
- `with_either_side_empty_there_is_no_area_under_the_curve`
- property `the_area_is_between_zero_and_one_and_swapping_the_sides_gives_one_minus_it`
- property `the_area_depends_only_on_the_order_of_the_scores` (unchanged by any strictly increasing map)

**`similarity-train`, `data.rs`:**

- `a_pair_in_the_validation_split_is_neither_trained_on_nor_held_out`
- `a_validation_pair_below_the_minimum_confidence_is_left_out_and_counted`
- `catalog_groups_of_an_owner_with_validation_or_holdout_pairs_are_left_out_and_counted`
- `a_catalog_group_without_an_owner_is_kept_and_counted`
- `reading_the_owner_leaves_the_fingerprint_of_the_training_data_unchanged`

**`similarity-train`, `select.rs`:**

- `the_best_candidate_has_the_highest_area_and_a_tie_goes_to_the_earlier_step`
- `a_candidate_without_both_labels_is_never_chosen`
- `the_model_a_run_started_from_can_be_the_best`

**`similarity-train`, `main.rs`** (end to end through `Cli::run` on the tiny model, as
`a_trained_model_and_its_checkpoints_record_where_they_came_from` does):

- `every_checkpoint_reports_its_validation_separation_on_stdout`
- `finished_names_the_best_model_and_the_holdout_is_reported_only_for_it`
- `pruning_keeps_the_best_checkpoint_beyond_the_number_to_keep`
- `measuring_checkpoints_does_not_change_the_weights_a_run_ends_on` (`--no-dropout`, the same file with and
  without `--no-validation`, equal final digests)
- `a_resumed_run_rescores_what_is_on_disk_and_names_the_same_best_as_an_uninterrupted_one`
- `without_validation_pairs_a_run_reports_exactly_what_it_did_before` (the event list of
  `results_are_json_lines_on_stdout_that_no_log_level_can_hide` unchanged)
- `a_selection_width_that_is_not_trained_is_refused_before_training`
- `a_checkpoint_that_cannot_be_scored_is_a_warning_and_training_continues`

**`similarity-server`:**

- `carving_moves_whole_owners_from_one_split_to_another_and_nothing_else`
- `the_same_salt_carves_the_same_owners_and_another_salt_carves_others`
- `a_pair_without_an_owner_stays_where_it_was_and_is_counted`
- `comments_blank_lines_and_other_splits_pass_through_byte_for_byte`
- `carving_twice_with_the_same_salt_moves_nothing_more`
- property `carving_is_a_partition_by_owner` (every owner's pairs end on one side; the line count is unchanged)
- `calibrate_with_a_model_override_scores_with_that_model_and_names_it_in_the_header`
- (step 6) `the_calibration_summary_reports_the_area_under_the_curve_of_the_overall_score`

**Slow, `#[ignore]`d, existing:** `just test-parity` and `just test-real` after step 3, to prove the training step
and the saved model are unchanged on the real checkpoint. No new slow test is needed: tier 1 scores with the same
`eval::separation` the real tests already exercise.

**Only at work, on real data:** whether a validation fold of this size ranks checkpoints stably (run two salts
and see whether `best_step` moves); whether the selected model beats the final one on the holdout; whether
tier 1's AUC and tier 2's operating points agree on the ranking. None of this can be judged on 53 synthetic
pairs.

## Documentation to update

- **README, "Fine-tuning the model":** a "Choosing a checkpoint" passage (validation fold, carving, what
  `best_model` means, tier 2 recipe); the options table (`--validation-split`, `--validation-min-confidence`,
  `--select-width`, `--no-validation`; `--keep-checkpoints` now "plus the best"); the events table (the new
  `separation` stages and fields, the `finished` fields); a real example line. **Remove** "That is how to pick the
  best pass" from the checkpoint bullet and make step 3 say the holdout is measured once, for the chosen model.
- **README, "Choosing thresholds":** `calibrate --model`, and that thresholds are still set on the holdout of the
  chosen model only (D3 unchanged).
- **README, command reference:** `pairs carve`, `just calibrate-checkpoints`.
- **`CLAUDE.md`:** the commands; under "easy to get wrong": never select a checkpoint on the holdout; the
  validation fold is chosen by `split` alone, so carve the file rather than filtering in code; selection settings
  stay out of `RunIdentity`; a measurement never goes into a checkpoint directory.
- **`docs/decisions.md`:** a new **D22, "A checkpoint is chosen on a validation fold, and the holdout is measured
  once"**: why the fold is a `split` value written into the file, why AUC, why the full policy stays in
  `calibrate` (layering, and thresholds that do not transfer between checkpoints), why pruning keeps the best,
  how a resume re-scores. A sentence in D13's pruning bullet; a sentence in D17 that `calibration` gained
  `model_dir` and `model_step`.
- **`docs/data-request-prompt.md`:** the optional 70/10/20 assignment.
- **`docs/feature-suggestions.md` and `docs/plan-features.md`:** mark feature 3 built and say what differs.

## Other features this assumes

- **Feature 2 (run registry), not built.** Nothing here depends on it. The events go to stdout as today; if a run
  directory exists later, they land in its `events.ndjson` unchanged. The interface assumed is only that feature
  2 records stdout lines verbatim.
- **Feature 5 (run report), not built.** It would chart `separation` lines with `split: "validation"` by `step`.
  The interface it can rely on is the table above.
- **Feature 6 (sweeps), not built.** It would rank sweep points by `finished.best_auc`. If it needs the full
  policy in process, that is when moving `offline::{pairs, calibrate, stats}` and the schema and policy half of
  `config.rs` into a shared crate pays for itself; this plan deliberately does not. Its planner proposed an
  in-memory `similarity_core::fold::Fold { index, count }` (`--validation-fold 1/5` in the trainer, `calibrate
  --fold 1/5`) for k-fold use. That is the alternative under decision 1; if the owner wants k folds, it replaces
  `pairs carve` and the `validation` split value, and the rest of this plan is unchanged (the fold decides which
  pairs are validation, nothing else). Also from that planner, and true: a checkpoint at the end of pass k is not
  the model `--epochs k` would give, because the learning rate decays over the whole run's steps. Checkpoint
  selection chooses a point on one schedule; it does not replace sweeping the number of passes.
- **Feature 4 (promotion gate), not built.** Its "calibration ran on the holdout split" check fits: the holdout
  run is the one made for the chosen model.

## Decisions for the owner

Each has the default this plan uses.

1. **Where the validation fold comes from.** Default: a `split: "validation"` value in the file, written by
   `pairs carve` for an existing export or directly by a new one. Alternative: the trainer hashes owners in
   memory, which needs a matching option in `calibrate` and a shared hash function.
2. **How big, and which salt.** Default: `--fraction 0.15` of `calibrate` (about 68/12/20 overall), salt given
   on the command line and recorded in the README example. A larger fold ranks checkpoints more reliably and
   trains on less.
3. **The selection metric.** Default: AUC of field similarities at the selection width. Alternatives:
   `gap_of_means` (sensitive to a few extreme pairs, rewards moving means over reducing overlap) or the fewest
   errors at the best cut (coarse at small counts).
4. **The selection width.** Default: the narrowest trained width (256 by default, matching `output_dim` in
   `config.example.toml`). Alternative: require `--select-width` explicitly, in the spirit of D4.
5. **Holdout reporting with validation on.** Default: only for the selected model (`stage: "selected"`), the
   final model's `held_out` `after` line left out unless it was selected. Alternative: keep the `after` line too,
   at the risk of a person choosing between the two by eye on the holdout.
6. **Pruning.** Default: `--keep-checkpoints N` keeps N newest plus the best. Alternative: keep N including the
   best (the best can then push out a newer one).
7. **Full policy at every checkpoint.** Default: no; `calibrate --model` and a recipe over the checkpoints, plus
   AUC in the calibration summary. Alternative: move the offline statistics into a shared crate now.
8. **On by default.** Default: validation runs whenever the pairs have a `validation` split; `--no-validation`
   turns it off. Alternative: an explicit `--validation` flag.
9. **Owner-disjoint catalog groups.** Default: groups of validation and holdout owners are dropped, groups with
   no owner kept and counted. This changes the training data, and so `data_digest`, for an export that did not
   already exclude holdout brands; a run started before the change then cannot be resumed after it.
10. **`--out` holds the final model**, not the best. Default: keep it so (D13 saves it before measuring). The
    best is named in `finished`; copying it is a `cp -r`.

## Size

**Medium.** No new dependency (`sha2` and `serde_json` are already in both crates; `proptest` is already a
dev-dependency of core and server), no change to the training step, the loss, or the model format, and every
piece has an existing pattern to copy (`Checkpoints`, `Outcome::report`, `calibrate`'s arguments). The bulk is
the trainer plumbing in step 3 (resume re-scoring and pruning that keeps the best are the fiddly parts) and the
documentation. It would be large only if decision 7 went the other way.
