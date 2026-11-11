# Plan: a pinned verdict snapshot, checked in CI

Feature 9 of `docs/feature-suggestions.md`. Written on 2026-11-11 against `initial-implementation` at `545ef20`,
before any of it was built, so that a session with no memory of the conversation that produced it can carry it
out. Read `CLAUDE.md` first, then this file. Delete this file, or fold what is still true into `docs/decisions.md`,
when the work is done.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

The tests that touch the real model (`just test-parity`, `just test-real`) run only on someone's machine. An
encoder, tokenizer, loader, or scoring change can merge without anyone having run them, and `CLAUDE.md`'s "run
`just test-real` after touching the encoder" is a rule, not a check. This feature turns it into a check:

- a committed file of curated, synthetic hard cases (`data/regression.jsonl`);
- a committed calibration of those cases under the pinned model and `config.example.toml`
  (`data/regression.snapshot.ndjson`), in the format `calibrate --ndjson` already writes;
- a CI job that fetches the pinned model (cached), runs the parity tests, scores the cases afresh, and fails on any
  verdict flip, any change in why a pair was rejected, or any score that moved more than a tolerance;
- a check in plain `cargo test`, with no model, that judges the committed snapshot again under the current
  `config.example.toml` and the current scoring code, and fails on the same things. Most changes that move a verdict
  (weights, floors, thresholds, judging code) are caught there, on every machine, in seconds.

The snapshot pins behavior, not correctness. Several curated cases are ones the shipped configuration is expected
to get "wrong" (same-brand variant changes, labeled `unsure`); the point is that nobody changes what happens to
them without seeing it. The snapshot is never a calibration: its cases are chosen to be hard, not sampled, and no
threshold may be read from it.

## What already exists

Verified in the tree at `545ef20`:

- **The run format.** `offline::events::emit` (`crates/similarity-server/src/offline/events.rs`) writes a
  `calibration` header (fingerprint, dimension, max tokens, threshold, anchor threshold, `pairs_sha256`, seed),
  one `schema_field`/`schema_quantity` line per schema item, and per pair a `pair` line plus `pair_field`,
  `pair_quantity`, and `pair_reason` lines. Similarities go through `six` (six decimals); factors are written
  unrounded. `offline::events::read` reads a run back into `Run { info: RunInfo, calibration }`, skipping lines
  whose `target` is not `similarity_log::RESULT` and ignoring keys it does not use (so a line without `timestamp`
  reads fine).
- **The comparison.** `offline::diff::diff(&Calibration, &Calibration) -> Diff` matches pairs by id and lists
  `flips`, totals, slices, and field-range shifts. `diff::failure(&Diff, FailOn)` implements `--fail-on
  new-false-accepts | any-flip | errors-up`. `diff::same_pairs` refuses two runs over different inputs unless
  `--intersect`. `diff::rejudge(&Run, &Schema, &Policy)` judges a run again under another config with no model
  (`Evaluation::rejudge` in `similarity-core/src/score.rs`), and refuses a change of field or quantity names.
- **The CLI.** `CalibrateArgs` and `CalibrateDiffArgs` in `crates/similarity-server/src/main.rs`. `--fail-on` takes
  one value today.
- **Pairs.** `offline::pairs::{parse_pairs, parse_full_pairs}` and `Pair` (`pairs.rs`). The full format carries
  `pair_id`, `variant_change`, `note`, `label` (`unsure` is scored and counted nowhere). `content_id` hashes
  `before` and `after` when there is no `pair_id`.
- **Scoring a pair.** `Offline::compare` (`offline/mod.rs`) passes `None` as the anchor to `score::evaluate` and an
  empty anchor to `judge_quantities`: "A pair has no history, so there is no anchor and nothing can be restored".
- **Anchor helpers, public in core.** `score::extend_anchor` (`similarity-core/src/score.rs:182`) and
  `quantity::first_values` (`similarity-core/src/quantity.rs:325`), the same two calls `Engine::store_import` uses
  to build an imported record's anchor from its original version (D20).
- **Capturing results in a test.** `similarity_log::testing::capture` (behind the `testing` feature, already a
  dev-dependency feature of `similarity-server`) returns the lines `emit` would write to stdout.
- **The model pin.** `fetch::DEFAULT_REVISION` and `DEFAULT_CHECKSUMS` (`crates/similarity-server/src/fetch.rs`).
  `model fetch` re-hashes a file already on disk and skips the download when it matches, so a restored cache is
  verified for free. The pinned fingerprint is `f46e48ee0ea2dd02` (D18), `-d256-t256` under the example config.
- **Real-model tests.** `similarity-core/tests/real_parity.rs` (embeddings to cosine above 0.99999 of PyTorch,
  max component error under 2e-4), `similarity-train/tests/real_parity.rs`, and
  `similarity-server/tests/real_e2e.rs` (needs Qdrant too). All `#[ignore]`d, reading `./models/mdbr-leaf-mt`.
  Dev builds compile dependencies at `opt-level = 3` (`[profile.dev.package."*"]`), so the model runs at a usable
  speed under `cargo test`.
- **CI.** `.github/workflows/ci.yml` has `fmt`, `clippy`, `test` (ubuntu and macOS), `qdrant` (the store contract
  against a service container), and `deny`. A comment there says the model-dependent tests stay local on purpose
  because of the 90 MB download. CI calls `cargo` directly, not `just`.
- **Text normalization and casing.** Whitespace is normalized before hashing (`record::normalize_text`, property
  tested), and the pinned tokenizer lowercases (`"lowercase": true`). So a whitespace-only edit is `Unchanged` and
  an all-caps rewrite tokenizes identically, whatever the model.

## What the suggestion got wrong or left open

1. **The blank-and-refill swap cannot be written as a calibration pair.** It is two revisions and an anchor;
   `Offline::compare` has no anchor, so the refilled field comes out `Added` and is left out of the score, which
   is the opposite of what the engine does (`Restored`, scored against the anchor, D8). The same goes for "a
   quantity removed and refilled". Either pairs learn to carry an anchor, or these cases stay out of the snapshot.
   This plan adds an optional `anchor` record to a pair (decision 1).
2. **"Scores to three decimals" would be a second format.** `calibrate --ndjson` already writes six decimals and
   `events::read` already reads it. The snapshot should be a run file, so that `calibrate-diff` can read it and no
   new parser is needed. The tolerance is applied when comparing, not by rounding what is stored.
3. **`calibrate-diff` cannot fail on a score change.** It reports flips and field ranges only. "Fails on a score
   change above 0.005" needs a new comparison (`moved`) and a new gate. A verdict that stays `rejected` but for
   another reason (a floor instead of the threshold) is also invisible to it today.
4. **The output is not byte-stable.** Every line carries a `timestamp`, so committing raw `--ndjson` output makes
   every regeneration a whole-file diff, and the reviewer of a snapshot change cannot see what moved. Six-decimal
   values can also differ in the last digit between platforms. The committed file drops `timestamp` (decision 6),
   and the check is by tolerance, never by bytes.
5. **"The thresholds in `config.example.toml` match a fresh run" is not checkable as stated.** A threshold is
   chosen from a suggestion, not equal to it (the config has 0.90, the sample suggests 0.895). What can be checked
   is that the report lines the README quotes are what `calibrate` prints. The config comments quote the same
   ranges at two decimals and are left to that check by proxy.
6. **Two of the listed hard cases do not exercise the model.** Whitespace-only edits are `Unchanged` by hash, and
   all-caps rewrites tokenize identically under a lowercasing tokenizer. They are still worth a line each: they pin
   `normalize_text` and the tokenizer's casing setting, which a model change could alter.
7. **Most changes can be caught with no model at all,** which the suggestion did not notice. `rejudge` applies the
   current weights, floors, thresholds, limits, and judging code to the snapshot's stored similarities. Only a
   change to what the similarities are (model files, encoder, loader, tokenizer, pooling, truncation) needs the
   CI job.
8. **`plan-features.md` says a committed run plus `calibrate-diff --fail-on any-flip` is "most of it".** That is
   right for flips. Missing: the score gate (3), the anchor (1), the byte stability (4), a guard against cases that
   sit on a threshold and would flip on platform noise, and the no-model check (7).

## Design

### Anchored pairs

A pair may carry a third record, `anchor`: the record as it was first created. It means "this record was created
as `anchor`, its current stored version is `before`, and `after` arrives". It is exactly a record imported with an
`anchor` (D20) followed by one `Evaluate`, so it is built the same way:

- `Pair` gains `pub anchor: Option<RecordFields>`. The strict format allows `anchor` as an optional fifth key
  (still `deny_unknown_fields` for anything else); the full format reduces it to the schema like `before` and
  `after`, and counts what it drops under the same `ignored_key` lines.
- `content_id` hashes the anchor only when there is one, so every existing pair keeps its id.
- `Offline::compare` takes the pair's anchor. With one: embed it (plan with no baseline), build the anchor
  composite as `score::extend_anchor(schema, dimension, &anchor.composite, &before.composite)`, pass it to
  `score::evaluate`, and pass `quantity::first_values(&anchor_quantities, &before_quantities)` as the anchor to
  `judge_quantities`. `policy.anchor_threshold` then applies to the pair, as it would in the engine.
- The `pair` event gains `anchored` (boolean) and `anchor_similarity` (six decimals, left out when absent).
  `events::read` reads `anchor_similarity` back into `Evaluation::anchor_similarity`, so `rejudge` applies
  `anchor_threshold` to an anchored pair. Fields are added; nothing is renamed (D17).
- The text report marks an anchored pair (`anchored` after its verdict) and otherwise treats it like any pair.

A property test holds the offline path to the engine's: for any anchored pair, `Offline::compare` and
`Engine::import` with that anchor followed by a `DryRun` evaluate give the same evaluation. That keeps the anchor
construction from being duplicated in two places that could drift.

### Scores that moved

In `offline/diff.rs`:

```rust
/// What about a pair moved between two runs, beyond the tolerance or at all.
#[derive(Debug, Clone, PartialEq)]
pub enum Movement {
  /// The overall similarity.
  Overall { old: f32, new: f32 },
  /// One field's similarity, or its change kind (`modified` to `restored`).
  Field { field: String, old: Option<f32>, new: Option<f32>, old_change: FieldChange, new_change: FieldChange },
  /// The similarity to the anchor.
  Anchor { old: Option<f32>, new: Option<f32> },
  /// A quantity's factor or change kind.
  Quantity { quantity: String, old: Option<f64>, new: Option<f64> },
  /// The set of reason codes, by subject.
  Reasons { old: Vec<String>, new: Vec<String> },
}

/// One pair, and what moved about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Moved {
  pub id: String,
  pub movements: Vec<Movement>,
}

/// Pairs both runs have whose scores moved by more than `tolerance`, or whose kinds or reasons changed at all.
pub fn moved(old: &Calibration, new: &Calibration, tolerance: f32) -> Vec<Moved>;
```

Similarities are compared with `tolerance`. Change kinds and reason codes (`Reason::code` plus `subject`) must be
equal. Factors are plain arithmetic on the record, independent of the model, and compared relative to their size
at `1e-9`.

`Diff` gains `moved: Vec<Moved>` and `diff` gains a `tolerance: f32` argument. `FailOn` gains `Moved`.
`CalibrateDiffArgs` gains `--tolerance` (default `0.001`, decision 2) and `--fail-on` becomes repeatable
(`Vec<FailOn>`, `action = Append`), so one call can gate on flips and movement together; one `--fail-on WHAT`
still means what it did. The text report gains a "scores that moved more than 0.001" section, printed only when
something moved. The NDJSON form gains one `moved` line per movement (`pair_id`, `what` = `overall`, `field`,
`anchor`, `quantity`, or `reasons`, `subject`, `old`, `new`, and for kinds `old_change`/`new_change`), and the
`diff` line gains a `moved` count. A list of codes cannot be one value (D15 forbids the preformatted string it
would need), so a `reasons` movement is written as one `moved` line per code that appeared or disappeared, with
`what = "reason_added"` or `"reason_removed"` and `subject` = the code.

### Cases on a border

```rust
/// A score in `calibration` closer than `margin` to a limit it is judged against.
pub struct Border { pub id: String, pub what: String, pub value: f64, pub limit: f64 }

pub fn borderline(calibration: &Calibration, schema: &Schema, policy: &Policy, margin: f32) -> Vec<Border>;
```

Checks the overall similarity against `policy.threshold`, each field similarity against its `reject_below`, the
anchor similarity against `anchor_threshold`, and each factor against `reject_above_factor`. Regenerating the
snapshot refuses to write while any pair is on a border (decision 5): a curated case that sits within the
tolerance of a threshold would flip on platform noise, and a flaky gate gets switched off.

### The files

`data/regression.jsonl`, full format, synthetic, every line with a `pair_id` (prefix by kind, `r-size-1`), a
`note`, and `variant_change` where it is one. Labeled `edit` or `recycle` where the intended verdict is not in
doubt, `unsure` for same-brand variant changes (the data request expects those to be undecided). A comment block
at the top says it is synthetic, what it is for, and that no threshold is read from it. About 30 pairs:

| Kind | Cases |
|---|---|
| Same-brand variants (`unsure`) | flavor, strain, size (an eighth to a quarter, 3.5 g to 7 g), potency (100 mg to 200 mg) |
| Unit rewrites (`edit`) | 3.5 g to 0.125 oz; 1000 mg to 1 g (no change); 1 fl oz to 30 ml |
| `%` to `mg` (`edit`) | `thc_per_item` 20 % to 700 mg (`incomparable`, rejects nothing under the shipped config) |
| A quantity removed (`edit`) | `weight_per_item` dropped, text unchanged |
| A quantity removed and refilled (`recycle`, anchored) | anchor 3.5 g, before none, after 28 g: `Restored`, factor 8 |
| Blank-and-refill text swap (`recycle`, anchored) | anchor has a description and brand, before has them blank, after fills them with another product's |
| The first step of that swap (`edit`) | the blanking alone: `Removed`, out of the score |
| Whitespace only (`edit`) | doubled spaces, trailing newline, tabs in the description |
| Unicode look-alikes | Cyrillic `а` in a name (`edit`), full-width digits `１００mg`, a non-breaking space, `Café` against `Cafe` |
| All-caps rewrite (`edit`) | the whole name upper-cased |
| Past the token limit (`edit`) | a description changed only after its first 256 tokens (D21): `Modified` at 1.000 |
| Ordinary recycles | a different product in the same category and brand; a different category |

`name` is `required` in the example schema, so swaps blank `brand` and `description`, not `name`.

`data/regression.snapshot.ndjson`: the result lines of `calibrate --full --ndjson --seed 0` on that file under
`config.example.toml`, with `timestamp` removed from each line and nothing else changed. It is a valid run for
`events::read` and therefore for `calibrate-diff`.

### Where the check runs

**In `cargo test`, no model** (`crates/similarity-server/tests/regression_snapshot.rs`, not ignored):

- the snapshot reads with `events::read`;
- its `pairs_sha256` is the SHA-256 of `data/regression.jsonl`, so editing the cases without regenerating fails
  here with "run `just snapshot`";
- `data/regression.jsonl` parses with `parse_full_pairs` under `config.example.toml`'s schema, every pair has a
  `pair_id`, and nothing is dropped as an ignored key;
- `rejudge` of the snapshot under `config.example.toml` is compared with the snapshot by `diff` and `moved`: no
  flip, no movement. This is the gate for weights, floors, thresholds, limits, and judging code.

**In CI, with the model** (`crates/similarity-server/tests/real_snapshot.rs`, `#[ignore]`d): loads
`models/mdbr-leaf-mt` and `config.example.toml`, scores `data/regression.jsonl` with `calibrate`, captures the
events with `similarity_log::testing::capture`, reads them back, and fails with the rendered `calibrate-diff`
report if `diff` has a flip or `moved` is non-empty, or if the fingerprint differs from the snapshot's (a
different model needs a regenerated snapshot, said in the message). Going through the events, rather than
comparing `Calibration`s directly, applies the same rounding to both sides.

With `UPDATE_SNAPSHOT=1` the same test writes the file instead, after printing the diff against the old one and
refusing if `borderline` finds anything. That is `just snapshot`, the counterpart of `just golden`. Writing from a
test keeps the regeneration in Rust and in the code that checks, with no `jq` or script in the pipeline.

### justfile

```just
# Score the curated hard cases with the real model and compare them with the committed snapshot (needs `just fetch-model`)
test-snapshot:
    cargo test -p similarity-server --test real_snapshot -- --ignored

# Regenerate the committed verdict snapshot after an intended change, and show what moved
snapshot:
    UPDATE_SNAPSHOT=1 cargo test -p similarity-server --test real_snapshot -- --ignored --nocapture

# Gate a calibration of real pairs against an earlier one (at work, where the data is)
verdict-gate old pairs config="config.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- calibrate --config {{ config }} {{ pairs }} --full --ndjson {{ flags }} > target/verdict-gate.ndjson
    cargo run --release -q -p similarity-server -- calibrate-diff {{ old }} target/verdict-gate.ndjson --fail-on any-flip --fail-on moved
```

`test-real` gains `test-snapshot` as a prerequisite next to `test-parity`.

### CI

A new job in `.github/workflows/ci.yml`, on every push and pull request like the others (decision 3):

```yaml
  # The real model: parity with PyTorch, and the curated verdicts. The model is cached under the hash of fetch.rs,
  # which pins its revision and checksums; `model fetch` re-verifies a restored cache and downloads what is missing.
  real-model:
    name: real model (parity and verdict snapshot)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: 1.96.0
      - uses: Swatinem/rust-cache@v2
      - run: sudo apt-get update && sudo apt-get install -y protobuf-compiler
      - uses: actions/cache@v4
        with:
          path: models/mdbr-leaf-mt
          key: model-${{ hashFiles('crates/similarity-server/src/fetch.rs') }}
      - run: cargo run -p similarity-server -- model fetch
      - run: cargo test -p similarity-core --test real_parity -- --ignored
      - run: cargo test -p similarity-train --test real_parity -- --ignored
      - run: cargo test -p similarity-server --test real_snapshot -- --ignored
```

`model fetch` runs in the dev profile on purpose: the release profile is fat LTO with one codegen unit and would
add a long second compile for a command that only downloads and hashes. The comment on the `qdrant` job that says
the model tests stay local is rewritten. The snapshot is generated on a developer's machine (macOS) and checked on
Linux; the tolerance has to absorb that difference, and the first CI run is where it is measured (decision 2).

### README quotes

`calibrate`'s quoted reports in README "Choosing thresholds" are preceded by a marker comment naming the command
that produced them, for example `<!-- measured: calibrate data/calibration.sample.jsonl config.example.toml -->`.
An ignored test (`real_snapshot.rs`, run by the same job) finds each marked block, runs that calibration in
process, renders the text report, and checks that every line of the block other than `...` occurs in the report
(decision 7). The D3 and README model tables need four other models and stay a manual measurement.

### Dependencies

None new. SHA-256 is already in the server for `pairs_sha256`; `similarity-log`'s `testing` feature is already a
dev-dependency feature. `actions/cache@v4` is a GitHub action, not a crate, and `cargo deny` does not see it.

## The work, in order

1. **Anchored pairs.** `Pair::anchor`, both parsers, `content_id`, `Offline::compare`, the `anchored` and
   `anchor_similarity` fields on the `pair` event and in `events::read`, the report marker. Example tests and the
   offline-equals-engine property. README "Choosing thresholds" documents the key. Run `PROPTEST_CASES=5000 just
   test` once, since this touches how an anchor is built.
2. **Movement in `calibrate-diff`.** `Movement`, `Moved`, `moved`, `Diff::moved`, `FailOn::Moved`, `--tolerance`,
   repeatable `--fail-on`, the report section, the `moved` events and the `diff` line's count. README "Comparing
   two calibrations".
3. **Borders.** `borderline` and its tests.
4. **The cases and the snapshot.** `data/regression.jsonl`; `tests/real_snapshot.rs` with its update mode; run
   `just snapshot` locally with the pinned model and commit the snapshot; `tests/regression_snapshot.rs` (the
   no-model checks); the `snapshot` and `test-snapshot` recipes; `test-real` depends on `test-snapshot`.
5. **README quotes.** The markers and the ignored check. Re-run `just calibrate` for both sample files first; if a
   quoted line no longer matches, the README was stale and is corrected in this commit, with the figures as
   measured.
6. **CI.** The `real-model` job, the rewritten comment. Read the first run's movement against the macOS-made
   snapshot (the largest similarity difference, printed with `--nocapture`) and set the tolerance from it if 0.001
   is too tight.
7. **The real-data gate.** The `verdict-gate` recipe. Built and run here only on the synthetic sample; its first
   real use is at work.
8. **Docs.** D22, `CLAUDE.md`, README "Development", `docs/feature-suggestions.md` (mark 9 built, say what came out
   differently), `docs/plan-features.md`.

Each step ends with `just ci`; steps 1, 4, 5, and 6 also with `just test-real`.

## Tests

No model, no network (in `cargo test`):

- `offline/pairs.rs`: `a_pair_may_carry_the_record_as_it_was_first_created`;
  `a_pair_without_an_anchor_keeps_the_id_it_had`; `an_anchor_is_reduced_to_the_schema_and_its_dropped_keys_counted`;
  `a_strict_pair_with_any_fifth_key_but_anchor_is_refused`.
- `offline/mod.rs`: `a_field_blanked_and_refilled_is_scored_against_the_anchor`;
  `a_quantity_removed_and_refilled_is_compared_with_the_anchor`; `an_anchored_pair_is_held_to_the_anchor_threshold`.
- `offline/events.rs`: `an_anchored_pair_reads_back_with_its_anchor_similarity`;
  `a_run_without_timestamps_reads_back_the_same`.
- `offline/diff.rs`: `a_similarity_within_the_tolerance_has_not_moved`;
  `a_similarity_beyond_the_tolerance_is_reported_with_both_values`;
  `a_verdict_rejected_for_another_reason_has_moved`; `a_field_that_became_restored_has_moved`;
  `a_quantity_factor_is_compared_almost_exactly`; `two_gates_can_be_asked_for_at_once`;
  `the_moved_events_carry_the_pair_id_and_count_into_the_diff_line`;
  `a_pair_within_the_margin_of_a_threshold_or_floor_is_on_a_border`.
- `properties.rs` (server): `an_anchored_pair_is_judged_as_an_import_followed_by_an_evaluate`.
- `tests/regression_snapshot.rs`: `the_snapshot_was_made_from_the_committed_cases`;
  `every_curated_case_fits_the_example_schema_and_has_an_id`;
  `the_snapshot_judged_again_under_the_example_config_changes_nothing`; `the_snapshot_holds_no_timestamp`.

Real model (`#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-snapshot`"]`), no
Qdrant:

- `tests/real_snapshot.rs`: `the_pinned_model_gives_the_committed_verdicts_and_scores`;
  `the_calibration_reports_quoted_in_the_readme_are_what_calibrate_prints`.

Only at work, on real pairs: `just verdict-gate` against a kept `--ndjson` run of the real holdout. Nothing here
can show it is useful there; it is the same code as the synthetic gate, run on other data.

## Documentation to update

- **`docs/decisions.md`: D22, "Verdicts on hard cases are pinned, and checked without the model where they can
  be".** Why the snapshot is a run file (no second format, D17), why timestamps are dropped, why the check is by
  tolerance and the tolerance's measured basis, why a case on a border is refused, why anchored pairs exist and
  that they are built as an import is (D8, D20), that the snapshot is not a calibration (D3), and what the
  no-model rejudge covers versus what needs the model. D17 gains a sentence on the `anchored`,
  `anchor_similarity`, and `moved` fields.
- **`CLAUDE.md`, "Things that are easy to get wrong":** `data/regression.jsonl` is synthetic and stays that way;
  its snapshot is regenerated with `just snapshot` only for an intended change, and the regeneration's printed
  diff belongs in the commit message; no threshold is ever read from it; a change to `config.example.toml` that
  moves a curated verdict fails `cargo test` until the snapshot is regenerated, which needs the model. "Testing"
  gains `just test-snapshot` beside `just test-real`. "Commands" gains `snapshot`, `test-snapshot`,
  `verdict-gate`.
- **`README.md`:** "Choosing thresholds" documents `anchor` in a pair; "Comparing two calibrations" documents
  `--tolerance`, `--fail-on moved`, repeated `--fail-on`, and the `moved` events; the event table gains the new
  fields; "Development" gains the two recipes and says CI now runs the real model; the project layout lists the
  two new data files. Every example re-run before it is edited.
- **`docs/feature-suggestions.md`** marks 9 built with what differed; **`docs/plan-features.md`** updates its row.

## Dependencies on other features

None is required. What this assumes or offers:

- **Feature 19 (reason codes on the wire):** independent. Both rely on `Reason::code` and `subject` being stable,
  which D17 already requires.
- **Feature 4 (promotion gate):** would call `calibrate-diff --fail-on any-flip --fail-on moved` as built here.
- **Feature 11 (drift calibration):** anchored pairs are a one-step form of a history. Feature 11 replays whole
  sequences through the engine and does not need them, but may reuse the `anchored`/`anchor_similarity` fields
  for its events.
- **Feature 15 (mutation testing):** the no-model snapshot check is a cheap extra killer of mutants in
  `score.rs`, `quantity.rs`, and `offline/stats.rs`.

## Decisions for the owner

1. **Pairs may carry an `anchor`.** Without it the two-step swap and the quantity refill (both named in the task)
   cannot be in the snapshot, and they are the cases D8 exists for. The rule itself is already property tested
   with a fake embedder; what the snapshot adds is how the real model scores them. *Default: yes, add `anchor`.*
2. **The score tolerance.** The suggestion says 0.005. The parity test holds embeddings to cosine 0.99999 of
   PyTorch, so platform noise should be around 1e-5 per similarity. *Default: 0.001, confirmed against the first
   Linux run of a macOS-made snapshot.*
3. **When the job runs.** *Default: every push and pull request, like the other jobs.* The cache makes the model a
   one-time download per pinned revision; the cost is one more test build. Path filters were considered and
   rejected: a scoring change anywhere would need listing.
4. **Whether CI also runs the trainer's parity test.** It is heavier than the embedding one. *Default: run both;
   drop the trainer's from CI if the job passes ten minutes.*
5. **A case on a border.** *Default: `just snapshot` refuses to write while any pair is within the tolerance of a
   threshold, floor, or limit; the case is changed instead.*
6. **The committed snapshot drops `timestamp`.** *Default: yes*, so that a regeneration's git diff shows only what
   moved. The alternative, committing raw output, needs no code but makes every snapshot change unreadable.
7. **Checking the README's quoted calibration reports.** *Default: yes, with marker comments,* in the same job.
   It turns "every number in `README.md` was measured" into a check for the figures most likely to go stale.
8. **Labels for same-brand variant changes.** *Default: `unsure`,* so that the snapshot pins what happens to them
   without claiming what should.
9. **Where the snapshot is regenerated.** *Default: locally, with `just snapshot`; CI only checks.* A CI job that
   regenerated and committed would make an unreviewed change to a pinned file.

## Size

**Medium.** No new dependency and no new format, and most of the machinery exists (`events`, `diff`, `rejudge`,
`capture`). The work is two additive extensions (anchored pairs, movement), one new CI job, about 30 hand-written
cases, and two test files. The anchored pair is the largest single piece, because it touches the parser, the
offline scorer, the events, and needs the engine-equivalence property. Without decision 1 it is small.
