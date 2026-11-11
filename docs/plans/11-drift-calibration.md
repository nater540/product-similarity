# Plan: drift-threshold calibration from edit histories (feature 11)

A plan for `record-similarity calibrate-drift`, written against the code on `initial-implementation` at `545ef20`.
Nothing here is built. Read `CLAUDE.md` first, then D7, D8, D16, D17, and D20 in `docs/decisions.md`.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [What can only be validated at work](#what-can-only-be-validated-at-work)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

`calibrate` measures pairs: one edit against the version before it. Two settings cannot be read from pairs,
because they are about a record's whole life:

1. **`policy.anchor_threshold`**, the drift check of D8. It rejects a revision whose similarity to the record's
   anchor (the first value of every field it has had) is too low, which is what stops a recycler walking a
   product somewhere else in steps that each pass. It must sit below how far a legitimate product drifts over its
   lifetime, and that is a property of histories.
2. **Whether quantities need a drift check at all.** A `reject_above_factor` holds each revision against the one
   before it (D16, "What this does not do"), so a weight can go 3.5 g, 7 g, 14 g, 28 g in steps of 2. Whether
   real products do that, legitimately or not, is also a property of histories.

The command replays each history in `sequences.jsonl` (data request, section 3) through the real engine on a
throwaway in-memory store, measures every step against the previous version and against the anchor, judges each
step under the configured policy, and reports what a person needs to set `anchor_threshold` and to decide on a
quantity drift check. Like `calibrate`, it prints a report by default and writes flat NDJSON events with
`--ndjson`.

## What already exists

Verified in the code as it is now:

- **The engine and its anchor rules.** `Engine::evaluate` and the private `Engine::revise` in
  `crates/similarity-server/src/engine.rs`. An accepted `ENFORCE` write extends the anchor with
  `score::extend_anchor` and `quantity::first_values`; `FORCE` resets both; a rejection and `DRY_RUN` write
  nothing. Re-submitting what is stored writes nothing (the `identical` check). `EvaluateOutput` carries
  `similarity`, `fields: Vec<FieldScore>`, `anchor_similarity`, `quantities: Vec<QuantityScore>`,
  `measure_changed`, `stored`, `revision`, and the reasons as **strings**.
- **Seeding a history.** `Engine::import` with `ImportInput { record, anchor: Option<Original> }` creates a record
  at revision 1 whose anchor is the original version extended with the current one (D20). `Engine::store()`
  exists but is `#[cfg(test)] pub(crate)`.
- **A synchronous model behind the async trait.** `worker::DirectEmbedder(pub Embedder)` implements
  `TextEmbedder`, as `import` uses it: no worker thread or queue is needed offline.
- **Scoring.** In `crates/similarity-core/src/score.rs`: `Policy { threshold, anchor_threshold,
  reject_on_measure_change }`, `Reason::AnchorDrift` (code `anchor_drift`), `Reason::code`, `subject`, `value`,
  `limit`, `judge`, `judge_quantities`, `compare_quantities`, `weighted_similarity` (the anchor similarity: a
  weighted mean over the fields both composites have), and `Evaluation::rejudge`, which re-applies weights,
  floors, thresholds, and quantity limits to measurements. D19's property says judging again equals evaluating
  afresh.
- **Quantities.** `similarity_core::quantity::compare(field, baseline, anchor, candidate)` measures against the
  anchor's value only when the baseline has none (`Restored`). Nothing compares a present quantity with its first
  value. `StoredRecord::anchor_quantities` holds those first values.
- **The offline toolkit** in `crates/similarity-server/src/offline/`: `RecordFields` (text and quantities in one
  JSON object), the private `pairs::project` (keeps what the schema compares, counts the rest by key and
  `Ignored` reason), the private `pairs::data_lines` (skips blank and `#` lines), `stats` (every number the report
  and the events state, including the private owner-resampling `bootstrap` with `RESAMPLES = 1000` and the
  `Rate`/`Interval` types), `events::emit_schema` and `six`, and `report`. `calibrate` is the model for the new
  command: `CalibrateArgs` in `main.rs`, the `calibrate` recipe in the `justfile`.
- **A synthetic slow walk.** The engine test `small_steps_walk_a_record_anywhere_unless_drift_is_enforced` walks
  "blueberry gummies" into "sour diesel resin vape cartridge" in steps the pair threshold accepts.

Nothing reads `sequences.jsonl`. There is no sample history file in `data/`.

## What the suggestion got wrong or left open

1. **Replaying in `ENFORCE` mode measures the wrong thing after the first rejection.** Once a step is rejected the
   engine keeps the old baseline, so every later step is compared with a version the product had already left. A
   legitimate history that the policy rejects once (which is the case most worth seeing) would show a made-up
   tail, and its real lifetime drift would never be measured. The history happened: every version in it was
   saved. The replay should therefore **write every version** (measure), and **judge each step afterwards** under
   the configured policy, the split D17 made for pairs. Up to and including the first rejection this gives
   exactly what `ENFORCE` would, because every earlier step was accepted under both; a property test states it.
   Steps after the first rejection are still reported, marked as what would have happened had the edit been let
   through.
2. **The engine's response is not enough on its own.** It has no per-field anchor similarities and no
   anchor-relative quantity factors, and its reasons are strings. The replay rebuilds an `Evaluation` from the
   response (all its parts are public), reads the stored anchor before each step, and needs two small additions
   to the core (below).
3. **"`Engine::import` with `anchor` is the way to seed each history"** (`docs/plan-features.md`) is half right.
   A history that starts at the product's creation needs no anchor: its first version is created like any record.
   `import` with `anchor` matters for a history that starts later than creation while the original is known,
   which is exactly the situation a production re-index creates (D20). The replay seeds every history through
   `Engine::import`, with the original when the line carries one, so both cases go through one path.
4. **An anchor threshold is not placed like a pair threshold.** Like a quantity limit (D17), the drift check is an
   additional way to reject. Most recycles are one large step that the pair threshold or the `name` floor already
   rejects, so there is usually no gap between "lowest legitimate drift" and "highest recycled drift" to put a
   threshold in. The suggestion has to come from the legitimate histories (it must block none of them) and from
   the recycles **the pair rules miss**. With few or no such recycles in the data, the honest output is a ceiling
   ("anything below 0.71 blocks no legitimate history here") and no suggested value.
5. **The data request's history format cannot say where a recycle happened.** `sequences.jsonl` labels a whole
   history `edit` or `recycle`, but a recycled product was legitimate until some version. Without that index the
   tool can say whether a recycled history was caught, not whether it was caught at the right step or which
   earlier legitimate steps were wrongly rejected. The format also leaves open whether `fields` holds quantities as
   objects (it should, as in `pairs.full.jsonl`), whether the first version is the product's creation, and how
   sure the label is. See "Format" below for the optional keys to add.
6. **The shipped `anchor_threshold = 0.85` is not a measurement.** `config.example.toml` enforces it, and no
   number in the config comments, `README.md`, or `docs/decisions.md` says where it came from, unlike
   `threshold` (0.90, from the sample pairs) and the `name` floor. That is the situation `CLAUDE.md` rules out for
   every other threshold. This feature is how it gets measured; it must not be "fixed" from the synthetic sample.
7. **`Evaluation::rejudge` cannot re-weigh an anchor similarity.** It is a scalar computed under the weights in
   force when it was measured, and `rejudge` passes it through unchanged. Harmless today, because calibration
   pairs have no anchor, but a drift run judged again under other weights would silently keep the old weights for
   drift. Recording per-field anchor similarities (below) keeps that possible; re-judging drift runs is left for
   later.
8. **Quantities have no drift measurement anywhere.** Confirmed: `judge_quantities` sees only the step's
   comparison. Reporting each quantity's largest factor against its first value, per history, is new.

## Design

### Format: `sequences.jsonl`

The data request's shape, read leniently like `pairs.full.jsonl`: each version's `fields` is reduced to what the
schema compares by the same projection, and every key left out is counted by key and reason.

```json
{"product": "h0007", "owner": "b12", "label": "recycle", "recycled_at": 4,
 "label_source": "moderator_revert", "label_confidence": "high",
 "original": {"name": "Blueberry Gummies 100mg"},
 "versions": [{"at": "2025-11-02T09:00:00Z",
               "fields": {"name": "...", "category": "Edibles > Gummies",
                          "weight_per_item": {"value": 40, "unit": "g"}}}]}
```

| Key | Required | Meaning |
|---|---|---|
| `product` | yes | The history's id (a pseudonym). Unique in the file; two lines with one id are refused with both line numbers. |
| `owner` | no | The brand, as a pseudonym. Used as the tenant in the replay and as the cluster for intervals. |
| `label` | yes | `edit` or `recycle` take part in counts; anything else (`unsure`) is replayed, shown, and never counted. |
| `versions` | yes | At least two, in order. `at` is kept as text and never parsed (no date crate needed); file order is the order. |
| `recycled_at` | no | For `recycle`: the index in `versions` of the first version that is no longer the product (at least 1). |
| `label_source`, `label_confidence` | no | As in `pairs.full.jsonl`. `--min-confidence` filters labeled histories like `calibrate --full`. |
| `original` | no | The product as created, when `versions[0]` is not its creation. Becomes the anchor through `Engine::import`. |
| `truncated` | no | `true` when `versions[0]` is not the creation and no `original` is known. Reported apart: drift is understated. |

Unknown top-level keys are refused (`deny_unknown_fields`), as in the full pair format, so a misspelled
`recyled_at` is an error and not a silently unlabeled history. A version whose `fields` do not fit the schema
(a blank `name`, a malformed quantity) is an error naming the product and the version index, as a bad pair is for
`calibrate` (owner decision 3).

### Replay

New module `crates/similarity-server/src/offline/drift.rs` (parsing in `offline/histories.rs`):

```rust
/// One history as the file holds it, after projection onto the schema.
pub struct History {
  pub id: String,
  pub owner: Option<String>,
  pub label: Option<Label>,
  pub label_text: String,
  pub label_source: Option<String>,
  pub label_confidence: Option<Confidence>,
  pub recycled_at: Option<usize>,
  pub truncated: bool,
  pub original: Option<RecordFields>,
  pub versions: Vec<Version>,
}

pub struct Version { pub at: Option<String>, pub fields: RecordFields }

pub struct Histories { pub histories: Vec<History>, pub input: Input }

/// One step: version `step` (1-based) judged as a revision of version `step - 1`.
pub struct Step {
  pub step: usize,
  pub at: Option<String>,
  /// Measured with every limit off, then judged under the configured schema and policy.
  pub evaluation: Evaluation,
  /// The same measurement judged with `anchor_threshold` removed: the pair rules alone.
  pub pair_rules_accepted: bool,
  /// Per field, the cosine with the anchor's block, `None` where either side has no value.
  pub anchor_fields: Vec<(String, Option<f32>)>,
  /// Per quantity, the comparison with its first value (`quantity::against_first`).
  pub anchor_quantities: Vec<QuantityScore>,
  /// Whether the engine wrote this version (false only when it was identical to the last).
  pub written: bool,
}

pub struct ReplayedHistory { pub history: History, pub steps: Vec<Step> }

pub struct DriftRun { pub histories: Vec<ReplayedHistory>, pub input: Input }

pub async fn replay<E: TextEmbedder>(
  embedder: E, schema: &Schema, policy: &Policy, histories: Histories,
) -> anyhow::Result<DriftRun>;
```

How `replay` works:

1. **A measuring engine.** `Engine::new(MemoryStore, embedder, measuring_schema, measuring_policy)`, where the
   measuring schema is the configured one with every `reject_below`, `reject_above_factor`, and
   `reject_incomparable` cleared (same fields, weights, and order, so the same layout id), and the measuring
   policy is `Policy::new(-1.0, None)` with `reject_on_measure_change` off. Nothing can be rejected, so every
   version is written by the ordinary `ENFORCE` path with the real anchor rules. One store holds every history;
   ids are unique, so histories cannot see each other. Nothing is written anywhere else.
2. **Seed.** `Engine::import` with `versions[0]` as the record and `original` (if any) as the anchor, the owner as
   tenant.
3. **Each later version.** Read the stored record first (its `anchor` and `anchor_quantities` are what this step
   is held to), then `Engine::evaluate` in `ENFORCE`. From the response, rebuild
   `Evaluation { comparison: Comparison { similarity, fields }, anchor_similarity, reasons: vec![], quantities,
   measure_changed }` and call `rejudge(configured_schema, configured_policy)`. Judge a copy again with
   `anchor_threshold: None` for `pair_rules_accepted`. Compute `anchor_fields` with the new
   `score::field_similarities(schema, dimension, &stored.anchor, &response.vector)`, and `anchor_quantities` with
   `quantity::against_first` for each schema quantity.
4. A rejection, an error from the engine, or an embedder failure ends the run with the product and step named.

The two core additions, both in `similarity-core` and both pure:

- `score::field_similarities(schema, dimension, left, right) -> Result<Vec<(String, Option<f32>)>>`, the
  per-field cosines `weighted_similarity` already computes inside its loop. `weighted_similarity` is rewritten on
  top of it with no change of result.
- `quantity::against_first(field, first, candidate) -> Option<QuantityScore>`: `compare(field, first, first,
  candidate)` under a name that says what it is for. `Added` when the anchor has never had the quantity (no
  factor yet), `Removed` when the candidate lacks it.

One engine change: `Engine::store()` loses `#[cfg(test)]` and stays `pub(crate)`, so the replay, which is in the
same crate, can read the anchor before a step. No new engine behavior and no new write path.

### Statistics (`offline::stats`)

Every number in the report and the events comes from here, per `CLAUDE.md`. `stats.rs` is 787 lines already, so
it becomes `offline/stats/mod.rs` with the drift functions in `offline/stats/drift.rs`; `bootstrap`, `Rate`, and
`Interval` become `pub(super)`-visible to it and are reused, not copied.

Per history (`HistorySummary`):

- `steps`, `min_anchor_similarity` and the step it occurred at, `final_anchor_similarity`;
- `first_rejected` (step and reason codes) under the configured policy, `first_rejected_by_pair_rules`, and
  `first_anchor_drift` (the first step whose anchor similarity is below `anchor_threshold`, if one is set);
- for a recycle with `recycled_at`: `caught` is `at_recycle` (first rejection at `recycled_at`), `later` (after
  it), `early` (before it: a legitimate step was rejected), or `never`; without `recycled_at` it is `caught` or
  `never`. For drift, the recycled part is the steps from `recycled_at` on, and the legitimate part the steps
  before it;
- per quantity (`HistoryQuantity`): the largest step factor, the largest factor against the first value, and
  whether it ever became incomparable with, or crossed zero against, its first value.

Over the run (`DriftStats`), counting labeled histories only, and leaving `truncated` ones out of the suggestion:

- **Ranges of lifetime drift.** The lowest, fifth-percentile, and median `min_anchor_similarity` among `edit`
  histories, and the same over the recycled part of `recycle` histories. Sliced by history length (2 to 4, 5 to
  9, 10 to 19, 20 or more steps), because drift grows with the number of edits and the data request asks only for
  products with at least five.
- **Blocked legitimate histories.** Of the `edit` histories, how many the configured policy rejects at some step,
  how many the pair rules alone do, and how many the drift check alone does. Each is a `Rate` with a 95% interval
  from resampling owners (D17: one brand's histories are not independent).
- **Recycles.** Counts by `caught`, and how many were caught by the drift check and by nothing else.
- **`AnchorSuggestion`**, placed the way D17 places a quantity limit:
  - `legit_low`: the lowest `min_anchor_similarity` of any `edit` history (and of the legitimate part of any
    `recycle` history with `recycled_at`). A drift threshold must be below it.
  - `uncaught`: the recycled histories whose recycled part the pair rules never reject, with the lowest anchor
    similarity each reaches there. These are the only recycles a drift threshold adds anything for.
  - `next_below`: the highest of those that is below `legit_low`.
  - `anchor_threshold`: the midpoint of `legit_low` and `next_below`, with `margin` (half the gap) and
    `recycles_caught` (how many `uncaught` histories it rejects). `None` when there is no `next_below`, in which
    case the report states `legit_low` as a ceiling and says that no recycle in the data needs the check.
  - `above_threshold`: set when the suggestion is above `policy.threshold`. At step 1 the anchor is the previous
    version, so such a drift threshold would be stricter than the pair threshold on every first edit; that is
    worth a line, not a silent suggestion.
- **`QuantityDrift`**, per quantity, answering whether quantities need a drift check:
  - among `edit` histories: the largest step factor and the largest factor against the first value;
  - among recycled parts: the same;
  - `walked`: recycled histories whose factor against the first value exceeds the largest that any `edit` history
    reaches, while every one of their steps stays at or below the largest step factor any `edit` history shows.
    These are the recycles a per-step limit placed from edits cannot catch and a drift limit could. A non-zero
    count is the evidence for building a quantity drift check; zero, on enough histories, is the evidence against.
  - counts of incomparable-against-first and zero-crossing-against-first, per label.
  It suggests no value: there is no configuration key for it (owner decision 5).

### Command and report

```text
record-similarity calibrate-drift [--config config.toml] [--min-confidence medium] [--ndjson] [--seed 0] <histories>
```

`CalibrateDriftArgs` in `main.rs` beside `CalibrateArgs`: it loads the embedder with `load_embedder`, takes
`config.schema()` and `config.policy()`, wraps the embedder in `DirectEmbedder`, and runs `replay` on a
current-thread Tokio runtime, as `import` does. `justfile`:

```just
# Replay whole edit histories to choose `anchor_threshold`: `just calibrate-drift sequences.jsonl config.toml`
calibrate-drift histories="data/drift.sample.jsonl" config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- calibrate-drift --config {{ config }} {{ histories }} {{ flags }}
```

The text report, rendered in `offline/report.rs` from `DriftStats` to a locked stdout with `writeln!`, in the
shape of the calibration report:

```text
model     f46e48ee0ea2dd02 (256 dimensions per field, up to 256 tokens)
histories 14 (8 edit, 5 recycle, 1 other), 61 steps

lowest similarity to the anchor     edit histories            recycled part
all                                 0.712 (p5 0.731, median 0.884)   0.031 to 0.802 (5)
5 to 9 steps                        ...

configured policy (threshold 0.900, anchor_threshold 0.850, floors and quantity limits):
  legitimate histories blocked: 2 of 8 (by the pair rules 1, by drift alone 1)
  recycles caught at the recycled version 3, later 1, never 1; caught by drift alone 1

anchor_threshold: 0.512 is below every legitimate history (lowest 0.712) and catches 1 of 1 recycle the pair rules miss (margin 0.200)

quantity            edit histories (step / vs first)   recycled part (step / vs first)   walked
weight_per_item     1.01 / 1.14                        2.00 / 8.00                         1
...

histories the configured policy blocks or misses, worst first:
h0003 edit     step 4  anchor_drift  0.842 < 0.850   "..." 
```

Numbers in the sketch are placeholders for layout; the README quotes only what `just calibrate-drift` prints.

### Events (`--ndjson`)

Through `similarity-log` as `target: similarity_log::RESULT` events (D15, D17): flat, scalars only, similarities
through `six`, optional values left out rather than `null`. All names are new, so nothing `calibrate` or
`calibrate-diff` reads changes. Once shipped they are an interface: add fields, never rename.

| Event | Fields |
|---|---|
| `drift_run` (first) | `fingerprint`, `dimension`, `max_tokens`, `threshold`, `anchor_threshold`, `reject_on_measure_change`, `histories_sha256`, `histories`, `min_confidence`, `seed` |
| `schema_field`, `schema_quantity` | as `calibrate` writes them (`events::emit_schema`, reused) |
| `history` | `history_id`, `label`, `owner`, `label_source`, `label_confidence`, `recycled_at`, `truncated`, `seeded_from_original`, `versions`, `first_at`, `last_at`, `min_anchor_similarity`, `min_anchor_step`, `final_anchor_similarity`, `first_rejected_step`, `first_rejected_by_pair_rules_step`, `first_anchor_drift_step`, `caught` |
| `step` | `history_id`, `step`, `at`, `similarity`, `anchor_similarity`, `accepted`, `accepted_by_pair_rules`, `written`, `recycled`, `measure_changed` |
| `step_field` | `history_id`, `step`, `field`, `change`, `similarity`, `anchor_similarity`, `weight`, `below_floor` |
| `step_quantity` | `history_id`, `step`, `quantity`, `change`, `factor`, `zero_crossing`, `incomparable`, `previous_unit`, `unit`, `beyond_limit`, `first_change`, `first_factor`, `first_zero_crossing`, `first_incomparable`, `first_unit` |
| `step_reason` | `history_id`, `step`, `code` (`Reason::code`), `subject`, `value`, `limit` |
| `history_quantity` | `history_id`, `quantity`, `max_step_factor`, `max_first_factor`, `first_incomparable`, `first_zero_crossing` |
| `drift_range` | `label`, `slice`, `count`, `lowest`, `p5`, `median` |
| `drift_blocked` | `what` (`policy`, `pair_rules`, `drift_only`), `errors`, `of`, `rate`, `low`, `high`, `owners` |
| `drift_suggestion` | `anchor_threshold`, `margin`, `legit_low`, `next_below`, `recycles_caught`, `uncaught`, `above_threshold` |
| `drift_quantity` | `quantity`, `label`, `max_step_factor`, `max_first_factor`, `walked`, `first_incomparable`, `first_zero_crossings` |
| `ignored_key` | as `calibrate` writes it |
| `drift_summary` (last) | `histories`, `steps`, `blocked`, `caught`, `missed` |

Per-field anchor similarities (`step_field.anchor_similarity`) are what make a later re-judging of a drift run
under other weights possible, and what would show whether one field (the name, most likely) carries the walk.

### Sample data

`data/drift.sample.jsonl`, synthetic and saying so in a `#` header line, hand-written like
`data/calibration.sample.jsonl`. About fifteen histories across a few invented owners:

- legitimate: steady rewording of a description over eight versions; a name gaining and losing a potency suffix;
  a description blanked and refilled with the same text; batch potency updates (21.3% to 22.8% and back); a unit
  rewrite (`1 g` to `1000 mg`); a category refined from `Edibles` to `Edibles > Gummies`;
- recycled in one step (caught by the pair rules at `recycled_at`);
- recycled by a slow walk, the engine test's gummies-to-cartridge sequence, with `recycled_at` at the first
  version that is no longer gummies;
- a quantity walk: weight 3.5 g, 7 g, 14 g, 28 g with unchanged text, labeled `recycle`;
- a remove-and-refill swap of a description, labeled `recycle`;
- one history with `original` (a truncated history seeded from its creation) and one `unsure`.

Its output in the README is an example of the report, like the calibration sample's, and is not read from.

## The work, in order

Each step leaves `just ci` green; tests are written first where practical.

1. **Core measurements.** `score::field_similarities` with `weighted_similarity` rebuilt on it, and
   `quantity::against_first`. Tests and one property (below). Run `PROPTEST_CASES=5000 just test`, since the
   scoring is touched. `just test-real` is not needed: nothing an embedding or a verdict depends on changes, and
   the property proves `weighted_similarity` unchanged.
2. **History format.** `offline/histories.rs`: `History`, `Version`, `Histories`, `parse_histories(jsonl, schema,
   min_confidence)`. `pairs::project` and `pairs::data_lines` become `pub(super)` and are reused; `Input` gains
   nothing (its `ignored`, `sha256`, and `skipped_confidence` fit as they are). Tests on literal JSON.
3. **Replay.** `Engine::store()` made available outside tests; `offline/drift.rs` with `replay`, the measuring
   schema and policy, and the rebuilt-and-rejudged `Evaluation` per step. Tests with `WordBagEmbedder` and
   `MemoryStore`, and the property that ties the replay to an enforcing engine.
4. **Statistics.** `stats.rs` becomes `stats/mod.rs`; `stats/drift.rs` with `HistorySummary`, `DriftStats`,
   `AnchorSuggestion`, `QuantityDrift`, reusing `bootstrap`. Tests on hand-built `DriftRun`s, no embedder.
5. **Report, events, command.** `report::render_drift`, `events::emit_drift`, `CalibrateDriftArgs`, the `justfile`
   recipe. A test that the report and the events state the same numbers.
6. **Sample and docs.** `data/drift.sample.jsonl`, `just calibrate-drift` run with the real model, its output
   pasted into the README, the decision record, the data request, `CLAUDE.md`. Add one ignored real-model test.
   Then `just ci` and `just test-real`.

## Tests

Names are sentences. All run under `cargo test` with no model, no network, and no Qdrant, except the last.

`similarity-core`:

- `score::tests::field_similarities_name_every_field_and_skip_those_one_side_lacks`
- `properties::the_weighted_mean_of_the_field_similarities_is_the_weighted_similarity` (proptest)
- `quantity::tests::a_quantity_is_measured_against_its_first_value_however_it_got_there`
- `quantity::tests::a_quantity_the_anchor_has_never_held_has_no_factor_against_its_first_value`

`similarity-server`, `offline::histories`:

- `a_history_keeps_what_the_schema_compares_and_counts_the_rest_by_key`
- `a_history_with_fewer_than_two_versions_is_refused_by_line`
- `two_histories_with_one_product_id_are_refused_with_both_line_numbers`
- `a_recycled_at_outside_the_versions_or_on_an_edit_history_is_refused`
- `a_misspelled_top_level_key_is_an_error_not_an_unlabeled_history`
- `any_label_other_than_edit_or_recycle_is_replayed_and_left_out_of_every_count`
- `a_version_that_does_not_fit_the_schema_is_reported_with_its_product_and_index`

`offline::drift` (with `WordBagEmbedder` and `MemoryStore`):

- `every_version_of_a_history_is_measured_even_after_the_policy_would_have_rejected_one`
- `a_slow_walk_is_caught_by_drift_and_not_by_the_pair_rules`
- `a_field_blanked_and_refilled_in_a_history_is_scored_against_its_first_value`
- `a_quantity_walked_in_small_steps_reaches_a_large_factor_against_its_first_value`
- `a_history_with_its_original_measures_drift_from_the_original`
- `an_unchanged_version_is_a_step_that_writes_nothing`
- `the_pair_rules_alone_are_the_configured_policy_without_its_anchor_threshold`
- `properties::the_first_rejection_found_by_replay_is_the_one_an_enforcing_engine_gives` (proptest over generated
  histories of field and quantity edits: the first step the replay judges rejected is the first step an engine
  built with the configured schema and policy rejects in `ENFORCE`, and every step before it has identical
  similarity and anchor similarity in both)

`offline::stats::drift`:

- `the_suggested_anchor_threshold_sits_between_the_lowest_legitimate_drift_and_the_next_uncaught_recycle`
- `no_anchor_threshold_is_suggested_when_the_pair_rules_catch_every_recycle`
- `a_suggestion_above_the_pair_threshold_is_flagged`
- `truncated_histories_are_reported_and_left_out_of_the_suggestion`
- `blocked_legitimate_histories_carry_an_interval_only_when_there_are_two_owners`
- `a_recycle_caught_before_its_recycled_version_counts_as_a_blocked_legitimate_step`
- `a_quantity_walk_is_counted_only_when_no_single_step_exceeds_what_edits_show`

`offline::report`, `offline::events`, `main`:

- `the_drift_report_and_the_drift_events_state_the_same_numbers`
- `drift_events_are_flat_and_every_one_after_the_first_names_its_history`
- `calibrate_drift_parses_its_flags`

Needs the real model (no Qdrant), `#[ignore = "needs ./models/mdbr-leaf-mt; run with just test-real"]`, in
`crates/similarity-server/tests/real_e2e.rs`:

- `the_sample_histories_block_no_legitimate_product_and_catch_the_slow_walk_by_drift`. It reads
  `data/drift.sample.jsonl` with `config.example.toml`. Its `just test-real` recipe also starts Qdrant for the
  other two tests; this one does not need it.

## Documentation to update

- **`docs/decisions.md`**: a new record (the next free number; D22 when this was written, but other plans may
  claim it), "Drift is calibrated from whole histories, measured first and judged after". It holds: why the replay
  writes every version (item 1 above) and the property that ties it to `ENFORCE`; why the anchor threshold is
  placed from legitimate histories and uncaught recycles, not from a gap; that the shipped 0.85 was not measured;
  what `QuantityDrift.walked` means and what would justify a quantity drift check; that `at` is never parsed.
  D8 gains one sentence pointing to it. D16's "What this does not do" bullet on quantity drift gains "measured by
  `calibrate-drift`". D17 gains a line that drift events are a separate family with their own names.
- **`README.md`**: in "Choosing thresholds", replace the one-line `policy.anchor_threshold` bullet with a
  subsection, "Choosing the drift threshold", showing `just calibrate-drift` on the sample (run and pasted, not
  written by hand) and how to read it; the event table gains the drift events; "Limitations", "No drift check for
  quantities", says the drift is now measured by `calibrate-drift` and still not enforced; the commands list gains
  the recipe.
- **`config.example.toml`**: the `anchor_threshold` comment says the value has not been measured from histories
  and is to be set from `calibrate-drift` on real ones. **The value does not change** (owner decision 4).
- **`CLAUDE.md`**: "Thresholds are measurements" names `just calibrate-drift` for `anchor_threshold`; "Every
  number in the text report and in the events comes from `offline::stats`" covers the drift report; the commands
  list gains `just calibrate-drift`; "Calibration events are flat" covers the drift events too.
- **`docs/data-request-prompt.md`**, section 3: the optional keys `recycled_at`, `label_source`,
  `label_confidence`, `original`, and `truncated`; `fields` in the same shape as `before` in file 1, quantities as
  objects; every version must have a non-blank `name`; the history should start at the product's creation; and a
  line under "Before you finish" to run `record-similarity calibrate-drift --config config.example.toml
  sequences.jsonl` and paste the report.
- **`docs/feature-suggestions.md`** and **`docs/plan-features.md`**: mark 11 built, and correct the "seed each
  history" note (item 3).

## Dependencies on other features

None. It assumes only what exists: `Reason::code`, `RecordFields`, `pairs::project`, `stats::bootstrap`,
`Engine::import`, `DirectEmbedder`, and `events::emit_schema`. Where other planned features touch it:

- **19 (reason codes on the wire)** may change how `Reason` is mapped in `grpc.rs`. The replay uses
  `Reason::code`, `subject`, `value`, and `limit` only, and assumes those keep their meaning.
- **2 (run registry)** would keep `calibrate-drift --ndjson` output with other runs. Interface assumed: it stores
  any result stream as written, keyed by its first event (`drift_run` here, as `calibration` for `calibrate`).
- **9 (pinned snapshot)** could add `data/drift.sample.jsonl` to the snapshot. Interface assumed: a committed
  `--ndjson` run compared against a fresh one; that would need a `calibrate-drift` counterpart of
  `calibrate-diff`, which this plan does not build.

## What can only be validated at work

Everything numerical. Here: the format, the replay mechanics, the property that ties it to `ENFORCE`, and a report
on synthetic histories. At work, on the real `sequences.jsonl`:

- whether 0.85 blocks legitimate products, and what `legit_low` actually is;
- whether any real recycle gets past the pair rules, which decides whether the drift check is worth enforcing;
- whether real quantities walk (`walked`), which decides owner decision 5;
- whether drift grows with history length enough that 200 histories with at least five edits are a fair sample of
  a product's whole life;
- whether the work database has `versions[0]` at creation, or needs `original` or `truncated`.

None of those numbers may enter `config.example.toml` from the synthetic sample.

## Decisions for the owner

1. **Measure every step, then judge** (as above), or replay literally in `ENFORCE`. Recommended: measure then
   judge. It gives the same first rejection and also the real lifetime drift.
2. **Extend the history format** with optional `recycled_at`, `label_source`, `label_confidence`, `original`, and
   `truncated`, and update the data request. Recommended: yes, all optional, so a file in the current shape still
   reads.
3. **A version that does not fit the schema**: fail the run naming it (as `calibrate` does for a bad pair), or skip
   that history and count it. Recommended: fail, so a bad export is fixed rather than quietly thinned.
4. **The shipped `anchor_threshold = 0.85`**: keep it and say in the comment that it is unmeasured, or comment it
   out (report-only) until real histories exist. Recommended: keep the value and fix the comment. Changing it in
   either direction is itself a threshold change with no measurement behind it.
5. **A quantity drift check** (`reject_above_first_factor` per quantity): build it now, or only if real histories
   show `walked` above zero. Recommended: measure only, build it if the data asks.
6. **Where `anchor_threshold` lives**: one policy value as now, or per-field drift floors (a `name` floor against
   the anchor). Recommended: keep the one value; the per-field drift lines in the events will show whether a
   field floor is needed.
7. **Re-judging a drift run under another config** (a drift `--config-b`): later, not now. The events carry what
   it needs.
8. **The command's name**: `calibrate-drift` (the suggestion's) or a `--histories` mode of `calibrate`.
   Recommended: `calibrate-drift`; the input, the statistics, and the events are all different.

## Size

**Medium, at the upper end.** No new dependency (`cargo deny` is unaffected; `at` is kept as text so no date
crate is needed), no change to the server's behavior, the proto, or the store, and two small pure functions in the
core. The bulk is in the offline module: a parser, the replay, a statistics module, a report, about fifteen event
kinds, and a hand-written sample, in the shape `calibrate` already established. About three to four days, most of
it statistics and their tests. Its value arrives only with the real `sequences.jsonl`.
