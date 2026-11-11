# Plan: reviewer feedback capture (feature 17)

A plan for feature 17 of `docs/feature-suggestions.md`, written against the code on `initial-implementation` at
`545ef20`. Nothing here is built. Read `CLAUDE.md` first; D7, D8, D14, D15, D17, and D20 in `docs/decisions.md`
are the decisions this rests on.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [Where the text goes](#where-the-text-goes)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [What can only be validated at work](#what-can-only-be-validated-at-work)
- [Size](#size)

## What it is for

Labeled pairs are the bottleneck: every threshold and limit is a measurement (D3), and measurements need labels.
A reviewer who looks at a rejected edit, or at a sampled accepted one, and decides "edit", "recycle", or "unsure"
produces exactly the strongest kind of evidence the data request ranks first (a moderator action). Today that
decision is lost. This feature lets the caller hand the decision to the service, which checks that the pair is
consistent with what it has stored and records it as result events; a command then turns those events into a
`pairs.full.jsonl` file that `calibrate --full` reads unchanged. A second, smaller part flags a random fraction of
accepted edits for review, so that reviewers do not only ever see what the model already rejected.

## What already exists

Verified in the code:

- **The pair format and its reader.** `offline::pairs::parse_full_pairs` (`crates/similarity-server/src/offline/
  pairs.rs`) reads `pair_id`, `label` (any string; only `edit` and `recycle` count, anything else such as `unsure`
  is scored and shown), `note`, `split`, `owner`, `variant_change`, `label_source`, `label_confidence` (must be
  `high`, `medium`, or `low`), and `before`/`after` as nested objects. Unknown top-level keys are ignored, so an
  export can carry extra bookkeeping. Two pairs with one `pair_id` are refused (`unique`).
- **Slicing by `label_source` and `owner`.** `offline::stats` slices labeled pairs by `variant_change`,
  `category`, `label_source`, and `owner`, and bootstraps intervals over owners (D17). Reviewer pairs need no
  calibration code at all if they arrive in this format with a meaningful `label_source` and an `owner`.
- **A record as a JSON object.** `offline::RecordFields` serializes text as strings and quantities as objects,
  keys sorted, and reads back the same (a property test pins that).
- **Field fingerprints.** `similarity_core::record::{normalize_text, text_hash}` give the whitespace-normalized
  SHA-256 prefix that `RecordVectors::hashes` stores per field. `PreparedRecord` computes the same hash per field
  but keeps it private; there is no public accessor yet.
- **Stored quantities.** `StoredRecord::quantities` holds the baseline's quantities as values, and
  `similarity_core::quantity::compare` reports `QuantityChange::Unchanged` for the same amount in another unit.
  `Engine::revise` already uses exactly that to decide "nothing changed".
- **The tenant check.** `engine::same_tenant` (D14).
- **The input type for a version with no id.** `engine::Original` (fields and quantities, no id or tenant), made
  for `import`'s anchor.
- **Result events and their capture.** `similarity_log::RESULT` goes to stdout unfiltered;
  `similarity_log::testing::capture` lets a test read both streams. The calibration events in
  `offline/events.rs` are the model to copy: flat lines joined on an id, written and read back by one module.
- **`float_roundtrip`** is set for the workspace (D19), so a quantity written as a JSON number reads back exactly.
- **The write modes.** `WRITE_MODE_FORCE` is already "a human reviewer overrides the policy" (D7), and the
  README's "At edit time" step 4 tells the caller to force-write an approved edit.

Not present: any feedback RPC, any sampling, any reader of server-emitted result events. The server emits no
result events while serving today (the `evaluate` line is an ordinary `info` log).

## What the suggestion got wrong or left open

1. **"A `feedback` result event in the `pairs.full.jsonl` shape" cannot be written.** A pair line nests `before`
   and `after` as objects keyed by field name. `tracing` records scalars under names fixed at the call site, so a
   nested event could only be a preformatted JSON string, which D15 forbids; this is the same finding D17 records
   for calibration events. The review's requirement that the event be "a line `calibrate --full` can read back"
   is therefore met in two steps: flat events, and `feedback export`, which reassembles them into pair lines. The
   suggestion already had the export command; it is the only bridge, and it must be strict about completeness
   (below), because a log pipeline can drop or reorder lines.
2. **"`before` matches the stored baseline's field hashes" is too narrow and, for sampled edits, wrong.**
   - It must include quantities, compared by amount (`1 g` matches `1000 mg`), as `revise` compares them, and not
     by bytes: a caller's database can hold `1000 mg` where the store kept `1 g` because the resubmission was not a
     write.
   - It must include the tenant (D14).
   - For an accepted edit flagged for review, the edit has already been written, so the stored baseline is
     `after`, not `before`. The check has to accept a match on either side and say which.
   - Either way only one side is verifiable: the store never holds both versions. The check is a consistency
     check like D14 (it catches a stale or misaligned pair), not proof that the transition happened. An optional
     `revision` the caller got back from `Evaluate` tightens it to "nothing has moved since".
3. **No model is needed.** The check is hashes and numbers. The RPC never touches the embedding queue, so it can
   never be `RESOURCE_EXHAUSTED`, and it works during an outage of the model worker.
4. **"`offline.rs` (the export)"**: `offline` is now a directory, and the event names are an interface (D17). The
   emitter and the reader belong in one module, as `offline/events.rs` holds both for calibrations.
5. **Left open: pair ids and repeats.** Two reviewers, or one reviewer changing their mind, produce two decisions
   for one pair, and `calibrate` refuses duplicate ids. The export must deduplicate, and the plan has to say which
   decision wins.
6. **Left open: `owner` and pseudonyms.** The data request gives `owner` as a pseudonym, and the calibration
   report lists the largest owners by name in a text report that is meant to be pasted out of the work
   environment. Writing raw tenant ids into `owner` would leak them through that report.
7. **Left open: `label_confidence`, `variant_change`, and the origin of a review.** Without the origin (a rejected
   edit, a random sample, a complaint), the selection bias the sampling is meant to counter cannot be seen in the
   results. Encoding it in `label_source` makes `calibrate`'s existing slice show it with no new code.
8. **A limit the suggestion did not mention: a pair has no history.** The server judges a `RESTORED` field or
   quantity against the anchor; `Offline::compare` scores a pair with no anchor, so the same field is `ADDED`
   there and leaves the score. A reviewer pair whose edit turned on a restored field will score differently in
   `calibrate` from how the server scored it. The label is still right; the replay is a one-step edit. The
   anchor's text is not stored (the service keeps vectors), so the export cannot supply it. Drift and restoring
   belong to feature 11's histories. The README should say this.
9. **Anchoring of reviewers.** The data request warns that labels drawn from how different two texts look make a
   calibration circular. A reviewer shown the model's verdict and similarity is nudged towards it. That is a
   product decision about the review screen, not code here, but the README should advise showing the reasons
   only after the decision, and the origin slice will show whether sampled and rejected reviews disagree.
10. **"The text lands only in the operator's own log stream"** is true of the code and incomplete as a data
    statement: see [Where the text goes](#where-the-text-goes).

## Design

### The RPC

Added to `similarity.proto`, all additive:

```proto
// Records a reviewer's decision about one edit of a stored record, so that it can become a labeled pair.
//
// Never changes what is stored. To make an approved edit the record's baseline, call `Evaluate` with
// `WRITE_MODE_FORCE` as well, before or after this call: the check below accepts either order.
//
// The record must be stored, and must currently be either `before` (the edit was rejected, or previewed, and not
// written) or `after` (the edit was accepted and written, as for an edit flagged with `review_sample`). Text is
// compared after whitespace normalization and quantities by amount, as `Evaluate` compares them.
//
// Errors: `INVALID_ARGUMENT` for a version that does not fit the schema, a `decision` or `confidence` that is not
// given or not known, or two versions that do not differ; `NOT_FOUND` if no record is stored for the id;
// `FAILED_PRECONDITION` for another tenant's record, a stored record that is neither version, or a `revision`
// that is not the stored one; `UNAVAILABLE` if the vector store cannot be reached.
rpc RecordFeedback(RecordFeedbackRequest) returns (RecordFeedbackResponse);

// A version of a record: its fields and quantities, with no id or tenant of its own.
message Version {
  map<string, string> fields = 1;
  map<string, Quantity> quantities = 2;
}

enum ReviewDecision {
  REVIEW_DECISION_UNSPECIFIED = 0;
  // The same record, legitimately edited.
  REVIEW_DECISION_EDIT = 1;
  // A different record under the old id.
  REVIEW_DECISION_RECYCLE = 2;
  // The reviewer could not decide, or the policy does not settle it (many same-brand variant changes).
  REVIEW_DECISION_UNSURE = 3;
}

// Why the edit was in front of a reviewer. Reviews of rejections alone over-represent what the model already
// catches; the origin is what lets a calibration tell the two populations apart.
enum ReviewOrigin {
  REVIEW_ORIGIN_UNSPECIFIED = 0;
  // `Evaluate` rejected it.
  REVIEW_ORIGIN_REJECTED = 1;
  // `Evaluate` accepted it and set `review_sample`.
  REVIEW_ORIGIN_SAMPLED = 2;
  // Someone reported it (a retailer, a customer, support).
  REVIEW_ORIGIN_REPORTED = 3;
}

enum ReviewConfidence {
  REVIEW_CONFIDENCE_UNSPECIFIED = 0;
  REVIEW_CONFIDENCE_HIGH = 1;
  REVIEW_CONFIDENCE_MEDIUM = 2;
  REVIEW_CONFIDENCE_LOW = 3;
}

// The kind of same-brand variant change an edit is, as the data request defines them.
enum VariantChange {
  VARIANT_CHANGE_UNSPECIFIED = 0; // not a variant change
  VARIANT_CHANGE_FLAVOR = 1;
  VARIANT_CHANGE_STRAIN = 2;
  VARIANT_CHANGE_SIZE = 3;
  VARIANT_CHANGE_COUNT = 4;
  VARIANT_CHANGE_POTENCY = 5;
}

message RecordFeedbackRequest {
  string id = 1;
  // Required if the record is stored under a tenant (D14).
  optional string tenant = 2;
  Version before = 3;
  Version after = 4;
  ReviewDecision decision = 5;
  ReviewOrigin origin = 6;
  ReviewConfidence confidence = 7;
  VariantChange variant_change = 8;
  // The reviewer's note, free text. Becomes the pair's `note`.
  optional string note = 9;
  // The `revision` the `Evaluate` response for this edit reported. If given, it must be the stored revision.
  optional uint64 revision = 10;
}

enum FeedbackMatch {
  FEEDBACK_MATCH_UNSPECIFIED = 0;
  // The stored record is `before`: the edit was not written.
  FEEDBACK_MATCH_BEFORE = 1;
  // The stored record is `after`: the edit was written.
  FEEDBACK_MATCH_AFTER = 2;
}

message RecordFeedbackResponse {
  // The id the pair will have in the exported file.
  string pair_id = 1;
  FeedbackMatch matched = 2;
  // The stored record's revision.
  uint64 revision = 3;
}
```

And on `EvaluateResponse`:

```proto
  // This accepted edit was picked at random, at the rate the server is configured with, for a human to review. Route
  // it to review as you would a rejection, and report the decision with `RecordFeedback` and
  // `REVIEW_ORIGIN_SAMPLED`. Only ever set on a `VERDICT_ACCEPTED` that was stored in `WRITE_MODE_ENFORCE`.
  bool review_sample = 12;
```

Unknown enum values are refused, as `write_mode` refuses them: `decision`, `origin`, and `confidence` are
required, and `UNSPECIFIED` is `INVALID_ARGUMENT`. `variant_change` treats `UNSPECIFIED` as "none". `Version`
mirrors `engine::Original`; reusing `Record` for the two sides would invite an id or tenant inside each side that
has to be refused.

### The engine

In `engine.rs`, beside `ImportInput`:

```rust
/// A reviewer's decision about one edit of a stored record.
pub struct FeedbackInput {
  pub id: String,
  pub tenant: Option<String>,
  pub before: Original,
  pub after: Original,
  pub decision: Decision,          // Edit, Recycle, Unsure
  pub origin: ReviewOrigin,        // Rejected, Sampled, Reported
  pub confidence: Confidence,      // offline::pairs::Confidence, moved or re-exported (see below)
  pub variant_change: Option<VariantChange>,
  pub note: Option<String>,
  pub revision: Option<u64>,
}

/// Which version of a reviewed edit the store holds.
pub enum Matched { Before, After }

/// A reviewer's decision, checked against the store and ready to be recorded.
pub struct Feedback {
  pub pair_id: String,
  pub matched: Matched,
  pub revision: u64,
  pub owner: Option<String>,       // the stored tenant, else the request's
  pub input: FeedbackInput,        // as sent: the text is recorded exactly as received
  pub hashes: [BTreeMap<String, String>; 2], // field fingerprints of before and after, for the events
}
```

`Engine::feedback(&self, input: FeedbackInput) -> Result<Feedback, EngineError>`:

1. `required_text` on the id and tenant; `prepare` both versions (schema validation, the same messages as
   `import`'s "the original version: ..." prefix, here "`before`: ..." and "`after`: ...").
2. Refuse two versions that are the same record: equal field hashes and every quantity `Unchanged` by
   `quantity::compare`. That is the rule `revise` uses for "nothing changed", and a pair with no change is not a
   pair (the data request skips them too).
3. `store.get(id)`; `None` is `NotFound`. `same_tenant` (D14).
4. Match: a version matches the stored record when its field hashes equal `stored.vectors.hashes` and, for every
   quantity name either side has, both lack it or `compare(name, Some(stored), None, Some(version))` is
   `Unchanged`. `before` is tried first. Neither is `FailedPrecondition` naming which fields and quantities
   differ from the stored record, by name only (never echoing text into an error that a log line may carry).
5. If `revision` is given and differs from `stored.revision`, `FailedPrecondition`.
6. `pair_id`: `fb-` and the first 16 hex digits of the SHA-256 of the record id, the two hash maps, and the two
   quantity maps as JSON (sorted keys). The decision is deliberately **not** in it, so two decisions about one pair
   collide and the export can choose between them.

It takes no lock and writes nothing: a single `get` is atomic, and nothing it does depends on a second read. It
needs `PreparedRecord::hashes()` (a new public accessor in `similarity-core`, returning field name to hash for the
fields that have a value, the same map `RecordVectors::hashes` holds) so that hashing does not need a model.

`offline::pairs::Confidence` and the decision labels are wanted on both sides (engine and export). Move
`Confidence` to `engine` or a small shared module and re-export it from `offline::pairs`, so the spelling `high`,
`medium`, `low` exists once.

### The events

A new module `crates/similarity-server/src/feedback.rs` owns the event names, the emitter, and the reader, as
`offline/events.rs` does for calibrations. `grpc.rs` calls `feedback::emit(&feedback)` after the engine returns,
as it logs the `evaluate` line; it decides nothing. All lines are `target: similarity_log::RESULT`, so no
`RUST_LOG` hides a reviewer's decision (D15). Flat, joined on `pair_id` (D17):

| `message` | One per | Fields |
|---|---|---|
| `feedback` | decision | `pair_id`, `record_id`, `owner` (if any), `label` (`edit`, `recycle`, `unsure`), `label_source`, `label_confidence`, `variant_change` (if any), `note` (if any), `matched` (`before`, `after`), `revision`, `model_fingerprint`, `layout`, `before_texts`, `before_quantities`, `after_texts`, `after_quantities` (counts) |
| `feedback_text` | field per side | `pair_id`, `side` (`before`, `after`), `field`, `text` (exactly as received), `hash` (the field fingerprint) |
| `feedback_quantity` | quantity per side | `pair_id`, `side`, `quantity`, `min`, `max`, `unit` (as received) |

`label_source` is `reviewer_rejected`, `reviewer_sampled`, or `reviewer_reported`, from the origin, so the
existing `label_source` slice in `calibrate` separates the populations. The counts in the `feedback` line and the
`hash` on each text line let the export prove a pair is complete and intact: some log transports split long lines
(Docker's default driver splits at 16 KB), and a description can be longer than that. Numbers are numbers; an
absent optional value is left out (D15). No similarity, verdict, or reason is recorded: the export is re-scored by
`calibrate` under whatever model and config are current, which is the only scoring that means anything for a
threshold (D3).

Each event's names are an interface from the first commit, like D17's: add fields, never rename.

### The export command

`record-similarity feedback export --out pairs.reviewer.full.jsonl --owner-salt <secret> <events.ndjson>...`

- Reads NDJSON as the server wrote it. Lines that are not JSON, not `target: result`, or not one of the three
  messages are skipped and counted. Unwrapping a log pipeline's envelope (`jq -r .log`) is the operator's job,
  and the README says so.
- Groups by `pair_id`. A pair is complete when it has its `feedback` line, exactly the counted text and quantity
  lines for each side, and every text's fingerprint matches its text. An incomplete pair is reported with its id
  and what is missing, and left out.
- When one `pair_id` has several complete decisions, the one with the latest `timestamp` wins, and the summary
  counts how many were superseded and how many of those disagreed with the winner (decision 3).
- Writes one `pairs.full.jsonl` line per pair: `pair_id`, `label`, `label_source`, `label_confidence`,
  `variant_change`, `note`, `owner`, `before`, `after` (through `RecordFields`'s serializer, so the format is the
  one `calibrate --full` already reads), plus bookkeeping `calibrate` ignores: `product` (the record id), `matched`,
  `revision`, `decided_at` (the event timestamp), `model_fingerprint`. No `split` (decision 7).
- `owner` and `product` are `hex_prefix(SHA-256(salt, value), 8)` (decision 2), so the calibration report's owner
  slice shows pseudonyms. `--owner-salt` is required unless `--raw-ids` is given.
- The output is written to a sibling and renamed, so an interrupted run leaves no half file. The summary
  (`lines`, `skipped_lines`, `pairs`, `incomplete`, `superseded`, `disagreeing`, by label) is one `finished` result
  event on stdout, as `import` reports, and is never mixed into the pairs file.
- Exit status: zero when it wrote a file, even with incomplete pairs: a lossy pipeline is ordinary, and the counts
  are in the summary. `--fail-on-incomplete` for whoever wants it strict.

The command is offline: no model, no store, no network, and it lives in `main.rs` as a `Feedback` subcommand group
with `Export` in it, calling `feedback::read` and `feedback::export`.

### Sampling accepted edits

Config, a new optional section (absent means off; `deny_unknown_fields` as elsewhere):

```toml
[feedback]
# The fraction of accepted, stored edits flagged `review_sample` for a human to review. 0 turns it off. Set it from
# how many reviews the team can do, not from a measurement: it changes no verdict.
sample_accepted = 0.0
```

`sample_accepted` must be finite and in `[0, 1]`; anything else is a config error. `config.example.toml` ships
`0.0`. In the engine, `Engine::with_review_sampling(Sampling { fraction, seed })` (a builder, so `Engine::new`
and its many test callers do not change). After `revise` writes under `WriteMode::Enforce` with `Verdict::Accepted`,
`review_sample` is set when `SHA-256(seed, id, revision)` read as a `u64` falls below `fraction * u64::MAX`.
Keyed by id and revision, the choice is a pure function and is testable; the seed is drawn once at startup from
`RandomState` (as the lock hasher is), so a merchant cannot predict which revision will be looked at. A dry run, a
forced write, a creation, an unchanged resubmission, and a rejection are never sampled. The `evaluate` log line
gains `review_sample`. This is not a verdict and not part of `Policy`: it changes no write.

## Where the text goes

Feedback carries real product text, and in production that is real product data.

- **The events** go to the server's stdout, with every other result, and from there wherever the operator's log
  pipeline sends stdout. If that pipeline forwards to a hosted log service, product text goes there. That is the
  same exposure as the request bodies a load balancer may log, but it is new for this service, and the owner
  should confirm it with whoever runs the pipeline (decision 1). Operators who want the text kept apart can route
  the server's stdout separately from stderr; the split already exists (D15).
- **The exported file** belongs in the work environment beside `pairs.full.jsonl` from the data request, outside
  any git repository. It is never committed here. Nothing in this repository will ever hold a reviewer pair; the
  tests use synthetic text only.
- **Errors never echo text.** A mismatch names fields, not values, because error messages end up in logs too.
- No reviewer identity is recorded. It would be personal data about employees, and nothing in the calibration
  needs it.

## The work, in order

Each step is one commit and leaves `just ci` passing.

1. **`PreparedRecord::hashes()`** in `similarity-core/src/record.rs`, with a test that it equals what `finish`
   puts in `RecordVectors::hashes`. Move `Confidence` to where both the engine and `offline::pairs` can use it.
2. **`Engine::feedback`** with `FeedbackInput`, `Feedback`, `Matched`, `Decision`, `ReviewOrigin`,
   `VariantChange`, and the engine tests below. No proto yet.
3. **`feedback.rs`: emit, read, export** as pure functions over strings, with the capture-based tests and the
   round trip through `parse_full_pairs`.
4. **The proto, `grpc.rs`, and the client example.** `RecordFeedback` mapped in `grpc.rs` (decoding enums,
   refusing unknown values, calling the engine, calling `feedback::emit`). A `feedback` subcommand in
   `examples/client.rs` taking two record files and a decision. gRPC tests in `lib.rs`.
5. **`feedback export`** in `main.rs`: arguments, the sibling-and-rename write, the summary event, exit status.
6. **Review sampling**: `[feedback]` in `config.rs` and `config.example.toml`, `Engine::with_review_sampling`,
   `EvaluateOutput::review_sample`, the proto field, the log line. Run `just test-real` once, because
   `config.example.toml` changed (CLAUDE.md), though nothing that affects a verdict did.
7. **Properties**: a `Feedback` request in the state machine of `similarity-server`'s `properties` module, and
   the sampling rule as a property. `PROPTEST_CASES=5000 just test` once, since `evaluate`'s output changed.
8. **Documentation**, below, and the new decision record.

## Tests

None needs the model or Qdrant: the check is hashes and numbers, and everything else is events and files.
`MemoryStore`, `WordBagEmbedder`, and `similarity_log::testing::capture` cover all of it.

Engine (`engine.rs`):

- `feedback_on_a_rejected_edit_matches_the_stored_before_and_writes_nothing`
- `feedback_on_an_accepted_edit_matches_the_stored_after`
- `feedback_matches_whichever_order_it_comes_in_around_a_forced_write`
- `a_quantity_in_other_units_still_matches_the_stored_baseline`
- `a_quantity_that_differs_from_the_stored_baseline_is_a_failed_precondition`
- `feedback_whose_versions_both_differ_from_the_stored_record_is_refused_and_names_the_fields_not_their_text`
- `feedback_for_a_record_that_is_not_stored_is_not_found`
- `feedback_must_name_the_tenant_the_record_is_stored_under`
- `a_revision_that_is_not_the_stored_one_is_a_failed_precondition`
- `two_versions_that_do_not_differ_are_not_a_pair`
- `feedback_never_calls_the_model` (a failing embedder, as `embedder_failures_keep_their_meaning` builds one)
- `two_decisions_about_one_edit_get_the_same_pair_id_whatever_was_decided`
- `an_accepted_edit_is_sampled_at_the_configured_rate` (fixed seed, 10,000 ids, fraction 0.1, the count within a
  fixed band; deterministic, so not flaky)
- `only_an_accepted_enforced_write_is_ever_sampled` (dry run, force, create, rejection, unchanged resubmission)
- `sampling_is_off_unless_configured`

Events and export (`feedback.rs`):

- `a_decision_is_written_as_flat_result_events_joined_on_its_pair_id`
- `text_and_quantities_are_recorded_exactly_as_received`
- `feedback_events_are_written_whatever_the_log_level`
- `an_exported_pair_reads_back_through_calibrate_full_as_the_pair_that_was_reviewed`
- `the_origin_of_a_review_becomes_its_label_source`
- `a_pair_missing_a_line_is_left_out_and_counted`
- `a_text_whose_fingerprint_does_not_match_is_left_out_and_counted` (a line cut short)
- `the_latest_decision_about_a_pair_wins_and_the_disagreement_is_counted`
- `lines_that_are_not_feedback_results_are_skipped_and_counted`
- `owners_and_record_ids_are_pseudonymized_with_the_salt`
- `an_unsure_decision_is_exported_and_counted_in_no_error_rate`

gRPC (`lib.rs`):

- `a_reviewer_decision_crosses_the_wire_and_reports_which_version_is_stored`
- `an_unknown_or_missing_decision_is_an_invalid_argument`
- `review_sample_crosses_the_wire`
- extend `errors_arrive_as_the_right_status_codes` with the feedback cases.

Properties (`properties.rs`):

- `every_request_in_any_sequence_obeys_the_write_rules` gains a `Feedback` request built from the last stored
  version and the next request's record; the rule checked is that it never changes what is stored.
- `feedback_is_accepted_exactly_when_one_version_is_what_is_stored`

Config (`config.rs`):

- `a_sampling_fraction_outside_zero_to_one_is_a_config_error`

## Documentation to update

- **README.md**:
  - "Integrating it", "At edit time": a step 5, "When a reviewer decides, call `RecordFeedback`", and that a
    `review_sample` on an accepted edit should be routed to review. Advise showing the reviewer the reasons only
    after they decide.
  - "What `Evaluate` returns": `review_sample`.
  - A new section under "Choosing thresholds", "Labels from reviewers": the export command, where the file
    belongs, that `--full` reads it, the `label_source` slice, and the one-step-edit limit (a restored field scores
    as added).
  - "Configuration reference": `[feedback] sample_accepted`.
  - "Making requests": a `just client feedback ...` example, run and its output pasted as the README rule
    requires.
- **CLAUDE.md**, under "Things that are easy to get wrong":
  - `RecordFeedback` never writes. An approved edit becomes the baseline only through `Evaluate` with `FORCE`.
  - The feedback event names are an interface, as the calibration events are (D17).
  - Reviewer pairs and server stdout from production carry product text and never enter this repository.
  - Workspace layout: `feedback.rs` in `similarity-server`.
- **docs/decisions.md**: a new record (the next free number, D22 unless another feature takes it first), "A
  reviewer's decision is recorded, never applied": why the RPC does not write, why either version may match, why
  flat events and an export instead of a pair line, why the decision is not in the pair id, why sampling is keyed by
  id and revision, and the no-history limit. D7's table is unchanged; its "Three write modes" paragraph gains one
  sentence pointing at the new record.
- **docs/feature-suggestions.md** and **docs/plan-features.md**: mark 17 built.

## Dependencies on other features

None is required. Two touch it:

- **Feature 19 (reason codes)** would let a review screen show `Reason::code`s instead of sentences. Assumed
  interface: none; feedback records no reasons.
- **Feature 10 (production telemetry)** could join `feedback` events to `evaluate` lines on `record_id` and
  `revision` to measure how often reviewers overturn the model. Assumed interface: the `evaluate` log line keeps
  `id` and `revision` (it has both today).
- **Feature 11 (drift from histories)** is where a restored-field judgment can be replayed; nothing here waits on
  it.

## Decisions for the owner

1. **Is product text in the server's result stream acceptable for the production log pipeline?** Recommended:
   yes, with stdout routed to a store inside the work environment. Alternative if not: a `[feedback] path` the
   server appends pair lines to directly, which keeps text out of logs but adds file handling, rotation, and one
   file per replica to the server.
2. **Pseudonymize `owner` and `product` in the export?** Recommended: yes, with a required salt, because the
   calibration report lists owners and is meant to be pasted out of the work environment. `--raw-ids` for local
   joins.
3. **Several decisions about one pair.** Recommended: the latest wins, and the export counts disagreements.
   Alternative: any disagreement makes the pair `unsure`.
4. **The sampling rate.** Recommended: ship `0.0`; the owner sets it from reviewer capacity. A rate is a budget,
   not a measurement, so this is not a threshold rule exception.
5. **`label_source` values.** Recommended: `reviewer_rejected`, `reviewer_sampled`, `reviewer_reported`, so the
   existing slice shows the selection bias. Alternative: `reviewer` plus a separate `review_origin` key that
   `calibrate` does not slice on.
6. **Is `confidence` required?** Recommended: yes. A pair with no confidence passes every `--min-confidence`
   filter, which is the wrong default for a label a reviewer was unsure of.
7. **`split`.** Recommended: the export writes none. The data request's split is by a salted brand hash the work
   export chose, and guessing it here would mix brands across the split.
8. **Record the server's verdict and similarity in the event?** Recommended: no. It would cost an embedding per
   feedback, and `calibrate` re-scores the pair under the current model anyway.

## What can only be validated at work

- That reviewer labels agree with the data request's other evidence on overlapping pairs.
- Whether the `reviewer_sampled` and `reviewer_rejected` slices disagree, which is the point of the sampling.
- How the production log pipeline handles long lines and envelopes, and whether the export's completeness counts
  stay at zero.
- Any threshold or limit read from reviewer pairs, by `calibrate --full`, which this feature does not change.

## Size

**Medium.** No model, no store change, no new dependency (`sha2`, `serde_json`, `tracing`, and `clap` are already
in `similarity-server`; `cargo deny` sees nothing new). The work is one read-only engine method, one proto
addition, one event module with a strict reader, one command, and a small sampling rule, each with tests that run
in memory. It is not small because the export's completeness and deduplication rules and the event interface have
to be right the first time: other tools will read them.
