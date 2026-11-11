# Plan: production score telemetry and drift report (feature 10)

A plan for feature 10 of `docs/feature-suggestions.md`, written against the code on `initial-implementation` at
`545ef20`, before any of it was built. Read `CLAUDE.md` first, then D7, D8, D12, D15, D16, and D17 in
`docs/decisions.md`. Nothing here has been run; every claim about the code below was checked by reading it.

## Contents

- [What it is for](#what-it-is-for)
- [What exists to build on](#what-exists-to-build-on)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [What can only be validated at work](#what-can-only-be-validated-at-work)
- [Size](#size)

## What it is for

The README's rollout plan is to run the service on real traffic first and log what it would have done. The log
line the server writes today cannot answer the questions that rollout exists to answer: which field caused a
rejection, which tenant is behind most of them, whether the score distribution has moved since calibration, and
what a candidate threshold would do to real traffic. This feature makes the per-request record of a verdict
complete enough to be that dataset, and adds one offline command, `drift`, that reads it.

It has three parts, in order of value:

1. **A complete, flat record of every `Evaluate`**, as result events joined on a request id, in the same shapes
   the calibration events already use (D17). Small, and should go first.
2. **`record-similarity drift`**: per-window distributions, rejection rates by reason and tenant, shift against
   the previous window and against a calibration, and (cheaply, by reusing `Evaluation::rejudge`) what a second
   configuration would have done to the same traffic.
3. **A Prometheus endpoint.** Recommended to defer; see the decisions.

## What exists to build on

Verified in the code:

- **The log line.** `GrpcService::evaluate` in `crates/similarity-server/src/grpc.rs` logs one `info!` event,
  message `evaluate`, with `id`, `verdict` (as `?Debug`, so `"Created"`, capitalized), `similarity`,
  `anchor_similarity`, `stored`, `revision`, and `measure_changed`. The similarities are `f32` printed unrounded.
  It goes to stderr under `RUST_LOG`, and the README tells operators that `RUST_LOG=warn` "quiets the per-request
  lines". `find_similar` and `delete` log one line each. Engine errors are not logged at all except
  `Unavailable` (warn) and `Internal` (error) in `impl From<EngineError> for Status`.
- **What the engine returns.** `EvaluateOutput` in `engine.rs` has `verdict`, `similarity`, `threshold`, `fields:
  Vec<FieldScore>`, `anchor_similarity`, `stored`, `reasons: Vec<String>`, `vector`, `revision`, `quantities:
  Vec<QuantityScore>`, `measure_changed`. The reasons are rendered to sentences in `Engine::revise`
  (`evaluation.reasons.iter().map(ToString::to_string)`), so the structured `Reason` is gone by the time
  `grpc.rs` sees it. Nothing is timed anywhere in the engine.
- **Reasons.** `similarity_core::score::Reason` with `code()`, `subject()`, `value()`, `limit()`. The six codes
  are an interface (D17).
- **Flat events.** `crates/similarity-server/src/offline/events.rs`: `six` (round to six places), `emit_schema`
  (`schema_field`, `schema_quantity` lines), and the `pair` / `pair_field` / `pair_quantity` / `pair_reason`
  shapes joined on `pair_id`. The reader there (`Event`, `field_change`, `quantity_change`, `reason`) already
  turns those lines back into `FieldScore`, `QuantityScore`, and `Reason`. `change_name` and
  `quantity_change_name` are `pub(crate)` in `offline/mod.rs`.
- **Judging again.** `Evaluation::rejudge(schema, policy)` in `score.rs` re-applies weights, floors, thresholds,
  the anchor threshold, and quantity limits to stored measurements. `offline/diff.rs::rejudge` uses it for
  `calibrate-diff --config-b`.
- **Logging.** `similarity-log` routes target `result` to stdout, never filtered, and everything else to stderr
  under `RUST_LOG`. `similarity_log::testing::capture` captures both streams for tests.
- **What a request knows.** `EvaluateInput.threshold` is a per-request override of the overall threshold
  (`Policy::with_threshold`). `StoredRecord.updated_at` (seconds) and `revision` are on the baseline the engine
  reads. The model fingerprint is `"<digest>-d<dim>-t<max_tokens>"` (`Embedder::fingerprint`).
- **Units.** `similarity_core::quantity::unit(raw)` returns `Dimension::Other(normalized)` for a unit the table
  does not know.
- **HTTP.** `axum 0.8.9` and `hyper 1.12` are already in `Cargo.lock`, pulled in by `tonic`'s `router` feature.
  `metrics` and `metrics-exporter-prometheus` are not, and are not in the local registry, so their dependency
  trees could not be checked here.
- **Feature 19's plan** (`docs/plans/19-reason-codes.md`) makes the same engine change this one needs:
  `EvaluateOutput.reasons` becomes `Vec<Reason>`.

## What the suggestion got wrong or left open

1. **The line already carries more, and is already wrong in two ways.** `measure_changed` is there. The
   similarities are unrounded (the one event in the codebase that is; `docs/plan-features.md` asks for this to be
   fixed here), and `verdict` is a Rust `Debug` name rather than the lowercase names every other event uses.
2. **"Add reason codes" cannot be done in `grpc.rs` today.** The engine throws the `Reason` away. The engine has
   to return it (the same change feature 19 needs).
3. **"Per-field similarity" cannot go on one line.** `tracing` field names are fixed at the call site, so a value
   per field name is impossible on one event, and a list could only be a preformatted string (D15). The answer is
   the one D17 found: item lines joined on an id. Because requests run concurrently, their lines interleave, so the
   join id is required, not just tidy.
4. **"The time spent embedding and in Qdrant" is measured nowhere.** The engine has to time its own calls.
5. **The suggestion predates quantities.** The record should carry per-quantity changes and factors, and the
   drift report should list units the table does not know, since `CLAUDE.md` says to extend `UNITS` "when a new
   unit turns up" and nothing currently says when one has.
6. **"Compare with calibration" needs care.** A calibration mixes edits and recycles; production traffic is
   mostly edits with an unknown share of recycles. The fair reference is the calibration's edit-labeled pairs, and
   the report must say that a shift can mean more recycling as well as a changed population. And the calibration
   here is synthetic: comparing production with `data/calibration.sample.*` means nothing.
7. **Unchanged fields would swamp every distribution.** An unchanged field reuses its stored block and scores
   about 1. Field distributions have to be taken over `modified` and `restored` fields only.
8. **The rollout plan, as written, does not produce the dataset it promises.** In `DRY_RUN` nothing is ever
   written, so a record's second, third, and tenth edits are all compared with the baseline the import stored,
   not with the edit before. Rollout telemetry would measure distance from the catalog snapshot, growing over the
   weeks of the rollout, and look like drift. And dry runs fired while a merchant types cannot be told from the
   one at save. See decision 3.
9. **Logs filtered by `RUST_LOG` are the wrong home for a dataset.** D15 moved the trainer's results off the log
   filter because `RUST_LOG=warn` produced a run that reported nothing; the README currently recommends exactly
   that setting to quiet these lines. See decision 1.
10. **PSI and KS have no thresholds to ship.** The usual "PSI above 0.25 is a shift" is folklore, not a
    measurement of this system. The report prints the statistics; a gate takes a limit the operator gives, with no
    default (as `CLAUDE.md` requires of thresholds).
11. **Where it goes.** `offline.rs` is now the `offline/` directory, and `similarity-runs` does not exist
    (feature 2 is unbuilt). The command belongs in `offline/`.
12. **Prometheus is less free than it looks.** The embedding queue is a `std::sync::mpsc::SyncSender`, which has
    no length, so "queue depth" needs a counter in the worker. The two `metrics` crates are a new dependency tree
    that could not be inspected offline. And every number the suggestion wants on the endpoint can be derived by a
    log pipeline from the events in part 1.

## Design

### Part 1: the telemetry events

**Where.** A new module, `crates/similarity-server/src/telemetry.rs`, holds the emitting functions; `grpc.rs`
calls them and stays free of decisions. The emitting functions are synchronous and take plain data, so they are
tested with `similarity_log::testing::capture` and no runtime.

**Target.** Results (`target: similarity_log::RESULT`), on stdout, never filtered (decision 1). No change to
`similarity-log`. The existing `find_similar` and `delete` lines stay logs.

**Shared helpers.** Move `six`, `emit_schema`, `change_name`, and `quantity_change_name` to a small
`crate::events` module (or make them `pub(crate)` where they are) so that calibration and telemetry spell every
name and round every number the same way. The calibration's output must not change: its tests pin it.

**Request id.** `request_id`: 16 lowercase hex characters, `similarity_core::rng::Rng::new(seed ^ counter)
.next_u64()`, where `seed` is drawn once at start from `RandomState` (the same source `Engine::lock_hasher` uses)
and `counter` is an `AtomicU64`. Unique across replicas and restarts with overwhelming probability, and needs no
dependency. Optionally, an `x-request-id` gRPC metadata value, if the caller sends one, is logged beside it as
`caller_request_id`, cut to 64 bytes (decision 6).

**Engine changes** (`engine.rs`), all additive except the first:

```rust
pub struct EvaluateOutput {
  // ...as today, except:
  /// Why the record was rejected; empty means accepted.
  pub reasons: Vec<Reason>, // was Vec<String>; `grpc.rs` renders the sentences (shared with feature 19)
  /// How many texts were sent to the model (fields whose text changed, or every field of a new record).
  pub texts_embedded: usize,
  /// The baseline's revision and age when there was one, so that telemetry shows what a revision was compared with.
  pub baseline: Option<BaselineInfo>,
  /// Where the time went.
  pub timings: Timings,
}

pub struct BaselineInfo {
  pub revision: u64,
  /// Seconds since the baseline was written, saturating at zero for a clock that went backwards.
  pub age_s: u64,
}

/// Wall time of each stage, in microseconds. Embedding includes the wait in the worker's queue.
pub struct Timings {
  pub store_read_us: u64,
  pub embed_us: u64,
  pub store_write_us: u64, // zero when nothing was written
}
```

Measured with `std::time::Instant` around `self.store.get`, `self.embed`, and `self.store.put`; durations
converted with `u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)`. Timing is not a decision, so it sits in
the engine without contradicting `CLAUDE.md`. The grpc layer adds `total_us` around the engine call (it includes
the per-id lock wait, which the engine's stages do not).

**Events.** Every value a scalar; absent optionals left out; similarities through `six`; factors as they are;
durations as integers. Names follow the calibration events, so one parser reads both.

| `message` | One per | Fields |
|---|---|---|
| `server` | start | `fingerprint`, `dimension`, `max_tokens`, `threshold`, `anchor_threshold`, `reject_on_measure_change`, `version` (crate version), followed by `schema_field` and `schema_quantity` lines as in a calibration |
| `evaluate` | request | `request_id`, `caller_request_id`, `id`, `tenant`, `mode`, `verdict`, `stored`, `revision`, `baseline_revision`, `baseline_age_s`, `similarity`, `anchor_similarity`, `threshold`, `threshold_overridden`, `measure_changed`, `fingerprint`, `texts_embedded`, `store_read_us`, `embed_us`, `store_write_us`, `total_us`, `field_count`, `quantity_count`, `reason_count` |
| `evaluate_field` | request and field | `request_id`, `field`, `change`, `similarity`, `weight`, `below_floor` |
| `evaluate_quantity` | request and quantity | `request_id`, `quantity`, `change`, `factor`, `zero_crossing`, `incomparable`, `previous_unit`, `unit`, `unit_known`, `beyond_limit` |
| `evaluate_reason` | request and reason | `request_id`, `code`, `field`, `value`, `limit` |
| `evaluate_failed` | request that returned an error | `request_id`, `caller_request_id`, `id`, `tenant`, `mode`, `code` (the gRPC code in snake case: `resource_exhausted`, `failed_precondition`, ...), `total_us` |

- `mode` is `enforce`, `dry_run`, or `force`; `verdict` is `embedded`, `created`, `accepted`, or `rejected`.
- The `evaluate` line keeps its message name and its existing field names, so `jq` filters on it keep working;
  what changes is its stream, its verdict spelling, its rounding, and that it has more fields (decision 5).
- `field_count`, `quantity_count`, and `reason_count` let a streaming reader know when a request's lines are all
  in, since concurrent requests interleave. The `evaluate` line is written first, so a person reading with `jq`
  sees the header before the items.
- `fingerprint` is on every line because log files rotate and the `server` line is then gone; it is about 80
  bytes, and the report groups by it (a verdict means nothing without the model that gave it, D3, D12).
- `evaluate_failed` does not carry the error message: a tenant mismatch message names both tenants, and the code
  is what is counted. The ordinary `warn`/`error` logs for `Unavailable` and `Internal` stay as they are.

**What is never logged.** No field text, no hash of field text, no vector. The record `id` is logged today and
stays; the `tenant` is an identifier, not content (decision 2). Units are caller-supplied strings and could carry
anything, so `previous_unit` and `unit` are cut to 32 bytes on a character boundary; `unit_known` says whether
`quantity::unit` recognized it, which is what the drift report needs to list unknown units. `caller_request_id`
is cut to 64 bytes.

**Volume.** A four-field, four-quantity revision that is rejected for one reason writes ten lines, about 2.5 KB.
A creation or an embed-only call writes one. That is the cost of making the stream a dataset, and why it is a
decision, not a default nobody chose.

### Part 2: `record-similarity drift`

```text
record-similarity drift TELEMETRY.ndjson... [--reference CALIBRATION.ndjson] [--config-b CONFIG]
  [--window week|day] [--modes enforce,dry_run,force] [--tenants N] [--min-values N]
  [--fail-on-psi LIMIT] [--fail-on-ks LIMIT] [--ndjson]
```

- Reads one or more files, or `-` for stdin (so `zcat logs/*.gz | record-similarity drift -` works with no
  compression dependency). Reads line by line with `BufRead`, never the whole file: a week of traffic is large.
- Skips lines that are not result events, as `events::read` does, so a file that captured stderr too still reads.
- A request is complete when its counts are met; one still incomplete at end of input is counted and reported
  (`incomplete`), never guessed at.
- The window comes from the envelope's `timestamp` (`2026-11-11T05:52:43.254934Z`): its first ten characters are
  the UTC date, turned into a day number with the usual days-from-civil arithmetic (written with `checked_*` and
  `saturating_*` for the lints) and into an ISO week. No date dependency.
- Windows are keyed by `(fingerprint, window)`. No statistic compares two fingerprints: a change of model starts
  a new series, and the report says where it happened.

**What it reports, per fingerprint and window:**

- Requests by mode and verdict; the rejection rate over judged requests (accepted plus rejected) by mode; failures
  by code; `p50`, `p95`, `p99` of `total_us`, `embed_us`, and `store_read_us`; the share of requests whose
  baseline is older than the window began (a direct view of problem 8 above).
- Rejections by reason code.
- Per field, over `modified` and `restored` fields only: count, `p5`, `p25`, `p50`, the number below its floor,
  and PSI and KS against the previous window and against the reference. The same for the overall `similarity` and
  for `anchor_similarity`.
- Per quantity: how many changed, `p50`, `p95`, and the largest factor, zero crossings, incomparable changes.
- Units the table does not know, normalized by `quantity::unit`, with counts.
- The `--tenants` tenants with the most rejections (default 10), with judged count, rejections, and rate.
- With `--config-b`: each judged request rebuilt as an `Evaluation` (fields, similarity, anchor similarity,
  quantities) and judged again under that file's `[policy]`, `[fields]`, and `[quantities]`, exactly as
  `calibrate-diff --config-b` does; reported as the rejection rate under each configuration, the flips in each
  direction, and the reason codes behind new rejections. A request that overrode the threshold keeps its own.
  Field and quantity names that differ from the traffic's are refused, as `calibrate-diff` refuses them.

**The statistics** go in `offline/stats.rs`, so the text report and the events cannot disagree (`CLAUDE.md`):

- `quantile(sorted, q)`: nearest-rank.
- `ks(a, b) -> f64`: the two-sample Kolmogorov-Smirnov statistic, the largest gap between the two empirical
  distribution functions, computed by merging the sorted samples. No p-value: with tens of thousands of values
  every difference is significant, and a p-value would invite reading it as a verdict.
- `psi(reference, current) -> f64`: bins at the reference's deciles (edges deduplicated, so ties make fewer
  bins), each proportion floored at `1e-4` so an empty bin is finite, `sum((c - r) * ln(c / r))`.
- Neither is computed when either side has fewer than `--min-values` values (default 30, a display choice, said
  to be one in the help text). Neither has a built-in limit; `--fail-on-psi` and `--fail-on-ks` set the exit
  status only when given.

**The reference.** `--reference` takes a `calibrate --ndjson` file, read with the existing `events::read`. Only
pairs labeled `edit` are used, and the report says so in its header. A reference whose fingerprint differs from a
window's is not compared with it.

**Output.** A text report for a person by default, written to a locked stdout with `writeln!` like the
calibration report, and with `--ndjson` the same numbers as flat result events: `drift` (header: files, lines,
requests, incomplete, reference), `drift_window`, `drift_score`, `drift_field`, `drift_quantity`, `drift_reason`,
`drift_unit`, `drift_tenant`, `drift_rejudged`, each carrying `fingerprint` and `window`. These names become an
interface on the same terms as D17's.

**Where.** `offline/traffic.rs` (the reader: lines to `Request { meta, evaluation }`, reusing the parsing
helpers from `events.rs`, made `pub(super)`), `offline/drift.rs` (windows, summaries, rejudging), the statistics
in `stats.rs`, rendering beside `report.rs`, the subcommand in `main.rs`, and a `just drift *args` recipe.

### Part 3: Prometheus (deferred by recommendation)

If the owner wants it now: a `[server] metrics_listen` address, off when absent; counters of verdicts by mode,
failures by code; fixed-bucket histograms of `total_us` and of overall similarity; a gauge of jobs waiting in the
worker (an `AtomicUsize` incremented on `try_send` and decremented when a batch is taken) and a histogram of batch
sizes. Served by a hand-written `/metrics` handler in the Prometheus text format on `axum`, which is already
compiled into the binary through `tonic`, so `cargo deny` sees no new crate. The alternative, `metrics` plus
`metrics-exporter-prometheus`, is MIT-licensed but adds a dependency tree that has to be run through `just deny`
before it is chosen. Histogram buckets for similarity are not thresholds, but they should be placed from a
calibration's ranges, not guessed.

## The work, in order

Each step is one commit, and each passes `just ci`.

1. **The engine keeps its reasons.** `EvaluateOutput.reasons: Vec<Reason>`; `grpc.rs` renders the sentences.
   No wire change. (If feature 19 lands first, this step is already done.)
2. **The engine reports what it did and how long it took.** `texts_embedded`, `baseline`, `timings`. Engine tests
   for `texts_embedded` and `baseline`.
3. **Shared event helpers.** Move `six`, `emit_schema`, and the change names where both emitters can use them.
   The calibration's tests must pass unchanged.
4. **Telemetry events.** `telemetry.rs` (request ids, `server`, `evaluate` and its items, `evaluate_failed`);
   `grpc.rs` calls it on success and failure; `lib.rs::serve` (or `main.rs`) writes the `server` lines at start.
   Tests with `capture`. Update the README's logging paragraph and example in the same commit, since the example
   line changes stream and spelling.
5. **The traffic reader.** `offline/traffic.rs`, with a round-trip test against what step 4 writes.
6. **Statistics.** `quantile`, `ks`, `psi`, and the ISO-week arithmetic in `stats.rs`, with example tests and
   properties.
7. **`drift`.** Windows, per-field and per-quantity summaries, reasons, tenants, units, the reference, the text
   report, `--ndjson`, the gates, the CLI, and the `just` recipe.
8. **`drift --config-b`.** Rebuilding evaluations and judging them again.
9. **Documentation**: the README section, D22, `CLAUDE.md`, the plan-features note, and the rollout paragraph
   per decision 3. Run every README example before quoting it.
10. **Optional, later: the metrics endpoint** (part 3), only on the owner's say-so.

After step 2 (the engine's write path is touched only to time it), run `PROPTEST_CASES=5000 just test` once.

## Tests

None needs the model, the network, or Qdrant. All use `WordBagEmbedder`, `MemoryStore`, `tiny_embedder`, and
`similarity_log::testing::capture`. Names state the behavior.

Engine (`engine.rs`):

- `a_resubmitted_record_sends_no_text_to_the_model`
- `a_revision_reports_the_revision_and_age_of_the_baseline_it_was_compared_with`
- `a_record_without_a_baseline_reports_no_baseline`
- `a_rejected_revision_returns_its_reasons_as_values_not_sentences` (shared with feature 19 if it lands first)
- `nothing_is_timed_as_written_when_nothing_was_written`

Telemetry (`telemetry.rs`):

- `an_evaluation_is_written_as_flat_result_lines_even_when_logs_are_off`
- `every_item_line_carries_the_request_id_of_its_evaluate_line`
- `the_counts_on_the_evaluate_line_match_the_item_lines_that_follow`
- `similarities_are_rounded_to_six_places_like_every_other_event`
- `the_verdict_and_mode_are_written_in_lowercase`
- `no_field_text_appears_anywhere_in_the_telemetry` (evaluate a record with distinctive text, assert no captured
  line contains any of its words)
- `a_unit_longer_than_the_cap_is_cut_on_a_character_boundary`
- `an_unknown_unit_is_logged_with_unit_known_false`
- `a_request_that_fails_writes_one_evaluate_failed_line_with_its_code_and_no_message`
- `request_ids_from_one_server_do_not_repeat`
- `the_field_lines_of_a_request_add_up_to_its_similarity_under_their_weights` (mirrors the calibration test)

gRPC (`grpc.rs`, driving `GrpcService::evaluate` on a current-thread runtime inside `capture`):

- `the_response_still_carries_reasons_as_sentences`
- `a_threshold_given_in_the_request_is_logged_as_overridden`

Traffic and drift (`offline/traffic.rs`, `offline/drift.rs`, `offline/stats.rs`):

- `telemetry_reads_back_as_the_evaluations_that_were_written`
- `interleaved_requests_are_put_back_together_by_request_id`
- `a_request_whose_lines_are_cut_off_is_counted_as_incomplete_and_not_guessed_at`
- `lines_that_are_not_result_events_are_skipped`
- `unchanged_fields_take_no_part_in_a_field_distribution`
- `windows_never_compare_two_model_fingerprints`
- `a_timestamp_falls_in_the_iso_week_of_its_utc_date` (with the year boundary: 2026-12-31 and 2027-01-01)
- `only_edit_labeled_pairs_of_a_reference_calibration_are_used`
- `psi_and_ks_are_left_out_below_the_minimum_sample_size`
- `there_is_no_drift_gate_unless_one_is_asked_for`
- `units_the_table_does_not_know_are_listed_with_their_counts`
- `judging_traffic_again_under_a_stricter_threshold_reports_the_new_rejections`
- `judging_traffic_again_under_other_field_names_is_refused`
- `the_text_report_and_the_events_carry_the_same_numbers`

Properties (`similarity-server`'s `properties` module, D19):

- `ks_is_symmetric_and_between_zero_and_one`
- `ks_of_a_sample_against_itself_is_zero`
- `psi_of_a_sample_against_itself_is_zero` and `psi_is_never_negative`
- `judging_logged_traffic_again_under_the_configuration_that_produced_it_changes_no_verdict` (drive the existing
  engine state machine, capture its telemetry, read it back, rejudge, compare: this is what proves the logged
  measurements are enough to reproduce a verdict, including anchor drift and quantity limits)

Nothing here needs `#[ignore]`. The real-model and Qdrant paths are untouched apart from timing around calls.

## Documentation to update

- **README.md.** The logging paragraph and the example `evaluate` line under "Running the server" (stream, verdict
  spelling, rounding; and `RUST_LOG=warn` no longer quiets them). A new "Production telemetry" section under
  Operations with the event table above, a `jq` example, and the `drift` command, in the style of "The calibration
  as data". The command table gains `drift`. "Rolling it out" changes per decision 3. Any example output quoted
  must be produced by running it (on the bundled sample traffic generated with `just serve-in-memory` and the
  client, not invented).
- **docs/decisions.md.** A D22, "What the server decided is output, and never carries text": why results and not
  logs (D15's argument), the flat shape and request id (D17's), what is never logged and why units are cut, why
  no drift limits ship, why windows never cross fingerprints, and the rollout problem with `DRY_RUN`. D15 gains a
  sentence that the server's per-request records are results. D7 gains a sentence if decision 3 adds a mode.
- **CLAUDE.md.** Under "Things that are easy to get wrong": telemetry events are an interface like the calibration
  events (add fields, never rename or repurpose); never log record text or anything derived from it; drift
  statistics live in `offline::stats`. Under Commands: `just drift`. The `similarity-server` line in the layout
  gains `telemetry` and `drift`.
- **docs/plan-features.md.** The row for feature 10 and the "still open" note about the unrounded line.
- **docs/feature-suggestions.md.** Mark feature 10 built when it is, with what came out differently.

## Dependencies on other features

- **Feature 19 (reason codes on the wire).** Shares step 1. Interface assumed: `EvaluateOutput.reasons:
  Vec<similarity_core::score::Reason>`, rendered to strings only in `grpc.rs`. Whichever lands first does it.
- **Feature 18 (shadow evaluation).** Not required. If built, its `shadow` event should reuse these shapes with
  the same `request_id`, so that `drift --config-b` and a shadow comparison read alike.
- **Feature 17 (reviewer feedback).** Not required. A feedback event that carries the `request_id` of the
  `evaluate` it answers would join a label onto logged traffic; worth asking for when 17 is designed.
- **Feature 12 (latency).** Not required. The timings here are the per-stage numbers a load test would want.
- **Feature 2 (run registry).** Not required; nothing here writes run directories.
- No new crate is needed for parts 1 and 2.

## Decisions for the owner

1. **Is the telemetry a result (stdout, never filtered) or a log (stderr, under `RUST_LOG`)?**
   Recommended: a result. It is what the server decided, the dataset the rollout depends on, and D15 already
   ruled that output must not be thinned by a log level. The cost is that `RUST_LOG=warn` no longer quiets it and
   the deployment has to collect stdout as well as stderr.
2. **Log the tenant as given, hashed, or not at all?**
   Recommended: as given. It is an identifier, not record content, the record `id` is already logged, and
   "which brand is behind the rejections" is half the point. Hashing (a keyed SHA-256 prefix) is a small change if
   the log pipeline's retention rules forbid merchant ids.
3. **What should the rollout run in?** `DRY_RUN` never moves the baseline (problem 8).
   Recommended: change the README's guidance to run in `ENFORCE` with the verdict ignored by the caller. No code;
   accepted edits advance the baseline as they will in production, and only records the service would have
   rejected keep a stale one, which `baseline_age_s` shows. The cleaner alternative is a new `WRITE_MODE_OBSERVE`
   that judges and always writes as an ordinary accepted write would (anchor values never replaced, never reset);
   that is a proto change, an amendment to D7, and new cases in the state machine, so it belongs in its own plan.
4. **Prometheus now or later?**
   Recommended: later, or never. Part 1's events give a log pipeline every count and distribution the endpoint
   would. If wanted, hand-write it on the `axum` already in the build rather than adding `metrics` crates.
5. **Change the existing `evaluate` line's verdict spelling (`Created` to `created`)?**
   Recommended: yes. The line moves to another stream anyway, so every consumer is touched once; doing it later
   would touch them twice.
6. **Log a caller's `x-request-id`?**
   Recommended: yes, cut to 64 bytes, so a rejection can be found from the caller's own logs.

## What can only be validated at work

Everything in this plan can be written and tested here on synthetic traffic. What cannot:

- Whether the per-field distributions of real edits look like the calibration's edits, and therefore whether PSI
  and KS against the reference say anything.
- How many lines a day the stream produces, and whether decision 1's volume is acceptable to the log pipeline.
- Which tenants dominate rejections, and whether that is recycling or one brand's editing habits.
- Which units occur that the table does not know.
- Any limit for `--fail-on-psi` or `--fail-on-ks`: those come from watching real weeks, as thresholds come from
  `calibrate`, and none should be written into the repository until then.

## Size

**Medium.** Part 1 is small (an engine type change shared with feature 19, three measured numbers, one emitting
module, about fifteen tests). Part 2 is the bulk: a streaming reader, three statistics, windowing, a report in two
renderers, and the rejudging, most of which reuses `events.rs` and `Evaluation::rejudge`. Part 3 would add a
small-to-medium piece and is recommended out.
