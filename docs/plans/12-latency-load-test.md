# Plan: latency tracking and a load test (feature 12)

A plan for measuring what the service does under concurrent requests, written so that a session with no memory of
the conversation that produced it can carry it out. Nothing here is built. Read `CLAUDE.md` first, then D9, D15,
D17, and D20 in `docs/decisions.md`.

## Contents

- [What it is for](#what-it-is-for)
- [What exists today](#what-exists-today)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Which command yields which figure](#which-command-yields-which-figure)
- [Documentation to update](#documentation-to-update)
- [Dependencies on unbuilt features](#dependencies-on-unbuilt-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

Every performance figure in the README is a forward pass measured on its own (`just throughput`) or an import
measured once by hand. Nothing measures a request the way a caller sees it: through tonic, the engine, the
per-id lock, the store, and the single embedding thread's queue (D9), with other requests arriving at the same
time. That is the figure that sizes a deployment ("how many edits per second does one replica take, and at what
p99?"), and it is the one the README still lacks.

This feature adds three things:

1. **A load generator**, `record-similarity load`, in Rust, on synthetic records. It drives concurrent `Evaluate`
   calls over a real gRPC socket and reports latency percentiles, requests and texts per second, the
   `RESOURCE_EXHAUSTED` count, and (when it runs the server itself) how the worker actually merged requests into
   batches.
2. **Counters in the embedding worker** (`WorkerStats`), so that batch sizes, busy time, and queue wait are
   observable at all.
3. **Per-request timing on the server's request log lines** (`elapsed_ms`), the smallest piece of production
   latency tracking, which today does not exist.

## What exists today

Verified against the tree at `545ef20`:

- `crates/similarity-server/src/worker.rs`: `EmbedWorker::spawn(embedder, queue_depth, max_batch_texts)` owns one
  thread; `run` takes the first `Job` with `recv`, then `try_recv`s more while the text count is below
  `max_batch_texts`, and calls `serve`, which calls `serve_with(&|texts| embedder.embed(texts), batch)`.
  `TextEmbedder::embed` maps `TrySendError::Full` to `EmbedError::Overloaded`. The worker records nothing: no
  counter, no timing, and the only log lines are a `debug` on stop, an `error` on a backend panic, and a `warn`
  when a merged batch is retried. `queue_depth` bounds the number of **requests** waiting, not texts.
- `crates/similarity-server/src/main.rs:313`: the server passes `config.model.batch_size` (32 in
  `config.example.toml`) as `max_batch_texts`, so a merged batch is about 32 texts: eight whole records, or 32
  one-field edits. The cap is checked before each `try_recv`, so a batch can overshoot it by one request.
- `crates/similarity-server/src/engine.rs`: `Engine::evaluate` takes the per-id stripe lock (`lock_for`) **before**
  embedding and holds it across the store read, the `embed` await, and the write. `Engine::embed` sends only the
  texts `PreparedRecord::plan` says changed, so a create sends every field, a one-field edit sends one text, and a
  re-save sends none and never touches the worker.
- `crates/similarity-server/src/grpc.rs`: the `evaluate`, `find_similar`, and `delete` log lines carry the id,
  verdict, similarities, revision, and `measure_changed`. **No timing of any kind.** Failures are logged only
  for `Unavailable` and `Internal`.
- `crates/similarity-server/src/lib.rs`: `serve(engine, listener, shutdown)` takes a bound `TcpListener`, so a
  caller can bind port 0. Its own tests and `tests/real_e2e.rs` already start an in-process server this way.
  `TcpIncoming` sets `TCP_NODELAY`.
- `crates/similarity-core/examples/throughput.rs`: loads the model and times four batch shapes with the median of
  20 runs, printing text. No fingerprint, commit, CPU, or thread count. `just throughput` runs it.
- `crates/similarity-server/examples/client.rs`: a `current_thread` tool that sends one request and prints the
  answer. Not a base for a load generator.
- `crates/similarity-server/src/import.rs`: `ImportLine { id, tenant, record: RecordFields, anchor }` is public
  and `RecordFields` serializes to the shape it is read from. `import::check` validates an export with no model.
- `similarity_core::rng::Rng` (SplitMix64) gives a sequence fixed by its seed on every platform.
- `similarity_core::provenance::Build` holds the version, commit, and dirty flag; `main.rs` fills one `BUILD`
  constant from `build.rs` (`RECORD_SIMILARITY_BUILD_COMMIT`, `RECORD_SIMILARITY_BUILD_DIRTY`), named for
  `model fetch`.
- `similarity_log`: `RESULT` target to stdout, `failed`, and a `testing::capture` helper that tests use to read
  what was emitted. `offline/events.rs` is the model for flat result events (D17).
- `similarity_core::testing::tiny_embedder` gives a real `Embedder` with tiny random weights, so an `EmbedWorker`
  can be spawned in `cargo test` without the model.
- tonic 0.14.9 passes `max_concurrent_streams: None` to hyper (`tonic/src/transport/server/mod.rs`), which
  overrides hyper's default of 200. One HTTP/2 connection therefore does not cap the concurrency a test can apply.
- README figures today: the `## Performance` table (4, 12, 20, 86 ms from `just throughput`), and "2,000
  four-field records were imported in 19 seconds" (`## Integrating it`, also in D20). No recipe reproduces the
  second one: it was an ad hoc export.

## What the suggestion got wrong or left open

1. **"Bootstrap and re-index time estimates rest on that figure."** No longer true. Since feature 16, bootstrap
   and re-index go through `import`, which uses `DirectEmbedder` on the calling thread and never the queue (D20),
   and the README quotes a measured import time. The queue figure now matters for edit-time capacity, and for a
   caller who bootstraps through `Evaluate` because they cannot run `import` next to Qdrant.
2. **"Reports batch sizes."** A client cannot see them. The worker counts nothing, and no RPC reports it. The
   load tool can only report batch sizes if it runs the server in its own process and reads counters the worker
   does not have yet. Against an external server it can report only what a client sees.
3. **"The `RESOURCE_EXHAUSTED` rate."** At the default `queue_depth` of 256 requests, a test with fewer than
   roughly 256 requests in flight never sees one. The rate is a figure only above the queue depth, or under a
   config with a smaller queue. The plan measures it deliberately at one concurrency above the depth.
4. **`throughput` into "the run registry" and "`runs compare` flags a slowdown."** Neither exists (features 2 and
   5 are unbuilt), and `throughput` writes text with nothing that identifies the machine, build, or model. The
   event shape can be designed now; the registry and the comparison cannot be.
5. **Fit: `examples/client.rs`.** Wrong home. It is a one-request `current_thread` tool. The load generator is a
   module of `similarity-server` and a subcommand of the binary (see decision 1).
6. **The per-id lock is held across embedding.** A load test that reuses ids measures lock contention, not the
   queue. Every concurrent request in a phase must name a different id. Not mentioned by the suggestion.
7. **The `iai-callgrind` gate** needs valgrind and Linux, the owner works on macOS, and the scoring path is
   microseconds next to a forward pass of milliseconds. It would add a dev-dependency and a CI job to guard the
   cheap part. Left out (decision 5).
8. **Production latency tracking was not in the suggestion at all**, though its title says "latency tracking".
   The request log lines carry no time. Adding `elapsed_ms` is the one production-side piece here.

## Design

### Where the code lives

```text
crates/similarity-server/src/
  worker.rs          + WorkerStats, EmbedWorker::stats, run_with (the loop over a closure, for tests)
  grpc.rs            + elapsed_ms on the evaluate, find_similar, and delete lines
  load/mod.rs        Options, Target, run(): phases, levels, the closed loop
  load/catalog.rs    the synthetic catalog: records, edits, ids; reading an export instead
  load/stats.rs      Latencies, PhaseStats, percentiles; every number in the events comes from here
  load/events.rs     the `load_run` and `load_phase` result events
  main.rs            + Command::Load(LoadArgs)
justfile             + load recipe
```

A subcommand and not an example, so that the tool is in the container image and can be run in the owner's work
environment against a staging replica without a Rust toolchain there, and so that its logic has ordinary unit
tests (an example's are not run by `cargo test`).

### Worker counters

```rust
/// What the embedding thread has done since it started. Counters only grow; subtract two snapshots for a window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerSnapshot {
  pub batches: u64,
  pub jobs: u64,
  pub texts: u64,
  /// Requests refused because the queue was full.
  pub overloaded: u64,
  /// Time spent inside the model, including a retry after a failed merged batch.
  pub busy_ns: u64,
  /// Summed over jobs: from `try_send` to the moment the worker took the job.
  pub queue_wait_ns: u64,
  /// Batches by texts merged: 1, 2 to 4, 5 to 8, 9 to 16, 17 to 32, 33 and more.
  pub batches_by_texts: [u64; 6],
}
```

`EmbedWorker` holds an `Arc<WorkerStats>` of `AtomicU64`s (relaxed ordering; these are statistics, not
synchronization), shared with the thread. `Job` gains `queued: Instant`. `EmbedWorker::stats() -> WorkerSnapshot`
reads them; `WorkerSnapshot::since(&earlier)` subtracts with `saturating_sub`. Bucket bounds follow the default
`batch_size` of 32 and are a display choice, not a limit. The counters cost a few atomic adds per batch, beside a
forward pass of milliseconds. Nothing is logged by the worker per batch: a line per batch under load is the
volume D15 warns about, and the counters make it unnecessary.

`run` becomes `run_with(embed: &F, jobs, max_batch_texts, stats)` generic over the same closure type `serve_with`
already takes, with `run` a one-line wrapper. That lets a test gate the "model" on a channel and fill the queue
deterministically, which is the only reliable way to test the overload count.

### Request timing on the log lines

`grpc.rs` takes an `Instant` at the top of `evaluate`, `find_similar`, and `delete` and adds
`elapsed_ms = hundredths(started.elapsed())` (milliseconds, two decimals, a number) to the existing success line.
No new line, no decision: `grpc.rs` stays free of them. The failure path is left as it is (decision 4). This is
the line feature 10 also touches, to round the similarities on it; whichever lands first should do both.

### The load tool

```text
record-similarity load [--config config.toml]
  [--server URL]                   drive a running server instead of starting one in this process
  [--qdrant-collection NAME]       in-process only: use Qdrant at the configured url, in this collection
  [--concurrency 1,4,16,64]        levels, run in this order, one set of phases each
  [--records 1000]                 records per level
  [--phases create,edit,resave]    which phases, in order
  [--edit-field name]              the field each edit changes
  [--seed 1]                       the synthetic catalog
  [--records-from export.jsonl]    take records from an export (the import format) instead of generating them
  [--write-export FILE]            write the synthetic catalog as an import export and stop
  [--keep]                         leave the records in the store afterwards
```

**Target.** Without `--server`, it loads the model from the config, spawns an `EmbedWorker` with the config's
`queue_depth` and `batch_size`, builds an `Engine` over a `MemoryStore` (or a `QdrantStore` when
`--qdrant-collection` is given), binds `127.0.0.1:0`, and serves with `similarity_server::serve` on the same
runtime. Client and server then share the machine, as they do when a person runs `just serve` and a client on one
machine; the client's share is a few microseconds per request beside milliseconds of model time. With `--server`
it reports only what a client can see, and the worker fields are left out of the events, not written as zero
(D15). Qdrant is never used without a collection named on the command line, so a load test cannot write into the
configured production collection by default.

**Records.** `catalog.rs` generates records from word lists and `Rng`: a name from strain, form, and strength, a
category from a fixed tree, a brand, and a description of two to five template sentences (so descriptions vary in
length the way the README's sample does), plus the quantities the schema defines with plausible units. Only
fields and quantities the loaded schema has are filled; if a required field is not one the generator knows, the
command stops with an error naming `--records-from`. Ids are `load-{run}-c{level}-{n}` under tenant `load-test`,
where `run` is the seed plus the start time in seconds, so two runs and two levels never share an id. An edit
changes exactly one field (`--edit-field`, `name` by default, "the usual edit" of the README table) in a way the
shipped threshold accepts most of the time: a suffix such as " - Indica" or a corrected strength. The verdicts are
counted and reported, not asserted: what matters is the cost, and a rejected edit costs the same model time and
skips one store write.

`--records-from` reads `ImportLine`s, so the owner can run the same measurement on a real export at work. Edits are
then mechanical (a suffix appended to `--edit-field`). No record text appears in any event, only counts, so the
results of such a run can leave the work environment without carrying product data.

**Phases.** Per concurrency level, after a warm-up of 32 requests that are not counted:

- `create`: one `Evaluate` per record, default mode. Every field goes to the model. This is the caller who
  bootstraps through the API.
- `edit`: one `Evaluate` per record with `--edit-field` changed. One short text per request. This is edit-time
  traffic, the figure that sizes a replica.
- `resave`: the edited record sent again unchanged. No text reaches the model; what is left is gRPC, the lock, and
  the store. It shows the floor under the other two.

Then, unless `--keep`, every record is removed with `Delete` (not timed).

**Closed loop.** `concurrency` tasks share one `AtomicUsize` index over the phase's records; each sends a request,
waits for the reply, records it, and takes the next index. All share one tonic `Channel` (enough, see above). A
request refused with `RESOURCE_EXHAUSTED` is counted and sent again after 5 ms (a tool setting, not a policy), and
its latency is measured from the **first** attempt, which is what a caller with a retry loop experiences. Other
errors are counted by code and not retried. Latency at a fixed concurrency is what this measures; latency at a
fixed arrival rate (open loop, free of coordinated omission) is decision 3.

**Statistics.** `stats.rs` keeps every successful request's latency in a `Vec<Duration>`, sorted once per phase.
Percentiles are nearest rank (`ceil(p * n)`th smallest), which is exact and needs no new dependency; a phase is at
most a few hundred thousand values. A phase with no successful request reports no percentiles at all. Rates are
counts over the phase's wall time, never over summed latency. Like `offline::stats` for calibration, every number
in an event is computed here and nowhere else.

### Events

Results on stdout through `similarity-log` (`target: RESULT`), flat, as D17 requires. Logs (progress per level)
on stderr. Two event names, which become an interface like the calibration events: add fields, never rename.

`load_run`, once at the start:

| Field | Meaning |
|---|---|
| `run_id` | Joins the phase lines. |
| `version`, `commit`, `dirty` | From the binary's `Build` (a second constant, `record-similarity load`). |
| `os`, `arch`, `threads`, `rayon_threads` | `std::env::consts`, `available_parallelism`, `RAYON_NUM_THREADS` if set. |
| `target` | `in_process` or `external`. |
| `store` | `memory` or `qdrant` (in process only). |
| `model_fingerprint`, `dimension` | From the engine, or from `Describe` when external. |
| `device`, `batch_size`, `queue_depth`, `max_tokens` | From the config (in process only). |
| `seed`, `records`, `records_from_export` | What was sent. |

The CPU model name is not in `std`. It is left out rather than read from a subprocess; whoever keeps a run file
writes the machine down beside it (decision 6).

`load_phase`, once per level and phase:

| Field | Meaning |
|---|---|
| `run_id`, `concurrency`, `phase` | Which. |
| `requests`, `ok`, `overloaded`, `failed` | `overloaded` counts refused attempts, so it can exceed `requests`. |
| `created`, `accepted`, `rejected` | Verdicts of the `ok` requests. |
| `field_texts` | Texts the client's requests asked to embed (all fields of a create, one per edit, none per re-save). |
| `elapsed_s` | Wall time of the phase. |
| `requests_per_s`, `texts_per_s` | Over `elapsed_s`. |
| `p50_ms`, `p90_ms`, `p99_ms`, `max_ms` | Absent if `ok` is zero. |
| `worker_texts`, `worker_jobs`, `batches` | In process only, from the counters for this phase. |
| `mean_texts_per_batch`, `max_bucket` | In process only. |
| `batches_1`, `batches_2_4`, `batches_5_8`, `batches_9_16`, `batches_17_32`, `batches_33_up` | In process only. |
| `busy_fraction` | Worker time in the model over `elapsed_s`. Near 1 means the encoder is the bottleneck. |
| `mean_queue_wait_ms` | In process only. |

Rounding: milliseconds to two decimals, rates to one, fractions to three. `worker_texts` equal to `field_texts` in
process is a check that the client's idea of what changed matches the engine's.

### The recipe

```make
# Concurrent requests through gRPC and the embedding queue: `just load`, or `just load config.toml --server URL`
load config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- load --config {{ config }} {{ flags }}
```

## The work, in order

One commit each.

1. **Worker counters.** `WorkerStats`, `WorkerSnapshot`, `Job::queued`, `run_with`, `EmbedWorker::stats`. Tests
   below. No behavior change.
2. **Timing on the request log lines.** `elapsed_ms` in `grpc.rs`, with a test through `similarity_log::testing`.
3. **The synthetic catalog and `--write-export`.** `load/catalog.rs` and the subcommand skeleton that only writes
   the export. Measure `just import` on a 2,000-record export into Qdrant to give the README's import figure a
   command that reproduces it; replace the figure if it moved.
4. **Statistics and events.** `load/stats.rs`, `load/events.rs`, with unit tests and a property.
5. **The closed loop, in process and external.** `load/mod.rs`, `LoadArgs`, the recipe, and the end-to-end tests on
   the tiny encoder.
6. **Measure and document.** Run the commands in
   [Which command yields which figure](#which-command-yields-which-figure), write the README tables, add D22, update `CLAUDE.md`, `docs/plan-features.md`, and the feature-suggestions
   "Things noticed" entry. `just ci`.
7. **Optional: `--records-from`**, if the owner wants to run it on a real export at work (decision 2). Small, since
   `ImportLine` exists.

## Tests

All run under `cargo test` with no model, no network, and no Qdrant, unless marked.

Worker (`worker.rs`):

- `the_worker_counts_every_batch_job_and_text_it_serves`
- `requests_waiting_together_are_counted_as_one_batch` (gated closure: hold the model, queue three jobs, release)
- `a_full_queue_refuses_at_once_and_the_refusal_is_counted` (gated closure, `queue_depth` 1)
- `queue_wait_is_measured_from_when_a_request_was_queued`
- `a_snapshot_minus_an_earlier_one_is_what_happened_between_them`
- `a_retried_merged_batch_counts_its_model_time_but_not_extra_texts`

Server log line (`grpc.rs` or `lib.rs` tests):

- `every_successful_request_logs_its_elapsed_time_in_milliseconds_as_a_number`

Catalog (`load/catalog.rs`):

- `the_synthetic_catalog_is_the_same_for_the_same_seed`
- `every_synthetic_record_is_valid_under_the_example_config` (prepares each with `config.example.toml`'s schema)
- `an_edit_changes_exactly_one_field_and_nothing_else`
- `no_two_levels_or_runs_share_an_id`
- `a_schema_with_a_required_field_the_generator_does_not_know_is_refused_with_a_pointer_to_records_from`
- `the_written_export_passes_import_check_with_no_failures`

Statistics and events (`load/stats.rs`, `load/events.rs`):

- `percentiles_are_the_nearest_rank_of_the_sorted_latencies` (1 to 100 ms: p50 is 50, p99 is 99)
- `a_phase_with_no_successful_request_reports_no_percentiles_rather_than_zero`
- `rates_are_counts_over_wall_time_not_over_summed_latency`
- `the_load_events_are_flat_and_every_phase_line_carries_the_run_id`
- `an_external_run_leaves_the_worker_fields_out_instead_of_writing_zero`

Property (`similarity-server`'s `properties` module, D19):

- `percentiles_never_decrease_with_the_rank` (any latencies: p50 <= p90 <= p99 <= max)
- `a_merged_batch_exceeds_the_text_cap_by_at_most_one_request` (any job sizes through `run_with`)

End to end, in process, tiny encoder through a real `EmbedWorker` and `MemoryStore` (`load/mod.rs`):

- `a_load_run_accounts_for_every_request_it_sent` (`requests == ok + failed`, `created == records` in `create`)
- `in_process_the_worker_embeds_exactly_the_texts_the_client_changed` (`worker_texts == field_texts`; zero for
  `resave`)
- `a_queue_smaller_than_the_concurrency_is_refused_and_every_record_still_lands` (`queue_depth` 1, concurrency 16:
  `overloaded > 0` is not asserted, since the tiny encoder may drain fast enough; that every record is stored is)
- `the_load_records_are_deleted_afterwards_unless_kept`

CLI (`main.rs`):

- `load_defaults_to_an_in_process_server_on_memory`
- `load_never_uses_qdrant_without_a_named_collection`

Ignored, real model:

- `#[ignore = "needs the model: just fetch-model, then just test-real"]`
  `under_concurrent_edits_the_real_model_merges_requests_into_batches` in `tests/real_e2e.rs`, in process on
  `MemoryStore`, concurrency 32: `mean_texts_per_batch > 1`. This is D9's claim, not a threshold.

Not tested: the absolute numbers. Wall time is too noisy to assert (see decision 5).

## Which command yields which figure

All on the README's machine (an M4 Pro Mac mini, release build, default model), every figure read from the
`load_phase` lines with `jq`. Nothing is quoted that one of these did not print.

| Figure | Command |
|---|---|
| Edits per second and p50/p99, per concurrency, encoder only | `just load` |
| The same through Qdrant (the README's headline) | `just qdrant-up`, then `just load config.example.toml --qdrant-collection load-test` |
| Creates per second through the API (bootstrap by `Evaluate`) | the `create` lines of the run above |
| Mean texts per batch and busy fraction at each level | the same lines |
| Floor cost of a request that embeds nothing | the `resave` lines |
| `RESOURCE_EXHAUSTED` rate above the queue depth | `just load config.example.toml --concurrency 512 --phases edit` |
| Same figures against a separately started server | `just serve-in-memory`, then `just load config.example.toml --server http://127.0.0.1:50051` |
| The import figure, reproducibly | `record-similarity load --write-export target/catalog.jsonl --records 2000 --concurrency 1`, then `just import target/catalog.jsonl` |

Only the owner's work environment can validate the figures that matter for production: real field lengths (real
descriptions may be longer than the synthetic ones, and cost grows with the square of tokens), the real share of
creates, edits, and re-saves, the production CPU and container limits, and Qdrant over a real network. Run
`record-similarity load --server <staging> --records-from <export>` there (decision 2). Those numbers stay there,
or come back as event files with no record text in them.

## Documentation to update

- **README.md**
  - `## Performance`: keep the forward-pass table, add "Under concurrent requests" with edits per second, p50, and
    p99 at each concurrency level, creates per second, mean texts per batch, and the overload rate at 512, each
    with its command. Say which machine.
  - `### Capacity`: point to `just load` for sizing; say that `queue_depth` counts requests and that refusals start
    only above it.
  - `## Integrating it`: the `Evaluate`-by-record bootstrap paragraph can quote the measured creates per second;
    the import figure gets its reproducible command.
  - `## Development`: add `just load` to the command list.
  - `## Operations`: the request log lines now carry `elapsed_ms`.
- **docs/decisions.md**: D22, "Latency is measured through the queue". Why a subcommand (it ships with the image);
  why in process by default (only then can batches be seen) and never Qdrant without a named collection; closed
  loop, and what it does not measure; nearest-rank percentiles with no dependency; why no CI gate on wall time or
  instruction counts. D9 gains one line pointing to the counters. D20's import figure gets its command.
- **CLAUDE.md**: `just load` under Commands; in "Every number in README.md" name `just load` beside `just
  throughput` and `just calibrate`; one bullet that the `load_run` and `load_phase` event names and fields are an
  interface, as calibration's are.
- **docs/plan-features.md**: row 12 moves to "What was built". **docs/feature-suggestions.md**: mark 12 built and
  update the "Things noticed" entry about the unmeasured queue figure.

## Dependencies on unbuilt features

- **Feature 2 (run registry).** Assumed interface: a run is a directory holding the NDJSON result lines a command
  wrote to stdout, plus a manifest. `load` needs nothing from it; its `load_run` line carries the build and
  machine fields a manifest would want. Until then, results are kept with `> load.ndjson`.
- **Feature 5 (run compare).** Assumed to read result lines by event name and join on fields. A latency comparison
  would join `load_phase` lines on `(concurrency, phase)`. Not built here.
- **Feature 10 (production telemetry).** Shares the `evaluate` log line. Assumed to add fields to it, not replace
  it; `elapsed_ms` is one such field. It is also where periodic `WorkerSnapshot` lines from a running server would
  belong (decision 4).

## Decisions for the owner

1. **Subcommand or example.** Recommended: a `load` subcommand of `record-similarity`, so it ships in the image
   and can be run at work against staging without a toolchain. The alternative is an example in
   `similarity-server/examples`, which keeps the release binary smaller but untested by `cargo test`.
2. **`--records-from` an export.** Recommended: yes, as the last step. It is the only way to measure with real
   field lengths, and it can only be run at work. Its events carry counts, never text.
3. **Open loop (`--rate`).** Recommended: not now. Closed loop answers "what does one replica sustain"; open loop
   answers "what is p99 at a given arrival rate" without coordinated omission, and is worth adding once feature
   10 gives a real arrival rate to aim at.
4. **Production timing beyond `elapsed_ms`.** Recommended: only `elapsed_ms` on success lines now. Timing on
   failure lines and a periodic worker-counter line from the server belong with feature 10, which owns
   production telemetry.
5. **Regression gating.** Recommended: no CI gate. Wall time on shared runners is noise, and an instruction-count
   gate (`iai-callgrind`) needs valgrind on Linux and guards the scoring path, which is microseconds beside the
   model. Compare `load_phase` files by hand, then with feature 5. `just throughput` stays text, as the README
   quotes it; giving it a `--ndjson` mode can wait for feature 2.
6. **CPU name in `load_run`.** Recommended: leave it out (std cannot read it, and a subprocess per platform is
   more code than the field is worth); record the machine beside the file.

No new dependency. Percentiles are a sorted vector, counters are `std::sync::atomic`, the client is the tonic
already in the workspace (with `channel` on), and the catalog uses `similarity_core::rng`. `cargo deny` sees no
change.

## Size

**Medium**, a few days. The worker counters and the log field are a day between them; the load module (catalog,
closed loop, statistics, events) and its tests are two; measuring on the development machine and against Qdrant, and writing
the README and D22 to match, is most of another. It would be small without the in-process target, but without it
batch sizes, the one thing a client cannot see, could not be reported.
