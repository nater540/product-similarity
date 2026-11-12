# Plan: shadow evaluation (feature 18)

A plan for running a candidate beside the live configuration on real traffic, written so that a session with no
memory of the conversation that produced it can carry it out. Read `CLAUDE.md` first, then D7, D8, D9, D12, D15,
D17, D20, and D30 in `docs/decisions.md`, then this file.

**Status: the policy half is built (D30); the model half is not.** What exists differs from steps 1 and 2 below
in these ways, and the model half has to start from what exists:

- The seam is `engine::Observer`, with one method. `Observation` borrows (`id`, `mode`, the request's
  `threshold`, the live `policy`, `stored`, `revision`, and `judged`: the baseline, the candidate's vectors and
  quantities, the live evaluation). A model shadow that queues work has to copy what it keeps, and needs the
  prepared texts added to the observation. There is no `deleted` callback yet.
- `next_anchor` was not extracted; the mirror write will want it.
- `shadow_reason` says `subject`, and a policy shadow writes no `shadow_field` or `shadow_quantity` lines: both
  sides measured the same values. `in_sync`, `missing`, `queued_ms`, and `embed_ms` belong to the model shadow and are not written.
- `events = "disagreements"` is the default; `"all"` writes every judged revision.
- `[shadow.model]` and `[shadow.qdrant]` are parsed only to be refused with the reason. There is no `queue` key.
- Not built: `shadow-report`, the shadow in `Describe`, `import --shadow`.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on unbuilt features](#dependencies-on-unbuilt-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

`calibrate` and `calibrate-diff` measure a candidate (a model, thresholds, weights, limits) on labeled pairs: the
cases somebody thought to label. A shadow measures it on the traffic itself. Every `Evaluate` with an id is judged
twice, once by the live configuration, which decides and writes, and once by the candidate, which only reports.
The output is a stream of events saying where the two disagree, how often, for which tenants, and on which
fields, plus a report over that stream.

Two cases, which cost very different amounts and are built in that order:

- **A shadow policy on the live model**: other weights, floors, thresholds, or quantity limits. The live vectors
  are judged a second time. Nothing is embedded and nothing is stored twice. Microseconds per request.
- **A shadow model**: another model directory, its own collection, its own policy. Every request is embedded a
  second time on a second worker, and the shadow collection is kept in step with the live one by mirroring the
  live writes. This is the expensive case, and the one the README's "the choice of model is a prior, not a
  measured win" is about.

A by-product of the model case that the suggestion did not mention: once the shadow collection has been imported
and mirrored for a while, it *is* a re-indexed collection with drift history. Cutting over is a config change
(the shadow's `[model]` and `qdrant.collection` become the live ones), with no re-index on the day.

## What already exists

Verified against the tree at `545ef20`:

- `crates/similarity-server/src/engine.rs`: `Engine<S: VectorStore, E: TextEmbedder>` with `evaluate`, the private
  `revise` (judge, then write by `WriteMode`), `import` (D20, only creates), `delete`, per-id striped locks
  (`lock_for`), and `store_layout`. `revise` holds the live `baseline: StoredRecord`, the `candidate:
  RecordVectors`, and the anchor rules (`score::extend_anchor`, `quantity::first_values`) in one place. The engine
  does no logging today; `grpc.rs` writes the `evaluate` log line.
- `crates/similarity-core/src/score.rs`: `score::evaluate(schema, policy, dimension, baseline, anchor, candidate)`
  returns an `Evaluation`; `Evaluation::judge_quantities`; `Evaluation::rejudge(schema, policy)`;
  `Comparison::weigh`; `weighted_similarity`; `Reason::code`, `subject`, `value`, `limit`.
- `crates/similarity-core/src/schema.rs`: `Schema::layout_id(fingerprint)` is the fingerprint plus field names;
  weights and quantities are deliberately absent.
- `crates/similarity-core/src/record.rs`: `RecordVectors::hashes` holds a SHA-256 of each field's normalized
  text. It does not depend on the model or on `max_tokens` (D21 hashes the whole text), so a live record and a
  shadow record of the same text carry identical hashes. That is what makes staleness checkable (below).
- `crates/similarity-server/src/worker.rs`: `EmbedWorker::spawn(embedder, queue_depth, max_batch_texts)`, one
  OS thread, bounded `sync_channel`, `try_send` so a full queue is `EmbedError::Overloaded` at once. `TextEmbedder`
  is the seam; `worker::testing::WordBagEmbedder` (with `fail_with` and a counter) is the stand-in.
- `crates/similarity-server/src/main.rs`: `ServeArgs::run` loads the embedder, spawns the worker, opens
  `MemoryStore` or `QdrantStore::connect(settings, layout, dimension)`, and `serve_with` builds the `Engine`.
  `lib.rs` only holds `serve`; the wiring is in `main.rs`, not `lib.rs` as the suggestion says.
- `crates/similarity-server/src/config.rs`: `Config` with `deny_unknown_fields`, `ModelConfig`, `QdrantConfig`,
  `PolicyConfig`, `FieldConfig`, `QuantityConfig`, and `schema()` / `policy()` that validate by building.
- `crates/similarity-store/src/qdrant.rs`: every point carries its layout; `get` returns `LayoutMismatch` for
  another layout; `connect` refuses a collection of the wrong width. A point id is the UUIDv5 of the record id,
  so two layouts in one collection overwrite each other's points on `put`.
- `crates/similarity-server/src/import.rs` and `Engine::import`: loads an export, with optional `anchor` (the
  original version) per record.
- `crates/similarity-server/src/offline/diff.rs`: `calibrate-diff`, `diff::rejudge`, `Flip`, `FailOn`, and
  `offline/events.rs` (the flat `calibration`, `pair`, `pair_field`, `pair_quantity`, `pair_reason` events, the
  `Run` reader, `six()`), and `offline/stats.rs` (`Range`, the owner-clustered `bootstrap`).
- `similarity-log`: `RESULT` events go to stdout and are never filtered.
- Burn `flex` is built with `rayon` (root `Cargo.toml`), and its ops and its `gemm` matrix products run on the
  **global** rayon pool (`burn-flex-0.22.1/src/ops/matmul.rs`: `gemm::Parallelism::Rayon(0)`, all threads).

## What the suggestion got wrong or left open

1. **"A second low-priority embedding worker" cannot be made low priority by the thread.** The worker thread only
   dispatches; the forward pass runs on rayon's global pool, which a second worker would share with the first.
   Lowering the worker thread's OS priority changes nothing. The design below gives the shadow its own small
   rayon pool and makes it wait while the live worker has work.
2. **"Feature 8 reports the disagreements" does not hold as written.** `calibrate-diff` reads two
   `calibrate --ndjson` runs: it needs a `calibration` header, a `pair` per labeled pair, and labels, and it
   reports error counts, slices, and ranges by label. Production traffic is unlabeled, so every flip would be
   `unlabeled` and every other section empty. And the server never holds the baseline's text, so a disagreement
   cannot become a pair here. What closes the loop is a `shadow-report` over shadow events, whose list of
   disagreements (id, tenant, a key) is taken to the work database, turned into labeled pairs there, and then run
   through `calibrate` and `calibrate-diff` as usual.
3. **Mirroring is more than "store when the primary stored".** Left open by the suggestion, decided below:
   - a dropped job (the point of a lossy queue) leaves the shadow's baseline behind the live one for that id, and
     every later shadow verdict on it would be against the wrong baseline unless detected;
   - the shadow may not hold a record the live store does (not imported yet);
   - `Delete` must be mirrored, or a deleted and re-created id is judged in the shadow against its old self;
   - jobs run asynchronously, so two quick revisions of one id must still reach the shadow in live order.
4. **`rejudge` is not exact for a shadow policy with other weights.** It keeps `anchor_similarity` as measured,
   which is a weighted mean under the live weights. Online, the anchor composite is in hand, so the shadow policy
   calls `score::evaluate` again under the shadow schema and is exact. (The offline `--config-b` path has the
   same limitation; it does not matter there because calibration pairs have no anchor.)
5. **A shadow policy overlaps `calibrate-diff --config-b`.** On labeled pairs that already answers "would these
   thresholds be better". What the online policy shadow adds is the rate and location of disagreements on real
   traffic, with history and anchors, which no labeled file has. Worth building because it is small; say so in
   the README so that nobody uses it in place of calibration.
6. **Feature 12 does not exist yet**, so "measure CPU contention with feature 12" needs a measurement here (step 9).

## Design

### Configuration

One optional table. With no `[shadow]` the server behaves exactly as today, and `config.example.toml` ships
without one (a commented example only, with no numbers: thresholds are measurements).

```toml
[shadow]
# A name for the candidate, written on every shadow event.
name = "leaf-ft-2026-11"
# How many shadow jobs may wait, over all shards, before new ones are dropped and counted. Model shadow only.
queue = 1024
# How often a `shadow_summary` line is written, in seconds.
summary_every_s = 60

[shadow.policy]
threshold = 0.0               # required, from `just calibrate` on the candidate
# anchor_threshold, reject_on_measure_change: as in [policy]

# Optional per-field and per-quantity overrides. Only `weight` and `reject_below` for fields, and the two limit
# keys for quantities. A name that is not in [fields] or [quantities] is an error.
[shadow.fields.name]
reject_below = 0.0

# Present: a second model, with its own collection. Absent: the shadow judges the live vectors.
[shadow.model]
dir = "models/leaf-ft-2026-11"
# device, max_tokens, batch_size, output_dim, queue_depth: as in [model], with no inheritance (D4: output_dim has
# no default on purpose, and a shadow that inherited it from the live model would hide the choice).
threads = 1                   # size of the shadow's own rayon pool

[shadow.qdrant]
collection = "records-leaf-ft-2026-11"   # url and timeout_ms default to [qdrant]'s
```

Types in `config.rs`: `ShadowConfig { name, queue, summary_every_s, policy: PolicyConfig, fields:
BTreeMap<String, ShadowFieldConfig>, quantities: BTreeMap<String, QuantityConfig>, model: Option<ShadowModelConfig>,
qdrant: Option<ShadowQdrantConfig> }`, all `deny_unknown_fields`. `ShadowModelConfig` flattens a `ModelConfig`
plus `threads`. Methods `Config::shadow_schema()` (the live schema with overrides applied: same field and quantity
names, same `required`, by construction) and `Config::shadow_policy()`.

`Config::parse` refuses, before anything starts:

- `[shadow.model]` without `[shadow.qdrant]`, or `[shadow.qdrant]` without `[shadow.model]`;
- a shadow collection equal to the live one on the same URL (the shadow's `put` would overwrite live points, since
  point ids depend only on the record id);
- `queue == 0`, `threads == 0`, an override naming an unknown field or quantity.

### The seam in the engine

`Engine` gains one optional field, `observer: Option<Arc<dyn ShadowObserver>>`, set by `Engine::with_observer`.
`dyn` because the model shadow's own types (`Engine<QdrantStore, EmbedWorker>` in production,
`Engine<MemoryStore, WordBagEmbedder>` in tests) should not become type parameters of the live engine.

```rust
/// Something told about every judged revision, after the live outcome is final. Must not block.
pub trait ShadowObserver: Send + Sync + 'static {
  /// Called with the per-id lock still held, so that observations of one id arrive in live order.
  fn observe(&self, observation: Observation);
  /// Called after a live `delete` that removed a record.
  fn deleted(&self, id: &str, tenant: Option<&str>);
}

pub struct Observation {
  pub key: String,                         // `<server start, unix seconds>-<sequence>`, joins the events
  pub id: String,
  pub tenant: Option<String>,
  pub mode: WriteMode,
  pub record: Arc<PreparedRecord>,         // validated texts, hashes, quantities: model-independent
  pub live: LiveOutcome,
  pub baseline: Option<Arc<StoredRecord>>, // the live baseline it was judged against; None for `Created`
  pub candidate: Arc<RecordVectors>,       // the live vectors (used by the policy shadow only)
}

pub struct LiveOutcome {
  pub verdict: Verdict,
  pub evaluation: Option<Evaluation>,      // None for `Created`
  pub stored: bool,
}
```

`evaluate` calls `observe` at its two exits that have an id (the `Created` branch and the end of `revise`), only on
success, still under `lock_for(id)`. Requests without an id (`Embedded`) and failed requests are not observed:
there is no live verdict to compare. The live path's added work is building `Observation` (moves and `Arc`s, no
copy of a vector) and whatever `observe` does, which for both observers below is bounded and never awaits.

The anchor-on-write rule in `revise` (force resets; otherwise `extend_anchor` and `first_values`) moves into a
private `fn next_anchor(...)` so that the mirror write uses the same code.

### A shadow policy (same model)

`shadow::PolicyShadow { name, schema: Schema, policy: Policy, emit: Emitter }`. `observe` runs, synchronously:

```rust
let mut shadow = score::evaluate(&self.schema, &self.policy, dimension, &baseline.vectors,
  Some(&baseline.anchor), &observation.candidate)?;
shadow.judge_quantities(&self.schema, &self.policy, &baseline.quantities, &baseline.anchor_quantities,
  observation.record.quantities());
```

then emits (below). `Created` observations are counted, not emitted per event. An error (only possible from a
layout fault, which the live call would already have hit) is counted and logged at `warn`, never returned. Cost:
one more weighted comparison of the same vectors, a few microseconds; no I/O.

### A shadow model

`shadow::ModelShadow` owns:

- a second `Engine<S2, E2>` (the shadow engine) over the shadow store and a shadow `EmbedWorker`, with the shadow
  schema and policy;
- `shards` (default 4) tokio `mpsc` channels of capacity `queue / shards`, one consumer task each. A job goes to
  shard `hash(id) % shards`, so one id is always handled by one task in order, and different ids in parallel;
- counters (`AtomicU64`): offered, dropped, judged, failed, disagreed, stale, missing, deleted.

`observe` is `try_send` and nothing else; a full shard increments `dropped`. `deleted` enqueues a delete job the
same way. The consumer calls `Engine::mirror(job)` on the shadow engine and emits the result.

`Engine::mirror` (crate-private) is the shadow's half of the write rules:

1. Lock the id (the shadow engine's own stripes). Read the shadow baseline. Tenant rule as everywhere (D14); a
   mismatch is a failed job.
2. Embed the record under the shadow model, reusing the shadow baseline's blocks for unchanged fields
   (`PreparedRecord::plan` with the shadow baseline), exactly as `evaluate` does. A typical edit embeds one
   short text (D9).
3. Judge with `score::evaluate` and `judge_quantities` under the shadow schema and policy.
4. `in_sync`: the shadow baseline's `hashes` and `quantities` equal those of the live baseline in the observation.
   A shadow that missed a write (a dropped job, an import from an older export) is told apart this way without any
   bookkeeping, because hashes are model-independent.
5. Write **if and only if the live call wrote**, with the live mode's anchor rule (`next_anchor`), at the shadow's
   own revision plus one. One exception: if the shadow has no baseline and the live call judged against one, the
   shadow does not write and the job is counted as `missing`, so that a later `import` into the shadow can still
   create the record with its original anchor (D20 only creates). When both had no baseline (`Created`), the
   shadow creates too.
6. Never write when the live call did not (`DRY_RUN`, a rejection, an unchanged resubmission).

A shadow verdict is therefore always "what the candidate would have said, given the history the live service
actually accepted", which is the comparison wanted. A shadow that rejected an edit the live service accepted
still stores it, because live did.

### Keeping the shadow off the live path

What is guaranteed, and how each is tested (test names under [Tests](#tests)):

- **Verdicts and writes.** The live outcome is final before `observe` is called, and `observe` returns nothing.
  The shadow engine has its own store handle and collection, and config parsing refuses a shared collection.
  Tested by a lockstep property: any sequence of operations gives identical live outputs and identical live store
  contents with and without a shadow, including a shadow whose embedder fails or never answers.
- **Latency, the live thread.** `observe` never awaits: policy shadow is a bounded computation, model shadow is a
  `try_send`. Tested with a shadow whose embedder never resolves and a full queue: live calls still finish, and
  the drops are counted.
- **Latency, the CPU.** This is the real cost and can only be bounded, not removed:
  - the shadow worker runs every forward pass inside `rayon::ThreadPool::install` on its own pool of
    `shadow.model.threads` threads (default 1), so it can occupy at most that many cores;
  - before each pass it waits while the live worker has jobs pending (an `AtomicUsize` the live `EmbedWorker`
    increments on `try_send` and decrements when it takes a job). Under sustained live load the shadow therefore
    stalls, its queue fills, and it drops, which is the intended order of sacrifice;
  - a pass already running when a live request arrives is not interrupted; the live request contends with at
    most `threads` cores for at most one shadow pass (a batch of at most `batch_size` texts).

  The bound is stated as a measurement, step 9: live p50 and p99 with the shadow on and off at fixed offered load.
  No number is quoted until it was measured; the owner sets the acceptable difference.
- **Memory.** A second model (about 90 MB for the default size) plus its activations, and the shadow queue
  (`queue` jobs, each holding one prepared record).
- **Start-up and failure.** A config error stops the server, as today. A shadow that cannot start (model directory
  unreadable, shadow collection unreachable or of the wrong width) is an owner decision; recommended default:
  serve without the shadow, log at `error`, and write `shadow_summary` with `enabled: false`.
- **Shutdown.** The shadow does not delay it. Queued jobs are abandoned and counted in a last `shadow_summary`.

`worker.rs` changes: `EmbedWorker::spawn` gains a `Pacing { pool_threads: Option<usize>, yield_to:
Option<Arc<AtomicUsize>> }` argument (live: no pool, no yield; shadow: both), and the live handle exposes
`pending()`. This adds `rayon` as a **direct** dependency of `similarity-server`. It is already in the lock file
through `burn-flex` (MIT or Apache-2.0), so `cargo deny` sees no new crate or license. Whether `gemm`'s
`Rayon(0)` honors an installed pool must be confirmed in its source and by the measurement in step 9; if it does
not, the pool bound is replaced by the idle gate alone and the README says so.

### Events

Through `similarity-log`, target `RESULT` (stdout, never filtered), flat, numbers as numbers, similarities via
`six()`, absent values left out (D15, D17). No record text is ever written. Names are an interface from the first
commit.

- `shadow`: `key`, `id`, `tenant`, `shadow` (the configured name), `kind` (`policy` or `model`),
  `shadow_fingerprint`, `mode` (`enforce`, `dry_run`, `force`), `live_verdict` and `shadow_verdict` (`created`,
  `accepted`, `rejected`), `agree`, `live_similarity`, `shadow_similarity`, `live_anchor_similarity`,
  `shadow_anchor_similarity`, `live_measure_changed`, `shadow_measure_changed`, `live_stored`, `shadow_stored`,
  `in_sync`, `missing`, `queued_ms`, `embed_ms`.
- `shadow_field`: `key`, `field`, `change`, `live_similarity`, `shadow_similarity`, `live_below_floor`,
  `shadow_below_floor`.
- `shadow_quantity`: `key`, `quantity`, `change`, `factor`, `live_beyond_limit`, `shadow_beyond_limit`.
- `shadow_reason`: `key`, `side` (`live` or `shadow`), `code`, `subject`, `value`, `limit` (as `pair_reason`).
- `shadow_summary`, every `summary_every_s` and at shutdown: `shadow`, `kind`, `enabled`, and the counters.
- `shadow_started`, once: `shadow`, `kind`, `fingerprint`, `dimension`, `max_tokens`, `collection`, the shadow
  policy, and one `shadow_schema_field` / `shadow_schema_quantity` line each, mirroring `schema_field`, so a file
  of shadow events says what produced it.

The emitter is a small `Emitter` value (a function pointer in production, a collecting closure in tests) so tests
can assert on outcomes without capturing stdout.

### Filling the shadow collection

`record-similarity import --shadow export.jsonl config.toml`: the existing command with the embedder, schema,
policy, and store taken from `[shadow]` instead of `[model]` / `[qdrant]`, so that there is no second config file
to drift. Run it with the export's `anchor` column wherever the work database has original versions. It can run
while the server is shadowing: import only creates, and the mirror never creates over a live baseline (rule 5).
Records edited between the export and the import come up `in_sync: false` until their next live write.

### Reporting

`record-similarity shadow-report shadow.ndjson [--ids disagreements.jsonl] [--ndjson]`, in
`offline/shadow.rs`, reading the events above (and skipping everything else, as `events::read` does):

- the live by shadow verdict table, counting only `in_sync` events with a baseline;
- the disagreement rate in each direction, with a 95% interval resampled by tenant (the same `bootstrap` over
  clusters as D17, for the same reason: one brand's edits are not independent);
- the tenants with the most disagreements, the reason codes behind them on each side, and per-field similarity
  ranges on each side among disagreements;
- coverage: offered, dropped, stale, missing, failed, so that a report on 20% of traffic says so;
- `--ids` writes one line per disagreement (`key`, `id`, `tenant`, both verdicts) for the work environment to join
  to its own history and label. Those labeled pairs then go through `calibrate --full` and `calibrate-diff`.

Every number comes from new functions in `offline::stats` (D17: one source for the text and the events).
`--fail-on shadow-accepts-recycles` is not offered: without labels there is no such thing. A gate belongs on the
labeled follow-up, where `calibrate-diff --fail-on` exists.

### `Describe`

Additive: `optional ShadowDescription shadow = N` on `DescribeResponse` with the name, kind, fingerprint, and the
shadow policy, so an operator can see what is running. Recommended, and cheap; the proto stays the documentation.

## The work, in order

Each step is one commit and leaves `just ci` passing.

1. **Engine seam, no behavior change.** `next_anchor` extracted; `ShadowObserver`, `Observation`, `LiveOutcome`;
   `Engine::with_observer`; `observe` at both exits and `deleted` in `delete`. A recording observer in tests.
2. **Policy shadow.** `config.rs` `[shadow]` with `policy`, `fields`, `quantities` (no `model` yet) and its
   validation; `shadow.rs` with `PolicyShadow`, the `Emitter`, and the events; `shadow_started` and
   `shadow_summary` (a tokio interval task); wiring in `main.rs` `serve_with`.
3. **Mirror write.** `Engine::mirror` and `Engine::mirror_delete`, crate-private, tested in memory directly.
4. **Model shadow runner.** `ModelShadow` (shards, `try_send`, counters, consumers, abandonment on shutdown),
   emitting `kind: model` events.
5. **Worker pacing.** `Pacing`, the live pending counter, the idle gate, the dedicated rayon pool; `rayon` as a
   direct dependency. `just deny`.
6. **Model shadow wiring.** `[shadow.model]`, `[shadow.qdrant]`, the shared-collection refusal, start-up failure
   handling, `serve` and `serve --in-memory` (a `MemoryStore` shadow, empty at start), `Describe`.
7. **`import --shadow`.**
8. **Properties.** The lockstep and mirror properties below in `similarity-server`'s `properties` module, then
   `PROPTEST_CASES=5000 just test` once, as CLAUDE.md requires after touching the write path.
9. **Measurement and Qdrant test.** A `just shadow-latency` recipe (fixed offered load through the gRPC client,
   shadow off then on, same machine; release build), the ignored Qdrant test, and the numbers it produced in the
   README. If feature 12 lands first, use its load test instead.
10. **`shadow-report`** and its stats.
11. **Documentation** (below), and D22.

## Tests

Fast (no model, no network, no Qdrant), using `MemoryStore`, `WordBagEmbedder` under a second fingerprint
(`word-bag-v2`) for the shadow model, and a policy that differs from the live one to force disagreements:

`engine.rs`:

- `an_observer_sees_each_judged_revision_once_after_the_live_outcome_is_final`
- `a_record_without_an_id_and_a_failed_request_are_never_observed`
- `observations_of_one_id_arrive_in_the_order_the_live_engine_judged_them`
- `a_live_delete_that_removed_a_record_is_observed`
- `the_mirror_writes_exactly_when_the_live_engine_wrote`
- `the_mirror_resets_its_anchor_only_when_the_live_write_was_forced`
- `the_mirror_lets_its_anchor_take_only_the_first_value_of_a_new_field`
- `a_shadow_without_the_record_does_not_create_it_over_a_live_baseline`
- `a_shadow_that_missed_a_write_reports_itself_out_of_sync_and_catches_up_on_the_next_one`
- `the_mirror_reuses_its_own_blocks_for_unchanged_fields`
- `the_mirror_refuses_a_record_stored_under_another_tenant`

`shadow.rs`:

- `a_shadow_policy_judges_the_live_vectors_without_embedding_anything`
- `a_shadow_policy_with_other_weights_measures_drift_under_its_own_weights`
- `a_shadow_policy_that_fails_is_counted_and_the_live_response_is_unchanged`
- `a_full_shadow_queue_drops_and_counts_without_slowing_the_live_call` (paused tokio time, a never-resolving
  shadow embedder)
- `a_shadow_whose_embedder_fails_leaves_live_verdicts_and_writes_alone`
- `jobs_for_one_id_reach_the_shadow_in_order_while_other_ids_run_in_parallel`
- `queued_shadow_jobs_are_abandoned_and_counted_at_shutdown`
- `shadow_events_are_flat_and_carry_no_record_text`

`worker.rs`:

- `a_shadow_pass_waits_while_the_live_worker_has_jobs_pending`
- `the_live_worker_counts_jobs_pending_and_taken`

`config.rs`:

- `a_config_without_a_shadow_parses_as_before`
- `a_shadow_sharing_the_live_collection_is_refused`
- `a_shadow_override_for_a_field_the_schema_lacks_is_refused`
- `a_shadow_model_without_a_shadow_collection_is_refused`
- `the_shadow_schema_keeps_the_live_names_and_required_fields`

`properties` (D19):

- `a_shadow_never_changes_a_live_output_or_a_live_write` (lockstep: two engines, one with a model shadow whose
  embedder is chosen per case among working, failing, and stalled; same operations; equal outputs and stores)
- `with_no_drops_every_shadow_record_has_the_live_record_s_text_hashes_and_quantities` (the mirror invariant)
- `with_no_drops_the_shadow_writes_the_same_ids_the_same_number_of_times`

`main.rs` and `offline/shadow.rs`:

- `serve_accepts_a_shadow_and_import_accepts_shadow`
- `the_shadow_report_counts_only_in_sync_judgments_with_a_baseline`
- `the_shadow_report_resamples_tenants_for_its_intervals`
- `the_shadow_report_lists_every_disagreement_for_labeling`

Slow, `#[ignore]`:

- `a_shadow_model_mirrors_the_live_collection_over_qdrant` (`just test-qdrant`): real Qdrant, the word-bag
  embedders, two collections, a delete, a forced write.
- `the_real_model_as_its_own_shadow_agrees_on_every_revision` (`just test-real`): the shipped model as both live
  and shadow, different collections; every shadow verdict equals the live one. This catches a mirror bug that
  only shows with real vectors.
- `just shadow-latency`: not a test, a measurement; its numbers go to the README.

**Only at work:** whether a candidate is better (it needs real traffic and then labels on the disagreements);
latency on the production hardware and load; the share of jobs dropped at real peak; whether `in_sync` stays high
after an import from a real export.

## Documentation to update

- **README.md**: a section "Trying a candidate in the shadow" under Operations: the two kinds, the config, `import
  --shadow`, `shadow-report`, what a disagreement list is for, and that a shadow policy is not a substitute for
  `calibrate`. A paragraph in "Rolling it out" and in "Re-indexing" (cutover from a shadow collection). The
  configuration reference gains `[shadow]`. Capacity gains the measured cost. Status item 2 points here.
- **CLAUDE.md**, "Things that are easy to get wrong": a shadow never decides; the mirror writes iff the live call
  wrote and uses the live anchor rule; never let the live path await the shadow; never point the shadow at the live
  collection; the shadow event names are an interface like the calibration ones. Workspace layout: `shadow.rs`.
- **docs/decisions.md**: D22, "A candidate runs in the shadow of the live service": the observer seam, the mirror
  rule and the `missing` exception, staleness by text hash, the CPU bound (own pool, idle gate) and why thread
  priority would not work, lossy by design, the events, why `calibrate-diff` is the follow-up and not the reader.
  D9 gains a sentence pointing to D22. D12 gains the cutover note.
- **docs/feature-suggestions.md** and **docs/plan-features.md**: mark 18 built when it is.
- **config.example.toml**: a commented `[shadow]` with placeholders, no numbers.

## Dependencies on unbuilt features

None is required. Assumed interfaces where they would help:

- **12 (latency tracking)**: if built first, step 9 uses its load generator. Otherwise `just shadow-latency` is a
  minimal one and should be folded into 12 later.
- **10 (telemetry)**: independent. If 10 adds a metrics endpoint, the shadow counters belong there too; the
  `shadow_summary` event stays.
- **17 (reviewer feedback)**: a reviewer's label on a disagreement, keyed by `key` or by `(id, revision)`, is the
  ideal way to label the `--ids` output. Assumed: 17 emits a line with `id` and a label that `calibrate --full`
  can read.
- **4 (promotion gate)**: a gate could require a shadow run with a disagreement rate under a set value. Not
  designed here.

## Decisions for the owner

1. **One shadow at a time?** Recommended: yes. Several candidates multiply the CPU cost and the events, and one
   at a time is how a cutover happens anyway.
2. **What the shadow does when it lacks a record the live store has.** Recommended: does not create it, counts
   `missing`, so that `import --shadow` can still create it with its original anchor.
3. **A shadow that cannot start.** Recommended: serve without it, log an error, `shadow_summary enabled: false`.
   The alternative (refuse to start) lets an experiment take the live service down.
4. **Events on stdout as results.** Recommended: yes (`RESULT`), so `RUST_LOG` cannot hide them and they can be
   captured alone. The server writes no results today, so this is a new stream on its stdout.
5. **The CPU bound.** Recommended: an own rayon pool of 1 thread plus waiting while the live worker is busy. This
   adds `rayon` as a direct dependency (already in the tree). Accept the measured live p99 difference before
   enabling in production; the acceptable value is the owner's.
6. **Mirror deletes.** Recommended: yes, through the same queue (and so lossy, caught by `in_sync`).
7. **Same field and quantity names as live, overrides only.** Recommended: yes. A candidate with another field
   set changes the request contract and belongs in a calibration, not a shadow.
8. **`Describe` reports the shadow.** Recommended: yes, additive.

## Size

**Large.** The policy shadow alone (steps 1 and 2) is small, a day or two. The model shadow adds a second worker
with pacing, an asynchronous mirror with its own write rules, a lossy ordered queue, a new command, a report, and
properties over the write path, plus a measurement that only means something on the production hardware; and the
question it exists to answer can only be answered in the work environment.
