# Feature suggestions

Twenty features that could be added to this project, with what each is, why it matters here, how it would fit the
code, and what it would cost. **Features 1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 15, 16, and 19 are built** (D17 to D28, and the README); the rest are not. They were proposed in November 2026 from a read of the
repository as it stood then (`README.md`, `docs/decisions.md`, and the crates); check a suggestion against the
code before starting on it, since the code will have moved. It has moved once already: records now carry
quantities, compared as numbers (D16), and the features that touch a verdict or a report say below what that
means for them. `docs/plan-features.md` plans the five listed under "What to build first" against the code as it
is now.

Effort is **S** (a day or less), **M** (a few days), or **L** (a week or more). "Needs real data" says whether a
feature is only meaningful once real labeled pairs exist. Real data never enters this repository (see
`docs/data-request-prompt.md`), so anything marked that way has to be finished and judged where the data lives;
it can still be written and tested here on synthetic data.

## Contents

- [The gaps these address](#the-gaps-these-address)
- [Summary](#summary)
- [A. Experiment tracking, lineage, and a model registry](#a-experiment-tracking-lineage-and-a-model-registry)
- [B. Regression reporting](#b-regression-reporting)
- [C. Testing depth](#c-testing-depth)
- [D. Serving, operations, and product](#d-serving-operations-and-product)
- [What to build first](#what-to-build-first)
- [Things noticed along the way](#things-noticed-along-the-way)

## The gaps these address

The serving path and the training math are solid. The gaps are in what surrounds them:

- **No machine-readable record of an evaluation.** `calibrate` prints a text report, and its strict four-key input
  format drops the `pair_id`, `owner`, `variant_change`, and `split` that the data request collects.
- **No lineage.** A fine-tuned model directory does not say which base model, data, or options produced it. Only
  checkpoints carry a state file, and that holds the options and a data fingerprint but not the base model.
- **Nothing compares two runs,** whether two trainings or two calibrations.
- **Shipping a better model is a hand-driven re-index** that also resets every record's drift history, because
  the server stores vectors and text hashes, never the text.
- **Testing is example-based only.** There are no property tests, no fuzzing, and no mutation testing.

**On Weights & Biases.** W&B is used for other projects in the organization, and this one should not depend on
it: a training run must work, and be fully recorded, with no tracker reachable. Section A is therefore built on
local files and the NDJSON events the trainer already emits. That choice keeps W&B available as an option
instead of ruling it out: once every run is a directory of events (feature 2), a small exporter can replay a run
into W&B, or into anything else with an HTTP API, after the fact. Build that exporter only if someone wants these
runs to appear next to other projects' runs. MLflow and Aim were considered and set aside for the same reason,
and because each would put a Python service in the loop.

## Summary

| # | Feature | Area | Effort | Needs real data |
|---|---|---|---|---|
| 1 | Lineage file in every model directory (built) | Tracking | S | No |
| 2 | Local run registry | Tracking | M | No |
| 3 | Evaluation at every checkpoint | Tracking | M | For meaningful results |
| 4 | Model registry and promotion gate | Tracking | M | No |
| 5 | Run report and compare | Tracking | M | No |
| 6 | Hyperparameter sweeps | Tracking | M to L | Yes |
| 7 | Structured calibration output with slices (built) | Regression | M | For the slices |
| 8 | Calibration diff (built) | Regression | M | No |
| 9 | Pinned verdict snapshot gated in CI | Regression | M | No |
| 10 | Production score telemetry and drift report | Regression | M | Real traffic |
| 11 | Drift-threshold calibration from histories | Regression | M | Yes |
| 12 | Latency regression tracking and a load test | Regression | S to M | No |
| 13 | Property-based and state-machine tests (built) | Testing | M | No |
| 14 | Fuzzing | Testing | M | No |
| 15 | Mutation testing | Testing | S to start | No |
| 16 | Bulk import and re-index that keeps drift history (built) | Serving | M to L | No |
| 17 | Reviewer feedback capture | Serving | M | No (it produces it) |
| 18 | Shadow evaluation of a candidate model | Serving | L | Real traffic |
| 19 | Structured reason codes and explanations | Serving | M | No |
| 20 | Menu-item to brand-product matching | Serving | L | Yes |

## A. Experiment tracking, lineage, and a model registry

### 1. Lineage file in every model directory

**Built** (D18). One difference from the description below: the file records what produced a model and not what
the model scored, because a model directory is saved before anything is measured and is never modified in place.

**What.** The trainer writes `provenance.json` next to the weights in `--out` and in every checkpoint:

- the base model's directory and fingerprint digest, with the base's own `provenance.json` nested inside, so the
  chain back to the published model is recorded;
- each data file's path and SHA-256, the existing data fingerprint, and the example counts;
- the full training options, `--max-tokens`, device, and seed;
- the git commit (with a dirty flag) and crate version;
- the chain of `--resume` points;
- the final `separation` results.

`model fetch` writes the same file with the repository, revision, and checksums it already verifies.

**Why here.** The README tells the owner to pick the best pass by pointing configs at checkpoints. After a few
rounds nobody can say which data and options made `models/mdbr-leaf-mt-ft`, and D3 says thresholds belong to one
specific model. This information cannot be recovered later if it is not written when the model is.

**Fit.** `similarity-train/src/checkpoint.rs` (`save_model`, `write`) and `similarity-server/src/fetch.rs`. The
digest loop in `Embedder::load` would move into a shared function in `similarity-core`. The fingerprint hashes
only the files the encoder reads plus `tokenizer.json`, so an extra file does not change it (D12 holds), and
sentence-transformers ignores files it does not know. A small `build.rs` captures the commit. No new
dependencies.

**Effort.** S: everything is already in memory when the model is saved. **Needs real data:** no.

### 2. Local run registry

**Built** (D23). What came out differently: the record is a third logging layer with a level of its own, not a
copy of the two streams (the loss curve is made of logs); `killed` is worked out when runs are read, not written;
and a resumed run is a run of its own that names the one it continues.

**What.** Every `record-similarity-train`, `calibrate`, and `throughput` invocation creates a run directory,
`runs/<utc>-<slug>/`, holding:

- `run.json`: the kind of run, arguments, options, git commit, host and CPU, device, model fingerprint, input file
  hashes, start and end times, and a status (`running`, `finished`, `failed`, `killed`);
- `events.ndjson`: both output streams, logs and results together.

A resumed run appends to its original directory. `--run-dir` moves it, `--no-run` disables it, and `runs list`
prints one line per run from the manifests.

**Why here.** This is what W&B calls a run. The events the trainer already emits (`training`, `checkpoint`,
`separation`, `finished`) are the metric history; today they survive only if someone remembered to redirect them
to a file.

**Fit.** A file sink in `similarity-log` (a second writer on the same formatter, so D15's "one format, one
crate" holds). The manifest types and reader go in a new small crate, say `similarity-runs`, with no Burn
dependency, used by both binaries. `runs/` joins `.gitignore`.

**Effort.** M: logging plumbing plus manifests for three commands. **Needs real data:** no.

### 3. Evaluation at every checkpoint

**Built** (D24). What came out differently: results are not written into a checkpoint (D18) but emitted as
events; checkpoints are compared by AUC and not at the configured thresholds, which belong to the base model;
the validation split is carved by whole owners with `pairs carve`; and catalog groups of measured owners are
left out of training. A resumed run does not re-score its predecessor's checkpoints.

**What.** The trainer measures separation on an evaluation split at each checkpoint, not only at the start and
end. It emits `separation` events with `stage: "checkpoint"` and the step, writes `separation.json` into the
checkpoint, and names a `best_checkpoint` in `finished`. Optionally, `--eval-config config.toml` also runs the
full-policy calibration (feature 7) at each checkpoint.

**Why here.** Choosing a pass is manual today: calibrate each checkpoint by hand. This gives loss and separation
curves over steps, which is the chart people open W&B for.

**Caveat.** Selecting a checkpoint on the brand-disjoint `holdout` split turns the holdout into a validation
set, and it stops being a clean test. Select on a separate `validation` fold carved from `calibrate` by brand
hash, and report `holdout` only for the checkpoint that was chosen.

**Fit.** `trainer.rs` (the checkpoint hook), `main.rs` (events), `data.rs` (a third split). The cost is one
forward pass over the split per checkpoint.

**Effort.** M. **Needs real data:** the code no, meaningful results yes.

### 4. Model registry and promotion gate

**What.** An append-only `models/registry.ndjson` of `registered`, `calibrated`, `promoted`, and `retired` events,
keyed by model fingerprint. `record-similarity model promote models/x --calibration runs/<id>` refuses when:

- the calibration's fingerprint is not the model's;
- the calibration ran on something other than the holdout split;
- the model is worse than the currently promoted one on the same holdout file (using feature 8).

On success it prints a config fragment: `model.dir`, a new `qdrant.collection` derived from the layout id, and
the thresholds from that calibration. At start-up the server warns if its model is unregistered or its
thresholds differ from the promoted calibration.

**Why here.** The two mistakes `CLAUDE.md` calls easiest to make become things a tool checks: thresholds that do
not transfer between models (D3), and a model change without a new collection (D12). Today both depend on
someone having read the README.

**Fit.** `similarity-runs`, plus a `model promote` subcommand in `similarity-server/src/main.rs` (the `model`
group exists). No conflicts.

**Effort.** M. **Needs real data:** no, though it is most useful once real holdouts exist.

### 5. Run report and compare

**Built** (D28). What came out differently: the commands are `record-similarity runs compare` and `runs report`,
not a binary of their own, since comparing calibrations needs code the trainer must not link; two calibrations
are compared by `calibrate-diff`'s own code; a resumed run is stitched into one curve; and the page has no
script. Grouping calibrations under their training run and per-field range bars were left out.

**What.** `record-similarity-runs report runs/A runs/B --out report.html` writes one self-contained HTML file
with inline SVG drawn in Rust (no JavaScript, no CDN, opens offline):

- loss, gradient-norm, and learning-rate curves;
- separation over steps;
- overlaid edit and recycle score histograms per calibration;
- per-field range bars;
- a side-by-side diff of options and lineage.

`runs compare A B` prints the same as a text table.

**Why here.** This is the dashboard, without a service. There is deliberately no terminal UI (D15), and a static
file can sit next to the data it describes.

**Fit.** A binary in `similarity-runs`. The SVG is string formatting, so no dependency is needed (or `plotters`
with only its SVG backend). The optional W&B exporter described above would live here too.

**Effort.** M: charts are tedious but straightforward. **Needs real data:** no.

### 6. Hyperparameter sweeps

**Half built** (D29). The policy sweep exists as `calibrate-sweep`: a grid of weights, floors, and limits judged
on one recorded calibration with no model, each point at its own best threshold, with an optional second run
that the sweep was not fitted to. The training sweep does not: every training option is now a flag and every
run is recorded and comparable, so a sweep is a shell loop, but there is no command that runs it.

**What.** `record-similarity-train sweep sweep.toml`. The file gives fixed options and axes (learning rate,
epochs, `--dims`, batch size, dropout, and three or more seeds per point). Each point runs as a child run under a
parent sweep run (feature 2), keeps one checkpoint by default, and is scored by full-policy calibration on the
validation fold. The sweep ends with a leaderboard event.

**Why here.** The README's only tuning advice is to lower the learning rate if the holdout gets worse. At a few
thousand examples the spread between seeds can exceed the effect being measured, so seeds have to be part of
every sweep. Runs are sequential because there is one GPU, and checkpoints are 265 MB each, hence the pruning.

**Fit.** The trainer would need to call the calibration code, which lives in the server crate, and the trainer
must not link the server (D13). The scoring half of `offline.rs` uses only core types and could move into
`similarity-core` or a shared crate.

**Effort.** M to L: mostly orchestration, plus that refactor. **Needs real data:** yes. Sweeping on 46 synthetic
pairs would mostly tune to noise.

## B. Regression reporting

### 7. Structured calibration output with slices

**Built.** `calibrate --full` and `--ndjson`, as described in the README and D17. Two things came out differently
from the description below: the per-pair events are flat (`pair`, `pair_field`, `pair_quantity`, `pair_reason`
lines joined on `pair_id`) because the logging layer cannot write a nested object, and a quantity limit is
suggested from the largest factor among edits, not from a gap between the two labels.

**What.**

- `calibrate --ndjson` emits `pair` events (pair id, label, overall score, per-field similarity and change,
  per-quantity change, factor, and flags, `measure_changed`, verdict, reason codes), then `field_range`,
  `quantity_range`, `suggestion`, and `summary`.
- `--full` reads `pairs.full.jsonl` directly, with `--split` and `--min-confidence` filters, in place of the `jq`
  projection the README describes.
- The report gains slices: by `variant_change`, by top-level category, by `label_source`, and by owner.
- It adds brand-clustered bootstrap intervals on the false-reject and false-accept rates, and an operating-point
  table (false-reject rate at 1%, 2%, and 5% false-accept rate).
- It shows the score distribution of `unsure` pairs, scored but not counted as errors.
- For each quantity it suggests a `reject_above_factor` the way it suggests a threshold, when the factors seen in
  edits and in recycles leave a gap, and says so when they do not. Today the report only prints the two ranges.

**Why here.** The data request calls same-brand variant changes the most valuable pairs and expects many of them
to be `unsure`. Today `calibrate` cannot read those fields at all. The data request also asks for a measured
base rate, which this could turn into an expected review load per thousand edits.

**Fit.** `similarity-server/src/offline.rs` and `main.rs`. The text report stays the default (D11, D15); the
NDJSON form follows D15's "results are output". The bootstrap needs only a small seeded generator like the
trainer's.

**Effort.** M. **Needs real data:** the format no, the slices and intervals yes.

### 8. Calibration diff

**Built**, as `record-similarity calibrate-diff`. The `--config-b` mode judges the first run's cached measurements
again and needs no embedding cache, which was dropped.

**What.** `record-similarity calibrate-diff old.ndjson new.ndjson` joins pairs by `pair_id` and reports:

- every verdict flip (an edit that went from accepted to rejected, a recycle that went from rejected to
  accepted), sorted by score change;
- per-slice changes in error rates and in margin;
- per-field distribution shifts.

`--fail-on new-false-accepts|any-flip|errors-up` sets the exit code. A `--config-b` mode re-scores under a
second config (weights, floors, thresholds) from cached per-field similarities, with no re-embedding.

**Why here.** It answers "did the new model, thresholds, weights, or code make verdicts worse?" directly.
Weights and thresholds need no re-index (D12), so most experiments are config changes, and cached re-scoring
makes those instant.

**Fit.** `offline.rs`, plus a small embedding cache under `target/` keyed by fingerprint and text hash (the text
hash exists already). It should refuse to diff runs over different pair sets unless told to intersect them.

**Effort.** M. **Needs real data:** no to build; the value arrives with the real holdout.

### 9. Pinned verdict snapshot gated in CI

**Built** (D22). What came out differently: the snapshot is a `calibrate --ndjson` run and not a format of its
own; it is checked by tolerance (0.001), not at three decimals; most of the check needs no model, because the
snapshot is judged again under the current config in `cargo test`; and pairs may carry an `anchor`, without which
the blank-and-refill swap could not be a case.

**What.** A committed `data/regression.jsonl` of curated synthetic hard cases:

- same-brand flavor, strain, size, and potency changes;
- the two-step blank-and-refill swap;
- whitespace-only edits;
- Unicode look-alikes;
- all-caps rewrites.

With it, a committed snapshot of verdicts and scores (three decimals) under the pinned model. A CI job caches
`models/` keyed on the hash of `fetch.rs`, runs the parity tests, and fails on any verdict flip or a score change
above 0.005. `just snapshot` regenerates it, as `just golden` does for the fixtures. The same job checks that
the thresholds in `config.example.toml` and the calibration block quoted in the README match a fresh run.

**Why here.** The real-model tests run only locally today, so an encoder, tokenizer, or scoring change can merge
without anyone having run `just test-real`. Two rules in `CLAUDE.md`, "every README number was measured" and
"never change thresholds without recalibrating", would become checks. Real pairs cannot be committed, so a gate
on the real holdout stays a local recipe built on feature 8.

**Fit.** `.github/workflows/ci.yml` (a new job), `similarity-server/tests/`, the `justfile`. It adds one 90 MB
download per pinned revision to CI, which the current workflow avoids on purpose, so the cache is essential.

**Effort.** M. **Needs real data:** no.

### 10. Production score telemetry and drift report

**What.**

1. A richer `evaluate` log event. It carries `id`, `verdict`, `similarity`, `anchor_similarity`, `stored`, and
   `revision` today. Add the tenant, mode, model fingerprint, per-field similarity and change, reason codes, and
   the time spent embedding and in Qdrant.
2. An optional Prometheus endpoint (`server.metrics_listen`): verdict counters by mode, per-field similarity
   histograms, queue depth, batch size, `RESOURCE_EXHAUSTED` count, latency.
3. `record-similarity drift --logs prod.ndjson --reference calibration.ndjson`: compares weekly per-field
   distributions with the previous window and with calibration (population stability index and a KS statistic),
   and reports rejection-rate trends by tenant.

**Why here.** The rollout plan in the README is to run in `DRY_RUN` and log the responses. The current log line
cannot show which field caused a rejection or which brand is behind most of them, so the logs are not yet the
dataset that plan assumes.

**Fit.** `grpc.rs` for the logging (decisions stay in the engine), `config.rs` for the endpoint. The endpoint
would use `metrics` and `metrics-exporter-prometheus`, which need a `cargo deny` check. The drift command goes in
`offline.rs` or `similarity-runs`.

**Effort.** M; the log enrichment alone is S and should go first. **Needs real data:** real traffic, not labels.

### 11. Drift-threshold calibration from histories

**Built** (D27). What came out differently: every version is replayed with nothing able to reject and judged
afterwards, since a replay under the configured policy invents everything after the first rejection; the
threshold is placed below legitimate drift and above the recycles the per-step rules miss, not in a gap between
labels; and the history format gained `recycled_at`, `original`, and `truncated`.

**What.** `record-similarity calibrate-drift sequences.jsonl` replays each edit history, version by version,
through the real engine on an in-memory store in `ENFORCE` mode. For each legitimate history it reports the
lowest anchor similarity reached, the step and reason of the first rejection if there is one, and whether the
pair thresholds alone would block a legitimate long-lived product. It suggests an `anchor_threshold` with a
margin, and says whether recycled histories are caught and at which step.

**Why here.** The README says `anchor_threshold` "is chosen from whole edit histories rather than pairs", and
the data request collects about 200 histories for exactly that. Nothing in the repository reads
`sequences.jsonl`.

**Fit.** The engine is already driven against `MemoryStore` in tests, so this is a new subcommand and a report.
It writes only to a throwaway store.

**Effort.** M. **Needs real data:** yes.

### 12. Latency regression tracking and a load test

**Built** (D26). What came out differently: the worker keeps counters, because a client cannot see batch sizes;
the loop is closed and says so; the store is in memory unless a running server is named; and there is no gate
on the figures. An open-loop mode, Qdrant in process, and records from a real export were left out.

**What.**

- `throughput` emits NDJSON results (fingerprint, CPU, threads, git revision) into the run registry, and
  `runs compare` flags a slowdown beyond a set percentage.
- A new `just load` drives N concurrent `Evaluate` calls at a running server and reports p50 and p99 latency,
  texts per second achieved, batch sizes, and the `RESOURCE_EXHAUSTED` rate.
- Optionally, a CI gate on instruction counts (`iai-callgrind`, Linux only) for the scoring path and the tiny
  test encoder. Wall-clock time on shared runners is too noisy to gate on.

**Why here.** See "Things noticed along the way": the README's concurrent throughput figure is derived from a
batch measurement, and nothing measures it through the gRPC queue. Bootstrap and re-index time estimates rest on
that figure.

**Fit.** `similarity-core/examples/throughput.rs`, `similarity-server/examples/client.rs`, the `justfile`.

**Effort.** S to M. **Needs real data:** no.

## C. Testing depth

### 13. Property-based and state-machine tests

**Built** (D19), with the quantity properties added. It found that numbers read from JSON depended on which crates
were in the build, which is fixed.

**What.** Properties checked on generated inputs:

- **Composite identity (D5).** The cosine of two composites equals the weighted mean of the field cosines when
  both records have the same fields filled.
- **Engine state machine.** Random sequences of `Evaluate` (every mode), `Delete`, and `FindSimilar` run against
  a simple reference model, on the in-memory store and the bag-of-words embedder. Invariants: a rejected or
  `DRY_RUN` revision never changes stored state (D7); an ordinary accepted write never replaces an anchor value
  (D8); `FORCE` resets the anchor; the tenant rules hold (D14).
- **Threshold suggestion.** `suggest_threshold` equals a brute-force optimum.
- **Normalization.** Whitespace normalization is idempotent and the text hash is stable under it.
- **Batching (D9).** Embedding a text alone and inside a merged, length-bucketed batch agree within tolerance.
- **Scores.** Always finite and within `[-1, 1]`.

**Why here.** D7 and D8 are the product's safety properties, and they are covered by hand-written scenarios
today. The coming churn in models, thresholds, and re-indexing is when they could quietly break.

**Fit.** `proptest` as a dev-dependency in `similarity-core` and `similarity-server`. It fits the rule that
`cargo test` needs no model, network, or Qdrant.

**Effort.** M. **Needs real data:** no.

### 14. Fuzzing

**What.** A `fuzz/` crate outside the workspace, using `cargo-fuzz`. It needs nightly Rust while the toolchain
is pinned to 1.96, so it runs as a scheduled CI job and not on every pull request. Targets, by how exposed each
boundary is:

1. **Model-directory parsing** (`modules.json`, the pooling config, `config.json`). These files arrive from the
   network through `model fetch`. Invariants: no stage path leaves the model directory, and `Normalize` is only
   accepted last (D2).
2. **A structure-aware request stream into the engine** (`arbitrary`), on the in-memory store and the
   bag-of-words embedder: no panics, and the invariants of feature 13. The API is unauthenticated, so this is
   the surface any caller can reach.
3. **Record normalization and the tokenizer** on pathological Unicode and very long fields.
4. **TOML config into the schema and policy**: NaN or infinite thresholds, zero or negative weights.
5. **Calibration and training JSONL, and `training_state.json`** on resume.

Safetensors header parsing is better left to `burn-store` upstream.

**Why here.** `unsafe_code = "forbid"` means fuzzing here finds panics in dependencies (the tokenizer, Burn's
shape checks), logic holes, and cost blow-ups, not memory bugs. It should be framed and budgeted that way.
Target 2 doubles as a deep invariant test.

**Fit.** A new `fuzz/` directory; `cargo-fuzz` and `arbitrary` in that crate only; a CI cron job. Some
internals may need a public entry point behind a `fuzzing` feature.

**Effort.** M. **Needs real data:** no.

### 15. Mutation testing

**Built** (D25). What came out differently: an allowlist of eight files, not a list of exclusions; a fixed
proptest seed, without which the survivors change from run to run; no CI job, since there is no remote; and a
full run takes a quarter of an hour, not hours. It found 18 unguarded boundaries, among them a real bug in the
threshold suggestion.

**What.** `cargo mutants` over `similarity-core` (`score.rs`, `record.rs`, `schema.rs`, `vector.rs`),
`engine.rs`, `offline.rs`, and `checkpoint.rs`:

- weekly in CI;
- `--in-diff` on pull requests that touch those files;
- a committed `mutants.toml` excluding `model/bert.rs` and the loader, whose real coverage is the ignored parity
  tests, so mutants there would always survive the fast suite.

Surviving mutants are triaged into new tests.

**Why here.** The policy code is short and full of comparisons (`<` against `<=` at a floor, saturating
arithmetic). A surviving mutant there is an untested boundary, and a boundary is a verdict.

**Fit.** A cargo tool with no workspace dependency, running entirely within the fast suite.

**Effort.** S to start, M to work through the survivors. **Needs real data:** no.

## D. Serving, operations, and product

### 16. Bulk import and re-index that keeps drift history

**Built** (D20), as `record-similarity import`. It only creates, which is what makes it resumable, so the store
trait did not have to grow. The streaming `Import` RPC was not built.

**What.** `record-similarity reindex --config new.toml export.jsonl` writes straight to a new collection
through the store, bypassing the gRPC queue. It embeds in full batches, resumes by skipping ids already stored
under the target layout, and reports progress as NDJSON. Each input line is
`{id, tenant, fields, anchor_fields?}`. When the caller still has a record's original version, `anchor_fields`
is embedded as the anchor, so a re-index no longer resets drift detection. A client-streaming `Import` RPC could
offer the same thing remotely.

**Why here.** "No bulk API" is a listed limitation, and the README says re-indexing resets drift history. Every
model change forces a re-index (D12). The server never stores field text, so the input has to come from the
caller's database, and this gives it a defined format.

**Fit.** `similarity-server` (the engine and a new subcommand). The store trait needs a way to ask what is
already stored, so `MemoryStore`, `QdrantStore`, and the contract tests all grow. Importing an anchor needs an
engine entry point that respects D8: it sets an anchor only when creating a record.

**Effort.** M to L. **Needs real data:** no.

### 17. Reviewer feedback capture

**What.** A `RecordFeedback` RPC: the caller sends the id, tenant, the before and after fields, and the
reviewer's decision with its source. The server checks that `before` matches the stored baseline's field hashes
and emits a `feedback` result event in the `pairs.full.jsonl` shape, with `label_source: "reviewer"`.
`record-similarity feedback export logs.ndjson` turns those events into a calibration and training file.

To counter selection bias (reviewers otherwise only ever see rejections), the server can flag a configurable
random fraction of accepted edits as `review_sample` in the response, for the caller to route to review as well.

**Why here.** Labeled data is the bottleneck the README's status section names. Moderator actions are the
strongest evidence class in the data request, and this would produce them continuously, already in the right
format.

**Fit.** The proto, `grpc.rs`, `engine.rs` (the check), `offline.rs` (the export). The text lands only in the
operator's own log stream. No new dependencies.

**Effort.** M. **Needs real data:** no; it produces it.

### 18. Shadow evaluation of a candidate model

**What.** A `[shadow]` config section naming a second model directory, collection, and policy. After each
primary `Evaluate`, a second low-priority embedding worker evaluates the same revision against the shadow
collection and logs a `shadow` event with both verdicts.

- Shadow writes mirror the primary decision: the shadow stores when the primary stored and resets its anchor
  only on `FORCE`, so its baselines track production instead of drifting from it.
- The shadow queue drops work when full, and counts what it drops, so it can never slow the primary.
- The shadow collection is filled with feature 16, and feature 8 reports the disagreements.

**Why here.** The README calls the choice of model "a prior, not a measured win". Labeled pairs cover the cases
someone thought to label; shadowing covers the real traffic before a re-index and a cutover.

**Fit.** `config.rs`, `lib.rs` (a second worker and engine), and an internal mirror-write mode in `engine.rs`.
A second 90 MB model in memory is fine; CPU contention with the primary is the real cost and should be measured
with feature 12.

**Effort.** L. **Needs real data:** real traffic, not labels.

### 19. Structured reason codes and explanations

**Built** (D17), as `EvaluateResponse.reason_details`. The word-level diff and the neighbor search on rejection
were left out: the server stores no text to diff against, and a neighbor cut-off would be an unmeasured threshold.

**What.** `EvaluateResponse` gains `repeated ReasonDetail`: a kind (`BELOW_THRESHOLD`, `FIELD_BELOW_FLOOR`,
`ANCHOR_DRIFT`, and the three quantity reasons that now exist, `QUANTITY_BEYOND_FACTOR`, `QUANTITY_INCOMPARABLE`,
`MEASURE_CHANGED`), the field or quantity, the similarity or factor, and the limit, alongside the existing
strings. Each modified field
gets a word-level diff (`Blueberry` to `Peach`). On a rejection, optionally, a tenant-scoped neighbor search
with the new text: "the edited record now matches sku-77 at 0.97".

**Why here.** The `Reason` enum exists in `score.rs` but reaches the API only as English sentences, so a UI or
an analysis (feature 10) has to parse prose. In brand recycling the new content often duplicates another
product of the same brand, which is strong evidence that a reviewer can read at a glance.

**Fit.** The proto (additive), the mapping in `grpc.rs`, and the neighbor lookup in `engine.rs` behind a request
flag, so the common path stays one lookup. The diff is plain string work.

**Effort.** M. **Needs real data:** no.

### 20. Menu-item to brand-product matching

**What.** A `Match` RPC. It takes a retailer menu item's fields (often only a name), searches brand products
across tenants, optionally narrowed by a brand-name hint, over-fetches five times the wanted number of
candidates from Qdrant, and re-ranks them with the exact weighted score over the fields both sides have. It
returns the top matches with per-field scores and a same-product decision. `calibrate-match
catalog-groups.jsonl` measures recall and mean reciprocal rank and picks the match threshold.

**Why here.** This is the stated next use, and the reason the `brand` field is being kept in the schema. It is
also where the search limitation in D5 hurts most: a name-only query against fully filled records scores well
below 1 even for an exact match. D5 already names the fix (over-fetch and re-rank), and `catalog-groups.jsonl`
is exactly the evaluation set.

**Fit.** The proto, `engine.rs`, `offline.rs`. The exact re-rank reuses the block slicing in `score.rs`. It
probably wants its own field configuration, since menu items do not have the same fields as brand products.

**Effort.** L. **Needs real data:** yes (catalog groups).

## What to build first

1. **Structured calibration output (7).** Real labeled data is about to arrive, and nearly every other feature
   here consumes these events.
2. **Calibration diff (8).** With 7 in place, a few days' work answers the central regression question for
   models, thresholds, weights, and code alike.
3. **Lineage files (1).** A day's work, and every model trained before it exists has lineage that cannot be
   recovered.
4. **Property and state-machine tests (13).** The coming churn is when D7 and D8 could quietly break, and these
   guard them without a model or a network.
5. **Bulk re-index with anchor carry-over (16).** Whatever the first four show to be better can only reach
   production through a re-index, and today that is slow, manual, and resets drift history.

## Things noticed along the way

Not features, but found while preparing this list:

- **A README figure was derived, not measured end to end.** "Roughly 375 field texts per second when requests
  arrive concurrently" came from the batch throughput measurement, not from a test through the gRPC queue. It has
  been replaced by a measured figure for the bulk import (2,000 four-field records in 19 seconds). Throughput
  through the gRPC queue is still unmeasured, and feature 12 would measure it.
- **Choosing a checkpoint on the holdout contaminates the holdout.** See the caveat under feature 3.
- **Very long fields were a CPU cost risk (verified, and fixed).** The tokenizer processed a whole field before
  truncating it to `max_tokens`: one 1 MB field cost about 0.7 s of the single embedding thread, and 3 MB about
  2 s, under the default 4 MB gRPC message limit. Only the part of a field that could be read is tokenized now
  (D21).
