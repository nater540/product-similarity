# Plan: what was built from the feature suggestions, and what is next

`docs/feature-suggestions.md` lists twenty features that could be added. This file started as a plan for the five
it puts first. All five are built. What is left here is a record of how that went, what the review of the
suggestions found, and notes for the fifteen that remain. Delete it when it stops being useful.

## Contents

- [What was built](#what-was-built)
- [What the review found](#what-the-review-found)
- [Decisions that were taken](#decisions-that-were-taken)
- [The other fifteen, briefly](#the-other-fifteen-briefly)
- [What to build next](#what-to-build-next)

## What was built

In this order, one commit each, on `initial-implementation`. Each has a decision record that says why it is the
way it is; the README says how to use it.

| What | Where to read about it |
|---|---|
| Step 0: measuring separated from judging (`Comparison::weigh`, `score::judge`, `Evaluation::rejudge`), and a stable `Reason::code` | D17 |
| Feature 7: `calibrate --full`, `--ndjson`, slices, intervals, operating points, quantity limit suggestions | D17, README "Choosing thresholds" |
| Feature 8: `calibrate-diff`, with `--config-b` to judge a run again under another config without a model | D17, README "Comparing two calibrations" |
| Feature 1: `provenance.json` in every model directory | D18, README "Where a model came from" |
| Feature 13: property tests and the engine state machine | D19, README "Development" |
| Feature 16: `import`, which only creates and can carry a record's original version over as its anchor | D20, README "Importing a catalog" |
| A cap on how much of a field is handed to the tokenizer (not one of the twenty; found by the review) | D21 |

Real product data never entered this repository. Everything was built and tested on synthetic data; features 7
and 8 in particular only show their worth on real labeled pairs, where the data lives.

## What the review found

Seven things in the suggestions needed correcting or deciding, and each is now reflected in what was built:

1. **Per-pair calibration events cannot be nested objects.** The logging layer writes scalars under fixed names.
   The events are flat instead: one `pair` line, then one line per field, quantity, and reason.
2. **The strict calibration format has no pair id.** Every pair now has one: the export's, or a hash of its
   content.
3. **Re-scoring under a second config needs no embedding cache.** Weights, floors, thresholds, and quantity
   limits are re-applied to measurements the first run wrote down.
4. **The final measurements cannot go into the lineage file.** A model is saved before anything is measured and
   never modified in place. `provenance.json` records what produced a model, not what it scored.
5. **The store trait did not have to grow for a resumable import.** An import that only creates is resumable by
   being run again.
6. **A build inside Docker has no `.git`.** The commit is passed in as a build argument. The image was built
   once that way to check: the `provenance.json` of the model baked into it names the commit.
7. **Very long fields were a CPU cost risk: verified, and fixed.** One 1 MB field cost about 0.7 s of the single
   embedding thread, and 3 MB about 2 s, because the tokenizer walks the whole field before truncating to
   `max_tokens`. Only the part that could be read is tokenized now (D21).

Building them turned up three more:

- **Numbers read from JSON depended on which crates were in the build.** `serde_json` is off by one bit about one
  time in ten without `float_roundtrip`, which only the trainer enabled, and cargo unifies features per build. A
  property test found it; the feature is now set for the workspace (D19).
- **A quantity limit is not placed like a threshold.** Most recycles leave a quantity alone, so there is never a
  gap between the two labels to put a limit in. It is suggested from the largest factor any edit shows (D17).
- **A README figure was derived, not measured.** "Roughly 375 field texts per second" under concurrent requests
  was replaced by a measured figure for the bulk import. Throughput through the gRPC queue is still unmeasured.

Still open, noticed along the way and not acted on:

- **The server's `evaluate` log line prints similarities unrounded** (`0.5670740008354187`). D15 says values are
  rounded where that helps, and every other event now does. Round it when feature 10 touches that line.
- **Quantities have no drift check.** A factor limit applies to each revision against the one before it. Feature
  11 is where real histories would show whether that matters.

## Decisions that were taken

The owner asked for these to be taken on best judgment. Each is easy to reverse.

1. **Flat calibration events through `similarity-log`**, not nested JSON written beside it.
2. **A model directory records inputs, not results.**
3. **`proptest` as a dev-dependency**, without its process-forking features. `just deny` passes.
4. **The import takes the original version as optional.** Whether the work database still has each product's
   first version decides whether drift history can be carried over; the data request asks.
5. **The import is a command, named `import`**, not `reindex` as first planned: loading a catalog for the first
   time is the same operation. No streaming RPC.
6. **A field too long to be worth tokenizing is cut, not refused.** The tokenizer is given the first
   `max_tokens * 64` bytes; the whole text is still hashed. No request that succeeds today fails, and no
   embedding of ordinary text changes (D21). The alternative, refusing the request above a size, is a one-line
   policy change if that is preferred.

## The other fifteen, briefly

Not planned in detail. What the review would add to each when its turn comes:

| # | Feature | Note |
|---|---|---|
| 2 | Local run registry | `similarity_log::subscriber` is already generic over its two writers, so a file sink is a tee, not a second subscriber. The manifest and `provenance.json` should share types; the build stamp (`provenance::Build`) is the start of that. |
| 3 | Evaluation at every checkpoint | The hook is `Checkpoints::after` in the trainer's `main.rs`. A validation fold by brand hash needs `owner`, which the trainer's pair parser does not read yet. |
| 4 | Model registry and promotion gate | Now mostly assembly: `provenance.json` identifies a model, `calibrate --ndjson` records a calibration with its fingerprint, and `calibrate-diff --fail-on` is the gate. |
| 5 | Run report and compare | Depends on 2. The calibration half can be drawn from the events that exist now. |
| 6 | Hyperparameter sweeps | The scoring it needs is in `similarity-core` already (step 0). Needs real data. |
| 9 | Pinned verdict snapshot in CI | A committed `calibrate --ndjson` run of curated hard cases, and `calibrate-diff --fail-on any-flip` against a fresh one, is most of it. The hard cases should include a unit rewrite, a size change, `%` to `mg`, and a quantity removed and refilled. |
| 10 | Production score telemetry | `measure_changed` is already on the `evaluate` log line. Round the similarities there. Add per-quantity factors and reason codes (`Reason::code`). |
| 11 | Drift-threshold calibration | Replays `sequences.jsonl` through the engine; `Engine::import` with `anchor` is the way to seed each history. Also report, per history, the largest factor each quantity reaches against its first value. |
| 12 | Latency tracking and a load test | Would replace the last unmeasured throughput figure. |
| 14 | Fuzzing | The state machine in `similarity-server`'s `properties` module is the structure-aware target, ready to be driven by a fuzzer. Add a target for `RecordFields` and `Quantity` parsing. |
| 15 | Mutation testing | Add `quantity.rs` and `offline/stats.rs`; their boundaries (`>` against `>=` at a limit, the tolerance) are verdicts. |
| 17 | Reviewer feedback capture | "`before` matches the stored baseline" has to include the quantities. The event it emits should be a line `calibrate --full` reads. |
| 18 | Shadow evaluation | The shadow collection is filled with `import`; disagreements are a `calibrate-diff`. |
| 19 | Structured reason codes | `Reason::code`, `subject`, `value`, and `limit` exist. What is left is the proto message and the mapping in `grpc.rs`. Six kinds now, not three. |
| 20 | Menu-item to brand-product matching | Quantities are a natural re-ranking signal for a match and are deliberately not used by `FindSimilar` today (D16). Decide when this is designed. |

## What to build next

With real labeled pairs about to exist, in this order:

1. **Nothing here, first.** Run `calibrate --full` on the real `pairs.full.jsonl`, read the slices, the `unsure`
   pairs, and the quantity lines, and set thresholds and limits from that. Keep the `--ndjson` output of every
   run: it is what the next one is compared against.
2. **Feature 19** (reason codes on the wire). Small now, and the first thing a reviewer's screen will want.
3. **Feature 9** (a pinned snapshot in CI). Also small now, and it turns "run `just test-real` after touching
   the encoder" from a rule into a check.
4. **Feature 11** (drift calibration from histories), once `sequences.jsonl` exists. It answers the two open
   questions about drift: where `anchor_threshold` belongs, and whether quantities need a drift check.
5. **Feature 2** (the run registry), before the first real fine-tuning runs, so that their results are kept.
