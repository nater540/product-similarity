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
   `max_tokens`. A long field is now tokenized a part at a time, until the token limit is full (D21). The first
   version of that fix cut at a byte count and let padding hide a field; see D21.

Building them turned up three more:

- **Numbers read from JSON depended on which crates were in the build.** `serde_json` is off by one bit about one
  time in ten without `float_roundtrip`, which only the trainer enabled, and cargo unifies features per build. A
  property test found it; the feature is now set for the workspace (D19).
- **A quantity limit is not placed like a threshold.** Most recycles leave a quantity alone, so there is never a
  gap between the two labels to put a limit in. It is suggested from the largest factor any edit shows (D17).
- **A README figure was derived, not measured.** "Roughly 375 field texts per second" under concurrent requests
  was replaced by a measured figure for the bulk import, and throughput through the gRPC queue is now measured
  by `just load` (D26).

Still open, noticed along the way and not acted on:

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
6. **A field too long to be worth tokenizing whole is read in parts, not refused.** The tokenizer is given a
   head of the text and a longer one only if the first did not fill the token limit; the whole text is still
   hashed. No request that succeeds today fails, and no embedding changes (D21). The alternative, refusing the
   request above a size, is a one-line policy change if that is preferred.

## The other fifteen, briefly

Not planned in detail. What the review would add to each when its turn comes:

| # | Feature | Note |
|---|---|---|
| 2 | Local run registry | **Built** (D23): `calibrate` and the trainer each leave a directory under `runs/`; `runs list`. |
| 3 | Evaluation at every checkpoint | **Built** (D24): `pairs carve`, the trainer scores every checkpoint on the validation split by AUC and names the best, `calibrate --model`. |
| 4 | Model registry and promotion gate | **Built** (D31): `model.expected_fingerprint` (opt-in), `model promote` with its gate, and `models/registry.ndjson` with `model list`, `register`, and `retire`. |
| 5 | Run report and compare | **Built** (D28): `runs compare` as text, `runs report` as one HTML file with curves and calibration histograms. |
| 6 | Hyperparameter sweeps | **Half built** (D29): `calibrate-sweep` for weights, floors, and limits, with no model; every training option is a flag. The command that trains at every point of a grid is not built. |
| 9 | Pinned verdict snapshot in CI | **Built** (D22): curated cases, a snapshot checked with and without the model, `calibrate-diff --fail-on moved`, pairs with an `anchor`. |
| 10 | Production score telemetry | `measure_changed` is already on the `evaluate` log line. Round the similarities there. Add per-quantity factors and reason codes (`Reason::code`). |
| 11 | Drift-threshold calibration | **Built** (D27): `calibrate-drift` replays histories, measured first and judged afterwards; reports legitimate drift, the suggestion, and quantities walked. |
| 12 | Latency tracking and a load test | **Built** (D26): `record-similarity load`, worker counters, `elapsed_ms` on request lines. The figure is in README "Performance". |
| 14 | Fuzzing | The state machine in `similarity-server`'s `properties` module is the structure-aware target, ready to be driven by a fuzzer. Add a target for `RecordFields` and `Quantity` parsing. |
| 15 | Mutation testing | **Built** (D25): `just mutants` over the files that decide a verdict; 517 mutants at the last run, none surviving, after a test for each survivor earlier runs found. |
| 17 | Reviewer feedback capture | "`before` matches the stored baseline" has to include the quantities. The event it emits should be a line `calibrate --full` reads. |
| 18 | Shadow evaluation | **Half built** (D30): a candidate policy judges every live revision and decides none. A candidate model in the shadow is not built; it would fill its collection with `import`. |
| 19 | Structured reason codes | **Built** (D17, "The API carries the same codes"): `EvaluateResponse.reason_details`. The word diff and the neighbor lookup were left out, for the reasons given there. |
| 20 | Menu-item to brand-product matching | Quantities are a natural re-ranking signal for a match and are deliberately not used by `FindSimilar` today (D16). Decide when this is designed. |

## What to build next

With real labeled pairs about to exist, in this order:

1. **Nothing here, first.** Run `calibrate --full` on the real `pairs.full.jsonl`, read the slices, the `unsure`
   pairs, and the quantity lines, and set thresholds and limits from that. Keep the `--ndjson` output of every
   run: it is what the next one is compared against.
2. **Feature 19** (reason codes on the wire). Built.
3. **Feature 9** (a pinned snapshot in CI). Built.
4. **Feature 11** (drift calibration from histories). Built; run it once `sequences.jsonl` exists. It answers the two open
   questions about drift: where `anchor_threshold` belongs, and whether quantities need a drift check.
5. **Feature 2** (the run registry). Built.
