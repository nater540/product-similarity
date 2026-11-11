# Plan: the first five features

A review of `docs/feature-suggestions.md` against the code as it stands after the quantity work (D16), and a plan
for the five features that document lists under "What to build first". Written so that a session with no memory
of the conversation that produced it can carry it out. Each section is deleted as its feature lands, and the file
when the last one does.

**Done so far:** step 0 (measuring separated from judging: `Comparison::weigh`, `score::judge`,
`Evaluation::rejudge`, `Reason::code`), feature 7 (`calibrate --full`, `--ndjson`, slices, intervals; D17), and
feature 8 (`calibrate-diff`, with `--config-b`; D17), feature 1 (`provenance.json` in every model directory; D18),
and feature 13 (property tests and the engine state machine; D19). What remains is below.

Read `CLAUDE.md` first (conventions, invariants, commands), then `docs/feature-suggestions.md` for what each
feature is and why, then this file for how. The feature numbers here are that document's.

## Contents

- [Where things stand](#where-things-stand)
- [What the review found](#what-the-review-found)
- [Decisions needed from the owner](#decisions-needed-from-the-owner)
- [Order](#order)
- [Feature 16. Bulk import and re-index that keeps drift history](#feature-16-bulk-import-and-re-index-that-keeps-drift-history)
- [The other fifteen, briefly](#the-other-fifteen-briefly)
- [Out of scope](#out-of-scope)

## Where things stand

- Branch `initial-implementation`. The quantity work (`docs/plan-quantity-fields.md`, D16) is finished, verified,
  and committed. The owner asked for the features below to be carried out on best judgment, one commit each.
- `just ci` passes. `just test-parity`, `just test-qdrant`, and the full-stack test behind
  `just test-real` pass. Docker (OrbStack) is off by default on the owner's machine; `orbctl start` and
  `orbctl stop` around anything that needs Qdrant.
- Settled by the owner, and binding on everything below: Rust end to end, NDJSON-only logging through
  `similarity-log`, no Weights & Biases dependency, and **real product data never enters this repository**.
  Anything marked "needs real data" is built and tested here on synthetic data and judged where the data lives.

## What the review found

The suggestions hold up. Six things in them need correcting or deciding before work starts, and one of their
unverified notes is now verified.

1. **Per-pair calibration events cannot be nested objects (feature 7).** The suggestion has one `pair` event
   carrying per-field similarities and a list of reason codes. Logging goes through `tracing`, whose fields are a
   fixed set of names per call site with scalar values. A map keyed by field name, or a list, can only be logged
   as a preformatted string, which D15 forbids. The plan below uses flat events instead: one `pair` line, then one
   `pair_field`, `pair_quantity`, and `pair_reason` line per item, joined on `pair_id`. That is also the easier
   shape to query with `jq` and to load into a table.
2. **The strict calibration format has no pair id (features 7 and 8).** `calibrate-diff` joins two runs by
   `pair_id`, but `pairs.jsonl` has exactly four keys and carries the id only as the first word of `note`, by
   convention. The plan gives every pair an id either way: the `pair_id` key when reading `pairs.full.jsonl`, and
   otherwise a hash of the pair's content, so the same pair joins across two runs of the same file.
3. **Re-scoring under a second config needs no embedding cache (feature 8).** Weights, floors, thresholds, and
   quantity limits can all be re-applied from the per-field similarities and per-quantity factors the first run
   already wrote. An embedding cache would only matter for a change of model or field set, which is a new
   calibration run anyway. The cache is dropped from the plan.
4. **The final measurements cannot go into the lineage file as described (feature 1).** The suggestion puts "the
   final `separation` results" in `provenance.json`. The trainer deliberately saves the model before it measures
   anything (D13: the last pass must not sit in memory through minutes of scoring), and a model directory is
   written whole and never modified in place. The plan records what produced the model and leaves what the model
   scored to the results stream, which is where feature 2 will collect it.
5. **The store trait does not have to grow for a resumable import (feature 16).** `VectorStore::get` under the
   target layout already answers "is this id stored?". A cheaper `contains` is an optimization to add only if a
   measured import shows `get` to matter.
6. **A build inside Docker has no `.git` (feature 1).** `.dockerignore` excludes it, so capturing the commit with
   a `build.rs` has to fall back to an environment variable and then to `unknown`.
7. **Very long fields are a CPU cost risk: verified.** The suggestions' last note is right. Scoring one record
   whose description is 1 MB of text takes about 0.7 s longer than a normal one on this laptop, and 3 MB about
   2 s (measured with `compare`; ordinary and unbroken text alike). The tokenizer walks the whole field before
   truncating to `max_tokens`, and there is one embedding thread, so a few such requests stall every other
   caller. It is a small fix with a decision in it; see the decisions below.

The quantity work changed four suggestions, and `docs/feature-suggestions.md` now says so: features 7 and 8 carry
quantity factors, feature 19 has three more reason kinds, and features 9, 11, 13, 14, 15, and 17 each gain a
quantity case (noted per feature at the end of this file).

Two more things noticed while reviewing, neither a feature:

- **The server's `evaluate` log line prints similarities unrounded.** A 32-bit similarity widened to 64 bits is
  logged as `0.5670740008354187`. D15 says values are rounded where that helps, and the trainer does round. Round
  to four places in `grpc.rs` when feature 10 touches that line.
- **Quantities have no drift check.** D16 says so. Feature 11 (drift calibration from histories) is where real
  data will show whether one is needed; the anchor already stores what it would be measured from.

## Decisions needed from the owner

None of these blocks starting. Each has the default that will be used if no answer comes.

1. **Flat calibration events, or nested JSON written outside `similarity-log`?** Default: flat events through
   `similarity-log`, as in finding 1. The alternative is a second JSON writer with its own envelope, which D15
   was written to prevent.
2. **Should a model directory record what it scored?** Default: no, as in finding 4. Lineage records inputs.
3. **Is `proptest` acceptable as a dev-dependency (feature 13)?** It is the first test-only dependency beyond
   `tempfile`, MIT or Apache-2.0, and never reaches a release build. Default: yes, pending `just deny`.
4. **Does the work database still hold each product's original version (feature 16)?** Carrying the anchor over
   a re-index is only possible if it does. The data request already asks whether a version history exists.
   Default: build the import with `anchor` optional, so it is useful either way.
5. **What should happen to a field too long to be worth tokenizing?** Options: refuse the request above a size
   (`INVALID_ARGUMENT`), or tokenize only a prefix. Default recommendation: tokenize a prefix of
   `max_tokens * 64` bytes, cut at a character boundary, and keep hashing the whole text. Ordinary text reaches
   256 tokens well inside its first 16 KB, so its embedding does not change, and nothing a caller sends today
   starts failing. (Text made of very long unbroken runs could embed differently; it is a row of unknown tokens
   either way.) It touches what is tokenized, so it needs `just test-real`. Not started.
6. **Commit the quantity work?** It is one coherent change and the tree is clean of anything else.

## Order

1. Step 0, the small refactor that features 7, 8, and 19 all stand on. Done.
2. Feature 7, then feature 8. Both done.
3. Feature 1. Done.
4. Feature 13. Done.
5. Feature 16.

Each step should leave `just ci` green. Tests first where practical; test names are sentences.

## Feature 16. Bulk import and re-index that keeps drift history

### Design

`record-similarity reindex --config NEW.toml EXPORT.jsonl`, writing through the store and not through gRPC.

- **Input**, one line per record, in the record shape every other tool reads (a string is a text field, an
  object is a quantity):

  ```json
  {"id": "sku-1", "tenant": "brand-7", "record": {"name": "...", "weight_per_item": {"value": 3.5, "unit": "g"}},
   "anchor": {"name": "...", "weight_per_item": {"value": 3.5, "unit": "g"}}}
  ```

  `anchor` is optional: the record as it was first created, if the caller still has it.
- **Engine entry point**, `Engine::import`, with one rule that keeps D8 intact: it only ever creates. For an id
  that is not stored it writes the record at revision 1, with the anchor taken from `anchor` when given (text
  embedded as the anchor composite, quantities as the anchor's quantities) and from the record itself otherwise.
  For an id that is stored under the target layout it does nothing and reports `skipped`. It never updates, so it
  can never replace an anchor value, and running it twice is harmless. That is also what makes it resumable.
- **Embedding** in full batches on the calling thread, as `Offline` does, not through the request queue. The
  engine is generic over `TextEmbedder`, so the command supplies one that calls `Embedder::embed` directly.
- **Progress** as NDJSON: a `reindex` log line every N records, and a `finished` result with `imported`,
  `skipped`, `failed`, and elapsed seconds. A line that does not parse or does not fit the schema is reported by
  line number and counted; `--stop-on-error` makes it fatal.
- **Anchors and the first value of a field.** When `anchor` lacks a field the record has, the anchor takes the
  record's value, exactly as `extend_anchor` and `first_values` would have over time.
- A client-streaming `Import` RPC is **not** in this plan. The command covers the need; add the RPC only if a
  caller cannot run a command next to Qdrant.

### The work, in order

1. `Engine::import`, with engine tests on `MemoryStore`: creates at revision 1; skips an existing id; takes the
   anchor from `anchor`; a later `Evaluate` reports drift against the imported anchor and `Restored` against an
   anchor-only field or quantity; tenant stored; an invalid record stores nothing.
2. The line type and reader, sharing `RecordFields`.
3. The command, batching, progress events, and resume (run it twice in a test; the second run imports nothing).
4. `just reindex`. README: "Re-indexing" in Operations is rewritten around the command, and the "No bulk API"
   limitation and the "re-indexing resets drift history" sentence change to say what is now true.
5. A decision record: import only creates.

With Docker running: an ignored test that imports into a real Qdrant collection and reads back anchors and
quantities, run by `just test-qdrant` or `just test-real`.

**Effort:** M to L. **Needs real data:** no.

## The other fifteen, briefly

Not planned in detail. What the review would add to each when its turn comes:

| # | Feature | Note |
|---|---|---|
| 2 | Local run registry | `similarity_log::subscriber` is already generic over its two writers, so a file sink is a tee, not a second subscriber. Do it after feature 1; the manifest and `provenance.json` should share types. |
| 3 | Evaluation at every checkpoint | The hook is `Checkpoints::after` in the trainer's `main.rs`. A validation fold by brand hash needs `owner`, which the trainer's pair parser does not read yet. |
| 4 | Model registry and promotion gate | Depends on 2, 7, and 8. |
| 5 | Run report and compare | Depends on 2 and 7. Add a per-quantity factor chart. |
| 6 | Hyperparameter sweeps | Step 0 here is the first half of the refactor it asks for. |
| 9 | Pinned verdict snapshot in CI | The hard cases should include a unit rewrite, a size change, `%` to `mg`, and a quantity removed and refilled. |
| 10 | Production score telemetry | `measure_changed` is already on the `evaluate` log line. Round the similarities there (see above). Add per-quantity factors. |
| 11 | Drift-threshold calibration | Also report, per history, the largest factor each quantity reaches against its first value. That is the evidence for or against a quantity drift check. |
| 12 | Latency tracking and a load test | Unchanged. The long-field measurement above belongs in its report. |
| 14 | Fuzzing | Add a target for `RecordFields` and `Quantity` parsing, and quantities to the request stream. |
| 15 | Mutation testing | Add `quantity.rs`; its boundaries (`>` against `>=` at a limit, the tolerance) are verdicts. |
| 17 | Reviewer feedback capture | "`before` matches the stored baseline" has to include the quantities. |
| 18 | Shadow evaluation | Unchanged; quantities are judged once and are the same for both models. |
| 19 | Structured reason codes | Step 0's `Reason::code` is its core. Six kinds now, not three. |
| 20 | Menu-item to brand-product matching | Quantities are a natural re-ranking signal for a match and are deliberately not used by `FindSimilar` today (D16). Decide when this is designed. |

## Out of scope

- Any change to the encoder, the trainer's math, the text thresholds, or a quantity limit.
- Weights & Biases, MLflow, or any tracker that would put a service in the loop.
- The long-field cap (decision 5), until the owner picks an option.
