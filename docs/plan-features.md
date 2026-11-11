# Plan: the first five features

A review of `docs/feature-suggestions.md` against the code as it stands after the quantity work (D16), and a plan
for the five features that document lists under "What to build first". Written so that a session with no memory
of the conversation that produced it can carry it out. **Nothing in it is built.** Delete each section as its
feature lands, and the file when the last one does.

Read `CLAUDE.md` first (conventions, invariants, commands), then `docs/feature-suggestions.md` for what each
feature is and why, then this file for how. The feature numbers here are that document's.

## Contents

- [Where things stand](#where-things-stand)
- [What the review found](#what-the-review-found)
- [Decisions needed from the owner](#decisions-needed-from-the-owner)
- [Order](#order)
- [Step 0. Separate measuring from judging](#step-0-separate-measuring-from-judging)
- [Feature 7. Structured calibration output with slices](#feature-7-structured-calibration-output-with-slices)
- [Feature 8. Calibration diff](#feature-8-calibration-diff)
- [Feature 1. Lineage file in every model directory](#feature-1-lineage-file-in-every-model-directory)
- [Feature 13. Property-based and state-machine tests](#feature-13-property-based-and-state-machine-tests)
- [Feature 16. Bulk import and re-index that keeps drift history](#feature-16-bulk-import-and-re-index-that-keeps-drift-history)
- [The other fifteen, briefly](#the-other-fifteen-briefly)
- [Out of scope](#out-of-scope)

## Where things stand

- Branch `initial-implementation`. The quantity work (`docs/plan-quantity-fields.md`, D16) is finished, verified,
  and committed. The owner asked for the features below to be carried out on best judgment, one commit each.
- `just ci` passes with 251 fast tests. `just test-parity`, `just test-qdrant`, and the full-stack test behind
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

1. Step 0, the small refactor that features 7, 8, and 19 all stand on.
2. Feature 7, then feature 8. Real labeled pairs are about to exist, and nearly everything else consumes these
   events.
3. Feature 1. A day's work, and it has to exist before the first real fine-tuning run, not before the first
   calibration.
4. Feature 13. Independent of the others; it can be done at any point, and is worth doing before feature 16
   changes the engine.
5. Feature 16.

Each step should leave `just ci` green. Tests first where practical; test names are sentences.

## Step 0. Separate measuring from judging

`score::evaluate` measures (per-field cosines) and judges (thresholds, floors) in one call, and
`Evaluation::judge_quantities` does both for quantities. Re-scoring a cached calibration under another config
(feature 8) needs the judging half alone, and structured reasons (features 7, 10, 19) need a stable name for each
reason.

In `crates/similarity-core/src/score.rs`:

- `Reason::code(&self) -> &'static str`: `below_threshold`, `field_below_floor`, `anchor_drift`,
  `quantity_beyond_factor`, `quantity_incomparable`, `measure_changed`. Add accessors for the field or quantity
  name, the measured value, and the limit, each `Option`. These six strings become API; a test pins them.
- `judge(schema, policy, comparison: &mut Comparison, anchor_similarity) -> Vec<Reason>`: recomputes
  `similarity` from the field scores and the schema's current weights, resets and sets `below_floor`, and returns
  the reasons. `evaluate` becomes `compare` followed by `judge`.
- The same split for quantities: a function that only measures (what `quantity::compare` does per quantity, over
  a schema) and one that only judges a slice of `QuantityScore` (resets and sets `beyond_limit`, derives
  `measure_changed`, returns reasons). `Evaluation::judge_quantities` keeps its signature and calls both.

Tests: judging a comparison twice gives the same result; judging under other weights equals comparing from
scratch under those weights; `evaluate` is unchanged (every existing test still passes untouched).

`crates/similarity-server/src/offline.rs` is 927 lines after the quantity work. Before adding to it, turn it into
a module directory: `offline/mod.rs` (`Offline`, `RecordFields`, `LabeledPair`, parsing), `offline/calibrate.rs`
(scoring and summarizing), `offline/report.rs` (the text renderers). No behavior change; the tests move with the
code.

## Feature 7. Structured calibration output with slices

### Design

**Input.** `calibrate` gains `--full`, which reads `pairs.full.jsonl` as the data request defines it:

- A lenient line type beside the strict `LabeledPair`: `pair_id`, `label` (any string), `split`, `owner`,
  `variant_change`, `label_source`, `label_confidence`, `note`, `before`, `after`. Unknown keys are ignored.
- `before` and `after` are projected onto the schema: a key the schema defines as a text field is kept if its
  value is a string, a key it defines as a quantity is kept if its value is a quantity object, and every other
  key is dropped. The dropped keys are **counted by name and reported**, so a misspelled `descripton` shows up as
  812 ignored values and not as a quietly weaker score.
- `--split NAME` (default: all) and `--min-confidence high|medium|low` (default `medium`, matching what the data
  request puts in `pairs.jsonl`).
- Pairs labeled anything but `edit` or `recycle` are scored and reported as a distribution, and never counted as
  errors or used for a suggestion.

The strict format keeps working unchanged, and its `deny_unknown_fields` stays.

**Pair ids.** `pair_id` from the line when present; otherwise the first 16 hex digits of the SHA-256 of the
pair's canonical `before` and `after` (keys sorted, as `serde_json` writes a `BTreeMap`).

**Output.** `--ndjson` replaces the text report with result events (`target: similarity_log::RESULT`, stdout,
never filtered). All values are numbers, booleans, or short strings; an absent optional is left out.

| `message` | One per | Fields |
|---|---|---|
| `calibration` | run | `fingerprint`, `dimension`, `max_tokens`, `threshold`, `anchor_threshold`, `pairs_sha256`, `pairs`, `split`, `min_confidence` |
| `pair` | pair | `pair_id`, `label`, `split`, `owner`, `variant_change`, `label_source`, `label_confidence`, `category`, `similarity`, `accepted`, `wrong`, `measure_changed` |
| `pair_field` | pair and field | `pair_id`, `field`, `change`, `similarity`, `weight`, `below_floor` |
| `pair_quantity` | pair and quantity | `pair_id`, `quantity`, `change`, `factor`, `zero_crossing`, `incomparable`, `previous_unit`, `unit`, `beyond_limit` |
| `pair_reason` | pair and reason | `pair_id`, `code`, `field`, `value`, `limit` |
| `field_range` | field and label | `field`, `label`, `count`, `low`, `high` |
| `quantity_range` | quantity and label | `quantity`, `label`, `count`, `low`, `high`, `incomparable`, `zero_crossings` |
| `suggestion` | run | `threshold`, `margin`, `false_rejects`, `false_accepts` |
| `quantity_suggestion` | quantity | `quantity`, `reject_above_factor`, `edits_high`, `recycles_low`, or `overlap: true` |
| `slice` | slice value | `by` (`variant_change`, `category`, `label_source`, `owner`), `value`, `edits`, `recycles`, `false_rejects`, `false_accepts`, score lows and highs |
| `operating_point` | target | `false_accept_rate`, `threshold`, `false_reject_rate`, `low`, `high` (the interval) |
| `ignored_key` | key | `key`, `count` |
| `summary` | run | `pairs`, `edits`, `recycles`, `unsure`, `false_rejects`, `false_accepts` |

`category` on a `pair` event is the top-level category: the text of `before.category` up to the first ` > `,
which is the separator the README tells callers to build the path with.

**Slices and intervals.**

- Slices by `variant_change`, top-level category, `label_source`, and `owner`. The `owner` slice prints only the
  ten largest, since it exists to show one brand dominating.
- Intervals on the two error rates by bootstrap, **resampling owners and not pairs**: pairs from one brand are
  not independent, and resampling pairs would understate the spread. 1,000 resamples, seeded, default seed 0.
  Without `owner` (the strict format) the report says intervals need `--full`.
- Operating points: the threshold and false-reject rate at 1%, 2%, and 5% false-accept rate, from the overall
  score alone, with the same intervals.
- A `reject_above_factor` suggestion per quantity: the midpoint (geometric, since factors are ratios) of the gap
  between the largest factor in edits and the smallest in recycles, or a statement that they overlap.

The bootstrap needs a seeded generator. The trainer has one (`SplitMix64` in `similarity-train/src/data.rs`), and
the server must not depend on the trainer (D13). Move it to `similarity-core` (a small `rng` module) and have the
trainer use it from there. `just test-parity` must still pass, since the batch order depends on it.

**The text report** stays the default and gains the slices, the operating points, and the quantity suggestions
when the input has what they need.

### The work, in order

1. Step 0.
2. The lenient reader and the projection, with the ignored-key count. Tests: unknown keys counted; a quantity
   under a text field's name is dropped and counted; `_id` and `_one_sided` companions are ignored; a strict file
   still refuses an unknown key.
3. Pair ids, and `--split` and `--min-confidence`.
4. `unsure` pairs: scored, excluded from errors and suggestions. Test with a file that is all `unsure`.
5. The events. Test by capturing the result stream (`similarity_log::testing` does this for the trainer) and
   parsing each line: every number is a JSON number, no line has a `null`, and `pair_field` similarities re-add
   to the `pair` similarity under the schema's weights.
6. Slices, the generator move, the bootstrap, operating points, quantity suggestions. Tests: a bootstrap with one
   owner has zero width; the same seed gives the same interval; a slice's counts sum to the whole.
7. Text report additions.

### Documentation and verification

- README, "Choosing thresholds": `--full`, `--ndjson`, reading slices and intervals. Replace the `jq` projection
  in "Fine-tuning the model", step 3, with `calibrate --full --split holdout`.
- `docs/data-request-prompt.md`: the final check becomes `calibrate --full` on file 1, and file 2 can be dropped
  from the request if the owner agrees (it exists only because the tool could not read file 1).
- A new decision record for the event shapes and the brand-clustered bootstrap.
- A small synthetic `data/calibration.sample.full.jsonl`, saying it is synthetic, so that the README's examples
  can be run. `just calibrate` on the existing sample must print the numbers it prints today.

**Effort:** M. **Needs real data:** the slices and intervals are only meaningful on it.

## Feature 8. Calibration diff

### Design

`record-similarity calibrate-diff OLD.ndjson NEW.ndjson`, reading two files of feature 7's events.

- A reader for those events in `offline/` (plain `serde` structs, one per `message`), shared with nothing else
  yet. It ignores lines whose `target` is not `result`, so a file that captured both streams still reads.
- Refuses two runs over different pair sets (`pairs_sha256` differs) unless `--intersect`, which diffs the pairs
  both have and says how many each side had alone.
- Reports, as text by default and as events with `--ndjson`:
  - every verdict flip, sorted by score change, labeled as a new false accept, a new false reject, or a fix;
  - per-slice change in error rates and in margin;
  - per-field and per-quantity shifts in the two ranges.
- `--fail-on new-false-accepts|any-flip|errors-up` sets the exit code, for scripts.
- `--config-b CONFIG` takes one run and re-judges it under another config, with no model loaded: step 0's
  `judge` over the cached `pair_field` and `pair_quantity` lines. It refuses a config whose field or quantity
  names differ from the run's, and says that a change of model or field set needs a new calibration run.

### The work, in order

1. The event reader, round-tripped against feature 7's writer in a test.
2. The join and the flip report. Tests: a pair that flips each way; identical runs report nothing and exit 0.
3. `--intersect` and the refusal without it.
4. `--config-b`. Test: re-judging a run under its own config reproduces it exactly; raising a field's weight
   changes the overall score to what a fresh calibration under that config gives (use the tiny test encoder).
5. `--fail-on` and exit codes.

README: a "Comparing two calibrations" subsection under "Choosing thresholds". `just calibrate-diff`.

**Effort:** M. **Needs real data:** no.

## Feature 1. Lineage file in every model directory

### Design

`provenance.json`, written by `record-similarity-train` into `--out` and every checkpoint, and by `model fetch`
into the directory it fills. It records what produced the model, not what the model scored (finding 4).

```json
{"format": 1, "written_at": "2026-11-11T05:52:42Z", "tool": "record-similarity-train", "version": "0.1.0",
 "commit": "f2471b2", "dirty": false, "digest": "0123456789abcdef",
 "base": {"dir": "models/mdbr-leaf-mt", "digest": "f46e48ee0ea2dd02", "provenance": {"...": "..."}},
 "data": [{"role": "pairs", "path": "pairs.full.jsonl", "sha256": "...", "examples": 4120}],
 "data_fingerprint": "...", "options": {"...": "..."}, "max_tokens": 256, "device": "metal",
 "resumed": [{"from": "models/x.checkpoints/step-000120", "step": 120}], "step": 240, "total_steps": 240}
```

What makes this safe, each checked against the code:

- **It does not change the fingerprint.** `Embedder::load` hashes the files the encoder read plus
  `tokenizer.json` (`embed.rs`), and nothing else in the directory.
- **It is not inherited by accident.** `SentenceEncoder::save` copies only the files it read and a fixed list of
  side files, so a base model's `provenance.json` is never copied into a model trained from it. It is nested
  under `base.provenance` on purpose instead.
- **It is written whole.** Inside the `write_whole` closures of `checkpoint::save_model` and `checkpoint::write`,
  after `encoder.save`, so a directory under its final name always has it.

The pieces:

- `similarity_core::model`: a public function for the content digest (the loop now inside `Embedder::load`), so
  the trainer can compute the base's digest and the saved model's own digest without building an `Embedder`. A
  test pins the digest of the tiny test model before and after the move; `just test-real` shows the real
  fingerprint is still `f46e48ee0ea2dd02`.
- The commit: a `build.rs` in the two binary crates that reads `RECORD_SIMILARITY_COMMIT`, then asks `git`, then
  settles for `unknown`, with a dirty flag. The `Dockerfile` passes the variable as a build argument.
- Data files are hashed as read (`data::read` already returns the whole text).
- A resume reads the checkpoint's `provenance.json` and appends to `resumed`.
- `model fetch` writes `{tool, repo, revision, files: [{name, sha256, verified}]}`. It adds the file to an
  already-fetched directory too, which changes no fingerprint.
- The server logs `provenance` fields on its `model loaded` line when the file is there (base digest, commit),
  and says nothing when it is not. It never refuses a model for lacking one.

### The work, in order

1. The shared digest function, with the pinning test.
2. The `provenance` type (serde) in `similarity-core`, since both binaries write it and the server reads it.
3. `build.rs` and the Dockerfile argument.
4. The trainer: base, data, options, resume chain; written in `save_model` and `write`. Tests: a saved model and
   a checkpoint both carry it; a model trained from a model with provenance nests it; a resumed run lists the
   resume; an extra file in a model directory does not change the fingerprint.
5. `model fetch`. `fetch` downloads over HTTPS and its tests have no stand-in for the Hub, so keep writing the
   file apart from downloading and test the writer on a temporary directory of files.
6. The server's log line.

README: a short "Where a model came from" under "Fine-tuning the model". `models/README.md`. A decision record.
`just test-parity` and `just test-real` (the loader is touched).

**Effort:** S. **Needs real data:** no.

## Feature 13. Property-based and state-machine tests

`proptest` as a dev-dependency of `similarity-core` and `similarity-server` (decision 3). `cargo test` stays free
of model, network, and Qdrant: everything runs on the tiny test encoder, `MemoryStore`, and `WordBagEmbedder`.
Keep each property to about 64 cases so the fast suite stays fast, and commit the `proptest-regressions` files.

Properties, with the quantity ones the suggestion predates:

| Area | Property |
|---|---|
| Composite (D5) | The cosine of two composites equals the weighted mean of the field cosines when both have the same fields filled. |
| Scores | Every similarity is finite and within `[-1, 1]`. |
| Normalization | `normalize_text` is idempotent, and `text_hash` is stable under it. |
| Thresholds | `suggest_threshold` equals a brute-force search over every candidate cut. |
| Batching (D9) | A text embedded alone and inside a merged, length-bucketed batch agree within tolerance. |
| Quantity factor | `compare(a, b)` and `compare(b, a)` give the same factor, and it is at least 1. |
| Quantity units | Rewriting both values in another unit of the same dimension changes no factor (within 1e-9 relative). |
| Quantity identity | A quantity compared with itself, in any known unit spelling, is `Unchanged`. |
| Quantity judging | `judge_quantities` never changes `similarity`, never removes a reason, and with no limit configured adds none. |
| Anchor (D8, D16) | `extend_anchor` and `first_values` never replace a value the anchor holds. |
| Record files | A `RecordFields` written as JSON reads back equal. |

**The engine state machine.** Random sequences of `Evaluate` (every mode, with and without quantities, tenants,
and a per-request threshold), `Delete`, and `FindSimilar`, run against a reference model that is deliberately
naive: a map from id to the last accepted record and the first value of each field and quantity. After every
step:

- a rejected or `DRY_RUN` revision left the stored record byte-for-byte as it was (D7);
- an accepted `ENFORCE` write changed no anchor value, text or quantity, that the anchor already held (D8);
- a `FORCE` write made the anchor equal the record;
- the revision counter rose by exactly one if and only if `stored`;
- the tenant rules held (D14);
- `stored` is false whenever the response is an error.

Work order: the pure properties in `similarity-core` first (cheap, and they exercise step 0), then the state
machine in `similarity-server`. Anything a property finds is fixed in the same change, with the shrunk case kept
as an ordinary named test.

**Effort:** M. **Needs real data:** no.

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
