# Plan: structured reason codes on the wire (feature 19)

A plan for carrying each rejection reason through the gRPC API as data (a code, the field or quantity it is about,
the measured value, and the limit it was held to) beside the English sentence that is there today. Written before
any of it was built, so that a session with no memory of the planning can carry it out. Delete this file, or fold
what is still true into `docs/decisions.md`, when the work is done.

Read `CLAUDE.md` first, then this file. D7, D15, D16, and D17 in `docs/decisions.md` are the decisions it rests on.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [Decisions for the owner, with the default to use meanwhile](#decisions-for-the-owner-with-the-default-to-use-meanwhile)
- [Size](#size)
- [Out of scope](#out-of-scope)

## What it is for

A rejected `Evaluate` says why only in `EvaluateResponse.reasons`, a list of English sentences such as
"field `name` similarity 0.509 is below its floor 0.800". A reviewer's screen that wants to highlight the `name`
field, or an analysis that wants to count rejections by kind, has to parse that prose, and the prose is not a
promise: its wording and its three decimal places can change. The calibration events already solved this for the
offline tools (D17: `pair_reason` with a stable `code`). This feature gives the API the same thing.

## What already exists

Verified against the tree at `545ef20`.

- **`similarity_core::score::Reason`** (`crates/similarity-core/src/score.rs`, re-exported as
  `similarity_core::Reason`) has six variants: `BelowThreshold`, `FieldBelowFloor`, `AnchorDrift`,
  `QuantityBeyondFactor`, `QuantityIncomparable`, `MeasureChanged`.
- **`Reason::code`** returns the six stable strings `below_threshold`, `field_below_floor`, `anchor_drift`,
  `quantity_beyond_factor`, `quantity_incomparable`, `measure_changed`. Its doc comment already says they are an
  interface and must not be renamed.
- **`Reason::subject`** (the field or quantity name, or `None`), **`Reason::value`** (a similarity or a factor, as
  `f64`, or `None`), and **`Reason::limit`** (the threshold, floor, or factor limit, or `None`).
- **`impl Display for Reason`** writes the sentence the API returns today.
- **`pair_reason`** in `crates/similarity-server/src/offline/events.rs` writes `code`, `field` (from `subject`),
  `value`, and `limit`, rounded to six places, and `events::reason` reads them back. The units of an incomparable
  quantity are not on that line; the reader recovers them from the pair's `pair_quantity` line.
- **`EvaluateOutput.reasons`** in `crates/similarity-server/src/engine.rs` is a `Vec<String>`: the engine turns
  each `Reason` into its sentence (`evaluation.reasons.iter().map(ToString::to_string)`, in `Engine::revise`, near
  line 532) and the structure is lost there, before `grpc.rs` ever sees it.
- **`grpc.rs`** copies those strings into `EvaluateResponse.reasons` (field 7). It has the same pattern this plan
  needs for every other enum: `verdict`, `field_change`, `quantity_change`, each an exhaustive `match`, with a test
  that every core value has a wire value of its own.
- **`EvaluateResponse`** in `crates/similarity-proto/proto/recordsimilarity/v1/similarity.proto` uses fields 1 to
  11. **Field 12 is free.** Every categorical on the wire is a proto enum with an `_UNSPECIFIED = 0`
  (`Verdict`, `FieldChange`, `QuantityChange`, `WriteMode`).
- **`examples/client.rs`** prints `reason    {sentence}` for each string. The README's first example shows that
  output, and the README's `grpcurl` example shows the JSON with `"reasons": [...]`.
- **`FindSimilar`** already accepts an arbitrary `Record` (embedded, not stored, its own id excluded) with an
  optional tenant filter.
- **What is stored about a field's text is its hash**, not the text: `RecordVectors.hashes` holds `text_hash` of
  the normalized text (`crates/similarity-core/src/record.rs`), and `StoredRecord`
  (`crates/similarity-store/src/lib.rs`) holds no text at all.

## What the suggestion got wrong or left open

1. **Most of it is built.** The suggestion predates step 0 of the feature work. The kinds, the stable codes, the
   subject, the value, and the limit all exist in `similarity-core`. What is left is a proto message, the engine
   keeping the `Reason` instead of its sentence, and the mapping in `grpc.rs`. `docs/plan-features.md` says so too.
2. **The word-level diff cannot be done on the server.** The server never stores a field's text, only its hash
   (above), so on a revision it has the new text and nothing to diff it against. Storing the text in Qdrant's
   payload would put every product description in the vector store for a display feature, grow every point, and
   change what a stored record is (a re-import for every existing collection to get diffs on old records). The
   caller already holds both versions: it is the marketplace that is being edited. **The diff belongs in the
   reviewer's UI**, and this plan drops it. The suggestion's "plain string work" is true; it is just not this
   service's string work.
3. **The neighbor lookup on rejection is a separate feature, and partly exists.** A caller can already ask
   `FindSimilar` with the rejected record and its tenant. Doing it inside `Evaluate` would save one embedding (the
   candidate's composite is already computed there) and nothing else. Two things the suggestion did not notice:
   - `FindSimilar` scores on a different rule from `Evaluate` (a field only one side has lowers the score), so
     "matches sku-77 at 0.97" is on the search scale, which is the right scale for "is this another product", but
     must be labeled as such.
   - "Strong evidence a reviewer can read at a glance" implies showing a neighbor only above some similarity. That
     number is a measurement (CLAUDE.md: thresholds are measurements) and nothing measures it yet. Real labeled
     recycles at work would: how often a recycle's new text matches another product of the same tenant, and at
     what similarity.

   This plan leaves it out. See "Out of scope" for the interface it would have.
4. **Left undecided: enum or string on the wire, whether the sentences stay, and what an old client sees.** Settled
   below.
5. **"Needs real data: no" is right for the wire format.** Nothing here sets a threshold or a limit. The only thing
   that needs the real model is that `just test-real`'s end-to-end test, which matches on the sentence today, is
   rewritten to match on the structure; it needs only the checkpoint, not product data.

## Design

### The proto

Added to `similarity.proto`, after `QuantityScore`:

```proto
// Why a revision was rejected, as data. Each value names one kind of check; the strings beside it are the
// `code` values `calibrate --ndjson` writes on its `pair_reason` lines (the enum name without its prefix, in lower
// case), so a reason seen here and one seen in a calibration are joined on the same name.
//
// A server newer than a client may send a value the client does not know. Show the sentence at the same index of
// `EvaluateResponse.reasons` for it; never treat an unknown reason as no reason.
enum ReasonCode {
  REASON_CODE_UNSPECIFIED = 0;

  // `below_threshold`: the overall similarity is below the threshold that applied.
  REASON_CODE_BELOW_THRESHOLD = 1;

  // `field_below_floor`: one field's similarity is below that field's own floor.
  REASON_CODE_FIELD_BELOW_FLOOR = 2;

  // `anchor_drift`: the overall similarity to the record's anchor is below the drift threshold.
  REASON_CODE_ANCHOR_DRIFT = 3;

  // `quantity_beyond_factor`: one quantity changed by more than its factor limit, or moved between zero and a
  // value above zero.
  REASON_CODE_QUANTITY_BEYOND_FACTOR = 4;

  // `quantity_incomparable`: one quantity is now in a unit that cannot be converted from the one it had, and its
  // schema entry rejects that.
  REASON_CODE_QUANTITY_INCOMPARABLE = 5;

  // `measure_changed`: the measures the record is described by changed (`EvaluateResponse.measure_changed`), and
  // the server's policy rejects that.
  REASON_CODE_MEASURE_CHANGED = 6;
}

// One reason a revision was rejected.
message ReasonDetail {
  ReasonCode code = 1;

  // The field or quantity the reason is about. Set for `FIELD_BELOW_FLOOR` (a `FieldScore.field`, whose
  // `below_floor` is set) and for the two quantity codes (a `QuantityScore.field`, whose `beyond_limit` is set);
  // look there for the units and the change. Absent for the others, which are about the whole record.
  optional string subject = 2;

  // What was measured: the similarity for the three similarity codes, the factor for `QUANTITY_BEYOND_FACTOR`.
  // Absent where no number describes it: a quantity that moved between zero and a value above zero
  // (`QuantityScore.zero_crossing`), `QUANTITY_INCOMPARABLE`, and `MEASURE_CHANGED`. A similarity is the same
  // 32-bit value as the `similarity` it repeats, widened; narrowing it gives that value back exactly.
  optional double value = 3;

  // The configured limit `value` was held to: the threshold, the field's floor, the drift threshold, or the factor
  // limit. Absent for `QUANTITY_INCOMPARABLE` and `MEASURE_CHANGED`, which are yes-or-no rules.
  optional double limit = 4;
}
```

And in `EvaluateResponse`:

```proto
  // Human-readable reasons for a `VERDICT_REJECTED`. Empty otherwise. One sentence per entry of `reason_details`,
  // in the same order. The wording is for people and may change; read `reason_details` instead of parsing it.
  repeated string reasons = 7;

  // ...

  // Why a `VERDICT_REJECTED` was rejected, as data: one entry per sentence in `reasons`, in the same order. Empty
  // otherwise. A rejected response with `reasons` but no `reason_details` came from a server older than this field.
  repeated ReasonDetail reason_details = 12;
```

Why these choices:

- **An enum, not a string.** Every other categorical on this wire is an enum, and a generated client gets a typed
  value it can `match`. The interface the strings already are (D17) is kept by naming: `REASON_CODE_` plus the
  code in upper case. A test pins that rule for all six, so the enum and `Reason::code` cannot drift apart. The
  cost of an enum (a new kind needs a proto change) is a cost this repository pays anyway, since a new kind means
  a new `Reason` variant and a new sentence.
- **`subject`, `value`, `limit`: the four things `pair_reason` carries, under the names `Reason` uses.** Not
  `field`, which `pair_reason` uses, because on this wire `field` means a `FieldScore` and a subject may be a
  quantity. A `subject` always names an entry in `fields` or `quantities` of the same response, which is where the
  change, the units, and `zero_crossing` already are. Repeating the units on the reason (what
  `Reason::QuantityIncomparable` holds) would put the same fact in two places of one message; the join is cheap and
  guaranteed, and a property test holds it.
- **`double` for both numbers**, as `Reason::value` and `Reason::limit` return. A factor is a `double` on the wire
  already (`QuantityScore.factor`); a similarity widened from `f32` is exact. Not rounded: D15's rounding is for
  log lines read by people, and a client comparing `value` with `FieldScore.similarity` must get equality.
- **The sentences stay, unchanged and not deprecated.** They are what an old client shows, what a new client shows
  for a code it does not know, and what the README teaches. Marking the field `[deprecated = true]` would make
  prost put `#[deprecated]` on `EvaluateResponse::reasons`, which `grpc.rs` itself writes, and clippy at
  `-D warnings` would fail the build.
- **No `message` string on `ReasonDetail`**: the parallel `reasons` list is that string, and one copy is enough.

### How each kind of client behaves

| Client built against | Server | What it sees |
|---|---|---|
| The old proto | New | Exactly what it sees today. Field 12 is unknown to it and skipped by every proto3 decoder; `reasons` is byte for byte the same sentences in the same order. |
| The new proto | Old | `reason_details` empty on a rejection that has `reasons`. Documented above as "an older server"; show the sentences. |
| The new proto | A newer one, with a seventh code | An enum value it does not know. prost keeps it as an `i32` (`ReasonDetail::code()` returns `UNSPECIFIED`, `ReasonCode::try_from` fails); other languages' proto3 enums are open too. Show `reasons[i]`. |
| `grpcurl` (reflection) | New | A `reasonDetails` array beside `reasons` in the JSON. Reflection serves the compiled descriptor, so nothing to do. |

### The engine

`EvaluateOutput.reasons` becomes `Vec<Reason>` (the core type), and the engine stops rendering:

```rust
  /// Why the record was rejected: empty unless the verdict is [`Verdict::Rejected`], in the order they were found.
  pub reasons: Vec<Reason>,
```

```rust
      reasons: evaluation.reasons,
```

That is the whole engine change. No decision moves: which reasons exist, and in what order, is still decided by
`score::judge` and `score::judge_quantities`. The order today is: below threshold, each field below its floor in
the comparison's field order, anchor drift, then per quantity in schema order its factor and incomparable
reasons, then measure changed. The order is reported, not promised; the proto promises only that the two lists
line up.

### `grpc.rs`

Two translation functions beside `quantity_score`, and the response builds both lists from the one `Vec<Reason>`:

```rust
fn reason_code(reason: &Reason) -> pb::ReasonCode {
  match reason {
    Reason::BelowThreshold { .. } => pb::ReasonCode::BelowThreshold,
    Reason::FieldBelowFloor { .. } => pb::ReasonCode::FieldBelowFloor,
    Reason::AnchorDrift { .. } => pb::ReasonCode::AnchorDrift,
    Reason::QuantityBeyondFactor { .. } => pb::ReasonCode::QuantityBeyondFactor,
    Reason::QuantityIncomparable { .. } => pb::ReasonCode::QuantityIncomparable,
    Reason::MeasureChanged => pb::ReasonCode::MeasureChanged,
  }
}

fn reason_detail(reason: &Reason) -> pb::ReasonDetail {
  pb::ReasonDetail {
    code: reason_code(reason).into(),
    subject: reason.subject().map(str::to_owned),
    value: reason.value(),
    limit: reason.limit(),
  }
}
```

```rust
      reasons: output.reasons.iter().map(ToString::to_string).collect(),
      reason_details: output.reasons.iter().map(reason_detail).collect(),
```

The `match` is exhaustive on purpose: a seventh `Reason` fails to compile here until it has a wire value. Rendering
the sentence in `grpc.rs` is translation, not a decision; the wording still lives in `impl Display for Reason`.

### The `evaluate` log line

Unchanged. Logging the codes would mean a list on one event, which D15 and D17 rule out, or one event per reason,
which is feature 10's design to make (it is also where the line's unrounded similarities get fixed). Noted there,
not done here.

### The example client

`print_evaluation` in `examples/client.rs` prints one line per reason from the details, falling back to the
sentence for a code it does not know or a server too old to send details:

```
reason    below_threshold         0.562 < 0.900
reason    field_below_floor name  0.509 < 0.800
reason    anchor_drift            0.567 < 0.850
```

The code printed is `as_str_name()` with the prefix stripped and lower-cased, i.e. the same string as the event.
Keep the sentence too? The default here prints only the structured line, since the other lines of the client's
output are already structured; the sentences remain in `--json`-style tools such as `grpcurl`. If the owner prefers
the sentence, print `reason    {code}  {sentence}` instead; it is a one-line change.

## The work, in order

Each step leaves `just ci` passing and is one commit.

1. **Engine keeps the `Reason`.** `EvaluateOutput.reasons: Vec<Reason>`; `grpc.rs` renders the sentences
   (`ToString::to_string`), so the wire is unchanged. Rewrite the engine tests that search the sentence
   (`reason.contains("`name`")`, `contains("factor of 8.00")`, `contains("`%` to `mg`")`, `contains("measures")`,
   `contains("original record")`, in `engine.rs`) to match on the variant, which is what they meant. In
   `properties.rs` (the `rejudge` agreement near line 659) compare `Reason::code` and `Reason::subject` lists, not
   the full `Reason`: that test already tolerates `1e-6` in the similarity, so exact `PartialEq` on the `f32`
   inside a reason could fail where the sentence at three places did not. Update `tests/real_e2e.rs` likewise.
   Run `PROPTEST_CASES=5000 just test` once (the engine's output changed shape) and `just test-real`.
2. **Proto and mapping.** Add `ReasonCode`, `ReasonDetail`, and field 12; update the doc comment on field 7; add
   `reason_code` and `reason_detail` to `grpc.rs`; the unit tests and the gRPC end-to-end test below.
3. **Client and docs.** `examples/client.rs`; the README (re-run both examples, see below); D17 gains a paragraph.

## Tests

Fast, no model, no network, no Qdrant:

- `grpc.rs`
  - `every_reason_has_a_wire_code_of_its_own`: all six variants map to six distinct values, none `UNSPECIFIED`
    (the pattern of `every_quantity_change_has_a_wire_value_of_its_own`).
  - `a_wire_reason_code_is_named_after_the_code_calibrate_writes`: for each variant,
    `reason_code(&r).as_str_name()` is `"REASON_CODE_"` plus `r.code()` in upper case. This is the test that keeps
    D17's interface and the API's in step.
  - `a_reason_detail_carries_the_subject_value_and_limit_of_its_reason`: one case per variant, including
    `QuantityBeyondFactor { factor: None, .. }` (value absent, limit present) and `MeasureChanged` (all absent).
  - `a_similarity_on_the_wire_narrows_back_to_the_similarity_it_repeats`: `value as f32 == similarity` for a
    `FieldBelowFloor` built from an awkward `f32`.
- `similarity-server/src/lib.rs` (the in-process tonic test that already checks the quantity rejection)
  - `a_rejection_carries_one_reason_detail_per_sentence_in_the_same_order`: the existing three-reason text
    rejection and the existing weight rejection; `reasons.len() == reason_details.len()` and each pair agrees on
    code (the sentence is the `Display` of the detail's reason).
  - `an_accepted_or_created_record_carries_no_reason_details`.
- `engine.rs`: the rewritten tests from step 1 keep their names where they are already sentences; the ones that
  matched prose now assert the variant and its subject, e.g.
  `a_recycled_record_is_rejected_and_the_baseline_is_untouched` asserts `FieldBelowFloor { field: "name", .. }`.
- `properties.rs` (engine level, so the rule holds for every generated history)
  - `every_reason_about_a_field_or_quantity_names_one_the_response_reports_beyond_its_limit`: for each reason with
    a subject, a `FieldBelowFloor` names a `fields` entry with `below_floor`, and a quantity reason names a
    `quantities` entry with `beyond_limit`. This is the guarantee the proto's `subject` comment makes.
  - The existing `rejected iff reasons non-empty` assertion (near line 357) stays as is and now covers the
    structured list.

Needs the real model (`just test-real`, ignored by default):

- `tests/real_e2e.rs`: the recycled-record assertion switches from `contains("`name`")` to the variant. No
  numbers change, so the shipped thresholds are untouched and nothing needs re-calibrating.

Needs Qdrant: nothing new. The store is not touched.

Needs real data: nothing. Whether a reviewer's screen is better for this can only be seen at work, with the UI that
consumes it, but no number in this plan comes from data.

## Documentation to update

- **`README.md`**
  - The first example's client output (the three `reason` lines): re-run `just client evaluate sku-1
    data/examples/vape.json` against `just serve-in-memory` and paste what it prints. CLAUDE.md: every command in
    the README was run.
  - The `grpcurl` JSON example: re-run it to get the `reasonDetails` array. `grpcurl` is not installed on the
    owner's machine (`brew install grpcurl`, or have the owner run it); do not hand-write the JSON.
  - "What `Evaluate` returns": add `reason_details` to the list of response fields, with one paragraph: the six
    codes, that they are the `pair_reason` codes, that `subject` names an entry in `fields` or `quantities`, and
    what to do with a code a client does not know.
  - The Rust and Python client snippets: show matching on `reason_details` (the Rust one by `code()` and
    `subject`; the Python one by `pb.ReasonCode.Name`). Re-run the Python one as the memory note describes
    (`uv run --with grpcio-tools` in a scratch directory), and the Rust one via the example.
  - Line 808, which lists the reason codes for the events: add that the API carries the same codes.
  - The table row "`VERDICT_REJECTED`, with `reasons`": "with `reasons` and `reason_details`".
- **`docs/decisions.md`**: D17 gains a short paragraph, "The API carries the same codes", saying why an enum, the
  naming rule and the test that pins it, why the sentences stay, and why the word diff and the neighbor lookup were
  left out (the server stores no text; a neighbor cut-off would be an unmeasured threshold). No new D-number: the
  decision is D17's, extended to the wire.
- **`CLAUDE.md`**: in the "Calibration events are flat, and their names are an interface" item, add that the
  same codes are `ReasonCode` on the wire, named `REASON_CODE_` plus the code, and that a new `Reason` needs a
  new code in both places, never a reused number.
- **`docs/plan-features.md`**: mark 19 done in the table, with the two parts that were dropped.
- **The proto** is the API's documentation (CLAUDE.md); the comments above are part of the work.

## Dependencies on other features

None unbuilt. It assumes only what is already there: `Reason` with `code`, `subject`, `value`, `limit`.

Two later features read what this one adds:

- **Feature 10 (production telemetry)** would log reason codes. The interface it can assume: `EvaluateOutput.reasons`
  is a `Vec<Reason>`, so it can log `reason.code()` per reason without parsing anything.
- **Feature 17 (reviewer feedback)** would show a reviewer why a revision was rejected. The interface it can assume:
  `EvaluateResponse.reason_details`, with `subject` joining to `fields` and `quantities`.

## Decisions for the owner, with the default to use meanwhile

1. **Enum or string for the code.** Default: **enum** (`ReasonCode`), named so that it maps one to one onto the
   `pair_reason` strings, pinned by a test. A string would match the events literally and add a kind without a
   proto change, but loses the typed client and breaks with every other categorical on this wire.
2. **Keep the English sentences.** Default: **keep `reasons` as it is, not deprecated**, parallel to
   `reason_details`. It is what old clients and unknown codes fall back to, and deprecating it would trip our own
   clippy through prost's generated `#[deprecated]`.
3. **Units on an incomparable reason.** Default: **not repeated**; `subject` joins to the `QuantityScore`, which
   has `previous_unit` and `unit`. If the reviewer UI at work would rather not join, add `previous_unit` and `unit`
   as fields 5 and 6 of `ReasonDetail`; additive, any time.
4. **The word-level diff.** Default: **dropped from this feature**; it belongs in the reviewer UI, which holds both
   texts. Doing it here means storing every field's text in the vector store, which is a larger decision (storage,
   what a stored record is, a re-import) than this feature.
5. **The neighbor lookup on rejection.** Default: **deferred**, as its own feature; callers can use `FindSimilar`
   with the rejected record and tenant today. Build it once real recycles at work show how often a recycle matches
   another product of the same tenant and at what similarity, since showing a neighbor only above some score sets
   a threshold that must be measured.
6. **What the example client prints.** Default: the structured line only (`reason    field_below_floor name
   0.509 < 0.800`). Alternative: the code and the sentence.

## Size

**Small.** One proto message, one enum, one field; one type change in `EvaluateOutput` that only moves where a
`to_string` happens; two translation functions; tests mostly rewritten from prose matching to variant matching.
No store, config, threshold, model, or event change. The time goes into the README, whose two examples have to be
re-run rather than edited, and one of which needs `grpcurl` installed.

## Out of scope

- **A word diff of modified fields.** See decision 4.
- **Neighbors on rejection.** See decision 5. If it is built later, the interface this plan would leave room for:
  `EvaluateRequest.neighbors_on_reject` (`uint32`, field 5, zero for none, capped like `FindSimilar.limit`), and
  `EvaluateResponse.neighbors` (`repeated Neighbor`, field 13), searched with the candidate's composite already in
  hand, filtered to the request's tenant, excluding the record's own id, run after the per-id lock is released so
  it never lengthens the write path, and documented as on `FindSimilar`'s scale. A failed search would be logged
  and leave the verdict alone: a verdict must never depend on a display aid.
- **Reason codes on the `evaluate` log line.** Feature 10.
- **Explanations beyond the code**, such as which tokens moved a similarity. Nothing in the encoder offers that
  cheaply, and an attribution method would be a model-side feature with its own validation.
