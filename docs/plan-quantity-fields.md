# Plan: quantity fields

> **Status, 2026-11-11: carried out.** Everything under "The work, in order", "Documentation to update", and
> "Verification" is done, on `initial-implementation`. What is still true here has been folded into
> `docs/decisions.md` (D16), including the open questions. This file is kept only so that the result can be checked
> against the plan; delete it once that is done. Where the result differs from the plan:
>
> - `QuantityScore` has one field the plan did not list, `incomparable`. A quantity that comes back in a unit of
>   another dimension is `Restored` (that is what happened to it) and needs a way to say it cannot be compared.
> - The two units are reported as `previous_unit` and `unit`.
> - The unit table also accepts spelled-out names (`grams`, `ounces`, `milliliters`, ...), `lbs`, `floz`, and `pct`.
> - A quantity re-sent as the same amount in another unit is not a write, as text re-saved with other whitespace is
>   not. Any other change in a quantity is.
> - `Policy::with_threshold` replaces the engine's `Policy::new(threshold, anchor)` for a per-request threshold,
>   which would otherwise have dropped `reject_on_measure_change`.
> - The calibration report's summary line now reads "plus field floors and any quantity limits".
> - The README's `merchant-1` tenant was left alone, as the plan asks.

A plan for teaching the service about the numeric fields on a product record (weight, volume, THC, CBD), written
so that a session with no memory of the conversation that produced it can carry it out. It was written before any
of it was built. Delete this file, or fold what is still true into `docs/decisions.md`, when the work is done.

Read `CLAUDE.md` first (conventions, invariants, commands), then this file. `README.md` and `docs/decisions.md`
(D1 to D15) explain everything the plan refers to.

## Contents

- [Where things stand](#where-things-stand)
- [What was decided, and by whom](#what-was-decided-and-by-whom)
- [How a product record maps onto a request](#how-a-product-record-maps-onto-a-request)
- [Why quantities are not embedded](#why-quantities-are-not-embedded)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Documentation to update](#documentation-to-update)
- [Verification](#verification)
- [Open questions, with the default to use meanwhile](#open-questions-with-the-default-to-use-meanwhile)
- [Out of scope](#out-of-scope)

## Where things stand

- Branch `initial-implementation`, three commits, the last being `f2471b2`. There is no `main` and no remote.
- Uncommitted when this was written: a one-line addition to `README.md`, `docs/feature-suggestions.md`, and this
  file. Commit only when the owner asks.
- `just ci` passes (205 fast tests, clippy, cargo-deny), as do `just test-parity`, `just test-real`, and
  `just test-qdrant`. Docker (OrbStack) is off by default on the owner's machine; `orbctl start` and
  `orbctl stop` around anything that needs Qdrant.
- The service today knows only **text fields**: `name` (weight 4, required, floor 0.80), `category` (3), `brand`
  (1), `description` (2). Each is embedded, compared by cosine similarity, and combined into a weighted score.

## What was decided, and by whom

The owner's decisions and facts. Treat them as settled:

- **First use is brand products, not retailers' menu items.** The tenant is the brand that owns the product.
- **`brand` on the record is a numeric foreign key.** The caller sends the brand's *name* as the `brand` text
  field. The field is kept in the schema, although it is constant per brand, because a later use may match menu
  items to brand products.
- **Categories live in a join table** (a product can have a root category and a sub category). The caller sends
  them **as text**, as one `category` field.
- **Rust end to end.** No Python in any build, serve, or training path.
- **All logging is NDJSON** through `similarity-log`.
- **Real product data never enters this repository or this account.** Training and calibration on real data
  happen in the owner's work environment. Everything here is built and tested on synthetic data.
- **Weights & Biases must not become a dependency.**

My recommendations, which the owner has seen and not objected to. Reasonable to revisit:

- Quantities are compared as numbers and never embedded (evidence below).
- Quantity checks are **report-only by default**; enforcement is switched on per field in the config.
- Quantities stay out of the weighted similarity score, so the existing thresholds keep their meaning.

One earlier idea was dropped: comparing category ids exactly ("key" fields). Categories arrive as text, so there
is nothing to build for them.

## How a product record maps onto a request

The owner listed these as common fields on a product record:

```text
name                        description
weight_per_item_value       weight_per_item_unit
volume_per_item_value       volume_per_item_unit
thc_per_item_min_value      thc_per_item_max_value      thc_per_item_unit
cbd_per_item_min_value      cbd_per_item_max_value      cbd_per_item_unit
```

| On the record | In the request | Notes |
|---|---|---|
| Brand foreign key | `tenant`, as a string | A product moving to another brand is refused by the tenant check (D14). |
| Brand name, looked up from the key | `fields["brand"]` | Constant per brand. Kept for later matching. |
| Root and sub category, through the join | `fields["category"]`, e.g. `Edibles > Gummies` | Names, never ids. Root first, then sub. If a product has several category rows, a fixed sort order, or a reordering looks like an edit. |
| `name`, `description` | `fields["name"]`, `fields["description"]` | Unchanged from today. |
| `weight_per_item_value` and `_unit` | `quantities["weight_per_item"]` | New. A single value. |
| `volume_per_item_value` and `_unit` | `quantities["volume_per_item"]` | New. A single value. |
| `thc_per_item_min_value`, `_max_value`, `_unit` | `quantities["thc_per_item"]` | New. A range. |
| `cbd_per_item_min_value`, `_max_value`, `_unit` | `quantities["cbd_per_item"]` | New. A range. |

Twelve columns become two text fields and four quantities. The four quantity names are configuration, not code:
the server knows the kind "quantity", and the config names the ones in use.

## Why quantities are not embedded

Measured with the shipped model (`mdbr-leaf-mt`, 256 dimensions), scoring each pair of strings as a `name`:

| Before | After | What it is | Similarity |
|---|---|---|---|
| `1 g` | `1000 mg` | The same quantity | 0.742 |
| `3.5 g` | `3.5 grams` | The same quantity | 0.970 |
| `weight 3.5 g` | `weight 0.125 oz` | The same quantity (roughly) | 0.832 |
| `THC 21.3 %` | `THC 22.8 %` | A routine batch update | 0.892 |
| `THC 18 % to 22 %` | `THC 19 % to 23 %` | A routine batch update | 0.951 |
| `3.5 g` | `28 g` | Eight times the size | 0.758 |
| `THC 100 mg` | `THC 10 mg` | A tenth of the potency | 0.924 |
| `THC 100 mg` | `THC 1000 mg` | Ten times the potency | 0.961 |
| `THC 22 %` | `THC 22 mg` | A different kind of product | 0.873 |
| `weight 3.5 g, THC 22 %` | `volume 355 ml, THC 10 mg` | A different kind of product | 0.727 |

The ordering is backwards: a tenfold potency change scores higher than a routine batch update, and an identical
weight in other units scores lowest of all. Embedded as text these fields would add noise to every verdict. This
table belongs in the new decision record (D16).

Compared as numbers they give three signals the text model cannot:

- **How much a quantity changed**, after unit conversion. `1 g` to `1000 mg` is no change; `3.5 g` to `28 g` is a
  factor of 8. This is the size and potency evidence for same-brand variant changes, the hard cases.
- **A quantity that stopped being comparable.** THC in `%` becoming THC in `mg` is not a bigger or smaller
  number, it is a different kind of product.
- **Which measures a product has.** A weight disappearing as a volume appears is flower becoming a beverage.

## Design

### A quantity

```rust
pub struct Quantity { pub min: f64, pub max: f64, pub unit: String }
```

A single value is `min == max`. Valid means: both finite, both `>= 0`, `min <= max`, unit not blank.

### Units

A built-in table maps a unit (trimmed, lowercased) to a dimension and a factor to that dimension's base unit:

| Dimension | Base | Units to start with |
|---|---|---|
| Mass | mg | `mcg`/`µg` 0.001, `mg` 1, `g` 1000, `kg` 1e6, `oz` 28349.523125, `lb` 453592.37 |
| Volume | ml | `ml` 1, `cl` 10, `l` 1000, `fl oz` 29.5735295625 |
| Fraction | % | `%`/`percent` 1 |

A unit not in the table is its own dimension, comparable only with the identical string. The real unit
vocabulary is unknown here (see the open questions), so the table must be easy to extend and an unknown unit must
never be an error.

### Comparing two quantities

Per quantity field, with the baseline `B`, the candidate `C`, and the anchor's first value `A` where needed:

| Change | When | Factor |
|---|---|---|
| `Unchanged` | Equal after conversion, within a relative tolerance of 1e-6 | 1 |
| `Modified` | Both present, same dimension, different | see below |
| `Incomparable` | Both present, different dimensions | none |
| `Added` | Only `C`, and the anchor has never had it | none |
| `Removed` | Only `B` | none |
| `Restored` | Only `C`, but the anchor has it: compared against the anchor | as `Modified` or `Incomparable` |

**Factor**: after conversion, `max(ratio(B.min, C.min), ratio(B.max, C.max))` where `ratio(x, y)` is the larger
over the smaller, so it is always at least 1 and "eight times bigger" and "an eighth" both read 8. If exactly one
of a pair is zero the factor is unbounded: report no factor and set `zero_crossing`, and treat it as beyond any
limit. Zero to zero is unchanged.

`Restored` exists for the same reason as for text fields (D8): without it a quantity could be swapped in two
steps, blank then refill. So the anchor keeps the first value of every quantity, exactly as it keeps the first
value of every text field.

**Measure change.** One derived flag per evaluation, `measure_changed`: true when any quantity is `Incomparable`,
or when in the same edit at least one quantity is `Removed` and a different one is `Added` or `Restored`. Filling
in a quantity that was simply missing does not set it.

### Policy

Quantities never enter the weighted similarity. They can reject on their own, and only if configured:

```toml
[quantities.weight_per_item]
# Both optional. With neither, the quantity is compared and reported and never rejects.
reject_above_factor = 4.0
reject_incomparable = true

[quantities.thc_per_item]

[policy]
# Optional, default false.
reject_on_measure_change = true
```

New rejection reasons: quantity beyond its factor (field, factor, limit), quantity incomparable (field, both
units), measure changed. A rejected revision still writes nothing (D7).

### What is stored

`StoredRecord` gains the current quantities and the anchor's quantities, both `BTreeMap<String, Quantity>`. In
Qdrant they go in the point payload, read with `#[serde(default)]` so that points written before this change load
as "no quantities". Quantities are not part of any vector, so **the layout id does not change** and no re-index
is needed (D12): an existing record simply reports `Added` the first time a quantity arrives.

Engine rules, mirroring the text rules:

- A change in quantities alone is a change: it must be written, and must bump the revision.
- An accepted ordinary write stores the candidate's quantities, and gives the anchor the first value of any
  quantity it has never had. It never replaces an anchor value.
- A `FORCE` write resets the anchor's quantities along with its vector.
- `DRY_RUN` and a rejection write nothing.

### API

- `Record` gains `map<string, Quantity> quantities = 4`, with `message Quantity { double min = 1; double max = 2;
  string unit = 3; }`.
- `EvaluateResponse` gains `repeated QuantityScore quantities = 10` (field, change, optional factor,
  `zero_crossing`, the two units, `beyond_limit`) and `bool measure_changed = 11`.
- A `QuantityChange` enum beside `FieldChange`, with `QUANTITY_CHANGE_INCOMPARABLE` as its extra value.
- `DescribeResponse` lists the configured quantities and their limits.
- An unknown quantity name, or an invalid quantity, is `INVALID_ARGUMENT`, like an unknown text field.
- `FindSimilar` ignores quantities: it searches vectors.

All additions are new field numbers; nothing existing changes.

### Files people write by hand

`compare`, `calibrate`, and the example client read a record as one JSON object. A text field stays a string. A
quantity is an object, in either spelling:

```json
{"name": "Blueberry Gummies 100mg", "category": "Edibles > Gummies", "brand": "WYLD",
 "weight_per_item": {"value": 3.5, "unit": "g"},
 "thc_per_item": {"min": 18, "max": 22, "unit": "%"}}
```

The value's JSON type decides which it is, and the key must be configured as that kind, or the error names the
key. Calibration pairs use the same shape inside `before` and `after`.

## The work, in order

Each step should leave `just ci` green. Write the tests first where that is practical (see `CLAUDE.md`). Test
names are sentences.

### 1. Core: the quantity type and its comparison

New module `crates/similarity-core/src/quantity.rs`, exported from `lib.rs`:

- `Quantity` (serde, with the `value` spelling accepted on input), validation, `Dimension`, the unit table,
  conversion.
- `QuantityChange`, `QuantityScore`, and `compare(baseline, anchor, candidate) -> QuantityScore` for one field.
- `first_values(anchor, candidate)`: the anchor's quantities after an accepted write.

Tests: every row of the comparison table; unit conversion both ways (`1 g` and `1000 mg` unchanged; `3.5 g` to
`28 g` is 8; `28 g` to `3.5 g` is 8); ranges (the factor is the worse end); zero crossing; unknown units equal
only to themselves; a restored quantity scored against the anchor; invalid quantities refused with the reason.

### 2. Core: schema and policy

- `crates/similarity-core/src/schema.rs`: a `QuantitySpec { name, reject_above_factor, reject_incomparable }`
  and a way to attach quantities to a `Schema` without changing `Schema::new` (it has dozens of callers): for
  example `Schema::new(fields)?.with_quantities(specs)?`. A quantity name must not collide with a text field
  name. `layout_id` must not include quantities.
- `crates/similarity-core/src/score.rs`: `Policy` gains `reject_on_measure_change`; `Reason` gains the three
  new variants with `Display`; `Evaluation` gains the quantity scores and `measure_changed`; a function that
  scores all quantities of a record and appends reasons. Keep the existing `evaluate` signature working for
  callers that have no quantities.
- `crates/similarity-core/src/record.rs`: `PreparedRecord` validates quantities against the schema.

Tests: report-only by default (a factor of 100 with no limit configured is reported and not a reason); each
limit rejects when set; `measure_changed` in each case of its definition; the layout id is unchanged by adding a
quantity.

### 3. Store

- `crates/similarity-store/src/lib.rs`: the two new maps on `StoredRecord`.
- `memory.rs`: nothing beyond carrying them.
- `qdrant.rs`: the payload fields, with `#[serde(default)]`.
- `tests/contract.rs`: quantities round-trip through `put` and `get` for both stores. For Qdrant, a point
  written without the new payload keys reads back with empty maps (write one with the raw client, as the
  index-repair test does).

### 4. Server: config, engine, gRPC

- `crates/similarity-server/src/config.rs`: `[quantities.<name>]` tables and `policy.reject_on_measure_change`.
  `deny_unknown_fields` is on, so add the fields explicitly. The shipped example must still parse.
- `engine.rs`: `RecordInput` gains quantities; `evaluate` and `revise` apply the engine rules above; the
  "identical resubmission" check must include quantities; `EvaluateOutput` gains the scores and the flag.
- `grpc.rs`: conversions only. Keep decisions out of it.
- The proto, with documentation comments in the style already there.

Engine tests: a quantity-only change is accepted, stored, and bumps the revision; an identical resubmission
with quantities does not; blank then refill is `Restored` and scored against the first value; `FORCE` resets the
anchor's quantities; with a limit set, a quantity beyond it rejects and nothing is written; `measure_changed`;
a record stored before quantities existed reports `Added`. One gRPC round trip in `lib.rs`. Extend
`tests/real_e2e.rs` with a quantity on one of its evaluations.

### 5. Offline tools and the client

- `crates/similarity-server/src/offline.rs`: replace the `FieldMap` alias (string to string) with a type holding
  texts and quantities, deserialized from one JSON object by value type. `LabeledPair` keeps
  `deny_unknown_fields` at the top level.
- `render_comparison`: a quantities section.
- `render_calibration`: per quantity field, the range of change factors among edits and among recycles, the
  count of `Incomparable` per label, and the count of `measure_changed` per label. That is what will show where
  `reject_above_factor` belongs. Keep `render_calibration` under clippy's line limit by extracting helpers, as
  `render_field_ranges` already does.
- `examples/client.rs`: `evaluate` accepts quantity objects in its JSON file and prints the quantity scores.
- `crates/similarity-train`: no change. Its parser takes only string values, so quantity objects are ignored;
  add a test that says so.

### 6. Sample data

- `data/examples/`: add quantities to the three example records.
- `data/calibration.sample.jsonl`: add quantities to enough pairs that the new report section is not empty,
  including a unit rewrite (`1 g` to `1000 mg`) labeled `edit`, a batch potency update labeled `edit`, a size
  change, and a `%` to `mg` switch labeled `recycle`. The file is synthetic and says so; keep it saying so.
- `config.example.toml`: the four `[quantities.*]` tables, **with no limits set**, and comments explaining that
  limits are to be chosen from calibration on real data. Do not invent thresholds.

The text thresholds must not move. `just calibrate` should print the same text numbers as before (edits from
0.941, recycles up to 0.850, best threshold 0.895); if it does not, something entered the text score.

## Documentation to update

- **`docs/decisions.md`**: a new D16, "Quantities are compared, not embedded", with the measured table above,
  the report-only default, why they stay out of the score, and why the layout does not change.
- **`README.md`** (every command and number in it must be run and measured, per `CLAUDE.md`):
  - "Making requests": quantities in the examples, and the new response fields in "What `Evaluate` returns".
  - "Integrating it": a new subsection, "Mapping your records", carrying the mapping table above, including the
    two caller rules for category text (names not ids; a fixed order).
  - "How a verdict is reached": the quantity rules.
  - "Choosing thresholds": reading the new calibration section.
  - "Configuration reference": `[quantities.*]` and `policy.reject_on_measure_change`.
  - The examples still use `merchant-1` as the tenant. The owner has not asked for that to change; ask before
    renaming it to a brand.
- **`CLAUDE.md`**: quantities never enter the weighted score or the layout id; limits are measurements, like
  thresholds.
- **`docs/data-request-prompt.md`**: name the twelve columns; ask for the four quantities as numbers with their
  units, under the four names above, never rendered into text; ask for the distinct unit strings per quantity
  with counts (this answers the first open question); extend the strict `pairs.jsonl` format to allow the four
  quantity keys with object values; say that `brand` is the brand's name and `category` the display path built
  root first.
- **`docs/feature-suggestions.md`**: feature 7 (structured calibration output) and feature 19 (reason codes)
  should mention quantities once this exists.

## Verification

Before calling it done:

1. `just ci`
2. `just test-parity` (the encoder and trainer must be untouched; this proves it)
3. With Docker running: `just test-real` and `just test-qdrant`
4. `just calibrate`: text numbers unchanged, new quantity section present
5. By hand against `just serve-in-memory`, with the example client: a unit rewrite reports `Unchanged`; an
   eightfold size change reports a factor of 8 and is still accepted (report-only); with a limit set in a copy of
   the config, the same edit is rejected and the stored record is untouched.

## Open questions, with the default to use meanwhile

None of these blocks the work.

1. **What unit strings actually occur?** Unknown here. Default: the table above, case-insensitive, with unknown
   units comparable only to themselves. The data-request prompt change will get the real list.
2. **Is THC ever expressed as `mg/g`, or per package instead of per item?** Default: treat an unknown unit as its
   own dimension; add `mg/g` to the Fraction dimension (factor 0.1) only once it is confirmed to occur.
3. **Are `min` and `max` always both set?** Default: if the caller has only one, it sends it as both.
4. **Does zero mean "none" or "not measured"** (CBD of 0, say)? Default: zero is a value; a move between zero and
   non-zero is reported as a zero crossing and, like everything else, rejects only if a limit is configured.
5. **Is there a pack-size or item-count field?** "Per item" suggests one. It is not in the owner's list. Default:
   leave it out; it can be added as a fifth quantity with no code change.
6. **Where do the limits go?** Unknown until real pairs are calibrated in the owner's work environment. Default:
   ship with none.

## Out of scope

- Exact comparison of category or brand ids. Both arrive as text.
- Using quantities in `FindSimilar` or in the menu-item matching use.
- Any change to the encoder, the trainer, or the text thresholds.
- Features from `docs/feature-suggestions.md`.
