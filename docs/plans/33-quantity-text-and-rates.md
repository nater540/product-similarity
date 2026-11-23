# Plan: amounts read from text, and units per something (features 33 and 34)

A plan for two additions to the quantity system that D47 lists as not done: reading an amount that arrives as text
(`"3.5g"`, `"1/8 oz"`, `"18-22%"`) into a `Quantity`, and understanding a unit that is one thing per another
(`mg/ml`, `mg/g`, `mg per serving`). Written on 2026-11-22, before any of it was built, so that a session with no
memory of the planning can carry it out. Reviewed the same day against the owner's answers, which are recorded
under [the decisions](#decisions-for-the-owner-with-the-default-to-use-meanwhile). Delete this file, or fold what
is still true into `docs/decisions.md`, when the work is done.

Read `CLAUDE.md` first, then this file. D16 (quantities are compared, not embedded), D25 (mutation testing), D44
(the judging digest), and D47 (the `Quantities` service) in `docs/decisions.md` are the decisions it rests on.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [Found while planning](#found-while-planning)
- [Design: amounts from text (33)](#design-amounts-from-text-33)
- [Design: units per something (34)](#design-units-per-something-34)
- [Where the two meet](#where-the-two-meet)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [What can only be validated at work](#what-can-only-be-validated-at-work)
- [Decisions for the owner, with the default to use meanwhile](#decisions-for-the-owner-with-the-default-to-use-meanwhile)
- [Size](#size)
- [Out of scope](#out-of-scope)

## What it is for

The `Quantities` service (D47) made the comparison of amounts callable by other systems. Those systems do not
always hold an amount as a number and a unit. They hold `"3.5g"` in one column, `"1/8 oz"` on a menu, `"18-22%"`
in a lab result. Today each caller writes its own parser before it can ask this service anything, and each of
those parsers is a place where `1,000 mg` becomes 1, or `1/8` becomes 1. Reading an amount is as intricate as
comparing one, and it belongs in the same place, with the same tests.

Units are the other half. A potency is rarely a bare mass: it is `mg/g` on flower, `mg/ml` in a tincture, `mg`
per serving on an edible. To the unit table all of those are unknown units, each comparable only with its own
exact spelling. So `200 mg/g` against `20 %`, which are one amount, reads as `INCOMPARABLE`, and so does
`10 mg/ml` against `10 g/l`. D16 saw the first of these coming ("add it to the percentage dimension only once it
is confirmed to occur") and left it for real data.

Neither changes what a verdict is. The first adds a way in. The second makes more pairs of amounts comparable
that already should be.

## What already exists

- **`similarity_core::quantity`** (`quantity.rs`, about 800 lines with its tests): `Quantity { min, max, unit }`,
  `Quantity::check`, the `UNITS` table of eleven units in three dimensions, `unit(raw) -> (Dimension, factor)`,
  `compare`, and since D47 `Quantity::in_unit`, `in_base_unit`, `dimension`, and `units()`. A unit is matched
  after trimming, collapsing inner whitespace, and lowercasing. Anything not in the table is
  `Dimension::Other(normalized)`, never an error.
- **The `Quantities` service** (`crates/similarity-server/src/quantities.rs`): `Compare`, `Convert`, `ListUnits`.
  It translates and calls the core and holds no arithmetic of its own. It answers "these do not convert" as data,
  not as a status code.
- **A record takes numbers.** `Record.quantities` is a map of name to `Quantity`. A quantity sent as text in
  `fields` is refused by name (``"`weight_per_item` is a quantity, not a text field"``, `record.rs`), and the
  README's rule for callers is "a quantity is never rendered into text".
- **The questions are already asked.** `docs/data-request-prompt.md` asks for every distinct value of the four
  unit columns with a count, and whether THC is ever recorded per package. No answer has come back into this
  repository, and none can: real data stays at work.
- **Where "per item" lives today:** in the quantity's name (`thc_per_item`, unit `mg`), not in its unit.
- **The fuzzing plan** (`14-fuzzing.md`, deferred) already names `Quantity` parsing and `quantity::unit` as
  targets. A text parser is a better one.

## Found while planning

1. **The judging digest does not cover the unit table.** `Description::judging` (`engine.rs`) writes out the
   policy, the matching settings, every field, and every `QuantitySpec`. Which units convert is not among them,
   and it is a rule of judging: with `reject_incomparable` or `policy.reject_on_measure_change` set, a server
   whose table knows `mg/g` and one whose table does not give two verdicts for one edit, and report one digest.
   This is true today for any line added to `UNITS`. Feature 34 changes the unit rules more than a line does, so
   it has to fix this (step 6 below), and the fix is worth making even if 34 is never built.
2. **`mg/g` and `mg / g` are two different unknown units today.** Collapsing whitespace leaves the spaces around
   the slash. A record whose potency is re-saved with a space reads as `INCOMPARABLE` against itself. Feature 34
   removes this as a side effect.
3. **A parsed amount needs no new type.** Everything `"1 1/2 oz"` or `"18-22%"` can say fits in
   `Quantity { min, max, unit }`. The parser's whole output is a `Quantity` or a reason it is not one.
4. **The two features share one rule.** A unit may contain a digit only as the amount of a denominator
   (`mg/100 g`). The text parser needs that rule to tell `25 mg/5 ml` from `10 x 10 mg`, so the grammar of 33 is
   written against the unit syntax of 34 even if 34 comes second.

## Design: amounts from text (33)

### The rule: the whole string is the amount, or it is not an amount

The parser reads a string that *is* an amount. It does not find amounts inside prose: `"Blueberry Gummies 100mg"`
is a name, and what a name says stays in the name (D16). It never guesses. A string it cannot read exactly is
refused with a code that says why, because a wrong number that looks right is worse than no number: it becomes a
baseline.

### What is read

After trimming and collapsing whitespace. `unit` is everything after the number, kept as written, and handed to
`quantity::unit` like any other unit, so an unknown unit is still not an error.

| Text | Read as | Note |
|---|---|---|
| `3.5 g`, `3.5g`, `3.5 G` | 3.5 `g` | A space before the unit is optional. |
| `.5 g`, `0.5 g` | 0.5 `g` | |
| `1,000 mg`, `12,500 mg` | 1000 `mg`, 12500 `mg` | A comma only between groups of exactly three digits. |
| `1/8 oz` | 0.125 `oz` | A fraction of two positive integers. |
| `1 1/2 oz`, `1-1/2 oz` | 1.5 `oz` | A whole number and a proper fraction. See the note on the hyphen. |
| `½ oz`, `1½ oz`, `1 ½ oz`, `⅛ oz` | 0.5, 1.5, 1.5, 0.125 `oz` | The Unicode fractions, alone or after a whole number. |
| `18-22%`, `18 - 22 %`, `18–22 %`, `18 to 22 %` | 18 to 22 `%` | A range: a hyphen, a dash, or `to`. |
| `18% - 22%`, `18 g - 22 grams` | 18 to 22 `%`, 18 to 22 `grams` | Both ends may carry the unit, if it is one unit to the table. The last end's spelling is kept. |
| `1/8 - 1/4 oz` | 0.125 to 0.25 `oz` | Either end may be any number form. |
| `10 mg/ml`, `25 mg / 5 ml`, `5mg per serving` | 10 `mg/ml`, 25 `mg / 5 ml`, 5 `mg per serving` | The unit is feature 34's. |
| `2 pack`, `3.5 g net` | 2 `pack`, 3.5 `g net` | Unknown units, comparable with themselves. The survey below is where a person sees `g net`. |

**The hyphen.** `1-1/2 oz` is one and a half ounces, not a range from 1 to a half. A hyphen with no space on
either side, between a whole number that is at least 1 and a proper fraction, makes a mixed number. Those are
exactly the strings whose reading as a range would run backwards, so no string has two readings: `0-1/2 oz` is a
range from nothing to a half, and `1 - 1/2 oz`, with its spaces, is a range and is refused as a backwards one.

**A comma groups thousands, and does nothing else.** The first group is one to three digits and does not start
with a zero, and every later group is exactly three. So `0,500 g` is not five hundred grams: it is somebody's
half a gram, and is refused with `3,5 g`.

**What a unit may be made of.** Letters, spaces, `%`, `.`, `/`, a hyphen between two letters (`pre-rolls`), and
digits only as the amount of a denominator (`mg/100 g`). Anything else after the number is not a unit this reads:
`3.5 g (approx)`, `20 %+`, and `3.5 g, 1/8 oz` are refused, where a looser rule would have stored
`g (approx)` as a unit comparable with itself. (Found in review: the first draft refused a digit and nothing
else.)

### What is refused, and the code for each

The codes are an interface in the sense `Reason::code` is (D17): add, never rename or repurpose. The owner's
rule for all of them (2026-11-22): data that is wrong is refused, and nothing here tries to rescue it. The codes
exist so that a survey can count what kind of wrong a column holds, not so that each can grow a reading later.

| Code | Example | Why it is not guessed |
|---|---|---|
| `empty` | `""`, `"  "` | |
| `no_number` | `g`, `THC 22%`, `eighth`, `up to 22 %` | Does not start with a number. A word in front makes it prose. |
| `no_unit` | `3.5`, `18% - 22` | A number is not a measurement (`Quantity::check` refuses a blank unit). |
| `ambiguous_separator` | `3,5 g`, `1,00 mg`, `0,500 g` | Three and a half, or thirty-five with a typo. US English reads neither. Any comma that is not grouping thousands. |
| `bad_number` | `1.2.3 g`, `1/0 oz`, `0/8 oz`, `5. g` | |
| `negative` | `-1 g` | |
| `backwards_range` | `22-18 %`, `1 - 1/2 oz` | |
| `mixed_units` | `500 mg - 1 g` | A range in two units is somebody's mistake more often than their meaning. |
| `multiple` | `10 x 10 mg`, `2 × 0.5 g`, `10 mg x 10` | Ten pieces of 10 mg is 100 mg a package or 10 mg an item. Only the caller knows which its quantity means. |
| `qualified` | `<1 %`, `~3.5 g`, `≥ 20 %` | Starts with `<`, `>`, `≤`, `≥`, `~`, `≈`, or `±`. "Less than one" is a claim about a range this cannot make for the caller. |
| `unreadable_unit` | `3.5 g (1/8 oz)`, `5 mg 2`, `20 %+`, `1e3 mg` | What follows the number is not made of what a unit is made of (above). |
| `too_large` | a 1 followed by 400 zeros, in `g` | It is a number and a unit, and too large to compare: `Quantity::check` refuses it. |

Changed in review: there is no `position` (where reading stopped). Nothing asked for it, a caller cleaning a
column acts on the code, and it would have been an interface with a rule for every refusal. A field can be added.
A qualifier is recognized by its symbol only: a list of words (`up to`, `about`, `approx`) would never be
complete, and those are `no_number` like any other word in front. `not_a_measurement` is named for the one thing
it can be, `too_large`.

### Core: `crates/similarity-core/src/amount.rs` (new, synchronous, no I/O)

```rust
/// Reads a string that is an amount, the whole of it.
///
/// # Errors
///
/// [`NotAnAmount`] with the reason and where reading stopped. Never a guess.
pub fn parse(text: &str) -> Result<Quantity, NotAnAmount>;

pub enum NotAnAmount { Empty, NoNumber, NoUnit, AmbiguousSeparator, /* one for each row above */ }

impl NotAnAmount {
  /// The stable name: `ambiguous_separator`.
  pub fn code(&self) -> &'static str;
}
```

- Hand-written and small: a cursor over `char_indices`, one function for each number form. No regular
  expressions (the workspace has none on this path) and no indexing, per the lints.
- The owner prefers a maintained crate to hand-written infrastructure, so crates.io was searched on 2026-11-22
  before any of this was written. What exists: implementations of UCUM, a strict standard that refuses `fl oz`
  and `1 1/2`; parsers of typesetting and SI dimensions; parsers of Kubernetes quantities; and recipe-ingredient
  parsers (`ingredient`, `ingreedy-rs`), which read prose and guess, the opposite of the rule above. None gives a
  refusal a reason. The grammar is this domain's vocabulary, which is the part worth owning.
- A decimal is parsed by `str::parse::<f64>` after the separators are checked and removed, so `0.1` is the
  nearest double and a fraction is one division. `1/3 oz` is 0.333... and that is what is stored.
- The result is passed through `Quantity::check` and returned `trimmed()`, so everything `parse` returns is a
  quantity a record would accept.
- It writes nothing back: no "canonical form" of the text. The unit is kept as written, as the README tells
  callers to send it.

**Named amounts are not read, and no table for them is built yet.** `eighth`, `quarter`, `half`, and `zip` are
how the trade names weights, and what each means is the owner's company's to say (D36): an eighth is an eighth of
an ounce, 3.54 g, and is sold as 3.5 g; `half` is half an ounce of flower and half a gram of concentrate. Until
the owner supplies a row (decision 2), `eighth` is refused as `no_number`, and the survey's `--examples` is where
such words would be seen if a column holds them. The first draft shipped an empty `NAMED` table; review removed
it, since a lookup with no rows is code no test can reach. When there is a row: `NAMED: &[(&[&str], f64, &str)]`
beside `UNITS`, consulted when the string starts with no number. A row is one line and a test.

### API: `Quantities.Parse` (additive)

```proto
// Reads amounts written as text. One answer for each text, in order. A text that is not an amount is an answer,
// with the reason, and not an error: a caller checking a column wants every answer and a count of each reason.
//
// Errors: `INVALID_ARGUMENT` for more than 1000 texts.
rpc Parse(ParseAmountsRequest) returns (ParseAmountsResponse);

message ParseAmountsRequest { repeated string texts = 1; }

message ParsedAmount {
  // The amount, when the text is one. Usable as a `Record` quantity as it is.
  optional Quantity quantity = 1;
  // Why not, when it is not. `AMOUNT_PROBLEM_UNSPECIFIED` when `quantity` is present.
  AmountProblem problem = 2;
  // What the amount's unit measures. `DIMENSION_OTHER` says the unit is not one the table converts, which is
  // how a caller tells `3.5 g` from `3.5 gr` without a second call. Unspecified when `quantity` is absent.
  Dimension dimension = 3;
}

message ParseAmountsResponse { repeated ParsedAmount amounts = 1; }
```

`AmountProblem` is the enum of the codes, named as `ReasonCode` is (`AMOUNT_PROBLEM_` plus the code in
capitals), held to `NotAnAmount::code` by a test in `quantities.rs` the way `grpc.rs` holds `ReasonCode`.
`dimension` was added in review: an unknown unit is an answer and not a refusal, so a caller needs to be told
when it got one. There is no limit on one text's length beyond the transport's on the message; reading is one
pass.
`quantities.rs` stays a translation. The span is `parse_amounts`, carrying how many texts and how many were
amounts, and the log line the same two numbers: never a text (D40).

### Offline: `record-similarity amounts survey FILE`

One text to a line. It reports how many lines are amounts, a count for each unit as normalized with its
dimension, and a count for each problem code. `--ndjson` writes the same as flat events through `similarity-log`
(`amount_survey`, `amount_unit`, `amount_problem`), whose names are an interface like the calibration ones.
`--examples N` (0 unless given) adds up to N of the refused texts for each code.

This is the tool that answers D16's first open question, "which unit strings actually occur?", where the data
is, without a person reading a column. Its output is counts and unit spellings. With `--examples` it also
prints raw values of a quantity column, which are not product text but are the company's data: the default is
off for that reason.

### What does not change

`Evaluate`, `Import`, `Match`, the export format, and `compare` take a `Quantity` and nothing else. A record does
not gain a text spelling of a quantity, and the refusal in `record.rs` stays. Parsing is a call a caller makes
before it has a record, so that what was understood is visible to the caller and not decided inside a write
(decision 1, confirmed by the owner).

So a client whose source holds `"3.5g"` as text calls `Parse` (up to 1000 texts to a call, so an import is not a
call for each row), puts the `Quantity` it gets into the record, and decides for itself what to do with a
refusal. A client that already holds a number and a unit never calls it. Neither normalizes a unit first:
`Evaluate` converts when it compares, and a unit is stored as it was sent.

## Design: units per something (34)

### The model: a unit is a numerator, or a numerator per a denominator

```rust
pub enum Dimension {
  Mass,
  Volume,
  Fraction,
  Other(String),
  /// One thing per another, when that is not a share: a mass per volume, a mass per serving.
  Per(Box<Dimension>, Box<Dimension>),
}
```

`quantity::unit(raw)` normalizes as it does now and then:

1. Looks the whole string up in `UNITS`. A spelling in the table always wins.
2. Otherwise splits it once, at a `/` or at the word `per` (a word: `percent` does not split). More than one
   separator, or a blank side, and it is `Other(whole string)` as today. Never an error.
3. Reads an optional amount at the front of the denominator: `mg/100 g`, `mg / 5 ml`. A plain positive decimal;
   anything else makes the whole string `Other`.
4. Looks each side up on its own: the table, then the count nouns below, then `Other`.
5. **Reduces a ratio of one known dimension to a share.** Mass over mass and volume over volume are `Fraction`,
   with the factor that makes them percent: `mg/g` is 0.1, `mg/kg` is 0.0001, `g/g` and `ml/ml` are 100,
   `g/100 g` is 1. Everything else is `Per(numerator, denominator)` with the factor
   `numerator / (amount x denominator)`.

Whitespace around the separator stops mattering, since each side is normalized on its own.

### What that makes comparable

| Before | After | Today | With this |
|---|---|---|---|
| `200 mg/g` | `20 %` | `INCOMPARABLE` | `UNCHANGED` |
| `200 mg/g` | `200 mg / g` | `INCOMPARABLE` | `UNCHANGED` |
| `10 mg/ml` | `10 g/l` | `INCOMPARABLE` | `UNCHANGED` |
| `25 mg/5 ml` | `5 mg/ml` | `INCOMPARABLE` | `UNCHANGED` |
| `10 mg/ml` | `100 mg/ml` | `INCOMPARABLE` | `MODIFIED`, factor 10 |
| `5 mg/serving` | `0.005 g/serving` | `INCOMPARABLE` | `UNCHANGED` |
| `5 mg/serving` | `5 mg/servings` | `INCOMPARABLE` | `UNCHANGED` (a count noun, below) |
| `5 mg/serving` | `5 mg/package` | `INCOMPARABLE` | `INCOMPARABLE`: a serving is not a package |
| `10 mg/ml` | `1 %` | `INCOMPARABLE` | `INCOMPARABLE`: a mass in a volume is not a share without a density |
| `10 mg` | `10 mg/serving` | `INCOMPARABLE` | `INCOMPARABLE`: an amount is not a rate (decision 8) |

`compare`, `in_unit`, the limits, `measure_changed`, the anchor's first values, the audit, and the restore all go
through `unit()` and need no change of their own. `Dimension::base_unit` of a `Per` is its two base units
joined: `mg/ml`, `mg/serving`.

### Count nouns

The denominator of `mg/serving` is not a unit of measure, and it is not noise either. `COUNT_NOUNS:
&[(&[&str], &str)]` lists the spellings of one noun (`serving`, `servings`) under one name, so that a plural or
an abbreviation is the same denominator. It does not say that two nouns are the same thing. Whether an `item` is
a `piece` or a `package` is exactly what D16 says this service cannot know ("the caller has to be consistent"),
and merging them here would make a tenth of a package compare as a whole one. A noun not in the table is
`Other(noun)` and works, without its plural.

### The judging digest says which unit rules a server has

`quantity::rules_digest()` hashes the rows of `UNITS` and `COUNT_NOUNS` in order, their factors as bits, and a
rule version that this feature sets to 2. `Description::judging` gains one last line, `units <hex>`. D44 allows
exactly this: a line added for a rule, none reworded or reordered. Every server's digest changes once, at the
upgrade, which is the truth: a server with these unit rules and one without them do not judge alike. From then
on a line added to either table changes the digest, which closes the gap found above.

### What it changes for records already stored

Nothing is rewritten: a stored quantity keeps the unit it was sent with. What changes is how two stored
spellings compare. A record stored with `200 mg/g` and re-sent as `20 %` was an incomparable change that set
`measure_changed`; it becomes no change at all, and by D16's rule not a write. Every such move is from wrong to
right, and none makes a comparable pair incomparable. It can still flip a verdict where `reject_incomparable` or
`reject_on_measure_change` is configured, so it goes out as a judging change: the digest moves, and a calibration
made before it is re-run after it (`calibrate-diff` shows what moved, which on pairs that use no such unit is
nothing).

### API (additive)

- `Dimension` gains `DIMENSION_RATE = 5`.
- `ConvertQuantityResponse` gains `UnitKind kind = 4` and `UnitKind to_kind = 5`, where `UnitKind` is
  `{ Dimension numerator; optional Dimension denominator; optional string denominator_unit; }`. The two existing
  dimension fields keep their meaning, and read `DIMENSION_RATE` for a rate.
- `ListUnitsResponse` gains `repeated CountNoun count_nouns = 2` and `string rules_digest = 3`.
- `quantities.rs` already does not compile until a new `Dimension` has a wire value, and a test holds the values
  apart.

## Where the two meet

The text parser hands everything after the number to `unit()`. With 34 built, `"25 mg / 5 ml"` parses to a
quantity that compares equal to `5 mg/ml`. With only 33 built it parses to an unknown unit that is comparable
with itself, which is still correct, and the survey counts it as one. So 33 can be built and run on real data
first, and what it reports (which rates occur, which nouns, which named amounts) is the evidence 34's tables
and decisions 2, 6, and 7 are waiting for.

## The work, in order

Each step ends with `just ci`. Commit a feature at a time.

1. **Tests first for the grammar**: the conformance file (below) and the table of refusals, failing.
2. `amount.rs` in `similarity-core`: `parse` and `NotAnAmount`. Export from `lib.rs`. Add the file to
   `.cargo/mutants.toml`.
3. Properties for `parse` in `similarity-core`'s `properties` module.
4. `Quantities.Parse` in the `.proto` and `quantities.rs`; `just client parse <text>...`.
5. `record-similarity amounts survey` in the `offline` module, with its events. D48. README. **Stop here and
   have the survey run at work** before step 6 if the tables are to be right the first time.
6. `quantity::rules_digest()` and the `units` line of the judging digest. Worth doing alone if 34 is deferred.
7. `Dimension::Per`, the split, the reduction, and `COUNT_NOUNS` in `quantity.rs`, tests first. Fix every
   `match` on `Dimension` the compiler names.
8. The wire: `DIMENSION_RATE`, `UnitKind`, `count_nouns`, `rules_digest`. The client's `convert` and `units`.
9. Curated cases with rates in `data/regression.jsonl`, then `just snapshot` (it needs the model) and what it
   prints in the commit message. No existing case moves: the sample data uses `g`, `mg`, `%`, `ml`, `fl oz`,
   and `oz` only.
10. D49, README, `CLAUDE.md`, the data request. `PROPTEST_CASES=5000 just test` once, `just test-real`, and
    `just mutants -j 3` from a clean checkout (the quantity comparison changed).

## Tests

- **A conformance file, `data/amounts.cases.jsonl`**, synthetic and checked in: one line for each text, with
  the quantity it must read as or the code it must be refused with. Every row of the two tables above, and the
  boundaries between them (`1,000` and `1,00` and `0,500`; `1-1/2` and `1 - 1/2` and `0-1/2`; `½` and `1½`;
  `18%-22%` and `18 mg-22 %`; `pre-rolls` and `g (approx)`). One test reads it through `amount::parse` and one
  through the service, so the two cannot
  differ, and a client in another language can run the same file against a server. This is the "lot of tests"
  the feature was asked for, and adding a case is adding a line.
- **Properties** (D19), for every input:
  - `parse` never panics and never returns a quantity that fails `check`, for any string.
  - A quantity written out as `"{min} {unit}"` or `"{min}-{max} {unit}"` parses back to itself.
  - Whitespace around the text, and the case of a known unit, change nothing.
  - A unit and its rate spellings: `x a/b` compares `UNCHANGED` with `x` times the factor in the base unit,
    and `in_unit` there and back is the same amount.
  - A ratio of one dimension is a share: for any two mass units, `1 a/b` in `%` is `100 x factor(a) /
    factor(b)`.
  - `unit()` of any string is total: a dimension and a finite positive factor, never a panic.
- **Examples in `quantity.rs`**, written to the boundaries mutation testing will try: the separator that splits
  and the one that does not (`percent`), the denominator's amount, a table spelling that contains a slash,
  `Per` of two unknown units, the digest moving when a row is added.
- **`quantities.rs`**: every `NotAnAmount` and every `Dimension` has a wire value of its own; `Parse` answers in
  order and answers a refusal as data; more than 1000 texts is `INVALID_ARGUMENT`.
- **The digest**: the pinned text of `judging` in `engine.rs`'s tests gains its last line; a test adds a unit
  row behind a test-only table and sees the digest move.
- **Fuzzing** (plan 14, deferred): `amount::parse` and `quantity::unit` are its first two targets when it is
  built. Nothing here waits for it.

## Documentation to update

- `docs/decisions.md`: D48 (amounts from text: the whole string or nothing, the codes, why a record still takes
  numbers) and D49 (units per something: the reduction to a share, count nouns, the digest line, what changed
  for stored records). D16's open-questions table: the `mg/g` row is answered. D47's "Not done": two items go.
- `README.md`: "Quantities on their own" gains `Parse` and the rate examples, each run; the command table gains
  `amounts survey`; "Mapping your records" keeps its rule and points at `Parse` for a source that holds text;
  "Replicas that judge alike" says the digest covers the unit rules; Limitations gains what is out of scope.
- `CLAUDE.md`: `amount.rs` in the layout; under "easy to get wrong": never guess an amount, the codes are an
  interface, the named and count-noun tables are the owner's vocabulary, and a unit row is a judging change
  that moves the digest.
- `docs/data-request-prompt.md`: ask for the survey's output in place of a hand count of unit strings, and for
  the trade's named amounts and count nouns.
- `.cargo/mutants.toml`, the Bruno collection, `tools/demo.py` (one step: a menu's `"1/8 oz"` parsed and
  compared with a catalog's `3.5 g`).

## What can only be validated at work

- Which unit strings, rates, count nouns, and named amounts occur, and how often each refusal does. That is the
  survey's output, and it decides the tables.
- How often each refusal occurs. The owner's rule is that wrong data is refused and not rescued, so a count is
  information for whoever cleans the source, and only a very large one would be a reason to ask again.
- Whether `%` on a tincture or a beverage means a share by weight. If it means a mass in a volume there, the
  reduction rule is still right and the data is what is ambiguous.

## Decisions for the owner, with the default to use meanwhile

Answered on 2026-11-22 where it says so. The rest stand at their defaults until the owner says otherwise.

1. **Does a record ever take a text amount?** **Decided: no.** `Evaluate` and the export keep `Quantity`; a
   caller whose source holds text calls `Parse` first. The alternative, accepting a string in `quantities`, hides
   what was understood inside a write.
2. **The named amounts.** Whether menus or catalogs ever hold a weight as a word (`eighth`, `quarter`, `half`,
   `zip`) with no number, and if so what each word equals. **Open: asked again in plainer words.** Default: none
   are read, and such a text is refused as `no_number`.
3. **Is an eighth 3.5 g?** `1/8 oz` is 3.54 g, a factor of 1.0125 from 3.5 g. So a record whose weight goes from
   `3.5 g` to `1/8 oz` has its weight reported as `MODIFIED` (the comparison's word for "both sides have a value
   and the values differ") with that factor, and not as `UNCHANGED`. It is rejected only if a
   `reject_above_factor` below 1.0125 is ever configured for that quantity, and none is configured at all today.
   **Open: asked again in plainer words.** Default: leave it so. The alternative is a table of nominal
   equivalents, which is a statement about products and not about units.
4. **The decimal comma.** **Decided: refused.** `3,5 g` is `ambiguous_separator`.
5. **Qualifiers.** **Decided: refused**, with the comma: wrong data is not rescued. `<1 %` is `qualified`.
6. **A mass over a mass is a share.** Default: yes, as D16 planned for `mg/g`, and for volume over volume. A
   mass over a volume never is. (Feature 34. Not put to the owner as a question yet.)
7. **The count nouns**, and whether any two are one thing. **Decided: the default, for now.** The plural and
   abbreviated spellings of `serving`, `piece`, `item`, and `package`, each its own noun.
8. **An implied denominator.** `thc_per_item` in `mg` and a later `mg/item` are an amount and a rate, and
   incomparable. **Decided: the default, for now.** A unit is what is written. The alternative is
   `per = "item"` on a `[quantities.<name>]` table, which only `Evaluate` could honor, since `Quantities` reads
   no schema.
9. **Numbers nothing has measured:** 1000 texts to a `Parse` call; `--examples` off by default.

## Size

**33 is medium**, three or four days: the parser and its conformance file are two, the call and the survey are
one or two. **34 is medium**, three days: the unit model and its tests are two, the digest line, the wire, and
the snapshot are one. Step 6 alone is half a day. The mutation run is 40 minutes; one run after 34 covers both
if they land together, and 33 alone owes one for `amount.rs`.

## Out of scope

- **Finding an amount inside prose.** A name that says `100mg` keeps saying it as text.
- **One quantity derived from two.** `mg` a package from `mg` a serving and servings a package needs a rule
  that reads several quantities at once, which nothing in `compare` does. A caller that has both multiplies.
- **Densities.** `mg/ml` to `%` needs one, and it differs by product.
- **A text spelling written back out.** `parse` reads; nothing here formats an amount for display.
- **Other locales and other languages' unit words.** The service is English only (README, Limitations).
- **A limit, a threshold, or a suggestion of one.** None of this measures anything.
