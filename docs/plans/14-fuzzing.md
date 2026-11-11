# Plan: fuzzing (feature 14)

A plan for feature 14 of `docs/feature-suggestions.md`, written so that a session with no memory of the
conversation that produced it can carry it out. Read `CLAUDE.md` first, then D2, D16, D19, D20, and D21 in
`docs/decisions.md`. Nothing here has been built. Delete this file, or fold what is still true into a decision
record, when the work is done.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Findings made while planning](#findings-made-while-planning)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

The property tests of D19 check the product's rules on inputs drawn from small, hand-picked vocabularies: seven
names, five units, six amounts. A coverage-guided fuzzer looks for the input nobody picked: an unusual character,
a float at the edge of its range, a file shaped in a way no test imagined. With `unsafe_code = "forbid"` it will
not find memory bugs in this workspace. What it can find, and what it should be framed and budgeted for:

- **Logic holes**: an input for which a rule of D7, D8, D14, D16, D20, or D21 does not hold.
- **Panics**, in this code or in a dependency (the tokenizer, `serde_json`, Burn's shape checks). Fuzz builds have
  debug assertions and overflow checks on.
- **Cost blow-ups**: an input that takes seconds or gigabytes, on the one embedding thread every request shares
  (D9).

Planning found two logic holes already, one verified (see [Findings](#findings-made-while-planning)). That is the
argument for the feature.

## What already exists

Verified in the tree on 2026-11-11.

- **The engine state machine**, `crates/similarity-server/src/properties.rs`: the `Request` enum (Evaluate,
  Delete, Import, Search), `step` (runs one request and checks D7, D8, D14, D16, the import rules, and that no
  other record changed), `check_anchor_kept`, and the proptest
  `every_request_in_any_sequence_obeys_the_write_rules`. It runs on `MemoryStore` and `WordBagEmbedder`.
- **The scoring properties**, `crates/similarity-core/src/properties.rs`, including the quantity factor properties
  (`a_factor_is_the_same_in_both_directions_and_at_least_one`) with amounts limited to `1e-9..1e12`.
- **Parsers that already take bytes or text and are public**, so a target can call them directly:
  - `similarity_core::model::Pipeline::parse(&[u8], &Path)` and `Pipeline::files()` (`model/sentence.rs`);
  - `Quantity` deserialization (`serde(try_from = "QuantityInput")`), `Quantity::check`, `quantity::unit`,
    `quantity::compare`, `quantity::first_values` (`quantity.rs`);
  - `record::normalize_text`, `record::text_hash`, `PreparedRecord::new` and `with_quantities` (`record.rs`);
  - `TextTokenizer::encode` (`tokenize.rs`), whose private `head` is the D21 cut;
  - `RecordFields` (its own `Deserialize`, `offline/mod.rs`);
  - `parse_pairs(&str)` and `parse_full_pairs(&str, &Schema, Selection)` (`offline/pairs.rs`);
  - `offline::events::read(&str) -> Run`, `offline::diff::{rejudge, diff, failure, render_diff}`;
  - `import::ImportLine` and `import::check(&Schema, impl BufRead, report)` (`import.rs`);
  - `Config::parse(&str)`, `Config::schema`, `Config::policy` (`config.rs`);
  - `similarity_train::data::{pair_examples, group_examples}` (take `&str`).
- **The wire surface is callable without a network**: `grpc::GrpcService::new(Arc<Engine>)` implements the
  generated `RecordSimilarity` trait, whose methods take a `tonic::Request<pb::...>`.
- **Test support**: `similarity_core::testing` (behind the `testing` feature, with `#![allow(clippy::unwrap_used)]`
  and a stated reason) provides `tiny_tokenizer` (a `WordLevel` tokenizer with a whitespace pre-tokenizer),
  `tiny_embedder`, and `write_tiny_model`.
- **Toolchain**: `rust-toolchain.toml` pins `1.96.0`. On the owner's machine `cargo-fuzz 0.13.2` and a nightly
  (`1.99.0-nightly`, 2026-07-19) are installed.
- **Already in `Cargo.lock`, though not in the built graph**: `arbitrary 1.5.0` and `libfuzzer-sys 0.4.13`, both
  pulled in optionally by `rav1e`.

## What the suggestion got wrong or left open

1. **"It needs nightly Rust."** Not with `--sanitizer none`. A probe crate built and ran under
   `cargo +1.96.0 fuzz run -s none` (cargo-fuzz 0.13.2, aarch64-apple-darwin): with no sanitizer, every flag
   cargo-fuzz passes (`-Cpasses=sancov-module`, `-Cllvm-args=-sanitizer-coverage-*`, `--cfg fuzzing`,
   `-Cdebug-assertions`) is a stable one. Only AddressSanitizer and the other sanitizers need nightly, and with
   unsafe code forbidden here they would only watch the dependencies' unsafe code. The fuzzer can run on the
   pinned toolchain. Linux is expected to behave the same; the first CI run confirms it.
2. **"The state machine is ready to be driven by a fuzzer"** (`docs/plan-features.md`). Its logic is; its form is
   not. `properties` is `#[cfg(test)] mod properties`, private to the crate. `WordBagEmbedder` is
   `#[cfg(test)] pub(crate)` in `worker::testing` and `assert!`s that a text has a word. `step` returns proptest's
   `TestCaseError`. And `Request` holds indexes into fixed lists (`NAMES`, `CATEGORIES`, ...), so a fuzzer could only
   choose among texts already chosen. The checker has to move into a `testing` module behind a feature, take
   owned strings, and return its own error type (step 3).
3. **Target 4, "NaN or infinite thresholds, zero or negative weights"**: already refused. `Schema::new` refuses a
   weight that is not positive and finite, `Policy::new` and `Policy::with_threshold` refuse a threshold outside
   `[-1, 1]` (NaN included), `with_quantities` refuses a factor limit below 1 or not finite, and `schema.rs` tests
   it. What is not checked is a combination of valid values (see finding 3).
4. **Target 1 asks for invariants that are enforced, but through parsers a target cannot reach.** `stage_dir` and
   the `Normalize`-last rule are in `Pipeline::parse`, which is public. But `BertConfig::from_file`, the pooling
   config, and the `Dense` config are read from paths, and the last two are private. Fuzzing
   `SentenceEncoder::load` end to end would mostly find out-of-memory aborts: it calls `config.init(device)` for
   whatever sizes `config.json` states before it looks at the weights, and `validate` has no upper bound. Bytes
   level parse functions are needed (step 2), and the allocation is a decision, not a fuzz target.
5. **"Some internals may need a public entry point behind a `fuzzing` feature."** Most do not: the parsers above
   are public already. The few that are not (the three model config files, `training_state.json`,
   `provenance.json`) are file readers whose parse step should simply be a public function the reader calls. No
   `fuzzing` feature is proposed. What does need a feature is test support (the word-bag embedder and the
   checker), and that is a `testing` feature as in `similarity-core`.
6. **Target 5 predates three parsers.** Since the suggestion, feature 7 added the full calibration format, feature 8
   the events reader (`events::read`, which `calibrate-diff` runs on files written elsewhere), and feature 16 the
   import line. Those are now more exposed than the trainer's files, which only the owner's own tools write.
7. **The tokenizer target needs a tokenizer shaped like the real one.** `tiny_tokenizer` has no BERT normalizer,
   no BERT pre-tokenizer, and no WordPiece, which is where the interesting behavior is (finding 1 depends on the
   normalizer).
8. **Left open: how a crash becomes a regression test, where targets live so `just ci` is unaffected, and what
   `cargo deny` says.** All three are answered below. On the last: `libfuzzer-sys` is licensed
   `(MIT OR Apache-2.0) AND NCSA`, and NCSA is not on `deny.toml`'s allow list.

The suggestion is right that safetensors header parsing belongs to `burn-store`. The same holds for parsing a
`tokenizer.json`, which belongs to `tokenizers`.

## Findings made while planning

These are what the first targets should confirm and then turn into named tests. Only the first was run.

1. **Verified: the D21 cut lets invisible characters hide a field's text.** BERT's normalizer (`clean_text`) drops
   control and format characters (Unicode Cc, Cf, and also Co and Cn), and `normalize_text` keeps them because
   they are not whitespace. `head` counts their bytes against the 16 KB budget. So a field that starts with 16 KB
   of zero-width spaces (U+200B) or word joiners (U+2060) reaches the tokenizer as nothing but invisible
   characters, and encodes as `[CLS] [SEP]` whatever follows. Run in a scratch crate against the real
   `models/mdbr-leaf-mt/tokenizer.json`: `"\u{200B}" * 6000 + "blue dream live resin cartridge"` and the same
   prefix before `"sour diesel gummies ten pack"` both encoded to `[101, 102]`; without the cut the first encodes
   to its five words. Two different products then score 1.0 on that field, and since the text hash still changes,
   the field reads as modified at similarity 1. A page renders the field as its visible words. Before D21 the
   normalizer removed the prefix and the text was read. Soft hyphens (U+00AD, two bytes each) at 12 KB did not
   reach the limit and did not hide anything, so the effect depends on byte length. D21's claim "changes nothing a
   caller can see" does not hold for this input. This is a bug to fix whether or not fuzzing is built; the fix is
   owner decision 1.
2. **Predicted, not run: a quantity factor can be infinite.** `Quantity::check` accepts any finite, non-negative
   value whose base-unit value is finite. `ratio` returns `high / low`, which for `1e-300 g` against `1e300 g` is
   `inf`. The policy still rejects it (`inf` is above every limit), but `inf` is not a JSON number: an event
   carrying it would be written as `null` and `calibrate-diff` would then fail to read the run back. D16's
   property only draws amounts from `1e-9..1e12`.
3. **Predicted, not run: an extreme weighting hides a field.** In a composite with every field present, a field's
   block has length `sqrt(w / total)`. `Schema::blocks` treats a block shorter than `ABSENT_BLOCK_NORM` (`1e-6`) as
   absent. So a field whose weight is below about `1e-12` of the total is read back as missing, and any change to
   it is out of the score. Each weight on its own is valid. No real config is near this, but `Config::parse`
   accepts one.

## Design

### Where things live

Three places, so that nothing nightly-only, unlinted, or untested holds logic:

1. **`similarity_server::testing`**, a new module behind a new `testing` feature (as in `similarity-core`), holding
   the word-bag embedder and the state machine's checker. `properties.rs` keeps the proptest strategies and calls
   the checker; the fuzz harness calls the same checker. One copy of the rules.
2. **`crates/similarity-fuzz`**, a new workspace member (`publish = false`, library only). One module per target,
   each with `pub fn run(data: &[u8]) -> Result<(), Violation>`. It is under the workspace lints, `just clippy`,
   `just test`, and `just deny` like every other crate. Its tests replay every committed seed and every saved
   crash, so a fixed crash stays fixed under plain `cargo test`, with no libFuzzer.
3. **`fuzz/`**, the `cargo-fuzz` crate, outside the workspace: a `Cargo.toml` with its own `[workspace]` table, its
   own committed `Cargo.lock`, and one three-line file per target. The root `Cargo.toml` adds
   `exclude = ["fuzz"]`.

```rust
// fuzz/fuzz_targets/quantity.rs
#![no_main]
libfuzzer_sys::fuzz_target!(|data: &[u8]| {
  if let Err(violation) = similarity_fuzz::quantity::run(data) {
    panic!("{violation}");
  }
});
```

The panic is the only way to tell libFuzzer a rule broke, and it is in a crate the workspace lints do not cover.
Nothing in `similarity-fuzz` panics on purpose.

`similarity-fuzz` depends on `similarity-server` with `testing` as a normal dependency, so `cargo build
--workspace` compiles the server library with the test support module in it. `cargo build -p similarity-server`
(`just release`, the Dockerfile) does not. Clippy then lints the `testing` modules under the full rules, since they
are no longer only `cfg(test)`; write them to pass, with no `allow` beyond what `similarity_core::testing` has.

### The harness crate

```text
crates/similarity-fuzz/
  Cargo.toml
  src/
    lib.rs          //! header, `Violation`, `TARGETS`
    quantity.rs     record_json.rs   tokenize.rs   model_dir.rs   config.rs
    import_line.rs  pairs.rs         events.rs     engine.rs      wire.rs
  tests/replay.rs
  examples/write_wire_seeds.rs
```

```rust
/// A rule that did not hold for an input. The fuzz target turns it into a crash.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Violation(pub String);

/// Every target, by the name of its file under `fuzz/fuzz_targets/` and of its seed and regression directories.
pub const TARGETS: &[(&str, fn(&[u8]) -> Result<(), Violation>)] = &[
  ("quantity", quantity::run),
  // ...
];
```

Dependencies: `similarity-core` (`testing`), `similarity-server` (`testing`), `similarity-store`,
`similarity-proto`, `arbitrary` (with `derive`), `prost`, `tonic`, `tokio` (`rt`), `serde_json`, `toml`,
`thiserror`. Only `arbitrary` is new to the graph: MIT OR Apache-2.0, no dependencies of its own beyond
`derive_arbitrary` (same licenses, built on `proc-macro2`, `quote`, and `syn`, which are already in the tree).
`cargo deny` passes it with today's `deny.toml`. Why it and not proptest's strategies: libFuzzer hands a target
bytes, and `arbitrary::Unstructured` is how those bytes become a structured request sequence while still letting
the fuzzer's mutations steer it.

### Targets, by how exposed the input is

Each states what it feeds in and which rule it checks beyond "no panic, no timeout, no out-of-memory".

| Target | Input | Rule checked |
|---|---|---|
| `wire` | bytes split into (method, protobuf message), decoded with `prost`, sent to `GrpcService` over `MemoryStore` and the word-bag embedder | An error status leaves the store exactly as it was (D7); every float in a response is finite; similarities lie in `[-1, 1]`. This is the unauthenticated surface. |
| `engine` | `arbitrary` session: policy choice, then up to 64 requests (Evaluate in every mode, Delete, Import, FindSimilar) | Every rule `step` checks today (D7, D8, D14, D16, D20), after every request. Ids come from a pool of three, so sequences revisit records; texts mix a word list with arbitrary strings; quantity amounts are arbitrary `f64`, units mix `UNITS` spellings with arbitrary strings. Also: every reported factor is `None` or finite and at least 1 (finding 2). |
| `tokenize` | arbitrary text, run through `normalize_text` and then a BERT-shaped in-memory tokenizer with `max_tokens = 16` (so the cut is at 1 KB and the fuzzer gets past it cheaply) | At most `max_tokens` ids; and the D21 claim itself: the ids with the cut equal the ids without it, unless the text holds an unbroken run longer than 64 bytes (D21's stated exception). Fails today on finding 1. |
| `record_json` | bytes as a `RecordFields` JSON object, prepared against a fixed schema shaped like `config.example.toml` | Serialize then parse gives the same record; a record `PreparedRecord` accepts has only quantities that pass `check`, and each compares with itself as unchanged with factor 1. |
| `quantity` | two quantities from arbitrary `f64`s and unit strings | `unit` of a normalized unit is the unit of the raw one; for two quantities that pass `check`, `compare` is symmetric in change, factor, and comparability, and the factor is finite and at least 1. Expected to fail on finding 2. |
| `import_line` | bytes as an export (lines) | Differential, for D20's promise that `--check` finds what an import would refuse: a line `import::check` passes is never refused as `InvalidArgument` by `Engine::import` on an empty store, and a line it fails is never created. |
| `pairs` | bytes as a strict file and as a full file (fixed schema, default `Selection`) | Every parsed pair has a non-empty `pair_id`, and ids are unique within a file, or the parse is an error. |
| `events` | bytes as `calibrate --ndjson` output | If `events::read` accepts it, `rejudge` under the fixed schema and policy, `diff` of the result with itself, and `render_diff` run; the diff has no flips (`failure(&diff, FailOn::AnyFlip)` is `None`). |
| `config` | bytes as TOML | If `Config::parse` accepts it: a record with every field present composes and reads back from `Schema::blocks` with every field present (finding 3); `layout_id` does not depend on weights. |
| `model_dir` | bytes as `modules.json`; in step 2 also `config.json`, the pooling config, and the `Dense` config through new bytes-level parsers | If `Pipeline::parse` accepts it, every path in `files()` is relative, `/`-separated, without `..`, a root, or `\`; and, reading the same JSON as plain values, no stage follows a `Normalize` (D2). Config parsers: parse equals what the file reader returns for the same bytes. |

Phase two, lower exposure (only the owner's tools write these): `train_data` (`pair_examples`, `group_examples`),
`checkpoint_state` (through a new `checkpoint::parse`), `provenance` (through a new `Provenance::parse`). Putting
`similarity-train` in the harness crate turns on Burn's `autodiff` in the harness build, as `cargo test
--workspace` already does; D13 forbids only the server depending on the trainer, and it still does not.

The tokenizer for `tokenize` is a new `similarity_core::testing::bert_tokenizer(max_tokens) -> TextTokenizer`:
`BertNormalizer` (clean text, Chinese characters, lowercase, accents left alone), `BertPreTokenizer`, `WordPiece`
with a vocabulary of a few hundred pieces including `##` continuations and `max_input_chars_per_word = 100`, and
BERT's `[CLS]`/`[SEP]` processor: the settings of `models/mdbr-leaf-mt/tokenizer.json`. It can call the
`pub(crate) TextTokenizer::new`, and also needs an uncut encode for the oracle, as a `testing`-only
`TextTokenizer::encode_uncut`. Set `FUZZ_TOKENIZER_DIR=models/mdbr-leaf-mt` to fuzz with the real tokenizer instead
(after `just fetch-model`); that run is local only.

### Seeds, dictionaries, and the corpus

- `fuzz/seeds/<target>/`: small, synthetic, committed: `data/examples/*.json`, lines of
  `data/calibration.sample*.jsonl`, `config.example.toml`, the `modules.json` and configs `write_tiny_model`
  writes, the import line from the README, a `calibrate --ndjson` run of the sample. Wire seeds are protobuf, so
  they are written by `cargo run -p similarity-fuzz --example write_wire_seeds` (`just fuzz-seeds`) and committed.
- `fuzz/dicts/<target>.dict`: libFuzzer dictionaries: JSON punctuation, every spelling in `UNITS`, the schema's
  field and quantity names, `Normalize`, `Pooling`, `Dense`, event names.
- `fuzz/corpus/` and `fuzz/artifacts/`: what a run grows and what it finds. Git-ignored.

**Real product data never enters this repository.** Seeds are synthetic. A run seeded from a real export can only
happen in the owner's work environment, and the corpus it grows holds real product text: it is never committed and
never uploaded as a CI artifact. A crash found that way is reproduced as a synthetic input before it comes here.

### From a crash to a named test

1. A run stops with `fuzz/artifacts/<target>/crash-<hash>` (or `slow-unit-`, `oom-`).
2. `just fuzz-tmin <target> <file>` minimizes it; `just fuzz <target> <file>` replays it.
3. Write the ordinary test that states the broken rule, beside the code that broke it (`quantity.rs`,
   `tokenize.rs`, ...), named as a sentence, with the input written out literally when it is readable.
4. Copy the minimized file to `fuzz/regressions/<target>/<that test's name>`. `tests/replay.rs` runs every file
   there through its target, so the harness keeps accepting it.
5. Fix the code, or the rule if the rule was wrong. Commit the three together, as D19 says for proptest.

### Commands (justfile, four-space indented like the rest of it)

```text
just fuzz <target> [seconds=60]    # cargo fuzz run -s none <target> fuzz/corpus/<target> fuzz/seeds/<target>
                                   #   -- -max_total_time=<seconds> -dict=fuzz/dicts/<target>.dict
                                   #      -timeout=10 -rss_limit_mb=2048
just fuzz-all [seconds=300]        # every target in turn
just fuzz-tmin <target> <file>     # cargo fuzz tmin -s none
just fuzz-seeds                    # regenerate the wire seeds
just fuzz-check                    # fmt-check and build of fuzz/, plus cargo deny on its lockfile
```

All run on the pinned 1.96.0 toolchain. `cargo +nightly fuzz run -s address <target>` remains available for an
occasional sanitizer run. `just ci` does not change: `similarity-fuzz` is a workspace member and is covered by it;
`fuzz/` is not built on every pull request because it compiles Burn with coverage instrumentation, which is slow,
and its shims cannot break without the harness, which `just ci` tests.

### CI

`.github/workflows/fuzz.yml`, on `schedule` (weekly) and `workflow_dispatch` (with a per-target seconds input):
toolchain 1.96.0, `cargo install cargo-fuzz --locked --version 0.13.2`, `just fuzz-check`, then
`just fuzz-all 600`. The corpus is kept between runs with `actions/cache`, keyed by target. On failure the job
uploads `fuzz/artifacts/` (synthetic inputs only; see above). `RUSTFLAGS: -D warnings` from the main workflow is
not set here: cargo-fuzz appends its own flags to `RUSTFLAGS`, and a warning in a dependency built with
instrumentation is not this job's concern.

### cargo deny

The harness crate adds `arbitrary` and `derive_arbitrary` to the root graph; `just deny` passes them. `fuzz/` has its
own lockfile, which the root `cargo deny check` does not see. `just fuzz-check` runs
`cargo deny --manifest-path fuzz/Cargo.toml check --config fuzz/deny.toml`, where `fuzz/deny.toml` repeats the root
policy and adds one exception, `{ crate = "libfuzzer-sys", allow = ["NCSA"] }`: LLVM's libFuzzer is under the
University of Illinois/NCSA license, permissive, and only ever in a fuzzing binary that ships nowhere.

### Events

None. The harness does not log. libFuzzer prints its own progress on stderr, which is a tool's output and not the
service's (D15 governs the two binaries).

## The work, in order

Each step is one commit and leaves `just ci` passing.

0. **Fix finding 1.** Test first with the BERT-shaped tokenizer (it lands in this commit),
   `invisible_characters_before_the_cut_do_not_hide_the_text_after_them`, which fails today. Fix per owner
   decision 1; amend D21. Run `just test-real` (tokenizer change) and `just calibrate`, which must print the
   numbers it printed before.
1. **Bytes-level parsers.** `BertConfig::parse(&[u8], &Path)`, `sentence::parse_pooling(&[u8], &Path)`,
   `sentence::parse_dense(&[u8], &Path)` (or one public `ModelConfigs` parse, whichever reads better), and later
   `checkpoint::parse` and `Provenance::parse`. Each file reader becomes read plus parse. No behavior change.
2. **`similarity_server::testing`.** Move `WordBagEmbedder` and `word_bag` there behind `testing`; `word_bag`
   returns an error for a text with no words instead of asserting. Move `step` and `check_anchor_kept` into
   `testing::machine`, with `Request` holding owned strings and `Quantity` values, the invalidity oracle written
   against the public validation (`PreparedRecord::new` then `with_quantities`, present-but-blank id or tenant,
   threshold outside `[-1, 1]`), and errors as `machine::Violation`. `properties.rs` maps its indexes onto the new
   `Request` and turns a `Violation` into `TestCaseError::fail`. The property must still catch the three
   deliberate breaks D19 lists; check that again by hand.
3. **`crates/similarity-fuzz`** with `quantity`, `record_json`, `tokenize`, `model_dir`, `config`, their seeds and
   dictionaries, and `tests/replay.rs`. Add it to the workspace members and the README layout.
4. **Server targets**: `engine`, `wire`, `import_line`, `pairs`, `events`, and `write_wire_seeds`.
5. **`fuzz/`**: the shim crate, its lockfile, `fuzz/deny.toml`, `.gitignore` entries, `exclude = ["fuzz"]`, the
   justfile recipes.
6. **First campaign**, locally: `just fuzz-all 600`. Findings 2 and 3 are expected here. Each finding is its own
   commit (test, saved input, fix), and any fix that touches scoring is followed by
   `PROPTEST_CASES=5000 just test`. Record what was found, and that each oracle can fail (break the engine as D19
   did and time how long the `engine` target takes to notice), for D22.
7. **CI workflow** `fuzz.yml`.
8. **Documentation**: D22, README, CLAUDE.md.
9. Later, phase two: `train_data`, `checkpoint_state`, `provenance`.

## Tests

None need the model or Qdrant. Names as they would appear:

In `similarity-core`:

- `invisible_characters_before_the_cut_do_not_hide_the_text_after_them` (fails today)
- `a_bert_shaped_tokenizer_gives_ordinary_text_the_same_tokens_with_and_without_the_cut`
- `a_config_json_parses_from_bytes_as_it_reads_from_a_file`
- `a_pooling_config_parses_from_bytes_in_both_formats` (D2)
- `a_quantity_at_the_edge_of_the_float_range_has_a_finite_factor_or_is_refused` (expected to fail until decision 3)
- `a_schema_whose_weights_would_hide_a_field_is_refused` (expected to fail until decision 4)

In `similarity-server`:

- `the_word_bag_embedder_refuses_a_text_with_no_words_instead_of_panicking`
- `every_request_in_any_sequence_obeys_the_write_rules` (kept, now through `testing::machine`)

In `similarity-fuzz`:

- `every_seed_runs_clean_through_its_target`
- `every_saved_fuzz_input_runs_clean`
- `every_fuzz_target_file_has_a_harness_and_every_harness_a_target_file` (reads `fuzz/fuzz_targets/`)
- `a_line_that_import_check_accepts_is_never_refused_as_malformed_by_the_import`
- `a_calibration_diffed_against_itself_has_no_flips`
- `the_wire_target_reads_every_written_seed_as_a_request`

In `similarity-train` (phase two): `a_checkpoint_state_parses_from_bytes_as_it_reads_from_a_directory`.

The only checks that need the real model are the ones step 0 requires anyway: `just test-real` and
`just calibrate`. Fuzzing with the real tokenizer (`FUZZ_TOKENIZER_DIR`) is a local run, not a test.

## Documentation to update

- **`docs/decisions.md`**: a new **D22, fuzzing finds the input nobody picked**: what is fuzzed and why (logic
  holes, panics, cost, not memory bugs), the three-place layout, `-s none` on the pinned toolchain, the crash to
  test path, the real-data rule for corpora, what the first campaign found, and proof the oracles can fail.
  **Amend D21** for finding 1 and its fix, and D16 for finding 2's resolution.
- **`README.md`**, Development: a "Fuzzing" paragraph after "Property tests" with `just fuzz` and what a crash
  becomes; `just fuzz*` in the command block; `similarity-fuzz/` and `fuzz/` in the project layout; the decisions
  count to D22. Every command quoted must have been run.
- **`CLAUDE.md`**: the workspace layout gains `similarity-fuzz` ("one harness function per fuzz target; no logic
  in `fuzz/`"). Under Testing: "A fuzz crash becomes a named test beside the code it broke and a file under
  `fuzz/regressions/<target>/`, committed together. Fuzz corpora are never committed, and never seeded from real
  data here." Under Commands: `just fuzz <target>`, `just fuzz-all`, `just fuzz-tmin`. Under "easy to get wrong": a
  new parser of outside input gets a target in the same change.
- **`docs/feature-suggestions.md`**: mark 14 built, as 13 is. **`docs/plan-features.md`**: its one-line note on
  14.

## Dependencies on other features

None blocks this. Interfaces assumed:

- **Feature 19** (reason codes on the wire) adds fields to `pb::EvaluateResponse`. The `wire` target's oracle
  should then also check that every code is one `Reason::code` produces. Nothing else changes.
- **Feature 10** (score telemetry) logs factors and similarities. Finding 2 decides whether a factor can be
  infinite in a log line; that decision should land first.
- **Feature 11** (drift calibration) replays histories through the engine; it can reuse `testing::machine`.
- **Feature 17** (reviewer feedback) and any later parser of outside input add a target each.
- **Feature 15** (mutation testing) should exclude `similarity-fuzz`, whose mutants the fast suite cannot kill by
  design.

## Decisions for the owner

1. **How to close the hole the cut opened (finding 1).** Options: (a) run the model's own normalizer over the
   whole text and cut what it returns, so that only characters the model would keep count against the budget;
   (b) cut as now, and if the head yields fewer than `max_tokens` tokens, double it and encode again, up to the
   whole text; (c) strip format and control characters in `normalize_text`, which changes stored text hashes.
   **Recommended: (a), if it measures within D21's numbers** (`compare` on 1 MB and 3 MB fields, as D21 measured);
   it is exact and keeps D21's cost for ordinary text. (b) is exact too but brings back the full cost for a base64
   blob, which D21 cut on purpose. (c) leaks a tokenizer detail into the hash.
2. **Where harness code lives.** **Recommended: the `similarity-fuzz` workspace crate**, so all logic is linted and
   tested and `fuzz/` holds only shims. The alternative, a `fuzzing` feature in each crate, scatters the targets.
3. **An infinite factor (finding 2, once confirmed).** **Recommended: `ratio` saturates at `f64::MAX`**: still
   beyond every limit, symmetric, and a number JSON can hold. The alternative, refusing such quantities in
   `check`, needs a bound on the ratio of two values that nobody has measured.
4. **A weighting that hides a field (finding 3, once confirmed).** **Recommended: `Schema::new` refuses it**,
   derived from `ABSENT_BLOCK_NORM` rather than from a new constant, with a message naming the field.
5. **Sanitizers.** **Recommended: `-s none` on the pinned 1.96.0**, verified to work; nightly ASan only by hand.
6. **CI budget.** **Recommended: weekly, ten minutes a target**, about two hours with the instrumented build, plus a
   manual trigger for longer runs. Not on pull requests.
7. **The NCSA license of `libfuzzer-sys`.** **Recommended: allow it in `fuzz/deny.toml` only**; it never reaches a
   shipped binary.
8. **The loader allocating whatever `config.json` says before it reads the weights.** **Recommended: leave it for
   now and record it in D22.** The default model is pinned by checksum, and another is chosen by an operator. If it
   is fixed later, compare the config's sizes with the safetensors header before `init`, rather than inventing an
   upper bound.

## Size

**Medium.** The shims and CI are small. The work is the move of the state machine into a shared `testing` module
without weakening it, ten harnesses with real oracles, and the seeds. Each finding then needs its own decision and
fix, and finding 1 touches the tokenizer, so it needs `just test-real` and a re-measurement. With phase two and a
long triage of the first campaign it would become large.
