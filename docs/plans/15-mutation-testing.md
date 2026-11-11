# Plan: mutation testing (feature 15)

A plan for checking, systematically, that the tests would notice if the code that reaches a verdict were wrong.
Written on 2026-11-11 against `initial-implementation` at `545ef20`, before any of it was built. Read `CLAUDE.md`
first, then `docs/decisions.md` D7, D8, D16, D17, and D19, which this builds on.

Nothing here was run: this plan was written while other planners were reading the tree, so no tool was installed,
nothing was built, and every count and duration below is an estimate that step 1 replaces with a measurement.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)

## What it is for

The policy code is short and made of comparisons: a similarity against a threshold, a field against its floor, a
quantity's factor against its limit, a hash against a hash to decide whether anything changed. Each comparison is a
verdict. If `<` became `<=` at the threshold, a revision scoring exactly the threshold would flip from accepted to
rejected, and nothing in the suite might notice.

A mutation tester makes small changes like that one at a time (a mutant), runs the tests against each, and reports
the mutants no test failed on. A surviving mutant in the verdict code is an untested boundary. D19 already did this
by hand for the engine state machine (three deliberate breaks, all caught); this feature does it for every
operator and every function in the files whose boundaries are verdicts, and keeps doing it as the code changes.

It needs no model, no Qdrant, no network, and no real data: it runs the fast suite (`just test`) many times.

## What already exists

Verified in the tree as it is now:

- **The code under test.**
  - `crates/similarity-core/src/score.rs`: `compare`, `Comparison::weigh` (`found < floor` sets `below_floor`),
    `extend_anchor`, `weighted_similarity`, `weighted_mean` (`total <= 0.0`), `Policy::new` and
    `Policy::with_threshold`, `judge` (`comparison.similarity < policy.threshold`, `similarity < threshold` for
    drift), `judge_quantities` (`factor > limit`, `zero_crossing`, `reject_incomparable`, the `measure_changed`
    rule), `Evaluation::judge_quantities` and `Evaluation::rejudge`, `Reason::code`, `subject`, `value`, `limit`.
  - `crates/similarity-core/src/quantity.rs`: `RELATIVE_TOLERANCE` (1e-6), `Quantity::check` (`value < 0.0`,
    `self.min > self.max`), `unit`, `ratio` (`left <= right`, `high - low <= RELATIVE_TOLERANCE * high`,
    `low <= 0.0`), `measure` (`factor <= 1.0 && change == QuantityChange::Modified`), `compare`, `first_values`.
  - `crates/similarity-core/src/schema.rs`: `Schema::new` and `with_quantities` (validation: weight `> 0.0`, floor
    in `[-1, 1]`, factor limit `>= 1.0`), `compose`, `blocks` (`norm(block) > ABSENT_BLOCK_NORM`), `layout_id`.
  - `crates/similarity-core/src/vector.rs` (`dot`, `norm`, `normalize`, `cosine`) and `record.rs`
    (`normalize_text`, `text_hash`, `PreparedRecord::new` with the `required` check, `plan` with the reuse-by-hash
    rule, `EmbedPlan::finish`).
  - `crates/similarity-server/src/engine.rs`: `Engine::evaluate`, `revise` (the `identical` conjunction, the
    `write` match on `WriteMode`, the anchor branch on `WriteMode::Force`, `revision.saturating_add(1)`), `import`,
    `store_import`, `find_similar` (`limit <= MAX_NEIGHBORS`), `delete`, `same_tenant`, `required_text`, `now`.
  - `crates/similarity-server/src/offline/stats.rs` (ranges, quantity suggestions, slices, the owner bootstrap,
    error rates, operating points) and `offline/calibrate.rs` (`suggest_threshold`).
- **The tests that should kill mutants there.** Unit tests in each file's `#[cfg(test)] mod tests`; the
  `properties` modules of `similarity-core` (`src/properties.rs`, eleven properties) and `similarity-server`
  (`src/properties.rs`: the engine state machine `every_request_in_any_sequence_obeys_the_write_rules`,
  `the_suggested_threshold_is_what_trying_every_cut_finds`); the engine scenarios in `engine.rs`; the stats tests.
  Every one of these is in a crate's library test target. The integration test targets in `tests/` that run
  without `--ignored` are only `similarity-store`'s contract against `MemoryStore`, which is outside this feature.
- **Some boundaries are already pinned.** `a_factor_limit_rejects_what_exceeds_it_and_nothing_else` checks that a
  factor exactly at its limit is within it (3.5 g to 14 g at a limit of 4), so `factor > limit` becoming
  `factor >= limit` is caught. `one_owner_or_a_pair_without_one_means_no_interval` pins `groups.len() > 1`.
- **Proptest is configured through the environment.** Both `properties` modules build their `Config` as
  `Config { cases, ..Config::default() }`, and proptest 1.12 (the version in `Cargo.lock`) fills
  `Config::default()` from `PROPTEST_RNG_SEED` and `PROPTEST_DISABLE_FAILURE_PERSISTENCE` (verified in
  `proptest-1.12.0/src/test_runner/config.rs`). That is what makes a mutation run deterministic without touching
  the test code.
- **The build script never fails without git.** `build_commit.rs` takes `RECORD_SIMILARITY_COMMIT` if set, else
  asks `git`, else writes `unknown`. A mutation tester builds in a copy of the tree with no `.git`, so this matters.
- **CI.** `.github/workflows/ci.yml` runs fmt, clippy, tests on Linux and macOS, the Qdrant contract, and
  `cargo deny`. But the repository has **no remote and one branch** (`git remote -v` is empty; `git branch -a`
  shows only `initial-implementation`), so that workflow has never run.
- **Nothing for mutation testing yet.** `cargo-mutants` is not installed on the owner's machine (`cargo mutants`
  is "no such command"), there is no `.cargo/` directory, and nothing in the tree mentions mutants.

## What the suggestion got wrong or left open

1. **`offline.rs` no longer exists.** It is the `offline/` module now: `calibrate.rs`, `stats.rs`, `diff.rs`,
   `pairs.rs`, `report.rs`, `events.rs`. The verdict-bearing parts are `stats.rs` and `calibrate.rs`
   (`suggest_threshold`). `quantity.rs` did not exist when the suggestion was written; `plan-features.md` already
   says to add it and `stats.rs`.
2. **"Saturating arithmetic" is invisible to the tool.** `cargo-mutants` mutates binary and unary operators,
   whole function bodies, match arms and guards, and struct-literal fields. It does not mutate method calls. The
   workspace lints (`arithmetic_side_effects`) push every integer operation into `saturating_*`/`checked_*` calls,
   so integer arithmetic here is almost entirely out of its reach, `revision.saturating_add(1)` included. What is
   left for it is float arithmetic, comparisons, and boolean logic, which is where the verdicts are anyway.
   The same blind spot covers `low.max(high)` in `quantity::measure`, `clamp` in `weighted_mean`, and every
   `(-1.0..=1.0).contains(..)` range check (`Policy::new`, `Schema::new`, `find_similar`).
3. **"The tolerance" cannot be mutated either.** Literals and constants are never changed: `RELATIVE_TOLERANCE`,
   `ABSENT_BLOCK_NORM`, `MAX_NEIGHBORS`, `BYTES_PER_TOKEN`, and every factor in `UNITS` keep their values in every
   mutant. The operators around them are mutated (`<=` in `ratio` becomes `>`, `*` becomes `+` or `/`). The values
   themselves stay pinned by named tests that exist today
   (`a_rounded_conversion_is_a_small_change_and_not_an_identical_value` for the tolerance, `the_factor_of_a_range_is_that_of_the_end_that_moved_more` for `max`,
   `thresholds_outside_the_cosine_range_are_refused` for the range). Mutation testing does not replace them, and
   the README should say what it cannot see.
4. **Which boundary mutations exist depends on the version.** `<` to `<=` and `>` to `>=` were added in
   cargo-mutants 25.1.0. `<=` is only ever turned into `>`, and `>=` into `<`, never into `<` and `>`. So the
   version must be pinned, and "`<` against `<=` at a floor" is generated only from `<` (which is how every floor
   and threshold here is written, so the case the suggestion cares about is covered).
5. **The exclusion list is the wrong shape.** Excluding `model/bert.rs` and the loader leaves in everything else
   whose real coverage is ignored tests or that is not a verdict: `model/sentence.rs`, `embed.rs`, `tokenize.rs`,
   `QdrantStore` (whose contract tests are `#[ignore]`d, so every mutant there survives), `fetch.rs`, `worker.rs`,
   `grpc.rs`, the report and event renderers, and the `testing` modules, which are not `cfg(test)` (they are
   exported under the `testing` feature) and would be mutated. An **allowlist** of the files whose boundaries are
   verdicts is shorter, says what it is for, and does not silently grow when a file is added.
6. **"Weekly in CI; `--in-diff` on pull requests"** assumes a CI that runs. There is no remote, so there are no
   pull requests and no schedule. The deliverable is a `just` recipe; a workflow is a decision for when a remote
   exists (below).
7. **Nondeterminism was not mentioned.** The `properties` modules draw a fresh random seed on every run. A mutant
   that only one in five seeds exposes would be "caught" in one run and "missed" in the next, and a missed-mutant
   list that changes between identical runs cannot be triaged. Fixing the seed for the mutation run solves it.
   Failure persistence must also be off: a failure under a mutant would otherwise write a `proptest-regressions/`
   file into the copied tree and be replayed against every later mutant built there.
8. **Running time was not estimated, and is not small** (below). It is too slow for `just ci`.
9. **`checkpoint.rs` is a different kind of target.** Its rule (restore parameter ids before the optimizer state)
   is a safety property of the trainer, tested by fast tests in `trainer.rs`
   (`resuming_without_the_recorded_parameter_ids_would_not_be_the_same_run`,
   `a_run_resumed_from_a_checkpoint_ends_exactly_where_an_uninterrupted_one_does`,
   `a_checkpoint_is_only_resumed_by_the_run_that_wrote_it`). It belongs in a second, separately run set, because
   those tests run real optimizer steps and the trainer is changed far less often than the policy code.
10. **What to do with a survivor** was "triaged into new tests". It also has to say when a survivor may be
    excluded instead, or the exclusion list becomes the place untested code goes. That rule is below.

## Design

### The tool

`cargo-mutants`, pinned at **27.1.0** (released 2026-06-02; it needs Rust 1.88 or later, and the toolchain is
1.96.0). It is the maintained mutation tester for Rust, and the only one that works on a stable toolchain without
instrumenting the code. Installed once per machine:

```sh
cargo install --locked cargo-mutants@27.1.0
```

It is a cargo subcommand, not a dependency. Nothing is added to any `Cargo.toml` dependency table, `Cargo.lock`
does not change, and `cargo deny` never sees it (it checks the workspace's dependency graph). Its own license is
MIT. Rust end to end: no script, no Python, no `jq`.

What it generates (27.1.0): each function body replaced by plausible values of its return type (`()`, `0`, `1`,
`true`, `false`, `None`, `Ok(Default::default())`, an empty `Vec` or `String`, ...); `==` and `!=` swapped; `&&`
and `||` swapped; `<` to `==`, `>`, `<=`; `>` to `==`, `<`, `>=`; `<=` to `>`; `>=` to `<`; `+ - * /` swapped
among themselves; `!` and unary `-` deleted; match guards replaced by `true` and `false`; arms deleted from matches
with a wildcard. It skips anything under `#[cfg(test)]` or `#[test]`, so the `properties` modules and every
`mod tests` are never mutated. The plan's first step confirms this list with `cargo mutants --list` on the tree,
including whether `Policy::new` and `Schema::new` (functions named `new`) get body replacements or only operator
mutants.

### Configuration: `.cargo/mutants.toml`

The default location, read on every run. A new `.cargo/` directory holds only this file; it is not
`.cargo/config.toml` and changes nothing about how cargo builds.

```toml
# cargo-mutants configuration: `just mutants`. https://mutants.rs
#
# The files listed here are the ones whose boundaries are verdicts (docs/decisions.md, D22). A mutant that survives
# in one is an untested boundary: kill it with a test, or, only if it cannot change any observable behavior,
# exclude it below with the reason. Never exclude a mutant because it is hard to kill.

examine_globs = [
  "crates/similarity-core/src/score.rs",
  "crates/similarity-core/src/quantity.rs",
  "crates/similarity-core/src/schema.rs",
  "crates/similarity-core/src/vector.rs",
  "crates/similarity-core/src/record.rs",
  "crates/similarity-server/src/engine.rs",
  "crates/similarity-server/src/offline/stats.rs",
  "crates/similarity-server/src/offline/calibrate.rs",
]

# Every test that can kill a mutant in these files is in a crate's library target. The integration targets either
# need the model or Qdrant (and are ignored) or test the store, so building and linking them for every mutant buys
# nothing. Add `--tests` back if a fast integration test for one of these files is ever written.
additional_cargo_test_args = ["--lib"]

# A mutant can leave a parameter unused or a value unread. That is a warning, and must not make the mutant fail to
# build: an unviable mutant is a mutant nobody tested.
cap_lints = true

profile = "mutants"

# Mutants that cannot change behavior, each with why. Matched on the function and the replacement, never on a
# line number, so that editing the file does not silently un-exclude or mis-exclude one.
exclude_re = [
]
```

Two choices in it deserve a word:

- **Each crate's mutants are tested by that crate's tests only** (the tool's default; no `test_workspace`). A
  mutant in `score.rs` must be killed by `similarity-core`'s tests, not by the engine's scenarios in
  `similarity-server`. That is stricter (the rule is pinned next to the code that implements it, which is how
  `CLAUDE.md` wants tests written) and much faster (a core mutant rebuilds one crate, not the server and its
  gRPC stack). The cost is that a few survivors in core may be "covered" by an engine test already; the answer is
  still a core test, usually a two-line one.
- **The allowlist is the scope.** `examine_globs` is used, not `exclude_globs`.

### A profile for the run: `Cargo.toml`

```toml
# Mutation testing (`just mutants`) builds the workspace once per mutant. Debug info is most of a debug build's
# size and link time and nothing reads it there. Dependencies stay optimized, as in `dev`, or the tests that run
# the tiny encoders take as long as the build.
[profile.mutants]
inherits = "test"
debug = "none"

[profile.mutants.package."*"]
opt-level = 3
```

The package override repeats the one in `[profile.dev.package."*"]` explicitly rather than relying on whether a
custom profile inherits package overrides. The `profile` is not a dependency; `cargo deny` does not look at it.

### A second set for the trainer: `.cargo/mutants-train.toml`

```toml
# The trainer's checkpoint rules (docs/decisions.md, D13): `just mutants-train`. Kept apart from
# `mutants.toml` because these tests run real optimizer steps, and the trainer changes rarely.
examine_globs = ["crates/similarity-train/src/checkpoint.rs"]
additional_cargo_test_args = ["--lib"]
cap_lints = true
profile = "mutants"
exclude_re = []
```

It is a separate file because list options from a config file are appended to the command line's, so a `--file`
flag on top of `mutants.toml` would test both sets. It is read with `--config .cargo/mutants-train.toml`, which
replaces the default file.

### Recipes: `justfile`

In the `test and lint` section, after `test-qdrant`:

```just
# Environment for a mutation run: one fixed proptest seed, so that a mutant is caught or missed the same way on
# every run, and no regression files written into the copied tree. The commit is fixed so that the build script
# does not ask `git` in a tree that has none.
mutants_env := "PROPTEST_RNG_SEED=1 PROPTEST_DISABLE_FAILURE_PERSISTENCE=1 RECORD_SIMILARITY_COMMIT=mutants"

# Mutation testing of the files whose boundaries are verdicts (slow; see README "Mutation testing")
mutants *flags:
    {{ mutants_env }} cargo mutants --output target {{ flags }}

# Only the mutants in what changed since `base`: `just mutants-diff` (uncommitted), `just mutants-diff main`
mutants-diff base="HEAD":
    git diff --no-ext-diff {{ base }} > target/mutants.diff
    {{ mutants_env }} cargo mutants --output target --in-diff target/mutants.diff

# Mutation testing of the trainer's checkpoint rules
mutants-train *flags:
    {{ mutants_env }} cargo mutants --config .cargo/mutants-train.toml --output target/mutants-train {{ flags }}
```

- `--output target` puts the report in `target/mutants.out/`, which `.gitignore` already covers, so `.gitignore`
  does not change. It holds `outcomes.json`, one file per outcome (`caught.txt`, `missed.txt`, `timeout.txt`,
  `unviable.txt`), a diff per mutant, and the build and test log of every mutant.
- `{{ flags }}` passes through what is useful by hand: `--list` (what would be tested, no build), `-f <file>` to
  narrow a run to one file, `--re <regex>` to re-test one mutant after writing the test that should kill it,
  `-j 3` for parallel jobs.
- The exit code is the tool's: 0 when every viable mutant was caught, 2 when some were missed, 3 on timeouts, 4
  when the unmutated tests already fail. A recipe that fails on a survivor is the gate (see the decisions).
- `--in-diff` only sees changed lines in files that produce mutants. A change that only deletes a test does not
  trigger anything, and an untracked new file is not in `git diff`. The full run is the real check;
  `mutants-diff` is the quick one before a commit.
- `mutants` is **not** part of `just ci`. Its running time is measured in hours of CPU, not seconds.

### Where it runs, and how long it takes

On the owner's machine (12 cores, 24 GB), by hand, after changing the engine's write path, the anchor, the
scoring, the quantity comparison, or the calibration statistics: the same trigger as the existing
`PROPTEST_CASES=5000 just test` rule in `CLAUDE.md`. `mutants-diff` before committing such a change, `mutants`
after a series of them.

A rough estimate from counting operator sites and functions outside the test modules of the eight files: about 300
to 450 mutants. Each costs an incremental rebuild of one crate and its library test binary (for the server, a link
against tonic, Qdrant's client, and Burn) plus one run of that crate's library tests, so perhaps 15 to 45 seconds.
That is two to five hours serially, and roughly an hour at `-j 3`, after a first full build in each job's scratch
copy. The scratch copies live under `TMPDIR`, each with its own `target`; at `debug = "none"` that is a few GB per
job, not the 35 GB of the current `target/debug`. None of these figures goes into the README until step 1 has
measured them (`CLAUDE.md`: every number there was measured).

Step 1 also decides `-j`. Three jobs is the starting point: Burn's test builds are memory-hungry, and the tool
itself warns above 8.

### What to do with a surviving mutant

In this order of preference, and each survivor gets exactly one:

1. **A named example test** that fails under the mutant and passes on the real code. This is the usual answer for
   a boundary: "a score exactly at the threshold passes" is a sentence the suite should contain anyway.
2. **A property**, in the crate's `properties` module, when the survivor shows a rule that holds for every input
   rather than one case (D19). Keep the case that exposed it as a named example test too.
3. **A simplification**, when the survivor shows that the code does not need what was mutated (a condition already
   implied by another). Remove it, and say so in the commit.
4. **An exclusion** in `exclude_re`, only for an equivalent mutant: one that no input can tell apart from the real
   code. The entry names the function and the replacement, never a line number, and carries a comment saying why
   no input can tell. "Hard to test" is not a reason; "only observable through the clock" might be, and is
   discussed below.

Timeouts are treated like survivors: either the mutant made a loop infinite (then the tool already counts it as
detected, and nothing is needed) or a test is slow enough to be mistaken for one, which is a fact to fix in the
test.

No event and no log line is added for any of this. The report is the tool's own, in `target/mutants.out/`; D15
governs the product's logs, and a development tool's output is not one of them.

### Survivors to expect

From reading the code, without running anything. The first run decides; these are where to look first.

- **`judge` and `Comparison::weigh` at equality.** No test scores a revision exactly at the threshold, a field
  exactly at its floor, or a record exactly at its drift threshold, so `similarity < policy.threshold` becoming
  `<=` (and the same for the floor and the drift threshold) probably survives. Exact equality is easy to
  construct without guessing a float: measure once, then judge again with the threshold set to the measured
  value (`Evaluation::rejudge` with a `Policy` whose `threshold` is `evaluation.comparison.similarity`). That is
  three mutants and one test.
- **`Quantity::check` at zero.** `value < 0.0` becoming `value <= 0.0` would refuse a quantity of zero, which is a
  legitimate value (a CBD of 0 %). If no test passes a zero through `check`, it survives.
- **`Engine`'s `now`.** Replacing its body with `0` or `1` changes only `updated_at`, which no engine test reads.
  This is the one case where an exclusion could be argued (the clock is not a verdict), but D20 makes the
  timestamp part of what an import must leave alone, so the recommended answer is a test that a write stamps a
  time no earlier than the request.
- **The `identical` conjunction in `revise`.** `candidate.hashes == baseline.vectors.hashes` looks implied by every
  field being `Unchanged` (an unchanged field needs both hashes present and equal, and a field on one side only is
  `Added`, `Removed`, or `Restored`). If the mutants of that conjunct survive, the clause is redundant and the
  answer is a simplification, not a test, and the engine state machine already checks the write rules it serves.
- **`stats.rs` guards on empty denominators** (`chances > 0`, `of > 0` becoming `>=`): only a test with zero
  chances notices the `NaN`. The interval and rate tests may or may not include one.

## The work, in order

Each step is one commit. Nothing is committed until the owner asks.

### 1. Tooling, and the first measured run

- Add `.cargo/mutants.toml`, `.cargo/mutants-train.toml`, `[profile.mutants]` in `Cargo.toml`, and the three
  recipes.
- `cargo install --locked cargo-mutants@27.1.0`; `just mutants --list` to confirm the mutant count per file and
  the mutation kinds listed above (in particular for functions named `new`). Adjust the allowlist if a file
  produces nothing.
- Run `just mutants -j 3` once, timed. Record the count, the outcome per file, and the wall time. That run is the
  input to the triage below; the time is what the README quotes.
- `just ci` passes (the profile and the config change nothing it builds).

### 2. Triage `score.rs` and `quantity.rs`

Tests first: for each survivor, write the test, then `just mutants -f <file> --re '<mutant>'` to see it caught.
Expected: the equality tests for threshold, floor, and drift; zero as a valid quantity. Any exclusion goes into
`exclude_re` with its reason in the same commit.

### 3. Triage `schema.rs`, `vector.rs`, `record.rs`

Same procedure. These are validation and arithmetic on vectors; expect survivors in `Schema::new`'s validation
messages and in `normalize`'s degenerate-length branch.

### 4. Triage `engine.rs`

Same procedure, plus the decision about the `identical` clause and the `now` test. Run
`PROPTEST_CASES=5000 just test` once after any change to the write path, per `CLAUDE.md`.

### 5. Triage `offline/stats.rs` and `offline/calibrate.rs`

Same procedure. Any new statistic test lives beside the existing ones in `stats.rs`; nothing moves into a renderer
(`CLAUDE.md`: every number comes from `offline::stats`).

### 6. The trainer set

`just mutants-train`, timed, and the same triage for `checkpoint.rs`. Separate because it is slower and changes
for different reasons.

### 7. Documentation and the decision record

A full `just mutants` with no survivors (or only commented exclusions), the README paragraph with the measured
time, the `CLAUDE.md` rules, and the decision record.

## Tests

None of these needs the model or Qdrant: they are ordinary fast tests, run by `just test`. The names below are
the expected ones; the first run decides which are needed, and each is kept only if it kills a mutant that
survived.

In `similarity-core/src/score.rs`:

- `a_score_exactly_at_the_threshold_passes`
- `a_field_exactly_at_its_floor_is_not_below_it`
- `a_record_exactly_at_the_drift_threshold_has_not_drifted`

(Or one test, `a_score_exactly_at_any_threshold_floor_or_drift_limit_passes`, if the owner prefers fewer.)

In `similarity-core/src/quantity.rs`:

- `a_quantity_of_zero_is_a_valid_measurement`

In `similarity-server/src/engine.rs`:

- `a_write_stamps_the_time_it_was_made`

In `similarity-server/src/offline/stats.rs`, if the guards survive:

- `a_label_with_no_pairs_has_no_error_rate`

And the rest as the run finds them, each named as the behavior it pins. A survivor that shows a rule for every
input becomes a property in the crate's `properties` module, with the shrunk case kept as an example test and the
`proptest-regressions/` file committed, as `CLAUDE.md` already requires.

The mutation run itself is not a test target and has no `#[ignore]`d counterpart.

## Documentation to update

- **README.md, "Development".** Add `just mutants`, `just mutants-diff`, and `just mutants-train` to the command
  block, and a paragraph after "Property tests": what mutation testing is checking, which files and why, that it
  is slow (the measured time from step 1 and the job count it was measured at), that it needs
  `cargo install --locked cargo-mutants@27.1.0` once, where the report lands, and what it cannot see (constants,
  method calls, which the named tests cover instead). Also add `.cargo/` to "Project layout", and correct the
  stale `docs/` lines there while editing it (`feature-suggestions.md` "none built yet" and `plan-features.md`
  "a plan for the first five" are both out of date).
- **CLAUDE.md, "Testing".** One rule: after changing `score.rs`, `quantity.rs`, `schema.rs`, `vector.rs`,
  `record.rs`, `engine.rs`, `offline/stats.rs`, or `offline/calibrate.rs`, run `just mutants-diff`, and a full
  `just mutants` before calling a series of such changes done; a surviving mutant is killed by a test or excluded
  in `.cargo/mutants.toml` with the reason it is equivalent, never because it is hard to kill. Add a file to the
  allowlist when a new file starts deciding a verdict. **"Commands".** The three recipes.
- **CLAUDE.md, "Things that are easy to get wrong".** Mutation testing cannot see constants or method calls: a
  change to `RELATIVE_TOLERANCE`, a `UNITS` factor, or a `saturating_*` call is covered only by the named tests,
  so a clean mutation run says nothing about it.
- **docs/decisions.md.** A new record, **D22. The verdict code is checked by mutation** (or whatever number is
  next when this lands; other planned features may take D22 first): why mutation testing on top of D19's
  properties (properties check rules on generated inputs, mutation checks that the suite would notice a wrong
  operator), why an allowlist, why per-crate test scope, why a fixed proptest seed, why not in `just ci`, the
  survivor rule, the blind spots, and what the first run found. D19's "It can fail" paragraph gets one sentence
  pointing to it.
- **docs/plan-features.md.** Move feature 15 to "What was built".

## Dependencies on other features

None of the unbuilt features is required. Where one lands later, the allowlist grows:

- **Feature 9 (pinned verdict snapshot)** and **feature 4 (promotion gate)** turn `calibrate-diff --fail-on` into a
  gate, so the function in `offline/diff.rs` that decides a flip becomes a verdict. Assumed interface: a pure
  function from two runs' pair outcomes and a `--fail-on` choice to pass or fail. When that exists, add
  `crates/similarity-server/src/offline/diff.rs` to `examine_globs`.
- **Feature 19 (reason codes on the wire)** adds a mapping in `grpc.rs`. `CLAUDE.md` keeps `grpc.rs` free of
  decisions, so it stays out of the allowlist; if the mapping turns out to decide anything, that is the bug.
- **Feature 14 (fuzzing)** is complementary and shares nothing with this: it looks for inputs that crash, this
  looks for operators no test pins.
- **Feature 17 (reviewer feedback)**, if it adds a "does `before` match the stored baseline" check to the engine,
  is covered automatically, being in `engine.rs`.

Nothing here can only be validated in the owner's work environment: mutation testing runs the synthetic fast suite
and reads no data. The one thing that differs between machines is the running time, which is why the README
quotes the machine and the job count with it.

## Decisions for the owner

1. **Is a survivor a failure?** Recommended: yes, once steps 2 to 6 are done. `just mutants` exits non-zero on any
   missed mutant, and the rule in `CLAUDE.md` says a series of changes to these files is not done until it passes.
   The alternative, report-only, lets survivors accumulate, which is the state this feature exists to end.
2. **How equivalent mutants are excluded.** Recommended: `exclude_re` in `.cargo/mutants.toml`, with a comment per
   entry. The alternatives are `#[mutants::skip]` in the source, which needs the `mutants` crate as a regular
   (not dev) dependency of each library (tiny, MIT, but a dependency in the release graph for a test tool's
   sake), or `#[cfg_attr(mutants, mutants::skip)]` with a never-set `cfg`, which needs no dependency but needs
   `unexpected_cfgs` taught about `mutants` in `[workspace.lints]`. Both put a tool's annotation into the code.
3. **Test scope.** Recommended: each crate's own library tests (the default). The alternative, the whole workspace
   for every mutant, finds fewer survivors in core because the engine's tests also exercise `score.rs`, and is
   several times slower.
4. **CI.** Recommended: no workflow now; there is no remote for it to run on. When there is one, a weekly
   scheduled job on Linux running `just mutants -j 2` (the snippet goes in the decision record), and no
   `--in-diff` job on pull requests until the full run's time is known. If the code also lives in a work
   repository with CI, the same recipe runs there unchanged.
5. **Proptest cases during a mutation run.** Recommended: the default 64, with the seed fixed. More cases kill a
   few more mutants and multiply the time; the state machine already caught all three deliberate breaks within 64.
6. **The trainer set.** Recommended: include it, as its own recipe (`just mutants-train`), run after changing
   `checkpoint.rs` or the resume path. The alternative is to leave the trainer out until feature 3 or 6 changes
   it.
7. **A decision record of its own or a paragraph in D19.** Recommended: its own (D22 or next), because the survivor
   rule and the blind spots are rules someone needs to find.

## Size

**Small to set up, medium overall.** The tooling is two config files, a profile, three recipes, and documentation:
a day including the first measured run. The triage is the real work and cannot be sized until that run: at a
guessed 5 to 10 percent survival over 300 to 450 mutants, it is 15 to 45 survivors, most of them a short test
each, a few needing a decision (a redundant clause, an equivalent mutant). That matches the suggestion's estimate
("S to start, M to work through the survivors"), and nothing in it needs real data, the model, or Qdrant.
