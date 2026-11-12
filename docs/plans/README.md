# Plans for the fifteen features that are not built

One plan per remaining suggestion in `docs/feature-suggestions.md`, each written by a separate planner that read
the code as it stands and checked the suggestion against it. Nothing here is built. This file is the index: what
the plans found, where they meet, and an order. Delete a plan when its feature lands.

## Found while planning

1. **Fixed: the long-field cut (D21) let padding hide a field.** Two unrelated products score 0.519 and are
   rejected; with every field prefixed by 18 KB of zero-width spaces they scored 1.000 and were accepted. 18 KB of
   combining marks did the same, and so did one visible 20,000-letter word followed by a space. The cut counted
   bytes that the tokenizer drops or folds into one token. A long text is now read a part at a time until the
   tokenizer reports the limit full, and D21 was rewritten to say so.
2. **Fixed in the text: the README told the reader to select on the holdout.** "Fine-tuning" said to set
   thresholds from the holdout calibration and to lower the learning rate if the holdout got worse. Both turn the
   holdout into a validation set. Plans 3 and 4 found this independently. The README now says to keep a third
   `split` value for choosing; plan 3's `pairs carve` is the tool that would write it.
3. **`anchor_threshold = 0.85` ships enforced and nothing records how it was chosen.** D8 explains the anchor, not
   the number. Plan 11 is how it would be measured.
4. **Fixed: two more that the fuzzing plan predicted.** A quantity factor could be infinite (`1e-300 g` against
   `1e300 g`), which an event cannot carry; it is now the largest finite number. A weight below about `1e-12` of
   the total made a field read as absent; `Schema::new` now refuses a weighting that lopsided.

`Evaluation::rejudge` reuses the anchor similarity as measured, so it is not exact under other weights when a
record has an anchor (plans 11 and 18). `calibrate-diff --config-b` is not affected: a pair has no anchor.

## The plans

| # | Plan | Size | Needs first | The main correction to the suggestion |
|---|---|---|---|---|
| 2 | [Run registry](02-run-registry.md) | medium, **built** | | Loss lines are filterable logs, so a tee loses them; a killed run is worked out on read |
| 3 | [Checkpoint evaluation](03-checkpoint-evaluation.md) | medium, **built** | | Results cannot go in a checkpoint (D18); catalog groups leak validation brands |
| 4 | [Model registry and gate](04-model-registry.md) | medium | | A model has a digest, not a fingerprint; the server cannot read a local registry |
| 5 | [Run report and compare](05-run-report.md) | medium, **built** | 2, 3 for curves | Comparing calibrations is already `calibrate-diff`; resumes need a chain |
| 6 | [Sweeps](06-hyperparameter-sweeps.md) | large, **half built** (the policy sweep) | 3 | Calibration code is not in core; epochs cannot be swept by picking checkpoints |
| 9 | [Verdict snapshot](09-verdict-snapshot.md) | medium, **built** | | Refill cases need an anchor on a pair; output has timestamps, so not byte-stable |
| 10 | [Score telemetry](10-score-telemetry.md) | medium | shares a step with 19 | Per-field values need flat lines joined on a request id; nothing is timed today |
| 11 | [Drift calibration](11-drift-calibration.md) | medium, **built** | `sequences.jsonl` to be useful | Replaying in `ENFORCE` makes up everything after the first rejection |
| 12 | [Latency and load test](12-latency-load-test.md) | medium, **built** | | Batch sizes are invisible to a client; the worker needs counters |
| 14 | [Fuzzing](14-fuzzing.md) | medium | | Does not need nightly; the state machine is private and index-driven |
| 15 | [Mutation testing](15-mutation-testing.md) | small, **built** | | The tool never mutates method calls or constants; proptest seeds must be fixed |
| 17 | [Reviewer feedback](17-reviewer-feedback.md) | medium | | A feedback event cannot nest a pair; a restored field replays differently |
| 18 | [Shadow evaluation](18-shadow-evaluation.md) | large, **policy half built** | 12 helps | A low-priority thread does nothing: Burn runs on the global rayon pool |
| 19 | [Reason codes](19-reason-codes.md) | small, **built** | | Mostly exists; a server-side word diff is impossible (only hashes are stored) |
| 20 | [Menu-item matching](20-menu-item-matching.md) | large | | A brand-name filter is impossible; search returns no vectors |

## Where the plans meet

- **`EvaluateOutput.reasons` becomes `Vec<Reason>`.** Step 1 of both 10 and 19, described the same way. Build it
  once, with 19.
- **The validation split.** 3 and 6 agree: `pairs carve` writes `split: "validation"` into the pairs file, and the
  trainer and `calibrate` select by it. The alternative (k folds in memory) is one decision for both.
- **The run directory.** 5 codes against the section of 2 named "What features 4 and 5 may rely on". 4 works on
  plain `calibrate --ndjson` files and takes a run directory once 2 exists.
- **An anchor on a pair.** 9 adds an optional `anchor` record to a calibration pair. That is also what would let
  17's exported feedback replay a restored field the way the server judged it, which 17 lists as a limit.
- **Two things called drift.** 10 names its command `drift` (production scores moving away from calibration); 11
  names its command `calibrate-drift` (a record moving away from its anchor). Rename 10's before building either.
- **Every plan but 19 calls its decision record D22.** Feature 9 took that number; the rest are numbered as they
  are written.
- **CI.** `.github/workflows/ci.yml` exists and the repository has no remote. 9 and 14 plan CI jobs; 15 defers its
  job until there is somewhere to run it. All three are right about half of that.

## Decisions that are the owner's and are not only technical

- **17:** reviewer feedback puts product text on the server's stdout (the plan's default is yes).
- **20:** a match searches every tenant unless the request names one (the plan's default).
- **10:** the tenant is logged as given. **2:** a run records the hostname and CPU model.
- **Numbers proposed as defaults that nothing has measured:** a 15% validation split (3), a warning below 40
  holdout recycles (4), a 0.001 score tolerance (9), a 5 times over-fetch (20).

Each plan ends with its full list of decisions and a recommended default for each.

## An order

1. Fix the D21 cut, and correct the README's holdout advice. Both done.
2. Run `calibrate --full` on the real pairs (nothing to build).
3. **19** and **9**: both built. Both small to medium, and 9 turns "run `just test-real`" from a rule into a check.
4. **15** (built) and **14**, at any point. 14's first targets confirm the two predicted findings above.
5. **11** is built, on synthetic histories; it is run for real once `sequences.jsonl` exists.
6. Before real fine-tuning: **2**, **3**, and **5** (all built), then **4**. **6** last; it is large and builds on 3.
7. Before production: **10** and **12** (12 is built). In production: **17**, then **18** (its policy-only half is built, D30; the model half waits for a candidate model).
8. **20** when the matching question is taken up; it is a separate product question.
