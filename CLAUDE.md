# record-similarity

A gRPC server (tonic) that decides whether an edit to a record is still the same record, to catch "recycled"
marketplace listings. Each revision is embedded field by field with a small BERT-class sentence encoder running on
[Burn](https://burn.dev), compared with the stored baseline, and accepted or rejected against calibrated
thresholds. A record's quantities (weight, volume, potency) are compared as numbers beside that, never embedded.
Vectors live in Qdrant, which also provides neighbor search.

Read `README.md` for behavior and `docs/decisions.md` (D1 to D32) for why each piece is the way it is before
changing a design. Infer conventions from the existing code first.

## Code style: non-negotiable

- **Two-space indentation, exclusively.** Rust, TOML, YAML, Python, everything except the `justfile`.
  `.editorconfig` and `rustfmt.toml` enforce it.
- **Aim for ~120-character lines**, including comments.
- Comments and doc comments explain constraints and the non-obvious *why*, not what the next line does. Every file
  opens with a `//!` header; every public item has `///` with `# Errors` where it can fail.
- **No `unwrap()`, `expect()`, `panic!`, indexing, or unchecked integer arithmetic in non-test code.** The workspace
  lints deny them (`[workspace.lints]` in the root `Cargo.toml`). Use `?`, `.get()`, iterators, and
  `saturating_*`/`checked_*`. For Burn tensors use the method forms (`.add()`, `.mul()`, `.div()`), since the
  arithmetic lint also fires on operator overloads. Tests may unwrap and index (`clippy.toml`).
- No `println!`. Libraries log through `tracing`. **All logging is NDJSON, set up only in `similarity-log`**
  (D15): never add a text format, a second subscriber, or a preformatted string where a number would do. Log
  values as numbers and booleans. A training result is an event with `target: similarity_log::RESULT` (stdout,
  never filtered); a failure goes through `similarity_log::failed`. The server's `model fetch`, `compare`, and
  `calibrate` reports are for people and are written to a locked `stdout` with `writeln!`.
- `thiserror` enums in the libraries (`similarity-core`, `similarity-store`, `similarity-train`'s `lib`); `anyhow`
  only in the two binaries.
- Imports grouped std, external, `crate::`, separated by blank lines.
- No em-dashes in prose. US English.
- Clippy-clean at `-D warnings`, pedantic included.

## Testing: write tests first wherever possible

- `cargo test` must need **no model, no network, and no Qdrant**. It uses tiny randomly initialized encoders and
  an in-memory tokenizer (`similarity_core::testing`, exported to other crates' tests by the `testing` feature),
  `MemoryStore`, and a bag-of-words `WordBagEmbedder`.
- Anything needing the 90 MB checkpoint or a Qdrant server is `#[ignore = "..."]` with the reason and the `just`
  recipe that runs it.
- Test names are sentences stating the behavior (`a_recycled_record_is_rejected_and_the_baseline_is_untouched`).
- Store implementations must pass the shared contract in `crates/similarity-store/tests/contract.rs`.
- **A rule that must hold for every input belongs in a `properties` module** (`similarity-core`, `similarity-server`;
  proptest, D19), beside the example tests and not in place of them. After changing the engine's write path, the
  anchor, or the scoring, run `PROPTEST_CASES=5000 just test` once. When a property fails, fix the code or the
  rule, keep the shrunk case as an ordinary named test, and commit the `proptest-regressions/` file it wrote.
- **The tests of the verdict-deciding files are themselves tested** (D25). After changing the scoring, the
  quantity comparison, the schema, the engine's write path, the calibration statistics, the match ranking, or
  the promotion gate, run `just mutants -j 3`
  (25 minutes; `just mutants-diff` for a quick look) from a clean checkout, never from a tree being
  edited. A surviving mutant gets a test that states the boundary. Exclude one in `.cargo/mutants.toml` only if
  the change cannot alter behavior, with the reason, and never because it is hard to kill.
- **Verdicts on curated hard cases are pinned** (D22). `data/regression.jsonl` is synthetic and stays that way;
  `data/regression.snapshot.ndjson` is what the pinned model and `config.example.toml` make of it. A change to the
  config or the scoring code that moves any case fails `cargo test` (the snapshot is judged again, with no model);
  a change to the encoder or tokenizer that moves one fails `just test-snapshot`. If the move is intended, run
  `just snapshot` (it needs the model) and put what it prints in the commit message. Never read a threshold from
  these cases, and never edit the snapshot by hand.
- After touching the encoder, the loader, the tokenizer settings, the trainer, or `config.example.toml`, run
  `just test-real`. It is the only thing that proves numerical parity with PyTorch (embeddings and training steps)
  and that the shipped thresholds still behave.
- Before declaring a task done: `just ci` (fmt-check, clippy, test, deny).

## Workspace layout

- `crates/similarity-log`: the one logging setup both binaries use.
- `crates/similarity-proto`: the `.proto` (the source of truth for the API, and where its docs live) and generated
  code.
- `crates/similarity-core`: encoder, strict weight loading and saving, tokenization, composite vectors, quantities,
  scoring, policy, and scoring a listing against stored records (`matching`). Synchronous, no I/O beyond reading and writing a model directory, no network code ever (D11).
- `crates/similarity-store`: the `VectorStore` trait, `MemoryStore`, `QdrantStore`.
- `crates/similarity-runs`: the local record of runs (D23): the manifest, starting and ending a run, reading runs
  back. It records and lists; it decides nothing.
- `crates/similarity-server`: config, the embedding worker thread, `Engine` (all behavior), the thin gRPC layer,
  `model fetch`, `import`, `compare`/`calibrate`/`calibrate-drift`/`calibrate-match`/`calibrate-sweep` (the `offline` module), `load` (the load
  test), `runs` (reading recorded runs back, comparing and reporting them), `shadow` (a candidate policy judging
  live requests), `registry` (which model was put into service, and the gate), and the `record-similarity`
  binary. Keep `grpc.rs` free of decisions.
- `crates/similarity-train`: fine-tuning (data, losses, schedule, training loop) and the `record-similarity-train`
  binary. The only crate that enables Burn's `autodiff` and `optim`; the server must never depend on it (D13).

## Things that are easy to get wrong here

- **Thresholds are measurements, not opinions.** Never change `threshold`, `anchor_threshold`, or a `reject_below`
  in `config.example.toml` without re-running `just calibrate` and updating the numbers quoted in the config
  comments, `README.md`, and `docs/decisions.md` (D3). Thresholds do not transfer between models:
  `model.expected_fingerprint` is how a config says which model its thresholds belong to, and it is opt-in by the
  owner's choice. Do not set it in `config.example.toml`, and do not make it part of the fingerprint or layout.
- **The model fingerprint and the layout id guard stored data** (D12). Anything that changes what vectors mean
  (model files, `output_dim`, `max_tokens`, the set of field names) must change the fingerprint or layout. Weights
  and thresholds must not.
- **A rejected revision must never write** (D7). A normal accepted write must never replace a value the anchor
  holds; it may only give the anchor the first value of a field it has never had (D8). Fields present on one side
  only stay out of the score, except a `Restored` field, which is scored against the anchor. Do not "simplify"
  that away: it is what stops a field being swapped by blanking it and filling it in again.
- **Quantities never enter the weighted score or the layout id** (D16). They are compared as numbers
  (`similarity_core::quantity`), stored in the payload beside the vectors, and judged by
  `Evaluation::judge_quantities` after the text is scored. Do not embed one, do not fold one into `similarity`, and
  do not add one to `Schema::layout_id`: the first changes what every threshold means, the second forces a
  re-index for no reason. The anchor rules above apply to them unchanged (first value kept, `Restored` compared
  against it, `FORCE` resets).
- **Quantity limits are measurements, like thresholds.** `config.example.toml` ships every quantity with no
  `reject_above_factor`, no `reject_incomparable`, and `reject_on_measure_change` off. Never add a limit there
  that was not read from `just calibrate` on real labeled pairs, and never from the synthetic sample.
- **An unknown unit is never an error.** It is its own dimension, comparable only with itself. Extend the `UNITS`
  table in `quantity.rs` when a new unit turns up; do not make the server refuse it.
- **Calibration events are flat, and their names are an interface** (D17). `calibrate --ndjson` writes one `pair`
  line and then one line per field, quantity, and reason, joined on `pair_id`, through `similarity-log`. Do not
  nest an object or a list into an event (it could only be a preformatted string), and do not rename an event, a
  field, or a `Reason::code`: other tools read them. Add fields; never repurpose one. The API carries the same
  codes as `ReasonCode` (`REASON_CODE_` plus the code in capitals, held by a test in `grpc.rs`): a new `Reason`
  needs a new code in both places and a new enum number, never a reused one.
- **Every number in the text report and in the events comes from `offline::stats`**, so the two cannot disagree.
  Put a new statistic there, not in a renderer.
- **An import only creates** (D20). `Engine::import` writes an id that is not stored and leaves one that is
  exactly alone, timestamp included. That is what makes it safe to run twice and unable to replace an anchor.
  Never give it an update path; an update is an `Evaluate`, which judges.
- **A tenant, once stored, must be named by every later request for that id**, including `Delete` (D14).
- **What a supported model directory is lives in one place**, `similarity_core::model::Pipeline`. The loader and
  `model fetch` both use it; never parse `modules.json` anywhere else.
- **Burn 0.22 is not the Burn in older tutorials**: models are not generic over a backend, a runtime `Device`
  selects it, `ndarray` is deprecated in favor of `flex`, and records/recorders were replaced by `burn-store`.
  Check the crate source under `~/.cargo/registry` rather than trusting memory.
- If the model or its revision changes: update the pinned revision and checksums in `fetch.rs`, regenerate the
  parity fixture (`just golden`), re-calibrate, and update `models/README.md`.
- **`output_dim` has no default on purpose** (D4): truncation is only valid for Matryoshka-trained models. When
  comparing models with `calibrate`, check the logged `dimension=` to confirm each ran at the width you meant.
- **Every command and number in `README.md` was run and measured.** Keep it that way: re-run an example before
  editing it, and never quote a figure that was not produced by `just throughput`, `just load`, or `just calibrate`.
- **Rust end to end, by the owner's choice** (D13): training is `similarity-train`, in Burn. Do not add Python to
  any pipeline. `tools/*.py` are referees that write PyTorch fixtures for the parity tests and nothing else.
- **Histories are measured first and judged afterwards** (D27). `calibrate-drift` replays every version on an
  engine that can reject nothing, then judges each step under the config. Never replay under the configured
  policy: after the first rejection the baseline stops moving and the rest of the history is fiction.
  `anchor_threshold = 0.85` in `config.example.toml` is a starting point, not a measurement; do not replace it
  with a number from `data/drift.sample.jsonl`, which is synthetic.
- **A match is a read, scored exactly** (D32). `Match` never writes, a listing is mapped onto the product schema
  and never gets a layout of its own, and a listing that carries an `id` or a `tenant` is refused. The engine and
  `calibrate-match` both rank with `similarity_core::matching::rank`: do not score a match anywhere else, do not
  apply a floor or a quantity limit to one, and do not let a quantity move a match score or decision (D16).
  `matching.threshold` is a measurement and `config.example.toml` ships none; `matching.overfetch` is a cost that
  `calibrate-match` measures. Never read a threshold from `data/catalog-groups.sample.jsonl`, which is synthetic.
  The `match_*` event names are an interface like the calibration ones. `VectorStore::search_vectors` must
  return what `search` returns, in the same order: both are built from one request, and the contract test holds
  them to it.
- **The promotion gate measures and chooses nothing** (D31). `model promote` never suggests or writes a
  threshold, and never accepts a calibration of another model or of any split but `holdout`, with or without
  `--override`. Each model is judged under its own config (the candidate's, and the one the model in service was
  promoted with), never one model at another's thresholds. The registry (`models/registry.ndjson`) is appended
  to and never rewritten, lives beside the model directories and never in one, and is never read by the server.
  Its entry names are an interface: add fields, never rename or repurpose one.
- **A shadow decides nothing** (D30). `[shadow]` is a candidate policy that the engine tells about each judged
  request through `Observer`, after the outcome is final and anything it writes is written. Never give `observe`
  a return value, a way to fail the request, or anything to wait on, and never let a shadow judge with
  `Evaluation::rejudge` (the similarity to the anchor has to be computed under its own weights). The `shadow*`
  event names are an interface like the calibration ones, and a line carries an id, a tenant, scores, and codes:
  never a record's text. `config.example.toml` ships no shadow and no number for one. A shadow measures
  agreement, not quality: do not add anything that ranks or adopts a candidate by its disagreements.
- **A sweep is a search, and what it finds is fitted** (D29). `calibrate-sweep` ranks on one calibration and may
  evaluate the best points on another at the threshold already chosen. Never report a swept point's errors on
  the run it was ranked on as if they were a measurement, and never let it choose a threshold on the evaluation
  run.
- **The holdout chooses nothing** (D24). A checkpoint, a threshold, or a setting is chosen on the `validation`
  split (`record-similarity pairs carve` makes one), and the holdout is measured once, for what was chosen. Do
  not add a code path, a recipe, or a sentence in the README that picks anything by its holdout numbers. Models
  are compared by AUC (`similarity_core::separation`), never by errors at a configured threshold, which belongs
  to the model it was measured on. Catalog groups of an owner in the validation or holdout split are not trained
  on.
- **A report computes nothing** (D28). `runs compare` and `runs report` lay out numbers from `runs::training`
  (a training run's events) and `offline::stats` (a calibration). Put a new number in one of those, never in
  `runs/html.rs` or `runs/compare.rs`. The page is one file with no script and nothing fetched; a chart shows one
  measure, never two axes, and is followed by a table of what it draws.
- **A run is recorded by `main`, never by the code it calls** (D23). `calibrate` and the trainer start a
  `similarity_runs::Run` before any work and pass it down as an `Option`; tests pass `None`, so no test writes
  into `runs/`. The record is a third layer in `similarity-log`, built by the same function as the two streams:
  never a second subscriber, and never a writer with a format of its own. `killed` is never stored; it is
  concluded on read from the lock. Do not record a command whose arguments can hold a secret.
- **`provenance.json` is a note about a model, never part of it** (D18). Whatever writes a model directory writes
  it, inside the same `write_whole`, and it records what produced the model, not what the model scored. Never add
  it to the files `ModelFiles::digest` hashes, never make loading depend on it, and never write into a model
  directory after the fact to add results.
- **Dropout is never in the module tree.** `forward_training` takes it as an argument; `forward` cannot apply it.
- **A checkpoint must restore parameter ids before the optimizer state** (`similarity-train/src/checkpoint.rs`).
  Burn files optimizer state under ids that change on every load; skipping this makes a resume silently train
  without its optimizer state. Anything that changes what a training step sees or does belongs in `RunIdentity`,
  or a resume under a different value will be accepted. Model directories and checkpoints are written through
  `write_whole` (sibling, flush, rename), never in place.
- **The trainer is held to PyTorch step for step** (`similarity-train/tests/real_parity.rs`). After changing a
  loss, the clipping, the optimizer settings, or the training forward pass, run `just test-parity`; if the change
  is intended, change `tools/train_golden.py` to match and regenerate with `just golden`.
- The loader must keep reading both sentence-transformers pooling-config formats (D2).

## Commands

- Build: `just build` (`cargo build --workspace`); release binary: `just release`; with Metal: `just release-metal`
- Test (fast): `just test`, or one crate: `just test similarity-core`
- Lint: `just clippy`; format: `just fmt`; everything CI runs: `just ci`
- Model: `just fetch-model`; Qdrant: `just qdrant-up` / `just qdrant-down`
- Slow tests: `just test-parity`, `just test-snapshot`, `just test-qdrant`, `just test-real`; `just snapshot` rewrites
  the pinned verdicts
- Mutation testing (needs `cargo install cargo-mutants`): `just mutants -j 3`, `just mutants-diff`,
  `just mutants-train`
- Run: `just serve` (needs Qdrant) or `just serve-in-memory`; then `just client describe`
- Load a catalog: `just import export.jsonl config.toml` (`--check` validates only)
- Measure: `just throughput`, `just load`, `just calibrate`, `just calibrate-drift`, `just calibrate-match`, `just calibrate-sweep`, `just calibrate-diff`, `just verdict-gate old.ndjson pairs.full.jsonl`
- Put a model into service: `just promote --config candidate.toml --calibration holdout.ndjson`; then
  `record-similarity model list`, `model register <dir>`, `model retire <digest> --reason "..."`
- See what was run: `just runs` (`calibrate` and the trainer each leave a directory under `runs/`; `--no-run`
  turns that off); `record-similarity runs compare A B`; `record-similarity runs report A B --out report.html`
- Make a validation split: `record-similarity pairs carve pairs.full.jsonl --fraction F > pairs.carved.jsonl`
- Fine-tune: `just finetune --base ... --out ... --pairs ...` (CPU); GPU: `just release-train metal`, then
  `./target/release/record-similarity-train --device metal ...`. Continue an interrupted run with `--resume
  <out>.checkpoints/step-NNNNNN` in place of `--base`. Results are NDJSON on stdout, logs NDJSON on stderr.
- PyTorch referees (need `uv`): `just golden` regenerates both fixtures; `just verify-model <dir>` is optional
