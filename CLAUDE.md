# record-similarity

A gRPC server (tonic) that decides whether an edit to a record is still the same record, to catch "recycled"
marketplace listings. Each revision is embedded field by field with a small BERT-class sentence encoder running on
[Burn](https://burn.dev), compared with the stored baseline, and accepted or rejected against calibrated
thresholds. A record's quantities (weight, volume, potency) are compared as numbers beside that, never embedded.
Vectors live in Qdrant, which also provides neighbor search.

Read `README.md` for behavior and `docs/decisions.md` (D1 to D20) for why each piece is the way it is before
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
- After touching the encoder, the loader, the tokenizer settings, the trainer, or `config.example.toml`, run
  `just test-real`. It is the only thing that proves numerical parity with PyTorch (embeddings and training steps)
  and that the shipped thresholds still behave.
- Before declaring a task done: `just ci` (fmt-check, clippy, test, deny).

## Workspace layout

- `crates/similarity-log`: the one logging setup both binaries use.
- `crates/similarity-proto`: the `.proto` (the source of truth for the API, and where its docs live) and generated
  code.
- `crates/similarity-core`: encoder, strict weight loading and saving, tokenization, composite vectors, quantities,
  scoring, policy. Synchronous, no I/O beyond reading and writing a model directory, no network code ever (D11).
- `crates/similarity-store`: the `VectorStore` trait, `MemoryStore`, `QdrantStore`.
- `crates/similarity-server`: config, the embedding worker thread, `Engine` (all behavior), the thin gRPC layer,
  `model fetch`, `import`, `compare`/`calibrate` (the `offline` module), and the `record-similarity` binary. Keep
  `grpc.rs` free of decisions.
- `crates/similarity-train`: fine-tuning (data, losses, schedule, training loop) and the `record-similarity-train`
  binary. The only crate that enables Burn's `autodiff` and `optim`; the server must never depend on it (D13).

## Things that are easy to get wrong here

- **Thresholds are measurements, not opinions.** Never change `threshold`, `anchor_threshold`, or a `reject_below`
  in `config.example.toml` without re-running `just calibrate` and updating the numbers quoted in the config
  comments, `README.md`, and `docs/decisions.md` (D3). Thresholds do not transfer between models.
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
  field, or a `Reason::code`: other tools read them. Add fields; never repurpose one.
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
  editing it, and never quote a figure that was not produced by `just throughput` or `just calibrate`.
- **Rust end to end, by the owner's choice** (D13): training is `similarity-train`, in Burn. Do not add Python to
  any pipeline. `tools/*.py` are referees that write PyTorch fixtures for the parity tests and nothing else.
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
- Slow tests: `just test-parity`, `just test-qdrant`, `just test-real`
- Run: `just serve` (needs Qdrant) or `just serve-in-memory`; then `just client describe`
- Load a catalog: `just import export.jsonl config.toml` (`--check` validates only)
- Measure: `just throughput`, `just calibrate`, `just calibrate-diff`
- Fine-tune: `just finetune --base ... --out ... --pairs ...` (CPU); GPU: `just release-train metal`, then
  `./target/release/record-similarity-train --device metal ...`. Continue an interrupted run with `--resume
  <out>.checkpoints/step-NNNNNN` in place of `--base`. Results are NDJSON on stdout, logs NDJSON on stderr.
- PyTorch referees (need `uv`): `just golden` regenerates both fixtures; `just verify-model <dir>` is optional
