# record-similarity

A gRPC server (tonic) that decides whether an edit to a record is still the same record, to catch "recycled"
marketplace products. Each revision is embedded field by field with a small BERT-class sentence encoder (defined
and trained in [Burn](https://burn.dev), run by ONNX Runtime), compared with the stored baseline, and accepted or
rejected against calibrated thresholds. A record's quantities (weight, volume, potency) are compared as numbers
beside that, never embedded. Vectors live in Qdrant, which also provides neighbor search.

Read `README.md` for behavior and `docs/decisions.md` (D1 to D48) for why each piece is the way it is before
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
- **Mutation testing is for extreme cases, and is not part of ordinary work** (D25; the owner's decision,
  2026-11-23, because it takes a long time). Do not run `just mutants`, `just mutants-diff`, `just mutants-train`,
  or a `cargo mutants` narrowed to a file after a change, and do not count a run among what a change needs to be
  done: `just ci` is that, with the thorough property pass where the rule above asks for it. An extreme case is
  a rewrite of how a verdict is decided (the scoring, the quantity comparison, the engine's write path), or the
  last check before the service first judges real edits. If it is not plainly one of those, it is not one; ask
  the owner before starting a run, and never start one unasked. Write the tests of a new boundary by hand
  instead, to the changes the tool would make (`<` against `<=`, a body replaced by a constant). When a run is
  made: from a clean checkout, never from a tree being edited; a surviving mutant gets a test that states the
  boundary; exclude one in `.cargo/mutants.toml` only if the change cannot alter behavior, with the reason, and
  never because it is hard to kill. A new file whose boundaries are verdicts still goes into that list, so that
  the next run covers it.
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
  amounts read from text (`amount`), scoring, policy, and scoring a menu item against stored records (`matching`). Synchronous, no I/O beyond reading and writing a model directory, no network code ever (D11).
- `crates/similarity-store`: the `VectorStore` trait, the `Scan` trait (listing what a store holds, which only the
  audit and the anchor backup do), `MemoryStore`, `QdrantStore`.
- `crates/similarity-runs`: the local record of runs (D23): the manifest, starting and ending a run, reading runs
  back. It records and lists; it decides nothing.
- `crates/similarity-server`: config, the embedding worker threads, `Engine` (all behavior), the thin gRPC layer, `quantities` (the
  `Quantities` service, thinner still: no engine behind it),
  `import_stream` (the loop behind the streaming `Import` call), `locks` (one lock for each record a request is
  at), `metrics` (the few numbers the server keeps
  about itself), `telemetry` (where its traces and metrics go: the exporters, the page, the traced store),
  `model fetch`, `import`, `audit` (the store checked against an export: `compare.rs` for one record and the counts,
  `mod.rs` for the run, `report.rs` for the report and the events), `anchors` (a backup of every record's anchor:
  `file.rs` for the file, `mod.rs` for the export and the restore), `compare`/`calibrate`/`calibrate-drift`/`calibrate-match`/`calibrate-sweep` (the `offline` module), `load` (the load
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
- **The words are the owner's company's** (D36). A record's brand is its `brand_id` (never `tenant`, never
  `owner`): the key of the brand that owns it, in the API, the store, the files, and the events. `brand` alone is
  the scored text field that holds the brand's name. What `Match` is given is a *menu item* (`menu_item`,
  `menu_items` in group files), never a "listing": where the owner works, a listing is a dispensary. A product
  entry is a product or a record. The old names are refused wherever they could be read without an error and
  mean nothing (a tolerant file format, a stored point), and reserved in the `.proto`. "The owner" in these
  documents is the person who owns the project, and is not a name for anything in the data.
- **Live and ready are two answers** (D37). The health service's empty name is liveness and is `SERVING` while
  the process runs; `recordsimilarity.v1.RecordSimilarity` is readiness and is `Engine::check`: the embedding
  thread is alive and the store answered its last check. Never make liveness depend on the store (an
  orchestrator would restart a healthy process because Qdrant is down), and never make a request wait on or be
  refused by readiness: it is advice to whoever routes requests. Stopping is in one order: not ready, the
  notice, the listener, the deadline. A new `VectorStore` needs a `check` that is cheap and reads no record.
- **Metrics are few, about the process, and their names are an interface** (D39). They are OpenTelemetry
  instruments in `metrics.rs`, sent to a collector over gRPC when `telemetry.otlp_endpoint` is set, served as a
  Prometheus page when `telemetry.metrics_listen` is, and not kept at all with neither. **The owner prefers
  maintained crates to hand-written infrastructure** (2026-11-12): do not bring back a hand-written counter,
  histogram, or text format, or the `metrics` crate and a bridge to OpenTelemetry, and reach for the
  ecosystem's crate first for the next thing of this kind. **The owner values a request's trace over counts**
  (2026-11-12): a metric is for what no single trace shows (how busy the model is, who is waiting, who was
  turned away, what verdicts go out). Before adding one, ask whether a span attribute says it; do not bring
  back per-call request counts and latencies, store timings, or lock waits as metrics. To add one: a name, a
  field, a line in `Metrics::new` (which also gives it its zero), and a method; call the method from the rest
  of the server, never an instrument. Never make an attribute value out of anything a request carries: no id,
  no brand, no text, no score. Add a metric or an attribute value; never rename one or change what it counts.
  Nothing in a request may wait on, or fail because of, a metric. `Metrics` is made from a meter and knows no
  exporter; exporters are built in `telemetry.rs` and nowhere else. The OpenTelemetry crates move together
  (`opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`, `opentelemetry-prometheus` at one version, with
  the `tracing-opentelemetry` and `tonic-tracing-opentelemetry` releases built for it).
- **A request is a trace, and anything it waits for is a span in it** (D40). The transport's layer makes the
  call's span; the handler makes one for its work (`evaluate`, `delete`, ...) at `debug`; the record's lock,
  each store call (`telemetry::Traced`), and the model (`embed`) are spans inside it. A new wait gets a span,
  or nobody will know it is there; a new call gets a handler span; work moved to another task is given the
  request's span (`in_current_span`). A span may carry a record's id and what came of the request (mode,
  verdict, stored, revision, decision), and never a record's text, its brand, or a score. Record a number as
  `i64` or `f64` (an unsigned one is sent as text). The spans exist only for the trace: keep them at `debug`
  under the `similarity_server` target, which is what the tracing layer's own filter lets through, and never
  make the log lines depend on them. The layer is handed to `similarity-log` through `Init::tracing` before
  logging starts; do not use `tracing_subscriber::reload` for it (it breaks the trace context), and do not
  set up a second subscriber. A store is traced from outside by `Traced`; do not put spans into a store.
  Health checks are not traced. Before proposing a performance change, find out where the time goes; it has
  been guessed wrong before.
- **A record's lock is its own** (D41). `locks.rs` makes a lock when the first request for a record arrives
  and forgets it when the last has left; requests for one record take turns in the order they came, and a
  request never waits for another record's. Do not bring back a fixed number of locks picked by a hash (one
  request in five waited out a stranger's whole request), do not keep a lock for a record nobody is at (the
  table would grow with the catalog), and never hold the table's own mutex across an `.await`. It is one
  process's lock: what holds across replicas is the conditional write (D35).
- **A request can be dropped anywhere it waits, and a write never is** (D38). The transport drops a handler
  when its caller's deadline runs out or the caller hangs up, so every `.await` in a request is a place it can
  end. Before the write that is free, and has to stay free: nothing may be left half done there. The write
  itself goes through `Engine::committed`, which runs the store call, the observer, and the making of the answer
  on a task of their own with the record's lock, and whose result logs itself if nobody takes it. Never call
  `put_if` or `delete_if` on the request's own task, never put anything after the write that the detached task
  does not do, and never log that line for a request that was answered. The embedding threads (one for each of
  `model.sessions`, sharing one backlog under one lock, D46) pass over a job whose reply nobody holds; do not
  make `EmbedWorker::embed` keep working for a caller that is gone (no spawning it, no holding the receiver
  elsewhere). With several sessions, a request left waiting is refused only if no session can be free before
  its deadline: never while one is idle. A deadline reaches the thread through `worker::by`, a
  task-local: work moved to another task does not carry it. The thread judges "cannot be answered in time" on
  the *fastest* rate it has seen, a lower bound on purpose: a request is refused only for a deadline that
  cannot be met, never for one that might not be, a request alone in a pass is never refused on a guess at its
  own texts, and nothing but a deadline that has passed is refused on arrival. Do not replace that rate with an
  average. Requests are served in the order they came until one is lost, and newest first while the thread is
  behind: do not "simplify" that to one order. Always in order serves requests at the edge of their deadlines
  once a queue forms, and always newest first starves callers that are being served fine. None of it governs
  the store: a server short of Qdrant misses deadlines however the model is shared out, and the README says so.
- **Every write the engine makes is conditional on what it read** (D35). `Engine` never calls `VectorStore::put`
  or `delete`: it calls `put_if` with the version it judged against (or `Expected::Absent` for a record it found
  missing) and `delete_if`, and when one comes back `false` it has written nothing and reads and judges again,
  at most `WRITE_ATTEMPTS` times, before answering `Conflict` (`ABORTED`). That is what lets replicas share a
  store and no lock, and what makes "an import only creates" true beside another writer. Do not write
  unconditionally "because the lock is held": the lock is one process's. Do not observe, log an outcome for, or
  count an attempt that lost. A store implementation must pass the conditional half of the contract test, and
  `just test-qdrant` is what proves it against the real thing.
- **Quantities never enter the weighted score or the layout id** (D16). They are compared as numbers
  (`similarity_core::quantity`), stored in the payload beside the vectors, and judged by
  `Evaluation::judge_quantities` after the text is scored. Do not embed one, do not fold one into `similarity`, and
  do not add one to `Schema::layout_id`: the first changes what every threshold means, the second forces a
  re-index for no reason. The anchor rules above apply to them unchanged (first value kept, `Restored` compared
  against it, `FORCE` resets).
- **Quantity limits are measurements, like thresholds.** `config.example.toml` ships every quantity with no
  `reject_above_factor`, no `reject_incomparable`, and `reject_on_measure_change` off. Never add a limit there
  that was not read from `just calibrate` on real labeled pairs, and never from the synthetic sample.
- **The `Quantities` service computes nothing of its own** (D47). `quantities.rs` translates and calls
  `similarity_core::quantity`: `compare` for `Compare`, `Quantity::in_unit` for `Convert`, `units()` for
  `ListUnits`. That is what makes its answers the ones `Evaluate` reports. Do not give it arithmetic, a unit, or
  a rule of its own, do not let it read the schema, a limit, the store, or the model (it must answer while the
  API is not ready, and the same request must always get the same answer), and do not set `beyond_limit` in it.
  Units that do not convert are an answer, not a status code. A new conversion rule goes into `quantity.rs`,
  which is mutation-tested, and a new `Dimension` needs a wire value (a test in `quantities.rs` holds that).
  `Parse` is `similarity_core::amount::parse` for each text and nothing more.
- **An amount is read whole from a text, or refused, and never guessed** (D48). `amount::parse` reads a string that is
  an amount and nothing else: it does not look for one inside prose, and data that is wrong is refused and not
  rescued, by the owner's decision (2026-11-22). Do not add a reading for a decimal comma, a qualifier (`<1 %`), a
  multiple (`10 x 10 mg`), or a word in front, and do not loosen what a unit may be made of. Every text must have one
  reading: the comma rule, the rule for `1-1/2`, and the unit's characters are what make that so, and a change to any
  of them needs the others checked. A record still takes a `Quantity` and never a text (the owner decided,
  2026-11-22): do not accept a string in `quantities`, and keep the refusal in `record.rs`. A weight named by a word
  is read as the weight it is sold as, in grams (`eighth` 3.5 g, `quarter` 7 g, `half ounce` 14 g, `ounce` and `zip`
  28 g: the `NAMED` table, the owner's values, 2026-11-22), and a number and a unit stay exact (`1/8 oz` is 3.54 g):
  do not snap a written ounce to a trade weight, and do not make `oz` 28 g in `UNITS`, which the owner declined. A
  bare `half` is refused (`ambiguous_name`) by the owner's decision. A new word is the owner's company's vocabulary
  (D36) and its value is theirs to give. `ParsedAmount.normalized` is `Quantity::in_base_unit` and nothing else.
  `NotAnAmount::code` and `AmountProblem` are an interface: add, never rename, and a new reason needs a wire value and
  a line in `data/amounts.cases.jsonl` (tests hold both). That file is synthetic and is the specification: a new form
  or refusal is a line there first, and both the parser and the service are held to every line. `amounts survey`
  (`offline/amounts.rs`) counts what a column of texts holds by calling the same `parse`: never give it a reading of
  its own, keep every number of its report and its events a field of `Survey`, and never let it write a text of the
  column unless `--examples` asks (the default is 0 on purpose: the values are the data's owner's). Its `amount_*`
  event names and fields, and the dimension names (`mass`, `volume`, `fraction`, `other`), are an interface like the
  calibration ones. The plan for what is not built (units per something) is
  `docs/plans/33-quantity-text-and-rates.md`.
- **An unknown unit is never an error.** It is its own dimension, comparable only with itself. Extend the `UNITS`
  table in `quantity.rs` when a new unit turns up; do not make the server refuse it.
- **A unit added to the table is a judging change** (D44). `quantity::rules_digest()` covers every spelling with
  its factor, the tolerance, and `RULES_VERSION`, and is the last line of the judging digest's text, so a new row
  changes the digest of every server that takes the build. A test in `quantity.rs` pins the digest of the table
  that ships: when it fails, say in the change that the digest moves, then write the new value. Raise
  `RULES_VERSION` for a change to how amounts are compared that no row shows (how a unit is normalized, how a
  change is told from none). The lines are sorted by spelling, so moving a row changes nothing; keep it so. The
  `NAMED` words of `amount.rs` are not in it: reading decides nothing.
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
  Never give it an update path; an update is an `Evaluate`, which judges. The streaming `Import` call is the same
  function behind a wire (D33) and takes the same rule: do not add a mode that replaces, and do not let one bad
  record end a stream. A record that is already stored must not be embedded; the early look for it is outside
  the lock and is only a saving, and whether a record is created is still decided under the lock.
- **A stream shares the model with every other request, a small batch at a time** (D33). `server.import_batch_records`
  trades a stream's speed against how long every other request waits; the README's table was measured with
  `just load`. Do not make the stream wait for a batch to fill, and do not buffer more than one batch on either
  side of the loop in `import_stream.rs`: back-pressure on the sender is the transport's job.
- **An audit reads, and vouches only for what it compared** (D42). `audit` is written against `Scan`, which has
  no write, and opens its collection with `QdrantStore::open`, never `connect` (which creates a collection and
  its indexes). Do not give it a repair mode or a write path: a record the store lacks is created by `import`,
  and replacing a baseline is a forced write that is the caller's to decide. Keep `Scan` a trait of its own and
  not a method of `VectorStore`: the engine never scans. A scan lists every point whatever its layout, names
  what it cannot read instead of failing, and reads no vector but an anchor it is asked for (D43); a store's `scan` must pass the `scanning` half of
  the contract test. A record is `not_stored` only once the whole store has been listed: never conclude anything
  from part of a scan. What a valid line of an export is lives in `import::prepared`, for `import --check` and
  the audit alike, and the line compared is the first one an import would have taken. Text is compared by
  fingerprint (`PreparedRecord::hashes`, which `Plan::finish` stores) and quantities by `quantity::compare`: do
  not compare either another way. The anchor has no fingerprints and is not audited; the report says so, and
  nothing may approximate it. Every count comes from `audit::Summary`, and everything looked at ends as exactly
  one of agree, differ, not stored, not in the export, not compared (a property holds that). The `audit_*`
  event names and fields, and the codes of `Kind` and `Difference`, are an interface like the calibration ones.
  A line carries ids and brands, never a record's text or a fingerprint of it.
- **A restore gives an anchor only to a record with nothing to lose** (D43). `Engine::restore_anchor` is the one
  write besides a forced one that changes an anchor, and `engine::restorable` is its whole rule: the record is
  stored under the backup's brand, has never been written since it was created (revision 1), is its own anchor,
  and is fingerprint for fingerprint and amount for amount what the backup was taken of. Do not widen it: a
  record that changed since the backup, has a history, or was reset by a forced write is left alone and named,
  and whether a `changed` record may ever get its kept anchor is the owner's decision (D43 says what the
  choices are), not a default to pick. It is a write like any other: conditional on the version read, through
  `committed`, counted as a revision. `SAME_VECTOR` is a tolerance for telling a vector from itself after
  Qdrant has kept it at unit length, not a similarity threshold: do not tune it, and do not use it to judge
  anything. A backup belongs to its layout: `anchors::restore` reads the whole file first and refuses, having
  written nothing, one that is cut short (no last line) or of another layout. Keep vectors in the file as
  bytes in base64, never as numbers. `anchors export` opens the collection with `QdrantStore::open`, writes
  under a `.partial` name and then links it to its own, and never writes over a file (a rename would). A backup holds no text and must not gain
  any; a scan reads an anchor only when asked (`Read::Anchor`) and never a current composite. The file's line
  names and fields, and the codes of `Restored`, are an interface: add, never rename.
- **The judging digest covers everything a verdict depends on, and its text is an interface** (D44).
  `Description::judging` writes the rules out one to a line and `judging_digest` is its hash: what `Describe`
  and the `serving gRPC` line give so that replicas can be seen to agree. Servers of different versions compare
  digests, so add a line for a new rule and never reword, reorder, or drop one. It takes `Policy`, `Matching`,
  `FieldSpec`, and `QuantitySpec` apart by name with nothing left over on purpose: do not add `..` to make a new
  field compile, write the field out. Numbers go in as their bits, never formatted. Nothing that is not a
  rule of judging belongs in it (an address, a batch size, the shadow). Its last line, `units`, is the one rule
  no configuration holds: the digest of the unit table the build compares amounts by.
- **A brand, once stored, must be named by every later request for that id**, including `Delete` (D14).
- **What a supported model directory is lives in one place**, `similarity_core::model::Pipeline`. The loader and
  `model fetch` both use it; never parse `modules.json` anywhere else.
- **Burn 0.22 is not the Burn in older tutorials**: models are not generic over a backend, a runtime `Device`
  selects it, `ndarray` is deprecated in favor of `flex`, and records/recorders were replaced by `burn-store`.
  Check the crate source under `~/.cargo/registry` rather than trusting memory.
- If the model or its revision changes: update the pinned revision and checksums in `fetch.rs`, regenerate the
  parity fixture (`just golden`), export it again (`just export-onnx`), re-calibrate, and update
  `models/README.md`.
- **`output_dim` has no default on purpose** (D4): truncation is only valid for Matryoshka-trained models. When
  comparing models with `calibrate`, check the logged `dimension=` to confirm each ran at the width you meant.
- **Every command and number in `README.md` was run and measured.** Keep it that way: re-run an example before
  editing it, and never quote a figure that was not produced by `just throughput`, `just load`, or `just calibrate`.
- **Rust first, by the owner's choice** (D13, revised 2026-11-12 in D34): serving, training, and evaluation are Rust,
  and nothing at run time may need Python. Python is allowed where it is the best tool for a job, and not as a
  default: say why in the change. `tools/golden.py` and `tools/train_golden.py` write PyTorch fixtures for the
  parity tests; `tools/export_onnx.py` converts a model directory for the ONNX engine; `tools/demo.py` is a
  client that walks an audience through the API on a synthetic catalog (`just demo`), and prints what the server
  answers without asserting any of it.
- **Burn defines the model; ONNX Runtime computes it** (D34). Every command of `record-similarity` that embeds
  loads the directory's weights in Burn and runs the forward pass in C++ from `onnx/model.onnx`. There is no
  setting for it and no Burn serving path: the owner dropped that on 2026-11-12, and `model.engine`,
  `model.device`, and the server's `metal` and `onnx` features went with it. Do not bring a choice back. Never
  let the exported file be used unchecked: the loader compares it with the directory's weights in Burn at every
  load and refuses a graph that differs, because the file goes stale silently when the weights change. Never put
  the engine into the fingerprint or the layout id: the vectors are the same, and that is what the check is for.
  `onnx` stays a feature of `similarity-core` (the `ort` crate is behind it) that the server always enables and the
  trainer never does, with `ort` pinned
  exactly. Its threads spin, always, and `model.threads` (left out by default) says how many there are (D45):
  neither changes a vector, so neither goes into the fingerprint, the layout id, or the judging digest. The tests
  stay on Burn, the reference: the tiny encoders, and the real-model tests, which load with
  `Embedder::load`. After changing the encoder, the pooling, or the export, run `just test-onnx` (it needs
  `just export-onnx` first), which is what holds the two engines to each other.
- **A model directory is not usable until it is exported.** `just export-onnx <dir>` after `model fetch`, after a
  fine-tune, and for any checkpoint that is to be calibrated. It is the one step that needs Python, and nothing
  at run time does.
- **Histories are measured first and judged afterwards** (D27). `calibrate-drift` replays every version on an
  engine that can reject nothing, then judges each step under the config. Never replay under the configured
  policy: after the first rejection the baseline stops moving and the rest of the history is fiction.
  `anchor_threshold = 0.85` in `config.example.toml` is a starting point, not a measurement; do not replace it
  with a number from `data/drift.sample.jsonl`, which is synthetic.
- **A match is a read, scored exactly** (D32). `Match` never writes, a menu item is mapped onto the product schema
  and never gets a layout of its own, and a menu item that carries an `id` or a `brand_id` is refused. The engine and
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
  event names are an interface like the calibration ones, and a line carries an id, a brand, scores, and codes:
  never a record's text. `config.example.toml` ships no shadow and no number for one. A shadow measures
  agreement, not quality: do not add anything that ranks or adopts a candidate by its disagreements.
  `shadow-report` takes totals from the `shadow_summary` lines and everything else from the `shadow` lines, and
  never mixes the two.
- **A sweep is a search, and what it finds is fitted** (D29). `calibrate-sweep` ranks on one calibration and may
  evaluate the best points on another at the threshold already chosen. Never report a swept point's errors on
  the run it was ranked on as if they were a measurement, and never let it choose a threshold on the evaluation
  run.
- **The holdout chooses nothing** (D24). A checkpoint, a threshold, or a setting is chosen on the `validation`
  split (`record-similarity pairs carve` makes one), and the holdout is measured once, for what was chosen. Do
  not add a code path, a recipe, or a sentence in the README that picks anything by its holdout numbers. Models
  are compared by AUC (`similarity_core::separation`), never by errors at a configured threshold, which belongs
  to the model it was measured on. Catalog groups of a brand in the validation or holdout split are not trained
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

- Build: `just build` (`cargo build --workspace`); release binary: `just release`. The first build of the server
  downloads ONNX Runtime's prebuilt library; on Linux it links only against GCC 13's C++ library or later
- Test (fast): `just test`, or one crate: `just test similarity-core`
- Lint: `just clippy`; format: `just fmt`; everything CI runs: `just ci`
- Model: `just fetch-model`, then `just export-onnx` (needs `uv`); Qdrant: `just qdrant-up` / `just qdrant-down`
- Slow tests: `just test-parity`, `just test-snapshot`, `just test-qdrant`, `just test-real`; `just snapshot` rewrites
  the pinned verdicts
- The engine: `just export-onnx <model-dir>` (needs `uv`), `just test-onnx`, `just throughput <model-dir>` and
  `just throughput <model-dir> burn` for the reference beside it; `just image <tag>` exports inside the build
  (Debian 13: the prebuilt runtime does not link on Debian 12)
- Mutation testing, in extreme cases only (40 minutes; needs `cargo install cargo-mutants`): `just mutants -j 3`,
  `just mutants-diff`, `just mutants-train`
- Run: `just serve` (needs Qdrant) or `just serve-in-memory`; then `just client describe`
- Watch its traces and metrics: `just observe-up` (an OpenTelemetry Collector on 4317, Jaeger on 16686, and
  Prometheus on 9090, in Docker; the server needs `[telemetry]` set), `just observe-down`
- Show it to an audience: `just serve-in-memory config.toml`, then `just demo --pause` (needs `uv`)
- Load a catalog: `just import export.jsonl config.toml` (`--check` validates only)
- Check the store against the catalog: `just audit export.jsonl config.toml` (`--ndjson` for every finding as an
  event, `--exported-at SECONDS` to set aside what was written since, `--fail-on-differences` for an exit status)
- Back up the anchors, and put them back after a collection was lost and imported again:
  `just anchors export --config config.toml --out anchors.ndjson`, `just anchors check anchors.ndjson`,
  `just anchors restore --config config.toml anchors.ndjson` (`--dry-run` writes nothing)
- Measure: `just throughput`, `just load`, `just calibrate`, `just calibrate-drift`, `just calibrate-match`, `just calibrate-sweep`, `just calibrate-diff`, `just verdict-gate old.ndjson pairs.full.jsonl`
- Read a shadowed server's stdout: `just shadow-report shadow.ndjson --ids to-label.jsonl`
- Count what a column of amounts written as text holds: `just amounts survey column.txt` (`--ndjson` for events,
  `--examples N` to also see refused texts, `-` for standard input)
- Put a model into service: `just promote --config candidate.toml --calibration holdout.ndjson`; then
  `record-similarity model list`, `model register <dir>`, `model retire <digest> --reason "..."`
- See what was run: `just runs` (`calibrate` and the trainer each leave a directory under `runs/`; `--no-run`
  turns that off); `record-similarity runs compare A B`; `record-similarity runs report A B --out report.html`
- Make a validation split: `record-similarity pairs carve pairs.full.jsonl --fraction F > pairs.carved.jsonl`
- Fine-tune: `just finetune --base ... --out ... --pairs ...` (CPU); GPU: `just release-train metal`, then
  `./target/release/record-similarity-train --device metal ...`. Continue an interrupted run with `--resume
  <out>.checkpoints/step-NNNNNN` in place of `--base`. Results are NDJSON on stdout, logs NDJSON on stderr.
- PyTorch referees (need `uv`): `just golden` regenerates both fixtures; `just verify-model <dir>` is optional
