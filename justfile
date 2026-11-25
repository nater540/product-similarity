# Project actions. `just` with no arguments lists them. https://just.systems

set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments := false

# Where the ignored Qdrant tests look for a server.
qdrant_url := env_var_or_default("QDRANT_URL", "http://localhost:6334")

# Where ONNX Runtime's library is: every recipe that runs the model opens the one `just fetch-onnxruntime` put
# here, unless the variable already names another. The binary run by hand needs the variable, or the library
# beside it.
export ORT_DYLIB_PATH := env_var_or_default("ORT_DYLIB_PATH", justfile_directory() / "onnxruntime" / if os() == "macos" { "libonnxruntime.dylib" } else { "libonnxruntime.so" })

default:
    @just --list --unsorted

# --- build ------------------------------------------------------------------

# Debug build of the whole workspace
build:
    cargo build --workspace

# Release build of the server binary (ONNX Runtime is not in it: `just fetch-onnxruntime` gets the library it opens)
release:
    cargo build --release -p similarity-server

# Write `onnx/model.onnx` into a model directory: the file the model is run from (needs `uv`; again after fine-tuning)
export-onnx model="models/mdbr-leaf-mt":
    uv run tools/export_onnx.py {{ model }}

# Release build of the fine-tuning binary; `just release-train metal` or `cuda` adds that GPU backend
release-train backend="":
    cargo build --release -p similarity-train {{ if backend == "" { "" } else { "--features " + backend } }}

# Container image with the model baked in, exported for ONNX Runtime in a build stage of its own
image tag="record-similarity:dev":
    docker build --build-arg RECORD_SIMILARITY_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)" -t {{ tag }} .

# --- test and lint ----------------------------------------------------------

# Run every fast test (no model, no network, no Qdrant), or one crate's: `just test similarity-core`
test crate="":
    {{ if crate == "" { "cargo test --workspace" } else { "cargo test -p " + crate } }}

# Tests that need the downloaded model: parity with PyTorch, the pinned verdicts, and the full stack against Qdrant
test-real: test-parity test-snapshot
    QDRANT_URL={{ qdrant_url }} cargo test -p similarity-server --test real_e2e -- --ignored

# The Burn port against PyTorch on the real model: the embeddings, then the training steps (needs `just fetch-model`)
test-parity:
    cargo test -p similarity-core --test real_parity -- --ignored
    cargo test -p similarity-train --test real_parity -- --ignored

# The ONNX Runtime engine against PyTorch and against Burn, on the real model (needs `just export-onnx`, `just fetch-onnxruntime`)
test-onnx:
    cargo test -p similarity-core --features onnx --test real_onnx -- --ignored

# The curated hard cases against the committed verdict snapshot, and the README's quoted calibration reports
# against what `calibrate` prints (needs `just fetch-model`)
test-snapshot:
    cargo test -p similarity-server --test real_snapshot -- --ignored

# Rewrite the verdict snapshot after an intended change, printing what moved (needs `just fetch-model`)
snapshot:
    UPDATE_SNAPSHOT=1 cargo test -p similarity-server --test real_snapshot the_pinned_model -- --ignored --nocapture

# Gate a calibration of labeled pairs against an earlier `--ndjson` run of the same pairs: fails on a flipped
# verdict or a score that moved. For real pairs, where the data is: `just verdict-gate kept.ndjson pairs.full.jsonl`
verdict-gate old pairs config="config.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- calibrate --config {{ config }} {{ pairs }} --full --ndjson {{ flags }} > target/verdict-gate.ndjson
    cargo run --release -q -p similarity-server -- calibrate-diff {{ old }} target/verdict-gate.ndjson --fail-on any-flip --fail-on moved

# For a mutation run: one fixed proptest seed, so that a mutant is caught or missed the same way every time, and no
# regression files written into the copied tree. The commit is given so that the build script does not ask `git`
# in a copy that has none.
mutants_env := "PROPTEST_RNG_SEED=1 PROPTEST_DISABLE_FAILURE_PERSISTENCE=1 RECORD_SIMILARITY_COMMIT=mutants"

# Mutation testing of the files whose boundaries are verdicts. About 40 minutes: for extreme cases only, D25
# (needs `cargo install cargo-mutants`)
mutants *flags:
    {{ mutants_env }} cargo mutants --output target {{ flags }}

# Only the mutants in what changed since `base`: `just mutants-diff` (uncommitted), `just mutants-diff main`
mutants-diff base="HEAD":
    git diff --no-ext-diff {{ base }} > target/mutants.diff
    {{ mutants_env }} cargo mutants --output target --in-diff target/mutants.diff

# Mutation testing of the trainer's checkpoint rules
mutants-train *flags:
    {{ mutants_env }} cargo mutants --config .cargo/mutants-train.toml --output target/mutants-train {{ flags }}

# The storage contract against a live Qdrant (needs `just qdrant-up`)
test-qdrant:
    QDRANT_URL={{ qdrant_url }} cargo test -p similarity-store --test contract -- --ignored

# Clippy at the zero-warning policy
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Clippy on the trainer with the Metal backend compiled in (macOS only; a long first build)
clippy-metal:
    cargo clippy -p similarity-train --features metal --all-targets -- -D warnings

# Format the workspace
fmt:
    cargo fmt --all

# Fail if anything is not formatted
fmt-check:
    cargo fmt --all -- --check

# License and advisory gate
deny:
    cargo deny check

# Everything CI checks, cheapest first
ci: fmt-check clippy test deny

# --- model ------------------------------------------------------------------

# Download the default model into ./models, verifying checksums
fetch-model:
    cargo run --release -p similarity-server -- model fetch

# Download Microsoft's build of ONNX Runtime, which runs the model, into ./onnxruntime, verifying checksums
fetch-onnxruntime:
    cargo run --release -p similarity-server -- runtime fetch

# Regenerate both PyTorch parity fixtures (needs `uv`; downloads PyTorch on first run)
golden:
    uv run tools/golden.py models/mdbr-leaf-mt crates/similarity-core/tests/fixtures/golden_mdbr-leaf-mt.json
    uv run tools/train_golden.py models/mdbr-leaf-mt crates/similarity-train/tests/fixtures/training_mdbr-leaf-mt.json

# Fine-tune the model: `just finetune --base models/mdbr-leaf-mt --out models/my-model --pairs pairs.full.jsonl`
finetune *args:
    @cargo run --release -q -p similarity-train -- {{ args }}

# Optional: check that PyTorch reads a model directory and agrees with the Burn encoder on it (needs `uv`)
verify-model dir:
    uv run tools/golden.py {{ dir }} target/parity-fixture.json
    PARITY_MODEL_DIR={{ dir }} PARITY_FIXTURE=target/parity-fixture.json cargo test -p similarity-core --test real_parity -- --ignored

# Forward-pass latency on this machine's CPU
throughput model="models/mdbr-leaf-mt" engine="onnx" threads="":
    cargo run --release -p similarity-core {{ if engine == "onnx" { "--features onnx" } else { "" } }} --example throughput -- {{ model }} {{ engine }} {{ threads }}

# Score the labeled pairs and report the threshold that separates edits from recycled records.
# Flags go last: `just calibrate data/calibration.sample.full.jsonl config.example.toml --full --split holdout`
calibrate pairs="data/calibration.sample.jsonl" config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- calibrate --config {{ config }} {{ pairs }} {{ flags }}

# Compare two `calibrate --ndjson` runs, or one run under a second config: `just calibrate-diff old.ndjson --config-b b.toml`
calibrate-diff *args:
    cargo run --release -q -p similarity-server -- calibrate-diff {{ args }}

# Put a model into service if its holdout calibration passes the gate: `just promote --config c.toml --calibration holdout.ndjson`
promote *args:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- model promote {{ args }}

# Replay whole edit histories to see how far products drift and where `anchor_threshold` belongs:
# `just calibrate-drift sequences.jsonl config.toml`
calibrate-drift histories="data/drift.sample.jsonl" config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- calibrate-drift --config {{ config }} {{ histories }} {{ flags }}

# Score catalog groups as `Match` would, to see how often the right product comes first and where
# `matching.threshold` belongs: `just calibrate-match catalog-groups.jsonl config.toml --split holdout`
calibrate-match groups="data/catalog-groups.sample.jsonl" config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- calibrate-match --config {{ config }} {{ groups }} {{ flags }}

# Say what a server with a `[shadow]` table wrote to its stdout: `just shadow-report shadow.ndjson --ids to-label.jsonl`
shadow-report *args:
    cargo run --release -q -p similarity-server -- shadow-report {{ args }}

# Judge one recorded calibration under a grid of weights, floors, and limits, with no model:
# `just calibrate-sweep select.ndjson --grid grid.toml --config config.toml [--evaluate other.ndjson]`
calibrate-sweep *args:
    cargo run --release -q -p similarity-server -- calibrate-sweep {{ args }}

# What was recorded: every calibration and every training run, newest first (`--kind train`, `--limit 5`)
runs *flags:
    cargo run --release -q -p similarity-server -- runs list {{ flags }}

# Latency and throughput through the API at several levels of concurrency, on synthetic records and a store in
# memory: `just load config.example.toml --concurrency 1,4,16,64 --records 300`. Results are NDJSON on stdout
load config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- load --config {{ config }} {{ flags }}

# --- run --------------------------------------------------------------------

# Start a local Qdrant (dashboard at http://localhost:6333/dashboard)
qdrant-up:
    docker compose up -d qdrant

# Stop the local Qdrant (its data stays in ./qdrant_storage)
qdrant-down:
    docker compose down

# Start an OpenTelemetry Collector (OTLP gRPC on 4317), Jaeger (traces, http://localhost:16686), and Prometheus
# (metrics, http://localhost:9090) to watch a server that has `[telemetry]` set:
# `otlp_endpoint = "http://127.0.0.1:4317"` and `metrics_listen = "127.0.0.1:9464"`
observe-up:
    docker compose --profile observability up -d otel-collector jaeger prometheus

# Stop them. Nothing is kept: the traces and Prometheus's data go with their containers
observe-down:
    docker compose --profile observability rm -sf otel-collector jaeger prometheus

# Run the server against the local Qdrant
serve config="config.example.toml":
    cargo run --release -p similarity-server -- serve --config {{ config }}

# Load an export into the configured collection: `just import export.jsonl config.toml` (add `--check` to only validate)
import export config="config.example.toml" *flags:
    cargo run --release -q -p similarity-server -- import --config {{ config }} {{ export }} {{ flags }}

# Check the configured collection against an export: `just audit export.jsonl config.toml` (`--ndjson` for every
# finding as an event, `--fail-on-differences` for an exit code, `--exported-at $(date +%s)` taken with the export)
audit export config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- audit --config {{ config }} {{ export }} {{ flags }}

# Keep a copy of every record's anchor, or put anchors back after a collection was lost:
# `just anchors export --config config.toml --out anchors.ndjson`, `just anchors check anchors.ndjson`,
# `just anchors restore --config config.toml anchors.ndjson --dry-run`
anchors *args:
    cargo run --release -q -p similarity-server -- anchors {{ args }}

# Count what a column of amounts written as text holds, one text to a line: `just amounts survey weights.txt`
# (`--ndjson` for events, `--examples 5` to also see refused texts, `-` for standard input)
amounts *args:
    cargo run --release -q -p similarity-server -- amounts {{ args }}

# Run the server with no Qdrant at all; nothing is persisted
serve-in-memory config="config.example.toml":
    cargo run --release -p similarity-server -- serve --config {{ config }} --in-memory

# A walk through the API for an audience, against a running server (needs `uv`): `just serve-in-memory config.toml`
# in one terminal, `just demo --pause` in another. It loads 300 synthetic products and sends real requests about them
demo *args:
    uv run tools/demo.py {{ args }}

# Talk to a running server: `just client evaluate sku-1 data/examples/gummies.json`
client *args:
    cargo run --release -q -p similarity-server --example client -- {{ args }}
