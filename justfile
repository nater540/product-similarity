# Project actions. `just` with no arguments lists them. https://just.systems

set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments := false

# Where the ignored Qdrant tests look for a server.
qdrant_url := env_var_or_default("QDRANT_URL", "http://localhost:6334")

default:
    @just --list --unsorted

# --- build ------------------------------------------------------------------

# Debug build of the whole workspace
build:
    cargo build --workspace

# Release build of the server binary
release:
    cargo build --release -p similarity-server

# Release build that can also run the encoder on the GPU through Metal (Apple silicon)
release-metal:
    cargo build --release -p similarity-server --features metal

# Release build of the fine-tuning binary; `just release-train metal` or `cuda` adds that GPU backend
release-train backend="":
    cargo build --release -p similarity-train {{ if backend == "" { "" } else { "--features " + backend } }}

# Container image with the model baked in
image tag="record-similarity:dev":
    docker build -t {{ tag }} .

# --- test and lint ----------------------------------------------------------

# Run every fast test (no model, no network, no Qdrant), or one crate's: `just test similarity-core`
test crate="":
    {{ if crate == "" { "cargo test --workspace" } else { "cargo test -p " + crate } }}

# Tests that need the downloaded model: parity with PyTorch, and the full stack against Qdrant
test-real: test-parity
    QDRANT_URL={{ qdrant_url }} cargo test -p similarity-server --test real_e2e -- --ignored

# The Burn port against PyTorch on the real model: the embeddings, then the training steps (needs `just fetch-model`)
test-parity:
    cargo test -p similarity-core --test real_parity -- --ignored
    cargo test -p similarity-train --test real_parity -- --ignored

# The storage contract against a live Qdrant (needs `just qdrant-up`)
test-qdrant:
    QDRANT_URL={{ qdrant_url }} cargo test -p similarity-store --test contract -- --ignored

# Clippy at the zero-warning policy
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Clippy with the Metal backend compiled in (macOS only; a long first build)
clippy-metal:
    cargo clippy -p similarity-server -p similarity-train --features metal --all-targets -- -D warnings

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
throughput model="models/mdbr-leaf-mt":
    cargo run --release -p similarity-core --example throughput -- {{ model }}

# Score the labeled pairs and report the threshold that separates edits from recycled records.
# Flags go last: `just calibrate data/calibration.sample.full.jsonl config.example.toml --full --split holdout`
calibrate pairs="data/calibration.sample.jsonl" config="config.example.toml" *flags:
    RUST_LOG=warn cargo run --release -q -p similarity-server -- calibrate --config {{ config }} {{ pairs }} {{ flags }}

# --- run --------------------------------------------------------------------

# Start a local Qdrant (dashboard at http://localhost:6333/dashboard)
qdrant-up:
    docker compose up -d qdrant

# Stop the local Qdrant (its data stays in ./qdrant_storage)
qdrant-down:
    docker compose down

# Run the server against the local Qdrant
serve config="config.example.toml":
    cargo run --release -p similarity-server -- serve --config {{ config }}

# Run the server with no Qdrant at all; nothing is persisted
serve-in-memory config="config.example.toml":
    cargo run --release -p similarity-server -- serve --config {{ config }} --in-memory

# Talk to a running server: `just client evaluate sku-1 data/examples/gummies.json`
client *args:
    cargo run --release -q -p similarity-server --example client -- {{ args }}
