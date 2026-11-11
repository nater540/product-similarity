# record-similarity

A gRPC service that answers one question about an edit to a record: **is this still the same record?**

Marketplace listings get "recycled": a merchant takes an existing product, say a blueberry gummy, and edits it into
an entirely different one, keeping the id and everything attached to it (reviews, rankings, history). This service
embeds each revision of a record field by field with a small sentence-embedding model running on
[Burn](https://burn.dev), compares it with the stored version, and accepts or rejects the change. Accepted vectors
are kept in [Qdrant](https://qdrant.tech), which also makes the catalog searchable by similarity.

```text
$ just client evaluate sku-1 data/examples/gummies.json
verdict   Created (stored: true, revision 1)

$ just client evaluate sku-1 data/examples/gummies-edited.json      # name suffix, reworded description
verdict   Accepted (stored: true, revision 2)
overall   0.982 against a threshold of 0.900
drift     0.982 similarity to the original record
field     brand           1.000  Unchanged
field     category        1.000  Unchanged
field     description     0.981  Modified
field     name            0.963  Modified

$ just client evaluate sku-1 data/examples/vape.json                # a different product under the same id
verdict   Rejected (stored: false, revision 2)
overall   0.562 against a threshold of 0.900
drift     0.567 similarity to the original record
field     brand           0.642  Modified
field     category        0.559  Modified
field     description     0.635  Modified
field     name            0.509  Modified  << below floor
reason    overall similarity 0.562 is below the threshold 0.900
reason    field `name` similarity 0.509 is below its floor 0.800
reason    similarity to the original record 0.567 is below the drift threshold 0.850
```

## Contents

- [Status](#status)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Running the server](#running-the-server)
- [Making requests](#making-requests)
- [Integrating it](#integrating-it)
- [How a verdict is reached](#how-a-verdict-is-reached)
- [Choosing thresholds](#choosing-thresholds)
- [Fine-tuning the model](#fine-tuning-the-model)
- [Using a different model](#using-a-different-model)
- [Configuration reference](#configuration-reference)
- [Operations](#operations)
- [Performance](#performance)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Status

Working end to end and tested at every layer, including a full-stack test against a real Qdrant with the real model.
Three things stand between this and enforcing on production traffic:

1. **Thresholds are calibrated on synthetic data.** `data/calibration.sample.jsonl` holds 46 pairs written for this
   repository. Collect labeled edits from production (`docs/data-request-prompt.md` describes exactly what to
   collect), run `calibrate` on them, and set the thresholds from that. Start in `DRY_RUN`.
2. **The choice of model is a prior, not a measured win.** On the synthetic pairs the default model and
   `all-MiniLM-L6-v2` are close. Real pairs should decide ([Using a different model](#using-a-different-model)).
3. **Per-record locking is per process.** See [Limitations](#limitations).

## Dependencies

To build and run:

| Tool | Why | macOS | Debian / Ubuntu |
|---|---|---|---|
| Rust 1.96 | The toolchain. `rust-toolchain.toml` pins it, so `rustup` installs the right one on first build. | [rustup.rs](https://rustup.rs) | [rustup.rs](https://rustup.rs) |
| A C toolchain | Some dependencies compile C. | Xcode command line tools | `apt-get install build-essential` |
| `protoc` | Compiles the gRPC definitions at build time. | `brew install protobuf` | `apt-get install protobuf-compiler` |
| Docker with Compose | Runs Qdrant locally. Not needed with `--in-memory`. | Docker Desktop or OrbStack | Docker Engine |

Optional, by task:

| Tool | Needed for | Install |
|---|---|---|
| [`just`](https://just.systems) | The short commands used throughout this README. Every recipe is a plain `cargo`, `docker`, or `uv` command you can read in the `justfile` and run by hand. | `brew install just` or `cargo install just` |
| [`uv`](https://docs.astral.sh/uv/) | Only for regenerating the PyTorch reference fixtures (`just golden`) and the optional `just verify-model`. Building, testing, serving, and fine-tuning are all Rust and need no Python. It installs PyTorch itself on first use, a download of several hundred megabytes or more. | `brew install uv` |
| `cargo-deny` | `just deny` and `just ci`. | `cargo install cargo-deny --locked` |
| `grpcurl` | Calling the server from a shell. | `brew install grpcurl` |
| `jq` | One step of the fine-tuning workflow. | `brew install jq` |

Disk and hardware: the model is about 100 MB, and Rust build artifacts for this workspace run to several gigabytes.
No GPU is needed to serve. Fine-tuning on long texts is much faster with one
([Fine-tuning the model](#fine-tuning-the-model)). The first release build takes a few minutes; Burn is a large
dependency.

## Setup

```sh
git clone <this repository> && cd record-similarity

just fetch-model          # 1. download the model into ./models and verify its checksums
just qdrant-up            # 2. start Qdrant in Docker (dashboard: http://localhost:6333/dashboard)
cp config.example.toml config.toml     # 3. your own copy of the configuration
just serve config.toml    # 4. build and run the server on 127.0.0.1:50051
```

What each step does:

1. **Model.** `record-similarity model fetch` downloads `MongoDB/mdbr-leaf-mt` at a pinned revision from the
   Hugging Face Hub, checks the SHA-256 of every file against values compiled into the binary, and then loads the
   model once to prove it runs. Re-running it only downloads what is missing. This is the only command that touches
   the network; the server itself only ever reads a directory. See [`models/README.md`](models/README.md).
2. **Qdrant.** `docker compose up -d qdrant` starts Qdrant 1.20 with its data in `./qdrant_storage`. The server
   creates its collection on first start.
3. **Configuration.** The example is commented line by line. For a first run nothing needs changing.
4. **Server.** On start-up it loads the model, runs one warm-up embedding, connects to Qdrant, verifies or creates
   the collection, and logs `serving gRPC`.

Check that it works, from another terminal:

```sh
just client describe
just client evaluate sku-1 data/examples/gummies.json --tenant merchant-1
just client evaluate sku-1 data/examples/gummies-edited.json --tenant merchant-1
just client evaluate sku-1 data/examples/vape.json --tenant merchant-1
just client similar sku-1
```

Then run the test suites that use the real model and the real Qdrant:

```sh
just test-real            # parity with PyTorch, then the whole stack through gRPC
```

Without `just`, the same setup is:

```sh
cargo run --release -p similarity-server -- model fetch
docker compose up -d qdrant
cargo run --release -p similarity-server -- serve --config config.toml
```

## Running the server

The binary is `record-similarity` (`target/release/record-similarity` after `just release`).

| Command | What it does |
|---|---|
| `record-similarity serve [--config config.toml] [--listen ADDR] [--in-memory]` | Runs the gRPC server. |
| `record-similarity model fetch [--repo R] [--revision REV] [--dir DIR]` | Downloads and verifies a model directory. `--dir` defaults to `models/<repository name>`, so two models never share a directory. |
| `record-similarity compare [--config config.toml] BEFORE.json AFTER.json` | Scores one before/after pair offline. |
| `record-similarity calibrate [--config config.toml] PAIRS.jsonl` | Scores labeled pairs and suggests a threshold. |

Environment variables:

| Variable | Effect |
|---|---|
| `QDRANT_API_KEY` | The API key for a secured Qdrant or Qdrant Cloud. Never read from the config file. |
| `RUST_LOG` | Log filter, `info` by default. `RUST_LOG=warn` quiets the per-request lines; `RUST_LOG=debug` adds detail. Logs go to stderr. |

Ways to run it:

- **Against Qdrant** (normal): `just serve config.toml`.
- **With no Qdrant at all**: `just serve-in-memory`. Vectors live in process memory and are lost on exit. Useful
  for trying the API and for demos.
- **As a container**: `docker compose --profile server up --build` builds the image (the model is downloaded and
  verified at build time and baked in) and runs it next to Qdrant, publishing `127.0.0.1:50051`. `just image`
  builds the image alone. The container runs as a non-root user and reads `/etc/record-similarity/config.toml`;
  mount your own file there to override the defaults.
- **On the GPU on a Mac**: `just release-metal`, then set `device = "metal"` in the config. This is a development
  convenience; a model this small is already fast on a CPU.

The server stops cleanly on Ctrl-C or `SIGTERM`, finishing in-flight requests first.

## Making requests

The API is defined, with full documentation of every field, in
[`similarity.proto`](crates/similarity-proto/proto/recordsimilarity/v1/similarity.proto). The package is
`recordsimilarity.v1` and the service is `RecordSimilarity`.

| RPC | What it does |
|---|---|
| `Evaluate` | Embeds a record. With an `id`, compares it with the stored baseline, returns a verdict with per-field detail, and stores it if allowed. |
| `FindSimilar` | Returns the stored records most similar to a stored id or to an ad hoc record, optionally within one tenant. |
| `Delete` | Forgets a record. |
| `Describe` | Reports the model fingerprint, schema, and thresholds in force. |

A record is an optional `id`, an optional `tenant`, and a map of field name to text. The field names must be ones
the server's schema defines; `Describe` lists them.

**Tenants.** Once a record is stored under a tenant, every later `Evaluate` or `Delete` for its id must name the
same tenant. A request naming another tenant, or none, gets `FAILED_PRECONDITION`. A record stored without a
tenant takes the tenant of the first revision that names one. This catches a caller applying one merchant's edit
to another merchant's record. It is a consistency check, not access control: the server does not authenticate
its callers. If you use tenants at all, send one with every request.

### With the bundled client

The quickest way to poke a server. It is also a small, complete example of a Rust client
([`examples/client.rs`](crates/similarity-server/examples/client.rs)).

```sh
just client describe
just client evaluate <id> <fields.json> [--tenant T] [--dry-run | --force]
just client similar <id> [--limit N]
just client delete <id> [--tenant T]
just client --server http://other-host:50051 describe
```

`fields.json` is a JSON object of field name to text, like the files in `data/examples/`.

### With grpcurl

The server exposes gRPC reflection, so no `.proto` file is needed on the command line.

```sh
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext localhost:50051 describe recordsimilarity.v1.RecordSimilarity
grpcurl -plaintext localhost:50051 recordsimilarity.v1.RecordSimilarity/Describe
```

Create a record (the first `Evaluate` for an id stores it as the baseline):

```sh
grpcurl -plaintext -d '{
  "record": {
    "id": "sku-1",
    "tenant": "merchant-1",
    "fields": {
      "name": "Blueberry Gummies 100mg",
      "category": "Edibles > Gummies",
      "brand": "WYLD",
      "description": "Ten vegan gummies made with real fruit. 10mg THC per piece, 100mg per package."
    }
  }
}' localhost:50051 recordsimilarity.v1.RecordSimilarity/Evaluate
```

```json
{
  "verdict": "VERDICT_CREATED",
  "threshold": 0.9,
  "stored": true,
  "revision": "1"
}
```

Preview what would happen to an edit, without storing anything:

```sh
grpcurl -plaintext -d '{
  "record": {
    "id": "sku-1",
    "tenant": "merchant-1",
    "fields": {
      "name": "Sour Diesel Vape Cartridge 1g",
      "category": "Vape Pens > Cartridges",
      "brand": "STIIIZY",
      "description": "One gram 510-thread cartridge filled with premium distillate."
    }
  },
  "mode": "WRITE_MODE_DRY_RUN"
}' localhost:50051 recordsimilarity.v1.RecordSimilarity/Evaluate
```

```json
{
  "verdict": "VERDICT_REJECTED",
  "similarity": 0.5413923,
  "threshold": 0.9,
  "fields": [
    { "field": "brand", "change": "FIELD_CHANGE_MODIFIED", "similarity": 0.6418042, "weight": 1 },
    { "field": "category", "change": "FIELD_CHANGE_MODIFIED", "similarity": 0.55878574, "weight": 3 },
    { "field": "description", "change": "FIELD_CHANGE_MODIFIED", "similarity": 0.52455044, "weight": 2 },
    { "field": "name", "change": "FIELD_CHANGE_MODIFIED", "similarity": 0.51166517, "weight": 4, "belowFloor": true }
  ],
  "anchorSimilarity": 0.5413923,
  "reasons": [
    "overall similarity 0.541 is below the threshold 0.900",
    "field `name` similarity 0.512 is below its floor 0.800",
    "similarity to the original record 0.541 is below the drift threshold 0.850"
  ],
  "revision": "1"
}
```

Other requests:

```sh
# Store a change regardless of the verdict (a reviewer approved it). Also resets the drift anchor.
grpcurl -plaintext -d '{"record": {"id": "sku-1", "tenant": "merchant-1", "fields": {"name": "OG Kush Pre-Roll 1g"}}, "mode": "WRITE_MODE_FORCE"}' \
  localhost:50051 recordsimilarity.v1.RecordSimilarity/Evaluate

# Embed only: no id, nothing compared or stored. `return_vector` returns the vector.
grpcurl -plaintext -d '{"record": {"fields": {"name": "Blueberry Gummies"}}, "return_vector": true}' \
  localhost:50051 recordsimilarity.v1.RecordSimilarity/Evaluate

# The stored records most similar to a stored one, within one tenant.
grpcurl -plaintext -d '{"id": "sku-1", "limit": 5, "tenant": "merchant-1"}' \
  localhost:50051 recordsimilarity.v1.RecordSimilarity/FindSimilar

# The stored records most similar to text that is not stored.
grpcurl -plaintext -d '{"record": {"fields": {"name": "blueberry gummy 100 mg"}}, "limit": 3, "min_similarity": 0.5}' \
  localhost:50051 recordsimilarity.v1.RecordSimilarity/FindSimilar

grpcurl -plaintext -d '{"id": "sku-1", "tenant": "merchant-1"}' localhost:50051 recordsimilarity.v1.RecordSimilarity/Delete

# Health, as a load balancer would ask.
grpcurl -plaintext -d '{"service": "recordsimilarity.v1.RecordSimilarity"}' localhost:50051 grpc.health.v1.Health/Check
```

In JSON, enum values are written by name (`"WRITE_MODE_DRY_RUN"`), 64-bit integers come back as strings
(`"revision": "1"`), and fields at their default value (a `false`, an empty list) are omitted from responses.

### From Rust

Depend on the `similarity-proto` crate, which contains the generated client:

```rust
use similarity_proto::v1::{self as pb, record_similarity_client::RecordSimilarityClient};

let mut client = RecordSimilarityClient::connect("http://127.0.0.1:50051").await?;
let request = pb::EvaluateRequest {
  record: Some(pb::Record {
    id: Some("sku-1".to_owned()),
    tenant: Some("merchant-1".to_owned()),
    fields: [("name".to_owned(), "Blueberry Gummies 100mg".to_owned())].into(),
  }),
  mode: pb::WriteMode::DryRun.into(),
  ..Default::default()
};
let response = client.evaluate(request).await?.into_inner();
if response.verdict() == pb::Verdict::Rejected {
  eprintln!("rejected: {:?}", response.reasons);
}
```

### From other languages

Generate a client from the `.proto` with your language's gRPC tooling. For example, in Python:

```sh
python -m grpc_tools.protoc -I crates/similarity-proto/proto \
  --python_out=. --grpc_python_out=. recordsimilarity/v1/similarity.proto      # needs `pip install grpcio-tools`
```

```python
import grpc
from recordsimilarity.v1 import similarity_pb2 as pb, similarity_pb2_grpc as rpc

client = rpc.RecordSimilarityStub(grpc.insecure_channel("127.0.0.1:50051"))
record = pb.Record(id="sku-1", tenant="merchant-1", fields={"name": "Blueberry Gummies 100mg"})

response = client.Evaluate(pb.EvaluateRequest(record=record, mode=pb.WRITE_MODE_DRY_RUN))
print(pb.Verdict.Name(response.verdict), response.similarity, list(response.reasons))
```

### What `Evaluate` returns

| Request | Verdict | Stored? |
|---|---|---|
| No `id` | `VERDICT_EMBEDDED`: vector computed, nothing compared | never |
| `id` with no baseline | `VERDICT_CREATED`: this is the baseline now | yes |
| `id`, similar enough | `VERDICT_ACCEPTED` | yes (skipped if nothing changed) |
| `id`, too different | `VERDICT_REJECTED`, with `reasons` | no: the baseline is untouched |

`mode` changes the write, never the verdict:

| Mode | Behavior |
|---|---|
| `WRITE_MODE_ENFORCE` (default) | As in the table above. |
| `WRITE_MODE_DRY_RUN` | Never writes. |
| `WRITE_MODE_FORCE` | Always writes, and makes this revision the new reference point for drift. The verdict is still reported. |

Response fields: `similarity` (overall, absent when there was no baseline), `threshold` (the one applied),
`fields` (per field: `change`, `similarity`, `weight`, `below_floor`), `anchor_similarity` (similarity to the
first value each of the record's fields had), `stored`, `reasons`, `revision`, and `vector` if requested.
`threshold` on the request overrides the configured overall threshold for that one call.

A field's `change` is one of:

| Change | Meaning | Scored? |
|---|---|---|
| `FIELD_CHANGE_UNCHANGED` | Same text, ignoring whitespace. | Yes, as 1.0. |
| `FIELD_CHANGE_MODIFIED` | Both versions have a value and the text differs. | Yes, against the stored value. |
| `FIELD_CHANGE_ADDED` | The record has never had a value for this field. | No: new information is taken on trust. |
| `FIELD_CHANGE_REMOVED` | The stored version has a value and this one does not. | No. Mark a field `required` if it must not disappear. |
| `FIELD_CHANGE_RESTORED` | The stored version has no value, but the record once did, and has one again. | Yes, against the first value the field had. |

`RESTORED` exists so that a field cannot be swapped in two steps, by blanking it in one edit and filling it with
something else in the next. The refill is judged exactly as if the field had been changed directly.

### Errors

| Status | Meaning | What to do |
|---|---|---|
| `INVALID_ARGUMENT` | The record does not fit the schema (unknown field, missing required field, blank id), a parameter is out of range, or `mode` is a value this server does not know. | Fix the request. Retrying will not help. |
| `NOT_FOUND` | `FindSimilar` by an id that is not stored. | |
| `FAILED_PRECONDITION` | The stored record belongs to a tenant the request does not name (it names another, or none), or was written under a different model or field set. | Send the record's tenant, or see [Re-indexing](#re-indexing). |
| `RESOURCE_EXHAUSTED` | The embedding queue is full. | Retry with backoff. |
| `UNAVAILABLE` | Qdrant cannot be reached. | Retry with backoff. |

Nothing is written when an error is returned.

## Integrating it

**Bootstrapping an existing catalog.** Call `Evaluate` once for every existing record, in the default mode. Each
call returns `VERDICT_CREATED` and stores the baseline. On a laptop CPU the encoder gets through roughly 375 field
texts per second when requests arrive concurrently and are batched, so 50,000 four-field records take on the order
of ten minutes. Keep
fewer requests in flight than `model.queue_depth` (256 by default) and back off on `RESOURCE_EXHAUSTED`.

**At edit time.**

1. While the merchant is editing, optionally call `Evaluate` with `WRITE_MODE_DRY_RUN` to warn them early.
2. When they save, call `Evaluate` in the default mode. On `ACCEPTED` or `CREATED`, save the edit; the service has
   already stored the new baseline.
3. On `REJECTED`, do not save. Show the `reasons`, or route the edit to a human reviewer.
4. If a reviewer approves it, call `Evaluate` again with `WRITE_MODE_FORCE`, then save.

The service's stored baseline and your own database must agree, so make step 2 and your own save succeed or fail
together as far as you can: if your save fails after an `ACCEPTED`, re-submit the previous content with
`WRITE_MODE_FORCE` to put the baseline back (which also resets that record's drift anchor).

**Rolling it out.** Run it in `DRY_RUN` on real traffic first and log the responses. That costs nothing, shows what
would have been rejected, and produces exactly the data the next two sections need.

## How a verdict is reached

1. **Each field is embedded on its own**, into a unit vector. Fields whose text has not changed since the stored
   version (ignoring whitespace) are not re-embedded.
2. **Each field is compared with its stored counterpart** by cosine similarity.
3. **The overall similarity is the weighted mean** of the field similarities, using the weights in the config,
   over the fields that can be compared. A field that was removed, or added for the first time, is reported but
   does not count. A field that was blanked earlier and now has a value again does count, compared with the
   first value it had.
4. A revision is rejected if any of these holds:
   - the overall similarity is below `policy.threshold`;
   - a field's similarity is below that field's `reject_below`. This catches a swapped name hiding behind an
     untouched description;
   - the similarity to the record's *anchor* is below `policy.anchor_threshold`. The anchor holds the first
     value each field had since the record was created (or last force-written). This catches a record being
     walked somewhere else in a series of small, individually acceptable edits.

The field vectors are stored as one weighted, concatenated vector per record, built so that plain cosine similarity
between two of them is the weighted score. That is why `FindSimilar` results are on the same scale as verdicts
(exactly so when both records have the same fields filled in; see [Limitations](#limitations)). The reasoning
behind this and every other design choice is in [`docs/decisions.md`](docs/decisions.md).

## Choosing thresholds

Cosine similarities from an embedding model are not percentages. With the default model, two unrelated products
score about 0.5, and in the sample set no legitimately edited product drops below 0.94. A threshold has to be
measured, on labeled examples:

```sh
just calibrate                                            # the bundled sample pairs, the example config
just calibrate path/to/pairs.jsonl config.toml            # your pairs, your config
```

The input is one JSON object per line, labeled `edit` (should be accepted) or `recycle` (should be rejected).
Blank lines and lines starting with `#` are ignored. The field names must be ones the schema defines.

```json
{"label": "recycle", "note": "gummies become a vape", "before": {"name": "..."}, "after": {"name": "..."}}
```

The report starts with the model and the width it ran at, lists every pair with its score and verdict, flags the
ones the current config gets wrong, and ends with a summary. On the bundled sample:

```text
model     f46e48ee0ea2dd02-d256-t256 (256 dimensions per field, up to 256 tokens)
...
edits     overall similarity from 0.941 to 1.000
recycles  overall similarity from 0.495 to 0.850

field (if modified)  in edits               in recycles
brand                1.000 to 1.000 ( 1)    0.478 to 0.674 ( 9)
category             0.804 to 0.925 ( 3)    0.493 to 0.869 (16)
description          0.837 to 0.989 ( 8)    0.511 to 0.734 (17)
name                 0.875 to 1.000 (15)    0.319 to 0.725 (20)

configured policy (threshold 0.900 plus field floors): 0 edit(s) wrongly rejected, 0 recycle(s) wrongly accepted, of 46 pairs
best overall threshold for this data: 0.895 (margin 0.046; 0 edit(s) rejected, 0 recycle(s) accepted)
```

Reading it:

- **`policy.threshold`** belongs in the gap between the two overall ranges. The suggestion is the cut that
  misclassifies the fewest pairs; `margin` is how close the nearest pair comes to it.
- **`reject_below` for a field** belongs in the gap between that field's two ranges, if it has one. Here `name`
  does (0.725 to 0.875) and `category` does not (the ranges overlap), which is why the example config puts a floor
  on `name` only.
- **Weights** are worth adjusting when a field separates the groups well but is weighted low, or the reverse.
  Change them in the config and re-run; nothing needs re-indexing.
- **`policy.anchor_threshold`** is chosen from whole edit histories rather than pairs: it should sit below how far
  a legitimate record drifts over its lifetime.

If the two overall ranges overlap on real data, there are three things to try, in order of cost: weights and
floors, [a different model](#using-a-different-model), and [fine-tuning](#fine-tuning-the-model).

`record-similarity compare before.json after.json` scores a single pair the same way, to see why one verdict came
out as it did.

[`docs/data-request-prompt.md`](docs/data-request-prompt.md) is a ready-made brief for collecting labeled pairs from
production data. Its most important rule: labels must come from evidence other than how different the text looks
(moderator actions, identifier changes, complaints), or the calibration is circular.

## Fine-tuning the model

Fine-tuning teaches the encoder your catalog's own notion of "the same product": that `WYLD Blueberry 100mg (10pk)`
and `Blueberry Gummies` are one thing, and that `Blue Dream 3.5g` and `Wedding Cake 3.5g` are not.

**Do this last.** It is only worth it if calibration on real pairs shows edits and recycles overlapping in a way
that thresholds, weights, floors, and the alternative model cannot fix. It needs a few thousand training examples
to help rather than hurt, and it obliges you to re-index and re-calibrate.

Training is Rust end to end. The `record-similarity-train` binary (crate `similarity-train`) trains the same Burn
encoder the server runs, through the same tokenizer and the same weight files, and writes a model directory the
server loads like any other. It is a separate binary so that the server stays an inference-only build (D13).

**What has and has not been tested:**

- The trainer is checked against PyTorch on the real model (`just test-parity`). Replaying six optimizer steps that
  PyTorch took on fixed batches, it reproduces the loss and the gradient norm of every step to about one part in
  100,000, and ends on the same embeddings.
- The workflow below has been run end to end on synthetic data, on the CPU and on Metal.
- It has not been used to produce a better model, because that needs real data. The CUDA build compiles but has
  never been run.

### 1. Assemble training data

Two kinds of signal, either or both. The formats are the ones `docs/data-request-prompt.md` asks for.

- **Labeled pairs** (`--pairs`): the same before/after pairs used for calibration. Extra keys are ignored, so
  `pairs.full.jsonl` works directly. Pairs labeled anything other than `edit` or `recycle` are skipped. If pairs
  carry a `split` key, `calibrate` is trained on (`--split` changes that) and `holdout` is only ever measured
  (`--holdout-split`). Pairs with no `split` are all trained on.
- **Catalog groups** (`--catalog-groups`): listings from different merchants linked to the same catalog product.
  These are naturally occurring paraphrases and usually the larger, cleaner source.

  ```json
  {"group": "g1", "catalog": {"name": "Blueberry Gummies 100mg"}, "listings": [{"name": "WYLD Blueberry 100mg (10pk)"}]}
  ```

By default only `name` and `description` are learned from (`--fields`). Categorical fields are left out because the
same change (say, sativa to hybrid) appears in both edits and recycles, which is noise at the level of one field.

### 2. Train

```sh
just finetune --base models/mdbr-leaf-mt --out models/mdbr-leaf-mt-ft \
  --pairs pairs.full.jsonl --catalog-groups catalog-groups.jsonl
```

`just finetune` is `cargo run --release -p similarity-train --`. To train on a GPU, build that backend in and
select it:

```sh
just release-train metal        # or: just release-train cuda
./target/release/record-similarity-train --device metal --base models/mdbr-leaf-mt --out models/mdbr-leaf-mt-ft \
  --pairs pairs.full.jsonl --catalog-groups catalog-groups.jsonl
```

| Option | Default | Notes |
|---|---|---|
| `--base` | (required) | The model directory to start from. |
| `--out` | (required) | Where to write the fine-tuned directory. Put it under `models/`, which git ignores. It must not exist yet, or be empty: a model is never overwritten. |
| `--pairs`, `--catalog-groups` | | At least one. |
| `--fields` | `name,description` | Fields to learn from. |
| `--split` | `calibrate` | Which `split` of the pairs to train on. |
| `--holdout-split` | `holdout` | Which `split` to report on without training on it. |
| `--dims` | `1024,512,256` | Embedding widths to keep usable. Must include the server's `output_dim`. A width the model cannot produce is replaced by its native width, so a 384-wide model is trained at `384,256`. |
| `--epochs` | `2` | Passes over the data. |
| `--batch-size` | `32` | Larger batches give the catalog-group loss more negatives per step, and cost memory. |
| `--learning-rate` | `2e-5` | The peak, after a warm-up over the first 10% of steps. Lower it if the result gets worse on the holdout. |
| `--max-tokens` | `256` | Match `model.max_tokens` in the server config. |
| `--device` | `cpu` | `cpu`, `metal`, or `cuda`. The GPU ones need a binary built with that feature. |
| `--seed` | `0` | Fixes the batch order and dropout. |
| `--no-dropout` | off | Trains without the encoder's dropout. |
| `--weight-decay` | `0` | Applied to every parameter. |
| `--max-grad-norm` | `1` | Global gradient-norm clipping; `0` disables it. |
| `--log-every` | `10` | Steps between progress lines. |

What it does:

- **Labeled pairs** are trained with a contrastive loss: an edit's two texts are pulled together, a recycle's are
  pushed apart until they clear a margin.
- **Catalog groups** are trained with a ranking loss: each catalog entry must pick out its own listing from the
  others in the batch. No batch holds the same text twice, so two listings of one product are never used as
  negatives for each other. A listing left with nothing to be ranked against sits that pass out (the log says
  how many), which happens when one catalog entry has far more listings than the rest.
- **Both losses are applied at every width in `--dims`**, so the model stays usable when truncated.
- The optimizer is AdamW with a linear warm-up and decay, and the encoder's own dropout (10%) is active. Dropout
  exists only inside the training pass; the server's forward pass has none to forget to turn off.

When it finishes it saves the model, loads it back the way the server does, and prints where edits and recycles sit
before and after. This run trained on 47 pair and 15 catalog examples (synthetic) in 4 seconds on a laptop CPU:

```text
held-out pairs (never trained on): 5 edit and 8 recycle field texts
  width          edits min / mean / max   recycles min / mean / max   gap of means
    256  before  0.936 / 0.963 / 0.978    0.589 / 0.644 / 0.734      +0.319
    256  after   0.934 / 0.962 / 0.977    0.550 / 0.611 / 0.712      +0.351
```

That table is a progress check on single field texts, with no field weights, floors, or thresholds. It tells you
whether training moved things the right way; step 3 tells you whether the model is better.

**How long it takes.** Each step encodes `2 x batch-size` texts and its cost follows their length. Measured on an
Apple-silicon laptop with the default batch size of 32:

| Texts | CPU | Metal |
|---|---|---|
| Names (about 10 tokens) | 0.3 s per step, 0.9 GB of memory | 0.2 s per step |
| Descriptions at the 256-token limit | 9 s per step, 9 GB of memory | 1.3 s per step |

A run has `examples / batch-size x epochs` steps, so 10,000 description examples for two passes is about 625 steps:
around an hour and a half on that CPU, or a quarter of an hour on its GPU. Use a GPU, or a smaller `--batch-size`,
when the CPU's memory use is a problem.

### 3. Measure whether it helped

Evaluate on pairs the model did not train on, with the full policy. Project the holdout pairs into the calibration
format:

```sh
jq -c 'select(.split == "holdout" and (.label == "edit" or .label == "recycle"))
  | def keep: with_entries(select(.key | IN("name", "category", "brand", "description")));
    {label, note: "\(.pair_id) \(.note)", before: (.before | keep), after: (.after | keep)}' \
  pairs.full.jsonl > holdout.jsonl
```

Then calibrate with the base model and with the fine-tuned one, using two configs that differ only in `model.dir`:

```sh
just calibrate holdout.jsonl config.toml             # base model
just calibrate holdout.jsonl config-finetuned.toml   # dir = "models/mdbr-leaf-mt-ft"
```

Keep the fine-tuned model only if the gap between the two overall ranges is clearly wider, or the number of
misclassified pairs clearly lower, on the holdout. A model that improves only on the pairs it trained on has
memorized them.

### 4. Deploy it

A fine-tuned model is a different model. Its fingerprint differs, so:

1. Set `model.dir` to the new directory and `qdrant.collection` to a **new** collection name.
2. Set the thresholds from the holdout calibration of the new model. The old thresholds do not carry over.
3. Re-index: see [Re-indexing](#re-indexing).
4. To ship it in the container image, `COPY` the directory into the image in place of the `model fetch` step in
   the `Dockerfile`, and remove `/models` from `.dockerignore` so the build can see it.

The saved directory is also a valid sentence-transformers model. If you want a second opinion from PyTorch,
`just verify-model models/mdbr-leaf-mt-ft` (needs `uv`) loads it there and requires the Burn encoder to reproduce
PyTorch's embeddings. Nothing in the workflow depends on it.

## Using a different model

Any sentence-transformers model with a BERT encoder, mean or CLS pooling, and optionally a linear projection can be
used without code changes. `all-MiniLM-L6-v2`, `gte-small`, and `bge-small-en-v1.5` have been loaded and run this
way, as has the default; other models of the same shape should work too. Anything else (RoBERTa, ModernBERT,
last-token or max pooling) is refused at load with a message saying what is unsupported, and `just verify-model
<dir>` proves that a model that does load computes what PyTorch computes.

```sh
record-similarity model fetch --repo sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --dir models/all-MiniLM-L6-v2
```

Then, in the config: set `model.dir`, **remove `output_dim`** unless the model was trained to be truncated, choose a
new `qdrant.collection`, and re-calibrate. Use a commit hash for `--revision`; checksums are pinned only for the
default model, and for others the command prints the digests of what it downloaded.

Measured on the bundled sample pairs (`just calibrate`), each model at the width shown. "Gap" is the distance
between the lowest-scoring edit and the highest-scoring recycle, which is the room a threshold has:

| Model | Width | Edits | Recycles | Gap | One record |
|---|---|---|---|---|---|
| `mdbr-leaf-mt` (default) | 256 of 1024 | 0.941 to 1.000 | 0.495 to 0.850 | 0.091 | 20 ms |
| `all-MiniLM-L6-v2` | 384 | 0.896 to 1.000 | 0.173 to 0.815 | 0.081 | 19 ms |
| `gte-small` | 384 | 0.968 to 1.000 | 0.791 to 0.949 | 0.019 | 41 ms |
| `bge-small-en-v1.5` | 384 | 0.928 to 1.000 | 0.583 to 0.911 | 0.017 | 42 ms |

The two six-layer models are close, and 46 synthetic pairs cannot tell them apart; the two twelve-layer models are
slower and leave almost no gap. When you have real pairs, calibrate with both `mdbr-leaf-mt` and `all-MiniLM-L6-v2`
before settling. More in `docs/decisions.md`, D3 and D4.

## Configuration reference

One TOML file. Unknown keys are errors, so a typo cannot silently become a default. `config.example.toml` is the
commented starting point.

| Key | Default | Meaning |
|---|---|---|
| `server.listen` | `127.0.0.1:50051` | Address to serve gRPC on. Use `0.0.0.0:50051` in a container. |
| `model.dir` | `models/mdbr-leaf-mt` | The model directory. |
| `model.device` | `cpu` | `cpu`, or `metal` in a binary built with `--features metal`. |
| `model.max_tokens` | `256` | Text beyond this many tokens is ignored. Cost grows with its square. |
| `model.output_dim` | none (native width) | Keep only the first N components of each field vector. Only for models trained for it. The example config sets 256 for the default model. |
| `model.batch_size` | `32` | The most texts in one forward pass. |
| `model.queue_depth` | `256` | Requests that may wait for the encoder before new ones get `RESOURCE_EXHAUSTED`. |
| `qdrant.url` | `http://localhost:6334` | Qdrant's gRPC endpoint (6334, not the REST port 6333). |
| `qdrant.collection` | `records` | The collection. Created if missing. |
| `qdrant.timeout_ms` | `5000` | Deadline for one Qdrant request. |
| `policy.threshold` | (required) | Minimum overall similarity to the stored record. |
| `policy.anchor_threshold` | none | Minimum overall similarity to the record's anchor (the first value of each field). Omit to report drift without enforcing it. |
| `fields.<name>.weight` | (required) | Relative importance of the field. Only ratios matter. |
| `fields.<name>.required` | `false` | Refuse records with no value for this field. |
| `fields.<name>.reject_below` | none | Reject a revision whose similarity for this field alone is below this. |

At least one field is required. Field names may contain letters, digits, and underscores.

What a change costs:

| Change | Consequence |
|---|---|
| A threshold, a floor, `required` | None. Takes effect on restart. |
| A field weight | Verdicts use the new weight at once. Search rankings use the weights records were written with until they are rewritten. |
| `model.dir`, `output_dim`, `max_tokens`, or the set of fields | Stored vectors are no longer comparable. Use a new collection and re-index. The server refuses to mix them. |

## Operations

### Re-indexing

Needed after changing the model, `output_dim`, `max_tokens`, or the set of fields. Stored vectors carry an identifier
of the model and field layout that produced them, and the server will not compare across layouts: reading an old
record returns `FAILED_PRECONDITION`, and a collection whose vectors have the wrong width stops the server at
start-up.

1. Point `qdrant.collection` at a new, empty collection and restart.
2. Replay every record from your own database through `Evaluate`, as in
   [bootstrapping](#integrating-it). Each becomes a new baseline.
3. Drop the old collection when you are satisfied.

Re-indexing resets drift history: each record's anchor becomes its content at the time of the re-index.

### Health and shutdown

The server implements the standard gRPC health service; the service name to check is
`recordsimilarity.v1.RecordSimilarity`. For Kubernetes:

```yaml
readinessProbe:
  grpc:
    port: 50051
    service: recordsimilarity.v1.RecordSimilarity
```

It shuts down gracefully on `SIGTERM`.

### Capacity

One thread runs the model, and the backend spreads each forward pass across the machine's cores, so a server scales
with cores rather than with replicas' thread counts. Under load, requests waiting for the encoder are merged into
one batch automatically. When more than `model.queue_depth` requests are waiting, new ones are refused immediately
with `RESOURCE_EXHAUSTED` rather than queued past their deadline. Add replicas for more throughput, with the caveat
under [Limitations](#limitations).

### Security

The server speaks plaintext gRPC and has no authentication. Run it on a private network or behind a mesh or proxy
that provides both. The Qdrant API key is taken from the environment, never from the config file.

## Performance

Measured with `just throughput` on an Apple-silicon laptop CPU, release build, default model:

| Input | Time |
|---|---|
| One short field (the usual edit) | 4 ms |
| One description | 12 ms |
| One whole record, four fields | 20 ms |
| Eight whole records in one batch | 86 ms |

Model load is about 0.3 s. A stored composite at the example settings is 1024 floats, so 50,000 records are about
400 MB in Qdrant across the two vectors each record keeps, of which half is memory-mapped from disk. Run
`just throughput` on your own hardware before sizing anything.

## Development

```sh
just                 # list every recipe
just test            # fast tests: no model, no network, no Qdrant
just clippy          # zero warnings, pedantic lints included
just fmt             # format
just ci              # fmt-check, clippy, test, cargo-deny: what CI runs
just test-qdrant     # the storage contract against a live Qdrant
just test-parity     # Burn against PyTorch on the real model: the embeddings, then the training steps
just test-real       # test-parity, then the full stack through gRPC against Qdrant
just throughput      # forward-pass latency on this machine
```

Run `just test-real` after touching the encoder, the loader, the tokenizer settings, the trainer, or
`config.example.toml`.

### Project layout

```text
crates/
  similarity-proto/     the .proto file and generated gRPC code
  similarity-core/      the encoder (Burn), tokenization, composite vectors, scoring, policy. No network, no async.
  similarity-store/     the VectorStore trait, an in-memory implementation, and the Qdrant one
  similarity-server/    config, embedding worker thread, engine, gRPC service, and the `record-similarity` binary
  similarity-train/     fine-tuning: data, losses, the training loop, and the `record-similarity-train` binary
data/
  calibration.sample.jsonl   synthetic labeled pairs
  examples/                  sample records for the client
docs/
  decisions.md               why things are the way they are (D1 to D14)
  data-request-prompt.md     a brief for collecting labeled production data
models/                 fetched and fine-tuned model directories (ignored by git)
tools/                  referees, not part of any pipeline: they write the fixtures the parity tests compare with
  golden.py             PyTorch reference embeddings
  train_golden.py       PyTorch reference training steps
```

Conventions (two-space indentation, no `unwrap` in library code, tests first) are in [`CLAUDE.md`](CLAUDE.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Build fails with `Could not find protoc` | Install `protoc` ([Dependencies](#dependencies)). |
| `cannot read config file config.toml` | `serve` looks for `config.toml` in the current directory. Create it (`cp config.example.toml config.toml`) or pass `--config`. |
| `cannot load the model from ...; run record-similarity model fetch first` | The model directory is missing or incomplete. Run `just fetch-model`. The cause is printed underneath. |
| `unsupported model at ...` | The directory is not a BERT-architecture sentence-transformers model, or uses a pooling mode other than mean or CLS. The message names what is unsupported. |
| `weight load failed for ...` | The checkpoint does not match its `config.json`. Re-download; do not edit model files by hand. |
| `cannot use Qdrant collection ... vector store unavailable` | Qdrant is not running, or `qdrant.url` points at the REST port. The gRPC port is 6334. `just qdrant-up`. |
| `collection ... is not usable: vector current has N components, but this server's model and schema produce M` | The collection was created under a different model, `output_dim`, or field set. Use a new `qdrant.collection` and [re-index](#re-indexing). |
| `FAILED_PRECONDITION: record ... was stored under layout ...` | Same cause, for a single record in a collection of the right width. Re-index. |
| `device = "metal" needs a binary built with --features metal` | Build with `just release-metal`, or set `device = "cpu"`. |
| `--device metal needs a binary built with --features metal` (fine-tuning) | Build with `just release-train metal` and run `./target/release/record-similarity-train`; `just finetune` always runs the CPU build. |
| `... already exists and is not empty` (fine-tuning) | `--out` names a directory that holds something. Choose a new one; a model directory is never overwritten. |
| `INVALID_ARGUMENT: invalid record: unknown field ...` | The request used a field name the schema does not define. `Describe` lists the valid ones. |
| Many `RESOURCE_EXHAUSTED` | More requests are arriving than the encoder can serve. Back off in the client, add CPU, or add replicas. |
| Everything is rejected, or nothing is | The thresholds do not suit the model or the data. Run `calibrate`. Thresholds never carry over between models. |
| The first Docker command fails with `Cannot connect to the Docker daemon` | Docker is not running. Start Docker Desktop, or `orbctl start` for OrbStack. |

## Limitations

- **Concurrent edits to one record across replicas.** One server serializes evaluations of the same id. Two
  replicas do not coordinate, and Qdrant has no compare-and-swap, so two simultaneous revisions of one record
  handled by different replicas could both be accepted against the same baseline. If callers do not already
  serialize edits per record, route by id, or see D6 and D10 in the decisions.
- **Search scores drop when records have different fields filled in.** `Evaluate` compares only the fields both
  versions have. `FindSimilar` compares whole stored vectors, in which a missing field is a block of zeros, so two
  records that are identical except that one has no description score 1.0 in `Evaluate` but about 0.89 in
  `FindSimilar`. Verdicts are unaffected; rankings favor records with the same fields filled in as the query,
  which is most visible when searching by a name alone. See D5.
- **No bulk API.** Bootstrapping and re-indexing are one `Evaluate` call per record.
- **English only.** The default model is English. Other languages need another model and a new calibration.
- **Text only.** Images are not considered. A recycler who changes only the photo is invisible to this service.
- **Truncation.** Text beyond `max_tokens` (256 by default) is ignored.
- **No transport security or authentication** in the server itself.

## Acknowledgements

- [Burn](https://github.com/tracel-ai/burn) for the tensor and module framework.
- [`MongoDB/mdbr-leaf-mt`](https://huggingface.co/MongoDB/mdbr-leaf-mt) (Apache-2.0), the default embedding model.
- [sentence-transformers](https://www.sbert.net), whose model directory layout this project reads and writes, and
  whose output is the reference both parity tests are held to.
